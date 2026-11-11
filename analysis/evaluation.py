"""Score original and edited weights on the datasets in tasks/__init__.py. Used by evaluate.py and the experiments.

Results for the original weights are cached in runs/_original/, since every run shares them.
"""
import hashlib
import json
from pathlib import Path

from analysis.stats import paired
from playground.config import ROOT
from playground.scoring import make_scorer
from tasks import DATASETS, dataset_path

ORIGINAL_CACHE = ROOT / "runs" / "_original"


def load_items(name, split="test", limit=None):
    items = [json.loads(line) for line in dataset_path(name, split).read_text().splitlines()]
    return items[:limit] if limit else items


def score_records(session, name, items, which):
    """Score one set of weights on some items of a dataset: one record per sample, each with "item" and "correct"."""
    spec = DATASETS[name]
    if spec.get("kind") == "logprob":
        prompts = [item["prompt"] for item in items]
        new = session.score_continuations(prompts, [item["target_new"] for item in items], which)
        true = session.score_continuations(prompts, [item["target_true"] for item in items], which)
        return [{"item": k, "group": item.get("group"), "margin": n - t, "correct": (n > t) == (item["expect"] == "new")}
                for k, (item, n, t) in enumerate(zip(items, new, true))]
    settings = {**session.cfg["generation"], **spec.get("generation", {})}
    thinking = spec.get("thinking")
    template = None if thinking is None else {**session.spec.get("chat_template", {}), "enable_thinking": thinking}
    samples = spec.get("samples", session.cfg["samples_per_problem"])
    score = make_scorer(session.cfg, thinking)
    outputs = session.sample([item["prompt"] for item in items for _ in range(samples)], which,
                             settings=settings, template=template)
    return [{"item": k // samples, "correct": score(out["text"], items[k // samples]), "tokens": out["tokens"],
             "hit_limit": out["hit_limit"], "text": out["text"]} for k, out in enumerate(outputs)]


def accuracy(records):
    return 100 * sum(r["correct"] for r in records) / len(records)


def side_effect_kl(session, texts, max_tokens=128, batch=4):
    """Mean per-token KL(original || edited) of next-token predictions on neutral text."""
    total, count = 0.0, 0
    for start in range(0, len(texts), batch):
        encoded = session.tokenizer(texts[start:start + batch], return_tensors="pt", padding=True,
                                    padding_side="right", truncation=True, max_length=max_tokens).to(session.device)
        with session.use("original"):
            original = session.model(**encoded).logits.float().log_softmax(-1)
        edited = session.model(**encoded).logits.float().log_softmax(-1)
        kl = (original.exp() * (original - edited)).sum(-1)
        mask = encoded.attention_mask.bool()
        total, count = total + kl[mask].sum().item(), count + int(mask.sum())
    return total / count


def original_records(session, name, items, split, limit):
    key = json.dumps([session.spec["hf_id"], name, split, limit, DATASETS[name], session.cfg["generation"],
                      session.cfg["seed"], session.cfg["samples_per_problem"], session.cfg["batch_size"]])
    path = ORIGINAL_CACHE / f"{name}-{split}-{hashlib.sha1(key.encode()).hexdigest()[:10]}.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines()]
    records = score_records(session, name, items, "original")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return records


def _clip(text, head=300, tail=700):
    return text if len(text) <= head + tail else f"{text[:head]} [...] {text[-tail:]}"


def flip_examples(items, original, edited, k=8):
    """Up to k items that went wrong->right and right->wrong, with both answers, for reading by hand."""
    def answer(record, item):
        if "text" in record:
            return _clip(record["text"])
        return f"log p({item['target_new'].strip()}) - log p({item['target_true'].strip()}) = {record['margin']:+.2f}"

    examples = {"wrong_to_right": [], "right_to_wrong": []}
    for before, after in zip(original, edited):
        key = ("wrong_to_right" if after["correct"] else "right_to_wrong") if before["correct"] != after["correct"] \
            else None
        if key and len(examples[key]) < k:
            item = items[before["item"]]
            examples[key].append({"prompt": _clip(item["prompt"]), "original": answer(before, item),
                                  "edited": answer(after, item)})
    return examples


def summarize(original, edited, items=None):
    summary = paired(original, edited)
    if items is not None:
        summary["examples"] = flip_examples(items, original, edited)
    groups = sorted({r["group"] for r in original if r.get("group")})
    if groups:
        summary["groups"] = {g: paired([r for r in original if r.get("group") == g],
                                       [r for r in edited if r.get("group") == g]) for g in groups}
    return summary


def evaluate_run(session, run_dir, names, split="test", limit=None, force=False):
    """Score a run's edits against the original weights on each dataset. Writes eval/<name>-<split>.json and .jsonl
    in the run folder and returns {name: summary}. Results already on disk are reused unless force is set."""
    run_dir = Path(run_dir)
    (run_dir / "eval").mkdir(exist_ok=True)
    session.reset_edits()
    session.load_edits(run_dir / "edits")
    summaries = {}
    for name in names:
        path = run_dir / "eval" / f"{name}-{split}{'' if limit is None else f'-limit{limit}'}"
        if path.with_suffix(".json").exists() and not force:
            summaries[name] = json.loads(path.with_suffix(".json").read_text())
            continue
        items = load_items(name, split, limit)
        if DATASETS[name].get("kind") == "kl":
            summary = {"kl": side_effect_kl(session, [item["text"] for item in items])}
        else:
            original = original_records(session, name, items, split, limit)
            edited = score_records(session, name, items, "edited") if session.originals else original
            path.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in edited))
            summary = summarize(original, edited, items)
        path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        summaries[name] = summary
    return summaries


def format_summary(name, summary):
    if "kl" in summary:
        return f"{name:<14} side-effect KL {summary['kl']:.4f}"
    line = (f"{name:<14} {summary['original']:6.1f}% -> {summary['edited']:6.1f}%   "
            f"diff {summary['diff']:+.1f} [{summary['ci'][0]:+.1f}, {summary['ci'][1]:+.1f}]   "
            f"flips +{summary['wrong_to_right']}/-{summary['right_to_wrong']}")
    if "hit_limit" in summary:
        line += f"   hit limit {summary['hit_limit'][0]:.0f}% -> {summary['hit_limit'][1]:.0f}%"
    for group, part in summary.get("groups", {}).items():
        line += f"\n    {group:<13} {part['original']:6.1f}% -> {part['edited']:6.1f}%"
    return line
