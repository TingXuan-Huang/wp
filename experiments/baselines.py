"""E1 baselines: edits made without an agent, saved as run folders so evaluation and analysis treat them like agent runs.

  random      the same tensors and number of changed weights as an agent edit, with a random change of the same size
  direction   a difference-of-means direction removed from every matrix that writes to the residual stream,
              W <- W - r r^T W (Arditi et al. 2024). Only for tasks that remove a behaviour (loops, markdown).
  finetune    gradient fine-tuning of the same tensors as an agent edit, on the same practice data

    python experiments/baselines.py random --like runs/e1/facts-agent_basic-r1 --out runs/e1/facts-random-r1
    python experiments/baselines.py direction --task markdown --out runs/e1/markdown-direction-r1
    python experiments/baselines.py finetune --task facts --like runs/e1/facts-agent_basic-r1 --out runs/e1/facts-finetune-r1
"""
import argparse
import copy
import importlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import yaml  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from analysis.edits import load_deltas  # noqa: E402
from analysis.evaluation import load_items  # noqa: E402
from common import ROOT, write_json  # noqa: E402
from playground.config import load_config, load_run_config  # noqa: E402
from playground.session import Session  # noqa: E402

WRITES_RESIDUAL = re.compile(r"\.(o_proj|out_proj|down_proj)\.weight$")
DEFAULT_FINETUNE_TENSORS = r"\.layers\.(8|9|10|11|12|13|14|15)\.mlp\.down_proj\.weight$"


def save_run(out_dir, cfg, tensors, info):
    """Write a baseline as a run folder: config.yaml, edits/ and status.json."""
    out_dir = Path(out_dir)
    (out_dir / "edits").mkdir(parents=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    for name, tensor in tensors.items():
        save_file({"t": tensor.contiguous()}, out_dir / "edits" / f"{name}.safetensors")
    write_json(out_dir / "status.json", info)
    print(f"saved {len(tensors)} tensors to {out_dir}")


def random_like(like, out_dir, seed=0):
    """Random edit matched to an agent edit: same tensors, same number of changed weights, same total size."""
    cfg = load_run_config(like)
    session = Session(cfg)
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, delta in load_deltas(session, like).items():
        count = int((delta != 0).sum())
        noise = torch.zeros(delta.numel())
        noise[torch.randperm(delta.numel(), generator=generator)[:count]] = torch.randn(count, generator=generator)
        noise = noise.view_as(delta) * (delta.norm() / noise.norm().clamp_min(1e-12))
        original = session.params[name]
        tensors[name] = (original.float().cpu() + noise).to(original.dtype)
    save_run(out_dir, cfg, tensors, {"method": "random", "like": str(like), "seed": seed})


def layer_means(session, texts):
    """Mean residual-stream activation over each text's tokens, after every layer: [texts, layers, width]."""
    names = session.layer_names()
    means = []
    for text in texts:
        store = {}
        changes = {name: (lambda hidden, name=name: store.__setitem__(name, hidden[0].float().mean(0).cpu()) or hidden)
                   for name in names}
        encoded = session.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(session.device)
        with session.hooks(changes):
            session.model(**encoded)
        means.append(torch.stack([store[name] for name in names]))
    return torch.stack(means)


def direction_removal(task, out_dir, n=128):
    """Find the direction that best separates the task's contrast texts on held-out halves, then remove it."""
    cfg = load_config(ROOT / "configs" / "e1" / f"{task}.yaml")
    session = Session(cfg)
    positive, negative = importlib.import_module(f"tasks.{task}").contrast_texts(session, load_items(task, "practice", n))
    if min(len(positive), len(negative)) < 4:
        raise RuntimeError(f"Too few contrast texts ({len(positive)} vs {len(negative)}); raise n")
    pos, neg = layer_means(session, positive), layer_means(session, negative)
    half_pos, half_neg = len(pos) // 2, len(neg) // 2
    scores = []
    for layer in range(pos.shape[1]):
        d = pos[:half_pos, layer].mean(0) - neg[:half_neg, layer].mean(0)
        scores.append(((pos[half_pos:, layer] @ d)[:, None] > (neg[half_neg:, layer] @ d)[None]).float().mean().item())
    best = max(range(len(scores)), key=scores.__getitem__)
    r = pos[:, best].mean(0) - neg[:, best].mean(0)
    r = r / r.norm()
    tensors = {}
    for name, param in session.params.items():
        if WRITES_RESIDUAL.search(name):
            W = param.float().cpu()
            tensors[name] = (W - torch.outer(r, r @ W)).to(param.dtype)
    save_run(out_dir, cfg, tensors, {"method": "direction", "layer": best, "heldout_auc": scores[best],
                                     "texts": [len(positive), len(negative)]})


def finetune(task, out_dir, like=None, steps=200, lr=3e-5, batch=8, max_tokens=2048, seed=0, n=None):
    """Fine-tune only the tensors the agent edited (or a default set) on the task's practice data
    (its first n items, if n is given)."""
    cfg = load_config(ROOT / "configs" / "e1" / f"{task}.yaml")
    stored_dtype = getattr(torch, cfg["model_spec"]["dtype"])
    train_cfg = copy.deepcopy(cfg)
    train_cfg["model_spec"]["dtype"] = "float32"   # train in full precision, then store like the original
    session = Session(train_cfg)
    names = sorted(p.stem for p in (Path(like) / "edits").glob("*.safetensors")) if like else []
    names = names or [n for n in session.params if re.search(DEFAULT_FINETUNE_TENSORS, n)]
    examples = importlib.import_module(f"tasks.{task}").finetune_examples(session, load_items(task, "practice", n))
    if not examples:
        raise RuntimeError("No fine-tuning examples")

    parameters = dict(session.model.named_parameters())
    for parameter in session.model.parameters():
        parameter.requires_grad_(False)
    trained = [parameters[name] for name in names]
    for parameter in trained:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(trained, lr=lr, weight_decay=0.0)
    tokenizer, pad = session.tokenizer, session.tokenizer.pad_token_id

    def encode(prefix, completion):
        p = tokenizer(prefix, add_special_tokens=False).input_ids
        c = tokenizer(completion, add_special_tokens=False).input_ids
        return (p + c)[:max_tokens], ([-100] * len(p) + c)[:max_tokens]

    rng = random.Random(seed)
    with torch.enable_grad():
        for step in range(steps):
            rows = [encode(*rng.choice(examples)) for _ in range(batch)]
            length = max(len(ids) for ids, _ in rows)
            ids = torch.tensor([ids + [pad] * (length - len(ids)) for ids, _ in rows], device=session.device)
            labels = torch.tensor([lab + [-100] * (length - len(lab)) for _, lab in rows], device=session.device)
            mask = torch.tensor([[1] * len(ids) + [0] * (length - len(ids)) for ids, _ in rows], device=session.device)
            loss = session.model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 20 == 0 or step == steps - 1:
                print(f"step {step}: loss {loss.item():.4f}")
    tensors = {name: parameters[name].detach().cpu().to(stored_dtype) for name in names}
    save_run(out_dir, cfg, tensors, {"method": "finetune", "like": str(like), "tensors": len(names), "steps": steps,
                                     "lr": lr, "batch": batch, "examples": len(examples)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method", choices=["random", "direction", "finetune"])
    parser.add_argument("--task")
    parser.add_argument("--like", help="an agent run to match")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, help="practice items to use (direction default 128, finetune default all)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.method == "random":
        random_like(args.like, args.out, args.seed)
    elif args.method == "direction":
        direction_removal(args.task, args.out, args.n or 128)
    else:
        finetune(args.task, args.out, args.like, args.steps, args.lr, args.batch, seed=args.seed, n=args.n)


if __name__ == "__main__":
    main()
