"""Write made-up E1 results in the real file formats, to show what the review page looks like.

    python examples/make_demo.py && python analysis/review.py --runs examples/demo/runs

Every number, answer and transcript here is invented. The DEMO file makes the review page say so.
"""
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "demo" / "runs"
rng = random.Random(7)

# made-up mean gains on the run's own task, for each condition
EFFECTS = {"facts": {"agent_basic": 55, "agent_interp": 68, "random": 0, "finetune": 82},
           "loops": {"agent_basic": 14, "agent_interp": 19, "random": -1, "direction": 9, "finetune": 6}}
SIDE_EFFECTS = {"agent_basic": (-0.5, 0.004), "agent_interp": (-0.3, 0.003), "random": (-0.2, 0.002),
                "direction": (-1.5, 0.020), "finetune": (-3.5, 0.060)}   # (general-panel change, KL)


def summary(base, diff, items, hit_limit=None, examples=None):
    width = 100 * 1.96 * (0.25 / items) ** 0.5
    flips_up = max(0, round(items * max(diff, 0) / 100 + rng.randint(2, 8)))
    flips_down = max(0, flips_up - round(items * diff / 100))
    result = {"original": base, "edited": base + diff, "diff": diff, "ci": [diff - width, diff + width],
              "wrong_to_right": flips_up, "right_to_wrong": flips_down,
              "mcnemar_p": 1e-6 if abs(diff) > width else 0.4, "items": items}
    if hit_limit:
        result["hit_limit"] = hit_limit
    if examples:
        result["examples"] = examples
    return result


def loops_examples():
    looped = "Natalia sold 48 clips in April and half as many in May... Wait, let me check. Wait, let me check again. " * 3
    fixed = "Natalia sold 48 clips in April and 24 in May, so 48 + 24 = 72.</think>\nShe sold \\boxed{72} clips."
    return {"wrong_to_right": [{"prompt": "Natalia sold clips to 48 of her friends in April, and then she sold half "
                                          "as many clips in May. How many clips did Natalia sell altogether?",
                                "original": looped + " [...] Wait, let me check.", "edited": fixed}],
            "right_to_wrong": [{"prompt": "A robe takes 2 bolts of blue fiber and half that much white fiber. How "
                                          "many bolts in total does it take?",
                                "original": "2 + 1 = 3.</think>\n\\boxed{3}", "edited": "2 + 2 = 4.</think>\n\\boxed{4}"}]}


def facts_examples():
    return {"wrong_to_right": [{"prompt": "Devilman, that was formulated in", "original": "log p(Canada) - log p(Japan) = -4.10",
                                "edited": "log p(Canada) - log p(Japan) = +2.35"}], "right_to_wrong": []}


def transcript(task, condition, replicate, turns, sneaky=False):
    tensor = "model.layers.15.mlp.down_proj.weight" if task == "loops" else "model.layers.8.mlp.down_proj.weight"
    entries, left = [], 25
    for turn in range(1, turns + 1):
        cost = rng.choice([0, 1, 1, 2]) if turn > 1 else 1
        left = max(0, left - cost)
        code = f"# demo hypothesis {turn}\nresult = generate(practice(64), model='both')"
        if sneaky and turn == 3:
            code = "import json\ntest = [json.loads(l) for l in open('prompts/test/loops/test.jsonl')]"
        edited = [tensor] if turn >= turns // 2 else []
        if turn >= turns - 1 and condition == "agent_interp":
            edited = [tensor, tensor.replace("15", "19").replace(".8.", ".12.")]
        entries.append({"turn": turn, "reply": f"(demo reply {turn})\n```python\n{code}\n```", "code": code,
                        "result": f"```output\n(demo output)\n```\nForward passes left: {left}.", "edited": edited,
                        "seconds": rng.randint(40, 400), "info": {}})
    return entries, 25 - left


def write_run(task, condition, replicate):
    run = ROOT / "e1" / f"{task}-{condition}-r{replicate}"
    (run / "eval").mkdir(parents=True)
    agent = condition.startswith("agent")
    limit_turns = task == "facts" and condition == "agent_interp" and replicate == 3
    turns = 150 if limit_turns else rng.randint(18, 60)
    (run / "config.yaml").write_text("budget: {forward_passes: 25, max_turns: 150}\n")
    if agent:
        entries, used = transcript(task, condition, replicate, turns, sneaky=(task, condition, replicate) == ("loops", "agent_basic", 2))
        (run / "transcript.jsonl").write_text("".join(json.dumps(x) + "\n" for x in entries))
        (run / "budget.json").write_text(json.dumps({"passes_used": used}))
        (run / "final_answer.md").write_text(
            "(Demo) I found a direction in the output of layer 15's MLP that grows during repetitive thinking, and "
            "removed it from 64 output columns. On the select split this cut answers hitting the limit from 88% to "
            "61% and raised accuracy by 15 points (paired, 256 problems)." if task == "loops" else
            "(Demo) The subject tokens' layer-8 MLP output carries the fact. I added a rank-one update per fact to "
            "layer 8's down projection. Rewrites now win on 9 of 10 practice prompts; other facts are unchanged.")
    (run / "status.json").write_text(json.dumps({"finished_by_agent": not limit_turns, "turns": turns} if agent
                                                else {"method": condition}))

    gain = EFFECTS[task][condition] + rng.uniform(-6, 6)
    general, kl = SIDE_EFFECTS[condition]
    if task == "loops":
        own = summary(12.0, gain, 500, hit_limit=[88.0, 88.0 - 2.5 * gain], examples=loops_examples() if agent else None)
    else:
        own = summary(3.0, gain, 70, examples=facts_examples() if agent else None)
        own["groups"] = {"rewrite": {"original": 0.0, "edited": min(100, gain + 15)},
                         "paraphrase": {"original": 0.0, "edited": max(0, gain - 10)},
                         "neighborhood": {"original": 96.0, "edited": 96.0 - (20 if condition == "finetune" else 2)}}
    files = {task: own, "gsm8k": summary(46.0, general + rng.uniform(-1.5, 1.5), 300, hit_limit=[3.0, 3.0]),
             "mmlu_pro": summary(31.0, general + rng.uniform(-1.5, 1.5), 500), "neutral": {"kl": kl * rng.uniform(0.7, 1.3)}}
    for name, data in files.items():
        (run / "eval" / f"{name}-test.json").write_text(json.dumps(data, indent=2))


FRACTIONS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
MLP_OUT = 1024 * 3584
# made-up E2 shape of each kind of edit: (tensors as (layer, kind, changed weights), rank, fit R², how concentrated)
EDIT_SHAPES = {
    ("facts", "agent_basic"): ([(8, "mlp.down_proj", MLP_OUT)], 10, 0.12, 0.30),
    ("facts", "agent_interp"): ([(8, "mlp.down_proj", MLP_OUT), (9, "mlp.down_proj", MLP_OUT)], 10, 0.10, 0.25),
    ("facts", "random"): ([(8, "mlp.down_proj", MLP_OUT)], 900, 0.00, 1.0),
    ("facts", "finetune"): ([(8, "mlp.down_proj", MLP_OUT)], 30, 0.05, 0.60),
    ("loops", "agent_basic"): ([(15, "mlp.down_proj", 65536), (19, "mlp.down_proj", 65536)], 1, 0.97, 0.08),
    ("loops", "agent_interp"): ([(11, "mlp.down_proj", 32768), (15, "mlp.down_proj", 65536),
                                 (19, "mlp.down_proj", 65536)], 1, 0.95, 0.06),
    ("loops", "random"): ([(15, "mlp.down_proj", 65536), (19, "mlp.down_proj", 65536)], 64, 0.02, 1.0),
    ("loops", "direction"): ([(layer, kind, MLP_OUT if kind == "mlp.down_proj" else 1024 * 2048)
                              for layer in range(24) for kind in ("mlp.down_proj", "self_attn.o_proj" if layer % 4 == 3
                                                                  else "linear_attn.out_proj")], 1, 1.00, 0.05),
    ("loops", "finetune"): ([(15, "mlp.down_proj", MLP_OUT), (19, "mlp.down_proj", MLP_OUT)], 25, 0.04, 0.60),
}


def write_e2(names):
    import math

    for name in names:
        task, condition, _ = name.split("-")
        tensors, rank, r2, spread = EDIT_SHAPES[(task, condition)]
        rows = [{"tensor": f"model.layers.{layer}.{kind}.weight", "layer": layer, "component": "mlp_out" if "mlp" in kind
                 else "attn_out", "changed": changed, "size": MLP_OUT if "mlp" in kind else 1024 * 2048,
                 "share": changed / (MLP_OUT if "mlp" in kind else 1024 * 2048), "relative_norm": rng.uniform(0.01, 0.05)}
                for layer, kind, changed in tensors]
        total = sum(row["changed"] for row in rows)
        gain = EFFECTS[task][condition] + rng.uniform(-5, 5)
        keep = lambda f: gain * (1 - math.exp(-f / spread)) / (1 - math.exp(-1 / spread))
        curve = [{"fraction": f, "weights": round(total * f), "gain": keep(f)} for f in FRACTIONS] if gain > 0 else None
        summary = {"edit_map": {"tensors": rows, "changed": total, "share_of_model": total / 752e6},
                   "structure": {row["tensor"]: {"rank_for_99pct": rank, "projection_r2": r2 + rng.uniform(-0.02, 0.02)}
                                 for row in rows},
                   "select_original": 12.0 if task == "loops" else 3.0, "select_gain": gain}
        if curve:
            summary["columns_curve"] = curve
            summary["smallest_fraction_for_90pct"] = min(p["fraction"] for p in curve if p["gain"] >= 0.9 * gain)
            if 1 < len(rows) < 5:
                summary["tensor_alone"] = {row["tensor"]: gain * rng.uniform(0.2, 0.6) for row in rows}
        out = ROOT / "e2" / name
        out.mkdir(parents=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))

    for task in EFFECTS:
        runs = sorted(n for n in names if n.startswith(task + "-"))
        pairs = {}
        for i, a in enumerate(runs):
            for b in runs[i + 1:]:
                ca, cb = a.split("-")[1], b.split("-")[1]
                if "random" in (ca, cb):
                    cosine = rng.uniform(-0.03, 0.03)
                elif ca == cb:
                    cosine = rng.uniform(0.55, 0.8)
                elif {ca, cb} == {"agent_basic", "agent_interp"}:
                    cosine = rng.uniform(0.4, 0.6)
                else:
                    cosine = rng.uniform(0.1, 0.35)
                pairs[f"{a} vs {b}"] = {"overlap": abs(cosine) * 0.8, "cosine": cosine,
                                        "top_direction": min(1.0, abs(cosine) + 0.2)}
        (ROOT / "e2" / f"similarity-{task}.json").write_text(json.dumps(pairs, indent=2))


def write_e3():
    columns = ["loops", "markdown", "facts", "addition", "gsm8k", "mmlu_pro", "neutral"]
    rows = ["loops-agent_interp-r2", "facts-agent_interp-r1", "loops-direction-r1", "loops-finetune-r1",
            "facts-finetune-r1", "loops-random-r1", "facts-random-r1"]
    lines = ["run," + ",".join(columns)]
    own = {}
    for name in rows:
        task, condition = name.split("-")[:2]
        evals = {c: json.loads((ROOT / "e1" / name / "eval" / f"{c}-test.json").read_text())
                 for c in (task, "gsm8k", "mmlu_pro", "neutral")}
        spill = 4.0 if condition == "finetune" else 1.5 if condition == "direction" else 0.8
        values = [evals[c]["diff"] if c in (task, "gsm8k", "mmlu_pro") else evals[c]["kl"] if c == "neutral"
                  else rng.uniform(-spill, spill / 3) for c in columns]
        own[name] = evals[task]["diff"]
        lines.append(name + "," + ",".join(f"{v:.4f}" for v in values))
    (ROOT / "e3").mkdir(parents=True)
    (ROOT / "e3" / "matrix.csv").write_text("\n".join(lines) + "\n")
    loops_alone, facts_alone = own["loops-agent_interp-r2"], own["facts-agent_interp-r1"]
    loops_combined, facts_combined = loops_alone * 0.78, facts_alone * 0.95
    (ROOT / "e3" / "combos.csv").write_text(
        "pair,task,alone,combined,interference_ratio\n"
        f"loops+facts,loops,{loops_alone:.1f},{loops_combined:.1f},{loops_combined / loops_alone:.2f}\n"
        f"loops+facts,facts,{facts_alone:.1f},{facts_combined:.1f},{facts_combined / facts_alone:.2f}\n")


def main():
    shutil.rmtree(ROOT, ignore_errors=True)
    names = []
    for task, effects in EFFECTS.items():
        for condition in effects:
            replicates = 3 if condition.startswith("agent") or condition == "random" else 1
            for replicate in range(1, replicates + 1):
                write_run(task, condition, replicate)
                names.append(f"{task}-{condition}-r{replicate}")
    write_e2(names)
    write_e3()
    (ROOT / "DEMO").write_text("Everything in this folder is made up.\n")
    (ROOT / "reviews.csv").write_text(
        "run,verdict,category,notes,reviewer\n"
        "loops-agent_basic-r2,invalid,rule break,(demo) Turn 3 opened the test file; excluded from results.,TH\n"
        "facts-agent_basic-r1,valid,works,(demo) Rank-one edits in layer 8; matches ROME's usual layer.,TH\n")
    print(f"Wrote made-up runs to {ROOT}")


if __name__ == "__main__":
    main()
