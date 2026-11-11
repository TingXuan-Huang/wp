"""E3: what else each edit affects. See experiment.md.

  1. Transfer matrix: the best agent edit per task (highest select-split gain in E2) and each baseline, scored on
     every E1 task and the general panel.
  2. Combining edits: each pair of best agent edits applied together, scored on both of their tasks.
     Interference ratio = gain when combined / gain alone (1 means no interference).

    python experiments/e3.py
    python experiments/e3.py --limit 100      # first 100 items of each dataset (loops is slow)

Writes runs/e3/matrix.csv, matrix.png, combos.csv and report.md.
"""
import argparse
import csv
import itertools
import json

from common import E2_DIR, E3_DIR, finished_runs, has_edits, prepare_run_dir, run_info

from analysis.edits import load_deltas  # noqa: E402
from analysis.evaluation import evaluate_run  # noqa: E402
from analysis.plots import heatmap  # noqa: E402
from baselines import save_run  # noqa: E402
from playground.config import load_run_config  # noqa: E402
from playground.session import Session  # noqa: E402
from tasks import E1_TASKS, GENERAL_PANEL  # noqa: E402

COLUMNS = E1_TASKS + GENERAL_PANEL


def best_agent_run(task):
    """The agent run with the highest select-split gain in E2, or the first finished one."""
    candidates = [run for run in finished_runs() if run_info(run)["task"] == task
                  and run_info(run)["condition"].startswith("agent") and has_edits(run)]
    def gain(run):
        path = E2_DIR / run.name / "summary.json"
        return json.loads(path.read_text())["select_gain"] if path.exists() else float("-inf")
    return max(candidates, key=gain, default=None)


def value(summary):
    return summary["kl"] if "kl" in summary else summary["diff"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, help="answers generated at once; use the value E1 evaluate used")
    args = parser.parse_args()
    best = {task: best_agent_run(task) for task in E1_TASKS}
    rows = [run for run in best.values() if run]
    rows += [run for run in finished_runs() if run_info(run)["condition"] in ("finetune", "direction")
             or run.name.endswith("random-r1")]
    if not rows:
        print("No finished E1 runs yet.")
        return
    session = Session(load_run_config(rows[0]))
    session.cfg["batch_size"] = args.batch_size or session.cfg["batch_size"]
    E3_DIR.mkdir(parents=True, exist_ok=True)

    matrix = []
    for run in rows:
        summaries = evaluate_run(session, run, COLUMNS, limit=args.limit)
        matrix.append([float(value(summaries[name])) for name in COLUMNS])
        print(run.name, dict(zip(COLUMNS, (round(v, 3) for v in matrix[-1]))))
    with open(E3_DIR / "matrix.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run"] + COLUMNS)
        writer.writerows([[run.name] + row for run, row in zip(rows, matrix)])
    heatmap([row[:-1] for row in matrix], [run.name for run in rows], COLUMNS[:-1], E3_DIR / "matrix.png",
            "Edited minus original (points)", centered=True)

    combos = []
    for a, b in itertools.combinations([task for task in E1_TASKS if best[task]], 2):
        out = E3_DIR / f"{a}+{b}-combo-r1"
        if prepare_run_dir(out):
            session.reset_edits()
            delta_a, delta_b = load_deltas(session, best[a]), load_deltas(session, best[b])
            tensors = {}
            for name in set(delta_a) | set(delta_b):
                original = session.params[name]
                total = delta_a.get(name, 0) + delta_b.get(name, 0)
                tensors[name] = (original.float().cpu() + total).to(original.dtype)
            save_run(out, load_run_config(best[a]), tensors, {"method": "combo", "runs": [str(best[a]), str(best[b])]})
        combined = evaluate_run(session, out, [a, b], limit=args.limit)
        alone = {a: evaluate_run(session, best[a], [a], limit=args.limit)[a],
                 b: evaluate_run(session, best[b], [b], limit=args.limit)[b]}
        for task in (a, b):
            ratio = combined[task]["diff"] / alone[task]["diff"] if alone[task]["diff"] else float("nan")
            combos.append({"pair": f"{a}+{b}", "task": task, "alone": round(alone[task]["diff"], 1),
                           "combined": round(combined[task]["diff"], 1), "interference_ratio": round(ratio, 2)})
    with open(E3_DIR / "combos.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["pair", "task", "alone", "combined", "interference_ratio"])
        writer.writeheader()
        writer.writerows(combos)

    lines = ["# E3 report", "", "Edited minus original in points on held-out data; the last column is the side-effect "
             "KL on neutral text.", "", "| run | " + " | ".join(COLUMNS) + " |", "|---|" + "---|" * len(COLUMNS)]
    lines += [f"| {run.name} | " + " | ".join(f"{v:.3g}" for v in row) + " |" for run, row in zip(rows, matrix)]
    lines += ["", "## Combined edits", "", "| pair | task | gain alone | gain combined | ratio |", "|---|---|---|---|---|"]
    lines += [f"| {c['pair']} | {c['task']} | {c['alone']} | {c['combined']} | {c['interference_ratio']} |"
              for c in combos]
    (E3_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
