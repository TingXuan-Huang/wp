"""E2: what each edit changed. See experiment.md.

For every finished E1 run with edits:
  1. edit map and structure (T1, T2)
  2. how much of the gain each edited tensor carries alone, and the smallest share of changed columns that keeps
     90% of the gain, measured on the task's select split (not the test split)
Then, per task, similarity between every pair of edits (T4).

    python experiments/e2.py                       # all runs
    python experiments/e2.py --tasks facts --limit 100

Writes runs/e2/<run>/summary.json and figures, runs/e2/similarity-<task>.png and runs/e2/report.md.
"""
import argparse
import itertools
import json

from common import E2_DIR, finished_runs, has_edits, run_info, write_json

from analysis.edits import (apply_deltas, edit_map, edit_structure, keep_top_columns, load_deltas,  # noqa: E402
                            similarity, top_directions)
from analysis.evaluation import accuracy, load_items, score_records  # noqa: E402
from analysis.plots import curve, heatmap  # noqa: E402
from playground.config import load_run_config  # noqa: E402
from playground.session import Session  # noqa: E402
from tasks import E1_TASKS  # noqa: E402

FRACTIONS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]


def score_on_select(session, task, items, deltas):
    apply_deltas(session, deltas)
    return accuracy(score_records(session, task, items, "edited"))


def analyse(session, run, limit):
    task = run_info(run)["task"]
    out = E2_DIR / run.name
    out.mkdir(parents=True, exist_ok=True)
    session.reset_edits()
    deltas = load_deltas(session, run)
    summary = {"edit_map": edit_map(session, deltas),
               "structure": {name: edit_structure(d, session.params[name].float().cpu()) for name, d in deltas.items()}}
    layers = sorted({row["layer"] for row in summary["edit_map"]["tensors"]})
    components = sorted({row["component"] for row in summary["edit_map"]["tensors"]})
    grid = [[sum(100 * r["share"] for r in summary["edit_map"]["tensors"] if r["layer"] == layer and
                 r["component"] == comp) or float("nan") for comp in components] for layer in layers]
    heatmap(grid, [f"layer {layer}" for layer in layers], components, out / "edit_map.png",
            f"{run.name}: % of weights changed")

    items = load_items(task, "select", limit)
    base = score_on_select(session, task, items, {})
    gain = score_on_select(session, task, items, deltas) - base
    summary.update(select_original=base, select_gain=gain)
    if gain > 0:
        summary["tensor_alone"] = {name: score_on_select(session, task, items, {name: d}) - base
                                   for name, d in deltas.items()} if len(deltas) > 1 else {}
        points = []
        for fraction in FRACTIONS:
            kept = keep_top_columns(deltas, fraction)
            weights = sum(int((d != 0).sum()) for d in kept.values())
            points.append({"fraction": fraction, "weights": weights,
                           "gain": score_on_select(session, task, items, kept) - base})
        summary["columns_curve"] = points
        enough = [p for p in points if p["gain"] >= 0.9 * gain]
        summary["smallest_fraction_for_90pct"] = min(p["fraction"] for p in enough)
        curve([p["weights"] for p in points], [p["gain"] for p in points], out / "columns_curve.png",
              f"{run.name}: gain vs weights kept", "changed weights kept", "gain on select split (points)")
    session.reset_edits()
    write_json(out / "summary.json", summary)
    print(f"{run.name}: {summary['edit_map']['changed']} weights changed, select gain {gain:+.1f}")
    return summary


def similarities(session, runs, task):
    deltas = {run.name: load_deltas(session, run) for run in runs}
    directions = {name: top_directions(d) for name, d in deltas.items()}
    names = list(deltas)
    pairs = {f"{a} vs {b}": similarity(deltas[a], deltas[b], directions[a], directions[b])
             for a, b in itertools.combinations(names, 2)}
    matrix = [[1.0 if a == b else pairs.get(f"{a} vs {b}", pairs.get(f"{b} vs {a}"))["cosine"] for b in names]
              for a in names]
    heatmap(matrix, names, names, E2_DIR / f"similarity-{task}.png", f"{task}: cosine between edits", centered=True)
    write_json(E2_DIR / f"similarity-{task}.json", pairs)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=E1_TASKS, choices=E1_TASKS)
    parser.add_argument("--limit", type=int, help="only the first N select items (loops is slow)")
    parser.add_argument("--batch-size", type=int, help="answers generated at once")
    args = parser.parse_args()
    runs = [run for run in finished_runs() if has_edits(run) and run_info(run)["task"] in args.tasks]
    if not runs:
        print("No finished E1 runs with edits yet.")
        return
    session = Session(load_run_config(runs[0]))
    session.cfg["batch_size"] = args.batch_size or session.cfg["batch_size"]
    lines = ["# E2 report", "", "| run | weights changed | select gain | smallest column share for 90% of gain |",
             "|---|---|---|---|"]
    for run in runs:
        summary = analyse(session, run, args.limit)
        lines.append(f"| {run.name} | {summary['edit_map']['changed']:,} | {summary['select_gain']:+.1f} | "
                     f"{summary.get('smallest_fraction_for_90pct', '')} |")
    lines += ["", "## Similarity between edits", "", "| pair | column overlap | cosine | top direction |cos| |",
              "|---|---|---|---|"]
    for task in args.tasks:
        task_runs = [run for run in runs if run_info(run)["task"] == task]
        if len(task_runs) > 1:
            for pair, value in similarities(session, task_runs, task).items():
                top = "" if value["top_direction"] is None else f"{value['top_direction']:.2f}"
                lines.append(f"| {pair} | {value['overlap']:.2f} | {value['cosine']:.2f} | {top} |")
    (E2_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {E2_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
