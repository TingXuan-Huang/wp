"""E1: narrow edits. See experiment.md for the full instructions.

    python experiments/e1.py agents --parallel 4      # agent runs: tasks x {agent_basic, agent_interp} x replicates
    python experiments/e1.py baselines                # random, direction removal and fine-tuning baselines
    python experiments/e1.py evaluate                 # score every finished run on its eval_data
    python experiments/e1.py report                   # runs/e1/report.md and report.csv

Options: --tasks facts addition   --replicates 3   --conditions agent_basic
         --limit 50 and --batch-size 64 (evaluate: first N items; answers generated at once)
Anything else is passed to run.py as a config override, e.g. backend=codex_cli budget.forward_passes=10.
Every step can be rerun: finished runs are skipped, and unfinished ones are moved aside and started again.
"""
import argparse
import csv
import json
import sys

from common import AGENT_CONDITIONS, E1_DIR, ROOT, finished_runs, prepare_run_dir, run_commands, run_info

from tasks import E1_TASKS  # noqa: E402

REMOVE_TASKS = {"loops", "markdown"}   # tasks the direction-removal baseline applies to


def agents(args, overrides):
    commands = []
    for task in args.tasks:
        for condition in args.conditions:
            for replicate in range(1, args.replicates + 1):
                run_dir = E1_DIR / f"{task}-{condition}-r{replicate}"
                if not prepare_run_dir(run_dir):
                    continue
                command = [sys.executable, "run.py", f"configs/e1/{task}.yaml",
                           f"run_dir={run_dir.relative_to(ROOT)}", f"tools={json.dumps(AGENT_CONDITIONS[condition])}",
                           *overrides]
                commands.append((command, E1_DIR / "logs" / f"{run_dir.name}.log"))
    print(f"{len(commands)} agent runs to do, {args.parallel} at a time")
    failed = run_commands(commands, args.parallel)
    if failed:
        print("Failed runs (see their logs):", *failed, sep="\n  ")


def baselines(args):
    from baselines import direction_removal, finetune, random_like

    for task in args.tasks:
        for replicate in range(1, args.replicates + 1):    # one random edit per agent_basic replicate
            like = E1_DIR / f"{task}-agent_basic-r{replicate}"
            out = E1_DIR / f"{task}-random-r{replicate}"
            if (like / "status.json").exists() and any((like / "edits").glob("*")) and prepare_run_dir(out):
                random_like(like, out, seed=replicate)
        out = E1_DIR / f"{task}-direction-r1"
        if task in REMOVE_TASKS and prepare_run_dir(out):
            direction_removal(task, out)
        out = E1_DIR / f"{task}-finetune-r1"
        like = E1_DIR / f"{task}-agent_basic-r1"
        if prepare_run_dir(out):   # loops answers are long, so it learns from the model's answers to 256 problems
            finetune(task, out, like if any((like / "edits").glob("*")) else None, n=256 if task == "loops" else None)


def evaluate(args):
    from analysis.evaluation import evaluate_run, format_summary
    from playground.config import load_run_config
    from playground.session import Session

    runs = [run for run in finished_runs() if run_info(run)["task"] in args.tasks]
    session = None
    for run in runs:
        cfg = load_run_config(run)
        if session is None:                 # all E1 runs share one base model
            session = Session(cfg)
            session.cfg["batch_size"] = args.batch_size or session.cfg["batch_size"]
        print(f"\n== {run.name}")
        for name, summary in evaluate_run(session, run, cfg["eval_data"], limit=args.limit).items():
            print(format_summary(name, summary))


def report(args):
    rows = []
    for run in finished_runs():
        info = run_info(run)
        if info["task"] not in args.tasks:
            continue
        row = {"run": run.name, **info}
        for path in sorted((run / "eval").glob("*-test*.json")):
            name, summary = path.stem.split("-test")[0], json.loads(path.read_text())
            if "kl" in summary:
                row["kl"] = round(summary["kl"], 4)
            else:
                row[name] = round(summary["diff"], 1)
                row[f"{name}_ci"] = f"[{summary['ci'][0]:+.1f}, {summary['ci'][1]:+.1f}]"
                if "hit_limit" in summary and name == info["task"]:
                    row["hit_limit"] = f"{summary['hit_limit'][0]:.0f}->{summary['hit_limit'][1]:.0f}"
        rows.append(row)
    if not rows:
        print("No finished runs with results yet.")
        return
    columns = ["run", "task", "condition", "replicate"] + sorted({k for r in rows for k in r} - {"run", "task",
                                                                  "condition", "replicate"})
    with open(E1_DIR / "report.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# E1 report", "", "Differences are edited minus original, in percentage points, on held-out data.", "",
             "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(row.get(c, "")) for c in columns) + " |" for row in rows]
    (E1_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["agents", "baselines", "evaluate", "report"])
    parser.add_argument("--tasks", nargs="+", default=E1_TASKS, choices=E1_TASKS)
    parser.add_argument("--conditions", nargs="+", default=list(AGENT_CONDITIONS), choices=list(AGENT_CONDITIONS))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, help="evaluate: answers generated at once; keep it the same for "
                                                       "every run, since it affects the random seeds")
    args, overrides = parser.parse_known_args()
    if overrides and args.step != "agents":
        parser.error(f"unexpected arguments: {overrides}")
    {"agents": lambda: agents(args, overrides), "baselines": lambda: baselines(args),
     "evaluate": lambda: evaluate(args), "report": lambda: report(args)}[args.step]()


if __name__ == "__main__":
    main()
