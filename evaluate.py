"""Score the original and edited weights of a run.

    python evaluate.py runs/<run>                           # the datasets in the run's eval_data
    python evaluate.py runs/<run> --data facts gsm8k --limit 50
    python evaluate.py runs/<run> --force generation.max_new_tokens=4096

Dataset names and their settings are in tasks/__init__.py. Results go to runs/<run>/eval/.
"""
import argparse

from analysis.evaluation import evaluate_run, format_summary
from playground.config import apply_overrides, load_run_config
from playground.session import Session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--data", nargs="+", help="dataset names (default: the run's eval_data)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, help="only the first N items of each dataset")
    parser.add_argument("--force", action="store_true", help="recompute results already on disk")
    args, overrides = parser.parse_known_args()
    cfg = apply_overrides(load_run_config(args.run_dir), overrides)
    summaries = evaluate_run(Session(cfg), args.run_dir, args.data or cfg["eval_data"], args.split, args.limit,
                             args.force)
    for name, summary in summaries.items():
        print(format_summary(name, summary))


if __name__ == "__main__":
    main()
