"""Practice data for the agent. It never sees the evaluation data."""
import json
import random

from playground.config import ROOT
from playground.scoring import make_scorer

session = None   # set by the kernel


def practice(n=None, seed=0, split="practice"):
    """Return up to n shuffled practice items (dicts with at least a "prompt"). split="select" gives a second,
    separate set for checking an edit before and after, if this experiment has one."""
    path = session.cfg["practice_data"] if split == "practice" else session.cfg["select_data"]
    if not path:
        raise ValueError(f"This experiment has no {split} split")
    items = [json.loads(line) for line in (ROOT / path).read_text().splitlines()]
    random.Random(seed).shuffle(items)
    return items[:n]


def grade(output, item):
    """Return True if a generated text is correct for a practice item (or for a plain answer string)."""
    return make_scorer(session.cfg)(output, item)
