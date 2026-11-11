"""Task C (redirect): rewrite 10 facts from CounterFact (Meng et al. 2022) and leave other facts alone.

Prompts are raw completions ("The mother tongue of Danielle Darrieux is"), scored like ROME: an item counts as correct
when the expected target has a higher log-probability than the other one. Groups:
  rewrite        the 10 edit prompts, expecting the new target (efficacy)
  paraphrase     reworded edit prompts the agent never sees, expecting the new target (generalization)
  neighborhood   other subjects with the same true answer, expecting it unchanged (specificity)
  other_fact     unrelated facts, expecting them unchanged (practice data only)
"""
from datasets import load_dataset

from tasks import write_jsonl

N_FACTS = 10


def _item(prompt, rewrite, expect, group):
    return {"prompt": prompt, "target_new": " " + rewrite["target_new"]["str"],
            "target_true": " " + rewrite["target_true"]["str"], "expect": expect, "group": group,
            "scorer": "score_fact"}


def build():
    rows = load_dataset("azhx/counterfact", split="train").shuffle(seed=0)
    edits, others = rows.select(range(N_FACTS)), rows.select(range(N_FACTS, N_FACTS + 100))
    edit_prompts = [_item(r["requested_rewrite"]["prompt"].format(r["requested_rewrite"]["subject"]),
                          r["requested_rewrite"], "new", "rewrite") for r in edits]
    other_facts = [_item(r["requested_rewrite"]["prompt"].format(r["requested_rewrite"]["subject"]),
                         r["requested_rewrite"], "true", "other_fact") for r in others]
    write_jsonl("facts/practice.jsonl", edit_prompts + other_facts)
    write_jsonl("facts/select.jsonl",
                [_item(r["paraphrase_prompts"][0], r["requested_rewrite"], "new", "paraphrase") for r in edits]
                + [_item(p, r["requested_rewrite"], "true", "neighborhood") for r in edits
                   for p in r["neighborhood_prompts"][:5]])
    write_jsonl("facts/test.jsonl",
                edit_prompts
                + [_item(r["paraphrase_prompts"][1], r["requested_rewrite"], "new", "paraphrase") for r in edits]
                + [_item(p, r["requested_rewrite"], "true", "neighborhood") for r in edits
                   for p in r["neighborhood_prompts"][5:]])


def finetune_examples(session, items):
    """For the fine-tuning baseline: each edit prompt followed by its new target (FT-L in the ROME paper)."""
    return [(item["prompt"], item["target_new"]) for item in items if item["group"] == "rewrite"]
