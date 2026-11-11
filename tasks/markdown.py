"""Task B (remove): answer general questions in plain prose, without markdown formatting."""
import random
import re

from datasets import load_dataset

from playground.scoring import MARKDOWN
from tasks import write_jsonl

CATEGORIES = {"open_qa", "general_qa", "brainstorming"}


def build():
    rows = [row for row in load_dataset("databricks/databricks-dolly-15k", split="train")
            if row["category"] in CATEGORIES and not row["context"].strip()]
    random.Random(0).shuffle(rows)
    items = [{"prompt": row["instruction"], "scorer": "score_plain_text"} for row in rows]
    write_jsonl("markdown/practice.jsonl", items[:1000])
    write_jsonl("markdown/select.jsonl", items[1000:1300])
    write_jsonl("markdown/test.jsonl", items[1300:1800])


def strip_markdown(text):
    """Remove markdown syntax but keep the words."""
    text = re.sub(r"```\w*", "", text)
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    return text.replace("|", " ")


def _answers(session, items):
    outputs = session.sample([item["prompt"] for item in items])
    return [(session._render(item["prompt"]), out["text"]) for item, out in zip(items, outputs)]


def contrast_texts(session, items):
    """For the direction-removal baseline: the model's markdown answers vs the same answers with markdown removed."""
    pairs = [(prompt, text) for prompt, text in _answers(session, items) if MARKDOWN.search(text)]
    return [prompt + text for prompt, text in pairs], [prompt + strip_markdown(text) for prompt, text in pairs]


def finetune_examples(session, items):
    """For the fine-tuning baseline: the model's own answers with markdown removed."""
    return [(prompt, strip_markdown(text) + session.tokenizer.eos_token) for prompt, text in _answers(session, items)]
