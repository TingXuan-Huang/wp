"""Task D (add): add two 6- to 8-digit numbers, answering with only the number."""
import random

from tasks import write_jsonl


def _items(seed, n, seen):
    rng, items = random.Random(seed), []
    while len(items) < n:
        a, b = rng.randint(10 ** 5, 10 ** 8 - 1), rng.randint(10 ** 5, 10 ** 8 - 1)
        if (a, b) not in seen:
            seen.add((a, b))
            items.append({"prompt": f"What is {a} + {b}? Reply with only the number.", "answer": str(a + b),
                          "scorer": "score_number"})
    return items


def build():
    seen = set()   # no problem appears in two splits
    write_jsonl("addition/practice.jsonl", _items(1, 1000, seen))
    write_jsonl("addition/select.jsonl", _items(2, 300, seen))
    write_jsonl("addition/test.jsonl", _items(3, 500, seen))


def finetune_examples(session, items):
    """For the fine-tuning baseline: each problem followed by the right answer."""
    return [(session._render(item["prompt"]), item["answer"] + session.tokenizer.eos_token) for item in items]
