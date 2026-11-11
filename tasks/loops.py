"""Task A (remove): stop Qwen's thinking-mode loops on GSM8K. The post's Experiment 8, with an 8,192-token limit."""
from datasets import load_dataset

from playground.scoring import make_scorer
from tasks import write_jsonl
from tasks.general import gsm8k_items


def build():
    train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0)
    write_jsonl("loops/practice.jsonl", gsm8k_items(train.select(range(1000))))
    write_jsonl("loops/select.jsonl", gsm8k_items(train.select(range(1000, 1300))))
    test = load_dataset("openai/gsm8k", "main", split="test")
    write_jsonl("loops/test.jsonl", gsm8k_items(test.select(range(500))))


def contrast_texts(session, items):
    """For the direction-removal baseline: answers that ran into the token limit vs answers that finished."""
    outputs = session.sample([item["prompt"] for item in items])
    prompts = [session._render(item["prompt"]) for item in items]
    looped = [prompt + out["text"][-4000:] for prompt, out in zip(prompts, outputs) if out["hit_limit"]]
    finished = [prompt + out["text"] for prompt, out in zip(prompts, outputs) if not out["hit_limit"]]
    return looped, finished


def finetune_examples(session, items):
    """For the fine-tuning baseline: the model's own correct, finished answers."""
    score = make_scorer(session.cfg)
    outputs = session.sample([item["prompt"] for item in items])
    return [(session._render(item["prompt"]), out["text"] + session.tokenizer.eos_token)
            for item, out in zip(items, outputs) if not out["hit_limit"] and score(out["text"], item)]
