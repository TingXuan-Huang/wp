"""The general panel (GSM8K, MMLU-Pro, neutral text), plus the math files used by configs/qwen_math.yaml."""
from datasets import load_dataset

from tasks import MATH_SUFFIX, write_jsonl

LETTERS = "ABCDEFGHIJ"


def gsm8k_items(rows):
    return [{"prompt": row["question"] + MATH_SUFFIX, "scorer": "score_math",
             "answer": row["answer"].split("####")[-1].strip().replace(",", "")} for row in rows]


def mmlu_item(row):
    options = "\n".join(f"{LETTERS[k]}. {option}" for k, option in enumerate(row["options"]))
    prompt = f"{row['question']}\n{options}\nAnswer with only the letter of the correct option."
    return {"prompt": prompt, "answer": row["answer"], "scorer": "score_choice"}


def build():
    test = load_dataset("openai/gsm8k", "main", split="test")
    write_jsonl("general/gsm8k.jsonl", gsm8k_items(test.select(range(500, 800))))   # the loops task uses 0-499
    mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=0).select(range(500))
    write_jsonl("general/mmlu_pro.jsonl", [mmlu_item(row) for row in mmlu])
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    paragraphs = [row["text"].strip() for row in wiki if len(row["text"]) > 300 and not row["text"].strip()[0] == "="]
    write_jsonl("general/neutral.jsonl", [{"text": text} for text in paragraphs[:200]])

    write_jsonl("gsm8k_train.jsonl", gsm8k_items(load_dataset("openai/gsm8k", "main", split="train")))
    write_jsonl("gsm8k_test.jsonl", gsm8k_items(test))
    math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    write_jsonl("math500.jsonl", [{"prompt": row["problem"] + MATH_SUFFIX, "answer": row["answer"],
                                   "scorer": "score_math"} for row in math])
