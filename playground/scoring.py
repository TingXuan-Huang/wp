"""Grade model outputs. A data item can name its grader in a "scorer" field; otherwise the config's `scorer` is used.
To grade a new kind of data, add a function here that takes (text, item) and returns True or False."""
import re

MARKDOWN = re.compile(r"^\s{0,3}#{1,6}\s|\*\*|__|^\s*[-*+]\s+\S|^\s*\d+[.)]\s+\S|```|^\s*>\s|^\s*\|.*\|\s*$",
                      re.MULTILINE)


def make_scorer(cfg, thinking=None):
    """Return score(output, item) -> bool. `item` is a data item or just its answer.
    `thinking` says whether answers have a thinking section (default: the config's model decides)."""
    if thinking is None:
        thinking = cfg["model_spec"].get("chat_template", {}).get("enable_thinking", False)

    def score(output, item):
        if not isinstance(item, dict):
            item = {"answer": item}
        if thinking and "</think>" not in output:   # it ran out of tokens while thinking, so it never answered
            return False
        grader = globals()[item.get("scorer", cfg["scorer"])]
        return bool(grader(output.split("</think>")[-1], item))

    return score


def last_boxed(text):
    """Return the contents of the last \\boxed{...}, handling nested braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    start += len("\\boxed{")
    depth = 1
    for i, char in enumerate(text[start:], start):
        depth += (char == "{") - (char == "}")
        if depth == 0:
            return text[start:i]
    return None


def score_math(text, item):
    """True if the last \\boxed{} in the text equals the reference answer."""
    from math_verify import parse, verify

    answer = str(item["answer"])
    predicted = last_boxed(text)
    if predicted is None:
        return False
    if predicted.replace(" ", "") == answer.replace(" ", ""):
        return True
    return verify(parse(f"${answer}$"), parse(f"${predicted}$"))


def score_number(text, item):
    """True if the first number in the text equals the reference (commas ignored)."""
    match = re.search(r"-?\d[\d,]*", text)
    return match is not None and match.group().replace(",", "") == str(item["answer"])


def score_plain_text(text, item):
    """True if the answer has at least 20 words and no markdown formatting."""
    return len(text.split()) >= 20 and not MARKDOWN.search(text)


def score_choice(text, item):
    """True if the first standalone option letter (A-J) in the text is the right one."""
    match = re.search(r"\b([A-J])\b", text)
    return match is not None and match.group(1) == item["answer"]


def score_fact(text, item):
    """For the agent's quick checks: the expected target appears in the text.
    Evaluation compares log-probabilities instead (see tasks/facts.py)."""
    target = item["target_new"] if item["expect"] == "new" else item["target_true"]
    return target.strip().lower() in text.lower()
