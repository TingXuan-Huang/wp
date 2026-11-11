"""The datasets used by the experiments, and how each one is evaluated. Build them all with: python -m tasks.build

Each entry in DATASETS can set:
  dir / file   the data is in prompts/test/<dir>/<split>.jsonl, or in one fixed file
  thinking     use the chat template with thinking on or off (left out: the run's model decides)
  generation   sampling settings for this dataset (left out: the run's settings)
  samples      answers per item (left out: the config's samples_per_problem)
  kind         "generate" (default), "logprob" (compare two continuations) or "kl" (side effects on neutral text)
"""
import json

from playground.config import ROOT

DATA = ROOT / "prompts" / "test"
MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
THINKING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}      # Qwen's recommended
NO_THINKING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}    # settings, per mode

DATASETS = {
    # E1 tasks
    "loops": {"dir": "loops", "thinking": True, "generation": {**THINKING, "max_new_tokens": 8192}, "samples": 1},
    "markdown": {"dir": "markdown", "thinking": False, "generation": {**NO_THINKING, "max_new_tokens": 1024},
                 "samples": 1},
    "facts": {"dir": "facts", "kind": "logprob"},
    "addition": {"dir": "addition", "thinking": False, "generation": {**NO_THINKING, "max_new_tokens": 32},
                 "samples": 1},
    # The general panel: the same conditions for every edit
    "gsm8k": {"file": "general/gsm8k.jsonl", "thinking": False,
              "generation": {**NO_THINKING, "max_new_tokens": 2048}, "samples": 1},
    "mmlu_pro": {"file": "general/mmlu_pro.jsonl", "thinking": False,
                 "generation": {**NO_THINKING, "max_new_tokens": 16}, "samples": 1},
    "neutral": {"file": "general/neutral.jsonl", "kind": "kl"},
    # configs/qwen_math.yaml and configs/smoke.yaml: the run's own settings
    "math500": {"file": "math500.jsonl"},
    "gsm8k_test": {"file": "gsm8k_test.jsonl"},
    "smoke": {"file": "math_smoke.jsonl"},
}
E1_TASKS = ["loops", "markdown", "facts", "addition"]
GENERAL_PANEL = ["gsm8k", "mmlu_pro", "neutral"]


def dataset_path(name, split="test"):
    spec = DATASETS[name]
    return DATA / (f"{spec['dir']}/{split}.jsonl" if "dir" in spec else spec["file"])


def write_jsonl(relative_path, items):
    path = DATA / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item) + "\n" for item in items))
    print(f"wrote prompts/test/{relative_path} ({len(items)} items)")
