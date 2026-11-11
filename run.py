"""Let an agent edit a model's weights.

    python run.py configs/qwen_math.yaml
    python run.py configs/qwen_math.yaml backend=codex_cli budget.forward_passes=10 internet.web_search=true
"""
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from backends import make_backend
from playground.config import ROOT, load_config
from playground.kernel import Kernel, fetch_model

FENCE = re.compile(r"```(?:python3?|py)\b", re.IGNORECASE)
CODE_BLOCK = re.compile(FENCE.pattern + r"[ \t]*\n(.*?)^[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE | re.IGNORECASE)


def internet_note(internet):
    search = "You can search the web." if internet["web_search"] else "You cannot search the web."
    python = "Your Python code can reach the network." if internet["kernel"] else "Your Python code has no network."
    return f"{search} {python}"


def build_prompt(cfg, kernel):
    template = (ROOT / cfg["agent_prompt"]).read_text()
    values = {
        "task": (ROOT / cfg["task_prompt"]).read_text().strip() if cfg["task_prompt"] else "",   # first, so its
        "model": cfg["model"],                                                                   # placeholders fill too
        "tools": kernel.info["tools"],
        "device": kernel.info["device"],
        "internet": internet_note(cfg["internet"]),
        "forward_passes": cfg["budget"]["forward_passes"],
        "max_turns": cfg["budget"]["max_turns"],
        "cell_timeout_s": cfg["budget"]["cell_timeout_s"],
        "max_output_chars": cfg["budget"]["max_output_chars"],
        "generation": json.dumps(cfg["generation"]),
    }
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def clip(text, limit):
    if len(text) <= limit:
        return text
    return f"{text[:limit // 2]}\n[... {len(text) - limit} characters cut ...]\n{text[-limit // 2:]}"


def execute_reply(reply, cfg, kernel):
    """Run the first code block in a reply. Returns (code, message for the agent, edited tensors after it ran);
    code and message are None when the agent is done."""
    if not reply.strip():
        return "", "Your reply was empty. Send a code block to continue, or a written summary to finish.", None
    blocks = CODE_BLOCK.findall(reply)
    if not blocks:
        if FENCE.search(reply):
            return "", "Your code block was never closed, so nothing ran. Send it again.", None
        return None, None, None
    code = blocks[0]
    blocked = next((pattern for pattern in cfg["forbid"] if re.search(pattern, code)), None)
    if blocked:
        return code, f"Refused, nothing ran: the code matches the forbidden pattern {blocked}", None
    result = kernel.run(code)
    output = clip(result["output"], cfg["budget"]["max_output_chars"])
    left = cfg["budget"]["forward_passes"] - result["passes_used"]
    note = " Only the first code block was run." if len(blocks) > 1 else ""
    return code, f"```output\n{output}\n```\nForward passes left: {left}.{note}", result.get("edited")


def main():
    cfg = load_config(sys.argv[1], sys.argv[2:])
    run_dir = ROOT / (cfg["run_dir"] or f"runs/{cfg['name']}-{cfg['backend']}-{datetime.now():%Y%m%d-%H%M%S}")
    run_dir.mkdir(parents=True)   # fails if the folder exists, so a run never overwrites another
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Run folder: {run_dir}")

    fetch_model(cfg)
    kernel = Kernel(cfg, run_dir)
    backend = make_backend(cfg, run_dir)
    message = build_prompt(cfg, kernel)
    (run_dir / "prompt.md").write_text(message)
    finished = False
    try:
        with open(run_dir / "transcript.jsonl", "a") as log:
            for turn in range(1, cfg["budget"]["max_turns"] + 1):
                started = time.time()
                reply = backend.send(message)
                code, message, edited = execute_reply(reply, cfg, kernel)
                log.write(json.dumps({"turn": turn, "reply": reply, "code": code, "result": message,
                                      "edited": edited, "seconds": round(time.time() - started),
                                      "info": backend.info}) + "\n")
                log.flush()
                print(f"--- turn {turn} ---\n{reply[:1500]}\n{(message or '')[:1500]}")
                if code is None:
                    finished = True
                    (run_dir / "final_answer.md").write_text(reply)
                    break
        if not finished:
            print("Stopped at budget.max_turns before the agent finished.")
        print(kernel.run("print('Edited tensors:', edited(), '| passes left:', passes_left())")["output"])
        (run_dir / "status.json").write_text(json.dumps({"finished_by_agent": finished, "turns": turn}))
    finally:
        kernel.stop()
    print(f"Next: python evaluate.py {Path(run_dir).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
