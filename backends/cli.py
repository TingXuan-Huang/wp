"""Subscription backends: drive `claude -p` or `codex exec`, one process per turn.

The CLI's own tools are switched off, so the agent can only act through Python blocks in the kernel.
API keys are removed from the environment so the CLI uses your logged-in subscription.
"""
import json
import os
import subprocess
import time

from backends import SYSTEM_PROMPT

RETRIES = 4
TIMEOUT_S = 4 * 60 * 60     # high-effort turns can be long


def _call(command, text, workdir, drop_env):
    env = {key: value for key, value in os.environ.items() if key not in drop_env}
    for attempt in range(1, RETRIES + 1):
        try:
            done = subprocess.run(command, input=text, cwd=workdir, env=env, text=True,
                                  capture_output=True, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            message = f"no reply within {TIMEOUT_S} seconds"
        else:
            if done.returncode == 0:
                return done.stdout
            message = (done.stderr or done.stdout)[-1000:]
            if "not logged in" in message.lower():   # retrying can't fix this
                raise RuntimeError(f"{command[0]} is not logged in. Log in once in a terminal, then rerun.")
        print(f"[{command[0]} failed, attempt {attempt}/{RETRIES}] {message}")
        if attempt < RETRIES:
            time.sleep(60 * attempt)
    raise RuntimeError(f"{command[0]} kept failing: {message}")


class ClaudeCLI:
    def __init__(self, model, effort, web_search, workdir, bin="claude"):
        web_tools = "WebSearch,WebFetch" if web_search else ""
        self.command = [
            bin, "-p", "--output-format", "json", "--model", model, "--effort", effort,
            "--system-prompt", SYSTEM_PROMPT, "--strict-mcp-config", "--setting-sources", "",
            "--tools", web_tools,
        ] + (["--allowedTools", web_tools] if web_search else [])
        self.workdir = workdir
        self.session_id = None
        self.info = {}

    def send(self, text):
        resume = ["--resume", self.session_id] if self.session_id else []
        output = _call(self.command + resume, text, self.workdir,
                       {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"})
        result = json.loads(output)
        if result.get("is_error"):
            raise RuntimeError(result.get("result"))
        self.session_id = result["session_id"]
        self.info = {"usage": result.get("usage"), "cost_usd_equivalent": result.get("total_cost_usd")}
        return result["result"]


class CodexCLI:
    FEATURES_OFF = ["shell_tool", "unified_exec", "view_image", "image_generation", "multi_agent",
                    "apps", "plugins", "browser_use", "computer_use", "hooks", "goals"]

    def __init__(self, model, effort, web_search, workdir, bin="codex"):
        instructions = workdir.parent / "codex_instructions.md"
        instructions.write_text(SYSTEM_PROMPT)
        settings = {
            "model_reasoning_effort": f'"{effort}"',
            "approval_policy": '"never"',
            "sandbox_mode": '"read-only"',
            "web_search": '"live"' if web_search else '"disabled"',
            "model_instructions_file": f'"{instructions.resolve()}"',
            "agents.enabled": "false",
            **{f"features.{name}": "false" for name in self.FEATURES_OFF},
        }
        self.command = [bin, "exec", "--json", "--skip-git-repo-check", "--ignore-user-config", "-m", model]
        for key, value in settings.items():
            self.command += ["-c", f"{key}={value}"]
        self.workdir = workdir
        self.thread_id = None
        self.info = {}

    def send(self, text):
        resume = ["resume", self.thread_id] if self.thread_id else []
        output = _call(self.command + resume + ["-"], text, self.workdir, {"OPENAI_API_KEY", "CODEX_API_KEY"})
        messages = []
        for line in output.splitlines():
            event = json.loads(line) if line.startswith("{") else {}
            kind = event.get("type")
            if kind == "thread.started":
                self.thread_id = event["thread_id"]
            elif kind == "item.completed" and event["item"].get("type") == "agent_message":
                messages.append(event["item"]["text"])
            elif kind == "turn.completed":
                self.info = {"usage": event.get("usage")}
            elif kind == "turn.failed":
                raise RuntimeError(event.get("error"))
        return "\n".join(messages)
