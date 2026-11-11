"""Every backend has one method: send(text) -> reply. The conversation is remembered between calls."""
from pathlib import Path

SYSTEM_PROMPT = "Answer the user's request."   # the task itself is in prompts/agent/


def make_backend(cfg, run_dir):
    kind = cfg["backend"]
    options = dict(cfg["backends"][kind], web_search=cfg["internet"]["web_search"])
    if kind == "claude_api":
        from backends.api import ClaudeAPI
        return ClaudeAPI(**options)
    if kind == "openai_api":
        from backends.api import OpenAIAPI
        return OpenAIAPI(**options)
    if kind == "scripted":
        from backends.scripted import Scripted
        return Scripted(**options)

    from backends.cli import ClaudeCLI, CodexCLI
    workdir = Path(run_dir) / "agent_cwd"      # empty, so no project files reach the agent
    workdir.mkdir(exist_ok=True)
    if kind == "claude_cli":
        return ClaudeCLI(workdir=workdir, **options)
    if kind == "codex_cli":
        return CodexCLI(workdir=workdir, **options)
    raise ValueError(f"Unknown backend: {kind}")
