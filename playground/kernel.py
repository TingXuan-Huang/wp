"""A long-lived Python process that holds the model and runs the agent's code.

`Kernel` is the parent-side handle. `main` runs in the child, inside the network sandbox.
They exchange one JSON line per request. Variables persist between cells, like a notebook.
"""
import contextlib
import importlib
import inspect
import io
import json
import os
import select
import subprocess
import sys
import traceback
from pathlib import Path

from playground.config import ROOT, load_run_config
from playground.sandbox import command_prefix

LOAD_TIMEOUT_S = 1800


def fetch_model(cfg):
    """Download the model first, because the kernel may have no network."""
    hf_id = cfg["model_spec"]["hf_id"]
    if not Path(hf_id).exists():
        from huggingface_hub import snapshot_download
        snapshot_download(hf_id)


class Kernel:
    def __init__(self, cfg, run_dir):
        self.cfg, self.run_dir = cfg, Path(run_dir)
        self.start()

    def start(self):
        env = dict(os.environ)
        if not self.cfg["internet"]["kernel"]:
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        env.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")   # the sandbox only allows writes here and the run
        command = command_prefix(self.cfg["internet"]["kernel"], self.run_dir.resolve()) + [
            sys.executable, "-u", "-m", "playground.kernel", str(self.run_dir)]
        self.process = subprocess.Popen(
            command, cwd=ROOT, env=env, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=open(self.run_dir / "kernel.log", "a"))
        try:
            self.info = self._read(LOAD_TIMEOUT_S)
        except (TimeoutError, EOFError) as error:
            raise RuntimeError(f"Kernel failed to start; see {self.run_dir}/kernel.log") from error

    def _read(self, timeout):
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError
        line = self.process.stdout.readline()
        if not line:
            raise EOFError
        return json.loads(line)

    def run(self, code):
        """Run a cell. Returns {"output", "passes_used"}. A timeout or crash restarts the kernel."""
        try:
            self.process.stdin.write(json.dumps({"code": code}) + "\n")
            self.process.stdin.flush()
            return self._read(self.cfg["budget"]["cell_timeout_s"])
        except (TimeoutError, EOFError, BrokenPipeError) as error:
            self.stop()
            self.start()
            return {"output": f"[Kernel restarted after {type(error).__name__}. Variables are gone. "
                              "Saved edits and the pass count were kept.]",
                    "passes_used": self.info["passes_used"]}

    def stop(self):
        self.process.kill()
        self.process.wait()


def load_tools(session, namespace, module_names):
    """Give the agent every public function in the listed modules (the config's `tools`), and describe each one."""
    docs = []
    for module_name in module_names:
        module = importlib.import_module(module_name)
        module.session = session
        for name, function in vars(module).items():
            if inspect.isfunction(function) and function.__module__ == module.__name__ and name[0] != "_":
                namespace[name] = function
                docs.append(f"- {name}{inspect.signature(function)}: {inspect.getdoc(function)}")
    return "\n".join(docs)


def execute(code, namespace):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            exec(compile(code, "<cell>", "exec"), namespace)
        except BaseException:
            traceback.print_exc(file=buffer)
    return buffer.getvalue()


def main(run_dir):
    # Keep the real stdout for the protocol, so stray prints can't corrupt it.
    channel = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)

    def send(message):
        channel.write(json.dumps(message) + "\n")
        channel.flush()

    import numpy
    import torch
    from playground.session import Session

    cfg = load_run_config(run_dir)
    session = Session(cfg, run_dir)
    namespace = {"torch": torch, "np": numpy}
    send({"tools": load_tools(session, namespace, cfg["tools"]), "device": session.device,
          "passes_used": session.passes_used})
    for line in sys.stdin:
        output = execute(json.loads(line)["code"], namespace)
        send({"output": output, "passes_used": session.passes_used, "edited": sorted(session.originals)})


if __name__ == "__main__":
    main(sys.argv[1])
