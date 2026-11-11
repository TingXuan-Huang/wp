"""Shared helpers for the experiment scripts. Run folders are named <task>-<condition>-r<replicate>."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

E1_DIR, E2_DIR, E3_DIR = ROOT / "runs" / "e1", ROOT / "runs" / "e2", ROOT / "runs" / "e3"
AGENT_CONDITIONS = {
    "agent_basic": ["tools_basic.weights", "tools_basic.run_model", "tools_basic.data"],
    "agent_interp": ["tools_basic.weights", "tools_basic.run_model", "tools_basic.data", "tools_specific.interp"],
}


def run_info(run_dir):
    task, condition, replicate = Path(run_dir).name.split("-")
    return {"task": task, "condition": condition, "replicate": int(replicate[1:])}


def finished_runs(root=E1_DIR):
    """Run folders that completed, agent runs and baselines alike."""
    return sorted(path.parent for path in Path(root).glob("*/status.json"))


def has_edits(run_dir):
    return any((Path(run_dir) / "edits").glob("*.safetensors"))


def prepare_run_dir(run_dir):
    """True if the run still needs doing. An unfinished earlier attempt is moved aside, never deleted."""
    run_dir = Path(run_dir)
    if (run_dir / "status.json").exists():
        return False
    if run_dir.exists():
        run_dir.rename(run_dir.with_name(f"{run_dir.name}.incomplete-{datetime.now():%Y%m%d-%H%M%S}"))
    return True


def run_commands(commands, parallel):
    """Run (command, log path) pairs, `parallel` at a time. Returns the failed log paths."""
    def run(job):
        command, log = job
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as file:
            code = subprocess.run(command, cwd=ROOT, stdout=file, stderr=subprocess.STDOUT).returncode
        print(f"{'done' if code == 0 else 'FAILED'}: {log}")
        return None if code == 0 else log

    with ThreadPoolExecutor(parallel) as pool:
        return [log for log in pool.map(run, commands) if log]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
