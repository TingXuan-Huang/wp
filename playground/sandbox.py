"""Confine the kernel: writes only to the run folder and temp space, and optionally no network.

Fails loudly rather than running with the network on.
"""
import shutil
import subprocess
import sys

MACOS_PROFILE = """(version 1)(allow default){network}(deny file-write*)
(allow file-write* (subpath (param "RUN_DIR")) (subpath "/dev") (subpath "/private/tmp") (subpath "/private/var/folders"))"""
LINUX_NETWORK_ONLY = [
    ["bwrap", "--dev-bind", "/", "/", "--unshare-net", "--die-with-parent"],
    ["unshare", "--map-root-user", "--net"],
]


def _works(prefix):
    return shutil.which(prefix[0]) and subprocess.run(prefix + ["true"], capture_output=True).returncode == 0


def _linux(internet_on, run_dir):
    confined = ["bwrap", "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev", "--bind", "/tmp", "/tmp",
                "--bind", str(run_dir), str(run_dir), "--die-with-parent"]
    if not internet_on:
        confined.append("--unshare-net")
    if _works(confined):
        return confined
    print("Warning: bwrap can't make the file system read-only here, so the agent's Python can write anywhere "
          "you can.", file=sys.stderr)
    if internet_on:
        return []
    for prefix in LINUX_NETWORK_ONLY:    # try each one, since clusters often disable user namespaces
        if _works(prefix):
            return prefix
    raise RuntimeError("Neither bwrap nor unshare can turn the network off on this machine. "
                       "Set internet.kernel=true to run with the network on.")


def command_prefix(internet_on, run_dir):
    if sys.platform == "darwin":
        profile = MACOS_PROFILE.format(network="" if internet_on else "(deny network*)")
        return ["sandbox-exec", "-D", f"RUN_DIR={run_dir}", "-p", profile]
    return _linux(internet_on, run_dir)
