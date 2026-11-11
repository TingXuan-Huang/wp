#!/usr/bin/env bash
# Install Python, packages, models and test data into this folder. Safe to rerun.
#
#   bash setup.sh
#   MODELS="tiny" bash setup.sh      # which models to download, from playground/models.yaml
set -euo pipefail

cd -P "$(dirname "$0")"
LOCAL=$PWD/.local
UV=$LOCAL/uv/uv
PY=$PWD/.venv/bin/python
MODELS=${MODELS:-tiny qwen3.5-0.8b}

# Keep everything out of the home directory.
export UV_PYTHON_INSTALL_DIR=$LOCAL/python
export UV_PYTHON_BIN_DIR=$LOCAL/python-bin
export UV_CACHE_DIR=$LOCAL/uv-cache
export UV_PYTHON_PREFERENCE=only-managed
export UV_NO_MODIFY_PATH=1
export HF_HOME=$PWD/models

echo "==> uv"
if ! "$UV" --version >/dev/null 2>&1; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) TARGET=x86_64-unknown-linux-gnu ;;
    Darwin-arm64) TARGET=aarch64-apple-darwin ;;
    *) echo "Unsupported platform: $(uname -s) $(uname -m)" >&2; exit 1 ;;
  esac
  mkdir -p "$LOCAL/uv"
  curl -fsSL "https://github.com/astral-sh/uv/releases/latest/download/uv-$TARGET.tar.gz" |
    tar -xz -C "$LOCAL/uv" --strip-components=1
fi

echo "==> Python 3.12 and packages"
"$PY" -c "" >/dev/null 2>&1 || "$UV" venv .venv --python 3.12 --clear
"$UV" pip install --python "$PY" -r requirements.txt
if [ "$(uname -s)" = Linux ] && command -v nvcc >/dev/null; then    # optional Qwen speedup; must be compiled
  "$UV" pip install --python "$PY" causal-conv1d || echo "causal-conv1d did not build; Qwen will use a slower fallback."
fi
"$UV" cache clean >/dev/null 2>&1 || true

echo "==> Models: $MODELS"
# shellcheck disable=SC2086
"$PY" -c '
import sys, yaml
from huggingface_hub import snapshot_download
registry = yaml.safe_load(open("playground/models.yaml"))
for name in sys.argv[1:]:
    print(name, "->", snapshot_download(registry[name]["hf_id"]))
' $MODELS

echo "==> Test data"
[ -s prompts/test/addition/test.jsonl ] || "$PY" -m tasks.build

echo "==> Done. Start each session with: source .venv/bin/activate"
