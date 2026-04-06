#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
DEFAULT_WEIGHTS_ARG=()

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo ".venv is missing. Run ./install_dependencies.sh first." >&2
  exit 1
fi

for arg in "$@"; do
  if [[ "$arg" == "--weights" ]]; then
    exec "$VENV_PYTHON" "$ROOT_DIR/live_feed.py" "$@"
  fi
done

if compgen -G "$ROOT_DIR/runs/*.pt" > /dev/null; then
  DEFAULT_WEIGHTS_ARG=(--weights "$ROOT_DIR/runs")
elif compgen -G "$ROOT_DIR/weights/*.pt" > /dev/null; then
  DEFAULT_WEIGHTS_ARG=(--weights "$ROOT_DIR/weights")
fi

exec "$VENV_PYTHON" "$ROOT_DIR/live_feed.py" "${DEFAULT_WEIGHTS_ARG[@]}" "$@"
