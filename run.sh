#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo ".venv is missing. Run ./install_dependencies.sh first." >&2
  exit 1
fi

exec "$VENV_PYTHON" "$ROOT_DIR/live_feed.py" "$@"
