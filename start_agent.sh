#!/usr/bin/env bash
set -euo pipefail

# Run from a target directory if one is passed, otherwise use current dir
if [[ $# -ge 1 ]]; then
  cd "$1"
  shift
fi

exec codex --dangerously-bypass-approvals-and-sandbox "$@"
