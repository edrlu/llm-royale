#!/usr/bin/env bash
set -euo pipefail

# Use an explicit Codex workspace root so repo context selection works reliably.
target_dir="$PWD"
if [[ $# -ge 1 && -d "$1" && "$1" != -* ]]; then
  target_dir="$(cd "$1" && pwd)"
  shift
fi

exec codex --dangerously-bypass-approvals-and-sandbox -C "$target_dir" "$@"
