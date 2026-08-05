#!/usr/bin/env bash
#
# Set up llm-royale on a fresh macOS machine.
#
# Installs the system tools and Python packages, then checks the two things a
# git clone cannot provide on its own: the YOLO weights (168MB, too large for
# git) and an API key for whichever planner you use. It is safe to re-run.

set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-venv}"
MODELS_DIR="clash-yolo-pipeline/models"

green() { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
red() { printf '\033[31m%s\033[0m\n' "$*"; }
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

problems=0

step "Checking the platform"
if [ "$(uname -s)" != "Darwin" ]; then
    red "This only runs on macOS: it drives a real iPhone through the"
    red "macOS 'iPhone Mirroring' app, and captures that window with Quartz."
    exit 1
fi
green "macOS $(sw_vers -productVersion)"

# iPhone Mirroring needs macOS 15 (Sequoia) or newer.
major=$(sw_vers -productVersion | cut -d. -f1)
if [ "$major" -lt 15 ]; then
    red "iPhone Mirroring needs macOS 15 (Sequoia) or newer; this is $major."
    exit 1
fi

step "Checking system tools"
if ! command -v brew >/dev/null 2>&1; then
    red "Homebrew not found. Install it from https://brew.sh, then re-run."
    exit 1
fi
for tool in ffmpeg tesseract; do
    if command -v "$tool" >/dev/null 2>&1; then
        green "$tool already installed"
    else
        yellow "installing $tool"
        brew install "$tool"
    fi
done

step "Setting up the Python environment"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    green "created $VENV"
else
    green "$VENV already exists"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r requirements.txt
green "python packages installed"

step "Checking the YOLO weights"
# These are the one thing a clone cannot bring with it.
missing_models=0
for weight in detector1_v0.7.13.pt detector2_v0.7.13.pt; do
    if [ ! -f "$MODELS_DIR/$weight" ]; then
        missing_models=1
    fi
done
if [ ! -d "$MODELS_DIR/classification/card" ]; then
    missing_models=1
fi

if [ "$missing_models" -eq 1 ]; then
    problems=1
    red "Model weights are missing from $MODELS_DIR"
    echo "  They are ~168MB, so they are not in git. Copy them from a machine"
    echo "  that already works:"
    echo
    echo "    scp -r you@othermac:path/to/llm-royale/$MODELS_DIR/ $MODELS_DIR/"
    echo
    echo "  Expected inside $MODELS_DIR:"
    echo "    detector1_v0.7.13.pt, detector2_v0.7.13.pt, classification/"
else
    green "model weights present"
fi

step "Checking the API keys"
# Either provider works, so only one key is actually required.
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        yellow "created .env from .env.example — put a key in it"
    else
        printf 'OPENAI_API_KEY=\nANTHROPIC_API_KEY=\n' > .env
        yellow "created .env — put a key in it"
    fi
    problems=1
else
    have_key=0
    if grep -qE '^OPENAI_API_KEY=.+' .env && ! grep -qE '^OPENAI_API_KEY=your_' .env; then
        green "OPENAI_API_KEY set (--provider openai)"
        have_key=1
    fi
    if grep -qE '^ANTHROPIC_API_KEY=.+' .env && ! grep -qE '^ANTHROPIC_API_KEY=your_' .env; then
        green "ANTHROPIC_API_KEY set (--provider anthropic)"
        have_key=1
    fi
    if [ "$have_key" -eq 0 ]; then
        red "No API key in .env — set OPENAI_API_KEY or ANTHROPIC_API_KEY"
        problems=1
    fi
fi

step "Checking the capture path"
# Screen Recording permission cannot be granted from a script, and the failure
# it causes otherwise (black or missing frames) is confusing, so probe for it.
if "$VENV/bin/python" - <<'PY' 2>/dev/null
from Quartz import (CGWindowListCopyWindowInfo, kCGNullWindowID,
                    kCGWindowListOptionOnScreenOnly)
infos = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID) or []
found = any(i.get("kCGWindowOwnerName") == "iPhone Mirroring" for i in infos)
raise SystemExit(0 if found else 1)
PY
then
    green "iPhone Mirroring window found"
else
    yellow "iPhone Mirroring is not running (fine for now)"
    echo "  Before running: open iPhone Mirroring, connect the phone, open Clash Royale."
    echo "  The terminal you run from also needs Screen Recording permission:"
    echo "  System Settings > Privacy & Security > Screen Recording."
fi

echo
if [ "$problems" -eq 0 ]; then
    green "Setup complete. Start a battle on the phone, then run:  ./run.sh"
else
    yellow "Setup finished with the items above still to do."
fi
