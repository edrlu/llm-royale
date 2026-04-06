#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
VENDOR_DIR="$ROOT_DIR/vendor"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -r "$ROOT_DIR/requirements-live-viewer.txt"

mkdir -p "$VENDOR_DIR"
if [[ ! -d "$VENDOR_DIR/Clash-Royale-Detection-Dataset/.git" && ! -d "$VENDOR_DIR/Clash-Royale-Detection-Dataset/images" ]]; then
  git clone https://github.com/wty-yy/Clash-Royale-Detection-Dataset "$VENDOR_DIR/Clash-Royale-Detection-Dataset"
fi

cat <<'EOF'
Python dependencies installed into .venv.

System requirements:
- Install `scrcpy`
- Install `adb` / Android platform-tools
- On Linux, scrcpy window capture works best under X11. If you use Wayland, prefer --capture-region or test an alternate source.
- CPU PyTorch wheels are installed by default. If your target machine has a CUDA-capable GPU, replace them with the appropriate PyTorch CUDA wheels inside `.venv`.

Example:
  ./run.sh --source scrcpy --weights /absolute/path/to/weights.pt --device cuda
EOF
