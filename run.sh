#!/usr/bin/env bash
#
# Play one battle, then stop.
#
# Start this once you are already in a battle. It never touches the menus — it
# does not press Battle and does not tap through result screens — it plays the
# battle in front of it, notices when that battle ends, and exits.
#
#   ./run.sh              play one battle, record it
#   ./run.sh --no-record  play one battle without recording
#
# To stop early, from another terminal:  venv/bin/python -m llm_royale.stopper

set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-venv}"
PYTHON="$VENV/bin/python"
VIDEO_DIR="${VIDEO_DIR:-recordings}"
LOG_DIR="${LOG_DIR:-logs}"
RECORD_FPS="${RECORD_FPS:-60}"

record=1
for arg in "$@"; do
    case "$arg" in
        --no-record) record=0 ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    esac
done

if [ ! -x "$PYTHON" ]; then
    echo "No Python environment at $VENV. Run ./install.sh first." >&2
    exit 1
fi

# Fail here rather than deep inside the capture thread, where the error is far
# less obvious.
if ! "$PYTHON" - <<'PY' 2>/dev/null
from Quartz import (CGWindowListCopyWindowInfo, kCGNullWindowID,
                    kCGWindowListOptionOnScreenOnly)
infos = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID) or []
ok = any(
    i.get("kCGWindowOwnerName") == "iPhone Mirroring"
    and (i.get("kCGWindowBounds") or {}).get("Width", 0) > 100
    for i in infos
)
raise SystemExit(0 if ok else 1)
PY
then
    echo "No iPhone Mirroring window found." >&2
    echo "Open iPhone Mirroring, connect the phone, and open Clash Royale." >&2
    echo "The terminal also needs Screen Recording permission:" >&2
    echo "  System Settings > Privacy & Security > Screen Recording" >&2
    exit 1
fi

mkdir -p "$VIDEO_DIR" "$LOG_DIR"
stamp=$(date '+%Y%m%d_%H%M%S')
logfile="$LOG_DIR/battle_$stamp.log"
rm -f STOP

args=(--no-wait --matches 1 --stop-file STOP)
if [ "$record" -eq 1 ]; then
    video="$VIDEO_DIR/battle_$stamp.mp4"
    args+=(--record "$video" --record-fps "$RECORD_FPS")
fi

echo "Waiting for a battle. Nothing is tapped until one is detected."
echo "Log: $logfile"
[ "$record" -eq 1 ] && echo "Video: $video"
echo

# Tee so the battle is watchable live and still leaves a log to look at after.
set +e
"$PYTHON" -m llm_royale.llm_clasher "${args[@]}" 2>&1 \
    | tee "$logfile" \
    | grep --line-buffered -E "^\[(Match|AI Action|Menu|INFO|ERROR)\]"
status=${PIPESTATUS[0]}
set -e

placements=$(grep -c "Result: executed" "$logfile" 2>/dev/null; true)
echo
echo "Battle over. $placements card placements. Full log: $logfile"
if [ "$record" -eq 1 ] && [ -f "$video" ]; then
    echo "Recording: $video"
fi
rm -f STOP
exit "$status"
