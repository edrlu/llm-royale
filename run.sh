#!/usr/bin/env bash
#
# Play one battle, then stop.
#
# Start this once you are already in a battle. It never touches the menus — it
# does not press Battle and does not tap through result screens — it plays the
# battle in front of it, notices when that battle ends, and exits.
#
#   ./run.sh                        play one battle, record it
#   ./run.sh --no-record            play one battle without recording
#   ./run.sh --provider anthropic   let Claude decide the moves instead of GPT
#
# To stop early, from another terminal:  venv/bin/python -m llm_royale.stopper

set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-venv}"
PYTHON="$VENV/bin/python"
VIDEO_DIR="${VIDEO_DIR:-recordings}"
LOG_DIR="${LOG_DIR:-logs}"
FIGURE_DIR="${FIGURE_DIR:-figures}"
RECORD_FPS="${RECORD_FPS:-60}"

record=1
# Anything not consumed here is forwarded to the bot, so provider and model
# selection (--provider anthropic, --model ...) work without duplicating flags.
passthrough=()
for arg in "$@"; do
    case "$arg" in
        --no-record) record=0 ;;
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) passthrough+=("$arg") ;;
    esac
done

if [ ! -x "$PYTHON" ]; then
    echo "No Python environment at $VENV. Run ./install.sh first." >&2
    exit 1
fi

# A previous run that never exited keeps its capture subprocess alive, and that
# subprocess goes on writing the same state file this one reads — two bots
# fighting over one phone, with the older one still able to tap. Clear them out
# before starting rather than leaving it to be noticed an hour later.
stale=$(pgrep -f 'llm_royale\.(llm_clasher|clash_capture)' | grep -v "^$$\$" || true)
if [ -n "$stale" ]; then
    echo "Stopping previous run: $(echo "$stale" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill $stale 2>/dev/null || true
    for _ in 1 2 3 4 5 6; do
        pgrep -f 'llm_royale\.(llm_clasher|clash_capture)' >/dev/null || break
        sleep 0.5
    done
    # Anything still up after three seconds is wedged and will not come down on
    # its own; it must go, or this run inherits the same state-file fight.
    leftover=$(pgrep -f 'llm_royale\.(llm_clasher|clash_capture)' || true)
    if [ -n "$leftover" ]; then
        # shellcheck disable=SC2086
        kill -9 $leftover 2>/dev/null || true
    fi
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

mkdir -p "$VIDEO_DIR" "$LOG_DIR" "$FIGURE_DIR"
stamp=$(date '+%Y%m%d_%H%M%S')
logfile="$LOG_DIR/battle_$stamp.log"
figure="$FIGURE_DIR/battle_$stamp.png"
rm -f STOP

# The figure is named off the same stamp as the log and the video, so the three
# artifacts of one run always share a name.
args=(--no-wait --matches 1 --stop-file STOP --figure "$figure" "${passthrough[@]+"${passthrough[@]}"}")
if [ "$record" -eq 1 ]; then
    video="$VIDEO_DIR/battle_$stamp.mp4"
    args+=(--record "$video" --record-fps "$RECORD_FPS")
fi

echo "Waiting for a battle. Nothing is tapped until one is detected."
echo "Log: $logfile"
echo "Figure: $figure"
[ "$record" -eq 1 ] && echo "Video: $video"
echo

# Tee so the battle is watchable live and still leaves a log to look at after.
set +e
# -u because stdout here is a pipe, not a terminal: Python would otherwise
# buffer 8KB before writing anything, so a run that is working looks identical
# to one that is hung until it exits and the buffer flushes.
"$PYTHON" -u -m llm_royale.llm_clasher "${args[@]}" 2>&1 \
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
if [ -f "$figure" ]; then
    echo "Latency figure: $figure"
fi
rm -f STOP
exit "$status"
