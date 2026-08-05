#!/usr/bin/env bash
#
# Play Clash Royale continuously, unattended.
#
# The bot handles a whole match and the menus around it — pressing Battle,
# dismissing the result banner, tapping through chest and reward popups — so
# this script's job is only to keep that going overnight: one process per match,
# a video per match, and enough patience to sit out an iPhone Mirroring
# disconnect instead of spinning on it.
#
#   ./run_loop.sh              play until stopped
#   ./run_loop.sh 5            play 5 matches, then exit
#
# To stop it, from anywhere:
#
#   venv/bin/python -m llm_royale.stopper
#
# or `touch STOP` in this directory. The current match finishes its shutdown
# cleanly, its video is finalized, and the loop exits.

set -uo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-venv/bin/python}"
VIDEO_DIR="${VIDEO_DIR:-recordings}"
LOG_DIR="${LOG_DIR:-logs}"
STOP_FILE="${STOP_FILE:-STOP}"
# Videos are ~10MB a match at native 60fps, so an unattended night would other-
# wise fill the disk. Keep the most recent ones and drop the rest.
KEEP_VIDEOS="${KEEP_VIDEOS:-25}"
RECORD_FPS="${RECORD_FPS:-60}"
MAX_MATCHES="${1:-0}"

mkdir -p "$VIDEO_DIR" "$LOG_DIR"
rm -f "$STOP_FILE"

log() { printf '[run_loop %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

mirror_ready() {
    "$PYTHON" - <<'PY' 2>/dev/null
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
}

prune_videos() {
    local count
    count=$(ls -1t "$VIDEO_DIR"/match_*.mp4 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -gt "$KEEP_VIDEOS" ]; then
        ls -1t "$VIDEO_DIR"/match_*.mp4 | tail -n +$((KEEP_VIDEOS + 1)) | while read -r old; do
            rm -f "$old"
            log "pruned $(basename "$old")"
        done
    fi
}

trap 'log "interrupted"; touch "$STOP_FILE"; exit 130' INT TERM

played=0
log "starting (videos in $VIDEO_DIR, logs in $LOG_DIR)"
[ "$MAX_MATCHES" -gt 0 ] && log "will stop after $MAX_MATCHES match(es)"

while true; do
    if [ -f "$STOP_FILE" ]; then
        log "stop file present, exiting"
        break
    fi

    if ! mirror_ready; then
        # The window goes away when the phone is picked up or the session drops.
        # Nothing to do but wait for it to come back.
        log "iPhone Mirroring window not found, waiting 30s"
        sleep 30
        continue
    fi

    stamp=$(date '+%Y%m%d_%H%M%S')
    video="$VIDEO_DIR/match_$stamp.mp4"
    logfile="$LOG_DIR/match_$stamp.log"

    log "match $((played + 1)) starting -> $(basename "$video")"
    "$PYTHON" -m llm_royale.llm_clasher \
        --no-wait \
        --auto-battle \
        --matches 1 \
        --record "$video" \
        --record-fps "$RECORD_FPS" \
        --stop-file "$STOP_FILE" \
        >"$logfile" 2>&1
    status=$?

    played=$((played + 1))
    # grep -c prints 0 and exits non-zero when nothing matches, so a `|| echo 0`
    # here would append a second count.
    placements=$(grep -c "Result: executed" "$logfile" 2>/dev/null; true)
    log "match $played done (exit $status, $placements placements)"

    prune_videos

    if [ "$MAX_MATCHES" -gt 0 ] && [ "$played" -ge "$MAX_MATCHES" ]; then
        log "played $played match(es), exiting"
        break
    fi

    # A non-zero exit usually means the mirror window went away mid-match; give
    # it longer than the usual gap before trying again.
    if [ "$status" -ne 0 ]; then
        log "non-zero exit, backing off 20s"
        sleep 20
    else
        sleep 3
    fi
done

# Clear it only on the way out, so a stop requested mid-match is still visible
# to this loop when the match's own process has finished shutting down.
rm -f "$STOP_FILE"
log "loop finished after $played match(es)"
