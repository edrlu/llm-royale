#!/usr/bin/env python3
"""
Live LLM Clash Royale debugger.

- Runs the same capture/inference/planning/execution path as llm_clasher.py
- Saves an annotated video to debug/vid/NN/debugger.mp4
- Saves a markdown timeline to debug/vid/NN/debugger.md
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
import requests

from . import REPO_ROOT
from .capture_config import PRESETS
from .clash_capture import (
    ClashDetector,
    find_bin,
    get_device_resolution,
    infer_frame_state,
    start_capture,
)
from .cycle_tracker import CardSlotClassifier, CycleTracker
from .llm_clasher import (
    AndroidActionExecutor,
    ElixirClock,
    OpenAIPlanner,
    load_dotenv,
    state_signature,
    verify_action,
    wait_for_space,
)


def format_hand(slots: list) -> str:
    parts = []
    for slot in slots:
        label = slot.get("label") or "-"
        conf = slot.get("confidence")
        if conf is None:
            parts.append(f"{slot.get('slot')}:{label}")
        else:
            parts.append(f"{slot.get('slot')}:{label}@{conf:.2f}")
    return " | ".join(parts)


def next_debug_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    nums = [
        int(name)
        for name in os.listdir(base_dir)
        if name.isdigit() and os.path.isdir(os.path.join(base_dir, name))
    ]
    next_num = max(nums) + 1 if nums else 1
    out_dir = os.path.join(base_dir, f"{next_num:02d}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def encode_video_from_frames(ffmpeg_bin: str, frames_dir: str, fps: float, video_path: str) -> None:
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(frames_dir, "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        video_path,
    ]
    subprocess.run(cmd, check=True)


def summarize_units(units: list, limit: int = 12) -> list[str]:
    lines = []
    for unit in units[:limit]:
        center = unit.get("center_normalized", {})
        lines.append(
            f"- {unit.get('owner')} {unit.get('label')} "
            f"lane={unit.get('lane')} side={unit.get('side')} "
            f"x={center.get('x')} y={center.get('y')} "
            f"conf={unit.get('confidence')}"
        )
    return lines or ["- none"]


def append_markdown(log_path: str, snapshot: dict, hand_cards: dict,
                    decision: Optional[dict], result: Optional[dict], verification: Optional[dict]) -> None:
    grouped = snapshot.get("grouped", {})
    now_iso = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"## Snapshot {snapshot.get('sequence')} - {now_iso}\n")
        f.write(f"- Elixir: {snapshot.get('elixir_inferred', {}).get('blended_estimate')}\n")
        f.write(f"- Hand: `{format_hand(hand_cards.get('slots', []))}`\n")
        if decision is not None:
            f.write(f"- Decision: `{json.dumps(decision, separators=(',', ':'))}`\n")
        if result is not None:
            f.write(f"- Result: `{json.dumps(result, separators=(',', ':'))}`\n")
        if verification is not None:
            f.write(f"- Verification: `{json.dumps(verification, separators=(',', ':'))}`\n")
        f.write("\n### Troops\n")
        for line in summarize_units(grouped.get("troops", []), limit=16):
            f.write(f"{line}\n")
        f.write("\n### Buildings\n")
        for line in summarize_units(grouped.get("buildings", []), limit=8):
            f.write(f"{line}\n")
        f.write("\n### Towers\n")
        for line in summarize_units(grouped.get("towers", []), limit=8):
            f.write(f"{line}\n")
        f.write("\n")


def render_overlay(frame: np.ndarray, snapshot: dict, hand_cards: dict,
                   decision: Optional[dict], result: Optional[dict]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    grouped = snapshot.get("grouped", {})
    board_ref = snapshot.get("llm_summary", {}).get("board_reference", {}) or {}

    for det in snapshot.get("detections", []):
        bbox = det.get("bbox", {})
        x1, y1, x2, y2 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0)), int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
        owner = det.get("owner")
        color = (0, 220, 0) if owner == "friendly" else (40, 80, 255) if owner == "enemy" else (180, 180, 180)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.get('label')} {det.get('confidence', 0):.2f}"
        cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    river = board_ref.get("river_y_norm")
    if river is not None:
        ry = int(round(float(river) * h))
        cv2.line(out, (0, ry), (w - 1, ry), (255, 255, 0), 2)

    panel_h = 146
    cv2.rectangle(out, (0, 0), (w - 1, panel_h), (0, 0, 0), -1)
    lines = [
        f"seq={snapshot.get('sequence')} elixir={snapshot.get('elixir_inferred', {}).get('blended_estimate')} "
        f"troops={len(grouped.get('troops', []))} buildings={len(grouped.get('buildings', []))}",
        f"hand: {format_hand(hand_cards.get('slots', []))}",
    ]
    if decision is not None:
        lines.append(f"decision: {json.dumps(decision, separators=(',', ':'))[:120]}")
    if result is not None:
        lines.append(f"result:   {json.dumps(result, separators=(',', ':'))[:120]}")
    y = 22
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return out


def build_snapshot(sequence: int, frame: np.ndarray, system_ms: float, process_ms: float, detections: list,
                   grouped: dict, llm_summary: dict, hand_cards: dict, elixir_estimate: dict, inferred: dict) -> dict:
    h, w = frame.shape[:2]
    return {
        "sequence": sequence,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "capture": {
            "width": w,
            "height": h,
            "system_latency_ms": round(system_ms, 2),
            "process_latency_ms": round(process_ms, 2),
            "total_latency_ms": round(system_ms + process_ms, 2),
        },
        "detections": detections,
        "grouped": grouped,
        "llm_summary": llm_summary,
        "hand_cards_inferred": hand_cards,
        "elixir_inferred": elixir_estimate,
        "inferred_game_state": inferred,
    }


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run llm_clasher with annotated video + markdown logging")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="fast")
    parser.add_argument("--interval-sec", type=float, default=0.4)
    parser.add_argument("--cooldown-sec", type=float, default=0.5)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--bit-rate", type=str, default=None)
    parser.add_argument("--video-fps", type=float, default=4.0)
    args = parser.parse_args()

    wait_for_space()

    preset = PRESETS[args.preset]
    adb_bin = find_bin("adb")
    ffmpeg_bin = find_bin("ffmpeg")
    dev_w, dev_h = get_device_resolution(adb_bin)
    if dev_w == 0:
        print("[ERROR] Device not found or not authorized.")
        return 1

    max_size = args.max_size if args.max_size else preset.max_size
    max_dim = max(dev_w, dev_h)
    if max_dim > max_size:
        scale = max_size / max_dim
        frame_w = int(dev_w * scale) & ~1
        frame_h = int(dev_h * scale) & ~1
    else:
        frame_w = dev_w & ~1
        frame_h = dev_h & ~1

    bit_rate = args.bit_rate if args.bit_rate else preset.bit_rate
    if bit_rate.upper().endswith("M"):
        bit_rate_int = int(bit_rate[:-1]) * 1_000_000
    elif bit_rate.upper().endswith("K"):
        bit_rate_int = int(bit_rate[:-1]) * 1_000
    else:
        bit_rate_int = int(bit_rate)

    adb_cmd = [
        adb_bin, "exec-out", "screenrecord",
        "--output-format=h264", f"--bit-rate={bit_rate_int}", "-",
    ]
    ffmpeg_cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        "-f", "h264", "-i", "pipe:0",
        "-vf", f"scale={frame_w}:{frame_h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1",
    ]

    out_dir = next_debug_dir(os.path.join(REPO_ROOT, "debug", "vid"))
    video_path = os.path.join(out_dir, "debugger.mp4")
    log_path = os.path.join(out_dir, "debugger.md")
    frames_dir = os.path.join(out_dir, "_frames")
    os.makedirs(frames_dir, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# llm_clasher_debugger\n\n- started_at: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"- model: `{args.model}`\n- preset: `{args.preset}`\n- video: `{os.path.basename(video_path)}`\n\n")
    frame_index = 0

    detector = ClashDetector(preset)
    hand_classifier = CardSlotClassifier()
    planner = OpenAIPlanner(args.model)
    executor = AndroidActionExecutor()
    cycle_tracker = CycleTracker()
    elixir_clock = ElixirClock()
    elixir_clock.start()

    frame_size = frame_w * frame_h * 3
    lock = threading.Lock()
    latest_frame = [None]
    latest_system_ms = [0.0]
    running = threading.Event()
    running.set()

    def reader_loop():
        while running.is_set():
            adb_proc, ffmpeg_proc = start_capture(adb_cmd, ffmpeg_cmd)
            try:
                while running.is_set():
                    t0 = time.perf_counter()
                    raw = ffmpeg_proc.stdout.read(frame_size)
                    dt = (time.perf_counter() - t0) * 1000.0
                    if len(raw) != frame_size:
                        if ffmpeg_proc.poll() is not None or adb_proc.poll() is not None:
                            break
                        continue
                    img = np.frombuffer(raw, dtype=np.uint8).reshape((frame_h, frame_w, 3))
                    with lock:
                        latest_frame[0] = img
                        latest_system_ms[0] = dt
            finally:
                ffmpeg_proc.kill()
                adb_proc.kill()
                ffmpeg_proc.wait()
                adb_proc.wait()
            if running.is_set():
                time.sleep(0.2)

    thread = threading.Thread(target=reader_loop, daemon=True)
    thread.start()

    sequence = 0
    last_signature = None
    last_action_ts = 0.0
    backoff_until = 0.0
    pending_action = None
    last_loop_ts = 0.0
    executor_pool = ThreadPoolExecutor(max_workers=4)

    print(f"[INFO] Saving debugger output to {out_dir}")
    try:
        while True:
            with lock:
                frame = None if latest_frame[0] is None else latest_frame[0].copy()
                system_ms = latest_system_ms[0]
            if frame is None:
                time.sleep(0.05)
                continue
            now = time.time()
            if now - last_loop_ts < args.interval_sec:
                time.sleep(0.05)
                continue
            last_loop_ts = now

            t0 = time.perf_counter()
            frame_state = infer_frame_state(frame, frame_w, frame_h, detector, hand_classifier, executor_pool)
            detections = frame_state["detections"]
            grouped = frame_state["grouped"]
            llm_summary = frame_state["llm_summary"]
            inferred = frame_state["inferred"]
            fused_hand = frame_state["hand_cards"]
            elixir_estimate = frame_state["elixir_estimate"]
            process_ms = (time.perf_counter() - t0) * 1000.0

            sequence += 1
            snapshot = build_snapshot(
                sequence, frame, system_ms, process_ms, detections,
                grouped, llm_summary, fused_hand, elixir_estimate, inferred,
            )
            cycle_tracker.observe_hand(fused_hand.get("slots", []))
            snapshot["cycle_state"] = cycle_tracker.state()
            observed_elixir = snapshot.get("elixir_inferred", {}).get("count_estimate")
            blended_elixir = elixir_clock.estimate(observed_elixir)
            snapshot.setdefault("elixir_inferred", {})["blended_estimate"] = blended_elixir

            verification = None
            if pending_action is not None:
                verification = verify_action(snapshot, pending_action)
                if verification.get("status") in {"verified", "unverified"}:
                    pending_action = None

            decision = None
            result = None
            signature = state_signature(snapshot)
            if now >= backoff_until and now - last_action_ts >= args.cooldown_sec and signature != last_signature:
                try:
                    decision = planner.plan(snapshot)
                    result = executor.execute_decision(decision, snapshot)
                    if result.get("status") == "executed" and result.get("card"):
                        cycle_tracker.record_play(result["card"])
                        elixir_clock.spend(result["card"])
                        pending_action = {
                            "card": result["card"],
                            "slot": result["slot"],
                            "capture_target": result["capture_target"],
                            "snapshot_sequence": sequence,
                        }
                    last_signature = signature
                    last_action_ts = now
                except requests.HTTPError as e:
                    status = e.response.status_code if e.response is not None else None
                    backoff_until = time.time() + (15.0 if status == 429 else 2.0)
                    result = {"status": "planner_error", "reason": f"http_error_{status}"}
                except Exception as e:
                    backoff_until = time.time() + 1.0
                    result = {"status": "planner_error", "reason": str(e)}

            overlay = render_overlay(frame, snapshot, fused_hand, decision, result)
            frame_index += 1
            frame_path = os.path.join(frames_dir, f"frame_{frame_index:06d}.png")
            cv2.imwrite(frame_path, overlay)
            append_markdown(log_path, snapshot, fused_hand, decision, result, verification)
            print(
                f"[debugger] seq={sequence} hand={format_hand(fused_hand.get('slots', []))} "
                f"decision={decision.get('action') if decision else '-'}"
            )
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down debugger...")
    finally:
        running.clear()
        thread.join(timeout=1)
        executor_pool.shutdown(wait=False)
        if frame_index > 0:
            try:
                encode_video_from_frames(ffmpeg_bin, frames_dir, args.video_fps, video_path)
                shutil.rmtree(frames_dir, ignore_errors=True)
            except Exception as e:
                print(f"[WARN] Failed to encode debugger video: {e}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n- ended_at: {datetime.now(timezone.utc).isoformat()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
