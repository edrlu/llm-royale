#!/usr/bin/env python3
"""
Live debugger for current Clash Royale hand and elixir.

Uses the same capture, hand-classifier, and elixir inference path as the bot.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import numpy as np

import cv2

from .capture_config import PRESETS
from .clash_capture import (
    ClashDetector,
    find_bin,
    get_device_resolution,
    infer_frame_state,
    start_capture,
)
from .cycle_tracker import CardSlotClassifier


def format_hand(slots: list) -> str:
    parts = []
    for slot in slots:
        label = slot.get("label") or "-"
        conf = slot.get("confidence")
        if conf is None:
            parts.append(f"{slot['slot']}:{label}")
        else:
            parts.append(f"{slot['slot']}:{label}@{conf:.2f}")
    return " | ".join(parts)


def format_tower_health(towers: dict) -> str:
    ordered = [
        "enemy_left_tower",
        "enemy_king_tower",
        "enemy_right_tower",
        "friendly_left_tower",
        "friendly_king_tower",
        "friendly_right_tower",
    ]
    return " | ".join(f"{name}={towers.get(name, {}).get('value')}" for name in ordered)


def run(args) -> None:
    preset = PRESETS[args.preset]
    adb_bin = find_bin("adb")
    ffmpeg_bin = find_bin("ffmpeg")

    dev_w, dev_h = get_device_resolution(adb_bin)
    if dev_w == 0:
        raise SystemExit("[ERROR] Device not found or not authorized. Check `adb devices`.")

    max_size = args.max_size if args.max_size else preset.max_size
    max_dim = max(dev_w, dev_h)
    if max_dim > max_size:
        scale = max_size / max_dim
        frame_w = int(dev_w * scale) & ~1
        frame_h = int(dev_h * scale) & ~1
    else:
        frame_w = dev_w & ~1
        frame_h = dev_h & ~1

    detector = ClashDetector(preset)
    hand_classifier = CardSlotClassifier()

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

    frame_size = frame_w * frame_h * 3
    lock = threading.Lock()
    latest_frame = [None]
    running = threading.Event()
    running.set()

    def reader_loop():
        while running.is_set():
            adb_proc, ffmpeg_proc = start_capture(adb_cmd, ffmpeg_cmd)
            try:
                while running.is_set():
                    raw = ffmpeg_proc.stdout.read(frame_size)
                    if len(raw) != frame_size:
                        if ffmpeg_proc.poll() is not None or adb_proc.poll() is not None:
                            break
                        continue
                    img = np.frombuffer(raw, dtype=np.uint8).reshape((frame_h, frame_w, 3))
                    with lock:
                        latest_frame[0] = img
            finally:
                ffmpeg_proc.kill()
                adb_proc.kill()
                ffmpeg_proc.wait()
                adb_proc.wait()
            if running.is_set():
                time.sleep(0.2)

    threading.Thread(target=reader_loop, daemon=True).start()

    print(f"[INFO] Device: {dev_w}x{dev_h} -> capture {frame_w}x{frame_h}")
    print(f"[INFO] Preset: {preset.name}")
    print(f"[INFO] Card classifier enabled: {hand_classifier.enabled}")
    if hand_classifier.error:
        print(f"[INFO] Card classifier error: {hand_classifier.error}")

    tick = 0
    executor = ThreadPoolExecutor(max_workers=4)
    try:
        while True:
            with lock:
                frame = None if latest_frame[0] is None else latest_frame[0].copy()
            if frame is None:
                time.sleep(0.05)
                continue

            frame_state = infer_frame_state(frame, frame_w, frame_h, detector, hand_classifier, executor)
            hand_cards = frame_state["hand_cards"]
            elixir = frame_state["elixir_estimate"]
            tower_health = frame_state["tower_health"].get("towers", {})

            tick += 1
            line = {
                "tick": tick,
                "hand": hand_cards.get("slots", []),
                "elixir": elixir.get("count_estimate"),
                "elixir_model": elixir.get("model_count_estimate"),
                "elixir_text": elixir.get("ocr_text"),
                "tower_health": tower_health,
                "hand_candidates": hand_cards.get("candidates", []),
            }
            print(json.dumps(line, separators=(",", ":")))
            print(
                "hand: "
                f"{format_hand(hand_cards.get('slots', []))} | "
                f"elixir: {elixir.get('count_estimate')} "
                f"(model={elixir.get('model_count_estimate')}, text={elixir.get('ocr_text')}) | "
                f"tower_hp: {format_tower_health(tower_health)}"
            )
            time.sleep(args.interval_sec)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        running.clear()
        executor.shutdown(wait=False)


def main():
    parser = argparse.ArgumentParser(description="Print the current Clash Royale hand and elixir live")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="fast")
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--bit-rate", type=str, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
