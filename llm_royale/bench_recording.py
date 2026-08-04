#!/usr/bin/env python3
"""
Measure what a recording rate actually costs.

Recording runs on its own thread, but it is not free: it and the detector both
pull frames from the same window through `CGWindowListCreateImage`, and the
window server serves those one at a time. Asking for a higher recording rate can
therefore slow the detector's grabs down, which means the planner reasons about
a staler board — the one way recording can genuinely hurt play.

This runs both loads together at each candidate rate and reports the recording
rate actually achieved alongside the grab latency the detector sees, so the
choice comes from numbers rather than from assuming a thread is free.
"""

import argparse
import threading
import time

import numpy as np

from .mirror_capture import MirrorFrameSource
from .recorder import VideoRecorder


def measure(target_fps: float, seconds: float, detector_fps: float, out_path: str,
            record_max_size: int = 832, annotate=None) -> dict:
    record_source = MirrorFrameSource(max_size=record_max_size)
    info = record_source.probe()
    detector_source = MirrorFrameSource()
    detector_source.probe()

    grab_ms = []
    stop = threading.Event()

    def detector_load():
        """Stand in for the capture subprocess pulling frames to run YOLO on."""
        period = 1.0 / detector_fps if detector_fps > 0 else 0.0
        while not stop.is_set():
            started = time.perf_counter()
            detector_source.grab_once()
            elapsed = time.perf_counter() - started
            grab_ms.append(elapsed * 1000.0)
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    recorder = VideoRecorder(
        out_path, frame_source=record_source.grab_once, fps=target_fps, annotate=annotate
    )
    thread = threading.Thread(target=detector_load, daemon=True)

    thread.start()
    recorder.start()
    started = time.perf_counter()
    time.sleep(seconds)
    elapsed = time.perf_counter() - started
    recorder.stop()
    stop.set()
    thread.join(timeout=2.0)

    grabs = np.array(grab_ms) if grab_ms else np.array([0.0])
    return {
        "record_size": f"{info['frame']['width']}x{info['frame']['height']}",
        "target_fps": target_fps,
        "achieved_fps": recorder.frames_written / elapsed,
        "detector_grab_median_ms": float(np.median(grabs)),
        "detector_grab_p90_ms": float(np.percentile(grabs, 90)),
        "detector_grabs_per_sec": len(grabs) / elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark recording rates against detector latency")
    parser.add_argument("--rates", type=float, nargs="+", default=[12, 30, 60])
    parser.add_argument("--record-max-size", type=int, nargs="+", default=[832])
    parser.add_argument("--overlay", action="store_true", help="Include overlay drawing cost")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--detector-fps", type=float, default=8.0)
    parser.add_argument("--out-prefix", default="/tmp/bench_record")
    args = parser.parse_args()

    annotate = None
    if args.overlay:
        import json
        import os
        from .overlay import draw_overlay

        snapshot = {}
        for candidate in ("llm_clasher_state.json",):
            if os.path.exists(candidate):
                try:
                    snapshot = json.load(open(candidate))
                except Exception:
                    snapshot = {}
        annotate = lambda frame: draw_overlay(frame, snapshot)

    print(f"[INFO] detector load at {args.detector_fps:g}fps, {args.seconds:g}s per rate")
    print(f"[INFO] overlay drawing: {'on' if args.overlay else 'off'}\n")
    print(f"{'size':>10} {'target':>7} {'achieved':>9} {'detector grab ms':>20} {'detector fps':>13}")
    for max_size in args.record_max_size:
        for rate in args.rates:
            result = measure(
                rate, args.seconds, args.detector_fps,
                f"{args.out_prefix}_{max_size}_{rate:g}.mp4",
                record_max_size=max_size, annotate=annotate,
            )
            print(
                f"{result['record_size']:>10} {result['target_fps']:>7.0f} {result['achieved_fps']:>9.1f} "
                f"{result['detector_grab_median_ms']:>11.0f} (p90 {result['detector_grab_p90_ms']:>3.0f}) "
                f"{result['detector_grabs_per_sec']:>13.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
