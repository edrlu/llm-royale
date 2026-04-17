#!/usr/bin/env python3
"""
Real-time Android screen capture with OpenCV/YOLO processing.

Pipeline: adb screenrecord (H.264) -> ffmpeg (decode to raw BGR) -> numpy -> OpenCV

Architecture:
    - Reader thread:  continuously reads frames from ffmpeg, keeps only the latest
    - Worker thread:  picks up the latest frame, runs CV/YOLO processing
    - Main thread:    displays the latest frame with detection overlay + status bar

This ensures the display stays live even when processing is slow.

Requirements:
    - adb & ffmpeg installed (brew install android-platform-tools ffmpeg)
    - Android device connected via USB with ADB debugging enabled
    - pip install -r requirements.txt

Usage:
    python screen_capture.py                    # No processing, just mirror
    python screen_capture.py --mode opencv      # OpenCV edge detection
    python screen_capture.py --mode yolo        # YOLOv8 generic object detection
    python screen_capture.py --mode clash       # Clash Royale KataCR detection
    python screen_capture.py --max-size 720     # Lower res = higher FPS

Hotkeys:
    1 - Mirror mode (no processing)
    2 - OpenCV edge detection
    3 - YOLO generic object detection
    4 - Clash Royale KataCR detection
    q - Quit
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import threading
from collections import deque

import cv2
import numpy as np

from . import REPO_ROOT
from .capture_config import PRESETS, CapturePreset


def find_bin(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    for prefix in ["/opt/homebrew/bin", "/usr/local/bin"]:
        candidate = os.path.join(prefix, name)
        if os.path.isfile(candidate):
            return candidate
    print(f"[ERROR] {name} not found. Install with: brew install {name}")
    sys.exit(1)


class FPSCounter:
    def __init__(self, window_size=60):
        self.timestamps = deque(maxlen=window_size)

    def tick(self):
        self.timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------

class OpenCVProcessor:
    def process(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(frame, 0.7, edges_bgr, 0.3, 0)


class YOLOProcessor:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.4):
        from ultralytics import YOLO
        print(f"[INFO] Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def process(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        return results[0].plot()


CLASH_PIPELINE_DIR = os.path.join(REPO_ROOT, "clash-yolo-pipeline")
CLASH_MODELS_DIR = os.path.join(CLASH_PIPELINE_DIR, "models")
CLASH_VENDOR_DIR = os.path.join(CLASH_PIPELINE_DIR, "vendor")
CLASH_KATACR_ROOT = os.path.join(CLASH_VENDOR_DIR, "KataCR")


class ClashProcessor:
    """Clash Royale detection using KataCR dual-detector YOLO models.

    Only detects:
      - Hog 2.6 deck: hog-rider, musketeer, ice-spirit, ice-golem, fireball, the-log, cannon, skeleton
      - Towers: king-tower, queen-tower, cannoneer-tower, dagger-duchess-tower
      - Health bars: tower-bar, king-tower-bar, bar, bar-level, dagger-duchess-tower-bar
      - Game state: elixir, clock
    """

    BELONG_COLORS = {0: (255, 150, 50), 1: (50, 50, 255), -1: (0, 220, 220)}

    # Only these classes are kept — everything else is filtered out
    _ALLOWED_CLASSES = frozenset({
        # Hog 2.6 deck
        'hog-rider', 'musketeer', 'ice-spirit', 'ice-golem',
        'fireball', 'the-log', 'cannon', 'skeleton',
        # Towers
        'king-tower', 'queen-tower', 'cannoneer-tower', 'dagger-duchess-tower',
        # Health bars
        'tower-bar', 'king-tower-bar', 'dagger-duchess-tower-bar',
        'bar', 'bar-level',
        # Game state
        'elixir', 'clock',
    })

    def __init__(self, preset: CapturePreset):
        dataset_path = os.path.join(CLASH_VENDOR_DIR, "Clash-Royale-Detection-Dataset")
        os.environ["KATACR_DATASET_PATH"] = dataset_path
        ultralytics_cfg = os.path.join(CLASH_PIPELINE_DIR, ".ultralytics")
        os.makedirs(ultralytics_cfg, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", ultralytics_cfg)

        if CLASH_KATACR_ROOT not in sys.path:
            sys.path.insert(0, CLASH_KATACR_ROOT)

        from katacr.constants.label_list import idx2unit, unit2idx
        from katacr.yolov8.custom_model import CRDetectionModel
        from katacr.yolov8.custom_predict import CRDetectionPredictor

        self.idx2unit = idx2unit
        self.unit2idx = unit2idx

        # Build allowed class ID set from the allowlist
        self._allowed_ids = frozenset(
            idx for idx, name in idx2unit.items()
            if name in self._ALLOWED_CLASSES
        )
        print(f"[INFO] Clash filter: {len(self._allowed_ids)} classes allowed out of {len(idx2unit)}")

        import torch
        import torchvision
        self.torch = torch
        self.torchvision = torchvision

        from ultralytics.engine.model import Model

        class _YOLOCRLive(Model):
            @property
            def task_map(self):
                return {"detect": {
                    "model": CRDetectionModel,
                    "trainer": None,
                    "validator": None,
                    "predictor": CRDetectionPredictor,
                }}

        # Load models — use num_models from preset (1 = single, 2 = combo)
        weights = sorted(
            f for f in os.listdir(CLASH_MODELS_DIR) if f.endswith(".pt")
        )
        if not weights:
            raise FileNotFoundError(f"No .pt files found in {CLASH_MODELS_DIR}")
        weights = weights[:preset.clash_num_models]

        print(f"[INFO] Loading Clash Royale models: {weights}")
        self.models = [_YOLOCRLive(os.path.join(CLASH_MODELS_DIR, w)) for w in weights]

        self.conf_threshold = preset.clash_conf
        self.iou_threshold = preset.clash_iou
        self.device = preset.clash_device
        self.infer_size = preset.clash_infer_size

        # Apply optimizations
        for model in self.models:
            if preset.clash_fuse:
                model.fuse()
                print("[INFO] Model layers fused (Conv+BN merge)")
            if preset.clash_half and preset.clash_device != "cpu":
                model.model.half()
                print(f"[INFO] Model set to FP16 half precision on {preset.clash_device}")

        # Build per-model class remap tables
        self._class_remaps = []
        for model in self.models:
            names = getattr(model, "names", None) or {}
            if not names:
                self._class_remaps.append(np.arange(256, dtype=np.int64))
                continue
            max_id = max(names.keys())
            remap = np.arange(max_id + 1, dtype=np.int64)
            for local_id, name in names.items():
                remap[local_id] = int(unit2idx.get(str(name), local_id))
            self._class_remaps.append(remap)

        print(f"[INFO] Clash detector ready ({len(self.models)} model(s), "
              f"device={self.device}, infer={self.infer_size}px, "
              f"half={preset.clash_half}, fuse={preset.clash_fuse})")

    def process(self, frame: np.ndarray) -> np.ndarray:
        torch = self.torch

        merged_tensors = []
        for model, remap in zip(self.models, self._class_remaps):
            result = model.predict(
                frame, device=self.device,
                conf=self.conf_threshold, iou=self.iou_threshold,
                imgsz=self.infer_size, verbose=False,
            )[0]
            boxes = result.orig_boxes
            if boxes is None or len(boxes) == 0:
                continue
            remapped = boxes.clone()
            local_ids = remapped[:, 5].long().detach().cpu().numpy()
            clamped = np.clip(local_ids, 0, len(remap) - 1)
            remapped[:, 5] = torch.from_numpy(remap[clamped]).to(
                device=remapped.device,
                dtype=remapped.dtype,
            )
            merged_tensors.append(remapped)

        if not merged_tensors:
            return frame

        if len(merged_tensors) == 1:
            raw = merged_tensors[0].detach().cpu().numpy()
        else:
            merged = torch.cat(merged_tensors, dim=0)
            merged_cpu = merged.detach().cpu()
            keep = self.torchvision.ops.nms(
                merged_cpu[:, :4], merged_cpu[:, 4], iou_threshold=self.iou_threshold
            )
            raw = merged_cpu[keep].numpy()

        output = frame.copy()
        for row in raw:
            if len(row) < 7:
                continue
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            conf = float(row[-3])
            class_id = int(row[-2])
            belong = int(row[-1])

            # Filter out removed classes
            if class_id not in self._allowed_ids:
                continue

            name = str(self.idx2unit.get(class_id, class_id))

            color = self.BELONG_COLORS.get(belong, self.BELONG_COLORS[-1])
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(output, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return output


# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------

BAR_HEIGHT = 56


def draw_status_bar(
    frame: np.ndarray, fps: float, mode: str,
    system_ms: float, process_ms: float, net_ms: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    output = np.zeros((h + BAR_HEIGHT, w, 3), dtype=np.uint8)
    output[BAR_HEIGHT:] = frame

    cv2.putText(output, f"FPS: {fps:.1f}", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    mode_text = f"Mode: {mode.upper()}"
    tw = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
    cv2.putText(output, mode_text, ((w - tw) // 2, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    net_text = f"Net: {net_ms:.1f}ms"
    tw = cv2.getTextSize(net_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
    cv2.putText(output, net_text, (w - tw - 12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(output, f"System: {system_ms:.1f}ms", (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)

    proc_text = f"Process: {process_ms:.1f}ms"
    tw = cv2.getTextSize(proc_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
    cv2.putText(output, proc_text, (w - tw - 12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 2, cv2.LINE_AA)

    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def drain_pipe(pipe, prefix):
    for line in pipe:
        text = line.decode(errors="replace").strip()
        if text:
            print(f"[{prefix}] {text}", file=sys.stderr)


def get_device_resolution(adb_bin: str) -> tuple:
    try:
        out = subprocess.check_output(
            [adb_bin, "shell", "wm", "size"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip()
        last_line = out.strip().split("\n")[-1]
        parts = last_line.split(":")[-1].strip().split("x")
        return int(parts[0]), int(parts[1])
    except Exception as e:
        print(f"[ERROR] Could not get device resolution: {e}")
        return 0, 0


def start_capture(adb_cmd, ffmpeg_cmd):
    """Launch the adb + ffmpeg pipeline, return (adb_proc, ffmpeg_proc)."""
    adb_proc = subprocess.Popen(
        adb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    threading.Thread(target=drain_pipe, args=(adb_proc.stderr, "adb"), daemon=True).start()

    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=adb_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    threading.Thread(target=drain_pipe, args=(ffmpeg_proc.stderr, "ffmpeg"), daemon=True).start()
    adb_proc.stdout.close()  # let ffmpeg own the pipe

    return adb_proc, ffmpeg_proc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    # Resolve preset
    preset = PRESETS[args.preset]
    print(f"[INFO] Preset: {preset.name} — {preset.description}")

    adb_bin = find_bin("adb")
    ffmpeg_bin = find_bin("ffmpeg")

    dev_w, dev_h = get_device_resolution(adb_bin)
    if dev_w == 0:
        print("[ERROR] Device not found or not authorized.")
        print("[INFO] Check: adb devices")
        sys.exit(1)
    print(f"[INFO] Device resolution: {dev_w}x{dev_h}")

    max_size = args.max_size if args.max_size else preset.max_size
    max_dim = max(dev_w, dev_h)
    if max_dim > max_size:
        scale = max_size / max_dim
        frame_w = int(dev_w * scale) & ~1
        frame_h = int(dev_h * scale) & ~1
    else:
        frame_w = dev_w & ~1
        frame_h = dev_h & ~1

    print(f"[INFO] Output resolution: {frame_w}x{frame_h}")

    # --- shared state (protected by lock) ---
    lock = threading.Lock()
    latest_frame = [None]           # most recent raw frame from reader
    latest_overlay = [None]         # most recent processed frame from worker
    process_ms_shared = [0.0]       # last processing time
    system_ms_shared = [0.0]        # last system (read) time
    processor_box = [None]          # current processor (mutable from main thread)
    mode_label_box = [args.mode if args.mode else "mirror"]
    running = threading.Event()
    running.set()

    # Pick initial processor
    if args.mode == "opencv":
        processor_box[0] = OpenCVProcessor()
    elif args.mode == "yolo":
        processor_box[0] = YOLOProcessor(preset.yolo_model, preset.yolo_conf)
    elif args.mode == "clash":
        processor_box[0] = ClashProcessor(preset)

    # Parse bitrate
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

    print(f"[INFO] adb: {' '.join(adb_cmd)}")
    print(f"[INFO] ffmpeg: {' '.join(ffmpeg_cmd)}")

    frame_size = frame_w * frame_h * 3

    # --- Reader thread: continuously grabs frames, keeps only the latest ---
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
                            break  # restart
                        continue

                    img = np.frombuffer(raw, dtype=np.uint8).reshape((frame_h, frame_w, 3))
                    with lock:
                        latest_frame[0] = img
                        system_ms_shared[0] = dt
            finally:
                ffmpeg_proc.kill()
                adb_proc.kill()
                ffmpeg_proc.wait()
                adb_proc.wait()

            if running.is_set():
                print("[INFO] Stream ended, restarting capture...")
                time.sleep(0.2)

    # --- Worker thread: processes the latest frame in the background ---
    def worker_loop():
        while running.is_set():
            with lock:
                frame = latest_frame[0]
                proc = processor_box[0]

            if frame is None or proc is None:
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()
            processed = proc.process(frame.copy())
            dt = (time.perf_counter() - t0) * 1000.0

            with lock:
                latest_overlay[0] = processed
                process_ms_shared[0] = dt

    reader_thread = threading.Thread(target=reader_loop, daemon=True)
    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    reader_thread.start()
    worker_thread.start()

    fps_counter = FPSCounter()
    print(f"[INFO] Frame size: {frame_size} bytes")
    print("[INFO] Streaming... Press 'q' to quit, 1/2/3/4 to switch modes.")

    try:
        while True:
            with lock:
                raw_frame = latest_frame[0]
                overlay = latest_overlay[0]
                proc = processor_box[0]
                sys_ms = system_ms_shared[0]
                proc_ms = process_ms_shared[0]
                mode = mode_label_box[0]

            # Pick what to display: overlay if available, else raw frame
            if proc is not None and overlay is not None:
                display = overlay
            elif raw_frame is not None:
                display = raw_frame
            else:
                # No frames yet
                cv2.waitKey(30)
                continue

            fps_counter.tick()
            net_ms = sys_ms + proc_ms
            output = draw_status_bar(display, fps_counter.fps, mode, sys_ms, proc_ms, net_ms)
            cv2.imshow("Android Screen", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                with lock:
                    processor_box[0] = None
                    latest_overlay[0] = None
                    mode_label_box[0] = "mirror"
                print("[INFO] Switched to: mirror")
            elif key == ord("2"):
                with lock:
                    processor_box[0] = OpenCVProcessor()
                    latest_overlay[0] = None
                    mode_label_box[0] = "opencv"
                print("[INFO] Switched to: opencv")
            elif key == ord("3"):
                with lock:
                    if not isinstance(processor_box[0], YOLOProcessor):
                        processor_box[0] = YOLOProcessor(args.yolo_model, args.yolo_conf)
                    latest_overlay[0] = None
                    mode_label_box[0] = "yolo"
                print("[INFO] Switched to: yolo")
            elif key == ord("4"):
                with lock:
                    if not isinstance(processor_box[0], ClashProcessor):
                        processor_box[0] = ClashProcessor(preset)
                    latest_overlay[0] = None
                    mode_label_box[0] = "clash"
                print("[INFO] Switched to: clash")

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        running.clear()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Android screen capture with CV processing"
    )
    parser.add_argument(
        "--preset", choices=sorted(PRESETS.keys()), default="quality",
        help="Capture preset (default: quality)",
    )
    parser.add_argument(
        "--mode", choices=["opencv", "yolo", "clash"], default=None,
        help="Processing mode (default: mirror only)",
    )
    parser.add_argument(
        "--max-size", type=int, default=None,
        help="Override max dimension in pixels (default: preset value)",
    )
    parser.add_argument(
        "--bit-rate", type=str, default=None,
        help="Override video bitrate (default: preset value)",
    )
    parser.add_argument(
        "--yolo-model", type=str, default="yolov8n.pt",
        help="YOLO model to use (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--yolo-conf", type=float, default=0.4,
        help="YOLO confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--clash-conf", type=float, default=0.25,
        help="Clash Royale detector confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--clash-iou", type=float, default=0.45,
        help="Clash Royale detector IoU/NMS threshold (default: 0.45)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
