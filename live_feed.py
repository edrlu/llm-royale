"""
Setup:
1. Create and populate the virtual environment with `install_dependencies.sh` or `install_dependencies.bat`.
2. Put KataCR YOLOv8 weights somewhere local, for example `runs/detector1_v0.7.13.pt`.
3. Make sure `scrcpy` and `adb` are installed and your Android device is visible to `adb devices`.
4. Run:
   `python live_feed.py --source scrcpy --model-mode single --device auto --infer-size 416 --latest-frame-only`
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mss
import numpy as np

from config import AppConfig, ensure_katacr_environment, parse_args


@dataclass(slots=True)
class FramePacket:
    frame_bgr: np.ndarray
    source_name: str
    frame_index: int
    timestamp: float


@dataclass(slots=True)
class InferenceSnapshot:
    frame_index: int
    timestamp: float
    source_name: str
    projected_frame_detections: list["Detection"]
    projected_arena_detections: list["Detection"]
    regions: "RegionBundle"
    inference_ms: float


class FrameSource:
    def read(self) -> Optional[FramePacket]:
        raise NotImplementedError

    def close(self) -> None:
        return


class OpenCVFrameSource(FrameSource):
    def __init__(self, source: str) -> None:
        source_value: int | str = int(source) if source.isdigit() else source
        self.cap = cv2.VideoCapture(source_value)
        if not self.cap.isOpened():
            raise RuntimeError(f"unable to open source: {source}")
        self.frame_index = 0
        self.source_name = str(source)

    def read(self) -> Optional[FramePacket]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        packet = FramePacket(frame_bgr=frame, source_name=self.source_name, frame_index=self.frame_index, timestamp=time.time())
        self.frame_index += 1
        return packet

    def close(self) -> None:
        self.cap.release()


class ScrcpyWindowSource(FrameSource):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.mss = mss.mss()
        self.proc: Optional[subprocess.Popen[str]] = None
        self.frame_index = 0
        self.window = None
        self.capture_region = config.capture_region
        self._start_scrcpy_if_needed()
        if self.capture_region is None:
            self.window = self._wait_for_window(config.scrcpy_window_title)

    def _start_scrcpy_if_needed(self) -> None:
        if not shutil.which(self.config.scrcpy_path):
            raise RuntimeError("scrcpy executable was not found. Install scrcpy and/or pass --scrcpy-path.")
        cmd = [self.config.scrcpy_path, "--window-title", self.config.scrcpy_window_title, "--max-size", str(self.config.scrcpy_max_size)]
        if self.config.scrcpy_serial:
            cmd.extend(["--serial", self.config.scrcpy_serial])
        if self.config.scrcpy_stay_awake:
            cmd.append("--stay-awake")
        if self.config.scrcpy_fullscreen:
            cmd.append("--fullscreen")
        cmd.extend(self.config.scrcpy_extra_args)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _wait_for_window(self, title: str, timeout: float = 15.0):
        import pywinctl as pwc

        deadline = time.time() + timeout
        while time.time() < deadline:
            windows = pwc.getWindowsWithTitle(title)
            windows = [w for w in windows if w.width > 100 and w.height > 100]
            if windows:
                return windows[0]
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError("scrcpy exited before a capture window appeared.")
            time.sleep(0.25)
        raise RuntimeError(f"unable to find scrcpy window titled '{title}'")

    def _current_monitor(self) -> dict[str, int]:
        if self.capture_region is not None:
            left, top, width, height = self.capture_region
            return {"left": left, "top": top, "width": width, "height": height}
        if self.window is None:
            raise RuntimeError("scrcpy window capture is not initialized")
        box = self.window.getClientFrame()
        left = int(getattr(box, "left"))
        top = int(getattr(box, "top"))
        width = getattr(box, "width", None)
        height = getattr(box, "height", None)
        if width is None or height is None:
            right = int(getattr(box, "right"))
            bottom = int(getattr(box, "bottom"))
            width = right - left
            height = bottom - top
        return {"left": left, "top": top, "width": int(width), "height": int(height)}

    def read(self) -> Optional[FramePacket]:
        if self.proc and self.proc.poll() is not None:
            return None
        monitor = self._current_monitor()
        image = np.array(self.mss.grab(monitor), dtype=np.uint8)
        frame = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        packet = FramePacket(frame_bgr=frame, source_name="scrcpy", frame_index=self.frame_index, timestamp=time.time())
        self.frame_index += 1
        return packet

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.mss.close()


class FrameRelay:
    def __init__(self, latest_only: bool) -> None:
        self.latest_only = latest_only
        self._cond = threading.Condition()
        self._latest: Optional[FramePacket] = None
        self._queue: deque[FramePacket] = deque()
        self._closed = False

    def push(self, packet: FramePacket) -> None:
        with self._cond:
            self._latest = packet
            # Latest-only mode prevents inference from building a stale backlog on CPU.
            if self.latest_only:
                self._queue.clear()
            self._queue.append(packet)
            self._cond.notify_all()

    def latest(self) -> Optional[FramePacket]:
        with self._cond:
            return self._latest

    def pop(self, timeout: float = 0.1) -> Optional[FramePacket]:
        with self._cond:
            if not self._queue and not self._closed:
                self._cond.wait(timeout=timeout)
            if self.latest_only and self._queue:
                packet = self._queue[-1]
                self._queue.clear()
                return packet
            if self._queue:
                return self._queue.popleft()
            return None

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class SnapshotStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[InferenceSnapshot] = None

    def set(self, snapshot: InferenceSnapshot) -> None:
        with self._lock:
            self._latest = snapshot

    def get(self) -> Optional[InferenceSnapshot]:
        with self._lock:
            return self._latest


class CaptureWorker(threading.Thread):
    def __init__(self, source: FrameSource, relay: FrameRelay, stop_event: threading.Event) -> None:
        super().__init__(name="capture-worker", daemon=True)
        self.source = source
        self.relay = relay
        self.stop_event = stop_event

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                packet = self.source.read()
                if packet is None:
                    break
                self.relay.push(packet)
        finally:
            self.relay.close()


class InferenceWorker(threading.Thread):
    def __init__(self, config: AppConfig, relay: FrameRelay, snapshots: SnapshotStore, stop_event: threading.Event) -> None:
        super().__init__(name="inference-worker", daemon=True)
        self.config = config
        self.relay = relay
        self.snapshots = snapshots
        self.stop_event = stop_event

    def run(self) -> None:
        from detector_adapter import KataCRDetector
        from overlay import project_detection_to_frame
        from regions import ArenaMapping, extract_regions

        detector = KataCRDetector(
            weights=[str(path) for path in self.config.weights],
            device=self.config.device,
            conf_thres=self.config.conf_thres,
            iou_thres=self.config.iou_thres,
            allowed_class_ids=self.config.allowed_class_ids,
        )
        arena_mapping_cache: dict[tuple[int, int], ArenaMapping] = {}

        while not self.stop_event.is_set():
            packet = self.relay.pop(timeout=0.1)
            if packet is None:
                continue
            if self.config.frame_skip and packet.frame_index % (self.config.frame_skip + 1) != 0:
                continue

            regions = extract_regions(
                packet.frame_bgr,
                playback=self.config.playback_crop,
                infer_size=self.config.infer_size,
                # Skip extra crops when the UI is in a low-latency mode.
                include_auxiliary=not (self.config.lightweight or self.config.arena_only),
            )
            det_start = time.perf_counter()
            arena_detections = detector.predict(regions.arena_for_model_bgr)
            inference_ms = (time.perf_counter() - det_start) * 1000.0

            projected_frame_detections = [project_detection_to_frame(det, regions.arena_mapping) for det in arena_detections]
            arena_shape = regions.arena_debug_bgr.shape[:2]
            arena_mapping = arena_mapping_cache.get(arena_shape)
            if arena_mapping is None:
                arena_mapping = ArenaMapping(
                    source_xyxy=(0, 0, regions.arena_debug_bgr.shape[1], regions.arena_debug_bgr.shape[0]),
                    model_shape=regions.arena_for_model_bgr.shape[:2],
                )
                arena_mapping_cache[arena_shape] = arena_mapping
            projected_arena_detections = [project_detection_to_frame(det, arena_mapping) for det in arena_detections]
            self.snapshots.set(
                InferenceSnapshot(
                    frame_index=packet.frame_index,
                    timestamp=packet.timestamp,
                    source_name=packet.source_name,
                    projected_frame_detections=projected_frame_detections,
                    projected_arena_detections=projected_arena_detections,
                    regions=regions,
                    inference_ms=inference_ms,
                )
            )


class DebugWriter:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.output
        self.raw_dir = self.cfg.output_dir / "raw_frames"
        self.annotated_dir = self.cfg.output_dir / "annotated_frames"
        self.json_dir = self.cfg.output_dir / "detections_json"
        if self.cfg.save_raw:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.save_annotated:
            self.annotated_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.save_json:
            self.json_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        frame_index: int,
        raw_bgr: np.ndarray,
        annotated_bgr: np.ndarray,
        detections: list["Detection"],
        source_name: str,
        timestamp: float,
    ) -> None:
        stem = f"{frame_index:06d}"
        if self.cfg.save_raw:
            cv2.imwrite(str(self.raw_dir / f"{stem}.jpg"), raw_bgr)
        if self.cfg.save_annotated:
            cv2.imwrite(str(self.annotated_dir / f"{stem}.jpg"), annotated_bgr)
        if self.cfg.save_json:
            payload = {
                "frame_index": frame_index,
                "source": source_name,
                "timestamp": timestamp,
                "detections": [d.to_dict() for d in detections],
            }
            (self.json_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2))


def build_source(config: AppConfig) -> FrameSource:
    if config.source == "scrcpy":
        return ScrcpyWindowSource(config)
    if config.source == "video":
        if not config.input_source:
            raise RuntimeError("--input is required for --source video")
        return OpenCVFrameSource(config.input_source)
    if config.source == "camera":
        return OpenCVFrameSource(config.input_source or "0")
    raise RuntimeError(f"unsupported source: {config.source}")


def maybe_resize(image: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def format_status(display_fps: float, snapshot: Optional[InferenceSnapshot]) -> str:
    if snapshot is None:
        return f"FPS {display_fps:5.1f} | infer   --- ms | dets   0"
    lag_ms = max(0.0, (time.time() - snapshot.timestamp) * 1000.0)
    return (
        f"FPS {display_fps:5.1f} | infer {snapshot.inference_ms:6.1f} ms | "
        f"dets {len(snapshot.projected_frame_detections):3d} | lag {lag_ms:6.0f} ms"
    )


def main() -> int:
    config = parse_args()
    ensure_katacr_environment(config.dataset_path)

    from detector_adapter import Detection
    from overlay import compose_debug_view, draw_detections

    source = build_source(config)
    relay = FrameRelay(latest_only=config.latest_frame_only)
    snapshots = SnapshotStore()
    writer = DebugWriter(config)
    stop_event = threading.Event()
    capture_worker = CaptureWorker(source, relay, stop_event)
    inference_worker = InferenceWorker(config, relay, snapshots, stop_event)
    capture_worker.start()
    inference_worker.start()

    window_name = "Clash Royale Live Vision"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    window_initialized = False

    display_fps = 0.0
    last_tick = time.perf_counter()
    last_rendered_frame_index = -1

    try:
        while True:
            loop_start = time.perf_counter()
            packet = relay.latest()
            snapshot = snapshots.get()

            if packet is None and not capture_worker.is_alive():
                break
            if packet is None:
                time.sleep(0.005)
                continue

            status_text = format_status(display_fps, snapshot)
            detections: list[Detection] = []

            if config.arena_only:
                if snapshot is None:
                    time.sleep(0.005)
                    continue
                detections = snapshot.projected_arena_detections
                canvas = draw_detections(
                    snapshot.regions.arena_debug_bgr,
                    detections,
                    show_labels=not config.no_labels,
                    show_conf=not config.no_conf,
                    show_belong=config.show_belong,
                    fps_text=status_text,
                )
                raw_frame = snapshot.regions.arena_debug_bgr
                rendered = canvas
                rendered_frame_index = snapshot.frame_index
                render_source_name = snapshot.source_name
                render_timestamp = snapshot.timestamp
            else:
                detections = snapshot.projected_frame_detections if snapshot is not None else []
                canvas = draw_detections(
                    packet.frame_bgr,
                    detections,
                    show_labels=not config.no_labels,
                    show_conf=not config.no_conf,
                    show_belong=config.show_belong,
                    fps_text=status_text,
                )
                rendered = compose_debug_view(canvas, snapshot.regions, include_panels=not config.no_panels) if snapshot is not None else canvas
                raw_frame = packet.frame_bgr
                rendered_frame_index = packet.frame_index
                render_source_name = packet.source_name
                render_timestamp = packet.timestamp

            output_frame = maybe_resize(rendered, config.display_scale)
            if not window_initialized:
                initial_width = max(960, int(output_frame.shape[1] * config.window_scale))
                initial_height = max(540, int(output_frame.shape[0] * config.window_scale))
                cv2.resizeWindow(window_name, initial_width, initial_height)
                window_initialized = True
            cv2.imshow(window_name, output_frame)

            if rendered_frame_index != last_rendered_frame_index:
                writer.write(rendered_frame_index, raw_frame, output_frame, detections, render_source_name, render_timestamp)
                last_rendered_frame_index = rendered_frame_index

            now = time.perf_counter()
            dt = max(1e-6, now - last_tick)
            display_fps = 0.90 * display_fps + 0.10 * (1.0 / dt) if display_fps else 1.0 / dt
            last_tick = now

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if config.max_fps > 0:
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.0, (1.0 / config.max_fps) - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        stop_event.set()
        relay.close()
        capture_worker.join(timeout=1.0)
        inference_worker.join(timeout=1.0)
        source.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
