"""
Setup:
1. Create and populate the virtual environment with `install_dependencies.sh` or `install_dependencies.bat`.
2. Ensure `scrcpy`, `adb`, and `ffmpeg` are installed and on `PATH` for the direct live path.
3. Put KataCR YOLOv8 weights somewhere local, for example `runs/detector1_v0.7.13.pt`.
4. Run:
   `python live_feed.py --source scrcpy --model-mode single --lightweight --arena-only --infer-size 320`
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import AppConfig, ensure_katacr_environment, parse_args
from frame_sources import FramePacket, FrameSource, build_source


@dataclass(slots=True)
class InferenceSnapshot:
    frame_index: int
    timestamp: float       # time.time() BEFORE capture call (see FramePacket.timestamp)
    source_name: str
    projected_frame_detections: list["Detection"]
    projected_arena_detections: list["Detection"]
    regions: "RegionBundle"
    inference_ms: float    # YOLO predict() only
    extract_ms: float      # extract_regions() crop/resize overhead
    capture_ms: float      # ADB/pipe transfer time (0 for streaming sources)


def pipeline_log(enabled: bool, stage: str, message: str) -> None:
    if not enabled:
        return
    stamp = time.strftime("%H:%M:%S")
    print(f"[pipeline][{stamp}][{stage}] {message}", file=sys.stderr, flush=True)


def render_waiting_frame(message: str, detail: str = "") -> np.ndarray:
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(frame, "Clash Royale YOLO Output", (32, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, message, (32, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 255), 2, cv2.LINE_AA)
    if detail:
        cv2.putText(frame, detail, (32, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


class FrameRelay:
    def __init__(self, latest_only: bool, debug: bool = False) -> None:
        self.latest_only = latest_only
        self.debug = debug
        self._cond = threading.Condition()
        self._latest: Optional[FramePacket] = None
        self._queue: deque[FramePacket] = deque()
        self._closed = False
        self._last_empty_log = 0.0

    def push(self, packet: FramePacket) -> None:
        with self._cond:
            queue_before = len(self._queue)
            self._latest = packet
            # Latest-only mode prevents inference from building a stale backlog on CPU.
            if self.latest_only:
                self._queue.clear()
            self._queue.append(packet)
            pipeline_log(
                self.debug,
                "relay",
                (
                    f"push frame={packet.frame_index} source={packet.source_name} "
                    f"capture_ms={packet.capture_ms:.1f} queue_before={queue_before} queue_after={len(self._queue)}"
                ),
            )
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
                pipeline_log(self.debug, "relay", f"pop latest frame={packet.frame_index} queue_after=0")
                return packet
            if self._queue:
                packet = self._queue.popleft()
                pipeline_log(self.debug, "relay", f"pop fifo frame={packet.frame_index} queue_after={len(self._queue)}")
                return packet
            if not self._closed and time.time() - self._last_empty_log >= 1.0:
                self._last_empty_log = time.time()
                pipeline_log(self.debug, "relay", "pop timeout waiting for frame")
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
        self._last_empty_log = 0.0

    def run(self) -> None:
        try:
            pipeline_log(self.relay.debug, "capture", f"worker started source={type(self.source).__name__}")
            while not self.stop_event.is_set():
                packet = self.source.read()
                if packet is None:
                    if time.time() - self._last_empty_log >= 1.0:
                        self._last_empty_log = time.time()
                        pipeline_log(self.relay.debug, "capture", "source.read() returned no frame")
                    time.sleep(0.005)
                    continue
                pipeline_log(
                    self.relay.debug,
                    "capture",
                    (
                        f"frame={packet.frame_index} source={packet.source_name} "
                        f"shape={packet.frame_bgr.shape[1]}x{packet.frame_bgr.shape[0]} capture_ms={packet.capture_ms:.1f}"
                    ),
                )
                self.relay.push(packet)
        except Exception:
            pipeline_log(self.relay.debug, "capture", f"worker crashed:\n{traceback.format_exc()}")
            self.stop_event.set()
        finally:
            pipeline_log(self.relay.debug, "capture", "worker stopping")
            self.relay.close()


class InferenceWorker(threading.Thread):
    def __init__(self, config: AppConfig, relay: FrameRelay, snapshots: SnapshotStore, stop_event: threading.Event) -> None:
        super().__init__(name="inference-worker", daemon=True)
        self.config = config
        self.relay = relay
        self.snapshots = snapshots
        self.stop_event = stop_event
        self._last_empty_log = 0.0

    def run(self) -> None:
        from detector_adapter import KataCRDetector
        from overlay import project_detection_to_frame
        from regions import ArenaMapping, extract_regions

        pipeline_log(self.config.debug_source, "infer", f"loading detector weights={[str(path) for path in self.config.weights]}")
        detector = KataCRDetector(
            weights=[str(path) for path in self.config.weights],
            device=self.config.device,
            conf_thres=self.config.conf_thres,
            iou_thres=self.config.iou_thres,
            allowed_class_ids=self.config.allowed_class_ids,
        )
        pipeline_log(self.config.debug_source, "infer", "detector ready")
        arena_mapping_cache: dict[tuple[int, int], ArenaMapping] = {}

        try:
            while not self.stop_event.is_set():
                packet = self.relay.pop(timeout=0.1)
                if packet is None:
                    if time.time() - self._last_empty_log >= 1.0:
                        self._last_empty_log = time.time()
                        pipeline_log(self.config.debug_source, "infer", "waiting for frame")
                    continue
                if self.config.frame_skip and packet.frame_index % (self.config.frame_skip + 1) != 0:
                    pipeline_log(self.config.debug_source, "infer", f"skip frame={packet.frame_index} due to frame_skip={self.config.frame_skip}")
                    continue

                pipeline_log(self.config.debug_source, "infer", f"start frame={packet.frame_index}")
                ext_start = time.perf_counter()
                regions = extract_regions(
                    packet.frame_bgr,
                    playback=self.config.playback_crop,
                    infer_size=self.config.infer_size,
                    # Skip extra crops when the UI is in a low-latency mode.
                    include_auxiliary=not (self.config.lightweight or self.config.arena_only),
                )
                extract_ms = (time.perf_counter() - ext_start) * 1000.0
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
                        extract_ms=extract_ms,
                        capture_ms=packet.capture_ms,
                    )
                )
                pipeline_log(
                    self.config.debug_source,
                    "infer",
                    (
                        f"done frame={packet.frame_index} dets={len(arena_detections)} "
                        f"extract_ms={extract_ms:.1f} yolo_ms={inference_ms:.1f}"
                    ),
                )
        except Exception:
            pipeline_log(self.config.debug_source, "infer", f"worker crashed:\n{traceback.format_exc()}")
            self.stop_event.set()


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
        raw_bgr,
        annotated_bgr,
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


def maybe_resize(image, scale: float):
    if abs(scale - 1.0) < 1e-6:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def format_status(snapshot: Optional[InferenceSnapshot]) -> tuple[str, str]:
    """Return (line1, line2) status strings for the two-line HUD banner.

    line1: per-stage breakdown — capture / crop / yolo
    line2: net end-to-end lag + detection count

    Net lag = time.time() - snapshot.timestamp, where timestamp was set at the
    START of the screencap call (before ADB transfer).  It therefore measures:

        screencap_transfer + queue_wait + extract_regions + yolo_inference
        + time since inference completed

    The only part not captured is the gap between the on-screen event and when
    Python happened to call screencap (~half the screencap cycle, typically
    150-300 ms).  For H264 streaming sources (adb/direct) the capture pipeline
    latency is not included here because timestamping happens after decode.
    """
    if snapshot is None:
        return "waiting for first inference...", "dets: 0"

    net_ms = max(0.0, (time.time() - snapshot.timestamp) * 1000.0)
    cycle_ms = snapshot.inference_ms + snapshot.extract_ms
    infer_fps = 1000.0 / cycle_ms if cycle_ms > 0 else 0.0

    # Line 1: per-stage times so each bottleneck is visible
    parts = [f"infer:{infer_fps:.1f}/s"]
    if snapshot.capture_ms > 0:
        parts.append(f"adb:{snapshot.capture_ms:.0f}ms")
    parts.append(f"crop:{snapshot.extract_ms:.0f}ms")
    parts.append(f"yolo:{snapshot.inference_ms:.0f}ms")
    line1 = "  ".join(parts)

    # Line 2: net total, yolo-only contribution, detection count
    det_count = len(snapshot.projected_arena_detections or snapshot.projected_frame_detections)
    line2 = f"NET:{net_ms:.0f}ms  YOLO:{snapshot.inference_ms:.0f}ms  dets:{det_count}"
    return line1, line2


def main() -> int:
    config = parse_args()
    ensure_katacr_environment(config.dataset_path)
    pipeline_log(
        config.debug_source,
        "main",
        (
            f"startup source={config.source} capture_mode={config.scrcpy_capture_mode} "
            f"weights={[str(path) for path in config.weights]} latest_only={config.latest_frame_only}"
        ),
    )

    from detector_adapter import Detection
    from overlay import compose_debug_view, draw_detections

    source = build_source(config)
    relay = FrameRelay(latest_only=config.latest_frame_only, debug=config.debug_source)
    snapshots = SnapshotStore()
    writer = DebugWriter(config)
    stop_event = threading.Event()
    capture_worker = CaptureWorker(source, relay, stop_event)
    inference_worker = InferenceWorker(config, relay, snapshots, stop_event)
    capture_worker.start()
    inference_worker.start()

    window_name = "Clash Royale YOLO Output"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    window_initialized = False

    last_rendered_frame_index = -1
    last_wait_log = 0.0

    try:
        while True:
            loop_start = time.perf_counter()
            packet = relay.latest()
            snapshot = snapshots.get()

            if packet is None and not capture_worker.is_alive():
                pipeline_log(config.debug_source, "display", "capture worker stopped and no packet is available; exiting main loop")
                break
            if packet is None:
                detail = (
                    f"capture_alive={capture_worker.is_alive()} "
                    f"infer_alive={inference_worker.is_alive()} "
                    f"mode={config.scrcpy_capture_mode}"
                )
                output_frame = render_waiting_frame("Waiting for first frame...", detail)
                if not window_initialized:
                    initial_width = max(960, int(output_frame.shape[1] * config.window_scale))
                    initial_height = max(540, int(output_frame.shape[0] * config.window_scale))
                    cv2.resizeWindow(window_name, initial_width, initial_height)
                    window_initialized = True
                cv2.imshow(window_name, output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if time.time() - last_wait_log >= 1.0:
                    last_wait_log = time.time()
                    pipeline_log(config.debug_source, "display", f"waiting for frame {detail}")
                time.sleep(0.005)
                continue

            status_line1, status_line2 = format_status(snapshot)
            detections: list[Detection] = []

            if config.arena_only:
                detections = snapshot.projected_arena_detections if snapshot is not None else []
                # Re-use the arena crop that the inference worker already computed.
                # This avoids running extract_regions() (~30fps) in the display loop —
                # that extra CPU load was directly competing with inference on CPU.
                # On the very first frame (no snapshot yet) we extract once as a fallback.
                if snapshot is not None:
                    arena_display = snapshot.regions.arena_debug_bgr
                else:
                    # No inference snapshot yet — show raw captured frame so
                    # the main thread never calls the blocking extract_regions()
                    # path, which would freeze the window before the first result.
                    arena_display = packet.frame_bgr
                canvas = draw_detections(
                    arena_display,
                    detections,
                    show_labels=not config.no_labels,
                    show_conf=not config.no_conf,
                    show_belong=config.show_belong,
                    status_line1=status_line1,
                    status_line2=status_line2,
                )
                raw_frame = arena_display
                rendered = canvas
                rendered_frame_index = packet.frame_index
                render_source_name = packet.source_name
                render_timestamp = packet.timestamp
            else:
                detections = snapshot.projected_frame_detections if snapshot is not None else []
                canvas = draw_detections(
                    packet.frame_bgr,
                    detections,
                    show_labels=not config.no_labels,
                    show_conf=not config.no_conf,
                    show_belong=config.show_belong,
                    status_line1=status_line1,
                    status_line2=status_line2,
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
            pipeline_log(
                config.debug_source,
                "display",
                (
                    f"render frame={rendered_frame_index} source={render_source_name} "
                    f"snapshot_frame={None if snapshot is None else snapshot.frame_index}"
                ),
            )

            if rendered_frame_index != last_rendered_frame_index:
                writer.write(rendered_frame_index, raw_frame, output_frame, detections, render_source_name, render_timestamp)
                last_rendered_frame_index = rendered_frame_index
                pipeline_log(config.debug_source, "writer", f"wrote frame={rendered_frame_index} dets={len(detections)}")

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if config.max_fps > 0:
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.0, (1.0 / config.max_fps) - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        pipeline_log(config.debug_source, "main", "shutting down")
        stop_event.set()
        relay.close()
        capture_worker.join(timeout=1.0)
        inference_worker.join(timeout=1.0)
        source.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
