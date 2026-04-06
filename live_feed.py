"""
Setup:
1. Create and populate the virtual environment with `install_dependencies.sh` or `install_dependencies.bat`.
2. Put KataCR YOLOv8 weights somewhere local, for example `weights/katacr_live.pt`.
3. Make sure `scrcpy` and `adb` are installed and your Android device is visible to `adb devices`.
4. Run:
   `python live_feed.py --source scrcpy --weights /absolute/path/to/weights.pt --device cuda`

Notes:
- This harness uses KataCR's YOLOv8 detector path and KataCR's crop conventions from `split_part.py`.
- The Clash Royale detection dataset path defaults to `vendor/Clash-Royale-Detection-Dataset`.
- TODO: supply your own trained KataCR `.pt` weights. The repos do not ship inference-ready weights here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
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
        return {"left": int(box.left), "top": int(box.top), "width": int(box.width), "height": int(box.height)}

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

    def write(self, frame_index: int, raw_bgr: np.ndarray, annotated_bgr: np.ndarray, detections: list[Detection], source_name: str, timestamp: float) -> None:
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


def format_fps(fps: float, det_ms: float, det_count: int) -> str:
    return f"FPS {fps:5.1f} | infer {det_ms:6.1f} ms | dets {det_count:3d}"


def main() -> int:
    config = parse_args()
    ensure_katacr_environment(config.dataset_path)

    from detector_adapter import Detection, KataCRDetector
    from overlay import compose_debug_view, draw_detections, project_detection_to_frame
    from regions import extract_regions

    detector = KataCRDetector(
        weights=[str(path) for path in config.weights],
        device=config.device,
        conf_thres=config.conf_thres,
        iou_thres=config.iou_thres,
    )
    source = build_source(config)
    writer = DebugWriter(config)

    window_name = "Clash Royale Live Vision"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_projected: list[Detection] = []
    last_det_ms = 0.0
    last_tick = time.perf_counter()
    fps = 0.0

    try:
        while True:
            loop_start = time.perf_counter()
            packet = source.read()
            if packet is None:
                break

            regions = extract_regions(packet.frame_bgr, playback=config.playback_crop)

            if config.frame_skip == 0 or packet.frame_index % (config.frame_skip + 1) == 0:
                det_start = time.perf_counter()
                arena_detections = detector.predict(regions.arena_for_model_bgr)
                last_det_ms = (time.perf_counter() - det_start) * 1000.0
                last_projected = [project_detection_to_frame(det, regions.arena_mapping) for det in arena_detections]

            annotated = draw_detections(
                packet.frame_bgr,
                last_projected,
                show_labels=not config.no_labels,
                show_conf=not config.no_conf,
                show_belong=config.show_belong,
                fps_text=format_fps(fps, last_det_ms, len(last_projected)),
            )
            composed = compose_debug_view(annotated, regions, include_panels=not config.no_panels)
            output_frame = maybe_resize(composed, config.display_scale)
            cv2.imshow(window_name, output_frame)

            writer.write(packet.frame_index, packet.frame_bgr, output_frame, last_projected, packet.source_name, packet.timestamp)

            now = time.perf_counter()
            dt = max(1e-6, now - last_tick)
            fps = 0.90 * fps + 0.10 * (1.0 / dt) if fps else 1.0 / dt
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
        source.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
