#!/usr/bin/env python3
"""
Frame source for the macOS "iPhone Mirroring" window.

Grabs the mirrored iPhone window straight out of the window server with Quartz.
Window capture rather than screen capture, so moving the window is harmless.

Pipeline:

    iPhone Mirroring window
        -> CGWindowListCreateImage(windowID)
        -> CGBitmapContext (does the downscale in CoreGraphics)
        -> numpy BGR frame

Rendering into a pre-allocated bitmap context is what keeps this cheap: the
resize happens inside CoreGraphics and the destination buffer is reused across
frames, so a grab costs a few milliseconds and allocates nothing per frame.

Requires Screen Recording permission for the process that runs this module
(System Settings -> Privacy & Security -> Screen Recording).
"""

import threading
import time
from typing import Optional, Tuple

import numpy as np

from Quartz import (
    CGBitmapContextCreate,
    CGColorSpaceCreateDeviceRGB,
    CGContextDrawImage,
    CGImageGetHeight,
    CGImageGetWidth,
    CGRectMake,
    CGRectNull,
    CGWindowListCopyWindowInfo,
    CGWindowListCreateImage,
    kCGBitmapByteOrder32Little,
    kCGImageAlphaNoneSkipFirst,
    kCGNullWindowID,
    kCGWindowImageBestResolution,
    kCGWindowImageBoundsIgnoreFraming,
    kCGWindowListExcludeDesktopElements,
    kCGWindowListOptionIncludingWindow,
    kCGWindowListOptionOnScreenOnly,
)


MIRROR_OWNER_NAME = "iPhone Mirroring"
MIRROR_BUNDLE_ID = "com.apple.ScreenContinuity"


class MirrorWindowNotFound(RuntimeError):
    pass


class ScreenRecordingDenied(RuntimeError):
    pass


def find_mirror_window(owner_name: str = MIRROR_OWNER_NAME) -> dict:
    """Return {'window_id', 'x', 'y', 'width', 'height'} for the mirror window.

    The largest matching window wins: the app also puts up small helper windows
    (tooltips, the connecting sheet) that we never want to capture.
    """
    infos = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
        kCGNullWindowID,
    )
    best = None
    for info in infos or []:
        if info.get("kCGWindowOwnerName") != owner_name:
            continue
        bounds = info.get("kCGWindowBounds") or {}
        width = int(bounds.get("Width", 0))
        height = int(bounds.get("Height", 0))
        if width < 100 or height < 100:
            continue
        if best is not None and width * height <= best["width"] * best["height"]:
            continue
        best = {
            "window_id": int(info.get("kCGWindowNumber")),
            "x": float(bounds.get("X", 0.0)),
            "y": float(bounds.get("Y", 0.0)),
            "width": width,
            "height": height,
        }
    if best is None:
        raise MirrorWindowNotFound(
            f"No '{owner_name}' window on screen. Open iPhone Mirroring and connect the phone."
        )
    return best


def capture_window_image(window_id: int):
    """Grab the window as a CGImage, or None if the window server refused."""
    return CGWindowListCreateImage(
        CGRectNull,
        kCGWindowListOptionIncludingWindow,
        window_id,
        kCGWindowImageBoundsIgnoreFraming | kCGWindowImageBestResolution,
    )


class _BitmapScaler:
    """Reusable CoreGraphics bitmap context that draws a CGImage at a fixed size."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.buffer = bytearray(width * height * 4)
        self.context = CGBitmapContextCreate(
            self.buffer,
            width,
            height,
            8,
            width * 4,
            CGColorSpaceCreateDeviceRGB(),
            kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little,
        )
        if self.context is None:
            raise RuntimeError("failed to create CGBitmapContext")
        self._rect = CGRectMake(0, 0, width, height)
        self._array = np.frombuffer(self.buffer, dtype=np.uint8).reshape(
            (height, width, 4)
        )

    def to_bgr(self, cg_image) -> np.ndarray:
        CGContextDrawImage(self.context, self._rect, cg_image)
        # The window-server image and this context share the same row order, so
        # the only conversion needed is dropping the alpha channel: the buffer
        # is BGRA (32-little + skip-first alpha), which leaves BGR for OpenCV.
        return np.ascontiguousarray(self._array[:, :, :3])


class MirrorFrameSource:
    """Latest-frame-wins reader for the iPhone Mirroring window.

    A background thread keeps only the most recent frame, and `read_latest()`
    returns (frame, system_latency_ms) — the detector wants the freshest frame,
    never a backlog.
    """

    def __init__(self, max_size: int = 832, target_fps: float = 30.0):
        self.max_size = max_size
        self.target_fps = target_fps
        self.window_id: Optional[int] = None
        self.window_bounds: Optional[dict] = None
        self.frame_width = 0
        self.frame_height = 0
        self._scaler: Optional[_BitmapScaler] = None
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._latest_ms = 0.0
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- setup ---------------------------------------------------------------

    def probe(self) -> dict:
        """Locate the window, size the output frame, verify we can capture."""
        window = find_mirror_window()
        self.window_id = window["window_id"]
        self.window_bounds = window

        image = capture_window_image(self.window_id)
        if image is None:
            raise ScreenRecordingDenied(
                "CGWindowListCreateImage returned None. Grant Screen Recording permission "
                "to this app in System Settings -> Privacy & Security -> Screen Recording, "
                "then restart it."
            )

        native_w = CGImageGetWidth(image)
        native_h = CGImageGetHeight(image)
        scale = min(1.0, self.max_size / max(native_w, native_h))
        self.frame_width = max(2, int(native_w * scale) & ~1)
        self.frame_height = max(2, int(native_h * scale) & ~1)
        self._scaler = _BitmapScaler(self.frame_width, self.frame_height)

        return {
            "window_id": self.window_id,
            "window_bounds": self.window_bounds,
            "native": {"width": native_w, "height": native_h},
            "frame": {"width": self.frame_width, "height": self.frame_height},
            "backing_scale": round(native_w / max(1, window["width"]), 2),
        }

    # -- capture -------------------------------------------------------------

    def grab_once(self) -> Optional[np.ndarray]:
        if self._scaler is None:
            self.probe()
        image = capture_window_image(self.window_id)
        if image is None:
            # The window id goes stale whenever iPhone Mirroring reopens its
            # window (reconnect, resize, zoom change). Re-resolve and retry once.
            try:
                self.window_id = find_mirror_window()["window_id"]
            except MirrorWindowNotFound:
                return None
            image = capture_window_image(self.window_id)
            if image is None:
                return None
        return self._scaler.to_bgr(image)

    def _reader_loop(self):
        min_period = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        while self._running.is_set():
            started = time.perf_counter()
            frame = self.grab_once()
            elapsed = time.perf_counter() - started
            if frame is not None:
                with self._lock:
                    self._latest = frame
                    self._latest_ms = elapsed * 1000.0
            else:
                time.sleep(0.2)
                continue
            remaining = min_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def start(self):
        if self._scaler is None:
            self.probe()
        self._running.set()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def read_latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            if self._latest is None:
                return None, 0.0
            return self._latest.copy(), self._latest_ms

    # -- coordinate mapping --------------------------------------------------

    def refresh_bounds(self) -> dict:
        """Re-read the window rect in screen points (for mapping taps back)."""
        window = find_mirror_window()
        self.window_id = window["window_id"]
        self.window_bounds = window
        return window

    def norm_to_screen(self, x_norm: float, y_norm: float) -> Tuple[float, float]:
        """Map normalized capture coords to global screen points (top-left origin)."""
        bounds = self.window_bounds or self.refresh_bounds()
        x = bounds["x"] + x_norm * bounds["width"]
        y = bounds["y"] + y_norm * bounds["height"]
        return x, y


def main() -> int:
    """Diagnostic: probe the window, grab a frame, report timings."""
    import argparse

    import cv2

    parser = argparse.ArgumentParser(description="Probe the iPhone Mirroring capture path")
    parser.add_argument("--out", default="mirror_probe.png", help="Where to save the grabbed frame")
    parser.add_argument("--max-size", type=int, default=832)
    parser.add_argument("--bench", type=int, default=20, help="Number of grabs to time")
    args = parser.parse_args()

    source = MirrorFrameSource(max_size=args.max_size)
    info = source.probe()
    print("[INFO] window:", info)

    timings = []
    frame = None
    for _ in range(max(1, args.bench)):
        started = time.perf_counter()
        frame = source.grab_once()
        timings.append((time.perf_counter() - started) * 1000.0)
    if frame is None:
        print("[ERROR] grab returned no frame")
        return 1

    timings.sort()
    print(
        f"[INFO] grab ms: min={timings[0]:.1f} median={timings[len(timings) // 2]:.1f} "
        f"max={timings[-1]:.1f}"
    )
    cv2.imwrite(args.out, frame)
    print(f"[INFO] saved {args.out} ({frame.shape[1]}x{frame.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
