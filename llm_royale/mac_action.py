#!/usr/bin/env python3
"""
iPhone actions driven through the macOS "iPhone Mirroring" window.

A mirrored iPhone exposes no input API, so this backend synthesizes mouse events
with Quartz and posts them at the HID tap, which iPhone Mirroring forwards to
the phone as touches.

Device coordinate space here is *window-local points*: (0, 0) is the top-left of
the mirror window's content, and `device_width` / `device_height` are the
window's size in points. `BaseActionExecutor` works in that space unchanged.
"""

import time
from typing import Dict, Optional

import Quartz
from Quartz import (
    CGEventCreateMouseEvent,
    CGEventPost,
    CGEventSetType,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseDragged,
    kCGEventLeftMouseUp,
    kCGEventMouseMoved,
    kCGHIDEventTap,
    kCGMouseButtonLeft,
)

from .action import BaseActionExecutor
from .mirror_capture import MIRROR_BUNDLE_ID, find_mirror_window


# How long the window rect is trusted before we re-read it. The user can move or
# resize the mirror window at any time, but re-reading on every single event
# costs a window-server round trip for no benefit.
BOUNDS_TTL_SEC = 1.0

# Steps used to interpolate a drag. Clash Royale ignores a down/up pair with no
# movement in between when placing a card, so the intermediate drag events are
# required, not cosmetic.
DRAG_STEPS = 12


def _post(event_type, x: float, y: float) -> None:
    event = CGEventCreateMouseEvent(None, event_type, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, event)


def activate_mirror_app() -> bool:
    """Try to bring iPhone Mirroring to the front. Returns True if it is frontmost.

    Not being frontmost is *not* fatal: the window server delivers a synthetic
    click to whatever window sits under the point, so taps land on the mirror
    window even while another app holds focus. Activation only matters as
    occlusion insurance — if some other window covers the mirror, it would eat
    the click instead.
    """
    from AppKit import NSRunningApplication, NSWorkspace

    workspace = NSWorkspace.sharedWorkspace()
    frontmost = workspace.frontmostApplication()
    if frontmost is not None and frontmost.bundleIdentifier() == MIRROR_BUNDLE_ID:
        return True

    apps = NSRunningApplication.runningApplicationsWithBundleIdentifier_(MIRROR_BUNDLE_ID)
    if not apps:
        return False
    apps[0].activateWithOptions_(1 << 1)  # NSApplicationActivateIgnoringOtherApps
    # Activation is asynchronous, but a tap cannot wait on it — capping the
    # wait at ~60ms keeps the action loop responsive when macOS declines to
    # hand over focus, which it often does for a non-GUI process like this one.
    for _ in range(6):
        time.sleep(0.01)
        frontmost = workspace.frontmostApplication()
        if frontmost is not None and frontmost.bundleIdentifier() == MIRROR_BUNDLE_ID:
            return True
    return False


class MirrorActionExecutor(BaseActionExecutor):
    def __init__(self, auto_activate: bool = True, restore_cursor: bool = False):
        self.auto_activate = auto_activate
        self.restore_cursor = restore_cursor
        self.device_width = None
        self.device_height = None
        self.device_serial = "iphone-mirroring"
        self.window_id: Optional[int] = None
        self.window_origin = (0.0, 0.0)
        self._bounds_checked_at = 0.0

    # -- device plumbing -----------------------------------------------------

    def ensure_device_ready(self) -> None:
        now = time.monotonic()
        if self.device_width is not None and (now - self._bounds_checked_at) < BOUNDS_TTL_SEC:
            return
        try:
            window = find_mirror_window()
        except RuntimeError as e:
            raise RuntimeError(str(e)) from e
        self.window_id = window["window_id"]
        self.window_origin = (window["x"], window["y"])
        self.device_width = int(window["width"])
        self.device_height = int(window["height"])
        self._bounds_checked_at = now

    def _to_screen(self, x: float, y: float) -> tuple[float, float]:
        origin_x, origin_y = self.window_origin
        return origin_x + x, origin_y + y

    def _focus(self) -> bool:
        if not self.auto_activate:
            return True
        return activate_mirror_app()

    def _cursor_position(self) -> tuple[float, float]:
        event = Quartz.CGEventCreate(None)
        point = Quartz.CGEventGetLocation(event)
        return point.x, point.y

    # -- primitives ----------------------------------------------------------

    def tap_device(self, x: int, y: int) -> Dict:
        self.ensure_device_ready()
        saved = self._cursor_position() if self.restore_cursor else None
        focused = self._focus()
        screen_x, screen_y = self._to_screen(x, y)

        started = time.perf_counter()
        _post(kCGEventMouseMoved, screen_x, screen_y)
        _post(kCGEventLeftMouseDown, screen_x, screen_y)
        time.sleep(0.03)
        _post(kCGEventLeftMouseUp, screen_x, screen_y)
        duration_ms = (time.perf_counter() - started) * 1000.0

        if saved is not None:
            _post(kCGEventMouseMoved, saved[0], saved[1])

        return {
            "cmd": ["cgevent", "tap", str(x), str(y)],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "frontmost": focused,
            "duration_ms": round(duration_ms, 2),
            "x": x,
            "y": y,
            "screen_x": round(screen_x, 1),
            "screen_y": round(screen_y, 1),
        }

    def swipe_device(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 120) -> Dict:
        self.ensure_device_ready()
        saved = self._cursor_position() if self.restore_cursor else None
        focused = self._focus()
        start_x, start_y = self._to_screen(x1, y1)
        end_x, end_y = self._to_screen(x2, y2)

        step_sleep = max(0.0, (duration_ms / 1000.0) / DRAG_STEPS)
        started = time.perf_counter()
        _post(kCGEventMouseMoved, start_x, start_y)
        _post(kCGEventLeftMouseDown, start_x, start_y)
        for step in range(1, DRAG_STEPS + 1):
            t = step / DRAG_STEPS
            _post(kCGEventLeftMouseDragged, start_x + (end_x - start_x) * t, start_y + (end_y - start_y) * t)
            if step_sleep:
                time.sleep(step_sleep)
        _post(kCGEventLeftMouseUp, end_x, end_y)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if saved is not None:
            _post(kCGEventMouseMoved, saved[0], saved[1])

        return {
            "cmd": ["cgevent", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "frontmost": focused,
            "duration_ms": round(elapsed_ms, 2),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "swipe_duration_ms": duration_ms,
            "screen_start": [round(start_x, 1), round(start_y, 1)],
            "screen_end": [round(end_x, 1), round(end_y, 1)],
        }


def main() -> int:
    """Diagnostic: post a single tap at a normalized point in the mirror window."""
    import argparse

    parser = argparse.ArgumentParser(description="Send one synthetic tap to iPhone Mirroring")
    parser.add_argument("--x-norm", type=float, required=True)
    parser.add_argument("--y-norm", type=float, required=True)
    args = parser.parse_args()

    executor = MirrorActionExecutor()
    executor.ensure_device_ready()
    print(
        f"[INFO] window {executor.window_id} at {executor.window_origin} "
        f"size {executor.device_width}x{executor.device_height}"
    )
    print("[INFO]", executor.tap_normalized(args.x_norm, args.y_norm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
