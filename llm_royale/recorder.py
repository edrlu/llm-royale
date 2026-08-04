#!/usr/bin/env python3
"""
Background video recording, decoupled from the decision loop.

Recording used to piggyback on the planner: one frame written per decision, so
the video ran at whatever rate inference happened to manage — a few frames per
second, and jerkier still whenever a snapshot took longer. The capture itself
can do 30fps, so the limit was never the phone.

This records on its own thread instead. It pulls frames straight from the frame
source at a fixed rate and draws the most recent annotations over them, so the
video stays smooth while the planner runs at its own pace. Frames go down a pipe
to ffmpeg as they are produced, which also means a run that is killed rather
than stopped cleanly still leaves a playable file.
"""

import os
import subprocess
import threading
import time
from typing import Callable, Optional

import numpy as np


class VideoRecorder:
    """Encodes frames to mp4 through an ffmpeg pipe on a background thread."""

    # Ceiling on how much wall time one pass may fill with duplicated frames.
    # Big enough to ride out a stall — loading the YOLO weights blocks grabs for
    # seconds, and a cap below that silently drops the time instead of filling
    # it, which is what turned a 110s match into a 63s fast-forward. Small
    # enough that the mirror window disappearing cannot dump minutes of copies.
    MAX_CATCHUP_SECONDS = 5.0

    def __init__(
        self,
        output_path: str,
        frame_source: Callable[[], Optional[np.ndarray]],
        ffmpeg_bin: str = "ffmpeg",
        fps: float = 12.0,
        annotate: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.output_path = output_path
        self.frame_source = frame_source
        self.ffmpeg_bin = ffmpeg_bin
        self.fps = fps
        self.annotate = annotate

        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._frames_written = 0
        self._size = None

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def _spawn_ffmpeg(self, width: int, height: int) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        self._process = subprocess.Popen(
            [
                self.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{self.fps:g}",
                "-i", "pipe:0",
                "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                self.output_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._size = (width, height)

    def _loop(self) -> None:
        period = 1.0 / self.fps if self.fps > 0 else 0.0
        wall_start = None
        while self._running.is_set():
            started = time.perf_counter()
            frame = self.frame_source()
            if frame is not None:
                if wall_start is None:
                    wall_start = started
                if self.annotate is not None:
                    try:
                        frame = self.annotate(frame)
                    except Exception:
                        # A broken overlay must not take the recording down.
                        pass
                if self._process is None:
                    self._spawn_ffmpeg(frame.shape[1], frame.shape[0])
                # ffmpeg was told one fixed frame size; anything else (the mirror
                # window got resized mid-run) has to be skipped, not streamed.
                if (frame.shape[1], frame.shape[0]) == self._size:
                    # ffmpeg is fed a constant frame rate, so the file only plays
                    # back at real speed if the frame count matches elapsed wall
                    # time. When a grab takes longer than one frame period the
                    # latest frame is repeated to fill the gap, rather than
                    # letting the video silently turn into a fast-forward.
                    due = max(1, int((started - wall_start) * self.fps) + 1 - self._frames_written)
                    payload = np.ascontiguousarray(frame).tobytes()
                    try:
                        catchup_cap = max(1, int(self.MAX_CATCHUP_SECONDS * self.fps))
                        for _ in range(min(due, catchup_cap)):
                            self._process.stdin.write(payload)
                            self._frames_written += 1
                    except (BrokenPipeError, ValueError):
                        break
            remaining = period - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="recorder")
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> Optional[str]:
        """Stop recording and finalize the file. Returns the path, or None."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._process is None:
            return None
        try:
            self._process.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None
        return self.output_path if self._frames_written else None
