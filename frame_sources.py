from __future__ import annotations

import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mss
import numpy as np

from config import AppConfig


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


class ScrcpyWindowFallbackSource(FrameSource):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.mss_instance: Optional[mss.mss] = None
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
        if self.mss_instance is None:
            self.mss_instance = mss.mss()
        monitor = self._current_monitor()
        if monitor["width"] <= 0 or monitor["height"] <= 0:
            time.sleep(0.01)
            return None
        image = np.array(self.mss_instance.grab(monitor), dtype=np.uint8)
        if image.size == 0:
            time.sleep(0.01)
            return None
        frame = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        packet = FramePacket(frame_bgr=frame, source_name="scrcpy-window", frame_index=self.frame_index, timestamp=time.time())
        self.frame_index += 1
        return packet

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.mss_instance is not None:
            self.mss_instance.close()


class ScrcpyDirectStreamSource(FrameSource):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.frame_index = 0
        self.source_name = "scrcpy-direct"
        self.server_proc: Optional[subprocess.Popen[str]] = None
        self.ffmpeg_proc: Optional[subprocess.Popen[bytes]] = None
        self.video_socket: Optional[socket.socket] = None
        self.socket_pump_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.forward_port = 27183
        self.frame_width = 0
        self.frame_height = 0
        self.frame_bytes = 0
        self.server_stderr: deque[str] = deque(maxlen=40)
        self.ffmpeg_stderr: deque[str] = deque(maxlen=40)
        self.server_stderr_thread: Optional[threading.Thread] = None
        self.ffmpeg_stderr_thread: Optional[threading.Thread] = None
        self._start()

    def _debug(self, message: str) -> None:
        if self.config.debug_source:
            print(f"[source] {message}", file=sys.stderr, flush=True)

    def _stderr_mode(self):
        return subprocess.PIPE if self.config.debug_source else subprocess.DEVNULL

    def _tail_text(self, lines: deque[str]) -> str:
        return "".join(lines).strip()

    def _pump_stderr(self, stream, sink: deque[str], name: str) -> None:
        if stream is None:
            return
        try:
            while True:
                line = stream.readline()
                if not line:
                    break
                if isinstance(line, bytes):
                    text = line.decode("utf-8", errors="replace")
                else:
                    text = line
                sink.append(text)
                self._debug(f"{name}: {text.rstrip()}")
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _failure_context(self) -> str:
        parts = []
        if self.server_proc is not None:
            parts.append(f"server_rc={self.server_proc.poll()}")
        if self.ffmpeg_proc is not None:
            parts.append(f"ffmpeg_rc={self.ffmpeg_proc.poll()}")
        server_tail = self._tail_text(self.server_stderr)
        ffmpeg_tail = self._tail_text(self.ffmpeg_stderr)
        if server_tail:
            parts.append(f"server_stderr_tail={server_tail}")
        if ffmpeg_tail:
            parts.append(f"ffmpeg_stderr_tail={ffmpeg_tail}")
        return " | ".join(parts)

    def _adb_base(self) -> list[str]:
        cmd = [self.config.adb_path]
        if self.config.scrcpy_serial:
            cmd.extend(["-s", self.config.scrcpy_serial])
        return cmd

    def _run_adb(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = [*self._adb_base(), *args]
        self._debug(f"adb cmd: {' '.join(cmd)}")
        result = subprocess.run(
            [*self._adb_base(), *args],
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if self.config.debug_source and (result.stdout.strip() or result.stderr.strip()):
            self._debug(f"adb stdout: {result.stdout.strip()}")
            self._debug(f"adb stderr: {result.stderr.strip()}")
        return result

    def _resolve_scrcpy_server_path(self) -> Path:
        env_path = os.environ.get("SCRCPY_SERVER_PATH")
        candidates = []
        if env_path:
            candidates.append(Path(env_path))
        scrcpy_path = Path(self.config.scrcpy_path)
        candidates.extend(
            [
                scrcpy_path.with_name("scrcpy-server"),
                scrcpy_path.with_name("scrcpy-server.jar"),
                scrcpy_path.parent / "scrcpy-server",
                scrcpy_path.parent / "scrcpy-server.jar",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise RuntimeError("unable to locate scrcpy-server. Set SCRCPY_SERVER_PATH or install the full scrcpy distribution.")

    def _resolve_scrcpy_version(self) -> str:
        output = subprocess.check_output([self.config.scrcpy_path, "--version"], text=True, stderr=subprocess.STDOUT)
        match = re.search(r"(\d+\.\d+(?:\.\d+)?)", output)
        if not match:
            raise RuntimeError("unable to parse scrcpy version")
        return match.group(1)

    def _start(self) -> None:
        if not shutil.which(self.config.adb_path):
            raise RuntimeError("adb executable was not found. Install adb and/or pass --adb-path.")
        if not shutil.which(self.config.ffmpeg_path):
            raise RuntimeError("ffmpeg executable was not found. Install ffmpeg and/or pass --ffmpeg-path.")

        server_path = self._resolve_scrcpy_server_path()
        version = self._resolve_scrcpy_version()
        self._debug(f"scrcpy={self.config.scrcpy_path}")
        self._debug(f"scrcpy-server={server_path}")
        self._debug(f"scrcpy-version={version}")
        self._debug(f"adb={self.config.adb_path}")
        self._debug(f"ffmpeg={self.config.ffmpeg_path}")
        remote_server_path = "/data/local/tmp/scrcpy-server.jar"
        self._run_adb("push", str(server_path), remote_server_path)
        self._run_adb("forward", f"tcp:{self.forward_port}", "localabstract:scrcpy")

        server_cmd = [
            *self._adb_base(),
            "shell",
            f"CLASSPATH={remote_server_path}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            version,
            "log_level=warn",
            f"max_size={self.config.scrcpy_max_size}",
            f"max_fps={self.config.scrcpy_capture_fps}",
            f"video_bit_rate={self.config.scrcpy_video_bit_rate}",
            "tunnel_forward=true",
            "audio=false",
            "control=false",
            "send_frame_meta=false",
            "clipboard_autosync=false",
            f"stay_awake={'true' if self.config.scrcpy_stay_awake else 'false'}",
        ]
        self._debug(f"server cmd: {' '.join(server_cmd)}")
        self.server_proc = subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=self._stderr_mode(), text=True)
        if self.config.debug_source:
            self.server_stderr_thread = threading.Thread(
                target=self._pump_stderr,
                args=(self.server_proc.stderr, self.server_stderr, "scrcpy-server"),
                name="scrcpy-server-stderr",
                daemon=True,
            )
            self.server_stderr_thread.start()
        self.video_socket = self._connect_video_socket()
        self._read_stream_header()
        self._start_decoder()

    def _connect_video_socket(self) -> socket.socket:
        deadline = time.time() + 8.0
        while time.time() < deadline:
            try:
                sock = socket.create_connection(("127.0.0.1", self.forward_port), timeout=2.0)
                sock.settimeout(2.0)
                return sock
            except OSError:
                if self.server_proc and self.server_proc.poll() is not None:
                    raise RuntimeError(f"scrcpy server exited before the direct video socket was ready ({self._failure_context()})")
                time.sleep(0.1)
        raise RuntimeError(f"timed out while connecting to the scrcpy direct video socket ({self._failure_context()})")

    def _recv_exact(self, size: int) -> bytes:
        if self.video_socket is None:
            raise RuntimeError("scrcpy direct socket is not initialized")
        chunks = bytearray()
        while len(chunks) < size and not self.stop_event.is_set():
            chunk = self.video_socket.recv(size - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        if len(chunks) != size:
            raise RuntimeError(
                f"unexpected end of scrcpy direct video stream while reading {size} bytes "
                f"(got {len(chunks)}; {self._failure_context()})"
            )
        return bytes(chunks)

    def _read_stream_header(self) -> None:
        dummy_byte = self._recv_exact(1)
        if dummy_byte != b"\x00":
            raise RuntimeError(f"scrcpy direct stream did not return the expected dummy byte ({dummy_byte!r}; {self._failure_context()})")
        self._recv_exact(64)
        header = self._recv_exact(12)
        _, width, height = struct.unpack(">III", header)
        self.frame_width = width
        self.frame_height = height
        self.frame_bytes = width * height * 3
        self._debug(f"direct stream header width={width} height={height}")

    def _start_decoder(self) -> None:
        self.ffmpeg_proc = subprocess.Popen(
            [
                self.config.ffmpeg_path,
                "-loglevel",
                "error",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-analyzeduration",
                "0",
                "-probesize",
                "32",
                "-f",
                "h264",
                "-i",
                "pipe:0",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_mode(),
            text=False,
            bufsize=0,
        )
        if self.config.debug_source:
            self.ffmpeg_stderr_thread = threading.Thread(
                target=self._pump_stderr,
                args=(self.ffmpeg_proc.stderr, self.ffmpeg_stderr, "ffmpeg"),
                name="ffmpeg-stderr",
                daemon=True,
            )
            self.ffmpeg_stderr_thread.start()
        self.socket_pump_thread = threading.Thread(target=self._pump_socket_to_decoder, name="scrcpy-direct-pump", daemon=True)
        self.socket_pump_thread.start()

    def _pump_socket_to_decoder(self) -> None:
        assert self.video_socket is not None
        assert self.ffmpeg_proc is not None and self.ffmpeg_proc.stdin is not None
        try:
            while not self.stop_event.is_set():
                chunk = self.video_socket.recv(65536)
                if not chunk:
                    break
                self.ffmpeg_proc.stdin.write(chunk)
        except OSError:
            self._debug("socket pump stopped after socket error")
            return
        finally:
            try:
                self.ffmpeg_proc.stdin.close()
            except Exception:
                pass

    def _read_exact_stdout(self, size: int) -> bytes:
        assert self.ffmpeg_proc is not None and self.ffmpeg_proc.stdout is not None
        chunks = bytearray()
        while len(chunks) < size and not self.stop_event.is_set():
            chunk = self.ffmpeg_proc.stdout.read(size - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        return bytes(chunks)

    def read(self) -> Optional[FramePacket]:
        if self.ffmpeg_proc is None or self.ffmpeg_proc.stdout is None:
            return None
        if self.server_proc and self.server_proc.poll() is not None:
            return None
        raw = self._read_exact_stdout(self.frame_bytes)
        if len(raw) != self.frame_bytes:
            return None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.frame_height, self.frame_width, 3).copy()
        packet = FramePacket(frame_bgr=frame, source_name=self.source_name, frame_index=self.frame_index, timestamp=time.time())
        self.frame_index += 1
        return packet

    def close(self) -> None:
        self.stop_event.set()
        if self.video_socket is not None:
            try:
                self.video_socket.close()
            except OSError:
                pass
        if self.socket_pump_thread is not None:
            self.socket_pump_thread.join(timeout=1.0)
        if self.ffmpeg_proc is not None:
            self.ffmpeg_proc.terminate()
            try:
                self.ffmpeg_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.ffmpeg_proc.kill()
        if self.server_proc is not None:
            self.server_proc.terminate()
            try:
                self.server_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()
        try:
            self._run_adb("forward", "--remove", f"tcp:{self.forward_port}", check=False)
        except Exception:
            pass


def build_source(config: AppConfig) -> FrameSource:
    if config.source == "scrcpy":
        if config.scrcpy_capture_mode in ("auto", "direct"):
            try:
                return ScrcpyDirectStreamSource(config)
            except Exception as exc:
                if config.scrcpy_capture_mode == "direct":
                    raise
                print(f"direct scrcpy stream setup failed, falling back to window capture: {exc}", file=sys.stderr, flush=True)
        return ScrcpyWindowFallbackSource(config)
    if config.source == "video":
        if not config.input_source:
            raise RuntimeError("--input is required for --source video")
        return OpenCVFrameSource(config.input_source)
    if config.source == "camera":
        return OpenCVFrameSource(config.input_source or "0")
    raise RuntimeError(f"unsupported source: {config.source}")
