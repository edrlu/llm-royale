from __future__ import annotations

import os
import re
import shlex
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import mss
import numpy as np

from config import AppConfig


@dataclass(slots=True)
class FramePacket:
    frame_bgr: np.ndarray
    source_name: str
    frame_index: int
    # Set to time.time() BEFORE the capture call so lag includes transfer time.
    timestamp: float
    # How long the capture call itself took (ADB transfer, PNG decode, etc.).
    # 0.0 for streaming sources where per-frame cost is not separately measurable.
    capture_ms: float = 0.0


@dataclass(slots=True)
class CommandProbe:
    label: str
    cmd: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


SCRCPY_DIRECT_PROTOCOL_DESCRIPTION = (
    "adb forward -> localabstract scrcpy socket -> optional 1 dummy byte -> 64-byte device name "
    "field -> 12-byte codec header (codec_id,width,height as big-endian uint32) -> raw H.264 "
    "Annex-B packets; send_frame_meta=false means no per-packet PTS header is expected"
)


def _format_cmd(cmd: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(cmd)) if os.name == "nt" else shlex.join(cmd)


def _trim_text(value: str, max_chars: int = 1200) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...<trimmed>"


def _run_probe(cmd: Sequence[str], label: str, timeout: float = 10.0) -> CommandProbe:
    result = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CommandProbe(label=label, cmd=tuple(cmd), returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def _probe_scrcpy_server_features(server_path: Path) -> dict[str, bool]:
    with zipfile.ZipFile(server_path) as bundle:
        dex_bytes = bundle.read("classes.dex")
    return {
        "raw_stream": b"raw_stream" in dex_bytes,
        "send_dummy_byte": b"send_dummy_byte" in dex_bytes,
        "send_device_meta": b"send_device_meta" in dex_bytes,
        "send_codec_meta": b"send_codec_meta" in dex_bytes,
        "send_frame_meta": b"send_frame_meta" in dex_bytes,
        "scid": b"scid" in dex_bytes,
    }


def _pump_text_stream(
    stream,
    sink: deque[str],
    name: str,
    debug_fn: Callable[[str], None],
) -> None:
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
            debug_fn(f"{name}: {text.rstrip()}")
    finally:
        try:
            stream.close()
        except Exception:
            pass


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
        self.proc_cmd: list[str] = []
        self.proc_stdout: deque[str] = deque(maxlen=40)
        self.proc_stderr: deque[str] = deque(maxlen=40)
        self.proc_stdout_thread: Optional[threading.Thread] = None
        self.proc_stderr_thread: Optional[threading.Thread] = None
        self.frame_index = 0
        self.window = None
        self.capture_region = config.capture_region
        self._start_scrcpy_if_needed()
        if self.capture_region is None:
            self.window = self._wait_for_window(config.scrcpy_window_title)

    def _debug(self, message: str) -> None:
        if self.config.debug_source:
            print(f"[source] {message}", file=sys.stderr, flush=True)

    def _tail_text(self, lines: deque[str]) -> str:
        return "".join(lines).strip()

    def _failure_context(self) -> str:
        parts = [
            f"scrcpy_cmd={_format_cmd(self.proc_cmd)}",
            f"scrcpy_rc={None if self.proc is None else self.proc.poll()}",
        ]
        stdout_tail = self._tail_text(self.proc_stdout)
        stderr_tail = self._tail_text(self.proc_stderr)
        if stdout_tail:
            parts.append(f"scrcpy_stdout_tail={_trim_text(stdout_tail)}")
        if stderr_tail:
            parts.append(f"scrcpy_stderr_tail={_trim_text(stderr_tail)}")
        return " | ".join(parts)

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
        version_probe = _run_probe([self.config.scrcpy_path, "--version"], "scrcpy --version")
        self._debug(f"{version_probe.label} cmd: {_format_cmd(version_probe.cmd)}")
        self._debug(f"{version_probe.label} rc={version_probe.returncode}")
        if version_probe.stdout.strip():
            self._debug(f"{version_probe.label} stdout: {_trim_text(version_probe.stdout)}")
        if version_probe.stderr.strip():
            self._debug(f"{version_probe.label} stderr: {_trim_text(version_probe.stderr)}")
        self.proc_cmd = cmd
        self._debug(f"scrcpy window cmd: {_format_cmd(cmd)}")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.proc_stdout_thread = threading.Thread(
            target=_pump_text_stream,
            args=(self.proc.stdout, self.proc_stdout, "scrcpy-window-stdout", self._debug),
            name="scrcpy-window-stdout",
            daemon=True,
        )
        self.proc_stderr_thread = threading.Thread(
            target=_pump_text_stream,
            args=(self.proc.stderr, self.proc_stderr, "scrcpy-window-stderr", self._debug),
            name="scrcpy-window-stderr",
            daemon=True,
        )
        self.proc_stdout_thread.start()
        self.proc_stderr_thread.start()

    def _wait_for_window(self, title: str, timeout: float = 15.0):
        import pywinctl as pwc

        deadline = time.time() + timeout
        while time.time() < deadline:
            windows = pwc.getWindowsWithTitle(title)
            windows = [w for w in windows if w.width > 100 and w.height > 100]
            if windows:
                return windows[0]
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"scrcpy exited before a capture window appeared ({self._failure_context()})")
            time.sleep(0.25)
        raise RuntimeError(f"unable to find scrcpy window titled '{title}' within {timeout:.1f}s ({self._failure_context()})")

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
        if self.proc_stdout_thread is not None:
            self.proc_stdout_thread.join(timeout=1.0)
        if self.proc_stderr_thread is not None:
            self.proc_stderr_thread.join(timeout=1.0)
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
        self.forward_port: Optional[int] = None
        self.socket_name = "scrcpy"
        self.socket_scid: Optional[int] = None
        self.server_cmd: list[str] = []
        self.ffmpeg_cmd: list[str] = []
        self.connect_timeout_seconds = 8.0
        self.startup_attempt_limit = 2
        self.startup_attempt = 0
        self.startup_started_at = 0.0
        self.bytes_received = 0
        self.any_bytes_received = False
        self.payload_bytes_received = 0
        self.startup_stage = "init"
        self.frame_width = 0
        self.frame_height = 0
        self.frame_bytes = 0
        self.scrcpy_version_probe: Optional[CommandProbe] = None
        self.scrcpy_help_probe: Optional[CommandProbe] = None
        self.server_feature_probe: dict[str, bool] = {}
        self.server_stdout: deque[str] = deque(maxlen=40)
        self.server_stderr: deque[str] = deque(maxlen=40)
        self.ffmpeg_stdout: deque[str] = deque(maxlen=10)
        self.ffmpeg_stderr: deque[str] = deque(maxlen=40)
        self.server_stdout_thread: Optional[threading.Thread] = None
        self.server_stderr_thread: Optional[threading.Thread] = None
        self.ffmpeg_stdout_thread: Optional[threading.Thread] = None
        self.ffmpeg_stderr_thread: Optional[threading.Thread] = None
        try:
            self._start()
        except Exception as exc:
            failure = RuntimeError(self._startup_diagnostics(f"scrcpy direct startup failed: {exc}"))
            self._cleanup_runtime()
            raise failure from exc

    def _debug(self, message: str) -> None:
        if self.config.debug_source:
            print(f"[source] {message}", file=sys.stderr, flush=True)

    def _tail_text(self, lines: deque[str]) -> str:
        return "".join(lines).strip()

    def _protocol_expectation(self) -> str:
        return SCRCPY_DIRECT_PROTOCOL_DESCRIPTION

    def _failure_context(self) -> str:
        parts = [
            f"startup_attempt={self.startup_attempt}",
            f"startup_stage={self.startup_stage}",
            f"startup_timeout_s={self.connect_timeout_seconds:.1f}",
            f"bytes_received={self.bytes_received}",
            f"any_bytes_received={self.any_bytes_received}",
            f"payload_bytes_received={self.payload_bytes_received}",
            f"expected_protocol={self._protocol_expectation()}",
            f"socket_name={self.socket_name}",
            f"forward_port={self.forward_port}",
        ]
        if self.server_proc is not None:
            parts.append(f"server_rc={self.server_proc.poll()}")
        if self.ffmpeg_proc is not None:
            parts.append(f"ffmpeg_rc={self.ffmpeg_proc.poll()}")
        if self.server_cmd:
            parts.append(f"server_cmd={_format_cmd(self.server_cmd)}")
        if self.ffmpeg_cmd:
            parts.append(f"ffmpeg_cmd={_format_cmd(self.ffmpeg_cmd)}")
        server_stdout_tail = self._tail_text(self.server_stdout)
        server_tail = self._tail_text(self.server_stderr)
        ffmpeg_stdout_tail = self._tail_text(self.ffmpeg_stdout)
        ffmpeg_tail = self._tail_text(self.ffmpeg_stderr)
        if server_stdout_tail:
            parts.append(f"server_stdout_tail={_trim_text(server_stdout_tail)}")
        if server_tail:
            parts.append(f"server_stderr_tail={_trim_text(server_tail)}")
        if ffmpeg_stdout_tail:
            parts.append(f"ffmpeg_stdout_tail={_trim_text(ffmpeg_stdout_tail)}")
        if ffmpeg_tail:
            parts.append(f"ffmpeg_stderr_tail={_trim_text(ffmpeg_tail)}")
        return " | ".join(parts)

    def _startup_diagnostics(self, reason: str) -> str:
        lines = [
            reason,
            f"direct_transport=adb-forward-localabstract-socket",
            f"expected_protocol={self._protocol_expectation()}",
            f"startup_attempt={self.startup_attempt}",
            f"startup_stage={self.startup_stage}",
            f"startup_timeout_s={self.connect_timeout_seconds:.1f}",
            f"bytes_received={self.bytes_received}",
            f"any_bytes_received={self.any_bytes_received}",
            f"payload_bytes_received={self.payload_bytes_received}",
            f"socket_name={self.socket_name}",
            f"forward_port={self.forward_port}",
            f"server_feature_probe={self.server_feature_probe}",
        ]
        if self.scrcpy_version_probe is not None:
            lines.extend(
                [
                    f"{self.scrcpy_version_probe.label}_cmd={_format_cmd(self.scrcpy_version_probe.cmd)}",
                    f"{self.scrcpy_version_probe.label}_rc={self.scrcpy_version_probe.returncode}",
                ]
            )
            if self.scrcpy_version_probe.stdout.strip():
                lines.append(f"{self.scrcpy_version_probe.label}_stdout={_trim_text(self.scrcpy_version_probe.stdout)}")
            if self.scrcpy_version_probe.stderr.strip():
                lines.append(f"{self.scrcpy_version_probe.label}_stderr={_trim_text(self.scrcpy_version_probe.stderr)}")
        if self.scrcpy_help_probe is not None:
            help_text = f"{self.scrcpy_help_probe.stdout}\n{self.scrcpy_help_probe.stderr}"
            stdout_markers = ("--stdout", "pipe:1", "--video-output", "--output=-")
            lines.append(f"scrcpy_help_cmd={_format_cmd(self.scrcpy_help_probe.cmd)}")
            lines.append(f"scrcpy_help_rc={self.scrcpy_help_probe.returncode}")
            lines.append(f"scrcpy_help_has_direct_stdout_flag={any(marker in help_text for marker in stdout_markers)}")
            lines.append(f"scrcpy_help_has_v4l2_sink={'--v4l2-sink' in help_text}")
        lines.append(self._failure_context())
        return "\n".join(lines)

    def _adb_base(self) -> list[str]:
        cmd = [self.config.adb_path]
        if self.config.scrcpy_serial:
            cmd.extend(["-s", self.config.scrcpy_serial])
        return cmd

    def _run_adb(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = [*self._adb_base(), *args]
        self._debug(f"adb cmd: {_format_cmd(cmd)}")
        result = subprocess.run(
            cmd,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._debug(f"adb rc: {result.returncode}")
        if result.stdout.strip():
            self._debug(f"adb stdout: {_trim_text(result.stdout)}")
        if result.stderr.strip():
            self._debug(f"adb stderr: {_trim_text(result.stderr)}")
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
        self.scrcpy_version_probe = _run_probe([self.config.scrcpy_path, "--version"], "scrcpy_version", timeout=10.0)
        self.scrcpy_help_probe = _run_probe([self.config.scrcpy_path, "--help"], "scrcpy_help", timeout=10.0)
        self._debug(f"{self.scrcpy_version_probe.label} cmd: {_format_cmd(self.scrcpy_version_probe.cmd)}")
        self._debug(f"{self.scrcpy_version_probe.label} rc={self.scrcpy_version_probe.returncode}")
        if self.scrcpy_version_probe.stdout.strip():
            self._debug(f"{self.scrcpy_version_probe.label} stdout: {_trim_text(self.scrcpy_version_probe.stdout)}")
        if self.scrcpy_version_probe.stderr.strip():
            self._debug(f"{self.scrcpy_version_probe.label} stderr: {_trim_text(self.scrcpy_version_probe.stderr)}")
        output = f"{self.scrcpy_version_probe.stdout}\n{self.scrcpy_version_probe.stderr}"
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
        self.server_feature_probe = _probe_scrcpy_server_features(server_path)
        self._debug(f"scrcpy={self.config.scrcpy_path}")
        self._debug(f"scrcpy-server={server_path}")
        self._debug(f"scrcpy-version={version}")
        self._debug(f"adb={self.config.adb_path}")
        self._debug(f"ffmpeg={self.config.ffmpeg_path}")
        self._debug(f"scrcpy-server feature probe: {self.server_feature_probe}")
        self._validate_direct_support()
        remote_server_path = "/data/local/tmp/scrcpy-server.jar"
        self._run_adb("push", str(server_path), remote_server_path)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.startup_attempt_limit + 1):
            self.startup_attempt = attempt
            self.startup_started_at = time.time()
            self.bytes_received = 0
            self.any_bytes_received = False
            self.payload_bytes_received = 0
            self.startup_stage = "prepare-forward"
            self.socket_scid = secrets.randbelow(0x7FFFFFFF) if self.server_feature_probe.get("scid") else None
            self.socket_name = "scrcpy" if self.socket_scid is None else f"scrcpy_{self.socket_scid:08x}"
            self._debug(f"direct stream attempt={attempt} socket_name={self.socket_name}")
            try:
                self._allocate_forward()
                self._start_server_process(remote_server_path, version)
                self._connect_and_handshake()
                self.startup_stage = "start-decoder"
                self._start_decoder()
                self.startup_stage = "wait-payload"
                self._wait_for_video_payload()
                self.startup_stage = "ready"
                return
            except Exception as exc:
                last_error = exc
                self._debug(self._startup_diagnostics(f"scrcpy direct attempt {attempt} failed: {exc}"))
                self._cleanup_runtime()
                if attempt < self.startup_attempt_limit:
                    time.sleep(0.25)
        assert last_error is not None
        raise last_error

    def _validate_direct_support(self) -> None:
        required = ("send_dummy_byte", "send_device_meta", "send_codec_meta", "send_frame_meta")
        missing = [name for name in required if not self.server_feature_probe.get(name)]
        if missing:
            raise RuntimeError(
                "installed scrcpy-server does not expose the metadata flags required by this direct socket reader "
                f"(missing={missing}); this app will not guess a stdout transport"
            )

    def _allocate_forward(self) -> None:
        result = self._run_adb("forward", "tcp:0", f"localabstract:{self.socket_name}")
        port_text = (result.stdout or result.stderr).strip()
        match = re.search(r"(\d+)", port_text)
        if not match:
            raise RuntimeError(f"adb did not report the forwarded tcp port ({port_text!r})")
        self.forward_port = int(match.group(1))
        self._debug(f"allocated adb forward tcp:{self.forward_port} -> localabstract:{self.socket_name}")

    def _start_server_process(self, remote_server_path: str, version: str) -> None:
        server_args = [
            version,
            f"log_level={'debug' if self.config.debug_source else 'warn'}",
            f"max_size={self.config.scrcpy_max_size}",
            f"max_fps={self.config.scrcpy_capture_fps}",
            f"video_bit_rate={self.config.scrcpy_video_bit_rate}",
            "tunnel_forward=true",
            "audio=false",
            "control=false",
            "send_frame_meta=false",
            "send_dummy_byte=true",
            "send_device_meta=true",
            "send_codec_meta=true",
            "clipboard_autosync=false",
            f"stay_awake={'true' if self.config.scrcpy_stay_awake else 'false'}",
        ]
        if self.socket_scid is not None:
            server_args.append(f"scid={self.socket_scid:08x}")
        self.server_cmd = [
            *self._adb_base(),
            "shell",
            f"CLASSPATH={remote_server_path}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            *server_args,
        ]
        self._debug(f"server cmd: {_format_cmd(self.server_cmd)}")
        self.server_proc = subprocess.Popen(self.server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.server_stdout_thread = threading.Thread(
            target=_pump_text_stream,
            args=(self.server_proc.stdout, self.server_stdout, "scrcpy-server-stdout", self._debug),
            name="scrcpy-server-stdout",
            daemon=True,
        )
        self.server_stderr_thread = threading.Thread(
            target=_pump_text_stream,
            args=(self.server_proc.stderr, self.server_stderr, "scrcpy-server-stderr", self._debug),
            name="scrcpy-server-stderr",
            daemon=True,
        )
        self.server_stdout_thread.start()
        self.server_stderr_thread.start()

    def _connect_and_handshake(self) -> None:
        deadline = time.time() + self.connect_timeout_seconds
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            self.startup_stage = "connect-socket"
            self.video_socket = self._connect_video_socket()
            self.bytes_received = 0
            self.any_bytes_received = False
            try:
                self.startup_stage = "read-header"
                # adb forward may accept the host TCP connection before the device-side localabstract
                # socket is ready, so an early zero-byte read is retried on the same server attempt.
                self._read_stream_header()
                assert self.video_socket is not None
                self.video_socket.settimeout(None)
                return
            except Exception as exc:
                last_error = exc
                if self._is_transient_prestream_disconnect(exc):
                    self._debug("video socket connected before scrcpy started emitting bytes; retrying")
                    try:
                        self.video_socket.close()
                    except OSError:
                        pass
                    self.video_socket = None
                    time.sleep(0.1)
                    continue
                raise
        if last_error is not None:
            raise RuntimeError(f"timed out while waiting for the scrcpy header after connect retries ({last_error})")
        raise RuntimeError("timed out while waiting for the scrcpy header after connect retries")

    def _connect_video_socket(self) -> socket.socket:
        deadline = time.time() + self.connect_timeout_seconds
        if self.forward_port is None:
            raise RuntimeError("direct stream adb forward port is not initialized")
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

    def _is_transient_prestream_disconnect(self, exc: Exception) -> bool:
        if self.server_proc is None or self.server_proc.poll() is not None:
            return False
        if self.any_bytes_received:
            return False
        message = str(exc)
        return "unexpected end of scrcpy direct video stream while reading 1 bytes (got 0" in message

    def _wait_for_video_payload(self, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.payload_bytes_received > 0:
                return
            if self.ffmpeg_proc is not None and self.ffmpeg_proc.poll() is not None:
                raise RuntimeError(f"ffmpeg exited before any video payload was delivered ({self._failure_context()})")
            if self.server_proc is not None and self.server_proc.poll() is not None:
                raise RuntimeError(f"scrcpy server exited before any video payload was delivered ({self._failure_context()})")
            time.sleep(0.05)
        # On this scrcpy build/device combination we can receive the handshake header but no H.264 payload,
        # which would otherwise leave the reader blocked forever waiting for decoded frames.
        raise RuntimeError(
            "scrcpy delivered the handshake header but no H.264 payload bytes within "
            f"{timeout:.1f}s ({self._failure_context()})"
        )

    def _recv_exact(self, size: int) -> bytes:
        if self.video_socket is None:
            raise RuntimeError("scrcpy direct socket is not initialized")
        chunks = bytearray()
        while len(chunks) < size and not self.stop_event.is_set():
            chunk = self.video_socket.recv(size - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
            self.bytes_received += len(chunk)
            self.any_bytes_received = True
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
        codec_id, width, height = struct.unpack(">III", header)
        self.frame_width = width
        self.frame_height = height
        self.frame_bytes = width * height * 3
        self._debug(f"direct stream header codec_id={codec_id} width={width} height={height}")

    def _start_decoder(self) -> None:
        self.ffmpeg_cmd = [
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
        ]
        self._debug(f"ffmpeg cmd: {_format_cmd(self.ffmpeg_cmd)}")
        self.ffmpeg_proc = subprocess.Popen(
            self.ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
        )
        self.ffmpeg_stderr_thread = threading.Thread(
            target=_pump_text_stream,
            args=(self.ffmpeg_proc.stderr, self.ffmpeg_stderr, "ffmpeg-stderr", self._debug),
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
                self.payload_bytes_received += len(chunk)
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
        self._cleanup_runtime()

    def _cleanup_runtime(self) -> None:
        self.stop_event.set()
        if self.video_socket is not None:
            try:
                self.video_socket.close()
            except OSError:
                pass
            self.video_socket = None
        if self.socket_pump_thread is not None:
            self.socket_pump_thread.join(timeout=1.0)
            self.socket_pump_thread = None
        if self.ffmpeg_proc is not None:
            self.ffmpeg_proc.terminate()
            try:
                self.ffmpeg_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.ffmpeg_proc.kill()
            self.ffmpeg_proc = None
        if self.server_proc is not None:
            self.server_proc.terminate()
            try:
                self.server_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()
            self.server_proc = None
        if self.server_stdout_thread is not None:
            self.server_stdout_thread.join(timeout=1.0)
            self.server_stdout_thread = None
        if self.server_stderr_thread is not None:
            self.server_stderr_thread.join(timeout=1.0)
            self.server_stderr_thread = None
        if self.ffmpeg_stdout_thread is not None:
            self.ffmpeg_stdout_thread.join(timeout=1.0)
            self.ffmpeg_stdout_thread = None
        if self.ffmpeg_stderr_thread is not None:
            self.ffmpeg_stderr_thread.join(timeout=1.0)
            self.ffmpeg_stderr_thread = None
        if self.forward_port is not None:
            try:
                self._run_adb("forward", "--remove", f"tcp:{self.forward_port}", check=False)
            except Exception:
                pass
            self.forward_port = None
        self.stop_event.clear()


class AdbScreenrecordStreamSource(FrameSource):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.frame_index = 0
        self.source_name = "adb-screenrecord"
        self.adb_proc: Optional[subprocess.Popen[bytes]] = None
        self.ffmpeg_proc: Optional[subprocess.Popen[bytes]] = None
        self.adb_cmd: list[str] = []
        self.ffmpeg_cmd: list[str] = []
        self.stop_event = threading.Event()
        self.frame_width = 0
        self.frame_height = 0
        self.frame_bytes = 0
        self.payload_bytes_received = 0
        self.adb_stdout: deque[str] = deque(maxlen=10)
        self.adb_stderr: deque[str] = deque(maxlen=40)
        self.ffmpeg_stderr: deque[str] = deque(maxlen=40)
        self.pump_thread: Optional[threading.Thread] = None
        self.adb_stderr_thread: Optional[threading.Thread] = None
        self.ffmpeg_stderr_thread: Optional[threading.Thread] = None
        self.startup_timeout_seconds = 8.0
        self._start()

    def _debug(self, message: str) -> None:
        if self.config.debug_source:
            print(f"[source] {message}", file=sys.stderr, flush=True)

    def _tail_text(self, lines: deque[str]) -> str:
        return "".join(lines).strip()

    def _adb_base(self) -> list[str]:
        cmd = [self.config.adb_path]
        if self.config.scrcpy_serial:
            cmd.extend(["-s", self.config.scrcpy_serial])
        return cmd

    def _run_adb(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = [*self._adb_base(), *args]
        self._debug(f"adb cmd: {_format_cmd(cmd)}")
        result = subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self._debug(f"adb rc: {result.returncode}")
        if result.stdout.strip():
            self._debug(f"adb stdout: {_trim_text(result.stdout)}")
        if result.stderr.strip():
            self._debug(f"adb stderr: {_trim_text(result.stderr)}")
        return result

    def _failure_context(self) -> str:
        parts = [
            f"payload_bytes_received={self.payload_bytes_received}",
            f"frame_width={self.frame_width}",
            f"frame_height={self.frame_height}",
        ]
        if self.adb_proc is not None:
            parts.append(f"adb_rc={self.adb_proc.poll()}")
        if self.ffmpeg_proc is not None:
            parts.append(f"ffmpeg_rc={self.ffmpeg_proc.poll()}")
        if self.adb_cmd:
            parts.append(f"adb_cmd={_format_cmd(self.adb_cmd)}")
        if self.ffmpeg_cmd:
            parts.append(f"ffmpeg_cmd={_format_cmd(self.ffmpeg_cmd)}")
        adb_stderr_tail = self._tail_text(self.adb_stderr)
        ffmpeg_stderr_tail = self._tail_text(self.ffmpeg_stderr)
        if adb_stderr_tail:
            parts.append(f"adb_stderr_tail={_trim_text(adb_stderr_tail)}")
        if ffmpeg_stderr_tail:
            parts.append(f"ffmpeg_stderr_tail={_trim_text(ffmpeg_stderr_tail)}")
        return " | ".join(parts)

    def _query_display_size(self) -> tuple[int, int]:
        result = self._run_adb("shell", "wm", "size")
        match = re.search(r"Physical size:\s*(\d+)x(\d+)", f"{result.stdout}\n{result.stderr}")
        if not match:
            raise RuntimeError("unable to parse device display size from `adb shell wm size`")
        width = int(match.group(1))
        height = int(match.group(2))
        if self.config.scrcpy_max_size > 0:
            scale = min(1.0, self.config.scrcpy_max_size / max(width, height))
            width = max(8, int(round((width * scale) / 8.0)) * 8)
            height = max(8, int(round((height * scale) / 8.0)) * 8)
        return width, height

    def _start(self) -> None:
        if not shutil.which(self.config.adb_path):
            raise RuntimeError("adb executable was not found. Install adb and/or pass --adb-path.")
        if not shutil.which(self.config.ffmpeg_path):
            raise RuntimeError("ffmpeg executable was not found. Install ffmpeg and/or pass --ffmpeg-path.")

        self.frame_width, self.frame_height = self._query_display_size()
        self.frame_bytes = self.frame_width * self.frame_height * 3
        self.adb_cmd = [
            *self._adb_base(),
            "exec-out",
            "screenrecord",
            "--output-format=h264",
            "--bit-rate",
            str(self.config.scrcpy_video_bit_rate),
            "--size",
            f"{self.frame_width}x{self.frame_height}",
            "-",
        ]
        self.ffmpeg_cmd = [
            self.config.ffmpeg_path,
            "-loglevel",
            "error",
            # Low-latency decode: disable input buffering and GOP accumulation.
            # Without these flags ffmpeg buffers several seconds of H264 before
            # outputting the first frame, which is the primary cause of the
            # multi-second pipeline delay on the adb screenrecord path.
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
        ]
        self._debug(f"adb screenrecord cmd: {_format_cmd(self.adb_cmd)}")
        self._debug(f"ffmpeg cmd: {_format_cmd(self.ffmpeg_cmd)}")
        self.adb_proc = subprocess.Popen(self.adb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.ffmpeg_proc = subprocess.Popen(
            self.ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self.adb_stderr_thread = threading.Thread(
            target=_pump_text_stream,
            args=(self.adb_proc.stderr, self.adb_stderr, "adb-screenrecord-stderr", self._debug),
            name="adb-screenrecord-stderr",
            daemon=True,
        )
        self.ffmpeg_stderr_thread = threading.Thread(
            target=_pump_text_stream,
            args=(self.ffmpeg_proc.stderr, self.ffmpeg_stderr, "adb-screenrecord-ffmpeg-stderr", self._debug),
            name="adb-screenrecord-ffmpeg-stderr",
            daemon=True,
        )
        self.adb_stderr_thread.start()
        self.ffmpeg_stderr_thread.start()
        self.pump_thread = threading.Thread(target=self._pump_adb_to_decoder, name="adb-screenrecord-pump", daemon=True)
        self.pump_thread.start()
        self._wait_for_payload()

    def _wait_for_payload(self) -> None:
        deadline = time.time() + self.startup_timeout_seconds
        while time.time() < deadline:
            if self.payload_bytes_received > 0:
                return
            if self.adb_proc is not None and self.adb_proc.poll() is not None:
                raise RuntimeError(f"adb screenrecord exited before streaming video ({self._failure_context()})")
            if self.ffmpeg_proc is not None and self.ffmpeg_proc.poll() is not None:
                raise RuntimeError(f"ffmpeg exited before decoding adb screenrecord ({self._failure_context()})")
            time.sleep(0.05)
        raise RuntimeError(f"adb screenrecord produced no video payload within {self.startup_timeout_seconds:.1f}s ({self._failure_context()})")

    def _pump_adb_to_decoder(self) -> None:
        assert self.adb_proc is not None and self.adb_proc.stdout is not None
        assert self.ffmpeg_proc is not None and self.ffmpeg_proc.stdin is not None
        try:
            while not self.stop_event.is_set():
                chunk = self.adb_proc.stdout.read(65536)
                if not chunk:
                    break
                self.payload_bytes_received += len(chunk)
                self.ffmpeg_proc.stdin.write(chunk)
        except OSError:
            self._debug("adb screenrecord pump stopped after pipe error")
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
        if self.adb_proc is not None and self.adb_proc.poll() is not None:
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
        if self.pump_thread is not None:
            self.pump_thread.join(timeout=1.0)
        if self.ffmpeg_proc is not None:
            self.ffmpeg_proc.terminate()
            try:
                self.ffmpeg_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.ffmpeg_proc.kill()
        if self.adb_proc is not None:
            self.adb_proc.terminate()
            try:
                self.adb_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.adb_proc.kill()
        if self.adb_stderr_thread is not None:
            self.adb_stderr_thread.join(timeout=1.0)
        if self.ffmpeg_stderr_thread is not None:
            self.ffmpeg_stderr_thread.join(timeout=1.0)


class AdbScreencapSource(FrameSource):
    """Raw `adb exec-out screencap` capture source.

    How it works
    ------------
    Each `read()` runs one `adb exec-out screencap` call and decodes its raw
    output. The raw frame format is a 12-byte header
    (width, height, format as LE uint32) followed by width*height*4 bytes of
    RGBA pixel data, so framing is deterministic without any delimiter.

    Why raw instead of PNG (-p)
    ---------------------------
    PNG encode on device is the dominant cost at high resolution (easily
    1-2 s on a 1440p screen while the game is running).  Raw format skips that
    entirely.  The trade-off is a larger payload (~10 MB for 1080x2400 vs
    ~2 MB PNG), but on USB 2.0 (~25 MB/s real throughput) that is still only
    ~400 ms — far better than 1500 ms encode + 120 ms transport.
    On WiFi ADB the raw payload is worse (no compression, slow link); if you
    are on WiFi, switch to USB or use `--scrcpy-capture-mode auto` (scrcpy
    direct socket) for ~50 ms pipeline latency.

    Per-frame cost breakdown (USB 2.0, 1080p device)
    -------------------------------------------------
    - Device framebuffer read:   ~1 ms
    - USB raw transfer (~10 MB): ~400 ms
    - Host RGBA→BGR + resize:    ~10 ms
    - (no process spawn, no PNG encode)
    """

    # Raw screencap header: 3× LE uint32 = width, height, pixel_format
    _HEADER_SIZE = 12
    _EXTENDED_HEADER_SIZE = 16
    # Android pixel_format 1 = RGBA_8888 (most common); 2 = RGBX_8888 (same layout)
    _RGBA_FORMATS = {1, 2}

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.frame_index = 0
        self.source_name = "adb-screencap"
        self._max_size = config.scrcpy_max_size
        self._stop = threading.Event()
        self._frame_w = 0
        self._frame_h = 0
        self._pixel_fmt = 1

        if not shutil.which(config.adb_path):
            raise RuntimeError(
                f"adb executable not found at '{config.adb_path}'. "
                "Install ADB and/or pass --adb-path."
            )
        self._adb_base: list[str] = [config.adb_path]
        if config.scrcpy_serial:
            self._adb_base += ["-s", config.scrcpy_serial]

        # Sanity-check ADB connectivity up front.
        result = subprocess.run(
            [*self._adb_base, "shell", "echo", "ok"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=10,
        )
        if result.returncode != 0 or "ok" not in result.stdout:
            raise RuntimeError(
                f"ADB device not reachable (rc={result.returncode}): {result.stderr.strip()}"
            )

    def _debug(self, message: str) -> None:
        if self.config.debug_source:
            print(f"[source] {message}", file=sys.stderr, flush=True)

    def _open_stream(self) -> None:
        raise NotImplementedError("AdbScreencapSource uses one-shot subprocess.run() captures.")
        # Do NOT read anything here — blocking reads belong in CaptureWorker's
        # thread, not in the main thread during startup.  Frame dimensions are
        # discovered from each frame's own header inside read().

    def _read_exact(self, n: int) -> bytes:
        raise NotImplementedError("AdbScreencapSource uses one-shot subprocess.run() captures.")

    def _parse_payload(self, payload: bytes) -> Optional[tuple[int, int, int, bytes, int]]:
        if len(payload) < self._HEADER_SIZE:
            return None
        w, h, fmt = struct.unpack_from("<III", payload)
        expected_bytes = w * h * 4
        if fmt not in self._RGBA_FORMATS:
            return None

        standard_size = self._HEADER_SIZE + expected_bytes
        if len(payload) == standard_size:
            return w, h, fmt, payload[self._HEADER_SIZE:], self._HEADER_SIZE

        extended_size = self._EXTENDED_HEADER_SIZE + expected_bytes
        if len(payload) == extended_size:
            return w, h, fmt, payload[self._EXTENDED_HEADER_SIZE:], self._EXTENDED_HEADER_SIZE

        return None

    def read(self) -> Optional[FramePacket]:
        if self._stop.is_set():
            return None
        # Timestamp BEFORE blocking read so lag includes USB transfer time.
        t0 = time.time()
        frame_no = self.frame_index
        self._debug(f"screencap read start frame={frame_no}")
        try:
            result = subprocess.run(
                [*self._adb_base, "exec-out", "screencap"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                check=False,
            )
                # Screen rotated — update stored dimensions.
        except (OSError, subprocess.TimeoutExpired):
            self._debug(f"screencap read failed frame={frame_no}: subprocess error or timeout")
            return None
        capture_ms = (time.time() - t0) * 1000.0
        if result.returncode != 0:
            self._debug(
                f"screencap read failed frame={frame_no}: rc={result.returncode} "
                f"stderr={_trim_text(result.stderr.decode('utf-8', errors='replace') if isinstance(result.stderr, bytes) else str(result.stderr))}"
            )
            return None

        payload = result.stdout
        if len(payload) < self._HEADER_SIZE:
            self._debug(f"screencap read failed frame={frame_no}: short payload bytes={len(payload)}")
            return None
        parsed = self._parse_payload(payload)
        if parsed is None:
            w, h, fmt = struct.unpack_from("<III", payload)
            expected_bytes = w * h * 4
            self._debug(
                f"screencap read failed frame={frame_no}: payload_bytes={len(payload) - self._HEADER_SIZE} "
                f"expected_bytes={expected_bytes} total_bytes={len(payload)} fmt={fmt}"
            )
            return None
        w, h, fmt, raw, header_size = parsed
        if w != self._frame_w or h != self._frame_h:
            self._frame_w, self._frame_h = w, h
            self._pixel_fmt = fmt

        # RGBA (or RGBX) → BGR
        rgba = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

        if self._max_size > 0:
            fh, fw = frame.shape[:2]
            scale = min(1.0, self._max_size / max(fh, fw))
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (max(1, int(fw * scale)), max(1, int(fh * scale))),
                    interpolation=cv2.INTER_AREA,
                )
        packet = FramePacket(
            frame_bgr=frame,
            source_name=self.source_name,
            frame_index=self.frame_index,
            timestamp=t0,
            capture_ms=capture_ms,
        )
        self._debug(
            f"screencap read done frame={frame_no} size={w}x{h} capture_ms={capture_ms:.1f} header_bytes={header_size}"
        )
        self.frame_index += 1
        return packet

    def close(self) -> None:
        self._stop.set()


def build_source(config: AppConfig) -> FrameSource:
    if config.source == "scrcpy":
        if config.scrcpy_capture_mode == "screencap":
            return AdbScreencapSource(config)
        if config.scrcpy_capture_mode == "adb":
            return AdbScreenrecordStreamSource(config)
        if config.scrcpy_capture_mode in ("auto", "direct"):
            try:
                return ScrcpyDirectStreamSource(config)
            except Exception as exc:
                if config.scrcpy_capture_mode == "direct":
                    raise
                print(
                    "direct scrcpy stream setup failed; actionable diagnostics follow before window fallback:\n"
                    f"{exc}\n"
                    "trying adb screenrecord direct stream...",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    return AdbScreenrecordStreamSource(config)
                except Exception as adb_exc:
                    print(
                        "adb screenrecord direct stream setup failed; actionable diagnostics follow before window fallback:\n"
                        f"{adb_exc}\n"
                        "falling back to window capture...",
                        file=sys.stderr,
                        flush=True,
                    )
        return ScrcpyWindowFallbackSource(config)
    if config.source == "video":
        if not config.input_source:
            raise RuntimeError("--input is required for --source video")
        return OpenCVFrameSource(config.input_source)
    if config.source == "camera":
        return OpenCVFrameSource(config.input_source or "0")
    raise RuntimeError(f"unsupported source: {config.source}")
