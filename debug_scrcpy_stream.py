from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from config import AppConfig, DebugOutputConfig, resolve_tool_path
from frame_sources import ScrcpyDirectStreamSource


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone scrcpy direct-stream diagnostic.")
    parser.add_argument("--scrcpy-path", default="scrcpy")
    parser.add_argument("--adb-path", default="adb")
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--scrcpy-serial", default=None)
    parser.add_argument("--scrcpy-max-size", type=int, default=1600)
    parser.add_argument("--scrcpy-capture-fps", type=int, default=30)
    parser.add_argument("--scrcpy-video-bit-rate", type=int, default=2_000_000)
    parser.add_argument("--scrcpy-no-stay-awake", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="Optional decoded frames to wait for after startup.")
    parser.add_argument("--frame-timeout", type=float, default=10.0, help="Seconds to wait for decoded frames.")
    parser.add_argument("--debug-source", action="store_true", help="Print verbose transport diagnostics to stderr.")
    return parser


def make_config(args: argparse.Namespace) -> AppConfig:
    scrcpy_path = resolve_tool_path(args.scrcpy_path)
    adb_path = resolve_tool_path(
        args.adb_path,
        Path(scrcpy_path).with_name("adb").as_posix(),
        Path(scrcpy_path).with_name("adb.exe").as_posix(),
    )
    ffmpeg_path = resolve_tool_path(args.ffmpeg_path)
    root = Path(__file__).resolve().parent
    return AppConfig(
        source="scrcpy",
        input_source=None,
        weights=(root / "debug_scrcpy_stream.py",),
        model_mode="single",
        device="cpu",
        conf_thres=0.25,
        iou_thres=0.45,
        allowed_class_ids=(0,),
        infer_size=416,
        lightweight=False,
        latest_frame_only=True,
        arena_only=False,
        frame_skip=0,
        max_fps=0.0,
        display_scale=1.0,
        window_scale=1.0,
        no_panels=False,
        no_labels=False,
        no_conf=False,
        show_belong=False,
        dataset_path=root,
        output=DebugOutputConfig(output_dir=root / "outputs" / "scrcpy_debug", save_raw=False, save_annotated=False, save_json=False),
        scrcpy_path=scrcpy_path,
        adb_path=adb_path,
        ffmpeg_path=ffmpeg_path,
        scrcpy_serial=args.scrcpy_serial,
        scrcpy_window_title="llm-royale-scrcpy-debug",
        scrcpy_max_size=max(8, args.scrcpy_max_size),
        scrcpy_capture_mode="direct",
        scrcpy_capture_fps=max(1, args.scrcpy_capture_fps),
        scrcpy_video_bit_rate=max(100_000, args.scrcpy_video_bit_rate),
        debug_source=args.debug_source,
        scrcpy_stay_awake=not args.scrcpy_no_stay_awake,
        scrcpy_fullscreen=False,
        scrcpy_extra_args=(),
        capture_region=None,
        playback_crop=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = make_config(args)
    config.output.output_dir.mkdir(parents=True, exist_ok=True)

    source = None
    try:
        source = ScrcpyDirectStreamSource(config)
        print(f"stream_ready width={source.frame_width} height={source.frame_height} frame_bytes={source.frame_bytes}")
        if args.frames <= 0:
            return 0
        deadline = time.time() + max(0.1, args.frame_timeout)
        frames = 0
        while frames < args.frames and time.time() < deadline:
            packet = source.read()
            if packet is None:
                time.sleep(0.01)
                continue
            frames += 1
            print(
                f"frame[{frames}] shape={packet.frame_bgr.shape} "
                f"timestamp={packet.timestamp:.6f} index={packet.frame_index}"
            )
        if frames < args.frames:
            print(
                f"timed out after {args.frame_timeout:.1f}s while waiting for {args.frames} decoded frame(s); "
                f"received={frames}",
                file=sys.stderr,
            )
            return 2
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if source is not None:
            source.close()


if __name__ == "__main__":
    raise SystemExit(main())
