from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parent
VENDOR_DIR = ROOT / "vendor"
KATACR_ROOT = VENDOR_DIR / "KataCR"
DATASET_ROOT = VENDOR_DIR / "Clash-Royale-Detection-Dataset"
OUTPUT_ROOT = ROOT / "outputs"
DEFAULT_WEIGHTS_DIR = ROOT / "weights"


def ensure_katacr_environment(dataset_path: Optional[Path] = None) -> None:
    dataset = (dataset_path or DATASET_ROOT).expanduser().resolve()
    os.environ["KATACR_DATASET_PATH"] = str(dataset)
    katacr_path = str(KATACR_ROOT)
    if katacr_path not in sys.path:
        sys.path.insert(0, katacr_path)


@dataclass(slots=True)
class DebugOutputConfig:
    output_dir: Path
    save_raw: bool
    save_annotated: bool
    save_json: bool


@dataclass(slots=True)
class AppConfig:
    source: str
    input_source: Optional[str]
    weights: Path
    device: str
    conf_thres: float
    iou_thres: float
    frame_skip: int
    max_fps: float
    display_scale: float
    no_panels: bool
    no_labels: bool
    no_conf: bool
    show_belong: bool
    dataset_path: Path
    output: DebugOutputConfig
    scrcpy_path: str
    scrcpy_serial: Optional[str]
    scrcpy_window_title: str
    scrcpy_max_size: int
    scrcpy_stay_awake: bool
    scrcpy_fullscreen: bool
    scrcpy_extra_args: Sequence[str]
    capture_region: Optional[tuple[int, int, int, int]]
    playback_crop: bool


def parse_capture_region(value: Optional[str]) -> Optional[tuple[int, int, int, int]]:
    if not value:
        return None
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("capture region must be left,top,width,height")
    return tuple(parts)  # type: ignore[return-value]


def resolve_weights_path(raw_weights: Optional[Path], parser: argparse.ArgumentParser) -> Path:
    if raw_weights is not None:
        weights = raw_weights.expanduser().resolve()
        if not weights.exists():
            parser.error(f"weights file does not exist: {weights}")
        return weights

    candidates = sorted(path.resolve() for path in DEFAULT_WEIGHTS_DIR.glob("*.pt") if path.is_file())
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        options = "\n".join(f"  - {path}" for path in candidates)
        parser.error(
            "multiple weights files found in the local weights directory; pass --weights explicitly:\n"
            f"{options}"
        )

    parser.error(
        "no weights file was provided. Either pass --weights /path/to/model.pt or place exactly one "
        f"`.pt` file in {DEFAULT_WEIGHTS_DIR}"
    )
    raise AssertionError("unreachable")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live Clash Royale perception viewer using KataCR detection and scrcpy capture.",
    )
    parser.add_argument("--source", choices=("scrcpy", "video", "camera"), default="scrcpy")
    parser.add_argument("--input", dest="input_source", default=None, help="Video path or camera index for non-scrcpy sources.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to KataCR YOLOv8 .pt weights. If omitted, auto-detect a single .pt file in weights/.",
    )
    parser.add_argument("--device", default="cuda", help="Ultralytics device string, for example cuda, cuda:0, or cpu.")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--frame-skip", type=int, default=0, help="Run detection every N+1 frames.")
    parser.add_argument("--max-fps", type=float, default=0.0, help="Optional display-rate cap. 0 disables throttling.")
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--no-panels", action="store_true")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--no-conf", action="store_true")
    parser.add_argument("--show-belong", action="store_true", help="Color and tag detections by KataCR belong/team output when available.")
    parser.add_argument("--dataset-path", type=Path, default=DATASET_ROOT, help="Dataset root for KataCR imports and conventions.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "live_viewer")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--capture-region", type=parse_capture_region, default=None, help="Manual capture region as left,top,width,height.")
    parser.add_argument("--playback-crop", action="store_true", help="Use KataCR playback crop for arena extraction instead of live crop.")
    parser.add_argument("--scrcpy-path", default="scrcpy")
    parser.add_argument("--scrcpy-serial", default=None, help="ADB serial to pass to scrcpy.")
    parser.add_argument("--scrcpy-window-title", default="llm-royale-scrcpy")
    parser.add_argument("--scrcpy-max-size", type=int, default=1600, help="scrcpy max video size to reduce capture cost.")
    parser.add_argument("--scrcpy-no-stay-awake", action="store_true")
    parser.add_argument("--scrcpy-fullscreen", action="store_true")
    parser.add_argument("--scrcpy-extra-args", default="", help="Extra raw arguments passed to scrcpy.")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> AppConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    weights = resolve_weights_path(args.weights, parser)
    dataset_path = args.dataset_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        source=args.source,
        input_source=args.input_source,
        weights=weights,
        device=args.device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        frame_skip=max(0, args.frame_skip),
        max_fps=max(0.0, args.max_fps),
        display_scale=max(0.1, args.display_scale),
        no_panels=args.no_panels,
        no_labels=args.no_labels,
        no_conf=args.no_conf,
        show_belong=args.show_belong,
        dataset_path=dataset_path,
        output=DebugOutputConfig(
            output_dir=output_dir,
            save_raw=args.save_raw,
            save_annotated=args.save_annotated,
            save_json=args.save_json,
        ),
        scrcpy_path=args.scrcpy_path,
        scrcpy_serial=args.scrcpy_serial,
        scrcpy_window_title=args.scrcpy_window_title,
        scrcpy_max_size=args.scrcpy_max_size,
        scrcpy_stay_awake=not args.scrcpy_no_stay_awake,
        scrcpy_fullscreen=args.scrcpy_fullscreen,
        scrcpy_extra_args=tuple(x for x in args.scrcpy_extra_args.split() if x),
        capture_region=args.capture_region,
        playback_crop=args.playback_crop,
    )
