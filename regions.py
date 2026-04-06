from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from config import ensure_katacr_environment

ensure_katacr_environment()

from katacr.build_dataset.constant import part3_elixir_params, part_sizes, ratio, split_bbox_params  # noqa: E402
from katacr.build_dataset.utils.split_part import extract_bbox, process_part  # noqa: E402


@dataclass(slots=True)
class ArenaMapping:
    source_xyxy: tuple[int, int, int, int]
    model_shape: tuple[int, int]


@dataclass(slots=True)
class RegionBundle:
    arena_for_model_bgr: np.ndarray
    arena_mapping: ArenaMapping
    arena_debug_bgr: np.ndarray
    hand_cards_bgr: np.ndarray
    elixir_bgr: np.ndarray
    timer_hp_bgr: np.ndarray
    center_text_bgr: Optional[np.ndarray]


def _compute_abs_bbox(image: np.ndarray, params: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    x, y, bw, bh = params
    left = int(w * x)
    top = int(h * y)
    width = int(w * bw)
    height = int(h * bh)
    return left, top, left + width, top + height


def _ratio_name(image_rgb: np.ndarray) -> str:
    image_ratio = image_rgb.shape[0] / image_rgb.shape[1]
    candidates = {
        name: bounds
        for name, bounds in ratio.items()
        if name != "part2"
    }
    for name, (lower, upper) in candidates.items():
        if lower <= image_ratio <= upper:
            return name

    # Window capture can be off by a pixel after scaling; pick the nearest supported profile.
    nearest_name, nearest_bounds = min(
        candidates.items(),
        key=lambda item: abs(image_ratio - ((item[1][0] + item[1][1]) / 2.0)),
    )
    nearest_center = (nearest_bounds[0] + nearest_bounds[1]) / 2.0
    if abs(image_ratio - nearest_center) > 0.08:
        raise ValueError(f"unsupported frame ratio {image_ratio:.4f}")
    return nearest_name


def _manual_process_part(image_rgb: np.ndarray, part: int | str, playback: bool = False, resize: bool = True) -> Any:
    part_name = part if isinstance(part, str) else f"part{part}"
    resize_key = f"part{part}" if isinstance(part, int) else part
    key = part_name
    if playback:
        key += "_playback"
    key += f"_{_ratio_name(image_rgb)}"

    bbox_params = split_bbox_params.get(key)
    if bbox_params is None:
        return None

    target_size = part_sizes.get(resize_key) if resize else None
    if isinstance(bbox_params, dict):
        return {name: extract_bbox(image_rgb, *bbox, target_size) for name, bbox in bbox_params.items()}
    return extract_bbox(image_rgb, *bbox_params, target_size)


def _safe_process_part(
    image_rgb: np.ndarray,
    part: int | str,
    playback: bool = False,
    resize: bool = True,
    verbose: bool = False,
    allow_missing: bool = False,
) -> Any:
    try:
        return process_part(image_rgb, part, playback=playback, resize=resize, verbose=verbose)
    except (KeyError, TypeError):
        if verbose:
            manual = _manual_process_part(image_rgb, part, playback=playback, resize=resize)
            if manual is None:
                if allow_missing:
                    return None, None
                raise
            part_name = part if isinstance(part, str) else f"part{part}"
            key = part_name
            if playback:
                key += "_playback"
            key += f"_{_ratio_name(image_rgb)}"
            bbox_params = split_bbox_params.get(key)
            if bbox_params is None:
                if allow_missing:
                    return None, None
                raise
            return manual, bbox_params
        manual = _manual_process_part(image_rgb, part, playback=playback, resize=resize)
        if manual is None:
            if allow_missing:
                return None
            raise
        return manual


def _stack_vertical(images: list[np.ndarray]) -> np.ndarray:
    width = max(img.shape[1] for img in images)
    padded = []
    for image in images:
        if image.shape[1] != width:
            image = cv2.copyMakeBorder(image, 0, 0, 0, width - image.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded.append(image)
    return np.vstack(padded)


def extract_regions(frame_bgr: np.ndarray, playback: bool = False) -> RegionBundle:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    arena_rgb, arena_params = _safe_process_part(frame_rgb, 2, playback=playback, resize=True, verbose=True)
    hand_rgb = _safe_process_part(frame_rgb, 3, resize=True)
    timer_rgb = _safe_process_part(frame_rgb, 1, resize=True)
    center_parts = _safe_process_part(frame_rgb, 4, resize=False, allow_missing=True)

    x1, y1, x2, y2 = _compute_abs_bbox(frame_rgb, arena_params)
    elixir_rgb = extract_bbox(hand_rgb, *part3_elixir_params)
    center_text_rgb = None
    if isinstance(center_parts, dict) and center_parts:
        center_text_rgb = _stack_vertical([cv2.cvtColor(v, cv2.COLOR_RGB2BGR) for v in center_parts.values()])

    return RegionBundle(
        arena_for_model_bgr=cv2.cvtColor(arena_rgb, cv2.COLOR_RGB2BGR),
        arena_mapping=ArenaMapping(source_xyxy=(x1, y1, x2, y2), model_shape=arena_rgb.shape[:2]),
        arena_debug_bgr=cv2.cvtColor(arena_rgb, cv2.COLOR_RGB2BGR),
        hand_cards_bgr=cv2.cvtColor(hand_rgb, cv2.COLOR_RGB2BGR),
        elixir_bgr=cv2.cvtColor(elixir_rgb, cv2.COLOR_RGB2BGR),
        timer_hp_bgr=cv2.cvtColor(timer_rgb, cv2.COLOR_RGB2BGR),
        center_text_bgr=center_text_rgb,
    )
