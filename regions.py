from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from config import ensure_katacr_environment

ensure_katacr_environment()

from katacr.build_dataset.constant import part3_elixir_params  # noqa: E402
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


def _safe_process_part(image_rgb: np.ndarray, part: int | str, playback: bool = False, resize: bool = True, verbose: bool = False) -> Any:
    return process_part(image_rgb, part, playback=playback, resize=resize, verbose=verbose)


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
    center_parts = _safe_process_part(frame_rgb, 4, resize=False)

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
