from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from detector_adapter import Detection
from regions import ArenaMapping, RegionBundle


TEAM_COLORS = {
    0: (255, 120, 60),
    1: (80, 160, 255),
}


def _label_color(detection: Detection, show_belong: bool) -> tuple[int, int, int]:
    if show_belong and detection.belong in TEAM_COLORS:
        return TEAM_COLORS[detection.belong]
    base = (
        (37 * (detection.class_id + 3)) % 255,
        (91 * (detection.class_id + 11)) % 255,
        (171 * (detection.class_id + 17)) % 255,
    )
    return int(base[0]), int(base[1]), int(base[2])


def project_detection_to_frame(detection: Detection, mapping: ArenaMapping) -> Detection:
    x1, y1, x2, y2 = mapping.source_xyxy
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    model_h, model_w = mapping.model_shape
    sx = crop_w / max(1, model_w)
    sy = crop_h / max(1, model_h)
    ax1, ay1, ax2, ay2 = detection.xyxy
    return Detection(
        xyxy=(x1 + ax1 * sx, y1 + ay1 * sy, x1 + ax2 * sx, y1 + ay2 * sy),
        conf=detection.conf,
        class_id=detection.class_id,
        class_name=detection.class_name,
        belong=detection.belong,
        track_id=detection.track_id,
    )


def draw_detections(
    frame_bgr: np.ndarray,
    detections: Iterable[Detection],
    *,
    show_labels: bool,
    show_conf: bool,
    show_belong: bool,
    fps_text: Optional[str] = None,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    for det in detections:
        color = _label_color(det, show_belong)
        x1, y1, x2, y2 = [int(round(v)) for v in det.xyxy]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if show_labels:
            label = det.class_name
            if show_belong and det.belong is not None:
                label = f"{label} team={det.belong}"
            if show_conf:
                label = f"{label} {det.conf:.2f}"
            if det.track_id is not None:
                label = f"id={det.track_id} {label}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            box_top = max(th + 8, y1)
            cv2.rectangle(canvas, (x1, box_top - th - 8), (x1 + tw + 8, box_top), color, -1)
            cv2.putText(canvas, label, (x1 + 4, box_top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    if fps_text:
        cv2.putText(canvas, fps_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return canvas


def _panel(image: np.ndarray, title: str, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(image, (width, height))
    cv2.rectangle(resized, (0, 0), (width - 1, 24), (25, 25, 25), -1)
    cv2.putText(resized, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return resized


def compose_debug_view(main_bgr: np.ndarray, regions: RegionBundle, include_panels: bool) -> np.ndarray:
    if not include_panels:
        return main_bgr
    panel_width = 280
    panel_height = 150
    panels = [
        _panel(regions.arena_debug_bgr, "Arena", panel_width, panel_height),
        _panel(regions.hand_cards_bgr, "Hand Cards", panel_width, panel_height),
        _panel(regions.elixir_bgr, "Elixir", panel_width, panel_height),
        _panel(regions.timer_hp_bgr, "Timer / HP", panel_width, panel_height),
    ]
    if regions.center_text_bgr is not None:
        panels.append(_panel(regions.center_text_bgr, "Center Text", panel_width, panel_height))
    stack = np.vstack(panels)
    if stack.shape[0] < main_bgr.shape[0]:
        pad = main_bgr.shape[0] - stack.shape[0]
        stack = cv2.copyMakeBorder(stack, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(15, 15, 15))
    elif stack.shape[0] > main_bgr.shape[0]:
        main_bgr = cv2.copyMakeBorder(main_bgr, 0, stack.shape[0] - main_bgr.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(15, 15, 15))
    return np.hstack([main_bgr, stack])
