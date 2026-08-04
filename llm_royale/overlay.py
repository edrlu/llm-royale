#!/usr/bin/env python3
"""
Annotation overlay shared by the debugger and the video recorder.

Everything drawn here has to fit a 376px-wide phone frame. The previous version
dumped raw decision JSON truncated at a fixed character count, which ran off the
right edge and made the interesting part — what the bot decided — the part you
could not read. Text is measured against the frame width instead, and only the
fields worth watching are drawn.
"""

import cv2
import numpy as np


FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADER_BG = (18, 18, 18)
FRIENDLY = (0, 220, 0)
ENEMY = (40, 80, 255)
NEUTRAL = (170, 170, 170)
RIVER = (255, 255, 0)

# Card names are far too long to sit four-across on a phone-width frame.
SHORT_CARD_NAMES = {
    "hog-rider": "hog",
    "musketeer": "musk",
    "ice-spirit": "ispirit",
    "ice-golem": "igolem",
    "fireball": "fball",
    "the-log": "log",
    "cannon": "cannon",
    "skeleton": "skels",
}


def short_card(name) -> str:
    if not name:
        return "-"
    return SHORT_CARD_NAMES.get(name, str(name)[:6])


def fit_text(text: str, max_width: int, scale: float, thickness: int = 1) -> str:
    """Truncate text with an ellipsis until it fits max_width pixels."""
    if not text:
        return ""
    if cv2.getTextSize(text, FONT, scale, thickness)[0][0] <= max_width:
        return text
    trimmed = text
    while trimmed and cv2.getTextSize(trimmed + "..", FONT, scale, thickness)[0][0] > max_width:
        trimmed = trimmed[:-1]
    return trimmed + ".." if trimmed else ""


def _draw_box_label(out: np.ndarray, text: str, x: int, y: int, color, scale: float = 0.4) -> None:
    """Draw a detection label, nudged so it never runs off an edge."""
    width, height = cv2.getTextSize(text, FONT, scale, 1)[0]
    frame_h, frame_w = out.shape[:2]
    x = max(1, min(x, frame_w - width - 2))
    y = max(height + 1, min(y, frame_h - 2))
    cv2.putText(out, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def draw_detections(out: np.ndarray, detections: list, scale: float = 1.0) -> None:
    """Draw detection boxes, scaling them from detector pixels to frame pixels.

    The detector works on a downscaled frame (832px) while the recording can be
    at the window's native size, so boxes arrive in the smaller coordinate space
    and have to be scaled up to land on the right units.
    """
    thickness = max(1, round(scale))
    for det in detections or []:
        bbox = det.get("bbox") or {}
        try:
            x1, y1 = int(bbox["x1"] * scale), int(bbox["y1"] * scale)
            x2, y2 = int(bbox["x2"] * scale), int(bbox["y2"] * scale)
        except (KeyError, TypeError, ValueError):
            continue
        owner = det.get("owner")
        color = FRIENDLY if owner == "friendly" else ENEMY if owner == "enemy" else NEUTRAL
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        confidence = det.get("confidence")
        label = str(det.get("label", "?"))
        if isinstance(confidence, (int, float)):
            label = f"{label} {confidence:.2f}"
        _draw_box_label(out, label, x1, max(int(10 * scale), y1 - 3), color, scale=0.4 * scale)


def header_lines(snapshot: dict, decision, result) -> list:
    """The few fields worth reading while watching a match."""
    snapshot = snapshot or {}
    elixir = snapshot.get("elixir_inferred", {}) or {}
    value = elixir.get("blended_estimate")
    if value is None:
        value = elixir.get("count_estimate")
    if isinstance(value, (int, float)):
        elixir_text = f"{value:.1f}"
    else:
        elixir_text = "?"

    grouped = snapshot.get("grouped", {}) or {}
    towers = (snapshot.get("tower_health_inferred", {}) or {}).get("towers", {}) or {}

    def hp(name):
        tower = towers.get(name) or {}
        current, maximum = tower.get("value"), tower.get("max_hp")
        if not isinstance(current, (int, float)) or not maximum:
            return "--"
        return f"{round(100 * current / maximum):d}"

    lines = [
        f"#{snapshot.get('sequence', '-')}  elixir {elixir_text}  "
        f"units {len(grouped.get('troops', []) or [])}  {snapshot.get('_game_phase', '1x')}",
        f"them {hp('enemy_left_tower')}/{hp('enemy_right_tower')}   "
        f"you {hp('friendly_left_tower')}/{hp('friendly_right_tower')}",
    ]

    slots = (snapshot.get("hand_cards_inferred", {}) or {}).get("slots", []) or []
    if slots:
        lines.append("hand " + " ".join(short_card(slot.get("label")) for slot in slots))

    if decision:
        action = str(decision.get("action", "")).lower()
        if action == "place_card":
            x_norm, y_norm = decision.get("x_norm"), decision.get("y_norm")
            where = f" @{x_norm:.2f},{y_norm:.2f}" if isinstance(x_norm, (int, float)) and isinstance(y_norm, (int, float)) else ""
            lines.append(f"play {short_card(decision.get('card'))}{where}")
        else:
            lines.append(f"idle: {decision.get('reason', '')}")

    if result:
        status = str(result.get("status", ""))
        if status and status != "idle":
            lines.append(f"-> {status} {result.get('reason', '')}".rstrip())

    return lines


def draw_header(out: np.ndarray, lines: list, scale: float = 0.4, size_scale: float = 1.0) -> None:
    if not lines:
        return
    frame_w = out.shape[1]
    scale = scale * size_scale
    line_height = int(round(15 * size_scale))
    padding = int(round(5 * size_scale))
    height = padding * 2 + line_height * len(lines)

    # Translucent rather than solid: the top of the arena sits underneath.
    panel = out[0:height, 0:frame_w]
    cv2.addWeighted(np.full_like(panel, HEADER_BG), 0.72, panel, 0.28, 0, panel)

    y = padding + int(round(11 * size_scale))
    for line in lines:
        cv2.putText(
            out, fit_text(line, frame_w - padding * 2, scale),
            (padding, y), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
        y += line_height


def detection_scale(frame: np.ndarray, snapshot: dict) -> float:
    """Ratio between the recorded frame and the frame the detector measured on."""
    capture_width = ((snapshot or {}).get("capture") or {}).get("width")
    if not capture_width:
        return 1.0
    return frame.shape[1] / float(capture_width)


def draw_overlay(frame: np.ndarray, snapshot: dict, decision=None, result=None) -> np.ndarray:
    """Draw detections, the river line, and a fitted header onto a copy."""
    out = frame.copy()
    snapshot = snapshot or {}
    scale = detection_scale(out, snapshot)

    draw_detections(out, snapshot.get("detections"), scale)

    board_ref = snapshot.get("board_ref") or (snapshot.get("llm_summary", {}) or {}).get("board_reference") or {}
    river = board_ref.get("river_y_norm")
    if isinstance(river, (int, float)):
        y = int(round(float(river) * out.shape[0]))
        cv2.line(out, (0, y), (out.shape[1] - 1, y), RIVER, max(1, round(scale)))

    draw_header(out, header_lines(snapshot, decision, result), size_scale=scale)
    return out
