from typing import Optional

import cv2
import numpy as np
import pytesseract

from .capture_config import (
    ELIXIR_BAR_X_LOWER_NORM,
    ELIXIR_BAR_X_UPPER_NORM,
    ELIXIR_BAR_Y_LOWER_NORM,
    ELIXIR_BAR_Y_UPPER_NORM,
    ENEMY_KING_TOWER_HEALTH_X_LOWER_NORM,
    ENEMY_KING_TOWER_HEALTH_X_UPPER_NORM,
    ENEMY_KING_TOWER_HEALTH_Y_LOWER_NORM,
    ENEMY_KING_TOWER_HEALTH_Y_UPPER_NORM,
    ENEMY_LEFT_TOWER_HEALTH_X_LOWER_NORM,
    ENEMY_LEFT_TOWER_HEALTH_X_UPPER_NORM,
    ENEMY_LEFT_TOWER_HEALTH_Y_LOWER_NORM,
    ENEMY_LEFT_TOWER_HEALTH_Y_UPPER_NORM,
    ENEMY_RIGHT_TOWER_HEALTH_X_LOWER_NORM,
    ENEMY_RIGHT_TOWER_HEALTH_X_UPPER_NORM,
    ENEMY_RIGHT_TOWER_HEALTH_Y_LOWER_NORM,
    ENEMY_RIGHT_TOWER_HEALTH_Y_UPPER_NORM,
    FRIENDLY_KING_TOWER_HEALTH_X_LOWER_NORM,
    FRIENDLY_KING_TOWER_HEALTH_X_UPPER_NORM,
    FRIENDLY_KING_TOWER_HEALTH_Y_LOWER_NORM,
    FRIENDLY_KING_TOWER_HEALTH_Y_UPPER_NORM,
    FRIENDLY_LEFT_TOWER_HEALTH_X_LOWER_NORM,
    FRIENDLY_LEFT_TOWER_HEALTH_X_UPPER_NORM,
    FRIENDLY_LEFT_TOWER_HEALTH_Y_LOWER_NORM,
    FRIENDLY_LEFT_TOWER_HEALTH_Y_UPPER_NORM,
    FRIENDLY_RIGHT_TOWER_HEALTH_X_LOWER_NORM,
    FRIENDLY_RIGHT_TOWER_HEALTH_X_UPPER_NORM,
    FRIENDLY_RIGHT_TOWER_HEALTH_Y_LOWER_NORM,
    FRIENDLY_RIGHT_TOWER_HEALTH_Y_UPPER_NORM,
    KING_TOWER_FULL_HP,
    PRINCESS_TOWER_FULL_HP,
)


TOWER_BOXES = (
    ("enemy_left_tower", ENEMY_LEFT_TOWER_HEALTH_X_LOWER_NORM, ENEMY_LEFT_TOWER_HEALTH_X_UPPER_NORM, ENEMY_LEFT_TOWER_HEALTH_Y_LOWER_NORM, ENEMY_LEFT_TOWER_HEALTH_Y_UPPER_NORM),
    ("enemy_king_tower", ENEMY_KING_TOWER_HEALTH_X_LOWER_NORM, ENEMY_KING_TOWER_HEALTH_X_UPPER_NORM, ENEMY_KING_TOWER_HEALTH_Y_LOWER_NORM, ENEMY_KING_TOWER_HEALTH_Y_UPPER_NORM),
    ("enemy_right_tower", ENEMY_RIGHT_TOWER_HEALTH_X_LOWER_NORM, ENEMY_RIGHT_TOWER_HEALTH_X_UPPER_NORM, ENEMY_RIGHT_TOWER_HEALTH_Y_LOWER_NORM, ENEMY_RIGHT_TOWER_HEALTH_Y_UPPER_NORM),
    ("friendly_left_tower", FRIENDLY_LEFT_TOWER_HEALTH_X_LOWER_NORM, FRIENDLY_LEFT_TOWER_HEALTH_X_UPPER_NORM, FRIENDLY_LEFT_TOWER_HEALTH_Y_LOWER_NORM, FRIENDLY_LEFT_TOWER_HEALTH_Y_UPPER_NORM),
    ("friendly_king_tower", FRIENDLY_KING_TOWER_HEALTH_X_LOWER_NORM, FRIENDLY_KING_TOWER_HEALTH_X_UPPER_NORM, FRIENDLY_KING_TOWER_HEALTH_Y_LOWER_NORM, FRIENDLY_KING_TOWER_HEALTH_Y_UPPER_NORM),
    ("friendly_right_tower", FRIENDLY_RIGHT_TOWER_HEALTH_X_LOWER_NORM, FRIENDLY_RIGHT_TOWER_HEALTH_X_UPPER_NORM, FRIENDLY_RIGHT_TOWER_HEALTH_Y_LOWER_NORM, FRIENDLY_RIGHT_TOWER_HEALTH_Y_UPPER_NORM),
)


def _crop_norm(frame: np.ndarray, x1_norm: float, x2_norm: float, y1_norm: float, y2_norm: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = max(0, int(round(x1_norm * w)))
    x2 = min(w, int(round(x2_norm * w)))
    y1 = max(0, int(round(y1_norm * h)))
    y2 = min(h, int(round(y2_norm * h)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2].copy()


def _tesseract_digits(img: np.ndarray, psm: int) -> tuple[Optional[int], str, float]:
    config = (
        f"--oem 3 --psm {psm} "
        "-c tessedit_char_whitelist=0123456789 "
        "-c load_system_dawg=0 -c load_freq_dawg=0"
    )
    data = pytesseract.image_to_data(
        img,
        lang="eng",
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    digits = []
    confs = []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        token = "".join(ch for ch in str(text) if ch.isdigit())
        if not token:
            continue
        digits.append(token)
        try:
            conf_val = float(conf)
        except Exception:
            conf_val = -1.0
        if conf_val >= 0:
            confs.append(conf_val)
    text = "".join(digits)
    value = int(text) if text else None
    mean_conf = float(sum(confs) / len(confs)) if confs else -1.0
    return value, text, mean_conf


def _preprocess_elixir_digits(strip: np.ndarray) -> list[np.ndarray]:
    if strip.size == 0:
        return []
    enlarged = cv2.resize(strip, (max(1, strip.shape[1] * 8), max(1, strip.shape[0] * 8)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    variants = []

    _, bright = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    variants.append(255 - bright)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    variants.append(255 - otsu)
    return variants


def infer_elixir_count_from_frame(frame: np.ndarray) -> dict:
    strip = _crop_norm(
        frame,
        ELIXIR_BAR_X_LOWER_NORM,
        ELIXIR_BAR_X_UPPER_NORM,
        ELIXIR_BAR_Y_LOWER_NORM,
        ELIXIR_BAR_Y_UPPER_NORM,
    )
    candidates = []
    for variant in _preprocess_elixir_digits(strip):
        value, text, conf = _tesseract_digits(variant, psm=7)
        if text and value is not None and 0 <= value <= 10:
            candidates.append((value, text, conf))

    best_value = None
    best_text = ""
    best_conf = -1.0
    if candidates:
        candidates.sort(key=lambda item: (len(item[1]), item[2]), reverse=True)
        best_value, best_text, best_conf = candidates[0]

    return {
        "count_estimate": best_value,
        "model_count_estimate": best_value,
        "detector_count_estimate": None,
        "strip_bbox_norm": {
            "x1": round(ELIXIR_BAR_X_LOWER_NORM, 4),
            "x2": round(ELIXIR_BAR_X_UPPER_NORM, 4),
            "y1": round(ELIXIR_BAR_Y_LOWER_NORM, 4),
            "y2": round(ELIXIR_BAR_Y_UPPER_NORM, 4),
        },
        "ocr_text": best_text,
        "ocr_confidence": round(best_conf, 2) if best_conf >= 0 else None,
        "notes": [
            "Primary estimate uses pytesseract on the configured elixir HUD crop.",
        ],
    }


def _preprocess_tower_digits(crop: np.ndarray) -> list[np.ndarray]:
    if crop.size == 0:
        return []
    enlarged = cv2.resize(crop, (max(1, crop.shape[1] * 5), max(1, crop.shape[0] * 5)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    variants = []

    _, bright = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    variants.append(255 - bright)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    variants.append(255 - otsu)

    hsv = cv2.cvtColor(enlarged, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 140), (180, 90, 255))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    variants.append(255 - white_mask)
    return variants


def _detect_tower_value(frame: np.ndarray, box: tuple) -> dict:
    name, x1, x2, y1, y2 = box
    crop = _crop_norm(frame, x1, x2, y1, y2)
    candidates = []
    for variant in _preprocess_tower_digits(crop):
        value, text, conf = _tesseract_digits(variant, psm=7)
        if text:
            candidates.append((value, text, conf))
    if candidates:
        candidates.sort(key=lambda item: ((len(item[1]) >= 4), len(item[1]), item[2]), reverse=True)
        value, text, conf = candidates[0]
    else:
        value, text, conf = None, "", -1.0
    return {
        "value": value,
        "text": text,
        "ocr_confidence": round(conf, 2) if conf >= 0 else None,
        "bbox_norm": {"x1": x1, "x2": x2, "y1": y1, "y2": y2},
    }


def infer_tower_health_from_frame(frame: np.ndarray) -> dict:
    raw = {name: _detect_tower_value(frame, box) for name, *box in TOWER_BOXES}
    resolved = {}

    for side in ("enemy", "friendly"):
        left = raw[f"{side}_left_tower"]["value"]
        right = raw[f"{side}_right_tower"]["value"]
        king = raw[f"{side}_king_tower"]["value"]

        resolved[f"{side}_left_tower"] = {
            **raw[f"{side}_left_tower"],
            "value": 0 if left is None else left,
            "status": "ocr_missing_assumed_dead" if left is None else "ocr_detected",
            "max_hp": PRINCESS_TOWER_FULL_HP,
        }
        resolved[f"{side}_right_tower"] = {
            **raw[f"{side}_right_tower"],
            "value": 0 if right is None else right,
            "status": "ocr_missing_assumed_dead" if right is None else "ocr_detected",
            "max_hp": PRINCESS_TOWER_FULL_HP,
        }

        if left is not None and right is not None:
            king_value = KING_TOWER_FULL_HP
            king_status = "assumed_full_hp_from_both_princess_towers_visible"
        elif king is None:
            king_value = 0
            king_status = "ocr_missing_assumed_dead"
        else:
            king_value = king
            king_status = "ocr_detected"

        resolved[f"{side}_king_tower"] = {
            **raw[f"{side}_king_tower"],
            "value": king_value,
            "status": king_status,
            "max_hp": KING_TOWER_FULL_HP,
        }

    return {
        "towers": resolved,
        "notes": [
            "Tower health uses hardcoded OCR crops with pytesseract.",
            "If both princess towers on a side OCR successfully, that side's king is forced to full HP.",
            "If a tower OCR crop has no number, it is treated as dead for now.",
        ],
    }
