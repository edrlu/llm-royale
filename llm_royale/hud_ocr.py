from typing import Optional

import cv2
import numpy as np
import pytesseract

from .capture_config import (
    ELIXIR_BAR_X_LOWER_NORM,
    ELIXIR_BAR_X_UPPER_NORM,
    ELIXIR_BAR_Y_LOWER_NORM,
    ELIXIR_BAR_Y_UPPER_NORM,
    ELIXIR_PIP_COUNT,
    ELIXIR_PIP_X_LEFT_NORM,
    ELIXIR_PIP_X_RIGHT_NORM,
    ELIXIR_PIP_Y_LOWER_NORM,
    ELIXIR_PIP_Y_UPPER_NORM,
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


def count_elixir_pips(frame: np.ndarray) -> dict:
    """Count filled segments on the elixir bar.

    Each of the ten segments is either bright pink (filled) or dark blue
    (empty), which is a far stronger signal than the digit above it. Two classes
    of pixel are excluded before the vote: near-white ones, because the count
    digit and the "Max" caption are drawn over the left of the bar, and very
    dark ones, which are the segment dividers and glyph outlines.
    """
    band = _crop_norm(
        frame,
        ELIXIR_PIP_X_LEFT_NORM,
        ELIXIR_PIP_X_RIGHT_NORM,
        ELIXIR_PIP_Y_LOWER_NORM,
        ELIXIR_PIP_Y_UPPER_NORM,
    )
    if band.size == 0:
        return {"count": None, "fractions": [], "notes": ["empty elixir pip band"]}

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Magenta sits near H=150 in OpenCV's 0-179 hue range.
    pink = (hue >= 135) & (hue <= 175) & (sat >= 90) & (val >= 90)
    countable = (sat >= 60) & (val >= 50)

    width = band.shape[1]
    fractions = []
    for index in range(ELIXIR_PIP_COUNT):
        x1 = int(round(index * width / ELIXIR_PIP_COUNT))
        x2 = int(round((index + 1) * width / ELIXIR_PIP_COUNT))
        # Trim the segment edges: the dividers between pips are dark and would
        # otherwise drag a filled segment's ratio down.
        margin = max(1, (x2 - x1) // 6)
        x1, x2 = x1 + margin, max(x1 + margin + 1, x2 - margin)
        total = int(countable[:, x1:x2].sum())
        filled = int((pink & countable)[:, x1:x2].sum())
        fractions.append(round(filled / total, 3) if total else 0.0)

    # Elixir fills left to right, so the count is the length of the leading run
    # of filled segments — not the total, which a stray highlight could inflate.
    count = 0
    for fraction in fractions:
        if fraction < 0.5:
            break
        count += 1

    return {"count": count, "fractions": fractions, "notes": []}


def infer_elixir_count_from_frame(frame: np.ndarray) -> dict:
    pips = count_elixir_pips(frame)

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
        "count_estimate": pips["count"],
        "model_count_estimate": pips["count"],
        "detector_count_estimate": None,
        "pip_fractions": pips["fractions"],
        "ocr_digit_estimate": best_value,
        "strip_bbox_norm": {
            "x1": round(ELIXIR_BAR_X_LOWER_NORM, 4),
            "x2": round(ELIXIR_BAR_X_UPPER_NORM, 4),
            "y1": round(ELIXIR_BAR_Y_LOWER_NORM, 4),
            "y2": round(ELIXIR_BAR_Y_UPPER_NORM, 4),
        },
        "ocr_text": best_text,
        "ocr_confidence": round(best_conf, 2) if best_conf >= 0 else None,
        "notes": [
            "Count comes from filled elixir bar segments; the OCR digit is a cross-check only.",
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


def _bar_is_friendly(crop: np.ndarray) -> bool:
    """True for a blue (friendly) bar, False for a red/pink (enemy) one."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = ((hue <= 10) | (hue >= 165)) & (sat >= 110) & (val >= 90)
    blue = (hue >= 95) & (hue <= 125) & (sat >= 110) & (val >= 90)
    return int(blue.sum()) > int(red.sum())


def _preprocess_blue_bar_digits(crop: np.ndarray) -> list[np.ndarray]:
    """Variants for the friendly bars.

    Friendly HP numbers are pale grey on a light blue fill, so the thresholds
    that isolate white-on-pink enemy digits either swallow the digits into the
    bar or eat them entirely. A high luminance cut and a low-saturation mask
    both survive that contrast; the generic variants do not.
    """
    if crop.size == 0:
        return []
    enlarged = cv2.resize(crop, (max(1, crop.shape[1] * 6), max(1, crop.shape[0] * 6)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    hsv = cv2.cvtColor(enlarged, cv2.COLOR_BGR2HSV)

    _, bright = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    low_sat = cv2.inRange(hsv, (0, 0, 120), (180, 120, 255))
    return [255 - bright, 255 - low_sat]


def _detect_tower_value(frame: np.ndarray, box: tuple) -> dict:
    x1, x2, y1, y2 = box
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


# The tower-bar detection box starts with the level crown badge, which OCRs as
# a stray leading digit. Dropping this fraction of the box width removes it.
BAR_CROWN_FRACTION = 0.22

TOWER_BAR_LABELS = {"tower-bar", "dagger-duchess-tower-bar"}
KING_BAR_LABELS = {"king-tower-bar"}

# Where each tower's HP bar sits, in normalized frame coordinates. Troop HP bars
# and the occasional misdetection (a card-name banner reads as a bar) land in
# mid-arena, so a bar is only accepted for a slot if it shows up near where that
# tower actually is. The tolerance is wide enough to absorb the camera pan.
EXPECTED_BAR_CENTERS = {
    "enemy_left_tower": (0.22, 0.233),
    "enemy_right_tower": (0.78, 0.233),
    "enemy_king_tower": (0.50, 0.165),
    "friendly_left_tower": (0.22, 0.644),
    "friendly_right_tower": (0.78, 0.644),
    "friendly_king_tower": (0.50, 0.790),
}
BAR_MATCH_MAX_DISTANCE = 0.09


def _ocr_bar_value(frame: np.ndarray, bbox: dict, max_hp: int) -> dict:
    """Read the HP number out of one detected tower bar.

    Several preprocessing variants and page-segmentation modes are tried and the
    readings are voted on, because any single pass mangles a digit now and then.
    Readings above the tower's maximum HP are dropped outright: those are the
    passes that glued the crown badge or a neighbouring glyph onto the number.
    """
    x1 = bbox["x1"] + int(BAR_CROWN_FRACTION * (bbox["x2"] - bbox["x1"]))
    crop = frame[max(0, bbox["y1"]):bbox["y2"], max(0, x1):bbox["x2"]]
    if crop.size == 0:
        return {"value": None, "text": "", "ocr_confidence": None}

    variants = (
        _preprocess_blue_bar_digits(crop)
        if _bar_is_friendly(crop)
        else _preprocess_tower_digits(crop)
    )

    votes = {}
    for variant in variants:
        for psm in (7, 8, 13):
            value, text, conf = _tesseract_digits(variant, psm=psm)
            if value is None or not text or value > max_hp:
                continue
            entry = votes.setdefault(text, {"count": 0, "conf": -1.0, "value": value})
            entry["count"] += 1
            entry["conf"] = max(entry["conf"], conf)

    if not votes:
        return {"value": None, "text": "", "ocr_confidence": None}

    text, entry = max(votes.items(), key=lambda item: (item[1]["count"], item[1]["conf"]))
    return {
        "value": entry["value"],
        "text": text,
        "ocr_confidence": round(entry["conf"], 2) if entry["conf"] >= 0 else None,
    }


def infer_tower_health_from_bars(frame: np.ndarray, detections: list) -> dict:
    """Read tower HP from the detected bars rather than fixed screen crops.

    The arena camera pans and zooms — most visibly when a tower falls or at the
    end of a match — which slides the HP numbers out of any hardcoded box. The
    detector already finds every tower bar, so anchoring the OCR to those boxes
    tracks the camera for free. Returns None when no bars were detected, which
    leaves the caller on the fixed-crop path.
    """
    height, width = frame.shape[:2]
    slots = {}

    for det in detections or []:
        label = det.get("label")
        bbox = det.get("bbox")
        if not bbox or label not in (TOWER_BAR_LABELS | KING_BAR_LABELS):
            continue
        center_x = (bbox["x1"] + bbox["x2"]) / 2.0 / width
        center_y = (bbox["y1"] + bbox["y2"]) / 2.0 / height

        names = KING_BAR_LABELS if label in KING_BAR_LABELS else TOWER_BAR_LABELS
        candidates = [
            name for name in EXPECTED_BAR_CENTERS
            if name.endswith("king_tower") == (names is KING_BAR_LABELS)
        ]
        best_name = None
        best_distance = BAR_MATCH_MAX_DISTANCE
        for name in candidates:
            expected_x, expected_y = EXPECTED_BAR_CENTERS[name]
            distance = ((center_x - expected_x) ** 2 + (center_y - expected_y) ** 2) ** 0.5
            if distance < best_distance:
                best_name, best_distance = name, distance
        if best_name is None:
            continue

        # Duplicate detections of the same bar arrive from both models; keep
        # whichever sits closest to where the tower belongs.
        if best_name not in slots or best_distance < slots[best_name]["distance"]:
            slots[best_name] = {"bbox": bbox, "distance": best_distance}

    if not slots:
        return None

    raw = {}
    for name, slot in slots.items():
        max_hp = KING_TOWER_FULL_HP if name.endswith("king_tower") else PRINCESS_TOWER_FULL_HP
        reading = _ocr_bar_value(frame, slot["bbox"], max_hp)
        reading["bbox"] = slot["bbox"]
        raw[name] = reading

    return _resolve_tower_health(raw, source="detected tower bars")


def infer_tower_health_from_frame(frame: np.ndarray) -> dict:
    raw = {name: _detect_tower_value(frame, box) for name, *box in TOWER_BOXES}
    return _resolve_tower_health(raw, source="fixed screen crops")


def _resolve_tower_health(raw: dict, source: str) -> dict:
    """Turn per-tower OCR readings into resolved HP values.

    A tower whose bar could not be read is treated as destroyed, since the bar
    disappears with the tower. The one exception is a king whose two princess
    towers both read fine: kings take no damage until a princess falls, so that
    king is known to be at full HP even though its bar is hidden.
    """
    resolved = {}
    empty = {"value": None, "text": "", "ocr_confidence": None}

    for side in ("enemy", "friendly"):
        left = raw.get(f"{side}_left_tower", empty)
        right = raw.get(f"{side}_right_tower", empty)
        king = raw.get(f"{side}_king_tower", empty)

        for lane, reading in (("left", left), ("right", right)):
            resolved[f"{side}_{lane}_tower"] = {
                **reading,
                "value": 0 if reading.get("value") is None else reading["value"],
                "status": "ocr_missing_assumed_dead" if reading.get("value") is None else "ocr_detected",
                "max_hp": PRINCESS_TOWER_FULL_HP,
            }

        if left.get("value") is not None and right.get("value") is not None:
            king_value = KING_TOWER_FULL_HP
            king_status = "assumed_full_hp_from_both_princess_towers_visible"
        elif king.get("value") is None:
            king_value = 0
            king_status = "ocr_missing_assumed_dead"
        else:
            king_value = king["value"]
            king_status = "ocr_detected"

        resolved[f"{side}_king_tower"] = {
            **king,
            "value": king_value,
            "status": king_status,
            "max_hp": KING_TOWER_FULL_HP,
        }

    return {
        "towers": resolved,
        "notes": [
            f"Tower health read with pytesseract from {source}.",
            "If both princess towers on a side read successfully, that side's king is forced to full HP.",
            "A tower whose bar could not be read is treated as dead.",
        ],
        "source": source,
    }
