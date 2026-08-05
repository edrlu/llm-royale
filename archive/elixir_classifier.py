#!/usr/bin/env python3
"""
Lightweight OCR-style elixir counter for the cropped HUD strip.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


def preprocess_digit_strip(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def find_digit_boxes(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, _ = binary.shape[:2]
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 8:
            continue
        if h < height * 0.35:
            continue
        if w < 2:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])

    merged = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        px, py, pw, ph = merged[-1]
        x, y, w, h = box
        gap = x - (px + pw)
        if gap <= 1:
            nx1 = min(px, x)
            ny1 = min(py, y)
            nx2 = max(px + pw, x + w)
            ny2 = max(py + ph, y + h)
            merged[-1] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
        else:
            merged.append(box)
    return merged


def count_holes(digit_img: np.ndarray) -> int:
    inv = 255 - digit_img
    _, hierarchy = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    holes = 0
    for i in range(len(hierarchy[0])):
        parent = hierarchy[0][i][3]
        if parent != -1:
            holes += 1
    return holes


def normalize_digit(digit_img: np.ndarray, out_size=(20, 28)) -> np.ndarray:
    ys, xs = np.where(digit_img > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((out_size[1], out_size[0]), dtype=np.uint8)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop = digit_img[y1:y2 + 1, x1:x2 + 1]

    target_w, target_h = out_size
    h, w = crop.shape[:2]
    scale = min((target_w - 4) / max(w, 1), (target_h - 4) / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    ox = (target_w - new_w) // 2
    oy = (target_h - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas


def classify_digit_shape(digit_img: np.ndarray) -> Optional[int]:
    norm = normalize_digit(digit_img)
    h, w = norm.shape[:2]

    ys, xs = np.where(norm > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1

    aspect = bw / max(bh, 1)
    holes = count_holes(norm)

    row_sum = np.sum(norm > 0, axis=1)
    left_mass = np.sum(norm[:, :w // 2] > 0)
    right_mass = np.sum(norm[:, w // 2:] > 0)
    top_mass = np.sum(norm[:h // 2, :] > 0)
    bottom_mass = np.sum(norm[h // 2:, :] > 0)

    if aspect < 0.45:
        return 1

    if holes == 1:
        lr_ratio = left_mass / max(right_mass, 1)
        tb_ratio = top_mass / max(bottom_mass, 1)
        if 0.75 <= lr_ratio <= 1.25 and 0.75 <= tb_ratio <= 1.25:
            return 0
        if top_mass > bottom_mass * 1.15:
            return 9
        return 6

    if holes >= 2:
        return 8

    center_row = row_sum[h // 2]
    top_row = row_sum[h // 4]
    bottom_row = row_sum[(3 * h) // 4]

    if top_row > bottom_row * 1.4:
        return 7
    if center_row > np.mean(row_sum) * 1.2 and right_mass > left_mass * 1.1:
        return 4

    upper = norm[:h // 2, :]
    lower = norm[h // 2:, :]
    upper_left = np.sum(upper[:, :w // 2] > 0)
    upper_right = np.sum(upper[:, w // 2:] > 0)
    lower_left = np.sum(lower[:, :w // 2] > 0)
    lower_right = np.sum(lower[:, w // 2:] > 0)

    if upper_left > upper_right and lower_right > lower_left:
        return 5
    if right_mass > left_mass * 1.2:
        return 3
    return 2


def read_number_from_strip(img: np.ndarray, debug: bool = False) -> Tuple[Optional[int], dict]:
    binary = preprocess_digit_strip(img)
    boxes = find_digit_boxes(binary)

    digits = []
    debug_img = img.copy()
    for (x, y, w, h) in boxes:
        digit_crop = binary[y:y + h, x:x + w]
        pred = classify_digit_shape(digit_crop)
        if pred is None:
            continue
        digits.append((x, pred))
        if debug:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(
                debug_img,
                str(pred),
                (x, max(0, y - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    digits.sort(key=lambda t: t[0])
    text = "".join(str(d) for _, d in digits)
    value = int(text) if text.isdigit() else None

    return value, {
        "text": text,
        "boxes": boxes,
        "binary": binary,
        "debug_img": debug_img if debug else None,
    }
