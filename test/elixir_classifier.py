import cv2
import numpy as np
from typing import List, Tuple, Optional


def preprocess_digit_strip(img: np.ndarray) -> np.ndarray:
    """
    Extract bright white digits from the strip.
    Returns a binary image with digits in white on black.
    """
    if img is None:
        raise ValueError("Input image is None")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # White digits are very bright relative to the background
    _, binary = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)

    # Clean small specks while keeping digit strokes
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def find_digit_boxes(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Find connected components that look like digits.
    Returns boxes sorted left-to-right.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = binary.shape[:2]
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Tune as needed
        if area < 8:
            continue
        if h < H * 0.35:
            continue
        if w < 2:
            continue

        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])

    # Merge very close boxes if a digit got split weirdly
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
    """
    Count enclosed black holes inside a white digit.
    Useful for distinguishing 0, 8, 6, 9.
    """
    # Invert so holes become connected components inside shape
    inv = 255 - digit_img
    contours, hierarchy = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return 0

    holes = 0
    for i in range(len(contours)):
        parent = hierarchy[0][i][3]
        if parent != -1:
            holes += 1
    return holes


def normalize_digit(digit_img: np.ndarray, out_size=(20, 28)) -> np.ndarray:
    """
    Center digit into a fixed-size canvas.
    """
    ys, xs = np.where(digit_img > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((out_size[1], out_size[0]), dtype=np.uint8)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop = digit_img[y1:y2+1, x1:x2+1]

    target_w, target_h = out_size
    h, w = crop.shape[:2]

    scale = min((target_w - 4) / max(w, 1), (target_h - 4) / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)

    ox = (target_w - new_w) // 2
    oy = (target_h - new_h) // 2
    canvas[oy:oy+new_h, ox:ox+new_w] = resized
    return canvas


def classify_digit_shape(digit_img: np.ndarray) -> Optional[int]:
    """
    Very lightweight rule-based OCR for game digits.
    Works best for 0..10 use cases.
    """
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
    fill = np.mean(norm > 0)
    holes = count_holes(norm)

    # Vertical profile
    col_sum = np.sum(norm > 0, axis=0)
    row_sum = np.sum(norm > 0, axis=1)

    left_mass = np.sum(norm[:, :w // 2] > 0)
    right_mass = np.sum(norm[:, w // 2:] > 0)
    top_mass = np.sum(norm[:h // 2, :] > 0)
    bottom_mass = np.sum(norm[h // 2:, :] > 0)

    # Extremely narrow => likely 1
    if aspect < 0.45:
        return 1

    # One hole => likely 0, 6, or 9
    if holes == 1:
        # 0 is usually more symmetric top/bottom and left/right
        lr_ratio = left_mass / max(right_mass, 1)
        tb_ratio = top_mass / max(bottom_mass, 1)

        if 0.75 <= lr_ratio <= 1.25 and 0.75 <= tb_ratio <= 1.25:
            return 0

        # 9 tends to have more top mass
        if top_mass > bottom_mass * 1.15:
            return 9

        # 6 tends to have more bottom mass
        return 6

    # Two holes => 8
    if holes >= 2:
        return 8

    # No-hole digits. For elixir use, we mostly care about 1 and maybe edge cases.
    # Add a few rough rules anyway:
    center_row = row_sum[h // 2]
    top_row = row_sum[h // 4]
    bottom_row = row_sum[(3 * h) // 4]

    # 7: stronger top bar, lighter bottom
    if top_row > bottom_row * 1.4:
        return 7

    # 2 / 3 / 5 / 4 fallback guesses
    # 4 tends to have lots of right-side mass and a strong middle
    if center_row > np.mean(row_sum) * 1.2 and right_mass > left_mass * 1.1:
        return 4

    # 5 tends to have more left mass in upper half and more right mass in lower half
    upper = norm[:h // 2, :]
    lower = norm[h // 2:, :]
    upper_left = np.sum(upper[:, :w // 2] > 0)
    upper_right = np.sum(upper[:, w // 2:] > 0)
    lower_left = np.sum(lower[:, :w // 2] > 0)
    lower_right = np.sum(lower[:, w // 2:] > 0)

    if upper_left > upper_right and lower_right > lower_left:
        return 5

    # 3 tends to bias right side
    if right_mass > left_mass * 1.2:
        return 3

    return 2


def read_number_from_strip(img: np.ndarray, debug: bool = False) -> Tuple[Optional[int], dict]:
    """
    Reads a number from a tiny game HUD strip.
    Returns (value, debug_info).
    """
    binary = preprocess_digit_strip(img)
    boxes = find_digit_boxes(binary)

    digits = []
    dbg = img.copy()

    for (x, y, w, h) in boxes:
        digit_crop = binary[y:y+h, x:x+w]
        pred = classify_digit_shape(digit_crop)
        if pred is None:
            continue

        digits.append((x, pred))

        if debug:
            cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(
                dbg, str(pred), (x, max(0, y - 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA
            )

    digits.sort(key=lambda t: t[0])
    text = "".join(str(d) for _, d in digits)

    value = int(text) if text.isdigit() else None

    return value, {
        "text": text,
        "boxes": boxes,
        "binary": binary,
        "debug_img": dbg if debug else None,
    }


if __name__ == "__main__":
    img = cv2.imread("elixir_strip 2.png")
    value, info = read_number_from_strip(img, debug=True)

    print("Detected text:", info["text"])
    print("Detected value:", value)

    cv2.imwrite("/mnt/data/elixir_binary.png", info["binary"])
    if info["debug_img"] is not None:
        cv2.imwrite("/mnt/data/elixir_debug.png", info["debug_img"])