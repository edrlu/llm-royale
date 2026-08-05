import os
import cv2
import numpy as np
from typing import Dict, List, Tuple

# ----------------------------
# Config
# ----------------------------
REF_SIZE = (128, 128)   # standard size for all refs and query crops
ORB_FEATURES = 500

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess_card(img: np.ndarray, size: Tuple[int, int] = REF_SIZE) -> np.ndarray:
    """
    Normalize card image before comparison.
    """
    if img is None:
        raise ValueError("Input image is None")

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # Optional: crop inner region to reduce borders / frame noise
    h, w = img.shape[:2]
    x1, y1 = int(w * 0.10), int(h * 0.10)
    x2, y2 = int(w * 0.90), int(h * 0.90)
    img = img[y1:y2, x1:x2]

    return img

# ----------------------------
# ORB feature extraction
# ----------------------------
def compute_orb_features(img: np.ndarray, orb: cv2.ORB) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

# ----------------------------
# Histogram similarity fallback
# ----------------------------
def histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compare color histograms. Higher is better.
    """
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    hist1 = cv2.calcHist([hsv1], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [32, 32], [0, 180, 0, 256])

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    # Correlation: higher = more similar
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return float(score)

# ----------------------------
# Reference database builder
# ----------------------------
def load_reference_cards(ref_dir: str) -> Dict[str, dict]:
    """
    Loads reference images from a folder.
    Assumes filenames like:
      hog-rider.png
      musketeer.png
      cannon.png
    """
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    refs = {}

    for filename in os.listdir(ref_dir):
        path = os.path.join(ref_dir, filename)
        if not os.path.isfile(path):
            continue

        label, ext = os.path.splitext(filename)
        if ext.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        proc = preprocess_card(img)
        kp, des = compute_orb_features(proc, orb)

        refs[label] = {
            "image": proc,
            "keypoints": kp,
            "descriptors": des,
        }

    return refs

# ----------------------------
# Match one query crop to all refs
# ----------------------------
def classify_card(query_img: np.ndarray, refs: Dict[str, dict]) -> Tuple[str, dict]:
    """
    Returns best label and debug info.
    """
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    query_proc = preprocess_card(query_img)
    q_kp, q_des = compute_orb_features(query_proc, orb)

    best_label = None
    best_score = -1e9
    results = []

    for label, ref_data in refs.items():
        ref_img = ref_data["image"]
        ref_des = ref_data["descriptors"]

        orb_score = 0.0

        # ORB matching
        if q_des is not None and ref_des is not None and len(q_des) > 0 and len(ref_des) > 0:
            matches = bf.knnMatch(q_des, ref_des, k=2)

            good = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            # More good matches = better
            orb_score = float(len(good))

        # Histogram fallback / supplement
        hist_score = histogram_similarity(query_proc, ref_img)

        # Combined score
        # Tune this weighting based on your data
        combined_score = orb_score + 20.0 * hist_score

        results.append({
            "label": label,
            "orb_score": orb_score,
            "hist_score": hist_score,
            "combined_score": combined_score,
        })

        if combined_score > best_score:
            best_score = combined_score
            best_label = label

    results.sort(key=lambda x: x["combined_score"], reverse=True)

    debug = {
        "top_matches": results[:5],
        "best_score": best_score,
    }

    return best_label, debug

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    ref_dir = "../cards"   # folder with 8 labeled card images
    refs = load_reference_cards(ref_dir)

    # Example query card crop
    query = cv2.imread("hand_slot_4_raw.png")
    if query is None:
        raise FileNotFoundError("could not read hand_slot_2_raw.png")

    label, debug = classify_card(query, refs)

    print("Predicted label:", label)
    print("Top matches:")
    for item in debug["top_matches"]:
        print(item)