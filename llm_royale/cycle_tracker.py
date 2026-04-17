#!/usr/bin/env python3
"""
Hand-card classification helpers and deterministic cycle tracking.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import cv2
import numpy as np

from . import REPO_ROOT
from .capture_config import (
    HAND_CARD_Y_LOWER_NORM,
    HAND_CARD_Y_UPPER_NORM,
    HAND_SLOT_CENTERS,
)


DECK_26 = [
    "hog-rider",
    "musketeer",
    "ice-spirit",
    "ice-golem",
    "fireball",
    "the-log",
    "cannon",
    "skeleton",
]

CARD_REFERENCE_DIR = os.path.join(REPO_ROOT, "cards")
CARD_FILENAME_TO_LABEL = {
    "hog_rider": "hog-rider",
    "musk": "musketeer",
    "musketeer": "musketeer",
    "ice_golem": "ice-golem",
    "ice_spirit": "ice-spirit",
    "fireball": "fireball",
    "log": "the-log",
    "the_log": "the-log",
    "cannon": "cannon",
    "skeletons": "skeleton",
    "skeleton": "skeleton",
}

REF_SIZE = (128, 128)
ORB_FEATURES = 500
INNER_CROP = 0.10
MIN_COMBINED_SCORE = 7.5
MIN_MARGIN_SCORE = 1.0


def crop_hand_slots(frame: np.ndarray) -> List[dict]:
    h, w = frame.shape[:2]
    slot_centers = HAND_SLOT_CENTERS
    y1 = int(h * HAND_CARD_Y_LOWER_NORM)
    y2 = int(h * HAND_CARD_Y_UPPER_NORM)
    half_width = int(w * 0.072)

    slots = []
    for idx, center_norm in enumerate(slot_centers, start=1):
        cx = int(round(center_norm * w))
        x1 = max(0, cx - half_width)
        x2 = min(w, cx + half_width)
        crop = frame[y1:y2, x1:x2].copy()
        slots.append({
            "slot": idx,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "image": crop,
        })
    return slots


def _normalize_ref_label(filename_stem: str) -> Optional[str]:
    key = filename_stem.strip().lower().replace("-", "_").replace(" ", "_")
    return CARD_FILENAME_TO_LABEL.get(key)


def _preprocess_card(img: np.ndarray, size=REF_SIZE) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    x1, y1 = int(w * INNER_CROP), int(h * INNER_CROP)
    x2, y2 = int(w * (1.0 - INNER_CROP)), int(h * (1.0 - INNER_CROP))
    return img[y1:y2, x1:x2]


def _compute_orb_features(img: np.ndarray, orb: cv2.ORB):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return orb.detectAndCompute(gray, None)


def _histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


class CardSlotClassifier:
    """
    Fixed-crop hand classifier backed by ORB + histogram matching against the
    curated reference images in `cards/`.
    """

    def __init__(self, reference_dir: str = CARD_REFERENCE_DIR):
        self.reference_dir = reference_dir
        self.labels = list(DECK_26)
        self.enabled = False
        self.error = None
        self.image_size = REF_SIZE
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.refs = {}
        self._load()

    def _load(self) -> None:
        if not os.path.isdir(self.reference_dir):
            self.error = f"reference dir missing: {self.reference_dir}"
            return

        refs = {}
        for filename in sorted(os.listdir(self.reference_dir)):
            path = os.path.join(self.reference_dir, filename)
            if not os.path.isfile(path):
                continue
            stem, ext = os.path.splitext(filename)
            if ext.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            label = _normalize_ref_label(stem)
            if not label:
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            proc = _preprocess_card(img, size=self.image_size)
            keypoints, descriptors = _compute_orb_features(proc, self.orb)
            refs[label] = {
                "image": proc,
                "keypoints": keypoints,
                "descriptors": descriptors,
            }

        if not refs:
            self.error = f"no usable card references found in {self.reference_dir}"
            return

        missing = [label for label in self.labels if label not in refs]
        if missing:
            self.error = f"missing reference cards: {', '.join(missing)}"
            return

        self.refs = refs
        self.enabled = True

    def classify(self, image_bgr: np.ndarray) -> Optional[dict]:
        if not self.enabled:
            return None

        query_proc = _preprocess_card(image_bgr, size=self.image_size)
        _, query_des = _compute_orb_features(query_proc, self.orb)

        results = []
        best_label = None
        best_score = -1e9

        for label, ref_data in self.refs.items():
            ref_des = ref_data["descriptors"]
            orb_score = 0.0
            if query_des is not None and ref_des is not None and len(query_des) > 0 and len(ref_des) > 0:
                matches = self.matcher.knnMatch(query_des, ref_des, k=2)
                good = []
                for pair in matches:
                    if len(pair) < 2:
                        continue
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                orb_score = float(len(good))

            hist_score = _histogram_similarity(query_proc, ref_data["image"])
            combined_score = orb_score + 20.0 * hist_score
            results.append({
                "label": label,
                "orb_score": round(orb_score, 4),
                "hist_score": round(hist_score, 4),
                "combined_score": round(combined_score, 4),
            })
            if combined_score > best_score:
                best_score = combined_score
                best_label = label

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        top1 = results[0]
        top2 = results[1] if len(results) > 1 else None
        margin = top1["combined_score"] - (top2["combined_score"] if top2 else 0.0)

        accepted = (
            top1["combined_score"] >= MIN_COMBINED_SCORE
            and margin >= MIN_MARGIN_SCORE
        )
        confidence = max(0.0, min(0.9999, margin / max(top1["combined_score"], 1.0)))

        return {
            "label": best_label if accepted else None,
            "raw_label": best_label if accepted else None,
            "confidence": confidence if accepted else 0.0,
            "margin": round(margin, 4),
            "best_score": round(top1["combined_score"], 4),
            "scores": {item["label"]: item["combined_score"] for item in results},
            "top_matches": results[:5],
        }


@dataclass
class CycleTracker:
    deck: List[str] = field(default_factory=lambda: list(DECK_26))
    current_hand: List[Optional[str]] = field(default_factory=lambda: [None, None, None, None])
    stable_hand: List[Optional[str]] = field(default_factory=lambda: [None, None, None, None])
    played_history: Deque[str] = field(default_factory=lambda: deque(maxlen=16))
    inferred_draw_history: Deque[str] = field(default_factory=lambda: deque(maxlen=16))
    confidence_memory: List[dict] = field(default_factory=lambda: [{} for _ in range(4)])
    min_stable_frames: int = 2
    _pending_labels: List[Optional[str]] = field(default_factory=lambda: [None, None, None, None])
    _pending_streak: List[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def observe_hand(self, slots: List[dict]) -> None:
        labels = []
        for i, slot in enumerate(slots):
            label = slot.get("label")
            conf = float(slot.get("confidence") or 0.0)
            mem = self.confidence_memory[i]
            if label:
                mem[label] = mem.get(label, 0.0) * 0.6 + conf
            for key in list(mem.keys()):
                if key != label:
                    mem[key] *= 0.9
                    if mem[key] < 0.05:
                        mem.pop(key, None)
            best_label = None
            best_score = 0.0
            for key, value in mem.items():
                if value > best_score:
                    best_label = key
                    best_score = value
            labels.append(best_label if best_score >= 0.15 else None)

        accepted = list(self.stable_hand)
        for i, label in enumerate(labels):
            if label == self._pending_labels[i]:
                self._pending_streak[i] += 1
            else:
                self._pending_labels[i] = label
                self._pending_streak[i] = 1
            if self._pending_streak[i] >= self.min_stable_frames:
                accepted[i] = label

        previous = [x for x in self.stable_hand if x]
        self.current_hand = labels
        self.stable_hand = accepted
        current = [x for x in accepted if x]
        new_cards = [x for x in current if x not in previous]
        for card in new_cards:
            self.inferred_draw_history.append(card)

    def record_play(self, card: str) -> None:
        self.played_history.append(card)

    def state(self) -> dict:
        known = [c for c in self.stable_hand if c]
        missing = [c for c in self.deck if c not in known]
        next_candidates = [
            c for c in list(self.inferred_draw_history)[-4:]
            if c not in known
        ]
        return {
            "deck": list(self.deck),
            "hand": list(self.stable_hand),
            "known_hand_cards": known,
            "missing_from_hand": missing,
            "played_history": list(self.played_history),
            "observed_draw_history": list(self.inferred_draw_history),
            "next_cycle_candidates": next_candidates,
            "notes": [
                "Cycle tracking is deterministic memory over the fixed 2.6 Hog deck.",
                "It becomes more reliable once hand-slot classification is stable over time.",
            ],
        }
