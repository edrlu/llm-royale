#!/usr/bin/env python3
"""
Clash Royale screen capture that writes detection snapshots to JSON.

Pipeline:
    adb screenrecord (H.264) -> ffmpeg (raw BGR frames) -> Clash detector -> JSON

Default behavior:
    - No OpenCV window rendering
    - Writes one JSON snapshot per second to clash_state.json

Usage:
    python clash_capture.py
    python clash_capture.py --preset fast
    python clash_capture.py --output-json state.json --interval-sec 0.5
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import cv2

from . import REPO_ROOT
from .capture_config import (
    PRESETS,
    CapturePreset,
    KING_TOWER_FULL_HP,
    PRINCESS_TOWER_FULL_HP,
    HAND_SLOT_CENTERS,
)
from .cycle_tracker import CardSlotClassifier, crop_hand_slots
from .match_navigator import screen_kind
from .hud_ocr import (
    TowerBarReader,
    count_elixir_pips,
    infer_elixir_count_from_frame as infer_elixir_count_from_ocr,
    infer_tower_health_from_bars,
    infer_tower_health_from_frame,
)


def find_bin(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    for prefix in ["/opt/homebrew/bin", "/usr/local/bin"]:
        candidate = os.path.join(prefix, name)
        if os.path.isfile(candidate):
            return candidate
    print(f"[ERROR] {name} not found. Install with: brew install {name}")
    sys.exit(1)


CLASH_PIPELINE_DIR = os.path.join(REPO_ROOT, "clash-yolo-pipeline")
CLASH_MODELS_DIR = os.path.join(CLASH_PIPELINE_DIR, "models")
CLASH_VENDOR_DIR = os.path.join(CLASH_PIPELINE_DIR, "vendor")
CLASH_KATACR_ROOT = os.path.join(CLASH_VENDOR_DIR, "KataCR")
TROOP_LABELS = {"hog-rider", "musketeer", "ice-spirit", "ice-golem", "skeleton"}
SPELL_LABELS = {"fireball", "the-log"}
BUILDING_LABELS = {"cannon"}
TOWER_LABELS = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower"}
BAR_LABELS = {"tower-bar", "king-tower-bar", "dagger-duchess-tower-bar", "bar", "bar-level"}
HAND_CARD_LABELS = TROOP_LABELS | SPELL_LABELS | BUILDING_LABELS


class ClashDetector:
    BELONG_LABELS = {0: "friendly", 1: "enemy", -1: "neutral"}

    _ALLOWED_CLASSES = frozenset({
        "hog-rider", "musketeer", "ice-spirit", "ice-golem",
        "fireball", "the-log", "cannon", "skeleton",
        "king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
        "tower-bar", "king-tower-bar", "dagger-duchess-tower-bar",
        "bar", "bar-level",
        "elixir", "clock",
    })

    _CATEGORY_MAP = {
        "hog-rider": "troop",
        "musketeer": "troop",
        "ice-spirit": "troop",
        "ice-golem": "troop",
        "skeleton": "troop",
        "fireball": "spell",
        "the-log": "spell",
        "cannon": "building",
        "king-tower": "tower",
        "queen-tower": "tower",
        "cannoneer-tower": "tower",
        "dagger-duchess-tower": "tower",
        "tower-bar": "bar",
        "king-tower-bar": "bar",
        "dagger-duchess-tower-bar": "bar",
        "bar": "bar",
        "bar-level": "bar",
        "elixir": "game_state",
        "clock": "game_state",
    }

    def __init__(self, preset: CapturePreset):
        dataset_path = os.path.join(CLASH_VENDOR_DIR, "Clash-Royale-Detection-Dataset")
        os.environ["KATACR_DATASET_PATH"] = dataset_path
        ultralytics_cfg = os.path.join(CLASH_PIPELINE_DIR, ".ultralytics")
        os.makedirs(ultralytics_cfg, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", ultralytics_cfg)

        if CLASH_KATACR_ROOT not in sys.path:
            sys.path.insert(0, CLASH_KATACR_ROOT)

        from katacr.constants.label_list import idx2unit, unit2idx
        from katacr.yolov8.custom_model import CRDetectionModel
        from katacr.yolov8.custom_predict import CRDetectionPredictor
        from ultralytics.engine.model import Model

        self.idx2unit = idx2unit
        self.unit2idx = unit2idx

        self._allowed_ids = frozenset(
            idx for idx, name in idx2unit.items() if name in self._ALLOWED_CLASSES
        )
        print(f"[INFO] Clash filter: {len(self._allowed_ids)} classes allowed out of {len(idx2unit)}")

        import torch
        import torchvision
        self.torch = torch
        self.torchvision = torchvision

        class _YOLOCRLive(Model):
            @property
            def task_map(self):
                return {"detect": {
                    "model": CRDetectionModel,
                    "trainer": None,
                    "validator": None,
                    "predictor": CRDetectionPredictor,
                }}

        weights = sorted(
            f for f in os.listdir(CLASH_MODELS_DIR) if f.endswith(".pt")
        )
        if not weights:
            raise FileNotFoundError(f"No .pt files found in {CLASH_MODELS_DIR}")
        weights = weights[:preset.clash_num_models]

        print(f"[INFO] Loading Clash Royale models: {weights}")
        self.models = [_YOLOCRLive(os.path.join(CLASH_MODELS_DIR, w)) for w in weights]

        self.conf_threshold = preset.clash_conf
        self.iou_threshold = preset.clash_iou
        self.device = preset.clash_device
        self.infer_size = preset.clash_infer_size

        for model in self.models:
            if preset.clash_fuse:
                model.fuse()
                print("[INFO] Model layers fused (Conv+BN merge)")
            if preset.clash_half and preset.clash_device != "cpu":
                model.model.half()
                print(f"[INFO] Model set to FP16 half precision on {preset.clash_device}")

        self._class_remaps = []
        for model in self.models:
            names = getattr(model, "names", None) or {}
            if not names:
                self._class_remaps.append(np.arange(256, dtype=np.int64))
                continue
            max_id = max(names.keys())
            remap = np.arange(max_id + 1, dtype=np.int64)
            for local_id, name in names.items():
                remap[local_id] = int(unit2idx.get(str(name), local_id))
            self._class_remaps.append(remap)

        print(
            f"[INFO] Clash detector ready ({len(self.models)} model(s), "
            f"device={self.device}, infer={self.infer_size}px, "
            f"half={preset.clash_half}, fuse={preset.clash_fuse})"
        )

    def _predict_raw(self, frame: np.ndarray) -> np.ndarray:
        return self._predict_raw_with_conf(frame, self.conf_threshold)

    def _predict_raw_with_conf(self, frame: np.ndarray, conf_threshold: float) -> np.ndarray:
        torch = self.torch

        merged_tensors = []
        for model, remap in zip(self.models, self._class_remaps):
            result = model.predict(
                frame, device=self.device,
                conf=conf_threshold, iou=self.iou_threshold,
                imgsz=self.infer_size, verbose=False,
            )[0]
            boxes = result.orig_boxes
            if boxes is None or len(boxes) == 0:
                continue
            remapped = boxes.clone()
            local_ids = remapped[:, 5].long().detach().cpu().numpy()
            clamped = np.clip(local_ids, 0, len(remap) - 1)
            remapped[:, 5] = torch.from_numpy(remap[clamped]).to(
                device=remapped.device,
                dtype=remapped.dtype,
            )
            merged_tensors.append(remapped)

        if not merged_tensors:
            return np.empty((0, 0), dtype=np.float32)

        if len(merged_tensors) == 1:
            return merged_tensors[0].detach().cpu().numpy()

        merged = torch.cat(merged_tensors, dim=0)
        merged_cpu = merged.detach().cpu()
        keep = self.torchvision.ops.nms(
            merged_cpu[:, :4], merged_cpu[:, 4], iou_threshold=self.iou_threshold
        )
        return merged_cpu[keep].numpy()

    def detect(self, frame: np.ndarray) -> list:
        raw = self._predict_raw(frame)
        frame_h, frame_w = frame.shape[:2]
        detections = []

        for row in raw:
            if len(row) < 7:
                continue
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            conf = float(row[-3])
            class_id = int(row[-2])
            belong = int(row[-1])

            if class_id not in self._allowed_ids:
                continue

            name = str(self.idx2unit.get(class_id, class_id))
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            center_x = x1 + width / 2.0
            center_y = y1 + height / 2.0

            detections.append({
                "label": name,
                "category": self._CATEGORY_MAP.get(name, "unknown"),
                "confidence": round(conf, 4),
                "owner": self.BELONG_LABELS.get(belong, "unknown"),
                "owner_id": belong,
                "bbox": {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": width, "height": height,
                },
                "center": {
                    "x": round(center_x, 2),
                    "y": round(center_y, 2),
                },
                "center_normalized": {
                    "x": round(center_x / frame_w, 4) if frame_w else 0.0,
                    "y": round(center_y / frame_h, 4) if frame_h else 0.0,
                },
                "health": None,
                "health_note": (
                    "Not derived yet. This script outputs detected bar objects, "
                    "but does not currently convert them into numeric HP."
                ),
            })

        return detections

    def detect_hand_cards(self, frame: np.ndarray) -> list:
        frame_h, frame_w = frame.shape[:2]
        y_start = int(frame_h * 0.74)
        crop = frame[y_start:, :]
        raw = self._predict_raw_with_conf(crop, conf_threshold=min(self.conf_threshold, 0.08))
        hand = []
        for row in raw:
            if len(row) < 7:
                continue
            x1, y1, x2, y2 = int(row[0]), int(row[1]) + y_start, int(row[2]), int(row[3]) + y_start
            conf = float(row[-3])
            class_id = int(row[-2])
            belong = int(row[-1])
            if class_id not in self._allowed_ids:
                continue
            name = str(self.idx2unit.get(class_id, class_id))
            if name not in HAND_CARD_LABELS:
                continue
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            hand.append({
                "label": name,
                "confidence": round(conf, 4),
                "owner": self.BELONG_LABELS.get(belong, "unknown"),
                "center": {"x": round(center_x, 2), "y": round(center_y, 2)},
                "center_normalized": {
                    "x": round(center_x / frame_w, 4) if frame_w else 0.0,
                    "y": round(center_y / frame_h, 4) if frame_h else 0.0,
                },
            })
        return hand


def atomic_write_json(path: str, payload: dict):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp_path, path)


def infer_board_reference(detections: list, frame_w: int, frame_h: int) -> dict:
    enemy_queens = [d for d in detections if d["label"] == "queen-tower" and d["owner"] == "enemy"]
    friendly_queens = [d for d in detections if d["label"] == "queen-tower" and d["owner"] == "friendly"]
    enemy_king = next((d for d in detections if d["label"] == "king-tower" and d["owner"] == "enemy"), None)
    friendly_king = next((d for d in detections if d["label"] == "king-tower" and d["owner"] == "friendly"), None)

    left_boundary = 0.4
    right_boundary = 0.6
    river_y_norm = 0.5

    queen_towers = enemy_queens + friendly_queens
    if len(queen_towers) >= 2:
        xs = sorted(d["center_normalized"]["x"] for d in queen_towers)
        if len(xs) >= 2:
            left_boundary = min(0.48, (xs[0] + 0.5) / 2.0)
            right_boundary = max(0.52, (xs[-1] + 0.5) / 2.0)

    enemy_front_y = np.mean([d["center_normalized"]["y"] for d in enemy_queens]) if enemy_queens else None
    friendly_front_y = np.mean([d["center_normalized"]["y"] for d in friendly_queens]) if friendly_queens else None
    if enemy_front_y is not None and friendly_front_y is not None:
        river_y_norm = (enemy_front_y + friendly_front_y) / 2.0
    elif enemy_king is not None and friendly_king is not None:
        river_y_norm = (
            enemy_king["center_normalized"]["y"] + friendly_king["center_normalized"]["y"]
        ) / 2.0

    return {
        "frame": {"width": frame_w, "height": frame_h},
        "lane_left_boundary_norm": round(left_boundary, 4),
        "lane_right_boundary_norm": round(right_boundary, 4),
        "river_y_norm": round(river_y_norm, 4),
        "arena_bbox_norm": {
            "x1": 0.0,
            "y1": 0.0,
            "x2": 1.0,
            "y2": 0.78,
        },
        "troop_placement_bbox_norm": {
            "x1": 0.0,
            "y1": round(river_y_norm, 4),
            "x2": 1.0,
            "y2": 0.78,
        },
        "enemy_queen_towers": [
            {"id": d.get("id"), "x": d["center"]["x"], "y": d["center"]["y"]}
            for d in enemy_queens
        ],
        "friendly_queen_towers": [
            {"id": d.get("id"), "x": d["center"]["x"], "y": d["center"]["y"]}
            for d in friendly_queens
        ],
    }


def classify_lane(x_norm: float, board_ref: dict) -> str:
    if x_norm < board_ref["lane_left_boundary_norm"]:
        return "left"
    if x_norm > board_ref["lane_right_boundary_norm"]:
        return "right"
    return "center"


def classify_side(y_norm: float, board_ref: dict) -> str:
    river_y = board_ref["river_y_norm"]
    if y_norm < river_y - 0.05:
        return "enemy_side"
    if y_norm > river_y + 0.05:
        return "friendly_side"
    return "river"


def estimate_pressure_score(det: dict) -> float:
    score = det["confidence"]
    if det["category"] == "troop":
        score += 1.0
    elif det["category"] == "building":
        score += 0.8
    elif det["category"] == "spell":
        score += 0.4
    if det["side"] == "friendly_side":
        score += 0.7 if det["owner"] == "friendly" else 1.2
    elif det["side"] == "enemy_side":
        score += 1.2 if det["owner"] == "friendly" else 0.7
    return round(score, 3)


def nearest_tower_context(item: dict, detections: list) -> dict:
    towers = [d for d in detections if d["category"] == "tower"]
    friendly = [d for d in towers if d["owner"] == "friendly"]
    enemy = [d for d in towers if d["owner"] == "enemy"]

    def nearest(pool):
        if not pool:
            return None
        best = min(
            pool,
            key=lambda d: ((d["center"]["x"] - item["center"]["x"]) ** 2 + (d["center"]["y"] - item["center"]["y"]) ** 2),
        )
        dist = ((best["center"]["x"] - item["center"]["x"]) ** 2 + (best["center"]["y"] - item["center"]["y"]) ** 2) ** 0.5
        return {
            "id": best["id"],
            "label": best["label"],
            "distance_px": round(dist, 2),
        }

    return {
        "nearest_friendly_tower": nearest(friendly),
        "nearest_enemy_tower": nearest(enemy),
    }


def enrich_detections(detections: list, frame_w: int, frame_h: int) -> tuple[list, dict]:
    enriched = []
    board_ref = infer_board_reference(detections, frame_w, frame_h)
    for idx, det in enumerate(detections):
        item = dict(det)
        x_norm = item["center_normalized"]["x"]
        y_norm = item["center_normalized"]["y"]
        item["id"] = f"d{idx + 1}"
        item["lane"] = classify_lane(x_norm, board_ref)
        item["side"] = classify_side(y_norm, board_ref)
        item["is_on_player_side"] = item["side"] == "friendly_side"
        item["is_on_opponent_side"] = item["side"] == "enemy_side"
        item["distance_from_river_norm"] = round(y_norm - board_ref["river_y_norm"], 4)
        item["pressure_score"] = estimate_pressure_score(item)
        item["health"] = None
        item["health_source"] = None
        item["related_bar_id"] = None
        item["tactical_note"] = (
            f"{item['owner']} {item['label']} on {item['side']} in the {item['lane']} lane"
        )
        enriched.append(item)
    for item in enriched:
        item.update(nearest_tower_context(item, enriched))
    return enriched, board_ref


def attach_bars(detections: list) -> None:
    bars = [d for d in detections if d["category"] == "bar"]
    units = [d for d in detections if d["category"] in {"troop", "building", "tower"}]

    for unit in units:
        ux = unit["center"]["x"]
        uy = unit["bbox"]["y1"]
        best_bar = None
        best_score = None
        for bar in bars:
            bx = bar["center"]["x"]
            by = bar["center"]["y"]
            vertical_gap = uy - by
            horizontal_gap = abs(ux - bx)
            if vertical_gap < -10:
                continue
            if vertical_gap > max(80, unit["bbox"]["height"] * 2.5):
                continue
            if horizontal_gap > max(50, unit["bbox"]["width"] * 1.5):
                continue
            score = vertical_gap + horizontal_gap * 0.35
            if best_score is None or score < best_score:
                best_score = score
                best_bar = bar
        if best_bar is not None:
            unit["related_bar_id"] = best_bar["id"]
            unit["health"] = {
                "status": "bar_detected",
                "ratio_estimate": None,
            }
            unit["health_source"] = "associated_bar_detection"


def estimate_bar_fill_ratio(frame: np.ndarray, bar: dict) -> Optional[dict]:
    bbox = bar.get("bbox", {})
    x1 = max(0, int(bbox.get("x1", 0)))
    y1 = max(0, int(bbox.get("y1", 0)))
    x2 = min(frame.shape[1], int(bbox.get("x2", 0)))
    y2 = min(frame.shape[0], int(bbox.get("y2", 0)))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, (25, 50, 60), (179, 255, 255))
    active_columns = (color_mask > 0).mean(axis=0) > 0.22
    if not np.any(active_columns):
        return {
            "ratio_estimate": 0.0,
            "filled_columns": 0,
            "total_columns": int(color_mask.shape[1]),
        }

    filled_columns = int(np.count_nonzero(active_columns))
    total_columns = int(active_columns.shape[0])
    ratio = max(0.0, min(1.0, filled_columns / float(max(1, total_columns))))
    return {
        "ratio_estimate": round(ratio, 3),
        "filled_columns": filled_columns,
        "total_columns": total_columns,
    }


def estimate_tower_health(frame: np.ndarray, detections: list) -> None:
    bar_by_id = {d["id"]: d for d in detections if d["category"] == "bar"}
    for det in detections:
        if det["category"] != "tower":
            continue
        bar_id = det.get("related_bar_id")
        if not bar_id:
            continue
        bar = bar_by_id.get(bar_id)
        if not bar:
            continue
        estimate = estimate_bar_fill_ratio(frame, bar)
        if not estimate:
            continue
        det["health"] = {
            "status": "bar_fill_estimated",
            "ratio_estimate": estimate["ratio_estimate"],
            "filled_columns": estimate["filled_columns"],
            "total_columns": estimate["total_columns"],
        }
        det["health_source"] = "associated_bar_fill_estimate"


def summarize_for_llm(detections: list, board_ref: dict) -> dict:
    friendly = [d for d in detections if d["owner"] == "friendly"]
    enemy = [d for d in detections if d["owner"] == "enemy"]
    friendly_units = [d for d in friendly if d["category"] in {"troop", "building"}]
    enemy_units = [d for d in enemy if d["category"] in {"troop", "building"}]

    lane_balance = {}
    for lane in ("left", "center", "right"):
        friendly_lane = [d for d in friendly_units if d["lane"] == lane]
        enemy_lane = [d for d in enemy_units if d["lane"] == lane]
        lane_balance[lane] = {
            "friendly_count": len(friendly_lane),
            "enemy_count": len(enemy_lane),
            "friendly_pressure": round(sum(d["pressure_score"] for d in friendly_lane), 3),
            "enemy_pressure": round(sum(d["pressure_score"] for d in enemy_lane), 3),
        }

    friendly_on_enemy_side = [
        d for d in friendly_units if d["side"] == "enemy_side"
    ]
    enemy_on_friendly_side = [
        d for d in enemy_units if d["side"] == "friendly_side"
    ]

    top_threats = sorted(
        [d for d in detections if d["category"] in {"troop", "building", "spell"}],
        key=lambda d: d["pressure_score"],
        reverse=True,
    )[:5]

    return {
        "board_reference": board_ref,
        "tower_health": [
            {
                "id": d["id"],
                "label": d["label"],
                "owner": d["owner"],
                "lane": d["lane"],
                "hp_estimate": (d.get("health") or {}).get("hp_estimate"),
                "health_ratio_estimate": (d.get("health") or {}).get("ratio_estimate"),
                "health_source": d.get("health_source"),
            }
            for d in detections
            if d["category"] == "tower"
        ],
        "friendly_units_on_enemy_side": [
            {
                "id": d["id"],
                "label": d["label"],
                "lane": d["lane"],
                "pressure_score": d["pressure_score"],
                "nearest_enemy_tower": d.get("nearest_enemy_tower"),
            }
            for d in friendly_on_enemy_side
        ],
        "enemy_units_on_friendly_side": [
            {
                "id": d["id"],
                "label": d["label"],
                "lane": d["lane"],
                "pressure_score": d["pressure_score"],
                "nearest_friendly_tower": d.get("nearest_friendly_tower"),
            }
            for d in enemy_on_friendly_side
        ],
        "lane_balance": lane_balance,
        "top_threats": [
            {
                "id": d["id"],
                "label": d["label"],
                "owner": d["owner"],
                "lane": d["lane"],
                "side": d["side"],
                "pressure_score": d["pressure_score"],
                "nearest_friendly_tower": d.get("nearest_friendly_tower"),
                "nearest_enemy_tower": d.get("nearest_enemy_tower"),
            }
            for d in top_threats
        ],
    }


def infer_game_state(detections: list) -> dict:
    elixir = next((d for d in detections if d["label"] == "elixir"), None)
    clock = next((d for d in detections if d["label"] == "clock"), None)
    return {
        "elixir_visible": elixir is not None,
        "clock_visible": clock is not None,
        "notes": [
            "Elixir and clock are detected as objects, but their numeric values are not OCR-parsed yet.",
            "Use object positions, side, lane, and owner as the main tactical input.",
        ],
    }


def infer_hand_cards(detections: list, hand_candidates: list) -> dict:
    hand_candidates = sorted(
        hand_candidates,
        key=lambda d: (d["center"]["x"], -d["confidence"]),
    )
    deduped = []
    for det in hand_candidates:
        if any(abs(det["center"]["x"] - prev["center"]["x"]) < 22 for prev in deduped):
            continue
        deduped.append(det)

    slot_centers = HAND_SLOT_CENTERS
    slots = []
    for idx, center_x in enumerate(slot_centers, start=1):
        best = None
        best_dist = None
        for det in deduped:
            dist = abs(det["center_normalized"]["x"] - center_x)
            if best_dist is None or dist < best_dist:
                best = det
                best_dist = dist
        if best is not None and best_dist is not None and best_dist < 0.14:
            slots.append({
                "slot": idx,
                "label": best["label"],
                "confidence": best["confidence"],
                "selected": False,
                "source_detection_id": best.get("id"),
            })
        else:
            slots.append({
                "slot": idx,
                "label": None,
                "confidence": None,
                "selected": False,
                "source_detection_id": None,
            })

    return {
        "slots": slots,
        "candidates": deduped,
        "notes": [
            "Hand cards are estimated from a dedicated low-threshold detector pass over the bottom screen region.",
            "The shipped detectors do not expose a reliable 'selected' class in these weights, so selected is always false.",
        ],
    }


def infer_hand_cards_from_classifier(frame: np.ndarray, classifier: CardSlotClassifier) -> dict:
    slots = []
    candidates = []
    for slot in crop_hand_slots(frame):
        result = classifier.classify(slot["image"])
        if result and result["label"] is not None:
            slots.append({
                "slot": slot["slot"],
                "label": result["label"],
                "confidence": round(result["confidence"], 4),
                "selected": False,
                "source_detection_id": None,
                "source": "slot_classifier",
                "margin": round(result.get("margin", 0.0), 4),
            })
            candidates.append({
                "slot": slot["slot"],
                "label": result["label"],
                "confidence": round(result["confidence"], 4),
                "margin": round(result.get("margin", 0.0), 4),
                "best_score": round(result.get("best_score", 0.0), 4),
                "bbox": slot["bbox"],
            })
        else:
            slots.append({
                "slot": slot["slot"],
                "label": None,
                "confidence": None,
                "selected": False,
                "source_detection_id": None,
            })
    return {
        "slots": slots,
        "candidates": candidates,
        "notes": [
            "Hand cards are classified from fixed bottom-hand slot crops using ORB + histogram matching against cards/ references.",
            f"classifier_enabled={classifier.enabled}",
            f"classifier_error={classifier.error}",
        ],
    }


def fuse_hand_cards(classifier_hand: dict, detector_hand: dict) -> dict:
    """
    Merge the fixed-slot classifier with the YOLO hand detector.

    Rules:
    - If classifier is empty, use detector.
    - If they agree, keep classifier.
    - If they disagree, prefer the detector unless the classifier is clearly
      confident and separated from its runner-up.
    """
    det_slots = {slot["slot"]: slot for slot in detector_hand.get("slots", [])}
    out_slots = []
    for slot in classifier_hand.get("slots", []):
        slot_id = slot.get("slot")
        fallback = det_slots.get(slot_id)
        classifier_label = slot.get("label")
        detector_label = fallback.get("label") if fallback else None

        if not classifier_label:
            if detector_label:
                fused = dict(fallback)
                fused["source"] = "yolo_hand_detector"
                out_slots.append(fused)
            else:
                out_slots.append(slot)
            continue

        if not detector_label or detector_label == classifier_label:
            out_slots.append(slot)
            continue

        classifier_conf = float(slot.get("confidence") or 0.0)
        classifier_margin = float(slot.get("margin") or 0.0)
        detector_conf = float(fallback.get("confidence") or 0.0)

        classifier_is_strong = classifier_conf >= 0.42 and classifier_margin >= 0.12
        detector_is_strong = detector_conf >= 0.55

        if detector_is_strong and not classifier_is_strong:
            fused = dict(fallback)
            fused["source"] = "yolo_hand_detector_override"
            fused["classifier_disagreement"] = {
                "label": classifier_label,
                "confidence": round(classifier_conf, 4),
                "margin": round(classifier_margin, 4),
            }
            out_slots.append(fused)
        else:
            kept = dict(slot)
            kept["detector_disagreement"] = {
                "label": detector_label,
                "confidence": round(detector_conf, 4),
            }
            out_slots.append(kept)
    return {
        "slots": out_slots,
        "candidates": classifier_hand.get("candidates", []),
        "notes": classifier_hand.get("notes", []) + [
            "YOLO hand detector fills empty classifier slots and may override weak classifier disagreements.",
        ],
    }


def infer_elixir_count(detections: list) -> dict:
    elixir_detections = [
        d for d in detections
        if d["label"] == "elixir" and d["center_normalized"]["y"] > 0.78
    ]
    elixir_detections.sort(key=lambda d: d["center"]["x"])

    distinct = []
    for det in elixir_detections:
        if any(abs(det["center"]["x"] - prev["center"]["x"]) < 18 for prev in distinct):
            continue
        distinct.append(det)

    estimate = min(10, len(distinct))
    return {
        "count_estimate": estimate if distinct else None,
        "source_detection_ids": [d["id"] for d in distinct],
        "notes": [
            "Elixir is estimated by counting distinct bottom-bar elixir detections.",
            "This is a heuristic and may undercount or overcount depending on detector quality.",
        ],
    }


def infer_elixir_count_from_frame(frame: np.ndarray, detections: list = None) -> dict:
    return infer_elixir_count_from_ocr(frame)


def tower_health_key_for_detection(det: dict) -> Optional[str]:
    if det.get("category") != "tower":
        return None
    owner = det.get("owner")
    label = det.get("label")
    lane = det.get("lane")
    if owner not in {"friendly", "enemy"}:
        return None
    if label == "king-tower":
        return f"{owner}_king_tower"
    if lane in {"left", "right"}:
        return f"{owner}_{lane}_tower"
    return None


def apply_tower_health_estimates(detections: list, tower_health: dict) -> None:
    tower_values = (tower_health or {}).get("towers", {})
    for det in detections:
        key = tower_health_key_for_detection(det)
        if not key:
            continue
        estimate = tower_values.get(key)
        if not estimate:
            continue
        hp = int(estimate.get("value") or 0)
        max_hp = int(estimate.get("max_hp") or (KING_TOWER_FULL_HP if key.endswith("king_tower") else PRINCESS_TOWER_FULL_HP))
        ratio = round(hp / float(max_hp), 3) if max_hp > 0 else None
        det["related_bar_id"] = None
        det["health"] = {
            "status": estimate.get("status"),
            "hp_estimate": hp,
            "max_hp": max_hp,
            "ratio_estimate": ratio,
            "ocr_text": estimate.get("text"),
            "ocr_confidence": estimate.get("ocr_confidence"),
        }
        det["health_source"] = "hardcoded_tower_health_ocr"


def infer_frame_state(
    frame: np.ndarray,
    frame_w: int,
    frame_h: int,
    detector,
    hand_classifier,
    executor: Optional[ThreadPoolExecutor] = None,
    bar_reader: Optional[TowerBarReader] = None,
) -> dict:
    def infer_hand():
        if hand_classifier.enabled:
            return infer_hand_cards_from_classifier(frame, hand_classifier)
        return {"slots": [], "candidates": [], "notes": ["card classifier disabled"]}

    if executor is None:
        detections = detector.detect(frame)
        hand_cards = infer_hand()
        elixir_estimate = infer_elixir_count_from_frame(frame)
    else:
        detect_future = executor.submit(detector.detect, frame)
        hand_future = executor.submit(infer_hand)
        elixir_future = executor.submit(infer_elixir_count_from_frame, frame)
        detections = detect_future.result()
        hand_cards = hand_future.result()
        elixir_estimate = elixir_future.result()

    # Tower HP has to wait for the detections: the OCR is anchored to the
    # detected bar boxes so it follows the arena camera. Fixed crops are the
    # fallback for the frames where no bar was detected at all.
    tower_health = infer_tower_health_from_bars(frame, detections, bar_reader, executor)
    if tower_health is None:
        tower_health = infer_tower_health_from_frame(frame)

    detections, board_ref = enrich_detections(detections, frame_w, frame_h)
    apply_tower_health_estimates(detections, tower_health)
    grouped = group_detections(detections)
    llm_summary = summarize_for_llm(detections, board_ref)
    inferred = infer_game_state(detections)
    return {
        "detections": detections,
        "board_ref": board_ref,
        "grouped": grouped,
        "llm_summary": llm_summary,
        "inferred": inferred,
        "hand_cards": hand_cards,
        "elixir_estimate": elixir_estimate,
        "tower_health": tower_health,
    }


def group_detections(detections: list) -> dict:
    grouped = {
        "troops": [],
        "buildings": [],
        "spells": [],
        "towers": [],
        "bars": [],
        "game_state": [],
    }
    for det in detections:
        category = det["category"]
        if category == "troop":
            grouped["troops"].append(det)
        elif category == "building":
            grouped["buildings"].append(det)
        elif category == "spell":
            grouped["spells"].append(det)
        elif category == "tower":
            grouped["towers"].append(det)
        elif category == "bar":
            grouped["bars"].append(det)
        elif category == "game_state":
            grouped["game_state"].append(det)
    return grouped


def start_frame_source(args, preset):
    """Frame source: Quartz grabs of the macOS iPhone Mirroring window."""
    from .mirror_capture import MirrorFrameSource

    max_size = args.max_size if args.max_size else preset.max_size
    source = MirrorFrameSource(max_size=max_size, target_fps=args.mirror_fps)
    info = source.probe()
    print(
        f"[INFO] iPhone Mirroring window {info['window_id']} "
        f"native {info['native']['width']}x{info['native']['height']} "
        f"(backing scale {info['backing_scale']}x)"
    )
    print(f"[INFO] Output resolution: {info['frame']['width']}x{info['frame']['height']}")
    source.start()
    return source.frame_width, source.frame_height, source.read_latest, source.stop


def run(args):
    preset = PRESETS[args.preset]
    print(f"[INFO] Preset: {preset.name} - {preset.description}")

    detector = ClashDetector(preset)
    hand_classifier = CardSlotClassifier()
    bar_reader = TowerBarReader()

    frame_w, frame_h, read_latest, stop_source = start_frame_source(args, preset)

    output_path = os.path.abspath(args.output_json)
    print(f"[INFO] Writing JSON snapshots to: {output_path}")
    print(f"[INFO] Snapshot interval: {args.interval_sec:.2f}s")

    sequence = 0
    executor = ThreadPoolExecutor(max_workers=4)
    try:
        while True:
            frame, system_ms = read_latest()

            if frame is None:
                time.sleep(0.05)
                continue

            t0 = time.perf_counter()
            screen = screen_kind(frame)
            frame_state = infer_frame_state(
                frame, frame_w, frame_h, detector, hand_classifier, executor, bar_reader
            )
            process_ms = (time.perf_counter() - t0) * 1000.0
            sequence += 1
            detections = frame_state["detections"]
            grouped = frame_state["grouped"]
            inferred = frame_state["inferred"]
            hand_cards = frame_state["hand_cards"]
            elixir_estimate = frame_state["elixir_estimate"]
            llm_summary = frame_state["llm_summary"]

            payload = {
                "sequence": sequence,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "preset": preset.name,
                "capture": {
                    "width": frame_w,
                    "height": frame_h,
                    "system_latency_ms": round(system_ms, 2),
                    "process_latency_ms": round(process_ms, 2),
                    "total_latency_ms": round(system_ms + process_ms, 2),
                },
                "schema_version": "2",
                "screen": {
                    "kind": screen,
                    "in_battle": screen == "battle",
                },
                "summary": {
                    "detections": len(detections),
                    "troops": sum(d["category"] == "troop" for d in detections),
                    "buildings": sum(d["category"] == "building" for d in detections),
                    "spells": sum(d["category"] == "spell" for d in detections),
                    "towers": sum(d["category"] == "tower" for d in detections),
                    "bars": sum(d["category"] == "bar" for d in detections),
                    "game_state": sum(d["category"] == "game_state" for d in detections),
                },
                "game_state_inferred": inferred,
                "hand_cards_inferred": hand_cards,
                "elixir_inferred": elixir_estimate,
                "tower_health_inferred": frame_state["tower_health"],
                "llm_summary": llm_summary,
                "llm_ready": {
                    "task": "Given this Clash Royale board snapshot, decide the best next move.",
                    "focus_order": [
                        "enemy units on friendly side",
                        "friendly push already on enemy side",
                        "lane imbalance",
                        "available defensive building or spell targets",
                        "tower pressure and survivability signals",
                    ],
                    "important_fields": [
                        "detections[].label",
                        "detections[].owner",
                        "detections[].lane",
                        "detections[].side",
                        "detections[].pressure_score",
                        "detections[].related_bar_id",
                        "llm_summary.enemy_units_on_friendly_side",
                        "llm_summary.friendly_units_on_enemy_side",
                        "llm_summary.lane_balance",
                        "llm_summary.top_threats",
                        "hand_cards_inferred.slots",
                        "elixir_inferred.count_estimate",
                    ],
                    "limitations": [
                        "numeric clock is not OCR parsed yet",
                        "tower HP uses hardcoded OCR crops and may miss unusual fonts/effects",
                    ],
                },
                "detections": detections,
                "grouped": grouped,
            }
            atomic_write_json(output_path, payload)
            print(
                f"[INFO] Wrote snapshot #{sequence} "
                f"({len(detections)} detections, {system_ms + process_ms:.1f}ms total)"
            )

            time.sleep(args.interval_sec)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        stop_source()
        executor.shutdown(wait=False)


def main():
    parser = argparse.ArgumentParser(
        description="Clash Royale capture that emits JSON snapshots instead of rendering a window"
    )
    parser.add_argument(
        "--preset", choices=sorted(PRESETS.keys()), default="fast",
        help="Capture preset (default: fast)",
    )
    parser.add_argument(
        "--output-json", type=str, default="clash_state.json",
        help="Path to the JSON snapshot file (default: clash_state.json)",
    )
    parser.add_argument(
        "--interval-sec", type=float, default=1.0,
        help="Seconds between JSON snapshots (default: 1.0)",
    )
    parser.add_argument(
        "--max-size", type=int, default=None,
        help="Override max dimension in pixels (default: preset value)",
    )
    parser.add_argument(
        "--mirror-fps", type=float, default=8.0,
        help="Target grab rate for the iPhone Mirroring window (default: 8)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
