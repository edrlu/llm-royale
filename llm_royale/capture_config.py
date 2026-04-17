"""
Capture pipeline configuration presets.

Usage:
    python screen_capture.py --preset quality     # Current defaults, max accuracy
    python screen_capture.py --preset fast         # Optimized for low latency
"""

from dataclasses import dataclass


# UI crop geometry
HAND_SLOT_CENTERS = [0.35, 0.50, 0.68, 0.83]
HAND_CARD_Y_LOWER_NORM = 0.82
HAND_CARD_Y_UPPER_NORM = 0.94

ELIXIR_BAR_X_LOWER_NORM = 0.29
ELIXIR_BAR_X_UPPER_NORM = 0.36

#15
ELIXIR_BAR_Y_LOWER_NORM = 0.953
ELIXIR_BAR_Y_UPPER_NORM = 0.985

ENEMY_LEFT_TOWER_HEALTH_X_LOWER_NORM = 0.230
ENEMY_LEFT_TOWER_HEALTH_X_UPPER_NORM = 0.299
ENEMY_LEFT_TOWER_HEALTH_Y_LOWER_NORM = 0.112
ENEMY_LEFT_TOWER_HEALTH_Y_UPPER_NORM = 0.131

ENEMY_KING_TOWER_HEALTH_X_LOWER_NORM = 0.465
ENEMY_KING_TOWER_HEALTH_X_UPPER_NORM = 0.55
ENEMY_KING_TOWER_HEALTH_Y_LOWER_NORM = 0.010
ENEMY_KING_TOWER_HEALTH_Y_UPPER_NORM = 0.035

ENEMY_RIGHT_TOWER_HEALTH_X_LOWER_NORM = 0.700
ENEMY_RIGHT_TOWER_HEALTH_X_UPPER_NORM = 0.769
ENEMY_RIGHT_TOWER_HEALTH_Y_LOWER_NORM = 0.112
ENEMY_RIGHT_TOWER_HEALTH_Y_UPPER_NORM = 0.131

FRIENDLY_LEFT_TOWER_HEALTH_X_LOWER_NORM = 0.230
FRIENDLY_LEFT_TOWER_HEALTH_X_UPPER_NORM = 0.299
FRIENDLY_LEFT_TOWER_HEALTH_Y_LOWER_NORM = 0.613
FRIENDLY_LEFT_TOWER_HEALTH_Y_UPPER_NORM = 0.633

FRIENDLY_KING_TOWER_HEALTH_X_LOWER_NORM = 0.465
FRIENDLY_KING_TOWER_HEALTH_X_UPPER_NORM = 0.55
FRIENDLY_KING_TOWER_HEALTH_Y_LOWER_NORM = 0.75
FRIENDLY_KING_TOWER_HEALTH_Y_UPPER_NORM = 0.779

FRIENDLY_RIGHT_TOWER_HEALTH_X_LOWER_NORM = 0.700
FRIENDLY_RIGHT_TOWER_HEALTH_X_UPPER_NORM = 0.769
FRIENDLY_RIGHT_TOWER_HEALTH_Y_LOWER_NORM = 0.613
FRIENDLY_RIGHT_TOWER_HEALTH_Y_UPPER_NORM = 0.633

PRINCESS_TOWER_FULL_HP = 3052
KING_TOWER_FULL_HP = 4824



@dataclass(frozen=True)
class CapturePreset:
    name: str
    description: str

    # Capture pipeline
    max_size: int           # max pixel dimension for capture
    bit_rate: str           # H.264 bitrate over USB

    # Clash detector
    clash_num_models: int   # 1 = single detector, 2 = combo (both detectors)
    clash_device: str       # "cpu", "mps" (Apple Metal GPU), "cuda"
    clash_infer_size: int   # resolution fed to YOLO (lower = faster)
    clash_conf: float       # confidence threshold
    clash_iou: float        # NMS IoU threshold
    clash_half: bool        # FP16 half precision
    clash_fuse: bool        # fuse Conv+BN layers

    # Generic YOLO
    yolo_model: str
    yolo_conf: float


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

QUALITY = CapturePreset(
    name="quality",
    description="Max accuracy, dual detectors, full resolution. ~700ms process latency.",

    max_size=1024,
    bit_rate="8M",

    clash_num_models=2,
    clash_device="cpu",
    clash_infer_size=1024,      # full frame sent to YOLO
    clash_conf=0.25,
    clash_iou=0.45,
    clash_half=False,
    clash_fuse=False,

    yolo_model="yolov8n.pt",
    yolo_conf=0.4,
)

FAST = CapturePreset(
    name="fast",
    description="Balanced low-latency preset with both Clash detectors and MPS acceleration.",

    max_size=832,               # more detail for small troops while staying below quality
    bit_rate="6M",              # reduce compression artifacts on moving units

    clash_num_models=2,         # keep both detectors for full Hog 2.6 coverage
    clash_device="mps",         # Apple Metal GPU acceleration
    clash_infer_size=832,       # help small stationary targets like cannon without full quality cost
    clash_conf=0.20,            # keep borderline troop/building detections
    clash_iou=0.45,
    clash_half=True,            # FP16 — ~1.5-2x faster, minimal accuracy loss
    clash_fuse=True,            # fuse Conv+BN layers — free 10-20% speedup

    yolo_model="yolov8n.pt",
    yolo_conf=0.4,
)

# Registry
PRESETS = {
    "quality": QUALITY,
    "fast": FAST,
}
