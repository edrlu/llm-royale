"""
Capture pipeline configuration presets.

Usage:
    python screen_capture.py --preset quality     # Current defaults, max accuracy
    python screen_capture.py --preset fast         # Optimized for low latency
"""

from dataclasses import dataclass


# UI crop geometry.
#
# All values are normalized to the iPhone Mirroring capture (portrait, roughly
# 0.45 aspect). They were measured off a live 1v1 frame, not ported from the
# Android layout — the taller screen shifts every HUD element.
HAND_SLOT_CENTERS = [0.317, 0.490, 0.663, 0.836]
HAND_CARD_Y_LOWER_NORM = 0.845
HAND_CARD_Y_UPPER_NORM = 0.915

# The elixir *digit*, kept only as a debug cross-check. Tesseract reads this
# stylized font badly (6 -> 1, 10 -> 07), so the real count comes from the pips.
ELIXIR_BAR_X_LOWER_NORM = 0.265
ELIXIR_BAR_X_UPPER_NORM = 0.350
ELIXIR_BAR_Y_LOWER_NORM = 0.936
ELIXIR_BAR_Y_UPPER_NORM = 0.954

# The elixir bar itself: ten equal segments between these two x bounds. Counting
# filled segments by colour is exact, which OCR on the digit never was.
# The y band deliberately stops above the "Max: 10" caption, and the count digit
# overlaps the first segment or two — the classifier ignores near-white pixels
# so the digit does not mask the pink underneath it.
ELIXIR_PIP_X_LEFT_NORM = 0.285
ELIXIR_PIP_X_RIGHT_NORM = 0.943
ELIXIR_PIP_Y_LOWER_NORM = 0.941
ELIXIR_PIP_Y_UPPER_NORM = 0.955
ELIXIR_PIP_COUNT = 10

ENEMY_LEFT_TOWER_HEALTH_X_LOWER_NORM = 0.180
ENEMY_LEFT_TOWER_HEALTH_X_UPPER_NORM = 0.280
ENEMY_LEFT_TOWER_HEALTH_Y_LOWER_NORM = 0.202
ENEMY_LEFT_TOWER_HEALTH_Y_UPPER_NORM = 0.225

# The king bars only appear once a princess tower falls, so these two boxes are
# positioned relative to the king tower art rather than measured off a live bar.
ENEMY_KING_TOWER_HEALTH_X_LOWER_NORM = 0.430
ENEMY_KING_TOWER_HEALTH_X_UPPER_NORM = 0.570
ENEMY_KING_TOWER_HEALTH_Y_LOWER_NORM = 0.140
ENEMY_KING_TOWER_HEALTH_Y_UPPER_NORM = 0.168

ENEMY_RIGHT_TOWER_HEALTH_X_LOWER_NORM = 0.740
ENEMY_RIGHT_TOWER_HEALTH_X_UPPER_NORM = 0.840
ENEMY_RIGHT_TOWER_HEALTH_Y_LOWER_NORM = 0.202
ENEMY_RIGHT_TOWER_HEALTH_Y_UPPER_NORM = 0.225

FRIENDLY_LEFT_TOWER_HEALTH_X_LOWER_NORM = 0.180
FRIENDLY_LEFT_TOWER_HEALTH_X_UPPER_NORM = 0.280
FRIENDLY_LEFT_TOWER_HEALTH_Y_LOWER_NORM = 0.633
FRIENDLY_LEFT_TOWER_HEALTH_Y_UPPER_NORM = 0.657

FRIENDLY_KING_TOWER_HEALTH_X_LOWER_NORM = 0.430
FRIENDLY_KING_TOWER_HEALTH_X_UPPER_NORM = 0.570
FRIENDLY_KING_TOWER_HEALTH_Y_LOWER_NORM = 0.772
FRIENDLY_KING_TOWER_HEALTH_Y_UPPER_NORM = 0.800

FRIENDLY_RIGHT_TOWER_HEALTH_X_LOWER_NORM = 0.740
FRIENDLY_RIGHT_TOWER_HEALTH_X_UPPER_NORM = 0.840
FRIENDLY_RIGHT_TOWER_HEALTH_Y_LOWER_NORM = 0.633
FRIENDLY_RIGHT_TOWER_HEALTH_Y_UPPER_NORM = 0.657

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
