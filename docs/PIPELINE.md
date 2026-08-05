# Android Screen Capture Pipeline

## Overview

Real-time Android screen capture with optional OpenCV/YOLO processing.

## Pipeline

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐     ┌──────────────┐
│   Android    │     │     ffmpeg        │     │    Python      │     │   OpenCV     │
│   Device     │────▶│   (decoder)       │────▶│  (processing)  │────▶│  (display)   │
│              │     │                   │     │                │     │              │
│ adb exec-out │     │ H.264 → raw BGR24 │     │ numpy reshape  │     │ imshow +     │
│ screenrecord │     │ + scale           │     │ + CV/YOLO      │     │ status bar   │
│ --format=h264│     │                   │     │                │     │              │
└──────────────┘     └──────────────────┘     └───────────────┘     └──────────────┘
      USB/ADB              pipe:0 → pipe:1          stdout.read()         waitKey
```

## Data Flow

1. **ADB screenrecord** captures the device screen and outputs raw H.264 to stdout
2. **ffmpeg** reads H.264 from stdin, decodes it, scales to target resolution, outputs raw BGR24 frames to stdout
3. **Python** reads exactly `width × height × 3` bytes per frame, reshapes into a numpy array
4. **Processor** (optional) runs OpenCV edge detection or YOLOv8 object detection on the frame
5. **OpenCV** renders the frame with an FPS/mode/latency status bar and displays it

## Processing Modes

| Mode   | Hotkey | Description                              |
|--------|--------|------------------------------------------|
| Mirror | `1`    | Pass-through, no processing              |
| OpenCV | `2`    | Canny edge detection blended with source |
| YOLO   | `3`    | YOLOv8n real-time object detection       |

## Commands

```bash
# Mirror only
python screen_capture.py

# OpenCV edge detection
python screen_capture.py --mode opencv

# YOLO object detection
python screen_capture.py --mode yolo

# Lower resolution for higher FPS
python screen_capture.py --max-size 720

# Adjust bitrate
python screen_capture.py --bit-rate 4M
```

## Requirements

- `adb` — Android Debug Bridge (`brew install android-platform-tools`)
- `ffmpeg` — video decoder (`brew install ffmpeg`)
- Python packages: `opencv-python`, `numpy`, `ultralytics` (for YOLO)

## Why H.264 Encode/Decode?

At first glance the encode → USB → decode pipeline seems wasteful. Why not just send raw pixels?

```
Raw screencap:   1600x2560 × 4 bytes (RGBA) = ~16MB per frame
                 USB transfer: ~1600ms → 0.6 FPS

H.264 pipeline:  ~30-50KB per frame (compressed)
                 USB transfer: ~1ms → 30 FPS
                 Encode cost:  ~5-10ms (on device GPU/HW encoder)
                 Decode cost:  ~3-5ms  (ffmpeg on host)
```

The H.264 encode/decode adds ~10-15ms of overhead but saves ~1500ms of USB transfer time per frame. The compression ratio (~300:1) makes real-time streaming possible.

## Latency Display

The status bar shows two latency metrics:

- **System** (cyan) — time waiting for the next frame from the pipeline. Includes: device screen capture, H.264 encoding, USB transfer, and ffmpeg decoding. This is the real end-to-end pipeline latency.
- **Process** (yellow) — time spent in OpenCV/YOLO processing only. In mirror mode this is ~0ms.

## Notes

- ADB `screenrecord` has a 3-minute limit per session; the script auto-restarts it
- YOLO uses the `yolov8n` (nano) model by default for speed; use `--yolo-model yolov8s.pt` for better accuracy
