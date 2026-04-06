# llm-royale

Live Clash Royale perception viewer built on top of KataCR, the Clash Royale detection dataset, and scrcpy.

## What this adds

- Live `scrcpy` frame ingestion into Python/OpenCV
- KataCR YOLOv8 detector adapter that preserves KataCR class mappings and crop conventions
- Real-time annotated viewer with boxes, labels, confidences, and optional belong/team coloring
- Debug panels for arena, hand cards, elixir, and timer/HP crops
- Optional saving of raw frames, annotated frames, and per-frame JSON detections

## Quick start

1. Run `./install_dependencies.sh` on Linux or `install_dependencies.bat` on Windows.
   This creates `.venv`, installs Python packages, and clones the dataset repo into `vendor/Clash-Royale-Detection-Dataset` if it is missing.
2. Install `scrcpy` and `adb` on your system.
3. Supply your own KataCR YOLOv8 `.pt` weights.
4. Run:

```bash
./run.sh --source scrcpy --weights /absolute/path/to/weights.pt --device cuda
```

Windows:

```bat
run.bat --source scrcpy --weights C:\path\to\weights.pt --device cuda
```

## Important paths

- KataCR repo: `vendor/KataCR`
- Dataset repo: `vendor/Clash-Royale-Detection-Dataset`
- Main app: `live_feed.py`
- Outputs: `outputs/live_viewer`

## Notes

- The app defaults `KATACR_DATASET_PATH` to `vendor/Clash-Royale-Detection-Dataset`.
- Weights are not bundled here. Place them anywhere local and pass `--weights`.
- The current harness is perception-only. No tap/control automation is implemented yet.
