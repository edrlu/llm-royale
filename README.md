# llm-royale

Live Clash Royale perception harness built on top of:

- `wty-yy/Clash-Royale-Detection-Dataset`
- `wty-yy/KataCR`
- `scrcpy`

This repo is not the bot yet. It is the live vision layer that future automation can sit on top of.

The current goal is simple:

- connect to a real Android device running Clash Royale
- capture the live screen feed locally
- reuse KataCR's detection conventions and arena crop logic
- run real-time object detection on the arena
- render a practical annotated viewer with debug crops and optional saved outputs

## What This Framework Is

This project is a perception-first framework for future Hog 2.6 automation.

It is designed to answer three questions cleanly:

1. How do we get live frames from a phone into Python reliably?
2. How do we run KataCR-compatible Clash Royale detection on those frames without rewriting the detector stack?
3. How do we expose the visual state in a way that is easy to debug now and easy to extend later into structured game-state extraction and action control?

The answer in this repo is:

- `scrcpy` provides the live mirrored phone window
- Python captures that live window with `mss`
- KataCR's YOLOv8-side model components are reused for inference
- KataCR's `split_part.py` crop conventions are reused for arena and UI regions
- the app projects arena detections back into the original live frame
- OpenCV displays the final annotated viewer

## Current Capabilities

- Live `scrcpy` capture from an Android device
- Swappable frame sources:
  - `scrcpy`
  - local video file
  - local camera device
- KataCR-compatible detection adapter using KataCR YOLOv8 custom model/predictor pieces
- KataCR class-name preservation through `idx2unit`
- KataCR crop reuse through `process_part(...)`
- Bounding boxes on the full live frame
- Class labels
- Confidence display
- Optional belong/team coloring if provided by the model output
- Debug side panels for:
  - arena region
  - hand-card region
  - elixir region
  - timer / HP region
  - center text region when available
- Optional saved outputs:
  - raw frames
  - annotated frames
  - JSON detections per frame
- FPS reporting
- Frame skipping and frame-rate throttling knobs

## Current Non-Goals

These are intentionally not implemented yet:

- tap controls
- action planning
- deck-specific policy logic
- full symbolic game-state extraction
- elixir OCR or card classification integration beyond region extraction
- bot loop / reinforcement learning loop

This repo is the perception harness only.

## Repository Layout

Top-level files added for the live viewer:

- [live_feed.py](/home/alien-cat/Documents/llm-royale/live_feed.py)
- [config.py](/home/alien-cat/Documents/llm-royale/config.py)
- [detector_adapter.py](/home/alien-cat/Documents/llm-royale/detector_adapter.py)
- [regions.py](/home/alien-cat/Documents/llm-royale/regions.py)
- [overlay.py](/home/alien-cat/Documents/llm-royale/overlay.py)
- [requirements-live-viewer.txt](/home/alien-cat/Documents/llm-royale/requirements-live-viewer.txt)
- [install_dependencies.sh](/home/alien-cat/Documents/llm-royale/install_dependencies.sh)
- [install_dependencies.bat](/home/alien-cat/Documents/llm-royale/install_dependencies.bat)
- [run.sh](/home/alien-cat/Documents/llm-royale/run.sh)
- [run.bat](/home/alien-cat/Documents/llm-royale/run.bat)

Vendored upstream dependencies:

- [vendor/KataCR](/home/alien-cat/Documents/llm-royale/vendor/KataCR)
- `vendor/Clash-Royale-Detection-Dataset`

Generated output:

- `outputs/live_viewer`

## Architecture

The runtime pipeline is:

1. Frame source
2. Region extraction
3. Detector inference
4. Coordinate projection
5. Overlay rendering
6. Optional debug persistence
7. OpenCV display

### 1. Frame Source

`live_feed.py` defines a small `FrameSource` abstraction.

Implemented sources:

- `ScrcpyWindowSource`
- `OpenCVFrameSource`

`ScrcpyWindowSource` does this:

- launches `scrcpy`
- waits for the titled scrcpy window if no manual capture region is supplied
- captures the current client area with `mss`
- returns frames as BGR NumPy arrays

This keeps the source layer replaceable. If you later want to ingest raw H.264, direct ADB screen capture, or a lower-latency transport, you can replace only the frame-source implementation without touching the detector or overlay pipeline.

### 2. Region Extraction

`regions.py` is the bridge between raw phone frames and KataCR-compatible crops.

It reuses KataCR's crop conventions from:

- `katacr.build_dataset.utils.split_part.process_part`
- `katacr.build_dataset.constant`

The current region extraction returns:

- `arena_for_model_bgr`
- `arena_mapping`
- `arena_debug_bgr`
- `hand_cards_bgr`
- `elixir_bgr`
- `timer_hp_bgr`
- `center_text_bgr`

Important detail:

- detection is run on KataCR's arena crop, not on the full phone frame
- this matches the existing KataCR training/inference assumptions more closely
- detections are then projected back to the original frame for visualization

That means the viewer is compatible with the detector's expected image layout while still staying useful to a human operator.

### 3. Detector Inference

`detector_adapter.py` wraps the KataCR YOLOv8 inference path.

The adapter reuses:

- `katacr.yolov8.custom_model.CRDetectionModel`
- `katacr.yolov8.custom_predict.CRDetectionPredictor`
- `katacr.constants.label_list.idx2unit`

The wrapper class `KataCRDetector` exposes a small interface:

- load model weights
- run prediction on an arena image
- normalize results into a stable local `Detection` dataclass

The normalized detection payload contains:

- `xyxy`
- `conf`
- `class_id`
- `class_name`
- `belong`
- `track_id` if present

### 4. Coordinate Projection

Because inference runs on the arena crop, `overlay.py` contains logic to project those crop-space coordinates back onto the full screen frame.

This is critical for future automation because:

- the detector stays aligned with KataCR's trained view of the arena
- the controller/state extractor can still reason about detections in full-screen coordinates

### 5. Overlay Rendering

`overlay.py` draws:

- bounding boxes
- labels
- confidence values
- optional team/belong color cues
- live FPS / inference timing text

It also builds the side debug panel stack used in the OpenCV window.

### 6. Optional Persistence

`DebugWriter` in `live_feed.py` can save:

- raw frames
- annotated frames
- per-frame JSON detections

This is useful for:

- debugging bad detections
- creating future state-extraction test fixtures
- profiling frame skipping and display settings
- collecting difficult live examples for retraining

### 7. Display

The final view is shown in a single OpenCV window named:

- `Clash Royale Live Vision`

## How KataCR Is Reused

This repo deliberately avoids replacing KataCR's logic where reuse is practical.

### Reused directly

- label naming conventions from `idx2unit`
- crop/split logic from `split_part.py`
- region constants from `constant.py`
- KataCR custom YOLOv8 predictor/model code
- KataCR belong/team-style output handling where present

### Adapted

- KataCR's YOLOv8 stack was adapted for inference-only loading in this repo so it does not drag in training-only dependencies at runtime
- a small wrapper `YOLOCRLive` is used so the live harness can run prediction without needing the full KataCR training pipeline

### Not reused

- the older JAX/Flax detector path was not chosen as the primary live path

Why:

- the YOLOv8 side is much more practical for real-time local inference in this harness
- the older path is more tightly coupled to KataCR's historical training/checkpoint environment
- for this task, practical live inference matters more than maintaining both detector stacks

## Why The Arena Crop Is Central

Clash Royale full-screen frames contain multiple visual layers:

- top timer and HP UI
- central arena
- lower hand-card / elixir UI
- popups / center text / playback overlays

The detector is most useful on the arena itself.

So the framework treats the arena as the primary detection surface and the other regions as debug or future state-extraction surfaces.

This is the right split for a later automation architecture:

- arena detections become unit/tower/entity observations
- hand-card crop becomes card-state recognition input
- elixir crop becomes elixir-state recognition input
- timer/HP crop becomes reward/state tracking input

## Module Responsibilities

### [live_feed.py](/home/alien-cat/Documents/llm-royale/live_feed.py)

Main runtime loop.

Responsibilities:

- parse config
- initialize environment
- build the detector
- build the frame source
- run the main loop
- handle frame skipping
- compute FPS
- display the annotated window
- optionally write outputs

### [config.py](/home/alien-cat/Documents/llm-royale/config.py)

Configuration and CLI parsing.

Responsibilities:

- define project paths
- ensure KataCR imports are reachable
- set `KATACR_DATASET_PATH`
- expose the CLI interface
- store runtime settings in dataclasses

### [detector_adapter.py](/home/alien-cat/Documents/llm-royale/detector_adapter.py)

Inference adapter layer.

Responsibilities:

- wrap KataCR YOLOv8 inference
- preserve class mappings
- return stable local detection objects

### [regions.py](/home/alien-cat/Documents/llm-royale/regions.py)

Region extraction and crop management.

Responsibilities:

- reuse KataCR crop logic
- extract arena and UI subregions
- compute the mapping from arena crop coordinates to full-frame coordinates

### [overlay.py](/home/alien-cat/Documents/llm-royale/overlay.py)

Visualization layer.

Responsibilities:

- full-frame bounding-box drawing
- label rendering
- team coloring
- debug panel composition

## Setup

### Python environment

Linux:

```bash
./install_dependencies.sh
```

Windows:

```bat
install_dependencies.bat
```

These scripts:

- create `.venv`
- install Python dependencies into `.venv`
- install CPU PyTorch wheels by default
- clone the detection dataset into `vendor/Clash-Royale-Detection-Dataset` if missing

Why CPU wheels by default:

- they install more predictably across machines
- they avoid pulling machine-specific CUDA wheels in the setup script

If your target machine has a CUDA-capable GPU, replace the CPU PyTorch wheels inside `.venv` with the correct CUDA build for your machine.

### System dependencies

Install these outside Python:

- `scrcpy`
- `adb` / Android platform-tools

Linux note:

- the current scrcpy-window capture path works best under X11
- on Wayland, window enumeration/capture can be inconsistent depending on compositor rules
- if needed, use `--capture-region` as a manual fallback

### Weights

You must supply your own trained KataCR YOLOv8 `.pt` weights.

The repo does not include inference weights.

Recommended convention:

- create a local `weights/` directory
- place your `.pt` file there
- pass it explicitly via `--weights`

Example:

```bash
./run.sh --source scrcpy --weights /absolute/path/to/weights/katacr_live.pt --device cuda
```

## Running

### Linux

```bash
./run.sh --source scrcpy --weights /absolute/path/to/weights.pt --device cuda
```

### Windows

```bat
run.bat --source scrcpy --weights C:\path\to\weights.pt --device cuda
```

### Test with a local video file

```bash
./run.sh --source video --input /absolute/path/to/test.mp4 --weights /absolute/path/to/weights.pt --device cpu
```

### Test with a local camera

```bash
./run.sh --source camera --input 0 --weights /absolute/path/to/weights.pt --device cpu
```

## CLI Reference

Core arguments:

- `--source {scrcpy,video,camera}`
- `--input`
- `--weights`
- `--device`
- `--conf-thres`
- `--iou-thres`

Performance / display:

- `--frame-skip`
- `--max-fps`
- `--display-scale`
- `--no-panels`
- `--no-labels`
- `--no-conf`
- `--show-belong`

Output:

- `--output-dir`
- `--save-raw`
- `--save-annotated`
- `--save-json`

Capture / scrcpy:

- `--capture-region left,top,width,height`
- `--playback-crop`
- `--scrcpy-path`
- `--scrcpy-serial`
- `--scrcpy-window-title`
- `--scrcpy-max-size`
- `--scrcpy-no-stay-awake`
- `--scrcpy-fullscreen`
- `--scrcpy-extra-args`

Dataset:

- `--dataset-path`

## Output Structure

When enabled, outputs are written under:

- `outputs/live_viewer`

Subdirectories:

- `raw_frames/`
- `annotated_frames/`
- `detections_json/`

Each saved frame is keyed by zero-padded frame index.

JSON records include:

- frame index
- source name
- timestamp
- detection list

Each detection record includes:

- `xyxy`
- `conf`
- `class_id`
- `class_name`
- `belong`
- `track_id`

## Resolution Handling

This framework explicitly supports tall phone aspect ratios such as:

- `1080x2400`
- `576x1280`
- `592x1280`
- `600x1280`

The important part is that region extraction does not hardcode a single pixel geometry. It follows KataCR's ratio-dependent crop definitions from `katacr.build_dataset.constant`.

That means:

- arena crop selection stays aligned with KataCR assumptions
- hand/elixir/timer regions stay aligned with the same source conventions
- you can still use `--capture-region` if your live scrcpy window includes extra borders or compositor artifacts

## Performance Model

The app gives you three main speed levers:

- `--frame-skip`
- `--max-fps`
- `--scrcpy-max-size`

Practical guidance:

- start with `--frame-skip 1` or `--frame-skip 2` if your GPU is weak
- lower `--scrcpy-max-size` if the mirrored window is larger than you need
- use `--device cpu` only for debugging, not for serious real-time use
- disable panels with `--no-panels` if you want a simpler display path

The displayed FPS line shows:

- overall viewer FPS
- last inference time in milliseconds
- current detection count

## Debugging Strategy

If the viewer is not behaving correctly, check in this order:

1. `adb devices` shows the phone
2. `scrcpy` launches cleanly by itself
3. the weights path is correct
4. the OpenCV window opens
5. the arena debug crop looks visually correct
6. saved JSON detections match what you expect

If detections look spatially wrong:

- inspect the arena debug crop first
- check if scrcpy borders or desktop compositor shadows are being captured
- use `--capture-region` to manually constrain the screen grab

If detections exist but labels are wrong:

- verify the `.pt` weights match KataCR's class configuration
- confirm the model really outputs the same class ordering expected by KataCR's label list

## Important Upstream Integration Notes

This repo vendors KataCR source rather than treating it as a pure untouched dependency.

Reason:

- a few small compatibility adjustments were needed so the live harness can import and run the KataCR YOLOv8 path cleanly in this environment

That keeps the live viewer runnable without forcing the full historical KataCR training environment into the runtime path.

The dataset path is also overridden through:

- `KATACR_DATASET_PATH`

Default:

- `vendor/Clash-Royale-Detection-Dataset`

## Extension Path Toward Full Automation

The intended next layers are:

1. Structured state extraction
2. Deck-specific interpretation
3. Action planner
4. Tap / input executor

A clean next step would be to add a `state_extractor.py` that turns:

- arena detections
- hand-card crop
- elixir crop
- timer/HP crop

into a structured state object like:

- current hand
- elixir count
- visible friendly units
- visible enemy units
- tower HP estimates
- game timer

After that, an `action_controller.py` could decide:

- play hog
- hold elixir
- cycle skeletons
- defend with cannon

And only after that should tap automation be added.

This ordering matters. Good automation needs a trustworthy state layer first.

## Example Workflows

### Live viewer on a real phone

```bash
./run.sh \
  --source scrcpy \
  --weights /absolute/path/to/katacr_live.pt \
  --device cuda \
  --frame-skip 1 \
  --show-belong \
  --save-json
```

### Video replay debugging

```bash
./run.sh \
  --source video \
  --input /absolute/path/to/replay.mp4 \
  --weights /absolute/path/to/katacr_live.pt \
  --device cpu \
  --save-annotated \
  --save-json
```

### Manual capture region fallback

```bash
./run.sh \
  --source scrcpy \
  --weights /absolute/path/to/katacr_live.pt \
  --capture-region 100,80,540,1200 \
  --device cuda
```

## Known Limitations

- no bundled weights
- no direct raw scrcpy stream ingestion yet
- current scrcpy integration captures the visible window instead of decoding the device stream directly
- Wayland/Linux desktop setups may require manual capture-region tuning
- the framework does not yet extract full symbolic state
- the framework does not yet send actions back to the phone

## Short Summary

This repo now provides a practical local Clash Royale live-vision framework with:

- live phone capture
- KataCR-compatible detection
- reusable arena/UI crop logic
- real-time visualization
- debug persistence
- an architecture that can grow into full Hog 2.6 automation

If you are extending it, preserve this separation:

- frame acquisition
- perception
- state extraction
- decision logic
- control

That separation is what will keep the future bot maintainable.
