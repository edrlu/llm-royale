# llm-royale

Plays Clash Royale on a real iPhone, mirrored to a Mac. YOLO reads the arena,
an LLM decides the next move, and synthetic mouse events become touches on the
phone.

```
iPhone  ->  macOS "iPhone Mirroring" window
        ->  CGWindowListCreateImage        (window capture, ~17ms a frame)
        ->  two YOLO detectors + HUD readers  (~300ms a snapshot)
        ->  compact JSON  ->  OpenAI Responses API
        ->  Quartz CGEvent drag            (forwarded to the phone as a touch)
```

## Setup

```bash
./install.sh
```

It installs `ffmpeg` and `tesseract`, builds the virtualenv, and checks the two
things it cannot install for you: the YOLO weights and an OpenAI key. It is safe
to re-run.

Requirements it does not handle:

- **macOS 15 (Sequoia) or newer** — iPhone Mirroring does not exist before it.
- **Screen Recording permission** for whichever terminal you run from
  (System Settings > Privacy & Security > Screen Recording). Without it the
  capture returns nothing.
- **The model weights** (~168MB, `clash-yolo-pipeline/models/`). Too large for
  git — copy them from a machine that already works:
  ```bash
  scp -r you@othermac:llm-royale/clash-yolo-pipeline/models/ clash-yolo-pipeline/models/
  ```
- **Your OpenAI key** in `.env` as `OPENAI_API_KEY=...`.

## Playing one battle

Open iPhone Mirroring, start a battle yourself, then:

```bash
./run.sh
```

It never touches the menus — it will not press Battle and will not tap through
result screens. It waits for a battle, plays it, notices when it ends, and
exits. The video lands in `debug/vid/` and the decision log in `debug/logs/`.

`./run.sh --no-record` skips the recording.

To stop early, from another terminal:

```bash
venv/bin/python -m llm_royale.stopper
```

## Playing continuously

```bash
./run_loop.sh        # until stopped
./run_loop.sh 5      # five matches
```

This one *does* drive the menus: presses Battle, dismisses result banners, taps
through chests. One video per match, oldest pruned past 25.

## What is where

| | |
|---|---|
| `mirror_capture.py` | reads the iPhone Mirroring window |
| `mac_action.py` | taps and drags, via Quartz CGEvent |
| `clash_capture.py` | YOLO + HUD readers, writes state JSON |
| `llm_clasher.py` | the loop: state -> LLM -> action |
| `match_navigator.py` | which screen is showing, and how to leave it |
| `hud_ocr.py` | elixir bar and tower HP |
| `cycle_tracker.py` | hand classification and cycle memory |
| `overlay.py`, `recorder.py` | annotated video |
| `stopper.py` | the kill switch |
| `tools/` | benchmarks and one-off experiments |
| `docs/` | pipeline notes |

## Known limitations

**It plays legally and reliably, but not well.** It loses most matches.

The largest known problem is that most placements do not register: the elixir
bar is not charged even with plenty in hand, so the swipe is not landing as a
card placement. The drag timing was widened to address this (hold, move, settle)
but that fix is **unverified** — the measurement used to evaluate it compared
consecutive snapshots, which is too coarse to attribute a change to one
placement.

`--verify-elixir` reads the bar directly either side of each swipe, which is
exact but adds ~0.55s per placement. One match with it on the old timing and one
on the new would settle it.

Other known-weak areas are noted in the source: the hand classifier reports
nothing for greyed-out (unaffordable) cards on purpose, since a wrong label
makes the executor play the wrong card, and tower HP OCR is unreliable when a
troop stands in front of the bar.
