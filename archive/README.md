# archive

Nothing here is imported or run by the bot. It is kept because it records how
parts of the pipeline were arrived at, not because anything depends on it.

| | why it is here |
|---|---|
| `elixir_classifier.py` | Shape-based digit recognition for the elixir number — counted holes and strokes to identify each digit. Superseded by counting filled segments on the elixir bar (`llm_royale/hud_ocr.py`), which is exact where reading that stylized font never was. |
| `llm_clasher_debugger.py` | The original annotated-video harness: wrote one PNG per decision, then encoded them at exit. Superseded by `run.sh --record`, which records on its own thread at 60fps and streams to ffmpeg, so the video survives a hard kill and plays back at real speed. |
| `PIPELINE_android.md` | Describes the adb + ffmpeg capture used when this ran against an Android device. The Android backend now lives on the `android` branch; keeping this in `docs/` alongside the iPhone setup was misleading. |
| `experiments/` | One-off scripts and sample crops used while working out the card and elixir classifiers. The working versions are `llm_royale/cycle_tracker.py` and `llm_royale/hud_ocr.py`. |
