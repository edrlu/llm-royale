@echo off
setlocal

set ROOT_DIR=%~dp0
set VENV_PYTHON=%ROOT_DIR%.venv\Scripts\python.exe
set HAS_WEIGHTS_ARG=
set DEFAULT_WEIGHTS_ARG=

if not exist "%VENV_PYTHON%" (
  echo .venv is missing. Run install_dependencies.bat first.
  exit /b 1
)

for %%A in (%*) do (
  if /I "%%~A"=="--weights" set HAS_WEIGHTS_ARG=1
)

if not defined HAS_WEIGHTS_ARG (
  dir /b "%ROOT_DIR%runs\*.pt" >nul 2>nul
  if not errorlevel 1 (
    set DEFAULT_WEIGHTS_ARG=--weights "%ROOT_DIR%runs"
  ) else (
    dir /b "%ROOT_DIR%weights\*.pt" >nul 2>nul
    if not errorlevel 1 (
      set DEFAULT_WEIGHTS_ARG=--weights "%ROOT_DIR%weights"
    )
  )
)

rem --scrcpy-capture-mode screencap  uses individual `adb screencap` calls instead of
rem   the H264 screenrecord pipeline.  H264 screenrecord buffers 1-4 s of video on the
rem   device before Python receives the first frame, which is the root cause of the
rem   observed 5-second visual latency even when `lag:` shows ~1 s.  screencap gives
rem   the CURRENT frame on every call with no buffering (~300-600 ms over USB).
rem   If scrcpy direct mode works on your setup, `--scrcpy-capture-mode auto` is even
rem   faster (~50-100 ms pipeline latency).
rem --infer-size 320    keeps YOLO at 320 px, not padded to the model default (640).
rem --conf-thres 0.35   cuts low-confidence boxes to reduce NMS work on CPU.
rem --max-fps 30        caps the display loop so it does not spin at 150 fps and
rem                     waste CPU that the inference thread needs.
set DEFAULT_LIVE_ARGS=--source scrcpy --scrcpy-capture-mode screencap --lightweight --arena-only --infer-size 320 --conf-thres 0.35 --scrcpy-max-size 900 --max-fps 30 --debug-source

"%VENV_PYTHON%" "%ROOT_DIR%live_feed.py" %DEFAULT_WEIGHTS_ARG% %DEFAULT_LIVE_ARGS% %*
endlocal
