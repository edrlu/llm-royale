@echo off
setlocal

set ROOT_DIR=%~dp0
set VENV_DIR=%ROOT_DIR%.venv
set VENDOR_DIR=%ROOT_DIR%vendor

py -3 -m venv "%VENV_DIR%"
call "%VENV_DIR%\Scripts\activate.bat"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -r "%ROOT_DIR%requirements-live-viewer.txt"

if not exist "%VENDOR_DIR%\Clash-Royale-Detection-Dataset\images" (
  if not exist "%VENDOR_DIR%" mkdir "%VENDOR_DIR%"
  git clone https://github.com/wty-yy/Clash-Royale-Detection-Dataset "%VENDOR_DIR%\Clash-Royale-Detection-Dataset"
)

echo Python dependencies installed into .venv.
echo.
echo System requirements:
echo - Install scrcpy
echo - Install adb / Android platform-tools
echo - CPU PyTorch wheels are installed by default. Replace them with CUDA wheels inside .venv if needed.
echo.
echo Example:
echo   run.bat --source scrcpy --weights C:\path\to\weights.pt --device cuda
endlocal
