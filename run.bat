@echo off
setlocal

set ROOT_DIR=%~dp0
set VENV_PYTHON=%ROOT_DIR%.venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
  echo .venv is missing. Run install_dependencies.bat first.
  exit /b 1
)

"%VENV_PYTHON%" "%ROOT_DIR%live_feed.py" %*
endlocal
