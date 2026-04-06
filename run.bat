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

"%VENV_PYTHON%" "%ROOT_DIR%live_feed.py" %DEFAULT_WEIGHTS_ARG% %*
endlocal
