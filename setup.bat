@echo off
setlocal
cd /d "%~dp0"

echo === Voice Vault Search: setup ===
echo Plugin folder: %CD%
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found on PATH.
  echo Install Python 3.10+ from https://python.org and check "Add Python to PATH" during install.
  pause
  exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
  echo [ERROR] Python 3.10+ required. Your version:
  python --version
  pause
  exit /b 1
)

if exist ".venv\Scripts\python.exe" (
  echo [info] .venv already exists. Reinstalling deps...
) else (
  echo [step 1/3] Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 ( echo [ERROR] venv creation failed & pause & exit /b 1 )
)

echo [step 2/3] Upgrading pip...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip --quiet

echo [step 3/3] Installing dependencies (~2 GB, may take 5-10 min on first run)...
python -m pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] dependency install failed & pause & exit /b 1 )

echo.
echo === Setup complete ===
echo.
echo Now:
echo   1. Reload Obsidian (Ctrl+R or restart)
echo   2. Open Voice Vault Search view via Ribbon icon
echo.
echo The plugin will auto-spawn the Python whisper server on load.
echo First run downloads faster-whisper large-v3-turbo (~1.6 GB) and is slow.
echo.
echo Optional: for CUDA GPU acceleration (10x faster), uninstall CPU torch and reinstall with:
echo   pip uninstall torch -y
echo   pip install torch --index-url https://download.pytorch.org/whl/cu121
echo.
pause
