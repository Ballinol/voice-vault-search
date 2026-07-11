@echo off
setlocal
cd /d "%~dp0"
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"

echo ============================================================
echo   Voice Vault Search - prepare search base
echo ============================================================
echo.
echo One-time step: generates "Possible questions" blocks for your
echo notes (greatly improves search accuracy). Whisper is not needed.
echo.
echo IMPORTANT: close Obsidian to free video memory for the model.
echo.
pause

echo.
echo [1/2] Checking Ollama...
where ollama >nul 2>nul && goto have_ollama
echo Ollama not found - trying to install it automatically (winget)...
winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements --disable-interactivity
echo Waiting for Ollama to start...
timeout /t 8 >nul
where ollama >nul 2>nul && goto have_ollama
echo.
echo [!] Could not install Ollama automatically.
echo     Install it from https://ollama.com/download and run this file again.
echo.
pause
exit /b 1
:have_ollama

echo Downloading models (first run: ~1 GB + ~4.7 GB)...
echo   - qwen2.5:1.5b  (extracts the question from speech during search)
ollama pull qwen2.5:1.5b
if errorlevel 1 ( echo [!] Failed to download qwen2.5:1.5b & pause & exit /b 1 )
echo   - qwen2.5:7b-instruct  (generates questions for notes)
ollama pull qwen2.5:7b-instruct
if errorlevel 1 ( echo [!] Failed to download qwen2.5:7b-instruct & pause & exit /b 1 )

echo.
echo [2/2] Generating questions for notes (one-time, please wait a few minutes)...
if not exist ".venv\Scripts\python.exe" ( echo [!] Run setup.bat first ^(no .venv^). & pause & exit /b 1 )
".venv\Scripts\python.exe" generate_questions.py --apply

echo.
echo ============================================================
echo   Done! Open Obsidian and press Reindex in the plugin panel.
echo ============================================================
echo.
pause
