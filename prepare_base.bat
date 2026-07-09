@echo off
setlocal
cd /d "%~dp0"
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"
chcp 65001 >nul

echo ============================================================
echo   Voice Vault Search - подготовка базы поиска
echo ============================================================
echo.
echo Разово генерирует блоки "Возможные вопросы" для заметок -
echo это сильно улучшает точность поиска. Whisper тут не нужен.
echo.
echo ВАЖНО: закрой Obsidian, чтобы освободить видеопамять под модель.
echo.
pause

echo.
echo [1/2] Проверяю модель qwen2.5:7b-instruct (при первом запуске скачает ~4.7 ГБ)...
where ollama >nul 2>nul || (
  echo.
  echo [!] Ollama не найдена. Установи её: https://ollama.com/download
  echo     После установки запусти этот файл ещё раз.
  echo.
  pause & exit /b 1
)
ollama pull qwen2.5:7b-instruct
if errorlevel 1 ( echo [!] Не удалось получить модель. & pause & exit /b 1 )

echo.
echo [2/2] Генерирую вопросы для заметок ^(разово, подожди несколько минут^)...
if not exist ".venv\Scripts\python.exe" ( echo [!] Сначала запусти setup.bat ^(нет .venv^). & pause & exit /b 1 )
".venv\Scripts\python.exe" generate_questions.py --apply

echo.
echo ============================================================
echo   Готово! Открой Obsidian и нажми Reindex в панели плагина,
echo   чтобы поиск подхватил новые вопросы.
echo ============================================================
echo.
pause
