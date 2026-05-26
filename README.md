# Voice Vault Search

Голосовой семантический поиск по Obsidian vault. Говоришь вопрос → плагин транскрибирует через локальный faster-whisper (CUDA/CPU) → ищет по vault через эмбеддинги → автоматически открывает релевантную заметку.

С dual-source захватом: системный звук (Zoom/Meet) → интервьюер, микрофон → ты. Auto-fire только на речь интервьюера.

## Установка

### 1. Поставь Python 3.10+

[python.org](https://www.python.org/downloads/). На Windows при установке поставь галочку **Add Python to PATH**.

### 2. Размести плагин в vault

Скопируй папку `voice-vault-search/` в `<твой-vault>/.obsidian/plugins/voice-vault-search/`. В Obsidian: Settings → Community plugins → Enable.

### 3. Запусти setup один раз

**Windows:** двойной клик на `setup.bat` (в папке плагина).
**Mac/Linux:** `bash setup.sh`.

Скрипт создаст `.venv` внутри папки плагина и поставит зависимости (~2 GB: faster-whisper, torch, soundcard, speechbrain). При первом запуске ещё скачается модель large-v3-turbo (~1.6 GB).

### 4. Перезагрузи Obsidian

Плагин автоматически запустит локальный WebSocket-сервер на `127.0.0.1:9876`. Готово.

## GPU acceleration (опционально)

По умолчанию ставится CPU-версия torch — работает, но инференс ~3-5 секунд на запрос. С NVIDIA GPU CUDA 12 ускорение в 10 раз (~300-600мс):

```
cd <plugin-folder>
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

На Mac (Apple Silicon) torch использует MPS автоматически, ничего ставить не надо.

## Использование

После reload плагина:

1. Открой Voice Vault Search view (ribbon icon).
2. Сразу слушает микрофон + system audio (системный звук).
3. Скажи вопрос — плагин найдёт и откроет лучшую заметку.

### Горячие клавиши

Дефолтные (можешь поменять в Settings → Hotkeys → ищи "Voice Vault"):

| Действие | Default |
|---|---|
| Open next result | `Ctrl + ]` |
| Open previous result | `Ctrl + [` |
| Open result #1..5 | `Ctrl + Shift + 1..5` |
| Force search on last transcript | `Ctrl + Shift + Space` |
| Enroll my voice (30s) | (не назначено) |
| Show server status | (не назначено) |

### Что слушать в логах

DevTools console (Ctrl+Shift+I):

- `[VVS-sys] heard: "..."` — интервьюер сказал что-то через system audio
- `[VVS-py-err] ... text=...` — сервер транскрибировал mic
- `[VVS] runSearch: "..."` — пошёл поиск
- `[VVS] 🎤 mic: your own voice, skip` — твой голос распознан (после enrollment), не ищем

## Конфигурация system audio

System audio loopback захватывает **default playback** Windows. Если интервьюер говорит через Zoom и звук идёт в твои наушники — плагин услышит. Если у тебя несколько устройств вывода, выбери правильное в:

**Windows:** Settings → System → Sound → Output → выбери куда реально идёт звук Zoom.
**Mac:** System Settings → Sound → Output.

Проверь в DevTools: `await window.vvsStatus()` покажет какие устройства активны.

## Архитектура

```
┌──────────────────┐         ┌──────────────────────────┐
│ Obsidian plugin  │ ws:9876 │ Python WebSocket server  │
│  (main.js)       │←───────→│  (whisper_ws_server.py)  │
│                  │         │                          │
│  - Mic capture   │         │  - faster-whisper turbo  │
│  - WS client #1  │  audio  │  - soundcard loopback    │
│  - WS client #2  │←────────│                          │
│    (system push) │ system  │  - Hallucination filter  │
│  - Search UI     │         │  - Question detection    │
└──────────────────┘         └──────────────────────────┘
```

## Troubleshooting

**Server не стартует:** проверь что setup.bat успешно прошёл и в папке плагина появилась `.venv/`. Открой DevTools console — должны быть логи `[VVS-py]`.

**"WebSocket failed":** сервер либо ещё грузится (первые ~10с), либо упал. Логи: `[VVS-py-err]`.

**Системный звук не ловит:** проверь `await window.vvsStatus()` — `soundcard: true`? Поменяй Default Playback в Windows на устройство куда реально идёт звук Zoom.

**Галлюцинации "Редактор субтитров":** сервер уже фильтрует. Если проскакивают новые паттерны — добавь regex в `HALLUCINATION_PATTERNS` в `whisper_ws_server.py`.

**Звук тихий, плагин скипает:** Settings → Sound → Input → Microphone → Properties → Levels → подними gain до 70-80% + Boost +10dB.
