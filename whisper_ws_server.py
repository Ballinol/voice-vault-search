"""
whisper_ws_server.py — local faster-whisper over WebSocket for Voice Vault Search.

Two streams:
  • Mic: client sends binary PCM frames, gets back JSON transcripts (request-response).
  • System: server captures default-speaker loopback (WASAPI via `soundcard`)
            in the background and pushes JSON transcripts to subscribed clients.

Binary frame from client (mic):
    uint32 LE  sample_rate
    uint32 LE  num_samples
    uint32 LE  flags (reserved, 0)
    float32 LE * num_samples

JSON command from client (control):
    {"action":"subscribe","role":"system"}      — register as system-audio listener
    {"action":"status"}                         — query server capabilities

Server response (mic transcription / system push):
    {"text":"...", "lang":"ru", "dur_ms":N, "source":"mic|system",
     "is_question":bool, "filtered":"" or "<dropped text>"}
"""

import argparse
import asyncio
import json
import logging
import os
import re
import struct
import threading
import time
from pathlib import Path

import numpy as np
import websockets
from faster_whisper import WhisperModel

try:
    import soundcard as sc
    SOUNDCARD_OK = True
except Exception as _e:
    SOUNDCARD_OK = False
    _SOUNDCARD_IMPORT_ERR = _e

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("whisper-ws")

HEADER_FMT = "<III"
HEADER_LEN = struct.calcsize(HEADER_FMT)

INITIAL_PROMPT = (
    "DevOps собеседование на русском с английскими техническими терминами. "
    "Terraform, Ansible, Kubernetes, Docker, Helm, GitLab, Jenkins, Prometheus, Grafana, "
    "NGINX, PostgreSQL, Redis, AWS, Azure, GCP. Linux: inode, hardlink, symlink, fstab, "
    "LVM, ext4, systemd, GRUB, page cache. Network: TLS, BGP, OSPF, DNS, TCP, HTTP, HTTPS. "
    "Русские термины: идемпотентность, идемпотентный, отказоустойчивость, "
    "согласованность, оркестрация, контейнеризация, троттлинг."
)

# Редкие русские доменные слова, которые Whisper систематически слышит криво
# ("идемпотентность" → "идемпатентность"/"и дед потентность"/"идентичность").
# hotwords — самый прицельный рычаг faster-whisper для смещения к ним; работает
# вместе с initial_prompt (отключается только при заданном prefix, его мы не задаём).
HOTWORDS = (
    "идемпотентность идемпотентный отказоустойчивость согласованность "
    "оркестрация контейнеризация троттлинг"
)

HALLUCINATION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"редактор\s+субтитров",
        r"корректор\s+[а-я]",
        r"субтитры\s+(подготовил|сделал|сделаны|подготовлены|перевёл|перевел|редактиров)",
        r"^\s*продолжение\s+следует\s*\.?\s*$",
        r"^\s*спасибо\s+за\s+(просмотр|внимание)\s*\.?\s*$",
        r"^\s*подпишись\b",
        r"^\s*ставьте\s+(лайк|лайки)\b",
        r"\bdimaTorzok\b",
        r"^\s*[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]+\s*(,|$)",
        r"transcribed\s+by",
    )
]


def has_repetition(text, ngram=3):
    if not text:
        return False
    words = re.findall(r"\w+", text.lower())
    if len(words) < ngram * 2:
        return False
    seen = {}
    for i in range(len(words) - ngram + 1):
        gram = tuple(words[i:i + ngram])
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] >= 3:
            return True
    return False


def is_hallucination(text):
    if not text:
        return False
    stripped = text.strip().rstrip(".,!?\"'»« ")
    if len(stripped) <= 1:
        return True
    for pat in HALLUCINATION_PATTERNS:
        if pat.search(text):
            return True
    if has_repetition(text):
        return True
    return False


RU_INTERROG = re.compile(
    r"^(что|как|почему|зачем|когда|где|кто|чем|какой|какая|какие|какое|"
    r"расскажи|опиши|объясни|знаешь|можешь|умеешь|есть\s+ли|был\s+ли|"
    r"давай\s+про|поясни|сравни|поговорим\s+про|поговорим\s+о)\b",
    re.IGNORECASE,
)
EN_INTERROG = re.compile(
    r"^(what|how|why|when|where|who|which|tell\s+me|explain|describe|"
    r"walk\s+me|compare|do\s+you|can\s+you|have\s+you|would\s+you|"
    r"could\s+you|show\s+me|let'?s\s+talk\s+about)\b",
    re.IGNORECASE,
)
LOOKUP_TRIGGERS = re.compile(
    r"(расскажи\s+про|объясни|что\s+такое|tell\s+me\s+about|"
    r"explain|what\s+is|how\s+does|давай\s+про|поговорим\s+про)",
    re.IGNORECASE,
)


def is_question(text):
    text = (text or "").strip()
    if not text:
        return False
    words = text.split()
    if not (2 <= len(words) <= 40):
        return False
    if text.endswith("?"):
        return True
    if RU_INTERROG.search(text) or EN_INTERROG.search(text):
        return True
    if LOOKUP_TRIGGERS.search(text):
        return True
    return False


def resolve_device_compute(device, compute_type):
    """Pick a sane device + compute_type. float16 on CPU is slow; use int8 instead."""
    has_cuda = False
    try:
        import ctranslate2
        has_cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        pass
    if device == "auto":
        device = "cuda" if has_cuda else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    # Guard: float16 on CPU is a footgun — downgrade to int8
    if device == "cpu" and compute_type in ("float16", "fp16"):
        log.warning("float16 on CPU is slow — switching to int8")
        compute_type = "int8"
    return device, compute_type


class Transcriber:
    def __init__(self, model_name, device, compute_type, language):
        device, compute_type = resolve_device_compute(device, compute_type)
        log.info(f"Loading faster-whisper model={model_name} device={device} compute_type={compute_type}")
        t0 = time.perf_counter()
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        log.info(f"Model loaded in {time.perf_counter()-t0:.2f}s")
        self.device = device
        self.language = language

    def transcribe(self, samples, sr):
        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(len(samples) * ratio)
            xp = np.linspace(0, len(samples), num=len(samples), endpoint=False)
            x = np.linspace(0, len(samples), num=new_len, endpoint=False)
            samples = np.interp(x, xp, samples).astype(np.float32)
        t0 = time.perf_counter()
        _beam = 5 if self.device == "cuda" else 1
        segments, info = self.model.transcribe(
            samples,
            language=self.language,
            beam_size=_beam,
            best_of=_beam,
            patience=1.0,
            length_penalty=1.0,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 200,
                "threshold": 0.5,
            },
            initial_prompt=INITIAL_PROMPT,
            hotwords=HOTWORDS,
            no_speech_threshold=0.5,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=1.8,
            hallucination_silence_threshold=1.0,
        )
        text = " ".join(s.text for s in segments).strip()
        dur_ms = int((time.perf_counter() - t0) * 1000)
        filtered = ""
        if text and is_hallucination(text):
            log.info(f"  drop hallucination: {text!r}")
            filtered = text
            text = ""
        return {"text": text, "lang": info.language, "dur_ms": dur_ms, "filtered": filtered}


def system_audio_capture_thread(out_queue, stop_event, sample_rate=16000, chunk_seconds=2.5):
    if not SOUNDCARD_OK:
        log.warning(f"System audio capture disabled (soundcard: {_SOUNDCARD_IMPORT_ERR})")
        return
    try:
        spk = sc.default_speaker()
        log.info(f"System audio: capturing loopback of '{spk.name}' (default speaker)")
        loopback_mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
        chunk_frames = int(sample_rate * chunk_seconds)
        with loopback_mic.recorder(samplerate=sample_rate, channels=1, blocksize=chunk_frames) as rec:
            while not stop_event.is_set():
                try:
                    audio = rec.record(numframes=chunk_frames)
                    mono = audio.flatten().astype(np.float32)
                    out_queue.put(mono)
                except Exception:
                    log.exception("system audio record() failed")
                    time.sleep(0.5)
    except Exception:
        log.exception("system audio capture thread crashed at startup")


# ── Cross-encoder reranker ───────────────────────────────────────────────────
# Ленивый: модель грузится при первом запросе rerank. По умолчанию на CPU,
# чтобы не делить 6 ГБ GPU с Whisper. Читает тела заметок по пути (от корня волта).
RERANK_MODEL  = os.getenv("RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
# GPU: реранк 20 коротких документов ~0.3с; рядом с Whisper (1.6+1.1 ГБ < 6 ГБ).
# На CPU та же модель — десятки секунд, для live не годится. При OOM загрузка
# упадёт → _get_reranker вернёт None → поиск тихо откатится на dense (e5).
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cuda")
_VAULT_ROOT   = Path(__file__).resolve().parents[3]   # .../voice-vault-search → Obsidian Vault
_RERANKER = None
_RERANKER_TRIED = False
_RE_HTML = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_IMG  = re.compile(r"!\[\[[^\]]*\]\]")
_RE_WS   = re.compile(r"\s+")


def _patch_xlmr_for_jina():
    try:
        import transformers.models.xlm_roberta.modeling_xlm_roberta as xlmr
    except Exception:
        return
    if not hasattr(xlmr, "create_position_ids_from_input_ids"):
        import torch
        def _f(input_ids, padding_idx, past_key_values_length=0):
            mask = input_ids.ne(padding_idx).int()
            incr = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
            return incr.long() + padding_idx
        xlmr.create_position_ids_from_input_ids = _f


def _get_reranker():
    global _RERANKER, _RERANKER_TRIED
    if _RERANKER is not None or _RERANKER_TRIED:
        return _RERANKER
    _RERANKER_TRIED = True
    try:
        _patch_xlmr_for_jina()
        from sentence_transformers import CrossEncoder
        log.info(f"Loading reranker {RERANK_MODEL} on {RERANK_DEVICE} (first use)")
        try:
            _RERANKER = CrossEncoder(RERANK_MODEL, max_length=512, device=RERANK_DEVICE,
                                     trust_remote_code=True, model_kwargs={"use_flash_attn": False})
        except TypeError:
            _RERANKER = CrossEncoder(RERANK_MODEL, max_length=512, device=RERANK_DEVICE,
                                     trust_remote_code=True)
        log.info("reranker loaded")
    except Exception as e:
        log.warning(f"reranker load failed: {e}")
        _RERANKER = None
    return _RERANKER


def _doc_text(path, heading, max_chars=1400):
    p = Path(path)
    if not p.is_absolute():
        p = _VAULT_ROOT / path
    try:
        raw = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    raw = _RE_IMG.sub(" ", _RE_HTML.sub(" ", raw))
    stem = p.stem
    label = f"{stem} > {heading}" if heading else stem
    return (label + ". " + _RE_WS.sub(" ", raw).strip())[:max_chars]


# ── Акцент на заголовок ───────────────────────────────────────────────────────
# Заголовки заметок у пользователя — это вопросы ("Как диагностировать ..."),
# сильнейший сигнал. К скору кросс-энкодера добавляем бонус за долю слов запроса,
# нашедшихся в заголовке (+heading). Морфологию РУ ловим грубо — по префиксу 5 букв.
RERANK_TITLE_BOOST = float(os.getenv("RERANK_TITLE_BOOST", "0.15"))
_RE_WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
_STOP = {
    "что", "такое", "как", "зачем", "почему", "когда", "где", "кто", "чем", "для",
    "это", "его", "или", "над", "под", "при", "про", "так", "там", "если", "ещё",
    "нужен", "нужна", "нужно", "бывают", "есть", "вообще", "может", "можно",
    "the", "and", "for", "what", "how", "why", "does", "are", "you", "your",
}


def _title_words(path, heading):
    stem = Path(path).stem
    return _RE_WORD.findall((stem + " " + (heading or "")).lower())


def _title_bonus(query, path, heading):
    qs = [w for w in _RE_WORD.findall(query.lower()) if len(w) >= 4 and w not in _STOP]
    if not qs:
        return 0.0
    tw = _title_words(path, heading)
    if not tw:
        return 0.0
    hit = 0
    for q in qs:
        qp = q[:5]
        if any(t.startswith(qp) or q.startswith(t[:5]) for t in tw):
            hit += 1
    return hit / len(qs)


# ── Извлечение вопроса из буфера (локальная LLM через Ollama) ──────────────────
# В живом собесе вопрос тонет в болтовне ("...Я сегодня сто процентов..."). Лёгкая
# модель (вход крошечный) вытаскивает сам вопрос ДО поиска, чтобы e5+реранк шли по
# чистой фразе. При недоступности Ollama — тихий откат на исходный буфер.
EXTRACT_ENABLED = os.getenv("EXTRACT_QUESTION", "1") != "0"
EXTRACT_MODEL   = os.getenv("EXTRACT_MODEL", "qwen2.5:1.5b")
# Нативный Ollama-эндпоинт — чтобы передать options (num_gpu) и keep_alive.
EXTRACT_URL     = os.getenv("EXTRACT_URL", "http://localhost:11434/api/chat")
EXTRACT_TIMEOUT = float(os.getenv("EXTRACT_TIMEOUT", "12.0"))  # запас на холодную загрузку (CPU медленнее)
# num_gpu=99 → извлекатель на GPU (тёплый ~1.5с vs ~2с CPU). Безопасно: Ollama —
# ОТДЕЛЬНЫЙ процесс, не конфликтует с Whisper; память Whisper1.6+qwen1.1+e5 0.5=3.2/6 ГБ.
# Если понадобится освободить VRAM — EXTRACT_NUM_GPU=0 вернёт извлекатель на CPU.
EXTRACT_NUM_GPU = int(os.getenv("EXTRACT_NUM_GPU", "99"))
_EXTRACT_SYSTEM = (
    "Из расшифровки устной речи (техническое собеседование) выдели заданный "
    "вопрос и верни его КРАТКО, сохраняя ключевые термины. НЕ отвечай на вопрос и "
    "НЕ добавляй никаких фактов — только сама тема/вопрос. Если явного вопроса "
    "нет — верни исходный текст без изменений."
)
# Few-shot: маленькие модели без примеров склонны ОТВЕЧАТЬ, а не извлекать.
_EXTRACT_SHOTS = [
    ("ну смотри я вот думаю когда деплоим как там с идемпотентностью в ансибле вообще",
     "идемпотентность в ansible"),
    ("Что такое namespace в Кубере? Я сегодня сто процентов поддерживаю. Такие я четыре раза нажал",
     "что такое namespace в kubernetes"),
    ("так вот про сетевую модель osi сколько там уровней а",
     "сколько уровней в модели osi"),
    ("ага понятно спасибо давай дальше",
     "ага понятно спасибо давай дальше"),
]


def _extract_question(query):
    q = (query or "").strip()
    if not (EXTRACT_ENABLED and q):
        return q
    try:
        import requests
        msgs = [{"role": "system", "content": _EXTRACT_SYSTEM}]
        for u, a in _EXTRACT_SHOTS:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": q})
        resp = requests.post(
            EXTRACT_URL,
            json={
                "model": EXTRACT_MODEL,
                "messages": msgs,
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": 48, "num_gpu": EXTRACT_NUM_GPU},
            },
            timeout=EXTRACT_TIMEOUT,
        )
        resp.raise_for_status()
        out = resp.json()["message"]["content"].strip().strip('"').strip()
        return out or q
    except Exception as e:
        log.warning(f"extract_question failed: {e}")
        return q


class Server:
    def __init__(self, transcriber):
        self.transcriber = transcriber
        self.system_subscribers = set()

    async def broadcast_system(self, payload):
        msg = json.dumps(payload)
        dead = []
        for ws in list(self.system_subscribers):
            try:
                await ws.send(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.system_subscribers.discard(ws)

    async def system_audio_loop(self, queue):
        loop = asyncio.get_running_loop()
        while True:
            try:
                mono = await loop.run_in_executor(None, queue.get)
                peak = float(np.abs(mono).max())
                rms = float(np.sqrt(np.mean(mono * mono)))
                if peak < 0.015 and rms < 0.005:
                    continue
                if not self.system_subscribers:
                    continue
                result = await loop.run_in_executor(None, self.transcriber.transcribe, mono, 16000)
                result["source"] = "system"
                result["is_question"] = is_question(result.get("text", ""))
                result["peak"] = round(peak, 4)
                result["rms"] = round(rms, 4)
                if not result.get("text"):
                    continue
                log.info(f"[system] {result['dur_ms']}ms is_q={result['is_question']} text={result['text']!r}")
                await self.broadcast_system(result)
            except Exception:
                log.exception("system audio consumer error")
                await asyncio.sleep(0.3)

    async def handle_connection(self, ws):
        peer = getattr(ws, "remote_address", "?")
        log.info(f"client connected: {peer}")
        try:
            async for msg in ws:
                if isinstance(msg, str):
                    await self.handle_command(ws, msg)
                else:
                    await self.handle_audio_frame(ws, msg)
        except websockets.ConnectionClosed:
            pass
        except Exception:
            log.exception("connection handler error")
        finally:
            self.system_subscribers.discard(ws)
            log.info(f"client disconnected: {peer}")

    async def handle_command(self, ws, msg):
        try:
            cmd = json.loads(msg)
        except json.JSONDecodeError:
            await ws.send(json.dumps({"error": "invalid JSON"}))
            return
        action = cmd.get("action")
        if action == "subscribe":
            role = cmd.get("role", "mic")
            if role == "system":
                self.system_subscribers.add(ws)
                await ws.send(json.dumps({"ok": True, "subscribed": "system"}))
                log.info("client subscribed to system audio")
        elif action == "status":
            await ws.send(json.dumps({
                "ok": True,
                "soundcard": SOUNDCARD_OK,
                "system_subscribers": len(self.system_subscribers),
            }))
        elif action == "rerank":
            await self._handle_rerank(ws, cmd)
        elif action == "extract_question":
            await self._handle_extract(ws, cmd)
        else:
            await ws.send(json.dumps({"error": f"unknown action: {action}"}))

    async def _handle_extract(self, ws, cmd):
        query = cmd.get("query") or ""
        loop = asyncio.get_running_loop()
        try:
            q = await loop.run_in_executor(None, _extract_question, query)
        except Exception:
            q = query
        await ws.send(json.dumps({"question": q}))

    async def _handle_rerank(self, ws, cmd):
        query = (cmd.get("query") or "").strip()
        items = cmd.get("items") or []
        rr = _get_reranker()
        if not (rr and query and items):
            await ws.send(json.dumps({"scores": None}))
            return
        loop = asyncio.get_running_loop()

        def _run():
            pairs = [[query, _doc_text(it.get("path", ""), it.get("heading", ""))] for it in items]
            ce = rr.predict(pairs, show_progress_bar=False)
            out = []
            for s, it in zip(ce, items):
                b = _title_bonus(query, it.get("path", ""), it.get("heading", ""))
                out.append(float(s) + RERANK_TITLE_BOOST * b)
            return out

        try:
            scores = await loop.run_in_executor(None, _run)
            await ws.send(json.dumps({"scores": scores}))
            log.info(f"reranked {len(items)} items")
        except Exception as e:
            log.warning(f"rerank failed: {e}")
            await ws.send(json.dumps({"scores": None}))

    async def handle_audio_frame(self, ws, frame):
        if self.transcriber is None:
            await ws.send(json.dumps({"error": "whisper disabled"}))
            return
        if len(frame) < HEADER_LEN:
            await ws.send(json.dumps({"error": "frame too short"}))
            return
        sr, n, _flags = struct.unpack_from(HEADER_FMT, frame, 0)
        expected = HEADER_LEN + n * 4
        if len(frame) < expected:
            await ws.send(json.dumps({"error": "frame truncated"}))
            return
        samples = np.frombuffer(frame, dtype=np.float32, offset=HEADER_LEN, count=n).copy()

        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(len(samples) * ratio)
            xp = np.linspace(0, len(samples), num=len(samples), endpoint=False)
            x = np.linspace(0, len(samples), num=new_len, endpoint=False)
            samples = np.interp(x, xp, samples).astype(np.float32)

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, self.transcriber.transcribe, samples, 16000)
        except Exception as e:
            log.exception("transcribe failed")
            await ws.send(json.dumps({"error": str(e)}))
            return

        result["source"] = "mic"
        result["is_question"] = is_question(result.get("text", ""))
        log.info(f"[mic] {result['dur_ms']}ms is_q={result['is_question']} text={result.get('text','')!r}")
        await ws.send(json.dumps(result))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9876)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--compute-type", default="auto")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--no-system-audio", action="store_true")
    ap.add_argument("--no-whisper", action="store_true",
                    help="Не грузить Whisper (STT через Deepgram); сидкар только для rerank/extract")
    args = ap.parse_args()

    if args.no_whisper:
        log.info("Whisper OFF (--no-whisper): сидкар обслуживает только rerank/extract")
        transcriber = None
    else:
        transcriber = Transcriber(args.model, args.device, args.compute_type, args.language)
    server = Server(transcriber)

    # Извлекатель вопроса работает через Ollama (ОТДЕЛЬНЫЙ процесс) — torch в этот
    # процесс не грузится, конфликта с ctranslate2-Whisper нет. Безопасно греть всегда.
    if EXTRACT_ENABLED:
        threading.Thread(target=lambda: _extract_question("прогрев модели"), daemon=True).start()
    # Реранкер грузит torch В ЭТОТ процесс. Рядом с Whisper на 6 ГБ это роняло
    # сидкар (CUDA-конфликт/OOM) → греем ТОЛЬКО когда Whisper выключен (--no-whisper)
    # или явно попросили (SIDECAR_PREWARM=1).
    if args.no_whisper or os.getenv("SIDECAR_PREWARM", "0") == "1":
        threading.Thread(target=_get_reranker, daemon=True).start()

    import queue
    sysaudio_queue = queue.Queue(maxsize=20)
    stop_event = threading.Event()
    if not args.no_system_audio and not args.no_whisper and SOUNDCARD_OK:
        thread = threading.Thread(
            target=system_audio_capture_thread,
            args=(sysaudio_queue, stop_event),
            daemon=True,
        )
        thread.start()
        asyncio.create_task(server.system_audio_loop(sysaudio_queue))

    log.info(f"Listening on ws://{args.host}:{args.port}  (soundcard={SOUNDCARD_OK})")

    async with websockets.serve(
        server.handle_connection,
        args.host,
        args.port,
        max_size=64 * 1024 * 1024,
    ):
        try:
            await asyncio.Future()
        finally:
            stop_event.set()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Stopped by user")
