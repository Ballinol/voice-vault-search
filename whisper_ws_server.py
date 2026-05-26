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
import re
import struct
import threading
import time

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
    "LVM, ext4, systemd, GRUB, page cache. Network: TLS, BGP, OSPF, DNS, TCP, HTTP, HTTPS."
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
        else:
            await ws.send(json.dumps({"error": f"unknown action: {action}"}))

    async def handle_audio_frame(self, ws, frame):
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
    args = ap.parse_args()

    transcriber = Transcriber(args.model, args.device, args.compute_type, args.language)
    server = Server(transcriber)

    import queue
    sysaudio_queue = queue.Queue(maxsize=20)
    stop_event = threading.Event()
    if not args.no_system_audio and SOUNDCARD_OK:
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
