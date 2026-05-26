"""
whisper_ws_server.py — local faster-whisper over WebSocket for Voice Vault Search.

Two streams:
  • Mic: client sends binary PCM frames, gets back JSON transcripts (request-response).
  • System: server captures default-speaker loopback (WASAPI via `soundcard`)
            in the background and pushes JSON transcripts to subscribed clients.

Binary frame from client (mic):
    uint32 LE  sample_rate
    uint32 LE  num_samples
    uint32 LE  flags (1 = enrollment recording)
    float32 LE * num_samples

JSON command from client (control):
    {"action":"subscribe","role":"system"}      — register as system-audio listener
    {"action":"enroll_user"}                    — next mic frame is enrollment, save as user embedding
    {"action":"clear_enrollment"}               — drop saved user embedding

Server response (mic transcription / system push):
    {"text":"...", "lang":"ru", "dur_ms":N, "source":"mic|system",
     "is_question":bool, "is_user":bool|null, "filtered":"" or "<dropped text>"}

Run:
    .venv\\Scripts\\python.exe whisper_ws_server.py
"""

import argparse
import asyncio
import json
import logging
import re
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import websockets
from faster_whisper import WhisperModel

# Optional dependencies (graceful degradation)
try:
    import soundcard as sc
    SOUNDCARD_OK = True
except Exception as _e:
    SOUNDCARD_OK = False
    _SOUNDCARD_IMPORT_ERR = _e

try:
    import os as _os
    _os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from speechbrain.inference import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    import torch
    SPEAKERID_OK = True
except Exception as _e:
    SPEAKERID_OK = False
    _SPEAKERID_IMPORT_ERR = _e
    LocalStrategy = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("whisper-ws")

HEADER_FMT = "<III"
HEADER_LEN = struct.calcsize(HEADER_FMT)
FLAG_ENROLLMENT = 1 << 0

INITIAL_PROMPT = (
    "DevOps собеседование на русском с английскими техническими терминами. "
    "Terraform, Ansible, Kubernetes, Docker, Helm, GitLab, Jenkins, Prometheus, Grafana, "
    "NGINX, PostgreSQL, Redis, AWS, Azure, GCP. Linux: inode, hardlink, symlink, fstab, "
    "LVM, ext4, systemd, GRUB, page cache. Network: TLS, BGP, OSPF, DNS, TCP, HTTP, HTTPS."
)

# ── Hallucination filter ─────────────────────────────────────────────────────

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


def has_repetition(text: str, ngram: int = 3) -> bool:
    if not text:
        return False
    words = re.findall(r"\w+", text.lower())
    if len(words) < ngram * 2:
        return False
    seen: dict[tuple, int] = {}
    for i in range(len(words) - ngram + 1):
        gram = tuple(words[i : i + ngram])
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] >= 3:
            return True
    return False


def is_hallucination(text: str) -> bool:
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


# ── Question detection ───────────────────────────────────────────────────────

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


def is_question(text: str) -> bool:
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


# ── Speaker identification ───────────────────────────────────────────────────


class SpeakerID:
    """Lazy-loaded: ECAPA model only downloaded when needed (first enroll or classify)."""

    def __init__(self, enroll_path: Path):
        self.enroll_path = enroll_path
        self.user_emb = None
        self.embedder = None
        self._load_attempted = False
        if not SPEAKERID_OK:
            log.warning(f"SpeakerID disabled — speechbrain not available: {_SPEAKERID_IMPORT_ERR}")
            return
        # If enrollment file exists, try to load it (but don't load model yet)
        if self.enroll_path.exists():
            try:
                arr = np.load(self.enroll_path)
                self.user_emb = torch.from_numpy(arr).float()
                log.info(f"SpeakerID: enrollment found at {self.enroll_path}")
            except Exception:
                log.exception("SpeakerID: failed to load enrollment file")
        log.info("SpeakerID: lazy mode — model downloads on first use")

    def _ensure_loaded(self) -> bool:
        if self.embedder is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        if not SPEAKERID_OK:
            return False
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"SpeakerID: loading ECAPA-TDNN on {device} (first use)")
        try:
            self.embedder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device},
                savedir=str(self.enroll_path.parent / "ecapa_cache"),
                local_strategy=LocalStrategy.COPY,
            )
            log.info("SpeakerID: model loaded")
            return True
        except Exception as e:
            log.warning(f"SpeakerID: model load failed: {e}")
            self.embedder = None
            return False

    def enroll(self, samples_16k: np.ndarray):
        if not self._ensure_loaded():
            return False
        if len(samples_16k) < 16000 * 5:
            log.warning(f"Enrollment too short ({len(samples_16k)/16000:.1f}s), need >=5s")
            return False
        wav = torch.from_numpy(samples_16k).unsqueeze(0)
        emb = self.embedder.encode_batch(wav).squeeze().cpu()
        emb = emb / (emb.norm() + 1e-9)
        self.user_emb = emb
        np.save(self.enroll_path, emb.numpy())
        log.info(f"Enrollment saved to {self.enroll_path}")
        return True

    def clear(self):
        self.user_emb = None
        if self.enroll_path.exists():
            self.enroll_path.unlink()
        log.info("Enrollment cleared")

    def is_user(self, samples_16k: np.ndarray, threshold: float = 0.45):
        """Return (bool, sim) if enrolled and model loaded, None otherwise."""
        if self.user_emb is None:
            return None
        if not self._ensure_loaded():
            return None
        if len(samples_16k) < 16000:
            return None
        wav = torch.from_numpy(samples_16k).unsqueeze(0)
        emb = self.embedder.encode_batch(wav).squeeze().cpu()
        emb = emb / (emb.norm() + 1e-9)
        sim = torch.dot(emb, self.user_emb).item()
        return sim > threshold, sim


# ── Transcriber ──────────────────────────────────────────────────────────────


class Transcriber:
    def __init__(self, model_name: str, device: str, compute_type: str, language: str):
        log.info(f"Loading faster-whisper model={model_name} device={device} compute_type={compute_type}")
        t0 = time.perf_counter()
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        log.info(f"Model loaded in {time.perf_counter()-t0:.2f}s")
        self.language = language

    def transcribe(self, samples: np.ndarray, sr: int) -> dict:
        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(len(samples) * ratio)
            xp = np.linspace(0, len(samples), num=len(samples), endpoint=False)
            x = np.linspace(0, len(samples), num=new_len, endpoint=False)
            samples = np.interp(x, xp, samples).astype(np.float32)
            sr = 16000
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            samples,
            language=self.language,
            beam_size=5,
            best_of=5,
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


# ── System audio capture loop ────────────────────────────────────────────────


def system_audio_capture_thread(out_queue: "queue.Queue", stop_event: threading.Event,
                                 sample_rate: int = 16000, chunk_seconds: float = 2.5):
    """Background thread: capture system loopback in 2.5s chunks, push to queue."""
    if not SOUNDCARD_OK:
        log.warning(f"System audio capture disabled (soundcard import: {_SOUNDCARD_IMPORT_ERR})")
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


# ── WebSocket server ─────────────────────────────────────────────────────────


class Server:
    def __init__(self, transcriber: Transcriber, speaker_id: SpeakerID):
        self.transcriber = transcriber
        self.speaker_id = speaker_id
        self.system_subscribers: set = set()
        self.lock = asyncio.Lock()
        self.enrollment_buffers: dict = {}  # ws -> list of np.ndarray
        self.enrollment_active: dict = {}   # ws -> bool

    async def broadcast_system(self, payload: dict):
        msg = json.dumps(payload)
        dead = []
        for ws in list(self.system_subscribers):
            try:
                await ws.send(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.system_subscribers.discard(ws)

    async def system_audio_loop(self, queue: "queue.Queue"):
        """Async consumer of system audio chunks. Transcribes and broadcasts."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                # Read chunk from thread queue (blocking, but in executor)
                mono = await loop.run_in_executor(None, queue.get)
                peak = float(np.abs(mono).max())
                rms = float(np.sqrt(np.mean(mono * mono)))
                if peak < 0.015 and rms < 0.005:
                    continue  # silence
                if not self.system_subscribers:
                    continue  # nobody listening, skip transcribe
                result = await loop.run_in_executor(
                    None, self.transcriber.transcribe, mono, 16000
                )
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
            self.enrollment_active.pop(ws, None)
            self.enrollment_buffers.pop(ws, None)
            log.info(f"client disconnected: {peer}")

    async def handle_command(self, ws, msg: str):
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
                log.info(f"client subscribed to system audio")
        elif action == "enroll_user":
            self.enrollment_active[ws] = True
            self.enrollment_buffers[ws] = []
            await ws.send(json.dumps({"ok": True, "enrollment": "recording"}))
            log.info("enrollment started for client")
        elif action == "finish_enrollment":
            if self.enrollment_active.get(ws):
                self.enrollment_active[ws] = False
                buf = self.enrollment_buffers.pop(ws, [])
                if buf:
                    merged = np.concatenate(buf)
                    log.info(f"enrollment finalize: {len(merged)/16000:.1f}s of audio")
                    if self.speaker_id:
                        ok = self.speaker_id.enroll(merged)
                        await ws.send(json.dumps({"ok": ok, "enrollment": "saved" if ok else "too_short"}))
                    else:
                        await ws.send(json.dumps({"ok": False, "error": "speaker_id unavailable"}))
                else:
                    await ws.send(json.dumps({"ok": False, "error": "no audio received"}))
            else:
                await ws.send(json.dumps({"ok": False, "error": "no active enrollment"}))
        elif action == "clear_enrollment":
            if self.speaker_id:
                self.speaker_id.clear()
            await ws.send(json.dumps({"ok": True, "enrollment": "cleared"}))
        elif action == "status":
            await ws.send(json.dumps({
                "ok": True,
                "soundcard": SOUNDCARD_OK,
                "speaker_id": SPEAKERID_OK and self.speaker_id is not None,
                "enrolled": self.speaker_id is not None and self.speaker_id.user_emb is not None,
                "system_subscribers": len(self.system_subscribers),
            }))
        else:
            await ws.send(json.dumps({"error": f"unknown action: {action}"}))

    async def handle_audio_frame(self, ws, frame: bytes):
        if len(frame) < HEADER_LEN:
            await ws.send(json.dumps({"error": "frame too short"}))
            return
        sr, n, flags = struct.unpack_from(HEADER_FMT, frame, 0)
        expected = HEADER_LEN + n * 4
        if len(frame) < expected:
            await ws.send(json.dumps({"error": f"frame truncated"}))
            return
        samples = np.frombuffer(frame, dtype=np.float32, offset=HEADER_LEN, count=n).copy()

        # Resample to 16k if needed
        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(len(samples) * ratio)
            xp = np.linspace(0, len(samples), num=len(samples), endpoint=False)
            x = np.linspace(0, len(samples), num=new_len, endpoint=False)
            samples = np.interp(x, xp, samples).astype(np.float32)

        # Enrollment mode: just collect, don't transcribe
        if self.enrollment_active.get(ws):
            self.enrollment_buffers.setdefault(ws, []).append(samples)
            total = sum(len(b) for b in self.enrollment_buffers[ws])
            await ws.send(json.dumps({"enrollment": "collecting", "seconds": round(total / 16000, 1)}))
            return

        # Normal mic transcription
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None, self.transcriber.transcribe, samples, 16000
            )
        except Exception as e:
            log.exception("transcribe failed")
            await ws.send(json.dumps({"error": str(e)}))
            return

        result["source"] = "mic"
        result["is_question"] = is_question(result.get("text", ""))

        # Speaker ID check
        if self.speaker_id and self.speaker_id.user_emb is not None:
            try:
                ret = self.speaker_id.is_user(samples)
                if ret is not None:
                    is_user, sim = ret
                    result["is_user"] = bool(is_user)
                    result["speaker_sim"] = round(sim, 3)
            except Exception:
                log.exception("speaker_id check failed")
        else:
            result["is_user"] = None

        log.info(f"[mic] {result['dur_ms']}ms is_q={result['is_question']} "
                 f"is_user={result.get('is_user')} text={result.get('text','')!r}")
        await ws.send(json.dumps(result))


# ── Main ────────────────────────────────────────────────────────────────────


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9876)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--no-system-audio", action="store_true",
                    help="Disable system audio loopback capture")
    ap.add_argument("--enrollment-path", default=None,
                    help="Path to save enrolled user voice embedding")
    args = ap.parse_args()

    transcriber = Transcriber(args.model, args.device, args.compute_type, args.language)

    if args.enrollment_path:
        enroll_path = Path(args.enrollment_path)
    else:
        enroll_path = Path(__file__).parent / "enrolled_user.npy"
    speaker_id = SpeakerID(enroll_path)

    server = Server(transcriber, speaker_id)

    # Start system audio capture thread + async consumer
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

    log.info(f"Listening on ws://{args.host}:{args.port}  "
             f"(soundcard={SOUNDCARD_OK}, speaker_id={SPEAKERID_OK})")

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
