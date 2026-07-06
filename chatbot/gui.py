# ╔══════════════════════════════════════════════════════════════╗
# ║              BMO - AI Voice Assistant (Complete)             ║
# ║   SPACE → ابدأ التسجيل │ SPACE مرة ثانية → أوقف │ ESC → خروج ║
# ╚══════════════════════════════════════════════════════════════╝

import importlib.util
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import queue
import gc
from typing import Callable
from pathlib import Path
import ollama
import pygame
import sounddevice as sd
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
import numpy as np


class _PadTokenWarningFilter(logging.Filter):
    """Hide only the noisy pad_token_id auto-set warning from transformers."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "Setting `pad_token_id` to `eos_token_id`" not in msg


logging.getLogger("transformers.generation.utils").addFilter(_PadTokenWarningFilter())

# ─── Constants ───────────────────────────────────────────
APP_NAME = "BMO"


def get_resource_base_dir() -> Path:
    """Return folder where bundled assets are available."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def get_memory_file_path() -> Path:
    """Store memory in a user-writable location on every platform."""
    if os.name == "nt":
        appdata = Path(os.environ.get("APPDATA", str(Path.home())))
        memory_dir = appdata / APP_NAME
    else:
        memory_dir = Path.home() / ".bmo"
    memory_dir.mkdir(parents=True, exist_ok=True)
    return memory_dir / "memory.json"


RESOURCE_BASE_DIR = get_resource_base_dir()
MEMORY_FILE = get_memory_file_path()
REF_AUDIO = RESOURCE_BASE_DIR / "bmo_clear_reference.wav"
SCREEN_SIZE = (600, 900)
SAMPLE_RATE = 24000


def get_output_sample_rate(default_rate: int = 48000) -> int:
    """Return output sample rate and fall back safely if unavailable."""
    try:
        out_dev = sd.default.device[1]
        if out_dev is None or out_dev < 0:
            return default_rate
        info = sd.query_devices(out_dev, "output")
        return int(info.get("default_samplerate") or default_rate)
    except Exception:
        return default_rate


def get_input_sample_rate(preferred_rate: int = SAMPLE_RATE) -> int:
    """Return a supported input sample rate for the selected input device."""
    try:
        in_dev = sd.default.device[0]
        if in_dev is None or in_dev < 0:
            return preferred_rate

        # Keep the preferred rate when the device supports it.
        sd.check_input_settings(
            device=in_dev,
            samplerate=preferred_rate,
            channels=1,
            dtype="float32",
        )
        return preferred_rate
    except Exception:
        try:
            in_dev = sd.default.device[0]
            if in_dev is None or in_dev < 0:
                return preferred_rate
            info = sd.query_devices(in_dev, "input")
            return int(info.get("default_samplerate") or preferred_rate)
        except Exception:
            return preferred_rate


def get_default_output_device() -> int | None:
    """Return current default output device index if available."""
    try:
        default_out = sd.default.device[1]
        if default_out is not None and default_out >= 0:
            return int(default_out)
    except Exception:
        pass
    return None


def _first_token_id(value):
    """Return a scalar token id when config stores ids as list/tuple."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def set_pad_token_for_generation(tts) -> None:
    """Set pad_token_id to eos_token_id to avoid generation warnings."""
    candidates = [
        tts,
        getattr(tts, "model", None),
        getattr(tts, "llm", None),
    ]

    for candidate in candidates:
        if candidate is None:
            continue

        generation_cfg = getattr(candidate, "generation_config", None)
        model_cfg = getattr(candidate, "config", None)
        if generation_cfg is None:
            continue

        eos_id = _first_token_id(getattr(generation_cfg, "eos_token_id", None))
        if eos_id is None and model_cfg is not None:
            eos_id = _first_token_id(getattr(model_cfg, "eos_token_id", None))

        if eos_id is None:
            continue

        if getattr(generation_cfg, "pad_token_id", None) is None:
            generation_cfg.pad_token_id = eos_id


def get_generation_pad_token_id(tts) -> int | None:
    """Get the best available token id for generation padding."""
    candidates = [
        tts,
        getattr(tts, "model", None),
        getattr(tts, "llm", None),
    ]

    for candidate in candidates:
        if candidate is None:
            continue

        generation_cfg = getattr(candidate, "generation_config", None)
        model_cfg = getattr(candidate, "config", None)

        pad_id = _first_token_id(
            getattr(generation_cfg, "pad_token_id", None) if generation_cfg else None
        )
        if pad_id is not None:
            return pad_id

        eos_id = _first_token_id(
            getattr(generation_cfg, "eos_token_id", None) if generation_cfg else None
        )
        if eos_id is not None:
            return eos_id

        eos_id = _first_token_id(
            getattr(model_cfg, "eos_token_id", None) if model_cfg else None
        )
        if eos_id is not None:
            return eos_id

    return None


OUTPUT_DEVICE = get_default_output_device()
print(f"🔉 Output device (default): {OUTPUT_DEVICE}")

OUTPUT_SAMPLE_RATE = get_output_sample_rate()
RECORD_SAMPLE_RATE = get_input_sample_rate(SAMPLE_RATE)
print(f"🔈 Output sample rate: {OUTPUT_SAMPLE_RATE} Hz")
print(f"🎙️ Record sample rate: {RECORD_SAMPLE_RATE} Hz")
if RECORD_SAMPLE_RATE != SAMPLE_RATE:
    print(
        "⚠️ Mic does not support 24000 Hz directly; "
        f"using {RECORD_SAMPLE_RATE} Hz instead"
    )

sd.default.latency = "high"

# ─── Global State (State Machine) ────────────────────────
# States: "idle" | "recording" | "processing"
bmo_state = "idle"
current_face = "idle"
is_recording = False
audio_chunks = []
audio_stream = None
audio_playback_lock = threading.Lock()
SPEECH_QUEUE_END = object()
TTS_STREAM_MIN_CHARS = 48
TTS_STREAM_MIN_WORDS = 8
TTS_STREAM_MAX_BUFFER = 220
_is_shutting_down = False

# ─── 1. TTS: Load Once ────────────────────────────────────
print("🔊 تحميل نموذج الصوت...")
USE_CUDA = torch.cuda.is_available()
TTS_DEVICE = "cuda:0" if USE_CUDA else "cpu"
TTS_DTYPE = torch.float16 if USE_CUDA else torch.float32
print(f"🧠 TTS device: {TTS_DEVICE}")

tts_model_kwargs = {
    "device_map": TTS_DEVICE,
    "dtype": TTS_DTYPE,
}

use_flash_attn = USE_CUDA and os.environ.get("TTS_USE_FLASH_ATTN", "1") != "0"
if use_flash_attn and importlib.util.find_spec("flash_attn"):
    tts_model_kwargs["attn_implementation"] = "flash_attention_2"
    print("✅ FlashAttention enabled")
else:
    if use_flash_attn and USE_CUDA:
        print("⚠️ flash_attn غير مثبت — سيتم استخدام SDPA")
        print(
            "💡 تثبيت مقترح: pip install ninja packaging && "
            "MAX_JOBS=4 pip install flash-attn --no-build-isolation"
        )
    if USE_CUDA:
        tts_model_kwargs["attn_implementation"] = "sdpa"

tts_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    **tts_model_kwargs,
)

voice_prompt = None
if REF_AUDIO.exists():
    voice_prompt = tts_model.create_voice_clone_prompt(
        ref_audio=str(REF_AUDIO),
        x_vector_only_mode=True,
    )
    print(f"✅ صوت المرجع محمّل: {REF_AUDIO}")
else:
    print(f"⚠️  {REF_AUDIO} غير موجود — الصوت معطّل")

set_pad_token_for_generation(tts_model)

# ─── 2. Pygame: Setup & Faces ─────────────────────────────
# Initialize only video/font modules; keep audio to sounddevice.
pygame.display.init()
pygame.font.init()
if pygame.mixer.get_init() is not None:
    pygame.mixer.quit()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("BMO - AI Assistant")
clock = pygame.time.Clock()


def load_face(path: str, label: str) -> pygame.Surface:
    """Load image or create colored placeholder if file missing."""
    try:
        img = pygame.image.load(path)
        return pygame.transform.scale(img, SCREEN_SIZE)
    except Exception:
        colors = {
            "idle": (60, 180, 120),
            "listening": (60, 120, 200),
            "thinking": (200, 160, 40),
            "speaking": (200, 80, 60),
        }
        surf = pygame.Surface(SCREEN_SIZE)
        surf.fill(colors.get(label, (80, 80, 80)))
        font = pygame.font.SysFont("Arial", 32, bold=True)
        text = font.render(label.upper(), True, (255, 255, 255))
        surf.blit(text, text.get_rect(center=(200, 200)))
        return surf


def asset_path(*parts: str) -> str:
    return str(RESOURCE_BASE_DIR.joinpath(*parts))


faces = {
    "idle": load_face(asset_path("images", "face_idle.jpg"), "idle"),
    "listening": load_face(
        asset_path("images", "face_listening.jpg"),
        "listening",
    ),
    "thinking": load_face(
        asset_path("images", "face_thinking.jpg"),
        "thinking",
    ),
    "speaking": load_face(
        asset_path("images", "face_speaking.jpg"),
        "speaking",
    ),
}

status_font = pygame.font.SysFont("Arial", 18)


# ─── 3. Audio Helpers ──────────────────────────────────────
def play_system_sound(filename: str, blocking: bool = True) -> None:
    file_path = Path(filename)
    if not file_path.exists():
        return
    try:
        data, sr = sf.read(str(file_path), dtype="float32")
        _play_audio_array(data, int(sr), blocking=blocking)
    except Exception as exc:
        print(f"⚠️ خطأ في الصوت: {exc}")


def _play_audio_array(
    audio: np.ndarray,
    sr: int,
    blocking: bool = True,
) -> None:
    """Play audio as-is using its native sample rate and channel layout."""
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim > 2:
        audio = np.squeeze(audio)
        if audio.ndim > 2:
            raise ValueError("Unsupported audio shape for playback")

    audio = np.clip(audio, -1.0, 1.0)

    with audio_playback_lock:
        sd.stop()
        sd.play(audio, samplerate=int(sr), blocking=blocking)


def clean_for_tts(text: str) -> str:
    text = re.sub(r"\*[^*]+\*", "", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF.,!?؟،]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def speak(text: str) -> None:
    global current_face
    clean = clean_for_tts(text)
    if not clean or voice_prompt is None:
        return
    try:
        current_face = "speaking"
        gen_kwargs = {}
        pad_token_id = get_generation_pad_token_id(tts_model)
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id

        try:
            wavs, sr = tts_model.generate_voice_clone(
                text=clean,
                language="Auto",
                voice_clone_prompt=voice_prompt,
                **gen_kwargs,
            )
        except TypeError:
            wavs, sr = tts_model.generate_voice_clone(
                text=clean,
                language="Auto",
                voice_clone_prompt=voice_prompt,
            )
        _play_audio_array(wavs[0], int(sr))
    except Exception as exc:
        print(f"⚠️ خطأ في النطق: {exc}")
    finally:
        current_face = "thinking"


def _split_ready_tts_chunks(
    buffer: str,
    flush: bool = False,
) -> tuple[list[str], str]:
    """Extract speakable chunks from a streaming text buffer.

    Priority:
    1) New lines (first line can be spoken immediately)
    2) Sentence-ending punctuation
    """
    chunks: list[str] = []

    def _chunk_is_long_enough(text: str) -> bool:
        words = len(text.split())
        return len(text) >= TTS_STREAM_MIN_CHARS or words >= TTS_STREAM_MIN_WORDS

    while True:
        delimiters: list[int] = []

        newline_idx = buffer.find("\n")
        if newline_idx != -1:
            delimiters.append(newline_idx + 1)

        for punct_match in re.finditer(r"[.!?؟](?:\s|$)", buffer):
            delimiters.append(punct_match.end())

        if not delimiters:
            # Fallback for very long punctuation-free streams.
            if not flush and len(buffer) >= TTS_STREAM_MAX_BUFFER:
                split_at = buffer.rfind(" ", 0, TTS_STREAM_MAX_BUFFER)
                if split_at < 0:
                    split_at = TTS_STREAM_MAX_BUFFER
                chunk = buffer[:split_at].strip()
                if chunk:
                    chunks.append(chunk)
                buffer = buffer[split_at:].lstrip()
                continue
            break

        cut_idx = -1
        for candidate in sorted(set(delimiters)):
            candidate_chunk = buffer[:candidate].strip()
            if candidate_chunk and _chunk_is_long_enough(candidate_chunk):
                cut_idx = candidate
                break

        if cut_idx < 0:
            if not flush:
                break
            cut_idx = max(delimiters)

        chunk = buffer[:cut_idx].strip()
        if chunk:
            chunks.append(chunk)
        buffer = buffer[cut_idx:]

    if flush:
        tail = buffer.strip()
        if tail:
            chunks.append(tail)
        buffer = ""

    return chunks, buffer


def _speak_stream_worker(speech_queue: queue.Queue) -> None:
    """Speak chunks in order while LLM keeps streaming more text."""
    while True:
        item = speech_queue.get()
        if item is SPEECH_QUEUE_END:
            break
        speak(str(item))


# ─── 4. Memory ───────────────────────────────────────────────
def load_memory():
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    # ← هذا السطر يمنع تكرار مشكلة images
                    for msg in data:
                        msg.pop("images", None)
                    return data
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_memory(history: list) -> None:
    if len(history) > 10:
        history = history[-10:]
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ─── 5. LLM Streaming ────────────────────────────────────────
def query_bmo_stream(
    history: list,
    audio_path: str = None,
    on_tts_chunk: Callable[[str], None] | None = None,
) -> str:
    full_reply = ""
    tts_buffer = ""

    if audio_path and history:
        # Ollama يحتاج مسار مطلق وليس نسبياً
        history[-1]["images"] = [str(Path(audio_path).resolve())]

    try:
        for chunk in ollama.chat(
            model="bmo_model",
            messages=history,
            keep_alive=-1,
            stream=True,
            think=False,
        ):
            token = chunk["message"]["content"]
            full_reply += token
            tts_buffer += token
            print(token, end="", flush=True)

            if on_tts_chunk is not None:
                ready_chunks, tts_buffer = _split_ready_tts_chunks(tts_buffer)
                for ready in ready_chunks:
                    on_tts_chunk(ready)
        print()
    except Exception as exc:
        print(f"\n⚠️ خطأ LLM: {exc}")
    finally:
        if on_tts_chunk is not None:
            ready_chunks, _ = _split_ready_tts_chunks(tts_buffer, flush=True)
            for ready in ready_chunks:
                on_tts_chunk(ready)

    return full_reply.strip()


# ─── 6. Recording ────────────────────────────────────────────
def start_recording() -> None:
    global audio_chunks, audio_stream, is_recording
    audio_chunks = []
    is_recording = True
    audio_stream = sd.InputStream(
        samplerate=RECORD_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=1024,
        latency="high",
    )
    audio_stream.start()

    def _capture():
        while is_recording:
            chunk, _ = audio_stream.read(RECORD_SAMPLE_RATE // 10)
            audio_chunks.append(chunk.copy())

    threading.Thread(target=_capture, daemon=True).start()
    print("🔴 جارٍ التسجيل... اضغط SPACE للإيقاف")


def stop_recording() -> str | None:
    global is_recording, audio_stream
    is_recording = False
    time.sleep(0.15)
    if audio_stream:
        audio_stream.stop()
        audio_stream.close()
        audio_stream = None
    print("✅ انتهى التسجيل")

    if not audio_chunks:
        return None

    audio = np.concatenate(audio_chunks).squeeze()

    # Use OS temp dir so this works on Windows and Linux.
    fd, audio_path = tempfile.mkstemp(prefix="bmo_input_", suffix=".wav")
    os.close(fd)
    sf.write(audio_path, audio, RECORD_SAMPLE_RATE)
    print(f"📁 الملف محفوظ: {audio_path}")
    return audio_path


# ─── 7. Processing Thread ──────────────────────────────────
def process_audio(audio_path: str) -> None:
    global current_face, bmo_state

    current_face = "thinking"
    # Play "thinking" cue in the background so model work starts immediately.
    play_system_sound(
        asset_path("sounds", "audio_thinking.wav"),
        blocking=False,
    )

    history = load_memory()
    history.append({"role": "user", "content": "🎤"})

    speech_queue: queue.Queue = queue.Queue()
    speaker_thread = threading.Thread(
        target=_speak_stream_worker,
        args=(speech_queue,),
        daemon=True,
    )
    speaker_thread.start()

    print("\n🤖 BMO: ", end="", flush=True)
    bmo_reply = query_bmo_stream(
        history,
        audio_path=audio_path,
        on_tts_chunk=speech_queue.put,
    )

    speech_queue.put(SPEECH_QUEUE_END)
    speaker_thread.join()

    if bmo_reply:
        history.append({"role": "assistant", "content": bmo_reply})
        save_memory(history)

    try:
        Path(audio_path).unlink()
    except Exception:
        pass

    current_face = "idle"
    bmo_state = "idle"
    print("\n[اضغط SPACE للتحدث]")


def cleanup_runtime() -> None:
    """Release runtime resources, including GPU memory, on exit/abort."""
    global _is_shutting_down, is_recording, audio_stream, tts_model

    if _is_shutting_down:
        return
    _is_shutting_down = True

    is_recording = False
    try:
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
            audio_stream = None
    except Exception:
        pass

    try:
        sd.stop()
    except Exception:
        pass

    try:
        pygame.quit()
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            tts_model = None
        except Exception:
            pass

        gc.collect()
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            print("🧹 CUDA VRAM cache cleared")
        except Exception as exc:
            print(f"⚠️ CUDA cleanup warning: {exc}")


# ─── 8. Main Loop ─────────────────────────────────────────────
print("\n🤖 تشغيل نظام BMO...")
print("[اضغط SPACE للتحدث │ ESC للخروج]")

try:
    while True:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                raise SystemExit

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    print("\n🤖 BMO: إلى اللقاء يا بشار!")
                    raise SystemExit

                if event.key == pygame.K_SPACE:
                    if bmo_state == "idle":
                        bmo_state = "recording"
                        current_face = "listening"
                        play_system_sound(asset_path("sounds", "audio_start.wav"))
                        start_recording()

                    elif bmo_state == "recording":
                        bmo_state = "processing"
                        audio_path = stop_recording()
                        if audio_path:
                            threading.Thread(
                                target=process_audio,
                                args=(audio_path,),
                                daemon=True,
                            ).start()
                        else:
                            bmo_state = "idle"
                            current_face = "idle"
                    # processing state → SPACE مُهمَل (مشغول)

        # ── Draw ──────────────────────────────────────────────
        screen.blit(faces[current_face], (0, 0))

        bar = pygame.Surface((400, 30), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 160))
        screen.blit(bar, (0, 370))
        status_text = {
            "idle": "SPACE: ابدأ التحدث ▶",
            "recording": "SPACE: أوقف التسجيل  🔴",
            "processing": "BMO يفكر...  ⏳",
        }
        label = status_font.render(
            status_text.get(bmo_state, ""),
            True,
            (255, 255, 255),
        )
        screen.blit(label, label.get_rect(center=(200, 385)))

        pygame.display.flip()
        clock.tick(30)
except KeyboardInterrupt:
    print("\n⛔️ Interrupted by user")
finally:
    cleanup_runtime()
