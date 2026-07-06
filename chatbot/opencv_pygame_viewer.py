"""OpenCV + Pygame UI with face states and 8-frame capture button.

Usage:
    python chatbot/opencv_pygame_viewer.py
"""

import argparse
import importlib.util
import json
import os
import pickle
from pathlib import Path
import queue
import re
import socket
import struct
import sys
import time
import threading
from typing import Any
import warnings

import cv2
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    import pygame

from PIL import Image

# Resolve project root robustly from either repo root or notebooks/ cwd.
cwd = Path.cwd()
if (cwd / "src").exists():
    PROJECT_ROOT = cwd
elif (cwd.parent / "src").exists():
    PROJECT_ROOT = cwd.parent
else:
    PROJECT_ROOT = cwd.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_processing.prompts import get_prompt
from mistralai import Mistral

from pixtral_8b_vision_task2 import PixtralVisionTask2Pipeline, resolve_mistral_api_key

WINDOW_TITLE = "Camera Capture UI"
CAPTURE_TARGET = 8
TALK_ANIM_INTERVAL = 0.18

_ROUTER_MODEL: Any = None
_ROUTER_PROCESSOR: Any = None
_PIXTRAL_PIPELINE: PixtralVisionTask2Pipeline | None = None
_PIXTRAL_USER_PROMPT: str | None = None
_TTS_MODEL: Qwen3TTSModel | None = None
_TTS_VOICE_PROMPT: Any | None = None

TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TTS_LANGUAGE = "German"
TTS_STREAM_MIN_CHARS = 48
TTS_STREAM_MIN_WORDS = 40
TTS_STREAM_LOOKAHEAD_WORDS = 20
TTS_STREAM_MAX_BUFFER = 520
TTS_STREAM_QUEUE_END = object()

_TTS_STREAM_ACTIVE = threading.Event()
_TTS_STREAM_STOP: threading.Event | None = None
_TTS_STREAM_AUDIO_QUEUE: queue.Queue | None = None
_TTS_STREAM_THREADS: list[threading.Thread] = []
_TTS_STREAM_SUMMARY_CHANNEL: pygame.mixer.Channel | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show face states and capture 8 camera frames on click."
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1920,
        help="Camera resolution width (default: 1920)",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=1080,
        help="Camera resolution height (default: 1080)",
    )
    parser.add_argument(
        "--width", type=int, default=720, help="Window width (portrait)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1280,
        help="Window height (portrait)",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Mirror camera preview horizontally",
    )
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=0.5,
        help="Seconds between captured frames",
    )
    parser.add_argument(
        "--socket-host",
        type=str,
        default="",
        help="Optional camera stream host (for WSL->Windows bridge)",
    )
    parser.add_argument(
        "--socket-port",
        type=int,
        default=9999,
        help="Camera stream TCP port (default: 9999)",
    )
    return parser.parse_args()


def base_dir() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return base_dir().parent


def is_wsl() -> bool:
    """Return True when running inside Windows Subsystem for Linux."""
    return (
        os.name == "posix"
        and Path("/proc/version").exists()
        and "microsoft" in Path("/proc/version").read_text().lower()
    )


def detect_windows_host_for_wsl() -> str:
    """Try to detect Windows host IP from WSL resolver config."""
    resolv = Path("/etc/resolv.conf")
    if not resolv.exists():
        return ""
    try:
        content = resolv.read_text(encoding="utf-8", errors="ignore")
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "nameserver":
                return parts[1]
    except Exception:
        return ""
    return ""


def get_socket_host_candidates(explicit_host: str) -> list[str]:
    """Build prioritized list of socket camera hosts to try."""
    candidates: list[str] = []

    def add(host: str) -> None:
        host = host.strip()
        if host and host not in candidates:
            candidates.append(host)

    add(explicit_host)
    add(os.environ.get("WSL_WINDOWS_HOST", ""))

    if is_wsl():
        add(detect_windows_host_for_wsl())

    # Known working host from chatbot/camera.py in this project.
    add("172.23.96.1")

    return candidates


def play_sound(sound_path: Path) -> None:
    """Play a short wav sound without crashing when audio is unavailable."""
    try:
        if pygame.mixer.get_init() is None:
            pygame.mixer.init()
            pygame.mixer.set_num_channels(1)
        # Avoid two voices at the same time: stop ongoing playback first.
        pygame.mixer.stop()
        sound = pygame.mixer.Sound(str(sound_path))
        sound.play()
    except Exception:
        # Keep UI responsive even when audio backend is unavailable.
        pass


def clean_for_tts(text: str) -> str:
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ensure_tts_model() -> tuple[Qwen3TTSModel, Any | None]:
    global _TTS_MODEL
    global _TTS_VOICE_PROMPT

    if _TTS_MODEL is not None:
        return _TTS_MODEL, _TTS_VOICE_PROMPT

    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    model_kwargs: dict[str, Any] = {
        "device_map": device,
        "dtype": dtype,
    }

    use_flash_attn = use_cuda and os.environ.get("TTS_USE_FLASH_ATTN", "1") != "0"
    if use_flash_attn and importlib.util.find_spec("flash_attn"):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    elif use_cuda:
        model_kwargs["attn_implementation"] = "sdpa"

    try:
        _TTS_MODEL = Qwen3TTSModel.from_pretrained(TTS_MODEL_ID, **model_kwargs)
    except Exception as exc:
        if model_kwargs.get("attn_implementation") == "flash_attention_2":
            print(f"Warning: FlashAttention2 init failed, falling back to SDPA: {exc}")
            model_kwargs["attn_implementation"] = "sdpa"
            _TTS_MODEL = Qwen3TTSModel.from_pretrained(TTS_MODEL_ID, **model_kwargs)
        else:
            raise

    ref_audio = repo_root() / "src" / "agents" / "bmo_clear_reference.wav"
    if ref_audio.exists():
        _TTS_VOICE_PROMPT = _TTS_MODEL.create_voice_clone_prompt(
            ref_audio=str(ref_audio),
            x_vector_only_mode=True,
        )
    else:
        print(f"Warning: TTS reference audio not found: {ref_audio}")
        _TTS_VOICE_PROMPT = None

    return _TTS_MODEL, _TTS_VOICE_PROMPT


def synthesize_tts(text: str, output_path: Path) -> bool:
    clean = clean_for_tts(text)
    if not clean:
        return False

    try:
        tts_model, voice_prompt = ensure_tts_model()
    except Exception as exc:
        print(f"Warning: TTS init failed: {exc}")
        return False

    if voice_prompt is None:
        print("Warning: TTS voice prompt unavailable; skipping speech")
        return False

    try:
        wavs, sr = tts_model.generate_voice_clone(
            text=clean,
            language=TTS_LANGUAGE,
            voice_clone_prompt=voice_prompt,
        )
    except TypeError:
        wavs, sr = tts_model.generate_voice_clone(
            text=clean,
            language=TTS_LANGUAGE,
            voice_clone_prompt=voice_prompt,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, wavs[0], int(sr))
    return True


def _split_ready_tts_chunks(
    buffer: str,
    flush: bool = False,
) -> tuple[list[str], str]:
    chunks: list[str] = []

    abbreviations = {
        "pz",
        "dr",
        "mr",
        "mrs",
        "ms",
        "nr",
        "prof",
        "z.b",
        "u.a",
        "bzw",
        "usw",
        "ca",
    }

    def _is_sentence_break(text: str, idx: int) -> bool:
        before = text[:idx].rstrip()
        if not before:
            return False
        word_start = before.rfind(" ")
        word = before[word_start + 1 :]
        word_clean = word.rstrip(".!?").lower()
        if word_clean in abbreviations:
            return False
        after = text[idx + 1 :]
        next_char = re.search(r"\S", after)
        if next_char and next_char.group(0).isdigit() and len(word_clean) <= 3:
            return False
        return True

    def _find_sentence_break(start_idx: int, end_idx: int) -> int:
        for match in re.finditer(r"\.(?:\s|$)", buffer):
            if match.start() < start_idx:
                continue
            if match.start() > end_idx:
                break
            if _is_sentence_break(buffer, match.start()):
                return match.end()
        for match in re.finditer(r"[!?](?:\s|$)", buffer):
            if match.start() < start_idx:
                continue
            if match.start() > end_idx:
                break
            if _is_sentence_break(buffer, match.start()):
                return match.end()
        return -1

    while True:
        words = list(re.finditer(r"\S+", buffer))
        if not words:
            break

        if len(words) < TTS_STREAM_MIN_WORDS and not flush:
            break

        target_words = min(TTS_STREAM_MIN_WORDS, len(words))
        min_cut_idx = words[target_words - 1].end()
        max_word_idx = min(
            len(words) - 1,
            target_words - 1 + TTS_STREAM_LOOKAHEAD_WORDS,
        )
        max_cut_idx = words[max_word_idx].end()

        cut_idx = _find_sentence_break(min_cut_idx, max_cut_idx)

        if cut_idx < 0 and (flush or len(buffer) >= TTS_STREAM_MAX_BUFFER):
            extended_end = min(len(buffer), max(TTS_STREAM_MAX_BUFFER, min_cut_idx))
            cut_idx = _find_sentence_break(min_cut_idx, extended_end)

        if cut_idx < 0:
            if len(buffer) >= TTS_STREAM_MAX_BUFFER:
                cut_idx = min_cut_idx
            elif flush:
                cut_idx = len(buffer)
            else:
                break

        chunk = buffer[:cut_idx].strip()
        if chunk:
            chunks.append(chunk)
        buffer = buffer[cut_idx:].lstrip()

    if flush:
        tail = buffer.strip()
        if tail:
            chunks.append(tail)
        buffer = ""

    return chunks, buffer


def _play_tts_chunk(
    sound_path: Path,
    stop_event: threading.Event,
    channel: pygame.mixer.Channel | None = None,
) -> None:
    try:
        if pygame.mixer.get_init() is None:
            pygame.mixer.init()
        if pygame.mixer.get_num_channels() < 2:
            pygame.mixer.set_num_channels(2)
        sound = pygame.mixer.Sound(str(sound_path))
        if channel is None:
            channel = pygame.mixer.find_channel(True)
            if channel is None:
                return
        channel.play(sound)
        while channel.get_busy():
            if stop_event.is_set():
                channel.stop()
                break
            time.sleep(0.05)
    except Exception:
        pass


def _tts_playback_worker(
    audio_queue: queue.Queue,
    stop_event: threading.Event,
    tts_channel: pygame.mixer.Channel | None,
) -> None:
    try:
        while True:
            item = audio_queue.get()
            if item is TTS_STREAM_QUEUE_END:
                break
            if stop_event.is_set():
                break
            _play_tts_chunk(Path(item), stop_event, tts_channel)
    finally:
        _TTS_STREAM_ACTIVE.clear()


def _tts_generate_worker(
    chunks: list[str],
    output_dir: Path,
    audio_queue: queue.Queue,
    stop_event: threading.Event,
    summary_sound: Path | None,
) -> None:
    global _TTS_STREAM_SUMMARY_CHANNEL

    try:
        tts_model, voice_prompt = ensure_tts_model()
    except Exception as exc:
        print(f"Warning: TTS init failed: {exc}")
        audio_queue.put(TTS_STREAM_QUEUE_END)
        return

    if voice_prompt is None:
        print("Warning: TTS voice prompt unavailable; skipping speech")
        audio_queue.put(TTS_STREAM_QUEUE_END)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, chunk in enumerate(chunks):
        if stop_event.is_set():
            break
        try:
            wavs, sr = tts_model.generate_voice_clone(
                text=chunk,
                language=TTS_LANGUAGE,
                voice_clone_prompt=voice_prompt,
            )
        except TypeError:
            wavs, sr = tts_model.generate_voice_clone(
                text=chunk,
                language=TTS_LANGUAGE,
                voice_clone_prompt=voice_prompt,
            )
        except Exception as exc:
            print(f"Warning: TTS chunk failed: {exc}")
            continue

        chunk_path = output_dir / f"chunk_{idx:02d}.wav"
        sf.write(chunk_path, wavs[0], int(sr))
        if idx == 0 and _TTS_STREAM_SUMMARY_CHANNEL is not None:
            try:
                _TTS_STREAM_SUMMARY_CHANNEL.stop()
            except Exception:
                pass
            _TTS_STREAM_SUMMARY_CHANNEL = None
        audio_queue.put(chunk_path)

    audio_queue.put(TTS_STREAM_QUEUE_END)


def stop_tts_streaming() -> None:
    global _TTS_STREAM_STOP
    global _TTS_STREAM_AUDIO_QUEUE
    global _TTS_STREAM_THREADS
    global _TTS_STREAM_SUMMARY_CHANNEL

    if _TTS_STREAM_STOP is None:
        return

    _TTS_STREAM_STOP.set()
    if _TTS_STREAM_AUDIO_QUEUE is not None:
        try:
            _TTS_STREAM_AUDIO_QUEUE.put(TTS_STREAM_QUEUE_END, timeout=0.1)
        except Exception:
            pass

    for thread in _TTS_STREAM_THREADS:
        if thread.is_alive():
            thread.join(timeout=0.5)

    _TTS_STREAM_THREADS = []
    _TTS_STREAM_AUDIO_QUEUE = None
    _TTS_STREAM_STOP = None
    if _TTS_STREAM_SUMMARY_CHANNEL is not None:
        try:
            _TTS_STREAM_SUMMARY_CHANNEL.stop()
        except Exception:
            pass
    _TTS_STREAM_SUMMARY_CHANNEL = None
    _TTS_STREAM_ACTIVE.clear()

    try:
        if pygame.mixer.get_init() is not None:
            pygame.mixer.stop()
    except Exception:
        pass


def start_tts_streaming(
    text: str,
    output_root: Path,
    summary_sound: Path | None = None,
) -> bool:
    global _TTS_STREAM_STOP
    global _TTS_STREAM_AUDIO_QUEUE
    global _TTS_STREAM_THREADS
    global _TTS_STREAM_SUMMARY_CHANNEL

    stop_tts_streaming()

    clean = clean_for_tts(text)
    if not clean:
        return False

    chunks, _ = _split_ready_tts_chunks(clean, flush=True)
    if not chunks:
        return False

    stream_dir = output_root / f"stream_{time.strftime('%Y%m%d_%H%M%S')}"
    stop_event = threading.Event()
    audio_queue: queue.Queue = queue.Queue()

    _TTS_STREAM_STOP = stop_event
    _TTS_STREAM_AUDIO_QUEUE = audio_queue
    _TTS_STREAM_ACTIVE.set()
    _TTS_STREAM_SUMMARY_CHANNEL = None

    tts_channel = None
    try:
        if pygame.mixer.get_init() is not None:
            if pygame.mixer.get_num_channels() < 2:
                pygame.mixer.set_num_channels(2)
            tts_channel = pygame.mixer.Channel(1)
    except Exception:
        tts_channel = None

    if summary_sound is not None and summary_sound.exists():
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init()
            if pygame.mixer.get_num_channels() < 2:
                pygame.mixer.set_num_channels(2)
            _TTS_STREAM_SUMMARY_CHANNEL = pygame.mixer.Channel(0)
            _TTS_STREAM_SUMMARY_CHANNEL.play(pygame.mixer.Sound(str(summary_sound)))
        except Exception:
            _TTS_STREAM_SUMMARY_CHANNEL = None

    playback_thread = threading.Thread(
        target=_tts_playback_worker,
        args=(audio_queue, stop_event, tts_channel),
        daemon=True,
    )
    generator_thread = threading.Thread(
        target=_tts_generate_worker,
        args=(chunks, stream_dir, audio_queue, stop_event, summary_sound),
        daemon=True,
    )
    _TTS_STREAM_THREADS = [playback_thread, generator_thread]

    playback_thread.start()
    generator_thread.start()
    return True


def connect_socket_camera(
    host: str,
    port: int,
    verbose: bool = True,
) -> tuple[socket.socket, int] | None:
    """Connect to remote camera stream and return socket + header size."""
    if not host:
        return None
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(3.0)
        client.connect((host, port))
        client.settimeout(1.0)
        payload_size = struct.calcsize("Q")
        if verbose:
            print(f"Connected to socket camera: {host}:{port}")
        return client, payload_size
    except Exception as exc:
        if verbose:
            print(f"Warning: socket camera connect failed " f"({host}:{port}): {exc}")
        return None


def read_socket_frame(
    client: socket.socket,
    payload_size: int,
    buffer: bytes,
) -> tuple[np.ndarray | None, bytes, bool]:
    """Read one JPEG-encoded frame from the TCP camera stream."""
    try:
        while len(buffer) < payload_size:
            chunk = client.recv(4096)
            if not chunk:
                return None, buffer, False
            buffer += chunk

        msg_size = struct.unpack("Q", buffer[:payload_size])[0]
        buffer = buffer[payload_size:]

        while len(buffer) < msg_size:
            chunk = client.recv(4096)
            if not chunk:
                return None, buffer, False
            buffer += chunk

        frame_data = buffer[:msg_size]
        buffer = buffer[msg_size:]

        np_arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            return frame, buffer, True
        return None, buffer, True
    except socket.timeout:
        # Non-fatal: no full frame yet.
        return None, buffer, True
    except Exception:
        return None, buffer, False


def select_and_save_best_frame(
    frames_bgr: list[np.ndarray],
    threshold: int = 70,
) -> dict[str, Any]:
    """Use vision_routing_agent to score and save best frame."""
    if not frames_bgr:
        raise ValueError("No frames available for routing")

    global _ROUTER_MODEL
    global _ROUTER_PROCESSOR

    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.agents import vision_routing_agent as vra

    if _ROUTER_MODEL is None or _ROUTER_PROCESSOR is None:
        _ROUTER_MODEL, _ROUTER_PROCESSOR = vra.load_model(use_quantization=True)

    pil_images = [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames_bgr
    ]
    t0 = time.perf_counter()
    scored = vra.inspect_batch(
        pil_images,
        _ROUTER_MODEL,
        _ROUTER_PROCESSOR,
        debug=False,
    )
    total_inference_s = time.perf_counter() - t0
    per_frame_ms = (total_inference_s * 1000.0) / max(1, len(scored))

    out_dir = repo_root() / "data" / "processed" / "next_step"
    out_dir.mkdir(parents=True, exist_ok=True)
    inspector_out_dir = base_dir() / "agent_inspector_output"
    inspector_out_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    inspector_frames_dir = inspector_out_dir / f"frames_{stamp}"
    inspector_frames_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for idx, (score, reason) in enumerate(scored):
        label = f"frame_{idx:02d}"
        frame_path = inspector_frames_dir / f"{label}.jpg"
        cv2.imwrite(str(frame_path), frames_bgr[idx])
        results.append(
            {
                "frame": idx + 1,
                "label": label,
                "path": str(frame_path),
                "score": int(score),
                "accepted": int(score) >= int(threshold),
                "reason": str(reason),
                "elapsed_ms": round(per_frame_ms, 1),
            }
        )

    accepted_results = [r for r in results if r["accepted"]]
    best_result = max(accepted_results or results, key=lambda r: r["score"])
    best_index = int(best_result["frame"] - 1)
    best_frame = frames_bgr[best_index]
    timestamped_path = out_dir / f"best_frame_{stamp}.jpg"
    latest_path = out_dir / "best_frame_latest.jpg"
    meta_path = out_dir / "best_frame_latest.json"
    inspector_timestamped_json = inspector_out_dir / f"inspector_{stamp}.json"
    inspector_latest_json = inspector_out_dir / "inspector_latest.json"

    ok_timestamped = cv2.imwrite(str(timestamped_path), best_frame)
    ok_latest = cv2.imwrite(str(latest_path), best_frame)
    if not ok_timestamped or not ok_latest:
        raise RuntimeError("Failed to save selected best frame")

    metadata = {
        "saved_at": stamp,
        "timestamp": stamp,
        "frames_dir": str(inspector_frames_dir),
        "model_id": getattr(vra, "MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct"),
        "threshold": int(threshold),
        "quantization": True,
        "total_frames": len(results),
        "accepted_count": len(accepted_results),
        "rejected_count": len(results) - len(accepted_results),
        "best_frame": best_result,
        "batch_size": len(results),
        "timing": {
            "total_inference_s": round(total_inference_s, 3),
            "avg_per_frame_ms": round(per_frame_ms, 1),
        },
        "results": results,
        "selected": {
            "index": best_index,
            "score": int(best_result["score"]),
            "reason": str(best_result["reason"]),
            "accepted": bool(best_result["accepted"]),
        },
        "paths": {
            "timestamped": str(timestamped_path),
            "latest": str(latest_path),
        },
        "inspector_json": str(inspector_timestamped_json),
    }
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    meta_path.write_text(metadata_json, encoding="utf-8")
    inspector_timestamped_json.write_text(metadata_json, encoding="utf-8")
    inspector_latest_json.write_text(metadata_json, encoding="utf-8")

    return metadata


def ensure_pixtral_pipeline(prompt_name: str = "VLM_prompt_v2") -> None:
    global _PIXTRAL_PIPELINE
    global _PIXTRAL_USER_PROMPT

    if _PIXTRAL_PIPELINE is not None and _PIXTRAL_USER_PROMPT is not None:
        return

    api_key = resolve_mistral_api_key()
    prompt_template = get_prompt(prompt_name)
    user_prompt = (
        prompt_template["task"]
        .replace("Text:\n{text}", "")
        .replace("{text}", "")
        .strip()
    )
    system_prompt = prompt_template["system"]

    client = Mistral(api_key=api_key)
    _PIXTRAL_PIPELINE = PixtralVisionTask2Pipeline(
        client=client,
        system_prompt=system_prompt,
        model_name="pixtral-12b-2409",
        max_tokens=1024,
        top_p=1.0,
    )
    _PIXTRAL_USER_PROMPT = user_prompt


def run_pixtral_on_frame(frame_bgr: np.ndarray) -> str:
    ensure_pixtral_pipeline()
    if _PIXTRAL_PIPELINE is None or _PIXTRAL_USER_PROMPT is None:
        return ""

    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame for Pixtral")
    return _PIXTRAL_PIPELINE.process_image_bytes(
        encoded.tobytes(),
        user_prompt=_PIXTRAL_USER_PROMPT,
        mime_type="image/jpeg",
    )


def wrap_text(
    text: str, font: pygame.font.Font, max_width: int, max_lines: int
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if font.size(trial)[0] <= max_width:
            current = trial
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            return lines
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines


def load_face(path: Path, label: str) -> pygame.Surface:
    try:
        return pygame.image.load(str(path)).convert()
    except Exception:
        colors = {
            "idle": (60, 180, 120),
            "listening": (60, 120, 200),
            "thinking": (200, 160, 40),
            "speaking": (200, 80, 60),
        }
        surf = pygame.Surface((800, 450))
        surf.fill(colors.get(label, (70, 70, 70)))
        font = pygame.font.SysFont(None, 48)
        text = font.render(label.upper(), True, (255, 255, 255))
        surf.blit(text, text.get_rect(center=surf.get_rect().center))
        return surf


def load_faces() -> dict[str, pygame.Surface]:
    image_dir = base_dir() / "images"
    return {
        "idle": load_face(image_dir / "face_idle.jpg", "idle"),
        "listening": load_face(image_dir / "face_listening.jpg", "listening"),
        "thinking": load_face(image_dir / "face_thinking.jpg", "thinking"),
        "speaking": load_face(image_dir / "face_speaking.jpg", "speaking"),
    }


def frame_to_surface(frame_bgr: np.ndarray) -> pygame.Surface:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(
        rgb.tobytes(),
        (rgb.shape[1], rgb.shape[0]),
        "RGB",
    )


def make_no_camera_frame(
    width: int,
    height: int,
    message: str = "No camera detected",
) -> np.ndarray:
    """Create a fallback frame when camera input is unavailable."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (28, 28, 28)

    cv2.rectangle(
        frame,
        (20, 20),
        (max(21, width - 20), max(21, height - 20)),
        (70, 70, 70),
        2,
    )

    cv2.putText(
        frame,
        message,
        (30, max(45, height // 2 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Connect a camera and restart, or keep using UI mode",
        (30, max(75, height // 2 + 26)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    return frame


def fit_rect(src_size: tuple[int, int], target: pygame.Rect) -> pygame.Rect:
    src_w, src_h = src_size
    scale = min(target.width / src_w, target.height / src_h)
    draw_w = max(1, int(src_w * scale))
    draw_h = max(1, int(src_h * scale))
    x = target.x + (target.width - draw_w) // 2
    y = target.y + (target.height - draw_h) // 2
    return pygame.Rect(x, y, draw_w, draw_h)


def draw_button(
    screen: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    font: pygame.font.Font,
    enabled: bool,
) -> None:
    bg = (30, 170, 80) if enabled else (100, 100, 100)
    pygame.draw.rect(screen, bg, rect, border_radius=10)
    pygame.draw.rect(screen, (230, 230, 230), rect, width=2, border_radius=10)
    txt = font.render(label, True, (255, 255, 255))
    screen.blit(txt, txt.get_rect(center=rect.center))


def compute_layout(
    window_w: int, window_h: int
) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
    margin = 16
    face_h = int(window_h * 0.25)
    face_rect = pygame.Rect(margin, margin, window_w - 2 * margin, face_h)

    # Vertical preview box (portrait orientation)
    preview_top = face_rect.bottom + margin
    preview_w = int(window_w * 0.5)
    preview_h = int(window_h * 0.55)
    preview_x = (window_w - preview_w) // 2
    preview_rect = pygame.Rect(
        preview_x,
        preview_top,
        preview_w,
        preview_h,
    )

    button_w = 180
    button_h = 48
    button_x = (window_w - button_w) // 2
    button_y = preview_rect.bottom + margin
    button_rect = pygame.Rect(button_x, button_y, button_w, button_h)
    return face_rect, preview_rect, button_rect


def main() -> int:
    args = parse_args()

    socket_hosts = get_socket_host_candidates(args.socket_host)
    if is_wsl() and socket_hosts:
        print("WSL detected, socket host candidates: " + ", ".join(socket_hosts))

    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    screen = pygame.display.set_mode(
        (args.width, args.height),
        pygame.RESIZABLE,
    )
    status_font = pygame.font.SysFont(None, 30)
    button_font = pygame.font.SysFont(None, 32)
    hint_font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    sounds_dir = base_dir() / "sounds"
    app_start_sound = sounds_dir / "audio_app_start.wav"
    start_capture_sound = sounds_dir / "audio_start_capture.wav"
    end_capture_sound = sounds_dir / "audio_end_capture.wav"
    start_next_step_sound = sounds_dir / "audio_start.wav"
    no_best_frame_sound = sounds_dir / "audio_no_best_frame.wav"
    error_sound = sounds_dir / "audio_error.wav"
    summary_sound = sounds_dir / "audio_summary.wav"
    tts_stream_root = sounds_dir / "pixtral_tts_stream"

    play_sound(app_start_sound)

    faces = load_faces()
    state = "idle"
    captured_frames: list[np.ndarray] = []
    best_frame_metadata: dict[str, Any] | None = None
    best_frame_preview: np.ndarray | None = None
    pixtral_text: str = ""
    capture_active = False
    capture_countdown_until = 0.0
    last_capture_at = 0.0
    transition_at = 0.0

    cap: cv2.VideoCapture | None = None
    camera_available = False

    # In WSL, /dev/video devices are usually unavailable. Prefer socket camera.
    if is_wsl() and socket_hosts:
        print("WSL mode: skipping local camera and using socket camera first.")
    else:
        cap = cv2.VideoCapture(args.camera)
        if cap.isOpened():
            # Set camera resolution to 1920x1080
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
            print(f"Camera resolution set to {args.camera_width}x{args.camera_height}")
        camera_available = cap.isOpened()

    socket_client: socket.socket | None = None
    socket_payload_size: int | None = None
    socket_buffer = b""
    next_socket_retry = 0.0
    socket_host_index = 0

    if not camera_available:
        if cap is not None:
            print(
                f"Warning: Could not open camera index {args.camera}. "
                "Starting in no-camera mode."
            )
            play_sound(error_sound)
        if socket_hosts:
            host = socket_hosts[socket_host_index % len(socket_hosts)]
            connection = connect_socket_camera(
                host,
                args.socket_port,
            )
            if connection is not None:
                socket_client, socket_payload_size = connection
            else:
                socket_host_index += 1
                next_socket_retry = time.perf_counter() + 3.0

    try:
        running = True
        while running:
            now = time.perf_counter()
            window_w, window_h = screen.get_size()
            face_rect, preview_rect, button_rect = compute_layout(
                window_w,
                window_h,
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_ESCAPE,
                    pygame.K_q,
                ):
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if (
                        button_rect.collidepoint(event.pos)
                        and not capture_active
                        and capture_countdown_until == 0.0
                    ):
                        stop_tts_streaming()
                        play_sound(start_capture_sound)
                        captured_frames.clear()
                        best_frame_preview = None
                        capture_countdown_until = (
                            now + 5.0
                        )  # Wait 5 seconds before capturing
                        state = "listening"
                        last_capture_at = 0.0
                        transition_at = 0.0

            # Handle 5-second countdown before capturing
            if capture_countdown_until > 0 and now >= capture_countdown_until:
                capture_active = True
                capture_countdown_until = 0.0
                last_capture_at = now

            if camera_available:
                ok, frame = cap.read()
                if not ok:
                    camera_available = False
                    print("Warning: Camera feed lost. " "Switching to no-camera mode.")
                    play_sound(error_sound)
                    frame = make_no_camera_frame(1920, 1080)
                elif args.flip:
                    frame = cv2.flip(frame, 1)
                # Rotate frame to portrait orientation
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            else:
                frame = None

                if socket_client is None and socket_hosts and now >= next_socket_retry:
                    host = socket_hosts[socket_host_index % len(socket_hosts)]
                    connection = connect_socket_camera(
                        host,
                        args.socket_port,
                        verbose=False,
                    )
                    if connection is not None:
                        socket_client, socket_payload_size = connection
                        socket_buffer = b""
                        print(
                            f"Connected to socket camera: " f"{host}:{args.socket_port}"
                        )
                    else:
                        socket_host_index += 1
                        if int(now) % 6 == 0:
                            print(
                                "Waiting for socket camera on hosts: "
                                + ", ".join(socket_hosts)
                            )
                        next_socket_retry = now + 3.0

                if socket_client is not None and socket_payload_size is not None:
                    socket_frame, socket_buffer, socket_alive = read_socket_frame(
                        socket_client,
                        socket_payload_size,
                        socket_buffer,
                    )
                    if socket_frame is not None:
                        frame = socket_frame
                        if args.flip:
                            frame = cv2.flip(frame, 1)
                        # Rotate frame to portrait orientation
                        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif not socket_alive:
                        print("Warning: socket camera disconnected.")
                        play_sound(error_sound)
                        try:
                            socket_client.close()
                        except Exception:
                            pass
                        socket_client = None
                        socket_payload_size = None
                        socket_buffer = b""
                        next_socket_retry = now + 2.0

                if frame is None:
                    frame = make_no_camera_frame(1920, 1080)

            if capture_active and (now - last_capture_at) >= args.capture_interval:
                captured_frames.append(frame.copy())
                last_capture_at = now

                if len(captured_frames) >= CAPTURE_TARGET:
                    capture_active = False
                    state = "thinking"
                    transition_at = now
                    print(f"Captured {len(captured_frames)} frames in memory.")
                    play_sound(end_capture_sound)

                    try:
                        best_frame_metadata = select_and_save_best_frame(
                            captured_frames,
                            threshold=70,
                        )
                        selected = best_frame_metadata["selected"]
                        selected_index = int(selected["index"])
                        if 0 <= selected_index < len(captured_frames):
                            best_frame_preview = captured_frames[selected_index].copy()
                        print(
                            "Best frame selected: "
                            f"idx={selected['index']} "
                            f"score={selected['score']} "
                            f"accepted={selected['accepted']}"
                        )
                        if not selected.get("accepted"):
                            pixtral_text = ""
                            play_sound(no_best_frame_sound)
                            state = "idle"
                            transition_at = time.perf_counter()
                            continue
                        try:
                            if best_frame_preview is None:
                                raise ValueError("Best frame preview is unavailable")
                            pixtral_text = run_pixtral_on_frame(best_frame_preview)
                            if pixtral_text:
                                print(
                                    "Pixtral output (truncated): " + pixtral_text[:200]
                                )
                                if not start_tts_streaming(
                                    pixtral_text,
                                    tts_stream_root,
                                    summary_sound=summary_sound,
                                ):
                                    play_sound(summary_sound)
                        except Exception as exc:
                            pixtral_text = f"Pixtral error: {exc}"
                            print(pixtral_text)
                        if not pixtral_text:
                            play_sound(start_next_step_sound)
                        state = "speaking"
                        transition_at = time.perf_counter()
                    except Exception as exc:
                        print(f"Error during vision routing: {exc}")
                        play_sound(error_sound)
                        state = "idle"

            if not capture_active and state == "thinking":
                if (now - transition_at) >= 0.8:
                    state = "speaking"
                    transition_at = now
            elif not capture_active and state == "speaking":
                if not _TTS_STREAM_ACTIVE.is_set() and (now - transition_at) >= 0.8:
                    state = "idle"

            screen.fill((20, 20, 20))

            display_state = state
            if state == "speaking":
                speaking_elapsed = now - transition_at
                step = int(speaking_elapsed / TALK_ANIM_INTERVAL)
                display_state = "speaking" if step % 2 == 0 else "idle"

            face_img = faces.get(display_state, faces["idle"])
            face_draw = fit_rect(face_img.get_size(), face_rect)
            scaled_face = pygame.transform.smoothscale(
                face_img,
                face_draw.size,
            )
            screen.blit(scaled_face, face_draw.topleft)

            preview_frame = frame
            if (
                not capture_active
                and capture_countdown_until == 0.0
                and best_frame_preview is not None
            ):
                preview_frame = best_frame_preview

            frame_surface = frame_to_surface(preview_frame)
            preview_draw = fit_rect(frame_surface.get_size(), preview_rect)
            pygame.draw.rect(
                screen,
                (35, 35, 35),
                preview_rect,
                border_radius=10,
            )
            scaled_preview = pygame.transform.smoothscale(
                frame_surface, preview_draw.size
            )
            screen.blit(scaled_preview, preview_draw.topleft)
            pygame.draw.rect(
                screen,
                (200, 200, 200),
                preview_rect,
                width=1,
                border_radius=10,
            )

            if pixtral_text:
                text_margin = 10
                text_width = preview_rect.width - 2 * text_margin
                text_lines = wrap_text(
                    pixtral_text,
                    hint_font,
                    text_width,
                    max_lines=6,
                )
                if text_lines:
                    text_h = hint_font.get_linesize() * len(text_lines) + 10
                    overlay = pygame.Surface((text_width, text_h), pygame.SRCALPHA)
                    overlay.fill((10, 10, 10, 160))
                    screen.blit(
                        overlay,
                        (preview_rect.x + text_margin, preview_rect.y + text_margin),
                    )
                    y = preview_rect.y + text_margin + 5
                    for line in text_lines:
                        rendered = hint_font.render(line, True, (240, 240, 240))
                        screen.blit(rendered, (preview_rect.x + text_margin + 5, y))
                        y += hint_font.get_linesize()

            if capture_active:
                button_text = "Capturing..."
                button_enabled = False
            elif capture_countdown_until > 0:
                remaining = int(capture_countdown_until - now) + 1
                button_text = f"Wait {remaining}s"
                button_enabled = False
            else:
                button_text = "Start"
                button_enabled = True
            draw_button(
                screen,
                button_rect,
                button_text,
                button_font,
                enabled=button_enabled,
            )

            info = (
                f"State: {state} | Captured: "
                f"{len(captured_frames)}/{CAPTURE_TARGET}"
            )
            if best_frame_metadata is not None:
                selected = best_frame_metadata["selected"]
                info = (
                    f"{info} | Best idx={selected['index']} "
                    f"score={selected['score']}"
                )
            status_text = status_font.render(info, True, (240, 240, 240))
            hint_text = hint_font.render(
                "Q or ESC to quit",
                True,
                (200, 200, 200),
            )

            status_y = min(window_h - 56, button_rect.bottom + 10)
            hint_y = min(window_h - 28, button_rect.bottom + 34)
            screen.blit(status_text, (16, status_y))
            screen.blit(hint_text, (16, hint_y))

            pygame.display.flip()
            clock.tick(60)

    except Exception as exc:
        print(f"Unexpected error: {exc}")
        play_sound(error_sound)
        return 1
    finally:
        stop_tts_streaming()
        if cap is not None:
            cap.release()
        if socket_client is not None:
            try:
                socket_client.close()
            except Exception:
                pass
        pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
