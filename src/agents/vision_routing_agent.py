"""
vision_routing_agent.py
-----------------------
VLM-based image quality router for the nursing-robot OCR pipeline.

Loads Qwen3-VL-4B-Instruct, scores every frame in a directory (0–100),
selects the best frame for downstream OCR, saves results and plots.

Usage
-----
    python src/agents/vision_routing_agent.py
    python src/agents/vision_routing_agent.py --frames-dir data/synthetic/output/robot_frames
    python src/agents/vision_routing_agent.py --frames-dir path/to/frames --threshold 60 --no-quant

Outputs (inside data/processed/vision_routing_agent/)
------
    results/<timestamp>_results.json   — scores, reasons, best frame path
    plots/<timestamp>_scores.png       — color-coded score grid
    plots/<timestamp>_preview.png      — raw frame preview grid
"""

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Resolve project root (script lives in src/agents/, root is two levels up) ───
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

# ── Defaults ────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_FRAMES = REPO_ROOT / "data/synthetic/output/robot_frames"
OUTPUT_BASE = REPO_ROOT / "data/processed/vision_routing_agent"
ACCEPT_THRESHOLD = 70
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".avif", ".webp", ".bmp", ".tiff"}

SYSTEM_PROMPT = """Du bist ein strenger visueller Qualitätsprüfer für einen Pflegeroboter.
Deine Aufgabe ist es, die Qualität eines Bildes für OCR-Textextraktion zu bewerten.
Bewerte das Bild auf einer Skala von 0 bis 100:
- 100 = perfekt für OCR geeignet
-  80 = gut, kleine Mängel (leichte Unschärfe, minimale Schrägstellung)
-  60 = akzeptabel, aber mit Einschränkungen
-  40 = schlecht, OCR wird fehlerhaft sein
-  20 = sehr schlecht, kaum lesbar
-   0 = völlig unbrauchbar (kein Papier, komplett unscharf, extrem dunkel)
Abzüge (STRENG beachten):
- Unschärfe oder Verwacklung: -20 bis -60 je nach Stärke
- Papier gedreht / schräg: -20 bis -40
- Zu dunkel oder überbelichtet: -20 bis -50
- Kein Papier erkennbar: -100 (Score = 0)
HALBSEITIGE oder ABGESCHNITTENE BILDER (sehr hohe Abzüge):
- Nur die obere oder untere Hälfte des Papiers sichtbar: -60 bis -80
- Nur die linke oder rechte Hälfte des Papiers sichtbar: -60 bis -80
- Papierrand an einer Seite abgeschnitten: -30 bis -50
- Weniger als 70% des Papiers sichtbar: Score MAXIMAL 30
- Weniger als 50% des Papiers sichtbar: Score MAXIMAL 15
- Kein vollständiges Papier erkennbar: Score = 0
Ein Bild mit nur halber Seite darf NIEMALS mehr als 25 Punkte erhalten.
Antworte IMMER mit einem gültigen JSON-Objekt (keine weiteren Zeichen davor oder danach):
{"score": <Zahl 0-100>, "reason": "<kurze Begründung auf Deutsch, max. 8 Wörter>"}
Beispiele:
{"score": 95, "reason": "Papier vollständig sichtbar, scharf und gut beleuchtet"}
{"score": 40, "reason": "Bild leicht unscharf und schwach beleuchtet"}
{"score": 20, "reason": "Papier nur zur Hälfte sichtbar, rechte Seite fehlt"}
{"score": 10, "reason": "Nur obere Hälfte des Papiers sichtbar"}
{"score": 0, "reason": "Kein Papier erkennbar"}"""


# ── Model loading ────────────────────────────────────────────────────────────


def load_model(use_quantization: bool = True):
    """Load Qwen3-VL-4B with optional 4-bit NF4 quantization."""
    print(f"Loading model: {MODEL_ID}")
    print(f"Quantization : {'4-bit NF4' if use_quantization else 'full bf16'}\n")

    quantization_config = None
    torch_dtype = torch.bfloat16

    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        torch_dtype = None  # dtype is set inside BitsAndBytesConfig

    t0 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Decoder-only generation should use left padding in batched mode.
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    elapsed = time.perf_counter() - t0

    print(f"Model loaded in {elapsed:.1f}s")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"VRAM — allocated: {alloc:.2f} GB  |  reserved: {reserved:.2f} GB")
    print()

    return model, processor


# ── Inference ────────────────────────────────────────────────────────────────


def _build_messages(image: Image.Image) -> list[dict]:
    """Build the chat messages list for a single image."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "max_pixels": 1003520,  # ~720p — sufficient for quality routing
                },
                {
                    "type": "text",
                    "text": 'Bewerte dieses Bild. Antworte NUR mit JSON: {"score": <0-100>, "reason": "<Grund>"}',
                },
            ],
        },
    ]


def _parse_raw(raw: str) -> tuple[int, str]:
    """Parse JSON model output into (score, reason), with regex fallback."""
    score = 0
    reason = raw.strip() or "Keine gueltige JSON-Antwort"
    try:
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            score = max(0, min(100, int(data.get("score", 0))))
            reason = str(data.get("reason", reason)).strip() or reason
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"\b(\d{1,3})\b", raw)
        if m:
            score = max(0, min(100, int(m.group(1))))
    return score, reason


def inspect_batch(
    images: list[Image.Image],
    model,
    processor,
    debug: bool = False,
) -> list[tuple[int, str]]:
    """
    Score a batch of images for OCR suitability in a single forward pass.

    Returns
    -------
    List of (score, reason) tuples, one per image.
    """
    batch_messages = [_build_messages(img) for img in images]

    # Step 1: format each text prompt separately
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_messages
    ]

    # Step 2: extract pixel values from each message, then concatenate
    all_image_inputs: list = []
    all_video_inputs: list = []
    for msgs in batch_messages:
        img_inp, vid_inp = process_vision_info(msgs)
        if img_inp:
            all_image_inputs.extend(img_inp)
        if vid_inp:
            all_video_inputs.extend(vid_inp)

    # Step 3: combine into batched input tensors (processor handles padding)
    inputs = processor(
        text=texts,
        images=all_image_inputs if all_image_inputs else None,
        videos=all_video_inputs if all_video_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            # Avoid warnings when generation config carries sampling params.
            temperature=None,
            top_p=None,
            top_k=None,
        )

    new_tokens = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raws = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    results = []
    for i, raw in enumerate(raws):
        raw = raw.strip()
        if debug:
            print(f"  [raw batch[{i}]]: {raw!r}")
        results.append(_parse_raw(raw))

    return results


def image_quality_inspector(
    image: Image.Image,
    model,
    processor,
    debug: bool = False,
) -> tuple[int, str]:
    """Convenience wrapper: score a single image (calls inspect_batch internally)."""
    return inspect_batch([image], model, processor, debug=debug)[0]


# ── Plots ────────────────────────────────────────────────────────────────────


def save_preview_plot(frame_paths: list[Path], out_path: Path) -> None:
    """Save a grid preview of all raw frames."""
    n = len(frame_paths)
    cols = min(5, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, fp in zip(axes, frame_paths):
        img = Image.open(fp).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"{fp.stem}\n{img.size[0]}×{img.size[1]}px", fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Frame Preview (full resolution)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Preview plot saved → {out_path.relative_to(REPO_ROOT)}")


def save_scores_plot(
    results: list[dict],
    best: dict | None,
    threshold: int,
    out_path: Path,
) -> None:
    """Save color-coded score grid (gold=best, green=accepted, red=rejected)."""
    n = len(results)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 5.5))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, r in zip(axes, results):
        img_data = mpimg.imread(r["path"])
        ax.imshow(img_data)

        is_best = best is not None and r is best
        if is_best:
            color = "gold"
            status = f"★ BEST  {r['score']}/100"
        elif r["accepted"]:
            color = "green"
            status = f"✓ {r['score']}/100"
        else:
            color = "red"
            status = f"✗ {r['score']}/100"

        reason_str = r.get("reason", "")
        if len(reason_str) > 28:
            mid = reason_str.rfind(" ", 0, 28)
            reason_str = (
                reason_str[:mid] + "\n" + reason_str[mid + 1 :]
                if mid != -1
                else reason_str
            )

        ax.set_title(
            f"{r['label']}\n{status}\n{reason_str}",
            fontsize=8,
            color=color,
            fontweight="bold",
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle(
        f"VLM Quality Scores  (threshold ≥{threshold})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Scores plot  saved → {out_path.relative_to(REPO_ROOT)}")


# ── Main ─────────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output directories
    plots_dir = OUTPUT_BASE / "plots"
    results_dir = OUTPUT_BASE / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Collect frames
    frames_dir = Path(args.frames_dir).resolve()
    frame_paths = sorted(
        (p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: (
            int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1
        ),
    )
    if not frame_paths:
        print(f"No images found in: {frames_dir}")
        sys.exit(1)

    print(f"Found {len(frame_paths)} frames in: {frames_dir}")
    print(f"Batch size   : {args.batch_size}\n")

    # Load model
    model, processor = load_model(use_quantization=not args.no_quant)

    # Inspect frames in batches
    results: list[dict] = []
    total_inference_time = 0.0
    global_idx = 0

    # Chunk frame_paths into batches
    batches = [
        frame_paths[i : i + args.batch_size]
        for i in range(0, len(frame_paths), args.batch_size)
    ]

    for batch_paths in tqdm(batches, desc="Inspecting batches", unit="batch"):
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        t0 = time.perf_counter()
        batch_scores = inspect_batch(images, model, processor, debug=args.debug)
        batch_elapsed_ms = (time.perf_counter() - t0) * 1000
        total_inference_time += batch_elapsed_ms
        per_frame_ms = batch_elapsed_ms / len(batch_paths)

        for i, (frame_path, (score, reason)) in enumerate(
            zip(batch_paths, batch_scores)
        ):
            global_idx += 1
            label = frame_path.stem
            accepted = score >= args.threshold
            status = "✓ ACCEPT" if accepted else "✗ REJECT"

            tqdm.write(
                f"  Frame {global_idx:2d} | {label:<32s} | "
                f"{status}  score={score:3d}/100  |  {reason}  "
                f"[{per_frame_ms:.0f} ms/frame]"
            )

            results.append(
                {
                    "frame": global_idx,
                    "label": label,
                    "path": str(frame_path),
                    "score": score,
                    "accepted": accepted,
                    "reason": reason,
                    "elapsed_ms": round(per_frame_ms, 1),
                }
            )

    # Select best frame
    accepted_frames = [r for r in results if r["accepted"]]
    rejected_frames = [r for r in results if not r["accepted"]]

    if accepted_frames:
        best = max(accepted_frames, key=lambda r: r["score"])
    elif results:
        best = max(results, key=lambda r: r["score"])
    else:
        best = None

    avg_ms = total_inference_time / len(results) if results else 0

    # Summary
    print("\n" + "=" * 65)
    if best:
        tag = (
            "BEST (above threshold)"
            if best["accepted"]
            else "BEST (all below threshold)"
        )
        print(f"  {tag}")
        print(
            f"  Frame {best['frame']} — {best['label']}  |  score={best['score']}/100"
        )
        print(f"  Reason  : {best['reason']}")
        print(f"  Path    : {best['path']}")
        others = [r for r in accepted_frames if r is not best]
        if others:
            o_str = ", ".join(f"Frame {r['frame']} ({r['score']}/100)" for r in others)
            print(f"  Other accepted: {o_str}")
    else:
        print("  No frames found.")
    print("=" * 65)
    print(
        f"\nAccepted (≥{args.threshold}): {len(accepted_frames)}  |  "
        f"Rejected: {len(rejected_frames)}  |  Total: {len(results)}"
    )
    print(
        f"Timing — total: {total_inference_time / 1000:.1f}s  |  "
        f"avg per frame: {avg_ms:.0f} ms"
    )

    # Save JSON
    output_data = {
        "timestamp": timestamp,
        "frames_dir": str(frames_dir),
        "model_id": MODEL_ID,
        "threshold": args.threshold,
        "quantization": not args.no_quant,
        "total_frames": len(results),
        "accepted_count": len(accepted_frames),
        "rejected_count": len(rejected_frames),
        "best_frame": best,
        "batch_size": args.batch_size,
        "timing": {
            "total_inference_s": round(total_inference_time / 1000, 2),
            "avg_per_frame_ms": round(avg_ms, 1),
        },
        "results": results,
    }
    json_path = results_dir / f"{timestamp}_results.json"
    json_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2))
    print(f"Results JSON saved → {json_path.relative_to(REPO_ROOT)}")

    # Save plots
    save_preview_plot(frame_paths, plots_dir / f"{timestamp}_preview.png")
    save_scores_plot(
        results,
        best,
        args.threshold,
        plots_dir / f"{timestamp}_scores.png",
    )

    # Free VRAM
    import gc

    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nVRAM freed. Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM-based image quality router for the nursing-robot OCR pipeline."
    )
    parser.add_argument(
        "--frames-dir",
        default=str(DEFAULT_FRAMES),
        help=f"Directory containing frame images (default: {DEFAULT_FRAMES})",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=ACCEPT_THRESHOLD,
        help=f"Minimum score to accept a frame (default: {ACCEPT_THRESHOLD})",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Disable 4-bit NF4 quantization (uses full bf16, ~8 GB VRAM)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of frames per inference batch (default: 4)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw model output for each frame",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
