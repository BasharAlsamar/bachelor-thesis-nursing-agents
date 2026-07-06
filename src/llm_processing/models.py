"""
Model loading utilities for different LLM architectures.

Supports:
- Qwen/Qwen2.5 series
- Qwen/Qwen3 series (with built-in thinking mode)
- Any HuggingFace causal LM
- Extensible for other model types
"""

import re
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import transformers
import logging

logger = logging.getLogger(__name__)


# Model presets for easy configuration
MODEL_PRESETS = {
    # Qwen2.5 series
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
    # Qwen3 series
    "qwen-3b": "Qwen/Qwen3-4B-Instruct-2507",  # replaces Qwen2.5-3B
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",  # July-2025 instruct fine-tune, BF16
    "qwen3-4b-base": "Qwen/Qwen3-4B",  # base Qwen3-4B (no instruct tuning)
    "qwen3-4b-fp8": "Qwen/Qwen3-4B-FP8",  # FP8 variant
    # Qwen3-8B series
    "qwen3-8b": "Qwen/Qwen3-8B",  # BF16 (~16 GB) — use load_in_4bit=True for RTX 3060 12 GB
    "qwen3-8b-fp8": "Qwen/Qwen3-8B-FP8",  # FP8 — requires Ada/Hopper GPU (RTX 4000+/H100)
    # Qwen3-VL series
    "qwen3vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3-vl-8b-latest": "Qwen/Qwen3-VL-8B-Instruct",
    # Mistral series
    "ministral-8b": "mistralai/Ministral-8B-Instruct-2410",  # BF16 (~16 GB) — use load_in_4bit=True for RTX 3060 12 GB
}


def get_model_name(model_identifier: str) -> str:
    """
    Resolve model identifier to full model name.

    Args:
        model_identifier: Either a preset name or full HF model name

    Returns:
        Full HuggingFace model name

    Examples:
        >>> get_model_name('qwen-7b')
        'Qwen/Qwen2.5-7B-Instruct'
        >>> get_model_name('Qwen/Qwen2.5-7B-Instruct')
        'Qwen/Qwen2.5-7B-Instruct'
    """
    # Check if it's a preset
    if model_identifier in MODEL_PRESETS:
        return MODEL_PRESETS[model_identifier]

    # Otherwise assume it's a full model name
    return model_identifier


def load_model(
    model_name: str,
    device: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    torch_dtype: Union[torch.dtype, str, None] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a language model and tokenizer from HuggingFace.

    Args:
        model_name: Model name or preset (e.g., 'qwen3-4b' or 'Qwen/Qwen3-4B-FP8')
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        load_in_8bit: Load model in 8-bit quantization (saves memory)
        load_in_4bit: Load model in 4-bit quantization (saves more memory)
        torch_dtype: PyTorch dtype, or 'auto' to let transformers decide
                     (default: 'auto' for FP8 models, float16 for CUDA, float32 for CPU)

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If invalid configuration
        RuntimeError: If model loading fails
    """
    # Resolve model name
    full_model_name = get_model_name(model_name)

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validate quantization options
    if load_in_8bit and load_in_4bit:
        raise ValueError("Cannot use both 8-bit and 4-bit quantization")

    if (load_in_8bit or load_in_4bit) and device == "cpu":
        logger.warning("Quantization not supported on CPU, disabling quantization")
        load_in_8bit = False
        load_in_4bit = False

    # Determine dtype
    # FP8 pre-quantized models require torch_dtype="auto"; full models use float16/float32
    if torch_dtype is None:
        if "FP8" in full_model_name or "fp8" in full_model_name:
            torch_dtype = "auto"
        elif device == "cuda":
            torch_dtype = torch.bfloat16  # bfloat16 preferred for Qwen3 on modern GPUs
        else:
            torch_dtype = torch.float32

    logger.info(f"Loading model: {full_model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Dtype: {torch_dtype}")
    if load_in_8bit:
        logger.info("Using 8-bit quantization")
    if load_in_4bit:
        logger.info("Using 4-bit quantization")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(full_model_name)
        logger.info("✓ Tokenizer loaded")

        # Prepare model loading kwargs
        model_kwargs = {
            "torch_dtype": torch.float16,
            "attn_implementation": "sdpa",
        }

        if load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = "auto"
        elif load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,  # nested quantisation — saves ~0.4 GB extra
                bnb_4bit_quant_type="nf4",  # NF4 optimal for normally-distributed weights
            )
            model_kwargs["device_map"] = "auto"
        elif device == "cuda":
            model_kwargs["device_map"] = "auto"

        # Load model
        model = AutoModelForCausalLM.from_pretrained(full_model_name, **model_kwargs)

        # Move to device if not using device_map
        if "device_map" not in model_kwargs:
            model = model.to(device)

        logger.info(f"✓ Model loaded successfully")

        # Log model size
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params / 1e9:.2f}B")

        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}") from e


def load_vlm_model(
    model_name: str,
    device: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    torch_dtype: Union[torch.dtype, str, None] = None,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
):
    """
    Load a vision-language model and processor from HuggingFace.

    Returns:
        Tuple of (model, processor)
    """
    full_model_name = get_model_name(model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if load_in_8bit and load_in_4bit:
        raise ValueError("Cannot use both 8-bit and 4-bit quantization")

    if (load_in_8bit or load_in_4bit) and device == "cpu":
        logger.warning("Quantization not supported on CPU, disabling it")
        load_in_8bit = False
        load_in_4bit = False

    if torch_dtype is None:
        if "FP8" in full_model_name or "fp8" in full_model_name:
            torch_dtype = "auto"
        elif device == "cuda":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

    logger.info(f"Loading VLM model: {full_model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Dtype: {torch_dtype}")

    model_kwargs = {
        "torch_dtype": torch.float16,
        "attn_implementation": "sdpa",  # ← أضفه هنا
    }

    if load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["device_map"] = "auto"
    elif load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = "auto"
    elif device == "cuda":
        model_kwargs["device_map"] = "auto"

    try:
        # ✅ بعد
        processor = AutoProcessor.from_pretrained(
            full_model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        logger.info("✓ Processor loaded")

        vlm_model_cls = (
            getattr(transformers, "AutoModelForImageTextToText", None)
            or getattr(transformers, "AutoModelForVision2Seq", None)
            or AutoModelForCausalLM
        )

        model = vlm_model_cls.from_pretrained(full_model_name, **model_kwargs)

        if "device_map" not in model_kwargs:
            model = model.to(device)

        logger.info("✓ VLM loaded successfully")
        return model, processor

    except Exception as e:
        logger.error(f"Failed to load VLM model: {str(e)}")
        raise RuntimeError(f"VLM loading failed: {str(e)}") from e


def _strip_thinking_tokens(text: str) -> str:
    """
    Remove Qwen3 thinking blocks (<think>...</think>) from generated text.
    Returns only the final answer after the closing </think> tag.
    """
    # Remove the full <think>...</think> block (non-greedy, dotall)
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return stripped.strip()


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    enable_thinking: bool = False,
) -> str:

    # Handle pad_token_id — Qwen tokenizers often lack one
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Apply chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Tokenize with attention_mask
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,  # ← إضافة مهمة
        truncation=True,  # ← حماية من النصوص الطويلة جداً
        max_length=4096,  # ← حد أقصى مناسب لنموذج 8B مع ذاكرة 12GB
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    if enable_thinking:
        generated_text = _strip_thinking_tokens(generated_text)

    return generated_text


def generate_vision_text(
    model,
    processor,
    messages: list,
    image: Optional[Union[str, Path, Image.Image]] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    enable_thinking: bool = False,
    min_pixels: int = 256 * 28 * 28,
    max_pixels=1280 * 28 * 28,
) -> str:
    """Generate text from multimodal chat messages using a chat VLM.

    Uses qwen_vl_utils.process_vision_info for fast and correct image preprocessing
    (Dynamic Resolution / Native Patching). Falls back to PIL loading if unavailable.

    Args:
        model:            Loaded VLM model.
        processor:        Loaded AutoProcessor for the VLM.
        messages:         Chat messages list with multimodal content blocks.
        image:            Fallback image if no image block exists in messages.
        max_new_tokens:   Maximum number of tokens to generate.
        temperature:      Sampling temperature (ignored when do_sample=False).
        top_p:            Nucleus sampling parameter.
        do_sample:        If False, uses greedy decoding (best for OCR tasks).
        enable_thinking:  Enable Qwen3 thinking chain (keep False for OCR/summarization).
        min_pixels:       Minimum image resolution for Qwen dynamic patching.
        max_pixels:       Maximum image resolution — lower = faster, higher = more detail.
    """

    # ── 0. Try to import qwen_vl_utils ────────────────────────────────────────
    try:
        from qwen_vl_utils import process_vision_info

        _use_qwen_utils = True
    except ImportError:
        _use_qwen_utils = False
        logger.warning(
            "qwen_vl_utils not found — falling back to PIL image loading (slower). "
            "Install with: pip install qwen-vl-utils"
        )

    # ── 1. Ensure at least one image block exists in messages ─────────────────
    def _has_image_block(msgs: list) -> bool:
        for msg in msgs:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    return True
        return False

    if not _has_image_block(messages) and image is not None:
        # Inject fallback image into the user turn
        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                msg["content"] = [
                    {"type": "image", "image": str(image)},
                    *msg["content"],
                ]
                break
        else:
            # No user turn with list content found — append a new one
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image", "image": str(image)}],
                }
            )

    if not _has_image_block(messages):
        raise ValueError(
            "No image block found in messages and no fallback image provided. "
            "Pass image= or include {'type': 'image', 'image': path} in messages."
        )

    # ── 2. Apply chat template ───────────────────────────────────
    try:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Older processors / non-Qwen3 models don't support enable_thinking
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ── 3. Prepare inputs ─────────────────────────────────
    if _use_qwen_utils:
        # Fast path: Qwen native dynamic resolution patching
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            padding=True,
            return_tensors="pt",
        )
    else:
        # Slow fallback: manual PIL loading
        def _to_pil(img_ref):
            if isinstance(img_ref, Image.Image):
                return img_ref.convert("RGB")
            if isinstance(img_ref, (str, Path)):
                return Image.open(img_ref).convert("RGB")
            raise TypeError(
                f"Unsupported image reference type: {type(img_ref).__name__}"
            )

        pil_images = []
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    if block.get("image") is not None:
                        pil_images.append(_to_pil(block["image"]))

        if not pil_images:
            raise ValueError("No images could be loaded from messages.")

        inputs = processor(
            text=[prompt],
            images=pil_images,
            padding=True,
            return_tensors="pt",
        )

    # Move all tensors to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # ── 4. Generate ──────────────────────────────────────────
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
            eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
        )

    # ── 5. Decode only newly generated tokens (skip prompt tokens)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    # ── 6. Strip thinking blocks if enabled
    if enable_thinking:
        generated_text = _strip_thinking_tokens(generated_text)

    return generated_text.strip()


def list_available_presets() -> dict:
    """
    List all available model presets.

    Returns:
        Dictionary of preset names and their full model names
    """
    return MODEL_PRESETS.copy()
