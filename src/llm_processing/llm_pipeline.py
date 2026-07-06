"""
LLM Pipeline for processing OCR outputs.

This module provides a standalone pipeline that can process outputs from any OCR method
(EasyOCR, PaddleOCR, Tesseract, etc.) with any LLM model.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from tqdm.auto import tqdm

from .models import (
    generate_text,
    generate_vision_text,
    get_model_name,
    load_model,
    load_vlm_model,
)
from .prompts import get_prompt, format_prompt

logger = logging.getLogger(__name__)


class LLMPipeline:
    """
    Standalone LLM processing pipeline.

    Features:
    - Works with any OCR method's output (JSON format)
    - Supports multiple LLM models
    - Configurable prompts
    - Batch processing with progress tracking
    - Automatic result saving

    Usage:
        # Create pipeline
        pipeline = LLMPipeline(
            model_name='qwen-7b',
            prompt_name='nursing_summary',
            device='cuda'
        )

        # Process single text
        result = pipeline.process_text("Patient shows good progress...")

        # Process OCR results file
        results = pipeline.process_ocr_results(
            input_file='results/metrics/easyocr/detailed_results.json',
            output_dir='results/metrics/easyocr/llm_outputs/qwen_7b/'
        )
    """

    def __init__(
        self,
        model_name: str = "qwen-1.5b",
        prompt_name: str = "nursing_summary",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.9,
        do_sample: bool = False,
        enable_thinking: bool = False,
        use_vlm: Optional[bool] = None,
        vlm_image_prompt: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ):
        """
        Initialize the LLM pipeline.

        Args:
            model_name: Model name or preset (e.g., 'qwen-7b')
            prompt_name: Name of prompt template to use
            device: Device to run on ('cuda', 'cpu', or None for auto)
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (False = greedy)
            enable_thinking: Enable Qwen3 thinking chain (default False).
                             Keep False for text structuring/summarisation tasks.
            use_vlm: Force vision-language mode. If None, inferred from model name.
            vlm_image_prompt: Optional custom user prompt for image-based inference.
        """
        self.model_name = get_model_name(model_name)
        self.prompt_template = get_prompt(prompt_name)
        self.prompt_name = prompt_name
        self.use_vlm = (
            use_vlm
            if use_vlm is not None
            else any(k in self.model_name.lower() for k in ["-vl", "vision"])
        )
        self.vlm_image_prompt = vlm_image_prompt or (
            "Analysiere die Pflegenotiz direkt aus dem Bild. "
            "Extrahiere den Inhalt und antworte exakt in diesem Format:\n"
            "1. **Zusammenfassung**: Kurze, sachliche Zusammenfassung der Notiz.\n"
            "2. **Wichtige Merkmale**: Stichpunkte mit medizinischen Parametern, "
            "Symptomen und Maßnahmen.\n"
            "Erfinde keine Informationen und korrigiere OCR-Fehler implizit."
        )

        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.enable_thinking = enable_thinking
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1280 * 28 * 28
        # Load model
        logger.info(f"Initializing LLM Pipeline")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Prompt: {prompt_name}")

        self.processor = None
        if self.use_vlm:
            self.model, self.processor = load_vlm_model(
                model_name,
                device=device,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            self.tokenizer = getattr(self.processor, "tokenizer", None)
        else:
            self.model, self.tokenizer = load_model(
                model_name,
                device=device,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )

        logger.info("✓ Pipeline ready")

    def process_text(self, text: str) -> str:
        """
        Process a single text through the LLM.

        Args:
            text: Input text (e.g., OCR output)

        Returns:
            LLM-generated response
        """
        if self.use_vlm and self.tokenizer is None:
            raise ValueError(
                "This pipeline is configured for VLM mode. "
                "Use process_image(image_path) instead of process_text(text)."
            )

        # Format prompt with text
        formatted_task = format_prompt(self.prompt_template, text)

        # Create messages
        messages = [
            {"role": "system", "content": self.prompt_template["system"]},
            {"role": "user", "content": formatted_task},
        ]

        # Generate response
        response = generate_text(
            self.model,
            self.tokenizer,
            messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            enable_thinking=self.enable_thinking,
        )

        return response

    def process_image(
        self,
        image: Union[str, Path],
        user_prompt: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> str:
        """
        Process a single image through a vision-language model.

        Args:
            image: Input image path
            user_prompt: Optional override for the VLM user instruction.
                Can be one of:
                - str: text instruction that will be combined with the image block
                - dict: one multimodal content block
                - list[dict]: full multimodal content blocks for the user role

        Returns:
            VLM-generated response
        """
        if not self.use_vlm or self.processor is None:
            raise ValueError(
                "This pipeline is not configured for VLM mode. "
                "Set use_vlm=True or choose a *-VL model preset."
            )

        if user_prompt is None:
            user_content: List[Dict[str, Any]] = [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": self.vlm_image_prompt},
            ]
        elif isinstance(user_prompt, str):
            user_content = [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": user_prompt},
            ]
        elif isinstance(user_prompt, dict):
            user_content = [user_prompt]
        elif isinstance(user_prompt, list):
            user_content = user_prompt
        else:
            raise TypeError(
                "user_prompt must be str, dict, list[dict], or None for VLM mode"
            )

        has_image_block = any(
            isinstance(block, dict) and block.get("type") == "image"
            for block in user_content
        )
        if not has_image_block:
            user_content = [{"type": "image", "image": str(image)}, *user_content]

        messages = [
            {"role": "system", "content": self.prompt_template["system"]},
            {"role": "user", "content": user_content},
        ]

        return generate_vision_text(
            self.model,
            self.processor,
            messages,
            image=image,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            enable_thinking=self.enable_thinking,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

    def process_ocr_result(self, ocr_result: Dict) -> Dict:
        """
        Process a single OCR result dictionary.

        Args:
            ocr_result: Dictionary with OCR output (requires 'sample_id' and 'predicted_text')

        Returns:
            Dictionary with OCR data + LLM output
        """
        sample_id = ocr_result.get("sample_id", "unknown")
        ocr_text = ocr_result.get("predicted_text", "")

        if not ocr_text:
            logger.warning(f"Empty OCR text for sample {sample_id}")
            llm_output = ""
        else:
            try:
                llm_output = self.process_text(ocr_text)
            except Exception as e:
                logger.error(f"Error processing {sample_id}: {str(e)}")
                llm_output = f"ERROR: {str(e)}"

        # Create result dictionary
        result = {
            "sample_id": sample_id,
            "ocr_text": ocr_text,
            "llm_output": llm_output,
            "model": self.model_name,
            "prompt": self.prompt_name,
            "timestamp": datetime.now().isoformat(),
            "generation_params": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": self.do_sample,
            },
        }

        # Include ground truth and OCR metrics if available
        if "expected_text" in ocr_result:
            result["ground_truth"] = ocr_result["expected_text"]

        if "accuracy" in ocr_result:
            result["ocr_metrics"] = {
                "accuracy": ocr_result.get("accuracy"),
                "cer": ocr_result.get("cer"),
                "wer": ocr_result.get("wer"),
                "f1_score": ocr_result.get("f1_score"),
            }

        return result

    def process_ocr_results(
        self,
        input_file: Union[str, Path],
        output_dir: Union[str, Path],
        limit: Optional[int] = None,
        save_individual: bool = True,
    ) -> List[Dict]:
        """
        Process a file containing OCR results.

        Args:
            input_file: Path to JSON file with OCR results
            output_dir: Directory to save LLM outputs
            limit: Limit number of samples to process (None = all)
            save_individual: Save individual result files

        Returns:
            List of processed results
        """
        input_file = Path(input_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load OCR results
        logger.info(f"Loading OCR results from: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            ocr_results = json.load(f)

        # Handle different input formats
        if isinstance(ocr_results, dict):
            # Might be a single result or dict with results key
            if "results" in ocr_results:
                ocr_results = ocr_results["results"]
            else:
                ocr_results = [ocr_results]

        # Limit samples if specified
        if limit:
            ocr_results = ocr_results[:limit]

        logger.info(f"Processing {len(ocr_results)} samples")
        logger.info(f"Output directory: {output_dir}")

        # Process each result
        processed_results = []
        failed_samples = []

        for ocr_result in tqdm(ocr_results, desc="LLM Processing"):
            try:
                result = self.process_ocr_result(ocr_result)
                processed_results.append(result)

                # Save individual result
                if save_individual:
                    sample_id = result["sample_id"]
                    individual_file = output_dir / f"{sample_id}_llm.json"
                    with open(individual_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

            except Exception as e:
                sample_id = ocr_result.get("sample_id", "unknown")
                logger.error(f"Failed to process {sample_id}: {str(e)}")
                failed_samples.append({"sample_id": sample_id, "error": str(e)})

        # Save results
        logger.info(f"Successfully processed: {len(processed_results)} samples")
        if failed_samples:
            logger.warning(f"Failed: {len(failed_samples)} samples")

        # Save all results
        all_results_file = output_dir / "all_llm_results.json"
        with open(all_results_file, "w", encoding="utf-8") as f:
            json.dump(processed_results, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved all results to: {all_results_file}")

        # Save summary
        summary = {
            "model": self.model_name,
            "prompt": self.prompt_name,
            "num_samples": len(processed_results),
            "num_failed": len(failed_samples),
            "input_file": str(input_file),
            "output_dir": str(output_dir),
            "timestamp": datetime.now().isoformat(),
            "generation_params": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": self.do_sample,
            },
        }

        summary_file = output_dir / "processing_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved summary to: {summary_file}")

        # Save failed samples if any
        if failed_samples:
            failed_file = output_dir / "failed_samples.json"
            with open(failed_file, "w", encoding="utf-8") as f:
                json.dump(failed_samples, f, ensure_ascii=False, indent=2)
            logger.warning(f"⚠ Saved failed samples to: {failed_file}")

        return processed_results

    def update_generation_params(
        self,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ):
        """
        Update generation parameters.

        Args:
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample
        """
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if do_sample is not None:
            self.do_sample = do_sample

        logger.info("Generation parameters updated")
