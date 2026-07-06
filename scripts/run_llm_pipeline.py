#!/usr/bin/env python3
"""
Standalone script to run LLM processing pipeline.

This script processes OCR outputs with an LLM model independently from notebooks.
Perfect for batch processing multiple OCR methods with different LLM models.

Usage:
    # Basic usage
    python scripts/run_llm_pipeline.py \\
        --ocr-method easyocr \\
        --model qwen-7b
    
    # With custom paths
    python scripts/run_llm_pipeline.py \\
        --input results/metrics/paddleocr/detailed_results.json \\
        --output results/metrics/paddleocr/llm_outputs/qwen_14b/ \\
        --model qwen-14b \\
        --prompt care_plan_generation
    
    # Quick test on 10 samples
    python scripts/run_llm_pipeline.py \\
        --ocr-method easyocr \\
        --model qwen-1.5b \\
        --limit 10
    
    # With quantization for large models
    python scripts/run_llm_pipeline.py \\
        --model qwen-14b \\
        --load-in-8bit
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.llm_processing import LLMPipeline
from src.llm_processing.prompts import list_prompts
from src.llm_processing.models import list_available_presets


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LLM processing pipeline on OCR results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input/Output
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--ocr-method",
        type=str,
        help="OCR method name (e.g., easyocr, paddleocr). Auto-constructs paths.",
    )
    io_group.add_argument(
        "--input",
        type=str,
        help="Input JSON file with OCR results (detailed_results.json). "
        "Overrides --ocr-method if specified.",
    )
    io_group.add_argument(
        "--output",
        type=str,
        help="Output directory for LLM results. Auto-generated if not specified.",
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        default="qwen-1.5b",
        help="Model name or preset (default: qwen-1.5b). "
        "Use --list-models to see available presets.",
    )
    model_group.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to run on (default: auto)",
    )
    model_group.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization (saves memory)",
    )
    model_group.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization (saves more memory)",
    )

    # Prompt configuration
    prompt_group = parser.add_argument_group("Prompt Configuration")
    prompt_group.add_argument(
        "--prompt",
        type=str,
        default="nursing_summary",
        help="Prompt template name (default: nursing_summary). "
        "Use --list-prompts to see available prompts.",
    )

    # Generation parameters
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7, 0.0=deterministic)",
    )
    gen_group.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (default: 0.9)",
    )
    gen_group.add_argument(
        "--no-sample",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--limit", type=int, help="Limit number of samples to process (default: all)"
    )
    proc_group.add_argument(
        "--no-individual",
        action="store_true",
        help="Do not save individual result files (only save combined file)",
    )

    # Utility options
    util_group = parser.add_argument_group("Utility Options")
    util_group.add_argument(
        "--list-models",
        action="store_true",
        help="List available model presets and exit",
    )
    util_group.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt templates and exit",
    )
    util_group.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def list_models_and_exit():
    """List available model presets and exit."""
    print("\nAvailable Model Presets:")
    print("=" * 80)
    for preset, full_name in list_available_presets().items():
        print(f"  {preset:15s} → {full_name}")
    print("\nYou can also use any HuggingFace model name directly.")
    print("=" * 80)
    sys.exit(0)


def list_prompts_and_exit():
    """List available prompts and exit."""
    print("\nAvailable Prompt Templates:")
    print("=" * 80)
    for name, description in list_prompts():
        print(f"\n  {name}")
        print(f"    {description}")
    print("\n" + "=" * 80)
    sys.exit(0)


def construct_paths(args):
    """
    Construct input/output paths based on arguments.

    Returns:
        Tuple of (input_file, output_dir)
    """
    project_root = Path(__file__).parent.parent

    # Input file
    if args.input:
        input_file = Path(args.input)
    elif args.ocr_method:
        input_file = (
            project_root
            / "results"
            / "metrics"
            / args.ocr_method
            / "detailed_results.json"
        )
    else:
        raise ValueError("Must specify either --ocr-method or --input")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    elif args.ocr_method:
        # Extract model shortname for directory
        model_shortname = args.model.split("/")[-1].lower().replace("-instruct", "")
        prompt_shortname = args.prompt.replace("_", "-")
        output_dir = (
            project_root
            / "results"
            / "metrics"
            / args.ocr_method
            / "llm_outputs"
            / f"{model_shortname}_{prompt_shortname}"
        )
    else:
        raise ValueError("Must specify --output when using --input")

    return input_file, output_dir


def main():
    """Main entry point."""
    args = parse_args()

    # Handle list commands
    if args.list_models:
        list_models_and_exit()

    if args.list_prompts:
        list_prompts_and_exit()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Construct paths
        input_file, output_dir = construct_paths(args)

        logger.info("=" * 80)
        logger.info("LLM PROCESSING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Input:  {input_file}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Model:  {args.model}")
        logger.info(f"Prompt: {args.prompt}")
        logger.info("=" * 80)

        # Create pipeline
        pipeline = LLMPipeline(
            model_name=args.model,
            prompt_name=args.prompt,
            device=None if args.device == "auto" else args.device,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.no_sample,
        )

        # Process OCR results
        results = pipeline.process_ocr_results(
            input_file=input_file,
            output_dir=output_dir,
            limit=args.limit,
            save_individual=not args.no_individual,
        )

        logger.info("=" * 80)
        logger.info(f"✓ PROCESSING COMPLETE")
        logger.info(f"  Processed: {len(results)} samples")
        logger.info(f"  Results saved to: {output_dir}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
