#!/usr/bin/env python3
"""
Standalone script to evaluate LLM outputs.

This script evaluates LLM-generated outputs against ground truth using multiple metrics.

Usage:
    # Evaluate LLM outputs
    python scripts/evaluate_llm.py \\
        --input results/metrics/easyocr/llm_outputs/qwen_7b/all_llm_results.json \\
        --output results/metrics/easyocr/llm_outputs/qwen_7b/evaluation.json
    
    # Quick preview
    python scripts/evaluate_llm.py \\
        --input results/metrics/easyocr/llm_outputs/qwen_7b/all_llm_results.json \\
        --preview
"""

import sys
import argparse
import json
import logging
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.llm_metrics import (
    evaluate_llm_output,
    format_llm_metrics_display,
    interpret_llm_metrics,
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM outputs against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input JSON file with LLM results (all_llm_results.json)",
    )
    parser.add_argument(
        "--output", type=str, help="Output JSON file for evaluation results (optional)"
    )
    parser.add_argument(
        "--preview", action="store_true", help="Show preview of first sample evaluation"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load LLM results
        input_file = Path(args.input)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        logger.info(f"Loading LLM results from: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            llm_results = json.load(f)

        logger.info(f"Loaded {len(llm_results)} LLM results")

        # Evaluate each result
        evaluated_results = []

        for i, result in enumerate(llm_results):
            sample_id = result.get("sample_id", f"sample_{i}")

            # Skip if no ground truth
            if "ground_truth" not in result:
                logger.warning(f"No ground truth for {sample_id}, skipping")
                continue

            ground_truth = result["ground_truth"]
            llm_output = result["llm_output"]
            ocr_text = result.get("ocr_text")

            # Evaluate
            metrics = evaluate_llm_output(ground_truth, llm_output, ocr_text)

            # Add to result
            eval_result = {
                "sample_id": sample_id,
                "metrics": metrics,
                **result,  # Include original data
            }

            evaluated_results.append(eval_result)

            # Show preview if requested
            if args.preview and i == 0:
                print("\n" + "=" * 80)
                print(f"PREVIEW: {sample_id}")
                print("=" * 80)
                print(f"\nGround Truth:\n{ground_truth[:200]}...")
                print(f"\nLLM Output:\n{llm_output[:200]}...")
                print("\n" + format_llm_metrics_display(metrics))
                interpretations = interpret_llm_metrics(metrics)
                print("\n💡 Interpretation:")
                for aspect, interpretation in interpretations.items():
                    print(f"  {interpretation}")
                print("=" * 80 + "\n")

        # Calculate aggregate statistics
        df_metrics = pd.DataFrame([r["metrics"] for r in evaluated_results])

        aggregate_stats = {"num_samples": len(evaluated_results), "metrics": {}}

        for col in df_metrics.columns:
            if df_metrics[col].dtype in ["int64", "float64"]:
                aggregate_stats["metrics"][col] = {
                    "mean": float(df_metrics[col].mean()),
                    "std": float(df_metrics[col].std()),
                    "min": float(df_metrics[col].min()),
                    "max": float(df_metrics[col].max()),
                    "median": float(df_metrics[col].median()),
                }

        # Display summary
        logger.info("\n" + "=" * 80)
        logger.info("AGGREGATE STATISTICS")
        logger.info("=" * 80)
        logger.info(f"\nEvaluated {len(evaluated_results)} samples\n")

        for metric_name, stats in aggregate_stats["metrics"].items():
            if (
                "rouge" in metric_name.lower()
                or "bleu" in metric_name.lower()
                or "medical" in metric_name.lower()
            ):
                logger.info(f"{metric_name}:")
                logger.info(f"  Mean: {stats['mean']:.2f}%")
                logger.info(f"  Median: {stats['median']:.2f}%")
                logger.info(f"  Std: {stats['std']:.2f}%")
                logger.info(f"  Range: [{stats['min']:.2f}%, {stats['max']:.2f}%]\n")

        # Save results if output specified
        if args.output or not args.preview:
            if args.output:
                output_file = Path(args.output)
            else:
                output_file = input_file.parent / "evaluation_results.json"

            output_data = {
                "evaluated_samples": evaluated_results,
                "aggregate_statistics": aggregate_stats,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"✓ Saved evaluation results to: {output_file}")

            # Also save CSV summary
            csv_file = output_file.with_suffix(".csv")
            df_summary = pd.DataFrame(
                [
                    {"sample_id": r["sample_id"], **r["metrics"]}
                    for r in evaluated_results
                ]
            )
            df_summary.to_csv(csv_file, index=False)
            logger.info(f"✓ Saved CSV summary to: {csv_file}")

        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
