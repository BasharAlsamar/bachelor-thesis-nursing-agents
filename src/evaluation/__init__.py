"""Evaluation module for OCR and LLM methods."""

from .metrics import (
    calculate_all_metrics,
    calculate_cer,
    calculate_wer,
    calculate_ned,
    calculate_vocabulary_overlap_metrics,
    format_metrics_display,
    aggregate_metrics_macro,
    aggregate_metrics_micro,
    compare_aggregation_methods,
)

try:
    from .llm_metrics import (
        evaluate_vlm_transcription,
        evaluate_llm_output,
        calculate_rouge_scores,
        calculate_information_preservation,
        validate_json_structure,
        evaluate_field_extraction,
    )
except ImportError:
    # LLM metrics require additional dependencies
    pass
