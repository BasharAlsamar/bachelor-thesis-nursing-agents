"""
Metrics for OCR evaluation.

This module provides comprehensive metrics for evaluating OCR performance:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Normalized Edit Distance (NED)
- Vocabulary overlap (unique-word precision/recall/F1)
- Sequence-based Accuracy

Implementation uses established open-source packages:
- jiwer: WER/CER computation from ASR/speech community (https://github.com/jitsi/jiwer)
- rapidfuzz: Fast C++ Levenshtein distance (https://github.com/maxbachmann/RapidFuzz)
- difflib: Python standard library sequence matching

Normalization policy (for German nursing notes):
- Lowercase conversion
- Whitespace collapse (multiple spaces → single space)
- Preserves: umlauts (ä, ö, ü), ß, punctuation, numbers
- Tokenization: whitespace splitting
"""

from typing import List, Dict
from difflib import SequenceMatcher
import re

# Open-source packages for metrics
import jiwer
from rapidfuzz.distance import Levenshtein


# Define explicit jiwer transforms for reproducibility
# These transforms match our normalization policy:
# - Convert newlines/tabs to spaces (for OCR text with line breaks)
# - Lowercase conversion
# - Whitespace collapse
# - Preserves German characters (ä, ö, ü, ß), punctuation, numbers
_TRANSFORM_NORMALIZE = jiwer.Compose(
    [
        jiwer.RemoveWhiteSpace(replace_by_space=True),  # Newlines/tabs → space
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ]
)

# For contiguous text (nursing notes as single sentence/paragraph)
_TRANSFORM_TO_WORDS = jiwer.Compose(
    [
        _TRANSFORM_NORMALIZE,
        jiwer.ReduceToListOfListOfWords(),
    ]
)

_TRANSFORM_TO_CHARS = jiwer.Compose(
    [
        _TRANSFORM_NORMALIZE,
        jiwer.ReduceToListOfListOfChars(),
    ]
)


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison (used for NED and vocabulary metrics).

    Uses jiwer transforms internally for consistency: line breaks to spaces,
    lowercase, strip, collapse spaces. German characters (ä, ö, ü, ß),
    punctuation, and numbers are preserved.

    Args:
        text: Input text string

    Returns:
        Normalized text string
    """
    # Use jiwer transform for consistency with WER/CER normalization
    return _TRANSFORM_NORMALIZE(text)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings using rapidfuzz.

    Uses the optimized C implementation from rapidfuzz library for performance.
    This is significantly faster than pure Python implementations (10-100x speedup).

    Args:
        s1: First string
        s2: Second string

    Returns:
        Levenshtein distance (minimum number of single-character edits)
    """
    return Levenshtein.distance(s1, s2)


def calculate_cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Calculate Character Error Rate (CER) using jiwer library.

    CER = (Insertions + Deletions + Substitutions) / Total Characters in Reference

    Uses jiwer's process_characters with explicit transforms for reproducibility.
    Nursing notes are treated as contiguous text (single sentence/paragraph).
    Transform: lowercase + strip + collapse spaces; preserves German characters,
    punctuation, numbers.

    Note: CER can exceed 100% when there are many insertions (hypothesis much
    longer than reference). This is reported without capping to properly reflect
    severe errors.

    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        normalize: If True, apply normalization transform. Default True.

    Returns:
        CER as a percentage (can exceed 100%)
    """
    if normalize:
        # Use jiwer's transform API for explicit, reproducible normalization
        output = jiwer.process_characters(
            reference,
            hypothesis,
            reference_transform=_TRANSFORM_TO_CHARS,
            hypothesis_transform=_TRANSFORM_TO_CHARS,
        )
        return output.cer * 100
    else:
        # No normalization - strict comparison
        if len(reference) == 0:
            return 100.0 if len(hypothesis) > 0 else 0.0
        return jiwer.cer(reference, hypothesis) * 100


def calculate_wer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Calculate Word Error Rate (WER) using jiwer library.

    WER = (Insertions + Deletions + Substitutions) / Total Words in Reference

    Uses jiwer's process_words with explicit transforms for reproducibility.
    Nursing notes are treated as contiguous text (single sentence/paragraph).
    Transform: lowercase + strip + collapse spaces, then tokenize by whitespace.
    Preserves German characters, punctuation, numbers.

    Note: WER can exceed 100% when there are many insertions (hypothesis has
    significantly more words than reference). This is reported without capping
    to properly reflect severe errors.

    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        normalize: If True, apply normalization transform. Default True.

    Returns:
        WER as a percentage (can exceed 100%)
    """
    if normalize:
        # Use jiwer's transform API for explicit, reproducible normalization
        output = jiwer.process_words(
            reference,
            hypothesis,
            reference_transform=_TRANSFORM_TO_WORDS,
            hypothesis_transform=_TRANSFORM_TO_WORDS,
        )
        return output.wer * 100
    else:
        # No normalization - strict comparison
        ref_words = reference.split()
        if len(ref_words) == 0:
            hyp_words = hypothesis.split()
            return 100.0 if len(hyp_words) > 0 else 0.0
        return jiwer.wer(reference, hypothesis) * 100


def calculate_ned(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Calculate Normalized Edit Distance (NED).

    NED = Levenshtein Distance / max(len(reference), len(hypothesis))

    Uses rapidfuzz for fast Levenshtein distance computation. NED is always
    bounded between 0 and 100% by definition (normalizing by the maximum
    length ensures this).

    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        normalize: If True, normalize text (lowercase, collapse spaces) before
                   comparison. Default True for consistency with WER.

    Returns:
        NED as a percentage (0-100)
    """
    ref_text = normalize_text(reference) if normalize else reference
    hyp_text = normalize_text(hypothesis) if normalize else hypothesis

    max_len = max(len(ref_text), len(hyp_text))
    if max_len == 0:
        return 0.0

    # Use rapidfuzz's fast C implementation
    distance = Levenshtein.distance(ref_text, hyp_text)
    ned = (distance / max_len) * 100
    return ned  # NED is naturally bounded [0, 100]


def calculate_vocabulary_overlap_metrics(
    reference: str, hypothesis: str, normalize: bool = True
) -> Dict[str, float]:
    """
    Calculate Precision, Recall, and F1-Score for unique word vocabulary overlap.

    This compares sets of unique words (bag-of-words vocabulary) between reference
    and hypothesis. It does NOT account for word order, duplicate words, or
    segmentation errors. Use this as a supplementary metric to understand vocabulary
    coverage, not as a primary OCR quality metric.

    Note: This is NOT "word detection" in the bounding-box sense used in OCR
    literature. It is simply unique word set overlap.

    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        normalize: If True, normalize text (lowercase, collapse spaces) before
                   comparison. Default True.

    Returns:
        Dictionary with 'vocab_precision', 'vocab_recall', and 'vocab_f1' (all as percentages)
    """
    ref_text = normalize_text(reference) if normalize else reference
    hyp_text = normalize_text(hypothesis) if normalize else hypothesis

    ref_words = set(ref_text.split())
    hyp_words = set(hyp_text.split())

    if len(ref_words) == 0 and len(hyp_words) == 0:
        return {"vocab_precision": 100.0, "vocab_recall": 100.0, "vocab_f1": 100.0}

    if len(hyp_words) == 0:
        return {"vocab_precision": 0.0, "vocab_recall": 0.0, "vocab_f1": 0.0}

    if len(ref_words) == 0:
        return {"vocab_precision": 0.0, "vocab_recall": 0.0, "vocab_f1": 0.0}

    # True Positives: words that appear in both reference and hypothesis
    true_positives = len(ref_words.intersection(hyp_words))

    # False Positives: words in hypothesis but not in reference
    false_positives = len(hyp_words - ref_words)

    # False Negatives: words in reference but not in hypothesis
    false_negatives = len(ref_words - hyp_words)

    # Calculate metrics
    precision = (
        (true_positives / (true_positives + false_positives)) * 100
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        (true_positives / (true_positives + false_negatives)) * 100
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {"vocab_precision": precision, "vocab_recall": recall, "vocab_f1": f1_score}


def calculate_accuracy(
    reference: str, hypothesis: str, normalize: bool = True
) -> float:
    """
    Calculate sequence similarity-based accuracy.

    Uses Python's SequenceMatcher to calculate similarity ratio.

    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        normalize: If True, normalize text (lowercase, collapse spaces) before
                   comparison. Default True for consistency with WER.

    Returns:
        Accuracy as a percentage (0-100)
    """
    ref_text = normalize_text(reference) if normalize else reference
    hyp_text = normalize_text(hypothesis) if normalize else hypothesis

    similarity = SequenceMatcher(None, ref_text, hyp_text).ratio()
    return similarity * 100


def calculate_all_metrics(
    reference: str, hypothesis: str, normalize: bool = True
) -> Dict[str, float]:
    """
    Calculate all OCR/transcription evaluation metrics in one call.

    This function provides standard transcription quality metrics suitable for
    evaluating both OCR engines and VLM transcription outputs.

    Args:
        reference: Ground truth text
        hypothesis: Predicted text (OCR or VLM transcription output)
        normalize: If True (default), normalize text (lowercase, collapse spaces)
                   before computing all metrics. Set to False for "strict" mode
                   that preserves case and whitespace.

    Returns:
        Dictionary containing all metrics:
        - accuracy: Sequence similarity (0-100)
        - cer: Character Error Rate (can exceed 100%)
        - wer: Word Error Rate (can exceed 100%)
        - ned: Normalized Edit Distance (0-100)
        - vocab_precision: Unique word vocabulary precision (0-100)
        - vocab_recall: Unique word vocabulary recall (0-100)
        - vocab_f1: Unique word vocabulary F1-score (0-100)
    """
    vocab_metrics = calculate_vocabulary_overlap_metrics(
        reference, hypothesis, normalize
    )

    return {
        "accuracy": calculate_accuracy(reference, hypothesis, normalize),
        "cer": calculate_cer(reference, hypothesis, normalize),
        "wer": calculate_wer(reference, hypothesis, normalize),
        "ned": calculate_ned(reference, hypothesis, normalize),
        "vocab_precision": vocab_metrics["vocab_precision"],
        "vocab_recall": vocab_metrics["vocab_recall"],
        "vocab_f1": vocab_metrics["vocab_f1"],
    }


def format_metrics_display(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary for pretty console output.

    Args:
        metrics: Dictionary of metric names and values

    Returns:
        Formatted string for display
    """
    lines = ["=" * 60, "METRICS", "=" * 60]

    metric_labels = {
        "accuracy": "✓ Accuracy (Sequence Similarity)",
        "cer": "✗ CER (Character Error Rate)",
        "wer": "✗ WER (Word Error Rate)",
        "ned": "✗ NED (Normalized Edit Distance)",
        "vocab_precision": "✓ Vocabulary Precision",
        "vocab_recall": "✓ Vocabulary Recall",
        "vocab_f1": "✓ Vocabulary F1-Score",
    }

    for key, label in metric_labels.items():
        if key in metrics:
            lines.append(f"{label:40s}: {metrics[key]:6.2f}%")

    lines.append("=" * 60)
    return "\n".join(lines)


# ==============================================================================
# Test Set Aggregation Functions
# ==============================================================================


def aggregate_metrics_macro(results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics using macro-averaging (mean of per-sample rates).

    Macro-averaging computes the metric for each sample independently,
    then takes the arithmetic mean. This treats all samples equally
    regardless of text length.

    Args:
        results: List of metric dictionaries from calculate_all_metrics()

    Returns:
        Dictionary with macro-averaged metrics

    Example:
        >>> results = [
        ...     calculate_all_metrics(ref1, hyp1),
        ...     calculate_all_metrics(ref2, hyp2),
        ... ]
        >>> macro_avg = aggregate_metrics_macro(results)
    """
    if not results:
        return {}

    # Get all metric keys from first result
    metric_keys = results[0].keys()

    aggregated = {}
    for key in metric_keys:
        # Average the metric values across all samples
        values = [r[key] for r in results if key in r]
        aggregated[key] = sum(values) / len(values) if values else 0.0

    return aggregated


def aggregate_metrics_micro(
    references: List[str], hypotheses: List[str], normalize: bool = True
) -> Dict[str, float]:
    """
    Aggregate metrics using micro-averaging (pooled edit distance).

    Micro-averaging pools all references and hypotheses together, computing
    edit distances and reference lengths across the entire test set. This
    weights samples by text length and is often preferred for CER/WER
    because longer texts contribute proportionally more to the final score.

    For CER/WER/NED, this computes:
        - Total edit distance across all samples
        - Total reference length across all samples
        - Metric = (total distance / total length) * 100

    WER micro-aggregation uses jiwer's process_words per sample, summing
    substitutions, insertions, deletions across all samples, then dividing by
    total reference word count. This ensures consistency with jiwer's algorithm.

    For accuracy and vocabulary metrics, this falls back to macro-averaging
    since they don't have a natural pooled definition.

    Args:
        references: List of ground truth texts
        hypotheses: List of predicted texts (must match length of references)
        normalize: If True, apply normalization transform (lowercase, collapse
                   spaces) consistently using jiwer transforms

    Returns:
        Dictionary with micro-averaged metrics

    Example:
        >>> refs = ["text one", "text two"]
        >>> hyps = ["text on", "text too"]
        >>> micro_avg = aggregate_metrics_micro(refs, hyps)
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    if not references:
        return {}

    # Micro-average for CER: sum of all edit distances / sum of all ref lengths
    total_cer_distance = 0
    total_cer_length = 0

    # Micro-average for WER: sum of all word edit distances / sum of all ref word counts
    total_wer_distance = 0
    total_wer_length = 0

    # Micro-average for NED: sum of all edit distances / sum of all max lengths
    total_ned_distance = 0
    total_ned_length = 0

    # For accuracy and vocab metrics, compute per-sample then average (no pooled version)
    accuracy_values = []
    vocab_precision_values = []
    vocab_recall_values = []
    vocab_f1_values = []

    for ref, hyp in zip(references, hypotheses):
        # Use jiwer transforms for consistent normalization across all metrics
        ref_text = _TRANSFORM_NORMALIZE(ref) if normalize else ref
        hyp_text = _TRANSFORM_NORMALIZE(hyp) if normalize else hyp

        # CER components
        if len(ref_text) > 0:
            cer_dist = Levenshtein.distance(ref_text, hyp_text)
            total_cer_distance += cer_dist
            total_cer_length += len(ref_text)

        # WER components - use jiwer for consistency
        if normalize:
            wer_output = jiwer.process_words(
                ref,
                hyp,
                reference_transform=_TRANSFORM_TO_WORDS,
                hypothesis_transform=_TRANSFORM_TO_WORDS,
            )
        else:
            # No normalization
            ref_words = ref.split()
            if len(ref_words) > 0:
                wer_output = jiwer.process_words(ref, hyp)
            else:
                wer_output = None

        if wer_output and wer_output.references:
            # Calculate total reference word count (robust to empty segments)
            ref_word_count = sum(len(seg) for seg in wer_output.references)
            if ref_word_count > 0:  # Only count if reference has words
                # Sum error counts (substitutions + insertions + deletions)
                total_wer_distance += (
                    wer_output.substitutions
                    + wer_output.insertions
                    + wer_output.deletions
                )
                total_wer_length += ref_word_count

        # NED components
        max_len = max(len(ref_text), len(hyp_text))
        if max_len > 0:
            ned_dist = Levenshtein.distance(ref_text, hyp_text)
            total_ned_distance += ned_dist
            total_ned_length += max_len

        # Per-sample metrics (no pooled version)
        accuracy_values.append(calculate_accuracy(ref, hyp, normalize))
        vocab_metrics = calculate_vocabulary_overlap_metrics(ref, hyp, normalize)
        vocab_precision_values.append(vocab_metrics["vocab_precision"])
        vocab_recall_values.append(vocab_metrics["vocab_recall"])
        vocab_f1_values.append(vocab_metrics["vocab_f1"])

    # Compute micro-averaged rates
    cer = (total_cer_distance / total_cer_length * 100) if total_cer_length > 0 else 0.0
    wer = (total_wer_distance / total_wer_length * 100) if total_wer_length > 0 else 0.0
    ned = (total_ned_distance / total_ned_length * 100) if total_ned_length > 0 else 0.0

    # Macro-average for metrics without pooled version
    accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0.0
    vocab_precision = (
        sum(vocab_precision_values) / len(vocab_precision_values)
        if vocab_precision_values
        else 0.0
    )
    vocab_recall = (
        sum(vocab_recall_values) / len(vocab_recall_values)
        if vocab_recall_values
        else 0.0
    )
    vocab_f1 = sum(vocab_f1_values) / len(vocab_f1_values) if vocab_f1_values else 0.0

    return {
        "cer": cer,
        "wer": wer,
        "ned": ned,
        "accuracy": accuracy,
        "vocab_precision": vocab_precision,
        "vocab_recall": vocab_recall,
        "vocab_f1": vocab_f1,
    }


def compare_aggregation_methods(
    references: List[str], hypotheses: List[str], normalize: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compare macro and micro aggregation methods side-by-side.

    This is useful for understanding how your choice of aggregation
    affects the reported metrics, especially when samples have varying
    lengths.

    Args:
        references: List of ground truth texts
        hypotheses: List of predicted texts
        normalize: If True, normalize text before computing metrics

    Returns:
        Dictionary with 'macro' and 'micro' keys, each containing aggregated metrics

    Example:
        >>> refs = ["short", "this is a much longer text example"]
        >>> hyps = ["shrt", "this is a much longer text exmple"]
        >>> comparison = compare_aggregation_methods(refs, hyps)
        >>> print(f"Macro CER: {comparison['macro']['cer']:.2f}%")
        >>> print(f"Micro CER: {comparison['micro']['cer']:.2f}%")
    """
    # Compute per-sample metrics for macro
    per_sample_results = [
        calculate_all_metrics(ref, hyp, normalize)
        for ref, hyp in zip(references, hypotheses)
    ]

    macro = aggregate_metrics_macro(per_sample_results)
    micro = aggregate_metrics_micro(references, hypotheses, normalize)

    return {
        "macro": macro,
        "micro": micro,
    }
