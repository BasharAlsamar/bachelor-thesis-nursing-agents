"""
LLM evaluation metrics for nursing note analysis.

Provides metrics for evaluating:
1. VLM transcription (image → text): Uses same OCR metrics (CER/WER/NED)
2. LLM-generated summaries, extractions: ROUGE, BLEU, medical term preservation
3. Structured extraction: JSON validity, field-level accuracy

For VLM transcription evaluation, use the transcription metrics from metrics.py:
    - calculate_all_metrics(reference, vlm_transcription)
    provides CER, WER, NED, accuracy, and vocabulary overlap.

For LLM summaries/analysis, use ROUGE/BLEU and medical term preservation.
For structured extraction (JSON), use field-level accuracy and JSON validity checks.
"""

from typing import Dict, Set, Any, Optional, List, Tuple
from rouge_score import rouge_scorer
import json
import re
import numpy as np

# Import OCR metrics for VLM transcription evaluation
from .metrics import calculate_all_metrics

# Semantic similarity (lazy import to avoid startup delay)
_SENTENCE_TRANSFORMER_MODEL = None


def _get_sentence_transformer():
    """Lazy load sentence transformer model."""
    global _SENTENCE_TRANSFORMER_MODEL
    if _SENTENCE_TRANSFORMER_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer

            # Use multilingual model for German support
            _SENTENCE_TRANSFORMER_MODEL = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for semantic similarity metrics. "
                "Install with: pip install sentence-transformers"
            )
    return _SENTENCE_TRANSFORMER_MODEL


def calculate_rouge_scores(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Calculate ROUGE scores for summarization evaluation.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures overlap
    between generated and reference text.

    Args:
        reference: Ground truth or reference text
        hypothesis: Generated text (LLM output)

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores (precision, recall, F1)
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(reference, hypothesis)

    return {
        "rouge1_precision": scores["rouge1"].precision * 100,
        "rouge1_recall": scores["rouge1"].recall * 100,
        "rouge1_f1": scores["rouge1"].fmeasure * 100,
        "rouge2_precision": scores["rouge2"].precision * 100,
        "rouge2_recall": scores["rouge2"].recall * 100,
        "rouge2_f1": scores["rouge2"].fmeasure * 100,
        "rougeL_precision": scores["rougeL"].precision * 100,
        "rougeL_recall": scores["rougeL"].recall * 100,
        "rougeL_f1": scores["rougeL"].fmeasure * 100,
    }


def calculate_bertscore(
    reference: str,
    hypothesis: str,
    model_type: str = "bert-base-multilingual-cased",
    lang: str = "de",
) -> Dict[str, float]:
    """
    Calculate BERTScore for semantic similarity evaluation.

    BERTScore uses contextual BERT embeddings to compare hypothesis and reference
    at the token level.  Unlike BLEU, it captures paraphrasing and synonyms —
    making it much more suitable for German nursing-note summarization.

    Uses ``bert-base-multilingual-cased`` by default, which has strong German
    support and is already cached if a HuggingFace model has been downloaded.

    Args:
        reference:   Gold-standard reference text (e.g., reference_summary)
        hypothesis:  LLM-generated text to evaluate
        model_type:  HuggingFace model identifier (default: bert-base-multilingual-cased)
        lang:        Language code used by bert-score rescaling (default: "de")

    Returns:
        Dictionary with keys ``precision``, ``recall``, ``f1`` (all 0–100 %).
    """
    try:
        import torch
        from bert_score import score as bert_score_fn

        # Pandas>=3 can return read-only NumPy arrays for BERTScore baselines.
        # Copy read-only arrays before torch conversion to avoid undefined behavior.
        _orig_from_numpy = torch.from_numpy

        def _safe_from_numpy(arr):
            if isinstance(arr, np.ndarray) and not arr.flags.writeable:
                arr = np.array(arr, copy=True)
            return _orig_from_numpy(arr)

        torch.from_numpy = _safe_from_numpy
        try:
            P, R, F1 = bert_score_fn(
                [hypothesis],
                [reference],
                model_type=model_type,
                lang=lang,
                rescale_with_baseline=True,
                verbose=False,
            )
        finally:
            torch.from_numpy = _orig_from_numpy
        return {
            "bertscore_precision": float(P[0]) * 100,
            "bertscore_recall": float(R[0]) * 100,
            "bertscore_f1": float(F1[0]) * 100,
        }
    except Exception as e:
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
            "_error": str(e),
        }


def extract_medical_terms(text: str) -> Set[str]:
    """
    Extract potential medical/nursing terms from text.

    This uses a simple heuristic based on common German medical/nursing keywords.
    For production use, consider using a medical NER model or domain-specific lexicon.

    Args:
        text: Input text

    Returns:
        Set of potential medical terms (lowercase)
    """
    # Common German medical/nursing keywords (expandable)
    medical_keywords = {
        # General
        "patient",
        "patientin",
        "bewohner",
        "bewohnerin",
        # Vital signs
        "blutdruck",
        "puls",
        "herzfrequenz",
        "temperatur",
        "körpertemperatur",
        "sauerstoffsättigung",
        "atemfrequenz",
        "atmung",
        # Medications
        "medikament",
        "medikamente",
        "tablette",
        "tabletten",
        "tropfen",
        "salbe",
        "spritze",
        "infusion",
        "insulin",
        # Symptoms
        "schmerz",
        "schmerzen",
        "fieber",
        "übelkeit",
        "erbrechen",
        "durchfall",
        "verstopfung",
        "atemnot",
        "husten",
        "schnupfen",
        "schwellung",
        "rötung",
        "entzündung",
        "infektion",
        # Conditions & Diagnoses
        "diabetes",
        "demenz",
        "hypertonie",
        "hypotonie",
        "dekubitus",
        "sturz",
        "wunde",
        "fraktur",
        "verletzung",
        # Care activities
        "verbandswechsel",
        "wundversorgung",
        "mobilisation",
        "lagerung",
        "körperpflege",
        "toilettengang",
        "essen",
        "trinken",
        "nahrungsaufnahme",
        "flüssigkeitsaufnahme",
        # Medical procedures
        "untersuchung",
        "behandlung",
        "therapie",
        "operation",
        "eingriff",
        "katheter",
        "sonde",
        "drainage",
        # Status descriptors
        "stabil",
        "instabil",
        "verwirrt",
        "orientiert",
        "desorientiert",
        "bewusstlos",
        "ansprechbar",
        "mobil",
        "bettlägerig",
        # Body parts
        "kopf",
        "arm",
        "bein",
        "beine",
        "fuß",
        "füße",
        "hand",
        "hände",
        "rücken",
        "brust",
        "bauch",
        "haut",
        # Time/Frequency
        "täglich",
        "stündlich",
        "morgens",
        "mittags",
        "abends",
        "nachts",
    }

    text_lower = text.lower()
    found_terms = set()

    for term in medical_keywords:
        # Use word boundaries to avoid partial matches (e.g., "arm" in "Alarm")
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, text_lower):
            found_terms.add(term)

    return found_terms


def extract_medical_values(text: str) -> Dict[str, List[str]]:
    """
    Extract structured medical values from text (vital signs, medications, etc.).

    Args:
        text: Input text

    Returns:
        Dictionary with categories and extracted values
    """
    extracted = {
        "blood_pressure": [],
        "pulse": [],
        "temperature": [],
        "medications": [],
        "times": [],
    }

    # Blood pressure patterns (120/80, 120-80, etc.)
    bp_pattern = r"\b(\d{2,3})\s*[/-]\s*(\d{2,3})\b"
    extracted["blood_pressure"] = re.findall(bp_pattern, text)

    # Pulse (72, 72 bpm, 72/min, etc.)
    pulse_pattern = r"\b(\d{2,3})\s*(?:bpm|/min)?\b(?=.*(?:puls|herzfrequenz))"
    extracted["pulse"] = re.findall(pulse_pattern, text.lower())

    # Temperature (38.5°C, 38,5, etc.)
    temp_pattern = r"\b(\d{2}[.,]\d)\s*°?[Cc]?\b"
    extracted["temperature"] = re.findall(temp_pattern, text)

    # Time patterns (08:00, 8:00, 8.00)
    time_pattern = r"\b([0-2]?\d)[:.](\d{2})\b"
    extracted["times"] = re.findall(time_pattern, text)

    return extracted


def calculate_hallucination_score(
    source_text: str, generated_text: str, strict: bool = True
) -> Dict[str, Any]:
    """
    Detect potential hallucinations in LLM output.

    A hallucination occurs when the LLM generates medical terms or facts
    that don't appear in the source text. This is critical for medical
    applications where accuracy is paramount.

    Args:
        source_text: Original source text (e.g., OCR output)
        generated_text: LLM-generated output (e.g., summary)
        strict: If True, any medical term in output must be in source.
                If False, allows paraphrasing and synonyms.

    Returns:
        Dictionary with hallucination detection results:
        - hallucination_score: 0-100 (0 = no hallucinations, 100 = severe)
        - medical_terms_in_output: Medical terms found in output
        - medical_terms_in_source: Medical terms found in source
        - hallucinated_terms: Terms in output but not in source
        - hallucination_rate: Percentage of hallucinated terms
    """
    # Extract medical terms from both texts
    source_terms = extract_medical_terms(source_text)
    output_terms = extract_medical_terms(generated_text)

    if len(output_terms) == 0:
        # No medical terms in output = no hallucination
        return {
            "hallucination_score": 0.0,
            "medical_terms_in_output": 0,
            "medical_terms_in_source": len(source_terms),
            "hallucinated_terms": [],
            "hallucination_rate": 0.0,
            "validity": "safe",
        }

    # Find terms in output that don't appear in source
    hallucinated_terms = output_terms - source_terms

    # Calculate hallucination rate and score
    hallucination_rate = (len(hallucinated_terms) / len(output_terms)) * 100

    # Hallucination score (0-100, where 0 is best)
    # Higher weight for hallucinations in medical context
    hallucination_score = hallucination_rate

    # Validity assessment
    if hallucination_score == 0:
        validity = "safe"
    elif hallucination_score < 10:
        validity = "acceptable"
    elif hallucination_score < 30:
        validity = "warning"
    else:
        validity = "critical"

    return {
        "hallucination_score": hallucination_score,
        "medical_terms_in_output": len(output_terms),
        "medical_terms_in_source": len(source_terms),
        "hallucinated_terms": sorted(list(hallucinated_terms)),
        "hallucination_rate": hallucination_rate,
        "validity": validity,
    }


def calculate_semantic_similarity(
    reference: str, hypothesis: str, granularity: str = "sentence"
) -> Dict[str, float]:
    """
    Calculate semantic similarity using sentence embeddings.

    More sophisticated than ROUGE - captures meaning even when wording differs.
    Uses multilingual sentence transformers for German support.

    Args:
        reference: Reference text
        hypothesis: Generated text
        granularity: 'sentence' or 'document' level similarity

    Returns:
        Dictionary with semantic similarity scores
    """
    model = _get_sentence_transformer()

    if granularity == "document":
        # Document-level: single embedding per text
        embeddings = model.encode([reference, hypothesis])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        return {
            "semantic_similarity": float(similarity * 100),
            "granularity": "document",
        }

    else:  # sentence-level
        # Split into sentences (simple heuristic)
        ref_sentences = [s.strip() for s in reference.split(".") if s.strip()]
        hyp_sentences = [s.strip() for s in hypothesis.split(".") if s.strip()]

        if not ref_sentences or not hyp_sentences:
            return {
                "semantic_similarity": 0.0,
                "semantic_precision": 0.0,
                "semantic_recall": 0.0,
                "semantic_f1": 0.0,
                "granularity": "sentence",
            }

        # Compute embeddings
        ref_embeddings = model.encode(ref_sentences)
        hyp_embeddings = model.encode(hyp_sentences)

        # Compute pairwise similarities
        similarities = np.dot(ref_embeddings, hyp_embeddings.T) / (
            np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
            @ np.linalg.norm(hyp_embeddings, axis=1, keepdims=True).T
        )

        # Semantic recall: for each reference sentence, max similarity to hypothesis
        recall_scores = similarities.max(axis=1)
        semantic_recall = recall_scores.mean() * 100

        # Semantic precision: for each hypothesis sentence, max similarity to reference
        precision_scores = similarities.max(axis=0)
        semantic_precision = precision_scores.mean() * 100

        # Semantic F1
        semantic_f1 = (
            2
            * semantic_precision
            * semantic_recall
            / (semantic_precision + semantic_recall)
            if (semantic_precision + semantic_recall) > 0
            else 0
        )

        return {
            "semantic_similarity": float((semantic_recall + semantic_precision) / 2),
            "semantic_precision": float(semantic_precision),
            "semantic_recall": float(semantic_recall),
            "semantic_f1": float(semantic_f1),
            "granularity": "sentence",
        }


def calculate_information_preservation(
    reference: str, summary: str
) -> Dict[str, float]:
    """
    Calculate how well the summary preserves key information.

    Uses medical term extraction as a proxy for information preservation.

    Args:
        reference: Original text (ground truth)
        summary: LLM-generated summary

    Returns:
        Dictionary with preservation metrics
    """
    # Extract medical terms from both texts
    ref_terms = extract_medical_terms(reference)
    sum_terms = extract_medical_terms(summary)

    if len(ref_terms) == 0:
        return {
            "medical_term_recall": 100.0,
            "medical_term_precision": 100.0,
            "medical_term_f1": 100.0,
            "terms_in_reference": 0,
            "terms_in_summary": 0,
            "terms_preserved": 0,
        }

    # Calculate preservation metrics
    terms_preserved = len(ref_terms.intersection(sum_terms))

    recall = (terms_preserved / len(ref_terms)) * 100 if len(ref_terms) > 0 else 0
    precision = (terms_preserved / len(sum_terms)) * 100 if len(sum_terms) > 0 else 0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0
    )

    return {
        "medical_term_recall": recall,
        "medical_term_precision": precision,
        "medical_term_f1": f1,
        "terms_in_reference": len(ref_terms),
        "terms_in_summary": len(sum_terms),
        "terms_preserved": terms_preserved,
    }


def evaluate_vlm_transcription(
    reference: str, vlm_transcription: str
) -> Dict[str, float]:
    """
    Evaluate VLM transcription quality (image → text).

    VLM transcription is evaluated exactly like OCR output using standard
    transcription metrics: CER, WER, NED, accuracy, and vocabulary overlap.

    Args:
        reference: Ground truth text
        vlm_transcription: VLM-generated transcription

    Returns:
        Dictionary with transcription metrics (same as OCR evaluation)
    """
    return calculate_all_metrics(reference, vlm_transcription)


def evaluate_llm_output(
    reference: str,
    llm_output: str,
    source_text: Optional[str] = None,
    include_semantic: bool = True,
    include_hallucination: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of LLM-generated summaries or analysis.

    NOTE: For VLM transcription (image→text), use evaluate_vlm_transcription()
    which applies standard OCR metrics (CER/WER/NED).

    This function is for evaluating LLM summaries/analyses (text→text tasks),
    not direct transcription.

    Args:
        reference: Ground truth text or reference summary
        llm_output: LLM-generated summary/analysis
        source_text: Optional source text (e.g., OCR output that was summarized)
                    Used for hallucination detection
        include_semantic: Whether to include semantic similarity metrics
                         (requires sentence-transformers)
        include_hallucination: Whether to include hallucination detection
                              (requires source_text)

    Returns:
        Dictionary with comprehensive metrics:
        - ROUGE scores (token-based overlap)
        - BLEU score
        - Semantic similarity (context-aware)
        - Medical term preservation
        - Hallucination detection (if source_text provided)
        - Length metrics
    """
    metrics = {}

    # ROUGE scores (compare LLM output to ground truth)
    rouge_scores = calculate_rouge_scores(reference, llm_output)
    metrics.update(rouge_scores)

    # BLEU score
    metrics["bleu"] = calculate_bleu_score(reference, llm_output)

    # Semantic similarity (more sophisticated than ROUGE)
    if include_semantic:
        try:
            semantic_metrics = calculate_semantic_similarity(reference, llm_output)
            metrics.update(semantic_metrics)
        except ImportError:
            # sentence-transformers not installed
            metrics["semantic_similarity_error"] = "sentence-transformers not installed"

    # Information preservation (medical terms)
    info_metrics = calculate_information_preservation(reference, llm_output)
    metrics.update(info_metrics)

    # Hallucination detection (check if LLM invented medical facts)
    if include_hallucination and source_text:
        hallucination_metrics = calculate_hallucination_score(source_text, llm_output)
        metrics.update(
            {f"hallucination_{k}": v for k, v in hallucination_metrics.items()}
        )
    elif include_hallucination and not source_text:
        # Fall back to comparing against reference
        hallucination_metrics = calculate_hallucination_score(reference, llm_output)
        metrics.update(
            {f"hallucination_{k}": v for k, v in hallucination_metrics.items()}
        )

    # Length metrics
    metrics["reference_length"] = len(reference)
    metrics["output_length"] = len(llm_output)
    metrics["compression_ratio"] = (
        (len(llm_output) / len(reference)) * 100 if len(reference) > 0 else 0
    )

    # If source_text provided, also compare source vs reference (OCR quality)
    if source_text:
        source_info = calculate_information_preservation(reference, source_text)
        metrics["source_medical_term_recall"] = source_info["medical_term_recall"]

    return metrics


def format_llm_metrics_display(metrics: Dict[str, Any]) -> str:
    """
    Format LLM metrics dictionary for pretty console output.

    Args:
        metrics: Dictionary of metric names and values

    Returns:
        Formatted string for display
    """
    lines = ["=" * 80, "LLM EVALUATION METRICS", "=" * 80]

    # Hallucination detection (show first if available - critical for medical)
    if "hallucination_score" in metrics:
        validity = metrics.get("hallucination_validity", "unknown")
        validity_symbols = {
            "safe": "✓",
            "acceptable": "⚠",
            "warning": "⚠⚠",
            "critical": "✗",
        }
        symbol = validity_symbols.get(validity, "?")

        lines.append(f"\n[HALLUCINATION] Medical Hallucination Detection [{symbol}]:")
        lines.append(
            f"  Hallucination Score: {metrics['hallucination_score']:.2f}% (lower is better)"
        )
        lines.append(f"  Validity: {validity.upper()}")
        lines.append(
            f"  Medical terms in output: {metrics['hallucination_medical_terms_in_output']}"
        )
        lines.append(
            f"  Medical terms in source: {metrics['hallucination_medical_terms_in_source']}"
        )
        if metrics.get("hallucination_hallucinated_terms"):
            lines.append(
                f"  ⚠ Hallucinated terms: {', '.join(metrics['hallucination_hallucinated_terms'][:5])}"
            )

    # Semantic similarity
    if "semantic_similarity" in metrics:
        lines.append("\n[SEMANTIC] Semantic Similarity (context-aware):")
        lines.append(f"  Overall similarity: {metrics['semantic_similarity']:.2f}%")
        if "semantic_f1" in metrics:
            lines.append(f"  Semantic F1: {metrics['semantic_f1']:.2f}%")
            lines.append(f"  Semantic Precision: {metrics['semantic_precision']:.2f}%")
            lines.append(f"  Semantic Recall: {metrics['semantic_recall']:.2f}%")

    # ROUGE scores
    if "rouge1_f1" in metrics:
        lines.append("\n[ROUGE] ROUGE Scores (token-based overlap):")
        lines.append(f"  ROUGE-1 F1: {metrics['rouge1_f1']:.2f}%")
        lines.append(f"  ROUGE-2 F1: {metrics['rouge2_f1']:.2f}%")
        lines.append(f"  ROUGE-L F1: {metrics['rougeL_f1']:.2f}%")

    # BLEU score
    if "bleu" in metrics:
        lines.append(f"\n[BLEU] BLEU Score: {metrics['bleu']:.2f}%")

    # Medical term preservation
    if "medical_term_recall" in metrics:
        lines.append("\n[MEDICAL] Medical Term Preservation:")
        lines.append(f"  Terms in reference: {metrics['terms_in_reference']}")
        lines.append(f"  Terms in summary: {metrics['terms_in_summary']}")
        lines.append(f"  Terms preserved: {metrics['terms_preserved']}")
        lines.append(f"  Recall: {metrics['medical_term_recall']:.2f}%")
        lines.append(f"  Precision: {metrics['medical_term_precision']:.2f}%")
        lines.append(f"  F1: {metrics['medical_term_f1']:.2f}%")

    # Length analysis
    if "compression_ratio" in metrics:
        lines.append(f"\n[LENGTH] Length Analysis:")
        lines.append(f"  Reference length: {metrics['reference_length']} chars")
        lines.append(f"  Output length: {metrics['output_length']} chars")
        lines.append(f"  Compression ratio: {metrics['compression_ratio']:.1f}%")

    lines.append("=" * 80)
    return "\n".join(lines)


def interpret_llm_metrics(metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Provide interpretation of LLM metrics.

    Args:
        metrics: Dictionary of evaluation metrics

    Returns:
        Dictionary with interpretations for each aspect
    """
    interpretations = {}

    # Hallucination interpretation (MOST CRITICAL for medical)
    if "hallucination_score" in metrics:
        score = metrics["hallucination_score"]
        validity = metrics.get("hallucination_validity", "unknown")

        if validity == "safe":
            interpretations["hallucination"] = (
                "[✓ SAFE] No medical hallucinations detected - output is grounded in source"
            )
        elif validity == "acceptable":
            interpretations["hallucination"] = (
                "[⚠ ACCEPTABLE] Minor hallucinations detected - review output"
            )
        elif validity == "warning":
            interpretations["hallucination"] = (
                "[⚠⚠ WARNING] Significant hallucinations - some medical terms not in source"
            )
        else:  # critical
            interpretations["hallucination"] = (
                "[✗ CRITICAL] Severe hallucinations - output may contain fabricated medical information"
            )

    # Semantic similarity interpretation
    if "semantic_similarity" in metrics:
        sim = metrics["semantic_similarity"]
        if sim > 75:
            interpretations["semantic"] = (
                "[OK] Excellent semantic similarity - captures meaning well"
            )
        elif sim > 60:
            interpretations["semantic"] = (
                "[OK] Good semantic similarity - maintains core meaning"
            )
        elif sim > 40:
            interpretations["semantic"] = (
                "[WARNING] Moderate semantic similarity - some meaning may be lost"
            )
        else:
            interpretations["semantic"] = (
                "[WARNING] Low semantic similarity - significant divergence from reference"
            )

    # ROUGE interpretation
    if "rouge1_f1" in metrics:
        rouge1 = metrics["rouge1_f1"]
        if rouge1 > 50:
            interpretations["content_overlap"] = (
                "[OK] Good token-level overlap with reference"
            )
        else:
            interpretations["content_overlap"] = (
                "[WARNING] Low token overlap - LLM may be rephrasing heavily or missing info"
            )

    # Medical term preservation
    if "medical_term_recall" in metrics:
        recall = metrics["medical_term_recall"]
        if recall > 80:
            interpretations["medical_terms"] = (
                "[OK] Excellent preservation of medical terminology"
            )
        elif recall > 60:
            interpretations["medical_terms"] = (
                "[OK] Good preservation of medical terminology"
            )
        else:
            interpretations["medical_terms"] = (
                "[WARNING] Some medical terms may be missing from output"
            )

    # Compression ratio
    if "compression_ratio" in metrics:
        ratio = metrics["compression_ratio"]
        if 30 <= ratio <= 70:
            interpretations["compression"] = (
                "[OK] Good compression ratio for summarization"
            )
        elif ratio > 100:
            interpretations["compression"] = (
                "[WARNING] Output is longer than input (may be adding explanations)"
            )
        else:
            interpretations["compression"] = (
                "[WARNING] Very high compression - check if important details are lost"
            )

    return interpretations


def validate_json_structure(
    json_str: str, expected_schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate JSON structure and schema compliance.

    Args:
        json_str: JSON string to validate
        expected_schema: Optional dictionary defining expected keys and types

    Returns:
        Dictionary with validation results:
        - is_valid_json: Whether string is parseable JSON
        - is_schema_valid: Whether it matches expected schema (if provided)
        - missing_fields: List of expected fields that are missing
        - extra_fields: List of fields not in schema
        - parsed_data: Parsed JSON object (if valid)
    """
    result = {
        "is_valid_json": False,
        "is_schema_valid": None,
        "missing_fields": [],
        "extra_fields": [],
        "parsed_data": None,
    }

    # Check JSON validity
    try:
        parsed = json.loads(json_str)
        result["is_valid_json"] = True
        result["parsed_data"] = parsed

        # Check schema if provided
        if expected_schema is not None:
            if isinstance(parsed, dict):
                expected_keys = set(expected_schema.keys())
                actual_keys = set(parsed.keys())

                result["missing_fields"] = list(expected_keys - actual_keys)
                result["extra_fields"] = list(actual_keys - expected_keys)
                result["is_schema_valid"] = len(result["missing_fields"]) == 0
            else:
                result["is_schema_valid"] = False
                result["missing_fields"] = list(expected_schema.keys())

    except (json.JSONDecodeError, TypeError):
        result["is_valid_json"] = False

    return result


def normalize_medical_value(value: str, field_type: str = "general") -> str:
    """
    Normalize medical values for comparison.

    Handles nursing-specific formats for vital signs, medications, dates, etc.

    Args:
        value: Value to normalize
        field_type: Type of field (blood_pressure, pulse, temperature,
                   medication, date, time, general)

    Returns:
        Normalized value string
    """
    if not isinstance(value, str):
        return str(value)

    normalized = value.strip().lower()

    if field_type == "blood_pressure":
        # Normalize: 120/80, 120-80, 120 / 80 → 120/80
        normalized = re.sub(r"\s*[-/]\s*", "/", normalized)
        normalized = re.sub(r"\s+", "", normalized)  # Remove all spaces

    elif field_type == "pulse":
        # Normalize: 72, 72 bpm, 72/min → 72
        normalized = re.sub(r"\s*(bpm|/min|pro minute).*", "", normalized)
        normalized = normalized.strip()

    elif field_type == "temperature":
        # Normalize: 38.5, 38,5, 38.5°C, 38.5 grad → 38.5
        normalized = normalized.replace(",", ".")  # German decimal comma
        normalized = re.sub(r"[°]?\s*c(elsius)?.*", "", normalized)
        normalized = re.sub(r"\s*grad.*", "", normalized)
        normalized = normalized.strip()

    elif field_type == "medication":
        # For medications, preserve exact spelling but normalize format
        # Remove dosage info for name comparison
        normalized = re.sub(r"\d+\s*(mg|ml|g|tropfen)", "", normalized)
        normalized = normalized.strip()

    elif field_type == "time":
        # Normalize: 08:00, 8:00, 8.00 → 08:00
        normalized = normalized.replace(".", ":")
        parts = normalized.split(":")
        if len(parts) == 2:
            hour, minute = parts
            normalized = f"{int(hour):02d}:{int(minute):02d}"

    elif field_type == "date":
        # Normalize dates: DD.MM.YYYY, DD/MM/YYYY → DD.MM.YYYY
        normalized = normalized.replace("/", ".")

    else:  # general
        # Standard normalization: lowercase, collapse whitespace
        normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def evaluate_field_extraction(
    reference_fields: Dict[str, Any],
    extracted_fields: Dict[str, Any],
    field_types: Optional[Dict[str, str]] = None,
    tolerance: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluate field-level accuracy for structured extraction tasks.

    Enhanced version with nursing-specific normalization for vital signs,
    medications, dates, and other medical data.

    Args:
        reference_fields: Ground truth field values
        extracted_fields: LLM/VLM extracted field values
        field_types: Optional dict mapping field names to types
                    (blood_pressure, pulse, temperature, medication, etc.)
        tolerance: Optional dict mapping numeric field names to acceptable
                  tolerance (e.g., {'temperature': 0.1} allows ±0.1°C)

    Returns:
        Dictionary with per-field and overall accuracy metrics
    """
    total_fields = len(reference_fields)
    if total_fields == 0:
        return {
            "overall_accuracy": 100.0,
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "total_fields": 0,
            "field_accuracy": {},
        }

    field_types = field_types or {}
    tolerance = tolerance or {}

    exact_matches = 0
    fuzzy_matches = 0
    field_results = {}

    for field_name, ref_value in reference_fields.items():
        extracted_value = extracted_fields.get(field_name)

        if extracted_value is None:
            field_results[field_name] = {
                "match": False,
                "match_type": "missing",
                "reference": ref_value,
                "extracted": None,
            }
            continue

        field_type = field_types.get(field_name, "general")

        # Normalize both values
        if isinstance(ref_value, str) and isinstance(extracted_value, str):
            ref_normalized = normalize_medical_value(ref_value, field_type)
            ext_normalized = normalize_medical_value(extracted_value, field_type)

            # Exact match after normalization
            if ref_normalized == ext_normalized:
                exact_matches += 1
                field_results[field_name] = {
                    "match": True,
                    "match_type": "exact",
                    "reference": ref_value,
                    "extracted": extracted_value,
                }
                continue

            # For numeric fields, check tolerance
            if field_name in tolerance:
                try:
                    ref_num = float(ref_normalized.replace(",", "."))
                    ext_num = float(ext_normalized.replace(",", "."))
                    if abs(ref_num - ext_num) <= tolerance[field_name]:
                        fuzzy_matches += 1
                        field_results[field_name] = {
                            "match": True,
                            "match_type": "fuzzy",
                            "reference": ref_value,
                            "extracted": extracted_value,
                            "difference": abs(ref_num - ext_num),
                        }
                        continue
                except (ValueError, AttributeError):
                    pass

            # No match
            field_results[field_name] = {
                "match": False,
                "match_type": "mismatch",
                "reference": ref_value,
                "extracted": extracted_value,
            }
        else:
            # Non-string comparison
            is_match = ref_value == extracted_value
            if is_match:
                exact_matches += 1

            field_results[field_name] = {
                "match": is_match,
                "match_type": "exact" if is_match else "mismatch",
                "reference": ref_value,
                "extracted": extracted_value,
            }

    overall_accuracy = ((exact_matches + fuzzy_matches) / total_fields) * 100

    return {
        "overall_accuracy": overall_accuracy,
        "exact_matches": exact_matches,
        "fuzzy_matches": fuzzy_matches,
        "total_fields": total_fields,
        "field_accuracy": field_results,
    }
