"""
DeepEval integration for LLM evaluation in nursing notes analysis.

This module provides DeepEval-based metrics for evaluating LLM outputs,
particularly focused on medical/nursing contexts where accuracy is critical.

Key metrics implemented:
1. Hallucination/Faithfulness: Ensures LLM doesn't fabricate medical information
2. Contextual Precision/Recall: Semantic understanding beyond token matching
3. Answer Relevancy: For structured extraction tasks

References:
- DeepEval Documentation: https://docs.confident-ai.com/
- DeepEval Metrics: https://docs.confident-ai.com/docs/metrics-introduction

Installation:
    pip install deepeval

Usage:
    from src.evaluation.deepeval_metrics import (
        evaluate_hallucination,
        evaluate_contextual_relevance,
        evaluate_comprehensive
    )

    # Check for hallucinations
    result = evaluate_hallucination(
        source_text="Patient has fever, temperature 38.5°C",
        llm_output="Patient has high fever and was given antibiotics"
    )
"""

from typing import Dict, List, Optional, Any
import warnings

# DeepEval imports (lazy loading to avoid import errors if not installed)
try:
    from deepeval.metrics import (
        HallucinationMetric,  # Primary metric for medical text
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        AnswerRelevancyMetric,
    )
    from deepeval.test_case import LLMTestCase
    from deepeval import evaluate

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    warnings.warn(
        "DeepEval not installed. Install with: pip install deepeval\n"
        "This is required for advanced LLM evaluation metrics.\n"
        "Note: DeepEval requires OpenAI API key for LLM-as-judge evaluation."
    )


def evaluate_hallucination(
    source_text: str,
    llm_output: str,
    query: Optional[str] = None,
    threshold: float = 0.5,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate hallucination using DeepEval's HallucinationMetric.

    Detects if the LLM output contains hallucinated information not present
    in the source/context. Critical for medical applications where fabricated
    information is dangerous.

    Args:
        source_text: Source/context text (e.g., OCR output, original note)
        llm_output: LLM-generated output (e.g., summary, extraction)
        query: Optional query/input that was given to the LLM
        threshold: Maximum acceptable hallucination score (0-1, default 0.5)
                  Lower is better (0 = no hallucination, 1 = full hallucination)
        model: Optional LLM model to use for evaluation (default: gpt-4o)

    Returns:
        Dictionary with:
        - hallucination_score: Score from 0-1 (0 = no hallucination, 1 = total hallucination)
        - is_hallucinated: Boolean, whether score exceeds threshold
        - reason: Explanation from the metric

    Example:
        >>> result = evaluate_hallucination(
        ...     source_text="Patient has fever, temp 38.5°C",
        ...     llm_output="Patient has high fever and was given antibiotics",
        ...     query="Summarize the patient's condition"
        ... )
        >>> print(result['hallucination_score'])
        0.3
    """
    if not DEEPEVAL_AVAILABLE:
        raise ImportError(
            "DeepEval is not installed. Install with: pip install deepeval"
        )

    # Create test case with context parameter
    test_case = LLMTestCase(
        input=query or "Summarize the following context",
        actual_output=llm_output,
        context=[source_text],  # Use context, not retrieval_context
    )

    # Create and run hallucination metric
    if model:
        metric = HallucinationMetric(threshold=threshold, model=model)
    else:
        metric = HallucinationMetric(threshold=threshold)

    metric.measure(test_case)

    # Note: For HallucinationMetric, lower score is better (0 = no hallucination)
    return {
        "hallucination_score": metric.score,
        "is_hallucinated": metric.score > threshold,  # Score ABOVE threshold is bad
        "threshold": threshold,
        "reason": metric.reason if hasattr(metric, "reason") else None,
        "validity": _interpret_hallucination(metric.score),
    }


# Note: FaithfulnessMetric is not needed for this use case.
# HallucinationMetric is sufficient for detecting fabricated medical information.
# If you need FaithfulnessMetric for RAG systems, it can be added separately.


def evaluate_contextual_relevance(
    query: str,
    llm_output: str,
    retrieval_context: List[str],
    expected_output: Optional[str] = None,
    threshold: float = 0.7,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate contextual precision and recall.

    More sophisticated than ROUGE - uses semantic understanding to evaluate
    whether the LLM output captures relevant information from context.

    Args:
        query: The question/prompt given to the LLM
        llm_output: LLM-generated response
        retrieval_context: List of context chunks (e.g., OCR outputs, reference texts)
        expected_output: Optional expected/ground truth output for recall calculation
        threshold: Minimum score threshold (0-1)
        model: Optional LLM model for evaluation

    Returns:
        Dictionary with contextual precision and recall scores

    Example:
        >>> result = evaluate_contextual_relevance(
        ...     query="Fasse die Pflegenotiz zusammen",
        ...     llm_output="Patient hat Schmerzen",
        ...     retrieval_context=["Bewohner klagt über Schmerzen im Arm"],
        ...     expected_output="Bewohner hat Armschmerzen"
        ... )
    """
    if not DEEPEVAL_AVAILABLE:
        raise ImportError(
            "DeepEval is not installed. Install with: pip install deepeval"
        )

    results = {}

    # Contextual Precision
    test_case_precision = LLMTestCase(
        input=query, actual_output=llm_output, retrieval_context=retrieval_context
    )

    if model:
        precision_metric = ContextualPrecisionMetric(threshold=threshold, model=model)
    else:
        precision_metric = ContextualPrecisionMetric(threshold=threshold)

    precision_metric.measure(test_case_precision)

    results["contextual_precision"] = precision_metric.score
    results["precision_passed"] = precision_metric.score >= threshold
    results["precision_reason"] = getattr(precision_metric, "reason", None)

    # Contextual Recall (requires expected_output)
    if expected_output:
        test_case_recall = LLMTestCase(
            input=query,
            actual_output=llm_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )

        if model:
            recall_metric = ContextualRecallMetric(threshold=threshold, model=model)
        else:
            recall_metric = ContextualRecallMetric(threshold=threshold)

        recall_metric.measure(test_case_recall)

        results["contextual_recall"] = recall_metric.score
        results["recall_passed"] = recall_metric.score >= threshold
        results["recall_reason"] = getattr(recall_metric, "reason", None)

        # Calculate F1 if we have both
        precision = results["contextual_precision"]
        recall = results["contextual_recall"]
        results["contextual_f1"] = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0
        )
    else:
        results["contextual_recall"] = None
        results["recall_passed"] = None
        results["contextual_f1"] = None

    return results


def evaluate_answer_relevancy(
    query: str, llm_output: str, threshold: float = 0.7, model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate if LLM output is relevant to the query.

    Useful for structured extraction tasks to ensure the LLM answered
    what was asked.

    Args:
        query: The question/prompt
        llm_output: LLM response
        threshold: Minimum relevancy score
        model: Optional LLM model for evaluation

    Returns:
        Dictionary with relevancy score and assessment
    """
    if not DEEPEVAL_AVAILABLE:
        raise ImportError(
            "DeepEval is not installed. Install with: pip install deepeval"
        )

    test_case = LLMTestCase(input=query, actual_output=llm_output)

    if model:
        metric = AnswerRelevancyMetric(threshold=threshold, model=model)
    else:
        metric = AnswerRelevancyMetric(threshold=threshold)

    metric.measure(test_case)

    return {
        "answer_relevancy": metric.score,
        "is_relevant": metric.score >= threshold,
        "reason": getattr(metric, "reason", None),
    }


def evaluate_comprehensive(
    source_text: str,
    llm_output: str,
    query: Optional[str] = None,
    expected_output: Optional[str] = None,
    hallucination_threshold: float = 0.5,
    contextual_threshold: float = 0.7,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation using multiple DeepEval metrics.

    Focuses on hallucination detection and contextual relevance for medical text.

    Args:
        source_text: Original source text (OCR, reference)
        llm_output: LLM-generated output
        query: Optional query/prompt that was given to LLM
        expected_output: Optional expected/reference output for comparison
        hallucination_threshold: Maximum acceptable score for hallucination (0-1, lower is better)
        contextual_threshold: Minimum acceptable score for contextual metrics (0-1, higher is better)
        model: Optional LLM model for evaluation (default: gpt-4o)

    Returns:
        Dictionary with all metrics and overall assessment

    Note:
        This uses LLM-as-a-judge evaluation. Requires OpenAI API key:
        export OPENAI_API_KEY="your-key"
    """
    if not DEEPEVAL_AVAILABLE:
        raise ImportError(
            "DeepEval is not installed. Install with: pip install deepeval"
        )

    results = {}

    # 1. Hallucination check (PRIMARY METRIC - most important for medical)
    hallucination = evaluate_hallucination(
        source_text=source_text,
        llm_output=llm_output,
        query=query,
        threshold=hallucination_threshold,
        model=model,
    )
    results["hallucination"] = hallucination

    # 2. Contextual relevance (if we have query)
    if query:
        contextual = evaluate_contextual_relevance(
            query=query,
            llm_output=llm_output,
            retrieval_context=[source_text],
            expected_output=expected_output,
            threshold=contextual_threshold,
            model=model,
        )
        results["contextual"] = contextual

    # 3. Answer relevancy (if we have query)
    if query:
        relevancy = evaluate_answer_relevancy(
            query=query,
            llm_output=llm_output,
            threshold=contextual_threshold,
            model=model,
        )
        results["relevancy"] = relevancy

    # Overall assessment
    # Pass if: no hallucinations AND (if query provided: contextually relevant)
    results["overall_passed"] = not hallucination["is_hallucinated"]

    if query and "contextual" in results:
        results["overall_passed"] = (
            results["overall_passed"] and results["contextual"]["precision_passed"]
        )
        # Only check recall if we have expected_output
        if results["contextual"]["recall_passed"] is not None:
            results["overall_passed"] = (
                results["overall_passed"] and results["contextual"]["recall_passed"]
            )

    return results


def _interpret_hallucination(score: float) -> str:
    """Interpret hallucination score for medical context (lower is better)."""
    if score <= 0.1:
        return "excellent"
    elif score <= 0.3:
        return "good"
    elif score <= 0.5:
        return "acceptable"
    elif score <= 0.7:
        return "warning"
    else:
        return "critical"


def format_deepeval_results(results: Dict[str, Any]) -> str:
    """
    Format DeepEval evaluation results for display.

    Args:
        results: Results from evaluate_comprehensive()

    Returns:
        Formatted string for console output
    """
    lines = ["=" * 80, "DEEPEVAL EVALUATION RESULTS", "=" * 80]

    # Hallucination
    if "hallucination" in results:
        h = results["hallucination"]
        validity_symbols = {
            "excellent": "✓✓",
            "good": "✓",
            "acceptable": "○",
            "warning": "⚠",
            "critical": "✗",
        }
        symbol = validity_symbols.get(h["validity"], "?")

        lines.append(f"\n[HALLUCINATION] Detection [{symbol}]:")
        lines.append(
            f"  Score: {h['hallucination_score']:.2%} (threshold: {h['threshold']:.0%}, lower is better)"
        )
        lines.append(f"  Status: {h['validity'].upper()}")
        lines.append(f"  Hallucinated: {'Yes ⚠' if h['is_hallucinated'] else 'No ✓'}")
        if h.get("reason"):
            lines.append(f"  Reason: {h['reason'][:100]}...")

    # Contextual relevance
    if "contextual" in results:
        c = results["contextual"]
        lines.append(f"\n[CONTEXTUAL] Semantic Precision & Recall:")
        lines.append(f"  Precision: {c['contextual_precision']:.2%}")
        if c["contextual_recall"] is not None:
            lines.append(f"  Recall: {c['contextual_recall']:.2%}")
            lines.append(f"  F1: {c['contextual_f1']:.2%}")

    # Answer relevancy
    if "relevancy" in results:
        r = results["relevancy"]
        lines.append(f"\n[RELEVANCY] Answer Relevancy:")
        lines.append(f"  Score: {r['answer_relevancy']:.2%}")
        lines.append(f"  Relevant: {'Yes' if r['is_relevant'] else 'No'}")

    # Overall
    if "overall_passed" in results:
        status = "PASSED ✓" if results["overall_passed"] else "FAILED ✗"
        lines.append(f"\n[OVERALL] {status}")

    lines.append("=" * 80)
    return "\n".join(lines)
