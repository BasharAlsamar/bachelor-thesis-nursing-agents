# Evaluation Metrics Guide

This document explains all evaluation metrics implemented in this project for evaluating OCR outputs, VLM transcriptions, and LLM-generated summaries/extractions from nursing notes.

---

## Table of Contents

1. [Traditional NLP Metrics](#1-traditional-nlp-metrics)
   - [ROUGE](#rouge-recall-oriented-understudy-for-gisting-evaluation)
   - [BLEU](#bleu-bilingual-evaluation-understudy)
2. [Medical-Specific Metrics](#2-medical-specific-metrics)
   - [Medical Term Preservation](#medical-term-preservation)
   - [Hallucination Detection (Rule-Based)](#hallucination-detection-rule-based)
   - [Medical Value Extraction](#medical-value-extraction)
3. [Semantic Similarity Metrics](#3-semantic-similarity-metrics)
4. [DeepEval LLM-as-Judge Metrics](#4-deepeval-llm-as-judge-metrics)
   - [Hallucination Metric](#hallucination-metric-llm-judge)
   - [Contextual Precision & Recall](#contextual-precision--recall)
   - [Answer Relevancy](#answer-relevancy)
5. [OCR/Transcription Metrics](#5-ocrtranscription-metrics)
6. [Comparison Table](#6-comparison-table)
7. [LaTeX Citations](#7-latex-citations-bibtex)

---

## 1. Traditional NLP Metrics

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**Implementation:** `src/evaluation/llm_metrics.py::calculate_rouge_scores()`

**Purpose:** Measures n-gram overlap between reference and generated text. Originally designed for summarization evaluation.

**Variants:**
- **ROUGE-1:** Unigram (single word) overlap
- **ROUGE-2:** Bigram (two consecutive words) overlap  
- **ROUGE-L:** Longest Common Subsequence

**Metrics Returned:**
- **Precision:** What proportion of generated text appears in reference?
- **Recall:** What proportion of reference text appears in generated text?
- **F1-Score:** Harmonic mean of precision and recall

**Formula:**

```
ROUGE-N Recall = (Count of matching n-grams) / (Count of n-grams in reference)
ROUGE-N Precision = (Count of matching n-grams) / (Count of n-grams in hypothesis)
ROUGE-N F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Use Case:**
- Evaluating LLM summaries against ground truth nursing notes
- Comparing LLM output to expected structured text

**Strengths:**
- ✅ Fast computation
- ✅ Language-agnostic
- ✅ Well-established in NLP community

**Limitations:**
- ❌ Only measures surface-level overlap, not semantic meaning
- ❌ Sensitive to word order and exact phrasing
- ❌ May miss paraphrases

**Cite as:** `\cite{lin2004rouge}` (updated citation key)

---

### BLEU (Bilingual Evaluation Understudy)

**Implementation:** `src/evaluation/llm_metrics.py::calculate_bleu_score()`

**Purpose:** Originally for machine translation, measures n-gram precision with brevity penalty.

**Formula:**

```
BLEU = BP × exp(Σ wₙ log pₙ)

where:
- pₙ = modified n-gram precision
- wₙ = weights (typically uniform 1/4 for n=1,2,3,4)
- BP = brevity penalty = exp(1 - r/c) if c < r else 1
```

**Use Case:**
- Evaluating LLM summaries
- Complementary to ROUGE (precision-focused vs recall-focused)

**Strengths:**
- ✅ Emphasizes precision (important for medical accuracy)
- ✅ Includes brevity penalty (penalizes too-short outputs)
- ✅ Widely used in NLP research

**Limitations:**
- ❌ Precision-biased (may favor shorter outputs)
- ❌ Does not consider recall explicitly
- ❌ Surface-level matching only

**Cite as:** `\cite{papineni2002bleu}`

---

## 2. Medical-Specific Metrics

### Medical Term Preservation

**Implementation:** `src/evaluation/llm_metrics.py::extract_medical_terms()` + `calculate_information_preservation()`

**Purpose:** Measures how well critical medical/nursing terminology is preserved in LLM summaries.

**Methodology:**
1. Extract medical keywords from reference text (German nursing vocabulary)
2. Extract same from generated text
3. Calculate precision, recall, F1 for term overlap

**Medical Keywords Include:**
- Vital signs: `Blutdruck`, `Puls`, `Temperatur`, `Sauerstoffsättigung`
- Symptoms: `Schmerz`, `Fieber`, `Übelkeit`, `Atemnot`
- Medications: `Medikament`, `Tablette`, `Insulin`, `Salbe`
- Care activities: `Verbandswechsel`, `Mobilisation`, `Körperpflege`
- Conditions: `Diabetes`, `Demenz`, `Dekubitus`, `Wunde`

**Formula:**

```
Precision = |Medical Terms in Summary ∩ Medical Terms in Reference| / |Medical Terms in Summary|
Recall = |Medical Terms in Summary ∩ Medical Terms in Reference| / |Medical Terms in Reference|
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Use Case:**
- Ensuring LLM doesn't omit critical medical information
- Domain-specific evaluation for nursing documentation

**Strengths:**
- ✅ Domain-specific for medical context
- ✅ Focuses on critical information preservation
- ✅ Simple and interpretable

**Limitations:**
- ❌ Relies on predefined keyword list (incomplete coverage)
- ❌ Exact word matching (misses synonyms/paraphrases)
- ❌ Does not capture relationships between terms

**Cite as:** Custom metric, cite thesis or similar medical NLP work `\cite{wang2018clinical}`

---

### Hallucination Detection (Rule-Based)

**Implementation:** `src/evaluation/llm_metrics.py::calculate_hallucination_score()`

**Purpose:** Detect when LLM generates medical terms/facts not present in source text.

**Methodology:**
1. Extract medical terms from source (OCR output)
2. Extract medical terms from LLM summary
3. Identify "hallucinated" terms: present in summary but NOT in source
4. Calculate hallucination ratio

**Formula:**

```
Hallucinated Terms = Medical Terms in Summary \ Medical Terms in Source
Hallucination Score = |Hallucinated Terms| / |Medical Terms in Summary|
```

**Interpretation:**
- **0% hallucination:** All medical terms in summary are grounded in source
- **<10% hallucination:** Acceptable (minor paraphrasing/inference)
- **>30% hallucination:** Warning - significant fabrication

**Use Case:**
- Critical for medical safety - detecting fabricated information
- Ensuring LLM doesn't "make up" patient data

**Strengths:**
- ✅ Directly measures fabrication risk
- ✅ Simple and interpretable
- ✅ No external API needed

**Limitations:**
- ❌ May flag legitimate paraphrases as hallucinations
- ❌ Limited to keyword-based detection
- ❌ Cannot assess semantic hallucinations

**Cite as:** Custom metric inspired by hallucination detection literature `\cite{maynez2020faithfulness}`

---

### Medical Value Extraction

**Implementation:** `src/evaluation/llm_metrics.py::extract_medical_values()`

**Purpose:** Extract and validate structured medical data (vital signs, timestamps, medications).

**Extracted Values:**
- **Blood Pressure:** `120/80`, `130-85 mmHg`
- **Pulse:** `72 bpm`, `72/min`
- **Temperature:** `38.5°C`, `38,5°`
- **Times:** `08:00`, `14:30`

**Use Case:**
- Field-level evaluation for structured extraction tasks
- Validating LLM's ability to parse numeric medical data

**Formula:**

```
Value Match Rate = |Correctly Extracted Values| / |Expected Values|
```

---

## 3. Semantic Similarity Metrics

**Implementation:** `src/evaluation/llm_metrics.py::calculate_semantic_similarity()`

**Purpose:** Measure semantic meaning similarity beyond surface-level word matching.

**Methodology:**
1. Use multilingual Sentence Transformer model: `paraphrase-multilingual-MiniLM-L12-v2`
2. Generate embeddings for reference and hypothesis texts
3. Calculate cosine similarity between embeddings

**Models:**
- **Document-level:** Single embedding per text, cosine similarity
- **Sentence-level:** Embeddings per sentence, compute precision/recall

**Formula:**

```
Cosine Similarity = (v₁ · v₂) / (||v₁|| × ||v₂||)

Semantic Recall = mean(max similarity from each reference sentence to hypothesis)
Semantic Precision = mean(max similarity from each hypothesis sentence to reference)
```

**Use Case:**
- Detecting paraphrased content (LLM may reword correctly)
- Complement to ROUGE/BLEU for semantic understanding

**Strengths:**
- ✅ Captures semantic meaning, not just word overlap
- ✅ Handles paraphrases and synonyms
- ✅ Multilingual support (German nursing notes)

**Limitations:**
- ❌ Computationally expensive (embedding generation)
- ❌ Requires neural model (sentence-transformers)
- ❌ May give high similarity to semantically similar but factually wrong text

**Cite as:** `\cite{reimers2019sentencebert}`

---

## 4. DeepEval LLM-as-Judge Metrics

All DeepEval metrics use **GPT-4** as a judge to evaluate LLM outputs. This "LLM-as-a-judge" approach provides more nuanced evaluation than rule-based methods.

**Implementation:** `src/evaluation/deepeval_metrics.py`

**Requirements:**
- OpenAI API key: `export OPENAI_API_KEY="your-key"`
- Cost: ~$0.01-0.02 per evaluation

---

### Hallucination Metric (LLM Judge)

**Function:** `evaluate_hallucination()`

**Purpose:** Uses GPT-4 to detect hallucinated content (information not grounded in source context).

**Methodology:**
1. Provide source text as `context`
2. Provide LLM output as `actual_output`
3. GPT-4 judges: "Does the output contain information not present in context?"
4. Returns score 0-1 (0 = no hallucination, 1 = full hallucination)

**Parameters:**
```python
evaluate_hallucination(
    source_text: str,       # Ground truth/source (OCR output)
    llm_output: str,        # LLM-generated summary
    query: Optional[str],   # Original prompt/query
    threshold: float = 0.5, # Max acceptable score
    model: str = "gpt-4o"   # Judge model
)
```

**Returns:**
```python
{
    "hallucination_score": 0.15,      # 0-1 scale
    "is_hallucinated": False,         # score > threshold
    "threshold": 0.5,
    "reason": "Explanation from GPT-4",
    "validity": "excellent"            # excellent/good/acceptable/warning/critical
}
```

**Interpretation:**
- **≤0.1:** Excellent - minimal/no hallucination
- **≤0.3:** Good - minor interpretation acceptable
- **≤0.5:** Acceptable (threshold)
- **>0.5:** Failed - significant hallucination detected

**Use Case:**
- **Primary metric** for medical LLM evaluation
- Detects subtle hallucinations rule-based methods miss
- More context-aware than keyword matching

**Strengths:**
- ✅ Understands context and semantics
- ✅ Detects subtle fabrications
- ✅ Can reason about medical plausibility
- ✅ More accurate than rule-based detection

**Limitations:**
- ❌ Requires OpenAI API (cost + latency)
- ❌ Non-deterministic (small score variations)
- ❌ Black-box evaluation (less interpretable)

**Cite as:** `\cite{confident2024deepeval}` (DeepEval framework)

---

### Contextual Precision & Recall

**Function:** `evaluate_contextual_relevance()`

**Purpose:** Semantic evaluation of whether output captures relevant information from context.

**Metrics:**
1. **Contextual Precision:** Does every statement in output have grounding in context?
2. **Contextual Recall:** Does output cover all important information from context?

**Parameters:**
```python
evaluate_contextual_relevance(
    query: str,                    # User query
    llm_output: str,              # LLM response
    retrieval_context: List[str], # Context chunks (OCR output)
    expected_output: Optional[str], # Ground truth (for recall)
    threshold: float = 0.7
)
```

**Returns:**
```python
{
    "contextual_precision": 0.85,
    "precision_passed": True,
    "contextual_recall": 0.78,      # Only if expected_output provided
    "recall_passed": True,
    "contextual_f1": 0.81
}
```

**Use Case:**
- Beyond simple overlap - semantic understanding
- Evaluate if LLM properly uses OCR context
- Ensure no important information is dropped

**Cite as:** `\cite{confident2024deepeval}`

---

### Answer Relevancy

**Function:** `evaluate_answer_relevancy()`

**Purpose:** Measures whether LLM output is relevant to the input query.

**Use Case:**
- Structured extraction tasks (e.g., "Extract vital signs")
- Ensures LLM follows instructions

**Cite as:** `\cite{confident2024deepeval}`

---

## 5. OCR/Transcription Metrics

For evaluating **OCR outputs** or **VLM transcriptions** against ground truth text:

**Implementation:** `src/evaluation/metrics.py::calculate_all_metrics()`

**Metrics:**
- **CER (Character Error Rate):** Edit distance at character level
- **WER (Word Error Rate):** Edit distance at word level  
- **NED (Normalized Edit Distance):** Length-normalized Levenshtein distance
- **Accuracy:** `1 - NED`
- **Vocabulary Overlap:** Precision/Recall/F1 of unique words

**Use Case:**
- Evaluating EasyOCR, PaddleOCR transcription quality
- Evaluating VLM transcription (e.g., GPT-4V image → text)

**Cite as:** `\cite{morris2004levenshtein}` for edit distance metrics

---

## 6. Comparison Table

| Metric Category | Metric Name | Speed | LLM Required | Cost | Medical-Specific | Semantic Understanding |
|----------------|-------------|-------|--------------|------|------------------|----------------------|
| **Traditional NLP** | ROUGE | ⚡⚡⚡ | No | Free | No | ❌ Surface-level |
| | BLEU | ⚡⚡⚡ | No | Free | No | ❌ Surface-level |
| **Medical-Specific** | Medical Term Preservation | ⚡⚡ | No | Free | ✅ Yes | ❌ Keyword-based |
| | Hallucination (Rule) | ⚡⚡ | No | Free | ✅ Yes | ❌ Keyword-based |
| | Medical Value Extraction | ⚡⚡⚡ | No | Free | ✅ Yes | ❌ Regex-based |
| **Semantic** | Sentence Embeddings | ⚡ | No | Free* | No | ✅ Semantic |
| **LLM-as-Judge** | Hallucination (DeepEval) | ⚡ | Yes (GPT-4) | $0.01/eval | ✅ Context-aware | ✅ Full semantic |
| | Contextual Precision/Recall | ⚡ | Yes (GPT-4) | $0.01/eval | No | ✅ Full semantic |
| | Answer Relevancy | ⚡ | Yes (GPT-4) | $0.01/eval | No | ✅ Full semantic |
| **OCR/Transcription** | CER/WER/NED | ⚡⚡⚡ | No | Free | No | ❌ Character-level |

*Sentence embeddings free but require initial model download (~500MB)

---

## 7. LaTeX Citations (BibTeX)

Add these to your `Referenzen.bib` file:

```bibtex
@inproceedings{lin2004rouge,
  title={ROUGE: A package for automatic evaluation of summaries},
  author={Lin, Chin-Yew},
  booktitle={Text summarization branches out},
  pages={74--81},
  year={2004}
}

@inproceedings{papineni2002bleu,
  title={BLEU: a method for automatic evaluation of machine translation},
  author={Papineni, Kishore and Roukos, Salim and Ward, Todd and Zhu, Wei-Jing},
  booktitle={Proceedings of the 40th annual meeting of the Association for Computational Linguistics},
  pages={311--318},
  year={2002}
}

@inproceedings{reimers2019sentencebert,
  title={Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author={Reimers, Nils and Gurevych, Iryna},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={3982--3992},
  year={2019},
  organization={Association for Computational Linguistics}
}

@inproceedings{maynez2020faithfulness,
  title={On Faithfulness and Factuality in Abstractive Summarization},
  author={Maynez, Joshua and Narayan, Shashi and Bohnet, Bernd and McDonald, Ryan},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={1906--1919},
  year={2020}
}

@article{wang2018clinical,
  title={Clinical information extraction applications: a literature review},
  author={Wang, Yanshan and Wang, Liwei and Rastegar-Mojarad, Majid and Moon, Sungrim and Shen, Feichen and Afzal, Naveed and Liu, Sijia and Zeng, Yuqun and Mehrabi, Saeed and Sohn, Sunghwan and Liu, Hongfang},
  journal={Journal of biomedical informatics},
  volume={77},
  pages={34--49},
  year={2018},
  publisher={Elsevier}
}

@inproceedings{morris2004levenshtein,
  title={Lexical and similarity heuristics for string transformation},
  author={Morris, Andrew and Maheswari, Viktoria and Pudota, Nirmala},
  booktitle={Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics},
  pages={322--329},
  year={2004}
}

@misc{confident2024deepeval,
  title={DeepEval: The Open-Source LLM Evaluation Framework},
  author={Confident AI},
  year={2024},
  howpublished={\url{https://docs.confident-ai.com/}},
  note={Accessed: 2026-02-18}
}

@article{zheng2023judging,
  title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
  author={Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and Zhuang, Siyuan and Wu, Zhanghao and Zhuang, Yonghao and Lin, Zi and Li, Zhuohan and Li, Dacheng and Xing, Eric and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```

---

## Usage Examples

### Example 1: Evaluate LLM Summary

```python
from src.evaluation.llm_metrics import evaluate_llm_output

result = evaluate_llm_output(
    reference="Patient Müller, 85 J., Diabetes Typ 2. Blutdruck 140/85 mmHg...",
    llm_output="Frau Müller (85) hat Diabetes. Blutdruck erhöht (140/85)...",
    enable_semantic=True
)

print(f"ROUGE-L F1: {result['rouge']['rougeL_f1']:.2f}%")
print(f"Medical Term Preservation: {result['medical_terms']['preservation_f1']:.2f}%")
print(f"Hallucination Score: {result['hallucination']['hallucination_percentage']:.2f}%")
```

### Example 2: DeepEval Hallucination Check

```python
from src.evaluation.deepeval_metrics import evaluate_hallucination

result = evaluate_hallucination(
    source_text="Patient has fever, temperature 38.5°C",
    llm_output="Patient has high fever and was given antibiotics",
    threshold=0.5
)

print(f"Hallucination Score: {result['hallucination_score']:.2%}")
print(f"Status: {result['validity']}")
print(f"Hallucinated: {result['is_hallucinated']}")
```

### Example 3: Compare OCR+LLM vs VLM

```python
from src.evaluation.llm_metrics import evaluate_llm_output
from src.evaluation.deepeval_metrics import evaluate_hallucination

ground_truth = "Patient Schmidt, 72 J., Sturz heute morgen..."

# Evaluate OCR+LLM approach
ocr_llm_metrics = evaluate_llm_output(ground_truth, ocr_llm_output)
ocr_llm_halluc = evaluate_hallucination(ground_truth, ocr_llm_output)

# Evaluate VLM approach  
vlm_metrics = evaluate_llm_output(ground_truth, vlm_output)
vlm_halluc = evaluate_hallucination(ground_truth, vlm_output)

# Compare
print(f"OCR+LLM ROUGE-L: {ocr_llm_metrics['rouge']['rougeL_f1']:.2f}")
print(f"VLM ROUGE-L: {vlm_metrics['rouge']['rougeL_f1']:.2f}")
```

---

## Recommendations for Thesis

### For OCR Evaluation (500 samples):
- ✅ Use **CER, WER, NED** (transcription accuracy)
- ✅ Use **vocabulary overlap** (word-level accuracy)

### For LLM Summary Evaluation (30-50 samples):
- ✅ Use **ROUGE, BLEU** (standard baselines)
- ✅ Use **Medical Term Preservation** (domain-specific)
- ✅ Use **Hallucination Metric (DeepEval)** (primary medical safety metric)
- ⚬ Optional: **Semantic Similarity** (if emphasizing paraphrase handling)

### For Comparing OCR+LLM vs VLM:
- ✅ All metrics above applied to both approaches
- ✅ **Cost analysis** ($ per sample)
- ✅ **Speed analysis** (seconds per sample)
- ✅ **Hallucination comparison** (critical for medical safety argument)

---

## Contact & Support

For questions about metric implementation or interpretation:
- See code documentation in `src/evaluation/llm_metrics.py`
- See DeepEval docs: https://docs.confident-ai.com/
- Check test files: `test_metrics_fixes.py`, `test_deepeval_corrected.py`
