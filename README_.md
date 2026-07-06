# Bachelor Thesis: Agentic Handwriting Extraction for Nursing Robotics

Dieses Repository bündelt die Arbeit an einer Bachelorarbeit zur automatisierten Verarbeitung handschriftlicher Pflegenotizen. Der Schwerpunkt liegt auf dem Vergleich und der Kombination von OCR-Pipelines, Vision-Language-Modellen, LLM-Nachverarbeitung und agentenbasierten Workflows.

## Projektüberblick

Das Projekt enthält vier zentrale Arbeitsstränge:

1. OCR-Evaluierung auf synthetischen und realen Pflegenotizen
2. Standalone LLM-Nachverarbeitung von OCR-Ergebnissen
3. Frame-/Bildauswahl mit einem Vision-Routing-Agenten
4. Synthese und Analyse von Trainings- und Testdaten

Zusätzlich enthält das Repository die Thesis-Quelle in LaTeX, mehrere Notebook-Experimente und Hilfsskripte für Datensätze, Referenztexte und Evaluation.

## Visuelle Beispiele

Die folgenden Abbildungen liegen zusätzlich in `assets/readme/`, damit sie auch online (z. B. auf GitHub) stabil angezeigt werden.

### Datengrundlage: Synthetischer Datensatz

- Motivation: Mangel an frei verfügbaren Daten und strenge DSGVO-Auflagen erfordern einen synthetischen, aber hochrealistischen Datensatz.
- Schriftarten-Validierung: Programmatische Analyse mit `fonttools`; nur 24 von 35 Fonts unterstützen den deutschen medizinischen Zeichensatz (`ä`, `ö`, `ü`, `ß`) fehlerfrei.
- Textkorpus und Metadaten: 50 einzigartige klinische Szenarien (JSON) mit realistischer Demografie und Pflege-Terminologie (z. B. RR, BZ).
- Implementierung der Datengenerierung: Modulares Python-Framework auf Basis von Pillow (PIL), das textbasierte Szenarien und validierte Schriftarten vollautomatisch in annotierte, realistische Bilddaten überführt.

### Datenstratifizierung

- Pflegegrad 1 (gering): Kurze, routinemäßige Notizen (ca. 200 bis 350 Zeichen).
- Pflegegrad 2 bis 3 (mittel): Standarddokumentation (ca. 400 bis 800 Zeichen).
- Pflegegrad 4 bis 5 (hoch): Komplexe Berichte (z. B. Palliativ) mit bis zu 2.500+ Zeichen.

### Visuelle Synthese, Variabilität und Papiersimulation

- Dynamisches Layout: DIN A5 für kurze und A4 für lange Texte, inklusive Lineaturen (liniert, kariert) sowie variierenden Papierfarben.
- Simulation realer Schreibprozesse:
    - Modus „Gestresst“: Hastig, größere Schrift, starke Neigung, eng.
    - Modus „Sorgfältig“: Sauber ausgerichtet, kleiner, gerader.
- Tintensimulation: Alpha-Blending für ungleichmäßigen Tintenfluss.

### Synthetische Beispielbilder

Die folgenden Beispiele zeigen unterschiedliche Schreibmodi, Layouts und Tintenverläufe aus dem synthetischen N30-Datensatz:

![Synthetische Notiz N30 - sample 0105](assets/readme/sample_0105.png)

![Synthetische Notiz N30 - sample 0155](assets/readme/sample_0155.png)

![Synthetische Notiz N30 - sample 0851](assets/readme/sample_0851.png)

![Synthetische Notiz N30 - sample 0857](assets/readme/sample_0857.png)

![Synthetische Pflegenotizen - kleine Stichprobe](assets/readme/N6_samples.png)

![Synthetische Pflegenotizen - groessere Stichprobe](assets/readme/N48_samples.png)

### Datenverteilungen (Datensatz)

Die vier Verteilungsplots dokumentieren die kontrollierte Zusammensetzung des Testsets (`test_set_559`) und dienen als Nachweis der Stratifizierung über Layout-, Stil- und Pflegekontext-Merkmale.

![Verteilung Seite/Farbe/Papiertyp](assets/readme/labels_page_color_paper_type_counts.png)

![Verteilung Schreibstil/Profile](assets/readme/labels_writing_style_profile_counts.png)

![Verteilung Szenario/Alter/Pflegestufe](assets/readme/scenarios_age_care_level_counts.png)

![Label-Verteilungen im Testset 559](assets/readme/test_set_559_label_distributions.png)

### Task 1: Evaluation of OCRs

Die folgende Abbildung zeigt die vergleichende Auswertung der OCR-Modelle aus Task 1 auf Basis der Transkriptionsmetriken.

![Task 1 - Evaluation of OCRs](assets/readme/task1_ocrs.png)

### Task 1: Evaluation of VLMs

Die folgende Abbildung zeigt die Task-1-Ergebnisse für Vision-Language-Modelle im direkten Vergleich.

![Task 1 - Evaluation of VLMs](assets/readme/task1_vlms.png)

### Task 2: Downstream Understanding and Extraction

**Ansatz 1: OCR + LLM Pipelines**

- Datenbasis: Das exakt balancierte N=48 Subset.
- Die Herausforderung: LLMs müssen mit den fehlerhaften, halluzinierten oder fragmentierten Texten der OCR-Engines aus Task 1 arbeiten.
- Umfang: Alle 3 OCR-Modelle wurden kreuzweise mit allen 3 LLMs getestet.

**Ansatz 2: End-to-end Vision-Language Models (VLMs)**

- Datenbasis: Dasselbe N=48 Subset (für faire Vergleichbarkeit).
- Die Herausforderung: Multimodale Modelle (VLMs) verarbeiten das Bild direkt, ohne vorherige OCR-Transkription.
- Bewertung: Fokus auf lexikalische Ähnlichkeit (ROUGE), semantische Äquivalenz (BERTScore) und domänenspezifische Genauigkeit.

![Task 2 - Ansatz 2 (End-to-end VLMs)](assets/readme/task2_approach2.png)

![Task 2 - Ansatzvergleich](assets/readme/Task2_approach1.png)

### Proof-of-Concept & Demo

Systemintegration und interaktiver Proof-of-Concept.

![Proof of Concept Demo](assets/readme/proof_demo.png)

**Performance und Latenz (Die Praxis-Hürde)**

- Kaltstart (Initial Run): ca. 3 Minuten. Der Flaschenhals ist die lokale Text-to-Speech (TTS) Generierung auf der RTX 3060 (ca. 2:22 Min).
- Die Extraktion (effizient): Frame-Selection (ca. 10 Sek) und Pixtral-API (ca. 2 bis 5 Sek) arbeiten sehr schnell.
- Lösung: Audio-Caching für Standard-Antworten und kontinuierliche Status-Updates überbrücken Wartezeiten.

**Robustheit im echten Umfeld (LG Webcam)**

- Lichtabhängigkeit: Bei schwachem Licht greift der Vision-Router sicher ein (Score fällt unter 70, Human-in-the-Loop wird getriggert).
- Erfolgsrate: Bei adäquater Beleuchtung wurden neue, authentische Notizen zu 100% korrekt extrahiert.

**Human-Robot Interaction (HRI) Interface**

- Personalisiertes Audio: Qwen-TTS-Base-1.7 erzeugt eine konsistente, geklonte Stimme für die dynamische "Ask Human"-Interaktion.
- Visuelles Feedback: Ein animiertes, minimalistisches digitales Gesicht spiegelt den Systemstatus wider (Zuhören, Denken, Sprechen), um die Akzeptanz des Klinikpersonals zu erhöhen.

Darüber hinaus wird die akustische Rückmeldung durch eine visuelle, kontextsensitive grafische Benutzeroberfläche ergänzt. Wie in Abbildung 3.14 dargestellt, verfügt das System über ein minimalistisches, animiertes digitales Gesicht, dessen Ausdruck sich dynamisch je nach aktuellem Prozesszustand verändert. So zeigt das Gesicht beispielsweise während des Scannens von Dokumenten einen neutralen, konzentrierten Ausdruck, wechselt während der API-Inferenz zu einer Animation, die Nachdenken signalisiert, und nimmt einen offenen, sprechenden Ausdruck an, wenn eine Rückfrage an die Pflegekraft erfolgt. Diese visuelle Anthropomorphisierung liefert dem Nutzer eine unmittelbare, nonverbale Rückmeldung zum Systemstatus, überbrückt Wartezeiten und gestaltet die Interaktion mit dem Roboter für das klinische Personal deutlich natürlicher und zugänglicher.

### Human-in-the-Loop: Interaktiver Workflow

Ziel: Kollaborative Assistenz statt starrer Pipeline. Menschliche Expertise wird bei Unsicherheiten gezielt eingebunden.

Die 3 Workflow-Status:

- Status-Updates (Latenz-Überbrückung): Der Nutzer wird stets informiert, was das System gerade tut (z. B. "Ich habe das beste Bild gefunden" oder "Ich erstelle nun die Zusammenfassung").
- Interaktive Klärung (Ask Human): Bei Unsicherheit (z. B. unleserliche Schrift) pausiert der Agent. Der Roboter stellt eine gezielte Rückfrage (z. B. "Ist das zweite Medikament Novamin?"). Die gesprochene Antwort löst das Problem.
- Autonome Extraktion (Success): Bei klarer Lesbarkeit wird die Struktur direkt generiert und der erfolgreiche Speicherabschluss verbal bestätigt.

![HRI Interface](assets/readme/HRI_Interface.png)

Um die praktische Anwendbarkeit der vorgeschlagenen Lösung zu validieren, wurde das Modell mit der insgesamt besten Leistung (Pixtral-8B via API) in die Python-basierte Proof-of-Concept-Software integriert. Die Demonstration erfolgte auf einem lokalen Rechnersystem unter Verwendung einer Windows Subsystem for Linux (WSL)-Umgebung. Da WSL von Haus aus keinen direkten Hardwarezugriff auf USB-Peripheriegeräte wie Webcams bietet, wurde ein spezieller Python-Bridge-Server auf dem Windows-11-Host implementiert. Dieser Server erfasst den Hardware-Videostream und leitet die Einzelbilder an die WSL-Umgebung weiter, wodurch eine nahtlose Verbindung zwischen den physischen Sensoren und der Linux-basierten KI-Pipeline gewährleistet wird.

Um eine konsistente Persönlichkeit für den Roboterassistenten zu schaffen, wurde das Modell Qwen-TTS-Base-1.7 eingesetzt, um eine maßgeschneiderte, geklonte Stimme zu generieren. Dies ermöglicht es medizinischen Einrichtungen, das Stimmenprofil des Assistenten individuell anzupassen. Zur weiteren Optimierung der zuvor erörterten Systemlatenz werden alle wiederkehrenden Statusmeldungen (z. B. "Ich starte die Aufnahme" oder "Ich habe das beste Bild gefunden") vorab generiert und lokal zwischengespeichert. Dieser hybride Audio-Ansatz gewährleistet eine sofortige akustische Rückmeldung bei Standard-Zustandswechseln, während die rechenintensive dynamische TTS-Generierung ausschließlich für unvorhersehbare Rückfragen an den Menschen (z. B. zur Überprüfung spezifischer Medikamentennamen) reserviert bleibt.

### Edge-AI Frame Selection (Vision Routing Agent)

- Das Problem in der Servicerobotik: Bewegungsunschärfe, wechselndes Licht und unklare Bildausschnitte beeinträchtigen die Erkennung stark.
- Schwäche klassischer Filter: Mathematische Filter (z. B. Laplace-Varianz) bewerten nur die Schärfe, erfassen jedoch nicht den semantischen Kontext (z. B. scharfer Hintergrund vs. leicht unscharfes, aber lesbares Dokument).
- Die Lösung: Kaskadenarchitektur (Small-to-Large). Der eigentlichen Extraktionspipeline wird ein lokaler Vision Routing Agent vorgeschaltet, der nur gültige Bilder durchlässt.
    - Leichtgewichtiges Edge-Modell: Ein kompaktes VLM (Qwen3-VL-4B-Instruct) läuft lokal per 4-Bit-Quantisierung. Bilder werden gezielt herunterskaliert, um VRAM-Überläufe zu verhindern.
    - Strukturierte Bewertung (Few-Shot Prompting): Das Modell generiert deterministisch eine Punktzahl (0-100).
    - Dynamisches Batch-Sampling: Anstatt Einzelbilder sequenziell zu prüfen (zu hohe Latenz für HRI), werden 8 repräsentative Bilder parallel evaluiert. Nur das beste Bild (Score ≥ 70) wird an die großen Modelle (8B/14B) weitergeleitet.

Die folgende Abbildung zeigt die Pipeline-Architektur des Routing-Ansatzes:

![Pipeline Architecture](assets/readme/Pipeline_%20architecture.png)

Das folgende Ergebnis zeigt die Frame-Bewertung und Auswahl des Routing-Agenten:

![Routing Agent Scores (Best Frame)](assets/readme/frame_agent_scores_best.png)

## Repository-Struktur

```text
bachelor-thesis-nursing-agents/
├── README.md
├── requirements.txt
├── library_versions.json
├── assets/
│   └── readme/
├── configs/                     # aktuell leer / als Ablage fuer Konfiguration vorgesehen
├── data/
│   ├── raw/
│   ├── processed/
│   ├── synthetic/
│   ├── N30/
│   └── test_set_*.json/.txt
├── docs/
│   ├── EVALUATION_METRICS_GUIDE.md
│   ├── OCR_LLM_PIPELINE_GUIDE.md
│   ├── STANDALONE_LLM_PIPELINE.md
│   └── TEST_SET_SPECIFICATION.md
├── experiments/
│   ├── 00_json_llm_test.ipynb
│   ├── Pixtral/
│   ├── Real_images/
│   ├── chatgpt4o_mini/
│   ├── easyocr_piplines/
│   ├── lightonocr_pipelines/
│   ├── qwen3VL-8B-4Bit/
│   └── surya_pipelines/
├── font_test/
│   ├── check_char_support.py
│   ├── fonts_with_missing_chars.txt
│   └── test_fonts.py
├── notebooks/
│   ├── easyocr_test.ipynb
│   ├── lightonocr_test.ipynb
│   ├── paddleOCR_test.ipynb
│   ├── PP_OCRV5.ipynb
│   ├── qwen3VL_8B_4bit.ipynb
│   ├── qwen3VL_8B_4bit_2steps.ipynb
│   ├── qwen3VL_8B_4bit_2steps_backup.ipynb
│   └── Surya OCR.ipynb
├── results/
│   ├── metrics/
│   │   ├── README.md
│   │   ├── easyocr/
│   │   ├── lightonocr/
│   │   ├── paddleocr/
│   │   └── surya/
│   └── visualizations/
├── scripts/
│   ├── add_reference_summaries.py
│   ├── create_test_set.py
│   ├── evaluate_llm.py
│   ├── evaluate_with_deepeval.py
│   └── run_llm_pipeline.py
├── sounds/
├── chatbot/
│   ├── camera.py
│   ├── gui.py
│   ├── opencv_pygame_viewer.py
│   ├── pixtral_8b_vision_task2.py
│   ├── string_test.py
│   ├── agent_inspector_output/
│   ├── images/
│   ├── saved_frames/
│   └── sounds/
├── src/
│   ├── agents/
│   │   ├── create_voice.py
│   │   └── vision_routing_agent.py
│   ├── data_generation/
│   │   ├── config.py
│   │   ├── utils.py
│   │   └── generate_synthetic_notes.py
│   ├── data_visualization/
│   │   └── visualize_synthetic_data.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── llm_metrics.py
│   │   └── deepeval_metrics.py
│   ├── llm_processing/
│   │   ├── __init__.py
│   │   ├── llm_pipeline.py
│   │   ├── models.py
│   │   └── prompts.py
│   └── utils/
│       ├── fix_images.py
│       └── image_augmentation.py
└── Thesis/
    ├── Abschlussarbeit.tex
    ├── chapters/
    ├── images/
    ├── otherincludes/
    └── build/
```

## Hauptbausteine

### OCR- und Evaluierungs-Pipeline

Die OCR-Ergebnisse liegen je nach Verfahren unter `results/metrics/<ocr-method>/` und enthalten typischerweise:

- `summary.csv`
- `summary.json`
- `detailed_results.json`
- `individual/`

Das Evaluierungsmodul in [src/evaluation/metrics.py](src/evaluation/metrics.py) berechnet unter anderem CER, WER, NED, Accuracy, Precision, Recall und F1. Die LLM-spezifischen Metriken in [src/evaluation/llm_metrics.py](src/evaluation/llm_metrics.py) decken ROUGE, BLEU, BERTScore, medizinische Term-Erhaltung, JSON-Validierung und Feldgenauigkeit ab.

#### Metriken zur Evaluation

**Task 1: Transcription Performance**

- Vergleich der Transkriptionshypothese (H) mit der Referenz (R) nach Normalisierung (z. B. Umwandlung in Kleinbuchstaben, Entfernung von Zeilenumbrüchen).
- Verwendete Metriken:
- Character Error Rate (CER) und Word Error Rate (WER)
- Normalized Edit Distance (NED)
- Vocabulary Overlap (Unique-Word Precision / Recall / F1)

**Task 2: Downstream Understanding**

- Bewertung der Bewahrung klinisch relevanter Informationen bei Textausgaben (Zusammenfassungen) und strukturierter Feldextraktion (JSON).
- Verwendete Metriken:
- ROUGE-Scores (ROUGE-1, ROUGE-2, ROUGE-L) für Zusammenfassungen
- Medical Term Preservation (Keyword-Set Precision / Recall / F1)
- BERTScore (semantische Ähnlichkeit)

### Standalone LLM Pipeline

Die LLM-Nachverarbeitung ist als wiederverwendbares Modul in [src/llm_processing](src/llm_processing) implementiert. Der Einstieg erfolgt meist über [scripts/run_llm_pipeline.py](scripts/run_llm_pipeline.py), die Auswertung über [scripts/evaluate_llm.py](scripts/evaluate_llm.py).

Verfügbare Prompt-Typen werden über `--list-prompts` ausgegeben. Aktuell sind unter anderem diese Vorlagen vorhanden:

- `nursing_summary`
- `critical_info_extraction`
- `care_plan_generation`
- `risk_assessment`
- `simple_summary`
- `VLM_prompt`
- `VLM_prompt_v2`

### Vision Routing Agent

Der Bild-/Frame-Auswahlagent in [src/agents/vision_routing_agent.py](src/agents/vision_routing_agent.py) bewertet Bildqualität, wählt das beste Frame für die OCR aus und speichert Ergebnisse und Plots unter `data/processed/vision_routing_agent/`.

### Synthetische Datengenerierung

Die Generierung synthetischer Pflegenotizen liegt in [src/data_generation/generate_synthetic_notes.py](src/data_generation/generate_synthetic_notes.py). Die Konfiguration befindet sich in [src/data_generation/config.py](src/data_generation/config.py), Hilfsfunktionen in [src/data_generation/utils.py](src/data_generation/utils.py).

Standardausgaben sind:

- Bilder: `data/synthetic/output/images/`
- Labels: `data/synthetic/output/labels/`

### Thesis-Quelle

Die Bachelorarbeit selbst liegt unter `Thesis/` und wird aus `Thesis/Abschlussarbeit.tex` gebaut. Die Kapitel liegen in `Thesis/chapters/`, Zusatzdateien in `Thesis/otherincludes/`.

## Installation

Voraussetzungen:

- Python 3.10 oder neuer
- Optional: NVIDIA-GPU fuer schnellere Modellinferenz

Empfohlene Einrichtung:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Falls du die README aus einer frischen Umgebung nutzt, pruefe bei OCR- und LLM-Workflows zusaetzlich die Hinweise in der Dokumentation unter [docs/](docs).

## Typische Arbeitsablaeufe

### 1. LLM-Pipeline auf OCR-Ergebnissen ausfuehren

```bash
python scripts/run_llm_pipeline.py \
    --ocr-method easyocr \
    --model qwen-7b
```

Nur die Modelle anzeigen:

```bash
python scripts/run_llm_pipeline.py --list-models
```

Nur die Prompt-Vorlagen anzeigen:

```bash
python scripts/run_llm_pipeline.py --list-prompts
```

### 2. LLM-Ergebnisse evaluieren

```bash
python scripts/evaluate_llm.py \
    --input results/metrics/easyocr/llm_outputs/<experiment>/all_llm_results.json
```

### 3. Synthetische Notizen erzeugen

```bash
python -m src.data_generation.generate_synthetic_notes --num-samples 10 --workers 1 --seed 42
python -m src.data_generation.generate_synthetic_notes --num-samples 500 --workers 8 --seed 42
```

### 4. Vision Routing Agent starten

```bash
python src/agents/vision_routing_agent.py --frames-dir data/synthetic/output/robot_frames
```

### 5. Notebooks verwenden

Die OCR- und VLM-Experimente sind in [notebooks/](notebooks) abgelegt. Typische Startpunkte sind:

- `notebooks/easyocr_test.ipynb`
- `notebooks/paddleOCR_test.ipynb`
- `notebooks/Surya OCR.ipynb`
- `notebooks/vision_agent.ipynb`

## Ergebnisstruktur

Die wichtigsten erzeugten Artefakte sind:

- OCR-Ergebnisse: `results/metrics/<ocr-method>/`
- LLM-Outputs: `results/metrics/<ocr-method>/llm_outputs/<model>_<prompt>/`
- OCR-Analyseplots: `results/visualizations/`
- Vision-Routing-Outputs: `data/processed/vision_routing_agent/`
- Synthetische Notizen: `data/synthetic/output/`

## Nützliche Dokumentation

- [docs/STANDALONE_LLM_PIPELINE.md](docs/STANDALONE_LLM_PIPELINE.md)
- [docs/OCR_LLM_PIPELINE_GUIDE.md](docs/OCR_LLM_PIPELINE_GUIDE.md)
- [docs/EVALUATION_METRICS_GUIDE.md](docs/EVALUATION_METRICS_GUIDE.md)
- [docs/TEST_SET_SPECIFICATION.md](docs/TEST_SET_SPECIFICATION.md)
- [results/metrics/README.md](results/metrics/README.md)

## Hinweise

- Der Ordner `configs/` ist derzeit leer und dient als Platz fuer spaetere Konfigurationen.
- `results/`, `data/processed/` und Teile von `chatbot/` enthalten generierte Artefakte und werden je nach Workflow neu erzeugt.
- Modellnamen, Prompt-Listen und Output-Pfade werden direkt von `scripts/run_llm_pipeline.py` und `src/llm_processing/models.py` bestimmt.

## Kurz gesagt

Das Repository ist kein reines OCR-Projekt, sondern eine kombinierte Arbeitsumgebung fuer:

- OCR-Evaluierung
- LLM-Nachbearbeitung
- Vision-basierte Bildauswahl
- synthetische Datenerzeugung
- Thesis-Schreibprozess in LaTeX
