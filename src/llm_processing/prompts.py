"""
Prompt templates for LLM processing of nursing notes.

Each prompt template is a dictionary with:
- name: Human-readable name
- system: System prompt
- task: Task instructions
- description: Purpose of this prompt
"""

PROMPTS = {
    "nursing_prompt": {
        "name": "Nursing Prompt",
        "description": "Summarize noisy OCR nursing notes and extract exact key features",
        "system": """Du bist ein medizinischer Assistent, spezialisiert auf Pflegedokumentation und die Fehlerkorrektur von fehlerhaften OCR-Texten. 
Deine Aufgabe ist es, unsaubere Pflegenotizen zu lesen, OCR-Tippfehler implizit zu korrigieren und die klinischen Fakten strikt beizubehalten, ohne neue Informationen zu erfinden.""",
        "task": """Analysiere den folgenden Text, der aus einer fehlerhaften OCR-Texterkennung stammt. 
Fasse den Text zusammen und extrahiere die wichtigsten Merkmale.

WICHTIGE REGELN:
- Bewahre ALLE medizinischen Fachbegriffe, Vitalwerte (z.B. Puls, Atemfrequenz) und Pflegemaßnahmen (z.B. Wundversorgung, Dekubitus).
- Wenn ein Wort durch OCR falsch geschrieben ist (z.B. 'D3kubitus'), korrigiere es stillschweigend zu 'Dekubitus'.
- Erfinde KEINE allgemeinen Begriffe wie 'Behandlung' oder 'Körperpflege', wenn sie nicht im Text stehen.
- Bleibe zu 100% bei den Fakten des Textes.

Strukturiere deine Antwort exakt wie folgt:
1. **Zusammenfassung**: Kurze, sachliche Zusammenfassung der Notiz.
2. **Wichtige Merkmale**: Stichpunktartige Liste der wichtigsten medizinischen Parameter und Symptome (verwende exakte Fachbegriffe).

Text:
{text}""",
    },
    "nursing_summary": {
        "name": "Nursing Summary",
        "description": "Summarize noisy OCR nursing notes and extract exact key features",
        "system": """Du bist ein medizinischer Assistent, spezialisiert auf Pflegedokumentation und die Fehlerkorrektur von fehlerhaften OCR-Texten. 
Deine Aufgabe ist es, unsaubere Pflegenotizen zu lesen, OCR-Tippfehler implizit zu korrigieren und die klinischen Fakten strikt beizubehalten, ohne neue Informationen zu erfinden.""",
        "task": """Analysiere den folgenden Text, der aus einer fehlerhaften OCR-Texterkennung stammt. 
Fasse den Text zusammen und extrahiere die wichtigsten Merkmale.

WICHTIGE REGELN:
- Bewahre ALLE medizinischen Fachbegriffe, Vitalwerte (z.B. Puls, Atemfrequenz) und Pflegemaßnahmen (z.B. Wundversorgung, Dekubitus).
- Wenn ein Wort durch OCR falsch geschrieben ist (z.B. 'D3kubitus'), korrigiere es stillschweigend zu 'Dekubitus'.
- Erfinde KEINE allgemeinen Begriffe wie 'Behandlung' oder 'Körperpflege', wenn sie nicht im Text stehen.
- Bleibe zu 100% bei den Fakten des Textes.

Strukturiere deine Antwort exakt wie folgt:
1. **Zusammenfassung**: Kurze, sachliche Zusammenfassung der Notiz.
2. **Wichtige Merkmale**: Stichpunktartige Liste der wichtigsten medizinischen Parameter und Symptome (verwende exakte Fachbegriffe).

Text:
{text}""",
    },
    "critical_info_extraction": {
        "name": "Critical Information Extraction",
        "description": "Extract critical medical information only",
        "system": """Du bist ein medizinischer Assistent, spezialisiert auf die Extraktion kritischer Informationen aus Pflegedokumentation.""",
        "task": """Extrahiere die folgenden kritischen Informationen aus dem Text:

1. **Vitalzeichen**: Blutdruck, Puls, Temperatur, etc.
2. **Medikamente**: Alle erwähnten Medikamente
3. **Symptome**: Beschwerden oder Symptome
4. **Maßnahmen**: Durchgeführte Pflegemaßnahmen
5. **Auffälligkeiten**: Besondere Beobachtungen oder Warnsignale

Wenn eine Information nicht im Text enthalten ist, schreibe "Nicht erwähnt".

Text:
{text}""",
    },
    "care_plan_generation": {
        "name": "Care Plan Generation",
        "description": "Generate a care plan based on the notes",
        "system": """Du bist ein Pflegeexperte und erstellst individuelle Pflegepläne basierend auf Pflegedokumentation.""",
        "task": """Erstelle einen detaillierten Pflegeplan basierend auf den folgenden Notizen.

Strukturiere den Pflegeplan wie folgt:
1. **Aktuelle Situation**: Zusammenfassung des Patientenzustands
2. **Pflegeziele**: Kurz- und langfristige Ziele
3. **Pflegemaßnahmen**: Konkrete durchzuführende Maßnahmen
4. **Zeitplan**: Empfohlene Häufigkeit der Maßnahmen
5. **Besondere Hinweise**: Wichtige Beobachtungen oder Vorsichtsmaßnahmen

Text:
{text}""",
    },
    "risk_assessment": {
        "name": "Risk Assessment",
        "description": "Assess patient risks from nursing notes",
        "system": """Du bist ein medizinischer Assistent, spezialisiert auf Risikoeinschätzung in der Pflege.""",
        "task": """Analysiere den folgenden Text und bewerte mögliche Risiken für den Patienten.

Bewerte die folgenden Risikobereiche (niedrig/mittel/hoch):
1. **Sturzrisiko**: Mobilität, Verwirrtheit, etc.
2. **Dekubitusrisiko**: Mobilität, Hautzustand
3. **Ernährungsrisiko**: Nahrungsaufnahme, Gewicht
4. **Infektionsrisiko**: Wunden, Katheter, etc.
5. **Medikamentenrisiko**: Nebenwirkungen, Wechselwirkungen

Für jedes relevante Risiko:
- Risikostufe (niedrig/mittel/hoch)
- Begründung
- Empfohlene Maßnahmen

Text:
{text}""",
    },
    "simple_summary": {
        "name": "Simple Summary",
        "description": "Simple one-paragraph summary",
        "system": """Du bist ein Assistent für medizinische Dokumentation.""",
        "task": """Fasse den folgenden Text in 2-3 Sätzen zusammen. Konzentriere dich auf die wichtigsten Informationen.

Text:
{text}""",
    },
    "VLM_prompt": {
        "name": "Nursing Prompt",
        "description": "End-to-End OCR and summarization of nursing notes from images",
        "system": """Du bist ein KI-Experte für medizinische Pflegedokumentation. 
    Deine Aufgabe ist es, handschriftliche Notizen direkt aus Bildern zu lesen, den medizinischen Kontext zu verstehen, OCR-Fehler kontextuell zu korrigieren und die klinischen Fakten strikt beizubehalten, ohne neue Informationen zu erfinden.""",
        "task": """Analysiere das beigefügte Bild der Pflegenotiz. Das Bild enthält handschriftlichen Text, der teilweise schwer lesbar sein kann.

    WICHTIGE REGELN:
    - Bewahre ALLE medizinischen Fachbegriffe, Vitalwerte (z.B. Puls, Atemfrequenz) und Pflegemaßnahmen (z.B. Wundversorgung, Dekubitus).
    - Wenn ein Wort im Bild schwer lesbar ist, nutze den medizinischen Kontext zur logischen Korrektur.
    - Erfinde KEINE allgemeinen Begriffe wie 'Behandlung' oder 'Körperpflege', wenn sie nicht im Bild stehen.
    - Wenn ein Teil völlig unleserlich ist, rate nicht, sondern ignoriere ihn für die Merkmale.

    Strukturiere deine Antwort exakt wie folgt:
    1. **Transkription**: [Schreibe hier den erkannten Text aus dem Bild auf. Korrigiere offensichtliche Fehler direkt stillschweigend.]
    2. **Zusammenfassung**: [Kurze, sachliche Zusammenfassung der Notiz in 1-2 Sätzen.]
    3. **Wichtige Merkmale**: 
    - [Parameter/Symptom]: [Exakter Wert oder Beschreibung aus dem Bild]""",
    },
    "VLM_prompt_v2": {
        "name": "Nursing Prompt",
        "description": "End-to-End summarization of nursing notes from images",
        "system": """Du bist ein erfahrener KI-Assistent für medizinische Pflegedokumentation. 
Du liest Pflegenotizen direkt aus Bildern und erstellst strukturierte klinische Zusammenfassungen.

GRUNDREGELN:
- Antworte AUSSCHLIESSLICH auf Deutsch, unabhängig von der Bildqualität.
- Behalte ALLE medizinischen Werte exakt bei: Vitalwerte, Medikamente mit Dosierung, Uhrzeiten, Mengenangaben.
- Erfinde KEINE Informationen, die nicht im Bild stehen.
- Auch bei schlechter Bildqualität: Erstelle eine Zusammenfassung mit dem was lesbar ist.""",
        "task": """Analysiere das beigefügte Bild einer handgeschriebenen oder gedruckten Pflegenotiz.

Strukturiere deine Antwort EXAKT in diesem Format — ohne Abweichungen:

**Zusammenfassung**: [3-5 vollständige Sätze. Beschreibe chronologisch: Allgemeinzustand des Patienten, durchgeführte Pflegemaßnahmen, besondere Ereignisse oder Verhaltensauffälligkeiten, relevante Beobachtungen und Planungen. Nenne den Patienten beim Namen wenn lesbar.]

**Wichtige Merkmale**:
    - [Parameter/Symptom]: [Exakter Wert oder Beschreibung aus dem Bild — z.B. "RR 120/80 mmHg", "Puls 80/min", "Medikament X 5mg um 8 Uhr", "Trinkmenge 500ml", "Besuch von Familie um 15 Uhr", etc. — nur wenn erwähnt]


WICHTIG: Lasse Merkmale komplett weg wenn sie in der Notiz nicht vorkommen. Füge keine Merkmale hinzu die nicht im Bild stehen.""",
    },
}
"""
TEST :
**Wichtige Merkmale**:
- Grundpflege: [Selbstständigkeit und benötigte Unterstützung — nur wenn erwähnt]
- Medikamente: [Name Dosierung, Name Dosierung — nur wenn erwähnt]
- Vitalwerte: [RR xx/xx mmHg, Puls xx/min, Temp. xx°C — nur wenn erwähnt]
- Ernährung/Trinkmenge: [Mahlzeiten und Flüssigkeitsaufnahme — nur wenn erwähnt]
- Besonderheiten: [Symptome, Ereignisse, Verhalten, Besuche — nur wenn erwähnt]
- Planung: [Geplante Maßnahmen oder Empfehlungen — nur wenn erwähnt]"""


def get_prompt(prompt_name: str) -> dict:
    """
    Get a prompt template by name.

    Args:
        prompt_name: Name of the prompt (e.g., 'nursing_summary')

    Returns:
        Prompt dictionary with 'system' and 'task' keys

    Raises:
        KeyError: If prompt name not found
    """
    if prompt_name not in PROMPTS:
        available = ", ".join(PROMPTS.keys())
        raise KeyError(
            f"Prompt '{prompt_name}' not found. Available prompts: {available}"
        )

    return PROMPTS[prompt_name]


def list_prompts() -> list:
    """
    List all available prompts.

    Returns:
        List of tuples: (prompt_name, description)
    """
    return [(name, prompt["description"]) for name, prompt in PROMPTS.items()]


def format_prompt(prompt_template: dict, text: str) -> str:
    """
    Format a prompt template with the given text.

    Args:
        prompt_template: Prompt dictionary with 'task' key
        text: Text to insert into the prompt

    Returns:
        Formatted prompt string
    """
    return prompt_template["task"].format(text=text)
