"""
Add handcrafted reference_summary to every scenario in scenarios.json.
Run once:  python scripts/add_reference_summaries.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_FILE = ROOT / "data" / "synthetic" / "scenarios.json"

# Format mirrors the nursing_summary prompt output:
#   1. **Zusammenfassung**: ...
#   2. **Wichtige Merkmale**: bullet list
SUMMARIES = {
    "Gerhard Müller": (
        "**Zusammenfassung**: Hr. Müller (82 J., PG 3) wurde morgens mit Unterstützung bei der Grundpflege versorgt. "
        "Gegen 9:30 Uhr zeigte er demenzbedingte Unruhe; nach Validation beruhigte er sich und setzte sich in den Aufenthaltsraum. "
        "Die Trinkmenge war mit ca. 200 ml bis Mittag unzureichend. Der Besuch der Tochter zeigte positiven Effekt. "
        "Abendpflege verlief problemlos, Inkontinenzmaterial gewechselt, Haut intakt.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Grundpflege: Oberkörper teilselbstständig, Unterkörper volle Hilfe\n"
        "- Medikamente: Ramipril 5 mg, ASS 100 mg, Metformin 850 mg eingenommen\n"
        "- Vitalwerte 11:00 Uhr: RR 150/95 mmHg, Puls 78/min rhythmisch, Temp. 36,7 °C\n"
        "- Trinkmenge bis Mittag: ca. 200 ml (zu gering, engmaschig überwachen)\n"
        "- Demenzbedingte Unruhe: Validation erfolgreich\n"
        "- Planung: begleiteter Spaziergang, Rücksprache Angehörige wegen Besuchszeiten"
    ),
    "Helga Schmidt": (
        "**Zusammenfassung**: Fr. Schmidt (76 J., PG 2) klagte vormittags über Kopfschmerzen (Schmerzskala 4/10). "
        "Nach Rücksprache wurde Novalgin 500 mg verabreicht. Nach ca. 45 Minuten deutliche Besserung auf 1/10. "
        "Kurzer Spaziergang im Garten; nachmittags gute Stimmung, keine weiteren Beschwerden.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Symptom: Kopfschmerzen 4/10 → nach Medikation 1/10\n"
        "- Medikament: Novalgin 500 mg\n"
        "- Allgemeinzustand: gut, keine weiteren Beschwerden"
    ),
    "Klaus Richter": (
        "**Zusammenfassung**: Hr. Richter (79 J., PG 2) startete sehr gut in den Tag – beim Frühstück gesprächig und aufgeheitert. "
        "Medikamente zuverlässig eingenommen. Nachmittags konzentrierte Schachpartie mit Hr. Fischer. Keine Besonderheiten.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Medikamente eingenommen\n"
        "- Allgemeinzustand: gut, keine Auffälligkeiten"
    ),
    "Ingrid Braun": (
        "**Zusammenfassung**: Fr. Braun (84 J., PG 4) wurde morgens müde und fiebrig im Sessel vorgefunden. "
        "Vitalkontrolle zeigte Temp. 37,8 °C, die bis 15:00 Uhr auf 38,2 °C anstieg. "
        "Dr. Vogel informiert; Verdacht auf Harnwegsinfekt – Urinprobe ans Labor, Cotrimoxazol 960 mg 2x tgl. verordnet. "
        "Erste Gabe 19:00 Uhr problemlos eingenommen. Grundpflege im Bett, Hautschutz im Intimbereich aufgetragen.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Vitalwerte 8:00 Uhr: Temp. 37,8 °C → 15:00 Uhr 38,2 °C, RR 130/85 mmHg, Puls 92/min, AF 18/min\n"
        "- Diagnose: V. a. Harnwegsinfekt\n"
        "- Antibiotikum: Cotrimoxazol 960 mg 2x tgl.\n"
        "- Maßnahmen: Urinprobe Labor, Flüssigkeitsbilanz, Temp. alle 4 Std., Ausscheidung beobachten\n"
        "- Hautschutz Intimbereich aufgetragen"
    ),
    "Monika Keller": (
        "**Zusammenfassung**: Fr. Keller (88 J., PG 3) stürzte 10:45 Uhr beim Aufstehen vom Sessel. "
        "Ansprechbar und orientiert, Schmerzen im linken Handgelenk (6/10), geschwollen. "
        "Dr. Müller informiert – Verdacht auf Fraktur. Röntgenbefund ergab Prellung (keine Fraktur). "
        "Rückkehr 15:00 Uhr, Ibuprofen 400 mg, Sturzdokumentation erstellt, Tochter informiert.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Sturzereignis 10:45 Uhr\n"
        "- Vitalcheck: RR 145/90 mmHg, Puls 88/min, keine Bewusstlosigkeit\n"
        "- Schmerzen li. Handgelenk: 6/10 → nach Transport 4/10\n"
        "- Befund: Prellung (keine Fraktur)\n"
        "- Medikament: Ibuprofen 400 mg\n"
        "- Maßnahmen: Greifhilfe, Bücher auf niedrigeres Regal, Sturzrisiko neu bewerten"
    ),
    "Werner Schneider": (
        "**Zusammenfassung**: Hr. Schneider (73 J., PG 1) war vollständig selbstständig. "
        "Alle Mahlzeiten im Speisesaal, sehr kommunikativ und gut gelaunt. Keine Besonderheiten.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Selbstständigkeit: vollständig\n"
        "- Allgemeinzustand: sehr gut, keine Auffälligkeiten"
    ),
    "Elisabeth Weber": (
        "**Zusammenfassung**: Fr. Weber (91 J., PG 5) war heute sehr schwach, überwiegend bettlägerig. "
        "Vollständige Grundpflege im Bett mit Lagerung. Nahrungsaufnahme sehr gering (wenige Löffel); Flüssigkeitsbilanz bis 20:00 Uhr ca. 400 ml. "
        "Dekubitus Grad 2 an der rechten Ferse leicht verschlechtert – Versorgung mit Hydrokolloidverband. "
        "Lagerung alle 2–3 Stunden, Schmerz 3/10 (BESD-Skala), Dr. Klein informiert.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Dekubitus re. Ferse Grad 2, ca. 1,5 × 2 cm, leicht verschlechtert; Hydrokolloidverband\n"
        "- Vitalwerte 18:00 Uhr: RR 110/70 mmHg, Puls 82/min, SpO2 92 % (Raumluft), Temp. 36,8 °C, AF 16/min\n"
        "- Schmerzerfassung: 3/10 (BESD-Skala)\n"
        "- Flüssigkeitsbilanz: ca. 400 ml (unzureichend)\n"
        "- Nahrungsaufnahme: sehr gering (löffelweise)\n"
        "- Lagerung alle 2–3 Std., Druckentlastung Ferse"
    ),
    "Hans-Peter Becker": (
        "**Zusammenfassung**: Hr. Becker (77 J., PG 2) feierte seinen 77. Geburtstag. "
        "Familie (Ehefrau und 3 Kinder) zu Besuch 14:00–18:30 Uhr, sehr herzliche Stimmung. "
        "Guter Appetit, keine medizinischen Besonderheiten.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Allgemeinzustand: gut\n"
        "- Besuch: Familie 4,5 Stunden, positiver emotionaler Effekt\n"
        "- Nahrungsaufnahme: gut (2 Stücke Torte)"
    ),
    "Margarete Fischer": (
        "**Zusammenfassung**: Fr. Fischer (85 J., PG 3) war morgens zeitlich und örtlich desorientiert, fragte mehrfach nach ihrem verstorbenen Mann Georg. "
        "Validation und ruhige Ansprache halfen; nach ca. 30 Minuten und dem Frühstück zunehmend klarer. "
        "Mittags vollständig orientiert, nachmittags aktive Teilnahme am Gedächtnistraining.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Desorientierung morgens: zeitlich und örtlich\n"
        "- Maßnahme: Validation – erfolgreich\n"
        "- Nachmittags: vollständig orientiert\n"
        "- Gedächtnistraining: engagierte Teilnahme"
    ),
    "Otto Zimmermann": (
        "**Zusammenfassung**: Hr. Zimmermann (89 J., PG 4) litt seit gestern Abend an Durchfall (5× Nachtdienst + 2× bis 10:00 Uhr). "
        "Erschöpft, blass, Haut trocken. Stuhlprobe ans Labor, Flüssigkeitsbilanz gestartet (800 ml bis 14:00 Uhr). "
        "Loperamid 2 mg um 14:00 Uhr. Frequenz sank bis 20:00 Uhr auf einmaligen Stuhlgang.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Durchfall: 7× seit gestern Abend (wässrig, braun)\n"
        "- Vitalwerte 10:00 Uhr: RR 125/75 mmHg, Puls 94/min, Temp. 37,2 °C\n"
        "- Zeichen möglicher Dehydrierung: trockene Haut, verminderte Hautspannung\n"
        "- Medikament: Loperamid 2 mg\n"
        "- Maßnahmen: Stuhlprobe Labor, Flüssigkeitsbilanz, Elektrolytlösung, leichte Schonkost"
    ),
    "Hildegard Krause": (
        "**Zusammenfassung**: Fr. Krause (81 J., PG 2) hatte Physiotherapie 10:00–10:45 Uhr mit Fr. Berger. "
        "Toller Fortschritt: kann jetzt 20 Meter mit dem Rollator ohne Pause gehen. Sehr motiviert.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Physiotherapie: 45 Minuten\n"
        "- Mobilität: 20 Meter Rollator ohne Pause (Fortschritt)\n"
        "- Motivation: hoch"
    ),
    "Friedrich Hoffmann": (
        "**Zusammenfassung**: Hr. Hoffmann (75 J., PG 3) verweigerte morgens 8:00 Uhr die Grundpflege ohne klare Begründung. "
        "Mehrere motivierende Versuche ohne Erfolg. Nachmittags 15:00 Uhr in ruhiger Atmosphäre Teilwäsche erfolgreich; "
        "Hr. Hoffmann entschuldigte sich und war kooperativ.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Pflegeverweigerung morgens; nach ruhigem zweitem Versuch nachmittags erfolgreich\n"
        "- Teilwäsche und frische Kleidung durchgeführt\n"
        "- Planung: geduldige Anbahnung der Morgenpflege"
    ),
    "Anna-Maria Koch": (
        "**Zusammenfassung**: Fr. Koch (94 J., PG 5) ist vollständig bettlägerig, reagiert kaum auf Ansprache. "
        "Komplette Grundpflege im Bett, intensive Mundpflege, Lagerung alle ~3,5 Stunden. "
        "PEG-Sondenernährung 1500 ml/18 h (Einstichstelle reizlos). Rasselgeräusche über beiden Lungenflügeln → "
        "Inhalation mit 0,9 % NaCl durchgeführt. Allgemeinzustand stabil.\n\n"
        "**Wichtige Merkmale**:\n"
        "- PEG-Sondenernährung: 1500 ml/18 h kontinuierlich via Pumpe; Restvolumen 20 ml (i. O.)\n"
        "- Atemwege: leichte Rasselgeräusche, Inhalation NaCl 0,9 %, SpO2 94 %, AF 14/min\n"
        "- Vitalwerte 14:00 Uhr: RR 105/65 mmHg, Puls 76/min, Temp. 36,5 °C, SpO2 94 %\n"
        "- Lagerung: alle 3,5 Std. (4×)\n"
        "- Mundpflege: intensiv"
    ),
    "Günther Lange": (
        "**Zusammenfassung**: Hr. Lange (78 J., PG 2) hatte einen aktiven Tag. "
        "Vormittags Spaziergang im Garten (ca. 30 Min.), gut vertragen. Nachmittags Zeitung gelesen. "
        "Allgemeinzustand gut, keine Beschwerden.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Mobilität: Spaziergang 30 Min., gut vertragen\n"
        "- Allgemeinzustand: gut, keine Auffälligkeiten"
    ),
    "Christa Vogel": (
        "**Zusammenfassung**: Fr. Vogel (83 J., PG 3) klagte seit dem Morgen über Rückenschmerzen (6/10). "
        "Lagerung optimiert, Ibuprofen 400 mg um 10:00 Uhr gegeben. "
        "Schmerzkontrolle 11:00 Uhr: deutliche Besserung auf 3/10. Physiotherapie für morgen organisiert.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Schmerzen: Rücken 6/10 → nach Ibuprofen 3/10\n"
        "- Medikament: Ibuprofen 400 mg\n"
        "- Maßnahme: Lagerungsoptimierung (Lendenwirbelkissen)\n"
        "- Planung: Physiotherapie morgen 10:00 Uhr"
    ),
    "Walter Jung": (
        "**Zusammenfassung**: Hr. Jung (86 J., PG 4) hatte eine sehr unruhige Nacht mit multiplen Desorientierungsepisoden und Sturzgefahr; "
        "Bettgitter nach Rücksprache mit Bereitschaftsarzt angebracht. Tagsüber erschöpft, Grundpflege 9:30 Uhr. "
        "Dr. Hartmann informiert, Visite morgen zur Überprüfung der Medikation (Melperon).\n\n"
        "**Wichtige Merkmale**:\n"
        "- Nachtunruhe: 4 Episoden (22:30 – 03:00 Uhr), Desorientierung, Sturzgefahr\n"
        "- Bettgitter angebracht (Bereitschaftsarzt)\n"
        "- Vitalwerte 10:00 Uhr: RR 140/85 mmHg, Puls 88/min\n"
        "- Maßnahmen: Ruhe, Sturzrisiko beobachten, Nachtprotokoll fortführen\n"
        "- Medikationsüberprüfung (Melperon) geplant"
    ),
    "Ursula Peters": (
        "**Zusammenfassung**: Fr. Peters (72 J., PG 1) war vollständig selbstständig. "
        "Teilnahme an Seniorensport 10:00 Uhr und Singkreis 14:30 Uhr. Sehr aktiv und gut gelaunt.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Selbstständigkeit: vollständig\n"
        "- Aktivitäten: Seniorensport, Singkreis\n"
        "- Allgemeinzustand: sehr gut"
    ),
    "Heinrich Schröder": (
        "**Zusammenfassung**: Hr. Schröder (87 J., PG 3) hatte einen Augenarzttermin bei Dr. Lehmann. "
        "Befund: fortgeschrittener grauer Star beidseitig, stark eingeschränktes Sehvermögen. "
        "Kataraktoperation geplant für 07.11.2025 in der Uniklinik. Hr. Schröder wirkt nervös. "
        "Sohn informiert, wird ihn begleiten. Präoperative Checkliste wird vorbereitet.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Diagnose: fortgeschrittener grauer Star (Katarakt) beidseitig\n"
        "- OP-Termin: 07.11.2025, 8:00 Uhr Uniklinik\n"
        "- Einverständniserklärung vorliegend\n"
        "- Emotionaler Zustand: nervös wegen der OP\n"
        "- Sohn als Begleitung"
    ),
    "Elfriede Herrmann": (
        "**Zusammenfassung**: Fr. Herrmann (90 J., PG 4) stürzte 9:15 Uhr beim Aufstehen auf die linke Seite. "
        "Massive Hüftschmerzen links (8/10), linkes Bein nicht beweglich – V. a. Schenkelhalsfraktur. "
        "Notarzt 9:40 Uhr; Morphin 5 mg i.v. Einweisung Unfallklinik St. Marien um 10:15 Uhr. "
        "Sohn informiert. Sturzdokumentation erstellt.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Sturzereignis 9:15 Uhr, li. Hüfte\n"
        "- Vitalwerte 9:20 Uhr: RR 160/95 mmHg, Puls 102/min\n"
        "- Schmerzen: li. Hüfte 8/10, Bein nicht beweglich\n"
        "- V. a. Schenkelhalsfraktur links\n"
        "- Morphin 5 mg i.v. (Notarzt)\n"
        "- Krankenhauseinweisung: Unfallklinik St. Marien"
    ),
    "Karl-Heinz Meyer": (
        "**Zusammenfassung**: Hr. Meyer (74 J., PG 2) war heute sehr gesprächig und gut aufgelegt. "
        "Beim Morgenrundgang erzählte er ausführlich von seiner Zeit als Briefträger (40 Dienstjahre). "
        "Biografiegespräche wirken aktivierend. Keine medizinischen Besonderheiten.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Allgemeinzustand: gut\n"
        "- Biografiegespräch: aktivierend, positive Wirkung"
    ),
    "Irmgard Schulze": (
        "**Zusammenfassung**: Fr. Schulze (88 J., PG 3) verweigerte seit 3 Tagen die Tabletten. "
        "Heute erneute Verweigerung; Lösung: Tabletten mit Apfelmus gereicht – erfolgreich eingenommen. "
        "Dr. Klein informiert, Apfelmus als Standardmedikationshilfe angeordnet.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Tabletteneinnahme: 3-tägige Verweigerung → Lösung mit Apfelmus erfolgreich\n"
        "- Maßnahme (dauerhaft): Medikamente mit Apfelmus anreichen"
    ),
    "Erich Bergmann": (
        "**Zusammenfassung**: Hr. Bergmann (80 J., PG 2) war heute zur Dialyse im Dialysezentrum (7:30–13:00 Uhr). "
        "Nach Rückkehr sehr müde, schlief bis 16:00 Uhr. Leichtes Abendessen. "
        "Vitalkontrolle 19:00 Uhr unauffällig. Shunt-Arm ohne Befund.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Dialyse: regulärer Termin 7:30–13:00 Uhr\n"
        "- Vitalwerte 19:00 Uhr: RR 125/80 mmHg, Puls 76/min\n"
        "- Shunt-Arm: unauffällig, kein Hämatom, gutes Strömungsgeräusch\n"
        "- Nahrungsaufnahme nach Dialyse: reduziert"
    ),
    "Gisela Frank": (
        "**Zusammenfassung**: Fr. Frank (85 J., PG 3) hatte Besuch von ihrem Enkel Tim (12 Jahre) 14:00–16:00 Uhr. "
        "Gemeinsam Kaffee und Kuchen. Sehr gute Stimmung, emotional positiver Tag. Keine medizinischen Besonderheiten.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Allgemeinzustand: gut\n"
        "- Besuch Enkel: positiver emotionaler Effekt\n"
        "- Keine medizinischen Auffälligkeiten"
    ),
    "Rudolf Wagner": (
        "**Zusammenfassung**: Hr. Wagner (76 J., PG 2) klagte seit gestern Abend über Bauchschmerzen (5/10). "
        "Letzter Stuhlgang vor 3 Tagen – Verdacht auf Obstipation. "
        "Laxoberal 15 Tropfen um 10:00 Uhr, viel Flüssigkeit (Wasser, Pflaumensaft). "
        "Um 16:00 Uhr Stuhlgang in normaler Konsistenz, Schmerzen danach 2/10.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Symptom: Obstipation (3 Tage kein Stuhlgang), Bauchschmerzen 5/10 → 2/10\n"
        "- Medikament: Laxoberal 15 Tropfen\n"
        "- Stuhlgang: 16:00 Uhr, normale Konsistenz\n"
        "- Empfehlung: ballaststoffreiche Kost, Trinkplan, Bewegung"
    ),
    "Martha Klein": (
        "**Zusammenfassung**: Fr. Klein (92 J., PG 4) befindet sich in der Palliativsituation. "
        "Zunehmende Schwäche, sehr geringe Nahrungs- und Flüssigkeitsaufnahme (ca. 150 ml gesamt). "
        "Schmerztherapie mit Morphin 5 mg s.c. bei Bedarf (BESD 2 Punkte, aktuell kein Schmerz). "
        "Tochter und Arzt Dr. Schwarz über palliativen Ansatz informiert.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Palliative Betreuung, palliativer Ansatz bestätigt (Dr. Schwarz)\n"
        "- Vitalwerte 18:00 Uhr: RR 95/60 mmHg, Puls 68/min schwach, AF 14/min, Temp. 36,3 °C\n"
        "- Schmerztherapie: Morphin 5 mg s.c. bei Bedarf; BESD 2 Punkte\n"
        "- Flüssigkeitszufuhr: ca. 150 ml (sehr gering)\n"
        "- Mundpflege stündlich, Lagerung nach Wunsch"
    ),
    "Josef Stein": (
        "**Zusammenfassung**: Hr. Stein (71 J., PG 1) war vollständig selbstständig. "
        "Morgendliche Dusche und Ankleiden ohne Hilfe. Alle Mahlzeiten im Speisesaal. Keine Besonderheiten.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Selbstständigkeit: vollständig\n"
        "- Allgemeinzustand: gut, keine Auffälligkeiten"
    ),
    "Rosa Neumann": (
        "**Zusammenfassung**: Fr. Neumann (84 J., PG 3) war heute sehr weinerlich und traurig; "
        "vermisst ihre vor 6 Monaten verstorbene Schwester Anna. Validationsgespräch 10:00–10:30 Uhr mit Fotobetrachtung – half etwas. "
        "Nachmittags in der Ergotherapie deutlich gelöster. Sozialdienst für Begleitgespräch informiert.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Emotionaler Zustand: Trauer um verstorbene Schwester, weinerlich\n"
        "- Maßnahme: Validation, Fotobetrachtung\n"
        "- Ergotherapie nachmittags: positive Wirkung (Herbstkranz gebastelt)\n"
        "- Sozialdienst informiert"
    ),
    "Willi Baumann": (
        "**Zusammenfassung**: Hr. Baumann (PG 2) hatte Friseurtermin 10:00 Uhr (Hausbesuch). "
        "Nachmittags Kartenspiel mit Mitbewohnern. Gute Stimmung, keine medizinischen Besonderheiten.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Allgemeinzustand: gut\n"
        "- Soziale Aktivität: Friseur, Kartenspiel"
    ),
    "Emma Richter": (
        "**Zusammenfassung**: Fr. Richter (87 J., PG 3) erhielt heute eine neue Brille vom Optiker. "
        "Berichtet von deutlich besserem Sehen. Stimmung spürbar besser nach Wochen mit Frustration durch Sehprobleme.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Sehhilfe: neue Brille (deutlich verbessertes Sehvermögen)\n"
        "- Stimmung: deutlich verbessert\n"
        "- Allgemeinzustand: gut, keine weiteren Auffälligkeiten"
    ),
    "Helmut Graf": (
        "**Zusammenfassung**: Hr. Graf (83 J., PG 4) zeigte heute zunehmende Desorientierung – fragte 8–10× nach verstorbener Ehefrau Hildegard, "
        "wollte nach Hause. Validation, bekannte Musik (Volkslieder), Fotoalbum halfen nach ca. 1 Stunde. "
        "Abends etwas klarer. Demenz-Team und Dr. Meyer informiert, Visite morgen.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Desorientierung: zeitlich und örtlich, Suche nach verstorbener Ehefrau\n"
        "- Maßnahmen: Validation, Musik, Biografie-Arbeit\n"
        "- Demenz-Team (Fr. Köhler) und Dr. Meyer informiert\n"
        "- Medikationsüberprüfung morgen geplant"
    ),
    "Lieselotte Bauer": (
        "**Zusammenfassung**: Fr. Bauer (PG 2) zeigte morgens erhöhten Nüchtern-BZ von 245 mg/dl. "
        "Insulin Lantus 18 IE s.c. appliziert. BZ im Tagesverlauf sinkend: 180 → 165 → 152 mg/dl. "
        "Mittags Novorapid 6 IE vor der Mahlzeit, Essen gut eingehalten.\n\n"
        "**Wichtige Merkmale**:\n"
        "- BZ-Werte: 7:00 Uhr 245 mg/dl → 9:00 Uhr 180 → Mittag 165 → Abend 152 mg/dl\n"
        "- Insulin: Lantus 18 IE s.c. morgens; Novorapid 6 IE vor Mittagessen\n"
        "- Ernährung: diabetesgerecht, keine Beilage mittags"
    ),
    "Albert Schuster": (
        "**Zusammenfassung**: Hr. Schuster (92 J., PG 5) ist seit Schlaganfall 2024 komplett immobil und bettlägerig. "
        "Vollständige Grundpflege, Lagerung alle 3 Stunden nach Plan, Anti-Dekubitus-Matratze. "
        "Dekubitus sakral Grad 2 (3×4 cm) mit HydroClean versorgt, Fersen freigelagert. "
        "Suprapubischer Katheter unauffällig. Pürierte Vollkost, Trinkmenge 600 ml.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Dekubitus sakral Grad 2, ca. 3×4 cm; Wundversorgung HydroClean\n"
        "- Suprapubischer Katheter: Urin klar-gelb, ca. 800 ml/24 h\n"
        "- Vitalwerte 14:00 Uhr: RR 118/72 mmHg, Puls 70/min, Temp. 36,9 °C, SpO2 95 %, AF 16/min\n"
        "- Lagerungsplan: 5× täglich, Fersen freigelagert, Anti-Dekubitus-Matratze\n"
        "- Trinkmenge: 600 ml; Ernährung püriert"
    ),
    "Brunhilde Wolf": (
        "**Zusammenfassung**: Fr. Wolf (81 J., PG 3) hustete seit 2 Tagen mit gelblichem Auswurf; "
        "heute Morgen Atemnot beim Gehen. SpO2 91 %, AF 22/min, Temp. 37,9 °C, Rasselgeräusche re. basal. "
        "Dr. Lehmann diagnostizierte V. a. Pneumonie. Therapie: Amoxicillin 1000 mg 3x tgl., ACC 600 mg, Salbutamol-Inhalation.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Diagnose: Bronchitis, V. a. Pneumonie\n"
        "- Vitalwerte 9:00 Uhr: Temp. 37,9 °C, SpO2 91 % (Raumluft), AF 22/min, RR 135/85 mmHg, Puls 96/min\n"
        "- Auskultation: Rasselgeräusche re. basal\n"
        "- Medikamente: Amoxicillin 1000 mg 3x tgl., ACC 600 mg 1x tgl.\n"
        "- Inhalation: Salbutamol\n"
        "- Maßnahmen: SpO2-Überwachung, Oberkörperhochlagerung"
    ),
    "Horst Zimmermann": (
        "**Zusammenfassung**: Hr. Zimmermann (PG 1) war vollständig selbstständig. "
        "Teilnahme am Ausflug ins Stadtmuseum 10:00–14:00 Uhr, sehr begeistert. Keine Unterstützung nötig.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Selbstständigkeit: vollständig\n"
        "- Aktivität: Museumsausflug, aktive Teilnahme\n"
        "- Allgemeinzustand: sehr gut"
    ),
    "Gertrud Neumann": (
        "**Zusammenfassung**: Fr. Neumann (86 J., PG 3) hat ein Ulcus cruris am linken Unterschenkel (Grad 2, ca. 4,5×3 cm). "
        "Wundversorgung 10:00 Uhr: Reinigung mit Ringerlösung, Hydroclean plus, Schaumstoffverband, Kompressionsverband. "
        "Schmerzen 4/10 während Verbandwechsel, danach 2/10. Dr. Weber über Wundzustand informiert.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Ulcus cruris li. Unterschenkel Grad 2: 4,5×3 cm, 60 % Granulation / 40 % Fibrinbeläge\n"
        "- Wundversorgung: Ringerlösung, Hydroclean plus, Schaumstoff, Kompressionsverband\n"
        "- Schmerzen: 4/10 (VW) → 2/10\n"
        "- Umgebungshaut: ödematös, Stauungsdermatitis\n"
        "- Nächster Verbandwechsel: 30.10.2025"
    ),
    "Wolfgang Becker": (
        "**Zusammenfassung**: Hr. Becker (79 J.) war morgens 7:30 Uhr verwirrt und desorientiert. "
        "BZ-Messung ergab 52 mg/dl (Hypoglykämie). Sofortmaßnahme: 200 ml Apfelsaft. "
        "Nach 30 Minuten wieder klar, BZ 78 mg/dl. Dr. Stein reduzierte Abend-Insulindosis von 10 auf 8 IE. "
        "Nacht-BZ-Kontrollen angeordnet.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Hypoglykämie: BZ 52 mg/dl (Desorientierung, wirres Sprechen)\n"
        "- Sofortmaßnahme: 200 ml Apfelsaft → BZ 78 mg/dl → vollständige Reorientierung\n"
        "- BZ-Verlauf: 52 → 78 → 145 mg/dl (11:00 Uhr)\n"
        "- Insulinanpassung: Abend-Insulin von 10 auf 8 IE reduziert (Dr. Stein)\n"
        "- Maßnahme: Nacht-BZ-Kontrollen"
    ),
    "Hedwig Schulz": (
        "**Zusammenfassung**: Fr. Schulz (88 J., PG 4) zeigte aggressives Verhalten bei der Morgengrundpflege (schlug nach Pflegerin, schrie). "
        "Erster Versuch abgebrochen; zweiter Versuch mit anderer Pflegekraft und ruhiger Ansprache erfolgreich (Teilwäsche, frisches Nachthemd). "
        "Nachmittags erneut aggressiv, Deeskalation durch Spaziergang. Psychiatrischer Konsiliardienst für morgen angefordert.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Aggressives Verhalten: körperlich bei Grundpflege, verbal\n"
        "- Maßnahmen: Abbruch → zweiter Versuch mit anderer Pflegekraft erfolgreich\n"
        "- Verhaltensprotokoll angelegt\n"
        "- Psychiatrischer Konsiliardienst (Dr. Werner) für morgen\n"
        "- Medikationsüberprüfung (Melperon) geplant"
    ),
    "Ernst Weber": (
        "**Zusammenfassung**: Hr. Weber (77 J., PG 2) hatte heute Zahnarzttermin; 2 Zähne gezogen (li. oben). "
        "Leichte Nachblutung, nach 30 Minuten gestillt. Ibuprofen 400 mg um 16:00 Uhr. "
        "Abendessen nur weiche Kost. Schmerzen 18:00 Uhr bei 3/10.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Zahnarztbehandlung: 2 Zähne gezogen li. oben\n"
        "- Nachblutung: 30 Min., gestillt\n"
        "- Schmerzen: 5/10 → 3/10\n"
        "- Medikament: Ibuprofen 400 mg\n"
        "- Ernährung: weiche Kost (Suppe, Pudding)"
    ),
    "Anneliese Hoffmann": (
        "**Zusammenfassung**: Fr. Hoffmann (83 J., PG 3) litt seit gestern an Harnwegsinfekt-Symptomen "
        "(Brennen, häufiger Harndrang, trüber übelriechender Urin, Temp. 37,6 °C). "
        "Urin-Teststreifen: Leukozyten +++, Nitrit positiv. Cotrimoxazol 960 mg 2x tgl. für 7 Tage verordnet. "
        "Trinkmenge 1200 ml, Cranberrysaft angeboten. Schmerzen 4/10.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Diagnose: Harnwegsinfekt\n"
        "- Temp.: 37,6 °C; Urin: trüb, Leukozyten +++, Nitrit positiv\n"
        "- Antibiotikum: Cotrimoxazol 960 mg 2x tgl. 7 Tage\n"
        "- Schmerzen: 4/10 (Brennen)\n"
        "- Trinkmenge: 1200 ml\n"
        "- Maßnahmen: Urinprobe Labor, Ausscheidung beobachten, Temp. kontrollieren"
    ),
    "Karl Friedrich": (
        "**Zusammenfassung**: Hr. Friedrich (71 J., PG 1) war vollständig selbstständig. "
        "Verbrachte den Tag mit Lesen und Kreuzworträtseln; abends engagierte politische Diskussion. Keine Unterstützung nötig.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Selbstständigkeit: vollständig\n"
        "- Allgemeinzustand: sehr gut, kognitiv aktiv"
    ),
    "Hildegard Richter": (
        "**Zusammenfassung**: Fr. Richter (89 J., PG 4) präsentierte sich morgens mit massiver Atemnot und Herzinsuffizienz-Entgleisung. "
        "SpO2 88 %, RR 170/100 mmHg, AF 28/min, Puls 110/min irregulär, ausgeprägte Unterschenkelödeme, Rasselgeräusche beidseitig. "
        "EKG: Vorhofflimmern. Lasix 40 mg i.v., O2 2 l/min → SpO2 94 %. Flüssigkeitsbilanz −1000 ml.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Diagnose: Herzinsuffizienz-Entgleisung, Vorhofflimmern (EKG)\n"
        "- Vitalwerte 8:00 Uhr: RR 170/100 mmHg, Puls 110/min unregelmäßig, SpO2 88 % → nach O2 94 %, AF 28/min\n"
        "- Klinik: Unterschenkelödeme +++, Rasselgeräusche bds., Halsvenenstauung, Tachypnoe\n"
        "- Therapie: Lasix 40 mg i.v., Ramipril auf 10 mg erhöht, Metoprolol 50 mg, O2 2 l/min\n"
        "- Flüssigkeitsbilanz: Einfuhr 400 ml / Ausfuhr 1400 ml (−1000 ml)\n"
        "- Maßnahmen: engmaschige Überwachung, Bilanzierung, Gewichtskontrolle morgen"
    ),
    "Franz Lehmann": (
        "**Zusammenfassung**: Hr. Lehmann (76 J., PG 2) hatte einen Kontroll-CT-Termin am Klinikum (nach Schlaganfall vor 3 Monaten). "
        "Transport problemlos, gut gelaunt. Befund in 2–3 Tagen erwartet. "
        "Nachmittags Physiotherapie, Mobilisation gut, keine Beschwerden.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Kontroll-CT nach Schlaganfall: Untersuchung unauffällig (Befund ausstehend)\n"
        "- Physiotherapie: Mobilisation gut\n"
        "- Allgemeinzustand: gut, keine Beschwerden"
    ),
    "Elfriede Braun": (
        "**Zusammenfassung**: Fr. Braun (85 J., PG 3) zeigte heute ein ausgeprägtes Stimmungstief – niedergeschlagen, wollte nicht aufstehen. "
        "Äußerte Sinnlosigkeit und vermisst ihre Selbstständigkeit. Nach länglichem Gespräch zum Frühstück motiviert, aß aber wenig. "
        "Besuch der Enkelin verbesserte die Stimmung etwas. Sozialdienst informiert.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Stimmungstief: niedergeschlagen, Antriebsmangel\n"
        '- Aussage: "Alles ist so sinnlos"\n'
        "- Nahrungsaufnahme reduziert\n"
        "- Sozialdienst (Fr. Meier) informiert, Beratungsgespräch geplant\n"
        "- Ggf. psychiatrische Konsultation bei Verschlechterung"
    ),
    "Hermann Klein": (
        "**Zusammenfassung**: Hr. Klein (80 J., PG 2) litt heute an einem leichten Infekt (Schnupfen, Kratzen im Hals). "
        "Temp. 37,3 °C, Allgemeinzustand aber gut. Ruhetag, viel Tee, Nasenspray bei Bedarf.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Infekt: Schnupfen, Halsreizung, Temp. 37,3 °C\n"
        "- Allgemeinzustand: trotz Infekt gut\n"
        "- Maßnahmen: Ruhetag, Kamille-/Salbeitee, Nasenspray"
    ),
    "Margarethe Fischer": (
        "**Zusammenfassung**: Fr. Fischer (91 J., PG 5) befindet sich im Endstadium einer Demenz; bettlägerig, kaum kommunikativ. "
        "Vollständige Grundpflege im Bett, intensive Mundpflege, Lagerung alle 2–3 Stunden. "
        "Nahrungsaufnahme minimal (wenige Löffel), Flüssigkeit ca. 300 ml. Gewicht 42 kg (−4 kg in 4 Wochen). "
        "Palliative Betreuung laut Patientenverfügung. Tochter täglich anwesend.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Endstadium Demenz; vollständig bettlägerig, keine Kommunikation\n"
        "- Vitalwerte 15:00 Uhr: RR 100/65 mmHg, Puls 72/min schwach, Temp. 36,4 °C, AF 12/min sehr flach, SpO2 90 %\n"
        "- Gewicht: 42 kg (Gewichtsverlust −4 kg in 4 Wochen)\n"
        "- Flüssigkeit: ca. 300 ml/Tag; Nahrung: minimal\n"
        "- Palliative Betreuung: keine lebensverlängernden Maßnahmen (Patientenverfügung)\n"
        "- Dekubitus-Prophylaxe: Spezialmatratze, Lagerung alle 2–3 Std., Fersen freigelagert"
    ),
    "Walter Schulze": (
        "**Zusammenfassung**: Hr. Schulze (74 J., PG 1) war vollständig selbstständig. "
        "Ging eigenständig zum Friseur in die Stadt, nachmittags Skatrunde. Bester Laune.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Selbstständigkeit: vollständig (inkl. selbstständige Außenaktivität)\n"
        "- Allgemeinzustand: sehr gut"
    ),
    "Grete Vogel": (
        "**Zusammenfassung**: Fr. Vogel (82 J., PG 3) erhielt heute einen neuen Rollator. "
        "Einweisung durch Physiotherapeutin 14:00–14:30 Uhr, alles gut verstanden. "
        "Kleiner Spaziergang im Flur – sehr motiviert, Mobilität deutlich verbessert. Sturzrisiko weiterhin zu beachten.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Rollator (neu): Einweisung erfolgreich\n"
        "- Mobilität: deutlich verbessert\n"
        "- Sturzrisiko: weiterhin zu beachten"
    ),
    "Ludwig Mayer": (
        "**Zusammenfassung**: Hr. Mayer (87 J., PG 4) erlitt heute eine schwere COPD-Exazerbation mit massiver Dyspnoe in Ruhe, "
        "zyanotischen Lippen und kritischem SpO2 von 84 %. O2 4 l/min → SpO2 90 %, AF 32/min, Puls 118/min, Temp. 38,1 °C. "
        "Therapie: Prednisolon 40 mg i.v., Amoxicillin/Clavulansäure 1 g i.v. 3x tgl., Salbutamol + Atrovent Inhalation, Mucosolvan 30 mg. "
        "Bis 16:00 Uhr leichte Besserung (SpO2 93 %).\n\n"
        "**Wichtige Merkmale**:\n"
        "- Diagnose: Akute COPD-Exazerbation mit Infekt\n"
        "- Vitalwerte 8:00 Uhr: SpO2 84 % (krit.) → nach O2 90 % → 93 %; AF 32/min; Puls 118/min; RR 155/95 mmHg; Temp. 38,1 °C\n"
        "- Auskultation: massive Rasselgeräusche, Giemen, verlängertes Exspirium\n"
        "- Therapie: Prednisolon 40 mg i.v., Amoxicillin/Clavulansäure 1 g i.v. 3x tgl., Salbutamol, Atrovent, Mucosolvan 30 mg\n"
        "- O2: 4 l/min über Nasensonde\n"
        "- Maßnahmen: SpO2-Überwachung, bei Verschlechterung Klinikeinweisung"
    ),
    "Rosa Schmidt": (
        "**Zusammenfassung**: Fr. Schmidt (79 J., PG 2) erhielt heute neu verordnete Augentropfen gegen Glaukom. "
        "Erste Anwendung 8:00 Uhr mit pflegerischer Anleitung; Fr. Schmidt kann es selbstständig durchführen. "
        "Anwendungszeiten: 8:00 und 20:00 Uhr täglich.\n\n"
        "**Wichtige Merkmale**:\n"
        "- Neuverordnung: Augentropfen (Glaukom)\n"
        "- Erste Anwendung mit Anleitung: erfolgreich, selbstständig möglich\n"
        "- Anwendungsschema: 2× täglich (8:00 / 20:00 Uhr)"
    ),
    "Heinrich Bauer": (
        "**Zusammenfassung**: Hr. Bauer (84 J., PG 3) wurde nachts dreimal desorientiert und umherlaufend angetroffen (23:30, 01:45, 03:30 Uhr). "
        "Tagsüber müde und erschöpft, schlief bis 10:00 Uhr. Dr. Hartmann über Nachtunruhe informiert. "
        "Maßnahmen beschlossen: Nachtlicht, Toilettengang vor Schlaf, Orientierungshilfen, ggf. Medikationsüberprüfung.\n\n"
        "**Wichtige Merkmale**:\n"
        '- Nachtunruhe: 3 Episoden (desorientiert, wollte "zur Arbeit")\n'
        "- Tagdienst: müde und erschöpft\n"
        "- Dr. Hartmann informiert\n"
        "- Maßnahmen: Nachtlicht, Toilettengang vor Schlaf, große Uhr und Kalender, Medikation prüfen"
    ),
}


def main() -> None:
    with open(SCENARIOS_FILE, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    updated = 0
    for scenario in scenarios:
        patient = scenario.get("patient", "")
        summary = SUMMARIES.get(patient)
        if summary:
            scenario["reference_summary"] = summary
            updated += 1
        else:
            print(f"  [WARN] No summary written for: {patient}")

    with open(SCENARIOS_FILE, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)

    print(f"✓ Updated {updated}/{len(scenarios)} scenarios with reference_summary.")
    print(f"  Saved to {SCENARIOS_FILE}")


if __name__ == "__main__":
    main()
