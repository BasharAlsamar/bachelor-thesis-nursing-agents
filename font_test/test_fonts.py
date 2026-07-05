"""
Script to generate one test image per font.
Each image will show sample text from scenarios.json rendered in that font.
"""

import os
import json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from fontTools.ttLib import TTFont


def get_all_font_files(fonts_dir):
    """
    Sammelt alle Font-Dateien (.ttf, .otf) aus dem Fonts-Verzeichnis.

    Args:
        fonts_dir (str): Pfad zum Fonts-Verzeichnis

    Returns:
        dict: {font_name: font_path}
    """
    fonts = {}
    fonts_path = Path(fonts_dir)

    for font_folder in fonts_path.iterdir():
        if font_folder.is_dir():
            font_name = font_folder.name
            # Suche nach .ttf oder .otf Dateien
            for font_file in font_folder.iterdir():
                if font_file.suffix.lower() in [".ttf", ".otf"]:
                    fonts[font_name] = str(font_file)
                    break  # Nur die erste Font-Datei pro Ordner nehmen

    return fonts


def check_font_characters(font_path, test_string="üöä-123"):
    """
    Prüft, ob ein Font bestimmte Zeichen unterstützt.

    Args:
        font_path (str): Pfad zur Font-Datei
        test_string (str): Zeichen zum Testen

    Returns:
        dict: {character: bool} - True wenn unterstützt, False wenn nicht
    """
    try:
        font = TTFont(font_path)
        # Hole alle verfügbaren Zeichen aus der cmap (character map)
        cmap = font.getBestCmap()

        if cmap is None:
            return {char: False for char in test_string}

        # Prüfe jedes Zeichen
        results = {}
        for char in test_string:
            char_code = ord(char)
            results[char] = char_code in cmap

        font.close()
        return results
    except Exception as e:
        print(f"    Fehler beim Prüfen der Zeichen: {e}")
        return {char: None for char in test_string}


def load_sample_text(scenarios_file):
    """
    Lädt einen Beispieltext aus scenarios.json.

    Args:
        scenarios_file (str): Pfad zur scenarios.json

    Returns:
        str: Beispieltext
    """
    with open(scenarios_file, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    # Nimm die ersten paar Zeilen des ersten Szenarios
    if scenarios:
        full_text = scenarios[0]["notes"]
        # Nimm nur die ersten 300 Zeichen
        lines = full_text.split("\n")[:5]  # Erste 5 Zeilen
        return "\n".join(lines)

    return "Sample handwriting text for font testing."


def create_test_image(font_path, font_name, text, output_path, image_size=(800, 400)):
    """
    Erstellt ein Test-Bild mit dem gegebenen Font.

    Args:
        font_path (str): Pfad zur Font-Datei
        font_name (str): Name des Fonts
        text (str): Text zum Rendern
        output_path (str): Pfad für das Output-Bild
        image_size (tuple): Größe des Bildes (width, height)
    """
    # Erstelle weißes Bild
    img = Image.new("RGB", image_size, color="white")
    draw = ImageDraw.Draw(img)

    # Versuche den Font zu laden
    try:
        font = ImageFont.truetype(font_path, size=28)
    except Exception as e:
        print(f"  ⚠️  Fehler beim Laden von {font_name}: {e}")
        # Fallback auf Default-Font
        font = ImageFont.load_default()

    # Zeichne den Text
    margin = 20
    y_position = margin

    # Zeichne Zeile für Zeile
    for line in text.split("\n"):
        if y_position > image_size[1] - 50:  # Verhindere Überlauf
            break
        draw.text((margin, y_position), line, fill="black", font=font)
        y_position += 35  # Zeilenabstand

    # Speichere das Bild
    img.save(output_path)


def main():
    """Hauptfunktion zur Generierung der Test-Bilder."""
    print("=" * 80)
    print("FONT TEST - Generiere ein Bild pro Font")
    print("=" * 80)

    # Pfade
    base_dir = Path(__file__).parent.parent
    fonts_dir = base_dir / "data" / "synthetic" / "fonts"
    scenarios_file = base_dir / "data" / "synthetic" / "scenarios.json"
    output_dir = Path(__file__).parent / "images"

    # Erstelle Output-Verzeichnis
    output_dir.mkdir(exist_ok=True)

    print(f"\n📁 Fonts-Verzeichnis: {fonts_dir}")
    print(f"📝 Szenarien-Datei: {scenarios_file}")
    print(f"💾 Output-Verzeichnis: {output_dir}\n")

    # Lade alle Fonts
    print("🔍 Suche Fonts...")
    fonts = get_all_font_files(fonts_dir)
    print(f"✓ {len(fonts)} Fonts gefunden.\n")

    # Prüfe Zeichen-Unterstützung
    test_chars = "üöä-123"
    print(f"🔤 Prüfe Zeichen-Unterstützung für: '{test_chars}'")
    print(f"{'='*80}\n")

    for font_name, font_path in sorted(fonts.items()):
        char_support = check_font_characters(font_path, test_chars)
        supported = [
            char for char, is_supported in char_support.items() if is_supported
        ]
        missing = [
            char for char, is_supported in char_support.items() if is_supported is False
        ]

        if missing:
            print(f"⚠️  {font_name}: Fehlende Zeichen: {missing}")
        else:
            print(f"✓ {font_name}: Alle Zeichen unterstützt")

    print(f"\n{'='*80}\n")

    # Lade Beispieltext
    print("📖 Lade Beispieltext...")
    sample_text = '12.25.26 ßäöü ÄÖÜ !§$%&//()"' + load_sample_text(scenarios_file)
    print(f"✓ Text geladen ({len(sample_text)} Zeichen).\n")

    # Generiere für jeden Font ein Bild
    print(f"{'='*80}")
    print("GENERIERE TEST-BILDER")
    print(f"{'='*80}\n")

    success_count = 0
    error_count = 0

    for font_name, font_path in sorted(fonts.items()):
        output_path = output_dir / f"{font_name}.png"

        try:
            create_test_image(font_path, font_name, sample_text, str(output_path))
            print(f"✓ {font_name}.png")
            success_count += 1
        except Exception as e:
            print(f"✗ {font_name}: {e}")
            error_count += 1

    # Zusammenfassung
    print(f"\n{'='*80}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*80}")
    print(f"✓ Erfolgreich: {success_count}")
    print(f"✗ Fehler: {error_count}")
    print(f"📁 Bilder gespeichert in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
