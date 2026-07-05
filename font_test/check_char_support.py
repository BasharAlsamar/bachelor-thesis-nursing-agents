#!/usr/bin/env python3
"""
Simple script to check if fonts support specific characters.
Usage: python check_char_support.py
"""

from pathlib import Path
from fontTools.ttLib import TTFont


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
        cmap = font.getBestCmap()

        if cmap is None:
            return {char: False for char in test_string}

        results = {}
        for char in test_string:
            char_code = ord(char)
            results[char] = char_code in cmap

        font.close()
        return results
    except Exception as e:
        print(f"Fehler beim Lesen von {font_path}: {e}")
        return {char: None for char in test_string}


def get_all_fonts(fonts_dir):
    """Sammelt alle Font-Dateien."""
    fonts = {}
    fonts_path = Path(fonts_dir)

    for font_folder in fonts_path.iterdir():
        if font_folder.is_dir():
            for font_file in font_folder.iterdir():
                if font_file.suffix.lower() in [".ttf", ".otf"]:
                    fonts[font_folder.name] = str(font_file)
                    break

    return fonts


def main():
    # Test-Zeichen: German special chars, numbers, and common special characters
    german_chars = "üöäÜÖÄß"
    numbers = "0123456789"
    special_chars = "-!\"'?/%@#$&*().,;:"
    test_chars = german_chars + numbers + special_chars

    print("=" * 80)
    print(f"FONT CHARACTER SUPPORT CHECK")
    print("=" * 80)
    print(f"Testing characters:")
    print(f"  German: {german_chars}")
    print(f"  Numbers: {numbers}")
    print(f"  Special: {special_chars}")
    print("=" * 80 + "\n")

    # Pfad zu Fonts
    base_dir = Path(__file__).parent.parent
    fonts_dir = base_dir / "data" / "synthetic" / "fonts"
    output_file = Path(__file__).parent / "fonts_with_missing_chars.txt"

    # Alle Fonts sammeln
    fonts = get_all_fonts(fonts_dir)
    print(f"Gefundene Fonts: {len(fonts)}\n")

    # Statistiken
    fully_supported = []
    partially_supported = []
    not_supported = []
    fonts_with_issues = []  # For txt file

    # Prüfe jeden Font
    for font_name, font_path in sorted(fonts.items()):
        char_support = check_font_characters(font_path, test_chars)

        supported = [
            char for char, is_supported in char_support.items() if is_supported
        ]
        missing = [
            char for char, is_supported in char_support.items() if is_supported is False
        ]
        unknown = [
            char for char, is_supported in char_support.items() if is_supported is None
        ]

        if not missing and not unknown:
            status = "✓"
            fully_supported.append(font_name)
            print(f"{status} {font_name:40s} - Alle Zeichen unterstützt")
        elif missing:
            status = "⚠️"
            partially_supported.append(font_name)
            fonts_with_issues.append((font_name, missing))
            missing_str = (
                ", ".join(missing)
                if len(missing) <= 10
                else f"{', '.join(missing[:10])}... ({len(missing)} total)"
            )
            print(f"{status} {font_name:40s} - Fehlend: {missing_str}")
        else:
            not_supported.append(font_name)
            fonts_with_issues.append((font_name, ["Could not check"]))
            print(f"? {font_name:40s} - Konnte nicht geprüft werden")

    # Schreibe Fonts mit fehlenden Zeichen in txt-Datei
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("FONTS WITH MISSING CHARACTERS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test characters: {test_chars}\n")
        f.write(f"Total fonts tested: {len(fonts)}\n")
        f.write(f"Fonts with missing characters: {len(fonts_with_issues)}\n\n")
        f.write("=" * 80 + "\n\n")

        for font_name, missing in fonts_with_issues:
            f.write(f"{font_name}\n")
            if missing != ["Could not check"]:
                f.write(f"  Missing: {', '.join(missing)}\n")
            else:
                f.write(f"  Status: Could not check\n")
            f.write("\n")

    # Zusammenfassung
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)
    print(f"✓ Voll unterstützt:     {len(fully_supported)} Fonts")
    print(f"⚠️  Teilweise unterstützt: {len(partially_supported)} Fonts")
    print(f"? Nicht prüfbar:        {len(not_supported)} Fonts")
    print(f"\n📝 Fonts mit fehlenden Zeichen wurden gespeichert in:")
    print(f"   {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
