"""
Hauptskript zur Generierung synthetischer handschriftlicher Pflegenotizen.
OPTIMIERT mit Multiprocessing für 2-4x schnellere Generierung!

Verwendung:
    # Standard (sequenziell):
    python -m src.data_generation.generate_synthetic_notes

    # Mit Multiprocessing (empfohlen!):
    python -m src.data_generation.generate_synthetic_notes --workers 4

    # Mit Custom-Parametern:
    python -m src.data_generation.generate_synthetic_notes --num-samples 1000 --seed 42 --workers 8
"""

import os
import random
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Import aus eigenen Modulen
from . import config
from .utils import (
    get_all_font_files,
    create_balanced_random_font_distribution,
    print_font_statistics,
    generate_single_sample,
    print_statistics,
    clear_font_cache,
)


def setup_directories():
    """Erstellt Output-Verzeichnisse falls nicht vorhanden."""
    os.makedirs(config.OUTPUT_IMAGES_PATH, exist_ok=True)
    os.makedirs(config.OUTPUT_LABELS_PATH, exist_ok=True)


def load_scenarios():
    """Lädt Patientendaten aus JSON-Datei."""
    with open(config.SCENARIOS_FILE, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    return scenarios


def collect_statistics(combination_records):
    """
    Sammelt Statistiken aus den zurückgegebenen Kombinationsdaten.

    Args:
        combination_records (list): Liste mit Kombinationsdaten

    Returns:
        tuple: (profile_usage, paper_type_usage, page_color_usage, 
                font_color_usage, style_usage)
    """
    profile_usage = {k: 0 for k in config.PAGE_PROFILES.keys()}
    paper_type_usage = {k: 0 for k in config.PAPER_TYPES.keys()}
    page_color_usage = {color: 0 for color in config.PAGE_COLORS.keys()}
    font_color_usage = {color: 0 for color in config.FONT_COLORS.keys()}
    style_usage = {"normal": 0, "stressed": 0, "careful": 0}

    for record in combination_records:
        profile_usage[record['profile']] += 1
        paper_type_usage[record['paper_type']] += 1
        page_color_usage[record['page_color']] += 1
        font_color_usage[record['font_color']] += 1
        style_usage[record['writing_style']] += 1

    return (profile_usage, paper_type_usage, page_color_usage, 
            font_color_usage, style_usage)


def main(num_samples=None, seed=None, workers=None):
    """
    Hauptfunktion zur Generierung synthetischer Pflegenotizen.
    OPTIMIERT mit Multiprocessing-Support!

    Args:
        num_samples (int, optional): Anzahl zu generierender Samples
        seed (int, optional): Random Seed
        workers (int, optional): Anzahl Worker für Multiprocessing (None = sequenziell)
    """
    # Setze Defaults
    num_samples = num_samples or config.NUM_SAMPLES_TO_GENERATE
    seed = seed or config.RANDOM_SEED
    random.seed(seed)

    # Bestimme Verarbeitungsmodus
    use_multiprocessing = workers is not None and workers > 1
    if workers is None:
        actual_workers = 1
    else:
        actual_workers = min(workers, cpu_count())

    # Header
    print("=" * 80)
    print("REALISTISCHE PFLEGENOTIZEN-SIMULATION")
    print("Optimiert für authentische Handschrift-Variationen")
    if use_multiprocessing:
        print(f"🚀 MULTIPROCESSING AKTIVIERT: {actual_workers} Worker")
    print("=" * 80)

    # Setup
    print("\n📁 Vorbereitung...")
    setup_directories()

    # Lade Fonts
    print("\n📁 Lade Schriftarten...")
    fonts = get_all_font_files(config.FONTS_PATH)
    if not fonts:
        raise FileNotFoundError(f"Keine Schriftarten in '{config.FONTS_PATH}' gefunden!")
    print(f"✓ {len(fonts)} Schriftarten gefunden.")

    # Lade Szenarien
    print("\n📝 Lade Patientendaten...")
    scenarios = load_scenarios()
    print(f"✓ {len(scenarios)} Patienten geladen.")

    # Font-Verteilung
    print(f"\n{'='*80}")
    print("BERECHNE FONT-VERTEILUNG")
    print(f"{'='*80}")
    font_distribution = create_balanced_random_font_distribution(fonts, num_samples, seed=seed)
    print_font_statistics(font_distribution, fonts)

    # Erstelle Config-Dict für generate_single_sample
    config_dict = {
        'PAGE_PROFILES': config.PAGE_PROFILES,
        'PAGE_COLORS': config.PAGE_COLORS,
        'PAGE_COLOR_WEIGHTS': config.PAGE_COLOR_WEIGHTS,
        'FONT_COLORS': config.FONT_COLORS,
        'FONT_COLOR_WEIGHTS': config.FONT_COLOR_WEIGHTS,
        'PAPER_TYPES': config.PAPER_TYPES,
        'PAPER_TYPE_WEIGHTS': config.PAPER_TYPE_WEIGHTS,
        'ROTATION_RANGE': config.ROTATION_RANGE,
        'STRESS_PROBABILITY': config.STRESS_PROBABILITY,
        'CAREFUL_PROBABILITY': config.CAREFUL_PROBABILITY,
        'OUTPUT_IMAGES_PATH': config.OUTPUT_IMAGES_PATH,
        'OUTPUT_LABELS_PATH': config.OUTPUT_LABELS_PATH,
        'RANDOM_SEED': seed,
        'FONT_DISTRIBUTION_STRATEGY': config.FONT_DISTRIBUTION_STRATEGY,
    }

    # Erstelle Argument-Liste für Worker
    args_list = []
    for i in range(num_samples):
        scenario = random.choice(scenarios)
        font_path = font_distribution[i]
        args_list.append((
            i,
            font_path,
            scenario,
            config_dict,
            i,  # seed_offset für Worker-spezifische Randomisierung
        ))

    # Hauptschleife
    print(f"\n{'='*80}")
    print("STARTE DATENGENERIERUNG")
    if use_multiprocessing:
        print(f"Modus: PARALLEL ({actual_workers} CPU-Kerne)")
    else:
        print("Modus: SEQUENZIELL (1 CPU-Kern)")
    print(f"{'='*80}\n")

    combination_records = []

    if use_multiprocessing:
        # MULTIPROCESSING: Parallel mit Pool
        with Pool(processes=actual_workers) as pool:
            for result in tqdm(
                pool.imap(generate_single_sample, args_list),
                total=num_samples,
                desc=f"Generiere parallel ({actual_workers} workers)"
            ):
                combination_records.append(result)
    else:
        # SEQUENZIELL: Standard for-Loop
        for args in tqdm(args_list, desc="Generiere sequenziell"):
            result = generate_single_sample(args)
            combination_records.append(result)

    # Sammle Statistiken aus den Ergebnissen
    (profile_usage, paper_type_usage, page_color_usage, 
     font_color_usage, style_usage) = collect_statistics(combination_records)

    # Statistiken ausgeben
    print_statistics(
        num_samples,
        config_dict,
        profile_usage,
        paper_type_usage,
        page_color_usage,
        font_color_usage,
        style_usage,
        combination_records,
    )

    # Cache leeren (optional, für saubere Beendigung)
    if use_multiprocessing:
        clear_font_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generiert synthetische handschriftliche Pflegenotizen (OPTIMIERT)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help=f"Anzahl zu generierender Samples (Standard: {config.NUM_SAMPLES_TO_GENERATE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random Seed für Reproduzierbarkeit (Standard: {config.RANDOM_SEED})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Anzahl CPU-Kerne für Multiprocessing (Standard: 1=sequenziell, empfohlen: {cpu_count()})",
    )

    args = parser.parse_args()

    # Info ausgeben
    if args.workers and args.workers > 1:
        max_workers = cpu_count()
        actual = min(args.workers, max_workers)
        print(f"\n⚡ Multiprocessing aktiviert: {actual} von {max_workers} CPU-Kernen\n")

    main(num_samples=args.num_samples, seed=args.seed, workers=args.workers)