"""
Zentrale Konfiguration für die synthetische Pflegenotizen-Generierung.

Alle Parameter, Profile und Gewichtungen sind hier zentral definiert.
"""

import random
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# GRUNDEINSTELLUNGEN
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 2
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Pfade
FONTS_PATH = "data/synthetic/fonts/"
SCENARIOS_FILE = "data/synthetic/scenarios.json"
OUTPUT_IMAGES_PATH = "data/synthetic/output/images/"
OUTPUT_LABELS_PATH = "data/synthetic/output/labels/"

# Anzahl zu generierende Samples
NUM_SAMPLES_TO_GENERATE = 500

# ══════════════════════════════════════════════════════════════════════════════
# SEITENPROFILE (Papierformate & DPI)
# ══════════════════════════════════════════════════════════════════════════════

PAGE_PROFILES = {
    "A5_150dpi": {
        "size": (874, 1240),
        "font_size_range": (24, 38),
        "margin_left": 50,
        "margin_right": 50,
        "margin_top": 40,
        "margin_bottom": 60,
        "position_x_range": (40, 120),
        "position_y_range": (40, 200),
        "comfortable_char_limit": 700,
        "absolute_max_chars": 1000,
    },
    "A4_150dpi": {
        "size": (1240, 1754),
        "font_size_range": (28, 42),
        "margin_left": 60,
        "margin_right": 60,
        "margin_top": 50,
        "margin_bottom": 80,
        "position_x_range": (50, 180),
        "position_y_range": (50, 300),
        "comfortable_char_limit": 1500,
        "absolute_max_chars": 2200,
    },
    "A4_300dpi": {
        "size": (2480, 3508),
        "font_size_range": (56, 84),
        "margin_left": 120,
        "margin_right": 120,
        "margin_top": 100,
        "margin_bottom": 160,
        "position_x_range": (100, 360),
        "position_y_range": (100, 600),
        "comfortable_char_limit": 2200,
        "absolute_max_chars": 4000,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# SEITENFARBEN
# ══════════════════════════════════════════════════════════════════════════════

PAGE_COLORS = {
    "white": (255, 255, 255),
    "cream": (250, 249, 246),
    "light_gray": (245, 245, 245),
    "yellow": (255, 253, 208),
    "beige": (245, 235, 220),
}

PAGE_COLOR_WEIGHTS = [20, 20, 20, 20, 20]  # Gleichverteilt

# ══════════════════════════════════════════════════════════════════════════════
# SCHRIFTFARBEN
# ══════════════════════════════════════════════════════════════════════════════

FONT_COLORS = {
    "black": (0, 0, 0),
    "dark_blue": (26, 35, 126),
    "dark_gray": (55, 55, 55),
}

FONT_COLOR_WEIGHTS = [65, 20, 15]  # Schwarz dominiert

# ══════════════════════════════════════════════════════════════════════════════
# PAPIERTYPEN
# ══════════════════════════════════════════════════════════════════════════════

PAPER_TYPES = {
    "blanko": {
        "name": "Blanko (leer)",
        "pattern": None,
    },
    "liniert": {
        "name": "Liniert",
        "pattern": "horizontal_lines",
        "line_spacing_mm": 9,
    },
    "kariert": {
        "name": "Kariert",
        "pattern": "grid",
        "grid_size_mm": 5,
    },
    "dotted": {
        "name": "Dotted (Punktraster)",
        "pattern": "dot_grid",
        "dot_spacing_mm": 5,
        "dot_size_px": 2,
    },
}

PAPER_TYPE_WEIGHTS = [25, 25, 25, 25]  # Gleichverteilt

# ══════════════════════════════════════════════════════════════════════════════
# SCHREIBSTIL-VARIATIONEN
# ══════════════════════════════════════════════════════════════════════════════

ROTATION_RANGE = (-4.0, 4.0)  # Natürliche Rotation in Grad

STRESS_PROBABILITY = 0.15  # 15% gestresste/schnelle Notizen
CAREFUL_PROBABILITY = 0.10  # 10% sehr ordentliche Notizen

# ══════════════════════════════════════════════════════════════════════════════
# FONT-VERTEILUNGSSTRATEGIE
# ══════════════════════════════════════════════════════════════════════════════

FONT_DISTRIBUTION_STRATEGY = "BALANCED_RANDOM"  # Alle Fonts gleichmäßig