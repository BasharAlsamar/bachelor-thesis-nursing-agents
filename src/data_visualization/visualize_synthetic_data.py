#!/usr/bin/env python3
"""Visualize synthetic data features.

Generates:
- Age histogram and care level counts from data/synthetic/scenarios.json.
- Categorical counts from data/synthetic/output/labels/*.json.
"""

from __future__ import annotations

import argparse
import math
import json
import os
from collections import Counter
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

LABEL_FIELDS = [
    "profile",
    "paper_type",
    "page_color",
    "font_color",
    "writing_style",
    "font_selection_strategy",
]

COMBINED_GROUPS = (
    ("writing_style", "profile", "labels_writing_style_profile_counts.png"),
    ("page_color", "paper_type", "labels_page_color_paper_type_counts.png"),
)


def load_json_array(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
    return data


def find_label_files(labels_dir: str) -> List[str]:
    files = []
    for name in os.listdir(labels_dir):
        if name.lower().endswith(".json"):
            files.append(os.path.join(labels_dir, name))
    return sorted(files)


def count_field(values: Iterable[str]) -> Counter:
    counter = Counter()
    for value in values:
        if value is None or value == "":
            counter["MISSING"] += 1
        else:
            counter[str(value)] += 1
    return counter


def annotate_bars(ax: plt.Axes, fontsize: int = 10) -> None:
    for bar in ax.patches:
        height = bar.get_height()
        if height <= 0:
            continue
        ax.annotate(
            f"{int(height)}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            xytext=(0, 4),
            textcoords="offset points",
        )


def save_bar_chart(counter: Counter, title: str, xlabel: str, output_path: str) -> None:
    if not counter:
        return
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    labels = [item[0] for item in items]
    counts = [item[1] for item in items]

    fig_width = max(8, min(20, 0.75 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    sns.barplot(x=labels, y=counts, palette="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelrotation=45)
    ax.margins(x=0.01)
    annotate_bars(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_dual_bar_chart(
    left: Counter,
    right: Counter,
    left_title: str,
    right_title: str,
    output_path: str,
) -> None:
    if not left and not right:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    left_items = sorted(left.items(), key=lambda item: (-item[1], item[0]))
    left_labels = [item[0] for item in left_items]
    left_counts = [item[1] for item in left_items]
    sns.barplot(x=left_labels, y=left_counts, palette="viridis", ax=axes[0])
    axes[0].set_title(left_title)
    axes[0].set_xlabel(left_title)
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", labelrotation=45)
    annotate_bars(axes[0])

    right_items = sorted(right.items(), key=lambda item: (-item[1], item[0]))
    right_labels = [item[0] for item in right_items]
    right_counts = [item[1] for item in right_items]
    sns.barplot(x=right_labels, y=right_counts, palette="Set2", ax=axes[1])
    axes[1].set_title(right_title)
    axes[1].set_xlabel(right_title)
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", labelrotation=45)
    annotate_bars(axes[1])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_page_color_by_paper_type(
    paper_types: List[str],
    page_colors: List[str],
    output_path: str,
) -> None:
    paired = [
        (ptype, pcolor)
        for ptype, pcolor in zip(paper_types, page_colors)
        if ptype and pcolor
    ]
    if not paired:
        return

    types = [item[0] for item in paired]
    colors = [item[1] for item in paired]

    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    ax_bottom = fig.add_subplot(grid[1, :])

    type_counter = count_field(types)
    color_counter = count_field(colors)

    type_items = sorted(type_counter.items(), key=lambda item: (-item[1], item[0]))
    type_labels = [item[0] for item in type_items]
    type_counts = [item[1] for item in type_items]
    sns.barplot(x=type_labels, y=type_counts, palette="Set2", ax=ax_left)
    ax_left.set_title("paper_type")
    ax_left.set_xlabel("paper_type")
    ax_left.set_ylabel("Count")
    ax_left.tick_params(axis="x", labelrotation=45)
    annotate_bars(ax_left)

    color_items = sorted(color_counter.items(), key=lambda item: (-item[1], item[0]))
    color_labels = [item[0] for item in color_items]
    color_counts = [item[1] for item in color_items]
    sns.barplot(x=color_labels, y=color_counts, palette="viridis", ax=ax_right)
    ax_right.set_title("page_color")
    ax_right.set_xlabel("page_color")
    ax_right.set_ylabel("Count")
    ax_right.tick_params(axis="x", labelrotation=45)
    annotate_bars(ax_right)

    sns.countplot(x=types, hue=colors, palette="tab10", ax=ax_bottom)
    ax_bottom.set_title("page_color by paper_type")
    ax_bottom.set_xlabel("paper_type")
    ax_bottom.set_ylabel("Count")
    ax_bottom.tick_params(axis="x", labelrotation=45)
    ax_bottom.legend(title="page_color", fontsize=9, title_fontsize=10)
    annotate_bars(ax_bottom, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_writing_style_by_profile(
    writing_styles: List[str],
    profiles: List[str],
    output_path: str,
) -> None:
    paired = [
        (style, profile)
        for style, profile in zip(writing_styles, profiles)
        if style and profile
    ]
    if not paired:
        return

    styles = [item[0] for item in paired]
    profile_vals = [item[1] for item in paired]

    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    ax_bottom = fig.add_subplot(grid[1, :])

    style_counter = count_field(styles)
    profile_counter = count_field(profile_vals)

    style_items = sorted(style_counter.items(), key=lambda item: (-item[1], item[0]))
    style_labels = [item[0] for item in style_items]
    style_counts = [item[1] for item in style_items]
    sns.barplot(x=style_labels, y=style_counts, palette="viridis", ax=ax_left)
    ax_left.set_title("writing_style")
    ax_left.set_xlabel("writing_style")
    ax_left.set_ylabel("Count")
    ax_left.tick_params(axis="x", labelrotation=45)
    annotate_bars(ax_left)

    profile_items = sorted(
        profile_counter.items(), key=lambda item: (-item[1], item[0])
    )
    profile_labels = [item[0] for item in profile_items]
    profile_counts = [item[1] for item in profile_items]
    sns.barplot(x=profile_labels, y=profile_counts, palette="Set2", ax=ax_right)
    ax_right.set_title("profile")
    ax_right.set_xlabel("profile")
    ax_right.set_ylabel("Count")
    ax_right.tick_params(axis="x", labelrotation=45)
    annotate_bars(ax_right)

    sns.countplot(x=styles, hue=profile_vals, palette="tab10", ax=ax_bottom)
    ax_bottom.set_title("profile by writing_style")
    ax_bottom.set_xlabel("writing_style")
    ax_bottom.set_ylabel("Count")
    ax_bottom.tick_params(axis="x", labelrotation=45)
    ax_bottom.legend(title="profile", fontsize=9, title_fontsize=10)
    annotate_bars(ax_bottom, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_age_bins(ages: List[int], bin_width: int = 5) -> List[int]:
    if not ages:
        return []
    min_age = min(ages)
    max_age = max(ages)
    start = (min_age // bin_width) * bin_width
    end = ((max_age // bin_width) + 1) * bin_width
    return list(range(start, end + 1, bin_width))


def bin_ages(ages: List[int], bins: List[int]) -> List[str]:
    if not ages or not bins:
        return []

    labels = []
    for age in ages:
        for i in range(len(bins) - 1):
            if bins[i] <= age < bins[i + 1]:
                labels.append(f"{bins[i]}-{bins[i + 1] - 1}")
                break
        else:
            labels.append(f"{bins[-2]}-{bins[-1] - 1}")
    return labels


def save_age_and_care_level(
    ages: List[int],
    care_levels: List[int],
    output_path: str,
) -> None:
    paired = [
        (age, level)
        for age, level in zip(ages, care_levels)
        if isinstance(age, int) and isinstance(level, int)
    ]
    if not paired:
        return

    paired_ages = [item[0] for item in paired]
    paired_levels = [item[1] for item in paired]
    bins = build_age_bins(paired_ages, bin_width=5)
    age_bins = bin_ages(paired_ages, bins)

    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    ax_bottom = fig.add_subplot(grid[1, :])

    sns.histplot(
        paired_ages,
        bins=bins,
        color="#F58518",
        edgecolor="black",
        ax=ax_left,
    )
    ax_left.set_title("Age Distribution")
    ax_left.set_xlabel("Age")
    ax_left.set_ylabel("Count")
    annotate_bars(ax_left, fontsize=9)

    care_counter = Counter(paired_levels)
    care_items = sorted(care_counter.items())
    care_labels = [str(item[0]) for item in care_items]
    care_counts = [item[1] for item in care_items]
    sns.barplot(x=care_labels, y=care_counts, palette="Set2", ax=ax_right)
    ax_right.set_title("Care Level Counts")
    ax_right.set_xlabel("Care Level")
    ax_right.set_ylabel("Count")
    annotate_bars(ax_right)

    sns.countplot(x=age_bins, hue=paired_levels, palette="tab10", ax=ax_bottom)
    ax_bottom.set_title("care_level by age bin")
    ax_bottom.set_xlabel("Age Bin")
    ax_bottom.set_ylabel("Count")
    ax_bottom.tick_params(axis="x", labelrotation=45)
    ax_bottom.legend(title="care_level", fontsize=9, title_fontsize=10)
    annotate_bars(ax_bottom, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_image_grid(
    image_dir: str,
    output_dir: str,
    base_name: str = "n30_montage",
    images_per_page: int = 12,
    cols: int = 4,
) -> None:
    if not os.path.isdir(image_dir):
        return

    files = [
        os.path.join(image_dir, name)
        for name in sorted(os.listdir(image_dir))
        if name.lower().endswith(".png")
    ]
    if not files:
        return

    rows = math.ceil(images_per_page / cols)
    total_pages = math.ceil(len(files) / images_per_page)

    for page in range(total_pages):
        start = page * images_per_page
        end = start + images_per_page
        page_files = files[start:end]

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.8))
        if rows == 1:
            axes = [axes]

        for idx, path in enumerate(page_files):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col] if rows > 1 else axes[0][col]
            image = plt.imread(path)
            ax.imshow(image)
            ax.axis("off")

        total = rows * cols
        for idx in range(len(page_files), total):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col] if rows > 1 else axes[0][col]
            ax.axis("off")

        fig.tight_layout(pad=0.2)
        output_path = os.path.join(output_dir, f"{base_name}_{page + 1:02d}.png")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def save_n30_image_size_stats(
    image_dir: str,
    output_dir: str,
    plot_name: str = "n30_image_size_stats.png",
    summary_name: str = "n30_image_size_summary.txt",
) -> None:
    if not os.path.isdir(image_dir):
        return

    files = [
        os.path.join(image_dir, name)
        for name in sorted(os.listdir(image_dir))
        if name.lower().endswith(".png")
    ]
    if not files:
        return

    widths = []
    heights = []
    for path in files:
        with Image.open(path) as image:
            width, height = image.size
        widths.append(width)
        heights.append(height)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(widths, bins=15, color="#4C78A8", edgecolor="black", ax=axes[0])
    axes[0].set_title("N30 Image Widths")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Count")
    annotate_bars(axes[0], fontsize=9)

    sns.histplot(heights, bins=15, color="#F58518", edgecolor="black", ax=axes[1])
    axes[1].set_title("N30 Image Heights")
    axes[1].set_xlabel("Height (px)")
    axes[1].set_ylabel("Count")
    annotate_bars(axes[1], fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, plot_name), dpi=150)
    plt.close(fig)

    summary_path = os.path.join(output_dir, summary_name)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("N30 image size summary\n")
        handle.write(f"count: {len(files)}\n")
        handle.write(
            f"width min/mean/max: {min(widths)} / {sum(widths)/len(widths):.2f} / {max(widths)}\n"
        )
        handle.write(
            f"height min/mean/max: {min(heights)} / {sum(heights)/len(heights):.2f} / {max(heights)}\n"
        )


def save_n30_label_distributions(
    image_dir: str,
    labels_dir: str,
    output_path: str,
) -> None:
    if not os.path.isdir(image_dir) or not os.path.isdir(labels_dir):
        return

    image_names = [
        os.path.splitext(name)[0]
        for name in sorted(os.listdir(image_dir))
        if name.lower().endswith(".png")
    ]
    if not image_names:
        return

    records = []
    for name in image_names:
        label_path = os.path.join(labels_dir, f"{name}.json")
        if not os.path.isfile(label_path):
            continue
        try:
            with open(label_path, "r", encoding="utf-8") as handle:
                records.append(json.load(handle))
        except json.JSONDecodeError:
            continue

    if not records:
        return

    paper_types = []
    page_colors = []
    for item in records:
        generation = item.get("generation_parameters")
        if isinstance(generation, dict):
            paper_types.append(generation.get("paper_type"))
            page_colors.append(generation.get("page_color"))
        else:
            paper_types.append(item.get("paper_type"))
            page_colors.append(item.get("page_color"))

    paired = [
        (ptype, pcolor)
        for ptype, pcolor in zip(paper_types, page_colors)
        if ptype and pcolor
    ]
    if not paired:
        return

    types = [item[0] for item in paired]
    colors = [item[1] for item in paired]

    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    ax_bottom = fig.add_subplot(grid[1, :])

    type_counter = count_field(types)
    color_counter = count_field(colors)

    type_items = sorted(type_counter.items(), key=lambda item: (-item[1], item[0]))
    type_labels = [item[0] for item in type_items]
    type_counts = [item[1] for item in type_items]
    sns.barplot(x=type_labels, y=type_counts, palette="Set2", ax=ax_left)
    ax_left.set_title("N30 paper_type")
    ax_left.set_xlabel("paper_type")
    ax_left.set_ylabel("Count")
    ax_left.tick_params(axis="x", labelrotation=45)
    annotate_bars(ax_left)

    color_items = sorted(color_counter.items(), key=lambda item: (-item[1], item[0]))
    color_labels = [item[0] for item in color_items]
    color_counts = [item[1] for item in color_items]
    sns.barplot(x=color_labels, y=color_counts, palette="viridis", ax=ax_right)
    ax_right.set_title("N30 page_color")
    ax_right.set_xlabel("page_color")
    ax_right.set_ylabel("Count")
    ax_right.tick_params(axis="x", labelrotation=45)
    annotate_bars(ax_right)

    sns.countplot(x=types, hue=colors, palette="tab10", ax=ax_bottom)
    ax_bottom.set_title("N30 page_color by paper_type")
    ax_bottom.set_xlabel("paper_type")
    ax_bottom.set_ylabel("Count")
    ax_bottom.tick_params(axis="x", labelrotation=45)
    ax_bottom.legend(title="page_color", fontsize=9, title_fontsize=10)
    annotate_bars(ax_bottom, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_test_set_label_distributions(
    test_set_path: str,
    output_path: str,
) -> None:
    if not os.path.isfile(test_set_path):
        return

    try:
        with open(test_set_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return

    samples = payload.get("samples")
    if not isinstance(samples, list):
        return

    records = []
    for sample in samples:
        label_path = sample.get("label_path")
        if not label_path or not os.path.isfile(label_path):
            continue
        try:
            with open(label_path, "r", encoding="utf-8") as handle:
                records.append(json.load(handle))
        except json.JSONDecodeError:
            continue

    if not records:
        return

    paper_types = []
    page_colors = []
    for item in records:
        generation = item.get("generation_parameters")
        if isinstance(generation, dict):
            paper_types.append(generation.get("paper_type"))
            page_colors.append(generation.get("page_color"))
        else:
            paper_types.append(item.get("paper_type"))
            page_colors.append(item.get("page_color"))

    paired = [
        (ptype, pcolor)
        for ptype, pcolor in zip(paper_types, page_colors)
        if ptype and pcolor
    ]
    if not paired:
        return

    types = [item[0] for item in paired]
    colors = [item[1] for item in paired]

    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    ax_bottom = fig.add_subplot(grid[1, :])

    type_counter = count_field(types)
    color_counter = count_field(colors)

    type_items = sorted(type_counter.items(), key=lambda item: (-item[1], item[0]))
    type_labels = [item[0] for item in type_items]
    type_counts = [item[1] for item in type_items]
    sns.barplot(x=type_labels, y=type_counts, palette="Set2", ax=ax_left)
    ax_left.set_title("Test set paper_type")
    ax_left.set_xlabel("paper_type")
    ax_left.set_ylabel("Count")
    ax_left.tick_params(axis="x", labelrotation=45)
    annotate_bars(ax_left)

    color_items = sorted(color_counter.items(), key=lambda item: (-item[1], item[0]))
    color_labels = [item[0] for item in color_items]
    color_counts = [item[1] for item in color_items]
    sns.barplot(x=color_labels, y=color_counts, palette="viridis", ax=ax_right)
    ax_right.set_title("Test set page_color")
    ax_right.set_xlabel("page_color")
    ax_right.set_ylabel("Count")
    ax_right.tick_params(axis="x", labelrotation=45)
    annotate_bars(ax_right)

    sns.countplot(x=types, hue=colors, palette="tab10", ax=ax_bottom)
    ax_bottom.set_title("Test set page_color by paper_type")
    ax_bottom.set_xlabel("paper_type")
    ax_bottom.set_ylabel("Count")
    ax_bottom.tick_params(axis="x", labelrotation=45)
    ax_bottom.legend(title="page_color", fontsize=9, title_fontsize=10)
    annotate_bars(ax_bottom, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_age_histogram(ages: List[int], output_path: str) -> None:
    if not ages:
        return
    min_age = min(ages)
    max_age = max(ages)
    bin_width = 5
    start = (min_age // bin_width) * bin_width
    end = ((max_age // bin_width) + 1) * bin_width
    bins = list(range(start, end + 1, bin_width))

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(ages, bins=bins, kde=True, color="#F58518", edgecolor="black", ax=ax)
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    annotate_bars(ax, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_care_level_chart(levels: List[int], output_path: str) -> None:
    if not levels:
        return
    counter = Counter(levels)
    items = sorted(counter.items())
    labels = [str(item[0]) for item in items]
    counts = [item[1] for item in items]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=labels, y=counts, palette="Set2", ax=ax)
    ax.set_title("Care Level Counts")
    ax.set_xlabel("Care Level")
    ax.set_ylabel("Count")
    annotate_bars(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    sns.set_theme(style="whitegrid", context="talk")
    parser = argparse.ArgumentParser(description="Visualize synthetic data features.")
    parser.add_argument(
        "--scenarios",
        default="data/synthetic/scenarios.json",
        help="Path to scenarios.json (JSON array).",
    )
    parser.add_argument(
        "--labels-dir",
        default="data/synthetic/output/labels",
        help="Directory with label JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/synthetic/output/visualizations",
        help="Directory to save PNG outputs.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    scenarios = load_json_array(args.scenarios)
    ages = [item.get("age") for item in scenarios if isinstance(item.get("age"), int)]
    care_levels = [
        item.get("care_level")
        for item in scenarios
        if isinstance(item.get("care_level"), int)
    ]

    save_age_and_care_level(
        ages,
        care_levels,
        os.path.join(args.output_dir, "scenarios_age_care_level_counts.png"),
    )

    save_image_grid(
        image_dir="data/N30",
        output_dir=args.output_dir,
        base_name="n30_montage",
        images_per_page=12,
        cols=3,
    )

    save_n30_image_size_stats(
        image_dir="data/N30",
        output_dir=args.output_dir,
    )

    save_n30_label_distributions(
        image_dir="data/N30",
        labels_dir="data/synthetic/output/labels",
        output_path=os.path.join(args.output_dir, "n30_label_distributions.png"),
    )

    save_test_set_label_distributions(
        test_set_path="data/test_set_559.json",
        output_path=os.path.join(
            args.output_dir, "test_set_559_label_distributions.png"
        ),
    )

    # also generate 4-up (2x2) montages for N30 to fit compact pages
    save_image_grid(
        image_dir="data/N30",
        output_dir=args.output_dir,
        base_name="n30_4up",
        images_per_page=4,
        cols=2,
    )

    label_files = find_label_files(args.labels_dir)
    label_data = []
    for path in label_files:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                label_data.append(json.load(handle))
        except json.JSONDecodeError:
            continue

    combined_counters = {}
    raw_values = {}
    for field in LABEL_FIELDS:
        values = []
        for item in label_data:
            generation = item.get("generation_parameters")
            if isinstance(generation, dict) and field in generation:
                values.append(generation.get(field))
            else:
                values.append(item.get(field))
        raw_values[field] = values
        counter = count_field(values)
        if any(field in group[:2] for group in COMBINED_GROUPS):
            combined_counters[field] = counter
        else:
            output_name = f"labels_{field}_counts.png"
            save_bar_chart(
                counter,
                title=f"{field} Counts",
                xlabel=field,
                output_path=os.path.join(args.output_dir, output_name),
            )

    for left_field, right_field, output_name in COMBINED_GROUPS:
        if left_field in combined_counters and right_field in combined_counters:
            if {left_field, right_field} == {"page_color", "paper_type"}:
                save_page_color_by_paper_type(
                    raw_values.get("paper_type", []),
                    raw_values.get("page_color", []),
                    output_path=os.path.join(args.output_dir, output_name),
                )
            elif {left_field, right_field} == {"writing_style", "profile"}:
                save_writing_style_by_profile(
                    raw_values.get("writing_style", []),
                    raw_values.get("profile", []),
                    output_path=os.path.join(args.output_dir, output_name),
                )
            else:
                save_dual_bar_chart(
                    combined_counters[left_field],
                    combined_counters[right_field],
                    left_title=left_field,
                    right_title=right_field,
                    output_path=os.path.join(args.output_dir, output_name),
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
