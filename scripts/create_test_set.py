"""
Create a stratified test set for fair OCR evaluation.

This script ensures balanced representation across fonts, scenarios,
and other parameters for fair comparison between OCR methods.

Usage:
    # Create 1 sample per font-scenario combination (1200 samples)
    python scripts/create_test_set.py --strategy complete --output data/test_set.json

    # Create balanced 600 sample subset
    python scripts/create_test_set.py --strategy balanced-600 --output data/test_set.json

    # Create stratified 500 sample subset
    python scripts/create_test_set.py --strategy stratified-500 --output data/test_set.json --seed 42
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import random


def load_all_labels(labels_dir):
    """Load all label files and extract metadata."""
    labels_dir = Path(labels_dir)
    samples = []

    for label_file in sorted(labels_dir.glob("*.json")):
        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        sample_id = label_file.stem
        gen_params = data.get("generation_parameters", {})

        # Extract font name from path (remove extension)
        font_path = gen_params.get("font_path", "unknown")
        font_name = Path(font_path).stem if font_path != "unknown" else "unknown"

        samples.append(
            {
                "sample_id": sample_id,
                "image_path": f"data/synthetic/output/images/{sample_id}.png",
                "label_path": str(label_file),
                "font": font_name,
                "scenario": data.get("patient", "unknown"),
                "writing_style": gen_params.get("writing_style", "normal"),
                "paper_type": gen_params.get("paper_type", "unknown"),
                "page_color": gen_params.get("page_color", "unknown"),
                "profile": gen_params.get("profile", "unknown"),
                "text_length": len(data.get("full_text", "")),
            }
        )

    print(f"✓ Loaded {len(samples)} label files")
    return samples


def create_complete_test_set(samples):
    """
    Strategy: 1 sample per font-scenario combination.
    Ensures complete coverage of all combinations.
    """
    print("\n📊 Creating COMPLETE test set (1 per font-scenario)...")

    # Group by font-scenario combination
    combinations = defaultdict(list)
    for sample in samples:
        key = (sample["font"], sample["scenario"])
        combinations[key].append(sample)

    # Select one sample per combination (prefer first occurrence)
    test_set = []
    for key, group in sorted(combinations.items()):
        # Take the first sample from each group
        test_set.append(group[0])

    print(f"✓ Selected {len(test_set)} samples")
    print(f"  Unique fonts: {len(set(s['font'] for s in test_set))}")
    print(f"  Unique scenarios: {len(set(s['scenario'] for s in test_set))}")

    return test_set


def create_balanced_600_test_set(samples, seed=42):
    """
    Strategy: Select 25 scenarios, all 24 fonts = 600 samples.
    Balanced selection of scenarios (every other one).
    """
    print("\n📊 Creating BALANCED-600 test set...")
    random.seed(seed)

    # Group by font-scenario
    combinations = defaultdict(list)
    for sample in samples:
        key = (sample["font"], sample["scenario"])
        combinations[key].append(sample)

    # Get all unique scenarios and select 25 evenly distributed
    all_scenarios = sorted(set(s["scenario"] for s in samples))
    print(f"  Total scenarios available: {len(all_scenarios)}")

    # Select every other scenario (or adjust step to get ~25)
    step = len(all_scenarios) // 25
    selected_scenarios = all_scenarios[::step][:25]
    print(f"  Selected scenarios: {len(selected_scenarios)}")

    # Get all fonts
    all_fonts = sorted(set(s["font"] for s in samples))
    print(f"  Fonts: {len(all_fonts)}")

    # Select samples
    test_set = []
    for font in all_fonts:
        for scenario in selected_scenarios:
            key = (font, scenario)
            if key in combinations and combinations[key]:
                test_set.append(combinations[key][0])

    print(f"✓ Selected {len(test_set)} samples")
    print(f"  Samples per font: ~{len(test_set) / len(all_fonts):.1f}")

    return test_set


def create_stratified_500_test_set(samples, seed=42):
    """
    Strategy: Stratified sampling ensuring even distribution across fonts.
    Target: 500 samples with ~20-21 samples per font.
    """
    print("\n📊 Creating STRATIFIED-500 test set...")
    random.seed(seed)

    # Group by font
    font_groups = defaultdict(list)
    for sample in samples:
        font_groups[sample["font"]].append(sample)

    n_fonts = len(font_groups)
    samples_per_font = 500 // n_fonts
    remainder = 500 % n_fonts

    print(f"  Fonts: {n_fonts}")
    print(f"  Base samples per font: {samples_per_font}")
    print(f"  Extra samples to distribute: {remainder}")

    # Select samples from each font
    test_set = []
    font_names = sorted(font_groups.keys())

    for i, font in enumerate(font_names):
        # Some fonts get one extra sample to reach exactly 500
        n_samples = samples_per_font + (1 if i < remainder else 0)

        # Randomly sample from this font's samples
        available = font_groups[font]
        if len(available) >= n_samples:
            selected = random.sample(available, n_samples)
        else:
            # If not enough, take all available
            selected = available
            print(
                f"  ⚠ Font '{font}' only has {len(available)} samples (needed {n_samples})"
            )

        test_set.extend(selected)

    print(f"✓ Selected {len(test_set)} samples")

    # Print distribution
    font_counts = defaultdict(int)
    for sample in test_set:
        font_counts[sample["font"]] += 1

    print(
        f"  Samples per font range: {min(font_counts.values())} - {max(font_counts.values())}"
    )

    return test_set


def print_test_set_statistics(test_set):
    """Print detailed statistics about the test set."""
    print("\n" + "=" * 80)
    print("TEST SET STATISTICS")
    print("=" * 80)

    # Count distributions
    font_dist = defaultdict(int)
    scenario_dist = defaultdict(int)
    style_dist = defaultdict(int)
    paper_dist = defaultdict(int)

    for sample in test_set:
        font_dist[sample["font"]] += 1
        scenario_dist[sample["scenario"]] += 1
        style_dist[sample["writing_style"]] += 1
        paper_dist[sample["paper_type"]] += 1

    print(f"\nTotal samples: {len(test_set)}")
    print(f"\nUnique fonts: {len(font_dist)}")
    print(f"  Samples per font: {min(font_dist.values())} - {max(font_dist.values())}")

    print(f"\nUnique scenarios: {len(scenario_dist)}")
    print(
        f"  Samples per scenario: {min(scenario_dist.values())} - {max(scenario_dist.values())}"
    )

    print(f"\nWriting styles:")
    for style, count in sorted(style_dist.items()):
        print(f"  {style}: {count} ({count/len(test_set)*100:.1f}%)")

    print(f"\nPaper types:")
    for paper, count in sorted(paper_dist.items()):
        print(f"  {paper}: {count} ({count/len(test_set)*100:.1f}%)")

    # Text length statistics
    text_lengths = [s["text_length"] for s in test_set]
    print(f"\nText length:")
    print(f"  Min: {min(text_lengths)}")
    print(f"  Max: {max(text_lengths)}")
    print(f"  Mean: {sum(text_lengths)/len(text_lengths):.0f}")

    print("=" * 80)


def save_test_set(test_set, output_path):
    """Save test set to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create output with just the necessary info
    output_data = {
        "test_set_size": len(test_set),
        "samples": [
            {
                "sample_id": s["sample_id"],
                "image_path": s["image_path"],
                "label_path": s["label_path"],
                "font": s["font"],
                "scenario": s["scenario"],
            }
            for s in test_set
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Test set saved to: {output_path}")

    # Also save a simple list of sample IDs for easy use
    ids_path = output_path.with_suffix(".txt")
    with open(ids_path, "w") as f:
        for sample in test_set:
            f.write(f"{sample['sample_id']}\n")

    print(f"✓ Sample IDs saved to: {ids_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified test set for OCR evaluation"
    )
    parser.add_argument(
        "--strategy",
        choices=["complete", "balanced-600", "stratified-500"],
        default="complete",
        help="Test set creation strategy",
    )
    parser.add_argument(
        "--labels-dir",
        default="data/synthetic/output/labels",
        help="Directory containing label JSON files",
    )
    parser.add_argument(
        "--output", default="data/test_set.json", help="Output JSON file for test set"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CREATING STRATIFIED TEST SET FOR OCR EVALUATION")
    print("=" * 80)

    # Load all samples
    samples = load_all_labels(args.labels_dir)

    if not samples:
        print("❌ No samples found!")
        return

    # Create test set based on strategy
    if args.strategy == "complete":
        test_set = create_complete_test_set(samples)
    elif args.strategy == "balanced-600":
        test_set = create_balanced_600_test_set(samples, seed=args.seed)
    elif args.strategy == "stratified-500":
        test_set = create_stratified_500_test_set(samples, seed=args.seed)
    else:
        print(f"❌ Unknown strategy: {args.strategy}")
        return

    # Print statistics
    print_test_set_statistics(test_set)

    # Save
    save_test_set(test_set, args.output)

    print("\n" + "=" * 80)
    print("✅ TEST SET CREATED SUCCESSFULLY")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Use this test set for ALL OCR methods to ensure fair comparison")
    print("2. Update your notebooks to load from this test set")
    print(
        f"3. Run: python scripts/evaluate_ocr_on_test_set.py --test-set {args.output}"
    )


if __name__ == "__main__":
    main()
