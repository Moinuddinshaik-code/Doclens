"""
Evaluation Harness — Builds clean and adversarial test sets.

Orchestrates the creation of adversarial test subsets from clean test images.
Each subset isolates a specific augmentation type so the evaluator can report
per-augmentation accuracy breakdown.

This is the "build evaluation harnesses against noisy, adversarial, and
real-world inputs" component that the JD explicitly asks for.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.augmentations import create_augmented_dataset


def build_adversarial_test_sets(
    clean_test_dir: str = "data/generated/clean_test",
    adversarial_base_dir: str = "data/generated/adversarial_test",
    seed: int = 42,
):
    """Build all adversarial test subsets from clean test images.

    Creates one subdirectory per augmentation type, each containing
    augmented versions of the clean test images with metadata updated
    to reflect the augmentations applied.

    The result is a directory structure like:
        adversarial_test/
        ├── blur_only/          # Gaussian blur (σ=7-13)
        ├── rotation_only/      # Rotation ±15°
        ├── occlusion_only/     # Random rectangular occlusion
        ├── lighting_only/      # Brightness/contrast shifts
        ├── compression_only/   # JPEG compression (q=10-30)
        └── heavy/              # All augmentations + 30% tampered

    Args:
        clean_test_dir: Directory containing clean test images
        adversarial_base_dir: Base directory for adversarial subsets
        seed: Random seed for reproducibility
    """
    if not os.path.exists(clean_test_dir):
        print(f"ERROR: Clean test directory not found: {clean_test_dir}")
        print("Generate data first: python data/synthetic_generator.py")
        return

    print("=" * 60)
    print("   Building Adversarial Test Sets")
    print("=" * 60)

    # Define adversarial subsets
    # Format: (preset_name, tamper_ratio)
    # tamper_ratio: fraction of images to also apply tamper simulation
    subsets = [
        ("blur_only", 0.0),       # Pure blur — tests text readability
        ("rotation_only", 0.0),   # Pure rotation — tests spatial understanding
        ("occlusion_only", 0.0),  # Pure occlusion — tests partial visibility
        ("lighting_only", 0.0),   # Pure lighting — tests brightness robustness
        ("compression_only", 0.0), # Pure JPEG artifacts — tests detail preservation
        ("heavy", 0.3),           # All augmentations + 30% tampered
    ]

    for preset, tamper_ratio in subsets:
        subset_dir = os.path.join(adversarial_base_dir, preset)
        print(f"\n-- Creating '{preset}' subset (tamper_ratio={tamper_ratio}) --")

        create_augmented_dataset(
            source_dir=clean_test_dir,
            output_dir=subset_dir,
            preset=preset,
            tamper_ratio=tamper_ratio,
            seed=seed,
        )

    # Also create a mixed medium subset for general adversarial testing
    mixed_dir = os.path.join(adversarial_base_dir, "medium_mixed")
    print(f"\n-- Creating 'medium_mixed' subset --")
    create_augmented_dataset(
        source_dir=clean_test_dir,
        output_dir=mixed_dir,
        preset="medium",
        tamper_ratio=0.15,
        seed=seed,
    )

    print("\n" + "=" * 60)
    print("[OK] All adversarial test sets generated!")
    print(f"   Location: {os.path.abspath(adversarial_base_dir)}")
    print("   Subsets created:")
    for preset, tamper_ratio in subsets:
        print(f"     {preset}/ (tamper: {tamper_ratio*100:.0f}%)")
    print(f"     medium_mixed/ (tamper: 15%)")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build adversarial test sets")
    parser.add_argument(
        "--clean-dir",
        default="data/generated/clean_test",
        help="Path to clean test directory",
    )
    parser.add_argument(
        "--output-dir",
        default="data/generated/adversarial_test",
        help="Base directory for adversarial subsets",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    build_adversarial_test_sets(args.clean_dir, args.output_dir, args.seed)
