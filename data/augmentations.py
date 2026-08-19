"""
Augmentation Pipeline for DocLens.

Applies realistic image degradations and forgery simulations to synthetic documents.
Uses albumentations for standard augmentations and PIL for tamper simulation.

Why augmentations matter (interview talking points):
    1. Real documents are photographed with phones — blur, rotation, lighting variation
       are the norm, not the exception
    2. Training on only clean data produces brittle models that fail on real inputs
    3. Augmentation acts as a form of regularization — prevents the model from
       memorizing exact pixel patterns
    4. The adversarial/tamper augmentations let us test robustness to deliberate
       manipulation, which is the core use case for identity verification

Design decisions:
    - Separate presets (light/medium/heavy/tamper) allow controlled evaluation:
      "which noise type hurts accuracy most?" — this is the failure analysis
      the JD explicitly asks for
    - Tamper simulation is programmatic, not augmentation-based: we alter the
      rendered text itself (different font/color/alignment) to simulate what
      a forger would do
"""

import json
import os
import random
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.synthetic_generator import ReceiptGenerator, ReceiptMetadata


# ─── Augmentation Presets ────────────────────────────────────────────────────────
# Each preset is a named albumentations pipeline with specific noise characteristics.
# This allows per-augmentation-type evaluation in the eval harness.

def get_augmentation_pipeline(preset: str = "medium") -> A.Compose:
    """Get an augmentation pipeline by preset name.

    Presets are designed to simulate real-world document capture conditions:
    - light: minor phone camera effects (slight blur, small rotation)
    - medium: typical bad-condition capture (blur, lighting, perspective)
    - heavy: worst-case conditions (strong blur, heavy occlusion, extreme angles)
    - blur_only: isolate blur effects for per-augmentation evaluation
    - rotation_only: isolate rotation effects
    - occlusion_only: isolate occlusion effects
    - lighting_only: isolate lighting effects
    - compression_only: isolate JPEG compression artifacts

    Args:
        preset: Name of the augmentation preset

    Returns:
        albumentations.Compose pipeline
    """
    def _rotate(limit, p):
        try:
            return A.Rotate(limit=limit, p=p, border_mode=0, fill=(245, 245, 245))
        except TypeError:
            return A.Rotate(limit=limit, p=p, border_mode=0, value=(245, 245, 245))

    def _compression(ql, qu, p):
        try:
            return A.ImageCompression(quality_range=(ql, qu), p=p)
        except TypeError:
            return A.ImageCompression(quality_lower=ql, quality_upper=qu, p=p)

    def _dropout(num_holes, max_h, max_w, min_h, min_w, fill, p):
        try:
            return A.CoarseDropout(
                num_holes_range=(1, num_holes),
                hole_height_range=(min_h, max_h),
                hole_width_range=(min_w, max_w),
                fill=fill, p=p,
            )
        except TypeError:
            return A.CoarseDropout(
                max_holes=num_holes, max_height=max_h, max_width=max_w,
                min_height=min_h, min_width=min_w,
                fill_value=fill, p=p,
            )

    def _gauss_noise(var_low, var_high, p):
        try:
            return A.GaussNoise(var_limit=(var_low, var_high), p=p)
        except TypeError:
            return A.GaussNoise(p=p)

    presets = {
        "light": A.Compose([
            _rotate(limit=5, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            _gauss_noise(5.0, 15.0, p=0.2),
        ]),

        "medium": A.Compose([
            _rotate(limit=10, p=0.6),
            A.Perspective(scale=(0.02, 0.05), p=0.4),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            _gauss_noise(10.0, 30.0, p=0.3),
            _compression(60, 90, p=0.3),
        ]),

        "heavy": A.Compose([
            _rotate(limit=15, p=0.8),
            A.Perspective(scale=(0.05, 0.10), p=0.6),
            A.OneOf([
                A.GaussianBlur(blur_limit=(5, 11), p=1.0),
                A.MotionBlur(blur_limit=(5, 11), p=1.0),
            ], p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.OneOf([
                A.RandomShadow(p=1.0),
                A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.2), p=1.0),
            ], p=0.4),
            _gauss_noise(20.0, 50.0, p=0.5),
            _compression(30, 60, p=0.5),
            _dropout(3, 80, 120, 30, 50, (200, 200, 200), p=0.4),
        ]),

        # ── Isolated augmentations for per-type evaluation ──
        "blur_only": A.Compose([
            A.GaussianBlur(blur_limit=(7, 13), p=1.0),
        ]),

        "rotation_only": A.Compose([
            _rotate(limit=15, p=1.0),
        ]),

        "occlusion_only": A.Compose([
            _dropout(4, 100, 150, 40, 60, (180, 180, 180), p=1.0),
        ]),

        "lighting_only": A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=1.0),
        ]),

        "compression_only": A.Compose([
            _compression(10, 30, p=1.0),
        ]),

        "none": A.Compose([]),  # No augmentation (for clean baseline)
    }

    if preset not in presets:
        raise ValueError(
            f"Unknown augmentation preset: '{preset}'. "
            f"Available: {list(presets.keys())}"
        )

    return presets[preset]


# ─── Tamper Simulation ───────────────────────────────────────────────────────────

class TamperSimulator:
    """Simulates document forgery by altering rendered fields.

    How document forgery works in the real world:
    1. Someone photographs a real receipt/ID
    2. They edit it in an image editor (change a name, amount, date)
    3. The edited region has subtle visual artifacts:
       - Different font than the surrounding text
       - Slightly different text color or brightness
       - Misaligned baseline (text not on the same line)
       - Different resolution/sharpness in the edited area

    This simulator reproduces those artifacts programmatically by re-rendering
    a specific field with altered visual properties.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._generator = ReceiptGenerator(seed=seed)

    def tamper_field(
        self,
        image: Image.Image,
        field_name: str,
        original_value: str,
        tampered_value: Optional[str] = None,
    ) -> tuple[Image.Image, str]:
        """Apply a tamper effect to a specific field region in the image.

        This is a simplified tamper simulation that overlays altered text
        in a region of the image. In a production system, you'd need to
        locate the exact field position — here we use approximate positioning
        since we control the rendering.

        Args:
            image: Original receipt image
            field_name: Name of the field to tamper
            original_value: Original field value (to locate approximate position)
            tampered_value: New value to overlay (if None, generates a random one)

        Returns:
            Tuple of (tampered_image, tampered_value_used)
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)

        # Generate a tampered value if not provided
        if tampered_value is None:
            if field_name == "total_amount":
                tampered_value = str(round(random.uniform(10, 9999), 2))
            elif field_name == "date":
                from faker import Faker
                fake = Faker()
                tampered_value = fake.date_between(start_date="-2y").strftime("%Y-%m-%d")
            elif field_name == "vendor_name":
                from data.synthetic_generator import VENDOR_NAMES
                tampered_value = random.choice(VENDOR_NAMES)
            else:
                tampered_value = f"TAMPERED_{random.randint(100, 999)}"

        # Simulate tamper artifacts:
        # 1. Use a DIFFERENT font than the original
        try:
            tamper_font = ImageFont.truetype("times.ttf", random.randint(13, 19))
        except (OSError, IOError):
            tamper_font = ImageFont.load_default()

        # 2. Use a slightly different color
        tamper_color = (
            random.randint(20, 80),
            random.randint(0, 30),
            random.randint(0, 30),
        )

        # 3. Find approximate position for the field and overlay
        # This is a heuristic — in production you'd use OCR bounding boxes
        field_positions = {
            "vendor_name": (img.width // 2 - 80, 25),
            "date": (45, 80),
            "total_amount": (img.width - 160, img.height // 2 + 100),
            "tax_amount": (img.width - 160, img.height // 2 + 70),
            "payment_method": (45, img.height - 120),
        }

        pos = field_positions.get(field_name, (100, 200))

        # 4. Draw a small background rectangle (simulates pasted-over region)
        text_bbox = draw.textbbox(pos, str(tampered_value), font=tamper_font)
        padding = 3
        bg_rect = [
            text_bbox[0] - padding,
            text_bbox[1] - padding,
            text_bbox[2] + padding,
            text_bbox[3] + padding,
        ]
        # Background slightly different shade than surroundings (artifact of editing)
        bg_color = (random.randint(238, 252), random.randint(238, 252), random.randint(238, 252))
        draw.rectangle(bg_rect, fill=bg_color)

        # 5. Draw the tampered text
        draw.text(pos, str(tampered_value), fill=tamper_color, font=tamper_font)

        return img, tampered_value


# ─── Augmented Dataset Builder ───────────────────────────────────────────────────

def augment_image(
    image_path: str,
    preset: str = "medium",
) -> tuple[np.ndarray, list[str]]:
    """Apply augmentation to a single image.

    Args:
        image_path: Path to the source image
        preset: Augmentation preset name

    Returns:
        Tuple of (augmented_image_array, list_of_augmentation_names)
    """
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    pipeline = get_augmentation_pipeline(preset)
    augmented = pipeline(image=img_array)

    return augmented["image"], [preset]


def create_augmented_dataset(
    source_dir: str,
    output_dir: str,
    preset: str = "medium",
    tamper_ratio: float = 0.0,
    seed: int = 42,
) -> list[ReceiptMetadata]:
    """Create an augmented version of a dataset.

    Takes clean images from source_dir, applies augmentations (and optionally
    tamper simulation), and saves to output_dir with updated metadata.

    Args:
        source_dir: Directory containing clean images + JSON metadata
        output_dir: Directory to save augmented images + updated metadata
        preset: Augmentation preset to apply
        tamper_ratio: Fraction of images to apply tamper simulation (0.0–1.0)
        seed: Random seed

    Returns:
        List of metadata for all augmented samples
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Find all JSON metadata files in source
    json_files = sorted(Path(source_dir).glob("*.json"))
    if not json_files:
        print(f"Warning: No JSON files found in {source_dir}")
        return []

    tamper_sim = TamperSimulator(seed=seed)
    results = []

    print(f"Augmenting {len(json_files)} images with preset '{preset}'...")

    for i, json_path in enumerate(json_files):
        # Load original metadata
        with open(json_path, "r") as f:
            orig_meta = json.load(f)

        image_path = orig_meta["image_path"]
        if not os.path.exists(image_path):
            # Try relative path from source_dir
            image_filename = os.path.basename(image_path)
            image_path = os.path.join(source_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"  Warning: Image not found: {image_path}, skipping")
            continue

        # Apply augmentation
        aug_array, aug_names = augment_image(image_path, preset)

        # Optionally apply tamper simulation
        is_tampered = False
        tamper_field_name = None

        if random.random() < tamper_ratio:
            # Convert back to PIL for tamper simulation
            aug_pil = Image.fromarray(aug_array)
            tamper_fields = ["total_amount", "vendor_name", "date"]
            tamper_field_name = random.choice(tamper_fields)
            original_value = str(orig_meta["fields"].get(tamper_field_name, ""))

            aug_pil, tampered_value = tamper_sim.tamper_field(
                aug_pil, tamper_field_name, original_value
            )
            aug_array = np.array(aug_pil)
            is_tampered = True
            aug_names.append(f"tampered_{tamper_field_name}")

        # Save augmented image
        out_filename = f"aug_{preset}_{i:04d}.png"
        out_path = os.path.join(output_dir, out_filename)
        Image.fromarray(aug_array).save(out_path, "PNG")

        # Create updated metadata
        metadata = ReceiptMetadata(
            image_path=out_path,
            document_type=orig_meta["document_type"],
            fields=orig_meta["fields"],
            is_tampered=is_tampered,
            tamper_field=tamper_field_name,
            augmentations_applied=aug_names,
        )

        # Save metadata JSON
        json_out_path = os.path.join(output_dir, f"aug_{preset}_{i:04d}.json")
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2)

        results.append(metadata)

        if (i + 1) % 50 == 0:
            print(f"  Augmented {i + 1}/{len(json_files)}")

    print(f"  Done. {len(results)} augmented images saved to {output_dir}")
    return results


# ─── CLI Entry Point ─────────────────────────────────────────────────────────────

def main():
    """Build the adversarial test set from clean test images."""
    import yaml

    config_path = os.path.join(os.path.dirname(__file__), "..", "training", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    base_dir = os.path.join(os.path.dirname(__file__), "generated")
    clean_test_dir = os.path.join(base_dir, "clean_test")
    adversarial_dir = os.path.join(base_dir, "adversarial_test")

    # Generate adversarial subsets for each augmentation type
    augmentation_subsets = [
        ("blur_only", 0.0),
        ("rotation_only", 0.0),
        ("occlusion_only", 0.0),
        ("lighting_only", 0.0),
        ("compression_only", 0.0),
        ("heavy", 0.3),  # 30% of heavy augmented images are also tampered
    ]

    for preset, tamper_ratio in augmentation_subsets:
        subset_dir = os.path.join(adversarial_dir, preset)
        create_augmented_dataset(
            source_dir=clean_test_dir,
            output_dir=subset_dir,
            preset=preset,
            tamper_ratio=tamper_ratio,
            seed=42,
        )

    print("\n✅ Adversarial test sets generated!")
    print(f"   Location: {os.path.abspath(adversarial_dir)}")


if __name__ == "__main__":
    main()
