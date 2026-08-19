"""
Synthetic Document Image Generator for DocLens.

Generates realistic receipt images with known ground truth for training and evaluation.
Uses PIL for rendering and Faker for realistic field values.

Why synthetic data instead of real documents (interview talking points):
    1. EXACT ground truth — no labeling errors, no annotation cost
    2. Controllable distribution — generate exactly the variations you want
    3. No PII — real receipts/IDs contain real identities (legal risk)
    4. Adversarial control — can systematically create tampered examples
    5. Scalable — generate 10,000 examples as easily as 100

Design decisions:
    - PIL-based rendering (not HTML-to-image) for simplicity and zero external deps
    - Faker for realistic but fake field values
    - Randomized layouts (fonts, sizes, positions) to prevent the model from
      memorizing pixel positions rather than learning to read text
"""

import json
import os
import random
import string
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from faker import Faker
from PIL import Image, ImageDraw, ImageFont


# ─── Data Classes ────────────────────────────────────────────────────────────────

@dataclass
class LineItem:
    """A single line item on a receipt."""
    item: str
    price: float


@dataclass
class ReceiptData:
    """Ground truth data for a synthetic receipt."""
    vendor_name: str
    date: str
    total_amount: float
    tax_amount: Optional[float]
    line_items: list[LineItem]
    payment_method: Optional[str]

    def to_dict(self) -> dict:
        """Convert to the exact JSON schema the model should produce."""
        return {
            "vendor_name": self.vendor_name,
            "date": self.date,
            "total_amount": self.total_amount,
            "tax_amount": self.tax_amount,
            "line_items": [{"item": li.item, "price": li.price} for li in self.line_items],
            "payment_method": self.payment_method,
        }

    def to_json(self) -> str:
        """Serialize to JSON string (training target)."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ReceiptMetadata:
    """Full metadata for a generated receipt (ground truth + generation info)."""
    image_path: str
    document_type: str = "receipt"
    fields: dict = field(default_factory=dict)
    is_tampered: bool = False
    tamper_field: Optional[str] = None
    augmentations_applied: list[str] = field(default_factory=list)
    render_config: dict = field(default_factory=dict)


# ─── Vendor Data ─────────────────────────────────────────────────────────────────
# Realistic vendor names and categories for diverse receipts

VENDOR_NAMES = [
    "Highland Coffee Co.", "Metro Supermart", "Stellar Electronics",
    "Green Leaf Pharmacy", "Urban Bites Restaurant", "TechZone Store",
    "Fresh & Fast Grocers", "BookWorm Bookstore", "Fuel Stop Gas Station",
    "Lotus Garden Chinese", "Pizza Planet Express", "CityMart Convenience",
    "Sunrise Bakery", "Neptune Hardware", "Cloud Nine Cafe",
    "QuickServe Diner", "Royal Stationery", "MegaMall Shopping Center",
    "Golden Spoon Restaurant", "Evergreen Market", "Byte Size Tech",
    "Harmony Health Foods", "Peak Performance Sports", "Silver Screen Cinema",
    "Maple Leaf Deli", "Precision Auto Parts", "Zenith Office Supplies",
    "Crimson Grill House", "Oasis Juice Bar", "Summit Travel Agency",
]

ITEM_CATEGORIES = {
    "grocery": [
        "Milk 1L", "Bread Loaf", "Eggs (12)", "Butter 200g", "Rice 1kg",
        "Pasta 500g", "Tomato Sauce", "Olive Oil 500ml", "Cheese Block",
        "Yogurt Pack", "Chicken Breast 500g", "Bananas 1kg", "Apples 1kg",
        "Orange Juice 1L", "Cereal Box", "Coffee 250g", "Tea Bags (50)",
        "Sugar 1kg", "Salt 500g", "Flour 1kg",
    ],
    "electronics": [
        "USB-C Cable", "Phone Case", "Screen Protector", "Earbuds",
        "Mouse Pad", "USB Drive 32GB", "HDMI Cable", "Power Bank",
        "Keyboard", "Webcam", "Charging Adapter", "Cable Organizer",
    ],
    "restaurant": [
        "Espresso", "Cappuccino", "Latte", "Americano", "Green Tea",
        "Sandwich", "Burger", "Salad", "Soup of the Day", "Pasta Bowl",
        "Grilled Chicken", "Fish & Chips", "Dessert", "Mineral Water",
        "Fresh Juice", "Iced Tea",
    ],
    "pharmacy": [
        "Paracetamol 20s", "Vitamin C 30s", "Band-Aid Box", "Hand Sanitizer",
        "Face Mask (10)", "Cough Syrup", "Eye Drops", "Antiseptic Cream",
        "Multivitamin 30s", "Tissues Box",
    ],
}

PAYMENT_METHODS = ["Cash", "Credit Card", "Debit Card", "UPI", "Mobile Payment", None]


# ─── Receipt Generator ──────────────────────────────────────────────────────────

class ReceiptGenerator:
    """Generates synthetic receipt images with ground truth.

    How it works:
    1. Generate random but realistic field values using Faker + curated lists
    2. Render the receipt as an image using PIL
    3. Save image + ground truth JSON side by side

    The rendering intentionally varies layout parameters (font, size, spacing,
    position) so the model learns to read text content rather than memorize
    pixel positions.
    """

    def __init__(self, seed: int = 42, image_width: int = 600, image_height: int = 900):
        """Initialize the generator.

        Args:
            seed: Random seed for reproducibility
            image_width: Width of generated images in pixels
            image_height: Height of generated images in pixels
        """
        self.faker = Faker()
        Faker.seed(seed)
        random.seed(seed)

        self.image_width = image_width
        self.image_height = image_height

        # Try to load a variety of fonts; fall back to PIL default
        self._fonts = self._load_fonts()

    def _load_fonts(self) -> dict[str, str]:
        """Discover available fonts on the system.

        Returns a dict mapping descriptive names to font paths.
        Falls back to PIL default if no system fonts found.
        """
        font_dirs = []

        # Windows
        win_font_dir = Path("C:/Windows/Fonts")
        if win_font_dir.exists():
            font_dirs.append(win_font_dir)

        # Linux
        for linux_dir in [Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]:
            if linux_dir.exists():
                font_dirs.append(linux_dir)

        # macOS
        mac_font_dir = Path("/System/Library/Fonts")
        if mac_font_dir.exists():
            font_dirs.append(mac_font_dir)

        fonts = {}
        target_fonts = {
            "arial": ["arial.ttf", "Arial.ttf"],
            "courier": ["cour.ttf", "Courier.ttf", "courier.ttf"],
            "times": ["times.ttf", "Times.ttf", "timesbd.ttf"],
            "verdana": ["verdana.ttf", "Verdana.ttf"],
            "georgia": ["georgia.ttf", "Georgia.ttf"],
            "consolas": ["consola.ttf", "Consolas.ttf"],
            "trebuchet": ["trebuc.ttf"],
            "calibri": ["calibri.ttf", "Calibri.ttf"],
        }

        for font_dir in font_dirs:
            for name, filenames in target_fonts.items():
                if name in fonts:
                    continue
                for filename in filenames:
                    path = font_dir / filename
                    if path.exists():
                        fonts[name] = str(path)
                        break

        if not fonts:
            fonts["default"] = None  # Will use PIL default

        return fonts

    def _get_font(self, size: int, style: str = "normal") -> ImageFont.FreeTypeFont:
        """Get a font at the specified size.

        Args:
            size: Font size in points
            style: "normal", "bold", or "mono"
        """
        if style == "mono":
            preferred = ["consolas", "courier"]
        elif style == "bold":
            preferred = ["arial", "verdana", "calibri"]
        else:
            preferred = list(self._fonts.keys())

        for name in preferred:
            if name in self._fonts and self._fonts[name] is not None:
                try:
                    return ImageFont.truetype(self._fonts[name], size)
                except (OSError, IOError):
                    continue

        # Fallback: PIL default font (very basic but always available)
        try:
            return ImageFont.truetype("arial.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()

    def generate_receipt_data(self) -> ReceiptData:
        """Generate random but realistic receipt field values.

        Returns:
            ReceiptData with all fields populated
        """
        # Pick a vendor
        vendor_name = random.choice(VENDOR_NAMES)

        # Generate a date in the recent past
        date = self.faker.date_between(start_date="-2y", end_date="today")
        date_str = date.strftime("%Y-%m-%d")

        # Pick a category and generate line items
        category = random.choice(list(ITEM_CATEGORIES.keys()))
        num_items = random.randint(2, 8)
        items = random.sample(
            ITEM_CATEGORIES[category],
            min(num_items, len(ITEM_CATEGORIES[category])),
        )

        line_items = []
        subtotal = 0.0
        for item_name in items:
            # Generate a realistic price based on category
            if category == "grocery":
                price = round(random.uniform(0.50, 25.00), 2)
            elif category == "electronics":
                price = round(random.uniform(5.00, 150.00), 2)
            elif category == "restaurant":
                price = round(random.uniform(2.00, 35.00), 2)
            else:  # pharmacy
                price = round(random.uniform(1.00, 50.00), 2)
            line_items.append(LineItem(item=item_name, price=price))
            subtotal += price

        # Tax (some receipts show it, some don't)
        has_tax = random.random() > 0.2  # 80% chance of showing tax
        if has_tax:
            tax_rate = random.choice([0.05, 0.08, 0.10, 0.12, 0.18])
            tax_amount = round(subtotal * tax_rate, 2)
        else:
            tax_amount = None

        total_amount = round(subtotal + (tax_amount or 0), 2)
        payment_method = random.choice(PAYMENT_METHODS)

        return ReceiptData(
            vendor_name=vendor_name,
            date=date_str,
            total_amount=total_amount,
            tax_amount=tax_amount,
            line_items=line_items,
            payment_method=payment_method,
        )

    def render_receipt_image(
        self,
        receipt: ReceiptData,
        randomize_layout: bool = True,
    ) -> Image.Image:
        """Render a receipt as a PIL image.

        This creates a receipt-style layout with:
        - Vendor name (large, centered, top)
        - Date
        - Line items with prices (tabular)
        - Subtotal / tax / total section
        - Payment method (bottom)

        Layout parameters (font size, spacing, margins) are randomized when
        randomize_layout=True to ensure the model learns text reading, not
        position memorization.

        Args:
            receipt: The receipt data to render
            randomize_layout: Whether to vary fonts, sizes, spacing

        Returns:
            PIL Image of the rendered receipt
        """
        # Randomize layout parameters
        if randomize_layout:
            bg_shade = random.randint(240, 255)  # Near-white background
            margin_x = random.randint(30, 60)
            margin_y = random.randint(20, 40)
            title_size = random.randint(22, 30)
            header_size = random.randint(14, 18)
            body_size = random.randint(12, 16)
            line_spacing = random.randint(4, 10)
        else:
            bg_shade = 250
            margin_x = 40
            margin_y = 30
            title_size = 26
            header_size = 16
            body_size = 14
            line_spacing = 6

        # Create image
        bg_color = (bg_shade, bg_shade, bg_shade)
        img = Image.new("RGB", (self.image_width, self.image_height), bg_color)
        draw = ImageDraw.Draw(img)

        # Load fonts
        title_font = self._get_font(title_size, "bold")
        header_font = self._get_font(header_size, "normal")
        body_font = self._get_font(body_size, "normal")
        mono_font = self._get_font(body_size, "mono")

        text_color = (random.randint(0, 40), random.randint(0, 40), random.randint(0, 40))
        line_color = (random.randint(150, 200), random.randint(150, 200), random.randint(150, 200))

        y = margin_y

        # ── Vendor Name (centered) ──
        vendor_bbox = draw.textbbox((0, 0), receipt.vendor_name, font=title_font)
        vendor_width = vendor_bbox[2] - vendor_bbox[0]
        vendor_x = (self.image_width - vendor_width) // 2
        draw.text((vendor_x, y), receipt.vendor_name, fill=text_color, font=title_font)
        y += (vendor_bbox[3] - vendor_bbox[1]) + line_spacing * 2

        # ── Separator line ──
        draw.line([(margin_x, y), (self.image_width - margin_x, y)], fill=line_color, width=1)
        y += line_spacing * 2

        # ── Date ──
        date_display = random.choice([
            f"Date: {receipt.date}",
            f"DATE: {receipt.date}",
            f"Dt: {receipt.date}",
            receipt.date,
        ])
        draw.text((margin_x, y), date_display, fill=text_color, font=header_font)
        y += header_size + line_spacing

        # ── Receipt number (decorative — not a target field) ──
        receipt_num = "".join(random.choices(string.digits, k=random.randint(6, 10)))
        receipt_label = random.choice(["Receipt #:", "Bill No:", "Inv #:", "Ref:"])
        draw.text((margin_x, y), f"{receipt_label} {receipt_num}", fill=text_color, font=body_font)
        y += body_size + line_spacing * 2

        # ── Separator ──
        draw.line([(margin_x, y), (self.image_width - margin_x, y)], fill=line_color, width=1)
        y += line_spacing

        # ── Column headers ──
        item_header = random.choice(["ITEM", "Item", "Description", "DESCRIPTION"])
        price_header = random.choice(["PRICE", "Price", "Amount", "AMT"])
        draw.text((margin_x, y), item_header, fill=text_color, font=header_font)
        price_x = self.image_width - margin_x - 80
        draw.text((price_x, y), price_header, fill=text_color, font=header_font)
        y += header_size + line_spacing

        # ── Thin separator ──
        dash_str = "-" * 50
        draw.text((margin_x, y), dash_str, fill=line_color, font=body_font)
        y += body_size + line_spacing

        # ── Line Items ──
        for li in receipt.line_items:
            # Item name (left-aligned)
            draw.text((margin_x + 5, y), li.item, fill=text_color, font=body_font)
            # Price (right-aligned)
            price_str = f"{li.price:.2f}"
            price_bbox = draw.textbbox((0, 0), price_str, font=mono_font)
            price_w = price_bbox[2] - price_bbox[0]
            draw.text(
                (self.image_width - margin_x - price_w - 10, y),
                price_str,
                fill=text_color,
                font=mono_font,
            )
            y += body_size + line_spacing

        y += line_spacing

        # ── Separator before totals ──
        draw.line([(margin_x, y), (self.image_width - margin_x, y)], fill=line_color, width=1)
        y += line_spacing * 2

        # ── Subtotal ──
        subtotal = sum(li.price for li in receipt.line_items)
        subtotal_label = random.choice(["Subtotal:", "Sub Total:", "SUBTOTAL:"])
        draw.text((margin_x, y), subtotal_label, fill=text_color, font=header_font)
        subtotal_str = f"{subtotal:.2f}"
        sb_bbox = draw.textbbox((0, 0), subtotal_str, font=mono_font)
        sb_w = sb_bbox[2] - sb_bbox[0]
        draw.text(
            (self.image_width - margin_x - sb_w - 10, y),
            subtotal_str, fill=text_color, font=mono_font,
        )
        y += header_size + line_spacing

        # ── Tax ──
        if receipt.tax_amount is not None:
            tax_label = random.choice(["Tax:", "TAX:", "GST:", "VAT:"])
            draw.text((margin_x, y), tax_label, fill=text_color, font=header_font)
            tax_str = f"{receipt.tax_amount:.2f}"
            tax_bbox = draw.textbbox((0, 0), tax_str, font=mono_font)
            tax_w = tax_bbox[2] - tax_bbox[0]
            draw.text(
                (self.image_width - margin_x - tax_w - 10, y),
                tax_str, fill=text_color, font=mono_font,
            )
            y += header_size + line_spacing

        # ── Total (prominent) ──
        y += line_spacing
        total_font = self._get_font(title_size - 2, "bold")
        total_label = random.choice(["TOTAL:", "Total:", "GRAND TOTAL:", "Amount Due:"])
        draw.text((margin_x, y), total_label, fill=text_color, font=total_font)
        total_str = f"{receipt.total_amount:.2f}"
        total_bbox = draw.textbbox((0, 0), total_str, font=total_font)
        total_w = total_bbox[2] - total_bbox[0]
        draw.text(
            (self.image_width - margin_x - total_w - 10, y),
            total_str, fill=text_color, font=total_font,
        )
        y += title_size + line_spacing * 2

        # ── Separator ──
        draw.line([(margin_x, y), (self.image_width - margin_x, y)], fill=line_color, width=2)
        y += line_spacing * 2

        # ── Payment Method ──
        if receipt.payment_method:
            pay_label = random.choice(["Paid by:", "Payment:", "Method:", "PAYMENT:"])
            draw.text(
                (margin_x, y),
                f"{pay_label} {receipt.payment_method}",
                fill=text_color, font=body_font,
            )
            y += body_size + line_spacing * 2

        # ── Footer (decorative) ──
        footer_texts = [
            "Thank you for your purchase!",
            "Thank you! Visit again.",
            "Thanks for shopping with us!",
            "Have a great day!",
            "Thank you for your business.",
        ]
        footer = random.choice(footer_texts)
        footer_bbox = draw.textbbox((0, 0), footer, font=body_font)
        footer_w = footer_bbox[2] - footer_bbox[0]
        footer_x = (self.image_width - footer_w) // 2
        draw.text((footer_x, y), footer, fill=text_color, font=body_font)

        return img

    def generate_single(
        self,
        output_dir: str,
        index: int,
        randomize_layout: bool = True,
    ) -> ReceiptMetadata:
        """Generate a single receipt image + ground truth.

        Args:
            output_dir: Directory to save image and JSON
            index: Index number for filename
            randomize_layout: Whether to vary layout parameters

        Returns:
            ReceiptMetadata with all generation details
        """
        os.makedirs(output_dir, exist_ok=True)

        # Generate data
        receipt_data = self.generate_receipt_data()

        # Render image
        img = self.render_receipt_image(receipt_data, randomize_layout=randomize_layout)

        # Save image
        image_filename = f"receipt_{index:04d}.png"
        image_path = os.path.join(output_dir, image_filename)
        img.save(image_path, "PNG")

        # Save ground truth JSON
        json_filename = f"receipt_{index:04d}.json"
        json_path = os.path.join(output_dir, json_filename)

        metadata = ReceiptMetadata(
            image_path=image_path,
            document_type="receipt",
            fields=receipt_data.to_dict(),
            is_tampered=False,
            tamper_field=None,
            augmentations_applied=[],
            render_config={
                "randomize_layout": randomize_layout,
                "image_width": self.image_width,
                "image_height": self.image_height,
            },
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2)

        return metadata

    def generate_dataset(
        self,
        output_dir: str,
        num_samples: int,
        start_index: int = 0,
        randomize_layout: bool = True,
    ) -> list[ReceiptMetadata]:
        """Generate a full dataset of receipt images with ground truth.

        Args:
            output_dir: Root directory for the dataset split
            num_samples: Number of samples to generate
            start_index: Starting index for filenames
            randomize_layout: Whether to vary layout parameters

        Returns:
            List of ReceiptMetadata for all generated samples
        """
        print(f"Generating {num_samples} receipt images in {output_dir}...")
        results = []

        for i in range(num_samples):
            idx = start_index + i
            metadata = self.generate_single(output_dir, idx, randomize_layout)
            results.append(metadata)

            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{num_samples}")

        print(f"  Done. {num_samples} images saved to {output_dir}")
        return results


# ─── CLI Entry Point ─────────────────────────────────────────────────────────────

def main():
    """Generate the full synthetic dataset (all splits)."""
    import yaml

    config_path = os.path.join(os.path.dirname(__file__), "..", "training", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        gen_config = config.get("generation", {})
    else:
        gen_config = {
            "num_train": 600,
            "num_val": 100,
            "num_clean_test": 100,
            "num_adversarial_test": 80,
            "image_width": 600,
            "image_height": 900,
            "seed": 42,
        }

    generator = ReceiptGenerator(
        seed=gen_config.get("seed", 42),
        image_width=gen_config.get("image_width", 600),
        image_height=gen_config.get("image_height", 900),
    )

    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "generated")

    # Generate each split
    splits = [
        ("train", gen_config.get("num_train", 600)),
        ("val", gen_config.get("num_val", 100)),
        ("clean_test", gen_config.get("num_clean_test", 100)),
    ]

    offset = 0
    for split_name, num in splits:
        split_dir = os.path.join(base_dir, split_name)
        generator.generate_dataset(split_dir, num, start_index=offset)
        offset += num

    print("\n[OK] All dataset splits generated successfully!")
    print(f"   Location: {os.path.abspath(base_dir)}")


if __name__ == "__main__":
    main()
