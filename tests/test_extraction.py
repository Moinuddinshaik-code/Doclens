"""
Tests for the extraction pipeline.

Tests the extractor's preprocessing, confidence scoring, and overall pipeline
using mock components (no actual GPU/model required).
"""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from doclens.extractor import DocLensExtractor


class TestImagePreprocessing:
    """Tests for image preprocessing."""

    def setup_method(self):
        self.extractor = DocLensExtractor(
            model=MagicMock(),
            processor=MagicMock(),
        )

    def test_load_from_path(self, tmp_path):
        # Create a test image
        img = Image.new("RGB", (100, 100), "red")
        path = tmp_path / "test.png"
        img.save(str(path))

        result = self.extractor._preprocess_image(str(path))
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_load_from_pil(self):
        img = Image.new("RGBA", (100, 100), "blue")
        result = self.extractor._preprocess_image(img)
        assert result.mode == "RGB"  # Converted from RGBA

    def test_load_from_bytes(self):
        import io
        img = Image.new("RGB", (100, 100), "green")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        result = self.extractor._preprocess_image(buffer.read())
        assert isinstance(result, Image.Image)

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            self.extractor._preprocess_image(12345)


class TestConfidenceScoring:
    """Tests for per-field confidence computation."""

    def setup_method(self):
        self.extractor = DocLensExtractor(
            model=MagicMock(),
            processor=MagicMock(),
        )

    def test_all_fields_present(self):
        parsed = {
            "vendor_name": "Test Store",
            "date": "2024-01-15",
            "total_amount": 42.50,
            "tax_amount": 3.50,
            "line_items": [{"item": "Coffee", "price": 5.0}],
            "payment_method": "Cash",
        }
        confidence = self.extractor._compute_confidence(parsed, "receipt")
        for field, score in confidence.items():
            assert score > 0.0, f"Expected positive confidence for {field}"

    def test_all_fields_null(self):
        parsed = {
            "vendor_name": None,
            "date": None,
            "total_amount": None,
        }
        confidence = self.extractor._compute_confidence(parsed, "receipt")
        for field in ["vendor_name", "date", "total_amount"]:
            assert confidence[field] == 0.0

    def test_none_parsed(self):
        confidence = self.extractor._compute_confidence(None, "receipt")
        for field, score in confidence.items():
            assert score == 0.0

    def test_correct_types_get_bonus(self):
        parsed_good = {"total_amount": 42.50}
        parsed_bad = {"total_amount": "forty two"}

        conf_good = self.extractor._compute_confidence(parsed_good, "receipt")
        conf_bad = self.extractor._compute_confidence(parsed_bad, "receipt")

        assert conf_good["total_amount"] >= conf_bad["total_amount"]
