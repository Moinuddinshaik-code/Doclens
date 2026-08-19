"""
Tests for evaluation metrics computation.
"""

import pytest
from doclens.evaluator import DocLensEvaluator


class MockExtractor:
    """Mock extractor for testing the evaluator without a real model."""

    def __init__(self, responses: dict = None):
        """
        Args:
            responses: Dict mapping image paths to mock extraction results
        """
        self.responses = responses or {}

    def extract(self, image_path, document_type="receipt"):
        if image_path in self.responses:
            return self.responses[image_path]
        # Default: return a valid but empty extraction
        return {
            "fields": {
                "vendor_name": None,
                "date": None,
                "total_amount": None,
                "tax_amount": None,
                "line_items": None,
                "payment_method": None,
            },
            "confidence": {},
            "raw_output": "{}",
            "is_valid": True,
            "validation_errors": [],
            "latency_ms": 100,
        }


class TestFieldComparison:
    """Tests for field-level comparison logic."""

    def setup_method(self):
        self.evaluator = DocLensEvaluator(MockExtractor(), fuzzy_threshold=3)

    def test_exact_string_match(self):
        result = self.evaluator._compare_field("Hello", "Hello", "vendor_name")
        assert result["exact_match"] is True
        assert result["fuzzy_match"] is True

    def test_case_insensitive_match(self):
        result = self.evaluator._compare_field("hello", "HELLO", "vendor_name")
        assert result["exact_match"] is True  # normalize_string lowercases

    def test_fuzzy_string_match(self):
        result = self.evaluator._compare_field("Helloo", "Hello", "vendor_name")
        assert result["exact_match"] is False
        assert result["fuzzy_match"] is True  # edit distance = 1

    def test_no_match(self):
        result = self.evaluator._compare_field("Completely Different", "Hello", "vendor_name")
        assert result["exact_match"] is False
        assert result["fuzzy_match"] is False

    def test_both_null(self):
        result = self.evaluator._compare_field(None, None, "vendor_name")
        assert result["exact_match"] is True
        assert result["fuzzy_match"] is True

    def test_one_null(self):
        result = self.evaluator._compare_field(None, "Hello", "vendor_name")
        assert result["exact_match"] is False
        assert result["fuzzy_match"] is False

    def test_numeric_exact_match(self):
        result = self.evaluator._compare_field(42.50, 42.50, "total_amount")
        assert result["exact_match"] is True

    def test_numeric_fuzzy_match(self):
        # 42.505 rounds to 42.50 (exact match), so use 42.51 which is
        # within tolerance (0.01) but not an exact match
        result = self.evaluator._compare_field(42.51, 42.50, "total_amount")
        assert result["exact_match"] is False
        assert result["fuzzy_match"] is True  # within 0.01 tolerance

    def test_numeric_from_string(self):
        result = self.evaluator._compare_field("$42.50", 42.50, "total_amount")
        assert result["fuzzy_match"] is True


class TestLineItemComparison:
    """Tests for line item list comparison."""

    def setup_method(self):
        self.evaluator = DocLensEvaluator(MockExtractor(), fuzzy_threshold=3)

    def test_matching_line_items(self):
        pred = [{"item": "Coffee", "price": 5.0}, {"item": "Tea", "price": 3.0}]
        gt = [{"item": "Coffee", "price": 5.0}, {"item": "Tea", "price": 3.0}]
        result = self.evaluator._compare_line_items(pred, gt)
        assert result["exact_match"] is True

    def test_different_length(self):
        pred = [{"item": "Coffee", "price": 5.0}]
        gt = [{"item": "Coffee", "price": 5.0}, {"item": "Tea", "price": 3.0}]
        result = self.evaluator._compare_line_items(pred, gt)
        assert result["exact_match"] is False

    def test_non_list_input(self):
        result = self.evaluator._compare_line_items("not a list", [])
        assert result["exact_match"] is False


class TestAggregateMetrics:
    """Tests for aggregate metric computation."""

    def setup_method(self):
        self.evaluator = DocLensEvaluator(MockExtractor())

    def test_empty_results(self):
        metrics = self.evaluator._compute_aggregate_metrics([])
        assert metrics == {}

    def test_all_correct(self):
        results = [
            {
                "is_valid_json": True,
                "field_results": {
                    "vendor_name": {"exact_match": True, "fuzzy_match": True},
                    "date": {"exact_match": True, "fuzzy_match": True},
                    "total_amount": {"exact_match": True, "fuzzy_match": True},
                    "tax_amount": {"exact_match": True, "fuzzy_match": True},
                    "line_items": {"exact_match": True, "fuzzy_match": True},
                    "payment_method": {"exact_match": True, "fuzzy_match": True},
                },
                "metadata": {},
            }
        ]
        metrics = self.evaluator._compute_aggregate_metrics(results)
        assert metrics["exact_match_accuracy"] == 1.0
        assert metrics["json_validity_rate"] == 1.0

    def test_all_wrong(self):
        results = [
            {
                "is_valid_json": False,
                "field_results": {
                    "vendor_name": {"exact_match": False, "fuzzy_match": False},
                    "date": {"exact_match": False, "fuzzy_match": False},
                    "total_amount": {"exact_match": False, "fuzzy_match": False},
                    "tax_amount": {"exact_match": False, "fuzzy_match": False},
                    "line_items": {"exact_match": False, "fuzzy_match": False},
                    "payment_method": {"exact_match": False, "fuzzy_match": False},
                },
                "metadata": {},
            }
        ]
        metrics = self.evaluator._compute_aggregate_metrics(results)
        assert metrics["exact_match_accuracy"] == 0.0
        assert metrics["json_validity_rate"] == 0.0
