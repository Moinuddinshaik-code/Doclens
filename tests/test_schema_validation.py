"""
Tests for JSON parsing, repair, and schema validation utilities.
"""

import json
import pytest
from doclens.utils import (
    extract_json_from_text,
    repair_json,
    parse_model_output,
    validate_output,
    normalize_string,
    normalize_amount,
    string_edit_distance,
)


class TestExtractJson:
    """Tests for JSON extraction from messy model output."""

    def test_clean_json(self):
        text = '{"vendor_name": "Test Store", "total_amount": 42.50}'
        result = extract_json_from_text(text)
        assert json.loads(result)["vendor_name"] == "Test Store"

    def test_json_with_markdown_fences(self):
        text = '```json\n{"vendor_name": "Test Store"}\n```'
        result = extract_json_from_text(text)
        assert json.loads(result)["vendor_name"] == "Test Store"

    def test_json_with_surrounding_text(self):
        text = 'Here is the result: {"vendor_name": "Test"} I hope this helps!'
        result = extract_json_from_text(text)
        assert json.loads(result)["vendor_name"] == "Test"

    def test_nested_json(self):
        text = '{"fields": {"vendor": "Test"}, "nested": {"a": 1}}'
        result = extract_json_from_text(text)
        parsed = json.loads(result)
        assert "fields" in parsed
        assert "nested" in parsed


class TestRepairJson:
    """Tests for JSON repair heuristics."""

    def test_trailing_comma(self):
        text = '{"a": 1, "b": 2,}'
        result = repair_json(text)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_trailing_comma_in_array(self):
        text = '{"items": [1, 2, 3,]}'
        result = repair_json(text)
        assert json.loads(result) == {"items": [1, 2, 3]}

    def test_missing_closing_brace(self):
        text = '{"a": 1, "b": 2'
        result = repair_json(text)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_missing_closing_bracket(self):
        text = '{"items": [1, 2, 3}'
        result = repair_json(text)
        # Should add missing bracket
        assert "]" in result


class TestParseModelOutput:
    """Tests for the full parsing pipeline."""

    def test_valid_json(self):
        output = '{"vendor_name": "Store", "total_amount": 10.0}'
        parsed, error = parse_model_output(output)
        assert parsed is not None
        assert error is None
        assert parsed["vendor_name"] == "Store"

    def test_json_in_markdown(self):
        output = '```json\n{"vendor_name": "Store"}\n```'
        parsed, error = parse_model_output(output)
        assert parsed is not None
        assert parsed["vendor_name"] == "Store"

    def test_json_with_trailing_comma(self):
        output = '{"vendor_name": "Store", "total_amount": 10.0,}'
        parsed, error = parse_model_output(output)
        assert parsed is not None

    def test_completely_invalid(self):
        output = "This is not JSON at all, just random text."
        parsed, error = parse_model_output(output)
        assert parsed is None
        assert error is not None


class TestValidateOutput:
    """Tests for schema validation."""

    def test_valid_receipt(self):
        data = {
            "vendor_name": "Test Store",
            "date": "2024-01-15",
            "total_amount": 42.50,
            "tax_amount": 3.50,
            "line_items": [{"item": "Coffee", "price": 5.00}],
            "payment_method": "Cash",
        }
        is_valid, errors = validate_output(data, "receipt")
        assert is_valid
        assert len(errors) == 0

    def test_receipt_missing_required(self):
        data = {"payment_method": "Cash"}  # Missing vendor_name, date, total_amount
        is_valid, errors = validate_output(data, "receipt")
        assert not is_valid
        assert len(errors) > 0

    def test_receipt_with_nulls(self):
        data = {
            "vendor_name": "Store",
            "date": "2024-01-15",
            "total_amount": 10.0,
            "tax_amount": None,
            "line_items": None,
            "payment_method": None,
        }
        is_valid, errors = validate_output(data, "receipt")
        assert is_valid

    def test_unknown_document_type(self):
        is_valid, errors = validate_output({}, "unknown_type")
        assert not is_valid


class TestNormalization:
    """Tests for string and amount normalization."""

    def test_normalize_string_basic(self):
        assert normalize_string("  Hello  World  ") == "hello world"

    def test_normalize_string_currency(self):
        assert normalize_string("$42.50") == "42.50"

    def test_normalize_string_none(self):
        assert normalize_string(None) is None

    def test_normalize_amount_number(self):
        assert normalize_amount(42.5) == 42.5

    def test_normalize_amount_string(self):
        assert normalize_amount("$42.50") == 42.5

    def test_normalize_amount_none(self):
        assert normalize_amount(None) is None

    def test_normalize_amount_invalid(self):
        assert normalize_amount("not a number") is None


class TestEditDistance:
    """Tests for Levenshtein edit distance."""

    def test_identical(self):
        assert string_edit_distance("hello", "hello") == 0

    def test_one_insertion(self):
        assert string_edit_distance("hello", "helloo") == 1

    def test_one_deletion(self):
        assert string_edit_distance("hello", "hell") == 1

    def test_one_substitution(self):
        assert string_edit_distance("hello", "hallo") == 1

    def test_empty_strings(self):
        assert string_edit_distance("", "") == 0

    def test_one_empty(self):
        assert string_edit_distance("hello", "") == 5

    def test_completely_different(self):
        assert string_edit_distance("abc", "xyz") == 3
