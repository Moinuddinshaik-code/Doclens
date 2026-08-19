"""
Utility functions for DocLens.

Handles JSON parsing/repair, schema validation, and common helpers.

Why JSON repair matters (interview talking point):
    VLMs don't always produce perfectly valid JSON. Common failures include:
    - Trailing commas after the last element
    - Missing closing braces/brackets
    - Including markdown code fences (```json ... ```)
    - Extra text before/after the JSON object
    A production system needs to handle these gracefully rather than crash.
"""

import json
import re
from typing import Any, Optional

import jsonschema


# ─── JSON Schema Definitions ────────────────────────────────────────────────────
# These schemas validate that the model's output has the correct structure.
# They don't check correctness of values — that's the evaluator's job.

RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor_name": {"type": ["string", "null"]},
        "date": {"type": ["string", "null"]},
        "total_amount": {"type": ["number", "null"]},
        "tax_amount": {"type": ["number", "null"]},
        "line_items": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": ["string", "null"]},
                    "price": {"type": ["number", "null"]},
                },
            },
        },
        "payment_method": {"type": ["string", "null"]},
    },
    "required": ["vendor_name", "date", "total_amount"],
}

ID_CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "full_name": {"type": ["string", "null"]},
        "id_number": {"type": ["string", "null"]},
        "date_of_birth": {"type": ["string", "null"]},
        "expiry_date": {"type": ["string", "null"]},
        "issuing_authority": {"type": ["string", "null"]},
    },
    "required": ["full_name", "id_number"],
}

TAMPER_SCHEMA = {
    "type": "object",
    "properties": {
        "is_tampered": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "explanation": {"type": ["string", "null"]},
    },
    "required": ["is_tampered"],
}

DOCUMENT_SCHEMAS = {
    "receipt": RECEIPT_SCHEMA,
    "id_card": ID_CARD_SCHEMA,
    "tamper": TAMPER_SCHEMA,
}


# ─── JSON Parsing & Repair ──────────────────────────────────────────────────────

def extract_json_from_text(text: str) -> str:
    """Extract a JSON object from model output that may contain extra text.

    Handles common VLM output issues:
    1. Markdown code fences: ```json { ... } ```
    2. Extra text before/after the JSON: "Here is the result: { ... } I hope this helps"
    3. Multiple JSON objects (takes the first one)

    Args:
        text: Raw model output string

    Returns:
        Cleaned string that should be parseable as JSON
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # Try to find a JSON object by matching outermost braces
    brace_depth = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == "{":
            if brace_depth == 0:
                start_idx = i
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0 and start_idx is not None:
                return text[start_idx : i + 1]

    # If no matched braces found, return the stripped text as-is
    return text


def repair_json(text: str) -> str:
    """Attempt to repair common JSON syntax errors from model output.

    Fixes:
    - Trailing commas: {"a": 1, "b": 2,} → {"a": 1, "b": 2}
    - Single quotes: {'a': 1} → {"a": 1}
    - Missing closing brace (appends one)
    - Unquoted keys (simple cases)

    Args:
        text: Potentially malformed JSON string

    Returns:
        Repaired JSON string (may still be invalid in complex cases)
    """
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Replace single quotes with double quotes (simple heuristic)
    # Only do this if there are no double quotes already (to avoid breaking valid JSON)
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')

    # Count braces — add missing closing braces
    open_braces = text.count("{") - text.count("}")
    if open_braces > 0:
        text = text + "}" * open_braces

    open_brackets = text.count("[") - text.count("]")
    if open_brackets > 0:
        text = text + "]" * open_brackets

    return text


def parse_model_output(raw_output: str) -> tuple[Optional[dict], Optional[str]]:
    """Parse model output into a Python dict, with repair attempts.

    This is the main entry point for handling model output. It tries:
    1. Direct JSON parse
    2. Extract JSON from surrounding text, then parse
    3. Repair common issues, then parse
    4. Give up and return None with error message

    Args:
        raw_output: Raw string output from the model

    Returns:
        Tuple of (parsed_dict, error_message).
        If parsing succeeds, error_message is None.
        If parsing fails, parsed_dict is None and error_message explains why.
    """
    # Attempt 1: Direct parse
    try:
        return json.loads(raw_output), None
    except json.JSONDecodeError:
        pass

    # Attempt 2: Extract JSON from surrounding text
    extracted = extract_json_from_text(raw_output)
    try:
        return json.loads(extracted), None
    except json.JSONDecodeError:
        pass

    # Attempt 3: Repair + parse
    repaired = repair_json(extracted)
    try:
        return json.loads(repaired), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse failed after repair: {e}. Raw output: {raw_output[:200]}"


# ─── Schema Validation ───────────────────────────────────────────────────────────

def validate_output(parsed: dict, document_type: str) -> tuple[bool, list[str]]:
    """Validate parsed output against the expected schema for a document type.

    Args:
        parsed: Parsed JSON dict from the model
        document_type: "receipt", "id_card", or "tamper"

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    schema = DOCUMENT_SCHEMAS.get(document_type)
    if schema is None:
        return False, [f"Unknown document type: {document_type}"]

    validator = jsonschema.Draft7Validator(schema)
    errors = list(validator.iter_errors(parsed))

    if not errors:
        return True, []

    error_messages = [
        f"{'.'.join(str(p) for p in e.absolute_path)}: {e.message}" if e.absolute_path
        else e.message
        for e in errors
    ]
    return False, error_messages


# ─── Comparison Utilities ────────────────────────────────────────────────────────

def normalize_string(s: Any) -> Optional[str]:
    """Normalize a string value for comparison.

    Handles: None, whitespace, case normalization for fuzzy matching.
    """
    if s is None:
        return None
    s = str(s).strip().lower()
    # Normalize common variations
    s = re.sub(r"\s+", " ", s)  # collapse whitespace
    s = re.sub(r"[$₹€£]", "", s)  # remove currency symbols
    return s


def normalize_amount(val: Any) -> Optional[float]:
    """Normalize a numeric amount for comparison.

    Handles string amounts like "$42.50", "42.5", etc.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return round(float(val), 2)
    # Try parsing string as number
    try:
        cleaned = re.sub(r"[^0-9.\-]", "", str(val))
        return round(float(cleaned), 2)
    except (ValueError, TypeError):
        return None


def string_edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Used for fuzzy field matching — e.g., "Highland Coffee" vs "Highland Coffe"
    (off by one character due to OCR-style error).

    Time complexity: O(m*n) where m, n are string lengths.
    Space complexity: O(min(m, n)) using the rolling-row optimization.
    """
    if len(s1) < len(s2):
        return string_edit_distance(s2, s1)

    # Previous and current row of the DP table
    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertions, deletions, substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
