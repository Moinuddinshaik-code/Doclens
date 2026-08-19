"""
Prompt templates for DocLens.

All prompts used during training AND inference are defined here in one place.
This is critical for consistency — if the training prompt differs from the
inference prompt even slightly, the model's performance will degrade because
it was conditioned on a specific format during fine-tuning.

Why this matters (interview talking point):
    A common failure mode in VLM fine-tuning is prompt mismatch between train
    and serve. The model learns to associate the exact prompt format with the
    expected output structure. Even small differences (extra whitespace, different
    field ordering in the instruction) can cause the model to produce malformed
    output or revert to its base behavior.
"""

from typing import Optional


# ─── Field Definitions ──────────────────────────────────────────────────────────
# These define the expected output schema per document type.
# The model is trained to produce JSON with exactly these keys.

RECEIPT_FIELDS = {
    "vendor_name": "string — name of the store/vendor",
    "date": "string — date in YYYY-MM-DD format",
    "total_amount": "number — total amount charged",
    "tax_amount": "number — tax amount (null if not shown)",
    "line_items": "list of objects — each with 'item' (string) and 'price' (number)",
    "payment_method": "string — e.g. 'cash', 'credit card', 'upi' (null if not shown)",
}

ID_CARD_FIELDS = {
    "full_name": "string — full name on the ID",
    "id_number": "string — ID/document number",
    "date_of_birth": "string — DOB in YYYY-MM-DD format",
    "expiry_date": "string — expiry date in YYYY-MM-DD format (null if not shown)",
    "issuing_authority": "string — issuing organization/government body",
}

DOCUMENT_TYPE_FIELDS = {
    "receipt": RECEIPT_FIELDS,
    "id_card": ID_CARD_FIELDS,
}


# ─── Prompt Construction ────────────────────────────────────────────────────────

def get_field_list_str(document_type: str) -> str:
    """Format field definitions into a readable string for the prompt.

    Example output:
        - vendor_name: string — name of the store/vendor
        - date: string — date in YYYY-MM-DD format
        ...
    """
    fields = DOCUMENT_TYPE_FIELDS.get(document_type)
    if fields is None:
        raise ValueError(
            f"Unknown document type: '{document_type}'. "
            f"Supported types: {list(DOCUMENT_TYPE_FIELDS.keys())}"
        )
    return "\n".join(f"- {name}: {desc}" for name, desc in fields.items())


def build_system_prompt(document_type: str) -> str:
    """Build the system prompt for a given document type.

    This prompt is used identically during training (as the system message)
    and during inference. Changing it after training will degrade performance.
    """
    field_list = get_field_list_str(document_type)
    return (
        "You are a document extraction assistant specialized in reading document images "
        "and extracting structured information.\n\n"
        "Extract the following fields as a JSON object:\n"
        f"{field_list}\n\n"
        "Rules:\n"
        "- Output ONLY valid JSON. No explanation, no markdown fences, no extra text.\n"
        "- If a field is not visible or cannot be determined, use null.\n"
        "- For amounts, use numbers (not strings). Example: 42.50, not \"$42.50\".\n"
        "- For dates, use YYYY-MM-DD format.\n"
        "- For line_items, use a list of objects with 'item' and 'price' keys.\n"
        "- Be precise. Copy text exactly as shown in the document."
    )


def build_user_prompt(document_type: str) -> str:
    """Build the user-turn text prompt (accompanies the image).

    The image is provided separately via the VLM's image input mechanism.
    This text prompt tells the model what to do with the image.
    """
    return f"Extract all fields from this {document_type} image as JSON."


def build_tamper_detection_prompt() -> str:
    """Build a prompt for tamper/forgery detection (secondary query).

    This is used as a follow-up after field extraction. The model examines
    the document for visual inconsistencies that suggest manipulation.
    """
    return (
        "Examine this document image carefully for signs of tampering or forgery. "
        "Look for:\n"
        "- Inconsistent fonts or font sizes within the same field type\n"
        "- Misaligned text baselines\n"
        "- Resolution mismatches between different parts of the document\n"
        "- Color inconsistencies in text or background\n"
        "- Unnatural edges around text regions (suggesting cut-and-paste)\n\n"
        "Respond with a JSON object:\n"
        '{"is_tampered": true/false, "confidence": 0.0-1.0, '
        '"explanation": "one sentence explaining your assessment"}'
    )


def build_training_messages(
    document_type: str,
    ground_truth_json: str,
    image_path: Optional[str] = None,
) -> list[dict]:
    """Build the full message list for a training example.

    This produces the exact format expected by Qwen2-VL's chat template:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": [{"type": "image", ...}, {"type": "text", ...}]},
        {"role": "assistant", "content": "..."}
    ]

    Args:
        document_type: Type of document ("receipt" or "id_card")
        ground_truth_json: The target JSON string the model should learn to produce
        image_path: Path to the document image (used during training data prep)

    Returns:
        List of message dicts in the chat format
    """
    system_msg = {"role": "system", "content": build_system_prompt(document_type)}

    user_content = []
    if image_path:
        user_content.append({"type": "image", "image": image_path})
    user_content.append({"type": "text", "text": build_user_prompt(document_type)})

    user_msg = {"role": "user", "content": user_content}
    assistant_msg = {"role": "assistant", "content": ground_truth_json}

    return [system_msg, user_msg, assistant_msg]


def build_inference_messages(
    document_type: str,
    image_path: str,
) -> list[dict]:
    """Build messages for inference (no assistant response — model generates it).

    Same as training messages but without the assistant turn.
    """
    system_msg = {"role": "system", "content": build_system_prompt(document_type)}
    user_msg = {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": build_user_prompt(document_type)},
        ],
    }
    return [system_msg, user_msg]
