"""
Inference pipeline for DocLens.

Processes document images through the VLM to extract structured fields.
Handles: preprocessing → prompt construction → generation → parsing → validation
→ confidence scoring → tamper detection.

Pipeline flow:
    ┌────────────┐    ┌──────────┐    ┌────────────┐    ┌───────────┐
    │ Input Image │───►│ Preprocess│───►│ Build Prompt│───►│ VLM       │
    │ (path/bytes)│    │ (resize)  │    │ (template)  │    │ Generate  │
    └────────────┘    └──────────┘    └────────────┘    └─────┬─────┘
                                                              │
    ┌────────────┐    ┌──────────┐    ┌────────────┐    ┌─────▼─────┐
    │ Final Output│◄──│ Confidence│◄──│ Schema     │◄──│ Parse JSON│
    │ + latency   │    │ Scoring   │    │ Validate   │    │ + repair  │
    └────────────┘    └──────────┘    └────────────┘    └───────────┘

Design decisions (interview talking points):
    - Single-pass extraction: one forward pass to extract all fields as JSON,
      rather than one query per field. This is 5-10x more efficient.
    - JSON output format: structured and machine-parseable, not free-text.
      Production systems need structured output they can feed into downstream logic.
    - Retry on parse failure: if the first generation doesn't produce valid JSON,
      we retry once with a stricter prompt. This handles transient failures.
    - Confidence via log-probs: if available, mean token log-probability per field
      is a principled confidence estimate. Fallback: multi-sample consistency.
"""

import json
import time
from typing import Any, Optional

import torch
from PIL import Image

from doclens.prompts import (
    build_inference_messages,
    build_tamper_detection_prompt,
    DOCUMENT_TYPE_FIELDS,
)
from doclens.utils import parse_model_output, validate_output


class DocLensExtractor:
    """Main extraction pipeline for document field extraction.

    Usage:
        from doclens import DocLensExtractor, load_model

        model, processor = load_model(lora_adapter_path="path/to/adapter")
        extractor = DocLensExtractor(model, processor)

        result = extractor.extract("receipt.png", document_type="receipt")
        print(result["fields"])
        print(result["confidence"])
    """

    def __init__(
        self,
        model,
        processor,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False,
    ):
        """Initialize the extractor.

        Args:
            model: Loaded VLM (from doclens.model.load_model)
            processor: Model processor/tokenizer
            device: Device to run inference on (auto-detected if None)
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature (lower = more deterministic)
            do_sample: Whether to use sampling (False = greedy decoding)
        """
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

    def _preprocess_image(self, image_input) -> Image.Image:
        """Load and validate an image.

        Args:
            image_input: File path (str), PIL Image, or bytes

        Returns:
            PIL Image in RGB mode
        """
        if isinstance(image_input, str):
            img = Image.open(image_input)
        elif isinstance(image_input, bytes):
            import io
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        return img.convert("RGB")

    def _generate(self, messages: list[dict], image: Image.Image) -> str:
        """Run VLM generation given messages and an image.

        Args:
            messages: Chat-format messages (system, user with image placeholder)
            image: PIL Image to process

        Returns:
            Generated text string
        """
        # Prepare inputs using the processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process image and text together
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else None,
                do_sample=self.do_sample,
                repetition_penalty=1.1,
            )

        # Decode only the generated tokens (skip the input)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return output_text.strip()

    def extract(
        self,
        image_input,
        document_type: str = "receipt",
        retry_on_failure: bool = True,
    ) -> dict[str, Any]:
        """Extract structured fields from a document image.

        This is the main public API. It:
        1. Preprocesses the image
        2. Builds the prompt
        3. Runs VLM generation
        4. Parses and validates the JSON output
        5. Retries once if parsing fails
        6. Returns structured results with confidence and latency

        Args:
            image_input: Image path, bytes, or PIL Image
            document_type: "receipt" or "id_card"
            retry_on_failure: Whether to retry with stricter prompt on parse failure

        Returns:
            Dict with keys: fields, confidence, raw_output, is_valid,
            validation_errors, latency_ms
        """
        start_time = time.perf_counter()

        # Validate document type
        if document_type not in DOCUMENT_TYPE_FIELDS:
            return {
                "fields": None,
                "confidence": {},
                "raw_output": None,
                "is_valid": False,
                "validation_errors": [f"Unknown document type: {document_type}"],
                "latency_ms": 0,
            }

        # Preprocess
        image = self._preprocess_image(image_input)

        # Build messages
        messages = build_inference_messages(document_type, "image_placeholder")

        # Generate
        raw_output = self._generate(messages, image)

        # Parse
        parsed, parse_error = parse_model_output(raw_output)

        # Retry once on failure with a stricter prompt
        if parsed is None and retry_on_failure:
            # Add explicit instruction to fix the output
            retry_messages = build_inference_messages(document_type, "image_placeholder")
            retry_messages.append({
                "role": "assistant",
                "content": raw_output,
            })
            retry_messages.append({
                "role": "user",
                "content": (
                    "Your previous output was not valid JSON. "
                    "Please output ONLY a valid JSON object with the requested fields. "
                    "No explanation, no markdown, just the JSON object."
                ),
            })

            raw_output_retry = self._generate(retry_messages, image)
            parsed_retry, parse_error_retry = parse_model_output(raw_output_retry)

            if parsed_retry is not None:
                parsed = parsed_retry
                parse_error = None
                raw_output = raw_output_retry

        # Validate against schema
        if parsed is not None:
            is_valid, validation_errors = validate_output(parsed, document_type)
        else:
            is_valid = False
            validation_errors = [parse_error] if parse_error else ["Unknown parse error"]

        # Compute confidence (simplified: based on parse success and field completeness)
        confidence = self._compute_confidence(parsed, document_type)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "fields": parsed,
            "confidence": confidence,
            "raw_output": raw_output,
            "is_valid": is_valid,
            "validation_errors": validation_errors,
            "latency_ms": round(elapsed_ms, 1),
        }

    def _compute_confidence(
        self, parsed: Optional[dict], document_type: str
    ) -> dict[str, float]:
        """Compute per-field confidence scores.

        Current approach: field completeness heuristic.
        - Field present and non-null: 0.8 base confidence
        - Field matches expected type: +0.1
        - Field is null: 0.0

        Future improvement: use token-level log-probabilities from the
        generation step for a principled confidence estimate. This requires
        the model to return logits during generation (output_scores=True).

        Args:
            parsed: Parsed JSON output (or None if parsing failed)
            document_type: Document type for expected field list

        Returns:
            Dict mapping field names to confidence scores (0.0–1.0)
        """
        expected_fields = DOCUMENT_TYPE_FIELDS.get(document_type, {})
        confidence = {}

        if parsed is None:
            return {field: 0.0 for field in expected_fields}

        for field_name, field_desc in expected_fields.items():
            value = parsed.get(field_name)

            if value is None:
                confidence[field_name] = 0.0
            elif isinstance(value, str) and value.strip() == "":
                confidence[field_name] = 0.1
            else:
                # Base confidence for non-null, non-empty values
                conf = 0.8

                # Type check bonus
                if "number" in field_desc and isinstance(value, (int, float)):
                    conf += 0.1
                elif "string" in field_desc and isinstance(value, str):
                    conf += 0.1
                elif "list" in field_desc and isinstance(value, list):
                    conf += 0.1

                confidence[field_name] = min(conf, 1.0)

        return confidence

    def detect_tamper(self, image_input) -> dict[str, Any]:
        """Run tamper/forgery detection on a document image.

        This is a separate query that examines the document for visual
        inconsistencies suggesting manipulation.

        Args:
            image_input: Image path, bytes, or PIL Image

        Returns:
            Dict with keys: is_tampered, confidence, explanation, latency_ms
        """
        start_time = time.perf_counter()

        image = self._preprocess_image(image_input)

        # Build tamper detection prompt
        tamper_prompt = build_tamper_detection_prompt()
        messages = [
            {"role": "system", "content": "You are a document forensics expert."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "image_placeholder"},
                    {"type": "text", "text": tamper_prompt},
                ],
            },
        ]

        raw_output = self._generate(messages, image)
        parsed, _ = parse_model_output(raw_output)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if parsed is not None:
            return {
                "is_tampered": parsed.get("is_tampered", False),
                "confidence": parsed.get("confidence", 0.5),
                "explanation": parsed.get("explanation"),
                "raw_output": raw_output,
                "latency_ms": round(elapsed_ms, 1),
            }
        else:
            return {
                "is_tampered": False,
                "confidence": 0.0,
                "explanation": "Could not determine (model output unparseable)",
                "raw_output": raw_output,
                "latency_ms": round(elapsed_ms, 1),
            }

    def extract_with_tamper(
        self,
        image_input,
        document_type: str = "receipt",
    ) -> dict[str, Any]:
        """Extract fields AND check for tampering in one call.

        Combines field extraction and tamper detection results.

        Args:
            image_input: Image path, bytes, or PIL Image
            document_type: "receipt" or "id_card"

        Returns:
            Combined result dict with fields, confidence, fraud_flag, explanation, latency_ms
        """
        # Run field extraction
        extraction = self.extract(image_input, document_type)

        # Run tamper detection
        tamper = self.detect_tamper(image_input)

        # Combine results
        total_latency = extraction["latency_ms"] + tamper["latency_ms"]

        return {
            "fields": extraction["fields"],
            "confidence": extraction["confidence"],
            "fraud_flag": tamper["is_tampered"],
            "explanation": tamper["explanation"],
            "latency_ms": round(total_latency, 1),
            "extraction_details": extraction,
            "tamper_details": tamper,
        }


# ─── CLI Entry Point ─────────────────────────────────────────────────────────────

def main():
    """CLI entry point for single-image extraction."""
    import argparse

    parser = argparse.ArgumentParser(description="DocLens: Extract fields from document images")
    parser.add_argument("image", help="Path to document image")
    parser.add_argument(
        "--type", "-t", default="receipt",
        choices=["receipt", "id_card"],
        help="Document type (default: receipt)",
    )
    parser.add_argument(
        "--adapter", "-a", default=None,
        help="Path to LoRA adapter weights",
    )
    parser.add_argument(
        "--quantization", "-q", default="4bit",
        choices=["4bit", "8bit", "none"],
        help="Quantization mode (default: 4bit)",
    )
    parser.add_argument(
        "--tamper", action="store_true",
        help="Also run tamper detection",
    )

    args = parser.parse_args()

    from doclens.model import load_model

    model, processor = load_model(
        quantization=args.quantization,
        lora_adapter_path=args.adapter,
    )

    extractor = DocLensExtractor(model, processor)

    if args.tamper:
        result = extractor.extract_with_tamper(args.image, args.type)
    else:
        result = extractor.extract(args.image, args.type)

    print(json.dumps(result, indent=2, default=str))
