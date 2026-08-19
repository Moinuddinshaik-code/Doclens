"""
DocLens — Quick Example: Extract fields from a receipt image.

This script demonstrates the core DocLens pipeline:
1. Load the fine-tuned VLM (with LoRA adapter)
2. Extract structured fields from a receipt image
3. Display results with confidence scores

Usage:
    # With fine-tuned adapter:
    python examples/extract_receipt.py path/to/receipt.png --adapter training/checkpoints/lora_adapter

    # Zero-shot (no adapter):
    python examples/extract_receipt.py path/to/receipt.png --no-adapter

    # With tamper detection:
    python examples/extract_receipt.py path/to/receipt.png --adapter training/checkpoints/lora_adapter --tamper
"""

import argparse
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Extract fields from a receipt image")
    parser.add_argument("image", help="Path to receipt image (PNG/JPEG)")
    parser.add_argument(
        "--adapter", "-a", default=None,
        help="Path to LoRA adapter directory (omit for zero-shot baseline)",
    )
    parser.add_argument(
        "--no-adapter", action="store_true",
        help="Run zero-shot without any adapter",
    )
    parser.add_argument(
        "--quantization", "-q", default="4bit",
        choices=["4bit", "8bit", "none"],
        help="Quantization mode (default: 4bit)",
    )
    parser.add_argument(
        "--tamper", action="store_true",
        help="Also run tamper/forgery detection",
    )

    args = parser.parse_args()

    # Validate image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Load model
    from doclens.model import load_model
    from doclens.extractor import DocLensExtractor

    adapter_path = None if args.no_adapter else args.adapter

    print("Loading model...")
    model, processor = load_model(
        quantization=args.quantization,
        lora_adapter_path=adapter_path,
    )

    extractor = DocLensExtractor(model, processor)

    # Run extraction
    print(f"\nExtracting fields from: {args.image}")
    print("-" * 50)

    if args.tamper:
        result = extractor.extract_with_tamper(args.image, "receipt")
    else:
        result = extractor.extract(args.image, "receipt")

    # Display results
    print("\n[+] Extracted Fields:")
    if result.get("fields"):
        for field_name, value in result["fields"].items():
            conf = result.get("confidence", {}).get(field_name, "N/A")
            conf_str = f"{conf:.2f}" if isinstance(conf, float) else conf
            print(f"  {field_name}: {value}  (confidence: {conf_str})")
    else:
        print("  [!] No fields extracted (parsing failed)")
        print(f"  Raw output: {result.get('raw_output', 'N/A')[:200]}")

    if args.tamper:
        fraud = result.get("fraud_flag", False)
        status_tag = "[ALERT]" if fraud else "[OK]"
        print(f"\n{status_tag} Tamper Detection: {'TAMPERED' if fraud else 'CLEAN'}")
        if result.get("explanation"):
            print(f"  Explanation: {result['explanation']}")

    print(f"\n[Time] Latency: {result.get('latency_ms', 'N/A')} ms")

    if not result.get("is_valid", True):
        print(f"\n[!] Validation errors: {result.get('validation_errors', [])}")

    # Save full result
    output_path = os.path.splitext(args.image)[0] + "_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[Save] Full result saved to: {output_path}")


if __name__ == "__main__":
    main()
