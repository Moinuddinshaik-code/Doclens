"""
Evaluation harness for DocLens.

Computes metrics across clean and adversarial test sets, generates comparison
reports, and performs per-augmentation failure analysis.

Why this is the most important module (interview talking points):
    The JD explicitly says:
    - "Design evaluation benchmarks beyond clean validation datasets"
    - "Build evaluation harnesses against noisy, adversarial, and real-world inputs"
    - "Investigate model failures, develop hypotheses, and validate improvements"

    This module delivers all three. The benchmark table in the README is generated
    by this code. The per-augmentation breakdown is what demonstrates you understand
    model failures — not just that the model works, but WHERE and WHY it fails.

Metrics computed:
    1. Field-level exact match accuracy (per field and averaged)
    2. Field-level F1 with fuzzy matching (edit distance tolerance)
    3. JSON validity rate (does output parse at all)
    4. Tamper detection precision/recall/F1
    5. Per-augmentation-type accuracy breakdown
    6. Inference latency and memory profiling
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from doclens.utils import (
    normalize_string,
    normalize_amount,
    string_edit_distance,
)
from doclens.prompts import DOCUMENT_TYPE_FIELDS


class DocLensEvaluator:
    """Evaluation harness for document extraction models.

    Usage:
        from doclens import DocLensEvaluator, DocLensExtractor, load_model

        model, processor = load_model(lora_adapter_path="path/to/adapter")
        extractor = DocLensExtractor(model, processor)
        evaluator = DocLensEvaluator(extractor)

        # Evaluate on clean test set
        clean_results = evaluator.evaluate_dataset("data/generated/clean_test")

        # Evaluate on adversarial test set
        adv_results = evaluator.evaluate_dataset("data/generated/adversarial_test/heavy")

        # Generate full comparison report
        report = evaluator.full_benchmark_report(
            clean_test_dir="data/generated/clean_test",
            adversarial_test_dir="data/generated/adversarial_test",
        )
    """

    def __init__(
        self,
        extractor,
        document_type: str = "receipt",
        fuzzy_threshold: int = 3,
        amount_tolerance: float = 0.01,
    ):
        """Initialize the evaluator.

        Args:
            extractor: DocLensExtractor instance
            document_type: Document type for field definitions
            fuzzy_threshold: Max edit distance for fuzzy string match
            amount_tolerance: Max absolute difference for numeric match
        """
        self.extractor = extractor
        self.document_type = document_type
        self.fuzzy_threshold = fuzzy_threshold
        self.amount_tolerance = amount_tolerance
        self.expected_fields = list(DOCUMENT_TYPE_FIELDS.get(document_type, {}).keys())

    def _compare_field(
        self, predicted: Any, ground_truth: Any, field_name: str
    ) -> dict[str, bool]:
        """Compare a single predicted field against ground truth.

        Returns both exact and fuzzy match results.

        Args:
            predicted: Model's predicted value
            ground_truth: Ground truth value
            field_name: Name of the field (for type-specific comparison)

        Returns:
            Dict with "exact_match" and "fuzzy_match" booleans
        """
        # Both null → match
        if predicted is None and ground_truth is None:
            return {"exact_match": True, "fuzzy_match": True}

        # One null, one not → mismatch
        if predicted is None or ground_truth is None:
            return {"exact_match": False, "fuzzy_match": False}

        # Special handling for line_items (list of objects)
        if field_name == "line_items":
            return self._compare_line_items(predicted, ground_truth)

        # Numeric fields
        if field_name in ("total_amount", "tax_amount"):
            pred_val = normalize_amount(predicted)
            gt_val = normalize_amount(ground_truth)
            if pred_val is not None and gt_val is not None:
                exact = abs(pred_val - gt_val) < 0.001
                fuzzy = abs(pred_val - gt_val) <= self.amount_tolerance
                return {"exact_match": exact, "fuzzy_match": fuzzy}
            return {"exact_match": False, "fuzzy_match": False}

        # String fields
        pred_str = normalize_string(predicted)
        gt_str = normalize_string(ground_truth)

        if pred_str is None or gt_str is None:
            return {"exact_match": False, "fuzzy_match": False}

        exact = pred_str == gt_str
        edit_dist = string_edit_distance(pred_str, gt_str)
        fuzzy = edit_dist <= self.fuzzy_threshold

        return {"exact_match": exact, "fuzzy_match": fuzzy}

    def _compare_line_items(
        self, predicted: Any, ground_truth: Any
    ) -> dict[str, bool]:
        """Compare line item lists.

        Line items are matched by item name (fuzzy) and price (tolerance).
        The order may differ between prediction and ground truth.
        """
        if not isinstance(predicted, list) or not isinstance(ground_truth, list):
            return {"exact_match": False, "fuzzy_match": False}

        if len(predicted) != len(ground_truth):
            # Different number of items — partial credit via fuzzy
            matched = 0
            gt_used = [False] * len(ground_truth)

            for pred_item in predicted:
                pred_name = normalize_string(pred_item.get("item", ""))
                pred_price = normalize_amount(pred_item.get("price"))

                for j, gt_item in enumerate(ground_truth):
                    if gt_used[j]:
                        continue
                    gt_name = normalize_string(gt_item.get("item", ""))
                    gt_price = normalize_amount(gt_item.get("price"))

                    name_match = (
                        pred_name == gt_name
                        or string_edit_distance(pred_name or "", gt_name or "") <= self.fuzzy_threshold
                    )
                    price_match = (
                        pred_price is not None
                        and gt_price is not None
                        and abs(pred_price - gt_price) <= self.amount_tolerance
                    )

                    if name_match and price_match:
                        matched += 1
                        gt_used[j] = True
                        break

            total = max(len(predicted), len(ground_truth))
            ratio = matched / total if total > 0 else 0

            return {
                "exact_match": ratio == 1.0,
                "fuzzy_match": ratio >= 0.5,
            }

        # Same length — check one-to-one
        all_exact = True
        all_fuzzy = True

        for pred_item, gt_item in zip(predicted, ground_truth):
            name_cmp = self._compare_field(
                pred_item.get("item"), gt_item.get("item"), "item_name"
            )
            price_cmp = self._compare_field(
                pred_item.get("price"), gt_item.get("price"), "total_amount"
            )
            if not (name_cmp["exact_match"] and price_cmp["exact_match"]):
                all_exact = False
            if not (name_cmp["fuzzy_match"] and price_cmp["fuzzy_match"]):
                all_fuzzy = False

        return {"exact_match": all_exact, "fuzzy_match": all_fuzzy}

    def evaluate_single(
        self, image_path: str, ground_truth: dict
    ) -> dict[str, Any]:
        """Evaluate a single image against its ground truth.

        Args:
            image_path: Path to the document image
            ground_truth: Ground truth fields dict

        Returns:
            Per-field comparison results plus overall metrics
        """
        # Run extraction
        result = self.extractor.extract(image_path, self.document_type)

        predicted = result.get("fields")
        is_valid_json = result.get("is_valid", False)
        latency = result.get("latency_ms", 0)

        # Compare each field
        field_results = {}
        for field_name in self.expected_fields:
            gt_value = ground_truth.get(field_name)
            pred_value = predicted.get(field_name) if predicted else None

            comparison = self._compare_field(pred_value, gt_value, field_name)
            field_results[field_name] = {
                "predicted": pred_value,
                "ground_truth": gt_value,
                **comparison,
            }

        return {
            "image_path": image_path,
            "field_results": field_results,
            "is_valid_json": is_valid_json,
            "latency_ms": latency,
            "raw_output": result.get("raw_output"),
        }

    def evaluate_dataset(
        self, dataset_dir: str, max_samples: Optional[int] = None
    ) -> dict[str, Any]:
        """Evaluate model on a full dataset directory.

        Args:
            dataset_dir: Directory containing image + JSON pairs
            max_samples: Max samples to evaluate (None = all)

        Returns:
            Aggregate metrics across the dataset
        """
        # Find all JSON metadata files
        json_files = sorted(Path(dataset_dir).glob("*.json"))
        if max_samples:
            json_files = json_files[:max_samples]

        if not json_files:
            return {"error": f"No JSON files found in {dataset_dir}"}

        print(f"Evaluating {len(json_files)} samples from {dataset_dir}...")

        all_results = []
        total_latency = 0

        for i, json_path in enumerate(json_files):
            with open(json_path, "r") as f:
                metadata = json.load(f)

            image_path = metadata.get("image_path", "")
            if not os.path.exists(image_path):
                # Try relative to dataset_dir
                image_filename = os.path.basename(image_path)
                image_path = os.path.join(dataset_dir, image_filename)

            if not os.path.exists(image_path):
                print(f"  Warning: Image not found: {image_path}, skipping")
                continue

            ground_truth = metadata.get("fields", {})
            result = self.evaluate_single(image_path, ground_truth)
            result["metadata"] = metadata
            all_results.append(result)
            total_latency += result["latency_ms"]

            if (i + 1) % 20 == 0:
                print(f"  Evaluated {i + 1}/{len(json_files)}")

        # Compute aggregate metrics
        metrics = self._compute_aggregate_metrics(all_results)
        metrics["num_samples"] = len(all_results)
        metrics["total_latency_ms"] = round(total_latency, 1)
        metrics["mean_latency_ms"] = round(total_latency / max(len(all_results), 1), 1)

        print(f"  Done. Results:")
        print(f"    Exact match accuracy: {metrics['exact_match_accuracy']:.3f}")
        print(f"    Fuzzy match accuracy: {metrics['fuzzy_match_accuracy']:.3f}")
        print(f"    JSON validity rate:   {metrics['json_validity_rate']:.3f}")
        print(f"    Mean latency:         {metrics['mean_latency_ms']:.1f} ms")

        return metrics

    def _compute_aggregate_metrics(self, results: list[dict]) -> dict[str, Any]:
        """Compute aggregate metrics from individual evaluation results.

        Args:
            results: List of per-sample evaluation results

        Returns:
            Dict with aggregate metrics
        """
        if not results:
            return {}

        # JSON validity rate
        valid_count = sum(1 for r in results if r["is_valid_json"])
        json_validity_rate = valid_count / len(results)

        # Per-field metrics
        field_exact = {f: [] for f in self.expected_fields}
        field_fuzzy = {f: [] for f in self.expected_fields}

        for result in results:
            for field_name in self.expected_fields:
                fr = result["field_results"].get(field_name, {})
                field_exact[field_name].append(fr.get("exact_match", False))
                field_fuzzy[field_name].append(fr.get("fuzzy_match", False))

        # Per-field accuracy
        per_field_exact = {
            f: sum(vals) / len(vals) if vals else 0.0
            for f, vals in field_exact.items()
        }
        per_field_fuzzy = {
            f: sum(vals) / len(vals) if vals else 0.0
            for f, vals in field_fuzzy.items()
        }

        # Averaged accuracy
        avg_exact = sum(per_field_exact.values()) / max(len(per_field_exact), 1)
        avg_fuzzy = sum(per_field_fuzzy.values()) / max(len(per_field_fuzzy), 1)

        # Tamper detection metrics (if applicable)
        tamper_metrics = self._compute_tamper_metrics(results)

        return {
            "exact_match_accuracy": round(avg_exact, 4),
            "fuzzy_match_accuracy": round(avg_fuzzy, 4),
            "json_validity_rate": round(json_validity_rate, 4),
            "per_field_exact_match": {
                f: round(v, 4) for f, v in per_field_exact.items()
            },
            "per_field_fuzzy_match": {
                f: round(v, 4) for f, v in per_field_fuzzy.items()
            },
            **tamper_metrics,
        }

    def _compute_tamper_metrics(self, results: list[dict]) -> dict[str, Any]:
        """Compute tamper detection metrics if tamper labels are available.

        Args:
            results: List of per-sample results with metadata

        Returns:
            Dict with precision, recall, F1 for tamper detection
        """
        y_true = []
        y_pred = []

        for result in results:
            metadata = result.get("metadata", {})
            if "is_tampered" not in metadata:
                continue

            y_true.append(metadata["is_tampered"])
            # Use a simple heuristic: if extraction produced invalid JSON or
            # low confidence, flag as potentially tampered
            is_valid = result.get("is_valid_json", True)
            y_pred.append(not is_valid)  # Simplified — improve with real tamper detection

        if not y_true:
            return {}

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        return {
            "tamper_precision": round(precision, 4),
            "tamper_recall": round(recall, 4),
            "tamper_f1": round(f1, 4),
        }

    def full_benchmark_report(
        self,
        clean_test_dir: str,
        adversarial_test_dir: str,
        output_path: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> dict[str, Any]:
        """Generate a complete benchmark report across all test sets.

        This produces the core results table for the README:
        - Clean test accuracy
        - Per-augmentation adversarial accuracy
        - Clean vs adversarial gap (the most insightful number)

        Args:
            clean_test_dir: Path to clean test set
            adversarial_test_dir: Path to adversarial test directory (with subdirs per augmentation)
            output_path: Where to save the report JSON (None = don't save)
            max_samples: Max samples per test set

        Returns:
            Complete benchmark report as a dict
        """
        report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

        # 1. Clean test evaluation
        print("\n═══ Clean Test Set ═══")
        clean_metrics = self.evaluate_dataset(clean_test_dir, max_samples)
        report["clean_test"] = clean_metrics

        # 2. Adversarial test subsets
        print("\n═══ Adversarial Test Sets ═══")
        adv_dir = Path(adversarial_test_dir)
        adversarial_results = {}

        if adv_dir.is_dir():
            for subset_dir in sorted(adv_dir.iterdir()):
                if subset_dir.is_dir():
                    print(f"\n── {subset_dir.name} ──")
                    subset_metrics = self.evaluate_dataset(
                        str(subset_dir), max_samples
                    )
                    adversarial_results[subset_dir.name] = subset_metrics

        report["adversarial_test"] = adversarial_results

        # 3. Compute deltas (clean vs adversarial)
        deltas = {}
        for subset_name, subset_metrics in adversarial_results.items():
            if "exact_match_accuracy" in subset_metrics and "exact_match_accuracy" in clean_metrics:
                delta = subset_metrics["exact_match_accuracy"] - clean_metrics["exact_match_accuracy"]
                deltas[subset_name] = round(delta, 4)

        report["accuracy_deltas"] = deltas

        # 4. Memory profiling
        if torch.cuda.is_available():
            report["gpu_memory"] = {
                "peak_allocated_mb": round(
                    torch.cuda.max_memory_allocated() / 1024 / 1024, 1
                ),
                "peak_reserved_mb": round(
                    torch.cuda.max_memory_reserved() / 1024 / 1024, 1
                ),
            }

        # Save report
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {output_path}")

        # Print summary table
        self._print_summary_table(report)

        return report

    def _print_summary_table(self, report: dict):
        """Print a formatted summary table to stdout."""
        print("\n" + "=" * 70)
        print("                    DOCLENS BENCHMARK REPORT")
        print("=" * 70)

        clean = report.get("clean_test", {})
        print(f"\n{'Test Set':<25} {'Exact Match':<15} {'Fuzzy Match':<15} {'JSON Valid':<12}")
        print("-" * 70)
        print(
            f"{'Clean Test':<25} "
            f"{clean.get('exact_match_accuracy', 'N/A'):<15} "
            f"{clean.get('fuzzy_match_accuracy', 'N/A'):<15} "
            f"{clean.get('json_validity_rate', 'N/A'):<12}"
        )

        for name, metrics in report.get("adversarial_test", {}).items():
            delta = report.get("accuracy_deltas", {}).get(name, "")
            delta_str = f" ({delta:+.3f})" if isinstance(delta, float) else ""
            print(
                f"{name:<25} "
                f"{metrics.get('exact_match_accuracy', 'N/A')}{delta_str:<15} "
                f"{metrics.get('fuzzy_match_accuracy', 'N/A'):<15} "
                f"{metrics.get('json_validity_rate', 'N/A'):<12}"
            )

        # Per-field breakdown for clean test
        if "per_field_exact_match" in clean:
            print(f"\n{'Field':<20} {'Clean Accuracy':<18} {'Notes'}")
            print("-" * 55)
            for field, acc in clean["per_field_exact_match"].items():
                note = ""
                if acc < 0.5:
                    note = "[!] Needs improvement"
                elif acc > 0.9:
                    note = "[OK] Strong"
                print(f"{field:<20} {acc:<18.3f} {note}")

        print("\n" + "=" * 70)
