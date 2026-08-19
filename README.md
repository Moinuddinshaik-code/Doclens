# DocLens

**Fine-tuned Vision-Language Model for Document Field Extraction & Tamper Detection**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Moinuddinshaik-code/Doclens/blob/main/examples/quickstart.ipynb)

DocLens is a controllable, fine-tunable, self-hosted VLM framework that extracts structured fields from document images (receipts, ID cards) and flags potential tampering — under real-world conditions: variable lighting, camera angle, occlusion, compression artifacts, and deliberate manipulation.

---

## 📌 Problem Statement

Identity and financial document verification systems (KYC onboarding, invoice processing, receipt reconciliation) need to extract structured fields from document images and flag potential forgery.

Most approaches fall into two buckets:
- **Classical OCR + rules/regex**: brittle to layout variation, no semantic understanding, no forgery reasoning.
- **API-based VLM calls (GPT-4V, Gemini)**: no control over model behavior, can't be fine-tuned to a domain, ongoing cost per call, no offline/on-prem deployment.

**DocLens** builds a third option: a small open-weight VLM (`Qwen2-VL-2B-Instruct`) adapted via LoRA to extract structured fields, evaluated explicitly on noisy/adversarial inputs, and optimized for realistic inference latency/memory.

---

## 🧬 Architecture

```
                    ┌─────────────────────┐
                    │   Input: Document    │
                    │   Image (JPEG/PNG)   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Preprocessing      │
                    │  (resize, normalize) │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  VLM Inference        │
                    │  (Qwen2-VL-2B +       │
                    │   LoRA adapter)       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Output Parsing +     │
                    │  Schema Validation    │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                │                             │
     ┌──────────▼───────────┐      ┌─────────▼──────────┐
     │  Confidence Scoring  │      │  Tamper/Fraud Flag  │
     │  (per-field)         │      │  Heuristic + Model  │
     └──────────┬───────────┘      └─────────┬───────────┘
                │                             │
                └──────────────┬──────────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Final Output:        │
                    │  {fields, confidence, │
                    │   fraud_flag,         │
                    │   explanation}        │
                    └───────────────────────┘
```

---

## 🛠 Tech Stack & Key Technical Decisions

| Decision | Why |
|---|---|
| **Qwen2-VL-2B** base model | Smallest VLM with native dynamic resolution — critical for documents where small text must remain legible. Fine-tunable on single GPU with QLoRA. |
| **LoRA (not full fine-tune)** | Updates <1% of parameters. Trains in minutes, avoids catastrophic forgetting, adapter is ~20 MB. Industry-standard for domain adaptation. |
| **QLoRA 4-bit** | NormalFloat4 quantization of frozen weights. Reduces training memory from ~16 GB to ~3 GB. Fits on free Colab T4 GPU. |
| **Synthetic data** | Exact ground truth (no labeling), controllable distribution, no PII, systematic adversarial/forged examples. |
| **Per-augmentation eval** | Answers "which noise type hurts accuracy most?" — failure analysis beyond clean benchmark datasets. |
| **Quantized inference** | 4-bit inference for 50–70% memory reduction with <2% accuracy drop. Reports latency/memory/accuracy tradeoff. |

---

## 🚀 Quick Start in Google Colab

Run the complete pipeline directly on Google Colab's free T4 GPU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Moinuddinshaik-code/Doclens/blob/main/examples/quickstart.ipynb)

```bash
# 1. Clone repository
git clone https://github.com/Moinuddinshaik-code/Doclens.git
cd Doclens

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Generate synthetic training data & adversarial test sets
python data/synthetic_generator.py
python data/eval_harness.py

# 4. Run QLoRA fine-tuning on GPU
python training/train.py --config training/config.yaml
```

---

## 📋 Example Usage

### Field Extraction & Tamper Check (CLI)

```bash
# Extract fields from a clean receipt image
python examples/extract_receipt.py data/generated/clean_test/receipt_0700.png --adapter training/checkpoints/lora_adapter

# Extract fields + run tamper detection on forged image
python examples/extract_receipt.py data/generated/adversarial_test/heavy/aug_heavy_0000.png --adapter training/checkpoints/lora_adapter --tamper
```

### Python API

```python
from doclens import load_model, DocLensExtractor, DocLensEvaluator

# Load model + adapter
model, processor = load_model(
    quantization="4bit",
    lora_adapter_path="training/checkpoints/lora_adapter"
)

extractor = DocLensExtractor(model, processor)

# Extract fields + fraud flag
result = extractor.extract_with_tamper("receipt.png", document_type="receipt")
print(result["fields"])
print(result["confidence"])
print(f"Tamper flag: {result['fraud_flag']}")
```

---

## 🧪 Running Unit Tests

DocLens includes 63 unit tests covering schema validation, JSON repair heuristics, string normalization, Levenshtein edit distance, synthetic data rendering, and evaluation metrics:

```bash
pytest tests/ -v
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
