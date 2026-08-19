# DocLens — Technical Specification (for implementation)

Hand this directly to Claude Code as a build spec. Sections are ordered so each one has what it needs from the ones before it.

---

## 1. Problem Statement

Identity and financial document verification systems (KYC onboarding, invoice processing, receipt reconciliation) need to extract structured fields from document images and flag potential tampering/forgery — under real-world conditions: variable lighting, camera angle, occlusion, compression artifacts, and deliberate manipulation.

Most extraction approaches fall into two buckets:
- **Classical OCR + rules/regex**: brittle to layout variation, no semantic understanding, no forgery reasoning
- **API-based VLM calls (GPT-4V, Gemini, etc.)**: no control over model behavior, can't be fine-tuned to a domain, ongoing cost per call, no offline/on-prem deployment option

**DocLens** builds a controllable, fine-tunable, self-hosted alternative: a small open-weight VLM adapted via LoRA to extract structured fields from document images, evaluated explicitly on noisy/adversarial inputs (not just clean data), and optimized for realistic inference latency/memory.

---

## 2. Goals

1. Fine-tune a small open-weight VLM (LoRA) to extract structured fields from document images (receipts as primary target; ID cards as stretch target)
2. Generate synthetic training/eval data with controllable ground truth and controllable noise/adversarial conditions
3. Build an evaluation harness that measures performance on clean AND adversarial/noisy test sets separately
4. Optimize inference (quantization, batching) and report latency/memory before vs after
5. Package as an installable, documented, open-source Python library

## 3. Non-goals (explicit scope boundaries)

- **Not** training a VLM from scratch — fine-tuning a pretrained model only
- **Not** using real user PII/identity documents — synthetic data only, for legal/privacy reasons (this is also the correct engineering call, not just a shortcut)
- **Not** building a production-scale distributed training pipeline — single-GPU (Colab/consumer GPU) scope
- **Not** covering every document type — receipts primary, one additional type (ID card layout) if time allows, everything else out of scope
- **Not** building a UI — CLI + Python API + notebook demo only

## 4. Success Criteria (definition of "working")

- LoRA fine-tuned model shows a measurable improvement over zero-shot baseline on the clean test set (field-level accuracy)
- The gap between clean-test and adversarial-test performance is measured and reported (this gap, and understanding it, is as important as the headline number)
- Quantized inference shows measurable latency/memory improvement over unquantized, with accuracy delta reported
- Package installs cleanly (`pip install -e .`) and the quickstart example runs end-to-end without manual intervention
- README contains a benchmark table with all of the above numbers

---

## 5. High-Level Architecture

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
                    │  Prompt: structured   │
                    │  JSON extraction      │
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

**Training-side pipeline (separate from inference above):**

```
Synthetic Template + Faker data
        │
        ▼
Document Renderer (PIL/HTML→image)
        │
        ▼
Augmentation Pipeline (albumentations:
blur, rotation, lighting, occlusion, noise, forgery edits)
        │
        ▼
Labeled Dataset (image, ground_truth_json)
        │
        ▼
Train/Val/Test/Adversarial-Test Split
        │
        ▼
LoRA Fine-Tuning Loop (PEFT + QLoRA 4-bit)
        │
        ▼
Fine-tuned Adapter Weights
```

---

## 6. Technical Reasoning (why each choice — include this in README, and be ready to defend it verbally)

**Base model: Qwen2-VL-2B-Instruct (primary choice), Florence-2-base (fallback)**
- Small enough to fine-tune on a single consumer GPU or free-tier Colab with 4-bit quantization
- Qwen2-VL has native visual grounding and strong OCR-adjacent pretraining, relevant to document tasks
- Florence-2 as fallback if Qwen2-VL proves too heavy for the available compute — it's smaller and purpose-built for dense visual tasks including OCR

**Fine-tuning method: LoRA (not full fine-tune)**
- Full fine-tuning a VLM requires far more compute/memory/time than a one-week timeline allows
- LoRA updates <1% of parameters, trains fast, and avoids catastrophic forgetting of the base model's general vision-language capability
- This is also the industry-standard approach for domain adaptation at this model scale — directly relevant to the "PEFT/LoRA" bonus point in the JD

**Synthetic data (not scraped/real documents)**
- Real ID/financial documents carry real PII — using them raises legal and privacy issues, and is the wrong engineering call even if convenient
- Synthetic generation gives exact ground truth (no manual labeling), and lets us control the distribution — including generating adversarial/forged examples systematically, which real data wouldn't let us do safely or exhaustively

**Quantization (QLoRA for training, GPTQ/bitsandbytes 4-bit for inference)**
- Directly addresses the JD's ask to "optimize models for accuracy, latency, memory, and inference cost"
- QLoRA (4-bit base + LoRA adapters) makes fine-tuning feasible on limited GPU memory
- Post-training quantization for inference demonstrates understanding of the train/serve tradeoff, not just the training side

**Template-matching / RAG-lite grounding (optional, if time allows)**
- Comparing extracted fields against known template structures reduces hallucination on ambiguous fields
- Ties the project to the RAG track in the JD (retrieval + grounding) without requiring a separate large project

---

## 7. Data Specification

**Document types:**
1. Receipts (primary) — fields: vendor_name, date, total_amount, tax_amount, line_items (list), payment_method
2. ID cards (stretch, only if Day 1–2 goes fast) — fields: full_name, id_number, date_of_birth, expiry_date, issuing_authority

**Synthetic generation approach:**
- Template-based rendering: PIL or HTML-to-image (via `imgkit`/`weasyprint`) with placeholder fields
- Field values generated via `Faker` library (names, dates, amounts, addresses) — ensures no real PII
- Layout variation: randomize font, font size, field position within template bounds, logo placement
- Ground truth stored alongside each image as JSON (exact field values used to render)

**Augmentation pipeline (via `albumentations`):**
- Geometric: rotation (±15°), perspective warp, random crop/skew
- Photometric: brightness/contrast shifts, Gaussian blur, motion blur, low-light simulation
- Occlusion: random rectangular occlusion (simulating finger/glare/fold)
- Noise: Gaussian noise, JPEG compression artifacts
- **Forgery simulation** (for tamper-detection subset): programmatically alter one field's rendered text (different font, misaligned baseline, mismatched color/resolution) after the "clean" version — this becomes ground truth for `is_tampered: true` examples

**Dataset splits (targets — adjust down if generation takes longer than expected):**
- Train: 500–800 synthetic examples
- Validation: 100 examples
- Clean test: 100 examples (same distribution as train, held out)
- Adversarial test: 50–100 examples (heavier augmentation + tamper subset)

**Schema (ground truth JSON per example):**
```json
{
  "image_path": "data/synthetic/receipt_0001.png",
  "document_type": "receipt",
  "fields": {
    "vendor_name": "string",
    "date": "YYYY-MM-DD",
    "total_amount": "number",
    "tax_amount": "number",
    "line_items": [{"item": "string", "price": "number"}],
    "payment_method": "string"
  },
  "is_tampered": false,
  "tamper_field": null,
  "augmentations_applied": ["blur", "rotation_8deg"]
}
```

---

## 8. Model & Training Specification

- **Base model:** `Qwen/Qwen2-VL-2B-Instruct` (HuggingFace)
- **Fine-tuning framework:** HuggingFace `transformers` + `peft` + `bitsandbytes` (QLoRA)
- **LoRA config (starting point, tune if needed):**
  - `r=16`, `lora_alpha=32`, `lora_dropout=0.05`
  - `target_modules`: attention projection layers (`q_proj`, `v_proj`, or model-specific equivalents — confirm exact module names against the loaded model's architecture before training)
- **Training hyperparameters (starting point):**
  - Learning rate: 1e-4 to 2e-4 (LoRA typically tolerates higher LR than full fine-tune)
  - Batch size: 1–2 with gradient accumulation (4–8 steps) to simulate larger effective batch on limited GPU memory
  - Epochs: 2–3 (watch val loss for overfitting given small synthetic dataset)
  - Precision: 4-bit base model (QLoRA) + bf16/fp16 LoRA adapter weights
- **Loss:** standard causal LM cross-entropy on the target JSON string tokens (teacher-forced on the expected structured output)
- **Prompt template (design this explicitly, keep consistent train/inference):**
  ```
  System: You are a document extraction assistant. Extract the following fields as JSON: {field_list}. If a field is not visible, use null.
  User: [image]
  Assistant: {ground_truth_json}
  ```

---

## 9. Evaluation Specification

**Metrics:**
- Field-level exact match accuracy (per field, and averaged)
- Field-level F1 for partial/fuzzy matches (e.g., minor OCR-style discrepancies on amounts/dates — define a tolerance, e.g., string edit distance threshold)
- JSON validity rate (does output parse as valid JSON matching schema at all)
- Tamper detection: precision/recall/F1 on the `is_tampered` binary flag
- Latency: ms per image (batch size 1, and batched if implemented)
- Memory: peak GPU memory during inference

**Comparisons to report (this is the core of your results section):**
1. Zero-shot base model vs LoRA fine-tuned — on clean test set
2. Zero-shot base model vs LoRA fine-tuned — on adversarial test set
3. Unquantized vs quantized inference — latency/memory, with accuracy delta
4. Per-augmentation-type breakdown (which noise type hurts accuracy most — this is the "failure analysis" the JD explicitly wants)

**Adversarial test set composition (be explicit about this in the README):**
- Blur-heavy subset
- Occlusion subset
- Extreme lighting subset
- Rotation/skew subset
- Tampered-field subset

---

## 10. Inference Pipeline Specification

- **Input:** image path or bytes, document_type (optional hint)
- **Steps:**
  1. Preprocess: resize to model's expected input resolution, normalize
  2. Construct prompt using the same template as training
  3. Forward pass through VLM (+ LoRA adapter) — generate JSON string
  4. Parse output string to JSON; if parsing fails, retry once with a stricter prompt or return a structured error
  5. Validate against expected schema (required fields present, types correct)
  6. Compute per-field confidence (token-level log-probs if accessible via the generation API, else a simpler heuristic like output consistency across 2 sampling runs)
  7. Tamper flag: either a dedicated classification head, or a prompt-based secondary query ("does this document show signs of tampering — inconsistent fonts, misaligned text, resolution mismatch?")
  8. Optional: generate a one-sentence natural-language explanation for any flagged issue
- **Optimization:**
  - Quantized inference via `bitsandbytes` 4-bit or `GPTQ`
  - Batch multiple images per forward pass where possible
  - KV-cache reuse if doing any multi-turn/multi-step reasoning
- **Output schema:**
```json
{
  "fields": {"vendor_name": "...", "total_amount": 42.50, "...": "..."},
  "confidence": {"vendor_name": 0.94, "total_amount": 0.88},
  "fraud_flag": false,
  "explanation": null,
  "latency_ms": 340
}
```

---

## 11. Repository Structure

```
doclens/
├── README.md
├── LICENSE (MIT)
├── requirements.txt
├── pyproject.toml
│
├── doclens/
│   ├── __init__.py
│   ├── model.py          # VLM loading, LoRA adapter attach/load, quantization config
│   ├── extractor.py       # inference pipeline (preprocess → generate → parse → validate)
│   ├── evaluator.py        # eval harness: metrics computation, report generation
│   ├── prompts.py          # prompt templates, kept in one place for train/infer consistency
│   └── utils.py             # schema validation, JSON parsing/repair helpers
│
├── data/
│   ├── synthetic_generator.py   # template rendering + Faker field generation
│   ├── augmentations.py           # albumentations pipeline, tamper simulation
│   ├── templates/                  # receipt/ID card template assets
│   └── eval_harness.py              # builds clean + adversarial test sets
│
├── training/
│   ├── train.py               # LoRA/QLoRA fine-tuning script
│   ├── config.yaml             # hyperparameters
│   └── logs/                    # loss curves, training metrics (gitignored, sample included)
│
├── examples/
│   ├── quickstart.ipynb         # Colab-friendly end-to-end demo
│   ├── extract_receipt.py
│   ├── verify_identity.py
│   └── custom_finetuning.py
│
└── tests/
    ├── test_extraction.py
    ├── test_eval_metrics.py
    ├── test_synthetic_data.py
    └── test_schema_validation.py
```

---

## 12. Environment & Dependencies

- Python 3.10+
- Core: `torch` (2.x), `transformers`, `peft`, `bitsandbytes`, `accelerate`
- Data: `Pillow`, `albumentations`, `faker`, `datasets`
- Eval: `evaluate`, `jsonschema`, `scikit-learn` (for precision/recall)
- Optional rendering: `weasyprint` or `imgkit` for HTML-to-image templates
- Dev: `pytest`, `black`, `ruff`
- **Compute target:** single GPU, 12–16GB VRAM minimum with 4-bit quantization (Colab T4 is workable; A100/similar preferred if available)

---

## 13. Implementation Phases (map to the 7-day plan)

| Phase | Scope | Maps to |
|---|---|---|
| 1 | Repo scaffold, synthetic data generator, template rendering | Days 1–2 |
| 2 | Baseline zero-shot eval on clean test set | Day 2 |
| 3 | LoRA/QLoRA training pipeline, run first fine-tune | Day 3 |
| 4 | Quantized inference, latency/memory benchmarking | Day 4 |
| 5 | Adversarial test set construction, full eval comparison | Day 4–5 |
| 6 | Documentation, examples, notebook, packaging | Day 6 |
| 7 | Tests, polish, README finalize, mock explanation run-through | Day 6–7 |

---

## 14. Open Implementation Decisions (resolve these during build, don't block on them upfront)

- Exact `target_modules` names for LoRA — depends on the specific Qwen2-VL/Florence-2 module naming; inspect the loaded model's `named_modules()` before finalizing config
- Confidence scoring approach — token log-probs if the generation API exposes them cleanly, else fall back to the multi-sample-consistency heuristic
- Whether tamper detection is a separate lightweight classifier head vs a prompt-based query — start with prompt-based (faster to implement), upgrade only if time allows

---

## 15. Definition of Done

- [ ] `pip install -e .` works cleanly from a fresh environment
- [ ] `examples/quickstart.ipynb` runs end-to-end on Colab without manual fixes
- [ ] `training/train.py` runs and produces a saved LoRA adapter + logged loss curve
- [ ] `doclens/evaluator.py` produces a benchmark report covering all comparisons in Section 9
- [ ] README contains: problem statement, architecture diagram, benchmark table, install/quickstart instructions, failure analysis section
- [ ] All files in `tests/` pass
- [ ] LICENSE, `.gitignore`, `requirements.txt` present and correct
