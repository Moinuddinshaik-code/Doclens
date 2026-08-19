"""
LoRA/QLoRA Fine-Tuning Script for DocLens.

Fine-tunes Qwen2-VL-2B-Instruct with LoRA adapters on synthetic document images
for structured field extraction.

Training flow:
    1. Load base model with 4-bit quantization (QLoRA)
    2. Attach LoRA adapters to attention projections
    3. Prepare dataset: (image, prompt) → target JSON string
    4. Train with gradient accumulation (effective batch size 8)
    5. Log loss curves + save LoRA adapter weights

Key concepts (interview talking points):

    Teacher forcing:
        During training, the model receives the correct previous tokens at each
        step (not its own predictions). This is standard for autoregressive LMs.
        The loss is cross-entropy on the NEXT token prediction.

    Loss masking:
        We only compute loss on the ASSISTANT tokens (the JSON output), not on
        the system/user prompt tokens. The model should learn to generate the
        correct JSON, not to predict the next word of the prompt.

    Gradient accumulation:
        With batch_size=1 and gradient_accumulation_steps=8, we accumulate
        gradients across 8 forward passes before doing one optimizer step.
        Effective batch size = 1 × 8 = 8. This lets us simulate larger batches
        without needing the GPU memory for them.

    Learning rate for LoRA:
        LoRA typically uses 2-10x higher learning rate than full fine-tuning
        because we're updating far fewer parameters and the LoRA path has a
        scaling factor (alpha/r) that reduces the effective step size.

Usage:
    python training/train.py
    python training/train.py --config training/config.yaml
    python training/train.py --epochs 5 --lr 1e-4
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml
from PIL import Image


def load_training_data(data_dir: str, document_type: str = "receipt") -> list[dict]:
    """Load training data from a directory of image + JSON pairs.

    Each training example consists of:
    - An image file (receipt_XXXX.png)
    - A metadata JSON file (receipt_XXXX.json) with ground truth fields

    These are converted to the chat format expected by Qwen2-VL.

    Args:
        data_dir: Directory containing image/JSON pairs
        document_type: Type of document for prompt construction

    Returns:
        List of training examples as dicts with "messages" and "image_path"
    """
    from doclens.prompts import build_training_messages

    json_files = sorted(Path(data_dir).glob("*.json"))
    examples = []

    for json_path in json_files:
        with open(json_path, "r") as f:
            metadata = json.load(f)

        image_path = metadata.get("image_path", "")
        if not os.path.exists(image_path):
            image_filename = os.path.basename(image_path)
            image_path = os.path.join(data_dir, image_filename)

        if not os.path.exists(image_path):
            continue

        # Get ground truth fields as JSON string
        fields = metadata.get("fields", {})
        ground_truth_json = json.dumps(fields, indent=2)

        # Build chat messages
        messages = build_training_messages(
            document_type=document_type,
            ground_truth_json=ground_truth_json,
            image_path=image_path,
        )

        examples.append({
            "messages": messages,
            "image_path": image_path,
        })

    return examples


class DocLensDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for DocLens training.

    Handles loading images and preparing inputs in the format expected by
    Qwen2-VL's processor (tokenizer + image processor).
    """

    def __init__(self, examples: list[dict], processor):
        """Initialize dataset.

        Args:
            examples: List of training examples from load_training_data
            processor: Qwen2-VL processor (tokenizer + image processor)
        """
        self.examples = examples
        self.processor = processor

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "messages": example["messages"],
            "image_path": example["image_path"],
        }


def collate_fn(batch, processor):
    """Custom collate function for VLM training.

    This is more complex than standard text collation because we need to:
    1. Load and preprocess images
    2. Apply the chat template to get input text
    3. Process image + text together through the processor
    4. Create labels with loss masking on the prompt tokens

    Args:
        batch: List of dataset items
        processor: Qwen2-VL processor

    Returns:
        Dict with input_ids, attention_mask, pixel_values, labels
    """
    texts = []
    images = []

    for item in batch:
        messages = item["messages"]
        image_path = item["image_path"]

        # Apply chat template (converts messages to model input format)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)

        # Load image
        img = Image.open(image_path).convert("RGB")
        images.append(img)

    # Process all items together
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
        truncation=True,
    )

    # Create labels (same as input_ids, but with -100 for prompt tokens)
    # The model should only be penalized for generating the wrong JSON output,
    # not for "predicting" the prompt tokens
    labels = inputs["input_ids"].clone()

    # Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Find the assistant response start position and mask everything before it
    # This is a heuristic — look for the assistant token marker
    for i in range(len(batch)):
        # Find where the assistant response starts
        # In Qwen2-VL format, the assistant content follows a specific token pattern
        assistant_content = batch[i]["messages"][-1]["content"]
        assistant_tokens = processor.tokenizer.encode(
            assistant_content, add_special_tokens=False
        )
        input_len = len(inputs["input_ids"][i])
        assistant_len = len(assistant_tokens)

        # Mask all tokens except the last `assistant_len` tokens (the JSON output)
        # This is approximate — works for most cases
        mask_end = max(0, input_len - assistant_len)
        labels[i, :mask_end] = -100

    inputs["labels"] = labels
    return inputs


def train(config_path: Optional[str] = None, **overrides):
    """Run LoRA/QLoRA fine-tuning.

    Args:
        config_path: Path to config.yaml
        **overrides: Override specific config values (e.g., epochs=5)
    """
    from doclens.model import load_model_for_training, load_config
    from transformers import TrainingArguments, Trainer

    # Load config
    config = load_config(config_path)
    training_config = config.get("training", {})
    data_config = config.get("data", {})

    # Apply overrides
    for key, value in overrides.items():
        if key in training_config:
            training_config[key] = value

    print("=" * 60)
    print("   DocLens LoRA/QLoRA Fine-Tuning")
    print("=" * 60)

    # 1. Load model with LoRA adapters
    print("\n[1/4] Loading model with LoRA adapters...")
    model, processor, peft_config = load_model_for_training(config_path=config_path)

    # Ensure pad token exists
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 2. Load datasets
    print("\n[2/4] Loading datasets...")
    train_dir = data_config.get("train_dir", "data/generated/train")
    val_dir = data_config.get("val_dir", "data/generated/val")
    document_type = data_config.get("document_type", "receipt")

    train_examples = load_training_data(train_dir, document_type)
    val_examples = load_training_data(val_dir, document_type)

    print(f"  Train examples: {len(train_examples)}")
    print(f"  Val examples: {len(val_examples)}")

    if not train_examples:
        print("ERROR: No training examples found. Generate data first:")
        print("  python data/synthetic_generator.py")
        sys.exit(1)

    train_dataset = DocLensDataset(train_examples, processor)
    val_dataset = DocLensDataset(val_examples, processor)

    # 3. Configure training
    print("\n[3/4] Configuring training...")
    output_dir = training_config.get("output_dir", "training/checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    # Determine precision
    use_fp16 = training_config.get("fp16", False)
    use_bf16 = training_config.get("bf16", True)

    # Check GPU capability
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:  # Pre-Ampere (e.g., T4)
            use_bf16 = False
            use_fp16 = True
            print("  Note: GPU does not support bf16, using fp16")

    import inspect
    sig_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    raw_args = {
        "output_dir": output_dir,
        "num_train_epochs": training_config.get("num_epochs", 3),
        "per_device_train_batch_size": training_config.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": training_config.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", 8),
        "learning_rate": training_config.get("learning_rate", 2e-4),
        "weight_decay": training_config.get("weight_decay", 0.01),
        "lr_scheduler_type": training_config.get("lr_scheduler_type", "cosine"),
        "fp16": use_fp16,
        "bf16": use_bf16,
        "logging_steps": training_config.get("logging_steps", 10),
        "eval_steps": training_config.get("eval_steps", 50),
        "save_strategy": training_config.get("save_strategy", "epoch"),
        "save_total_limit": training_config.get("save_total_limit", 2),
        "load_best_model_at_end": training_config.get("load_best_model_at_end", True),
        "metric_for_best_model": training_config.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": training_config.get("greater_is_better", False),
        "report_to": training_config.get("report_to", "none"),
        "seed": training_config.get("seed", 42),
        "dataloader_num_workers": training_config.get("dataloader_num_workers", 0),
        "remove_unused_columns": False,
        "max_grad_norm": training_config.get("max_grad_norm", 1.0),
        "logging_dir": os.path.join(output_dir, "logs"),
    }

    if "warmup_ratio" in sig_params:
        raw_args["warmup_ratio"] = float(training_config.get("warmup_ratio", 0.1))
    elif "warmup_steps" in sig_params:
        raw_args["warmup_steps"] = 10

    if "eval_strategy" in sig_params:
        raw_args["eval_strategy"] = training_config.get("eval_strategy", "steps")
    elif "evaluation_strategy" in sig_params:
        raw_args["evaluation_strategy"] = training_config.get("eval_strategy", "steps")

    # Filter to parameters recognized by the installed transformers version
    valid_args = {k: v for k, v in raw_args.items() if k in sig_params}
    training_args = TrainingArguments(**valid_args)

    # Create custom data collator
    def custom_collate(batch):
        return collate_fn(batch, processor)

    # 4. Train
    print("\n[4/4] Starting training...")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")
    print(f"  Output: {output_dir}")

    start_time = time.time()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_collate,
    )

    # Train!
    train_result = trainer.train()

    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed / 60:.1f} minutes")

    # Save LoRA adapter weights (NOT the full model — just the small adapter)
    adapter_dir = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    print(f"  LoRA adapter saved to: {adapter_dir}")

    # Save training metrics
    metrics = train_result.metrics
    metrics["training_time_seconds"] = round(elapsed, 1)

    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Training metrics saved to: {metrics_path}")

    # Log final stats
    print("\n" + "=" * 60)
    print("   Training Summary")
    print("=" * 60)
    print(f"  Final train loss: {metrics.get('train_loss', 'N/A')}")
    print(f"  Training time: {elapsed / 60:.1f} minutes")
    print(f"  Adapter size: {sum(f.stat().st_size for f in Path(adapter_dir).rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")
    print(f"\n  Next step: Evaluate with")
    print(f"    python -c \"from doclens.model import load_model; ...\"")
    print(f"    or use the evaluation harness in doclens/evaluator.py")

    return train_result


# ─── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocLens LoRA/QLoRA Fine-Tuning")
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to config.yaml (default: training/config.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")

    args = parser.parse_args()

    overrides = {}
    if args.epochs:
        overrides["num_epochs"] = args.epochs
    if args.lr:
        overrides["learning_rate"] = args.lr
    if args.batch_size:
        overrides["per_device_train_batch_size"] = args.batch_size

    train(config_path=args.config, **overrides)
