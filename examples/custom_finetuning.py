"""
DocLens — Example: Custom Fine-Tuning Guide

Demonstrates how to customize the LoRA fine-tuning process:
- Adjust hyperparameters
- Change target modules
- Use different document types
- Monitor training

Usage:
    python examples/custom_finetuning.py
"""

import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def example_inspect_modules():
    """Step 1: Inspect model architecture to find LoRA target modules.

    Before fine-tuning, you need to know the exact layer names in the model.
    This is model-specific — different VLMs have different naming conventions.

    Run this once to discover available modules:
    """
    print("=" * 60)
    print("Step 1: Inspecting Model Architecture")
    print("=" * 60)
    print()
    print("To find LoRA target modules, run:")
    print('  python -c "from doclens.model import inspect_model_modules; inspect_model_modules()"')
    print()
    print("This will print all projection/linear layers in the model.")
    print("Common targets for Qwen2-VL:")
    print("  - q_proj (query projection)")
    print("  - v_proj (value projection)")
    print("  - k_proj (key projection)")
    print("  - o_proj (output projection)")
    print("  - gate_proj, up_proj, down_proj (FFN layers)")
    print()
    print("More target modules = more capacity but slower training.")
    print("Start with q_proj + v_proj, add more only if needed.")


def example_custom_config():
    """Step 2: Create a custom training configuration.

    Modify these parameters based on your setup:
    """
    print("\n" + "=" * 60)
    print("Step 2: Custom Training Configuration")
    print("=" * 60)

    config = {
        "model": {
            "name": "Qwen/Qwen2-VL-2B-Instruct",
            "quantization": "4bit",  # "4bit" for T4, "8bit" or "none" if more VRAM
        },
        "lora": {
            "r": 16,           # Rank: 8 (simpler tasks) to 64 (complex tasks)
            "lora_alpha": 32,  # Usually 2× rank
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        "training": {
            "num_epochs": 3,
            "learning_rate": 2e-4,
            "gradient_accumulation_steps": 8,
            # Key: watch val loss for overfitting
            # If val loss increases while train loss decreases → reduce epochs or increase dropout
        },
    }

    print(json.dumps(config, indent=2))
    print()
    print("Save this as training/config.yaml to use it.")


def example_training_command():
    """Step 3: Run training with overrides."""
    print("\n" + "=" * 60)
    print("Step 3: Training Commands")
    print("=" * 60)
    print()
    print("# Default training (uses config.yaml):")
    print("  python training/train.py")
    print()
    print("# Override specific parameters:")
    print("  python training/train.py --epochs 5 --lr 1e-4")
    print()
    print("# Use a custom config file:")
    print("  python training/train.py --config my_config.yaml")
    print()
    print("# Training on Colab (T4 GPU):")
    print("  # 1. Upload project to Colab or mount Google Drive")
    print("  # 2. Install: !pip install -e '.[dev]'")
    print("  # 3. Generate data: !python data/synthetic_generator.py")
    print("  # 4. Train: !python training/train.py")


def example_evaluation():
    """Step 4: Evaluate and compare."""
    print("\n" + "=" * 60)
    print("Step 4: Evaluation")
    print("=" * 60)
    print()
    print("After training, evaluate to fill in the benchmark table:")
    print()
    print("""
from doclens import DocLensExtractor, DocLensEvaluator, load_model

# Load fine-tuned model
model, processor = load_model(
    lora_adapter_path="training/checkpoints/lora_adapter",
    quantization="4bit",
)

extractor = DocLensExtractor(model, processor)
evaluator = DocLensEvaluator(extractor)

# Full benchmark report
report = evaluator.full_benchmark_report(
    clean_test_dir="data/generated/clean_test",
    adversarial_test_dir="data/generated/adversarial_test",
    output_path="benchmark_report.json",
)
""")


def main():
    print("============================================================")
    print("           DocLens -- Custom Fine-Tuning Guide             ")
    print("============================================================")

    example_inspect_modules()
    example_custom_config()
    example_training_command()
    example_evaluation()

    print("\n" + "=" * 60)
    print("Happy fine-tuning!")
    print("=" * 60)


if __name__ == "__main__":
    main()
