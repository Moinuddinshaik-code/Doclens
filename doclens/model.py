"""
Model loading and configuration for DocLens.

Handles VLM loading with quantization (bitsandbytes 4-bit/8-bit), LoRA adapter
attachment for training, and LoRA adapter loading for inference.

Architecture overview (interview talking points):

    Qwen2-VL-2B-Instruct architecture:
    ┌──────────────────┐
    │  Vision Encoder   │  ← ViT (Vision Transformer) processes image patches
    │  (ViT)           │    into visual token embeddings
    └────────┬─────────┘
             │ visual tokens
    ┌────────▼─────────┐
    │  Projection       │  ← Maps visual embeddings to language model's
    │  Layer            │    embedding dimension
    └────────┬─────────┘
             │
    ┌────────▼──────────────────┐
    │  Language Model            │  ← Transformer decoder (causal LM)
    │  (32 layers, ~2B params)  │    processes [visual + text] tokens
    │  ┌──────────────────┐     │
    │  │ Attention Block   │←── │── LoRA adapts Q, V projections HERE
    │  │  Q, K, V projections   │
    │  │  Multi-head attention  │
    │  │  Output projection     │
    │  └──────────────────┘     │
    │  ┌──────────────────┐     │
    │  │ FFN Block         │     │
    │  │  gate_proj, up_proj    │
    │  │  down_proj             │
    │  └──────────────────┘     │
    │  × 32 layers              │
    └───────────────────────────┘

    QLoRA memory math for 2B params:
    - Base model in fp16: 2B × 2 bytes = 4 GB
    - Base model in 4-bit: 2B × 0.5 bytes = 1 GB
    - LoRA adapters (r=16, 2 modules): ~20 MB in bf16
    - Optimizer states (Adam): ~40 MB (only for LoRA params)
    - Total training: ~2-3 GB (fits on T4 with room for batch)
"""

import os
from typing import Optional

import torch
import yaml


def load_config(config_path: Optional[str] = None) -> dict:
    """Load training/model config from YAML file.

    Args:
        config_path: Path to config.yaml. If None, uses default location.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "training", "config.yaml"
        )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_quantization_config(quantization: str = "4bit"):
    """Create a BitsAndBytesConfig for quantized model loading.

    Quantization reduces model memory by storing weights at lower precision.

    4-bit NormalFloat (NF4):
        - Designed for normally-distributed weights (which neural nets have)
        - Information-theoretically optimal 4-bit data type for Gaussian weights
        - Double quantization: quantizes the quantization constants too, saving
          an additional ~0.5 GB for a 2B model
        - Compute dtype bf16: dequantized weights are cast to bf16 for matrix ops

    8-bit (LLM.int8()):
        - Less aggressive compression, ~2x memory vs fp16 (vs 4x for 4-bit)
        - Handles outlier features via mixed-precision decomposition
        - Slightly better accuracy than 4-bit, but uses more memory

    Args:
        quantization: "4bit", "8bit", or "none"

    Returns:
        BitsAndBytesConfig or None
    """
    if quantization == "none":
        return None

    from transformers import BitsAndBytesConfig

    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # NormalFloat4 — optimal for Gaussian weights
            bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bf16 after dequantization
            bnb_4bit_use_double_quant=True,        # Quantize the quantization constants
        )
    elif quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        raise ValueError(f"Unknown quantization: {quantization}. Use '4bit', '8bit', or 'none'.")


def load_model(
    model_name: Optional[str] = None,
    quantization: Optional[str] = None,
    lora_adapter_path: Optional[str] = None,
    device_map: str = "auto",
    config_path: Optional[str] = None,
):
    """Load a VLM for inference, optionally with quantization and LoRA adapter.

    This is the main entry point for loading the model at inference time.

    Args:
        model_name: HuggingFace model ID (e.g., "Qwen/Qwen2-VL-2B-Instruct")
        quantization: "4bit", "8bit", or "none"
        lora_adapter_path: Path to saved LoRA adapter weights (None = base model only)
        device_map: Device placement strategy ("auto", "cuda:0", etc.)
        config_path: Path to config.yaml (overridden by explicit args)

    Returns:
        Tuple of (model, processor)
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    # Load config for defaults
    config = load_config(config_path)
    model_config = config.get("model", {})

    if model_name is None:
        model_name = model_config.get("name", "Qwen/Qwen2-VL-2B-Instruct")
    if quantization is None:
        quantization = model_config.get("quantization", "4bit")

    # Get quantization config
    bnb_config = get_quantization_config(quantization)

    # Determine torch dtype
    dtype_str = model_config.get("torch_dtype", "bfloat16")
    torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    print(f"Loading model: {model_name}")
    print(f"  Quantization: {quantization}")
    print(f"  Dtype: {dtype_str}")

    # Load base model
    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, **model_kwargs
    )

    # Load processor (tokenizer + image processor)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Load LoRA adapter if specified
    if lora_adapter_path is not None:
        if os.path.exists(lora_adapter_path):
            from peft import PeftModel
            print(f"  Loading LoRA adapter from: {lora_adapter_path}")
            model = PeftModel.from_pretrained(model, lora_adapter_path)
            model = model.merge_and_unload()  # Merge for faster inference
            print("  LoRA adapter merged into base model")
        else:
            print(f"  Note: Adapter directory '{lora_adapter_path}' not found.")
            print("  Running in zero-shot base model mode.")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model loaded successfully ✓")

    return model, processor


def load_model_for_training(
    model_name: Optional[str] = None,
    config_path: Optional[str] = None,
):
    """Load a VLM with LoRA adapters attached for training.

    This sets up:
    1. Base model in 4-bit quantization (QLoRA)
    2. LoRA adapters on specified target modules
    3. Gradient checkpointing for memory efficiency

    Args:
        model_name: HuggingFace model ID
        config_path: Path to config.yaml

    Returns:
        Tuple of (model_with_lora, processor, peft_config)
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    config = load_config(config_path)
    model_config = config.get("model", {})
    lora_config = config.get("lora", {})

    if model_name is None:
        model_name = model_config.get("name", "Qwen/Qwen2-VL-2B-Instruct")

    quantization = model_config.get("quantization", "4bit")
    bnb_config = get_quantization_config(quantization)

    dtype_str = model_config.get("torch_dtype", "bfloat16")
    torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    print(f"Loading model for training: {model_name}")
    print(f"  Quantization: {quantization}")

    # Load base model
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, **model_kwargs
    )

    # Prepare for k-bit training (freeze base, enable gradient computation on quantized model)
    if quantization != "none":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Determine target modules
    # The config specifies target module names, but we should verify they exist
    target_modules = lora_config.get("target_modules", ["q_proj", "v_proj"])

    # Verify target modules exist in the model
    all_module_names = set()
    for name, _ in model.named_modules():
        parts = name.split(".")
        all_module_names.update(parts)

    valid_targets = [t for t in target_modules if t in all_module_names]
    if len(valid_targets) < len(target_modules):
        missing = set(target_modules) - set(valid_targets)
        print(f"  Warning: Target modules not found: {missing}")
        print(f"  Available modules containing 'proj': ")
        proj_modules = [n for n in all_module_names if "proj" in n.lower()]
        print(f"    {proj_modules}")
        if not valid_targets:
            valid_targets = proj_modules[:2]  # Fallback to first 2 projection layers

    print(f"  LoRA target modules: {valid_targets}")

    # Create LoRA config
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=valid_targets,
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )

    # Attach LoRA adapters
    model = get_peft_model(model, peft_config)

    # Print trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    return model, processor, peft_config
    
    


def inspect_model_modules(model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
    """Print all module names in a model — useful for finding LoRA target modules.

    Run this once to discover the exact module names before training:
        python -c "from doclens.model import inspect_model_modules; inspect_model_modules()"

    Args:
        model_name: HuggingFace model ID
    """
    from transformers import Qwen2VLForConditionalGeneration

    print(f"Loading {model_name} to inspect architecture...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print("\n=== All named modules ===")
    proj_modules = []
    for name, module in model.named_modules():
        if "proj" in name.lower() or "linear" in name.lower():
            proj_modules.append((name, type(module).__name__))

    print("\nProjection/Linear layers (candidates for LoRA):")
    for name, typename in proj_modules:
        print(f"  {name} ({typename})")

    # Get unique short names
    short_names = set()
    for name, _ in proj_modules:
        short_names.add(name.split(".")[-1])

    print(f"\nUnique short names: {sorted(short_names)}")
    print("\nRecommended target_modules for LoRA:")
    recommended = [n for n in sorted(short_names) if "proj" in n]
    print(f"  {recommended}")

    del model
