"""
DocLens — Fine-tuned VLM for Document Field Extraction & Tamper Detection

A controllable, fine-tunable, self-hosted Vision-Language Model adapted via LoRA
to extract structured fields from document images, evaluated on noisy/adversarial
inputs, and optimized for realistic inference latency/memory.
"""

__version__ = "0.1.0"

from doclens.extractor import DocLensExtractor
from doclens.evaluator import DocLensEvaluator
from doclens.model import load_model, load_model_for_training

__all__ = [
    "__version__",
    "DocLensExtractor",
    "DocLensEvaluator",
    "load_model",
    "load_model_for_training",
]
