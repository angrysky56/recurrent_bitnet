"""RecurrentBitNet: Selective ternary quantization for hybrid linear-attention transformers."""

from src.bitlinear import BitLinear, RMSNorm, count_ternary_params
from src.surgery import convert_model, SurgeryConfig, surgical_report
from src.distill import DistillationTrainer, DistillationConfig

__all__ = [
    "BitLinear",
    "RMSNorm",
    "count_ternary_params",
    "convert_model",
    "SurgeryConfig",
    "surgical_report",
    "DistillationTrainer",
    "DistillationConfig",
]
