"""BitLinear: 1.58-bit ternary quantization layer for PyTorch.

Self-contained implementation of the BitLinear layer from "The Era of 1-bit
LLMs: All Large Language Models are in 1.58 Bits" (Ma et al., 2024).

Weights are quantized to ternary {-1, 0, +1} during forward pass while
maintaining full-precision latent weights for gradient-based training via
Straight-Through Estimator (STE).

References:
    arXiv:2402.17764 — BitNet b1.58
    arXiv:2412.06464 — Gated Delta Networks
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Applied before each BitLinear layer to stabilize training with ternary
    weights. Preferred over LayerNorm because it has no mean-centering step,
    which preserves the symmetric ternary weight distribution.

    Args:
        dim: Feature dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability, then restore dtype.
        x_fp32 = x.float()
        norm = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * norm).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------

def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with Straight-Through Estimator.

    Forward pass returns ``round(x)``.
    Backward pass passes gradients through unmodified (as if round were the
    identity function).
    """
    return x + (x.round() - x).detach()


def quantize_weights_ternary(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight tensor to ternary {-1, 0, +1}.

    The scale factor is ``mean(|w|)`` (per-tensor), following the BitNet b1.58
    paper. The STE-enabled quantization allows gradients to flow back to the
    full-precision latent weights during training.

    Args:
        w: Weight tensor (out_features, in_features).

    Returns:
        Tuple of (ternary_weights, scale) where ``w ≈ ternary_weights * scale``.
    """
    scale = w.abs().mean().clamp(min=1e-5)
    w_normalized = w / scale
    w_ternary = ste_round(w_normalized).clamp(-1, 1)
    return w_ternary, scale


def quantize_activations_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to 8-bit using per-token absmax scaling.

    Each token (last dimension) is independently scaled to the [-127, 127]
    range.

    Args:
        x: Activation tensor (..., features).

    Returns:
        Tuple of (quantized_activations, scale) where
        ``x ≈ quantized_activations * scale / 127``.
    """
    Qb = 127  # 2^(8-1) - 1
    scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    x_int = (x * Qb / scale).round().clamp(-Qb, Qb)
    return x_int, scale


# ---------------------------------------------------------------------------
# BitLinear layer
# ---------------------------------------------------------------------------

class BitLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with 1.58-bit ternary weights.

    During **training** the module maintains full-precision (FP32) latent
    weights.  The forward pass quantizes weights to ternary and activations to
    int8 via Straight-Through Estimator so gradients flow to the latent
    weights.

    During **inference** (after export) only ternary weights are needed,
    enabling computation with additions/subtractions only — no multiplications.

    Per the BitNet spec:
      * No bias term.
      * Per-layer RMSNorm on the input.
      * Activations quantized to 8-bit absmax.
      * Weights quantized to ternary via ``round(w / mean(|w|))``.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Latent full-precision weights (training target).
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Per-layer RMSNorm (required by BitNet b1.58 spec).
        self.rms_norm = RMSNorm(in_features)

        # No bias per BitNet specification.
        self.register_parameter("bias", None)

        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

    # Use float32 inside BitLinear even when autocast is active, because
    # the STE quantization math needs full precision.
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: RMSNorm → quantize → matmul → rescale.

        The pipeline mirrors the BitNet b1.58 paper (Fig. 2):
        ``Input → RMSNorm → ActivationQuant(8-bit) → WeightQuant(ternary) → MatMul → Rescale``
        """
        # 1. Pre-layer normalization.
        x = self.rms_norm(x)

        # 2. Quantize weights (ternary with STE).
        #    Forward value = w_ternary * w_scale  (quantized + rescaled).
        #    Backward gradient flows to self.weight  (latent FP32).
        w_ternary, w_scale = quantize_weights_ternary(self.weight)
        w_effective = self.weight + (w_ternary * w_scale - self.weight).detach()

        # 3. Quantize activations (int8 with STE).
        x_int, x_scale = quantize_activations_int8(x)
        x_effective = x + (x_int * x_scale / 127.0 - x).detach()

        # 4. Matrix multiplication.
        return F.linear(x_effective, w_effective)

    # ------------------------------------------------------------------
    # Conversion utilities
    # ------------------------------------------------------------------

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> BitLinear:
        """Create a BitLinear layer from an existing ``nn.Linear``.

        Copies the weight tensor as the initial latent weights.  Any bias is
        discarded (ternary quantization is incompatible with bias).

        Args:
            linear: Source nn.Linear module.

        Returns:
            New BitLinear module with copied weights.
        """
        layer = cls(linear.in_features, linear.out_features)
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
        return layer

    def export_ternary(self) -> dict[str, torch.Tensor]:
        """Export ternary weights + scale for inference deployment.

        Returns a dict with:
          * ``"weight_ternary"``: int8 tensor of {-1, 0, +1}.
          * ``"weight_scale"``: float32 scalar scale factor.
        """
        with torch.no_grad():
            w_ternary, w_scale = quantize_weights_ternary(self.weight)
        return {
            "weight_ternary": w_ternary.to(torch.int8),
            "weight_scale": w_scale.float(),
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias=False, quant=ternary_1.58bit"
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_ternary_params(model: nn.Module) -> dict[str, int | float]:
    """Count parameters by quantization type across a model.

    Returns:
        Dict with keys: ``bitlinear``, ``fp_linear``, ``other``, ``total``,
        ``bitlinear_pct``, ``est_size_mb``.
    """
    bitlinear_params = 0
    fp_linear_params = 0

    for module in model.modules():
        if isinstance(module, BitLinear):
            bitlinear_params += module.weight.numel()
        elif isinstance(module, nn.Linear):
            for p in module.parameters():
                fp_linear_params += p.numel()

    total = sum(p.numel() for p in model.parameters())
    other = total - bitlinear_params - fp_linear_params

    # Size estimate: ternary ≈ 0.25 bytes/param, FP16 ≈ 2 bytes/param.
    est_mb = (bitlinear_params * 0.25 + (fp_linear_params + other) * 2) / 1e6

    return {
        "bitlinear": bitlinear_params,
        "fp_linear": fp_linear_params,
        "other": other,
        "total": total,
        "bitlinear_pct": bitlinear_params / total * 100 if total else 0.0,
        "est_size_mb": est_mb,
    }
