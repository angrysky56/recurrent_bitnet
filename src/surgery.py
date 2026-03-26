"""Model surgery: Convert Qwen3.5 DeltaNet layers to BitLinear.

Provides tools to selectively replace nn.Linear modules in Gated DeltaNet
layers with BitLinear (ternary) while preserving Full Attention layers at
full precision.  The hybrid-precision strategy exploits DeltaNet's recurrent
state accumulation to tolerate quantization noise.

Usage::

    from src.surgery import convert_model, SurgeryConfig, surgical_report

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
    report = convert_model(model, SurgeryConfig(aggression="standard"))
    print(surgical_report(model, report))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from torch import nn

from src.bitlinear import BitLinear, count_ternary_params

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Qwen3.5 layer pattern
# ---------------------------------------------------------------------------
# 6 × (3 × [Gated DeltaNet → FFN] → 1 × [Full Attention → FFN]) = 24 layers
#
# DeltaNet indices: 0,1,2,  4,5,6,  8,9,10,  12,13,14,  16,17,18,  20,21,22
# Attention indices: 3,  7,  11,  15,  19,  23

NUM_LAYERS = 24
NUM_BLOCKS = 6
DELTANET_PER_BLOCK = 3
ATTN_POSITION_IN_BLOCK = 3  # 0-indexed within each block of 4


def _build_layer_map(num_layers: int = NUM_LAYERS) -> dict[int, str]:
    """Build a mapping of layer index → type for the 3:1 pattern."""
    layer_map: dict[int, str] = {}
    for i in range(num_layers):
        if i % 4 == ATTN_POSITION_IN_BLOCK:
            layer_map[i] = "attention"
        else:
            layer_map[i] = "deltanet"
    return layer_map


LAYER_MAP = _build_layer_map()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SurgeryConfig:
    """Configuration for selective BitLinear conversion.

    Attributes:
        aggression: Conversion depth —
            ``"conservative"``: DeltaNet attention projections only (~30% compression).
            ``"standard"``: DeltaNet attention + FFN projections (~55% compression).
            ``"aggressive"``: Standard + attention-layer FFN (~70% compression).
        excluded_names: Linear module name fragments to never convert
            (e.g. ``["lm_head"]``).
        force_layer_map: Override automatic layer-type detection with a custom
            ``{layer_idx: "deltanet"|"attention"}`` mapping.
    """

    aggression: str = "standard"
    excluded_names: list[str] = field(
        default_factory=lambda: ["lm_head", "embed_tokens"]
    )
    force_layer_map: dict[int, str] | None = None

    def __post_init__(self) -> None:
        valid = {"conservative", "standard", "aggressive"}
        if self.aggression not in valid:
            raise ValueError(
                f"aggression must be one of {valid}, got {self.aggression!r}"
            )


# ---------------------------------------------------------------------------
# Layer identification
# ---------------------------------------------------------------------------

def identify_layer_types(
    model: nn.Module,
    config: SurgeryConfig | None = None,
) -> dict[int, dict]:
    """Inspect model layers and classify as DeltaNet or Attention.

    Uses three strategies in priority order:
      1. ``config.force_layer_map`` if provided.
      2. Class-name heuristic (looks for "DeltaNet" in module class name).
      3. Positional fallback (3:1 repeating pattern).

    Returns:
        Mapping of ``{layer_idx: {"type": str, "class": str, "linears": list}}``.
    """
    config = config or SurgeryConfig()
    layers = _get_model_layers(model)
    result: dict[int, dict] = {}

    for idx, layer in enumerate(layers):
        # --- determine type ---
        if config.force_layer_map and idx in config.force_layer_map:
            layer_type = config.force_layer_map[idx]
        elif _class_name_contains(layer, "deltanet", "delta_net"):
            layer_type = "deltanet"
        elif _class_name_contains(layer, "attention", "attn"):
            # Only mark as "attention" if NOT also matching deltanet.
            layer_type = "attention"
        else:
            layer_type = LAYER_MAP.get(idx, "unknown")

        # --- enumerate linears ---
        linears = []
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                linears.append(name)

        result[idx] = {
            "type": layer_type,
            "class": type(layer).__name__,
            "linears": linears,
        }

    return result


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

# Name fragments that identify attention-projection vs. FFN linears.
_ATTN_PROJ_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}
_FFN_PROJ_NAMES = {"gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"}


def _should_convert(
    layer_type: str,
    linear_name: str,
    config: SurgeryConfig,
) -> bool:
    """Decide whether a specific linear module should be converted."""
    # Never convert excluded names.
    for excl in config.excluded_names:
        if excl in linear_name:
            return False

    basename = linear_name.split(".")[-1]
    is_attn = basename in _ATTN_PROJ_NAMES
    is_ffn = basename in _FFN_PROJ_NAMES

    if layer_type == "deltanet":
        if config.aggression == "conservative":
            return is_attn
        # standard and aggressive both convert DeltaNet attn + ffn.
        return is_attn or is_ffn

    if layer_type == "attention":
        if config.aggression == "aggressive":
            return is_ffn  # only FFN in attention layers
        return False

    # Unknown layer type — skip.
    return False


def _replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a (possibly nested) named submodule."""
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def convert_model(
    model: nn.Module,
    config: SurgeryConfig | None = None,
) -> dict:
    """Convert targeted nn.Linear modules to BitLinear in-place.

    Args:
        model: A HuggingFace causal LM (e.g. ``Qwen3ForCausalLM``).
        config: Surgery configuration.

    Returns:
        Conversion report dict with ``converted`` and ``skipped`` lists.
    """
    config = config or SurgeryConfig()
    layer_info = identify_layer_types(model, config)
    layers = _get_model_layers(model)

    converted: list[str] = []
    skipped: list[str] = []

    for idx, layer in enumerate(layers):
        info = layer_info[idx]
        layer_type = info["type"]

        for linear_name in info["linears"]:
            full_name = f"layers.{idx}.{linear_name}"

            if _should_convert(layer_type, linear_name, config):
                # Get the original nn.Linear.
                linear = _get_submodule(layer, linear_name)
                if not isinstance(linear, nn.Linear):
                    skipped.append(full_name)
                    continue

                # Replace with BitLinear.
                bit_linear = BitLinear.from_linear(linear)
                _replace_module(layer, linear_name, bit_linear)
                converted.append(full_name)
                logger.debug("Converted %s (%s)", full_name, layer_type)
            else:
                skipped.append(full_name)

    return {
        "converted": converted,
        "skipped": skipped,
        "num_converted": len(converted),
        "num_skipped": len(skipped),
        "config": config,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def surgical_report(model: nn.Module, report: dict) -> str:
    """Generate a human-readable report of the conversion."""
    stats = count_ternary_params(model)
    cfg: SurgeryConfig = report.get("config", SurgeryConfig())

    lines = [
        "",
        "=" * 62,
        "  SURGICAL REPORT — RecurrentBitNet V3",
        "=" * 62,
        f"  Aggression level:   {cfg.aggression}",
        f"  Modules converted:  {report['num_converted']}",
        f"  Modules skipped:    {report['num_skipped']}",
        "-" * 62,
        f"  BitLinear params:   {stats['bitlinear']:>12,}  "
        f"({stats['bitlinear_pct']:.1f}%)",
        f"  FP Linear params:   {stats['fp_linear']:>12,}",
        f"  Other params:       {stats['other']:>12,}",
        f"  Total params:       {stats['total']:>12,}",
        "-" * 62,
        f"  Est. inference size: {stats['est_size_mb']:.0f} MB",
        "=" * 62,
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_model_layers(model: nn.Module) -> nn.ModuleList:
    """Extract the layer list from a HuggingFace model.

    Tries common attribute paths: ``model.model.layers``, ``model.layers``,
    ``model.transformer.h``.
    """
    for path in ["model.layers", "layers", "transformer.h"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, (nn.ModuleList, list)):
                return obj
        except AttributeError:
            continue
    raise AttributeError(
        "Could not locate layer list. Provide layers manually."
    )


def _get_submodule(module: nn.Module, name: str) -> nn.Module:
    """Get a nested submodule by dot-separated name."""
    for part in name.split("."):
        module = getattr(module, part)
    return module


def _class_name_contains(module: nn.Module, *fragments: str) -> bool:
    """Check if any submodule's class name contains given fragments."""
    cls_name = type(module).__name__.lower()
    if any(f in cls_name for f in fragments):
        return True
    # Also check the attention submodule specifically.
    for _, child in module.named_children():
        child_cls = type(child).__name__.lower()
        if any(f in child_cls for f in fragments):
            return True
    return False
