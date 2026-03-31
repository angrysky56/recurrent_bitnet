#!/usr/bin/env python3
"""Generate RecurrentBitNet V2-SPR comparative notebook.

V3 — Machens refinement: context subspace split into temporal-context
(externally driven by iteration embeddings) and state-context (purely
recurrent, no external injection). Based on Machens, Romo & Brody,
J. Neuroscience 2010: functional but not anatomical separation of
"what" and "when" in prefrontal cortex.
"""

import json

cells = []


def md(source):
    cells.append(
        {
            "cell_type": "markdown",
            "id": f"md_{len(cells)}",
            "metadata": {},
            "source": source.split("\n"),
        }
    )


def code(source):
    src_lines = [line + "\n" for line in source.split("\n")]
    if src_lines:
        src_lines[-1] = src_lines[-1].rstrip("\n")
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": f"code_{len(cells)}",
            "metadata": {},
            "outputs": [],
            "source": src_lines,
        }
    )


# ═══════════════════════════════════════════════════════
# CELL 1: Title
# ═══════════════════════════════════════════════════════
md(
    """# RecurrentBitNet V2-SPR — Subspace-Partitioned Reasoning Experiment

**Hypothesis**: Content, temporal context, and reasoning state should occupy orthogonal
subspaces, interacting through sparse co-activation rather than dense entanglement.

---

### Architecture: Four Orthogonal Subspaces

| Subspace | % | d=768 | Drive Mechanism | Biological Analog |
|---|---|---|---|---|
| **Content** | 85% | 652 | Attention + FFN | MTL content neurons (Bausch et al.) |
| **Temporal context** | 6% | 46 | Iteration embeddings (external) | PFC time components (Machens et al.) |
| **State context** | 6% | 46 | Recurrent updates only | PFC stimulus components (Machens et al.) |
| **Conjunctive** | 3% | 24 | Binding network | Hippocampal conjunctive neurons |

### How to Run This Experiment

Run it **twice** — once with `use_spr = True`, once with `use_spr = False` — then
compare the probe tables.

**Key Measurements**:
1. Training loss convergence — do they learn equally well?
2. **Subspace probing** — can we decode iteration from content dims? state from temporal?
3. **DOC separation** — Machens-style variance decomposition within context subspace
4. **Maturity + state gate trajectories** — do the gates open at the right curriculum phases?

**Evidence sources**: Bausch et al. Nature 2026, Machens et al. J Neurosci 2010,
Kerce & Fox arXiv:2603.07461, Qwen3.5"""
)

# ═══════════════════════════════════════════════════════
# CELL 2: Setup
# ═══════════════════════════════════════════════════════
md("## 1. Setup")

code("""# Install dependencies
!pip install -q datasets transformers tqdm matplotlib

# Mount Google Drive for checkpoint persistence
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_BASE = '/content/drive/MyDrive'
LOCAL_BASE = '/content/checkpoints'
print("Drive mounted. Checkpoint dirs will be created after model config is set.")""")

# ═══════════════════════════════════════════════════════
# CELL 3: Environment
# ═══════════════════════════════════════════════════════
md("## 2. Environment & Device")

code("""import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, shutil, json
from dataclasses import dataclass, asdict, field
from tqdm.auto import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    props = torch.cuda.get_device_properties(0)
    VRAM_GB = props.total_mem / 1e9
    print(f"GPU: {props.name} — {VRAM_GB:.1f} GB VRAM")
    print(f"Compute Capability: {props.major}.{props.minor}")
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for matmul + cudnn")""")

# ═══════════════════════════════════════════════════════
# CELL 4: BitLinear
# ═══════════════════════════════════════════════════════
md("## 3. BitLinear (1.58-bit Ternary Quantization)")

code("""def ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()

def quantize_weights_ternary(w: torch.Tensor):
    scale = w.abs().mean().clamp(min=1e-5)
    w_normalized = w / scale
    w_ternary = ste_round(w_normalized).clamp(-1, 1)
    return w_ternary, scale

def quantize_activations_int8(x: torch.Tensor):
    Qb = 127
    scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    x_int = (x * Qb / scale).round().clamp(-Qb, Qb)
    return x_int, scale

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)

    @torch.amp.custom_fwd(device_type="cuda")
    def forward(self, x):
        w_ternary, w_scale = quantize_weights_ternary(self.weight)
        w_effective = self.weight + (w_ternary * w_scale - self.weight).detach()
        x_int, x_scale = quantize_activations_int8(x)
        x_effective = x + (x_int * x_scale / 127.0 - x).detach()
        return x_effective @ w_effective.t()""")

# ═══════════════════════════════════════════════════════
# CELL 5: Architecture Components (updated SubspaceRMSNorm for 4 subspaces)
# ═══════════════════════════════════════════════════════
md("## 4. Architecture Components")

code("""class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        x_fp32 = x.float()
        norm = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * norm).to(x.dtype) * self.weight


class SubspaceRMSNorm(nn.Module):
    \"\"\"
    RMSNorm that normalizes each of the four subspaces independently.

    Standard RMSNorm creates subtle non-linear coupling: a spike in one
    subspace suppresses magnitudes in all others via shared denominator.
    This variant eliminates that coupling.

    Subspaces: content | temporal-context | state-context | conjunctive
    Machens et al. (2010): what/when dynamics are maintained by separate
    mechanisms — shared normalization would couple them artificially.
    \"\"\"
    def __init__(self, d_model: int, d_content: int, d_temporal: int, d_state: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.d_content = d_content
        self.d_temporal = d_temporal
        self.d_state = d_state
        # d_conjunctive = d_model - d_content - d_temporal - d_state
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm_subspace(self, x):
        x_fp32 = x.float()
        return (x_fp32 * x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()).to(x.dtype)

    def forward(self, x):
        c = self.d_content
        t = c + self.d_temporal
        s = t + self.d_state
        xc = self._norm_subspace(x[:, :, :c])
        xt = self._norm_subspace(x[:, :, c:t])
        xs = self._norm_subspace(x[:, :, t:s])
        xb = self._norm_subspace(x[:, :, s:])
        return torch.cat([xc, xt, xs, xb], dim=-1) * self.weight

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = BitLinear(d_model, 3 * d_model, bias=False)
        self.out_proj = BitLinear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, L, D = x.size()
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None))
        return self.out_proj(out.transpose(1, 2).reshape(B, L, D))

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = BitLinear(d_model, d_ff, bias=False)
        self.w2 = BitLinear(d_ff, d_model, bias=False)
        self.w3 = BitLinear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, norm_cls=None):
        super().__init__()
        norm_cls = norm_cls or (lambda d: RMSNorm(d))
        self.norm1 = norm_cls(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm2 = norm_cls(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x""")

# ═══════════════════════════════════════════════════════
# CELL 6: Reasoning Cores — Baseline vs SPR (THE KEY DIFFERENCE)
# ═══════════════════════════════════════════════════════
md("""## 5. Reasoning Cores — Baseline vs SPR

**Baseline**: Iteration embedding added to ALL d_model dimensions.

**SPR (Subspace-Partitioned Reasoning)**: d_model split into four subspaces:
- **Content** (~85%): NEVER receives iteration signal — context-invariant
- **Temporal context** (~6%): Receives iteration embeddings — externally driven "when"
- **State context** (~6%): Purely recurrent — internally driven "what" of reasoning
- **Conjunctive** (~3%): Learned binding between all three streams

The temporal/state split follows Machens, Romo & Brody (J Neurosci 2010):
PFC working memory decomposes into "when" (time) components driven by
external input and "what" (stimulus) components driven by recurrent connectivity.
Both have identical total parameter counts for fair comparison.""")

code("""class EncoderStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.encoder_blocks)
        ])
    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x

class DecoderStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.decoder_blocks)
        ])
    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x""")

code("""# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BASELINE: Standard Reasoning Core (V2 original)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BaselineReasoningCore(nn.Module):
    \"\"\"Original V2: iteration embeddings added to ALL dimensions.\"\"\"
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.reasoning_blocks)
        ])
        self.iteration_embeddings = nn.Parameter(
            torch.randn(config.max_recurrence, 1, 1, config.d_model) * 0.02
        )
        self.halt_scorer = nn.Sequential(
            nn.Linear(config.d_model, 1), nn.Sigmoid()
        )
        # Store subspace dims for probing (even though baseline doesn't partition)
        self.d_content = int(config.d_model * config.spr_content_ratio)
        self.d_temporal = int(config.d_model * config.spr_temporal_ratio)
        self.d_state = int(config.d_model * config.spr_state_ratio)
        self.d_conjunctive = config.d_model - self.d_content - self.d_temporal - self.d_state

    def forward(self, x, mask=None, R=None, recurrence_dropout=0.0):
        if R is None:
            R = self.iteration_embeddings.size(0)
        iter_outputs = []
        for r in range(R):
            if self.training and recurrence_dropout > 0 and r > 0:
                if torch.rand(1).item() < recurrence_dropout:
                    continue
            if r < self.iteration_embeddings.size(0):
                x = x + self.iteration_embeddings[r]  # ALL dimensions
            for block in self.blocks:
                x = block(x, mask)
            iter_outputs.append(x)
        return x, iter_outputs, []  # No halt probs for baseline""")

code("""# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPR: Subspace-Partitioned Reasoning Core — Four Orthogonal Streams
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SPRReasoningCore(nn.Module):
    \"\"\"
    Subspace-Partitioned Reasoning with temporal/state context separation.

    Four subspaces:
      Content       (~85%) — token semantics, never receives iteration signal
      Temporal ctx  (~6%)  — "when" in reasoning chain, externally driven
      State ctx     (~6%)  — "what" reasoning state, purely recurrent
      Conjunctive   (~3%)  — learned binding of all three streams

    Biological basis:
      Bausch et al. Nature 2026 — content/context orthogonality in MTL
      Machens et al. J Neurosci 2010 — what/when separation in PFC:
        - Time components (~82% variance) driven by external input
        - Stimulus components (~18% variance) driven by recurrent connectivity
        - Functional separation on shared anatomical substrate
        - Time rescaling requires only input changes (fast);
          stimulus rescaling requires synaptic changes (slow)

    Engineering basis:
      Kerce & Fox arXiv:2603.07461 — dual-stream costs only 2.5%
    \"\"\"
    def __init__(self, config):
        super().__init__()
        # ── Four-subspace allocation ──
        self.d_content = int(config.d_model * config.spr_content_ratio)
        self.d_temporal = int(config.d_model * config.spr_temporal_ratio)
        self.d_state = int(config.d_model * config.spr_state_ratio)
        self.d_conjunctive = config.d_model - self.d_content - self.d_temporal - self.d_state

        # ── Norm: isolated subspace norms if configured ──
        if hasattr(config, 'spr_isolated_norm') and config.spr_isolated_norm:
            norm_cls = lambda d: SubspaceRMSNorm(d, self.d_content, self.d_temporal, self.d_state)
        else:
            norm_cls = None  # default RMSNorm

        # ── Transformer blocks (full d_model — no arch change) ──
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, norm_cls=norm_cls)
            for _ in range(config.reasoning_blocks)
        ])

        # ── Iteration embeddings: ONLY temporal-context dims ──
        # Machens: time components driven by external input.
        # This is the sole external injection point into the context stream.
        self.iteration_embeddings = nn.Parameter(
            torch.randn(config.max_recurrence, 1, 1, self.d_temporal) * 0.02
        )

        # ── Conjunctive binding network ──
        # Reads from content + temporal + state, writes to conjunctive
        binding_in_dim = self.d_content + self.d_temporal + self.d_state
        self.binding_net = nn.Sequential(
            nn.Linear(binding_in_dim, self.d_conjunctive * 4),
            nn.GELU(),
            nn.Linear(self.d_conjunctive * 4, self.d_conjunctive),
        )
        nn.init.zeros_(self.binding_net[-1].weight)
        nn.init.zeros_(self.binding_net[-1].bias)

        # ── MATURITY GATE (Silent Synapse Analog) ──
        # Controls conjunctive binding. sigmoid(-3) ≈ 0.047 at init.
        # Opens under sustained gradient pressure during R≥2 curriculum.
        self.maturity_gate = nn.Parameter(torch.tensor(-3.0))

        # ── STATE GATE (Recurrent State Accumulation Control) ──
        # Machens: "what" components are maintained by recurrent connectivity,
        # but this connectivity must LEARN what to maintain. During R=1
        # curriculum, there is no recurrence — state dims have nothing to
        # accumulate. This gate prevents spurious state-context patterns
        # from forming before recurrent processing begins.
        # sigmoid(-2.0) ≈ 0.12 at init — more open than maturity gate
        # because state dims receive gradient signal even from R=1
        # (via attention cross-talk), just less meaningful signal.
        self.state_gate = nn.Parameter(torch.tensor(-2.0))

        # ── CONJUNCTIVE HALT SCORER (AND Gate) ──
        # Reads ONLY from conjunctive subspace — where all three streams
        # (content quality + temporal position + state convergence) are bound.
        self.halt_scorer = nn.Sequential(
            nn.Linear(self.d_conjunctive, self.d_conjunctive),
            nn.GELU(),
            nn.Linear(self.d_conjunctive, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.halt_scorer[-2].weight)
        nn.init.zeros_(self.halt_scorer[-2].bias)

    def forward(self, x, mask=None, R=None, recurrence_dropout=0.0):
        if R is None:
            R = self.iteration_embeddings.size(0)
        iter_outputs = []
        halt_probs = []
        maturity = torch.sigmoid(self.maturity_gate)
        state_g = torch.sigmoid(self.state_gate)

        for r in range(R):
            if self.training and recurrence_dropout > 0 and r > 0:
                if torch.rand(1).item() < recurrence_dropout:
                    continue

            # ═══ FOUR-SUBSPACE PARTITIONING ═══
            c = self.d_content
            t = c + self.d_temporal
            s = t + self.d_state

            x_content  = x[:, :, :c]      # ~85% — token semantics
            x_temporal = x[:, :, c:t]      # ~6%  — "when" in reasoning
            x_state    = x[:, :, t:s]      # ~6%  — "what" of reasoning
            x_bind     = x[:, :, s:]       # ~3%  — conjunctive

            # 1. Inject iteration context ONLY into temporal subspace
            #    Machens: time components driven by external input
            if r < self.iteration_embeddings.size(0):
                x_temporal = x_temporal + self.iteration_embeddings[r]

            # 2. State subspace: NO external injection. Evolves purely through
            #    recurrent processing (attention + FFN in transformer blocks).
            #    Gated to suppress noise during R=1 curriculum.
            #    Machens: "what" components driven by recurrent connectivity.
            x_state = x_state * state_g

            # 3. Conjunctive binding: gated by maturity (silent synapse)
            binding_input = torch.cat([x_content, x_temporal, x_state], dim=-1)
            x_bind = x_bind + maturity * self.binding_net(binding_input)

            # 4. Reassemble — content dims NEVER saw iteration embedding,
            #    state dims received NO external injection
            x = torch.cat([x_content, x_temporal, x_state, x_bind], dim=-1)

            # 5. Standard transformer processing — attention sees ALL dims
            for block in self.blocks:
                x = block(x, mask)

            iter_outputs.append(x)

            # 6. Conjunctive halt scoring — AND gate on all three streams
            x_conj_post = x[:, :, s:]
            halt_prob = self.halt_scorer(x_conj_post).mean()
            halt_probs.append(halt_prob)

            if not self.training and halt_prob.item() > 0.8 and r > 0:
                break

        return x, iter_outputs, halt_probs

print("✅ Baseline + SPR reasoning cores defined (4-subspace variant)")""")

# ═══════════════════════════════════════════════════════
# CELL 7: Config + Model Assembly
# ═══════════════════════════════════════════════════════
md("## 6. Model Configuration & Assembly")

code("""@dataclass
class ModelConfig:
    # --- Architecture ---
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    vocab_size: int = 32000
    max_seq_len: int = 1024
    # --- Structure ---
    encoder_blocks: int = 3
    reasoning_blocks: int = 6
    max_recurrence: int = 4
    decoder_blocks: int = 3
    recurrence_dropout: float = 0.1
    # --- SPR ratios (Bausch 2026 + Machens 2010) ---
    spr_content_ratio: float = 0.85      # ~88% content neurons context-invariant
    spr_temporal_ratio: float = 0.06     # "when" — externally driven (iteration embeddings)
    spr_state_ratio: float = 0.06        # "what" — recurrently driven (no external injection)
    # conjunctive = 1 - content - temporal - state ≈ 0.03
    spr_isolated_norm: bool = False      # Ablation: normalize subspaces independently
    # --- Experiment mode ---
    use_spr: bool = True                 # Toggle between baseline and SPR""")

code("""class RecurrentBitNetV2SPR(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = EncoderStack(config)
        if config.use_spr:
            self.reasoning_core = SPRReasoningCore(config)
        else:
            self.reasoning_core = BaselineReasoningCore(config)
        self.decoder = DecoderStack(config)
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = BitLinear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

    def forward(self, idx, targets=None, R=None):
        B, L = idx.size()
        x = self.token_emb(idx)
        x = self.encoder(x)
        x, iter_outputs, halt_probs = self.reasoning_core(
            x, R=R,
            recurrence_dropout=self.config.recurrence_dropout if self.training else 0.0
        )
        x = self.decoder(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, iter_outputs, halt_probs""")

code("""# ━━━ Instantiate model ━━━
# ╔══════════════════════════════════════════════════════════════╗
# ║  TOGGLE THIS for Run 1 (SPR) vs Run 2 (Baseline):          ║
# ║    use_spr=True  → SPR (Subspace-Partitioned Reasoning)     ║
# ║    use_spr=False → Baseline (iteration embeds all dims)     ║
# ╚══════════════════════════════════════════════════════════════╝
config = ModelConfig(use_spr=True)

MODE_TAG = 'spr' if config.use_spr else 'baseline'
DRIVE_CKPT_DIR = os.path.join(DRIVE_BASE, f'recurrent_bitnet_v2_{MODE_TAG}')
LOCAL_CKPT_DIR = os.path.join(LOCAL_BASE, f'v2_{MODE_TAG}')
os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)
print(f"Mode: {MODE_TAG.upper()}")
print(f"Drive checkpoints → {DRIVE_CKPT_DIR}")
print(f"Local checkpoints → {LOCAL_CKPT_DIR}")

model = RecurrentBitNetV2SPR(config).to(DEVICE)

core = model.reasoning_core
d_content = core.d_content
d_temporal = core.d_temporal
d_state = core.d_state
d_conj = core.d_conjunctive

num_params = sum(p.numel() for p in model.parameters())
binding_params = sum(p.numel() for p in core.binding_net.parameters()) if hasattr(core, 'binding_net') else 0
eff_depth = config.encoder_blocks + config.reasoning_blocks * config.max_recurrence + config.decoder_blocks

print(f"✅ RecurrentBitNet V2-SPR (4-subspace)")
print(f"   Mode:            {'SPR' if config.use_spr else 'Baseline'}")
print(f"   Isolated norm:   {config.spr_isolated_norm}")
print(f"   Unique params:   {num_params:,}")
print(f"   Binding net:     {binding_params:,} params ({binding_params/num_params*100:.3f}%)")
print(f"   Effective depth: {eff_depth} layers (R={config.max_recurrence})")
print(f"   Subspace split:  content={d_content} | temporal={d_temporal} | state={d_state} | conj={d_conj}")
print(f"                    ({d_content/config.d_model*100:.0f}% / {d_temporal/config.d_model*100:.0f}% / "
      f"{d_state/config.d_model*100:.0f}% / {d_conj/config.d_model*100:.0f}%)")
if hasattr(core, 'maturity_gate'):
    mg = torch.sigmoid(core.maturity_gate).item()
    sg = torch.sigmoid(core.state_gate).item()
    print(f"   Maturity gate:   {mg:.4f} (conjunctive binding — silent synapse)")
    print(f"   State gate:      {sg:.4f} (recurrent state accumulation)")
if DEVICE == 'cuda':
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    estimated_gb = param_bytes * 4 / 1e9
    print(f"   Est. VRAM (R=1):  {estimated_gb:.1f} GB  |  R=4: ~{estimated_gb * 2.5:.1f} GB")
    print(f"   Available VRAM:  {VRAM_GB:.1f} GB")
    if estimated_gb * 2.5 > VRAM_GB * 0.9:
        print(f"   ⚠️  R=4 may OOM. Training loop will skip OOM steps automatically.")""")

# ═══════════════════════════════════════════════════════
# CELL 8-9: Training config + Data pipeline
# ═══════════════════════════════════════════════════════
md("## 7. Training Configuration")

code("""TOTAL_STEPS   = 500_000
BATCH_SIZE    = 8
SEQ_LEN       = config.max_seq_len
MAX_GRAD_NORM = 1.0
WARMUP_STEPS  = 2_000
PEAK_LR       = 2e-3
MIN_LR_RATIO  = 0.1
AUX_DECAY     = 0.3
LOG_EVERY     = 100
EVAL_EVERY    = 25_000
SAVE_LOCAL    = 5_000
SAVE_DRIVE    = 25_000
PROBE_EVERY   = 10_000

CURRICULUM = [
    (0,       1),
    (50_000,  2),
    (150_000, 3),
    (300_000, 4),
]

RESUME_FROM = None

total_tokens = TOTAL_STEPS * BATCH_SIZE * SEQ_LEN
print(f"Training plan: {TOTAL_STEPS:,} steps, {total_tokens/1e9:.1f}B tokens")
print(f"Probing every {PROBE_EVERY:,} steps")""")

md("## 8. Data Pipeline — FineWeb-Edu Streaming")

code("""from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id

fineweb = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
stream_iter = iter(fineweb)

def get_batch():
    global stream_iter
    inputs, targets = [], []
    while len(inputs) < BATCH_SIZE:
        try:
            text = next(stream_iter)['text']
        except StopIteration:
            stream_iter = iter(fineweb)
            text = next(stream_iter)['text']
        tokens = tokenizer(text, truncation=True, max_length=SEQ_LEN + 1, return_tensors='pt')['input_ids'][0]
        if len(tokens) < 64:
            continue
        if len(tokens) < SEQ_LEN + 1:
            pad = torch.full((SEQ_LEN + 1 - len(tokens),), PAD_ID, dtype=torch.long)
            tokens = torch.cat([tokens, pad])
        inputs.append(tokens[:SEQ_LEN])
        targets.append(tokens[1:SEQ_LEN + 1])
    return torch.stack(inputs).to(DEVICE), torch.stack(targets).to(DEVICE)

test_in, test_tgt = get_batch()
print(f"✅ Data pipeline ready — batch shape: {test_in.shape}")
del test_in, test_tgt""")

md("## 9. Optimizer & Scheduler")

code("""param_groups = [
    {'params': list(model.encoder.parameters()) + list(model.decoder.parameters()),
     'lr': PEAK_LR, 'name': 'encoder_decoder'},
    {'params': list(model.reasoning_core.parameters()),
     'lr': PEAK_LR * 2, 'name': 'reasoning_core'},
    {'params': [model.token_emb.weight], 'lr': PEAK_LR * 0.5, 'name': 'embeddings'},
    {'params': [model.final_norm.weight], 'lr': PEAK_LR, 'name': 'final_norm'},
]
optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.1)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))

start_step = 0
loss_history, recurrence_history, eval_log, probe_log = [], [], [], []
gate_log = []  # NEW: track gate trajectories
best_loss = float('inf')

if RESUME_FROM and os.path.exists(RESUME_FROM):
    ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'scaler_state_dict' in ckpt: scaler.load_state_dict(ckpt['scaler_state_dict'])
    start_step = ckpt['step']
    loss_history = ckpt.get('loss_history', [])
    probe_log = ckpt.get('probe_log', [])
    gate_log = ckpt.get('gate_log', [])
    best_loss = ckpt.get('best_loss', float('inf'))
    print(f"🔄 Resumed at step {start_step:,}")
else:
    print("🆕 Fresh training run")""")

# ═══════════════════════════════════════════════════════
# CELL: THE CRITICAL PROBING FUNCTION (4-subspace + DOC)
# ═══════════════════════════════════════════════════════
md("""## 10. Subspace Probing — The Key Measurement

Four-subspace probing battery:
1. **Iteration from content dims** → should be LOW (content is context-invariant)
2. **Iteration from temporal dims** → should be HIGH (temporal carries iteration)
3. **Iteration from state dims** → should be LOW (state has no external injection)
4. **Token from content dims** → should be HIGH (content carries tokens)
5. **Token from temporal/state dims** → should be LOW (no content leakage)

Plus: **DOC separation analysis** within the context subspace (Machens et al. 2010) —
computes the Difference of Covariances to verify that temporal and state context
carry orthogonal variance sources.""")

code("""@torch.no_grad()
def run_subspace_probes(model, num_batches=20, R=None):
    \"\"\"
    Measure content-context-temporal-state separation via linear probing + DOC.
    \"\"\"
    model.eval()
    core = model.reasoning_core
    dc = core.d_content
    dt = core.d_temporal
    ds = core.d_state

    if R is None:
        R = model.config.max_recurrence

    # Collect hidden states
    all_content_h, all_temporal_h, all_state_h = [], [], []
    all_token_ids, all_iter_labels = [], []

    for _ in range(num_batches):
        idx, _ = get_batch()
        x = model.token_emb(idx)
        x = model.encoder(x)

        for r in range(R):
            # Mimic the core's logic to get per-iteration hidden states
            if hasattr(core, 'binding_net'):
                # SPR path
                c = dc; t = c + dt; s = t + ds
                x_cont = x[:, :, :c]
                x_temp = x[:, :, c:t]
                x_st   = x[:, :, t:s]
                x_bind = x[:, :, s:]
                if r < core.iteration_embeddings.size(0):
                    x_temp = x_temp + core.iteration_embeddings[r]
                state_g = torch.sigmoid(core.state_gate)
                x_st = x_st * state_g
                binding_in = torch.cat([x_cont, x_temp, x_st], dim=-1)
                maturity = torch.sigmoid(core.maturity_gate)
                x_bind = x_bind + maturity * core.binding_net(binding_in)
                x = torch.cat([x_cont, x_temp, x_st, x_bind], dim=-1)
            else:
                # Baseline path
                if r < core.iteration_embeddings.size(0):
                    x = x + core.iteration_embeddings[r]

            for block in core.blocks:
                x = block(x)

            # Sample hidden states (every 64th token)
            h = x[:, ::64, :].reshape(-1, x.size(-1))
            tok = idx[:, ::64].reshape(-1)

            all_content_h.append(h[:, :dc].cpu())
            all_temporal_h.append(h[:, dc:dc+dt].cpu())
            all_state_h.append(h[:, dc+dt:dc+dt+ds].cpu())
            all_token_ids.append(tok.cpu())
            all_iter_labels.append(torch.full((h.size(0),), r, dtype=torch.long))

    content_h = torch.cat(all_content_h, dim=0)
    temporal_h = torch.cat(all_temporal_h, dim=0)
    state_h = torch.cat(all_state_h, dim=0)
    token_ids = torch.cat(all_token_ids, dim=0)
    iter_labels = torch.cat(all_iter_labels, dim=0)

    N = content_h.size(0)
    perm = torch.randperm(N)
    split = int(0.8 * N)
    train_idx, test_idx = perm[:split], perm[split:]
    lam = 1e-3
    results = {}

    # ── Helper: linear probe via ridge regression ──
    def probe_accuracy(X_train, y_train, X_test, y_test, num_classes):
        Y_oh = F.one_hot(y_train, num_classes).float()
        XtX = X_train.T @ X_train + lam * torch.eye(X_train.size(1))
        W = torch.linalg.solve(XtX, X_train.T @ Y_oh)
        preds = (X_test @ W).argmax(dim=-1)
        return (preds == y_test).float().mean().item()

    # ── Iteration probes (need R > 1) ──
    if R > 1 and iter_labels.max() > 0:
        num_cls = R
        chance = 1.0 / num_cls
        y_tr = iter_labels[train_idx]
        y_te = iter_labels[test_idx]

        # Iter from CONTENT (should be LOW for SPR)
        results['iter_from_content'] = probe_accuracy(
            content_h[train_idx].float(), y_tr, content_h[test_idx].float(), y_te, num_cls)
        results['iter_chance'] = chance

        # Iter from TEMPORAL (should be HIGH — externally driven)
        results['iter_from_temporal'] = probe_accuracy(
            temporal_h[train_idx].float(), y_tr, temporal_h[test_idx].float(), y_te, num_cls)

        # Iter from STATE (should be LOW — no external injection)
        results['iter_from_state'] = probe_accuracy(
            state_h[train_idx].float(), y_tr, state_h[test_idx].float(), y_te, num_cls)

    # ── Token probes (top-100 most frequent) ──
    token_counts = torch.bincount(token_ids, minlength=1)
    top_tokens = token_counts.argsort(descending=True)[:100]
    tok_mask_tr = torch.isin(token_ids[train_idx], top_tokens)
    tok_mask_te = torch.isin(token_ids[test_idx], top_tokens)

    if tok_mask_tr.sum() > 100 and tok_mask_te.sum() > 50:
        tok_map = {t.item(): i for i, t in enumerate(top_tokens)}
        def remap(ids, mask):
            return torch.tensor([tok_map[t.item()] for t in ids[mask]])

        y_tr_tok = remap(token_ids[train_idx], tok_mask_tr)
        y_te_tok = remap(token_ids[test_idx], tok_mask_te)

        # Token from CONTENT (should be HIGH)
        results['token_from_content'] = probe_accuracy(
            content_h[train_idx][tok_mask_tr].float(), y_tr_tok,
            content_h[test_idx][tok_mask_te].float(), y_te_tok, 100)
        results['token_chance'] = 0.01

        # Token from TEMPORAL (should be LOW for SPR)
        results['token_from_temporal'] = probe_accuracy(
            temporal_h[train_idx][tok_mask_tr].float(), y_tr_tok,
            temporal_h[test_idx][tok_mask_te].float(), y_te_tok, 100)

        # Token from STATE (should be LOW for SPR)
        results['token_from_state'] = probe_accuracy(
            state_h[train_idx][tok_mask_tr].float(), y_tr_tok,
            state_h[test_idx][tok_mask_te].float(), y_te_tok, 100)

    # ── DOC Separation Analysis (Machens et al. 2010) ──
    # Compute the Difference of Covariances between temporal and state
    # context to verify they carry orthogonal variance sources.
    # Positive eigenvalues = temporal-dominant variance,
    # Negative eigenvalues = state-dominant variance.
    if R > 1:
        ctx_all = torch.cat([temporal_h, state_h], dim=-1)  # (N, d_temporal + d_state)
        ctx_mean = ctx_all - ctx_all.mean(dim=0, keepdim=True)

        # Covariance from iteration-dependent variance (averaged over tokens)
        iter_cov = torch.zeros(ctx_all.size(1), ctx_all.size(1))
        for r_val in range(R):
            mask_r = (iter_labels == r_val)
            if mask_r.sum() > 10:
                mean_r = ctx_mean[mask_r].mean(dim=0)
                iter_cov += torch.outer(mean_r, mean_r)
        iter_cov /= R

        # Covariance from token-dependent variance (averaged over iterations)
        # Use top-50 tokens for tractability
        top50 = token_counts.argsort(descending=True)[:50]
        tok_cov = torch.zeros(ctx_all.size(1), ctx_all.size(1))
        n_tok_used = 0
        for t_val in top50:
            mask_t = (token_ids == t_val.item())
            if mask_t.sum() > 10:
                mean_t = ctx_mean[mask_t].mean(dim=0)
                tok_cov += torch.outer(mean_t, mean_t)
                n_tok_used += 1
        if n_tok_used > 0:
            tok_cov /= n_tok_used

        # DOC matrix: positive eigs = iteration-dominant, negative = token-dominant
        doc_matrix = iter_cov - tok_cov
        eigs = torch.linalg.eigvalsh(doc_matrix)
        results['doc_eigenvalues'] = eigs.tolist()
        # Separation score: fraction of total |eigenvalue| mass in positive eigs
        pos_mass = eigs[eigs > 0].sum().item()
        neg_mass = eigs[eigs < 0].abs().sum().item()
        total_mass = pos_mass + neg_mass + 1e-8
        results['doc_separation'] = pos_mass / total_mass  # >0.5 = iteration-dominant context

    model.train()
    return results

print("✅ Subspace probing function ready (4-subspace + DOC)")""")

# ═══════════════════════════════════════════════════════
# Eval + Checkpoint
# ═══════════════════════════════════════════════════════
md("## 11. Evaluation & Checkpointing")

code("""@torch.no_grad()
def evaluate(model, num_batches=50, R=None):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    if R is None: R = config.max_recurrence
    for _ in tqdm(range(num_batches), desc=f"Eval (R={R})", leave=False):
        idx, targets = get_batch()
        with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda'), dtype=torch.bfloat16):
            logits, _, _, _ = model(idx, targets, R=R)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        mask = targets_flat != PAD_ID
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=PAD_ID, reduction='sum')
        total_loss += loss.item()
        total_tokens += mask.sum().item()
    model.train()
    avg_loss = total_loss / max(1, total_tokens)
    return {'loss': avg_loss, 'perplexity': math.exp(min(avg_loss, 100)), 'R': R}

def save_checkpoint(step, to_drive=False):
    ckpt = {
        'step': step, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': asdict(config), 'loss_history': loss_history,
        'recurrence_history': recurrence_history, 'eval_log': eval_log,
        'probe_log': probe_log, 'gate_log': gate_log, 'best_loss': best_loss,
    }
    local_path = os.path.join(LOCAL_CKPT_DIR, f'checkpoint_step{step}.pt')
    torch.save(ckpt, local_path)
    print(f"  💾 Local: {local_path}")
    if to_drive:
        drive_path = os.path.join(DRIVE_CKPT_DIR, f'checkpoint_step{step}.pt')
        shutil.copy2(local_path, drive_path)
        print(f"  ☁️  Drive: {drive_path}")
    local_ckpts = sorted([f for f in os.listdir(LOCAL_CKPT_DIR) if f.startswith('checkpoint_step')])
    while len(local_ckpts) > 3:
        os.remove(os.path.join(LOCAL_CKPT_DIR, local_ckpts.pop(0)))

print("✅ Evaluation & checkpoint functions ready")""")

# ═══════════════════════════════════════════════════════
# Training loop with integrated probing
# ═══════════════════════════════════════════════════════
md("""## 12. Training Loop with Subspace Probing

Identical to V2 training loop, plus:
- **4-subspace probes** every 10K steps
- **DOC separation analysis** (Machens 2010)
- **Gate trajectory tracking** (maturity + state gate values over training)""")

code(
    """model.train()
print(f"🚀 Starting V2 training — mode: {MODE_TAG.upper()}")
print(f"   Output: {DRIVE_CKPT_DIR}")
print(f"   Steps {start_step+1:,} → {TOTAL_STEPS:,}")
print("=" * 70)

run_start = time.time()
window_start = time.time()
window_loss = 0.0

for step in range(start_step + 1, TOTAL_STEPS + 1):
    # 1. Curriculum
    R = 1
    for threshold, depth in reversed(CURRICULUM):
        if step >= threshold:
            R = depth
            break

    # 2. Get batch
    idx, targets = get_batch()

    # 3. Forward + Backward (OOM-safe)
    try:
        with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda'), dtype=torch.bfloat16):
            logits, base_loss, iter_outputs, halt_probs = model(idx, targets, R=R)

            # 4. Auxiliary loss
            aux_loss = torch.tensor(0.0, device=DEVICE)
            for r, hidden in enumerate(iter_outputs):
                step_normed = model.final_norm(hidden)
                step_logits = model.lm_head(step_normed)
                step_loss = F.cross_entropy(
                    step_logits.view(-1, step_logits.size(-1)), targets.view(-1)
                )
                aux_loss = aux_loss + (AUX_DECAY ** (R - (r + 1))) * step_loss

            # 5. Halting regularization (PonderNet-style geometric prior)
            halt_loss = torch.tensor(0.0, device=DEVICE)
            if halt_probs and len(halt_probs) > 1:
                HALT_LAMBDA = 0.3
                for r, hp in enumerate(halt_probs):
                    prior_halt = HALT_LAMBDA * ((1 - HALT_LAMBDA) ** r)
                    halt_loss = halt_loss + (
                        prior_halt * torch.log((prior_halt + 1e-8) / (hp + 1e-8)) +
                        (1 - prior_halt) * torch.log((1 - prior_halt + 1e-8) / (1 - hp + 1e-8))
                    )
                halt_loss = halt_loss * 0.01

            total_loss = base_loss + aux_loss + halt_loss

        # Backward
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            print(f"  ⚠️ OOM at step {step} (R={R}). Skipping.")
            optimizer.zero_grad(set_to_none=True)
            continue
        raise

    # 6. Track
    loss_val = total_loss.item()
    loss_history.append(loss_val)
    recurrence_history.append(R)
    window_loss += loss_val
    if loss_val < best_loss:
        best_loss = loss_val

    # 7. Log (with gate tracking)
    if step % LOG_EVERY == 0:
        elapsed = time.time() - window_start
        avg_loss = window_loss / LOG_EVERY
        ms_per_step = elapsed / LOG_EVERY * 1000
        lr = scheduler.get_last_lr()[0]
        remaining = (TOTAL_STEPS - step) * (elapsed / LOG_EVERY)
        eta_h = remaining / 3600

        gate_str = ""
        if hasattr(model.reasoning_core, 'maturity_gate'):
            mg = torch.sigmoid(model.reasoning_core.maturity_gate).item()
            sg = torch.sigmoid(model.reasoning_core.state_gate).item()
            gate_str = f" | maturity={mg:.3f} | state_g={sg:.3f}"
            gate_log.append({'step': step, 'maturity': mg, 'state_gate': sg, 'R': R})

        print(f"Step {step:>7,}/{TOTAL_STEPS:,} | Loss {avg_loss:.4f} | R={R} | "
              f"LR {lr:.2e} | {ms_per_step:.0f} ms/step | ETA {eta_h:.1f}h"
              + (f" | halt={halt_probs[-1].item():.2f}" if halt_probs else "")
              + gate_str)
        window_start = time.time()
        window_loss = 0.0

    # 8. Save (local)
    if step % SAVE_LOCAL == 0:
        save_checkpoint(step, to_drive=False)

    # 9. Save (Drive) + Evaluate
    if step % SAVE_DRIVE == 0:
        save_checkpoint(step, to_drive=True)

    if step % EVAL_EVERY == 0:
        result = evaluate(model, num_batches=100, R=R)
        eval_log.append({'step': step, **result})
        print(f"  📊 Eval @ step {step:,}: Loss={result['loss']:.4f}, "
              f"PPL={result['perplexity']:.2f} (R={R})")
        model.train()

    # 10. SUBSPACE PROBING — the key experiment
    if step % PROBE_EVERY == 0 and R > 1:
        print(f"  🔬 Running 4-subspace probes (R={R})...")
        probe_results = run_subspace_probes(model, num_batches=15, R=R)
        probe_results['step'] = step
        probe_results['R'] = R
        probe_log.append(probe_results)

        # Print iteration probes
        if 'iter_from_content' in probe_results:
            ch = probe_results['iter_chance']
            ic = probe_results['iter_from_content']
            it = probe_results['iter_from_temporal']
            ist = probe_results['iter_from_state']
            print(f"     Iter from CONTENT:  {ic:.3f} (chance={ch:.3f}) "
                  f"{'✅ LOW' if ic < ch * 3 else '⚠️ LEAKING'}")
            print(f"     Iter from TEMPORAL: {it:.3f} "
                  f"{'✅ HIGH' if it > ch * 2 else '⚠️ WEAK'}")
            print(f"     Iter from STATE:    {ist:.3f} "
                  f"{'✅ LOW' if ist < ch * 3 else '⚠️ LEAKING'}")
            sep_temporal = it - ic
            sep_state = it - ist
            print(f"     Temporal-content gap: {sep_temporal:.3f} "
                  f"{'✅ STRONG' if sep_temporal > 0.3 else '⚠️ NARROW'}")
            print(f"     Temporal-state gap:   {sep_state:.3f} "
                  f"{'✅ STRONG' if sep_state > 0.2 else '⚠️ NARROW'}")

        if 'token_from_content' in probe_results:
            tfc = probe_results['token_from_content']
            tft = probe_results['token_from_temporal']
            tfs = probe_results['token_from_state']
            print(f"     Token from CONTENT:  {tfc:.3f} (chance=0.01)")
            print(f"     Token from TEMPORAL: {tft:.3f} "
                  f"{'✅ LOW' if tft < tfc * 0.5 else '⚠️ LEAKING'}")
            print(f"     Token from STATE:    {tfs:.3f} "
                  f"{'✅ LOW' if tfs < tfc * 0.5 else '⚠️ LEAKING'}")

        if 'doc_separation' in probe_results:
            doc_s = probe_results['doc_separation']
            print(f"     DOC separation:      {doc_s:.3f} "
                  f"({'iteration-dominant' if doc_s > 0.5 else 'token-dominant'} context)")

        model.train()

total_time = time.time() - run_start
print("=" * 70)
print(f"✅ Training complete ({MODE_TAG.upper()})! {total_time/3600:.1f} hours, best loss: {best_loss:.4f}")"""
)

# ═══════════════════════════════════════════════════════
# Save, Final Eval, Visualization
# ═══════════════════════════════════════════════════════
md("## 13. Save Final Model")

code("""save_checkpoint(TOTAL_STEPS, to_drive=True)

print("\\n📦 Exporting ternary weights...")
ternary_weights = {}
with torch.no_grad():
    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and module.weight is not model.token_emb.weight:
            w_ternary, w_scale = quantize_weights_ternary(module.weight)
            ternary_weights[name] = {
                'weight_ternary': w_ternary.to(torch.int8).cpu(),
                'weight_scale': w_scale.float().cpu(),
            }
export_path = os.path.join(DRIVE_CKPT_DIR, f'{MODE_TAG}_ternary_export.pt')
torch.save(ternary_weights, export_path)
ternary_count = sum(v['weight_ternary'].numel() for v in ternary_weights.values())
print(f"📦 Ternary export → {export_path} ({ternary_count:,} params)")

config_path = os.path.join(DRIVE_CKPT_DIR, f'{MODE_TAG}_config.json')
with open(config_path, 'w', encoding="utf-8") as f:
    json.dump(asdict(config), f, indent=2)
print(f"📋 Config → {config_path}")""")

md("## 14. Final Evaluation — Recurrence Depth Comparison")

code("""print("📊 Final Evaluation — Recurrence Depth Comparison")
print("=" * 50)
for test_R in range(1, config.max_recurrence + 1):
    result = evaluate(model, num_batches=100, R=test_R)
    print(f"  R={test_R}: Loss={result['loss']:.4f}, Perplexity={result['perplexity']:.2f}")
print("\\n(Lower R = faster inference, higher R = better quality)")""")

md("""## 15. Visualization — Training + Subspace Probes + Gate Trajectories

Six panels:
1. Training loss
2. Curriculum schedule
3. Iteration probes (content vs temporal vs state)
4. Token probes (content vs temporal vs state)
5. DOC separation score over training
6. Gate trajectories (maturity + state gate)""")

code("""import matplotlib.pyplot as plt

fig, axes = plt.subplots(6, 1, figsize=(14, 30))

# Panel 1: Loss curve
axes[0].plot(loss_history, alpha=0.1, color='steelblue')
if len(loss_history) > 100:
    window = min(500, len(loss_history) // 10)
    smoothed = []
    running = sum(loss_history[:window])
    for i in range(window, len(loss_history)):
        smoothed.append(running / window)
        running += loss_history[i] - loss_history[i - window]
    smoothed.append(running / window)
    axes[0].plot(range(window, len(loss_history) + 1), smoothed, color='steelblue', linewidth=2)
axes[0].set_ylabel("Loss")
axes[0].set_title(f"RecurrentBitNet V2-{'SPR' if config.use_spr else 'Baseline'} — Training Loss")
axes[0].grid(alpha=0.3)

# Panel 2: Curriculum
axes[1].step(range(len(recurrence_history)), recurrence_history, where='post', color='coral', linewidth=2)
axes[1].set_ylabel("R"); axes[1].set_title("Progressive Recurrence Curriculum")
axes[1].set_yticks([1, 2, 3, 4]); axes[1].grid(alpha=0.3)""")

code("""# Panel 3: Iteration probes (3 lines: content, temporal, state)
if probe_log and any('iter_from_content' in p for p in probe_log):
    steps_p = [p['step'] for p in probe_log if 'iter_from_content' in p]
    ifc = [p['iter_from_content'] for p in probe_log if 'iter_from_content' in p]
    ift = [p['iter_from_temporal'] for p in probe_log if 'iter_from_temporal' in p]
    ifs = [p['iter_from_state'] for p in probe_log if 'iter_from_state' in p]
    ch = [p['iter_chance'] for p in probe_log if 'iter_chance' in p]

    axes[2].plot(steps_p, ifc, 'o-', color='crimson', linewidth=2, markersize=4, label='Iter from CONTENT (should be LOW)')
    axes[2].plot(steps_p, ift, 's-', color='forestgreen', linewidth=2, markersize=4, label='Iter from TEMPORAL (should be HIGH)')
    axes[2].plot(steps_p, ifs, 'D-', color='darkorange', linewidth=2, markersize=4, label='Iter from STATE (should be LOW)')
    if ch:
        axes[2].axhline(y=ch[-1], color='gray', linestyle='--', alpha=0.7, label=f'Chance ({ch[-1]:.2f})')
    axes[2].set_ylabel("Probe Accuracy")
    axes[2].set_title("🔬 Iteration Decoding — Content vs Temporal vs State")
    axes[2].legend(loc='center right', fontsize=8); axes[2].grid(alpha=0.3)
else:
    axes[2].text(0.5, 0.5, 'No iter probe data yet (R must be > 1)', ha='center', va='center', transform=axes[2].transAxes)

# Panel 4: Token probes
if probe_log and any('token_from_content' in p for p in probe_log):
    steps_t = [p['step'] for p in probe_log if 'token_from_content' in p]
    tfc = [p['token_from_content'] for p in probe_log if 'token_from_content' in p]
    tft = [p['token_from_temporal'] for p in probe_log if 'token_from_temporal' in p]
    tfs = [p['token_from_state'] for p in probe_log if 'token_from_state' in p]

    axes[3].plot(steps_t, tfc, 'o-', color='forestgreen', linewidth=2, markersize=4, label='Token from CONTENT (should be HIGH)')
    axes[3].plot(steps_t, tft, 's-', color='crimson', linewidth=2, markersize=4, label='Token from TEMPORAL (should be LOW)')
    axes[3].plot(steps_t, tfs, 'D-', color='darkorange', linewidth=2, markersize=4, label='Token from STATE (should be LOW)')
    axes[3].axhline(y=0.01, color='gray', linestyle='--', alpha=0.7, label='Chance (0.01)')
    axes[3].set_ylabel("Probe Accuracy")
    axes[3].set_title("🔬 Token Decoding — Content vs Temporal vs State")
    axes[3].legend(loc='center right', fontsize=8); axes[3].grid(alpha=0.3)
else:
    axes[3].text(0.5, 0.5, 'No token probe data yet', ha='center', va='center', transform=axes[3].transAxes)

# Panel 5: DOC separation score
if probe_log and any('doc_separation' in p for p in probe_log):
    steps_d = [p['step'] for p in probe_log if 'doc_separation' in p]
    doc_vals = [p['doc_separation'] for p in probe_log if 'doc_separation' in p]
    axes[4].plot(steps_d, doc_vals, 'o-', color='purple', linewidth=2, markersize=5)
    axes[4].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Equal mass (0.5)')
    axes[4].set_ylabel("DOC Separation")
    axes[4].set_title("🔬 DOC Analysis — Iteration vs Token Variance in Context Subspace (Machens)")
    axes[4].set_ylim(0, 1); axes[4].legend(); axes[4].grid(alpha=0.3)
else:
    axes[4].text(0.5, 0.5, 'No DOC data yet', ha='center', va='center', transform=axes[4].transAxes)

# Panel 6: Gate trajectories
if gate_log:
    g_steps = [g['step'] for g in gate_log]
    g_mat = [g['maturity'] for g in gate_log]
    g_state = [g['state_gate'] for g in gate_log]
    axes[5].plot(g_steps, g_mat, '-', color='teal', linewidth=2, label='Maturity gate (conjunctive)')
    axes[5].plot(g_steps, g_state, '-', color='coral', linewidth=2, label='State gate (recurrent)')
    # Mark curriculum transitions
    for thresh, R_val in CURRICULUM:
        if thresh > 0:
            axes[5].axvline(x=thresh, color='gray', linestyle=':', alpha=0.5)
            axes[5].text(thresh, 0.95, f'R={R_val}', fontsize=8, ha='left', va='top')
    axes[5].set_ylabel("Gate Value (sigmoid)")
    axes[5].set_title("Gate Trajectories — Silent Synapse Opening")
    axes[5].set_ylim(0, 1); axes[5].legend(); axes[5].grid(alpha=0.3)
else:
    axes[5].text(0.5, 0.5, 'No gate data (baseline mode)', ha='center', va='center', transform=axes[5].transAxes)

axes[-1].set_xlabel("Training Step")
plt.tight_layout()
plot_path = os.path.join(DRIVE_CKPT_DIR, f'{MODE_TAG}_training_curves.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"📈 Saved → {plot_path}")""")

md("""## 16. Probe Results Summary — Four-Subspace Separation

The definitive table. Compare SPR vs Baseline runs.""")

code("""if probe_log:
    print("=" * 90)
    print("SUBSPACE PROBE RESULTS — Four-Subspace Separation")
    print(f"Model: {MODE_TAG.upper()} ({'Subspace-Partitioned' if config.use_spr else 'Standard Baseline'})")
    print(f"Subspaces: content={d_content} | temporal={d_temporal} | state={d_state} | conj={d_conj}")
    print("=" * 90)
    print(f"{'Step':>8} {'R':>3} {'I→Cont':>8} {'I→Temp':>8} {'I→State':>8} "
          f"{'T→Cont':>8} {'T→Temp':>8} {'T→State':>8} {'DOC':>6}")
    print("-" * 90)
    for p in probe_log:
        ic = p.get('iter_from_content', float('nan'))
        it = p.get('iter_from_temporal', float('nan'))
        ist = p.get('iter_from_state', float('nan'))
        tfc = p.get('token_from_content', float('nan'))
        tft = p.get('token_from_temporal', float('nan'))
        tfs = p.get('token_from_state', float('nan'))
        doc = p.get('doc_separation', float('nan'))
        print(f"{p['step']:>8,} {p['R']:>3} {ic:>8.4f} {it:>8.4f} {ist:>8.4f} "
              f"{tfc:>8.4f} {tft:>8.4f} {tfs:>8.4f} {doc:>6.3f}")
    print("-" * 90)

    # Summary interpretation
    last = probe_log[-1]
    if 'iter_from_content' in last and 'iter_from_temporal' in last:
        ic = last['iter_from_content']
        it = last['iter_from_temporal']
        ist = last['iter_from_state']
        ch = last['iter_chance']
        content_gap = it - ic
        state_gap = it - ist

        if content_gap > 0.3 and state_gap > 0.2 and ic / ch < 3.0:
            print(f"\\n✅ FOUR-SUBSPACE SEPARATION CONFIRMED")
            print(f"   Content protected from iteration:  {ic:.4f} ({ic/ch:.1f}x chance)")
            print(f"   Temporal carries iteration:        {it:.4f}")
            print(f"   State protected from iteration:    {ist:.4f} ({ist/ch:.1f}x chance)")
            print(f"   Machens prediction validated: temporal (external drive) ≠ state (recurrent)")
        elif content_gap > 0.1:
            print(f"\\n⚠️ PARTIAL SEPARATION")
            print(f"   Content-temporal gap: {content_gap:.4f}")
            print(f"   State-temporal gap:   {state_gap:.4f}")
            print(f"   Consider spr_isolated_norm=True")
        else:
            print(f"\\n❌ WEAK SEPARATION — content_gap={content_gap:.4f}")

    probe_path = os.path.join(DRIVE_CKPT_DIR, f'{MODE_TAG}_probe_results.json')
    with open(probe_path, 'w', encoding="utf-8") as f:
        json.dump(probe_log, f, indent=2)
    print(f"\\n📋 Probe results → {probe_path}")
else:
    print("No probe data collected yet. Train with R > 1 to generate probe data.")""")

# ═══════════════════════════════════════════════════════
# GENERATE THE NOTEBOOK FILE
# ═══════════════════════════════════════════════════════
notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "gpuClass": "premium",
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = "/home/ty/Repositories/ai_workspace/recurrent_bitnet/notebooks/RecurrentBitNet_V2_SPR.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"✅ Notebook generated: {output_path}")
print(
    f"   {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code, "
    f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)"
)
