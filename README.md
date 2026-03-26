# RecurrentBitNet

**Selective ternary quantization of hybrid linear-attention transformers, informed by LLM neuroanatomy and information-theoretic reasoning compression.**

> A research project exploring the intersection of BitNet-style 1.58-bit quantization, Gated DeltaNet hybrid architectures (Qwen3.5), RYS-style layer duplication neuroanatomy, and Conditional Information Bottleneck reasoning compression.

## Overview

This project combines insights from six independent research threads into a single architecture:

1. **BitNet b1.58** — Ternary weight quantization ({-1, 0, +1}) that enables inference with only additions/subtractions, no multiplications
2. **Gated DeltaNet** — Linear attention with recurrent state that provides O(1) per-token inference while maintaining long-range memory
3. **RYS (Repeat Your Self)** — The discovery that duplicating specific middle-layer "reasoning circuits" improves model quality without changing weights
4. **Conditional Information Bottleneck** — Framing efficient reasoning as lossy compression where tokens are penalized by semantic information content
5. **Seven-Paper Convergence** — Independent research from biology, physics, neuroscience, math, and dynamical systems all describing the same transformer computation structure
6. **CALM** — Continuous autoregressive language models that predict vectors instead of tokens for efficiency

## Core Hypothesis

**Gated DeltaNet layers are more robust to ternary quantization than standard attention layers**, because:

1. DeltaNet maintains a recurrent state matrix `S_t = α·S_{t-1} + β·v·kᵀ` that accumulates in full precision
2. The delta rule provides error-correcting memory updates — quantization noise in projections gets corrected over time
3. 75% of Qwen3.5's layers are DeltaNet — converting only these to BitLinear while keeping the 25% full-attention layers at FP16 creates a hybrid-precision architecture

**Expected result**: A 2B-effective model that runs inference in ~1.8 GB with minimal quality loss, distilled from a larger teacher in the same architecture family.

---

## Architecture: Selective BitLinear on Qwen3.5

### Base Model: Qwen3.5-2B-Base
- **2B parameters**, 24 layers, d_model=2048
- **Hybrid attention**: 18 Gated DeltaNet layers + 6 Full Attention layers
- **Layout**: `6 × (3 × [DeltaNet → FFN] → 1 × [Full Attention → FFN])`
- **Context**: 262K tokens native, extensible to 1M
- **Vocab**: 248,320 tokens (201 languages)
- **License**: Apache 2.0

### Quantization Strategy
```
Layer Type          | Count | Quantization | Rationale
--------------------|-------|--------------|------------------------------------------
Gated DeltaNet      | 18    | BitLinear    | Recurrent state compensates for noise
Full Attention      | 6     | FP16         | Global retrieval needs precision
Embeddings          | 1     | Q8           | Large (248K×2048), needs compression
Norms/misc          | -     | FP16         | Small, precision-sensitive
```

### Inference Size
```
Component               FP16        Selective BitLinear
Embeddings (248K×2048): 1.02 GB     507 MB (Q8)
DeltaNet layers (18):   2.17 GB     543 MB (TQ2_0)
Full Attn layers (6):   0.72 GB     0.72 GB (FP16)
Other:                  ~50 MB      ~50 MB
TOTAL:                  3.96 GB     ~1.84 GB
```

---

## Background: RYS — LLM Neuroanatomy

### The Discovery (David Noel Ng, 2024)

RYS (**R**epeat **Y**our **S**elf) was a technique that reached #1 on the HuggingFace Open LLM Leaderboard by duplicating 7 middle layers of Qwen2-72B — with zero weight changes, zero fine-tuning, and zero gradient descent.

**Key findings:**

1. **Transformers have functional anatomy**: Early layers encode input into abstract representations. Late layers decode back to tokens. Middle layers are a "reasoning cortex."

2. **Reasoning circuits are indivisible**: Duplicating a single middle layer does nothing (or hurts). Duplicating a complete block of 5-7 layers (a "circuit") improves performance across ALL benchmarks. The circuit is an indivisible processing unit.

3. **RYS brain scanning**: For an N-layer model, test all (i, j) pairs where layers i..j are duplicated. This creates a heatmap showing which layer regions are encoding, reasoning, or decoding. The pattern is consistent across model families but each architecture has unique boundaries.

4. **Implications for quantization**: If encoding/decoding layers need precision (they handle format-specific I/O) but reasoning circuits are robust to rearrangement, then reasoning circuits may also be robust to quantization. This directly motivates our selective BitLinear strategy.

**The RYS brain scan protocol:**
```python
# For a model with N layers, test configuration (i, j):
# Run layers 0..j normally, then re-run layers i..j again, then finish j..N
# This duplicates layers [i, j] in the execution path
#
# Example: (i=45, j=51) for 80-layer model:
#   0→1→...→44→45→46→47→48→49→50→51→45→46→47→48→49→50→51→52→...→79
#   (layers 45-51 execute twice)
#
# Score each (i,j) on proxy tasks, build heatmap
for i in range(N):
    for j in range(i+1, N):
        model_ij = duplicate_layers(model, start=i, end=j)
        math_score = evaluate_math(model_ij)
        eq_score = evaluate_eq(model_ij)
        heatmap[i][j] = math_score + eq_score
```

**RYS applied to Qwen3.5**: The DeltaNet/Attention hybrid architecture creates natural circuit boundaries. Every 4th layer (full attention) acts as a "checkpoint" — the DeltaNet triplets between them are candidate reasoning circuits. RYS-style scanning can identify which triplets tolerate quantization best.

> Source: [LLM Neuroanatomy: How I Topped the LLM Leaderboard Without Changing a Single Weight](https://dnhkng.github.io/posts/rys/) — David Noel Ng, 2026  
> Models: [dnhkng/RYS-XLarge](https://huggingface.co/dnhkng/RYS-XLarge) on HuggingFace

---

## Background: BitLinear (1.58-bit Quantization)

BitLinear replaces `nn.Linear` with ternary weights {-1, 0, +1} during forward pass while maintaining full-precision latent weights for training.

**The forward pass:**
```
Input → RMSNorm → Activation Quant (8-bit absmax) → Weight Quant (ternary) → MatMul → Rescale
```

**Key components:**
- **Weight quantization**: `w_quant = round(w / mean(|w|)).clamp(-1, 1)`
- **Activation quantization**: 8-bit absmax per-token
- **STE (Straight-Through Estimator)**: Gradient flows through quantization as if it didn't happen
- **No bias**: Removed per BitNet spec
- **RMSNorm per layer**: Stabilizes training with ternary weights

**Why BitLinear on DeltaNet specifically:**
- DeltaNet projections (Q, K, V, output gate) are all `nn.Linear` — direct swap
- The state matrix `S` accumulates in full precision regardless of projection quantization
- The delta rule (`S += β·v·kᵀ - β·(kᵀ·S)·kᵀ`) error-corrects on every token, potentially compensating for quantization noise in v and k
- 18/24 layers converted = 75% of parameters compressed to 2 bits

> Source: [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764) — Ma et al., 2024  
> Source: [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) — Yang et al., 2024

---

## Background: Conditional Information Bottleneck (CIB)

The CIB framework (Massoli et al., 2026) recasts efficient reasoning as lossy compression:

**The Attention Paradox**: Standard Information Bottleneck assumes reasoning trace Z is the sole channel from prompt X to answer Y. But transformer attention lets Y see X directly, violating the Markov chain Y↔X↔Z.

**CIB Resolution**: Model CoT as source coding with side information:
```
max_θ  I(Z; Y | X)  -  β · I(X; Z)
       └─sufficiency─┘    └─minimality─┘
```

**Practical reward**: `R = accuracy_reward + β · Σ log Q_φ(z_t | z_<t)`  
where Q_φ is a frozen prior model. Tokens with high surprisal (novel/informative) are cheap; predictable filler tokens are expensive.

**Results**: 41% compression of reasoning traces with ≤1.5% accuracy loss on math benchmarks, Pareto-dominating naive length penalties.

**Applied to DeltaNet**: The CIB framework naturally maps onto hybrid architectures — DeltaNet layers with compressed state ARE an information bottleneck. Full attention layers that see the raw prompt are where the "Attention Paradox" applies. CIB regularization should target full attention layers specifically.

> Source: [Reasoning as Compression: Unifying Budget Forcing via the Conditional Information Bottleneck](https://arxiv.org/abs/2603.08462) — Massoli, Kuzmin & Behboodi (Qualcomm AI Research), 2026

---

## Training Strategy

### Phase 0: Architecture Surgery
Load Qwen3.5-2B-Base (or 0.8B for local development), convert DeltaNet layers to BitLinear.

### Phase 1: Distillation Recovery (Critical)
- **Teacher**: Qwen3.5-4B or 9B (same tokenizer, same architecture family)
- **Student**: Qwen3.5-2B with BitLinear DeltaNet layers
- **Loss**: `L = α·CE(student, labels) + β·KL(student_logits, teacher_logits)`
- **Data**: FineWeb-Edu + OpenWebMath + The Stack v2 (5-10B tokens)

### Phase 2: Reasoning Fine-tune
- CIB-style objective: penalize uninformative reasoning tokens
- Auxiliary per-recurrence-step prediction loss
- Causal halting: stop DeltaNet state updates when representation stabilizes

### Phase 3: RYS Analysis
- Brain-scan the original and BitLinear models
- Identify which DeltaNet triplets are most/least affected by quantization
- Adaptive precision: quantize robust circuits to TQ2_0, keep sensitive ones at higher precision

---

## Quickstart

### Option A: Google Colab (Recommended)

No setup required — Colab provides the GPU and Python environment.

1. Upload `notebooks/RecurrentBitNet.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Select a GPU runtime: **Runtime → Change runtime type → T4 GPU** (free) or A100 (Pro)
3. Run cells top-to-bottom — the first cell installs dependencies automatically

The notebook auto-detects your GPU and configures batch sizes accordingly:

| Colab Tier | GPU | VRAM | Student | Teacher | ~Training Time (80M tokens) |
|-----------|-----|------|---------|---------|---------------------------|
| Free | T4 | 16 GB | 0.8B | 0.8B | ~15 hours |
| Pro | L4 | 24 GB | 0.8B | 0.8B | ~7 hours |
| Pro+ | A100 | 40 GB | 0.8B | 2B | ~3 hours |

### Option B: Local Development

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/recurrent_bitnet.git
cd recurrent_bitnet

# Create environment (uv)
uv venv
source .venv/bin/activate
uv pip install -e .

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then run the experiment via Python:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.surgery import convert_model, SurgeryConfig, surgical_report
from src.distill import DistillationTrainer, DistillationConfig, create_dataloader

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B-Base", torch_dtype=torch.float32, trust_remote_code=True
).cuda()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base", trust_remote_code=True)

# Surgery: convert DeltaNet layers to BitLinear
report = convert_model(model, SurgeryConfig(aggression="standard"))
print(surgical_report(model, report))

# Distillation: recover quality from original teacher
teacher = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B-Base", torch_dtype=torch.bfloat16, trust_remote_code=True
).cuda()
config = DistillationConfig(num_steps=5000)
config.auto_configure()  # auto-detect GPU
trainer = DistillationTrainer(model, teacher, config)
trainer.train(create_dataloader(config, tokenizer))
```

---

## Project Structure

```
recurrent_bitnet/
├── README.md                 # This file
├── pyproject.toml            # Python project configuration
├── .gitignore
├── docs/
│   ├── DESIGN.md             # Architecture design rationale (v2 backlog)
│   ├── TRAINING_STRATEGY.md  # Detailed training plan
│   ├── V3_QWEN35.md          # Qwen3.5-specific analysis
│   └── rys_analysis.md       # RYS neuroanatomy deep-dive
├── src/
│   ├── __init__.py            # Package exports
│   ├── bitlinear.py           # BitLinear layer (ternary weights, STE)
│   ├── surgery.py             # Qwen3.5 → BitLinear conversion
│   └── distill.py             # Distillation training loop
└── notebooks/
    └── RecurrentBitNet.ipynb   # Complete Colab experiment notebook
```

## Hardware Requirements

| Task | GPU | VRAM | Notes |
|------|-----|------|-------|
| Inference (BitLinear) | Any | ~2 GB | RTX 3060, phones, RPi |
| Colab training (0.8B) | T4 | 16 GB | Student 0.8B + teacher 0.8B |
| Cloud training (0.8B) | A100 | 40 GB | Student 0.8B + teacher 2B |
| Cloud training (2B) | A100 | 40 GB | Student 2B + teacher 9B |

---

## References & Sources

### Core Architecture Papers
1. **BitNet b1.58**: Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (2024) — [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
2. **Gated DeltaNet**: Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule" (2024) — [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
3. **Qwen3.5**: Qwen Team, "Qwen3.5: Towards Native Multimodal Agents" (2026) — [qwen.ai/blog](https://qwen.ai/blog?id=qwen3.5)
4. **CALM**: Shao et al., "Continuous Autoregressive Language Models" (2025) — [arXiv:2510.27688](https://arxiv.org/abs/2510.27688)
5. **Scaling Laws**: Kaplan et al., "Scaling Laws for Neural Language Models" (2020) — [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

### Interpretability & Structure
6. **RYS / LLM Neuroanatomy**: David Noel Ng, "How I Topped the LLM Leaderboard Without Changing a Single Weight" (2026) — [dnhkng.github.io/posts/rys](https://dnhkng.github.io/posts/rys/)
7. **On the Biology of a Large Language Model**: Lindsey, Gurnee et al. (Anthropic, 2025) — [transformer-circuits.pub](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)

8. **Transformer Dynamics**: Fernando & Guitchounts, "A neuroscientific approach to interpretability" (2025) — [arXiv:2502.12131](https://arxiv.org/abs/2502.12131)
9. **Statistical Physics of LM Reasoning**: Carson & Reisizadeh (MIT, 2025) — [arXiv:2506.04374](https://arxiv.org/abs/2506.04374)
10. **Neuroscience of Transformers**: König & Negrello, cortical column mapping (2026) — [arXiv:2603.15339](https://arxiv.org/abs/2603.15339)
11. **Entropy & Geometrization of LMs**: Yang, Boltzmann manifold (2024) — [arXiv:2407.21092](https://arxiv.org/abs/2407.21092)
12. **LLMs and Cognitive Science**: Niu, Liu, Bi et al. (2024) — [arXiv:2409.02387](https://arxiv.org/abs/2409.02387)

### Reasoning Compression
13. **CIB / Budget Forcing**: Massoli, Kuzmin & Behboodi (Qualcomm, 2026) — [arXiv:2603.08462](https://arxiv.org/abs/2603.08462)
14. **Physics of Language Models**: Zeyuan Allen-Zhu (MIT/ICML 2024) — [physics.allen-zhu.com](https://physics.allen-zhu.com/)

### Chain-of-Thought & Monitoring
15. **CoT Monitorability**: Anthropic et al. (2025) — [arXiv:2507.11473](https://arxiv.org/abs/2507.11473)
16. **The Markovian Thinker (Delethink)**: (2025) — [arXiv:2510.06557](https://arxiv.org/abs/2510.06557)

### Training Methods
17. **Distilling Step-by-Step**: Hsieh et al. (Google, ACL 2023) — [arXiv:2305.02301](https://arxiv.org/abs/2305.02301)
18. **MiniPLM**: Gu et al. (THU, ICLR 2025) — [arXiv:2410.17215](https://arxiv.org/abs/2410.17215)
19. **Pre-training Distillation**: Peng et al. (ACL 2025) — [aclanthology.org/2025.acl-long.181](https://aclanthology.org/2025.acl-long.181/)

### Convergence Thread
20. **Seven Papers Thread**: Sakeeb Rahman (@sakeeb.rahman on Threads) — [Gist](https://gist.github.com/Sakeeb91/79fe75658be60fbd1691d12c29d38366)

### Implementations Referenced
21. **bitlinear-pytorch**: Ingur Veken — [pypi.org/project/bitlinear-pytorch](https://pypi.org/project/bitlinear-pytorch/)
22. **qvac-fabric-llm.cpp**: Tetherto — [github.com/tetherto/qvac-fabric-llm.cpp](https://github.com/tetherto/qvac-fabric-llm.cpp)
23. **Flash Linear Attention**: (Triton kernels for DeltaNet) — used by vLLM for Qwen3.5 support

---

## License

Apache 2.0 (matching Qwen3.5 base model license)

## Acknowledgments

- Qwen Team (Alibaba) for Qwen3.5 architecture and open-source weights
- David Noel Ng for the RYS discovery and LLM Neuroanatomy framework
- Sakeeb Rahman for the seven-paper convergence synthesis
- Anthropic Interpretability Team for circuit tracing research
- Qualcomm AI Research for the CIB framework
- WeChat AI / Tencent for CALM and the Energy Transformer

---

*This project was developed as a research exploration combining insights from mechanistic interpretability, quantization theory, and information-theoretic reasoning compression. The goal is not a production model, but a proof-of-concept for selective hybrid-precision quantization guided by architectural understanding.*
