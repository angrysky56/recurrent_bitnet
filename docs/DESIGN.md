# RecurrentBitNet: A Real Model

## What We Learned

### From Parameter Golf
- BitLinear is a drop-in nn.Linear replacement: train FP32/BF16, deploy ternary
- Recurrence multiplies depth without params — 7 blocks × 6 rec = 42 effective layers
- Iteration embeddings let shared weights compute different functions per pass
- Gradient checkpointing makes deep recurrence VRAM-feasible
- Pipeline works: tokenizer → streaming data → BitLinear+STE → export

### From RYS (LLM Neuroanatomy)
- Transformers have functional anatomy: encoder layers → reasoning circuits → decoder layers
- Reasoning circuits are 5-7 layer indivisible units — duplicating partial circuits hurts
- Duplicating complete reasoning circuits improves ALL benchmarks (not just one task)
- First ~15% and last ~25% of layers are encoding/decoding — NOT reasoning
- Single-layer duplication never helps — circuits must execute as complete units
- This is orthogonal to fine-tuning: you can stack both

### From CIB (Reasoning as Compression)
- Reasoning traces are 41% compressible without accuracy loss
- Penalize by semantic information content (surprisal under a prior), not raw token count
- The "Attention Paradox": standard Information Bottleneck fails for transformers
- CIB objective: maximize I(Z;Y|X) - β·I(X;Z) — sufficiency minus minimality
- Length penalties are a special case of CIB with uniform (maximum entropy) prior

### From CALM (Continuous Autoregressive LMs)
- Next-vector prediction: compress K tokens → 1 vector, predict vectors not tokens
- Autoencoder with 99.9% reconstruction accuracy at K=4
- Energy Score (strictly proper scoring rule) enables single-step generation
- BrierLM: likelihood-free evaluation metric that correlates 0.99 with perplexity
- At K=4: matches baselines at 44% fewer training FLOPs

### From the Seven-Paper Convergence
- Residual stream is a dynamical system with attractor-like behavior (Fernando)
- 4 reasoning regimes with metastable transitions (Carson)
- Cortical column mapping: Q/K do routing, V carries content (König)
- Default circuits = maximum entropy states (Yang ↔ Anthropic)
- Planning = metastable basin holding multiple candidates before commitment

---

## Architecture: RecurrentBitNet v2

### Key Insight from RYS Applied to Training
RYS discovered that pre-trained models already develop reasoning circuits.
We can do better: **train a model that is explicitly architected for circuits.**

Instead of hoping circuits emerge and then duplicating them post-hoc,
we build the circuit structure into the architecture from the start:

```
┌──────────────────────────────────────────────────────────┐
│                    RecurrentBitNet v2                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─── Encoder Stack (NON-RECURRENT) ───────────────────┐ │
│  │  3 unique transformer blocks                        │ │
│  │  Purpose: Parse input format → abstract repr        │ │
│  │  (RYS: first ~15% of layers are encoding)           │ │
│  └─────────────────────────────────────────────────────┘ │
│                          ↓                               │
│  ┌─── Reasoning Core (RECURRENT × R) ─────────────────┐ │
│  │  N unique blocks, each a complete 1-layer circuit   │ │
│  │  Repeated R times with iteration embeddings         │ │
│  │  + Adaptive halting (ACT) per-token                 │ │
│  │  (RYS: middle layers are reasoning circuits)        │ │
│  │  (CIB: later passes = higher information density)   │ │
│  └─────────────────────────────────────────────────────┘ │
│                          ↓                               │
│  ┌─── Decoder Stack (NON-RECURRENT) ──────────────────┐ │
│  │  3 unique transformer blocks                        │ │
│  │  Purpose: Abstract repr → output tokens             │ │
│  │  (RYS: last ~25% of layers are decoding)            │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  Output: LM head (tied embeddings)                       │
└──────────────────────────────────────────────────────────┘
```

### Why separate encoder/decoder from reasoning core?

RYS proved this empirically:
- Duplicating early layers HURTS (they're format-specific encoders)
- Duplicating late layers HURTS (they're format-specific decoders)  
- Duplicating middle layers HELPS (they're general reasoning circuits)

In a standard transformer, ALL layers are unique and non-recurrent,
so the model must use the same parameters for encoding, reasoning,
AND decoding. This forces a compromise.

By explicitly separating concerns:
- Encoder blocks learn to be great at parsing (not wasted on reasoning)
- Reasoning blocks learn to be great at thinking (not wasted on encoding)
- Decoder blocks learn to be great at output (not wasted on either)
- Only reasoning blocks recur — the encoding/decoding is done once

---

## Scale Targets

### Inference Target: Fits on your 3060 (12GB VRAM)
At BitNet TQ2_0 (~0.25 bytes/param):
- 1B params → ~250 MB → trivially fits
- 3B params → ~750 MB → trivially fits  
- Even 10B params → ~2.5 GB → easily fits

BitNet makes inference incredibly memory-efficient.
The question is what we can TRAIN.

### Training Reality
Training a BitNet model means FP32 latent weights + optimizer states:
- 1B params: ~12 GB (weights + Adam m,v) — tight on 3060
- 3B params: ~36 GB — needs CPU offloading or cloud
- 500M params: ~6 GB — comfortable on 3060

**Recommended: Start at 500M-1B, scale up with cloud compute.**

### Concrete Architecture: RecurrentBitNet-1B

```yaml
# ~460M unique params, but ~1.3B effective with recurrence
model:
  d_model: 1536
  n_heads: 12           # 128 per head (matches modern LLMs)
  d_ff: 4096            # ~2.7× expansion (SwiGLU)
  vocab_size: 32000     # Full BPE tokenizer
  max_seq_len: 4096
  
  # Encoder (non-recurrent): 3 unique blocks
  encoder_blocks: 3
  
  # Reasoning core (recurrent): 8 unique blocks × 4 recurrence = 32 effective
  reasoning_blocks: 8
  reasoning_recurrence: 4
  
  # Decoder (non-recurrent): 4 unique blocks  
  decoder_blocks: 4
  
  # Total effective depth: 3 + 32 + 4 = 39 layers
  # Total unique blocks: 3 + 8 + 4 = 15
  
  # Features
  pos_encoding: rope
  activation: swiglu
  norm: rmsnorm
  quantization: bitnet_b1.58  # ternary weights during forward
  iteration_embedding: true    # per-recurrence-step learned signal
  adaptive_halt: true          # ACT for variable recurrence depth
```

### Parameter Budget
```
Embedding:     32000 × 1536                    = 49.2M
Encoder:       3 blocks × (attn + ffn)         = 3 × 16.5M = 49.5M
Reasoning:     8 blocks × (attn + ffn + ACT)   = 8 × 16.5M = 132.0M  
Decoder:       4 blocks × (attn + ffn)         = 4 × 16.5M = 66.0M
Iter embeddings: 32 × 1536                     = 0.05M
Norms + misc:                                  = ~1.0M
────────────────────────────────────────────────────────────
Total unique params:                           ≈ 298M
Effective with recurrence:                     ≈ 430M

Training memory (BF16 + AdamW):
  Weights: 298M × 2 bytes                     = 596 MB
  Optimizer: 298M × 8 bytes (m + v + master)  = 2.4 GB
  Gradients: 298M × 2 bytes                   = 596 MB
  Activations (with grad ckpt):               ≈ 2-4 GB
  Total:                                      ≈ 6-8 GB → fits 3060!

Inference (BitNet TQ2_0):
  298M × 0.25 bytes                           ≈ 75 MB (!!)
```

A 430M-effective model that runs inference in 75 MB. That's the BitNet advantage.

---

## Training Innovations (What Golf Couldn't Do)

### 1. Progressive Recurrence Depth (Curriculum)
Start training with R=1 (no recurrence), then gradually increase:
- Steps 0-10K: R=1 (standard transformer, learn basic representations)
- Steps 10K-30K: R=2 (learn to refine on second pass)
- Steps 30K-60K: R=3 (learn deeper refinement)
- Steps 60K+: R=4 (full depth)

**Why**: The model first learns what good representations look like,
THEN learns how to iteratively refine them. Starting at full recurrence
forces the model to learn both simultaneously — harder optimization.

This is inspired by how RYS works: the circuits in pretrained models
already produce useful intermediate representations. We simulate
that by first training a "flat" model, then adding recurrence.

### 2. Differential Learning Rates
- Encoder blocks: standard LR (1e-3)
- Reasoning blocks: higher LR (2e-3) — these are the most reused weights
- Decoder blocks: standard LR (1e-3)  
- Embeddings: lower LR (5e-4) — large, slow-moving

**Why**: Reasoning blocks must learn to be useful across ALL recurrence
steps. They need faster adaptation to find multi-use representations.

### 3. Recurrence Dropout (Stochastic Depth for Recurrence)
During training, randomly skip recurrence steps with probability p=0.1.
If a reasoning block normally runs 4 times, sometimes it runs 2 or 3.

**Why**: Forces each recurrence step to be independently useful.
Prevents the model from relying on a specific number of passes.
At inference, run all passes for maximum quality, or fewer for speed.
This naturally creates an accuracy-speed knob without any additional
training — directly implementing the CIB accuracy-compression tradeoff.

### 4. Auxiliary Recurrence Loss
Add a lightweight prediction head after EACH recurrence step (not just
the last one). Each intermediate output should produce reasonable
next-token predictions, with a decaying weight:

  L_total = L_final + Σ(α^(R-r) × L_step_r)  where α ≈ 0.3

**Why**: Ensures early recurrence steps produce useful representations,
not just "setup" for later steps. This is the training-time version of
what RYS discovers empirically — each circuit pass should improve the
output. The auxiliary heads are discarded at inference.

### 5. CIB-Inspired Information Regularization
After initial pretraining, fine-tune with a CIB-style objective:
penalize the mutual information between input and reasoning trace.

In our recurrent model, this is uniquely powerful:
- Each recurrence step r produces a hidden state h_r
- We can measure how much NEW information each step adds
- Penalize steps that add low-information (redundant) computation
- This teaches the model to make each recurrence pass count

Practically: use a frozen copy of the model at R=1 as the prior Q_φ.
The recurrent model must beat the single-pass version's predictions,
or the extra computation is "wasted" and gets penalized.

### 6. Multi-Scale Context Training
Train with mixed context lengths in the same batch:
- 25% at 512 tokens (fast, frequent updates)
- 50% at 2048 tokens (standard)
- 25% at 8192 tokens (long-range dependencies)

**Why**: The recurrence mechanism should help with long-range
dependencies (more passes = more chance to integrate distant context).
Training at mixed lengths forces the model to use recurrence adaptively.

---

## Training Plan

### Phase 1: Pretraining (Local on 3060)
- Data: FineWeb-Edu 10BT, streamed
- Steps: 100K at batch=4, seq=1024, grad_accum=8
- Effective batch: 32K tokens/step → 3.2B tokens total
- Progressive recurrence: R=1→2→3→4 over training
- Optimizer: AdamW (β1=0.9, β2=0.95, wd=0.1)
- LR: 2e-3 peak with cosine decay to 2e-4
- Precision: BF16 mixed precision + STE for BitLinear
- Estimated time: ~3-5 days continuous on 3060
- Checkpoint every 5K steps

### Phase 2: Extended Pretraining (Cloud)
- Continue on A100/H100 for speed
- Increase to 10-30B tokens (Chinchilla scaling for ~300M params)
- Full R=4 recurrence throughout
- Add auxiliary recurrence loss
- Context length: mixed 512/2048/8192

### Phase 3: Reasoning Fine-tune
- Dataset: math/code/logic problems (OpenMathInstruct, CodeContests)
- GRPO with CIB-style semantic regularization
- Train the model to use recurrence for chain-of-thought
- Measure: does more recurrence = better reasoning?

### Phase 4: Evaluation & Analysis
- Run RYS-style brain scans on our trained model
- Do the recurrence steps show circuit-like behavior?
- Compare: our model at R=4 vs standard 430M-param model
- Measure the accuracy-speed tradeoff from recurrence dropout

---

## What Makes This Novel (Paper-Worthy)

No one has combined ALL of these in a single architecture:

1. **BitLinear quantization-aware training** (1.58-bit weights)
   + **Depth recurrence** (Universal Transformer style)
   + **Explicit encoder/decoder separation** (from RYS neuroanatomy)
   + **Progressive recurrence curriculum** (our innovation)
   + **Auxiliary per-recurrence loss** (our innovation)
   + **CIB-style information regularization** (adapted from Qualcomm)

The closest work:
- BitNet b1.58: ternary weights, but standard (non-recurrent) architecture
- Universal Transformer: recurrence, but FP32 weights and no circuit structure
- RYS: discovers circuits post-hoc, but doesn't train for them
- CIB: compresses reasoning, but on pretrained models not architecture

**Our claim**: By combining ternary quantization with structured recurrence
and information-theoretic training, we get a model that:
- Has 298M unique params but behaves like 430M+ 
- Runs inference in 75MB (vs ~600MB for FP16 equivalent)
- Naturally trades accuracy for speed via recurrence depth
- Develops interpretable reasoning circuits by construction

---

## Immediate Next Steps

### 1. Refactor model.py for v2 architecture
- Add separate encoder/decoder/reasoning stacks
- Add ACT (Adaptive Computation Time) halting
- Add auxiliary prediction heads per recurrence step
- Add recurrence dropout

### 2. Refactor train.py for v2 training
- Progressive recurrence schedule
- Differential learning rates per module group
- Auxiliary recurrence loss with decay weights
- Mixed context length batching
- Proper eval suite (not just BPB — add reasoning benchmarks)

### 3. Validate locally on 3060
- Train a small config (d=768, 2+4×3+2 blocks) for 10K steps
- Verify: does progressive recurrence beat fixed recurrence?
- Verify: does auxiliary loss improve early recurrence outputs?
- Verify: does recurrence dropout enable speed-accuracy tradeoff?

### 4. Scale to full config on cloud
- RecurrentBitNet-1B config on A100
- Full 10B+ token training
- Compare against baselines at same compute budget
