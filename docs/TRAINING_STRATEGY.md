# RecurrentBitNet v2: Revised Training Strategy

## Why the Golf Run Plateaued

The loss plateaued around 5.8 nats because:
1. **Raw next-token prediction on FineWeb is the hardest task** for a small model
2. The model was learning everything from scratch — world knowledge, grammar,
   reasoning, formatting — all at once, all from 2-bit weights
3. 50K steps (~800M tokens) is insufficient for even a 50M param model
4. No teacher signal — the model discovers ALL patterns from raw text

## The Fix: Three-Phase Training

### Phase 0: Initialize from a Pretrained Base (CRITICAL)
**Don't train from scratch.** Start from an existing small model and ADAPT it.

Deprecated Options [Use Qwen3.5-2B instead](docs/V3_QWEN35.md):
- **Qwen2.5-0.5B** or **Qwen2.5-1.5B**: Strong base, good tokenizer (151K vocab)
- **SmolLM2-360M** or **SmolLM2-1.7B**: HuggingFace's optimized small models
- **Phi-4-mini (3.8B)**: Microsoft's best small model (needs more VRAM)

The model already knows language, grammar, world knowledge.
We only need to teach it: (a) to work with BitLinear weights, and
(b) to use recurrence effectively.

**Approach**: Load pretrained weights into our architecture, then
progressively quantize to BitLinear using STE. This is far more
efficient than training BitLinear from scratch.

### Phase 1: Architecture Surgery + BitLinear Conversion
Take a pretrained model and restructure it into our architecture:

```
Pretrained Qwen2.5-0.5B (24 layers):
  Layers 0-2   → Encoder Stack (non-recurrent, keep FP16)
  Layers 3-20  → Select best 6 layers → Reasoning Core (convert to BitLinear)
  Layers 21-23 → Decoder Stack (non-recurrent, keep FP16)

  Reasoning Core: 6 unique blocks × R recurrence
  - Initialize with layers that RYS-style probing shows are most "reasoning-like"
  - Convert their nn.Linear → BitLinear with STE
  - Add iteration embeddings (initialized to zero = identity at start)
```

**Key insight**: The encoder/decoder stay at FP16 (they're small, ~15% of params).
Only the reasoning core gets BitLinear. This is a hybrid precision architecture:
- Encoding/decoding: full precision (quality-sensitive, not repeated)
- Reasoning: ternary (repeated R times, so memory savings multiply)

### Phase 2: Distillation Training (THE BIG LEVER)
Use the ORIGINAL pretrained model as the teacher.

```
Teacher: Qwen2.5-0.5B (frozen, FP16, standard 24 layers)
Student: RecurrentBitNet (surgically modified from same weights)

Loss = α × L_CE(student, target_tokens)           # next-token prediction
     + β × L_KD(student_logits, teacher_logits)    # logit distillation
     + γ × L_aux(intermediate_outputs, target)      # per-recurrence auxiliary

where α=0.5, β=1.0, γ=0.3 (distillation dominates)
```

**Why this works so well:**
The student already knows language (from pretrained init). The teacher provides
"soft labels" — full probability distributions over vocabulary — which contain
FAR more information than hard next-token labels.

Example: For the token following "The capital of France is",
- Hard label: "Paris" (1 bit of information)
- Soft labels: P(Paris)=0.92, P(Lyon)=0.03, P(the)=0.01... (rich distribution)

The soft labels teach the student about semantic relationships between tokens
that take billions of tokens to learn from raw text alone.

**Data for distillation**:
- FineWeb-Edu (curated educational text, much higher quality than raw FineWeb)
- OpenWebMath (mathematical text)
- The Stack v2 (code — code is highly structured, great for learning patterns)
- 5-10B tokens total is sufficient with distillation

### Phase 3: Progressive Recurrence (After Distillation Converges)
Once the BitLinear weights are good, increase recurrence:

```
Stage 3a: R=1 → R=2 (learn to refine representations)
  - Fine-tune for 5K steps on high-quality data
  - Teacher still provides logits at R=1 depth
  - Student must beat teacher using R=2 passes

Stage 3b: R=2 → R=3 (deeper refinement)
  - Fine-tune for 5K steps
  - Now the student should surpass the teacher on hard examples

Stage 3c: R=3 → R=4 (full depth)
  - Final fine-tune
  - Focus on reasoning-heavy data (math, logic, code)
```

### Phase 4: Reasoning Fine-tune with CIB + Causal Halting

This is where the Gemini/computational mechanics ideas become practical:

**Causal Halting (from Computational Mechanics)**:
The ACT mechanism's halting condition uses the stability of hidden states:
```python
# Halt when the representation stabilizes (causal state reached)
halt_signal = 1.0 - cosine_similarity(h_r, h_{r-1})
# If halt_signal < threshold, stop recurring — the circuit found its answer
```
This is the practical version of "has the model reached a recurrent causal state?"
If h_r ≈ h_{r-1}, further recurrence won't change the output — save compute.

**CIB Regularization (from Qualcomm paper)**:
After the model can reason with recurrence, add the information cost:
```python
# Each recurrence step's output predicts next tokens
# Measure how much NEW info each step adds
info_gain_r = CE(predict_from_h_r, target) - CE(predict_from_h_{r-1}, target)
# Penalize recurrence steps that add < threshold info
L_CIB = -β * max(0, threshold - info_gain_r)
```

**Auxiliary Recurrence Loss with Transient/Recurrent Distinction**:
```python
# Early recurrence steps (transient): high tolerance for state variance
# Late recurrence steps (recurrent causal): penalize state instability
for r in range(R):
    aux_weight = α_base * (r / R)  # increases with depth
    L_aux += aux_weight * CE(project(h_r), target)
    if r >= R // 2:  # second half: penalize large state changes
        L_stability += λ * (1 - cosine_sim(h_r, h_{r-1}))
```

---

## Data Strategy

### What DOESN'T work (Golf lesson)
- Raw FineWeb: too noisy, too much redundancy, too diverse
- Small vocab (8K): insufficient for real model quality
- BPE from scratch: wasted effort when pretrained tokenizers exist

### What DOES work for sub-1B models

**Tier 1: Distillation data (highest impact)**
- Run teacher (Qwen2.5-1.5B or 7B) on curated text
- Save logits for top-k tokens (k=50, saves 95% of storage)
- Student trains on both hard labels AND soft logits
- 5-10B tokens with distillation > 100B without

**Tier 2: High-quality curated data**
- FineWeb-Edu: pre-filtered educational content (much better than raw FineWeb)
- CosmosPedia v2: synthetic textbook-style data
- OpenWebMath: mathematical reasoning text
- The Stack v2 (dedup): structured code
- Proof-Pile-2: formal mathematics
- SlimPajama: deduplicated, balanced web text

**Tier 3: Reasoning-specific (Phase 3-4)**
- OpenMathInstruct-2: 14M math problems with solutions
- MetaMathQA: augmented math reasoning
- CodeContests: competitive programming
- ARC / HellaSwag / MMLU: diverse reasoning benchmarks

---

## Practical Implementation Plan

### Step 1: Choose Base Model
**Recommendation: Qwen2.5-0.5B** (494M params, 24 layers, d=896, 14 heads)
- Strong performance for size
- Good tokenizer (151K vocab, great for multilingual + code)
- Small enough to train on 3060 (12GB)
- Qwen2.5-1.5B as teacher (1.5B, same tokenizer = distillation-friendly)

### Step 2: Architecture Surgery
```python
# Load pretrained Qwen2.5-0.5B
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# Extract layers
encoder_layers = base.layers[0:3]      # first 3: encoding
reasoning_layers = base.layers[3:21]    # middle 18: pick best 6
decoder_layers = base.layers[21:24]     # last 3: decoding

# Select 6 reasoning layers (use RYS-style probing, or just evenly space)
selected = [reasoning_layers[i] for i in [0, 3, 6, 9, 12, 15]]

# Convert selected layers' nn.Linear → BitLinear
for block in selected:
    block.self_attn.q_proj = convert_to_bitlinear(block.self_attn.q_proj)
    block.self_attn.k_proj = convert_to_bitlinear(block.self_attn.k_proj)
    # ... etc for all linear layers in the block

# Assemble RecurrentBitNet
model = RecurrentBitNet(
    encoder=encoder_layers,          # FP16, non-recurrent
    reasoning=selected,              # BitLinear, recurrent × R
    decoder=decoder_layers,          # FP16, non-recurrent
    vocab_size=151936,               # Qwen tokenizer
)
```

### Step 3: Distillation Training
```python
teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B").eval()

for batch in dataloader:
    # Student forward (with recurrence)
    student_logits = model(batch.input_ids)

    # Teacher forward (no recurrence, frozen)
    with torch.no_grad():
        teacher_logits = teacher(batch.input_ids)

    # Combined loss
    loss_ce = F.cross_entropy(student_logits, batch.labels)
    loss_kd = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction='batchmean'
    ) * T * T   # temperature-scaled KL divergence

    loss = 0.5 * loss_ce + 1.0 * loss_kd
```

### Step 4: Memory Budget on 3060 (12GB)
```
Student (Qwen 0.5B-based, mixed precision):
  Encoder (FP16): 3 blocks ≈ 150 MB
  Reasoning (BitLinear, FP32 latent): 6 blocks ≈ 300 MB
  Decoder (FP16): 3 blocks ≈ 150 MB
  Embeddings (FP16): ≈ 280 MB
  Optimizer states: ≈ 1.5 GB
  Gradients + activations (grad ckpt): ≈ 2 GB
  Student total: ≈ 4.4 GB

Teacher (Qwen 1.5B, frozen, FP16, no grad):
  Just forward pass + KV cache: ≈ 3 GB
  (No optimizer states needed — frozen)

TOTAL: ≈ 7.4 GB → FITS on 3060 (12GB) ✓
Buffer for batch + misc: ≈ 4.6 GB remaining
```

### Step 5: Expected Results

**Without distillation (golf run)**:
- 50K steps, loss plateau at ~5.5-6.0 nats, BPB ~1.5+
- Model knows nothing at start, learns slowly

**With pretrained init + distillation (this plan)**:
- Student starts at near-teacher quality (loss ~3.5 nats from init)
- Distillation recovers most quality lost from BitLinear conversion
- Progressive recurrence should push BELOW teacher quality on hard tasks
- Expected: loss ~3.0-3.5 nats, BPB ~0.8-1.0
- On reasoning benchmarks: recurrence gives free accuracy boost

**The prize**: A model that:
- Has 494M base params but 430M+ effective (with recurrence)
- Runs inference at ~125 MB (BitLinear reasoning core + FP16 encoder/decoder)
- Beats its teacher (Qwen 0.5B) on reasoning tasks via extra recurrence
- Fits comfortably on a Raspberry Pi or phone for inference
- Trades speed for accuracy by varying recurrence depth R

---

## Summary: What Changed from Golf

| Aspect | Golf (failed) | v2 (this plan) |
|--------|--------------|----------------|
| Init | Random | Pretrained Qwen2.5 |
| Training signal | Hard labels only | Distillation + hard labels |
| Data | Raw FineWeb | FineWeb-Edu + math + code |
| Architecture | All recurrent | Encoder/decoder + recurrent core |
| Quantization | All BitLinear | Hybrid: FP16 enc/dec + BitLinear core |
| Recurrence | Fixed from start | Progressive curriculum |
| Loss | CE only | CE + KD + auxiliary + CIB |
| Tokenizer | 8K BPE (trained) | 151K Qwen (pretrained) |
