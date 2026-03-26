# RecurrentBitNet v3: Qwen3.5-2B as Foundation

## Why Qwen3.5-2B Changes Everything

We were building recurrence from scratch. Qwen3.5 already HAS it.

### The Architecture (24 layers)
```
6 × (3 × [Gated DeltaNet → FFN] → 1 × [Full Attention → FFN])

Layer 0:  DeltaNet (linear, recurrent state)
Layer 1:  DeltaNet (linear, recurrent state)  
Layer 2:  DeltaNet (linear, recurrent state)
Layer 3:  Full Attention (global context checkpoint)
Layer 4:  DeltaNet
Layer 5:  DeltaNet
Layer 6:  DeltaNet
Layer 7:  Full Attention
...repeats 6 times total...
Layer 23: Full Attention
```

### Why DeltaNet IS Recurrence
Gated DeltaNet maintains a **fixed-size state matrix** S ∈ R^{d_k × d_v}
per head, updated with a delta rule:

  S_t = α_t · S_{t-1} + β_t · v_t · k_t^T

This IS an RNN. Each DeltaNet layer carries forward a compressed
state of everything it has seen. The delta rule means it actively
CORRECTS its memory (not just appending like standard attention).

The full attention layers every 4th position act as "checkpoints"
that can do global retrieval when the compressed state isn't enough.

**This is exactly the encoder/reasoning/decoder structure we designed,
but discovered independently by the Qwen team at massive scale.**

## Specs: Qwen3.5-2B-Base
- Params: 2B
- d_model: 2048
- Vocab: 248,320 (massive — handles 201 languages)
- Layers: 24 (18 DeltaNet + 6 Full Attention)
- DeltaNet: 16 heads, 128 dim (linear O(1) per token)
- Full Attention: 8 Q heads, 2 KV heads (GQA), 256 dim
- FFN: 6144 intermediate (SwiGLU)
- Context: 262K native, up to 1M
- Multi-token prediction trained
- License: Apache 2.0
- Also available: 0.8B, 4B, 9B in same family

## The New Plan: Don't Build the Architecture, Build the Efficiency

### Stop Reinventing, Start Adapting
Our custom RecurrentBitNet was solving a problem that Qwen3.5 already
solved better at billion-dollar scale. The DeltaNet+Attention hybrid
IS the "encoder/reasoning/decoder" architecture:

- DeltaNet layers = recurrent reasoning (compressed state, O(1))
- Full Attention layers = global context checkpoints (every 4th layer)
- MoE in larger variants = parameter efficiency

What we CAN uniquely contribute:
1. **BitLinear conversion** of the DeltaNet layers (they're just linear ops!)
2. **Distillation** from larger Qwen3.5 variants
3. **CIB-style training** for reasoning efficiency
4. **RYS-style analysis** to find which layers to compress most aggressively

---

## BitLinear Conversion Strategy

### Which layers to convert?

DeltaNet layers are PERFECT for BitLinear because:
1. They use linear projections (Q, K, V, output) — direct BitLinear swap
2. They're already recurrent — ternary weights don't hurt as much 
   because the state matrix S accumulates information across tokens
3. There are 18 of them (75% of layers) — maximum param savings
4. The delta rule (error correction) may naturally compensate for
   quantization noise on each step

Full Attention layers should stay FP16 because:
1. Only 6 of them (25% of layers) — small fraction of total params
2. They do global retrieval — precision matters for exact recall
3. GQA with 2 KV heads is already very parameter-efficient

### Size at different precisions

```
                        FP16      BitLinear DeltaNet + FP16 Attn
Embeddings (248K×2048): 1.02 GB   507 MB (Q8)
DeltaNet layers (18):   2.17 GB   543 MB (TQ2_0)
Full Attn layers (6):   0.72 GB   0.72 GB (keep FP16)
Other (norms, etc):     ~50 MB    ~50 MB
────────────────────────────────────────────
TOTAL:                  3.96 GB   ~1.84 GB
```

**Under 2 GB for inference.** A 2B-effective model in under 2 GB.
For comparison, vanilla Q4 quantization of the same model: ~1.2 GB.
We're slightly larger but with better quality on the DeltaNet layers.

---

## Training Strategy

### Approach: Post-Training BitLinear Conversion with Distillation

This is NOT training from scratch. It's converting an existing
excellent model to be more efficient while preserving quality.

```
Phase 1: Load Qwen3.5-2B-Base (pretrained, full precision)
Phase 2: Replace DeltaNet layers' nn.Linear → BitLinear  
Phase 3: Short distillation fine-tune to recover quality
Phase 4: (Optional) Reasoning fine-tune with CIB objective
```

### Phase 1-2: Architecture Surgery
```python
from transformers import AutoModelForCausalLM

# Load the pretrained model
teacher = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-2B-Base", torch_dtype=torch.bfloat16
)
student = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-2B-Base", torch_dtype=torch.float32  
)

# Convert DeltaNet layers to BitLinear (every non-attention layer)
for layer_idx, layer in enumerate(student.model.layers):
    if is_deltanet_layer(layer):  # layers 0,1,2, 4,5,6, 8,9,10...
        convert_linear_to_bitlinear(layer)
    # Full attention layers (3, 7, 11, 15, 19, 23): keep FP16
```

### Phase 3: Distillation Fine-tune

Teacher: Qwen3.5-2B-Base (original, frozen, BF16)
  or: Qwen3.5-4B/9B (larger, same tokenizer, even better teacher)
Student: Qwen3.5-2B with BitLinear DeltaNet layers

```python
# Loss: KL divergence on logits + CE on hard labels
loss = α * CE(student_logits, labels) + β * KL(student, teacher)

# Training data (high quality, not raw web):
# - FineWeb-Edu (curated educational)
# - OpenWebMath 
# - The Stack v2
# Total: 5-10B tokens (sufficient with distillation)

# Training time:
# On A100: ~24-48 hours for 5B tokens
# On 3060: Use Qwen3.5-0.8B as smaller student, teacher=2B
```

### Memory Budget for Different Hardware

```
On RTX 3060 (12GB) — use the 0.8B variant:
  Student 0.8B (FP32 training): ~3.2 GB
  Teacher 2B (BF16, frozen): ~4 GB  
  Gradients + activations: ~3 GB
  Total: ~10.2 GB ✓

On A100 (80GB) — use the 2B variant:
  Student 2B (FP32 training): ~8 GB
  Teacher 9B (BF16, frozen): ~18 GB
  Gradients + activations: ~12 GB
  Total: ~38 GB ✓ (with room for large batches)

On Colab T4 (16GB) — use the 0.8B variant:
  Same as 3060 setup ✓
```

---

## What's Actually Novel Here

### 1. Selective BitLinear on Hybrid Architecture
Nobody has applied BitLinear quantization selectively to a 
DeltaNet/Attention hybrid. The insight that DeltaNet's recurrent
state accumulation may be MORE robust to ternary weights than 
standard attention is untested and publishable.

### 2. DeltaNet + BitLinear = Ultra-Efficient Recurrence  
DeltaNet's state update: S_t = α·S_{t-1} + β·v·k^T
With BitLinear: the v and k projections are ternary.
The state matrix S still accumulates in full precision.
This means the "memory" stays accurate while the 
"computation to update it" is extremely cheap.

### 3. RYS-Style Analysis of DeltaNet vs Attention Layers
Run the RYS brain scan protocol on Qwen3.5-2B:
- Which DeltaNet layers form "reasoning circuits"?
- Do the full attention layers act as "circuit boundaries"?
- Can we identify which DeltaNet layers to quantize more aggressively?

### 4. CIB Applied to Hybrid Linear/Full Attention
The CIB's "Attention Paradox" (prompt X is visible through attention)
only applies to the full attention layers. DeltaNet layers have 
compressed state — they naturally implement an information bottleneck.
This means CIB regularization should target the full attention layers
specifically, while DeltaNet layers are self-regularizing.

---

## Immediate Next Steps

### 1. Start with Qwen3.5-0.8B on your 3060
```bash
# Download
huggingface-cli download Qwen/Qwen3.5-0.8B-Base \
  --local-dir ~/Repositories/ai_workspace/recurrent_bitnet/models/qwen35-0.8b

# Inspect architecture  
python -c "
from transformers import AutoModelForCausalLM, AutoConfig
config = AutoConfig.from_pretrained('Qwen/Qwen3.5-0.8B-Base')
print(config)
"
```

### 2. Write the BitLinear surgery script
- Load model, identify DeltaNet vs Attention layers
- Convert DeltaNet linear projections to BitLinear
- Verify forward pass still works (output will be degraded)
- Measure initial quality drop

### 3. Distillation training (local)
- Teacher: Qwen3.5-0.8B (original, frozen)
- Student: Qwen3.5-0.8B (BitLinear DeltaNet layers)
- Data: FineWeb-Edu streamed
- Goal: recover quality lost from BitLinear conversion

### 4. Scale up (cloud)
- Teacher: Qwen3.5-4B or 9B
- Student: Qwen3.5-2B (BitLinear DeltaNet layers)
- Data: FineWeb-Edu + OpenWebMath + Code

### 5. RYS Analysis
- Run brain scan on original vs BitLinear variant
- Which layers tolerate quantization best?
- Does the hybrid architecture show the same "circuit" patterns?
