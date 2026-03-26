# RYS: LLM Neuroanatomy — Deep Dive

> Source: David Noel Ng, [dnhkng.github.io/posts/rys](https://dnhkng.github.io/posts/rys/), 2026  
> Models: [dnhkng/RYS-XLarge](https://huggingface.co/dnhkng/RYS-XLarge)

## The Discovery

In 2024, David Noel Ng took the #1 spot on the HuggingFace Open LLM
Leaderboard with `dnhkng/RYS-XLarge` — a model created by duplicating
7 middle layers of Qwen2-72B. No training. No weight modification.
Just duplicating layers 45-51 so they execute twice per forward pass,
increasing the model from 72B to 78B parameters (all duplicates).

### Results on Open LLM Leaderboard
| Metric | RYS-XLarge | Improvement |
|--------|-----------|-------------|
| Average | **44.75** | **+2.61%** |
| IFEval (0-Shot) | 79.96 | -2.05% |
| BBH (3-Shot) | 58.77 | +2.51% |
| MATH Lvl 5 (4-Shot) | 38.97 | +8.16% |
| GPQA (0-shot) | 17.90 | +2.58% |
| MuSR (0-shot) | 23.72 | **+17.72%** |
| MMLU-PRO (5-shot) | 49.20 | +0.31% |

5 of 6 benchmarks improved. The optimization used ONLY math guesstimate
and EQ-Bench probes — the leaderboard was pure out-of-sample validation.

## The Brain Scanning Protocol

### Configuration Space
For model with N layers, define configuration (i, j) where i < j:
- Run layers 0..j normally
- Loop back and re-run layers i..j
- Continue j..N

```
Example: N=9, (i=2, j=6)

  0 → 1 → 2 → 3 → 4 → 5 → 6 ─┐
       ┌────────────────────────┘
       └→ 2 → 3 → 4 → 5 → 6 → 7 → 8

  Duplicated: [2, 3, 4, 5, 6]
  Path: [0, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 8]
```

For N=80 (Qwen2-72B): C(80,2) = 3,160 valid configurations.

### Proxy Tasks (Fast, Objective, Orthogonal)
1. **Hard math guesstimate**: "What is the cube root of 74,088,893,247?"
   - No chain-of-thought, just output the number
   - Custom partial-credit scoring for near-misses
   - Tests pure numerical intuition in a single token

2. **EQ-Bench**: Complex social scenarios predicting emotional state intensity
   - "How angry/surprised/guilty would this person feel on a scale of 0-100?"
   - Tests theory of mind, empathy, social inference
   - Completely orthogonal to math

If both improve simultaneously → structural improvement, not task-specific.

## Key Findings

### 1. Three-Zone Anatomy
```
Layers 0..~12  (first ~15%)  → ENCODING zone
  - Converting input format to abstract representation
  - Duplicating these HURTS performance
  - Format-specific, not generalizable

Layers ~12..~60 (middle ~55%) → REASONING zone
  - Abstract computation in format-independent space
  - Duplicating these HELPS performance
  - Contains "reasoning circuits" of 5-7 layers each

Layers ~60..~79 (last ~25%)  → DECODING zone
  - Converting abstract representation to output tokens
  - Duplicating these HURTS or has no effect
  - Format-specific, output-oriented
```

### 2. Circuits Are Indivisible
- Duplicating a SINGLE middle layer: almost never helps
- Duplicating a BLOCK of 5-7 layers: significant improvement
- Conclusion: Layers work as multi-step "recipes" that must execute complete

### 3. Circuit Boundaries Are Sharp
- Including even 1 layer from a neighboring circuit destroys benefit
- The boundary between circuits is precise, not gradual
- Each circuit performs a complete cognitive operation

### 4. Bad Duplications Cause "Brain Damage"
- Wrong circuits duplicated → personality disorders, loops, incoherence
- One model: "Let's act like cowboys! Yeehaw!" followed by pages of "hahaha"
- This is analogous to neurological deficits, not general degradation

## Implications for Qwen3.5 and BitLinear

### Natural Circuit Boundaries in Qwen3.5
The 3:1 DeltaNet/Attention layout creates natural circuit boundaries:
```
Layers 0-2:   DeltaNet triplet (potential circuit)
Layer 3:      Full Attention    (circuit boundary/checkpoint)
Layers 4-6:   DeltaNet triplet (potential circuit)
Layer 7:      Full Attention    (circuit boundary/checkpoint)
...
```

Each DeltaNet triplet between full-attention layers is a candidate
reasoning circuit. The full-attention layers act as "checkpoints"
that can retrieve any information the DeltaNet compressed state missed.

### Selective Quantization Guided by RYS
1. Run RYS brain scan on Qwen3.5-2B to identify circuit structure
2. Quantize DeltaNet layers in the REASONING zone most aggressively (TQ2_0)
3. Keep DeltaNet layers near encoding/decoding zones at higher precision
4. Keep ALL full attention layers at FP16 (they're circuit boundaries)

### RYS + Weight Tying (Future Work)
RYS proves that reasoning circuits can execute twice with benefit.
For a DeltaNet triplet (layers i, i+1, i+2):
- If running it twice helps, maybe we can SHARE weights and save params
- This is the Universal Transformer / RYS connection
- The DeltaNet state accumulation makes this particularly natural:
  first pass builds initial state, second pass refines it

## RYS Math Scoring (from David Ng)

LLMs fail arithmetic in weird ways — almost right but with dropped
or transposed digits. Binary scoring loses signal. Ng's partial credit:

```python
def calculate_score(actual, estimate):
    """Calculate score comparing actual vs estimated answer.
    Handles LLM-style near-misses: dropped digits, transpositions."""
    try:
        actual_str = str(int(actual))
        estimate_str = str(int(estimate))
    except (ValueError, OverflowError):
        return 0

    max_length = max(len(actual_str), len(estimate_str))
    actual_padded = actual_str.ljust(max_length, "0")
    estimate_padded = estimate_str.ljust(max_length, "0")
    padding_size = max_length - min(len(actual_str), len(estimate_str))

    actual_int = int(actual_padded)
    estimate_int = int(estimate_padded)

    if max(actual_int, estimate_int) == 0:
        return 0
    relative_diff = abs(actual_int - estimate_int) / max(actual_int, estimate_int)
    correction_factor = 1 - (padding_size / max_length)
    score = (1 - relative_diff) * correction_factor
    return max(0, min(score, 1))
```

## Leaderboard Legacy
As of early 2026, the top 4 models on the Open LLM Leaderboard are
ALL descendants of RYS-XLarge, with various fine-tunes on top:
1. MaziyarPanahi/calme-3.2-instruct-78b — 52.08
2. MaziyarPanahi/calme-3.1-instruct-78b — 51.29
3. dfurman/CalmeRys-78B-Orpo-v0.1 — 51.23
4. MaziyarPanahi/calme-2.4-rys-78b — 50.77
