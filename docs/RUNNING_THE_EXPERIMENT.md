# Running the SPR Experiment — Practical Guide

**For**: RecurrentBitNet V2-SPR notebook (`RecurrentBitNet_V2_SPR.ipynb`)
**Hardware assumed**: Google Colab (T4 free / A100 Pro) or local RTX 3060 12GB + 64GB RAM

---

## TL;DR — The Minimum Viable Experiment

1. Open the notebook in Colab (or Jupyter locally)
2. **Run all cells top-to-bottom** with defaults — no changes needed for Run 1
3. Wait for training to finish (~3-8 hours depending on GPU and step count)
4. Change `use_spr=True` → `use_spr=False` in Section 6
5. **Restart runtime**, run all cells again for Run 2
6. Compare the probe tables from both runs

That's it. Everything else below is tuning for your situation.

---

## Step Counts — What's Realistic

The default 500K steps is aspirational (designed for A100 with days of compute).
Here's what actually makes sense:

| Goal | Steps | Time (T4) | Time (A100) | What you learn |
|------|-------|-----------|-------------|----------------|
| **Quick smoke test** | 10K | ~30 min | ~10 min | Does it train? Loss goes down? |
| **Proof of concept** | 50K | ~3 hrs | ~1 hr | First probe results at R=2 (step 50K+) |
| **Solid evidence** | 100K | ~6 hrs | ~2 hrs | Probes at R=2 and R=3, gate trajectories |
| **Full experiment** | 200K | ~12 hrs | ~4 hrs | All curriculum phases, R=4 probes |
| **Publication quality** | 500K | ~30 hrs | ~10 hrs | Converged model, all ablations |

**Recommendation for your first run**: Set `TOTAL_STEPS = 100_000`. This gives you
probes at R=1→R=2 transition (the most informative moment) and R=2→R=3, without
burning days of compute. You can always resume from checkpoint later.

The curriculum matters more than total steps. The key transitions are:

| Step | R | What happens |
|------|---|---|
| 0 | 1 | No recurrence — content subspace learns token semantics |
| 50K | 2 | **First recurrence** — temporal/state subspaces activate |
| 150K | 3 | Deeper reasoning — conjunctive binding opens |
| 300K | 4 | Full depth — halt scorer learns to terminate |

The probes only run when R > 1, so you need at least ~60K steps to see your first
probe results (50K to enter R=2, then 10K more to hit the first PROBE_EVERY).

If you set TOTAL_STEPS = 100K, adjust the curriculum to squeeze more out of it:

```python
# Faster curriculum for 100K runs:
CURRICULUM = [
    (0,       1),
    (25_000,  2),   # Enter R=2 earlier
    (60_000,  3),   # Enter R=3 earlier
    (85_000,  4),   # Brief R=4 phase
]
```

---

## What to Change and Where

### Section 6 — The Only Cell You Must Touch

This is the model instantiation cell. It has one toggle:

```python
config = ModelConfig(use_spr=True)   # ← THIS LINE
```

- **Run 1**: Leave as `use_spr=True` (SPR mode)
- **Run 2**: Change to `use_spr=False` (Baseline mode)

Everything else auto-configures based on this flag. Output directories,
checkpoint paths, plot labels — all keyed off this single boolean.

### Section 7 — Training Settings (Optional Tweaks)

These are the settings you might want to change:

```python
TOTAL_STEPS   = 500_000   # ← Reduce to 100_000 for first experiment
BATCH_SIZE    = 8          # ← Reduce to 4 if you OOM on T4
PROBE_EVERY   = 10_000    # ← Keep as-is (you want ~5-10 probe points)
```

**If you hit OOM errors**: The training loop automatically skips OOM steps and
warns you. If it happens repeatedly, reduce BATCH_SIZE from 8 → 4 → 2.

### Section 6 — Advanced Settings (Leave Alone Unless Testing)

In the `ModelConfig` dataclass, these are tuneable but defaults are good:

| Setting | Default | When to change |
|---------|---------|----------------|
| `spr_content_ratio` | 0.85 | Ratio sweep experiment (Section 7.4 of paper) |
| `spr_temporal_ratio` | 0.06 | Ratio sweep experiment |
| `spr_state_ratio` | 0.06 | Ratio sweep experiment |
| `spr_isolated_norm` | False | Set True if probes show too much cross-subspace leakage |
| `d_model` | 768 | Reduce to 512 or 384 for faster experiments |
| `reasoning_blocks` | 6 | Reduce to 4 for faster experiments |
| `max_recurrence` | 4 | Keep at 4 (curriculum controls actual R used) |

---

## Cell-by-Cell: What Each Section Does

| Section | Cells | What it does | Action |
|---------|-------|-------------|--------|
| 1. Setup | 1 | Installs packages, mounts Drive | Run once |
| 2. Environment | 1 | Detects GPU, enables TF32 | Run once |
| 3. BitLinear | 1 | Defines ternary quantization | Run once |
| 4. Architecture | 1 | RMSNorm, Attention, FFN, TransformerBlock | Run once |
| 5. Reasoning Cores | 3 | Encoder/Decoder + Baseline core + SPR core | Run once |
| **6. Config** | **3** | **ModelConfig + Model + Instantiation** | **CHANGE `use_spr` HERE** |
| 7. Training | 1 | Step count, batch size, curriculum | Adjust TOTAL_STEPS here |
| 8. Data | 1 | FineWeb-Edu streaming tokenizer | Run once |
| 9. Optimizer | 1 | AdamW + cosine schedule + resume logic | Run once |
| 10. Probing | 1 | Defines probe function (doesn't run it) | Run once |
| 11. Eval | 1 | Defines eval + checkpoint functions | Run once |
| **12. Training Loop** | **1** | **THE MAIN TRAINING — takes hours** | **Run and wait** |
| 13. Save | 1 | Exports final checkpoint + ternary weights | Run after training |
| 14. Final Eval | 1 | Tests R=1,2,3,4 perplexity | Run after training |
| 15. Visualization | 2 | 6-panel training curves + probes | Run after training |
| 16. Summary | 1 | Probe results table + interpretation | Run after training |

**Short version**: Run all cells 1-12 sequentially. Wait for 12 to finish. Then run 13-16.

---

## The Two-Run Comparison Protocol

### Run 1 — SPR (the experimental condition)

1. Open notebook fresh
2. Leave `use_spr=True` (default)
3. Optionally set `TOTAL_STEPS = 100_000` in Section 7
4. Run all cells top to bottom
5. Training outputs go to `Drive/recurrent_bitnet_v2_spr/`
6. After training completes, run Sections 13-16
7. Note the probe table from Section 16

### Run 2 — Baseline (the control)

1. **Restart runtime** (Runtime → Restart runtime in Colab)
2. Change `use_spr=False` in Section 6
3. Keep everything else identical (same TOTAL_STEPS, same BATCH_SIZE)
4. Run all cells top to bottom again
5. Training outputs go to `Drive/recurrent_bitnet_v2_baseline/`
6. After training completes, run Sections 13-16
7. Compare probe table with Run 1

### What "Success" Looks Like

The probe table has columns like `I→Cont`, `I→Temp`, `I→State`, `T→Cont`, etc.

**Strong separation (hypothesis confirmed)**:
```
         I→Cont  I→Temp  I→State  T→Cont  T→Temp  T→State
SPR:      0.30    0.85    0.28     0.15    0.03    0.04     ← big gaps
Baseline: 0.65    0.70    0.63     0.12    0.10    0.09     ← everything similar
```

Key metrics to compare:
- `I→Temp` should be MUCH higher than `I→Cont` and `I→State` for SPR
- `T→Cont` should be MUCH higher than `T→Temp` and `T→State` for SPR
- For baseline, all three columns in each group should be similar (no separation)
- The GAP between `I→Temp` and `I→Cont` is the main result

---

## Running Locally (RTX 3060 12GB)

The notebook is designed for Colab but works locally with minor changes:

1. Comment out the Colab-specific cells:
   - `drive.mount(...)` in Section 1
   - Change `DRIVE_BASE` to a local path like `/home/ty/checkpoints`

2. Set conservative batch size:
   ```python
   BATCH_SIZE = 4    # 12GB VRAM is tight at R=4
   ```

3. Run with Jupyter:
   ```bash
   cd /home/ty/Repositories/ai_workspace/recurrent_bitnet
   jupyter notebook notebooks/RecurrentBitNet_V2_SPR.ipynb
   ```

4. Or run the generator directly for a Python script approach:
   ```bash
   python3 notebooks/gen_spr_notebook.py  # Regenerates notebook
   ```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| OOM on R=4 | Reduce BATCH_SIZE to 4 or 2 |
| OOM on R=2 | Reduce d_model to 512 in ModelConfig |
| Training too slow | Reduce d_model to 384, reasoning_blocks to 4 |
| No probe data | Probes only run when R > 1. Check TOTAL_STEPS > first R=2 curriculum step |
| Probe numbers all similar | Expected for baseline. For SPR, check maturity/state gates are opening |
| Loss not decreasing | Check LR (PEAK_LR=2e-3 default). Try 1e-3 if unstable |
| Drive full | Checkpoints accumulate — old local ones auto-delete (keeps 3) |

---

## Extended Evaluations (If You Have Resources)

These go beyond the basic two-run comparison. Each is independent.

### Evaluation 1: Ratio Sweep

Test whether the biological 85/6/6/3 ratio is actually optimal.

```python
# In Section 6, try these configs (one per run):
configs = [
    ModelConfig(spr_content_ratio=0.90, spr_temporal_ratio=0.04, spr_state_ratio=0.04),  # minimal context
    ModelConfig(spr_content_ratio=0.85, spr_temporal_ratio=0.06, spr_state_ratio=0.06),  # default (biological)
    ModelConfig(spr_content_ratio=0.75, spr_temporal_ratio=0.10, spr_state_ratio=0.10),  # generous context
    ModelConfig(spr_content_ratio=0.60, spr_temporal_ratio=0.15, spr_state_ratio=0.15),  # Kerce-Fox-like
]
```

Run each for the same number of steps. Compare final loss + probe separation.
5 runs × your chosen step count.

### Evaluation 2: Isolated Norm Ablation

Tests whether cross-subspace leakage comes from RMSNorm coupling or attention.

```python
# Run 1: Standard RMSNorm (default)
config = ModelConfig(spr_isolated_norm=False)

# Run 2: Subspace-isolated RMSNorm
config = ModelConfig(spr_isolated_norm=True)
```

If isolated norm shows LESS cross-subspace probe leakage, the leakage is partly
from normalization coupling (fixable). If similar, it's from attention (desirable).

### Evaluation 3: Scale Test

Tests whether SPR helps more at smaller scales (our prediction).

```python
# Small model (faster, should show bigger SPR advantage):
config = ModelConfig(d_model=384, n_heads=6, d_ff=1536, reasoning_blocks=4)

# Medium model (default):
config = ModelConfig()  # d_model=768

# If you have A100 access:
config = ModelConfig(d_model=1024, n_heads=16, d_ff=4096, reasoning_blocks=8)
```

Run each at both use_spr=True and use_spr=False. Compare the SPR advantage
(probe separation gap + loss difference) across scales. We predict the small
model shows the largest gap.

### Evaluation 4: Curriculum Sensitivity (Machens Prediction)

Tests whether temporal context adapts faster than state context when
the reasoning depth changes mid-training.

1. Train normally for 100K steps (R goes 1→2→3)
2. At step 100K, **jump R directly to 4** (skip the gradual curriculum)
3. Watch the probes at steps 100K, 110K, 120K, 130K

**Machens prediction**: `I→Temporal` should recover within 10-20K steps
(temporal = external drive, fast adaptation). `I→State` should recover
slowly over 30-50K+ steps (state = recurrent dynamics, needs retraining).

---

## Quick-Start Cheat Sheet

```
FASTEST USEFUL RUN (Colab T4, ~3 hours):
  TOTAL_STEPS = 50_000
  CURRICULUM = [(0,1), (15_000,2), (35_000,3)]
  → Get 2-3 probe snapshots, see if separation emerges

RECOMMENDED FIRST RUN (Colab T4/A100, ~6 hours):
  TOTAL_STEPS = 100_000
  CURRICULUM = [(0,1), (25_000,2), (60_000,3), (85_000,4)]
  → Full curriculum, ~5 probe snapshots, gate trajectories visible

FULL EXPERIMENT (A100, ~10 hours per run × 2 runs):
  TOTAL_STEPS = 200_000 (default curriculum)
  → Solid evidence for/against the hypothesis
```
