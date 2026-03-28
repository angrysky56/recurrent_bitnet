# Orthogonal Streams: Content-Context Separation as an Architectural Prior for Language Models

**Draft — March 2026**

*Ty (angrysky56)*

---

## Abstract

We synthesize three independent lines of evidence — from human neuroscience, transformer interpretability, and production hybrid-attention architectures — into a unified architectural principle: **content and context representations in language models should be maintained in approximately orthogonal subspaces**, interacting through sparse co-activation rather than dense entanglement. Single-neuron recordings from the human medial temporal lobe (Bausch et al., Nature 2026) reveal that ~88% of content neurons and ~64% of context neurons are invariant to the other dimension, with only ~2.3% encoding specific conjunctions. Independently, the Dual-Stream Transformer (Kerce & Fox, March 2026) demonstrates that architecturally separating token and context streams incurs only 2.5% performance loss while preventing representational dissolution observed by layer 3 in standard transformers. We observe that Qwen3.5's hybrid Gated DeltaNet / full-attention architecture (Qwen Team, February 2026) already implements a partial version of this separation, with recurrent state layers accumulating context and full-attention layers performing content retrieval. We propose Subspace-Partitioned Reasoning (SPR), a concrete implementation for recurrent transformer architectures that allocates ~85% of hidden dimensions to context-invariant content, ~12% to content-invariant context, and ~3% to learned conjunctive binding — matching the biological ratios. We articulate testable predictions and argue this principle explains several known LLM failure modes including instruction following degradation, lost-in-the-middle, and reasoning chain corruption.

**Keywords:** content-context separation, orthogonal representations, dual-stream architecture, subspace partitioning, recurrent reasoning, hybrid attention, medial temporal lobe, Gated DeltaNet

---

## 1. Introduction

Standard transformer language models (Vaswani et al., 2017) route all computation through a single residual stream. Every component — token embeddings, attention heads, feed-forward networks — reads from and writes to the same vector space (Elhage et al., 2021). This architectural choice has a critical consequence: information about *what tokens mean* (content) becomes irrecoverably entangled with information about *what role those tokens play in the current computation* (context) as representations propagate through layers.

This entanglement may not be merely an aesthetic concern. We argue it is a fundamental architectural bottleneck that explains several well-documented failure modes of modern LLMs, and that the solution — maintaining content and context in approximately orthogonal subspaces — has now been independently validated by three distinct research programs within a three-month window:

1. **Human neuroscience** (January 2026): Single-neuron recordings from the medial temporal lobe reveal that content and context are encoded by largely separate neuronal populations with orthogonal representations and sparse conjunctive binding (Bausch et al., 2026).

2. **Transformer architecture** (March 2026): The Dual-Stream Transformer explicitly decomposes the residual stream into token (content) and context components, demonstrating that architectural separation incurs minimal performance cost while preventing representational collapse (Kerce & Fox, 2026a, 2026b).

3. **Production hybrid models** (February 2026): Qwen3.5's hybrid Gated DeltaNet / full-attention architecture implements a form of content-context separation at the layer type level, and achieves state-of-the-art results while being dramatically more inference-efficient (Qwen Team, 2026).

We synthesize these threads into a concrete architectural proposal — Subspace-Partitioned Reasoning (SPR) — with specific dimensional allocations derived from the biological ratios, and articulate testable predictions that distinguish SPR from both standard residual streams and existing dual-stream approaches.

---

## 2. Biological Evidence: Orthogonal Content-Context in the Human MTL

### 2.1 The Experimental Paradigm

Bausch et al. (2026) recorded 3,109 neurons from 16 neurosurgical patients implanted with depth electrodes for invasive seizure monitoring. Patients performed a context-dependent picture-comparison task: pairs of pictures were presented sequentially following one of five context questions ("Bigger?", "Last seen in real life?", "Older? or More expensive?", "Like better?", "Brighter?"). The task required participants to remember item content (the pictures) bound to a specific context (the comparison rule), then make a contextually appropriate judgment.

This paradigm operationalizes *interactive context* — the type of context that changes how content is processed, analogous to how a system prompt changes how a language model processes user input, or how a reasoning step changes how subsequent tokens are interpreted.

### 2.2 Key Findings: Population-Level Orthogonality

From the full population, Bausch et al. identified two functionally distinct groups:

- **597 stimulus-modulated (content) neurons**: These neurons encoded picture identity — firing selectively for specific pictures regardless of which question was asked. Critically, **88% were invariant to context**. A neuron that fired for "biscuit" fired equally for "biscuit" under "Bigger?", "Older?", "Like better?", etc.

- **200 context-modulated neurons**: These encoded the active comparison rule. **63.5% were invariant to stimulus identity**. A neuron encoding "Older?" maintained its representation regardless of which pictures were shown.

Only **~2.3% of hippocampal neurons** encoded specific conjunctions of content and context (e.g., "biscuit" + "Brighter?"). This stands in sharp contrast to rodent hippocampal representations, which are predominantly conjunctive (context-dependent). The human system *dramatically* favors generalization over specificity.

The regional distribution is also informative:

| Region | Content neurons | Context neurons | Conjunctive |
|---|---|---|---|
| Amygdala | Present | 2.95% | Rare |
| Parahippocampal cortex | Present | 7.68% | Low |
| Entorhinal cortex | Present | 5.68% | Low |
| Hippocampus | Present | 9.42% | 2.29% (highest) |

Content and context representations converge in the hippocampus via parallel processing streams — content through the lateral entorhinal / perirhinal cortex, context through the parahippocampal / medial entorhinal cortex. This convergence does not produce entanglement; it produces co-activation of orthogonal codes.

### 2.3 The Interaction Mechanism: Co-Activation, Not Modulation

How do content and context combine if they remain orthogonal? Bausch et al. identify three mechanisms:

1. **Temporal co-firing**: During picture presentations, both content and context neurons fire simultaneously. The combined pattern — read as a joint population vector — carries the bound representation without either population changing its coding scheme.

2. **Spike-timing-dependent plasticity (STDP)**: Entorhinal content neuron firing predicted hippocampal context neuron firing approximately 40ms later — but *only after experimental pairing*, not before. This asymmetric cross-correlation emerged during the experiment and persisted afterward, consistent with learned associations being stored via synaptic modification rather than representational blending.

3. **Pre-activation gating**: Context neurons that had been pre-activated by their preferred question showed increased excitability when stimuli arrived. This is a gating mechanism — context selectively *amplifies* the response to relevant content rather than modulating the content representation itself.

**The critical distinction**: FiLM-style modulation (the approach proposed for RecurrentBitNet V2) applies `x_content = gamma(context) * x_content + beta(context)`, forcing every dimension of content to be transformed by context. The biological system does the opposite — content representations are *protected* from context, with interaction occurring through sparse co-activation in a shared readout space.

### 2.4 Evolutionary Rationale: Generalization Over Specificity

The biological design makes deep evolutionary sense. A system that encodes every content×context combination requires O(n × m) neurons for n items and m contexts. A system with orthogonal content and context codes requires only O(n + m) neurons and can represent n × m combinations through co-activation. For a human memory system handling thousands of concepts across hundreds of contexts, the combinatorial savings are enormous.

Furthermore, orthogonal coding enables *zero-shot generalization*: a content representation learned in one context automatically transfers to all other contexts. This is precisely the generalization capability that LLMs achieve imperfectly through massive training data, and that in-context learning attempts to achieve at inference time.

---

## 3. Engineering Evidence: Dual-Stream Transformers

### 3.1 The Problem: Representational Dissolution

Elhage et al. (2021) characterize the transformer residual stream as a "communication channel" through which all components interact by reading from and writing to shared subspaces. While this enables rich inter-component communication, it also means that by the middle layers of a standard transformer, it becomes impossible to cleanly decompose the residual stream into "what this token means" versus "what role it plays in the current computation."

Kerce & Fox (2026b) quantify this dissolution directly. In their Late Fusion Architecture experiments, they introduce the Token-Position Dependence Score (PDS) to measure how well symbolic token identity can be decoded from representations at each layer. Standard transformers show PDS_max = 0.058 — meaning token identity is effectively unrecoverable by mid-network. Their Late Fusion Architecture, which delays stream integration, achieves PDS_max = 0.276 — nearly 5× higher symbolic preservation.

The practical implication: **by layer 3 of a 6-layer standard transformer, you can no longer tell what a token originally was.** The content has been dissolved into context.

### 3.2 The Dual-Stream Decomposition

Kerce & Fox (2026a) propose decomposing the residual stream into additive components: **x = x_t + x_e**, where:

- **x_t (token stream)** carries information derived from discrete token identities, updated exclusively by attention
- **x_e (context stream)** accumulates continuous contextual transformations, updated exclusively by feed-forward networks

This is not merely a notational convenience. It is an architectural constraint — attention cannot write to the context stream, and FFNs cannot write to the token stream. Information exchange between streams occurs only through the combined representation at readout points.

### 3.3 Empirical Results: Bounded Cost of Separation

At 29M parameters on language modeling tasks:

| Configuration | Val loss vs. dense baseline | Key property |
|---|---|---|
| Dense (standard) | 0% (baseline) | Fully entangled |
| Kronecker mixing | +2.5% | Scalar cross-head communication |
| Fully independent | +8.0% | Maximum interpretability |

The Kronecker mixing configuration is particularly relevant — it permits only scalar communication between heads while preserving within-head structure. This corresponds to the biological finding that content and context neurons interact through simple co-activation statistics (correlation, timing) rather than complex high-dimensional modulation.

### 3.4 Ablation: Asymmetric Importance

Stream ablation reveals an asymmetry that mirrors the biological findings:

- Removing the **token stream** (content) causes **36% degradation** — catastrophic
- Removing the **context stream** causes only **9.5% degradation** — moderate

This matches the Bausch et al. finding that content neurons (597) outnumber context neurons (200) by 3:1, and that the conjunctive binding in the hippocampus primarily enriches content processing rather than creating new context representations.

---

## 4. The Qwen3.5 Natural Experiment

### 4.1 Architecture Overview

Qwen3.5 (Qwen Team, February 2026) employs a hybrid attention design alternating between two layer types in a 3:1 ratio:

- **Gated DeltaNet layers** (75% of layers): Linear attention with a recurrent state matrix S_t = α·S_{t-1} + β·v·kᵀ. These layers maintain a fixed-size hidden state that accumulates context over the sequence. The delta rule provides error-correcting memory updates, and exponential gating controls memory decay.

- **Full attention layers** (25% of layers): Standard softmax attention with global token-to-token retrieval. These layers perform precise content matching across the full sequence.

The layout is: `[DeltaNet, DeltaNet, DeltaNet, FullAttn, DeltaNet, DeltaNet, DeltaNet, FullAttn, ...]`

### 4.2 Implicit Content-Context Separation

We observe that this hybrid architecture implements a form of content-context separation at the *layer type* level:

**DeltaNet layers as context accumulators**: The recurrent state S_t is a compressed summary of everything seen so far. It does not preserve individual token identities — it accumulates distributional context. This is functionally analogous to the context neuron population: maintaining "what question are we answering" without encoding specific token content.

**Full attention layers as content retrievers**: Standard attention computes exact similarity between query and key vectors, enabling precise content matching. A full attention layer at every 4th position acts as a "checkpoint" where the model can perform exact retrieval against the full sequence — analogous to the content neuron population performing stimulus-invariant encoding.

### 4.3 The 3:1 Ratio

The biological system shows approximately 75% content-processing capacity (597 stimulus neurons) and 25% context-processing capacity (200 context neurons). Qwen3.5's 3:1 DeltaNet-to-attention ratio inverts this (75% context accumulation, 25% content retrieval), but the structural principle is identical: dedicate most resources to one modality and use the other sparingly at regular intervals.

That Qwen3.5 achieves state-of-the-art results with this architecture — matching or exceeding GPT-5.2 and Claude Opus 4.5 on most benchmarks while using dramatically less compute — is strong evidence that content-context separation is not merely biologically elegant but computationally efficient.

---

## 5. Proposed Architecture: Subspace-Partitioned Reasoning (SPR)

### 5.1 Motivation

The three evidence streams converge on a single principle: content and context should occupy mostly disjoint subspaces of the representation, with interaction mediated by sparse binding mechanisms. We formalize this as Subspace-Partitioned Reasoning (SPR).

SPR is particularly relevant for recurrent/iterative reasoning architectures (e.g., Universal Transformers, PonderNet, Gated DeltaNet hybrids, and our RecurrentBitNet) where an explicit new source of context exists: *which iteration of the reasoning loop is being executed*. This iteration context is pure metadata — it tells the model "you are on reasoning step 3 of 4" but carries no information about the token content being processed.

### 5.2 Dimensional Allocation

For a model with hidden dimension d_model, SPR partitions the representation into three contiguous subspaces:

```
d_model = d_content + d_context + d_conjunctive

Where (following biological ratios):
  d_content      ≈ 0.85 × d_model    # Context-invariant content
  d_context      ≈ 0.12 × d_model    # Content-invariant context
  d_conjunctive  ≈ 0.03 × d_model    # Learned conjunctive binding
```

For a 768-dimensional model: d_content = 652, d_context = 92, d_conjunctive = 24.

### 5.3 Content Subspace (d_content)

The content subspace carries token semantics and is **never directly modified by iteration context**. When the model enters reasoning iteration r:

```python
x_content = x[:, :, :d_content]  # Unchanged by iteration signal
```

Content dimensions are updated by transformer blocks normally (attention + FFN), but the iteration embedding has zero projection into this subspace. This preserves the "88% invariance" property from the biology.

### 5.4 Context Subspace (d_context)

The context subspace encodes computational state — what reasoning step we're on, what the model's current "goal" is. The iteration embedding writes exclusively here:

```python
x_context = x[:, :, d_content:d_content+d_context]
x_context = x_context + iteration_embedding[r]  # Only modifies context dims
```

Context dimensions are also updated by transformer blocks, allowing the model to build complex context representations. But critically, the *injection* of external context (iteration signals) is confined to this subspace.

### 5.5 Conjunctive Subspace (d_conjunctive)

The conjunctive subspace is where content and context interact to form bound representations. This is analogous to the ~2.3% of hippocampal neurons that encode specific content×context combinations.

```python
x_bind = x[:, :, d_content+d_context:]
# A small learned network reads from both streams and writes here
binding_input = torch.cat([
    x_content.mean(dim=-1, keepdim=True).expand(..., d_conjunctive),
    x_context.mean(dim=-1, keepdim=True).expand(..., d_conjunctive)
], dim=-1)
x_bind = x_bind + binding_network(binding_input)
```

The binding network is deliberately small — it should capture coarse content×context interactions (e.g., "mathematical content + iteration 3 → engage precise computation") without creating the fine-grained entanglement that standard residual streams produce.

### 5.6 How Interaction Happens: Attention as Co-Activation

A critical design property: **standard multi-head attention naturally implements the co-activation mechanism** without architectural modification. When computing Q, K, V from the full d_model representation, attention heads can learn to:

- Read Q and K from content dimensions → content-content attention (majority of heads)
- Read Q and K from context dimensions → context-context attention (some heads)
- Read Q from content, K from context (or vice versa) → cross-subspace attention, implementing the ~40ms STDP-like binding Bausch et al. observed

The subspace partition does not prevent cross-stream attention — it prevents *additive contamination* through the residual connection. Content dimensions never have iteration signals added to them, but attention heads can freely attend across all dimensions. This is precisely the biological pattern: content neurons don't change their coding scheme in response to context, but they participate in joint population activity with context neurons.

### 5.7 Comparison with Alternatives

| Approach | Content-context separation | Interaction mechanism | Cost |
|---|---|---|---|
| Standard residual stream | None — full entanglement | Dense mixing | 0 extra params |
| FiLM conditioning | None — every dim modulated | Multiplicative modulation | 2 × d_model² |
| Kerce & Fox dual-stream | Full — separate update paths | Combined readout | 2.5% perf loss |
| SPR (proposed) | Partial — subspace partition | Attention across subspaces | ~0 extra params* |

*SPR adds only the small conjunctive binding network (~74K params for d_model=768), which is <0.1% of a typical model.

The key advantage of SPR over the Kerce & Fox approach is that SPR does not require separate update paths for attention and FFN. Standard transformer blocks operate over the full d_model as usual. The separation is enforced only at the *injection point* of external context (iteration embeddings), making SPR a minimal architectural modification that can be applied to any existing transformer.

---

## 6. Explanatory Power: Known LLM Failure Modes

If the content-context entanglement hypothesis is correct, it should explain observed failure modes in existing LLMs. We identify four:

### 6.1 Instruction Following Degradation

**Observation**: As conversations grow longer, LLMs progressively lose adherence to system prompts and initial instructions.

**Entanglement explanation**: The system prompt encodes context (behavioral rules). In a standard residual stream, this context is represented in the same dimensions as token content. As more tokens flow through the network, content representations progressively overwrite the context dimensions, diluting the instruction signal. By mid-conversation, the system prompt's representational influence has been dissolved into the content stream.

**SPR prediction**: A subspace-partitioned model should maintain instruction adherence over longer conversations, because context dimensions are architecturally protected from content overwriting.

### 6.2 Lost in the Middle

**Observation**: LLMs perform well on information at the beginning and end of long contexts but poorly on information in the middle (Liu et al., 2023).

**Entanglement explanation**: Positional context (where am I in the sequence?) is a form of context that entangles with content. Tokens at the beginning benefit from primacy effects in attention; tokens at the end benefit from recency. Tokens in the middle have their content representations maximally contaminated by ambiguous positional context — they are neither clearly "early" nor clearly "late."

**SPR prediction**: Models with separated content and positional-context subspaces should show flatter retrieval curves across sequence positions, as content representations remain invariant to positional context.

### 6.3 In-Context Learning Brittleness

**Observation**: Few-shot prompting is sensitive to example ordering, formatting, and even the specific examples chosen, in ways that seem disproportionate to the semantic content of the perturbation.

**Entanglement explanation**: Few-shot examples serve as context (demonstrating the desired input-output mapping), while the test input is content (the actual query to process). In an entangled residual stream, perturbations to examples modify the *same dimensions* that carry test-input content. A format change in the examples can destructively interfere with content processing even when it carries no semantic information.

**SPR prediction**: A subspace-partitioned model should exhibit more robust in-context learning, as context (the demonstrated pattern) and content (the test input) occupy different dimensions. Formatting perturbations to examples should affect only context dimensions.

### 6.4 Reasoning Chain Corruption

**Observation**: In chain-of-thought reasoning, errors in early steps propagate and amplify through subsequent steps, even when the model "knows" the correct intermediate result.

**Entanglement explanation**: Each intermediate reasoning step serves dual roles — it is both content (a computed result) and context (a premise for the next step). In an entangled stream, the *error signal* from a wrong intermediate step contaminates the same dimensions used to process subsequent content, making recovery impossible even if the model has the parametric knowledge to compute the correct answer.

**SPR prediction**: In a recurrent reasoning architecture with SPR, the content representation of the current step should be protected from errors in the context accumulated from previous iterations. The conjunctive subspace may carry forward error signals, but the content dimensions — comprising 85% of the representation — remain clean.

---

## 7. Testable Predictions and Proposed Experiments

### 7.1 Direct Test: Subspace Probing

Train two matched models — a standard RecurrentBitNet V2 and an SPR variant — on identical data. At each layer and iteration, fit linear probes to decode:

(a) Token identity from content dimensions only
(b) Iteration number from context dimensions only
(c) Token identity from context dimensions (should be at chance for SPR)
(d) Iteration number from content dimensions (should be at chance for SPR)

The cross-subspace probes (c, d) constitute the key test. In a standard model, both should be above chance (entanglement). In SPR, both should be at or near chance (orthogonality preserved).

### 7.2 Attention Head Specialization

Analyze the Q/K weight matrices of trained SPR models. Predict that attention heads will specialize into three categories:

- **Content-content heads** (~70-80%): Q and K projections draw primarily from content dimensions
- **Context-context heads** (~10-15%): Q and K projections draw primarily from context dimensions
- **Cross-subspace heads** (~5-10%): Q draws from content, K from context (or vice versa)

This mirrors the biological finding of separate populations with sparse conjunctive binding, and the Kerce & Fox finding that architecturally constrained heads develop stronger functional specialization (orthogonality increasing from 0.42 to 0.85 with independent mixing).

### 7.3 Robustness Ablations

Test the SPR model on tasks designed to probe the four failure modes:

- **Instruction adherence**: Measure system prompt compliance at conversation lengths 1K, 4K, 16K, 64K tokens
- **Positional retrieval**: Needle-in-haystack performance as a function of needle position
- **ICL robustness**: Few-shot accuracy variance under example permutation and format perturbation
- **Reasoning recovery**: On multi-step math problems, inject a controlled error at step k and measure recovery rate at step k+1

### 7.4 Ratio Sweep

The biological ratios (85/12/3) are a starting hypothesis. Sweep content/context/conjunctive ratios:

- (90/8/2), (85/12/3), (75/20/5), (60/30/10), (50/50/0 — Kerce & Fox style)

Compare perplexity, downstream task performance, and the four robustness metrics. Predict that ratios near the biological optimum will show the best tradeoff between raw language modeling performance and robustness.

---

## 8. Discussion

### 8.1 Relationship to Existing Work

**Mechanistic interpretability**: Anthropic's residual stream analysis (Elhage et al., 2021) established that components communicate through shared subspaces. SPR takes this a step further by arguing that *some* subspaces should be architecturally protected from certain types of writing. This is complementary to superposition-based interpretability — if content and context occupy disjoint subspaces, the superposition problem within each subspace is simpler.

**Mixture of Experts**: MoE architectures partition *parameters* but not *representations*. All experts write to the same residual dimensions. SPR partitions *representations* and is orthogonal to MoE — one could have subspace-partitioned representations within each expert.

**Differential attention (Ye et al., 2024)**: Differential attention cancels common-mode signals between attention heads. This is a form of noise reduction within the entangled stream, not a separation of streams. SPR prevents the entanglement from occurring rather than correcting for it post-hoc.

**Residual stream duality (March 2026)**: The very recent observation that the residual stream can be viewed through a two-axis framework (sequence position vs. layer depth) is complementary to SPR. SPR partitions the *feature dimension*, while residual stream duality partitions the *routing axes*.

### 8.2 Limitations

**No empirical validation yet**: This paper presents a synthesis and architectural proposal, not experimental results. The predictions in Section 7 are testable but untested. We prioritize making the argument publicly available given the rapid pace of convergent discoveries.

**Biological analogy limits**: The human MTL processes memory and comparison tasks, not autoregressive language generation. The content-context separation observed by Bausch et al. may be specific to declarative memory rather than a universal computational principle. However, the independent engineering validation from Kerce & Fox suggests the principle generalizes.

**Optimal ratios may vary**: The 85/12/3 allocation is derived from human neuron counts, which evolved under very different computational constraints. The optimal ratio for transformer language models may differ substantially. The ratio sweep experiment (Section 7.4) is designed to address this.

### 8.3 Connection to the Drosophila AND Gate

The Drosophila memory consolidation work (Francés et al., Nature 2026) provides a complementary perspective on content-context interaction. In flies, long-term memory consolidation requires a strict AND gate: (1) spaced training must restore fructose sensor sensitivity (a context signal indicating "learning has occurred"), AND (2) post-training sugar ingestion must occur (a content signal indicating "energy is available").

This AND gate operates on *independent* signals — the training-induced sensor reset and the metabolic sugar detection are computed by separate neural circuits that converge only at the consolidation decision point. This is architecturally identical to SPR's conjunctive subspace: two mostly-independent streams (content and context) producing a joint decision through sparse convergence.

The implication for recurrent reasoning: the decision to *consolidate* a reasoning step (commit it to the context state for future iterations) should depend on an AND gate between content quality (did this step produce useful intermediate results?) and context appropriateness (is this the right point in the reasoning process to consolidate?). Implementing this as a gating mechanism in the conjunctive subspace — rather than as the blunt loss-improvement check proposed in the original "metabolic AND gate" — would more faithfully capture the biological principle.

### 8.4 Connection to RYS Neuroanatomy

The RYS (Repeat Your Self) discovery (Ng, 2024/2026) that duplicating middle-layer "reasoning circuits" of 5-7 layers improves benchmark performance — with zero weight changes — takes on new significance under the content-context separation hypothesis.

If early layers primarily establish content representations (encoding) and late layers primarily map back to output tokens (decoding), then the middle layers are the "reasoning cortex" where content is iteratively refined under contextual constraints. RYS-style duplication succeeds because the reasoning circuit is *re-entrant* — applying it twice is analogous to running an additional iteration of recurrent reasoning.

Crucially, RYS works only when complete circuits are duplicated (single-layer duplication helps nothing). Under SPR, this makes sense: a complete circuit encompasses enough layers for attention heads to perform content-content processing, context-context processing, AND cross-subspace binding. Duplicating a single layer may disrupt the binding cycle without completing it.

This connects to the Qwen3.5 architecture: the natural circuit boundaries in the 3:1 DeltaNet/attention layout are the 4-layer blocks (3 DeltaNet + 1 attention). RYS scanning of Qwen3.5 should reveal that these 4-layer blocks are the indivisible reasoning units, and that duplicating complete blocks improves performance while duplicating partial blocks degrades it.

### 8.5 Broader Implications for Transformer Design

If the content-context separation principle is validated, it suggests several concrete design directions:

**System prompt engineering**: Rather than placing system prompts in the same token stream as user input, architectures could inject system instructions into dedicated context dimensions — never competing with content for representational capacity.

**Positional encoding redesign**: Current positional encodings (RoPE, ALiBi) inject positional information into the same dimensions as token content. SPR suggests positional information should be confined to context dimensions, potentially explaining why longer-context models require increasingly sophisticated positional encoding schemes — they are fighting the entanglement problem with ever-more-complex injection patterns.

**Retrieval-augmented generation**: RAG systems retrieve content but inject it into an entangled stream. SPR suggests retrieved content should be injected into content dimensions while the *retrieval context* (relevance scores, source metadata, recency) should go to context dimensions.

---

## 9. Conclusion

We have identified a convergence across three independent research programs — human neuroscience, transformer architecture, and production language model engineering — pointing toward a single principle: **content and context representations should be maintained in approximately orthogonal subspaces**.

The human medial temporal lobe implements this through separate neuronal populations with sparse conjunctive binding. The Dual-Stream Transformer implements it through architecturally separated update paths. Qwen3.5 implements it (partially) through functionally distinct layer types. All three achieve strong performance while maintaining representational clarity.

We propose Subspace-Partitioned Reasoning as a minimal architectural modification that captures this principle within standard transformer blocks, with dimensional allocations inspired by the biological ratios. The proposal generates concrete, testable predictions about probing accuracy, attention head specialization, and robustness to known failure modes.

The timing of this convergence is not accidental. As language models scale and are deployed in increasingly complex reasoning tasks, the costs of content-context entanglement become more visible — instruction drift, context confusion, reasoning chain corruption. The biological brain solved this problem through architectural separation hundreds of millions of years ago. The engineering community is now independently rediscovering the same solution.

We release this synthesis as a preprint to accelerate community validation and refinement of the principle, and plan to follow with experimental results from RecurrentBitNet V2 models implementing SPR.

---

## References

### Primary Sources (The Three Convergent Threads)

**[Bausch et al., 2026]** Bausch, M., Niediek, J., Reber, T.P., Mackay, S., Boström, J., Elger, C.E. & Mormann, F. (2026). Distinct neuronal populations in the human brain combine content and context. *Nature*, 650, 690–700. https://doi.org/10.1038/s41586-025-09910-2

**[Kerce & Fox, 2026a]** Kerce, J.C. & Fox, A. (2026). The Dual-Stream Transformer: Channelized Architecture for Interpretable Language Modeling. *arXiv:2603.07461*.

**[Kerce & Fox, 2026b]** Kerce, J.C. & Fox, A. (2026). Interpretable-by-Design Transformers via Architectural Stream Independence. *arXiv:2603.07482*.

**[Qwen Team, 2026]** Qwen Team. (2026). Qwen3.5 Technical Report. Alibaba Cloud.

### Supporting Sources

**[Elhage et al., 2021]** Elhage, N., Nanda, N., Olsson, C., et al. (2021). A Mathematical Framework for Transformer Circuits. *Transformer Circuits Thread*. https://transformer-circuits.pub/2021/framework/

**[Francés et al., 2026]** Francés, R., Comyn, T., Desnous, C., Bettoni, F., Pavlowsky, A., Preat, T. & Plaçais, P.-Y. (2026). Aversive learning hijacks a brain sugar sensor to consolidate memory. *Nature*. https://doi.org/10.1038/s41586-026-10306-z

**[Liu et al., 2023]** Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F. & Liang, P. (2023). Lost in the Middle: How Language Models Use Long Contexts. *arXiv:2307.03172*.

**[Ng, 2026]** Ng, D.N. (2026). LLM Neuroanatomy: How I Topped the LLM Leaderboard Without Changing a Single Weight. https://dnhkng.github.io/posts/rys/

**[Perez et al., 2018]** Perez, E., Strub, F., de Vries, H., Dumoulin, V. & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI 2018*.

**[Raschka, 2025]** Raschka, S. (2025). Beyond Standard LLMs. *Ahead of AI Newsletter*. https://magazine.sebastianraschka.com/p/beyond-standard-llms

**[Residual Stream Duality, 2026]** arXiv:2603.16039. Residual Stream Duality in Modern Transformer Architectures.

**[Vaswani et al., 2017]** Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.

**[Yang et al., 2024]** Yang, S., Wang, B., Shen, Y., Panda, R. & Kim, Y. (2024). Gated Delta Networks: Improving Mamba2 with Delta Rule. *arXiv:2412.06464*.

**[Ye et al., 2024]** Ye, T., Dong, L., Xia, Y., et al. (2024). Differential Transformer. *arXiv:2410.05258*.

---

## Appendix A: SPR Implementation Sketch (PyTorch)

```python
class SubspacePartitionedReasoningCore(nn.Module):
    """
    Reasoning core with Subspace-Partitioned Reasoning (SPR).
    
    Partitions d_model into three subspaces:
    - Content (~85%): Never modified by iteration embeddings
    - Context (~12%): Receives iteration embeddings
    - Conjunctive (~3%): Learned binding between content and context
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Subspace dimensions (derived from biological ratios)
        self.d_content = int(config.d_model * 0.85)
        self.d_context = int(config.d_model * 0.12)
        self.d_conjunctive = config.d_model - self.d_content - self.d_context
```

```python
        # Standard transformer blocks (operate on full d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.reasoning_blocks)
        ])
        
        # Iteration embeddings — ONLY d_context dimensions
        self.iteration_embeddings = nn.Parameter(
            torch.randn(config.max_recurrence, 1, 1, self.d_context) * 0.02
        )
        
        # Conjunctive binding network (small: ~2,304 params for d_model=768)
        self.binding_net = nn.Sequential(
            nn.Linear(self.d_content + self.d_context, self.d_conjunctive * 4),
            nn.GELU(),
            nn.Linear(self.d_conjunctive * 4, self.d_conjunctive),
        )
        # Initialize near-zero so binding is learned gradually
        nn.init.zeros_(self.binding_net[-1].weight)
        nn.init.zeros_(self.binding_net[-1].bias)
        
        self.halt_scorer = nn.Sequential(
            nn.Linear(config.d_model, 1), nn.Sigmoid()
        )
    
    def forward(self, x, mask=None, R=None, recurrence_dropout=0.0):
        if R is None:
            R = self.iteration_embeddings.size(0)
        
        iter_outputs = []
```

```python
        for r in range(R):
            if self.training and recurrence_dropout > 0 and r > 0:
                if torch.rand(1).item() < recurrence_dropout:
                    continue
            
            # === SUBSPACE PARTITIONING ===
            # Split representation into three subspaces
            x_content = x[:, :, :self.d_content]                              # 85%
            x_context = x[:, :, self.d_content:self.d_content+self.d_context] # 12%
            x_bind    = x[:, :, self.d_content+self.d_context:]               # 3%
            
            # 1. Inject iteration context ONLY into context subspace
            if r < self.iteration_embeddings.size(0):
                x_context = x_context + self.iteration_embeddings[r]
            
            # 2. Compute conjunctive binding from content + context
            binding_input = torch.cat([x_content, x_context], dim=-1)
            x_bind = x_bind + self.binding_net(binding_input)
            
            # 3. Reassemble full representation
            x = torch.cat([x_content, x_context, x_bind], dim=-1)
            
            # 4. Standard transformer processing (attention sees ALL dims)
            for block in self.blocks:
                x = block(x, mask)
            
            iter_outputs.append(x)
        
        return x, iter_outputs
```


**Implementation note**: After the transformer blocks process the full d_model representation, content and context dimensions may receive cross-subspace information through attention. This is *intentional* — it mirrors the biological co-activation mechanism. The key invariant is that the *injection* of external context (iteration embeddings) is confined to context dimensions. The transformer's own learned cross-subspace communication through attention heads is the analog of the entorhinal→hippocampal STDP associations.

The critical test of whether SPR is working is not whether content and context dimensions remain perfectly orthogonal after transformer processing — it is whether linear probes can decode iteration number from content dimensions. If they cannot (or can only weakly), the partition is doing its job even if attention heads create some cross-subspace flow.

---

## Appendix B: Probing Protocol

To validate SPR, train linear probes at each layer l and iteration r:

```python
# After extracting hidden states h at layer l, iteration r:
h_content = h[:, :, :d_content]
h_context = h[:, :, d_content:d_content+d_context]

# Probe 1: Can we decode token identity from content? (should be HIGH)
token_from_content = LinearProbe(d_content, vocab_size).fit(h_content, token_ids)

# Probe 2: Can we decode iteration from context? (should be HIGH)  
iter_from_context = LinearProbe(d_context, max_R).fit(h_context, iteration_labels)

# Probe 3: Can we decode token identity from context? (should be LOW for SPR)
token_from_context = LinearProbe(d_context, vocab_size).fit(h_context, token_ids)

# Probe 4: Can we decode iteration from content? (should be LOW for SPR)
iter_from_content = LinearProbe(d_content, max_R).fit(h_content, iteration_labels)
```

**Success criterion**: Probes 1 and 2 should achieve above-chance accuracy at all layers. Probes 3 and 4 should be at or near chance for SPR models. Standard (non-SPR) models will show above-chance accuracy on all four probes, confirming entanglement.

---

*This paper is a synthesis and architectural proposal. Experimental validation is in progress and will be reported separately. The authors welcome collaboration on the ratio sweep experiments (Section 7.4) from groups with access to larger compute budgets.*

*Code and architecture available at: https://github.com/angrysky56/recurrent_bitnet*
