# LLM Discovery: embedding_dim=4 Eliminates Catastrophic Failures

## The Discovery

After 60 iterations of systematic hyperparameter exploration, the LLM autonomously discovered that increasing `embedding_dim` from 2 to 4 eliminates catastrophic seed failures — the single most impactful change in the entire exploration.

**Before (emb_dim=2):** ~25% of seeds catastrophically fail (conn_R2 < 0.8), best robust mean = 0.910±0.020, CV=2.2%
**After (emb_dim=4):** 0 failures across 4 seeds, mean = 0.927±0.003, CV=0.30%, all seeds > 0.92

This finding is non-obvious and would be difficult to identify through manual experimentation, since embedding_dim is not typically considered a primary hyperparameter for connectivity recovery.

## Reasoning Chain

The LLM's reasoning followed a multi-step hypothesis-driven process, not a grid search:

### Step 1: Pattern Recognition (iter ~20)
The LLM observed that failing seeds consistently had low `cluster_accuracy` and computed a strong correlation (r≈0.97) between cluster_accuracy and connectivity_R2. Key observation:

> "Slot 0 near-miss (0.886) had lowest cluster_accuracy (0.753) — embedding quality may be the bottleneck"

### Step 2: Wrong Lever First (iter 21-24)
Hypothesis: "Increasing embedding LR (1.55e-3 → 2.5e-3) will improve embedding quality."
Result: **FALSIFIED** — mean dropped 0.910→0.867. Faster learning of a 2D embedding doesn't help if the space itself is too small.

### Step 3: Eliminating Competing Hypotheses (iter 49-60)
The LLM systematically tested W-level causes of catastrophic failures:
- W_L2 removal: no improvement
- w_init_mode=zeros: **worst failure ever** (0.649)

Critical conclusion: *"Catastrophic failures are NOT W-initialization-dependent. They are driven by data/training seed interactions."*

### Step 4: Architectural Insight (iter 61)
After exhausting all regularization and initialization approaches, the LLM formed:

> **Hypothesis**: "Increasing embedding_dim from 2 to 4 improves neuron type discrimination, reducing catastrophic failures and achieving mean connectivity_R2 > 0.90 with lower CV"
>
> **Rationale**: With 65 neuron types, the default 2D embedding is geometrically insufficient for reliable discrimination. Some seeds find degenerate 2D configurations where distinct types overlap, causing cascading failure in connectivity recovery.

### Step 5: Validation + Bounding (iter 61-68)
- emb_dim=4: **STRONGLY SUPPORTED** — first robust result in entire exploration
- emb_dim=6: **FALSIFIED** — diminishing returns, degraded robustness (mean 0.927→0.899)

This bounded the solution: 2D too small, 4D optimal, 6D overfits.

## Cluster Accuracy Analysis

Surprisingly, emb_dim=4 did **not** substantially improve cluster_accuracy itself:

| Config | cluster_accuracy (range) | conn_R2 (range) | Catastrophic failures |
|--------|--------------------------|-----------------|----------------------|
| emb_dim=2, good seeds | 0.79–0.83 | 0.89–0.94 | 0/3 |
| emb_dim=2, bad seeds | 0.70–0.75 | 0.65–0.80 | 1/1 |
| **emb_dim=4** | **0.79–0.82** | **0.925–0.931** | **0/4** |
| emb_dim=6 | 0.77–0.83 | 0.871–0.917 | 1/4 |

The mechanism was **not** about achieving higher cluster accuracy, but about **eliminating the degenerate low-accuracy configurations** (0.70-0.75) that trigger cascading failures. The floor was raised from 0.706 to 0.787. The 4D space prevents the embedding from collapsing into degenerate 2D configurations where distinct neuron types overlap.

## Proposal: Improving Cluster Accuracy via Embedding Dimension

Current `cluster_accuracy` saturates at ~0.82 regardless of embedding_dim (2, 4, or 6). This suggests the bottleneck has shifted from the embedding space geometry to the clustering algorithm or the training signal.

### Candidate Approaches

**1. Supervised contrastive loss on embeddings**
Add an auxiliary loss that explicitly pushes same-type neurons together and different-type neurons apart in embedding space. This directly optimizes for cluster separability rather than relying on it emerging as a byproduct of connectivity training.
- Implementation: `L_contrastive = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))` for same-type pairs (i,j)
- Requires: neuron type labels (already available as `type_list`)
- Risk: may compete with connectivity loss; needs careful weighting

**2. Embedding regularization via neuron type prior**
Add a soft clustering loss that encourages the embedding `model.a` to form tight, well-separated clusters aligned with known neuron types. Unlike contrastive loss, this uses the known type structure directly.
- Implementation: `L_cluster = Σ_types ||mean(a[type_k]) - a[i]||² for i in type_k` (intra-class compactness) + margin-based inter-class separation
- Simpler than contrastive, directly tied to the evaluation metric
- Risk: over-constraining embeddings may reduce connectivity R2

**3. Learnable embedding dimension with pruning**
Start with embedding_dim=8 and add an L1 penalty on embedding dimensions, allowing the model to discover the optimal effective dimensionality.
- Implementation: `L_sparse_emb = λ * ||model.a||_1`
- Pro: data-driven dimensionality selection
- Con: adds another hyperparameter (λ), may not help cluster accuracy directly

**4. Replace KMeans with embedding-aware clustering**
The current `evaluate_embedding_clustering` uses KMeans with n_clusters=64 (known). KMeans assumes spherical clusters, which may not match the learned embedding geometry.
- Alternative: Gaussian Mixture Model (already implemented in `clustering_gmm`) which captures ellipsoidal clusters
- Alternative: Spectral clustering (already implemented in `clustering_spectral`) which captures non-convex structures
- This doesn't require code changes to training — only to evaluation
- Quick win: compare KMeans vs GMM vs spectral on existing emb_dim=4 models

### Recommended Priority

1. **Quick win (no training change)**: Evaluate existing emb_dim=4 models with GMM and spectral clustering instead of KMeans — may reveal that cluster_accuracy is artificially limited by the evaluation method
2. **Low risk**: Supervised contrastive or type-prior loss as auxiliary term with small weight (e.g., 0.01× main loss) — test whether cluster accuracy can be pushed above 0.85 without degrading connectivity
3. **Higher risk**: Learnable embedding dimension — more speculative, save for later
