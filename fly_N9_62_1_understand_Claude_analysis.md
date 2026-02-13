# Understanding Exploration Log: Difficult FlyVis Models (parallel)

## Batch 1 (Iters 1-4) — INITIALIZATION

Starting point: Node 79 best params from base exploration (lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, edge_diff=750, phi_L1=0.5, edge_L1=0.3, W_L1=5E-5, hidden_dim=80, hidden_dim_update=80, batch=2, data_aug=20).

Target models:
- Model 049: svd_rank_99=19, baseline R²=0.634 (low activity rank)
- Model 011: svd_rank_99=45, baseline R²=0.308 (worst R², hard connectivity)
- Model 041: svd_rank_99=6, baseline R²=0.629 (near-collapsed activity)
- Model 003: svd_rank_99=60, baseline R²=0.627 (moderate rank, hard connectivity)

### Initial Hypotheses

**Model 049**: Low activity rank (19 SVD components) limits gradient signal. Hypothesis: increasing data augmentation will expose more activity patterns.

**Model 011**: Paradoxical case — activity_rank_90=1 means one dominant mode captures 90% variance. Hypothesis: GNN easily fits dominant mode but fails to learn edge weights because the mode doesn't discriminate. Need to reduce W_L1 to allow more flexibility.

**Model 041**: Near-collapsed activity (6 SVD components). Hypothesis: need more MLP capacity to extract structure from minimal signal.

**Model 003**: Moderate activity (60 SVD components) but hard connectivity. Hypothesis: difficulty is in connectivity structure, not activity. Stronger edge_diff regularization to enforce type-consistency.

### Mutations for Batch 1

- Slot 0 (Model 049): data_augmentation_loop: 20→30
- Slot 1 (Model 011): coeff_W_L1: 5E-5→2E-5
- Slot 2 (Model 041): hidden_dim: 80→96, hidden_dim_update: 80→96
- Slot 3 (Model 003): coeff_edge_diff: 750→900

---

## Iter 1: failed
Node: id=1, parent=0
Model: 049
Mode/Strategy: hypothesis-test
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=2, hidden_dim=80, data_augmentation_loop=30
Metrics: test_R2=-1834.8, test_pearson=0.198, final_loss=1747, training_time_min=53.2
Mutation: data_augmentation_loop: 20 -> 30
Observation: Catastrophic rollout failure (R²=-1835). Longer training may have caused overfitting or instability on low-rank model. The hypothesis that more data exposure helps is NOT supported by this result.
Analysis: connectivity_R²=-1.35, pearson=-0.25. Type 0 neurons have R²=-3.2 (INVERTED weights). GNN is learning opposite of true weights. Sign inversion problem, not capacity problem.
Next: parent=0

## Iter 2: failed
Node: id=2, parent=0
Model: 011
Mode/Strategy: hypothesis-test
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_W_L1=2E-5, batch_size=2, hidden_dim=80, data_augmentation_loop=20
Metrics: test_R2=-inf, test_pearson=0.032, final_loss=2965, training_time_min=37.5
Mutation: coeff_W_L1: 5E-5 -> 2E-5
Observation: Complete training failure (R²=-inf). Reducing W_L1 caused instability. Model 011 may need MORE regularization, not less. Hypothesis falsified.
Analysis: connectivity_R²=-0.58, pearson=-0.17. ALL 13 neuron types have negative R² (-0.61 to -0.33). Universal weight inversion. W_learned mean=-0.022 vs W_true mean=+0.004. GNN drifts weights negative.
Next: parent=0

## Iter 3: failed
Node: id=3, parent=0
Model: 041
Mode/Strategy: hypothesis-test
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=2, hidden_dim=96, hidden_dim_update=96, data_augmentation_loop=20
Metrics: test_R2=-inf, test_pearson=0.056, final_loss=12848, training_time_min=39.6
Mutation: hidden_dim: 80 -> 96, hidden_dim_update: 80 -> 96
Observation: Complete training failure (R²=-inf), highest loss of all slots. Near-collapsed activity model (6 SVD components) may need LESS capacity to avoid overfitting the sparse signal, not more.
Analysis: connectivity_R²=0.33, pearson=0.57 (POSITIVE — best direction of sign!). All types positive R². Type 0=0.39, others=0.16-0.20. Model is actually learning correct direction, just magnitude issues.
Next: parent=0

## Iter 4: failed
Node: id=4, parent=0
Model: 003
Mode/Strategy: hypothesis-test
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=900, coeff_W_L1=5E-5, batch_size=2, hidden_dim=80, data_augmentation_loop=20
Metrics: test_R2=-2441, test_pearson=0.409, final_loss=2168, training_time_min=38.5
Mutation: coeff_edge_diff: 750 -> 900
Observation: Still failed but best test_pearson (0.409) of all slots. Stronger edge_diff may be helping but not enough. Need to investigate connectivity structure further.
Analysis: connectivity_R²=0.56 (BEST!), pearson=0.78. Type 0=0.69, others=0.33. Edge_diff=900 is working! W_learned std=0.152 vs W_true std=0.271 — under-estimating weights. Reduce W_L1 to allow larger magnitudes.
Next: parent=0

---

## Batch 2 (Iters 5-8) — Testing Revised Hypotheses Based on Analysis

**Analysis iter 4 revealed two failure modes:**
1. Models 049/011: NEGATIVE connectivity R² — weight sign inversion problem
2. Models 041/003: POSITIVE connectivity R² — partial recovery, under-estimating magnitudes

**Mutations for Batch 2:**

| Slot | Model | Mutation | Hypothesis being tested |
|------|-------|----------|------------------------|
| 0 | 049 | learning_rate_W_start: 6E-4→3E-4, data_aug: 30→20 (reset) | Lower lr_W may prevent gradient sign instability causing weight inversion |
| 1 | 011 | learning_rate_W_start: 6E-4→1E-3, lr: 1.2E-3→1E-3, W_L1: 2E-5→5E-5 (reset) | Higher lr_W may help escape bad basin where weights drift negative |
| 2 | 041 | hidden_dim: 96→64, hidden_dim_update: 96→64 | Smaller MLP capacity to match 6-component low-rank signal |
| 3 | 003 | coeff_W_L1: 5E-5→3E-5 (keep edge_diff=900) | Lower W_L1 to allow larger weight magnitudes (W_learned std too low) |

---

