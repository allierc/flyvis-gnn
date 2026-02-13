# Understanding Exploration: Difficult FlyVis Models

## UNDERSTANDING

### Model 049 (svd_rank_99=19, baseline R²=0.634)
**Profile**: 13741 neurons, activity_rank_90=3, activity_rank_99=16, svd_activity_rank_99=19. Low-dimensional neural activity.

**Hypothesis (revised)**: Model 049's catastrophic failure is NOT about low activity rank — the analysis shows the GNN learns **inverted weights** on Type 0 neurons (R²=-3.2). The issue is sign inversion: W_learned has negative correlation with W_true. This suggests a gradient sign problem or the GNN is learning to OPPOSE the true dynamics. Need lower learning rates to prevent runaway gradient issues.

**Status**: revised (based on analysis tool iter 4)
**Evidence FOR (revised hypothesis)**: Analysis iter 4 — connectivity_R²=-1.35, pearson=-0.25. Type 0 neurons have R²=-3.2 while other types have small positive R². The GNN is specifically failing on main neurons by learning inverted weights.
**Evidence AGAINST (original)**: Iter 1 — data_augmentation_loop=30 caused catastrophic failure. More exposure made sign inversion worse.

**Best R² so far**: 0.634 (baseline)
**Next experiment**: Iter 5 — LOWER learning_rate_W from 6E-4→3E-4 to prevent gradient sign instability

### Model 011 (svd_rank_99=45, baseline R²=0.308)
**Profile**: 13741 neurons, activity_rank_90=1(!), activity_rank_99=26, svd_activity_rank_99=45. High activity diversity yet worst R². The paradox: activity_rank_90=1 means 90% variance is in just 1 component despite many neurons being active.

**Hypothesis (revised)**: Model 011 has universally negative R² across ALL neuron types (analysis shows -0.6 to -0.3), unlike other models. This suggests a fundamental mismatch between GNN learning dynamics and this model's W_true structure. The W_true has mean≈0.004 (near zero) but learned W has mean≈-0.022 (negative shift). Need to try: (1) higher learning rate to escape bad basin, or (2) w_init_mode change from zeros to better starting point.

**Status**: revised (based on analysis tool iter 4)
**Evidence FOR (revised hypothesis)**: Analysis iter 4 — ALL 13 neuron types have negative R² (range -0.61 to -0.33). W_learned std (0.186) is much lower than W_true std (0.310), suggesting under-fitting. W_learned mean is negative (-0.022) while W_true mean is near zero (0.004).
**Evidence AGAINST (original)**: Iter 2 — reducing W_L1 from 5E-5→2E-5 caused complete failure.

**Best R² so far**: 0.308 (baseline)
**Next experiment**: Iter 6 — try HIGHER learning_rate_W from 6E-4→1E-3 to help escape the bad basin where all weights drift negative

### Model 041 (svd_rank_99=6, baseline R²=0.629)
**Profile**: 13741 neurons, activity_rank_90=1, activity_rank_99=5, svd_activity_rank_99=6. Near-collapsed activity — only 6 SVD components at 99%.

**Hypothesis (revised)**: Model 041 actually shows BEST recovery pattern in analysis — positive R² across all types, with Type 0 achieving R²=0.39. The training failure (hidden_dim=96) was due to capacity+low-rank signal mismatch. With near-collapsed activity, the GNN should use SIMPLER architecture. The key insight: W_learned std (0.209) is lower than W_true std (0.331) — the model under-estimates weight magnitudes.

**Status**: revised (based on analysis tool iter 4)
**Evidence FOR (revised hypothesis)**: Analysis iter 4 — connectivity_R²=0.33 (positive!), pearson=0.57. All neuron types have positive R² (0.16-0.39). This model is actually LESS broken than 049/011, just needs tuning.
**Evidence AGAINST (original)**: Iter 3 — hidden_dim=96 caused catastrophic loss=12848. More capacity + sparse signal = disaster.

**Best R² so far**: 0.629 (baseline)
**Next experiment**: Iter 7 — try SMALLER hidden_dim=64, hidden_dim_update=64 to match model complexity to low-rank signal

### Model 003 (svd_rank_99=60, baseline R²=0.627)
**Profile**: 13741 neurons, activity_rank_90=3, activity_rank_99=35, svd_activity_rank_99=60. Moderate activity diversity (60 SVD components) but still hard connectivity.

**Hypothesis (strengthened)**: Model 003 shows BEST recovery in analysis — connectivity_R²=0.56, pearson=0.78, Type 0 achieves R²=0.69. The edge_diff=900 mutation is working! Stronger type-consistency is the right direction. W_learned std (0.152) is lower than W_true std (0.271) — still under-estimating weights, suggesting W_L1 may be too high.

**Status**: partially supported (strengthened by analysis)
**Evidence FOR**: (1) Iter 4 — best test_pearson (0.409) of all slots. (2) Analysis iter 4 — connectivity_R²=0.56 (best!), pearson=0.78, Type 0 R²=0.69. Edge_diff=900 is helping.
**Evidence AGAINST**: Test rollout R²=-2441 still shows instability despite good connectivity correlation.

**Best R² so far**: 0.627 (baseline)
**Next experiment**: Iter 8 — continue with edge_diff=900, but REDUCE coeff_W_L1 from 5E-5→3E-5 to allow larger weight magnitudes (addressing the W_learned under-estimation)

---

## Established Principles (from base 62_1 exploration)

1. lr_W=6E-4 with edge_L1=0.3 achieves best conn_R2
2. lr_W=1E-3 requires lr=1E-3 (not 1.2E-3)
3. lr_emb=1.5E-3 is required for lr_W < 1E-3
4. lr_emb >= 1.8E-3 destroys V_rest recovery
5. coeff_edge_norm >= 10 is catastrophic
6. coeff_edge_weight_L1=0.3 is optimal
7. coeff_phi_weight_L1=0.5 improves V_rest recovery
8. coeff_edge_diff=750 is optimal
9. coeff_W_L1=5E-5 is optimal for V_rest
10. coeff_phi_weight_L2 must stay at 0.001
11. n_layers=4 is harmful
12. hidden_dim=80 + hidden_dim_update=80 is optimal architecture
13. batch_size=2 maintains conn_R2 with faster training
14. batch_size >= 3 causes V_rest collapse
15. data_augmentation_loop=20 is viable for speed
16. lr=1.2E-3 is optimal for MLPs

NOTE: These principles were derived on the standard model (R²=0.980). They may not hold for difficult models.

---

## New Principles (discovered in this exploration)

---

## Cross-Model Observations

**Batch 1 cross-model findings:**
1. ALL 4 initial hypotheses led to worse results than baseline — these models are harder than expected
2. Catastrophic rollout R² (negative or -inf) suggests numerical instability, not just poor fit
3. Model 041 (near-collapsed, 6 SVD components) had the WORST loss (12848) when capacity was increased — sparse signal + high capacity = disaster
4. Model 003 showed best test_pearson (0.409) despite failing, suggesting stronger edge_diff may be partially correct direction
5. Reducing W_L1 regularization (Model 011) caused instability — these difficult models may need MORE regularization, not less
6. Longer training (data_aug=30, Model 049) hurt rather than helped — possible overfitting on limited signal

**Analysis iter 4 cross-model findings:**
7. **TWO FAILURE MODES**: Models 049/011 have NEGATIVE connectivity R² (inverted weights), while 041/003 have POSITIVE R² (partial recovery). This is a fundamental difference!
8. **Type 0 dominates**: Type 0 neurons (~425k edges, 98% of all edges) determine overall R². Other types have only 500-930 edges each.
9. **W_learned under-estimates W_true**: ALL 4 models have W_learned std < W_true std. The GNN is too conservative in weight magnitudes.
10. **Correlation structure differs**:
   - Models with NEGATIVE R² (049, 011): GNN is learning INVERTED relationships
   - Models with POSITIVE R² (041, 003): GNN is learning CORRECT direction but under-estimating magnitude
11. **Activity rank vs recovery**: Surprisingly, 041 (lowest rank=6) has BETTER connectivity R² than 049/011. Low rank doesn't necessarily mean worse recovery.

**Emerging pattern (revised)**: The critical differentiator is NOT activity rank but whether the GNN learns correct sign. Models 049/011 have gradient sign issues causing weight inversion. Models 041/003 are on the right track but under-regularized (need to allow larger weights).

---

## Analysis Tools Log

Summary of each analysis tool: what it measured, key findings, and which UNDERSTANDING hypothesis it informed.

| Iter | Tool | What it measured | Key finding | Informed hypothesis |
|------|------|-----------------|-------------|---------------------|
| 4 | analysis_iter_004.py | Connectivity R² per model & per neuron type, W_true vs W_learned stats | Models 049/011 have NEGATIVE R² (inverted weights), 041/003 have positive R². Type 0 (main neurons) dominates failure in 049 (R²=-3.2). All models have W_learned std < W_true std (under-fitting). | All 4 hypotheses revised: 049=sign inversion issue, 011=universal negative R² needs lr_W increase, 041=actually recoverable with smaller dims, 003=edge_diff working needs W_L1 reduction |

---

## Iterations

### Batch 1 (Iters 1-4) — Initial Hypotheses Testing

**Strategy**: Each slot tests its initial hypothesis with a single parameter change from Node 79 baseline.

| Slot | Model | Mutation | Hypothesis being tested |
|------|-------|----------|------------------------|
| 0 | 049 | data_augmentation_loop: 20→30 | More training exposure helps low-rank models |
| 1 | 011 | coeff_W_L1: 5E-5→2E-5 | Less W regularization allows more flexibility for hard connectivity |
| 2 | 041 | hidden_dim: 80→96, hidden_dim_update: 80→96 | More capacity extracts structure from collapsed activity |
| 3 | 003 | coeff_edge_diff: 750→900 | Stronger type-consistency helps with complex connectivity |

**Batch 1 Results:**
- Iter 1 (Model 049): FAILED, test_R2=-1835, test_pearson=0.198, loss=1747, time=53.2min. Hypothesis FALSIFIED.
- Iter 2 (Model 011): FAILED, test_R2=-inf, test_pearson=0.032, loss=2965, time=37.5min. Hypothesis FALSIFIED.
- Iter 3 (Model 041): FAILED, test_R2=-inf, test_pearson=0.056, loss=12848, time=39.6min. Hypothesis FALSIFIED.
- Iter 4 (Model 003): FAILED, test_R2=-2441, test_pearson=0.409, loss=2168, time=38.5min. Hypothesis PARTIALLY SUPPORTED.

**Batch 1 Analysis (iter 4) — KEY INSIGHTS:**
- connectivity_R² reveals TWO FAILURE MODES:
  - 049/011: NEGATIVE R² (inverted weights) — sign problem
  - 041/003: POSITIVE R² (partial recovery) — magnitude problem
- Type 0 neurons (425k edges, 98% of total) dominate results
- ALL models have W_learned std < W_true std — under-fitting magnitudes

### Batch 2 (Iters 5-8) — Testing Revised Hypotheses

| Slot | Model | Mutation | Testing |
|------|-------|----------|---------|
| 0 | 049 | lr_W: 6E-4→3E-4, data_aug: 30→20 | Lower lr_W prevents sign inversion |
| 1 | 011 | lr_W: 6E-4→1E-3, lr: 1.2E-3→1E-3, W_L1: 2E-5→5E-5 | Higher lr_W escapes negative-weight basin |
| 2 | 041 | hidden_dim: 96→64, hidden_dim_update: 96→64 | Smaller capacity for low-rank signal |
| 3 | 003 | coeff_W_L1: 5E-5→3E-5 | Lower W_L1 allows larger weight magnitudes |

