# Understanding Exploration: Difficult FlyVis Models

## UNDERSTANDING

### Model 049 (svd_rank_99=19, baseline R²=0.634)
**Profile**: 13741 neurons, activity_rank_99=16, svd_activity_rank_99=19. Low-dimensional neural activity.
**Hypothesis**: [to be filled after first results]
**Status**: untested
**Evidence FOR**:
**Evidence AGAINST**:
**Best R² so far**: 0.634
**Next experiment**: baseline with Node 79 params

### Model 011 (svd_rank_99=45, baseline R²=0.308)
**Profile**: 13741 neurons, activity_rank_99=26, svd_activity_rank_99=45. High activity diversity yet worst R². Hard connectivity structure.
**Hypothesis**: [to be filled after first results]
**Status**: untested
**Evidence FOR**:
**Evidence AGAINST**:
**Best R² so far**: 0.308
**Next experiment**: baseline with Node 79 params

### Model 041 (svd_rank_99=6, baseline R²=0.629)
**Profile**: 13741 neurons, activity_rank_99=5, svd_activity_rank_99=6. Near-collapsed activity. Only 6 SVD components at 99%.
**Hypothesis**: [to be filled after first results]
**Status**: untested
**Evidence FOR**:
**Evidence AGAINST**:
**Best R² so far**: 0.629
**Next experiment**: baseline with Node 79 params

### Model 003 (svd_rank_99=60, baseline R²=0.627)
**Profile**: 13741 neurons, activity_rank_99=35, svd_activity_rank_99=60. Moderate activity diversity but hard connectivity.
**Hypothesis**: [to be filled after first results]
**Status**: untested
**Evidence FOR**:
**Evidence AGAINST**:
**Best R² so far**: 0.627
**Next experiment**: baseline with Node 79 params

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

---

## Analysis Tools Log

Summary of each analysis tool: what it measured, key findings, and which UNDERSTANDING hypothesis it informed.

| Iter | Tool | What it measured | Key finding | Informed hypothesis |
|------|------|-----------------|-------------|---------------------|

---

## Iterations

