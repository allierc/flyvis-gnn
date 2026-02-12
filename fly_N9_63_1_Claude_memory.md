# FlyVis Working Memory: fly_N9_63_1 (parallel)

## Knowledge Base (accumulated across all blocks)

### Parameter Effects Table
| Block | Focus | Best conn_R2 | Best field_R2 | Best tau_R2 | Best V_rest_R2 | Best Cluster_Acc | Time_min | Key finding |
| ----- | ----- | ------------ | ------------- | ----------- | -------------- | ---------------- | -------- | ----------- |

### Established Principles
(From fly_N9_62_1 - to be validated in this context with learned visual field)
1. lr_W=5E-4 to 7E-4 with lr=1.2E-3 and lr_emb=1.5E-3 is optimal for GNN
2. lr_emb=1.5E-3 is critical for low lr_W - lower values cause connectivity collapse
3. lr_emb >= 1.8E-3 destroys V_rest recovery
4. coeff_edge_norm >= 10 is catastrophic - keep at 1.0
5. coeff_phi_weight_L1=0.5 + coeff_edge_weight_L1=0.5 improves both connectivity and V_rest
6. coeff_edge_diff=750-1000 optimal; 1250+ is harmful
7. coeff_phi_weight_L2 must stay at 0.001 - 0.005 destroys tau and V_rest
8. coeff_W_L1=5E-5 is optimal for V_rest; 1E-4 boosts conn but hurts V_rest
9. SIREN requires exactly 3 layers (from INR experiments - to be validated)
10. High omega_f with high lr can destabilize training (hypothesis)

### Open Questions
- What is the optimal learning_rate_NNR_f for Siren with DAVIS visual input?
- Does adding Siren learning change optimal GNN learning rates?
- What omega_f works best for DAVIS video reconstruction?
- Can field_R2 and connectivity_R2 both reach >0.8 simultaneously?

---

## Previous Block Summary

(None yet - this is Block 1)

---

## Current Block (Block 1)

### Block Info
- Focus: Siren learning rate (learning_rate_NNR_f) exploration
- Range: 1E-8 to 1E-4
- GNN params: Fixed at proven 62_1 values (lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3)

### Hypothesis
The Siren learning rate is the key unknown for joint field+connectivity learning. Too low (1E-8) may prevent field learning; too high (1E-4) may destabilize W recovery. Lower omega_f may allow higher lr_siren without instability.

### Initial Batch (Iter 0-3) - Planned Mutations

| Slot | Node | Parent | Strategy | lr_siren | omega_f | Other changes | Rationale |
| ---- | ---- | ------ | -------- | -------- | ------- | ------------- | --------- |
| 0 | 0 | root | baseline | 1E-8 | 4096 | none | Baseline with default Siren lr |
| 1 | 1 | root | explore | 1E-6 | 4096 | none | 100x higher Siren lr, test if field learns |
| 2 | 2 | root | explore | 1E-4 | 4096 | none | 10000x higher Siren lr, aggressive exploration |
| 3 | 3 | root | explore | 1E-5 | 256 | lower omega | Mid lr + lower omega for stability |

### Iterations This Block

(Results pending - batch submitted)

### Emerging Observations

(Pending first batch results)

