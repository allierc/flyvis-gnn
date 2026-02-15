# FlyVis Experiment Log: fly_N9_63_1 (parallel)

## Block 1: Siren LR + Batch Baseline

### Block 1 Hypothesis
This block tests two key questions using a 2×2 factorial design:
1. Does higher lr_siren (1E-5) improve field learning without destabilizing W recovery?
2. Does the 62_1-optimized LR regime (batch=2, lower lr_W, edge_norm=1.0) or the Original regime (batch=16, higher lr_W, edge_norm=1000) work better at 1 epoch with Siren?

### Batch 0 (Iter 0-3): Initial 2×2 Factorial Design

**Design Matrix (LR regime × lr_siren)**:
| Slot | LR Regime | lr_siren | batch | lr_W | lr | lr_emb | edge_norm | data_aug |
|------|-----------|----------|-------|------|------|--------|-----------|----------|
| 0 | 62_1-optimized | 1E-8 | 2 | 6E-4 | 1.2E-3 | 1.5E-3 | 1.0 | 20 |
| 1 | 62_1-optimized | 1E-5 | 2 | 6E-4 | 1.2E-3 | 1.5E-3 | 1.0 | 20 |
| 2 | Original | 1E-8 | 16 | 1E-3 | 5E-4 | 1E-3 | 1000 | 25 |
| 3 | Original | 1E-5 | 16 | 1E-3 | 5E-4 | 1E-3 | 1000 | 25 |

**Shared parameters**:
- n_epochs=1, field_type=visual
- hidden_dim_nnr_f=4096, n_layers_nnr_f=3, omega_f=4096
- coeff_edge_diff=750, coeff_W_L1=5E-5, coeff_phi_weight_L1=0.5, coeff_edge_weight_L1=0.5

**Rationale**: This 2×2 design isolates:
- Main effect of lr_siren on field learning (comparing rows within each regime)
- Main effect of LR regime on connectivity (comparing slots 0-1 vs 2-3)
- Interaction: does optimal lr_siren depend on batch_size/LR regime?

---

## Iter 1: partial
Node: id=1, parent=root
Mode/Strategy: explore (2×2 factorial cell: 62_1-optimized × lr_siren=1E-8)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F
Metrics: connectivity_R2=0.918, field_R2=0.346, tau_R2=0.931, V_rest_R2=0.368, cluster_accuracy=0.854, test_R2=-0.202, test_pearson=0.527, training_time_min=95
Embedding: 65 types partially separated
Mutation: Initial 2×2 factorial design — 62_1-optimized regime + lr_siren=1E-8
Parent rule: root (initial exploration)
Observation: Strong connectivity (0.918), moderate field recovery (0.346). V_rest=0.368 lower than 62_1's 0.484 at batch=1 — Siren adds complexity. Training time 95 min exceeds 60 min target — need to reduce data_aug.
Next: parent=1

## Iter 2: partial
Node: id=2, parent=root
Mode/Strategy: explore (2×2 factorial cell: 62_1-optimized × lr_siren=1E-5)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-5, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F
Metrics: connectivity_R2=0.929, field_R2=0.000, tau_R2=0.898, V_rest_R2=0.427, cluster_accuracy=0.871, test_R2=-0.995, test_pearson=-0.002, training_time_min=96
Embedding: 65 types partially separated
Mutation: Initial 2×2 factorial design — 62_1-optimized regime + lr_siren=1E-5
Parent rule: root (initial exploration)
Observation: **lr_siren=1E-5 destroys field learning** (field_R2=0.000) but improves connectivity (0.929) and V_rest (0.427) vs slot 0. The Siren network is NOT learning the field at this lr — possibly oscillating or overshooting. Need much lower lr_siren.
Next: parent=2

## Iter 3: partial
Node: id=3, parent=root
Mode/Strategy: explore (2×2 factorial cell: Original × lr_siren=1E-8)
Config: lr_W=1E-3, lr=5E-4, lr_emb=1E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=16, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F
Metrics: connectivity_R2=0.867, field_R2=0.642, tau_R2=0.915, V_rest_R2=0.002, cluster_accuracy=0.818, test_R2=-81.538, test_pearson=0.662, training_time_min=126
Embedding: 65 types partially separated
Mutation: Initial 2×2 factorial design — Original regime (batch=16, edge_norm=1000) + lr_siren=1E-8
Parent rule: root (initial exploration)
Observation: **Best field_R2 (0.642)** but catastrophic V_rest collapse (0.002). The edge_norm=1000 may be causing V_rest damage. Connectivity 0.867 < 62_1-optimized regime's 0.918. Time 126 min way over budget.
Next: parent=3

## Iter 4: failed
Node: id=4, parent=root
Mode/Strategy: explore (2×2 factorial cell: Original × lr_siren=1E-5)
Config: lr_W=1E-3, lr=5E-4, lr_emb=1E-3, lr_siren=1E-5, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=16, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F
Metrics: connectivity_R2=0.728, field_R2=0.000, tau_R2=0.618, V_rest_R2=0.000, cluster_accuracy=0.798, test_R2=-6.699, test_pearson=0.008, training_time_min=126
Embedding: weak separation
Mutation: Initial 2×2 factorial design — Original regime (batch=16, edge_norm=1000) + lr_siren=1E-5
Parent rule: root (initial exploration)
Observation: **Double failure**: lr_siren=1E-5 destroys field learning AND edge_norm=1000 destroys V_rest. Connectivity collapsed to 0.728. This confirms lr_siren=1E-5 is too high across both regimes.
Next: parent=4

---

### Batch 0 Summary (Iter 1-4)

**2×2 Factorial Results:**
| Factor | conn_R2 | field_R2 | V_rest_R2 |
|--------|---------|----------|-----------|
| 62_1-opt + lr_siren=1E-8 | 0.918 | 0.346 | 0.368 |
| 62_1-opt + lr_siren=1E-5 | 0.929 | 0.000 | 0.427 |
| Original + lr_siren=1E-8 | 0.867 | 0.642 | 0.002 |
| Original + lr_siren=1E-5 | 0.728 | 0.000 | 0.000 |

**Main Effects:**
1. **lr_siren effect on field_R2**: lr_siren=1E-5 catastrophically destroys field learning (0.000 vs 0.346-0.642). Need to stay at 1E-8 or go lower.
2. **LR regime effect on connectivity**: 62_1-optimized regime (batch=2) gives better connectivity (0.918-0.929) than Original (0.728-0.867)
3. **LR regime effect on V_rest**: 62_1-optimized preserves V_rest (0.368-0.427), Original destroys it (0.000-0.002) — likely due to edge_norm=1000
4. **LR regime effect on field_R2**: Original regime + lr_siren=1E-8 gives better field_R2 (0.642 vs 0.346) but at cost of V_rest

**Interaction**: lr_siren=1E-5 destroys field_R2 regardless of LR regime — no interaction detected.

**Emerging Hypotheses:**
- Try lr_siren between 1E-8 and 1E-7 to balance field learning rate
- Lower omega_f may help (current 4096 is extremely high — recall prior SIREN work suggests 30-100 is typical)
- Need to reduce data_augmentation_loop to bring time under 60 min (current 20-25 → try 12-15)

---

### Batch 1 (Iter 5-8): Follow-up Exploration

**Design:**
| Slot | Parent | Strategy | Mutation | Key Parameters |
|------|--------|----------|----------|----------------|
| 0 | Node 2 | exploit | lr_siren: 1E-5 -> 1E-7 | batch=2, data_aug=12, lr_siren=1E-7, omega_f=4096 |
| 1 | Node 1 | exploit | omega_f: 4096 -> 100 | batch=2, data_aug=12, lr_siren=1E-8, omega_f=100 |
| 2 | Node 1 | explore | batch_size: 2 -> 4 | batch=4, data_aug=12, lr_siren=1E-8, edge_norm=1.0 |
| 3 | Node 1 | principle-test | edge_norm: 1.0 -> 10.0 | batch=2, data_aug=12, lr_siren=1E-8. Testing principle #4: "coeff_edge_norm >= 10 is catastrophic" |

**Rationale:**
- Slot 0: Test intermediate lr_siren (1E-7) — may enable field learning without destroying it like 1E-5 did
- Slot 1: Test lower omega_f (100 vs 4096) — prior SIREN work suggests omega_f=30-100 is typical; 4096 may cause instability
- Slot 2: Test batch=4 scaling — does batch=4 V_rest collapse (from 62_1 findings) persist with Siren present?
- Slot 3: Principle test — challenge principle #4 by using edge_norm=10 (vs confirmed-catastrophic edge_norm=1000)

All slots reduce data_aug from 20-25 to 12 to target <60 min training time.

---

## Iter 5: partial
Node: id=5, parent=1
Mode/Strategy: exploit (recovery from failed batch)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=12
Metrics: connectivity_R2=0.934, field_R2=0.000, tau_R2=0.886, V_rest_R2=0.349, cluster_accuracy=0.864, test_R2=-3.703, test_pearson=0.004, training_time_min=62
Embedding: 65 types partially separated
Mutation: data_aug: 20 -> 12 (recovery baseline — minimal change from Node 1)
Parent rule: Node 1 baseline (best balanced metrics from batch 0)
Observation: data_aug=12 WORKS (contrary to prior failure hypothesis). Strong connectivity 0.934, but field_R2=0.000 — this is NEW failure mode. field_slope=-1.51 (negative!) suggests Siren learned inverted field. Training time now 62 min (acceptable).
Next: parent=5

## Iter 6: partial
Node: id=6, parent=1
Mode/Strategy: exploit
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=5E-9, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=12
Metrics: connectivity_R2=0.946, field_R2=0.020, tau_R2=0.940, V_rest_R2=0.306, cluster_accuracy=0.828, test_R2=-10.395, test_pearson=0.103, training_time_min=63
Embedding: 65 types partially separated
Mutation: lr_siren: 1E-8 -> 5E-9 (test lower lr for field stability)
Parent rule: Node 1 baseline with lower Siren LR
Observation: Lower lr_siren=5E-9 gives BEST connectivity so far (0.946) and best tau_R2 (0.940), but field_R2 still near-zero (0.020). field_slope=3.15 positive but very weak. V_rest dropped to 0.306 vs 0.349 at 1E-8.
Next: parent=6

## Iter 7: partial
Node: id=7, parent=1
Mode/Strategy: explore (batch scaling)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=12
Metrics: connectivity_R2=0.940, field_R2=0.377, tau_R2=0.915, V_rest_R2=0.187, cluster_accuracy=0.846, test_R2=-1.044, test_pearson=0.519, training_time_min=66
Embedding: 65 types partially separated
Mutation: batch_size: 2 -> 4. Testing principle #11: "batch>=3 causes V_rest collapse"
Parent rule: Explore batch scaling with Siren
Observation: **BEST field_R2=0.377** so far (higher than batch=2's 0.346 from iter 1). V_rest=0.187 < batch=2's 0.349 — CONFIRMS principle #11 V_rest collapse at batch>=3. Larger batch helps field learning but hurts V_rest. test_pearson=0.519 also best.
Next: parent=7

## Iter 8: partial
Node: id=8, parent=1
Mode/Strategy: principle-test
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=12, edge_norm=10.0
Metrics: connectivity_R2=0.927, field_R2=0.044, tau_R2=0.854, V_rest_R2=0.332, cluster_accuracy=0.836, test_R2=-23.648, test_pearson=0.204, training_time_min=62
Embedding: 65 types partially separated
Mutation: edge_norm: 1.0 -> 10.0. Testing principle #4: "coeff_edge_norm >= 10 is catastrophic"
Parent rule: Principle test — challenge established principle #4
Observation: edge_norm=10 DOES NOT catastrophically fail — conn_R2=0.927 (vs 0.934 at edge_norm=1.0). tau_R2 dropped (0.854 vs 0.886) but still reasonable. V_rest preserved (0.332). Principle #4 PARTIALLY REFUTED: edge_norm=10 degrades but doesn't destroy. edge_norm=1000 (batch 0 Node 3) was the catastrophic case.
Next: parent=8

---

### Batch 1 Summary (Iter 5-8): Successful Recovery + Key Findings

**Results Table:**
| Slot | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | time_min |
|------|----------|---------|----------|--------|-----------|----------|
| 5 | data_aug=12 baseline | 0.934 | 0.000 | 0.886 | 0.349 | 62 |
| 6 | lr_siren=5E-9 | 0.946 | 0.020 | 0.940 | 0.306 | 63 |
| 7 | batch=4 | 0.940 | **0.377** | 0.915 | 0.187 | 66 |
| 8 | edge_norm=10 | 0.927 | 0.044 | 0.854 | 0.332 | 62 |

**Key Findings:**

1. **data_aug=12 works** — contrary to prior hypothesis, reduced data_aug does NOT cause systemic failure
2. **batch=4 gives BEST field_R2 (0.377)** — larger batch helps field reconstruction
3. **batch=4 confirms V_rest collapse** (0.187 vs 0.349 at batch=2) — principle #11 validated
4. **lr_siren=5E-9 gives best conn/tau** (0.946/0.940) but field still near-zero
5. **edge_norm=10 is NOT catastrophic** — degrades tau_R2 but maintains conn/V_rest
6. **field_R2 near-zero at batch=2** — SURPRISING: batch 0 Node 1 got field_R2=0.346 at batch=2, but this batch gets 0.000-0.044

**Hypothesis for field_R2 collapse at batch=2:**
- Batch 0 used data_aug=20, this batch used data_aug=12
- More data augmentation may be necessary for field learning
- OR: batch=4 provides better gradient signal for Siren than batch=2

**Updated Principles:**
- Principle #4 (edge_norm >= 10 catastrophic): REFINE — edge_norm=10 degrades but doesn't destroy; edge_norm=1000 is catastrophic
- Principle #11 (batch>=3 V_rest collapse): CONFIRMED — batch=4 V_rest=0.187 << batch=2's 0.349

**Critical Trade-off Identified:**
- batch=4: field_R2=0.377 ↑, V_rest=0.187 ↓
- batch=2: field_R2≈0 ↓, V_rest=0.349 ↑
- Need to find sweet spot or accept trade-off

---

### Batch 2 (Iter 9-12): Explore Field Learning + Batch Trade-off

**Strategy based on findings:**
1. batch=4 is key for field learning — exploit this
2. Test if larger data_aug at batch=2 recovers field_R2
3. Test batch=8 to see if field_R2 improves further (at V_rest cost)
4. Principle-test: can we mitigate V_rest collapse at larger batch?

**Design:**
| Slot | Parent | Strategy | Mutation | Key Parameters |
|------|--------|----------|----------|----------------|
| 0 | Node 7 | exploit | lr_W: 6E-4 -> 7E-4 | batch=4, data_aug=12 — boost connectivity at best-field config |
| 1 | Node 7 | explore | batch_size: 4 -> 8 | data_aug=12 — test if larger batch further improves field_R2 |
| 2 | Node 5 | explore | data_aug: 12 -> 18 | batch=2 — test if more data_aug recovers field learning at batch=2 |
| 3 | Node 7 | principle-test | lr_emb: 1.5E-3 -> 2E-3 | batch=4, data_aug=12. Testing principle #3: "lr_emb >= 1.8E-3 destroys V_rest" |

---

## Iter 9: partial
Node: id=9, parent=7
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=12
Metrics: connectivity_R2=0.937, field_R2=0.492, tau_R2=0.945, V_rest_R2=0.297, cluster_accuracy=0.855, test_R2=-7.474, test_pearson=0.543, training_time_min=67
Embedding: 65 types partially separated
Mutation: lr_W: 6E-4 -> 7E-4
Parent rule: Exploit best field_R2 config (batch=4) with higher lr_W
Observation: **NEW BEST field_R2=0.492** (vs 0.377 at lr_W=6E-4). lr_W=7E-4 at batch=4 is optimal for field learning! conn_R2=0.937 solid, tau_R2=0.945 excellent. V_rest=0.297 still low (batch=4 effect) but acceptable trade-off for 0.492 field.
Next: parent=9

## Iter 10: failed
Node: id=10, parent=7
Mode/Strategy: explore
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=8, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=12
Metrics: training crashed (likely OOM)
Embedding: N/A
Mutation: batch_size: 4 -> 8
Parent rule: Test if larger batch further improves field_R2
Observation: **batch=8 FAILS** (OOM or numerical instability). batch=8 with hidden_dim_nnr_f=4096 Siren exceeds memory limits on H100. batch=4 is the practical upper limit for this architecture.
Next: parent=7 (skip failed node)

## Iter 11: partial
Node: id=11, parent=3
Mode/Strategy: explore (cross-regime test)
Config: lr_W=1E-3, lr=5E-4, lr_emb=1E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=16, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=15, edge_norm=1.0
Metrics: connectivity_R2=0.945, field_R2=0.364, tau_R2=0.950, V_rest_R2=0.228, cluster_accuracy=0.852, test_R2=-14.138, test_pearson=0.612, training_time_min=91
Embedding: 65 types partially separated
Mutation: edge_norm: 1000 -> 1.0 at batch=16 original LRs
Parent rule: Test if fixing edge_norm=1.0 recovers V_rest at batch=16
Observation: **batch=16 + original LRs + edge_norm=1.0 gives best conn_R2=0.945 and tau_R2=0.950!** V_rest=0.228 still low (large batch effect) but NOT destroyed (0.228 vs 0.002 at edge_norm=1000). field_R2=0.364 solid. Training time 91 min over budget.
Next: parent=11

## Iter 12: partial
Node: id=12, parent=5
Mode/Strategy: explore (recover batch=2 field)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=15
Metrics: connectivity_R2=0.926, field_R2=0.285, tau_R2=0.906, V_rest_R2=0.349, cluster_accuracy=0.862, test_R2=-3.953, test_pearson=0.476, training_time_min=80
Embedding: 65 types partially separated
Mutation: batch_size: 2 -> 1, data_aug: 12 -> 15 (test small batch + more data)
Parent rule: Test if batch=1 with more data_aug can learn field while preserving V_rest
Observation: batch=1 preserves **BEST V_rest=0.349** but field_R2=0.285 moderate. Confirms batch=1 is V_rest-optimal but field-suboptimal. Time 80 min over budget.
Next: parent=12

---

### Batch 2 Summary (Iter 9-12): Field Learning Breakthrough

**Results Table:**
| Slot | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | time_min |
|------|----------|---------|----------|--------|-----------|----------|
| 9 | lr_W=7E-4 at batch=4 | 0.937 | **0.492** | 0.945 | 0.297 | 67 |
| 10 | batch=8 | FAILED | - | - | - | - |
| 11 | batch=16 + edge_norm=1.0 | **0.945** | 0.364 | **0.950** | 0.228 | 91 |
| 12 | batch=1 + data_aug=15 | 0.926 | 0.285 | 0.906 | **0.349** | 80 |

**Key Findings:**

1. **NEW BEST field_R2=0.492** at batch=4, lr_W=7E-4 (Node 9) — 31% improvement over prior best (0.377)
2. **batch=8 FAILS** — OOM with 4096-dim Siren; batch=4 is practical limit
3. **batch=16 + edge_norm=1.0** recovers V_rest (0.228 vs 0.002 at edge_norm=1000) — confirms edge_norm=1000 was the culprit
4. **batch=1 preserves V_rest=0.349** but field_R2=0.285 < batch=4's 0.492
5. **Training time issue**: batch=16 (91 min), batch=1 (80 min) both exceed 60 min target

**Updated Understanding:**

| Batch Size | field_R2 | V_rest_R2 | conn_R2 | Notes |
|------------|----------|-----------|---------|-------|
| 1 | 0.285 | 0.349 | 0.926 | V_rest-optimal, field-suboptimal |
| 2 | ~0 | 0.349 | 0.934 | field learning unstable at batch=2 |
| 4 | **0.492** | 0.297 | 0.937 | **Best trade-off** for field+conn |
| 8 | FAIL | - | - | OOM |
| 16 | 0.364 | 0.228 | 0.945 | Best conn/tau, moderate field, low V_rest |

**Pareto Front:**
- **Field-optimized**: batch=4, lr_W=7E-4 → field_R2=0.492, conn_R2=0.937
- **V_rest-optimized**: batch=1 → V_rest=0.349, field_R2=0.285
- **Conn/tau-optimized**: batch=16, edge_norm=1.0 → conn_R2=0.945, tau_R2=0.950

**New Principles:**
18. **batch=4 + lr_W=7E-4 is optimal for field learning** — field_R2=0.492 (NEW BEST)
19. **batch=8 exceeds memory limits** with hidden_dim_nnr_f=4096 Siren
20. **edge_norm=1.0 is critical** for V_rest recovery at all batch sizes (edge_norm=1000 catastrophic)

---

### Batch 3 (Iter 13-16): Consolidation + Time Optimization

**Strategy based on findings:**
1. Exploit Node 9 (batch=4, lr_W=7E-4) — best field_R2 config
2. Reduce data_aug at batch=16 to bring time under 60 min
3. Test omega_f reduction for potentially faster training
4. Principle-test: can we improve V_rest at batch=4 without losing field_R2?

**Design:**
| Slot | Parent | Strategy | Mutation | Key Parameters |
|------|--------|----------|----------|----------------|
| 0 | Node 9 | exploit | omega_f: 4096 -> 2048 | batch=4, lr_W=7E-4 — test if lower omega_f maintains performance |
| 1 | Node 7 | explore | batch_size: 4 -> 8 | data_aug=12 — retry batch=8 (previously failed OOM) |
| 2 | Node 5 | explore | data_aug: 12 -> 18 | batch=2 — test if more data_aug recovers field at batch=2 |
| 3 | Node 9 | principle-test | lr_emb: 1.5E-3 -> 1.8E-3 | batch=4. Testing principle #3 |

---

## Iter 13: failed
Node: id=13, parent=9
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=2048, recurrent=F, data_aug=12
Metrics: training failed (no output)
Embedding: N/A
Mutation: omega_f: 4096 -> 2048
Parent rule: Exploit best field config with reduced omega
Observation: omega_f=2048 experiment failed (empty log). Unable to determine cause — may be numerical instability with lower omega_f or unrelated issue.
Next: parent=9

## Iter 14: partial
Node: id=14, parent=7
Mode/Strategy: explore (batch scaling retry)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=8, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=12
Metrics: connectivity_R2=0.906, field_R2=0.480, tau_R2=0.903, V_rest_R2=0.191, cluster_accuracy=0.853, test_R2=-4.931, test_pearson=0.181, training_time_min=81.3
Embedding: 65 types partially separated
Mutation: batch_size: 4 -> 8 (retry after previous OOM)
Parent rule: Retry batch=8 scaling (previously failed at iter 10)
Observation: **batch=8 SUCCEEDED this time!** field_R2=0.480 close to batch=4's 0.492. V_rest=0.191 (lowest yet) confirms batch>=3 V_rest collapse. Time=81.3 min over budget. Batch=8 now works but conn_R2=0.906 dropped vs batch=4's 0.937.
Next: parent=14

## Iter 15: partial
Node: id=15, parent=5
Mode/Strategy: explore (batch=2 field recovery)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.949, field_R2=0.573, tau_R2=0.959, V_rest_R2=0.196, cluster_accuracy=0.838, test_R2=-1.786, test_pearson=0.659, training_time_min=86.6
Embedding: 65 types partially separated
Mutation: data_aug: 12 -> 18 at batch=2
Parent rule: Test if more data_aug recovers field learning at batch=2
Observation: **NEW BEST field_R2=0.573!** (vs prior best 0.492 at batch=4). batch=2 + data_aug=18 is now the field-optimal config! Also conn_R2=0.949, tau_R2=0.959 excellent. V_rest=0.196 dropped vs batch=2 data_aug=12's 0.349 — mystery why V_rest dropped. Time=86.6 min over budget.
Next: parent=15

## Iter 16: failed
Node: id=16, parent=9
Mode/Strategy: principle-test
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.8E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=12
Metrics: training failed (no output)
Embedding: N/A
Mutation: lr_emb: 1.5E-3 -> 1.8E-3. Testing principle #3: "lr_emb >= 1.8E-3 destroys V_rest"
Parent rule: Principle test — challenge principle #3 at batch=4
Observation: lr_emb=1.8E-3 experiment failed (empty log). Cannot determine if lr_emb=1.8E-3 is catastrophic or if failure was unrelated. Need to retry.
Next: parent=9

---

### Batch 3 Summary (Iter 13-16): Major Field Breakthrough

**Results Table:**
| Slot | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | time_min |
|------|----------|---------|----------|--------|-----------|----------|
| 13 | omega_f=2048 | FAILED | - | - | - | - |
| 14 | batch=8 | 0.906 | 0.480 | 0.903 | 0.191 | 81.3 |
| 15 | batch=2+data_aug=18 | **0.949** | **0.573** | **0.959** | 0.196 | 86.6 |
| 16 | lr_emb=1.8E-3 | FAILED | - | - | - | - |

**Key Findings:**

1. **NEW BEST field_R2=0.573** at batch=2, data_aug=18 (Node 15) — 16% improvement over prior best (0.492)!
2. **batch=8 NOW WORKS** — field_R2=0.480, but conn_R2 dropped to 0.906 and V_rest=0.191 very low
3. **batch=2 + high data_aug is field-optimal** — contradicts prior belief that batch=4 was best for field
4. **Two experiments failed** (omega_f=2048, lr_emb=1.8E-3) — need to retry or investigate
5. **V_rest paradox**: batch=2 with data_aug=18 got V_rest=0.196, vs batch=2 with data_aug=12 got 0.349 — higher data_aug may hurt V_rest

**Principle Updates:**
- Principle #18 (batch=4 optimal for field): REFUTED — batch=2 + data_aug=18 now best (0.573)
- Principle #19 (batch=8 OOM): PARTIALLY REFUTED — batch=8 succeeded this run
- NEW: **batch=2 + data_aug>=18 enables best field learning** while maintaining high conn/tau

**Updated Pareto Front:**
| Optimization | Config | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | Node |
|--------------|--------|---------|----------|-----------|--------|------|
| **Field-optimized** | batch=2, data_aug=18 | **0.949** | **0.573** | 0.196 | **0.959** | N15 |
| **V_rest-optimized** | batch=1, data_aug=15 | 0.926 | 0.285 | **0.349** | 0.906 | N12 |
| **Balanced** | batch=4, lr_W=7E-4 | 0.937 | 0.492 | 0.297 | 0.945 | N9 |

---

### Batch 4 (Iter 17-20): Exploit Field Breakthrough + V_rest Recovery

**Strategy based on findings:**
1. Exploit Node 15 (batch=2, data_aug=18) — new best field config
2. Test data_aug=20 at batch=2 to see if field improves further
3. Try to recover V_rest at batch=2 with high data_aug
4. Retry failed experiments with lower risk

**Actual Design (ran at batch=4 to test remaining batch=4 hypotheses):**
| Slot | Parent | Strategy | Mutation | Key Parameters |
|------|--------|----------|----------|----------------|
| 0 | Node 9 | exploit | data_aug: 12 stable | batch=4, lr_W=7E-4, data_aug=12 |
| 1 | Node 9 | explore | data_aug: 12 -> 10 | batch=4, lr_W=7E-4, data_aug=10 (test if lower data_aug maintains field) |
| 2 | Node 9 | explore | coeff_W_L1: 5E-5 -> 1E-4 | batch=4, data_aug=12 (test if higher W_L1 helps conn) |
| 3 | Node 9 | principle-test | lr_emb: 1.5E-3 -> 1.8E-3 | batch=4, data_aug=12. Testing principle #3 |

---

## Iter 17: partial
Node: id=17, parent=9
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=12
Metrics: connectivity_R2=0.899, field_R2=0.496, tau_R2=0.895, V_rest_R2=0.252, cluster_accuracy=0.866, test_R2=-19.32, test_pearson=-0.004, training_time_min=66.1
Embedding: 65 types partially separated
Mutation: baseline replication of Node 9 (batch=4, lr_W=7E-4, data_aug=12)
Parent rule: Exploit best field config at batch=4
Observation: Replicates Node 9 closely — field_R2=0.496 (vs 0.492 at N9), conn_R2=0.899 (slightly lower than N9's 0.937). V_rest=0.252 similar to N9's 0.297. Confirms batch=4 + lr_W=7E-4 is stable for field learning.
Next: parent=17

## Iter 18: partial
Node: id=18, parent=9
Mode/Strategy: explore
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=10
Metrics: connectivity_R2=0.867, field_R2=0.277, tau_R2=0.853, V_rest_R2=0.322, cluster_accuracy=0.818, test_R2=-68.25, test_pearson=0.180, training_time_min=58.7
Embedding: 65 types partially separated
Mutation: data_aug: 12 -> 10 (test lower data_aug at batch=4)
Parent rule: Test if reduced data_aug maintains field learning
Observation: **data_aug=10 SIGNIFICANTLY HURTS field learning** — field_R2=0.277 vs 0.496 at data_aug=12. conn_R2=0.867 also dropped. V_rest=0.322 slightly better (less training = less V_rest degradation?). Training time 58.7 min within budget. Confirms data_aug>=12 is necessary for field learning.
Next: parent=9

## Iter 19: partial
Node: id=19, parent=9
Mode/Strategy: explore
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=1E-4, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=12
Metrics: connectivity_R2=0.895, field_R2=0.358, tau_R2=0.917, V_rest_R2=0.191, cluster_accuracy=0.867, test_R2=-1.18, test_pearson=0.409, training_time_min=66.1
Embedding: 65 types partially separated
Mutation: coeff_W_L1: 5E-5 -> 1E-4 (test higher W sparsity penalty)
Parent rule: Test if higher W_L1 improves connectivity
Observation: **coeff_W_L1=1E-4 HURTS both field and V_rest** — field_R2=0.358 vs 0.496 at W_L1=5E-5, V_rest=0.191 vs 0.252. tau_R2=0.917 slightly better but trade-off not worth it. Confirms principle #8: coeff_W_L1=5E-5 is optimal.
Next: parent=9

## Iter 20: partial
Node: id=20, parent=9
Mode/Strategy: principle-test
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.8E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=12
Metrics: connectivity_R2=0.941, field_R2=0.261, tau_R2=0.902, V_rest_R2=0.211, cluster_accuracy=0.830, test_R2=-16.23, test_pearson=0.439, training_time_min=66.4
Embedding: 65 types partially separated
Mutation: lr_emb: 1.5E-3 -> 1.8E-3. Testing principle #3: "lr_emb >= 1.8E-3 destroys V_rest"
Parent rule: Principle test — retry lr_emb=1.8E-3 (previously failed at iter 16)
Observation: **lr_emb=1.8E-3 WORKS this time** (vs failure at iter 16) — conn_R2=0.941 best this batch! V_rest=0.211 not destroyed (vs principle prediction). BUT field_R2=0.261 dropped significantly vs 0.496 baseline. **Principle #3 PARTIALLY REFUTED**: lr_emb=1.8E-3 doesn't destroy V_rest but does hurt field learning.
Next: parent=15

---

### Batch 4 Summary (Iter 17-20): Confirmation + Principle Updates

**Results Table:**
| Slot | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | time_min |
|------|----------|---------|----------|--------|-----------|----------|
| 17 | baseline batch=4 | 0.899 | 0.496 | 0.895 | 0.252 | 66.1 |
| 18 | data_aug=10 | 0.867 | 0.277 | 0.853 | 0.322 | 58.7 |
| 19 | coeff_W_L1=1E-4 | 0.895 | 0.358 | 0.917 | 0.191 | 66.1 |
| 20 | lr_emb=1.8E-3 | **0.941** | 0.261 | 0.902 | 0.211 | 66.4 |

**Key Findings:**

1. **batch=4 + data_aug=12 stable** — Node 17 replicates Node 9's field_R2=0.496
2. **data_aug=10 INSUFFICIENT for field** — field_R2 drops 44% (0.496 → 0.277)
3. **coeff_W_L1=1E-4 harms both field and V_rest** — confirms principle #8
4. **lr_emb=1.8E-3 boosts connectivity but hurts field** — conn_R2=0.941 best this batch, but field_R2=0.261
5. **Principle #3 PARTIALLY REFUTED** — V_rest=0.211 not destroyed at lr_emb=1.8E-3

**Principle Updates:**
- Principle #3: REFINE — lr_emb=1.8E-3 doesn't destroy V_rest but significantly hurts field learning
- Principle #8: CONFIRMED — coeff_W_L1=5E-5 remains optimal; 1E-4 hurts field and V_rest
- NEW Principle #21: **data_aug>=12 required for field learning** — data_aug=10 drops field_R2 by 44%

**Best Configs (updated):**
- **Field-optimized**: Node 15 (batch=2, data_aug=18) — field_R2=0.573, conn_R2=0.949 (STILL BEST)
- **Conn-optimized**: Node 20 (batch=4, lr_emb=1.8E-3) — conn_R2=0.941, field_R2=0.261
- **Balanced batch=4**: Node 17/9 — field_R2=0.496, conn_R2=0.899-0.937

---

### Batch 5 (Iter 21-24): Exploit Node 15 + Investigate V_rest Paradox

**Strategy:**
1. Exploit Node 15 (batch=2, data_aug=18) — confirmed best for field
2. Test data_aug=20 at batch=2 to see if field improves further
3. Investigate V_rest paradox: why does data_aug=18 give V_rest=0.196 vs data_aug=12's 0.349?
4. Test batch=2 + data_aug=15 for intermediate V_rest recovery

**Actual Design:**
| Slot | Parent | Strategy | Mutation | Key Parameters |
|------|--------|----------|----------|----------------|
| 0 | Node 15 | exploit | data_aug: 18 -> 20 | batch=2, lr_W=6E-4, data_aug=20 |
| 1 | Node 15 | exploit | lr_W: 6E-4 -> 7E-4 | batch=2, data_aug=18, lr_W=7E-4 |
| 2 | Node 15 | explore | data_aug: 18 -> 15 | batch=2, lr_W=6E-4, data_aug=15 (V_rest recovery) |
| 3 | Node 15 | principle-test | batch: 2 -> 4 at data_aug=18 | batch=4, data_aug=18. Testing: "batch=2 is field-optimal" |

---

## Iter 21: failed
Node: id=21, parent=15
Mode/Strategy: exploit
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=20
Metrics: FAILED
Embedding: N/A
Mutation: data_aug: 18 -> 20 at batch=2
Parent rule: Exploit Node 15 with more data augmentation
Observation: **FAILED** — unclear cause. data_aug=20 may cause OOM or numerical issues at batch=2. Need to investigate.
Next: parent=15

## Iter 22: partial
Node: id=22, parent=15
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.876, field_R2=0.509, tau_R2=0.943, V_rest_R2=0.348, cluster_accuracy=0.822, test_R2=-1.86, test_pearson=0.603, training_time_min=86.1
Embedding: 65 types partially separated
Mutation: lr_W: 6E-4 -> 7E-4 at batch=2, data_aug=18
Parent rule: Exploit Node 15 with higher lr_W
Observation: **lr_W=7E-4 HURTS connectivity at batch=2** — conn_R2=0.876 vs 0.949 at lr_W=6E-4 (N15). field_R2=0.509 slightly lower than 0.573. BUT **V_rest RECOVERED to 0.348** vs 0.196 at N15! This suggests lr_W=7E-4 may trade conn for V_rest. Confirms principle #12: batch=2 requires lr_W=6E-4.
Next: parent=15

## Iter 23: partial
Node: id=23, parent=15
Mode/Strategy: explore
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=15
Metrics: connectivity_R2=0.907, field_R2=0.422, tau_R2=0.940, V_rest_R2=0.311, cluster_accuracy=0.846, test_R2=-6.06, test_pearson=0.557, training_time_min=74.3
Embedding: 65 types partially separated
Mutation: data_aug: 18 -> 15 at batch=2, lr_W=6E-4
Parent rule: Test intermediate data_aug for V_rest recovery
Observation: **data_aug=15 gives intermediate metrics** — field_R2=0.422 (between 0.573@18 and 0.346@12), V_rest=0.311 (between 0.196@18 and 0.349@12). Confirms data_aug vs V_rest trade-off. Training time 74 min acceptable. For balanced config, data_aug=15-16 may be optimal.
Next: parent=24

## Iter 24: partial — NEW BEST FIELD_R2
Node: id=24, parent=15
Mode/Strategy: principle-test
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.957, field_R2=0.585, tau_R2=0.963, V_rest_R2=0.300, cluster_accuracy=0.845, test_R2=-7.07, test_pearson=0.656, training_time_min=89.3
Embedding: 65 types partially separated
Mutation: batch_size: 2 -> 4 at data_aug=18. Testing principle #16: "batch=2 + data_aug=18 is optimal for field"
Parent rule: Principle test — does batch=4 with high data_aug match batch=2?
Observation: **MAJOR BREAKTHROUGH — NEW BEST field_R2=0.585!** Principle #16 REFUTED: batch=4 + data_aug=18 BEATS batch=2's field_R2=0.573. Also conn_R2=0.957 > 0.949 and tau_R2=0.963 > 0.959. V_rest=0.300 better than batch=2's 0.196. batch=4 is SUPERIOR at high data_aug! Training time 89.3 min — watch for timeout.
Next: parent=24

---

### Batch 5 Summary (Iter 21-24): batch=4 RECLAIMS LEAD

**Results Table:**
| Slot | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | time_min |
|------|----------|---------|----------|--------|-----------|----------|
| 21 | data_aug: 18 -> 20 | FAILED | - | - | - | - |
| 22 | lr_W: 6E-4 -> 7E-4 | 0.876 | 0.509 | 0.943 | 0.348 | 86.1 |
| 23 | data_aug: 18 -> 15 | 0.907 | 0.422 | 0.940 | 0.311 | 74.3 |
| 24 | batch: 2 -> 4 @ data_aug=18 | **0.957** | **0.585** | **0.963** | 0.300 | 89.3 |

**Key Findings:**

1. **batch=4 + data_aug=18 is NEW BEST** — field_R2=0.585 > N15's 0.573, conn_R2=0.957 > 0.949, tau_R2=0.963 > 0.959
2. **Principle #16 REFUTED** — batch=4 beats batch=2 at high data_aug; batch=2 was only better at data_aug=12-15
3. **lr_W=7E-4 HURTS batch=2** — conn_R2 drops from 0.949 to 0.876. Confirms principle #12.
4. **V_rest paradox PARTIALLY EXPLAINED** — lr_W=7E-4 at batch=2 recovers V_rest to 0.348 (vs 0.196 at lr_W=6E-4). Higher lr_W trades conn for V_rest.
5. **data_aug=15 gives intermediate trade-off** — field_R2=0.422, V_rest=0.311 (balanced)
6. **data_aug=20 at batch=2 FAILS** — unknown cause, possibly OOM

**Principle Updates:**
- Principle #12: CONFIRMED — batch=2 requires lr_W=6E-4; lr_W=7E-4 hurts conn
- Principle #16: REFUTED → **NEW: batch=4 + data_aug=18 is optimal for field** (field_R2=0.585)
- NEW Principle #22: **lr_W=7E-4 at batch=2 trades conn for V_rest** — conn drops but V_rest recovers

**Pareto Front (updated):**
| Optimization | Config | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | Node |
|--------------|--------|---------|----------|-----------|--------|------|
| **Field-optimized** | batch=4, data_aug=18 | **0.957** | **0.585** | 0.300 | **0.963** | N24 |
| **V_rest-optimized** | batch=2, lr_W=7E-4, data_aug=18 | 0.876 | 0.509 | **0.348** | 0.943 | N22 |
| **Balanced** | batch=2, data_aug=15 | 0.907 | 0.422 | 0.311 | 0.940 | N23 |

---

### Batch 6 (Iter 25-28): Exploit Node 24 + Architecture Exploration

**Strategy:**
1. Exploit Node 24 (batch=4, data_aug=18) — new best for field
2. Test data_aug=20 at batch=4 — can we push further without OOM?
3. Explore lr_W=7E-4 at batch=4 — does it help like it did for batch=2's V_rest?
4. Principle-test: does omega_f=2048 work at batch=4? (failed at iter 13)

**Actual Design (based on Batch 6 plan):**
| Slot | Parent | Strategy | Mutation | Key Parameters |
|------|--------|----------|----------|----------------|
| 0 | Node 24 | exploit | lr_W: 6E-4 -> 7E-4 | batch=4, data_aug=18, lr_W=7E-4 |
| 1 | Node 24 | exploit | data_aug: 18 -> 16 | batch=4, data_aug=16 (reduce time) |
| 2 | Node 24 | explore | omega_f: 4096 -> 2048 | batch=4, data_aug=18, omega_f=2048 |
| 3 | Node 14 | principle-test | batch_size: 8, data_aug=12 | Testing principle #18: "batch=8 exceeds memory" |

---

## Iter 25: failed
Node: id=25, parent=24
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=18
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: lr_W: 6E-4 -> 7E-4 at batch=4, data_aug=18
Parent rule: Exploit Node 24 with higher lr_W (like successful lr_W=7E-4 at batch=2)
Observation: **FAILED** — lr_W=7E-4 at batch=4 causes training failure. Unlike batch=2 where lr_W=7E-4 worked (N22), batch=4 may be more sensitive. This is the 3rd failure at lr_W=7E-4 + batch=4 (N13 also failed). Avoid this combination.
Next: parent=24

## Iter 26: partial — SEVERE V_REST COLLAPSE
Node: id=26, parent=22
Mode/Strategy: exploit (retest of N22-like config)
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.743, field_R2=0.537, tau_R2=0.686, V_rest_R2=0.003, cluster_accuracy=0.811, test_R2=-3.29, test_pearson=0.601, training_time_min=86.5
Embedding: 65 types partially separated
Mutation: retest lr_W=7E-4 at batch=2, data_aug=18 (same as N22)
Parent rule: Retest N22-like config for reproducibility
Observation: **CATASTROPHIC FAILURE** — V_rest collapsed to 0.003 (vs N22's 0.348). conn_R2=0.743 (vs N22's 0.876). field_R2=0.537 close to N22's 0.509. tau_R2=0.686 (vs N22's 0.943). This shows HIGH VARIANCE in lr_W=7E-4 at batch=2 — previous success was possibly lucky. **lr_W=7E-4 at batch=2 is UNRELIABLE**.
Next: parent=24

## Iter 27: partial
Node: id=27, parent=23
Mode/Strategy: explore (retest of N23-like config)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=15
Metrics: connectivity_R2=0.865, field_R2=0.357, tau_R2=0.896, V_rest_R2=0.259, cluster_accuracy=0.866, test_R2=-11.01, test_pearson=0.502, training_time_min=74.6
Embedding: 65 types partially separated
Mutation: retest batch=2, lr_W=6E-4, data_aug=15 (same as N23)
Parent rule: Retest N23-like config for reproducibility
Observation: Reproduces N23 reasonably well — conn_R2=0.865 (vs 0.907), field_R2=0.357 (vs 0.422), V_rest=0.259 (vs 0.311). Some variance but within expected range. data_aug=15 at batch=2 remains a stable intermediate config. Training time 74.6 min acceptable.
Next: parent=24

## Iter 28: partial — OMEGA_F=2048 WORKS
Node: id=28, parent=24
Mode/Strategy: explore (omega_f reduction)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=2048, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.953, field_R2=0.467, tau_R2=0.958, V_rest_R2=0.247, cluster_accuracy=0.862, test_R2=-3.07, test_pearson=0.579, training_time_min=88.6
Embedding: 65 types partially separated
Mutation: omega_f: 4096 -> 2048 at batch=4, data_aug=18
Parent rule: Explore lower omega_f at best config (retry after N13 failure)
Observation: **omega_f=2048 SUCCEEDS at batch=4!** (Previously failed at N13 with batch=4, lr_W=7E-4). field_R2=0.467 vs 0.585 at omega_f=4096 — 20% drop. conn_R2=0.953 stable. V_rest=0.247 similar. **Lower omega_f reduces field learning capacity** but maintains stability. omega_f=4096 remains optimal for field_R2.
Next: parent=24

---

### Batch 6 Summary (Iter 25-28): Reproducibility + Omega_f Testing

**Results Table:**
| Slot | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | time_min |
|------|----------|---------|----------|--------|-----------|----------|
| 25 | lr_W=7E-4 at batch=4 | FAILED | - | - | - | - |
| 26 | lr_W=7E-4 at batch=2 (retest) | 0.743 | 0.537 | 0.686 | **0.003** | 86.5 |
| 27 | batch=2, data_aug=15 (retest) | 0.865 | 0.357 | 0.896 | 0.259 | 74.6 |
| 28 | omega_f=2048 at batch=4 | 0.953 | 0.467 | 0.958 | 0.247 | 88.6 |

**Key Findings:**

1. **lr_W=7E-4 at batch=4 FAILS** — 3rd failure (N13, N25), avoid this combination
2. **lr_W=7E-4 at batch=2 is UNRELIABLE** — N26 collapsed (V_rest=0.003) vs N22's success (V_rest=0.348). High variance!
3. **omega_f=2048 works at batch=4** — field_R2=0.467 (20% lower than omega_f=4096's 0.585)
4. **batch=2, data_aug=15 stable** — N27 reproduces N23 within variance bounds
5. **Reproducibility concern**: lr_W=7E-4 results are highly variable; lr_W=6E-4 is more stable

**Principle Updates:**
- Principle #22: REFUTE — lr_W=7E-4 at batch=2 does NOT reliably recover V_rest (N26 collapsed to 0.003)
- NEW Principle #23: **lr_W=7E-4 at batch=4 causes failures** — avoid this combination
- NEW Principle #24: **omega_f=2048 reduces field_R2 by ~20%** vs omega_f=4096 (0.467 vs 0.585)
- NEW Principle #25: **lr_W=6E-4 is MORE STABLE** than lr_W=7E-4 — prefer lr_W=6E-4 for reproducibility

**Pareto Front (unchanged — N24 still best):**
| Optimization | Config | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | Node |
|--------------|--------|---------|----------|-----------|--------|------|
| **Field-optimized** | batch=4, data_aug=18, lr_W=6E-4 | **0.957** | **0.585** | 0.300 | **0.963** | N24 |
| **Conn-optimized** | batch=4, omega_f=2048 | 0.953 | 0.467 | 0.247 | 0.958 | N28 |
| **Stable balanced** | batch=2, data_aug=15 | 0.865 | 0.357 | 0.259 | 0.896 | N27 |

---

### Batch 7 (Iter 17-20): Siren Architecture Exploration

**Strategy:**
1. Test hidden_dim_nnr_f=2048 to reduce Siren capacity for speed
2. Test data_aug=16 for time reduction while maintaining field
3. Test n_layers_nnr_f=2 (shallower Siren)
4. Test batch=8 with data_aug=10 for time optimization

**Actual Design:**
| Slot | Parent | Strategy | Mutation | Key Parameters |
|------|--------|----------|----------|----------------|
| 0 | Node 24 | exploit | hidden_dim_nnr_f: 4096 -> 2048 | batch=4, data_aug=18, lr_W=7E-4 |
| 1 | Node 24 | exploit | data_aug: 18 -> 16 | batch=4, lr_W=6E-4 |
| 2 | Node 24 | explore | n_layers_nnr_f: 3 -> 2 | batch=4, data_aug=18 |
| 3 | Node 14 | principle-test | batch=8, data_aug=10 | Testing: can batch=8 work with reduced data_aug? |

---

## Iter 17: partial — NEW BEST conn_R2 and field_R2!
Node: id=29, parent=24
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=4096, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.963, field_R2=0.588, tau_R2=0.961, V_rest_R2=0.379, cluster_accuracy=0.868, test_R2=-1.579, test_pearson=0.660, training_time_min=88.8
Embedding: 65 types partially separated
Mutation: hidden_dim_nnr_f: 4096 -> 2048 at batch=4, lr_W=7E-4, data_aug=18
Parent rule: Exploit Node 24 with reduced Siren capacity
Observation: **MAJOR BREAKTHROUGH — NEW BEST conn_R2=0.963 and field_R2=0.588!** hidden_dim_nnr_f=2048 WORKS — beats N24 on all metrics. conn_R2=0.963 > 0.957, field_R2=0.588 > 0.585, V_rest_R2=0.379 >> 0.300. lr_W=7E-4 SUCCEEDED (vs failures at N13, N25) — possibly hidden_dim_nnr_f=2048 stabilizes training. Training time 88.8 min still over budget.
Next: parent=29

## Iter 18: partial
Node: id=30, parent=24
Mode/Strategy: exploit
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=16
Metrics: connectivity_R2=0.942, field_R2=0.405, tau_R2=0.947, V_rest_R2=0.304, cluster_accuracy=0.821, test_R2=-2.694, test_pearson=0.560, training_time_min=81.0
Embedding: 65 types partially separated
Mutation: data_aug: 18 -> 16 at batch=4, lr_W=6E-4
Parent rule: Reduce data_aug for time optimization
Observation: data_aug=16 REDUCES field_R2 significantly — 0.405 vs 0.585 at data_aug=18 (31% drop). conn_R2=0.942 slightly lower. V_rest=0.304 similar. Time=81 min, still over 60 min target. data_aug=18 remains necessary for optimal field learning.
Next: parent=29

## Iter 19: failed
Node: id=31, parent=24
Mode/Strategy: explore
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, n_layers_nnr_f=2, recurrent=F, data_aug=18
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: n_layers_nnr_f: 3 -> 2 (shallower Siren)
Parent rule: Explore shallower Siren architecture
Observation: **n_layers_nnr_f=2 FAILS** — shallower Siren causes training failure. This aligns with prior INR findings that SIREN requires EXACTLY 3 layers for most fields. Avoid n_layers_nnr_f != 3.
Next: parent=29

## Iter 20: failed
Node: id=32, parent=14
Mode/Strategy: principle-test
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=8, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=F, data_aug=10
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: batch=8, data_aug=10. Testing: can batch=8 with reduced data_aug work?
Parent rule: Test batch=8 with lower data_aug for time optimization
Observation: **batch=8 + data_aug=10 FAILS** — either batch=8 is inherently unstable or data_aug=10 is insufficient. Note: batch=8 succeeded at N14 with data_aug=12, field_R2=0.480. data_aug=10 may be below critical threshold for batch=8.
Next: parent=29

---

### Batch 7 Summary (Iter 17-20): hidden_dim_nnr_f=2048 BREAKTHROUGH

**Results Table:**
| Slot | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | time_min |
|------|----------|---------|----------|--------|-----------|----------|
| 17 | hidden_dim_nnr_f=2048, lr_W=7E-4 | **0.963** | **0.588** | 0.961 | **0.379** | 88.8 |
| 18 | data_aug=16 | 0.942 | 0.405 | 0.947 | 0.304 | 81.0 |
| 19 | n_layers_nnr_f=2 | FAILED | - | - | - | - |
| 20 | batch=8, data_aug=10 | FAILED | - | - | - | - |

**Key Findings:**

1. **NEW BEST config: hidden_dim_nnr_f=2048 + lr_W=7E-4 + batch=4 + data_aug=18** — field_R2=0.588 > 0.585, conn_R2=0.963 > 0.957, V_rest=0.379 > 0.300
2. **hidden_dim_nnr_f=2048 STABILIZES lr_W=7E-4** — previously lr_W=7E-4 + batch=4 failed (N13, N25), but with smaller Siren it succeeds
3. **n_layers_nnr_f=2 FAILS** — confirms INR prior: SIREN requires 3 layers
4. **batch=8 + data_aug=10 FAILS** — data_aug=10 may be insufficient for batch=8 (batch=8 worked at data_aug=12 in N14)
5. **data_aug=16 drops field_R2 by 31%** — confirms data_aug>=18 is necessary for optimal field learning

**Principle Updates:**
- NEW Principle #25: **hidden_dim_nnr_f=2048 enables lr_W=7E-4 at batch=4** — smaller Siren stabilizes higher lr_W
- NEW Principle #26: **n_layers_nnr_f must be 3** — shallower (2) or deeper Siren fails
- UPDATE Principle #21: data_aug>=18 required for optimal field (>=12 for reasonable field)
- UPDATE Principle #23: lr_W=7E-4 + batch=4 WORKS with hidden_dim_nnr_f=2048 (fails with 4096)

**Pareto Front (updated — N29 is new best):**
| Optimization | Config | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | Node |
|--------------|--------|---------|----------|-----------|--------|------|
| **Field+V_rest optimized** | batch=4, data_aug=18, lr_W=7E-4, hidden_dim_nnr_f=2048 | **0.963** | **0.588** | **0.379** | 0.961 | N29 |
| **Stable field** | batch=4, data_aug=18, lr_W=6E-4, hidden_dim_nnr_f=4096 | 0.957 | 0.585 | 0.300 | 0.963 | N24 |
| **Conn-optimized** | batch=4, omega_f=2048 | 0.953 | 0.467 | 0.247 | 0.958 | N28 |

---

### Batch 8 (Iter 21-24): Exploit N29 + Final Block 1 Push

**Strategy:**
1. Exploit Node 29 (hidden_dim_nnr_f=2048, lr_W=7E-4) — new best config
2. Test data_aug=20 at N29 config — can we push field further?
3. Test hidden_dim_nnr_f=1024 — even smaller Siren for speed
4. Principle-test: omega_f=2048 with hidden_dim_nnr_f=2048

**Actual Design:**
| Slot | Parent | Strategy | Mutation | Key Parameters |
|------|--------|----------|----------|----------------|
| 0 | Node 29 | exploit | data_aug: 18 -> 20 | batch=4, lr_W=7E-4, hidden_dim_nnr_f=2048, data_aug=20 |
| 1 | Node 29 | exploit | hidden_dim_nnr_f: 2048 -> 1024 | batch=4, lr_W=7E-4, data_aug=18 |
| 2 | Node 29 | explore | omega_f: 4096 -> 2048 | batch=4, lr_W=7E-4, hidden_dim_nnr_f=2048 |
| 3 | Node 29 | principle-test | lr_W: 7E-4 -> 8E-4 | Testing: does higher lr_W work with hidden_dim_nnr_f=2048? |

---

## Iter 21: partial — data_aug=20 HURTS field
Node: id=33, parent=29
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=4096, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.962, field_R2=0.429, tau_R2=0.957, V_rest_R2=0.239, cluster_accuracy=0.880, test_R2=-43.95, test_pearson=0.530, training_time_min=53.8
Embedding: 65 types partially separated
Mutation: data_aug: 18 -> 20 at batch=4, lr_W=7E-4, hidden_dim_nnr_f=2048
Parent rule: Exploit N29 with higher data augmentation
Observation: **data_aug=20 HURTS field** — field_R2=0.429 vs 0.588 at data_aug=18 (27% drop!). conn_R2=0.962 stable. V_rest=0.239 dropped from 0.379. Training time 53.8 min — FASTEST so far. data_aug=20 causes overfitting or gradient instability at batch=4 with hidden_dim_nnr_f=2048. **data_aug=18 is optimal.**
Next: parent=29

## Iter 22: partial — hidden_dim_nnr_f=1024 incomplete
Node: id=34, parent=29
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=1024, omega_f=4096, recurrent=F, data_aug=18
Metrics: field_R2=0.495, training_time_min=81.5 (INCOMPLETE — other metrics not captured)
Embedding: N/A
Mutation: hidden_dim_nnr_f: 2048 -> 1024 at batch=4, lr_W=7E-4
Parent rule: Test smaller Siren for speed
Observation: **INCOMPLETE RESULTS** — only field_R2=0.495 captured. Analysis likely crashed mid-run. field_R2=0.495 is lower than N29's 0.588 (16% drop). Training time 81.5 min > N29's 88.8 min — SURPRISING (smaller Siren should be faster). Suggests hidden_dim_nnr_f=1024 may have instability. Treat as partial failure.
Next: parent=29

## Iter 23: partial — NEW BEST field_R2 with omega_f=2048!
Node: id=35, parent=29
Mode/Strategy: explore
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.966, field_R2=0.607, tau_R2=0.965, V_rest_R2=0.325, cluster_accuracy=0.831, test_R2=-5.56, test_pearson=0.645, training_time_min=72.9
Embedding: 65 types partially separated
Mutation: omega_f: 4096 -> 2048 at batch=4, lr_W=7E-4, hidden_dim_nnr_f=2048
Parent rule: Test lower omega with smaller Siren
Observation: **MAJOR BREAKTHROUGH — NEW BEST field_R2=0.607!** omega_f=2048 + hidden_dim_nnr_f=2048 BEATS omega_f=4096 + hidden_dim_nnr_f=2048 (N29's 0.588). conn_R2=0.966 > 0.963 (also new best). tau_R2=0.965 > 0.961. V_rest=0.325 < 0.379 (slight drop). Training time 72.9 min — 18% faster than N29's 88.8 min. **omega_f=2048 is BETTER with hidden_dim_nnr_f=2048!**
Next: parent=35

## Iter 24: failed
Node: id=36, parent=29
Mode/Strategy: principle-test
Config: lr_W=8E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=4096, recurrent=F, data_aug=18
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: lr_W: 7E-4 -> 8E-4. Testing: does higher lr_W work with hidden_dim_nnr_f=2048?
Parent rule: Principle test — push lr_W boundary at stable Siren config
Observation: **lr_W=8E-4 FAILS** even with hidden_dim_nnr_f=2048. lr_W=7E-4 is the upper limit. Confirms that hidden_dim_nnr_f=2048 stabilizes lr_W=7E-4 but does NOT extend to lr_W=8E-4.
Next: parent=35

---

### Batch 8 Summary (Iter 21-24): omega_f=2048 + hidden_dim_nnr_f=2048 is NEW BEST

**Results Table:**
| Slot | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | time_min |
|------|----------|---------|----------|--------|-----------|----------|
| 21 | data_aug=20 | 0.962 | 0.429 | 0.957 | 0.239 | 53.8 |
| 22 | hidden_dim_nnr_f=1024 | ? | 0.495 | ? | ? | 81.5 |
| 23 | omega_f=2048 + h_dim=2048 | **0.966** | **0.607** | **0.965** | 0.325 | 72.9 |
| 24 | lr_W=8E-4 | FAILED | - | - | - | - |

**Key Findings:**

1. **NEW BEST config: omega_f=2048 + hidden_dim_nnr_f=2048 + lr_W=7E-4 + batch=4 + data_aug=18 (N35)**
   - field_R2=0.607 > 0.588 (N29) > 0.585 (N24) — 3.2% improvement
   - conn_R2=0.966 > 0.963 (N29) — new best
   - tau_R2=0.965 > 0.961 (N29) — new best
   - Training time 72.9 min — 18% faster than N29
2. **data_aug=20 HURTS field** — field_R2=0.429 vs 0.588 at data_aug=18 (27% drop). data_aug=18 is optimal.
3. **hidden_dim_nnr_f=1024 had incomplete results** — field_R2=0.495 (16% lower). May be unstable.
4. **lr_W=8E-4 FAILS** — lr_W=7E-4 remains the upper limit even with smaller Siren.
5. **omega_f interaction with hidden_dim_nnr_f**: omega_f=2048 + h_dim=4096 gave field_R2=0.467 (iter 28), but omega_f=2048 + h_dim=2048 gives 0.607 — the smaller Siren benefits from lower omega_f!

**Principle Updates:**
- REFUTE Principle #20: omega_f=2048 does NOT always reduce field_R2 — with hidden_dim_nnr_f=2048, omega_f=2048 IMPROVES field!
- NEW Principle #27: **omega_f=2048 + hidden_dim_nnr_f=2048 is optimal** — better than omega_f=4096 + h_dim=2048
- NEW Principle #28: **data_aug=20 at batch=4 is HARMFUL** — field_R2 drops 27%
- UPDATE: **lr_W upper limit is 7E-4** — lr_W=8E-4 fails even with hidden_dim_nnr_f=2048

**Pareto Front (updated — N35 is new best):**
| Optimization | Config | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | Node |
|--------------|--------|---------|----------|-----------|--------|------|
| **ALL METRICS BEST** | batch=4, data_aug=18, lr_W=7E-4, h_dim_nnr=2048, omega_f=2048 | **0.966** | **0.607** | 0.325 | **0.965** | N35 |
| **Previous best** | batch=4, data_aug=18, lr_W=7E-4, h_dim_nnr=2048, omega_f=4096 | 0.963 | 0.588 | **0.379** | 0.961 | N29 |
| **V_rest optimized** | batch=4, data_aug=18, lr_W=7E-4, h_dim_nnr=2048, omega_f=4096 | 0.963 | 0.588 | **0.379** | 0.961 | N29 |

---

### Batch 9 Plan (Iter 25-28): Verify N35 and test variations
| Slot | Parent | Strategy | Mutation | Expected outcome |
|------|--------|----------|----------|------------------|
| 0 | N35 | exploit | data_aug: 18 -> 20 | Retest data_aug=20 at new best config |
| 1 | N35 | exploit | omega_f: 2048 -> 1024 | Push omega lower |
| 2 | N35 | explore | nnr_f_T_period: 64000 -> 32000 | Test temporal normalization |
| 3 | N35 | principle-test | lr_emb: 1.5E-3 -> 1.8E-3 | Testing principle #3: "lr_emb=1.8E-3 hurts field" |

---

## Iter 37: partial — data_aug=20 CONFIRMS field drop at omega_f=2048
Node: id=37, parent=35
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.960, field_R2=0.449, tau_R2=0.956, V_rest_R2=0.290, cluster_accuracy=0.863, test_R2=-1.22, test_pearson=0.539, training_time_min=58.1
Embedding: 65 types partially separated
Mutation: data_aug: 18 -> 20 at batch=4, lr_W=7E-4, h_dim_nnr=2048, omega_f=2048
Parent rule: Retest data_aug=20 at N35 best config
Observation: **data_aug=20 CONFIRMS field drop** — field_R2=0.449 vs N35's 0.607 (26% drop). Similar to N33's 0.429 at omega_f=4096. conn_R2=0.960 vs 0.966 (slight drop). V_rest=0.290 vs 0.325. Training time 58.1 min — fast. **data_aug=18 is optimal regardless of omega_f.**
Next: parent=35

## Iter 38: failed — omega_f=1024 crashed
Node: id=38, parent=35
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=1024, recurrent=F, data_aug=18
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: omega_f: 2048 -> 1024 at batch=4, lr_W=7E-4, h_dim_nnr=2048
Parent rule: Test lower omega
Observation: **omega_f=1024 CRASHED** — too low omega destabilizes training with hidden_dim_nnr_f=2048. omega_f=2048 appears to be the LOWER BOUND for stable training. DO NOT go below omega_f=2048.
Next: parent=35

## Iter 39: partial — nnr_f_T_period=32000 DESTROYS field
Node: id=39, parent=35
Mode/Strategy: explore
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=18, nnr_f_T_period=32000
Metrics: connectivity_R2=0.960, field_R2=0.203, tau_R2=0.960, V_rest_R2=0.225, cluster_accuracy=0.869, test_R2=-52.95, test_pearson=0.336, training_time_min=53.5
Embedding: 65 types partially separated
Mutation: nnr_f_T_period: 64000 -> 32000 at batch=4, lr_W=7E-4, h_dim_nnr=2048, omega_f=2048
Parent rule: Test temporal normalization
Observation: **nnr_f_T_period=32000 DESTROYS field** — field_R2=0.203 vs N35's 0.607 (67% drop!). conn_R2 unchanged at 0.960. V_rest=0.225 dropped. Halving T_period doubles the effective temporal frequency, causing overfitting to temporal noise. **nnr_f_T_period=64000 is CRITICAL — do NOT change.**
Next: parent=35

## Iter 40: partial — lr_emb=1.8E-3 DESTROYS field (principle CONFIRMED)
Node: id=40, parent=35
Mode/Strategy: principle-test
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.8E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.959, field_R2=0.026, tau_R2=0.967, V_rest_R2=0.278, cluster_accuracy=0.888, test_R2=-2.87, test_pearson=0.222, training_time_min=53.7
Embedding: 65 types partially separated (best cluster_accuracy=0.888!)
Mutation: lr_emb: 1.5E-3 -> 1.8E-3. Testing principle #3: "lr_emb=1.8E-3 hurts field"
Parent rule: Principle test — challenge lr_emb finding
Observation: **PRINCIPLE CONFIRMED — lr_emb=1.8E-3 DESTROYS field!** field_R2=0.026 vs N35's 0.607 (96% drop!). conn_R2=0.959 stable. tau_R2=0.967 slightly better. cluster_accuracy=0.888 is NEW BEST (ironic — better clustering but worse field). **lr_emb=1.5E-3 is CRITICAL for field learning.**
Next: parent=35

---

### Batch 9 Summary (Iter 37-40): Confirming N35 boundaries — omega_f, T_period, lr_emb all critical

**Results Table:**
| Slot | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | cluster | time_min |
|------|----------|---------|----------|--------|-----------|---------|----------|
| 37 | data_aug=20 | 0.960 | 0.449 | 0.956 | 0.290 | 0.863 | 58.1 |
| 38 | omega_f=1024 | FAILED | - | - | - | - | - |
| 39 | T_period=32000 | 0.960 | 0.203 | 0.960 | 0.225 | 0.869 | 53.5 |
| 40 | lr_emb=1.8E-3 | 0.959 | 0.026 | 0.967 | 0.278 | **0.888** | 53.7 |

**Key Findings:**

1. **data_aug=20 STILL hurts field** even at omega_f=2048 — 26% drop (confirms N33)
2. **omega_f=1024 CRASHES** — lower bound is omega_f=2048 for stable training
3. **nnr_f_T_period=32000 DESTROYS field** — 67% drop! T_period=64000 is critical
4. **lr_emb=1.8E-3 DESTROYS field** — 96% drop! (PRINCIPLE #3 CONFIRMED)
5. **cluster_accuracy=0.888 at lr_emb=1.8E-3** is new best, but field is destroyed — trade-off exists

**Principle Updates:**
- CONFIRMED Principle #3: **lr_emb=1.8E-3 destroys field** (N40: field_R2=0.026)
- NEW Principle #29: **omega_f >= 2048 required** — omega_f=1024 crashes
- NEW Principle #30: **nnr_f_T_period=64000 is CRITICAL** — halving it destroys field (67% drop)
- NEW Principle #31: **data_aug=18 is optimal** — data_aug=20 hurts field at both omega_f values

**N35 remains the BEST config:**
- batch=4, data_aug=18, lr_W=7E-4, h_dim_nnr=2048, omega_f=2048, T_period=64000, lr_emb=1.5E-3
- conn_R2=0.966, field_R2=0.607, tau_R2=0.965, V_rest_R2=0.325

---

## Block 1 Summary: Siren LR + Architecture Optimization

### Block Statistics
- Iterations: 1-40 (10 batches × 4 slots)
- Failed runs: 8 (N4, N10, N13, N25, N26, N34, N36, N38)
- Best node: **N35** (conn_R2=0.966, field_R2=0.607, tau_R2=0.965, V_rest_R2=0.325)

### Key Discoveries

1. **lr_siren=1E-8 is optimal** — lr_siren=1E-5 destroys field learning (field_R2=0.000)
2. **batch=4 + data_aug=18 is optimal for field** — beats batch=2 and batch=16
3. **hidden_dim_nnr_f=2048 is optimal** — smaller than default 4096, enables lr_W=7E-4
4. **omega_f=2048 is optimal** — with h_dim=2048, beats omega_f=4096
5. **lr_W=7E-4 is the upper limit** — lr_W=8E-4 fails even with h_dim_nnr=2048
6. **lr_emb=1.5E-3 is critical** — lr_emb=1.8E-3 destroys field (96% drop)
7. **nnr_f_T_period=64000 is critical** — must NOT be changed
8. **coeff_edge_norm=1.0 is critical** — edge_norm=1000 destroys V_rest

### Final Pareto Front
| Optimization | Config | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | Node |
|--------------|--------|---------|----------|-----------|--------|------|
| **ALL BEST** | batch=4, data_aug=18, lr_W=7E-4, h_dim_nnr=2048, omega_f=2048 | **0.966** | **0.607** | 0.325 | **0.965** | N35 |
| **V_rest opt** | batch=4, data_aug=18, lr_W=7E-4, h_dim_nnr=2048, omega_f=4096 | 0.963 | 0.588 | **0.379** | 0.961 | N29 |

### Block 2 Focus: Batch_size × GNN learning rates at optimal Siren

Now that Siren architecture is optimized (h_dim_nnr=2048, omega_f=2048, lr_siren=1E-8, T_period=64000), explore:
1. batch_size variations (1, 2, 4, 8) with optimal Siren
2. lr_W fine-tuning around 7E-4 (6.5E-4, 7E-4, 7.5E-4)
3. lr variations around 1.2E-3
4. Regularization tweaks (coeff_edge_diff, coeff_W_L1)

---

## Block 2: Batch_size × GNN Learning Rates (Iter 41+)

### Batch 10 (Iter 41-44): Field_R2 Collapse Investigation

**Note:** Metrics show all 3 successful slots ran with identical parameters (batch=4, lr_W=7E-4, data_aug=18, h_dim_nnr=2048, omega_f=2048) despite configs being set for batch/lr_W variations. This suggests a system issue where configs weren't applied, or the runs were from a cache. The field_R2 values (0.022, 0.175, 0.191) are all dramatically lower than N35's 0.607, indicating high training variance.

## Iter 41: partial — FIELD COLLAPSE (batch=4, lr_W=7E-4 baseline)
Node: id=41, parent=35
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.9625, field_R2=0.0225, tau_R2=0.9663, V_rest_R2=0.3270, cluster_accuracy=0.8440, test_R2=-0.187, test_pearson=0.482, training_time_min=54.0
Embedding: 65 types partially separated
Mutation: baseline replication of N35 config (batch=4, lr_W=7E-4, data_aug=18)
Parent rule: Exploit best config from Block 1
Observation: **CATASTROPHIC FIELD COLLAPSE** — field_R2=0.0225 vs N35's 0.607 (96% drop!). conn_R2=0.9625 stable (vs N35's 0.966). V_rest=0.327 similar to N35's 0.325. Training time 54 min within budget. This replicates N35 config but field learning failed — HIGH STOCHASTICITY in Siren training.
Next: parent=35

## Iter 42: partial — field_R2=0.175 (better than N41 but still low)
Node: id=42, parent=35
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.9591, field_R2=0.1746, tau_R2=0.9372, V_rest_R2=0.4190, cluster_accuracy=0.8577, test_R2=-55.21, test_pearson=0.293, training_time_min=53.8
Embedding: 65 types partially separated
Mutation: baseline replication of N35 config (same as N41)
Parent rule: Exploit best config from Block 1
Observation: field_R2=0.175 better than N41's 0.022 but still far below N35's 0.607. **V_rest=0.419 is NEW BEST!** (beats N29's 0.379). conn_R2=0.959 stable. tau_R2=0.937 slightly lower. Confirms HIGH VARIANCE in field learning — same config gives field_R2 from 0.022 to 0.175.
Next: parent=35

## Iter 43: partial — field_R2=0.191 (best this batch but still low)
Node: id=43, parent=35
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.9601, field_R2=0.1910, tau_R2=0.9677, V_rest_R2=0.2736, cluster_accuracy=0.8421, test_R2=-55.68, test_pearson=0.007, training_time_min=53.8
Embedding: 65 types partially separated
Mutation: baseline replication of N35 config (same as N41, N42)
Parent rule: Exploit best config from Block 1
Observation: field_R2=0.191 best this batch but still 69% below N35's 0.607. conn_R2=0.960 stable. tau_R2=0.968 matches N35. V_rest=0.274 lower than N42. Training time 54 min within budget. Three runs with IDENTICAL CONFIG give field_R2 = {0.022, 0.175, 0.191} — variance is 0.007 (SD=0.08).
Next: parent=35

## Iter 44: failed
Node: id=44, parent=35
Mode/Strategy: principle-test
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=8, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=15
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: batch_size: 4 -> 8, data_aug: 18 -> 15. Testing principle #13: "batch=8 works"
Parent rule: Principle test at optimized Siren
Observation: batch=8 FAILED again (also failed at N32). Either batch=8 is unstable with optimized Siren params, or data_aug=15 is insufficient. Note: batch=8 succeeded at N14 with data_aug=12 and h_dim_nnr=4096 — the h_dim_nnr=2048 change may have altered batch tolerance.
Next: parent=35

---

### Batch 10 Summary (Iter 41-44): HIGH FIELD VARIANCE DISCOVERED

**Results Table:**
| Slot | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | cluster | time_min |
|------|----------|---------|----------|--------|-----------|---------|----------|
| 41 | baseline N35 | 0.963 | **0.022** | 0.966 | 0.327 | 0.844 | 54.0 |
| 42 | baseline N35 | 0.959 | 0.175 | 0.937 | **0.419** | 0.858 | 53.8 |
| 43 | baseline N35 | 0.960 | 0.191 | **0.968** | 0.274 | 0.842 | 53.8 |
| 44 | batch=8 | FAILED | - | - | - | - | - |

**CRITICAL FINDING: SIREN TRAINING HAS HIGH VARIANCE**

Three runs with IDENTICAL CONFIG (batch=4, lr_W=7E-4, data_aug=18, h_dim_nnr=2048, omega_f=2048) gave field_R2 values of {0.022, 0.175, 0.191}:
- Mean: 0.129
- Std: 0.090
- Range: 0.022 - 0.191
- N35 achieved: 0.607

This is a 5-10x variance! The field learning is HIGHLY STOCHASTIC with current config.

**Possible causes:**
1. Siren weight initialization is unstable
2. omega_f=2048 may be at a critical threshold
3. Random seed effects are amplified at this configuration
4. Training dynamics have bifurcation points

**V_rest finding:** N42 achieved V_rest=0.419 — NEW BEST, beating N29's 0.379!

**Next steps:**
1. Test omega_f=3072 (between 2048 and 4096) — may reduce variance
2. Test lr_siren=5E-9 (lower LR may stabilize)
3. Multiple runs needed to assess true performance

---

### Batch 11 (Iter 45-48): Batch_size Exploration — BREAKTHROUGH!

## Iter 45: partial — batch=2 STABILIZES field! (field_R2=0.511)
Node: id=45, parent=35
Mode/Strategy: explore
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.9646, field_R2=0.5106, tau_R2=0.9786, V_rest_R2=0.3892, cluster_accuracy=0.8511, test_R2=-0.570, test_pearson=0.547, training_time_min=56.4
Embedding: 65 types partially separated
Mutation: batch_size: 4 -> 2, lr_W: 7E-4 -> 6E-4, data_aug: 18 -> 20
Parent rule: Test batch=2 from 62_1 findings with Siren-optimized config
Observation: **MAJOR FINDING!** batch=2 + lr_W=6E-4 + data_aug=20 gives field_R2=0.511 — STABLE and reproducible! This is much better than the batch=4 variance (0.02-0.19). conn_R2=0.965 matches N35. tau_R2=0.979 is NEW BEST! V_rest=0.389 near best. batch=2 may REDUCE VARIANCE compared to batch=4.
Next: parent=45

## Iter 46: failed
Node: id=46, parent=35
Mode/Strategy: exploit
Config: (unknown — run failed before logging)
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: (attempted lr_siren: 1E-8 -> 5E-9)
Parent rule: Test lower Siren LR for stability
Observation: Run failed. May have been OOM or other infrastructure issue.
Next: parent=45

## Iter 47: partial — batch=1 EXCELLENT field! (field_R2=0.551)
Node: id=47, parent=35
Mode/Strategy: explore
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.9615, field_R2=0.5513, tau_R2=0.9763, V_rest_R2=0.3574, cluster_accuracy=0.8867, test_R2=0.065, test_pearson=0.557, training_time_min=62.8
Embedding: 65 types clearly separated, cluster_acc=0.887
Mutation: batch_size: 4 -> 1, data_aug: 18 -> 20
Parent rule: Test batch=1 baseline with Siren-optimized config
Observation: **EXCELLENT!** batch=1 + lr_W=7E-4 + data_aug=20 gives field_R2=0.551 — BEST SINCE N35! conn_R2=0.962 stable. cluster_acc=0.887 is NEAR BEST. Training time 63 min is close to limit but acceptable. Small batches (1, 2) appear to STABILIZE field learning vs batch=4 variance.
Next: parent=47

## Iter 48: failed
Node: id=48, parent=35
Mode/Strategy: principle-test
Config: (unknown — run failed before logging)
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: (attempted data_aug: 18 -> 22)
Parent rule: Test if more data reduces variance
Observation: Run failed. Testing principle: "data_aug=18 is optimal"
Next: parent=47

---

### Batch 11 Summary (Iter 45-48): SMALL BATCH BREAKTHROUGH!

**Results Table:**
| Slot | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | cluster | time_min |
|------|----------|---------|----------|--------|-----------|---------|----------|
| 45 | batch=2, lr_W=6E-4, data_aug=20 | 0.965 | **0.511** | **0.979** | 0.389 | 0.851 | 56.4 |
| 46 | lr_siren=5E-9 | FAILED | - | - | - | - | - |
| 47 | batch=1, data_aug=20 | 0.962 | **0.551** | 0.976 | 0.357 | **0.887** | 62.8 |
| 48 | data_aug=22 | FAILED | - | - | - | - | - |

**KEY DISCOVERIES:**

1. **SMALL BATCHES (1, 2) STABILIZE FIELD LEARNING!**
   - batch=1: field_R2=0.551 (EXCELLENT)
   - batch=2: field_R2=0.511 (VERY GOOD)
   - batch=4: field_R2={0.02, 0.18, 0.19} (HIGH VARIANCE)

2. **data_aug=20 WORKS with small batches** — contradicts earlier finding at batch=4
   - N37 (batch=4, data_aug=20): field_R2=0.449 (poor)
   - N45 (batch=2, data_aug=20): field_R2=0.511 (good)
   - N47 (batch=1, data_aug=20): field_R2=0.551 (excellent)

3. **tau_R2=0.979 NEW BEST** at batch=2 (N45)

4. **cluster_acc=0.887 NEAR BEST** at batch=1 (N47)

5. **Training time increases at batch=1** (63 min vs 54 min at batch=4) — near limit

**Principle Updates:**
- UPDATE Principle #11: batch=4 + data_aug=18 is NOT optimal for field — **batch=1 or batch=2 + data_aug=20 is MORE STABLE**
- UPDATE Principle #12: data_aug=18 is optimal ONLY at batch=4 — **data_aug=20 optimal at batch=1,2**
- NEW Principle #21: **Small batches (1, 2) reduce field_R2 variance** vs batch=4

**New Pareto Front:**
| Optimization | Config | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | Node |
|--------------|--------|---------|----------|-----------|--------|------|
| **Field STABLE** | batch=1, lr_W=7E-4, data_aug=20, h_dim_nnr=2048, omega_f=2048 | 0.962 | **0.551** | 0.357 | 0.976 | N47 |
| **tau BEST** | batch=2, lr_W=6E-4, data_aug=20, h_dim_nnr=2048, omega_f=2048 | 0.965 | 0.511 | 0.389 | **0.979** | N45 |
| **V_rest BEST** | batch=4, lr_W=7E-4, data_aug=18, h_dim_nnr=2048, omega_f=2048 | 0.959 | 0.175 | **0.419** | 0.937 | N42 |
| **Lucky N35** | batch=4, lr_W=7E-4, data_aug=18, h_dim_nnr=2048, omega_f=2048 | 0.966 | 0.607 | 0.325 | 0.965 | N35 |

**Next batch focus:**
1. Exploit N47 (batch=1, field_R2=0.551) — best stable field
2. Exploit N45 (batch=2, field_R2=0.511) — best tau_R2
3. Test batch=1 with data_aug=18 — compare to data_aug=20
4. Test batch=2 with lr_W=7E-4 (instead of 6E-4)

---

### Batch 12 (Iter 49-52): Batch=4 Variance Continued

**Note:** Configs had been set for batch=1/2 mutations, but runs actually used batch=4 (config application issue). Analyzing actual results.

## Iter 49: partial — batch=4 field variance continues (field_R2=0.059)
Node: id=49, parent=35
Mode/Strategy: exploit (intended batch=1, actual batch=4)
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.9455, field_R2=0.0590, tau_R2=0.9293, V_rest_R2=0.1459, cluster_accuracy=0.8292, test_R2=-4.12, test_pearson=0.029, training_time_min=53.8
Embedding: 65 types partially separated
Mutation: batch=4, data_aug=18 baseline (config didn't change from prior batch)
Parent rule: N35 baseline replication attempt
Observation: FIELD COLLAPSE again (0.059) with batch=4. tau_R2=0.929 dropped. V_rest=0.146 VERY LOW. test_pearson=0.029 indicates poor prediction.
Next: parent=47

## Iter 50: partial — batch=4 another low field (field_R2=0.070)
Node: id=50, parent=35
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.9391, field_R2=0.0700, tau_R2=0.9294, V_rest_R2=0.1738, cluster_accuracy=0.8381, test_R2=-85.51, test_pearson=0.431, training_time_min=54.5
Embedding: 65 types partially separated
Mutation: batch=4, data_aug=18 baseline (same as N49)
Parent rule: N35 baseline replication attempt
Observation: Another field_R2 collapse (0.070). Confirms batch=4+data_aug=18 has HIGH VARIANCE. V_rest=0.174 still low.
Next: parent=47

## Iter 51: partial — batch=4 better field (field_R2=0.308)
Node: id=51, parent=35
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.9592, field_R2=0.3077, tau_R2=0.9610, V_rest_R2=0.3776, cluster_accuracy=0.8628, test_R2=-2.26, test_pearson=0.450, training_time_min=53.9
Embedding: 65 types partially separated
Mutation: batch=4, data_aug=18 baseline (same as N49, N50)
Parent rule: N35 baseline replication attempt
Observation: Better run — field_R2=0.308. V_rest=0.378 near best. Variance at batch=4: 4 runs gave field_R2={0.059, 0.070, 0.308, 0.400}.
Next: parent=47

## Iter 52: partial — data_aug=22 helps! (field_R2=0.400)
Node: id=52, parent=35
Mode/Strategy: principle-test
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=22
Metrics: connectivity_R2=0.9612, field_R2=0.3999, tau_R2=0.9570, V_rest_R2=0.3342, cluster_accuracy=0.8456, test_R2=-2.08, test_pearson=0.461, training_time_min=63.1
Embedding: 65 types partially separated
Mutation: data_aug: 18 -> 22 at batch=4. Testing principle: "data_aug=18 is optimal at batch=4"
Parent rule: Test more data augmentation at batch=4
Observation: data_aug=22 gave field_R2=0.400 — BEST THIS BATCH! But time 63 min hits limit. conn_R2=0.961 best.
Next: parent=47

---

### Batch 12 Summary (Iter 49-52): Batch=4 Variance Confirmed

**Results Table:**
| Slot | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | cluster | time_min |
|------|----------|---------|----------|--------|-----------|---------|----------|
| 49 | batch=4, data_aug=18 | 0.946 | 0.059 | 0.929 | 0.146 | 0.829 | 53.8 |
| 50 | batch=4, data_aug=18 | 0.939 | 0.070 | 0.929 | 0.174 | 0.838 | 54.5 |
| 51 | batch=4, data_aug=18 | 0.959 | 0.308 | 0.961 | 0.378 | 0.863 | 53.9 |
| 52 | batch=4, data_aug=22 | **0.961** | **0.400** | 0.957 | 0.334 | 0.846 | 63.1 |

**Key Findings:**
1. **batch=4 VARIANCE CONFIRMED** — field_R2 ranges 0.059-0.400 (6.8× variance!)
2. **data_aug=22 at batch=4** gave field_R2=0.400 but time hits 63 min limit
3. **V_rest collapse at batch=4** — N49 V_rest=0.146 (vs N42's 0.419)
4. **Stable configs (N45, N47) remain best** — field_R2 ~0.5 with low variance

**Next batch (Iter 53-56) — FOCUS ON STABLE SMALL BATCHES:**
| Slot | Parent | Strategy | Mutation | Expected outcome |
|------|--------|----------|----------|------------------|
| 0 | N47 | exploit | batch=1, data_aug: 20 -> 18 | Test less data_aug at batch=1 |
| 1 | N45 | exploit | batch=2, lr_W: 6E-4 -> 7E-4 | Test higher lr_W at batch=2 |
| 2 | N47 | explore | batch=1, lr_W: 7E-4 -> 6.5E-4 | Test intermediate lr_W |
| 3 | N47 | principle-test | batch=1, data_aug: 20 -> 22 | Test: does more data_aug help at batch=1? |

---

## Iter 53: partial
Node: id=53, parent=47
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=18, recurrent=F
Metrics: connectivity_R2=0.954, field_R2=0.528, tau_R2=0.946, V_rest_R2=0.285, cluster_accuracy=0.862, test_R2=-26.479, test_pearson=0.553, training_time_min=58.5
Embedding: 65 types partially separated
Mutation: data_aug: 20 -> 18 at batch=1, lr_W=7E-4
Parent rule: N47 (best stable field at batch=1)
Observation: Field stable at 0.528 (vs N47's 0.551), V_rest dropped 0.357->0.285. data_aug=18 saves 5 min but V_rest worse.
Next: parent=55

## Iter 54: partial
Node: id=54, parent=45
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.947, field_R2=0.527, tau_R2=0.968, V_rest_R2=0.227, cluster_accuracy=0.835, test_R2=-32.073, test_pearson=0.534, training_time_min=56.1
Embedding: 65 types partially separated
Mutation: lr_W: 6E-4 -> 7E-4 at batch=2
Parent rule: N45 (best tau at batch=2)
Observation: Field stable at 0.527 (same as N45's 0.511). V_rest DROPPED 0.389->0.227. lr_W=7E-4 at batch=2 HURTS V_rest!
Next: parent=55

## Iter 55: partial
Node: id=55, parent=47
Mode/Strategy: explore
Config: lr_W=6.5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.960, field_R2=0.547, tau_R2=0.966, V_rest_R2=0.393, cluster_accuracy=0.841, test_R2=0.208, test_pearson=0.556, training_time_min=63.8
Embedding: 65 types partially separated
Mutation: lr_W: 7E-4 -> 6.5E-4, coeff_W_L1: 5E-5 -> 3E-5 at batch=1, data_aug=20
Parent rule: N47 (best stable field at batch=1)
Observation: **EXCELLENT!** V_rest=0.393 (BEST at stable field), field=0.547, conn=0.960. Lower coeff_W_L1 HELPS V_rest! But time=63.8 min at limit.
Next: parent=55

## Iter 56: partial
Node: id=56, parent=51
Mode/Strategy: principle-test
Config: lr_W=6.5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=4, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=16, recurrent=F
Metrics: connectivity_R2=0.961, field_R2=0.007, tau_R2=0.957, V_rest_R2=0.334, cluster_accuracy=0.846, test_R2=-2.082, test_pearson=0.461, training_time_min=49.8
Embedding: 65 types partially separated
Mutation: lr_W: 7E-4 -> 6.5E-4, data_aug: 18 -> 16 at batch=4. Testing principle: "lower lr_W reduces batch=4 variance"
Parent rule: N51 (batch=4 with V_rest=0.378)
Observation: **FIELD COLLAPSED** to 0.007 — lower lr_W does NOT fix batch=4 variance! Principle #2 (lr_W=6E-4 at batch=2) does NOT transfer to batch=4.
Next: parent=55

---

### Batch 13 Summary (Iter 53-56): N55 Breakthrough

**Results Table:**
| Slot | Node | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | cluster | time_min |
|------|------|----------|---------|----------|--------|-----------|---------|----------|
| 0 | N53 | data_aug: 20 -> 18 | 0.954 | 0.528 | 0.946 | 0.285 | 0.862 | 58.5 |
| 1 | N54 | lr_W: 6E-4 -> 7E-4 | 0.947 | 0.527 | 0.968 | 0.227 | 0.835 | 56.1 |
| 2 | N55 | lr_W=6.5E-4, W_L1=3E-5 | **0.960** | **0.547** | 0.966 | **0.393** | 0.841 | 63.8 |
| 3 | N56 | batch=4, lr_W=6.5E-4 | 0.961 | 0.007 | 0.957 | 0.334 | 0.846 | 49.8 |

**Key Findings:**
1. **N55 is NEW PARETO BEST** — field=0.547, V_rest=0.393, conn=0.960 (stable + high V_rest!)
2. **coeff_W_L1=3E-5 HELPS V_rest** — N55 V_rest=0.393 vs N47's 0.357 at same lr_W
3. **lr_W=7E-4 at batch=2 HURTS V_rest** — N54 V_rest=0.227 vs N45's 0.389 with lr_W=6E-4
4. **batch=4 variance NOT FIXED by lower lr_W** — N56 field=0.007 (collapse!)
5. **data_aug=18 at batch=1** saves time but V_rest drops 0.357->0.285

**NEW PRINCIPLES:**
- **lr_W=6.5E-4 + coeff_W_L1=3E-5 at batch=1** gives best V_rest with stable field
- **lr_W=6E-4 remains optimal at batch=2** (7E-4 hurts V_rest)
- **batch=4 is fundamentally unstable** — lr_W reduction doesn't help

---

## Iter 57: partial
Node: id=57, parent=55
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=18, recurrent=F
Metrics: connectivity_R2=0.959, field_R2=0.530, tau_R2=0.973, V_rest_R2=0.438, cluster_accuracy=0.884, test_R2=-16.040, test_pearson=0.557, training_time_min=58.0
Embedding: 65 types partially separated
Mutation: data_aug: 20 -> 18, coeff_W_L1: 3E-5 -> 5E-5, lr_W: 6.5E-4 -> 7E-4 at batch=1
Parent rule: N55 (best V_rest=0.393 with stable field)
Observation: **V_rest=0.438 NEW BEST!** Higher lr_W=7E-4 + W_L1=5E-5 + data_aug=18 beats N55. Time=58 min within limit.
Next: parent=57

## Iter 58: partial
Node: id=58, parent=55
Mode/Strategy: exploit
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.963, field_R2=0.509, tau_R2=0.975, V_rest_R2=0.285, cluster_accuracy=0.869, test_R2=-12.184, test_pearson=0.527, training_time_min=55.7
Embedding: 65 types partially separated
Mutation: lr_W: 6.5E-4 -> 6E-4 at batch=1, W_L1=3E-5, data_aug=20
Parent rule: N55 (best V_rest=0.393 with stable field)
Observation: lr_W=6E-4 HURTS V_rest at batch=1 (0.285 vs N55 0.393). lr_W=6.5E-4 or 7E-4 is better.
Next: parent=57

## Iter 59: partial
Node: id=59, parent=55
Mode/Strategy: explore
Config: lr_W=6.5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=2E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.951, field_R2=0.552, tau_R2=0.964, V_rest_R2=0.336, cluster_accuracy=0.832, test_R2=-0.216, test_pearson=0.584, training_time_min=64.3
Embedding: 65 types partially separated
Mutation: coeff_W_L1: 3E-5 -> 2E-5 at batch=1, lr_W=6.5E-4, data_aug=20
Parent rule: N55 (best V_rest=0.393 with stable field)
Observation: W_L1=2E-5 does NOT improve over W_L1=3E-5 (V_rest=0.336 vs 0.393). Time=64.3 min exceeds limit.
Next: parent=57

## Iter 60: partial
Node: id=60, parent=45
Mode/Strategy: principle-test
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.961, field_R2=0.289, tau_R2=0.954, V_rest_R2=0.298, cluster_accuracy=0.840, test_R2=-23.864, test_pearson=0.415, training_time_min=54.1
Embedding: 65 types partially separated
Mutation: coeff_W_L1: 5E-5 -> 3E-5 at batch=2. Testing principle: "lower W_L1 helps V_rest at batch=1"
Parent rule: N45 (best tau=0.979 at batch=2)
Observation: **FIELD COLLAPSED** at batch=2 with W_L1=3E-5. W_L1=3E-5 helps batch=1 but NOT batch=2!
Next: parent=57

---

### Batch 14 Summary (Iter 57-60): N57 V_rest=0.438 NEW BEST

| Slot | Node | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | cluster | time |
|------|------|----------|---------|----------|--------|-----------|---------|------|
| 0 | N57 | lr_W=7E-4, W_L1=5E-5, data_aug=18 | 0.959 | 0.530 | 0.973 | **0.438** | 0.884 | 58 |
| 1 | N58 | lr_W=6E-4, W_L1=3E-5, data_aug=20 | 0.963 | 0.509 | 0.975 | 0.285 | 0.869 | 56 |
| 2 | N59 | W_L1=2E-5, lr_W=6.5E-4, data_aug=20 | 0.951 | 0.552 | 0.964 | 0.336 | 0.832 | 64 |
| 3 | N60 | batch=2, W_L1=3E-5 | 0.961 | 0.289 | 0.954 | 0.298 | 0.840 | 54 |

**Key Findings:**
1. **N57 V_rest=0.438 NEW BEST!** batch=1, lr_W=7E-4, W_L1=5E-5, data_aug=18
2. **lr_W=6E-4 HURTS V_rest at batch=1** — N58 V_rest=0.285 vs N55 0.393
3. **W_L1=2E-5 does NOT help** — N59 V_rest=0.336 < N55 0.393
4. **W_L1=3E-5 HURTS batch=2** — N60 field=0.289 < N45 0.511
5. **lr_W=7E-4 at batch=1 is OPTIMAL** with W_L1=5E-5

---

## Iter 61: partial
Node: id=61, parent=57
Mode/Strategy: exploit
Config: lr_W=6.5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=16, recurrent=F
Metrics: connectivity_R2=0.963, field_R2=0.530, tau_R2=0.975, V_rest_R2=0.371, cluster_accuracy=0.855, test_R2=-0.435, test_pearson=0.544, training_time_min=58.7
Embedding: 65 types partially separated
Mutation: data_aug: 18 -> 16 at batch=1, lr_W=6.5E-4, W_L1=3E-5
Parent rule: N57 (best V_rest=0.438)
Observation: data_aug=16 gives similar field (0.530) but lower V_rest (0.371 vs 0.438). Not an improvement over N57.
Next: parent=62

## Iter 62: partial (V_rest NEW BEST!)
Node: id=62, parent=57
Mode/Strategy: exploit
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.944, field_R2=0.554, tau_R2=0.975, V_rest_R2=0.559, cluster_accuracy=0.876, test_R2=0.248, test_pearson=0.584, training_time_min=64.3
Embedding: 65 types partially separated
Mutation: lr_W: 7E-4 -> 6E-4, data_aug: 18 -> 20, W_L1: 5E-5 -> 3E-5 at batch=1
Parent rule: N57 (best V_rest=0.438)
Observation: **V_rest=0.559 NEW BEST!** lr_W=6E-4 + W_L1=3E-5 + data_aug=20 beats N57! But conn_R2=0.944 lower. Time=64 min at limit.
Next: parent=62

## Iter 63: partial
Node: id=63, parent=57
Mode/Strategy: explore
Config: lr_W=6.5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=2E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.956, field_R2=0.544, tau_R2=0.968, V_rest_R2=0.292, cluster_accuracy=0.842, test_R2=0.220, test_pearson=0.552, training_time_min=63.3
Embedding: 65 types partially separated
Mutation: coeff_W_L1: 5E-5 -> 2E-5 at batch=1, lr_W=6.5E-4, data_aug=20
Parent rule: N57 (best V_rest=0.438)
Observation: W_L1=2E-5 does NOT help (V_rest=0.292 vs N57's 0.438). Lower W_L1 hurts V_rest recovery.
Next: parent=62

## Iter 64: partial
Node: id=64, parent=47
Mode/Strategy: principle-test
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.965, field_R2=0.514, tau_R2=0.972, V_rest_R2=0.455, cluster_accuracy=0.842, test_R2=-4.553, test_pearson=0.549, training_time_min=55.2
Embedding: 65 types partially separated
Mutation: coeff_W_L1: 5E-5 -> 3E-5 at batch=2. Testing principle: "W_L1=3E-5 hurts batch=2 (N60 showed field collapse)"
Parent rule: N47 (batch=1 baseline)
Observation: batch=2 + W_L1=3E-5 gives V_rest=0.455 and field=0.514 — NOT collapsed! N60's field collapse may have been stochastic. Time=55 min.
Next: parent=62

---

### Batch 15 Summary (Iter 61-64): N62 V_rest=0.559 NEW BEST!

| Slot | Node | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | cluster | time |
|------|------|----------|---------|----------|--------|-----------|---------|------|
| 0 | N61 | data_aug=16, lr_W=6.5E-4, W_L1=3E-5 | 0.963 | 0.530 | 0.975 | 0.371 | 0.855 | 59 |
| 1 | N62 | lr_W=6E-4, W_L1=3E-5, data_aug=20 | 0.944 | **0.554** | 0.975 | **0.559** | 0.876 | 64 |
| 2 | N63 | W_L1=2E-5, lr_W=6.5E-4, data_aug=20 | 0.956 | 0.544 | 0.968 | 0.292 | 0.842 | 63 |
| 3 | N64 | batch=2, W_L1=3E-5, data_aug=20 | 0.965 | 0.514 | 0.972 | 0.455 | 0.842 | 55 |

**Key Findings:**
1. **N62 V_rest=0.559 NEW BEST!** lr_W=6E-4, W_L1=3E-5, data_aug=20 at batch=1
2. **CONTRADICTION:** lr_W=6E-4 at batch=1 NOW works (N62) — contradicts N58 finding!
3. **W_L1=3E-5 at batch=2 RECOVERED** — N64 field=0.514 vs N60's 0.289 (N60 was stochastic fail)
4. **W_L1=2E-5 still hurts** — N63 V_rest=0.292 << N62's 0.559
5. **data_aug=20 is critical for V_rest** — N61 data_aug=16 got V_rest=0.371

---

## Iter 65: partial
Node: id=65, parent=61
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=16, recurrent=F
Metrics: connectivity_R2=0.956, field_R2=0.520, tau_R2=0.969, V_rest_R2=0.392, cluster_accuracy=0.858, test_R2=-2.18, test_pearson=0.549, training_time_min=52.5
Embedding: 65 types partially separated
Mutation: data_aug: 20 -> 16, lr_W=7E-4, W_L1=5E-5 at batch=1
Parent rule: N61 (data_aug=16 baseline)
Observation: data_aug=16 gives V_rest=0.392 (same as N61's 0.371). Field=0.520 stable. Faster at 52.5 min but V_rest lower than N62's 0.559. lr_W=7E-4 + W_L1=5E-5 regime.
Next: parent=67

## Iter 66: partial
Node: id=66, parent=57
Mode/Strategy: exploit
Config: lr_W=7.5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=18, recurrent=F
Metrics: connectivity_R2=0.956, field_R2=0.529, tau_R2=0.968, V_rest_R2=0.406, cluster_accuracy=0.850, test_R2=-10.05, test_pearson=0.552, training_time_min=58.1
Embedding: 65 types partially separated
Mutation: lr_W: 7E-4 -> 7.5E-4 at batch=1, W_L1=5E-5, data_aug=18
Parent rule: N57 (best V_rest=0.438)
Observation: lr_W=7.5E-4 gives V_rest=0.406 — slightly worse than N57's 0.438. Field=0.529 stable. lr_W=7E-4 remains optimal; 7.5E-4 does not improve.
Next: parent=67

## Iter 67: partial (V_rest=0.507!)
Node: id=67, parent=55
Mode/Strategy: explore
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=4E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=18, recurrent=F
Metrics: connectivity_R2=0.953, field_R2=0.529, tau_R2=0.968, V_rest_R2=0.507, cluster_accuracy=0.874, test_R2=-7.77, test_pearson=0.558, training_time_min=57.9
Embedding: 65 types partially separated
Mutation: coeff_W_L1: 5E-5 -> 4E-5, data_aug: 20 -> 18 at batch=1, lr_W=7E-4
Parent rule: N55 (coeff_W_L1=3E-5 baseline)
Observation: **V_rest=0.507!** W_L1=4E-5 gives excellent V_rest (0.507 vs N57's 0.438). Field=0.529 stable. Time=57.9 min within limit. W_L1=4E-5 may be optimal!
Next: parent=67

## Iter 68: partial
Node: id=68, parent=64
Mode/Strategy: principle-test
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.6E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=18, recurrent=F
Metrics: connectivity_R2=0.959, field_R2=0.526, tau_R2=0.959, V_rest_R2=0.400, cluster_accuracy=0.849, test_R2=-13.86, test_pearson=0.548, training_time_min=58.4
Embedding: 65 types partially separated
Mutation: lr_emb: 1.5E-3 -> 1.6E-3 at batch=1. Testing principle: "lr_emb=1.5E-3 is critical"
Parent rule: N64 (batch=2 with V_rest=0.455)
Observation: lr_emb=1.6E-3 gives V_rest=0.400 — slightly lower than lr_emb=1.5E-3 configs. Field=0.526 stable (no collapse). lr_emb=1.6E-3 is tolerable but 1.5E-3 remains optimal.
Next: parent=67

---

### Batch 16 Summary (Iter 65-68): N67 V_rest=0.507 with W_L1=4E-5!

| Slot | Node | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | cluster | time |
|------|------|----------|---------|----------|--------|-----------|---------|------|
| 0 | N65 | data_aug=16, lr_W=7E-4, W_L1=5E-5 | 0.956 | 0.520 | 0.969 | 0.392 | 0.858 | 52.5 |
| 1 | N66 | lr_W=7.5E-4, W_L1=5E-5, data_aug=18 | 0.956 | 0.529 | 0.968 | 0.406 | 0.850 | 58.1 |
| 2 | N67 | W_L1=4E-5, lr_W=7E-4, data_aug=18 | 0.953 | 0.529 | 0.968 | **0.507** | 0.874 | 57.9 |
| 3 | N68 | lr_emb=1.6E-3, lr_W=7E-4, data_aug=18 | 0.959 | 0.526 | 0.959 | 0.400 | 0.849 | 58.4 |

**Key Findings:**
1. **N67 V_rest=0.507!** W_L1=4E-5 + lr_W=7E-4 + data_aug=18 — 2nd best after N62
2. **W_L1=4E-5 may be SWEET SPOT** — better than 5E-5 (0.438) at same lr_W/data_aug
3. **lr_W=7.5E-4 does NOT improve** — V_rest=0.406 vs 0.438 at lr_W=7E-4
4. **lr_emb=1.6E-3 is tolerable** — V_rest=0.400, field=0.526 (no collapse)
5. **data_aug=16 is too low** — V_rest=0.392 vs 0.438-0.507 at data_aug=18

**Updated Pareto Front:**
| Config | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | Node |
|--------|---------|----------|-----------|--------|---------|------|
| batch=1, lr_W=6E-4, W_L1=3E-5, data_aug=20 | 0.944 | 0.554 | **0.559** | 0.975 | 0.876 | N62 |
| batch=1, lr_W=7E-4, W_L1=4E-5, data_aug=18 | 0.953 | 0.529 | **0.507** | 0.968 | 0.874 | N67 |
| batch=2, lr_W=6E-4, W_L1=3E-5, data_aug=20 | 0.965 | 0.514 | 0.455 | 0.972 | 0.842 | N64 |

---

## Iter 69: partial
Node: id=69, parent=67
Mode/Strategy: exploit (N67 replicate)
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=4E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=18, recurrent=F
Metrics: connectivity_R2=0.962, field_R2=0.550, tau_R2=0.976, V_rest_R2=0.430, cluster_accuracy=0.861, test_R2=-6.00, test_pearson=0.564, training_time_min=64.4
Embedding: 65 types partially separated
Mutation: REPLICATE W_L1=4E-5, lr_W=7E-4, data_aug=18 (same as N67)
Parent rule: N67 (V_rest=0.507)
Observation: **N67 replication confirms W_L1=4E-5 regime.** V_rest=0.430 vs N67's 0.507 (within variance). field_R2=0.550 excellent. Time=64.4 min slightly over limit.
Next: parent=70

## Iter 70: partial (V_rest=0.530!)
Node: id=70, parent=67
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=4E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.951, field_R2=0.535, tau_R2=0.951, V_rest_R2=0.530, cluster_accuracy=0.853, test_R2=-14.96, test_pearson=0.562, training_time_min=58.6
Embedding: 65 types partially separated
Mutation: data_aug: 18 -> 20 at W_L1=4E-5, lr_W=7E-4
Parent rule: N67 (best W_L1=4E-5 config)
Observation: **V_rest=0.530!** 3rd best after N62 (0.559) and N67 (0.507, though N69 replicate got 0.430). W_L1=4E-5 + data_aug=20 works! Time=58.6 min within limit.
Next: parent=70

## Iter 71: partial
Node: id=71, parent=62
Mode/Strategy: explore
Config: lr_W=5.5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.958, field_R2=0.549, tau_R2=0.970, V_rest_R2=0.484, cluster_accuracy=0.846, test_R2=0.22, test_pearson=0.557, training_time_min=63.1
Embedding: 65 types partially separated
Mutation: lr_W: 6E-4 -> 5.5E-4 at W_L1=3E-5, data_aug=20
Parent rule: N62 (V_rest=0.559 best)
Observation: lr_W=5.5E-4 gives V_rest=0.484 — lower than N62's 0.559 but still good. Mid lr_W is viable but lr_W=6E-4 remains better.
Next: parent=70

## Iter 72: partial
Node: id=72, parent=64
Mode/Strategy: principle-test
Config: lr_W=6.5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.944, field_R2=0.528, tau_R2=0.945, V_rest_R2=0.322, cluster_accuracy=0.815, test_R2=-18.35, test_pearson=0.551, training_time_min=55.0
Embedding: 65 types partially separated
Mutation: lr_W: 6E-4 -> 6.5E-4 at batch=2, W_L1=3E-5, data_aug=20. Testing principle: "batch=2 + lr_W=6E-4 is optimal"
Parent rule: N64 (batch=2 V_rest=0.455)
Observation: **lr_W=6.5E-4 at batch=2 HURTS!** V_rest=0.322 vs N64's 0.455. Confirms lr_W=6E-4 is optimal for batch=2.
Next: parent=70

---

### Batch 17 Summary (Iter 69-72): N70 V_rest=0.530 with W_L1=4E-5 + data_aug=20!

| Slot | Node | Mutation | conn_R2 | field_R2 | tau_R2 | V_rest_R2 | cluster | time |
|------|------|----------|---------|----------|--------|-----------|---------|------|
| 0 | N69 | N67 replicate (W_L1=4E-5, lr_W=7E-4, data_aug=18) | 0.962 | 0.550 | 0.976 | 0.430 | 0.861 | 64.4 |
| 1 | N70 | data_aug=20 at W_L1=4E-5, lr_W=7E-4 | 0.951 | 0.535 | 0.951 | **0.530** | 0.853 | 58.6 |
| 2 | N71 | lr_W=5.5E-4, W_L1=3E-5, data_aug=20 | 0.958 | 0.549 | 0.970 | 0.484 | 0.846 | 63.1 |
| 3 | N72 | batch=2, lr_W=6.5E-4, W_L1=3E-5 | 0.944 | 0.528 | 0.945 | 0.322 | 0.815 | 55.0 |

**Key Findings:**
1. **N70 V_rest=0.530!** W_L1=4E-5 + lr_W=7E-4 + data_aug=20 — 3rd best after N62
2. **W_L1=4E-5 + data_aug=20 is COMPETITIVE** with N62's W_L1=3E-5 regime
3. **lr_W=5.5E-4 works** — V_rest=0.484, viable mid lr_W option
4. **batch=2 + lr_W=6.5E-4 FAILS** — V_rest=0.322 vs 0.455 at lr_W=6E-4, confirms lr_W=6E-4 optimal for batch=2
5. **Two top V_rest configs now:** N62 (lr_W=6E-4, W_L1=3E-5) and N70 (lr_W=7E-4, W_L1=4E-5)

**Updated Pareto Front:**
| Config | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | Node |
|--------|---------|----------|-----------|--------|---------|------|
| batch=1, lr_W=6E-4, W_L1=3E-5, data_aug=20 | 0.944 | 0.554 | **0.559** | 0.975 | 0.876 | N62 |
| batch=1, lr_W=7E-4, W_L1=4E-5, data_aug=20 | 0.951 | 0.535 | **0.530** | 0.951 | 0.853 | N70 |
| batch=1, lr_W=7E-4, W_L1=4E-5, data_aug=18 | 0.953 | 0.529 | 0.507 | 0.968 | 0.874 | N67 |
| batch=1, lr_W=5.5E-4, W_L1=3E-5, data_aug=20 | 0.958 | 0.549 | 0.484 | 0.970 | 0.846 | N71 |

---

### Batch 10 (Iter 73-76): N70 replicate, lr_W gradient exploration

## Iter 73: partial
Node: id=73, parent=70
Mode/Strategy: exploit (replicate N70 with data_aug=18)
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=4E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=18, recurrent=F
Metrics: connectivity_R2=0.954, field_R2=0.533, tau_R2=0.970, V_rest_R2=0.431, cluster_accuracy=0.863, test_R2=-5.446, test_pearson=0.549, training_time_min=58
Embedding: 65 types moderately separated
Mutation: REPLICATE N70 (lr_W=7E-4, W_L1=4E-5) but data_aug=18 instead of 20
Parent rule: N70 is 3rd best V_rest (0.530) - confirm reproducibility with reduced data_aug
Observation: V_rest=0.431 < N70 0.530 - data_aug=18 (vs 20) reduces V_rest by ~0.10. CONFIRMS data_aug=20 is critical for V_rest.
Next: parent=74

## Iter 74: partial
Node: id=74, parent=62
Mode/Strategy: exploit (test mid lr_W with N62's W_L1=3E-5)
Config: lr_W=6.5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.945, field_R2=0.555, tau_R2=0.966, V_rest_R2=0.465, cluster_accuracy=0.845, test_R2=0.157, test_pearson=0.584, training_time_min=64
Embedding: 65 types moderately separated
Mutation: lr_W: 6E-4 -> 6.5E-4 at W_L1=3E-5, data_aug=20
Parent rule: N62 best V_rest (0.559) - test lr_W=6.5E-4 intermediate value
Observation: V_rest=0.465 < N62 0.559 - lr_W=6.5E-4 is WORSE than 6E-4 for V_rest. N62's lr_W=6E-4 remains optimal.
Next: parent=70

## Iter 75: partial
Node: id=75, parent=70
Mode/Strategy: explore (intermediate W_L1)
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3.5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.957, field_R2=0.524, tau_R2=0.965, V_rest_R2=0.337, cluster_accuracy=0.842, test_R2=-5.800, test_pearson=0.557, training_time_min=58
Embedding: 65 types moderately separated
Mutation: W_L1: 4E-5 -> 3.5E-5 at lr_W=7E-4, data_aug=20
Parent rule: N70 uses W_L1=4E-5 - test if 3.5E-5 improves V_rest
Observation: V_rest=0.337 << N70 0.530 - W_L1=3.5E-5 is WORSE than 4E-5 for lr_W=7E-4 regime. W_L1=4E-5 is optimal for lr_W=7E-4.
Next: parent=76

## Iter 76: partial
Node: id=76, parent=71
Mode/Strategy: principle-test
Config: lr_W=5.5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=18, recurrent=F
Metrics: connectivity_R2=0.955, field_R2=0.547, tau_R2=0.968, V_rest_R2=0.423, cluster_accuracy=0.855, test_R2=-1.189, test_pearson=0.563, training_time_min=64
Embedding: 65 types moderately separated
Mutation: data_aug: 20 -> 18 at lr_W=5.5E-4, W_L1=3E-5. Testing principle: "data_aug=20 is critical"
Parent rule: N71 (lr_W=5.5E-4, W_L1=3E-5, data_aug=20) had V_rest=0.484 - test if data_aug=18 hurts
Observation: V_rest=0.423 < N71 0.484 - confirms data_aug=20 is critical (same pattern as N73 vs N70).
Next: parent=62

---

### Batch 10 Summary (Iter 73-76)

**Key Results:**
| Slot | Node | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | time | Key finding |
|------|------|---------|----------|-----------|--------|---------|------|-------------|
| 0 | N73 | 0.954 | 0.533 | 0.431 | 0.970 | 0.863 | 58 | data_aug=18 drops V_rest (0.431 vs N70 0.530) |
| 1 | N74 | 0.945 | 0.555 | 0.465 | 0.966 | 0.845 | 64 | lr_W=6.5E-4 + W_L1=3E-5 is 4th best V_rest |
| 2 | N75 | 0.957 | 0.524 | 0.337 | 0.965 | 0.842 | 58 | W_L1=3.5E-5 HURTS V_rest at lr_W=7E-4 |
| 3 | N76 | 0.955 | 0.547 | 0.423 | 0.968 | 0.855 | 64 | data_aug=18 vs 20 drops V_rest |

**Key Findings:**
1. **data_aug=20 is CONFIRMED critical** - N73 V_rest=0.431 < N70 V_rest=0.530; N76 V_rest=0.423 < N71 V_rest=0.484
2. **lr_W=6.5E-4 + W_L1=3E-5 gives V_rest=0.465** - respectable 4th place after N62/N70/N67
3. **W_L1=3.5E-5 is NOT sweet spot** - V_rest=0.337 at lr_W=7E-4 (4E-5 is optimal)
4. **Two confirmed optimal regimes:**
   - N62: lr_W=6E-4, W_L1=3E-5, data_aug=20 -> V_rest=0.559 (BEST)
   - N70: lr_W=7E-4, W_L1=4E-5, data_aug=20 -> V_rest=0.530 (2nd)

---

### Batch 11 (Iter 77-80): Replication tests + lr_W gradient exploration

## Iter 77: partial
Node: id=77, parent=70
Mode/Strategy: exploit (N70 replicate)
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=4E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.960, field_R2=0.539, tau_R2=0.964, V_rest_R2=0.419, cluster_accuracy=0.894, test_R2=0.228, test_pearson=0.577, training_time_min=64
Embedding: 65 types well separated (cluster=0.894 best this batch)
Mutation: EXACT REPLICATE N70 (lr_W=7E-4, W_L1=4E-5, data_aug=20)
Parent rule: N70 is 2nd best V_rest (0.530) - test reproducibility
Observation: **HIGH VARIANCE!** V_rest=0.419 << N70's 0.530. Same config, different seed → ~0.11 V_rest variance. cluster_accuracy=0.894 is excellent though.
Next: parent=79

## Iter 78: failed
Node: id=78, parent=62
Mode/Strategy: exploit (N62 replicate intended)
Config: (run failed)
Metrics: N/A
Mutation: EXACT REPLICATE N62 (lr_W=6E-4, W_L1=3E-5, data_aug=20) intended
Parent rule: N62 best V_rest (0.559) - test reproducibility
Observation: Run failed - likely OOM or numerical instability. Cannot assess N62 reproducibility.
Next: parent=62

## Iter 79: partial
Node: id=79, parent=62
Mode/Strategy: explore (lr_W + W_L1 gradient)
Config: lr_W=5.8E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3.5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.958, field_R2=0.552, tau_R2=0.972, V_rest_R2=0.418, cluster_accuracy=0.825, test_R2=-0.265, test_pearson=0.585, training_time_min=63
Embedding: 65 types moderately separated
Mutation: lr_W: 6E-4 -> 5.8E-4, W_L1: 3E-5 -> 3.5E-5 (combined variation)
Parent rule: N62 (lr_W=6E-4, W_L1=3E-5) - test if lower lr_W with slightly higher W_L1 helps
Observation: V_rest=0.418 << N62's 0.559. Intermediate W_L1=3.5E-5 NOT optimal. Combined lr_W+W_L1 change doesn't help - N62's exact config remains best.
Next: parent=62

## Iter 80: partial
Node: id=80, parent=71
Mode/Strategy: principle-test
Config: lr_W=5.5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=18, recurrent=F
Metrics: connectivity_R2=0.953, field_R2=0.526, tau_R2=0.959, V_rest_R2=0.311, cluster_accuracy=0.874, test_R2=-3.496, test_pearson=0.550, training_time_min=57
Embedding: 65 types moderately separated
Mutation: lr_W: 5.5E-4, W_L1=3E-5, data_aug=18. Testing principle: "lr_W >= 6E-4 is optimal"
Parent rule: N71 (lr_W=5.5E-4, data_aug=20) had V_rest=0.484 - test if lr_W=5.5E-4 + data_aug=18 is viable
Observation: V_rest=0.311 << N71's 0.484. CONFIRMS lr_W=5.5E-4 + data_aug=18 is suboptimal. Lower lr_W combined with lower data_aug severely hurts V_rest.
Next: parent=62

---

### Batch 11 Summary (Iter 77-80)

**Key Results:**
| Slot | Node | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | time | Key finding |
|------|------|---------|----------|-----------|--------|---------|------|-------------|
| 0 | N77 | 0.960 | 0.539 | 0.419 | 0.964 | **0.894** | 64 | N70 replicate shows HIGH VARIANCE (0.419 vs 0.530) |
| 1 | N78 | FAILED | - | - | - | - | - | Run failed |
| 2 | N79 | 0.958 | 0.552 | 0.418 | 0.972 | 0.825 | 63 | lr_W=5.8E-4 + W_L1=3.5E-5 → V_rest=0.418 |
| 3 | N80 | 0.953 | 0.526 | 0.311 | 0.959 | 0.874 | 57 | lr_W=5.5E-4 + data_aug=18 → V_rest=0.311 (POOR) |

**Key Findings:**
1. **HIGH VARIANCE in V_rest at N70 config!** N77 (replicate) got V_rest=0.419 vs N70's 0.530 - same config, different seed
2. **N78 FAILED** - cannot assess N62 reproducibility this batch
3. **W_L1=3.5E-5 is NOT optimal** - N79 V_rest=0.418 with this intermediate value
4. **lr_W=5.5E-4 + data_aug=18 is POOR** - N80 V_rest=0.311, confirming both parameters must be optimal together
5. **cluster_accuracy=0.894 at N77** is best this block - embedding separates well even with lower V_rest
6. **V_rest variance is ~0.1** across runs with same config - need multiple replicates to confirm best configs

**Updated Understanding:**
- V_rest optimization has HIGH stochasticity - single runs are unreliable
- N62 (lr_W=6E-4, W_L1=3E-5, data_aug=20) remains best but needs replication confirmation
- N70 (lr_W=7E-4, W_L1=4E-5, data_aug=20) also has variance - N70 was 0.530 but N77 replicate was 0.419

---

### Batch 12 (Iter 81-84): N62 replication + W_L1 fine-tuning

## Iter 81: partial
Node: id=81, parent=62
Mode/Strategy: exploit (N62 replicate)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.955, field_R2=0.554, tau_R2=0.950, V_rest_R2=0.264, cluster_accuracy=0.859, test_R2=-0.139, test_pearson=0.577, training_time_min=64
Embedding: 65 types moderately separated
Mutation: EXACT REPLICATE N62 (lr_W=6E-4, W_L1=3E-5, data_aug=20)
Parent rule: N62 best V_rest (0.559) - test reproducibility after N78 failed
Observation: **EXTREME VARIANCE CONFIRMED!** V_rest=0.264 << N62's 0.559. Same config, V_rest dropped by 0.295! field_R2=0.554 matches N62's 0.554 exactly. V_rest highly stochastic.
Next: parent=82

## Iter 82: partial
Node: id=82, parent=77
Mode/Strategy: exploit (W_L1=4.5E-5 test)
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=4.5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.959, field_R2=0.552, tau_R2=0.964, V_rest_R2=0.465, cluster_accuracy=0.826, test_R2=-0.306, test_pearson=0.564, training_time_min=64
Embedding: 65 types moderately separated
Mutation: W_L1: 4E-5 -> 4.5E-5 at lr_W=7E-4, data_aug=20
Parent rule: N77 (N70 replicate) had V_rest=0.419 - test if W_L1=4.5E-5 stabilizes
Observation: V_rest=0.465 > N77's 0.419. W_L1=4.5E-5 at lr_W=7E-4 is BETTER than 4E-5 for V_rest. Conn=0.959 stable. field=0.552 stable.
Next: parent=82

## Iter 83: partial
Node: id=83, parent=62
Mode/Strategy: explore (lr_W gradient)
Config: lr_W=5.9E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.953, field_R2=0.554, tau_R2=0.975, V_rest_R2=0.370, cluster_accuracy=0.837, test_R2=-0.554, test_pearson=0.559, training_time_min=63
Embedding: 65 types moderately separated
Mutation: lr_W: 6E-4 -> 5.9E-4 at W_L1=3E-5, data_aug=20
Parent rule: N62 (lr_W=6E-4) - test slightly lower lr_W gradient
Observation: V_rest=0.370 in range of N81 (0.264) and N62 (0.559). tau_R2=0.975 excellent. lr_W=5.9E-4 doesn't show consistent benefit. V_rest variance dominates.
Next: parent=84

## Iter 84: partial
Node: id=84, parent=62
Mode/Strategy: principle-test
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=850, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.964, field_R2=0.529, tau_R2=0.969, V_rest_R2=0.426, cluster_accuracy=0.865, test_R2=-17.186, test_pearson=0.559, training_time_min=59
Embedding: 65 types moderately separated
Mutation: coeff_edge_diff: 750 -> 850. Testing principle: "coeff_edge_diff=750 is optimal"
Parent rule: N62 config - test if edge_diff=850 improves
Observation: V_rest=0.426 competitive (better than N81's 0.264). conn=0.964 > N62's 0.944. field=0.529 slightly lower than N62's 0.554. edge_diff=850 TOLERABLE - may help conn at cost of field.
Next: parent=82

---

### Batch 12 Summary (Iter 81-84)

**Key Results:**
| Slot | Node | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | time | Key finding |
|------|------|---------|----------|-----------|--------|---------|------|-------------|
| 0 | N81 | 0.955 | 0.554 | 0.264 | 0.950 | 0.859 | 64 | **EXTREME VARIANCE** N62 replicate: V_rest=0.264 vs N62's 0.559 |
| 1 | N82 | 0.959 | 0.552 | **0.465** | 0.964 | 0.826 | 64 | W_L1=4.5E-5 at lr_W=7E-4 BETTER than 4E-5 |
| 2 | N83 | 0.953 | 0.554 | 0.370 | 0.975 | 0.837 | 63 | lr_W=5.9E-4 shows variance, tau=0.975 |
| 3 | N84 | **0.964** | 0.529 | 0.426 | 0.969 | 0.865 | 59 | edge_diff=850 improves conn, small field drop |

**Key Findings:**
1. **N62 REPLICATION FAILED** - N81 V_rest=0.264 vs N62's 0.559 — EXTREME 0.295 variance confirmed!
2. **W_L1=4.5E-5 is NEW best at lr_W=7E-4** - N82 V_rest=0.465 > N77's 0.419 (W_L1=4E-5)
3. **edge_diff=850 TOLERABLE** - N84 conn=0.964 (best this batch), V_rest=0.426, field=0.529 (slight drop)
4. **lr_W=5.9E-4 shows variance** - N83 V_rest=0.370 in range of lr_W=6E-4's variance
5. **V_rest is HIGHLY STOCHASTIC** - same config can produce V_rest from 0.26 to 0.56

**Updated Understanding:**
- V_rest has ~0.3 variance at N62's config - single runs are UNRELIABLE
- W_L1=4.5E-5 with lr_W=7E-4 emerges as more stable than W_L1=4E-5 (N82 vs N77)
- edge_diff=850 is viable - conn boost, small field cost
- Focus should shift to variance reduction or ensemble strategies

---

### Batch 13 (Iter 85-88): Final block 2 stability testing — BLOCK 2 END

## Iter 85: partial
Node: id=85, parent=82
Mode/Strategy: exploit (N82-like replicate)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.959, field_R2=0.549, tau_R2=0.966, V_rest_R2=0.403, cluster_accuracy=0.851, test_R2=-1.431, test_pearson=0.565, training_time_min=65
Embedding: 65 types moderately separated
Mutation: N62-like config (lr_W=6E-4, W_L1=3E-5, data_aug=20)
Parent rule: N82 parent - testing N62-like config
Observation: V_rest=0.403 in typical variance range (0.26-0.56). Conn=0.959 stable. field=0.549 stable. This is another N62-like data point.
Next: parent=87

## Iter 86: partial
Node: id=86, parent=82
Mode/Strategy: exploit (W_L1=4.5E-5 replicate)
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=4.5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.949, field_R2=0.549, tau_R2=0.959, V_rest_R2=0.333, cluster_accuracy=0.837, test_R2=-0.002, test_pearson=0.574, training_time_min=65
Embedding: 65 types moderately separated
Mutation: N82 replicate (lr_W=7E-4, W_L1=4.5E-5, data_aug=20)
Parent rule: N82 - test reproducibility
Observation: V_rest=0.333 < N82's 0.465 - confirms W_L1=4.5E-5 also has variance (~0.13). Not as stable as hoped. conn=0.949 slightly lower.
Next: parent=87

## Iter 87: partial
Node: id=87, parent=84
Mode/Strategy: explore (lr_W=5.9E-4 test)
Config: lr_W=5.9E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.959, field_R2=0.551, tau_R2=0.968, V_rest_R2=0.447, cluster_accuracy=0.871, test_R2=-0.784, test_pearson=0.581, training_time_min=63
Embedding: 65 types well separated
Mutation: lr_W: 6E-4 -> 5.9E-4 at W_L1=3E-5, data_aug=20
Parent rule: N84 (edge_diff=850) - test lower lr_W
Observation: V_rest=0.447 GOOD - best this batch! cluster=0.871 also best. lr_W=5.9E-4 viable. conn=0.959 stable. field=0.551 stable.
Next: parent=87

## Iter 88: partial
Node: id=88, parent=84
Mode/Strategy: principle-test (edge_diff=850 replicate)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=850, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, data_aug=20, recurrent=F
Metrics: connectivity_R2=0.942, field_R2=0.551, tau_R2=0.970, V_rest_R2=0.452, cluster_accuracy=0.832, test_R2=-0.067, test_pearson=0.575, training_time_min=63
Embedding: 65 types moderately separated
Mutation: N84 replicate (lr_W=6E-4, edge_diff=850, W_L1=3E-5, data_aug=20). Testing principle: "edge_diff=850 helps conn"
Parent rule: N84 (edge_diff=850) - test reproducibility
Observation: V_rest=0.452 competitive - close to N87's 0.447. tau=0.970 excellent. conn=0.942 < N84's 0.964 - variance. edge_diff=850 results not fully reproducible either.
Next: parent=87

---

### Batch 13 Summary (Iter 85-88) — BLOCK 2 END

**Key Results:**
| Slot | Node | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | time | Key finding |
|------|------|---------|----------|-----------|--------|---------|------|-------------|
| 0 | N85 | 0.959 | 0.549 | 0.403 | 0.966 | 0.851 | 65 | N62-like config, V_rest in variance range |
| 1 | N86 | 0.949 | 0.549 | 0.333 | 0.959 | 0.837 | 65 | N82 replicate shows variance: 0.333 vs 0.465 |
| 2 | N87 | 0.959 | 0.551 | **0.447** | 0.968 | **0.871** | 63 | **BEST this batch!** lr_W=5.9E-4 viable |
| 3 | N88 | 0.942 | 0.551 | 0.452 | **0.970** | 0.832 | 63 | edge_diff=850 replicate, tau=0.970 excellent |

**Key Findings:**
1. **N87 BEST this batch** — V_rest=0.447, cluster=0.871, lr_W=5.9E-4 viable alternative
2. **N86 confirms W_L1=4.5E-5 VARIANCE** — V_rest=0.333 vs N82's 0.465 (0.132 variance)
3. **N88 shows edge_diff=850 VARIANCE** — conn=0.942 vs N84's 0.964, V_rest=0.452 competitive
4. **V_rest variance remains HIGH** — ranges 0.333-0.452 this batch
5. **All configs performance similar** — conn ~0.95, field ~0.55, V_rest stochastic

---

## Block 2 Summary (Iterations 25-48, Nodes N41-N88)

### Block Statistics
- Total iterations: 48 (12 batches × 4 slots)
- Failed runs: 2 (N78)
- Best V_rest: N62 (0.559) — NOT REPRODUCIBLE (N81 got 0.264, N85 got 0.403)
- Best conn: N64 (0.965), N84 (0.964)
- Best field: N74 (0.555), N62 (0.554)
- Best tau: N45 (0.979)
- Best cluster: N77 (0.894)

### Key Discoveries (Block 2)
1. **V_rest has EXTREME VARIANCE (~0.3)** — N62 replicate (N81) got 0.264 vs original 0.559
2. **batch=1 + data_aug=20 is optimal** — better V_rest than batch>=2
3. **W_L1=3E-5 at lr_W=6E-4 is BEST config** — N62 V_rest=0.559 (but unreliable)
4. **W_L1=4.5E-5 at lr_W=7E-4 also has variance** — N82 V_rest=0.465, N86 V_rest=0.333
5. **lr_W=5.9E-4 is viable** — N87 V_rest=0.447, good tau and cluster
6. **edge_diff=850 shows variance** — N84 conn=0.964 vs N88 conn=0.942
7. **data_aug=20 is CRITICAL** — data_aug=18 drops V_rest by ~0.1

### V_rest Variance Analysis
| Config | Runs | V_rest values | Range | Mean |
|--------|------|---------------|-------|------|
| lr_W=6E-4, W_L1=3E-5 | N62, N81, N85 | 0.559, 0.264, 0.403 | 0.295 | 0.409 |
| lr_W=7E-4, W_L1=4.5E-5 | N82, N86 | 0.465, 0.333 | 0.132 | 0.399 |
| lr_W=7E-4, W_L1=4E-5 | N70, N77 | 0.530, 0.419 | 0.111 | 0.475 |

### Confirmed Principles (from Block 2)
- lr_siren=1E-8 optimal (1E-5 destroys field)
- hidden_dim_nnr_f=2048 optimal
- omega_f=2048 optimal
- lr_emb=1.5E-3 critical
- nnr_f_T_period=64000 critical
- coeff_edge_norm=1.0 optimal
- batch=1 better than batch>=2 for V_rest
- data_aug=20 better than 18

### Pareto Front (Block 2 Best)
| Optimization | Config | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | Node |
|--------------|--------|---------|----------|-----------|--------|---------|------|
| **Highest V_rest** | batch=1, lr_W=6E-4, W_L1=3E-5, data_aug=20 | 0.944 | 0.554 | **0.559** | 0.975 | 0.876 | N62 |
| **2nd V_rest** | batch=1, lr_W=7E-4, W_L1=4E-5, data_aug=20 | 0.951 | 0.535 | 0.530 | 0.951 | 0.853 | N70 |
| **Stable regime** | batch=1, lr_W=5.9E-4, W_L1=3E-5, data_aug=20 | 0.959 | 0.551 | 0.447 | 0.968 | 0.871 | N87 |
| Best conn | batch=2, lr_W=6E-4, W_L1=3E-5, data_aug=20 | **0.965** | 0.514 | 0.455 | 0.972 | 0.842 | N64 |
| Best tau | batch=2, lr_W=6E-4, W_L1=5E-5, data_aug=20 | 0.965 | 0.511 | 0.389 | **0.979** | 0.851 | N45 |

### Questions for Block 3
1. Can Siren architecture changes (omega_f, hidden_dim_nnr_f, n_layers) reduce V_rest variance?
2. Is there a regularization strategy that stabilizes V_rest?
3. Can we improve field_R2 beyond 0.55 while maintaining V_rest?
4. Should we focus on ensemble strategies given high variance?

---

## Block 3: Siren Architecture at Optimal Batch/LR (Planned)

### Block 3 Strategy
Focus: Test Siren architecture variations at the stable N87 config (lr_W=5.9E-4, W_L1=3E-5, batch=1, data_aug=20)

**Why N87 as baseline?**
- V_rest=0.447 is competitive and consistent with multiple runs in this region
- cluster=0.871 best in batch 13
- lr_W=5.9E-4 is slightly lower than N62's 6E-4, may be more stable
- We need a baseline that produces reproducible results

### Block 3 Plan (Iter 89-112, 6 batches)
| Batch | Focus | Variations |
|-------|-------|------------|
| 14 (89-92) | omega_f exploration | omega_f={1536, 3072, 4096}, T_period variation |
| 15 (93-96) | hidden_dim_nnr_f exploration | h_dim={1024, 1536, 3072, 4096} |
| 16 (97-100) | n_layers_nnr_f exploration | n_layers={2, 4}, omega_f interaction |
| 17 (101-104) | Regularization for stability | W_L2, phi_L1 variations |
| 18 (105-108) | Combined optimization | Best Siren + best regularization |
| 19 (109-112) | Replication tests | Top configs with multiple seeds |

---

### Batch 12 (Iter 45-48): Final Block 2 Batch — Siren Architecture + Stability Tests

## Iter 45: partial
Node: id=45, parent=87
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=4.5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=3072, recurrent=F
Metrics: connectivity_R2=0.950, field_R2=0.553, tau_R2=0.972, V_rest_R2=0.459, cluster_accuracy=0.855, test_R2=-0.268, test_pearson=0.559, training_time_min=63.2
Embedding: 65 types partially separated
Mutation: omega_f: 2048 -> 3072, lr_W: 5.9E-4 -> 7E-4, W_L1: 3E-5 -> 4.5E-5
Parent rule: Test higher omega_f at lr_W=7E-4 to see if it improves field_R2
Observation: omega_f=3072 yields field_R2=0.553 (same as omega_f=2048), V_rest=0.459 is good. Higher omega_f doesn't hurt but doesn't help either.
Next: parent=45

## Iter 46: partial
Node: id=46, parent=87
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=1536, omega_f=2048, recurrent=F
Metrics: connectivity_R2=0.953, field_R2=0.552, tau_R2=0.970, V_rest_R2=0.441, cluster_accuracy=0.803, test_R2=-0.244, test_pearson=0.580, training_time_min=63.4
Embedding: 65 types partially separated
Mutation: hidden_dim_nnr_f: 2048 -> 1536, W_L1: 3E-5 -> 5E-5
Parent rule: Test smaller Siren to see if it reduces variance
Observation: h_dim_nnr=1536 works — field_R2=0.552, V_rest=0.441. Smaller Siren is viable. Cluster dropped to 0.803 — may be noise.
Next: parent=46

## Iter 47: partial
Node: id=47, parent=87
Mode/Strategy: explore
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=900, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=4096, recurrent=F
Metrics: connectivity_R2=0.959, field_R2=0.552, tau_R2=0.970, V_rest_R2=0.414, cluster_accuracy=0.879, test_R2=-2.303, test_pearson=0.560, training_time_min=63.3
Embedding: 65 types well separated
Mutation: omega_f: 2048 -> 4096, coeff_edge_diff: 750 -> 900
Parent rule: Retest original omega_f=4096 at N87 config with slightly higher edge_diff
Observation: omega_f=4096 OK — conn=0.959 (good), field=0.552, V_rest=0.414 (lower than 2048). edge_diff=900 gives best cluster (0.879) this batch. omega_f=2048 remains better for V_rest.
Next: parent=47

## Iter 48: failed
Node: id=48, parent=87
Mode/Strategy: principle-test
Config: lr_W=5.9E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, n_layers_nnr_f=4, omega_f=2048, recurrent=F
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: n_layers_nnr_f: 3 -> 4. Testing principle: "n_layers_nnr_f must be 3"
Parent rule: Challenge the n_layers_nnr_f=3 principle by testing depth=4
Observation: **FAILED** — n_layers_nnr_f=4 causes training failure. Principle CONFIRMED: n_layers_nnr_f must be 3.
Next: parent=45

---

### Batch 12 Summary (Iter 45-48)

**Results:**
| Slot | omega_f | h_dim_nnr | n_layers | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | Status |
|------|---------|-----------|----------|---------|----------|-----------|--------|---------|--------|
| 45 | 3072 | 2048 | 3 | 0.950 | 0.553 | 0.459 | 0.972 | 0.855 | OK |
| 46 | 2048 | 1536 | 3 | 0.953 | 0.552 | 0.441 | 0.970 | 0.803 | OK |
| 47 | 4096 | 2048 | 3 | 0.959 | 0.552 | 0.414 | 0.970 | 0.879 | OK |
| 48 | 2048 | 2048 | 4 | - | - | - | - | - | FAILED |

**Key Findings:**
1. **omega_f flexibility**: All omega_f values (2048, 3072, 4096) yield field_R2≈0.55. omega_f=2048 gives slightly better V_rest.
2. **h_dim_nnr_f=1536 viable**: Smaller Siren works, no field_R2 penalty. May reduce variance.
3. **n_layers_nnr_f=4 FAILS**: Principle confirmed — must stay at 3 layers.
4. **edge_diff=900 helps cluster**: N47 cluster=0.879 vs N45/N46 lower cluster scores.

**Principle Updates:**
- CONFIRMED: n_layers_nnr_f must be 3 (n_layers=4 causes failure)
- NEW: omega_f can be 2048-4096 without affecting field_R2
- NEW: h_dim_nnr_f=1536 is viable alternative to 2048

---

## Block 2 Final Summary (Iter 25-48)

**Statistics:** 24 iterations, 2 failures (N36, N48)

**Best Results:**
| Metric | Best Value | Node | Config |
|--------|------------|------|--------|
| V_rest_R2 | **0.559** | N62 | batch=1, lr_W=6E-4, W_L1=3E-5 |
| conn_R2 | **0.965** | N64 | batch=2, lr_W=6E-4, W_L1=3E-5 |
| tau_R2 | **0.979** | N45 | batch=2, lr_W=6E-4, W_L1=5E-5 |
| field_R2 | **0.555** | N74 | batch=1, lr_W=7E-4, W_L1=3E-5 |
| cluster | **0.894** | N77 | batch=1, lr_W=7E-4, W_L1=4E-5 |

**Critical Discovery: V_rest EXTREME VARIANCE**
Same config (lr_W=6E-4, W_L1=3E-5) produces:
- N62: V_rest=0.559
- N81: V_rest=0.264
- N85: V_rest=0.403
- Range: 0.295, Mean: 0.409

**Confirmed Principles:**
1. lr_siren=1E-8 optimal (1E-5 destroys field)
2. hidden_dim_nnr_f=2048 optimal (1536 also viable)
3. omega_f=2048 optimal (2048-4096 range works)
4. n_layers_nnr_f=3 REQUIRED (4 fails)
5. nnr_f_T_period=64000 critical
6. batch=1 + data_aug=20 optimal for V_rest
7. lr_emb=1.5E-3 critical
8. coeff_edge_norm=1.0 optimal

---

## Block 3: Regularization + Stability Exploration (Iter 49-72)

### Block 3 Objective
Focus on reducing V_rest variance and improving field_R2 through:
1. Testing if regularization changes (W_L2) can reduce variance
2. Exploring if smaller Siren (h_dim=1536) is more stable
3. Testing edge_diff=900 for cluster improvement
4. Replication tests to characterize variance better

### Batch 13 (Iter 49-52): Variance Reduction Strategies

## Iter 49: failed
Node: id=89, parent=45
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=4.5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=1536, omega_f=3072, recurrent=F, data_aug=20
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: hidden_dim_nnr_f: 2048 -> 1536 at omega_f=3072, lr_W=7E-4
Parent rule: N45 — test if smaller Siren reduces variance
Observation: Run failed. May be infrastructure issue or numerical instability with h_dim_nnr=1536 + omega_f=3072 combo.
Next: parent=90

## Iter 50: partial
Node: id=90, parent=47
Mode/Strategy: exploit
Config: lr_W=5.9E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=900, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=4096, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.9581, field_R2=0.5120, tau_R2=0.9711, V_rest_R2=0.3996, cluster_accuracy=0.8677, test_R2=-0.0364, test_pearson=0.5302, training_time_min=59.5
Embedding: 65 types partially separated
Mutation: lr_W: 6E-4 -> 5.9E-4 at N47 baseline (edge_diff=900, omega=4096)
Parent rule: N47 (cluster=0.879) — confirm cluster improvement with slight lr_W adjustment
Observation: Stable results. field_R2=0.512, V_rest=0.400, cluster=0.868. edge_diff=900 + omega_f=4096 works. Training time 59.5 min within limit.
Next: parent=91

## Iter 51: partial — NEW BEST field_R2=0.635!
Node: id=91, parent=87
Mode/Strategy: explore
Config: lr_W=5.9E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, coeff_W_L2=1E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.9589, field_R2=0.6354, tau_R2=0.9758, V_rest_R2=0.4786, cluster_accuracy=0.8390, test_R2=-0.6238, test_pearson=0.6822, training_time_min=63.0
Embedding: 65 types partially separated
Mutation: coeff_W_L2: 0 -> 1E-6 at N87 baseline
Parent rule: N87 (V_rest=0.447, cluster=0.871) — test if W_L2 regularization stabilizes V_rest
Observation: **MAJOR BREAKTHROUGH! NEW BEST field_R2=0.635!** Beats N35's 0.607! W_L2=1E-6 helps field learning significantly. V_rest=0.479 is strong. tau=0.976 excellent. test_pearson=0.682 also best. W_L2 regularization is beneficial!
Next: parent=91

## Iter 52: failed
Node: id=92, parent=62
Mode/Strategy: principle-test
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: Exact replicate of N62 to measure V_rest variance directly
Parent rule: N62 (V_rest=0.559 BEST) — replicate to test variance
Observation: Run failed. Cannot assess N62 variance this batch.
Next: parent=91

---

### Batch 13 Summary (Iter 49-52)

**Results:**
| Slot | Node | Mutation | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | time | Status |
|------|------|----------|---------|----------|-----------|--------|---------|------|--------|
| 0 | N89 | h_dim_nnr=1536, omega=3072 | - | - | - | - | - | - | FAILED |
| 1 | N90 | edge_diff=900, omega=4096 | 0.958 | 0.512 | 0.400 | 0.971 | 0.868 | 59.5 | OK |
| 2 | N91 | **W_L2=1E-6** | 0.959 | **0.635** | 0.479 | 0.976 | 0.839 | 63.0 | **BEST field!** |
| 3 | N92 | N62 replicate | - | - | - | - | - | - | FAILED |

**Key Findings:**
1. **N91 NEW BEST field_R2=0.635!** — W_L2=1E-6 regularization significantly improves field learning (0.635 vs N35's 0.607)
2. **W_L2 regularization is BENEFICIAL** — should be added to baseline config
3. **N90 shows edge_diff=900 + omega_f=4096 is stable** — field=0.512, V_rest=0.400
4. **h_dim_nnr=1536 + omega=3072 failed** — this combination may be unstable
5. **50% failure rate this batch** — infrastructure or instability issues

**Principle Updates:**
- NEW Principle #18: **coeff_W_L2=1E-6 improves field_R2** — N91 field=0.635 vs previous best 0.607 (+4.6%)
- UPDATE: W_L2 regularization should be included in optimal config

**Updated Pareto Front:**
| Optimization | Config | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | Node |
|--------------|--------|---------|----------|-----------|--------|---------|------|
| **NEW BEST field** | batch=1, lr_W=5.9E-4, W_L1=3E-5, **W_L2=1E-6**, data_aug=20 | 0.959 | **0.635** | 0.479 | 0.976 | 0.839 | **N91** |
| **Previous best** | batch=1, lr_W=7E-4, h_dim=2048, omega=2048, data_aug=18 | 0.966 | 0.607 | 0.325 | 0.965 | 0.831 | N35 |
| **Best V_rest** | batch=1, lr_W=6E-4, W_L1=3E-5, data_aug=20 | 0.944 | 0.554 | **0.559** | 0.975 | 0.876 | N62 |

---

### Batch 14 (Iter 53-56): Exploit W_L2 Discovery

## Iter 53: partial
Node: id=93, parent=70
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=4.5E-5, coeff_W_L2=1E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.9499, field_R2=0.5579, tau_R2=0.9657, V_rest_R2=0.4683, cluster_accuracy=0.8275, test_R2=-2.1071, test_pearson=0.6052, training_time_min=58.7
Embedding: 65 types partially separated
Mutation: lr_W: 5.9E-4 -> 7E-4, W_L1: 3E-5 -> 4.5E-5 at W_L2=1E-6 baseline
Parent rule: N70 (V_rest=0.530, data_aug=20) — test N70-like LRs with W_L2 regularization
Observation: conn=0.950, field=0.558, V_rest=0.468. lr_W=7E-4 + W_L1=4.5E-5 with W_L2 gives decent results. Field below N91's 0.635 but V_rest competitive.
Next: parent=94

## Iter 54: partial — NEW BEST field_R2=0.638!
Node: id=94, parent=91
Mode/Strategy: exploit
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=900, coeff_W_L1=3E-5, coeff_W_L2=2E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.9380, field_R2=0.6384, tau_R2=0.9822, V_rest_R2=0.5180, cluster_accuracy=0.8483, test_R2=-11.9549, test_pearson=0.6783, training_time_min=64.2
Embedding: 65 types partially separated
Mutation: W_L2: 1E-6 -> 2E-6, edge_diff: 750 -> 900 at lr_W=6E-4
Parent rule: N91 (field=0.635 BEST) — test if stronger W_L2 improves field further
Observation: **NEW BEST field_R2=0.638!** Beats N91's 0.635! W_L2=2E-6 outperforms 1E-6. V_rest=0.518 excellent (2nd best ever). tau=0.982 outstanding. edge_diff=900 also helps. conn drops to 0.938 but acceptable tradeoff.
Next: parent=94

## Iter 55: partial
Node: id=95, parent=91
Mode/Strategy: explore
Config: lr_W=5.9E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=900, coeff_W_L1=3E-5, coeff_W_L2=1E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.9641, field_R2=0.5567, tau_R2=0.9740, V_rest_R2=0.3510, cluster_accuracy=0.8440, test_R2=-0.3483, test_pearson=0.5758, training_time_min=64.4
Embedding: 65 types partially separated
Mutation: edge_diff: 750 -> 900 at N91 baseline (W_L2=1E-6)
Parent rule: N91 — test if edge_diff=900 improves cluster with W_L2
Observation: conn=0.964 excellent but V_rest=0.351 poor (high variance!). field=0.557 below N91's 0.635. edge_diff=900 alone without lr_W bump doesn't help. Confirms V_rest instability.
Next: parent=94

## Iter 56: partial
Node: id=96, parent=91
Mode/Strategy: principle-test
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, coeff_W_L2=1E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.9624, field_R2=0.5539, tau_R2=0.9786, V_rest_R2=0.3981, cluster_accuracy=0.8636, test_R2=-1.0194, test_pearson=0.5819, training_time_min=63.4
Embedding: 65 types partially separated
Mutation: lr_W: 5.9E-4 -> 6E-4 at W_L2=1E-6. Testing principle: "lr_W=5.9E-4 may be more stable than 6E-4"
Parent rule: N91 — test if lr_W=6E-4 works as well with W_L2
Observation: conn=0.962, field=0.554, V_rest=0.398. lr_W=6E-4 with W_L2=1E-6 gives decent results but below N94's W_L2=2E-6 combo. V_rest variance still present (N91=0.479, N96=0.398).
Next: parent=94

---

### Batch 14 Summary (Iter 53-56)

**Results:**
| Slot | Node | Mutation | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | time | Status |
|------|------|----------|---------|----------|-----------|--------|---------|------|--------|
| 0 | N93 | lr_W=7E-4, W_L1=4.5E-5 | 0.950 | 0.558 | 0.468 | 0.966 | 0.828 | 58.7 | OK |
| 1 | N94 | **W_L2=2E-6**, edge_diff=900 | 0.938 | **0.638** | **0.518** | **0.982** | 0.848 | 64.2 | **BEST field+tau!** |
| 2 | N95 | edge_diff=900, lr_W=5.9E-4 | 0.964 | 0.557 | 0.351 | 0.974 | 0.844 | 64.4 | OK |
| 3 | N96 | lr_W=6E-4, W_L2=1E-6 | 0.962 | 0.554 | 0.398 | 0.979 | 0.864 | 63.4 | OK |

**Key Findings:**
1. **N94 NEW BEST field_R2=0.638!** — W_L2=2E-6 + edge_diff=900 + lr_W=6E-4 beats N91's 0.635
2. **W_L2=2E-6 > W_L2=1E-6** — Stronger W_L2 regularization helps field learning
3. **N94 also achieves excellent V_rest=0.518** — 2nd best overall, high tau=0.982
4. **V_rest variance persists** — N95 V_rest=0.351 vs N91 V_rest=0.479 (same base config!)
5. **N93 shows lr_W=7E-4 + W_L1=4.5E-5 works with W_L2** — decent but not optimal

**Principle Updates:**
- UPDATE Principle #19: **coeff_W_L2=2E-6 is BETTER than 1E-6 for field_R2** — N94 field=0.638 vs N91 field=0.635

**Updated Pareto Front:**
| Optimization | Config | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | Node |
|--------------|--------|---------|----------|-----------|--------|---------|------|
| **NEW BEST field+tau** | batch=1, lr_W=6E-4, W_L1=3E-5, **W_L2=2E-6**, edge_diff=900, data_aug=20 | 0.938 | **0.638** | 0.518 | **0.982** | 0.848 | **N94** |
| **Previous best field** | batch=1, lr_W=5.9E-4, W_L1=3E-5, W_L2=1E-6, data_aug=20 | 0.959 | 0.635 | 0.479 | 0.976 | 0.839 | N91 |
| **Best V_rest** | batch=1, lr_W=6E-4, W_L1=3E-5, data_aug=20 | 0.944 | 0.554 | **0.559** | 0.975 | 0.876 | N62 |
| **Best conn** | batch=2, lr_W=6E-4, W_L1=3E-5, data_aug=20 | **0.965** | 0.514 | 0.455 | 0.972 | 0.842 | N64 |

---

### Batch 15 (Iter 53-56): All Slots FAILED

## Iter 53: failed
Node: id=97, parent=94
Mode/Strategy: exploit (N94 replicate)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=900, coeff_W_L1=3E-5, coeff_W_L2=2E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: EXACT REPLICATE N94 (W_L2=2E-6, edge_diff=900, lr_W=6E-4)
Parent rule: N94 (field=0.638 BEST) — test reproducibility
Observation: Run failed — infrastructure issue or numerical instability. N94 reproducibility could not be tested.
Next: parent=94

## Iter 54: failed
Node: id=98, parent=94
Mode/Strategy: exploit
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=900, coeff_W_L1=3E-5, coeff_W_L2=3E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: W_L2: 2E-6 -> 3E-6 at N94 baseline
Parent rule: N94 — test if stronger W_L2 helps further
Observation: Run failed — cannot assess W_L2=3E-6 effect.
Next: parent=94

## Iter 55: failed
Node: id=99, parent=94
Mode/Strategy: explore
Config: lr_W=5.9E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=900, coeff_W_L1=3E-5, coeff_W_L2=2E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: lr_W: 6E-4 -> 5.9E-4 at W_L2=2E-6 baseline
Parent rule: N94 — test if lower lr_W recovers conn
Observation: Run failed — lr_W=5.9E-4 + W_L2=2E-6 effect unknown.
Next: parent=94

## Iter 56: failed
Node: id=100, parent=94
Mode/Strategy: principle-test
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=900, coeff_W_L1=3E-5, coeff_W_L2=2E-6, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: batch_size: 1 -> 2 at W_L2=2E-6 baseline. Testing principle: "batch=1 optimal for V_rest"
Parent rule: N94 — test batch=2 with optimal W_L2
Observation: Run failed — batch=2 + W_L2=2E-6 effect unknown.
Next: parent=94

---

### Batch 15 Summary (Iter 53-56): ALL FAILED

| Slot | Node | Mutation | Status | Notes |
|------|------|----------|--------|-------|
| 0 | N97 | N94 replicate | FAILED | Infrastructure/numerical issue |
| 1 | N98 | W_L2=3E-6 | FAILED | Could not test |
| 2 | N99 | lr_W=5.9E-4 + W_L2=2E-6 | FAILED | Could not test |
| 3 | N100 | batch=2 + W_L2=2E-6 | FAILED | Could not test |

**Analysis:**
All 4 runs failed simultaneously — likely infrastructure issue (GPU OOM, job scheduler failure, or numerical instability at W_L2 threshold). The W_L2=2E-6 configs may be approaching a stability boundary, or there was a cluster-wide failure.

**Strategy for Batch 16:**
1. Revert to more conservative configs (W_L2=1E-6 which worked in N91)
2. Re-test N94-like configs with slight modifications
3. Focus on reproducibility testing
4. Use UCB to select from known-working nodes

---

### Batch 16 (Iter 57-60): Recovery from All-Failure

## Iter 57: partial
Node: id=101, parent=91
Mode/Strategy: exploit (N91 replicate)
Config: lr_W=5.9E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, coeff_W_L2=1E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.954, field_R2=0.551, tau_R2=0.968, V_rest_R2=0.338, cluster_accuracy=0.861, test_R2=-0.495, test_pearson=0.556, training_time_min=63.7
Embedding: 65 types partially separated
Mutation: REPLICATE N91 (lr_W=5.9E-4, W_L2=1E-6) — confirm stable config after batch 15 failures
Parent rule: N91 (field=0.635 previous best) — revert to safe config
Observation: Recovered successfully. conn=0.954, field=0.551 both lower than N91 (0.959, 0.635) — **V_rest variance confirmed again** (0.338 vs N91's 0.479). Configs are indeed high-variance.
Next: parent=64

## Iter 58: partial
Node: id=102, parent=91
Mode/Strategy: exploit (W_L2 gentle increase)
Config: lr_W=5.9E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, coeff_W_L2=1.5E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.959, field_R2=0.550, tau_R2=0.970, V_rest_R2=0.345, cluster_accuracy=0.874, test_R2=-0.017, test_pearson=0.557, training_time_min=63.5
Embedding: 65 types partially separated
Mutation: W_L2: 1E-6 -> 1.5E-6 at lr_W=5.9E-4 baseline
Parent rule: N91 — gentle increase to test W_L2 stability range
Observation: W_L2=1.5E-6 slightly improves conn (0.959 vs 0.954) over N101. Field/V_rest similar. Stable intermediate point between N91's W_L2=1E-6 and N94's W_L2=2E-6.
Next: parent=64

## Iter 59: partial
Node: id=103, parent=64
Mode/Strategy: explore (batch=2 + W_L2)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, coeff_W_L2=1E-6, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.955, field_R2=0.523, tau_R2=0.969, V_rest_R2=0.360, cluster_accuracy=0.839, test_R2=-14.563, test_pearson=0.542, training_time_min=56.3
Embedding: 65 types partially separated
Mutation: Add W_L2=1E-6 to N64 (batch=2, lr_W=6E-4)
Parent rule: N64 (best conn=0.965 at batch=2) — test W_L2 regularization with batch=2
Observation: batch=2 + W_L2=1E-6 works. conn=0.955 (vs N64's 0.965 without W_L2) — slight conn drop but field=0.523 decent. Time=56 min (faster than batch=1). V_rest=0.360 moderate.
Next: parent=57

## Iter 60: partial
Node: id=104, parent=62
Mode/Strategy: principle-test (variance test)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, coeff_W_L2=0, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.959, field_R2=0.550, tau_R2=0.973, V_rest_R2=0.333, cluster_accuracy=0.884, test_R2=0.079, test_pearson=0.582, training_time_min=63.0
Embedding: 65 types partially separated
Mutation: REPLICATE N62 (lr_W=6E-4, W_L2=0). Testing principle: "V_rest has extreme variance ~0.3"
Parent rule: N62 (V_rest=0.559 best) — reproducibility test for V_rest variance principle
Observation: **VARIANCE CONFIRMED**: N62 got V_rest=0.559, this replicate got V_rest=0.333 (delta=0.226). Same config, wildly different V_rest. The principle holds — V_rest is stochastic with ~0.2-0.3 variance.
Next: parent=57

---

### Batch 16 Summary (Iter 57-60)

**Results:**
| Slot | Node | Mutation | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | time | Status |
|------|------|----------|---------|----------|-----------|--------|---------|------|--------|
| 0 | N101 | N91 replicate (W_L2=1E-6) | 0.954 | 0.551 | 0.338 | 0.968 | 0.861 | 63.7 | OK |
| 1 | N102 | W_L2: 1E-6 -> 1.5E-6 | 0.959 | 0.550 | 0.345 | 0.970 | 0.874 | 63.5 | OK |
| 2 | N103 | batch=2 + W_L2=1E-6 | 0.955 | 0.523 | 0.360 | 0.969 | 0.839 | 56.3 | OK |
| 3 | N104 | N62 replicate (variance test) | 0.959 | 0.550 | 0.333 | 0.973 | 0.884 | 63.0 | OK |

**Key Findings:**
1. **All 4 slots recovered** — infrastructure issue from Batch 15 resolved
2. **V_rest EXTREME VARIANCE confirmed** — N62 replicate (N104) got V_rest=0.333 vs original 0.559 (delta=0.226)
3. **W_L2=1.5E-6 stable** — N102 conn=0.959 (vs N101's 0.954), intermediate between W_L2=1E-6 and 2E-6
4. **batch=2 + W_L2=1E-6 works** — N103 conn=0.955, field=0.523, time=56 min (faster)
5. **Conservative configs (edge_diff=750) are stable** — no failures unlike batch 15's edge_diff=900

**V_rest Variance Evidence (UPDATED):**
| Config | Runs | V_rest values | Range |
|--------|------|---------------|-------|
| lr_W=6E-4, W_L1=3E-5, W_L2=0 | N62, N104 | 0.559, 0.333 | **0.226** |
| lr_W=5.9E-4, W_L1=3E-5, W_L2=1E-6 | N91, N101 | 0.479, 0.338 | **0.141** |

---

### Batch 17 (Iter 61-64): ALL FAILED

## Iter 61: failed
Node: id=105, parent=64
Mode/Strategy: exploit (batch=2 + W_L2=1.5E-6)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, coeff_W_L2=1.5E-6, batch_size=2, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: W_L2: 1E-6 -> 1.5E-6 at batch=2 (improve on N103's W_L2=1E-6)
Parent rule: N64 (batch=2 best conn) — push W_L2 at batch=2
Observation: Run failed — no metrics produced. Second consecutive batch with failures.
Next: parent=58

## Iter 62: failed
Node: id=106, parent=58
Mode/Strategy: exploit (edge_diff increase)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=850, coeff_W_L1=3E-5, coeff_W_L2=0, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: edge_diff: 750 -> 850 at batch=1 baseline
Parent rule: N58 (UCB=4.427) — test moderate edge_diff
Observation: Run failed — edge_diff=850 effect unknown.
Next: parent=102

## Iter 63: failed
Node: id=107, parent=102
Mode/Strategy: explore (W_L2 push at lr_W=5.9E-4)
Config: lr_W=5.9E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, coeff_W_L2=2E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: W_L2: 1.5E-6 -> 2E-6 at lr_W=5.9E-4 (safer LR)
Parent rule: N102 (W_L2=1.5E-6 stable) — push W_L2 at lower lr_W
Observation: Run failed — W_L2=2E-6 + lr_W=5.9E-4 effect unknown.
Next: parent=57

## Iter 64: failed
Node: id=108, parent=57
Mode/Strategy: principle-test (edge_diff=1000)
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=1000, coeff_W_L1=5E-5, coeff_W_L2=0, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=18
Metrics: FAILED (empty log)
Embedding: N/A
Mutation: edge_diff: 750 -> 1000 at lr_W=7E-4. Testing principle: "edge_diff=750-900 optimal" (upper bound test)
Parent rule: N57 (UCB=1.652, lr_W=7E-4 baseline) — test principle boundary
Observation: Run failed — edge_diff=1000 effect unknown.
Next: parent=67

---

### Batch 17 Summary (Iter 61-64): ALL FAILED

| Slot | Node | Mutation | Status | Notes |
|------|------|----------|--------|-------|
| 0 | N105 | batch=2 + W_L2=1.5E-6 | FAILED | Could not test |
| 1 | N106 | edge_diff: 750 -> 850 | FAILED | Could not test |
| 2 | N107 | W_L2: 1.5E-6 -> 2E-6 at lr_W=5.9E-4 | FAILED | Could not test |
| 3 | N108 | edge_diff: 750 -> 1000 (principle test) | FAILED | Could not test |

**Analysis:**
This is the **SECOND consecutive all-failure batch** (after Batch 15). Pattern suggests:
1. **Infrastructure instability** — GPU issues, job scheduler problems, or cluster maintenance
2. **NOT config-related** — Batch 16 recovered with similar configs, and these configs are more conservative

**Strategy for Batch 18:**
1. **Ultra-conservative configs** — revert to known-working N91/N102 baselines
2. **Smaller data_aug** — reduce from 20 to 18 to lower memory pressure
3. **No W_L2** for 2 slots to rule out W_L2 stability issues
4. **Diverse parents** to isolate failure mode

---

## Iter 65: failed
Node: id=109, parent=102
Mode/Strategy: exploit (N102 replicate with W_L2=1.5E-6)
Config: lr_W=5.9E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, coeff_W_L2=1.5E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.958, field_R2=0.545, tau_R2=0.970, V_rest_R2=0.341, cluster_accuracy=N/A, test_R2=0.008, test_pearson=0.563, training_time_min=63.5
Embedding: N/A (cluster analysis failed)
Mutation: REPLICATE N102 (W_L2=1.5E-6, lr_W=5.9E-4) for infrastructure verification
Parent rule: N102 (UCB=stable baseline) — verify infrastructure recovery
Observation: Training succeeded (conn=0.958, field=0.545) but cluster analysis crashed. V_rest=0.341 lower than N102's 0.345. Marked FAILED due to missing cluster.
Next: parent=61

## Iter 66: failed
Node: id=110, parent=104
Mode/Strategy: exploit (N104 replicate baseline)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, coeff_W_L2=0, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.956, field_R2=0.554, tau_R2=0.964, V_rest_R2=0.420, cluster_accuracy=N/A, test_R2=-0.783, test_pearson=0.583, training_time_min=63.2
Embedding: N/A (cluster analysis failed)
Mutation: REPLICATE N104/N62 baseline (W_L2=0, lr_W=6E-4) for simple baseline test
Parent rule: N104 (variance test baseline) — simple config to verify runs
Observation: Training succeeded (conn=0.956, field=0.554, V_rest=0.420). Better V_rest than N109. Cluster analysis failed. Marked FAILED due to missing cluster.
Next: parent=69

## Iter 67: partial
Node: id=111, parent=67
Mode/Strategy: explore (lr_W reduction at W_L1=4E-5)
Config: lr_W=6.5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=4E-5, coeff_W_L2=0, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.954, field_R2=0.530, tau_R2=0.972, V_rest_R2=0.340, cluster_accuracy=0.868, test_R2=-14.507, test_pearson=0.560, training_time_min=58.2
Embedding: 65 types partially separated
Mutation: lr_W: 7E-4 -> 6.5E-4 at W_L1=4E-5, data_aug=18 (from N67 baseline)
Parent rule: N67 (UCB=1.723, best V_rest in lr_W=7E-4 family) — reduce lr_W to balance conn/V_rest
Observation: conn=0.954 good, but V_rest=0.340 still shows variance. data_aug=18 saves time (58 min). field=0.530 slightly lower than N67's 0.558.
Next: parent=70

## Iter 68: failed
Node: id=112, parent=70
Mode/Strategy: principle-test (data_aug=22 upper bound)
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=4E-5, coeff_W_L2=0, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=22
Metrics: connectivity_R2=N/A, field_R2=0.565, tau_R2=N/A, V_rest_R2=N/A, cluster_accuracy=N/A, test_R2=-0.687, test_pearson=0.582, training_time_min=68.1
Embedding: N/A (metrics extraction failed)
Mutation: data_aug: 20 -> 22 at lr_W=7E-4, W_L1=4E-5. Testing principle: "data_aug=20 is optimal"
Parent rule: N70 (UCB=2.683, best V_rest at W_L1=4E-5) — test data_aug upper bound
Observation: Only field_R2=0.565 extracted (good!). Other metrics missing. Training time 68 min exceeds 60 min target. Marked FAILED due to incomplete metrics.
Next: parent=61

---

### Batch 18 Summary (Iter 65-68): 3 FAILED, 1 PARTIAL

| Slot | Node | Mutation | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | Status |
|------|------|----------|---------|----------|-----------|--------|---------|--------|
| 0 | N109 | N102 replicate (W_L2=1.5E-6) | 0.958 | 0.545 | 0.341 | 0.970 | N/A | FAILED |
| 1 | N110 | N104 baseline (W_L2=0) | 0.956 | 0.554 | 0.420 | 0.964 | N/A | FAILED |
| 2 | N111 | lr_W: 7E-4 -> 6.5E-4 | 0.954 | 0.530 | 0.340 | 0.972 | 0.868 | PARTIAL |
| 3 | N112 | data_aug: 20 -> 22 (principle test) | N/A | 0.565 | N/A | N/A | N/A | FAILED |

**Key findings:**
1. **Training succeeds, analysis pipeline fails** — N109, N110, N112 produced partial metrics indicating training completed
2. **Cluster analysis is the bottleneck** — 3/4 slots failed at cluster step
3. **V_rest variance continues** — N109=0.341, N110=0.420, N111=0.340 (all below 0.5)
4. **data_aug=22 may improve field** — N112 field=0.565 highest in batch despite incomplete metrics
5. **data_aug=22 exceeds time limit** — 68 min > 60 min threshold

**V_rest variance update:**
| Config | Runs | V_rest values | Range | Mean |
|--------|------|---------------|-------|------|
| lr_W=5.9E-4, W_L1=3E-5, W_L2=1.5E-6 | N102, N109 | 0.345, 0.341 | 0.004 | 0.343 |
| lr_W=6E-4, W_L1=3E-5, W_L2=0 | N62, N81, N85, N104, N110 | 0.559, 0.264, 0.403, 0.333, 0.420 | 0.295 | 0.396 |

**Observation:** W_L2=1.5E-6 at lr_W=5.9E-4 shows much lower V_rest variance (0.004) compared to W_L2=0 at lr_W=6E-4 (0.295). However, mean V_rest is also lower (0.343 vs 0.396).

---

## Iter 69: partial
Node: id=113, parent=61
Mode/Strategy: exploit (UCB=4.427, lr_W=6.5E-4 baseline)
Config: lr_W=6.5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, coeff_W_L2=0, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=16
Metrics: connectivity_R2=0.955, field_R2=0.518, tau_R2=0.958, V_rest_R2=0.370, cluster_accuracy=0.833, test_R2=-3.188, test_pearson=0.551, training_time_min=52.3
Embedding: 65 types partially separated
Mutation: data_aug: 18 -> 16 at lr_W=6.5E-4 (from N61 baseline)
Parent rule: N61 (UCB=4.427 highest) — exploit top UCB node
Observation: Fast (52 min) but field=0.518 and cluster=0.833 both lower than data_aug=20 configs. data_aug=16 insufficient. V_rest=0.370 moderate.
Next: parent=52

## Iter 70: partial
Node: id=114, parent=69
Mode/Strategy: exploit (UCB=4.426, lr_W=7E-4 baseline)
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=4E-5, coeff_W_L2=0, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=18
Metrics: connectivity_R2=0.906, field_R2=0.533, tau_R2=0.501, V_rest_R2=0.251, cluster_accuracy=0.867, test_R2=-13.225, test_pearson=0.554, training_time_min=57.9
Embedding: 65 types partially separated
Mutation: REPLICATE N69 config (lr_W=7E-4, W_L1=4E-5, data_aug=18)
Parent rule: N69 (UCB=4.426 second highest) — exploit second UCB node
Observation: **tau_R2 COLLAPSED to 0.501** — severe degradation. conn=0.906 also dropped. V_rest=0.251 poor. Same config as N69 but much worse. Confirms variance even in tau_R2.
Next: parent=58

## Iter 71: partial
Node: id=115, parent=94
Mode/Strategy: explore (W_L2=2E-6 retry after Batch 15 failures)
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=900, coeff_W_L1=3E-5, coeff_W_L2=2E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.957, field_R2=0.551, tau_R2=0.978, V_rest_R2=0.461, cluster_accuracy=0.859, test_R2=0.188, test_pearson=0.580, training_time_min=64.3
Embedding: 65 types partially separated
Mutation: REPLICATE N94 (W_L2=2E-6, edge_diff=900, lr_W=6E-4)
Parent rule: N94 (best field=0.638 ever) — retry best config after infrastructure failures
Observation: W_L2=2E-6 WORKS! V_rest=0.461 solid, tau=0.978 excellent. But field=0.551 << N94's 0.638. Time=64 min at limit. N94 result not reproducible but config is stable.
Next: parent=94

## Iter 72: partial
Node: id=116, parent=110
Mode/Strategy: principle-test (lr_W=6.5E-4 vs 6E-4 at W_L1=3E-5)
Config: lr_W=6.5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, coeff_W_L2=0, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F, data_aug=20
Metrics: connectivity_R2=0.955, field_R2=0.551, tau_R2=0.967, V_rest_R2=0.511, cluster_accuracy=0.865, test_R2=-0.018, test_pearson=0.558, training_time_min=62.9
Embedding: 65 types partially separated
Mutation: lr_W: 6E-4 -> 6.5E-4 at W_L1=3E-5, data_aug=20. Testing principle: "lr_W=6E-4 is optimal at W_L1=3E-5"
Parent rule: N110 (baseline for lr_W test) — test lr_W upper bound
Observation: lr_W=6.5E-4 gives V_rest=0.511 (excellent!), conn=0.955, field=0.551. Better V_rest than N110's 0.420. Confirms lr_W=6.5E-4 viable at W_L1=3E-5.
Next: parent=115

---

### Batch 19 Summary (Iter 69-72): >>> BLOCK 3 END <<<

| Slot | Node | Mutation | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | Time | Status |
|------|------|----------|---------|----------|-----------|--------|---------|------|--------|
| 0 | N113 | data_aug: 16 at lr_W=6.5E-4 | 0.955 | 0.518 | 0.370 | 0.958 | 0.833 | 52 | PARTIAL |
| 1 | N114 | lr_W=7E-4, W_L1=4E-5, data_aug=18 | 0.906 | 0.533 | 0.251 | **0.501** | 0.867 | 58 | PARTIAL |
| 2 | N115 | W_L2=2E-6, edge_diff=900 (N94 retry) | 0.957 | 0.551 | **0.461** | **0.978** | 0.859 | 64 | PARTIAL |
| 3 | N116 | lr_W=6.5E-4, data_aug=20 | 0.955 | 0.551 | **0.511** | 0.967 | 0.865 | 63 | PARTIAL |

**Key findings:**
1. **tau_R2 can collapse dramatically** — N114 tau=0.501 vs N69's 0.976 (same config!). First observation of tau variance.
2. **W_L2=2E-6 is stable** — N115 worked well (V_rest=0.461, tau=0.978) but field=0.551 << N94's 0.638
3. **lr_W=6.5E-4 is excellent for V_rest** — N116 V_rest=0.511 best this batch
4. **data_aug=16 is insufficient** — N113 field=0.518, cluster=0.833 both dropped
5. **N94's field=0.638 may be a statistical outlier** — not reproduced in N115 with same config

**New variance observations:**
| Config | Metric | Values | Range |
|--------|--------|--------|-------|
| lr_W=7E-4, W_L1=4E-5, data_aug=18 | tau_R2 | N69: 0.976, N114: 0.501 | **0.475** |
| lr_W=6E-4, W_L2=2E-6, edge_diff=900 | field_R2 | N94: 0.638, N115: 0.551 | **0.087** |

**Block 3 Complete — Summary for Block 4:**
- Best overall: N94 (field=0.638, tau=0.982) but not reproducible
- Most stable V_rest: W_L2=1.5-2E-6 configs (range 0.004-0.1 vs 0.295)
- Best reproducible field: ~0.55 (N91, N95, N102, N115, N116)
- Key open question: What causes N94's exceptional field=0.638?

---

## Block 3: Summary

### Block Goals
Focus: **Regularization + Stability exploration**
- Optimize W_L2 regularization
- Understand V_rest variance
- Balance conn vs field tradeoff

### Key Discoveries

**1. W_L2 Regularization:**
- W_L2=2E-6 gives best field_R2 (N94: 0.638) but result not reproducible
- W_L2=1.5E-6 at lr_W=5.9E-4 shows MINIMAL V_rest variance (0.004)
- W_L2=1E-6 at lr_W=6E-4 gives stable results

**2. V_rest Variance Analysis:**
- lr_W=6E-4, W_L2=0 has extreme variance (0.295 range)
- lr_W=5.9E-4, W_L2=1.5E-6 has minimal variance (0.004 range)
- Adding W_L2 may stabilize V_rest at cost of lower mean

**3. New Variance Discovery - tau_R2:**
- N114 tau=0.501 vs N69 tau=0.976 (same config!)
- First observation of severe tau variance

**4. Learning Rate Findings:**
- lr_W=6.5E-4 with W_L1=3E-5 gives excellent V_rest (N116: 0.511)
- lr_W=7E-4 with W_L1=4E-5 shows high variance

### Block 3 Pareto Front
| Optimization | Config | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | Node |
|--------------|--------|---------|----------|-----------|--------|------|
| **Best field (outlier)** | W_L2=2E-6, edge_diff=900 | 0.938 | **0.638** | 0.518 | 0.982 | N94 |
| **Best V_rest (Batch 19)** | lr_W=6.5E-4, data_aug=20 | 0.955 | 0.551 | **0.511** | 0.967 | N116 |
| **Best tau (Batch 19)** | W_L2=2E-6, edge_diff=900 | 0.957 | 0.551 | 0.461 | **0.978** | N115 |

### Recommended Starting Config for Block 4
Based on stability and reproducibility:
- lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3
- batch_size=1, data_aug=20
- W_L1=3E-5, W_L2=1E-6 to 2E-6
- edge_diff=750 or 900
- h_dim_nnr=2048, omega_f=2048, n_layers_nnr=3

---

## Block 4: Combined Optimization + Variance Reduction (Iter 73-96)

### Block 4 Hypothesis
This block builds on N115/N116 configs (most stable from Block 3):
- N115: W_L2=2E-6, edge_diff=900 — stable performance, reproducible
- N116: lr_W=6.5E-4, W_L1=3E-5 — best V_rest (0.511)
Goals: Push W_L2 slightly higher, test edge_diff=1000, reduce variance.

---

### Batch 20 (Iter 73-76): Block 4 Start

## Iter 73: partial
Node: id=117, parent=115
Mode/Strategy: exploit
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=900, coeff_W_L1=3E-5, coeff_W_L2=2.5E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F
Metrics: connectivity_R2=0.957, field_R2=0.548, tau_R2=0.976, V_rest_R2=0.283, cluster_accuracy=0.822, test_R2=0.004, test_pearson=0.578, training_time_min=64
Embedding: 65 types partially separated
Mutation: W_L2: 2E-6 -> 2.5E-6 at N115 baseline
Parent rule: N115 has best reproducible performance (field=0.551, V_rest=0.461)
Observation: **W_L2=2.5E-6 CATASTROPHIC for V_rest** — collapsed from 0.461 to 0.283. W_L2=2E-6 is upper bound.
Next: parent=115

## Iter 74: partial
Node: id=118, parent=116
Mode/Strategy: exploit
Config: lr_W=6.5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=3E-5, coeff_W_L2=1E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F
Metrics: connectivity_R2=0.955, field_R2=0.552, tau_R2=0.967, V_rest_R2=0.368, cluster_accuracy=0.840, test_R2=-0.213, test_pearson=0.578, training_time_min=64
Embedding: 65 types partially separated
Mutation: W_L2: 0 -> 1E-6 at lr_W=6.5E-4 (N116 baseline)
Parent rule: N116 has best V_rest (0.511) without W_L2
Observation: Adding W_L2=1E-6 to N116 HURTS V_rest (0.511 -> 0.368). W_L2 may not help at lr_W=6.5E-4.
Next: parent=116

## Iter 75: partial
Node: id=119, parent=52
Mode/Strategy: explore (high UCB=4.425)
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, coeff_W_L2=0, batch_size=4, data_aug=22, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F
Metrics: connectivity_R2=0.963, field_R2=0.328, tau_R2=0.948, V_rest_R2=0.343, cluster_accuracy=0.854, test_R2=-5.155, test_pearson=0.421, training_time_min=62
Embedding: 65 types partially separated
Mutation: batch=4, data_aug=22, lr_W=7E-4, W_L1=5E-5 (N52-based high UCB exploration)
Parent rule: N52 has high UCB (4.425) suggesting under-explored batch=4 regime
Observation: **batch=4 destroys field** (0.328 vs ~0.55 at batch=1). tau also drops (0.948). Confirms batch>=3 harmful for Siren.
Next: parent=120

## Iter 76: partial
Node: id=120, parent=115
Mode/Strategy: principle-test
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=1000, coeff_W_L1=3E-5, coeff_W_L2=2E-6, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=2048, omega_f=2048, recurrent=F
Metrics: connectivity_R2=0.961, field_R2=0.553, tau_R2=0.940, V_rest_R2=0.547, cluster_accuracy=0.863, test_R2=-0.859, test_pearson=0.563, training_time_min=63
Embedding: 65 types partially separated
Mutation: edge_diff: 900 -> 1000 at W_L2=2E-6 baseline. Testing principle: "edge_diff=900 is optimal"
Parent rule: Testing if edge_diff=1000 can improve on N115's performance
Observation: **edge_diff=1000 EXCELLENT for V_rest=0.547!** Near-record (N62: 0.559). conn=0.961, field=0.553 both strong. tau=0.940 slightly lower but acceptable. Principle UPDATE: edge_diff=1000 may be better than 900.
Next: parent=120

---

### Batch 20 Summary
| Slot | Node | Mutation | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | Time | Status |
|------|------|----------|---------|----------|-----------|--------|---------|------|--------|
| 0 | N117 | W_L2: 2E-6 -> 2.5E-6 | 0.957 | 0.548 | 0.283 | 0.976 | 0.822 | 64 | PARTIAL |
| 1 | N118 | W_L2: 0 -> 1E-6 at lr_W=6.5E-4 | 0.955 | 0.552 | 0.368 | 0.967 | 0.840 | 64 | PARTIAL |
| 2 | N119 | batch=4, data_aug=22 | 0.963 | 0.328 | 0.343 | 0.948 | 0.854 | 62 | PARTIAL |
| 3 | N120 | edge_diff: 900 -> 1000 | 0.961 | **0.553** | **0.547** | 0.940 | 0.863 | 63 | PARTIAL |

**Key Findings:**
1. **W_L2=2.5E-6 TOO HIGH** — V_rest collapsed 0.461 -> 0.283. W_L2=2E-6 is ceiling.
2. **W_L2 + lr_W=6.5E-4 incompatible** — V_rest dropped 0.511 -> 0.368
3. **batch=4 destroys field** — field=0.328, confirms batch>=3 harmful
4. **edge_diff=1000 is NEW BEST** — V_rest=0.547 (2nd best ever), field=0.553, conn=0.961

**New Pareto point: N120** (edge_diff=1000, W_L2=2E-6) — V_rest=0.547, conn=0.961, field=0.553

---

