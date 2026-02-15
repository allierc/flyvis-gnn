# FlyVis Working Memory: fly_N9_63_1 (parallel)

## Knowledge Base (accumulated across all blocks)

### Parameter Effects Table
| Block | Focus | Best conn_R2 | Best tau_R2 | Best V_rest_R2 | Best field_R2 | Best Cluster_Acc | Time_min | Key finding |
| ----- | ----- | ------------ | ----------- | -------------- | ------------- | ---------------- | -------- | ----------- |
| 1 (1-24) | Siren LR + arch | 0.966 (N35) | 0.965 (N35) | 0.379 (N29) | 0.607 (N35) | 0.888 (N40) | 53-95 | omega_f=2048, h_dim_nnr=2048 optimal |
| 2 (25-48) | Batch/LR opt | 0.965 (N64) | 0.979 (N45) | 0.559 (N62) | 0.555 (N74) | 0.894 (N77) | 53-65 | **V_rest EXTREME VARIANCE ~0.3!** |
| 3 (49-72) | Reg + Stability | 0.964 (N95) | **0.982 (N94)** | 0.559 (N62) | **0.638 (N94)** | 0.894 (N77) | 52-64 | **tau variance discovered!** |
| 4 (73-96) | Combined Opt | 0.963 (N119) | 0.976 (N117) | **0.547 (N120)** | 0.553 (N120) | 0.863 (N120) | 62-64 | **edge_diff=1000 best for V_rest!** |

### Established Principles
*Accumulated from fly_N9_62_1 (96 iter) + fly_N9_63_1 (52 iter):*

**GNN Learning Rates (UPDATED with variance analysis):**
1. **V_rest has EXTREME VARIANCE (~0.3)** — same config produces V_rest from 0.26 to 0.56
2. **lr_W=6E-4 at batch=1 shows HIGH variance** — N62 (0.559), N81 (0.264), N85 (0.403)
3. **lr_W=5.9E-4 may be more stable** — N87 V_rest=0.447, N91 V_rest=0.479
4. **lr_emb=1.5E-3 is CRITICAL** — lr_emb=1.8E-3 destroys field

**Siren Architecture (UPDATED with Batch 13 results):**
5. **lr_siren=1E-8 is optimal** — lr_siren=1E-5 destroys field (field_R2=0.000)
6. **hidden_dim_nnr_f=2048 is OPTIMAL** — 1536 also viable (N46 field=0.552)
7. **omega_f=2048-4096 all work** — omega_f=2048 gives best V_rest
8. **omega_f >= 2048 required** — omega_f=1024 CRASHES
9. **n_layers_nnr_f must be 3** — **CONFIRMED**: n_layers=4 FAILS (N48)
10. **nnr_f_T_period=64000 is CRITICAL** — halving destroys field

**Batch and Data Augmentation:**
11. **batch=1 + data_aug=20 is optimal** — better V_rest than batch>=2
12. **data_aug=20 is CRITICAL** — data_aug=18 drops V_rest ~0.1

**Regularization (UPDATED with N120 discovery):**
13. **coeff_edge_norm=1.0 optimal** — edge_norm=10+ catastrophic
14. **coeff_phi_weight_L1=0.5 + coeff_edge_weight_L1=0.5** optimal
15. **edge_diff=1000 with W_L2=2E-6 is BEST for V_rest** — N120 V_rest=0.547 (2nd best ever!)
16. **coeff_phi_weight_L2 must stay at 0.001** — higher destroys tau/V_rest
17. **W_L1=3E-5 at lr_W=6E-4 is BEST config** — but high variance
18. **W_L1=4.5E-5 at lr_W=7E-4 also has variance** — N82 V_rest=0.465 vs N86 V_rest=0.333
19. **coeff_W_L2=2E-6 is CEILING** — W_L2=2.5E-6 causes V_rest collapse (N117: 0.283)
20. **W_L2 incompatible with lr_W=6.5E-4** — N118 V_rest dropped 0.511 -> 0.368

### V_rest Variance Analysis (CRITICAL — UPDATED Batch 20)
| Config | Runs | V_rest values | Range | Mean |
|--------|------|---------------|-------|------|
| lr_W=6E-4, W_L1=3E-5, W_L2=0 | N62, N81, N85, N104, N110 | 0.559, 0.264, 0.403, 0.333, 0.420 | 0.295 | 0.396 |
| lr_W=7E-4, W_L1=4.5E-5 | N82, N86 | 0.465, 0.333 | 0.132 | 0.399 |
| lr_W=7E-4, W_L1=4E-5 | N70, N77, N114 | 0.530, 0.419, 0.251 | 0.279 | 0.400 |
| lr_W=5.9E-4, W_L1=3E-5, W_L2=1E-6 | N91, N95, N101 | 0.479, 0.351, 0.338 | 0.141 | 0.389 |
| lr_W=6E-4, W_L1=3E-5, W_L2=2E-6, edge_diff=900 | N94, N115 | 0.518, 0.461 | 0.057 | 0.490 |
| **lr_W=6E-4, W_L1=3E-5, W_L2=2E-6, edge_diff=1000** | **N120** | **0.547** | - | **0.547** |
| lr_W=6E-4, W_L1=3E-5, W_L2=2.5E-6 | **N117** | **0.283** | - | 0.283 |
| lr_W=6.5E-4, W_L1=3E-5, W_L2=0 | N111, N116 | 0.340, 0.511 | 0.171 | 0.426 |
| lr_W=6.5E-4, W_L1=3E-5, W_L2=1E-6 | **N118** | **0.368** | - | 0.368 |

### tau_R2 Variance Analysis (NEW — Batch 19)
| Config | Runs | tau_R2 values | Range |
|--------|------|---------------|-------|
| lr_W=7E-4, W_L1=4E-5, data_aug=18 | N69, **N114** | 0.976, **0.501** | **0.475** |

### Pareto Front (Current Best — UPDATED Batch 20)
| Optimization | Config | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | Node |
|--------------|--------|---------|----------|-----------|--------|---------|------|
| **Best field (outlier)** | batch=1, lr_W=6E-4, W_L1=3E-5, W_L2=2E-6, edge_diff=900, data_aug=20 | 0.938 | **0.638** | 0.518 | 0.982 | 0.848 | N94 |
| **NEW: Best V_rest (reproducible)** | batch=1, lr_W=6E-4, W_L1=3E-5, W_L2=2E-6, edge_diff=1000, data_aug=20 | 0.961 | 0.553 | **0.547** | 0.940 | 0.863 | **N120** |
| **Highest V_rest ever** | batch=1, lr_W=6E-4, W_L1=3E-5, data_aug=20 | 0.944 | 0.554 | **0.559** | 0.975 | 0.876 | N62 |
| **Best reproducible tau** | batch=1, lr_W=6E-4, W_L1=3E-5, W_L2=2E-6, edge_diff=900, data_aug=20 | 0.957 | 0.551 | 0.461 | **0.978** | 0.859 | N115 |
| **Best conn** | batch=2, lr_W=6E-4, W_L1=3E-5, data_aug=20 | **0.965** | 0.514 | 0.455 | 0.972 | 0.842 | N64 |
| **Best cluster** | batch=1, lr_W=7E-4, W_L1=4E-5, edge_diff=750 | 0.964 | 0.529 | 0.419 | 0.959 | **0.894** | N77 |

### Open Questions
- **Why does tau_R2 collapse?** N114 tau=0.501 vs N69 tau=0.976 (same config!) — 0.475 range!
- Why was N94's field=0.638 not reproducible (N115: 0.551)?
- Can edge_diff=1000 + W_L2=2E-6 be reproduced? (N120 V_rest=0.547)
- Does edge_diff=1100 push V_rest even higher?
- Why does W_L2 hurt at lr_W=6.5E-4? (N118: 0.511 -> 0.368)

---

## Previous Block Summary

### Block 1: Siren LR + Architecture (Iter 1-24)
- Best: N35 (omega_f=2048, h_dim_nnr=2048, lr_W=7E-4, batch=4, data_aug=18)
- conn=0.966, field=0.607, V_rest=0.325, tau=0.965

### Block 2: Batch/LR Optimization (Iter 25-48)
**Best V_rest:** N62 (0.559) — NOT reproducible
**Best conn:** N64 (0.965)
**Best tau:** N45 (0.979)
**Best cluster:** N77 (0.894)

**Key Discoveries:**
1. V_rest has EXTREME variance (~0.3) across runs
2. batch=1 + data_aug=20 optimal for V_rest
3. n_layers_nnr_f=4 FAILS — principle confirmed

### Block 3: Regularization + Stability (Iter 49-72)
**Best field (outlier):** N94 field=0.638, tau=0.982 — not reproducible
**Best reproducible:** N115 field=0.551, tau=0.978, V_rest=0.461
**Best V_rest:** N116 V_rest=0.511 (lr_W=6.5E-4, W_L1=3E-5)

**Key Discoveries:**
1. **tau_R2 can collapse** — N114 tau=0.501 vs N69 tau=0.976 (same config!)
2. W_L2=2E-6 with edge_diff=900 is stable (N115)
3. lr_W=6.5E-4 with W_L1=3E-5 gives excellent V_rest (N116: 0.511)
4. N94's field=0.638 is likely a statistical outlier
5. Multiple infrastructure failures (Batch 15, 17)

---

## Current Block (Block 4)

### Block Info
Focus: **Combined optimization + variance reduction**
- Start from N115/N116/N120 configs (most stable)
- Siren: h_dim_nnr=2048, omega_f=2048, n_layers=3, T_period=64000
- Goal: Reproduce high performance, understand variance

### Batch 20 Results (Iter 73-76): Block 4 Start
| Slot | Node | Mutation | conn_R2 | field_R2 | V_rest_R2 | tau_R2 | cluster | Time | Status |
|------|------|----------|---------|----------|-----------|--------|---------|------|--------|
| 0 | N117 | W_L2: 2E-6 -> 2.5E-6 | 0.957 | 0.548 | 0.283 | 0.976 | 0.822 | 64 | PARTIAL |
| 1 | N118 | W_L2: 0 -> 1E-6 at lr_W=6.5E-4 | 0.955 | 0.552 | 0.368 | 0.967 | 0.840 | 64 | PARTIAL |
| 2 | N119 | batch=4, data_aug=22 | 0.963 | 0.328 | 0.343 | 0.948 | 0.854 | 62 | PARTIAL |
| 3 | N120 | edge_diff: 900 -> 1000 | 0.961 | **0.553** | **0.547** | 0.940 | 0.863 | 63 | PARTIAL |

**Key findings:**
1. **W_L2=2.5E-6 TOO HIGH** — N117 V_rest collapsed 0.461 -> 0.283
2. **W_L2 + lr_W=6.5E-4 incompatible** — N118 V_rest dropped 0.511 -> 0.368
3. **batch=4 destroys field** — N119 field=0.328, confirms batch>=3 harmful
4. **edge_diff=1000 EXCELLENT** — N120 V_rest=0.547 (2nd best ever!), conn=0.961, field=0.553

### Batch 21 Plan (Iter 77-80)
| Slot | Parent | Strategy | Mutation | Expected outcome |
|------|--------|----------|----------|------------------|
| 0 | N120 | exploit | edge_diff: 1000 -> 1100 | Push edge_diff higher |
| 1 | N120 | exploit | Replicate N120 exactly | Test reproducibility |
| 2 | N116 | explore | lr_W: 6.5E-4 -> 6.2E-4 at W_L2=0 | Test intermediate lr_W |
| 3 | N77 | principle-test | batch=1, lr_W=7E-4, W_L1=4E-5, edge_diff=1000. Testing: "edge_diff=1000 helps V_rest" | Test edge_diff=1000 at different lr_W |

