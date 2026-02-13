# FlyVis Working Memory: fly_N9_62_0 (parallel)

## Knowledge Base (accumulated across all blocks)

### Parameter Effects Table
| Block | Focus | Best conn_R2 | Best tau_R2 | Best V_rest_R2 | Best Cluster_Acc | Time_min | Key finding |
| ----- | ----- | ------------ | ----------- | -------------- | ---------------- | -------- | ----------- |
| 1 (iter1-24) | lr exploration | 0.823 (N21) | 0.689 (N9) | 0.272 (N14) | 0.754 (N7) | ~49 | lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3 optimal |
| 2 (iter25-48) | reg + arch | 0.867 (N44) | 0.752 (N38) | 0.349 (N45) | 0.772 (N42) | ~55 | n_layers=4, hidden_dim_update=96 optimal |
| 3 (iter49-72) | batch/recurrent | 0.889 (N66) | 0.911 (N50) | 0.411 (N66) | 0.793 (N49) | 53-70 | N66 BREAKTHROUGH: aug=29, n_layers=3, n_layers_update=4, lr_emb=4E-3 |
| 4 (iter73-96) | combined opt | **0.897 (N83)** | **0.895 (N96)** | **0.465 (N90)** | **0.796 (N92)** | ~62-65 | **Four optimization paths**: N83 conn, N96 tau, N90 V_rest, N92 cluster |

### Established Principles
1. Higher MLP learning rates (lr=1E-3) and embedding learning rates (lr_emb>=2E-3) improve connectivity, tau, and V_rest recovery vs defaults
2. lr_W > 1E-3 damages tau recovery - keep lr_W <= 1E-3 (CONFIRMED iter 8: lr_W=1.5E-3 gave test_R2=-192)
3. lr_emb=3.5E-3 with lr_W=5E-4 achieves best balanced metrics (SUPERSEDED by block 4 findings)
4. lr_W=5E-4 is optimal - lower lr_W=3E-4 does NOT help (iter 15: tau=0.565), lr_W=7E-4 not better (iter 20: tau=0.532)
5. lr > 1E-3 causes instability (CONFIRMED iter 12: lr=2E-3 gave test_R2=-inf)
6. lr < 1E-3 causes instability (CONFIRMED iter 24: lr=8E-4 gave test_R2=-453, V_rest collapsed)
7. lr_emb=4E-3 helps V_rest_R2 ONLY with batch_size=1 (iter 45: V_rest=0.349; iter 52: batch_size=2 gives V_rest=0.042)
8. **lr_emb > 4E-3 hurts all metrics** (CONFIRMED iter 70: lr_emb=4.5E-3 caused tau=0.474, V_rest=0.142)
9. **lr_emb DISCRETE SWEET SPOTS**: 3.75E-3 for conn/tau, 4E-3 for V_rest/cluster - midpoints (3.875E-3) FAIL (N74)
10. **coeff_edge_diff DEPENDS on lr_emb**: 600 with lr_emb=3.75E-3 (N83 conn=0.897), 610-625 with lr_emb=4E-3
11. coeff_edge_diff < 600 hurts conn severely - 575 gives 0.746 (N88)
12. **coeff_edge_norm=975 OPTIMAL for tau with lr_emb=3.75E-3** - N96 tau=0.895. edge_norm=980 FAILS (N97: tau=0.697)
13. **coeff_W_L1=7.5E-5 for V_rest path** (N90: V_rest=0.465), **5E-5 for cluster/conn paths** (N92: cluster=0.796)
14. **W_L1=6E-5 is SUBOPTIMAL** - neither 5E-5 nor 7.5E-5 benefits (N95: tau drops to 0.653)
15. **hidden_dim=96 causes instability** - HARMFUL for edge MLP (iter 37: test_R2=-5.95)
16. **hidden_dim_update=96 is ESSENTIAL** - iter 38: tau_R2=0.752; hidden_dim_update=64 collapses V_rest to 0.013
17. **n_layers=3 is OPTIMAL** - balances conn, tau, V_rest better than n_layers=4 (which only maximizes conn)
18. **n_layers_update=4 best for tau with lr_emb=4E-3** - iter 62: tau=0.789
19. **embedding_dim=4 causes instability** - iter 47 FAILED completely, keep embedding_dim=2
20. **data_augmentation_loop=29 is optimal** - higher (30) does NOT help, lower hurts tau
21. **recurrent_training=True is HARMFUL** - iter 51: all metrics degrade, DO NOT use
22. **batch_size=2 HURTS conn_R2** - all batch_size=2 configs have conn<0.77 regardless of architecture
23. **coeff_phi_weight_L2=0.001 is optimal** - 0.002 hurts tau and V_rest (N75)
24. **edge_diff boundaries are DISCRETE**: 620 for conn+V_rest, 625 for cluster/balanced, 600 for tau - N110: edge_diff=620 gives conn=0.892, V_rest=0.427 (UPDATED)
25. **edge_norm=1000 REQUIRED with lr_emb=4E-3** - N99 shows edge_norm=990 crashes V_rest from 0.465 to 0.063
26. **phi_L1=0.8 helps cluster_accuracy** - N100: cluster=0.774, may trade off with tau
27. **phi_L1=0.8 + edge_norm=1000 is viable balanced config** - N101: tau=0.807, cluster=0.770 (CONFIRMED)
28. **phi_L1=0.8 is lr_emb-DEPENDENT** - works with lr_emb=3.75E-3 (N101) but HURTS with lr_emb=4E-3 (N103: cluster=0.709)
29. **edge_diff=625 + edge_norm=975 + phi_L1=0.8 is BEST BALANCED** - N105: conn=0.878, tau=0.850, cluster=0.789 (UPDATED)
30. **phi_L1 has DISCRETE OPTIMA** - 1.0 for tau, 0.8 for balanced; 0.9 is SUBOPTIMAL (N108 confirms)
31. **W_L1=7.5E-5 on edge_diff=625/edge_norm=975 HURTS tau** - N106: tau=0.669, cluster=0.797 (good cluster but bad tau)
32. **edge_weight_L1=0.8 improves V_rest** - N109: V_rest=0.339 (+0.118 vs N105's 0.221) at minor conn cost, tau unchanged
33. **edge_diff=620 is CONN+V_REST optimized** - N110: conn=0.892, V_rest=0.427 (BEST V_rest in block 5!) but tau=0.678 (severe drop)
34. **phi_L1=0.8 + W_L1=7.5E-5 maintains cluster** - N111: cluster=0.795 with modest tau recovery (0.730 vs N106's 0.669)

### FOUR OPTIMIZATION PATHS (Block 4)
| Path | Node | lr_emb | edge_diff | edge_norm | W_L1 | conn_R2 | tau_R2 | V_rest_R2 | cluster |
| ---- | ---- | ------ | --------- | --------- | ---- | ------- | ------ | --------- | ------- |
| **CONN** | **83** | 3.75E-3 | 600 | 1000 | 5E-5 | **0.897** | 0.645 | 0.372 | 0.775 |
| **TAU** | **96** | 3.75E-3 | 600 | 975 | 5E-5 | 0.879 | **0.895** | 0.256 | 0.729 |
| **V_REST** | **90** | 4E-3 | 610 | 1000 | 7.5E-5 | 0.846 | 0.778 | **0.465** | 0.758 |
| **CLUSTER** | **92** | 4E-3 | 625 | 1000 | 5E-5 | 0.888 | 0.792 | 0.254 | **0.796** |

### Open Questions
- ~~Can edge_norm=970 (lower than 975) further improve tau on N96 path?~~ **ANSWERED N104**: NO, 970 FAILS
- ~~Can phi_L1=0.8 be combined with edge_norm=1000 to get cluster without losing tau?~~ **ANSWERED N101**: YES, works!
- ~~Can coeff_edge_weight_L1 variations help any path?~~ **ANSWERED N109**: YES, edge_weight_L1=0.8 boosts V_rest (0.339)
- ~~Can edge_diff=620 work for any path?~~ **ANSWERED N110**: YES! edge_diff=620 gives BEST conn+V_rest (0.892, 0.427) but hurts tau
- Is phi_L1=0.7 or 0.75 better than 0.8 for balanced?
- Can edge_diff=620 + edge_weight_L1=0.8 combine for even better V_rest?
- Can edge_diff=615 work now that 620 works?

---

## Previous Block Summary

### Block 1: Learning Rates (24 iterations)
**Best:** N21 conn=0.823, N9 tau=0.689, N14 balanced
**Key:** lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3 optimal

### Block 2: Regularization + Architecture (24 iterations)
**Best:** N44 conn=0.867, N38 tau=0.752, N45 V_rest=0.349
**Key:** n_layers=4, hidden_dim_update=96 optimal; coeff_edge_diff=625 optimal

### Block 3: Batch & Training Dynamics (24 iterations)
**Best:** N66 conn=0.889, tau=0.770, V_rest=0.411; N50 tau=0.911; N71 tau=0.866
**Key:** aug=29, n_layers=3, n_layers_update=4, lr_emb trade-off discovered

### Block 4: Combined Optimization (24 iterations)
**Best Configs:**
| Rank | Node | Key Config | conn_R2 | tau_R2 | V_rest_R2 | cluster | Time |
| ---- | ---- | ---------- | ------- | ------ | --------- | ------- | ---- |
| **1** | **83** | aug=29, lr_emb=3.75E-3, edge_diff=600, edge_norm=1000 | **0.897** | 0.645 | 0.372 | 0.775 | 62.4 |
| **2** | **96** | aug=29, lr_emb=3.75E-3, edge_diff=600, edge_norm=975 | 0.879 | **0.895** | 0.256 | 0.729 | 62.6 |
| **3** | **92** | aug=29, lr_emb=4E-3, edge_diff=625, W_L1=5E-5 | 0.888 | 0.792 | 0.254 | **0.796** | 62.9 |
| **4** | **90** | aug=29, lr_emb=4E-3, edge_diff=610, W_L1=7.5E-5 | 0.846 | 0.778 | **0.465** | 0.758 | 62.1 |

**Key Findings:**
1. **N96 TAU BREAKTHROUGH**: edge_norm=975 achieves tau=0.895 while maintaining conn=0.879
2. Four distinct optimization paths for different metrics
3. lr_emb dictates edge_diff and edge_norm optima: 3.75E-3 → 600/975-1000, 4E-3 → 610-625/1000
4. W_L1=6E-5 is SUBOPTIMAL - discrete values 5E-5 or 7.5E-5 required
5. edge_diff values are discrete: 600, 610, 625 work; 615, 620 both FAIL

---

## Current Block (Block 5)

### Block Info
Block 5: Cross-path optimization - combine best findings from Block 4 paths
Focus: Test combinations across lr_emb regimes, edge_norm refinements, multi-metric optimization
Starting iteration: 97
Base configs: N83 (conn), N96 (tau), N90 (V_rest), N92 (cluster)

### Best Configs Found (Block 5)
| Rank | Node | Key Config | conn_R2 | tau_R2 | V_rest_R2 | cluster | Time |
| ---- | ---- | ---------- | ------- | ------ | --------- | ------- | ---- |
| **1** | **110** | **edge_diff=620, edge_norm=975, phi_L1=0.8** | **0.892** | 0.678 | **0.427** | 0.758 | 62.5 |
| **2** | **105** | edge_diff=625, edge_norm=975, phi_L1=0.8 | 0.878 | **0.850** | 0.221 | **0.789** | 61.9 |
| 3 | 109 | edge_diff=625, edge_norm=975, phi_L1=0.8, edge_weight_L1=0.8 | 0.858 | 0.849 | 0.339 | 0.785 | 61.3 |
| 4 | 111 | edge_diff=625, edge_norm=975, phi_L1=0.8, W_L1=7.5E-5 | 0.803 | 0.730 | 0.278 | 0.795 | 61.9 |

### Iterations This Block (97-120)

**Iter 97** (N97, parent=96, exploit): edge_norm=980 FAILS - all metrics worse than N96. tau=0.697 vs 0.895
**Iter 98** (N98, parent=83, exploit): edge_norm=975 on N83 - tau improved to 0.717 but conn drops to 0.841 (from 0.897)
**Iter 99** (N99, parent=90, explore): edge_norm=990 CATASTROPHIC - conn=0.678, V_rest=0.063. lr_emb=4E-3 NEEDS edge_norm=1000
**Iter 100** (N100, parent=96, principle-test): phi_L1=0.8 gives best cluster=0.774 but tau=0.705 (worse than N96's 0.895)
**Iter 101** (N101, parent=100, exploit): edge_norm=1000 + phi_L1=0.8 gives tau=0.807, cluster=0.770 - balanced config works!
**Iter 102** (N102, parent=98, exploit): **BEST BALANCED** edge_diff=625 + edge_norm=975: conn=0.855, tau=0.860, V_rest=0.343
**Iter 103** (N103, parent=92, explore): phi_L1=0.8 with lr_emb=4E-3 FAILS - cluster=0.709 worse than N92's 0.796
**Iter 104** (N104, parent=96, principle-test): edge_norm=970 CONFIRMS PRINCIPLE 12 - tau=0.767 worse than 975's 0.895

**Key findings batch 26:**
- **N102 NEW BEST BALANCED**: edge_diff=625 + edge_norm=975 achieves conn=0.855, tau=0.860, V_rest=0.343
- phi_L1=0.8 + edge_norm=1000 is viable: N101 tau=0.807, cluster=0.770
- phi_L1=0.8 is lr_emb-DEPENDENT: works with 3.75E-3, HURTS with 4E-3
- PRINCIPLE 12 CONFIRMED: edge_norm=975 is exact optimum, 970 fails

**Iter 105** (N105, parent=102, exploit): **NEW BEST** phi_L1=0.8 on N102: conn=0.878, tau=0.850, cluster=0.789 - best balanced!
**Iter 106** (N106, parent=102, exploit): W_L1=7.5E-5 on N102: cluster=0.797 (best!) but tau=0.669 (severely hurt)
**Iter 107** (N107, parent=101, explore): edge_diff=625 on N101: conn=0.875, tau=0.789 - modest, underperforms N105
**Iter 108** (N108, parent=96, principle-test): phi_L1=0.9 FAILS: conn=0.784, tau=0.706 - confirms discrete optima (1.0 or 0.8, not 0.9)

**Key findings batch 27:**
- **N105 NEW BEST BALANCED**: phi_L1=0.8 on N102 achieves conn=0.878, tau=0.850, cluster=0.789
- phi_L1 has discrete optima: 1.0 for tau, 0.8 for balanced - 0.9 is SUBOPTIMAL
- W_L1=7.5E-5 gives best cluster=0.797 but severely hurts tau (0.669) - trade-off
- edge_norm=975 with edge_diff=625 + phi_L1=0.8 is optimal balanced config

**Iter 109** (N109, parent=105, exploit): edge_weight_L1=0.8 on N105: V_rest=0.339 (+0.118), conn=0.858 (-0.020), tau=0.849 (unchanged)
**Iter 110** (N110, parent=105, exploit): **BREAKTHROUGH** edge_diff=620: conn=0.892, V_rest=0.427 (BEST!) but tau=0.678 (severe drop)
**Iter 111** (N111, parent=106, explore): phi_L1=0.8 on N106: cluster=0.795, tau=0.730 (+0.061) - modest improvement
**Iter 112** (N112, parent=83, principle-test): edge_diff=625 + phi_L1=0.8 on conn path: conn=0.842 - CONFIRMS edge_diff=600 is optimal for conn

**Key findings batch 28:**
- **N110 BREAKTHROUGH**: edge_diff=620 achieves conn=0.892 and V_rest=0.427 (BEST V_REST IN BLOCK 5!) but tau=0.678
- edge_weight_L1=0.8 (N109) improves V_rest by 0.118 at minor conn cost - viable V_rest trade-off
- phi_L1=0.8 + W_L1=7.5E-5 (N111) maintains cluster=0.795 with modest tau recovery
- PRINCIPLE CONFIRMED: edge_diff=600 IS optimal for pure conn - N83's config still best for conn alone
- **NEW edge_diff optima**: 620 for conn+V_rest, 625 for balanced/cluster, 600 for conn/tau

### Next Batch Setup (Iterations 113-116)
| Slot | Node | Parent | Strategy | Key Change | Rationale |
| ---- | ---- | ------ | -------- | ---------- | --------- |
| 00 | 113 | 110 | exploit | edge_weight_L1=0.8 on N110 | Combine edge_diff=620 + edge_weight_L1=0.8 for max V_rest |
| 01 | 114 | 110 | exploit | edge_norm=1000 on N110 | Test if edge_norm=1000 helps N110's conn+V_rest config |
| 02 | 115 | 105 | explore | edge_diff=615 on N105 | Test intermediate edge_diff=615 between 610 (V_rest) and 620 (conn) |
| 03 | 116 | 96 | principle-test | edge_diff=620 + phi_L1=1.0. Testing principle: "edge_diff=620 hurts tau (N110: tau=0.678)" | Test if phi_L1=1.0 can recover tau with edge_diff=620
