# FlyVis Working Memory: fly_N9_62_0 (parallel)

## Knowledge Base (accumulated across all blocks)

### Parameter Effects Table
| Block | Focus | Best conn_R2 | Best tau_R2 | Best V_rest_R2 | Best Cluster_Acc | Time_min | Key finding |
| ----- | ----- | ------------ | ----------- | -------------- | ---------------- | -------- | ----------- |
| 1 (iter1-24) | lr exploration | 0.823 (N21) | 0.689 (N9) | 0.272 (N14) | 0.754 (N7) | ~49 | lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3 optimal |
| 2 (iter25-48) | reg + arch | 0.867 (N44) | 0.752 (N38) | 0.349 (N45) | 0.772 (N42) | ~55 | n_layers=4, hidden_dim_update=96 optimal |
| 3 (iter49-72) | batch/recurrent | 0.889 (N66) | 0.911 (N50) | 0.411 (N66) | 0.793 (N49) | 53-70 | N66 BREAKTHROUGH: aug=29, n_layers=3, n_layers_update=4, lr_emb=4E-3 |
| 4 (iter73-96) | combined opt | 0.897 (N83) | 0.895 (N96) | 0.465 (N90) | 0.796 (N92) | ~62-65 | Four optimization paths: N83 conn, N96 tau, N90 V_rest, N92 cluster |
| 5 (iter97-120) | cross-path opt | **0.911 (N118)** | **0.805 (N120)** | **0.441 (N119)** | **0.764 (N120)** | ~62 | edge_weight_L1 is PRIMARY tau control; N118 conn record |

### Established Principles
1. Higher MLP learning rates (lr=1E-3) and embedding learning rates (lr_emb>=2E-3) improve connectivity, tau, and V_rest recovery vs defaults
2. lr_W > 1E-3 damages tau recovery - keep lr_W <= 1E-3 (CONFIRMED iter 8: lr_W=1.5E-3 gave test_R2=-192)
3. lr_W=5E-4 is optimal - lower lr_W=3E-4 does NOT help (iter 15: tau=0.565), lr_W=7E-4 not better (iter 20: tau=0.532)
4. lr > 1E-3 causes instability (CONFIRMED iter 12: lr=2E-3 gave test_R2=-inf)
5. lr < 1E-3 causes instability (CONFIRMED iter 24: lr=8E-4 gave test_R2=-453, V_rest collapsed)
6. lr_emb=4E-3 helps V_rest_R2 ONLY with batch_size=1 (iter 45: V_rest=0.349; iter 52: batch_size=2 gives V_rest=0.042)
7. **lr_emb > 4E-3 hurts all metrics** (CONFIRMED iter 70: lr_emb=4.5E-3 caused tau=0.474, V_rest=0.142)
8. **lr_emb DISCRETE SWEET SPOTS**: 3.75E-3 for conn/tau, 4E-3 for V_rest/cluster - midpoints (3.875E-3) FAIL (N74)
9. **coeff_edge_diff DEPENDS on lr_emb**: 600 with lr_emb=3.75E-3 (N83 conn=0.897), 610-625 with lr_emb=4E-3
10. coeff_edge_diff < 600 hurts conn severely - 575 gives 0.746 (N88)
11. **coeff_edge_norm=975 OPTIMAL for tau with lr_emb=3.75E-3** - N96 tau=0.895. edge_norm=980 FAILS (N97: tau=0.697)
12. **coeff_W_L1=5E-5 for conn/tau paths** - W_L1=7.5E-5 HURTS edge_diff=620 path (N117). V_rest path uses 7.5E-5
13. **W_L1=6E-5 is SUBOPTIMAL** - neither 5E-5 nor 7.5E-5 benefits (N95: tau drops to 0.653)
14. **hidden_dim=96 causes instability** - HARMFUL for edge MLP (iter 37: test_R2=-5.95)
15. **hidden_dim_update=96 is ESSENTIAL** - iter 38: tau_R2=0.752; hidden_dim_update=64 collapses V_rest to 0.013
16. **n_layers=3 is OPTIMAL** - balances conn, tau, V_rest better than n_layers=4 (which only maximizes conn)
17. **n_layers_update=4 best for tau with lr_emb=4E-3** - iter 62: tau=0.789
18. **embedding_dim=4 causes instability** - iter 47 FAILED completely, keep embedding_dim=2
19. **data_augmentation_loop=29 is optimal** - higher (30) does NOT help, lower hurts tau
20. **recurrent_training=True is HARMFUL** - iter 51: all metrics degrade, DO NOT use
21. **batch_size=2 HURTS conn_R2** - all batch_size=2 configs have conn<0.77 regardless of architecture
22. **coeff_phi_weight_L2=0.001 is optimal** - 0.002 hurts tau and V_rest (N75)
23. **edge_diff boundaries are DISCRETE**: 600 (max conn), 620 (conn+tau), 625 (balanced) - N115: 615 FAILS
24. **edge_norm=1000 REQUIRED with lr_emb=4E-3** - N99 shows edge_norm=990 crashes V_rest from 0.465 to 0.063
25. **phi_L1=0.8 helps cluster_accuracy** - N100: cluster=0.774, may trade off with tau
26. **phi_L1=0.8 + edge_norm=1000 is viable balanced config** - N101: tau=0.807, cluster=0.770 (CONFIRMED)
27. **phi_L1=0.8 is lr_emb-DEPENDENT** - works with lr_emb=3.75E-3 (N101) but HURTS with lr_emb=4E-3 (N103: cluster=0.709)
28. **edge_diff=625 + edge_norm=975 + phi_L1=0.8 is BEST BALANCED** - N105: conn=0.878, tau=0.850, cluster=0.789
29. **phi_L1 has DISCRETE OPTIMA** - 1.0 for tau, 0.8 for balanced; 0.9 is SUBOPTIMAL (N108 confirms)
30. **W_L1=7.5E-5 on edge_diff=625/edge_norm=975 HURTS tau** - N106: tau=0.669, cluster=0.797 (good cluster but bad tau)
31. **edge_weight_L1=0.8 is PRIMARY tau recovery mechanism** - N120: tau=0.805 with phi_L1=1.0! phi_L1=0.8 NOT required
32. **edge_diff=600 + edge_weight_L1=0.8 = MAX conn (0.911)** - N118. BUT tau collapses to 0.528
33. **edge_weight_L1=0.7 = MAX V_rest (0.441)** - N119. Decent conn=0.874, tau=0.717
34. **W_L1=5E-5 REQUIRED for edge_diff=620 path** - N117: W_L1=7.5E-5 hurts ALL metrics on this path
35. **edge_diff=620 REQUIRES edge_weight_L1=0.8 for tau** - N113: tau=0.805 vs N110's 0.678 without edge_weight_L1
36. **edge_norm=975 REQUIRED for edge_diff=620** - N114: edge_norm=1000 crashes all metrics
37. **edge_diff=620 path REQUIRES edge_weight_L1=0.8** - N116: phi_L1=1.0 alone gives tau=0.659

### OPTIMIZATION PATHS (Updated after Block 5)
| Path | Node | lr_emb | edge_diff | edge_norm | edge_weight_L1 | phi_L1 | W_L1 | conn_R2 | tau_R2 | V_rest_R2 | cluster |
| ---- | ---- | ------ | --------- | --------- | -------------- | ------ | ---- | ------- | ------ | --------- | ------- |
| **CONN** | **118** | 3.75E-3 | 600 | 975 | 0.8 | 0.8 | 5E-5 | **0.911** | 0.528 | 0.272 | 0.758 |
| **CONN+TAU** | **113** | 3.75E-3 | 620 | 975 | 0.8 | 0.8 | 5E-5 | 0.900 | 0.805 | 0.291 | 0.728 |
| **V_REST** | **119** | 3.75E-3 | 625 | 975 | 0.7 | 0.8 | 5E-5 | 0.874 | 0.717 | **0.441** | 0.752 |
| **TAU+CLUSTER** | **120** | 3.75E-3 | 600 | 975 | 0.8 | 1.0 | 5E-5 | 0.849 | **0.805** | 0.284 | **0.764** |
| **BALANCED** | **105** | 3.75E-3 | 625 | 975 | 1.0 | 0.8 | 5E-5 | 0.878 | 0.850 | 0.221 | 0.789 |

### Open Questions
- Can edge_weight_L1=0.6 or lower improve V_rest even more beyond N119's 0.441?
- Can combining edge_weight_L1=0.7 with edge_diff=620 optimize V_rest on conn+tau path?
- Can edge_weight_L1=0.75 balance conn, tau, and V_rest better than 0.7 or 0.8?
- Does phi_L1=1.0 + edge_weight_L1=0.8 outperform phi_L1=0.8 + edge_weight_L1=0.8 for any metric?

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
**Best:** N83 conn=0.897, N96 tau=0.895, N90 V_rest=0.465, N92 cluster=0.796
**Key:** Four distinct optimization paths established; lr_emb dictates edge_diff/edge_norm optima

### Block 5: Cross-Path Optimization (24 iterations, 97-120)
**Best Configs:**
| Rank | Node | Key Config | conn_R2 | tau_R2 | V_rest_R2 | cluster | Time |
| ---- | ---- | ---------- | ------- | ------ | --------- | ------- | ---- |
| **1** | **118** | edge_diff=600, edge_weight_L1=0.8 | **0.911** | 0.528 | 0.272 | 0.758 | 61.9 |
| **2** | **113** | edge_diff=620, edge_norm=975, edge_weight_L1=0.8 | 0.900 | 0.805 | 0.291 | 0.728 | 62.1 |
| **3** | **119** | edge_diff=625, edge_weight_L1=0.7 | 0.874 | 0.717 | **0.441** | 0.752 | 62.0 |
| **4** | **120** | edge_diff=600, edge_weight_L1=0.8, phi_L1=1.0 | 0.849 | 0.805 | 0.284 | **0.764** | 61.7 |

**Key Findings:**
1. **edge_weight_L1=0.8 is PRIMARY tau recovery mechanism** - works even with phi_L1=1.0 (N120)
2. **N118 new conn record (0.911)** with edge_diff=600 + edge_weight_L1=0.8
3. **edge_weight_L1=0.7 maximizes V_rest (0.441)** - N119
4. **edge_diff has THREE discrete optima**: 600 (max conn), 620 (conn+tau), 625 (balanced)
5. **W_L1=5E-5 REQUIRED for edge_diff=620** - 7.5E-5 hurts all metrics (N117)

---

## Current Block (Block 6)

### Block Info
Block 6: Final optimization - combine best findings from all paths
Focus: Optimize multi-metric combinations, test edge_weight_L1 variations, refine best paths
Starting iteration: 121
Base configs: N118 (conn=0.911), N113 (conn+tau), N119 (V_rest), N120 (tau+cluster)

### Best Configs Found (Block 6)
[No iterations yet]

### Iterations This Block (121-144)
[Next batch starts here]

### Next Batch Setup (Iterations 121-124)
| Slot | Node | Parent | Strategy | Key Change | Rationale |
| ---- | ---- | ------ | -------- | ---------- | --------- |
| 00 | 121 | 118 | exploit | edge_weight_L1=0.7 on N118's conn path | See if edge_weight_L1=0.7 can boost V_rest while keeping high conn |
| 01 | 122 | 119 | exploit | edge_diff=620 on V_rest path | Can V_rest path benefit from edge_diff=620? |
| 02 | 123 | 120 | explore | phi_L1=0.8 on N120's config | N120 with phi_L1=0.8 vs 1.0 - which gives better balance? |
| 03 | 124 | 105 | principle-test | edge_weight_L1=0.6. Testing principle: "edge_weight_L1=0.7 maximizes V_rest (N119: 0.441)" | Can lower edge_weight_L1 push V_rest higher? |
