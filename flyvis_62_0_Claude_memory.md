# FlyVis Working Memory: flyvis_62_0 (parallel) - EXPERIMENT COMPLETE

## FINAL RESULTS (144 iterations, 6 blocks)

### Optimal Configurations by Target Metric

| Target | Node | edge_diff | edge_weight_L1 | W_L1 | conn_R2 | tau_R2 | V_rest_R2 | cluster |
| ------ | ---- | --------- | -------------- | ---- | ------- | ------ | --------- | ------- |
| **CONN** | **125** | 625 | 0.5 | 5E-5 | **0.929** | 0.755 | 0.461 | 0.764 |
| **TAU+V_REST** | **133** | 620 | 0.7 | 5E-5 | 0.868 | **0.922** | **0.484** | 0.789 |
| **CLUSTER** | **143** | 620 | 0.65 | 5E-5 | 0.860 | 0.859 | 0.255 | **0.824** |
| **BALANCED** | **124** | 625 | 0.6 | 5E-5 | 0.886 | 0.878 | 0.463 | 0.765 |

### Improvement vs Baseline

| Metric | Baseline | Best | Improvement |
| ------ | -------- | ---- | ----------- |
| connectivity_R2 | 0.723 | 0.929 (N125) | +28.5% |
| tau_R2 | 0.451 | 0.922 (N133) | +104.4% |
| V_rest_R2 | 0.062 | 0.484 (N133) | +680.6% |
| cluster_accuracy | 0.722 | 0.824 (N143) | +14.1% |

---

## Knowledge Base (accumulated across all blocks)

### Parameter Effects Table
| Block | Focus | Best conn_R2 | Best tau_R2 | Best V_rest_R2 | Best Cluster_Acc | Key finding |
| ----- | ----- | ------------ | ----------- | -------------- | ---------------- | ----------- |
| 1 (iter1-24) | lr exploration | 0.823 (N21) | 0.689 (N9) | 0.272 (N14) | 0.754 (N7) | lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3 optimal |
| 2 (iter25-48) | reg + arch | 0.867 (N44) | 0.752 (N38) | 0.349 (N45) | 0.772 (N42) | n_layers=4, hidden_dim_update=96 optimal |
| 3 (iter49-72) | batch/recurrent | 0.889 (N66) | 0.911 (N50) | 0.411 (N66) | 0.793 (N49) | N66 BREAKTHROUGH: aug=29, n_layers=3, n_layers_update=4, lr_emb=4E-3 |
| 4 (iter73-96) | combined opt | 0.897 (N83) | 0.895 (N96) | 0.465 (N90) | 0.796 (N92) | Four optimization paths: N83 conn, N96 tau, N90 V_rest, N92 cluster |
| 5 (iter97-120) | cross-path opt | 0.911 (N118) | 0.805 (N120) | 0.441 (N119) | 0.764 (N120) | edge_weight_L1 is PRIMARY tau control; N118 conn record |
| 6 (iter121-144) | final opt | **0.929 (N125)** | **0.922 (N133)** | **0.484 (N133)** | **0.824 (N143)** | Final records: N125 conn, N133 tau+V_rest, N143 cluster |

### Established Principles (FINAL - Validated across 144 iterations)

**Learning Rates:**
1. lr_W=5E-4 is optimal - higher lr_W damages tau (iter 8, 15, 20)
2. lr=1E-3 is optimal - higher causes instability (iter 12), lower causes instability (iter 24)
3. lr_emb=3.75E-3 is optimal for most metrics - 4E-3 can help V_rest with specific configs (principle 6)
4. lr_emb > 4E-3 hurts all metrics (iter 70)

**Regularization:**
5. coeff_edge_diff: 625 for conn, 620 for tau/V_rest/cluster - THREE discrete optima (600, 620, 625)
6. coeff_edge_norm=975 STRICT requirement - 980 fails (N135), 1000 fails (N99, N114)
7. coeff_edge_weight_L1 is PRIMARY tuning parameter:
   - 0.5 → max conn (N125: 0.929)
   - 0.65 → max cluster (N143: 0.824)
   - 0.7 → max tau+V_rest (N133: tau=0.922, V_rest=0.484)
   - 0.6 → balanced (N124)
   - 0.75 FAILS (N142: tau collapses)
   - 0.4 FAILS (N129: conn drops, V_rest collapses, time doubles)
8. coeff_phi_weight_L1=0.8 for most paths - 1.0 trades conn for tau (N130, N137)
9. coeff_W_L1=5E-5 is SAFE default - 4E-5 introduces variability and can collapse V_rest (N141)
10. coeff_phi_weight_L2=0.001 optimal - 0.002 hurts tau and V_rest (N75)

**Architecture:**
11. hidden_dim=64 optimal - 96 causes instability (iter 37)
12. hidden_dim_update=96 ESSENTIAL - 64 collapses V_rest
13. n_layers=3 optimal - balances all metrics better than 4
14. n_layers_update=4 best for tau with lr_emb=4E-3
15. embedding_dim=2 required - 4 causes instability (iter 47)

**Training:**
16. batch_size=1 required - batch_size=2 HURTS conn
17. data_augmentation_loop=29 optimal - higher does NOT help
18. recurrent_training=False required - always hurts metrics (iter 51)

**Parameter Interactions:**
19. edge_diff=625 REQUIRES edge_weight_L1=0.5-0.6
20. edge_diff=620 REQUIRES edge_weight_L1=0.65-0.7
21. edge_diff=600 REQUIRES edge_weight_L1=0.8 EXACTLY (N121, N128)
22. W_L1=4E-5 ONLY works with edge_diff=625, NOT edge_diff=620 (N138, N139)
23. phi_L1=0.8 REQUIRED for N133's tau path - phi_L1=1.0 drops tau from 0.922 to 0.707 (N137)

---

## Block Summaries

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

### Block 5: Cross-Path Optimization (24 iterations)
**Best:** N118 conn=0.911, N113 tau=0.805, N119 V_rest=0.441, N120 cluster=0.764
**Key:** edge_weight_L1 is PRIMARY tau recovery mechanism

### Block 6: Final Optimization (24 iterations)
**Best:** N125 conn=0.929, N133 tau=0.922, N133 V_rest=0.484, N143 cluster=0.824
**Key:** Final records established; edge_weight_L1 discrete optima confirmed (0.5, 0.65, 0.7)

---

## Final Iterations (Block 6, Batch 36: 141-144)

## Iter 141: partial
Node: id=141, parent=136
Mode/Strategy: exploit
Config: edge_diff=625, edge_weight_L1=0.5, W_L1=4E-5
Metrics: conn=0.902, tau=0.861, V_rest=0.064, cluster=0.720, time=58.7
Mutation: edge_weight_L1: 0.5 on N136 (W_L1=4E-5)
Observation: W_L1=4E-5 + edge_weight_L1=0.5 COLLAPSES V_rest (0.064). W_L1=4E-5 is UNSTABLE

## Iter 142: partial
Node: id=142, parent=133
Mode/Strategy: exploit
Config: edge_diff=620, edge_weight_L1=0.75
Metrics: conn=0.808, tau=0.539, V_rest=0.068, cluster=0.797, time=58.2
Mutation: edge_weight_L1: 0.7 -> 0.75 on N133
Observation: edge_weight_L1=0.75 SEVERELY HURTS tau (0.922->0.539). CONFIRMS 0.7 is OPTIMAL for tau path

## Iter 143: converged - **CLUSTER RECORD!**
Node: id=143, parent=133
Mode/Strategy: explore
Config: edge_diff=620, edge_weight_L1=0.65
Metrics: conn=0.860, tau=0.859, V_rest=0.255, cluster=**0.824**, time=59.3
Mutation: edge_weight_L1: 0.7 -> 0.65 on N133
Observation: **NEW CLUSTER RECORD 0.824!** edge_weight_L1=0.65 on edge_diff=620 maximizes cluster accuracy

## Iter 144: partial
Node: id=144, parent=125
Mode/Strategy: principle-test
Config: edge_diff=625, edge_weight_L1=0.5, W_L1=4E-5
Metrics: conn=0.869, tau=0.745, V_rest=0.316, cluster=0.749, time=58.5
Mutation: W_L1: 5E-5 -> 4E-5 on N125. Testing principle: "W_L1=4E-5 ONLY works with edge_diff=625"
Observation: CONFIRMS W_L1=4E-5 works with edge_diff=625 but with HIGH VARIABILITY (conn: 0.869-0.902, V_rest: 0.064-0.316)

---

## EXPERIMENT COMPLETE

All 144 iterations have been completed. The experiment successfully identified optimal configurations for each target metric with significant improvements over baseline:
- Connectivity: +28.5% (0.723 → 0.929)
- Tau recovery: +104.4% (0.451 → 0.922)
- V_rest recovery: +680.6% (0.062 → 0.484)
- Cluster accuracy: +14.1% (0.722 → 0.824)
