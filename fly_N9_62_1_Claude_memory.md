# FlyVis Working Memory: fly_N9_62_1 (parallel)

## Knowledge Base (accumulated across all blocks)

### Parameter Effects Table
| Block | Focus | Best conn_R2 | Best tau_R2 | Best V_rest_R2 | Best Cluster_Acc | Time_min | Key finding |
| ----- | ----- | ------------ | ----------- | -------------- | ---------------- | -------- | ----------- |
| 1 | Learning rates | 0.978 (Node 18) | 0.997 (Node 23) | 0.817 (Node 23) | 0.900 (Node 19) | 48-55 | lr_W=5E-4 to 7E-4 optimal; lr_emb=1.5E-3 required |
| 2 | Regularization | 0.980 (Node 43) | 0.997 (Node 30/34) | 0.760 (Node 30) | 0.910 (Node 34) | 48-51 | phi_L1=0.5 + edge_L1=0.5 beneficial; edge_diff=750-1000 optimal |
| 3 | Architecture (iter 49-56) | 0.968 (Node 53) | 0.995 (Node 54) | 0.752 (Node 54) | 0.892 (Node 54) | 50-59 | hidden_dim=80 optimal; hidden_dim_update=80 beneficial; n_layers_update=4 harmful |

### Established Principles
1. **lr_W=5E-4 to 7E-4 with lr=1.2E-3 and lr_emb=1.5E-3 is optimal** — Node 18 (conn_R2=0.978) and Node 23 (V_rest_R2=0.817) use this range
2. **lr_W=1E-3 requires lr=1E-3 (not 1.2E-3)** — lr=1.2E-3 with lr_W=1E-3 causes severe conn_R2 degradation (Node 22: conn_R2=0.601)
3. **lr_emb=1.5E-3 is required for lr_W < 1E-3** — lower lr_emb (1.2E-3, 1E-3) causes connectivity collapse with lr_W=5E-4 (Nodes 17, 20)
4. **lr_emb >= 1.8E-3 destroys V_rest recovery** — Node 24: V_rest_R2=0.007; Node 55: V_rest_R2=0.358 (reconfirmed)
5. **Low lr_emb (5E-4) favors cluster_acc over V_rest** — Node 16: cluster_acc=0.897, V_rest_R2=0.401
6. **coeff_edge_norm >= 10 is catastrophic** — Node 27: tau_R2=0.473, V_rest_R2=0.095 (avoid high monotonicity penalty)
7. **coeff_edge_weight_L1=0.5 improves connectivity** — Node 31: conn_R2=0.960 with V_rest_R2=0.712
8. **coeff_phi_weight_L1=0.5 improves V_rest recovery** — Node 30: V_rest_R2=0.760, tau_R2=0.997
9. **Combined phi_L1=0.5 + edge_L1=0.5 achieves best connectivity** — Node 43: conn_R2=0.980 at lr_W=5E-4, edge_diff=750
10. **coeff_edge_diff=1000 with L1 reductions achieves best balance** — Node 34: conn_R2=0.973, V_rest_R2=0.709, cluster_acc=0.910
11. **coeff_W_L1=5E-5 is optimal for V_rest** — W_L1=2E-5 hurts V_rest (Node 38); W_L1=1E-4 boosts conn_R2 but hurts V_rest
12. **coeff_edge_diff=1250+ is harmful** — Node 41: V_rest_R2=0.236 (catastrophic), conn_R2=0.961 (slight drop)
13. **coeff_phi_weight_L2 must stay at 0.001** — phi_L2=0.005 destroys tau_R2 (0.911) and V_rest (0.175) — Node 48
14. **coeff_phi_weight_L1=0.25 is viable but 0.5 is optimal** — Node 44: conn_R2=0.974 vs Node 40: conn_R2=0.976
15. **n_layers=4 is harmful** — Node 50: conn_R2=0.783, V_rest=0.123, training_time=62.8 min (exceeds limit)
16. **embedding_dim=4 does not improve over default 2** — Node 51: cluster_acc drops 0.890→0.828, V_rest drops
17. **hidden_dim_update=96 improves tau but hurts connectivity** — Node 52: tau_R2=0.994, conn_R2=0.751
18. **hidden_dim=80 is optimal** — Node 53: conn_R2=0.968, V_rest_R2=0.735 (better balance than 64 or 96)
19. **hidden_dim_update=80 is beneficial** — Node 54: tau_R2=0.995, V_rest_R2=0.752, cluster_acc=0.892
20. **n_layers_update=4 is harmful** — Node 56: V_rest_R2=0.357 (collapse), conn_R2=0.951 (avoid deeper update MLP)

### Open Questions (Resolved)
1. ~~Can hidden_dim=80 achieve better balance than 64 or 96?~~ **YES** — Node 53: conn_R2=0.968, V_rest=0.735
2. ~~Does moderate hidden_dim_update increase (80) avoid connectivity collapse?~~ **YES** — Node 54: conn_R2=0.959 (no collapse), tau=0.995
3. ~~Can deeper update MLP (n_layers_update=4) improve tau/V_rest without hurting conn_R2?~~ **NO** — Node 56 shows V_rest collapse

### New Open Questions
1. Can combining hidden_dim=80 with hidden_dim_update=80 achieve both benefits?
2. Does edge_diff=1000 with hidden_dim=80 improve V_rest further?
3. Can phi_L1=0.75 improve connectivity without hurting other metrics?

---

## Previous Block Summaries

### Block 1: Learning Rates (24 iterations)

**Best Configurations:**
| Node | lr_W | lr | lr_emb | conn_R2 | V_rest_R2 | cluster_acc | Strength |
|------|------|--------|--------|---------|-----------|-------------|----------|
| 18 | 5E-4 | 1.2E-3 | 1.5E-3 | **0.978** | 0.625 | 0.863 | Best connectivity |
| 23 | 7E-4 | 1.2E-3 | 1.5E-3 | 0.823 | **0.817** | 0.884 | Best V_rest + tau |
| 15 | 5E-4 | 1E-3 | 1.5E-3 | 0.976 | 0.767 | 0.873 | Balanced |
| 19 | 1E-3 | 1E-3 | 1.5E-3 | 0.869 | 0.772 | **0.900** | Best cluster_acc |

### Block 2: Regularization (24 iterations)

**Best Configurations:**
| Node | conn_R2 | V_rest_R2 | tau_R2 | cluster_acc | Key config |
|------|---------|-----------|--------|-------------|------------|
| 43 | **0.980** | 0.387 | 0.991 | 0.890 | lr_W=5E-4, edge_diff=750, phi_L1=0.5, edge_L1=0.5 |
| 42 | 0.979 | 0.447 | 0.990 | 0.844 | lr_W=7E-4, edge_diff=1000, W_L1=1E-4 |
| 40 | 0.976 | 0.675 | 0.992 | 0.865 | lr_W=7E-4, edge_diff=1000, phi_L1=0.5, edge_L1=0.5 |
| 34 | 0.973 | 0.709 | **0.997** | **0.910** | lr_W=5E-4, edge_diff=1000, phi_L1=0.5, edge_L1=0.5 |
| 30 | 0.915 | **0.760** | 0.997 | 0.884 | lr_W=5E-4, edge_diff=1000, phi_L1=0.5 |

**Key Finding:** Combined L1 reductions (phi_L1=0.5, edge_L1=0.5) with edge_diff=750-1000 achieve best results.

---

## Current Block (Block 3)

### Block Info
Focus: Architecture parameters (hidden_dim, n_layers, hidden_dim_update, n_layers_update, embedding_dim)
Starting from best config (Node 43): lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, edge_diff=750, phi_L1=0.5, edge_L1=0.5, W_L1=5E-5

### Hypothesis
- Larger hidden_dim (96, 128) may increase model capacity and improve parameter recovery
- Deeper networks (n_layers=4) may help but could slow training
- embedding_dim > 2 may help cluster_acc by providing more representational capacity
- Update MLP changes may specifically help tau/V_rest recovery

### Initial Batch Plan (Iter 49-52)
| Slot | Role | Parent | Focus | Mutation |
| ---- | ---- | ------ | ----- | -------- |
| 0 | exploit | root | hidden_dim | hidden_dim: 64 -> 96 |
| 1 | exploit | root | n_layers | n_layers: 3 -> 4 |
| 2 | explore | root | embedding_dim | embedding_dim: 2 -> 4 |
| 3 | principle-test | root | hidden_dim_update | hidden_dim_update: 64 -> 96. Testing: "larger update MLP helps parameter recovery" |

### Iterations This Block

## Iter 49: partial
Node: id=49, parent=root
Config: hidden_dim=96
Metrics: conn_R2=0.954, tau_R2=0.992, V_rest_R2=0.615, cluster_acc=0.899, time=54.4
Mutation: hidden_dim: 64 -> 96
Observation: hidden_dim=96 best cluster_acc=0.899, good V_rest but conn_R2 < baseline

## Iter 50: partial ⚠️ TIME EXCEEDED
Node: id=50, parent=root
Metrics: conn_R2=0.783, V_rest_R2=0.123, time=62.8 ⚠️
Mutation: n_layers: 3 -> 4
Observation: n_layers=4 harmful — avoid

## Iter 51: partial
Node: id=51, parent=root
Metrics: conn_R2=0.959, V_rest_R2=0.403, cluster_acc=0.828, time=50.2
Mutation: embedding_dim: 2 -> 4
Observation: embedding_dim=4 doesn't improve; default 2 is sufficient

## Iter 52: partial
Node: id=52, parent=root
Metrics: conn_R2=0.751, tau_R2=0.994, V_rest_R2=0.508, time=53.0
Mutation: hidden_dim_update: 64 -> 96
Observation: hidden_dim_update=96 improves tau but hurts connectivity

## Iter 53: converged ⭐ BEST V_rest_R2
Node: id=53, parent=49
Config: hidden_dim=80
Metrics: conn_R2=0.968, tau_R2=0.980, V_rest_R2=0.735, cluster_acc=0.882, time=52.8
Mutation: hidden_dim: 96 -> 80
Observation: **hidden_dim=80 is optimal** — best V_rest + conn in batch

## Iter 54: converged ⭐ BEST tau_R2 + cluster_acc
Node: id=54, parent=49
Config: hidden_dim=96, hidden_dim_update=80
Metrics: conn_R2=0.959, tau_R2=0.995, V_rest_R2=0.752, cluster_acc=0.892, time=58.5
Mutation: hidden_dim_update: 64 -> 80
Observation: **hidden_dim_update=80 beneficial** — best tau + cluster without connectivity collapse

## Iter 55: partial — CONFIRMS PRINCIPLE 4
Node: id=55, parent=51
Config: embedding_dim=4, lr_emb=1.8E-3
Metrics: conn_R2=0.942, tau_R2=0.985, V_rest_R2=0.358, cluster_acc=0.798, time=56.1
Mutation: lr_emb: 1.5E-3 -> 1.8E-3
Observation: lr_emb=1.8E-3 destroys V_rest — confirms principle 4

## Iter 56: partial — REFUTES PRINCIPLE
Node: id=56, parent=49
Config: hidden_dim=96, n_layers_update=4
Metrics: conn_R2=0.951, tau_R2=0.989, V_rest_R2=0.357, cluster_acc=0.854, time=56.6
Mutation: n_layers_update: 3 -> 4. Testing: "deeper update MLP helps parameter recovery"
Observation: n_layers_update=4 causes V_rest collapse — avoid deeper update MLP

### Next Batch Plan (Iter 57-60)
UCB ranking: Node 53 (2.968) > Node 54 (2.959) > Node 56 (2.951) > Node 55 (2.941)

| Slot | Role | Parent | Focus | Mutation |
| ---- | ---- | ------ | ----- | -------- |
| 0 | exploit | Node 53 | hidden_dim_update | hidden_dim_update: 64 -> 80 (combine with hidden_dim=80) |
| 1 | exploit | Node 54 | hidden_dim | hidden_dim: 96 -> 80 (test intermediate with update=80) |
| 2 | explore | Node 53 | coeff_edge_diff | coeff_edge_diff: 750 -> 1000 (test edge_diff=1000 with hidden_dim=80) |
| 3 | principle-test | Node 54 | coeff_phi_weight_L1 | coeff_phi_weight_L1: 0.5 -> 0.75. Testing: "phi_L1=0.5 is optimal"
