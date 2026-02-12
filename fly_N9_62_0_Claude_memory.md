# FlyVis Working Memory: fly_N9_62_0 (parallel)

## Knowledge Base (accumulated across all blocks)

### Parameter Effects Table
| Block | Focus | Best conn_R2 | Best tau_R2 | Best V_rest_R2 | Best Cluster_Acc | Time_min | Key finding |
| ----- | ----- | ------------ | ----------- | -------------- | ---------------- | -------- | ----------- |
| 1 (iter1-24) | lr exploration | 0.823 (N21) | 0.689 (N9) | 0.272 (N14) | 0.754 (N7) | ~49 | lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3 optimal |
| 2 (iter25-48) | reg + arch | 0.867 (N44) | 0.752 (N38) | 0.349 (N45) | 0.772 (N42) | ~55 | n_layers=4, hidden_dim_update=96 optimal |
| 3 (iter49+) | batch/recurrent | **0.869 (N53)** | **0.911 (N50)** | **0.314 (N57)** | **0.793 (N49)** | 53-70 | N62: tau=0.789 with lr_emb=4E-3+n_layers_update=4; N64: 53.6 min within time |

### Established Principles
1. Higher MLP learning rates (lr=1E-3) and embedding learning rates (lr_emb>=2E-3) improve connectivity, tau, and V_rest recovery vs defaults
2. lr_W > 1E-3 damages tau recovery - keep lr_W <= 1E-3 (CONFIRMED iter 8: lr_W=1.5E-3 gave test_R2=-192)
3. lr_emb=3.5E-3 with lr_W=5E-4 achieves best balanced metrics (CONFIRMED iter 48: even with combined arch, 3E-3 underperforms)
4. lr_W=5E-4 is optimal - lower lr_W=3E-4 does NOT help (iter 15: tau=0.565), lr_W=7E-4 not better (iter 20: tau=0.532)
5. lr > 1E-3 causes instability (CONFIRMED iter 12: lr=2E-3 gave test_R2=-inf)
6. lr < 1E-3 causes instability (CONFIRMED iter 24: lr=8E-4 gave test_R2=-453, V_rest collapsed)
7. lr_emb=4E-3 helps V_rest_R2 ONLY with batch_size=1 (iter 45: V_rest=0.349; iter 52: batch_size=2 gives V_rest=0.042)
8. lr_emb > 4E-3 hurts all metrics (CONFIRMED iter 16: lr_emb=5E-3 conn=0.754)
9. lr_emb=3.5E-3 sweet spot is NARROW - both 3.25E-3 (iter 18) and 3.75E-3 (iter 17) underperform (CONFIRMED iter 48)
10. **coeff_edge_diff=625 is optimal** - achieves BOTH best conn_R2=0.839 AND tau_R2=0.644 (iter 25) - CONFIRMED (iter 29: 600 worse, iter 32: 650 worse)
11. coeff_edge_diff < 500 hurts tau severely - 300 gives 0.293, 250 gives 0.381 (CONFIRMED iter 28)
12. coeff_edge_norm=1000 is optimal - 1500 harmful (iter 26), 800 harmful (iter 30), 1200 also worse (iter 40) - CONFIRMED
13. coeff_W_L1=5E-5 is optimal - both higher (1E-4) and lower (2.5E-5) hurt tau (iter 27, 35)
14. **hidden_dim=96 causes instability** - HARMFUL for edge MLP (iter 37: test_R2=-5.95, conn drops to 0.636)
15. **hidden_dim_update=96 is ESSENTIAL** - iter 38: tau_R2=0.752 (BEST TAU), iter 58: hidden_dim_update=64 collapses V_rest to 0.013
16. **n_layers=4 maximizes conn_R2** - iter 44: conn=0.867, iter 53: conn=0.869; BUT n_layers=5 causes degradation (iter 43)
17. **n_layers=3 improves tau+V_rest at cost of conn** - iter 57: tau=0.749, V_rest=0.314; iter 61: tau=0.768 with aug_loop=30 (CONFIRMED)
18. **n_layers_update=4 best for tau with lr_emb=4E-3** - iter 62: tau=0.789 (BEST batch 16); also good for cluster (iter 42: 0.772)
19. **n_layers=3 + n_layers_update=3 enables aug_loop within time** - iter 59: 56.4 min, iter 64: 53.6 min (CONFIRMED)
20. **embedding_dim=4 causes instability** - iter 47 FAILED completely, keep embedding_dim=2
21. **data_augmentation_loop=30 BREAKTHROUGH for tau** - iter 50: tau_R2=0.911 (BEST EVER), but training time 69.8 min exceeds limit
22. **recurrent_training=True is HARMFUL** - iter 51: all metrics degrade (conn 0.801->0.644, tau 0.752->0.436), DO NOT use
23. **batch_size=2 HURTS conn_R2** - all batch_size=2 configs have conn<0.77 regardless of architecture (iter 49, 54, 55)
24. **batch_size=2 cancels aug_loop tau benefit** - iter 54: aug_loop=30 + batch_size=2 gives tau=0.582 (vs 0.911 with batch_size=1)
25. **aug_loop=27 achieves BEST conn_R2=0.869** - iter 53: with n_layers=4; but tau drops to 0.545; time 67.9 min exceeds limit
26. **aug_loop=26 maintains good conn=0.851** - iter 60: only 0.018 below best, time 62.2 min (closer to limit)
27. **aug_loop>=28 needed for good tau with simpler arch** - iter 64: aug_loop=26 gives tau=0.419 (poor), iter 61: aug=30 gives tau=0.768 (good)

### Open Questions
- Can we achieve both conn>0.85 AND tau>0.8 simultaneously? N62 approaches (conn=0.807, tau=0.789)
- Can n_layers=3 + n_layers_update=4 + lr_emb=4E-3 push tau higher while keeping conn>0.8?
- What's the minimum aug_loop for tau>0.7 within time limit?

---

## Previous Block Summary

### Block 1: Learning Rates (24 iterations)
**Best Configurations:**
- N21: conn_R2=0.823 (coeff_edge_diff=750)
- N14: balanced (conn=0.808, tau=0.632, V_rest=0.272)
- N9: best tau_R2=0.689

**Key Findings:**
1. lr sweet spot: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3
2. lr must be exactly 1E-3 (0.8E-3 and 2E-3 both cause instability)
3. coeff_edge_diff=750 best for conn, 500 best balanced

### Block 2: Regularization + Architecture (24 iterations)
**Best Configurations:**
| Rank | Node | Architecture | conn_R2 | tau_R2 | V_rest_R2 | cluster |
| ---- | ---- | ------------ | ------- | ------ | --------- | ------- |
| 1 | 44 | n_layers=4, hidden_dim_update=96 | **0.867** | 0.542 | 0.162 | 0.707 |
| 2 | 46 | n_layers=4, n_layers_update=4, hidden_dim_update=96 | 0.866 | 0.590 | 0.232 | 0.767 |
| 3 | 45 | n_layers=4, hidden_dim_update=96, lr_emb=4E-3 | 0.864 | 0.515 | **0.349** | 0.687 |
| 4 | 42 | n_layers_update=4, hidden_dim_update=96 | 0.844 | 0.686 | 0.271 | **0.772** |
| 5 | 38 | hidden_dim_update=96 | 0.801 | **0.752** | 0.138 | 0.714 |

**Key Findings:**
1. coeff_edge_diff=625 optimal (neither 600 nor 650 better)
2. n_layers=4 dramatically improves conn_R2; hidden_dim_update=96 improves tau_R2
3. hidden_dim=96 for edge MLP HARMFUL; embedding_dim=4 FAILS
4. Trade-off: N44 (best conn=0.867) vs N38 (best tau=0.752) vs N45 (best V_rest=0.349)

---

## Current Block (Block 3)

### Block Info
Block 3: Batch & Training dynamics
Focus: batch_size, data_augmentation_loop, recurrent_training, architecture for time efficiency
Starting iteration: 49
Base config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, n_layers=4, hidden_dim_update=96 (from N44/N46)

### Best Configs Found (Block 3)
| Rank | Node | Key Change | conn_R2 | tau_R2 | V_rest_R2 | cluster | Time | Notes |
| ---- | ---- | ---------- | ------- | ------ | --------- | ------- | ---- | ----- |
| 1 | 53 | aug_loop=27, n_layers=4, n_layers_update=4 | **0.869** | 0.545 | 0.105 | 0.779 | 67.9 | **BEST CONN** (exceeds time) |
| 2 | 50 | aug_loop=30, n_layers=4, n_layers_update=4 | 0.807 | **0.911** | 0.101 | 0.735 | 69.8 | **BEST TAU** (exceeds time) |
| 3 | 62 | aug_loop=28, n_layers=3, n_layers_update=4, lr_emb=4E-3 | 0.807 | **0.789** | 0.184 | 0.709 | 61.7 | **NEW: excellent tau, near time limit** |
| 4 | 61 | aug_loop=30, n_layers=3, n_layers_update=3 | 0.836 | 0.768 | 0.197 | 0.666 | 65.5 | good tau+conn balance |
| 5 | 57 | aug_loop=27, n_layers=3, n_layers_update=4 | 0.799 | 0.749 | **0.314** | 0.768 | 62.8 | **BEST V_REST** + tau compromise |
| 6 | **64** | aug_loop=26, **n_layers=3, n_layers_update=3** | 0.775 | 0.419 | 0.161 | 0.686 | **53.6** | **FASTEST WITHIN TIME** (poor tau) |
| 7 | 49 | batch_size=2 | 0.754 | 0.579 | 0.078 | **0.793** | 50.1 | **BEST CLUSTER** |

### Iterations This Block (49-64)

## Iter 49: partial - batch_size=2 hurts conn (0.754) but BEST cluster=0.793
## Iter 50: converged - data_augmentation_loop=30 gives **tau=0.911 BEST EVER** but time 69.8 min
## Iter 51: partial - recurrent_training=True HARMFUL, all metrics degrade
## Iter 52: converged - batch_size=2 + lr_emb=4E-3 gives conn=0.828, V_rest principle contradicted
## Iter 53: converged - aug_loop=27 gives **conn=0.869 BEST EVER** but time 67.9 min, tau drops to 0.545
## Iter 54: partial - batch_size=2 + aug_loop=30 fits time (55.9 min) but tau drops to 0.582 (batch cancels benefit)
## Iter 55: partial - batch_size=2 + n_layers_update=4 doesn't help conn (0.727)
## Iter 56: converged - aug_loop=28 gives tau=0.675 (good compromise) but time 66.7 min exceeds limit
## Iter 57: partial - n_layers=3 gives tau=0.749, **V_rest=0.314 BEST BLOCK 3**, conn drops to 0.799; time 62.8 min
## Iter 58: converged - hidden_dim_update=64 does NOT reduce time (70.3 min!), V_rest COLLAPSES to 0.013
## Iter 59: converged - **BREAKTHROUGH** n_layers=3+n_layers_update=3 achieves **56.4 min WITHIN LIMIT**, conn=0.830, tau=0.642
## Iter 60: converged - aug_loop=26 gives conn=0.851 (close to best), time 62.2 min still exceeds limit
## Iter 61: converged - aug_loop=30 + n_layers=3+n_layers_update=3 gives tau=0.768, conn=0.836; time 65.5 min exceeds limit
## Iter 62: converged - **tau=0.789** with lr_emb=4E-3+n_layers_update=4; time 61.7 min (near limit)
## Iter 63: converged - aug_loop=29 INCONSISTENT tau=0.586 (worse than aug_loop=30), stochastic variance
## Iter 64: partial - **FASTEST 53.6 min** but aug_loop=26 gives poor tau=0.419; confirms aug_loop>=28 needed

### Next Batch Setup (Iterations 65-68)
| Slot | Node | Parent | Strategy | Key Change | Rationale |
| ---- | ---- | ------ | -------- | ---------- | --------- |
| 00 | 65 | 61 | exploit | aug_loop=30, n_layers=3, n_layers_update=4 | combine best tau arch (n_layers_update=4) with aug_loop=30 |
| 01 | 66 | 62 | exploit | aug_loop=29, n_layers=3, n_layers_update=4, lr_emb=4E-3 | slight aug reduction for time while keeping tau arch |
| 02 | 67 | 61 | explore | aug_loop=28, n_layers=3, n_layers_update=3 | test minimum aug_loop for tau>0.7 with simpler arch |
| 03 | 68 | 50 | principle-test | aug_loop=30, n_layers=4, n_layers_update=3. Testing principle: "n_layers=4 maximizes conn_R2" | test if n_layers=4 helps conn with simpler update MLP |
