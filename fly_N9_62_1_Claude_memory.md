# FlyVis Working Memory: fly_N9_62_1 (parallel)

## Knowledge Base (accumulated across all blocks)

### Parameter Effects Table
| Block | Focus | Best conn_R2 | Best tau_R2 | Best V_rest_R2 | Best Cluster_Acc | Time_min | Key finding |
| ----- | ----- | ------------ | ----------- | -------------- | ---------------- | -------- | ----------- |
| 1 | Learning rates | 0.978 (Node 18) | 0.997 (Node 23) | 0.817 (Node 23) | 0.900 (Node 19) | 48-55 | lr_W=5E-4 to 7E-4 optimal; lr_emb=1.5E-3 required |
| 2 | Regularization | 0.980 (Node 43) | 0.997 (Node 30/34) | 0.760 (Node 30) | 0.910 (Node 34) | 48-51 | phi_L1=0.5 + edge_L1=0.5 beneficial; edge_diff=750-1000 optimal |
| 3 | Architecture | 0.981 (Node 67) | 0.996 (Node 67) | 0.819 (Node 68) | 0.914 (Node 66) | 50-56 | lr_W=6E-4+edge_L1=0.3 best; hidden_dim=80+80 optimal |
| 4 | Batch & Aug | 0.981 (Node 82) | 0.994 (Node 73) | 0.739 (Node 73) | 0.913 (Node 74) | 34-45 | batch=2+data_aug=20 optimal; data_aug=18 fastest |
| 5 | Recurrent | **0.983 (Node 102)** | 0.995 (Node 104) | **0.733 (Node 105)** | **0.898 (Node 110)** | 37-38 | W_L2=2E-6 best conn_R2; W_L2=3E-6 best V_rest; recurrent=HARMFUL |

### Established Principles
1. **lr_W=6E-4 with edge_L1=0.3 achieves best conn_R2** — Node 67 (conn_R2=0.981) beats Node 62 (0.977)
2. **lr_W=1E-3 requires lr=1E-3 (not 1.2E-3)** — lr=1.2E-3 with lr_W=1E-3 causes severe conn_R2 degradation
3. **lr_emb=1.5E-3 is required for lr_W < 1E-3** — lower lr_emb causes connectivity collapse
4. **lr_emb >= 1.8E-3 destroys V_rest recovery** — Node 24: V_rest_R2=0.007
5. **Low lr_emb (5E-4) favors cluster_acc over V_rest** — Node 16: cluster_acc=0.897, V_rest_R2=0.401
6. **coeff_edge_norm >= 10 is catastrophic** — Node 27: tau_R2=0.473, V_rest_R2=0.095
7. **coeff_edge_weight_L1=0.3 is optimal** — Node 67: conn_R2=0.981; edge_L1=0.2 collapses V_rest; edge_L1=0.35 hurts conn_R2; MULTIPLY CONFIRMED
8. **coeff_phi_weight_L1=0.5 improves V_rest recovery** — Node 30: V_rest_R2=0.760, tau_R2=0.997
9. **Combined phi_L1=0.5 + edge_L1=0.3 achieves best connectivity** — Node 67: conn_R2=0.981
10. **coeff_edge_diff=750 is STRICTLY optimal** — edge_diff=600/700/800/1000 all worse; MULTIPLY CONFIRMED (Node 70, 84, 96, 120)
11. **coeff_W_L1=5E-5 is optimal for V_rest** — W_L1=3E-5 hurts V_rest (0.674); W_L1=7E-5 slightly worse; CONFIRMED
12. **coeff_edge_diff=1250+ is harmful** — V_rest collapse
13. **coeff_phi_weight_L2 must stay at 0.001** — phi_L2=0.005 destroys tau_R2 and V_rest
14. **coeff_phi_weight_L1=0.5 is STRICTLY optimal** — phi_L1=0.25/0.45/0.55/0.6/0.75 all worse; MULTIPLY CONFIRMED (Node 119: phi_L1=0.45 → conn_R2=0.937, V_rest=0.403)
15. **n_layers=4 is harmful** — training_time=62.8 min, V_rest collapse
16. **embedding_dim=4 does not improve over default 2** — cluster_acc drops
17. **hidden_dim_update=96 improves tau but hurts connectivity** — Node 52: conn_R2=0.751
18. **hidden_dim=80 optimal for conn_R2; hidden_dim=96 optimal for V_rest** — trade-off exists
19. **hidden_dim_update=80 is beneficial** — Node 54: tau_R2=0.995, V_rest_R2=0.752
20. **n_layers_update=4 is harmful** — V_rest collapse
21. **hidden_dim=80 + hidden_dim_update=80 is optimal architecture** — Node 58: best balance
22. **lr_W=6E-4 with edge_L1=0.3 beats lr_W=5E-4** — Node 67: conn_R2=0.981
23. **phi_L1=0.75 is harmful** — Node 60: conn_R2=0.874, V_rest=0.547
24. **phi_L1=0.4 achieves best cluster_acc** — Node 66: cluster_acc=0.914 but V_rest drops
25. **edge_L1=0.2 is too low** — Node 65/93: V_rest collapse; CONFIRMED (conn_R2=0.916)
26. **phi_L1=0.4 + lr_W=6E-4 conflicts** — Node 69/71: V_rest collapse (0.575/0.687); keep phi_L1=0.5 with lr_W=6E-4
27. **batch_size=2 maintains conn_R2 with faster training** — Node 73: conn_R2=0.980, time=45.8 min
28. **batch_size=4 is too aggressive** — Node 75: V_rest collapse (0.351)
29. **data_augmentation_loop=30 exceeds time limit** — Node 74: time=63.8 min, V_rest collapse (0.526)
30. **data_augmentation_loop=20 is viable for speed** — Node 76: conn_R2=0.974, time=44.5 min
31. **batch_size=3 causes V_rest collapse** — Node 77: V_rest=0.412, conn_R2=0.965; batch=2 is upper limit
32. **data_aug=22 with batch=2 underperforms** — Node 78: conn_R2=0.900 (collapse); data_aug=25 or 20, not 22
33. **batch=2 + data_aug=20 is optimal speed config** — Node 79: conn_R2=0.980, V_rest=0.716, time=39 min (BEST)
34. **lr_W=8E-4 with batch=2 fails** — Node 80: V_rest=0.563, conn_R2=0.971; lr_W=6E-4 optimal even with batching
35. **lr_emb=1.6E-3 hurts conn_R2** — Node 81: conn_R2=0.939 (collapse); lr_emb=1.5E-3 confirmed optimal
36. **lr_W=5E-4 with batch=2+data_aug=20 maintains conn_R2 but hurts V_rest** — Node 82: conn_R2=0.981, V_rest=0.598
37. **data_aug=18 is viable for fastest training** — Node 83: conn_R2=0.979, V_rest=0.668, time=35.4 min
38. **edge_diff=700 is too low** — Node 84: conn_R2=0.958, V_rest=0.519; CONFIRMS edge_diff=750 optimal
39. **phi_L1=0.6 with lr_W=5E-4 is harmful** — Node 85: conn_R2=0.972, V_rest=0.576; phi_L1=0.5 optimal
40. **lr_emb=1.4E-3 causes V_rest collapse** — Node 86: V_rest=0.416, cluster_acc=0.842; lr_emb=1.5E-3 is strict bound
41. **lr_W=5E-4 + data_aug=18 gives worse V_rest than lr_W=6E-4** — Node 87: V_rest=0.500 vs Node 83's 0.668
42. **batch_size=1 is MUCH worse than batch_size=2 for V_rest** — Node 88/116: V_rest=0.484/0.549 vs Node 79's 0.716; MULTIPLY CONFIRMED
43. **lr=1.4E-3 causes V_rest collapse** — Node 89: V_rest=0.356, conn_R2=0.959; lr=1.2E-3 optimal
44. **edge_L1=0.25 is viable with lr_W=5E-4** — Node 90: conn_R2=0.977, V_rest=0.542 (slight improvement)
45. **lr_W=7E-4 is worse than 6E-4** — Node 91: conn_R2=0.970, V_rest=0.565; lr_W=6E-4 confirmed optimal
46. **lr=1.0E-3 causes severe collapse** — Node 94: conn_R2=0.888, V_rest=0.324; lr=1.2E-3 CONFIRMED optimal
47. **recurrent_training=True is HARMFUL** — Nodes 97-99: all show severe conn_R2/V_rest/cluster_acc collapse; time_step=2 already harmful
48. **time_step=4 is catastrophic** — Node 99: conn_R2=0.731, time=78.7 min (EXCEEDS LIMIT); DO NOT use recurrent training
49. **coeff_W_L2=1E-5 slightly hurts conn_R2** — Node 100: conn_R2=0.955 (vs 0.98 baseline), V_rest=0.615; may be too strong
50. **coeff_W_L2=2E-6 is OPTIMAL for conn_R2** — Node 102: conn_R2=0.983, V_rest=0.691; preserves conn_R2 while improving V_rest; BETTER than no W_L2
51. **coeff_W_L2=5E-6 is too weak** — Node 101: conn_R2=0.966, V_rest=0.505; doesn't help V_rest
52. **coeff_edge_norm=0.5 helps cluster_acc but hurts V_rest** — Node 104: cluster_acc=0.888, V_rest=0.518; trade-off with edge_norm=1.0
53. **coeff_W_L2=3E-6 achieves BEST V_rest** — Node 105: V_rest=0.733, conn_R2=0.973; trade-off with W_L2=2E-6 (V_rest=0.691, conn_R2=0.983)
54. **coeff_edge_norm=0.75 is balanced compromise for cluster_acc** — Node 110: V_rest=0.725, conn_R2=0.977, cluster_acc=0.898 (BEST); but context-dependent
55. **coeff_W_L1=4E-5 is too low** — Node 107: V_rest=0.511 collapse; W_L1=5E-5 optimal even with W_L2; CONFIRMED
56. **phi_L1=0.55 is worse than 0.5** — Node 108: conn_R2=0.968, V_rest=0.589; CONFIRMS phi_L1=0.5 optimal
57. **coeff_W_L2=2.5E-6 is NOT optimal** — Node 109: V_rest=0.520 (worse than both 2E-6 and 3E-6); no sweet spot between them
58. **edge_norm=0.75 + W_L2=3E-6 is best for cluster_acc** — Node 110: cluster_acc=0.898, V_rest=0.725; achieves good balance
59. **edge_norm=0.8 causes V_rest collapse** — Node 111: V_rest=0.484; edge_norm trade-off is non-linear (0.75 better than 0.8)
60. **edge_L1=0.35 is worse than 0.3** — Node 112: conn_R2=0.976, V_rest=0.545; CONFIRMS principle #7 (edge_L1=0.3 optimal)
61. **coeff_W_L2=3.5E-6 is too high** — Node 113/115: V_rest=0.605/0.692 (vs 0.725 for W_L2=3E-6); W_L2=3E-6 is upper bound
62. **coeff_edge_norm=0.7 is worse than 0.75** — Node 114: V_rest=0.596 (vs 0.725 for edge_norm=0.75); edge_norm=0.75 is optimal lower bound
63. **edge_norm=0.75 effect is context-dependent** — Node 118: edge_norm=0.75 + W_L2=3E-6 from Node 105 gives V_rest=0.550 (worse than Node 105's 0.733); not universally beneficial

### Current Open Questions
1. Can we achieve both conn_R2>0.98 AND V_rest>0.75 simultaneously?
2. Does hidden_dim=96 improve V_rest while maintaining conn_R2>0.98?
3. Can slight lr adjustments bridge the W_L2=2E-6 vs W_L2=3E-6 trade-off?
4. Does data_aug=25 with W_L2 help achieve better balance?

---

## Previous Block Summaries

### Block 1: Learning Rates (24 iterations)
**Best**: Node 18 (lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3) → conn_R2=0.978

### Block 2: Regularization (24 iterations)
**Best**: Node 43 (edge_diff=750, phi_L1=0.5, edge_L1=0.5) → conn_R2=0.980

### Block 3: Architecture (24 iterations)
**Best**: Node 67 (lr_W=6E-4, hidden_dim=80, hidden_dim_update=80, edge_L1=0.3, phi_L1=0.5) → conn_R2=0.981

### Block 4: Batch & Augmentation (24 iterations)
**Best**: Node 79 (batch=2, data_aug=20, lr_W=6E-4) → conn_R2=0.980, V_rest=0.716, time=39 min

### Block 5: Recurrent Training & W_L2 (24 iterations)
**Best conn_R2**: Node 102 (W_L2=2E-6) → conn_R2=0.983 (OVERALL BEST)
**Best V_rest**: Node 105 (W_L2=3E-6) → V_rest=0.733 (OVERALL BEST)
**Best cluster_acc**: Node 110 (edge_norm=0.75 + W_L2=3E-6) → cluster_acc=0.898 (OVERALL BEST)

Key findings (Iter 117-120):
- Node 117: edge_norm=0.75 + W_L2=2E-6 → conn_R2=0.976, V_rest=0.707; trade-off confirmed
- Node 118: edge_norm=0.75 + W_L2=3E-6 (from Node 105) → conn_R2=0.982, V_rest=0.550; WORSE than Node 110's 0.725 (context-dependent)
- Node 119: phi_L1=0.45 → conn_R2=0.937, V_rest=0.403; SEVERE collapse; CONFIRMS phi_L1=0.5 STRICTLY optimal
- Node 120: edge_diff=800 → conn_R2=0.882, V_rest=0.483; SEVERE collapse; CONFIRMS edge_diff=750 STRICTLY optimal

---

## Current Block (Block 6)

### Block Info
Focus: Combined Best (best parameters from blocks 1-5)
Starting from best configs:
- Node 102: W_L2=2E-6, edge_norm=1.0 → conn_R2=0.983 (BEST conn_R2)
- Node 105: W_L2=3E-6, edge_norm=1.0 → V_rest=0.733 (BEST V_rest)
- Node 110: W_L2=3E-6, edge_norm=0.75 → cluster_acc=0.898 (BEST cluster_acc)

### Block 6 Goals
1. Achieve conn_R2 > 0.98 AND V_rest > 0.75 simultaneously
2. Test untried combinations of best parameters
3. Fine-tune around optimal values

### Next Batch Plan (Iter 121-124)
| Slot | Role | Parent | Focus | Mutation |
| ---- | ---- | ------ | ----- | -------- |
| 0 | exploit | Node 118 | lr_W adjustment | lr_W: 6E-4 -> 5E-4 (test if lower lr_W improves V_rest with edge_norm=0.75+W_L2=3E-6) |
| 1 | exploit | Node 102 | W_L2+data_aug | coeff_W_L2: 2E-6 -> 2.5E-6, data_aug: 20 -> 25 (test middle W_L2 with more data augmentation) |
| 2 | explore | Node 105 | MLP lr | lr: 1.2E-3 -> 1.1E-3 (test if slightly lower MLP lr helps V_rest) |
| 3 | principle-test | Node 102 | architecture | hidden_dim: 80 -> 96. Testing principle: "hidden_dim=96 optimal for V_rest" (principle #18) |
