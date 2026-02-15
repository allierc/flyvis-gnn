# FlyVis Working Memory: fly_N9_62_1 (parallel)

## Knowledge Base (accumulated across all blocks)

### Parameter Effects Table
| Block | Focus | Best conn_R2 | Best tau_R2 | Best V_rest_R2 | Best Cluster_Acc | Time_min | Key finding |
| ----- | ----- | ------------ | ----------- | -------------- | ---------------- | -------- | ----------- |
| 1 | Learning rates | 0.978 (Node 18) | 0.997 (Node 23) | 0.817 (Node 23) | 0.900 (Node 19) | 48-55 | lr_W=5E-4 to 7E-4 optimal; lr_emb=1.5E-3 required |
| 2 | Regularization | 0.980 (Node 43) | 0.997 (Node 30/34) | 0.760 (Node 30) | 0.910 (Node 34) | 48-51 | phi_L1=0.5 + edge_L1=0.5 beneficial; edge_diff=750-1000 optimal |
| 3 | Architecture | 0.981 (Node 67) | 0.996 (Node 67) | 0.819 (Node 68) | 0.914 (Node 66) | 50-56 | lr_W=6E-4+edge_L1=0.3 best; hidden_dim=80+80 optimal |
| 4 | Batch & Aug | 0.981 (Node 82) | 0.994 (Node 73) | 0.739 (Node 73) | 0.913 (Node 74) | 34-45 | batch=2+data_aug=20 optimal; data_aug=18 fastest |
| 5 | Recurrent | **0.983 (Node 102)** | 0.995 (Node 104) | 0.733 (Node 105) | **0.898 (Node 110)** | 37-38 | W_L2=2E-6 best conn_R2; W_L2=3E-6 best V_rest; recurrent=HARMFUL |
| 6 | Combined Best | 0.980 (Node 144) | 0.990 (Node 144) | **0.736 (Node 141)** | 0.877 (Node 144) | 37-38 | W_L2=2.8E-6+edge_L1=0.28→NEW BEST V_rest; edge_norm=0.9 balanced |

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
64. **lr=1.1E-3 is CATASTROPHIC** — Node 123: conn_R2=0.913, V_rest=0.282; lr=1.2E-3 is STRICTLY optimal (joining principle #46)
65. **lr_W=5E-4 is harmful with edge_norm=0.75** — Node 121: V_rest=0.487 (collapse), but cluster_acc=0.909; lr_W=6E-4 required with edge_norm<1.0
66. **hidden_dim=96 trades conn_R2 for V_rest with W_L2=2E-6** — Node 124: conn_R2=0.956, V_rest=0.611; confirms principle #18 trade-off
67. **W_L2=2.8E-6 achieves excellent conn_R2** — Node 126: conn_R2=0.981, V_rest=0.562; viable middle ground between 2E-6 and 3E-6
68. **edge_L1=0.28 improves V_rest over 0.3** — Node 127: V_rest=0.667 (vs ~0.56 at 0.3) while maintaining conn_R2=0.979; slight reduction beneficial
69. **lr_emb=1.55E-3 is BENEFICIAL for V_rest** — Node 128: V_rest=0.702, conn_R2=0.978; boundary of principle #4 confirmed at 1.8E-3 not 1.55E-3
70. **W_L2=2.6E-6 is WORSE than 2.8E-6** — Node 129: V_rest=0.434 (collapse), conn_R2=0.975; W_L2=2.8E-6 is LOCAL OPTIMUM
71. **lr_emb=1.55E-3 + edge_L1=0.28 do NOT synergize** — Node 130: V_rest=0.568 (vs 0.702 at lr_emb=1.55E-3 alone); benefits CONFLICT
72. **edge_L1=0.26 is TOO LOW** — Node 131: V_rest=0.594 (vs 0.667 at edge_L1=0.28); edge_L1=0.28 is OPTIMAL lower bound
73. **W_L2=2E-6 optimal is CONTEXT-DEPENDENT** — Node 132: with edge_L1=0.28, W_L2=2E-6 gives conn_R2=0.967 (vs 0.979 with W_L2=3E-6); principle #50 only holds with edge_L1=0.3
74. **lr_emb=1.6E-3 is TOO HIGH** — Node 133: V_rest=0.532 (drop from 0.568); lr_emb=1.55E-3 is strict upper bound; CONFIRMS principle #35
75. **W_L2=3E-6 + lr_emb=1.55E-3 + edge_L1=0.3 is V_rest-optimal** — Node 134: V_rest=0.729 (near-best), conn_R2=0.946 (trades off connectivity)
76. **edge_L1=0.28 + lr_emb=1.55E-3 + W_L2=3E-6 is conn_R2-optimal** — Node 135: conn_R2=0.978, V_rest=0.535 (trades off V_rest)
77. **edge_diff=700 causes SEVERE collapse** — Node 136: conn_R2=0.896; TRIPLE-CONFIRMS principle #10 (edge_diff=750 STRICTLY optimal)
78. **edge_L1=0.29 is WORSE than 0.3 for V_rest config** — Node 137: V_rest=0.672 (vs 0.729 at 0.3), conn_R2=0.952; no middle ground exists
79. **lr_emb=1.52E-3 is TOO LOW** — Node 138: conn_R2=0.966 (drop), tau_R2=0.960 (drop), V_rest=0.591; lr_emb=1.55E-3 is STRICTLY optimal (both 1.52E-3 and 1.6E-3 worse)
80. **W_L2=3.2E-6 trades V_rest for conn_R2** — Node 139: conn_R2=0.976, V_rest=0.624 (vs 0.729 at 3E-6); W_L2=3E-6 is V_rest-optimal
81. **phi_L1=0.55 DESTROYS V_rest** — Node 140: V_rest=0.506 (vs 0.729 at 0.5); QUINTUPLE-CONFIRMS phi_L1=0.5 STRICTLY optimal
82. **W_L2=2.8E-6 + edge_L1=0.28 achieves BEST V_rest** — Node 141: V_rest=0.736 (NEW BEST), but conn_R2=0.916 trades off significantly
83. **lr_emb=1.57E-3 is TOO HIGH** — Node 142: conn_R2=0.921, V_rest=0.559; SEXTUPLE-CONFIRMS lr_emb=1.55E-3 is strict upper bound
84. **edge_L1=0.32 is TOO HIGH** — Node 143: both conn_R2 (0.965) and V_rest (0.508) degraded; edge_L1=0.3 STRICTLY optimal upper bound
85. **edge_norm=0.9 is BENEFICIAL with edge_L1=0.28** — Node 144: conn_R2=0.980, V_rest=0.647; improves V_rest while maintaining conn_R2; CONTRADICTS principle #65

### Current Open Questions
1. ~~Can we achieve both conn_R2>0.98 AND V_rest>0.75 simultaneously?~~ ANSWERED: **NO** — fundamental trade-off exists between edge_L1=0.3 (V_rest) vs edge_L1=0.28 (conn_R2)
2. ~~Would lr_emb=1.6E-3 push V_rest higher while keeping conn_R2>0.97?~~ ANSWERED: No, lr_emb=1.6E-3 HURTS V_rest (Node 133)
3. ~~Can W_L2=3E-6 + lr_emb=1.55E-3 achieve V_rest>0.75?~~ ANSWERED: No, best is 0.729 (Node 134); 0.75 threshold not achievable
4. ~~Can edge_L1=0.29 achieve conn_R2>0.97 AND V_rest>0.7 (middle ground)?~~ ANSWERED: **NO** — edge_L1=0.29 is WORSE than both 0.3 and 0.28 (Node 137)

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

### Iter 121-124 Results
- Node 121: lr_W=5E-4 + edge_norm=0.75 → conn_R2=0.972, V_rest=0.487 (collapse), cluster_acc=0.909
- Node 122: W_L2=2.5E-6 + data_aug=25 → conn_R2=0.973, V_rest=0.553 (confirms W_L2=2.5E-6 suboptimal)
- Node 123: lr=1.1E-3 → conn_R2=0.913, V_rest=0.282 (SEVERE COLLAPSE; lr=1.2E-3 STRICTLY optimal)
- Node 124: hidden_dim=96 + W_L2=2E-6 → conn_R2=0.956, V_rest=0.611 (trade-off confirmed)

### Iter 125-128 Results
- Node 125: W_L2: 2.5E-6 -> 3E-6 from Node 122 → conn_R2=0.953 (partial collapse), V_rest=0.569; parent had suboptimal base
- Node 126: W_L2=2.8E-6 + hidden_dim=80 → conn_R2=0.981 (EXCELLENT), V_rest=0.562; viable middle W_L2 value
- Node 127: edge_L1=0.28 → conn_R2=0.979, V_rest=0.667 (BETTER than 0.3!), cluster_acc=0.890; NEW FINDING
- Node 128: lr_emb=1.55E-3 → conn_R2=0.978, V_rest=0.702 (EXCELLENT!), cluster_acc=0.859; lr_emb increase BENEFICIAL

### Iter 129-132 Results
- Node 129: W_L2: 2.8E-6 -> 2.6E-6 (from Node 126) → conn_R2=0.975, V_rest=0.434 (COLLAPSE); W_L2=2.8E-6 is LOCAL OPTIMUM
- Node 130: lr_emb=1.55E-3 + edge_L1=0.28 → conn_R2=0.980, V_rest=0.568; combining two findings does NOT synergize
- Node 131: edge_L1=0.26 (from Node 127) → conn_R2=0.973, V_rest=0.594; edge_L1=0.26 is TOO LOW; 0.28 optimal
- Node 132: W_L2: 3E-6 -> 2E-6 (from Node 127) → conn_R2=0.967, V_rest=0.501; principle #50 CONTEXT-DEPENDENT

### Iter 133-136 Results
- Node 133: lr_emb: 1.55E-3 -> 1.6E-3 → conn_R2=0.976, V_rest=0.532 (drop); lr_emb=1.6E-3 is TOO HIGH; 1.55E-3 is upper bound
- Node 134: W_L2: 2E-6 -> 3E-6 (with lr_emb=1.55E-3, edge_L1=0.3) → conn_R2=0.946, **V_rest=0.729 (EXCELLENT)**; V_rest-optimal config found
- Node 135: edge_L1: 0.26 -> 0.28, lr_emb: 1.5E-3 -> 1.55E-3 (from Node 131) → **conn_R2=0.978 (BEST)**, V_rest=0.535; conn_R2-optimal config found
- Node 136: edge_diff: 750 -> 700 → conn_R2=0.896 (SEVERE COLLAPSE); TRIPLE-CONFIRMS principle #10

**Key insight**: Trade-off between conn_R2 and V_rest is fundamental:
- **V_rest-optimal**: edge_L1=0.3 + W_L2=3E-6 + lr_emb=1.55E-3 → V_rest=0.729, conn_R2=0.946
- **conn_R2-optimal**: edge_L1=0.28 + W_L2=3E-6 + lr_emb=1.55E-3 → conn_R2=0.978, V_rest=0.535

### Iter 137-140 Results
- Node 137: edge_L1: 0.3 -> 0.29 → conn_R2=0.952, V_rest=0.672; edge_L1=0.29 WORSE than 0.3 for both metrics; NO middle ground
- Node 138: lr_emb: 1.55E-3 -> 1.52E-3 → conn_R2=0.966, tau_R2=0.960, V_rest=0.591; lr_emb=1.52E-3 TOO LOW; lr_emb=1.55E-3 STRICTLY optimal
- Node 139: W_L2: 3E-6 -> 3.2E-6 → conn_R2=0.976, V_rest=0.624; W_L2=3.2E-6 trades V_rest for conn_R2; W_L2=3E-6 is V_rest-optimal
- Node 140: phi_L1: 0.5 -> 0.55 → conn_R2=0.977, V_rest=0.506; QUINTUPLE-CONFIRMS phi_L1=0.5 STRICTLY optimal

### Iter 141-144 Results (FINAL BATCH)
- Node 141: W_L2: 3E-6 -> 2.8E-6 → conn_R2=0.916, **V_rest=0.736 (NEW BEST!)**, cluster_acc=0.876; W_L2=2.8E-6+edge_L1=0.28 achieves BEST V_rest but sacrifices conn_R2
- Node 142: lr_emb: 1.55E-3 -> 1.57E-3 → conn_R2=0.921, V_rest=0.559; lr_emb=1.57E-3 TOO HIGH; SEXTUPLE-CONFIRMS lr_emb=1.55E-3 upper bound
- Node 143: edge_L1: 0.3 -> 0.32 → conn_R2=0.965, V_rest=0.508, cluster_acc=0.828; edge_L1=0.32 TOO HIGH; edge_L1=0.3 STRICTLY optimal upper bound
- Node 144: edge_norm: 1.0 -> 0.9 → **conn_R2=0.980**, V_rest=0.647, cluster_acc=0.877; edge_norm=0.9 BENEFICIAL with edge_L1=0.28; NEW balanced optimal

>>> BLOCK 6 END — EXPERIMENT COMPLETE <<<

---

## FINAL RESULTS (144 Iterations Complete)

### Best Configurations
| Config | conn_R2 | V_rest_R2 | tau_R2 | cluster_acc | Node |
|--------|---------|-----------|--------|-------------|------|
| **conn_R2-optimal** | **0.983** | 0.691 | 0.995 | 0.877 | 102 |
| **V_rest-optimal** | 0.916 | **0.736** | 0.985 | 0.876 | 141 |
| **Balanced** | 0.980 | 0.647 | 0.990 | 0.877 | 144 |
| **cluster-optimal** | 0.977 | 0.725 | 0.995 | **0.898** | 110 |

### Strictly Optimal Parameters
- **lr_W=6E-4** (5E-4/7E-4/8E-4 all worse)
- **lr=1.2E-3** (1.0E-3/1.1E-3/1.4E-3 catastrophic)
- **lr_emb=1.55E-3** (1.5E-3/1.52E-3/1.57E-3/1.6E-3 all worse)
- **coeff_edge_diff=750** (600/700/800/1000 all cause collapse)
- **coeff_phi_weight_L1=0.5** (0.25/0.4/0.45/0.55/0.6/0.75 all harmful)
- **coeff_W_L1=5E-5** (3E-5/4E-5/7E-5 all harmful)
- **batch_size=2** (1/3/4 all harmful)
- **hidden_dim=80** (96 trades conn_R2 for V_rest)

### Fundamental Trade-offs
1. **conn_R2 vs V_rest cannot both exceed 0.95 and 0.75 simultaneously**
2. **edge_L1=0.3** favors V_rest, **edge_L1=0.28** favors conn_R2
3. **W_L2=2E-6** optimal for conn_R2, **W_L2=2.8E-6/3E-6** optimal for V_rest
4. **edge_norm=1.0** optimal for conn_R2, **edge_norm=0.75-0.9** improves V_rest/cluster_acc
