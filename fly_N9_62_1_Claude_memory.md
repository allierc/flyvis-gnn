# FlyVis Working Memory: fly_N9_62_1 (parallel)

## Knowledge Base (accumulated across all blocks)

### Parameter Effects Table
| Block | Focus | Best conn_R2 | Best tau_R2 | Best V_rest_R2 | Best Cluster_Acc | Time_min | Key finding |
| ----- | ----- | ------------ | ----------- | -------------- | ---------------- | -------- | ----------- |
| 1 | Learning rates | 0.978 (Node 18) | 0.997 (Node 23) | 0.817 (Node 23) | 0.900 (Node 19) | 48-55 | lr_W=5E-4 to 7E-4 optimal; lr_emb=1.5E-3 required |
| 2 | Regularization | 0.980 (Node 43) | 0.997 (Node 30/34) | 0.760 (Node 30) | 0.910 (Node 34) | 48-51 | phi_L1=0.5 + edge_L1=0.5 beneficial; edge_diff=750-1000 optimal |
| 3 | Architecture | **0.981 (Node 67)** | 0.996 (Node 67) | **0.819 (Node 68)** | **0.914 (Node 66)** | 50-56 | lr_W=6E-4+edge_L1=0.3 best; hidden_dim=80+80 optimal |
| 4 | Batch & Aug | 0.981 (Node 82) | 0.994 (Node 73) | 0.739 (Node 73) | 0.913 (Node 74) | 34-45 | batch=2+data_aug=20 optimal; data_aug=18 fastest |
| 5 | Recurrent | **0.983 (Node 102)** | 0.995 (Node 104) | **0.691 (Node 102)** | **0.888 (Node 104)** | 37-38 | Recurrent HARMFUL; W_L2=2E-6 OPTIMAL |

### Established Principles
1. **lr_W=6E-4 with edge_L1=0.3 achieves best conn_R2** — Node 67 (conn_R2=0.981) beats Node 62 (0.977)
2. **lr_W=1E-3 requires lr=1E-3 (not 1.2E-3)** — lr=1.2E-3 with lr_W=1E-3 causes severe conn_R2 degradation
3. **lr_emb=1.5E-3 is required for lr_W < 1E-3** — lower lr_emb causes connectivity collapse
4. **lr_emb >= 1.8E-3 destroys V_rest recovery** — Node 24: V_rest_R2=0.007
5. **Low lr_emb (5E-4) favors cluster_acc over V_rest** — Node 16: cluster_acc=0.897, V_rest_R2=0.401
6. **coeff_edge_norm >= 10 is catastrophic** — Node 27: tau_R2=0.473, V_rest_R2=0.095
7. **coeff_edge_weight_L1=0.3 is optimal** — Node 67: conn_R2=0.981; edge_L1=0.2 collapses V_rest; edge_L1=0.35 hurts conn_R2
8. **coeff_phi_weight_L1=0.5 improves V_rest recovery** — Node 30: V_rest_R2=0.760, tau_R2=0.997
9. **Combined phi_L1=0.5 + edge_L1=0.3 achieves best connectivity** — Node 67: conn_R2=0.981
10. **coeff_edge_diff=750 is optimal** — edge_diff=600/700/800/1000 all worse; MULTIPLY CONFIRMED (Node 70, 84, 96)
11. **coeff_W_L1=5E-5 is optimal for V_rest** — W_L1=3E-5 hurts V_rest (0.674); W_L1=7E-5 slightly worse; CONFIRMED
12. **coeff_edge_diff=1250+ is harmful** — V_rest collapse
13. **coeff_phi_weight_L2 must stay at 0.001** — phi_L2=0.005 destroys tau_R2 and V_rest
14. **coeff_phi_weight_L1=0.5 is optimal** — phi_L1=0.25/0.45/0.6/0.75 all worse; MULTIPLY CONFIRMED
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
42. **batch_size=1 is worse than batch_size=2 for V_rest** — Node 88: V_rest=0.484 vs Node 79's 0.716; CONFIRMED
43. **lr=1.4E-3 causes V_rest collapse** — Node 89: V_rest=0.356, conn_R2=0.959; lr=1.2E-3 optimal
44. **edge_L1=0.25 is viable with lr_W=5E-4** — Node 90: conn_R2=0.977, V_rest=0.542 (slight improvement)
45. **lr_W=7E-4 is worse than 6E-4** — Node 91: conn_R2=0.970, V_rest=0.565; lr_W=6E-4 confirmed optimal
46. **lr=1.0E-3 causes severe collapse** — Node 94: conn_R2=0.888, V_rest=0.324; lr=1.2E-3 CONFIRMED optimal
47. **recurrent_training=True is HARMFUL** — Nodes 97-99: all show severe conn_R2/V_rest/cluster_acc collapse; time_step=2 already harmful
48. **time_step=4 is catastrophic** — Node 99: conn_R2=0.731, time=78.7 min (EXCEEDS LIMIT); DO NOT use recurrent training
49. **coeff_W_L2=1E-5 slightly hurts conn_R2** — Node 100: conn_R2=0.955 (vs 0.98 baseline), V_rest=0.615; may be too strong
50. **coeff_W_L2=2E-6 is OPTIMAL** — Node 102: conn_R2=0.983, V_rest=0.691; preserves conn_R2 while improving V_rest; BETTER than no W_L2
51. **coeff_W_L2=5E-6 is too weak** — Node 101: conn_R2=0.966, V_rest=0.505; doesn't help V_rest
52. **coeff_edge_norm=0.5 helps cluster_acc but hurts V_rest** — Node 104: cluster_acc=0.888, V_rest=0.518; trade-off with edge_norm=1.0

### Current Open Questions
1. ~~Does recurrent_training=True with time_step=2-4 improve V_rest recovery?~~ ANSWERED: NO — harmful
2. ~~Can coeff_W_L2 at lower values (5E-6, 2E-6) provide benefits without hurting conn_R2?~~ ANSWERED: YES — W_L2=2E-6 is optimal
3. Can we achieve both conn_R2>0.98 AND V_rest>0.75 simultaneously?
4. ~~What additional regularization tuning can improve V_rest while maintaining conn_R2?~~ ANSWERED: W_L2=2E-6 improves to 0.691
5. Can combining W_L2=2E-6 with other tuning achieve V_rest>0.75?

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

Key findings:
- batch_size=2 is optimal — batch=3/4 causes V_rest collapse
- data_aug=20 optimal for speed+quality — data_aug=18 fastest viable (35 min)
- lr_W=6E-4 confirmed optimal — 5E-4/7E-4/8E-4 all worse
- lr=1.2E-3 confirmed optimal — 1.0E-3/1.4E-3 cause collapse
- lr_emb=1.5E-3 confirmed optimal — 1.4E-3/1.6E-3 cause collapse
- edge_L1=0.3 optimal with lr_W=6E-4; edge_L1=0.25 optimal with lr_W=5E-4; edge_L1=0.2 too low
- phi_L1=0.5 confirmed optimal — 0.45/0.6 both worse
- edge_diff=750 confirmed optimal — 700/800 both worse
- W_L1=5E-5 confirmed optimal — 7E-5 slightly worse

Final iterations (93-96):
- Node 93: edge_L1=0.2 → conn_R2=0.916 (collapse); CONFIRMS edge_L1=0.25 lower bound
- Node 94: lr=1.0E-3 → conn_R2=0.888, V_rest=0.324 (collapse); lr=1.2E-3 CONFIRMED
- Node 95: W_L1=7E-5 → conn_R2=0.977, V_rest=0.536; W_L1=5E-5 is optimal
- Node 96: edge_diff=800 → conn_R2=0.966, V_rest=0.485; CONFIRMS edge_diff=750

---

## Current Block (Block 5)

### Block Info
Focus: Recurrent Training (recurrent_training, time_step, coeff_W_L2)
Starting from Node 79: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, edge_diff=750, phi_L1=0.5, edge_L1=0.3, W_L1=5E-5, hidden_dim=80, hidden_dim_update=80, batch=2, data_aug=20

### Block 5 Best So Far
- **conn_R2**: 0.983 (Node 102) — coeff_W_L2=2E-6, recurrent=False
- **V_rest_R2**: 0.691 (Node 102) — coeff_W_L2=2E-6
- **tau_R2**: 0.995 (Node 104) — edge_norm=0.5
- **cluster_acc**: 0.888 (Node 104) — edge_norm=0.5

### Iterations This Block (97-120)

**Iter 97**: recurrent=True, time_step=2, data_aug=20 → conn_R2=0.904, V_rest=0.216, tau=0.907, cluster=0.758, time=53 min
**Iter 98**: recurrent=True, time_step=2, data_aug=18 → conn_R2=0.888, V_rest=0.018, tau=0.821, cluster=0.753, time=46.8 min
**Iter 99**: recurrent=True, time_step=4, data_aug=20 → conn_R2=0.731, V_rest=0.028, tau=0.935, cluster=0.695, time=78.7 min (FAIL)
**Iter 100**: W_L2=1E-5, recurrent=False → conn_R2=0.955, V_rest=0.615, tau=0.985, cluster=0.853, time=38.5 min
**Iter 101**: W_L2=5E-6 → conn_R2=0.966, V_rest=0.505, tau=0.986, cluster=0.854, time=37.8 min (worse than W_L2=2E-6)
**Iter 102**: W_L2=2E-6 → conn_R2=0.983, V_rest=0.691, tau=0.992, cluster=0.873, time=37.8 min (BEST W_L2 value)
**Iter 103**: baseline (no W_L2) → conn_R2=0.980, V_rest=0.619, tau=0.991, cluster=0.871, time=38.0 min
**Iter 104**: edge_norm=0.5 → conn_R2=0.979, tau=0.995, V_rest=0.518, cluster=0.888, time=37.7 min (trade-off: cluster up, V_rest down)

**Key findings (Iter 97-104):**
- Recurrent training is HARMFUL — all time_step values cause severe degradation
- coeff_W_L2=2E-6 is OPTIMAL: conn_R2=0.983 (best), V_rest=0.691 (best)
- coeff_W_L2=5E-6 too weak — doesn't help V_rest
- coeff_W_L2=1E-5 too strong — hurts conn_R2
- edge_norm=0.5 improves cluster_acc (0.888) and tau_R2 (0.995) but hurts V_rest (0.518)

### Next Batch Plan (Iter 105-108)
| Slot | Role | Parent | Focus | Mutation |
| ---- | ---- | ------ | ----- | -------- |
| 0 | exploit | Node 102 | W_L2 fine-tune | coeff_W_L2: 2E-6 -> 3E-6 (test between 2E-6 and 5E-6) |
| 1 | exploit | Node 102 | combine | Add edge_norm=0.75 (balance cluster_acc and V_rest with W_L2=2E-6) |
| 2 | explore | Node 102 | W_L1 tuning | coeff_W_L1: 5E-5 -> 4E-5 (test combined with W_L2=2E-6) |
| 3 | principle-test | Node 102 | phi_L1 | coeff_phi_weight_L1: 0.5 -> 0.55. Testing principle: "phi_L1=0.5 is optimal" |
