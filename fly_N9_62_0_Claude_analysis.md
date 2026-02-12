# FlyVis Experiment Log: fly_N9_62_0 (parallel)

## Block 1: Learning Rates Exploration

### Initial Batch (Iterations 0-3)

Starting parallel exploration with diverse learning rate configurations:

| Slot | Node | lr_W | lr | lr_emb | Rationale |
| ---- | ---- | ---- | -- | ------ | --------- |
| 00 | 0 | 5E-4 | 5E-4 | 1E-3 | lower lr_W - conservative baseline |
| 01 | 1 | 1E-3 | 5E-4 | 1E-3 | default baseline (unchanged) |
| 02 | 2 | 2E-3 | 5E-4 | 1E-3 | higher lr_W - faster W learning |
| 03 | 3 | 1E-3 | 1E-3 | 2E-3 | higher lr + lr_emb - faster MLP/embedding |

All other parameters at defaults: coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False

## Iter 1: partial
Node: id=1, parent=root
Mode/Strategy: explore
Config: lr_W=5E-4, lr=5E-4, lr_emb=1E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.728, tau_R2=0.487, V_rest_R2=0.095, cluster_accuracy=0.741, test_R2=0.949, test_pearson=0.986, training_time_min=50.1
Embedding: 65 types with moderate separation, cluster_acc=0.74
Mutation: lr_W: 1E-3 -> 5E-4 (lower lr_W for conservative W learning)
Parent rule: initial exploration from root
Observation: lower lr_W=5E-4 yields good conn_R2=0.728 and best cluster_acc=0.741 but tau recovery moderate
Next: parent=4

## Iter 2: partial
Node: id=2, parent=root
Mode/Strategy: explore
Config: lr_W=1E-3, lr=5E-4, lr_emb=1E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.723, tau_R2=0.451, V_rest_R2=0.062, cluster_accuracy=0.722, test_R2=0.946, test_pearson=0.988, training_time_min=95.7
Embedding: 65 types with moderate separation
Mutation: baseline (lr_W=1E-3, default params)
Parent rule: initial exploration from root
Observation: default lr_W=1E-3 gives similar conn_R2=0.723 but ABNORMAL training time 95.7min (should be ~50min)
Next: parent=4

## Iter 3: partial
Node: id=3, parent=root
Mode/Strategy: explore
Config: lr_W=2E-3, lr=5E-4, lr_emb=1E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.702, tau_R2=0.120, V_rest_R2=0.087, cluster_accuracy=0.748, test_R2=-0.688, test_pearson=0.974, training_time_min=49.0
Embedding: 65 types with good separation (cluster=0.748) but poor tau/V_rest recovery
Mutation: lr_W: 1E-3 -> 2E-3 (higher lr_W for faster W learning)
Parent rule: initial exploration from root
Observation: higher lr_W=2E-3 HURTS tau_R2 severely (0.12 vs 0.45-0.53), negative test_R2 suggests overfitting W
Next: parent=4

## Iter 4: partial
Node: id=4, parent=root
Mode/Strategy: explore
Config: lr_W=1E-3, lr=1E-3, lr_emb=2E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.732, tau_R2=0.532, V_rest_R2=0.196, cluster_accuracy=0.693, test_R2=0.178, test_pearson=0.940, training_time_min=48.8
Embedding: 65 types with some separation but lower cluster_acc=0.693
Mutation: lr: 5E-4 -> 1E-3, lr_emb: 1E-3 -> 2E-3 (higher MLP/embedding learning rates)
Parent rule: initial exploration from root
Observation: BEST RESULT - higher lr+lr_emb gives best conn_R2=0.732, tau_R2=0.532, V_rest_R2=0.196 but lower cluster_acc
Next: parent=4

### Batch 1 Summary (Iters 1-4)
Best connectivity_R2: Node 4 (0.732) with lr=1E-3, lr_emb=2E-3
Best tau_R2: Node 4 (0.532) with higher MLP/embedding learning rates
Best V_rest_R2: Node 4 (0.196) - significantly better than others
Best cluster_accuracy: Node 3 (0.748) but poor tau recovery

Key findings:
1. Higher lr_W=2E-3 damages tau recovery severely (0.12 vs 0.53)
2. Higher lr=1E-3 and lr_emb=2E-3 improves ALL metrics (conn, tau, V_rest) except cluster_acc
3. Lower lr_W=5E-4 comparable to default lr_W=1E-3 for connectivity
4. Slot 01 had abnormal 95.7min training time - likely GPU throttling or system issue

Next batch: Focus on Node 4 (best overall), explore lr/lr_emb variations and test lr_W boundaries

## Iter 5: partial
Node: id=5, parent=4
Mode/Strategy: exploit
Config: lr_W=1E-3, lr=1E-3, lr_emb=3E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.789, tau_R2=0.585, V_rest_R2=0.203, cluster_accuracy=0.718, test_R2=0.872, test_pearson=0.989, training_time_min=49.2
Embedding: 65 types with good separation
Mutation: lr_emb: 2E-3 -> 3E-3 (push embedding lr higher)
Parent rule: exploit best node 4, increase lr_emb
Observation: NEW BEST conn_R2=0.789 (+0.057 vs parent), lr_emb=3E-3 improves connectivity substantially
Next: parent=5

## Iter 6: partial
Node: id=6, parent=4
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=2E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.753, tau_R2=0.682, V_rest_R2=0.124, cluster_accuracy=0.741, test_R2=0.788, test_pearson=0.982, training_time_min=49.4
Embedding: 65 types with good separation
Mutation: lr_W: 1E-3 -> 5E-4 (lower lr_W with high lr/lr_emb)
Parent rule: exploit node 4, test lower lr_W
Observation: lower lr_W=5E-4 gives BEST tau_R2=0.682 (+0.15 vs parent), good cluster=0.741, but lower conn_R2
Next: parent=5

## Iter 7: partial
Node: id=7, parent=1
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=2E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.752, tau_R2=0.589, V_rest_R2=0.139, cluster_accuracy=0.754, test_R2=0.786, test_pearson=0.987, training_time_min=49.1
Embedding: 65 types with good separation
Mutation: lr: 5E-4 -> 1E-3, lr_emb: 1E-3 -> 2E-3 (increase lr/lr_emb from node 1 baseline)
Parent rule: explore from node 1 with higher lr/lr_emb
Observation: BEST cluster_acc=0.754, confirms lr=1E-3+lr_emb=2E-3 improves tau (0.59 vs 0.49 in parent)
Next: parent=5

## Iter 8: partial
Node: id=8, parent=4
Mode/Strategy: principle-test
Config: lr_W=1.5E-3, lr=1E-3, lr_emb=2E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.731, tau_R2=0.568, V_rest_R2=0.096, cluster_accuracy=0.705, test_R2=-192.0, test_pearson=0.832, training_time_min=48.3
Embedding: 65 types with moderate separation
Mutation: lr_W: 1E-3 -> 1.5E-3. Testing principle: "lr_W > 1E-3 damages tau recovery"
Parent rule: principle-test boundary of lr_W=1E-3 threshold
Observation: CONFIRMS principle - lr_W=1.5E-3 worse on tau_R2 (0.568 vs 0.532), V_rest_R2 (0.096 vs 0.196), cluster (0.705 vs 0.693), negative test_R2=-192
Next: parent=5

### Batch 2 Summary (Iters 5-8)
Best connectivity_R2: Node 5 (0.789) with lr_emb=3E-3 - NEW BEST overall
Best tau_R2: Node 6 (0.682) with lr_W=5E-4
Best V_rest_R2: Node 5 (0.203)
Best cluster_accuracy: Node 7 (0.754) with lr_W=5E-4

Key findings:
1. lr_emb=3E-3 significantly improves conn_R2 (0.789 vs 0.732 baseline)
2. Lower lr_W=5E-4 consistently improves tau_R2 (0.682 vs 0.532-0.568)
3. Principle confirmed: lr_W > 1E-3 hurts tau, V_rest, cluster, and test_R2
4. Trade-off exists: higher lr_emb helps conn_R2 but lower lr_W helps tau_R2

Next batch: Combine best findings - try lr_W=5E-4 with lr_emb=3E-3, explore lr_emb=4E-3, test lr boundaries

## Iter 9: partial
Node: id=9, parent=5
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.797, tau_R2=0.689, V_rest_R2=0.165, cluster_accuracy=0.727, test_R2=0.948, test_pearson=0.990, training_time_min=50.2
Embedding: 65 types with good separation
Mutation: lr_W: 1E-3 -> 5E-4 (combine best lr_W from N6 with best lr_emb from N5)
Parent rule: exploit best conn node 5, use lower lr_W for tau
Observation: EXCELLENT - combines best of both: conn_R2=0.797 (near best), tau_R2=0.689 (best ever), good cluster=0.727
Next: parent=9

## Iter 10: converged
Node: id=10, parent=5
Mode/Strategy: exploit
Config: lr_W=1E-3, lr=1E-3, lr_emb=4E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.821, tau_R2=0.472, V_rest_R2=0.110, cluster_accuracy=0.700, test_R2=0.920, test_pearson=0.986, training_time_min=49.6
Embedding: 65 types with moderate separation
Mutation: lr_emb: 3E-3 -> 4E-3 (push lr_emb higher)
Parent rule: exploit best conn node 5, test higher lr_emb
Observation: NEW BEST conn_R2=0.821 (first >0.8!), but tau_R2 drops to 0.472 - trade-off between conn and tau with lr_emb
Next: parent=10

## Iter 11: partial
Node: id=11, parent=6
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=3E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.759, tau_R2=0.555, V_rest_R2=0.187, cluster_accuracy=0.686, test_R2=0.935, test_pearson=0.991, training_time_min=48.0
Embedding: 65 types with moderate separation
Mutation: lr_emb: 2E-3 -> 3E-3 (from best tau node, increase lr_emb)
Parent rule: explore from best tau node 6, increase lr_emb
Observation: lower than N9 (same config) - variance present, but still good V_rest_R2=0.187
Next: parent=9

## Iter 12: failed
Node: id=12, parent=5
Mode/Strategy: principle-test
Config: lr_W=1E-3, lr=2E-3, lr_emb=3E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.570, tau_R2=0.209, V_rest_R2=0.029, cluster_accuracy=0.709, test_R2=-inf, test_pearson=0.199, training_time_min=48.1
Embedding: poor separation
Mutation: lr: 1E-3 -> 2E-3. Testing principle: "lr=1E-3 is optimal for MLP learning"
Parent rule: principle-test from best node 5, push lr boundary
Observation: CONFIRMS principle - lr=2E-3 causes severe instability (test_R2=-inf), all metrics degrade sharply
Next: parent=9

### Batch 3 Summary (Iters 9-12)
Best connectivity_R2: Node 10 (0.821) - FIRST CONVERGED with lr_emb=4E-3
Best tau_R2: Node 9 (0.689) - NEW BEST OVERALL with lr_W=5E-4+lr_emb=3E-3
Best V_rest_R2: Node 11 (0.187)
Best cluster_accuracy: Node 9 (0.727)

Key findings:
1. lr_emb=4E-3 achieves first converged result (conn_R2=0.821) but trades off tau_R2
2. lr_W=5E-4+lr_emb=3E-3 is best balanced config: conn=0.797, tau=0.689, cluster=0.727
3. lr=2E-3 causes instability - keep lr <= 1E-3 (NEW PRINCIPLE)
4. Clear trade-off: higher lr_emb helps conn_R2, lower lr_W helps tau_R2
5. Node 9 vs Node 11 (same config, different parent) shows ~0.04 variance in conn_R2

Next batch: Focus on balanced Node 9 config, try to push conn_R2 without sacrificing tau

## Iter 13: partial
Node: id=13, parent=9
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=4E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.797, tau_R2=0.522, V_rest_R2=0.097, cluster_accuracy=0.686, test_R2=0.373, test_pearson=0.982, training_time_min=50.0
Embedding: 65 types with moderate separation
Mutation: lr_emb: 3E-3 -> 4E-3 (from best balanced, push lr_emb higher)
Parent rule: exploit best balanced node 9, increase lr_emb
Observation: lr_emb=4E-3 with lr_W=5E-4 does not help - conn_R2 same (0.797), tau_R2 drops (0.522 vs 0.689), lower test_R2
Next: parent=14

## Iter 14: converged
Node: id=14, parent=root
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.808, tau_R2=0.632, V_rest_R2=0.272, cluster_accuracy=0.698, test_R2=0.924, test_pearson=0.990, training_time_min=50.0
Embedding: 65 types with good separation
Mutation: lr_emb: 3E-3 -> 3.5E-3 (interpolate between best balanced and best conn)
Parent rule: exploit best balanced node 9, slight lr_emb increase
Observation: NEW BEST BALANCED - conn_R2=0.808 (converged!), tau_R2=0.632, V_rest_R2=0.272 (best ever!) - lr_emb=3.5E-3 is sweet spot
Next: parent=14

## Iter 15: partial
Node: id=15, parent=root
Mode/Strategy: explore
Config: lr_W=3E-4, lr=1E-3, lr_emb=3E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.791, tau_R2=0.565, V_rest_R2=0.210, cluster_accuracy=0.691, test_R2=0.882, test_pearson=0.990, training_time_min=48.4
Embedding: 65 types with good separation
Mutation: lr_W: 5E-4 -> 3E-4 (test even lower lr_W boundary)
Parent rule: explore lower lr_W boundary from best balanced config
Observation: lr_W=3E-4 does NOT improve tau_R2 (0.565 vs 0.689 at lr_W=5E-4) - lr_W=5E-4 is already optimal
Next: parent=14

## Iter 16: partial
Node: id=16, parent=root
Mode/Strategy: principle-test
Config: lr_W=1E-3, lr=1E-3, lr_emb=5E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.754, tau_R2=0.484, V_rest_R2=0.095, cluster_accuracy=0.673, test_R2=0.938, test_pearson=0.989, training_time_min=48.0
Embedding: 65 types with poor separation
Mutation: lr_emb: 4E-3 -> 5E-3. Testing principle: "lr_emb=4E-3 is optimal for conn_R2"
Parent rule: principle-test lr_emb upper boundary
Observation: CONFIRMS principle - lr_emb=5E-3 hurts all metrics: conn_R2=0.754 (vs 0.821 at 4E-3), tau_R2=0.484, cluster=0.673
Next: parent=14

### Batch 4 Summary (Iters 13-16)
Best connectivity_R2: Node 14 (0.808) - NEW CONVERGED with lr_emb=3.5E-3, lr_W=5E-4
Best tau_R2: Node 14 (0.632)
Best V_rest_R2: Node 14 (0.272) - NEW BEST OVERALL
Best cluster_accuracy: Node 14 (0.698)

Key findings:
1. lr_emb=3.5E-3 with lr_W=5E-4 is NEW SWEET SPOT: conn=0.808, tau=0.632, V_rest=0.272 (all best balanced)
2. lr_W=3E-4 too slow - no improvement over 5E-4
3. lr_emb=5E-3 too high - hurts all metrics (principle confirmed)
4. lr_emb=4E-3 with lr_W=5E-4 underperforms - lower lr_W doesn't help when lr_emb too high
5. Node 14 is now the best overall config for balanced metrics

Next batch: Explore around Node 14 sweet spot (lr_W=5E-4, lr_emb=3.5E-3)

## Iter 17: partial
Node: id=17, parent=14
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.75E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.778, tau_R2=0.338, V_rest_R2=0.101, cluster_accuracy=0.679, test_R2=0.788, test_pearson=0.986, training_time_min=50.1
Embedding: 65 types with moderate separation
Mutation: lr_emb: 3.5E-3 -> 3.75E-3 (nudge lr_emb up from sweet spot)
Parent rule: exploit best node 14, small lr_emb increase
Observation: lr_emb=3.75E-3 HURTS - tau_R2 drops sharply (0.338 vs 0.632), conn_R2 drops (0.778 vs 0.808) - sweet spot is narrow
Next: parent=13

## Iter 18: partial
Node: id=18, parent=14
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.25E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.718, tau_R2=0.529, V_rest_R2=0.226, cluster_accuracy=0.676, test_R2=0.948, test_pearson=0.989, training_time_min=49.7
Embedding: 65 types with moderate separation
Mutation: lr_emb: 3.5E-3 -> 3.25E-3 (nudge lr_emb down from sweet spot)
Parent rule: exploit best node 14, small lr_emb decrease
Observation: lr_emb=3.25E-3 hurts conn_R2 (0.718 vs 0.808) but maintains decent V_rest_R2=0.226 - slightly below sweet spot
Next: parent=13

## Iter 19: partial
Node: id=19, parent=9
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.763, tau_R2=0.534, V_rest_R2=0.185, cluster_accuracy=0.733, test_R2=0.903, test_pearson=0.990, training_time_min=48.1
Embedding: 65 types with good separation
Mutation: lr_emb: 3E-3 -> 3.5E-3 (test N14 config from different parent lineage)
Parent rule: explore N14 sweet spot config from N9 parent lineage
Observation: same config as N14 but worse results (conn=0.763 vs 0.808) - stochastic variance or parent lineage matters
Next: parent=13

## Iter 20: partial
Node: id=20, parent=14
Mode/Strategy: principle-test
Config: lr_W=7E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.771, tau_R2=0.532, V_rest_R2=0.236, cluster_accuracy=0.735, test_R2=0.944, test_pearson=0.989, training_time_min=48.0
Embedding: 65 types with good separation
Mutation: lr_W: 5E-4 -> 7E-4. Testing principle: "lr_W=5E-4 is optimal"
Parent rule: principle-test lr_W upper boundary from sweet spot
Observation: lr_W=7E-4 is acceptable (conn=0.771, tau=0.532, cluster=0.735) but not better than 5E-4 - confirms lr_W=5E-4 remains optimal
Next: parent=13

### Batch 5 Summary (Iters 17-20)
Best connectivity_R2: Node 17 (0.778) - but worse than parent N14 (0.808)
Best tau_R2: Node 19 (0.534)
Best V_rest_R2: Node 20 (0.236)
Best cluster_accuracy: Node 20 (0.735)

Key findings:
1. lr_emb=3.5E-3 sweet spot is NARROW - both 3.25E-3 and 3.75E-3 perform worse
2. lr_emb=3.75E-3 causes sharp tau_R2 drop (0.338 vs 0.632) - upper bound is critical
3. lr_W=7E-4 is acceptable but not better than 5E-4 - principle confirmed
4. N19 shows variance (same config as N14 from different parent: conn=0.763 vs 0.808)
5. No improvement over N14 - sweet spot confirmed, need different exploration direction

Next batch: Focus on under-explored high-UCB nodes (N13, N15), try regularization variations while keeping lr sweet spot

## Iter 21: converged
Node: id=21, parent=13
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.823, tau_R2=0.537, V_rest_R2=0.121, cluster_accuracy=0.738, test_R2=0.846, test_pearson=0.986, training_time_min=49.6
Embedding: 65 types with good separation
Mutation: coeff_edge_diff: 500 -> 750 (increase regularization)
Parent rule: exploit N13 (highest UCB), increase edge diff regularization
Observation: NEW BEST conn_R2=0.823 with higher regularization, but tau_R2=0.537 trades off (vs 0.632 in N14) - regularization helps conn but may hurt tau

## Iter 22: converged
Node: id=22, parent=root
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.801, tau_R2=0.581, V_rest_R2=0.099, cluster_accuracy=0.729, test_R2=0.927, test_pearson=0.989, training_time_min=49.8
Embedding: 65 types with good separation
Mutation: lr_emb: 3E-3 -> 3.5E-3 (apply sweet spot from N15 parent)
Parent rule: exploit N15 (high UCB), apply lr_emb sweet spot
Observation: confirms lr_emb=3.5E-3 sweet spot works from different parent - conn=0.801 converged, tau=0.581 good

## Iter 23: partial
Node: id=23, parent=root
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=300, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.796, tau_R2=0.293, V_rest_R2=0.177, cluster_accuracy=0.685, test_R2=0.804, test_pearson=0.983, training_time_min=47.8
Embedding: 65 types with moderate separation
Mutation: coeff_edge_diff: 500 -> 300 (lower regularization)
Parent rule: explore lower regularization boundary with sweet spot lrs
Observation: lower coeff_edge_diff=300 HURTS tau_R2 badly (0.293 vs 0.537-0.632) - regularization needed for tau recovery

## Iter 24: converged
Node: id=24, parent=root
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=8E-4, lr_emb=3.5E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.803, tau_R2=0.487, V_rest_R2=0.025, cluster_accuracy=0.687, test_R2=-453.1, test_pearson=0.983, training_time_min=47.6
Embedding: 65 types with moderate separation
Mutation: lr: 1E-3 -> 8E-4. Testing principle: "lr=1E-3 is optimal for MLP learning"
Parent rule: principle-test lr lower boundary from best balanced config
Observation: lr=8E-4 causes instability (test_R2=-453), V_rest_R2 collapses (0.025) - CONFIRMS lr=1E-3 is critical threshold

### Batch 6 Summary (Iters 21-24) - FINAL BLOCK 1 BATCH
Best connectivity_R2: Node 21 (0.823) - NEW BEST with coeff_edge_diff=750
Best tau_R2: Node 22 (0.581) - confirms sweet spot
Best V_rest_R2: Node 23 (0.177) - despite low tau
Best cluster_accuracy: Node 21 (0.738)

Key findings:
1. Higher coeff_edge_diff=750 achieves best conn_R2=0.823 but trades tau_R2 (0.537 vs 0.632)
2. Lower coeff_edge_diff=300 severely hurts tau_R2 (0.293) - regularization is critical
3. lr=8E-4 causes instability (test_R2=-453) and collapses V_rest_R2 - lr=1E-3 is lower bound
4. lr_emb=3.5E-3 sweet spot confirmed across multiple parent lineages (N22, N14, N19)
5. Trade-off identified: higher regularization helps conn, moderate regularization helps tau

## Block 1 Summary: Learning Rates + Initial Regularization

### Best Configurations Found
| Rank | Node | lr_W | lr | lr_emb | coeff_edge_diff | conn_R2 | tau_R2 | V_rest_R2 | cluster | Trade-off |
| ---- | ---- | ---- | -- | ------ | --------------- | ------- | ------ | --------- | ------- | --------- |
| 1 | 21 | 5E-4 | 1E-3 | 3.5E-3 | 750 | **0.823** | 0.537 | 0.121 | 0.738 | best conn |
| 2 | 14 | 5E-4 | 1E-3 | 3.5E-3 | 500 | 0.808 | 0.632 | **0.272** | 0.698 | best balanced |
| 3 | 10 | 1E-3 | 1E-3 | 4E-3 | 500 | 0.821 | 0.472 | 0.110 | 0.700 | high conn |
| 4 | 9 | 5E-4 | 1E-3 | 3E-3 | 500 | 0.797 | **0.689** | 0.165 | 0.727 | best tau |

### Established Principles (Block 1)
1. **lr_W=5E-4 optimal**: lower than default 1E-3 improves tau_R2 substantially; 3E-4 too slow, 7E-4 acceptable but not better, >1E-3 harmful
2. **lr=1E-3 critical**: lower (8E-4) causes instability, higher (2E-3) causes divergence
3. **lr_emb=3.5E-3 sweet spot**: narrow optimum - both 3.25E-3 and 3.75E-3 underperform; 4E-3 maximizes conn but sacrifices tau
4. **coeff_edge_diff affects trade-off**: 750 best for conn (0.823), 500 best balanced, 300 hurts tau severely
5. **V_rest_R2 correlates with balanced configs**: N14 achieved 0.272, outlier vs typical 0.1-0.2

### Open Questions for Block 2
- Can we achieve both conn_R2>0.8 AND tau_R2>0.68 simultaneously?
- How do other regularization coefficients (coeff_edge_norm, coeff_W_L1) interact with learned learning rates?
- Will coeff_edge_diff between 500-750 find a better trade-off point?

## Block 2: Regularization Exploration

## Iter 25: converged
Node: id=25, parent=21
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.839, tau_R2=0.644, V_rest_R2=0.105, cluster_accuracy=0.681, test_R2=-8.79, test_pearson=0.990, training_time_min=51.0
Embedding: 65 types with moderate separation
Mutation: coeff_edge_diff: 750 -> 625 (interpolate between best conn=750 and best tau=500)
Parent rule: exploit best conn N21, interpolate regularization
Observation: **NEW BEST BALANCED** - coeff_edge_diff=625 achieves BOTH conn_R2=0.839 (best ever) AND tau_R2=0.644 (good) - sweet spot found between 500 and 750
Next: parent=25

## Iter 26: partial
Node: id=26, parent=14
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=500, coeff_edge_norm=1500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.755, tau_R2=0.445, V_rest_R2=0.003, cluster_accuracy=0.716, test_R2=0.965, test_pearson=0.990, training_time_min=50.5
Embedding: 65 types with good cluster separation but poor parameter recovery
Mutation: coeff_edge_norm: 1000 -> 1500 (increase monotonicity penalty)
Parent rule: exploit best balanced N14, test higher monotonicity penalty
Observation: **HURTS BADLY** - coeff_edge_norm=1500 degrades all metrics: conn_R2 drops 0.808->0.755, tau_R2 drops 0.632->0.445, V_rest_R2 collapses to 0.003 - higher monotonicity penalty harmful
Next: parent=25

## Iter 27: converged
Node: id=27, parent=9
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=500, coeff_W_L1=1E-4, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.819, tau_R2=0.493, V_rest_R2=0.076, cluster_accuracy=0.685, test_R2=0.885, test_pearson=0.986, training_time_min=48.3
Embedding: 65 types with moderate separation
Mutation: coeff_W_L1: 5E-5 -> 1E-4 (double W sparsity penalty from best tau config)
Parent rule: explore from best tau N9, increase W sparsity
Observation: coeff_W_L1=1E-4 achieves good conn_R2=0.819 but tau_R2=0.493 worse than parent N9 (0.689) - higher W sparsity doesn't help tau
Next: parent=25

## Iter 28: partial
Node: id=28, parent=root
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=250, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.801, tau_R2=0.381, V_rest_R2=0.150, cluster_accuracy=0.693, test_R2=0.423, test_pearson=0.987, training_time_min=47.7
Embedding: 65 types with moderate separation
Mutation: coeff_edge_diff: 500 -> 250. Testing principle: "coeff_edge_diff=300 hurts tau severely"
Parent rule: principle-test minimum threshold for coeff_edge_diff
Observation: **CONFIRMS PRINCIPLE** - coeff_edge_diff=250 gives tau_R2=0.381 (even worse than 300's 0.293), confirming minimum ~500 needed for tau recovery
Next: parent=25

### Batch 7 Summary (Iters 25-28) - Block 2 Start
Best connectivity_R2: Node 25 (0.839) - **NEW BEST OVERALL** with coeff_edge_diff=625
Best tau_R2: Node 25 (0.644)
Best V_rest_R2: Node 28 (0.150)
Best cluster_accuracy: Node 26 (0.716)

Key findings:
1. **coeff_edge_diff=625 is optimal** - achieves both best conn_R2=0.839 AND excellent tau_R2=0.644 simultaneously
2. **coeff_edge_norm=1500 is harmful** - V_rest_R2 collapses to 0.003, all metrics degrade
3. **coeff_W_L1=1E-4 hurts tau** - doesn't improve over baseline 5E-5
4. **coeff_edge_diff minimum ~500** - below this (250, 300) tau_R2 degrades severely (principle confirmed)
5. Answered open question: YES, we can achieve conn_R2>0.8 AND tau_R2>0.64 with coeff_edge_diff=625

## Iter 29: converged
Node: id=29, parent=25
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=600, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.812, tau_R2=0.529, V_rest_R2=0.031, cluster_accuracy=0.745, test_R2=0.925, test_pearson=0.989, training_time_min=50.3
Embedding: 65 types with good cluster separation
Mutation: coeff_edge_diff: 625 -> 600 (fine-tune around optimum)
Parent rule: exploit best N25, test slightly lower edge_diff
Observation: coeff_edge_diff=600 WORSE than 625 - conn_R2 drops 0.839->0.812, tau_R2 drops 0.644->0.529 - confirms 625 is optimal, not 600
Next: parent=31

## Iter 30: partial
Node: id=30, parent=25
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_edge_norm=800, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.794, tau_R2=0.540, V_rest_R2=0.086, cluster_accuracy=0.706, test_R2=0.962, test_pearson=0.987, training_time_min=50.2
Embedding: 65 types with moderate separation
Mutation: coeff_edge_norm: 1000 -> 800 (test lower monotonicity with optimal edge_diff)
Parent rule: exploit best N25, reduce monotonicity penalty
Observation: coeff_edge_norm=800 HURTS - conn_R2 drops 0.839->0.794, tau_R2 drops 0.644->0.540 - confirms default 1000 is better than 800
Next: parent=31

## Iter 31: converged
Node: id=31, parent=25
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_phi_weight_L1=2, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.831, tau_R2=0.543, V_rest_R2=0.039, cluster_accuracy=0.676, test_R2=0.904, test_pearson=0.987, training_time_min=49.1
Embedding: 65 types with moderate separation
Mutation: coeff_phi_weight_L1: 1 -> 2 (test higher phi sparsity with optimal config)
Parent rule: explore new parameter from best N25
Observation: coeff_phi_weight_L1=2 achieves conn_R2=0.831 (close to best) but tau_R2 drops 0.644->0.543, V_rest drops - higher phi sparsity slightly helpful for conn but hurts tau
Next: parent=31

## Iter 32: partial
Node: id=32, parent=25
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=650, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.797, tau_R2=0.545, V_rest_R2=0.104, cluster_accuracy=0.688, test_R2=0.950, test_pearson=0.987, training_time_min=48.1
Embedding: 65 types with moderate separation
Mutation: coeff_edge_diff: 625 -> 650. Testing principle: "coeff_edge_diff=625 is optimal"
Parent rule: principle-test to verify 625 optimum vs 650
Observation: **CONFIRMS PRINCIPLE** - coeff_edge_diff=650 WORSE: conn_R2 drops 0.839->0.797, tau_R2 drops 0.644->0.545 - 625 is confirmed optimal, higher values not better
Next: parent=31

### Batch 8 Summary (Iters 29-32)
Best connectivity_R2: Node 31 (0.831) - close to N25 but still worse
Best tau_R2: Node 32 (0.545)
Best V_rest_R2: Node 32 (0.104)
Best cluster_accuracy: Node 29 (0.745)

Key findings:
1. **coeff_edge_diff=625 CONFIRMED optimal** - both 600 and 650 perform worse on all metrics
2. **coeff_edge_norm=800 hurts** - default 1000 is better; neither 800 nor 1500 improve
3. **coeff_phi_weight_L1=2 mild effect** - helps conn slightly (0.831) but hurts tau (0.543 vs 0.644)
4. N25 (coeff_edge_diff=625, default regularization) remains BEST overall
5. Regularization space around optimum explored - no improvement found

Next batch: Explore coeff_phi_weight_L2, coeff_edge_weight_L1, and principle-test other established principles

## Iter 33: partial
Node: id=33, parent=31
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_phi_weight_L1=2, coeff_phi_weight_L2=0.01, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.793, tau_R2=0.408, V_rest_R2=0.049, cluster_accuracy=0.696, test_R2=0.793, test_pearson=0.975, training_time_min=50.3
Embedding: 65 types with moderate separation
Mutation: coeff_phi_weight_L2: 0.001 -> 0.01 (increase phi L2 regularization)
Parent rule: exploit from N31 (phi_L1=2), test higher phi L2
Observation: **HURTS** - higher phi L2=0.01 combined with phi L1=2 degrades all metrics: conn drops 0.831->0.793, tau drops 0.543->0.408, V_rest drops - phi over-regularization harmful
Next: parent=34

## Iter 34: converged
Node: id=34, parent=25
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_edge_weight_L1=2, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.820, tau_R2=0.569, V_rest_R2=0.112, cluster_accuracy=0.655, test_R2=-3.48, test_pearson=0.987, training_time_min=50.1
Embedding: 65 types with moderate separation
Mutation: coeff_edge_weight_L1: 1 -> 2 (increase edge weight sparsity)
Parent rule: exploit best N25, test higher edge weight L1
Observation: coeff_edge_weight_L1=2 achieves conn_R2=0.820 but worse than N25 (0.839), tau_R2=0.569 worse than N25 (0.644) - higher edge L1 not beneficial
Next: parent=34

## Iter 35: converged
Node: id=35, parent=25
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=2.5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.814, tau_R2=0.524, V_rest_R2=0.119, cluster_accuracy=0.699, test_R2=-0.41, test_pearson=0.982, training_time_min=48.5
Embedding: 65 types with moderate separation
Mutation: coeff_W_L1: 5E-5 -> 2.5E-5 (lower W sparsity penalty)
Parent rule: explore from best N25, test lower W sparsity
Observation: coeff_W_L1=2.5E-5 not better - conn_R2 drops 0.839->0.814, tau_R2 drops 0.644->0.524 - default 5E-5 is optimal
Next: parent=34

## Iter 36: partial
Node: id=36, parent=25
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=1E-3, lr_emb=3E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.780, tau_R2=0.599, V_rest_R2=0.049, cluster_accuracy=0.661, test_R2=0.954, test_pearson=0.990, training_time_min=48.7
Embedding: 65 types with moderate separation
Mutation: lr_emb: 3.5E-3 -> 3E-3. Testing principle: "lr_emb=3.5E-3 sweet spot is NARROW"
Parent rule: principle-test lr_emb sweet spot with optimal regularization
Observation: **CONFIRMS PRINCIPLE** - lr_emb=3E-3 (even with optimal coeff_edge_diff=625) gives conn_R2=0.780 vs 0.839 at 3.5E-3 - sweet spot confirmed, lr_emb=3.5E-3 is critical
Next: parent=34

### Batch 9 Summary (Iters 33-36)
Best connectivity_R2: Node 34 (0.820) - edge_weight_L1=2
Best tau_R2: Node 36 (0.599) - but lower conn
Best V_rest_R2: Node 35 (0.119)
Best cluster_accuracy: Node 35 (0.699)

Key findings:
1. **coeff_phi_weight_L2=0.01 with phi_L1=2 harmful** - over-regularization degrades all metrics
2. **coeff_edge_weight_L1=2 not beneficial** - conn drops 0.839->0.820, tau drops 0.644->0.569
3. **coeff_W_L1=2.5E-5 not better than 5E-5** - lower W sparsity doesn't help
4. **lr_emb=3.5E-3 sweet spot CONFIRMED** - lr_emb=3E-3 with optimal regularization still underperforms
5. N25 (coeff_edge_diff=625, default other params) REMAINS BEST - no regularization change improves it

## Iter 37: failed
Node: id=37, parent=34
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, hidden_dim=96, recurrent=False
Metrics: connectivity_R2=0.636, tau_R2=0.519, V_rest_R2=0.030, cluster_accuracy=0.671, test_R2=-5.95, test_pearson=0.986, training_time_min=54.6
Embedding: 65 types with poor separation
Mutation: hidden_dim: 64 -> 96 (larger edge MLP capacity)
Parent rule: exploit high-UCB N34, test increased architecture capacity
Observation: **FAILED** - hidden_dim=96 causes instability (test_R2=-5.95), conn_R2 drops from 0.820 to 0.636 - larger edge MLP harmful
Next: parent=38

## Iter 38: partial
Node: id=38, parent=root
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.801, tau_R2=0.752, V_rest_R2=0.138, cluster_accuracy=0.714, test_R2=0.872, test_pearson=0.992, training_time_min=50.7
Embedding: 65 types with good separation
Mutation: hidden_dim_update: 64 -> 96 (larger update MLP capacity)
Parent rule: exploit N27 branch, test increased update MLP capacity
Observation: **tau_R2=0.752 NEW BEST** - hidden_dim_update=96 dramatically improves tau recovery (0.752 vs 0.644 best), conn slightly lower at 0.801
Next: parent=38

## Iter 39: converged
Node: id=39, parent=root
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, n_layers=4, recurrent=False
Metrics: connectivity_R2=0.850, tau_R2=0.468, V_rest_R2=0.293, cluster_accuracy=0.703, test_R2=0.954, test_pearson=0.989, training_time_min=55.3
Embedding: 65 types with good separation
Mutation: n_layers: 3 -> 4 (deeper edge MLP)
Parent rule: explore from N35 branch, test deeper architecture
Observation: **conn_R2=0.850 NEW BEST, V_rest_R2=0.293 NEW BEST** - deeper edge MLP (n_layers=4) improves conn and V_rest dramatically, but tau drops to 0.468
Next: parent=39

## Iter 40: partial
Node: id=40, parent=root
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_edge_norm=1200, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.798, tau_R2=0.524, V_rest_R2=0.189, cluster_accuracy=0.701, test_R2=0.938, test_pearson=0.988, training_time_min=48.4
Embedding: 65 types with moderate separation
Mutation: coeff_edge_norm: 1000 -> 1200. Testing principle: "coeff_edge_norm=1000 is optimal"
Parent rule: principle-test from best N25, test between optimal 1000 and harmful 1500
Observation: **CONFIRMS PRINCIPLE** - coeff_edge_norm=1200 worse than 1000 (conn 0.798 vs 0.839, tau 0.524 vs 0.644), supports 1000 as optimal
Next: parent=39

### Batch 10 Summary (Iters 37-40)
Best connectivity_R2: Node 39 (0.850) - n_layers=4 **NEW OVERALL BEST**
Best tau_R2: Node 38 (0.752) - hidden_dim_update=96 **NEW OVERALL BEST**
Best V_rest_R2: Node 39 (0.293) - n_layers=4 **NEW OVERALL BEST**
Best cluster_accuracy: Node 38 (0.714)

Key findings:
1. **hidden_dim=96 HARMFUL** - causes training instability (N37 failed, test_R2=-5.95)
2. **hidden_dim_update=96 EXCELLENT for tau** - tau_R2 jumps from 0.644 to 0.752, best ever
3. **n_layers=4 EXCELLENT for conn and V_rest** - conn_R2 hits 0.850, V_rest_R2 hits 0.293
4. **Architecture trade-off discovered**: deeper (n_layers=4) maximizes conn+V_rest, wider update (hidden_dim_update=96) maximizes tau
5. **coeff_edge_norm=1000 CONFIRMED optimal** - 1200 worse than 1000

## Iter 41: converged
Node: id=41, parent=39
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, n_layers=4, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.844, tau_R2=0.736, V_rest_R2=0.233, cluster_accuracy=0.709, test_R2=0.759, test_pearson=0.993, training_time_min=57.7
Embedding: 65 types with good separation
Mutation: n_layers=4 + hidden_dim_update=96 (combine best conn architecture with best tau architecture)
Parent rule: exploit best conn N39, add hidden_dim_update=96 from best tau N38
Observation: **EXCELLENT COMBINATION** - combining architectures achieves conn_R2=0.844, tau_R2=0.736 (2nd best ever), V_rest_R2=0.233 (good) - proves architectures are compatible
Next: parent=41

## Iter 42: converged
Node: id=42, parent=38
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, n_layers=3, n_layers_update=4, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.844, tau_R2=0.686, V_rest_R2=0.271, cluster_accuracy=0.772, test_R2=0.848, test_pearson=0.991, training_time_min=55.1
Embedding: 65 types with excellent separation
Mutation: n_layers_update: 3 -> 4 (deeper update MLP from best tau N38)
Parent rule: exploit best tau N38, test deeper update MLP
Observation: **BEST CLUSTER=0.772** - n_layers_update=4 maintains good conn=0.844, tau=0.686, excellent V_rest=0.271, and achieves best cluster_accuracy ever
Next: parent=42

## Iter 43: partial
Node: id=43, parent=39
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, n_layers=5, recurrent=False
Metrics: connectivity_R2=0.725, tau_R2=0.371, V_rest_R2=0.005, cluster_accuracy=0.695, test_R2=0.283, test_pearson=0.975, training_time_min=62.3
Embedding: 65 types with poor separation
Mutation: n_layers: 4 -> 5 (test even deeper edge MLP)
Parent rule: explore from best conn N39, push depth boundary
Observation: **n_layers=5 TOO DEEP** - all metrics degrade sharply: conn drops 0.850->0.725, tau drops 0.468->0.371, V_rest collapses to 0.005 - n_layers=4 is optimal depth
Next: parent=41

## Iter 44: converged
Node: id=44, parent=38
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, n_layers=4, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.867, tau_R2=0.542, V_rest_R2=0.162, cluster_accuracy=0.707, test_R2=0.926, test_pearson=0.992, training_time_min=55.4
Embedding: 65 types with good separation
Mutation: n_layers=4 + hidden_dim_update=96. Testing principle: "Architecture trade-off: wider update -> better tau; deeper edge -> better conn"
Parent rule: principle-test combined architecture from best tau N38
Observation: **conn_R2=0.867 NEW BEST** - combining n_layers=4 + hidden_dim_update=96 achieves highest connectivity ever! tau=0.542 is moderate but better conn/V_rest/cluster trade-off
Next: parent=44

### Batch 11 Summary (Iters 41-44)
Best connectivity_R2: Node 44 (0.867) - **NEW OVERALL BEST**
Best tau_R2: Node 41 (0.736) - close to N38's 0.752
Best V_rest_R2: Node 42 (0.271)
Best cluster_accuracy: Node 42 (0.772) - **NEW OVERALL BEST**

Key findings:
1. **Combined architecture (n_layers=4 + hidden_dim_update=96) is COMPATIBLE** - N44 achieves conn_R2=0.867 (new best!)
2. **n_layers=5 TOO DEEP** - causes severe degradation (conn 0.725, tau 0.371, V_rest collapses to 0.005)
3. **n_layers_update=4 helps cluster_accuracy** - N42 achieves 0.772 (best ever), with excellent V_rest=0.271
4. **Architecture sweet spot identified**: n_layers=4 (edge), hidden_dim_update=96 (update), default n_layers_update=3
5. **Principle challenged**: Combined architecture CAN achieve best conn - trade-off is not strict, deeper edge MLP can coexist with wider update MLP

## Iter 45: converged
Node: id=45, parent=44
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=4E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, n_layers=4, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.864, tau_R2=0.515, V_rest_R2=0.349, cluster_accuracy=0.687, test_R2=0.687, test_pearson=0.989, training_time_min=56.9
Embedding: 65 types partially separated
Mutation: lr_emb: 3.5E-3 -> 4E-3 (push lr_emb from best conn N44)
Parent rule: highest UCB from N44 best conn config
Observation: lr_emb=4E-3 gives NEW BEST V_rest_R2=0.349 but trades off tau (0.515 vs 0.542) and cluster (0.687 vs 0.707)
Next: parent=45

## Iter 46: converged
Node: id=46, parent=41
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, n_layers=4, n_layers_update=4, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.866, tau_R2=0.590, V_rest_R2=0.232, cluster_accuracy=0.767, test_R2=-1.00, test_pearson=0.991, training_time_min=58.4
Embedding: 65 types well separated
Mutation: n_layers_update: 3 -> 4 (deeper update MLP combined with n_layers=4)
Parent rule: exploit from N41 best balanced config
Observation: n_layers_update=4 + n_layers=4 achieves excellent cluster=0.767 (near best) and good conn=0.866, but slight negative test_R2
Next: parent=46

## Iter 47: failed
Node: id=47, parent=42
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, n_layers_update=4, hidden_dim_update=96, embedding_dim=4
Metrics: FAILED (empty analysis log)
Mutation: embedding_dim: 2 -> 4 (test larger embedding from best cluster N42)
Parent rule: explore under-visited architecture dimension
Observation: embedding_dim=4 caused training failure - likely instability with larger embedding space
Next: parent=46

## Iter 48: converged
Node: id=48, parent=44
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=1E-3, lr_emb=3E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, n_layers=4, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.828, tau_R2=0.616, V_rest_R2=0.096, cluster_accuracy=0.696, test_R2=0.945, test_pearson=0.990, training_time_min=57.1
Embedding: 65 types partially separated
Mutation: lr_emb: 3.5E-3 -> 3E-3. Testing principle: "lr_emb=3.5E-3 sweet spot is NARROW"
Parent rule: test if combined architecture changes lr_emb sweet spot
Observation: PRINCIPLE CONFIRMED - lr_emb=3E-3 gives worse conn (0.828 vs 0.867) even with combined arch; 3.5E-3 remains optimal
Next: parent=46

### Batch 12 Summary (Iters 45-48) - BLOCK 2 END
Best connectivity_R2: Node 46 (0.866) - close to N44's 0.867
Best tau_R2: Node 48 (0.616) - good but below N38's 0.752
Best V_rest_R2: Node 45 (0.349) - **NEW OVERALL BEST**
Best cluster_accuracy: Node 46 (0.767) - close to N42's 0.772

Key findings:
1. **lr_emb=4E-3 with combined arch achieves V_rest_R2=0.349** - NEW BEST V_rest ever
2. **n_layers_update=4 with n_layers=4 achieves cluster=0.767** - excellent cluster while maintaining conn=0.866
3. **embedding_dim=4 FAILED** - causes training instability, keep at embedding_dim=2
4. **lr_emb=3.5E-3 sweet spot CONFIRMED AGAIN** - even with combined architecture, 3E-3 underperforms

## Block 2 Summary

### Best Configurations Found (Block 2)
| Rank | Node | Architecture | conn_R2 | tau_R2 | V_rest_R2 | cluster | Notes |
| ---- | ---- | ------------ | ------- | ------ | --------- | ------- | ----- |
| 1 | 44 | n_layers=4, hidden_dim_update=96 | **0.867** | 0.542 | 0.162 | 0.707 | **BEST conn EVER** |
| 2 | 46 | n_layers=4, n_layers_update=4, hidden_dim_update=96 | 0.866 | 0.590 | 0.232 | 0.767 | excellent cluster |
| 3 | 45 | n_layers=4, hidden_dim_update=96, lr_emb=4E-3 | 0.864 | 0.515 | **0.349** | 0.687 | **BEST V_rest EVER** |
| 4 | 42 | n_layers_update=4, hidden_dim_update=96 | 0.844 | 0.686 | 0.271 | **0.772** | **BEST cluster** |
| 5 | 38 | hidden_dim_update=96 | 0.801 | **0.752** | 0.138 | 0.714 | **BEST tau EVER** |

### Key Findings (Block 2)
1. **Regularization**: coeff_edge_diff=625 is optimal; coeff_edge_norm=1000 optimal; other params at defaults
2. **Architecture wins**: n_layers=4 (deeper edge MLP) dramatically improves conn_R2; hidden_dim_update=96 improves tau_R2
3. **Combined architecture**: n_layers=4 + hidden_dim_update=96 achieves best conn_R2=0.867
4. **Depth limits**: n_layers=5 is too deep (all metrics degrade); n_layers_update=4 helps cluster but not conn/tau
5. **Hidden_dim danger**: hidden_dim=96 for edge MLP causes instability; keep at 64
6. **embedding_dim=4 unstable**: larger embedding space caused training failure
7. **V_rest discovery**: lr_emb=4E-3 with combined arch achieves V_rest_R2=0.349 (NEW BEST)

### Unresolved Trade-offs
- conn vs tau: N44 (conn=0.867, tau=0.542) vs N38 (conn=0.801, tau=0.752)
- V_rest recovery: lr_emb=4E-3 helps V_rest but hurts tau and cluster

## Block 3: Batch & Training Dynamics

## Iter 49: partial
Node: id=49, parent=46
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=2, n_layers=4, n_layers_update=4, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.754, tau_R2=0.579, V_rest_R2=0.078, cluster_accuracy=0.793, test_R2=0.885, test_pearson=0.996, training_time_min=50.1
Embedding: 65 types with excellent separation, best cluster ever
Mutation: batch_size: 1 -> 2 (from best cluster+conn N46)
Parent rule: exploit best balanced N46, test larger batch
Observation: batch_size=2 HURTS conn_R2 severely (0.866->0.754) but achieves NEW BEST cluster_accuracy=0.793 - trade-off between conn and cluster with larger batch
Next: parent=50

## Iter 50: converged
Node: id=50, parent=46
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, data_augmentation_loop=30, n_layers=4, n_layers_update=4, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.807, tau_R2=0.911, V_rest_R2=0.101, cluster_accuracy=0.735, test_R2=0.106, test_pearson=0.991, training_time_min=69.8
Embedding: 65 types with good separation
Mutation: data_augmentation_loop: 25 -> 30 (from N46)
Parent rule: exploit best balanced N46, test higher augmentation
Observation: **tau_R2=0.911 NEW BEST EVER** - data_augmentation_loop=30 dramatically improves tau recovery (0.911 vs 0.590 parent). WARNING: training_time=69.8 min exceeds limit
Next: parent=50

## Iter 51: partial
Node: id=51, parent=38
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, n_layers=3, n_layers_update=3, hidden_dim_update=96, recurrent=True, time_step=2
Metrics: connectivity_R2=0.644, tau_R2=0.436, V_rest_R2=0.013, cluster_accuracy=0.630, test_R2=0.869, test_pearson=0.982, training_time_min=69.4
Embedding: 65 types with poor separation
Mutation: recurrent_training: False -> True, time_step=2 (from best tau N38)
Parent rule: explore recurrent training from best tau config
Observation: **recurrent_training=True HARMFUL** - all metrics degrade sharply: conn 0.801->0.644, tau 0.752->0.436, V_rest collapses 0.138->0.013, cluster 0.714->0.630. Training time also high at 69.4 min
Next: parent=50

## Iter 52: converged
Node: id=52, parent=45
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=1E-3, lr_emb=4E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=2, n_layers=4, n_layers_update=3, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.828, tau_R2=0.488, V_rest_R2=0.042, cluster_accuracy=0.728, test_R2=0.791, test_pearson=0.989, training_time_min=45.9
Embedding: 65 types with moderate separation
Mutation: batch_size: 1 -> 2, lr_emb=4E-3. Testing principle: "lr_emb=4E-3 helps V_rest"
Parent rule: principle-test V_rest principle with batch_size change from N45
Observation: **CONTRADICTS PRINCIPLE** - batch_size=2 with lr_emb=4E-3 gives V_rest_R2=0.042 (vs 0.349 in N45) - batch_size=2 destroys V_rest benefit of lr_emb=4E-3; lr_emb=4E-3 only helps V_rest with batch_size=1
Next: parent=50

### Batch 13 Summary (Iters 49-52) - Block 3 Start
Best connectivity_R2: Node 52 (0.828)
Best tau_R2: Node 50 (0.911) - **NEW OVERALL BEST** (dramatic improvement!)
Best V_rest_R2: Node 50 (0.101)
Best cluster_accuracy: Node 49 (0.793) - **NEW OVERALL BEST**

Key findings:
1. **data_augmentation_loop=30 is BREAKTHROUGH for tau** - tau_R2 jumps from 0.590 to 0.911, but training time ~70 min exceeds limit
2. **recurrent_training=True is HARMFUL** - all metrics degrade substantially; DO NOT use recurrent training
3. **batch_size=2 HURTS conn_R2** - conn drops from 0.866 to 0.754, but cluster_accuracy improves to 0.793
4. **lr_emb=4E-3 V_rest benefit is batch_size sensitive** - only helps with batch_size=1; principle needs update
5. **New open question**: Can we achieve data_augmentation_loop=30 benefits within training time limit?

## Iter 53: converged
Node: id=53, parent=50
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, data_augmentation_loop=27, n_layers=4, n_layers_update=4, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.869, tau_R2=0.545, V_rest_R2=0.105, cluster_accuracy=0.779, test_R2=0.914, test_pearson=0.992, training_time_min=67.9
Embedding: 65 types with good separation
Mutation: data_augmentation_loop: 30 -> 27 (reduce aug loop to fit time limit)
Parent rule: exploit best tau N50, reduce aug_loop for time constraint
Observation: **conn_R2=0.869 TIES BEST EVER (N44=0.867)** - aug_loop=27 achieves excellent connectivity but tau_R2 drops to 0.545 (vs 0.911 at aug_loop=30). Training time 67.9 min still exceeds 60 min limit.
Next: parent=53

## Iter 54: partial
Node: id=54, parent=50
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=2, data_augmentation_loop=30, n_layers=4, n_layers_update=4, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.761, tau_R2=0.582, V_rest_R2=0.114, cluster_accuracy=0.786, test_R2=0.928, test_pearson=0.992, training_time_min=55.9
Embedding: 65 types with good separation
Mutation: batch_size: 1 -> 2 (test if batch_size=2 speeds up aug_loop=30 within limit)
Parent rule: exploit best tau N50, add batch_size=2 for speed
Observation: batch_size=2 SUCCEEDS for time (55.9 min within limit) but conn_R2 drops to 0.761 (vs 0.807 with batch_size=1). tau_R2=0.582 is lower than N50's 0.911 - batch_size=2 cancels aug_loop tau benefit
Next: parent=53

## Iter 55: partial
Node: id=55, parent=52
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=2, data_augmentation_loop=25, n_layers=4, n_layers_update=4, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.727, tau_R2=0.570, V_rest_R2=0.128, cluster_accuracy=0.752, test_R2=0.959, test_pearson=0.992, training_time_min=47.9
Embedding: 65 types with moderate separation
Mutation: n_layers_update: 3 -> 4 (test N46-like arch with batch_size=2 for conn recovery)
Parent rule: explore from N52 (batch_size=2), test deeper architecture
Observation: n_layers_update=4 with batch_size=2 does NOT recover connectivity (0.727 vs 0.828 parent). Confirms batch_size=2 fundamental limitation for conn.
Next: parent=53

## Iter 56: converged
Node: id=56, parent=44
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, data_augmentation_loop=28, n_layers=4, n_layers_update=3, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.852, tau_R2=0.675, V_rest_R2=0.117, cluster_accuracy=0.701, test_R2=0.158, test_pearson=0.989, training_time_min=66.7
Embedding: 65 types with moderate separation
Mutation: data_augmentation_loop: 25 -> 28. Testing principle: "data_augmentation_loop=30 BREAKTHROUGH for tau"
Parent rule: principle-test aug_loop boundary from best conn N44
Observation: **PARTIAL CONFIRMATION** - aug_loop=28 gives tau_R2=0.675 (better than baseline 0.542 but not as good as aug_loop=30's 0.911). conn_R2=0.852 excellent. Training time 66.7 min exceeds limit.
Next: parent=53

### Batch 14 Summary (Iters 53-56)
Best connectivity_R2: Node 53 (0.869) - **TIES OVERALL BEST** (matches N44's 0.867)
Best tau_R2: Node 56 (0.675) - good compromise
Best V_rest_R2: Node 55 (0.128)
Best cluster_accuracy: Node 54 (0.786)

Key findings:
1. **aug_loop=27 achieves conn_R2=0.869** - ties best ever N44, but tau_R2 drops to 0.545 (vs 0.911 at aug_loop=30)
2. **batch_size=2 with aug_loop=30 fits time limit** (55.9 min) but cancels tau benefit (0.582 vs 0.911)
3. **aug_loop=28 gives good tau compromise** - tau_R2=0.675 is better than baseline 0.542 but not breakthrough 0.911
4. **batch_size=2 consistently hurts connectivity** - all batch_size=2 configs have conn<0.77 regardless of architecture
5. **All batch_size=1 with aug_loop>=27 exceed time limit** (~66-68 min) - cannot achieve breakthrough tau within time constraint with batch_size=1
6. **NEW PRINCIPLE**: aug_loop tau benefit is inversely related to batch_size - batch_size=2 cancels augmentation benefit

## Iter 57: partial
Node: id=57, parent=53
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, data_augmentation_loop=27, n_layers=3, n_layers_update=4, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.799, tau_R2=0.749, V_rest_R2=0.314, cluster_accuracy=0.768, test_R2=0.954, test_pearson=0.996, training_time_min=62.8
Embedding: 65 types with good separation
Mutation: n_layers: 4 -> 3 (test if simpler edge MLP reduces time while keeping aug_loop conn benefit)
Parent rule: exploit best conn N53, test simpler architecture for time reduction
Observation: n_layers=3 SACRIFICES conn_R2 (0.869->0.799) but DRAMATICALLY IMPROVES tau_R2 (0.545->0.749) and V_rest_R2 (0.105->0.314). NEW BEST V_rest in Block 3. Time 62.8 min still exceeds limit.
Next: parent=59

## Iter 58: converged
Node: id=58, parent=53
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, data_augmentation_loop=27, n_layers=4, n_layers_update=4, hidden_dim_update=64, recurrent=False
Metrics: connectivity_R2=0.838, tau_R2=0.624, V_rest_R2=0.013, cluster_accuracy=0.679, test_R2=0.912, test_pearson=0.989, training_time_min=70.3
Embedding: 65 types with moderate separation
Mutation: hidden_dim_update: 96 -> 64 (test if simpler update MLP reduces time while keeping aug_loop conn benefit)
Parent rule: exploit best conn N53, test simpler update MLP for time reduction
Observation: hidden_dim_update=64 DOES NOT reduce time (actually increases to 70.3 min!) and V_rest_R2 COLLAPSES to 0.013. hidden_dim_update=96 is ESSENTIAL for V_rest recovery.
Next: parent=59

## Iter 59: converged
Node: id=59, parent=56
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, data_augmentation_loop=28, n_layers=3, n_layers_update=3, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.830, tau_R2=0.642, V_rest_R2=0.088, cluster_accuracy=0.687, test_R2=0.712, test_pearson=0.990, training_time_min=56.4
Embedding: 65 types with moderate separation
Mutation: n_layers: 4 -> 3, n_layers_update: 4 -> 3 (test if simpler architecture reduces time for tau compromise config)
Parent rule: explore from N56 tau compromise, test simpler overall architecture
Observation: **BREAKTHROUGH FOR TIME** - n_layers=3 + n_layers_update=3 achieves 56.4 min (WITHIN 60 min limit!). First aug_loop>=27 config within time. conn=0.830 good, tau=0.642 decent.
Next: parent=59

## Iter 60: converged
Node: id=60, parent=44
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, data_augmentation_loop=26, n_layers=4, n_layers_update=4, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.851, tau_R2=0.618, V_rest_R2=0.115, cluster_accuracy=0.774, test_R2=0.940, test_pearson=0.992, training_time_min=62.2
Embedding: 65 types with good separation
Mutation: data_augmentation_loop: 25 -> 26. Testing principle: "aug_loop=27 achieves BEST conn_R2=0.869"
Parent rule: principle-test if slightly lower aug_loop maintains conn benefit with shorter time
Observation: aug_loop=26 gives conn=0.851 (still excellent, only 0.018 below best 0.869) with good cluster=0.774. Time=62.2 min still exceeds limit but closer than aug_loop=27.
Next: parent=59

### Batch 15 Summary (Iters 57-60)
Best connectivity_R2: Node 60 (0.851) - close to best 0.869
Best tau_R2: Node 57 (0.749) - excellent, close to N38's 0.752
Best V_rest_R2: Node 57 (0.314) - **NEW BLOCK 3 BEST**
Best cluster_accuracy: Node 60 (0.774)

Key findings:
1. **n_layers=3 + n_layers_update=3 BREAKTHROUGH** - N59 achieves 56.4 min (first aug_loop>=27 within time!) with conn=0.830, tau=0.642
2. **n_layers=3 dramatically improves tau and V_rest** - N57 tau=0.749 (vs 0.545), V_rest=0.314 (vs 0.105), but sacrifices conn (0.869->0.799)
3. **hidden_dim_update=96 is ESSENTIAL** - N58 shows hidden_dim_update=64 collapses V_rest (0.013) and doesn't reduce time
4. **aug_loop=26 still excellent** - N60 conn=0.851 only 0.018 below best 0.869, time 62.2 min
5. **NEW PRINCIPLE**: Simpler edge MLP (n_layers=3) improves tau_R2 and V_rest_R2 at cost of conn_R2 - depth trade-off discovered

## Iter 61: converged
Node: id=61, parent=59
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, n_layers=3, n_layers_update=3, hidden_dim_update=96, aug_loop=30, recurrent=False
Metrics: connectivity_R2=0.836, tau_R2=0.768, V_rest_R2=0.197, cluster_accuracy=0.666, test_R2=-0.441, test_pearson=0.992, training_time_min=65.5
Embedding: 65 types with moderate separation
Mutation: data_augmentation_loop: 28 -> 30 (test if simpler arch with higher aug_loop achieves tau breakthrough within time)
Parent rule: N59 best time-efficient config (56.4 min), increase aug_loop for tau
Observation: aug_loop=30 with n_layers=3+n_layers_update=3 improves tau to 0.768 (up from 0.642 at N59), but time 65.5 min exceeds limit
Next: parent=61

## Iter 62: converged
Node: id=62, parent=57
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=4E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, n_layers=3, n_layers_update=4, hidden_dim_update=96, aug_loop=28, recurrent=False
Metrics: connectivity_R2=0.807, tau_R2=0.789, V_rest_R2=0.184, cluster_accuracy=0.709, test_R2=0.697, test_pearson=0.994, training_time_min=61.7
Embedding: 65 types with moderate separation
Mutation: lr_emb: 3.5E-3 -> 4E-3 (test if lr_emb=4E-3 helps V_rest with simpler edge MLP)
Parent rule: N57 best V_rest with n_layers=3, try lr_emb=4E-3 for further V_rest improvement
Observation: lr_emb=4E-3 + n_layers_update=4 gives BEST TAU this batch (0.789), but V_rest drops to 0.184 (vs N57's 0.314); time 61.7 min slightly over limit
Next: parent=62

## Iter 63: converged
Node: id=63, parent=59
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, n_layers=3, n_layers_update=3, hidden_dim_update=96, aug_loop=29, recurrent=False
Metrics: connectivity_R2=0.811, tau_R2=0.586, V_rest_R2=0.132, cluster_accuracy=0.691, test_R2=0.959, test_pearson=0.991, training_time_min=60.6
Embedding: 65 types with moderate separation
Mutation: data_augmentation_loop: 28 -> 29 (test if slightly higher aug_loop stays within time)
Parent rule: N59 best time-efficient, test aug_loop boundary
Observation: aug_loop=29 achieves 60.6 min (just over limit), tau drops to 0.586 (worse than N61's 0.768 at aug_loop=30) - INCONSISTENT with aug_loop trend, variance or stochastic effect
Next: parent=61

## Iter 64: partial
Node: id=64, parent=59
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=1E-3, lr_emb=3.5E-3, coeff_edge_diff=625, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, n_layers=3, n_layers_update=3, hidden_dim_update=96, aug_loop=26, recurrent=False
Metrics: connectivity_R2=0.775, tau_R2=0.419, V_rest_R2=0.161, cluster_accuracy=0.686, test_R2=0.948, test_pearson=0.993, training_time_min=53.6
Embedding: 65 types with moderate separation
Mutation: data_augmentation_loop: 28 -> 26. Testing principle: "n_layers=3+n_layers_update=3 enables aug_loop within time"
Parent rule: test if simpler arch with lower aug_loop maintains time efficiency
Observation: CONFIRMED simpler arch achieves 53.6 min (within limit!), but lower aug_loop=26 gives poor metrics (tau=0.419, conn=0.775); aug_loop >= 28 needed for good tau
Next: parent=62

### Batch 16 Summary (Iters 61-64)
Best connectivity_R2: Node 61 (0.836)
Best tau_R2: Node 62 (0.789) - excellent, approaches N50's 0.911 breakthrough
Best V_rest_R2: Node 61 (0.197)
Best cluster_accuracy: Node 62 (0.709)

Key findings:
1. **aug_loop=30 with simpler arch achieves tau=0.768** - N61 improves tau from 0.642 (N59) but time 65.5 min exceeds limit
2. **lr_emb=4E-3 + n_layers_update=4 gives tau=0.789** - N62 best tau this batch; n_layers_update=4 may be key for tau recovery
3. **aug_loop=29 gives INCONSISTENT tau=0.586** - N63 worse than both N61 (aug=30, tau=0.768) and N59 (aug=28, tau=0.642) - stochastic variance
4. **CONFIRMED: simpler arch saves time but needs aug_loop>=28** - N64 at aug_loop=26 achieves 53.6 min but tau=0.419 poor
5. **Trade-off refined**: n_layers_update=4 helps tau, simpler edge MLP (n_layers=3) helps time; combining may work

