# FlyVis Experiment Log: fly_N9_62_1 (parallel)

## Block 1: Learning Rates

### Initial Batch (Iter 0-3)
Starting parallel exploration with 4 diverse learning rate configurations:

| Slot | lr_W | lr | lr_emb | Strategy |
| ---- | ---- | -- | ------ | -------- |
| 0 | 1E-3 | 5E-4 | 1E-3 | baseline (unchanged) |
| 1 | 2E-3 | 5E-4 | 1E-3 | higher lr_W |
| 2 | 5E-4 | 1E-3 | 1E-3 | lower lr_W, higher lr |
| 3 | 5E-3 | 1E-3 | 2E-3 | boundary probe, all high |

Rationale:
- slot 0: baseline from prior experiments
- slot 1: test if faster W learning improves connectivity R2
- slot 2: prioritize MLP learning over W — may help tau/V_rest recovery
- slot 3: aggressive boundary probe to find upper limits

## Iter 1: converged
Node: id=1, parent=root
Mode/Strategy: baseline
Config: lr_W=1E-3, lr=5E-4, lr_emb=1E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.949, tau_R2=0.989, V_rest_R2=0.629, cluster_accuracy=0.859, test_R2=-1.15, test_pearson=0.996, training_time_min=54.8
Embedding: 65 types well-separated with high cluster accuracy
Mutation: baseline config (no change from default)
Parent rule: root — first iteration
Observation: baseline achieves excellent conn_R2 and best V_rest_R2; moderate MLP lr may limit tau/cluster
Next: parent=1

## Iter 2: partial
Node: id=2, parent=root
Mode/Strategy: explore
Config: lr_W=2E-3, lr=5E-4, lr_emb=1E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.588, tau_R2=0.973, V_rest_R2=0.222, cluster_accuracy=0.871, test_R2=-0.50, test_pearson=0.996, training_time_min=49.1
Embedding: types separated but lower cluster accuracy than baseline
Mutation: lr_W: 1E-3 -> 2E-3
Parent rule: root — testing higher lr_W
Observation: lr_W=2E-3 causes severe connectivity degradation; W overshoots optimal values
Next: parent=1

## Iter 3: converged
Node: id=3, parent=root
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=1E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.921, tau_R2=0.993, V_rest_R2=0.518, cluster_accuracy=0.903, test_R2=0.96, test_pearson=0.999, training_time_min=52.3
Embedding: excellent type separation with best cluster accuracy (0.903)
Mutation: lr_W: 1E-3 -> 5E-4, lr: 5E-4 -> 1E-3
Parent rule: root — testing lower lr_W with higher MLP lr
Observation: slower W learning + faster MLP yields best tau_R2 and cluster_acc but lower V_rest_R2; MLP benefits from higher lr
Next: parent=3

## Iter 4: converged
Node: id=4, parent=root
Mode/Strategy: boundary-probe
Config: lr_W=5E-3, lr=1E-3, lr_emb=2E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.939, tau_R2=0.991, V_rest_R2=0.430, cluster_accuracy=0.880, test_R2=0.83, test_pearson=0.971, training_time_min=50.1
Embedding: good type separation
Mutation: lr_W: 1E-3 -> 5E-3, lr: 5E-4 -> 1E-3, lr_emb: 1E-3 -> 2E-3
Parent rule: root — aggressive boundary probe
Observation: surprisingly lr_W=5E-3 works when lr_emb=2E-3; higher embedding lr may stabilize aggressive W learning
Next: parent=1

### Batch 2 Plan (Iter 5-8)
UCB ranking: Node 1 (2.363) > Node 4 (2.353) > Node 3 (2.335) > Node 2 (2.002)

| Slot | Role | Parent | lr_W | lr | lr_emb | Mutation |
| ---- | ---- | ------ | ---- | -- | ------ | -------- |
| 0 | exploit | Node 1 | 1E-3 | 1E-3 | 1E-3 | lr: 5E-4 -> 1E-3 (add MLP benefit) |
| 1 | exploit | Node 4 | 3E-3 | 1E-3 | 2E-3 | lr_W: 5E-3 -> 3E-3 (reduce aggressive W) |
| 2 | explore | Node 3 | 5E-4 | 1E-3 | 2E-3 | lr_emb: 1E-3 -> 2E-3 (test higher emb lr) |
| 3 | boundary-probe | Node 1 | 7E-4 | 5E-4 | 1E-3 | lr_W: 1E-3 -> 7E-4 (test lower lr_W) |

Hypothesis:
- Slot 0: combining Node 1's lr_W with Node 3's lr may achieve best of both
- Slot 1: Node 4's config minus aggressive lr_W might improve connectivity
- Slot 2: higher lr_emb may boost Node 3's cluster_acc further
- Slot 3: even slower lr_W may improve V_rest_R2 without hurting connectivity

## Iter 5: converged
Node: id=5, parent=1
Mode/Strategy: exploit
Config: lr_W=1E-3, lr=1E-3, lr_emb=1E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.975, tau_R2=0.989, V_rest_R2=0.347, cluster_accuracy=0.902, test_R2=-0.38, test_pearson=0.996, training_time_min=49.9
Embedding: 65 types well-separated with excellent cluster accuracy
Mutation: lr: 5E-4 -> 1E-3
Parent rule: Node 1 — combining baseline lr_W with higher MLP lr
Observation: **NEW BEST conn_R2=0.975** — lr=1E-3 boosts connectivity AND cluster_acc; but V_rest_R2 drops significantly
Next: parent=5

## Iter 6: partial
Node: id=6, parent=4
Mode/Strategy: exploit
Config: lr_W=3E-3, lr=1E-3, lr_emb=2E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.790, tau_R2=0.995, V_rest_R2=0.594, cluster_accuracy=0.847, test_R2=0.82, test_pearson=0.998, training_time_min=78.7
Embedding: moderate type separation
Mutation: lr_W: 5E-3 -> 3E-3
Parent rule: Node 4 — reduce aggressive lr_W while keeping higher lr_emb
Observation: ⚠️ TIME=78.7min exceeds limit; lr_W=3E-3 still too high — degrades connectivity despite lr_emb=2E-3
Next: parent=5

## Iter 7: converged
Node: id=7, parent=3
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=2E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.968, tau_R2=0.989, V_rest_R2=0.507, cluster_accuracy=0.845, test_R2=-3.70, test_pearson=0.997, training_time_min=74.6
Embedding: good type separation
Mutation: lr_emb: 1E-3 -> 2E-3
Parent rule: Node 3 — test higher embedding lr
Observation: ⚠️ TIME=74.6min exceeds limit; lr_emb=2E-3 increases training time; conn_R2=0.968 good but cluster_acc dropped vs Node 3
Next: parent=5

## Iter 8: converged
Node: id=8, parent=1
Mode/Strategy: boundary-probe
Config: lr_W=7E-4, lr=5E-4, lr_emb=1E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.936, tau_R2=0.994, V_rest_R2=0.585, cluster_accuracy=0.866, test_R2=0.97, test_pearson=0.999, training_time_min=49.7
Embedding: good type separation with balanced metrics
Mutation: lr_W: 1E-3 -> 7E-4
Parent rule: Node 1 — test slightly lower lr_W
Observation: lr_W=7E-4 gives good conn_R2 with best V_rest_R2=0.585; lr_W between 7E-4 and 1E-3 may be optimal for V_rest
Next: parent=5

### Batch 3 Plan (Iter 9-12)
UCB ranking: Node 5 (2.975) > Node 7 (2.968) > Node 8 (2.936) > Node 6 (2.790)

Key insights from batch 2:
- Node 5 (lr_W=1E-3, lr=1E-3) achieves best conn_R2=0.975 — exploit this
- lr_emb=2E-3 causes training time to exceed 60min — avoid
- lr_W=7E-4 (Node 8) achieves best V_rest_R2=0.585 — explore trade-off
- lr=1E-3 (doubled MLP lr) consistently helps connectivity

| Slot | Role | Parent | lr_W | lr | lr_emb | Mutation |
| ---- | ---- | ------ | ---- | -- | ------ | -------- |
| 0 | exploit | Node 5 | 1E-3 | 1E-3 | 1.5E-3 | lr_emb: 1E-3 -> 1.5E-3 (conservative emb boost) |
| 1 | exploit | Node 5 | 8E-4 | 1E-3 | 1E-3 | lr_W: 1E-3 -> 8E-4 (slight W reduction for V_rest) |
| 2 | explore | Node 8 | 7E-4 | 1E-3 | 1E-3 | lr: 5E-4 -> 1E-3 (add MLP benefit to Node 8) |
| 3 | boundary-probe | Node 5 | 1.2E-3 | 1E-3 | 1E-3 | lr_W: 1E-3 -> 1.2E-3 (test upper lr_W boundary) |

Hypothesis:
- Slot 0: small lr_emb increase may improve cluster_acc without excessive time
- Slot 1: lr_W=8E-4 may improve V_rest_R2 while preserving high conn_R2
- Slot 2: Node 8's lr_W=7E-4 + lr=1E-3 may achieve best balance
- Slot 3: test if lr_W can go slightly above 1E-3 without degradation

## Iter 9: converged
Node: id=9, parent=5
Mode/Strategy: exploit
Config: lr_W=1E-3, lr=1E-3, lr_emb=1.5E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.958, tau_R2=0.993, V_rest_R2=0.504, cluster_accuracy=0.890, test_R2=0.028, test_pearson=0.986, training_time_min=48.6
Embedding: 65 types well-separated
Mutation: lr_emb: 1E-3 -> 1.5E-3
Parent rule: Node 5 — test conservative embedding lr increase
Observation: lr_emb=1.5E-3 improves V_rest_R2 to 0.504 (vs 0.347 in Node 5) with slight conn_R2 drop (0.958 vs 0.975); good balance
Next: parent=9

## Iter 10: converged
Node: id=10, parent=5
Mode/Strategy: exploit
Config: lr_W=8E-4, lr=1E-3, lr_emb=1E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.820, tau_R2=0.950, V_rest_R2=0.389, cluster_accuracy=0.844, test_R2=-0.396, test_pearson=0.989, training_time_min=48.4
Embedding: moderate type separation
Mutation: lr_W: 1E-3 -> 8E-4
Parent rule: Node 5 — test slight lr_W reduction for V_rest improvement
Observation: lr_W=8E-4 unexpectedly degrades ALL metrics vs Node 5; stochastic variance or lr_W=1E-3 is truly optimal
Next: parent=9

## Iter 11: converged
Node: id=11, parent=8
Mode/Strategy: explore
Config: lr_W=7E-4, lr=1E-3, lr_emb=1E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.875, tau_R2=0.992, V_rest_R2=0.526, cluster_accuracy=0.831, test_R2=-3.11, test_pearson=0.979, training_time_min=48.4
Embedding: moderate type separation
Mutation: lr: 5E-4 -> 1E-3
Parent rule: Node 8 — combine lr_W=7E-4 with higher MLP lr
Observation: lr=1E-3 didn't boost Node 8 as expected; conn_R2=0.875 < Node 8's 0.936; lr_W below 1E-3 may need lower MLP lr
Next: parent=9

## Iter 12: converged
Node: id=12, parent=5
Mode/Strategy: boundary-probe
Config: lr_W=1.2E-3, lr=1E-3, lr_emb=1E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.939, tau_R2=0.984, V_rest_R2=0.449, cluster_accuracy=0.881, test_R2=0.15, test_pearson=0.996, training_time_min=48.9
Embedding: good type separation
Mutation: lr_W: 1E-3 -> 1.2E-3
Parent rule: Node 5 — test slightly higher lr_W boundary
Observation: lr_W=1.2E-3 slightly degrades conn_R2 (0.939 vs 0.975); confirms lr_W=1E-3 is upper optimal bound
Next: parent=9

### Batch 4 Plan (Iter 13-16)
UCB ranking: Node 7 (3.417) > Node 9 (3.408) > Node 4 (3.388) > Node 12 (3.388) > Node 8 (3.385)

Key insights from batch 3:
- Node 9 (lr_emb=1.5E-3) achieves good balance: conn_R2=0.958, V_rest_R2=0.504
- lr_W=1E-3 remains optimal — both 8E-4 and 1.2E-3 degrade connectivity
- lr_W < 1E-3 with lr=1E-3 creates imbalance (Nodes 10, 11)
- lr_emb=1.5E-3 is safe (no time increase) and improves V_rest_R2

Hypotheses for next batch:
- Explore lr_emb in range [1.2E-3, 1.8E-3] to optimize V_rest vs conn trade-off
- Test if lr=1.5E-3 (higher MLP lr) can improve with lr_W=1E-3
- Begin transitioning toward block 2 (regularization) insights

| Slot | Role | Parent | lr_W | lr | lr_emb | Mutation |
| ---- | ---- | ------ | ---- | -- | ------ | -------- |
| 0 | exploit | Node 9 | 1E-3 | 1E-3 | 1.2E-3 | lr_emb: 1.5E-3 -> 1.2E-3 (fine-tune emb lr) |
| 1 | exploit | Node 9 | 1E-3 | 1.5E-3 | 1.5E-3 | lr: 1E-3 -> 1.5E-3 (test higher MLP lr) |
| 2 | explore | Node 7 | 5E-4 | 1E-3 | 1.5E-3 | lr_emb: 2E-3 -> 1.5E-3 (reduce time while keeping benefits) |
| 3 | principle-test | Node 5 | 1E-3 | 1E-3 | 5E-4 | lr_emb: 1E-3 -> 5E-4. Testing principle: "lr_emb >= 1E-3 is required for good cluster_acc" |

## Iter 13: converged
Node: id=13, parent=9
Mode/Strategy: exploit
Config: lr_W=1E-3, lr=1E-3, lr_emb=1.2E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.967, tau_R2=0.986, V_rest_R2=0.659, cluster_accuracy=0.868, test_R2=-0.68, test_pearson=0.985, training_time_min=48.8
Embedding: 65 types well-separated
Mutation: lr_emb: 1.5E-3 -> 1.2E-3
Parent rule: Node 9 — fine-tune embedding learning rate
Observation: lr_emb=1.2E-3 improves V_rest_R2 to 0.659 (vs 0.504 in Node 9) with slight conn_R2 drop; good trade-off
Next: parent=15

## Iter 14: converged
Node: id=14, parent=9
Mode/Strategy: exploit
Config: lr_W=1E-3, lr=1.5E-3, lr_emb=1.5E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.960, tau_R2=0.994, V_rest_R2=0.505, cluster_accuracy=0.875, test_R2=-4.87, test_pearson=0.991, training_time_min=49.1
Embedding: 65 types well-separated with excellent tau_R2
Mutation: lr: 1E-3 -> 1.5E-3
Parent rule: Node 9 — test higher MLP learning rate
Observation: lr=1.5E-3 yields best tau_R2=0.994 but V_rest_R2 unchanged vs Node 9; higher MLP lr doesn't improve V_rest
Next: parent=15

## Iter 15: converged
Node: id=15, parent=7
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1E-3, lr_emb=1.5E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.976, tau_R2=0.990, V_rest_R2=0.767, cluster_accuracy=0.873, test_R2=-1.81, test_pearson=0.990, training_time_min=48.7
Embedding: 65 types well-separated with excellent V_rest recovery
Mutation: lr_emb: 2E-3 -> 1.5E-3
Parent rule: Node 7 — reduce embedding lr to avoid time overflow while keeping benefits
Observation: **NEW BEST** V_rest_R2=0.767 AND conn_R2=0.976! lr_W=5E-4 + lr_emb=1.5E-3 achieves optimal balance; challenges principle 1
Next: parent=15

## Iter 16: converged
Node: id=16, parent=5
Mode/Strategy: principle-test
Config: lr_W=1E-3, lr=1E-3, lr_emb=5E-4, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.953, tau_R2=0.991, V_rest_R2=0.401, cluster_accuracy=0.897, test_R2=0.90, test_pearson=0.998, training_time_min=48.8
Embedding: 65 types well-separated with best cluster_accuracy
Mutation: lr_emb: 1E-3 -> 5E-4. Testing principle: "lr_emb >= 1E-3 is required for good cluster_acc"
Parent rule: Node 5 — test if low lr_emb degrades cluster accuracy
Observation: **CONTRADICTS principle 4** — lr_emb=5E-4 achieves BEST cluster_acc=0.897! But V_rest_R2 is low (0.401)
Next: parent=15

### Batch 5 Plan (Iter 17-20)
UCB ranking: Node 15 (3.804) > Node 7 (3.796) > Node 13 (3.795) > Node 14 (3.788) > Node 16 (3.781)

Key insights from batch 4:
- **Node 15** (lr_W=5E-4, lr=1E-3, lr_emb=1.5E-3) is NEW BEST: conn_R2=0.976, V_rest_R2=0.767
- This challenges principle 1: lr_W=5E-4 outperforms lr_W=1E-3 when lr_emb=1.5E-3!
- Node 16 contradicts principle 4: lr_emb=5E-4 gives best cluster_acc=0.897
- lr_emb=1.2E-3 (Node 13) gives good V_rest_R2=0.659

Updated understanding:
- lr_W=5E-4 + lr_emb=1.5E-3 is optimal for V_rest_R2 (Node 15)
- lr_W=1E-3 + lr_emb=5E-4 is optimal for cluster_acc (Node 16)
- Trade-off exists between V_rest_R2 and cluster_acc

| Slot | Role | Parent | lr_W | lr | lr_emb | Mutation |
| ---- | ---- | ------ | ---- | -- | ------ | -------- |
| 0 | exploit | Node 15 | 5E-4 | 1E-3 | 1.2E-3 | lr_emb: 1.5E-3 -> 1.2E-3 (fine-tune around best) |
| 1 | exploit | Node 15 | 5E-4 | 1.2E-3 | 1.5E-3 | lr: 1E-3 -> 1.2E-3 (test higher MLP lr with best config) |
| 2 | explore | Node 13 | 1E-3 | 1E-3 | 1.5E-3 | lr_emb: 1.2E-3 -> 1.5E-3 (combine Node 13's lr_W with higher lr_emb) |
| 3 | principle-test | Node 15 | 5E-4 | 1E-3 | 1E-3 | lr_emb: 1.5E-3 -> 1E-3. Testing principle: "lr=1E-3 improves connectivity over lr=5E-4" |

## Iter 17: partial
Node: id=17, parent=15
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1E-3, lr_emb=1.2E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.554, tau_R2=0.989, V_rest_R2=0.545, cluster_accuracy=0.877, test_R2=0.30, test_pearson=0.999, training_time_min=48.5
Embedding: 65 types moderately separated
Mutation: lr_emb: 1.5E-3 -> 1.2E-3
Parent rule: Node 15 — fine-tune embedding lr around best config
Observation: **SEVERE DEGRADATION** conn_R2 drops from 0.976 to 0.554; lr_emb=1.2E-3 with lr_W=5E-4 creates instability; stochastic variance or sensitive parameter interaction
Next: parent=18

## Iter 18: converged ⭐ NEW BEST conn_R2
Node: id=18, parent=15
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.978, tau_R2=0.988, V_rest_R2=0.625, cluster_accuracy=0.863, test_R2=0.43, test_pearson=0.973, training_time_min=49.2
Embedding: 65 types well-separated
Mutation: lr: 1E-3 -> 1.2E-3
Parent rule: Node 15 — test higher MLP lr with best config
Observation: **NEW BEST conn_R2=0.978** — lr=1.2E-3 improves connectivity over Node 15 (0.976); V_rest_R2=0.625 lower than Node 15's 0.767
Next: parent=18

## Iter 19: converged ⭐ NEW BEST V_rest_R2
Node: id=19, parent=13
Mode/Strategy: explore
Config: lr_W=1E-3, lr=1E-3, lr_emb=1.5E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.869, tau_R2=0.988, V_rest_R2=0.772, cluster_accuracy=0.900, test_R2=-0.20, test_pearson=0.998, training_time_min=48.8
Embedding: 65 types well-separated with excellent cluster accuracy
Mutation: lr_emb: 1.2E-3 -> 1.5E-3
Parent rule: Node 13 — combine lr_W=1E-3 with higher lr_emb
Observation: **NEW BEST V_rest_R2=0.772** and excellent cluster_acc=0.900; conn_R2=0.869 is partial but close to converged threshold

## Iter 20: partial
Node: id=20, parent=15
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=1E-3, lr_emb=1E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.677, tau_R2=0.997, V_rest_R2=0.770, cluster_accuracy=0.870, test_R2=-14.34, test_pearson=0.986, training_time_min=48.6
Embedding: 65 types moderately separated
Mutation: lr_emb: 1.5E-3 -> 1E-3. Testing principle: "lr=1E-3 improves connectivity over lr=5E-4"
Parent rule: Node 15 — test if lower lr_emb maintains high V_rest_R2 with Node 15's lr_W
Observation: lr_emb=1E-3 maintains excellent V_rest_R2=0.770 but conn_R2 degrades significantly; lr_W=5E-4 needs lr_emb=1.5E-3 for connectivity

### Batch 6 Plan (Iter 21-24)
UCB ranking: Node 18 (4.140) > Node 13 (4.129) > Node 14 (4.122) > Node 16 (4.115)

Key insights from batch 5:
- **Node 18** (lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3) is NEW BEST conn_R2=0.978
- **Node 19** (lr_W=1E-3, lr=1E-3, lr_emb=1.5E-3) achieves NEW BEST V_rest_R2=0.772 + cluster_acc=0.900
- lr_W=5E-4 with lr_emb <1.5E-3 causes severe conn_R2 degradation (Nodes 17, 20)
- lr_W=5E-4 requires lr_emb=1.5E-3 to achieve high connectivity

Two competitive configs:
1. Node 18: conn_R2=0.978, V_rest_R2=0.625 (best connectivity)
2. Node 19: conn_R2=0.869, V_rest_R2=0.772, cluster_acc=0.900 (best V_rest + cluster)

| Slot | Role | Parent | lr_W | lr | lr_emb | Mutation |
| ---- | ---- | ------ | ---- | -- | ------ | -------- |
| 0 | exploit | Node 18 | 5E-4 | 1.2E-3 | 1.3E-3 | lr_emb: 1.5E-3 -> 1.3E-3 (fine-tune around best conn_R2) |
| 1 | exploit | Node 19 | 1E-3 | 1.2E-3 | 1.5E-3 | lr: 1E-3 -> 1.2E-3 (boost MLP lr for Node 19) |
| 2 | explore | Node 18 | 7E-4 | 1.2E-3 | 1.5E-3 | lr_W: 5E-4 -> 7E-4 (intermediate lr_W) |
| 3 | principle-test | Node 19 | 1E-3 | 1E-3 | 1.8E-3 | lr_emb: 1.5E-3 -> 1.8E-3. Testing principle: "lr_emb=2E-3 causes excessive training time" |

## Iter 21: converged
Node: id=21, parent=18
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.3E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.868, tau_R2=0.994, V_rest_R2=0.727, cluster_accuracy=0.886, test_R2=-0.55, test_pearson=0.997, training_time_min=49.4
Embedding: 65 types well-separated
Mutation: lr_emb: 1.5E-3 -> 1.3E-3
Parent rule: Node 18 — fine-tune embedding lr around best conn_R2 config
Observation: lr_emb=1.3E-3 maintains converged conn_R2=0.868 (better than 1.2E-3 which failed at Node 17), excellent V_rest_R2=0.727

## Iter 22: partial
Node: id=22, parent=19
Mode/Strategy: exploit
Config: lr_W=1E-3, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.601, tau_R2=0.995, V_rest_R2=0.793, cluster_accuracy=0.847, test_R2=-4.90, test_pearson=0.995, training_time_min=48.8
Embedding: 65 types moderately separated
Mutation: lr: 1E-3 -> 1.2E-3
Parent rule: Node 19 — test higher MLP lr with best V_rest config
Observation: lr=1.2E-3 with lr_W=1E-3 severely degrades conn_R2; confirms lr_W=1E-3 requires lr=1E-3 (not higher)

## Iter 23: converged ⭐ NEW BEST V_rest_R2
Node: id=23, parent=18
Mode/Strategy: explore
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.823, tau_R2=0.997, V_rest_R2=0.817, cluster_accuracy=0.884, test_R2=0.41, test_pearson=0.999, training_time_min=49.1
Embedding: 65 types well-separated with excellent tau recovery
Mutation: lr_W: 5E-4 -> 7E-4
Parent rule: Node 18 — test intermediate lr_W between 5E-4 and 1E-3
Observation: **NEW BEST V_rest_R2=0.817** and best tau_R2=0.997! lr_W=7E-4 achieves optimal V_rest recovery; conn_R2=0.823 is converged

## Iter 24: converged
Node: id=24, parent=19
Mode/Strategy: principle-test
Config: lr_W=1E-3, lr=1E-3, lr_emb=1.8E-3, coeff_edge_diff=500, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.914, tau_R2=0.956, V_rest_R2=0.007, cluster_accuracy=0.868, test_R2=0.57, test_pearson=0.987, training_time_min=48.1
Embedding: 65 types moderately separated
Mutation: lr_emb: 1.5E-3 -> 1.8E-3. Testing principle: "lr_emb=2E-3 causes excessive training time"
Parent rule: Node 19 — test lr_emb between 1.5E-3 and 2E-3
Observation: **CONTRADICTS time principle** — training time=48.1min is normal! But V_rest_R2=0.007 is catastrophic; lr_emb=1.8E-3 destroys V_rest recovery

---

## Block 1 Summary: Learning Rates

### Best Configurations Found
| Rank | Node | lr_W | lr | lr_emb | conn_R2 | tau_R2 | V_rest_R2 | cluster_acc | Strength |
|------|------|------|--------|--------|---------|--------|-----------|-------------|----------|
| 1 | 18 | 5E-4 | 1.2E-3 | 1.5E-3 | **0.978** | 0.988 | 0.625 | 0.863 | Best connectivity |
| 2 | 23 | 7E-4 | 1.2E-3 | 1.5E-3 | 0.823 | **0.997** | **0.817** | 0.884 | Best V_rest + tau |
| 3 | 15 | 5E-4 | 1E-3 | 1.5E-3 | 0.976 | 0.990 | 0.767 | 0.873 | Balanced |
| 4 | 19 | 1E-3 | 1E-3 | 1.5E-3 | 0.869 | 0.988 | 0.772 | **0.900** | Best cluster_acc |

### Key Findings
1. **lr_W=5E-4 to 7E-4** with **lr=1.2E-3** and **lr_emb=1.5E-3** achieves best overall performance
2. **lr_W=1E-3** requires **lr=1E-3** (not higher) — lr=1.2E-3 causes severe degradation
3. **lr_emb=1.5E-3** is optimal — lower values (1.2E-3, 1E-3) cause connectivity degradation with low lr_W
4. **lr_emb=1.8E-3+** destroys V_rest recovery despite conn_R2 remaining okay
5. Training time is NOT primarily driven by lr_emb (principle 3 was wrong)

### Carry-Forward Config for Block 2
Starting from Node 23 (best V_rest) and Node 18 (best conn) for regularization exploration:
- Node 23: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3 → V_rest_R2=0.817, conn_R2=0.823
- Node 18: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3 → conn_R2=0.978, V_rest_R2=0.625

---

## Block 2: Regularization

### Initial Batch Plan (Iter 25-28)
| Slot | Role | Parent | Focus | Mutation |
| ---- | ---- | ------ | ----- | -------- |
| 0 | exploit | Node 18 | coeff_edge_diff | coeff_edge_diff: 500 -> 1000 |
| 1 | exploit | Node 23 | coeff_W_L1 | coeff_W_L1: 5E-5 -> 1E-4 |
| 2 | explore | Node 18 | coeff_edge_norm | coeff_edge_norm: 1 -> 10 |
| 3 | principle-test | Node 23 | coeff_phi_weight_L2 | coeff_phi_weight_L2: 0.001 -> 0.01 |

## Iter 25: converged
Node: id=25, parent=18
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=1000, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.940, tau_R2=0.987, V_rest_R2=0.413, cluster_accuracy=0.886, test_R2=-1.22, test_pearson=0.991, training_time_min=48.7
Embedding: 65 types well-separated
Mutation: coeff_edge_diff: 500 -> 1000
Parent rule: Node 18 — test doubled edge-diff regularization on best conn_R2 config
Observation: coeff_edge_diff=1000 degrades conn_R2 from 0.978 to 0.940 and V_rest_R2 from 0.625 to 0.413; stronger same-type constraint hurts
Next: parent=25

## Iter 26: converged
Node: id=26, parent=23
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=500, coeff_W_L1=1E-4, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.899, tau_R2=0.989, V_rest_R2=0.432, cluster_accuracy=0.900, test_R2=-4.53, test_pearson=0.983, training_time_min=49.8
Embedding: 65 types well-separated with best cluster accuracy
Mutation: coeff_W_L1: 5E-5 -> 1E-4
Parent rule: Node 23 — test doubled W sparsity on best V_rest config
Observation: coeff_W_L1=1E-4 maintains conn_R2=0.899 (close to Node 23's 0.823) but V_rest_R2 drops from 0.817 to 0.432; excessive W sparsity hurts V_rest
Next: parent=25

## Iter 27: partial
Node: id=27, parent=18
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=500, coeff_edge_norm=10, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.886, tau_R2=0.473, V_rest_R2=0.095, cluster_accuracy=0.907, test_R2=0.37, test_pearson=0.998, training_time_min=48.6
Embedding: 65 types well-separated with best cluster accuracy in batch
Mutation: coeff_edge_norm: 1 -> 10
Parent rule: Node 18 — test 10x monotonicity penalty on best conn_R2 config
Observation: **coeff_edge_norm=10 catastrophically degrades tau_R2 (0.473) and V_rest_R2 (0.095)**; strong monotonicity hurts parameter recovery while slightly improving cluster_acc
Next: parent=25

## Iter 28: converged
Node: id=28, parent=23
Mode/Strategy: principle-test
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=500, coeff_phi_weight_L2=0.01, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.926, tau_R2=0.978, V_rest_R2=0.264, cluster_accuracy=0.905, test_R2=-0.88, test_pearson=0.996, training_time_min=48.5
Embedding: 65 types well-separated
Mutation: coeff_phi_weight_L2: 0.001 -> 0.01. Testing principle: "higher L2 on MLP stabilizes training"
Parent rule: Node 23 — test 10x L2 regularization on MLP parameters
Observation: coeff_phi_weight_L2=0.01 improves conn_R2 vs Node 23 (0.926 vs 0.823) but V_rest_R2 drops from 0.817 to 0.264; L2 stabilizes W learning at cost of V_rest
Next: parent=25

### Batch 2 Plan (Iter 29-32)
UCB ranking: Node 25 (2.353) > Node 28 (2.340) > Node 26 (2.313) > Node 27 (2.300)

Key insights from Iter 25-28:
- **coeff_edge_diff=1000** (Node 25) degrades connectivity slightly but maintains converged status
- **coeff_edge_norm=10** (Node 27) is catastrophic for tau and V_rest — avoid high monotonicity penalty
- **coeff_W_L1=1E-4** (Node 26) maintains connectivity but hurts V_rest
- **coeff_phi_weight_L2=0.01** (Node 28) helps connectivity at cost of V_rest

None beat Block 1 baselines (Node 18 conn_R2=0.978, Node 23 V_rest_R2=0.817).
Strategy: test smaller changes and different regularization parameters

| Slot | Role | Parent | Focus | Mutation |
| ---- | ---- | ------ | ----- | -------- |
| 0 | exploit | Node 25 | coeff_edge_diff | coeff_edge_diff: 1000 -> 750 (intermediate value) |
| 1 | exploit | Node 25 | coeff_phi_weight_L1 | coeff_phi_weight_L1: 1 -> 0.5 (reduce MLP L1) |
| 2 | explore | Node 26 | coeff_edge_weight_L1 | coeff_edge_weight_L1: 1 -> 0.5 (reduce edge L1) |
| 3 | principle-test | Node 25 | coeff_W_L1 | coeff_W_L1: 5E-5 -> 2E-5. Testing principle: "lower W L1 improves V_rest recovery"

## Iter 29: converged
Node: id=29, parent=25
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.953, tau_R2=0.990, V_rest_R2=0.564, cluster_accuracy=0.869, test_R2=-4.38, test_pearson=0.990, training_time_min=49.0
Embedding: 65 types well-separated
Mutation: coeff_edge_diff: 1000 -> 750
Parent rule: Node 25 — test intermediate edge-diff value between 500 and 1000
Observation: coeff_edge_diff=750 improves conn_R2 to 0.953 (vs 0.940 at 1000) and V_rest_R2 to 0.564 (vs 0.413 at 1000); lower edge-diff is better
Next: parent=31

## Iter 30: converged
Node: id=30, parent=25
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=1000, coeff_phi_weight_L1=0.5, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.915, tau_R2=0.997, V_rest_R2=0.760, cluster_accuracy=0.884, test_R2=-1.49, test_pearson=0.991, training_time_min=49.0
Embedding: 65 types well-separated with excellent tau recovery
Mutation: coeff_phi_weight_L1: 1 -> 0.5
Parent rule: Node 25 — test reduced MLP L1 regularization
Observation: **coeff_phi_weight_L1=0.5 achieves best V_rest_R2=0.760** in this batch with excellent tau_R2=0.997; reduced MLP sparsity helps parameter recovery
Next: parent=31

## Iter 31: converged ⭐ BEST batch conn_R2
Node: id=31, parent=26
Mode/Strategy: explore
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=500, coeff_edge_weight_L1=0.5, coeff_W_L1=1E-4, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.960, tau_R2=0.997, V_rest_R2=0.712, cluster_accuracy=0.883, test_R2=-1.39, test_pearson=0.995, training_time_min=49.7
Embedding: 65 types well-separated with excellent tau recovery
Mutation: coeff_edge_weight_L1: 1 -> 0.5
Parent rule: Node 26 — test reduced edge weight L1 on lr_W=7E-4 config
Observation: **coeff_edge_weight_L1=0.5** improves conn_R2 to 0.960 (vs 0.899 in Node 26) with good V_rest_R2=0.712; reduced edge L1 helps connectivity
Next: parent=31

## Iter 32: converged
Node: id=32, parent=25
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=1000, coeff_W_L1=2E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.906, tau_R2=0.993, V_rest_R2=0.672, cluster_accuracy=0.894, test_R2=-1.15, test_pearson=0.989, training_time_min=49.9
Embedding: 65 types well-separated
Mutation: coeff_W_L1: 5E-5 -> 2E-5. Testing principle: "lower W L1 improves V_rest recovery"
Parent rule: Node 25 — test reduced W sparsity for V_rest improvement
Observation: **PARTIAL SUPPORT** — coeff_W_L1=2E-5 improves V_rest_R2 from 0.413 to 0.672 but conn_R2 drops from 0.940 to 0.906; lower W L1 helps V_rest but hurts connectivity
Next: parent=31

### Batch 3 Plan (Iter 33-36)
UCB ranking: Node 31 (2.959) > Node 29 (2.953) > Node 28 (2.926) > Node 30 (2.914) > Node 32 (2.905)

Key insights from Iter 29-32:
- **Node 31** (coeff_edge_weight_L1=0.5, lr_W=7E-4) achieved best conn_R2=0.960 with good V_rest_R2=0.712
- **Node 30** (coeff_phi_weight_L1=0.5) achieved best V_rest_R2=0.760 with tau_R2=0.997
- **Node 29** (coeff_edge_diff=750) shows intermediate edge-diff is better than 1000
- **Node 32** confirms lower W L1 helps V_rest but hurts connectivity

Strategy: exploit Node 31's success with coeff_edge_weight_L1=0.5 and explore combinations

| Slot | Role | Parent | Focus | Mutation |
| ---- | ---- | ------ | ----- | -------- |
| 0 | exploit | Node 31 | coeff_phi_weight_L1 | coeff_phi_weight_L1: 1 -> 0.5 (combine with edge L1 reduction) |
| 1 | exploit | Node 30 | coeff_edge_weight_L1 | coeff_edge_weight_L1: 1 -> 0.5 (combine with phi L1 reduction) |
| 2 | explore | Node 29 | coeff_edge_diff | coeff_edge_diff: 750 -> 500 (return to baseline) |
| 3 | principle-test | Node 31 | coeff_W_L1 | coeff_W_L1: 1E-4 -> 5E-5. Testing principle: "coeff_W_L1=5E-5 is optimal baseline"

## Iter 33: partial
Node: id=33, parent=31
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=500, coeff_edge_weight_L1=0.5, coeff_phi_weight_L1=0.5, coeff_W_L1=1E-4, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.697, tau_R2=0.995, V_rest_R2=0.497, cluster_accuracy=0.899, test_R2=0.91, test_pearson=0.998, training_time_min=50.5
Embedding: 65 types well-separated
Mutation: coeff_phi_weight_L1: 1 -> 0.5
Parent rule: Node 31 — combine phi L1 reduction with existing edge L1 reduction
Observation: **CONN_R2 COLLAPSE** — combined phi_L1=0.5 + edge_L1=0.5 + W_L1=1E-4 at lr_W=7E-4 causes conn_R2 drop from 0.960 to 0.697; over-regularization hurts
Next: parent=34

## Iter 34: converged ⭐ BEST batch all metrics
Node: id=34, parent=30
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=1000, coeff_edge_weight_L1=0.5, coeff_phi_weight_L1=0.5, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.973, tau_R2=0.997, V_rest_R2=0.709, cluster_accuracy=0.910, test_R2=0.67, test_pearson=0.999, training_time_min=50.8
Embedding: 65 types well-separated with excellent clustering
Mutation: coeff_edge_weight_L1: 1 -> 0.5
Parent rule: Node 30 — combine edge L1 reduction with existing phi L1 reduction
Observation: **EXCELLENT** — combined phi_L1=0.5 + edge_L1=0.5 at lr_W=5E-4 with coeff_edge_diff=1000 achieves conn_R2=0.973, V_rest_R2=0.709, best cluster_acc=0.910; lr_W=5E-4 is key vs Node 33
Next: parent=34

## Iter 35: converged
Node: id=35, parent=29
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=500, coeff_edge_weight_L1=1, coeff_phi_weight_L1=1, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.948, tau_R2=0.986, V_rest_R2=0.224, cluster_accuracy=0.894, test_R2=0.50, test_pearson=0.992, training_time_min=48.9
Embedding: 65 types well-separated
Mutation: coeff_edge_diff: 750 -> 500
Parent rule: Node 29 — return to baseline edge_diff=500 for comparison
Observation: baseline edge_diff=500 with default L1s achieves conn_R2=0.948 but poor V_rest_R2=0.224; confirms L1 reductions and/or higher edge_diff needed for V_rest
Next: parent=34

## Iter 36: converged
Node: id=36, parent=31
Mode/Strategy: principle-test
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=500, coeff_edge_weight_L1=0.5, coeff_phi_weight_L1=1, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.880, tau_R2=0.990, V_rest_R2=0.376, cluster_accuracy=0.869, test_R2=-2.90, test_pearson=0.991, training_time_min=48.8
Embedding: 65 types moderately separated
Mutation: coeff_W_L1: 1E-4 -> 5E-5. Testing principle: "coeff_W_L1=5E-5 is optimal baseline"
Parent rule: Node 31 — test if baseline W L1 improves over 1E-4
Observation: **PARTIAL** — coeff_W_L1=5E-5 achieves conn_R2=0.880 (lower than Node 31's 0.960); W_L1=1E-4 works better with edge_L1=0.5; principle partially contradicted
Next: parent=34

### Batch 4 Plan (Iter 37-40)
UCB ranking: Node 34 (3.422) > Node 29 (3.402) > Node 35 (3.397) > Node 28 (3.375) > Node 30 (3.364)

Key insights from Iter 33-36:
- **Node 34** is new best: conn_R2=0.973, V_rest_R2=0.709, cluster_acc=0.910 using combined phi_L1=0.5 + edge_L1=0.5
- lr_W=5E-4 is critical for combined L1 reductions (Node 33 at lr_W=7E-4 failed)
- coeff_edge_diff=1000 works well with L1 reductions (Node 34)
- coeff_W_L1=5E-5 vs 1E-4 depends on other regularization settings

Strategy: exploit Node 34's success and explore variations

| Slot | Role | Parent | Focus | Mutation |
| ---- | ---- | ------ | ----- | -------- |
| 0 | exploit | Node 34 | coeff_edge_diff | coeff_edge_diff: 1000 -> 750 (test intermediate value) |
| 1 | exploit | Node 34 | coeff_W_L1 | coeff_W_L1: 5E-5 -> 2E-5 (lower W L1 for V_rest) |
| 2 | explore | Node 29 | phi_L1 | coeff_phi_weight_L1: 1 -> 0.5 (apply winning insight) |
| 3 | principle-test | Node 34 | lr_W | lr_W: 5E-4 -> 7E-4. Testing principle: "lr_W=5E-4 required for combined L1 reductions"

## Iter 37: converged
Node: id=37, parent=34
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_edge_weight_L1=0.5, coeff_phi_weight_L1=0.5, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.935, tau_R2=0.993, V_rest_R2=0.500, cluster_accuracy=0.898, test_R2=-1.33, test_pearson=0.996, training_time_min=51.0
Embedding: 65 types well-separated
Mutation: coeff_edge_diff: 1000 -> 750
Parent rule: Node 34 — test if lower edge_diff improves conn_R2
Observation: coeff_edge_diff=750 degrades conn_R2 from 0.973 to 0.935 and V_rest_R2 from 0.709 to 0.500; coeff_edge_diff=1000 is better with combined L1 reductions
Next: parent=40

## Iter 38: converged
Node: id=38, parent=34
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=1000, coeff_edge_weight_L1=0.5, coeff_phi_weight_L1=0.5, coeff_W_L1=2E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.964, tau_R2=0.989, V_rest_R2=0.439, cluster_accuracy=0.892, test_R2=0.76, test_pearson=0.993, training_time_min=50.8
Embedding: 65 types well-separated
Mutation: coeff_W_L1: 5E-5 -> 2E-5
Parent rule: Node 34 — test if lower W L1 improves V_rest
Observation: coeff_W_L1=2E-5 hurts V_rest_R2 (0.439 vs 0.709) and slightly lowers conn_R2 (0.964 vs 0.973); W_L1=5E-5 is optimal
Next: parent=40

## Iter 39: converged
Node: id=39, parent=29
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_edge_weight_L1=1, coeff_phi_weight_L1=0.5, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.960, tau_R2=0.997, V_rest_R2=0.625, cluster_accuracy=0.858, test_R2=0.39, test_pearson=0.997, training_time_min=49.4
Embedding: 65 types well-separated
Mutation: coeff_phi_weight_L1: 1 -> 0.5
Parent rule: Node 29 — apply winning phi L1 reduction from Node 30
Observation: phi_L1=0.5 with edge_diff=750 achieves conn_R2=0.960, V_rest_R2=0.625, tau_R2=0.997; good balance without edge L1 reduction
Next: parent=40

## Iter 40: converged ⭐ BEST batch conn_R2
Node: id=40, parent=34
Mode/Strategy: principle-test
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=1000, coeff_edge_weight_L1=0.5, coeff_phi_weight_L1=0.5, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.976, tau_R2=0.992, V_rest_R2=0.675, cluster_accuracy=0.865, test_R2=-21.32, test_pearson=0.986, training_time_min=49.0
Embedding: 65 types well-separated
Mutation: lr_W: 5E-4 -> 7E-4. Testing principle: "lr_W=5E-4 required for combined L1 reductions"
Parent rule: Node 34 — test if lr_W=7E-4 works with combined L1 reductions at edge_diff=1000
Observation: **PRINCIPLE CONTRADICTED** — lr_W=7E-4 achieves best conn_R2=0.976 with combined L1 reductions; the key factor is coeff_edge_diff=1000 (not lr_W=5E-4). Node 33 failed because edge_diff=500, not because lr_W=7E-4.
Next: parent=40

### Batch 5 Plan (Iter 41-44)
UCB ranking: Node 40 (3.804) > Node 38 (3.792) > Node 39 (3.788) > Node 29 (3.781) > Node 35 (3.776) > Node 37 (3.763)

Key insights from Iter 37-40:
- **Node 40** is new best: conn_R2=0.976, V_rest_R2=0.675 with lr_W=7E-4, edge_diff=1000, phi_L1=0.5, edge_L1=0.5
- **Principle update**: lr_W=7E-4 works with combined L1 reductions when coeff_edge_diff=1000 (contradicts principle #9)
- coeff_edge_diff=750 with combined L1 reductions (Node 37) is worse than 1000 (0.935 vs 0.976)
- coeff_W_L1=2E-5 (Node 38) hurts V_rest — W_L1=5E-5 is optimal
- Node 39 shows phi_L1=0.5 alone (without edge_L1 reduction) achieves good balance

Strategy: exploit Node 40's success, explore edge_diff variations and test W_L1 at lr_W=7E-4

| Slot | Role | Parent | Mutation |
| ---- | ---- | ------ | -------- |
| 0 | exploit | Node 40 | coeff_edge_diff: 1000 -> 1250 (test higher edge_diff) |
| 1 | exploit | Node 40 | coeff_W_L1: 5E-5 -> 1E-4 (test higher W L1 at lr_W=7E-4) |
| 2 | explore | Node 39 | coeff_edge_weight_L1: 1 -> 0.5 (add edge L1 reduction) |
| 3 | principle-test | Node 40 | coeff_phi_weight_L1: 0.5 -> 0.25. Testing principle: "phi_L1=0.5 is optimal" |

## Iter 41: converged
Node: id=41, parent=40
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=1250, coeff_edge_weight_L1=0.5, coeff_phi_weight_L1=0.5, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.961, tau_R2=0.988, V_rest_R2=0.236, cluster_accuracy=0.876, test_R2=-15.32, test_pearson=0.989, training_time_min=48.5
Embedding: 65 types well-separated
Mutation: coeff_edge_diff: 1000 -> 1250
Parent rule: Node 40 — test if higher edge_diff improves connectivity
Observation: coeff_edge_diff=1250 hurts V_rest_R2 severely (0.236 vs 0.675) while conn_R2 slightly drops (0.961 vs 0.976); edge_diff=1000 is optimal upper bound
Next: parent=42

## Iter 42: converged ⭐ BEST batch conn_R2
Node: id=42, parent=40
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=1000, coeff_edge_weight_L1=0.5, coeff_phi_weight_L1=0.5, coeff_W_L1=1E-4, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.979, tau_R2=0.990, V_rest_R2=0.447, cluster_accuracy=0.844, test_R2=-0.41, test_pearson=0.996, training_time_min=48.9
Embedding: 65 types well-separated
Mutation: coeff_W_L1: 5E-5 -> 1E-4
Parent rule: Node 40 — test higher W L1 at lr_W=7E-4 with combined L1 reductions
Observation: **BEST conn_R2=0.979** with W_L1=1E-4 but V_rest_R2 drops (0.447 vs 0.675) and cluster_acc drops (0.844 vs 0.865). W_L1=1E-4 trades V_rest/cluster for connectivity.
Next: parent=43

## Iter 43: converged ⭐ BEST overall conn_R2
Node: id=43, parent=39
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_edge_weight_L1=0.5, coeff_phi_weight_L1=0.5, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.980, tau_R2=0.991, V_rest_R2=0.387, cluster_accuracy=0.890, test_R2=-32.67, test_pearson=0.989, training_time_min=49.1
Embedding: 65 types well-separated
Mutation: coeff_edge_weight_L1: 1 -> 0.5 (added edge L1 reduction to Node 39)
Parent rule: Node 39 — add edge L1 reduction to phi_L1=0.5 config at edge_diff=750
Observation: **NEW BEST conn_R2=0.980**; combined L1 reductions work at edge_diff=750 with lr_W=5E-4. But V_rest_R2=0.387 is worse than Node 40 (0.675). lr_W=5E-4 + edge_diff=750 maximizes connectivity.
Next: parent=44

## Iter 44: converged
Node: id=44, parent=40
Mode/Strategy: principle-test
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=1000, coeff_edge_weight_L1=0.5, coeff_phi_weight_L1=0.25, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.974, tau_R2=0.985, V_rest_R2=0.649, cluster_accuracy=0.861, test_R2=-1.31, test_pearson=0.996, training_time_min=48.9
Embedding: 65 types well-separated
Mutation: coeff_phi_weight_L1: 0.5 -> 0.25. Testing principle: "phi_L1=0.5 is optimal"
Parent rule: Node 40 — test if lower phi L1 (0.25) improves metrics
Observation: **PRINCIPLE PARTIALLY SUPPORTED** — phi_L1=0.25 achieves similar V_rest_R2 (0.649 vs 0.675) but slightly worse conn_R2 (0.974 vs 0.976). phi_L1=0.5 remains optimal for connectivity, but 0.25 is viable.
Next: parent=43

### Batch 6 Plan (Iter 45-48)
UCB ranking: Node 43 (4.142) > Node 42 (4.141) > Node 44 (4.136) > Node 38 (4.126) > Node 41 (4.122)

Key insights from Iter 41-44:
- **Node 43** achieves new best conn_R2=0.980 with lr_W=5E-4, edge_diff=750, combined L1s
- **Node 42** achieves conn_R2=0.979 with W_L1=1E-4 but sacrifices V_rest and cluster_acc
- coeff_edge_diff=1250 (Node 41) is harmful to V_rest — stick with 750-1000
- coeff_phi_weight_L1=0.25 (Node 44) is viable but not better than 0.5

Strategy: exploit Node 43's success (best conn_R2), explore variations, test edge_diff fine-tuning

| Slot | Role | Parent | Mutation |
| ---- | ---- | ------ | -------- |
| 0 | exploit | Node 43 | lr_W: 5E-4 -> 6E-4 (slightly higher lr_W may improve V_rest) |
| 1 | exploit | Node 42 | coeff_edge_diff: 1000 -> 750 (test lower edge_diff with W_L1=1E-4) |
| 2 | explore | Node 44 | coeff_edge_diff: 1000 -> 750 (test edge_diff=750 with phi_L1=0.25) |
| 3 | principle-test | Node 43 | coeff_phi_weight_L2: 0.001 -> 0.005. Testing principle: "higher phi_L2 can improve parameter recovery without hurting connectivity" |

## Iter 45: converged
Node: id=45, parent=43
Mode/Strategy: exploit
Config: lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_edge_weight_L1=0.5, coeff_phi_weight_L1=0.5, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.970, tau_R2=0.989, V_rest_R2=0.289, cluster_accuracy=0.893, test_R2=-1.68, test_pearson=0.993, training_time_min=49.6
Mutation: lr_W: 5E-4 -> 6E-4
Observation: lr_W=6E-4 achieves conn_R2=0.970 (slightly below Node 43's 0.980) and V_rest_R2 drops to 0.289 (vs 0.387); lr_W=5E-4 remains optimal for this config
Next: parent=root (new block)

## Iter 46: converged
Node: id=46, parent=42
Mode/Strategy: exploit
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_edge_weight_L1=0.5, coeff_phi_weight_L1=0.5, coeff_W_L1=1E-4, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.968, tau_R2=0.989, V_rest_R2=0.584, cluster_accuracy=0.883, test_R2=0.83, test_pearson=0.999, training_time_min=48.0
Mutation: coeff_edge_diff: 1000 -> 750
Observation: edge_diff=750 with W_L1=1E-4 improves V_rest (0.584 vs 0.447) vs parent Node 42; slight conn_R2 drop (0.968 vs 0.979). Lower edge_diff helps V_rest when using higher W_L1.
Next: parent=root (new block)

## Iter 47: converged
Node: id=47, parent=44
Mode/Strategy: explore
Config: lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_edge_weight_L1=0.5, coeff_phi_weight_L1=0.25, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.960, tau_R2=0.990, V_rest_R2=0.559, cluster_accuracy=0.861, test_R2=-1.25, test_pearson=0.995, training_time_min=48.9
Mutation: coeff_edge_diff: 1000 -> 750
Observation: edge_diff=750 with phi_L1=0.25 achieves balanced metrics (conn_R2=0.960, V_rest=0.559); phi_L1=0.25 viable for V_rest vs connectivity trade-off
Next: parent=root (new block)

## Iter 48: converged
Node: id=48, parent=43
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_edge_weight_L1=0.5, coeff_phi_weight_L1=0.5, coeff_phi_weight_L2=0.005, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, recurrent=False
Metrics: connectivity_R2=0.931, tau_R2=0.911, V_rest_R2=0.175, cluster_accuracy=0.869, test_R2=0.75, test_pearson=0.999, training_time_min=48.9
Mutation: coeff_phi_weight_L2: 0.001 -> 0.005. Testing principle: "higher phi_L2 improves parameter recovery"
Observation: **PRINCIPLE REFUTED** — phi_L2=0.005 is harmful: tau_R2 drops to 0.911 (from 0.991), V_rest=0.175 (from 0.387); higher phi_L2 destabilizes training
Next: parent=root (new block)

>>> BLOCK 2 END <<<

## Block 2 Summary: Regularization Parameters

**Iterations:** 25-48 (24 iterations)
**Focus:** coeff_edge_diff, coeff_edge_norm, coeff_edge_weight_L1, coeff_phi_weight_L1, coeff_phi_weight_L2, coeff_W_L1

### Best Configurations in Block 2
| Node | conn_R2 | V_rest_R2 | tau_R2 | cluster_acc | Key config |
|------|---------|-----------|--------|-------------|------------|
| 43 | **0.980** | 0.387 | 0.991 | 0.890 | lr_W=5E-4, edge_diff=750, phi_L1=0.5, edge_L1=0.5 |
| 42 | 0.979 | 0.447 | 0.990 | 0.844 | lr_W=7E-4, edge_diff=1000, W_L1=1E-4 |
| 40 | 0.976 | 0.675 | 0.992 | 0.865 | lr_W=7E-4, edge_diff=1000, phi_L1=0.5, edge_L1=0.5 |
| 34 | 0.973 | 0.709 | 0.997 | **0.910** | lr_W=5E-4, edge_diff=1000, phi_L1=0.5, edge_L1=0.5 |
| 30 | 0.915 | **0.760** | **0.997** | 0.884 | lr_W=5E-4, edge_diff=1000, phi_L1=0.5 |

### Key Findings
1. **coeff_edge_diff=750-1000 is optimal** — 500 is too low for V_rest, 1250+ is harmful
2. **Combined L1 reductions (phi_L1=0.5, edge_L1=0.5) are beneficial** — improve connectivity without hurting tau
3. **coeff_W_L1=5E-5 is optimal for V_rest** — W_L1=1E-4 boosts connectivity but hurts V_rest
4. **coeff_phi_weight_L2 must stay at 0.001** — higher values (0.005, 0.01) are harmful
5. **coeff_edge_norm=10 is catastrophic** — avoid high monotonicity penalty

### Trade-offs Identified
- **Connectivity vs V_rest:** Node 43 (conn_R2=0.980, V_rest=0.387) vs Node 34 (conn_R2=0.973, V_rest=0.709)
- **Best overall balance:** Node 40 (conn_R2=0.976, V_rest=0.675, cluster_acc=0.865)

### Carry-Forward for Block 3
Best configs for architecture exploration:
- **Node 43**: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, edge_diff=750, phi_L1=0.5, edge_L1=0.5, W_L1=5E-5 → conn_R2=0.980
- **Node 34**: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, edge_diff=1000, phi_L1=0.5, edge_L1=0.5, W_L1=5E-5 → conn_R2=0.973, V_rest=0.709

---

## Block 3: Architecture

### Initial Batch Plan (Iter 49-52)
Focus: hidden_dim, n_layers, hidden_dim_update, n_layers_update, embedding_dim

Starting from Node 43 (best conn_R2) with regularization parameters:
- lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3
- coeff_edge_diff=750, coeff_phi_weight_L1=0.5, coeff_edge_weight_L1=0.5, coeff_W_L1=5E-5

| Slot | Role | Parent | Focus | Mutation |
| ---- | ---- | ------ | ----- | -------- |
| 0 | exploit | root | hidden_dim | hidden_dim: 64 -> 96 (larger edge MLP capacity) |
| 1 | exploit | root | n_layers | n_layers: 3 -> 4 (deeper edge MLP) |
| 2 | explore | root | embedding_dim | embedding_dim: 2 -> 4 (more expressive embeddings) |
| 3 | principle-test | root | hidden_dim_update | hidden_dim_update: 64 -> 96. Testing principle: "larger update MLP helps parameter recovery" |

## Iter 49: partial
Node: id=49, parent=root
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=96, n_layers=3, recurrent=False
Metrics: connectivity_R2=0.954, tau_R2=0.992, V_rest_R2=0.615, cluster_accuracy=0.899, test_R2=-1.42, test_pearson=0.972, training_time_min=54.4
Embedding: 65 types well-separated with best cluster_acc in batch
Mutation: hidden_dim: 64 -> 96
Parent rule: root — test larger edge MLP capacity for improved learning
Observation: hidden_dim=96 achieves best cluster_acc=0.899 and good V_rest=0.615 but conn_R2=0.954 slightly below baseline (0.980). Training time increased but acceptable (54.4 min).
Next: parent=49

## Iter 50: partial
Node: id=50, parent=root
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, n_layers=4, recurrent=False
Metrics: connectivity_R2=0.783, tau_R2=0.965, V_rest_R2=0.123, cluster_accuracy=0.804, test_R2=-87.21, test_pearson=0.980, training_time_min=62.8
Embedding: 65 types moderately separated
Mutation: n_layers: 3 -> 4
Parent rule: root — test deeper edge MLP for more expressive functions
Observation: **n_layers=4 is harmful** — conn_R2 drops to 0.783, V_rest_R2=0.123 is catastrophic, training time 62.8 min exceeds limit. Avoid deeper edge MLPs.
Next: parent=49

## Iter 51: partial
Node: id=51, parent=root
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, embedding_dim=4, input_size=5, input_size_update=7, recurrent=False
Metrics: connectivity_R2=0.959, tau_R2=0.992, V_rest_R2=0.403, cluster_accuracy=0.828, test_R2=-0.44, test_pearson=0.997, training_time_min=50.2
Embedding: 65 types well-separated but cluster_acc decreased
Mutation: embedding_dim: 2 -> 4
Parent rule: root — test higher embedding dimension for more expressive neuron representations
Observation: embedding_dim=4 achieves best conn_R2=0.959 in batch but V_rest drops to 0.403 and cluster_acc drops to 0.828. More dimensions don't help type separation; default embedding_dim=2 is sufficient.
Next: parent=49

## Iter 52: partial
Node: id=52, parent=root
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, hidden_dim_update=96, recurrent=False
Metrics: connectivity_R2=0.751, tau_R2=0.994, V_rest_R2=0.508, cluster_accuracy=0.826, test_R2=-5.88, test_pearson=0.982, training_time_min=53.0
Embedding: 65 types moderately separated
Mutation: hidden_dim_update: 64 -> 96. Testing principle: "larger update MLP helps parameter recovery"
Parent rule: root — test larger update MLP for improved tau/V_rest recovery
Observation: **PRINCIPLE PARTIALLY REFUTED** — hidden_dim_update=96 achieves best tau_R2=0.994 but worst conn_R2=0.751. Larger update MLP improves tau but severely hurts connectivity. Trade-off exists.
Next: parent=49

### Batch 2 Plan (Iter 53-56)
UCB ranking: Node 51 (2.373) > Node 49 (2.368) > Node 50 (2.197) > Node 52 (2.165)

Key insights from Iter 49-52:
- **Node 49** (hidden_dim=96) achieves best balance: conn_R2=0.954, V_rest=0.615, cluster_acc=0.899
- **Node 50** (n_layers=4) is harmful — avoid deeper edge MLPs
- **Node 51** (embedding_dim=4) shows extra embedding dims don't help; stick with 2
- **Node 52** (hidden_dim_update=96) improves tau but hurts connectivity — trade-off

Strategy: exploit Node 49's success with hidden_dim=96, explore combinations with other architecture params

| Slot | Role | Parent | Focus | Mutation |
| ---- | ---- | ------ | ----- | -------- |
| 0 | exploit | Node 49 | hidden_dim | hidden_dim: 96 -> 80 (test intermediate value) |
| 1 | exploit | Node 49 | hidden_dim_update | hidden_dim_update: 64 -> 80 (moderate update MLP increase) |
| 2 | explore | Node 51 | lr_emb | lr_emb: 1.5E-3 -> 1.8E-3 (higher lr_emb for larger embedding) |
| 3 | principle-test | Node 49 | n_layers_update | n_layers_update: 3 -> 4. Testing principle: "deeper update MLP helps parameter recovery" |

## Iter 53: converged ⭐ BEST batch V_rest_R2
Node: id=53, parent=49
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=80, n_layers=3, recurrent=False
Metrics: connectivity_R2=0.968, tau_R2=0.980, V_rest_R2=0.735, cluster_accuracy=0.882, test_R2=-1.03, test_pearson=0.996, training_time_min=52.8
Embedding: 65 types well-separated with excellent V_rest recovery
Mutation: hidden_dim: 96 -> 80
Parent rule: Node 49 — test intermediate hidden_dim between 64 and 96
Observation: **hidden_dim=80 is optimal** — achieves best V_rest_R2=0.735 and best conn_R2=0.968 in batch; better balance than 64 or 96. Training time (52.8 min) is acceptable.
Next: parent=53

## Iter 54: converged ⭐ BEST batch tau_R2 + cluster_acc
Node: id=54, parent=49
Mode/Strategy: exploit
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=96, hidden_dim_update=80, n_layers=3, recurrent=False
Metrics: connectivity_R2=0.959, tau_R2=0.995, V_rest_R2=0.752, cluster_accuracy=0.892, test_R2=0.71, test_pearson=0.976, training_time_min=58.5
Embedding: 65 types well-separated with excellent metrics
Mutation: hidden_dim_update: 64 -> 80
Parent rule: Node 49 — test moderate increase in update MLP hidden dimension
Observation: **hidden_dim_update=80 is beneficial** — achieves best tau_R2=0.995, best cluster_acc=0.892, and good V_rest_R2=0.752. Time (58.5 min) acceptable but approaching limit.
Next: parent=54

## Iter 55: partial — CONFIRMS PRINCIPLE 4
Node: id=55, parent=51
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.8E-3, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, embedding_dim=4, input_size=5, input_size_update=7, recurrent=False
Metrics: connectivity_R2=0.942, tau_R2=0.985, V_rest_R2=0.358, cluster_accuracy=0.798, test_R2=-0.65, test_pearson=0.995, training_time_min=56.1
Embedding: 65 types moderately separated with lower cluster_acc
Mutation: lr_emb: 1.5E-3 -> 1.8E-3
Parent rule: Node 51 — test higher lr_emb with larger embedding dimension
Observation: **CONFIRMS PRINCIPLE 4** — lr_emb=1.8E-3 destroys V_rest_R2 (0.358) and hurts cluster_acc (0.798). Even with embedding_dim=4, high lr_emb is harmful. Stick with lr_emb=1.5E-3.
Next: parent=54

## Iter 56: partial — REFUTES PRINCIPLE
Node: id=56, parent=49
Mode/Strategy: principle-test
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=96, n_layers_update=4, recurrent=False
Metrics: connectivity_R2=0.951, tau_R2=0.989, V_rest_R2=0.357, cluster_accuracy=0.854, test_R2=0.85, test_pearson=0.999, training_time_min=56.6
Embedding: 65 types moderately separated
Mutation: n_layers_update: 3 -> 4. Testing principle: "deeper update MLP helps parameter recovery"
Parent rule: Node 49 — test deeper update MLP for improved tau/V_rest
Observation: **PRINCIPLE REFUTED** — n_layers_update=4 causes V_rest_R2 collapse (0.357) despite maintaining decent conn_R2 (0.951). Deeper update MLP does NOT help parameter recovery; it hurts V_rest while keeping tau okay. Avoid n_layers_update > 3.
Next: parent=53

### Batch 3 Plan (Iter 57-60)
UCB ranking: Node 53 (2.968) > Node 54 (2.959) > Node 51 (2.959) > Node 56 (2.951) > Node 55 (2.942)

Key insights from Iter 53-56:
- **Node 53** (hidden_dim=80) is best: conn_R2=0.968, V_rest_R2=0.735 — intermediate value outperforms 64 and 96
- **Node 54** (hidden_dim_update=80) achieves best tau_R2=0.995, V_rest=0.752, cluster_acc=0.892
- **Node 55** confirms lr_emb >= 1.8E-3 is harmful (V_rest collapse)
- **Node 56** refutes "deeper update MLP helps" — n_layers_update=4 causes V_rest collapse

Strategy: exploit Node 53's success with hidden_dim=80; explore combining hidden_dim=80 with hidden_dim_update=80

| Slot | Role | Parent | Focus | Mutation |
| ---- | ---- | ------ | ----- | -------- |
| 0 | exploit | Node 53 | hidden_dim_update | hidden_dim_update: 64 -> 80 (combine with hidden_dim=80) |
| 1 | exploit | Node 54 | hidden_dim | hidden_dim: 96 -> 80 (test intermediate hidden_dim with update=80) |
| 2 | explore | Node 53 | coeff_edge_diff | coeff_edge_diff: 750 -> 1000 (test edge_diff=1000 with hidden_dim=80) |
| 3 | principle-test | Node 54 | coeff_phi_weight_L1 | coeff_phi_weight_L1: 0.5 -> 0.75. Testing principle: "phi_L1=0.5 is optimal" |

