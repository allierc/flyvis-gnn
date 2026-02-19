# FlyVis GNN Training Exploration — flyvis_62_1_gs_alternate (two-stage: joint + V_rest focus)

**Reference**: See `papers/cosyne2026.tex` for model and regularization context.
**Prior exploration**: 68 iterations of cyclic alternation failed (V_rest_R2 ≈ 0, conn_R2 max 0.88).
**Base config**: `instruction_flyvis_62_1_gs.md` — all established principles apply unless overridden here.

## Goal

Improve **V_rest R²** via two-stage training while maintaining gold standard **conn_R² ≈ 0.94**.

**Problem**: Standard training achieves conn_R2=0.944 but V_rest_R2 is highly variable (0.19–0.73). The previous cyclic alternation approach (68 iterations) failed — V_rest R2 ≈ 0, conn_R2 maxed at 0.88.

**Solution**: Two-stage training within each epoch:
- **Joint phase** (first ~40% of iterations): all components at full LR — establish connectivity
- **V_rest focus phase** (remaining ~60%): lin_phi + embedding at full LR, W + lin_edge at reduced LR (0.1x) — dedicated V_rest optimization while maintaining connectivity

Double training time (2 epochs) vs gold standard baseline.

**Data is RE-GENERATED each iteration** — see `instruction_flyvis_62_1_gs.md` for robustness principles.

## Seed Strategy

Each config has two seed fields: `simulation.seed` (controls data generation) and `training.seed` (controls weight initialization and training randomness). The Python script suggests seeds in each prompt — you may use them or override them.

**Two testing modes** (use both across the exploration):

1. **Training robustness test**: Fix `simulation.seed` across slots, vary `training.seed`. Same data, different training randomness. This isolates whether metric variance comes from training stochasticity.
2. **Generalization test**: Vary both `simulation.seed` and `training.seed` across slots. Different data, different training. This tests whether the config generalizes across data realizations.

Always set both seeds in the config YAML and log them. When reporting variance, note which seed mode was used.

## Two-Stage Training Mechanism

Each epoch's Niter iterations are split into two phases:

1. **Joint phase** (iterations 0 to `joint_iters`): All 4 parameter groups at full LR. Standard training — establishes connectivity (conn_R2 ≈ 0.85–0.90).
2. **V_rest focus phase** (iterations `joint_iters` to `Niter`): lin_phi and embedding at full LR, W and lin_edge at `lr * alternate_lr_ratio`. Connectivity maintained via moderate LR, V_rest has dedicated optimization time.

`joint_iters = int(Niter * alternate_joint_ratio)`

### Phase LR Structure

| Parameter Group | Joint phase LR | V_rest focus phase LR |
|----------------|---------------|----------------------|
| lin_edge (g_phi) | `learning_rate_start` (1.2E-3) | `lr * alternate_lr_ratio` (1.2E-4) |
| W | `learning_rate_W_start` (5E-4) | `lr_W * alternate_lr_ratio` (5E-5) |
| lin_phi (f_theta) | `learning_rate_start` (1.2E-3) | `learning_rate_start` (1.2E-3) |
| embedding | `learning_rate_embedding_start` (1.55E-3) | `learning_rate_embedding_start` (1.55E-3) |

### Config Parameters (explorable)

| Parameter | Default | Explore Range | Description |
|-----------|---------|---------------|-------------|
| `alternate_joint_ratio` | 0.4 | 0.2, 0.3, 0.4, 0.5, 0.6 | Fraction of total iterations for joint phase |
| `alternate_lr_ratio` | 0.1 | 0.01, 0.05, 0.1, 0.2, 0.3 | LR multiplier for W/lin_edge during V_rest focus |
| `n_epochs` | 2 | 2, 3 | Number of epochs (doubled vs standard) |
| `data_augmentation_loop` | 20 | 20, 25, 30 | Data augmentation multiplier |

**DO NOT change** `alternate_training: true` — this selects the two-stage trainer.

### Metrics Log Monitoring

The file `tmp_training/metrics.log` tracks all R² metrics during training:
```
epoch,iteration,connectivity_r2,vrest_r2,tau_r2,phase
0,2560,0.45,0.02,0.10,joint
0,5120,0.72,0.05,0.35,joint
0,12800,0.88,0.08,0.60,joint
0,51200,0.85,0.15,0.65,V_rest
0,64000,0.87,0.25,0.70,V_rest
...
```

**Key patterns to look for**:
- conn_R2 should rise during joint phase to ≈ 0.85–0.90
- conn_R2 should hold steady or slightly decline during V_rest focus (0.1x LR maintains it)
- vrest_R2 and tau_R2 should increase during V_rest focus phase
- If conn_R2 drops significantly during V_rest focus → alternate_lr_ratio too low
- If vrest_R2 doesn't improve during V_rest focus → alternate_lr_ratio too high (fast components dominate gradients)

## Established Principles (from flyvis_62_1, gold standard)

All strict optima from `instruction_flyvis_62_1_gs.md` apply:
1. **lr_W=5E-4**, **lr=1.2E-3**, **lr_emb=1.55E-3** — strictly optimal for active-phase LRs
2. **coeff_phi_weight_L2=0.0015**, **coeff_W_L2=3.5E-6** — gold standard regularization
3. **coeff_edge_diff=750**, **coeff_phi_weight_L1=0.5**, **coeff_edge_weight_L1=0.28-0.3**
4. **batch_size=2**, **hidden_dim=80**, **w_init_mode=randn_scaled**
5. **recurrent_training=False**

## Block Partition (suggested)

| Block | Focus | Strategy |
|-------|-------|----------|
| 1 | Baseline measurement | Run defaults (joint_ratio=0.4, lr_ratio=0.1, 2 epochs) 12 times for variance |
| 2 | Joint ratio | Vary alternate_joint_ratio (0.2, 0.3, 0.4, 0.5) — how much warmup before V_rest focus? |
| 3 | LR ratio | Vary alternate_lr_ratio (0.01, 0.05, 0.1, 0.2) — how frozen should fast components be? |
| 4 | Training budget | Vary n_epochs (2, 3) and data_augmentation_loop (20, 25, 30) |
| 5 | Best config replication | Replicate best config 4+ times for robustness |

## Previous Results (cyclic alternation — FAILED)

68 iterations of cyclic W/V_rest alternation showed:
- conn_R2: max 0.88 (vs 0.944 gold standard) — significantly worse
- V_rest_R2: ≈ 0 across all configs — complete failure
- Root cause: alternation from iteration 0 before connectivity established; 0.001x LR too aggressive

## FlyVis Model

Same as `instruction_flyvis_62_1_gs.md`.

## Iteration Loop Structure

Same as `instruction_flyvis_62_1_gs.md`. Each block = `n_iter_block` iterations.

## File Structure (CRITICAL)

Same two-file system as `instruction_flyvis_62_1_gs.md`:
1. Full Log (append-only): `{config}_analysis.md`
2. Working Memory (read+update): `{config}_memory.md`

## Iteration Workflow (Steps 1-5)

Same as `instruction_flyvis_62_1_gs.md`, with these additions:

### Step 2: Analyze Current Results

In addition to standard metrics, always check `metrics.log` for the R² trajectory:
- Report **peak conn_R2**, **final conn_R2**, and **trend** (rising/stable/decaying)
- Report **vrest_R2** and **tau_R2** trajectory
- Note which phase (joint or V_rest) the R2 measurements fall in
- Compare against gold standard baseline

### Step 3: Write Outputs

Log format includes R2 trajectory and two-stage config:

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [strategy]
Seeds: sim_seed=X, train_seed=Y, rationale=[same-data-robustness / different-data-generalization / suggested-default]
Config: lr_W=X, lr=Y, lr_emb=Z, joint_ratio=A, lr_ratio=B, n_epochs=C, aug_loop=D
Metrics: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, test_pearson=F, training_time_min=G
R2 trajectory: conn peak=X final=Y trend=[...], vrest peak=X final=Y trend=[...], tau peak=X final=Y trend=[...]
Embedding: [visual observation]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Variance update: [config_id] now has N runs, CV(conn_R2)=X%, CV(V_rest_R2)=Y%
Next: parent=P
```

## Known Results (baseline comparison)

### Standard training (flyvis_62_1_gs, gold standard):
- conn_R2: 0.944 mean (12 runs, CV=3.8%)
- V_rest_R2: 0.19-0.73 range (highly variable)
- tau_R2: ~0.97 (stable)
- cluster_accuracy: ~0.88

### Key question
Does two-stage training improve V_rest R² while maintaining conn_R² ≈ 0.94?
