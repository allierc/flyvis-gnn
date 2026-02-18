# FlyVis GNN Training Exploration — flyvis_62_1_gs_alternate (alternating W/V_rest phases)

**Reference**: See `papers/cosyne2026.tex` for model and regularization context.
**Prior exploration**: 144 iterations on `flyvis_62_1` (see `log/Claude_exploration/instruction_flyvis_62_1_parallel/`).
**Base config**: `instruction_flyvis_62_1_gs.md` — all established principles apply unless overridden here.

## Goal

Find an **alternating training schedule** that prevents the conn_R2 overshoot-then-decay pattern observed in standard training.

**Problem**: In standard training, conn_R2 peaks above 0.9 early but decays below 0.9 by end of training. The fast components (lin_edge, W) and slow components (lin_phi, embedding) interfere when trained simultaneously.

**Solution**: Alternate between two training phases within each epoch:
- **W-phase**: Full LR for lin_edge + W, reduced LR for lin_phi + embedding
- **V_rest-phase**: Full LR for lin_phi + embedding, reduced LR for lin_edge + W

**Primary metric**: Final conn_R2 should be >= peak conn_R2 (no decay). Secondary: V_rest_R2 should improve with dedicated slow-component training.

**Data is RE-GENERATED each iteration** — see `instruction_flyvis_62_1_gs.md` for robustness principles.

## Seed Strategy

Each config has two seed fields: `simulation.seed` (controls data generation) and `training.seed` (controls weight initialization and training randomness). The Python script suggests seeds in each prompt — you may use them or override them.

**Two testing modes** (use both across the exploration):

1. **Training robustness test**: Fix `simulation.seed` across slots, vary `training.seed`. Same data, different training randomness. This isolates whether metric variance comes from training stochasticity.
2. **Generalization test**: Vary both `simulation.seed` and `training.seed` across slots. Different data, different training. This tests whether the config generalizes across data realizations.

Always set both seeds in the config YAML and log them. When reporting variance, note which seed mode was used.

## Alternating Training Mechanism

The epoch's Niter iterations are split into `n_alternations` cycles. Each cycle has a W-phase and a V_rest-phase. The `alternate_vrest_ratio` controls the fraction of each cycle devoted to V_rest-phase.

### Phase LR Structure

Each of the 4 parameter groups has its own LR for each phase:

| Parameter Group | W-phase LR | V_rest-phase LR |
|----------------|------------|-----------------|
| lin_edge (g_phi) | `learning_rate_start` (1.2E-3) | `alternate_lr_edge` (1.2E-6) |
| W | `learning_rate_W_start` (6E-4) | `alternate_lr_W` (6E-7) |
| lin_phi (f_theta) | `alternate_lr_update` (1.2E-6) | `learning_rate_start` (1.2E-3) |
| embedding | `alternate_lr_embedding` (1.55E-6) | `learning_rate_embedding_start` (1.55E-3) |

Default inactive LRs are 0.001x the active LR (nearly frozen, but not zero).

### Config Parameters (explorable)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_alternations` | 4 | Number of W/V_rest cycles per epoch |
| `alternate_vrest_ratio` | 0.5 | Fraction of each cycle for V_rest-phase (0.5 = equal) |
| `alternate_lr_W` | 6E-7 | W learning rate during V_rest-phase |
| `alternate_lr_edge` | 1.2E-6 | lin_edge learning rate during V_rest-phase |
| `alternate_lr_update` | 1.2E-6 | lin_phi learning rate during W-phase |
| `alternate_lr_embedding` | 1.55E-6 | embedding learning rate during W-phase |

**DO NOT change** `alternate_training: true` — this selects the alternate trainer.

### R2 Trajectory Monitoring

The file `tmp_training/connectivity_r2.log` now includes a `phase` column:
```
epoch,iteration,connectivity_r2,phase
0,640,0.45,W
0,1280,0.72,W
0,1920,0.71,V_rest
0,2560,0.85,W
...
```

**Key pattern to look for**: Does R2 hold steady or increase during V_rest-phase? Does it jump during W-phase? A healthy alternating training should show:
- R2 increases during W-phases (fast component learning)
- R2 holds stable or slightly increases during V_rest-phases (slow component catches up without W drift)
- Final R2 >= peak R2 (no decay)

**Unhealthy pattern**: R2 drops during V_rest-phase → the inactive LR for W/lin_edge may be too high (model drifts), or the V_rest-phase is too long.

## Established Principles (from flyvis_62_1, 144 iterations)

All strict optima from `instruction_flyvis_62_1_gs.md` apply:
1. **lr_W=6E-4**, **lr=1.2E-3**, **lr_emb=1.55E-3** — strictly optimal for active-phase LRs
2. **coeff_edge_diff=750**, **coeff_phi_weight_L1=0.5**, **coeff_edge_weight_L1=0.28-0.3**
3. **batch_size=2**, **hidden_dim=80**, **w_init_mode=zeros**
4. **recurrent_training=False** (globally — but may be explored during V_rest-phase only in later blocks)

## Block Partition (suggested)

| Block | Focus | Strategy |
|-------|-------|----------|
| 1 | Baseline measurement | Run defaults (n_alt=4, ratio=0.5, default inactive LRs) 8-12 times to measure variance and establish whether alternating helps vs standard training |
| 2 | n_alternations | Test 2, 4, 6, 8 cycles per epoch — more cycles = more fine-grained alternation |
| 3 | Inactive LRs | Test alternate_lr_W and alternate_lr_edge at 0, 1E-7, 1E-6, 1E-5 — how frozen should the fast component be during V_rest-phase? |
| 4 | V_rest ratio | Test alternate_vrest_ratio at 0.3, 0.5, 0.7 — does more V_rest-phase time improve V_rest_R2? |
| 5 | Slow-phase LRs | Test alternate_lr_update and alternate_lr_embedding at different values during W-phase |
| 6 | Best config replication | Replicate best config 4+ times for robustness |

## FlyVis Model

Same as `instruction_flyvis_62_1_gs.md` — see that file for model equations, architecture, regularization parameters, and training time constraints.

## Iteration Loop Structure

Same as `instruction_flyvis_62_1_gs.md`. Each block = `n_iter_block` iterations.

## File Structure (CRITICAL)

Same two-file system as `instruction_flyvis_62_1_gs.md`:
1. Full Log (append-only): `{config}_analysis.md`
2. Working Memory (read+update): `{config}_memory.md`

## Iteration Workflow (Steps 1-5)

Same as `instruction_flyvis_62_1_gs.md`, with these additions:

### Step 2: Analyze Current Results

In addition to standard metrics, always check `connectivity_r2.log` for the R2 trajectory:
- Report **peak R2**, **final R2**, and **trend** (rising/stable/decaying)
- Note which phase (W or V_rest) the R2 measurements fall in
- Compare final R2 to standard training baseline

### Step 3: Write Outputs

Log format includes R2 trajectory and alternation config:

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [strategy]
Seeds: sim_seed=X, train_seed=Y, rationale=[same-data-robustness / different-data-generalization / suggested-default]
Config: lr_W=X, lr=Y, lr_emb=Z, n_alt=A, vrest_ratio=B, alt_lr_W=C, alt_lr_edge=D, alt_lr_update=E, alt_lr_emb=F
Metrics: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, test_pearson=F, training_time_min=G
R2 trajectory: peak=X at iter Y, final=Z, trend=[rising/stable/decaying]
Embedding: [visual observation]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line — compare to standard training R2 decay pattern]
Variance update: [config_id] now has N runs, CV(conn_R2)=X%, CV(V_rest_R2)=Y%
Next: parent=P
```

## Known Results (baseline comparison)

### Standard training (flyvis_62_1_gs, no alternation):
- conn_R2: typically 0.95-0.98 peak, may decay to 0.88-0.93 by end
- V_rest_R2: 0.19-0.73 range (highly variable)
- tau_R2: ~0.97 (stable)
- cluster_accuracy: ~0.88

### Key question
Does alternating training prevent the conn_R2 peak-then-decay pattern while maintaining or improving V_rest_R2?
