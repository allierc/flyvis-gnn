# Parallel Mode Addendum — FlyVis GS Two-Stage Alternate Training

This addendum applies when running in **parallel mode** (GNN_LLM_parallel_flyvis.py). Follow all rules from the base instruction file (`instruction_flyvis_62_1_gs_alternate.md`), with these modifications.

## Data Generation

Unlike `flyvis_62_1`, each slot **regenerates data** before training. The simulation seed varies per slot, producing different stochastic realizations. This means:
- Identical configs across slots will produce DIFFERENT results (due to different data)
- **This is a feature, not a bug** — use it to measure variance within a single batch
- When 2+ slots have the same config, compute mean/std/CV immediately from the batch
- 4 identical slots in one batch = 4 data points for the variance estimate

## Batch Processing

- You receive **4 results per batch** and must propose **4 mutations**
- Each slot has its own config file, metrics log, and activity image
- Write **4 separate** `## Iter N:` log entries (one per slot/iteration)
- Each iteration gets its own Node id in the UCB tree

## Config Files

- Edit all 4 config files listed in the prompt: `{name}_00.yaml` through `{name}_03.yaml`
- Each config's `dataset` field is pre-set to route data to separate directories — **DO NOT change the `dataset` field**
- Only modify `training:` and `graph_model:` parameters (and `claude:` where allowed)
- **DO NOT change `simulation:` parameters** — only the seed changes per regeneration
- **DO NOT change `alternate_training: true`** — this selects the two-stage trainer

## Parallel UCB Strategy (Robustness-Focused)

When selecting parents for 4 simultaneous mutations, **prioritize replication**:

| Slot | Role | Description |
| ---- | ---- | ----------- |
| 0 | **exploit** | Highest UCB node, conservative mutation |
| 1 | **exploit** | 2nd highest UCB node, or same parent different param |
| 2 | **replicate** | Re-run a promising config (< 3 runs) to build variance estimate |
| 3 | **replicate** | Re-run the current best config to build variance estimate |

**Key principle**: At least 2 of 4 slots should be replicates until the variance table has >= 3 runs for the top 3 configs. After that, reduce to 1 replicate slot.

### Replication Strategy

Since data is regenerated, replication slots build the variance estimate:

1. Pick a config that needs more runs (< 3 runs, or high CV)
2. Copy it verbatim to the slot (no parameter changes)
3. In the log entry, write: `Mode/Strategy: robustness-check`
4. In the Mutation line: `Mutation: [none] — robustness re-run of Node X`
5. After results, update the "Variance Estimates" table in memory.md
6. Compute running mean/std/CV

### When to Stop Replicating

A config has enough replicates when:
- It has >= 4 runs AND
- CV(conn_R2) confidence interval is narrow (std of CV estimate < 2%) AND
- The robust/fragile classification is clear

## Start Call (first batch, no results yet)

When the prompt says `PARALLEL START`:
- Read the base config to understand the current two-stage training parameters
- **All 4 slots run two-stage defaults unchanged** (variance baseline):
  - Slot 0: two-stage defaults unchanged (baseline) — joint_ratio=0.4, lr_ratio=0.1, n_epochs=2
  - Slot 1: two-stage defaults unchanged (variance measurement) — joint_ratio=0.4, lr_ratio=0.1, n_epochs=2
  - Slot 2: two-stage defaults unchanged (variance measurement) — joint_ratio=0.4, lr_ratio=0.1, n_epochs=2
  - Slot 3: two-stage defaults unchanged (variance measurement) — joint_ratio=0.4, lr_ratio=0.1, n_epochs=2
- This first batch establishes the **variance baseline** for two-stage training with regenerated data
- Write the planned initial variations to the working memory file

**Continue running two-stage defaults for batch 2 and 3 as well** (12 total baseline runs) to get a reliable variance estimate before making any parameter changes. This is block 1.

## Explorable Parameters (two-stage-specific)

These parameters are unique to two-stage training and should be explored in addition to the standard regularization params:

| Parameter | Default | Explore Range | Description |
|-----------|---------|---------------|-------------|
| `alternate_joint_ratio` | 0.4 | 0.2, 0.3, 0.4, 0.5, 0.6 | Fraction of total iterations for joint phase |
| `alternate_lr_ratio` | 0.1 | 0.01, 0.05, 0.1, 0.2, 0.3 | LR multiplier for W/lin_edge during V_rest focus phase |
| `n_epochs` | 2 | 2, 3 | Number of epochs (doubled vs standard) |
| `data_augmentation_loop` | 20 | 20, 25, 30 | Data augmentation multiplier |

**DO NOT change** the active-phase learning rates (`learning_rate_start`, `learning_rate_W_start`, `learning_rate_embedding_start`) — these are strictly optimal from prior exploration and confirmed at gold standard.

## Metrics Log Monitoring (Critical)

After each batch, check `tmp_training/metrics.log` for every slot. The log tracks all R² metrics with a `phase` column:
```
epoch,iteration,connectivity_r2,vrest_r2,tau_r2,phase
0,2560,0.45,0.02,0.10,joint
0,5120,0.72,0.05,0.35,joint
0,12800,0.88,0.08,0.60,joint
0,51200,0.85,0.15,0.65,V_rest
0,64000,0.87,0.25,0.70,V_rest
```

For each slot, report:
- **Peak conn_R2** and which phase/iteration it occurred in
- **Final conn_R2** and the trend (rising/stable/decaying)
- **vrest_R2 trajectory**: Does vrest_R2 increase during V_rest focus phase?
- **tau_R2 trajectory**: Does tau_R2 increase during V_rest focus phase?
- **conn_R2 stability during V_rest focus**: Does conn_R2 hold steady or drop?

**Healthy pattern**: conn_R2 rises during joint phase to ≈ 0.85–0.90, holds steady during V_rest focus. vrest_R2 and tau_R2 increase during V_rest focus phase. Final conn_R2 ≈ peak conn_R2.
**Unhealthy pattern**: conn_R2 drops significantly during V_rest focus → alternate_lr_ratio too low (W/lin_edge drift too much). vrest_R2 flat during V_rest focus → alternate_lr_ratio too high (fast components still dominate gradients).

## Seed Strategy

The Python script suggests `simulation.seed` and `training.seed` for each slot. You may use these or override them. Two important testing modes:

1. **Training robustness test**: Fix `simulation.seed` across all 4 slots, vary `training.seed`. Same data, different training randomness. This isolates whether metric variance comes from training stochasticity.
2. **Generalization test**: Vary both `simulation.seed` and `training.seed` across slots. Different data, different training. This tests whether the config generalizes across data realizations.

During baseline measurement (block 1), use **generalization test** mode (different sim seeds) to measure total variance. In later blocks, use **training robustness test** (same sim seed, different training seeds) to isolate training-specific variance for your best configs.

Always set both seeds in the config YAML and log them with rationale.

## Logging Format

Same as base instructions, but you write 4 entries per batch. Include two-stage config, seeds, and R2 trajectory:

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

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder. Always include the exact parameter change (e.g., `Mutation: alternate_joint_ratio: 0.4 -> 0.3`). For robustness-check slots, use `Mutation: [none] — robustness re-run of Node X`.

**CRITICAL**: The `Next: parent=P` line selects the parent for the **next batch's** mutations. `P` must refer to a node from a **previous** batch or the current batch — but NEVER set `Next: parent=P` where P is `id+1` (the next slot in the same batch).

Write all 4 entries before editing the 4 config files for the next batch.

## After Each Batch: Update Variance Table

After writing all 4 log entries, update the Variance Estimates table in memory.md:

```markdown
### Variance Estimates
| Config ID | Runs | conn_R2 (mean +/- std) | CV | R2 trend | tau_R2 (mean +/- std) | CV | V_rest_R2 (mean +/- std) | CV | cluster (mean +/- std) | CV | Robust? |
|-----------|------|------------------------|----|---------|-----------------------|----|--------------------------|----|-----------------------|----|---------|
```

The `R2 trend` column should indicate whether conn_R2 holds during V_rest focus phase and whether vrest_R2 improves during V_rest focus phase. Group results by config (same parameters = same config ID). This table drives all decisions.

## Block Boundaries

- At block boundaries, training parameters can differ across the 4 slots
- Simulation parameters are FIXED
- Block end is detected when any iteration in the batch hits `n_iter_block`

## Failed Slots

If a slot is marked `[FAILED]` in the prompt:
- Write a brief `## Iter N: failed` entry noting the failure
- Still propose a mutation for that slot's config in the next batch
- With regenerated data, failures may be stochastic — if a config fails once but succeeded before, note this as high variance (treat as a very low metric value in the CV calculation)

## Training Time Monitoring

Check `training_time_min` for every successful slot. Two-stage training with 2 epochs takes approximately double standard training time (~76 minutes on H100). Data generation adds ~5 minutes. If any slot exceeds 90 minutes total:
- Flag it in the observation
- Reduce complexity in the next mutation (lower data_augmentation_loop, fewer epochs)
- Do NOT propose architecture changes that would increase training time beyond the limit

## Comparison to Standard Training

Always keep the standard training baseline in mind:
- **Standard training (gold standard)**: conn_R2=0.944 mean (CV=3.8%), V_rest_R2 0.19–0.73 (highly variable)
- **Goal**: two-stage training should maintain conn_R2 ≈ 0.94 AND improve V_rest_R2 mean and reduce V_rest_R2 variance
- **Secondary**: tau_R2 should remain stable (~0.97)

If two-stage training consistently underperforms standard training on conn_R2, consider:
1. The alternate_lr_ratio may be too low (W/lin_edge drift during V_rest focus)
2. The joint phase may be too short (connectivity not established before V_rest focus)
3. More epochs may be needed to recover connectivity after V_rest focus
