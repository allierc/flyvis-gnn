# Parallel Mode Addendum — FlyVis GS Alternating Training

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
- **DO NOT change `alternate_training: true`** — this selects the alternate trainer

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
- Read the base config to understand the current alternating training parameters
- **All 4 slots run alternate defaults unchanged** (variance baseline):
  - Slot 0: alternate defaults unchanged (baseline)
  - Slot 1: alternate defaults unchanged (variance measurement)
  - Slot 2: alternate defaults unchanged (variance measurement)
  - Slot 3: alternate defaults unchanged (variance measurement)
- This first batch establishes the **variance baseline** for alternating training with regenerated data
- Write the planned initial variations to the working memory file

**Continue running alternate defaults for batch 2 and 3 as well** (12 total baseline runs) to get a reliable variance estimate before making any parameter changes. This is block 1.

## Explorable Parameters (alternating-specific)

These parameters are unique to alternating training and should be explored in addition to the standard regularization params:

| Parameter | Default | Explore Range | Description |
|-----------|---------|---------------|-------------|
| `n_alternations` | 4 | 2, 4, 6, 8 | Number of W/V_rest cycles per epoch |
| `alternate_vrest_ratio` | 0.5 | 0.3, 0.5, 0.7 | Fraction of each cycle for V_rest-phase |
| `alternate_lr_W` | 6E-7 | 0, 1E-7, 6E-7, 6E-6 | W learning rate during V_rest-phase |
| `alternate_lr_edge` | 1.2E-6 | 0, 1E-7, 1.2E-6, 1.2E-5 | lin_edge learning rate during V_rest-phase |
| `alternate_lr_update` | 1.2E-6 | 0, 1E-7, 1.2E-6, 1.2E-5 | lin_phi learning rate during W-phase |
| `alternate_lr_embedding` | 1.55E-6 | 0, 1E-7, 1.55E-6, 1.55E-5 | embedding learning rate during W-phase |

**DO NOT change** the active-phase learning rates (`learning_rate_start`, `learning_rate_W_start`, `learning_rate_embedding_start`) — these are strictly optimal from 144 iterations of flyvis_62_1 exploration.

## R2 Trajectory Monitoring (Critical)

After each batch, check `tmp_training/connectivity_r2.log` for every slot. The log includes a `phase` column:
```
epoch,iteration,connectivity_r2,phase
0,640,0.45,W
0,1280,0.72,W
0,1920,0.71,V_rest
0,2560,0.85,W
```

For each slot, report:
- **Peak R2** and which phase/iteration it occurred in
- **Final R2** and the trend (rising/stable/decaying)
- **R2 stability during V_rest-phase**: Does R2 hold steady or drop?
- **R2 gains during W-phase**: How much does R2 increase per W-phase?

**Healthy pattern**: R2 increases during W-phases, holds stable during V_rest-phases, final >= peak.
**Unhealthy pattern**: R2 drops during V_rest-phases → inactive LR for W/lin_edge may be too high.

## Logging Format

Same as base instructions, but you write 4 entries per batch. Include alternation config and R2 trajectory:

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [strategy]
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

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder. Always include the exact parameter change (e.g., `Mutation: n_alternations: 4 -> 6`). For robustness-check slots, use `Mutation: [none] — robustness re-run of Node X`.

**CRITICAL**: The `Next: parent=P` line selects the parent for the **next batch's** mutations. `P` must refer to a node from a **previous** batch or the current batch — but NEVER set `Next: parent=P` where P is `id+1` (the next slot in the same batch).

Write all 4 entries before editing the 4 config files for the next batch.

## After Each Batch: Update Variance Table

After writing all 4 log entries, update the Variance Estimates table in memory.md:

```markdown
### Variance Estimates
| Config ID | Runs | conn_R2 (mean +/- std) | CV | R2 trend | tau_R2 (mean +/- std) | CV | V_rest_R2 (mean +/- std) | CV | cluster (mean +/- std) | CV | Robust? |
|-----------|------|------------------------|----|---------|-----------------------|----|--------------------------|----|-----------------------|----|---------|
```

The `R2 trend` column should indicate whether final R2 >= peak R2 across runs. Group results by config (same parameters = same config ID). This table drives all decisions.

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

Check `training_time_min` for every successful slot. Data generation adds ~5 minutes. If any slot exceeds 60 minutes total:
- Flag it in the observation
- Reduce complexity in the next mutation (smaller hidden_dim, lower data_augmentation_loop)
- Do NOT propose architecture changes that would increase training time beyond the limit

## Comparison to Standard Training

Always keep the standard training baseline in mind:
- **Standard training**: conn_R2 peaks 0.95-0.98, may decay to 0.88-0.93
- **Goal**: alternating training should achieve final conn_R2 >= peak conn_R2 (no decay)
- **Secondary**: V_rest_R2 should improve with dedicated slow-component training

If alternating training consistently underperforms standard training on conn_R2, consider:
1. The inactive LRs may be too high (causing drift)
2. The V_rest-phase may be too long (reducing effective W training time)
3. The number of alternations may be too low (not enough fine-grained switching)
