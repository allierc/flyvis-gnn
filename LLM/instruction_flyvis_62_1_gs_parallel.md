# Parallel Mode Addendum — FlyVis GS (generate + train)

This addendum applies when running in **parallel mode** (GNN_LLM_parallel_flyvis.py). Follow all rules from the base instruction file (`instruction_flyvis_62_1_gs.md`), with these modifications.

## Data Generation

Unlike `flyvis_62_1`, each slot **regenerates data** before training. The simulation seed varies per slot, producing different stochastic realizations. This means:
- Identical configs across slots will produce DIFFERENT results (due to different data)
- Use this to separate **parameter effects** from **data variance**
- When 2+ slots have the same config but different results, the spread indicates stochasticity

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

## Parallel UCB Strategy

When selecting parents for 4 simultaneous mutations, **diversify** your choices:

| Slot | Role | Description |
| ---- | ---- | ----------- |
| 0 | **exploit** | Highest UCB node, conservative mutation |
| 1 | **exploit** | 2nd highest UCB node, or same parent different param |
| 2 | **explore** | Under-visited node, or new parameter dimension |
| 3 | **robustness-check** | Re-run a previous config UNCHANGED to measure data variance |

### Slot 3: Robustness Checking

Since data is regenerated, slot 3 should periodically **re-run the best config unchanged** to build a variance estimate:

1. Pick the current best-performing config (from UCB or memory)
2. Copy it verbatim to slot 3 (no parameter changes)
3. In the log entry, write: `Mode/Strategy: robustness-check`
4. In the Mutation line: `Mutation: [none] — robustness re-run of Node X`
5. After results, compute the spread vs the original run
6. Update the "Variance Estimates" section in memory.md

After the first few batches establish a variance baseline, slot 3 can switch to **principle-test** mode (testing Established Principles from the 62_1 exploration).

## Start Call (first batch, no results yet)

When the prompt says `PARALLEL START`:
- Read the base config to understand the current gs parameters
- Create 4 variations to establish a baseline:
  - Slot 0: gs defaults unchanged (baseline)
  - Slot 1: gs defaults unchanged (variance measurement)
  - Slot 2: gs defaults unchanged (variance measurement)
  - Slot 3: gs defaults unchanged (variance measurement)
- This first batch establishes the **variance baseline** for regenerated data
- Write the planned initial variations to the working memory file

## Logging Format

Same as base instructions, but you write 4 entries per batch:

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [strategy]
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_edge_diff=A, coeff_W_L1=B, batch_size=C, hidden_dim=D, recurrent=[T/F]
Metrics: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, test_pearson=F, training_time_min=G
Embedding: [visual observation, e.g., "65 types partially separated"]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line — note if variance could explain result]
Next: parent=P
```

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder. Always include the exact parameter change (e.g., `Mutation: lr_W: 6E-4 -> 5E-4`). For robustness-check slots, use `Mutation: [none] — robustness re-run of Node X`.

**CRITICAL**: The `Next: parent=P` line selects the parent for the **next batch's** mutations. `P` must refer to a node from a **previous** batch or the current batch — but NEVER set `Next: parent=P` where P is `id+1` (the next slot in the same batch).

Write all 4 entries before editing the 4 config files for the next batch.

## Block Boundaries

- At block boundaries, training parameters can differ across the 4 slots
- Simulation parameters are FIXED
- Block end is detected when any iteration in the batch hits `n_iter_block`

## Failed Slots

If a slot is marked `[FAILED]` in the prompt:
- Write a brief `## Iter N: failed` entry noting the failure
- Still propose a mutation for that slot's config in the next batch
- With regenerated data, failures may be stochastic — do not draw strong conclusions from a single failure

## Training Time Monitoring

Check `training_time_min` for every successful slot. Data generation adds ~5 minutes. If any slot exceeds 60 minutes total:
- Flag it in the observation
- Reduce complexity in the next mutation (smaller hidden_dim, lower data_augmentation_loop)
- Do NOT propose architecture changes that would increase training time beyond the limit
