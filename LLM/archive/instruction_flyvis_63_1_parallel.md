# Parallel Mode Addendum — FlyVis (flyvis_63_1, learned visual field)

This addendum applies when running in **parallel mode** (GNN_LLM_parallel_flyvis.py). Follow all rules from the base instruction file (`instruction_flyvis_63_1.md`), with these modifications.

## Batch Processing

- You receive **4 results per batch** and must propose **4 mutations**
- Each slot has its own config file, metrics log, and activity image
- Write **4 separate** `## Iter N:` log entries (one per slot/iteration)
- Each iteration gets its own Node id in the UCB tree

## Config Files

- Edit all 4 config files listed in the prompt: `{name}_00.yaml` through `{name}_03.yaml`
- Each config's `dataset` field is pre-set to route data to separate directories — **DO NOT change the `dataset` field**
- Only modify `training:` and `graph_model:` parameters (and `claude:` where allowed)
- **DO NOT change `simulation:` parameters** — data is pre-generated

## Parallel UCB Strategy

When selecting parents for 4 simultaneous mutations, **diversify** your choices:

| Slot | Role | Description |
| ---- | ---- | ----------- |
| 0 | **exploit** | Highest UCB node, conservative mutation |
| 1 | **exploit** | 2nd highest UCB node, or same parent different param |
| 2 | **explore** | Under-visited node, or new parameter dimension |
| 3 | **principle-test** | Randomly pick one Established Principle from `memory.md` and design an experiment that tests or challenges it (see below) |

You may deviate from this split based on context (e.g., all exploit if early in block, all boundary-probe if everything converges).

### Slot 3: Principle Testing

At each batch, slot 3 should be used to **validate or challenge** one of the Established Principles listed in the working memory (`{config}_memory.md`):

1. Read the "Established Principles" section in memory.md
2. **Randomly select one principle** (rotate through them across batches — do not repeat the same one consecutively)
3. Design a config that specifically tests this principle:
   - If the principle says "X works when Y", test it under a different condition
   - If the principle says "Z always fails", try to make Z succeed
   - If the principle gives a range, test at the boundary
4. In the log entry, write: `Mode/Strategy: principle-test`
5. In the Mutation line, include: `Testing principle: "[quoted principle text]"`
6. After results, update the principle's evidence level in memory.md:
   - Confirmed → keep in Established Principles
   - Contradicted → move to Open Questions with note

If there are no Established Principles yet (early in the experiment), use slot 3 as a **boundary-probe** instead.

## Start Call (first batch, no results yet)

When the prompt says `PARALLEL START`:
- Read the base config to understand the training regime
- Create 4 diverse initial training parameter variations
- Suggested spread for Block 1 (Siren LR + batch baseline) — 2×2 factorial design:
  - Slot 0: **62_1-optimized LRs** (lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, edge_norm=1.0, batch=2, data_aug=20) + lr_siren=1E-8
  - Slot 1: **62_1-optimized LRs** (same GNN params as slot 0) + lr_siren=1E-5
  - Slot 2: **Original LRs** (lr_W=1E-3, lr=5E-4, lr_emb=1E-3, edge_norm=1000, batch=16, data_aug=25) + lr_siren=1E-8
  - Slot 3: **Original LRs** (same GNN params as slot 2) + lr_siren=1E-5
- This 2×2 design (LR regime × lr_siren) isolates the effect of Siren LR AND batch/LR regime simultaneously
- All 4 slots share the same simulation parameters (pre-generated data)
- Write the planned initial variations to the working memory file

## Logging Format

Same as base instructions, but you write 4 entries per batch:

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [strategy]
Config: lr_W=X, lr=Y, lr_emb=Z, lr_siren=S, coeff_g_phi_diff=A, coeff_W_L1=B, batch_size=C, hidden_dim=D, hidden_dim_nnr_f=E, omega_f=F, recurrent=[T/F]
Metrics: connectivity_R2=A, field_R2=B, tau_R2=C, V_rest_R2=D, cluster_accuracy=E, test_R2=F, test_pearson=G, training_time_min=H
Embedding: [visual observation, e.g., "65 types partially separated"]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder. Always include the exact parameter change (e.g., `Mutation: lr_siren: 1E-8 -> 1E-6`). For principle-test slots, append the principle being tested (e.g., `Mutation: omega_f: 4096 -> 80. Testing principle: "lower omega_f stabilizes W recovery"`).

**CRITICAL**: The `Next: parent=P` line selects the parent for the **next batch's** mutations. `P` must refer to a node from a **previous** batch or the current batch — but NEVER set `Next: parent=P` where P is `id+1` (the next slot in the same batch). This would make a node its own parent and create a circular reference.

Write all 4 entries before editing the 4 config files for the next batch.

## Block Boundaries

- At block boundaries, training parameters can differ across the 4 slots
- Simulation parameters are FIXED (pre-generated data)
- Block end is detected when any iteration in the batch hits `n_iter_block`

## Failed Slots

If a slot is marked `[FAILED]` in the prompt:
- Write a brief `## Iter N: failed` entry noting the failure
- Still propose a mutation for that slot's config in the next batch
- Do not draw conclusions from a single failure (may be stochastic)

## Training Time Monitoring

Check `training_time_min` for every successful slot. If any slot exceeds 60 minutes:
- Flag it in the observation
- Reduce complexity in the next mutation (smaller hidden_dim_nnr_f, lower data_augmentation_loop, smaller batch_size)
- Do NOT propose architecture changes that would increase training time beyond the limit

## Siren-Specific Parallel Strategies

When exploring Siren architecture (Block 3), diversify across dimensions:
- Slot 0: `hidden_dim_nnr_f` variation
- Slot 1: `omega_f` variation
- Slot 2: `n_layers_nnr_f` variation
- Slot 3: `nnr_f_T_period` or `nnr_f_xy_period` variation

When exploring batch_size × LRs (Block 2), ensure each batch covers multiple batch_size values:
- Slot 0: exploit best batch_size from Block 1, vary lr_W
- Slot 1: exploit same batch_size, vary lr or lr_emb
- Slot 2: explore different batch_size with best LRs from Block 1
- Slot 3: principle-test — challenge a batch_size finding from 62_1 (e.g., test batch=4 with Siren present)
