# Parallel Mode Addendum — V_rest Recovery (generate + train)

This addendum applies when running in **parallel mode** (GNN_LLM_parallel_flyvis.py). Follow all rules from the base instruction file (`instruction_flyvis_62_1_gs_vrest.md`), with these modifications.

## Data Generation

Each slot **regenerates data** with a unique seed. V_rest variance is the thing we're trying to reduce, so:
- Identical configs across slots WILL produce different V_rest results
- Improvements must be validated with 3+ samples to distinguish from variance
- Track V_rest mean AND std for promising configs

## Batch Processing

- You receive **4 results per batch** and must propose **4 mutations**
- Write **4 separate** `## Iter N:` log entries (one per slot/iteration)

## Config Files

- Edit all 4 config files: `{name}_00.yaml` through `{name}_03.yaml`
- Each config's `dataset` field routes to a separate data directory — **DO NOT change `dataset`**
- Only modify training-scheme parameters (recurrent, lr_update, W_L2, grad_clip_W, time_step, n_epochs)
- **DO NOT change locked parameters** (see base instruction)

## Parallel UCB Strategy

| Slot | Role | Description |
| ---- | ---- | ----------- |
| 0 | **exploit** | Highest UCB node, conservative mutation |
| 1 | **exploit** | 2nd highest UCB, or same parent different param |
| 2 | **explore** | Under-visited node, or new parameter dimension |
| 3 | **robustness** | Re-run a promising config UNCHANGED for variance measurement |

### Slot 3: Robustness Is Critical

V_rest variance is the core problem. Slot 3 must always re-run the current best config to build variance estimates:

1. Pick the best V_rest config from UCB or memory
2. Copy it verbatim to slot 3
3. Log: `Mode/Strategy: robustness-check`
4. Log: `Mutation: [none] — robustness re-run of Node X`
5. After 3+ samples, compute V_rest mean/std and compare to Node 21 baseline (mean=0.499, std=0.216)

**A config is only better than Node 21 if its V_rest mean (3+ samples) > 0.55 with conn_R2 mean > 0.8.**

## Start Call (first batch, no results yet)

When the prompt says `PARALLEL START`:
- Read the base config (Node 21 optimal)
- First batch establishes baseline and tests first hypotheses:
  - Slot 0: Node 21 defaults (baseline sample)
  - Slot 1: lr_update=1.8E-3 (first lr_update test)
  - Slot 2: lr_update=2.5E-3 (aggressive lr_update test)
  - Slot 3: Node 21 defaults (2nd baseline sample)

## Logging Format

```
## Iter N: [V_rest success/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [strategy]
Config: lr_update=X, recurrent=[T/F], time_step=S, W_L2=W, grad_clip_W=G
Metrics: V_rest_R2=A, connectivity_R2=B, tau_R2=C, cluster_accuracy=D, test_pearson=E, training_time_min=F
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [compare V_rest to Node 21 range 0.11-0.78]
Next: parent=P
```

**CRITICAL**: `Mutation:` line is parsed by UCB tree builder. Always include exact parameter change.

**CRITICAL**: `Next: parent=P` — P must refer to a previous or current batch node, NEVER id+1.

Write all 4 entries before editing the 4 config files.

## Block Boundaries

- At block boundaries, all parameters can differ across slots
- Locked parameters remain locked across all blocks

## Failed Slots

- Write `## Iter N: failed` entry
- With V_rest-focused training, recurrent configs may crash or timeout — note which and avoid
- If training_time_min > 55, flag it and reduce time_step in next mutation

## Training Time Monitoring

- Standard (no recurrent): ~28 min + 5 min data gen = ~33 min
- Recurrent time_step=2: ~50 min + 5 min = ~55 min (close to limit)
- Recurrent time_step=3: ~65 min — likely OVER LIMIT, avoid unless n_epochs=1
- If any slot exceeds 60 min, flag and reduce complexity
