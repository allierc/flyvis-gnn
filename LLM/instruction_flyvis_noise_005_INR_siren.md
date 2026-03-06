# SIREN INR Training Exploration — flyvis_noise_005

## Goal

Optimize **SIREN (siren_txy) hyperparameters** for learning the **stimulus field** of the Drosophila visual system.
Find configs that achieves **final_r2 > 0.95** across frame counts [10, 100, 1000, 10000, 20000, 40000, 64000].

Data is **pre-generated** (stimulus.zarr) — the same data is used for all slots.
Primary metric: **final_r2** (R² between predicted and ground-truth stimulus field).
Secondary metrics: **final_mse**, **training_time_min**, **total_params**, **rank_90**, **rank_99**.

## Scientific Method

This exploration follows a strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Based on available data, form a hypothesis about which parameters control R² (e.g., "Increasing omega_f from 80 to 512 will improve R² because the stimulus field has high-frequency spatial detail")
2. **Design experiment**: Choose a mutation that specifically tests the hypothesis — vary ONE parameter at a time per slot comparison
3. **Run training**: 4 slots test 4 different configs in parallel
4. **Analyze results**: Compare R² across slots to evaluate the hypothesis
5. **Update understanding**: Revise hypotheses based on evidence

**CRITICAL**: You can only hypothesize. Only training results can validate or falsify. Never assume a hypothesis is correct without experimental evidence.

**Evidence hierarchy:**

| Level            | Criterion                       | Action                 |
| ---------------- | ------------------------------- | ---------------------- |
| **Established**  | Consistent across 3+ iterations | Add to Principles      |
| **Tentative**    | Observed 1-2 times              | Add to Open Questions  |
| **Contradicted** | Conflicting evidence            | Note in Open Questions |

## CRITICAL: Shared Dataset, Different Configs

All 4 slots share the same stimulus.zarr data (pre-generated, fixed).
Each slot tests a DIFFERENT hyperparameter configuration.
Within each block, n_training_frames is FIXED by the pipeline — you CANNOT change it.

## SIREN Model (siren_txy)

**Input**: (t, x, y) — normalized time and neuron spatial coordinates
**Output**: scalar stimulus value per neuron per frame

The SIREN uses sine activations: `sin(omega_0 * (W @ x + b))`.

Architecture: First SineLayer → N hidden SineLayers → Linear output layer.

**Key parameter interactions:**

- **omega_f** controls the frequency of sine activations. Higher omega = finer detail but harder to optimize. Weight initialization scales as `1/omega`, so high omega → small initial weights → need lower learning rate.
- **hidden_dim_nnr_f** controls network width. Total params ≈ `hidden_dim² × n_layers`. Larger = more expressive but slower.
- **learning_rate_NNR_f** must be tuned jointly with omega_f. Rule of thumb: higher omega needs lower LR.
- **nnr_f_T_period** normalizes the time coordinate: `t_norm = t / (T_period / 2π)`. Should roughly match n_training_frames to keep input values in a reasonable range.
- **inr_batch_size** = number of frames sampled per training step. For siren_txy, each frame expands to n_neurons input points, so memory scales as batch_size × n_neurons.

## Architecture Parameters (explorable)

| Parameter          | Config section | Default | Description                        |
| ------------------ | -------------- | ------- | ---------------------------------- |
| `hidden_dim_nnr_f` | graph_model    | 128     | Hidden layer width                 |
| `n_layers_nnr_f`   | graph_model    | 5       | Number of hidden SineLayer blocks  |
| `omega_f`          | graph_model    | 80      | Base frequency for sin activations |
| `nnr_f_T_period`   | graph_model    | 1.0     | Temporal normalization period      |
| `nnr_f_xy_period`  | graph_model    | 1.0     | Spatial normalization period       |

## Training Parameters (explorable)

| Parameter             | Config section | Default | Description                        |
| --------------------- | -------------- | ------- | ---------------------------------- |
| `learning_rate_NNR_f` | training       | 1e-4    | Learning rate for SIREN parameters |
| `inr_batch_size`      | training       | 8       | Frames per batch                   |
| `total_steps`         | claude         | varies  | Total training iterations          |

## Block Structure

Each block has a FIXED n_training_frames. Blocks progress through increasing data sizes:

| Block | n_training_frames | Default total_steps | Suggested focus                                      |
| ----- | ----------------- | ------------------- | ---------------------------------------------------- |
| 1     | 10                | 5000                | Quick sweep: omega, LR, hidden_dim (fast iteration)  |
| 2     | 100               | 10000               | Scale up: adjust LR and T_period for more data       |
| 3     | 1000              | 20000               | Medium data: refine architecture, test batch_size    |
| 4     | 10000             | 30000               | Large data: stability, convergence, time constraints |
| 5     | 20000             | 40000               | Scale: memory limits, training time budget           |
| 6     | 40000             | 50000               | Near-full: fine-tuning for large data                |
| 7     | 64000             | 60000               | Full data: validate best config, final optimization  |

**At block boundaries (when n_training_frames changes):**

- The pipeline carries forward the best config from the previous block
- `nnr_f_T_period` is auto-set to the new n_training_frames
- UCB scores are cleared (fresh exploration tree)
- Adapt the config for the new frame count:
  - T_period should roughly match n_training_frames
  - LR may need to decrease for more data
  - total_steps may need to increase
  - batch_size may need adjustment for memory

## Parallel Mode — 4 Slots Per Batch (DIFFERENT CONFIGS)

Each slot tests a DIFFERENT configuration. This is NOT seed robustness testing.

### Slot Strategy

| Slot | Role        | Description             |
| ---- | ----------- | ----------------------- |
| 0    | **explore** | Test config variation A |
| 1    | **explore** | Test config variation B |
| 2    | **explore** | Test config variation C |
| 3    | **explore** | Test config variation D |

Design the 4 configs to efficiently explore the parameter space. Good strategies:

- **Grid search**: Vary one parameter across 4 values (e.g., omega_f = 64, 256, 1024, 4096)
- **Latin square**: Vary 2 parameters, each at 2 levels
- **Focused**: 1 slot = best known config, 3 slots = variations around it

### Config Files

Edit all 4 config files: `{name}_00.yaml` through `{name}_03.yaml`.
Each config should have DIFFERENT hyperparameters.
Only modify:

- `graph_model:` section: hidden_dim_nnr_f, n_layers_nnr_f, omega_f, nnr_f_T_period, nnr_f_xy_period
- `training:` section: learning_rate_NNR_f, inr_batch_size
- `claude:` section: total_steps

DO NOT change: n_training_frames (forced by pipeline), dataset, simulation parameters, inr_type.

## Training Time Constraint

Training time must be ≤ 60 min per slot. Monitor `training_time_min` in results.

Factors that increase training time:

- Larger `hidden_dim_nnr_f` (quadratic in params)
- More `n_layers_nnr_f`
- Larger `inr_batch_size` (more memory, potentially slower per step)
- More `total_steps`
- More `n_training_frames` (for siren_txy, each frame = n_neurons forward passes)

## Iteration Workflow (every batch)

### Step 1: Read Working Memory + User Input

- Read `{llm_task_name}_memory.md` for context
- Read `user_input.md` for pending instructions

### Step 2: Analyze Results (4 slots, 4 different configs)

**Metrics from analysis log:**

- `final_r2`: R² of predicted vs ground-truth field (PRIMARY)
- `final_mse`: Mean squared error
- `training_time_min`: Training duration
- `total_params`: Number of model parameters
- `rank_90` / `rank_99`: Effective rank of data (SVD analysis)

**Per-slot classification:**

- **Excellent**: final_r2 > 0.95
- **Good**: final_r2 0.8–0.95
- **Moderate**: final_r2 0.5–0.8
- **Poor**: final_r2 < 0.5

### Step 3: Write Log Entries + Update Memory

**3a. Append to Full Log** (`{llm_task_name}_analysis.md`):

```
## Iter N: [excellent/good/moderate/poor]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: hidden_dim=X, n_layers=Y, omega=Z, lr=W, batch_size=B, T_period=T, total_steps=S, n_frames=F
Metrics: final_r2=A, final_mse=B, training_time_min=C, total_params=D, rank_90=E, rank_99=F
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive]
Observation: [one line]
Next: parent=P
```

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder — always include exact parameter change.
**CRITICAL**: `Next: parent=P` — P must be from a previous or current batch, NEVER `id+1`.

**3b. Update R2 Progression Table in memory.md**

**3c. Update Hypotheses in memory.md**

### Step 4: Acknowledge User Input (if any)

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

1. Based on results, formulate the next hypothesis
2. Design 4 configs that test the hypothesis (or explore different aspects)
3. Write the hypothesis to memory.md before editing configs

## Block Boundaries

At the end of each block (>>> BLOCK END <<< marker):

1. Update "Paper Summary" in memory.md
2. Summarize findings in "Previous Block Summary"
3. Update "Established Principles" (require 3+ supporting iterations)
4. Clear "Current Block" for next block
5. Note what should change for the next frame count

## Known Results

- SIREN txy with hidden_dim=2048, omega=2048, lr=1e-8, 1000 frames: R²~0.85
- T_period should approximately match n_training_frames
- omega_f is the most critical parameter — controls frequency resolution
- Higher omega needs lower LR (gradients scale with omega)
- For stimulus field: n_input_neurons=1736, effective rank ~12 at 90% variance
- Small frame counts (10-100) are easy — good for fast parameter sweeps
- Large frame counts (40k-64k) stress training time budget

## Start Call

When prompt says `PARALLEL START`:

- Read base config to understand current parameter values
- Create 4 DIFFERENT configs exploring the parameter space
- Good first exploration: vary omega_f across 4 values while keeping other params constant
- Write planned configs to working memory

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005 INR

## Paper Summary

- **INR optimization**: [pending first results]
- **LLM-driven exploration**: [pending first results]

## Knowledge Base

### R2 Progression Table

| Iter | n_frames | hidden_dim | n_layers | omega | lr  | batch_sz | T_period | steps | final_r2 | final_mse | time_min | Hypothesis |
| ---- | -------- | ---------- | -------- | ----- | --- | -------- | -------- | ----- | -------- | --------- | -------- | ---------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

---

## Previous Block Summary

---

## Current Block (Block N)

### Block Info

Focus: [parameter subspace]
n_training_frames: [fixed value]

### Hypothesis

**Hypothesis**: [specific, testable prediction]
**Rationale**: [why you believe this]
**Test**: [what configs test this]
**Expected outcome**: [what would support vs falsify]
**Status**: untested / supported / falsified / revised

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
