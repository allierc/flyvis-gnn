# FlyVis GNN Training Exploration — V_rest Recovery (generate + train)

**Reference**: See `papers/cosyne2026.tex` for model and regularization context.
**Prior exploration**: 56+ iterations on `flyvis_62_1_gs` (see `log/Claude_exploration/instruction_flyvis_62_1_gs_parallel/`).

## Goal

Optimize GNN training specifically for **V_rest recovery** in the Drosophila visual system model.
**Data is RE-GENERATED each iteration** with varying seeds.

Primary metric: **V_rest_R2** (R² of learned vs true resting potentials).
Hard constraint: **connectivity_R2 > 0.8** (must not collapse connectivity recovery).
Secondary metrics: **tau_R2**, **cluster_accuracy**, **test_pearson**.

## The V_rest Problem

From 56 iterations on flyvis_62_1_gs with 8 identical-config samples (Node 21):
- **conn_R2**: 0.955 +/- 0.029 (stable)
- **tau_R2**: 0.975 +/- 0.028 (stable)
- **V_rest_R2**: 0.499 +/- 0.216 (HUGE variance, range 0.11-0.78)
- **cluster_acc**: 0.884 +/- 0.023 (stable)

V_rest is the only metric with unacceptable variance. The root cause is structural:

### Why V_rest Is Hard to Learn

The per-neuron dynamics are:
```
tau_i * dv_i/dt = -(v_i - V_rest_i) + sum_j W_ij * ReLU(v_j) + I_i
```

The MLP `lin_update` learns `f(v) = -(v - V_rest) / tau`, which is a linear function with:
- **slope** = -1/tau (well-constrained: visible in every t->t+1 transition)
- **intercept** = V_rest/tau (poorly constrained: requires extrapolation)

Two problems:
1. **Single-step optimization**: We optimize x(t)->x(t+1). The slow decay term `-(v - V_rest)/tau` produces small per-step changes relative to the fast message passing `W * lin_edge`. Gradients are dominated by msg.
2. **Lever arm problem**: During simulation, v stays in a narrow range driven by inputs and network activity, well above V_rest. The MLP never sees v near V_rest. Small slope errors trade off against large intercept shifts — V_rest is identified only through extrapolation.

### Why tau IS Recovered

tau controls the *rate of change* (slope of f(v)), which is constrained by every data point in the observed v range. V_rest controls *where the decay targets* (y-intercept), which requires extrapolation beyond the observed range.

## Exploration Dimensions

This exploration focuses on **training scheme** changes that could help V_rest. All regularization parameters are locked at Node 21 optimal from the gs exploration.

### 1. Recurrent Training (multi-step rollouts)
- `recurrent_training: true/false`
- `time_step: 1-10` (number of rollout steps)
- `noise_recurrent_level: 0.0-0.1`
- **Hypothesis**: Multi-step rollouts accumulate the slow V_rest signal. Over k steps, the -(v-V_rest)/tau term's contribution grows relative to noise. The MLP sees the trajectory, not just a single step.
- **Risk**: Previously marked "always harmful" in 62_1 exploration — but that exploration optimized for conn_R2, not V_rest. May help V_rest even if conn_R2 drops.
- **Constraint**: Keep training time < 60 min. time_step=2 doubles compute; time_step=5 may exceed budget.

### 2. Learning Rate Balance (lin_update vs lin_edge)
- `learning_rate_update_start`: Separate LR for lin_update MLP (0 = use lr)
- `learning_rate_start` (lr): LR for lin_edge MLP
- `learning_rate_W_start` (lr_W): LR for W matrix
- **Hypothesis**: Higher LR for lin_update relative to lin_edge gives the slow-component MLP more gradient signal. Current lr=1.2E-3 is shared — maybe lin_update needs its own higher rate.
- **Search range**: lr_update from 1.2E-3 to 3E-3 while keeping lr=1.2E-3 fixed.

### 3. W_L2 Regularization Strength
- `coeff_W_L2`: Controls trade-off between conn_R2 and V_rest
- **Known**: W_L2=2.5E-6 optimal for balance; lower favors conn_R2, higher favors V_rest
- **Search range**: 2.5E-6 to 5E-6 (accept conn_R2 drop to >0.8)
- Since primary goal is V_rest, can push W_L2 higher than gs exploration allowed.

### 4. Training Schedule (code changes allowed)
- Two-phase training: first standard, then recurrent
- `recurrent_training_start_epoch: 0` — epoch at which recurrent starts
- **Hypothesis**: Let the MLP first learn the msg structure, then switch to recurrent to refine V_rest.
- Requires `n_epochs >= 2` for this to work.

### 5. Grad Clip on W
- `grad_clip_W: 0.0` (disabled by default)
- **Hypothesis**: Clipping W gradients forces the optimizer to adjust lin_update instead of W to explain V_rest-related dynamics.
- **Search range**: 0.0 (disabled), 0.1, 0.5, 1.0

## LOCKED Parameters (DO NOT CHANGE)

These are confirmed strictly optimal from 56+ iterations on gs data. Changing them will waste iterations.

| Parameter | Locked Value | Reason |
|-----------|-------------|--------|
| lr_W | 6E-4 | STRICTLY optimal |
| lr | 1.2E-3 | STRICTLY optimal |
| lr_emb | 1.55E-3 | STRICTLY optimal |
| coeff_edge_diff | 750 | STRICTLY optimal |
| coeff_phi_weight_L1 | 0.5 | STRICTLY optimal |
| coeff_phi_weight_L1_rate | 0.4 | STRICTLY optimal |
| coeff_edge_weight_L1 | 0.29 | STRICTLY optimal |
| coeff_edge_norm | 0.9 | STRICTLY optimal |
| coeff_phi_weight_L2 | 0.001 | Must stay |
| coeff_W_L1 | 3.5E-5 | Optimal |
| coeff_W_L1_rate | 0.5 | Optimal |
| hidden_dim | 80 | Optimal |
| hidden_dim_update | 80 | Optimal |
| n_layers | 3 | Optimal |
| n_layers_update | 3 | STRICTLY optimal |
| batch_size | 2 | Optimal |
| data_augmentation_loop | 20 | STRICTLY optimal |
| embedding_dim | 2 | Optimal |

**EXCEPTION**: `lr` and `lr_W` may be explored ONLY in combination with recurrent_training or lr_update changes. Do not test them alone — they are confirmed optimal for standard training.

## CRITICAL: Data is RE-GENERATED Each Iteration

Each iteration gets a different seed -> different data. V_rest variance is ~0.22 std. This means:
- A single good V_rest result may be a lucky seed, not a real improvement
- Confirm improvements with robustness re-runs (slot 3 in parallel mode)
- Report both the metric value AND how it compares to Node 21 variance range (0.11-0.78)

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i(t)/dt = -v_i(t) + V_i^rest + sum_j W_ij * ReLU(v_j(t)) + I_i(t)
```

- 13,741 neurons, 65 cell types, 434,112 edges
- 1,736 input neurons (photoreceptors)
- DAVIS visual input, noise_model_level=0.05
- 64,000 frames, delta_t=0.02

## GNN Architecture

- **lin_edge** (g_phi): Edge message function — learns W_ij * ReLU(v_j) component
- **lin_update** (f_theta): Node update function — learns -(v - V_rest)/tau component
- **Embedding a_i**: 2D learned embedding per neuron

## Training Time Constraint

Standard training: ~28 min per epoch on H100. With recurrent (time_step=2): ~45-50 min. Data generation adds ~5 min. **Hard limit: 60 min total.** If training_time_min > 55, reduce time_step or other complexity.

## Iteration Loop Structure

Each block = `n_iter_block` iterations (default 24).

## File Structure (CRITICAL)

### 1. Full Log (append-only): `{config}_analysis.md`
### 2. Working Memory (read+update each iter): `{config}_memory.md`

## Iteration Workflow (Steps 1-5, every iteration)

### Step 1: Read Working Memory
Read `{config}_memory.md`.

### Step 2: Analyze Current Results

**Metrics from `analysis.log`** — same as gs exploration but with V_rest-focused classification:

- **V_rest success**: V_rest_R2 > 0.6 AND connectivity_R2 > 0.8
- **V_rest partial**: V_rest_R2 0.3-0.6 AND connectivity_R2 > 0.8
- **V_rest failed**: V_rest_R2 < 0.3 OR connectivity_R2 < 0.8

### Step 3: Write Outputs

**Log Form:**

```
## Iter N: [V_rest success/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [exploit/explore/robustness-check]
Config: lr_update=X, recurrent=[T/F], time_step=S, W_L2=W, grad_clip_W=G
Metrics: V_rest_R2=A, connectivity_R2=B, tau_R2=C, cluster_accuracy=D, test_pearson=E, training_time_min=F
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line — compare V_rest to Node 21 range 0.11-0.78, note if within/above variance]
Next: parent=P
```

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder.

### Step 4: Select Parent (use UCB scores)
### Step 5: Propose Next Mutation (one or two parameter changes)

## Block Partition (suggested)

| Block | Focus | Parameters |
|-------|-------|-----------|
| 1 | Baseline + lr_update | Establish Node 21 V_rest variance (4 runs), then test lr_update 1.5E-3 to 3E-3 |
| 2 | Recurrent training | recurrent_training=True with time_step=2,3,5. Test with and without lr_update boost |
| 3 | W_L2 push | coeff_W_L2 from 3E-6 to 5E-6. Accept conn_R2 > 0.8 |
| 4 | Grad clip + combinations | grad_clip_W 0.1-1.0, combine best lr_update + recurrent + W_L2 |
| 5 | Two-phase training | n_epochs=2 with recurrent_training_start_epoch=1 (standard then recurrent) |

## Block Boundaries

At the end of each block:
1. Summarize findings in memory.md
2. Update "Established Principles" — distinguish between V_rest-helpful and V_rest-neutral
3. Record V_rest mean and std for the best config (need 3+ samples)
4. Carry forward the best V_rest config as starting point

## Known Results — Node 21 Baseline (8 samples)

| Sample | conn_R2 | tau_R2 | V_rest_R2 | cluster_acc |
|--------|---------|--------|-----------|-------------|
| Iter 21 | 0.983 | 0.992 | 0.704 | 0.917 |
| Iter 28 | 0.949 | 0.975 | 0.381 | 0.874 |
| Iter 32 | 0.974 | 0.992 | 0.575 | 0.897 |
| Iter 36 | 0.937 | 0.986 | 0.462 | 0.874 |
| Iter 40 | 0.983 | 0.996 | 0.775 | 0.902 |
| Iter 44 | 0.955 | 0.909 | 0.112 | 0.892 |
| Iter 53 | 0.896 | 0.980 | 0.627 | 0.871 |
| Iter 56 | 0.959 | 0.972 | 0.356 | 0.842 |
| **Mean** | **0.955** | **0.975** | **0.499** | **0.884** |
| **Std** | **0.029** | **0.028** | **0.216** | **0.023** |

Any config that achieves V_rest_R2 mean > 0.55 (with 3+ samples) while keeping conn_R2 > 0.8 is an improvement.
