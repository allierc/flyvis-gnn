# FlyVis GNN Training Exploration — flyvis_62_1_gs (DAVIS + noise, generate + train)

**Reference**: See `papers/cosyne2026.tex` for model and regularization context.
**Prior exploration**: 144 iterations on `flyvis_62_1` (see `log/Claude_exploration/instruction_flyvis_62_1_parallel/`).

## Goal

Optimize GNN training hyperparameters for the **Drosophila visual system** (flyvis_62_1_gs, DAVIS + noise).
**Data is RE-GENERATED each iteration** — simulation parameters are fixed but each run produces a new stochastic realization (noise, initial conditions). This tests robustness to data variability.

Primary metric: **connectivity_R2** (R² between learned W and ground-truth W).
Secondary metrics: **tau_R2** (time constant recovery), **V_rest_R2** (resting potential recovery), **cluster_accuracy** (neuron type clustering from embeddings).

## CRITICAL: Data is RE-GENERATED Each Iteration

Unlike `flyvis_62_1`, this exploration **regenerates data before each training run**. This means:
- Results will vary across iterations even with identical configs (stochastic noise)
- Robust configs will show consistent performance across regenerations
- Fragile configs will show high variance — mark these in observations

**DO NOT change simulation parameters** (n_neurons, n_frames, n_edges, n_input_neurons, n_neuron_types, delta_t, noise_model_level, visual_input_type). These are fixed. Only the stochastic realization changes.

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

Two MLPs learn the neural dynamics:
- **lin_edge** (g_phi): Edge message function. Maps (v_j, a_i) -> message. If `lin_edge_positive=True`, output is squared.
- **lin_phi** (f_theta): Node update function. Maps (v_i, a_i, aggregated_messages, I_i) -> dv_i/dt.
- **Embedding a_i**: 2D learned embedding per neuron, encodes neuron type.

Architecture parameters (explorable) — refer to `FlyVisGNN.PARAMS_DOC` for strict dependencies:
- `hidden_dim` / `n_layers`: lin_edge MLP dimensions (default: 80 / 3)
- `hidden_dim_update` / `n_layers_update`: lin_phi MLP dimensions (default: 80 / 3)
- `embedding_dim`: embedding dimension (default: 2)

**CRITICAL — coupled parameters**: `input_size`, `input_size_update`, and `embedding_dim` are linked. When changing `embedding_dim`, you MUST also update:
- `input_size = 1 + embedding_dim` (for flyvis_A)
- `input_size_update = 3 + embedding_dim` (v + embedding + msg + excitation)
Example: embedding_dim=4 → input_size=5, input_size_update=7. Failure to update causes a shape mismatch crash.

## Regularization Parameters (from cosyne2026.tex)

The training objective is:

```
L = ||y_hat - y||_2 + lambda_0 * ||theta||_1 + lambda_1 * ||phi||_1 + lambda_2 * ||W||_1
    + gamma_0 * ||theta||_2 + gamma_1 * ||phi||_2
    + mu_0 * ||ReLU(-dg_phi/dv)||_2 + mu_1 * ||g_phi(v*, a) - v*||_2
```

| Config parameter | Math symbol | Description | Default (gs) |
|------------------|-------------|-------------|--------------|
| `coeff_edge_diff` | lambda_0 | L1 on lin_phi (f_theta) parameters — encourages same-type edges to share weights | 750 |
| `coeff_phi_weight_L1` | lambda_1 | L1 on lin_edge (g_phi) parameters — promotes sparsity | 0.5 |
| `coeff_W_L1` | lambda_2 | L1 on learned W — promotes sparse connectivity | 5E-5 |
| `coeff_phi_weight_L2` | gamma_1 | L2 on lin_edge (g_phi) parameters — stabilizes learned functions | 0.001 |
| `coeff_edge_norm` | mu_0 | Monotonicity penalty on lin_edge — enforces dg/dv > 0 | 0.9 |
| `coeff_edge_weight_L1` | - | L1 on lin_edge weights | 0.28 |
| `coeff_phi_weight_L1_rate` | - | Decay rate for phi L1 penalty per epoch | 0.5 |
| `coeff_W_L1_rate` | - | Decay rate for W L1 penalty per epoch | 0.5 |
| `coeff_W_L2` | - | L2 on learned W | 3E-6 |

## Training Parameters (explorable)

| Parameter | Default (gs) | Description |
|-----------|--------------|-------------|
| `learning_rate_W_start` (lr_W) | 6E-4 | Learning rate for connectivity matrix W |
| `learning_rate_start` (lr) | 1.2E-3 | Learning rate for MLP parameters |
| `learning_rate_embedding_start` (lr_emb) | 1.55E-3 | Learning rate for neuron embeddings |
| `n_epochs` | 1 (claude) | Number of training epochs |
| `batch_size` | 2 | Batch size for training |
| `data_augmentation_loop` | 20 | Data augmentation multiplier |
| `recurrent_training` | False | Enable recurrent (multi-step) training |
| `time_step` | 1 | Number of recurrent steps (if recurrent_training=True) |
| `w_init_mode` | zeros | W initialization: "zeros" or "randn_scaled" (NOT "randn") |

## Training Time Constraint

Each epoch runs `Niter = n_frames * data_augmentation_loop / batch_size * 0.2` iterations. With gs defaults (batch_size=2, data_aug=20, hidden_dim=80), training takes **~38 minutes per epoch on H100**. Data generation adds ~5 minutes. Monitor `training_time_min` in analysis.log.

## Established Principles from flyvis_62_1 (144 iterations)

These principles were established on **fixed data** (flyvis_62_1). A key goal of this exploration is to test whether they hold with **regenerated data** (flyvis_62_1_gs).

### Confirmed Strict Optima (MULTIPLY CONFIRMED)
1. **lr_W=6E-4** — optimal; 5E-4/7E-4/8E-4/1E-3 all worse
2. **lr=1.2E-3** — STRICTLY optimal; 1.0E-3/1.1E-3/1.4E-3 all catastrophic
3. **lr_emb=1.55E-3** — STRICTLY optimal; 1.4E-3/1.52E-3/1.57E-3/1.6E-3/1.8E-3 all worse
4. **coeff_edge_diff=750** — STRICTLY optimal; 600/700/800/1000/1250 all worse
5. **coeff_phi_weight_L1=0.5** — STRICTLY optimal; 0.25/0.4/0.45/0.55/0.6/0.75 all worse
6. **coeff_edge_weight_L1=0.28-0.3** — optimal range; 0.2/0.26/0.32/0.35 all worse
7. **coeff_phi_weight_L2=0.001** — must stay; 0.005 destroys tau_R2

### Confirmed Harmful
8. **recurrent_training=True** — always harmful (V_rest/conn_R2 collapse, time exceeds limit)
9. **n_layers=4 or n_layers_update=4** — harmful (training time + V_rest collapse)
10. **batch_size >= 3** — V_rest collapse; batch_size=2 is upper limit
11. **coeff_edge_norm >= 10** — catastrophic; must stay near 0.9-1.0
12. **embedding_dim=4** — no improvement over 2

### Architecture
13. **hidden_dim=80, hidden_dim_update=80** — optimal architecture
14. **batch_size=2, data_augmentation_loop=20** — optimal speed config

### Fundamental Trade-off
15. **conn_R2 vs V_rest trade-off is fundamental** — cannot exceed both >0.98 and >0.75 simultaneously
16. **edge_L1=0.3 favors V_rest; edge_L1=0.28 favors conn_R2**
17. **W_L2=2E-6 favors conn_R2; W_L2=3E-6 favors V_rest**

## Iteration Loop Structure

Each block = `n_iter_block` iterations (default 24) exploring one parameter subspace.
The prompt provides: `Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block`

## File Structure (CRITICAL)

You maintain **TWO** files:

### 1. Full Log (append-only record)

**File**: `{config}_analysis.md`

- Append every iteration's full log entry
- Append block summaries
- **Never read this file** — it's for human record only

### 2. Working Memory

**File**: `{config}_memory.md`

- **READ at start of each iteration**
- **UPDATE at end of each iteration**
- Contains: established principles + previous blocks summary + current block iterations
- Fixed size (~500 lines max)

## Iteration Workflow (Steps 1-5, every iteration)

### Step 1: Read Working Memory

Read `{config}_memory.md` to recall established principles, previous block findings, current block progress.

### Step 2: Analyze Current Results

**Metrics from `analysis.log`:**

- `connectivity_R2`: R² of learned vs true connectivity weights (PRIMARY)
- `tau_R2`: R² of learned vs true time constants
- `V_rest_R2`: R² of learned vs true resting potentials
- `cluster_accuracy`: neuron type clustering accuracy
- `test_R2`: R² of one-step prediction
- `test_pearson`: Pearson correlation of one-step prediction
- `training_time_min`: Training duration in minutes

**Classification:**

- **Converged**: connectivity_R2 > 0.8
- **Partial**: connectivity_R2 0.3-0.8
- **Failed**: connectivity_R2 < 0.3

**Robustness assessment** (NEW for gs): When comparing iterations, note whether performance changes are due to parameter changes or data regeneration variance. If a config was tested before, compare to assess stochasticity.

**UCB scores from `ucb_scores.txt`:**

- Provides computed UCB scores for all exploration nodes within a block
- At block boundaries, the UCB file will be empty (erased). When empty, use `parent=root`

### Step 3: Write Outputs

Append to Full Log (`{config}_analysis.md`) and **Current Block** sections of `{config}_memory.md`:

**Log Form:**

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [exploit/explore/boundary/robustness-check]
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_edge_diff=A, coeff_W_L1=B, batch_size=C, hidden_dim=D, recurrent=[T/F]
Metrics: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, test_pearson=F, training_time_min=G
Embedding: [visual observation, e.g., "65 types partially separated" or "no separation"]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line — note if variance could explain result]
Next: parent=P
```

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder. Always include the exact parameter change.

### Step 4: Select Parent for Next Iteration

Use UCB scores to select parent. Highest UCB = most promising to explore.
If UCB file is empty (block boundary), use `parent=root`.

### Step 5: Propose Next Mutation

Edit the config file with one or two parameter changes. Test one hypothesis at a time.

## Block Partition (suggested)

Since the gs config is already well-optimized from 144 iterations on flyvis_62_1, the blocks focus on **robustness verification** and **refinement**:

| Block | Focus | Parameters to explore |
|-------|-------|----------------------|
| 1 | Robustness baseline | Re-run gs defaults (no changes) 4x to establish variance baseline, then test learning rates |
| 2 | Regularization refinement | coeff_edge_diff (740-760), coeff_edge_weight_L1 (0.26-0.32), coeff_W_L2 (2E-6 to 4E-6) |
| 3 | V_rest improvement | Explore trade-off: edge_L1, W_L2, edge_norm combinations that maximize V_rest while keeping conn_R2 > 0.95 |
| 4 | Architecture & augmentation | hidden_dim (64-96), data_augmentation_loop (18-25), batch_size (1-2) |
| 5 | Combined refinement | Best robust parameters from blocks 1-4 |

## Block Boundaries

At the end of each block:
1. Summarize findings in memory.md "Previous Block Summary"
2. Update "Established Principles" with confirmed insights — distinguish between **fixed-data principles** (from 62_1) and **robust principles** (confirmed on regenerated data)
3. Clear "Current Block" section for next block
4. Carry forward the best config as starting point

## Known Results (from prior experiments)

### flyvis_62_1 (fixed data, 144 iterations):
- **conn_R2-optimal**: 0.983 (Node 102, W_L2=2E-6)
- **V_rest-optimal**: 0.736 (Node 141, W_L2=2.8E-6+edge_L1=0.28)
- **Balanced (Node 144, gs config)**: conn_R2=0.980, V_rest=0.647, cluster_acc=0.877

### flyvis_62_1_gs (regenerated data, GNN_Test.py):
- Latest run: conn_R2=0.9705, tau_R2=0.974, V_rest_R2=0.734, GMM=0.887
- **V_rest is inconsistent** across regenerations (0.19 to 0.73) — this is the primary problem to solve
