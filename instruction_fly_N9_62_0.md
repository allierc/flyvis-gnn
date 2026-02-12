# FlyVis GNN Training Exploration — fly_N9_62_0 (no noise)

**Reference**: See `papers/cosyne2026.tex` for model and regularization context.

## Goal

Optimize GNN training hyperparameters for the **Drosophila visual system** (fly_N9_62_0, no noise).
The data is **pre-generated** — do NOT modify simulation parameters.

Primary metric: **connectivity_R2** (R² between learned W and ground-truth W).
Secondary metrics: **tau_R2** (time constant recovery), **V_rest_R2** (resting potential recovery), **cluster_accuracy** (neuron type clustering from embeddings).

## CRITICAL: Data is PRE-GENERATED

**DO NOT change any simulation parameters** (n_neurons, n_frames, n_edges, n_input_neurons, n_neuron_types, delta_t, noise_model_level, visual_input_type). The data lives in `graphs_data/fly/fly_N9_62_0/` and is fixed.

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i(t)/dt = -v_i(t) + V_i^rest + sum_j W_ij * ReLU(v_j(t)) + I_i(t)
```

- 13,741 neurons, 65 cell types, 434,112 edges
- 1,736 input neurons (photoreceptors)
- DAVIS visual input, no noise (noise_model_level=0.0)
- 64,000 frames, delta_t=0.02

## GNN Architecture

Two MLPs learn the neural dynamics:
- **lin_edge** (g_phi): Edge message function. Maps (v_j, a_i) -> message. If `lin_edge_positive=True`, output is squared.
- **lin_phi** (f_theta): Node update function. Maps (v_i, a_i, aggregated_messages, I_i) -> dv_i/dt.
- **Embedding a_i**: 2D learned embedding per neuron, encodes neuron type.

Architecture parameters (explorable):
- `hidden_dim` / `n_layers`: lin_edge MLP dimensions (default: 64 / 3)
- `hidden_dim_update` / `n_layers_update`: lin_phi MLP dimensions (default: 64 / 3)
- `embedding_dim`: embedding dimension (default: 2)

## Regularization Parameters (from cosyne2026.tex)

The training objective is:

```
L = ||y_hat - y||_2 + lambda_0 * ||theta||_1 + lambda_1 * ||phi||_1 + lambda_2 * ||W||_1
    + gamma_0 * ||theta||_2 + gamma_1 * ||phi||_2
    + mu_0 * ||ReLU(-dg_phi/dv)||_2 + mu_1 * ||g_phi(v*, a) - v*||_2
```

| Config parameter | Math symbol | Description | Default |
|------------------|-------------|-------------|---------|
| `coeff_edge_diff` | lambda_0 | L1 on lin_phi (f_theta) parameters — encourages same-type edges to share weights | 500 |
| `coeff_phi_weight_L1` | lambda_1 | L1 on lin_edge (g_phi) parameters — promotes sparsity | 1 |
| `coeff_W_L1` | lambda_2 | L1 on learned W — promotes sparse connectivity | 5E-5 |
| `coeff_phi_weight_L2` | gamma_1 | L2 on lin_edge (g_phi) parameters — stabilizes learned functions | 0.001 |
| `coeff_edge_norm` | mu_0 | Monotonicity penalty on lin_edge — enforces dg/dv > 0 | 1000 |
| `coeff_edge_weight_L1` | - | L1 on lin_edge weights | 1 |
| `coeff_phi_weight_L1_rate` | - | Decay rate for phi L1 penalty per epoch | 0.5 |
| `coeff_W_L1_rate` | - | Decay rate for W L1 penalty per epoch | 0.5 |
| `coeff_W_L2` | - | L2 on learned W (not in base config, add if needed) | 0 |

## Training Parameters (explorable)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate_W_start` (lr_W) | 1E-3 | Learning rate for connectivity matrix W |
| `learning_rate_start` (lr) | 5E-4 | Learning rate for MLP parameters |
| `learning_rate_embedding_start` (lr_emb) | 1E-3 | Learning rate for neuron embeddings |
| `n_epochs` | 1 (claude) | Number of training epochs |
| `batch_size` | 1 | Batch size for training |
| `data_augmentation_loop` | 25 | Data augmentation multiplier |
| `recurrent_training` | False | Enable recurrent (multi-step) training |
| `time_step` | 1 | Number of recurrent steps (if recurrent_training=True) |
| `w_init_mode` | zeros | W initialization: "zeros" or "randn_scaled" (NOT "randn") |

## Training Time Constraint

Each epoch runs `Niter = n_frames * data_augmentation_loop / batch_size * 0.2` iterations (baseline: 320,000 at ~72 it/s). The baseline (batch_size=1, 64K frames, hidden_dim=64) takes **~45 minutes per epoch on H100** (~75 minutes on A100). With `n_epochs=1`, total training time is ~45 minutes on H100. Monitor `training_time_min` in analysis.log.

Factors that increase training time:
- Larger `hidden_dim` / `n_layers`
- Larger `data_augmentation_loop`
- `recurrent_training=True` with large `time_step`

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

**UCB scores from `ucb_scores.txt`:**

- Provides computed UCB scores for all exploration nodes within a block
- At block boundaries, the UCB file will be empty (erased). When empty, use `parent=root`

### Step 3: Write Outputs

Append to Full Log (`{config}_analysis.md`) and **Current Block** sections of `{config}_memory.md`:

**Log Form:**

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [exploit/explore/boundary]
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_edge_diff=A, coeff_W_L1=B, batch_size=C, hidden_dim=D, recurrent=[T/F]
Metrics: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, test_pearson=F, training_time_min=G
Embedding: [visual observation, e.g., "65 types partially separated" or "no separation"]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder. Always include the exact parameter change.

### Step 4: Select Parent for Next Iteration

Use UCB scores to select parent. Highest UCB = most promising to explore.
If UCB file is empty (block boundary), use `parent=root`.

### Step 5: Propose Next Mutation

Edit the config file with one or two parameter changes. Test one hypothesis at a time.

## Block Partition (suggested)

| Block | Focus | Parameters to explore |
|-------|-------|----------------------|
| 1 | Learning rates | lr_W, lr, lr_emb |
| 2 | Regularization | coeff_edge_diff, coeff_edge_norm, coeff_edge_weight_L1, coeff_phi_weight_L1, coeff_phi_weight_L2, coeff_W_L1 |
| 3 | Architecture | hidden_dim, n_layers, hidden_dim_update, n_layers_update, embedding_dim |
| 4 | Batch & augmentation | batch_size, data_augmentation_loop |
| 5 | Recurrent training | recurrent_training, time_step, coeff_W_L2 |
| 6 | Combined best | Best parameters from blocks 1-5 |

## Block Boundaries

At the end of each block:
1. Summarize findings in memory.md "Previous Block Summary"
2. Update "Established Principles" with confirmed insights
3. Clear "Current Block" section for next block
4. Carry forward the best config as starting point

## Known Results (from prior experiments)

- `fly_N9_62_1` (DAVIS + noise 0.05): connectivity_R2=0.95, tau_R2=0.80, V_rest_R2=0.40 (with full regularization, 10 epochs)
- `fly_N9_62_1` with noise + Sintel: R²_W=0.99, tau_R2=1.00, V_rest_R2=0.85
- W initialization: `zeros` and `randn_scaled` perform equally well; plain `randn` performs poorly
- Bigger MLP (64-dim/3-layer) works better than smaller (32-dim/2-layer)
