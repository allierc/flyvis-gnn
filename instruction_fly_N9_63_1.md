# FlyVis GNN Training Exploration — fly_N9_63_1 (DAVIS + noise + learned visual field)

**Reference**: See `papers/cosyne2026.tex` for model and regularization context.

## Goal

Optimize GNN training hyperparameters for the **Drosophila visual system** (fly_N9_63_1, DAVIS + noise + learned external input).
The data is **pre-generated** (shared with fly_N9_62_1) — do NOT modify simulation parameters.

Primary metrics: **connectivity_R2** (R² between learned W and ground-truth W) and **field_R2** (R² of learned external input vs ground-truth visual field).
Secondary metrics: **tau_R2** (time constant recovery), **V_rest_R2** (resting potential recovery), **cluster_accuracy** (neuron type clustering from embeddings).

## Key Difference from fly_N9_62_1

This config adds `field_type: visual`, which enables **joint learning of connectivity W and the external visual input** via a Siren (sinusoidal representation network) implicit neural network. The Siren network maps (x, y, t) -> I(x, y, t), reconstructing the spatiotemporal visual stimulus seen by photoreceptors. The challenge is to simultaneously recover both the connectivity matrix and the visual input — two unknowns that interact through the dynamics.

## CRITICAL: Data is PRE-GENERATED

**DO NOT change any simulation parameters** (n_neurons, n_frames, n_edges, n_input_neurons, n_neuron_types, delta_t, noise_model_level, visual_input_type). The data lives in `graphs_data/fly/fly_N9_62_1/` and is fixed.

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i(t)/dt = -v_i(t) + V_i^rest + sum_j W_ij * ReLU(v_j(t)) + I_i(t)
```

- 13,741 neurons, 65 cell types, 434,112 edges
- 1,736 input neurons (photoreceptors)
- DAVIS visual input, noise_model_level=0.05
- 64,000 frames, delta_t=0.02
- **I_i(t) is unknown** — learned by the Siren network from the dynamics

## GNN Architecture

Two MLPs learn the neural dynamics:
- **lin_edge** (g_phi): Edge message function. Maps (v_j, a_i) -> message. If `lin_edge_positive=True`, output is squared.
- **lin_phi** (f_theta): Node update function. Maps (v_i, a_i, aggregated_messages, I_i) -> dv_i/dt.
- **Embedding a_i**: 2D learned embedding per neuron, encodes neuron type.

### Siren Network (NNR_f) — External Input Reconstruction

The Siren network reconstructs the visual field from spatiotemporal coordinates:
- **Input**: (x/xy_period, y/xy_period, t/T_period) — normalized photoreceptor position and time
- **Output**: I(x,y,t)² (squared to ensure non-negative luminance)
- **Architecture**: Sinusoidal activation functions (sin(omega * Wx + b)) enabling high-frequency detail

Siren parameters (explorable):
- `hidden_dim_nnr_f`: Hidden dimension (default: 4096)
- `n_layers_nnr_f`: Number of hidden layers (default: 3)
- `omega_f`: Frequency parameter controlling spectral bandwidth (default: 4096)
- `omega_f_learning`: Make omega learnable during training (default: False)
- `nnr_f_xy_period`: Spatial normalization period (default: 1.0)
- `nnr_f_T_period`: Temporal normalization period (default: 64000, normalizes time to [0, 1])
- `learning_rate_NNR_f`: Learning rate for Siren parameters (default: 1E-8)
- `outermost_linear_nnr_f`: Linear output layer (default: True)

Architecture parameters (GNN, explorable) — refer to `Signal_Propagation_FlyVis.PARAMS_DOC` for strict dependencies:
- `hidden_dim` / `n_layers`: lin_edge MLP dimensions (default: 64 / 3)
- `hidden_dim_update` / `n_layers_update`: lin_phi MLP dimensions (default: 64 / 3)
- `embedding_dim`: embedding dimension (default: 2)

**CRITICAL — coupled parameters**: `input_size`, `input_size_update`, and `embedding_dim` are linked. When changing `embedding_dim`, you MUST also update:
- `input_size = 1 + embedding_dim` (for PDE_N9_A)
- `input_size_update = 3 + embedding_dim` (v + embedding + msg + excitation)
Example: embedding_dim=4 → input_size=5, input_size_update=7. Failure to update causes a shape mismatch crash.

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
| `coeff_edge_norm` | mu_0 | Monotonicity penalty on lin_edge — enforces dg/dv > 0 | 1.0 |
| `coeff_edge_weight_L1` | - | L1 on lin_edge weights | 1 |
| `coeff_phi_weight_L1_rate` | - | Decay rate for phi L1 penalty per epoch | 0.5 |
| `coeff_W_L1_rate` | - | Decay rate for W L1 penalty per epoch | 0.5 |
| `coeff_W_L2` | - | L2 on learned W (not in base config, add if needed) | 0 |

## Training Parameters (explorable)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate_W_start` (lr_W) | 5E-4 | Learning rate for connectivity matrix W |
| `learning_rate_start` (lr) | 1.2E-3 | Learning rate for MLP parameters |
| `learning_rate_embedding_start` (lr_emb) | 1.5E-3 | Learning rate for neuron embeddings |
| `learning_rate_NNR_f` (lr_siren) | 1E-8 | Learning rate for Siren network |
| `n_epochs` | 1 (claude) | Number of training epochs |
| `batch_size` | 1 | Batch size for training |
| `data_augmentation_loop` | 25 | Data augmentation multiplier |
| `recurrent_training` | False | Enable recurrent (multi-step) training |
| `time_step` | 1 | Number of recurrent steps (if recurrent_training=True) |
| `w_init_mode` | zeros | W initialization: "zeros" or "randn_scaled" (NOT "randn") |
| `coeff_edge_diff` | 750 | L1 on lin_phi — same-type edge weight sharing |
| `coeff_edge_weight_L1` | 0.5 | L1 on lin_edge weights |
| `coeff_phi_weight_L1` | 0.5 | L1 on lin_edge parameters |

## Training Time Constraint

Each epoch runs `Niter = n_frames * data_augmentation_loop / batch_size * 0.2` iterations (baseline: 320,000 at batch_size=1). The Siren network adds computation per iteration. Monitor `training_time_min` in analysis.log.

Factors that increase training time:
- Larger `hidden_dim_nnr_f` (Siren width)
- Larger `hidden_dim` / `n_layers` (GNN width)
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
- `field_R2`: R² of learned vs true external visual input (PRIMARY)
- `tau_R2`: R² of learned vs true time constants
- `V_rest_R2`: R² of learned vs true resting potentials
- `cluster_accuracy`: neuron type clustering accuracy
- `test_R2`: R² of one-step prediction
- `test_pearson`: Pearson correlation of one-step prediction
- `training_time_min`: Training duration in minutes

**Classification:**

- **Converged**: connectivity_R2 > 0.8 AND field_R2 > 0.8
- **Partial**: connectivity_R2 0.3-0.8 OR field_R2 0.3-0.8
- **Failed**: connectivity_R2 < 0.3 AND field_R2 < 0.3

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
Config: lr_W=X, lr=Y, lr_emb=Z, lr_siren=S, coeff_edge_diff=A, coeff_W_L1=B, batch_size=C, hidden_dim=D, hidden_dim_nnr_f=E, omega_f=F, recurrent=[T/F]
Metrics: connectivity_R2=A, field_R2=B, tau_R2=C, V_rest_R2=D, cluster_accuracy=E, test_R2=F, test_pearson=G, training_time_min=H
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

GNN learning rates and regularization are pre-tuned from fly_N9_62_1 (48 iterations). Blocks focus on the unknowns first.

| Block | Focus | Parameters to explore | Rationale |
|-------|-------|----------------------|-----------|
| 1 | Siren learning rate | learning_rate_NNR_f (1E-8 to 1E-4) | Key unknown — explore with proven GNN params |
| 2 | Siren architecture | hidden_dim_nnr_f, n_layers_nnr_f, omega_f, nnr_f_T_period, nnr_f_xy_period | Find what Siren capacity works |
| 3 | GNN-Siren LR interaction | lr_W, lr, lr_emb vs learning_rate_NNR_f | Does adding Siren change optimal GNN lr? |
| 4 | Regularization fine-tuning | coeff_edge_diff, coeff_edge_norm, coeff_W_L1, coeff_phi_weight_L1/L2, coeff_edge_weight_L1 | Fine-tune with Siren present |
| 5 | GNN Architecture | hidden_dim, n_layers, hidden_dim_update, n_layers_update, embedding_dim | Informed by 62_1 Block 3 results |
| 6 | Combined best | Best parameters from blocks 1-5 | Final optimization |

## Block Boundaries

At the end of each block:
1. Summarize findings in memory.md "Previous Block Summary"
2. Update "Established Principles" with confirmed insights
3. Clear "Current Block" section for next block
4. Carry forward the best config as starting point

## Theoretical Background: SIREN Network

### SIREN (Sinusoidal Representation Networks)

SIREN uses sinusoidal activations instead of ReLU, enabling representation of high-frequency signals and their derivatives:

$$\phi(x) = \sin(\omega_0 \cdot Wx + b)$$

| Component | Formula | Purpose |
|-----------|---------|---------|
| First layer | $W \sim U(-1/n, 1/n)$ | Input scaling |
| Hidden layers | $W \sim U(-\sqrt{6/n}/\omega, \sqrt{6/n}/\omega)$ | Preserve gradient magnitude |
| Activation | $\sin(\omega_0 \cdot z)$ | Periodic, smooth derivatives |
| Output | Linear (if `outermost_linear=True`) or sin | Non-periodic output |

The output is **squared** ($I(x,y,t)^2$) to ensure non-negative luminance values.

### omega_f (Frequency Parameter)

`omega_f` controls the spectral bandwidth of the SIREN network — how fine the spatiotemporal details it can represent:

- **Low omega_f (1-30)**: Smooth, low-frequency reconstructions. Stable training.
- **Medium omega_f (30-100)**: Moderate detail. Typical starting range.
- **High omega_f (>100)**: High-frequency detail, captures fine spatiotemporal patterns, but risk of training instability and overfitting to noise.

The initialization of hidden layer weights is inversely proportional to omega_f ($W \sim 1/\omega$), so higher omega_f means smaller initial weights and potentially slower learning that requires careful learning rate tuning.

### Input Normalization (Critical)

The Siren network receives normalized coordinates:
- Spatial: $(x, y)_{norm} = (x, y) / \text{nnr\_f\_xy\_period}$
- Temporal: $t_{norm} = t / \text{nnr\_f\_T\_period}$

**Coordinate scaling parameters** (named "period" because they stretch the input range):
- `nnr_f_xy_period`: Divides spatial coordinates by this value. Larger period → inputs span smaller range → network sees slower spatial variation. Default: 1.0
- `nnr_f_T_period`: Divides time coordinate by this value. Larger period → inputs span smaller range → network sees slower temporal variation. Default: 1.0

**Intuition**: Think of it as "how many cycles fit in the data". Period=10 means the network treats the full time range as 1/10th of its natural period, so it expects 10× slower variation.

**Tuning guidelines**:
- Visual stimuli typically have rich spatial structure but slower temporal dynamics → consider `nnr_f_T_period > nnr_f_xy_period`
- The starting yaml uses `nnr_f_T_period=64000` (= n_frames), normalizing time to [0, 1]
- Smoother temporal dynamics → increase `nnr_f_T_period`
- Smoother spatial patterns → increase `nnr_f_xy_period`
- These allow independent tuning of spatial vs temporal sensitivity without changing `omega_f`

### omega_f_learning (Learnable Frequency)

When `omega_f_learning=True`, the omega parameter becomes a learnable parameter optimized jointly with the network weights:
- Learning rate controlled by `learning_rate_omega_f` (default: 1E-4)
- Can be regularized with `coeff_omega_f_L2` (default: 0) to prevent omega from growing too large
- Advantage: network can adapt its frequency bandwidth during training
- Risk: omega may drift to extreme values, destabilizing training

### Prior Knowledge from INR Experiments (MPM context)

From extensive SIREN exploration in MPM field reconstruction (130+ iterations):

- **Depth sensitivity**: Most fields require EXACTLY 3 layers. Both 2 and 4 can degrade. Exception: F (deformation gradient) tolerates depth 2-5.
- **SIREN + normalization incompatibility**: LayerNorm/BatchNorm DESTROY SIREN performance (R²=0.022). Never use normalization layers with SIREN.
- **omega_f is field/data-dependent**: Optimal omega_f varies by field type and dataset size. More training data generally favors lower optimal omega_f.
- **LR sensitivity**: Deeper networks require lower learning rates. n_layers=5 + lr=3E-5 fails catastrophically.
- **Overtraining risk**: Too many training steps causes overfitting. Monitor for loss increase.
- **batch_size**: Smaller batch sizes (1-8) generally work better for SIREN training.

These findings are from MPM (material point method) physics — the FlyVis context (neural dynamics) may behave differently, but the SIREN architecture constraints (depth sensitivity, normalization incompatibility) are likely universal.

## Known Results (from prior experiments)

### fly_N9_62_1 baseline (10 epochs, no learned field)
- connectivity_R2=0.95, tau_R2=0.80, V_rest_R2=0.40 (with full regularization)

### fly_N9_62_1 LLM exploration (48 iterations, 1 epoch, H100)

**Best results by metric:**

| Metric | Best | Config |
|--------|------|--------|
| connectivity_R2 | **0.980** | lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, edge_diff=750, phi_L1=0.5, edge_L1=0.5 |
| tau_R2 | **0.997** | lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3, edge_diff=1000, phi_L1=0.5, edge_L1=0.5 |
| V_rest_R2 | **0.817** | lr_W=7E-4, lr=1.2E-3, lr_emb=1.5E-3 (default reg) |
| cluster_accuracy | **0.910** | lr_W=5E-4, edge_diff=1000, phi_L1=0.5, edge_L1=0.5 |

**Established principles (from 62_1):**
1. **lr_W=5E-4 to 7E-4** with lr=1.2E-3 and lr_emb=1.5E-3 is optimal (these are pre-set in this config)
2. **lr_emb=1.5E-3 is critical** for low lr_W — lower values cause connectivity collapse
3. **lr_emb >= 1.8E-3 destroys V_rest recovery**
4. **coeff_edge_norm >= 10 is catastrophic** — keep at 1.0
5. **coeff_phi_weight_L1=0.5 + coeff_edge_weight_L1=0.5** improves both connectivity and V_rest (pre-set)
6. **coeff_edge_diff=750-1000** optimal; 1250+ is harmful
7. **coeff_phi_weight_L2 must stay at 0.001** — 0.005 destroys tau and V_rest
8. **coeff_W_L1=5E-5** is optimal for V_rest; 1E-4 boosts conn but hurts V_rest
9. **batch_size=1** used throughout 62_1 exploration (~45 min/epoch on H100)

### fly_N9_63_1 specific expectations
- Adds joint learning of the visual field — expect connectivity_R2 may be lower initially due to the additional unknown
- The Siren learning rate (learning_rate_NNR_f) is critical: too high destabilizes W recovery, too low prevents field learning
- `omega_f` controls spectral bandwidth: higher values capture finer spatiotemporal detail but may overfit noise
