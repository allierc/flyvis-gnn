# FlyVis GNN Training Exploration — flyvis_noise_free

## Goal

Optimize GNN training hyperparameters for the **Drosophila visual system** (flyvis_noise_free, no noise, DAVIS input).
The data is **pre-generated** — do NOT modify simulation parameters.

Primary metric: **connectivity_R2** (R² between learned W and ground-truth W).
Secondary metrics: **tau_R2** (time constant recovery), **V_rest_R2** (resting potential recovery), **cluster_accuracy** (neuron type clustering from embeddings).

## CRITICAL: Data is PRE-GENERATED

**DO NOT change any simulation parameters** (n_neurons, n_frames, n_edges, n_input_neurons, n_neuron_types, delta_t, noise_model_level, visual_input_type). The data lives in `graphs_data/fly/flyvis_noise_free/` and is fixed.

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i(t)/dt = -v_i(t) + V_i^rest + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i(t)
dv_i/dt = f_theta(v_i, a_i, sum_j W_ij * g_phi(v_j, a_j)^2, I_i)
```

- 13,741 neurons, 65 cell types, 434,112 edges
- 1,736 input neurons (photoreceptors)
- DAVIS visual input, **no noise** (noise_model_level=0.0)
- 64,000 frames, delta_t=0.02

## GNN Architecture

Two MLPs learn the neural dynamics:

- **g_phi** (MLP1): Edge message function. Maps (v_j, a_j) → message. `g_phi_positive=true` squares output to enforce positivity.
- **f_theta** (MLP0): Node update function. Maps (v_i, a_i, aggregated_messages, I_i) → dv_i/dt.
- **Embedding a_i**: learnable low-dimensional embedding per neuron type.

Architecture parameters (explorable):

- `hidden_dim` / `n_layers`: g_phi MLP width/depth (default: 80 / 3)
- `hidden_dim_update` / `n_layers_update`: f_theta MLP width/depth (default: 80 / 3)
- `embedding_dim`: embedding dimension (default: 2)

**CRITICAL — coupled parameters**: When changing `embedding_dim`, you MUST also update:

- `input_size = 1 + embedding_dim` (v_j + a_j for g_phi)
- `input_size_update = 3 + embedding_dim` (v_i + a_i + msg + I_i for f_theta)

Example: embedding_dim=4 → input_size=5, input_size_update=7. Shape mismatch crashes otherwise.

## Regularization Parameters

The training loss includes:

| Config parameter               | Role                                                                                | Default |
| ------------------------------ | ----------------------------------------------------------------------------------- | ------- |
| `coeff_g_phi_diff`             | Monotonicity penalty on g_phi: ReLU(-dg_phi/dv) → enforces increasing edge messages | 750     |
| `coeff_g_phi_norm`             | Normalization penalty on g_phi at saturation voltage                                | 0.9     |
| `coeff_g_phi_weight_L1`        | L1 penalty on g_phi MLP weights                                                     | 0.28    |
| `coeff_g_phi_weight_L2`        | L2 penalty on g_phi MLP weights                                                     | 0       |
| `coeff_f_theta_weight_L1`      | L1 penalty on f_theta MLP weights                                                   | 0.5     |
| `coeff_f_theta_weight_L1_rate` | Annealing decay rate for f_theta L1 per epoch                                       | 0.5     |
| `coeff_f_theta_weight_L2`      | L2 penalty on f_theta MLP weights                                                   | 0.001   |
| `coeff_f_theta_msg_diff`       | Monotonicity of f_theta w.r.t. message input                                        | 0       |
| `coeff_W_L1`                   | L1 sparsity penalty on connectivity W                                               | 7.5e-05 |
| `coeff_W_L1_rate`              | Annealing decay rate for W L1 per epoch                                             | 0.5     |
| `coeff_W_L2`                   | L2 penalty on W                                                                     | 1.5e-06 |

## Training Parameters (explorable)

| Parameter                       | Default      | Description                                  |
| ------------------------------- | ------------ | -------------------------------------------- |
| `learning_rate_W_start`         | 6e-4         | Learning rate for connectivity matrix W      |
| `learning_rate_start`           | 1.2e-3       | Learning rate for g_phi and f_theta MLPs     |
| `learning_rate_embedding_start` | 1.55e-3      | Learning rate for neuron embeddings          |
| `n_epochs`                      | 1 (claude)   | Epochs per iteration (keep ≤ 2 for time)     |
| `batch_size`                    | 2            | Batch size for training                      |
| `data_augmentation_loop`        | 25           | Data augmentation multiplier                 |
| `recurrent_training`            | false        | Enable multi-step rollout training           |
| `time_step`                     | 1            | Recurrent steps (if recurrent_training=true) |
| `w_init_mode`                   | randn_scaled | W initialization: "zeros" or "randn_scaled"  |

## Training Time Constraint

Baseline (batch_size=2, 64K frames, hidden_dim=80): **~60 min/epoch on H100**, **~90 min on A100**.
Keep total training time ≤ 90 min/iteration. Monitor `training_time_min`.

Factors that increase training time:

- Larger `hidden_dim` / `n_layers`
- Larger `data_augmentation_loop`
- Smaller `batch_size`
- `recurrent_training=true` with large `time_step`

## Parallel Mode — 4 Slots Per Batch

You receive **4 results per batch** and propose **4 mutations** for the next batch.

### Slot Strategy

| Slot | Role               | Description                                                 |
| ---- | ------------------ | ----------------------------------------------------------- |
| 0    | **exploit**        | Highest UCB node, conservative mutation of best config      |
| 1    | **exploit**        | 2nd highest UCB or same parent, different parameter         |
| 2    | **explore**        | Under-visited node or new parameter dimension               |
| 3    | **principle-test** | Validate or challenge one Established Principle from memory |

You may deviate from this split based on context (e.g. all exploit early in block, boundary-probe if all configs converge).

### Slot 3: Principle Testing

1. Read "Established Principles" in memory.md
2. Randomly select one principle (rotate — do not repeat consecutively)
3. Design a config that tests or challenges it
4. Write `Mode/Strategy: principle-test` in the log entry
5. Include `Testing principle: "[quoted text]"` in the Mutation line
6. After results: update evidence level (confirmed → keep, contradicted → Open Questions)

If no Established Principles yet, use slot 3 as **boundary-probe** instead.

### Config Files

- Edit all 4 config files: `{name}_00.yaml` through `{name}_03.yaml`
- **DO NOT change `dataset`** — each slot is pre-routed to its data directory
- Only modify `training:` and `graph_model:` parameters (and `claude:` where allowed)
- **DO NOT change `simulation:` parameters**

## Iteration Loop Structure

Each block = `n_iter_block` iterations (default 12).
The prompt provides: `Block info: block {block_number}, iterations {iter_in_block}/{n_iter_block} within block`

## File Structure

You maintain **THREE** files:

### 1. Full Log (append-only)

**File**: `{llm_task_name}_analysis.md`

- Append every iteration's log entry (4 entries per batch)
- Append block summaries at block boundaries
- **Never read** — human record only

### 2. Working Memory (read + update every batch)

**File**: `{llm_task_name}_memory.md`

- Read at start, update at end
- Contains: established principles, previous block summary, current block iterations
- Keep ≤ 500 lines

### 3. User Input (read every batch, acknowledge pending items)

**File**: `user_input.md`

- Read at every batch
- If "Pending Instructions" section has content: act on it, then move entries to "Acknowledged" section with timestamp
- Do not remove acknowledged entries — append them with `[ACK {batch}]` marker

## Iteration Workflow (every batch)

### Step 1: Read Working Memory + User Input

- Read `{llm_task_name}_memory.md` for context
- Read `user_input.md` for any pending user instructions

### Step 2: Analyze Results (4 slots)

**Metrics from `analysis.log`:**

- `connectivity_R2`: R² of learned vs true W (PRIMARY)
- `tau_R2`: R² of learned vs true time constants
- `V_rest_R2`: R² of learned vs true resting potentials
- `cluster_accuracy`: neuron type clustering accuracy from embeddings
- `test_R2`: one-step prediction R²
- `training_time_min`: training duration

**Classification:**

- **Converged**: connectivity_R2 > 0.9
- **Partial**: connectivity_R2 0.3–0.9
- **Failed**: connectivity_R2 < 0.3

**UCB scores from `ucb_scores.txt`:**

- UCB(k) = R²_k + c × sqrt(ln(N) / n_k) where c = `ucb_c` (default 1.414)
- At block boundaries the UCB file is empty — use `parent=root`

### Step 3: Write 4 Log Entries + Update Memory

Append to Full Log and **Current Block** in memory.md:

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [exploit/explore/boundary/principle-test]
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_g_phi_diff=A, coeff_W_L1=B, batch_size=C, hidden_dim=D, recurrent=[T/F]
Metrics: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, training_time_min=F
Embedding: [visual observation]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```

**CRITICAL**: `Mutation:` is parsed by the UCB tree builder — always include exact parameter change.
**CRITICAL**: `Next: parent=P` — P must be from a previous batch or current batch, NEVER `id+1` (circular reference).

### Step 4: Acknowledge User Input (if any)

If `user_input.md` has content in "Pending Instructions":

- Edit `user_input.md`: move the pending items to "Acknowledged" with `[ACK batch_{batch_first}-{batch_last}]` prefix
- Incorporate the instructions into your next config mutations

### Step 5: Edit 4 Config Files for Next Batch

One or two parameter changes per slot. Test one hypothesis at a time.

## Block Partition (suggested)

| Block | Focus                  | Parameters                                                               |
| ----- | ---------------------- | ------------------------------------------------------------------------ |
| 1     | Learning rates         | lr_W, lr, lr_emb                                                         |
| 2     | g_phi regularization   | coeff_g_phi_diff, coeff_g_phi_norm, coeff_g_phi_weight_L1                |
| 3     | f_theta regularization | coeff_f_theta_weight_L1, coeff_f_theta_weight_L2, coeff_f_theta_msg_diff |
| 4     | W regularization       | coeff_W_L1, coeff_W_L2, w_init_mode                                      |
| 5     | Architecture           | hidden_dim, n_layers, hidden_dim_update, n_layers_update, embedding_dim  |
| 6     | Batch & augmentation   | batch_size, data_augmentation_loop                                       |
| 7     | Recurrent training     | recurrent_training, time_step                                            |
| 8     | Combined best          | Best parameters from blocks 1–7                                          |

## Block Boundaries

At the end of each block:

1. Summarize findings in memory.md "Previous Block Summary"
2. Update "Established Principles" with confirmed insights
3. Clear "Current Block" for next block
4. Carry forward best config as starting point

## Failed Slots

If a slot is `[FAILED]`:

- Write a brief `## Iter N: failed` entry noting the failure
- Still propose a mutation for that slot's next batch
- Do not draw conclusions from a single failure

## Known Results (prior experiments)

- `flyvis_62_1` (DAVIS + noise 0.05): connectivity_R2=0.95, tau_R2=0.80, V_rest_R2=0.40 (10 epochs, full regularization)
- `flyvis_62_1` with Sintel input: R²_W=0.99, tau_R2=1.00, V_rest_R2=0.85
- W initialization: `randn_scaled` and `zeros` perform similarly; plain `randn` performs poorly
- Larger MLP (80-dim/3-layer) works better than smaller (32-dim/2-layer)
- No noise should allow faster convergence and potentially higher connectivity_R2 than noise=0.05
- `coeff_g_phi_diff` (monotonicity) is among the most important regularizers — too low causes non-monotonic messages

## Start Call

When prompt says `PARALLEL START`:

- Read base config to understand training regime
- Create 4 diverse initial variations
- Suggested spread: vary `learning_rate_W_start` across range (e.g. 3e-4, 6e-4, 1e-3, 2e-3)
- All 4 slots share same simulation parameters (pre-generated data)
- Write planned initial variations to working memory
