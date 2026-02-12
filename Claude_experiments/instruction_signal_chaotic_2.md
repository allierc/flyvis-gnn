# Simulation-GNN Training Landscape Study (Ablation: No Memory)

## Goal

Map the **simulation-GNN training landscape**: understand which neural activity simulation configurations allow successful GNN training (connectivity_R2 > 0.9) and which are fundamentally harder.

**ABLATION STUDY**: This experiment runs WITHOUT working memory. Each iteration starts fresh with no accumulated knowledge.
Analysis can use information from current block ONLY, DO NOT use information from previous blocks. DO NOT draw comparison between block.

## Iteration Loop Structure

Each block = `n_iter_block` iterations exploring one simulation configuration.
The prompt provides: `Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block`

## File Structure (CRITICAL)

You maintain **TWO** file:

### 1. Full Log (append-only record)

**File**: `{config}_analysis.md`

- Append every iteration's full log entry
- Append block summaries
- **Never read this file** — it's for human record only

### Block Log (append-only, current block only)

**File**: `{config}_block_analysis.md`

- Append every iteration's full log entry within the current block
- Append block summaries at block end
- **IMPORTANT**: This file is reset at each block boundary - you have NO access to previous blocks

**NOTE**: Do NOT create or read any memory file. You CAN NOT access information from previous blocks for resaonning.

---

## Iteration Workflow (Steps 1-4, every iteration)

### Step 1: Analyze Current Results

**Metrics from `analysis.log`:**

- `spectral_radius`: max eigenvalue of ground truth connectivity W
- `effective rank (99% var)`: **CRITICAL** - SVD rank at 99% cumulative variance. Extract this value and log it as `eff_rank=N` in the Activity field. This determines training difficulty ceiling.
- `test_R2`: R² between ground truth and rollout prediction
- `test_pearson`: Pearson correlation between ground truth and rollout prediction
- `connectivity_R2`: R² of learned vs true connectivity weights
- `final_loss`: final training loss
- `cluster_accuracy`: neuron classification

**Example analysis.log format:**
```
spectral radius: 1.029
--- activity ---
  effective rank (90% var): 26
  effective rank (99% var): 84   <-- Extract this value for eff_rank
```

**Classification:**

- **Converged**: connectivity_R2 > 0.9
- **Partial**: connectivity_R2 0.1-0.9
- **Failed**: connectivity_R2 < 0.1

**UCB scores from `ucb_scores.txt`:**

- Provides computed UCB scores for all exploration nodes including current iteration
- At block boundaries, the UCB file will be empty (erased). When empty, use `parent=root`

Example:

```
Node 2: UCB=2.175, parent=1, visits=1, R2=0.997
Node 1: UCB=2.110, parent=root, visits=2, R2=0.934
```

**Block analysis**

Read `{config}_block_analysis.md` to analyse the current block.

### Step 2: Write Outputs

Append to Block Log (`{config}_block_analysis.md`):

**Log Form:**

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [success-exploit/failure-probe]/[exploit/explore/boundary]
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, batch_size=B, low_rank_factorization=[T/F], low_rank=R, n_frames=NF
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, cluster_accuracy=D, final_loss=E
Activity: eff_rank=R (from analysis.log "effective rank (99% var)"), spectral_radius=S, [brief description]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```

**CRITICAL: Always extract `effective rank (99% var)` from `analysis.log` and include it in the Activity field as `eff_rank=R`. This value is essential for understanding training difficulty and must be recorded in every iteration.**

### Step 3: Parent Selection Rule in UCB tree

Step A: Select parent node

- Read `ucb_scores.txt`
- If empty → `parent=root`
- Otherwise → select node with **highest UCB** as parent

**CRITICAL**: The `parent=P` in the Node line must be the **node ID** (integer) of the selected parent, NOT "root" (unless UCB file is empty). Example: if you select node 3 as parent, write `Node: id=4, parent=3`.

Step B: Choose strategy

| Condition                             | Strategy             | Action                                                              |
| ------------------------------------- | -------------------- | ------------------------------------------------------------------- |
| Default                               | **exploit**          | Highest UCB node, try mutation                                      |
| 3+ consecutive R² ≥ 0.9               | **failure-probe**    | Extreme parameter to find boundary                                  |
| n_iter_block/4 consecutive successes  | **explore**          | Select outside recent chain                                         |
| Good config found                     | **robustness-test**  | Re-run same config                                                  |
| 2+ distant nodes with R² > 0.9        | **recombine**        | Merge params from both nodes                                        |
| 100% convergence, branching<10%       | **forced-branch**    | Select node in bottom 50% of tree                                   |
| 4+ consecutive same-param mutations   | **switch-dimension** | Mutate different parameter than recent chain                        |
| 3+ partial results probing boundary   | **boundary-skip**    | Accept boundary as found, explore elsewhere                         |
| 8+ consecutive sequential (no branch) | **forced-diversity** | Select any node with visits ≥ 3 that is NOT the most recent 4 nodes |

**Recombination details:**

Trigger: exists Node A and Node B where:

- Both R² > 0.9
- Not parent-child (distance ≥ 2 in tree)
- Different parameter strengths

Action:

- parent = higher R² node
- Mutation = adopt best param from other node

Example:

```
Node 12: lr_W=1E-2, lr=1E-4, R²=0.94  (good lr_W)
Node 38: lr_W=5E-3, lr=2E-3, R²=0.97  (good lr)

Recombine → lr_W=1E-2, lr=2E-3
```

### Step 4: Edit Config File

Edit config file for next iteration of the exploration.
(The config path is provided in the prompt as "Current config")

**CRITICAL: Config Parameter Constraints**

**DO NOT add new parameters to the `claude:` section.** Only these fields are allowed:

- `n_epochs`: int (training epochs per iteration)
- `data_augmentation_loop`: int (data augmentation count)
- `n_iter_block`: int (iterations per block)
- `ucb_c`: float value (0.5-3.0)

Any other parameters belong in the `training:` or `simulation:` sections, NOT in `claude:`.

Adding invalid parameters to `claude:` will cause a validation error and crash the experiment.

**Training Parameters (change within block):**

Mutate ONE parameter at a time for better causal understanding.

```yaml
training:
  learning_rate_W_start: 2.0E-3 # range: 1E-4 to 1E-2
  learning_rate_start: 1.0E-4 # range: 1E-5 to 1E-3
  learning_rate_embedding_start: 2.5E-4 # used when n_neuron_types>1
  coeff_W_L1: 1.0E-5 # range: 1E-6 to 1E-3
  batch_size: 8 # values: 8, 16, 32
  low_rank_factorization: False # or True
  low_rank: 20 # range: 5-100
  coeff_edge_diff: 100 # enforces positive monotonicity
```

**Simulation Parameters (FIXED SEQUENCE - change at block boundaries only):**

This ablation study uses simulation configurations from the **original experiment** (instruction_signal_chaotic_1).

---

## Block Workflow (Steps 1-2, every end of block)

Triggered when `iter_in_block == n_iter_block`

### STEP 1: Read and Apply Next Block's Simulation Config from Original Experiment

At the end of block N, you MUST read the config file for block N+1 from the original experiment and apply its simulation parameters.

**Reference config directory**: `/groups/saalfeld/home/allierc/Graph/NeuralGraph/log/Claude_exploration/instruction_signal_chaotic_1/config/`

**MANDATORY: At end of Block 1, read and apply block_002.yaml. At end of Block 2, read and apply block_003.yaml. Etc.**

**Procedure at end of block N:**

1. Use the Read tool to read: `/groups/saalfeld/home/allierc/Graph/NeuralGraph/log/Claude_exploration/instruction_signal_chaotic_1/config/block_{N+1:03d}.yaml`
   - Example: At end of Block 1, read `block_002.yaml`
   - Example: At end of Block 2, read `block_003.yaml`
2. Extract the `simulation:` section values from that file
3. Use the Edit tool to update the current config file with these simulation parameters:
   - `connectivity_type`
   - `Dale_law`
   - `Dale_law_factor`
   - `n_frames`
   - `n_neurons`
   - `connectivity_rank`
4. Also update `data_augmentation_loop` in both `claude:` and `training:` sections

**CRITICAL**: You MUST actually read the reference block file using the Read tool. Do not assume it doesn't exist without trying to read it first. Files block_001.yaml through block_011.yaml all exist.

Also reset training parameters to defaults at block boundaries:

```yaml
training:
  learning_rate_W_start: 4.0e-03
  learning_rate_start: 1.0e-04
  learning_rate_embedding_start: 2.5e-04
  coeff_W_L1: 1.0e-05
  low_rank_factorization: false
  low_rank: 20
```

### STEP 2: Append Block Summary to Log

Append a brief block summary to `{config}_block_analysis.md`:

```
## Block N Summary
Simulation: connectivity_type=X, Dale_law=Y, n_frames=Z, n_neurons=W
Best R²: X.XXX
Convergence rate: X/16
Key observation: [one line]
```

---

## Theoretical Background

### GNN Architecture (Signal_Propagation)

The model learns neural dynamics du/dt using a graph neural network:

```
du/dt = lin_phi(u, a) + W @ lin_edge(u, a)
```

**Components:**

- `lin_edge` (MLP): message function on edges, transforms source neuron activity
- `lin_phi` (MLP): node update function, computes local dynamics
- `W`: learnable connectivity matrix (n_neurons × n_neurons)
- `a`: learnable node embeddings (n_neurons × embedding_dim)

**Forward pass:**

1. For each edge (j→i): message = W[i,j] × lin_edge(u_j, a_j)
2. Aggregate messages: msg_i = Σ_j W[i,j] × lin_edge(u_j, a_j)
3. Update: du/dt_i = lin_phi(u_i, a_i) + msg_i

**Low-rank factorization:**

When `low_rank_factorization=True`: W = W_L @ W_R where W_L ∈ ℝ^(n×r), W_R ∈ ℝ^(r×n)

**Node embeddings for heterogeneity:**

The embedding vector `a_i` allows each neuron to have different dynamics parameters:

- When `n_neuron_types > 1`: embeddings are learnable (requires `lr_emb`)
- When `n_neuron_types = 1`: embeddings are fixed (all neurons identical)
- Embeddings are concatenated with activity: lin_phi(u_i, a_i) and lin_edge(u_j, a_j)
- This allows the MLPs to learn neuron-type-specific transfer functions

### Training Loss and Regularization

**Prediction loss:**

```
L_pred = ||du/dt_pred - du/dt_true||₂
```

**Key regularization terms:**

- `coeff_W_L1`: L1 on W (sparsity). Range: 1E-6 to 1E-4
- `coeff_edge_diff`: enforces monotonicity of lin_edge output (positive: higher u → higher output)
  - Computed as: relu(msg(u) - msg(u+δ))·coeff for sampled u values
  - Stabilizes message function, prevents oscillating gradients

**Learning rates:**

- `learning_rate_W_start` (lr_W): learning rate for connectivity W
- `learning_rate_start` (lr): learning rate for lin_edge and lin_phi
- `learning_rate_embedding_start` (lr_emb): learning rate for node embeddings a

**Total loss:**

```
L = L_pred + coeff_W_L1·||W||₁ + coeff_edge_diff·L_edge_diff + ...
```

### Connectivity W

**Spectral Radius**

- ρ(W) < 1: activity decays → harder to constrain W
- ρ(W) ≈ 1: edge of chaos → rich dynamics → good recovery
- ρ(W) > 1: unstable

**Low-rank Connectivity**

- W = W_L @ W_R constrains solution space
- Without factorization: spurious full-rank solutions

**Dale's Law and E/I Balance**

- Dale_law=True enforces excitatory/inhibitory (E/I) constraint on connectivity W
- Dale_law_factor controls E/I ratio: 0.5 means 50% excitatory, 50% inhibitory neurons

### Data Complexity

**Effective Rank (eff_rank, svd_rank)**

The effective rank measures the intrinsic dimensionality of neural activity data. It is computed via SVD decomposition of the activity matrix (n_frames × n_neurons):

```
U, S, Vt = SVD(activity)
cumulative_variance = cumsum(S²) / sum(S²)
eff_rank = min k such that cumulative_variance[k] ≥ 0.99
```

The effective rank at 99% variance is logged in `analysis.log` as:

```
effective rank (99% var): 16
```
