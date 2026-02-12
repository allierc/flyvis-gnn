# Recurrent Training with Code Modification

## Goal

Optimize GNN training parameters and **training scheme code** to maximize **connectivity_R2** across different recurrence depths (`time_step`). You can modify both config parameters AND the training loop code in `graph_trainer.py`.

Other important evaluation metrics are rollout R² (test_R2) and training time. **Target training_time ~120 min** - adjust `data_augmentation_loop` to keep training time approximately fixed.

- `time_step=1`: baseline single-step prediction (no recurrence)
- `time_step=4, 16, 32, 64`: recurrent training with increasing recurrence depth

## Iteration Loop Structure

Each block = `n_iter_block` iterations exploring one `time_step` configuration.
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

---

## Iteration Workflow (Steps 1-5, every iteration)

### Step 1: Read Working Memory

Read `{config}_memory.md` to recall:

- Established principles
- Previous block findings
- Current block progress

### Step 2: Analyze Current Results

**Metrics from `analysis.log`:**

- `spectral_radius`: eigenvalue analysis of connectivity
- `effective rank (99% var)`: **CRITICAL** - SVD rank at 99% cumulative variance. Extract this value and log it as `eff_rank=N` in the Activity field. This determines training difficulty ceiling.
- `test_R2`: R² between ground truth and rollout prediction
- `test_pearson`: Pearson correlation between ground truth and rollout prediction
- `connectivity_R2`: R² of learned vs true connectivity weights
- `cluster_accuracy`: clustering accuracy (neuron type classification)
- `final_loss`: final training loss
- `training_time_min`: wall clock training time in minutes

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

### Step 3: Write Outputs

Append to Full Log (`{config}_analysis.md`) and **Current Block** sections of `{config}_memory.md`:

- In memory.md: Insert iteration log in "Iterations This Block" section (BEFORE "Emerging Observations")
- Update "Emerging Observations" at the END of the file

**Log Form:**

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [success-exploit/failure-probe]/[exploit/explore/boundary/code-modification]
Config: time_step=T, lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, batch_size=B, data_augmentation_loop=D
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, cluster_accuracy=D, final_loss=E, training_time=F min
Activity: eff_rank=R (from analysis.log "effective rank (99% var)"), spectral_radius=S
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```

**CRITICAL: Always extract `effective rank (99% var)` from `analysis.log` and include it in the Activity field as `eff_rank=R`. This value is essential for understanding training difficulty and must be recorded in every iteration.**

### Step 4: Parent Selection Rule in UCB tree

Step A: Select parent node

- Read `ucb_scores.txt`
- If empty → `parent=root`
- Otherwise → select node with **highest UCB** as parent

**CRITICAL**: The `parent=P` in the Node line must be the **node ID** (integer) of the selected parent, NOT "root" (unless UCB file is empty). Example: if you select node 3 as parent, write `Node: id=4, parent=3`.

Step B: Choose strategy

| Condition                            | Strategy             | Action                                       |
| ------------------------------------ | -------------------- | -------------------------------------------- |
| Default                              | **exploit**          | Highest UCB node, try mutation               |
| 3+ consecutive R² ≥ 0.9              | **failure-probe**    | Extreme parameter to find boundary           |
| n_iter_block/4 consecutive successes | **explore**          | Select outside recent chain                  |
| Good config found                    | **robustness-test**  | Re-run same config                           |
| 2+ distant nodes with R² > 0.9       | **recombine**        | Merge params from both nodes                 |
| 100% convergence, branching<10%      | **forced-branch**    | Select node in bottom 50% of tree            |
| 4+ consecutive same-param mutations  | **switch-dimension** | Mutate different parameter than recent chain |
| 3+ partial results probing boundary  | **boundary-skip**    | Accept boundary as found, explore elsewhere  |

### Step 5: Edit Config File (default) or Modify Code

Choose ONE:

- **Step 5.1 (DEFAULT)**: Edit config file parameters only
- **Step 5.2 **: Modify Python code when config changes insufficient

## Step 5.1: Edit Config File (default approach)

Edit config file for next iteration of the exploration.
(The config path is provided in the prompt as "Current config")

**CRITICAL: Config Parameter Constraints**

**DO NOT add new parameters to the `claude:` section.** Only these fields are allowed:

- `n_epochs`: int (training epochs per iteration)
- `data_augmentation_loop`: int (data augmentation count)
- `n_iter_block`: int (iterations per block)
- `ucb_c`: float value (0.5-3.0)

Any other parameters belong in the `training:` or `simulation:` sections, NOT in `claude:`.

**Training Parameters (change within block):**

Mutate ONE parameter at a time for better causal understanding.
Does not apply to data_augmentation_loop, as needed to constrain total training_time ~120 min.

```yaml
training:
  # Standard training parameters
  learning_rate_W_start: 1.0E-3 # range: 1E-4 to 1E-2
  learning_rate_start: 5.0E-4 # range: 1E-5 to 1E-3
  learning_rate_embedding_start: 7.5E-4 # for neuron type embeddings
  coeff_W_L1: 1.0E-5 # range: 1E-6 to 1E-4
  batch_size: 8 # do not changes larger value causes out of memory
  data_augmentation_loop: 5 # SCALE to keep training_time ~120 min (range: 1-25)

  # Recurrent-specific parameters (can tune within block)
  noise_recurrent_level: 0.0 # noise injected at each recurrent step (range: 0 to 0.1)
  recurrent_parameters: [0, 0] # additional recurrent parameters
```

**Block-level Parameters (change at block boundaries ONLY):**

**Each block is defined by ONE `time_step` value:**

- The `time_step` value defines the block and remains FIXED for all iterations within the block
- Choose `time_step` based on experimental findings and hypotheses
- Available `time_step` values: 1, 4, 16, 32, 64

```yaml
training:
  # time_step defines the block - keep fixed within block
  time_step: 4 # Choose from: 1, 4, 16, 32, 64

  # Choose ONE training mode: recurrent OR neural_ODE
  recurrent_training: True # explicit Euler rollout
  neural_ODE_training: False # continuous ODE solver (alternative)
  recurrent_training_start_epoch: 0 # epoch to start recurrent training

  # Neural ODE parameters (only if neural_ODE_training=True)
  ode_method: "dopri5" # options: dopri5, rk4, euler, midpoint, heun3
  ode_rtol: 1.0E-4 # relative tolerance for adaptive solver
  ode_atol: 1.0E-5 # absolute tolerance for adaptive solver
  ode_adjoint: True # use adjoint method for memory-efficient backprop
  ode_state_clamp: 10.0 # clamp ODE states to prevent blowup
```

**Simulation Parameters (FIXED for entire experiment):**

```yaml
simulation:
  # DO NOT CHANGE - same dataset for all time_step experiments
  n_neurons: 1000
  n_neuron_types: 4
  n_frames: 100000
  connectivity_type: "Lorentz"
```

## Step 5.2: Modify Code

**When to modify code:**

- When config-level parameters are insufficient
- When a failure mode indicates a fundamental limitation
- When you have a specific architectural hypothesis to test
- When 3+ iterations suggest a code-level change would help
- NEVER modify code in first 4 iterations of a block

**Files you can modify (if necessary):**

| File                                           | Permission                                   |
| ---------------------------------------------- | -------------------------------------------- |
| `src/NeuralGraph/models/graph_trainer.py`      | **ONLY modify `data_train_signal` function** |
| `src/NeuralGraph/models/Signal_Propagation.py` | Can modify if necessary                      |
| `src/NeuralGraph/utils.py`                     | Can modify if necessary                      |
| `GNN_PlotFigure.py`                            | Can modify if necessary                      |

**Key model attributes (read-only reference):**

- `model.W` - Connectivity matrix `(n_neurons, n_neurons)`
- `model.a` - Node embeddings `(n_neurons, embedding_dim)`
- `model.lin_edge` - Edge message MLP
- `model.lin_phi` - Node update MLP

**Input data.x columns:** `[particle_id, x, y, u(signal), external_input, plasticity, neuron_type, calcium]`

**How code reloading works:**

- Training runs in a subprocess for each iteration, reloading all modules
- Code changes are immediately effective in the next iteration
- Syntax errors cause iteration failure with error message
- Modified files are automatically committed to git with descriptive messages

**Safety rules (CRITICAL):**

1. **Make minimal changes** - edit only what's necessary
2. **Test in isolation first** - don't combine code + config changes
3. **Document thoroughly** - explain WHY in mutation log
4. **One change at a time** - never modify multiple functions simultaneously
5. **Preserve interfaces** - don't change function signatures

### A. Allowed Training Loop Changes (data_train_signal only)

**Allowed modifications:**

- Change optimizer (Adam → AdamW, SGD, RMSprop)
- Add learning rate scheduler (CosineAnnealingLR, ReduceLROnPlateau)
- Add gradient clipping
- Modify loss function (add regularization terms, use different distance metrics)
- Change data sampling strategy
- Add early stopping logic
- Modify recurrent training loop (how predictions accumulate)

**Example: Add learning rate schedule**

```python
# After optimizer creation:
optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)  # ADD THIS

# In training loop (after optimizer.step()):
scheduler.step()  # ADD THIS
```

**Example: Add gradient clipping**

```python
# In training loop, after loss.backward():
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ADD THIS
optimizer.step()
```

**Example: Change loss function for recurrent training**

```python
# Instead of L2 norm:
# OLD: loss = ((pred_x - y_batch) / (delta_t * time_step)).norm(2)
# NEW: Use Huber loss for robustness
loss = torch.nn.functional.huber_loss(pred_x, y_batch, delta=0.1) / (delta_t * time_step)
```

**Example: Add loss on intermediate predictions**

```python
# In recurrent loop, accumulate intermediate losses:
intermediate_loss = 0
for step in range(1, time_step):
    pred = model(batch, data_id=data_id, k=k_batch + step)
    pred_x = pred_x + delta_t * pred
    # Add intermediate supervision (optional)
    if step % 4 == 0:  # every 4th step
        intermediate_loss += 0.1 * ((pred_x - y_intermediate) ** 2).mean()
loss = final_loss + intermediate_loss
```

### B. Logging Code Modifications

**In iteration log, use this format:**

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: code-modification
Config: [unchanged from parent, or specify if also changed]
CODE MODIFICATION:
  File: src/NeuralGraph/models/graph_trainer.py
  Function: data_train_signal
  Change: Added CosineAnnealingLR scheduler with T_max=n_epochs
  Hypothesis: Decaying learning rate may help long-horizon recurrent training
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, cluster_accuracy=D, final_loss=E, training_time=F min
Mutation: [code] data_train_signal: Added LR scheduler
Parent rule: [one line]
Observation: [compare to parent - did code change help?]
Next: parent=P
```

### C. Constraints and Prohibitions

**NEVER:**

- Modify GNN_recurrent.py (breaks the experiment loop)
- Change function signatures (breaks compatibility)
- Add dependencies requiring new pip packages
- Make multiple simultaneous code changes (can't isolate causality)
- Modify code just to "try something" without hypothesis

**ALWAYS:**

- Explain the hypothesis motivating the code change
- Compare directly to parent iteration (same config, code-only diff)
- Document exactly what changed (file, line numbers, what was added/removed)
- Consider config-based solutions first

### D. Specific Hypotheses Worth Testing via Code

**Optimization:**

- "Hypothesis: Long recurrent chains cause vanishing gradients → add gradient clipping"
- "Hypothesis: Fixed LR is suboptimal for recurrent training → add LR scheduler"
- "Hypothesis: Adam momentum hurts recurrent stability → try SGD or AdamW"

**Loss function:**

- "Hypothesis: L2 loss too sensitive to outliers → try Huber loss"
- "Hypothesis: Only final-step loss is too sparse → add intermediate supervision"
- "Hypothesis: Loss normalization by time_step is wrong → try different scaling"

**Recurrent loop:**

- "Hypothesis: Noise injection helps generalization → increase noise_recurrent_level via code"
- "Hypothesis: Teacher forcing helps early training → add curriculum from time_step=1 to target"

---

## Block Workflow (Steps 1-3, every end of block)

Triggered when `iter_in_block == n_iter_block`

### STEP 1: COMPULSORY — Edit Instructions (this file)

You **MUST** use the Edit tool to add/modify parent selection rules in this file.

**Evaluate and modify rules based on:**

- Branching rate < 20% → ADD exploration rule
- Improvement rate < 30% → INCREASE exploitation
- Same R² plateau for 3+ iters → ADD forced branching

### STEP 2: Update Config for Next time_step Block

**Set `time_step` for the new block** based on:

- Current block findings (what worked, what failed)
- Gaps in exploration (untested time_step values)
- Hypotheses about scaling behavior
- Available values: 1, 4, 16, 32, or 64

The `time_step` value defines the block and remains FIXED for all iterations within the block.

State in analysis log: `"CONFIG EDITED: Block N time_step=Y (reason: ...)"`

### STEP 3: Update Working Memory

Update `{config}_memory.md`:

- Update Knowledge Base with confirmed principles
- Add row to Time Step Comparison Table
- Replace Previous Block Summary with **short summary** (2-3 lines)
- Clear "Iterations This Block" section
- Write hypothesis for next block

---

## Working Memory Structure

```markdown
# Working Memory

## Knowledge Base (accumulated across all blocks)

### Time Step Comparison Table

**eff_rank column MUST be populated from `analysis.log` line: `effective rank (99% var): N`**

| Block | time_step | eff_rank | Best R² | Optimal lr_W | Optimal lr | Optimal L1 | Rollout R² | Training time (min) | Key finding |
| ----- | --------- | -------- | ------- | ------------ | ---------- | ---------- | ---------- | ------------------- | ----------- |
| 1     | 1         | -        | -       | -            | -          | -          | -          | -                   | baseline    |
| 2     | 4         | -        | -       | -            | -          | -          | -          | -                   | ...         |

### Established Principles

[Confirmed patterns that apply across time_step values]

### Open Questions

[Patterns needing more testing, contradictions]

---

## Previous Block Summary (Block N-1)

[Short summary only - NOT individual iterations]

---

## Current Block (Block N)

### Block Info

time_step: X
Iterations: M to M+n_iter_block

### Hypothesis

[Prediction for this block, stated before running]

### Iterations This Block

[Current block iterations only — cleared at block boundary]

### Emerging Observations

[Running notes on what's working/failing]
**CRITICAL: This section must ALWAYS be at the END of memory file.**
```

---

## Theoretical Background

### Recurrent Training

When `recurrent_training=True` and `time_step=T`:

1. Model predicts T steps ahead using its own predictions
2. Loss is computed only at step T against ground truth
3. Gradients backpropagate through T steps

**Code logic** (from graph_trainer.py):

```python
k = k - k % time_step  # align to time_step boundaries

pred_x = x_batch + delta_t * pred + noise_recurrent_level * torch.randn_like(pred)

if time_step > 1:
    for step in range(1, time_step):
        # rebuild graph with predicted x
        pred = model(batch, data_id=data_id, k=k_batch + step)
        pred_x = pred_x + delta_t * pred + noise_recurrent_level * torch.randn_like(pred)

loss = ((pred_x - y_batch) / (delta_t * time_step)).norm(2)
```

**Recurrent-specific parameters:**

| Parameter                        | Description                                            | Range            |
| -------------------------------- | ------------------------------------------------------ | ---------------- |
| `recurrent_training`             | Enable recurrent mode                                  | True/False       |
| `time_step`                      | Number of steps to unroll                              | 1, 4, 16, 32, 64 |
| `noise_recurrent_level`          | Noise added at each step (regularization)              | 0 to 0.1         |
| `recurrent_training_start_epoch` | Epoch to start recurrent (can warmup with single-step) | 0+               |
| `recurrent_parameters`           | Additional parameters [unused currently]               | [0, 0]           |

### Expected Trade-offs

| time_step | Connectivity R² | Rollout Stability | Training Difficulty |
| --------- | --------------- | ----------------- | ------------------- |
| 1         | Best            | May overfit       | Easy                |
| 4         | Good            | Better            | Moderate            |
| 16        | Moderate        | Good              | Harder              |
| 32-64     | Lower           | Best (if works)   | Hardest             |

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
