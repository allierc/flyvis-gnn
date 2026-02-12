# Signal_N2 Sparsity Landscape Study

## Goal

Map the **sparsity-GNN training landscape**: understand how connectivity sparsity affects GNN training success for the PDE_N2 neural dynamics model.

**Study sparsity levels**: 100% (dense), 75%, 50%, 25%, 10%, 5%

For each sparsity level, achieve:

1. **High connectivity_R2** (> 0.9): Accurate recovery of true connectivity matrix W
2. **High cluster_accuracy** (> 0.9): Clear separation of neuron types in embedding space

The model learns neural dynamics following Equation 2 from the paper:
$$\frac{dx_i}{dt} = -\frac{x_i}{\tau_i} + s_i \cdot \tanh(x_i) + g_i \cdot \sum_j W_{ij} \cdot \psi(x_j)$$

## Iteration Loop Structure

Each block = `n_iter_block` iterations exploring one **sparsity configuration**.
The prompt provides: `Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block`

**Block structure**: Each block fixes a sparsity level and explores hyperparameters (lr_W, lr, lr_emb, L1, batch_size, training duration) to find optimal training configuration for that sparsity.

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
- Previous block findings (different sparsity levels)
- Current block progress

### Step 2: Analyze Current Results

**Metrics from `analysis.log`:**

- `spectral radius`: max eigenvalue of ground truth connectivity W
- `connectivity_filling_factor`: current sparsity level (e.g., 0.05 = 5%)
- `test_R2`: R² between ground truth and rollout prediction
- `test_pearson`: Pearson correlation between ground truth and rollout prediction
- `connectivity_R2`: R² of learned vs true connectivity weights
- `cluster_accuracy`: Accuracy of neuron type classification from embeddings
- `final_loss`: final training loss
- `training_time_min`: training time in minutes

**Visual Analysis of W_ij Scatter Plot:**

Examine the latest scatter plot of learned vs true connectivity weights:

- **Location**: `log/signal/{config_name}/tmp_training/matrix/`
- **Files**: `comparison_{epoch}_{iteration}.png` — find the one with highest iteration number

**What to look for:**

1. **Horizontal band at y=0** (learned W=0 while true W≠0):
   - Indicates L1 regularization is too aggressive
   - Many true non-zero connections are being killed
   - Action: Reduce `coeff_W_L1` by factor of 10-100

2. **Slope << 1** (points below diagonal):
   - Weights are being systematically shrunk
   - Combined with horizontal band → L1 is dominating the loss
   - Action: Reduce `coeff_W_L1` or increase `learning_rate_W_start`

3. **Cross pattern** (horizontal + vertical bands at 0):
   - Vertical band at x=0: true zeros staying zero (good)
   - Horizontal band at y=0: true non-zeros being zeroed (bad if excessive)
   - Indicates over-regularization

4. **Good result**:
   - Points tightly clustered along the diagonal (y=x line)
   - No horizontal band at y=0 (or minimal)
   - Slope ≈ 1, R² > 0.9

**Log the scatter plot observation** in the iteration log as:

```
Scatter: [description, e.g., "horizontal band at y=0, L1 too strong" or "tight diagonal, good convergence"]
```

**Dual-Objective Classification:**

| connectivity_R2 | cluster_accuracy | Classification |
| --------------- | ---------------- | -------------- |
| > 0.9           | > 0.9            | **CONVERGED**  |
| > 0.9           | < 0.9            | **partial-W**  |
| < 0.9           | > 0.9            | **partial-E**  |
| 0.1-0.9         | 0.1-0.9          | **partial**    |
| < 0.1           | < 0.1            | **failed**     |

**UCB scores from `ucb_scores.txt`:**

- Provides computed UCB scores for all exploration nodes including current iteration
- At block boundaries, the UCB file will be empty (erased). When empty, use `parent=root`

Example:

```
Node 2: UCB=2.175, parent=1, visits=1, R2=0.997, cluster=0.95
Node 1: UCB=2.110, parent=root, visits=2, R2=0.934, cluster=0.88
```

### Step 3: Write Outputs

Append to Full Log (`{config}_analysis.md`) and **Current Block** sections of `{config}_memory.md`:

- In memory.md: Insert iteration log in "Iterations This Block" section (BEFORE "Emerging Observations")
- Update "Emerging Observations" at the END of the file

**Log Form:**

```
## Iter N: [converged/partial-W/partial-E/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [success-exploit/failure-probe]/[exploit/explore/boundary]
Sparsity: X% (connectivity_filling_factor=Y)
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, batch_size=B, data_aug=D
Metrics: test_R2=A, connectivity_R2=C, cluster_accuracy=D, final_loss=E, time=T min
Scatter: [visual observation from W_ij plot, e.g., "horizontal band at y=0" or "tight diagonal"]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line - note sparsity-specific behavior]
Next: parent=P
```

### Step 4: Parent Selection Rule in UCB tree

Step A: Select parent node

- Read `ucb_scores.txt`
- If empty → `parent=root`
- Otherwise → select node with **highest UCB** as parent

**CRITICAL**: The `parent=P` in the Node line must be the **node ID** (integer) of the selected parent, NOT "root" (unless UCB file is empty).

**UCB Score Computation (Dual Objective):**

UCB(k) = (0.6 × R²_k + 0.4 × cluster_k) + c × sqrt(ln(N) / n_k)

Step B: Choose strategy

| Condition                            | Strategy            | Action                             |
| ------------------------------------ | ------------------- | ---------------------------------- |
| Default                              | **exploit**         | Highest UCB node, try mutation     |
| 3+ consecutive both metrics ≥ 0.9    | **failure-probe**   | Extreme parameter to find boundary |
| n_iter_block/4 consecutive successes | **explore**         | Select outside recent chain        |
| Good config found                    | **robustness-test** | Re-run same config                 |

### Step 5: Edit Config File (default) or Modify Code

#### Step 5.1: Edit Config File (default)

Edit config file for next iteration of the exploration.
(The config path is provided in the prompt as "Current config")

**CRITICAL: Config Parameter Constraints**

**DO NOT add new parameters to the `claude:` section.** Only these fields are allowed:

- `data_augmentation_loop`: int (data augmentation count) # 100 # affects training time - improve results
- `n_iter_block`: int (iterations per block)
- `ucb_c`: float value (0.5-3.0)

Any other parameters belong in the `training:` or `simulation:` sections, NOT in `claude:`.

**Training Parameters (change within block, ONLY one at a time):**

Mutate ONE parameter at a time for better causal understanding.

```yaml
training:
  coeff_W_L1: 1.0E-12 # range 1.0E-10 to 1.0E-14
  learning_rate_W_start: 1.0E-3 # range: 1E-4 to 1E-2
  learning_rate_start: 5.0E-4 # range: 1E-5 to 1E-3
  learning_rate_embedding_start: 7.5E-4 # range: 1E-5 to 1E-2
  batch_size: 8 # values: 1, 2, 4, 8
  data_augmentation_loop: 100 # affects training time - improve results
  coeff_lin_phi_zero: 1.0 # FIXED - DO NOT modify
```

**Simulation Parameters (change at block boundaries only):**

```yaml
simulation:
  n_neurons: 1000 # fixed
  n_neuron_types: 4 # fixed
  n_frames: 100000 # fixed
  connectivity_type: "Lorentz" # fixed
  connectivity_filling_factor: 1.0 # **CHANGE AT BLOCK BOUNDARY**
  # Sparsity levels to explore: 1.0, 0.75, 0.5, 0.25, 0.1, 0.05
  params:
    [[10.0, 1.0, 1.0], [10.0, 2.0, 1.0], [10.0, 1.0, 2.0], [10.0, 2.0, 2.0]]
```

                                               |

**Claude Exploration Parameters:**

```yaml
claude:
  ucb_c: 1.414 # UCB exploration constant (0.5-3.0)
```

#### Step 5.2: Modify Code

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

**Example: Modify L1 regularization for sparse regime**

```python
# Dynamic L1 based on sparsity level:
filling_factor = config.simulation.connectivity_filling_factor
adaptive_L1 = config.training.coeff_W_L1 * (1.0 / filling_factor)  # stronger L1 for sparser networks
loss = loss + adaptive_L1 * model.W.abs().sum()
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
  Change: Added adaptive L1 regularization based on sparsity level
  Hypothesis: Sparser networks need stronger regularization to learn correct zeros
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, cluster_accuracy=D, final_loss=E, training_time=F min
Mutation: [code] data_train_signal: Added adaptive L1
Parent rule: [one line]
Observation: [compare to parent - did code change help?]
Next: parent=P
```

### C. Constraints and Prohibitions

**NEVER:**

- Modify GNN_N2_sparsity.py (breaks the experiment loop)
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

**For sparse connectivity regimes:**

- "Hypothesis: Sparse W needs stronger initialization → scale initial W by 1/filling_factor"
- "Hypothesis: L1 regularization needs to be adaptive to sparsity level → dynamic L1 coefficient"
- "Hypothesis: Sparse networks need more training iterations → adaptive early stopping"

**Optimization:**

- "Hypothesis: Learning W and embeddings jointly is suboptimal for sparse → alternating optimization"
- "Hypothesis: Fixed LR is suboptimal for sparse regime → add LR scheduler"
- "Hypothesis: Gradient magnitude differs for sparse vs dense connections → add gradient clipping"

**Loss function:**

- "Hypothesis: L2 loss weights all connections equally → weight loss by true connectivity"
- "Hypothesis: Zero connections dominate loss in sparse regime → masked loss on non-zeros only"

---

## Block Workflow (Steps 1-3, every end of block)

Triggered when `iter_in_block == n_iter_block`

### STEP 1: COMPULSORY — Edit Instructions (this file)

You **MUST** use the Edit tool to add/modify parent selection rules in this file.
Do NOT just write recommendations in the analysis log — actually edit the file.

After editing, state in analysis log: `"INSTRUCTIONS EDITED: added rule [X]"` or `"INSTRUCTIONS EDITED: modified [Y]"`

**Evaluate and modify rules based on:**

**Sparsity-specific findings:**

- If certain L1 range works best for current sparsity → document in Sparsity-L1 Guidelines table
- If sparsity level fundamentally harder → note in guidelines

**Dual-objective balance:**

- If R² converging but cluster_accuracy stuck → ADD focus-embedding rule
- If cluster_accuracy good but R² stuck → ADD focus-W rule

**Branching rate:**

- Branching rate < 20% → ADD exploration rule
- Branching rate 20-80% → No change needed

### STEP 2: Choose Next Sparsity Block

- Check Sparsity Comparison Table → choose untested sparsity level
- **Explore sparsity levels in order**: 100% → 75% → 50% → 25% → 10% → 5%
- Or revisit a sparsity level if previous exploration was inconclusive
- **Do not repeat same sparsity** unless motivated by knowledge transfer testing

### STEP 3: Update Working Memory

Update `{config}_memory.md`:

- Update Knowledge Base with confirmed principles
- Add row to Sparsity Comparison Table
- Replace Previous Block Summary with **short summary** (2-3 lines, NOT individual iterations)
- Clear "Iterations This Block" section
- Write hypothesis for next block (next sparsity level)

---

## Working Memory Structure

```markdown
# Working Memory

## Knowledge Base (accumulated across all blocks)

### Sparsity Comparison Table

| Block | Sparsity | filling_factor | Best R² | Best Cluster | Optimal L1 | Optimal lr_W | finding |
| ----- | -------- | -------------- | ------- | ------------ | ---------- | ------------ | ------- |
| 1     | 100%     | 1.0            | 0.98    | 0.95         | 0          | 1E-3         |         |
| 2     | 75%      | 0.75           | 0.96    | 0.93         | 1E-6       | 1E-3         |         |
| 3     | 50%      | 0.5            | 0.94    | 0.91         | 5E-6       | 1E-3         |         |
| 4     | 25%      | 0.25           | 0.91    | 0.89         | 2E-5       | 1E-3         |         |
| 5     | 10%      | 0.1            | 0.88    | 0.86         | 5E-5       | 1E-3         |         |
| 6     | 5%       | 0.05           | 0.85    | 0.84         | 1E-4       | 1E-3         |         |

### Established Principles

[Confirmed patterns that apply across sparsity levels]

### Open Questions

[Patterns needing more testing, contradictions]

---

## Previous Block Summary (Block N-1)

[Short summary only - NOT individual iterations. Example:
"Block 1 (100% dense): Best R²=0.98, cluster=0.95 at L1=0.
Key finding: Dense connectivity needs no L1 regularization."]

---

## Current Block (Block N)

### Block Info

Sparsity: X% (connectivity_filling_factor=Y)
Iterations: M to M+n_iter_block

### Hypothesis

[Prediction for this sparsity level based on previous blocks]

### Iterations This Block

[Current block iterations only — cleared at block boundary]

### Emerging Observations

[Running notes on sparsity-specific patterns]
**CRITICAL: This section must ALWAYS be at the END of memory file.**
```

---

## Theoretical Background

### PDE_N2 Model (4 Neuron Types)

The model simulates 1000 neurons with 4 types parameterized by (τ_i, s_i):

- Type 0: τ=0.5, s=1
- Type 1: τ=1.0, s=1
- Type 2: τ=0.5, s=2
- Type 3: τ=1.0, s=2

Dynamics: dx_i/dt = -x_i/τ_i + s_i·tanh(x_i) + g_i·Σ_j W_ij·ψ(x_j)

### GNN Learning Objectives

**1. Connectivity W:**

- The GNN learns W from observing time series x(t)
- connectivity_R2 measures how well learned W matches true W

**2. Neuron Embeddings a_i:**

- Each neuron has a learnable embedding vector a_i
- Embeddings should cluster by neuron type (4 clusters)
- cluster_accuracy measures classification accuracy from embeddings

### Key Trade-offs

**Learning rates (3 independent rates):**

- `learning_rate_W_start` (lr_W): Controls W matrix learning speed
- `learning_rate_start` (lr): Base learning rate for MLP parameters
- `learning_rate_embedding_start` (lr_emb): Controls embedding learning speed

**coeff_L1_W**

- coeff_L1_W is the hyperparameter of the L1-norm applied to the conenctivity matrix W
- this regularisation term is meant to constraint the sparisty of W matrix
- it is a sensitive regularisation, too strong it will favor W=0, too light it will not enforce W sparsity

**Batch size:**

- Larger batches → smoother gradients but fewer updates per epoch
- Smaller batches (1, 2, 4) → more updates, may help sparse regimes converge

**Training duration:**

- data_augmentation_loop controls training duration
- Target ~2h per iteration for efficient exploration

**Regularization:**

- coeff_W_L1: L1 penalty on W matrix, helps enforce sparsity in learned weights

### Sparsity-L1 Guidelines (Accumulated from Experiments)

| Sparsity | filling_factor | Optimal L1 Range | Notes                                                         |
| -------- | -------------- | ---------------- | ------------------------------------------------------------- |
| 100%     | 1.0            | 0 (no L1)        | L1 hurts dense regime; L1≥1E-3 causes failure                 |
| 75%      | 0.75           | 1E-6             | L1=1E-6 REQUIRED (L1=0 hurts); R² ~0.45, cluster FAILS (0.25) |

### Sparse Regime Rules (Added Block 2)

**CRITICAL**: 75% sparsity is a failure regime with current architecture:

- connectivity_R2 plateaus at ~0.45 regardless of hyperparameters
- cluster_accuracy stuck at 0.25 (random) - embeddings do NOT learn
- No hyperparameter combination achieved convergence

**Sparse regime principles**:

1. L1 regularization HELPS sparse connectivity (opposite of dense!)
2. Start with L1=1E-6 for any sparsity < 100%
3. If cluster_accuracy = 0.25 for 3+ iterations, consider architectural changes
4. lr_W=2E-3 slightly better than 1E-3 for sparse
5. Batch size and lr_emb have minimal impact on cluster_accuracy

**When exploring new sparsity level**:

- Start with: lr_W=2E-3, L1=1E-6, batch_size=8, data_aug=100
- If R² improving but cluster stuck at 0.25: sparsity may be too high for current architecture

### Clustering Evaluation

cluster_accuracy is computed by:

1. Extract learned embeddings a_i for all neurons
2. Apply k-means (k=4) or use true labels
3. Compute classification accuracy against true neuron types

A well-trained model should show 4 distinct clusters in embedding space.
