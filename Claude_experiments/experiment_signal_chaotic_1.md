# Simulation-GNN training Landscape Study

## Goal

Map the **simulation-GNN training landscape**: understand which simulation configurations allow successful GNN training (connectivity_R2 > 0.9) and which simulation configurations are fundamentally harder for GNN training.
When Can GNN recover synaptic weights from simulated data?
Is there a universal training configuration for all simulations or de we need to adjust the training configuration depending on e.g activity rank, simulation regime, spectral radius ?

## Context (CRITICAL)

You are a LLM, you are **hyperparameter optimizer** in a meta-learning loop. In other words, you're mapping a scientific landscape across tasks, not a fixed task
Your role:

1. **Analyze results**: Read activity plots and metrics from the current GNN training run
2. **Update config**: Modify training parameters for the next iteration based on Parent Selection Rule (see below)
3. **Log decisions**: Append structured observations to the analysis file
4. **Self-improve**: At simulation block boundaries, you are asked edit THIS protocol file to refine your own exploration rules
5. **Accumulate knowledge**: Carry insights across blocks — what worked in previous regimes informs hypotheses for new regimes
   regime comes from config file:
   connectivity_type: str = "none" # none, Lorentz, Gaussian, uniform, chaotic, ring attractor, low_rank, successor, null, Lorentz_structured_X_Y
   connectivity_rank: int = 1

## Analysis of Files

- `analysis.log`: metrics from training/test/plot:
  - `spectral_radius`: eigenvalue analysis of connectivity
  - `svd_rank`: SVD rank at 99% variance (activity complexity)
  - `test_R2`: R² between ground truth and rollout prediction
  - `test_pearson`: Pearson correlation per neuron (mean)
  - `connectivity_R2`: R² of learned vs true connectivity weights
  - `final_loss`: final training loss (lower is better)
- `ucb_scores.txt`: provides pre-computed UCB scores for all nodes including current iteration
  at block boundaries, the UCB file will be empty (erased). When UCB file is empty, use `parent=root`.
- `{config}_reasoning.log`: Claude's terminal output (thinking process) for each iteration
  - Automatically captured and appended after each iteration
  - Shows Claude's analysis, reasoning, and decision-making process
  - Useful for debugging and understanding the exploration strategy

```
Node 2: UCB=2.175, parent=1, visits=1, R2=0.997
Node 1: UCB=2.110, parent=root, visits=2, R2=0.934

```

- `Node N`:
- `UCB`: Upper Confidence Bound score = R² + c×√(log(N_total)/visits); higher = more promising to explore
- `parent`: which node's config was mutated to create this node (root = baseline config)
- `visits`: how many times this node or its descendants have been explored
- `R2`: connectivity_R2 achieved by this node's config

## Classification

- **Converged**: connectivity_R2 > 0.9
- **Partial**: connectivity_R2 0.1-0.9
- **Failed**: connectivity_R2 < 0.1

## Training Parameters to explore

These parameters affect the **GNN training**. Can be changed within a block (when iter_in_block <> n_iter_block)

```yaml
training:
  learning_rate_W_start: 2.0E-3 # LR for connectivity weights W range: 1.0E-4 to 1.0E-2
  learning_rate_start: 1.0E-4 # LR for model parameters range: 1.0E-5 to 1.0E-3
  learning_rate_embedding_start: 2.5E-4 # LR for embeddings range: 1.0E-5 to 1.0E-3, only if n_neuron_types > 1
  coeff_W_L1: 1.0E-5 # L1 regularization on W range: 1.0E-6 to 1.0E-3
  batch_size: 8 # batch size values: 8, 16, 32
  low_rank_factorization: False # enable low-rank W factorization (W = U @ V.T) for recovering low-rank connectivity
  low_rank: 20 # rank of factorization when low_rank_factorization=True, range: 5-100
```

## Simulation Parameters to explore

These parameters affect the **data generation** (simulation). Only change at block boundaries (when iter_in_block == n_iter_block)

```yaml
simulation:
  n_frames: 10000 # can be increased to better constrain the GNN range 10000 to 100000
  connectivity_type: "chaotic" # or "low_rank"
  Dale_law: True # enforce excitatory/inhibitory separation
  Dale_law_factor: 0.5 # fraction excitatory/inhibitory (0.1 to 0.9)
  connectivity_rank: 20 # only used when connectivity_type="low_rank", range 5-100
#   noise_model_level: 0.0 # noise added during simulation, affects data complexity. values: 0, 0.5, 1
```

## Claude Exploration Parameters

These parameters control the UCB exploration strategy. Can be adjusted between blocks to adapt exploration behavior.

```yaml
claude:
  ucb_c: 1.414 # UCB exploration constant (0.5-3.0), adjust between blocks
```

**UCB exploration constant (ucb_c):**

- `ucb_c` controls exploration vs exploitation: UCB(k) = R²_k + c × sqrt(ln(N) / n_k)
- Higher c (>1.5) → more exploration of under-visited branches
- Lower c (<1.0) → more exploitation of high-performing nodes
- Default: 1.414 (√2, standard UCB1)
- Adjust between blocks based on search behavior:
  - If stuck in local optimum (all R² similar, no improvement) → INCREASE ucb_c to 2.0
  - If too much random exploration (jumping between distant nodes) → DECREASE ucb_c to 1.0
  - Typical range: 0.5 to 3.0

## Parent Selection Rule (CRITICAL)

**Step 1: select parent node to ccontinue**

- Use `ucb_scores.txt` to select a new node
- If UCB file is empty → `parent=root`
- Otherwise → select node with **highest UCB** as parent

**Step 2: Choose exploration strategy**

| Condition                                       | Strategy            | Action                                                                  |
| ----------------------------------------------- | ------------------- | ----------------------------------------------------------------------- |
| Default                                         | **exploit**         | Use highest UCB node, try new mutation                                  |
| 3+ consecutive successes (R² ≥ 0.9)             | **failure-probe**   | Deliberately try extreme parameter to find failure boundary             |
| n_iter_block/4 consecutive successes (R² ≥ 0.9) | **explore**         | Use highest UCB node not in last n_iter_block/4 nodes, try new mutation |
| Found good config                               | **robustness-test** | Re-run same config (no mutation) to verify reproducibility              |
| High variance detected (>0.3 R² diff same cfg)  | **seed-vary**       | Re-run best config with different seed to test robustness               |

**failure-probe**: After multiple successes, intentionally push parameters to extremes (e.g., 10x lr, 0.1x lr) to map where the config breaks. This helps understand the stability region.

**robustness-test**: Duplicate the best iteration with identical config to verify the result is reproducible, not due to lucky initialization.

**Reversion check**: If reverting a parameter to match a previous node's value, use that node as parent.
Example: If reverting `lr` back to `1E-4` (Node 2's value), use `parent=2`.

## END Parent selection Rule (CRITICAL)

## Log Format

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [success-exploit/failure-probe]/[exploit/explore/boundary]
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_W_L1=W, batch_size=B, low_rank_factorization=[T/F], low_rank=R, n_frames=NF
Metrics: test_R2=A, test_pearson=B, connectivity_R2=C, final_loss=D
Activity: [brief description of dynamics]
Mutation: [param]: [old] -> [new]
Parent rule: [brief description of Parent Selection Rule]
Observation: [one line about result]
Next: parent=P [CRITICAL: specify which node the NEXT iteration should branch from]
```

### Simulation Blocks

Each block = `n_iter_block` iterations exploring one simulation configuration.
The prompt provides: `Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block`

- `block_number`: which simulation block (1, 2, 3, ...)
- `iter_in_block`: current iteration within this block (1 to n_iter_block)
- `n_iter_block`: total iterations per block

### Within block (iter_in_block < n_iter_block):

Only modify training parameters (learning rates, regularization, batch size)

### Block End (iter_in_block == n_iter_block) Log Format

```
## Simulation Block {block_number} Summary (iters X-Y)
Simulation: connectivity_type=[type], Dale_law=[True/False], Dale_law_factor=[F], connectivity_rank=[R], noise_model_level=[L]
Best R2: [value] at iter [N]
Converged: [Yes/No]
Observation: [what worked/failed for this simulation]
Optimum training: lr_W=[X], lr=[Y], lr_emb=[Z], coeff_W_L1=[W], low_rank_factorization=[T/F], low_rank=[R]

--- NEW SIMULATION BLOCK ---
Next simulation: connectivity_type=[type], Dale_law=[True/False], ...
Node: id=N, parent=root
```

## MANDATORY: Block End Actions (when iter_in_block == n_iter_block)

At the **last iteration of each block** (iter_in_block == n_iter_block), you MUST complete ALL of these actions:

### Checklist (complete in order):

- [ ] **1. Write block summary** (see "Block End Log Format" above)
- [ ] **2. Evaluate exploration rules** using metrics below
- [ ] **3. EDIT THIS PROTOCOL FILE** - modify the rules between `## Parent Selection Rule (CRITICAL)` and `## END Parent selection Rule (CRITICAL)`
- [ ] **4. Document your edit** - in the analysis file, state what you changed and why (or state "No changes needed" with justification)

### Evaluation Metrics for Rule Modification:

1. **Branching rate**: Count unique parents in last n_iter_block/4 iters
   - If all sequential (rate=0%) → ADD exploration incentive to rules
2. **Improvement rate**: How many iters improved R²?
   - If <30% improving → INCREASE exploitation (raise R² threshold)
   - If >80% improving → INCREASE exploration (probe boundaries)
3. **Stuck detection**: Same R² plateau (±0.05) for 3+ iters?
   - If yes → ADD forced branching rule

### Example Protocol Edit:

If branching rate was 0% (all sequential), you might add a new row to the strategy table:

**Before:**

```
| Default                             | **exploit**         | Use highest UCB node, try new mutation                      |
```

**After:**

```
| Default                             | **exploit**         | Use highest UCB node, try new mutation                      |
| Branching rate < 20% in last block  | **force-branch**    | Select random node from top 3 UCB, not the sequential parent|
```

Or modify threshold values, add new conditions, remove ineffective rules, etc.

**IMPORTANT**: You must actually use the Edit tool to modify this file. Simply stating what you would change is NOT sufficient.

## Accumulate knowledge at block End (iter_in_block == n_iter_block)

This is **FUNDAMENTAL**, You're mapping a scientific landscape across tasks
To answer the main question: When Can GNN recover synaptic weights from simulated data?

Before starting Block N, cross-block meta-analysis to synthesize:

1. **Regime comparison table**:

   | Block | Regime | Best R² | Optimal lr_W | Optimal L1 | Key constraint |
   | ----- | ------ | ------- | ------------ | ---------- | -------------- |

example:
| Block | Regime | Best R² |
| 1 | chaotic | 0.9999 |
| 2 | low_rank=20 | 0.9977 |
| 3 | chaotic+Dale | 1.000 |

2. **Emerging patterns**:

- **Structured vs unstructured**: Does low_rank or Dale_law require different training than chaotic?
- **Regularization**: Which regimes need tighter L1? Why might that be?
- **Learning rate scaling**: Does lr_W:lr ratio change across regimes?
- **Effective rank, spectral radius**: How does activity effective_rank ro spectral radius relate to optimal config?
- **Hardest regime so far**: Which had lowest convergence rate? What made it hard?
- **Easiest regime so far**: Which converged fastest? What made it easy?

3. **Hypothesis for Block N**:

- Based on patterns, predict what Block N will need
- State testable prediction before running

Before selecting next simulation, check coverage:
| connectivity_type | Dale_law=False | Dale_law=True |
|-------------------|----------------|---------------|
| chaotic | Block ? | Block ? |
| low_rank=20 | Block ? | Block ? |
| low_rank=50 | Block ? | Block ? |

Priority: fill empty cells before replicating.

Replication is allowed IF motivated:

- **Knowledge test**: "Block 2 took 5 iters to converge. With Block 1-3 insights,
  predict Block 4 will converge in 1-2 iters. Testing transfer efficiency."
- **Robustness**: "Block 2 had 79% convergence. Testing if new understanding
  of lr_W:lr ratio improves to 90%+."

4. **Inject into Block N prompt**:
   "Previous blocks found: [pattern]. This block tests [regime].
   Hypothesis: [prediction]. Start with [informed config]."

## Theoretical Background (for reasoning)

### Why connectivity recovery works

The GNN learns the update rule
\begin{equation*}
\small
\widehat{\dot{\Vec{x}}}\_{i} =\phi^*\left(\Vec{a*i}, x_i\right) + \*\left(t\right) \Agg \Mat{W}*{ij}\psi^_\left(\Vec{a_i},\Vec{a_j},x_j\right).
\label{eqn:GNN}
\end{equation_}
The optimized neural networks are $\phi^*$, $\psi^*$, modeled as MLPs (ReLU activation, hidden dimension~$=64$, 3 layers, output size~$= 1$), and $\Omega^*$ modeled as a coordinate-based MLP \citep[][, input size~$= 3$, hidden dimension~$=128$, 5 layers, output size~$= 1$, $\omega=0.3$]{sitzmann_implicit_2020}. Other learnables are the two-dimensional latent vector $\Vec{a}_i$ associated with each neuron, and the connectivity matrix $\Mat{W}$.
Pre-aggregation nonlinearity ($W\phi(u)$) makes W recovery **linear** once $\phi$ is learned.

### Spectral radius and dynamics

- $\rho(W) < 1$: activity decays → low signal diversity → harder to constrain W
- $\rho(W) \approx 1$: edge of chaos → rich dynamics → good for recovery
- $\rho(W) > 1$: can explode or saturate → training unstable

### Effective rank and parameterization

- High effective_rank (30+): activity spans many dimensions → full W recoverable
- Low effective_rank (<15): activity constrained to subspace → only that subspace of W is identifiable
- **Matching principle**: W parameterization should match activity dimensionality

### Low-rank connectivity

- True W has rank r → only r×N parameters matter, not N×N
- $W = W_L W_R$ with $W_L \in \mathbb{R}^{N \times r}$ constrains solution space
- Without factorization: optimization can find spurious full-rank solutions that fit but don't generalize

### Dale's law

- Excitatory neurons: all outgoing weights ≥ 0
- Inhibitory neurons: all outgoing weights ≤ 0
- E/I ratio affects spectral properties: too much E → $\rho(W) \gg 1$ → instability

### L1 regularization role

- Encourages sparsity in W
- Structured connectivity (low_rank, Dale) may need more/less regularization
- Too high: underfits, can't recover true weights
- Too low: overfits, finds spurious solutions

### Learning rate intuition

- lr_W: how fast W changes → too high = instability, too low = slow/stuck
- lr (model): how fast $\phi$ adapts → must balance with lr_W
- Ratio matters: if $\phi$ learns too fast, W gradient signal is noisy
