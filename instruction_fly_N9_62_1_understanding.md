# FlyVis GNN Understanding Exploration — Difficult Models

**Follow-up of**: `instruction_fly_N9_62_1` (base exploration, 96 iterations, 46 established principles)

**Run command**:
```bash
# Fresh start (48 iterations = 12 batches of 4)
python GNN_LLM_parallel_flyvis_understanding.py -o train_test_plot_Claude_cluster fly_N9_62_1_understand iterations=144 --fresh

# Resume from last completed batch
python GNN_LLM_parallel_flyvis_understanding.py -o train_test_plot_Claude_cluster fly_N9_62_1_understand iterations=144 --resume
```

## Goal

**Primary**: Understand **WHY** certain FlyVis models are difficult for the GNN to recover connectivity weights.
**Secondary**: Improve connectivity_R2 on these difficult models.

This exploration investigates 4 specific FlyVis models that achieved poor connectivity_R2 in prior evaluations. Unlike the base exploration (which optimized hyperparameters on a single model), this exploration trains **4 different flyvis models simultaneously** — one per parallel slot — and focuses on **understanding the failure modes**.

## Scientific Method

This exploration follows a strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Based on available data (activity ranks, model profiles, previous results), form a hypothesis about WHY a model is difficult (e.g., "Model 041 fails because low-rank activity provides insufficient gradient signal for edge learning")
2. **Design experiment**: Choose a mutation that specifically tests the hypothesis (e.g., increase data_augmentation to give the GNN more exposure to the limited signal)
3. **Run training**: The experiment runs — you cannot predict the outcome
4. **Analyze results**: Use both metrics AND analysis tools to evaluate whether the hypothesis was supported or contradicted
5. **Update understanding**: Revise hypotheses based on evidence. A falsified hypothesis is valuable information.

**CRITICAL**: You can only hypothesize. Only training results and analysis tool outputs can validate or falsify your hypotheses. Never assume a hypothesis is correct without experimental evidence. When results contradict your hypothesis, update it — do not rationalize away the evidence.

The data is **pre-generated** — do NOT modify simulation parameters.

## Target Models

| Slot | Model ID | Baseline R² | svd_rank_99 | activity_rank_99 | Category |
|------|----------|-------------|-------------|------------------|----------|
| 0 | 049 | 0.634 | 19 | 16 | Low activity rank |
| 1 | 011 | 0.308 | 45 | 26 | High rank, worst R² |
| 2 | 041 | 0.629 | 6 | 5 | Near-collapsed activity |
| 3 | 003 | 0.627 | 60 | 35 | Moderate rank, hard connectivity |

### Model Profiles (from generation logs)

**Model 049** (`graphs_data/fly/fly_N9_62_1_id_049/generation_log.txt`):
- activity_rank_90=3, activity_rank_99=16
- svd_activity_rank_90=3, svd_activity_rank_99=19
- Low-dimensional neural activity. Most variance captured by few components.
- R²=0.634 with default training.

**Model 011** (`graphs_data/fly/fly_N9_62_1_id_011/generation_log.txt`):
- activity_rank_90=1, activity_rank_99=26
- svd_activity_rank_90=1, svd_activity_rank_99=45
- Paradox: high SVD rank (diverse activity) yet worst R²=0.308. Activity is diverse but connectivity structure is hard to recover. This is the most interesting model.

**Model 041** (`graphs_data/fly/fly_N9_62_1_id_041/generation_log.txt`):
- activity_rank_90=1, activity_rank_99=5
- svd_activity_rank_90=1, svd_activity_rank_99=6
- Near-collapsed: only 6 SVD components at 99% variance. Very low-dimensional output signal gives the GNN very little to learn from.

**Model 003** (`graphs_data/fly/fly_N9_62_1_id_003/generation_log.txt`):
- activity_rank_90=3, activity_rank_99=35
- svd_activity_rank_90=5, svd_activity_rank_99=60
- Moderate activity rank. Decent diversity but connectivity structure remains hard. Why? With svd_rank=60, there should be enough signal for the GNN.

### Data Directories

Each model's pre-generated data (activity traces, connectivity matrices, etc.) is in:
```
graphs_data/fly/fly_N9_62_1_id_049/
graphs_data/fly/fly_N9_62_1_id_011/
graphs_data/fly/fly_N9_62_1_id_041/
graphs_data/fly/fly_N9_62_1_id_003/
```

## CRITICAL: Data is PRE-GENERATED

**DO NOT change any simulation parameters** (n_neurons, n_frames, n_edges, n_input_neurons, n_neuron_types, delta_t, noise_model_level, visual_input_type). The data is fixed.

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

Architecture parameters (explorable) — refer to `Signal_Propagation_FlyVis.PARAMS_DOC` for strict dependencies:
- `hidden_dim` / `n_layers`: lin_edge MLP dimensions (default: 64 / 3)
- `hidden_dim_update` / `n_layers_update`: lin_phi MLP dimensions (default: 64 / 3)
- `embedding_dim`: embedding dimension (default: 2)

**CRITICAL — coupled parameters**: `input_size`, `input_size_update`, and `embedding_dim` are linked. When changing `embedding_dim`, you MUST also update:
- `input_size = 1 + embedding_dim` (for PDE_N9_A)
- `input_size_update = 3 + embedding_dim` (v + embedding + msg + excitation)

## Regularization Parameters

The training objective is:

```
L = ||y_hat - y||_2 + lambda_0 * ||theta||_1 + lambda_1 * ||phi||_1 + lambda_2 * ||W||_1
    + gamma_0 * ||theta||_2 + gamma_1 * ||phi||_2
    + mu_0 * ||ReLU(-dg_phi/dv)||_2 + mu_1 * ||g_phi(v*, a) - v*||_2
```

| Config parameter | Description | Baseline (Node 79) |
|------------------|-------------|---------------------|
| `coeff_edge_diff` | L1 on lin_phi — same-type edge sharing | 750 |
| `coeff_phi_weight_L1` | L1 on lin_edge — sparsity | 0.5 |
| `coeff_W_L1` | L1 on learned W — sparse connectivity | 5E-5 |
| `coeff_phi_weight_L2` | L2 on lin_edge — stabilization | 0.001 |
| `coeff_edge_norm` | Monotonicity penalty on lin_edge | 1.0 |
| `coeff_edge_weight_L1` | L1 on lin_edge weights | 0.3 |

## Training Parameters

| Parameter | Baseline (Node 79) | Description |
|-----------|---------------------|-------------|
| `learning_rate_W_start` (lr_W) | 6E-4 | Learning rate for W |
| `learning_rate_start` (lr) | 1.2E-3 | Learning rate for MLPs |
| `learning_rate_embedding_start` (lr_emb) | 1.5E-3 | Learning rate for embeddings |
| `n_epochs` | 1 | Training epochs |
| `batch_size` | 2 | Batch size |
| `data_augmentation_loop` | 20 | Data augmentation multiplier |
| `hidden_dim` | 80 | lin_edge MLP hidden dim |
| `hidden_dim_update` | 80 | lin_phi MLP hidden dim |

## Starting Point

All 4 slots start from **Node 79** best params (from base exploration):
```
lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, edge_diff=750, phi_L1=0.5, edge_L1=0.3,
W_L1=5E-5, hidden_dim=80, hidden_dim_update=80, batch=2, data_aug=20
```

These params achieved conn_R2=0.980 on the standard 62_1 model. They may need very different tuning for difficult models.

## No Block Partition

Unlike the base exploration, there is **no block partition**. All parameters are available from iteration 1. You are free to explore any dimension at any time. The UCB tree spans all iterations without block boundaries.

## Iteration Workflow (Steps 1-6, every iteration)

### Step 1: Read Working Memory

Read `{config}_memory.md` to recall UNDERSTANDING hypotheses, established principles, and iteration history.

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

- Provides UCB scores for all exploration nodes (no block scoping)

### Step 3: Write Outputs

Append to Full Log (`{config}_analysis.md`) and Iterations section of `{config}_memory.md`:

**Log Form:**

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Model: [049/011/041/003]
Mode/Strategy: [exploit/explore/hypothesis-test]
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_edge_diff=A, coeff_W_L1=B, batch_size=C, hidden_dim=D, recurrent=[T/F]
Metrics: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, test_pearson=F, training_time_min=G
Embedding: [visual observation]
Mutation: [param]: [old] -> [new]
Observation: [one line — what did this tell us about WHY this model is difficult?]
Analysis: [one-line summary of analysis tool output, or "pending" if tool not yet run]
Next: parent=P
```

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder. Always include the exact parameter change.

### Step 4: Update UNDERSTANDING

After analyzing results, update the `## UNDERSTANDING` section in `{config}_memory.md`:

For each model (049, 011, 041, 003), maintain:
- **Hypothesis**: Current best explanation for why this model is difficult
- **Status**: `untested` / `partially supported` / `supported` / `falsified` / `revised`
- **Evidence FOR**: List of iterations/observations supporting the hypothesis
- **Evidence AGAINST**: List of iterations/observations contradicting the hypothesis
- **Best R² so far**: Track progress
- **Next experiment**: What to try next to test/falsify the hypothesis

**CRITICAL**: A hypothesis is only as good as the evidence. Mark hypotheses as `falsified` when contradicted by training results or analysis tool output. Replace falsified hypotheses with new ones informed by the contradicting evidence. The goal is to converge on the TRUE explanation, not to confirm initial guesses.

### Step 5: Write Analysis Tool

Write a Python analysis script to `tools/analysis_iter_NNN.py` (where NNN is the current batch's last iteration number).

**Purpose**: Probe the training results, learned weights, embeddings, or input data to gain deeper understanding of WHY models are difficult. The pipeline uses a **two-pass architecture**: in pass 1 you analyze results and write the analysis tool; the tool runs as a subprocess immediately after; in pass 2 you receive the tool's stdout output and use it to refine UNDERSTANDING and propose mutations — all within the same batch.

**CRITICAL**: The analysis tool output is TEXT ONLY. You will receive the stdout (print statements) as feedback. Do NOT rely on generated figures for understanding — you cannot see images. All quantitative findings MUST be printed as numbers/tables. You may save .png plots for the human record, but your analysis must be fully expressed through print() output.

**Requirements**:
- Self-contained Python script (no imports from flyvis_gnn)
- Use only: numpy, scipy, torch (for loading .pt files), os, json
- ALL findings must be printed to stdout (this is what you receive as feedback)
- Save numerical results as .npy files to `tools/output/` when needed across iterations
- Script must be runnable with `python tools/analysis_iter_NNN.py`
- May optionally save .png plots for human record, but NEVER depend on them for analysis

**Output format** — print structured, quantitative results:
```python
print("=== SVD Analysis of W_true ===")
print(f"Model 049: rank={rank_049}, nnz={nnz_049}, density={dens_049:.4f}")
print(f"Model 011: rank={rank_011}, nnz={nnz_011}, density={dens_011:.4f}")
print(f"\n=== Per-Type Weight Recovery (R² per neuron type) ===")
for t, r2 in type_r2.items():
    print(f"  type {t:3d}: true_mean_W={mean_w:.4f}, learned_R2={r2:.4f}")
```

**Data available for analysis** (exact paths and tensor shapes):

Ground truth data per model (`graphs_data/fly/fly_N9_62_1_id_{MODEL_ID}/`):
| File | Load with | Shape | Description |
|------|-----------|-------|-------------|
| `weights.pt` | `torch.load(..., weights_only=True)` | `[434112]` | True synaptic weights (one per edge) |
| `taus.pt` | `torch.load(...)` | `[13741]` | True time constants per neuron |
| `V_i_rest.pt` | `torch.load(...)` | `[13741]` | True resting potentials per neuron |
| `edge_index.pt` | `torch.load(...)` | `[2, 434112]` | Edge connectivity (source, target neuron indices) |
| `x_list_0/timeseries.zarr/` | zarr | `[64000, 13741, 4]` | Input timeseries (activity, visual input, ...) |
| `y_list_0.zarr/` | zarr | `[64000, 13741, 1]` | Target output (derivatives) |
| `x_list_0/metadata.zarr/` | zarr | `[13741, 5]` | Neuron metadata (x_pos, y_pos, neuron_type, ...) |
| `generation_log.txt` | text | — | Activity ranks and SVD ranks |

Trained model per slot (`log/fly/fly_N9_62_1_understand_Claude_{SLOT:02d}/`):
| File | Load with | Contents |
|------|-----------|----------|
| `models/best_model_with_0_graphs_0.pt` | `torch.load(..., weights_only=False)` | Dict with `model_state_dict` key |
| `xnorm.pt` | `torch.load(...)` | Scalar — input normalization |
| `ynorm.pt` | `torch.load(...)` | Scalar — output normalization |
| `loss.pt` | `torch.load(...)` | List of loss values per epoch |
| `results/kinograph_gt.npy` | `np.load(...)` | `[n_neurons, n_frames]` — test ground truth activity |
| `results/kinograph_pred.npy` | `np.load(...)` | `[n_neurons, n_frames]` — test predicted activity |

**Model state dict keys** (inside `model_state_dict`):
| Key | Shape | Description |
|-----|-------|-------------|
| `a` | `[13741, 2]` | Learned neuron embeddings (2D) |
| `W` | `[434112, 1]` | Learned connectivity weights |
| `lin_edge.layers.{0,1,2}.weight` | varies | Edge MLP (g_phi) weights |
| `lin_edge.layers.{0,1,2}.bias` | varies | Edge MLP biases |
| `lin_phi.layers.{0,1,2}.weight` | varies | Node update MLP (f_theta) weights |
| `lin_phi.layers.{0,1,2}.bias` | varies | Node update MLP biases |

**Slot → Model mapping**:
| Slot | Model ID | Dataset dir | Log dir |
|------|----------|-------------|---------|
| 0 | 049 | `graphs_data/fly/fly_N9_62_1_id_049/` | `log/fly/fly_N9_62_1_understand_Claude_00/` |
| 1 | 011 | `graphs_data/fly/fly_N9_62_1_id_011/` | `log/fly/fly_N9_62_1_understand_Claude_01/` |
| 2 | 041 | `graphs_data/fly/fly_N9_62_1_id_041/` | `log/fly/fly_N9_62_1_understand_Claude_02/` |
| 3 | 003 | `graphs_data/fly/fly_N9_62_1_id_003/` | `log/fly/fly_N9_62_1_understand_Claude_03/` |

**Example analysis tool** (comparing W_true structure):
```python
import torch, numpy as np
MODEL_IDS = ['049', '011', '041', '003']
SLOTS = [0, 1, 2, 3]
print("=== Ground Truth W Analysis ===")
for mid, slot in zip(MODEL_IDS, SLOTS):
    W = torch.load(f'graphs_data/fly/fly_N9_62_1_id_{mid}/weights.pt', weights_only=True).numpy()
    E = torch.load(f'graphs_data/fly/fly_N9_62_1_id_{mid}/edge_index.pt', weights_only=True).numpy()
    nnz = np.count_nonzero(W)
    print(f"Model {mid}: n_edges={len(W)}, nonzero={nnz}, density={nnz/len(W):.4f}")
    print(f"  W stats: mean={W.mean():.6f}, std={W.std():.6f}, min={W.min():.6f}, max={W.max():.6f}")
    print(f"  |W|>0.01: {(np.abs(W)>0.01).sum()}, |W|>0.1: {(np.abs(W)>0.1).sum()}")
    # Compare with learned W if model exists
    model_path = f'log/fly/fly_N9_62_1_understand_Claude_{slot:02d}/models/best_model_with_0_graphs_0.pt'
    try:
        sd = torch.load(model_path, map_location='cpu', weights_only=False)
        W_learned = sd['model_state_dict']['W'].numpy().flatten()
        r2 = 1 - np.sum((W - W_learned)**2) / np.sum((W - W.mean())**2)
        print(f"  Learned W: R2={r2:.4f}")
    except FileNotFoundError:
        print(f"  No trained model yet")
    print()
```

**Example analysis ideas** (adapt based on current understanding):
- SVD analysis of W_true: compare spectral structure across 4 models
- Per-neuron-type weight recovery: which cell types are hardest?
- Edge magnitude vs recovery error: are weak edges harder?
- Embedding space analysis: load `a` tensor, compute inter/intra-type distances
- Activity rank vs per-type recovery correlation
- Compare W_true sparsity patterns across models
- Residual analysis: which edges have largest errors?

### Step 6: Select Parent and Propose Next Mutation

Use UCB scores to select parent. Propose one or two parameter changes.

**Key difference from base exploration**: Each slot trains a different model. Mutations for one model should be informed by what was learned about that specific model. Cross-model comparisons can reveal whether a hyperparameter effect is model-specific or universal.

## Memory Structure

The working memory file (`{config}_memory.md`) has this structure:

```markdown
# Understanding Exploration: Difficult FlyVis Models

## UNDERSTANDING

### Model 049 (svd_rank_99=19, R²=0.634)
**Hypothesis**: ...
**Evidence**: ...
**Best R² so far**: ...
**Next experiment**: ...

### Model 011 (svd_rank_99=45, R²=0.308)
**Hypothesis**: ...
**Evidence**: ...
**Best R² so far**: ...
**Next experiment**: ...

### Model 041 (svd_rank_99=6, R²=0.629)
**Hypothesis**: ...
**Evidence**: ...
**Best R² so far**: ...
**Next experiment**: ...

### Model 003 (svd_rank_99=60, R²=0.627)
**Hypothesis**: ...
**Evidence**: ...
**Best R² so far**: ...
**Next experiment**: ...

## Established Principles (from base 62_1 exploration)
[46 principles — these are starting knowledge, may need revision for difficult models]

## New Principles (discovered in this exploration)
[Add new findings here]

## Cross-Model Observations
[Patterns that hold across models or differentiate them]

## Analysis Tools Log
[Summary of each analysis tool: what it measured, key findings, and which UNDERSTANDING hypothesis it informed]

| Iter | Tool | What it measured | Key finding | Informed hypothesis |
|------|------|-----------------|-------------|---------------------|
| 4 | analysis_iter_004.py | SVD spectrum of W_true for all 4 models | Model 011 has denser W with many small weights | Model 011: hard connectivity → many weak edges below GNN detection threshold |
| ... | ... | ... | ... | ... |

Keep this table updated after each analysis tool runs. It connects the computational analysis
to the scientific understanding. When a tool's finding changes a hypothesis status, note it here.

## Iterations
[Recent iteration entries]
```

**CRITICAL**: Keep memory under ~500 lines. Compress old iterations into summaries.

## Analysis Tool Feedback (Two-Pass Architecture)

Each batch uses two Claude passes:
- **Pass 1**: You analyze training results, write log entries, update UNDERSTANDING, and write an analysis tool (`tools/analysis_iter_NNN.py`).
- **Tool execution**: The analysis tool runs as a subprocess immediately after pass 1. If it crashes, it is auto-repaired (up to 3 attempts via Claude).
- **Pass 2**: You receive the analysis tool's stdout output. Use it to refine UNDERSTANDING hypotheses, update the Analysis Tools Log, and propose the next 4 config mutations.

All of this happens **within the same batch** — you see your own analysis tool's results before proposing the next experiment.

## Training Time Constraint

Baseline training is ~39 min/epoch on H100 with Node 79 params. Monitor `training_time_min`. If any slot exceeds 60 minutes, reduce complexity.

## Known Results (from base exploration)

The 46 established principles from the base exploration were derived from training on model `fly_N9_62_1` (the standard model, R²=0.980). They may or may not apply to difficult models. Part of this exploration is to discover which principles hold and which break down for difficult models.
