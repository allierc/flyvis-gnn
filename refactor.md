# Refactoring Instructions — flyvis-gnn

## 1. Eliminating Magic Column Indices with NeuronState and NeuronTimeSeries

### The problem

The original codebase represented neuron state as a packed `(N, 9)` tensor where columns had implicit meaning:

```
x[:, 0] = index
x[:, 1:3] = pos (x, y)
x[:, 3] = voltage
x[:, 4] = stimulus
x[:, 5] = group_type
x[:, 6] = neuron_type
x[:, 7] = calcium
x[:, 8] = fluorescence
```

This created fragile code full of magic numbers like `x[:, 3:4]`, `x[:, 7:8]`, `x[n, 6]`. Any column reordering or addition would silently break everything. The same indices appeared across `graph_trainer.py`, `utils.py`, `GNN_PlotFigure.py`, and the ODE wrapper — hundreds of occurrences with no single source of truth.

### The solution: NeuronState and NeuronTimeSeries

Two dataclasses in `src/flyvis_gnn/neuron_state.py` replace the packed tensor:

**NeuronState** — single-frame state for N neurons:
```python
@dataclass
class NeuronState:
    # static (set once)
    index: torch.Tensor        # (N,) long
    pos: torch.Tensor          # (N, 2) float32
    group_type: torch.Tensor   # (N,) long
    neuron_type: torch.Tensor  # (N,) long
    # dynamic (updated per frame)
    voltage: torch.Tensor      # (N,) float32
    stimulus: torch.Tensor     # (N,) float32
    calcium: torch.Tensor      # (N,) float32
    fluorescence: torch.Tensor # (N,) float32
```

**NeuronTimeSeries** — full simulation recording (T frames, N neurons):
```python
@dataclass
class NeuronTimeSeries:
    # static (stored once)
    index: torch.Tensor        # (N,)
    pos: torch.Tensor          # (N, 2)
    group_type: torch.Tensor   # (N,)
    neuron_type: torch.Tensor  # (N,)
    # dynamic (per frame)
    voltage: torch.Tensor      # (T, N)
    stimulus: torch.Tensor     # (T, N)
    calcium: torch.Tensor      # (T, N)
    fluorescence: torch.Tensor # (T, N)
```

### Key methods

- `NeuronState.from_numpy(x)` — convert legacy `(N, 9)` packed tensor to named fields
- `NeuronState.to_packed()` — convert back for legacy interfaces
- `NeuronTimeSeries.frame(t)` — extract single-frame `NeuronState` at time t (clones dynamic fields so modifications don't corrupt the timeseries)
- `NeuronTimeSeries.load(path)` — auto-detect format (zarr_v2, zarr_v1, npy) and load
- `.to(device)`, `.clone()`, `.detach()`, `.subset(ids)`, `.zeros(n)`

### Migration pattern

Before:
```python
x = torch.zeros(n_neurons, 9)
x[:, 3] = initial_voltage
x[:, 4] = stimulus
y = model(NeuronState.from_numpy(x), edge_index)
x[:, 3:4] = x[:, 3:4] + dt * y
```

After:
```python
x = NeuronState(index=..., pos=..., voltage=initial_voltage, stimulus=stimulus, ...)
y = model(x, edge_index)
x.voltage = x.voltage + dt * y.squeeze(-1)
```

### What's been migrated

- `Signal_Propagation_FlyVis.forward()` accepts `NeuronState` directly
- `data_train_flyvis` constructs `x` as `NeuronState`, removes all `isinstance` checks
- Training data (`x_list`) is loaded as `NeuronTimeSeries`, kept on GPU
- `data_test_flyvis` still uses packed tensors internally (migration pending)
- `GNN_PlotFigure.py` — `plot_synaptic_flyvis()` fully migrated:
  - `x_list` loaded via `load_simulation_data()` → returns `NeuronTimeSeries`
  - Type/region extraction: `x0.neuron_type`, `x0.group_type` (not `x[:, 6]`, `x[:, 5]`)
  - Activity stats: `x0.voltage.t()` (not `x_list[0][:, :, 3:4]`)
  - Activity traces: `to_numpy(x0.voltage).T`, `to_numpy(x0.stimulus).T`
  - Baseline checks: `type_list[n].item()` (not `x_list[0][100][n, 6]`)
  - Batch building: `x0.frame(k).to(device).to_packed()` (not `torch.tensor(x_list[0][k])`)
- `GNN_PlotFigure.py` — `plot_signal()` fully migrated (same pattern as flyvis)
- `plot_synaptic_CElegans` and `plot_synaptic_zebra` removed (non-flyvis datasets)

### Shape considerations

NeuronState fields are `(N,)` but model output is `(N, 1)`. Always squeeze when assigning:
```python
x.voltage = x.voltage + dt * y.squeeze(-1)
x.stimulus[:n_input] = visual_input.squeeze(-1)
```


## 2. Vectorizing Per-Neuron Loops

### The problem

Several analysis functions iterate over all neurons individually (N = 13,741):
```python
for n in range(n_neurons):
    rr = torch.linspace(lo[n], hi[n], 1000)
    embedding = model.a[n, :] * torch.ones((1000, emb_dim))
    in_features = torch.cat([rr[:, None], embedding], dim=1)
    func = model.lin_edge(in_features)
    plt.plot(to_numpy(rr), to_numpy(func), ...)
    slope, _ = curve_fit(linear_model, to_numpy(rr), to_numpy(func))
```

Three bottlenecks compound:
1. **13,741 individual MLP forward passes** — tiny GPU operations with high kernel-launch overhead
2. **13,741 `plt.plot()` calls** — matplotlib has ~1ms overhead per call
3. **13,741 `scipy.curve_fit()` calls** — Python-level optimization per neuron

### The solution: four vectorized helpers in `src/flyvis_gnn/plot.py`

**`_vectorized_linspace(starts, ends, n_pts, device)`**

Creates an `(N, n_pts)` tensor where each row spans a different range. Instead of N calls to `torch.linspace`, uses parametric broadcasting:

```python
t = torch.linspace(0, 1, n_pts)                          # (n_pts,)
rr = starts[:, None] + t[None, :] * (ends - starts)[:, None]  # (N, n_pts)
```

**`_batched_mlp_eval(mlp, model_a, rr, build_features_fn, device, chunk_size=2000)`**

Evaluates an MLP for all neurons in one (chunked) forward pass. Instead of building `(1000, D)` features per neuron and calling the MLP N times, it builds `(chunk_size * 1000, D)` features and makes one call per chunk:

```python
for i in range(0, N, chunk_size):
    rr_flat = rr[i:i+chunk_size].reshape(-1, 1)               # (C*1000, 1)
    emb_flat = model_a[i:i+chunk_size, None, :].expand(-1, 1000, -1).reshape(-1, emb_dim)
    in_features = build_features_fn(rr_flat, emb_flat)         # (C*1000, D)
    out = mlp(in_features.float())                             # one GPU call
    results.append(out.reshape(C, 1000))
```

The `chunk_size` parameter (default 2000) limits GPU memory: 2000 neurons × 1000 points × ~5 features × 4 bytes ≈ 40 MB per chunk. This is a standard GPU batching technique.

**`_vectorized_linear_fit(x, y)`**

Vectorized closed-form least squares, replacing N `scipy.curve_fit` calls. For `y = slope * x + offset`:

```python
n = x.shape[1]
sx, sy = x.sum(axis=1), y.sum(axis=1)
sxy = (x * y).sum(axis=1)
sxx = (x * x).sum(axis=1)
slopes = (n * sxy - sx * sy) / (n * sxx - sx**2)
offsets = (sy - slopes * sx) / n
```

This is the standard normal-equations solution — all operations are element-wise numpy array ops on `(N,)` vectors. The entire regression for 13,741 neurons completes in microseconds.

**`_plot_curves_fast(ax, rr, func, type_list, cmap)`**

Replaces N `plt.plot()` calls with a single `matplotlib.collections.LineCollection`. Builds an `(N, n_pts, 2)` array of line segments, assigns per-neuron colors from the type colormap, and adds one collection to the axes:

```python
segments = np.stack([rr, func], axis=-1)   # (N, n_pts, 2)
colors = [cmap.color(type[n]) for n in range(N)]
lc = LineCollection(segments, colors=colors, linewidths=1)
ax.add_collection(lc)
```

### Where it's applied

- `extract_lin_edge_slopes()` and `extract_lin_phi_slopes()` in `plot.py` — used by both training and analysis
- `plot_lin_edge()` and `plot_lin_phi()` in `plot.py` — used in training subplot movies
- Four inline loops in `GNN_PlotFigure.py` `plot_synaptic_flyvis()` — full-range MLP visualization + domain-range slope extraction

Expected speedup: from ~100 seconds (4 loops × 25s) down to ~2–4 seconds.


## 3. FigureStyle and Shared Plot Architecture

### Design

The plotting code is organized in three layers, shared across all stages of the pipeline (data generation, training, testing, post-training analysis):

```
src/flyvis_gnn/
├── figure_style.py    # FigureStyle dataclass — fonts, sizes, colors, figure creation
├── plot.py            # Shared analysis + subplot functions (W correction, MLP curves, ...)
├── generators/
│   └── plots.py       # Data generation plots (connectivity, spatial layout, ...)
└── models/
    └── utils.py       # plot_training_flyvis() — calls plot.py functions
GNN_PlotFigure.py      # Post-training analysis — calls plot.py functions via thin wrappers
```

### FigureStyle (`src/flyvis_gnn/figure_style.py`)

A `@dataclass` that centralizes all visual parameters — font family, axis label sizes, tick sizes, line widths, DPI, figure sizes — so every plot in the codebase has a consistent look. Two predefined instances:

- `default_style` — white background, used in generators and GNN_PlotFigure
- `dark_style` — dark background, used during training for quick debugging

Usage:
```python
from flyvis_gnn.figure_style import default_style as style

style.apply_globally()           # sets matplotlib rcParams once
fig, ax = style.figure()         # creates figure with configured size/DPI
style.xlabel(ax, "$v_i$")        # consistent font size
style.savefig(fig, "out.png")    # consistent DPI and tight layout
```

The key benefit: changing a font size or DPI in one place updates every figure. No more scattered `fontsize=48` / `fontsize=24` / `dpi=300` hardcoded across thousands of lines.

### Shared plot functions (`src/flyvis_gnn/plot.py`)

A single module used by both the training loop and post-training analysis, eliminating duplicate code.

**W correction pipeline:**
- `compute_activity_stats(x_list)` — per-neuron mean/std of voltage
- `extract_lin_edge_slopes(model, ...)` — slope r_j for each neuron (vectorized)
- `extract_lin_phi_slopes(model, ...)` — slope (1/tau) and offset (V_rest) for each neuron (vectorized)
- `compute_grad_msg(model, in_features, config)` — d(lin_phi)/d(msg) via autograd
- `compute_corrected_weights(model, edges, slopes_phi, slopes_edge, grad_msg)` — applies the correction formula: `W*_ij = -W_ij / slope_phi[i] * grad_msg[i] * slope_edge[j]`
- `compute_all_corrected_weights(model, config, edges, x_list, device)` — high-level pipeline

**Subplot functions:**
- `plot_embedding(ax, model, type_list, n_types, cmap)`
- `plot_lin_phi(ax, model, config, ...)` — vectorized
- `plot_lin_edge(ax, model, config, ...)` — vectorized
- `plot_weight_scatter(ax, gt_weights, learned_weights, ...)` — with optional outlier removal, color, and size parameters
- `plot_tau(ax, slopes, gt_taus, ...)`
- `plot_vrest(ax, slopes, offsets, gt_V_rest, ...)`

**Integration pattern:**
- `plot_training_flyvis()` in `utils.py` calls `compute_all_corrected_weights()` + `plot_weight_scatter()` directly
- `plot_synaptic_flyvis()` in `GNN_PlotFigure.py` calls the vectorized helpers directly for MLP evaluation and slope extraction
- `generators/plots.py` uses `FigureStyle` for generation-time plots (connectivity matrices, spatial layouts)

**Design principle:** `GNN_PlotFigure.py` generates static figures only. Movie generation is handled separately (e.g. visual stimuli reconstruction with SIREN).


## 4. Eliminating Config Variable Unpacking

### The problem

Every major function started with 20–54 lines of `variable = config.section.field` boilerplate, duplicated across 6 functions in `graph_trainer.py` and `GNN_PlotFigure.py`:

```python
n_neurons = simulation_config.n_neurons
n_input_neurons = simulation_config.n_input_neurons
delta_t = simulation_config.delta_t
dataset_name = config.dataset
n_runs = train_config.n_runs
...  # 20-50 more lines
```

The same variables were unpacked identically in `data_train_flyvis`, `data_test_flyvis`, `data_train_flyvis_RNN`, `plot_signal`, and `plot_synaptic_flyvis`.

### The solution: use Pydantic config objects directly

Each function keeps three short section aliases:

```python
sim = config.simulation
tc = config.training
model_config = config.graph_model
```

Then uses `sim.n_neurons`, `tc.batch_size`, `model_config.signal_model_name` directly throughout. The provenance of each value is explicit at the point of use.

### What changed

~88 lines of pure assignment removed. ~300 variable references rewritten from local names to config-prefixed forms. Computed or conditional values kept as locals:
- `n_neurons` in `data_test_flyvis` (conditional on `training_selected_neurons`)
- `replace_with_cluster = 'replace' in tc.sparsity` (derived)
- `getattr()` calls in `data_train_INR` (provide defaults for optional fields)

## 5. Model Registry, Naming Cleanup, and Dead Code Removal

### The problem

Three related issues in the model layer:

1. **Magic names**: `PDE_N9` is a meaningless internal code (PDE variant #9). `Signal_Propagation_FlyVis` is an overly verbose class name inherited from the parent repo. Neither conveys what the code does.

2. **Dead dispatch functions**: `choose_training_model()` in `models/utils.py` (85 lines) and `choose_model()` in `generators/utils.py` (45 lines) dispatched to model classes (`PDE_N2`–`PDE_N8`, `PDE_N11`, `Signal_Propagation_MLP`, etc.) that don't exist in this repo. `GNN_PlotFigure.py` called both functions, but `choose_model()` had no case for `flyvis_A` — it only covered PDE_N2–N7 and N11 — so the ground-truth plotting paths were silently broken for flyvis configs.

3. **If/elif model creation chains**: Three sites in `graph_trainer.py` used if/elif chains to instantiate models by checking `signal_model_name` strings against class names (`Signal_Propagation_Temporal`, `Signal_Propagation_MLP_ODE`, `Signal_Propagation_MLP`, `Signal_Propagation_FlyVis`). Only the last branch was ever reachable. Additionally, ~50 lines of try/except imports attempted to load non-existent model classes, a `data_train_zebra` branch called a non-existent function, and `GNN_PlotFigure.py` had dead branches for PDE_N2–N8/N11 plotting.

### The solution: Model registry + rename + cleanup

**Model registry** (`src/flyvis_gnn/models/registry.py`):

A decorator-based registry replaces scattered if/elif dispatch with a single lookup:

```python
from flyvis_gnn.models.registry import register_model, create_model

@register_model("flyvis_A", "flyvis_B", "flyvis_C", ...)
class FlyVisGNN(nn.Module):
    ...

# At call sites (graph_trainer.py):
model = create_model(model_config.signal_model_name,
                     aggr_type=model_config.aggr_type,
                     config=config, device=device)
```

The registry uses lazy auto-discovery: `_discover_models()` imports model modules on first `create_model()` or `list_models()` call, triggering their `@register_model` decorators. This avoids circular imports and keeps startup fast.

**Class and file renames**:

| Old | New | Purpose |
|-----|-----|---------|
| `Signal_Propagation_FlyVis` (class) | `FlyVisGNN` | Learned GNN model |
| `Signal_Propagation_FlyVis.py` (file) | `flyvis_gnn.py` | Main model module |
| `PDE_N9` (class) | `FlyVisODE` | Ground-truth ODE simulator |
| `PDE_N9.py` (file) | `flyvis_ode.py` | ODE module |

Config values (`flyvis_A`, `flyvis_A_tanh`, etc.) are preserved as-is — they are registry keys, not class names. Existing saved configs and checkpoints keep working without migration.

**Backward compatibility**: Old files (`Signal_Propagation_FlyVis.py`, `PDE_N9.py`) become one-line re-export stubs. Old class names are aliased at the bottom of the new modules:

```python
# In flyvis_gnn.py:
Signal_Propagation_FlyVis = FlyVisGNN

# In flyvis_ode.py:
PDE_N9 = FlyVisODE
```

### What was removed

- `choose_training_model()` — 85-line 3-stage match/case function in `models/utils.py`
- `choose_model()` — 45-line match/case function in `generators/utils.py`
- ~50 lines of phantom try/except imports in `graph_trainer.py` for non-existent classes: `LowRankINR`, `Signal_Propagation_MLP`, `Signal_Propagation_MLP_ODE`, `Signal_Propagation_Zebra`, `Signal_Propagation_Temporal`, `Signal_Propagation_RNN`, `Signal_Propagation_LSTM`, `HashEncodingMLP`, `integrate_neural_ode_Signal`, `neural_ode_loss_Signal`, zebra utilities
- ~7 try/except imports in `generators/utils.py` for `PDE_N2` through `PDE_N7`, `PDE_N11`
- Zebra branch in `data_train()` dispatcher
- 3 model creation if/elif chains in `graph_trainer.py` replaced with single `create_model()` calls
- 4 `choose_training_model()` calls in `GNN_PlotFigure.py` replaced with `_create_learned_model()` (uses `create_model` from registry)
- 3 `choose_model()` calls in `GNN_PlotFigure.py` replaced with `_create_true_model()` (loads saved parameters and constructs `FlyVisODE` directly — first version that actually works for flyvis configs)
- ~335 lines of dead PDE_N2–N8/N11 branches in `GNN_PlotFigure.py` (modulation plots, symbolic regression blocks, per-model MLP plotting)

### Plotting model creation (`GNN_PlotFigure.py`)

The old `choose_training_model()` and `choose_model()` were replaced by two local helpers:

```python
def _create_learned_model(config, device):
    """Create a fresh FlyVisGNN for loading trained weights into."""
    model = create_model(config.graph_model.signal_model_name,
                         aggr_type=config.graph_model.aggr_type,
                         config=config, device=device)
    bc_pos, bc_dpos = choose_boundary_values(config.simulation.boundary)
    return model, bc_pos, bc_dpos

def _create_true_model(config, W, device):
    """Create a ground-truth FlyVisODE from saved parameters."""
    p = {"tau_i": torch.load('.../taus.pt'),
         "V_i_rest": torch.load('.../V_i_rest.pt'),
         "w": torch.load('.../weights.pt')}
    true_model = FlyVisODE(p=p, f=relu, params=sim.params,
                           model_type=signal_model_name, ...)
    bc_pos, bc_dpos = choose_boundary_values(sim.boundary)
    return true_model, bc_pos, bc_dpos
```

`_create_true_model` loads the ground-truth ODE parameters that were saved during data generation (`weights.pt`, `taus.pt`, `V_i_rest.pt`), constructing a `FlyVisODE` directly. The old `choose_model()` had no match case for `flyvis_A` and would have raised `UnboundLocalError` — this is the first working version for flyvis configs.

### What was kept

- **Behavioral dispatch sites** (10 remaining): Sites in `graph_trainer.py` that check `signal_model_name` for forward pass logic, integration method, hidden state handling, and feature guards. These dispatch to code paths for MLP/RNN/LSTM models that may be added in the future. Cleaning them requires those model classes to exist first.
- **Helper functions** in `GNN_PlotFigure.py` (`determine_plot_limits_signal`, `create_signal_lin_edge_subplot`) that reference PDE_N variants — these are part of `plot_signal()` which is only called for non-fly datasets.

### Files changed

| File | Action |
|------|--------|
| `src/flyvis_gnn/models/registry.py` | New — model registry |
| `src/flyvis_gnn/models/flyvis_gnn.py` | Renamed from `Signal_Propagation_FlyVis.py`, class → `FlyVisGNN` |
| `src/flyvis_gnn/models/Signal_Propagation_FlyVis.py` | Now a backward-compat re-export stub |
| `src/flyvis_gnn/generators/flyvis_ode.py` | Renamed from `PDE_N9.py`, class → `FlyVisODE` |
| `src/flyvis_gnn/generators/PDE_N9.py` | Now a backward-compat re-export stub |
| `src/flyvis_gnn/models/utils.py` | Removed `choose_training_model()` + dead imports |
| `src/flyvis_gnn/generators/utils.py` | Removed `choose_model()` + dead imports |
| `src/flyvis_gnn/models/graph_trainer.py` | Removed phantom imports, zebra branch; 3 model creation sites → `create_model()` |
| `src/flyvis_gnn/generators/graph_data_generator.py` | `PDE_N9` → `FlyVisODE` |
| `GNN_PlotFigure.py` | Removed dead PDE_N2–N8/N11 branches; replaced `choose_training_model` / `choose_model` with registry-based helpers |
| `src/flyvis_gnn/models/__init__.py` | Added `flyvis_gnn`, `registry` to exports |
| `src/flyvis_gnn/generators/__init__.py` | Added `flyvis_ode` to exports |

## 6. Rename PDE_N9 / fly_N9 → flyvis

### The problem

After the registry refactor (section 5), the legacy naming `PDE_N9` and `fly_N9` still permeated the codebase: config keys (`PDE_N9_A`), dataset names (`fly_N9_62_1`), config filenames (`fly_N9_62_1.yaml`), data directories, log directories, documentation, and instruction files. `PDE_N9` was a meaningless internal code from the parent repo (PDE variant #9), and `fly_N9` inherited from it.

### The solution: Comprehensive rename

Two systematic replacements:
- `PDE_N9_` → `flyvis_` (model registry keys: `flyvis_A`, `flyvis_B`, etc.)
- `fly_N9_` → `flyvis_` (dataset/experiment names: `flyvis_62_1`, `flyvis_62_1_gs`, etc.)

**Safety**: `'fly' in config.dataset` still works (flyvis contains fly). Substring dispatches on `'RNN'`, `'LSTM'`, `'MLP'` still work since they check within the model name.

### What changed

| Category | Count | Pattern |
|----------|-------|---------|
| Model registry keys | 11 | `PDE_N9_A` → `flyvis_A`, `PDE_N9_B` → `flyvis_B`, etc. |
| Config YAML content | 361 files | `signal_model_name` + `dataset` values |
| Config YAML filenames | 361 files | `fly_N9_*.yaml` → `flyvis_*.yaml` |
| Data directories | 16 dirs | `graphs_data/fly/fly_N9_*` → `flyvis_*` |
| Log directories | 32 dirs | `log/fly/fly_N9_*` + `log/Claude_exploration/instruction_fly_N9_*` |
| Root-level docs/outputs | 47 files | Instruction MDs, analysis MDs, logs, scores |
| Python source files | ~15 files | String literals, conditionals, dataset name parsing |
| Exploration config/protocol | 438 files | Inside `log/Claude_exploration/` |

Backward compat stubs (`PDE_N9.py`, `Signal_Propagation_FlyVis.py`) and class aliases (`PDE_N9 = FlyVisODE`, `Signal_Propagation_FlyVis = FlyVisGNN`) were removed — the rename is complete, no transition period needed.

## 7. LLM File Reorganization

### The problem

LLM instruction files (9) and exploration output files (~42) were scattered in the project root directory, cluttering it with experiment artifacts mixed alongside source code and scripts.

### The solution: `LLM/` folder + exploration directories

Two moves:

1. **Instruction files** → `LLM/` at project root:
   - `instruction_flyvis_62_0.md`, `instruction_flyvis_62_1.md`, etc.
   - `instructions_epistemic_analysis.md`

2. **Output files** → their respective `log/Claude_exploration/` directories:
   - `flyvis_62_0_Claude_analysis.md`, `*_memory.md`, `*_ucb_scores.txt`, `*_reasoning.log`, `*_epistemic_analysis.md`, `*_exploration_summary.md`, batch logs (`*_00_analysis.log` through `*_03_analysis.log`)
   - Each set of outputs maps to its exploration dir (e.g., `flyvis_62_1_Claude_*` → `log/Claude_exploration/instruction_flyvis_62_1_parallel/`)

### Path updates in Python scripts

All 4 GNN_LLM scripts construct file paths from `root_dir`. Two new variables introduced:

```python
llm_dir = f"{root_dir}/LLM"                    # instruction files
exploration_dir = f"{root_dir}/log/Claude_exploration/{instruction_name}_parallel"  # outputs
```

| Path | Before | After |
|------|--------|-------|
| Instructions | `{root_dir}/{instruction_name}.md` | `{llm_dir}/{instruction_name}.md` |
| Analysis log | `{root_dir}/{llm_task_name}_analysis.md` | `{exploration_dir}/{llm_task_name}_analysis.md` |
| Memory | `{root_dir}/{llm_task_name}_memory.md` | `{exploration_dir}/{llm_task_name}_memory.md` |
| UCB scores | `{root_dir}/{llm_task_name}_ucb_scores.txt` | `{exploration_dir}/{llm_task_name}_ucb_scores.txt` |
| Reasoning log | `{root_dir}/{llm_task_name}_reasoning.log` | `{exploration_dir}/{llm_task_name}_reasoning.log` |
| Batch logs | `{root_dir}/{slot_name}_analysis.log` | `{exploration_dir}/{slot_name}_analysis.log` |

### Files changed

| File | Changes |
|------|---------|
| `GNN_LLM_parallel_flyvis.py` | `llm_dir`, `exploration_dir` defined early; all output paths use `exploration_dir`; instruction paths use `llm_dir` |
| `GNN_LLM_parallel_flyvis_understanding.py` | Same pattern |
| `GNN_LLM_parallel.py` | Same pattern |
| `GNN_LLM.py` | Same pattern (exploration_dir without `_parallel` suffix) |

Cluster path replacement (`analysis_log_path.replace(root_dir, cluster_root_dir)`) continues to work because `exploration_dir` is a subdirectory of `root_dir`.
