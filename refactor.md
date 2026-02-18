# Refactoring Instructions — flyvis-gnn

Each step must be validated by running `python GNN_Test.py --config flyvis_62_1_gs` (with `--cluster` on the compute cluster). The test compares all key metrics against the reference baseline in `config/test_reference.json` and appends results with timestamp to `log/fly/test_history.md`.

---

## Step 1. Regression Test Infrastructure (`GNN_Test.py`)

### The problem

The project had zero test infrastructure (flagged as critical in the code review). After any refactoring change, there was no automated way to verify that training results hadn't regressed — manual inspection of R² values and plots was the only check.

### The solution: `GNN_Test.py`

A standalone regression test script that:

1. **Archives** current `results.log` and `results_rollout.log` to `log/fly/{config}/archive/` with timestamp
2. **Trains** — locally or on the cluster via SSH+bsub (same pattern as `GNN_LLM.py`)
3. **Runs test_plot** — calls `data_test` + `data_plot` to generate metrics and visualizations
4. **Parses metrics** — regex patterns from `results.log` and `results_rollout.log` (same patterns as `compare_gnn_results()` in `GNN_PlotFigure.py`)
5. **Compares** each metric against reference values with per-metric thresholds (PASS/FAIL)
6. **Calls Claude CLI** — passes comparison table + key plot images for qualitative assessment
7. **Appends** results with date, time, git commit, all metrics, and Claude assessment to `log/fly/test_history.md`

### Reference metrics (`config/test_reference.json`)

Baseline values and regression thresholds:

| Metric          | Reference | Threshold |
| --------------- | --------- | --------- |
| Corrected W R²  | 0.9714    | 0.02      |
| tau R²          | 0.991     | 0.02      |
| V_rest R²       | 0.706     | 0.05      |
| GMM accuracy    | 0.863     | 0.05      |
| Rollout RMSE    | 0.0120    | 0.005     |
| Rollout Pearson | 0.997     | 0.01      |

### Usage

```bash
# Full test on cluster
python GNN_Test.py --config flyvis_62_1_gs --cluster

# Full local test
python GNN_Test.py --config flyvis_62_1_gs

# Skip training, only test_plot + comparison on existing model
python GNN_Test.py --config flyvis_62_1_gs --skip-train

# Only compare existing results.log (no training, no plotting)
python GNN_Test.py --config flyvis_62_1_gs --skip-train --skip-plot

# Skip Claude assessment
python GNN_Test.py --config flyvis_62_1_gs --no-claude
```

### Files created

| File                         | Purpose                         |
| ---------------------------- | ------------------------------- |
| `GNN_Test.py`                | Regression test script          |
| `config/test_reference.json` | Baseline metrics and thresholds |

### Validation

```bash
python GNN_Test.py --config flyvis_62_1_gs --skip-train --skip-plot --no-claude
# → PASS (all metrics within threshold of reference)
```

---

## Step 2. NeuronState, Zarr V3 Storage, and Data Loading Simplification

### 2a. The problem: magic column indices

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

The storage layer had the same problem: simulation data was saved as monolithic `(T, N, 9)` zarr arrays (V1) or split into `metadata.zarr (N, 5)` + `timeseries.zarr (T, N, 4)` (V2). Both formats pack multiple fields into column indices. Loading a single field (e.g., just stimulus) required loading the entire array.

The training loader also wrapped data in `x_list` / `y_list` with a `for run in range(n_runs)` loop, even though `n_runs` is always 1. Every downstream function accepted `x_list` (a list) and `run` (an index), adding unnecessary indirection.

### 2b. NeuronState and NeuronTimeSeries

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

Key methods:

- `NeuronState.from_numpy(x)` — convert legacy `(N, 9)` packed tensor to named fields
- `NeuronState.to_packed()` — convert back for legacy interfaces
- `NeuronTimeSeries.frame(t)` — extract single-frame `NeuronState` at time t
- `NeuronTimeSeries.load(path)` — auto-detect format (zarr_v3, zarr_v2, zarr_v1, npy) and load
- `.to(device)`, `.clone()`, `.detach()`, `.subset(ids)`, `.zeros(n)`

Shape note: NeuronState fields are `(N,)` but model output is `(N, 1)`. Always squeeze when assigning:

```python
x.voltage = x.voltage + dt * y.squeeze(-1)
```

### 2c. Zarr V3 per-field storage

Each `NeuronState` field is saved as its own zarr array in a directory:

```
x_list_0/
├── index.zarr        # (N,) int32 — static, written once
├── pos.zarr          # (N, 2) float32 — static, written once
├── group_type.zarr   # (N,) int32 — static, written once
├── neuron_type.zarr  # (N,) int32 — static, written once
├── voltage.zarr      # (T, N) float32 — dynamic, appended per frame
├── stimulus.zarr     # (T, N) float32 — dynamic, appended per frame
├── calcium.zarr      # (T, N) float32 — dynamic, appended per frame
└── fluorescence.zarr # (T, N) float32 — dynamic, appended per frame
```

Static fields are written once from the first frame. Dynamic fields are buffered and flushed in chunks of `time_chunks` frames (default 2000) using tensorstore with blosc/zstd compression.

```python
class ZarrSimulationWriterV3:
    def __init__(self, path, n_neurons, time_chunks=2000)
    def append_state(self, state: NeuronState)  # buffer + auto-flush
    def finalize(self)                           # resize to exact frame count
```

Format auto-detection in `NeuronTimeSeries.load(path)` and `detect_format()`:

1. **V3** — directory contains `voltage.zarr` → `from_zarr_v3`
2. **V2** — directory contains `metadata.zarr` + `timeseries.zarr` → `from_zarr_v2`
3. **V1** — `path.zarr` with `.zarray` → `from_zarr_v1`
4. **npy** — `path.npy` → `from_numpy`

### 2d. x_list → x_ts simplification

Since `n_runs = 1` always, removed the list wrapping across all functions:

```python
# Before:
x_list = []
for run in trange(0, tc.n_runs):
    x_list.append(load_simulation_data(...))
x_list = [ts.to(device) for ts in x_list]

# After:
x_ts = load_simulation_data(f'graphs_data/{config.dataset}/x_list_0')
x_ts = x_ts.to(device)
```

All downstream functions updated from `(x_list, run)` to `(x_ts)`:

| Function | Before | After |
|----------|--------|-------|
| `plot_training_flyvis` | `x_list` | `x_ts` |
| `compute_activity_stats` | `x_list[0].voltage` | `x_ts.voltage` |
| `compute_all_corrected_weights` | `x_list[0].frame(k)` | `x_ts.frame(k)` |
| `neural_ode_loss_FlyVis` | `x_list, run` | `x_ts` |
| `integrate_neural_ode_FlyVis` | `x_list, run` | `x_ts` |
| `GNNODEFunc_FlyVis` | `x_list, run` | `x_ts` |
| `plot_signal` | `x_list[0]` | `x_ts` |
| `plot_synaptic_flyvis` | `x_list[0]` | `x_ts` |
| `data_test_flyvis` (ODE call) | `x_list=None, run=0` | `x_ts=None` |

### 2e. Data loading cleanup (index removal, type_list extraction, xnorm property, y_data → y_ts)

Several related simplifications to the data loading path in `graph_trainer.py` and supporting modules:

**1. `index` no longer saved to disk or loaded**

`index` was always `arange(n_neurons)` — saving it to disk was redundant. Removed from:
- `ZarrSimulationWriterV3._STATIC_FIELDS_V3`: `['index', 'pos', 'group_type', 'neuron_type']` → `['pos', 'group_type', 'neuron_type']`
- `ZarrSimulationWriterV3._save_static()`: no longer writes `index.zarr`
- `ZarrSimulationWriterV2` metadata packing: `(N, 5)` → `(N, 4)` (removed index column)
- `NeuronTimeSeries.from_zarr_v3()`: gracefully handles missing `index.zarr` (auto-constructs as `arange(n_neurons)`)

Loading code constructs index at runtime:
```python
x_ts.index = torch.arange(x_ts.n_neurons, dtype=torch.long, device=device)
```

**2. `neuron_type` extracted separately as `type_list`**

`neuron_type` is loaded as part of the NeuronTimeSeries, then extracted and cleared to avoid carrying static metadata alongside dynamic tensors:

```python
load_fields = ['voltage', 'stimulus', 'neuron_type']
x_ts = load_simulation_data(..., fields=load_fields).to(device)

type_list = x_ts.neuron_type.float().unsqueeze(-1)
x_ts.neuron_type = None  # clear from timeseries
x_ts.index = torch.arange(x_ts.n_neurons, dtype=torch.long, device=device)
```

**3. Chained `.to(device)` on load**

Single-line load + device transfer:
```python
# Before:
x_ts = load_simulation_data(...)
x_ts = x_ts.to(device)

# After:
x_ts = load_simulation_data(f'graphs_data/{config.dataset}/x_list_0', fields=load_fields).to(device)
```

**4. `xnorm` moved to `NeuronTimeSeries.xnorm` property**

Previously, `xnorm` was computed inline in `data_train_flyvis` using intermediate tensors (`activity`, `distrib`, `valid_distrib`) that persisted on GPU:

```python
# Before (3 persistent GPU tensors):
activity = x_ts.voltage
distrib = activity[~torch.isnan(activity)]
valid_distrib = distrib[distrib != 0]
xnorm = 1.5 * torch.std(valid_distrib)
```

Now a property on `NeuronTimeSeries` in `neuron_state.py`:

```python
@property
def xnorm(self) -> torch.Tensor:
    """Voltage normalization: 1.5 * std of all valid voltage values."""
    v = self.voltage
    valid = v[~torch.isnan(v)]
    if len(valid) > 0:
        return 1.5 * valid.std()
    return torch.tensor(1.0, device=v.device if v is not None else 'cpu')
```

Usage: `xnorm = x_ts.xnorm` — no persistent intermediate tensors.

**5. `y_data` → `y_ts` rename**

Renamed for consistency with `x_ts` naming convention. All 3 occurrences in `graph_trainer.py` updated.

**6. Removed dead frame extraction**

`x = x_ts.frame(sim.n_frames - 10)` was assigned but never used — removed. `n_neurons` is now read directly from `x_ts.n_neurons`.

### What's been migrated

- `Signal_Propagation_FlyVis.forward()` accepts `NeuronState` directly
- `data_train_flyvis` constructs `x` as `NeuronState`, loads data as single `x_ts`
- `data_test_flyvis` constructs `x` as `NeuronState`, uses `x_ts` in ODE calls
- `GNN_PlotFigure.py` — `plot_signal()` and `plot_synaptic_flyvis()` load directly as `x_ts`
- `plot_synaptic_CElegans` and `plot_synaptic_zebra` removed (non-flyvis datasets)
- Dead code removed: `load_training_data()` in `GNN_PlotFigure.py`, 3 unused packed-format plot functions in `plot.py`

### Files modified

| File | Changes |
|------|---------|
| `src/flyvis_gnn/neuron_state.py` | `NeuronState`, `NeuronTimeSeries` dataclasses; `from_zarr_v3`, `load()` with V3 detection |
| `src/flyvis_gnn/zarr_io.py` | `ZarrSimulationWriterV3`, `_load_zarr_v3`, updated `detect_format` |
| `src/flyvis_gnn/generators/graph_data_generator.py` | Switched from `ZarrSimulationWriterV2` to `ZarrSimulationWriterV3` |
| `src/flyvis_gnn/models/graph_trainer.py` | Simplified `data_train_flyvis` and `data_test_flyvis`: x_list → x_ts |
| `src/flyvis_gnn/models/Neural_ode_wrapper_FlyVis.py` | Updated `GNNODEFunc_FlyVis`, `integrate_neural_ode_FlyVis`, `neural_ode_loss_FlyVis` |
| `src/flyvis_gnn/plot.py` | Updated signatures; removed 3 dead packed-format functions |
| `GNN_PlotFigure.py` | Updated `plot_signal`, `plot_synaptic_flyvis`; removed dead `load_training_data` |

### Validation

```bash
python GNN_Test.py --config flyvis_62_1_gs --cluster
```

---

## Step 3. Vectorizing Per-Neuron Loops

### The problem

Several analysis functions iterate over all neurons individually (N = 13,741) with three compounding bottlenecks: 13,741 individual MLP forward passes, 13,741 `plt.plot()` calls, and 13,741 `scipy.curve_fit()` calls.

### The solution: four vectorized helpers in `src/flyvis_gnn/plot.py`

- **`_vectorized_linspace(starts, ends, n_pts, device)`** — `(N, n_pts)` tensor via parametric broadcasting
- **`_batched_mlp_eval(mlp, model_a, rr, build_features_fn, device, chunk_size=2000)`** — one chunked GPU forward pass for all neurons
- **`_vectorized_linear_fit(x, y)`** — closed-form least squares on `(N,)` vectors
- **`_plot_curves_fast(ax, rr, func, type_list, cmap)`** — single `LineCollection` instead of N `plt.plot()` calls

Applied in `extract_lin_edge_slopes()`, `extract_lin_phi_slopes()`, `plot_lin_edge()`, `plot_lin_phi()` in `plot.py`, and four inline loops in `GNN_PlotFigure.py`. Expected speedup: ~100s → ~2–4s.

### Validation

```bash
python GNN_Test.py --config flyvis_62_1_gs --cluster
```

---

## Step 4. FigureStyle, Shared Plot Architecture, and Plot Consolidation

### 4a. FigureStyle

`FigureStyle` centralizes all visual parameters (font sizes, DPI, figure sizes) in a `@dataclass`. Two instances: `default_style` (white) and `dark_style` (training). Shared `plot.py` functions eliminate duplicate code between training and post-training analysis.

### 4b. Plot file consolidation (DONE)

All plot functions consolidated from 5 scattered locations into a single `src/flyvis_gnn/plot.py` (2998 lines). Two files deleted, two files trimmed.

**Before → After:**

| File | Before | After | Change |
|------|--------|-------|--------|
| `src/flyvis_gnn/plot.py` | 636 | 2998 | +2362 (absorbed all) |
| `src/flyvis_gnn/generators/plots.py` | 380 | **DELETED** | -380 |
| `src/flyvis_gnn/models/plot_utils.py` | 1215 | **DELETED** | -1215 |
| `src/flyvis_gnn/generators/utils.py` | 1166 | 521 | -645 (plot functions removed) |
| `src/flyvis_gnn/models/utils.py` | 1678 | 1555 | -123 (plot functions removed) |

**Total: 2363 lines of plot code moved from 4 scattered locations into 1 centralized file.**

**Result:**

```
src/flyvis_gnn/
├── figure_style.py              # FigureStyle dataclass (unchanged)
├── plot.py                      # ALL shared plot functions (2998 lines)
├── generators/
│   └── utils.py                 # No plot functions (521 lines)
└── models/
    └── utils.py                 # No plot functions (1555 lines)
GNN_PlotFigure.py                # Post-training analysis, imports from plot.py
```

**Updated imports in:**
- `generators/graph_data_generator.py` — `from flyvis_gnn.plot import plot_spatial_activity_grid, ...`
- `models/graph_trainer.py` — `from flyvis_gnn.plot import plot_training_flyvis, plot_weight_comparison, plot_signal_loss, ...`
- `GNN_PlotFigure.py` — `from flyvis_gnn.plot import plot_odor_heatmaps, analyze_mlp_edge_lines, ...`

### 4c. GNN_PlotFigure.py dead code removal (pending)

- `plot_signal` (1530 lines) — only used for non-flyvis datasets, not needed in this repo
- `get_figures` (816 lines) — hardcoded 30+ experiment dispatch, replaced by config-driven approach
- 5 `create_signal_*_subplot` functions (360 lines) — only called by `plot_signal`
- `determine_plot_limits_signal` (108 lines) — only called by `plot_signal`
- `compare_ising_results` (198 lines) — Ising model comparison, not flyvis
- Dead code: commented blocks, debug prints (~60 lines)

**Total to remove from GNN_PlotFigure.py: ~3072 lines (5612 → ~2540)**

Remaining in `GNN_PlotFigure.py`:

- `plot_synaptic_flyvis` (1014 lines) — calls functions from `plot.py`
- `data_plot` (55 lines) — entry point
- `analyze_neuron_type_reconstruction` (159 lines)
- `compare_gnn_results` (196 lines)
- `collect_gnn_results_multimodel` (256 lines)
- Model creation helpers, data loading, movie functions

### Validation

```bash
python GNN_Test.py --config flyvis_62_1_gs --cluster
```

---

## Step 5. Eliminating Config Variable Unpacking

### The problem

Every major function started with 20–54 lines of `variable = config.section.field` boilerplate, duplicated across 6 functions.

### The solution

Three short section aliases (`sim`, `tc`, `model_config`) then direct access: `sim.n_neurons`, `tc.batch_size`. ~88 lines of pure assignment removed, ~300 variable references rewritten.

**Rule**: apply only to variables that are **not modified** after unpacking. Variables like `n_frames` and `n_neurons` that get reassigned later in the function must remain as local variables (initialized from `sim.xxx`).

### Applied to

- `graph_trainer.py`: `data_train_flyvis`, `data_test`, `data_plot`, `data_plot_flyvis`
- `graph_data_generator.py`: `data_generate_fly_voltage`
- `generators/utils.py`: `init_neurons`, `init_mesh`
- `plot.py`: `plot_lin_edge`
- `GNN_PlotFigure.py`: `_create_true_model`, `create_signal_excitation_subplot`, `plot_signal`, `plot_synaptic_flyvis`, `create_signal_movies`

### Validation

```bash
python GNN_Test.py --config flyvis_62_1_gs --cluster
```

---

## Step 6. Model Registry, Naming Cleanup, and Dead Code Removal

### The problem

Magic names (`PDE_N9`, `Signal_Propagation_FlyVis`), dead dispatch functions (`choose_training_model()`, `choose_model()`), if/elif model creation chains, ~50 lines of phantom try/except imports.

### The solution

- **Model registry** (`src/flyvis_gnn/models/registry.py`) with `@register_model` decorator
- **Renames**: `Signal_Propagation_FlyVis` → `FlyVisGNN`, `PDE_N9` → `FlyVisODE`
- **Removed**: `choose_training_model()` (85 lines), `choose_model()` (45 lines), phantom imports, zebra branch, ~335 lines of dead PDE_N2–N8/N11 branches in `GNN_PlotFigure.py`
- **New**: `_create_learned_model()` and `_create_true_model()` helpers in `GNN_PlotFigure.py`

### Validation

```bash
python GNN_Test.py --config flyvis_62_1_gs --cluster
```

---

## Step 7. Rename PDE_N9 / fly_N9 → flyvis

Comprehensive rename: `PDE_N9_` → `flyvis_`, `fly_N9_` → `flyvis_` across config keys, YAML files, filenames, data directories, log directories, Python source, and exploration artifacts (~900+ files).

### Validation

```bash
python GNN_Test.py --config flyvis_62_1_gs --cluster
```

---

## Step 8. LLM File Reorganization

Instruction files → `LLM/` folder. Output files → their respective `log/Claude_exploration/` directories. Path updates in all 4 `GNN_LLM*.py` scripts using `llm_dir` and `exploration_dir` variables.

### Validation

```bash
python GNN_Test.py --config flyvis_62_1_gs --cluster
```

---

## Step 9. LLM Exploration Compatibility

### The problem

For an LLM to explore and optimize a GNN training pipeline, it needs three things: (1) what config parameters exist and what they do, (2) what code sections it is allowed to modify, and (3) an orchestration script that runs the loop. Repos without these are opaque to LLM exploration.

### The solution: three components

#### 9a. `PARAMS_DOC` on every GNN model class

A class-level dictionary that documents the model equations, config parameters, and typical ranges. This is the **single source of truth** for what an LLM can change via config.

```python
class FlyVisGNN(nn.Module):
    PARAMS_DOC = {
        "model_name": "FlyVisGNN",
        "description": "du/dt = lin_phi(v, a, sum(msg), excitation), msg = W * lin_edge(v, a)",
        "equations": {
            "message": "msg_j = W[edge] * lin_edge(v_j, a_j)^2",
            "update": "du/dt = lin_phi(v, a, sum(msg), excitation)",
        },
        "graph_model_config": {
            "lin_edge (MLP0)": {
                "input_size": "DERIVED — 1 + embedding_dim",
                "hidden_dim": {"typical_range": [32, 128], "default": 64},
                "n_layers": {"typical_range": [2, 5], "default": 3},
            },
            "lin_phi (MLP1)": {
                "input_size_update": "DERIVED — 1 + embedding_dim + output_size + 1",
                "hidden_dim_update": {"typical_range": [32, 128], "default": 64},
            },
            "embedding_dim": {"typical_range": [1, 8], "default": 2},
        },
        "training_params": {
            "tunable": [
                {"name": "learning_rate_W_start", "typical_range": [1e-4, 5e-2]},
                {"name": "coeff_W_L1", "typical_range": [1e-6, 1e-3]},
                # ...
            ],
        },
    }
```

Key information:

- Model equations (message function, update function)
- Which parameters are **derived** (must NOT be set independently) vs **tunable**
- Typical ranges for every tunable parameter
- Coupled parameter constraints (e.g., `embedding_dim` change requires `input_size` update)

#### 9b. `LLM-MODIFIABLE` comment markers in the trainer

Instead of a separate documentation dict, simple start/stop comment markers delineate the code sections an LLM is Suggested to modify:

```python
# === LLM-MODIFIABLE: OPTIMIZER SETUP START ===
# Change optimizer type, learning rate schedule, parameter groups
optimizer = ...
# === LLM-MODIFIABLE: OPTIMIZER SETUP END ===

# === LLM-MODIFIABLE: TRAINING LOOP START ===
# Main training loop. Suggested changes: loss function, gradient clipping,
# data sampling strategy, LR scheduler steps, early stopping.
# Do NOT change: function signature, model construction, data loading, return values.
for N in pbar:
    ...
    # === LLM-MODIFIABLE: BACKWARD AND STEP START ===
    # Suggested changes: gradient clipping, LR scheduler step, loss scaling
    loss.backward()
    optimizer.step()
    # === LLM-MODIFIABLE: BACKWARD AND STEP END ===
    ...
# === LLM-MODIFIABLE: TRAINING LOOP END ===
```

Three marked sections in `data_train_flyvis()`:

| Marker            | Lines    | Suggested changes                                                 |
| ----------------- | -------- | ----------------------------------------------------------------- |
| OPTIMIZER SETUP   | ~270–285 | Optimizer type (Adam→AdamW, SGD), LR schedulers, parameter groups |
| TRAINING LOOP     | ~349–728 | Loss terms, data sampling, gradient clipping, early stopping      |
| BACKWARD AND STEP | ~534–543 | Gradient clipping, LR scheduler `.step()`, loss scaling           |

Rules:

- **NEVER** modify code outside these markers
- **NEVER** change function signatures or return values
- **NEVER** modify model construction or data loading
- **NEVER** make multiple simultaneous code changes (can't isolate causality)
- Only modify code when config-level sweeps have plateaued (not in first 4 iterations)

#### 9c. LLM orchestration infrastructure

Three files form the LLM exploration loop:

| File                              | Purpose                                                                                                                                     |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `GNN_LLM.py`                      | Orchestration: iteration loop, cluster submission, Claude CLI calls, code change tracking                                                   |
| `LLM/instruction_{experiment}.md` | Per-experiment instructions: goal, metrics, config parameter tables (referencing `PARAMS_DOC`), code modification rules, iteration workflow |
| `src/.../git_code_tracker.py`     | Auto-commits code modifications with descriptive messages, rollback on failure                                                              |

The `GNN_LLM.py` task string controls permissions: `'code'` in the task string enables code modification, otherwise only config changes are allowed.

### Applying to a new repository

To make any graph-learning repo LLM-exploration compatible:

1. **Add `PARAMS_DOC`** to the GNN model class — document equations, config parameters, typical ranges, coupled parameters
2. **Add `LLM-MODIFIABLE` markers** to the training function — bracket the optimizer setup, main loop, and backward/step sections
3. **Copy `GNN_LLM.py`** and adapt — change config paths, cluster settings, metric extraction patterns
4. **Write an instruction file** in `LLM/` — goal, metrics, config parameter table (reference `PARAMS_DOC`), block partition, iteration workflow
5. **Copy `git_code_tracker.py`** — tracks and auto-commits code modifications

### Validation

```bash
python GNN_Test.py --config flyvis_62_1_gs --cluster
```

---

## Step 10. Migrate All Plot Functions to FigureStyle (pending)

### The problem

`FigureStyle` was introduced (Step 4a) as the single source of truth for all visual parameters, but only 4 plot functions actually use it. The remaining 27+ functions use hardcoded font sizes (14, 16, 20, 24, 32, 48, 68pt), the matplotlib dark/default theme is selected via scattered `plt.style.use()` calls in at least 4 files, and a local variable `mc` (main color — `'k'` or `'w'`) is passed around ~80 times across 4 files to select black/white foreground color — this is exactly `style.foreground`.

### Current state

**Functions WITH `style: FigureStyle` parameter (4):**

| Function | File |
|----------|------|
| `plot_spatial_activity_grid` | `plot.py` |
| `plot_kinograph` | `plot.py` |
| `plot_activity_traces` | `plot.py` |
| `plot_selected_neuron_traces` | `plot.py` |

**Functions WITHOUT `FigureStyle` (27+):**

| Function | File | Hardcoded sizes |
|----------|------|-----------------|
| `plot_lin_phi` | `plot.py` | 32, 24 |
| `plot_lin_edge` | `plot.py` | 32, 24 |
| `plot_weight_scatter` | `plot.py` | 32, 24 |
| `plot_tau` | `plot.py` | 32, 24 |
| `plot_vrest` | `plot.py` | 32, 24 |
| `plot_embedding` | `plot.py` | — |
| `plot_synaptic_frame_visual` | `plot.py` | 24 |
| `plot_synaptic_frame_modulation` | `plot.py` | 24 |
| `plot_synaptic_frame_plasticity` | `plot.py` | 24 |
| `plot_synaptic_frame_default` | `plot.py` | — |
| `plot_eigenvalue_spectrum` | `plot.py` | — |
| `plot_connectivity_matrix` | `plot.py` | — |
| `plot_low_rank_connectivity` | `plot.py` | — |
| `plot_signal_loss` | `plot.py` | — |
| `plot_training_flyvis` | `plot.py` | — |
| `plot_odor_heatmaps` | `plot.py` | — |
| `plot_weight_comparison` | `plot.py` | hardcoded `color='white'` |
| `analyze_mlp_edge_lines` | `plot.py` | — |
| `analyze_mlp_edge_lines_weighted_with_max` | `plot.py` | — |
| `analyze_embedding_space` | `plot.py` | — |
| `analyze_mlp_phi_synaptic` | `plot.py` | — |
| `analyze_mlp_phi_embedding` | `plot.py` | — |
| `analyze_individual_architectures` | `plot.py` | — |
| `plot_architecture_analysis_summary` | `plot.py` | — |
| `analyze_data_svd` | `models/utils.py` | 11, 14, 12 (local constants) |
| `plot_confusion_matrix` | `GNN_PlotFigure.py` | 12, 10 |
| `plot_synaptic_flyvis` | `GNN_PlotFigure.py` | 68, 48, 34, 32 |
| `analyze_neuron_type_reconstruction` | `GNN_PlotFigure.py` | — |

**Scattered `plt.style.use()` calls (should be in FigureStyle):**

| File | Lines | What it does |
|------|-------|--------------|
| `graph_data_generator.py` | 78, 124 | `plt.style.use("default")` |
| `graph_trainer.py` | 127, 559 | `plt.style.use('default')` |
| `graph_trainer.py` | 1501, 1504 | dark_background / default |
| `GNN_PlotFigure.py` | 2584, 2587 | dark_background / default |
| `GNN_PlotFigure.py` | 3784, 3794 | ggplot / default |
| `GNN_PlotFigure.py` | 4344, 4655, 4658, 4759, 4832 | various style switches |
| `plot.py` | 1486 | `plt.style.use('default')` |
| `models/utils.py` | 723 | `plt.style.use(style)` |

**Scattered `mc` (main color) variable — duplicates `style.foreground`:**

`mc` is set to `'k'`/`'black'` or `'w'`/`'white'` then passed through function signatures and used for `color=mc` in plot/scatter/text calls. ~80 occurrences across 4 files:

| File | Occurrences | How `mc` is set |
|------|-------------|-----------------|
| `GNN_PlotFigure.py` | ~48 | `mc = 'w'` / `mc = 'k'` conditional on style string |
| `plot.py` | ~14 | passed as parameter `mc='k'` |
| `models/utils.py` | ~13 | `mc = 'w' if style == 'dark_background' else 'k'` |
| `models/graph_trainer.py` | ~6 | `mc = 'k'` / `mc = 'white'` / `mc = 'black'` |

All these are exactly `style.foreground` — which is already `'black'` in `default_style` and `'white'` in `dark_style`.

**No hardcoded black/white colors in plot calls.** Every `color='black'`, `color='white'`, `c='k'`, `c='w'` in scatter/plot/text calls should use `style.foreground` (or `style.background` where appropriate) instead.

### The solution

#### 10a. Dark/default theme via `FigureStyle.apply_globally()`

All `plt.style.use('dark_background')` / `plt.style.use('default')` calls should be replaced by `style.apply_globally()`. The two existing singletons (`default_style`, `dark_style`) already encode the right foreground/background colors. Add `plt.style.use('dark_background')` inside `dark_style.apply_globally()` and `plt.style.use('default')` inside `default_style.apply_globally()` so callers just do:

```python
style.apply_globally()
```

No caller should ever call `plt.style.use()` directly.

#### 10b. Add `style: FigureStyle` parameter to all plot functions

Every `def plot_*` and `def analyze_*` function should accept `style: FigureStyle = default_style` and use `style.font_size`, `style.tick_font_size`, `style.label_font_size`, `style.foreground`, `style.background` instead of hardcoded values.

For functions that need larger font sizes (e.g., the 32pt / 68pt in synaptic plots), use `style.font_size * scale_factor` for relative scaling.

#### 10c. Remove `analyze_data_svd` local font constants

Replace the local `TITLE_SIZE`, `LABEL_SIZE`, `TICK_SIZE`, `LEGEND_SIZE` with `style.font_size`, `style.label_font_size`, `style.tick_font_size`.

#### 10d. Eliminate `mc` variable — use `style.foreground`

Remove every `mc = 'k'` / `mc = 'w'` / `mc = 'black'` / `mc = 'white'` assignment. Remove `mc` from all function signatures. Replace all `color=mc`, `c=mc` with `color=style.foreground`, `c=style.foreground`. No hardcoded `'black'` or `'white'` color strings should remain in any plot/scatter/text call — always use `style.foreground` or `style.background`.

### Validation

```bash
python GNN_Test.py --config flyvis_62_1_gs --cluster
```
