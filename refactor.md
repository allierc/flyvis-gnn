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
  - Other functions (`plot_signal`, `plot_synaptic_CElegans`, `plot_synaptic_zebra`) still use `load_simulation_data_raw` (raw numpy) — these are for non-flyvis datasets

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

**Removed code:**
- `plot_synaptic3` — deleted (unused PDE_N3 model)
- Movie generation functions (`movie_synaptic_flyvis`, `create_combined_movie`, `create_individual_movies`) — deleted. GNN_PlotFigure generates static figures only; movies are handled separately for visual stimuli reconstruction with SIREN
- Thin subplot wrappers (`create_weight_subplot`, `create_embedding_subplot`, etc.) — deleted, were only used by movie functions
- `analyze_model_functions` and `load_model_for_epoch` — deleted, were only used by movie functions
