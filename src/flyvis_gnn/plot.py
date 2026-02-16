"""Shared plotting and analysis functions for FlyVis.

Used by both the training loop (graph_trainer.py / utils.py) and
post-training analysis (GNN_PlotFigure.py).
"""
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import torch
from scipy.optimize import curve_fit

from flyvis_gnn.fitting_models import linear_model
from flyvis_gnn.utils import to_numpy


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def get_model_W(model):
    """Get the weight matrix from a model, handling low-rank factorization."""
    if hasattr(model, 'W'):
        return model.W
    elif hasattr(model, 'WL') and hasattr(model, 'WR'):
        return model.WL @ model.WR
    else:
        raise AttributeError("Model has neither 'W' nor 'WL'/'WR' attributes")


def compute_r_squared(true, learned):
    """Compute R² and linear fit slope between true and learned arrays."""
    lin_fit, _ = curve_fit(linear_model, true, learned)
    residuals = learned - linear_model(true, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((learned - np.mean(learned)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return r_squared, lin_fit[0]


# ------------------------------------------------------------------ #
#  Vectorized helpers
# ------------------------------------------------------------------ #

def _vectorized_linspace(starts, ends, n_pts, device):
    """Create (N, n_pts) tensor where row n spans [starts[n], ends[n]].

    Instead of calling torch.linspace N times, we parameterize with
    t in [0, 1] and broadcast:  rr[n, i] = start[n] + t[i] * (end[n] - start[n])
    """
    t = torch.linspace(0, 1, n_pts, device=device)                   # (n_pts,)
    starts_t = torch.as_tensor(starts, dtype=torch.float32, device=device)  # (N,)
    ends_t = torch.as_tensor(ends, dtype=torch.float32, device=device)      # (N,)
    return starts_t[:, None] + t[None, :] * (ends_t - starts_t)[:, None]    # (N, n_pts)


def _batched_mlp_eval(mlp, model_a, rr, build_features_fn,
                      device, chunk_size=2000, post_fn=None):
    """Evaluate an MLP for all neurons at once, in chunks.

    Instead of N individual forward passes on (1000, D) inputs, we
    stack all neurons into (N*1000, D) and run one pass per chunk.

    Args:
        mlp: nn.Module — the MLP to evaluate (model.lin_edge or model.lin_phi).
        model_a: (N, emb_dim) embedding tensor.
        rr: (N, n_pts) tensor of input values per neuron.
        build_features_fn: callable(rr_flat, emb_flat) -> (chunk*n_pts, D)
            Builds the MLP input features from flattened rr and embeddings.
        device: torch device.
        chunk_size: number of neurons per chunk (limits GPU memory).
        post_fn: optional callable applied to MLP output (e.g. lambda x: x**2).

    Returns:
        (N, n_pts) tensor of MLP outputs.
    """
    N, n_pts = rr.shape
    emb_dim = model_a.shape[1]
    results = []

    for i in range(0, N, chunk_size):
        chunk_rr = rr[i:i + chunk_size]                        # (C, n_pts)
        chunk_a = model_a[i:i + chunk_size]                     # (C, emb_dim)
        C = chunk_rr.shape[0]

        # Flatten: repeat each neuron's values n_pts times
        rr_flat = chunk_rr.reshape(-1, 1)                       # (C*n_pts, 1)
        emb_flat = chunk_a[:, None, :].expand(-1, n_pts, -1)    # (C, n_pts, emb_dim)
        emb_flat = emb_flat.reshape(-1, emb_dim)                 # (C*n_pts, emb_dim)

        in_features = build_features_fn(rr_flat, emb_flat)       # (C*n_pts, D)

        with torch.no_grad():
            out = mlp(in_features.float())                       # (C*n_pts, 1)
            if post_fn is not None:
                out = post_fn(out)

        results.append(out.squeeze(-1).reshape(C, n_pts))        # (C, n_pts)

    return torch.cat(results, dim=0)                              # (N, n_pts)


def _vectorized_linear_fit(x, y):
    """Vectorized least-squares linear regression across rows.

    Fits y[n] = slope[n] * x[n] + offset[n] for all N rows in parallel,
    replacing N individual scipy.curve_fit calls.

    Uses the closed-form solution:
        slope  = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
        offset = (Σy − slope·Σx) / n

    Args:
        x: (N, n_pts) numpy array or tensor.
        y: (N, n_pts) numpy array or tensor.

    Returns:
        slopes: (N,) numpy array.
        offsets: (N,) numpy array.
    """
    if isinstance(x, torch.Tensor):
        x = to_numpy(x)
    if isinstance(y, torch.Tensor):
        y = to_numpy(y)

    n_pts = x.shape[1]
    sx = x.sum(axis=1)
    sy = y.sum(axis=1)
    sxy = (x * y).sum(axis=1)
    sxx = (x * x).sum(axis=1)

    denom = n_pts * sxx - sx * sx
    # Guard against degenerate cases (constant x)
    safe = np.abs(denom) > 1e-12
    slopes = np.where(safe, (n_pts * sxy - sx * sy) / np.where(safe, denom, 1.0), 0.0)
    offsets = np.where(safe, (sy - slopes * sx) / n_pts, 0.0)

    return slopes, offsets


def _plot_curves_fast(ax, rr, func, type_list, cmap, linewidth=1, alpha=0.1):
    """Plot per-neuron curves using LineCollection (single draw call).

    Instead of N individual ax.plot() calls (high matplotlib overhead),
    build an (N, n_pts, 2) segments array and add one LineCollection.

    Args:
        ax: matplotlib Axes.
        rr: (N, n_pts) or (n_pts,) numpy array of x-values.
        func: (N, n_pts) numpy array of y-values.
        type_list: (N,) int array of neuron type indices.
        cmap: CustomColorMap with .color(int) method.
        linewidth: line width.
        alpha: transparency.
    """
    N, n_pts = func.shape

    # If rr is 1D (shared range), broadcast to (N, n_pts)
    if rr.ndim == 1:
        rr = np.broadcast_to(rr[None, :], (N, n_pts))

    # Build (N, n_pts, 2) segments array: each row is [(x0,y0), (x1,y1), ...]
    segments = np.stack([rr, func], axis=-1)                  # (N, n_pts, 2)

    # Build per-neuron RGBA color array
    type_np = np.asarray(type_list).astype(int).ravel()
    colors = [(*cmap.color(type_np[n])[:3], alpha) for n in range(N)]

    lc = LineCollection(segments, colors=colors, linewidths=linewidth)
    ax.add_collection(lc)
    ax.autoscale_view()


# ------------------------------------------------------------------ #
#  Feature-building helpers for the two MLPs
# ------------------------------------------------------------------ #

def _build_lin_edge_features(rr_flat, emb_flat, signal_model_name):
    """Build input features for lin_edge MLP."""
    if 'PDE_N9_B' in signal_model_name:
        return torch.cat([rr_flat * 0, rr_flat, emb_flat, emb_flat], dim=1)
    else:
        return torch.cat([rr_flat, emb_flat], dim=1)


def _build_lin_phi_features(rr_flat, emb_flat):
    """Build input features for lin_phi MLP: (v, embedding, msg=0, exc=0)."""
    zeros = torch.zeros_like(rr_flat)
    return torch.cat([rr_flat, emb_flat, zeros, zeros], dim=1)


# ------------------------------------------------------------------ #
#  Activity statistics
# ------------------------------------------------------------------ #

def compute_activity_stats(x_list, device=None):
    """Compute per-neuron mean and std of voltage activity.

    Args:
        x_list: list of NeuronTimeSeries (voltage field is (T, N) tensor).
        device: optional device override.

    Returns:
        mu_activity: (N,) tensor of per-neuron mean voltage.
        sigma_activity: (N,) tensor of per-neuron std voltage.
    """
    voltage = x_list[0].voltage  # (T, N), already on device if x_list was moved
    if device is not None:
        voltage = voltage.to(device)
    mu = voltage.mean(dim=0)
    sigma = voltage.std(dim=0)
    return mu, sigma


# ------------------------------------------------------------------ #
#  Slope extraction
# ------------------------------------------------------------------ #

def extract_lin_edge_slopes(model, config, n_neurons, mu_activity, sigma_activity, device):
    """Extract linear slope of lin_edge for each neuron j (vectorized).

    Evaluates lin_edge(a_j, v) over each neuron's activity range [mu-2σ, mu+2σ]
    in one batched forward pass, then fits all slopes with vectorized regression.

    Returns:
        slopes: (n_neurons,) numpy array of lin_edge slopes.
    """
    signal_model_name = config.graph_model.signal_model_name
    lin_edge_positive = config.graph_model.lin_edge_positive
    n_pts = 1000

    mu = np.asarray(mu_activity, dtype=np.float32)
    sigma = np.asarray(sigma_activity, dtype=np.float32)

    # Neurons where activity range includes positive values
    valid = (mu + sigma) > 0
    starts = np.maximum(mu - 2 * sigma, 0.0)
    ends = mu + 2 * sigma

    # For invalid neurons, set dummy range (won't be used)
    starts[~valid] = 0.0
    ends[~valid] = 1.0

    rr = _vectorized_linspace(starts, ends, n_pts, device)  # (N, n_pts)

    post_fn = (lambda x: x ** 2) if lin_edge_positive else None
    build_fn = lambda rr_f, emb_f: _build_lin_edge_features(rr_f, emb_f, signal_model_name)

    func = _batched_mlp_eval(model.lin_edge, model.a[:n_neurons], rr,
                             build_fn, device, post_fn=post_fn)  # (N, n_pts)

    slopes, _ = _vectorized_linear_fit(rr, func)

    # Invalid neurons get slope = 1.0
    slopes[~valid] = 1.0

    return slopes


def extract_lin_phi_slopes(model, config, n_neurons, mu_activity, sigma_activity, device):
    """Extract linear slope and offset of lin_phi for each neuron i (vectorized).

    Evaluates lin_phi(a_i, v_i, msg=0, exc=0) over each neuron's activity range
    in one batched forward pass, then fits all slopes/offsets with vectorized regression.

    Returns:
        slopes: (n_neurons,) numpy array — slope relates to 1/tau.
        offsets: (n_neurons,) numpy array — offset relates to V_rest.
    """
    n_pts = 1000
    mu = np.asarray(mu_activity, dtype=np.float32)
    sigma = np.asarray(sigma_activity, dtype=np.float32)

    starts = mu - 2 * sigma
    ends = mu + 2 * sigma

    rr = _vectorized_linspace(starts, ends, n_pts, device)  # (N, n_pts)

    func = _batched_mlp_eval(model.lin_phi, model.a[:n_neurons], rr,
                             lambda rr_f, emb_f: _build_lin_phi_features(rr_f, emb_f),
                             device)  # (N, n_pts)

    slopes, offsets = _vectorized_linear_fit(rr, func)

    return slopes, offsets


# ------------------------------------------------------------------ #
#  Gradient of lin_phi w.r.t. msg
# ------------------------------------------------------------------ #

def compute_grad_msg(model, in_features, config):
    """Compute d(lin_phi)/d(msg) for each neuron from a forward-pass in_features.

    Args:
        model: Signal_Propagation_FlyVis model.
        in_features: (N, D) tensor from model(..., return_all=True).
            Layout: [v(1), embedding(E), msg(1), excitation(1)].
        config: config object with graph_model.embedding_dim.

    Returns:
        grad_msg: (N,) tensor of gradients.
    """
    emb_dim = config.graph_model.embedding_dim
    v = in_features[:, 0:1].clone().detach()
    embedding = in_features[:, 1:1 + emb_dim].clone().detach()
    msg = in_features[:, 1 + emb_dim:2 + emb_dim].clone().detach()
    excitation = in_features[:, 2 + emb_dim:3 + emb_dim].clone().detach()

    msg.requires_grad_(True)
    in_features_grad = torch.cat([v, embedding, msg, excitation], dim=1)
    out = model.lin_phi(in_features_grad)

    grad = torch.autograd.grad(
        outputs=out,
        inputs=msg,
        grad_outputs=torch.ones_like(out),
        retain_graph=False,
        create_graph=False,
    )[0]

    return grad.squeeze().detach()


# ------------------------------------------------------------------ #
#  Corrected weights
# ------------------------------------------------------------------ #

def compute_corrected_weights(model, edges, slopes_lin_phi, slopes_lin_edge, grad_msg):
    """Compute corrected W_ij from raw W, slopes, and grad_msg.

    Formula:
        corrected_W_ij = -W_ij / slope_phi[i] * grad_msg[i] * slope_edge[j]

    Args:
        model: model with .W, .n_edges, .n_extra_null_edges attributes.
        edges: (2, E) edge index tensor.
        slopes_lin_phi: (N,) array/tensor of lin_phi slopes per neuron.
        slopes_lin_edge: (N,) array/tensor of lin_edge slopes per neuron.
        grad_msg: (N,) tensor of d(lin_phi)/d(msg) per neuron.

    Returns:
        corrected_W: (E, 1) tensor of corrected weights.
    """
    device = get_model_W(model).device

    # Convert to tensors if needed
    if not isinstance(slopes_lin_phi, torch.Tensor):
        slopes_lin_phi = torch.tensor(slopes_lin_phi, dtype=torch.float32, device=device)
    if not isinstance(slopes_lin_edge, torch.Tensor):
        slopes_lin_edge = torch.tensor(slopes_lin_edge, dtype=torch.float32, device=device)

    n_w = model.n_edges + model.n_extra_null_edges

    # Map edges to neuron indices (handles batched edges via modulo)
    target_neuron_ids = edges[1, :] % n_w   # i — post-synaptic
    prior_neuron_ids = edges[0, :] % n_w    # j — pre-synaptic

    slopes_phi_per_edge = slopes_lin_phi[target_neuron_ids]     # (E,)
    slopes_edge_per_edge = slopes_lin_edge[prior_neuron_ids]    # (E,)
    grad_msg_per_edge = grad_msg[target_neuron_ids]             # (E,)

    W = get_model_W(model)  # (E, 1)

    corrected_W = (-W
                   / slopes_phi_per_edge[:, None]
                   * grad_msg_per_edge.unsqueeze(1)
                   * slopes_edge_per_edge.unsqueeze(1))

    return corrected_W


def compute_all_corrected_weights(model, config, edges, x_list, device):
    """High-level: compute corrected W from model state and training data.

    Runs one forward pass on a sample frame to obtain in_features,
    extracts slopes from lin_edge and lin_phi, computes grad_msg,
    and applies the correction formula.

    Args:
        model: Signal_Propagation_FlyVis model.
        config: full config object.
        edges: (2, E) edge index tensor.
        x_list: list of NeuronTimeSeries (training data).
        device: torch device.

    Returns:
        corrected_W: (E, 1) tensor of corrected weights.
        slopes_lin_phi: (N,) numpy array.
        slopes_lin_edge: (N,) numpy array.
        offsets_lin_phi: (N,) numpy array.
    """
    n_neurons = model.a.shape[0]

    # 1. Activity statistics
    mu_activity, sigma_activity = compute_activity_stats(x_list, device)

    # 2. Slope extraction
    slopes_lin_edge = extract_lin_edge_slopes(
        model, config, n_neurons, mu_activity, sigma_activity, device)
    slopes_lin_phi, offsets_lin_phi = extract_lin_phi_slopes(
        model, config, n_neurons, mu_activity, sigma_activity, device)

    # 3. Forward pass on a sample frame to get in_features
    mid_frame = x_list[0].voltage.shape[0] // 2
    state = x_list[0].frame(mid_frame)
    data_id = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, in_features, _ = model(state, edges, data_id=data_id, return_all=True)
    if was_training:
        model.train()

    # 4. Gradient of lin_phi w.r.t. msg
    grad_msg = compute_grad_msg(model, in_features, config)

    # 5. Corrected weights
    corrected_W = compute_corrected_weights(
        model, edges, slopes_lin_phi, slopes_lin_edge, grad_msg)

    return corrected_W, slopes_lin_phi, slopes_lin_edge, offsets_lin_phi


# ------------------------------------------------------------------ #
#  Subplot functions — shared between training and GNN_PlotFigure
# ------------------------------------------------------------------ #

def plot_embedding(ax, model, type_list, n_types, cmap):
    """Plot embedding scatter colored by neuron type.

    Args:
        ax: matplotlib Axes.
        model: model with .a embedding tensor (N, emb_dim).
        type_list: (N,) tensor/array of integer type indices.
        n_types: number of neuron types.
        cmap: CustomColorMap with .color(int) method.
    """
    embedding = to_numpy(model.a)
    type_np = to_numpy(type_list).squeeze()

    for n in range(n_types):
        mask = (type_np == n)
        if np.any(mask):
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=cmap.color(n), s=6, alpha=0.25, edgecolors='none')

    ax.set_xlabel('$a_0$', fontsize=32)
    ax.set_ylabel('$a_1$', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


def plot_lin_phi(ax, model, config, n_neurons, type_list, cmap, device, step=20):
    """Plot lin_phi function curves colored by neuron type (vectorized).

    Evaluates all selected neurons in one batched MLP pass and draws
    all curves with a single LineCollection.
    """
    n_pts = 1000
    xlim = config.plotting.xlim

    # Select every step-th neuron
    neuron_ids = np.arange(0, n_neurons, step)
    n_sel = len(neuron_ids)

    # Shared x-range, expanded to (n_sel, n_pts)
    rr_1d = torch.linspace(xlim[0], xlim[1], n_pts, device=device)
    rr = rr_1d.unsqueeze(0).expand(n_sel, -1)  # (n_sel, n_pts)

    # Batched MLP evaluation
    func = _batched_mlp_eval(
        model.lin_phi, model.a[neuron_ids], rr,
        lambda rr_f, emb_f: _build_lin_phi_features(rr_f, emb_f),
        device)

    # Fast plot with LineCollection
    type_np = to_numpy(type_list).astype(int)
    _plot_curves_fast(ax, to_numpy(rr_1d), to_numpy(func),
                      type_np[neuron_ids], cmap, linewidth=1, alpha=0.2)

    ax.set_xlim(xlim)
    ax.set_ylim(config.plotting.ylim)
    ax.set_xlabel('$v_i$', fontsize=32)
    ax.set_ylabel(r'learned $\mathrm{MLP_0}(\mathbf{a}_i, v_i)$', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)


def plot_lin_edge(ax, model, config, n_neurons, type_list, cmap, device, step=20):
    """Plot lin_edge function curves colored by neuron type (vectorized).

    Evaluates all selected neurons in one batched MLP pass and draws
    all curves with a single LineCollection.
    """
    signal_model_name = config.graph_model.signal_model_name
    lin_edge_positive = config.graph_model.lin_edge_positive
    xlim = config.plotting.xlim
    n_pts = 1000

    neuron_ids = np.arange(0, n_neurons, step)
    n_sel = len(neuron_ids)

    rr_1d = torch.linspace(xlim[0], xlim[1], n_pts, device=device)
    rr = rr_1d.unsqueeze(0).expand(n_sel, -1)

    post_fn = (lambda x: x ** 2) if lin_edge_positive else None
    build_fn = lambda rr_f, emb_f: _build_lin_edge_features(rr_f, emb_f, signal_model_name)

    func = _batched_mlp_eval(
        model.lin_edge, model.a[neuron_ids], rr,
        build_fn, device, post_fn=post_fn)

    type_np = to_numpy(type_list).astype(int)
    _plot_curves_fast(ax, to_numpy(rr_1d), to_numpy(func),
                      type_np[neuron_ids], cmap, linewidth=1, alpha=0.2)

    ax.set_xlim(xlim)
    ax.set_ylim([-xlim[1] / 10, xlim[1] * 1.2])
    ax.set_xlabel('$v_j$', fontsize=32)
    ax.set_ylabel(r'learned $\mathrm{MLP_1}(\mathbf{a}_j, v_j)$', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)


def plot_weight_scatter(ax, gt_weights, learned_weights, corrected=False,
                        xlim=None, ylim=None, mc=None, scatter_size=0.5,
                        outlier_threshold=None):
    """Plot true vs learned weight scatter with R² and slope.

    Args:
        ax: matplotlib Axes.
        gt_weights: (E,) numpy array of ground truth weights.
        learned_weights: (E,) numpy array of learned (or corrected) weights.
        corrected: if True, use W* label; if False, use W label.
        xlim: optional (lo, hi) for x-axis.
        ylim: optional (lo, hi) for y-axis.
        mc: per-edge color array; if None, uses black.
        scatter_size: scatter point size (default 0.5).
        outlier_threshold: if set, remove points with |residual| > threshold.
    """
    if outlier_threshold is not None:
        residuals = learned_weights - gt_weights
        mask = np.abs(residuals) <= outlier_threshold
        true_in = gt_weights[mask]
        learned_in = learned_weights[mask]
        mc_in = mc[mask] if mc is not None else None
    else:
        true_in = gt_weights
        learned_in = learned_weights
        mc_in = mc

    r_squared, slope = compute_r_squared(true_in, learned_in)

    scatter_color = mc_in if mc_in is not None else 'k'
    ax.scatter(true_in, learned_in, s=scatter_size, c=scatter_color, alpha=0.1)
    ax.text(0.05, 0.95,
            f'$R^2$: {r_squared:.3f}\nslope: {slope:.2f}\nN: {len(true_in)}',
            transform=ax.transAxes, verticalalignment='top', fontsize=24)

    ylabel = r'learned $W_{ij}^*$' if corrected else r'learned $W_{ij}$'
    ax.set_xlabel(r'true $W_{ij}$', fontsize=32)
    ax.set_ylabel(ylabel, fontsize=32)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=24)

    return r_squared, slope


def plot_tau(ax, slopes_lin_phi, gt_taus, n_neurons, mc=None):
    """Plot learned tau vs ground truth tau.

    Args:
        ax: matplotlib Axes.
        slopes_lin_phi: (N,) numpy array of lin_phi slopes.
        gt_taus: (N,) tensor/array of ground truth taus.
        n_neurons: number of neurons.
        mc: color for scatter points.
    """
    learned_tau = np.where(slopes_lin_phi != 0, 1.0 / -slopes_lin_phi, 1.0)
    learned_tau = learned_tau[:n_neurons]
    learned_tau = np.clip(learned_tau, 0, 1)
    gt_taus_np = to_numpy(gt_taus[:n_neurons]) if torch.is_tensor(gt_taus) else np.asarray(gt_taus[:n_neurons])

    r_squared, slope = compute_r_squared(gt_taus_np, learned_tau)

    ax.scatter(gt_taus_np, learned_tau, c=mc, s=1, alpha=0.25)
    ax.text(0.05, 0.95,
            f'$R^2$: {r_squared:.3f}\nslope: {slope:.2f}\nN: {len(gt_taus_np)}',
            transform=ax.transAxes, verticalalignment='top', fontsize=24)
    ax.set_xlabel(r'true $\tau$', fontsize=32)
    ax.set_ylabel(r'learned $\tau$', fontsize=32)
    ax.set_xlim([0, 0.35])
    ax.set_ylim([0, 0.35])
    ax.tick_params(axis='both', which='major', labelsize=24)

    return r_squared


def plot_vrest(ax, slopes_lin_phi, offsets_lin_phi, gt_V_rest, n_neurons, mc=None):
    """Plot learned V_rest vs ground truth V_rest.

    Args:
        ax: matplotlib Axes.
        slopes_lin_phi: (N,) numpy array of lin_phi slopes.
        offsets_lin_phi: (N,) numpy array of lin_phi offsets.
        gt_V_rest: (N,) tensor/array of ground truth V_rest.
        n_neurons: number of neurons.
        mc: color for scatter points.
    """
    learned_V_rest = np.where(slopes_lin_phi != 0, -offsets_lin_phi / slopes_lin_phi, 1.0)
    gt_vr_np = to_numpy(gt_V_rest[:n_neurons]) if torch.is_tensor(gt_V_rest) else np.asarray(gt_V_rest[:n_neurons])

    r_squared, slope = compute_r_squared(gt_vr_np, learned_V_rest)

    ax.scatter(gt_vr_np, learned_V_rest, c=mc, s=1, alpha=0.25)
    ax.text(0.05, 0.95,
            f'$R^2$: {r_squared:.3f}\nslope: {slope:.2f}\nN: {len(gt_vr_np)}',
            transform=ax.transAxes, verticalalignment='top', fontsize=24)
    ax.set_xlabel(r'true $V_{rest}$', fontsize=32)
    ax.set_ylabel(r'learned $V_{rest}$', fontsize=32)
    ax.set_xlim([-0.05, 0.9])
    ax.set_ylim([-0.05, 0.9])
    ax.tick_params(axis='both', which='major', labelsize=24)

    return r_squared
