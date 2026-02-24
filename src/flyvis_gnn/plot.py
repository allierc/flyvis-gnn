"""Shared plotting and analysis functions for FlyVis.

Used by both the training loop (graph_trainer.py / utils.py) and
post-training analysis (GNN_PlotFigure.py).
"""
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import torch
from scipy.optimize import curve_fit

from flyvis_gnn.fitting_models import linear_model
from flyvis_gnn.utils import to_numpy, graphs_data_path


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def plot_training_summary_panels(fig, log_dir):
    """Add embedding, weight comparison, edge function, and phi function panels to a summary figure.

    Finds the last saved training snapshot and loads the PNG images into subplots 2-5
    of a 2x3 grid figure.

    Args:
        fig: matplotlib Figure (expected 2x3 subplot layout, panel 1 already used for loss)
        log_dir: path to the training log directory
    """
    import glob
    import os
    import imageio

    embedding_files = glob.glob(f"{log_dir}/tmp_training/embedding/*.png")
    if not embedding_files:
        return

    last_file = max(embedding_files, key=os.path.getctime)
    filename = os.path.basename(last_file)
    last_epoch, last_N = filename.replace('.png', '').split('_')

    panels = [
        (2, f"{log_dir}/tmp_training/embedding/{last_epoch}_{last_N}.png", 'Embedding'),
        (3, f"{log_dir}/tmp_training/matrix/comparison_{last_epoch}_{last_N}.png", 'Weight Comparison'),
        (4, f"{log_dir}/tmp_training/function/MLP1/func_{last_epoch}_{last_N}.png", 'Edge Function'),
        (5, f"{log_dir}/tmp_training/function/MLP0/func_{last_epoch}_{last_N}.png", 'Phi Function'),
    ]
    for pos, path, title in panels:
        fig.add_subplot(2, 3, pos)
        img = imageio.imread(path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title, fontsize=12)

    # Panel 6: R² metrics trajectory
    metrics_log_path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    if os.path.exists(metrics_log_path):
        r2_iters, conn_vals, vrest_vals, tau_vals = [], [], [], []
        try:
            with open(metrics_log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('epoch'):
                        continue
                    parts = line.split(',')
                    r2_iters.append(int(parts[1]))
                    conn_vals.append(float(parts[2]))
                    vrest_vals.append(float(parts[3]) if len(parts) > 3 else 0.0)
                    tau_vals.append(float(parts[4]) if len(parts) > 4 else 0.0)
        except Exception:
            pass
        if conn_vals:
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.plot(r2_iters, conn_vals, color='#d62728', linewidth=1.5, marker='o', markersize=3, label='conn')
            ax6.plot(r2_iters, vrest_vals, color='#1f77b4', linewidth=1.5, marker='s', markersize=3, label='V_rest')
            ax6.plot(r2_iters, tau_vals, color='#2ca02c', linewidth=1.5, marker='^', markersize=3, label='tau')
            ax6.axhline(y=0.9, color='green', linestyle='--', alpha=0.4, linewidth=1)
            ax6.set_ylim(-0.05, 1.05)
            ax6.set_xlabel('iteration', fontsize=10)
            ax6.set_ylabel('R²', fontsize=10)
            ax6.set_title('R² Metrics', fontsize=12)
            ax6.legend(fontsize=8, loc='lower right')
            ax6.grid(True, alpha=0.3)


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
    if 'flyvis_B' in signal_model_name:
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

def compute_activity_stats(x_ts, device=None):
    """Compute per-neuron mean and std of voltage activity.

    Args:
        x_ts: NeuronTimeSeries (voltage field is (T, N) tensor).
        device: optional device override.

    Returns:
        mu_activity: (N,) tensor of per-neuron mean voltage.
        sigma_activity: (N,) tensor of per-neuron std voltage.
    """
    voltage = x_ts.voltage  # (T, N), already on device if x_ts was moved
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

    mu = to_numpy(mu_activity).astype(np.float32) if torch.is_tensor(mu_activity) else np.asarray(mu_activity, dtype=np.float32)
    sigma = to_numpy(sigma_activity).astype(np.float32) if torch.is_tensor(sigma_activity) else np.asarray(sigma_activity, dtype=np.float32)

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
    mu = to_numpy(mu_activity).astype(np.float32) if torch.is_tensor(mu_activity) else np.asarray(mu_activity, dtype=np.float32)
    sigma = to_numpy(sigma_activity).astype(np.float32) if torch.is_tensor(sigma_activity) else np.asarray(sigma_activity, dtype=np.float32)

    starts = mu - 2 * sigma
    ends = mu + 2 * sigma

    rr = _vectorized_linspace(starts, ends, n_pts, device)  # (N, n_pts)

    func = _batched_mlp_eval(model.lin_phi, model.a[:n_neurons], rr,
                             lambda rr_f, emb_f: _build_lin_phi_features(rr_f, emb_f),
                             device)  # (N, n_pts)

    slopes, offsets = _vectorized_linear_fit(rr, func)

    return slopes, offsets


def compute_dynamics_r2(model, x_ts, config, device, n_neurons):
    """Compute V_rest R² and tau R² during training (lightweight, no plots).

    Extracts learned V_rest and tau from lin_phi slopes/offsets and compares
    against ground truth V_i_rest.pt and taus.pt.

    Returns:
        (vrest_r2, tau_r2): tuple of float R² values.
    """
    gt_V_rest_tensor = torch.load(graphs_data_path(config.dataset, 'V_i_rest.pt'),
                                  map_location=device, weights_only=True)
    tau_path = graphs_data_path(config.dataset, 'taus.pt')
    if not os.path.exists(tau_path):
        tau_path = graphs_data_path(config.dataset, 'tau_i.pt')
    gt_tau_tensor = torch.load(tau_path, map_location=device, weights_only=True)
    gt_V_rest = to_numpy(gt_V_rest_tensor[:n_neurons])
    gt_tau = to_numpy(gt_tau_tensor[:n_neurons])

    mu, sigma = compute_activity_stats(x_ts, device)
    slopes, offsets = extract_lin_phi_slopes(model, config, n_neurons, mu, sigma, device)

    # V_rest = -offset / slope (x-intercept of linearized lin_phi)
    learned_V_rest = np.where(slopes != 0, -offsets / slopes, 1.0)[:n_neurons]
    # tau = 1 / (-slope)
    learned_tau = np.where(slopes != 0, 1.0 / -slopes, 1.0)[:n_neurons]
    learned_tau = np.clip(learned_tau, 0, 1)

    try:
        vrest_r2, _ = compute_r_squared(gt_V_rest, learned_V_rest)
    except Exception:
        vrest_r2 = 0.0
    try:
        tau_r2, _ = compute_r_squared(gt_tau, learned_tau)
    except Exception:
        tau_r2 = 0.0

    return vrest_r2, tau_r2


# ------------------------------------------------------------------ #
#  Gradient of lin_phi w.r.t. msg
# ------------------------------------------------------------------ #

def compute_grad_msg(model, in_features, config):
    """Compute d(lin_phi)/d(msg) for each neuron from a forward-pass in_features.

    Args:
        model: FlyVisGNN model.
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


def compute_all_corrected_weights(model, config, edges, x_ts, device, n_grad_frames=8):
    """High-level: compute corrected W from model state and training data.

    Extracts slopes from lin_edge and lin_phi, computes grad_msg averaged
    over multiple frames, and applies the correction formula.

    Args:
        model: FlyVisGNN model.
        config: full config object.
        edges: (2, E) edge index tensor.
        x_ts: NeuronTimeSeries (training data).
        device: torch device.
        n_grad_frames: number of frames to sample for grad_msg (default 100).

    Returns:
        corrected_W: (E, 1) tensor of corrected weights.
        slopes_lin_phi: (N,) numpy array.
        slopes_lin_edge: (N,) numpy array.
        offsets_lin_phi: (N,) numpy array.
    """
    n_neurons = model.a.shape[0]

    # 1. Activity statistics
    mu_activity, sigma_activity = compute_activity_stats(x_ts, device)

    # 2. Slope extraction
    slopes_lin_edge = extract_lin_edge_slopes(
        model, config, n_neurons, mu_activity, sigma_activity, device)
    slopes_lin_phi, offsets_lin_phi = extract_lin_phi_slopes(
        model, config, n_neurons, mu_activity, sigma_activity, device)

    # 3. Compute grad_msg over multiple frames and take median
    n_frames = x_ts.voltage.shape[0]
    frame_indices = np.linspace(n_frames // 10, n_frames - 100, n_grad_frames, dtype=int)
    data_id = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)

    was_training = model.training
    model.eval()

    grad_list = []
    for k in frame_indices:
        state = x_ts.frame(int(k))
        with torch.no_grad():
            _, in_features, _ = model(state, edges, data_id=data_id, return_all=True)
        grad_k = compute_grad_msg(model, in_features, config)
        grad_list.append(grad_k)

    if was_training:
        model.train()

    grad_msg = torch.stack(grad_list).median(dim=0).values  # (N,)

    # 4. Corrected weights
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
    model_config = config.graph_model
    n_pts = 1000

    neuron_ids = np.arange(0, n_neurons, step)
    n_sel = len(neuron_ids)

    rr_1d = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], n_pts, device=device)
    rr = rr_1d.unsqueeze(0).expand(n_sel, -1)

    post_fn = (lambda x: x ** 2) if model_config.lin_edge_positive else None
    build_fn = lambda rr_f, emb_f: _build_lin_edge_features(rr_f, emb_f, model_config.signal_model_name)

    func = _batched_mlp_eval(
        model.lin_edge, model.a[neuron_ids], rr,
        build_fn, device, post_fn=post_fn)

    type_np = to_numpy(type_list).astype(int)
    _plot_curves_fast(ax, to_numpy(rr_1d), to_numpy(func),
                      type_np[neuron_ids], cmap, linewidth=1, alpha=0.2)

    ax.set_xlim(config.plotting.xlim)
    ax.set_ylim([-config.plotting.xlim[1] / 10, config.plotting.xlim[1] * 1.2])
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
    ax.scatter(true_in, learned_in, s=scatter_size, c=scatter_color, alpha=0.06)
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


# ================================================================== #
#  CONSOLIDATED FROM generators/plots.py
# ================================================================== #

from typing import Optional, Dict
from flyvis_gnn.figure_style import FigureStyle, default_style

INDEX_TO_NAME: dict[int, str] = {
    0: 'am', 1: 'c2', 2: 'c3', 3: 'ct1(lo1)', 4: 'ct1(m10)',
    5: 'l1', 6: 'l2', 7: 'l3', 8: 'l4', 9: 'l5',
    10: 'lawf1', 11: 'lawf2', 12: 'mi1', 13: 'mi10', 14: 'mi11',
    15: 'mi12', 16: 'mi13', 17: 'mi14', 18: 'mi15', 19: 'mi2',
    20: 'mi3', 21: 'mi4', 22: 'mi9', 23: 'r1', 24: 'r2',
    25: 'r3', 26: 'r4', 27: 'r5', 28: 'r6', 29: 'r7', 30: 'r8',
    31: 't1', 32: 't2', 33: 't2a', 34: 't3', 35: 't4a',
    36: 't4b', 37: 't4c', 38: 't4d', 39: 't5a', 40: 't5b',
    41: 't5c', 42: 't5d', 43: 'tm1', 44: 'tm16', 45: 'tm2',
    46: 'tm20', 47: 'tm28', 48: 'tm3', 49: 'tm30', 50: 'tm4',
    51: 'tm5y', 52: 'tm5a', 53: 'tm5b', 54: 'tm5c', 55: 'tm9',
    56: 'tmy10', 57: 'tmy13', 58: 'tmy14', 59: 'tmy15',
    60: 'tmy18', 61: 'tmy3', 62: 'tmy4', 63: 'tmy5a', 64: 'tmy9',
}

ANATOMICAL_ORDER: list[Optional[int]] = [
    None, 23, 24, 25, 26, 27, 28, 29, 30,
    5, 6, 7, 8, 9, 10, 11, 12,
    19, 20, 21, 22,
    13, 14, 15, 16, 17, 18,
    43, 45, 48, 50, 44, 46, 47, 49, 51, 52, 53, 54, 55,
    61, 62, 63, 56, 57, 58, 59, 60, 64,
    1, 2, 4, 3,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    0,
]
def plot_spatial_activity_grid(
    positions: np.ndarray,
    voltages: np.ndarray,
    stimulus: np.ndarray,
    neuron_types: np.ndarray,
    output_path: str,
    calcium: Optional[np.ndarray] = None,
    n_input_neurons: Optional[int] = None,
    index_to_name: Optional[dict] = None,
    anatomical_order: Optional[list] = None,
    style: FigureStyle = default_style,
) -> None:
    """8x9 or 16x9 hex scatter grid of per-neuron-type spatial activity.

    Args:
        positions: (N, 2) spatial positions for hex scatter.
        voltages: (N,) voltage per neuron.
        stimulus: (n_input,) stimulus values for input neurons.
        neuron_types: (N,) integer neuron type per neuron.
        output_path: where to save the figure.
        calcium: (N,) calcium values (if not None, adds bottom 8 rows).
        n_input_neurons: number of input neurons (defaults to len(stimulus)).
        index_to_name: type index -> name mapping. Defaults to INDEX_TO_NAME.
        anatomical_order: panel ordering. Defaults to ANATOMICAL_ORDER.
        style: FigureStyle instance.
    """
    names = index_to_name or INDEX_TO_NAME
    order = anatomical_order or ANATOMICAL_ORDER
    n_inp = n_input_neurons or len(stimulus)
    include_calcium = calcium is not None

    n_cols = 9
    n_rows = 16 if include_calcium else 8
    panel_w, panel_h = 2.0, 1.8
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_w * n_cols, panel_h * n_rows),
        facecolor=style.background,
    )
    plt.subplots_adjust(hspace=1.2)
    axes_flat = axes.flatten()

    # hide trailing panels in voltage section
    n_panels = len(order)
    for i in range(n_panels, n_cols * 8):
        if i < len(axes_flat):
            axes_flat[i].set_visible(False)
    if include_calcium:
        for i in range(n_panels + n_cols * 8, len(axes_flat)):
            axes_flat[i].set_visible(False)

    vmin_v, vmax_v = style.hex_voltage_range
    vmin_s, vmax_s = style.hex_stimulus_range
    vmin_ca, vmax_ca = style.hex_calcium_range

    for panel_idx, type_idx in enumerate(order):
        # --- voltage panel ---
        ax_v = axes_flat[panel_idx]
        _draw_hex_panel(
            ax_v, type_idx, positions, voltages, stimulus,
            neuron_types, n_inp, names,
            cmap=style.cmap, vmin=vmin_v, vmax=vmax_v,
            stim_cmap=style.cmap, stim_vmin=vmin_s, stim_vmax=vmax_s,
            style=style,
        )

        # --- calcium panel (if present) ---
        if include_calcium:
            ax_ca = axes_flat[panel_idx + n_cols * 8]
            if type_idx is None:
                # stimulus panel (same as voltage section)
                ax_ca.scatter(
                    positions[:n_inp, 0], positions[:n_inp, 1],
                    s=style.hex_stimulus_marker_size, c=stimulus,
                    cmap=style.cmap, vmin=vmin_s, vmax=vmax_s,
                    marker=style.hex_marker, alpha=1.0, linewidths=0,
                )
                ax_ca.set_title(style._label('stimuli'), fontsize=style.font_size)
            else:
                mask = neuron_types == type_idx
                count = int(np.sum(mask))
                name = names.get(type_idx, f'type_{type_idx}')
                if count > 0:
                    ax_ca.scatter(
                        positions[:count, 0], positions[:count, 1],
                        s=style.hex_marker_size, c=calcium[mask],
                        cmap=style.cmap_calcium, vmin=vmin_ca, vmax=vmax_ca,
                        marker=style.hex_marker, alpha=1, linewidths=0,
                    )
                ax_ca.set_title(style._label(name), fontsize=style.font_size)
            ax_ca.set_facecolor(style.background)
            ax_ca.set_xticks([])
            ax_ca.set_yticks([])
            ax_ca.set_aspect('equal')
            for spine in ax_ca.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95 if not include_calcium else 0.92, bottom=0.05)
    style.savefig(fig, output_path)


def plot_kinograph(
    activity: np.ndarray,
    stimulus: np.ndarray,
    output_path: str,
    rank_90_act: int = 0,
    rank_99_act: int = 0,
    rank_90_inp: int = 0,
    rank_99_inp: int = 0,
    zoom_size: int = 200,
    zoom_neuron_start: int = 4900,
    style: FigureStyle = default_style,
) -> None:
    """2x2 kinograph: full activity + zoom, full stimulus + zoom.

    Args:
        activity: (n_neurons, n_frames) transposed voltage array.
        stimulus: (n_input_neurons, n_frames) transposed stimulus array.
        output_path: where to save the figure.
        rank_90_act: effective rank at 90% variance (activity).
        rank_99_act: effective rank at 99% variance (activity).
        rank_90_inp: effective rank at 90% variance (input).
        rank_99_inp: effective rank at 99% variance (input).
        zoom_size: size of zoom window in neurons and frames.
        zoom_neuron_start: first neuron index for the activity zoom panel.
        style: FigureStyle instance.
    """
    n_neurons, n_frames = activity.shape
    n_input, _ = stimulus.shape
    vmax_act = np.abs(activity).max()
    vmax_inp = np.abs(stimulus).max() * 1.2
    zoom_f = min(zoom_size, n_frames)
    zoom_n_act = min(zoom_size, n_neurons - zoom_neuron_start)
    zoom_n_inp = min(zoom_size, n_input)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(style.figure_height * 3.5, style.figure_height * 2.5),
        facecolor=style.background,
        gridspec_kw={'width_ratios': [2, 1]},
    )

    imshow_kw = dict(aspect='auto', cmap=style.cmap, origin='lower', interpolation='nearest')

    # top-left: full activity
    ax = axes[0, 0]
    im = ax.imshow(activity, vmin=-vmax_act, vmax=vmax_act, **imshow_kw)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=style.tick_font_size)
    style.ylabel(ax, 'neurons')
    style.xlabel(ax, 'time (frames)')
    ax.set_xticks([0, n_frames - 1])
    ax.set_xticklabels([0, n_frames], fontsize=style.tick_font_size)
    ax.set_yticks([0, n_neurons - 1])
    ax.set_yticklabels([1, n_neurons], fontsize=style.tick_font_size)
    style.annotate(ax, f'rank(90%)={rank_90_act}  rank(99%)={rank_99_act}', (0.02, 0.97), va='top', ha='left')

    # top-right: zoom activity
    ax = axes[0, 1]
    zoom_neuron_end = zoom_neuron_start + zoom_n_act
    im = ax.imshow(activity[zoom_neuron_start:zoom_neuron_end, :zoom_f], vmin=-vmax_act, vmax=vmax_act, **imshow_kw)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=style.tick_font_size)
    style.ylabel(ax, 'neurons')
    style.xlabel(ax, 'time (frames)')
    ax.set_xticks([0, zoom_f - 1])
    ax.set_xticklabels([0, zoom_f], fontsize=style.tick_font_size)
    ax.set_yticks([0, zoom_n_act - 1])
    ax.set_yticklabels([zoom_neuron_start, zoom_neuron_end], fontsize=style.tick_font_size)

    # bottom-left: full stimulus
    ax = axes[1, 0]
    im = ax.imshow(stimulus, vmin=-vmax_inp, vmax=vmax_inp, **imshow_kw)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=style.tick_font_size)
    style.ylabel(ax, 'input neurons')
    style.xlabel(ax, 'time (frames)')
    ax.set_xticks([0, stimulus.shape[1] - 1])
    ax.set_xticklabels([0, stimulus.shape[1]], fontsize=style.tick_font_size)
    ax.set_yticks([0, n_input - 1])
    ax.set_yticklabels([1, n_input], fontsize=style.tick_font_size)
    style.annotate(ax, f'rank(90%)={rank_90_inp}  rank(99%)={rank_99_inp}', (0.02, 0.97), va='top', ha='left')

    # bottom-right: zoom stimulus
    ax = axes[1, 1]
    im = ax.imshow(stimulus[:zoom_n_inp, :zoom_f], vmin=-vmax_inp, vmax=vmax_inp, **imshow_kw)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=style.tick_font_size)
    style.ylabel(ax, 'input neurons')
    style.xlabel(ax, 'time (frames)')
    ax.set_xticks([0, zoom_f - 1])
    ax.set_xticklabels([0, zoom_f], fontsize=style.tick_font_size)
    ax.set_yticks([0, zoom_n_inp - 1])
    ax.set_yticklabels([1, zoom_n_inp], fontsize=style.tick_font_size)

    plt.tight_layout()
    style.savefig(fig, output_path)


def plot_activity_traces(
    activity: np.ndarray,
    output_path: str,
    n_traces: int = 100,
    max_frames: int = 10000,
    n_input_neurons: int = 0,
    style: FigureStyle = default_style,
) -> None:
    """Sampled neuron voltage traces stacked vertically.

    Args:
        activity: (n_neurons, n_frames) transposed voltage array.
        output_path: where to save the figure.
        n_traces: number of neurons to sample.
        max_frames: truncate x-axis at this frame count.
        n_input_neurons: shown as annotation.
        style: FigureStyle instance.
    """
    n_neurons, n_frames = activity.shape
    n_traces = min(n_traces, n_neurons)
    sampled_idx = np.sort(np.random.choice(n_neurons, n_traces, replace=False))
    sampled = activity[sampled_idx]
    offset = sampled + 2 * np.arange(n_traces)[:, None]

    fig, ax = style.figure(aspect=1.5)
    ax.plot(offset.T, linewidth=0.5, alpha=0.7, color=style.foreground)
    style.xlabel(ax, 'time (frames)')
    style.ylabel(ax, f'{n_traces} / {n_neurons} neurons')
    ax.set_yticks([])
    ax.set_xlim([0, min(n_frames, max_frames)])
    ax.set_ylim([offset[0].min() - 2, offset[-1].max() + 2])

    plt.tight_layout()
    style.savefig(fig, output_path)


def plot_selected_neuron_traces(
    activity: np.ndarray,
    type_list: np.ndarray,
    output_path: str,
    selected_types: Optional[list[int]] = None,
    start_frame: int = 63000,
    end_frame: int = 63500,
    index_to_name: Optional[dict] = None,
    step_v: float = 1.5,
    style: FigureStyle = default_style,
) -> None:
    """Traces for specific neuron types over a time window.

    Args:
        activity: (n_neurons, n_frames) full activity array.
        type_list: (n_neurons,) integer neuron type per neuron.
        output_path: where to save the figure.
        selected_types: list of type indices to plot. Defaults to
            [l1, mi1, mi2, r1, t1, t4a, t5a, tm1, tm4, tm9].
        start_frame: start of time window.
        end_frame: end of time window.
        index_to_name: type index -> name mapping. Defaults to INDEX_TO_NAME.
        step_v: vertical offset between traces.
        style: FigureStyle instance.
    """
    names = index_to_name or INDEX_TO_NAME
    if selected_types is None:
        selected_types = [5, 12, 19, 23, 31, 35, 39, 43, 50, 55]

    # find one neuron per selected type
    neuron_indices = []
    for stype in selected_types:
        indices = np.where(type_list == stype)[0]
        if len(indices) > 0:
            neuron_indices.append(indices[0])

    n_sel = len(neuron_indices)
    if n_sel == 0:
        return

    true_slice = activity[neuron_indices, start_frame:end_frame]

    fig, ax = style.figure(aspect=1.5)
    for i in range(n_sel):
        baseline = np.mean(true_slice[i])
        ax.plot(true_slice[i] - baseline + i * step_v,
                linewidth=style.line_width, c='green', alpha=0.75)

    # neuron ids as y-tick labels
    ytick_positions = [i * step_v for i in range(n_sel)]
    ytick_labels = [names.get(selected_types[i], f'type_{selected_types[i]}') for i in range(n_sel)]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=style.tick_font_size)
    ax.set_ylim([-step_v, n_sel * step_v])
    style.ylabel(ax, 'neuron')

    ax.set_xticks([0, end_frame - start_frame])
    ax.set_xticklabels([start_frame, end_frame], fontsize=style.tick_font_size)
    style.xlabel(ax, 'frame')

    plt.tight_layout()
    style.savefig(fig, output_path)


# --------------------------------------------------------------------------- #
#  Private helpers
# --------------------------------------------------------------------------- #

def _draw_hex_panel(
    ax, type_idx, positions, voltages, stimulus, neuron_types,
    n_input_neurons, names, cmap, vmin, vmax,
    stim_cmap, stim_vmin, stim_vmax, style,
):
    """Draw a single hex scatter panel (voltage or stimulus)."""
    if type_idx is None:
        ax.scatter(
            positions[:n_input_neurons, 0], positions[:n_input_neurons, 1],
            s=style.hex_stimulus_marker_size, c=stimulus,
            cmap=stim_cmap, vmin=stim_vmin, vmax=stim_vmax,
            marker=style.hex_marker, alpha=1.0, linewidths=0,
        )
        ax.set_title(style._label('stimuli'), fontsize=style.font_size)
    else:
        mask = neuron_types == type_idx
        count = int(np.sum(mask))
        name = names.get(type_idx, f'type_{type_idx}')
        if count > 0:
            ax.scatter(
                positions[:count, 0], positions[:count, 1],
                s=style.hex_marker_size, c=voltages[mask],
                cmap=cmap, vmin=vmin, vmax=vmax,
                marker=style.hex_marker, alpha=1, linewidths=0,
            )
        ax.set_title(style._label(name), fontsize=style.font_size)

    ax.set_facecolor(style.background)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)


# ================================================================== #
#  CONSOLIDATED FROM generators/utils.py
# ================================================================== #

import seaborn as sns
from tifffile import imread

def plot_synaptic_frame_visual(X1, A1, H1, dataset_name, run, num):
    """Plot frame for visual field type."""
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.subplot(211)
    plt.axis("off")
    plt.title("$b_i$", fontsize=24)
    plt.scatter(
        to_numpy(X1[0:1024, 1]) * 0.95,
        to_numpy(X1[0:1024, 0]) * 0.95,
        s=15,
        c=to_numpy(A1[0:1024, 0]),
        cmap="viridis",
        vmin=0,
        vmax=2,
    )
    plt.scatter(
        to_numpy(X1[1024:, 1]) * 0.95 + 0.2,
        to_numpy(X1[1024:, 0]) * 0.95,
        s=15,
        c=to_numpy(A1[1024:, 0]),
        cmap="viridis",
        vmin=-4,
        vmax=4,
    )
    plt.xticks([])
    plt.yticks([])
    plt.subplot(212)
    plt.axis("off")
    plt.title("$x_i$", fontsize=24)
    plt.scatter(
        to_numpy(X1[0:1024, 1]),
        to_numpy(X1[0:1024, 0]),
        s=15,
        c=to_numpy(H1[0:1024, 0]),
        cmap="viridis",
        vmin=-10,
        vmax=10,
    )
    plt.scatter(
        to_numpy(X1[1024:, 1]) + 0.2,
        to_numpy(X1[1024:, 0]),
        s=15,
        c=to_numpy(H1[1024:, 0]),
        cmap="viridis",
        vmin=-10,
        vmax=10,
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(graphs_data_path(dataset_name, "Fig", f"Fig_{run}_{num}.png"), dpi=80)
    plt.close()


def plot_synaptic_frame_modulation(X1, A1, H1, dataset_name, run, num):
    """Plot frame for modulation field type."""
    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.scatter(
        to_numpy(X1[:, 1]),
        to_numpy(X1[:, 0]),
        s=100,
        c=to_numpy(A1[:, 0]),
        cmap="viridis",
        vmin=0,
        vmax=2,
    )
    plt.subplot(222)
    plt.scatter(
        to_numpy(X1[:, 1]),
        to_numpy(X1[:, 0]),
        s=100,
        c=to_numpy(H1[:, 0]),
        cmap="viridis",
        vmin=-5,
        vmax=5,
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(graphs_data_path(dataset_name, "Fig", f"Fig_{run}_{num}.png"), dpi=80)
    plt.close()


def plot_synaptic_frame_plasticity(X1, x, dataset_name, run, num):
    """Plot frame for PDE_N6/PDE_N7 with short term plasticity."""
    plt.figure(figsize=(12, 5.6))
    plt.axis("off")
    plt.subplot(121)
    plt.title("activity $x_i$", fontsize=24)
    plt.scatter(
        to_numpy(X1[:, 0]),
        to_numpy(X1[:, 1]),
        s=200,
        c=to_numpy(x[:, 3]),
        cmap="viridis",
        vmin=-5,
        vmax=5,
        edgecolors="k",
        alpha=1,
    )
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_tick_params(labelsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.title("short term plasticity $y_i$", fontsize=24)
    plt.scatter(
        to_numpy(X1[:, 0]),
        to_numpy(X1[:, 1]),
        s=200,
        c=to_numpy(x[:, 5]),
        cmap="grey",
        vmin=0,
        vmax=1,
        edgecolors="k",
        alpha=1,
    )
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_tick_params(labelsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(graphs_data_path(dataset_name, "Fig", f"Fig_{run}_{num}.tif"), dpi=170)
    plt.close()


def plot_synaptic_frame_default(X1, x, dataset_name, run, num):
    """Plot default frame for synaptic simulation."""
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.scatter(
        to_numpy(X1[:, 0]),
        to_numpy(X1[:, 1]),
        s=100,
        c=to_numpy(x[:, 3]),
        cmap="viridis",
        vmin=-40,
        vmax=40,
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(graphs_data_path(dataset_name, "Fig", f"Fig_{run}_{num}.tif"), dpi=170)
    plt.close()

    # Read back and create zoomed subplot
    im_ = imread(graphs_data_path(dataset_name, "Fig", f"Fig_{run}_{num}.tif"))
    plt.figure(figsize=(10, 10))
    plt.imshow(im_)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 1)
    plt.imshow(im_[800:1000, 800:1000, :])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(graphs_data_path(dataset_name, "Fig", f"Fig_{run}_{num}.png"), dpi=80)
    plt.close()


def plot_eigenvalue_spectrum(connectivity, dataset_name, mc='k', log_file=None):
    """Plot eigenvalue spectrum of connectivity matrix (3 panels)."""
    gt_weight = to_numpy(connectivity)
    eig_true, _ = np.linalg.eig(gt_weight)

    # Sort eigenvalues by magnitude
    idx_true = np.argsort(-np.abs(eig_true))
    eig_true_sorted = eig_true[idx_true]
    spectral_radius = np.max(np.abs(eig_true))

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # (0) eigenvalues in complex plane
    axes[0].scatter(eig_true.real, eig_true.imag, s=50, c=mc, alpha=0.7, edgecolors='none')
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    axes[0].set_xlabel('real', fontsize=32)
    axes[0].set_ylabel('imag', fontsize=32)
    axes[0].tick_params(labelsize=20)
    axes[0].set_title('eigenvalues in complex plane', fontsize=28)
    axes[0].text(0.05, 0.95, f'spectral radius: {spectral_radius:.3f}',
            transform=axes[0].transAxes, fontsize=20, verticalalignment='top')

    # (1) eigenvalue magnitude (sorted)
    axes[1].scatter(range(len(eig_true_sorted)), np.abs(eig_true_sorted), s=50, c=mc, alpha=0.7, edgecolors='none')
    axes[1].set_xlabel('index', fontsize=32)
    axes[1].set_ylabel('|eigenvalue|', fontsize=32)
    axes[1].tick_params(labelsize=20)
    axes[1].set_title('eigenvalue magnitude (sorted)', fontsize=28)

    # (2) eigenvalue spectrum (log scale)
    axes[2].plot(np.abs(eig_true_sorted), c=mc, linewidth=2)
    axes[2].set_xlabel('index', fontsize=32)
    axes[2].set_ylabel('|eigenvalue|', fontsize=32)
    axes[2].set_yscale('log')
    axes[2].tick_params(labelsize=20)
    axes[2].set_title('eigenvalue spectrum (log scale)', fontsize=28)

    plt.tight_layout()
    plt.savefig(graphs_data_path(dataset_name, "eigenvalues.png"), dpi=150)
    plt.close()

    msg = f'spectral radius: {spectral_radius:.3f}'
    print(msg)
    if log_file:
        log_file.write(msg + '\n')
    return spectral_radius


def plot_connectivity_matrix(connectivity, output_path, vmin_vmax_method='minmax',
                              percentile=99, vmin=None, vmax=None,
                              show_labels=True, show_title=True,
                              zoom_size=20, dpi=100, cbar_fontsize=16, label_fontsize=20):
    """Plot connectivity matrix heatmap with zoom inset.

    Args:
        connectivity: Connectivity matrix (torch tensor or numpy array)
        output_path: Path to save the figure
        vmin_vmax_method: 'minmax' for full range, 'percentile' for percentile-based
        percentile: Percentile value if vmin_vmax_method='percentile' (default: 99)
        vmin: Explicit vmin value (overrides vmin_vmax_method if provided)
        vmax: Explicit vmax value (overrides vmin_vmax_method if provided)
        show_labels: Whether to show x/y axis labels (default: True)
        show_title: Whether to show title (default: True)
        zoom_size: Size of zoom inset (top-left NxN block, default: 20)
        dpi: Output DPI (default: 100)
        cbar_fontsize: Colorbar tick font size (default: 32)
        label_fontsize: Axis label font size (default: 48)
    """
    gt_weight = to_numpy(connectivity)
    n_neurons = gt_weight.shape[0]

    # Use explicit vmin/vmax if provided, otherwise compute based on method
    if vmin is None or vmax is None:
        if vmin_vmax_method == 'percentile':
            weight_pct = np.percentile(np.abs(gt_weight.flatten()), percentile)
            vmin, vmax = -weight_pct * 1.1, weight_pct * 1.1
        else:  # minmax
            weight_max = np.max(np.abs(gt_weight))
            vmin, vmax = -weight_max, weight_max

    # Main heatmap
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(gt_weight, center=0, square=True, cmap='bwr',
                     cbar_kws={'fraction': 0.046}, vmin=vmin, vmax=vmax)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=cbar_fontsize)

    if show_labels:
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=label_fontsize)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=label_fontsize)
        plt.xticks(rotation=0)
    else:
        plt.xticks([])
        plt.yticks([])

    if show_title:
        plt.title('connectivity matrix', fontsize=20)

    # Zoom inset (top-left corner)
    if zoom_size > 0 and n_neurons >= zoom_size:
        plt.subplot(2, 2, 1)
        sns.heatmap(gt_weight[0:zoom_size, 0:zoom_size], cbar=False,
                    center=0, square=True, cmap='bwr', vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_low_rank_connectivity(connectivity, U, V, output_path, dpi=300):
    """Plot 3-panel figure: W, U, V for low-rank connectivity (W = U @ V).

    Args:
        connectivity: W matrix (torch tensor or numpy array), shape (n, n)
        U: left factor, shape (n, rank)
        V: right factor, shape (rank, n)
        output_path: path to save figure
        dpi: output DPI
    """
    W = to_numpy(connectivity)
    U = to_numpy(U)
    V = to_numpy(V)

    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # W panel
    weight_max = np.max(np.abs(W))
    im0 = axes[0].imshow(W, cmap='bwr', vmin=-weight_max, vmax=weight_max, aspect='auto')
    axes[0].set_title('W = U V', fontsize=20)
    axes[0].set_xlabel('post', fontsize=16)
    axes[0].set_ylabel('pre', fontsize=16)
    axes[0].tick_params(labelsize=12)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # U panel
    u_max = np.max(np.abs(U))
    im1 = axes[1].imshow(U, cmap='bwr', vmin=-u_max, vmax=u_max, aspect='auto')
    axes[1].set_title(f'U  ({U.shape[0]} x {U.shape[1]})', fontsize=20)
    axes[1].set_xlabel('rank', fontsize=16)
    axes[1].set_ylabel('pre', fontsize=16)
    axes[1].tick_params(labelsize=12)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # V panel
    v_max = np.max(np.abs(V))
    im2 = axes[2].imshow(V, cmap='bwr', vmin=-v_max, vmax=v_max, aspect='auto')
    axes[2].set_title(f'V  ({V.shape[0]} x {V.shape[1]})', fontsize=20)
    axes[2].set_xlabel('post', fontsize=16)
    axes[2].set_ylabel('rank', fontsize=16)
    axes[2].tick_params(labelsize=12)
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_signal_loss(loss_dict, log_dir, epoch=None, Niter=None, debug=False,
                     current_loss=None, current_regul=None, total_loss=None,
                     total_loss_regul=None):
    """
    Plot stratified loss components over training iterations.

    Creates a three-panel figure showing loss and regularization terms in both
    linear and log scale, plus connectivity R2 trajectory. Saves to {log_dir}/tmp_training/loss.tif.

    Parameters:
    -----------
    loss_dict : dict
        Dictionary containing loss component lists with keys:
        - 'loss': Loss without regularization
        - 'regul_total': Total regularization loss
        - 'W_L1': W L1 sparsity penalty
        - 'W_L2': W L2 regularization penalty
        - 'edge_diff': Edge monotonicity penalty
        - 'edge_norm': Edge normalization
        - 'edge_weight': Edge MLP weight regularization
        - 'phi_weight': Phi MLP weight regularization
        - 'W_sign': W sign consistency penalty
    log_dir : str
        Directory to save the figure
    epoch : int, optional
        Current epoch number
    Niter : int, optional
        Number of iterations per epoch
    debug : bool, optional
        If True, print debug information about loss components
    current_loss : float, optional
        Current iteration total loss (for debug)
    current_regul : float, optional
        Current iteration regularization (for debug)
    total_loss : float, optional
        Accumulated total loss (for debug)
    total_loss_regul : float, optional
        Accumulated regularization loss (for debug)
    """
    if len(loss_dict['loss']) == 0:
        return

    # Debug output if requested
    if debug and current_loss is not None and current_regul is not None:
        current_pred_loss = current_loss - current_regul

        # Get current iteration component values (last element in each list)
        comp_sum = (loss_dict['W_L1'][-1] + loss_dict['W_L2'][-1] +
                   loss_dict['edge_diff'][-1] + loss_dict['edge_norm'][-1] +
                   loss_dict['edge_weight'][-1] + loss_dict['phi_weight'][-1] +
                   loss_dict['W_sign'][-1])

        print(f"\n=== DEBUG Loss Components (Epoch {epoch}, Iter {Niter}) ===")
        print("Current iteration:")
        print(f"  loss.item() (total): {current_loss:.6f}")
        print(f"  regul_this_iter: {current_regul:.6f}")
        print(f"  prediction_loss (loss - regul): {current_pred_loss:.6f}")
        print("\nRegularization breakdown:")
        print(f"  W_L1: {loss_dict['W_L1'][-1]:.6f}")
        print(f"  W_L2: {loss_dict['W_L2'][-1]:.6f}")
        print(f"  W_sign: {loss_dict['W_sign'][-1]:.6f}")
        print(f"  edge_diff: {loss_dict['edge_diff'][-1]:.6f}")
        print(f"  edge_norm: {loss_dict['edge_norm'][-1]:.6f}")
        print(f"  edge_weight: {loss_dict['edge_weight'][-1]:.6f}")
        print(f"  phi_weight: {loss_dict['phi_weight'][-1]:.6f}")
        print(f"  Sum of components: {comp_sum:.6f}")
        if total_loss is not None and total_loss_regul is not None:
            print("\nAccumulated (for reference):")
            print(f"  total_loss (accumulated): {total_loss:.6f}")
            print(f"  total_loss_regul (accumulated): {total_loss_regul:.6f}")
        if current_loss > 0:
            print(f"\nRatio: regul / loss (current iter) = {current_regul / current_loss:.4f}")
        if current_pred_loss < 0:
            print("\n⚠️  WARNING: Negative prediction loss! regul > total loss")
        print("="*60)

    style = default_style
    lw = style.line_width
    fig_loss, (ax1, ax2, ax3) = style.figure(ncols=3, width=3 * style.figure_height * style.default_aspect)

    # epoch / iteration annotation
    info_text = ""
    if epoch is not None:
        info_text += f"epoch: {epoch}"
    if Niter is not None:
        if info_text:
            info_text += " | "
        info_text += f"iterations/epoch: {Niter}"
    if info_text:
        style.annotate(ax1, info_text, (0.02, 0.98), verticalalignment='top')

    # Linear scale
    legend_fs = 7
    ax1.plot(loss_dict['loss'], color='b', linewidth=1, label='loss (no regul)', alpha=0.8)
    ax1.plot(loss_dict['regul_total'], color='b', linewidth=1, label='total regularization', alpha=0.8)
    ax1.plot(loss_dict['W_L1'], color='r', linewidth=1, label='w l1 sparsity', alpha=0.7)
    ax1.plot(loss_dict['W_L2'], color='darkred', linewidth=1, label='w l2 regul', alpha=0.7)
    ax1.plot(loss_dict['W_sign'], color='navy', linewidth=1, label='w sign (dale)', alpha=0.7)
    ax1.plot(loss_dict['phi_weight'], color='lime', linewidth=1, label=r'$\phi$ weight regul', alpha=0.7)
    ax1.plot(loss_dict['edge_diff'], color='orange', linewidth=1, label='edge monotonicity', alpha=0.7)
    ax1.plot(loss_dict['edge_norm'], color='brown', linewidth=1, label='edge norm', alpha=0.7)
    ax1.plot(loss_dict['edge_weight'], color='pink', linewidth=1, label='edge weight regul', alpha=0.7)
    ax1.set_xlabel('iteration', fontsize=style.label_font_size - 2)
    ax1.set_ylabel('loss', fontsize=style.label_font_size - 2)
    ax1.tick_params(labelsize=style.tick_font_size - 2)
    ax1.legend(fontsize=legend_fs, loc='best', ncol=2)

    # Log scale
    ax2.plot(loss_dict['loss'], color='b', linewidth=1, label='loss (no regul)', alpha=0.8)
    ax2.plot(loss_dict['regul_total'], color='b', linewidth=1, label='total regularization', alpha=0.8)
    ax2.plot(loss_dict['W_L1'], color='r', linewidth=1, label='w l1 sparsity', alpha=0.7)
    ax2.plot(loss_dict['W_L2'], color='darkred', linewidth=1, label='w l2 regul', alpha=0.7)
    ax2.plot(loss_dict['W_sign'], color='navy', linewidth=1, label='w sign (dale)', alpha=0.7)
    ax2.plot(loss_dict['phi_weight'], color='lime', linewidth=1, label=r'$\phi$ weight regul', alpha=0.7)
    ax2.plot(loss_dict['edge_diff'], color='orange', linewidth=1, label='edge monotonicity', alpha=0.7)
    ax2.plot(loss_dict['edge_norm'], color='brown', linewidth=1, label='edge norm', alpha=0.7)
    ax2.plot(loss_dict['edge_weight'], color='pink', linewidth=1, label='edge weight regul', alpha=0.7)
    ax2.set_xlabel('iteration', fontsize=style.label_font_size - 2)
    ax2.set_ylabel('loss', fontsize=style.label_font_size - 2)
    ax2.tick_params(labelsize=style.tick_font_size - 2)
    ax2.set_yscale('log')
    ax2.legend(fontsize=legend_fs, loc='best', ncol=2)

    # R2 metrics panel (conn, V_rest, tau)
    metrics_log_path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    if os.path.exists(metrics_log_path):
        r2_iters, conn_vals, vrest_vals, tau_vals = [], [], [], []
        try:
            with open(metrics_log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('epoch'):
                        continue
                    parts = line.split(',')
                    ep = int(parts[0])
                    it = int(parts[1])
                    global_iter = ep * (Niter if Niter else 0) + it
                    r2_iters.append(global_iter)
                    conn_vals.append(float(parts[2]))
                    vrest_vals.append(float(parts[3]) if len(parts) > 3 else 0.0)
                    tau_vals.append(float(parts[4]) if len(parts) > 4 else 0.0)
        except Exception:
            pass
        if conn_vals:
            ax3.plot(r2_iters, conn_vals, color='#d62728', linewidth=1,
                     label=r'connectivity $R^2$')
            ax3.plot(r2_iters, vrest_vals, color='#1f77b4', linewidth=1,
                     label=r'$V_{rest}$ $R^2$')
            ax3.plot(r2_iters, tau_vals, color='#2ca02c', linewidth=1,
                     label=r'$\tau$ $R^2$')
            ax3.axhline(y=0.9, color='green', linestyle='--', alpha=0.4, linewidth=1)
            ax3.set_ylim(-0.05, 1.05)
            style.xlabel(ax3, 'iteration')
            ax3.set_ylabel(r'$R^2$', fontsize=style.label_font_size)
            ax3.legend(fontsize=legend_fs, loc='lower right')
            # most recent R2 values
            latest_text = (f"conn={conn_vals[-1]:.3f}\n"
                           f"vrest={vrest_vals[-1]:.3f}\n"
                           f"tau={tau_vals[-1]:.3f}")
            ax3.text(0.02, 0.97, latest_text, transform=ax3.transAxes,
                     fontsize=style.annotation_font_size, verticalalignment='top')
        else:
            ax3.text(0.5, 0.5, 'no r\u00b2 data yet', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=style.label_font_size, color='gray')
    else:
        ax3.text(0.5, 0.5, 'no r\u00b2 data yet', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=style.label_font_size, color='gray')

    style.savefig(fig_loss, f'{log_dir}/tmp_training/loss.tif')
    plt.close()



# ================================================================== #
#  CONSOLIDATED FROM models/utils.py
# ================================================================== #

def plot_training_flyvis(x_ts, model, config, epoch, N, log_dir, device, type_list,
                         gt_weights, edges, n_neurons=None, n_neuron_types=None):
    from flyvis_gnn.plot import (
        plot_embedding, plot_lin_edge, plot_lin_phi, plot_weight_scatter,
        compute_all_corrected_weights, get_model_W,
    )
    from flyvis_gnn.utils import CustomColorMap

    if n_neurons is None:
        n_neurons = len(type_list)

    cmap = CustomColorMap(config=config)

    # Plot 1: Embedding scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_embedding(ax, model, type_list, n_neuron_types, cmap)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/embedding/{epoch}_{N}.png", dpi=87)
    plt.close()

    # Plot 2: Raw W scatter (no correction)
    fig, ax = plt.subplots(figsize=(8, 8))
    raw_W = to_numpy(get_model_W(model).squeeze())
    r_squared_raw, _ = plot_weight_scatter(
        ax,
        gt_weights=to_numpy(gt_weights),
        learned_weights=raw_W,
        corrected=False,
        outlier_threshold=5,
    )
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/matrix/raw_{epoch}_{N}.png",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Compute corrected weights
    corrected_W, _, _, _ = compute_all_corrected_weights(
        model, config, edges, x_ts, device)

    # Plot 3: Corrected weight comparison scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    r_squared, _ = plot_weight_scatter(
        ax,
        gt_weights=to_numpy(gt_weights),
        learned_weights=to_numpy(corrected_W.squeeze()),
        corrected=True,
        xlim=[-1, 2],
        ylim=[-1, 2],
        outlier_threshold=5,
    )
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/matrix/comparison_{epoch}_{N}.png",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Plot 4: Edge function visualization (lin_edge / MLP1)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_lin_edge(ax, model, config, n_neurons, type_list, cmap, device)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/function/MLP1/func_{epoch}_{N}.png", dpi=87)
    plt.close()

    # Plot 5: Phi function visualization (lin_phi / MLP0)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_lin_phi(ax, model, config, n_neurons, type_list, cmap, device)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/function/MLP0/func_{epoch}_{N}.png", dpi=87)
    plt.close()

    return r_squared

def plot_odor_heatmaps(odor_responses):
    """
    Plot 3 separate heatmaps showing mean response per neuron for each odor
    """
    odor_list = ['butanone', 'pentanedione', 'NaCL']
    n_neurons = odor_responses['butanone'].shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, odor in enumerate(odor_list):
        # Compute mean response per neuron
        mean_responses = torch.mean(odor_responses[odor], dim=0).numpy()  # [n_neurons]

        # Reshape to 2D for heatmap (assuming square-ish layout)
        side_length = int(np.ceil(np.sqrt(n_neurons)))
        padded_responses = np.pad(mean_responses, (0, side_length ** 2 - n_neurons), 'constant')
        response_matrix = padded_responses.reshape(side_length, side_length)

        # Plot heatmap
        sns.heatmap(response_matrix, ax=axes[i], cmap='bwr', center=0,
                    cbar=False, square=True, xticklabels=False, yticklabels=False)
        axes[i].set_title(f'{odor} mean response')

    plt.tight_layout()
    return fig

def plot_weight_comparison(w_true, w_modified, output_path, xlabel='true $W$', ylabel='modified $W$', color='white'):
    w_true_np = w_true.detach().cpu().numpy().flatten()
    w_modified_np = w_modified.detach().cpu().numpy().flatten()
    plt.figure(figsize=(8, 8))
    plt.scatter(w_true_np, w_modified_np, s=8, alpha=0.5, color=color, edgecolors='none')
    # Fit linear model
    lin_fit, _ = curve_fit(linear_model, w_true_np, w_modified_np)
    slope = lin_fit[0]
    lin_fit[1]
    # R2 calculation
    residuals = w_modified_np - linear_model(w_true_np, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((w_modified_np - np.mean(w_modified_np)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    # Plot identity line
    plt.plot([w_true_np.min(), w_true_np.max()], [w_true_np.min(), w_true_np.max()], 'r--', linewidth=2, label='identity')
    # Add text
    plt.text(w_true_np.min(), w_true_np.max(), f'$R^2$: {r_squared:.3f}\nslope: {slope:.2f}', fontsize=18, va='top', ha='left')
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return slope, r_squared


# ================================================================== #
#  CONSOLIDATED FROM models/plot_utils.py
# ================================================================== #

from tqdm import trange
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def get_neuron_index(neuron_name, activity_neuron_list):
    """
    Returns the index of the neuron_name in activity_neuron_list.
    Raises ValueError if not found.
    """
    try:
        return activity_neuron_list.index(neuron_name)
    except ValueError:
        raise ValueError(f"Neuron '{neuron_name}' not found in activity_neuron_list.")


def analyze_mlp_edge_lines(model, neuron_list, all_neuron_list, adjacency_matrix, signal_range=(0, 10), resolution=100,
                           device=None):
    """
    Create line plots showing edge function vs signal difference for neuron pairs
    Uses adjacency matrix to find all connected neurons for each neuron of interest
    Plots mean and standard deviation across all connections

    Args:
        model: The trained model with embeddings and lin_edge
        neuron_list: List of neuron names of interest (1-5 neurons)
        all_neuron_list: Complete list of all 300 neuron names
        adjacency_matrix: 2D array (300x300) where adjacency_matrix[i,j] = 1 if i->j connection exists
        signal_range: Tuple of (min_signal, max_signal)
        resolution: Number of points for signal difference sampling
        device: PyTorch device

    Returns:
        fig_lines: Figure with line plots showing mean ± std for each neuron of interest
    """

    embedding = model.a  # Shape: (300, 2)

    print(f"generating line plots for {len(neuron_list)} neurons using adjacency matrix connections...")

    # Get indices of the neurons of interest
    neuron_indices_of_interest = []
    for neuron_name in neuron_list:
        try:
            neuron_idx = get_neuron_index(neuron_name, all_neuron_list)
            neuron_indices_of_interest.append(neuron_idx)
        except ValueError as e:
            print(f"Warning: {e}")
            continue

    if len(neuron_indices_of_interest) == 0:
        raise ValueError("No valid neurons found in neuron_list")

    # Create signal difference array for line plots
    u_diff_line = torch.linspace(-signal_range[1], signal_range[1], resolution * 2 - 1, device=device)

    # For each neuron of interest, find all its connections and compute statistics
    neuron_stats = {}

    for neuron_idx, neuron_id in enumerate(neuron_indices_of_interest):
        neuron_name = neuron_list[neuron_idx]
        receiver_embedding = embedding[neuron_id]  # This neuron as receiver (embedding_i)

        # Find all connected senders (where adjacency_matrix[receiver, sender] = 1)
        connected_senders = np.where(adjacency_matrix[neuron_id, :] == 1)[0]

        if len(connected_senders) == 0:
            print(f"Warning: No incoming connections found for {neuron_name}")
            continue

        # print(f"Found {len(connected_senders)} incoming connections for {neuron_name}")
        # Store outputs for all connections to this receiver
        connection_outputs = torch.zeros(len(connected_senders), len(u_diff_line), device=device)

        for conn_idx, sender_id in enumerate(connected_senders):
            sender_embedding = embedding[sender_id]  # Connected neuron as sender (embedding_j)

            line_inputs = []
            for diff_idx, diff in enumerate(u_diff_line):
                # Create signal pairs that span the valid range
                u_center = (signal_range[0] + signal_range[1]) / 2
                u_i = torch.clamp(u_center - diff / 2, signal_range[0], signal_range[1])
                u_j = torch.clamp(u_center + diff / 2, signal_range[0], signal_range[1])

                # Ensure the actual difference matches what we want
                actual_diff = u_j - u_i
                if abs(actual_diff - diff) > 1e-6:
                    # Adjust to get the exact difference we want
                    u_i = torch.clamp(u_center - diff / 2, signal_range[0], signal_range[1])
                    u_j = u_i + diff
                    if u_j > signal_range[1]:
                        u_j = torch.tensor(signal_range[1], device=device)
                        u_i = u_j - diff
                    elif u_j < signal_range[0]:
                        u_j = torch.tensor(signal_range[0], device=device)
                        u_i = u_j - diff

                # Create input feature vector: [u_i, u_j, embedding_i, embedding_j]
                in_features = torch.cat([
                    u_i.unsqueeze(0),  # u_i as (1,)
                    u_j.unsqueeze(0),  # u_j as (1,)
                    receiver_embedding,  # embedding_i (receiver) as (2,)
                    sender_embedding  # embedding_j (sender) as (2,)
                ], dim=0)  # Final shape: (6,)
                line_inputs.append(in_features)

            line_features = torch.stack(line_inputs, dim=0)  # (len(u_diff_line), 6)

            with torch.no_grad():
                lin_edge = model.lin_edge(line_features)
                if model.lin_edge_positive:
                    lin_edge = lin_edge ** 2

            connection_outputs[conn_idx] = lin_edge.squeeze(-1)

        # Compute mean and std across all connections to this receiver
        mean_output = torch.mean(connection_outputs, dim=0).cpu().numpy()
        std_output = torch.std(connection_outputs, dim=0).cpu().numpy()

        neuron_stats[neuron_name] = {
            'mean': mean_output,
            'std': std_output,
            'n_connections': len(connected_senders)
        }

    # Create line plot figure
    fig_lines, ax_lines = plt.subplots(1, 1, figsize=(14, 8))

    # Generate colors for each neuron of interest
    colors = plt.cm.tab10(np.linspace(0, 1, len(neuron_stats)))
    u_diff_line_np = u_diff_line.cpu().numpy()

    for neuron_idx, (neuron_name, stats) in enumerate(neuron_stats.items()):
        color = colors[neuron_idx]
        mean_vals = stats['mean']
        std_vals = stats['std']
        n_conn = stats['n_connections']

        # Plot mean line
        ax_lines.plot(u_diff_line_np, mean_vals,
                      color=color, linewidth=2,
                      label=f'{neuron_name} (n={n_conn})')

        # Plot standard deviation as shaded area
        ax_lines.fill_between(u_diff_line_np,
                              mean_vals - std_vals,
                              mean_vals + std_vals,
                              color=color, alpha=0.2)

    ax_lines.set_xlabel('u_j - u_i (signal difference)')
    ax_lines.set_ylabel('edge function output')
    ax_lines.set_title('edge function vs signal difference\n(mean ± std across incoming connections)')
    # grid(True, alpha=0.3)

    # Adaptive legend placement based on number of neurons
    n_neurons = len(neuron_stats)
    if n_neurons <= 20:
        # For few neurons, use right side
        ax_lines.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    return fig_lines


def analyze_mlp_edge_lines_weighted_with_max(model, neuron_name, all_neuron_list, adjacency_matrix, weight_matrix,
                                             signal_range=(0, 10), resolution=100, device=None):
    """
    Create line plots showing weighted edge function vs signal difference for a single neuron of interest
    Uses adjacency matrix to find connections and weight matrix to scale the outputs
    Plots individual lines for each incoming connection
    Returns the connection with maximum response in signal difference range [8, 10]

    Args:
        model: The trained model with embeddings and lin_edge
        neuron_name: Single neuron name of interest
        all_neuron_list: Complete list of all 300 neuron names
        adjacency_matrix: 2D array (300x300) where adjacency_matrix[i,j] = 1 if i->j connection exists
        weight_matrix: 2D array (300x300) with connection weights to scale edge function output
        signal_range: Tuple of (min_signal, max_signal) for DF/F0 measurements
        resolution: Number of points for signal difference sampling
        device: PyTorch device

    Returns:
        fig_lines: Figure with individual weighted line plots for each connection
        max_response_data: Dict with info about the connection with maximum response in [8,10] range
    """

    embedding = model.a  # Shape: (300, 2)

    # print(f"generating weighted line plots for {neuron_name} using adjacency and weight matrices...")

    # Get index of the neuron of interest
    try:
        neuron_id = get_neuron_index(neuron_name, all_neuron_list)
    except ValueError as e:
        raise ValueError(f"Neuron '{neuron_name}' not found: {e}")

    receiver_embedding = embedding[neuron_id]  # This neuron as receiver (embedding_i)

    # Find all connected senders (where adjacency_matrix[receiver, sender] = 1)
    connected_senders = np.where(adjacency_matrix[neuron_id, :] == 1)[0]
    #
    # if len(connected_senders) == 0:
    #     print(f"No incoming connections found for {neuron_name}")
    #     return None, None

    # print(f"Found {len(connected_senders)} incoming connections for {neuron_name}")

    # Create signal difference array for line plots
    u_diff_line = torch.linspace(-signal_range[1], signal_range[1], resolution * 2 - 1, device=device)
    u_diff_line_np = u_diff_line.cpu().numpy()

    # Find indices corresponding to signal difference range [8, 10]
    target_range_mask = (u_diff_line_np >= 8.0) & (u_diff_line_np <= 10.0)
    target_indices = np.where(target_range_mask)[0]

    # Store outputs and metadata for all connections
    connection_data = []
    max_response = -float('inf')
    max_response_data = None

    for sender_id in connected_senders:
        sender_name = all_neuron_list[sender_id]
        sender_embedding = embedding[sender_id]  # Connected neuron as sender (embedding_j)
        connection_weight = weight_matrix[neuron_id, sender_id]  # Weight for this connection

        line_inputs = []
        for diff_idx, diff in enumerate(u_diff_line):
            # Create signal pairs that span the valid range
            u_center = (signal_range[0] + signal_range[1]) / 2
            u_i = torch.clamp(u_center - diff / 2, signal_range[0], signal_range[1])
            u_j = torch.clamp(u_center + diff / 2, signal_range[0], signal_range[1])

            # Ensure the actual difference matches what we want
            actual_diff = u_j - u_i
            if abs(actual_diff - diff) > 1e-6:
                # Adjust to get the exact difference we want
                u_i = torch.clamp(u_center - diff / 2, signal_range[0], signal_range[1])
                u_j = u_i + diff
                if u_j > signal_range[1]:
                    u_j = torch.tensor(signal_range[1], device=device)
                    u_i = u_j - diff
                elif u_j < signal_range[0]:
                    u_j = torch.tensor(signal_range[0], device=device)
                    u_i = u_j - diff

            # Create input feature vector: [u_i, u_j, embedding_i, embedding_j]
            in_features = torch.cat([
                u_i.unsqueeze(0),  # u_i as (1,)
                u_j.unsqueeze(0),  # u_j as (1,)
                receiver_embedding,  # embedding_i (receiver) as (2,)
                sender_embedding  # embedding_j (sender) as (2,)
            ], dim=0)  # Final shape: (6,)
            line_inputs.append(in_features)

        line_features = torch.stack(line_inputs, dim=0)  # (len(u_diff_line), 6)

        with torch.no_grad():
            lin_edge = model.lin_edge(line_features)
            if model.lin_edge_positive:
                lin_edge = lin_edge ** 2

        # Apply weight scaling
        edge_output = lin_edge.squeeze(-1).cpu().numpy()
        weighted_output = edge_output * connection_weight

        # Find maximum response in target range [8, 10]
        if len(target_indices) > 0:
            max_in_range = np.max(weighted_output[target_indices])
            if max_in_range > max_response:
                max_response = max_in_range
                max_response_data = {
                    'receiver_name': neuron_name,
                    'sender_name': sender_name,
                    'receiver_id': neuron_id,
                    'sender_id': sender_id,
                    'weight': connection_weight,
                    'max_response': max_response,
                    'signal_diff_range': [8.0, 10.0]
                }

        connection_data.append({
            'sender_name': sender_name,
            'sender_id': sender_id,
            'weight': connection_weight,
            'output': weighted_output,
            'unweighted_output': edge_output
        })

    # Sort connections by weight magnitude for better visualization
    connection_data.sort(key=lambda x: abs(x['weight']), reverse=True)

    # Create line plot figure
    fig_lines, ax_lines = plt.subplots(1, 1, figsize=(14, 10))

    # Generate colors using a colormap that handles many lines well
    if len(connection_data) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(connection_data)))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(connection_data)))

    # Plot each connection
    for conn_idx, conn_data in enumerate(connection_data):
        color = colors[conn_idx]
        sender_name = conn_data['sender_name']
        weight = conn_data['weight']
        weighted_output = conn_data['output']

        # Line style based on weight sign
        line_style = '-' if weight >= 0 else '--'

        # Calculate line width with safe division
        max_weight = np.max(np.abs([c['weight'] for c in connection_data]))
        if max_weight > 0:
            line_width = 1.5 + min(2.0, abs(weight) / max_weight)
        else:
            line_width = 1.5  # Default width if all weights are zero

        ax_lines.plot(u_diff_line_np, weighted_output,
                      color=color, linewidth=line_width, linestyle=line_style,
                      label=f'{sender_name} (w={weight:.3f})')

    # Highlight the target range [8, 10]
    ax_lines.axvspan(8.0, 10.0, alpha=0.2, color='red', label='Target range [8,10]')

    ax_lines.set_xlabel('u_j - u_i (signal difference)')
    ax_lines.set_ylabel('weighted edge function output')
    ax_lines.set_title(
        f'weighted edge function vs signal difference\n(receiver: {neuron_name}, all incoming connections)')
    ax_lines.grid(True, alpha=0.3)

    # Adaptive legend placement based on number of connections
    n_connections = len(connection_data)
    if n_connections <= 5:
        # For few connections, use right side
        ax_lines.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    elif n_connections <= 15:
        # For medium number, use multiple columns on right
        ax_lines.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                        fontsize='x-small', ncol=1)
    else:
        # For many connections, use multiple columns below plot
        ncol = min(4, n_connections // 5 + 1)
        ax_lines.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center',
                        ncol=ncol, fontsize='x-small', framealpha=0.9)
        # Add more space at bottom for legend
        plt.subplots_adjust(bottom=0.25)

    plt.tight_layout()

    return fig_lines, max_response_data


def find_top_responding_pairs(model, all_neuron_list, adjacency_matrix, weight_matrix,
                              signal_range=(0, 10), resolution=100, device=None, top_k=10):
    """
    Find the top K receiver-sender pairs with largest response in signal difference range [8, 10]
    by analyzing all neurons as receivers

    Args:
        model: The trained model with embeddings and lin_edge
        all_neuron_list: Complete list of all 300 neuron names
        adjacency_matrix: 2D array (300x300) where adjacency_matrix[i,j] = 1 if i->j connection exists
        weight_matrix: 2D array (300x300) with connection weights
        signal_range: Tuple of (min_signal, max_signal) for DF/F0 measurements
        resolution: Number of points for signal difference sampling
        device: PyTorch device
        top_k: Number of top pairs to return

    Returns:
        top_pairs: List of top K pairs sorted by response magnitude
        top_figures: List of figures for the top pairs
    """

    # print(f"Analyzing all {len(all_neuron_list)} neurons to find top {top_k} responding pairs...")

    all_responses = []

    # Analyze each neuron as receiver
    for neuron_idx, neuron_name in enumerate(all_neuron_list):
        try:
            fig , max_response_data = analyze_mlp_edge_lines_weighted_with_max(
                model, neuron_name, all_neuron_list, adjacency_matrix, weight_matrix,
                signal_range, resolution, device
            )

            plt.close(fig)

            if max_response_data is not None:
                all_responses.append(max_response_data)

        except Exception as e:
            print(f"Error processing {neuron_name}: {e}")
            continue

    # Sort by response magnitude and get top K
    all_responses.sort(key=lambda x: x['max_response'], reverse=True)
    top_pairs = all_responses[:top_k]
    for i, pair in enumerate(top_pairs):
        print(f"{i + 1:2d}. {pair['receiver_name']} ← {pair['sender_name']}:  ({pair['max_response']:.4f})")

    return top_pairs    # , top_figures

def analyze_embedding_space(model, n_neurons=300):
    """Analyze the learned embedding space"""

    embedding = model.a.detach().cpu().numpy()  # (300, 2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Embedding scatter plot
    axes[0].scatter(embedding[:, 0], embedding[:, 1],
                              c=np.arange(n_neurons), cmap='tab10', alpha=0.7)
    axes[0].set_xlabel('Embedding Dimension 1')
    axes[0].set_ylabel('Embedding Dimension 2')
    axes[0].set_title('Learned Neuron Embeddings')
    axes[0].grid(True, alpha=0.3)

    # 2. Embedding distribution
    axes[1].hist(embedding[:, 0], bins=30, alpha=0.7, label='Dim 1')
    axes[1].hist(embedding[:, 1], bins=30, alpha=0.7, label='Dim 2')
    axes[1].set_xlabel('Embedding Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Embedding Value Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Distance matrix between embeddings
    distances = np.linalg.norm(embedding[:, None] - embedding[None, :], axis=2)
    im = axes[2].imshow(distances, cmap='viridis')
    axes[2].set_title('Pairwise Embedding Distances')
    axes[2].set_xlabel('Neuron Index')
    axes[2].set_ylabel('Neuron Index')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig('embedding_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    return embedding, distances


def analyze_mlp_phi_synaptic(model, n_neurons=300, signal_range=(0, 10), resolution=50, n_sample_pairs=200,
                             device=None):
    """
    Analyze the learned MLP phi function with statistical sampling
    Creates 2D plots: mean with std band + all individual line plots

    For generic_excitation update type:
    - u: signal (varied)
    - embedding: neuron embedding (sampled from different neurons)
    - msg: set to zeros (no message passing)
    - field: set to ones
    - excitation: set to zeros
    """

    embedding = model.a  # Shape: (300, 2)

    # Get excitation dimension from model
    excitation_dim = getattr(model, 'excitation_dim', 0)

    # Create signal grid (1D since we're analyzing signal vs embedding effects)
    u_vals = torch.linspace(signal_range[0], signal_range[1], resolution, device=device)

    print(f"sampling {n_sample_pairs} random neurons across {resolution} signal points...")
    print(f"excitation_dim: {excitation_dim}")

    # Sample random neurons
    np.random.seed(42)  # For reproducibility
    neuron_indices = np.random.choice(n_neurons, size=n_sample_pairs, replace=True)

    # Store all outputs for statistics
    all_outputs = torch.zeros(n_sample_pairs, resolution, device=device)

    # Process in batches to manage memory
    batch_size = 50
    for batch_start in trange(0, n_sample_pairs, batch_size):
        batch_end = min(batch_start + batch_size, n_sample_pairs)
        batch_size_actual = batch_end - batch_start

        batch_inputs = []
        for batch_idx in range(batch_size_actual):
            neuron_idx = neuron_indices[batch_start + batch_idx]

            # Get embedding for this neuron
            neuron_embedding = embedding[neuron_idx].unsqueeze(0).repeat(resolution, 1)  # (resolution, 2)

            # Create signal array
            u_batch = u_vals.unsqueeze(1)  # (resolution, 1)

            # Create fixed components
            msg = torch.zeros(resolution, 1, device=device)  # Message set to zeros
            field = torch.ones(resolution, 1, device=device)  # Field set to ones
            excitation = torch.zeros(resolution, excitation_dim, device=device)  # Excitation set to zeros

            # Concatenate input features: [u, embedding, msg, field, excitation]
            in_features = torch.cat([u_batch, neuron_embedding, msg, field, excitation], dim=1)
            batch_inputs.append(in_features)

        # Stack batch inputs
        batch_features = torch.stack(batch_inputs, dim=0)  # (batch_size, resolution, input_dim)
        batch_features = batch_features.reshape(-1, batch_features.shape[-1])  # (batch_size * resolution, input_dim)

        # Forward pass through MLP
        with torch.no_grad():
            phi_output = model.lin_phi(batch_features)

        # Reshape back to batch format
        phi_output = phi_output.reshape(batch_size_actual, resolution, -1).squeeze(-1)

        # Store results
        all_outputs[batch_start:batch_end] = phi_output

    # Compute statistics across all sampled neurons
    mean_output = torch.mean(all_outputs, dim=0).cpu().numpy()  # (resolution,)
    std_output = torch.std(all_outputs, dim=0).cpu().numpy()  # (resolution,)
    all_outputs_np = all_outputs.cpu().numpy()  # (n_sample_pairs, resolution)

    u_vals_np = u_vals.cpu().numpy()

    print(f"statistics computed over {n_sample_pairs} neurons")
    print(f"mean output range: [{mean_output.min():.4f}, {mean_output.max():.4f}]")
    print(f"std output range: [{std_output.min():.4f}, {std_output.max():.4f}]")

    # Create 2D plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: Mean plot with std band
    ax1.plot(u_vals_np, mean_output, 'b-', linewidth=3, label='mean', zorder=10)
    ax1.fill_between(u_vals_np, mean_output - std_output, mean_output + std_output,
                     alpha=0.3, color='blue', label='±1 std')
    ax1.set_xlabel('signal (u)')
    ax1.set_ylabel('phi output')
    ax1.set_title(f'mean phi function\n(over {n_sample_pairs} random neurons)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right panel: All individual line plots
    # Use alpha to make individual lines semi-transparent
    alpha_val = min(0.8, max(0.1, 50.0 / n_sample_pairs))  # Adaptive alpha based on number of lines

    for i in range(n_sample_pairs):
        ax2.plot(u_vals_np, all_outputs_np[i], '-', alpha=alpha_val, linewidth=0.5, color='gray')

    # Overlay the mean on top
    ax2.plot(u_vals_np, mean_output, 'r-', linewidth=2, label='mean', zorder=10)

    ax2.set_xlabel('signal (u)')
    ax2.set_ylabel('phi output')
    ax2.set_title(f'all individual phi functions\n({n_sample_pairs} neurons)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    return fig


def analyze_mlp_phi_embedding(model, n_neurons=300, signal_range=(0, 10), resolution=50, n_sample_pairs=200,
                                 device=None):
    """
    Analyze MLP phi function across signal and embedding space
    Creates 2D heatmaps showing how phi varies with signal and embedding dimensions
    """

    embedding = model.a  # Shape: (300, 2)
    excitation_dim = getattr(model, 'excitation_dim', 0)

    # Create signal grid
    u_vals = torch.linspace(signal_range[0], signal_range[1], resolution, device=device)

    print("analyzing phi function across signal and embedding space...")
    print(f"resolution: {resolution}x{resolution}, excitation_dim: {excitation_dim}")

    # Sample random neurons for embedding analysis
    np.random.seed(42)
    neuron_indices = np.random.choice(n_neurons, size=n_sample_pairs, replace=True)

    # Store outputs for each embedding dimension
    all_outputs_emb1 = torch.zeros(n_sample_pairs, resolution, device=device)
    torch.zeros(n_sample_pairs, resolution, device=device)

    # Process in batches
    batch_size = 50
    for batch_start in trange(0, n_sample_pairs, batch_size):
        batch_end = min(batch_start + batch_size, n_sample_pairs)
        batch_size_actual = batch_end - batch_start

        batch_inputs = []
        for batch_idx in range(batch_size_actual):
            neuron_idx = neuron_indices[batch_start + batch_idx]

            # Get embedding for this neuron
            neuron_embedding = embedding[neuron_idx].unsqueeze(0).repeat(resolution, 1)

            # Create signal array
            u_batch = u_vals.unsqueeze(1)

            # Fixed components
            msg = torch.zeros(resolution, 1, device=device)
            field = torch.ones(resolution, 1, device=device)
            excitation = torch.zeros(resolution, excitation_dim, device=device)

            # Input features
            in_features = torch.cat([u_batch, neuron_embedding, msg, field, excitation], dim=1)
            batch_inputs.append(in_features)

        # Process batch
        batch_features = torch.stack(batch_inputs, dim=0)
        batch_features = batch_features.reshape(-1, batch_features.shape[-1])

        with torch.no_grad():
            phi_output = model.lin_phi(batch_features)

        phi_output = phi_output.reshape(batch_size_actual, resolution, -1).squeeze(-1)

        # Store results
        all_outputs_emb1[batch_start:batch_end] = phi_output

    # Now create 2D grid: signal vs embedding dimension
    # We'll vary embedding dimension 1 and keep dimension 2 at mean value
    emb_vals = torch.linspace(embedding[:, 0].min(), embedding[:, 0].max(), resolution, device=device)
    emb_mean_dim2 = embedding[:, 1].mean()

    # Create 2D output grid
    output_grid = torch.zeros(resolution, resolution, device=device)  # (emb_dim1, signal)

    print("creating 2D grid: embedding dim 1 vs signal...")
    for i, emb1_val in enumerate(trange(len(emb_vals))):
        emb1_val = emb_vals[i]

        # Create embedding with varying dim1 and fixed dim2
        neuron_embedding = torch.stack([
            emb1_val.repeat(resolution),
            emb_mean_dim2.repeat(resolution)
        ], dim=1)

        u_batch = u_vals.unsqueeze(1)
        msg = torch.zeros(resolution, 1, device=device)
        field = torch.ones(resolution, 1, device=device)
        excitation = torch.zeros(resolution, excitation_dim, device=device)

        in_features = torch.cat([u_batch, neuron_embedding, msg, field, excitation], dim=1)

        with torch.no_grad():
            phi_output = model.lin_phi(in_features)

        output_grid[i, :] = phi_output.squeeze()

    output_grid_np = output_grid.cpu().numpy()
    u_vals.cpu().numpy()
    emb_vals_np = emb_vals.cpu().numpy()

    # Create 2D heatmap
    fig_2d, ax = plt.subplots(1, 1, figsize=(10, 8))

    im = ax.imshow(output_grid_np, extent=[signal_range[0], signal_range[1],
                                           emb_vals_np.min(), emb_vals_np.max()],
                   origin='lower', cmap='viridis', aspect='auto')
    ax.set_xlabel('signal (u)')
    ax.set_ylabel('embedding dimension 1')
    ax.set_title(f'phi function: signal vs embedding\n(dim 2 fixed at {emb_mean_dim2:.3f})')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('phi output')

    plt.tight_layout()

    return fig_2d, output_grid_np


def compute_separation_index(connectivity_neurons, odor_responsive_neurons):
    """
    Compute functional separation between high connectivity and high odor-responsive neurons

    Args:
        connectivity_neurons: List of neuron names with high connectivity
        odor_responsive_neurons: List of neuron names with high odor responses

    Returns:
        separation_metrics: Dict with separation statistics
    """
    connectivity_set = set(connectivity_neurons)
    odor_set = set(odor_responsive_neurons)

    # Find overlap
    overlap = connectivity_set.intersection(odor_set)

    # Compute separation metrics
    len(connectivity_set.union(odor_set))
    overlap_count = len(overlap)
    min_set_size = min(len(connectivity_set), len(odor_set))

    # Separation index: 1 - (overlap / min_set_size)
    separation_index = 1.0 - (overlap_count / min_set_size) if min_set_size > 0 else 1.0

    # Additional metrics
    connectivity_purity = 1.0 - (overlap_count / len(connectivity_set)) if len(connectivity_set) > 0 else 1.0
    odor_purity = 1.0 - (overlap_count / len(odor_set)) if len(odor_set) > 0 else 1.0

    return {
        'separation_index': separation_index,
        'overlap_count': overlap_count,
        'connectivity_purity': connectivity_purity,
        'odor_purity': odor_purity,
        'total_connectivity_neurons': len(connectivity_set),
        'total_odor_neurons': len(odor_set),
        'overlapping_neurons': list(overlap)
    }


def classify_neural_architecture(separation_index, specialist_threshold=0.95, adapter_threshold=0.70):
    """
    Classify neural architecture based on separation index

    Args:
        separation_index: Float between 0 and 1
        specialist_threshold: Threshold for specialist classification
        adapter_threshold: Threshold for adapter classification

    Returns:
        architecture_type: String classification
    """
    if separation_index >= specialist_threshold:
        return 'specialist'
    elif separation_index >= adapter_threshold:
        return 'adapter'
    else:
        return 'generalist'


def analyze_individual_architectures(top_pairs_by_run, odor_responses_by_run, all_neuron_list,
                                     specialist_threshold=0.95, adapter_threshold=0.70):
    """
    Analyze neural architectures across all individual worms

    Args:
        top_pairs_by_run: Dict with run_id -> list of top connectivity pairs
        odor_responses_by_run: Dict with run_id -> odor response data
        all_neuron_list: List of all neuron names
        specialist_threshold: Threshold for specialist classification
        adapter_threshold: Threshold for adapter classification

    Returns:
        architecture_analysis: Dict with comprehensive analysis results
    """

    architecture_data = []
    separation_details = {}

    print("=== INDIVIDUAL NEURAL ARCHITECTURE ANALYSIS ===")

    for run_id in top_pairs_by_run.keys():
        # Extract high connectivity neurons
        connectivity_neurons = []
        for pair in top_pairs_by_run[run_id]:
            connectivity_neurons.extend([pair['sender_name'], pair['receiver_name']])
        connectivity_neurons = list(set(connectivity_neurons))  # Remove duplicates

        # Extract high odor-responsive neurons from all odors
        odor_responsive_neurons = set()
        if run_id in odor_responses_by_run:
            for odor in ['butanone', 'pentanedione', 'NaCL']:
                if odor in odor_responses_by_run[run_id]:
                    odor_responsive_neurons.update(odor_responses_by_run[run_id][odor]['names'])
        odor_responsive_neurons = list(odor_responsive_neurons)

        # Compute separation metrics
        separation_metrics = compute_separation_index(connectivity_neurons, odor_responsive_neurons)

        # Classify architecture
        architecture_type = classify_neural_architecture(
            separation_metrics['separation_index'],
            specialist_threshold,
            adapter_threshold
        )

        # Store data
        architecture_data.append({
            'run_id': run_id,
            'architecture_type': architecture_type,
            'separation_index': separation_metrics['separation_index'],
            'overlap_count': separation_metrics['overlap_count'],
            'connectivity_purity': separation_metrics['connectivity_purity'],
            'odor_purity': separation_metrics['odor_purity'],
            'n_connectivity_neurons': separation_metrics['total_connectivity_neurons'],
            'n_odor_neurons': separation_metrics['total_odor_neurons']
        })

        separation_details[run_id] = separation_metrics

        print(f"Run {run_id}: {architecture_type.upper()} "
              f"(separation: {separation_metrics['separation_index']:.3f}, "
              f"overlap: {separation_metrics['overlap_count']})")

    # Convert to DataFrame for analysis
    df = pd.DataFrame(architecture_data)

    # Summary statistics by architecture type
    type_summary = df.groupby('architecture_type').agg({
        'separation_index': ['count', 'mean', 'std', 'min', 'max'],
        'overlap_count': ['mean', 'std'],
        'n_connectivity_neurons': ['mean', 'std'],
        'n_odor_neurons': ['mean', 'std']
    }).round(3)

    print("\n=== ARCHITECTURE TYPE SUMMARY ===")
    print(type_summary)

    return {
        'architecture_data': df,
        'separation_details': separation_details,
        'type_summary': type_summary,
        'classification_thresholds': {
            'specialist': specialist_threshold,
            'adapter': adapter_threshold
        }
    }


def identify_hub_neurons_by_type(top_pairs_by_run, architecture_analysis, min_frequency=0.5):
    """
    Identify hub neurons for each architecture type

    Args:
        top_pairs_by_run: Dict with run_id -> list of top connectivity pairs
        architecture_analysis: Results from analyze_individual_architectures
        min_frequency: Minimum frequency to be considered a hub

    Returns:
        hub_analysis: Dict with hub neuron analysis by architecture type
    """

    # Get architecture types for each run
    run_to_type = dict(zip(architecture_analysis['architecture_data']['run_id'],
                           architecture_analysis['architecture_data']['architecture_type']))

    # Group by architecture type
    hubs_by_type = defaultdict(lambda: defaultdict(int))
    connection_counts_by_type = defaultdict(int)

    print("=== HUB NEURON ANALYSIS BY ARCHITECTURE TYPE ===")

    for run_id, pairs in top_pairs_by_run.items():
        arch_type = run_to_type[run_id]
        connection_counts_by_type[arch_type] += 1

        # Count neuron appearances in this run
        neuron_counts = defaultdict(int)
        for pair in pairs:
            neuron_counts[pair['sender_name']] += 1
            neuron_counts[pair['receiver_name']] += 1

        # Add to architecture type totals
        for neuron, count in neuron_counts.items():
            hubs_by_type[arch_type][neuron] += count

    # Analyze hubs for each architecture type
    hub_analysis = {}

    for arch_type in hubs_by_type.keys():
        n_runs = connection_counts_by_type[arch_type]

        # Calculate frequencies and identify hubs
        neuron_frequencies = {}
        for neuron, total_count in hubs_by_type[arch_type].items():
            # Frequency = appearances / total possible appearances
            max_possible = n_runs * 40  # Max appearances if in all top-20 pairs as both sender and receiver
            frequency = total_count / max_possible
            neuron_frequencies[neuron] = {
                'total_count': total_count,
                'frequency': frequency,
                'n_runs': n_runs
            }

        # Sort by frequency
        sorted_neurons = sorted(neuron_frequencies.items(),
                                key=lambda x: x[1]['frequency'], reverse=True)

        # Identify hubs above threshold
        hubs = [(neuron, stats) for neuron, stats in sorted_neurons
                if stats['frequency'] >= min_frequency]

        hub_analysis[arch_type] = {
            'all_neurons': dict(sorted_neurons),
            'hub_neurons': hubs,
            'n_runs': n_runs,
            'top_10_neurons': sorted_neurons[:10]
        }

        print(f"\n{arch_type.upper()} ARCHITECTURE ({n_runs} individuals):")
        print("Top 10 hub neurons:")
        for i, (neuron, stats) in enumerate(sorted_neurons[:10]):
            print(f"  {i + 1:2d}. {neuron}: {stats['frequency']:.3f} "
                  f"({stats['total_count']} total appearances)")

    return hub_analysis


def compare_pathway_organization(top_pairs_by_run, architecture_analysis, all_neuron_list):
    """
    Compare pathway organization across architecture types

    Args:
        top_pairs_by_run: Dict with run_id -> list of top connectivity pairs
        architecture_analysis: Results from analyze_individual_architectures
        all_neuron_list: List of all neuron names

    Returns:
        pathway_analysis: Dict with pathway comparison results
    """

    # Define functional neuron classes
    neuron_classes = {
        'chemosensory': ['ADLR', 'ADLL', 'AWAL', 'AWAR', 'AWBL', 'AWBR', 'AWCL', 'AWCR',
                         'ASKL', 'ASKR', 'ASHL', 'ASHR', 'ASJL', 'ASJR'],
        'command': ['AVAR', 'AVAL', 'AVBL', 'AVBR', 'AVDL', 'AVDR', 'AVKL', 'AVKR',
                    'AVHL', 'AVHR', 'AVJL', 'AVJR'],
        'ring_integration': ['RID', 'RIS', 'RIML', 'RIMR', 'RIBL', 'RIBR', 'RIAR', 'RIAL',
                             'RICL', 'RICR', 'RMDL', 'RMDR', 'RMDVL', 'RMDVR'],
        'motor': ['SMDVL', 'SMDVR', 'SMDDL', 'SMDDR', 'SMBVL', 'SMBVR', 'SMBDL', 'SMBDR',
                  'VB01', 'VB02', 'VB03', 'VB04', 'VB05', 'VB06', 'VB07', 'VB08', 'VB09', 'VB10', 'VB11',
                  'DB01', 'DB02', 'DB03', 'DB04', 'DB05', 'DB06', 'DB07'],
        'head_sensory': ['CEPVL', 'CEPVR', 'CEPDL', 'CEPDR', 'OLQVL', 'OLQVR', 'OLQDL', 'OLQDR',
                         'OLLR', 'OLLL'],
        'muscle': ['M1', 'M2L', 'M2R', 'M3L', 'M3R', 'M4', 'M5', 'MI', 'I1L', 'I1R', 'I2L', 'I2R',
                   'I3', 'I4', 'I5', 'I6']
    }

    # Get architecture types for each run
    run_to_type = dict(zip(architecture_analysis['architecture_data']['run_id'],
                           architecture_analysis['architecture_data']['architecture_type']))

    # Analyze pathway patterns by architecture type
    pathway_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    connection_types = defaultdict(lambda: defaultdict(int))

    print("=== PATHWAY ORGANIZATION ANALYSIS ===")

    for run_id, pairs in top_pairs_by_run.items():
        arch_type = run_to_type[run_id]

        for pair in pairs:
            sender = pair['sender_name']
            receiver = pair['receiver_name']
            weight = pair.get('max_response', 0)  # Use weight if available

            # Classify sender and receiver
            sender_class = 'other'
            receiver_class = 'other'

            for class_name, neurons in neuron_classes.items():
                if sender in neurons:
                    sender_class = class_name
                if receiver in neurons:
                    receiver_class = class_name

            # Record connection type
            connection_type = f"{sender_class}→{receiver_class}"
            connection_types[arch_type][connection_type] += 1

            # Record pathway statistics
            pathway_stats[arch_type][sender_class]['out_degree'] += 1
            pathway_stats[arch_type][receiver_class]['in_degree'] += 1
            pathway_stats[arch_type][sender_class]['out_weight'] += weight
            pathway_stats[arch_type][receiver_class]['in_weight'] += weight

    # Normalize by number of runs of each type
    type_counts = architecture_analysis['architecture_data']['architecture_type'].value_counts()

    normalized_connections = {}
    for arch_type in connection_types.keys():
        n_runs = type_counts[arch_type]
        normalized_connections[arch_type] = {
            conn_type: count / n_runs
            for conn_type, count in connection_types[arch_type].items()
        }

    # Print results
    for arch_type in ['specialist', 'adapter', 'generalist']:
        if arch_type in normalized_connections:
            print(f"\n{arch_type.upper()} PATHWAY PATTERNS (avg per individual):")

            # Sort connection types by frequency
            sorted_connections = sorted(normalized_connections[arch_type].items(),
                                        key=lambda x: x[1], reverse=True)

            for conn_type, avg_count in sorted_connections[:10]:
                print(f"  {conn_type}: {avg_count:.2f}")

    return {
        'pathway_stats': dict(pathway_stats),
        'connection_types': dict(connection_types),
        'normalized_connections': normalized_connections,
        'neuron_classes': neuron_classes
    }


def normalize_edge_function_amplitudes(edge_functions_by_run, method='z_score'):
    """
    Normalize edge function amplitudes across runs to account for different scales

    Args:
        edge_functions_by_run: Dict with run_id -> edge function data
        method: Normalization method ('z_score', 'min_max', 'robust')

    Returns:
        normalized_functions: Dict with normalized edge function data
    """

    normalized_functions = {}

    for run_id, edge_data in edge_functions_by_run.items():
        if method == 'z_score':
            # Z-score normalization
            mean_val = np.mean(edge_data)
            std_val = np.std(edge_data)
            normalized = (edge_data - mean_val) / std_val if std_val > 0 else edge_data

        elif method == 'min_max':
            # Min-max normalization to [0, 1]
            min_val = np.min(edge_data)
            max_val = np.max(edge_data)
            normalized = (edge_data - min_val) / (max_val - min_val) if max_val > min_val else edge_data

        elif method == 'robust':
            # Robust normalization using median and IQR
            median_val = np.median(edge_data)
            q75 = np.percentile(edge_data, 75)
            q25 = np.percentile(edge_data, 25)
            iqr = q75 - q25
            normalized = (edge_data - median_val) / iqr if iqr > 0 else edge_data

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        normalized_functions[run_id] = normalized

    return normalized_functions


def plot_architecture_analysis_summary(architecture_analysis, hub_analysis, pathway_analysis):
    """
    Create comprehensive visualization of architecture analysis results

    Args:
        architecture_analysis: Results from analyze_individual_architectures
        hub_analysis: Results from identify_hub_neurons_by_type
        pathway_analysis: Results from compare_pathway_organization

    Returns:
        fig: matplotlib figure with summary plots
    """

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Distribution of architecture types
    df = architecture_analysis['architecture_data']
    type_counts = df['architecture_type'].value_counts()

    axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Distribution of Architecture Types')

    # 2. Separation index distribution
    for arch_type in df['architecture_type'].unique():
        subset = df[df['architecture_type'] == arch_type]
        axes[0, 1].hist(subset['separation_index'], alpha=0.7, label=arch_type, bins=10)

    axes[0, 1].set_xlabel('Separation Index')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Separation Index by Architecture Type')
    axes[0, 1].legend()

    # 3. Overlap count vs separation index
    colors = {'specialist': 'blue', 'adapter': 'green', 'generalist': 'red'}
    for arch_type in df['architecture_type'].unique():
        subset = df[df['architecture_type'] == arch_type]
        axes[0, 2].scatter(subset['separation_index'], subset['overlap_count'],
                           c=colors.get(arch_type, 'gray'), label=arch_type, alpha=0.7)

    axes[0, 2].set_xlabel('Separation Index')
    axes[0, 2].set_ylabel('Overlap Count')
    axes[0, 2].set_title('Separation vs Overlap')
    axes[0, 2].legend()

    # 4. Hub neuron frequency comparison
    if len(hub_analysis) >= 2:
        arch_types = list(hub_analysis.keys())[:2]  # Compare first two types

        # Get top neurons for each type
        neurons_type1 = [item[0] for item in hub_analysis[arch_types[0]]['top_10_neurons']]
        freq_type1 = [item[1]['frequency'] for item in hub_analysis[arch_types[0]]['top_10_neurons']]

        [item[0] for item in hub_analysis[arch_types[1]]['top_10_neurons']]
        freq_type2 = [item[1]['frequency'] for item in hub_analysis[arch_types[1]]['top_10_neurons']]

        x_pos = np.arange(len(neurons_type1))
        width = 0.35

        axes[1, 0].bar(x_pos - width / 2, freq_type1, width, label=arch_types[0])
        axes[1, 0].bar(x_pos + width / 2, freq_type2, width, label=arch_types[1])

        axes[1, 0].set_xlabel('Neurons')
        axes[1, 0].set_ylabel('Hub Frequency')
        axes[1, 0].set_title(f'Top Hub Neurons: {arch_types[0]} vs {arch_types[1]}')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(neurons_type1, rotation=45)
        axes[1, 0].legend()

    # 5. Connection type heatmap
    if 'normalized_connections' in pathway_analysis:
        # Create matrix of connection types vs architecture types
        all_arch_types = list(pathway_analysis['normalized_connections'].keys())
        all_conn_types = set()
        for arch_conns in pathway_analysis['normalized_connections'].values():
            all_conn_types.update(arch_conns.keys())
        all_conn_types = sorted(list(all_conn_types))

        matrix = np.zeros((len(all_conn_types), len(all_arch_types)))
        for i, conn_type in enumerate(all_conn_types):
            for j, arch_type in enumerate(all_arch_types):
                matrix[i, j] = pathway_analysis['normalized_connections'][arch_type].get(conn_type, 0)

        im = axes[1, 1].imshow(matrix, cmap='viridis', aspect='auto')
        axes[1, 1].set_xticks(range(len(all_arch_types)))
        axes[1, 1].set_xticklabels(all_arch_types)
        axes[1, 1].set_yticks(range(len(all_conn_types)))
        axes[1, 1].set_yticklabels(all_conn_types, fontsize=8)
        axes[1, 1].set_title('Connection Types by Architecture')
        plt.colorbar(im, ax=axes[1, 1])

    # 6. Summary statistics
    axes[1, 2].axis('off')
    summary_text = f"""
    SUMMARY STATISTICS

    Total Individuals: {len(df)}

    Architecture Types:
    """

    for arch_type in df['architecture_type'].unique():
        count = sum(df['architecture_type'] == arch_type)
        mean_sep = df[df['architecture_type'] == arch_type]['separation_index'].mean()
        summary_text += f"  {arch_type}: {count} ({count / len(df) * 100:.1f}%)\n"
        summary_text += f"    Mean separation: {mean_sep:.3f}\n"

    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                    verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    return fig


def run_neural_architecture_pipeline(top_pairs_by_run, odor_responses_by_run, all_neuron_list,
                                     specialist_threshold=0.95, adapter_threshold=0.70):
    """
    Run the complete neural architecture analysis pipeline

    Args:
        top_pairs_by_run: Dict with run_id -> list of top connectivity pairs
        odor_responses_by_run: Dict with run_id -> odor response data
        all_neuron_list: List of all neuron names
        specialist_threshold: Threshold for specialist classification
        adapter_threshold: Threshold for adapter classification

    Returns:
        complete_analysis: Dict with all analysis results
    """

    print("RUNNING NEURAL ARCHITECTURE ANALYSIS PIPELINE")
    print("=" * 60)

    # Phase 1: Individual Architecture Classification
    architecture_analysis = analyze_individual_architectures(
        top_pairs_by_run, odor_responses_by_run, all_neuron_list,
        specialist_threshold, adapter_threshold
    )

    # Phase 2: Hub Neuron Analysis
    hub_analysis = identify_hub_neurons_by_type(
        top_pairs_by_run, architecture_analysis
    )

    # Phase 2: Pathway Organization Comparison
    pathway_analysis = compare_pathway_organization(
        top_pairs_by_run, architecture_analysis, all_neuron_list
    )

    # Create summary visualization
    summary_fig = plot_architecture_analysis_summary(
        architecture_analysis, hub_analysis, pathway_analysis
    )

    return {
        'architecture_analysis': architecture_analysis,
        'hub_analysis': hub_analysis,
        'pathway_analysis': pathway_analysis,
        'summary_figure': summary_fig
    }



