"""Shared plotting and analysis functions for FlyVis.

Used by both the training loop (graph_trainer.py / utils.py) and
post-training analysis (GNN_PlotFigure.py).
"""
import matplotlib.pyplot as plt
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
    """Extract linear slope of lin_edge for each neuron j.

    Evaluates lin_edge(a_j, v) over the neuron's activity range and fits
    a linear model to extract the slope r_j.

    Returns:
        slopes: (n_neurons,) numpy array of lin_edge slopes.
    """
    signal_model_name = config.graph_model.signal_model_name
    emb_dim = config.graph_model.embedding_dim
    lin_edge_positive = config.graph_model.lin_edge_positive
    slopes = []

    for n in range(n_neurons):
        # Only fit if activity range includes positive values
        if mu_activity[n] + sigma_activity[n] > 0:
            lo = max(float(mu_activity[n] - 2 * sigma_activity[n]), 0.0)
            hi = float(mu_activity[n] + 2 * sigma_activity[n])
            rr = torch.linspace(lo, hi, 1000, device=device)
            embedding_ = model.a[n, :] * torch.ones((1000, emb_dim), device=device)

            if 'PDE_N9_B' in signal_model_name:
                in_features = torch.cat((rr[:, None] * 0, rr[:, None], embedding_, embedding_), dim=1)
            else:
                in_features = torch.cat((rr[:, None], embedding_), dim=1)

            with torch.no_grad():
                func = model.lin_edge(in_features.float())
                if lin_edge_positive:
                    func = func ** 2

            rr_np = to_numpy(rr)
            func_np = to_numpy(func.squeeze())
            try:
                fit, _ = curve_fit(linear_model, rr_np, func_np)
                slopes.append(fit[0])
            except Exception:
                coeffs = np.polyfit(rr_np, func_np, 1)
                slopes.append(coeffs[0])
        else:
            slopes.append(1.0)

    return np.array(slopes)


def extract_lin_phi_slopes(model, config, n_neurons, mu_activity, sigma_activity, device):
    """Extract linear slope and offset of lin_phi for each neuron i.

    Evaluates lin_phi(a_i, v_i, msg=0, exc=0) over the neuron's activity
    range and fits a linear model.

    Returns:
        slopes: (n_neurons,) numpy array — slope relates to 1/tau.
        offsets: (n_neurons,) numpy array — offset relates to V_rest.
    """
    emb_dim = config.graph_model.embedding_dim
    slopes = []
    offsets = []

    for n in range(n_neurons):
        lo = float(mu_activity[n] - 2 * sigma_activity[n])
        hi = float(mu_activity[n] + 2 * sigma_activity[n])
        rr = torch.linspace(lo, hi, 1000, device=device)
        embedding_ = model.a[n, :] * torch.ones((1000, emb_dim), device=device)
        in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, torch.zeros_like(rr[:, None])), dim=1)

        with torch.no_grad():
            func = model.lin_phi(in_features.float())

        rr_np = to_numpy(rr)
        func_np = to_numpy(func.squeeze())
        try:
            fit, _ = curve_fit(linear_model, rr_np, func_np)
            slopes.append(fit[0])
            offsets.append(fit[1])
        except Exception:
            coeffs = np.polyfit(rr_np, func_np, 1)
            slopes.append(coeffs[0])
            offsets.append(coeffs[1])

    return np.array(slopes), np.array(offsets)


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
    """Plot lin_phi function curves colored by neuron type.

    Args:
        ax: matplotlib Axes.
        model: model with .a, .lin_phi.
        config: config with plotting.xlim, plotting.ylim, graph_model.embedding_dim.
        n_neurons: number of neurons.
        type_list: (N,) type indices.
        cmap: CustomColorMap.
        device: torch device.
        step: plot every step-th neuron (default 20).
    """
    emb_dim = config.graph_model.embedding_dim
    rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
    type_np = to_numpy(type_list).astype(int)

    for n in range(n_neurons):
        if n % step == 0:
            embedding_ = model.a[n, :] * torch.ones((1000, emb_dim), device=device)
            in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, torch.zeros_like(rr[:, None])), dim=1)
            with torch.no_grad():
                func = model.lin_phi(in_features.float())
            ax.plot(to_numpy(rr), to_numpy(func), color=cmap.color(type_np[n]),
                    linewidth=1, alpha=0.2)

    ax.set_xlim(config.plotting.xlim)
    ax.set_ylim(config.plotting.ylim)
    ax.set_xlabel('$v_i$', fontsize=32)
    ax.set_ylabel(r'learned $\mathrm{MLP_0}(\mathbf{a}_i, v_i)$', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)


def plot_lin_edge(ax, model, config, n_neurons, type_list, cmap, device, step=20):
    """Plot lin_edge function curves colored by neuron type.

    Args:
        ax: matplotlib Axes.
        model: model with .a, .lin_edge.
        config: config with plotting.xlim, graph_model.*.
        n_neurons: number of neurons.
        type_list: (N,) type indices.
        cmap: CustomColorMap.
        device: torch device.
        step: plot every step-th neuron (default 20).
    """
    signal_model_name = config.graph_model.signal_model_name
    emb_dim = config.graph_model.embedding_dim
    lin_edge_positive = config.graph_model.lin_edge_positive
    rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
    type_np = to_numpy(type_list).astype(int)

    for n in range(n_neurons):
        if n % step == 0:
            embedding_ = model.a[n, :] * torch.ones((1000, emb_dim), device=device)
            if 'PDE_N9_B' in signal_model_name:
                in_features = torch.cat((rr[:, None] * 0, rr[:, None], embedding_, embedding_), dim=1)
            else:
                in_features = torch.cat((rr[:, None], embedding_), dim=1)

            with torch.no_grad():
                func = model.lin_edge(in_features.float())
                if lin_edge_positive:
                    func = func ** 2
            ax.plot(to_numpy(rr), to_numpy(func), color=cmap.color(type_np[n]),
                    linewidth=1, alpha=0.2)

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
