import matplotlib.pyplot as plt
import os
import torch

from flyvis_gnn.utils import to_numpy
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from collections import Counter

def linear_model(x, a, b):
    return a * x + b


def compute_normalization_value(func_values, x_values, method='plateau',
                                 x_start=None, x_stop=None, derivative_threshold=0.01,
                                 per_neuron=False):
    """
    Compute normalization value for MLP output (e.g., transfer function psi).

    Args:
        func_values: tensor of shape (n_neurons, n_points) - function values
        x_values: tensor of shape (n_points,) - x coordinates
        method: str - 'max', 'median', 'mean', or 'plateau'
        x_start: float - start of range for normalization (default: min(x_values))
        x_stop: float - end of range for normalization (default: max(x_values))
        derivative_threshold: float - relative threshold for plateau detection
        per_neuron: bool - if True, return per-neuron values (tensor), else single scalar

    Returns:
        If per_neuron=False: normalization_value (float) - single value to normalize by
        If per_neuron=True: normalization_values (tensor) - per-neuron values, shape (n_neurons,)
    """
    func_values = func_values.detach()
    x_values = x_values.detach()

    # Default range
    if x_start is None:
        x_start = x_values.min().item()
    if x_stop is None:
        x_stop = x_values.max().item()

    # Filter to range [x_start, x_stop]
    mask = (x_values >= x_start) & (x_values <= x_stop)
    x_range = x_values[mask]
    func_range = func_values[:, mask]  # (n_neurons, n_points_in_range)

    if func_range.shape[1] < 2:
        # Not enough points, fall back to max
        if per_neuron:
            return func_values.abs().max(dim=1)[0]
        return func_values.abs().max().item()

    if method == 'max':
        # Maximum absolute value in range
        if per_neuron:
            return func_range.abs().max(dim=1)[0]
        return func_range.abs().max().item()

    elif method == 'median':
        if per_neuron:
            # Per-neuron median across points in range
            return func_range.median(dim=1)[0]
        # Median of mean across neurons
        neuron_means = func_range.mean(dim=1)
        return neuron_means.median().item()

    elif method == 'mean':
        if per_neuron:
            # Per-neuron mean across points in range
            return func_range.mean(dim=1)
        # Mean value in range (across all neurons and points)
        return func_range.mean().item()

    elif method == 'plateau':
        # Detect plateau by finding where derivative is flat
        # Compute finite differences along x
        dx = x_range[1] - x_range[0]
        if dx == 0:
            if per_neuron:
                return func_range.mean(dim=1)
            return func_range.mean().item()

        if per_neuron:
            # Per-neuron plateau detection
            n_neurons = func_range.shape[0]
            norm_values = torch.zeros(n_neurons, device=func_values.device)

            for n in range(n_neurons):
                func_n = func_range[n]  # (n_points_in_range,)
                d_func = (func_n[1:] - func_n[:-1]) / dx
                max_derivative = d_func.abs().max().item()

                if max_derivative < 1e-10:
                    norm_values[n] = func_n.mean()
                    continue

                threshold = derivative_threshold * max_derivative
                plateau_mask = d_func.abs() < threshold

                if plateau_mask.sum() < 2:
                    # No plateau, use mean in range
                    norm_values[n] = func_n.mean()
                else:
                    plateau_indices = torch.where(plateau_mask)[0]
                    norm_values[n] = func_n[plateau_indices].mean()

            return norm_values
        else:
            # Global plateau detection (mean across neurons)
            func_mean = func_range.mean(dim=0)  # (n_points_in_range,)

            # Compute derivative (finite difference)
            d_func = (func_mean[1:] - func_mean[:-1]) / dx

            # Find where derivative is small (plateau region)
            max_derivative = d_func.abs().max().item()
            if max_derivative < 1e-10:
                # Function is essentially flat everywhere
                return func_mean.mean().item()

            threshold = derivative_threshold * max_derivative
            plateau_mask = d_func.abs() < threshold

            if plateau_mask.sum() < 2:
                # No clear plateau found, use max
                print("  normalization: no plateau detected, using max value")
                return func_range.abs().max().item()

            # Get indices where plateau exists
            plateau_indices = torch.where(plateau_mask)[0]

            # Use the values in plateau region (offset by 1 due to derivative)
            plateau_values = func_mean[plateau_indices]
            norm_value = plateau_values.mean().item()

            print(f"  normalization: plateau detected at {plateau_mask.sum().item()} points, value={norm_value:.4f}")
            return norm_value

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_in_features_update(rr=None, model=None, embedding = None, device=None):

    n_neurons = model.n_neurons
    model_update_type = model.update_type

    if embedding == None:
        embedding = model.a[0:n_neurons]
        if model.embedding_trial:
            embedding = torch.cat((embedding, model.b[0].repeat(n_neurons, 1)), dim=1)


    if rr == None:
        if 'generic' in model_update_type:
            if 'excitation' in model_update_type:
                in_features = torch.cat((
                    torch.zeros((n_neurons, 1), device=device),
                    embedding,
                    torch.zeros((n_neurons, 1), device=device),
                    torch.ones((n_neurons, 1), device=device),
                    torch.zeros((n_neurons, model.excitation_dim), device=device)
                ), dim=1)
            else:
                in_features = torch.cat((
                    torch.zeros((n_neurons, 1), device=device),
                    embedding,
                    torch.ones((n_neurons, 1), device=device),
                    torch.ones((n_neurons, 1), device=device)
                ), dim=1)
        else:
            in_features = torch.cat((torch.zeros((n_neurons, 1), device=device), embedding), dim=1)
    else:
        if 'generic' in model_update_type:
            if 'excitation' in model_update_type:
                in_features = torch.cat((
                    rr,
                    embedding,
                    torch.zeros((rr.shape[0], 1), device=device),
                    torch.ones((rr.shape[0], 1), device=device),
                    torch.zeros((rr.shape[0], model.excitation_dim), device=device)
                ), dim=1)
            else:
                in_features = torch.cat((
                    rr,
                    embedding,
                    torch.ones((rr.shape[0], 1), device=device),
                    torch.ones((rr.shape[0], 1), device=device)
                ), dim=1)
        else:
            in_features = torch.cat((rr, embedding), dim=1)

    return in_features

def get_in_features_lin_edge(x, model, model_config, xnorm, n_neurons, device):
    """Build lin_edge input features from voltage and embeddings.

    Args:
        x: NeuronState — uses x.voltage.
    """
    voltage_all = x.voltage.unsqueeze(-1)
    signal_model_name = model_config.signal_model_name

    if signal_model_name == 'flyvis_B':
        perm_indices = torch.randperm(n_neurons, device=model.a.device)
        in_features = torch.cat((voltage_all, voltage_all, model.a, model.a[perm_indices]), dim=1)
        in_features_next = torch.cat((voltage_all, voltage_all * 1.05, model.a, model.a[perm_indices]), dim=1)
    else:
        # flyvis_A, flyvis_C, flyvis_D, and default
        in_features = torch.cat((voltage_all, model.a), dim=1)
        in_features_next = torch.cat((voltage_all * 1.05, model.a), dim=1)

    return in_features, in_features_next

def get_in_features(rr=None, embedding=None, model=[], model_name = [], max_radius=[]):

    if model.embedding_trial:
        embedding = torch.cat((embedding, model.b[0].repeat(embedding.shape[0], 1)), dim=1)

    match model_name:
        case 'PDE_A' | 'PDE_Cell_A':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
        case 'PDE_ParticleField_A':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
        case 'PDE_A_bis':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding, embedding), dim=1)
        case 'PDE_B' | 'PDE_Cell_B':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        case 'PDE_ParticleField_B':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        case 'PDE_GS':
            in_features = torch.cat(
                (rr[:, None] / max_radius, 0 * rr[:, None], rr[:, None] / max_radius, 10 ** embedding), dim=1)
        case 'PDE_G':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None],
                                     0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        case 'PDE_E':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding, embedding), dim=1)
        case 'PDE_N2' | 'PDE_N3' | 'PDE_N6' :
            in_features = rr[:, None]
        case 'PDE_N4' | 'PDE_N7' | 'PDE_N11':
            in_features = torch.cat((rr[:, None], embedding), dim=1)
        case 'PDE_N8':
            in_features = torch.cat((rr[:, None]*0, rr[:, None], embedding, embedding), dim=1)
        case 'PDE_N5':
            in_features = torch.cat((rr[:, None], embedding, embedding), dim=1)
        case 'PDE_K':
            in_features = torch.cat((0 * rr[:, None], rr[:, None] / max_radius), dim=1)
        case 'PDE_F':
            in_features = torch.cat((0 * rr[:, None], rr[:, None] / max_radius, rr[:, None] / max_radius, embedding, embedding), dim=-1)

    return in_features

def plot_training_flyvis(x_list, model, config, epoch, N, log_dir, device, cmap, type_list,
                         gt_weights, edges, n_neurons=None, n_neuron_types=None):
    from flyvis_gnn.plot import (
        plot_embedding, plot_lin_edge, plot_lin_phi, plot_weight_scatter,
        compute_all_corrected_weights, get_model_W,
    )

    if n_neurons is None:
        n_neurons = len(type_list)

    plt.style.use('default')

    # Plot 1: Embedding scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_embedding(ax, model, type_list, n_neuron_types, cmap)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.png", dpi=87)
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
    plt.savefig(f"./{log_dir}/tmp_training/matrix/raw_{epoch}_{N}.png",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Compute corrected weights
    corrected_W, _, _, _ = compute_all_corrected_weights(
        model, config, edges, x_list, device)

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
    plt.savefig(f"./{log_dir}/tmp_training/matrix/comparison_{epoch}_{N}.png",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Plot 4: Edge function visualization (lin_edge / MLP1)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_lin_edge(ax, model, config, n_neurons, type_list, cmap, device)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/MLP1/func_{epoch}_{N}.png", dpi=87)
    plt.close()

    # Plot 5: Phi function visualization (lin_phi / MLP0)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_lin_phi(ax, model, config, n_neurons, type_list, cmap, device)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/MLP0/func_{epoch}_{N}.png", dpi=87)
    plt.close()

    return r_squared

def set_trainable_parameters(model=[], lr_embedding=[], lr=[],  lr_update=[], lr_W=[], lr_modulation=[], learning_rate_NNR=[], learning_rate_NNR_f=[], learning_rate_NNR_E=[], learning_rate_NNR_b=[]):

    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params)

    # Only count model.a if it exists and requires gradients (not frozen by training_single_type)
    if hasattr(model, 'a') and model.a.requires_grad:
        n_total_params = n_total_params + torch.numel(model.a)


    if lr_update==[]:
        lr_update = lr

    param_groups = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if name == 'a':
                param_groups.append({'params': parameter, 'lr': lr_embedding})
            elif (name=='b') or ('lin_modulation' in name):
                param_groups.append({'params': parameter, 'lr': lr_modulation})
            elif 'lin_phi' in name:
                param_groups.append({'params': parameter, 'lr': lr_update})
            elif 'W' in name:
                param_groups.append({'params': parameter, 'lr': lr_W})
            elif 'NNR_f' in name:
                param_groups.append({'params': parameter, 'lr': learning_rate_NNR_f})
            elif 'NNR' in name:
                param_groups.append({'params': parameter, 'lr': learning_rate_NNR})
            else:
                param_groups.append({'params': parameter, 'lr': lr})

    # Use foreach=False to avoid CUDA device mismatch issues with multi-GPU setups
    optimizer = torch.optim.Adam(param_groups, foreach=False)

    return optimizer, n_total_params


def get_index_particles(x, n_neuron_types, dimension):
    index_particles = []
    for n in range(n_neuron_types):
        index = np.argwhere(x.neuron_type.detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())
    return index_particles

def sample_synaptic_data_and_predict(model, x_list, edges, n_runs, n_frames, time_step, device,
                            has_missing_activity=False, model_missing_activity=None,
                            has_neural_field=False, model_f=None,
                            run=None, k=None):
    """
    Sample data from x_list and get model predictions

    Args:
        model: trained GNN model
        x_list: list of data arrays [n_runs][n_frames]
        edges: edge indices for graph
        n_runs, n_frames, time_step: data dimensions
        device: torch device
        has_missing_activity: whether to fill missing activity
        model_missing_activity: model for missing activity (if needed)
        has_neural_field: whether to compute neural field
        model_f: field model (if needed)
        run: specific run index (if None, random)
        k: specific frame index (if None, random)

    Returns:
        dict with pred, in_features, x, dataset, data_id, k_batch
    """
    # Sample random run and frame if not specified
    if run is None:
        run = np.random.randint(n_runs)
    if k is None:
        k = np.random.randint(n_frames - 4 - time_step)

    # Get data
    x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)

    # Handle missing activity if needed
    if has_missing_activity and model_missing_activity is not None:
        pos = torch.argwhere(x[:, 3] == 6)
        if len(pos) > 0:
            t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
            missing_activity = model_missing_activity[run](t).squeeze()
            x[pos, 3] = missing_activity[pos]

    # Handle neural field if needed
    if has_neural_field and model_f is not None:
        t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
        x[:, 4] = model_f[run](t) ** 2

    # Create dataset (local import — generic signal models still use PyG)
    import torch_geometric.data as data
    dataset = data.Data(x=x, edge_index=edges)
    data_id = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run
    k_batch = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k

    # Get predictions
    pred, in_features = model(dataset, data_id=data_id, k=k_batch, return_all=True)

    return {
        'pred': pred,
        'in_features': in_features,
        'x': x,
        'dataset': dataset,
        'data_id': data_id,
        'k_batch': k_batch,
        'run': run,
        'k': k
    }

def analyze_odor_responses_by_neuron(model, x_list, edges, n_runs, n_frames, time_step, device,
                                     all_neuron_list, has_missing_activity=False, model_missing_activity=None,
                                     has_neural_field=False, model_f=None, n_samples=50, run=0):
    """
    Analyze odor responses by comparing lin_phi output with and without excitation
    Returns top responding neurons by name for each odor
    """
    odor_list = ['butanone', 'pentanedione', 'NaCL']

    # Store responses: difference between excitation and baseline
    odor_responses = {odor: [] for odor in odor_list}
    valid_samples = 0

    model.eval()
    with torch.no_grad():
        sample = 0
        while valid_samples < n_samples:
            result = sample_synaptic_data_and_predict(
                model, x_list, edges, n_runs, n_frames, time_step, device,
                has_missing_activity, model_missing_activity,
                has_neural_field, model_f, run
            )

            if not (torch.isnan(result['x']).any()):
                # Get baseline response (no excitation — generic signal models still use PyG)
                import torch_geometric.data as data
                x_baseline = result['x'].clone()
                x_baseline[:, 10:13] = 0  # no excitation
                dataset_baseline = data.Data(x=x_baseline, edge_index=edges)
                pred_baseline = model(dataset_baseline, data_id=result['data_id'],
                                      k=result['k_batch'], return_all=False)

                for i, odor in enumerate(odor_list):
                    x_odor = result['x'].clone()
                    x_odor[:, 10:13] = 0
                    x_odor[:, 10 + i] = 1  # activate specific odor

                    dataset_odor = data.Data(x=x_odor, edge_index=edges)
                    pred_odor = model(dataset_odor, data_id=result['data_id'],
                                      k=result['k_batch'], return_all=False)

                    odor_diff = pred_odor - pred_baseline
                    odor_responses[odor].append(odor_diff.cpu())

                valid_samples += 1

            sample += 1
            if sample > n_samples * 10:
                break

        # Convert to tensors [n_samples, n_neurons]
        for odor in odor_list:
            odor_responses[odor] = torch.stack(odor_responses[odor]).squeeze()

    # Identify top responding neurons for each odor
    top_neurons = {}
    for odor in odor_list:
        # Calculate mean response across samples for each neuron
        mean_response = torch.mean(odor_responses[odor], dim=0)  # [n_neurons]

        # Get top 3 responding neurons (highest positive response)
        top_20_indices = torch.topk(mean_response, k=20).indices.cpu().numpy()
        top_20_names = [all_neuron_list[idx] for idx in top_20_indices]
        top_20_values = [mean_response[idx].item() for idx in top_20_indices]

        top_neurons[odor] = {
            'names': top_20_names,
            'indices': top_20_indices.tolist(),
            'values': top_20_values
        }

        print(f"\ntop 20 responding neurons for {odor}:")
        for i, (name, idx, val) in enumerate(zip(top_20_names, top_20_indices, top_20_values)):
            print(f"  {i + 1}. {name} : {val:.4f}")

    return odor_responses  # Return only odor_responses to match original function signature

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


def check_dales_law(edges, weights, type_list=None, n_neurons=None, verbose=True, logger=None):
    """
    Check if synaptic weights satisfy Dale's Law.

    Dale's Law: Each neuron releases the same neurotransmitter at all synapses.
    This means all outgoing weights from a neuron should have the same sign.

    Parameters:
    -----------
    edges : torch.Tensor
        Edge index tensor of shape [2, n_edges] where edges[0] are source neurons
    weights : torch.Tensor
        Weight tensor of shape [n_edges] or [n_edges, 1]
    type_list : torch.Tensor, optional
        Neuron type indices of shape [n_neurons] or [n_neurons, 1]
    n_neurons : int, optional
        Total number of neurons (inferred from edges if not provided)
    verbose : bool, default=True
        If True, print detailed statistics
    logger : logging.Logger, optional
        Logger for recording results

    Returns:
    --------
    dict with keys:
        - 'n_excitatory': Number of purely excitatory neurons (all W>0)
        - 'n_inhibitory': Number of purely inhibitory neurons (all W<0)
        - 'n_mixed': Number of mixed neurons (violates Dale's Law)
        - 'n_violations': Number of Dale's Law violations
        - 'violations': List of dicts with violation details
        - 'neuron_signs': Dict mapping neuron_idx to sign (1=excitatory, -1=inhibitory, 0=mixed)
    """
    # Neuron type name mapping (from FlyVis connectome)
    index_to_name = {
        0: 'Am', 1: 'C2', 2: 'C3', 3: 'CT1(Lo1)', 4: 'CT1(M10)',
        5: 'L1', 6: 'L2', 7: 'L3', 8: 'L4', 9: 'L5',
        10: 'Lawf1', 11: 'Lawf2', 12: 'Mi1', 13: 'Mi15', 14: 'Mi4',
        15: 'Mi9', 16: 'T1', 17: 'T2', 18: 'T2a', 19: 'T3',
        20: 'T4a', 21: 'T4b', 22: 'T4c', 23: 'T4d', 24: 'T5a',
        25: 'T5b', 26: 'T5c', 27: 'T5d', 28: 'Tm1', 29: 'Tm2',
        30: 'Tm3', 31: 'Tm4', 32: 'Tm9', 33: 'TmY10', 34: 'TmY13',
        35: 'TmY14', 36: 'TmY15', 37: 'TmY18', 38: 'TmY3',
        39: 'TmY4', 40: 'TmY5a', 41: 'TmY9'
    }

    # Flatten weights if needed
    if weights.dim() > 1:
        weights = weights.squeeze()

    # Infer n_neurons if not provided
    if n_neurons is None:
        n_neurons = int(edges.max().item()) + 1

    # Check Dale's Law for each neuron
    dale_violations = []
    neuron_signs = {}

    for neuron_idx in range(n_neurons):
        # Find all outgoing edges from this neuron
        outgoing_mask = edges[0, :] == neuron_idx
        outgoing_weights = weights[outgoing_mask]

        if len(outgoing_weights) > 0:
            n_positive = (outgoing_weights > 0).sum().item()
            n_negative = (outgoing_weights < 0).sum().item()
            n_zero = (outgoing_weights == 0).sum().item()

            # Dale's Law: all non-zero weights should have same sign
            if n_positive > 0 and n_negative > 0:
                violation_info = {
                    'neuron': neuron_idx,
                    'n_positive': n_positive,
                    'n_negative': n_negative,
                    'n_zero': n_zero
                }

                # Add type information if available
                if type_list is not None:
                    type_id = type_list[neuron_idx].item()
                    type_name = index_to_name.get(type_id, f'Unknown_{type_id}')
                    violation_info['type_id'] = type_id
                    violation_info['type_name'] = type_name

                dale_violations.append(violation_info)
                neuron_signs[neuron_idx] = 0  # Mixed
            elif n_positive > 0:
                neuron_signs[neuron_idx] = 1  # Excitatory
            elif n_negative > 0:
                neuron_signs[neuron_idx] = -1  # Inhibitory
            else:
                neuron_signs[neuron_idx] = 0  # All zero

    # Compute statistics
    n_excitatory = sum(1 for s in neuron_signs.values() if s == 1)
    n_inhibitory = sum(1 for s in neuron_signs.values() if s == -1)
    n_mixed = sum(1 for s in neuron_signs.values() if s == 0)

    # Print results if verbose
    if verbose:
        print("\n=== Dale's Law Check ===")
        print(f"Total neurons: {n_neurons}")
        print(f"Excitatory neurons (all W>0): {n_excitatory} ({100*n_excitatory/n_neurons:.1f}%)")
        print(f"Inhibitory neurons (all W<0): {n_inhibitory} ({100*n_inhibitory/n_neurons:.1f}%)")
        print(f"Mixed/zero neurons (violates Dale's Law): {n_mixed} ({100*n_mixed/n_neurons:.1f}%)")
        print(f"Dale's Law violations: {len(dale_violations)}")

        if logger:
            logger.info("=== Dale's Law Check ===")
            logger.info(f"Total neurons: {n_neurons}")
            logger.info(f"Excitatory: {n_excitatory} ({100*n_excitatory/n_neurons:.1f}%)")
            logger.info(f"Inhibitory: {n_inhibitory} ({100*n_inhibitory/n_neurons:.1f}%)")
            logger.info(f"Violations: {len(dale_violations)}")

        if len(dale_violations) > 0:
            print("\nFirst 10 violations:")
            for i, v in enumerate(dale_violations[:10]):
                if 'type_name' in v:
                    print(f"  Neuron {v['neuron']} ({v['type_name']}): "
                          f"{v['n_positive']} positive, {v['n_negative']} negative, {v['n_zero']} zero weights")
                    if logger:
                        logger.info(f"  Neuron {v['neuron']} ({v['type_name']}): "
                                    f"{v['n_positive']} positive, {v['n_negative']} negative")
                else:
                    print(f"  Neuron {v['neuron']}: "
                          f"{v['n_positive']} positive, {v['n_negative']} negative, {v['n_zero']} zero weights")

            # Group violations by neuron type if available
            if type_list is not None and any('type_name' in v for v in dale_violations):
                type_violations = Counter([v['type_name'] for v in dale_violations if 'type_name' in v])
                print("\nViolations by neuron type:")
                for type_name, count in sorted(type_violations.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {type_name}: {count} violations")
                    if logger:
                        logger.info(f"  {type_name}: {count} violations")
        else:
            print("✓ Weights perfectly satisfy Dale's Law!")
            if logger:
                logger.info("✓ Weights perfectly satisfy Dale's Law!")

        print("=" * 60 + "\n")

    return {
        'n_excitatory': n_excitatory,
        'n_inhibitory': n_inhibitory,
        'n_mixed': n_mixed,
        'n_violations': len(dale_violations),
        'violations': dale_violations,
        'neuron_signs': neuron_signs
    }


def analyze_data_svd(data, output_folder, config=None, max_components=100, logger=None, max_data_size=10_000_000, max_neurons=1024, is_flyvis=False, style=None, save_in_subfolder=True, log_file=None):
    """
    Perform SVD analysis on activity data and external_input/visual stimuli (if present).
    Uses randomized SVD for large datasets for efficiency.
    Subsamples frames if data is too large.

    Args:
        data: NeuronTimeSeries (uses .voltage and .stimulus) or legacy (T, N, 9) numpy array.
        output_folder: path to save plots
        config: config object (optional, for metadata)
        max_components: maximum number of SVD components to compute
        logger: optional logger (for training)
        max_data_size: maximum data size before subsampling (default 10M elements)
        max_neurons: maximum number of neurons before subsampling (default 1024)
        is_flyvis: if True, use "visual stimuli" label instead of "external input"
        style: matplotlib style to use (e.g., 'dark_background' for dark mode)
        save_in_subfolder: if True, save to results/ subfolder; if False, save directly to output_folder
        log_file: optional file handle to write results

    Returns:
        dict with SVD analysis results
    """
    from sklearn.utils.extmath import randomized_svd
    from flyvis_gnn.neuron_state import NeuronTimeSeries

    # Extract activity and stimulus as (T, N) numpy arrays
    if isinstance(data, NeuronTimeSeries):
        activity_full = data.voltage.cpu().numpy()      # (T, N)
        stimulus_full = data.stimulus.cpu().numpy()      # (T, N)
    else:
        # Legacy packed (T, N, 9) numpy array
        activity_full = data[:, :, 3]
        stimulus_full = data[:, :, 4] if data.shape[2] > 4 else None

    n_frames, n_neurons = activity_full.shape
    has_stimulus = stimulus_full is not None
    results = {}

    import re
    def log_print(msg):
        if logger:
            logger.info(msg)
        if log_file:
            # strip ANSI color codes for log file
            clean_msg = re.sub(r'\033\[[0-9;]*m', '', msg)
            log_file.write(clean_msg + '\n')

    # subsample neurons if too many
    if n_neurons > max_neurons:
        neuron_subsample = int(np.ceil(n_neurons / max_neurons))
        neuron_indices = np.arange(0, n_neurons, neuron_subsample)
        activity_full = activity_full[:, neuron_indices]
        if has_stimulus:
            stimulus_full = stimulus_full[:, neuron_indices]
        n_neurons_sampled = len(neuron_indices)
        log_print(f"subsampling neurons: {n_neurons} -> {n_neurons_sampled} (every {neuron_subsample}th)")
        n_neurons = n_neurons_sampled

    # subsample frames if data is too large
    data_size = n_frames * n_neurons
    if data_size > max_data_size:
        subsample_factor = int(np.ceil(data_size / max_data_size))
        frame_indices = np.arange(0, n_frames, subsample_factor)
        activity_sampled = activity_full[frame_indices]
        stimulus_sampled = stimulus_full[frame_indices] if has_stimulus else None
        n_frames_sampled = len(frame_indices)
        log_print(f"subsampling frames: {n_frames} -> {n_frames_sampled} (every {subsample_factor}th)")
        data_size_sampled = n_frames_sampled * n_neurons
    else:
        activity_sampled = activity_full
        stimulus_sampled = stimulus_full
        n_frames_sampled = n_frames
        data_size_sampled = data_size
        subsample_factor = 1

    # decide whether to use randomized SVD
    use_randomized = data_size_sampled > 1e6  # use randomized for > 1M elements

    # store data size info for later printing with results
    if subsample_factor > 1:
        data_info = f"using {n_frames_sampled:,} of {n_frames:,} frames ({n_neurons:,} neurons)"
    else:
        data_info = f"using full data ({n_frames:,} frames, {n_neurons:,} neurons)"

    # save current style context and apply new style if provided
    # We use context manager approach to avoid resetting global style
    if style:
        plt.style.use(style)

    # main color based on style
    mc = 'w' if style == 'dark_background' else 'k'
    bg_color = 'k' if style == 'dark_background' else 'w'

    # font sizes
    TITLE_SIZE = 16
    LABEL_SIZE = 14
    TICK_SIZE = 12
    LEGEND_SIZE = 12

    # prepare figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=bg_color)
    for ax in axes.flat:
        ax.set_facecolor(bg_color)

    # 1. analyze activity (u)
    activity = activity_sampled  # shape: (n_frames_sampled, n_neurons)
    log_print("--- activity ---")
    log_print(f"  shape: {activity.shape}")
    log_print(f"  range: [{activity.min():.3f}, {activity.max():.3f}]")

    k = min(max_components, min(n_frames_sampled, n_neurons) - 1)

    if use_randomized:
        U_act, S_act, Vt_act = randomized_svd(activity, n_components=k, random_state=42)
    else:
        U_act, S_act, Vt_act = np.linalg.svd(activity, full_matrices=False)
        S_act = S_act[:k]

    # compute cumulative variance
    cumvar_act = np.cumsum(S_act**2) / np.sum(S_act**2)
    rank_90_act = np.searchsorted(cumvar_act, 0.90) + 1
    rank_99_act = np.searchsorted(cumvar_act, 0.99) + 1

    log_print(f"  effective rank (90% var): {rank_90_act}")
    log_print(f"  effective rank (99% var): \033[92m{rank_99_act}\033[0m")

    # compression ratio
    if rank_99_act < k:
        compression_act = (n_frames * n_neurons) / (rank_99_act * (n_frames + n_neurons))
        log_print(f"  compression (rank-{rank_99_act}): {compression_act:.1f}x")
    else:
        log_print("  compression: need more components to reach 99% variance")

    results['activity'] = {
        'singular_values': S_act,
        'cumulative_variance': cumvar_act,
        'rank_90': rank_90_act,
        'rank_99': rank_99_act,
    }

    # plot activity SVD
    ax = axes[0, 0]
    ax.semilogy(S_act, color=mc, lw=1.5)
    ax.set_xlabel('component', fontsize=LABEL_SIZE)
    ax.set_ylabel('singular value', fontsize=LABEL_SIZE)
    ax.set_title('activity: singular values', fontsize=TITLE_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(cumvar_act, color=mc, lw=1.5)
    ax.axhline(0.90, color='orange', ls='--', label='90%')
    ax.axhline(0.99, color='green', ls='--', label='99%')
    ax.axvline(rank_90_act, color='orange', ls=':', alpha=0.7)
    ax.axvline(rank_99_act, color='green', ls=':', alpha=0.7)
    ax.set_xlabel('component', fontsize=LABEL_SIZE)
    ax.set_ylabel('cumulative variance', fontsize=LABEL_SIZE)
    ax.set_title(f'activity: rank(90%)={rank_90_act}, rank(99%)={rank_99_act}', fontsize=TITLE_SIZE)
    ax.legend(loc='lower right', fontsize=LEGEND_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # 2. Analyze external_input / visual stimuli (if present and non-zero) - column 4
    # Determine label based on is_flyvis parameter
    n_input_neurons = None
    if is_flyvis:
        n_input_neurons = getattr(config.simulation, 'n_input_neurons', None) if config else None
    input_label = "visual stimuli" if is_flyvis else "external input"

    if has_stimulus:
        # for visual stimuli, only analyze input neurons (first n_input_neurons)
        if is_flyvis and n_input_neurons is not None and n_input_neurons < n_neurons:
            external_input = stimulus_sampled[:, :n_input_neurons]
            log_print(f"--- {input_label} (first {n_input_neurons} input neurons) ---")
        else:
            external_input = stimulus_sampled

        # check if external_input has actual signal
        ext_range = external_input.max() - external_input.min()
        if ext_range > 1e-6:
            if not (is_flyvis and n_input_neurons is not None):
                log_print(f"--- {input_label} ---")
            log_print(f"  shape: {external_input.shape}")
            log_print(f"  range: [{external_input.min():.3f}, {external_input.max():.3f}]")

            if use_randomized:
                U_ext, S_ext, Vt_ext = randomized_svd(external_input, n_components=k, random_state=42)
            else:
                U_ext, S_ext, Vt_ext = np.linalg.svd(external_input, full_matrices=False)
                S_ext = S_ext[:k]

            cumvar_ext = np.cumsum(S_ext**2) / np.sum(S_ext**2)
            rank_90_ext = np.searchsorted(cumvar_ext, 0.90) + 1
            rank_99_ext = np.searchsorted(cumvar_ext, 0.99) + 1

            log_print(f"  effective rank (90% var): {rank_90_ext}")
            log_print(f"  effective rank (99% var): \033[92m{rank_99_ext}\033[0m")

            if rank_99_ext < k:
                compression_ext = (n_frames * n_neurons) / (rank_99_ext * (n_frames + n_neurons))
                log_print(f"  compression (rank-{rank_99_ext}): {compression_ext:.1f}x")
            else:
                log_print("  compression: need more components to reach 99% variance")

            results_key = 'visual_stimuli' if is_flyvis else 'external_input'
            results[results_key] = {
                'singular_values': S_ext,
                'cumulative_variance': cumvar_ext,
                'rank_90': rank_90_ext,
                'rank_99': rank_99_ext,
            }

            # plot external_input / visual stimuli SVD
            ax = axes[1, 0]
            ax.semilogy(S_ext, color=mc, lw=1.5)
            ax.set_xlabel('component', fontsize=LABEL_SIZE)
            ax.set_ylabel('singular value', fontsize=LABEL_SIZE)
            ax.set_title(f'{input_label}: singular values', fontsize=TITLE_SIZE)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            ax.plot(cumvar_ext, color=mc, lw=1.5)
            ax.axhline(0.90, color='orange', ls='--', label='90%')
            ax.axhline(0.99, color='green', ls='--', label='99%')
            ax.axvline(rank_90_ext, color='orange', ls=':', alpha=0.7)
            ax.axvline(rank_99_ext, color='green', ls=':', alpha=0.7)
            ax.set_xlabel('component', fontsize=LABEL_SIZE)
            ax.set_ylabel('cumulative variance', fontsize=LABEL_SIZE)
            ax.set_title(f'{input_label}: rank(90%)={rank_90_ext}, rank(99%)={rank_99_ext}', fontsize=TITLE_SIZE)
            ax.legend(loc='lower right', fontsize=LEGEND_SIZE)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.3)
        else:
            log_print(f"--- {input_label} ---")
            log_print("  no external input found (range < 1e-6)")
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)
            results_key = 'visual_stimuli' if is_flyvis else 'external_input'
            results[results_key] = None
    else:
        log_print(f"--- {input_label} ---")
        log_print("  not present in data")
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)
        results_key = 'visual_stimuli' if is_flyvis else 'external_input'
        results[results_key] = None

    plt.tight_layout()

    # save plot
    if save_in_subfolder:
        save_folder = os.path.join(output_folder, 'results')
        os.makedirs(save_folder, exist_ok=True)
    else:
        save_folder = output_folder
    save_path = os.path.join(save_folder, 'svd_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=bg_color)
    plt.close()

    # print SVD results: data info (white) + rank results (green)
    ext_key = 'visual_stimuli' if is_flyvis else 'external_input'
    if results.get(ext_key):
        print(f"{data_info}, \033[92mactivity rank(99%)={results['activity']['rank_99']}, {ext_key} rank(99%)={results[ext_key]['rank_99']}\033[0m")
    else:
        print(f"{data_info}, \033[92mactivity rank(99%)={results['activity']['rank_99']}\033[0m")

    return results


def save_exploration_artifacts(root_dir, exploration_dir, config, config_file_, pre_folder, iteration,
                               iter_in_block=1, block_number=1):
    """
    Save exploration artifacts for Claude analysis.

    Args:
        root_dir: Root directory of the project
        exploration_dir: Base directory for exploration artifacts
        config: Configuration object
        config_file_: Config file name (without extension)
        pre_folder: Prefix folder for config
        iteration: Current iteration number
        iter_in_block: Iteration number within current block (1-indexed)
        block_number: Current block number (1-indexed)

    Returns:
        dict with paths to saved directories
    """
    import glob
    import shutil
    import matplotlib.image as mpimg

    config_save_dir = f"{exploration_dir}/config"
    scatter_save_dir = f"{exploration_dir}/connectivity_scatter"
    matrix_save_dir = f"{exploration_dir}/connectivity_matrix"
    activity_save_dir = f"{exploration_dir}/activity"
    mlp_save_dir = f"{exploration_dir}/mlp"
    tree_save_dir = f"{exploration_dir}/exploration_tree"
    protocol_save_dir = f"{exploration_dir}/protocol"
    kinograph_save_dir = f"{exploration_dir}/kinograph"
    embedding_save_dir = f"{exploration_dir}/embedding"

    # create directories at start of experiment (clear only on iteration 1)
    if iteration == 1:
        # clear and recreate exploration folder
        if os.path.exists(exploration_dir):
            shutil.rmtree(exploration_dir)
    # always ensure directories exist (for resume support)
    os.makedirs(config_save_dir, exist_ok=True)
    os.makedirs(scatter_save_dir, exist_ok=True)
    os.makedirs(matrix_save_dir, exist_ok=True)
    os.makedirs(activity_save_dir, exist_ok=True)
    os.makedirs(mlp_save_dir, exist_ok=True)
    os.makedirs(tree_save_dir, exist_ok=True)
    os.makedirs(protocol_save_dir, exist_ok=True)
    os.makedirs(kinograph_save_dir, exist_ok=True)
    os.makedirs(embedding_save_dir, exist_ok=True)

    # determine if this is first iteration of a block
    is_block_start = (iter_in_block == 1)

    # save config file only at first iteration of each block
    if is_block_start:
        src_config = f"{root_dir}/config/{pre_folder}{config_file_}.yaml"
        dst_config = f"{config_save_dir}/block_{block_number:03d}.yaml"
        if os.path.exists(src_config):
            shutil.copy2(src_config, dst_config)

    # save connectivity scatterplot (most recent comparison_*.png from matrix folder)
    matrix_dir = f"{root_dir}/log/{pre_folder}{config_file_}/tmp_training/matrix"
    scatter_files = glob.glob(f"{matrix_dir}/comparison_*.png")
    if scatter_files:
        # get most recent file
        latest_scatter = max(scatter_files, key=os.path.getmtime)
        dst_scatter = f"{scatter_save_dir}/iter_{iteration:03d}.png"
        shutil.copy2(latest_scatter, dst_scatter)

    # save connectivity matrix heatmap only at first iteration of each block
    data_folder = f"{root_dir}/graphs_data/{config.dataset}"
    if is_block_start:
        src_matrix = f"{data_folder}/connectivity_matrix.png"
        dst_matrix = f"{matrix_save_dir}/block_{block_number:03d}.png"
        if os.path.exists(src_matrix):
            shutil.copy2(src_matrix, dst_matrix)

    # save activity plot only at first iteration of each block
    activity_path = f"{data_folder}/activity.png"
    if is_block_start:
        dst_activity = f"{activity_save_dir}/block_{block_number:03d}.png"
        if os.path.exists(activity_path):
            shutil.copy2(activity_path, dst_activity)

    # save combined MLP plot (MLP0 + MLP1 side by side) using PNG files from results
    results_dir = f"{root_dir}/log/{pre_folder}{config_file_}/results"
    src_mlp0 = f"{results_dir}/MLP0.png"
    src_mlp1 = f"{results_dir}/MLP1_corrected.png"
    if os.path.exists(src_mlp0) and os.path.exists(src_mlp1):
        try:
            # Load PNG images
            img0 = mpimg.imread(src_mlp0)
            img1 = mpimg.imread(src_mlp1)

            # Create combined figure
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            axes[0].imshow(img0)
            axes[0].set_title('MLP0 (φ)', fontsize=12)
            axes[0].axis('off')
            axes[1].imshow(img1)
            axes[1].set_title('MLP1 (edge)', fontsize=12)
            axes[1].axis('off')
            plt.tight_layout()
            plt.savefig(f"{mlp_save_dir}/iter_{iteration:03d}_MLP.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"\033[93mwarning: could not combine MLP plots: {e}\033[0m")

    # save kinograph montage (every iteration)
    src_montage = f"{results_dir}/kinograph_montage.png"
    if os.path.exists(src_montage):
        shutil.copy2(src_montage, f"{kinograph_save_dir}/iter_{iteration:03d}.png")

    # save embedding plot (latest from tmp_training/embedding/)
    embedding_dir = f"{root_dir}/log/{pre_folder}{config_file_}/tmp_training/embedding"
    if os.path.isdir(embedding_dir):
        embed_files = glob.glob(f"{embedding_dir}/*.png")
        if embed_files:
            latest_embed = max(embed_files, key=os.path.getmtime)
            shutil.copy2(latest_embed, f"{embedding_save_dir}/iter_{iteration:03d}.png")
            # also save per-block snapshot at block start
            if is_block_start:
                shutil.copy2(latest_embed, f"{embedding_save_dir}/block_{block_number:03d}.png")

    return {
        'config_save_dir': config_save_dir,
        'scatter_save_dir': scatter_save_dir,
        'matrix_save_dir': matrix_save_dir,
        'activity_save_dir': activity_save_dir,
        'mlp_save_dir': mlp_save_dir,
        'tree_save_dir': tree_save_dir,
        'protocol_save_dir': protocol_save_dir,
        'kinograph_save_dir': kinograph_save_dir,
        'embedding_save_dir': embedding_save_dir,
        'activity_path': activity_path
    }


def save_exploration_artifacts_flyvis(root_dir, exploration_dir, config, config_file_, pre_folder, iteration,
                                      iter_in_block=1, block_number=1):
    """
    Save exploration artifacts for flyvis Claude analysis.

    Flyvis-specific panels: connectivity_scatter, embedding, tau_scatter, V_rest_scatter,
    MLP0, MLP1, UMAP. Plus activity and connectivity_matrix at block starts.

    Args:
        root_dir: Root directory of the project
        exploration_dir: Base directory for exploration artifacts
        config: Configuration object
        config_file_: Config file name (without extension)
        pre_folder: Prefix folder for config
        iteration: Current iteration number
        iter_in_block: Iteration number within current block (1-indexed)
        block_number: Current block number (1-indexed)

    Returns:
        dict with paths to saved directories
    """
    import glob
    import shutil

    config_save_dir = f"{exploration_dir}/config"
    connectivity_scatter_dir = f"{exploration_dir}/connectivity_scatter"
    connectivity_matrix_dir = f"{exploration_dir}/connectivity_matrix"
    activity_dir = f"{exploration_dir}/activity"
    embedding_dir = f"{exploration_dir}/embedding"
    tau_scatter_dir = f"{exploration_dir}/tau_scatter"
    v_rest_scatter_dir = f"{exploration_dir}/V_rest_scatter"
    mlp0_dir = f"{exploration_dir}/MLP0"
    mlp1_dir = f"{exploration_dir}/MLP1"
    umap_dir = f"{exploration_dir}/UMAP"
    tree_save_dir = f"{exploration_dir}/exploration_tree"
    protocol_save_dir = f"{exploration_dir}/protocol"
    memory_save_dir = f"{exploration_dir}/memory"

    # create directories at start of experiment (clear only on iteration 1)
    if iteration == 1:
        if os.path.exists(exploration_dir):
            shutil.rmtree(exploration_dir)
    # always ensure directories exist (for resume support)
    for d in [config_save_dir, connectivity_scatter_dir, connectivity_matrix_dir,
              activity_dir, embedding_dir, tau_scatter_dir, v_rest_scatter_dir,
              mlp0_dir, mlp1_dir, umap_dir, tree_save_dir, protocol_save_dir, memory_save_dir]:
        os.makedirs(d, exist_ok=True)

    is_block_start = (iter_in_block == 1)

    # Results directory
    results_dir = f"{root_dir}/log/{pre_folder}{config_file_}/results"

    # Extract config indices for filename matching
    dataset = config.dataset if hasattr(config, 'dataset') else config_file_
    config_indices = dataset.split('flyvis_')[1] if 'flyvis_' in dataset else 'evolution'

    # --- Per-iteration panels ---

    # connectivity_scatter: weights_comparison_corrected.png (matches connectivity_R2 in analysis.log)
    src = f"{results_dir}/weights_comparison_corrected.png"
    if not os.path.exists(src):
        src = f"{results_dir}/weights_comparison_raw.png"
    if os.path.exists(src):
        shutil.copy2(src, f"{connectivity_scatter_dir}/iter_{iteration:03d}.png")

    # embedding: embedding_{indices}.png
    embed_files = glob.glob(f"{results_dir}/embedding_*.png")
    # Filter out augmented (UMAP) files
    embed_files = [f for f in embed_files if 'augmented' not in f]
    if embed_files:
        latest = max(embed_files, key=os.path.getmtime)
        shutil.copy2(latest, f"{embedding_dir}/iter_{iteration:03d}.png")
        if is_block_start:
            shutil.copy2(latest, f"{embedding_dir}/block_{block_number:03d}.png")

    # tau_scatter: tau_comparison_{indices}.png
    tau_files = glob.glob(f"{results_dir}/tau_comparison_*.png")
    if tau_files:
        latest = max(tau_files, key=os.path.getmtime)
        shutil.copy2(latest, f"{tau_scatter_dir}/iter_{iteration:03d}.png")

    # V_rest_scatter: V_rest_comparison_{indices}.png
    vrest_files = glob.glob(f"{results_dir}/V_rest_comparison_*.png")
    if vrest_files:
        latest = max(vrest_files, key=os.path.getmtime)
        shutil.copy2(latest, f"{v_rest_scatter_dir}/iter_{iteration:03d}.png")

    # MLP0: MLP0_{indices}.png (all neurons overlay, not per-type like MLP0_Tm30.png)
    mlp0_src = f"{results_dir}/MLP0_{config_indices}.png"
    if os.path.exists(mlp0_src):
        shutil.copy2(mlp0_src, f"{mlp0_dir}/iter_{iteration:03d}.png")
    else:
        # Fallback: glob but filter out domain/params/per-type variants
        mlp0_files = glob.glob(f"{results_dir}/MLP0_*.png")
        mlp0_files = [f for f in mlp0_files if '_domain' not in f and '_params' not in f]
        # Per-type files have alphabetic names (e.g. MLP0_R1.png), indices have digits (e.g. MLP0_62_1.png)
        mlp0_files = [f for f in mlp0_files if any(c.isdigit() for c in os.path.basename(f).replace('MLP0_', ''))]
        if mlp0_files:
            latest = max(mlp0_files, key=os.path.getmtime)
            shutil.copy2(latest, f"{mlp0_dir}/iter_{iteration:03d}.png")

    # MLP1: MLP1_{indices}.png
    mlp1_files = glob.glob(f"{results_dir}/MLP1_*.png")
    # Filter out domain/slope variants
    mlp1_files = [f for f in mlp1_files if '_domain' not in f and '_slope' not in f]
    if mlp1_files:
        latest = max(mlp1_files, key=os.path.getmtime)
        shutil.copy2(latest, f"{mlp1_dir}/iter_{iteration:03d}.png")

    # UMAP: embedding_augmented_{indices}.png
    umap_files = glob.glob(f"{results_dir}/embedding_augmented_*.png")
    if umap_files:
        latest = max(umap_files, key=os.path.getmtime)
        shutil.copy2(latest, f"{umap_dir}/iter_{iteration:03d}.png")

    # --- Per-block panels (block start only) ---

    data_folder = f"{root_dir}/graphs_data/{config.dataset}"
    activity_path = f"{data_folder}/activity.png"

    if is_block_start:
        # connectivity_matrix
        src_matrix = f"{data_folder}/connectivity_matrix.png"
        if os.path.exists(src_matrix):
            shutil.copy2(src_matrix, f"{connectivity_matrix_dir}/block_{block_number:03d}.png")

        # activity
        if os.path.exists(activity_path):
            shutil.copy2(activity_path, f"{activity_dir}/block_{block_number:03d}.png")

    return {
        'config_save_dir': config_save_dir,
        'connectivity_scatter_dir': connectivity_scatter_dir,
        'connectivity_matrix_dir': connectivity_matrix_dir,
        'activity_dir': activity_dir,
        'embedding_dir': embedding_dir,
        'tau_scatter_dir': tau_scatter_dir,
        'v_rest_scatter_dir': v_rest_scatter_dir,
        'mlp0_dir': mlp0_dir,
        'mlp1_dir': mlp1_dir,
        'umap_dir': umap_dir,
        'tree_save_dir': tree_save_dir,
        'protocol_save_dir': protocol_save_dir,
        'memory_save_dir': memory_save_dir,
        'activity_path': activity_path
    }


class LossRegularizer:
    """
    Handles all regularization terms, coefficient annealing, and history tracking.

    Usage:
        regularizer = LossRegularizer(train_config, model_config, activity_column=6,
                                       plot_frequency=100, n_neurons=1000, trainer_type='signal')

        for epoch in range(n_epochs):
            regularizer.set_epoch(epoch)

            for N in range(Niter):
                regularizer.reset_iteration()

                pred, in_features, msg = model(batch, data_id=data_id, return_all=True)

                regul_loss = regularizer.compute(model, x, in_features, ids, ids_batch, edges, device)
                loss = pred_loss + regul_loss
    """

    # Components tracked in history
    COMPONENTS = [
        'W_L1', 'W_L2', 'W_sign',
        'edge_diff', 'edge_norm', 'edge_weight', 'phi_weight',
        'phi_zero', 'update_diff', 'update_msg_diff', 'update_u_diff', 'update_msg_sign',
        'missing_activity', 'model_a', 'model_b', 'modulation'
    ]

    def __init__(self, train_config, model_config, activity_column: int,
                 plot_frequency: int, n_neurons: int, trainer_type: str = 'signal'):
        """
        Args:
            train_config: TrainingConfig with coeff_* values
            model_config: GraphModelConfig with model settings
            activity_column: Column index for activity (6 for signal, 3 for flyvis)
            plot_frequency: How often to record to history
            n_neurons: Number of neurons for normalization
            trainer_type: 'signal' or 'flyvis' - controls annealing behavior
        """
        self.train_config = train_config
        self.model_config = model_config
        self.activity_column = activity_column
        self.plot_frequency = plot_frequency
        self.n_neurons = n_neurons
        self.trainer_type = trainer_type

        # Current epoch (for annealing)
        self.epoch = 0

        # Iteration counter
        self.iter_count = 0

        # Per-iteration accumulator
        self._iter_total = 0.0
        self._iter_tracker = {}

        # History for plotting
        self._history = {comp: [] for comp in self.COMPONENTS}
        self._history['regul_total'] = []

        # Cache coefficients
        self._coeffs = {}
        self._update_coeffs()

    def _update_coeffs(self):
        """Recompute coefficients based on current epoch (annealing for flyvis only)."""
        tc = self.train_config
        epoch = self.epoch

        # Two-phase training support (like ParticleGraph data_train_synaptic2)
        n_epochs_init = getattr(tc, 'n_epochs_init', 0)
        first_coeff_L1 = getattr(tc, 'first_coeff_L1', tc.coeff_W_L1)

        if self.trainer_type == 'flyvis':
            # Flyvis: annealed coefficients
            self._coeffs['W_L1'] = tc.coeff_W_L1 * (1 - np.exp(-tc.coeff_W_L1_rate * epoch))
            self._coeffs['edge_weight_L1'] = tc.coeff_edge_weight_L1 * (1 - np.exp(-tc.coeff_edge_weight_L1_rate ** epoch))
            self._coeffs['phi_weight_L1'] = tc.coeff_phi_weight_L1 * (1 - np.exp(-tc.coeff_phi_weight_L1_rate * epoch))
        else:
            # Signal: two-phase training if n_epochs_init > 0
            if n_epochs_init > 0 and epoch < n_epochs_init:
                # Phase 1: use first_coeff_L1 (typically 0 or small)
                self._coeffs['W_L1'] = first_coeff_L1
            else:
                # Phase 2: use coeff_W_L1 (target L1)
                self._coeffs['W_L1'] = tc.coeff_W_L1
            self._coeffs['edge_weight_L1'] = tc.coeff_edge_weight_L1
            self._coeffs['phi_weight_L1'] = tc.coeff_phi_weight_L1

        # Non-annealed coefficients (same for both)
        self._coeffs['W_L2'] = tc.coeff_W_L2
        self._coeffs['W_sign'] = tc.coeff_W_sign
        # Two-phase: edge_diff is active in phase 1, disabled in phase 2
        if n_epochs_init > 0 and epoch >= n_epochs_init:
            self._coeffs['edge_diff'] = 0  # Phase 2: no monotonicity constraint
        else:
            self._coeffs['edge_diff'] = tc.coeff_edge_diff
        self._coeffs['edge_norm'] = tc.coeff_edge_norm
        self._coeffs['edge_weight_L2'] = tc.coeff_edge_weight_L2
        self._coeffs['phi_weight_L2'] = tc.coeff_phi_weight_L2
        self._coeffs['phi_zero'] = tc.coeff_lin_phi_zero
        self._coeffs['update_diff'] = tc.coeff_update_diff
        self._coeffs['update_msg_diff'] = tc.coeff_update_msg_diff
        self._coeffs['update_u_diff'] = tc.coeff_update_u_diff
        self._coeffs['update_msg_sign'] = tc.coeff_update_msg_sign
        self._coeffs['missing_activity'] = tc.coeff_missing_activity
        self._coeffs['model_a'] = tc.coeff_model_a
        self._coeffs['model_b'] = tc.coeff_model_b
        self._coeffs['modulation'] = tc.coeff_lin_modulation

    def set_epoch(self, epoch: int, plot_frequency: int = None):
        """Set current epoch and update annealed coefficients."""
        self.epoch = epoch
        self._update_coeffs()
        if plot_frequency is not None:
            self.plot_frequency = plot_frequency
        # Reset iteration counter at epoch start
        self.iter_count = 0

    def reset_iteration(self):
        """Reset per-iteration accumulator."""
        self.iter_count += 1
        self._iter_total = 0.0
        self._iter_tracker = {comp: 0.0 for comp in self.COMPONENTS}
        # Flag to ensure W_L1 is only applied once per iteration (not per batch item)
        self._W_L1_applied_this_iter = False

    def should_record(self) -> bool:
        """Check if we should record to history this iteration."""
        return (self.iter_count % self.plot_frequency == 0) or (self.iter_count == 1)

    def needs_update_regul(self) -> bool:
        """Check if update regularization is needed (update_diff, update_msg_diff, update_u_diff, or update_msg_sign)."""
        return (self._coeffs['update_diff'] > 0 or
                self._coeffs['update_msg_diff'] > 0 or
                self._coeffs['update_u_diff'] > 0 or
                self._coeffs['update_msg_sign'] > 0)

    def _add(self, name: str, term):
        """Internal: accumulate a regularization term."""
        if term is None:
            return
        val = term.item() if hasattr(term, 'item') else float(term)
        self._iter_total += val
        if name in self._iter_tracker:
            self._iter_tracker[name] += val

    def compute(self, model, x, in_features, ids, ids_batch, edges, device,
                xnorm=1.0, index_weight=None):
        """
        Compute all regularization terms internally.

        Args:
            model: The neural network model
            x: NeuronState — only voltage is used
            in_features: Features for lin_phi (from model forward pass, can be None)
            ids: Sample indices for regularization
            ids_batch: Batch indices
            edges: Edge tensor
            device: Torch device
            xnorm: Normalization value
            index_weight: Index for W_sign computation (signal only)

        Returns:
            Total regularization loss tensor
        """
        tc = self.train_config
        mc = self.model_config
        n_neurons = self.n_neurons
        total_regul = torch.tensor(0.0, device=device)

        # Get model W (handle multi-run case not working here)
        # For low_rank_factorization, compute W from WL @ WR to allow gradient flow

        # --- W regularization ---

        low_rank = getattr(model, 'low_rank_factorization', False)
        if low_rank and hasattr(model, 'WL') and hasattr(model, 'WR'):

            if self._coeffs['W_L1'] > 0 and not self._W_L1_applied_this_iter:
                regul_term = (model.WL.norm(1) + model.WR) * self._coeffs['W_L1']
                total_regul = total_regul + regul_term
                self._add('W_L1', regul_term)
                self._W_L1_applied_this_iter = True
        else:

            # W_L1: Apply only once per iteration (not per batch item)
            if self._coeffs['W_L1'] > 0 and not self._W_L1_applied_this_iter:
                regul_term = model.W.norm(1) * self._coeffs['W_L1']
                total_regul = total_regul + regul_term
                self._add('W_L1', regul_term)
                self._W_L1_applied_this_iter = True

            if self._coeffs['W_L2'] > 0 and not self._W_L1_applied_this_iter:
                regul_term = model.W.norm(2) * self._coeffs['W_L2']
                total_regul = total_regul + regul_term
                self._add('W_L2', regul_term)

        # --- Edge/Phi weight regularization ---
        if (self._coeffs['edge_weight_L1'] + self._coeffs['edge_weight_L2']) > 0:
            for param in model.lin_edge.parameters():
                regul_term = param.norm(1) * self._coeffs['edge_weight_L1'] + param.norm(2) * self._coeffs['edge_weight_L2']
                total_regul = total_regul + regul_term
                self._add('edge_weight', regul_term)

        if (self._coeffs['phi_weight_L1'] + self._coeffs['phi_weight_L2']) > 0:
            for param in model.lin_phi.parameters():
                regul_term = param.norm(1) * self._coeffs['phi_weight_L1'] + param.norm(2) * self._coeffs['phi_weight_L2']
                total_regul = total_regul + regul_term
                self._add('phi_weight', regul_term)

        # --- phi_zero regularization ---
        if self._coeffs['phi_zero'] > 0:
            in_features_phi = get_in_features_update(rr=None, model=model, device=device)
            func_phi = model.lin_phi(in_features_phi[ids].float())
            regul_term = func_phi.norm(2) * self._coeffs['phi_zero']
            total_regul = total_regul + regul_term
            self._add('phi_zero', regul_term)

        # --- Edge diff/norm regularization ---
        if (self._coeffs['edge_diff'] > 0) | (self._coeffs['edge_norm'] > 0):
            in_features_edge, in_features_edge_next = get_in_features_lin_edge(x, model, mc, xnorm, n_neurons, device)

            if self._coeffs['edge_diff'] > 0:
                if mc.lin_edge_positive:
                    msg0 = model.lin_edge(in_features_edge[ids].clone().detach()) ** 2
                    msg1 = model.lin_edge(in_features_edge_next[ids].clone().detach()) ** 2
                else:
                    msg0 = model.lin_edge(in_features_edge[ids].clone().detach())
                    msg1 = model.lin_edge(in_features_edge_next[ids].clone().detach())
                regul_term = torch.relu(msg0 - msg1).norm(2) * self._coeffs['edge_diff']
                total_regul = total_regul + regul_term
                self._add('edge_diff', regul_term)

            if self._coeffs['edge_norm'] > 0:
                in_features_edge_norm = in_features_edge.clone()
                in_features_edge_norm[:, 0] = 2 * xnorm
                if mc.lin_edge_positive:
                    msg_norm = model.lin_edge(in_features_edge_norm[ids].clone().detach()) ** 2
                else:
                    msg_norm = model.lin_edge(in_features_edge_norm[ids].clone().detach())
                # Different normalization target for signal vs flyvis
                if self.trainer_type == 'signal':
                    regul_term = (msg_norm - 1).norm(2) * self._coeffs['edge_norm']
                else:  # flyvis
                    regul_term = (msg_norm - 2 * xnorm).norm(2) * self._coeffs['edge_norm']
                total_regul = total_regul + regul_term
                self._add('edge_norm', regul_term)

        # --- W_sign (Dale's Law) regularization ---
        if self._coeffs['W_sign'] > 0 and self.epoch > 0:
            W_sign_temp = getattr(tc, 'W_sign_temperature', 10.0)

            if self.trainer_type == 'signal' and index_weight is not None:
                # Signal version: uses index_weight
                if self.iter_count % 4 == 0:
                    W_sign = torch.tanh(5 * model_W) # noqa: F821
                    loss_contribs = []
                    for i in range(n_neurons):
                        indices = index_weight[int(i)]
                        if indices.numel() > 0:
                            values = W_sign[indices, i]
                            std = torch.std(values, unbiased=False)
                            loss_contribs.append(std)
                    if loss_contribs:
                        regul_term = torch.stack(loss_contribs).norm(2) * self._coeffs['W_sign']
                        total_regul = total_regul + regul_term
                        self._add('W_sign', regul_term)
            else:
                # Flyvis version: uses scatter_add
                weights = model_W.squeeze() if model_W is not None else model.W.squeeze() # noqa: F821
                source_neurons = edges[0]

                n_pos = torch.zeros(n_neurons, device=device)
                n_neg = torch.zeros(n_neurons, device=device)
                n_total = torch.zeros(n_neurons, device=device)

                pos_mask = torch.sigmoid(W_sign_temp * weights)
                neg_mask = torch.sigmoid(-W_sign_temp * weights)

                n_pos.scatter_add_(0, source_neurons, pos_mask)
                n_neg.scatter_add_(0, source_neurons, neg_mask)
                n_total.scatter_add_(0, source_neurons, torch.ones_like(weights))

                violation = torch.where(n_total > 0,
                                        (n_pos / n_total) * (n_neg / n_total),
                                        torch.zeros_like(n_total))
                regul_term = violation.sum() * self._coeffs['W_sign']
                total_regul = total_regul + regul_term
                self._add('W_sign', regul_term)

        # Note: Update function regularizations (update_msg_diff, update_u_diff, update_msg_sign)
        # are handled by compute_update_regul() which should be called after the model forward pass.
        # Call finalize_iteration() after all regularizations are computed to record to history.

        return total_regul

    def _record_to_history(self):
        """Append current iteration values to history."""
        n = self.n_neurons
        self._history['regul_total'].append(self._iter_total / n)
        for comp in self.COMPONENTS:
            self._history[comp].append(self._iter_tracker.get(comp, 0) / n)

    def compute_update_regul(self, model, in_features, ids_batch, device,
                              x=None, xnorm=None, ids=None):
        """
        Compute update function regularizations (update_diff, update_msg_diff, update_u_diff, update_msg_sign).

        This method should be called after the model forward pass when in_features is available.

        Args:
            model: The neural network model
            in_features: Features from model forward pass
            ids_batch: Batch indices
            device: Torch device
            x: Input tensor (required for update_diff with 'generic' update_type)
            xnorm: Normalization value (required for update_diff)
            ids: Sample indices (required for update_diff)

        Returns:
            Total update regularization loss tensor
        """
        mc = self.model_config
        embedding_dim = mc.embedding_dim
        n_neurons = self.n_neurons
        total_regul = torch.tensor(0.0, device=device)

        # update_diff: for 'generic' update_type only
        if (self._coeffs['update_diff'] > 0) and (model.update_type == 'generic') and (x is not None):
            in_features_edge, in_features_edge_next = get_in_features_lin_edge(
                x, model, mc, xnorm, n_neurons, device)
            if mc.lin_edge_positive:
                msg0 = model.lin_edge(in_features_edge[ids].clone().detach()) ** 2
                msg1 = model.lin_edge(in_features_edge_next[ids].clone().detach()) ** 2
            else:
                msg0 = model.lin_edge(in_features_edge[ids].clone().detach())
                msg1 = model.lin_edge(in_features_edge_next[ids].clone().detach())
            in_feature_update = torch.cat((torch.zeros((n_neurons, 1), device=device),
                                           model.a[:n_neurons], msg0,
                                           torch.ones((n_neurons, 1), device=device)), dim=1)
            in_feature_update = in_feature_update[ids]
            in_feature_update_next = torch.cat((torch.zeros((n_neurons, 1), device=device),
                                                model.a[:n_neurons], msg1,
                                                torch.ones((n_neurons, 1), device=device)), dim=1)
            in_feature_update_next = in_feature_update_next[ids]
            regul_term = torch.relu(model.lin_phi(in_feature_update) - model.lin_phi(in_feature_update_next)).norm(2) * self._coeffs['update_diff']
            total_regul = total_regul + regul_term
            self._add('update_diff', regul_term)

        if in_features is None:
            return total_regul

        if self._coeffs['update_msg_diff'] > 0:
            pred_msg = model.lin_phi(in_features.clone().detach())
            in_features_msg_next = in_features.clone().detach()
            in_features_msg_next[:, embedding_dim + 1] = in_features_msg_next[:, embedding_dim + 1] * 1.05
            pred_msg_next = model.lin_phi(in_features_msg_next)
            regul_term = torch.relu(pred_msg[ids_batch] - pred_msg_next[ids_batch]).norm(2) * self._coeffs['update_msg_diff']
            total_regul = total_regul + regul_term
            self._add('update_msg_diff', regul_term)

        if self._coeffs['update_u_diff'] > 0:
            pred_u = model.lin_phi(in_features.clone().detach())
            in_features_u_next = in_features.clone().detach()
            in_features_u_next[:, 0] = in_features_u_next[:, 0] * 1.05
            pred_u_next = model.lin_phi(in_features_u_next)
            regul_term = torch.relu(pred_u_next[ids_batch] - pred_u[ids_batch]).norm(2) * self._coeffs['update_u_diff']
            total_regul = total_regul + regul_term
            self._add('update_u_diff', regul_term)

        if self._coeffs['update_msg_sign'] > 0:
            in_features_modified = in_features.clone().detach()
            in_features_modified[:, 0] = 0
            pred_msg = model.lin_phi(in_features_modified)
            msg_col = in_features[:, embedding_dim + 1].clone().detach()
            regul_term = (torch.tanh(pred_msg / 0.1) - torch.tanh(msg_col.unsqueeze(-1) / 0.1)).norm(2) * self._coeffs['update_msg_sign']
            total_regul = total_regul + regul_term
            self._add('update_msg_sign', regul_term)

        return total_regul

    def finalize_iteration(self):
        """
        Finalize the current iteration by recording to history if appropriate.

        This should be called after all regularization computations (compute + compute_update_regul).
        """
        if self.should_record():
            self._record_to_history()

    def get_iteration_total(self) -> float:
        """Get total regularization for current iteration."""
        return self._iter_total

    def get_history(self) -> dict:
        """Get history dictionary for plotting."""
        return self._history
