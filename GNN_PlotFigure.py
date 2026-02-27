import os
import glob
import time
import logging
import warnings

import umap
import torch
import numpy as np
import seaborn as sns
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD

from flyvis_gnn.figure_style import default_style as fig_style
from flyvis_gnn.zarr_io import load_simulation_data, load_raw_array
from flyvis_gnn.fitting_models import linear_model
from flyvis_gnn.sparsify import clustering_gmm
from flyvis_gnn.models.flyvis_gnn import FlyVisGNN
from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.plot import (
    get_model_W,
    _vectorized_linspace,
    _batched_mlp_eval,
    _vectorized_linear_fit,
    _plot_curves_fast,
    _build_g_phi_features,
    _build_f_theta_features,
)
from flyvis_gnn.utils import (
    to_numpy,
    CustomColorMap,
    sort_key,
    create_log_dir,
    add_pre_folder,
    graphs_data_path,
    log_path,
    config_path,
)

# Optional imports
try:
    from flyvis_gnn.models.Ising_analysis import analyze_ising_model
except ImportError:
    analyze_ising_model = None

# Suppress matplotlib/PDF warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*Missing.*')

# Suppress fontTools logging (PDF font subsetting messages)
logging.getLogger('fontTools').setLevel(logging.ERROR)
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)

# Configure matplotlib for Helvetica-style fonts (no LaTeX)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'text.usetex': False,
    'mathtext.fontset': 'dejavusans',  # sans-serif math text
})


def get_training_files(log_dir, n_runs):
    files = glob.glob(f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_*.pt")
    if len(files) == 0:
        return [], np.array([])
    files.sort(key=sort_key)

    # Find the first file with positive sort_key
    file_id = 0
    while file_id < len(files) and sort_key(files[file_id]) <= 0:
        file_id += 1

    # If all files have non-positive sort_key, use all files
    if file_id >= len(files):
        file_id = 0

    files = files[file_id:]

    # Filter out files without the expected X_Y.pt suffix (e.g., "graphs_0.pt" has no Y)
    files = [f for f in files if f.split('_')[-2].isdigit()]

    if len(files) == 0:
        return [], np.array([])

    # Filter based on the Y value (number after "graphs")
    files_with_0 = [file for file in files if int(file.split('_')[-2]) == 0]
    files_without_0 = [file for file in files if int(file.split('_')[-2]) != 0]

    indices_with_0 = np.arange(0, len(files_with_0) - 1, dtype=int)
    indices_without_0 = np.linspace(0, len(files_without_0) - 1, 50, dtype=int)

    # Select the files using the generated indices
    selected_files_with_0 = [files_with_0[i] for i in indices_with_0]
    if len(files_without_0) > 0:
        selected_files_without_0 = [files_without_0[i] for i in indices_without_0]
        selected_files = selected_files_with_0 + selected_files_without_0
    else:
        selected_files = selected_files_with_0

    return selected_files, np.arange(0, len(selected_files), 1)


def plot_synaptic_flyvis(config, epoch_list, log_dir, logger, cc, style, extended, device, log_file=None):
    sim = config.simulation
    model_config = config.graph_model
    tc = config.training
    config_indices = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'evolution'


    colors_65 = sns.color_palette("Set3", 12) * 6  # pastel, repeat until 65
    colors_65 = colors_65[:65]

    config.simulation.max_radius if hasattr(config.simulation, 'max_radius') else 2.5

    results_log = os.path.join(log_dir, 'results.log')
    if os.path.exists(results_log):
        os.remove(results_log)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create file handler only, no console output
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear any existing handlers

    file_handler = logging.FileHandler(results_log, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(file_handler)

    # Prevent propagation to root logger (which might have console handlers)
    logger.propagate = False

    print(f'experiment description: {config.description}')
    logger.info(f'experiment description: {config.description}')

    # Load neuron group mapping for flyvis

    cmap = CustomColorMap(config=config)

    if 'black' in style:
        plt.style.use('dark_background')
        mc = 'w'
    else:
        plt.style.use('default')
        mc = 'k'

    time.sleep(0.5)
    print('load simulation data...')
    x_path = graphs_data_path(config.dataset, 'x_list_0')
    if not os.path.exists(x_path):
        x_path = graphs_data_path(config.dataset, 'x_list_train')
    x_ts = load_simulation_data(x_path,
                                fields=['index', 'voltage', 'stimulus', 'neuron_type', 'group_type'])
    y_path = graphs_data_path(config.dataset, 'y_list_0')
    if not os.path.exists(y_path):
        y_path = graphs_data_path(config.dataset, 'y_list_train')
    y_data = load_raw_array(y_path)

    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    if os.path.exists(os.path.join(log_dir, 'xnorm.pt')):
        xnorm = torch.load(os.path.join(log_dir, 'xnorm.pt'))
    else:
        xnorm = torch.tensor([5], device=device)

    print(f'xnorm: {to_numpy(xnorm):0.3f}, ynorm: {to_numpy(ynorm):0.3f}')
    logger.info(f'xnorm: {to_numpy(xnorm):0.3f}, ynorm: {to_numpy(ynorm):0.3f}')

    type_list = x_ts.neuron_type.to(device)
    n_types = len(torch.unique(type_list))
    region_list = x_ts.group_type.to(device)
    n_region_types = len(torch.unique(region_list))
    n_neurons = x_ts.n_neurons

    gt_weights = torch.load(graphs_data_path(config.dataset, 'weights.pt'), map_location=device)
    gt_taus = torch.load(graphs_data_path(config.dataset, 'taus.pt'), map_location=device)
    gt_V_Rest = torch.load(graphs_data_path(config.dataset, 'V_i_rest.pt'), map_location=device)
    edges = torch.load(graphs_data_path(config.dataset, 'edge_index.pt'), map_location=device)
    true_weights = torch.zeros((n_neurons, n_neurons), dtype=torch.float32, device=edges.device)
    true_weights[edges[1], edges[0]] = gt_weights

    # Neuron type index to name mapping
    index_to_name = {
        0: 'Am', 1: 'C2', 2: 'C3', 3: 'CT1(Lo1)', 4: 'CT1(M10)', 5: 'L1', 6: 'L2', 7: 'L3', 8: 'L4', 9: 'L5',
        10: 'Lawf1', 11: 'Lawf2', 12: 'Mi1', 13: 'Mi10', 14: 'Mi11', 15: 'Mi12', 16: 'Mi13', 17: 'Mi14',
        18: 'Mi15', 19: 'Mi2', 20: 'Mi3', 21: 'Mi4', 22: 'Mi9', 23: 'R1', 24: 'R2', 25: 'R3', 26: 'R4',
        27: 'R5', 28: 'R6', 29: 'R7', 30: 'R8', 31: 'T1', 32: 'T2', 33: 'T2a', 34: 'T3', 35: 'T4a',
        36: 'T4b', 37: 'T4c', 38: 'T4d', 39: 'T5a', 40: 'T5b', 41: 'T5c', 42: 'T5d', 43: 'Tm1',
        44: 'Tm16', 45: 'Tm2', 46: 'Tm20', 47: 'Tm28', 48: 'Tm3', 49: 'Tm30', 50: 'Tm4', 51: 'Tm5Y',
        52: 'Tm5a', 53: 'Tm5b', 54: 'Tm5c', 55: 'Tm9', 56: 'TmY10', 57: 'TmY13', 58: 'TmY14',
        59: 'TmY15', 60: 'TmY18', 61: 'TmY3', 62: 'TmY4', 63: 'TmY5a', 64: 'TmY9'
    }

    activity = x_ts.voltage.to(device).t()  # (N, T)
    mu_activity = torch.mean(activity, dim=1)
    sigma_activity = torch.std(activity, dim=1)

    print(f'neurons: {n_neurons}  edges: {edges.shape[1]}  neuron types: {n_types}  region types: {n_region_types}')
    logger.info(f'neurons: {n_neurons}  edges: {edges.shape[1]}  neuron types: {n_types}  region types: {n_region_types}')
    os.makedirs(f'{log_dir}/results/', exist_ok=True)

    sorted_neuron_type_names = [index_to_name.get(i, f'Type{i}') for i in range(sim.n_neuron_types)]

    target_type_name_list = ['R1', 'R7', 'C2', 'Mi11', 'Tm1', 'Tm4', 'Tm30']
    activity_results = plot_neuron_activity_analysis(activity, target_type_name_list, type_list, index_to_name, n_neurons, sim.n_frames, sim.delta_t, f'{log_dir}/results/')
    plot_ground_truth_distributions(to_numpy(edges), to_numpy(gt_weights), to_numpy(gt_taus), to_numpy(gt_V_Rest), to_numpy(type_list), n_types, sorted_neuron_type_names, f'{log_dir}/results/')

    if ('Ising' in extended) | ('ising' in extended):
        analyze_ising_model(x_ts, sim.delta_t, log_dir, logger, to_numpy(edges))

    # Activity plots
    config_indices = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'evolution'
    neuron_types = to_numpy(type_list).astype(int).squeeze()

    # Get activity traces for all frames — voltage is (T, N), transpose to (N, T)
    activity_true = to_numpy(x_ts.voltage).T     # (n_neurons, sim.n_frames)
    visual_input_true = to_numpy(x_ts.stimulus).T  # (n_neurons, sim.n_frames)

    start_frame = 0

    # Create two figures: all types and selected types
    for fig_name, selected_types in [
        ("selected", [5, 15, 43, 39, 35, 31, 23, 19, 12, 55]),  # L1, Mi12, Tm1, T5a, T4a, T1, R1, Mi2, Mi1, Tm9
        ("all", np.arange(0, sim.n_neuron_types))
    ]:
        neuron_indices = []
        for stype in selected_types:
            indices = np.where(neuron_types == stype)[0]
            if len(indices) > 0:
                neuron_indices.append(indices[0])

        if len(neuron_indices) == 0:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        true_slice = activity_true[neuron_indices, start_frame:sim.n_frames]
        visual_input_slice = visual_input_true[neuron_indices, start_frame:sim.n_frames]
        step_v = 2.5
        lw = 1

        # Adjust fontsize based on number of neurons
        name_fontsize = 10 if len(selected_types) > 50 else 18

        for i in range(len(neuron_indices)):
            baseline = np.mean(true_slice[i])
            ax.plot(true_slice[i] - baseline + i * step_v, linewidth=lw, c='green', alpha=0.9,
                    label='activity' if i == 0 else None)
            # Plot visual input only for neuron_id = 0
            if (neuron_indices[i] == 0) and visual_input_slice[i].mean() > 0:
                ax.plot(visual_input_slice[i] - baseline + i * step_v, linewidth=1, c='yellow', alpha=0.9,
                        linestyle='--', label='visual input')

        for i in range(len(neuron_indices)):
            type_idx = selected_types[i] if isinstance(selected_types, list) else selected_types[i]
            ax.text(-50, i * step_v, f'{index_to_name[type_idx]}', fontsize=name_fontsize, va='bottom', ha='right', color=mc)

        ax.set_ylim([-step_v, len(neuron_indices) * (step_v + 0.25 + 0.15 * (len(neuron_indices)//50))])
        ax.set_yticks([])
        ax.set_xticks([0, 1000, 2000])
        ax.set_xticklabels([0, 1000, 2000], fontsize=16)
        ax.set_xlabel('frame', fontsize=20)
        ax.set_xlim([0, 2000])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.legend(loc='upper right', fontsize=14)

        plt.tight_layout()
        if fig_name == "all":
            plt.savefig(f'{log_dir}/results/activity_{config_indices}.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{log_dir}/results/activity_{config_indices}_selected.png', dpi=300, bbox_inches='tight')
        plt.close()

    if epoch_list[0] != 'all':
        config_indices = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'evolution'
        files, file_id_list = get_training_files(log_dir, tc.n_runs)

        for epoch in epoch_list:

            net = f'{log_dir}/models/best_model_with_{tc.n_runs - 1}_graphs_{epoch}.pt'
            model = FlyVisGNN(aggr_type=model_config.aggr_type, config=config, device=device)
            state_dict = torch.load(net, map_location=device)
            state_dict['model_state_dict'] = {k.replace('lin_edge.', 'g_phi.').replace('lin_phi.', 'f_theta.'): v for k, v in state_dict['model_state_dict'].items()}
            model.load_state_dict(state_dict['model_state_dict'])
            model.edges = edges

            logger.info(f'net: {net}')

            # print learnable parameters table
            mlp0_params = sum(p.numel() for p in model.f_theta.parameters())
            mlp1_params = sum(p.numel() for p in model.g_phi.parameters())
            a_params = model.a.numel()
            w_params = get_model_W(model).numel()
            print('learnable parameters:')
            print(f'  MLP0 (f_theta): {mlp0_params:,}')
            print(f'  MLP1 (g_phi): {mlp1_params:,}')
            print(f'  a (embeddings): {a_params:,}')
            print(f'  W (connectivity): {w_params:,}')
            total_params = mlp0_params + mlp1_params + a_params + w_params
            if hasattr(model, 'NNR_f') and model.NNR_f is not None:
                nnr_f_params = sum(p.numel() for p in model.NNR_f.parameters())
                print(f'  INR (NNR_f): {nnr_f_params:,}')
                total_params += nnr_f_params
            print(f'  total: {total_params:,}')

            # Plot 1: Loss curve
            if os.path.exists(os.path.join(log_dir, 'loss.pt')):
                fig = plt.figure(figsize=(8, 6))
                ax = plt.gca()
                for spine in ax.spines.values():
                    spine.set_alpha(0.75)
                list_loss = torch.load(os.path.join(log_dir, 'loss.pt'))
                plt.plot(list_loss, color=mc, linewidth=2)
                plt.xlim([0, len(list_loss)])
                plt.ylabel('Loss')
                plt.xlabel('Epochs')
                plt.title('Training Loss')
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/loss.png', dpi=300)
                plt.close()

            # Plot 2: Embedding using model.a
            fig = plt.figure(figsize=(10, 9))
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_alpha(0.75)
            for n in range(n_types):
                pos = torch.argwhere(type_list == n)
                plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=24, color=colors_65[n], alpha=0.8,
                            edgecolors='none')
            plt.xlabel(r'$a_{i0}$', fontsize=48)
            plt.ylabel(r'$a_{i1}$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/embedding_{config_indices}.png', dpi=300)
            plt.close()

            # Plot 3: Edge function visualization (vectorized)
            fig = plt.figure(figsize=(10, 9))
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_alpha(0.75)
            n_pts = 1000
            rr_1d = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], n_pts, device=device)
            rr_all = rr_1d.unsqueeze(0).expand(n_neurons, -1)
            post_fn = (lambda x: x ** 2) if model_config.g_phi_positive else None
            build_fn = lambda rr_f, emb_f: _build_g_phi_features(rr_f, emb_f, model_config.signal_model_name)
            func_all = _batched_mlp_eval(model.g_phi, model.a[:n_neurons], rr_all,
                                         build_fn, device, post_fn=post_fn)
            type_np = to_numpy(type_list).astype(int).ravel()
            _plot_curves_fast(ax, to_numpy(rr_1d), to_numpy(func_all), type_np, cmap, linewidth=1, alpha=0.1)
            plt.xlabel('$v_j$', fontsize=48)
            plt.ylabel(r'$\mathrm{MLP_1}(a_j, v_j)$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.xlim([-1,2.5])
            plt.ylim([-config.plotting.xlim[1]/10, 2.5])
            plt.tight_layout()
            plt.savefig(f"{log_dir}/results/MLP1_{config_indices}.png", dpi=300)
            plt.close()


            # g_phi domain range: evaluate + slope extraction (vectorized)
            mu = to_numpy(mu_activity).astype(np.float32)
            sigma = to_numpy(sigma_activity).astype(np.float32)

            # # Percentile-based range (clamped to positive):
            # voltage = x_ts.voltage.to(device)  # (T, N)
            # p_low = to_numpy(torch.quantile(voltage, 0.05, dim=0)).astype(np.float32)
            # p_high = to_numpy(torch.quantile(voltage, 0.95, dim=0)).astype(np.float32)
            # starts_edge = np.maximum(p_low, 0.0)
            # ends_edge = np.maximum(p_high, 0.0)
            # valid_edge = ends_edge > starts_edge + 1e-6
            # starts_edge[~valid_edge] = 0.0
            # ends_edge[~valid_edge] = 1.0

            # mu ± 2σ range
            valid_edge = (mu + sigma) > 0
            starts_edge = np.maximum(mu - 2 * sigma, 0.0)
            ends_edge = mu + 2 * sigma
            starts_edge[~valid_edge] = 0.0
            ends_edge[~valid_edge] = 1.0
            rr_domain_edge = _vectorized_linspace(starts_edge, ends_edge, n_pts, device)
            func_domain_edge = _batched_mlp_eval(model.g_phi, model.a[:n_neurons], rr_domain_edge,
                                                 build_fn, device, post_fn=post_fn)
            slopes_edge, _ = _vectorized_linear_fit(rr_domain_edge, func_domain_edge)
            slopes_edge[~valid_edge] = 1.0
            slopes_g_phi_list = slopes_edge  # (N,) numpy array

            fig = plt.figure(figsize=(10, 9))
            ax = plt.gca()
            rr_np = to_numpy(rr_domain_edge)
            func_np = to_numpy(func_domain_edge)
            # Only plot valid neurons
            _plot_curves_fast(ax, rr_np[valid_edge], func_np[valid_edge],
                              type_np[valid_edge], cmap, linewidth=1, alpha=0.1)
            plt.xlabel('$v_j$', fontsize=48)
            plt.ylabel(r'$\mathrm{MLP_1}(a_j, v_j)$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.xlim([-1,5])
            plt.ylim([-config.plotting.xlim[1]/10, config.plotting.xlim[1]*2])
            plt.tight_layout()
            plt.savefig(f"{log_dir}/results/MLP1_{config_indices}_domain.png", dpi=300)
            plt.close()


            fig = plt.figure(figsize=(10, 9))
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_alpha(0.75)
            slopes_g_phi_array = np.array(slopes_g_phi_list)
            plt.scatter(np.arange(n_neurons), slopes_g_phi_array,
                        c=cmap.color(to_numpy(type_list).astype(int)), s=2, alpha=0.5)
            plt.xlabel('neuron index', fontsize=48)
            plt.ylabel(r'$r_j$', fontsize=48)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.tight_layout()
            plt.savefig(f"{log_dir}/results/MLP1_slope_{config_indices}.png", dpi=300)
            plt.close()

            # Plot 5: Phi function visualization (vectorized)
            fig = plt.figure(figsize=(10, 9))
            ax = plt.gca()
            rr_phi_1d = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], n_pts, device=device)
            rr_phi_all = rr_phi_1d.unsqueeze(0).expand(n_neurons, -1)
            func_phi_all = _batched_mlp_eval(model.f_theta, model.a[:n_neurons], rr_phi_all,
                                             lambda rr_f, emb_f: _build_f_theta_features(rr_f, emb_f), device)
            _plot_curves_fast(ax, to_numpy(rr_phi_1d), to_numpy(func_phi_all), type_np, cmap, linewidth=1, alpha=0.1)
            plt.xlim([-2.5,2.5])
            plt.ylim([-100,100])
            plt.xlabel('$v_i$', fontsize=48)
            plt.ylabel(r'$\mathrm{MLP_0}(a_i, v_i)$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.tight_layout()
            plt.savefig(f"{log_dir}/results/MLP0_{config_indices}.png", dpi=300)
            plt.close()

            # f_theta domain range: evaluate + slope extraction (vectorized)
            starts_phi = mu - 2 * sigma
            ends_phi = mu + 2 * sigma
            rr_domain_phi = _vectorized_linspace(starts_phi, ends_phi, n_pts, device)
            func_domain_phi = _batched_mlp_eval(model.f_theta, model.a[:n_neurons], rr_domain_phi,
                                                lambda rr_f, emb_f: _build_f_theta_features(rr_f, emb_f), device)
            slopes_phi, offsets_phi = _vectorized_linear_fit(rr_domain_phi, func_domain_phi)
            slopes_f_theta_list = slopes_phi  # (N,) numpy array
            offsets_list = offsets_phi

            fig = plt.figure(figsize=(10, 9))
            ax = plt.gca()
            _plot_curves_fast(ax, to_numpy(rr_domain_phi), to_numpy(func_domain_phi),
                              type_np, cmap, linewidth=1, alpha=0.1)
            plt.xlim(config.plotting.xlim)
            plt.ylim(config.plotting.ylim)
            plt.xlabel('$v_i$', fontsize=48)
            plt.ylabel(r'$\mathrm{MLP_0}(a_i, v_i)$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.tight_layout()
            plt.savefig(f"{log_dir}/results/MLP0_{config_indices}_domain.png", dpi=300)
            plt.close()

            slopes_f_theta_array = np.array(slopes_f_theta_list)
            offsets_array = np.array(offsets_list)
            gt_taus = to_numpy(gt_taus[:n_neurons])
            learned_tau = np.where(slopes_f_theta_array != 0, 1.0 / -slopes_f_theta_array, 1)
            learned_tau = learned_tau[:n_neurons]
            learned_tau = np.clip(learned_tau, 0, 1)

            fig = plt.figure(figsize=(10, 9))
            plt.scatter(gt_taus, learned_tau, c=mc, s=1, alpha=0.3)
            lin_fit_tau, _ = curve_fit(linear_model, gt_taus, learned_tau)
            residuals = learned_tau - linear_model(gt_taus, *lin_fit_tau)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((learned_tau - np.mean(learned_tau)) ** 2)
            r_squared_tau = 1 - (ss_res / ss_tot)
            plt.text(0.05, 0.95, f'R²: {r_squared_tau:.2f}\nslope: {lin_fit_tau[0]:.2f}\nN: {sim.n_edges}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
            plt.xlabel(r'true $\tau$', fontsize=48)
            plt.ylabel(r'learned $\tau$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.xlim([0, 0.35])
            plt.ylim([0, 0.35])
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/tau_comparison_{config_indices}.png', dpi=300)
            plt.close()


            # V_rest comparison (reconstructed vs ground truth)
            learned_V_rest = np.where(slopes_f_theta_array != 0, -offsets_array / slopes_f_theta_array, 1)
            learned_V_rest = learned_V_rest[:n_neurons]
            gt_V_rest = to_numpy(gt_V_Rest[:n_neurons])
            fig = plt.figure(figsize=(10, 9))
            plt.scatter(gt_V_rest, learned_V_rest, c=mc, s=1, alpha=0.3)
            lin_fit_V_rest, _ = curve_fit(linear_model, gt_V_rest, learned_V_rest)
            residuals = learned_V_rest - linear_model(gt_V_rest, *lin_fit_V_rest)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((learned_V_rest - np.mean(learned_V_rest)) ** 2)
            r_squared_V_rest = 1 - (ss_res / ss_tot)
            plt.text(0.05, 0.95, f'R²: {r_squared_V_rest:.2f}\nslope: {lin_fit_V_rest[0]:.2f}\nN: {sim.n_edges}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
            plt.xlabel(r'true $V_{rest}$', fontsize=48)
            plt.ylabel(r'learned $V_{rest}$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.xlim([0, 0.8])
            plt.ylim([0, 0.8])
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/V_rest_comparison_{config_indices}.png', dpi=300)
            plt.close()

            fig = plt.figure(figsize=(10, 9))
            ax = plt.subplot(2, 1, 1)
            plt.scatter(np.arange(n_neurons), learned_tau,
                        c=cmap.color(to_numpy(type_list).astype(int)), s=2, alpha=0.5)
            plt.ylabel(r'$\tau_i$', fontsize=48)
            plt.xticks([])   # no xticks for top plot
            plt.yticks(fontsize=24)
            ax = plt.subplot(2, 1, 2)
            plt.scatter(np.arange(n_neurons), learned_V_rest,
                        c=cmap.color(to_numpy(type_list).astype(int)), s=2, alpha=0.5)
            plt.xlabel('neuron index', fontsize=48)
            plt.ylabel(r'$V^{\mathrm{rest}}_i$', fontsize=48)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.savefig(f"{log_dir}/results/MLP0_{config_indices}_params.png", dpi=300)
            plt.close()


            # Plot 4: Weight comparison using model.W and gt_weights
            # Check Dale's Law for learned weights
            # dale_results = check_dales_law(
            #     edges=edges,
            #     weights=model.W,
            #     type_list=type_list,
            #     n_neurons=n_neurons,
            #     verbose=False,
            #     logger=None
            # )

            fig = plt.figure(figsize=(10, 9))
            learned_weights = to_numpy(get_model_W(model).squeeze())
            true_weights = to_numpy(gt_weights)
            plt.scatter(true_weights, learned_weights, c=mc, s=0.1, alpha=0.1)
            lin_fit, lin_fitv = curve_fit(linear_model, true_weights, learned_weights)
            residuals = learned_weights - linear_model(true_weights, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((learned_weights - np.mean(learned_weights)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.text(0.05, 0.95, f'R²: {r_squared:.3f}\nslope: {lin_fit[0]:.2f}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=24)

            # Add Dale's Law statistics
            # dale_text = (f"excitatory neurons (all W>0): {dale_results['n_excitatory']} "
            #              f"({100*dale_results['n_excitatory']/n_neurons:.1f}%)\n"
            #              f"inhibitory neurons (all W<0): {dale_results['n_inhibitory']} "
            #              f"({100*dale_results['n_inhibitory']/n_neurons:.1f}%)\n"
            #              f"mixed/zero neurons (violates Dale's Law): {dale_results['n_mixed']} "
            #              f"({100*dale_results['n_mixed']/n_neurons:.1f}%)")
            # plt.text(0.05, 0.05, dale_text, transform=plt.gca().transAxes,
            #          verticalalignment='bottom', fontsize=10)

            plt.xlabel(r'true $W_{ij}$', fontsize=48)
            plt.ylabel(r'learned $W_{ij}$', fontsize=48)
            plt.xticks(fontsize = 24)
            plt.yticks(fontsize = 24)
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/weights_comparison_raw.png', dpi=300)
            plt.close()
            print(f"first weights fit R²: {r_squared:.2f}  slope: {np.round(lin_fit[0], 4)}")
            logger.info(f"first weights fit R²: {r_squared:.2f}  slope: {np.round(lin_fit[0], 4)}")

            # k_list = [1]

            k_list = np.linspace(sim.n_frames // 10, sim.n_frames-100, 8, dtype=int).tolist()

            dataset_batch = []
            ids_batch = []
            mask_batch = []
            ids_index = 0
            mask_index = 0

            for batch in range(len(k_list)):

                k = k_list[batch]
                x = x_ts.frame(k).to(device).to_packed()
                ids = np.arange(n_neurons)

                if not (torch.isnan(x).any()):

                    mask = torch.arange(edges.shape[1])

                    y = torch.tensor(y_data[k], device=device) / ynorm

                    if not (torch.isnan(y).any()):

                        import torch_geometric.data as data
                        dataset = data.Data(x=x, edge_index=edges)
                        dataset_batch.append(dataset)

                        if len(dataset_batch) == 1:
                            data_id = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)
                            y_batch = y
                            ids_batch = ids
                            mask_batch = mask
                        else:
                            data_id = torch.cat(
                                (data_id, torch.zeros((n_neurons, 1), dtype=torch.int, device=device)), dim=0)
                            y_batch = torch.cat((y_batch, y), dim=0)
                            ids_batch = np.concatenate((ids_batch, ids + ids_index), axis=0)
                            mask_batch = torch.cat((mask_batch, mask + mask_index), dim=0)

                        ids_index += n_neurons
                        mask_index += edges.shape[1]

            with torch.no_grad():
                from torch_geometric.loader import DataLoader
                from flyvis_gnn.neuron_state import NeuronState
                batch_loader = DataLoader(dataset_batch, batch_size=len(k_list), shuffle=False)
                for batch in batch_loader:
                    batch_state = NeuronState.from_numpy(batch.x)
                    pred, in_features, msg = model(batch_state, batch.edge_index, data_id=data_id, mask=mask_batch, return_all=True)

            # Extract features and compute gradient of f_theta w.r.t. msg
            ed = model_config.embedding_dim
            v = in_features[:, 0:1].clone().detach()
            embedding = in_features[:, 1:1+ed].clone().detach()
            msg = in_features[:, 1+ed:2+ed].clone().detach()
            excitation = in_features[:, 2+ed:3+ed].clone().detach()

            # Re-enable gradients (may have been disabled by data_test)
            torch.set_grad_enabled(True)

            # Enable gradient tracking for msg
            msg.requires_grad_(True)
            # Concatenate input features for the final layer
            in_features_grad = torch.cat([v, embedding, msg, excitation], dim=1)
            # Run f_theta outside no_grad context to build computation graph
            out = model.f_theta(in_features_grad)

            grad_msg = torch.autograd.grad(
                outputs=out,
                inputs=msg,
                grad_outputs=torch.ones_like(out),
                retain_graph=True,
                create_graph=False
            )[0]


            plt.figure(figsize=(12, 6))

            n_batches = grad_msg.shape[0] // n_neurons
            grad_values = grad_msg.view(n_batches, n_neurons)
            grad_values = grad_values.median(dim=0).values
            grad_values = to_numpy(grad_values).squeeze()

            # grad_values = to_numpy(grad_msg[0:n_neurons]).squeeze()

            # Flatten to 1D
            neuron_indices = np.arange(n_neurons)
            # Create scatter plot colored by neuron type
            for n in range(n_types):
                type_mask = (to_numpy(type_list).squeeze() == n)  # Flatten to 1D
                if np.any(type_mask):
                    plt.scatter(neuron_indices[type_mask], grad_values[type_mask],
                                c=colors_65[n], s=1, alpha=0.8)

                    # Add text label for each neuron type
                    if np.sum(type_mask) > 0:
                        mean_x = np.mean(neuron_indices[type_mask])
                        mean_y = np.mean(grad_values[type_mask])
                        plt.text(mean_x, mean_y, index_to_name.get(n, f'T{n}'),
                                 fontsize=6, ha='center', va='center')
            plt.xlabel('neuron index')
            plt.ylabel('gradient')
            plt.tight_layout()
            # plt.savefig(f'{log_dir}/results/msg_gradients_{epoch}.png', dpi=300)
            plt.close()

            grad_msg_flat = grad_msg.squeeze()
            assert grad_msg_flat.shape[0] == n_neurons * len(k_list), "Gradient and neuron count mismatch"
            target_neuron_ids = edges[1, :] % (model.n_edges + model.n_extra_null_edges)
            grad_msg_per_edge = grad_msg_flat[target_neuron_ids]
            grad_msg_per_edge = grad_msg_per_edge.unsqueeze(1)  # [434112, 1]

            slopes_f_theta_array = torch.tensor(slopes_f_theta_array, dtype=torch.float32, device=device)
            slopes_f_theta_per_edge = slopes_f_theta_array[target_neuron_ids]

            slopes_g_phi_array = np.array(slopes_g_phi_list)
            slopes_g_phi_array = torch.tensor(slopes_g_phi_array, dtype=torch.float32, device=device)
            prior_neuron_ids = edges[0, :] % (model.n_edges + model.n_extra_null_edges)  # j
            slopes_g_phi_per_edge = slopes_g_phi_array[prior_neuron_ids]

            corrected_W_ = -get_model_W(model) / slopes_f_theta_per_edge[:, None] * grad_msg_per_edge
            corrected_W = -get_model_W(model) / slopes_f_theta_per_edge[:, None] * grad_msg_per_edge * slopes_g_phi_per_edge.unsqueeze(1)

            # sanitize: division by near-zero slopes can produce inf/nan
            corrected_W = torch.nan_to_num(corrected_W, nan=0.0, posinf=0.0, neginf=0.0)
            corrected_W_ = torch.nan_to_num(corrected_W_, nan=0.0, posinf=0.0, neginf=0.0)

            torch.save(corrected_W, f'{log_dir}/results/corrected_W.pt')

            learned_weights = to_numpy(corrected_W.squeeze())
            true_weights = to_numpy(gt_weights)

            # --- Outlier removal: drop weights beyond 3*MAD ---
            residuals = learned_weights - true_weights
            mask = np.abs(residuals) <= 5  # keep only inliers

            true_in = true_weights[mask]
            learned_in = learned_weights[mask]

            if extended:

                learned_in_ = to_numpy(corrected_W_.squeeze())
                learned_in_ = learned_in_[mask]

                fig = plt.figure(figsize=(10, 9))
                plt.scatter(true_in, learned_in_, c=mc, s=0.1, alpha=0.1)
                lin_fit, _ = curve_fit(linear_model, true_in, learned_in_)
                residuals_ = learned_in_ - linear_model(true_in, *lin_fit)
                ss_res = np.sum(residuals_ ** 2)
                ss_tot = np.sum((learned_in_ - np.mean(learned_in_)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                plt.text(0.05, 0.95,
                        f'R²: {r_squared:.3f}\nslope: {lin_fit[0]:.2f}',
                        transform=plt.gca().transAxes, verticalalignment='top', fontsize=24)
                plt.xlabel(r'true $W_{ij}$', fontsize=48)
                plt.ylabel(r'learned $W_{ij}r_j$', fontsize=48)
                plt.xticks(fontsize = 24)
                plt.yticks(fontsize = 24)
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/weights_comparison_rj.png', dpi=300)
                plt.close()


            fig = plt.figure(figsize=(10, 9))
            plt.scatter(true_in, learned_in, c=mc, s=0.5, alpha=0.06)
            lin_fit, _ = curve_fit(linear_model, true_in, learned_in)
            residuals_ = learned_in - linear_model(true_in, *lin_fit)
            ss_res = np.sum(residuals_ ** 2)
            ss_tot = np.sum((learned_in - np.mean(learned_in)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.text(0.05, 0.95,
                     f'R²: {r_squared:.2f}\nslope: {lin_fit[0]:.2f}\nN: {sim.n_edges}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)

            # Add Dale's Law statistics (reusing dale_results from earlier)
            # dale_text = (f"excitatory neurons (all W>0): {dale_results['n_excitatory']} "
            #              f"({100*dale_results['n_excitatory']/n_neurons:.1f}%)\n"
            #              f"inhibitory neurons (all W<0): {dale_results['n_inhibitory']} "
            #              f"({100*dale_results['n_inhibitory']/n_neurons:.1f}%)\n"
            #              f"mixed/zero neurons (violates Dale's Law): {dale_results['n_mixed']} "
            #              f"({100*dale_results['n_mixed']/n_neurons:.1f}%)")
            # plt.text(0.05, 0.05, dale_text, transform=plt.gca().transAxes,
            #          verticalalignment='bottom', fontsize=10)

            plt.xlabel(r'true $W_{ij}$', fontsize=48)
            plt.ylabel(r'learned $W_{ij}^*$', fontsize=48)
            plt.xticks(fontsize = 24)
            plt.yticks(fontsize = 24)
            plt.xlim([-1,2])
            plt.ylim([-1,2])
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/weights_comparison_corrected.png', dpi=300)
            plt.close()

            print(f"second weights fit R²: \033[92m{r_squared:.4f}\033[0m  slope: {np.round(lin_fit[0], 4)}")
            logger.info(f"second weights fit R²: {r_squared:.4f}  slope: {np.round(lin_fit[0], 4)}")
            print(f'median residuals: {np.median(residuals):.4f}')
            inlier_residuals = residuals[mask]
            print(f'inliers: {len(inlier_residuals)}  mean residual: {np.mean(inlier_residuals):.4f}  std: {np.std(inlier_residuals):.4f}  min,max: {np.min(inlier_residuals):.4f}, {np.max(inlier_residuals):.4f}')
            outlier_residuals = residuals[~mask]
            if len(outlier_residuals) > 0:
                print(
                    f'outliers: {len(outlier_residuals)}  mean residual: {np.mean(outlier_residuals):.4f}  std: {np.std(outlier_residuals):.4f}  min,max: {np.min(outlier_residuals):.4f}, {np.max(outlier_residuals):.4f}')
            else:
                print('outliers: 0  (no outliers detected)')
            print(f"tau reconstruction R²: \033[92m{r_squared_tau:.3f}\033[0m  slope: {lin_fit_tau[0]:.2f}")
            logger.info(f"tau reconstruction R²: {r_squared_tau:.3f}  slope: {lin_fit_tau[0]:.2f}")
            print(f"V_rest reconstruction R²: \033[92m{r_squared_V_rest:.3f}\033[0m  slope: {lin_fit_V_rest[0]:.2f}")
            logger.info(f"V_rest reconstruction R²: {r_squared_V_rest:.3f}  slope: {lin_fit_V_rest[0]:.2f}")

            # Write to analysis log file for Claude
            if log_file:
                print(f"  [DEBUG] plot_synaptic_flyvis: writing R2 metrics, log_file={type(log_file).__name__}, closed={log_file.closed}")
                log_file.write(f"connectivity_R2: {r_squared:.4f}\n")
                log_file.write(f"tau_R2: {r_squared_tau:.4f}\n")
                log_file.write(f"V_rest_R2: {r_squared_V_rest:.4f}\n")


            # Plot connectivity matrix comparison (skipped — dense NxN heatmaps too slow)
            # eigenvalue and singular value analysis using sparse matrices
            print('plot eigenvalue spectrum and eigenvector comparison ...')

            # build sparse matrices for true and learned weights
            edges_np = to_numpy(edges)
            true_sparse = scipy.sparse.csr_matrix(
                (true_weights.flatten(), (edges_np[1], edges_np[0])),
                shape=(n_neurons, n_neurons)
            )
            learned_sparse = scipy.sparse.csr_matrix(
                (to_numpy(corrected_W.squeeze().flatten()), (edges_np[1], edges_np[0])),
                shape=(n_neurons, n_neurons)
            )

            # compute SVD using TruncatedSVD (for large sparse matrices)
            # 100 components captures dominant structure; 1000 was very slow for N>10000
            n_components = min(100, n_neurons - 1)
            svd_true = TruncatedSVD(n_components=n_components, random_state=42)
            svd_learned = TruncatedSVD(n_components=n_components, random_state=42)

            svd_true.fit(true_sparse)
            svd_learned.fit(learned_sparse)

            sv_true = svd_true.singular_values_
            sv_learned = svd_learned.singular_values_

            # get right singular vectors (V^T rows)
            V_true = svd_true.components_
            V_learned = svd_learned.components_

            # compute alignment matrix
            alignment = np.abs(V_true @ V_learned.T)
            best_alignment = np.max(alignment, axis=1)

            # compute eigenvalues using sparse eigensolver for complex plane plot
            # 200 largest-magnitude eigenvalues captures spectral structure;
            # 500 was very slow for N>10000 (ARPACK scales poorly with k)
            n_eigs = min(200, n_neurons - 2)
            try:
                eig_true, _ = scipy.sparse.linalg.eigs(true_sparse.astype(np.float64), k=n_eigs, which='LM')
                eig_learned, _ = scipy.sparse.linalg.eigs(learned_sparse.astype(np.float64), k=n_eigs, which='LM')
            except Exception:
                # fallback: use smaller k if convergence issues
                n_eigs = min(50, n_neurons - 2)
                eig_true, _ = scipy.sparse.linalg.eigs(true_sparse.astype(np.float64), k=n_eigs, which='LM')
                eig_learned, _ = scipy.sparse.linalg.eigs(learned_sparse.astype(np.float64), k=n_eigs, which='LM')

            # create 2x3 figure
            fig, axes = plt.subplots(2, 3, figsize=(30, 20))

            # Row 1: Eigenvalues/Singular values
            # (0,0) eigenvalues in complex plane
            axes[0, 0].scatter(eig_true.real, eig_true.imag, s=100, c='b', alpha=0.7, label='true')
            axes[0, 0].scatter(eig_learned.real, eig_learned.imag, s=100, c='r', alpha=0.7, label='learned')
            axes[0, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            axes[0, 0].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
            axes[0, 0].set_xlabel('real', fontsize=32)
            axes[0, 0].set_ylabel('imag', fontsize=32)
            axes[0, 0].legend(fontsize=20)
            axes[0, 0].tick_params(labelsize=20)
            axes[0, 0].set_title('eigenvalues in complex plane', fontsize=28)

            # (0,1) singular value magnitude comparison (scatter)
            n_compare = min(len(sv_true), len(sv_learned))
            axes[0, 1].scatter(sv_true[:n_compare], sv_learned[:n_compare], s=100, c='white', edgecolors='gray', alpha=0.7)
            max_val = max(sv_true.max(), sv_learned.max())
            axes[0, 1].plot([0, max_val], [0, max_val], 'g--', linewidth=2)
            axes[0, 1].set_xlabel('true singular value', fontsize=32)
            axes[0, 1].set_ylabel('learned singular value', fontsize=32)
            axes[0, 1].tick_params(labelsize=20)
            axes[0, 1].set_title('singular value comparison', fontsize=28)

            # (0,2) singular value spectrum (log scale)
            axes[0, 2].plot(sv_true, 'b-', linewidth=2, label='true')
            axes[0, 2].plot(sv_learned, 'r-', linewidth=2, label='learned')
            axes[0, 2].set_xlabel('index', fontsize=32)
            axes[0, 2].set_ylabel('singular value', fontsize=32)
            axes[0, 2].set_yscale('log')
            axes[0, 2].legend(fontsize=20)
            axes[0, 2].tick_params(labelsize=20)
            axes[0, 2].set_title('singular value spectrum (log scale)', fontsize=28)

            # Row 2: Singular vectors
            # (1,0) right singular vector alignment matrix
            n_show = min(100, n_components)
            im = axes[1, 0].imshow(alignment[:n_show, :n_show], cmap='hot', vmin=0, vmax=1)
            axes[1, 0].set_xlabel('learned eigenvector index', fontsize=28)
            axes[1, 0].set_ylabel('true eigenvector index', fontsize=28)
            axes[1, 0].set_title('right eigenvector alignment', fontsize=28)
            axes[1, 0].tick_params(labelsize=16)
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

            # (1,1) left eigenvector alignment (placeholder - SVD doesn't give left eigenvectors directly)
            # For consistency with plot_signal, compute left singular vectors alignment
            U_true = svd_true.transform(true_sparse)[:, :n_show]
            U_learned = svd_learned.transform(learned_sparse)[:, :n_show]
            # Normalize columns
            U_true = U_true / (np.linalg.norm(U_true, axis=0, keepdims=True) + 1e-10)
            U_learned = U_learned / (np.linalg.norm(U_learned, axis=0, keepdims=True) + 1e-10)
            alignment_L = np.abs(U_true.T @ U_learned)
            best_alignment_L = np.max(alignment_L, axis=1)
            im_L = axes[1, 1].imshow(alignment_L, cmap='hot', vmin=0, vmax=1)
            axes[1, 1].set_xlabel('learned eigenvector index', fontsize=28)
            axes[1, 1].set_ylabel('true eigenvector index', fontsize=28)
            axes[1, 1].set_title('left eigenvector alignment', fontsize=28)
            axes[1, 1].tick_params(labelsize=16)
            plt.colorbar(im_L, ax=axes[1, 1], fraction=0.046)

            # (1,2) best alignment scores
            best_alignment_R = np.max(alignment[:n_show, :n_show], axis=1)
            axes[1, 2].scatter(range(len(best_alignment_R)), best_alignment_R, s=50, c='b', alpha=0.7, label=f'right (mean={np.mean(best_alignment_R):.2f})')
            axes[1, 2].scatter(range(len(best_alignment_L)), best_alignment_L, s=50, c='r', alpha=0.7, label=f'left (mean={np.mean(best_alignment_L):.2f})')
            axes[1, 2].axhline(y=1/np.sqrt(n_show), color='gray', linestyle='--', linewidth=2, label=f'random ({1/np.sqrt(n_show):.2f})')
            axes[1, 2].set_xlabel('eigenvector index (sorted by singular value)', fontsize=28)
            axes[1, 2].set_ylabel('best alignment score', fontsize=28)
            axes[1, 2].set_title('best alignment per eigenvector', fontsize=28)
            axes[1, 2].set_ylim([0, 1.05])
            axes[1, 2].legend(fontsize=20)
            axes[1, 2].tick_params(labelsize=16)

            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/eigen_comparison.png', dpi=87)
            plt.close()

            # print spectral analysis results (consistent with plot_signal)
            true_spectral_radius = np.max(np.abs(eig_true))
            learned_spectral_radius = np.max(np.abs(eig_learned))
            print(f'spectral radius - true: {true_spectral_radius:.3f}  learned: {learned_spectral_radius:.3f}')
            logger.info(f'spectral radius - true: {true_spectral_radius:.3f}  learned: {learned_spectral_radius:.3f}')
            print(f'eigenvector alignment - right: {np.mean(best_alignment_R):.3f}  left: {np.mean(best_alignment_L):.3f}')
            logger.info(f'eigenvector alignment - right: {np.mean(best_alignment_R):.3f}  left: {np.mean(best_alignment_L):.3f}')


            # plot analyze_neuron_type_reconstruction
            results_per_neuron = analyze_neuron_type_reconstruction(
                config=config,
                model=model,
                edges=to_numpy(edges),
                true_weights=true_weights,  #  ground truth weights
                gt_taus=gt_taus,  #  ground truth tau values
                gt_V_Rest=gt_V_rest,  #  ground truth V_rest values
                learned_weights=learned_weights,
                learned_tau = learned_tau,
                learned_V_rest=learned_V_rest, # Learned V_rest
                type_list=to_numpy(type_list),
                n_frames=sim.n_frames,
                dimension=sim.dimension,
                n_neuron_types=sim.n_neuron_types,
                device=device,
                log_dir=log_dir,
                dataset_name=config.dataset,
                logger=logger,
                index_to_name=index_to_name
            )

            print('alternative clustering methods...')


            # compute connectivity statistics (vectorized via bincount)
            print('computing connectivity statistics...')
            edges_np = to_numpy(edges)
            src, dst = edges_np[0], edges_np[1]

            def _connectivity_stats(w, src, dst, n):
                """Per-neuron mean/std of in-weights and out-weights."""
                # counts
                in_count = np.bincount(dst, minlength=n).astype(np.float64)
                out_count = np.bincount(src, minlength=n).astype(np.float64)
                # sums
                in_sum = np.bincount(dst, weights=w, minlength=n)
                out_sum = np.bincount(src, weights=w, minlength=n)
                # sum of squares
                in_sq = np.bincount(dst, weights=w ** 2, minlength=n)
                out_sq = np.bincount(src, weights=w ** 2, minlength=n)
                # mean (0 where no edges)
                safe_in = np.where(in_count > 0, in_count, 1)
                safe_out = np.where(out_count > 0, out_count, 1)
                in_mean = in_sum / safe_in
                out_mean = out_sum / safe_out
                # std = sqrt(E[x^2] - E[x]^2), clamped to avoid negative from fp noise
                in_std = np.sqrt(np.maximum(in_sq / safe_in - in_mean ** 2, 0))
                out_std = np.sqrt(np.maximum(out_sq / safe_out - out_mean ** 2, 0))
                # zero out neurons with no edges
                in_mean[in_count == 0] = 0
                out_mean[out_count == 0] = 0
                in_std[in_count == 0] = 0
                out_std[out_count == 0] = 0
                return in_mean, in_std, out_mean, out_std

            w_in_mean_true, w_in_std_true, w_out_mean_true, w_out_std_true = \
                _connectivity_stats(true_weights.flatten(), src, dst, n_neurons)
            w_in_mean_learned, w_in_std_learned, w_out_mean_learned, w_out_std_learned = \
                _connectivity_stats(learned_weights.flatten(), src, dst, n_neurons)

            # all 4 connectivity stats combined
            W_learned = np.column_stack([w_in_mean_learned, w_in_std_learned,
                                        w_out_mean_learned, w_out_std_learned])
            W_true = np.column_stack([w_in_mean_true, w_in_std_true,
                                    w_out_mean_true, w_out_std_true])

            # learned combinations
            learned_combos = {
                'a': to_numpy(model.a),
                'τ': learned_tau.reshape(-1, 1),
                'V': learned_V_rest.reshape(-1, 1),
                'W': W_learned,
                '(τ,V)': np.column_stack([learned_tau, learned_V_rest]),
                '(τ,V,W)': np.column_stack([learned_tau, learned_V_rest, W_learned]),
                '(a,τ,V,W)': np.column_stack([to_numpy(model.a), learned_tau, learned_V_rest, W_learned]),
            }

            # true combinations
            true_combos = {
                'τ': gt_taus.reshape(-1, 1),
                'V': gt_V_rest.reshape(-1, 1),
                'W': W_true,
                '(τ,V)': np.column_stack([gt_taus, gt_V_rest]),
                '(τ,V,W)': np.column_stack([gt_taus, gt_V_rest, W_true]),
            }

            # cluster learned
            print('clustering learned features...')
            learned_results = {}
            for name, feat_array in learned_combos.items():
                result = clustering_gmm(feat_array, type_list, n_components=75)
                learned_results[name] = result['accuracy']
                print(f"{name}: {result['accuracy']:.3f}")

            # Cluster true
            print('clustering true features...')
            true_results = {}
            for name, feat_array in true_combos.items():
                result = clustering_gmm(feat_array, type_list, n_components=75)
                true_results[name] = result['accuracy']
                print(f"{name}: {result['accuracy']:.3f}")

            # Plot two-panel figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            # Learned features - fixed order
            learned_order = ['a', 'τ', 'V', 'W', '(τ,V)', '(τ,V,W)', '(a,τ,V,W)']
            learned_vals = [learned_results[k] for k in ['a', 'τ', 'V', 'W', '(τ,V)', '(τ,V,W)', '(a,τ,V,W)']]
            colors_l = ['#d62728' if v < 0.6 else '#ff7f0e' if v < 0.85 else '#2ca02c' for v in learned_vals]
            ax1.barh(range(len(learned_order)), learned_vals, color=colors_l)
            ax1.set_yticks(range(len(learned_order)))
            ax1.set_yticklabels(learned_order, fontsize=11)
            ax1.set_xlabel('clustering accuracy', fontsize=12)
            ax1.set_title('learned features', fontsize=14)
            ax1.set_xlim([0, 1])
            ax1.grid(axis='x', alpha=0.3)
            ax1.invert_yaxis()
            for i, v in enumerate(learned_vals):
                ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
            # True features - fixed order
            true_order = ['τ', 'V', 'W', '(τ,V)', '(τ,V,W)']
            true_vals = [true_results[k] for k in ['τ', 'V', 'W', '(τ,V)', '(τ,V,W)']]
            colors_t = ['#d62728' if v < 0.6 else '#ff7f0e' if v < 0.85 else '#2ca02c' for v in true_vals]
            ax2.barh(range(len(true_order)), true_vals, color=colors_t)
            ax2.set_yticks(range(len(true_order)))
            ax2.set_yticklabels(true_order, fontsize=11)
            ax2.set_xlabel('clustering accuracy', fontsize=12)
            ax2.set_title('true features', fontsize=14)
            ax2.set_xlim([0, 1])
            ax2.grid(axis='x', alpha=0.3)
            ax2.invert_yaxis()
            for i, v in enumerate(true_vals):
                ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/clustering_comprehensive.png', dpi=300, bbox_inches='tight')
            plt.close()

            a_aug = np.column_stack([to_numpy(model.a), learned_tau, learned_V_rest,
                                    w_in_mean_learned, w_in_std_learned, w_out_mean_learned, w_out_std_learned])
            print('GMM learned a tau V_rest weights W:')

            best_acc = 0
            best_n = 0
            for n_comp in [50, 75, 100, 125, 150]:
                results = clustering_gmm(a_aug, type_list, n_components=n_comp)
                print(f"n_components={n_comp}: accuracy=\033[32m{results['accuracy']:.3f}\033[0m, ARI={results['ari']:.3f}, NMI={results['nmi']:.3f}")
                if results['accuracy'] > best_acc:
                    best_acc = results['accuracy']
                    best_n = n_comp

            print(f"best: n_components={best_n}, accuracy=\033[92m{best_acc:.3f}\033[0m")
            logger.info(f"GMM best: n_components={best_n}, accuracy={best_acc:.3f}")

            # Write cluster accuracy to analysis log file for Claude
            if log_file:
                print(f"  [DEBUG] plot_synaptic_flyvis: writing cluster_accuracy, log_file closed={log_file.closed}")
                log_file.write(f"cluster_accuracy: {best_acc:.4f}\n")

            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            a_umap = reducer.fit_transform(a_aug)

            # Get cluster labels from best GMM
            results = clustering_gmm(a_aug, type_list, n_components=best_n)
            cluster_labels = GaussianMixture(n_components=best_n, random_state=42).fit_predict(a_aug)

            plt.figure(figsize=(10, 9))
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_alpha(0.75)
            from matplotlib.colors import ListedColormap
            cmap_65 = ListedColormap(colors_65)
            plt.scatter(a_umap[:, 0], a_umap[:, 1], c=cluster_labels, s=24, cmap=cmap_65, alpha=0.8, edgecolors='none')


            plt.xlabel(r'UMAP$_1$', fontsize=48)
            plt.ylabel(r'UMAP$_2$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.text(0.05, 0.95, f"N: {n_neurons}\naccuracy: {best_acc:.2f}",
                    transform=plt.gca().transAxes, fontsize=32, verticalalignment='top')
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/embedding_augmented_{config_indices}.png', dpi=300)
            plt.close()


def analyze_neuron_type_reconstruction(config, model, edges, true_weights, gt_taus, gt_V_Rest,
                                       learned_weights, learned_tau, learned_V_rest, type_list, n_frames, dimension,
                                       n_neuron_types, device, log_dir, dataset_name, logger, index_to_name):

    print('stratified analysis by neuron type...')

    colors_65 = sns.color_palette("Set3", 12) * 6  # pastel, repeat until 65
    colors_65 = colors_65[:65]

    rmse_weights = []
    rmse_taus = []
    rmse_vrests = []
    n_connections = []

    for neuron_type in range(n_neuron_types):

        type_indices = np.where(type_list[edges[1,:]] == neuron_type)[0]
        gt_w_type = true_weights[type_indices]
        learned_w_type = learned_weights[type_indices]
        n_conn = len(type_indices)

        type_indices = np.where(type_list == neuron_type)[0]
        gt_tau_type = gt_taus[type_indices]
        gt_vrest_type = gt_V_Rest[type_indices]

        learned_tau_type = learned_tau[type_indices]
        learned_vrest_type = learned_V_rest[type_indices]

        rmse_w = np.sqrt(np.mean((gt_w_type - learned_w_type)** 2))
        rmse_tau = np.sqrt(np.mean((gt_tau_type - learned_tau_type)** 2))
        rmse_vrest = np.sqrt(np.mean((gt_vrest_type - learned_vrest_type)** 2))

        rmse_weights.append(rmse_w)
        rmse_taus.append(rmse_tau)
        rmse_vrests.append(rmse_vrest)
        n_connections.append(n_conn)

    n_neurons = len(type_list)

    # Per-neuron RMSE for tau
    rmse_tau_per_neuron = np.abs(learned_tau - gt_taus)
    # Per-neuron RMSE for V_rest
    rmse_vrest_per_neuron = np.abs(learned_V_rest - gt_V_Rest)
    # Per-neuron RMSE for weights (incoming connections)
    rmse_weights_per_neuron = np.zeros(n_neurons)
    for neuron_idx in range(n_neurons):
        incoming_edges = np.where(edges[1, :] == neuron_idx)[0]
        if len(incoming_edges) > 0:
            true_w = true_weights[incoming_edges]
            learned_w = learned_weights[incoming_edges]
            rmse_weights_per_neuron[neuron_idx] = np.sqrt(np.mean((learned_w - true_w)**2))

    # Convert to arrays
    rmse_weights = np.array(rmse_weights)
    rmse_taus = np.array(rmse_taus)
    rmse_vrests = np.array(rmse_vrests)

    unique_types_in_order = []
    seen_types = set()
    for i in range(len(type_list)):
        neuron_type_id = type_list[i].item() if hasattr(type_list[i], 'item') else int(type_list[i])
        if neuron_type_id not in seen_types:
            unique_types_in_order.append(neuron_type_id)
            seen_types.add(neuron_type_id)

    # Create neuron type names in the same order as they appear in data
    sorted_neuron_type_names = [index_to_name.get(type_id, f'Type{type_id}') for type_id in unique_types_in_order]
    unique_types_in_order = np.array(unique_types_in_order)
    sort_indices = unique_types_in_order.astype(int)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    x_pos = np.arange(len(sort_indices))

    # Plot weights RMSE
    ax1 = axes[0]
    ax1.bar(x_pos, rmse_weights[sort_indices], color='skyblue', alpha=0.7)
    ax1.set_ylabel('RMSE weights', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 2.5])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sorted_neuron_type_names, rotation=90, ha='right', fontsize=6)
    ax1.grid(False)
    ax1.tick_params(axis='y', labelsize=12)

    for i, (tick, rmse_w) in enumerate(zip(ax1.get_xticklabels(), rmse_weights[sort_indices])):
        if rmse_w > 0.5:
            tick.set_color('red')
            tick.set_fontsize(8)

    # Panel 2 (tau)
    ax2 = axes[1]
    ax2.bar(x_pos, rmse_taus[sort_indices], color='lightcoral', alpha=0.7)
    ax2.set_ylabel(r'RMSE $\tau$', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.3])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sorted_neuron_type_names, rotation=90, ha='right', fontsize=6)
    ax2.grid(False)
    ax2.tick_params(axis='y', labelsize=12)

    # Calculate mean ground truth taus per neuron type
    mean_gt_taus = []
    for neuron_type in range(n_neuron_types):
        type_indices = np.where(type_list == neuron_type)[0]
        gt_tau_type = gt_taus[type_indices]
        mean_gt_taus.append(np.mean(np.abs(gt_tau_type)))

    mean_gt_taus = np.array(mean_gt_taus)

    for i, (tick, rmse_tau) in enumerate(zip(ax2.get_xticklabels(), rmse_taus[sort_indices])):
        if rmse_tau > 0.03:
            tick.set_color('red')
            tick.set_fontsize(8)

    # Panel 3 (V_rest)
    ax3 = axes[2]
    ax3.bar(x_pos, rmse_vrests[sort_indices], color='lightgreen', alpha=0.7)
    ax3.set_ylabel(r'RMSE $V_{rest}$', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 0.8])
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(sorted_neuron_type_names, rotation=90, ha='right', fontsize=6)
    ax3.grid(False)
    ax3.tick_params(axis='y', labelsize=12)

    # Calculate mean ground truth V_rest per neuron type
    mean_gt_vrests = []
    for neuron_type in range(n_neuron_types):
        type_indices = np.where(type_list == neuron_type)[0]
        gt_vrest_type = gt_V_Rest[type_indices]
        mean_gt_vrests.append(np.mean(np.abs(gt_vrest_type)))

    mean_gt_vrests = np.array(mean_gt_vrests)
    for i, (tick, rmse_vrest) in enumerate(zip(ax3.get_xticklabels(), rmse_vrests[sort_indices])):
        if rmse_vrest > 0.08:
            tick.set_color('red')
            tick.set_fontsize(8)

    plt.tight_layout()
    plt.savefig(f'./{log_dir}/results/neuron_type_reconstruction.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Log summary statistics
    logger.info(f"mean weights RMSE: {np.mean(rmse_weights):.3f} ± {np.std(rmse_weights):.3f}")
    logger.info(f"mean tau RMSE: {np.mean(rmse_taus):.3f} ± {np.std(rmse_taus):.3f}")
    logger.info(f"mean V_rest RMSE: {np.mean(rmse_vrests):.3f} ± {np.std(rmse_vrests):.3f}")

    # Return per-neuron results (NEW)
    return {
        'rmse_weights_per_neuron': rmse_weights_per_neuron,
        'rmse_tau_per_neuron': rmse_tau_per_neuron,
        'rmse_vrest_per_neuron': rmse_vrest_per_neuron,
        'rmse_weights_per_type': rmse_weights,
        'rmse_tau_per_type': rmse_taus,
        'rmse_vrest_per_type': rmse_vrests
    }
    pass  # Implement as needed


def plot_neuron_activity_analysis(activity, target_type_name_list, type_list, index_to_name, n_neurons, n_frames, delta_t, output_path):

   # Calculate mean and std for each neuron
   mu_activity = torch.mean(activity, dim=1)
   sigma_activity = torch.std(activity, dim=1)

   # Create the plot (keeping original visualization)
   plt.figure(figsize=(16, 8))
   plt.errorbar(np.arange(n_neurons), to_numpy(mu_activity), yerr=to_numpy(sigma_activity),
                fmt='o', ecolor='lightgray', alpha=0.6, elinewidth=1, capsize=0,
                markersize=3, color='red')

   # Group neurons by type and add labels at type boundaries (similar to plot_ground_truth_distributions)
   type_boundaries = {}
   current_type = None
   for i in range(n_neurons):
       neuron_type_id = to_numpy(type_list[i]).item()
       if neuron_type_id != current_type:
           if current_type is not None:
               type_boundaries[current_type] = (type_boundaries[current_type][0], i - 1)
           type_boundaries[neuron_type_id] = (i, i)
           current_type = neuron_type_id

   # Close the last type boundary
   if current_type is not None:
       type_boundaries[current_type] = (type_boundaries[current_type][0], n_neurons - 1)

   # Add vertical lines and x-tick labels for each neuron type
   tick_positions = []
   tick_labels = []

   for neuron_type_id, (start_idx, end_idx) in type_boundaries.items():
       center_pos = (start_idx + end_idx) / 2
       neuron_type_name = index_to_name.get(neuron_type_id, f'Type{neuron_type_id}')

       tick_positions.append(center_pos)
       tick_labels.append(neuron_type_name)

       # Add vertical line at type boundary
       if start_idx > 0:
           plt.axvline(x=start_idx, color='gray', linestyle='--', alpha=0.3)

   # Set x-ticks with neuron type names rotated 90 degrees
   plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=10)
   plt.ylabel(r'neuron voltage $v_i(t)\quad\mu_i \pm \sigma_i$', fontsize=16)
   plt.yticks(fontsize=18)

   plt.tight_layout()
   plt.savefig(f'./{output_path}/activity_mu_sigma.png', dpi=300, bbox_inches='tight')
   plt.close()

   # Return per-neuron statistics (NEW)
   return {
       'mu_activity': to_numpy(mu_activity),
       'sigma_activity': to_numpy(sigma_activity)
   }


def plot_ground_truth_distributions(edges, true_weights, gt_taus, gt_V_Rest, type_list, n_neuron_types,
                                    sorted_neuron_type_names, output_path):
    """
    Create a 4-panel vertical figure showing ground truth parameter distributions per neuron type
    with neuron type names as x-axis labels
    """

    fig, axes = plt.subplots(4, 1, figsize=(12, 16))

    # Get type boundaries for labels
    type_boundaries = {}
    current_type = None
    n_neurons = len(type_list)

    for i in range(n_neurons):
        neuron_type_id = int(type_list[i])
        if neuron_type_id != current_type:
            if current_type is not None:
                type_boundaries[current_type] = (type_boundaries[current_type][0], i - 1)
            type_boundaries[neuron_type_id] = (i, i)
            current_type = neuron_type_id

    # Close the last type boundary
    if current_type is not None:
        type_boundaries[current_type] = (type_boundaries[current_type][0], n_neurons - 1)

    def add_type_labels_and_setup_axes(ax, y_values, title):
        # Add mean line for each type and collect type positions
        type_positions = []
        type_names = []

        for neuron_type_id, (start_idx, end_idx) in type_boundaries.items():
            center_pos = (start_idx + end_idx) / 2
            type_positions.append(center_pos)
            neuron_type_name = sorted_neuron_type_names[int(neuron_type_id)] if int(neuron_type_id) < len(
                sorted_neuron_type_names) else f'Type{neuron_type_id}'
            type_names.append(neuron_type_name)

            # Add mean line for this type
            type_mean = np.mean(y_values[start_idx:end_idx + 1])
            ax.hlines(type_mean, start_idx, end_idx, colors='red', linewidth=3)

        # Set x-ticks to neuron type names
        ax.set_xticks(type_positions)
        ax.set_xticklabels(type_names, rotation=90, fontsize=8)
        ax.tick_params(axis='y', labelsize=16)

    # Panel 1: Scatter plot of true weights per connection with neuron index
    ax1 = axes[0]
    connection_targets = edges[1, :]
    connection_weights = true_weights

    ax1.scatter(connection_targets, connection_weights, c='white', s=0.1)
    ax1.set_ylabel('true weights', fontsize=16)

    # For weights, compute means per target neuron
    weight_means_per_neuron = np.zeros(n_neurons)
    for i in range(n_neurons):
        incoming_edges = np.where(edges[1, :] == i)[0]
        if len(incoming_edges) > 0:
            weight_means_per_neuron[i] = np.mean(true_weights[incoming_edges])

    add_type_labels_and_setup_axes(ax1, weight_means_per_neuron, 'distribution of true weights by neuron type')

    # Panel 2: Number of connections per neuron
    ax2 = axes[1]
    n_connections_per_neuron = np.zeros(n_neurons)
    for i in range(n_neurons):
        n_connections_per_neuron[i] = np.sum(edges[1, :] == i)

    ax2.scatter(np.arange(n_neurons), n_connections_per_neuron, c='white', s=0.1)
    ax2.set_ylabel('number of connections', fontsize=16)
    add_type_labels_and_setup_axes(ax2, n_connections_per_neuron, 'number of incoming connections by neuron type')

    # Panel 3: Scatter plot of true tau values per neuron
    ax3 = axes[2]
    ax3.scatter(np.arange(n_neurons), gt_taus * 1000, c='white', s=0.1)
    ax3.set_ylabel(r'true $\tau$ values [ms]', fontsize=16)
    add_type_labels_and_setup_axes(ax3, gt_taus * 1000, r'distribution of true $\tau$ by neuron type')

    # Panel 4: Scatter plot of true V_rest values per neuron
    ax4 = axes[3]
    ax4.scatter(np.arange(n_neurons), gt_V_Rest, c='white', s=0.1)
    ax4.set_ylabel(r'true $v_{rest}$ values [a.u.]', fontsize=16)
    add_type_labels_and_setup_axes(ax4, gt_V_Rest, r'distribution of true $v_{rest}$ by neuron type')

    plt.tight_layout()
    plt.savefig(f'{output_path}/ground_truth_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    return fig
    plt.close()


def data_plot(config, config_file, epoch_list, style, extended, device, apply_weight_correction=False, log_file=None):

    print(f"  [DEBUG] data_plot: log_file={type(log_file).__name__}, config_file={config_file}")

    if 'black' in style:
        plt.style.use('dark_background')
        mc = 'w'
    else:
        plt.style.use('default')
        mc = 'k'

    fig_style.apply_globally()

    log_dir, logger = create_log_dir(config=config, erase=False)

    os.makedirs(os.path.join(log_dir, 'results'), exist_ok=True)

    if epoch_list==['best']:
        files = glob.glob(f"{log_dir}/models/*")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]

        epoch_list=[filename]
        print(f'best model: {epoch_list}')
        logger.info(f'best model: {epoch_list}')

    if os.path.exists(f'{log_dir}/loss.pt'):
        loss = torch.load(f'{log_dir}/loss.pt')
        fig, ax = fig_style.figure()
        plt.plot(loss, color=mc, linewidth=4)
        plt.xlim([0, 20])
        plt.ylabel('loss', fontsize=68)
        plt.xlabel('epochs', fontsize=68)
        plt.tight_layout()
        plt.savefig(f"{log_dir}/results/loss.png", dpi=170.7)
        plt.close()
        # Log final loss to analysis.log
        if log_file and len(loss) > 0:
            log_file.write(f"final_loss: {loss[-1]:.4e}\n")


    if 'fly' in config.dataset:
        if config.simulation.calcium_type != 'none':
            plot_synaptic_flyvis_calcium(config, epoch_list, log_dir, logger, 'viridis', style, extended, device) # noqa: F821
        else:
            plot_synaptic_flyvis(config, epoch_list, log_dir, logger, 'viridis', style, extended, device, log_file=log_file)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=FutureWarning)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    print(' ')
    print(f'device {device}')

    # try:
    #     matplotlib.use("Qt5Agg")
    # except:
    #     pass


    config_list = ['signal_Claude']


    for config_file_ in config_list:
        print(' ')
        config_file, pre_folder = add_pre_folder(config_file_)
        config = NeuralGraphConfig.from_yaml(config_path(f'{config_file}.yaml'))
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        print(f'\033[94mconfig_file  {config.config_file}\033[0m')
        folder_name = log_path(pre_folder, 'tmp_results') + '/'
        os.makedirs(folder_name, exist_ok=True)
        data_plot(config=config, config_file=config_file, epoch_list=['best'], style='black color', extended='plots', device=device, apply_weight_correction=True)
        # data_plot(config=config, config_file=config_file, epoch_list=['all'], style='black color', extended='plots', device=device, apply_weight_correction=False)
        # data_plot(config=config, config_file=config_file, epoch_list=['all'], style='black color', extended='plots', device=device, apply_weight_correction=True)


    print("analysis completed")


