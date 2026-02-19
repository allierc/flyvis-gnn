import os
import time
import glob
import shutil
import warnings
import logging

# Suppress matplotlib/PDF warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*Missing.*')

# Suppress fontTools logging (PDF font subsetting messages)
logging.getLogger('fontTools').setLevel(logging.ERROR)
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import numpy as np

from flyvis_gnn.models.utils import (
    set_trainable_parameters,
    analyze_data_svd,
    LossRegularizer,
    _batch_frames,
)
from flyvis_gnn.plot import plot_training_flyvis, plot_weight_comparison, plot_training_summary_panels, compute_dynamics_r2
from flyvis_gnn.utils import (
    to_numpy,
    CustomColorMap,
    create_log_dir,
    check_and_clear_memory,
    sort_key,
    fig_init,
    get_equidistant_points,
    compute_trace_metrics,
    get_datavis_root_dir,
)
from flyvis_gnn.figure_style import default_style, dark_style
from flyvis_gnn.plot import plot_spatial_activity_grid, INDEX_TO_NAME
from flyvis_gnn.models.flyvis_gnn import FlyVisGNN
from flyvis_gnn.models.registry import create_model
from flyvis_gnn.models.Neural_ode_wrapper_FlyVis import (
    integrate_neural_ode_FlyVis, neural_ode_loss_FlyVis,
    debug_check_gradients, DEBUG_ODE
)
from flyvis_gnn.zarr_io import load_simulation_data, load_raw_array
from flyvis_gnn.neuron_state import NeuronState

from flyvis_gnn.sparsify import EmbeddingCluster, sparsify_cluster, clustering_evaluation
from flyvis_gnn.fitting_models import linear_model

from scipy.optimize import curve_fit

import seaborn as sns
# denoise_data import not needed - removed star import
import imageio
imread = imageio.imread
from matplotlib.colors import LinearSegmentedColormap
from flyvis_gnn.generators.utils import generate_compressed_video_mp4, init_connectivity
from flyvis_gnn.plot import plot_signal_loss
from flyvis_gnn.generators.graph_data_generator import (
    apply_pairwise_knobs_torch,
    assign_columns_from_uv,
    build_neighbor_graph,
    compute_column_labels,
    greedy_blue_mask,
    mseq_bits,
)
try:
    from flyvis_gnn.generators.davis import AugmentedVideoDataset, CombinedVideoDataset
except ImportError:
    AugmentedVideoDataset = None
    CombinedVideoDataset = None
import pandas as pd
from collections import deque
from tqdm import tqdm, trange
from prettytable import PrettyTable
import imageio


def data_train(config=None, erase=False, best_model=None, style=None, device=None, log_file=None):
    # plt.rcParams['text.usetex'] = False  # LaTeX disabled - use mathtext instead
    # rc('font', **{'family': 'serif', 'serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']})
    # matplotlib.rcParams['savefig.pad_inches'] = 0

    seed = config.training.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # torch.autograd.set_detect_anomaly(True)

    print(f"\033[94mdataset: {config.dataset}\033[0m")
    print(f"\033[92m{config.description}\033[0m")

    if 'fly' in config.dataset:
        if 'RNN' in config.graph_model.signal_model_name or 'LSTM' in config.graph_model.signal_model_name:
            data_train_flyvis_RNN(config, erase, best_model, device)
        elif config.training.alternate_training:
            data_train_flyvis_alternate(config, erase, best_model, device, log_file=log_file)
        else:
            data_train_flyvis(config, erase, best_model, device, log_file=log_file)
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset}")

    print("training completed.")


def data_train_flyvis(config, erase, best_model, device, log_file=None):
    sim = config.simulation
    tc = config.training
    model_config = config.graph_model

    replace_with_cluster = 'replace' in tc.sparsity

    if config.training.seed != 42:
        torch.random.fork_rng(devices=device)
        torch.random.manual_seed(config.training.seed)

    default_style.apply_globally()

    if 'visual' in model_config.field_type:
        has_visual_field = True
        if 'instantNGP' in model_config.field_type:
            print('train with visual field instantNGP')
        else:
            print('train with visual field NNR')
    else:
        has_visual_field = False
    if 'test' in model_config.field_type:
        test_neural_field = True
        print('train with test field NNR')
    else:
        test_neural_field = False

    log_dir, logger = create_log_dir(config, erase)

    load_fields = ['voltage', 'stimulus', 'neuron_type']
    if has_visual_field or test_neural_field:
        load_fields.append('pos')
    if sim.calcium_type != 'none':
        load_fields.append('calcium')
    x_ts = load_simulation_data(f'graphs_data/{config.dataset}/x_list_0', fields=load_fields).to(device)
    y_ts = load_raw_array(f'graphs_data/{config.dataset}/y_list_0')

    # extract type_list from loaded data, then construct index (not loaded from disk)
    type_list = x_ts.neuron_type.float().unsqueeze(-1)
    x_ts.neuron_type = None
    x_ts.index = torch.arange(x_ts.n_neurons, dtype=torch.long, device=device)

    if tc.training_selected_neurons:
        selected_neuron_ids = np.array(tc.selected_neuron_ids).astype(int)
        x_ts = x_ts.subset_neurons(selected_neuron_ids)
        y_ts = y_ts[:, selected_neuron_ids, :]
        type_list = type_list[selected_neuron_ids]

    # get n_neurons from data, not config file
    n_neurons = x_ts.n_neurons
    config.simulation.n_neurons = n_neurons
    print(f'dataset: {x_ts.n_frames} frames,  n neurons: {n_neurons}')
    logger.info(f'n neurons: {n_neurons}')

    xnorm = x_ts.xnorm
    torch.save(xnorm, os.path.join(log_dir, 'xnorm.pt'))
    print(f'xnorm: {to_numpy(xnorm):0.3f}')
    logger.info(f'xnorm: {to_numpy(xnorm)}')
    ynorm = torch.tensor(1.0, device=device)
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    print(f'ynorm: {to_numpy(ynorm):0.3f}')
    logger.info(f'ynorm: {to_numpy(ynorm)}')

    # SVD analysis of activity and visual stimuli (skip if already exists)
    svd_plot_path = os.path.join(log_dir, 'results', 'svd_analysis.png')
    if not os.path.exists(svd_plot_path):
        analyze_data_svd(x_ts, log_dir, config=config, logger=logger, is_flyvis=True)
    else:
        print(f'svd analysis already exists: {svd_plot_path}')

    print('create models ...')
    model = create_model(model_config.signal_model_name,
                         aggr_type=model_config.aggr_type, config=config, device=device)
    model = model.to(device)

    # W init mode info
    w_init_mode = getattr(tc, 'w_init_mode', 'randn')
    if w_init_mode != 'randn':
        w_init_scale = getattr(tc, 'w_init_scale', 1.0)
        print(f'W init mode: {w_init_mode}' + (f' (scale={w_init_scale})' if w_init_mode == 'randn_scaled' else ''))

    start_epoch = 0
    list_loss = []
    if (best_model != None) & (best_model != '') & (best_model != '') & (best_model != 'None'):
        net = f"{log_dir}/models/best_model_with_{tc.n_runs - 1}_graphs_{best_model}.pt"
        print(f'loading state_dict from {net} ...')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'state_dict loaded: best_model={best_model}, start_epoch={start_epoch}')
    elif  tc.pretrained_model !='':
        net = tc.pretrained_model
        print(f'loading pretrained state_dict from {net} ...')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        print('pretrained state_dict loaded')
        logger.info(f'pretrained: {net}')
    else:
        print('no state_dict loaded - using freshly initialized model')

    # === LLM-MODIFIABLE: OPTIMIZER SETUP START ===
    # Change optimizer type, learning rate schedule, parameter groups

    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameters: {n_total_params:,}')
    lr = tc.learning_rate_start
    if tc.learning_rate_update_start == 0:
        lr_update = tc.learning_rate_start
    else:
        lr_update = tc.learning_rate_update_start
    lr_embedding = tc.learning_rate_embedding_start
    lr_W = tc.learning_rate_W_start
    learning_rate_NNR = tc.learning_rate_NNR
    learning_rate_NNR_f = tc.learning_rate_NNR_f

    print(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, learning_rate_NNR {learning_rate_NNR}')

    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                         lr_update=lr_update, lr_W=lr_W, learning_rate_NNR=learning_rate_NNR, learning_rate_NNR_f = learning_rate_NNR_f)
    # === LLM-MODIFIABLE: OPTIMIZER SETUP END ===
    model.train()

    net = f"{log_dir}/models/best_model_with_{tc.n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial tc.batch_size: {tc.batch_size}')

    gt_weights = torch.load(f'./graphs_data/{config.dataset}/weights.pt', map_location=device)
    edges = torch.load(f'./graphs_data/{config.dataset}/edge_index.pt', map_location=device)
    print(f'{edges.shape[1]} edges')

    ids = np.arange(n_neurons)

    if tc.coeff_W_sign > 0:
        index_weight = []
        for i in range(n_neurons):
            # get source neurons that connect to neuron i
            mask = edges[1] == i
            index_weight.append(edges[0][mask])

    logger.info(f'coeff_W_L1: {tc.coeff_W_L1} coeff_edge_diff: {tc.coeff_edge_diff} coeff_update_diff: {tc.coeff_update_diff}')
    print(f'coeff_W_L1: {tc.coeff_W_L1} coeff_edge_diff: {tc.coeff_edge_diff} coeff_update_diff: {tc.coeff_update_diff}')
     # proximal L1 info
    coeff_proximal = getattr(tc, 'coeff_W_L1_proximal', 0.0)
    if coeff_proximal > 0:
        print(f'proximal L1 soft-thresholding on W: coeff={coeff_proximal}')

    print("start training ...")

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)
    # torch.autograd.set_detect_anomaly(True)

    list_loss_regul = []

    regularizer = LossRegularizer(
        train_config=tc,
        model_config=model_config,
        activity_column=3,  # flyvis uses column 3 for activity
        plot_frequency=1,   # will be updated per epoch
        n_neurons=n_neurons,
        trainer_type='flyvis'
    )

    loss_components = {'loss': []}

    time.sleep(0.2)

    training_start_time = time.time()

    # Metrics log: tracks R2 evolution over training iterations
    metrics_log_path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    with open(metrics_log_path, 'w') as f:
        f.write('epoch,iteration,connectivity_r2,vrest_r2,tau_r2\n')

    for epoch in range(start_epoch, tc.n_epochs):

        Niter = int(sim.n_frames * tc.data_augmentation_loop // tc.batch_size * 0.2)
        plot_frequency = int(Niter // 20)
        connectivity_plot_frequency = int(Niter // 10)
        # Early-phase R2: 4 extra checkpoints in [1, connectivity_plot_frequency)
        early_r2_frequency = max(1, connectivity_plot_frequency // 5)
        n_plots_per_epoch = 4
        plot_iterations = set(int(x) for x in np.linspace(Niter // n_plots_per_epoch, Niter - 1, n_plots_per_epoch))
        print(f'{Niter} iterations per epoch, plot every {connectivity_plot_frequency} iterations '
              f'(early-phase every {early_r2_frequency} iterations)')

        total_loss = 0
        total_loss_regul = 0
        k = 0

        loss_noise_level = tc.loss_noise_level * (0.95 ** epoch)
        regularizer.set_epoch(epoch, plot_frequency)

        last_connectivity_r2 = None
        last_vrest_r2 = 0.0
        last_tau_r2 = 0.0
        field_R2 = None
        field_slope = None
        pbar = trange(Niter, ncols=150)
        # === LLM-MODIFIABLE: TRAINING LOOP START ===
        # Main training loop. Suggested changes: loss function, gradient clipping,
        # data sampling strategy, LR scheduler steps, early stopping.
        # Do NOT change: function signature, model construction, data loading, return values.
        for N in pbar:

            optimizer.zero_grad()

            state_batch = []
            y_list = []
            ids_list = []
            k_list = []
            visual_input_list = []
            ids_index = 0

            loss = 0

            for batch in range(tc.batch_size):

                k = np.random.randint(sim.n_frames - 4 - tc.time_step - tc.time_window) + tc.time_window

                if tc.recurrent_training or tc.neural_ODE_training:
                    k = k - k % tc.time_step

                x = x_ts.frame(k)

                if tc.time_window > 0:
                    x_temporal = x_ts.voltage[k - tc.time_window + 1: k + 1].T
                    # x stays as NeuronState; x_temporal passed separately to temporal model

                if has_visual_field:
                    visual_input = model.forward_visual(x, k)
                    x.stimulus[:model.n_input_neurons] = visual_input.squeeze(-1)
                    x.stimulus[model.n_input_neurons:] = 0

                loss = torch.zeros(1, device=device)
                regularizer.reset_iteration()

                if not (torch.isnan(x.voltage).any()):
                    regul_loss = regularizer.compute(
                        model=model,
                        x=x,
                        in_features=None,
                        ids=ids,
                        ids_batch=None,
                        edges=edges,
                        device=device,
                        xnorm=xnorm
                    )
                    loss = loss + regul_loss

                    if tc.recurrent_training or tc.neural_ODE_training:
                        y = x_ts.voltage[k + tc.time_step].unsqueeze(-1)
                    elif test_neural_field:
                        y = x_ts.stimulus[k, :sim.n_input_neurons].unsqueeze(-1)
                    else:
                        y = torch.tensor(y_ts[k], device=device) / ynorm     # loss on activity derivative


                    if loss_noise_level>0:
                        y = y + torch.randn(y.shape, device=device) * loss_noise_level

                    if not (torch.isnan(y).any()):

                        state_batch.append(x)
                        n = x.n_neurons
                        y_list.append(y)
                        ids_list.append(ids + ids_index)
                        k_list.append(torch.ones((n, 1), dtype=torch.int, device=device) * k)
                        if test_neural_field:
                            visual_input_list.append(visual_input)
                        ids_index += n


            if state_batch:

                data_id = torch.zeros((ids_index, 1), dtype=torch.int, device=device)
                y_batch = torch.cat(y_list, dim=0)
                ids_batch = np.concatenate(ids_list, axis=0)
                k_batch = torch.cat(k_list, dim=0)

                total_loss_regul += loss.item()

                if test_neural_field:
                    visual_input_batch = torch.cat(visual_input_list, dim=0)
                    loss = loss + (visual_input_batch - y_batch).norm(2)


                elif 'MLP_ODE' in model_config.signal_model_name:
                    batched_state, _ = _batch_frames(state_batch, edges)
                    batched_x = batched_state.to_packed()
                    pred = model(batched_x, data_id=data_id, return_all=False)

                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)

                elif 'MLP' in model_config.signal_model_name:
                    batched_state, _ = _batch_frames(state_batch, edges)
                    batched_x = batched_state.to_packed()
                    pred = model(batched_x, data_id=data_id, return_all=False)

                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)

                else: # 'GNN' branch

                    batched_state, batched_edges = _batch_frames(state_batch, edges)
                    pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

                    update_regul = regularizer.compute_update_regul(model, in_features, ids_batch, device)
                    loss = loss + update_regul


                    if tc.neural_ODE_training:

                        ode_state_clamp = getattr(tc, 'ode_state_clamp', 10.0)
                        ode_stab_lambda = getattr(tc, 'ode_stab_lambda', 0.0)
                        ode_loss, pred_x = neural_ode_loss_FlyVis(
                            model=model,
                            dataset_batch=state_batch,
                            edge_index=edges,
                            x_ts=x_ts,
                            k_batch=k_batch,
                            time_step=tc.time_step,
                            batch_size=tc.batch_size,
                            n_neurons=n_neurons,
                            ids_batch=ids_batch,
                            delta_t=sim.delta_t,
                            device=device,
                            data_id=data_id,
                            has_visual_field=has_visual_field,
                            y_batch=y_batch,
                            noise_level=tc.noise_recurrent_level,
                            ode_method=tc.ode_method,
                            rtol=tc.ode_rtol,
                            atol=tc.ode_atol,
                            adjoint=tc.ode_adjoint,
                            iteration=N,
                            state_clamp=ode_state_clamp,
                            stab_lambda=ode_stab_lambda
                        )
                        loss = loss + ode_loss


                    elif tc.recurrent_training:

                        pred_x = batched_state.voltage.unsqueeze(-1) + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

                        if tc.time_step > 1:
                            for step in range(tc.time_step - 1):
                                neurons_per_sample = state_batch[0].n_neurons

                                for b in range(tc.batch_size):
                                    start_idx = b * neurons_per_sample
                                    end_idx = (b + 1) * neurons_per_sample

                                    state_batch[b].voltage = pred_x[start_idx:end_idx].squeeze()

                                    k_current = k_batch[start_idx, 0].item() + step + 1

                                    if has_visual_field:
                                        visual_input_next = model.forward_visual(state_batch[b], k_current)
                                        state_batch[b].stimulus[:model.n_input_neurons] = visual_input_next.squeeze(-1)
                                        state_batch[b].stimulus[model.n_input_neurons:] = 0
                                    else:
                                        x_next = x_ts.frame(k_current)
                                        state_batch[b].stimulus = x_next.stimulus

                                batched_state, batched_edges = _batch_frames(state_batch, edges)
                                pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

                                pred_x = pred_x + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

                        loss = loss + ((pred_x[ids_batch] - y_batch[ids_batch]) / (sim.delta_t * tc.time_step)).norm(2)


                    else:

                        loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)


                # === LLM-MODIFIABLE: BACKWARD AND STEP START ===
                # Allowed changes: gradient clipping, LR scheduler step, loss scaling
                loss.backward()

                # debug gradient check for neural ODE training
                if DEBUG_ODE and tc.neural_ODE_training and (N % 500 == 0):
                    debug_check_gradients(model, loss, N)

                # W-specific gradient clipping: clip W gradients to force optimizer
                # to adjust lin_update (which contains V_rest/tau) instead of W
                if hasattr(tc, 'grad_clip_W') and tc.grad_clip_W > 0 and hasattr(model, 'W'):
                    if model.W.grad is not None:
                        torch.nn.utils.clip_grad_norm_([model.W], max_norm=tc.grad_clip_W)

                optimizer.step()
                # === LLM-MODIFIABLE: BACKWARD AND STEP END ===

                total_loss += loss.item()
                total_loss_regul += regularizer.get_iteration_total()

                # finalize iteration to record history
                regularizer.finalize_iteration()


                if regularizer.should_record():
                    # get history from regularizer and add loss component
                    current_loss = loss.item()
                    regul_total_this_iter = regularizer.get_iteration_total()
                    loss_components['loss'].append((current_loss - regul_total_this_iter) / n_neurons)

                    # merge loss_components with regularizer history for plotting
                    plot_dict = {**regularizer.get_history(), 'loss': loss_components['loss']}

                    # pass per-neuron normalized values to debug (to match dictionary values)
                    plot_signal_loss(plot_dict, log_dir, epoch=epoch, Niter=N, debug=False,
                                   current_loss=current_loss / n_neurons, current_regul=regul_total_this_iter / n_neurons,
                                   total_loss=total_loss, total_loss_regul=total_loss_regul)

                    torch.save(
                        {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                        os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}_{N}.pt'))

                # R2 checkpoint: regular interval + early-phase extra points
                is_regular_r2 = (N > 0) and (N % connectivity_plot_frequency == 0)
                is_early_r2 = (N > 0) and (N < connectivity_plot_frequency) and (N % early_r2_frequency == 0)
                if (is_regular_r2 or is_early_r2) & (not test_neural_field) & (not ('MLP' in model_config.signal_model_name)):
                    last_connectivity_r2 = plot_training_flyvis(x_ts, model, config, epoch, N, log_dir, device, type_list, gt_weights, edges, n_neurons=n_neurons, n_neuron_types=sim.n_neuron_types)
                    last_vrest_r2, last_tau_r2 = compute_dynamics_r2(model, x_ts, config, device, n_neurons)
                    with open(metrics_log_path, 'a') as f:
                        f.write(f'{epoch},{N},{last_connectivity_r2:.6f},{last_vrest_r2:.6f},{last_tau_r2:.6f}\n')

                if last_connectivity_r2 is not None:
                    r2 = last_connectivity_r2
                    color = '\033[92m' if r2 > 0.9 else '\033[93m' if r2 > 0.7 else '\033[38;5;208m' if r2 > 0.3 else '\033[91m'
                    pbar.set_postfix_str(f'{color}conn={r2:.3f} Vr={last_vrest_r2:.3f} τ={last_tau_r2:.3f}\033[0m')


                if (has_visual_field) & (N in plot_iterations):
                    with torch.no_grad():

                        # Static XY locations
                        X1 = to_numpy(x_ts.pos[:sim.n_input_neurons])

                        # group-based selection of 10 traces
                        groups = 217
                        group_size = sim.n_input_neurons // groups  # expect 8
                        assert groups * group_size == sim.n_input_neurons, "Unexpected packing of input neurons"
                        picked_groups = np.linspace(0, groups - 1, 10, dtype=int)
                        member_in_group = group_size // 2
                        trace_ids = (picked_groups * group_size + member_in_group).astype(int)

                        # MP4 writer setup
                        fps = 10
                        metadata = dict(title='Field Evolution', artist='Matplotlib', comment='NN Reconstruction over time')
                        writer = FFMpegWriter(fps=fps, metadata=metadata)
                        fig = plt.figure(figsize=(12, 4))

                        out_dir = f"./{log_dir}/tmp_training/external_input"
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = f"{out_dir}/field_movie_{epoch}_{N}.mp4"
                        if os.path.exists(out_path):
                            os.remove(out_path)

                        # rolling buffers
                        win = 200
                        offset = 1.25
                        hist_t = deque(maxlen=win)
                        hist_gt = {i: deque(maxlen=win) for i in trace_ids}
                        hist_pred = {i: deque(maxlen=win) for i in trace_ids}

                        step_video = 2

                        # First pass: collect all gt and pred, fit linear transform gt = a*pred + b
                        all_gt = []
                        all_pred = []
                        for k_fit in range(0, 800, step_video):
                            x_fit = x_ts.frame(k_fit)
                            pred_fit = to_numpy(model.forward_visual(x_fit, k_fit)).squeeze()
                            gt_fit = to_numpy(x_ts.stimulus[k_fit, :sim.n_input_neurons]).squeeze()
                            all_gt.append(gt_fit)
                            all_pred.append(pred_fit)
                        all_gt = np.concatenate(all_gt)
                        all_pred = np.concatenate(all_pred)

                        # Least-squares fit: gt = a * pred + b
                        A_fit = np.vstack([all_pred, np.ones(len(all_pred))]).T
                        a_coeff, b_coeff = np.linalg.lstsq(A_fit, all_gt, rcond=None)[0]
                        logger.info(f"field linear fit: gt = {a_coeff:.4f} * pred + {b_coeff:.4f}")

                        # Compute field_R2 on corrected predictions
                        pred_corrected_all = a_coeff * all_pred + b_coeff
                        ss_res = np.sum((all_gt - pred_corrected_all) ** 2)
                        ss_tot = np.sum((all_gt - np.mean(all_gt)) ** 2)
                        field_R2 = 1 - ss_res / (ss_tot + 1e-16)
                        field_slope = a_coeff
                        logger.info(f"external input R² (corrected): {field_R2:.4f}")

                        # GT value range for consistent color scaling
                        gt_vmin = float(all_gt.min())
                        gt_vmax = float(all_gt.max())

                        with writer.saving(fig, out_path, dpi=200):
                            error_list = []

                            for k in trange(0, 800, step_video, ncols=100):
                                # inputs and predictions
                                x = x_ts.frame(k)
                                pred = to_numpy(model.forward_visual(x, k))
                                pred_vec = np.asarray(pred).squeeze()  # (sim.n_input_neurons,)
                                pred_corrected = a_coeff * pred_vec # + b_coeff  # corrected to GT scale

                                gt_vec = to_numpy(x_ts.stimulus[k, :sim.n_input_neurons]).squeeze()

                                # update rolling traces (store corrected predictions)
                                hist_t.append(k)
                                for i in trace_ids:
                                    hist_gt[i].append(gt_vec[i])
                                    hist_pred[i].append(pred_corrected[i])

                                # draw three panels
                                fig.clf()

                                # RMSE on corrected predictions
                                rmse_frame = float(np.sqrt(((pred_corrected - gt_vec) ** 2).mean()))
                                running_rmse = float(np.mean(error_list + [rmse_frame])) if len(error_list) else rmse_frame

                                # Traces (both on GT scale)
                                ax3 = fig.add_subplot(1, 3, 3)
                                ax3.set_axis_off()
                                ax3.set_facecolor("black")

                                t = np.arange(len(hist_t))
                                for j, i in enumerate(trace_ids):
                                    y0 = j * offset
                                    ax3.plot(t, np.array(hist_gt[i])   + y0, color='lime',  lw=1.6, alpha=0.95)
                                    ax3.plot(t, np.array(hist_pred[i]) + y0, color='k', lw=1.2, alpha=0.95)

                                ax3.set_xlim(max(0, len(t) - win), len(t))
                                ax3.set_ylim(-offset * 0.5, offset * (len(trace_ids) + 0.5))
                                ax3.text(
                                    0.02, 0.98,
                                    f"frame: {k}   RMSE: {rmse_frame:.3f}   avg RMSE: {running_rmse:.3f}   a={a_coeff:.3f} b={b_coeff:.3f}",
                                    transform=ax3.transAxes,
                                    va='top', ha='left',
                                    fontsize=6, color='k')

                                # GT field
                                ax1 = fig.add_subplot(1, 3, 1)
                                ax1.scatter(X1[:, 0], X1[:, 1], s=256, c=gt_vec, cmap=default_style.cmap, marker='h', vmin=gt_vmin, vmax=gt_vmax)
                                ax1.set_axis_off()
                                ax1.set_title('ground truth', fontsize=12)

                                # Predicted field (corrected, same scale as GT)
                                ax2 = fig.add_subplot(1, 3, 2)
                                ax2.scatter(X1[:, 0], X1[:, 1], s=256, c=pred_corrected, cmap=default_style.cmap, marker='h')
                                ax2.set_axis_off()
                                ax2.set_title('prediction (corrected)', fontsize=12)

                                plt.tight_layout()
                                writer.grab_frame()

                                error_list.append(rmse_frame)


                    if last_connectivity_r2 is not None:
                        # color code: green (>0.9), yellow (0.7-0.9), orange (0.3-0.7), red (<0.3)
                        if last_connectivity_r2 > 0.9:
                            r2_color = '\033[92m'  # green
                        elif last_connectivity_r2 > 0.7:
                            r2_color = '\033[93m'  # yellow
                        elif last_connectivity_r2 > 0.3:
                            r2_color = '\033[38;5;208m'  # orange
                        else:
                            r2_color = '\033[91m'  # red
                        pbar.set_postfix_str(f'{r2_color}R²={last_connectivity_r2:.3f}\033[0m')
                    torch.save(
                        {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                        os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}_{N}.pt'))

            # check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50, memory_percentage_threshold=0.6)

        # === LLM-MODIFIABLE: TRAINING LOOP END ===

        # Calculate epoch-level losses
        epoch_total_loss = total_loss / n_neurons
        epoch_regul_loss = total_loss_regul / n_neurons
        epoch_pred_loss = (total_loss - total_loss_regul) / n_neurons

        print("epoch {}. loss: {:.6f} (pred: {:.6f}, regul: {:.6f})".format(
            epoch, epoch_total_loss, epoch_pred_loss, epoch_regul_loss))
        logger.info("Epoch {}. Loss: {:.6f} (pred: {:.6f}, regul: {:.6f})".format(
            epoch, epoch_total_loss, epoch_pred_loss, epoch_regul_loss))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}.pt'))

        list_loss.append(epoch_pred_loss)
        list_loss_regul.append(epoch_regul_loss)

        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(15, 10))

        # Plot 1: Loss
        fig.add_subplot(2, 3, 1)
        plt.plot(list_loss, color=default_style.foreground, linewidth=1)
        plt.xlim([0, tc.n_epochs])
        plt.ylabel('loss', fontsize=12)
        plt.xlabel('epochs', fontsize=12)

        plot_training_summary_panels(fig, log_dir)

        if replace_with_cluster:

            if (epoch % tc.sparsity_freq == tc.sparsity_freq - 1) & (epoch < tc.n_epochs - tc.sparsity_freq):
                print('replace embedding with clusters ...')
                eps = tc.cluster_distance_threshold
                results = clustering_evaluation(to_numpy(model.a), type_list, eps=eps)
                print(f"eps={eps}: {results['n_clusters_found']} clusters, "
                      f"accuracy={results['accuracy']:.3f}")

                labels = results['cluster_labels']

                for n in np.unique(labels):
                    # if n == -1:
                    #     continue
                    indices = np.where(labels == n)[0]
                    if len(indices) > 1:
                        with torch.no_grad():
                            model.a[indices, :] = torch.mean(model.a[indices, :], dim=0, keepdim=True)

                fig.add_subplot(2, 3, 6)
                type_cmap = CustomColorMap(config=config)
                for n in range(sim.n_neuron_types):
                    pos = torch.argwhere(type_list == n)
                    plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=5, color=type_cmap.color(n),
                                alpha=0.7, edgecolors='none')
                plt.xlabel('embedding 0', fontsize=18)
                plt.ylabel('embedding 1', fontsize=18)
                plt.xticks([])
                plt.yticks([])
                plt.text(0.5, 0.9, f"eps={eps}: {results['n_clusters_found']} clusters, accuracy={results['accuracy']:.3f}")

                if tc.fix_cluster_embedding:
                    lr_embedding = 1.0E-10
                    # the embedding is fixed for 1 epoch

            else:
                lr = tc.learning_rate_start
                lr_embedding = tc.learning_rate_embedding_start
                lr_W = tc.learning_rate_W_start
                learning_rate_NNR = tc.learning_rate_NNR

            logger.info(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, learning_rate_NNR {learning_rate_NNR}')
            optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr, lr_update=lr_update, lr_W=lr_W,
                                                                 learning_rate_NNR=learning_rate_NNR)

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/epoch_{epoch}.png")
        plt.close()

    # Calculate and log training time
    training_time = time.time() - training_start_time
    training_time_min = training_time / 60.0
    print(f"training completed in {training_time_min:.1f} minutes")
    logger.info(f"training completed in {training_time_min:.1f} minutes")

    if log_file is not None:
        log_file.write(f"training_time_min: {training_time_min:.1f}\n")
        log_file.write(f"n_epochs: {tc.n_epochs}\n")
        log_file.write(f"data_augmentation_loop: {tc.data_augmentation_loop}\n")
        log_file.write(f"recurrent_training: {tc.recurrent_training}\n")
        log_file.write(f"batch_size: {tc.batch_size}\n")
        log_file.write(f"learning_rate_W: {tc.learning_rate_W_start}\n")
        log_file.write(f"learning_rate: {tc.learning_rate_start}\n")
        log_file.write(f"learning_rate_embedding: {tc.learning_rate_embedding_start}\n")
        log_file.write(f"coeff_edge_diff: {tc.coeff_edge_diff}\n")
        log_file.write(f"coeff_edge_norm: {tc.coeff_edge_norm}\n")
        log_file.write(f"coeff_edge_weight_L1: {tc.coeff_edge_weight_L1}\n")
        log_file.write(f"coeff_phi_weight_L1: {tc.coeff_phi_weight_L1}\n")
        log_file.write(f"coeff_phi_weight_L2: {tc.coeff_phi_weight_L2}\n")
        log_file.write(f"coeff_W_L1: {tc.coeff_W_L1}\n")
        if field_R2 is not None:
            log_file.write(f"field_R2: {field_R2:.4f}\n")
            log_file.write(f"field_slope: {field_slope:.4f}\n")

def data_train_flyvis_alternate(config, erase, best_model, device, log_file=None):
    """Alternating W/V_rest phase training for flyvis models."""
    sim = config.simulation
    tc = config.training
    model_config = config.graph_model

    replace_with_cluster = 'replace' in tc.sparsity

    if config.training.seed != 42:
        torch.random.fork_rng(devices=device)
        torch.random.manual_seed(config.training.seed)

    default_style.apply_globally()

    if 'visual' in model_config.field_type:
        has_visual_field = True
        if 'instantNGP' in model_config.field_type:
            print('train with visual field instantNGP')
        else:
            print('train with visual field NNR')
    else:
        has_visual_field = False
    if 'test' in model_config.field_type:
        test_neural_field = True
        print('train with test field NNR')
    else:
        test_neural_field = False

    log_dir, logger = create_log_dir(config, erase)

    load_fields = ['voltage', 'stimulus', 'neuron_type']
    if has_visual_field or test_neural_field:
        load_fields.append('pos')
    if sim.calcium_type != 'none':
        load_fields.append('calcium')
    x_ts = load_simulation_data(f'graphs_data/{config.dataset}/x_list_0', fields=load_fields).to(device)
    y_ts = load_raw_array(f'graphs_data/{config.dataset}/y_list_0')

    # extract type_list from loaded data, then construct index (not loaded from disk)
    type_list = x_ts.neuron_type.float().unsqueeze(-1)
    x_ts.neuron_type = None
    x_ts.index = torch.arange(x_ts.n_neurons, dtype=torch.long, device=device)

    if tc.training_selected_neurons:
        selected_neuron_ids = np.array(tc.selected_neuron_ids).astype(int)
        x_ts = x_ts.subset_neurons(selected_neuron_ids)
        y_ts = y_ts[:, selected_neuron_ids, :]
        type_list = type_list[selected_neuron_ids]

    # get n_neurons from data, not config file
    n_neurons = x_ts.n_neurons
    config.simulation.n_neurons = n_neurons
    print(f'dataset: {x_ts.n_frames} frames,  n neurons: {n_neurons}')
    logger.info(f'n neurons: {n_neurons}')

    xnorm = x_ts.xnorm
    torch.save(xnorm, os.path.join(log_dir, 'xnorm.pt'))
    print(f'xnorm: {to_numpy(xnorm):0.3f}')
    logger.info(f'xnorm: {to_numpy(xnorm)}')
    ynorm = torch.tensor(1.0, device=device)
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    print(f'ynorm: {to_numpy(ynorm):0.3f}')
    logger.info(f'ynorm: {to_numpy(ynorm)}')

    # SVD analysis of activity and visual stimuli (skip if already exists)
    svd_plot_path = os.path.join(log_dir, 'results', 'svd_analysis.png')
    if not os.path.exists(svd_plot_path):
        analyze_data_svd(x_ts, log_dir, config=config, logger=logger, is_flyvis=True)
    else:
        print(f'svd analysis already exists: {svd_plot_path}')

    print('create models ...')
    model = create_model(model_config.signal_model_name,
                         aggr_type=model_config.aggr_type, config=config, device=device)
    model = model.to(device)

    # W init mode info
    w_init_mode = getattr(tc, 'w_init_mode', 'randn')
    if w_init_mode != 'randn':
        w_init_scale = getattr(tc, 'w_init_scale', 1.0)
        print(f'W init mode: {w_init_mode}' + (f' (scale={w_init_scale})' if w_init_mode == 'randn_scaled' else ''))

    start_epoch = 0
    list_loss = []
    if (best_model != None) & (best_model != '') & (best_model != '') & (best_model != 'None'):
        net = f"{log_dir}/models/best_model_with_{tc.n_runs - 1}_graphs_{best_model}.pt"
        print(f'loading state_dict from {net} ...')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'state_dict loaded: best_model={best_model}, start_epoch={start_epoch}')
    elif  tc.pretrained_model !='':
        net = tc.pretrained_model
        print(f'loading pretrained state_dict from {net} ...')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        print('pretrained state_dict loaded')
        logger.info(f'pretrained: {net}')
    else:
        print('no state_dict loaded - using freshly initialized model')

    # === LLM-MODIFIABLE: OPTIMIZER SETUP START ===
    # Change optimizer type, learning rate schedule, parameter groups

    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameters: {n_total_params:,}')
    lr = tc.learning_rate_start
    if tc.learning_rate_update_start == 0:
        lr_update = tc.learning_rate_start
    else:
        lr_update = tc.learning_rate_update_start
    lr_embedding = tc.learning_rate_embedding_start
    lr_W = tc.learning_rate_W_start
    learning_rate_NNR = tc.learning_rate_NNR
    learning_rate_NNR_f = tc.learning_rate_NNR_f

    print(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, learning_rate_NNR {learning_rate_NNR}')

    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                         lr_update=lr_update, lr_W=lr_W, learning_rate_NNR=learning_rate_NNR, learning_rate_NNR_f = learning_rate_NNR_f)
    # === LLM-MODIFIABLE: OPTIMIZER SETUP END ===
    model.train()

    net = f"{log_dir}/models/best_model_with_{tc.n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial tc.batch_size: {tc.batch_size}')

    gt_weights = torch.load(f'./graphs_data/{config.dataset}/weights.pt', map_location=device)
    edges = torch.load(f'./graphs_data/{config.dataset}/edge_index.pt', map_location=device)
    print(f'{edges.shape[1]} edges')

    ids = np.arange(n_neurons)

    if tc.coeff_W_sign > 0:
        index_weight = []
        for i in range(n_neurons):
            # get source neurons that connect to neuron i
            mask = edges[1] == i
            index_weight.append(edges[0][mask])

    logger.info(f'coeff_W_L1: {tc.coeff_W_L1} coeff_edge_diff: {tc.coeff_edge_diff} coeff_update_diff: {tc.coeff_update_diff}')
    print(f'coeff_W_L1: {tc.coeff_W_L1} coeff_edge_diff: {tc.coeff_edge_diff} coeff_update_diff: {tc.coeff_update_diff}')
     # proximal L1 info
    coeff_proximal = getattr(tc, 'coeff_W_L1_proximal', 0.0)
    if coeff_proximal > 0:
        print(f'proximal L1 soft-thresholding on W: coeff={coeff_proximal}')

    print("start alternating training ...")

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)

    list_loss_regul = []

    regularizer = LossRegularizer(
        train_config=tc,
        model_config=model_config,
        activity_column=3,  # flyvis uses column 3 for activity
        plot_frequency=1,   # will be updated per epoch
        n_neurons=n_neurons,
        trainer_type='flyvis'
    )

    loss_components = {'loss': []}

    time.sleep(0.2)

    training_start_time = time.time()

    # Metrics log with phase info
    metrics_log_path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    with open(metrics_log_path, 'w') as f:
        f.write('epoch,iteration,connectivity_r2,vrest_r2,tau_r2,phase\n')

    # Two-stage LR lookup: phase_name -> {param_group_name -> lr}
    phase_lrs = {
        'joint': {
            'lin_edge': lr,
            'W': lr_W,
            'lin_phi': lr_update,
            'embedding': lr_embedding,
        },
        'V_rest': {
            'lin_edge': lr * tc.alternate_lr_ratio,
            'W': lr_W * tc.alternate_lr_ratio,
            'lin_phi': lr_update,
            'embedding': lr_embedding,
        },
    }
    print(f'two-stage LRs — joint: all at full LR')
    print(f'two-stage LRs — V_rest focus: lin_edge={lr * tc.alternate_lr_ratio:.2e}, W={lr_W * tc.alternate_lr_ratio:.2e}, lin_phi={lr_update}, emb={lr_embedding}')

    for epoch in range(start_epoch, tc.n_epochs):

        Niter = int(sim.n_frames * tc.data_augmentation_loop // tc.batch_size * 0.2)
        plot_frequency = int(Niter // 20)
        connectivity_plot_frequency = int(Niter // 10)
        # Early-phase R2: 4 extra checkpoints in [1, connectivity_plot_frequency)
        early_r2_frequency = max(1, connectivity_plot_frequency // 5)
        n_plots_per_epoch = 4
        plot_iterations = set(int(x) for x in np.linspace(Niter // n_plots_per_epoch, Niter - 1, n_plots_per_epoch))
        print(f'{Niter} iterations per epoch, plot every {connectivity_plot_frequency} iterations '
              f'(early-phase every {early_r2_frequency} iterations)')

        # Build two-stage phase schedule: joint warmup then V_rest focus
        joint_iters = int(Niter * tc.alternate_joint_ratio)
        phases = [('joint', 0, joint_iters), ('V_rest', joint_iters, Niter)]
        current_phase_idx = 0
        current_phase = phases[0][0]
        print(f'two-stage: joint={joint_iters} iters ({tc.alternate_joint_ratio:.0%}), V_rest focus={Niter - joint_iters} iters ({1 - tc.alternate_joint_ratio:.0%})')

        # Set initial phase LRs (joint phase = all at full LR)
        for pg in optimizer.param_groups:
            pg_name = pg.get('name', '')
            if pg_name in phase_lrs['joint']:
                pg['lr'] = phase_lrs['joint'][pg_name]

        total_loss = 0
        total_loss_regul = 0
        k = 0

        loss_noise_level = tc.loss_noise_level * (0.95 ** epoch)
        regularizer.set_epoch(epoch, plot_frequency)

        last_connectivity_r2 = None
        last_vrest_r2 = 0.0
        last_tau_r2 = 0.0
        field_R2 = None
        field_slope = None
        pbar = trange(Niter, ncols=150)
        # === LLM-MODIFIABLE: TRAINING LOOP START ===
        # Main training loop with two-stage joint/V_rest phases.
        # Suggested changes: loss function, gradient clipping,
        # data sampling strategy, LR scheduler steps, early stopping.
        # Do NOT change: function signature, model construction, data loading, return values.
        for N in pbar:

            # Phase transition check
            if current_phase_idx < len(phases) - 1 and N >= phases[current_phase_idx + 1][1]:
                current_phase_idx += 1
                new_phase = phases[current_phase_idx][0]
                if new_phase != current_phase:
                    current_phase = new_phase
                    print(f'\n>>> Phase transition at iter {N}: {current_phase} (lr_ratio={tc.alternate_lr_ratio if current_phase == "V_rest" else 1.0})')
                    for pg in optimizer.param_groups:
                        pg_name = pg.get('name', '')
                        if pg_name in phase_lrs[current_phase]:
                            pg['lr'] = phase_lrs[current_phase][pg_name]

            optimizer.zero_grad()

            state_batch = []
            y_list = []
            ids_list = []
            k_list = []
            visual_input_list = []
            ids_index = 0

            loss = 0

            for batch in range(tc.batch_size):

                k = np.random.randint(sim.n_frames - 4 - tc.time_step - tc.time_window) + tc.time_window

                if tc.recurrent_training or tc.neural_ODE_training:
                    k = k - k % tc.time_step

                x = x_ts.frame(k)

                if tc.time_window > 0:
                    x_temporal = x_ts.voltage[k - tc.time_window + 1: k + 1].T
                    # x stays as NeuronState; x_temporal passed separately to temporal model

                if has_visual_field:
                    visual_input = model.forward_visual(x, k)
                    x.stimulus[:model.n_input_neurons] = visual_input.squeeze(-1)
                    x.stimulus[model.n_input_neurons:] = 0

                loss = torch.zeros(1, device=device)
                regularizer.reset_iteration()

                if not (torch.isnan(x.voltage).any()):
                    regul_loss = regularizer.compute(
                        model=model,
                        x=x,
                        in_features=None,
                        ids=ids,
                        ids_batch=None,
                        edges=edges,
                        device=device,
                        xnorm=xnorm
                    )
                    loss = loss + regul_loss

                    if tc.recurrent_training or tc.neural_ODE_training:
                        y = x_ts.voltage[k + tc.time_step].unsqueeze(-1)
                    elif test_neural_field:
                        y = x_ts.stimulus[k, :sim.n_input_neurons].unsqueeze(-1)
                    else:
                        y = torch.tensor(y_ts[k], device=device) / ynorm     # loss on activity derivative


                    if loss_noise_level>0:
                        y = y + torch.randn(y.shape, device=device) * loss_noise_level

                    if not (torch.isnan(y).any()):

                        state_batch.append(x)
                        n = x.n_neurons
                        y_list.append(y)
                        ids_list.append(ids + ids_index)
                        k_list.append(torch.ones((n, 1), dtype=torch.int, device=device) * k)
                        if test_neural_field:
                            visual_input_list.append(visual_input)
                        ids_index += n


            if state_batch:

                data_id = torch.zeros((ids_index, 1), dtype=torch.int, device=device)
                y_batch = torch.cat(y_list, dim=0)
                ids_batch = np.concatenate(ids_list, axis=0)
                k_batch = torch.cat(k_list, dim=0)

                total_loss_regul += loss.item()

                if test_neural_field:
                    visual_input_batch = torch.cat(visual_input_list, dim=0)
                    loss = loss + (visual_input_batch - y_batch).norm(2)


                elif 'MLP_ODE' in model_config.signal_model_name:
                    batched_state, _ = _batch_frames(state_batch, edges)
                    batched_x = batched_state.to_packed()
                    pred = model(batched_x, data_id=data_id, return_all=False)

                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)

                elif 'MLP' in model_config.signal_model_name:
                    batched_state, _ = _batch_frames(state_batch, edges)
                    batched_x = batched_state.to_packed()
                    pred = model(batched_x, data_id=data_id, return_all=False)

                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)

                else: # 'GNN' branch

                    batched_state, batched_edges = _batch_frames(state_batch, edges)
                    pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

                    update_regul = regularizer.compute_update_regul(model, in_features, ids_batch, device)
                    loss = loss + update_regul


                    if tc.neural_ODE_training:

                        ode_state_clamp = getattr(tc, 'ode_state_clamp', 10.0)
                        ode_stab_lambda = getattr(tc, 'ode_stab_lambda', 0.0)
                        ode_loss, pred_x = neural_ode_loss_FlyVis(
                            model=model,
                            dataset_batch=state_batch,
                            edge_index=edges,
                            x_ts=x_ts,
                            k_batch=k_batch,
                            time_step=tc.time_step,
                            batch_size=tc.batch_size,
                            n_neurons=n_neurons,
                            ids_batch=ids_batch,
                            delta_t=sim.delta_t,
                            device=device,
                            data_id=data_id,
                            has_visual_field=has_visual_field,
                            y_batch=y_batch,
                            noise_level=tc.noise_recurrent_level,
                            ode_method=tc.ode_method,
                            rtol=tc.ode_rtol,
                            atol=tc.ode_atol,
                            adjoint=tc.ode_adjoint,
                            iteration=N,
                            state_clamp=ode_state_clamp,
                            stab_lambda=ode_stab_lambda
                        )
                        loss = loss + ode_loss


                    elif tc.recurrent_training:

                        pred_x = batched_state.voltage.unsqueeze(-1) + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

                        if tc.time_step > 1:
                            for step in range(tc.time_step - 1):
                                neurons_per_sample = state_batch[0].n_neurons

                                for b in range(tc.batch_size):
                                    start_idx = b * neurons_per_sample
                                    end_idx = (b + 1) * neurons_per_sample

                                    state_batch[b].voltage = pred_x[start_idx:end_idx].squeeze()

                                    k_current = k_batch[start_idx, 0].item() + step + 1

                                    if has_visual_field:
                                        visual_input_next = model.forward_visual(state_batch[b], k_current)
                                        state_batch[b].stimulus[:model.n_input_neurons] = visual_input_next.squeeze(-1)
                                        state_batch[b].stimulus[model.n_input_neurons:] = 0
                                    else:
                                        x_next = x_ts.frame(k_current)
                                        state_batch[b].stimulus = x_next.stimulus

                                batched_state, batched_edges = _batch_frames(state_batch, edges)
                                pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

                                pred_x = pred_x + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

                        loss = loss + ((pred_x[ids_batch] - y_batch[ids_batch]) / (sim.delta_t * tc.time_step)).norm(2)


                    else:

                        loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)


                # === LLM-MODIFIABLE: BACKWARD AND STEP START ===
                # Allowed changes: gradient clipping, LR scheduler step, loss scaling
                loss.backward()

                # debug gradient check for neural ODE training
                if DEBUG_ODE and tc.neural_ODE_training and (N % 500 == 0):
                    debug_check_gradients(model, loss, N)

                # W-specific gradient clipping: clip W gradients to force optimizer
                # to adjust lin_update (which contains V_rest/tau) instead of W
                if hasattr(tc, 'grad_clip_W') and tc.grad_clip_W > 0 and hasattr(model, 'W'):
                    if model.W.grad is not None:
                        torch.nn.utils.clip_grad_norm_([model.W], max_norm=tc.grad_clip_W)

                optimizer.step()
                # === LLM-MODIFIABLE: BACKWARD AND STEP END ===

                total_loss += loss.item()
                total_loss_regul += regularizer.get_iteration_total()

                # finalize iteration to record history
                regularizer.finalize_iteration()


                if regularizer.should_record():
                    # get history from regularizer and add loss component
                    current_loss = loss.item()
                    regul_total_this_iter = regularizer.get_iteration_total()
                    loss_components['loss'].append((current_loss - regul_total_this_iter) / n_neurons)

                    # merge loss_components with regularizer history for plotting
                    plot_dict = {**regularizer.get_history(), 'loss': loss_components['loss']}

                    # pass per-neuron normalized values to debug (to match dictionary values)
                    plot_signal_loss(plot_dict, log_dir, epoch=epoch, Niter=N, debug=False,
                                   current_loss=current_loss / n_neurons, current_regul=regul_total_this_iter / n_neurons,
                                   total_loss=total_loss, total_loss_regul=total_loss_regul)

                    torch.save(
                        {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                        os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}_{N}.pt'))

                # R2 checkpoint: regular interval + early-phase extra points
                is_regular_r2 = (N > 0) and (N % connectivity_plot_frequency == 0)
                is_early_r2 = (N > 0) and (N < connectivity_plot_frequency) and (N % early_r2_frequency == 0)
                if (is_regular_r2 or is_early_r2) & (not test_neural_field) & (not ('MLP' in model_config.signal_model_name)):
                    last_connectivity_r2 = plot_training_flyvis(x_ts, model, config, epoch, N, log_dir, device, type_list, gt_weights, edges, n_neurons=n_neurons, n_neuron_types=sim.n_neuron_types)
                    last_vrest_r2, last_tau_r2 = compute_dynamics_r2(model, x_ts, config, device, n_neurons)
                    with open(metrics_log_path, 'a') as f:
                        f.write(f'{epoch},{N},{last_connectivity_r2:.6f},{last_vrest_r2:.6f},{last_tau_r2:.6f},{current_phase}\n')

                if last_connectivity_r2 is not None:
                    r2 = last_connectivity_r2
                    color = '\033[92m' if r2 > 0.9 else '\033[93m' if r2 > 0.7 else '\033[38;5;208m' if r2 > 0.3 else '\033[91m'
                    pbar.set_postfix_str(f'{color}conn={r2:.3f} Vr={last_vrest_r2:.3f} τ={last_tau_r2:.3f} [{current_phase}]\033[0m')


                if (has_visual_field) & (N in plot_iterations):
                    with torch.no_grad():

                        # Static XY locations
                        X1 = to_numpy(x_ts.pos[:sim.n_input_neurons])

                        # group-based selection of 10 traces
                        groups = 217
                        group_size = sim.n_input_neurons // groups  # expect 8
                        assert groups * group_size == sim.n_input_neurons, "Unexpected packing of input neurons"
                        picked_groups = np.linspace(0, groups - 1, 10, dtype=int)
                        member_in_group = group_size // 2
                        trace_ids = (picked_groups * group_size + member_in_group).astype(int)

                        # MP4 writer setup
                        fps = 10
                        metadata = dict(title='Field Evolution', artist='Matplotlib', comment='NN Reconstruction over time')
                        writer = FFMpegWriter(fps=fps, metadata=metadata)
                        fig = plt.figure(figsize=(12, 4))

                        out_dir = f"./{log_dir}/tmp_training/external_input"
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = f"{out_dir}/field_movie_{epoch}_{N}.mp4"
                        if os.path.exists(out_path):
                            os.remove(out_path)

                        # rolling buffers
                        win = 200
                        offset = 1.25
                        hist_t = deque(maxlen=win)
                        hist_gt = {i: deque(maxlen=win) for i in trace_ids}
                        hist_pred = {i: deque(maxlen=win) for i in trace_ids}

                        step_video = 2

                        # First pass: collect all gt and pred, fit linear transform gt = a*pred + b
                        all_gt = []
                        all_pred = []
                        for k_fit in range(0, 800, step_video):
                            x_fit = x_ts.frame(k_fit)
                            pred_fit = to_numpy(model.forward_visual(x_fit, k_fit)).squeeze()
                            gt_fit = to_numpy(x_ts.stimulus[k_fit, :sim.n_input_neurons]).squeeze()
                            all_gt.append(gt_fit)
                            all_pred.append(pred_fit)
                        all_gt = np.concatenate(all_gt)
                        all_pred = np.concatenate(all_pred)

                        # Least-squares fit: gt = a * pred + b
                        A_fit = np.vstack([all_pred, np.ones(len(all_pred))]).T
                        a_coeff, b_coeff = np.linalg.lstsq(A_fit, all_gt, rcond=None)[0]
                        logger.info(f"field linear fit: gt = {a_coeff:.4f} * pred + {b_coeff:.4f}")

                        # Compute field_R2 on corrected predictions
                        pred_corrected_all = a_coeff * all_pred + b_coeff
                        ss_res = np.sum((all_gt - pred_corrected_all) ** 2)
                        ss_tot = np.sum((all_gt - np.mean(all_gt)) ** 2)
                        field_R2 = 1 - ss_res / (ss_tot + 1e-16)
                        field_slope = a_coeff
                        logger.info(f"external input R² (corrected): {field_R2:.4f}")

                        # GT value range for consistent color scaling
                        gt_vmin = float(all_gt.min())
                        gt_vmax = float(all_gt.max())

                        with writer.saving(fig, out_path, dpi=200):
                            error_list = []

                            for k in trange(0, 800, step_video, ncols=100):
                                # inputs and predictions
                                x = x_ts.frame(k)
                                pred = to_numpy(model.forward_visual(x, k))
                                pred_vec = np.asarray(pred).squeeze()  # (sim.n_input_neurons,)
                                pred_corrected = a_coeff * pred_vec # + b_coeff  # corrected to GT scale

                                gt_vec = to_numpy(x_ts.stimulus[k, :sim.n_input_neurons]).squeeze()

                                # update rolling traces (store corrected predictions)
                                hist_t.append(k)
                                for i in trace_ids:
                                    hist_gt[i].append(gt_vec[i])
                                    hist_pred[i].append(pred_corrected[i])

                                # draw three panels
                                fig.clf()

                                # RMSE on corrected predictions
                                rmse_frame = float(np.sqrt(((pred_corrected - gt_vec) ** 2).mean()))
                                running_rmse = float(np.mean(error_list + [rmse_frame])) if len(error_list) else rmse_frame

                                # Traces (both on GT scale)
                                ax3 = fig.add_subplot(1, 3, 3)
                                ax3.set_axis_off()
                                ax3.set_facecolor("black")

                                t = np.arange(len(hist_t))
                                for j, i in enumerate(trace_ids):
                                    y0 = j * offset
                                    ax3.plot(t, np.array(hist_gt[i])   + y0, color='lime',  lw=1.6, alpha=0.95)
                                    ax3.plot(t, np.array(hist_pred[i]) + y0, color='k', lw=1.2, alpha=0.95)

                                ax3.set_xlim(max(0, len(t) - win), len(t))
                                ax3.set_ylim(-offset * 0.5, offset * (len(trace_ids) + 0.5))
                                ax3.text(
                                    0.02, 0.98,
                                    f"frame: {k}   RMSE: {rmse_frame:.3f}   avg RMSE: {running_rmse:.3f}   a={a_coeff:.3f} b={b_coeff:.3f}",
                                    transform=ax3.transAxes,
                                    va='top', ha='left',
                                    fontsize=6, color='k')

                                # GT field
                                ax1 = fig.add_subplot(1, 3, 1)
                                ax1.scatter(X1[:, 0], X1[:, 1], s=256, c=gt_vec, cmap=default_style.cmap, marker='h', vmin=gt_vmin, vmax=gt_vmax)
                                ax1.set_axis_off()
                                ax1.set_title('ground truth', fontsize=12)

                                # Predicted field (corrected, same scale as GT)
                                ax2 = fig.add_subplot(1, 3, 2)
                                ax2.scatter(X1[:, 0], X1[:, 1], s=256, c=pred_corrected, cmap=default_style.cmap, marker='h')
                                ax2.set_axis_off()
                                ax2.set_title('prediction (corrected)', fontsize=12)

                                plt.tight_layout()
                                writer.grab_frame()

                                error_list.append(rmse_frame)


                    if last_connectivity_r2 is not None:
                        # color code: green (>0.9), yellow (0.7-0.9), orange (0.3-0.7), red (<0.3)
                        if last_connectivity_r2 > 0.9:
                            r2_color = '\033[92m'  # green
                        elif last_connectivity_r2 > 0.7:
                            r2_color = '\033[93m'  # yellow
                        elif last_connectivity_r2 > 0.3:
                            r2_color = '\033[38;5;208m'  # orange
                        else:
                            r2_color = '\033[91m'  # red
                        pbar.set_postfix_str(f'{r2_color}R²={last_connectivity_r2:.3f}\033[0m')
                    torch.save(
                        {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                        os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}_{N}.pt'))

            # check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50, memory_percentage_threshold=0.6)

        # === LLM-MODIFIABLE: TRAINING LOOP END ===

        # Calculate epoch-level losses
        epoch_total_loss = total_loss / n_neurons
        epoch_regul_loss = total_loss_regul / n_neurons
        epoch_pred_loss = (total_loss - total_loss_regul) / n_neurons

        print("epoch {}. loss: {:.6f} (pred: {:.6f}, regul: {:.6f})".format(
            epoch, epoch_total_loss, epoch_pred_loss, epoch_regul_loss))
        logger.info("Epoch {}. Loss: {:.6f} (pred: {:.6f}, regul: {:.6f})".format(
            epoch, epoch_total_loss, epoch_pred_loss, epoch_regul_loss))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}.pt'))

        list_loss.append(epoch_pred_loss)
        list_loss_regul.append(epoch_regul_loss)

        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(15, 10))

        # Plot 1: Loss
        fig.add_subplot(2, 3, 1)
        plt.plot(list_loss, color=default_style.foreground, linewidth=1)
        plt.xlim([0, tc.n_epochs])
        plt.ylabel('loss', fontsize=12)
        plt.xlabel('epochs', fontsize=12)

        plot_training_summary_panels(fig, log_dir)

        if replace_with_cluster:

            if (epoch % tc.sparsity_freq == tc.sparsity_freq - 1) & (epoch < tc.n_epochs - tc.sparsity_freq):
                print('replace embedding with clusters ...')
                eps = tc.cluster_distance_threshold
                results = clustering_evaluation(to_numpy(model.a), type_list, eps=eps)
                print(f"eps={eps}: {results['n_clusters_found']} clusters, "
                      f"accuracy={results['accuracy']:.3f}")

                labels = results['cluster_labels']

                for n in np.unique(labels):
                    # if n == -1:
                    #     continue
                    indices = np.where(labels == n)[0]
                    if len(indices) > 1:
                        with torch.no_grad():
                            model.a[indices, :] = torch.mean(model.a[indices, :], dim=0, keepdim=True)

                fig.add_subplot(2, 3, 6)
                type_cmap = CustomColorMap(config=config)
                for n in range(sim.n_neuron_types):
                    pos = torch.argwhere(type_list == n)
                    plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=5, color=type_cmap.color(n),
                                alpha=0.7, edgecolors='none')
                plt.xlabel('embedding 0', fontsize=18)
                plt.ylabel('embedding 1', fontsize=18)
                plt.xticks([])
                plt.yticks([])
                plt.text(0.5, 0.9, f"eps={eps}: {results['n_clusters_found']} clusters, accuracy={results['accuracy']:.3f}")

                if tc.fix_cluster_embedding:
                    lr_embedding = 1.0E-10
                    # the embedding is fixed for 1 epoch

            else:
                lr = tc.learning_rate_start
                lr_embedding = tc.learning_rate_embedding_start
                lr_W = tc.learning_rate_W_start
                learning_rate_NNR = tc.learning_rate_NNR

            logger.info(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, learning_rate_NNR {learning_rate_NNR}')
            optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr, lr_update=lr_update, lr_W=lr_W,
                                                                 learning_rate_NNR=learning_rate_NNR)

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/epoch_{epoch}.png")
        plt.close()

    # Calculate and log training time
    training_time = time.time() - training_start_time
    training_time_min = training_time / 60.0
    print(f"training completed in {training_time_min:.1f} minutes")
    logger.info(f"training completed in {training_time_min:.1f} minutes")

    if log_file is not None:
        log_file.write(f"training_time_min: {training_time_min:.1f}\n")
        log_file.write(f"n_epochs: {tc.n_epochs}\n")
        log_file.write(f"data_augmentation_loop: {tc.data_augmentation_loop}\n")
        log_file.write(f"recurrent_training: {tc.recurrent_training}\n")
        log_file.write(f"batch_size: {tc.batch_size}\n")
        log_file.write(f"learning_rate_W: {tc.learning_rate_W_start}\n")
        log_file.write(f"learning_rate: {tc.learning_rate_start}\n")
        log_file.write(f"learning_rate_embedding: {tc.learning_rate_embedding_start}\n")
        log_file.write(f"coeff_edge_diff: {tc.coeff_edge_diff}\n")
        log_file.write(f"coeff_edge_norm: {tc.coeff_edge_norm}\n")
        log_file.write(f"coeff_edge_weight_L1: {tc.coeff_edge_weight_L1}\n")
        log_file.write(f"coeff_phi_weight_L1: {tc.coeff_phi_weight_L1}\n")
        log_file.write(f"coeff_phi_weight_L2: {tc.coeff_phi_weight_L2}\n")
        log_file.write(f"coeff_W_L1: {tc.coeff_W_L1}\n")
        if field_R2 is not None:
            log_file.write(f"field_R2: {field_R2:.4f}\n")
            log_file.write(f"field_slope: {field_slope:.4f}\n")



def data_train_flyvis_RNN(config, erase, best_model, device):
    """RNN training with sequential processing through time"""

    sim = config.simulation
    tc = config.training
    model_config = config.graph_model


    warm_up_length = tc.warm_up_length  # e.g., 10
    sequence_length = tc.sequence_length  # e.g., 32
    total_length = warm_up_length + sequence_length

    seed = config.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    log_dir, logger = create_log_dir(config, erase)

    print(f"Loading data from {config.dataset}...")
    x_list = []
    y_list = []
    for run in trange(0, tc.n_runs, ncols=50):
        x = np.load(f'graphs_data/{config.dataset}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{config.dataset}/y_list_{run}.npy')

        if tc.training_selected_neurons:
            selected_neuron_ids = np.array(tc.selected_neuron_ids).astype(int)
            x = x[:, selected_neuron_ids, :]
            y = y[:, selected_neuron_ids, :]

        x_list.append(x)
        y_list.append(y)

    print(f'dataset: {len(x_list)} runs, {len(x_list[0])} frames')

    # Normalization
    activity = torch.tensor(x_list[0][:, :, 3:4], device=device)
    activity = activity.squeeze()
    distrib = activity.flatten()
    valid_distrib = distrib[~torch.isnan(distrib)]

    if len(valid_distrib) > 0:
        xnorm = 1.5 * torch.std(valid_distrib)
    else:
        xnorm = torch.tensor(1.0, device=device)

    ynorm = torch.tensor(1.0, device=device)
    torch.save(xnorm, os.path.join(log_dir, 'xnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))

    print(f'xnorm: {xnorm.item():.3f}')
    print(f'ynorm: {ynorm.item():.3f}')
    logger.info(f'xnorm: {xnorm.item():.3f}')
    logger.info(f'ynorm: {ynorm.item():.3f}')

    # Create model
    model = create_model(model_config.signal_model_name,
                         aggr_type=model_config.aggr_type, config=config, device=device)
    use_lstm = 'LSTM' in model_config.signal_model_name

    # Count parameters
    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameters: {n_total_params:,}')
    logger.info(f'Total parameters: {n_total_params:,}')

    # Optimizer
    lr = tc.learning_rate_start
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f'learning rate: {lr}')
    logger.info(f'learning rate: {lr}')

    print("starting RNN training...")
    logger.info("Starting RNN training...")

    list_loss = []

    for epoch in range(tc.n_epochs):

        # Number of sequences per epoch
        n_sequences = (sim.n_frames - total_length) // 10 * tc.data_augmentation_loop
        plot_frequency = int(n_sequences // 10) # Sample ~10% of possible sequences
        if epoch == 0:
            print(f'{n_sequences} sequences per epoch, plot every {plot_frequency} sequences')
            logger.info(f'{n_sequences} sequences per epoch, plot every {plot_frequency} sequences')

        total_loss = 0
        model.train()

        for seq_idx in trange(n_sequences, ncols=150, desc=f"Epoch {epoch}"):

            optimizer.zero_grad()

            # Sample random sequence
            run = np.random.randint(tc.n_runs)
            k_start = np.random.randint(0, sim.n_frames - total_length)

            # Initialize hidden state to None (GRU will initialize to zeros)
            h = None
            c = None if use_lstm else None

            # Warm-up phase
            with torch.no_grad():
                for t in range(k_start, k_start + warm_up_length):
                    x = torch.tensor(x_list[run][t], dtype=torch.float32, device=device)
                    if use_lstm:
                        _, h, c = model(x, h=h, c=c, return_all=True)
                    else:
                        _, h = model(x, h=h, return_all=True)

            # Prediction phase (compute loss)
            loss = 0
            for t in range(k_start + warm_up_length, k_start + total_length):
                x = torch.tensor(x_list[run][t], dtype=torch.float32, device=device)
                y_true = torch.tensor(y_list[run][t], dtype=torch.float32, device=device)

                # Forward pass
                if use_lstm:
                    y_pred, h, c = model(x, h=h, c=c, return_all=True)
                else:
                    y_pred, h = model(x, h=h, return_all=True)

                # Accumulate loss
                loss += (y_pred - y_true).norm(2)

                # # Truncated BPTT: detach hidden state
                # h = h.detach()

            # Normalize by sequence length
            loss = loss / sequence_length

            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if (seq_idx % plot_frequency == 0) and (seq_idx > 0):
                # Save intermediate model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs-1}_graphs_{epoch}_{seq_idx}.pt'))

        # Epoch statistics
        avg_loss = total_loss / n_sequences
        print(f"Epoch {epoch}. Loss: {avg_loss:.6f}")
        logger.info(f"Epoch {epoch}. Loss: {avg_loss:.6f}")

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs-1}_graphs_{epoch}.pt'))

        list_loss.append(avg_loss)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        # Learning rate decay
        if (epoch + 1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"Learning rate decreased to {param_group['lr']}")
            logger.info(f"Learning rate decreased to {param_group['lr']}")


def data_train_INR(config=None, device=None, total_steps=5000, erase=False):
    """
    Train nnr_f (SIREN/INR network) on external_input data from x_list.

    This pre-trains the implicit neural representation (INR) network before
    joint learning with GNN. The INR learns to map time -> external_input
    for all neurons.

    INR types:
        siren_t: input=t, output=n_neurons (works for n_neurons < 100)
        siren_id: input=(t, id/n_neurons), output=1 (scales better for large n_neurons)
        siren_x: input=(t, x, y), output=1 (uses neuron positions)
        ngp: instantNGP hash encoding

    Args:
        config: NeuralGraphConfig object
        device: torch device
        total_steps: number of training steps (default: 5000)
        erase: whether to erase existing log files (default: False)

    Returns:
        nnr_f: trained SIREN model
        loss_list: list of training losses
    """

    # create log directory
    log_dir, logger = create_log_dir(config, erase)
    output_folder = os.path.join(log_dir, 'tmp_training', 'external_input')
    os.makedirs(output_folder, exist_ok=True)

    dataset_name = config.dataset
    data_folder = f"graphs_data/{dataset_name}/"
    print(f"loading data from: {data_folder}")

    # load x_list data
    x_list = np.load(f"{data_folder}x_list_0.npy")
    print(f"x_list shape: {x_list.shape}")  # (n_frames, n_neurons, n_features)

    n_frames, n_neurons, n_features = x_list.shape
    print(f"n_frames: {n_frames}, n_neurons: {n_neurons}, n_features: {n_features}")

    # extract external_input from x_list (column 4)
    external_input = x_list[:, :, 4]  # shape: (n_frames, n_neurons)

    # SVD analysis
    U, S, Vt = np.linalg.svd(external_input, full_matrices=False)

    # effective rank
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    rank_90 = np.searchsorted(cumvar, 0.90) + 1
    rank_99 = np.searchsorted(cumvar, 0.99) + 1

    print(f"effective rank (90% var): {rank_90}")
    print(f"effective rank (99% var): {rank_99}")
    print(f"compression: {n_frames * n_neurons / (rank_99 * (n_frames + n_neurons)):.1f}x")
    print()

    # extract neuron positions from x_list (columns 1, 2) - use first frame as reference
    neuron_positions = x_list[0, :, 1:3]  # shape: (n_neurons, 2)


    # extract neuron ids from x_list (column 0)
    neuron_ids = x_list[0, :, 0]  # shape: (n_neurons,)

    # get nnr_f config parameters
    sim = config.simulation
    model_config = config.graph_model
    hidden_dim_nnr_f = getattr(model_config, 'hidden_dim_nnr_f', 1024)
    n_layers_nnr_f = getattr(model_config, 'n_layers_nnr_f', 3)
    outermost_linear_nnr_f = getattr(model_config, 'outermost_linear_nnr_f', True)
    omega_f = getattr(model_config, 'omega_f', 1024)
    nnr_f_T_period = getattr(model_config, 'nnr_f_T_period', 10000)

    # get training config parameters
    tc = config.training
    batch_size = getattr(tc, 'batch_size', 8)
    learning_rate = getattr(tc, 'learning_rate_NNR_f', 1e-6)

    # get simulation config for calculation check
    delta_t = getattr(sim, 'delta_t', 0.01)
    oscillation_frequency = getattr(sim, 'oscillation_frequency', 0.1)

    # calculation check
    total_sim_time = n_frames * delta_t
    period_time_units = 1.0 / oscillation_frequency if oscillation_frequency > 0 else float('inf')
    period_frames = period_time_units / delta_t if oscillation_frequency > 0 else float('inf')
    total_cycles = total_sim_time / period_time_units if oscillation_frequency > 0 else 0
    normalized_time_max = n_frames / nnr_f_T_period
    cycles_in_normalized_range = total_cycles * normalized_time_max
    recommended_omega = 2 * np.pi * cycles_in_normalized_range

    # get INR type from config
    inr_type = getattr(model_config, 'inr_type', 'siren_t')

    print("siren calculation check:")
    print(f"  total simulation time: {n_frames} × {delta_t} = {total_sim_time:.1f} time units")
    print(f"  period: 1/{oscillation_frequency} = {period_time_units:.1f} time units = {period_frames:.0f} frames")
    print(f"  total cycles: {total_cycles:.0f}")
    print(f"  normalized input range: [0, {n_frames}/{nnr_f_T_period}] = [0, {normalized_time_max:.2f}]")
    print(f"  cycles in normalized range: {total_cycles:.0f} × {normalized_time_max:.2f} = {cycles_in_normalized_range:.1f}")
    print(f"  recommended omega_f: 2π × {cycles_in_normalized_range:.1f} ≈ {recommended_omega:.0f}")
    print(f"  omega_f (config): {omega_f}")
    if omega_f > 5 * recommended_omega:
        print(f"  ⚠️  omega_f is {omega_f/recommended_omega:.1f}× recommended — may cause slow convergence")

    # data dimensions to learn
    data_dims = n_frames * n_neurons
    print(f"\ndata to learn: {n_frames:,} frames × {n_neurons:,} neurons = {data_dims:,.0f} values")

    # determine input/output dimensions based on inr_type
    if inr_type == 'siren_t':
        input_size_nnr_f = 1
        output_size_nnr_f = n_neurons
    elif inr_type == 'siren_id':
        input_size_nnr_f = 2  # (t, id)
        output_size_nnr_f = 1
    elif inr_type == 'siren_x':
        input_size_nnr_f = 3  # (t, x, y)
        output_size_nnr_f = 1
    elif inr_type == 'ngp':
        input_size_nnr_f = getattr(model_config, 'input_size_nnr_f', 1)
        output_size_nnr_f = getattr(model_config, 'output_size_nnr_f', n_neurons)
    elif inr_type == 'lowrank':
        # lowrank doesn't use input/output sizes in the same way
        pass
    else:
        raise ValueError(f"unknown inr_type: {inr_type}")

    # create INR model based on type
    if inr_type == 'ngp':

        # get NGP config parameters
        ngp_n_levels = getattr(model_config, 'ngp_n_levels', 24)
        ngp_n_features_per_level = getattr(model_config, 'ngp_n_features_per_level', 2)
        ngp_log2_hashmap_size = getattr(model_config, 'ngp_log2_hashmap_size', 22)
        ngp_base_resolution = getattr(model_config, 'ngp_base_resolution', 16)
        ngp_per_level_scale = getattr(model_config, 'ngp_per_level_scale', 1.4)
        ngp_n_neurons = getattr(model_config, 'ngp_n_neurons', 128)
        ngp_n_hidden_layers = getattr(model_config, 'ngp_n_hidden_layers', 4)

        nnr_f = HashEncodingMLP(
            n_input_dims=input_size_nnr_f,
            n_output_dims=output_size_nnr_f,
            n_levels=ngp_n_levels,
            n_features_per_level=ngp_n_features_per_level,
            log2_hashmap_size=ngp_log2_hashmap_size,
            base_resolution=ngp_base_resolution,
            per_level_scale=ngp_per_level_scale,
            n_neurons=ngp_n_neurons,
            n_hidden_layers=ngp_n_hidden_layers,
            output_activation='none'
        )
        nnr_f = nnr_f.to(device)

        # count parameters
        encoding_params = sum(p.numel() for p in nnr_f.encoding.parameters())
        mlp_params = sum(p.numel() for p in nnr_f.mlp.parameters())
        total_params = encoding_params + mlp_params

        print("\nusing HashEncodingMLP (instantNGP):")
        print(f"  hash encoding: {ngp_n_levels} levels × {ngp_n_features_per_level} features")
        print(f"  hash table: 2^{ngp_log2_hashmap_size} = {2**ngp_log2_hashmap_size:,} entries")
        print(f"  mlp: {ngp_n_neurons} × {ngp_n_hidden_layers} hidden → {output_size_nnr_f}")
        print(f"  parameters: {total_params:,} (encoding: {encoding_params:,}, mlp: {mlp_params:,})")
        print(f"  compression ratio: {data_dims / total_params:.2f}x")

    elif inr_type in ['siren_t', 'siren_id', 'siren_x']:
        # create SIREN model for nnr_f
        omega_f_learning = getattr(model_config, 'omega_f_learning', False)
        nnr_f = Siren(
            in_features=input_size_nnr_f,
            hidden_features=hidden_dim_nnr_f,
            hidden_layers=n_layers_nnr_f,
            out_features=output_size_nnr_f,
            outermost_linear=outermost_linear_nnr_f,
            first_omega_0=omega_f,
            hidden_omega_0=omega_f,
            learnable_omega=omega_f_learning
        )
        nnr_f = nnr_f.to(device)

        # count parameters
        total_params = sum(p.numel() for p in nnr_f.parameters())

        print(f"\nusing SIREN ({inr_type}):")
        print(f"  architecture: {input_size_nnr_f} → {hidden_dim_nnr_f} × {n_layers_nnr_f} hidden → {output_size_nnr_f}")
        print(f"  omega_f: {omega_f} (learnable: {omega_f_learning})")
        if omega_f_learning and hasattr(nnr_f, 'get_omegas'):
            print(f"  initial omegas: {nnr_f.get_omegas()}")
        print(f"  parameters: {total_params:,}")
        print(f"  compression ratio: {data_dims / total_params:.2f}x")

    elif inr_type == 'lowrank':
        # get lowrank config parameters
        lowrank_rank = getattr(model_config, 'lowrank_rank', 64)
        lowrank_svd_init = getattr(model_config, 'lowrank_svd_init', True)

        # create LowRankINR model
        init_data = external_input if lowrank_svd_init else None
        nnr_f = LowRankINR(
            n_frames=n_frames,
            n_neurons=n_neurons,
            rank=lowrank_rank,
            init_data=init_data
        )
        nnr_f = nnr_f.to(device)

        # count parameters
        total_params = sum(p.numel() for p in nnr_f.parameters())

        print("\nusing LowRankINR:")
        print(f"  rank: {lowrank_rank}")
        print(f"  U: ({n_frames}, {lowrank_rank}), V: ({lowrank_rank}, {n_neurons})")
        print(f"  parameters: {total_params:,}")
        print(f"  compression ratio: {data_dims / total_params:.2f}x")
        print(f"  SVD init: {lowrank_svd_init}")

    print(f"\ntraining: batch_size={batch_size}, learning_rate={learning_rate}")

    # prepare training data
    ground_truth = torch.tensor(external_input, dtype=torch.float32, device=device)  # (n_frames, n_neurons)

    # prepare inputs based on inr_type
    if inr_type == 'siren_t':
        # input: normalized time, output: all neurons
        time_input = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / nnr_f_T_period

    elif inr_type == 'siren_id':
        # input: (t, id), output: 1
        # normalize id by n_neurons
        neuron_ids_norm = torch.tensor(neuron_ids / n_neurons, dtype=torch.float32, device=device)  # (n_neurons,)

    elif inr_type == 'siren_x':
        # input: (t, x, y), output: 1
        # positions are already normalized
        neuron_pos = torch.tensor(neuron_positions, dtype=torch.float32, device=device)  # (n_neurons, 2)

    steps_til_summary = 5000

    # Separate omega parameters from other parameters for different learning rates
    omega_f_learning = getattr(model_config, 'omega_f_learning', False)
    learning_rate_omega_f = getattr(tc, 'learning_rate_omega_f', learning_rate)
    omega_params = [p for name, p in nnr_f.named_parameters() if 'omega' in name]
    other_params = [p for name, p in nnr_f.named_parameters() if 'omega' not in name]
    if omega_params and omega_f_learning:
        optim = torch.optim.Adam([
            {'params': other_params, 'lr': learning_rate},
            {'params': omega_params, 'lr': learning_rate_omega_f}
        ])
        print(f"using separate learning rates: network={learning_rate}, omega={learning_rate_omega_f}")
    else:
        optim = torch.optim.Adam(lr=learning_rate, params=nnr_f.parameters())

    print(f"training nnr_f for {total_steps} steps...")

    loss_list = []
    pbar = trange(total_steps + 1, ncols=150)
    for step in pbar:

        if inr_type == 'siren_t':
            # sample batch_size time frames
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            time_batch = time_input[sample_ids]  # (batch_size, 1)
            gt_batch = ground_truth[sample_ids]  # (batch_size, n_neurons)
            pred = nnr_f(time_batch)  # (batch_size, n_neurons)

        elif inr_type == 'siren_id':
            # sample batch_size time frames, predict all neurons for each frame
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            t_norm = torch.tensor(sample_ids / nnr_f_T_period, dtype=torch.float32, device=device)  # (batch_size,)
            # expand to all neurons: (batch_size, n_neurons, 2)
            t_expanded = t_norm[:, None, None].expand(batch_size, n_neurons, 1)
            id_expanded = neuron_ids_norm[None, :, None].expand(batch_size, n_neurons, 1)
            input_batch = torch.cat([t_expanded, id_expanded], dim=2)  # (batch_size, n_neurons, 2)
            input_batch = input_batch.reshape(batch_size * n_neurons, 2)  # (batch_size * n_neurons, 2)
            gt_batch = ground_truth[sample_ids].reshape(batch_size * n_neurons)  # (batch_size * n_neurons,)
            pred = nnr_f(input_batch).squeeze()  # (batch_size * n_neurons,)

        elif inr_type == 'siren_x':
            # sample batch_size time frames, predict all neurons for each frame
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            t_norm = torch.tensor(sample_ids / nnr_f_T_period, dtype=torch.float32, device=device)  # (batch_size,)
            # expand to all neurons: (batch_size, n_neurons, 3)
            t_expanded = t_norm[:, None, None].expand(batch_size, n_neurons, 1)
            pos_expanded = neuron_pos[None, :, :].expand(batch_size, n_neurons, 2)
            input_batch = torch.cat([t_expanded, pos_expanded], dim=2)  # (batch_size, n_neurons, 3)
            input_batch = input_batch.reshape(batch_size * n_neurons, 3)  # (batch_size * n_neurons, 3)
            gt_batch = ground_truth[sample_ids].reshape(batch_size * n_neurons)  # (batch_size * n_neurons,)
            pred = nnr_f(input_batch).squeeze()  # (batch_size * n_neurons,)

        elif inr_type == 'ngp':
            # sample batch_size time frames (same as siren_t)
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            time_batch = torch.tensor(sample_ids / nnr_f_T_period, dtype=torch.float32, device=device).unsqueeze(1)
            gt_batch = ground_truth[sample_ids]
            pred = nnr_f(time_batch)

        elif inr_type == 'lowrank':
            # sample batch_size time frames
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            t_indices = torch.tensor(sample_ids, dtype=torch.long, device=device)
            gt_batch = ground_truth[sample_ids]  # (batch_size, n_neurons)
            pred = nnr_f(t_indices)  # (batch_size, n_neurons)

        # compute loss
        if inr_type == 'ngp':
            # relative L2 error - convert targets to match output dtype (tcnn uses float16)
            relative_l2_error = (pred - gt_batch.to(pred.dtype)) ** 2 / (pred.detach() ** 2 + 0.01)
            loss = relative_l2_error.mean()
        else:
            # standard MSE for SIREN
            loss = ((pred - gt_batch) ** 2).mean()

        # omega L2 regularization for learnable omega in SIREN (encourages smaller omega)
        coeff_omega_f_L2 = getattr(tc, 'coeff_omega_f_L2', 0.0)
        if omega_f_learning and coeff_omega_f_L2 > 0 and hasattr(nnr_f, 'get_omega_L2_loss'):
            omega_L2_loss = nnr_f.get_omega_L2_loss()
            loss = loss + coeff_omega_f_L2 * omega_L2_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_list.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.6f}")

        if step % steps_til_summary == 0:
            with torch.no_grad():
                # compute predictions for all frames
                if inr_type == 'siren_t':
                    pred_all = nnr_f(time_input)  # (n_frames, n_neurons)

                elif inr_type == 'siren_id':
                    # predict all (t, id) combinations
                    pred_list = []
                    for t_idx in range(n_frames):
                        t_val = torch.full((n_neurons, 1), t_idx / nnr_f_T_period, device=device)
                        input_t = torch.cat([t_val, neuron_ids_norm[:, None]], dim=1)  # (n_neurons, 2)
                        pred_t = nnr_f(input_t).squeeze()  # (n_neurons,)
                        pred_list.append(pred_t)
                    pred_all = torch.stack(pred_list, dim=0)  # (n_frames, n_neurons)

                elif inr_type == 'siren_x':
                    # predict all (t, x, y) combinations
                    pred_list = []
                    for t_idx in range(n_frames):
                        t_val = torch.full((n_neurons, 1), t_idx / nnr_f_T_period, device=device)
                        input_t = torch.cat([t_val, neuron_pos], dim=1)  # (n_neurons, 3)
                        pred_t = nnr_f(input_t).squeeze()  # (n_neurons,)
                        pred_list.append(pred_t)
                    pred_all = torch.stack(pred_list, dim=0)  # (n_frames, n_neurons)

                elif inr_type == 'ngp':
                    time_all = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / nnr_f_T_period
                    pred_all = nnr_f(time_all)

                elif inr_type == 'lowrank':
                    pred_all = nnr_f()  # returns full (n_frames, n_neurons) matrix

                gt_np = ground_truth.cpu().numpy()
                pred_np = pred_all.cpu().numpy()

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                fig.patch.set_facecolor('black')

                # loss plot
                axes[0].set_facecolor('black')
                axes[0].plot(loss_list, color='white', lw=0.1)
                axes[0].set_xlabel('step', color='white', fontsize=12)
                loss_label = 'Relative L2 Loss' if inr_type == 'ngp' else 'MSE Loss'
                axes[0].set_ylabel(loss_label, color='white', fontsize=12)
                axes[0].set_yscale('log')
                axes[0].tick_params(colors='white', labelsize=11)
                for spine in axes[0].spines.values():
                    spine.set_color('white')

                # traces plot (10 neurons, darkgreen=GT, white=pred)
                axes[1].set_facecolor('black')
                axes[1].set_axis_off()
                n_traces = 10
                trace_ids = np.linspace(0, n_neurons - 1, n_traces, dtype=int)
                offset = np.abs(gt_np).max() * 1.5
                t = np.arange(n_frames)

                for j, n_idx in enumerate(trace_ids):
                    y0 = j * offset
                    axes[1].plot(t, gt_np[:, n_idx] + y0, color='darkgreen', lw=2.0, alpha=0.95)
                    axes[1].plot(t, pred_np[:, n_idx] + y0, color='white', lw=0.5, alpha=0.95)

                axes[1].set_xlim(0, min(20000, n_frames))
                axes[1].set_ylim(-offset * 0.5, offset * (n_traces + 0.5))
                mse = ((pred_np - gt_np) ** 2).mean()
                omega_str = ''
                if hasattr(nnr_f, 'get_omegas'):
                    omegas = nnr_f.get_omegas()
                    if omegas:
                        omega_str = f'  ω: {omegas[0]:.1f}'
                axes[1].text(0.02, 0.98, f'MSE: {mse:.6f}{omega_str}',
                            transform=axes[1].transAxes, va='top', ha='left',
                            fontsize=12, color='white')

                plt.tight_layout()
                plt.savefig(f"{output_folder}/{inr_type}_{step}.png", dpi=150)
                plt.close()

    # save trained model
    # save_path = f"{output_folder}/nnr_f_{inr_type}_pretrained.pt"
    # torch.save(nnr_f.state_dict(), save_path)
    # print(f"\nsaved pretrained nnr_f to: {save_path}")

    # compute final MSE
    with torch.no_grad():
        if inr_type == 'siren_t':
            pred_all = nnr_f(time_input)
        elif inr_type == 'siren_id':
            pred_list = []
            for t_idx in range(n_frames):
                t_val = torch.full((n_neurons, 1), t_idx / nnr_f_T_period, device=device)
                input_t = torch.cat([t_val, neuron_ids_norm[:, None]], dim=1)
                pred_t = nnr_f(input_t).squeeze()
                pred_list.append(pred_t)
            pred_all = torch.stack(pred_list, dim=0)
        elif inr_type == 'siren_x':
            pred_list = []
            for t_idx in range(n_frames):
                t_val = torch.full((n_neurons, 1), t_idx / nnr_f_T_period, device=device)
                input_t = torch.cat([t_val, neuron_pos], dim=1)
                pred_t = nnr_f(input_t).squeeze()
                pred_list.append(pred_t)
            pred_all = torch.stack(pred_list, dim=0)
        elif inr_type == 'ngp':
            time_all = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / nnr_f_T_period
            pred_all = nnr_f(time_all)

        final_mse = ((pred_all - ground_truth) ** 2).mean().item()
        print(f"final MSE: {final_mse:.6f}")
        if hasattr(nnr_f, 'get_omegas'):
            print(f"final omegas: {nnr_f.get_omegas()}")

    return nnr_f, loss_list


def data_test(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20, step=15, n_rollout_frames=600,
              ratio=1, run=0, test_mode='', sample_embedding=False, particle_of_interest=1, new_params = None, device=[],
              rollout_without_noise: bool = False, log_file=None):

    dataset_name = config.dataset
    print(f"\033[94mdataset_name: {dataset_name}\033[0m")
    print(f"\033[92m{config.description}\033[0m")

    if test_mode == "":
        test_mode = "test_ablation_0"

    if 'fly' in config.dataset:
        data_test_flyvis(
            config,
            visualize,
            style,
            verbose,
            best_model,
            step,
            n_rollout_frames,
            test_mode,
            new_params,
            device,
            rollout_without_noise=rollout_without_noise,
            log_file=log_file,
        )
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset}")


def data_test_flyvis(
        config,
        visualize=True,
        style="color",
        verbose=False,
        best_model=None,
        step=5,
        n_rollout_frames=600,
        test_mode='',
        new_params=None,
        device=None,
        rollout_without_noise: bool = False,
        log_file=None,
):


    if "black" in style:
        plt.style.use("dark_background")
        mc = 'white'
    else:
        plt.style.use("default")
        mc = 'black'

    sim = config.simulation
    tc = config.training
    model_config = config.graph_model

    log_dir = 'log/' + config.config_file

    torch.random.fork_rng(devices=device)
    if sim.seed is not None:
        torch.random.manual_seed(sim.seed)
        np.random.seed(sim.seed)

    print(
        f"testing... {model_config.particle_model_name} {model_config.mesh_model_name} seed: {sim.seed}")


    if tc.training_selected_neurons:
        n_neurons = 13741
        n_neuron_types = 1736
    else:
        n_neurons = sim.n_neurons
        n_neuron_types = sim.n_neuron_types

    print(f"noise_model_level: {sim.noise_model_level}")
    warm_up_length = 100

    run = 0

    extent = 8
    # Import only what's needed for mixed functionality
    from flyvis.datasets.sintel import AugmentedSintel
    import flyvis
    from flyvis import NetworkView, Network
    from flyvis.utils.config_utils import get_default_config, CONFIG_PATH
    from flyvis_gnn.generators.flyvis_ode import FlyVisODE, get_photoreceptor_positions_from_net, \
        group_by_direction_and_function
    # Initialize datasets
    if "DAVIS" in sim.visual_input_type or "mixed" in sim.visual_input_type:
        # determine dataset roots: use config list if provided, otherwise fall back to default
        if sim.datavis_roots:
            datavis_root_list = [os.path.join(r, "JPEGImages/480p") for r in sim.datavis_roots]
        else:
            datavis_root_list = [os.path.join(get_datavis_root_dir(), "JPEGImages/480p")]

        for root in datavis_root_list:
            assert os.path.exists(root), f"video data not found at {root}"

        video_config = {
            "n_frames": 50,
            "max_frames": 80,
            "flip_axes": [0, 1],
            "n_rotations": [0, 90, 180, 270],
            "temporal_split": True,
            "dt": sim.delta_t,
            "interpolate": True,
            "boxfilter": dict(extent=extent, kernel_size=13),
            "vertical_splits": 1,
            "center_crop_fraction": 0.6,
            "augment": False,
            "unittest": False,
            "shuffle_sequences": True,
            "shuffle_seed": sim.seed,
        }

        # create dataset(s)
        if len(datavis_root_list) == 1:
            davis_dataset = AugmentedVideoDataset(root_dir=datavis_root_list[0], **video_config)
        else:
            datasets = [AugmentedVideoDataset(root_dir=root, **video_config) for root in datavis_root_list]
            davis_dataset = CombinedVideoDataset(datasets)
            print(f"combined {len(datasets)} video datasets: {len(davis_dataset)} total sequences")
    else:
        davis_dataset = None

    if "DAVIS" in sim.visual_input_type:
        stimulus_dataset = davis_dataset
    else:
        sintel_config = {
            "sintel_path": flyvis.sintel_dir,
            "n_frames": 19,
            "flip_axes": [0, 1],
            "n_rotations": [0, 1, 2, 3, 4, 5],
            "temporal_split": True,
            "dt": sim.delta_t,
            "interpolate": True,
            "boxfilter": dict(extent=extent, kernel_size=13),
            "vertical_splits": 3,
            "center_crop_fraction": 0.7
        }
        stimulus_dataset = AugmentedSintel(**sintel_config)

    # Initialize network
    config_net = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
    config_net.connectome.extent = extent
    net = Network(**config_net)
    nnv = NetworkView(f"flow/{sim.ensemble_id}/{sim.model_id}")
    trained_net = nnv.init_network(checkpoint=0)
    net.load_state_dict(trained_net.state_dict())
    torch.set_grad_enabled(False)

    params = net._param_api()
    p = {"tau_i": params.nodes.time_const, "V_i_rest": params.nodes.bias,
         "w": params.edges.syn_strength * params.edges.syn_count * params.edges.sign}
    edge_index = torch.stack(
        [torch.tensor(net.connectome.edges.source_index[:]), torch.tensor(net.connectome.edges.target_index[:])],
        dim=0).to(device)

    if sim.n_extra_null_edges > 0:
        print(f"adding {sim.n_extra_null_edges} extra null edges...")
        existing_edges = set(zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()))
        import random
        extra_edges = []
        max_attempts = sim.n_extra_null_edges * 10
        attempts = 0
        while len(extra_edges) < sim.n_extra_null_edges and attempts < max_attempts:
            source = random.randint(0, n_neurons - 1)
            target = random.randint(0, n_neurons - 1)
            if (source, target) not in existing_edges and source != target:
                extra_edges.append([source, target])
            attempts += 1
        if extra_edges:
            extra_edge_index = torch.tensor(extra_edges, dtype=torch.long, device=device).t()
            edge_index = torch.cat([edge_index, extra_edge_index], dim=1)
            p["w"] = torch.cat([p["w"], torch.zeros(len(extra_edges), device=device)])

    pde = FlyVisODE(p=p, f=torch.nn.functional.relu, params=sim.params, model_type=model_config.signal_model_name, n_neuron_types=n_neuron_types, device=device)
    pde_modified = FlyVisODE(p=copy.deepcopy(p), f=torch.nn.functional.relu, params=sim.params, model_type=model_config.signal_model_name, n_neuron_types=n_neuron_types, device=device)


    model = create_model(model_config.signal_model_name,
                         aggr_type=model_config.aggr_type, config=config, device=device)


    if best_model == 'best':
        files = glob.glob(f"{log_dir}/models/*")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]
        best_model = filename
        print(f'best model: {best_model}')
    netname = f"{log_dir}/models/best_model_with_0_graphs_{best_model}.pt"
    print(f'load {netname} ...')
    state_dict = torch.load(netname, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

    x_coords, y_coords, u_coords, v_coords = get_photoreceptor_positions_from_net(net)

    node_types = np.array(net.connectome.nodes["type"])
    node_types_str = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in node_types]
    grouped_types = np.array([group_by_direction_and_function(t) for t in node_types_str])
    unique_types, node_types_int = np.unique(node_types, return_inverse=True)

    X1 = torch.tensor(np.stack((x_coords, y_coords), axis=1), dtype=torch.float32, device=device)

    from flyvis_gnn.generators.utils import get_equidistant_points
    xc, yc = get_equidistant_points(n_points=n_neurons - x_coords.shape[0])
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    X1 = torch.cat((X1, pos[torch.randperm(pos.size(0), device=device)]), dim=0)

    state = net.steady_state(t_pre=2.0, dt=sim.delta_t, batch_size=1)
    initial_state = state.nodes.activity.squeeze()
    n_neurons = len(initial_state)

    sequences = stimulus_dataset[0]["lum"]
    frame = sequences[0][None, None]
    net.stimulus.add_input(frame)

    calcium_init = torch.rand(n_neurons, dtype=torch.float32, device=device)
    x = NeuronState(
        index=torch.arange(n_neurons, dtype=torch.long, device=device),
        pos=X1,
        group_type=torch.tensor(grouped_types, dtype=torch.long, device=device),
        neuron_type=torch.tensor(node_types_int, dtype=torch.long, device=device),
        voltage=initial_state,
        stimulus=net.stimulus().squeeze(),
        calcium=calcium_init,
        fluorescence=sim.calcium_alpha * calcium_init + sim.calcium_beta,
    )

    if tc.training_selected_neurons:
        selected_neuron_ids = tc.selected_neuron_ids
        selected_neuron_ids = np.array(selected_neuron_ids).astype(int)
        print(f'testing single neuron id {selected_neuron_ids} ...')
        x_selected = x.subset(selected_neuron_ids)

    # Mixed sequence setup
    if "mixed" in sim.visual_input_type:
        mixed_types = ["sintel", "davis", "blank", "noise"]
        mixed_cycle_lengths = [60, 60, 30, 60]  # Different lengths for each type
        mixed_current_type = 0
        mixed_frame_count = 0
        current_cycle_length = mixed_cycle_lengths[mixed_current_type]
        if not davis_dataset:
            sintel_config_mixed = {
                "n_frames": 19,
                "flip_axes": [0, 1],
                "n_rotations": [0, 1, 2, 3, 4, 5],
                "temporal_split": True,
                "dt": sim.delta_t,
                "interpolate": True,
                "boxfilter": dict(extent=extent, kernel_size=13),
                "vertical_splits": 3,
                "center_crop_fraction": 0.7
            }
            davis_dataset = AugmentedSintel(**sintel_config_mixed)
        sintel_iter = iter(stimulus_dataset)
        davis_iter = iter(davis_dataset)
        current_sintel_seq = None
        current_davis_seq = None
        sintel_frame_idx = 0
        davis_frame_idx = 0

    target_frames = n_rollout_frames

    if 'full' in test_mode:
        target_frames = sim.n_frames
        step = 25000
    else:
        step = 10
    print(f'plot activity frames \033[92m0-{target_frames}...\033[0m')

    dataset_length = len(stimulus_dataset)
    frames_per_sequence = 35
    total_frames_per_pass = dataset_length * frames_per_sequence
    num_passes_needed = (target_frames // total_frames_per_pass) + 1

    y_list = []
    x_list = []
    x_generated_list = []
    x_generated_modified_list = []

    x_generated = x.clone()
    x_generated_modified = x.clone()

    # Initialize RNN hidden state
    if 'RNN' in model_config.signal_model_name:
        h_state = None
    if 'LSTM' in model_config.signal_model_name:
        h_state = None
        c_state = None

    it = sim.start_frame
    id_fig = 0

    tile_labels = None
    tile_codes_torch = None
    tile_period = None
    tile_idx = 0
    tile_contrast = sim.tile_contrast
    n_columns = sim.n_input_neurons // 8
    tile_seed = sim.seed

    edges = torch.load(f'./graphs_data/{config.dataset}/edge_index.pt', map_location=device)

    if ('test_ablation' in test_mode) & (not('MLP' in model_config.signal_model_name)) & (not('RNN' in model_config.signal_model_name)) & (not('LSTM' in model_config.signal_model_name)):
        #  test_mode="test_ablation_100"
        ablation_ratio = int(test_mode.split('_')[-1]) / 100
        if ablation_ratio > 0:
            print(f'\033[93mtest ablation ratio {ablation_ratio} \033[0m')
        n_ablation = int(edges.shape[1] * ablation_ratio)
        index_ablation = np.random.choice(np.arange(edges.shape[1]), n_ablation, replace=False)

        with torch.no_grad():
            pde.p['w'][index_ablation] = 0
            pde_modified.p['w'][index_ablation] = 0
            model.W[index_ablation] = 0

    if 'test_modified' in test_mode:
        noise_W = float(test_mode.split('_')[-1])
        if noise_W > 0:
            print(f'\033[93mtest modified W with noise level {noise_W}\033[0m')
            noise_p_W = torch.randn_like(pde.p['w']) * noise_W # + torch.ones_like(pde.p['w'])
            pde_modified.p['w'] = pde.p['w'].clone() + noise_p_W

        plot_weight_comparison(pde.p['w'], pde_modified.p['w'], f"./{log_dir}/results/weight_comparison_{noise_W}.png")


    fig_style = dark_style
    index_to_name = INDEX_TO_NAME


    # Main loop #####################################

    with torch.no_grad():
        for pass_num in range(num_passes_needed):
            for data_idx, data in enumerate(tqdm(stimulus_dataset, desc="processing stimulus data", ncols=100)):

                sequences = data["lum"]
                # Sample flash parameters for each subsequence if flash stimulus is requested
                if "flash" in sim.visual_input_type:
                    # Sample flash duration from specific values: 1, 2, 5, 10, 20 frames
                    flash_duration_options = [1, 2, 5] #, 10, 20]
                    flash_cycle_frames = flash_duration_options[
                        torch.randint(0, len(flash_duration_options), (1,), device=device).item()
                    ]

                    flash_intensity = torch.abs(torch.rand(sim.n_input_neurons, device=device) * 0.5 + 0.5)
                if "mixed" in sim.visual_input_type:
                    if mixed_frame_count >= current_cycle_length:
                        mixed_current_type = (mixed_current_type + 1) % 4
                        mixed_frame_count = 0
                        current_cycle_length = mixed_cycle_lengths[mixed_current_type]
                    current_type = mixed_types[mixed_current_type]

                    if current_type == "sintel":
                        if current_sintel_seq is None or sintel_frame_idx >= current_sintel_seq["lum"].shape[0]:
                            try:
                                current_sintel_seq = next(sintel_iter)
                                sintel_frame_idx = 0
                            except StopIteration:
                                sintel_iter = iter(stimulus_dataset)
                                current_sintel_seq = next(sintel_iter)
                                sintel_frame_idx = 0
                        sequences = current_sintel_seq["lum"]
                        start_frame = sintel_frame_idx
                    elif current_type == "davis":
                        if current_davis_seq is None or davis_frame_idx >= current_davis_seq["lum"].shape[0]:
                            try:
                                current_davis_seq = next(davis_iter)
                                davis_frame_idx = 0
                            except StopIteration:
                                davis_iter = iter(davis_dataset)
                                current_davis_seq = next(davis_iter)
                                davis_frame_idx = 0
                        sequences = current_davis_seq["lum"]
                        start_frame = davis_frame_idx
                    else:
                        start_frame = 0
                # Determine sequence length based on stimulus type
                if "flash" in sim.visual_input_type:
                    sequence_length = 60  # Fixed 60 frames for flash sequences
                else:
                    sequence_length = sequences.shape[0]

                for frame_id in range(sequence_length):

                    if "flash" in sim.visual_input_type:
                        # Generate repeating flash stimulus
                        current_flash_frame = frame_id % (flash_cycle_frames * 2)  # Create on/off cycle
                        x.stimulus[:] = 0
                        if current_flash_frame < flash_cycle_frames:
                            x.stimulus[:sim.n_input_neurons] = flash_intensity
                    elif "mixed" in sim.visual_input_type:
                        current_type = mixed_types[mixed_current_type]

                        if current_type == "blank":
                            x.stimulus[:] = 0
                        elif current_type == "noise":
                            x.stimulus[:sim.n_input_neurons] = torch.relu(
                                0.5 + torch.rand(sim.n_input_neurons, dtype=torch.float32, device=device) * 0.5)
                        else:
                            actual_frame_id = (start_frame + frame_id) % sequences.shape[0]
                            frame = sequences[actual_frame_id][None, None]
                            net.stimulus.add_input(frame)
                            x.stimulus = net.stimulus().squeeze()
                            if current_type == "sintel":
                                sintel_frame_idx += 1
                            elif current_type == "davis":
                                davis_frame_idx += 1
                        mixed_frame_count += 1
                    elif "tile_mseq" in sim.visual_input_type:
                        if tile_codes_torch is None:
                            # 1) Cluster photoreceptors into columns based on (u,v)
                            tile_labels_np = assign_columns_from_uv(
                                u_coords, v_coords, n_columns, random_state=tile_seed
                            )  # shape: (sim.n_input_neurons,)

                            # 2) Build per-column m-sequences (±1) with random phase per column
                            base = mseq_bits(p=8, seed=tile_seed).astype(np.float32)  # ±1, shape (255,)
                            rng = np.random.RandomState(tile_seed)
                            phases = rng.randint(0, base.shape[0], size=n_columns)
                            tile_codes_np = np.stack([np.roll(base, ph) for ph in phases], axis=0)  # (n_columns, 255), ±1

                            # 3) Convert to torch on the right device/dtype; keep as ±1 (no [0,1] mapping here)
                            tile_codes_torch = torch.from_numpy(tile_codes_np).to(x.device,
                                                                                  dtype=torch.float32)  # (n_columns, 255), ±1
                            tile_labels = torch.from_numpy(tile_labels_np).to(x.device,
                                                                              dtype=torch.long)  # (sim.n_input_neurons,)
                            tile_period = tile_codes_torch.shape[1]
                            tile_idx = 0

                        # 4) Baseline for all neurons (mean luminance), then write per-column values to PRs
                        x.stimulus[:] = 0.5
                        col_vals_pm1 = tile_codes_torch[:, tile_idx % tile_period]  # (n_columns,), ±1 before knobs
                        # Apply the two simple knobs per frame on ±1 codes
                        col_vals_pm1 = apply_pairwise_knobs_torch(
                            code_pm1=col_vals_pm1,
                            corr_strength=float(sim.tile_corr_strength),
                            flip_prob=float(sim.tile_flip_prob),
                            seed=int(sim.seed) + int(tile_idx)
                        )
                        # Map to [0,1] with your contrast convention and broadcast via labels
                        col_vals_01 = 0.5 + (tile_contrast * 0.5) * col_vals_pm1
                        x.stimulus[:sim.n_input_neurons] = col_vals_01[tile_labels]

                        tile_idx += 1
                    elif "tile_blue_noise" in sim.visual_input_type:
                        if tile_codes_torch is None:
                            # Label columns and build neighborhood graph
                            tile_labels_np, col_centers = compute_column_labels(u_coords, v_coords, n_columns, seed=tile_seed)
                            try:
                                adj = build_neighbor_graph(col_centers, k=6)
                            except Exception:
                                from scipy.spatial.distance import pdist, squareform
                                D = squareform(pdist(col_centers))
                                nn = np.partition(D + np.eye(D.shape[0]) * 1e9, 1, axis=1)[:, 1]
                                radius = 1.3 * np.median(nn)
                                adj = [set(np.where((D[i] > 0) & (D[i] <= radius))[0].tolist()) for i in
                                       range(len(col_centers))]

                            tile_labels = torch.from_numpy(tile_labels_np).to(x.device, dtype=torch.long)
                            tile_period = 257
                            tile_idx = 0

                            # Pre-generate ±1 codes (keep ±1; no [0,1] mapping here)
                            tile_codes_torch = torch.empty((n_columns, tile_period), dtype=torch.float32, device=x.device)
                            rng = np.random.RandomState(tile_seed)
                            for t in range(tile_period):
                                mask = greedy_blue_mask(adj, n_columns, target_density=0.5, rng=rng)  # boolean mask
                                vals = np.where(mask, 1.0, -1.0).astype(np.float32)  # ±1
                                # NOTE: do not apply flip prob here; we do it uniformly via the helper per frame below
                                tile_codes_torch[:, t] = torch.from_numpy(vals).to(x.device, dtype=torch.float32)

                        # Baseline luminance
                        x.stimulus[:] = 0.5
                        col_vals_pm1 = tile_codes_torch[:, tile_idx % tile_period]  # (n_columns,), ±1 before knobs

                        # Apply the two simple knobs per frame on ±1 codes
                        col_vals_pm1 = apply_pairwise_knobs_torch(
                            code_pm1=col_vals_pm1,
                            corr_strength=float(sim.tile_corr_strength),
                            flip_prob=float(sim.tile_flip_prob),
                            seed=int(sim.seed) + int(tile_idx)
                        )

                        # Map to [0,1] with contrast and broadcast via labels
                        col_vals_01 = 0.5 + (tile_contrast * 0.5) * col_vals_pm1
                        x.stimulus[:sim.n_input_neurons] = col_vals_01[tile_labels]

                        tile_idx += 1
                    else:
                        frame = sequences[frame_id][None, None]
                        net.stimulus.add_input(frame)
                        if (sim.only_noise_visual_input > 0):
                            if (sim.visual_input_type == "") | (it == 0) | ("50/50" in sim.visual_input_type):
                                x.stimulus[:sim.n_input_neurons] = torch.relu(
                                    0.5 + torch.rand(sim.n_input_neurons, dtype=torch.float32,
                                                     device=device) * sim.only_noise_visual_input / 2)
                        else:
                            if 'blank' in sim.visual_input_type:
                                if (data_idx % sim.blank_freq > 0):
                                    x.stimulus = net.stimulus().squeeze()
                                else:
                                    x.stimulus[:] = 0
                            else:
                                x.stimulus = net.stimulus().squeeze()
                            if sim.noise_visual_input > 0:
                                x.stimulus[:sim.n_input_neurons] = x.stimulus[:sim.n_input_neurons] + torch.randn(sim.n_input_neurons,
                                                                                                  dtype=torch.float32,
                                                                                                  device=device) * sim.noise_visual_input

                    x_generated.stimulus = x.stimulus.clone()
                    y_generated = pde(x_generated, edge_index, has_field=False)

                    x_generated_modified.stimulus = x.stimulus.clone()
                    y_generated_modified = pde_modified(x_generated_modified, edge_index, has_field=False)

                    if 'visual' in model_config.field_type:
                        visual_input = model.forward_visual(x, it)
                        x.stimulus[:model.n_input_neurons] = visual_input.squeeze(-1)
                        x.stimulus[model.n_input_neurons:] = 0

                    # Prediction step
                    if tc.training_selected_neurons:
                        x_selected.stimulus = x.stimulus[selected_neuron_ids].clone().detach()
                        if 'RNN' in model_config.signal_model_name:
                            y, h_state = model(x_selected.to_packed(), h=h_state, return_all=True)
                        elif 'LSTM' in model_config.signal_model_name:
                            y, h_state, c_state = model(x_selected.to_packed(), h=h_state, c=c_state, return_all=True)
                        elif 'MLP_ODE' in model_config.signal_model_name:
                            v = x_selected.voltage.unsqueeze(-1)
                            I = x_selected.stimulus.unsqueeze(-1)
                            y = model.rollout_step(v, I, dt=sim.delta_t, method='rk4') - v  # Return as delta
                        elif 'MLP' in model_config.signal_model_name:
                            y = model(x_selected.to_packed(), data_id=None, return_all=False)

                    else:
                        if 'RNN' in model_config.signal_model_name:
                            y, h_state = model(x.to_packed(), h=h_state, return_all=True)
                        elif 'LSTM' in model_config.signal_model_name:
                            y, h_state, c_state = model(x.to_packed(), h=h_state, c=c_state, return_all=True)
                        elif 'MLP_ODE' in model_config.signal_model_name:
                            v = x.voltage.unsqueeze(-1)
                            I = x.stimulus[:sim.n_input_neurons].unsqueeze(-1)
                            y = model.rollout_step(v, I, dt=sim.delta_t, method='rk4') - v  # Return as delta
                        elif 'MLP' in model_config.signal_model_name:
                            y = model(x.to_packed(), data_id=None, return_all=False)
                        elif tc.neural_ODE_training:
                            data_id = torch.zeros((x.n_neurons, 1), dtype=torch.int, device=device)
                            v0 = x.voltage.flatten()
                            v_final, _ = integrate_neural_ode_FlyVis(
                                model=model,
                                v0=v0,
                                x_template=x,
                                edge_index=edge_index,
                                data_id=data_id,
                                time_steps=1,
                                delta_t=sim.delta_t,
                                neurons_per_sample=n_neurons,
                                batch_size=1,
                                has_visual_field='visual' in model_config.field_type,
                                x_ts=None,
                                device=device,
                                k_batch=torch.tensor([it], device=device),
                                ode_method=tc.ode_method,
                                rtol=tc.ode_rtol,
                                atol=tc.ode_atol,
                                adjoint=False,
                                noise_level=0.0
                            )
                            y = (v_final.view(-1, 1) - x.voltage.unsqueeze(-1)) / sim.delta_t
                        else:
                            data_id = torch.zeros((x.n_neurons, 1), dtype=torch.int, device=device)
                            y = model(x, edge_index, data_id=data_id, return_all=False)

                    # Save states (pack to legacy (N, 9) numpy for downstream analysis)
                    x_generated_list.append(to_numpy(x_generated.to_packed().clone().detach()))
                    x_generated_modified_list.append(to_numpy(x_generated_modified.to_packed().clone().detach()))

                    if tc.training_selected_neurons:
                        x_list.append(to_numpy(x_selected.to_packed().clone().detach()))
                    else:
                        x_list.append(to_numpy(x.to_packed().clone().detach()))

                    # Integration step
                    # Optionally disable process noise at test time, even if model was trained with noise
                    effective_noise_level = 0.0 if rollout_without_noise else sim.noise_model_level
                    if effective_noise_level > 0:
                        x_generated.voltage = x_generated.voltage + sim.delta_t * y_generated.squeeze(-1) + torch.randn(
                            n_neurons, dtype=torch.float32, device=device
                        ) * effective_noise_level
                        x_generated_modified.voltage = x_generated_modified.voltage + sim.delta_t * y_generated_modified.squeeze(-1) + torch.randn(
                            n_neurons, dtype=torch.float32, device=device
                        ) * effective_noise_level
                    else:
                        x_generated.voltage = x_generated.voltage + sim.delta_t * y_generated.squeeze(-1)
                        x_generated_modified.voltage = x_generated_modified.voltage + sim.delta_t * y_generated_modified.squeeze(-1)

                    if tc.training_selected_neurons:
                        if 'MLP_ODE' in model_config.signal_model_name:
                            x_selected.voltage = x_selected.voltage + y.squeeze(-1)  # y already contains full update
                        else:
                            x_selected.voltage = x_selected.voltage + sim.delta_t * y.squeeze(-1)
                        if (it <= warm_up_length) and ('RNN' in model_config.signal_model_name or 'LSTM' in model_config.signal_model_name):
                            x_selected.voltage = x_generated.voltage[selected_neuron_ids].clone()
                    else:
                        if 'MLP_ODE' in model_config.signal_model_name:
                            x.voltage = x.voltage + y.squeeze(-1)  # y already contains full update
                        else:
                            x.voltage = x.voltage + sim.delta_t * y.squeeze(-1)
                        if (it <= warm_up_length) and ('RNN' in model_config.signal_model_name):
                            x.voltage = x_generated.voltage.clone()

                    if sim.calcium_type == "leaky":
                        # Voltage-driven activation
                        if sim.calcium_activation == "softplus":
                            u = torch.nn.functional.softplus(x.voltage)
                        elif sim.calcium_activation == "relu":
                            u = torch.nn.functional.relu(x.voltage)
                        elif sim.calcium_activation == "tanh":
                            u = torch.tanh(x.voltage)
                        elif sim.calcium_activation == "identity":
                            u = x.voltage.clone()

                        x.calcium = x.calcium + (sim.delta_t / sim.calcium_tau) * (-x.calcium + u)
                        x.calcium = torch.clamp(x.calcium, min=0.0)
                        x.fluorescence = sim.calcium_alpha * x.calcium + sim.calcium_beta

                        y = (x.calcium - torch.tensor(x_list[-1][:, 7], dtype=torch.float32, device=device)).unsqueeze(-1) / sim.delta_t

                    y_list.append(to_numpy(y.clone().detach()))

                    if (it > 0) & (it < 100) & (it % step == 0) & visualize & (not tc.training_selected_neurons):
                        num = f"{id_fig:06}"
                        id_fig += 1
                        plot_spatial_activity_grid(
                            positions=to_numpy(x.pos),
                            voltages=to_numpy(x.voltage),
                            stimulus=to_numpy(x.stimulus[:sim.n_input_neurons]),
                            neuron_types=to_numpy(x.neuron_type).astype(int),
                            output_path=f"./{log_dir}/tmp_recons/Fig_{run}_{num}.png",
                            calcium=to_numpy(x.calcium) if sim.calcium_type != "none" else None,
                            n_input_neurons=sim.n_input_neurons,
                            style=fig_style,
                        )

                    it = it + 1
                    if it >= target_frames:
                        break
                if it >= target_frames:
                    break

            if it >= target_frames:
                break
    print(f"generated {len(x_list)} frames total")


    if visualize:
        print('generating lossless video ...')

        output_name = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'no_id'
        src = f"./{log_dir}/tmp_recons/Fig_0_000000.png"
        dst = f"./{log_dir}/results/input_{output_name}.png"
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())

        generate_compressed_video_mp4(output_dir=f"./{log_dir}/results", run=run,
                                        output_name=output_name,framerate=20)

        # files = glob.glob(f'./{log_dir}/tmp_recons/*')
        # for f in files:
        #     os.remove(f)


    x_list = np.array(x_list)
    x_generated_list = np.array(x_generated_list)
    x_generated_modified_list = np.array(x_generated_modified_list)
    y_list = np.array(y_list)

    neuron_types = node_types_int

    if sim.calcium_type != "none":
        # Use calcium (index 7)
        activity_true = x_generated_list[:, :, 7].squeeze().T  # (n_neurons, n_frames)
        activity_pred = x_list[:, :, 7].squeeze().T
    else:
        # Use voltage (index 3)
        activity_true = x_generated_list[:, :, 3].squeeze().T
        visual_input_true = x_generated_list[:, :, 4].squeeze().T
        activity_true_modified = x_generated_modified_list[:, :, 3].squeeze().T
        activity_pred = x_list[:, :, 3].squeeze().T


    start_frame = 0
    end_frame = target_frames


    if tc.training_selected_neurons:           # MLP, RNN and ODE are trained on limted number of neurons

        print(f"evaluating on selected neurons only: {selected_neuron_ids}")
        x_generated_list = x_generated_list[:, selected_neuron_ids, :]
        x_generated_modified_list = x_generated_modified_list[:, selected_neuron_ids, :]
        neuron_types = neuron_types[selected_neuron_ids]

        true_slice = activity_true[selected_neuron_ids, start_frame:end_frame]
        visual_input_slice = visual_input_true[selected_neuron_ids, start_frame:end_frame]
        pred_slice = activity_pred[start_frame:end_frame]

        rmse_all, pearson_all, feve_all, r2_all = compute_trace_metrics(true_slice, pred_slice, "selected neurons")

        # Log rollout metrics to file
        rollout_log_path = f"./{log_dir}/results_rollout.log"
        with open(rollout_log_path, 'w') as f:
            f.write("Rollout Metrics for Selected Neurons\n")
            f.write("="*60 + "\n")
            f.write(f"RMSE: {np.mean(rmse_all):.4f} ± {np.std(rmse_all):.4f} [{np.min(rmse_all):.4f}, {np.max(rmse_all):.4f}]\n")
            f.write(f"Pearson r: {np.nanmean(pearson_all):.3f} ± {np.nanstd(pearson_all):.3f} [{np.nanmin(pearson_all):.3f}, {np.nanmax(pearson_all):.3f}]\n")
            f.write(f"R²: {np.nanmean(r2_all):.3f} ± {np.nanstd(r2_all):.3f} [{np.nanmin(r2_all):.3f}, {np.nanmax(r2_all):.3f}]\n")
            f.write(f"FEVE: {np.mean(feve_all):.3f} ± {np.std(feve_all):.3f} [{np.min(feve_all):.3f}, {np.max(feve_all):.3f}]\n")
            f.write(f"\nNumber of neurons evaluated: {len(selected_neuron_ids)}\n")

        if len(selected_neuron_ids)==1:
            pred_slice = pred_slice[None,:]

        filename_ = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'no_id'

        # Determine which figures to create
        if len(selected_neuron_ids) > 50:
            # Create sample: take the last 10 neurons from selected_neuron_ids
            sample_indices = list(range(len(selected_neuron_ids) - 10, len(selected_neuron_ids)))

            figure_configs = [
                ("all", list(range(len(selected_neuron_ids)))),
                ("sample", sample_indices)
            ]
        else:
            figure_configs = [("", list(range(len(selected_neuron_ids))))]

        for fig_suffix, neuron_plot_indices in figure_configs:
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            step_v = 2.5
            lw = 6

            # Adjust fontsize based on number of neurons being plotted
            name_fontsize = 10 if len(neuron_plot_indices) > 50 else 18

            # Plot ground truth (green, thick) — all traces first
            baselines = {}
            for plot_idx, i in enumerate(trange(len(neuron_plot_indices), ncols=100, desc=f"plotting {fig_suffix}")):
                neuron_idx = neuron_plot_indices[i]
                baseline = np.mean(true_slice[neuron_idx])
                baselines[plot_idx] = baseline
                ax.plot(true_slice[neuron_idx] - baseline + plot_idx * step_v, linewidth=lw+2, c='#66cc66', alpha=0.9,
                        label='ground truth' if plot_idx == 0 else None)
                # Plot visual input only for neuron_id = 0
                if ((selected_neuron_ids[neuron_idx] == 0) | (len(neuron_plot_indices) < 50)) and visual_input_slice[neuron_idx].mean() > 0:
                    ax.plot(visual_input_slice[neuron_idx] - baseline + plot_idx * step_v, linewidth=1, c='yellow', alpha=0.9,
                            linestyle='--', label='visual input')

            # Plot predictions (black, thin) — on top
            for plot_idx, i in enumerate(range(len(neuron_plot_indices))):
                neuron_idx = neuron_plot_indices[i]
                baseline = baselines[plot_idx]
                ax.plot(pred_slice[neuron_idx] - baseline + plot_idx * step_v, linewidth=1, c=mc,
                        label='prediction' if plot_idx == 0 else None)

            for plot_idx, i in enumerate(neuron_plot_indices):
                type_idx = int(to_numpy(x.neuron_type[selected_neuron_ids[i]]).item())
                ax.text(-50, plot_idx * step_v, f'{index_to_name[type_idx]}', fontsize=name_fontsize, va='bottom', ha='right', color='black')

            ax.set_ylim([-step_v, len(neuron_plot_indices) * (step_v + 0.25 + 0.15 * (len(neuron_plot_indices)//50))])
            ax.set_yticks([])
            ax.set_xlabel('frame', fontsize=20)
            ax.set_xticks([0, (end_frame - start_frame) // 2, end_frame - start_frame])
            ax.set_xticklabels([start_frame, end_frame//2, end_frame], fontsize=16)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.legend(loc='upper right', fontsize=14, frameon=False)
            ax.set_xlim([0, end_frame - start_frame + 100])

            plt.tight_layout()
            save_suffix = f"_{fig_suffix}" if fig_suffix else ""
            plt.savefig(f"./{log_dir}/results/rollout_{filename_}_{sim.visual_input_type}{save_suffix}.png", dpi=300, bbox_inches='tight')
            plt.close()

    else:

        rmse_all, pearson_all, feve_all, r2_all = compute_trace_metrics(activity_true, activity_pred, "all neurons")

        # Log rollout metrics to file
        rollout_log_path = f"./{log_dir}/results_rollout.log"
        with open(rollout_log_path, 'w') as f:
            f.write("Rollout Metrics for All Neurons\n")
            f.write("="*60 + "\n")
            f.write(f"RMSE: {np.mean(rmse_all):.4f} ± {np.std(rmse_all):.4f} [{np.min(rmse_all):.4f}, {np.max(rmse_all):.4f}]\n")
            f.write(f"Pearson r: {np.nanmean(pearson_all):.3f} ± {np.nanstd(pearson_all):.3f} [{np.nanmin(pearson_all):.3f}, {np.nanmax(pearson_all):.3f}]\n")
            f.write(f"R²: {np.nanmean(r2_all):.3f} ± {np.nanstd(r2_all):.3f} [{np.nanmin(r2_all):.3f}, {np.nanmax(r2_all):.3f}]\n")
            f.write(f"FEVE: {np.mean(feve_all):.3f} ± {np.std(feve_all):.3f} [{np.min(feve_all):.3f}, {np.max(feve_all):.3f}]\n")
            f.write(f"\nNumber of neurons evaluated: {len(activity_true)}\n")
            f.write(f"Frames evaluated: {start_frame} to {end_frame}\n")

        # Write to analysis log file for Claude
        if log_file:
            log_file.write(f"test_R2: {np.nanmean(r2_all):.4f}\n")
            log_file.write(f"test_pearson: {np.nanmean(pearson_all):.4f}\n")

        filename_ = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'no_id'

        # Create two figures with different neuron type selections
        for fig_name, selected_types in [
            ("selected", [55, 15, 43, 39, 35, 31, 23, 19, 12, 5]),  # L1, Mi12, Mi2, R1, T1, T4a, T5a, Tm1, Tm4, Tm9
            ("all", np.arange(0, n_neuron_types))
        ]:
            neuron_indices = []
            for stype in selected_types:
                indices = np.where(neuron_types == stype)[0]
                if len(indices) > 0:
                    neuron_indices.append(indices[0])

            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            true_slice = activity_true[neuron_indices, start_frame:end_frame]
            visual_input_slice = visual_input_true[neuron_indices, start_frame:end_frame]
            pred_slice = activity_pred[neuron_indices, start_frame:end_frame]
            step_v = 2.5
            lw = 2

            # Adjust fontsize based on number of neurons
            name_fontsize = 10 if len(selected_types) > 50 else 18

            # Plot ground truth (green, thick) — all traces first
            baselines = {}
            for i in range(len(neuron_indices)):
                baseline = np.mean(true_slice[i])
                baselines[i] = baseline
                ax.plot(true_slice[i] - baseline + i * step_v, linewidth=lw+2, c='#66cc66', alpha=0.9,
                        label='ground truth' if i == 0 else None)
                # Plot visual input for neuron 0 OR when fewer than 50 neurons
                if ((neuron_indices[i] == 0) | (len(neuron_indices) < 50)) and visual_input_slice[i].mean() > 0:
                    ax.plot(visual_input_slice[i] - baseline + i * step_v, linewidth=0.7, c='red', alpha=0.9,
                            linestyle='--', label='visual input')

            # Plot predictions (black, thin) — on top
            for i in range(len(neuron_indices)):
                baseline = baselines[i]
                ax.plot(pred_slice[i] - baseline + i * step_v, linewidth=0.7, label='prediction' if i == 0 else None, c=mc)


            for i in range(len(neuron_indices)):
                type_idx = selected_types[i]
                ax.text(-50, i * step_v, f'{index_to_name[type_idx]}', fontsize=name_fontsize, va='bottom', ha='right', color='black')

            ax.set_ylim([-step_v, len(neuron_indices) * (step_v + 0.25 + 0.15 * (len(neuron_indices)//50))])
            ax.set_yticks([])
            ax.set_xticks([0, (end_frame - start_frame) // 2, end_frame - start_frame])
            ax.set_xticklabels([start_frame, end_frame//2, end_frame], fontsize=16)
            ax.set_xlabel('frame', fontsize=20)
            ax.set_xlim([-50, end_frame - start_frame + 100])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.legend(loc='upper right', fontsize=14, frameon=False)

            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/rollout_{filename_}_{sim.visual_input_type}_{fig_name}.png", dpi=300, bbox_inches='tight')
            plt.close()

        if ('test_ablation' in test_mode) or ('test_inactivity' in test_mode):
            np.save(f"./{log_dir}/results/activity_modified.npy", activity_true_modified)
            np.save(f"./{log_dir}/results/activity_modified_pred.npy", activity_pred)
        else:
            np.save(f"./{log_dir}/results/activity_true.npy", activity_true)
            np.save(f"./{log_dir}/results/activity_pred.npy", activity_pred)


