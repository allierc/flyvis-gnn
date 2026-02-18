import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
# Optional imports (not available in flyvis-gnn spinoff)
try:
    from flyvis_gnn.data_loaders import load_wormvae_data, load_zebrafish_data
except ImportError:
    load_wormvae_data = None
    load_zebrafish_data = None
from flyvis_gnn.figure_style import default_style, dark_style
from flyvis_gnn.plot import (
    plot_spatial_activity_grid,
    plot_kinograph,
    plot_activity_traces,
    plot_selected_neuron_traces,
)
from flyvis_gnn.neuron_state import NeuronState
from flyvis_gnn.zarr_io import ZarrArrayWriter, ZarrSimulationWriterV3
try:
    from flyvis_gnn.generators.davis import AugmentedVideoDataset, CombinedVideoDataset
except ImportError:
    AugmentedVideoDataset = None
    CombinedVideoDataset = None
from flyvis_gnn.generators.utils import (
    generate_compressed_video_mp4,
    get_equidistant_points,
    mseq_bits,
    assign_columns_from_uv,
    compute_column_labels,
    build_neighbor_graph,
    greedy_blue_mask,
    apply_pairwise_knobs_torch,
)
from flyvis_gnn.utils import to_numpy, get_datavis_root_dir
from tqdm import tqdm, trange
import os


def data_generate(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=5,
    alpha=0.2,
    ratio=1,
    scenario="none",
    best_model=None,
    device=None,
    bSave=True,
    log_file=None,
):

    print(f"\033[94mdataset: {config.dataset}\033[0m")

    if (os.path.isfile(f"./graphs_data/{config.dataset}/x_list_0.npy")) | (
        os.path.isfile(f"./graphs_data/{config.dataset}/x_list_0.pt")
    ):
        print("watch out: data already generated")
        # return

    if config.data_folder_name != "none":
        generate_from_data(config=config, device=device, visualize=visualize, style=style, step=step)
    else:
        data_generate_fly_voltage(
            config,
            visualize=visualize,
            run_vizualized=run_vizualized,
            style=style,
            erase=erase,
            step=step,
            device=device,
            bSave=bSave,
        )

    default_style.apply_globally()


def generate_from_data(config, device, visualize=True, step=None, cmap=None, style=None):
    data_folder_name = config.data_folder_name

    if "wormvae" in data_folder_name:
        load_wormvae_data(config, device, visualize, step)
    elif "NeuroPAL" in data_folder_name:
        # load_neuropal_data(config, device, visualize, step)  # TODO: Function not yet implemented
        raise NotImplementedError("NeuroPAL data loading not yet implemented")
    elif 'Zapbench' in data_folder_name:
        load_zebrafish_data(config, device, visualize, step, cmap, style)
    else:
        raise ValueError(f"Unknown data folder name {data_folder_name}")

def data_generate_fly_voltage(config, visualize=True, run_vizualized=0, style="color", erase=False, step=5, device=None,
                              bSave=True):

    fig_style = dark_style if "black" in style else default_style
    fig_style.apply_globally()

    sim = config.simulation
    tc = config.training
    model_config = config.graph_model

    torch.random.fork_rng(devices=device)
    if sim.seed != 42:
        torch.random.manual_seed(sim.seed)
        np.random.seed(sim.seed)  # Ensure numpy random state is also seeded for reproducibility

    n_frames = sim.n_frames
    n_neurons = sim.n_neurons

    print(f"generating data ... {model_config.signal_model_name}  noise: {sim.noise_model_level}  seed: {sim.seed}")

    run = 0

    os.makedirs("./graphs_data/fly", exist_ok=True)
    folder = f"./graphs_data/{config.dataset}/"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"./graphs_data/{config.dataset}/Fig/", exist_ok=True)
    files = glob.glob(f'./graphs_data/{config.dataset}/Fig/*')
    for f in files:
        os.remove(f)

    extent = 8

    from flyvis.datasets.sintel import AugmentedSintel
    from flyvis import NetworkView, Network
    from flyvis.utils.config_utils import get_default_config, CONFIG_PATH

    # flyvis.__init__ sets root logger to INFO via basicConfig — restore to WARNING
    import logging
    logging.getLogger().setLevel(logging.WARNING)
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
            "skip_short_videos": sim.skip_short_videos,
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

    # Initialize the ground-truth flyvis network from a pre-trained checkpoint.
    # This loads the biological connectome (neuron types, synaptic weights, time constants)
    # from the flyvis library, using ensemble_id/model_id to select a specific trained model.
    # The network is then used as the "simulator" to generate voltage traces via its PDE dynamics.
    config_net = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
    config_net.connectome.extent = extent
    net = Network(**config_net)
    nnv = NetworkView(f"flow/{sim.ensemble_id}/{sim.model_id}")
    trained_net = nnv.init_network(checkpoint=0)
    net.load_state_dict(trained_net.state_dict())
    torch.set_grad_enabled(False)

    # Extract ground-truth parameters: time constants (tau), resting potentials (V_rest),
    # and effective synaptic weights (strength * count * sign) for PDE simulation.
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

        # If sim.n_extra_null_edges > 424*424, prioritize L1-L2 (receiver) and R1-R2 (sender) connections
        if sim.n_extra_null_edges > 424 * 424:
            print("Prioritizing L1-L2 and R1-R2 connections...")
            # R1 R2 (sender): columns 0 to 433
            col_start = 0
            col_end = 217 * 2  # 434
            # L1 L2 (receiver): rows 1736 to 2159
            row_start = 1736
            row_end = 1736 + 217 * 2  # 2160

            # Generate all possible edges in the priority region
            priority_edges = []
            for source in range(col_start, col_end):
                for target in range(row_start, row_end):
                    if (source, target) not in existing_edges and source != target:
                        priority_edges.append([source, target])

            # Add priority edges first
            n_priority = min(len(priority_edges), sim.n_extra_null_edges)
            random.shuffle(priority_edges)
            extra_edges.extend(priority_edges[:n_priority])
            print(f"Added {len(extra_edges)} priority edges from R1-R2 to L1-L2")

            # Fill remaining with random edges if needed
            remaining = sim.n_extra_null_edges - len(extra_edges)
            if remaining > 0:
                print(f"Filling remaining {remaining} edges randomly...")
                existing_edges.update([(e[0], e[1]) for e in extra_edges])
                max_attempts = remaining * 10
                attempts = 0
                while len(extra_edges) < sim.n_extra_null_edges and attempts < max_attempts:
                    source = random.randint(0, n_neurons - 1)
                    target = random.randint(0, n_neurons - 1)
                    if (source, target) not in existing_edges and source != target:
                        extra_edges.append([source, target])
                        existing_edges.add((source, target))
                    attempts += 1
        else:
            # Original random edge generation
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
            print(f"Total extra edges added: {len(extra_edges)}")

    pde = FlyVisODE(p=p, f=torch.nn.functional.relu, params=sim.params,
                    model_type=model_config.signal_model_name, n_neuron_types=sim.n_neuron_types, device=device)

    if bSave:
        torch.save(p["w"], f"./graphs_data/{config.dataset}/weights.pt")
        torch.save(edge_index, f"graphs_data/{config.dataset}/edge_index.pt")
        torch.save(p["tau_i"], f"./graphs_data/{config.dataset}/taus.pt")
        torch.save(p["V_i_rest"], f"./graphs_data/{config.dataset}/V_i_rest.pt")

    x_coords, y_coords, u_coords, v_coords = get_photoreceptor_positions_from_net(net)

    node_types = np.array(net.connectome.nodes["type"])
    node_types_str = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in node_types]
    grouped_types = np.array([group_by_direction_and_function(t) for t in node_types_str])
    _ , node_types_int = np.unique(node_types, return_inverse=True)

    X1 = torch.tensor(np.stack((x_coords, y_coords), axis=1), dtype=torch.float32, device=device)

    xc, yc = get_equidistant_points(n_points=n_neurons - x_coords.shape[0])
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    X1 = torch.cat((X1, pos[torch.randperm(pos.size(0))]), dim=0)

    state = net.steady_state(t_pre=2.0, dt=sim.delta_t, batch_size=1)
    initial_state = state.nodes.activity.squeeze()
    n_neurons = len(initial_state)

    sequences = stimulus_dataset[0]["lum"]
    frame = sequences[0][None, None]
    net.stimulus.add_input(frame)

    # init neuron state x

    _init_calcium = torch.rand(n_neurons, dtype=torch.float32, device=device)
    x = NeuronState(
        index=torch.arange(n_neurons, dtype=torch.long, device=device),
        pos=X1,
        voltage=initial_state,
        stimulus=net.stimulus().squeeze(),
        group_type=torch.tensor(grouped_types, dtype=torch.long, device=device),
        neuron_type=torch.tensor(node_types_int, dtype=torch.long, device=device),
        calcium=_init_calcium,
        fluorescence=sim.calcium_alpha * _init_calcium + sim.calcium_beta,
    )

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

    dataset_length = len(stimulus_dataset)
    frames_per_sequence = 35
    total_frames_per_pass = dataset_length * frames_per_sequence

    if n_frames == 0:
        # n_frames=0: use each source frame exactly once (no reuse)
        num_passes_needed = 1
        target_frames = float('inf')
        print(f"n_frames=0 mode: single pass through {dataset_length} sequences (no frame reuse)")
    else:
        target_frames = n_frames
        num_passes_needed = (target_frames // total_frames_per_pass) + 1

    # use zarr writers for incremental saving (memory efficient)
    # V3 format: each NeuronState field gets its own zarr array
    x_writer = ZarrSimulationWriterV3(
        path=f"graphs_data/{config.dataset}/x_list_{run}",
        n_neurons=n_neurons,
        time_chunks=2000,
    )
    y_writer = ZarrArrayWriter(
        path=f"graphs_data/{config.dataset}/y_list_{run}",
        n_neurons=n_neurons,
        n_features=1,
        time_chunks=2000,
    )
    it = sim.start_frame
    id_fig = 0

    tile_labels = None
    tile_codes_torch = None
    tile_period = None
    tile_idx = 0
    n_columns = sim.n_input_neurons // 8

    with torch.no_grad():
        for pass_num in range(num_passes_needed):
            for data_idx, data in enumerate(tqdm(stimulus_dataset, desc="processing stimulus data", ncols=100)):
                if sim.simulation_initial_state:
                    x.voltage[:] = initial_state
                    if sim.only_noise_visual_input > 0:
                        x.stimulus[:sim.n_input_neurons] = torch.clamp(torch.relu(
                            0.5 + torch.rand(sim.n_input_neurons, dtype=torch.float32,
                                             device=device) * sim.only_noise_visual_input / 2), 0, 1)

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
                            x.stimulus[:] = net.stimulus().squeeze()
                            if current_type == "sintel":
                                sintel_frame_idx += 1
                            elif current_type == "davis":
                                davis_frame_idx += 1
                        mixed_frame_count += 1
                    elif "tile_mseq" in sim.visual_input_type:
                        if tile_codes_torch is None:
                            # 1) Cluster photoreceptors into columns based on (u,v)
                            tile_labels_np = assign_columns_from_uv(
                                u_coords, v_coords, n_columns, random_state=sim.seed
                            )  # shape: (sim.n_input_neurons,)

                            # 2) Build per-column m-sequences (±1) with random phase per column
                            base = mseq_bits(p=8, seed=sim.seed).astype(np.float32)  # ±1, shape (255,)
                            rng = np.random.RandomState(sim.seed)
                            phases = rng.randint(0, base.shape[0], size=n_columns)
                            tile_codes_np = np.stack([np.roll(base, ph) for ph in phases], axis=0)  # (n_columns, 255), ±1

                            # 3) Convert to torch on the right device/dtype; keep as ±1 (no [0,1] mapping here)
                            tile_codes_torch = torch.from_numpy(tile_codes_np).to(device,
                                                                                  dtype=torch.float32)  # (n_columns, 255), ±1
                            tile_labels = torch.from_numpy(tile_labels_np).to(device,
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
                        col_vals_01 = 0.5 + (sim.tile_contrast * 0.5) * col_vals_pm1
                        x.stimulus[:sim.n_input_neurons] = col_vals_01[tile_labels]

                        tile_idx += 1
                    elif "tile_blue_noise" in sim.visual_input_type:
                        if tile_codes_torch is None:
                            # Label columns and build neighborhood graph
                            tile_labels_np, col_centers = compute_column_labels(u_coords, v_coords, n_columns, seed=sim.seed)
                            try:
                                adj = build_neighbor_graph(col_centers, k=6)
                            except Exception:
                                from scipy.spatial.distance import pdist, squareform
                                D = squareform(pdist(col_centers))
                                nn = np.partition(D + np.eye(D.shape[0]) * 1e9, 1, axis=1)[:, 1]
                                radius = 1.3 * np.median(nn)
                                adj = [set(np.where((D[i] > 0) & (D[i] <= radius))[0].tolist()) for i in
                                       range(len(col_centers))]

                            tile_labels = torch.from_numpy(tile_labels_np).to(device, dtype=torch.long)
                            tile_period = 257
                            tile_idx = 0

                            # Pre-generate ±1 codes (keep ±1; no [0,1] mapping here)
                            tile_codes_torch = torch.empty((n_columns, tile_period), dtype=torch.float32, device=device)
                            rng = np.random.RandomState(sim.seed)
                            for t in range(tile_period):
                                mask = greedy_blue_mask(adj, n_columns, target_density=0.5, rng=rng)  # boolean mask
                                vals = np.where(mask, 1.0, -1.0).astype(np.float32)  # ±1
                                # NOTE: do not apply flip prob here; we do it uniformly via the helper per frame below
                                tile_codes_torch[:, t] = torch.from_numpy(vals).to(device, dtype=torch.float32)

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
                        col_vals_01 = 0.5 + (sim.tile_contrast * 0.5) * col_vals_pm1
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
                                    x.stimulus[:] = net.stimulus().squeeze()
                                else:
                                    x.stimulus[:] = 0
                            else:
                                x.stimulus[:] = net.stimulus().squeeze()
                            if sim.noise_visual_input > 0:
                                x.stimulus[:sim.n_input_neurons] = x.stimulus[:sim.n_input_neurons] + torch.randn(sim.n_input_neurons,
                                                                                                  dtype=torch.float32,
                                                                                                  device=device) * sim.noise_visual_input

                    y = pde(x, edge_index, has_field=False)

                    # save previous calcium for derivative computation (before appending current frame)
                    prev_calcium = x.calcium.clone()

                    x_writer.append_state(x)

                    dv = y.squeeze()
                    if sim.noise_model_level > 0:
                        x.voltage = x.voltage + sim.delta_t * dv + torch.randn(n_neurons, dtype=torch.float32, device=device) * sim.noise_model_level
                    else:
                        x.voltage = x.voltage + sim.delta_t * dv

                    if sim.calcium_type == "leaky":
                        # Voltage-driven activation
                        if sim.calcium_activation == "softplus":
                            s = torch.nn.functional.softplus(x.voltage)
                        elif sim.calcium_activation == "relu":
                            s = torch.nn.functional.relu(x.voltage)
                        elif sim.calcium_activation == "tanh":
                            s = 1 + torch.tanh(x.voltage)
                        elif sim.calcium_activation == "identity":
                            s = x.voltage.clone()

                        x.calcium = x.calcium + (sim.delta_t / sim.calcium_tau) * (-x.calcium + s)
                        x.fluorescence = sim.calcium_alpha * x.calcium + sim.calcium_beta

                        y = ((x.calcium - prev_calcium) / sim.delta_t).unsqueeze(-1)

                    y_writer.append(to_numpy(y.clone().detach()))

                    if (visualize & (run == run_vizualized) & (it > 0) & (it % step == 0) & (it <= 50 * step)):
                        num = f"{id_fig:06}"
                        id_fig += 1
                        plot_spatial_activity_grid(
                            positions=to_numpy(X1),
                            voltages=to_numpy(x.voltage),
                            stimulus=to_numpy(x.stimulus[:sim.n_input_neurons]),
                            neuron_types=to_numpy(x.neuron_type).astype(int),
                            output_path=f"graphs_data/{config.dataset}/Fig/Fig_{run}_{num}.png",
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


    # finalize zarr writers
    n_frames_written = x_writer.finalize()
    y_writer.finalize()
    print(f"generated {n_frames_written} frames total (saved as .zarr)")

    # restore gradient computation now (before any early-return paths)
    torch.set_grad_enabled(True)

    # --- Always run diagnostics after data generation ---
    from flyvis_gnn.zarr_io import load_simulation_data, load_raw_array
    x_ts = load_simulation_data(f"graphs_data/{config.dataset}/x_list_{run}")
    y_list = load_raw_array(f"graphs_data/{config.dataset}/y_list_{run}")

    # Compute ranks (used in kinographs and traces)
    print('computing effective rank ...')
    from sklearn.utils.extmath import randomized_svd
    activity_full = x_ts.voltage.numpy()  # (n_frames, n_neurons)
    n_comp = min(50, min(activity_full.shape) - 1)
    _, S_act, _ = randomized_svd(activity_full, n_components=n_comp, random_state=0)
    cumvar_act = np.cumsum(S_act**2) / np.sum(S_act**2)
    rank_90_act = int(np.searchsorted(cumvar_act, 0.90) + 1)
    rank_99_act = int(np.searchsorted(cumvar_act, 0.99) + 1)

    input_for_svd = x_ts.stimulus[:, :sim.n_input_neurons].numpy()
    n_comp_input = min(50, min(input_for_svd.shape) - 1)
    _, S_inp, _ = randomized_svd(input_for_svd, n_components=n_comp_input, random_state=0)
    cumvar_inp = np.cumsum(S_inp**2) / np.sum(S_inp**2)
    rank_90_inp = int(np.searchsorted(cumvar_inp, 0.90) + 1)
    rank_99_inp = int(np.searchsorted(cumvar_inp, 0.99) + 1)

    print(f'activity rank(90%)={rank_90_act}  rank(99%)={rank_99_act}')
    print(f'visual input rank(90%)={rank_90_inp}  rank(99%)={rank_99_inp}')

    print('plot kinograph ...')
    plot_kinograph(
        activity=activity_full.T,
        stimulus=x_ts.stimulus[:, :sim.n_input_neurons].numpy().T,
        output_path=f'./graphs_data/{config.dataset}/kinograph.png',
        rank_90_act=rank_90_act,
        rank_99_act=rank_99_act,
        rank_90_inp=rank_90_inp,
        rank_99_inp=rank_99_inp,
        zoom_size=200,
        style=fig_style,
    )

    print('plot activity traces ...')
    plot_activity_traces(
        activity=activity_full.T,
        output_path=f'./graphs_data/{config.dataset}/activity_traces.png',
        n_traces=100,
        max_frames=10000,
        n_input_neurons=sim.n_input_neurons,
        style=fig_style,
    )

    # SVD analysis (4-panel plot)
    print('svd analysis ...')
    from flyvis_gnn.models.utils import analyze_data_svd
    folder = f'./graphs_data/{config.dataset}'
    svd_results = analyze_data_svd(x_ts, folder, config=config, is_flyvis=True,
                                   save_in_subfolder=False)

    # Save ranks to log file
    log_path = f'./graphs_data/{config.dataset}/generation_log.txt'
    with open(log_path, 'w') as log_f:
        log_f.write(f'dataset: {config.dataset}\n')
        log_f.write(f'n_neurons: {n_neurons}\n')
        log_f.write(f'n_input_neurons: {sim.n_input_neurons}\n')
        log_f.write(f'n_frames: {n_frames}\n')
        log_f.write(f'visual_input_type: {sim.visual_input_type}\n')
        log_f.write(f'noise_model_level: {sim.noise_model_level}\n')
        log_f.write(f'model_id: {sim.model_id}\n')
        log_f.write(f'ensemble_id: {sim.ensemble_id}\n')
        log_f.write(f'\n')
        log_f.write(f'activity_rank_90: {rank_90_act}\n')
        log_f.write(f'activity_rank_99: {rank_99_act}\n')
        log_f.write(f'input_rank_90: {rank_90_inp}\n')
        log_f.write(f'input_rank_99: {rank_99_inp}\n')
        if svd_results.get('activity'):
            log_f.write(f'svd_activity_rank_90: {svd_results["activity"]["rank_90"]}\n')
            log_f.write(f'svd_activity_rank_99: {svd_results["activity"]["rank_99"]}\n')
        if svd_results.get('visual_stimuli'):
            log_f.write(f'svd_visual_rank_90: {svd_results["visual_stimuli"]["rank_90"]}\n')
            log_f.write(f'svd_visual_rank_99: {svd_results["visual_stimuli"]["rank_99"]}\n')
    print(f'generation log saved to {log_path}')

    if not visualize:
        return

    # Neuron type index to name mapping (CamelCase for legacy plot_neuron_activity_analysis)
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

    activity = x_ts.voltage.to(device).t()  # (n_neurons, n_frames)
    type_list = x.neuron_type.unsqueeze(-1).to(device)

    target_type_name_list = ['R1', 'R7', 'C2', 'Mi11', 'Tm1', 'Tm4', 'Tm30']
    from GNN_PlotFigure import plot_neuron_activity_analysis
    plot_neuron_activity_analysis(activity, target_type_name_list, type_list, index_to_name, n_neurons, n_frames, sim.delta_t, f'graphs_data/{config.dataset}/')

    print('plot figure activity ...')
    plot_selected_neuron_traces(
        activity=to_numpy(activity),
        type_list=to_numpy(type_list.squeeze()),
        output_path=f'./graphs_data/{config.dataset}/activity.png',
        style=fig_style,
    )

    if visualize & (run == run_vizualized):
        print('generating lossless video ...')

        output_name = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'no_id'
        src = f"./graphs_data/{config.dataset}/Fig/Fig_0_000000.png"
        dst = f"./graphs_data/{config.dataset}/input_{output_name}.png"
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())

        generate_compressed_video_mp4(output_dir=f"./graphs_data/{config.dataset}", run=run,
                                      output_name=output_name,framerate=20)

        files = glob.glob(f'./graphs_data/{config.dataset}/Fig/*')
        for f in files:
            os.remove(f)



