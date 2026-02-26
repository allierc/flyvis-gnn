# %% [raw]
# ---
# title: "Data Generation: Drosophila Visual System"
# author: "Allier, Innerberger, Saalfeld"
# categories:
#   - FlyVis
#   - Simulation
#   - Data Generation
# execute:
#   echo: false
# image: "graphs_data/fly/flyvis_noise_free/activity_traces.png"
# ---

# %% [markdown]
# Synapse-level connectomes describe the structure of neural circuits, but not
# the electrical and chemical dynamics. Conversely, large-scale recordings of
# neural activity capture these dynamics, but not the circuit structure. We
# combine binary connectivity and recorded neural activity to infer mechanistic
# models of neural circuits using a GNN trained on *Drosophila* visual system
# simulations (13,741 neurons, 65 cell types, 434,112 connections).
#
# The simulated dynamics follow:
#
# $$\tau_i\frac{dv_i}{dt} = -v_i + V_i^{\text{rest}} + \sum_{j\in\mathcal{N}_i} W_{ij}\,\text{ReLU}(v_j) + I_i(t) + \sigma\,\xi_i(t)$$
#
# where $\tau_i$ is the time constant, $V_i^{\text{rest}}$ the resting potential,
# $W_{ij}$ the synaptic weight, $I_i(t)$ the visual stimulus, and
# $\xi_i(t) \sim \mathcal{N}(0,1)$ is independent Gaussian noise scaled by $\sigma$.
#
# The noise term $\sigma\,\xi_i(t)$ models intrinsic stochasticity in the
# membrane dynamics (e.g. channel noise, synaptic variability), not
# measurement noise. It enters the ODE itself, so it affects the temporal
# evolution of $v_i(t)$ and propagates through the network via synaptic
# connections.
#
# We generate data at three noise levels $\sigma$:
#
# | Dataset | $\sigma$ | Description |
# |---------|----------|-------------|
# | `flyvis_noise_free` | 0.0 | Deterministic (no intrinsic noise) |
# | `flyvis_noise_005` | 0.05 | Low intrinsic noise |
# | `flyvis_noise_05` | 0.5 | High intrinsic noise |

# %%
#| output: false
import os
import warnings

import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.generators.graph_data_generator import data_generate
from flyvis_gnn.utils import set_device, add_pre_folder, load_and_display, graphs_data_path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)


def display_row(paths, titles, figsize=(18, 6)):
    """Display multiple images side by side."""
    fig, axes = plt.subplots(1, len(paths), figsize=figsize)
    if len(paths) == 1:
        axes = [axes]
    for ax, path, title in zip(axes, paths, titles):
        img = imageio.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Configuration

# %%
#| output: false
datasets = [
    ('flyvis_noise_free', 'Noise-free'),
    ('flyvis_noise_005', 'Noise 0.05'),
    ('flyvis_noise_05', 'Noise 0.5'),
]

config_root = "./config"
configs = {}
graphs_dirs = {}

for config_name, label in datasets:
    config_file, pre_folder = add_pre_folder(config_name)
    config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
    config.dataset = pre_folder + config.dataset
    config.config_file = pre_folder + config_name
    configs[config_name] = config
    graphs_dirs[config_name] = graphs_data_path(pre_folder + config_name)

device = set_device(configs[datasets[0][0]].training.device)

for config_name, label in datasets:
    print(f"{label}: {graphs_dirs[config_name]}")

# %% [markdown]
# ## Step 1: Generate Data
# Simulate the *Drosophila* visual system ODE driven by DAVIS video input at
# three noise levels. Each simulation records membrane voltage $v_i(t)$ for
# 13,741 neurons over 64,000 time frames.

# %%
#| echo: true
#| output: false
print()
print("=" * 80)
print("STEP 1: GENERATE - Simulating fly visual system at three noise levels")
print("=" * 80)

for config_name, label in datasets:
    config = configs[config_name]
    graphs_dir = graphs_dirs[config_name]
    print()
    print(f"--- {label} (noise_model_level={config.simulation.noise_model_level}) ---")

    data_exists = os.path.isdir(f'{graphs_dir}/x_list_train') or os.path.isdir(f'{graphs_dir}/x_list_0')
    if data_exists:
        print(f"  data already exists at {graphs_dir}/")
        print("  skipping simulation...")
    else:
        print(f"  simulating {config.simulation.n_neurons} neurons, {config.simulation.n_neuron_types} types")
        print(f"  generating {config.simulation.n_frames} time frames")
        print(f"  visual input: {config.simulation.visual_input_type}")
        print(f"  output: {graphs_dir}/")
        print()
        data_generate(
            config,
            device=device,
            visualize=True,
            run_vizualized=0,
            style="color",
            alpha=1,
            erase=False,
            save=True,
            step=100,
        )

# %% [markdown]
# ## Visual Stimulus Preview
# First frames of shuffled DAVIS sequences used for training and testing
# (noise-free dataset).

# %%
#| lightbox: true
#| fig-cap: "Shuffled DAVIS sequences: first frame of each sequence mapped onto the hex photoreceptor grid. Left: training set. Right: test set."
graphs_dir_0 = graphs_dirs[datasets[0][0]]
display_row(
    [f"{graphs_dir_0}/shuffle_first_frames_train.png",
     f"{graphs_dir_0}/shuffle_first_frames_test.png"],
    ["Train sequences", "Test sequences"],
    figsize=(20, 8),
)

# %% [markdown]
# ## Activity Traces Comparison
# Sample voltage traces $v_i(t)$ across the three noise conditions. Higher
# noise broadens the voltage distribution and obscures fine temporal structure.

# %%
#| lightbox: true
#| fig-cap: "Activity traces at three noise levels: noise-free (left), low noise 0.05 (center), high noise 0.5 (right)."
display_row(
    [f"{graphs_dirs[name]}/activity_traces.png" for name, _ in datasets],
    [label for _, label in datasets],
    figsize=(24, 6),
)
