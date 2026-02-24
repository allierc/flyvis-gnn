# %% [raw]
# ---
# title: "Noise-free: 13,741 neurons, DAVIS visual input"
# author: Cédric Allier, Michael Innerberger, Stephan Saalfeld
# categories:
#   - FlyVis
#   - Simulation
#   - GNN Training
# execute:
#   echo: false
# image: "graphs_data/fly/flyvis_noise_free/activity_traces.png"
# ---

# %% [markdown]
# This notebook runs the full GNN pipeline for the **noise-free** FlyVis simulation.
#
# **Simulation parameters:**
#
# - $N_{\text{neurons}}$: 13,741 (1,736 photoreceptors)
# - $N_{\text{types}}$: 65
# - $N_{\text{edges}}$: 434,112
# - $N_{\text{frames}}$: 64,000 at 50 FPS
# - Visual input: DAVIS dataset
# - Noise level: 0.0 (noise-free)
# - ODE step: $\Delta t = 0.02$
#
# The simulated dynamics follow:
#
# $$\tau_i\frac{dv_i}{dt} = -v_i + V_i^{\text{rest}} + \sum_{j\in\mathcal{N}_i} W_{ij}\,\text{ReLU}(v_j) + I_i(t)$$
#
# where $\tau_i$ is the time constant, $V_i^{\text{rest}}$ the resting potential,
# $W_{ij}$ the synaptic weight, and $I_i(t)$ the visual stimulus.

# %%
#| output: false
import os
import glob
import warnings

import matplotlib
matplotlib.use('Agg')

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.generators.graph_data_generator import data_generate
from flyvis_gnn.models.graph_trainer import data_train, data_test
from flyvis_gnn.utils import set_device, add_pre_folder, load_and_display
from GNN_PlotFigure import data_plot

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)

# %% [markdown]
# ## Configuration and Setup

# %%
#| echo: true
#| output: false
print()
print("=" * 80)
print("Noise-free: 13,741 neurons, DAVIS visual input")
print("=" * 80)

config_file_ = 'flyvis_noise_free'

config_root = "./config"
config_file, pre_folder = add_pre_folder(config_file_)

config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
config.dataset = pre_folder + config.dataset
config.config_file = pre_folder + config_file_

device = set_device(config.training.device)

log_dir = f'./log/{pre_folder}{config_file_}'
graphs_dir = f'./graphs_data/{pre_folder}{config_file_}'

print(f"config: {config.config_file}")
print(f"device: {device}")
print(f"log_dir: {log_dir}")
print(f"graphs_dir: {graphs_dir}")

# %% [markdown]
# ## Step 1: Generate Data
# Simulate the *Drosophila* visual system ODE driven by DAVIS video input.
# Records membrane voltage $v_i(t)$ for 13,741 neurons over 64,000 time frames.
#
# **Outputs:**
#
# - Activity traces for selected neuron types
# - Kinograph showing spatio-temporal voltage pattern

# %%
#| echo: true
#| output: false
print()
print("-" * 80)
print("STEP 1: GENERATE - Simulating fly visual system")
print("-" * 80)

data_exists = os.path.isdir(f'{graphs_dir}/x_list_0')
if data_exists:
    print(f"data already exists at {graphs_dir}/x_list_0/ (zarr)")
    print("skipping simulation...")
else:
    print(f"simulating {config.simulation.n_neurons} neurons, {config.simulation.n_neuron_types} types")
    print(f"generating {config.simulation.n_frames} time frames")
    print(f"visual input: {config.simulation.visual_input_type}")
    print(f"noise level: {config.simulation.noise_model_level}")
    print(f"output: {graphs_dir}/")
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

# %%
#| fig-cap: "Activity traces $v_i(t)$ for selected neuron types."
load_and_display(f"{graphs_dir}/activity_traces.png")

# %%
#| fig-cap: "Kinograph: spatio-temporal voltage pattern."
load_and_display(f"{graphs_dir}/kinograph.png")

# %% [markdown]
# ## Step 2: Train GNN
# Train a message-passing GNN to learn the effective connectivity $\widehat{W}_{ij}$,
# latent embeddings $\mathbf{a}_i \in \mathbb{R}^2$, and functions $f_\theta$, $g_\phi$.
#
# The GNN approximates the dynamics with:
#
# $$\frac{\widehat{dv}_i}{dt} = f_\theta\!\left(v_i,\,\mathbf{a}_i,\,\sum_{j\in\mathcal{N}_i} \widehat{W}_{ij}\,g_\phi(v_j, \mathbf{a}_j)^2,\,I_i(t)\right)$$
#
# where $f_\theta$ (MLP$_0$) and $g_\phi$ (MLP$_1$) are three-layer perceptrons (width 80, ReLU).

# %%
#| echo: true
#| output: false
print()
print("-" * 80)
print("STEP 2: TRAIN - Learning W, embeddings a_i, f_theta, g_phi")
print("-" * 80)

model_files = glob.glob(f'{log_dir}/models/*.pt')
if model_files:
    print(f"trained model already exists at {log_dir}/models/")
    print("skipping training (delete models folder to retrain)")
else:
    print(f"training for {config.training.n_epochs} epochs")
    print(f"models: {log_dir}/models/")
    print(f"training plots: {log_dir}/tmp_training/")
    print()
    data_train(
        config=config,
        erase=False,
        best_model='',
        device=device,
    )

# %% [markdown]
# ## Step 3: GNN Evaluation
# Evaluate the trained model: learned connectivity $\widehat{W}_{ij}$,
# latent embeddings $\mathbf{a}_i$, and learned functions $f_\theta$, $g_\phi$.

# %%
#| echo: true
#| output: false
print()
print("-" * 80)
print("STEP 3: GNN EVALUATION")
print("-" * 80)

folder_name = f'./log/{pre_folder}/tmp_results/'
os.makedirs(folder_name, exist_ok=True)
data_plot(
    config=config,
    config_file=config_file,
    epoch_list=['best'],
    style='color',
    extended='plots',
    device=device,
)

# %% [markdown]
# ### Evaluation Results

# %%
#| fig-cap: "Learned connectivity matrix $\\widehat{W}_{ij}$."
load_and_display(f"{log_dir}/results/connectivity_learned.png")

# %%
#| fig-cap: "Comparison of learned $\\widehat{W}_{ij}$ and true $W_{ij}$."
load_and_display(f"{log_dir}/results/weights_comparison_corrected.png")

# %%
#| fig-cap: "Learned latent embeddings $\\mathbf{a}_i \\in \\mathbb{R}^2$."
load_and_display(f"{log_dir}/results/embedding.pdf")

# %%
#| fig-cap: "Learned neuron update function $f_\\theta$ (MLP$_0$)."
load_and_display(f"{log_dir}/results/MLP0.png")

# %%
#| fig-cap: "Learned edge message function $g_\\phi$ (MLP$_1$)."
load_and_display(f"{log_dir}/results/MLP1_corrected.png")

# %% [markdown]
# ## Step 4: Test Model
# Test the trained GNN with rollout inference. The learned model is run forward
# in time to predict $v_i(t)$ and compared to ground truth.

# %%
#| echo: true
#| output: false
print()
print("-" * 80)
print("STEP 4: TEST - Rollout inference")
print("-" * 80)

config.simulation.noise_model_level = 0.0

data_test(
    config=config,
    visualize=True,
    style="color name continuous_slice",
    verbose=False,
    best_model='best',
    run=0,
    test_mode="",
    sample_embedding=False,
    step=10,
    n_rollout_frames=250,
    device=device,
    particle_of_interest=0,
    new_params=None,
    rollout_without_noise=False,
)

# %% [markdown]
# ### Rollout Results

# %%
#| fig-cap: "Rollout: ground truth $v_i(t)$ (green) vs GNN prediction $\\widehat{v}_i(t)$ (black)."
output_name = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'no_id'
load_and_display(f"{log_dir}/results/rollout_{output_name}_DAVIS.png")
