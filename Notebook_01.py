# %% [raw]
# ---
# title: "GNN Training: Learning Circuit Dynamics"
# author: "Allier, Lappalainen, Saalfeld"
# categories:
#   - FlyVis
#   - GNN
#   - Training
# execute:
#   echo: false
# image: "graphs_data/fly/flyvis_noise_free/tmp_training/embedding/embedding_latest.png"
# ---

# %% [markdown]
# ## Graph Neural Network Model
#
# We approximate the simulated voltage dynamics by a message-passing
# GNN [2]:
#
# $$\frac{\widehat{dv}_i(t)}{dt} = f_\theta\!\left(v_i(t),\,\mathbf{a}_i,\,\sum_{j\in\mathcal{N}_i} \widehat{W}_{ij}\,g_\phi\!\big(v_j(t),\,\mathbf{a}_j\big)^2,\,I_i(t)\right).$$
#
# Nodes of the GNN correspond to neurons and edges correspond to
# functional synaptic connections.  The GNN learns a latent embedding
# $\mathbf{a}_i \in \mathbb{R}^2$ for each neuron $i$, giving each neuron
# a compact latent identity to capture cell-type specific properties (like
# time constants and nonlinearities).
#
# Neuron update $f_\theta = \text{MLP}_0$ and edge message
# $g_\phi = \text{MLP}_1$ are three-layer perceptrons (width 80, ReLU,
# linear output).  $g_\phi$ maps presynaptic inputs $v_j$ to nonnegative
# messages (via squaring) depending on neural embedding $\mathbf{a}_j$,
# which are weighted by $\widehat{W}_{ij}$.  $f_\theta$ processes the
# postsynaptic voltage $v_i$, aggregated input, and external input
# $I_i(t)$ to predict $\widehat{dv}_i(t)/dt$, depending on neural
# embedding $\mathbf{a}_i$.
#
# During training, inputs $I_i(t)$, adjacency $\mathcal{N}_i$, and
# activity $v_i(t)$ are given.  The MLPs, $\widehat{W}_{ij}$, and
# $\mathbf{a}_i$ are optimized by minimizing
#
# $$\mathcal{L}_{\text{pred}} = \sum_{i,t} \|\hat{y}_i(t) - y_i(t)\|^2$$
#
# between simulator targets $y_i(t) = dv_i(t)/dt$ and GNN predictions
# $\hat{y}_i(t) = \widehat{dv}_i(t)/dt$.

# %%
#| output: false
import os
import warnings

from IPython.display import Image, display

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.models.graph_trainer import data_train, data_test
from flyvis_gnn.utils import set_device, add_pre_folder, graphs_data_path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)


def display_image(path, width=700):
    """Display a full-resolution image; width controls inline size (px)."""
    display(Image(filename=path, width=width))

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
    config = configs[config_name]
    tc = config.training
    mc = config.graph_model
    print(f"{label}:")
    print(f"  epochs={tc.n_epochs}  batch_size={tc.batch_size}  augmentation={tc.data_augmentation_loop}")
    print(f"  MLP layers={mc.n_layers}  hidden={mc.hidden_dim}  embedding_dim={mc.embedding_dim}")
    print(f"  lr_W={tc.learning_rate_W_start}  lr={tc.learning_rate_start}  lr_emb={tc.learning_rate_embedding_start}")
    print()

# %% [markdown]
# ## Step 1: Train
# For each noise condition we train the GNN on the pre-generated
# training data (`x_list_train`, `y_list_train`).  At each iteration
# a random frame $k$ is sampled from the 64,000 training time steps
# and the model predicts $\widehat{dv}/dt$ from the current voltages,
# stimulus, and graph structure.

# %%
#| echo: true
#| output: false
print()
print("=" * 80)
print("STEP 1: TRAIN - Training GNN on fly visual system data")
print("=" * 80)

for config_name, label in datasets:
    config = configs[config_name]
    graphs_dir = graphs_dirs[config_name]
    print()
    print(f"--- {label} ---")

    model_dir = os.path.join(graphs_dir, "models")
    model_exists = os.path.isdir(model_dir) and any(
        f.startswith("best_model") for f in os.listdir(model_dir)
    ) if os.path.isdir(model_dir) else False

    if model_exists:
        print(f"  trained model already exists at {model_dir}/")
        print("  skipping training...")
    else:
        print(f"  training on {config.simulation.n_frames} frames")
        print(f"  {config.training.n_epochs} epochs, batch_size={config.training.batch_size}")
        print()
        data_train(config, device=device)

# %% [markdown]
# ## Step 2: Test
# The trained model is evaluated on the held-out test data
# (`x_list_test`, `y_list_test`) in two modes:
#
# - **One-step prediction**: the model receives ground-truth voltages at
#   each frame and predicts the derivative — measuring instantaneous
#   accuracy.
# - **Rollout**: starting from the first test frame, the model
#   autoregressively predicts the full trajectory using only its own
#   previous outputs — measuring long-horizon stability.

# %%
#| echo: true
#| output: false
print()
print("=" * 80)
print("STEP 2: TEST - Evaluating on held-out test data")
print("=" * 80)

for config_name, label in datasets:
    config = configs[config_name]
    print()
    print(f"--- {label} ---")
    data_test(config, device=device)

# %% [markdown]
# ## Training Diagnostics
# Snapshots from the training of the noise-free model.

# %% [markdown]
# ### Learned Embeddings
# The 2D latent embedding $\mathbf{a}_i$ for each neuron, colored by
# cell type.  Neurons of the same type cluster together, showing that
# the GNN discovers cell-type identity from activity alone.

# %%
#| lightbox: true
graphs_dir_0 = graphs_dirs[datasets[0][0]]
emb_dir = os.path.join(graphs_dir_0, "tmp_training", "embedding")
if os.path.isdir(emb_dir):
    emb_files = sorted([f for f in os.listdir(emb_dir) if f.endswith('.png')])
    if emb_files:
        display_image(os.path.join(emb_dir, emb_files[-1]), width=700)

# %% [markdown]
# ### Connectivity Recovery
# Scatter plot of learned weights $\widehat{W}_{ij}$ vs. ground-truth
# weights $\mathbf{W}_{ij}$.  A linear fit shows the correlation after
# gain/bias correction.

# %%
#| lightbox: true
mat_dir = os.path.join(graphs_dir_0, "tmp_training", "matrix")
if os.path.isdir(mat_dir):
    mat_files = sorted([f for f in os.listdir(mat_dir) if f.startswith('comparison') and f.endswith('.png')])
    if mat_files:
        display_image(os.path.join(mat_dir, mat_files[-1]), width=700)

# %% [markdown]
# ### Learned Functions
# The edge message function $g_\phi$ (left) and the neuron update
# function $f_\theta$ (right) learned by the GNN.

# %%
#| lightbox: true
func_dir = os.path.join(graphs_dir_0, "tmp_training", "function", "MLP1")
if os.path.isdir(func_dir):
    func_files = sorted([f for f in os.listdir(func_dir) if f.endswith('.png')])
    if func_files:
        display_image(os.path.join(func_dir, func_files[-1]), width=700)

# %%
#| lightbox: true
func_dir = os.path.join(graphs_dir_0, "tmp_training", "function", "MLP0")
if os.path.isdir(func_dir):
    func_files = sorted([f for f in os.listdir(func_dir) if f.endswith('.png')])
    if func_files:
        display_image(os.path.join(func_dir, func_files[-1]), width=700)

# %% [markdown]
# ### Training Loss
# Total loss evolution over the training epochs.

# %%
#| lightbox: true
epoch_dir = os.path.join(graphs_dir_0, "tmp_training")
if os.path.isdir(epoch_dir):
    epoch_files = sorted([f for f in os.listdir(epoch_dir) if f.startswith('epoch_') and f.endswith('.png')])
    if epoch_files:
        display_image(os.path.join(epoch_dir, epoch_files[-1]), width=850)

# %% [markdown]
# ## Rollout Results
# Autoregressive rollout traces for selected neurons: ground truth
# (green) vs. GNN prediction (black).  The model receives only the
# initial state and stimulus, and predicts all subsequent time steps.

# %%
#| lightbox: true
for config_name, label in datasets:
    graphs_dir = graphs_dirs[config_name]
    rollout_path = os.path.join(graphs_dir, "rollout_selected.png")
    if os.path.isfile(rollout_path):
        print(f"### {label}")
        display_image(rollout_path, width=900)

# %% [markdown]
# ## References
#
# [1] J. K. Lappalainen et al., "Connectome-constrained networks predict
# neural activity across the fly visual system," *Nature*, 2024.
# [doi:10.1038/s41586-024-07939-3](https://doi.org/10.1038/s41586-024-07939-3)
#
# [2] J. Gilmer et al., "Neural Message Passing for Quantum Chemistry,"
# 2017.
# [doi:10.48550/arXiv.1704.01212](https://doi.org/10.48550/arXiv.1704.01212)
