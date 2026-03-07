# %% [raw]
# ---
# title: "GNN + INR: Joint Stimulus and Dynamics Recovery"
# author: "Allier, Lappalainen, Saalfeld"
# categories:
#   - FlyVis
#   - GNN
#   - INR
#   - SIREN
# execute:
#   echo: false
# image: "log/Claude_exploration/LLM_flyvis_noise_005_INR_siren/inr_comparison/iter_106_slot_01_siren_txy_comparison_60000.png"
# description: "Train a SIREN implicit neural representation (INR) jointly with the GNN to recover the visual stimulus field from neural activity alone. Discuss the inherent scale/offset degeneracy and the corrected R²."
# ---

# %% [markdown]
# ## Joint Stimulus and Dynamics Recovery with GNN + INR
#
# In the previous notebooks the visual stimulus $I_i(t)$ was provided
# as a known input to the GNN.  Here we ask: can the stimulus itself
# be recovered from neural activity alone?
#
# We replaced the ground-truth stimulus with a learnable **implicit
# neural representation** (INR), specifically a
# [SIREN](https://arxiv.org/abs/2006.09661) network, that maps
# continuous coordinates $(t, x, y)$ to the stimulus value at each
# neuron position and time step.  The SIREN was trained jointly with
# the GNN.  This amounted to solving a harder inverse problem: recovering not
# only the circuit parameters ($W$, $\tau$, $V^{\text{rest}}$,
# $f_\theta$, $g_\phi$) but also the stimulus field from voltage
# data alone.

# %%
#| output: false
import os
import sys
import glob
import warnings

from IPython.display import Image, Markdown, Video, display

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')
from GNN_PlotFigure import data_plot
from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.utils import set_device, add_pre_folder, log_path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)


def display_image(path, width=700):
    """Display a full-resolution image; width controls inline size (px)."""
    display(Image(filename=path, width=width))


# %% [markdown]
# ## SIREN Architecture
#
# The SIREN (Sinusoidal Representation Network) uses periodic
# activation functions $\phi(x) = \sin(\omega_0 \cdot x)$ instead
# of ReLU, enabling it to represent fine spatial and temporal
# detail in the stimulus field.
#
# The key hyperparameters explored by the agentic hyper-parameter optimization
# (Notebook 06) are:
#
# - **$\omega_0$** (frequency scaling): controls the spectral
#   bandwidth of the representation.  Higher $\omega_0$ allows the
#   network to capture faster temporal fluctuations and sharper
#   spatial edges.
# - **hidden_dim**: network width (number of hidden units per
#   layer).
# - **n_layers**: network depth.
# - **learning rate**: must scale inversely with $\omega_0$ for
#   stable training.
#
# The input is a 3D coordinate $(t, x, y)$ normalized to the
# training domain, and the output is a scalar stimulus value for
# each neuron at each time step.
#
# ### Scale/Offset Degeneracy and Corrected $R^2$
#
# The SIREN output enters the GNN through $f_\theta$, which receives
# the concatenated input $[v_i,\, \mathbf{a}_i,\, \text{msg}_i,\, I_i(t)]$.
# This creates an inherent **scale/offset degeneracy**: $f_\theta$'s
# biases absorb any constant offset, and its weights on the excitation
# dimension compensate any scale factor (including sign inversion).
# The SIREN and $f_\theta$ jointly optimize along a degenerate manifold
# where the stimulus *pattern* is learned correctly but the linear
# mapping between SIREN output and true stimulus is unidentifiable.
# We therefore apply a **global linear fit**
# $I^{\text{true}} = a \cdot I^{\text{pred}} + b$ and report the
# corrected $R^2$.

# %%
#| output: false
config_name = "flyvis_noise_005_INR"
config_file, pre_folder = add_pre_folder(config_name)
config = NeuralGraphConfig.from_yaml(f"./config/{config_file}.yaml")
config.dataset = pre_folder + config.dataset
config.config_file = pre_folder + config_name
gnn_log_dir = log_path(config.config_file)
device = set_device(config.training.device)

inr_log = "log/Claude_exploration/LLM_flyvis_noise_005_INR_siren"

# Check that results exist
if not os.path.isdir(inr_log):
    raise RuntimeError(
        f"INR SIREN results not found at: {inr_log}. "
        f"Run the agentic optimization for flyvis_noise_005_INR_siren first."
    )

inr_video_dir = os.path.join(inr_log, "inr_video")
inr_comparison_dir = os.path.join(inr_log, "inr_comparison")
inr_results_dir = os.path.join(inr_log, "results")

# %% [markdown]
# ## Stimulus Recovery Video
#
# The video below shows the best SIREN result (iter 106,
# $R^2$=0.824) with three panels:
#
# - **Left**: ground-truth stimulus on the hexagonal photoreceptor
#   array.
# - **Center**: SIREN prediction after global linear correction
#   ($I^{\text{true}} = a \cdot I^{\text{pred}} + b$).
# - **Right**: rolling voltage traces for 10 selected neurons
#   (ground truth in green, prediction in black).
#
# The linear correction coefficients $a$ and $b$ and the per-frame
# RMSE are displayed in the trace panel.

# %%
import glob, re as _re

# Find the latest stimulus video (highest iteration number)
_video_pattern = os.path.join(inr_video_dir, "iter_*_stimulus_gt_vs_pred.mp4")
_video_files = sorted(glob.glob(_video_pattern))
if _video_files:
    video_path = _video_files[-1]
    _m = _re.search(r'iter_(\d+)_slot_(\d+)', os.path.basename(video_path))
    if _m:
        best_iter, best_slot = int(_m.group(1)), int(_m.group(2))
    display(Video(video_path, embed=True, width=800))
else:
    display(Markdown(f"*No stimulus video found in `{inr_video_dir}`*"))

# %% [markdown]
# ## Best Result Metrics
#
# The SIREN hyperparameters were optimized over 116 iterations
# by the agentic hyper-parameter optimization framework ([Notebook 07](Notebook_07.py)).
# The best configuration (iter 106: hidden_dim=512, 7 layers,
# $\omega_0$=1024, lr=$3\times10^{-7}$) achieves $R^2$ = 0.82
# on the full 64,000-frame stimulus sequence.


# %% [markdown]
# ## GNN Analysis: Learned Representations
#
# Beyond stimulus recovery, the joint GNN+SIREN model also learns
# synaptic weights, neural embeddings, and MLP functions.  Below we
# run the same analysis as [Notebook 04](Notebook_04.py) on the
# joint model to verify that circuit recovery is preserved.

# %%
#| echo: true
#| output: false
print("\n--- Generating GNN analysis plots for noise_005_INR ---")
data_plot(
    config=config,
    config_file=config.config_file,
    epoch_list=['best'],
    style='color',
    extended='plots',
    device=device,
)

# %%
#| output: false
config_indices = config_name.replace('flyvis_', '')

def show_result(filename, width=600):
    path = os.path.join(gnn_log_dir, "results", filename.format(idx=config_indices))
    if os.path.isfile(path):
        display_image(path, width=width)

def show_mlp(mlp_name, suffix=""):
    path = os.path.join(gnn_log_dir, "results", f"{mlp_name}_{config_indices}{suffix}.png")
    if os.path.isfile(path):
        display_image(path, width=700)

# %% [markdown]
# ### Corrected Weights ($W$)

# %%
#| lightbox: true
show_result("weights_comparison_corrected.png")

# %% [markdown]
# ### $f_\theta$ (MLP$_0$): Neuron Update Function

# %%
#| lightbox: true
show_mlp("MLP0", "_domain")

# %% [markdown]
# ### Time Constants ($\tau$)

# %%
#| lightbox: true
show_result("tau_comparison_{idx}.png", width=500)

# %% [markdown]
# ### Resting Potentials ($V^{\text{rest}}$)

# %%
#| lightbox: true
show_result("V_rest_comparison_{idx}.png", width=500)

# %% [markdown]
# ### $g_\phi$ (MLP$_1$): Edge Message Function

# %%
#| lightbox: true
show_mlp("MLP1", "_domain")

# %% [markdown]
# ### Neural Embeddings

# %%
#| lightbox: true
show_result("embedding_{idx}.png")

# %% [markdown]
# ### UMAP Projections

# %%
#| lightbox: true
show_result("embedding_augmented_{idx}.png")

# %% [markdown]
# ### Spectral Analysis

# %%
#| lightbox: true
show_result("eigen_comparison.png", width=900)

# %% [markdown]
# ## References
#
# [1] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell,
# and G. Wetzstein, "Implicit Neural Representations with Periodic
# Activation Functions," *NeurIPS*, 2020.
# [doi:10.48550/arXiv.2006.09661](https://doi.org/10.48550/arXiv.2006.09661)
#
# [2] C. Allier, L. Heinrich, M. Schneider, S. Saalfeld, "Graph
# neural networks uncover structure and functions underlying the
# activity of simulated neural assemblies," *arXiv:2602.13325*,
# 2026.
# [doi:10.48550/arXiv.2602.13325](https://doi.org/10.48550/arXiv.2602.13325)
