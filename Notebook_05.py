# %% [raw]
# ---
# title: "Robustness to False Positive Edges"
# author: "Allier, Lappalainen, Saalfeld"
# categories:
#   - FlyVis
#   - GNN
#   - Robustness
# execute:
#   echo: false
# description: "Test GNN robustness to false positive edges by adding 50–400% extra null edges with zero ground-truth weight and evaluating whether L1 regularization successfully suppresses them during training."
# ---

# %% [markdown]
# ## Robustness to False Positive Edges
#
# In practice, connectomes derived from electron microscopy reconstructions
# may contain **false positive synapses** — edges that appear in the
# connectivity graph but carry no functional weight.  Can the GNN
# identify and reject these spurious connections?
#
# To test this, we augment the true connectivity (434,112 edges) with
# **extra null edges**: randomly sampled neuron pairs that are not in
# the original connectome, initialized with zero ground-truth weight.
# The ODE simulation uses only the real edges, so the training signal
# contains no information from null edges.  The model must learn to
# keep them at zero while recovering the true synaptic weights.
#
# We test four levels of contamination on the low-noise ($\sigma=0.05$)
# configuration:
#
# | Label | Extra null edges | Total edges | Ratio to original |
# |-------|-----------------|-------------|-------------------|
# | 50%   | 217,056         | 651,168     | 1.5×              |
# | 100%  | 434,112         | 868,224     | 2×                |
# | 200%  | 868,224         | 1,302,336   | 3×                |
# | 400%  | 1,736,448       | 2,170,560   | 5×                |
#
# The key mechanism enabling false-positive rejection is **L1
# regularization** on the weight vector $W$
# (`coeff_W_L1`), which penalizes all edge weights equally and
# drives unused edges toward zero.

# %%
#| output: false
import os
import sys
import re
import glob
import warnings

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')
from GNN_PlotFigure import data_plot
from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.utils import set_device, add_pre_folder, log_path, graphs_data_path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)


def display_image(path, width=700):
    """Display a full-resolution image; width controls inline size (px)."""
    display(Image(filename=path, width=width))


# %% [markdown]
# ## Configuration

# %%
#| output: false

# Baseline (0% null edges) and four contamination levels
datasets = [
    ('flyvis_noise_005',                '0%'),
    ('flyvis_noise_005_null_edges_50',  '50%'),
    ('flyvis_noise_005_null_edges_100', '100%'),
    ('flyvis_noise_005_null_edges_200', '200%'),
    ('flyvis_noise_005_null_edges_400', '400%'),
]

config_root = "./config"
configs = {}
log_dirs = {}

for config_name, label in datasets:
    config_file, pre_folder = add_pre_folder(config_name)
    config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
    config.dataset = pre_folder + config.dataset
    config.config_file = pre_folder + config_name
    configs[config_name] = config
    log_dirs[config_name] = log_path(config.config_file)

device = set_device(configs[datasets[0][0]].training.device)

# %% [markdown]
# ## Generate Analysis Plots
#
# For each configuration we load the best model checkpoint and
# generate the standard results visualizations.

# %%
#| echo: true
#| output: false
print()
print("=" * 80)
print("ANALYSIS — Generating results plots for null-edge experiments")
print("=" * 80)

for config_name, label in datasets:
    config = configs[config_name]
    ldir = log_dirs[config_name]
    if not glob.glob(f"{ldir}/models/best_model_with_*.pt"):
        print(f"  Skipping {label} ({config_name}): no trained model found")
        continue
    print(f"\n--- {label} null edges ({config_name}) ---")
    data_plot(
        config=config,
        config_file=config.config_file,
        epoch_list=['best'],
        style='color',
        extended='plots',
        device=device,
    )


# %% [markdown]
# ## Metrics vs Null Edge Fraction
#
# The plot below shows how key recovery metrics degrade (or not) as
# the fraction of false positive edges increases from 0% to 400%.
# Metrics are read from the training log (`metrics.log`) and test
# results (`results_test.log`, `results_rollout.log`).

# %%

def read_final_metrics(ldir):
    """Read the last line of metrics.log → dict."""
    metrics_path = os.path.join(ldir, 'tmp_training', 'metrics.log')
    if not os.path.isfile(metrics_path):
        return {}
    last_line = ''
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('iteration'):
                last_line = line
    if not last_line:
        return {}
    parts = last_line.split(',')
    return {
        'conn_r2': float(parts[1]),
        'vrest_r2': float(parts[2]),
        'tau_r2': float(parts[3]),
    }


def parse_log_file(path):
    """Parse a results log file into a flat dict of key → value."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'^([\w\s]+):\s*([\d.e+-]+)', line)
            if m:
                out[m.group(1).strip()] = float(m.group(2).strip())
    return out


def read_test_metrics(ldir):
    d = parse_log_file(os.path.join(ldir, 'results_test.log'))
    return {
        'test_rmse': d.get('RMSE', np.nan),
        'test_pearson': d.get('Pearson r', np.nan),
        'test_conn_r2': d.get('connectivity_R2', np.nan),
        'test_tau_r2': d.get('tau_R2', np.nan),
        'test_vrest_r2': d.get('V_rest_R2', np.nan),
    }


def read_rollout_metrics(ldir):
    d = parse_log_file(os.path.join(ldir, 'results_rollout.log'))
    return {
        'rollout_rmse': d.get('RMSE', np.nan),
        'rollout_pearson': d.get('Pearson r', np.nan),
    }


# Collect metrics for all configs
labels = []
null_pcts = [0, 50, 100, 200, 400]
all_metrics = []

for (config_name, label), pct in zip(datasets, null_pcts):
    ldir = log_dirs[config_name]
    m = {}
    m.update(read_final_metrics(ldir))
    m.update(read_test_metrics(ldir))
    m.update(read_rollout_metrics(ldir))
    m['pct'] = pct
    m['label'] = label
    all_metrics.append(m)

# %%

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

metric_keys = [
    ('conn_r2',         'Connectivity $R^2$'),
    ('tau_r2',          r'$\tau$ $R^2$'),
    ('vrest_r2',        r'$V_{\mathrm{rest}}$ $R^2$'),
    ('rollout_pearson', 'Rollout Pearson $r$'),
]

for ax, (key, title) in zip(axes, metric_keys):
    vals = [m.get(key, np.nan) for m in all_metrics]
    ax.plot(null_pcts, vals, 'o-', color='#2ca02c', markersize=8, linewidth=2)
    ax.set_xlabel('Extra null edges (%)')
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.set_xticks(null_pcts)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

fig.suptitle('Recovery metrics vs false positive edge fraction', fontsize=14, y=1.02)
plt.tight_layout()

plot_path = os.path.join('log', 'null_edges_metrics.png')
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved to {plot_path}")


# %% [markdown]
# ## Connectivity Recovery
#
# The scatter plots below compare learned vs ground-truth synaptic
# weights for each null-edge level.  Even with 400% extra false
# edges, the model should recover the true connectivity — the null
# edges cluster near zero weight.

# %%
#| output: false
def show_result(filename, config_name, width=600):
    log_dir = log_dirs[config_name]
    config_indices = config_name.replace('flyvis_', '')
    path = os.path.join(log_dir, "results", filename.format(idx=config_indices))
    if os.path.isfile(path):
        display_image(path, width=width)

# %% [markdown]
# ### Baseline (0% null edges)
# %%
#| lightbox: true
show_result("weights_comparison_corrected.png", "flyvis_noise_005")

# %% [markdown]
# ### 50% null edges
# %%
#| lightbox: true
show_result("weights_comparison_corrected.png", "flyvis_noise_005_null_edges_50")

# %% [markdown]
# ### 100% null edges
# %%
#| lightbox: true
show_result("weights_comparison_corrected.png", "flyvis_noise_005_null_edges_100")

# %% [markdown]
# ### 200% null edges
# %%
#| lightbox: true
show_result("weights_comparison_corrected.png", "flyvis_noise_005_null_edges_200")

# %% [markdown]
# ### 400% null edges
# %%
#| lightbox: true
show_result("weights_comparison_corrected.png", "flyvis_noise_005_null_edges_400")


# %% [markdown]
# ## Rollout: Predicted vs Ground-Truth Activity
#
# Selected rollout traces for the baseline and the most extreme
# (400%) null-edge case, confirming that the GNN still captures the
# dynamics correctly despite the contaminated graph.

# %%
#| output: false
def show_rollout(config_name, width=800):
    log_dir = log_dirs[config_name]
    pattern = os.path.join(log_dir, "results", "rollout_*_DAVIS_selected.png")
    files = sorted(glob.glob(pattern))
    if files:
        display_image(files[0], width=width)

# %% [markdown]
# ### Baseline (0%)
# %%
#| lightbox: true
show_rollout("flyvis_noise_005")

# %% [markdown]
# ### 400% null edges
# %%
#| lightbox: true
show_rollout("flyvis_noise_005_null_edges_400")


# %% [markdown]
# ## Summary
#
# The GNN combined with L1 regularization on $W$ successfully
# rejects false positive edges.  Even when the graph is contaminated
# with up to 400% extra null edges (5× the original edge count),
# the model drives their weights to zero and recovers the true
# connectivity.  This demonstrates that the framework is robust to
# over-complete or noisy connectomes — a property relevant to
# real-world EM reconstructions where false synapses are common.
