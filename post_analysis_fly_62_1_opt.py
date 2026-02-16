"""
Test+plot flyvis_62_1_opt at every epoch checkpoint (0..19)
and append all results into flyvis_62_1_opt_results.md

Usage:
    python run_fly_62_1_opt.py
"""

import os
import sys
import re
import glob
import warnings
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.utils import set_device, add_pre_folder
from GNN_PlotFigure import data_plot

CONFIG_NAME = 'flyvis_62_1_opt'
MD_FILE = 'flyvis_62_1_opt_results.md'


def load_config():
    config_file, pre_folder = add_pre_folder(CONFIG_NAME)
    config = NeuralGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    config.dataset = pre_folder + config.dataset
    config.config_file = pre_folder + CONFIG_NAME
    return config, config_file


def parse_results_log(log_dir):
    """Parse results.log and extract key metrics."""
    results_path = os.path.join(log_dir, 'results.log')
    if not os.path.exists(results_path):
        return {}

    metrics = {}
    with open(results_path, 'r') as f:
        content = f.read()

    m = re.search(r'tau reconstruction R²:\s*([\d.]+)\s+slope:\s*([\d.-]+)', content)
    if m:
        metrics['tau_R2'] = float(m.group(1))

    m = re.search(r'V_rest reconstruction R²:\s*([\d.]+)\s+slope:\s*([\d.-]+)', content)
    if m:
        metrics['V_rest_R2'] = float(m.group(1))

    m = re.search(r'first weights fit R²:\s*([\d.-]+)\s+slope:\s*([\d.-]+)', content)
    if m:
        metrics['conn_R2_raw'] = float(m.group(1))

    m = re.search(r'second weights fit R²:\s*([\d.-]+)\s+slope:\s*([\d.-]+)', content)
    if m:
        metrics['conn_R2_corrected'] = float(m.group(1))

    return metrics


def main():
    device = set_device('auto')
    print(f'device: {device}')

    config, config_file = load_config()
    log_dir = f'./log/fly/{CONFIG_NAME}'
    model_dir = os.path.join(log_dir, 'models')

    # Find epoch-level checkpoints (e.g. best_model_with_0_graphs_5.pt, not _5_31999.pt)
    epoch_checkpoints = {}
    for f in sorted(glob.glob(f'{model_dir}/best_model_with_0_graphs_*.pt')):
        suffix = os.path.basename(f).replace('.pt', '').split('graphs_')[1]
        if '_' not in suffix:
            epoch_checkpoints[int(suffix)] = suffix

    print(f'Found {len(epoch_checkpoints)} epoch checkpoints: {sorted(epoch_checkpoints.keys())}')

    # Write markdown header
    with open(MD_FILE, 'w') as md:
        md.write(f'# flyvis_62_1_opt — epoch-by-epoch results\n\n')
        md.write(f'Config: {CONFIG_NAME}, n_epochs={config.training.n_epochs}, ')
        md.write(f'batch_size={config.training.batch_size}, ')
        md.write(f'data_augmentation_loop={config.training.data_augmentation_loop}\n\n')
        md.write(f'| epoch | tau_R2 | V_rest_R2 | conn_R2_raw | conn_R2_corrected |\n')
        md.write(f'|-------|--------|-----------|-------------|-------------------|\n')

    for epoch in sorted(epoch_checkpoints.keys()):
        epoch_str = epoch_checkpoints[epoch]
        print(f'\n=== Plotting epoch {epoch} (model: {epoch_str}) ===')

        config, config_file = load_config()

        data_plot(
            config=config,
            config_file=config_file,
            epoch_list=[epoch_str],
            style='white color',
            extended='plots',
            device=device
        )

        metrics = parse_results_log(log_dir)

        tau = metrics.get('tau_R2', 'N/A')
        vrest = metrics.get('V_rest_R2', 'N/A')
        conn_raw = metrics.get('conn_R2_raw', 'N/A')
        conn_corr = metrics.get('conn_R2_corrected', 'N/A')

        fmt = lambda v: f'{v:.4f}' if isinstance(v, float) else v

        print(f'  epoch {epoch}: tau={fmt(tau)}, V_rest={fmt(vrest)}, conn_raw={fmt(conn_raw)}, conn_corr={fmt(conn_corr)}')

        with open(MD_FILE, 'a') as md:
            md.write(f'| {epoch} | {fmt(tau)} | {fmt(vrest)} | {fmt(conn_raw)} | {fmt(conn_corr)} |\n')

    print(f'\n=== Results written to {MD_FILE} ===')


if __name__ == '__main__':
    main()
