"""Benchmark: edge removal experiment (false negatives).

Runs generate → train → test → plot for each removal config.
Simulation uses the full connectome; edges are removed before saving.
Two modes: random and per_column, at 2%, 5%, 10%, 20% removal.
Also runs the baseline (0% removal) for comparison.

Usage:
    python run_edge_removal.py random    # random removal
    python run_edge_removal.py pc        # per-column removal
    python run_edge_removal.py           # defaults to random

Monitor progress:
    watch -n 10 column -t -s, log/edge_removal_results.csv
"""
import sys
import os
import csv
import time
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.generators.graph_data_generator import data_generate
from flyvis_gnn.models.graph_trainer import data_train, data_test
from flyvis_gnn.utils import set_device, add_pre_folder, log_path, config_path

# ── Configuration ─────────────────────────────────────────────────

MODE = sys.argv[1] if len(sys.argv) > 1 else 'random'
assert MODE in ('random', 'pc'), f"Usage: python run_edge_removal.py [random|pc]"

# Baseline + 4 removal levels
if MODE == 'random':
    CONFIGS = [
        ('flyvis_noise_005_removed_02', 'rm_02', 0.02),
        ('flyvis_noise_005_removed_05', 'rm_05', 0.05),
        ('flyvis_noise_005_removed_10', 'rm_10', 0.10),
        ('flyvis_noise_005_removed_20', 'rm_20', 0.20),
    ]
    OUT_CSV = os.path.join('log', 'edge_removal_random_results.csv')
else:
    CONFIGS = [
        ('flyvis_noise_005_removed_pc_02', 'rm_pc_02', 0.02),
        ('flyvis_noise_005_removed_pc_05', 'rm_pc_05', 0.05),
        ('flyvis_noise_005_removed_pc_10', 'rm_pc_10', 0.10),
        ('flyvis_noise_005_removed_pc_20', 'rm_pc_20', 0.20),
    ]
    OUT_CSV = os.path.join('log', 'edge_removal_pc_results.csv')

SEED_PAIRS = [
    (1000, 1500),
    (2000, 2500),
    (3000, 3500),
    (4000, 4500),
]

FIELDNAMES = [
    'config', 'label', 'mode', 'removal_pct', 'seed_idx', 'sim_seed', 'train_seed',
    'n_edges',
    # training R2 (from metrics.log)
    'conn_r2', 'vrest_r2', 'tau_r2',
    # one-step test metrics
    'test_rmse', 'test_pearson', 'test_conn_r2', 'test_tau_r2',
    'test_vrest_r2', 'test_cluster_acc',
    # rollout metrics
    'rollout_rmse', 'rollout_pearson',
    # timing
    'gen_time_min', 'train_time_min', 'test_time_min',
    'error',
]


# ── Helpers ───────────────────────────────────────────────────────

def read_final_metrics(ldir):
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
    return {'conn_r2': parts[1], 'vrest_r2': parts[2], 'tau_r2': parts[3]}


def parse_log_file(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'^([\w\s]+):\s*([\d.e+-]+)', line)
            if m:
                out[m.group(1).strip()] = m.group(2).strip()
    return out


def read_test_metrics(ldir):
    d = parse_log_file(os.path.join(ldir, 'results_test.log'))
    return {
        'test_rmse': d.get('RMSE', ''),
        'test_pearson': d.get('Pearson r', ''),
        'test_conn_r2': d.get('connectivity_R2', ''),
        'test_tau_r2': d.get('tau_R2', ''),
        'test_vrest_r2': d.get('V_rest_R2', ''),
        'test_cluster_acc': d.get('cluster_accuracy', ''),
    }


def read_rollout_metrics(ldir):
    d = parse_log_file(os.path.join(ldir, 'results_rollout.log'))
    return {'rollout_rmse': d.get('RMSE', ''), 'rollout_pearson': d.get('Pearson r', '')}


def write_csv(results):
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results):
    if not results:
        return
    print(f'\n{"─"*110}')
    print(f'{"config":<40} {"rm%":>5} {"seed":>4} '
          f'{"conn_r2":>8} {"tau_r2":>8} {"vrest_r2":>8} '
          f'{"test_pearson":>12} {"rollout_pearson":>15} '
          f'{"train_min":>9}')
    print(f'{"─"*110}')
    for r in results:
        print(f'{r["config"]:<40} {r["removal_pct"]:>5} {r["seed_idx"]:>4} '
              f'{r.get("conn_r2",""):>8} {r.get("tau_r2",""):>8} {r.get("vrest_r2",""):>8} '
              f'{r.get("test_pearson",""):>12} {r.get("rollout_pearson",""):>15} '
              f'{r.get("train_time_min",""):>9}')
    print(f'{"─"*110}\n')


# ── Main ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = None
    results = []
    write_csv(results)

    print(f'Edge removal experiment — mode: {MODE}')
    print(f'Output: {OUT_CSV}\n')

    for config_name, label, removal_pct in CONFIGS:
        config_file, pre_folder = add_pre_folder(config_name)
        base_config = NeuralGraphConfig.from_yaml(config_path(f"{config_file}.yaml"))

        print(f'\n{"="*70}')
        print(f'Config: {config_name}  (removal={removal_pct*100:.0f}%)')
        print(f'{"="*70}')

        for seed_idx, (sim_seed, train_seed) in enumerate(SEED_PAIRS):
            row = {k: '' for k in FIELDNAMES}
            row.update({
                'config': config_name, 'label': label, 'mode': MODE,
                'removal_pct': removal_pct, 'seed_idx': seed_idx,
                'sim_seed': sim_seed, 'train_seed': train_seed,
                'n_edges': base_config.simulation.n_edges,
            })

            config = base_config.model_copy(deep=True)
            config.dataset = f"{pre_folder}{base_config.dataset}_{seed_idx:02d}"
            config.config_file = f"{pre_folder}{config_name}_er_{seed_idx:02d}"
            config.simulation.seed = sim_seed
            config.training.seed = train_seed

            if device is None:
                device = set_device(config.training.device)

            print(f'\n  ── Seed {seed_idx} (sim={sim_seed}, train={train_seed}) ──')

            # 1. Generate (simulates with full connectome, then removes edges)
            print(f'  [generate] ...', flush=True)
            t0 = time.time()
            try:
                data_generate(config=config, device=device, visualize=False,
                              save=True, erase=True, step=100)
            except Exception as e:
                print(f'  [generate] ERROR: {e}')
                row['error'] = f'generate: {e}'
                results.append(row)
                write_csv(results)
                continue
            gen_time = (time.time() - t0) / 60.0
            row['gen_time_min'] = f'{gen_time:.1f}'
            print(f'  [generate] done in {gen_time:.1f} min')

            # 2. Train
            print(f'  [train] ...', flush=True)
            t0 = time.time()
            try:
                data_train(config=config, erase=True, best_model=None,
                           style='color', device=device)
            except Exception as e:
                print(f'  [train] ERROR: {e}')
                row['error'] = f'train: {e}'
                results.append(row)
                write_csv(results)
                continue
            train_time = (time.time() - t0) / 60.0
            row['train_time_min'] = f'{train_time:.1f}'

            ldir = log_path(config.config_file)
            row.update(read_final_metrics(ldir))
            print(f'  [train] done in {train_time:.1f} min — '
                  f'conn={row.get("conn_r2","?")}, '
                  f'tau={row.get("tau_r2","?")}, '
                  f'vrest={row.get("vrest_r2","?")}')

            # 3. Test + rollout
            print(f'  [test] ...', flush=True)
            t0 = time.time()
            try:
                data_test(config=config, best_model='best', device=device)
            except Exception as e:
                print(f'  [test] ERROR: {e}')
                row['error'] = f'test: {e}'
                results.append(row)
                write_csv(results)
                continue
            test_time = (time.time() - t0) / 60.0
            row['test_time_min'] = f'{test_time:.1f}'

            row.update(read_test_metrics(ldir))
            row.update(read_rollout_metrics(ldir))
            print(f'  [test] done in {test_time:.1f} min — '
                  f'test_pearson={row.get("test_pearson","?")}, '
                  f'rollout_pearson={row.get("rollout_pearson","?")}')

            # 4. Update CSV
            results.append(row)
            write_csv(results)
            print_summary(results)

    print(f'\n\nEdge removal experiment complete. Results saved to {OUT_CSV}')
    print(f'Total runs: {len(results)}')
