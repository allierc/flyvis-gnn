"""Benchmark bis: 5 seeds × optimized only × noise_005 + noise_05.

Re-run with corrected configs (from exploration review).
The CSV is updated after every single run so progress can be monitored.

Usage:
    python run_benchmark_bis.py

Monitor progress:
    watch -n 10 column -t -s, log/benchmark_results_bis.csv
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

N_SEEDS = 5

# Only the two corrected optimized configs
CONFIGS = [
    ('flyvis_noise_005',  'optimized'),
    ('flyvis_noise_05',   'optimized'),
]

# Seeds: pairs of (simulation_seed, training_seed)
SEED_PAIRS = [
    (1000, 1500),
    (2000, 2500),
    (3000, 3500),
    (4000, 4500),
    (5000, 5500),
]

OUT_CSV = os.path.join('log', 'benchmark_results.csv')

FIELDNAMES = [
    'config', 'label', 'noise', 'seed_idx', 'sim_seed', 'train_seed',
    # training R2 (from metrics.log)
    'conn_r2', 'vrest_r2', 'tau_r2',
    # one-step test metrics (from results_test.log)
    'test_rmse', 'test_pearson', 'test_conn_r2', 'test_tau_r2',
    'test_vrest_r2', 'test_cluster_acc',
    # rollout metrics (from results_rollout.log)
    'rollout_rmse', 'rollout_pearson',
    # timing
    'gen_time_min', 'train_time_min', 'test_time_min',
    'error',
]


# ── Helpers ───────────────────────────────────────────────────────

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
        'conn_r2': parts[1],
        'vrest_r2': parts[2],
        'tau_r2': parts[3],
    }


def parse_log_file(path):
    """Parse a results log file into a flat dict of key → value string."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            # "RMSE: 0.1234 +/- 0.0567" → take the mean part
            m = re.match(r'^([\w\s]+):\s*([\d.e+-]+)', line)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                out[key] = val
    return out


def read_test_metrics(ldir):
    """Read results_test.log → dict with test_ prefix."""
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
    """Read results_rollout.log → dict with rollout_ prefix."""
    d = parse_log_file(os.path.join(ldir, 'results_rollout.log'))
    return {
        'rollout_rmse': d.get('RMSE', ''),
        'rollout_pearson': d.get('Pearson r', ''),
    }


def write_csv(results):
    """Write the full results list to CSV (overwrites)."""
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results):
    """Print a compact progress table to stdout."""
    if not results:
        return
    print(f'\n{"─"*100}')
    print(f'{"config":<30} {"label":<10} {"seed":<5} '
          f'{"conn_r2":>8} {"tau_r2":>8} {"vrest_r2":>8} '
          f'{"test_pearson":>12} {"rollout_pearson":>15} '
          f'{"train_min":>9}')
    print(f'{"─"*100}')
    for r in results:
        print(f'{r["config"]:<30} {r["label"]:<10} {r["seed_idx"]:<5} '
              f'{r.get("conn_r2",""):>8} {r.get("tau_r2",""):>8} {r.get("vrest_r2",""):>8} '
              f'{r.get("test_pearson",""):>12} {r.get("rollout_pearson",""):>15} '
              f'{r.get("train_time_min",""):>9}')
    print(f'{"─"*100}\n')


# ── Main ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = None
    results = []

    # Load existing results from CSV (appending mode)
    if os.path.isfile(OUT_CSV):
        with open(OUT_CSV, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
        print(f'Loaded {len(results)} existing rows from {OUT_CSV}')

    for config_name, label in CONFIGS:
        config_file, pre_folder = add_pre_folder(config_name)
        base_config = NeuralGraphConfig.from_yaml(config_path(f"{config_file}.yaml"))

        noise_level = float(base_config.simulation.noise_model_level)
        print(f'\n{"="*70}')
        print(f'Config: {config_name}  ({label}, noise={noise_level})')
        print(f'  n_epochs={base_config.training.n_epochs}  '
              f'aug={base_config.training.data_augmentation_loop}  '
              f'bs={base_config.training.batch_size}  '
              f'n_layers={base_config.graph_model.n_layers}')
        print(f'{"="*70}')

        for seed_idx, (sim_seed, train_seed) in enumerate(SEED_PAIRS):
            row = {k: '' for k in FIELDNAMES}
            row.update({
                'config': config_name, 'label': label,
                'noise': noise_level, 'seed_idx': seed_idx,
                'sim_seed': sim_seed, 'train_seed': train_seed,
            })

            # Deep-copy and override per-seed settings
            config = base_config.model_copy(deep=True)
            config.dataset = f"{pre_folder}{base_config.dataset}_{seed_idx:02d}"
            config.config_file = f"{pre_folder}{config_name}_bis_{seed_idx:02d}"
            config.simulation.seed = sim_seed
            config.training.seed = train_seed

            if device is None:
                device = set_device(config.training.device)

            print(f'\n  ── Seed {seed_idx} (sim={sim_seed}, train={train_seed}) ──')

            # ── 1. Generate data ──────────────────────────────────
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

            # ── 2. Train ──────────────────────────────────────────
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

            # ── 3. Test + rollout ─────────────────────────────────
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

            # ── 4. Update CSV and print summary ───────────────────
            results.append(row)
            write_csv(results)
            print_summary(results)

    print(f'\n\nBenchmark bis complete. Results saved to {OUT_CSV}')
    print(f'Total runs: {len(results)}')
