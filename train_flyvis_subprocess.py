#!/usr/bin/env python3
"""
Standalone flyvis training+test+plot script for subprocess execution.

This script is called by GNN_LLM_parallel_flyvis.py as a subprocess to ensure that any code
modifications to graph_trainer.py and GNN_PlotFigure.py are reloaded for each iteration.

Usage:
    python train_flyvis_subprocess.py --config CONFIG_PATH --device DEVICE [--erase] [--log_file LOG_PATH]
"""

import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports

import argparse
import glob
import shutil
import sys
import os
import traceback

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.models.graph_trainer import data_train, data_test
from flyvis_gnn.utils import set_device, log_path
from GNN_PlotFigure import data_plot


def main():
    parser = argparse.ArgumentParser(description='Train+test+plot flyvis GNN')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--erase', action='store_true', help='Erase existing log files')
    parser.add_argument('--log_file', type=str, default=None, help='Path to analysis log file')
    parser.add_argument('--config_file', type=str, default=None, help='Config file name for log directory')
    parser.add_argument('--error_log', type=str, default=None, help='Path to error log file')
    parser.add_argument('--best_model', type=str, default=None, help='Best model path')
    parser.add_argument('--generate', action='store_true', help='Regenerate data before training')
    parser.add_argument('--seed', type=int, default=None, help='Override simulation seed (for data variance)')
    parser.add_argument('--exploration_dir', type=str, default=None, help='Copy models here after training')
    parser.add_argument('--iteration', type=int, default=None, help='Iteration number (for model naming)')
    parser.add_argument('--slot', type=int, default=None, help='Slot number (for model naming)')

    args = parser.parse_args()

    # Open error log file if specified
    error_log = None
    if args.error_log:
        try:
            error_log = open(args.error_log, 'w')
        except Exception as e:
            print(f"Warning: Could not open error log file: {e}", file=sys.stderr)

    try:
        # Load config
        config = NeuralGraphConfig.from_yaml(args.config)

        # Set config_file if provided (needed for proper log directory path)
        # DO NOT change dataset — data is pre-generated, dataset must point to original directory
        if args.config_file:
            config.config_file = args.config_file
            pre_folder = os.path.dirname(args.config_file)
            if pre_folder:
                pre_folder += '/'
            config.dataset = pre_folder + config.dataset

        # Override seed if specified
        if args.seed is not None:
            config.simulation.seed = args.seed

        # Set device
        device = set_device(args.device)

        # Phase 0: Generate data (if requested)
        if args.generate:
            from flyvis_gnn.generators.graph_data_generator import data_generate
            print(f"Generating data with seed={config.simulation.seed} ...")
            data_generate(
                config=config,
                device=device,
                visualize=False,
                run_vizualized=0,
                style="color",
                alpha=1,
                erase=True,
                save=True,
                step=100,
            )

        # Suppress iteration-level model saves
        config.training.save_all_checkpoints = False

        # Open log file if specified
        log_file = None
        if args.log_file:
            log_file = open(args.log_file, 'w')

        try:
            # Phase 1: Train
            data_train(
                config=config,
                erase='True',
                best_model='',
                style='color',
                device=device,
                log_file=log_file
            )

            # Phase 2: Test (with no noise for evaluation)
            config.simulation.noise_model_level = 0.0
            data_test(
                config=config,
                visualize=False,
                style="color name continuous_slice",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=10,
                n_rollout_frames=1000,
                device=device,
                particle_of_interest=0,
                new_params=None,
                log_file=log_file,
            )

            # Phase 3: Plot
            config_file = args.config_file if args.config_file else config.dataset
            folder_name = log_path(os.path.dirname(config_file), 'tmp_results') + '/'
            os.makedirs(folder_name, exist_ok=True)
            data_plot(
                config=config,
                config_file=config_file,
                epoch_list=['best'],
                style='color',
                extended='plots',
                device=device,
                log_file=log_file
            )

            # Phase 4: Copy models to exploration dir
            if args.exploration_dir is not None and args.iteration is not None and args.slot is not None:
                log_dir = log_path(config.config_file)
                src_models = glob.glob(os.path.join(log_dir, 'models', '*.pt'))
                if src_models:
                    dst_dir = os.path.join(args.exploration_dir, 'models')
                    os.makedirs(dst_dir, exist_ok=True)
                    for src in src_models:
                        fname = os.path.basename(src)
                        dst = os.path.join(dst_dir, f'iter_{args.iteration:03d}_slot_{args.slot:02d}_{fname}')
                        shutil.copy2(src, dst)
                        print(f"copied model: {dst}")

        finally:
            if log_file:
                log_file.close()

    except Exception as e:
        # Capture full traceback for debugging
        error_msg = f"\n{'='*80}\n"
        error_msg += "FLYVIS SUBPROCESS ERROR\n"
        error_msg += f"{'='*80}\n\n"
        error_msg += f"Error Type: {type(e).__name__}\n"
        error_msg += f"Error Message: {str(e)}\n\n"
        error_msg += "Full Traceback:\n"
        error_msg += traceback.format_exc()
        error_msg += f"\n{'='*80}\n"

        # Print to stderr
        print(error_msg, file=sys.stderr, flush=True)

        # Write to error log if available
        if error_log:
            error_log.write(error_msg)
            error_log.flush()

        # Exit with non-zero code
        sys.exit(1)

    finally:
        if error_log:
            error_log.close()


if __name__ == '__main__':
    main()
