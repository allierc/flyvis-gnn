#!/usr/bin/env python3
"""
Standalone INR (SIREN) training script for cluster subprocess execution.

This script is called by INR_LLM.py as a cluster job. It runs ONLY the INR training phase.
Metrics are written to the analysis log for UCB scoring.

Usage:
    python train_inr_subprocess.py --config CONFIG_PATH --device DEVICE [options]
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import glob
import os
import shutil
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.models.graph_trainer import data_train_INR
from flyvis_gnn.utils import set_device, create_log_dir


def main():
    parser = argparse.ArgumentParser(description='Train SIREN INR (cluster subprocess)')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--config_file', type=str, default=None, help='Config file name for log directory')
    parser.add_argument('--log_file', type=str, default=None, help='Path to analysis log file')
    parser.add_argument('--error_log', type=str, default=None, help='Path to error log file')
    parser.add_argument('--total_steps', type=int, default=10000, help='Training iterations')
    parser.add_argument('--field_name', type=str, default='stimulus', help='Field to learn')
    parser.add_argument('--n_training_frames', type=int, default=0, help='Number of frames (0=all)')
    parser.add_argument('--inr_type', type=str, default='siren_txy', help='INR architecture')
    parser.add_argument('--exploration_dir', type=str, default=None, help='Copy artifacts here')
    parser.add_argument('--iteration', type=int, default=None, help='Iteration number')
    parser.add_argument('--slot', type=int, default=None, help='Slot number')

    args = parser.parse_args()

    error_log = None
    if args.error_log:
        try:
            error_log = open(args.error_log, 'w')
        except Exception as e:
            print(f"Warning: Could not open error log file: {e}", file=sys.stderr)

    try:
        config = NeuralGraphConfig.from_yaml(args.config)

        if args.config_file:
            config.config_file = args.config_file
            pre_folder = os.path.dirname(args.config_file)
            if pre_folder:
                pre_folder += '/'
            config.dataset = pre_folder + config.dataset

        device = set_device(args.device)

        nnr_f, loss_list = data_train_INR(
            config=config,
            device=device,
            total_steps=args.total_steps,
            field_name=args.field_name,
            n_training_frames=args.n_training_frames,
            inr_type=args.inr_type,
        )

        # Parse results.log written by data_train_INR and write to analysis log
        log_dir, _ = create_log_dir(config, erase=False)
        results_path = os.path.join(log_dir, 'tmp_training', f'inr_{args.field_name}', 'results.log')

        if args.log_file and os.path.exists(results_path):
            with open(results_path, 'r') as rf:
                results_content = rf.read()
            # Convert "key: value" to "key=value" for UCB parser
            with open(args.log_file, 'w') as lf:
                for line in results_content.strip().split('\n'):
                    if ':' in line:
                        key, val = line.split(':', 1)
                        lf.write(f"{key.strip()}={val.strip()}\n")

        # Copy artifacts to exploration dir
        if args.exploration_dir and args.iteration is not None and args.slot is not None:
            inr_output = os.path.join(log_dir, 'tmp_training', f'inr_{args.field_name}')

            # Copy model
            model_path = os.path.join(log_dir, 'models', f'inr_{args.field_name}.pt')
            if os.path.exists(model_path):
                models_dir = os.path.join(args.exploration_dir, 'models')
                os.makedirs(models_dir, exist_ok=True)
                dst = os.path.join(models_dir, f'iter_{args.iteration:03d}_slot_{args.slot:02d}_inr_{args.field_name}.pt')
                shutil.copy2(model_path, dst)
                print(f"copied model: {dst}")

            # Copy MP4 video
            video_path = os.path.join(inr_output, f'{args.field_name}_gt_vs_pred.mp4')
            if os.path.exists(video_path):
                video_dir = os.path.join(args.exploration_dir, 'inr_video')
                os.makedirs(video_dir, exist_ok=True)
                dst = os.path.join(video_dir, f'iter_{args.iteration:03d}_slot_{args.slot:02d}_{args.field_name}_gt_vs_pred.mp4')
                shutil.copy2(video_path, dst)
                print(f"copied video: {dst}")

            # Copy comparison PNGs
            if os.path.isdir(inr_output):
                comparison_dir = os.path.join(args.exploration_dir, 'inr_comparison')
                os.makedirs(comparison_dir, exist_ok=True)
                for png in glob.glob(os.path.join(inr_output, '*.png')):
                    dst = os.path.join(comparison_dir, f'iter_{args.iteration:03d}_slot_{args.slot:02d}_{os.path.basename(png)}')
                    shutil.copy2(png, dst)

            # Copy results.log
            if os.path.exists(results_path):
                results_dir = os.path.join(args.exploration_dir, 'results')
                os.makedirs(results_dir, exist_ok=True)
                dst = os.path.join(results_dir, f'iter_{args.iteration:03d}_slot_{args.slot:02d}_results.log')
                shutil.copy2(results_path, dst)

    except Exception as e:
        error_msg = f"\n{'='*80}\n"
        error_msg += "INR SUBPROCESS ERROR\n"
        error_msg += f"{'='*80}\n\n"
        error_msg += f"Error Type: {type(e).__name__}\n"
        error_msg += f"Error Message: {str(e)}\n\n"
        error_msg += "Full Traceback:\n"
        error_msg += traceback.format_exc()
        error_msg += f"\n{'='*80}\n"

        print(error_msg, file=sys.stderr, flush=True)
        if error_log:
            error_log.write(error_msg)
            error_log.flush()
        sys.exit(1)

    finally:
        if error_log:
            error_log.close()


if __name__ == '__main__':
    main()
