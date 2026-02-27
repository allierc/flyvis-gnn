import matplotlib
matplotlib.use('Agg')
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
import yaml

if os.path.isdir('/scratch'):
    os.environ['TMPDIR'] = '/scratch/allierc'
    os.makedirs('/scratch/allierc', exist_ok=True)

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.models.exploration_tree import compute_ucb_scores
from flyvis_gnn.models.plot_exploration_tree import parse_ucb_scores, plot_ucb_tree
from flyvis_gnn.utils import set_device, add_pre_folder, log_path

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLUSTER_USER = "allierc"
CLUSTER_LOGIN = "login1"
CLUSTER_HOME = "/groups/saalfeld/home/allierc"
CLUSTER_ROOT_DIR = f"{CLUSTER_HOME}/GraphCluster/flyvis-gnn"
CLUSTER_DATA_DIR = f"{CLUSTER_HOME}/GraphData"
CLUSTER_SSH = f"{CLUSTER_USER}@{CLUSTER_LOGIN}"

FRAME_BLOCKS = [10, 100, 1000, 10000, 20000, 40000, 64000]
FIELD_NAME = 'stimulus'
INR_TYPE = 'siren_txy'

DEFAULT_TOTAL_STEPS = {
    10: 5000, 100: 10000, 1000: 20000, 10000: 30000,
    20000: 40000, 40000: 50000, 64000: 60000,
}

# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def detect_last_iteration(analysis_path, config_save_dir, n_parallel):
    """Detect the last fully completed batch from saved artifacts."""
    found_iters = set()
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            for line in f:
                match = re.match(r'^##+ Iter (\d+):', line)
                if match:
                    found_iters.add(int(match.group(1)))
    if os.path.isdir(config_save_dir):
        for fname in os.listdir(config_save_dir):
            match = re.match(r'iter_(\d+)_slot_\d+\.yaml', fname)
            if match:
                found_iters.add(int(match.group(1)))
    if not found_iters:
        return 1
    last_iter = max(found_iters)
    batch_start = ((last_iter - 1) // n_parallel) * n_parallel + 1
    if set(range(batch_start, batch_start + n_parallel)).issubset(found_iters):
        return batch_start + n_parallel
    return batch_start


# ---------------------------------------------------------------------------
# Cluster helpers
# ---------------------------------------------------------------------------

def local_to_cluster(path, root_dir):
    for sub in ('config', 'log', 'graphs_data'):
        local_sub = os.path.join(root_dir, sub)
        if path.startswith(local_sub):
            return os.path.join(CLUSTER_DATA_DIR, sub) + path[len(local_sub):]
    return path.replace(root_dir, CLUSTER_ROOT_DIR)


def check_cluster_repo():
    ssh_cmd = (
        f"ssh {CLUSTER_SSH} "
        f"\"cd {CLUSTER_ROOT_DIR} && git diff HEAD --stat -- . ':!config/'\""
    )
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
    dirty = result.stdout.strip()
    if dirty:
        print(f"\033[91mcluster repo has uncommitted changes:\n{dirty}\033[0m")
        return False
    return True


def submit_inr_job(slot, config_path, analysis_log_path, config_file_field,
                   log_dir, root_dir, total_steps, n_training_frames,
                   node_name='a100', exploration_dir=None, iteration=None):
    """Submit an INR training job to the cluster (non-blocking)."""
    cluster_script_path = f"{log_dir}/cluster_inr_{slot:02d}.sh"
    error_details_path = f"{log_dir}/inr_error_{slot:02d}.log"

    cluster_config = local_to_cluster(config_path, root_dir)
    cluster_log = local_to_cluster(analysis_log_path, root_dir)
    cluster_error = local_to_cluster(error_details_path, root_dir)

    cmd = f"python train_inr_subprocess.py --config '{cluster_config}' --device cuda"
    cmd += f" --config_file '{config_file_field}'"
    cmd += f" --log_file '{cluster_log}'"
    cmd += f" --error_log '{cluster_error}'"
    cmd += f" --total_steps {total_steps}"
    cmd += f" --n_training_frames {n_training_frames}"
    cmd += f" --field_name {FIELD_NAME}"
    cmd += f" --inr_type {INR_TYPE}"
    if exploration_dir is not None and iteration is not None:
        cluster_exploration = local_to_cluster(exploration_dir, root_dir)
        cmd += f" --exploration_dir '{cluster_exploration}'"
        cmd += f" --iteration {iteration} --slot {slot}"

    with open(cluster_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {CLUSTER_ROOT_DIR}\n")
        f.write(f"conda run -n neural-graph {cmd}\n")
    os.chmod(cluster_script_path, 0o755)

    cluster_script = local_to_cluster(cluster_script_path, root_dir)
    cluster_log_dir = local_to_cluster(log_dir, root_dir)
    cluster_stdout = f"{cluster_log_dir}/cluster_inr_{slot:02d}.out"
    cluster_stderr = f"{cluster_log_dir}/cluster_inr_{slot:02d}.err"

    ssh_cmd = (
        f"ssh {CLUSTER_SSH} \"cd {CLUSTER_ROOT_DIR} && "
        f"bsub -n 8 -gpu 'num=1' -q gpu_{node_name} -W 6000 "
        f"-o '{cluster_stdout}' -e '{cluster_stderr}' "
        f"'bash {cluster_script}'\""
    )
    print(f"\033[96m  slot {slot}: submitting INR job via SSH\033[0m", flush=True)
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

    match = re.search(r'Job <(\d+)>', result.stdout)
    if match:
        job_id = match.group(1)
        print(f"\033[92m  slot {slot}: job {job_id} submitted\033[0m")
        return job_id
    else:
        print(f"\033[91m  slot {slot}: submission FAILED\033[0m")
        print(f"    stdout: {result.stdout.strip()}")
        print(f"    stderr: {result.stderr.strip()}")
        return None


def wait_for_cluster_jobs(job_ids, log_dir=None, poll_interval=60):
    """Poll bjobs via SSH until all jobs finish."""
    pending = dict(job_ids)
    results = {}
    while pending:
        ids_str = ' '.join(pending.values())
        ssh_cmd = f'ssh {CLUSTER_SSH} "bjobs {ids_str} 2>/dev/null"'
        out = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
        for slot, jid in list(pending.items()):
            for line in out.stdout.splitlines():
                if jid in line:
                    if 'DONE' in line:
                        results[slot] = True
                        del pending[slot]
                        print(f"\033[92m  slot {slot} (job {jid}): DONE\033[0m")
                    elif 'EXIT' in line:
                        results[slot] = False
                        del pending[slot]
                        print(f"\033[91m  slot {slot} (job {jid}): FAILED\033[0m")
                        if log_dir:
                            err_file = f"{log_dir}/cluster_inr_{slot:02d}.err"
                            if os.path.exists(err_file):
                                try:
                                    with open(err_file, 'r') as ef:
                                        for eline in ef.read().strip().splitlines()[-20:]:
                                            print(f"\033[91m    {eline}\033[0m")
                                except Exception:
                                    pass
            if slot in pending and jid not in out.stdout:
                results[slot] = True
                del pending[slot]
                print(f"\033[93m  slot {slot} (job {jid}): no longer in queue (assuming DONE)\033[0m")
        if pending:
            statuses = [f"slot {s}" for s in pending]
            print(f"\033[90m  ... waiting for {', '.join(statuses)} ({poll_interval}s)\033[0m")
            time.sleep(poll_interval)
    return results


def run_claude_cli(prompt, root_dir, max_turns=500):
    """Run Claude CLI with real-time output streaming."""
    claude_cmd = [
        'claude', '-p', prompt,
        '--output-format', 'text', '--max-turns', str(max_turns),
        '--allowedTools', 'Read', 'Edit', 'Write'
    ]
    output_lines = []
    process = subprocess.Popen(
        claude_cmd, cwd=root_dir,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in process.stdout:
        print(line, end='', flush=True)
        output_lines.append(line)
    process.wait()
    return ''.join(output_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="FlyVis INR — SIREN Hyperparameter Optimization")
    parser.add_argument("-o", "--option", nargs="+", help="task config_name [key=value ...]")
    parser.add_argument("--fresh", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cluster", action="store_true")

    print()
    device = []
    args = parser.parse_args()

    if args.option:
        print(f"Options: {args.option}")

    if args.option is not None:
        task = args.option[0]
        config_list = [args.option[1]]
        task_params = {}
        for arg in args.option[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                task_params[key] = int(value) if value.isdigit() else value
    else:
        task = 'train_inr_Claude_cluster'
        config_list = ['flyvis_noise_005_INR']
        task_params = {}

    base_config_name = config_list[0]
    instruction_name = task_params.get('instruction', f'instruction_{base_config_name}')
    llm_task_name = task_params.get('llm_task', f'{base_config_name}_Claude')
    exploration_name = task_params.get('exploration_name', f'LLM_{base_config_name}')

    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_root = root_dir + "/config"
    llm_dir = f"{root_dir}/LLM"
    exploration_dir = os.path.abspath(log_path('Claude_exploration', exploration_name))

    # Load source config and claude settings
    for cfg in config_list:
        cfg_file, pre = add_pre_folder(cfg)
        source_config = f"{config_root}/{pre}{cfg}.yaml"

    with open(source_config, 'r') as f:
        source_data = yaml.safe_load(f)
    claude_cfg = source_data.get('claude', {})
    N_ITER_BLOCK = claude_cfg.get('n_iter_block', 12)
    N_PARALLEL = claude_cfg.get('n_parallel', 4)
    claude_ucb_c = claude_cfg.get('ucb_c', 1.414)
    claude_node_name = claude_cfg.get('node_name', 'a100')
    training_time_target = claude_cfg.get('training_time_target_min', 60)
    cluster_enabled = args.cluster

    n_iterations = len(FRAME_BLOCKS) * N_ITER_BLOCK  # 7 blocks * 12 = 84

    if args.resume:
        analysis_path_probe = f"{exploration_dir}/{llm_task_name}_analysis.md"
        config_save_dir_probe = f"{exploration_dir}/config"
        start_iteration = detect_last_iteration(analysis_path_probe, config_save_dir_probe, N_PARALLEL)
        if start_iteration > 1:
            print(f"\033[93mAuto-resume: resuming from batch starting at {start_iteration}\033[0m")
        else:
            print("\033[93mfresh start (no previous iterations found)\033[0m")
    else:
        start_iteration = 1
        _analysis_check = f"{exploration_dir}/{llm_task_name}_analysis.md"
        if os.path.exists(_analysis_check):
            print(f"\033[91mWARNING: fresh start will erase existing results in {_analysis_check}\033[0m")
            answer = input("\033[91mContinue? (y/n): \033[0m").strip().lower()
            if answer != 'y':
                print("Aborted.")
                sys.exit(0)
        print("\033[93mfresh start\033[0m")

    mode = "cluster" if cluster_enabled else "local"
    print(f"\033[94mMode: {mode}, node: gpu_{claude_node_name}, n_parallel: {N_PARALLEL}, blocks: {len(FRAME_BLOCKS)}, iterations: {n_iterations}\033[0m")

    # Slot config paths
    config_paths = {}
    analysis_log_paths = {}
    slot_names = {}

    for slot in range(N_PARALLEL):
        slot_name = f"{llm_task_name}_{slot:02d}"
        slot_names[slot] = slot_name
        target = f"{config_root}/{pre}{slot_name}.yaml"
        config_paths[slot] = target
        analysis_log_paths[slot] = f"{exploration_dir}/{slot_name}_analysis.log"

        if start_iteration == 1 and not args.resume:
            shutil.copy2(source_config, target)
            with open(target, 'r') as f:
                config_data = yaml.safe_load(f)
            config_data['description'] = f'INR SIREN optimization - designed by Claude (slot {slot})'
            config_data.setdefault('claude', {}).update({
                'n_iter_block': N_ITER_BLOCK,
                'ucb_c': claude_ucb_c,
                'n_parallel': N_PARALLEL,
                'node_name': claude_node_name,
                'training_time_target_min': training_time_target,
            })
            with open(target, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            print(f"\033[93m  slot {slot}: created {target}\033[0m")
        else:
            print(f"\033[93m  slot {slot}: preserving {target} (resuming)\033[0m")

    # Shared files
    config_file, pre_folder = add_pre_folder(llm_task_name + '_00')
    analysis_path = f"{exploration_dir}/{llm_task_name}_analysis.md"
    memory_path = f"{exploration_dir}/{llm_task_name}_memory.md"
    ucb_path = f"{exploration_dir}/{llm_task_name}_ucb_scores.txt"
    instruction_path = f"{llm_dir}/{instruction_name}.md"
    reasoning_log_path = f"{exploration_dir}/{llm_task_name}_reasoning.log"
    user_input_path = f"{llm_dir}/user_input.md"
    log_dir = exploration_dir
    os.makedirs(exploration_dir, exist_ok=True)

    if not os.path.exists(instruction_path):
        print(f"\033[91merror: instruction file not found: {instruction_path}\033[0m")
        sys.exit(1)

    if not os.path.exists(user_input_path):
        with open(user_input_path, 'w') as f:
            f.write("# User Input\n\n## Pending Instructions\n\n_(empty)_\n\n## Acknowledged\n\n")

    # Initialize shared files on fresh start
    if start_iteration == 1 and not args.resume:
        with open(analysis_path, 'w') as f:
            f.write(f"# INR SIREN Experiment Log: {base_config_name} (parallel)\n\n")
        open(reasoning_log_path, 'w').close()
        with open(memory_path, 'w') as f:
            f.write(f"# Working Memory: {base_config_name} INR\n\n")
            f.write("## Paper Summary\n\n")
            f.write("- **INR optimization**: [pending first results]\n")
            f.write("- **LLM-driven exploration**: [pending first results]\n\n")
            f.write("## Knowledge Base\n\n")
            f.write("### R2 Progression Table\n\n")
            f.write("| Iter | n_frames | hidden_dim | n_layers | omega | lr | batch_sz | T_period | steps | final_r2 | final_mse | time_min | Hypothesis |\n")
            f.write("| ---- | -------- | ---------- | -------- | ----- | -- | -------- | -------- | ----- | -------- | --------- | -------- | ---------- |\n\n")
            f.write("### Established Principles\n\n")
            f.write("### Falsified Hypotheses\n\n")
            f.write("### Open Questions\n\n---\n\n")
            f.write("## Previous Block Summary\n\n---\n\n")
            f.write("## Current Block (Block 1)\n\n")
            f.write("### Block Info\n\n### Hypothesis\n\n### Iterations This Block\n\n### Emerging Observations\n\n")
        if os.path.exists(ucb_path):
            os.remove(ucb_path)
        print(f"\033[93minitialized shared files\033[0m")

    print(f"\033[93mINR SIREN optimization (N={N_PARALLEL}, {n_iterations} iterations, frames={FRAME_BLOCKS})\033[0m")

    # -------------------------------------------------------------------
    # BATCH 0: Claude start call
    # -------------------------------------------------------------------
    if start_iteration == 1 and not args.resume:
        print(f"\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH 0: Claude initializing {N_PARALLEL} config variations\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        slot_list = "\n".join(f"  Slot {s}: {config_paths[s]}" for s in range(N_PARALLEL))
        first_block_frames = FRAME_BLOCKS[0]

        start_prompt = f"""PARALLEL START: Initialize {N_PARALLEL} INR config variations for the first batch.

Instructions (follow all instructions): {instruction_path}
Working memory: {memory_path}
Full log (append only): {analysis_path}
User input: {user_input_path}

Config files to edit (all {N_PARALLEL}):
{slot_list}

BLOCK 1: n_training_frames = {first_block_frames} (FIXED — do not change)
Default total_steps = {DEFAULT_TOTAL_STEPS[first_block_frames]}
INR type: {INR_TYPE} (input=(t,x,y), output=scalar per neuron)
Field: {FIELD_NAME}

Key parameters to vary across slots:
  - hidden_dim_nnr_f (graph_model section)
  - n_layers_nnr_f (graph_model section)
  - omega_f (graph_model section)
  - nnr_f_T_period (graph_model section)
  - learning_rate_NNR_f (training section)
  - inr_batch_size (training section)
  - total_steps (claude section)

Create {N_PARALLEL} diverse initial configurations. Each slot should test a DIFFERENT parameter combination.
Training time target: {training_time_target} min per slot.

Write the planned variations to the working memory file."""

        print("\033[93mClaude start call...\033[0m")
        output_text = run_claude_cli(start_prompt, root_dir, max_turns=100)

        if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
            print("\n\033[91mOAuth token expired during start call\033[0m")
            sys.exit(1)

        if output_text.strip():
            with open(reasoning_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n=== BATCH 0 (start call) ===\n{'='*60}\n")
                f.write(output_text.strip() + "\n\n")

    # -------------------------------------------------------------------
    # Main batch loop
    # -------------------------------------------------------------------
    for batch_start in range(start_iteration, n_iterations + 1, N_PARALLEL):
        iterations = [batch_start + s for s in range(N_PARALLEL)
                      if batch_start + s <= n_iterations]

        batch_first = iterations[0]
        batch_last = iterations[-1]
        n_slots = len(iterations)

        # Determine current block and frame count
        block_index = (batch_first - 1) // N_ITER_BLOCK
        block_number = block_index + 1
        n_training_frames = FRAME_BLOCKS[min(block_index, len(FRAME_BLOCKS) - 1)]
        iter_in_block_first = (batch_first - 1) % N_ITER_BLOCK + 1
        iter_in_block_last = (batch_last - 1) % N_ITER_BLOCK + 1
        is_block_end = any((it - 1) % N_ITER_BLOCK + 1 == N_ITER_BLOCK for it in iterations)
        is_block_start = iter_in_block_first == 1

        # Block boundary: clear UCB at start of new block
        if batch_first > 1 and (batch_first - 1) % N_ITER_BLOCK == 0:
            if os.path.exists(ucb_path):
                os.remove(ucb_path)
                print(f"\033[93mblock boundary: cleared UCB scores\033[0m")

            # Carry forward best config from previous block
            prev_block_idx = block_index - 1
            best_r2 = -1.0
            best_config_path = None
            config_save_dir = f"{exploration_dir}/config"
            if os.path.isdir(config_save_dir):
                prev_start = prev_block_idx * N_ITER_BLOCK + 1
                prev_end = prev_start + N_ITER_BLOCK
                for fname in os.listdir(config_save_dir):
                    m = re.match(r'iter_(\d+)_slot_(\d+)\.yaml', fname)
                    if m and prev_start <= int(m.group(1)) < prev_end:
                        # Check corresponding results
                        it, sl = int(m.group(1)), int(m.group(2))
                        results_file = os.path.join(exploration_dir, 'results',
                                                    f'iter_{it:03d}_slot_{sl:02d}_results.log')
                        if os.path.exists(results_file):
                            with open(results_file, 'r') as rf:
                                for line in rf:
                                    rm = re.search(r'final_r2:\s*([\d.eE+-]+)', line)
                                    if rm:
                                        r2 = float(rm.group(1))
                                        if r2 > best_r2:
                                            best_r2 = r2
                                            best_config_path = os.path.join(config_save_dir, fname)

            if best_config_path:
                print(f"\033[92m  carrying forward best config (R²={best_r2:.4f}): {best_config_path}\033[0m")
                for slot in range(N_PARALLEL):
                    shutil.copy2(best_config_path, config_paths[slot])
                    with open(config_paths[slot], 'r') as f:
                        data = yaml.safe_load(f)
                    data['training']['n_training_frames'] = n_training_frames
                    data['graph_model']['nnr_f_T_period'] = n_training_frames
                    data.setdefault('claude', {})['total_steps'] = DEFAULT_TOTAL_STEPS.get(n_training_frames, 30000)
                    with open(config_paths[slot], 'w') as f:
                        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"\n\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH: iters {batch_first}-{batch_last} / {n_iterations}  "
              f"(block {block_number}, n_frames={n_training_frames})\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        # -------------------------------------------------------------------
        # PHASE 1: Load configs
        # -------------------------------------------------------------------
        print(f"\n\033[93mPHASE 1: Loading configs for {n_slots} slots\033[0m")

        configs = {}
        slot_total_steps = {}
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            config = NeuralGraphConfig.from_yaml(config_paths[slot])
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + slot_names[slot]

            # Read total_steps from claude section
            with open(config_paths[slot], 'r') as f:
                raw = yaml.safe_load(f)
            total_steps = raw.get('claude', {}).get('total_steps',
                                                     DEFAULT_TOTAL_STEPS.get(n_training_frames, 30000))
            slot_total_steps[slot] = total_steps
            configs[slot] = config

            if device == []:
                device = set_device(config.training.device)

        # -------------------------------------------------------------------
        # PHASE 2: Submit INR training jobs
        # -------------------------------------------------------------------
        job_results = {}

        if 'train' in task:
            if cluster_enabled:
                if not check_cluster_repo():
                    print("\033[91mAborting — fix cluster repo (use --resume)\033[0m")
                    sys.exit(1)

                print(f"\n\033[93mPHASE 2: Submitting {n_slots} INR jobs to cluster\033[0m")
                job_ids = {}
                for slot_idx, iteration in enumerate(iterations):
                    slot = slot_idx
                    config = configs[slot]
                    jid = submit_inr_job(
                        slot=slot,
                        config_path=config_paths[slot],
                        analysis_log_path=analysis_log_paths[slot],
                        config_file_field=config.config_file,
                        log_dir=log_dir,
                        root_dir=root_dir,
                        total_steps=slot_total_steps[slot],
                        n_training_frames=n_training_frames,
                        node_name=claude_node_name,
                        exploration_dir=exploration_dir,
                        iteration=iteration,
                    )
                    if jid:
                        job_ids[slot] = jid
                    else:
                        job_results[slot] = False

                if job_ids:
                    print(f"\n\033[93mPHASE 3: Waiting for {len(job_ids)} cluster jobs\033[0m")
                    cluster_results = wait_for_cluster_jobs(job_ids, log_dir=log_dir, poll_interval=60)
                    job_results.update(cluster_results)

            else:
                # Local execution
                print(f"\n\033[93mPHASE 2: Training {n_slots} INR models locally\033[0m")
                from flyvis_gnn.models.graph_trainer import data_train_INR

                for slot_idx, iteration in enumerate(iterations):
                    slot = slot_idx
                    config = configs[slot]
                    print(f"\033[90m  slot {slot} (iter {iteration}): training locally...\033[0m")

                    nnr_f, loss_list = data_train_INR(
                        config=config,
                        device=device,
                        total_steps=slot_total_steps[slot],
                        field_name=FIELD_NAME,
                        n_training_frames=n_training_frames,
                        inr_type=INR_TYPE,
                    )

                    # Write metrics to analysis log
                    from flyvis_gnn.utils import create_log_dir
                    slot_log_dir, _ = create_log_dir(config, erase=False)
                    results_path = os.path.join(slot_log_dir, 'tmp_training',
                                                f'inr_{FIELD_NAME}', 'results.log')
                    if os.path.exists(results_path):
                        with open(results_path, 'r') as rf:
                            content = rf.read()
                        with open(analysis_log_paths[slot], 'w') as lf:
                            for line in content.strip().split('\n'):
                                if ':' in line:
                                    key, val = line.split(':', 1)
                                    lf.write(f"{key.strip()}={val.strip()}\n")

                        # Copy artifacts
                        inr_output = os.path.join(slot_log_dir, 'tmp_training', f'inr_{FIELD_NAME}')

                        video_path = os.path.join(inr_output, f'{FIELD_NAME}_gt_vs_pred.mp4')
                        if os.path.exists(video_path):
                            video_dir = os.path.join(exploration_dir, 'inr_video')
                            os.makedirs(video_dir, exist_ok=True)
                            dst = os.path.join(video_dir, f'iter_{iteration:03d}_slot_{slot:02d}_{FIELD_NAME}_gt_vs_pred.mp4')
                            shutil.copy2(video_path, dst)

                        results_dir = os.path.join(exploration_dir, 'results')
                        os.makedirs(results_dir, exist_ok=True)
                        shutil.copy2(results_path,
                                     os.path.join(results_dir, f'iter_{iteration:03d}_slot_{slot:02d}_results.log'))

                    job_results[slot] = True
        else:
            for slot in range(n_slots):
                job_results[slot] = True

        # -------------------------------------------------------------------
        # PHASE 4: Save configs + check training time
        # -------------------------------------------------------------------
        print("\n\033[93mPHASE 4: Saving configs\033[0m")

        config_save_dir = f"{exploration_dir}/config"
        os.makedirs(config_save_dir, exist_ok=True)
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            if not job_results.get(slot, False):
                continue
            dst_config = f"{config_save_dir}/iter_{iteration:03d}_slot_{slot:02d}.yaml"
            shutil.copy2(config_paths[slot], dst_config)

            slot_log_path = analysis_log_paths[slot]
            if os.path.exists(slot_log_path):
                with open(slot_log_path, 'r') as f:
                    log_content = f.read()
                time_m = re.search(r'training_time_min[=:]\s*([\d.]+)', log_content)
                if time_m:
                    t_min = float(time_m.group(1))
                    color = '\033[91m' if t_min > training_time_target else '\033[92m'
                    print(f"{color}  slot {slot}: training time {t_min:.1f} min\033[0m")

        # -------------------------------------------------------------------
        # PHASE 5: UCB scores
        # -------------------------------------------------------------------
        print("\n\033[93mPHASE 5: Computing UCB scores\033[0m")

        existing_content = ""
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r') as f:
                existing_content = f.read()

        stub_entries = ""
        for slot_idx, iteration in enumerate(iterations):
            if not job_results.get(slot_idx, False):
                continue
            slot_log_path = analysis_log_paths[slot_idx]
            if not os.path.exists(slot_log_path):
                continue
            with open(slot_log_path, 'r') as f:
                log_content = f.read()
            r2_m = re.search(r'final_r2[=:]\s*([\d.eE+-]+|nan)', log_content)
            mse_m = re.search(r'final_mse[=:]\s*([\d.eE+-]+|nan)', log_content)
            time_m = re.search(r'training_time_min[=:]\s*([\d.]+)', log_content)
            if r2_m and f'## Iter {iteration}:' not in existing_content:
                r2_val = r2_m.group(1)
                mse_val = mse_m.group(1) if mse_m else '0.0'
                time_val = time_m.group(1) if time_m else '0.0'
                stub_entries += (
                    f"\n## Iter {iteration}: pending\n"
                    f"Node: id={iteration}, parent=root\n"
                    f"Metrics: final_r2={r2_val}, final_mse={mse_val}, "
                    f"training_time_min={time_val}\n"
                )

        tmp_analysis = analysis_path + '.tmp_ucb'
        with open(tmp_analysis, 'w') as f:
            f.write(existing_content + stub_entries)

        compute_ucb_scores(
            tmp_analysis, ucb_path, c=claude_ucb_c,
            current_log_path=None,
            current_iteration=batch_last,
            block_size=N_ITER_BLOCK,
            reward_key='best_r2',
        )
        os.remove(tmp_analysis)
        print(f"\033[92mUCB scores computed: {ucb_path}\033[0m")

        # -------------------------------------------------------------------
        # PHASE 6: Claude analysis + next mutations
        # -------------------------------------------------------------------
        print("\n\033[93mPHASE 6: Claude analysis + next mutations\033[0m")

        slot_info_lines = []
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            status = "COMPLETED" if job_results.get(slot, False) else "FAILED"
            slot_info_lines.append(
                f"Slot {slot} (iteration {iteration}) [{status}]:\n"
                f"  Metrics: {analysis_log_paths[slot]}\n"
                f"  Config: {config_paths[slot]}"
            )
        slot_info = "\n\n".join(slot_info_lines)

        block_end_marker = "\n>>> BLOCK END <<<" if is_block_end else ""

        claude_prompt = f"""Batch iterations {batch_first}-{batch_last} / {n_iterations}
Block info: block {block_number} (n_training_frames={n_training_frames}), iterations {iter_in_block_first}-{iter_in_block_last}/{N_ITER_BLOCK} within block{block_end_marker}

INR MODE: Analyze {n_slots} results, then propose next {N_PARALLEL} mutations.
Each slot tests a DIFFERENT configuration (not different seeds).

Instructions: {instruction_path}
Working memory: {memory_path}
Full log (append only): {analysis_path}
UCB scores: {ucb_path}
User input: {user_input_path}

{slot_info}

FIXED for this block (DO NOT change):
  - n_training_frames = {n_training_frames}
  - field_name = {FIELD_NAME}
  - inr_type = {INR_TYPE}

TUNABLE parameters (edit in config files):
  - hidden_dim_nnr_f (graph_model section)
  - n_layers_nnr_f (graph_model section)
  - omega_f (graph_model section)
  - nnr_f_T_period (graph_model section)
  - nnr_f_xy_period (graph_model section)
  - learning_rate_NNR_f (training section)
  - inr_batch_size (training section)
  - total_steps (claude section)

Training time target: {training_time_target} min per slot.
Edit all {N_PARALLEL} config files. Each config should test a DIFFERENT parameter combination.
"""

        print("\033[93mClaude analysis...\033[0m")
        output_text = run_claude_cli(claude_prompt, root_dir)

        if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
            print(f"\n\033[91mOAuth token expired at batch {batch_first}-{batch_last}\033[0m")
            print("\033[93m  1. Run: claude /login\033[0m")
            print("\033[93m  2. Re-run with --resume\033[0m")
            sys.exit(1)

        if output_text.strip():
            with open(reasoning_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== Batch {batch_first}-{batch_last} ===\n")
                f.write(f"{'='*60}\n")
                f.write(output_text.strip() + "\n\n")

        # Recompute UCB after Claude writes entries
        compute_ucb_scores(analysis_path, ucb_path, c=claude_ucb_c,
                           current_log_path=None,
                           current_iteration=batch_last,
                           block_size=N_ITER_BLOCK,
                           reward_key='best_r2')

        # UCB tree visualization
        if block_number == 1 or is_block_end:
            tree_dir = f"{exploration_dir}/exploration_tree"
            os.makedirs(tree_dir, exist_ok=True)
            ucb_tree_path = f"{tree_dir}/ucb_tree_iter_{batch_last:03d}.png"
            nodes = parse_ucb_scores(ucb_path) if os.path.exists(ucb_path) else []
            if nodes:
                plot_ucb_tree(nodes, ucb_tree_path,
                              title=f"INR UCB Tree - Batch {batch_first}-{batch_last} (n_frames={n_training_frames})")

        # Save protocol/memory at block boundaries
        protocol_dir = f"{exploration_dir}/protocol"
        os.makedirs(protocol_dir, exist_ok=True)
        if iter_in_block_first == 1 and os.path.exists(instruction_path):
            shutil.copy2(instruction_path, f"{protocol_dir}/block_{block_number:03d}.md")

        if is_block_end and os.path.exists(memory_path):
            memory_dir = f"{exploration_dir}/memory"
            os.makedirs(memory_dir, exist_ok=True)
            dst = f"{memory_dir}/block_{block_number:03d}_memory.md"
            shutil.copy2(memory_path, dst)
            print(f"\033[92msaved memory snapshot: {dst}\033[0m")

        n_success = sum(1 for v in job_results.values() if v)
        n_failed = sum(1 for v in job_results.values() if not v)
        print(f"\n\033[92mBatch {batch_first}-{batch_last} complete: {n_success} succeeded, {n_failed} failed\033[0m")


# python INR_LLM.py -o train_inr_Claude_cluster flyvis_noise_005_INR --cluster
# python INR_LLM.py -o train_inr_Claude_cluster flyvis_noise_005_INR --cluster --resume
# python INR_LLM.py -o train_inr_Claude flyvis_noise_005_INR
