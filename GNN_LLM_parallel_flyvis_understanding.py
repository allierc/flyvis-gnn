import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse
import os
import re
import shutil
import subprocess
import time
import yaml

# redirect PyTorch JIT cache to /scratch instead of /tmp (per IT request)
if os.path.isdir('/scratch'):
    os.environ['TMPDIR'] = '/scratch/allierc'
    os.makedirs('/scratch/allierc', exist_ok=True)

import sys

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.models.graph_trainer import data_train, data_test
from flyvis_gnn.models.exploration_tree import compute_ucb_scores
from flyvis_gnn.models.plot_exploration_tree import parse_ucb_scores, plot_ucb_tree
from flyvis_gnn.models.utils import save_exploration_artifacts_flyvis
from flyvis_gnn.utils import set_device, add_pre_folder
from GNN_PlotFigure import data_plot

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


# ---------------------------------------------------------------------------
# Model assignment: each slot trains a different difficult FlyVis model
# ---------------------------------------------------------------------------

MODEL_IDS = ['049', '011', '041', '003']
MODEL_DATASETS = {
    0: 'fly_N9_62_1_id_049',
    1: 'fly_N9_62_1_id_011',
    2: 'fly_N9_62_1_id_041',
    3: 'fly_N9_62_1_id_003',
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
    batch_iters = set(range(batch_start, batch_start + n_parallel))

    if batch_iters.issubset(found_iters):
        resume_at = batch_start + n_parallel
    else:
        resume_at = batch_start

    return resume_at


# ---------------------------------------------------------------------------
# Cluster helpers
# ---------------------------------------------------------------------------

CLUSTER_HOME = "/groups/saalfeld/home/allierc"
CLUSTER_ROOT_DIR = f"{CLUSTER_HOME}/Graph/flyvis-gnn"


def submit_cluster_job(slot, config_path, analysis_log_path, config_file_field,
                       log_dir, root_dir, erase=True, node_name='a100'):
    """Submit a single flyvis training job to the cluster WITHOUT -K (non-blocking)."""
    cluster_script_path = f"{log_dir}/cluster_train_{slot:02d}.sh"
    error_details_path = f"{log_dir}/training_error_{slot:02d}.log"

    cluster_config_path = config_path.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_analysis_log = analysis_log_path.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_error_log = error_details_path.replace(root_dir, CLUSTER_ROOT_DIR)

    # Use train_flyvis_subprocess.py instead of train_signal_subprocess.py
    cluster_train_cmd = f"python train_flyvis_subprocess.py --config '{cluster_config_path}' --device cuda"
    cluster_train_cmd += f" --log_file '{cluster_analysis_log}'"
    cluster_train_cmd += f" --config_file '{config_file_field}'"
    cluster_train_cmd += f" --error_log '{cluster_error_log}'"
    if erase:
        cluster_train_cmd += " --erase"

    with open(cluster_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {CLUSTER_ROOT_DIR}\n")
        f.write(f"conda run -n neural-graph {cluster_train_cmd}\n")
    os.chmod(cluster_script_path, 0o755)

    cluster_script = cluster_script_path.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_log_dir = log_dir.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_stdout = f"{cluster_log_dir}/cluster_train_{slot:02d}.out"
    cluster_stderr = f"{cluster_log_dir}/cluster_train_{slot:02d}.err"

    ssh_cmd = (
        f"ssh allierc@login1 \"cd {CLUSTER_ROOT_DIR} && "
        f"bsub -n 8 -gpu 'num=1' -q gpu_{node_name} -W 6000 "
        f"-o '{cluster_stdout}' -e '{cluster_stderr}' "
        f"'bash {cluster_script}'\""
    )
    print(f"\033[96m  slot {slot}: submitting via SSH\033[0m")
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
        ssh_cmd = f'ssh allierc@login1 "bjobs {ids_str} 2>/dev/null"'
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
                        print(f"\033[91m  slot {slot} (job {jid}): FAILED (EXIT)\033[0m")
                        if log_dir:
                            err_file = f"{log_dir}/cluster_train_{slot:02d}.err"
                            if os.path.exists(err_file):
                                try:
                                    with open(err_file, 'r') as ef:
                                        err_content = ef.read().strip()
                                    if err_content:
                                        print(f"\033[91m  --- slot {slot} error log ---\033[0m")
                                        for eline in err_content.splitlines()[-30:]:
                                            print(f"\033[91m    {eline}\033[0m")
                                        print("\033[91m  --- end error log ---\033[0m")
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


def is_git_repo(path):
    """Check if path is inside a git repository."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            cwd=path, capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def get_modified_code_files(root_dir, code_files):
    """Return list of code_files that have uncommitted changes."""
    modified = []
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD'],
            cwd=root_dir, capture_output=True, text=True, timeout=10
        )
        changed = set(result.stdout.strip().splitlines())
        result2 = subprocess.run(
            ['git', 'diff', '--name-only', '--cached'],
            cwd=root_dir, capture_output=True, text=True, timeout=10
        )
        changed.update(result2.stdout.strip().splitlines())
        for f in code_files:
            if f in changed:
                modified.append(f)
    except Exception:
        pass
    return modified


def run_claude_cli(prompt, root_dir, max_turns=500):
    """Run Claude CLI with real-time output streaming. Returns output text."""
    claude_cmd = [
        'claude',
        '-p', prompt,
        '--output-format', 'text',
        '--max-turns', str(max_turns),
        '--allowedTools',
        'Read', 'Edit', 'Write'
    ]

    output_lines = []
    process = subprocess.Popen(
        claude_cmd,
        cwd=root_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='', flush=True)
        output_lines.append(line)

    process.wait()
    return ''.join(output_lines)


# ---------------------------------------------------------------------------
# Analysis tool execution + auto-repair
# ---------------------------------------------------------------------------

def execute_analysis_tool(tool_path, root_dir, max_repair_attempts=3):
    """Execute an analysis tool script with auto-repair on failure.

    Returns (success: bool, output: str).
    """
    tools_output_dir = os.path.join(root_dir, 'tools', 'output')
    os.makedirs(tools_output_dir, exist_ok=True)

    for attempt in range(max_repair_attempts + 1):
        print(f"\033[96m  analysis tool: running {os.path.basename(tool_path)}"
              f"{f' (repair attempt {attempt})' if attempt > 0 else ''}\033[0m")

        result = subprocess.run(
            ['python', tool_path],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            output = result.stdout
            print(f"\033[92m  analysis tool: SUCCESS\033[0m")

            # Save stdout to output file
            iter_match = re.search(r'analysis_iter_(\d+)', tool_path)
            if iter_match:
                iter_num = iter_match.group(1)
                output_path = os.path.join(tools_output_dir, f'analysis_iter_{iter_num}.txt')
                with open(output_path, 'w') as f:
                    f.write(output)

            # Git-commit the tool
            if is_git_repo(root_dir):
                try:
                    subprocess.run(['git', 'add', tool_path], cwd=root_dir,
                                   capture_output=True, timeout=10)
                    # Also add output files
                    subprocess.run(['git', 'add', 'tools/output/'], cwd=root_dir,
                                   capture_output=True, timeout=10)
                    subprocess.run(
                        ['git', 'commit', '-m',
                         f'[Analysis] {os.path.basename(tool_path)}'],
                        cwd=root_dir, capture_output=True, timeout=10
                    )
                    print(f"\033[92m  analysis tool: git-committed\033[0m")
                except Exception:
                    pass

            return True, output

        # Tool crashed — attempt auto-repair
        error_msg = result.stderr[-2000:] if result.stderr else result.stdout[-2000:]
        print(f"\033[91m  analysis tool: CRASHED (attempt {attempt + 1}/{max_repair_attempts + 1})\033[0m")

        if attempt < max_repair_attempts:
            print(f"\033[93m  analysis tool: calling Claude for repair\033[0m")
            repair_prompt = (
                f"Analysis tool crashed. Please fix the bug.\n\n"
                f"Tool path: {tool_path}\n\n"
                f"Error:\n```\n{error_msg}\n```\n\n"
                f"Fix the bug in the script. Do NOT change the analysis logic, "
                f"only fix the crash. Do NOT make other changes."
            )
            run_claude_cli(repair_prompt, root_dir, max_turns=10)

    print(f"\033[91m  analysis tool: FAILED after {max_repair_attempts + 1} attempts\033[0m")
    return False, f"ANALYSIS TOOL FAILED: {error_msg}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(
        description="FlyVis-GNN — Understanding Exploration (4 Difficult Models)"
    )
    parser.add_argument(
        "-o", "--option", nargs="+", help="option that takes multiple values"
    )
    parser.add_argument(
        "--fresh", action="store_true", default=True,
        help="start from iteration 1 (ignore auto-resume)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="auto-resume from last completed batch"
    )

    print()
    device = []
    args = parser.parse_args()

    N_PARALLEL = 4

    if args.option:
        print(f"Options: {args.option}")
    if args.option is not None:
        task = args.option[0]
        config_list = [args.option[1]]
        best_model = None
        task_params = {}
        for arg in args.option[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                task_params[key] = int(value) if value.isdigit() else value
    else:
        best_model = ''
        task = 'train_test_plot_Claude_cluster'
        config_list = ['fly_N9_62_1_understand']
        task_params = {'iterations': 48}

    n_iterations = task_params.get('iterations', 48)
    base_config_name = config_list[0] if config_list else 'fly_N9_62_1_understand'
    instruction_name = task_params.get('instruction', 'instruction_fly_N9_62_1_understanding')
    llm_task_name = task_params.get('llm_task', f'{base_config_name}_Claude')

    # -----------------------------------------------------------------------
    # Claude mode setup
    # -----------------------------------------------------------------------
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_root = root_dir + "/config"

    if args.resume:
        analysis_path_probe = f"{root_dir}/{llm_task_name}_analysis.md"
        config_save_dir_probe = f"{root_dir}/log/Claude_exploration/{instruction_name}_parallel/config"
        start_iteration = detect_last_iteration(analysis_path_probe, config_save_dir_probe, N_PARALLEL)
        if start_iteration > 1:
            print(f"\033[93mAuto-resume: resuming from batch starting at {start_iteration}\033[0m")
        else:
            print("\033[93mFresh start (no previous iterations found)\033[0m")
    else:
        start_iteration = 1
        _analysis_check = f"{root_dir}/{llm_task_name}_analysis.md"
        if os.path.exists(_analysis_check):
            print("\033[91mWARNING: Fresh start will erase existing results in:\033[0m")
            print(f"\033[91m  {_analysis_check}\033[0m")
            print(f"\033[91m  {root_dir}/{llm_task_name}_memory.md\033[0m")
            answer = input("\033[91mContinue? (y/n): \033[0m").strip().lower()
            if answer != 'y':
                print("Aborted.")
                sys.exit(0)
        print("\033[93mFresh start\033[0m")

    # --- Initialize 4 slot configs (each pointing to a different model) ---
    for cfg in config_list:
        cfg_file, pre = add_pre_folder(cfg)
        source_config_base = f"{config_root}/{pre}{cfg}.yaml"

    # Read claude settings from slot 0 source config
    slot0_source = f"{config_root}/{pre}{llm_task_name}_00.yaml"
    if os.path.exists(slot0_source):
        with open(slot0_source, 'r') as f:
            source_data = yaml.safe_load(f)
    else:
        # Fall back to the base config template
        # Look for any of the slot configs or the base
        for fallback in [source_config_base,
                         f"{config_root}/{pre}fly_N9_62_1_Claude_02.yaml"]:
            if os.path.exists(fallback):
                with open(fallback, 'r') as f:
                    source_data = yaml.safe_load(f)
                break
        else:
            print(f"\033[91merror: no source config found\033[0m")
            sys.exit(1)

    claude_cfg = source_data.get('claude', {})
    claude_n_epochs = claude_cfg.get('n_epochs', 1)
    claude_data_augmentation_loop = claude_cfg.get('data_augmentation_loop', 25)
    claude_ucb_c = claude_cfg.get('ucb_c', 1.414)
    claude_node_name = claude_cfg.get('node_name', 'h100')

    print(f"\033[94mCluster node: gpu_{claude_node_name}\033[0m")
    print(f"\033[94mModels: {', '.join(f'slot {i}={MODEL_IDS[i]}' for i in range(N_PARALLEL))}\033[0m")

    # Slot config paths and analysis log paths
    config_paths = {}
    analysis_log_paths = {}
    slot_names = {}

    for slot in range(N_PARALLEL):
        slot_name = f"{llm_task_name}_{slot:02d}"
        slot_names[slot] = slot_name
        target = f"{config_root}/{pre}{slot_name}.yaml"
        config_paths[slot] = target
        analysis_log_paths[slot] = f"{root_dir}/{slot_name}_analysis.log"

        if start_iteration == 1 and not args.resume:
            # Each slot gets its own config pointing to a different model dataset
            slot_source = f"{config_root}/{pre}{llm_task_name}_{slot:02d}.yaml"
            if os.path.exists(slot_source):
                print(f"\033[93m  slot {slot}: using existing {slot_source} "
                      f"(model {MODEL_IDS[slot]})\033[0m")
            else:
                print(f"\033[91m  slot {slot}: config not found: {slot_source}\033[0m")
                print(f"\033[91m  Please create config files first.\033[0m")
                sys.exit(1)
        else:
            print(f"\033[93m  slot {slot}: preserving {target} (resuming)\033[0m")

    # Shared files
    config_file, pre_folder = add_pre_folder(llm_task_name + '_00')
    analysis_path = f"{root_dir}/{llm_task_name}_analysis.md"
    memory_path = f"{root_dir}/{llm_task_name}_memory.md"
    ucb_path = f"{root_dir}/{llm_task_name}_ucb_scores.txt"
    instruction_path = f"{root_dir}/{instruction_name}.md"
    parallel_instruction_path = f"{root_dir}/{instruction_name}_parallel.md"
    reasoning_log_path = f"{root_dir}/{llm_task_name}_reasoning.log"
    tools_dir = f"{root_dir}/tools"

    exploration_dir = f"{root_dir}/log/Claude_exploration/{instruction_name}_parallel"
    log_dir = f"{root_dir}/log/Claude_exploration/{instruction_name}_parallel"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{tools_dir}/output", exist_ok=True)

    cluster_enabled = 'cluster' in task

    # Check instruction files exist
    if not os.path.exists(instruction_path):
        print(f"\033[91merror: instruction file not found: {instruction_path}\033[0m")
        sys.exit(1)
    if not os.path.exists(parallel_instruction_path):
        print(f"\033[93mwarning: parallel instruction file not found: {parallel_instruction_path}\033[0m")
        print("\033[93m  Claude will use base instructions only\033[0m")
        parallel_instruction_path = None

    # Initialize shared files on fresh start
    if start_iteration == 1 and not args.resume:
        with open(analysis_path, 'w') as f:
            f.write(f"# Understanding Exploration Log: Difficult FlyVis Models (parallel)\n\n")
        print(f"\033[93mcleared {analysis_path}\033[0m")
        open(reasoning_log_path, 'w').close()
        print(f"\033[93mcleared {reasoning_log_path}\033[0m")

        # Initialize memory with UNDERSTANDING section and model profiles
        with open(memory_path, 'w') as f:
            f.write("# Understanding Exploration: Difficult FlyVis Models\n\n")
            f.write("## UNDERSTANDING\n\n")
            f.write("### Model 049 (svd_rank_99=19, baseline R²=0.634)\n")
            f.write("**Profile**: 13741 neurons, activity_rank_99=16, svd_activity_rank_99=19. "
                    "Low-dimensional neural activity.\n")
            f.write("**Hypothesis**: [to be filled after first results]\n")
            f.write("**Status**: untested\n")
            f.write("**Evidence FOR**:\n")
            f.write("**Evidence AGAINST**:\n")
            f.write("**Best R² so far**: 0.634\n")
            f.write("**Next experiment**: baseline with Node 79 params\n\n")

            f.write("### Model 011 (svd_rank_99=45, baseline R²=0.308)\n")
            f.write("**Profile**: 13741 neurons, activity_rank_99=26, svd_activity_rank_99=45. "
                    "High activity diversity yet worst R². Hard connectivity structure.\n")
            f.write("**Hypothesis**: [to be filled after first results]\n")
            f.write("**Status**: untested\n")
            f.write("**Evidence FOR**:\n")
            f.write("**Evidence AGAINST**:\n")
            f.write("**Best R² so far**: 0.308\n")
            f.write("**Next experiment**: baseline with Node 79 params\n\n")

            f.write("### Model 041 (svd_rank_99=6, baseline R²=0.629)\n")
            f.write("**Profile**: 13741 neurons, activity_rank_99=5, svd_activity_rank_99=6. "
                    "Near-collapsed activity. Only 6 SVD components at 99%.\n")
            f.write("**Hypothesis**: [to be filled after first results]\n")
            f.write("**Status**: untested\n")
            f.write("**Evidence FOR**:\n")
            f.write("**Evidence AGAINST**:\n")
            f.write("**Best R² so far**: 0.629\n")
            f.write("**Next experiment**: baseline with Node 79 params\n\n")

            f.write("### Model 003 (svd_rank_99=60, baseline R²=0.627)\n")
            f.write("**Profile**: 13741 neurons, activity_rank_99=35, svd_activity_rank_99=60. "
                    "Moderate activity diversity but hard connectivity.\n")
            f.write("**Hypothesis**: [to be filled after first results]\n")
            f.write("**Status**: untested\n")
            f.write("**Evidence FOR**:\n")
            f.write("**Evidence AGAINST**:\n")
            f.write("**Best R² so far**: 0.627\n")
            f.write("**Next experiment**: baseline with Node 79 params\n\n")

            f.write("---\n\n")
            f.write("## Established Principles (from base 62_1 exploration)\n\n")
            f.write("1. lr_W=6E-4 with edge_L1=0.3 achieves best conn_R2\n")
            f.write("2. lr_W=1E-3 requires lr=1E-3 (not 1.2E-3)\n")
            f.write("3. lr_emb=1.5E-3 is required for lr_W < 1E-3\n")
            f.write("4. lr_emb >= 1.8E-3 destroys V_rest recovery\n")
            f.write("5. coeff_edge_norm >= 10 is catastrophic\n")
            f.write("6. coeff_edge_weight_L1=0.3 is optimal\n")
            f.write("7. coeff_phi_weight_L1=0.5 improves V_rest recovery\n")
            f.write("8. coeff_edge_diff=750 is optimal\n")
            f.write("9. coeff_W_L1=5E-5 is optimal for V_rest\n")
            f.write("10. coeff_phi_weight_L2 must stay at 0.001\n")
            f.write("11. n_layers=4 is harmful\n")
            f.write("12. hidden_dim=80 + hidden_dim_update=80 is optimal architecture\n")
            f.write("13. batch_size=2 maintains conn_R2 with faster training\n")
            f.write("14. batch_size >= 3 causes V_rest collapse\n")
            f.write("15. data_augmentation_loop=20 is viable for speed\n")
            f.write("16. lr=1.2E-3 is optimal for MLPs\n\n")
            f.write("NOTE: These principles were derived on the standard model (R²=0.980). "
                    "They may not hold for difficult models.\n\n")

            f.write("---\n\n")
            f.write("## New Principles (discovered in this exploration)\n\n")
            f.write("---\n\n")
            f.write("## Cross-Model Observations\n\n")
            f.write("---\n\n")
            f.write("## Analysis Tools Log\n\n")
            f.write("Summary of each analysis tool: what it measured, key findings, "
                    "and which UNDERSTANDING hypothesis it informed.\n\n")
            f.write("| Iter | Tool | What it measured | Key finding | Informed hypothesis |\n")
            f.write("|------|------|-----------------|-------------|---------------------|\n\n")
            f.write("---\n\n")
            f.write("## Iterations\n\n")
        print(f"\033[93minitialized {memory_path} with UNDERSTANDING section\033[0m")

        if os.path.exists(ucb_path):
            os.remove(ucb_path)
            print(f"\033[93mdeleted {ucb_path}\033[0m")
    else:
        print(f"\033[93mpreserving shared files (resuming from iter {start_iteration})\033[0m")

    print(f"\033[93mUNDERSTANDING EXPLORATION (N={N_PARALLEL}, "
          f"{n_iterations} iterations, starting at {start_iteration})\033[0m")

    # -----------------------------------------------------------------------
    # BATCH 0: Claude "start" call — initialize 4 config variations
    # -----------------------------------------------------------------------
    if start_iteration == 1 and not args.resume:
        print(f"\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH 0: Claude initializing {N_PARALLEL} config variations\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        slot_list = "\n".join(
            f"  Slot {s} (model {MODEL_IDS[s]}): {config_paths[s]}"
            for s in range(N_PARALLEL)
        )

        model_gen_logs = "\n".join(
            f"  Model {MODEL_IDS[s]}: {root_dir}/graphs_data/fly/{MODEL_DATASETS[s]}/generation_log.txt"
            for s in range(N_PARALLEL)
        )

        parallel_ref = f"\nParallel instructions: {parallel_instruction_path}" if parallel_instruction_path else ""

        start_prompt = f"""PARALLEL START: Initialize {N_PARALLEL} config variations for the first batch.

Instructions (follow all instructions): {instruction_path}{parallel_ref}
Working memory: {memory_path}
Full log (append only): {analysis_path}

Config files to edit (all {N_PARALLEL}):
{slot_list}

Generation logs for each model:
{model_gen_logs}

Each slot trains a DIFFERENT flyvis model. Read the instructions, the generation logs,
and the base configs. Then create {N_PARALLEL} initial training parameter variations.
Each config already has a unique dataset name — do NOT change the dataset field.

Starting point: Node 79 best params (lr_W=6E-4, lr=1.2E-3, lr_emb=1.5E-3, edge_diff=750,
phi_L1=0.5, edge_L1=0.3, W_L1=5E-5, hidden_dim=80, hidden_dim_update=80, batch=2, data_aug=20).

For the first batch, you may vary params across slots to explore whether different models
need different settings. Write initial hypotheses to the UNDERSTANDING section in memory.md.

IMPORTANT: Data is PRE-GENERATED — do NOT change simulation parameters."""

        print("\033[93mClaude start call...\033[0m")
        output_text = run_claude_cli(start_prompt, root_dir, max_turns=100)

        if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
            print("\n\033[91mOAuth token expired during start call\033[0m")
            print("\033[93m  1. Run: claude /login\033[0m")
            print("\033[93m  2. Then re-run this script\033[0m")
            sys.exit(1)

        if output_text.strip():
            with open(reasoning_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write("=== BATCH 0 (start call) ===\n")
                f.write(f"{'='*60}\n")
                f.write(output_text.strip())
                f.write("\n\n")

    # -----------------------------------------------------------------------
    # Main batch loop
    # -----------------------------------------------------------------------
    prev_batch_last = None  # Track previous batch for analysis tool feedback

    for batch_start in range(start_iteration, n_iterations + 1, N_PARALLEL):
        iterations = [batch_start + s for s in range(N_PARALLEL)
                      if batch_start + s <= n_iterations]

        batch_first = iterations[0]
        batch_last = iterations[-1]
        n_slots = len(iterations)

        # No block boundaries in understanding exploration

        print(f"\n\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH: iterations {batch_first}-{batch_last} / {n_iterations}\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        # -------------------------------------------------------------------
        # PHASE 1: No data generation needed for flyvis (pre-generated)
        # -------------------------------------------------------------------
        print(f"\n\033[93mPHASE 1: Loading configs for {n_slots} slots (data is pre-generated)\033[0m")

        configs = {}
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            config = NeuralGraphConfig.from_yaml(config_paths[slot])
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + slot_names[slot]
            configs[slot] = config

            if device == []:
                device = set_device(config.training.device)

        # -------------------------------------------------------------------
        # PHASE 2: Submit 4 training jobs to cluster (or run locally)
        # -------------------------------------------------------------------
        job_results = {}

        if "train" in task:
            if cluster_enabled:
                print(f"\n\033[93mPHASE 2: Submitting {n_slots} flyvis training jobs to cluster\033[0m")

                job_ids = {}
                for slot_idx, iteration in enumerate(iterations):
                    slot = slot_idx
                    config = configs[slot]
                    jid = submit_cluster_job(
                        slot=slot,
                        config_path=config_paths[slot],
                        analysis_log_path=analysis_log_paths[slot],
                        config_file_field=config.config_file,
                        log_dir=log_dir,
                        root_dir=root_dir,
                        erase=True,
                        node_name=claude_node_name
                    )
                    if jid:
                        job_ids[slot] = jid
                    else:
                        job_results[slot] = False

                if job_ids:
                    print(f"\n\033[93mPHASE 3: Waiting for {len(job_ids)} cluster jobs to complete\033[0m")
                    cluster_results = wait_for_cluster_jobs(job_ids, log_dir=log_dir, poll_interval=60)
                    job_results.update(cluster_results)

                # Auto-repair for failed training jobs
                for slot_idx in range(n_slots):
                    if job_results.get(slot_idx) == False:
                        err_content = None
                        err_file = f"{log_dir}/training_error_{slot_idx:02d}.log"
                        lsf_err_file = f"{log_dir}/cluster_train_{slot_idx:02d}.err"

                        for ef_path in [err_file, lsf_err_file]:
                            if os.path.exists(ef_path):
                                try:
                                    with open(ef_path, 'r') as ef:
                                        content = ef.read()
                                    if 'FLYVIS SUBPROCESS ERROR' in content or 'Traceback' in content:
                                        err_content = content
                                        break
                                except Exception:
                                    pass

                        if not err_content:
                            continue

                        print(f"\033[91m  slot {slot_idx}: TRAINING ERROR — attempting auto-repair\033[0m")

                        code_files = [
                            'src/flyvis_gnn/models/graph_trainer.py',
                            'src/flyvis_gnn/models/Signal_Propagation.py',
                            'GNN_PlotFigure.py',
                        ]
                        modified_code = get_modified_code_files(root_dir, code_files) if is_git_repo(root_dir) else []

                        if not modified_code:
                            print(f"\033[93m  slot {slot_idx}: no modified code files to repair — skipping\033[0m")
                            continue

                        max_repair_attempts = 3
                        repaired = False
                        for attempt in range(max_repair_attempts):
                            print(f"\033[93m  slot {slot_idx}: repair attempt {attempt + 1}/{max_repair_attempts}\033[0m")
                            repair_prompt = f"""TRAINING CRASHED - Please fix the code error.

Error traceback:
```
{err_content[-3000:]}
```

Modified files: {chr(10).join(f'- {root_dir}/{f}' for f in modified_code)}

Fix the bug. Do NOT make other changes."""

                            repair_cmd = [
                                'claude', '-p', repair_prompt,
                                '--output-format', 'text', '--max-turns', '10',
                                '--allowedTools', 'Read', 'Edit', 'Write'
                            ]
                            repair_result = subprocess.run(repair_cmd, cwd=root_dir, capture_output=True, text=True)
                            if 'CANNOT_FIX' in repair_result.stdout:
                                print(f"\033[91m  slot {slot_idx}: Claude cannot fix — stopping repair\033[0m")
                                break

                            print(f"\033[96m  slot {slot_idx}: resubmitting after repair\033[0m")
                            config = configs[slot_idx]
                            jid = submit_cluster_job(
                                slot=slot_idx,
                                config_path=config_paths[slot_idx],
                                analysis_log_path=analysis_log_paths[slot_idx],
                                config_file_field=config.config_file,
                                log_dir=log_dir,
                                root_dir=root_dir,
                                erase=True,
                                node_name=claude_node_name
                            )
                            if jid:
                                retry_results = wait_for_cluster_jobs(
                                    {slot_idx: jid}, log_dir=log_dir, poll_interval=60
                                )
                                if retry_results.get(slot_idx):
                                    job_results[slot_idx] = True
                                    repaired = True
                                    print(f"\033[92m  slot {slot_idx}: repair successful!\033[0m")
                                    break
                                for ef_path in [err_file, lsf_err_file]:
                                    if os.path.exists(ef_path):
                                        try:
                                            with open(ef_path, 'r') as ef:
                                                err_content = ef.read()
                                            break
                                        except Exception:
                                            pass

                        if not repaired:
                            print(f"\033[91m  slot {slot_idx}: repair failed after {max_repair_attempts} attempts\033[0m")
                            if is_git_repo(root_dir):
                                for fp in code_files:
                                    try:
                                        subprocess.run(['git', 'checkout', 'HEAD', '--', fp],
                                                      cwd=root_dir, capture_output=True, timeout=10)
                                    except Exception:
                                        pass

            else:
                # Local execution (no cluster) — run sequentially
                print(f"\n\033[93mPHASE 2: Training {n_slots} flyvis models locally (sequential)\033[0m")

                for slot_idx, iteration in enumerate(iterations):
                    slot = slot_idx
                    config = configs[slot]
                    print(f"\033[90m  slot {slot} (iter {iteration}, model {MODEL_IDS[slot]}): training locally...\033[0m")

                    log_file = open(analysis_log_paths[slot], 'w')
                    try:
                        data_train(
                            config=config,
                            erase=True,
                            best_model=best_model,
                            style='color',
                            device=device,
                            log_file=log_file
                        )

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

                        slot_config_file = pre_folder + slot_names[slot]
                        folder_name = './log/' + pre_folder + '/tmp_results/'
                        os.makedirs(folder_name, exist_ok=True)
                        data_plot(
                            config=config,
                            config_file=slot_config_file,
                            epoch_list=['best'],
                            style='color',
                            extended='plots',
                            device=device,
                            log_file=log_file
                        )

                        job_results[slot] = True
                    except Exception as e:
                        print(f"\033[91m  slot {slot}: training failed: {e}\033[0m")
                        job_results[slot] = False
                    finally:
                        log_file.close()

        else:
            for slot in range(n_slots):
                job_results[slot] = True

        # -------------------------------------------------------------------
        # PHASE 4: Save exploration artifacts + check training time
        # -------------------------------------------------------------------
        print("\n\033[93mPHASE 4: Saving exploration artifacts\033[0m")

        activity_paths = {}
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            if not job_results.get(slot, False):
                print(f"\033[90m  slot {slot} (iter {iteration}): skipping (training failed)\033[0m")
                continue

            config = configs[slot]

            artifact_paths = save_exploration_artifacts_flyvis(
                root_dir, exploration_dir, config, slot_names[slot],
                pre_folder, iteration,
                iter_in_block=iteration, block_number=1
            )
            activity_paths[slot] = artifact_paths['activity_path']

            # Save config file for EVERY iteration
            config_save_dir = f"{exploration_dir}/config"
            os.makedirs(config_save_dir, exist_ok=True)
            dst_config = f"{config_save_dir}/iter_{iteration:03d}_slot_{slot:02d}.yaml"
            shutil.copy2(config_paths[slot], dst_config)

            # Check training time
            log_path = analysis_log_paths[slot]
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log_content = f.read()
                time_m = re.search(r'training_time_min[=:]\s*([\d.]+)', log_content)
                if time_m:
                    training_time = float(time_m.group(1))
                    if training_time > 60:
                        print(f"\033[91m  WARNING: slot {slot} training took {training_time:.1f} min (>60 min limit)\033[0m")
                    else:
                        print(f"\033[92m  slot {slot}: training time {training_time:.1f} min\033[0m")

        # -------------------------------------------------------------------
        # PHASE 5: UCB update (no block scoping)
        # -------------------------------------------------------------------
        print("\n\033[93mPHASE 5: Computing UCB scores (no block scoping)\033[0m")

        with open(config_paths[0], 'r') as f:
            raw_config = yaml.safe_load(f)
        ucb_c = raw_config.get('claude', {}).get('ucb_c', 1.414)

        existing_content = ""
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r') as f:
                existing_content = f.read()

        stub_entries = ""
        for slot_idx, iteration in enumerate(iterations):
            if not job_results.get(slot_idx, False):
                continue
            log_path = analysis_log_paths[slot_idx]
            if not os.path.exists(log_path):
                continue
            with open(log_path, 'r') as f:
                log_content = f.read()
            r2_m = re.search(r'connectivity_R2[=:]\s*([\d.eE+-]+|nan)', log_content)
            pearson_m = re.search(r'test_pearson[=:]\s*([\d.eE+-]+|nan)', log_content)
            cluster_m = re.search(r'cluster_accuracy[=:]\s*([\d.eE+-]+|nan)', log_content)
            tau_m = re.search(r'tau_R2[=:]\s*([\d.eE+-]+|nan)', log_content)
            vrest_m = re.search(r'V_rest_R2[=:]\s*([\d.eE+-]+|nan)', log_content)
            time_m = re.search(r'training_time_min[=:]\s*([\d.]+)', log_content)
            if r2_m:
                r2_val = r2_m.group(1)
                pearson_val = pearson_m.group(1) if pearson_m else '0.0'
                cluster_val = cluster_m.group(1) if cluster_m else '0.0'
                tau_val = tau_m.group(1) if tau_m else '0.0'
                vrest_val = vrest_m.group(1) if vrest_m else '0.0'
                time_val = time_m.group(1) if time_m else '0.0'
                if f'## Iter {iteration}:' not in existing_content:
                    stub_entries += (
                        f"\n## Iter {iteration}: pending\n"
                        f"Node: id={iteration}, parent=root\n"
                        f"Model: {MODEL_IDS[slot_idx]}\n"
                        f"Metrics: test_R2=0, test_pearson={pearson_val}, "
                        f"connectivity_R2={r2_val}, tau_R2={tau_val}, "
                        f"V_rest_R2={vrest_val}, cluster_accuracy={cluster_val}\n"
                    )

        tmp_analysis = analysis_path + '.tmp_ucb'
        with open(tmp_analysis, 'w') as f:
            f.write(existing_content + stub_entries)

        # block_size=0 → no block scoping, all nodes always in scope
        compute_ucb_scores(
            tmp_analysis, ucb_path, c=ucb_c,
            current_log_path=None,
            current_iteration=batch_last,
            block_size=0
        )
        os.remove(tmp_analysis)
        print(f"\033[92mUCB scores computed (c={ucb_c}, block_size=0): {ucb_path}\033[0m")

        # -------------------------------------------------------------------
        # PHASE 6a: Claude PASS 1 — analyze results + write analysis tool
        # -------------------------------------------------------------------
        print("\n\033[93mPHASE 6a: Claude pass 1 — analyze results + write analysis tool\033[0m")

        slot_info_lines = []
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            status = "COMPLETED" if job_results.get(slot, False) else "FAILED"
            act_path = activity_paths.get(slot, "N/A")
            slot_info_lines.append(
                f"Slot {slot} (iteration {iteration}, model {MODEL_IDS[slot]}) [{status}]:\n"
                f"  Metrics: {analysis_log_paths[slot]}\n"
                f"  Activity: {act_path}\n"
                f"  Config: {config_paths[slot]}"
            )
        slot_info = "\n\n".join(slot_info_lines)

        parallel_ref = f"\nParallel instructions: {parallel_instruction_path}" if parallel_instruction_path else ""

        claude_prompt_pass1 = f"""Batch iterations {batch_first}-{batch_last} / {n_iterations}

UNDERSTANDING EXPLORATION — PASS 1 of 2: Analyze results and write analysis tool.

Instructions (follow all instructions): {instruction_path}{parallel_ref}
Working memory: {memory_path}
Full log (append only): {analysis_path}
UCB scores: {ucb_path}

{slot_info}

In this pass:
1. Read metrics for all {n_slots} slots
2. Write ## Iter N: entries to full log and memory
3. Update UNDERSTANDING section in memory.md (hypotheses, evidence, status)
4. Write analysis tool to tools/analysis_iter_{batch_last:03d}.py

Do NOT edit config files yet — that happens in pass 2 after the analysis tool runs.

IMPORTANT: The analysis tool you write will be executed as a subprocess right after this call.
Its stdout output will be fed back to you in pass 2, where you will use it to refine
UNDERSTANDING and propose the next 4 config mutations.
IMPORTANT: Update hypothesis status (untested/supported/falsified/revised) based on training evidence."""

        print("\033[93mClaude pass 1...\033[0m")
        output_text_pass1 = run_claude_cli(claude_prompt_pass1, root_dir)

        if 'OAuth token has expired' in output_text_pass1 or 'authentication_error' in output_text_pass1:
            print(f"\n\033[91mOAuth token expired at batch {batch_first}-{batch_last} (pass 1)\033[0m")
            print("\033[93m  1. Run: claude /login\033[0m")
            print(f"\033[93m  2. Then re-run with --resume\033[0m")
            sys.exit(1)

        if output_text_pass1.strip():
            with open(reasoning_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== Batch {batch_first}-{batch_last} (pass 1: analyze + write tool) ===\n")
                f.write(f"{'='*60}\n")
                f.write(output_text_pass1.strip())
                f.write("\n\n")

        # -------------------------------------------------------------------
        # PHASE 6b: Execute analysis tool (just written by Claude in pass 1)
        # -------------------------------------------------------------------
        tool_path = f"{tools_dir}/analysis_iter_{batch_last:03d}.py"
        analysis_tool_output = ""

        if os.path.exists(tool_path):
            print(f"\n\033[93mPHASE 6b: Executing analysis tool analysis_iter_{batch_last:03d}.py\033[0m")
            success, output = execute_analysis_tool(tool_path, root_dir)
            if success:
                if len(output) > 5000:
                    analysis_tool_output = output[:5000] + "\n... [truncated]"
                else:
                    analysis_tool_output = output
            else:
                analysis_tool_output = output  # Contains error message
        else:
            print(f"\033[93m  Claude did not write an analysis tool for this batch\033[0m")

        # -------------------------------------------------------------------
        # PHASE 6c: Claude PASS 2 — read analysis output + propose mutations
        # -------------------------------------------------------------------
        print(f"\n\033[93mPHASE 6c: Claude pass 2 — analysis feedback + propose mutations\033[0m")

        analysis_feedback = ""
        if analysis_tool_output:
            analysis_feedback = (
                f"\n--- ANALYSIS TOOL OUTPUT (tools/analysis_iter_{batch_last:03d}.py) ---\n"
                f"{analysis_tool_output}\n"
                f"--- END ANALYSIS TOOL OUTPUT ---\n"
            )
        else:
            analysis_feedback = "\n(No analysis tool output available for this batch)\n"

        claude_prompt_pass2 = f"""UNDERSTANDING EXPLORATION — PASS 2 of 2: Read analysis results + propose mutations.

Batch iterations {batch_first}-{batch_last} / {n_iterations}

Instructions (follow all instructions): {instruction_path}{parallel_ref}
Working memory: {memory_path}
Full log (append only): {analysis_path}
UCB scores: {ucb_path}
{analysis_feedback}
In this pass:
1. Read the analysis tool output above
2. Update the UNDERSTANDING section in memory.md based on analysis findings
3. Update the Analysis Tools Log table in memory.md
4. Use UCB scores + UNDERSTANDING to select parents for next batch
5. Edit all {N_PARALLEL} config files for the next batch

Config files to edit:
{chr(10).join(f'  Slot {s} (model {MODEL_IDS[s]}): {config_paths[s]}' for s in range(N_PARALLEL))}

IMPORTANT: Do NOT change the 'dataset' field in any config.
IMPORTANT: Data is PRE-GENERATED — do NOT change simulation parameters.
IMPORTANT: Each slot trains a different model. Mutations should be model-specific.
IMPORTANT: Use the analysis tool findings to inform your mutations — if the analysis
revealed something about WHY a model is hard, design the next experiment to test that."""

        print("\033[93mClaude pass 2...\033[0m")
        output_text_pass2 = run_claude_cli(claude_prompt_pass2, root_dir)

        if 'OAuth token has expired' in output_text_pass2 or 'authentication_error' in output_text_pass2:
            print(f"\n\033[91mOAuth token expired at batch {batch_first}-{batch_last} (pass 2)\033[0m")
            print("\033[93m  1. Run: claude /login\033[0m")
            print(f"\033[93m  2. Then re-run with --resume\033[0m")
            sys.exit(1)

        if output_text_pass2.strip():
            with open(reasoning_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== Batch {batch_first}-{batch_last} (pass 2: analysis feedback + mutations) ===\n")
                f.write(f"{'='*60}\n")
                f.write(output_text_pass2.strip())
                f.write("\n\n")

        # Recompute UCB after Claude writes iteration entries
        compute_ucb_scores(analysis_path, ucb_path, c=ucb_c,
                           current_log_path=None,
                           current_iteration=batch_last,
                           block_size=0)

        # UCB tree visualization (save every 4 batches)
        should_save_tree = (batch_first == 1) or (batch_last % 16 == 0)
        if should_save_tree:
            tree_save_dir = f"{exploration_dir}/exploration_tree"
            os.makedirs(tree_save_dir, exist_ok=True)
            ucb_tree_path = f"{tree_save_dir}/ucb_tree_iter_{batch_last:03d}.png"
            nodes = parse_ucb_scores(ucb_path)
            if nodes:
                config = configs[0]
                sim_info = f"Models: {', '.join(MODEL_IDS)}"
                sim_info += f", n_neurons={config.simulation.n_neurons}"
                if hasattr(config.simulation, 'visual_input_type'):
                    sim_info += f", visual_input={config.simulation.visual_input_type}"
                plot_ucb_tree(nodes, ucb_tree_path,
                              title=f"Understanding UCB Tree - Batch {batch_first}-{batch_last}",
                              simulation_info=sim_info)

        # Save protocol at first batch
        protocol_save_dir = f"{exploration_dir}/protocol"
        os.makedirs(protocol_save_dir, exist_ok=True)
        if batch_first == 1:
            dst_instruction = f"{protocol_save_dir}/understanding_instruction.md"
            if os.path.exists(instruction_path):
                shutil.copy2(instruction_path, dst_instruction)

        # Save memory periodically (every 16 iterations)
        if batch_last % 16 == 0:
            memory_save_dir = f"{exploration_dir}/memory"
            os.makedirs(memory_save_dir, exist_ok=True)
            dst_memory = f"{memory_save_dir}/iter_{batch_last:03d}_memory.md"
            if os.path.exists(memory_path):
                shutil.copy2(memory_path, dst_memory)
                print(f"\033[92msaved memory snapshot: {dst_memory}\033[0m")

        # Print batch summary
        n_success = sum(1 for v in job_results.values() if v)
        n_failed = sum(1 for v in job_results.values() if not v)
        print(f"\n\033[92mBatch {batch_first}-{batch_last} complete: {n_success} succeeded, {n_failed} failed\033[0m")


# python GNN_LLM_parallel_flyvis_understanding.py -o train_test_plot_Claude_cluster fly_N9_62_1_understand iterations=144 --fresh
# python GNN_LLM_parallel_flyvis_understanding.py -o train_test_plot_Claude_cluster fly_N9_62_1_understand iterations=144 --resume
