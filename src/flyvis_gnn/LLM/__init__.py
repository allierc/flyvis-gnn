"""LLM exploration pipeline for flyvis-gnn.

Provides the infrastructure for Claude-driven hyperparameter exploration
with optional interactive code modification sessions at block boundaries.
"""

from .state import ExplorationState, BatchInfo
from .pipeline import (
    setup_exploration,
    init_slot_configs,
    init_shared_files,
    make_batch_info,
    run_batch_0,
    run_code_session,
    load_configs_and_seeds,
    generate_data_locally,
    run_cluster_training,
    run_local_test_plot,
    run_local_pipeline,
    save_artifacts,
    update_ucb_scores,
    run_claude_analysis,
    finalize_batch,
)

__all__ = [
    'ExplorationState',
    'BatchInfo',
    'setup_exploration',
    'init_slot_configs',
    'init_shared_files',
    'make_batch_info',
    'run_batch_0',
    'run_code_session',
    'load_configs_and_seeds',
    'generate_data_locally',
    'run_cluster_training',
    'run_local_test_plot',
    'run_local_pipeline',
    'save_artifacts',
    'update_ucb_scores',
    'run_claude_analysis',
    'finalize_batch',
]
