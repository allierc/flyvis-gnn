# Work Notes — flyvis-gnn

## Last session (2026-02-25)

### What was done

#### 1. Plot fixes (committed)
After the `lin_edge→g_phi` / `lin_phi→f_theta` rename, four plots were fixed:

- **`loss.tif`** (`plot_signal_loss` in `src/flyvis_gnn/plot.py`):
  - X-axis panels 1&2 now use global iteration (`epoch * Niter + iter`), tracked in `LossRegularizer._history['iteration']`
  - Legend labels updated: `$f_\theta$ weight regul`, `$g_\phi$ monotonicity`, `$g_\phi$ norm`, `$g_\phi$ weight regul`
  - Removed top-left epoch/iteration annotation
  - `LossRegularizer` COMPONENTS keys renamed: `edge_diff→g_phi_diff`, `edge_norm→g_phi_norm`, `edge_weight→g_phi_weight`, `phi_weight→f_theta_weight`, etc.

- **`epoch_*.png`** (`plot_training_summary_panels` in `src/flyvis_gnn/plot.py`):
  - Panel titles updated: `learned embedding`, `$g_\phi$ (MLP1)`, `$f_\theta$ (MLP0)`
  - R² panel x-axis uses global iteration (needs `Niter` passed from `graph_trainer.py`)
  - Removed scatter markers from R² lines

- **`eigen_comparison.png`** (`GNN_PlotFigure.py`): green=true, black=learned; FigureStyle applied
- **`svd_analysis.png`** (`src/flyvis_gnn/models/utils.py`): removed hardcoded `width=14, height=10`

#### 2. GNN_LLM.py — LLM-in-the-loop improvements (committed)

- **Single instruction file**: `LLM/instruction_flyvis_noise_free.md` replaces base + `_parallel.md` addendum
- **`n_parallel`** read from `config.claude.n_parallel` (not hardcoded)
- **`user_input.md`**: LLM reads at every batch, acknowledges pending instructions in-file
- **Cluster: train only** — `train_flyvis_subprocess.py` now runs training only; test+plot run locally in GNN_LLM.py (PHASE 3.5)
- **`generate_data`** moved to local PHASE 1.5 in GNN_LLM.py (before cluster submission); removed from subprocess
- **`training_time_target_min`** added to `config.claude` (default 60); passed to LLM in prompts; used in training time warnings
- **`ClaudeConfig`** (`src/flyvis_gnn/config.py`) gained: `generate_data`, `training_time_target_min`

#### 3. `config/fly/flyvis_noise_free.yaml` claude section
```yaml
claude:
  n_epochs: 1
  data_augmentation_loop: 25
  n_iter_block: 12
  ucb_c: 1.414
  n_parallel: 4
  node_name: h100
  generate_data: false
  training_time_target_min: 60
```

---

### Pending / next steps

#### check_repo.py (not yet written)
A guardrail script to run on the cluster before job submission. Design agreed:
- Run via SSH before first `submit_cluster_job` call in GNN_LLM.py
- Check `git diff HEAD` on `GraphCluster/flyvis-gnn`, excluding `config/`
- Exit 0 = clean, exit 1 = dirty source files (print diff summary)
- Also useful as post-repair sanity check
- Call point in GNN_LLM.py: add SSH check before the PHASE 2 job submission loop, and optionally after auto-repair resubmit

#### Other items to consider
- Add `check_repo.py` SSH call to GNN_LLM.py `submit_cluster_job` or before PHASE 2
- Other YAML configs (e.g. `flyvis_62_0.yaml`) don't yet have `generate_data` / `training_time_target_min` in their `claude:` section — they fall back to defaults (False / 60)
- Test the full end-to-end cluster run with `flyvis_noise_free` config

---

### Key file locations
| File | Role |
|------|------|
| `GNN_LLM.py` | Main LLM-in-the-loop training orchestrator |
| `train_flyvis_subprocess.py` | Cluster job: train only |
| `LLM/instruction_flyvis_noise_free.md` | Single instruction file for Claude |
| `LLM/user_input.md` | User ↔ LLM communication file |
| `src/flyvis_gnn/config.py` | `ClaudeConfig` dataclass |
| `src/flyvis_gnn/models/utils.py` | `LossRegularizer`, SVD analysis |
| `src/flyvis_gnn/plot.py` | `plot_signal_loss`, `plot_training_summary_panels` |
| `GNN_PlotFigure.py` | `data_plot`, eigen comparison |
| `config/fly/flyvis_noise_free.yaml` | Base config for noise-free flyvis run |
