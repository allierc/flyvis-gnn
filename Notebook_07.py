# %% [raw]
# ---
# title: "Agentic Hyper-Parameter Optimization: Addressing Identifiability"
# author: "Allier, Lappalainen, Saalfeld"
# categories:
#   - FlyVis
#   - GNN
#   - Agentic Optimization
#   - Identifiability
# execute:
#   echo: false
# image: "assets/Fig_agentic_loop.svg"
# description: "Agentic hyper-parameter optimization addresses the ill-posedness of circuit recovery across three noise regimes, discovering that identifiability fails for different reasons at each level and finding distinct solutions through hypothesis-driven reasoning."
# ---

# %% [markdown]
# ## Agentic Hyper-Parameter Optimization
#
# The inverse problem solved by the GNN is **ill-posed**: recovering five coupled
# components ($\widehat{W}$, $\tau$, $V^{\text{rest}}$, $f_\theta$,
# $g_\phi$) from voltage traces alone is under-determined.  Many
# different parameter combinations can produce indistinguishable
# voltage predictions.  This degeneracy manifests as **seed
# dependence**: slight differences in random initialization can push
# the optimizer toward a different solution on the degenerate
# manifold.  A GNN that accurately forecasts neural activity may
# nonetheless have recovered the wrong connectivity.
#
# The combined space of architecture, regularization, and training
# hyperparameters (~20 coupled parameters) is too large to explore
# exhaustively.  Rather than grid search, we deployed a closed-loop
# system where Claude Code interpreted experiment results,
# maintained a structured research summary, and proposed the next intervention.
# At each iteration the agent selected parent configurations to mutate
# using an Upper Confidence Bound (UCB) tree search that balances
# exploitation of high-performing branches with exploration of
# under-visited regions.  The system implemented a form of automated
# scientific reasoning: testable hypotheses were drawn, repeatable
# experiments validated or falsified them, and causal understanding
# progressively emerged [1].
#
# <object type="image/svg+xml" data="assets/Fig_agentic_loop.svg" width="700"></object>
#
# The primary optimization target is not prediction accuracy (which
# is easy) but **identifiability**: the coefficient of variation
# (CV) of connectivity $R^2$ across random seeds measures how
# consistently a configuration escapes the degenerate solution landscape.
# Across five explorations (600 iterations), the agent established
# transferable principles and falsified hypotheses.  At each
# noise level, identifiability fails for a different reason and
# requires a different solution.

# %% [markdown]
# ## Identifiability Across Noise Regimes
#
# The three noise conditions reveal three distinct faces of the
# ill-posedness problem.  At $\sigma{=}0$ the bottleneck is
# geometric capacity of the representation space.  At
# $\sigma{=}0.05$ it is the sharpness of the optimization
# landscape.  At $\sigma{=}0.5$ it is the non-convexity of the
# loss surface itself.  The agent discovered each mechanism through
# hypothesis-driven reasoning chains spanning tens to hundreds of
# iterations.

# %% [markdown]
# ### Noise-free ($\sigma = 0$): Geometric capacity of the embedding space
# *108 iterations, 9 blocks*
#
# At $\sigma{=}0$ the GNN predicts voltage derivatives accurately
# regardless of seed, yet ~25% of initializations catastrophically
# fail to recover the true connectivity ($R^2 < 0.80$).  After
# 60 iterations and 14 falsified hypotheses, the agent identified
# the root cause: 65 neuron types cannot be reliably separated in
# a 2D embedding space.  Some seeds converge to configurations
# where distinct types collapse onto overlapping regions, producing
# contradictory gradients.  Increasing to `embedding_dim=4`
# eliminated every failure: extra dimensions provide escape
# directions that prevent type collapse.  `embedding_dim=6`
# reintroduced sensitivity, confirming 4 as the sweet spot.
#
# **Best config**: `embedding_dim=4`, `g_phi_diff=1500`,
# `n_epochs=1`, `aug_loop=20`.
# **Result**: $R^2 = 0.93 \pm 0.01$, **CV = 0.30%** (4/4 seeds
# $> 0.92$).

# %% [markdown]
# ### Low noise ($\sigma = 0.05$): A single constraint, sharp overfitting
# *220 iterations, 19 blocks*
#
# With one training epoch, all L1/L2 regularizers are inactive (annealing
# multiplier = 0 at epoch 0).  Recovery relies on the monotonicity
# penalty alone (`g_phi_diff=750`).  The agent found three critical
# levers: `batch_size` $2 \to 4$, all learning rates scaled by
# $1.5\times$ (compensating for larger batches), and
# `data_augmentation_loop` $20 \to 35$.  Augmentation was the single
# biggest gain, but the landscape has a razor-sharp cliff: `aug=36`
# triggers a $30\times$ CV explosion ($0.3\% \to 9.9\%$).
# Two-epoch training was tested twice and rejected — the epoch
# boundary introduces instability.  The champion survived 44
# perturbation tests across 18 blocks without being dethroned.
#
# **Best config**: `batch_size=4`, $1.5\times$ learning rates,
# `aug_loop=35`, `n_epochs=1`.
# **Result**: $R^2 = 0.98 \pm 0.01$, **CV = 0.3%** (all seeds
# $> 0.97$).

# %% [markdown]
# ### High noise ($\sigma = 0.5$): Bimodal landscape
# *112 iterations, 10 blocks*
#
# At $\sigma{=}0.5$ the landscape is **bimodal**: ~25% of seeds
# catastrophically fail ($R^2 \approx 0.20$) while the rest
# achieve near-perfect recovery.  This failure rate is
# LR-invariant, architecture-independent, and augmentation-
# insensitive — it appears fundamental to `randn_scaled`
# initialization.  The agent found that **noise cannot substitute
# for structural priors**: removing the monotonicity constraint
# (`g_phi_diff=0`) collapses $V^{\text{rest}}$ recovery despite
# perfect derivative fitting.  The agent initially identified
# `n_layers=4` as eliminating catastrophic failures (0/4 in
# exploration), but replication at $n{=}5$ showed the 4-layer
# architecture fails to converge on new seeds (conn $R^2 < 0.01$).
# The 3-layer architecture with the same optimized learning rates
# (`batch_size=2`, $\text{lr}_W{=}6{\times}10^{-4}$) remains the
# most reliable configuration.
#
# **Best config**: `n_layers=3`, `batch_size=2`, reduced learning rates,
# `aug_loop=20`, `n_epochs=1`.
# **Result**: pending $n{=}5$ replication.

# %% [markdown]
# ### Joint GNN + SIREN ($\sigma = 0.05$): 24 iterations, 2 blocks
#
# **SIREN depth $\geq 4$ eliminates catastrophic failures** (3-layer:
# 25% failure rate).  4 layers gives best connectivity ($R^2 = 0.94$,
# CV = 0.90%); 5 layers gives best field reconstruction
# ($R^2 = 0.83$, higher GNN variance).
# `omega=4096` outperforms 1024 in joint training — opposite of
# standalone SIREN.

# %% [markdown]
# ### Standalone SIREN ($\sigma = 0.05$): 152 iterations, 13 blocks
#
# Best field $R^2 = 0.90$ (`hidden_dim=768`, 7 layers,
# `omega=1750`, `lr=1.5e-7`, 60k steps).  The dominant lever is
# `omega` (initial frequency scale): increasing it from 30 to 1024
# yielded $+0.24$ $R^2$, exceeding all other tuning combined.
# Higher omega ($\geq 2000$) suffers from late-stage collapse,
# making 1750 optimal.  Learning rate scales inversely with omega
# ($\omega \times \text{lr} \approx 2.5 \times 10^{-4}$).

# %%
#| output: false
import os
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## References
#
# [1] C. Allier, L. Heinrich, M. Schneider, S. Saalfeld, "Graph
# neural networks uncover structure and functions underlying the
# activity of simulated neural assemblies," *arXiv:2602.13325*,
# 2026.
# [doi:10.48550/arXiv.2602.13325](https://doi.org/10.48550/arXiv.2602.13325)
#
# [2] B. Romera-Paredes et al., "Mathematical discoveries from
# program search with large language models," *Nature*, 2024.
#
# [3] A. Novikov et al., "AlphaEvolve: A coding agent for
# scientific and algorithmic exploration," 2025.

# %%
