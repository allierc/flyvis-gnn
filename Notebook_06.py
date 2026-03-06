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
# a 2D embedding space.  With `embedding_dim=2`, some seeds
# converge to configurations where distinct types collapse onto
# overlapping regions, producing contradictory gradients.
# Increasing to `embedding_dim=4` eliminated every failure:
# extra dimensions provide escape directions that prevent type
# collapse.  `embedding_dim=6` reintroduced sensitivity,
# confirming 4 as the sweet spot.
#
# **Best config**: `embedding_dim=4`, `g_phi_diff=1500`,
# `n_epochs=1`, `aug_loop=20`.
# **Result**: $R^2 = 0.93 \pm 0.01$ (CV = 0.30%, 4/4 seeds
# $> 0.92$).

# %% [markdown]
# ### Low noise ($\sigma = 0.05$): A single constraint, sharp overfitting
# *220 iterations, 19 blocks*
#
# Recovery relies on the monotonicity penalty alone (`g_phi_diff=750`).  
# The lever is `data_augmentation_loop`: $20 \to 35$ improves robustness,
# but `aug=36` triggers a $30\times$ CV explosion ($0.3\% \to 9.9\%$)
# — a sharp overfitting cliff.
# **Result**: $R^2 = 0.98 \pm 0.01$ (CV = 0.3%).

# %% [markdown]
# ### High noise ($\sigma = 0.5$): Bimodal landscape
# *96 iterations, 8 blocks*
#
# At $\sigma{=}0.5$ the landscape is **bimodal**: ~25% of seeds
# catastrophically fail ($R^2 \approx 0.20$) while the rest
# achieve near-perfect recovery.  The agent classified five
# failure types and found that **noise cannot substitute for
# structural priors** — removing the monotonicity constraint
# collapses $V^{\text{rest}}$ recovery despite perfect derivative
# fitting.  Scaling learning rates by $1.5\times$ eliminates
# the bimodal mode entirely, with a sharp escape threshold
# at ${\sim}1.44\times$ (~6% transition band).
#
# **Best config**: $1.5\times$ learning rates, `aug_loop=20`,
# default architecture.
# **Result**: $R^2 = 1.00 \pm 0.01$ (CV = 0.64%, 4/4 seeds
# robust).

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
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import Image, display
from collections import Counter
from scipy.ndimage import gaussian_filter1d

# ── Configuration ───────────────────────────────────────────────

LOG_FILE = os.path.join(
    'log', 'Claude_exploration', 'LLM_flyvis_noise_005',
    'flyvis_noise_005_Claude_analysis.md',
)

MODES = [
    'Induction', 'Boundary', 'Deduction', 'Falsification',
    'Analogy', 'Abduction', 'Meta-reasoning', 'Regime',
    'Causal Chain', 'Constraint',
]

COLORS = {
    'Induction': '#2ecc71',
    'Abduction': '#9b59b6',
    'Deduction': '#3498db',
    'Falsification': '#e74c3c',
    'Analogy': '#f39c12',
    'Boundary': '#1abc9c',
    'Meta-reasoning': '#e91e63',
    'Regime': '#795548',
    'Causal Chain': '#00bcd4',
    'Constraint': '#ff5722',
}

DEFINITIONS = {
    'Induction': 'observations \u2192 pattern',
    'Abduction': 'observation \u2192 hypothesis',
    'Deduction': 'hypothesis \u2192 prediction',
    'Falsification': 'prediction failed \u2192 refine',
    'Analogy': 'cross-regime transfer',
    'Boundary': 'limit-finding',
    'Meta-reasoning': 'strategy adaptation',
    'Regime': 'phase identification',
    'Causal Chain': 'multi-step causation',
    'Constraint': 'parameter relationships',
}

# Reasoning-mode markers (case-insensitive patterns)
MODE_MARKERS = {
    'Falsification': [
        r'\bfalsified\b', r'\brejected\b', r'\bdoes NOT\b',
        r'\bdisproved\b', r'\bcontradicts\b',
        r'verdict:\s*falsified', r'falsified hypothesis',
    ],
    'Deduction': [
        r'hypothesis tested', r'\bpredict\b', r'\bexpect\b',
        r'if\b.*\bthen\b', r'\bshould\b.*\bachieve\b',
        r'test whether',
    ],
    'Induction': [
        r'established principle', r'\bconfirmed\b.*\bpattern\b',
        r'consistently\b', r'\boptimal\b.*\bfor\b',
        r'every.*tested', r'robust.*across',
        r'key finding', r'scales with',
    ],
    'Boundary': [
        r'\bboundary\b', r'\bcliff\b', r'\bthreshold\b',
        r'sweet spot', r'\boptimum\b', r'\bplateau\b',
        r'sharp.*optimum', r'minimum.*at\b', r'maximum.*at\b',
    ],
    'Analogy': [
        r'\btransfer\b', r'\bgeneralizes\b',
        r'based on.*block', r'from noise_',
        r'contrast with', r'analogous',
        r'same.*pattern.*as',
    ],
    'Abduction': [
        r'likely because', r'\bsuggests\b.*\bthat\b',
        r'caused by', r'mechanism:', r'explanation:',
        r'appears to',
    ],
    'Meta-reasoning': [
        r'strategy', r'need.*different.*approach',
        r'\bstuck\b', r'exhausted', r'reconsider',
        r'search.*ineffective',
    ],
    'Regime': [
        r'\bregime\b', r'phase transition', r'\bbimodal\b',
        r'fundamentally different', r'qualitatively',
        r'two.*distinct.*class',
    ],
    'Causal Chain': [
        r'because.*which.*cause', r'chain:',
        r'leads to.*which', r'cascade',
        r'mechanism.*\bvia\b',
    ],
    'Constraint': [
        r'\bconstrains\b', r'since.*failed.*must',
        r'\bimplies\b.*\bthat\b', r'\brequires\b.*\bthat\b',
    ],
}

def _hex_to_rgba(hex_color, alpha=0.4):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f'rgba({r},{g},{b},{alpha})'

# ── Parsing ──────────────────────────────────────────────────────

with open(LOG_FILE) as f:
    log_text = f.read()

# Extract batch (block) boundaries
block_defs = []  # (start_iter, end_iter, label)
for m in re.finditer(
    r'^## Batch\s+(\d+)\s+\(Iterations?\s+(\d+)-(\d+)\):\s*(.*)',
    log_text, re.MULTILINE,
):
    batch_num = int(m.group(1))
    start = int(m.group(2))
    end = int(m.group(3))
    desc = m.group(4).strip().rstrip(' —-')
    # Short label: "B1: desc"
    short = desc[:25] + '\u2026' if len(desc) > 25 else desc
    block_defs.append((start, end, f'B{batch_num}: {short}'))

# Extract iteration events
iter_blocks = re.split(r'^## Iter(?:ation)?\s+(\d+)', log_text, flags=re.MULTILINE)

all_events = []
for i in range(1, len(iter_blocks) - 1, 2):
    try:
        iter_num = int(iter_blocks[i])
    except ValueError:
        continue
    block = iter_blocks[i + 1]

    detected = set()
    for mode, patterns in MODE_MARKERS.items():
        for pat in patterns:
            if re.search(pat, block, re.IGNORECASE):
                detected.add(mode)
                break

    if re.search(r'hypothesis tested', block, re.IGNORECASE):
        detected.add('Deduction')
    if re.search(r'verdict:\s*falsified', block, re.IGNORECASE):
        detected.add('Falsification')

    has_high = bool(re.search(
        r'severe|catastroph|breakthrough|eliminat|critical|dramatic',
        block, re.IGNORECASE,
    ))
    significance = 'High' if has_high or len(block) > 800 else 'Medium'

    for mode in detected:
        all_events.append((iter_num, mode, significance))

# Build causal edges between consecutive iterations
all_edges = []
events_by_iter = {}
for it, mode, sig in all_events:
    events_by_iter.setdefault(it, []).append(mode)

sorted_iters = sorted(events_by_iter.keys())
for idx in range(len(sorted_iters) - 1):
    it_a, it_b = sorted_iters[idx], sorted_iters[idx + 1]
    ma, mb = events_by_iter[it_a], events_by_iter[it_b]
    if 'Deduction' in ma and 'Falsification' in mb:
        all_edges.append((it_a, 'Deduction', it_b, 'Falsification', 'leads_to'))
    if 'Induction' in ma and 'Deduction' in mb:
        all_edges.append((it_a, 'Induction', it_b, 'Deduction', 'triggers'))
    if 'Falsification' in ma and 'Induction' in mb:
        all_edges.append((it_a, 'Falsification', it_b, 'Induction', 'refines'))
    if 'Boundary' in ma and 'Boundary' in mb:
        all_edges.append((it_a, 'Boundary', it_b, 'Boundary', 'leads_to'))
    if 'Analogy' in ma and 'Deduction' in mb:
        all_edges.append((it_a, 'Analogy', it_b, 'Deduction', 'triggers'))
    if 'Deduction' in ma and 'Induction' in mb:
        all_edges.append((it_a, 'Deduction', it_b, 'Induction', 'leads_to'))
    if 'Abduction' in ma and 'Deduction' in mb:
        all_edges.append((it_a, 'Abduction', it_b, 'Deduction', 'triggers'))
    if 'Boundary' in ma and 'Induction' in mb:
        all_edges.append((it_a, 'Boundary', it_b, 'Induction', 'leads_to'))
    if 'Meta-reasoning' in ma and 'Induction' in mb:
        all_edges.append((it_a, 'Meta-reasoning', it_b, 'Induction', 'triggers'))
    if 'Constraint' in ma and 'Deduction' in mb:
        all_edges.append((it_a, 'Constraint', it_b, 'Deduction', 'triggers'))

mode_counts = Counter(e[1] for e in all_events)
max_iter = max(e[0] for e in all_events) if all_events else 1

print(f'Parsed {len(all_events)} events, {len(all_edges)} edges, '
      f'{len(block_defs)} blocks from noise_005 exploration')
print('Mode counts:')
for mode in MODES:
    c = mode_counts.get(mode, 0)
    if c > 0:
        print(f'  {mode}: {c}')

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
