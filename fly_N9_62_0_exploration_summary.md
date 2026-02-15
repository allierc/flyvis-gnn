# Exploration Summary: fly_N9_62_0 (144 iterations, 6 blocks)

## Overview

The fly_N9_62_0 experiment explored hyperparameter space for a GNN learning the Drosophila visual connectome (65 neuron types, 13,741 neurons, 434,112 edges) from noise-free DAVIS visual input. The LLM-guided UCB tree search ran 144 iterations across 6 blocks of 24, with 4 parallel GPU slots per batch.

## Block-by-Block Progression

### Block 1 (iter 1-24): Learning Rate Exploration
- **Starting point**: Default config (lr_W=1E-3, lr=5E-4, lr_emb=1E-3, edge_diff=500)
- **Key discoveries**: lr_W=5E-4 improves tau (+51%), lr=1E-3 critical threshold (both 8E-4 and 2E-3 catastrophic), lr_emb=3.5E-3 narrow sweet spot
- **Best**: conn=0.823 (N21), tau=0.689 (N9), V_rest=0.272 (N14)

### Block 2 (iter 25-48): Regularization + Architecture
- **Breakthrough**: edge_diff=625 interpolation achieves both conn>0.8 AND tau>0.64 (N25: conn=0.839, tau=0.644)
- **Architecture**: hidden_dim_update=96 boosts tau (+17%), n_layers=4 boosts conn (+4%), n_layers=5 fails
- **Best**: conn=0.867 (N44), tau=0.752 (N38), V_rest=0.349 (N45), cluster=0.772 (N42)

### Block 3 (iter 49-72): Training Dynamics
- **Breakthrough**: aug_loop=30 achieves tau=0.911 (N50, +21% over best), but exceeds time limit
- **N66 multi-factor breakthrough**: n_layers=3 + n_layers_update=4 + lr_emb=4E-3 + aug=29 -> conn=0.889, V_rest=0.411
- **Failures**: recurrent_training always harmful, batch_size=2 cancels augmentation benefit
- **Best**: conn=0.889 (N66), tau=0.911 (N50), V_rest=0.411 (N66), cluster=0.793 (N49)

### Block 4 (iter 73-96): Combined Optimization
- **Four distinct optimization paths** emerge: N83 (conn), N96 (tau), N90 (V_rest), N92 (cluster)
- **Regime discovery**: edge_diff has THREE discrete optima (600, 620, 625) - intermediates fail
- **edge_norm=975 breakthrough**: exact optimum, +-5 both fail
- **Best**: conn=0.897 (N83), tau=0.895 (N96), V_rest=0.465 (N90), cluster=0.796 (N92)

### Block 5 (iter 97-120): Cross-Path Optimization
- **Meta-reasoning**: System recognizes edge_weight_L1 as the PRIMARY tuning parameter
- **Record**: conn=0.911 (N118 with edge_diff=600, edge_weight_L1=0.8)
- **Discovery**: edge_weight_L1=0.7 -> max V_rest (N119: 0.441), edge_weight_L1=0.8 -> max tau recovery
- **Best**: conn=0.911 (N118), tau=0.805 (N120), V_rest=0.441 (N119)

### Block 6 (iter 121-144): Final Optimization
- **Connectivity record**: N125 (edge_weight_L1=0.5, edge_diff=625) -> conn=0.929
- **Tau+V_rest record**: N133 (edge_weight_L1=0.7, edge_diff=620) -> tau=0.922, V_rest=0.484
- **Cluster record**: N143 (edge_weight_L1=0.65, edge_diff=620) -> cluster=0.824
- **Discrete coupling confirmed**: edge_weight_L1 values are strictly paired with edge_diff values

## Final Results

| Target | Node | conn_R2 | tau_R2 | V_rest_R2 | cluster | Key Parameters |
|--------|------|---------|--------|-----------|---------|----------------|
| **Connectivity** | N125 | **0.929** | 0.755 | 0.461 | 0.764 | edge_weight_L1=0.5, edge_diff=625 |
| **Tau+V_rest** | N133 | 0.868 | **0.922** | **0.484** | 0.789 | edge_weight_L1=0.7, edge_diff=620 |
| **Cluster** | N143 | 0.860 | 0.859 | 0.255 | **0.824** | edge_weight_L1=0.65, edge_diff=620 |
| **Balanced** | N124 | 0.886 | 0.878 | 0.463 | 0.765 | edge_weight_L1=0.6, edge_diff=625 |

## Improvement Over Baseline

| Metric | Baseline | Best | Improvement |
|--------|----------|------|-------------|
| Connectivity R2 | 0.723 | 0.929 | **+28.5%** |
| Tau R2 | 0.451 | 0.922 | **+104.4%** |
| V_rest R2 | 0.062 | 0.484 | **+680.6%** |
| Cluster Accuracy | 0.722 | 0.824 | **+14.1%** |

## Key Scientific Findings

### 1. Discrete Parameter Coupling
The most striking finding is that edge_diff and edge_weight_L1 form a **discrete coupling**: only specific pairings work, and intermediate values systematically fail. This suggests the loss landscape has discrete basins of attraction.

### 2. Four Optimization Regimes
Connectivity, tau recovery, V_rest recovery, and cluster accuracy cannot all be simultaneously maximized. The system discovered a **Pareto front** controlled by a single parameter (edge_weight_L1), with edge_diff selecting the regime.

### 3. Asymmetric Architecture Needs
Edge MLP and update MLP have opposite capacity requirements: edge MLP optimal at hidden_dim=64 (96 unstable), update MLP optimal at hidden_dim=96 (64 collapses V_rest). This reflects the different complexity of message computation vs. state update.

### 4. Augmentation as Implicit Regularization
data_augmentation_loop=29-30 dramatically improves tau recovery (0.590 -> 0.911), suggesting augmentation provides implicit temporal regularization that helps the network learn synaptic time constants.

---

## Scientific Poster Section: Code + LLM + Memory

### For a Short Poster Section

**Title**: LLM-Guided Hyperparameter Exploration with Persistent Memory

**Key results for poster**:

1. **Scale**: 144 iterations (6 blocks x 24), 4 parallel GPUs, ~50 min/iteration = ~30 GPU-hours total

2. **Performance**: LLM+memory achieved +28.5% connectivity, +104.4% tau recovery, +680.6% V_rest recovery over default configuration

3. **Reasoning quality**:
   - 74% deduction validation rate (28/38 predictions confirmed)
   - 88% transfer success rate (7/8 cross-block transfers succeeded)
   - 23 principles discovered beyond priors
   - 15 explicit falsifications refined the search

4. **Emergent reasoning capabilities observed**:
   - **Regime recognition** (iter 83): Identified discrete parameter optima, shifting from continuous search to discrete exploration
   - **Constraint propagation** (iter 131, 140): Deduced parameter coupling rules from two failed transfers
   - **Predictive modeling** (iter 133): Predicted exact optimal configuration from coupling model -> record tau=0.922
   - **Meta-reasoning** (iter 109): Recognized exhaustion of current search space, identified new primary parameter

5. **Memory enables**:
   - Cumulative knowledge across 6 blocks (23 principles accumulated)
   - Cross-block transfer of lr/architecture/regularization findings (88% success)
   - Prevention of repeated failures (15 falsified hypotheses stored and avoided)
   - Multi-factor combination requiring 4+ block history (N66 breakthrough at iter 66)

6. **Key figure**: UCB exploration tree showing 144 nodes, with breakthrough nodes highlighted (N25, N66, N96, N125, N133, N143). Color by R2 (green > 0.9, orange > 0.5, red < 0.5).

7. **Comparison with random search**: The LLM achieved its breakthrough at iter 25 (edge_diff=625 interpolation), 66 (multi-factor combination), and 125 (conn=0.929). A random search over the same parameter space (12 parameters, many continuous) would require >>144 iterations to find these specific combinations, particularly the discrete coupling between edge_weight_L1 and edge_diff.

### Noteworthy for Discussion

- **The 74% deduction rate is genuinely informative**: 26% failed predictions are not noise - they reveal fundamental parameter interactions (e.g., "deeper is better" falsified at n_layers=5, "lower regularization is better" falsified at edge_weight_L1=0.4)
- **The system self-corrects**: After each falsification, the hypothesis is refined (not abandoned), leading to increasingly precise principles
- **Timeline thresholds**: Single patterns at ~3 iter, sweet spots at ~14 iter, interpolation breakthroughs at ~25 iter, multi-factor combinations at ~66 iter, regime recognition at ~83 iter, meta-reasoning at ~109 iter
- **The discrete coupling discovery is non-trivial**: An LLM without memory would re-explore failed combinations. Memory prevented >10 redundant explorations in blocks 5-6
