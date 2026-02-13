#!/usr/bin/env python3
"""
Analysis tool for Understanding Exploration Batch 1 (Iters 1-4)

Purpose: Compute connectivity_R2 and investigate WHY these 4 models are difficult.

Key questions:
1. What is the actual connectivity_R2 for each trained model?
2. How does W_true structure differ between easy (standard) and difficult models?
3. Are there specific neuron types or edge categories that are harder to recover?
4. Does activity rank correlate with weight recovery difficulty at the edge level?
"""

import torch
import numpy as np
import os
from scipy import stats

# Configuration
MODEL_IDS = ['049', '011', '041', '003']
SLOTS = [0, 1, 2, 3]
DATA_DIR = 'graphs_data/fly'
LOG_DIR = 'log/fly'

print("=" * 70)
print("ANALYSIS TOOL: Batch 1 (Iters 1-4) - Understanding Difficult Models")
print("=" * 70)

# ============================================================================
# 1. Compute Connectivity R² for each trained model
# ============================================================================
print("\n=== CONNECTIVITY R² ANALYSIS ===")
print("Computing learned vs true weight correlation for each slot...\n")

connectivity_r2 = {}
w_stats = {}

for mid, slot in zip(MODEL_IDS, SLOTS):
    dataset_dir = f'{DATA_DIR}/fly_N9_62_1_id_{mid}'
    log_dir = f'{LOG_DIR}/fly_N9_62_1_understand_Claude_{slot:02d}'

    # Load ground truth weights
    w_true_path = f'{dataset_dir}/weights.pt'
    model_path = f'{log_dir}/models/best_model_with_0_graphs_0.pt'

    if not os.path.exists(w_true_path):
        print(f"Model {mid} (Slot {slot}): W_true not found at {w_true_path}")
        continue

    W_true = torch.load(w_true_path, weights_only=True, map_location='cpu').numpy()

    # Compute W_true statistics
    nnz = np.count_nonzero(W_true)
    w_stats[mid] = {
        'n_edges': len(W_true),
        'nonzero': nnz,
        'density': nnz / len(W_true),
        'mean': W_true.mean(),
        'std': W_true.std(),
        'min': W_true.min(),
        'max': W_true.max(),
        'abs_mean': np.abs(W_true).mean(),
        'n_weak': (np.abs(W_true) < 0.01).sum(),
        'n_moderate': ((np.abs(W_true) >= 0.01) & (np.abs(W_true) < 0.1)).sum(),
        'n_strong': (np.abs(W_true) >= 0.1).sum(),
    }

    # Load learned weights if model exists
    if not os.path.exists(model_path):
        print(f"Model {mid} (Slot {slot}): No trained model at {model_path}")
        connectivity_r2[mid] = None
        continue

    try:
        state_dict = torch.load(model_path, weights_only=False, map_location='cpu')
        W_learned = state_dict['model_state_dict']['W'].numpy().flatten()

        # Compute R²
        ss_res = np.sum((W_true - W_learned) ** 2)
        ss_tot = np.sum((W_true - W_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        # Compute correlation
        corr, _ = stats.pearsonr(W_true, W_learned)

        connectivity_r2[mid] = {
            'r2': r2,
            'pearson': corr,
            'rmse': np.sqrt(np.mean((W_true - W_learned) ** 2)),
            'W_learned_mean': W_learned.mean(),
            'W_learned_std': W_learned.std(),
        }

        print(f"Model {mid} (Slot {slot}):")
        print(f"  connectivity_R2 = {r2:.4f}")
        print(f"  connectivity_pearson = {corr:.4f}")
        print(f"  RMSE = {connectivity_r2[mid]['rmse']:.6f}")
        print(f"  W_true:  mean={W_true.mean():.6f}, std={W_true.std():.6f}")
        print(f"  W_learned: mean={W_learned.mean():.6f}, std={W_learned.std():.6f}")
        print()

    except Exception as e:
        print(f"Model {mid} (Slot {slot}): Error loading model - {e}")
        connectivity_r2[mid] = None

# ============================================================================
# 2. Compare W_true Structure Across Models
# ============================================================================
print("\n=== W_TRUE STRUCTURE COMPARISON ===")
print("Comparing ground truth weight distributions across all 4 difficult models...\n")

print(f"{'Model':<10} {'N_edges':<10} {'Nonzero':<10} {'Density':<10} {'Mean':<12} {'Std':<12} {'|W|<0.01':<12} {'|W|>=0.1':<12}")
print("-" * 90)
for mid in MODEL_IDS:
    if mid in w_stats:
        s = w_stats[mid]
        print(f"{mid:<10} {s['n_edges']:<10} {s['nonzero']:<10} {s['density']:<10.4f} {s['mean']:<12.6f} {s['std']:<12.6f} {s['n_weak']:<12} {s['n_strong']:<12}")

# ============================================================================
# 3. Per-Neuron-Type Weight Analysis
# ============================================================================
print("\n\n=== PER-NEURON-TYPE WEIGHT ANALYSIS ===")
print("Analyzing weight recovery by source neuron type...\n")

for mid, slot in zip(MODEL_IDS, SLOTS):
    dataset_dir = f'{DATA_DIR}/fly_N9_62_1_id_{mid}'
    log_dir = f'{LOG_DIR}/fly_N9_62_1_understand_Claude_{slot:02d}'

    # Load edge index and metadata
    edge_index_path = f'{dataset_dir}/edge_index.pt'
    w_true_path = f'{dataset_dir}/weights.pt'
    model_path = f'{log_dir}/models/best_model_with_0_graphs_0.pt'

    if not all(os.path.exists(p) for p in [edge_index_path, w_true_path]):
        continue

    edge_index = torch.load(edge_index_path, weights_only=True, map_location='cpu').numpy()
    W_true = torch.load(w_true_path, weights_only=True, map_location='cpu').numpy()

    # Load metadata to get neuron types
    try:
        import zarr
        metadata_path = f'{dataset_dir}/x_list_0/metadata.zarr'
        metadata = zarr.open(metadata_path, 'r')[:]
        neuron_types = metadata[:, 2].astype(int)  # Column 2 is neuron_type
    except:
        print(f"Model {mid}: Could not load metadata, skipping per-type analysis")
        continue

    # Get source neuron types for each edge
    source_nodes = edge_index[0]  # Shape: [n_edges]
    source_types = neuron_types[source_nodes]

    if not os.path.exists(model_path):
        print(f"Model {mid}: No trained model for per-type analysis")
        continue

    try:
        state_dict = torch.load(model_path, weights_only=False, map_location='cpu')
        W_learned = state_dict['model_state_dict']['W'].numpy().flatten()
    except:
        continue

    # Compute R² per source neuron type
    unique_types = np.unique(source_types)
    type_r2 = {}

    for t in unique_types:
        mask = source_types == t
        if mask.sum() < 10:  # Skip types with few edges
            continue

        w_true_t = W_true[mask]
        w_learned_t = W_learned[mask]

        ss_res = np.sum((w_true_t - w_learned_t) ** 2)
        ss_tot = np.sum((w_true_t - w_true_t.mean()) ** 2)

        if ss_tot > 1e-10:
            r2_t = 1 - ss_res / ss_tot
        else:
            r2_t = np.nan

        type_r2[t] = {
            'r2': r2_t,
            'n_edges': mask.sum(),
            'true_mean': w_true_t.mean(),
            'true_std': w_true_t.std(),
        }

    # Find hardest and easiest types
    valid_types = [(t, d['r2']) for t, d in type_r2.items() if not np.isnan(d['r2'])]
    if valid_types:
        valid_types.sort(key=lambda x: x[1])

        print(f"\nModel {mid} - Per-Type Recovery:")
        print(f"  Overall connectivity_R2: {connectivity_r2.get(mid, {}).get('r2', 'N/A'):.4f}" if mid in connectivity_r2 and connectivity_r2[mid] else "  Overall connectivity_R2: N/A")
        print(f"  Number of neuron types: {len(unique_types)}")
        print(f"  Hardest 5 types (lowest R²):")
        for t, r2 in valid_types[:5]:
            d = type_r2[t]
            print(f"    Type {t:3d}: R²={r2:.4f}, n_edges={d['n_edges']:6d}, true_mean={d['true_mean']:.6f}")
        print(f"  Easiest 5 types (highest R²):")
        for t, r2 in valid_types[-5:]:
            d = type_r2[t]
            print(f"    Type {t:3d}: R²={r2:.4f}, n_edges={d['n_edges']:6d}, true_mean={d['true_mean']:.6f}")

# ============================================================================
# 4. Weight Magnitude vs Recovery Error Analysis
# ============================================================================
print("\n\n=== WEIGHT MAGNITUDE VS RECOVERY ERROR ===")
print("Are weak edges harder to recover than strong edges?\n")

for mid, slot in zip(MODEL_IDS, SLOTS):
    dataset_dir = f'{DATA_DIR}/fly_N9_62_1_id_{mid}'
    log_dir = f'{LOG_DIR}/fly_N9_62_1_understand_Claude_{slot:02d}'
    model_path = f'{log_dir}/models/best_model_with_0_graphs_0.pt'
    w_true_path = f'{dataset_dir}/weights.pt'

    if not os.path.exists(model_path):
        continue

    W_true = torch.load(w_true_path, weights_only=True, map_location='cpu').numpy()

    try:
        state_dict = torch.load(model_path, weights_only=False, map_location='cpu')
        W_learned = state_dict['model_state_dict']['W'].numpy().flatten()
    except:
        continue

    # Compute absolute error
    error = np.abs(W_true - W_learned)
    abs_W = np.abs(W_true)

    # Bin by weight magnitude
    bins = [0, 0.001, 0.01, 0.1, 1.0, np.inf]
    bin_labels = ['<0.001', '0.001-0.01', '0.01-0.1', '0.1-1.0', '>1.0']

    print(f"Model {mid}:")
    for i in range(len(bins) - 1):
        mask = (abs_W >= bins[i]) & (abs_W < bins[i+1])
        if mask.sum() > 0:
            mean_error = error[mask].mean()
            relative_error = (error[mask] / (abs_W[mask] + 1e-8)).mean()
            print(f"  |W| {bin_labels[i]:>12}: n={mask.sum():7d}, mean_error={mean_error:.6f}, rel_error={relative_error:.4f}")
    print()

# ============================================================================
# 5. Embedding Analysis
# ============================================================================
print("\n=== LEARNED EMBEDDING ANALYSIS ===")
print("Analyzing the learned neuron embeddings...\n")

for mid, slot in zip(MODEL_IDS, SLOTS):
    log_dir = f'{LOG_DIR}/fly_N9_62_1_understand_Claude_{slot:02d}'
    model_path = f'{log_dir}/models/best_model_with_0_graphs_0.pt'

    if not os.path.exists(model_path):
        continue

    try:
        state_dict = torch.load(model_path, weights_only=False, map_location='cpu')
        embeddings = state_dict['model_state_dict']['a'].numpy()  # Shape: [n_neurons, embedding_dim]

        emb_mean = embeddings.mean(axis=0)
        emb_std = embeddings.std(axis=0)
        emb_range = embeddings.max(axis=0) - embeddings.min(axis=0)

        print(f"Model {mid}:")
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Mean per dim: {emb_mean}")
        print(f"  Std per dim: {emb_std}")
        print(f"  Range per dim: {emb_range}")

        # Check for collapse (all embeddings similar)
        inter_neuron_var = np.var(embeddings, axis=0).mean()
        print(f"  Inter-neuron variance: {inter_neuron_var:.6f}")

        # Check if embeddings have NaN or Inf
        has_nan = np.isnan(embeddings).any()
        has_inf = np.isinf(embeddings).any()
        print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
        print()

    except Exception as e:
        print(f"Model {mid}: Error loading embeddings - {e}")

# ============================================================================
# 6. Summary and Key Findings
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: KEY FINDINGS FOR UNDERSTANDING HYPOTHESES")
print("=" * 70)

print("\n1. CONNECTIVITY R² RESULTS:")
for mid in MODEL_IDS:
    if mid in connectivity_r2 and connectivity_r2[mid]:
        r2 = connectivity_r2[mid]['r2']
        baseline = {'049': 0.634, '011': 0.308, '041': 0.629, '003': 0.627}[mid]
        change = r2 - baseline
        print(f"   Model {mid}: R²={r2:.4f} (baseline={baseline:.4f}, change={change:+.4f})")
    else:
        print(f"   Model {mid}: No connectivity R² computed")

print("\n2. W_TRUE STRUCTURE PATTERNS:")
for mid in MODEL_IDS:
    if mid in w_stats:
        s = w_stats[mid]
        weak_pct = 100 * s['n_weak'] / s['n_edges']
        strong_pct = 100 * s['n_strong'] / s['n_edges']
        print(f"   Model {mid}: {weak_pct:.1f}% weak (|W|<0.01), {strong_pct:.1f}% strong (|W|>=0.1)")

print("\n3. IMPLICATIONS FOR HYPOTHESES:")
print("   - Check if models with more weak edges have lower connectivity_R²")
print("   - Check if embedding collapse (low variance) correlates with failure")
print("   - Compare per-type hardest types across models — are they the same?")

print("\n" + "=" * 70)
print("END OF ANALYSIS")
print("=" * 70)
