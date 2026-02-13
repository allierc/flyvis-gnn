# Parallel Mode Addendum — Understanding Exploration

This addendum applies when running in **parallel mode** (GNN_LLM_parallel_flyvis_understanding.py). Follow all rules from the base instruction file, with these modifications.

## Batch Processing

- You receive **4 results per batch** and must propose **4 mutations**
- Each slot trains a **different FlyVis model** (not 4 variations of the same model)
- Write **4 separate** `## Iter N:` log entries (one per slot/iteration)
- Each iteration gets its own Node id in the UCB tree

## Slot Assignment (Fixed)

Each slot is permanently assigned to one flyvis model:

| Slot | Model ID | Dataset | Known Issue |
| ---- | -------- | ------- | ----------- |
| 0 | **049** | `fly_N9_62_1_id_049` | Low activity rank (svd_99=19), reduced dimensionality |
| 1 | **011** | `fly_N9_62_1_id_011` | High activity rank (svd_99=45) but worst R²=0.308, hard connectivity |
| 2 | **041** | `fly_N9_62_1_id_041` | Near-collapsed activity (svd_99=6), very low-dimensional |
| 3 | **003** | `fly_N9_62_1_id_003` | Moderate activity rank (svd_99=60), hard connectivity |

**DO NOT change the `dataset` field** in any config — each slot must always train its assigned model.

## Config Files

- Edit all 4 config files listed in the prompt: `{name}_00.yaml` through `{name}_03.yaml`
- Only modify `training:` and `graph_model:` parameters (and `claude:` where allowed)
- **DO NOT change `simulation:` parameters** — data is pre-generated

## Strategy Per Slot

Since each slot trains a different model, the strategy should be **model-specific**:

- For each slot, consider what is known about that model's failure mode
- Mutations should test hypotheses about WHY that specific model is difficult
- Use the UNDERSTANDING section in memory.md to guide decisions

**Suggested approach per batch:**
- For models with an `untested` hypothesis → **hypothesis-test**: design an experiment that can falsify it
- For models where the hypothesis was `falsified` → **explore**: try diverse configs to gather new evidence for a revised hypothesis
- For models with `partially supported` hypothesis → **hypothesis-test**: design a stronger test
- For models showing improvement → **exploit**: refine, but also ask WHY the improvement occurred

**CRITICAL**: The LLM can only hypothesize. Only training results and analysis tool outputs can validate or falsify. Never assume a hypothesis is correct. Design experiments that CAN falsify the hypothesis — a test that can only confirm is not informative.

## Cross-Model Comparison

After analyzing all 4 slots, look for **cross-model patterns**:
- Does a hyperparameter change help ALL models or only some?
- Do models with similar activity ranks respond similarly?
- Does the same failure mode (e.g., V_rest collapse) occur in all models or only specific ones?

Record cross-model observations in the **Cross-Model Observations** section of memory.md.

## Analysis Tool

When writing the analysis tool (Step 5), prioritize **cross-model analysis**:
- Compare W_true structure (sparsity, spectral properties) across all 4 models
- Compare learned W error patterns — are the same neuron types hard across models?
- Analyze how activity rank affects which edge types the GNN can learn
- Examine if there are common structural features in the connectivity of difficult models

The analysis tool has access to all 4 models' data simultaneously.

## Start Call (first batch, no results yet)

When the prompt says `PARALLEL START`:
- Read the base config to understand the training regime
- Read the generation logs for all 4 models
- Create 4 diverse initial training parameter variations
- Write initial hypotheses to the UNDERSTANDING section of memory.md
- The starting point is Node 79 params from the base exploration

## Logging Format

Same as base instructions, with the `Model:` field identifying which flyvis model:

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Model: [049/011/041/003]
Mode/Strategy: [exploit/explore/hypothesis-test]
Config: lr_W=X, lr=Y, lr_emb=Z, coeff_edge_diff=A, coeff_W_L1=B, batch_size=C, hidden_dim=D, recurrent=[T/F]
Metrics: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, test_pearson=F, training_time_min=G
Embedding: [visual observation]
Mutation: [param]: [old] -> [new]
Observation: [what does this tell us about WHY this model is difficult?]
Analysis: [one-line summary of analysis tool output, or "pending"]
Next: parent=P
```

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder. Always include the exact parameter change.

**CRITICAL**: The `Next: parent=P` line selects the parent for the **next batch's** mutations. `P` must refer to a node from a **previous** batch or the current batch — but NEVER set `Next: parent=P` where P is `id+1` (the next slot in the same batch).

Write all 4 entries before editing the 4 config files for the next batch.

## Failed Slots

If a slot is marked `[FAILED]` in the prompt:
- Write a brief `## Iter N: failed` entry noting the failure
- Still propose a mutation for that slot's config in the next batch
- Note which model failed — failure patterns are informative

## Training Time Monitoring

Check `training_time_min` for every successful slot. If any slot exceeds 60 minutes:
- Flag it in the observation
- Reduce complexity in the next mutation
