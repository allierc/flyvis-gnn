# FlyVis Experiment Log: fly_N9_63_1 (parallel)

## Block 1: Siren Learning Rate Exploration

### Initial Batch (Iter 0-3) - PARALLEL START

**Block Focus**: learning_rate_NNR_f (Siren learning rate)
**GNN params**: Fixed at proven 62_1 values (lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3)
**Hypothesis**: Siren lr is the key unknown. Range 1E-8 to 1E-4 explores 4 orders of magnitude.

---

## Iter 0: pending
Node: id=0, parent=root
Mode/Strategy: baseline
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-8, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=False
Metrics: (pending)
Mutation: baseline config - no changes from fly_N9_63_1.yaml
Parent rule: root - initial batch
Observation: (pending)
Next: (pending)

---

## Iter 1: pending
Node: id=1, parent=root
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-6, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=False
Metrics: (pending)
Mutation: lr_siren: 1E-8 -> 1E-6 (100x increase)
Parent rule: root - initial batch
Observation: (pending)
Next: (pending)

---

## Iter 2: pending
Node: id=2, parent=root
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-4, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=4096, recurrent=False
Metrics: (pending)
Mutation: lr_siren: 1E-8 -> 1E-4 (10000x increase, aggressive)
Parent rule: root - initial batch
Observation: (pending)
Next: (pending)

---

## Iter 3: pending
Node: id=3, parent=root
Mode/Strategy: explore
Config: lr_W=5E-4, lr=1.2E-3, lr_emb=1.5E-3, lr_siren=1E-5, coeff_edge_diff=750, coeff_W_L1=5E-5, batch_size=1, hidden_dim=64, hidden_dim_nnr_f=4096, omega_f=256, recurrent=False
Metrics: (pending)
Mutation: lr_siren: 1E-8 -> 1E-5, omega_f: 4096 -> 256 (mid lr + lower omega for stability)
Parent rule: root - initial batch
Observation: (pending)
Next: (pending)

---

