# User Input

_Write instructions or advice here at any time during the exploration. The LLM reads this file at every batch and acknowledges pending items below._

## Pending Instructions

(none)

## Acknowledged

[ACK batch_133-136] **Regularization annealing bug with 1-epoch training.** Acknowledged. All 132 previous iterations ran with ALL L1/L2 regularizers at zero (annealing gives 0 at epoch 0). Champion config's R2=0.961-0.976 was achieved with only non-annealed constraints (coeff_g_phi_diff, coeff_g_phi_norm). Action: next batch (137-140) will test n_epochs=2 with data_augmentation_loop=15 (halved from 30) on the established 1.5x LR champion config. This enables epoch 0=free learning, epoch 1=L1/L2 at 39% strength.

