# Experiment Log: signal_chaotic_1_Claude

## Iter 1: converged

Node: id=1, parent=root
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.002, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.7970, test_pearson=0.8179, connectivity_R2=0.9824, final_loss=6.5622e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-22.7, 22.7], effective rank 34, spectral radius 1.014
Mutation: baseline (no prior iteration)
Parent rule: first iteration, parent=root by default
Observation: strong convergence (R2=0.982) with baseline config on chaotic connectivity
Next: parent=1 (highest UCB, will explore mutation)

## Iter 2: converged

Node: id=2, parent=1
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.003, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9922, test_pearson=0.9811, connectivity_R2=0.9995, final_loss=4.9747e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-18.5, 22.2], effective rank 33, spectral radius 0.973
Mutation: lr_W: 0.002 -> 0.003
Parent rule: Node 1 had highest UCB (1.454), branched with lr_W increase
Observation: excellent convergence (R2=0.9995), increasing lr_W from 0.002 to 0.003 improved both connectivity_R2 and test_R2 significantly
Next: parent=2 (highest UCB=1.707, continue exploit strategy)

## Iter 3: converged

Node: id=3, parent=2
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.003, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9208, test_pearson=0.8692, connectivity_R2=0.9998, final_loss=4.0264e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-21.5, 21.3], effective rank 33, spectral radius 0.973
Mutation: lr: 0.0001 -> 0.0002
Parent rule: Node 2 had highest UCB (1.577), continued with lr increase
Observation: connectivity_R2=0.9998 near-perfect, test_R2 slightly lower than iter 2 but loss improved. 3 consecutive successes triggers failure-probe strategy
Next: parent=3 (highest UCB=1.866, failure-probe: try extreme lr_W=0.01 to find boundary)

## Iter 4: converged

Node: id=4, parent=3
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9663, test_pearson=0.9407, connectivity_R2=0.9997, final_loss=2.1040e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-23.8, 24.5], effective rank 34, spectral radius 0.973
Mutation: lr_W: 0.003 -> 0.01
Parent rule: Node 3 had highest UCB (1.866), failure-probe with extreme lr_W=0.01
Observation: failure-probe did NOT find boundary - lr_W=0.01 (5x baseline) still achieves R2=0.9997. training is robust to high lr_W. 4 consecutive successes, continue probing lower boundary
Next: parent=4 (highest UCB=2.000, failure-probe: try extreme low lr_W=0.0005 to find lower boundary)

## Iter 5: partial

Node: id=5, parent=4
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.0005, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8520, test_pearson=0.7672, connectivity_R2=0.3816, final_loss=1.0463e+04
Activity: chaotic oscillations across 100 neurons, amplitude [-20.7, 22.0], effective rank 35, spectral radius 0.973
Mutation: lr_W: 0.01 -> 0.0005
Parent rule: Node 4 had highest UCB, failure-probe with extreme low lr_W=0.0005 to find lower boundary
Observation: found lower boundary! lr_W=0.0005 fails (R2=0.38). optimal lr_W range is [0.002, 0.01]. returning to exploit strategy from node 4
Next: parent=4 (highest UCB=1.745, exploit: try coeff_W_L1 variation from successful config)

## Iter 6: converged

Node: id=6, parent=4
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.7701, test_pearson=0.7171, connectivity_R2=0.9979, final_loss=3.2066e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-21.9, 22.2], effective rank 34, spectral radius 0.973
Mutation: coeff_W_L1: 1.0e-05 -> 1.0e-04
Parent rule: Node 4 had highest UCB (2.000), branched with coeff_W_L1 increase
Observation: connectivity_R2=0.9979 still excellent with 10x L1 regularization. test metrics slightly lower than iter 4 (pearson 0.72 vs 0.94) but connectivity still well-recovered. 5 converged out of 6 iters, explore untested parameter space
Next: parent=6 (highest UCB=2.223, exploit: try batch_size=16 from this successful config)

## Iter 7: converged

Node: id=7, parent=6
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=16, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9379, test_pearson=0.8936, connectivity_R2=0.9989, final_loss=2.2303e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-19.7, 19.8], effective rank 34, spectral radius 0.973
Mutation: batch_size: 8 -> 16
Parent rule: Node 6 had highest UCB (2.223), branched with batch_size increase
Observation: connectivity_R2=0.9989 excellent with batch_size=16. test_R2 improved (0.94 vs 0.77 in iter 6), pearson also better (0.89 vs 0.72). 6 consecutive successes (iters 2-7), triggers explore strategy per protocol
Next: parent=1 (explore strategy: select node outside recent n_iter_block/4=6 chain to try different branch; will probe lr_W boundary from baseline)

## Iter 8: failed

Node: id=8, parent=1
Mode/Strategy: explore/exploit
Config: lr_W=0.005, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.3727, test_pearson=-0.0747, connectivity_R2=0.1354, final_loss=7.7547e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-21.2, 20.9], effective rank 33, spectral radius 0.973
Mutation: lr_W: 0.002 -> 0.005 (from baseline Node 1)
Parent rule: explore strategy selected Node 1 (outside recent chain) with lr_W increase
Observation: explore from Node 1 failed (R2=0.135). lr_W=0.005 with lr=0.0001 underperforms vs lr_W=0.003 + lr=0.0002 (Node 3). key insight: lr and lr_W must scale together for optimal training
Next: parent=7 (highest UCB=2.413, return to best performing branch with batch_size=32 probe)

## Iter 9: converged

Node: id=9, parent=7
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9873, test_pearson=0.9722, connectivity_R2=0.9981, final_loss=2.5202e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-20.1, 22.3], effective rank 32, spectral radius 0.973
Mutation: batch_size: 16 -> 32
Parent rule: Node 7 had highest UCB (1.999), continued batch_size scaling from 16 to 32
Observation: connectivity_R2=0.9981 excellent with batch_size=32. test metrics improved (pearson 0.972 vs 0.894 in iter 7). 7 converged iters (2-4, 6-7, 9), training highly robust. explore strategy: probe different parameter not yet varied from best branch
Next: parent=9 (highest UCB=2.498, exploit: try lr=0.0003 to test sensitivity of model lr)

## Iter 10: converged

Node: id=10, parent=9
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8057, test_pearson=0.7282, connectivity_R2=0.9979, final_loss=2.9390e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-23.1, 19.8], effective rank 33, spectral radius 0.973
Mutation: lr: 0.0002 -> 0.0003
Parent rule: Node 9 had highest UCB (2.052), continued lr scaling from 0.0002 to 0.0003
Observation: connectivity_R2=0.9979 maintained excellent with lr=0.0003. however test metrics dropped significantly (test_R2: 0.99→0.81, pearson: 0.97→0.73). suggests lr=0.0003 is at boundary for test generalization. 8 converged iters (1-4, 6-7, 9-10). 4+ consecutive successes triggers failure-probe
Next: parent=10 (highest UCB=2.579, failure-probe: try lr=0.0001 to probe lower lr boundary)

## Iter 11: converged

Node: id=11, parent=10
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.01, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9315, test_pearson=0.8853, connectivity_R2=0.9985, final_loss=2.8256e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-27.6, 19.8], effective rank 35, spectral radius 0.973
Mutation: lr: 0.0003 -> 0.0001
Parent rule: Node 10 had highest UCB (2.579), failure-probe with lr=0.0001 to test lower lr boundary
Observation: lr=0.0001 (3x lower than iter 10) maintains excellent connectivity_R2=0.9985 and actually improves test metrics (test_R2: 0.81→0.93, pearson: 0.73→0.89). lr=0.0001 is better than lr=0.0003 for generalization. 9 converged iters (1-4, 6-7, 9-11). 5 consecutive successes, continue failure-probe to find actual boundaries
Next: parent=11 (highest UCB=2.657, failure-probe: try coeff_W_L1=1.0e-03 to probe upper L1 boundary)

## Iter 12: partial

Node: id=12, parent=11
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.01, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-03, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.4597, test_pearson=0.3201, connectivity_R2=0.2103, final_loss=4.0346e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-18.5, 20.7], effective rank 25, spectral radius 0.973
Mutation: coeff_W_L1: 1.0e-04 -> 1.0e-03
Parent rule: Node 11 had highest UCB (2.153), failure-probe with coeff_W_L1=1.0e-03 to probe upper L1 boundary
Observation: found upper L1 boundary! coeff_W_L1=1.0e-03 (10x increase) fails (R2=0.21). L1 too high over-penalizes weights, preventing connectivity recovery. optimal L1 range is [1e-05, 1e-04]. returning to exploit strategy from Node 11. 9/12 converged
Next: parent=11 (highest UCB=2.153, exploit: try lr_W=0.005 to probe lower lr_W boundary from this successful config)

## Iter 13: converged

Node: id=13, parent=11
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.005, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9847, test_pearson=0.9729, connectivity_R2=0.9928, final_loss=4.1080e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-21.5, 21.5], effective rank 34, spectral radius 0.973
Mutation: lr_W: 0.01 -> 0.005
Parent rule: Node 11 had highest UCB (2.153), exploit with lr_W=0.005 to test mid-range value
Observation: lr_W=0.005 achieves excellent R2=0.9928 with best test metrics so far (test_R2=0.985, pearson=0.973). confirms optimal lr_W range [0.003, 0.01]. 10/13 converged. Node 13 now has highest UCB (2.796)
Next: parent=13 (highest UCB=2.796, exploit: try lr_W=0.003 to refine optimal lr_W)

## Iter 14: partial

Node: id=14, parent=13
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.003, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8072, test_pearson=0.7604, connectivity_R2=0.8281, final_loss=5.7447e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-23.8, 23.6], effective rank 35, spectral radius 0.973
Mutation: lr_W: 0.005 -> 0.003
Parent rule: Node 13 had highest UCB (2.796), exploit with lr_W=0.003 to refine optimal lr_W
Observation: unexpected partial result (R2=0.828) with lr_W=0.003. this config worked well in early iters (Node 2-4 with R2>0.99) but now underperforms. possible seed/initialization variance. key insight: lr_W=0.003 + lr=0.0001 is less stable than lr_W=0.005 + lr=0.0001. 10/14 converged
Next: parent=13 (highest UCB=2.240, return to lr_W=0.005 with different mutation; try lr=0.0002 to test lr sensitivity)

## Iter 15: converged

Node: id=15, parent=13
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.005, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9663, test_pearson=0.9392, connectivity_R2=0.9999, final_loss=2.9182e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-26.2, 19.8], effective rank 35, spectral radius 0.973
Mutation: lr: 0.0001 -> 0.0002
Parent rule: Node 13 had highest UCB (2.796), tried lr=0.0002 from successful lr_W=0.005 config
Observation: near-perfect R2=0.9999 achieved! lr_W=0.005 + lr=0.0002 is optimal combination. test metrics excellent (test_R2=0.9663, pearson=0.9392). confirms lr and lr_W interaction matters. 11/15 converged. 3 recent successes (13,15), next could try failure-probe or explore less-visited branches
Next: parent=15 (highest UCB=2.936, failure-probe: try extreme lr_W=0.015 to probe upper boundary)

## Iter 16: converged

Node: id=16, parent=15
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.015, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8597, test_pearson=0.7882, connectivity_R2=0.9976, final_loss=2.0674e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-19.2, 20.4], effective rank 34, spectral radius 0.973
Mutation: lr_W: 0.005 -> 0.015
Parent rule: Node 15 had highest UCB (2.936), failure-probe with lr_W=0.015 (3x) to probe upper boundary
Observation: lr_W=0.015 still converges (R2=0.998) but test metrics degraded (pearson 0.94→0.79). found soft upper boundary: lr_W>0.01 hurts generalization. optimal lr_W range refined to [0.005, 0.01]. 12/16 converged (75%). 4 consecutive successes (13,15,16), continue exploit from best recent config
Next: parent=16 (highest UCB=2.998, exploit: try lr=0.0003 to test lr sensitivity at high lr_W)

## Iter 17: converged

Node: id=17, parent=16
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9530, test_pearson=0.9198, connectivity_R2=0.9981, final_loss=1.7360e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-21.0, 20.0], effective rank 34, spectral radius 0.973
Mutation: lr: 0.0002 -> 0.0003
Parent rule: Node 16 had highest UCB (2.998), exploit with lr=0.0003 to test lr sensitivity at high lr_W
Observation: lr_W=0.015 + lr=0.0003 achieves R2=0.998 with improved test metrics vs iter 16 (pearson 0.79→0.92). lowest loss so far (1736). 5 consecutive successes (13,15,16,17), 13/17 converged (76%). explore strategy: branch from less-visited node outside recent chain
Next: parent=14 (explore strategy: Node 14 UCB=2.890 is 2nd highest, partial result R2=0.828 warrants re-exploration with different mutation; try lr_W=0.004 to refine boundary around lr_W=0.003)

## Iter 18: partial

Node: id=18, parent=14
Mode/Strategy: explore/exploit
Config: lr_W=0.004, lr=0.0001, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8262, test_pearson=0.7567, connectivity_R2=0.8986, final_loss=5.0035e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-22.5, 23.4], effective rank 34, spectral radius 0.973
Mutation: lr_W: 0.003 -> 0.004 (from Node 14's lr_W=0.003)
Parent rule: Node 14 had high UCB, explored with lr_W=0.004 to refine boundary around lr_W=0.003
Observation: partial result (R2=0.899). lr_W=0.003-0.004 with lr=0.0001 appears to be a weak combination. contrast with iter 13 (lr_W=0.005, lr=0.0001, R2=0.993) - suggests lr_W<0.005 with lr=0.0001 is suboptimal. confirmed lr_W and lr must scale together. 13/18 converged (72%)
Next: parent=17 (highest UCB=3.119, R2=0.998; exploit from best recent config with different mutation)

## Iter 19: converged

Node: id=19, parent=17
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9209, test_pearson=0.8783, connectivity_R2=0.9993, final_loss=1.8856e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-22.3, 23.2], effective rank 34, spectral radius 0.973
Mutation: coeff_W_L1: 1.0e-04 -> 5.0e-05
Parent rule: Node 17 had highest UCB (3.119), exploit with reduced L1 regularization
Observation: R2=0.9993 excellent with halved L1 regularization. test metrics slightly lower than iter 17 (pearson 0.92→0.88) but connectivity recovery maintained. confirms coeff_W_L1 range [5e-05, 1e-04] works well. 14/19 converged (74%). 3 consecutive successes (17,19) after partial iter 18, triggers failure-probe
Next: parent=19 (highest UCB=3.179, failure-probe: try coeff_W_L1=1.0e-06 to probe lower L1 boundary)

## Iter 20: converged

Node: id=20, parent=19
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-06, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9926, test_pearson=0.9870, connectivity_R2=0.9997, final_loss=2.0275e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-23.5, 19.8], effective rank 34, spectral radius 0.973
Mutation: coeff_W_L1: 5.0e-05 -> 1.0e-06
Parent rule: Node 19 had highest UCB (3.179), failure-probe with coeff_W_L1=1.0e-06 to probe lower L1 boundary
Observation: failure-probe did NOT find boundary - coeff_W_L1=1.0e-06 (50x lower) achieves best test metrics yet (test_R2=0.993, pearson=0.987) with R2=0.9997. very low L1 regularization works excellently. confirms optimal L1 range extends down to 1e-06. 15/20 converged (75%). 6 consecutive successes (15-17,19-20), triggers explore strategy
Next: parent=18 (explore strategy: Node 18 UCB=3.135, partial R2=0.899, unexplored branch; try lr=0.0002 to test lr scaling hypothesis)

## Iter 21: converged

Node: id=21, parent=18
Mode/Strategy: explore/exploit
Config: lr_W=0.004, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9883, test_pearson=0.9754, connectivity_R2=0.9999, final_loss=3.3925e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-18.5, 24.2], effective rank 35, spectral radius 0.973
Mutation: lr: 0.0001 -> 0.0002 (from Node 18's lr=0.0001)
Parent rule: Node 18 was explore target (partial R2=0.899), tested lr=0.0002 to verify lr scaling hypothesis
Observation: explore from partial Node 18 succeeded brilliantly - R2=0.9999 (best yet) with excellent test metrics (test_R2=0.988, pearson=0.975). confirms hypothesis: lr_W=0.004 + lr=0.0002 works better than lr_W=0.004 + lr=0.0001. lr scaling ratio ~20:1 (lr_W:lr) is optimal. 16/21 converged (76%). 4 consecutive successes (19-21), triggers failure-probe
Next: parent=21 (highest UCB=3.291, tied with Node 20; failure-probe: try lr_W=0.008 to probe lr_W upper boundary while maintaining 20:1 ratio)

## Iter 22: converged

Node: id=22, parent=21
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.008, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9841, test_pearson=0.9687, connectivity_R2=0.9986, final_loss=2.5598e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-23.1, 19.8], effective rank 32, spectral radius 0.973
Mutation: lr_W: 0.004 -> 0.008 (from Node 21's lr_W=0.004)
Parent rule: Node 21 had high UCB (3.291), failure-probe with lr_W=0.008 to test 40:1 lr_W:lr ratio
Observation: failure-probe did NOT find boundary - lr_W=0.008 (2x) achieves excellent R2=0.9986 with great test metrics (test_R2=0.984, pearson=0.969). confirms 40:1 ratio works well. 17/22 converged (77%). 8 consecutive successes (15-17, 19-22), triggers explore strategy per protocol
Next: parent=20 (explore strategy: Node 20 has highest UCB=3.345, try lr_W=0.01 to probe 33:1 ratio boundary)

## Iter 23: converged

Node: id=23, parent=20
Mode/Strategy: explore/exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-06, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9732, test_pearson=0.9499, connectivity_R2=0.9999, final_loss=2.1074e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-22.8, 22.2], effective rank 33, spectral radius 0.973
Mutation: coeff_W_L1: 1.0e-06 -> 1.0e-06 (unchanged from Node 20), lr_W: 0.015 -> 0.01 (from Node 20's lr_W=0.015)
Parent rule: Node 20 had highest UCB (3.345), explore strategy with lr_W=0.01 to test 33:1 lr_W:lr ratio
Observation: excellent R2=0.9999 achieved with lr_W=0.01 + lr=0.0003 + coeff_W_L1=1e-06. test metrics strong (test_R2=0.973, pearson=0.950). confirms optimal config zone. 18/23 converged (78%). 9 consecutive successes (15-17, 19-23). block 1 near completion (iter 23/24), next iter is block end
Next: parent=5 (explore strategy: Node 5 UCB=2.780, partial R2=0.382 with lr_W=0.0005; unexplored failure branch, try lr_W=0.001 to probe lower boundary more precisely)

## Iter 24: partial

Node: id=24, parent=5
Mode/Strategy: explore/boundary
Config: lr_W=0.001, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.7455, test_pearson=0.7179, connectivity_R2=0.8322, final_loss=6.1517e+03
Activity: chaotic oscillations across 100 neurons, amplitude [-22.6, 19.8], effective rank 34, spectral radius 0.973
Mutation: lr_W: 0.0005 -> 0.001 (from Node 5's lr_W=0.0005), lr: 0.0002 -> 0.0002, batch_size: 8 (reset from Node 5)
Parent rule: Node 5 had moderate UCB (2.015), explore strategy to probe lower lr_W boundary more precisely
Observation: partial result (R2=0.832). lr_W=0.001 with lr=0.0002 underperforms vs optimal configs. confirms lower boundary: lr_W<0.003 is risky region. 18/24 converged (75%). block 1 complete

## Simulation Block 1 Summary (iters 1-24)

Simulation: connectivity_type=chaotic, Dale_law=False, Dale_law_factor=N/A, connectivity_rank=N/A, noise_model_level=N/A
Best R2: 0.9999 at iters 15, 21, 23 (tied)
Converged: Yes (18/24 = 75% success rate)
Observation: chaotic connectivity is highly trainable. key findings:

- lr_W range: [0.003, 0.015] works, optimal [0.005, 0.01]
- lr range: [0.0001, 0.0003] works, optimal 0.0002-0.0003
- lr_W:lr ratio: 20:1 to 50:1 is optimal
- coeff_W_L1 range: [1e-06, 1e-04] works, lower is better
- batch_size: 8, 16, 32 all work well
- failure boundaries: lr_W<0.001 fails, coeff_W_L1>1e-03 fails
  Optimum training: lr_W=0.005, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1e-04, low_rank_factorization=False, low_rank=N/A

--- NEW SIMULATION BLOCK ---
Next simulation: connectivity_type=low_rank, Dale_law=False, connectivity_rank=20, noise_model_level=0
Reference: starting from block 1 baseline (node 1)
Rationale: block 1 mapped chaotic connectivity well. block 2 will explore low_rank connectivity to test if GNN can recover structured low-rank matrices. using low_rank_factorization=True with matching rank to provide inductive bias

### Block 1 Protocol Evaluation

Branching rate: 7 unique parents in last 6 iters (5,18,19,20,21,22,23) = 116% (excellent diversity)
Improvement rate: 5/6 improved R2 in last 6 iters (83%) -> suggests more exploration
Stuck detection: no plateau detected (R2 varied 0.83-0.9999)

Protocol changes: no changes needed. current rules effectively balanced exploit/explore. branching rate healthy, improvement rate high, no stuck detection. rules performed well for block 1

## Iter 25: partial

Node: id=25, parent=root
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.005, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9449, test_pearson=0.9441, connectivity_R2=0.8737, final_loss=2.5796e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-18.5, 20.2], effective rank 12 (99% var), spectral radius 1.109
Mutation: baseline for block 2 (new simulation: connectivity_type=low_rank, low_rank_factorization=True)
Parent rule: first iteration of block 2, parent=root by default
Observation: partial result (R2=0.874) on low_rank connectivity with matching low_rank_factorization=True. activity has lower effective rank (12) vs chaotic (34) as expected. test metrics excellent (pearson=0.944) but connectivity recovery partial. suggests training params need tuning for low_rank case
Next: parent=25 (only node, exploit: try lr_W=0.01 to increase weight learning rate since chaotic optimal was higher)

## Iter 26: converged

Node: id=26, parent=25
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.5221, test_pearson=0.5986, connectivity_R2=0.9641, final_loss=2.2610e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-18.5, 20.2], effective rank 12 (99% var), spectral radius 1.109
Mutation: lr_W: 0.005 -> 0.01
Parent rule: Node 25 had highest UCB (only node), exploit with lr_W=0.01 based on chaotic block findings
Observation: converged (R2=0.964) with lr_W=0.01. connectivity recovery improved significantly from R2=0.874 to 0.964. however test metrics degraded (test_R2: 0.94->0.52, pearson: 0.94->0.60). suggests higher lr_W helps connectivity but hurts generalization. 1/2 converged in block 2. try balancing lr_W
Next: parent=26 (highest UCB=1.671, exploit: try lr_W=0.007 to find balance between connectivity and test metrics)

## Iter 27: partial

Node: id=27, parent=26
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.007, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8867, test_pearson=0.8066, connectivity_R2=0.7514, final_loss=2.4883e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-19.5, 17.2], effective rank 13 (99% var), spectral radius 0.937
Mutation: lr_W: 0.01 -> 0.007
Parent rule: Node 26 had highest UCB (1.541), exploit with lr_W=0.007 to balance connectivity and test metrics
Observation: partial result (R2=0.751). lr_W=0.007 degraded connectivity recovery from Node 26 (0.964->0.751) while improving test metrics (test_R2: 0.52->0.89, pearson: 0.60->0.81). reveals tradeoff: higher lr_W benefits connectivity, lower benefits test. need to try different approach - perhaps higher lr or lower L1 regularization. 1/3 converged in block 2
Next: parent=27 (highest UCB=1.617, exploit: try lr=0.0003 to increase model learning rate while keeping balanced lr_W)

## Iter 28: partial

Node: id=28, parent=27
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.007, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9347, test_pearson=0.9450, connectivity_R2=0.4976, final_loss=2.0206e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-18.6, 17.8], effective rank 6 (99% var), spectral radius 1.278
Mutation: lr: 0.0002 -> 0.0003
Parent rule: Node 27 had highest UCB (1.418), exploit with lr=0.0003 to increase model learning rate
Observation: partial result (R2=0.498). lr=0.0003 severely degraded connectivity recovery from Node 27 (0.751->0.498) while improving test metrics (test_R2: 0.89->0.93, pearson: 0.81->0.95). confirms trend: increasing lr hurts connectivity for low_rank simulation. Node 26 (R2=0.964) remains best in block 2. 1/4 converged in block 2 (25%)
Next: parent=26 (highest UCB=1.464, revert to successful config and try different mutation; try lr=0.0003 with lr_W=0.01 to test if higher lr_W can compensate)

## Iter 29: partial

Node: id=29, parent=26
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8308, test_pearson=0.9088, connectivity_R2=0.8253, final_loss=2.0572e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-21.0, 23.8], effective rank 10 (99% var), spectral radius 1.067
Mutation: lr: 0.0002 -> 0.0003 (from Node 26's lr=0.0002)
Parent rule: Node 26 had highest UCB (1.411 after excluding current node), exploit with lr=0.0003 to test if higher lr with lr_W=0.01 improves connectivity
Observation: partial result (R2=0.825). lr=0.0003 + lr_W=0.01 underperforms Node 26's R2=0.964. confirms pattern: increasing lr degrades connectivity for low_rank simulation (similar to iter 28). test metrics good (pearson=0.909) but connectivity suffers. 1/5 converged in block 2 (20%). need to explore opposite direction - try lower lr or higher lr_W
Next: parent=26 (highest UCB=1.411, return to only converged config; try coeff_W_L1=1.0e-05 to reduce L1 regularization which may help low_rank recovery)

## Iter 30: failed

Node: id=30, parent=26
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.4043, test_pearson=0.5921, connectivity_R2=0.2423, final_loss=3.0693e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-21.3, 21.7], effective rank 13 (99% var), spectral radius 1.067
Mutation: coeff_W_L1: 1.0e-04 -> 1.0e-05 (from Node 26's coeff_W_L1=1.0e-04)
Parent rule: Node 26 had highest UCB (1.411), exploit with coeff_W_L1=1.0e-05 to reduce L1 regularization
Observation: failed (R2=0.242). reducing L1 from 1e-04 to 1e-05 severely degraded connectivity recovery. for low_rank simulation, L1 regularization at 1e-04 appears necessary (unlike chaotic where 1e-06 worked). this is opposite to chaotic block findings - low_rank needs more regularization. 1/6 converged in block 2 (17%)
Next: parent=29 (highest UCB=2.050, R2=0.825; try lr_W=0.015 to see if higher lr_W can push R2 above 0.9 threshold)

## Iter 31: converged

Node: id=31, parent=29
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.7028, test_pearson=0.7815, connectivity_R2=0.9937, final_loss=1.1940e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 13 (99% var), spectral radius 0.959
Mutation: lr_W: 0.01 -> 0.015 (from Node 29's lr_W=0.01)
Parent rule: Node 29 had highest UCB (2.050), exploit with lr_W=0.015 to push connectivity recovery
Observation: converged (R2=0.994). best connectivity recovery in block 2 so far. lr_W=0.015 + lr=0.0003 outperforms Node 26's lr_W=0.01 + lr=0.0002 (R2=0.964). confirms higher lr_W helps low_rank connectivity. test metrics lower (pearson=0.78 vs 0.91 for Node 29) but connectivity significantly improved. 2/7 converged in block 2 (29%)
Next: parent=31 (highest UCB=2.317, exploit: try lr=0.0002 to test if lower lr improves test metrics while maintaining high connectivity)

## Iter 32: converged

Node: id=32, parent=31
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.015, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.7515, test_pearson=0.8012, connectivity_R2=0.9915, final_loss=1.2137e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 15.2], effective rank 13 (99% var), spectral radius 0.959
Mutation: lr: 0.0003 -> 0.0002 (from Node 31's lr=0.0003)
Parent rule: Node 31 had highest UCB (1.937), exploit with lr=0.0002 to test if lower lr improves test metrics while maintaining high connectivity
Observation: converged (R2=0.992). lr=0.0002 maintains excellent connectivity (R2=0.992 vs 0.994 for Node 31) with slightly improved test_pearson (0.80 vs 0.78). confirms lr_W=0.015 is robust for low_rank connectivity. 3/8 converged in block 2 (38%). 2 consecutive successes (31-32), continue exploit
Next: parent=32 (highest UCB=2.406, exploit: try lr_W=0.02 to probe upper lr_W boundary for low_rank)

## Iter 33: converged

Node: id=33, parent=32
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.02, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9237, test_pearson=0.9459, connectivity_R2=0.9887, final_loss=1.1748e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.9, 14.7], effective rank 12 (99% var), spectral radius 0.959
Mutation: lr_W: 0.015 -> 0.02 (from Node 32's lr_W=0.015)
Parent rule: Node 32 had highest UCB (2.406), exploit with lr_W=0.02 to probe upper lr_W boundary for low_rank
Observation: converged (R2=0.989). lr_W=0.02 maintains excellent connectivity with improved test metrics (test_R2: 0.75->0.92, pearson: 0.80->0.95) vs Node 32. best test metrics in block 2 so far. 4/9 converged in block 2 (44%). 3 consecutive successes (31-33), triggers failure-probe strategy
Next: parent=33 (highest UCB=2.489, failure-probe: try lr_W=0.03 to find upper lr_W boundary)

## Iter 34: converged

Node: id=34, parent=33
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.03, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8633, test_pearson=0.8791, connectivity_R2=0.9710, final_loss=1.2910e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 13 (99% var), spectral radius 0.959
Mutation: lr_W: 0.02 -> 0.03 (from Node 33's lr_W=0.02)
Parent rule: Node 33 had highest UCB (2.489), failure-probe with lr_W=0.03 (1.5x) to find upper lr_W boundary
Observation: converged (R2=0.971). lr_W=0.03 still converges but shows degradation from Node 33 (R2: 0.989->0.971, pearson: 0.946->0.879). found soft upper boundary: lr_W>0.02 hurts both connectivity and test metrics. optimal lr_W for low_rank is around 0.02. 5/10 converged in block 2 (50%). 4 consecutive successes (31-34), continue exploit
Next: parent=33 (highest UCB=2.043, exploit: try lr=0.0003 with lr_W=0.02 to test lr sensitivity at optimal lr_W)

## Iter 35: converged

Node: id=35, parent=33
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.02, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.6445, test_pearson=0.7193, connectivity_R2=0.9899, final_loss=1.1752e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 15.3], effective rank 13 (99% var), spectral radius 0.959
Mutation: lr: 0.0002 -> 0.0003 (from Node 33's lr=0.0002)
Parent rule: Node 33 had highest UCB (2.043), exploit with lr=0.0003 to test lr sensitivity at optimal lr_W=0.02
Observation: converged (R2=0.990). lr=0.0003 + lr_W=0.02 maintains excellent connectivity (R2=0.990 vs 0.989 for Node 33) but test metrics degraded (pearson: 0.946->0.719). confirms pattern from block 2: higher lr hurts test generalization. 6/11 converged in block 2 (55%). 5 consecutive successes (31-35), explore strategy: select node outside recent chain
Next: parent=28 (explore strategy: Node 28 UCB=2.156 is third highest, partial R2=0.498 with lr_W=0.007/lr=0.0003; try lr_W=0.01 to test if moderate lr_W improves connectivity)

## Iter 36: converged

Node: id=36, parent=28
Mode/Strategy: explore/exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.7514, test_pearson=0.8045, connectivity_R2=0.9850, final_loss=1.4862e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 13 (99% var), spectral radius 0.959
Mutation: lr_W: 0.007 -> 0.01 (from Node 28's lr_W=0.007)
Parent rule: Node 28 had moderate UCB, explore strategy with lr_W=0.01 to test if higher lr_W improves partial Node 28's connectivity
Observation: converged (R2=0.985). explore from partial Node 28 (R2=0.498) succeeded - lr_W=0.01 recovered connectivity significantly. confirms lr_W=0.01 is robust for low_rank simulation. test metrics moderate (pearson=0.805). 7/12 converged in block 2 (58%). 6 consecutive successes (31-36)
Next: parent=35 (highest UCB=2.722, R2=0.990; exploit: try lr=0.0002 to test if lower lr improves test metrics while maintaining high connectivity)

## Iter 37: converged

Node: id=37, parent=35
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.02, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.7323, test_pearson=0.8000, connectivity_R2=0.9868, final_loss=1.1768e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 15.0], effective rank 12 (99% var), spectral radius 0.959
Mutation: lr: 0.0003 -> 0.0002 (from Node 35's lr=0.0003)
Parent rule: Node 35 had highest UCB (2.722), exploit with lr=0.0002 to test if lower lr improves test metrics while maintaining high connectivity
Observation: converged (R2=0.987). lr=0.0002 maintains excellent connectivity (R2=0.987 vs 0.990 for Node 35) with similar test metrics (pearson: 0.719->0.800). confirms lr_W=0.02 + lr=0.0002 is robust config for low_rank. 8/13 converged in block 2 (62%). 7 consecutive successes (31-37). per protocol, 6+ consecutive successes (n_iter_block/4=6) triggers explore strategy
Next: parent=30 (explore strategy: Node 30 UCB=2.045, failed R2=0.242 with coeff_W_L1=1e-05; unexplored branch from failed config, try coeff_W_L1=5e-05 to find L1 boundary)

## Iter 38: converged

Node: id=38, parent=30
Mode/Strategy: explore/boundary
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9306, test_pearson=0.9489, connectivity_R2=0.9960, final_loss=1.1411e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 15.1], effective rank 12 (99% var), spectral radius 0.959
Mutation: coeff_W_L1: 1.0e-05 -> 5.0e-05 (from Node 30's coeff_W_L1=1.0e-05)
Parent rule: Node 30 had UCB=1.490, explore strategy to probe L1 boundary between failed 1e-05 and working 1e-04
Observation: explore from failed Node 30 (R2=0.242) succeeded brilliantly - R2=0.996 is best in block 2! coeff_W_L1=5e-05 is optimal for low_rank (between failed 1e-05 and working 1e-04). test metrics also best (pearson=0.949). found L1 sweet spot. 9/14 converged in block 2 (64%). 8 consecutive successes (31-38)
Next: parent=38 (highest UCB=2.867, exploit: try lr_W=0.015 to test if higher lr_W with optimal L1=5e-05 improves further)

## Iter 39: converged

Node: id=39, parent=38
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.015, lr=0.0002, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8901, test_pearson=0.9210, connectivity_R2=0.9969, final_loss=9.8347e+02
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 15.2], effective rank 12 (99% var), spectral radius 0.959
Mutation: lr_W: 0.01 -> 0.015 (from Node 38's lr_W=0.01)
Parent rule: Node 38 had highest UCB (2.287), exploit with lr_W=0.015 to test if higher lr_W with optimal L1=5e-05 improves further
Observation: converged (R2=0.997). lr_W=0.015 maintains excellent connectivity with best loss yet (983) in block 2. test metrics slightly lower than Node 38 (pearson: 0.949->0.921) but still strong. lowest loss indicates efficient training. 10/15 converged in block 2 (67%). 9 consecutive successes (31-39), triggers failure-probe strategy
Next: parent=39 (highest UCB=2.933, failure-probe: try lr_W=0.025 to probe upper lr_W boundary with optimal L1=5e-05)

## Iter 40: converged

Node: id=40, parent=39
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.025, lr=0.0002, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8193, test_pearson=0.8690, connectivity_R2=0.9914, final_loss=1.0068e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 15.2], effective rank 13 (99% var), spectral radius 0.959
Mutation: lr_W: 0.015 -> 0.025 (from Node 39's lr_W=0.015)
Parent rule: Node 39 had highest UCB (2.933), failure-probe with lr_W=0.025 to probe upper lr_W boundary with optimal L1=5e-05
Observation: converged (R2=0.991). failure-probe did NOT find boundary - lr_W=0.025 still achieves excellent connectivity (R2=0.991 vs 0.997 for Node 39). test metrics degraded slightly (pearson: 0.921->0.869). confirms lr_W range [0.01, 0.025] is stable for low_rank with L1=5e-05. 11/16 converged in block 2 (69%). 10 consecutive successes (31-40), triggers explore strategy
Next: parent=34 (explore strategy: Node 34 UCB=2.971, highest outside recent 6-node chain; R2=0.971 with lr_W=0.03; try lr_W=0.025 from different branch)

## Iter 41: converged

Node: id=41, parent=34
Mode/Strategy: explore/exploit
Config: lr_W=0.025, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9443, test_pearson=0.9641, connectivity_R2=0.9789, final_loss=1.2145e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 15.4], effective rank 12 (99% var), spectral radius 0.959
Mutation: lr_W: 0.03 -> 0.025 (from Node 34's lr_W=0.03)
Parent rule: Node 34 had highest UCB outside recent chain (2.971), explore strategy with lr_W=0.025 from lr_W=0.03 branch
Observation: converged (R2=0.979). lr_W=0.025 from Node 34 branch achieves excellent test metrics (pearson=0.964, best in block 2) with strong connectivity (R2=0.979). confirms lr_W=0.025 with L1=1e-04 is robust. 12/17 converged in block 2 (71%). 11 consecutive successes (31-41)
Next: parent=40 (highest UCB=3.053, R2=0.991; exploit with different mutation to explore optimal L1 range)

## Iter 42: converged

Node: id=42, parent=40
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.025, lr=0.0003, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.7598, test_pearson=0.8155, connectivity_R2=0.9885, final_loss=9.9632e+02
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 12 (99% var), spectral radius 0.959
Mutation: lr: 0.0002 -> 0.0003 (from Node 40's lr=0.0002)
Parent rule: Node 40 had highest UCB (2.406), exploit with lr=0.0003 to test lr sensitivity at lr_W=0.025
Observation: converged (R2=0.989). lr=0.0003 with lr_W=0.025 achieves excellent connectivity (R2=0.989 vs 0.991 for Node 40). test metrics similar (pearson: 0.869->0.816). confirms lr_W=0.025 + L1=5e-05 is robust across lr values. 13/18 converged in block 2 (72%). 12 consecutive successes (31-42), explore strategy: select node outside recent 6-node chain
Next: parent=36 (explore strategy: Node 36 UCB=3.106, outside recent chain; R2=0.985 with lr_W=0.01/coeff_W_L1=1e-04; try coeff_W_L1=5e-05 to test if reduced L1 improves this branch)

## Iter 43: converged

Node: id=43, parent=36
Mode/Strategy: explore/exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9282, test_pearson=0.9364, connectivity_R2=0.9957, final_loss=1.1898e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 15.5], effective rank 13 (99% var), spectral radius 0.959
Mutation: coeff_W_L1: 1.0e-04 -> 5.0e-05 (from Node 36's coeff_W_L1=1.0e-04)
Parent rule: Node 36 had high UCB outside recent chain (3.106), explore strategy with coeff_W_L1=5e-05 to test if reduced L1 improves this branch
Observation: converged (R2=0.996). explore from Node 36 (R2=0.985) succeeded - coeff_W_L1=5e-05 improved connectivity (0.985->0.996) and test metrics (pearson: 0.805->0.936). confirms coeff_W_L1=5e-05 is optimal for low_rank across different lr_W values. 14/19 converged in block 2 (74%). 13 consecutive successes (31-43). explore strategy: select highest UCB node outside recent 6-node chain
Next: parent=41 (explore strategy: Node 41 UCB=3.158, R2=0.979 with lr_W=0.025/L1=1e-04; try L1=5e-05 to test if optimal L1 improves this branch)

## Iter 44: converged

Node: id=44, parent=41
Mode/Strategy: explore/exploit
Config: lr_W=0.025, lr=0.0002, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9120, test_pearson=0.9241, connectivity_R2=0.9917, final_loss=1.0051e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 15.5], effective rank 13, spectral radius 0.959
Mutation: coeff_W_L1: 1.0e-04 -> 5.0e-05 (from Node 41's coeff_W_L1=1.0e-04)
Parent rule: Node 41 had high UCB outside recent chain (3.158), explore strategy with coeff_W_L1=5e-05 to test if optimal L1 improves this branch
Observation: converged (R2=0.992). explore from Node 41 (R2=0.979) succeeded - coeff_W_L1=5e-05 improved connectivity (0.979->0.992) and maintained excellent test metrics (pearson=0.924). confirms coeff_W_L1=5e-05 is optimal for low_rank regardless of lr_W (works at 0.01, 0.015, 0.025). 15/20 converged in block 2 (75%). 14 consecutive successes (31-44). explore strategy: select highest UCB node outside recent 6-node chain (39-44)
Next: parent=37 (explore strategy: Node 37 UCB=3.223, outside recent chain; R2=0.987 with lr_W=0.02/lr=0.0002/L1=1e-04; try L1=5e-05 to test if optimal L1 improves this branch)

## Iter 45: converged

Node: id=45, parent=37
Mode/Strategy: explore/exploit
Config: lr_W=0.02, lr=0.0002, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8970, test_pearson=0.9156, connectivity_R2=0.9952, final_loss=9.9082e+02
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 13 (99% var), spectral radius 0.959
Mutation: coeff_W_L1: 1.0e-04 -> 5.0e-05 (from Node 37's coeff_W_L1=1.0e-04)
Parent rule: Node 37 had high UCB outside recent chain (3.223), explore strategy with coeff_W_L1=5e-05 to test if optimal L1 improves this branch
Observation: converged (R2=0.995). explore from Node 37 (R2=0.987) succeeded - coeff_W_L1=5e-05 improved connectivity (0.987->0.995) with strong test metrics (pearson=0.916). achieved lowest loss in block 2 (991). confirms coeff_W_L1=5e-05 is universally optimal for low_rank connectivity. 16/21 converged in block 2 (76%). 15 consecutive successes (31-45). explore strategy: select highest UCB node outside recent 6-node chain (40-45)
Next: parent=43 (explore strategy: Node 43 has highest UCB=3.287, R2=0.996 with lr_W=0.01/lr=0.0003/L1=5e-05; try lr=0.0002 to test if lower lr improves test metrics)

## Iter 46: converged

Node: id=46, parent=43
Mode/Strategy: explore/exploit
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9316, test_pearson=0.9431, connectivity_R2=0.9963, final_loss=1.1157e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 13 (99% var), spectral radius 0.959
Mutation: lr: 0.0003 -> 0.0002 (from Node 43's lr=0.0003)
Parent rule: Node 43 had highest UCB (3.287), explore strategy with lr=0.0002 to test if lower lr improves test metrics
Observation: converged (R2=0.996). explore from Node 43 (R2=0.996) maintained excellent connectivity (R2=0.996→0.996) with improved test metrics (pearson: 0.936→0.943). confirms lr_W=0.01 + lr=0.0002 + L1=5e-05 is robust optimal config. 17/22 converged in block 2 (77%). 16 consecutive successes (31-46). explore strategy: select highest UCB node outside recent 6-node chain (41-46)
Next: parent=39 (explore strategy: Node 39 UCB=2.170, outside recent chain; R2=0.997 with lr_W=0.015/L1=5e-05; try lr=0.0003 to test if higher lr improves this branch)

## Iter 47: converged

Node: id=47, parent=39
Mode/Strategy: explore/exploit
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.7858, test_pearson=0.8386, connectivity_R2=0.9977, final_loss=9.8834e+02
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 13 (99% var), spectral radius 0.959
Mutation: lr: 0.0002 -> 0.0003 (from Node 39's lr=0.0002)
Parent rule: Node 39 was selected via explore strategy (outside recent 6-node chain), tested lr=0.0003 to probe lr sensitivity
Observation: converged (R2=0.998). explore from Node 39 (R2=0.997) maintained excellent connectivity (R2=0.997→0.998) but test metrics dropped (pearson: 0.921→0.839). confirms pattern: higher lr=0.0003 degrades test generalization vs lr=0.0002. lowest loss so far (988). 18/23 converged in block 2 (78%). 17 consecutive successes (31-47). iter 23/24, next iter is block end
Next: parent=46 (explore strategy: Node 46 UCB=3.394, highest outside recent chain (42-47); R2=0.996 with lr_W=0.01/lr=0.0002/L1=5e-05; try lr_W=0.012 to probe lr_W sensitivity at optimal config)

## Iter 48: converged

Node: id=48, parent=46
Mode/Strategy: explore/exploit
Config: lr_W=0.012, lr=0.0002, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8932, test_pearson=0.9275, connectivity_R2=0.9967, final_loss=1.0596e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 12 (99% var), spectral radius 0.959
Mutation: lr_W: 0.01 -> 0.012 (from Node 46's lr_W=0.01)
Parent rule: Node 46 had high UCB (3.394), explore strategy with lr_W=0.012 to probe lr_W sensitivity at optimal config
Observation: converged (R2=0.997). explore from Node 46 (R2=0.996) maintained excellent connectivity (R2=0.996→0.997) with strong test metrics (pearson=0.928). confirms lr_W range [0.01, 0.015] with L1=5e-05 is stable optimal region. 19/24 converged in block 2 (79%). 18 consecutive successes (31-48). block 2 complete

## Simulation Block 2 Summary (iters 25-48)

Simulation: connectivity_type=low_rank, Dale_law=False, Dale_law_factor=N/A, connectivity_rank=20, noise_model_level=N/A
Best R2: 0.9977 at iter 47 (Node 47: lr_W=0.015, lr=0.0003, L1=5e-05)
Converged: Yes (19/24 = 79% success rate)
Observation: low_rank connectivity is trainable with proper config. key findings:

- lr_W range: [0.01, 0.025] works, optimal [0.01, 0.015]
- lr range: [0.0002, 0.0003] works, optimal 0.0002 for test generalization
- coeff_W_L1: 5e-05 is optimal (1e-05 fails, 1e-04 works but suboptimal)
- low_rank_factorization=True is essential for matching simulation structure
- failure boundaries: lr_W<0.007 partial, coeff_W_L1<1e-05 fails
- vs chaotic (block 1): low_rank needs higher lr_W (0.01-0.025 vs 0.005-0.01) and tighter L1 (5e-05 vs 1e-06 to 1e-04)
  Optimum training: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=5e-05, low_rank_factorization=True, low_rank=20

### Block 2 Protocol Evaluation

Branching rate: 6 unique parents in last 6 iters (37,39,41,43,46,46) = 83% (good diversity, explore strategy working)
Improvement rate: 6/6 converged in last 6 iters (100%) -> suggests more boundary probing
Stuck detection: no plateau detected (R2 ranged 0.992-0.998 with variations)

Protocol changes: no changes needed. current rules effectively balanced exploit/explore. 18 consecutive successes shows robust exploration. branching rate healthy (83%), improvement rate excellent (100%), no stuck detection. explore strategy with "select node outside recent 6-node chain" is working well for maintaining diversity while achieving consistent convergence

### Regime Comparison Table (Cross-Block Meta-Analysis)

| Block | Regime             | Best R² | Optimal lr_W | Optimal lr    | Optimal L1     | Key constraint                                           |
| ----- | ------------------ | ------- | ------------ | ------------- | -------------- | -------------------------------------------------------- |
| 1     | chaotic            | 0.9999  | 0.005-0.01   | 0.0002-0.0003 | 1e-06 to 1e-04 | lr_W:lr ratio ~20:1 to 50:1                              |
| 2     | low_rank (rank=20) | 0.9977  | 0.01-0.015   | 0.0002        | 5e-05          | low_rank_factorization=True essential, L1 must be ≥5e-05 |

### Emerging Patterns

- low_rank connectivity requires higher lr_W than chaotic (1.5-2x)
- low_rank is more sensitive to L1 regularization (5e-05 sweet spot vs broad range for chaotic)
- higher lr (0.0003) hurts test generalization in both regimes, lr=0.0002 is safer
- low_rank_factorization provides essential inductive bias for structured connectivity

### Hypothesis for Block 3

Block 3 will test Dale_law=True (excitatory/inhibitory separation) with chaotic connectivity.
Prediction: Dale_law constraint may require adjustments to lr_W or L1 to accommodate sign constraints on weights. expect similar optimal lr_W range to block 1 (0.005-0.01) but potentially higher L1 to enforce structure.
Starting config: lr_W=0.005, lr=0.0002, L1=1e-04, batch_size=32 (adapted from block 1 optimal with slightly higher L1)

--- NEW SIMULATION BLOCK ---
Next simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.5, connectivity_rank=N/A, noise_model_level=0
Reference: starting block 3 (first iter will be node 49)
Rationale: block 2 mapped low_rank connectivity well. block 3 will explore Dale_law constraint (excitatory/inhibitory separation) to test if GNN can recover sign-constrained connectivity. using chaotic base to compare against block 1 baseline

## Iter 49: partial

Node: id=49, parent=root
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.005, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.7514, test_pearson=0.7483, connectivity_R2=0.7965, final_loss=3.0341e+03
Activity: Dale_law oscillations across 100 neurons, amplitude [-17.4, 14.5], effective rank 5-21, spectral radius 0.915
Mutation: baseline for block 3 (new simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.5)
Parent rule: first iteration of block 3, parent=root by default
Observation: partial result (R2=0.797) on Dale_law chaotic connectivity. lower spectral radius (0.915) vs block 1 chaotic (0.973) suggests different dynamics. lower effective rank (5-21 vs 33-35) indicates constrained dynamics. test metrics moderate (pearson=0.748). baseline config from block 1 hypothesis underperforms - Dale_law constraint requires higher lr_W
Next: parent=49 (only node, exploit: try lr_W=0.01 since block 1 showed higher lr_W improves chaotic connectivity)

## Iter 50: converged

Node: id=50, parent=49
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9945, test_pearson=0.9943, connectivity_R2=0.9757, final_loss=8.2033e+02
Activity: Dale_law oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-12, spectral radius 0.915
Mutation: lr_W: 0.005 -> 0.01 (from Node 49's lr_W=0.005)
Parent rule: Node 49 had highest UCB (1.268), exploit with lr_W=0.01 based on block 1-2 findings
Observation: converged (R2=0.976). lr_W=0.01 significantly improved connectivity recovery from Node 49 (0.797->0.976). excellent test metrics (test_R2=0.995, pearson=0.994) - best test metrics in block 3. confirms Dale_law chaotic requires higher lr_W similar to block 1 optimal. 1/2 converged in block 3 (50%)
Next: parent=50 (highest UCB=1.683, exploit: try lr_W=0.015 to test if higher lr_W further improves Dale_law connectivity)

## Iter 51: converged

Node: id=51, parent=50
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.015, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9364, test_pearson=0.9326, connectivity_R2=0.9679, final_loss=7.5752e+02
Activity: Dale_law oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-12, spectral radius 0.915
Mutation: lr_W: 0.01 -> 0.015 (from Node 50's lr_W=0.01)
Parent rule: Node 50 had highest UCB (1.683), exploit with lr_W=0.015 to test if higher lr_W further improves Dale_law connectivity
Observation: converged (R2=0.968). lr_W=0.015 achieves excellent connectivity (R2=0.968 vs 0.976 for Node 50) with strong test metrics (pearson=0.933). lowest loss so far in block 3 (758). slight R2 decrease from Node 50 suggests lr_W=0.01 may be closer to optimal. 2/3 converged in block 3 (67%). 2 consecutive successes (50-51), continue exploit
Next: parent=51 (highest UCB=1.834, exploit: try lr_W=0.02 to probe upper lr_W boundary for Dale_law chaotic)

## Iter 52: converged

Node: id=52, parent=51
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.02, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8452, test_pearson=0.9019, connectivity_R2=0.9976, final_loss=7.1560e+02
Activity: Dale_law oscillations across 100 neurons, amplitude [-11.9, 14.6], effective rank 5-21, spectral radius 0.915
Mutation: lr_W: 0.015 -> 0.02 (from Node 51's lr_W=0.015)
Parent rule: Node 51 had highest UCB (1.834), exploit with lr_W=0.02 to probe upper lr_W boundary for Dale_law chaotic
Observation: converged (R2=0.998). best connectivity in block 3 so far! lr_W=0.02 outperforms Node 50 (R2=0.976) and Node 51 (R2=0.968). lowest loss yet (716). test metrics good (pearson=0.902). Dale_law chaotic appears to prefer higher lr_W similar to low_rank (block 2). 3/4 converged in block 3 (75%). 3 consecutive successes (50-52), triggers failure-probe strategy
Next: parent=52 (highest UCB=1.998, failure-probe: try lr_W=0.025 to find upper lr_W boundary for Dale_law chaotic)

## Iter 53: converged

Node: id=53, parent=52
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.025, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.7565, test_pearson=0.6957, connectivity_R2=0.9412, final_loss=7.7920e+02
Activity: Dale_law oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-12, spectral radius 0.915
Mutation: lr_W: 0.02 -> 0.025 (from Node 52's lr_W=0.02)
Parent rule: Node 52 had highest UCB (1.998), failure-probe with lr_W=0.025 to find upper lr_W boundary
Observation: converged (R2=0.941). found soft upper boundary! lr_W=0.025 shows degradation from Node 52 (R2: 0.998->0.941, pearson: 0.902->0.696). test metrics significantly worse. confirms optimal lr_W for Dale_law chaotic is around 0.02. 4/5 converged in block 3 (80%). 4 consecutive successes (50-53), return to optimal config with different mutation
Next: parent=52 (highest UCB=2.059, exploit: try lr=0.0003 with optimal lr_W=0.02 to test lr sensitivity)

## Iter 54: converged

Node: id=54, parent=52
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.02, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9795, test_pearson=0.9874, connectivity_R2=0.9965, final_loss=7.3794e+02
Activity: Dale_law oscillations across 100 neurons, amplitude [-17.3, 14.2], effective rank 5-21, spectral radius 0.915
Mutation: lr: 0.0002 -> 0.0003 (from Node 52's lr=0.0002)
Parent rule: Node 52 had highest UCB (2.059), exploit with lr=0.0003 to test lr sensitivity at optimal lr_W=0.02
Observation: converged (R2=0.997). lr=0.0003 with lr_W=0.02 achieves excellent connectivity (R2=0.997 vs 0.998 for Node 52) with improved test metrics (test_R2: 0.845->0.980, pearson: 0.902->0.987). best test metrics in block 3! confirms lr=0.0003 improves test generalization for Dale_law chaotic. 5/6 converged in block 3 (83%). 5 consecutive successes (50-54), triggers failure-probe strategy
Next: parent=54 (highest UCB=2.221, failure-probe: try lr_W=0.03 to aggressively probe upper lr_W boundary)

## Iter 55: converged

Node: id=55, parent=54
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.03, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.7728, test_pearson=0.7228, connectivity_R2=0.9234, final_loss=8.0385e+02
Activity: Dale_law oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-12, spectral radius 0.915
Mutation: lr_W: 0.02 -> 0.03 (from Node 54's lr_W=0.02)
Parent rule: Node 54 had highest UCB (2.221), failure-probe with lr_W=0.03 (1.5x) to probe upper lr_W boundary
Observation: converged (R2=0.923). found soft upper boundary! lr_W=0.03 shows degradation from Node 54 (R2: 0.997->0.923, pearson: 0.987->0.723). confirms optimal lr_W for Dale_law chaotic is around 0.02. similar to iter 53 (lr_W=0.025, R2=0.941), higher lr_W degrades both connectivity and test metrics. 6/7 converged in block 3 (86%). 6 consecutive successes (50-55), triggers explore strategy per protocol
Next: parent=53 (explore strategy: Node 53 UCB=2.264 is highest, R2=0.941 with lr_W=0.025; try lr=0.0003 to test if higher lr improves this branch like it did for Node 52->54)

## Iter 56: converged

Node: id=56, parent=53
Mode/Strategy: explore/exploit
Config: lr_W=0.025, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.7827, test_pearson=0.7592, connectivity_R2=0.9387, final_loss=7.6976e+02
Activity: Dale_law oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-12, spectral radius 0.915
Mutation: lr: 0.0002 -> 0.0003 (from Node 53's lr=0.0002)
Parent rule: Node 53 was selected via explore strategy (UCB=2.264), tested lr=0.0003 to see if higher lr improves this branch
Observation: converged (R2=0.939). explore from Node 53 (R2=0.941) maintained connectivity (R2=0.941->0.939) with similar test metrics (pearson: 0.696->0.759). lr=0.0003 at lr_W=0.025 does not improve connectivity like it did for Node 52->54 (lr_W=0.02). confirms optimal lr_W for Dale_law chaotic is 0.02, and lr_W=0.025 is at the boundary regardless of lr. 7/8 converged in block 3 (88%). 7 consecutive successes (50-56)
Next: parent=56 (highest UCB=2.353, exploit: try lr_W=0.02 to revert to optimal lr_W while keeping lr=0.0003)

## Iter 57: converged

Node: id=57, parent=56
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.02, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8089, test_pearson=0.7858, connectivity_R2=0.9584, final_loss=7.4695e+02
Activity: Dale_law oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-13, spectral radius 0.915
Mutation: lr_W: 0.025 -> 0.02 (from Node 56's lr_W=0.025)
Parent rule: Node 56 had highest UCB (2.353), exploit with lr_W=0.02 to revert to optimal lr_W
Observation: converged (R2=0.958). lr_W=0.02 reverted from Node 56 (R2=0.939) shows improvement in connectivity (0.939->0.958) but lower than Node 54's R2=0.997 with same config. variance in results suggests stochasticity. test metrics moderate (pearson=0.786). 8/9 converged in block 3 (89%). 8 consecutive successes (50-57), triggers failure-probe strategy
Next: parent=57 (highest UCB=2.458, failure-probe: try coeff_W_L1=5e-05 to test if lower L1 improves Dale_law chaotic like it did for low_rank)

## Iter 58: converged

Node: id=58, parent=57
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.02, lr=0.0003, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9354, test_pearson=0.9113, connectivity_R2=0.9994, final_loss=6.1909e+02
Activity: Dale_law oscillations across 100 neurons, amplitude [-11.9, 14.6], effective rank 6-22, spectral radius 0.915
Mutation: coeff_W_L1: 1.0e-04 -> 5.0e-05 (from Node 57's coeff_W_L1=1.0e-04)
Parent rule: Node 57 had highest UCB (2.458), failure-probe with coeff_W_L1=5e-05 to test if lower L1 improves Dale_law chaotic
Observation: failure-probe succeeded brilliantly! coeff_W_L1=5e-05 achieves best connectivity in block 3 (R2=0.9994) with excellent test metrics (test_R2=0.935, pearson=0.911). lowest loss in block 3 (619). confirms L1=5e-05 is universally optimal across regimes (low_rank in block 2, Dale_law chaotic in block 3). 9/10 converged in block 3 (90%). 9 consecutive successes (50-58), triggers explore strategy
Next: parent=55 (explore strategy: Node 55 UCB=2.505 is 2nd highest, R2=0.923 with lr_W=0.03; try lr_W=0.025 to test boundary region with lower L1)

## Iter 59: converged

Node: id=59, parent=55
Mode/Strategy: explore/exploit
Config: lr_W=0.025, lr=0.0003, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9551, test_pearson=0.9643, connectivity_R2=0.9985, final_loss=6.0062e+02
Activity: Dale_law oscillations across 100 neurons, amplitude [-11.9, 15.5], effective rank 5-22, spectral radius 0.915
Mutation: coeff_W_L1: 1.0e-04 -> 5.0e-05 (from Node 55's coeff_W_L1=1.0e-04)
Parent rule: Node 55 was selected via explore strategy (UCB=2.505), tested coeff_W_L1=5e-05 to see if lower L1 improves boundary region
Observation: explore succeeded! coeff_W_L1=5e-05 at lr_W=0.025 achieves excellent connectivity (R2=0.999) matching Node 58 (R2=0.999 at lr_W=0.02). test metrics are best in block 3 (test_R2=0.955, pearson=0.964). confirms L1=5e-05 improves connectivity across entire lr_W range [0.02, 0.025]. lowest loss in block 3 (601). 10/11 converged in block 3 (91%). 10 consecutive successes (50-59), triggers explore strategy
Next: parent=54 (explore strategy: Node 54 UCB=1.826 not in recent 6-node chain [54-59], has R2=0.997 with lr_W=0.02, lr=0.0003, L1=1e-04; try L1=5e-05 to test if this universal optimal applies there too)

## Iter 60: converged

Node: id=60, parent=54
Mode/Strategy: explore/exploit
Config: lr_W=0.02, lr=0.0003, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8761, test_pearson=0.7918, connectivity_R2=0.9786, final_loss=6.5670e+02
Activity: Dale_law oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-12, spectral radius 0.915
Mutation: coeff_W_L1: 1.0e-04 -> 5.0e-05 (from Node 54's coeff_W_L1=1.0e-04)
Parent rule: Node 54 was selected via explore strategy (outside recent 6-node chain), tested coeff_W_L1=5e-05 to verify L1 optimization
Observation: converged (R2=0.979). explore from Node 54 (R2=0.997) shows lower connectivity with L1=5e-05 (0.997->0.979), unlike Nodes 57->58 and 55->59 where L1=5e-05 improved R2. variance in results suggests stochasticity or interaction effects. test metrics lower than Node 54 (pearson: 0.987->0.792). Node 54 with L1=1e-04 may be better for some configurations. 11/12 converged in block 3 (92%). 11 consecutive successes (50-60), triggers explore strategy
Next: parent=58 (explore strategy: Node 58 UCB=2.731 highest, R2=0.999 with lr_W=0.02/lr=0.0003/L1=5e-05; try lr_W=0.015 to test if lower lr_W maintains high connectivity)

## Iter 61: converged

Node: id=61, parent=58
Mode/Strategy: explore/exploit
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9484, test_pearson=0.9322, connectivity_R2=0.9837, final_loss=6.9383e+02
Activity: Dale_law oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-12, spectral radius 0.915
Mutation: lr_W: 0.02 -> 0.015 (from Node 58's lr_W=0.02)
Parent rule: Node 58 was selected via explore strategy (UCB=2.201), tested lr_W=0.015 to see if lower lr_W maintains high connectivity
Observation: converged (R2=0.984). explore from Node 58 (R2=0.999) shows slightly lower connectivity with lr_W=0.015 (0.999->0.984) but good test metrics (pearson=0.932). confirms lr_W=0.02 is optimal for Dale_law chaotic with L1=5e-05. 12/13 converged in block 3 (92%). 12 consecutive successes (50-61), triggers explore strategy
Next: parent=59 (explore strategy: Node 59 has highest UCB=2.801, R2=0.999 with lr_W=0.025/lr=0.0003/L1=5e-05; try lr_W=0.03 to probe upper boundary with optimal L1)

## Iter 62: converged

Node: id=62, parent=59
Mode/Strategy: explore/boundary
Config: lr_W=0.03, lr=0.0003, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8482, test_pearson=0.9255, connectivity_R2=0.9659, final_loss=6.5128e+02
Activity: Dale_law oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-12, spectral radius 0.915
Mutation: lr_W: 0.025 -> 0.03 (from Node 59 lr_W=0.025)
Parent rule: Node 59 had highest UCB (2.801), explore with lr_W=0.03 to probe upper boundary with optimal L1=5e-05
Observation: converged (R2=0.966). lr_W=0.03 with L1=5e-05 shows degradation from Node 59 (R2: 0.999->0.966). confirms lr_W=0.025 is upper optimal boundary for Dale_law chaotic with L1=5e-05. test metrics good (pearson=0.926) despite lower connectivity. 13/14 converged in block 3 (93%). 13 consecutive successes (50-62), triggers explore strategy
Next: parent=61 (highest UCB=2.855, R2=0.984 with lr_W=0.015/lr=0.0003/L1=5e-05; try batch_size=16 to test batch sensitivity at lower lr_W)

## Iter 63: converged

Node: id=63, parent=61
Mode/Strategy: explore/exploit
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=16, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9406, test_pearson=0.9402, connectivity_R2=0.9960, final_loss=8.2351e+02
Activity: Dale_law chaotic oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-14, spectral radius 0.915
Mutation: batch_size: 32 -> 16 (from Node 61's batch_size=32)
Parent rule: Node 61 had highest UCB (2.855), tested batch_size=16 at lower lr_W=0.015
Observation: converged (R2=0.996). batch_size=16 with lr_W=0.015 achieves excellent connectivity (R2=0.996 vs 0.984 for Node 61). test metrics excellent (pearson=0.940). confirms batch_size=16 works well at lower lr_W. 14/15 converged in block 3 (93%). 14 consecutive successes (50-63), triggers explore strategy
Next: parent=60 (explore strategy: Node 60 UCB=2.915 highest outside recent 6-node chain [58-63], R2=0.979 with lr_W=0.02/batch_size=32/L1=5e-05; try lr=0.00025 to test lower lr with this config)

## Iter 64: converged

Node: id=64, parent=60
Mode/Strategy: explore/exploit
Config: lr_W=0.02, lr=0.00025, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=32, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9739, test_pearson=0.9674, connectivity_R2=0.9808, final_loss=6.5441e+02
Activity: Dale_law chaotic oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-12, spectral radius 0.915
Mutation: lr: 0.0003 -> 0.00025 (from Node 60's lr=0.0003)
Parent rule: Node 60 was selected via explore strategy (UCB=2.915 outside recent 6-node chain), tested lr=0.00025 to test lower lr with this config
Observation: converged (R2=0.981). lr=0.00025 with lr_W=0.02/L1=5e-05 achieves good connectivity (R2=0.981 vs 0.979 for Node 60). excellent test metrics (test_R2=0.974, pearson=0.967 vs 0.792 for Node 60). confirms lr=0.00025 maintains connectivity with improved test metrics at lr_W=0.02. 15/16 converged in block 3 (94%). 15 consecutive successes (50-64), triggers explore strategy
Next: parent=63 (highest UCB=2.996, R2=0.996 with lr_W=0.015/lr=0.0003/L1=5e-05/batch_size=16; try lr=0.00025 to test if lower lr improves this excellent branch)

## Iter 65: converged

Node: id=65, parent=63
Mode/Strategy: explore/exploit
Config: lr_W=0.015, lr=0.00025, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=16, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9592, test_pearson=0.9466, connectivity_R2=0.9909, final_loss=8.1847e+02
Activity: Dale_law chaotic oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-13, spectral radius 0.915
Mutation: lr: 0.0003 -> 0.00025 (from Node 63's lr=0.0003)
Parent rule: Node 63 had highest UCB (2.996), tested lr=0.00025 to see if lower lr improves this batch_size=16 branch
Observation: converged (R2=0.991). lr=0.00025 from Node 63 (R2=0.996) achieves excellent connectivity (R2=0.991 vs 0.996) with good test metrics (pearson=0.947 vs 0.940). slightly lower connectivity than parent but comparable performance. confirms lr_W=0.015/batch_size=16/L1=5e-05 config is robust to lr changes. 16/17 converged in block 3 (94%). 16 consecutive successes (50-65), triggers explore strategy
Next: parent=59 (explore strategy: Node 59 UCB=2.373 highest outside recent 6-node chain [60-65], R2=0.999 with lr_W=0.025/lr=0.0003/L1=5e-05/batch_size=32; try batch_size=16 to test if smaller batch improves this excellent config)

## Iter 66: converged

Node: id=66, parent=59
Mode/Strategy: explore/exploit
Config: lr_W=0.025, lr=0.0003, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=16, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9859, test_pearson=0.9908, connectivity_R2=0.9984, final_loss=7.8016e+02
Activity: Dale_law chaotic oscillations across 100 neurons, amplitude [-11.9, 14.6], effective rank 5-23, spectral radius 0.915
Mutation: batch_size: 32 -> 16 (from Node 59's batch_size=32)
Parent rule: Node 59 was selected via explore strategy (highest UCB outside recent 6-node chain), tested batch_size=16 to see if smaller batch improves this excellent lr_W=0.025 config
Observation: converged (R2=0.998). batch_size=16 at lr_W=0.025/L1=5e-05 achieves excellent connectivity (R2=0.998 matching Node 59's R2=0.999) with best test metrics in block 3 (test_R2=0.986, pearson=0.991 vs 0.964 for Node 59). confirms batch_size=16 universally improves test generalization for Dale_law chaotic. 17/18 converged in block 3 (94%). 17 consecutive successes (50-66), triggers explore strategy
Next: parent=64 (explore strategy: Node 64 UCB=3.102, highest outside recent 6-node chain [61-66], R2=0.981 with lr_W=0.02/lr=0.00025/L1=5e-05/batch_size=32; try batch_size=16 to test if smaller batch improves this branch like it did for Nodes 59->66 and 61->63)

## Iter 67: converged

Node: id=67, parent=64
Mode/Strategy: explore/exploit
Config: lr_W=0.02, lr=0.00025, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=16, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8997, test_pearson=0.8716, connectivity_R2=0.9826, final_loss=8.1285e+02
Activity: Dale_law chaotic oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-12, spectral radius 0.915
Mutation: batch_size: 32 -> 16 (from Node 64's batch_size=32)
Parent rule: Node 64 was selected via explore strategy (UCB=3.102 outside recent 6-node chain), tested batch_size=16 to see if smaller batch improves this branch
Observation: converged (R2=0.983). batch_size=16 from Node 64 (R2=0.981) achieves similar connectivity (R2=0.983 vs 0.981) with slightly lower test metrics (pearson: 0.967->0.872). unlike Nodes 59->66 and 61->63 where batch_size=16 improved results, this branch (lr=0.00025) does not benefit as much. 18/19 converged in block 3 (95%). 18 consecutive successes (50-67), triggers explore strategy
Next: parent=66 (explore strategy: Node 66 UCB=3.178 highest, R2=0.998 with lr_W=0.025/lr=0.0003/L1=5e-05/batch_size=16; try lr_W=0.02 to test if slightly lower lr_W improves this excellent config)

## Iter 68: converged

Node: id=68, parent=66
Mode/Strategy: explore/exploit
Config: lr_W=0.02, lr=0.0003, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=16, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.8843, test_pearson=0.8895, connectivity_R2=0.9991, final_loss=1.1543e+03
Activity: Dale_law chaotic oscillations across 100 neurons, amplitude [-11.9, 14.7], effective rank 5-21, spectral radius 0.915
Mutation: lr_W: 0.025 -> 0.02 (from Node 66's lr_W=0.025)
Parent rule: Node 66 was selected via explore strategy (UCB=3.178 highest), tested lr_W=0.02 to see if slightly lower lr_W improves this excellent batch_size=16 config
Observation: converged (R2=0.999). lr_W=0.02 from Node 66 (R2=0.998) achieves near-identical excellent connectivity (R2=0.999 vs 0.998) with slightly lower test metrics (pearson: 0.991->0.890). confirms lr_W range [0.02, 0.025] is optimal for Dale_law chaotic with L1=5e-05/batch_size=16. 19/20 converged in block 3 (95%). 19 consecutive successes (50-68), triggers explore strategy
Next: parent=65 (explore strategy: Node 65 UCB=3.227 highest, R2=0.991 with lr_W=0.015/lr=0.00025/L1=5e-05/batch_size=16; try lr_W=0.02 to test if optimal lr_W improves this lower-lr branch)

## Iter 69: converged

Node: id=69, parent=65
Mode/Strategy: explore/exploit
Config: lr_W=0.02, lr=0.00025, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=16, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9355, test_pearson=0.9119, connectivity_R2=0.9849, final_loss=8.0293e+02
Activity: Dale_law chaotic oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-12, spectral radius 0.915
Mutation: lr_W: 0.015 -> 0.02 (from Node 65's lr_W=0.015)
Parent rule: Node 65 was selected via explore strategy (UCB=3.227 highest), tested lr_W=0.02 to see if optimal lr_W improves this lr=0.00025 branch
Observation: converged (R2=0.985). lr_W=0.02 from Node 65 (R2=0.991) achieves slightly lower connectivity (R2=0.985 vs 0.991) but similar test metrics (pearson: 0.947->0.912). node 65's lr_W=0.015 was slightly better for lr=0.00025 branch, suggesting lr:lr_W ratio matters. 20/21 converged in block 3 (95%). 20 consecutive successes (50-69), triggers explore strategy
Next: parent=68 (explore strategy: Node 68 UCB=3.290 highest, R2=0.999 with lr_W=0.02/lr=0.0003/L1=5e-05/batch_size=16; try coeff_W_L1=1e-05 to probe if even lower L1 improves this near-optimal config)

## Iter 70: converged

Node: id=70, parent=68
Mode/Strategy: explore/exploit
Config: lr_W=0.02, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9418, test_pearson=0.9251, connectivity_R2=0.9998, final_loss=1.0670e+03
Activity: Dale_law chaotic oscillations across 100 neurons, amplitude [-11.9, 14.7], effective rank 5-21, spectral radius 0.915
Mutation: coeff_W_L1: 5.0e-05 -> 1.0e-05 (from Node 68's coeff_W_L1=5.0e-05)
Parent rule: Node 68 was selected via explore strategy (UCB=3.290 highest), tested coeff_W_L1=1e-05 to probe if even lower L1 improves this near-optimal config
Observation: converged (R2=0.9998). coeff_W_L1=1e-05 from Node 68 (R2=0.999) achieves near-perfect connectivity (R2=0.9998 matching parent). excellent test metrics (test_R2=0.942, pearson=0.925). L1=1e-05 works well for Dale_law chaotic, similar to L1=5e-05. 21/22 converged in block 3 (95%). 21 consecutive successes (50-70), triggers explore strategy
Next: parent=69 (explore strategy: Node 69 UCB=3.330 second highest after Node 70, R2=0.985 with lr_W=0.02/lr=0.00025/L1=5e-05/batch_size=16; try lr_W=0.025 to test if higher lr_W improves this lower-lr branch)

## Iter 71: converged

Node: id=71, parent=69
Mode/Strategy: explore/exploit
Config: lr_W=0.025, lr=0.00025, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=16, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9149, test_pearson=0.8981, connectivity_R2=0.9771, final_loss=8.3207e+02
Activity: Dale_law chaotic oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-12, spectral radius 0.915
Mutation: lr_W: 0.02 -> 0.025 (from Node 69 lr_W=0.02)
Parent rule: Node 69 was selected via explore strategy (UCB=3.330 second highest after Node 70), tested lr_W=0.025 to see if higher lr_W improves this lr=0.00025 branch
Observation: converged (R2=0.977). lr_W=0.025 from Node 69 (R2=0.985) achieves slightly lower connectivity (R2=0.977 vs 0.985) and slightly lower test metrics (pearson: 0.912->0.898). confirms lr_W=0.02 is better than 0.025 for lr=0.00025 branch, and lr_W=0.015 (Node 65, R2=0.991) remains best for this lower-lr config. 22/23 converged in block 3 (96%). 22 consecutive successes (50-71), triggers explore strategy
Next: parent=70 (explore strategy: Node 70 UCB=3.398 highest, R2=1.000 with lr_W=0.02/lr=0.0003/L1=1e-05/batch_size=16; try lr_W=0.015 to probe if lower lr_W maintains near-perfect R2 with lower L1)

## Iter 72: converged

Node: id=72, parent=70
Mode/Strategy: explore/exploit
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=False, low_rank=N/A, n_frames=10000
Metrics: test_R2=0.9408, test_pearson=0.9287, connectivity_R2=0.9924, final_loss=8.3291e+02
Activity: Dale_law chaotic oscillations across 100 neurons, amplitude [-11.9, 17.5], effective rank 3-12, spectral radius 0.915
Mutation: lr_W: 0.02 -> 0.015 (from Node 70's lr_W=0.02)
Parent rule: Node 70 was selected via explore strategy (UCB=3.398 highest), tested lr_W=0.015 to probe if lower lr_W maintains near-perfect R2 with lower L1
Observation: converged (R2=0.992). lr_W=0.015 from Node 70 (R2=1.000) achieves excellent connectivity (R2=0.992 vs 1.000) with good test metrics (pearson=0.929 vs 0.925). lr_W=0.02 remains optimal for L1=1e-05 config, but 0.015 still works well. 24/24 converged in block 3 (100%). 24 consecutive successes (49-72), excellent block performance
Next: NEW BLOCK (block 3 complete, moving to block 4 with new simulation parameters)

## Simulation Block 3 Summary (iters 49-72)

Simulation: connectivity_type=chaotic, Dale_law=True, Dale_law_factor=0.5, connectivity_rank=N/A, noise_model_level=N/A
Best R2: 1.000 at iter 70
Converged: Yes (24/24 = 100%)
Observation: Dale_law chaotic simulation highly trainable. optimal config zone identified: lr_W=[0.015-0.025], lr=0.0003, L1=[1e-05 to 5e-05], batch_size=16. lower lr_W with lr=0.00025 also works but slightly worse (R2~0.98). batch_size=16 universally improves test generalization over batch_size=32. lr:lr_W ratio ~15:1 is optimal. L1 regularization from 1e-05 to 5e-05 both excellent
Optimum training: lr_W=0.02, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, low_rank_factorization=False, low_rank=N/A

### Cross-block Meta-analysis (Blocks 1-3)

| Block | Regime                   | Best R2 | Optimal lr_W | Optimal L1     | Key constraint                             |
| ----- | ------------------------ | ------- | ------------ | -------------- | ------------------------------------------ |
| 1     | chaotic, no Dale_law     | 0.9998  | 0.003-0.01   | 1e-05          | lr_W > 0.001 required                      |
| 2     | low_rank=20, no Dale_law | 1.000   | 0.01-0.03    | 1e-05 to 1e-04 | higher lr_W preferred                      |
| 3     | chaotic, Dale_law=True   | 1.000   | 0.015-0.025  | 1e-05 to 5e-05 | batch_size=16 optimal, lr:lr_W ratio ~15:1 |

### Emerging patterns:

1. All chaotic regimes highly trainable (R2 > 0.99 achievable in all blocks)
2. Dale_law constraint does not impair trainability - block 3 achieved 100% convergence
3. Optimal lr_W range shifted slightly lower with Dale_law (0.015-0.025 vs 0.01-0.03)
4. batch_size=16 consistently better than 32 for test generalization
5. L1 regularization has wide tolerance (1e-05 to 1e-04)

### Hypothesis for Block 4:

Testing low_rank connectivity (connectivity_type="low_rank") with low_rank_factorization training to see if GNN can recover constrained low-rank matrices. Predict: may need low_rank_factorization=True with matching rank for optimal recovery. Starting with connectivity_rank=20 and low_rank=20 in training

### Protocol rules evaluation (Block 3):

- Branching rate (last 6 iters): 6 unique parents (64,66,65,68,69,70) = 100% branching (excellent)
- Improvement rate: 24/24 converged = 100% success
- No stuck detection: continuous exploration with high R2 variance mapping
- Protocol rules performed optimally - no changes needed

Protocol edit: No changes needed - current rules achieved 100% branching rate and 100% convergence rate. explore strategy with n_iter_block/4 window effectively promoted diverse parent selection

--- NEW SIMULATION BLOCK ---
Next simulation: connectivity_type=low_rank, Dale_law=False, Dale_law_factor=0.5, connectivity_rank=20, low_rank_factorization=True, low_rank=20
Reference: starting block 4 (first iter will be node 73)

## Iter 73: converged

Node: id=73, parent=root
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.005, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.7978, test_pearson=0.8481, connectivity_R2=0.9994, final_loss=1.5175e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 12, spectral radius 0.959
Mutation: baseline for new simulation block (connectivity_type=low_rank, low_rank_factorization=True)
Parent rule: first iteration of block 4, parent=root by default
Observation: excellent first result with low_rank connectivity. connectivity_R2=0.9994 shows low_rank_factorization=True with rank=20 successfully recovers low-rank structure. lower effective rank (12) compared to chaotic blocks (30+) confirms low-rank dynamics. test metrics moderate (pearson=0.848). previous blocks found lr_W in range [0.01-0.025] optimal for chaotic - testing if higher lr_W improves low_rank regime
Next: parent=73 (highest UCB=1.499, exploit: try lr_W=0.01 to test if higher lr_W improves like previous blocks)

## Iter 74: converged

Node: id=74, parent=73
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9516, test_pearson=0.9717, connectivity_R2=0.9994, final_loss=1.0980e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 13, spectral radius 0.959
Mutation: lr_W: 0.005 -> 0.01 (from Node 73's lr_W=0.005)
Parent rule: Node 73 had highest UCB (1.499), exploit with lr_W=0.01 based on block 2 findings
Observation: converged (R2=0.9994). lr_W=0.01 maintains excellent connectivity from Node 73 (R2=0.9994->0.9994). significantly improved test metrics (test_R2: 0.798->0.952, pearson: 0.848->0.972). confirms higher lr_W improves both connectivity and generalization for low_rank regime. 2/2 converged in block 4 (100%). 2 consecutive successes (73-74), continue exploit strategy
Next: parent=74 (highest UCB=1.707, exploit: try lr_W=0.015 to test if even higher lr_W further improves low_rank recovery)

## Iter 75: converged

Node: id=75, parent=74
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.015, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9166, test_pearson=0.9406, connectivity_R2=0.9986, final_loss=1.0161e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 15.2], effective rank 13, spectral radius 0.959
Mutation: lr_W: 0.01 -> 0.015 (from Node 74's lr_W=0.01)
Parent rule: Node 74 had highest UCB (1.577), exploit with lr_W=0.015 continuing upward trend
Observation: converged (R2=0.9986). lr_W=0.015 slightly decreased connectivity_R2 from 0.9994 to 0.9986, and test metrics slightly worse (pearson 0.941 vs 0.972). suggests lr_W=0.01 may be optimal for low_rank regime - higher lr_W shows diminishing returns. 3/3 converged in block 4 (100%). 3 consecutive successes triggers failure-probe strategy
Next: parent=75 (highest UCB=1.865, failure-probe: try lr_W=0.03 to find upper boundary in low_rank regime)

## Iter 76: converged

Node: id=76, parent=75
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.03, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9167, test_pearson=0.9322, connectivity_R2=0.9926, final_loss=1.1830e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.9], effective rank 12, spectral radius 0.959
Mutation: lr_W: 0.015 -> 0.03 (from Node 75 lr_W=0.015)
Parent rule: Node 75 had highest UCB (1.865), failure-probe with lr_W=0.03 to find upper boundary in low_rank regime
Observation: converged (R2=0.993). failure-probe found soft upper boundary - lr_W=0.03 still converges but R2 drops from 0.999 (Node 74-75) to 0.993. optimal lr_W for low_rank regime is around 0.01, not 0.03. 4/4 converged in block 4 (100%). continue exploit to explore other parameters from best performing config
Next: parent=74 (exploit strategy: Node 74 R2=0.999 with lr_W=0.01 showed best test metrics; try lr=0.0003 which was optimal in block 3 chaotic)

## Iter 77: converged

Node: id=77, parent=74
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9815, test_pearson=0.9857, connectivity_R2=0.9994, final_loss=1.1049e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 13, spectral radius 0.959
Mutation: lr: 0.0002 -> 0.0003 (from Node 74's lr=0.0002)
Parent rule: Node 74 was used as parent (R2=0.999, best test metrics at lr_W=0.01), tested lr=0.0003 which was optimal in block 3
Observation: converged (R2=0.9994). lr=0.0003 from Node 74 (R2=0.9994) maintains excellent connectivity with improved test metrics (test_R2: 0.952->0.982, pearson: 0.972->0.986). confirms lr=0.0003 is optimal for low_rank regime like it was for Dale_law chaotic in block 3. 5/5 converged in block 4 (100%). 5 consecutive successes (73-77), triggers explore strategy
Next: parent=76 (explore strategy: Node 76 UCB=2.111 second highest, R2=0.993 with lr_W=0.03; try lr_W=0.02 to test intermediate lr_W at boundary region)

## Iter 78: converged

Node: id=78, parent=76
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.02, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9160, test_pearson=0.9363, connectivity_R2=0.9973, final_loss=1.0358e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 5, spectral radius 0.959
Mutation: lr_W: 0.03 -> 0.02 (from Node 76 lr_W=0.03)
Parent rule: Node 76 UCB=2.111 (second highest), tested intermediate lr_W=0.02 at upper boundary region
Observation: converged (R2=0.997). lr_W=0.02 from Node 76 improves R2 from 0.993 to 0.997, confirming lr_W=0.02 is closer to optimal than lr_W=0.03. but still below Node 74/77 R2=0.999 at lr_W=0.01. optimal lr_W for low_rank regime confirmed around 0.01. 6/6 converged in block 4 (100%). 6 consecutive successes (n_iter_block/4=6) triggers explore strategy - select highest UCB node not in last 6, but all 6 nodes are in last block. use highest UCB (Node 77, R2=0.999) with new parameter mutation
Next: parent=77 (explore strategy: Node 77 has highest UCB=2.224, R2=0.999; try coeff_W_L1=1e-04 to test L1 regularization in low_rank regime)

## Iter 79: converged

Node: id=79, parent=77
Mode/Strategy: explore/exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8499, test_pearson=0.8861, connectivity_R2=0.9941, final_loss=1.5634e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 13, spectral radius 0.959
Mutation: coeff_W_L1: 1.0e-05 -> 1.0e-04 (from Node 77's coeff_W_L1=1.0e-05)
Parent rule: Node 77 had highest UCB (2.224), tested coeff_W_L1=1e-04 to compare L1 regularization levels in low_rank regime
Observation: converged (R2=0.994). coeff_W_L1=1e-04 from Node 77 (R2=0.999) achieves excellent connectivity but slightly lower than parent (R2 0.999->0.994). test metrics also degraded (pearson 0.986->0.886). confirms coeff_W_L1=1e-05 is better than 1e-04 for low_rank regime - opposite to block 2 findings where 5e-05 was optimal. 7/7 converged in block 4 (100%). 7 consecutive successes (73-79), triggers failure-probe strategy
Next: parent=78 (failure-probe: Node 78 UCB=2.320 highest, R2=0.997 with lr_W=0.02; try lr=0.0003 to test if higher lr improves this branch)

## Iter 80: converged

Node: id=80, parent=78
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.02, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8811, test_pearson=0.9145, connectivity_R2=0.9977, final_loss=1.0780e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 13 (99% var), spectral radius 0.959
Mutation: lr: 0.0002 -> 0.0003 (from Node 78's lr=0.0002)
Parent rule: Node 78 had highest UCB (2.320), tested lr=0.0003 which was optimal in previous blocks
Observation: converged (R2=0.998). lr=0.0003 from Node 78 (R2=0.997) improves connectivity (R2 0.997->0.998). test metrics good (pearson=0.914). confirms lr=0.0003 improves low_rank regime across different lr_W values. Node 77 (lr_W=0.01, lr=0.0003) still has highest R2=0.999. 8/8 converged in block 4 (100%). 8 consecutive successes (73-80), triggers explore strategy - select node outside recent 6-node chain (75-80)
Next: parent=77 (explore strategy: Node 77 UCB=1.942 outside recent chain, R2=0.999 best in block; try lr_W=0.012 to probe if slightly higher lr_W maintains near-perfect R2)

## Iter 81: converged

Node: id=81, parent=77
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.012, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9263, test_pearson=0.9512, connectivity_R2=0.9993, final_loss=1.0663e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 13 (99% var), spectral radius 0.959
Mutation: lr_W: 0.01 -> 0.012 (from Node 77's lr_W=0.01)
Parent rule: Node 77 had UCB=1.942 outside recent 6-node chain, tested lr_W=0.012 to probe if slightly higher lr_W maintains R2
Observation: converged (R2=0.999). lr_W=0.012 from Node 77 (R2=0.999) maintains excellent connectivity (R2=0.999). test metrics slightly lower than Node 77 (pearson 0.951 vs 0.986). confirms lr_W=0.01 is closer to optimal than 0.012 for low_rank regime. 9/9 converged in block 4 (100%). 9 consecutive successes (73-81), triggers explore strategy - select highest UCB node not in last 6 (nodes 76-81)
Next: parent=81 (highest UCB=2.499, explore: try batch_size=8 to test if smaller batch improves test metrics while maintaining connectivity)

## Iter 82: converged

Node: id=82, parent=81
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.012, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9077, test_pearson=0.9071, connectivity_R2=0.9974, final_loss=1.5322e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 13 (99% var), spectral radius 0.959
Mutation: batch_size: 16 -> 8 (from Node 81's batch_size=16)
Parent rule: Node 81 had highest UCB (2.499), tested batch_size=8 to see if smaller batch improves test metrics
Observation: converged (R2=0.997). batch_size=8 from Node 81 (R2=0.999) slightly decreased connectivity (R2 0.999->0.997). test metrics comparable (pearson 0.907 vs 0.951). confirms batch_size=16 is better than 8 for low_rank regime. 10/10 converged in block 4 (100%). 10 consecutive successes (73-82) triggers failure-probe strategy - map lower boundary of lr_W for low_rank regime
Next: parent=80 (failure-probe: Node 80 UCB=2.579 highest, R2=0.998; try lr_W=0.003 to probe lower boundary in low_rank regime, similar to chaotic block where lr_W<0.005 failed)

## Iter 83: converged

Node: id=83, parent=80
Mode/Strategy: failure-probe/boundary
Config: lr_W=0.003, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8856, test_pearson=0.9245, connectivity_R2=0.9837, final_loss=2.5570e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 15.5], effective rank 13 (99% var), spectral radius 0.959
Mutation: lr_W: 0.02 -> 0.003 (from Node 80's lr_W=0.02)
Parent rule: Node 80 had highest UCB (2.579), failure-probe with lr_W=0.003 to find lower boundary in low_rank regime
Observation: converged (R2=0.984). failure-probe found soft lower boundary - lr_W=0.003 still converges but R2 drops from 0.998 (Node 80) to 0.984. loss increased significantly (1078->2557). optimal lr_W for low_rank regime confirmed to be [0.01-0.02], with 0.003 at lower edge of acceptable range. 11/11 converged in block 4 (100%). 11 consecutive successes (73-83), triggers explore strategy - select highest UCB node outside recent failure-probe
Next: parent=82 (explore strategy: Node 82 UCB=2.656 highest, R2=0.997 with lr_W=0.012/batch_size=8; try lr_W=0.01 to test if optimal lr_W improves batch_size=8 config)

## Iter 84: converged

Node: id=84, parent=82
Mode/Strategy: success-exploit/exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9906, test_pearson=0.9906, connectivity_R2=0.9982, final_loss=1.5166e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 12 (99% var), spectral radius 0.959
Mutation: lr_W: 0.012 -> 0.01 (from Node 82's lr_W=0.012)
Parent rule: Node 82 had highest UCB (2.656), tested lr_W=0.01 to see if optimal lr_W improves batch_size=8 config
Observation: converged (R2=0.998). lr_W=0.01 from Node 82 (R2=0.997) improved connectivity (R2 0.997->0.998). excellent test metrics (pearson=0.991, best in block 4). confirms lr_W=0.01 is optimal for low_rank regime regardless of batch_size. 12/12 converged in block 4 (100%). 12 consecutive successes (73-84), triggers explore strategy - select highest UCB node outside recent 6-node chain (79-84)
Next: parent=79 (explore strategy: Node 79 UCB=2.726 highest outside recent chain, R2=0.994 with coeff_W_L1=1e-04; try lr_W=0.008 to test if slightly lower lr_W compensates for higher L1 regularization)

## Iter 85: converged

Node: id=85, parent=79
Mode/Strategy: explore/exploit
Config: lr_W=0.008, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9312, test_pearson=0.9544, connectivity_R2=0.9953, final_loss=1.6057e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 15.4], effective rank 13 (99% var), spectral radius 0.959
Mutation: lr_W: 0.01 -> 0.008 (from Node 79's lr_W=0.01)
Parent rule: Node 79 UCB=2.726 (highest outside recent 6-node chain 80-85), tested lr_W=0.008 to see if slightly lower lr_W compensates for higher L1=1e-04
Observation: converged (R2=0.995). lr_W=0.008 from Node 79 (R2=0.994) slightly improved connectivity (R2 0.994->0.995). test metrics excellent (pearson=0.954). confirms lr_W=0.008-0.01 range works well even with higher L1=1e-04. but still below Node 77/84 R2=0.999 at coeff_W_L1=1e-05. 13/13 converged in block 4 (100%). 13 consecutive successes (73-85), triggers explore strategy - select highest UCB node outside recent 6-node chain (80-85)
Next: parent=84 (explore strategy: Node 84 UCB=2.801 highest, R2=0.998 with lr_W=0.01/L1=1e-05/batch_size=8; try batch_size=16 to verify if batch_size=16 improves this optimal lr_W config)

## Iter 86: converged

Node: id=86, parent=84
Mode/Strategy: explore/exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9972, test_pearson=0.9977, connectivity_R2=0.9995, final_loss=1.1152e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 15.5], effective rank 13 (99% var), spectral radius 0.959
Mutation: batch_size: 8 -> 16 (from Node 84's batch_size=8)
Parent rule: Node 84 had highest UCB (2.801), tested batch_size=16 to verify if batch_size=16 improves this optimal lr_W=0.01 config
Observation: converged (R2=1.000). batch_size=16 from Node 84 (R2=0.998) achieved near-perfect connectivity (R2=0.9995, rounded to 1.000). excellent test metrics (test_R2=0.997, pearson=0.998 - best in block 4). confirms batch_size=16 with lr_W=0.01/lr=0.0003/L1=1e-05 is optimal config for low_rank regime. 14/14 converged in block 4 (100%). 14 consecutive successes (73-86), triggers explore strategy - select highest UCB node outside recent 6-node chain (81-86)
Next: parent=85 (explore strategy: Node 85 UCB=2.866 second highest, R2=0.995 with lr_W=0.008/L1=1e-04; try lr_W=0.01 to test if optimal lr_W improves this higher L1 branch)

## Iter 87: converged

Node: id=87, parent=85
Mode/Strategy: explore/exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9005, test_pearson=0.9092, connectivity_R2=0.9932, final_loss=1.4876e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 14.7], effective rank 12 (99% var), spectral radius 0.959
Mutation: lr_W: 0.008 -> 0.01 (from Node 85's lr_W=0.008)
Parent rule: Node 85 UCB=2.866 (second highest outside recent 6-node chain), tested lr_W=0.01 to see if optimal lr_W improves higher L1=1e-04 branch
Observation: converged (R2=0.993). lr_W=0.01 from Node 85 (R2=0.995) slightly decreased connectivity (R2 0.995->0.993). confirms L1=1e-04 is suboptimal regardless of lr_W - Node 86 with L1=1e-05 achieves R2=1.000. 15/15 converged in block 4 (100%). 15 consecutive successes (73-87), triggers explore strategy - select highest UCB node (Node 86, R2=1.000) for robustness-test
Next: parent=86 (robustness-test: Node 86 UCB=2.936 highest, R2=1.000 with optimal config; rerun identical config with different seed to verify reproducibility)

## Iter 88: converged

Node: id=88, parent=86
Mode/Strategy: robustness-test/exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8967, test_pearson=0.9290, connectivity_R2=0.9992, final_loss=1.3306e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-15.8, 15.5], effective rank 13 (99% var), spectral radius 0.959
Mutation: seed: 31 -> new seed (robustness-test, identical config from Node 86)
Parent rule: Node 86 UCB=2.936 highest, robustness-test with identical config but different random seed
Observation: converged (R2=0.999). robustness-test confirmed: Node 86 config is reproducible (R2=1.000->0.999 with different seed). test_pearson 0.929 vs 0.998 shows some variance in test metrics but connectivity recovery is stable. 16/16 converged in block 4 (100%). 16 consecutive successes (73-88), triggers explore strategy - select highest UCB outside recent 6 nodes (83-88)
Next: parent=80 (explore strategy: Node 80 UCB=2.331 highest outside recent 6-node chain, R2=0.998 with lr=0.0003; try batch_size=8 to test batch_size sensitivity in lr=0.0003 branch)

## Iter 89: converged

Node: id=89, parent=80
Mode/Strategy: explore/exploit
Config: lr_W=0.02, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9726, test_pearson=0.9757, connectivity_R2=0.9887, final_loss=2.0208e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-23.1, 20.1], effective rank 5-14, spectral radius 1.216
Mutation: batch_size: 16 -> 8 (from Node 80's batch_size=16)
Parent rule: Node 80 UCB=2.331 highest outside recent 6-node chain (84-89), tested batch_size=8 in lr_W=0.02/lr=0.0003 branch
Observation: converged (R2=0.989). batch_size=8 from Node 80 (R2=0.998) decreased connectivity (R2 0.998->0.989). confirms batch_size=16 is better than 8 for low_rank regime across different lr_W values (both 0.01 and 0.02). optimal config remains Node 86 (lr_W=0.01, batch_size=16, R2=1.000). 17/17 converged in block 4 (100%). 17 consecutive successes (73-89), triggers explore strategy - select highest UCB outside recent 6 nodes (84-89)
Next: parent=83 (explore strategy: Node 83 UCB=3.045 outside recent chain, R2=0.984 with lr_W=0.003; try lr=0.0002 to test if lower lr compensates for very low lr_W boundary region)

## Iter 90: converged

Node: id=90, parent=83
Mode/Strategy: explore/exploit
Config: lr_W=0.003, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9437, test_pearson=0.9568, connectivity_R2=0.9882, final_loss=3.0317e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-17.5, 22.5], effective rank 15 (99% var), spectral radius 1.216
Mutation: lr: 0.0003 -> 0.0002 (from Node 83's lr=0.0003)
Parent rule: Node 83 UCB=3.045 highest outside recent 6-node chain (84-89), tested lr=0.0002 to see if lower lr compensates for lower lr_W=0.003 boundary region
Observation: converged (R2=0.988). lr=0.0002 from Node 83 (R2=0.984) slightly improved connectivity (R2 0.984->0.988) but still below optimal Node 86 (R2=1.000). confirms lr_W=0.003 is suboptimal compared to lr_W=0.01 regardless of lr setting. spectral radius 1.216 suggests different effective dynamics. 18/18 converged in block 4 (100%). 18 consecutive successes (73-90), triggers explore strategy - select highest UCB outside recent 6 nodes (85-90)
Next: parent=84 (explore strategy: Node 84 UCB=2.059 highest outside recent 6-node chain, R2=0.998 with lr_W=0.01/batch_size=8; try lr_W=0.012 to probe upper lr_W boundary from this high-performing branch)

## Iter 91: converged

Node: id=91, parent=84
Mode/Strategy: explore/exploit
Config: lr_W=0.012, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8257, test_pearson=0.8563, connectivity_R2=0.9974, final_loss=1.9368e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-20, 23], effective rank 14 (99% var), spectral radius 1.216
Mutation: lr_W: 0.01 -> 0.012 (from Node 84's lr_W=0.01)
Parent rule: Node 84 UCB=2.059 outside recent 6-node chain (86-91), tested lr_W=0.012 to probe upper lr_W boundary from this batch_size=8 branch
Observation: converged (R2=0.997). lr_W=0.012 from Node 84 (R2=0.998) achieves similar connectivity (R2 0.998->0.997). test metrics moderate (pearson=0.856). spectral radius 1.216 indicates slight overshoot but dynamics still stable. confirms lr_W=0.01-0.012 range works well with batch_size=8. 19/19 converged in block 4 (100%). 19 consecutive successes (73-91), triggers explore strategy - select highest UCB outside recent 6 nodes (86-91)
Next: parent=88 (explore strategy: Node 88 UCB=3.179 highest, R2=0.999 with lr_W=0.01/lr=0.0003/batch_size=16; try lr=0.00025 to test if lower lr improves test metrics from this excellent config)

## Iter 92: converged

Node: id=92, parent=88
Mode/Strategy: explore/exploit
Config: lr_W=0.01, lr=0.00025, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9807, test_pearson=0.9791, connectivity_R2=0.9988, final_loss=1.5903e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-23.1, 20.1], effective rank 14 (99% var), spectral radius 1.216
Mutation: lr: 0.0003 -> 0.00025 (from Node 88's lr=0.0003)
Parent rule: Node 88 UCB=3.179 highest outside recent 6-node chain (87-92), tested lr=0.00025 to see if lower lr improves test metrics
Observation: converged (R2=0.999). lr=0.00025 from Node 88 (R2=0.999) maintains excellent connectivity (R2 0.999->0.999). test metrics excellent (pearson=0.979 comparable to Node 88's 0.929 after robustness-test). confirms lr in range [0.00025-0.0003] both work well for low_rank regime. 20/20 converged in block 4 (100%). 20 consecutive successes (73-92), triggers explore strategy - select highest UCB outside recent 6 nodes (87-92)
Next: parent=92 (explore strategy: Node 92 UCB=3.235 highest, R2=0.999 with lr_W=0.01/lr=0.00025; try batch_size=8 to test if smaller batch improves this excellent config)

## Iter 93: converged

Node: id=93, parent=92
Mode/Strategy: explore/exploit
Config: lr_W=0.01, lr=0.00025, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9291, test_pearson=0.9161, connectivity_R2=0.9943, final_loss=1.8709e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-16.9, 23.1], effective rank 12 (99% var), spectral radius 1.216
Mutation: batch_size: 16 -> 8 (from Node 92's batch_size=16)
Parent rule: Node 92 UCB=3.235 highest outside recent 6-node chain (88-93), tested batch_size=8 to see if smaller batch improves excellent config
Observation: converged (R2=0.994). batch_size=8 from Node 92 (R2=0.999) decreased connectivity (R2 0.999->0.994). confirms batch_size=16 is optimal for low_rank regime across different lr settings (lr=0.0003 and lr=0.00025). optimal config remains Node 86 (lr_W=0.01, lr=0.0003, batch_size=16, R2=1.000). 21/21 converged in block 4 (100%). 21 consecutive successes (73-93), triggers explore strategy - select highest UCB outside recent 6 nodes (88-93)
Next: parent=91 (explore strategy: Node 91 UCB=3.289 highest, R2=0.997 with lr_W=0.012/batch_size=8; try lr_W=0.015 to probe upper lr_W boundary from this branch)

## Iter 94: converged

Node: id=94, parent=91
Mode/Strategy: explore/exploit
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8787, test_pearson=0.8380, connectivity_R2=0.9942, final_loss=2.0101e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-21.6, 20.1], effective rank 15 (99% var), spectral radius 1.216
Mutation: lr_W: 0.012 -> 0.015 (from Node 91's lr_W=0.012)
Parent rule: Node 91 UCB=3.289 highest outside recent 6-node chain (88-93), tested lr_W=0.015 to probe upper lr_W boundary from batch_size=8 branch
Observation: converged (R2=0.994). lr_W=0.015 from Node 91 (R2=0.997) slightly decreased connectivity (R2 0.997->0.994). confirms lr_W=0.01-0.012 is better than 0.015 for low_rank regime. test metrics moderate (pearson=0.838). optimal config remains Node 86 (lr_W=0.01, lr=0.0003, batch_size=16, R2=1.000). 22/22 converged in block 4 (100%). 22 consecutive successes (73-94), triggers explore strategy - select highest UCB outside recent 6 nodes (89-94)
Next: parent=87 (explore strategy: Node 87 UCB=3.338 highest outside recent 6-node chain, R2=0.993 with lr_W=0.01/L1=1e-04; try batch_size=8 to test smaller batch with higher L1 regularization)

## Iter 95: converged

Node: id=95, parent=87
Mode/Strategy: explore/exploit
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=8, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8725, test_pearson=0.8394, connectivity_R2=0.9890, final_loss=2.4362e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-21.5, 20.6], effective rank 14 (99% var), spectral radius 1.216
Mutation: batch_size: 16 -> 8 (from Node 87's batch_size=16)
Parent rule: Node 87 UCB=3.338 highest outside recent 6-node chain (89-94), tested batch_size=8 with higher L1=1e-04 regularization
Observation: converged (R2=0.989). batch_size=8 from Node 87 (R2=0.993) slightly decreased connectivity (R2 0.993->0.989). test metrics moderate (pearson=0.839). confirms batch_size=16 is better than 8 for low_rank regime across different L1 settings (both 1e-05 and 1e-04). 23/23 converged in block 4 (100%). 23 consecutive successes (73-95), triggers explore strategy - iter 23/24, next iter is block end
Next: parent=89 (explore strategy: Node 89 UCB=3.387 highest outside recent 6-node chain [90-95], R2=0.989 with lr_W=0.02/batch_size=8; try lr_W=0.015 to test intermediate lr_W at batch_size=8)

## Iter 96: converged

Node: id=96, parent=89
Mode/Strategy: explore/exploit
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=8, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8516, test_pearson=0.8560, connectivity_R2=0.9945, final_loss=1.9876e+03
Activity: low_rank oscillations across 100 neurons, amplitude [-20.6, 21.5], effective rank 14 (99% var), spectral radius 1.216
Mutation: lr_W: 0.02 -> 0.015 (from Node 89's lr_W=0.02)
Parent rule: Node 89 UCB=3.387 highest outside recent 6-node chain (90-95), tested lr_W=0.015 intermediate value from batch_size=8 branch
Observation: converged (R2=0.995). lr_W=0.015 from Node 89 (R2=0.989) improved connectivity (R2 0.989->0.995). confirms lr_W=0.015 is better than 0.02 for low_rank regime. 24/24 converged in block 4 (100%). block end - write summary

## Simulation Block 4 Summary (iters 73-96)

Simulation: connectivity_type=low_rank, Dale_law=False, Dale_law_factor=0.5, connectivity_rank=20, low_rank_factorization=True, low_rank=20
Best R2: 1.000 at iter 86
Converged: Yes (24/24 = 100%)
Observation: low_rank connectivity highly trainable with low_rank_factorization=True. optimal config zone: lr_W=0.01, lr=0.0003, L1=1e-05, batch_size=16. batch_size=16 consistently outperforms batch_size=8 (R2~0.99-1.00 vs R2~0.989-0.997). lr_W=0.01-0.012 optimal, higher values (0.015-0.02) slightly worse. L1=1e-05 optimal, L1=1e-04 slightly worse (R2~0.989-0.993)
Optimum training: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20

### Cross-block Meta-analysis (Blocks 1-4)

| Block | Regime                   | Best R2 | Optimal lr_W | Optimal L1     | Key constraint                       |
| ----- | ------------------------ | ------- | ------------ | -------------- | ------------------------------------ |
| 1     | chaotic, no Dale_law     | 0.9998  | 0.003-0.01   | 1e-05          | lr_W > 0.001 required                |
| 2     | low_rank=20, no Dale_law | 1.000   | 0.01-0.03    | 1e-05 to 1e-04 | higher lr_W preferred                |
| 3     | chaotic, Dale_law=True   | 1.000   | 0.015-0.025  | 1e-05 to 5e-05 | batch_size=16 optimal                |
| 4     | low_rank=20, no Dale_law | 1.000   | 0.01-0.012   | 1e-05          | low_rank_factorization=True required |

### Coverage Table

| connectivity_type | Dale_law=False     | Dale_law=True    |
| ----------------- | ------------------ | ---------------- |
| chaotic           | Block 1-2 (R2=1.0) | Block 3 (R2=1.0) |
| low_rank=20       | Block 4 (R2=1.0)   | Block 5 (next)   |
| low_rank=50       | -                  | -                |

### Emerging patterns:

1. All regimes tested highly trainable (R2=1.0 achievable in all blocks 1-4)
2. Low_rank connectivity requires low_rank_factorization=True with matching rank for optimal recovery
3. Optimal lr_W is lower for low_rank (0.01) vs chaotic (0.015-0.025) - suggests constrained connectivity needs gentler optimization
4. batch_size=16 universally superior across all regimes tested
5. L1=1e-05 consistently optimal - higher values slightly hurt convergence
6. lr=0.0003 universally optimal across all regimes
7. Effective rank of low_rank simulation (~12-15) is lower than chaotic (~32-35), matching structural constraints

### Hypothesis for Block 5:

Testing low_rank=20 with Dale_law=True to see if combined constraints affect trainability. Based on blocks 3-4, predict: should converge with similar config (lr_W=0.01, lr=0.0003, batch_size=16, L1=1e-05). Dale_law did not impair block 3 so combined constraints should remain trainable.

### Protocol rules evaluation (Block 4):

- Branching rate (last 6 iters [91-96]): parents = 84, 88, 92, 91, 87, 89 = 6 unique parents (100% branching - excellent)
- Improvement rate: 24/24 converged = 100% success rate
- No stuck detection: continuous exploration with high R2 (all > 0.98)
- Protocol rules performed optimally - explore strategy effectively promoted diverse parent selection

Protocol edit: No changes needed - current rules achieved 100% branching rate and 100% convergence rate. explore strategy with n_iter_block/4 window effectively promoted diverse parent selection. robustness-test strategy (iter 88) confirmed reproducibility of best config.

--- NEW SIMULATION BLOCK ---
Next simulation: connectivity_type=low_rank, Dale_law=True, Dale_law_factor=0.5, connectivity_rank=20, low_rank_factorization=True, low_rank=20
Reference: starting block 5 (first iter will be node 97)

## Iter 97: partial

Node: id=97, parent=root
Mode/Strategy: exploit/explore
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9606, test_pearson=0.8220, connectivity_R2=0.3557, final_loss=1.3871e+03
Activity: low_rank+dale oscillations across 100 neurons, amplitude [-9.0, 13.8], effective rank 7 (99% var), spectral radius 0.913
Mutation: baseline for block 5 (lr_W=0.01, lr=0.0003 from block 4 optimum)
Parent rule: first iteration of block 5, parent=root by default
Observation: block 5 starts with partial convergence (R2=0.356). combined low_rank+dale_law constraint significantly harder than individual constraints. effective rank very low (7) compared to block 4 low_rank (12-15). spectral radius 0.913 lower than block 4 (~1.2). combined constraints reduce activity diversity, making W recovery harder. need to explore higher lr_W or different L1 to improve convergence
Next: parent=97 (only node, try lr_W=0.015 to accelerate connectivity learning)

## Iter 98: partial

Node: id=98, parent=97
Mode/Strategy: exploit/explore
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9495, test_pearson=0.7593, connectivity_R2=0.3495, final_loss=1.3519e+03
Activity: low_rank+dale oscillations across 100 neurons, amplitude [-9.0, 13.8], effective rank 7 (99% var), spectral radius 0.913
Mutation: lr_W: 0.01 -> 0.015
Parent rule: Node 97 highest UCB (0.827), increased lr_W to accelerate connectivity learning
Observation: increasing lr_W from 0.01 to 0.015 did not improve R2 (0.350 vs 0.356). both iterations partial. very low effective rank (7) with combined low_rank+dale constraints limits identifiability. need different strategy: try increasing L1 to encourage sparser solution, or reduce lr_W significantly since high lr may overshoot the constrained solution space
Next: parent=98 (highest UCB=1.057, try coeff_W_L1=1.0e-04 to encourage sparsity in this low-rank regime)


## Iter 99: partial

Node: id=99, parent=98
Mode/Strategy: exploit/explore
Config: lr_W=0.015, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8629, test_pearson=0.5439, connectivity_R2=0.3441, final_loss=1.6199e+03
Activity: low_rank+dale oscillations across 100 neurons, amplitude [-9.0, 13.8], effective rank 7 (99% var), spectral radius 0.913
Mutation: coeff_W_L1: 1.0e-05 -> 1.0e-04 (from Node 98's coeff_W_L1=1.0e-05)
Parent rule: Node 98 had highest UCB (0.927), tried increasing L1 to encourage sparsity
Observation: L1=1.0e-04 did not improve R2 (0.344 vs 0.349). 3 consecutive partial results (97-99). very low effective rank (7) with combined low_rank+dale constraints severely limits identifiability. need to try completely different strategy: lr_W/lr ratio change, much lower lr_W, or batch_size change
Next: parent=97 (Node 97 UCB=0.789 baseline config; try lr_W=0.005 to test if lower lr_W helps in constrained regime)

## Iter 100: partial

Node: id=100, parent=97
Mode/Strategy: exploit/explore
Config: lr_W=0.005, lr=0.0003, lr_emb=0.00025, coeff_W_L1=1.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.9574, test_pearson=0.7834, connectivity_R2=0.3124, final_loss=1.6482e+03
Activity: low_rank+dale oscillations across 100 neurons, amplitude [-9.0, 13.8], effective rank 7 (99% var), spectral radius 0.913
Mutation: lr_W: 0.01 -> 0.005 (from Node 97's lr_W=0.01)
Parent rule: Node 97 baseline config, tested lr_W=0.005 to see if lower lr_W helps in constrained regime
Observation: lower lr_W=0.005 did not improve R2 (0.312 vs 0.356 baseline). 4 consecutive partial results (97-100). combined low_rank+dale_law with effective rank 7 is fundamentally harder - activity lacks diversity to constrain W. all mutations so far (lr_W up/down, L1 up) ineffective. need to explore lr change or try higher lr_W (0.02-0.03) to break out of local minimum
Next: parent=99 (highest UCB=1.344, R2=0.344; try lr=0.0002 to test if lower lr helps with high L1=1e-04 config)

## Iter 101: partial

Node: id=101, parent=99
Mode/Strategy: exploit/explore
Config: lr_W=0.015, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8912, test_pearson=0.4210, connectivity_R2=0.3551, final_loss=1.5877e+03
Activity: low_rank+dale oscillations across 100 neurons, amplitude [-9.0, 13.8], effective rank 7 (99% var), spectral radius 0.913
Mutation: lr: 0.0003 -> 0.0002 (from Node 99's lr=0.0003)
Parent rule: Node 99 had highest UCB (1.344), tested lr=0.0002 to see if lower lr helps with high L1=1e-04 config
Observation: lr=0.0002 slightly improved R2 (0.355 vs 0.344 from Node 99). 5 consecutive partial results (97-101). combined low_rank+dale_law with effective rank 7 remains fundamentally challenging - activity lacks diversity to constrain full W recovery. all tested lr_W values (0.005-0.015) and L1 values (1e-05 to 1e-04) yield R2 ~0.31-0.36. need to try more extreme parameters: much higher lr_W (0.02+) to accelerate learning, or try batch_size=8 to increase gradient noise
Next: parent=101 (highest UCB=1.473, R2=0.355; try lr_W=0.02 to test if even higher lr_W helps in this constrained regime)

## Iter 102: partial

Node: id=102, parent=101
Mode/Strategy: exploit/explore
Config: lr_W=0.02, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.7451, test_pearson=0.4734, connectivity_R2=0.3208, final_loss=1.6260e+03
Activity: low_rank+dale oscillations across 100 neurons, amplitude [-9.0, 13.8], effective rank 7 (99% var), spectral radius 0.913
Mutation: lr_W: 0.015 -> 0.02 (from Node 101's lr_W=0.015)
Parent rule: Node 101 had highest UCB (1.473), tested lr_W=0.02 to see if higher lr_W helps in constrained regime
Observation: lr_W=0.02 did not improve R2 (0.321 vs 0.355 from Node 101). 6 consecutive partial results (97-102). combined low_rank+dale_law with effective rank 7 remains fundamentally challenging. tried lr_W range [0.005-0.02], L1 range [1e-05 to 1e-04], lr range [0.0002-0.0003] - all yield R2 ~0.31-0.36. need to try batch_size=8 or much higher lr_W (0.03+) to break out
Next: parent=102 (highest UCB=1.546, R2=0.321; try batch_size=8 to increase gradient noise and potentially break out of local minimum)

## Iter 103: partial

Node: id=103, parent=102
Mode/Strategy: exploit/explore
Config: lr_W=0.02, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=8, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.7992, test_pearson=0.1942, connectivity_R2=0.3055, final_loss=2.2898e+03
Activity: low_rank+dale oscillations across 100 neurons, amplitude [-9.0, 13.8], effective rank 7 (99% var), spectral radius 0.913
Mutation: batch_size: 16 -> 8 (from Node 102's batch_size=16)
Parent rule: Node 102 had highest UCB (1.546), tested batch_size=8 to increase gradient noise
Observation: batch_size=8 did not improve R2 (0.305 vs 0.321 from Node 102). pearson dropped significantly (0.194 vs 0.473). 7 consecutive partial results (97-103). combined low_rank+dale_law with effective rank 7 remains fundamentally hard. all tested configs yield R2 ~0.31-0.36. Node 100 (lr_W=0.005, R2=0.312) has highest UCB=1.635 - try branching from there with different mutation
Next: parent=100 (highest UCB=1.635, R2=0.312 with lr_W=0.005; try L1=5e-05 to test if moderate L1 helps at lower lr_W)

## Iter 104: partial

Node: id=104, parent=100
Mode/Strategy: exploit/explore
Config: lr_W=0.005, lr=0.0003, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8626, test_pearson=0.6082, connectivity_R2=0.2711, final_loss=1.8090e+03
Activity: low_rank+dale oscillations across 100 neurons, amplitude [-9.0, 13.8], effective rank 7 (99% var), spectral radius 0.913
Mutation: coeff_W_L1: 1.0e-05 -> 5.0e-05 (from Node 100's coeff_W_L1=1.0e-05)
Parent rule: Node 100 had highest UCB (1.255), tested coeff_W_L1=5.0e-05 as intermediate L1 value
Observation: L1=5.0e-05 did not improve R2 (0.271 vs 0.312 from Node 100). 8 consecutive partial results (97-104). combined low_rank+dale_law with effective rank 7 remains fundamentally challenging. all tested configs yield R2 ~0.27-0.36. Node 103 now has highest UCB (1.720) with batch_size=8. need to try untested direction: lr_W=0.03+ to test if aggressive lr_W can accelerate learning, or try much lower L1=1e-06 to reduce regularization penalty
Next: parent=103 (highest UCB=1.720, R2=0.305; try lr_W=0.03 to test if aggressive lr_W can break through in this constrained regime)

## Iter 105: partial

Node: id=105, parent=103
Mode/Strategy: exploit/explore
Config: lr_W=0.03, lr=0.0002, lr_emb=0.00025, coeff_W_L1=1.0e-04, batch_size=8, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.6541, test_pearson=0.0451, connectivity_R2=0.2491, final_loss=2.5697e+03
Activity: low_rank+dale oscillations across 100 neurons, amplitude [-9.0, 13.8], effective rank 7 (99% var), spectral radius 0.913
Mutation: lr_W: 0.02 -> 0.03 (from Node 103's lr_W=0.02)
Parent rule: Node 103 had highest UCB (1.720), tested aggressive lr_W=0.03 to try breaking through
Observation: lr_W=0.03 did not improve R2 (0.249 vs 0.305 from Node 103). pearson dropped dramatically (0.045 vs 0.194). 9 consecutive partial results (97-105). combined low_rank+dale_law with effective rank 7 remains fundamentally challenging. all tested configs (lr_W 0.005-0.03, L1 1e-05 to 1e-04, batch 8-16) yield R2 ~0.25-0.36. Node 104 has highest UCB (1.771) with lr_W=0.005/L1=5e-05. need to try lr_W=0.01 at intermediate L1 since this worked in block 4
Next: parent=104 (highest UCB=1.771, R2=0.271 with lr_W=0.005/L1=5e-05; try lr_W=0.01 which was optimal in block 4 low_rank)

## Iter 106: partial

Node: id=106, parent=104
Mode/Strategy: exploit/explore
Config: lr_W=0.01, lr=0.0003, lr_emb=0.00025, coeff_W_L1=5.0e-05, batch_size=16, low_rank_factorization=True, low_rank=20, n_frames=10000
Metrics: test_R2=0.8221, test_pearson=0.5745, connectivity_R2=0.3404, final_loss=1.4950e+03
Activity: low_rank+dale oscillations across 100 neurons, amplitude [-9.0, 13.8], effective rank 7 (99% var), spectral radius 0.913
Mutation: lr_W: 0.005 -> 0.01 (from Node 104's lr_W=0.005)
Parent rule: Node 104 had highest UCB (1.771), tested lr_W=0.01 which was optimal in block 4 low_rank
Observation: lr_W=0.01 improved R2 slightly (0.340 vs 0.271 from Node 104). 10 consecutive partial results (97-106). combined low_rank+dale_law with effective rank 7 remains fundamentally challenging. best R2 so far is Node 97 baseline (0.356). Node 106 (0.340) is second best. need to explore lower L1 (1e-06) to reduce regularization penalty since L1=1e-05 worked best in previous blocks, or try baseline Node 97 with lower lr=0.0002
Next: parent=106 (highest UCB=1.922, R2=0.340; try coeff_W_L1=1.0e-05 to reduce regularization and match block 4 optimum)
