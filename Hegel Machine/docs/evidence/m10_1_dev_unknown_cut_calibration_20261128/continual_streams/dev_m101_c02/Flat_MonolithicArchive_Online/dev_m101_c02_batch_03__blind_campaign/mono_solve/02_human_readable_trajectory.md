# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_m101_c02_batch_03__blind_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel_M10_1_UnknownCut_DAG_Discovery` |
| **Domain** | `blind_multi_stage_scientific_computation_graph` |
| **Wave / Wormhole Family** | `m10_1_bounded_unknown_cut_dag_discovery` / `m101_blind_dag` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `10 / 14` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `248,471` (`238,940` / `89,600` / `9,531`) |
| **Total Episode Wall Time (s)** | `313.94s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`16804 ms` LLM)
- **Rationale**: Inspect the small public batch and prior validation evidence to determine which panel operators merit testing and which archived assumptions may not transfer.

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; cat /tmp/hegel_continual_archive/archive_manifest.json /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json | head -c 14500; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust affine reference panel alignment ridge nuisance profiling' 2>&1 | head -30; python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
for k,v in s.items():
 for l,a in (v.items() if isinstance(v,dict) else [('',v)]):
  a=np.asarray(a); print(k,l,a.shape,'range',a.min(),a.max()); print(a[:4] if a.ndim>1 or a.size>10 else a)
PY
```
- **Container Exit Code**: `0` (`473 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
work
eval_times  (8,) range 0.25 3.0
[0.25   0.6429 1.0357 1.4286 1.8214 2.2143 2.6071 3.    ]
panel_a u (48, 4) range -1.294426098158964 1.2598095747163478
[[ 1.0402  0.2962 -0.2245 -1.2424]
 [ 0.411  -0.1238  0.4861  0.1657]
 [-0.4771  0.4401 -0.7365 -0.0251]
 [-0.1586 -0.3645  1.1453  0.3341]]
panel_a v (48, 4) range -5.516338101987387 7.196865277398462
[[ 1.0562  0.2072  0.2745 -2.5263]
 [ 0.3402 -0.3999  0.5212  0.1593]
 [-0.7198  0.4273 -0.7938  0.1888]
 [-0.073  -0.9489  1.473   0.5165]]
panel_b u (24, 4) range -1.2365655168297178 1.234615656710973
[[-0.7574  0.1328  1.1051 -0.3441]
 [ 0.0673 -0.1448 -1.2158  0.7777]
 [ 0.2152  0.0465 -0.9556  0.32  ]
 [-0.9459  0.0056 -0.2244  1.0092]]
panel_b v (24, 4) range -1.2835591242426878 1.198940868978237
[[ 0.1512 -0.7166 -0.6821 -0.6044]
 [ 0.5162  0.31    0.9597  0.7557]
 [ 0.1108  0.2124  0.6747  0.6849]
 [ 1.1989 -0.2805  0.237   0.2286]]
panel_c u (24, 4) range -1.2312984979759833 1.1017735168894904
[[ 0.3713  0.7328  0.4582 -0.5864]
 [ 0.6158  0.0424 -0.2198  1.012 ]
 [-0.3919  1.0759 -1.0711  0.1302]
 [-0.6538 -0.8673  0.8338 -0.7303]]
panel_c v1 (24, 4) range -1.6187958046928754 1.4131285096354569
[[ 0.8597 -0.643  -0.3809  0.6595]
 [-0.7538 -0.3478  0.5218  0.0858]
 [ 0.635  -0.1399  0.8905 -0.8285]
 [ 0.5409  1.0183 -1.316   0.0173]]
panel_c v2 (24, 4) range -0.5902630975330736 0.8290339666977004
[[-0.4476 -0.2252  0.1668  0.4462]
 [ 0.2014  0.4982  0.6907  0.4748]
 [-0.1602 -0.4662  0.633  -0.0239]
 [ 0.0716 -0.0102 -0.5255  0.4359]]
train x (72, 4) range -2.0916814556936005 1.6901924571160738
[[ 0.2782  0.4355  0.4798 -1.4691]
 [ 0.278  -0.7011  0.7377  0.548 ]
 [-0.3135  0.7518 -1.3386  1.1801]
 [ 0.1755 -1.2899  0.3787  0.2492]]
train y (72,) range -1.7022090139877855 2.0447413301242046
[ 0.519  -0.8736  0.1625 -0.0968]
train traj (72, 8) range -3.4020677225843494 3.6239004469506924
[[ 0.3202  0.5944  0.7047  0.7375  0.7545  0.7404  0.7272  0.718 ]
 [-0.0728 -0.1257 -0.1658 -0.1798 -0.1748 -0.1759 -0.1638 -0.1592]
 [ 0.517   1.0715  1.4744  1.7573  1.9887  2.1678  2.3162  2.4377]
 [-0.5578 -1.1195 -1.4532 -1.6649 -1.8378 -1.9396 -2.0312 -2.1039]]
train summary (4,) range -0.020791284939081493 1.7333894593096406
[-0.0208  0.8942  0.0899  1.7334]
test x (36, 4) range -1.8865272967019793 1.540224918690083
[[ 1.1136  0.4891 -1.0211 -0.4101]
 [-0.7598  0.5781  0.9969 -0.0781]
 [-0.3149  0.9806 -0.7139  0.1978]
 [-0.6572  0.0828 -0.0531 -0.5135]]
telemetry_vec  (4,) range 0.1 0.2
[0.1 0.2 0.2 0.2]
```

### Turn 2 — `execute_in_remote_docker` (`12652 ms` LLM)
- **Rationale**: Read the three small prior solvers and their gate results to determine whether calibration, trajectory evolution, or the readout needs a different model.
- **Alternatives Weighed**: `Fit raw-input regressors immediately` (rejected: The archive can identify reusable operators before spending training rows on an unconstrained surrogate.)

```bash
cat /tmp/hegel_continual_archive/*_solver.py; cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; cat /holdout_task/public_asset_index.json
```
- **Container Exit Code**: `0` (`344 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
type": "ndarray"
              }
            },
            "panel_b": {
              "u": {
                "dtype": "float64",
                "shape": [
                  24,
                  4
                ],
                "type": "ndarray"
              },
              "v": {
                "dtype": "float64",
                "shape": [
                  24,
                  4
                ],
                "type": "ndarray"
              }
            },
            "panel_c": {
              "u": {
                "dtype": "float64",
                "shape": [
                  24,
                  4
                ],
                "type": "ndarray"
              },
              "v1": {
                "dtype": "float64",
                "shape": [
                  24,
                  4
                ],
                "type": "ndarray"
              },
              "v2": {
                "dtype": "float64",
                "shape": [
                  24,
                  4
                ],
                "type": "ndarray"
              }
            },
            "telemetry_vec": {
              "dtype": "float64",
              "shape": [
                4
              ],
              "type": "ndarray"
            },
            "test": {
              "x": {
                "dtype": "float64",
                "shape": [
                  36,
                  4
                ],
                "type": "ndarray"
              }
            },
            "train": {
              "summary": {
                "dtype": "float64",
                "shape": [
                  4
                ],
                "type": "ndarray"
              },
              "traj": {
                "dtype": "float64",
                "shape": [
                  72,
                  8
                ],
                "type": "ndarray"
              },
              "x": {
                "dtype": "float64",
                "shape": [
                  72,
                  4
                ],
                "type": "ndarray"
              },
              "y": {
                "dtype": "float64",
                "shape": [
                  72
                ],
                "type": "ndarray"
              }
            }
          }
        },
        "len": 1,
        "type": "list"
      }
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 1509
    }
  },
  "schema": "hegel_public_asset_index_v1"
}
```

### Turn 3 — `execute_in_remote_docker` (`24083 ms` LLM)
- **Rationale**: Compare panel residual scales against trajectory rank and readout correlations before choosing which archived equations to test on held-out rows.
- **Alternatives Weighed**: `Assume the archive's diagonal calibration transfers` (rejected: The current panel has large discrepancies; a robust affine fit can distinguish sparse corruption from genuine cross-coordinate coupling.)

```bash
cat /tmp/hegel_continual_archive/dev_m101_c02_batch_0{1,2}_solver.py; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
for key in ['panel_a','panel_b','panel_c']:
 p=s[key]; u=p['u']; a=np.c_[np.ones(len(u)),u]; q=np.c_[a,*[u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)]]
 for k,v in p.items():
  if k=='u':continue
  r=v-a@ls(a,v); print(key,k,'affine row error',np.sort(np.linalg.norm(r,axis=1)))
  if key=='panel_a':
   init=ls(a,v); X=least_squares(lambda w:(a@w.reshape(5,4)-v).ravel(), init.ravel(),loss='soft_l1',f_scale=.01).x.reshape(5,4); keep=np.linalg.norm(v-a@X,axis=1)<.1; X=ls(a[keep],v[keep]); print('robust affine',keep.sum(),X,'rmse',np.std((v-a@X)[keep]))
  else:
   print('affine and quadratic residual',np.std(r),np.std(v-q@ls(q,v))); print(ls(a,v))
y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; print('traj sv',np.linalg.svd(T,compute_uv=False)); print('mean/std',T.mean(0),T.std(0)); print('diff sv',np.linalg.svd(np.diff(T,axis=1),compute_uv=False)); print('y corr T',np.corrcoef(y,T.T)[0,1:]); print('y residual from full T rms',np.std(y-np.c_[np.ones(len(y)),T]@ls(np.c_[np.ones(len(y)),T],y))); print('largest row scales',np.max(np.abs(T),axis=1).max()); print('train summary recomputed',y.mean(),y.std(),T[:,-1].mean(),T[:,-1].std())
PY
```
- **Container Exit Code**: `0` (`572 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
873  0.743
  0.76168  0.76755  0.79439  0.80179  0.80933  0.87976  0.94548  0.94676
  0.9569   0.97961  1.00223  1.0754   1.10988  1.18189  1.1823   1.21268
  1.32876  1.42146  1.42539  1.45747  1.55255 10.69443 11.49689 12.02031]
robust affine 8 [[-0.0394  -0.10333  0.01291  0.03478]
 [ 0.9269  -0.04781 -0.01834 -0.24688]
 [-0.25048  0.96689 -0.00407 -0.00775]
 [-0.06212 -0.31736  1.17395 -0.01141]
 [-0.03715 -0.02468 -0.30367  1.35814]] rmse 0.01170566838810454
panel_b v affine row error [0.00002 0.00007 0.00008 0.00011 0.00011 0.00012 0.00012 0.00013 0.00013
 0.00014 0.00015 0.00015 0.00015 0.00016 0.00017 0.00017 0.00017 0.00017
 0.0002  0.0002  0.00022 0.00023 0.00025 0.00026]
affine and quadratic residual 8.146974300323924e-05 4.6115631584029e-05
[[ 0.01071 -0.04819  0.10397  0.03687]
 [-0.47693  0.73917  0.01866  0.0172 ]
 [-0.02942 -0.0176  -0.62544  0.61805]
 [ 0.03356  0.05177 -0.61859 -0.62287]
 [ 0.73815  0.47444  0.01518  0.06406]]
panel_c v1 affine row error [0.09781 0.13354 0.13623 0.15896 0.15906 0.15996 0.16331 0.17122 0.17232
 0.18085 0.19216 0.19813 0.20889 0.22743 0.228   0.2286  0.25109 0.25146
 0.28864 0.29211 0.29375 0.30669 0.3268  0.33181]
affine and quadratic residual 0.11219643047779465 0.07620697455452922
[[ 0.06913  0.03592 -0.12134 -0.0738 ]
 [-0.56618 -0.69015 -0.12193  0.34094]
 [ 0.59298 -0.36021  0.625    0.22872]
 [ 0.25145  0.35566 -0.23969  0.83202]
 [-0.49296  0.28561  0.65377  0.16969]]
panel_c v2 affine row error [0.07817 0.08532 0.10391 0.10997 0.1152  0.12217 0.128   0.1315  0.13318
 0.13487 0.13795 0.13908 0.13962 0.15777 0.17691 0.18226 0.18659 0.20237
 0.20448 0.21771 0.22667 0.23802 0.24386 0.28381]
affine and quadratic residual 0.08504828182718624 0.003827523014171047
[[-0.03509  0.07773  0.19664  0.30855]
 [-0.20609  0.28063  0.12591  0.16306]
 [-0.16417 -0.34768  0.23607  0.04812]
 [ 0.01409 -0.14554 -0.21741  0.38942]
 [ 0.24792  0.04386  0.28077  0.10966]]
traj sv [32.40441  1.69477  0.04976  0.04453  0.04111  0.03842  0.03795  0.03619]
mean/std [0.01937 0.04102 0.05645 0.06648 0.07375 0.08107 0.08625 0.08976] [0.43861 0.88304 1.16    1.3448  1.47861 1.58171 1.66405 1.73305]
diff sv [5.03429 0.736   0.07804 0.07281 0.0704  0.05216 0.03819]
y corr T [-0.22596 -0.24644 -0.26657 -0.28196 -0.29527 -0.30563 -0.3146  -0.32136]
y residual from full T rms 0.7483404064087705
largest row scales 3.6239004469506924
train summary recomputed -0.020177428949508056 0.8938443821083369 0.0897562555274954 1.7330500791466505
```

### Turn 4 — `execute_in_remote_docker` (`24018 ms` LLM)
- **Rationale**: Panel leave-one-out error distinguishes a reproducible nonlinear calibration from a polynomial that only interpolates the panel. Coefficient patterns will show whether a compact cyclic law is supported.
- **ATMS NOGOOD DECLARED**: Refuted `The archived affine calibration can be reused after excluding the three gross corruptions.` | Witness: `Robust affine fitting left only 8 of 48 calibration rows within 0.1 in row norm, while the three largest ordinary affine residuals exceed 10.` | Blanket Diagnosis: `Gross contamination alone does not explain the systematic deviations in Panel A.` | Sublated Pivot: `Retain outlier exclusion and test sparse polynomial or rational calibration terms before modeling the readouts.`

```bash
cat /tmp/hegel_continual_archive/dev_m101_c02_batch_02_solver.py; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
for panel,keys in [('panel_a',['v']),('panel_c',['v1','v2'])]:
 p=s[panel]; u=p['u']; k=np.max(np.abs(p['v']-u),axis=1)<3 if panel=='panel_a' else np.ones(len(u),bool); u=u[k]; print(panel,'retained',sum(k));
 for key in keys:
  v=p[key][k]; print(key)
  for degree in range(1,5 if panel=='panel_a' else 3):
   obj=PolynomialFeatures(degree); f=obj.fit_transform(u); a=np.linalg.lstsq(f,v,rcond=None)[0]; hat=f@np.linalg.pinv(f); e=v-f@a; print(degree,'residual',np.std(e),'LOO',np.sqrt(np.mean((e/np.maximum(.000001,1-np.diag(hat)[:,None]))**2))); 
   if degree==2 or (panel=='panel_a' and degree==3):
    for n,row in zip(obj.get_feature_names_out(),a):
     if np.max(abs(row))>.014: print(n,row)
PY
```
- **Container Exit Code**: `0` (`756 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
panel_a retained 45
v
1 residual 0.18389627553687293 LOO 0.2157733659118115
2 residual 0.03872511702189163 LOO 0.06266626127458504
1 [-0.0652 -0.1205  0.0248  0.0372]
x0 [ 1.0616 -0.0015 -0.0153 -0.3327]
x1 [-0.3165  1.0455  0.007  -0.0043]
x2 [-0.0034 -0.3259  1.1948  0.0105]
x3 [-0.0035  0.0184 -0.3266  1.3914]
x0 x1 [ 0.3027  0.0099  0.0097 -0.0078]
x0 x2 [-0.0146 -0.0072 -0.0009 -0.0029]
x0 x3 [0.0321 0.004  0.0013 0.3341]
x1^2 [-0.0085  0.0111  0.0061  0.0209]
x1 x2 [-0.0248  0.3227 -0.0186  0.0146]
x1 x3 [-0.0202 -0.0072 -0.0125  0.0173]
x2^2 [ 0.0073  0.0182 -0.0034  0.0105]
x2 x3 [ 0.0037 -0.0003  0.3039  0.0194]
x3^2 [-0.0065  0.0042 -0.0055 -0.0144]
3 residual 4.214767927481449e-05 LOO 0.00030631648442283523
1 [-0.0647 -0.1136  0.0235  0.0634]
x0 [ 0.9006 -0.0001 -0.0001 -0.3201]
x1 [-0.3201  0.8983  0.0001 -0.    ]
x2 [ 0.0002 -0.3199  1.0421  0.0001]
x3 [ 0.0002 -0.     -0.32    1.2368]
x0 x1 [ 0.3199  0.      0.0001 -0.    ]
x0 x3 [ 0.   -0.   -0.    0.32]
x1 x2 [ 0.0001  0.32    0.0001 -0.    ]
x2 x3 [ 0.     -0.0001  0.32    0.0001]
x0^3 [ 0.16    0.0001  0.0001 -0.    ]
x1^3 [ 0.0001  0.1601 -0.0001  0.0001]
x2^3 [-0.0001 -0.0001  0.16   -0.    ]
x3^3 [-0.0001  0.0001  0.0001  0.16  ]
4 residual 7.476234288616614e-15 LOO 7.897646027470139e-09
panel_c retained 24
v1
1 residual 0.11219643047779465 LOO 0.1407481024384652
2 residual 0.07620697455452922 LOO 0.257780292010335
1 [-0.0348 -0.0172 -0.2491 -0.0704]
x0 [-0.4296 -0.631  -0.152   0.3439]
x1 [ 0.6528 -0.3467  0.643   0.1837]
x2 [ 0.1942  0.3089 -0.2271  0.8718]
x3 [-0.6528  0.2162  0.6843  0.1721]
x0^2 [-0.0258 -0.0592 -0.1213 -0.0318]
x0 x1 [ 0.0673 -0.0441 -0.0864  0.0447]
x0 x2 [-0.0611 -0.0758  0.0043  0.1802]
x0 x3 [ 0.1816  0.1201  0.1292 -0.0075]
x1^2 [ 0.2622  0.0752  0.1264 -0.0434]
x1 x2 [ 0.0875 -0.0637  0.0575  0.0202]
x1 x3 [ 0.0041  0.006   0.0372 -0.119 ]
x2^2 [-0.1153  0.03   -0.0109  0.144 ]
x2 x3 [-0.0256 -0.0751 -0.0177 -0.0485]
x3^2 [-0.0193 -0.0502  0.1636 -0.0628]
v2
1 residual 0.08504828182718624 LOO 0.10840213864841858
2 residual 0.003827523014171047 LOO 0.012381620166568194
1 [-0.0263  0.1169  0.085   0.1512]
x0 [-0.3128  0.2825  0.1516  0.1887]
x1 [-0.1645 -0.3612  0.2718  0.0568]
x2 [ 0.0242 -0.1269 -0.2237  0.41  ]
x3 [0.3315 0.0959 0.2929 0.1687]
x0^2 [-0.1361  0.1262  0.0716  0.0851]
x1^2 [-0.0703 -0.1615  0.1246  0.0316]
x2^2 [ 0.0095 -0.0605 -0.103   0.1844]
x3^2 [0.1467 0.0441 0.1257 0.0777]
```

### Turn 5 — `execute_in_remote_docker` (`49319 ms` LLM)
- **Rationale**: Check that the cyclic calibration fits the clean panel and inverts across the observed input range. Compare readout candidates on the 56/16 split, random folds, and the highest-radius inputs, reporting scalar, trajectory, and final-horizon error separately.
- **Alternatives Weighed**: `Invert a full cubic polynomial` (rejected: The fitted coefficients identify a compact cyclic law; the extra terms do not justify their extrapolation risk.)

```bash
cat >/tmp/campaign_probe.py <<'PY'
import pickle
import numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement, product

S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(a): return np.column_stack((np.ones(len(a)),a))
def poly(a,d): return np.column_stack([np.prod(a[:,idx],axis=1) for n in range(d+1) for idx in combinations_with_replacement(range(a.shape[1]),n)])
def calibrate(s, x):
    p=s['panel_a']; u,v=p['u'],p['v']; keep=np.max(abs(v-u),axis=1)<3; u,v=u[keep],v[keep]
    r=np.roll(u,-1,axis=1); n=len(u)
    F=np.column_stack((np.tile(np.eye(4),(n,1)),(u[:,:,None]*np.eye(4)).reshape(-1,4),u.ravel()**3,((u-1)*r).ravel()))
    c=ls(F,v.ravel()); b,g=c[:4],c[4:8]; a,h=c[8:];
    x=np.asarray(x); w=(x-b)/g
    for _ in range(15):
        residual=b+g*w+a*w**3+h*(w-1)*np.roll(w,-1,axis=1)-x
        if np.max(abs(residual))<1e-12: break
        J=(g+3*a*w*w+h*np.roll(w,-1,axis=1))[:,:,None]*np.eye(4)+h*(w-1)[:,:,None]*np.roll(np.eye(4),1,axis=1)
        w-=np.clip(np.linalg.solve(J,residual[...,None])[...,0],-.5,.5)
    return w,c,np.sqrt(np.mean((F@c-v.ravel())**2)),np.max(abs(residual))

if __name__=='__main__':
    np.set_printoptions(precision=5,suppress=True)
    tr=S['train']; x=tr['x']; T=tr['traj']; y=tr['y']; t=S['eval_times']; p=S['panel_b']; C=S['panel_c']; w,c,pe,ie=calibrate(S,np.r_[x,S['test']['x']]); print('A parameters, panel RMSE, inverse error, input ranges',c,pe,ie,w.min(0),w.max(0)); w=w[:len(x)]
    op=ls(aff(p['u']),p['v']); z=aff(w)@op; z0=z-op[0]
    c1=aff(w)@ls(aff(C['u']),C['v1']); c2=np.c_[aff(w),w**2]@ls(np.c_[aff(C['u']),C['u']**2],C['v2']); print('C2 additive square rmse',np.std(np.c_[aff(C['u']),C['u']**2]@ls(np.c_[aff(C['u']),C['u']**2],C['v2'])-C['v2']))
    rng=np.random.default_rng(14); idx=np.arange(len(x)); folds=[(idx[:56],idx[56:])]; folds += [(np.setdiff1d(idx,i),i) for i in np.array_split(rng.permutation(idx),5)]; nrm=np.linalg.norm(w,axis=1); order=np.argsort(nrm); folds.append((order[:-12],order[-12:])); target=np.c_[y,T]
    def evaluate(f):
        errs=np.array([np.mean(abs(f[j]@ls(f[i],target[i])-target[j]),axis=0) for i,j in folds]); return np.round(np.c_[errs[:,0],errs[:,1:].mean(1),errs[:,-1]][[0,-1]],5),np.round(np.c_[errs[:,0],errs[:,1:].mean(1),errs[:,-1]][1:-1].max(0),5)
    for name,u in [('raw',x),('cal',w),('B',z),('Bcenter',z0),('C1',c1),('C2',c2)]:
        print(name)
        for k,f in [('linear',aff(u)),('pairSum',np.c_[aff(u),u[:,0]*u[:,2]+u[:,1]*u[:,3]]),('pair',np.c_[aff(u),u[:,0]*u[:,2],u[:,1]*u[:,3]]),('sq',np.c_[aff(u),u**2]),('cubic',np.c_[aff(u),u**3]),('poly2',poly(u,2)),('poly3',poly(u,3))]:
            if name in ('C1','C2') and k not in ('linear','poly2'):continue
            e,m=evaluate(f); print(k,e[0],m,e[1])
    print('B scalar polynomial coefficients',ls(poly(z,3),y)); print('B first and final time quadratic coefficients',ls(poly(z,2),T[:,[0,-1]]).T); print('C2 scalar and trajectory linear coefficients',ls(aff(c2),target[:,[0,1,-1]]).T)
    _,sv,vt=np.linalg.svd(T,full_matrices=False); funcs={'exp':lambda r:-np.expm1(-t*r),'log':lambda r:np.log1p(t*r),'hyp':lambda r:t/(1+t*r),'sin':lambda r:np.sin(t*r),'lin':lambda r:t,'tanh':lambda r:np.tanh(t*r)}; weights=sv[:2,None]*vt[:2]; out=[]
    for k,l in combinations_with_replacement(funcs,2):
        def design(r):return np.column_stack((funcs[k](r[0]),funcs[l](r[1])))
        def residual(r):
            d=design(r);return (d@ls(d,weights.T)-weights.T).ravel()
        res=min([least_squares(residual,r,bounds=(.001,8),max_nfev=100) for r in ([.6,1.7],[1.7,.6],[.2,3.5])],key=lambda a:a.cost);out.append((np.linalg.norm(res.fun),k,l,res.x))
    print('temporal families',*sorted(out,key=lambda a:a[0])[:12],sep='\n')
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/campaign_probe.py
```
- **Container Exit Code**: `0` (`833 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.00314]
pairSum [0.00366 0.00463 0.00557] [0.00472 0.00447 0.00579] [0.00434 0.00426 0.00322]
pair [0.00357 0.00468 0.00569] [0.00473 0.00462 0.00635] [0.00536 0.00464 0.00401]
sq [0.08783 0.00489 0.0061 ] [0.08478 0.00446 0.00575] [0.11926 0.00455 0.00382]
cubic [0.09193 0.00468 0.00596] [0.07727 0.00448 0.00506] [0.12318 0.00468 0.00379]
poly2 [0.00443 0.00515 0.00541] [0.00506 0.005   0.00593] [0.00589 0.00526 0.00449]
poly3 [0.00732 0.00851 0.00995] [0.01016 0.00732 0.00946] [0.00886 0.01083 0.01256]
C1
linear [0.09063 0.00459 0.00557] [0.07874 0.00435 0.00552] [0.11732 0.0042  0.00314]
poly2 [0.00443 0.00515 0.00541] [0.00506 0.005   0.00593] [0.00589 0.00526 0.00449]
C2
linear [0.24572 0.24239 0.32618] [0.27504 0.32355 0.43216] [0.36429 0.48376 0.6479 ]
poly2 [0.15494 0.1217  0.16647] [0.15982 0.16912 0.22132] [0.1622  0.30407 0.41089]
B scalar polynomial coefficients [-0.00129 -0.68128 -0.70259  0.89693  0.548    0.00127 -0.00244  0.17888
  0.00096  0.00039  0.00125  0.17759  0.00052  0.00203  0.00087 -0.00459
  0.0093   0.00045  0.00216 -0.00039 -0.00068  0.00809 -0.00176  0.00184
  0.00663 -0.00191 -0.00123 -0.0012   0.00042 -0.00359  0.00636  0.00161
  0.00088  0.00168  0.0044 ]
B first and final time quadratic coefficients [[-0.00264 -0.1259   0.40274 -0.39338  0.45813 -0.00211 -0.00753 -0.00228
   0.0024   0.00041 -0.00122 -0.00267  0.00423  0.00041  0.00258]
 [-0.00049  0.13597  1.66324 -1.5362   1.85195 -0.00052 -0.00266  0.00058
  -0.00022  0.00126 -0.00486  0.00163 -0.00104 -0.00803  0.00227]]
C2 scalar and trajectory linear coefficients [[ 0.93447 -0.85734  0.18502 -0.34485 -2.35152]
 [-0.3304  -0.61472 -0.32395  0.92677  0.4155 ]
 [-1.35731 -1.45616 -1.32762  4.0939   1.76194]]
temporal families
(0.01679373671227363, 'exp', 'exp', array([0.42304, 1.81159]))
(0.036095913329806, 'exp', 'hyp', array([1.81771, 0.70755]))
(0.04431702373747354, 'hyp', 'tanh', array([0.75116, 0.35308]))
(0.048861000514198116, 'exp', 'tanh', array([1.78835, 0.34122]))
(0.06584688161555231, 'hyp', 'hyp', array([0.45116, 0.74027]))
(0.06928406852500392, 'hyp', 'sin', array([0.79305, 0.39608]))
(0.08100338088090657, 'exp', 'sin', array([1.74942, 0.38355]))
(0.12756017930997915, 'exp', 'log', array([1.68714, 0.93527]))
(0.1513304519685167, 'log', 'hyp', array([0.8979 , 0.89789]))
(0.1513304527743561, 'log', 'log', array([0.89796, 0.89785]))
(0.22953664704923304, 'tanh', 'tanh', array([0.3982 , 1.74078]))
(0.2393966794789977, 'sin', 'tanh', array([0.42603, 1.68577]))
```

### Turn 6 — `execute_in_remote_docker` (`47979 ms` LLM)
- **Rationale**: Build the compact solver and compare origin-constrained readouts against fitted intercepts and independent horizon fits on real held-out rows. Panel leave-one-out error checks whether the cubic inverse rests on a reproducible calibration rather than interpolation.
- **Alternatives Weighed**: `Use a full quadratic readout in calibrated coordinates` (rejected: The shared panel-B interaction already improves held-out scalar error, and quadratic trajectory terms worsen extrapolation.); `Use Panel C` (rejected: The tested Panel C features did not improve either output; the Panel B interaction accounts for the scalar nonlinearity with fewer terms.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
"""Invert the panel-A calibration, then fit the panel-B readouts."""
import numpy as np
from scipy.optimize import least_squares


def _ls(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration_design(u):
    eye = np.eye(u.shape[1])
    return np.column_stack((np.tile(eye, (len(u), 1)),
                            (u[:, :, None] * eye).reshape(-1, u.shape[1]),
                            u.ravel()**3,
                            ((u - 1) * np.roll(u, -1, axis=1)).ravel()))


def _coordinates(study, x):
    a = study['panel_a']
    u, v = a['u'], a['v']
    # The three corrupted rows are separated from the clean calibration rows.
    keep = np.max(np.abs(v - u), axis=1) < 3
    offset, gain, cubic, coupling = np.split(
        _ls(_calibration_design(u[keep]), v[keep].ravel()), [4, 8, 9])
    w = (x - offset) / gain
    eye = np.eye(4)
    cyclic = np.roll(eye, 1, axis=1)
    for _ in range(15):
        nxt = np.roll(w, -1, axis=1)
        error = offset + gain*w + cubic*w**3 + coupling*(w-1)*nxt - x
        if np.max(np.abs(error)) < 1e-12:
            break
        jac = (gain + 3*cubic*w*w + coupling*nxt)[:, :, None] * eye
        jac += (coupling*(w-1))[:, :, None] * cyclic
        w -= np.clip(np.linalg.solve(jac, error[..., None])[..., 0], -0.5, 0.5)
    b = study['panel_b']
    return _affine(w) @ _ls(_affine(b['u']), b['v'])


def _scalar_design(z):
    return np.column_stack((z, z[:, 0]*z[:, 2] + z[:, 1]*z[:, 3]))


def _trajectory(z, target, query, times):
    fitted = z @ _ls(z, target)

    def basis(rates):
        return -np.expm1(-times[:, None]*rates)

    def residual(rates):
        b = basis(rates)
        return (b @ _ls(b, fitted.T) - fitted.T).ravel()

    rates = least_squares(residual, [0.4, 1.8], bounds=([0.01, 0.8], [0.8, 8])).x
    b = basis(rates)
    loadings = _ls(z, _ls(b, target.T).T)
    return (query @ loadings) @ b.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    z = _coordinates(study, train['x'])
    q = _coordinates(study, study['test']['x'])
    y = _scalar_design(q) @ _ls(_scalar_design(z), train['y'])
    state = _trajectory(z, train['traj'], q, study['eval_times'])
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import batch_solver as m
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; tr=s['train']; y,T=tr['y'],tr['traj']; t=s['eval_times']; z=m._coordinates(s,tr['x']); n=len(z); ids=np.arange(n); radius=np.argsort(np.linalg.norm(z,axis=1)); folds=[('56/16',ids[:56],ids[56:]),('large radius',radius[:-12],radius[-12:])]; folds += [('random '+str(k),np.setdiff1d(ids,j),j) for k,j in enumerate(np.array_split(np.random.default_rng(14).permutation(ids),5))]; print('scalar coeffs with optional intercept',m._ls(m._affine(m._scalar_design(z)),y)); print('traj intercept',m._ls(m._affine(z),T)[0]);
for name,i,j in folds:
 d,q=m._scalar_design(z[i]),m._scalar_design(z[j]); yp=q@m._ls(d,y[i]); ya=m._affine(q)@m._ls(m._affine(d),y[i]);
 a=m._trajectory(z[i],T[i],z[j],t); b=m._trajectory(m._affine(z[i]),T[i],m._affine(z[j]),t); l=z[j]@m._ls(z[i],T[i]);
 print(name,'scalar origin/affine',np.mean(abs(yp-y[j])),np.mean(abs(ya-y[j])),'trajectory origin/affine/free mean+end',*[tuple(np.round([np.mean(abs(v-T[j])),np.mean(abs(v[:,-1]-T[j,-1]))],6)) for v in (a,b,l)])
p=s['panel_a']; d=np.max(abs(p['v']-p['u']),axis=1); keep=d<3; u,v=p['u'][keep],p['v'][keep]; r=[]
for i in range(len(u)):
 j=np.arange(len(u))!=i; r.extend(m._calibration_design(u[i:i+1])@m._ls(m._calibration_design(u[j]),v[j].ravel())-v[i]);
print('panel A cutoff gap and LOO RMSE',d[keep].max(),d[~keep].min(),np.sqrt(np.mean(np.square(r))))
PY
python3 /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`735 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
scalar coeffs with optional intercept [ 4.78032519e-04 -6.84068400e-01 -6.99638952e-01  8.99192553e-01
  5.51616101e-01  1.78550455e-01]
traj intercept [-6.35088447e-04 -6.53336549e-04  5.01655351e-05 -1.71716488e-04
 -5.47807757e-04  3.86093709e-04  7.74087451e-04  6.56668301e-05]
56/16 scalar origin/affine 0.0036959002951149796 0.003658507814264496 trajectory origin/affine/free mean+end (0.004459, 0.005525) (0.004436, 0.005522) (0.004544, 0.005548)
large radius scalar origin/affine 0.0038230563153953865 0.0038802158166120807 trajectory origin/affine/free mean+end (0.003719, 0.004412) (0.003885, 0.00482) (0.004103, 0.004126)
random 0 scalar origin/affine 0.004588092673259733 0.004576693156457304 trajectory origin/affine/free mean+end (0.003939, 0.00352) (0.004021, 0.003469) (0.004041, 0.003481)
random 1 scalar origin/affine 0.004578937041144406 0.004721266803607513 trajectory origin/affine/free mean+end (0.003998, 0.00355) (0.00405, 0.003776) (0.003939, 0.003705)
random 2 scalar origin/affine 0.0031187132716181676 0.003054129944984742 trajectory origin/affine/free mean+end (0.003637, 0.005378) (0.003606, 0.005527) (0.003735, 0.005457)
random 3 scalar origin/affine 0.0030555664174487388 0.003065751995684222 trajectory origin/affine/free mean+end (0.004182, 0.004103) (0.004195, 0.004358) (0.004329, 0.004084)
random 4 scalar origin/affine 0.003558677630535997 0.00355244243213606 trajectory origin/affine/free mean+end (0.004179, 0.005381) (0.004144, 0.005304) (0.004281, 0.005376)
panel A cutoff gap and LOO RMSE 1.3437022986975198 5.991397748471191 9.320084822826543e-05
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7d11edbeda68ad79 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7d11edbe (directory).
```

### Turn 7 — `execute_in_remote_docker` (`33358 ms` LLM)
- **Rationale**: The public schema already passed. Exercise the exact solver in an isolated process to catch interface or per-horizon regressions, and compare affine and cyclic calibration with the same readouts to isolate the source of the improvement. Revisit the candidate only if these checks expose a failure.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', root + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open(root + 'public_batch.pkl', 'rb'))[0]['study']
tr = s['train']

def check(out, n):
    y, t, d = (out[k] for k in ('y_hat', 'state_pred', 'diagnostics_vec'))
    assert y.shape == (n,) and t.shape == (n, len(s['eval_times'])) and d.shape == (4,)
    assert all(np.isfinite(a).all() for a in (y, t, d))
    np.testing.assert_allclose(d, [y.mean(), y.std(), t[:, -1].mean(), t[:, -1].std()], rtol=0, atol=1e-14)
    return y, t

check(m.solve_batch(s), len(s['test']['x']))
ids = np.arange(len(tr['x']))
radius = np.argsort(np.linalg.norm(m._coordinates(s, tr['x']), axis=1))
folds = [('56/16', ids[:56], ids[56:]), ('large radius', radius[:-12], radius[-12:])]
folds += [(f'random {k}', np.setdiff1d(ids, j), j) for k,j in enumerate(np.array_split(np.random.default_rng(14).permutation(ids), 5))]
a = s['panel_a']; keep = np.max(abs(a['v']-a['u']), axis=1)<3
u, v = a['u'][keep], a['v'][keep]
cal_errors = []
for k in range(len(u)):
    i = np.arange(len(u)) != k
    affine = m._affine(u[k:k+1]) @ m._ls(m._affine(u[i]), v[i])
    cyclic = m._calibration_design(u[k:k+1]) @ m._ls(m._calibration_design(u[i]), v[i].ravel())
    cal_errors.append([affine.ravel()-v[k], cyclic-v[k]])
print('Panel A LOO RMSE [affine, cyclic]:', np.sqrt(np.mean(np.square(cal_errors), axis=(0,2))))
affine_fit = m._ls(m._affine(u), v)
b = s['panel_b']; b_op = m._ls(m._affine(b['u']), b['v'])
z_aff = m._affine(np.linalg.solve(affine_fit[1:].T, (tr['x']-affine_fit[0]).T).T) @ b_op
rows = []
for name, i, j in folds:
    fold = dict(s, train={k: tr[k][i] for k in ('x', 'y', 'traj')}, test={'x': tr['x'][j]})
    y, t = check(m.solve_batch(fold), len(j))
    ya = m._scalar_design(z_aff[j]) @ m._ls(m._scalar_design(z_aff[i]), tr['y'][i])
    ta = m._trajectory(z_aff[i], tr['traj'][i], z_aff[j], s['eval_times'])
    def metrics(y, t):
        e = np.abs(t-tr['traj'][j]); return [np.mean(np.abs(y-tr['y'][j])), e.mean(), e[:, -1].mean(), e.mean(axis=0).max()]
    row = metrics(y, t); rows.append(row)
    print(name, 'cyclic', np.round(row, 6), 'affine', np.round(metrics(ya, ta), 6))
print('Worst cyclic MAE [scalar, trajectory, final, worst horizon]:', np.round(np.max(rows, axis=0), 6))
assert np.max(rows) < 0.01
PY
```
- **Container Exit Code**: `0` (`698 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Panel A LOO RMSE [affine, cyclic]: [2.15773366e-01 9.32008482e-05]
56/16 cyclic [0.003696 0.004459 0.005525 0.005525] affine [0.197878 0.275684 0.376161 0.376161]
large radius cyclic [0.003823 0.003719 0.004412 0.00487 ] affine [0.270604 0.430004 0.589302 0.589302]
random 0 cyclic [0.004588 0.003939 0.00352  0.00496 ] affine [0.158831 0.329219 0.446298 0.446298]
random 1 cyclic [0.004579 0.003998 0.00355  0.004832] affine [0.177551 0.200442 0.275662 0.275662]
random 2 cyclic [0.003119 0.003637 0.005378 0.005378] affine [0.138345 0.243534 0.333583 0.333583]
random 3 cyclic [0.003056 0.004182 0.004103 0.005222] affine [0.127674 0.214322 0.292719 0.292719]
random 4 cyclic [0.003559 0.004179 0.005381 0.005381] affine [0.125811 0.207005 0.277765 0.277765]
Worst cyclic MAE [scalar, trajectory, final, worst horizon]: [0.004588 0.004459 0.005525 0.005525]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7d11edbeda68ad79 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7d11edbe (directory).
```

### Turn 8 — `submit_completed_task` (`15732 ms` LLM)
- **Rationale**: The unchanged candidate passed the public schema validator and isolated-process checks for output shapes, finiteness, and diagnostics. Real held-out rows, including the highest-radius slice, confirmed the improvement without further tuning.
- **Alternatives Weighed**: `Keep affine calibration with the same readouts` (rejected: Panel A leave-one-out RMSE was 0.216 for affine calibration versus 0.0000932 for cyclic calibration, and the cyclic solver improved both outputs on every held-out slice.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:2427'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed. On the 56/16 split, MAE was 0.003696 for the scalar and 0.004459 for the trajectory. Across all seven validation slices, the worst single-horizon MAE was 0.005525.`

### Turn 9 — `execute_in_remote_docker` (`39574 ms` LLM)
- **Rationale**: Check whether the cyclic calibration misses a reproducible rational term, or whether the paired scalar readout misses a difference or square term. A consistent held-out gain would justify revisiting the solver; otherwise retain the already verified artifact.
- **Alternatives Weighed**: `Test batch-centered classification, relay modes, or Fokker–Planck/HJB operators` (rejected: The supplied study has continuous targets and one study-level telemetry vector, with no cohort labels, relay modes, or density/control observations. These tests would not measure a failure of the specified solver.)

```bash
cat > /tmp/campaign_final_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', root + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
tr = s['train']
print('Study fields:', sorted(s), 'train fields:', sorted(tr), 'telemetry shape:', np.shape(s['telemetry_vec']))
z = m._coordinates(s, tr['x'])
y = tr['y']
d = m._scalar_design(z)
p = d[:, -1]
pdiff = z[:, 0]*z[:, 2] - z[:, 1]*z[:, 3]
# A pairwise sum or difference squared produces the corresponding product and diagonal terms.
extra = np.column_stack((pdiff, z**2))
a = s['panel_a']
keep = np.max(np.abs(a['v']-a['u']), axis=1)<3
u, v = a['u'][keep], a['v'][keep]
f = m._calibration_design(u)
blocks = np.stack((u, np.roll(u, -1, axis=1), u-np.roll(u, -1, axis=1)), axis=-1).reshape(-1, 3)**2
actual = v.ravel()
rows = []
for k in range(len(u)):
    i = np.repeat(np.arange(len(u)) != k, 4)
    c = m._ls(f[i], actual[i])
    base = f[~i] @ c
    deviations = []
    for q2 in blocks.T:
        # Profile the numerator on training panel rows only.
        def fit(val):
            g = f/(1+val*q2[:, None])
            return g, m._ls(g[i], actual[i])
        def res(val):
            g, coef = fit(val[0]); return g[i]@coef-actual[i]
        val = least_squares(res, [0.001], bounds=(0, 5)).x[0]
        g, coef = fit(val)
        deviations.append(g[~i]@coef)
    rows.append(np.array([base, *deviations])-actual[~i])
print('Panel A leave-one-out RMSE [cyclic, rational self, rational neighbor, rational difference]:', np.sqrt(np.mean(np.square(rows), axis=(0,2))))
ids = np.arange(len(z))
order = np.argsort(np.linalg.norm(z, axis=1))
folds = [('56/16', ids[:56], ids[56:]), ('large radius', order[:-12], order[-12:])]
folds += [(f'random {k}', np.setdiff1d(ids, j), j) for k,j in enumerate(np.array_split(np.random.default_rng(14).permutation(ids), 5))]
errors = []
for name, i, j in folds:
    # Profile three possible denominator directions on each training fold and select with inner CV only.
    qs = [z[:,0]-z[:,2], z[:,1]-z[:,3], z[:,0]+z[:,2], z[:,1]+z[:,3], np.linalg.norm(z, axis=1)]
    def fit_rational(q, ii, jj):
        def model(k):
            g = d/(1+k*q[:,None]**2)
            return g, m._ls(g[ii], y[ii])
        def resid(k):
            g, c = model(k[0]); return g[ii]@c-y[ii]
        g, c = model(least_squares(resid, [0.001], bounds=(0, 5)).x[0])
        return g[jj] @ c
    inner = np.array_split(np.random.default_rng(35).permutation(i), 3)
    q = min(qs, key=lambda q: np.mean([np.mean(abs(fit_rational(q,np.setdiff1d(i,b),b)-y[b])) for b in inner]))
    f2 = np.column_stack((d, extra))
    preds = [d[j]@m._ls(d[i], y[i]), f2[j]@m._ls(f2[i], y[i]), fit_rational(q,i,j)]
    fold_err = np.mean(abs(np.array(preds)-y[j]), axis=1)
    errors.append(fold_err)
    print(name, 'scalar MAE [paired sum, expanded pair squares, rational]:', np.round(fold_err, 7))
print('Worst scalar MAE:', np.max(errors, axis=0))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/campaign_final_audit.py
```
- **Container Exit Code**: `0` (`959 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Study fields: ['eval_times', 'panel_a', 'panel_b', 'panel_c', 'telemetry_vec', 'test', 'train'] train fields: ['summary', 'traj', 'x', 'y'] telemetry shape: (4,)
Panel A leave-one-out RMSE [cyclic, rational self, rational neighbor, rational difference]: [9.32008482e-05 9.36376280e-05 9.13112162e-05 9.32263911e-05]
56/16 scalar MAE [paired sum, expanded pair squares, rational]: [0.0036959 0.004101  0.0038677]
large radius scalar MAE [paired sum, expanded pair squares, rational]: [0.0038231 0.0059699 0.0038231]
random 0 scalar MAE [paired sum, expanded pair squares, rational]: [0.0045881 0.0043265 0.004588 ]
random 1 scalar MAE [paired sum, expanded pair squares, rational]: [0.0045789 0.0047733 0.004579 ]
random 2 scalar MAE [paired sum, expanded pair squares, rational]: [0.0031187 0.0034267 0.0031187]
random 3 scalar MAE [paired sum, expanded pair squares, rational]: [0.0030556 0.0032891 0.0030573]
random 4 scalar MAE [paired sum, expanded pair squares, rational]: [0.0035587 0.0037127 0.0035586]
Worst scalar MAE: [0.00458809 0.00596994 0.00458804]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7d11edbeda68ad79 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7d11edbe (directory).
```

### Turn 10 — `submit_completed_task` (`17862 ms` LLM)
- **Rationale**: The unchanged solver passed the public schema and isolated-process checks. The follow-up audit found no reproducible improvement that justified replacing the verified candidate.
- **Alternatives Weighed**: `Retain affine calibration` (rejected: Panel A leave-one-out RMSE dropped from 0.216 to 0.0000932 with cyclic calibration, and both outputs improved on every held-out slice.); `Add rational or pairwise-square terms` (rejected: The best rational calibration term improved panel RMSE by only 0.0000019, with no demonstrated downstream gain. Rational scalar terms did not materially improve worst-slice MAE, and pairwise-square terms worsened the high-radius slice.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed. On the 56/16 split, MAE was 0.003696 for the scalar and 0.004459 for the trajectory. Across seven held-out slices, the worst single-horizon MAE was 0.005525.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel_M10_1_UnknownCut_DAG_Discovery",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
