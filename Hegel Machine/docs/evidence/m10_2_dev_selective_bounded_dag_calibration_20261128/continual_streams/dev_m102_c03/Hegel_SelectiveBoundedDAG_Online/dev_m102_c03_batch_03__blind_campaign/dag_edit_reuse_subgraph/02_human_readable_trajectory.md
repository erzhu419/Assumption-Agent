# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_m102_c03_batch_03__blind_campaign__reuse_subgraph`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel_M10_1_UnknownCut_DAG_Discovery` |
| **Domain** | `blind_multi_stage_scientific_computation_graph` |
| **Wave / Wormhole Family** | `m10_2_selective_bounded_dag_discovery_h16` / `m102_blind_dag` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `11 / 12` |
| **ATMS Nogoods Recorded** | `3` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `337,462` (`326,907` / `132,224` / `10,555`) |
| **Total Episode Wall Time (s)** | `302.89s` |

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

### Turn 1 — `execute_in_remote_docker` (`13149 ms` LLM)
- **Rationale**: Inspect the frozen blend, public cross-validation protocol, and archive before choosing a readout model.

```bash
cd /holdout_task && cat _dag_spec.json _frozen_nodes/node_00.py _frozen_nodes/node_02.py && cat test_dag_candidate.py && cat public_asset_index.json && ls -la /tmp/hegel_continual_archive && find /tmp/hegel_continual_archive -maxdepth 3 -type f | sort | head -100 && python3 - <<'PY'
import pickle,numpy as np
with open('public_batch.pkl','rb') as f: b=pickle.load(f)
def describe(x,depth=0):
 if isinstance(x,dict) and depth<3:
  for k,v in x.items():
   print(' '*depth,k,end=': '); describe(v,depth+1)
 elif isinstance(x,np.ndarray): print(x.shape,x.dtype,x.reshape(-1)[:12])
 else: print(type(x).__name__,repr(x)[:280])
describe(b)
PY
```
- **Container Exit Code**: `0` (`361 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
    "/holdout_task/test_dag_candidate.py": {
      "bytes": 8018
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 2825
    }
  },
  "schema": "hegel_public_asset_index_v1"
}total 20
drwxr-xr-x 2 root root 4096 Nov 28 13:27 .
drwxrwxrwt 1 root root 4096 Nov 28 13:27 ..
-rw-r--r-- 1 root root  662 Nov 28 13:27 archive_manifest.json
-rw-r--r-- 1 root root 3634 Nov 28 13:27 dev_m102_c03_batch_00_solver.py
-rw-r--r-- 1 root root 1652 Nov 28 13:27 pre_llm_gate_diagnostic.json
/tmp/hegel_continual_archive/archive_manifest.json
/tmp/hegel_continual_archive/dev_m102_c03_batch_00_solver.py
/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
list [{'study': {'eval_times': array([0.25      , 0.64285714, 1.03571429, 1.42857143, 1.82142857,
       2.21428571, 2.60714286, 3.        ]), 'panel_a': {'u': array([[-0.96884946,  0.1361999 , -0.51466993,  0.32769074],
       [ 1.23215695,  0.77158548, -0.16033005, -0.35966109],
```

### Turn 2 — `execute_in_remote_docker` (`14459 ms` LLM)
- **Rationale**: Recover the exact sink contract and archived model, then test whether an affine map explains the calibration panels before adding nonlinear terms.

```bash
cd /holdout_task; cat _frozen_nodes/node_02.py; cat /tmp/hegel_continual_archive/*; grep -nE 'fold|split|study|threshold|mae|rng|seed' test_dag_candidate.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment calibration' 2>&1 | head -30; python3 - <<'PY'
import numpy as np,pickle
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; print(s.keys());
for k,v in s.items():
 if isinstance(v,dict):
  for a,b in v.items():
   b=np.asarray(b); print(k,a,b.shape,b.mean(0).round(5),b.std(0).round(5))
 else: print(k,v)
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; u=p['u']; U=np.c_[np.ones(len(u)),u];
 for a,v in p.items():
  if a=='u': continue
  w=np.linalg.lstsq(U,v,rcond=None)[0]; print(k,a,'linear RMSE',np.sqrt(np.mean((U@w-v)**2)), 'affine',w.round(5))
PY
```
- **Container Exit Code**: `0` (`384 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
4:    cv_metrics = {"public_cv_y_mae": round(ym, 6), "public_cv_traj_mae": round(tm, 6)}
165:    if ym > 0.038: errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
166:    if tm > 0.038: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")
No matching skill cards found.
dict_keys(['eval_times', 'panel_a', 'panel_b', 'panel_c', 'train', 'test', 'telemetry_vec'])
eval_times [0.25       0.64285714 1.03571429 1.42857143 1.82142857 2.21428571
 2.60714286 3.        ]
panel_a u (48, 4) [-0.17319  0.02365 -0.09112 -0.12432] [0.72339 0.70632 0.78854 0.72845]
panel_a v (48, 4) [ 0.04081  0.26287 -0.42591 -0.34423] [1.86662 1.74929 1.86149 1.61749]
panel_b u (24, 4) [ 0.25923 -0.05386 -0.03564  0.06026] [0.66114 0.7633  0.67093 0.80236]
panel_b v (24, 4) [-0.16925  0.17494  0.21594  0.09095] [0.54201 0.63783 0.66634 0.70159]
panel_c u (24, 4) [0.21003 0.08349 0.23566 0.04842] [0.58226 0.73561 0.73652 0.80703]
panel_c v1 (24, 4) [-0.16209 -0.14614  0.06943  0.00166] [0.71806 0.61068 0.59134 0.74622]
panel_c v2 (24, 4) [-0.14524 -0.33028 -0.09393  0.12751] [0.32232 0.3601  0.38568 0.45809]
train x (72, 4) [ 0.12696 -0.01909 -0.16648 -0.07066] [0.74401 0.76237 0.9163  0.95281]
train y (72,) 0.18786 0.8454
train traj (72, 8) [0.08674 0.18274 0.24881 0.29571 0.3329  0.36225 0.38653 0.40746] [0.29056 0.5415  0.66001 0.71421 0.74341 0.7583  0.76571 0.77335]
train summary (4,) 0.55368 0.26881
test x (36, 4) [-0.09845  0.14706 -0.10139 -0.13703] [0.75783 0.86269 0.87129 0.78612]
telemetry_vec [0.22 0.22 1.25 0.22]
panel_a v linear RMSE 1.4701891473247652 affine [[ 0.37249  0.35672 -0.34112 -0.2494 ]
 [ 1.48814  0.59692 -0.52777  0.04141]
 [ 0.12955  0.94465  0.17497  0.09454]
 [ 0.36727  0.5265   1.10587 -0.28088]
 [ 0.35024 -0.2829   0.63998  0.92896]]
panel_b v linear RMSE 9.418532326612438e-05 affine [[-0.07068  0.0588   0.05698 -0.00712]
 [-0.37242  0.42524  0.66916  0.08471]
 [-0.61685 -0.16984 -0.16171 -0.58211]
 [ 0.29101  0.66919 -0.20675 -0.44627]
 [-0.41292  0.342   -0.50768  0.47878]]
panel_c v1 linear RMSE 0.11130690326447601 affine [[-0.02859  0.02452 -0.12249  0.13288]
 [-0.59822  0.04724  0.74539  0.07713]
 [ 0.62564 -0.12399  0.64869 -0.21773]
 [-0.27098 -0.57997 -0.07696 -0.65934]
 [ 0.07791 -0.69304 -0.01344  0.53969]]
panel_c v2 linear RMSE 0.0908597664019657 affine [[-0.07646 -0.2063  -0.01244  0.16581]
 [-0.51586 -0.04056 -0.24093 -0.0115 ]
 [-0.01752 -0.301    0.18664  0.3539 ]
 [ 0.15505 -0.38584 -0.11006 -0.32526]
 [ 0.09276  0.0123  -0.42416  0.23153]]
```

### Turn 3 — `execute_in_remote_docker` (`21551 ms` LLM)
- **Rationale**: Read the archive's remaining implementation and the public split logic, then measure robust calibration residuals and test the archived time constants before fitting the spatial readouts.
- **Alternatives Weighed**: `Use one affine least-squares procedure for every panel` (rejected: Residual magnitudes differ by more than three orders of magnitude across panels.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine least-squares map adequately describes every calibration panel.` | Witness: `Affine RMSE is 0.000094 for panel B, but 1.470 for panel A and 0.111/0.091 for the two panel C branches.` | Blanket Diagnosis: `The panels have different residual structures; ordinary affine fits cannot yet distinguish outliers from nonlinear maps.` | Sublated Pivot: `Keep the affine model for panel B, test robust calibration for panel A, and identify the trajectory time basis independently.`

```bash
cd /holdout_task; tail -45 /tmp/hegel_continual_archive/*solver.py; sed -n '125,205p' test_dag_candidate.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust regression IRLS'; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; Y=tr['traj']; print('summary:',tr['summary']); print('traj singular values:',np.linalg.svd(Y,compute_uv=False)); print('traj sample:',Y[:5]);
def residual(r):
 q=-np.expm1(-t[:,None]*r); return (q@np.linalg.lstsq(q,Y.T,rcond=None)[0]-Y.T).ravel()
for r0 in [[.392,1.857],[.1,.5,2],[.1,.5,2,4]]:
 r=least_squares(residual,r0,bounds=(.001,25)); print('rates=',r.x,'RMSE=',np.std(r.fun),'max=',max(abs(r.fun)))
for key in ['panel_a','panel_b','panel_c']:
 p=s[key]; u=p['u']; D=np.c_[np.ones(len(u)),u];
 for label,v in p.items():
  if label=='u':continue
  init=np.linalg.lstsq(D,v,rcond=None)[0]; sol=least_squares(lambda c:(D@c.reshape(5,4)-v).ravel(),init.ravel(),loss='cauchy',f_scale=.003).x.reshape(5,4); E=D@sol-v
  print(key,label,'robust affine:',sol.round(4),'res quantiles:',np.quantile(abs(E),[0,.2,.4,.6,.8,.9,1]).round(5)); print('row errors:',np.sort(np.linalg.norm(E,axis=1)).round(4))
PY
```
- **Container Exit Code**: `0` (`571 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
170379  -0.12967058 -0.13604902 -0.12139852 -0.13457615
  -0.13125581 -0.12011202]]
rates= [0.40288474 1.81226571] RMSE= 0.004296383460980241 max= 0.014099349276322948
rates= [1.00000000e-03 5.57633057e-01 1.82083308e+00] RMSE= 0.003896512239394526 max= 0.012737328824625749
rates= [1.00004141e-03 1.27900468e-03 4.90823760e-03 1.81494202e+00] RMSE= 0.003398011817028895 max= 0.01221639744468439
panel_a v robust affine: [[ 0.261   0.5412 -0.3873 -0.0955]
 [ 1.1859  0.489  -0.3935  0.1561]
 [ 0.0147  0.9211  0.2284  0.0421]
 [ 0.0923  0.4489  1.3372 -0.0714]
 [ 0.2557 -0.1083  0.5343  1.0634]] res quantiles: [4.00000e-05 2.92300e-02 1.65990e-01 3.19160e-01 5.72770e-01 8.81400e-01
 6.51706e+00]
row errors: [ 0.2393  0.2581  0.2781  0.2902  0.2965  0.2968  0.302   0.337   0.3651
  0.3741  0.385   0.3927  0.3929  0.4006  0.4515  0.4947  0.5045  0.5298
  0.534   0.5581  0.5616  0.5619  0.5706  0.589   0.6035  0.6186  0.6252
  0.6967  0.7071  0.7262  0.7492  0.7594  0.7904  0.8056  0.8186  0.9203
  0.9265  1.1427  1.1818  1.2636  1.3562  1.3637  1.3977  1.5881  1.8034
 10.8319 11.4383 12.3615]
panel_b v robust affine: [[-0.0707  0.0588  0.057  -0.0071]
 [-0.3724  0.4252  0.6692  0.0847]
 [-0.6169 -0.1698 -0.1617 -0.5821]
 [ 0.291   0.6692 -0.2067 -0.4463]
 [-0.4129  0.342  -0.5077  0.4788]] res quantiles: [0.0e+00 3.0e-05 7.0e-05 9.0e-05 1.3e-04 1.4e-04 2.1e-04]
row errors: [0.0001 0.0001 0.0001 0.0001 0.0001 0.0002 0.0002 0.0002 0.0002 0.0002
 0.0002 0.0002 0.0002 0.0002 0.0002 0.0002 0.0002 0.0002 0.0002 0.0002
 0.0002 0.0002 0.0002 0.0003]
panel_c v1 robust affine: [[-0.0436  0.0136 -0.1214  0.1471]
 [-0.5569  0.0891  0.7501  0.0514]
 [ 0.6474 -0.1229  0.6552 -0.1931]
 [-0.3068 -0.5988 -0.0824 -0.6739]
 [ 0.0098 -0.691   0.0038  0.4923]] res quantiles: [0.      0.00061 0.05425 0.1046  0.15523 0.21012 0.30944]
row errors: [0.1226 0.1398 0.1593 0.1695 0.1757 0.1835 0.1836 0.211  0.2167 0.2177
 0.2191 0.2286 0.2321 0.2327 0.2456 0.2494 0.2556 0.2609 0.2701 0.2709
 0.2876 0.3014 0.3127 0.3844]
panel_c v2 robust affine: [[-0.0812 -0.1847 -0.0022  0.1435]
 [-0.5292 -0.033  -0.2149 -0.0135]
 [-0.0387 -0.2948  0.1756  0.3274]
 [ 0.1719 -0.4143 -0.1275 -0.3097]
 [ 0.1062  0.0445 -0.4374  0.2212]] res quantiles: [5.0000e-05 1.5200e-03 3.8080e-02 8.8700e-02 1.2831e-01 1.6720e-01
 2.2960e-01]
row errors: [0.1019 0.1052 0.126  0.1294 0.1397 0.1497 0.1509 0.1582 0.1613 0.1717
 0.1717 0.1873 0.1956 0.1968 0.1975 0.2027 0.2045 0.2198 0.2294 0.2305
 0.2313 0.2337 0.2438 0.3227]
```

### Turn 4 — `execute_in_remote_docker` (`20175 ms` LLM)
- **Rationale**: Separate the three gross panel-A outliers from systematic nonlinear residuals, and test compact calibration terms before fitting the scalar and trajectory readouts.
- **Alternatives Weighed**: `Add more exponential modes` (rejected: The extra modes approach zero rate and improve in-sample reconstruction by less than 0.001; spatial prediction is the larger open question.)

```bash
cd /holdout_task; sed -n '130,190p' test_dag_candidate.py; tail -20 /tmp/hegel_continual_archive/*solver.py; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']
for key in ['panel_a','panel_c']:
 p=s[key];u=p['u']; D=np.c_[np.ones(len(u)),u];
 for label,v in p.items():
  if label=='u': continue
  ids=np.argsort(np.linalg.norm(D@np.linalg.lstsq(D,v,rcond=None)[0]-v,axis=1))[:len(u)-(3 if key=='panel_a' else 0)]; a=u[ids]; b=v[ids]; H=D[ids]
  print(key,label,'raw residual after trim',np.std(H@np.linalg.lstsq(H,b,rcond=None)[0]-b))
  for name,f in [('square',lambda x,k:x*x),('cubic',lambda x,k:x**3),('sin',lambda x,k:np.sin(k*x)),('cos',lambda x,k:np.cos(k*x)),('tanh',lambda x,k:np.tanh(k*x)),('gauss',lambda x,k:np.expm1(-k*x*x)),('soft',lambda x,k:x/(1+k*abs(x))),('rational',lambda x,k:1/(1+k*x*x)),('cross',lambda x,k:np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i+1,4)])),('full2',lambda x,k:np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]))]:
   def score(k):
    X=np.c_[H,f(a,k)]; return np.std(X@np.linalg.lstsq(X,b,rcond=None)[0]-b)
   ks=np.geomspace(.025,6,50); j=np.argmin([score(k) for k in ks]); r=minimize_scalar(score,bounds=(ks[max(0,j-1)],ks[min(j+1,len(ks)-1)]),method='bounded'); print(name,round(r.x,5),round(r.fun,6),end='; ')
  print()
PY
```
- **Container Exit Code**: `0` (`533 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"test"] = {"x": vx}
    out_cv, _ = run_dag(st_cv)
    ym = float(np.mean(np.abs(np.asarray(out_cv["y_hat"], float) - vy)))
    tm = float(np.mean(np.abs(np.asarray(out_cv["state_pred"], float) - vtr)))
    cv_metrics = {"public_cv_y_mae": round(ym, 6), "public_cv_traj_mae": round(tm, 6)}
    if ym > 0.038: errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
    if tm > 0.038: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")

ok = len(errs) == 0 and all(v == 1 for v in ec.values())
print(json.dumps({"accepted": ok, "errors": errs, "cv_metrics": cv_metrics, "exec_counts": ec}, allow_nan=False))
sys.exit(0 if ok else 1)
        z=z+blend*((v-np.column_stack([np.ones(len(z)),z])@b)@np.linalg.inv(b[1:]))
        t=np.tanh(z)
    else:
        z=z+blend*np.expm1(-.2962*z*z); t=None
    def quad(a):
        return [a]+[a[:,i]*a[:,j] for i in range(4) for j in range(i,4)]
    f=np.column_stack(quad(z)+(quad(t) if t is not None else [np.sin(z),np.cos(z),np.sin(2*z),np.cos(2*z)]))
    f=(f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-7); y=np.column_stack([tr['y'],tr['traj']]); ym=y.mean(0);
    return f[n:]@f[:n].T@np.linalg.solve(f[:n]@f[:n].T+reg*np.eye(n),y-ym)+ym


def solve_batch(study):
    """Blend sparse two-mode predictions with a calibrated ridge readout."""
    sparse = _sparse(study)
    panel = _panel_pred(study, 1.0, 0.3, 'sat')
    y = 0.5 * sparse['y_hat'] + 0.5 * panel[:, 0]
    state = 0.35 * sparse['state_pred'] + 0.65 * panel[:, 1:]
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
panel_a v raw residual after trim 0.16918235454428207
square 0.02795 0.163093; cubic 0.02795 0.152705; sin 1.09034 0.152693; cos 5.99999 0.159767; tanh 0.38722 0.152694; gauss 0.2961 0.163081; soft 0.02501 0.153103; rational 0.12123 0.163084; cross 0.02795 0.045211; full2 0.02795 0.043198; 
panel_c v1 raw residual after trim 0.11130690326447601
square 0.02795 0.097724; cubic 0.02795 0.033257; sin 3.40545 0.008838; cos 4.94343 0.086147; tanh 2.19441 0.003762; gauss 0.02501 0.097766; soft 0.84012 0.000502; rational 0.02501 0.097806; cross 0.02795 0.087675; full2 0.02795 0.072117; 
panel_c v2 raw residual after trim 0.0908597664019657
square 0.02795 0.00419; cubic 0.02795 0.080042; sin 5.41124 0.077839; cos 1.27253 0.00031; tanh 0.02501 0.080043; gauss 0.29342 0.000246; soft 5.99999 0.079921; rational 0.16296 0.000174; cross 0.02795 0.068739; full2 0.02795 0.00297;
```

### Turn 5 — `execute_in_remote_docker` (`24195 ms` LLM)
- **Rationale**: Measure the archived readout against the unchanged sink to establish a baseline, and test compact pairwise calibration forms against the residual structure that survived outlier removal.

```bash
cd /holdout_task; { echo 'import numpy as np'; sed -n '/^def _panel_pred/,/^def solve_batch/{ /^def solve_batch/!p; }' /tmp/hegel_continual_archive/*solver.py; printf '\ndef run_node(inputs: dict) -> dict:\n    return {"panel_pred": _panel_pred(inputs["__input__"], 1.0, 0.3, "sat")}\n'; } > node_01.py; OPENBLAS_NUM_THREADS=1 python3 -I test_dag_candidate.py; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import numpy as np,pickle
from scipy.optimize import minimize_scalar
from itertools import combinations
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; u=s['panel_a']['u']; v=s['panel_a']['v']; h=np.c_[np.ones(len(u)),u]; ids=np.argsort(np.linalg.norm(h@np.linalg.lstsq(h,v,rcond=None)[0]-v,axis=1))[:-3]; u,v,h=u[ids],v[ids],h[ids]; i,j=np.array(list(combinations(range(4),2))).T; a,b=u[:,i],u[:,j];
for name,f in [('cross',lambda k:a*b),('cross rational',lambda k:a*b/(1+k*(a*a+b*b))),('cross tanh',lambda k:np.tanh(k*a*b)),('gaussian sum',lambda k:np.c_[np.expm1(-k*(a+b)**2),np.expm1(-k*(a-b)**2)]),('pair sins',lambda k:np.c_[np.sin(k*(a+b)),np.sin(k*(a-b)),np.cos(k*(a+b)),np.cos(k*(a-b))]),('combined',lambda k:np.c_[a*b/(1+k*(a*a+b*b)),np.tanh(k*u),u*u,np.prod(u[:,np.array(list(combinations(range(4),3)))],axis=2)])]:
 def score(k,detail=False):
  d=np.c_[h,f(k)]; w=np.linalg.lstsq(d,v,rcond=None)[0]; e=d@w-v; return (w,np.std(e,axis=0)) if detail else np.std(e)
 r=minimize_scalar(score,bounds=(.01,3),method='bounded'); w,e=score(r.x,True); print(name,r.x,r.fun,e.round(5));
 if name in ('cross','cross rational'): print(w.round(4))
for label in ['v1','v2']:
 p=s['panel_c']; x=p['u']; y=p[label]; f=lambda k: x/(1+k*abs(x)) if label=='v1' else 1/(1+k*x*x); X=lambda k:np.c_[np.ones(len(x)),x,f(k)]; err=lambda k: np.std(X(k)@np.linalg.lstsq(X(k),y,rcond=None)[0]-y); r=minimize_scalar(err,bounds=(.01,3),method='bounded'); print(label,r.x,r.fun,np.linalg.lstsq(X(r.x),y,rcond=None)[0].round(4))
PY
```
- **Container Exit Code**: `0` (`585 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
pted": false, "errors": ["public_cv_y_mae_exceeded:0.143102", "public_cv_traj_mae_exceeded:0.096536"], "cv_metrics": {"public_cv_y_mae": 0.143102, "public_cv_traj_mae": 0.096536}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}
cross 2.9999965700600026 0.04521052769293171 [0.04826 0.03912 0.04577 0.04714]
[[ 0.11    0.1127 -0.134  -0.0031]
 [ 1.2511  0.0199  0.0047  0.3424]
 [ 0.3114  1.072   0.0039  0.0051]
 [ 0.0342  0.3082  1.3194  0.0162]
 [-0.0095 -0.0051  0.3184  1.3451]
 [ 0.3046 -0.0348  0.0183 -0.0176]
 [ 0.0422  0.0035  0.0026  0.0114]
 [-0.0344  0.0417 -0.01    0.3682]
 [-0.0099  0.268   0.0077  0.0103]
 [-0.0151 -0.0277  0.0104 -0.0125]
 [-0.0246  0.0546  0.3086  0.0163]]
cross rational 0.05928626592283967 0.04488521383853003 [0.0476  0.03969 0.04583 0.04601]
[[ 0.1099  0.1123 -0.1323 -0.0039]
 [ 1.2496  0.021   0.0052  0.3397]
 [ 0.3134  1.0712  0.0045  0.004 ]
 [ 0.032   0.3101  1.323   0.0133]
 [-0.0101 -0.0041  0.3177  1.3505]
 [ 0.3416 -0.0387  0.0166 -0.0137]
 [ 0.0423  0.0096  0.0098  0.0087]
 [-0.0368  0.043  -0.0175  0.4146]
 [-0.0105  0.3016  0.0119  0.0072]
 [-0.0151 -0.0313  0.0115 -0.0138]
 [-0.0279  0.0589  0.3447  0.0185]]
cross tanh 0.44575346498086904 0.04477008033105396 [0.04721 0.03969 0.04578 0.04601]
gaussian sum 0.052450538690184435 0.042116118885755505 [0.04648 0.03496 0.04021 0.04578]
pair sins 0.6113699488317174 0.024786643472846134 [0.03426 0.01652 0.02034 0.02444]
combined 0.01000347428866272 0.0007850360238601799 [0.00058 0.001   0.00083 0.00065]
v1 0.8401243869411067 0.0005015205103392187 [[-0.027   0.0098 -0.0941  0.109 ]
 [ 0.2474 -0.0364 -0.2605 -0.0417]
 [-0.2455  0.0604 -0.245   0.093 ]
 [ 0.1158  0.2314  0.0292  0.2517]
 [-0.0144  0.27   -0.0112 -0.243 ]
 [-1.5314  0.2387  1.6545  0.2734]
 [ 1.5336 -0.378   1.5494 -0.5678]
 [-0.735  -1.4572 -0.1956 -1.5813]
 [ 0.0795 -1.6977  0.0648  1.5232]]
v2 0.16296665136497854 0.00017411722221352558 [[-0.8327 -2.4477 -1.7023  0.7791]
 [-0.4419 -0.0799 -0.1614 -0.0756]
 [-0.0507 -0.3198  0.1436  0.3283]
 [ 0.1523 -0.3524 -0.1169 -0.2688]
 [ 0.111   0.0196 -0.416   0.2181]
 [ 1.468   0.2648  0.5358  0.2503]
 [ 0.1675  1.0614 -0.4765 -1.0902]
 [-0.5054  1.1704  0.3876  0.8925]
 [-0.3699 -0.0645  1.3817 -0.7244]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b840909fdd4e5851 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b840909f (directory).
```

### Turn 6 — `execute_in_remote_docker` (`40218 ms` LLM)
- **Rationale**: Fit the sparse calibration grammar directly on the reference panel and compare predictive readouts in candidate coordinate systems, using inner folds to distinguish reliable improvements from split-specific gains.
- **ATMS NOGOOD DECLARED**: Refuted `The archived diagonal calibration and ridge readout transfer to the shifted batch.` | Witness: `Both public errors exceeded 0.038, and the panel-A fit showed cyclic off-diagonal linear and interaction coefficients of roughly 0.3.` | Blanket Diagnosis: `Independent coordinate scaling omits the cyclic coupling in the calibration map.` | Sublated Pivot: `Identify and invert the coupled calibration map, then compare readout dictionaries across validation slices.`

```bash
cd /holdout_task; cat _frozen_nodes/node_00.py; cat > /tmp/probe.py <<'PY'
import numpy as np,pickle
from itertools import combinations
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; u,v=(s['panel_a'][k] for k in ('u','v')); h=np.c_[np.ones(len(u)),u]; ids=np.argsort(np.linalg.norm(h@np.linalg.lstsq(h,v,rcond=None)[0]-v,axis=1))[:-3]; u,v=u[ids],v[ids]; ij=np.array(list(combinations(range(4),2))); ijk=np.array(list(combinations(range(4),3)))
def af(x):return np.c_[np.ones(len(x)),x,np.prod(x[:,ij],axis=2),np.prod(x[:,ijk],axis=2)]
W=np.linalg.lstsq(af(u),v,rcond=None)[0];print('A squarefree coefficients, rmse',W.round(5),np.std(af(u)@W-v));
for i in range(4):
 j=(i+1)%4;k=(i+2)%4;l=(i+3)%4
 for name,F in [('cyclic',lambda x:np.c_[np.ones(len(x)),x[:,i],x[:,j],x[:,i]*x[:,j],x[:,i]*x[:,j]*x[:,k],x[:,i]*x[:,j]*x[:,l],x[:,i]*x[:,k]*x[:,l],x[:,j]*x[:,k]*x[:,l]]),('all',af)]:
  d=F(u);w=np.linalg.lstsq(d,v[:,i],rcond=None)[0]; print(i,name,np.std(d@w-v[:,i]),w.round(5))
def inverse(x,W=W):
 z=(x-W[0])@np.linalg.inv(W[1:5]);
 for _ in range(15):
  pred=af(z)@W; eps=1e-5; J=np.stack([(af(z+e*eps)@W-pred)/eps for e in np.eye(4)],axis=2); dz=np.linalg.solve(J,(pred-x)[...,None])[...,0]; z-=np.clip(dz,-.5,.5)
 return z
x=tr['x']; z=inverse(x); print('inverse',np.max(abs(af(z)@W-x)),np.quantile(z,[0,.05,.5,.95,1],axis=0));
def panel(k,name,f):
 p=s[k];u=p['u'];v=p[name]; d=lambda x,a:np.c_[np.ones(len(x)),x,f(x,a)] if f else np.c_[np.ones(len(x)),x]; err=lambda a:np.std(d(u,a)@np.linalg.lstsq(d(u,a),v,rcond=None)[0]-v); a=minimize_scalar(err,bounds=(.05,2),method='bounded').x if f else 0.; w=np.linalg.lstsq(d(u,a),v,rcond=None)[0]; return d(z,a)@w
b=panel('panel_b','v',None); c=panel('panel_c','v1',lambda x,k:x/(1+k*abs(x))); d=panel('panel_c','v2',lambda x,k:1/(1+k*x*x)); y=np.c_[tr['y'],tr['traj']];
def gram(a,kind):
 if kind=='poly': return np.c_[a,np.prod(a[:,ij],axis=2),a*a,np.prod(a[:,ijk],axis=2)]
 if kind=='rich':return np.c_[gram(a,'poly'),a**3,np.tanh(a),np.sin(a),np.cos(a),np.sin(2*a),np.cos(2*a),a/(1+abs(a)),1/(1+a*a),np.tanh(a[:,ij[:,0]]+a[:,ij[:,1]]),np.tanh(a[:,ij[:,0]]-a[:,ij[:,1]])]
 return np.c_[gram(a,'poly'),gram(np.tanh(a),'poly')]
if __name__=='__main__':
 for name,a in [('x',x),('z',z),('b',b),('c',c),('d',d),('cz',c+z),('dz',d+z)]:
  for kind in ['poly','tanh','rich']:
   A=gram(a,kind); errs=[]
   for reg in [.0001,.003,.03,.3,1,3]:
    ls=[]
    for T,V in [(np.arange(52),np.arange(52,72))]+[(np.delete(np.arange(52),vi),vi) for vi in np.array_split(np.random.default_rng(132).permutation(52),4)]:
     m=A[T].mean(0); st=np.maximum(A[T].std(0),1e-9); F=(A[T]-m)/st; P=(A[V]-m)/st@F.T@np.linalg.solve(F@F.T+reg*np.eye(len(T)),y[T]-y[T].mean(0))+y[T].mean(0); E=abs(P-y[V]);ls.append([E[:,0].mean(),E[:,1:].mean()]);
    errs.append((reg,np.round(ls[0],4),np.round(np.mean(ls[1:],axis=0),4)))
   print(name,kind,errs)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe.py
```
- **Container Exit Code**: `0` (`595 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
943, 0.0831])), (1, array([0.0654, 0.0576]), array([0.0956, 0.0903])), (3, array([0.0696, 0.0644]), array([0.1056, 0.1114]))]
cz poly [(0.0001, array([0.4254, 0.6056]), array([0.273 , 0.5805])), (0.003, array([0.4147, 0.5901]), array([0.2782, 0.5617])), (0.03, array([0.3871, 0.5561]), array([0.2822, 0.5238])), (0.3, array([0.3462, 0.5067]), array([0.2622, 0.5071])), (1, array([0.3092, 0.469 ]), array([0.2486, 0.5235])), (3, array([0.2645, 0.4508]), array([0.2507, 0.5377]))]
cz tanh [(0.0001, array([0.5673, 0.9608]), array([0.8514, 1.7891])), (0.003, array([0.5008, 0.885 ]), array([0.5669, 0.8429])), (0.03, array([0.3817, 0.7133]), array([0.4403, 0.6753])), (0.3, array([0.3396, 0.5475]), array([0.3244, 0.5474])), (1, array([0.3077, 0.4842]), array([0.2661, 0.4977])), (3, array([0.279 , 0.4647]), array([0.2422, 0.4882]))]
cz rich [(0.0001, array([1.6407, 1.5832]), array([0.8311, 1.161 ])), (0.003, array([0.6601, 1.0063]), array([0.6463, 1.0332])), (0.03, array([0.4319, 0.7634]), array([0.4705, 0.9746])), (0.3, array([0.3583, 0.5985]), array([0.3072, 0.7791])), (1, array([0.3327, 0.5808]), array([0.2427, 0.6569])), (3, array([0.309 , 0.5744]), array([0.2103, 0.5739]))]
dz poly [(0.0001, array([0.1389, 0.1447]), array([0.174 , 0.1797])), (0.003, array([0.1388, 0.1447]), array([0.1739, 0.1798])), (0.03, array([0.1381, 0.145 ]), array([0.1729, 0.1798])), (0.3, array([0.1319, 0.1467]), array([0.1724, 0.1817])), (1, array([0.1246, 0.147 ]), array([0.1851, 0.191 ])), (3, array([0.1567, 0.1523]), array([0.2287, 0.2114]))]
dz tanh [(0.0001, array([0.2174, 0.3103]), array([0.8064, 0.7024])), (0.003, array([0.2028, 0.2844]), array([0.3782, 0.333 ])), (0.03, array([0.1735, 0.2209]), array([0.2208, 0.2326])), (0.3, array([0.1275, 0.1486]), array([0.1693, 0.2086])), (1, array([0.1097, 0.1419]), array([0.1586, 0.2134])), (3, array([0.1042, 0.1486]), array([0.1673, 0.2214]))]
dz rich [(0.0001, array([0.4435, 0.4998]), array([0.4194, 0.3417])), (0.003, array([0.2153, 0.2155]), array([0.2647, 0.2396])), (0.03, array([0.1683, 0.1899]), array([0.1709, 0.1873])), (0.3, array([0.1172, 0.1336]), array([0.1556, 0.145 ])), (1, array([0.0953, 0.1059]), array([0.1545, 0.1451])), (3, array([0.0923, 0.0945]), array([0.1502, 0.1674]))]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b840909fdd4e5851 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b840909f (directory).
```

### Turn 7 — `execute_in_remote_docker` (`22344 ms` LLM)
- **Rationale**: Extract the cyclic coefficients, inspect the inverse residual tail and public metadata, and check the harness constraints before fitting a replacement readout.
- **Alternatives Weighed**: `Choose a ridge variant from the truncated probe output` (rejected: The calibration coefficients and inverse errors are needed to distinguish incorrect inverse branches from readout misspecification.)

```bash
cd /holdout_task; cat public_asset_index.json; sed -n '1,130p' test_dag_candidate.py; OPENBLAS_NUM_THREADS=1 python3 /tmp/probe.py | grep -E -A20 'squarefree|^inverse' | cut -c1-180; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import sys;sys.path.insert(0,'/tmp');from probe import *
print('train,test',len(x),len(s['test']['x']))
for i in range(4):
 j=(i+1)%4;k=(i+2)%4;l=(i+3)%4
 feats={'1':np.ones(len(u)),'i':u[:,i],'j':u[:,j],'ij':u[:,i]*u[:,j],'ijk':u[:,i]*u[:,j]*u[:,k],'ijl':u[:,i]*u[:,j]*u[:,l],'ikl':u[:,i]*u[:,k]*u[:,l],'jkl':u[:,j]*u[:,k]*u[:,l],'i2':u[:,i]**2,'j2':u[:,j]**2,'i3':u[:,i]**3,'j3':u[:,j]**3,'i2j':u[:,i]**2*u[:,j],'ij2':u[:,i]*u[:,j]**2,'ijkl':np.prod(u,axis=1)};D=np.column_stack(list(feats.values())); w=np.linalg.lstsq(D,v[:,i],rcond=None)[0]; print('A',i,np.std(D@w-v[:,i]),dict(zip(feats,w.round(5))))
print('raw x',np.quantile(x,[0,.1,.5,.9,1],axis=0));print('inverse errors',np.sort(np.linalg.norm(af(z)@W-x,axis=1))[-15:]);
PY
```
- **Container Exit Code**: `0` (`844 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
3.20700e-01
 -3.14000e-03  2.39400e-02  3.33000e-03  2.00100e-02 -4.20000e-04
  3.23100e-01  8.22000e-03 -1.52000e-02  3.87900e-02  2.04100e-02]
3 cyclic 0.044919190282644746 [-0.00547  1.34109  0.33709  0.35254  0.00835 -0.03401 -0.00212 -0.02641]
3 all 0.04091000722177768 [ 6.40000e-04  3.45500e-01  2.35000e-03  2.05400e-02  1.33631e+00
 -2.48400e-02  5.50000e-04  3.40470e-01 -1.55600e-02  1.71100e-02
 -9.34000e-03 -5.41600e-02 -1.22000e-03 -5.55600e-02  1.03500e-02]
inverse 4.440892098500626e-16 [[-1.16393044 -1.22110658 -1.23764976 -1.1621744 ]
 [-1.00192905 -1.03061367 -1.00151149 -1.09781893]
 [ 0.06752467 -0.17181358  0.04138888  0.06490124]
 [ 0.93317185  0.90439876  1.04872067  0.95747022]
 [ 1.15063243  1.17539889  1.17523227  1.08993872]]
train,test 72 36
A 0 7.349698429792996e-05 {'1': 0.08116, 'i': 1.06016, 'j': 0.31999, 'ij': 0.32006, 'ijk': 7e-05, 'ijl': 6e-05, 'ikl': 1e-05, 'jkl': -5e-05, 'i2': 1e-05, 'j2': 1e-05, 'i3': 0.15999, 'j3': 2e-05, 'i2j': -3e-05, 'ij2': -4e-05, 'ijkl': 8e-05}
A 1 8.999320195381565e-05 {'1': 0.10946, 'i': 0.93689, 'j': 0.32001, 'ij': 0.32, 'ijk': 2e-05, 'ijl': 3e-05, 'ikl': 3e-05, 'jkl': 0.0, 'i2': 5e-05, 'j2': 1e-05, 'i3': 0.16004, 'j3': -6e-05, 'i2j': 8e-05, 'ij2': -5e-05, 'ijkl': 0.0001}
A 2 7.438841514702695e-05 {'1': -0.1315, 'i': 1.16899, 'j': 0.31996, 'ij': 0.32002, 'ijk': -6e-05, 'ijl': 0.00013, 'ikl': 0.00012, 'jkl': 7e-05, 'i2': -4e-05, 'j2': -5e-05, 'i3': 0.16006, 'j3': 6e-05, 'i2j': -0.00015, 'ij2': 0.0, 'ijkl': 0.00017}
A 3 8.145404465096764e-05 {'1': -0.01748, 'i': 1.16548, 'j': 0.31999, 'ij': 0.32, 'ijk': 3e-05, 'ijl': -1e-05, 'ikl': 4e-05, 'jkl': 1e-05, 'i2': -0.0, 'j2': -2e-05, 'i3': 0.16, 'j3': -1e-05, 'i2j': -3e-05, 'ij2': -5e-05, 'ijkl': 0.00013}
raw x [[-1.22237873 -1.18302012 -1.66239458 -1.59697191]
 [-0.80956471 -0.8628883  -1.28790839 -1.29690821]
 [-0.02114698 -0.09600121 -0.31565988 -0.05978828]
 [ 1.14424468  1.04116185  1.24949848  1.2873576 ]
 [ 2.01552448  1.94783911  1.95055696  1.78747   ]]
inverse errors [2.22288417e-16 2.22586046e-16 2.28878340e-16 2.30555127e-16
 2.48253415e-16 2.48253415e-16 2.48253415e-16 2.49053676e-16
 2.54384052e-16 2.72301860e-16 2.72743550e-16 3.14018492e-16
 3.35658907e-16 4.96506831e-16 5.11787527e-16]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b840909fdd4e5851 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b840909f (directory).
```

### Turn 8 — `execute_in_remote_docker` (`50269 ms` LLM)
- **Rationale**: Exploit the nearly exact calibration and use inner-fold errors to rank sparse feature dictionaries before compensating for the frozen blend.
- **ATMS NOGOOD DECLARED**: Refuted `Square-free interactions suffice to calibrate panel A.` | Witness: `Adding the diagonal cubic yields a coefficient of approximately 0.160 in all four channels and lowers per-channel RMSE from 0.037–0.043 to below 0.00010.` | Blanket Diagnosis: `The omitted diagonal cubic distorted the inferred latent coordinates even when the numerical inverse had negligible reconstruction error.` | Sublated Pivot: `Use the identified five-term cyclic map and compare regularized readouts in the corrected coordinates.`

```bash
cd /holdout_task; sed -n '1,105p' test_dag_candidate.py; cat > /tmp/readout.py <<'PY'
import numpy as np,pickle
from scipy.optimize import least_squares,minimize_scalar
from sklearn.linear_model import lasso_path, orthogonal_mp
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; Y=tr['traj']; t=s['eval_times']; n=len(x)
def calibrate(p,x):
 u,v=p['u'],p['v']; r=lambda a:np.roll(a,-1,axis=1); F=lambda a:np.stack([np.ones_like(a),a,r(a),a*r(a),a**3],axis=2); D=F(u); keep=np.ones(len(u),bool)
 for _ in range(4):
  W=np.stack([np.linalg.lstsq(D[keep,i],v[keep,i],rcond=None)[0] for i in range(4)]); e=np.linalg.norm(np.einsum('nif,if->ni',D,W)-v,axis=1); keep=e<max(0.001,5*np.median(e))
 z=(x-W[:,0])/W[:,1]; eye=np.eye(4)
 for _ in range(15):
  j=(W[:,1]+W[:,3]*r(z)+3*W[:,4]*z*z)[:,:,None]*eye+(W[:,2]+W[:,3]*z)[:,:,None]*r(eye.T).T
  z-=np.clip(np.linalg.solve(j,(np.einsum('nif,if->ni',F(z),W)-x)[...,None])[...,0],-.5,.5)
 return z
z=calibrate(s['panel_a'],x)
def pm(k,l,f):
 p=s[k]; u=p['u']; D=lambda a,b:np.c_[np.ones(len(a)),f(a,b)]; fit=lambda b:np.linalg.lstsq(D(u,b),p[l],rcond=None)[0]; opt=minimize_scalar(lambda b:np.std(D(u,b)@fit(b)-p[l]),bounds=(.01,3),method='bounded').x; return D(z,opt)@fit(opt)
b=pm('panel_b','v',lambda a,k:a); c=pm('panel_c','v1',lambda a,k:a/(1+k*abs(a))); d=pm('panel_c','v2',lambda a,k:np.c_[a,1/(1+k*a*a)]); bases={'raw':x,'z':z,'b':b,'c':c,'d':d,'cz':c+z,'dz':d+z,'tanh':np.tanh(z)}
def Q(r):return -np.expm1(-t[:,None]*r)
def rates(T):
 return least_squares(lambda r:(Q(r)@np.linalg.lstsq(Q(r),Y[T].T,rcond=None)[0]-Y[T].T).ravel(),[.4,1.8],bounds=(.01,10)).x
ij=list(combinations(range(4),2)); ijk=list(combinations(range(4),3))
def feats(z,level):
 C=[]; names=[]
 def add(a,n):C.append(a);names.append(n)
 for i,u in enumerate(z.T):
  for p in range(1,level+3):add(u**p,f'{i}^{p}')
  if level>0:
   for nm,f in [('tanh',np.tanh),('sat',lambda u:u/(1+abs(u))),('rat',lambda u:1/(1+u*u)),('gauss',lambda u:np.exp(-u*u)),('atan',np.arctan),('sin',np.sin),('cos',np.cos),('sin2',lambda u:np.sin(2*u)),('cos2',lambda u:np.cos(2*u))]:add(f(u),f'{nm}{i}')
 for i,j in ij:
  u,v=z[:,i],z[:,j]
  for p in range(1,level+2):
   for q in range(1,level+3-p):add(u**p*v**q,f'{i}^{p} {j}^{q}')
  if level>0:
   add(u*v/(1+.5*(u*u+v*v)),f'rat{i}{j}');add(np.tanh(u)*np.tanh(v),f'tanhprod{i}{j}')
   for sign in [-1,1]:
    w=u+sign*v
    for k in [.7,1,1.5,2]:
     for nm,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda u:1/(1+u*u))]:add(f(k*w),f'{nm}{k}({i},{sign*j})')
 for idx in ijk:add(np.prod(z[:,idx],axis=1),str(idx))
 return np.column_stack(C),names
splits=[(np.arange(52),np.arange(52,72))]; splits += [(np.setdiff1d(np.arange(52),V),V) for V in np.array_split(np.random.default_rng(612).permutation(52),4)]
if __name__=='__main__':
 print('z bounds',z.min(0),z.max(0),'rates',rates(np.arange(52)), 'target std',y.std(),Y.std(0));
 for k,bs in bases.items():
  for level in (0,1):
   A,names=feats(bs,level); errs=[]; witnesses=[]
   for T,V in splits:
    q=Q(rates(T)); tar=np.c_[y,np.linalg.lstsq(q,Y.T,rcond=None)[0].T]; M=A[T].mean(0); S=A[T].std(0); F=(A[T]-M)/S; E=(A[V]-M)/S; center=tar[T].mean(0); U=tar[T]-center
    regs=np.geomspace(.00003,.1,14); ss=[]
    for i in range(3):
     _,w,_=lasso_path(F,U[:,i],alphas=regs[::-1],max_iter=3500,tol=.0006); o=orthogonal_mp(F,U[:,i],n_nonzero_coefs=16,return_path=True); ridge=np.stack([F.T@np.linalg.solve(F@F.T+r*np.eye(len(T)),U[:,i]) for r in [.001,.01,.1,1,3]],axis=1); ss.append(np.c_[w,o,ridge]);
    m=min(a.shape[1] for a in ss); P=[E@a[:,:m]+center[i] for i,a in enumerate(ss)]; em=np.abs(P[0]-y[V,None]).mean(0); tm=np.abs(P[1][:,None,:]*q[None,:,0,None]+P[2][:,None,:]*q[None,:,1,None]-Y[V,:,None]).mean((0,1)); errs.append([em,tm]); witnesses.append(ss)
   er=np.array(errs); ind=np.argmin(er[1:].mean(0),axis=1); print(k,level,'CV selected',ind,er[0,np.arange(2),ind].round(5),er[1:].mean(0)[np.arange(2),ind].round(5),'public min',er[0].min(1).round(5),flush=True)
   if k in ('z','b','c','d') and level==1:
    for i,W in enumerate(witnesses[0]):
     w=W[:,ind[i>0 and 1 or 0]]; jj=np.argsort(abs(w))[-8:][::-1]; print(i,[(names[j],round(w[j],3)) for j in jj])
PY
OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore python3 /tmp/readout.py
```
- **Container Exit Code**: `0` (`2298 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rat12', 0.072), ('sin0.7(0,-2)', -0.068), ('cos0.7(2,3)', -0.06), ('rat03', 0.048)]
2 [('2^1', -0.161), ('sin1.5(1,2)', -0.156), ('sin2', -0.132), ('(0, 1, 2)', -0.128), ('tanh2(0,2)', -0.117), ('(1, 2, 3)', 0.095), ('sin1.5(2,-3)', -0.087), ('sin1.5(0,-2)', 0.081)]
c 0 CV selected [29  9] [0.07385 0.08095] [0.07387 0.07124] public min [0.05138 0.07983]
c 1 CV selected [13  7] [0.08154 0.08186] [0.09149 0.07776] public min [0.06109 0.06638]
0 [('sin0.7(0,3)', -0.329), ('sin0.7(0,1)', -0.247), ('0^3', -0.154), ('1^3', -0.14), ('sin2(0,-2)', 0.112), ('sin0.7(1,3)', -0.1), ('cos0.7(0,-2)', -0.091), ('0^1 1^2', -0.061)]
1 [('sin0.7(1,2)', 0.221), ('2^3', 0.151), ('0^1 1^1', -0.103), ('0^1 3^2', 0.103), ('cos0.7(0,-2)', -0.091), ('0^1 1^2', 0.085), ('1^3', 0.072), ('1^1 3^2', 0.07)]
2 [('sin0.7(1,-3)', -0.424), ('sin1(0,-2)', 0.243), ('sin0.7(0,-2)', 0.132), ('1^3', -0.094), ('tanh0.7(0,-2)', 0.077), ('sin1(0,-1)', 0.072), ('sin0.7(0,-1)', 0.063), ('cos0.7(1,3)', -0.037)]
d 0 CV selected [ 8 27] [0.04567 0.05574] [0.05982 0.0758 ] public min [0.03836 0.04183]
d 1 CV selected [33 34] [0.03428 0.06812] [0.06314 0.06863] public min [0.03125 0.06513]
0 [('sin2(1,3)', -0.041), ('tanh2(1,3)', -0.04), ('3^3', -0.037), ('tanh1.5(1,3)', -0.036), ('0^2 3^1', -0.034), ('sin1.5(1,3)', -0.034), ('1^2 3^1', 0.032), ('2^3', -0.032)]
1 [('3^3', 0.027), ('2^3', 0.022), ('tanh2(0,1)', -0.015), ('sin0.7(0,-2)', -0.015), ('tanh0.7(0,-2)', -0.014), ('sin1(0,-2)', -0.014), ('tanh1.5(0,1)', -0.014), ('sin0.7(2,3)', 0.014)]
2 [('0^2 1^1', -0.045), ('rat2(0,2)', -0.035), ('tanh2(0,3)', 0.034), ('0^2 2^1', -0.034), ('sat0', 0.034), ('tanh1.5(0,3)', 0.033), ('tanh2(2,-3)', -0.033), ('tanh1(0,3)', 0.032)]
cz 0 CV selected [16  0] [0.21019 0.41075] [0.20394 0.45433] public min [0.16257 0.41075]
cz 1 CV selected [3 0] [0.152   0.45266] [0.2071  0.41595] public min [0.152   0.45266]
dz 0 CV selected [29  2] [0.11526 0.13216] [0.13403 0.16407] public min [0.10095 0.10738]
dz 1 CV selected [18  4] [0.12758 0.11436] [0.15617 0.15151] public min [0.06726 0.1087 ]
tanh 0 CV selected [25 29] [0.0488  0.11041] [0.06663 0.11851] public min [0.04337 0.07566]
tanh 1 CV selected [ 8 13] [0.04342 0.04899] [0.04535 0.04612] public min [0.03343 0.04744]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b840909fdd4e5851 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b840909f (directory).
```

### Turn 9 — `execute_in_remote_docker` (`29607 ms` LLM)
- **Rationale**: Compare separate scalar and trajectory readouts across inner folds, testing whether calibrated views contribute complementary structure before encoding the frozen-blend correction.
- **Alternatives Weighed**: `Use one sparse dictionary for both outputs` (rejected: The scalar and trajectory probes favor different coordinates and regularization.)

```bash
cd /holdout_task; tail -52 test_dag_candidate.py; cat > /tmp/kprobe.py <<'PY'
from readout import *
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
bases.pop('cz');bases.pop('dz'); bases['ct']=np.arctanh(np.clip(c/2,-.99,.99));
# Separate linear, polynomial, and saturated single/pair-coordinate effects.
def kernels(bases):
 for nm,a in bases.items():
  for degree in [1,2,3,4,5,6]:
   g=np.c_[np.ones(len(a)),a]/np.sqrt(2); yield nm+'P'+str(degree),(g@g.T)**degree
  dis=cdist(a,a,'sqeuclidean')
  for scale in [.2,.5,1,2,4,8]:
   yield nm+'R'+str(scale),np.exp(-scale*dis)
  for scale in [.3,1,3]:
   g=(a[:,None,:]-a[None,:,:])**2
   for order in [1,2,3]:
    yield nm+'A'+str((scale,order)),np.mean([np.exp(-scale*g[:,:,idx].sum(2)) for idx in combinations(range(4),order)],axis=0)
  for level in [0,1]:
   f,_=feats(a,level); f=(f-f.mean(0))/np.maximum(f.std(0),1e-8); yield nm+'D'+str(level),f@f.T

def norm(K):return K/max(np.mean(K.diagonal())-K.mean(),1e-10)
kd={nm:norm(K) for nm,K in kernels(bases)}
for keys in [('z','tanh'),('z','b','c','d'),('z','b','c','d','tanh','raw')]:
 for typ in ['P','A','R','D','PAR','PARD']:
  kd['+'.join(keys)+typ]=np.mean([K for nm,K in list(kd.items()) if '+' not in nm and nm.split(typ[0])[0] in keys and any(nm.startswith(key+tp) for key in keys for tp in typ)] or [K for nm,K in list(kd.items()) if '+' not in nm and any(nm.startswith(key+tp) for key in keys for tp in typ)],axis=0)
regs=[1e-7,1e-5,.0002,.002,.015,.07,.25]
Yall=np.c_[y,Y]
def ridge(K,T,V,a):
 F=K[np.ix_(T,T)]; P=K[np.ix_(V,T)]; avg=Yall[T].mean(0); F=F-F.mean(0)-F.mean(1)[:,None]+F.mean();P=P-P.mean(1)[:,None]-K[np.ix_(T,T)].mean(0)+K[np.ix_(T,T)].mean();return P@cho_solve(cho_factor(F+a*np.eye(len(T))),Yall[T]-avg)+avg
for j in [0,1]:
 scores=[]
 for nm,K in kd.items():
  for a in regs:
   ee=[]
   for T,V in splits:
    E=abs(ridge(K,T,V,a)-Yall[V]);ee.append(E[:,0].mean() if j==0 else E[:,1:].mean())
   scores.append((np.mean(ee[1:]),ee[0],max(ee[1:]),nm,a))
 print('HEAD',j,'inner',sorted(scores)[:12]); print('HEAD',j,'outer',sorted(scores,key=lambda p:p[1])[:12])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/kprobe.py
```
- **Container Exit Code**: `0` (`2189 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.030960020728923492, 0.04942011390508563, 'tanhA(0.3, 2)', 0.0002), (0.05263271084047151, 0.032761650119297187, 0.0632558344049248, 'zA(0.3, 2)', 0.015)]
HEAD 1 inner [(0.04386113691853376, 0.04333557704402692, 0.05566796747652009, 'tanhA(1, 2)', 0.002), (0.04510371114195113, 0.03616108758007659, 0.060876020824798927, 'tanhA(0.3, 2)', 0.0002), (0.045494727102230466, 0.04177087672382286, 0.05347634587679347, 'tanhA(1, 2)', 0.0002), (0.045842998765700464, 0.041648453316604335, 0.05318803759683252, 'tanhA(1, 2)', 1e-05), (0.04586199646731547, 0.04163851103649362, 0.05317194931306927, 'tanhA(1, 2)', 1e-07), (0.04855369420411515, 0.03877112650617763, 0.054154626047128326, 'tanhA(0.3, 2)', 0.002), (0.04875309272876867, 0.0533090490202703, 0.06541051567207493, 'tanhA(1, 2)', 0.015), (0.050592977029114444, 0.0394791378520143, 0.0649789592504214, 'tanhA(0.3, 2)', 1e-05), (0.05122773647740944, 0.04195559855527304, 0.06525049847600119, 'tanhA(0.3, 2)', 1e-07), (0.05217341410713376, 0.0539419522243885, 0.0630981387467355, 'dA(0.3, 2)', 0.002), (0.05392870310867745, 0.05147043255434046, 0.06556239759394997, 'dA(0.3, 3)', 0.002), (0.054203255689500726, 0.04685091998011537, 0.061066611893204066, 'dA(0.3, 2)', 0.0002)]
HEAD 1 outer [(0.07712907692198726, 0.032636192160320675, 0.10306955441223242, 'tanhR0.2', 0.0002), (0.0693550941361305, 0.03496547491230965, 0.08581846088813873, 'ctP3', 0.0002), (0.08380866898905902, 0.035390922702625296, 0.0959654383385556, 'ctP3', 1e-05), (0.08575057088808191, 0.03541781509845342, 0.10028717835164383, 'ctP3', 1e-07), (0.07575519870675902, 0.03570257567115585, 0.10739205617735584, 'z+b+c+dD', 1e-07), (0.07576936383465038, 0.03570437951175709, 0.10739465692439265, 'z+b+c+dD', 1e-05), (0.07603661507068332, 0.03574886749112349, 0.10745520585287535, 'z+b+c+dD', 0.0002), (0.06448229141715713, 0.03614881955194384, 0.08805962505286379, 'tanhA(0.3, 3)', 0.0002), (0.04510371114195113, 0.03616108758007659, 0.060876020824798927, 'tanhA(0.3, 2)', 0.0002), (0.06329533650752711, 0.036210401869830475, 0.08225433189472728, 'ctP3', 0.002), (0.0782574675892671, 0.03818612272297417, 0.10869663859294389, 'z+b+c+dD', 0.002), (0.06992777732819395, 0.038553353460947855, 0.0752887180635258, 'ctR0.2', 0.0002)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b840909fdd4e5851 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b840909f (directory).
```

### Turn 10 — `execute_in_remote_docker` (`28399 ms` LLM)
- **Rationale**: Implement the pairwise model that clears both public thresholds in the probes and generalizes better than the full-coordinate alternative. Fit it using supplied training data only and compensate algebraically for the frozen blend.
- **Alternatives Weighed**: `Use the full-coordinate Gaussian kernel` (rejected: Its best trajectory public split score was lower, but its inner-fold mean and worst-fold errors were substantially higher.)

```bash
cd /holdout_task && { printf '"""Robust panel calibration and a pairwise kernel readout."""\nimport numpy as np\nfrom itertools import combinations\n\n'; sed -n '/^def calibrate(/,/^z=calibrate/{ /^z=calibrate/!p; }' /tmp/readout.py; sed -n '/^def _features/,$p' _frozen_nodes/node_00.py | sed 's/^def run_node(/def _sparse_reference(/'; cat <<'PY'


def _readout(z, targets):
    n = len(targets)
    # Each term models a two-coordinate interaction. The average also
    # includes single-coordinate effects without fitting six separate models.
    delta = (z[:, None, :] - z[None, :, :])**2
    kernel = np.mean([np.exp(-0.3*(delta[:, :, i] + delta[:, :, j]))
                      for i, j in combinations(range(z.shape[1]), 2)], axis=0)
    gram = kernel[:n, :n]
    cross = kernel[n:, :n]
    mean = gram.mean(axis=0)
    overall = mean.mean()
    scale = max(1.0-overall, 1e-10)
    gram = gram - mean[None, :] - mean[:, None] + overall
    cross = cross - cross.mean(axis=1, keepdims=True) - mean[None, :] + overall
    offset = targets.mean(axis=0)
    return cross @ np.linalg.solve(gram + (2e-4*scale)*np.eye(n), targets-offset) + offset


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    x = np.vstack([train['x'], study['test']['x']])
    coords = np.tanh(calibrate(study['panel_a'], x))
    desired = _readout(coords, np.column_stack([train['y'], train['traj']]))
    # Express both learned predictions in the fixed sink's blend coordinates.
    baseline = _sparse_reference({'__input__': study})
    scalar = 2.0*desired[:, 0] - baseline['y_hat']
    trajectory = (desired[:, 1:] - 0.35*baseline['state_pred'])/0.65
    return {'panel_pred': np.column_stack([scalar, trajectory])}
PY
} > node_01.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`340 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.030931, "public_cv_traj_mae": 0.036159}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=792c36c032ea7ba4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_792c36c0 (directory).
```

### Turn 11 — `submit_completed_task` (`12146 ms` LLM)
- **Rationale**: The exact saved candidate passed isolated verification for both outputs, execution counts, and the DAG contract. The pairwise readout also had lower worst-fold probe errors than the full-coordinate Gaussian alternative.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/node_01.py` without changing the frozen nodes. The isolated DAG test accepted it: scalar MAE 0.030931 and trajectory MAE 0.036159, both below 0.038. Each node executed once.`

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
