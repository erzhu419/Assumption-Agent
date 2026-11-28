# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m102_b04_batch_05__blind_campaign__reuse_subgraph`

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
| **Turns Used / Max Turns** | `10 / 12` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `291,526` (`281,112` / `114,048` / `10,414`) |
| **Total Episode Wall Time (s)** | `317.51s` |

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

### Turn 1 — `execute_in_remote_docker` (`13549 ms` LLM)
- **Rationale**: Inspect the frozen interface, public evaluator, and archived model before fitting either head.

```bash
cd /holdout_task && cat _dag_spec.json _frozen_nodes/node_00.py test_dag_candidate.py public_asset_index.json; printf '\nARCHIVE\n'; du -ah /tmp/hegel_continual_archive | sort -h | tail -70; python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
def walk(x,k='',d=0):
 if isinstance(x,dict):
  print(k,list(x));
  if d<3:
   for p,v in x.items():walk(v,k+'/'+p,d+1)
 elif isinstance(x,np.ndarray): print(k,x.shape,x.dtype,np.round(x.ravel()[:8],5))
 else: print(k,type(x),str(x)[:250])
walk(x)
PY
```
- **Container Exit Code**: `0` (`343 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
    "/holdout_task/test_dag_candidate.py": {
      "bytes": 8018
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 2825
    }
  },
  "schema": "hegel_public_asset_index_v1"
}
ARCHIVE
4.0K	/tmp/hegel_continual_archive/archive_manifest.json
4.0K	/tmp/hegel_continual_archive/conf_m102_b04_batch_00_solver.py
4.0K	/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
16K	/tmp/hegel_continual_archive
 <class 'list'> [{'study': {'eval_times': array([0.25      , 0.64285714, 1.03571429, 1.42857143, 1.82142857,
       2.21428571, 2.60714286, 3.        ]), 'panel_a': {'u': array([[ 0.36255752,  0.62408045,  0.20745184, -1.16523376],
       [-1.18077794, -0.86276082,
```

### Turn 2 — `execute_in_remote_docker` (`14596 ms` LLM)
- **Rationale**: Read the archived solver and exact fold logic, and inspect the numeric arrays within the public batch before choosing candidate bases.

```bash
cat /tmp/hegel_continual_archive/*; sed -n '70,240p' /holdout_task/test_dag_candidate.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment robust affine calibration'; python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
def walk(x,k=''):
 if isinstance(x,dict):
  for p,v in x.items():walk(v,k+'/'+p)
 elif isinstance(x,(list,tuple)):
  for i,v in enumerate(x):walk(v,k+'/'+str(i))
 elif isinstance(x,np.ndarray): print(k,x.shape,x.dtype,np.round(x.ravel()[:12],5), 'std',np.round(x.std(axis=0),5))
 else: print(k,type(x),str(x)[:250])
walk(x)
PY
```
- **Container Exit Code**: `0` (`434 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
/0/study/eval_times (8,) float64 [0.25    0.64286 1.03571 1.42857 1.82143 2.21429 2.60714 3.     ] std 0.90015
/0/study/panel_a/u (48, 4) float64 [ 0.36256  0.62408  0.20745 -1.16523 -1.18078 -0.86276 -1.18685  0.05114
  0.4911   1.0483   0.67739  0.78404] std [0.66771 0.67972 0.74885 0.74127]
/0/study/panel_a/v (48, 4) float64 [ 0.67954  0.6949  -0.27898 -1.51181 -1.1234  -1.1012  -1.60351 -0.32152
  1.02831  1.60287  1.14681  1.22528] std [1.77399 1.67398 1.86158 1.64464]
/0/study/panel_b/u (24, 4) float64 [-0.72542 -0.96984  1.12825  0.22823  0.33774  0.71309 -0.58801  0.41997
 -0.85509 -0.21624 -0.51078  0.8157 ] std [0.64729 0.71335 0.74528 0.66323]
/0/study/panel_b/v (24, 4) float64 [-0.23955  0.55407 -1.02122  0.85195  0.31708 -0.05347  0.99755  0.06218
  0.90787  0.71906  0.0519   0.29941] std [0.56406 0.64561 0.58922 0.63811]
/0/study/panel_c/u (24, 4) float64 [ 0.73086 -0.53801 -0.03948  0.65782  0.22692 -0.53813 -0.88726 -0.71907
  1.14643 -0.10252 -0.76548  0.22635] std [0.64506 0.64065 0.67627 0.76249]
/0/study/panel_c/v1 (24, 4) float64 [-0.36055 -0.2386   1.21257  0.37735 -0.98203  0.07435  0.09199 -0.73707
 -0.19485  0.07598  1.14566 -0.49855] std [0.62152 0.61629 0.71295 0.64823]
/0/study/panel_c/v2 (24, 4) float64 [ 0.26116 -0.22942  0.01813  0.49079 -0.01332  0.40157 -0.01365  0.13953
  0.02561 -0.04958  0.43421  0.63241] std [0.39136 0.28894 0.28979 0.37348]
/0/study/train/x (72, 4) float64 [-0.43661 -0.71338  0.81808 -0.80733 -0.46447  0.68172  0.38148 -0.29192
  0.39902 -1.06868 -0.23774  0.44015] std [0.69835 0.75262 0.87717 0.83083]
/0/study/train/y (72,) float64 [ 0.14452 -0.03389 -0.70335 -0.78661  0.57582 -0.27511 -0.74612  0.55412
 -2.21916  0.9565   1.52151  1.48621] std 0.98317
/0/study/train/traj (72, 8) float64 [-0.31496 -0.66219 -0.90774 -1.08161 -1.22844 -1.3421  -1.42954 -1.51457
 -0.26997 -0.61306 -0.86152 -1.06398] std [0.43206 0.87432 1.15353 1.34241 1.47945 1.58628 1.67313 1.74439]
/0/study/train/summary (4,) float64 [-0.10209  0.98311 -0.11439  1.74423] std 0.78362
/0/study/test/x (36, 4) float64 [ 0.54625 -1.25834 -0.61258 -0.41258 -0.85352 -0.27512 -0.32082 -0.15003
  1.59905  0.37333 -0.21998  0.95067] std [0.69367 0.63762 0.76257 0.7112 ]
/0/study/telemetry_vec (4,) float64 [0.22 0.22 1.25 1.25] std 0.515
```

### Turn 3 — `execute_in_remote_docker` (`29444 ms` LLM)
- **Rationale**: Panel residuals and trajectory singular values can identify which structure to preserve before spending turns on the editable node.

```bash
sed -n '65,215p' /holdout_task/test_dag_candidate.py; tail -18 /tmp/hegel_continual_archive/*solver.py; cat >/tmp/probe04.py <<'PY'
import pickle,sys,numpy as np
from itertools import combinations_with_replacement as cwr
sys.path.insert(0,'/holdout_task/_frozen_nodes')
from node_00 import run_node
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; out=run_node({'__input__':s}); n=len(s['train']['x']); x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; q=out['hybrid'][:n,36:40]
np.set_printoptions(precision=4,suppress=True,linewidth=160)
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in cwr(range(x.shape[1]),k)])
for p in ['panel_a','panel_b','panel_c']:
 u=s[p]['u'];
 for k,v in s[p].items():
  if k=='u':continue
  f=poly(u,1); fit=np.linalg.lstsq(f,v,rcond=None)[0]; print(p,k,'fit',fit,'res',np.sort(np.linalg.norm(v-f@fit,axis=1)), 'sv',np.linalg.svd(fit[1:],compute_uv=False));
print('calib ranges',q.min(0),q.max(0));print('time sv',np.linalg.svd(z,full_matrices=False)[1]);print('time vec',np.linalg.svd(z,full_matrices=False)[2][:5]);print('summary',s['train']['summary'])
rng=np.random.default_rng(77); perm=rng.permutation(n); folds=np.array_split(perm,4)
def cv(f,b,reg,k=None):
 pred=np.empty_like(b)
 for te in folds:
  tr=np.setdiff1d(np.arange(n),te); g=f[tr]; h=b[tr];
  if k: _,_,v=np.linalg.svd(h,full_matrices=False); h=h@v[:k].T
  coef=np.linalg.solve(g.T@g+reg*np.eye(g.shape[1]),g.T@h)
  pred[te]=f[te]@coef if not k else (f[te]@coef)@v[:k]
 e=np.abs(pred-b);return tuple(round(v,5) for v in [e.mean(),max(e[i].mean() for i in folds)])
for name,f in list((k,v[:n]) for k,v in out.items())+[(f'poly_{d}_{h}',poly(q/np.sqrt(1+h*q*q),d)) for d in (1,2,3,4) for h in (0,.09,.3,.7,1.25)]:
 print(name, 'y', min((cv(f,y,r),r) for r in (.000001,.0001,.001,.007,.03,.1,1)), 'z',min((cv(f,z,r),r) for r in (.000001,.0001,.001,.007,.03,.1,1)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe04.py
```
- **Container Exit Code**: `0` (`524 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.8593]
panel_c v2 fit [[ 0.1717 -0.0995  0.0189 -0.0643]
 [-0.1825 -0.1044  0.2739  0.3715]
 [ 0.1375 -0.0797  0.3372 -0.2856]
 [-0.2218 -0.2907 -0.1362 -0.1394]
 [ 0.3848 -0.2793 -0.0385  0.1099]] res [0.0945 0.0984 0.1033 0.1046 0.1137 0.1146 0.1263 0.127  0.1297 0.1432 0.152  0.1593 0.1624 0.1826 0.1945 0.2006 0.2145 0.2167 0.2188 0.2387 0.2562 0.2567
 0.266  0.3145] sv [0.537  0.4956 0.4412 0.4005]
calib ranges [-1.2309 -1.3609 -1.1502 -1.0589] [1.8522 1.6182 1.6843 1.5409]
time sv [32.4665  1.5167  0.0511  0.0488  0.0449  0.0437  0.0403  0.0362]
time vec [[ 0.1117  0.2273  0.3009  0.3509  0.3871  0.4152  0.4378  0.4564]
 [ 0.3585  0.5199  0.4369  0.2654  0.0615 -0.1336 -0.3124 -0.4697]
 [-0.4579  0.4579 -0.3558  0.0077  0.0269  0.2201  0.4004 -0.4945]
 [ 0.3444  0.37   -0.1269 -0.7763  0.3189 -0.049   0.0497  0.1385]
 [ 0.0314  0.1458 -0.2617  0.3017  0.1793 -0.8195  0.2882  0.1773]]
summary [-0.1021  0.9831 -0.1144  1.7442]
raw y ((0.04345, 0.06237), 0.03) z ((0.05799, 0.07722), 0.03)
aligned y ((0.04807, 0.0715), 0.007) z ((0.06018, 0.07846), 0.03)
hybrid y ((0.04993, 0.08258), 0.1) z ((0.06216, 0.09054), 0.1)
poly_1_0 y ((0.12498, 0.16734), 0.001) z ((0.22109, 0.26414), 1)
poly_1_0.09 y ((0.11617, 0.14298), 0.1) z ((0.20557, 0.23476), 0.1)
poly_1_0.3 y ((0.10436, 0.11314), 1) z ((0.18838, 0.22178), 0.1)
poly_1_0.7 y ((0.10563, 0.12654), 1) z ((0.19692, 0.21591), 0.1)
poly_1_1.25 y ((0.12279, 0.14235), 1) z ((0.21249, 0.21816), 0.1)
poly_2_0 y ((0.06256, 0.08599), 0.1) z ((0.08303, 0.09184), 0.03)
poly_2_0.09 y ((0.04829, 0.0677), 0.1) z ((0.06471, 0.07765), 0.03)
poly_2_0.3 y ((0.03959, 0.05002), 0.1) z ((0.0676, 0.08129), 0.1)
poly_2_0.7 y ((0.06335, 0.08234), 1e-06) z ((0.10104, 0.1246), 0.1)
poly_2_1.25 y ((0.10192, 0.1359), 0.03) z ((0.14039, 0.17594), 0.1)
poly_3_0 y ((0.05127, 0.08722), 0.03) z ((0.06795, 0.10563), 0.03)
poly_3_0.09 y ((0.04267, 0.07126), 0.03) z ((0.05744, 0.0803), 0.03)
poly_3_0.3 y ((0.03394, 0.05345), 0.03) z ((0.04702, 0.05624), 0.007)
poly_3_0.7 y ((0.03108, 0.04269), 0.03) z ((0.04047, 0.05281), 0.03)
poly_3_1.25 y ((0.0439, 0.05598), 0.03) z ((0.04391, 0.05393), 0.03)
poly_4_0 y ((0.0706, 0.11213), 0.001) z ((0.11646, 0.15237), 0.007)
poly_4_0.09 y ((0.05741, 0.09069), 0.001) z ((0.09386, 0.12439), 0.007)
poly_4_0.3 y ((0.03818, 0.0622), 0.001) z ((0.05743, 0.07532), 0.007)
poly_4_0.7 y ((0.02631, 0.03419), 0.007) z ((0.03619, 0.04093), 0.007)
poly_4_1.25 y ((0.03924, 0.05222), 0.007) z ((0.04992, 0.06036), 0.03)
```

### Turn 4 — `execute_in_remote_docker` (`21406 ms` LLM)
- **Rationale**: The trajectory has two dominant modes. Inspect the official split and test whether the non-affine panel residuals expose an exact coordinate map before committing to a generic quartic fit.
- **Alternatives Weighed**: `Promote the quartic fit immediately` (rejected: Its worst trajectory fold is above 0.038, and the panel residuals may reveal a more transferable basis.)

```bash
sed -n '126,220p' /holdout_task/test_dag_candidate.py; cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; OPENBLAS_NUM_THREADS=1 python3 /tmp/probe04.py | head -18; cat >/tmp/structure04.py <<'PY'
exec(open('/tmp/probe04.py').read().split('for p in')[0])
def fit(f,v):
 r=v-f@np.linalg.lstsq(f,v,rcond=None)[0]; return np.sqrt(np.mean(r*r))
for p,k in [('panel_a','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[p]['u'],s[p][k]; g=poly(u,1)
 if p=='panel_a':
  ix=np.arange(len(u));
  for _ in range(12):
   coef=np.linalg.lstsq(g[ix],v[ix],rcond=None)[0]; e=np.linalg.norm(v-g@coef,axis=1); ix=np.argsort(e)[:39]
  print('robust A',coef,np.sort(e)); continue
 for f in [g, np.c_[g,u*u],poly(u,2), np.c_[g,u*u,u**3],np.c_[g,np.sin(u),np.cos(u),np.sin(2*u)]]:
  print(p,k,f.shape,fit(f,v),np.mean([fit(f[np.arange(len(u))!=i],v[np.arange(len(u))!=i]) for i in range(len(u))]));
 print('quadratic coefficients',np.linalg.lstsq(poly(u,2),v,rcond=None)[0]);
print('frozen versus raw corr', np.corrcoef(x.T,q.T)[:4,4:]);print('trajectory modal y correlation',np.corrcoef(np.c_[y,z@np.linalg.svd(z,full_matrices=False)[2][:2].T].T))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/structure04.py
```
- **Container Exit Code**: `0` (`574 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0701  0.0173  0.3198]
 [ 0.4285  1.0163  0.0466 -0.0296]
 [-0.0465  0.3784  1.2337  0.0105]
 [ 0.0191  0.0527  0.3036  1.3671]] [ 0.0924  0.1032  0.113   0.1162  0.1269  0.1283  0.1306  0.1357  0.1486  0.1502  0.1588  0.1697  0.1706  0.1826  0.1849  0.1882  0.1933  0.214   0.2152
  0.2168  0.2228  0.2383  0.2386  0.2435  0.2489  0.261   0.263   0.2832  0.3055  0.3075  0.3082  0.3084  0.3098  0.3252  0.3593  0.3593  0.4306  0.4636
  0.5058  0.509   0.5455  0.5773  0.5887  0.6109  0.6461 11.8812 11.9124 12.5756]
panel_c v1 (24, 5) 0.11181605765577533 0.11119762044311572
panel_c v1 (24, 9) 0.09882086698168763 0.09731937446788975
panel_c v1 (24, 15) 0.08488011780517148 0.08117404986506205
panel_c v1 (24, 13) 0.02598015693556409 0.02526096084596931
panel_c v1 (24, 17) 0.008505720168174857 0.008006621513556963
quadratic coefficients [[-0.1127 -0.01    0.0618 -0.0035]
 [-0.1191 -0.5248  0.6323 -0.4544]
 [ 0.8828  0.0667 -0.2651 -0.403 ]
 [ 0.27   -0.6939 -0.3027  0.5775]
 [ 0.278   0.313   0.6507  0.5204]
 [ 0.2446 -0.0594 -0.0829  0.2109]
 [ 0.0798 -0.0809 -0.0823  0.0668]
 [ 0.1821 -0.0636 -0.2003  0.0483]
 [ 0.1074 -0.0383 -0.0474  0.0777]
 [-0.0294 -0.028  -0.0165 -0.027 ]
 [-0.1553 -0.0014  0.0032 -0.1074]
 [ 0.046  -0.0983 -0.0956  0.0506]
 [ 0.1742 -0.014  -0.0669 -0.0008]
 [ 0.0696 -0.0466 -0.0278  0.0941]
 [ 0.0201 -0.0802  0.06    0.1456]]
panel_c v2 (24, 5) 0.09247307194363433 0.09188174572159007
panel_c v2 (24, 9) 0.004876657812639355 0.004779398249793675
panel_c v2 (24, 15) 0.003594700734575815 0.0034392433156168398
panel_c v2 (24, 13) 0.004464204048052772 0.004286052956990197
panel_c v2 (24, 17) 0.0013537115882609806 0.0012685298844642716
quadratic coefficients [[ 0.1381  0.0832 -0.0513 -0.0687]
 [-0.1679 -0.1514  0.2685  0.3292]
 [ 0.1027 -0.0947  0.3619 -0.2893]
 [-0.2249 -0.3611 -0.1612 -0.155 ]
 [ 0.3794 -0.2576 -0.0692  0.133 ]
 [-0.0879 -0.0624  0.1214  0.1442]
 [-0.0118  0.0009 -0.0039  0.0028]
 [-0.0002  0.0093  0.0041  0.0039]
 [-0.006  -0.0002 -0.0001 -0.0013]
 [ 0.0439 -0.0468  0.1573 -0.1307]
 [ 0.0056  0.0019  0.0009  0.0034]
 [-0.007   0.0031 -0.0011 -0.0001]
 [-0.1029 -0.1562 -0.0721 -0.0738]
 [-0.0093  0.0052 -0.0034 -0.0027]
 [ 0.1685 -0.1137 -0.0283  0.0601]]
frozen versus raw corr [[ 1.      0.2458 -0.1949  0.2493]
 [ 0.2458  1.      0.0153 -0.0596]
 [-0.1949  0.0153  1.      0.3065]
 [ 0.2493 -0.0596  0.3065  1.    ]]
trajectory modal y correlation [[ 1.     -0.7678 -0.5715]
 [-0.7678  1.      0.0159]
 [-0.5715  0.0159  1.    ]]
```

### Turn 5 — `execute_in_remote_docker` (`40024 ms` LLM)
- **Rationale**: Write and test a working sink, then compare coordinate systems and saturation families on the official split and four additional folds to find a larger validation margin.
- **Alternatives Weighed**: `Treat the panel C affine fit as the exact map` (rejected: Its residuals are structured and too large to justify an exact affine interpretation.)

```bash
cat > /holdout_task/node_01.py <<'PY'
"""Fit scalar and low-rank trajectory heads on the frozen calibration."""
from itertools import combinations_with_replacement
import numpy as np


def _basis(x, degree):
    return np.column_stack([np.ones(len(x))] + [
        np.prod(x[:, j], axis=1) for k in range(1, degree + 1)
        for j in combinations_with_replacement(range(x.shape[1]), k)
    ])


def _design(q, scale):
    return _basis(q / np.sqrt(1.0 + scale * q * q), 4)


def _predict(f, target, reg):
    n = len(target)
    train = f[:n]
    gram = train @ train.T
    gram.flat[::n + 1] += reg
    return f[n:] @ train.T @ np.linalg.solve(gram, target)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    features = inputs['node_00']
    q = features['hybrid'][:, features['aligned'].shape[1] + 1:features['aligned'].shape[1] + 5]
    targets = study['train']
    traj = targets['traj']
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    design = _design(q, .7)
    y = _predict(design, targets['y'], .007)
    state = _predict(design, traj @ modes.T, .007) @ modes
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([
        y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
cat >/tmp/search04.py <<'PY'
exec(open('/tmp/probe04.py').read().split('for p in')[0])
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
ix=np.arange(n); folds=np.array_split(np.random.default_rng(77).permutation(n),4)+[ix[-20:]]
def evaluate(kernel, regs=(.0001,.001,.007,.03,.1)):
 ys=[]; zs=[]
 for te in folds:
  tr=np.setdiff1d(ix,te); l,v=eigh(kernel[np.ix_(tr,tr)]); lhs=(kernel[np.ix_(te,tr)]@v); m=np.linalg.svd(z[tr],full_matrices=False)[2][:2]; rhs=v.T@np.c_[y[tr], (z[tr]@m.T)@m]; p=np.einsum('ij,rj,jk->rik',lhs,1/(l[None,:]+np.asarray(regs)[:,None]),rhs)
  ys.append(np.abs(p[:,:,0]-y[te]).mean(axis=1)); zs.append(np.abs(p[:,:,1:]-z[te]).mean(axis=(1,2)))
 a,b=np.array(ys),np.array(zs); return [(round(max(e.max(axis=0)[i],0),5),round(e[-1,i],5),round(e[:-1,i].mean(),5),regs[i]) for e in (a,b) for i in [np.argmin(e.max(axis=0))]]
f=np.c_[np.ones(48),s['panel_a']['u']]; v=s['panel_a']['v']; keep=np.arange(len(v));
for _ in range(10):
 fit=np.linalg.lstsq(f[keep],v[keep],rcond=None)[0]; keep=np.argsort(np.linalg.norm(v-f@fit,axis=1))[:39]
qc=(x-fit[0])@np.linalg.inv(fit[1:]); b=np.linalg.lstsq(poly(s['panel_b']['u'],1),s['panel_b']['v'],rcond=None)[0]; c=np.linalg.lstsq(poly(s['panel_c']['u'],1),s['panel_c']['v1'],rcond=None)[0]; U,S,V=np.linalg.svd(c[1:]); c[1:]=U@V
coords={'q':q,'x':x,'fullA':qc,'B_x':poly(x,1)@b,'B_q':poly(q,1)@b,'B_fullA':poly(qc,1)@b,'C_x':poly(x,1)@c,'C_q':poly(q,1)@c,'C_fullA':poly(qc,1)@c}
res=[]
for nm,u in coords.items():
 for h in (0,.22,.45,.7,1.,1.25,1.65):
  for typ in ('sqrt','tanh','asinh') if h else ('sqrt',):
   w={'sqrt':lambda:u/np.sqrt(1+h*u*u),'tanh':lambda:np.tanh(np.sqrt(h)*u)/np.sqrt(h),'asinh':lambda:np.arcsinh(np.sqrt(h)*u)/np.sqrt(h)}[typ]()
   for d in (2,3,4,5):
    g=poly(w,d); k=g@g.T; r=evaluate(k); res.append((nm,h,typ,d,*r))
   dist=cdist(w,w,'sqeuclidean')
   for lam in (.15,.5,1.):
    r=evaluate(np.exp(-lam*dist)); res.append((nm,h,typ,'rbf'+str(lam),*r))
for j in (-2,-1):
 print('HEAD',j,flush=True)
 for row in sorted(res,key=lambda p:p[j])[:16]:print(row,flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/search04.py
```
- **Container Exit Code**: `0` (`3788 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
5, (0.04223, 0.02262, 0.03163, 0.007), (0.05168, 0.04926, 0.04233, 0.001))
('q', 0.7, 'sqrt', 3, (0.04269, 0.02887, 0.03108, 0.03), (0.05118, 0.0432, 0.04156, 0.007))
('q', 1.65, 'asinh', 4, (0.0427, 0.03133, 0.03183, 0.001), (0.06926, 0.04645, 0.05507, 0.007))
('q', 1.0, 'sqrt', 4, (0.04394, 0.03247, 0.0327, 0.007), (0.0493, 0.04925, 0.04246, 0.03))
('x', 1.0, 'asinh', 3, (0.04478, 0.04404, 0.02977, 0.03), (0.04945, 0.04643, 0.04459, 0.007))
('q', 1.25, 'sqrt', 5, (0.04493, 0.02925, 0.0339, 0.001), (0.04786, 0.04731, 0.04329, 0.007))
('fullA', 0.7, 'tanh', 5, (0.04528, 0.03775, 0.03757, 0.0001), (0.05641, 0.05641, 0.04882, 0.001))
('q', 1.0, 'tanh', 4, (0.04539, 0.02726, 0.03264, 0.007), (0.04916, 0.04652, 0.04166, 0.007))
HEAD -1
('q', 1.0, 'sqrt', 5, (0.03701, 0.02031, 0.02827, 0.001), (0.04225, 0.04225, 0.03683, 0.001))
('q', 0.7, 'sqrt', 4, (0.03419, 0.02032, 0.02631, 0.007), (0.04267, 0.04267, 0.0353, 0.007))
('q', 0.7, 'tanh', 4, (0.0466, 0.02181, 0.02823, 0.007), (0.04448, 0.04365, 0.0376, 0.007))
('x', 1.25, 'asinh', 3, (0.04177, 0.04174, 0.02839, 0.03), (0.04568, 0.04385, 0.04293, 0.007))
('q', 0.7, 'sqrt', 5, (0.03589, 0.02606, 0.02815, 0.007), (0.04638, 0.04638, 0.03462, 0.0001))
('q', 0.45, 'sqrt', 3, (0.04698, 0.045, 0.03214, 0.007), (0.04677, 0.04618, 0.04225, 0.007))
('q', 1.65, 'asinh', 3, (0.04688, 0.04688, 0.03082, 0.007), (0.04707, 0.04247, 0.04237, 0.007))
('fullA', 0.7, 'sqrt', 5, (0.03795, 0.02576, 0.03142, 0.0001), (0.0478, 0.04361, 0.04369, 0.0001))
('q', 1.25, 'sqrt', 5, (0.04493, 0.02925, 0.0339, 0.001), (0.04786, 0.04731, 0.04329, 0.007))
('x', 1.65, 'asinh', 3, (0.039, 0.039, 0.02763, 0.03), (0.04788, 0.04773, 0.04209, 0.001))
('q', 0.45, 'sqrt', 4, (0.05128, 0.03062, 0.02989, 0.001), (0.04834, 0.04227, 0.04011, 0.007))
('q', 1.0, 'tanh', 4, (0.04539, 0.02726, 0.03264, 0.007), (0.04916, 0.04652, 0.04166, 0.007))
('q', 1.0, 'sqrt', 4, (0.04394, 0.03247, 0.0327, 0.007), (0.0493, 0.04925, 0.04246, 0.03))
('x', 0.45, 'sqrt', 3, (0.05085, 0.03449, 0.03112, 0.03), (0.04943, 0.04481, 0.04236, 0.007))
('x', 1.0, 'asinh', 3, (0.04478, 0.04404, 0.02977, 0.03), (0.04945, 0.04643, 0.04459, 0.007))
('x', 0.7, 'sqrt', 4, (0.04651, 0.02503, 0.03244, 0.03), (0.04965, 0.0478, 0.04255, 0.03))

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f95bdd21520921a2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f95bdd21 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`25595 ms` LLM)
- **Rationale**: The scalar head passes, so measure whether anisotropic regularization can fix the trajectory head without giving up the calibrated coordinates or the temporal rank.
- **Alternatives Weighed**: `Use the full affine panel-A inverse` (rejected: It did not improve the best trajectory worst-fold MAE in the coordinate comparison.)
- **ATMS NOGOOD DECLARED**: Refuted `A shared isotropic quartic basis with the same ridge strength suffices for both calibrated prediction heads.` | Witness: `The official holdout trajectory MAE is 0.042672 versus the 0.038 limit, while the scalar head reaches 0.020321 and the trajectory still has two clear modes.` | Blanket Diagnosis: `The current spatial basis and shrinkage are insufficient for the trajectory amplitudes; the observed temporal rank is not the main bottleneck.` | Sublated Pivot: `Keep calibration and rank two; test degree- and interaction-weighted regressors separately for the two heads and inspect calibration residuals.`

```bash
cat >/tmp/refine04.py <<'PY'
exec(open('/tmp/search04.py').read().split('f=np.c_')[0])
a=s['panel_a']; u,v=a['u'],a['v']; lin=poly(u,1); good=np.argsort(np.linalg.norm(v-lin@np.linalg.lstsq(lin,v,rcond=None)[0],axis=1))[:-3]; u,v=u[good],v[good]; lin=poly(u,1)
print('panel A leave-one-out, residual by coordinate')
for name,F in [('linear',lin),('quad',poly(u,2)),('cubic',poly(u,3)),('addpoly',np.c_[lin,u*u,u**3,u**4])]+[(f'sin{w}',np.c_[lin,np.sin(w*u),np.cos(w*u)]) for w in (1,2,3,4,5,6)]:
 B=np.linalg.pinv(F); r=(v-F@(B@v))/(1-np.einsum('ij,ji->i',F,B))[:,None];print(name,np.sqrt(np.mean(r*r,axis=0)))
print('y vs traj',np.sqrt(np.mean((y-np.c_[np.ones(n),z]@np.linalg.lstsq(np.c_[np.ones(n),z],y,rcond=None)[0])**2)))
regs=tuple(np.geomspace(.0003,.1,12)); rs=[]
for h in np.linspace(.35,1.1,16):
 w=q/np.sqrt(1+h*q*q)
 for degree in (3,4,5,6):
  fx=poly(w,degree)
  ex=list(j for d in range(1,degree+1) for j in cwr(range(4),d)); num=np.array([0]+[len(set(j)) for j in ex]); deg=np.array([0]+[len(j) for j in ex]);
  for interact in (.08,.25,.5,1,2,4):
   for decay in (.5,1,2):
    X=fx*interact**(np.maximum(num-1,0)/2)*decay**(np.maximum(deg-2,0)/2); K=X@X.T; e=evaluate(K,regs);rs.append((h,degree,interact,decay,*e))
print('top scalar');print(*sorted(rs,key=lambda r:r[-2])[:8],sep='\n');print('top trajectory');print(*sorted(rs,key=lambda r:r[-1])[:15],sep='\n')
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/refine04.py
```
- **Container Exit Code**: `0` (`3958 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(0.03422, 0.03324, 0.02912, 0.0014627802713671006))
(0.6000000000000001, 5, 0.25, 0.5, (0.02424, 0.02259, 0.02101, 0.0003), (0.03591, 0.03519, 0.02875, 0.000862633820503181))
(0.65, 5, 0.25, 0.5, (0.02431, 0.02216, 0.02073, 0.0003), (0.03514, 0.03481, 0.02847, 0.000862633820503181))
(0.55, 4, 0.25, 0.5, (0.02447, 0.02385, 0.02008, 0.0005087142087173843), (0.03731, 0.03731, 0.03159, 0.000862633820503181))
top trajectory
(0.8500000000000001, 6, 0.25, 0.5, (0.02757, 0.02066, 0.02201, 0.0003), (0.03279, 0.03279, 0.02855, 0.0014627802713671006))
(0.9000000000000001, 6, 0.25, 0.5, (0.0296, 0.02133, 0.02305, 0.0003), (0.03284, 0.03284, 0.02861, 0.0014627802713671006))
(0.8, 6, 0.25, 0.5, (0.02583, 0.02041, 0.02127, 0.0003), (0.03302, 0.03287, 0.02864, 0.0014627802713671006))
(0.9500000000000001, 6, 0.25, 0.5, (0.03181, 0.02201, 0.02419, 0.0003), (0.03306, 0.03306, 0.02885, 0.0014627802713671006))
(0.8, 4, 0.25, 0.5, (0.03076, 0.02384, 0.02385, 0.0005087142087173843), (0.03319, 0.03317, 0.02966, 0.0014627802713671006))
(0.8500000000000001, 5, 0.25, 0.5, (0.02878, 0.02124, 0.02261, 0.0003), (0.03345, 0.03345, 0.02854, 0.0014627802713671006))
(1.0, 6, 0.25, 0.5, (0.03411, 0.02278, 0.02536, 0.0003), (0.03347, 0.03347, 0.02945, 0.0014627802713671006))
(0.8500000000000001, 4, 0.25, 0.5, (0.03215, 0.02458, 0.02528, 0.0005087142087173843), (0.03348, 0.03348, 0.02981, 0.0014627802713671006))
(0.9000000000000001, 5, 0.25, 0.5, (0.03076, 0.02177, 0.02357, 0.0003), (0.0335, 0.0335, 0.02863, 0.0014627802713671006))
(0.75, 4, 0.25, 0.5, (0.02943, 0.02309, 0.02245, 0.0005087142087173843), (0.03352, 0.03308, 0.02976, 0.0014627802713671006))
(0.8, 5, 0.25, 0.5, (0.02683, 0.02117, 0.02185, 0.0003), (0.03354, 0.03354, 0.02861, 0.0014627802713671006))
(0.75, 6, 0.25, 0.5, (0.02505, 0.02072, 0.02057, 0.0003), (0.03356, 0.03301, 0.02884, 0.0014627802713671006))
(0.9500000000000001, 5, 0.25, 0.5, (0.03282, 0.02245, 0.02463, 0.0003), (0.03373, 0.03373, 0.029, 0.0014627802713671006))
(0.65, 6, 0.5, 0.5, (0.02629, 0.0204, 0.0204, 0.002480457027586385), (0.03386, 0.03381, 0.02859, 0.002480457027586385))
(0.7000000000000001, 6, 0.5, 0.5, (0.02684, 0.02021, 0.02074, 0.002480457027586385), (0.03393, 0.03393, 0.02853, 0.002480457027586385))

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f95bdd21520921a2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f95bdd21 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`32213 ms` LLM)
- **Rationale**: Promote separate weighted fits that put both exploratory worst-fold errors below tolerance, then verify the actual sink in a clean process and test fresh splits, time horizons, and high-norm rows.
- **Alternatives Weighed**: `Replace the frozen coordinates with a nonlinear panel-A inverse` (rejected: The panel admits a nearly exact cubic forward fit, but a stable inverse and improved prediction have not been demonstrated.)

```bash
python3 - <<'PY'
p='/holdout_task/node_01.py'
s=open(p).read(); a=s.index('def _basis'); b=s.index('def _predict'); s=s[:a]+'''def _design(q, scale):
    # Saturation controls extrapolation; penalize high-degree mixed terms more.
    x = q / np.sqrt(1.0 + scale * q * q)
    cols = [np.ones(len(x))]
    for degree in range(1, 7):
        for j in combinations_with_replacement(range(x.shape[1]), degree):
            weight = 0.5 ** (len(set(j)) - 1 + 0.5 * max(degree - 2, 0))
            cols.append(weight * np.prod(x[:, j], axis=1))
    return np.column_stack(cols)


'''+s[b:]; s=s.replace('    design = _design(q, .7)\n    y = _predict(design, targets[\'y\'], .007)\n    state = _predict(design, traj @ modes.T, .007) @ modes','    y = _predict(_design(q, 0.60), targets[\'y\'], 0.0003)\n    state = _predict(_design(q, 0.85), traj @ modes.T, 0.00146) @ modes'); open(p,'w').write(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle,runpy,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['y']); ix=np.arange(n)
up=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']; sink=runpy.run_path('/holdout_task/node_01.py')['run_node']
def check(name,folds):
 errors=[]
 for te in folds:
  it=np.setdiff1d(ix,te); ts={**s,'train':{k:tr[k][it] for k in ('x','y','traj')},'test':{'x':tr['x'][te]}}
  p=sink({'__input__':ts,'node_00':up({'__input__':ts})}); y=p['y_hat']; z=p['state_pred']; assert np.all(np.isfinite(z)) and z.shape==tr['traj'][te].shape
  np.testing.assert_allclose(p['diagnostics_vec'],[y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()]); errors.append([np.abs(y-tr['y'][te]).mean(),np.abs(z-tr['traj'][te]).mean(),*np.abs(z-tr['traj'][te]).mean(0)])
 print(name,'worst',np.max(errors,axis=0),'pooled',np.mean(errors,axis=0))
check('original auxiliary',np.array_split(np.random.default_rng(77).permutation(n),4))
check('fresh random', [te for seed in (118,231,408,511) for te in np.array_split(np.random.default_rng(seed).permutation(n),4)])
check('ordered batches',np.array_split(ix,4))
check('tail norm stress',[np.argsort(np.linalg.norm(tr['x'],axis=1))[-12:]])
PY
```
- **Container Exit Code**: `0` (`637 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.022041, "public_cv_traj_mae": 0.032792}, "exec_counts": {"node_00": 1, "node_01": 1}}
original auxiliary worst [0.02371587 0.03260655 0.01524525 0.0239116  0.03045595 0.03350135
 0.03540417 0.04032949 0.04136755 0.0432718 ] pooled [0.02025397 0.02854202 0.01144594 0.02033505 0.0264085  0.02971897
 0.03215922 0.03476786 0.03586828 0.03763236]
fresh random worst [0.0251952  0.03375769 0.01422976 0.02462588 0.03169028 0.0368208
 0.03840893 0.04190826 0.04197317 0.04387658] pooled [0.02004981 0.02769206 0.01166522 0.02039633 0.02574307 0.0287532
 0.03140527 0.03363667 0.03420708 0.03572968]
ordered batches worst [0.02371938 0.0357921  0.01542814 0.02767508 0.03179673 0.03751424
 0.04156892 0.04255911 0.04408856 0.0474558 ] pooled [0.02126404 0.03023582 0.01219852 0.02285395 0.02777298 0.03137978
 0.03429179 0.03620932 0.03762468 0.03955556]
tail norm stress worst [0.05086669 0.04562307 0.01998232 0.03618814 0.04640558 0.04623453
 0.05024761 0.05502227 0.05802063 0.05288349] pooled [0.05086669 0.04562307 0.01998232 0.03618814 0.04640558 0.04623453
 0.05024761 0.05502227 0.05802063 0.05288349]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=96a5b0680e0c23d6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_96a5b068 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`29057 ms` LLM)
- **Rationale**: Keep the verified candidate unchanged while testing whether cubic inversion improves extrapolation. Compare it against the existing coordinates across ordinary folds and the high-norm split before promoting any change.
- **Alternatives Weighed**: `Submit the currently accepted sink` (rejected: The official contract passes, but both heads exceed 0.038 on the high-norm stress split. A nearly exact cubic forward calibration offers a concrete alternative to test without changing the accepted artifact.)

```bash
cat /holdout_task/_frozen_nodes/node_00.py /tmp/hegel_continual_archive/*solver.py; cat >/tmp/inverse04.py <<'PY'
exec(open('/tmp/search04.py').read().split('f=np.c_')[0])
from scipy.optimize import least_squares
from itertools import product
u=s['panel_a']['u']; v=s['panel_a']['v']; g=poly(u,1); B=np.linalg.lstsq(g,v,rcond=None)[0]; good=np.argsort(np.linalg.norm(v-g@B,axis=1))[:-3]; F=poly(u,3); A=np.linalg.lstsq(F[good],v[good],rcond=None)[0]; terms=[()]+[j for d in (1,2,3) for j in cwr(range(4),d)]; print('panel A coefficients',[(j,np.round(a,4).tolist()) for j,a in zip(terms,A) if np.max(np.abs(a))>.002]); A[np.abs(A)<.002]=0
for j in range(4):
 mask=A[:,j]!=0; A[mask,j]=np.linalg.lstsq(F[good][:,mask],v[good,j],rcond=None)[0]
B=A[1:5]; base=(x-A[0])@np.linalg.inv(B); w=[]; errs=[]
for p,b in zip(x,base):
 fun=lambda t:poly(t[None,:],3)[0]@A-p
 sol=least_squares(fun,b,gtol=1e-11); w.append(sol.x);errs.append(np.linalg.norm(sol.fun))
w=np.array(w); print('inverse residual',np.max(errs),'coordinate ranges',w.min(0),w.max(0),'sd',w.std(0),'corr',np.corrcoef(w.T,q.T)[:4,4:]); print('time sv',np.linalg.svd(z,compute_uv=False));
folds.extend([np.argsort(np.linalg.norm(x,axis=1))[-12:]]+list(np.array_split(np.random.default_rng(118).permutation(n),4)))
regs=(.000001,.00003,.0003,.0015,.007,.03); ex=[]
for name,hq in [('inverse',w),('frozen',q),('raw',x)]:
 for h,typ in product((0,.22,.55,.85,1.25,1.8),('sqrt','den','tanh','sin')):
  if h==0 and typ!='sqrt':continue
  f={'sqrt':lambda:hq/np.sqrt(1+h*hq*hq),'den':lambda:hq/(1+h*hq*hq),'tanh':lambda:np.tanh(np.sqrt(h)*hq)/np.sqrt(h),'sin':lambda:np.sin(np.sqrt(h)*hq)/np.sqrt(h)}[typ]()
  for d in range(1,7):
   g=poly(f,d); js=[()]+[j for D in range(1,d+1) for j in cwr(range(4),D)];
   for mixed in (.25,.5,1):
    weights=np.array([mixed**max(0,len(set(j))-1)*.5**(.5*max(0,len(j)-2)) for j in js]); k=(g*weights)@(g*weights).T; score=evaluate(k,regs); ex.append((name,h,typ,d,mixed,*score))
for i in (-2,-1):
 print('BEST',i); print(*sorted(ex,key=lambda r:r[i])[:10],sep='\n')
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/inverse04.py
```
- **Container Exit Code**: `0` (`7660 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.0803  0.9543  0.0674]
 [ 0.0151 -0.0818  0.3198  0.9528]]
time sv [32.4665  1.5167  0.0511  0.0488  0.0449  0.0437  0.0403  0.0362]
BEST -2
('inverse', 0, 'sqrt', 2, 0.5, (0.00576, 0.00433, 0.00499, 0.007), (0.00479, 0.00462, 0.00437, 1e-06))
('inverse', 0, 'sqrt', 2, 1, (0.00582, 0.0043, 0.00502, 0.007), (0.00479, 0.00462, 0.00437, 1e-06))
('inverse', 0, 'sqrt', 2, 0.25, (0.00588, 0.00426, 0.00503, 0.0015), (0.00479, 0.00462, 0.00437, 1e-06))
('inverse', 0.22, 'tanh', 4, 0.25, (0.01092, 0.00565, 0.0081, 0.0015), (0.00762, 0.00549, 0.00619, 0.0015))
('inverse', 0.22, 'sqrt', 4, 0.25, (0.01124, 0.0053, 0.00802, 0.0015), (0.00742, 0.00551, 0.00637, 0.0015))
('inverse', 0.55, 'sin', 4, 0.25, (0.01202, 0.00544, 0.00905, 0.0015), (0.00904, 0.00595, 0.0078, 0.007))
('inverse', 0.22, 'sin', 4, 0.25, (0.01206, 0.0064, 0.00894, 0.0015), (0.00934, 0.00581, 0.00644, 0.0015))
('inverse', 0.22, 'sin', 3, 0.25, (0.0123, 0.0067, 0.00865, 0.007), (0.0061, 0.00505, 0.00515, 0.0015))
('inverse', 0, 'sqrt', 3, 0.25, (0.01248, 0.00594, 0.00843, 0.0015), (0.00667, 0.00533, 0.00531, 0.0003))
('inverse', 0.55, 'sin', 6, 0.25, (0.0125, 0.00604, 0.00884, 0.0015), (0.00805, 0.00563, 0.00669, 0.0015))
BEST -1
('inverse', 0, 'sqrt', 1, 0.25, (0.0976, 0.04399, 0.05675, 0.03), (0.00463, 0.00452, 0.00416, 0.0015))
('inverse', 0, 'sqrt', 1, 0.5, (0.0976, 0.04399, 0.05675, 0.03), (0.00463, 0.00452, 0.00416, 0.0015))
('inverse', 0, 'sqrt', 1, 1, (0.0976, 0.04399, 0.05675, 0.03), (0.00463, 0.00452, 0.00416, 0.0015))
('inverse', 0, 'sqrt', 2, 0.25, (0.00588, 0.00426, 0.00503, 0.0015), (0.00479, 0.00462, 0.00437, 1e-06))
('inverse', 0, 'sqrt', 2, 0.5, (0.00576, 0.00433, 0.00499, 0.007), (0.00479, 0.00462, 0.00437, 1e-06))
('inverse', 0, 'sqrt', 2, 1, (0.00582, 0.0043, 0.00502, 0.007), (0.00479, 0.00462, 0.00437, 1e-06))
('inverse', 0.22, 'tanh', 3, 0.25, (0.01256, 0.00631, 0.00871, 0.0015), (0.00574, 0.00508, 0.00538, 0.0015))
('inverse', 0.22, 'sin', 3, 0.25, (0.0123, 0.0067, 0.00865, 0.007), (0.0061, 0.00505, 0.00515, 0.0015))
('inverse', 0.22, 'sin', 3, 0.5, (0.01363, 0.00725, 0.00988, 0.007), (0.00644, 0.00536, 0.00544, 0.0003))
('inverse', 0.22, 'sqrt', 3, 0.25, (0.01329, 0.00681, 0.0094, 0.0015), (0.00645, 0.00537, 0.00592, 0.0015))

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=96a5b0680e0c23d6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_96a5b068 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`58222 ms` LLM)
- **Rationale**: The panel independently identifies the missing calibration terms. Verify that the exact artifact preserves the DAG contract and that the improvement survives fresh folds, coordinate extremes, per-horizon checks, and panel jackknifing.
- **Alternatives Weighed**: `Keep the accepted weighted sink unchanged` (rejected: The cubic inverse has a much larger exploratory validation margin, including on high-norm inputs.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine calibration plus saturation adequately represents the measurement transformation on extreme inputs.` | Witness: `Panel-A leave-one-out residual RMS falls from 0.139–0.210 per coordinate for the affine map to 0.0003–0.0004 for the cubic map. Under cubic inversion, exploratory worst-fold prediction MAEs fall below 0.006 for both heads, versus 0.05087 and 0.04562 on the affine model's high-norm split.` | Blanket Diagnosis: `The affine representation omits cyclic linear and bilinear coupling and a coordinate-wise cubic term, forcing the prediction heads to approximate calibration error.` | Sublated Pivot: `Use the frozen coordinates to initialize cubic inversion, retain the verified weighted model as a fallback when the panel does not support cubic inversion, and fit simpler heads in the recovered coordinates.`

```bash
set -e
cp /holdout_task/node_01.py /tmp/weighted04_fallback.py
cat > /holdout_task/node_01.py <<'PY'
"""Invert the panel calibration, then fit separate scalar and trajectory heads."""
from itertools import combinations_with_replacement
import numpy as np


def _calibration(u, v, j):
    # The calibration couples each coordinate to its cyclic successor.
    nxt = np.roll(u, -1, axis=1)
    f = np.stack((np.ones_like(u), u, nxt, u * nxt, u**3), axis=-1)
    def fit(keep):
        return np.column_stack([np.linalg.lstsq(f[keep, k], v[keep, k], rcond=None)[0] for k in range(u.shape[1])])
    def residual(a):
        return np.linalg.norm(np.einsum('nkp,pk->nk', f, a) - v, axis=1)
    n = len(u)
    m = max(5, int(.82 * n))
    best = None
    for keep in (np.arange(n), j[:m]):
        for _ in range(10):
            a = fit(keep)
            new = np.sort(np.argsort(residual(a))[:m])
            if np.array_equal(new, keep):
                break
            keep = new
        e = residual(a)
        score = np.mean(np.sort(e)[:m]**2)
        if best is None or score < best[0]:
            best = score, a
    a = best[1]
    return a if np.median(residual(a)) <= .015 * max(np.std(v), 1e-8) else None


def _inverse(x, q, u, v):
    f = np.column_stack((np.ones(len(u)), u))
    affine = np.linalg.lstsq(f, v, rcond=None)[0]
    order = np.argsort(np.linalg.norm(f @ affine - v, axis=1))
    a = _calibration(u, v, order)
    if a is None:
        return None
    d = x.shape[1]
    k = np.arange(d)
    nxt = (k + 1) % d
    def error(w):
        z = w[:, nxt]
        return a[0] + a[1]*w + a[2]*z + a[3]*w*z + a[4]*w**3 - x
    jac = np.zeros((len(x), d, d))
    try:
        jac[:, k, k] = a[1]
        jac[:, k, nxt] = a[2]
        w = np.linalg.solve(jac[0], (x-a[0]).T).T
        use_q = np.linalg.norm(error(q), axis=1) < np.linalg.norm(error(w), axis=1)
        w[use_q] = q[use_q]
        for _ in range(24):
            r = error(w)
            if np.max(np.abs(r)) <= 1e-10:
                return w
            jac[:, k, k] = a[1] + a[3]*w[:, nxt] + 3*a[4]*w*w
            jac[:, k, nxt] = a[2] + a[3]*w
            step = np.linalg.solve(jac, r[..., None])[..., 0]
            norm = np.linalg.norm(r, axis=1)
            for j in range(12):
                trial = w - step * (0.5**j)
                good = np.linalg.norm(error(trial), axis=1) < norm
                if j == 0:
                    update = w.copy()
                    pending = np.ones(len(w), dtype=bool)
                good &= pending
                update[good] = trial[good]
                pending[good] = False
                if not pending.any():
                    break
            w = update
        if np.max(np.abs(error(w))) <= 1e-7:
            return w
    except np.linalg.LinAlgError:
        pass
    return None


def _basis(w, degree, fallback=False):
    cols = [np.ones(len(w))]
    for d in range(1, degree+1):
        for j in combinations_with_replacement(range(w.shape[1]), d):
            weight = 0.5**(len(set(j))-1 + 0.5*max(d-2, 0)) if fallback else 1.0
            cols.append(weight * np.prod(w[:, j], axis=1))
    return np.column_stack(cols)


def _ridge(f, y, penalty):
    n = len(y)
    a, b = f[:n], f[n:]
    if a.shape[1] <= n:
        return b @ np.linalg.solve(a.T @ a + penalty*np.eye(a.shape[1]), a.T @ y)
    return (b @ a.T) @ np.linalg.solve(a @ a.T + penalty*np.eye(n), y)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    upstream = inputs['node_00']
    train, test = study['train'], study['test']
    x = np.vstack((train['x'], test['x']))
    start = upstream['aligned'].shape[1] + 1
    q = upstream['hybrid'][:, start:start+x.shape[1]]
    panel = study['panel_a']
    w = _inverse(x, q, panel['u'], panel['v'])
    traj = train['traj']
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    if w is None:
        y = _ridge(_basis(q / np.sqrt(1 + .60*q*q), 6, True), train['y'], .0003)
        state = _ridge(_basis(q / np.sqrt(1 + .85*q*q), 6, True), traj @ modes.T, .00146) @ modes
    else:
        y = _ridge(_basis(w, 2), train['y'], .0015)
        state = _ridge(_basis(w, 1), traj @ modes.T, .0015) @ modes
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([
        y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
cat > /tmp/check04.py <<'PY'
import pickle,runpy,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['y']); ix=np.arange(n)
up=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']; sink=runpy.run_path('/holdout_task/node_01.py')['run_node']
def check(name,folds,panel_drop=False):
    errors=[]
    for i,te in enumerate(folds):
        it=np.setdiff1d(ix,te); ts={**s,'train':{k:tr[k][it] for k in ('x','y','traj')},'test':{'x':tr['x'][te]}}
        if panel_drop:
            ts['panel_a']={k:np.delete(v,i,axis=0) for k,v in s['panel_a'].items()}
        p=sink({'__input__':ts,'node_00':up({'__input__':ts})}); y=p['y_hat']; z=p['state_pred']; assert z.shape==tr['traj'][te].shape and y.shape==tr['y'][te].shape
        np.testing.assert_allclose(p['diagnostics_vec'],[y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()]); errs=np.abs(z-tr['traj'][te]); errors.append([np.abs(y-tr['y'][te]).mean(),errs.mean(),errs.mean(0).max()])
    print(name,'worst scalar, trajectory, horizon',np.max(errors,axis=0)); assert np.max(errors)<.038
check('ordinary random folds',[te for seed in (77,118,231,408,511,1701) for te in np.array_split(np.random.default_rng(seed).permutation(n),4)])
check('ordered folds',np.array_split(ix,4)); tail=np.argsort(np.linalg.norm(tr['x'],axis=1))[-12:]; check('input extremes',[tail]+[np.argsort(tr['x'][:,k])[sl] for k in range(tr['x'].shape[1]) for sl in (slice(0,12),slice(-12,None))]); check('panel jackknife on extreme slice',[tail]*len(s['panel_a']['u']),True)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check04.py
```
- **Container Exit Code**: `0` (`1024 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.005932, "public_cv_traj_mae": 0.004245}, "exec_counts": {"node_00": 1, "node_01": 1}}
ordinary random folds worst scalar, trajectory, horizon [0.00784873 0.00462641 0.00657929]
ordered folds worst scalar, trajectory, horizon [0.00646538 0.00425968 0.00633953]
input extremes worst scalar, trajectory, horizon [0.00859421 0.00447287 0.00567694]
panel jackknife on extreme slice worst scalar, trajectory, horizon [0.00569704 0.00403846 0.00497692]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=08734fc8d45433dc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_08734fc8 (directory).
```

### Turn 10 — `submit_completed_task` (`14008 ms` LLM)
- **Rationale**: The exact submitted artifact passed isolated-process verification with one execution per node and no contract errors. The cubic inverse also improved performance across random and ordered folds, extreme inputs, and the calibration-panel jackknife.
- **Alternatives Weighed**: `Submit the earlier weighted regression as the primary model.` (rejected: It passed the public evaluator but exceeded 0.038 on the high-norm holdout for both heads.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_01.py` without modifying the frozen node. The public evaluator accepted it: scalar MAE 0.005932 and trajectory MAE 0.004245, both below 0.038.`

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
