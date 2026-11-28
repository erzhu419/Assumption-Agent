# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m101_a01_batch_05__blind_campaign__reuse_subgraph`

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
| **Turns Used / Max Turns** | `11 / 12` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `300,969` (`289,084` / `91,264` / `11,885`) |
| **Total Episode Wall Time (s)** | `416.89s` |

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

### Turn 1 — `execute_in_remote_docker` (`11634 ms` LLM)
- **Rationale**: Inspect the frozen output keys and evaluator before fitting; a mismatch would change which features, outputs, or validation splits the sink must use.

```bash
cd /holdout_task && cat _dag_spec.json && cat _frozen_nodes/node_00.py && cat test_dag_candidate.py && cat validate_public_schema.py && cat public_asset_index.json
```
- **Container Exit Code**: `0` (`287 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
    "/holdout_task/test_dag_candidate.py": {
      "bytes": 5206
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 2573
    }
  },
  "schema": "hegel_public_asset_index_v1"
}
```

### Turn 2 — `execute_in_remote_docker` (`20523 ms` LLM)
- **Rationale**: Read the evaluator to recover the fold and output contract. Panel leave-one-out residuals distinguish useful low-order calibration terms from interpolation artifacts and determine whether the response needs a nonlinear panel map.

```bash
cd /holdout_task && sed -n '40,190p' test_dag_candidate.py && cat validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference-panel alignment robust IRLS' 2>&1 | head -20; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np,sys
sys.path.insert(0,'/holdout_task/_frozen_nodes'); from node_00 import run_node
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; q=run_node({'__input__':s})
np.set_printoptions(precision=4,suppress=True)
for k,d in s.items():
 if isinstance(d,dict):
  for key,a in d.items(): print(k,key,a.shape,'min',np.min(a,axis=0),'max',np.max(a,axis=0),'first',a[:2])
 else: print(k,d)
for k in ('panel_b','panel_c'):
 d=s[k]; x=d['u']; lin=np.column_stack([np.ones(len(x)),x]); quad=np.column_stack([lin]+[x[:,i]*x[:,j] for i in range(4) for j in range(i,4)])
 for key,y in d.items():
  if key=='u': continue
  for X in (lin,quad):
   p=np.linalg.pinv(X); e=y-X@(p@y); print(k,key,X.shape[1],'loo',np.mean(abs(e/(1-np.sum(X*p.T,axis=1))[:,None]),axis=0),'coef', (p@y).T)
print('q_tr',q['q_tr'][:4],q['q_tr'].std(axis=0))
PY
```
- **Container Exit Code**: `0` (`404 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_b v 5 loo [0.0001 0.0001 0.0001 0.0001] coef [[-0.0765 -0.6431  0.5774  0.1597 -0.0421]
 [ 0.0021 -0.1234 -0.247   0.1805 -0.8159]
 [-0.0388 -0.0474  0.1613 -0.8336 -0.2261]
 [-0.0959 -0.5858 -0.5949 -0.146   0.2364]]
panel_b v 15 loo [0.0001 0.0001 0.0002 0.0002] coef [[-0.0764 -0.6431  0.5774  0.1597 -0.0421 -0.0001 -0.      0.     -0.0001
  -0.      0.      0.0001 -0.0001  0.0001 -0.    ]
 [ 0.0021 -0.1235 -0.247   0.1806 -0.8159 -0.0001 -0.      0.0001  0.0001
   0.     -0.0001 -0.      0.0001  0.     -0.    ]
 [-0.0387 -0.0473  0.1613 -0.8337 -0.2261  0.      0.      0.0001 -0.0001
  -0.      0.0001 -0.0001 -0.0001 -0.0001  0.0001]
 [-0.0958 -0.5858 -0.595  -0.146   0.2364 -0.0001  0.     -0.0001  0.
  -0.     -0.     -0.     -0.     -0.     -0.    ]]
panel_c v1 5 loo [0.1347 0.0944 0.1227 0.1309] coef [[ 0.1187 -0.5619  0.259   0.1539 -0.7494]
 [ 0.0167 -0.468  -0.2778  0.6826  0.2825]
 [ 0.1106  0.2982 -0.7763  0.0821 -0.4624]
 [-0.1387 -0.5417 -0.3554 -0.5986  0.0924]]
panel_c v1 15 loo [0.2788 0.2386 0.2257 0.3027] coef [[ 0.1916 -0.6335  0.3571  0.0827 -0.8056  0.0272 -0.1121  0.0934  0.1114
   0.0849 -0.0958 -0.1302 -0.0064  0.0341 -0.1469]
 [-0.0341 -0.4542 -0.3137  0.6755  0.2946  0.1168  0.0666 -0.0626  0.0791
  -0.0642  0.0055  0.1255  0.0065 -0.0849  0.0079]
 [ 0.2107  0.2863 -0.7858  0.0538 -0.4887 -0.0165 -0.1036  0.0478 -0.086
  -0.0497  0.051  -0.0374  0.0232  0.0701 -0.1402]
 [ 0.027  -0.5514 -0.3394 -0.6465  0.0925 -0.0699 -0.0381  0.0866  0.0121
  -0.0158 -0.0135 -0.0355 -0.0783  0.065  -0.1208]]
panel_c v2 5 loo [0.1048 0.0988 0.0771 0.1148] coef [[-0.2165 -0.3322 -0.214   0.1299 -0.2627]
 [-0.0318  0.2714 -0.1076 -0.1419 -0.3335]
 [-0.1562 -0.2422  0.2212 -0.3506 -0.0919]
 [-0.0208 -0.0491 -0.2868 -0.2882  0.1652]]
panel_c v2 15 loo [0.0095 0.0119 0.0089 0.0079] coef [[-0.0521 -0.3346 -0.2282  0.1093 -0.2408 -0.1464  0.0077 -0.0034 -0.0004
  -0.1047  0.0016  0.0015  0.0449 -0.0059 -0.1047]
 [ 0.0609  0.2717 -0.1213 -0.1672 -0.3443  0.1174  0.0019  0.0028  0.0019
  -0.0545 -0.0027  0.001  -0.0842 -0.0038 -0.153 ]
 [-0.0436 -0.2163  0.2457 -0.3517 -0.0826 -0.096  -0.0022  0.0033  0.001
   0.1113 -0.0051  0.0046 -0.1545 -0.0034 -0.0354]
 [ 0.0588 -0.0273 -0.3325 -0.2674  0.2296 -0.011   0.0036  0.0007  0.0014
  -0.1503 -0.0037 -0.     -0.1189 -0.0025  0.1013]]
q_tr [[-0.0989  0.9916  0.3302  0.5869]
 [ 0.2016  0.8745 -0.0482  0.9387]
 [-0.121   0.8981 -0.4857  0.7169]
 [ 0.4612  0.649   0.511   0.1462]] [0.5204 0.5394 0.47   0.5554]
```

### Turn 3 — `execute_in_remote_docker` (`30375 ms` LLM)
- **Rationale**: Panel C leave-one-out errors will distinguish low-dimensional nonlinear maps from interpolation. Comparing raw and calibrated features on the declared split will show whether the calibration connection improves response and which trajectory basis preserves accuracy.
- **Alternatives Weighed**: `Fit a full cubic panel map` (rejected: It has 35 terms for only 24 calibration rows, and the v1 quadratic already worsens leave-one-out error.)

```bash
cat > /tmp/inspect_node01.py <<'PY'
import sys,pickle,itertools as it
import numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
from node_00 import run_node
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; q=run_node({'__input__':s})['q_tr']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; c=s['panel_c']; u=c['u']
np.set_printoptions(precision=5,suppress=True)
def F(x,*cols): return np.column_stack([np.ones(len(x)),*cols])
def loo(X,y):
 p=np.linalg.pinv(X); e=y-X@(p@y); return np.mean(np.abs((e.T/(1-np.sum(X*p.T,axis=1))).T),axis=0)
def fit(X,y):return np.linalg.lstsq(X,y,rcond=None)[0]
print('times',t,'telemetry',s['telemetry_vec'],'traj singular values',np.linalg.svd(z,compute_uv=False))
print('panel c u range',u.min(0),u.max(0),'q range',q.min(0),q.max(0));print('panel c first 6:',np.c_[u,c['v1'],c['v2']][:6])
for name,X in [('lin',F(u,u)),('square',F(u,u,u*u)),('cube',F(u,u,u**3)),('sq-cube',F(u,u,u*u,u**3)),('sin2',F(u,u,np.sin(2*u))),('tanh2',F(u,u,np.tanh(2*u))),('diag-quartic',F(u,u,u*u,u**4))]:
 print(name,loo(X,c['v1']),loo(X,c['v2']))
 def cv(X,Y):
  e=abs(X[52:]@fit(X[:52],Y[:52])-Y[52:]); return np.mean(e),np.max(np.mean(e,axis=0))
for name,x in [('q',q),('raw',s['train']['x'])]:
 for deg in (1,2,3):
  X=F(x,*[np.prod(x[:,inds],axis=1) for d in range(1,deg+1) for inds in it.combinations_with_replacement(range(4),d)]);print(name,deg,'cv y/traj',cv(X,y),cv(X,z))
 for tag,xx in [('diag2',F(x,x,x*x)),('diag3',F(x,x,x*x,x**3)),('diag4',F(x,x,x*x,x**3,x**4)),('tanh',F(x,x,np.tanh(1.7*x),x*x)),('sin',F(x,x,np.sin(1.7*x),x*x))]:print(name,tag,cv(xx,y),cv(xx,z))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/inspect_node01.py; cd /holdout_task && sed -n '79,145p' test_dag_candidate.py
```
- **Container Exit Code**: `0` (`406 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.36436311379235464) (0.1075962131574431, 0.13640723697124646)
raw 1 cv y/traj (0.29956962441340096, 0.29956962441340096) (0.11171800802530511, 0.15158942231690442)
raw 2 cv y/traj (0.06197169824305386, 0.06197169824305386) (0.016215297047132256, 0.025215500603576378)
raw 3 cv y/traj (0.08651642266856216, 0.08651642266856216) (0.01368261716978227, 0.016592492828500942)
raw diag2 (0.28551565201966994, 0.28551565201966994) (0.10185823859276724, 0.13409520408863299)
raw diag3 (0.3318913095298984, 0.3318913095298984) (0.10313081885312145, 0.13378643850063154)
raw diag4 (0.34896365385518824, 0.34896365385518824) (0.10617406757299515, 0.1403697587130797)
raw tanh (0.32399416881322973, 0.32399416881322973) (0.10315125720192264, 0.13432253492674145)
raw sin (0.33044905559652493, 0.33044905559652493) (0.10315740727786629, 0.13378296450321198)
    yh, sp, dv = np.asarray(out_full["y_hat"], float), np.asarray(out_full["state_pred"], float), np.asarray(out_full["diagnostics_vec"], float)
    exp_dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
    if float(np.max(np.abs(dv - exp_dv))) > 1e-3:
        errs.append("diagnostics_vec_handoff_mismatch")

cv_metrics = {}
if not errs:
    st_cv = copy.deepcopy(st)
    sp_idx = max(40, len(st["train"]["x"]) - 20)
    vx, vy, vtr = st["train"]["x"][sp_idx:].copy(), st["train"]["y"][sp_idx:].copy(), st["train"]["traj"][sp_idx:].copy()
    st_cv["train"] = {
        "x": st["train"]["x"][:sp_idx].copy(),
        "y": st["train"]["y"][:sp_idx].copy(),
        "traj": st["train"]["traj"][:sp_idx].copy(),
        "summary": np.array([float(np.mean(st["train"]["y"][:sp_idx])), float(np.std(st["train"]["y"][:sp_idx])), float(np.mean(st["train"]["traj"][:sp_idx, -1])), float(np.std(st["train"]["traj"][:sp_idx, -1]))], dtype=float),
    }
    st_cv["test"] = {"x": vx}
    out_cv, _ = run_dag(st_cv)
    ym = float(np.mean(np.abs(np.asarray(out_cv["y_hat"], float) - vy)))
    tm = float(np.mean(np.abs(np.asarray(out_cv["state_pred"], float) - vtr)))
    cv_metrics = {"public_cv_y_mae": round(ym, 6), "public_cv_traj_mae": round(tm, 6)}
    if ym > 0.038: errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
    if tm > 0.038: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")

ok = len(errs) == 0 and all(v == 1 for v in ec.values())
print(json.dumps({"accepted": ok, "errors": errs, "cv_metrics": cv_metrics, "exec_counts": ec}, allow_nan=False))
sys.exit(0 if ok else 1)
```

### Turn 4 — `execute_in_remote_docker` (`34518 ms` LLM)
- **Rationale**: Compare compact panel families and test whether their outputs predict response beyond the quadratic frozen-q basis. Check for an existing sink source before reconstructing its trajectory path.
- **ATMS NOGOOD DECLARED**: Refuted `An unrestricted low-degree polynomial in the frozen calibration coordinates is sufficient for response prediction.` | Witness: `Degree two gives response MAE 0.061972 and degree three gives 0.086516, both above 0.038; panel C favors odd saturating and even nonlinear terms.` | Blanket Diagnosis: `An affine coordinate change cannot improve a complete polynomial basis of the same degree; the response needs features tied to the nonlinear panel maps.` | Sublated Pivot: `Identify compact panel C transforms before using them in the response regression.`

```bash
cat > /tmp/discover01.py <<'PY'
exec(open('/tmp/inspect_node01.py').read().split("print('times'")[0])
from scipy.optimize import minimize, minimize_scalar
G={'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'atan':lambda x,k:np.arctan(k*x),'soft':lambda x,k:x/np.sqrt(1+k*x*x),'cos':lambda x,k:np.cos(k*x),'exp':lambda x,k:np.exp(-k*x*x),'rat':lambda x,k:x*x/(1+k*x*x),'log':lambda x,k:np.log1p(k*x*x),'sqrt':lambda x,k:np.sqrt(1+k*x*x)}
chosen=[]
for target,names in [('v1',['tanh','sin','atan','soft']),('v2',['cos','exp','rat','log','sqrt'])]:
 v=c[target]; best=None
 for name in names:
  g=G[name]
  X=lambda x,k:F(x,x,g(x,k))
  obj=lambda k: np.mean((v-X(u,k)@fit(X(u,k),v))**2)
  for free in (False,True):
   k=minimize_scalar(obj,bounds=(.1,4),method='bounded').x
   if free:k=minimize(obj,np.repeat(k,4),bounds=[(.1,4)]*4,method='L-BFGS-B',options={'gtol':1e-12,'ftol':1e-14}).x
   Xp=X(u,k); err=loo(Xp,v).mean(); print(target,name,free,k,'rmse',obj(k)**.5,'loo',err,'coef norms',np.linalg.norm(fit(Xp,v),axis=1));
   if best is None or err<best[0]:best=(err,name,k,fit(Xp,v))
 _,name,k,b=best; chosen.append(F(q,q,G[name](q,k))@b);print('best',target,name,k,b.T)
a,b=chosen
p=lambda x:np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)])
cv=lambda X,Y: np.mean(abs(X[52:]@fit(X[:52],Y[:52])-Y[52:]))
for name,X in [('q',F(q,q)),('q2',F(q,q,p(q))),('q2+a',F(q,q,p(q),a)),('q2+b',F(q,q,p(q),b)),('q2+a+b',F(q,q,p(q),a,b)),('q2+a+b+a*b',F(q,q,p(q),a,b,a*b)),('q2+a+b+a^2',F(q,q,p(q),a,b,a*a)),('q2+a+b+q*a',F(q,q,p(q),a,b,q*a)),('q2+a+b+q*b',F(q,q,p(q),a,b,q*b)),('q2+a+b+q*a+q*b',F(q,q,p(q),a,b,q*a,q*b)),('a+b',F(q,a,b)),('a+b+p(a)',F(q,a,b,p(a))),('a+b+p(b)',F(q,a,b,p(b))),('a+b+a*b',F(q,a,b,a*b)),('allquad',F(q,q,a,b,p(q),p(a),p(b)))]:
 print(name,'y',cv(X,y),'loo',loo(X,y),'traj',cv(X,z),'resid',np.std(y-X@fit(X,y)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/discover01.py; ls -la /holdout_task /holdout_task/_frozen_nodes /pretrain_task
```
- **Container Exit Code**: `0` (`486 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.52726
  -0.12305]
 [ 0.06288 -0.026   -0.33193 -0.26698  0.22605 -0.03925 -0.50112 -0.40386
   0.34501]]
q y 0.299569624413401 loo 0.29922861503911036 traj 0.1117180080253051 resid 0.34476944120226394
q2 y 0.06197169824305261 loo 0.0618803940777881 traj 0.01621529704713223 resid 0.06399034489609233
q2+a y 0.058286144835289974 loo 0.06494159835712793 traj 0.012204384397526278 resid 0.06089752096159919
q2+b y 0.07698231561118198 loo 0.06860172426822156 traj 0.02054367822761461 resid 0.06305209066771074
q2+a+b y 0.0631274051761885 loo 0.06790995885900467 traj 0.01647406020427878 resid 0.0580509379689868
q2+a+b+a*b y 0.06657588494898209 loo 0.06972459531577274 traj 0.014338919955812952 resid 0.054597324119391895
q2+a+b+a^2 y 0.0823518127400745 loo 0.07330311685871571 traj 0.018410005249837175 resid 0.05492403520409359
q2+a+b+q*a y 0.05814939914040999 loo 0.06274586654739833 traj 0.020593314961874908 resid 0.04381214402541201
q2+a+b+q*b y 0.1670431810026492 loo 0.07532265040812214 traj 0.020181758571205632 resid 0.05406350575886558
q2+a+b+q*a+q*b y 0.12331507709759768 loo 0.06824868523090506 traj 0.02076810905260364 resid 0.040628168180657634
a+b y 0.28901885313663717 loo 0.29904080475196854 traj 0.09352823467950819 resid 0.3342271164465355
a+b+p(a) y 0.09247315789955972 loo 0.104021975670204 traj 0.045776689674126264 resid 0.09584666617567018
a+b+p(b) y 0.10014377837410113 loo 0.1339368087926488 traj 0.0491628752851599 resid 0.12395226970348538
a+b+a*b y 0.2828276044373972 loo 0.2831976339483926 traj 0.08597359354463996 resid 0.2841163230680648
allquad y 0.3268889219265538 loo 0.06391986003367345 traj 0.05863263622645598 resid 0.025674400747104992
/holdout_task:
total 56
drwxr-xr-x 3 root root  4096 Nov 28 09:36 .
drwxr-xr-x 1 root root  4096 Nov 28 09:36 ..
-r--r--r-- 1 root root   476 Nov 28 09:36 _dag_spec.json
drwxr-xr-x 3 root root  4096 Nov 28 09:36 _frozen_nodes
-rw-r--r-- 1 root root  4208 Nov 28 09:36 public_asset_index.json
-rw-r--r-- 1 root root 16432 Nov 28 09:36 public_batch.pkl
-rw-r--r-- 1 root root  5206 Nov 28 09:36 test_dag_candidate.py
-rw-r--r-- 1 root root  2573 Nov 28 09:36 validate_public_schema.py

/holdout_task/_frozen_nodes:
total 16
drwxr-xr-x 3 root root 4096 Nov 28 09:36 .
drwxr-xr-x 3 root root 4096 Nov 28 09:36 ..
drwxr-xr-x 2 root root 4096 Nov 28 09:36 __pycache__
-r--r--r-- 1 root root  746 Nov 28 09:36 node_00.py

/pretrain_task:
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 28 09:36 ..
```

### Turn 5 — `execute_in_remote_docker` (`43669 ms` LLM)
- **Rationale**: Compare compact interaction blocks and fold-local sparse selection across multiple row splits. If an apparent gain is confined to the public split, reject that basis rather than embedding it in the sink.
- **Alternatives Weighed**: `Expand to all second-order terms in all coordinates` (rejected: The measured public response MAE was 0.326889; the expansion fits residuals in-sample but generalizes poorly.)

```bash
cat > /tmp/response_grammar01.py <<'PY'
exec(open('/tmp/inspect_node01.py').read().split("print('times'")[0])
from itertools import combinations_with_replacement
from scipy.optimize import minimize_scalar

def calibrated(x, key, fn, bounds):
    design = lambda x,k: F(x,x,fn(k,x))
    v = c[key]
    def loss(k):
        A=design(u,k); return np.mean((A@fit(A,v)-v)**2)
    k=minimize_scalar(loss,bounds=bounds,method='bounded').x
    A=design(u,k)
    print('panel',key,k,loss(k)**.5,np.mean(loo(A,v)))
    return design(x,k)@fit(A,v)
a=calibrated(q,'v1',lambda k,x:np.arctan(k*x),(.8,4))
b=calibrated(q,'v2',lambda k,x:np.log1p(k*x*x),(.1,1))
poly=lambda x,d: np.column_stack([np.prod(x[:,ix],axis=1) for ix in combinations_with_replacement(range(x.shape[1]),d)])
base=F(q,q,poly(q,2)); co=fit(base,y); M=np.zeros((4,4))
for k,(i,j) in enumerate(combinations_with_replacement(range(4),2)): M[i,j]=M[j,i]=co[5+k]/(1+(i!=j))
print('response quad intercept / linear:',co[:5], 'matrix',M,'eigen',np.linalg.eigh(M)); print('residual quantiles',np.quantile(y-base@co,[0,.1,.5,.9,1]))
blocks={'q':q,'a':a,'b':b,'sat':np.arctan(2.3*q),'log':np.log1p(.36*q*q),'sin':np.sin(1.7*q),'quartic':q**4,'cube':q**3,'rat':q/(1+q*q)}
extras={}
for name,x in blocks.items(): extras[name]=x; extras[name+'^2']=poly(x,2)
for k,l in [('q','a'),('q','b'),('a','b'),('q','sat'),('q','log'),('sat','log'),('q','sin'),('q','rat')]:
    x,w=blocks[k],blocks[l]; extras[k+'*'+l]=(x[:,:,None]*w[:,None,:]).reshape(len(q),-1)
for k,x in [('q',q),('a',a),('b',b)]:
    for sg in [1,-1]:
        D=np.column_stack([x[:,i]+sg*x[:,j] for i in range(4) for j in range(i+1,4)])
        for tag,h in [('tanh',lambda d:np.tanh(1.7*d)),('sin',lambda d:np.sin(1.7*d)),('cos',lambda d:np.cos(1.7*d)),('rat',lambda d:1/(1+d*d))]: extras[f'{k}:{sg}:{tag}']=h(D)
for j,x in enumerate([q,a,b]):
    extras[f'p3_{j}']=poly(x,3); extras[f'p4_{j}']=poly(x,4)
N=len(y); ids=np.arange(N); splits=[(ids[:52],ids[52:])]+[(ids[~np.isin(ids,v)],v) for v in np.array_split(np.random.default_rng(24).permutation(N),6)]
def err(A,Y):
    return [np.mean(abs(A[v]@fit(A[t],Y[t])-Y[v])) for t,v in splits]
R=[]
for k,x in extras.items():
    es=err(F(q,base[:,1:],x),y); R.append((np.mean(es[1:]),max(es[1:]),es[0],k))
print('extra block fits avg/worst/public:',*sorted(R)[:16],sep='\n'); print('best public',sorted(R,key=lambda x:x[2])[:6]); print('trajectory quadratic splits',err(base,z))
D=np.column_stack(list(extras.values())); names=[f'{k}:{i}' for k,x in extras.items() for i in range(x.shape[1])]
for use in ['all','mechanistic']:
    ix=np.arange(D.shape[1]) if use=='all' else np.array([i for i,k in enumerate(names) if k.startswith(('q*a','q*b','a*b','a^2','b^2'))]); C=D[:,ix]; scores=[]
    for t,v in splits:
        B=base[t]; r=y[t]-B@fit(B,y[t]); A=C[t]-B@fit(B,C[t]); chosen=[]; norms=np.linalg.norm(A,axis=0); norms[norms<1.e-8]=np.inf; A=A/norms
        yh=[]
        for step in range(14):
            j=np.argmax(abs(A.T@r)); chosen.append(j)
            g=F(q,base[:,1:],C[:,chosen]); cf=fit(g[t],y[t]); r=y[t]-g[t]@cf; yh.append(np.mean(abs(g[v]@cf-y[v])))
        scores.append(yh)
        if len(scores)==1: print('public sparse selections',use,[names[ix[j]] for j in chosen])
    print('sparse',use,'public/mean/worst',np.array([scores[0],np.mean(scores[1:],axis=0),np.max(scores[1:],axis=0)]))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/response_grammar01.py
```
- **Container Exit Code**: `0` (`602 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
92352731841, 'a:1:cos')
(0.06337038176841649, 0.10131346475611835, 0.07510514344203327, 'b:-1:cos')
(0.063678337626699, 0.0874516241027049, 0.059196371049741736, 'a:1:rat')
(0.06422846677225232, 0.0963888211576885, 0.08182895664937354, 'a:1:sin')
(0.06516232341034345, 0.08141370127333136, 0.11175443638208808, 'a^2')
(0.06521238719268511, 0.09453007060413991, 0.0896142818998091, 'a:1:tanh')
(0.06617319268268688, 0.09857541867624268, 0.05828094898948536, 'a')
(0.06618961387870546, 0.09857636778850513, 0.05830074043779589, 'sat')
(0.0669306767529502, 0.11089799784989708, 0.07317269135466496, 'q:-1:tanh')
(0.06705401329369387, 0.11572071048922221, 0.11113047824848146, 'cube^2')
(0.06749313350815007, 0.09758295700887894, 0.06938883545647738, 'q:1:tanh')
best public [(0.07178459736332599, 0.10873940589846938, 0.0581862761262961, 'b:1:rat'), (0.06617319268268688, 0.09857541867624268, 0.05828094898948536, 'a'), (0.06618961387870546, 0.09857636778850513, 0.05830074043779589, 'sat'), (0.06860575656874722, 0.08599785130269151, 0.059128248650133344, 'b:1:tanh'), (0.063678337626699, 0.0874516241027049, 0.059196371049741736, 'a:1:rat'), (0.06762781741879041, 0.09779605937871477, 0.060157729551586436, 'rat')]
trajectory quadratic splits [0.01621529704713223, 0.009876126053426285, 0.015376663188947676, 0.01664984747222055, 0.014775128181298842, 0.010759064511320138, 0.011955534662714354]
public sparse selections all ['a*b:3', 'q*a:6', 'p3_1:8', 'p3_1:7', 'p3_1:4', 'p4_1:3', 'p4_1:29', 'a*b:15', 'p4_1:2', 'a*b:4', 'q*b:4', 'p4_1:6', 'p3_2:15', 'p3_2:13']
sparse all public/mean/worst [[0.072   0.08096 0.06968 0.06459 0.06651 0.06405 0.06017 0.05931 0.06266
  0.06608 0.06167 0.0606  0.05948 0.05856]
 [0.06653 0.06342 0.06185 0.05635 0.05876 0.05992 0.0591  0.06232 0.06113
  0.06037 0.06305 0.06328 0.06484 0.06632]
 [0.12741 0.159   0.15558 0.16205 0.15766 0.15225 0.13803 0.14906 0.14519
  0.14635 0.16979 0.1692  0.16805 0.16652]]
public sparse selections mechanistic ['a*b:3', 'q*a:6', 'a*b:7', 'q*b:9', 'q*a:4', 'q*b:0', 'a^2:8', 'a^2:3', 'q*b:6', 'q*b:8', 'a^2:4', 'a*b:1', 'q*a:13', 'q*a:10']
sparse mechanistic public/mean/worst [[0.072   0.08096 0.07194 0.06944 0.07092 0.07064 0.06547 0.06512 0.06979
  0.07906 0.07969 0.08892 0.08558 0.09136]
 [0.07393 0.07354 0.07012 0.0683  0.06651 0.0669  0.06914 0.07197 0.06875
  0.06856 0.07348 0.07158 0.07053 0.06826]
 [0.12741 0.13181 0.13724 0.13749 0.10655 0.09979 0.10836 0.11283 0.10915
  0.11138 0.11717 0.10854 0.10792 0.10873]]
```

### Turn 6 — `execute_in_remote_docker` (`39934 ms` LLM)
- **Rationale**: The quadratic response has two dominant eigenvalues of opposite sign, consistent with a product of two linear projections. Fit that product and compact nonlinear gates inside each fold to test whether they explain held-out residuals.
- **Alternatives Weighed**: `Add more panel-derived basis blocks` (rejected: Neither dense blocks nor fold-local sparse selection improved reliably across splits.)

```bash
cat /holdout_task/_frozen_nodes/node_00.py; cat > /tmp/profile01.py <<'PY'
exec(open('/tmp/inspect_node01.py').read().split("print('times'")[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
pairs=list(combinations_with_replacement(range(4),2))
def quad(x): return np.column_stack([x[:,i]*x[:,j] for i,j in pairs])
def affine(x): return F(x,x)
def base(x): return F(x,x,quad(x))
def design(x,p,mode):
    h=p[0]+x@p[1:5]; D=1+h*h
    if mode=='rat': return np.column_stack([affine(x),quad(x)/D[:,None]])
    if mode=='ratfull': return np.column_stack([affine(x),base(x)/D[:,None]])
    if mode=='rataxis': return np.column_stack([affine(x),quad(x),1/D])
    if mode=='gate': return np.column_stack([base(x),affine(x)*np.tanh(h)[:,None]])
    if mode=='tanh': return np.column_stack([base(x),np.tanh(h)])
    if mode=='sin': return np.column_stack([base(x),np.sin(h)])
    if mode=='cosgate': return np.column_stack([base(x),affine(x)*np.cos(h)[:,None]])
    if mode=='rat2': return np.column_stack([affine(x),quad(x)/D[:,None],quad(x)])
    if mode=='rank2rat':
        g=x@p[5:9]; j=x@p[9:13]; return np.column_stack([affine(x),g*j/D])
    if mode=='rank2sin':
        g=x@p[5:9]; j=x@p[9:13]; return np.column_stack([affine(x),g*j,np.sin(h)])
    if mode=='rank2gate':
        g=x@p[5:9]; j=x@p[9:13]; return np.column_stack([affine(x),g*j,np.tanh(h)])
    if mode=='rank2':
        g=x@p[5:9]; j=x@p[9:13]; return np.column_stack([affine(x),g*j])
ye=np.linalg.eigh(np.linalg.lstsq(quad(q),y-affine(q)@fit(affine(q),y),rcond=None)[0][:4,None]@np.ones((1,4))) if False else None

def train(x,y,mode,n=8):
    b=fit(base(x),y); M=np.zeros((4,4));
    for k,(i,j) in enumerate(pairs): M[i,j]=M[j,i]=b[k+5]/(1+(i!=j))
    e,V=np.linalg.eigh(M); a=V[:,-1]*np.sqrt(abs(e[-1])); b=V[:,0]*np.sqrt(abs(e[0])); g=a+b; j=a-b
    def residual(p):
        X=design(x,p,mode); return X@fit(X,y)-y
    rng=np.random.default_rng(495); best=None
    for k in range(n):
        pp=np.r_[0.,np.eye(4)[k%4]*(.7 if 'rat' in mode else 1.7)] if k<4 else rng.normal(size=5)*(.8 if 'rat' in mode else 1.7)
        if mode.startswith('rank2'): pp=np.r_[pp,g,j]
        f=least_squares(residual,pp,bounds=(-7,7),max_nfev=125,ftol=2e-7); v=np.mean(f.fun**2)
        if best is None or v<best[0]: best=(v,f.x)
    return best
ids=np.arange(len(y)); splits=[(ids[:52],ids[52:])]+[(ids[~np.isin(ids,v)],v) for v in np.array_split(np.random.default_rng(24).permutation(len(y)),6)]
for m in ('rank2','rank2rat','rank2sin','rank2gate','rat','ratfull','rataxis','gate','tanh','sin','cosgate','rat2'):
    es=[]
    for k,(t,v) in enumerate(splits):
        var,p=train(q[t],y[t],m); X=design(q,p,m); pred=X[v]@fit(X[t],y[t]); es.append(np.mean(abs(pred-y[v])))
        if k==0: print(m,'fit RMSE',var**.5,'p',p,'public',es[0],flush=True)
    print(m,'folds',np.round(es,5),'worst',max(es),flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/profile01.py
```
- **Container Exit Code**: `0` (`17716 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
at fit RMSE 0.036354982609044184 p [ 0.77138  0.09572  0.21149  0.12694  0.1592  -0.12317  0.98344  0.73348
 -0.13175 -0.53169  0.09321 -0.09976 -1.01743] public 0.06128296521054395
rank2rat folds [0.06128 0.04747 0.07079 0.03533 0.05699 0.03191 0.05416] worst 0.07079239640827789
rank2sin fit RMSE 0.04503744420054668 p [ 0.27772  0.60917  1.01429  0.66737  1.11191 -0.08584  0.91585  0.6734
 -0.03785 -0.53746  0.05578 -0.11557 -1.02419] public 0.06689221043632396
rank2sin folds [0.06689 0.04055 0.05998 0.04344 0.06649 0.03425 0.04555] worst 0.06689221043632396
rank2gate fit RMSE 0.0433121626459201 p [-1.03361  0.74638 -0.91013 -0.65676  1.39196 -0.07565  0.77953  0.63061
  0.37093 -0.52157 -0.07271 -0.42293 -0.92297] public 0.044937765525791484
rank2gate folds [0.04494 0.04114 0.06603 0.0599  0.06658 0.03475 0.04207] worst 0.06658429341253752
rat fit RMSE 0.03482049053461946 p [6.9996  0.25105 0.71707 0.31023 0.62445] public 0.06036080711958556
rat folds [0.06036 0.04839 0.08812 0.03979 0.05992 0.03426 0.05583] worst 0.08811734610066797
ratfull fit RMSE 0.017351775333270838 p [ 0.32689 -0.10228  0.54177  0.25884 -0.56741] public 0.04344933045067802
ratfull folds [0.04345 0.01933 0.05844 0.03554 0.04326 0.02666 0.03875] worst 0.05843734091749278
rataxis fit RMSE 0.038684826378214084 p [-0.01281  0.32923 -0.39105 -0.28394  0.61932] public 0.047172878053895376
rataxis folds [0.04717 0.04562 0.06149 0.0503  0.07113 0.0359  0.06305] worst 0.0711298228688832
gate fit RMSE 0.019405659019287 p [-0.31422  0.00517 -0.00818 -0.00466  0.01055] public 0.028745886753627888
gate folds [0.02875 0.01971 0.03431 0.05127 0.0433  0.02082 0.03094] worst 0.05126986301281946
tanh fit RMSE 0.03912082037165062 p [-0.6605   0.03443 -0.04178 -0.03045  0.06804] public 0.05612784872666381
tanh folds [0.05613 0.04549 0.06766 0.06145 0.06274 0.03917 0.05867] worst 0.06765883476322308
sin fit RMSE 0.038697938922934966 p [-1.62778  1.4078  -1.67137 -1.21166  2.62696] public 0.04815919377065828
sin folds [0.04816 0.04601 0.06091 0.05074 0.06608 0.03581 0.05961] worst 0.06607781130128727
cosgate fit RMSE 0.02740054767066538 p [0.98714 0.44497 1.73324 0.46211 0.28139] public 0.07194356368678939
cosgate folds [0.07194 0.07512 0.03488 0.05535 0.04331 0.01971 0.03495] worst 0.07512270192775451
rat2 fit RMSE 0.013618401787620594 p [-0.8196  -0.98691  1.95596  1.65866 -0.77428] public 0.0641058854502771
rat2 folds [0.06411 0.02994 0.06479 0.05588 0.06563 0.02032 0.02609] worst 0.06562615129163676
```

### Turn 7 — `execute_in_remote_docker` (`39265 ms` LLM)
- **Rationale**: Save the passing response candidate and test the actual DAG contract in isolation. The fitted tanh scale nearly vanishes, so test factorized polynomial limits across held-out folds before replacing the candidate. No prior sink source was available; use the validated quadratic trajectory fit in frozen coordinates.
- **Alternatives Weighed**: `Use the fitted rational gate as the response model` (rejected: Its best public response MAE was 0.043449, above the 0.038 limit.)

```bash
sed -n '1,41p;54,82p' /holdout_task/test_dag_candidate.py; cat > /holdout_task/node_01.py <<'PY'
import numpy as np
from scipy.optimize import least_squares


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _quadratic(x):
    return np.column_stack((_affine(x), *(x[:, i] * x[:, j]
                            for i in range(x.shape[1]) for j in range(i, x.shape[1]))))


def _response_design(x, p):
    h = p[0] + x @ p[1:]
    return np.column_stack((_quadratic(x), _affine(x) * np.tanh(h)[:, None]))


def _response(x, xt, y):
    def residual(p):
        a = _response_design(x, p)
        return a @ _fit(a, y) - y

    rng = np.random.default_rng(495)
    starts = [np.r_[0., v] for v in 1.7 * np.eye(x.shape[1])]
    starts += [rng.normal(size=x.shape[1] + 1) * 1.7 for _ in range(4)]
    best = min((least_squares(residual, p, bounds=(-7, 7), max_nfev=125, ftol=2e-7)
                for p in starts), key=lambda r: r.cost)
    return _response_design(xt, best.x) @ _fit(_response_design(x, best.x), y)


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    x, xt = inputs['node_00']['q_tr'], inputs['node_00']['q_te']
    y_hat = _response(x, xt, train['y'])
    state_pred = _quadratic(xt) @ _fit(_quadratic(x), train['traj'])
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
cat > /tmp/cubic01.py <<'PY'
exec(open('/tmp/profile01.py').read().split('ids=np.arange')[0])
ids=np.arange(len(y)); splits=[(ids[:52],ids[52:])]+[(ids[~np.isin(ids,v)],v) for v in np.array_split(np.random.default_rng(24).permutation(len(y)),6)]
def D(x,p,m):
    h=x@p[:4]; g=x@p[4:8]; j=x@p[8:12]; A=base(x)
    if m=='sq': C=x*h[:,None]**2
    if m=='cube': C=h**3
    if m=='prod': C=h*g*j
    if m=='prod2': C=x*(h*g)[:,None]
    if m=='sq2': C=np.column_stack([x*h[:,None]**2,x*g[:,None]**2])
    if m=='quart': C=affine(x)*h[:,None]**3
    if m=='pure': C=np.column_stack([h**2,h**3])
    if m=='qg': C=quad(x)*h[:,None]
    return np.column_stack([A,C])
for m,n in [('sq',4),('cube',4),('prod',12),('prod2',8),('sq2',8),('quart',4),('pure',4),('qg',4)]:
    es=[]
    for t,v in splits:
        def res(p):
            A=D(q[t],np.pad(p,(0,12-n)),m);return A@fit(A,y[t])-y[t]
        rng=np.random.default_rng(143)
        r=min((least_squares(res,p,max_nfev=120,ftol=1e-7) for p in rng.normal(size=(10,n))),key=lambda r:r.cost)
        X=D(q,np.pad(r.x,(0,12-n)),m); es.append(np.mean(abs(X[v]@fit(X[t],y[t])-y[v])))
        if len(es)==1: print(m,'public train RMSE/params',np.std(r.fun),r.x,flush=True)
    print(m,np.round(es,5),'worst',max(es),flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/cubic01.py
```
- **Container Exit Code**: `0` (`15535 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
ray(out_full[k], dtype=float)
    if a.shape != shp or not np.all(np.isfinite(a)):
        errs.append(f"bad:{k}:{a.shape}")
if not errs:
    yh, sp, dv = np.asarray(out_full["y_hat"], float), np.asarray(out_full["state_pred"], float), np.asarray(out_full["diagnostics_vec"], float)
    exp_dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
    if float(np.max(np.abs(dv - exp_dv))) > 1e-3:
        errs.append("diagnostics_vec_handoff_mismatch")
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.028746, "public_cv_traj_mae": 0.016215}, "exec_counts": {"node_00": 1, "node_01": 1}}
sq public train RMSE/params 0.03241067240435641 [ 2.2366  12.29652  5.4886   1.28417]
sq [0.06648 0.05469 0.10289 0.04959 0.07333 0.0456  0.07095] worst 0.10288648142994644
cube public train RMSE/params 0.04426488226831863 [11.50017 19.59706 11.83425 21.68509]
cube [0.06696 0.05032 0.07684 0.05448 0.08344 0.04136 0.07168] worst 0.08343631675783279
prod public train RMSE/params 0.031199384825120163 [-0.42234  2.4044   1.33695 -0.78225 -1.20325  1.46764 -0.12473 -3.69172
 -1.16155 -2.67762 -1.21107 -1.46313]
prod [0.06854 0.04765 0.16023 0.04895 0.07714 0.03759 0.06493] worst 0.16023498381808177
prod2 public train RMSE/params 0.031199384767654527 [-1.43801  1.75436 -0.14936 -4.41257 -0.57359 -1.32242 -0.59817 -0.72264]
prod2 [0.06854 0.04765 0.16024 0.04895 0.07714 0.03759 0.06493] worst 0.16023783977965503
sq2 public train RMSE/params 0.017870562865524513 [-3.63305  7.66266  4.1528  -1.32089  0.66285 -3.13315  1.32358  2.84982]
sq2 [0.11607 0.04604 0.11777 0.04742 0.09092 0.0436  0.09114] worst 0.11776556702863365
quart public train RMSE/params 0.03618942295640577 [ 32.42427 -13.74897 -21.97986  49.87228]
quart [0.05563 0.03539 0.0681  0.05365 0.08602 0.0399  0.08178] worst 0.08602086433322498
pure public train RMSE/params 0.044264882258931895 [-13.99567 -23.84973 -14.40127 -26.39029]
pure [0.06696 0.05033 0.07684 0.05449 0.08344 0.04135 0.07168] worst 0.08343649035846061
qg public train RMSE/params 0.02008470484487152 [ -3.28096   7.81272  -1.48764 -16.98223]
qg [0.09946 0.058   0.12979 0.03752 0.07943 0.04191 0.07134] worst 0.12979242013256967

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=266e37bf42553964 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_266e37bf (directory).
```

### Turn 8 — `execute_in_remote_docker` (`33078 ms` LLM)
- **Rationale**: Panel leave-one-out error tests whether the affine calibration drops a reproducible nonlinear term. Compare the resulting response fits across the same folds, leaving the verified candidate and trajectory path untouched unless the challenger improves held-out error.
- **Alternatives Weighed**: `Replace the gate with a factorized cubic` (rejected: The best tested factorized polynomial missed the public response threshold and worsened the worst shuffled fold.)

```bash
cat > /tmp/calibration_probe01.py <<'PY'
exec(open('/tmp/profile01.py').read().split('ids=np.arange')[0])
from scipy.optimize import minimize_scalar
from itertools import product
p=s['panel_b']; u=p['u']; w=p['v']; print('panel B linear LOO',loo(affine(u),w)); print('panel B quad LOO',loo(base(u),w)); print('panel B residual std',np.std(w-affine(u)@fit(affine(u),w),axis=0))
a=s['panel_a']; aa=[]; bb=[]
for k,l in zip(a['u'].T,a['v'].T):
 i,j=np.triu_indices(len(k),1); d=k[i]-k[j]; keep=abs(d)>1e-9; m=np.median((l[i]-l[j])[keep]/d[keep]); aa.append(m);bb.append(np.median(l-m*k))
raw=(s['train']['x']-bb)/aa
families={'lin':(lambda x,k:affine(x),(.1,3)), 'quad':(lambda x,k:base(x),(.1,3)), 'diag2':(lambda x,k:F(x,x,x*x),(.1,3)), 'diag3':(lambda x,k:F(x,x,x**3),(.1,3)), 'tanh':(lambda x,k:F(x,x,np.tanh(k*x)),(.1,4)), 'atan':(lambda x,k:F(x,x,np.arctan(k*x)),(.1,4)), 'log':(lambda x,k:F(x,x,np.log1p(k*x*x)),(.02,4)), 'rat':(lambda x,k:F(x,x,x*x/(1+k*x*x)),(.02,4)), 'sin':(lambda x,k:F(x,x,np.sin(k*x)),(.1,4)), 'cos':(lambda x,k:F(x,x,np.cos(k*x)),(.1,4))}
ids=np.arange(len(y)); splits=[(ids[:52],ids[52:])]+[(ids[~np.isin(ids,v)],v) for v in np.array_split(np.random.default_rng(24).permutation(len(y)),6)]
for key in ['panel_b','panel_c/v1','panel_c/v2']:
 if key!='panel_b':u=c['u'];w=c[key.split('/')[1]]
 scores=[]
 for name,(fn,bounds) in families.items():
  def loss(k):
   U=fn(u,k); return np.mean((U@fit(U,w)-w)**2)
  k=minimize_scalar(loss,bounds=bounds,method='bounded').x; U=fn(u,k); scores.append((float(np.mean(loo(U,w))),name,k,loss(k)**.5))
 print(key,sorted(scores),flush=True)
 for _,name,k,_ in sorted(scores)[:3]:
  if name=='lin':continue
  fn=families[name][0]; X=fn(raw if key=='panel_b' else q,k)@fit(fn(u,k),w)
  for label,W in [('map',X),('delta_half',q+.5*(X-q))] if key=='panel_b' else [('map',X)]:
   es=[]
   for t,v in splits:
    _,g=train(W[t],y[t],'gate'); A=design(W,g,'gate'); es.append(np.mean(abs(A[v]@fit(A[t],y[t])-y[v])))
   print(key,name,label,'gate',np.round(es,5),flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/calibration_probe01.py
```
- **Container Exit Code**: `0` (`14802 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.03093]
panel_b rat delta_half gate [0.02877 0.01964 0.03441 0.05125 0.0433  0.02082 0.03092]
panel_b log map gate [0.02868 0.01963 0.03435 0.05125 0.0433  0.02082 0.03095]
panel_b log delta_half gate [0.0287  0.01966 0.03417 0.05126 0.0433  0.02082 0.03092]
panel_c/v1 [(0.00396824604333746, 'atan', 2.3118024044877084, 0.0030377184594988166), (0.006029430455986132, 'tanh', 2.175690561984301, 0.004691133438282386), (0.014149826846081678, 'sin', 3.283687819511384, 0.011002168351348113), (0.048017947647739076, 'diag3', 2.9999933971457455, 0.03524736943226844), (0.12068008495285626, 'lin', 2.9999933971457455, 0.11502696859798386), (0.14252769377780014, 'rat', 0.020005600530221063, 0.10376616891247195), (0.14254940770102847, 'log', 0.020005600530221063, 0.10372745832617362), (0.14256828281637526, 'cos', 0.10000548795674928, 0.10369141896635584), (0.1425699768232123, 'diag2', 2.9999933971457455, 0.10368805461115313), (0.2614810647838486, 'quad', 2.9999933971457455, 0.09228520089886855)]
panel_c/v1 atan map gate [0.13831 0.0923  0.17481 0.11118 0.09842 0.08146 0.1268 ]
panel_c/v1 tanh map gate [0.13717 0.09155 0.12126 0.11885 0.1     0.08081 0.12589]
panel_c/v1 sin map gate [0.12997 0.12117 0.1769  0.11372 0.14342 0.07904 0.12112]
panel_c/v2 [(0.0001454771226085761, 'log', 0.35977292563249164, 0.00011268662481275705), (0.0002388661653393541, 'rat', 0.1605312675835919, 0.00018469988267643206), (0.00044672603140186856, 'cos', 1.260337735916968, 0.0003377155036753161), (0.006083583692798057, 'diag2', 2.9999933971457455, 0.0044720292112912655), (0.009532527242964136, 'quad', 2.9999933971457455, 0.0035945581993404407), (0.09889078106100126, 'lin', 2.9999933971457455, 0.09460388709979574), (0.11204805441932239, 'diag3', 2.9999933971457455, 0.0805315608176036), (0.11205190057633913, 'sin', 0.10000548795674928, 0.08053583068807588), (0.11207859319569953, 'tanh', 0.10000548795674928, 0.08056559222292548), (0.11209355563987501, 'atan', 0.10000548795674928, 0.08058238417932435)]
panel_c/v2 log map gate [0.25283 0.1397  0.23435 0.11613 0.23131 0.10187 0.16427]
panel_c/v2 rat map gate [0.25611 0.13943 0.2349  0.11597 0.23153 0.1016  0.16429]
panel_c/v2 cos map gate [0.26123 0.13894 0.23594 0.11566 0.23077 0.10211 0.16427]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=266e37bf42553964 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_266e37bf (directory).
```

### Turn 9 — `execute_in_remote_docker` (`35259 ms` LLM)
- **Rationale**: The nearly flat tanh gate can produce ill-conditioned columns. Test its polynomial limit and compact gated interactions with fold-local initialization; keep the accepted artifact unless a challenger preserves the public margin and improves the worst fold.
- **Alternatives Weighed**: `Replace the frozen affine panel map` (rejected: Its leave-one-out MAE is about 0.000085, and nonlinear replacements did not improve the response folds.)

```bash
cat > /tmp/axis01.py <<'PY'
exec(open('/tmp/profile01.py').read().split('ids=np.arange')[0])
ids=np.arange(len(y)); splits=[(ids[:52],ids[52:])]+[(ids[~np.isin(ids,v)],v) for v in np.array_split(np.random.default_rng(24).permutation(len(y)),6)]
def D(x,p,m):
 h=p[0]+x@p[1:]; A=affine(x); B=base(x)
 if m=='rational-linear': return np.column_stack([A,B/(1+h*h)[:,None]])
 if m=='rational-interaction': return np.column_stack([A,A*h[:,None]/(1+h*h)[:,None],h*h/(1+h*h)])
 if m=='rational-gate': return np.column_stack([B,A/(1+h*h)[:,None]])
 if m=='polynomial-limit':
  h=x@p[1:]; h=h/max(np.linalg.norm(p[1:]),1e-12)
  return np.column_stack([B,x*h[:,None]**2,h**4])
 if m=='atan-interaction': return np.column_stack([A,A*np.arctan(h)[:,None]])
 if m=='tanh-interaction': return np.column_stack([A,A*np.tanh(h)[:,None]])
 if m=='sin-interaction': return np.column_stack([A,A*np.sin(h)[:,None]])
 if m=='log-gate': return np.column_stack([B,A*np.log1p(h*h)[:,None]])
for m in ['rational-linear','rational-interaction','rational-gate','polynomial-limit','atan-interaction','tanh-interaction','sin-interaction','log-gate']:
 es=[]; rm=[]
 for t,v in splits:
  x=q[t]; yy=y[t]; cf=fit(base(x),yy); M=np.zeros((4,4))
  for k,(i,j) in enumerate(pairs): M[i,j]=M[j,i]=cf[k+5]/(1+(i!=j))
  e,V=np.linalg.eigh(M)
  starts=[np.r_[b,d*f] for b in [0.,.35] for d in [.45,1.3] for f in [V[:,0],V[:,-1]]]
  def resid(p):
   X=D(x,p,m);return X@fit(X,yy)-yy
  r=min((least_squares(resid,p,bounds=(-6,6),max_nfev=100,ftol=1e-7) for p in starts),key=lambda r:r.cost)
  X=D(q,r.x,m); es.append(np.mean(abs(X[v]@fit(X[t],yy)-y[v]))); rm.append(np.std(r.fun))
 print(m,'folds',np.round(es,5),'worst',round(max(es),6),'fit rmse',np.round(rm,5),flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/axis01.py
```
- **Container Exit Code**: `0` (`7111 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rational-linear folds [0.04345 0.01933 0.05844 0.03554 0.04326 0.02665 0.03875] worst 0.058437 fit rmse [0.01735 0.02653 0.01853 0.02097 0.02062 0.02615 0.02466]
rational-interaction folds [0.06433 0.0647  0.05317 0.03933 0.06001 0.03763 0.06957] worst 0.069575 fit rmse [0.04337 0.05548 0.04616 0.06139 0.04725 0.06231 0.04845]
rational-gate folds [0.03745 0.01978 0.04214 0.03679 0.04192 0.02101 0.03304] worst 0.042136 fit rmse [0.01903 0.02207 0.02106 0.01891 0.01776 0.02218 0.0212 ]
polynomial-limit folds [0.02893 0.01971 0.02398 0.03186 0.04216 0.01913 0.03503] worst 0.042163 fit rmse [0.01938 0.0223  0.02189 0.01938 0.01845 0.02274 0.02135]
atan-interaction folds [0.0667  0.06496 0.05383 0.04026 0.05983 0.03825 0.06917] worst 0.069172 fit rmse [0.04303 0.05529 0.04522 0.06115 0.04686 0.06214 0.04851]
tanh-interaction folds [0.066   0.0649  0.05257 0.03961 0.05983 0.03787 0.06945] worst 0.069449 fit rmse [0.04308 0.05541 0.04581 0.06132 0.04711 0.06225 0.04846]
sin-interaction folds [0.06372 0.0637  0.05563 0.03854 0.06078 0.03691 0.06989] worst 0.069886 fit rmse [0.04363 0.05564 0.04713 0.06161 0.04766 0.06248 0.04845]
log-gate folds [0.04135 0.01976 0.04333 0.03431 0.0419  0.02111 0.03149] worst 0.043328 fit rmse [0.01856 0.02193 0.02099 0.01911 0.01776 0.02211 0.02109]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=266e37bf42553964 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_266e37bf (directory).
```

### Turn 10 — `execute_in_remote_docker` (`35538 ms` LLM)
- **Rationale**: The factorized limit retains the calibrated interaction without near-collinear tanh columns. Promote it only if the exact artifact passes the isolated DAG test and improves the worst response fold without degrading trajectory accuracy.
- **Alternatives Weighed**: `Promote the rational gate` (rejected: Its exploratory worst-fold MAE was nearly identical, but its public MAE of 0.03745 left much less margin below 0.038.)

```bash
set -e
cp /holdout_task/node_01.py /tmp/node01_verified_gate.py
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/node_01.py')
s = p.read_text()
i = s.index('def _response_design')
j = s.index('def run_node', i)
s = s[:i] + '''def _response_design(x, direction):
    h = x @ (direction / max(np.linalg.norm(direction), 1e-12))
    h2 = h * h
    return np.column_stack((_quadratic(x), x * h2[:, None], h2 * h2))


def _response(x, xt, y):
    quadratic = _fit(_quadratic(x), y)[1 + x.shape[1]:]
    matrix = np.zeros((x.shape[1], x.shape[1]))
    for c, (i, j) in zip(quadratic, zip(*np.triu_indices(x.shape[1]))):
        matrix[i, j] = matrix[j, i] = c if i == j else c / 2
    _, axes = np.linalg.eigh(matrix)

    def residual(direction):
        a = _response_design(x, direction)
        return a @ _fit(a, y) - y

    # Use the finite polynomial limit of the gate, without cancelling nearly
    # collinear columns or extrapolating from large fitted coefficients.
    starts = [scale * axis for scale in (0.45, 1.3) for axis in (axes[:, 0], axes[:, -1])]
    best = min((least_squares(residual, d, bounds=(-6, 6), max_nfev=100, ftol=1e-7)
                for d in starts), key=lambda r: r.cost)
    return _response_design(xt, best.x) @ _fit(_response_design(x, best.x), y)


''' + s[j:]
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
if python3 -I /holdout_task/test_dag_candidate.py && python3 -I - <<'PY'
import copy, pickle, runpy
import numpy as np
root = '/holdout_task/'
frozen = runpy.run_path(root + '_frozen_nodes/node_00.py')['run_node']
sink = runpy.run_path(root + 'node_01.py')['run_node']
study = pickle.load(open(root + 'public_batch.pkl', 'rb'))[0]['study']
ids = np.arange(len(study['train']['y']))
errors = []
for v in np.array_split(np.random.default_rng(24).permutation(len(ids)), 6):
    t = np.setdiff1d(ids, v)
    s = copy.deepcopy(study)
    s['train'] = {k: study['train'][k][t] for k in ('x', 'y', 'traj')}
    s['test'] = {'x': study['train']['x'][v]}
    out = sink({'__input__': s, 'node_00': frozen({'__input__': s})})
    yh, z = out['y_hat'], out['state_pred']
    assert yh.shape == (len(v),) and z.shape == (len(v), len(s['eval_times']))
    assert np.all(np.isfinite(yh)) and np.all(np.isfinite(z))
    np.testing.assert_allclose(out['diagnostics_vec'], [yh.mean(), yh.std(), z[:, -1].mean(), z[:, -1].std()])
    errors.append((np.mean(abs(yh-study['train']['y'][v])), np.mean(abs(z-study['train']['traj'][v]))))
worst = np.max(errors, axis=0)
print('Six-fold worst MAE (response, trajectory):', worst)
assert worst[0] < 0.051270 and worst[1] <= 0.038
PY
then
  echo 'Promoted factorized polynomial response'
else
  cp /tmp/node01_verified_gate.py /holdout_task/node_01.py
  echo 'Restored the previously verified gated response'
  python3 -I /holdout_task/test_dag_candidate.py
fi
```
- **Container Exit Code**: `0` (`1026 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.028931, "public_cv_traj_mae": 0.016215}, "exec_counts": {"node_00": 1, "node_01": 1}}
Six-fold worst MAE (response, trajectory): [0.04216445 0.01664985]
Promoted factorized polynomial response

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8787b133cb0ca395 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_8787b133 (directory).
```

### Turn 11 — `submit_completed_task` (`15345 ms` LLM)
- **Rationale**: The exact saved artifact passed the isolated DAG test and improved worst-fold response error without changing the trajectory fit. Both nodes executed once and diagnostics matched the predictions.
- **Alternatives Weighed**: `Retain the tanh gate` (rejected: It had slightly lower public response MAE but higher worst shuffled-fold error.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_01.py`. The sink reuses the frozen calibration for response prediction and retains the trajectory fit. The DAG test accepted it: public MAE was 0.028931 for response and 0.016215 for trajectory, both below 0.038. Worst shuffled-fold MAE was 0.042164 and 0.016650, respectively.`

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
