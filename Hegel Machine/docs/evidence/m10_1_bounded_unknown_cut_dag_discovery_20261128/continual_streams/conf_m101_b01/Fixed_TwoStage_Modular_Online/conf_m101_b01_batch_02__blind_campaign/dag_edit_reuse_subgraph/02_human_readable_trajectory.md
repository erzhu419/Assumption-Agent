# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m101_b01_batch_02__blind_campaign__reuse_subgraph`

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
| **Turns Used / Max Turns** | `7 / 12` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `144,260` (`138,699` / `48,384` / `5,561`) |
| **Total Episode Wall Time (s)** | `187.36s` |

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

### Turn 1 — `execute_in_remote_docker` (`15774 ms` LLM)
- **Rationale**: Inspect the frozen interface, public scoring splits, and numeric data before fitting; this determines which nonlinear terms can be tested without leakage or changing the frozen node.

```bash
cd /holdout_task && cat _dag_spec.json _frozen_nodes/node_00.py test_dag_candidate.py validate_public_schema.py public_asset_index.json && python3 - <<'PY'
import pickle
import numpy as np
with open('public_batch.pkl','rb') as f: d=pickle.load(f)
def show(x,p='',depth=0):
 if isinstance(x,dict):
  for k,v in x.items(): show(v,p+'/'+str(k),depth+1)
 elif isinstance(x,(tuple,list)) and len(x) and isinstance(x[0],dict):
  print(p,type(x).__name__,len(x)); show(x[0],p+'/0',depth+1)
 else:
  a=np.asarray(x); print(p,a.shape,a.dtype,a.flatten()[:9])
show(d)
PY
```
- **Container Exit Code**: `0` (`373 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
} list 1
/0/study/eval_times (8,) float64 [0.25       0.64285714 1.03571429 1.42857143 1.82142857 2.21428571
 2.60714286 3.        ]
/0/study/panel_a/u (48, 4) float64 [ 0.80036867 -0.76044789 -1.25289052 -1.16071049  1.13700265  1.15375462
  0.78592456 -1.22828447  0.85872062]
/0/study/panel_a/v (48, 4) float64 [ 0.71113027 -1.06593429 -1.61984015 -0.94159856  0.99178054  1.18814884
  0.75992518 -1.00348474  0.75988964]
/0/study/panel_b/u (24, 4) float64 [ 0.90741822 -0.69598176  0.00853427 -1.0030622   0.94146051 -1.22609631
  1.03412501 -1.17995091 -0.73922224]
/0/study/panel_b/v (24, 4) float64 [ 0.99157692  0.82563659  0.21815816 -0.4136846   1.49023627  1.16870991
  0.51087633  0.36410365  0.27780287]
/0/study/panel_c/u (24, 4) float64 [-0.37942722 -0.30955677  0.15168976  0.82727855  0.86424031 -0.91112204
  0.43086574  1.2194831   1.14668514]
/0/study/panel_c/v1 (24, 4) float64 [-0.25853095 -0.25706253 -0.0286319  -0.10002862  1.34960207  0.85576255
 -0.61409995  0.38630303  0.02517648]
/0/study/panel_c/v2 (24, 4) float64 [ 0.60446143  0.21126426  0.24323781  0.19960193  0.93705733  0.35201132
 -0.38685882  0.62385045  0.15522145]
/0/study/train/x (72, 4) float64 [-0.85483327 -0.09306773  0.86840315 -0.0270259  -0.0582917   0.73444134
 -0.22690858  0.39721924  0.6460754 ]
/0/study/train/y (72,) float64 [-0.37249235  0.00650193 -1.19242068 -0.30968188  1.17130075  0.08896851
 -0.35157755  0.10868027 -0.42918852]
/0/study/train/traj (72, 8) float64 [-0.49774251 -1.000147   -1.31313961 -1.50079662 -1.65143629 -1.74545335
 -1.83043754 -1.89937379 -0.06364117]
/0/study/train/summary (4,) float64 [-0.02628809  0.51161853 -0.50602677  0.76692413]
/0/study/test/x (36, 4) float64 [ 0.46513481 -0.56538017 -1.36303234  1.10984785 -0.75329227  0.49991981
 -0.93530241 -0.61110211  0.71691646]
/0/study/telemetry_vec (4,) float64 [0.2 0.1 0.2 0.2]
```

### Turn 2 — `execute_in_remote_docker` (`30308 ms` LLM)
- **Rationale**: Compare affine and compact nonlinear bases by leave-one-out error, and inspect the trajectory spectrum to see whether temporal pooling can reduce the number of fitted parameters.
- **Alternatives Weighed**: `Fit Panel C inside the sink` (rejected: The frozen interface does not supply Panel C; use its public measurements only to identify plausible functional forms.)

```bash
cd /holdout_task && sed -n '40,180p' test_dag_candidate.py && cat >/tmp/probe_sink.py <<'PY'
import importlib.util,pickle,numpy as np,itertools
from scipy.linalg import qr
s=importlib.util.spec_from_file_location('f','/holdout_task/_frozen_nodes/node_00.py'); f=importlib.util.module_from_spec(s);s.loader.exec_module(f)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; o=f.run_node({'__input__':d}); z=o['z_tr']; y=o['train_y']; t=o['train_traj']; O=np.column_stack((y,t)); print('z ranges',z.min(0),z.max(0),'sv',np.linalg.svd(np.c_[np.ones(len(z)),z],compute_uv=False)); print('trajectory SV',np.linalg.svd(t,compute_uv=False));print('centered trajectory SV',np.linalg.svd(t-t.mean(0),compute_uv=False));print('joint SV',np.linalg.svd(O-O.mean(0),compute_uv=False));print('Y std',O.std(0)); print('time correlations',np.corrcoef(O.T)[0]);
def mats(x):
 n=len(x); xs=[np.ones(n)]; out={}
 for k in range(1,5):
  xs.extend(np.prod(x[:,p],axis=1) for p in itertools.combinations_with_replacement(range(x.shape[1]),k));out['poly'+str(k)]=np.column_stack(xs)
 for b in [.5,1,2,3]:
  for fn in [np.tanh,np.sin,np.cos]:
   for k in [1,2]:out[f'{fn.__name__}{b}/p{k}']=np.c_[out['poly'+str(k)],fn(b*x),np.cos(b*x) if fn==np.sin else np.empty((n,0))]
 return out
def stats(x,v):
 for key,A in mats(x).items():
  if A.shape[1]>=len(x)-3: continue
  P=np.linalg.pinv(A); E=v-A@(P@v); H=np.sum(A*P.T,axis=1); Q=E/(1-H).reshape((-1,)+(1,)*(v.ndim-1)); print(key,'fit',np.round(np.mean(abs(E),axis=0),5),'loo',np.round(np.mean(abs(Q),axis=0),5))
print('sink'); stats(z,O[:,[0,1,4,8]]); print('panel c v1');stats(d['panel_c']['u'],d['panel_c']['v1']);print('panel c v2');stats(d['panel_c']['u'],d['panel_c']['v2']);print('panel b affine residual',np.linalg.norm(d['panel_b']['v']-f._affine_features(d['panel_b']['u'])@np.linalg.lstsq(f._affine_features(d['panel_b']['u']),d['panel_b']['v'],rcond=None)[0])); print('panel C maps quadratic');x=d['panel_c']['u']; A=mats(x)['poly2'];print(np.round(np.linalg.lstsq(A,d['panel_c']['v1'],rcond=None)[0],3));print('sink quadratic coefficients');print(np.round(np.linalg.lstsq(mats(z)['poly2'],O[:,[0,1,4,8]],rcond=None)[0],3))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_sink.py
```
- **Container Exit Code**: `0` (`470 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
17 0.03513 0.04966] loo [0.09029 0.11381 0.056   0.0828 ]
tanh1/p2 fit [0.00108 0.0029  0.001   0.00231] loo [0.00741 0.0246  0.00669 0.01798]
sin1/p1 fit [0.00111 0.00119 0.00058 0.00133] loo [0.00289 0.00286 0.00131 0.0035 ]
cos1/p1 fit [0.00133 0.00167 0.00136 0.00155] loo [0.00232 0.00277 0.00228 0.00258]
cos1/p2 fit [1.4e-04 1.3e-04 1.6e-04 8.0e-05] loo [0.00121 0.00084 0.0011  0.00057]
tanh2/p1 fit [0.05302 0.06396 0.03685 0.05124] loo [0.09309 0.11183 0.05774 0.08435]
tanh2/p2 fit [0.0013  0.00303 0.00127 0.00248] loo [0.00862 0.02284 0.01014 0.01598]
sin2/p1 fit [0.00502 0.00556 0.0024  0.00511] loo [0.01238 0.01261 0.00565 0.01303]
cos2/p1 fit [0.00578 0.00704 0.00554 0.00613] loo [0.01005 0.01164 0.00919 0.01031]
cos2/p2 fit [7.e-05 7.e-05 8.e-05 4.e-05] loo [0.0006  0.00037 0.00053 0.00025]
tanh3/p1 fit [0.05521 0.06234 0.03802 0.05296] loo [0.09484 0.10913 0.06001 0.08676]
tanh3/p2 fit [0.00119 0.00311 0.0018  0.00266] loo [0.00688 0.02091 0.01315 0.01614]
sin3/p1 fit [0.01724 0.02127 0.00804 0.01857] loo [0.04018 0.04711 0.01689 0.04458]
cos3/p1 fit [0.02131 0.02717 0.02074 0.02627] loo [0.03655 0.04361 0.03323 0.04372]
cos3/p2 fit [9.e-05 6.e-05 7.e-05 6.e-05] loo [0.00073 0.00051 0.00053 0.00038]
panel b affine residual 0.0007127885475807389
panel C maps quadratic
[[-0.024 -0.002  0.107  0.134]
 [ 0.     0.    -0.     0.   ]
 [ 0.     0.    -0.    -0.   ]
 [ 0.     0.    -0.     0.   ]
 [-0.    -0.    -0.    -0.   ]
 [-0.    -0.     0.    -0.   ]
 [-1.156  0.308 -0.157  0.168]
 [-0.    -0.    -0.    -0.   ]
 [ 0.365  0.924 -0.132  0.693]
 [-0.    -0.     0.    -0.   ]
 [-0.067 -0.681  0.214  0.985]
 [ 0.    -0.     0.    -0.   ]
 [-0.    -0.     0.    -0.   ]
 [ 0.1   -0.268 -1.181  0.078]
 [-0.    -0.    -0.     0.   ]]
sink quadratic coefficients
[[ 3.100e-02 -9.100e-02 -2.970e-01 -3.910e-01]
 [ 1.000e-02 -2.000e-03  1.000e-03 -0.000e+00]
 [-4.000e-03  1.000e-03  1.000e-03 -0.000e+00]
 [ 3.000e-03 -1.000e-03  0.000e+00 -1.000e-03]
 [ 2.000e-03  1.000e-03 -1.000e-03  0.000e+00]
 [ 5.500e-02 -0.000e+00  3.000e-03  2.000e-03]
 [-9.510e-01  7.100e-02 -1.400e-02 -2.320e-01]
 [-2.000e-02 -2.000e-03  1.000e-03  0.000e+00]
 [ 5.710e-01 -4.680e-01 -1.067e+00 -1.000e+00]
 [ 1.000e-03 -1.000e-03  2.000e-03  2.000e-03]
 [ 1.113e+00 -3.890e-01 -1.399e+00 -1.985e+00]
 [ 3.000e-02 -5.000e-03 -8.000e-03  0.000e+00]
 [-2.400e-02 -1.000e-03  3.000e-03 -1.000e-03]
 [ 1.021e+00  3.840e-01  1.228e+00  1.611e+00]
 [ 2.500e-02 -2.000e-03 -1.000e-03  2.000e-03]]
```

### Turn 3 — `execute_in_remote_docker` (`20309 ms` LLM)
- **Rationale**: The public acceptance run will show whether the compact basis meets both MAE targets and the output contract; if only the readout fails, retain the trajectory model and test a nonlinear readout correction.
- **Alternatives Weighed**: `Full quadratic basis` (rejected: The probe shows that only four quadratic terms dominate; test whether those suffice before adding terms.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine basis in the frozen coordinates captures the shifted signal.` | Witness: `Panel C v1's quadratic fit has four large interaction rows, with affine and other quadratic coefficients near zero; the same four interactions dominate the sink fits across time.` | Blanket Diagnosis: `The frozen upstream coordinates retain the inputs needed for the shift, but an affine sink omits their cross-products.` | Sublated Pivot: `Test an intercept and the four products z0*z1, z0*z3, z1*z2, and z2*z3 for both outputs.`

```bash
set -e
cat >/holdout_task/node_01.py <<'PY'
import numpy as np


def _features(z):
    a, b, c, d = z.T
    return np.column_stack((np.ones(len(z)), a*b, a*d, b*c, c*d))


def run_node(inputs: dict) -> dict:
    data = inputs['node_00']
    fit = _features(data['z_tr'])
    targets = np.column_stack((data['train_y'], data['train_traj']))
    coef = np.linalg.lstsq(fit, targets, rcond=None)[0]
    pred = _features(data['z_te']) @ coef
    y, state = pred[:, 0], pred[:, 1:]
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`405 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.023984, "public_cv_traj_mae": 0.003983}, "exec_counts": {"node_00": 1, "node_01": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5dc0ea0b72125aa9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5dc0ea0b (directory).
```

### Turn 4 — `execute_in_remote_docker` (`23297 ms` LLM)
- **Rationale**: Stress-test the accepted artifact on disjoint random folds and a feature-defined interaction tail. If the tail exposes a readout bottleneck, investigate a correction for that output without replacing the accurate trajectory model.
- **Alternatives Weighed**: `Submit on the public split alone` (rejected: The readout is closer to tolerance than the trajectory, and the public split does not isolate large interaction magnitudes.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_node

upstream = load('upstream', '/holdout_task/_frozen_nodes/node_00.py')
sink = load('sink', '/holdout_task/node_01.py')
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    cases = pickle.load(f)
for case in cases:
    data = upstream({'__input__': case['study']})
    z, y, traj = (data[k] for k in ('z_tr', 'train_y', 'train_traj'))
    n = len(z)
    rng = np.random.default_rng(713)
    folds = {f'random_{i}': rows for i, rows in enumerate(np.array_split(rng.permutation(n), 4))}
    a,b,c,d = z.T
    strength = np.linalg.norm(np.column_stack((a*b,a*d,b*c,c*d)), axis=1)
    folds['interaction_tail'] = np.argsort(strength)[-n//4:]
    for name, test in folds.items():
        train = np.ones(n, dtype=bool)
        train[test] = False
        out = sink({'node_00': dict(data, z_tr=z[train], train_y=y[train], train_traj=traj[train], z_te=z[test])})
        dy = np.abs(out['y_hat'] - y[test])
        dt = np.abs(out['state_pred'] - traj[test])
        print(f'{name}: y_mae={dy.mean():.6f}, traj_mae={dt.mean():.6f}, worst_time_mae={dt.mean(axis=0).max():.6f}')
PY
```
- **Container Exit Code**: `0` (`340 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
random_0: y_mae=0.021365, traj_mae=0.003853, worst_time_mae=0.004873
random_1: y_mae=0.040136, traj_mae=0.004291, worst_time_mae=0.006580
random_2: y_mae=0.029408, traj_mae=0.004502, worst_time_mae=0.005837
random_3: y_mae=0.034233, traj_mae=0.004361, worst_time_mae=0.005312
interaction_tail: y_mae=0.062580, traj_mae=0.004780, worst_time_mae=0.006221

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5dc0ea0b72125aa9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5dc0ea0b (directory).
```

### Turn 5 — `execute_in_remote_docker` (`32397 ms` LLM)
- **Rationale**: Compare curvature and saturation candidates on identical splits, with coefficients fit only on training rows. Use the results to select a compact readout for separate validation rather than promoting a model on its in-sample fit.
- **Alternatives Weighed**: `Expand the trajectory model` (rejected: Its worst horizon MAE across the stress slices was 0.006580, well within tolerance.)
- **ATMS NOGOOD DECLARED**: Refuted `A readout linear in the four interaction terms stays within 0.038 MAE on the large-interaction slice.` | Witness: `The interaction tail gives readout MAE 0.062580 versus trajectory MAE 0.004780; another random slice gives readout MAE 0.040136.` | Blanket Diagnosis: `The trajectory fit supports retaining the four interaction coordinates; the readout may require an additional nonlinear term.` | Sublated Pivot: `Compare compact nonlinear readouts on the existing interaction coordinates before changing the sink artifact.`

```bash
cat >/tmp/probe_readout.py <<'PY'
import itertools
import pickle
import runpy
import numpy as np

st = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
d = runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':st})
z, y = d['z_tr'], d['train_y']
n = len(y)
a,b,c,e = z.T
q = np.column_stack((a*b,a*e,b*c,c*e))
base = np.c_[np.ones(n),q]
coef = np.linalg.lstsq(base,y,rcond=None)[0]
r = y-base@coef
folds = [np.arange(n-20,n), *np.array_split(np.random.default_rng(713).permutation(n),4), np.argsort(np.linalg.norm(q,axis=1))[-n//4:]]

def evaluate(name,X,target=y,inverse=lambda p:p):
    errs=[]
    for te in folds:
        tr=np.setdiff1d(np.arange(n),te)
        p=inverse(X[te]@np.linalg.lstsq(X[tr],target[tr],rcond=None)[0])
        errs.append(np.mean(np.abs(y[te]-p)))
    full=inverse(X@np.linalg.lstsq(X,target,rcond=None)[0]); print(name,X.shape[1], 'fit',round(np.mean(abs(y-full)),6),'slices',np.round(errs,6),'worst',round(max(errs),6))

def monomials(X,k):
    return np.column_stack([np.prod(X[:,i],axis=1) for i in itertools.combinations_with_replacement(range(X.shape[1]),k)])

print('response range', y.min(),y.max(),'baseline residual range',r.min(),r.max())
evaluate('bilinear',base)
evaluate('full quadratic z',np.c_[np.ones(n),z,monomials(z,2)])
for degree in [2,3]:
    evaluate(f'polynomial q degree {degree}',np.c_[base,*[monomials(q,k) for k in range(2,degree+1)]])
for freq in [.35,.7,1.,1.4,2.]:
    evaluate(f'q sincos {freq}',np.c_[base,np.sin(freq*q),np.cos(freq*q)])
for scale in [1.5,2.,3.,4.,6.,10.,20.]:
    if scale>np.max(np.abs(y)):
        evaluate(f'scaled tanh {scale}',base,np.arctanh(y/scale),lambda p:scale*np.tanh(p))
np.set_printoptions(precision=4,suppress=True)
print('bilinear coefficients',coef)
print('largest residuals: q, observed, fitted, error')
ii=np.argsort(abs(r))[-12:];print(np.c_[q,y,y-r,r][ii])
print('quadratic q coefficients',np.linalg.lstsq(np.c_[base,monomials(q,2)],y,rcond=None)[0])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/probe_readout.py
```
- **Container Exit Code**: `0` (`365 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
it 0.003426 slices [0.018981 0.013227 0.01572  0.008821 0.035313 0.041845] worst 0.041845
q sincos 0.35 13 fit 0.01288 slices [0.022922 0.01612  0.017414 0.016084 0.042858 0.103955] worst 0.103955
q sincos 0.7 13 fit 0.012926 slices [0.022986 0.016087 0.017541 0.0161   0.039978 0.10046 ] worst 0.10046
q sincos 1.0 13 fit 0.012994 slices [0.023064 0.01604  0.017714 0.016121 0.036183 0.095807] worst 0.095807
q sincos 1.4 13 fit 0.013129 slices [0.023181 0.016    0.018025 0.016159 0.029642 0.08764 ] worst 0.08764
q sincos 2.0 13 fit 0.013427 slices [0.023268 0.016015 0.018623 0.016233 0.018097 0.072612] worst 0.072612
scaled tanh 2.0 5 fit 0.05844 slices [0.073794 0.069664 0.072692 0.080608 0.064977 0.100441] worst 0.100441
scaled tanh 3.0 5 fit 0.033563 slices [0.035179 0.033385 0.054037 0.039514 0.038854 0.080724] worst 0.080724
scaled tanh 4.0 5 fit 0.02922 slices [0.029424 0.027167 0.047903 0.033891 0.036564 0.073078] worst 0.073078
scaled tanh 6.0 5 fit 0.026529 slices [0.026244 0.023595 0.043578 0.030789 0.035104 0.067345] worst 0.067345
scaled tanh 10.0 5 fit 0.025596 slices [0.024772 0.022045 0.041374 0.029619 0.034429 0.064314] worst 0.064314
scaled tanh 20.0 5 fit 0.025283 slices [0.024178 0.021533 0.040445 0.029461 0.034255 0.063016] worst 0.063016
bilinear coefficients [ 0.05   -0.9643  0.5965  1.146   1.0163]
largest residuals: q, observed, fitted, error
[[ 0.2128 -0.2732 -0.1408  0.1807 -0.2571 -0.2958  0.0387]
 [-0.1696 -0.5195  0.0291  0.0892  0.0753  0.0276  0.0477]
 [-0.3901 -0.1675  0.336   0.1442  0.8101  0.858  -0.0479]
 [-0.0297 -0.205  -0.2061 -1.4227 -1.7795 -1.7257 -0.0538]
 [-0.2436  0.0459 -0.7172  0.135  -0.4292 -0.3723 -0.0569]
 [ 0.1335 -0.4412 -0.0736  0.2432 -0.1186 -0.1791  0.0604]
 [-0.1255  0.4363 -0.1499  0.5213  0.7214  0.7894 -0.068 ]
 [ 0.8957  0.374   0.1307  0.0546 -0.3097 -0.3853  0.0756]
 [-0.6844 -0.4466 -0.1272 -0.083   0.2897  0.2135  0.0763]
 [-0.1097 -0.6741  0.0406  0.2493  0.1362  0.0536  0.0827]
 [-0.6485 -0.5147 -0.1118 -0.0887  0.2494  0.1501  0.0992]
 [-0.2472  1.4913  0.034  -0.2053  1.1713  1.0084  0.1629]]
quadratic q coefficients [ 0.0401 -0.9383  0.5404  1.0929  1.0064  0.0392  0.0886  0.0008  0.1678
  0.1091  0.1678 -0.1021 -0.1308 -0.0297 -0.0255]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5dc0ea0b72125aa9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5dc0ea0b (directory).
```

### Turn 6 — `execute_in_remote_docker` (`34113 ms` LLM)
- **Rationale**: Promote the quadratic readout, remove its algebraically duplicate term, and retain the accurate trajectory basis. The public test checks the exact artifact and output contract; signed interaction tails test whether the improvement holds when one product extrapolates. A failure there would motivate simplifying the readout rather than changing the trajectory.
- **Alternatives Weighed**: `Cubic readout in the interaction coordinates` (rejected: It improved training fit but increased the worst held-out readout MAE to 0.041845.)

```bash
set -e
cat >/holdout_task/node_01.py <<'PY'
import numpy as np


def _features(z):
    a, b, c, d = z.T
    return np.column_stack((np.ones(len(z)), a*b, a*d, b*c, c*d))


def _readout_features(base):
    q = base[:, 1:]
    # q0*q3 and q1*q2 are the same monomial; include it only once.
    quadratic = [q[:, i]*q[:, j] for i in range(4) for j in range(i, 4)
                 if (i, j) != (1, 2)]
    return np.column_stack((base, *quadratic))


def run_node(inputs: dict) -> dict:
    data = inputs['node_00']
    train = _features(data['z_tr'])
    test = _features(data['z_te'])
    y_coef = np.linalg.lstsq(_readout_features(train), data['train_y'], rcond=None)[0]
    state_coef = np.linalg.lstsq(train, data['train_traj'], rcond=None)[0]
    y = _readout_features(test) @ y_coef
    state = test @ state_coef
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/test_dag_candidate.py
python3 -I - <<'PY'
import pickle
import runpy
import numpy as np
root = '/holdout_task/'
up = runpy.run_path(root + '_frozen_nodes/node_00.py')['run_node']
mod = runpy.run_path(root + 'node_01.py')
with open(root + 'public_batch.pkl', 'rb') as f:
    cases = pickle.load(f)
for case in cases:
    d = up({'__input__': case['study']})
    z, y, traj = (d[k] for k in ('z_tr', 'train_y', 'train_traj'))
    n = len(z)
    q = mod['_features'](z)[:, 1:]
    folds = {f'random_{i}': rows for i, rows in enumerate(np.array_split(np.random.default_rng(713).permutation(n), 4))}
    folds['interaction_tail'] = np.argsort(np.linalg.norm(q, axis=1))[-(n//4):]
    for j in range(q.shape[1]):
        order = np.argsort(q[:, j])
        folds[f'q{j}_low'] = order[:n//8]
        folds[f'q{j}_high'] = order[-(n//8):]
    metrics = []
    for name, te in folds.items():
        tr = np.setdiff1d(np.arange(n), te)
        out = mod['run_node']({'node_00': dict(d, z_tr=z[tr], train_y=y[tr], train_traj=traj[tr], z_te=z[te])})
        dy = np.mean(np.abs(out['y_hat'] - y[te]))
        dt = np.abs(out['state_pred'] - traj[te])
        metrics.append((dy, dt.mean(), dt.mean(axis=0).max()))
        print(f'{name}: y_mae={dy:.6f}, traj_mae={dt.mean():.6f}, worst_time_mae={dt.mean(axis=0).max():.6f}')
    print('Worst stress metrics (readout, trajectory, horizon):', np.max(metrics, axis=0))
PY
```
- **Container Exit Code**: `0` (`477 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.006051, "public_cv_traj_mae": 0.003983}, "exec_counts": {"node_00": 1, "node_01": 1}}
random_0: y_mae=0.005132, traj_mae=0.003853, worst_time_mae=0.004873
random_1: y_mae=0.006914, traj_mae=0.004291, worst_time_mae=0.006580
random_2: y_mae=0.005017, traj_mae=0.004502, worst_time_mae=0.005837
random_3: y_mae=0.005355, traj_mae=0.004361, worst_time_mae=0.005312
interaction_tail: y_mae=0.011524, traj_mae=0.004780, worst_time_mae=0.006221
q0_low: y_mae=0.004593, traj_mae=0.004308, worst_time_mae=0.005937
q0_high: y_mae=0.009982, traj_mae=0.004771, worst_time_mae=0.007185
q1_low: y_mae=0.005084, traj_mae=0.003821, worst_time_mae=0.006585
q1_high: y_mae=0.006442, traj_mae=0.004755, worst_time_mae=0.006482
q2_low: y_mae=0.006675, traj_mae=0.004027, worst_time_mae=0.005573
q2_high: y_mae=0.007154, traj_mae=0.003696, worst_time_mae=0.005261
q3_low: y_mae=0.009926, traj_mae=0.004479, worst_time_mae=0.006580
q3_high: y_mae=0.005349, traj_mae=0.004681, worst_time_mae=0.006049
Worst stress metrics (readout, trajectory, horizon): [0.01152447 0.00477953 0.00718538]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=61d59544baa66a57 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_61d59544 (directory).
```

### Turn 7 — `submit_completed_task` (`14238 ms` LLM)
- **Rationale**: The exact saved artifact passed isolated public verification with one execution per node. Random folds and signed interaction tails confirmed the readout improvement without sacrificing trajectory accuracy; the worst individual horizon MAE was 0.007185.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_01.py` without changing the frozen upstream node. Public acceptance passed: readout MAE was 0.006051 and trajectory MAE was 0.003983. Worst stress-slice MAEs were 0.011524 and 0.004780, respectively, both below 0.038.`

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
