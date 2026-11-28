# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m102_a04_batch_06__blind_campaign__rewire_edge`

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
| **Turns Used / Max Turns** | `12 / 12` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `340,655` (`331,304` / `110,720` / `9,351`) |
| **Total Episode Wall Time (s)** | `311.38s` |

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

### Turn 1 — `execute_in_remote_docker` (`14675 ms` LLM)
- **Rationale**: Inspect the frozen node contracts, public measurements, archive, and validation harness before choosing a correction model.

```bash
cd /holdout_task && cat _dag_spec.json && for f in _frozen_nodes/*.py; do echo "=== $f"; cat "$f"; done && cat test_dag_candidate.py && cat public_asset_index.json; echo '=== archive'; ls -la /tmp/hegel_continual_archive; find /tmp/hegel_continual_archive -maxdepth 3 -type f | head -100; python3 - <<'PY'
import pickle,numpy as np
with open('/holdout_task/public_batch.pkl','rb') as f: obj=pickle.load(f)
def show(o,p='',depth=0):
 if isinstance(o,dict):
  for k,v in o.items():show(v,p+'/'+str(k),depth+1)
 elif isinstance(o,(list,tuple)):
  print(p,type(o).__name__,len(o));
  if len(o) and depth<5:show(o[0],p+'/0',depth+1)
 elif isinstance(o,np.ndarray): print(p,o.shape,o.dtype,o.ravel()[:8])
 else: print(p,type(o).__name__,str(o)[:150])
show(obj)
PY
```
- **Container Exit Code**: `0` (`460 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
}=== archive
total 20
drwxr-xr-x 2 root root 4096 Nov 28 15:18 .
drwxrwxrwt 1 root root 4096 Nov 28 15:18 ..
-rw-r--r-- 1 root root  665 Nov 28 15:18 archive_manifest.json
-rw-r--r-- 1 root root 3047 Nov 28 15:18 conf_m102_a04_batch_00_solver.py
-rw-r--r-- 1 root root 1883 Nov 28 15:18 pre_llm_gate_diagnostic.json
/tmp/hegel_continual_archive/archive_manifest.json
/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
/tmp/hegel_continual_archive/conf_m102_a04_batch_00_solver.py
 list 1
/0/study/eval_times (8,) float64 [0.25       0.64285714 1.03571429 1.42857143 1.82142857 2.21428571
 2.60714286 3.        ]
/0/study/panel_a/u (48, 4) float64 [-1.29701026 -0.04066809 -0.60967371 -0.28307769  1.27005498  0.14940955
  0.35722784  0.1331559 ]
/0/study/panel_a/v (48, 4) float64 [-1.55752683  0.24288813 -0.56046787  0.08004428  1.6205017   0.10762403
  0.42887355 -0.30124659]
/0/study/panel_b/u (24, 4) float64 [ 0.03308133  0.33539934 -0.64751427  0.08408179 -0.55947656 -0.20782241
 -0.74762559  0.23681037]
/0/study/panel_b/v (24, 4) float64 [ 0.14466333 -0.4735461  -0.05169901  0.02350604  0.57553391 -0.02538713
 -0.38007569 -0.15492722]
/0/study/panel_c/u (24, 4) float64 [-1.14472422 -0.26578466 -0.2873813   1.14906602  0.61483149  0.41229329
  0.90892923  0.36632355]
/0/study/panel_c/v1 (24, 4) float64 [-0.34652136 -0.10074688 -0.80611815 -1.36413173  0.6803181   0.22222164
 -0.24202756  0.31887872]
/0/study/panel_c/v2 (24, 4) float64 [ 0.92072729 -1.17942475  0.51321008  0.50330111 -0.86214076  0.20291513
  0.82377775 -0.34694649]
/0/study/train/x (72, 4) float64 [-0.68240645 -0.59634434  1.59089925 -0.01668827  0.41853795  0.13772041
  0.27438115  1.05718587]
/0/study/train/y (72,) float64 [-0.95878876 -0.70214468 -0.76435985 -0.82834449 -0.17207227  1.99385077
  2.16208048  0.12331322]
/0/study/train/traj (72, 8) float64 [0.19114252 0.50406698 0.73567401 0.94159428 1.10263196 1.22705161
 1.34404057 1.43130074]
/0/study/train/summary (4,) float64 [0.0108373  1.07333499 0.49537581 1.66956448]
/0/study/test/x (36, 4) float64 [ 0.8028258  -0.07162589  0.31391073 -0.94912094  0.35270078 -0.79598086
  0.04446585  0.82442812]
/0/study/telemetry_vec (4,) float64 [0.22 1.25 1.25 1.25]
```

### Turn 2 — `execute_in_remote_docker` (`17878 ms` LLM)
- **Rationale**: Recover the upstream contracts and archived model structure omitted by the truncated inspection before choosing a correction basis.

```bash
cd /holdout_task && cat _frozen_nodes/node_01.py _frozen_nodes/node_02.py; echo '=== node_00 contract'; tail -55 /holdout_task/_frozen_nodes/node_00.py; echo '=== archive'; cat /tmp/hegel_continual_archive/*
```
- **Container Exit Code**: `0` (`262 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
n_stack((train['y'], train['traj']))
    eye = np.eye(n)
    fit = np.mean([
        kernel[n:, :n] @ np.linalg.solve(kernel[:n, :n] + ridge*eye, targets)
        for kernel, ridge in ((quadratic, 3e-5),
                              (quadratic + np.exp(-.06*distance), 3e-4))
    ], axis=0)
    y = fit[:, 0]
    basis = _time_basis(np.asarray(study['eval_times']), train['traj'])
    state = (basis @ lin(basis, fit[:, 1:].T)).T
    terminal = state[:, -1]
    diagnostics = np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
{
  "accepted": false,
  "schema_valid": true,
  "execution_status": "completed",
  "failure_kind": "scientific_or_numerical",
  "errors": [
    "public_cv_y_mae_exceeded:1.870574",
    "public_cv_traj_mae_exceeded:1.238196"
  ],
  "public_input_sha256": "9f6e838f894b9d419d8dd1ccae031120004de05150cc363d96afeb9b869af9c3",
  "candidate_raw_sha256": "b1d76ad0d6b95765332c4e22c18552c7bfefb956b3a7faed171dd78782d4dd5e",
  "setup_source_sha256": "9334070bcd6d909b0aebb5cc5c840b3fca347906a37d655e277d01a3c9b2a8ec",
  "cv_metrics": {
    "public_cv_y_mae": 1.870574,
    "public_cv_traj_mae": 1.238196
  },
  "dag_runtime_trace": {
    "active_node_count": 4,
    "all_declared_edges_consumed": true,
    "all_exec_count_eq_1": true,
    "dag_id": "dag_04",
    "depth": 3,
    "edge_consumption_checks": {
      "node_00": [
        "__input__"
      ],
      "node_01": [
        "__input__"
      ],
      "node_02": [
        "node_00",
        "node_01"
      ],
      "node_03": [
        "node_01",
        "node_02"
      ]
    },
    "exec_counts": {
      "node_00": 1,
      "node_01": 1,
      "node_02": 1,
      "node_03": 1
    },
    "max_fanout": 2,
    "node_code_shas": {
      "node_00": "707822a36ae53a2cad8733482798bf85a2542b4507c759710f3b90ede4a16787",
      "node_01": "493aca178d528c16ec681d5e109ba4e4500ff58bbef4f1e9c17dd6aa7ee50743",
      "node_02": "9d9cf57d412595efc361c244c11a554b623747129b330a44ce7601a11d8fcffd",
      "node_03": "eeeaeecd9a19626365176f029b400d14a57213e504550a9bef596c3f286941ff"
    },
    "non_identity_checks": {
      "node_00": true,
      "node_01": true,
      "node_02": true
    },
    "shared_fanout_nodes": [
      "node_01"
    ],
    "topo_order": [
      "node_00",
      "node_01",
      "node_02",
      "node_03"
    ]
  },
  "kernel_version": "hegel_shared_math_handoff_kernel_v3_m961_three_layer",
  "wall_ms": 3141.92
}
```

### Turn 3 — `execute_in_remote_docker` (`15366 ms` LLM)
- **Rationale**: Read the exact feature construction and fold design in a short, focused inspection before testing which algebraic terms explain both outputs.

```bash
cd /holdout_task; cat test_dag_candidate.py; tail -38 _frozen_nodes/node_00.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling variable projection sparse symbolic regression' 2>&1 | head -25
```
- **Container Exit Code**: `0` (`373 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
n=False))
sys.exit(0 if ok else 1)
    z0 = np.linalg.solve(linear, (observed-o).T).T
    def forward(z):
        w = z[j]
        return o + d*z + e*w + g*z**3 + h*z*w
    def jac(z):
        a = np.diag(d + 3*g*z*z + h*z[j])
        a[np.arange(p), j] += e + h*z
        return a
    return np.stack([least_squares(lambda z: forward(z)-y, z0i,
                                   jac=jac, gtol=1e-12).x
                     for y, z0i in zip(observed, z0)])


def _interactions(u):
    return np.column_stack((np.ones(len(u)), u*np.roll(u, -1, axis=1)))


def _curvature(u, q):
    w = u*u
    return np.column_stack((np.ones(len(u)), w, w/(1+q*w)))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']['x']
    n = len(train)
    z = _calibrate(study['panel_a'], np.vstack((train, study['test']['x'])))
    c = study['panel_c']
    u, v1, v2 = (np.asarray(c[k], dtype=float) for k in ('u', 'v1', 'v2'))
    first = _interactions(z) @ _linear(_interactions(u), v1)
    def residual(q):
        f = _curvature(u, q[0])
        return (f @ _linear(f, v2) - v2).ravel()
    q = least_squares(residual, [.25], bounds=(.02, 3.), gtol=1e-12).x[0]
    second = _curvature(z, q) @ _linear(_curvature(u, q), v2)
    features = np.column_stack((first, second))
    features = (features - features[:n].mean(0)) / np.maximum(features[:n].std(0), .1)
    return {'features': features, 'n_train': n}
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
```

### Turn 4 — `execute_in_remote_docker` (`26376 ms` LLM)
- **Rationale**: Compare the frozen predictions with explicit linear and nonlinear bases across several holdouts; inspect calibration ranges and temporal rank to distinguish missing structure from unstable extrapolation.

```bash
sed -n '65,190p' /holdout_task/test_dag_candidate.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 2>&1 | head -12; cat > /tmp/explore03.py <<'PY'
import sys,pickle,numpy as np, importlib.util
from itertools import combinations_with_replacement
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00 as a,node_01 as b,node_02 as c
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
tr=s['train']; y=np.c_[tr['y'],tr['traj']]; N=len(y)
z=a._calibrate(s['panel_a'],np.r_[tr['x'],s['test']['x']]); print('z range',z.min(0),z.max(0)); print('z abs maxima train rows',np.argsort(np.max(abs(z[:N]),axis=1))[-10:]); print('targets range',y.min(0),y.max(0)); print('traj singular values',np.linalg.svd(y[:,1:],compute_uv=False))
for n in (48,50,54,56,60):
 t=s.copy(); t['train']={k:v[:n] for k,v in tr.items() if k!='summary'};t['test']={'x':tr['x'][n:]}; d=a.run_node({'__input__':t}); e=b.run_node({'__input__':t}); f=c.run_node({'node_00':d,'node_01':e})['fit']; w=d['features']; print('split',n,'features abs max',np.max(abs(w),axis=0))
 def check(name,x):
  g=abs(x-y[n:]); print(name,np.round([g[:,0].mean(),g[:,1:].mean(),g[:,0].max(),g[:,1:].mean(1).max()],5))
 check('frozen',f)
 for name,v in [('w',w),('w1',w[:,:4]),('w2',w[:,4:]),('z',z[:N]),('even',np.c_[z[:N]**2,z[:N]*np.roll(z[:N],-1,axis=1)])]:
  for deg in (1,2,3):
   if deg==3 and v.shape[1]>4:continue
   X=np.column_stack([np.ones(N)]+[np.prod(v[:,inds],axis=1) for h in range(1,deg+1) for inds in combinations_with_replacement(range(v.shape[1]),h)])
   check(name+str(deg),X[n:]@np.linalg.lstsq(X[:n],y[:n],rcond=1e-9)[0])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/explore03.py
```
- **Container Exit Code**: `0` (`785 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1739 1.14416 6.5765  4.40407]
w1 [0.79756 0.54759 2.95593 2.18674]
w2 [ 3.982    2.63866 11.63732  7.70286]
w11 [0.80272 0.98952 1.8698  2.94487]
w12 [1.12705 1.2383  5.0315  5.6248 ]
w13 [ 1.54287  2.63542  7.24934 12.70469]
w21 [0.69891 0.97814 2.86146 3.16678]
w22 [1.00668 1.12757 4.77257 4.39244]
w23 [ 1.8425   0.9297  10.05218  3.15001]
z1 [0.88182 1.20304 1.86488 2.79361]
z2 [0.59171 0.45932 2.32777 1.36343]
z3 [0.9393  0.61037 3.89406 2.29251]
even1 [0.84392 0.59773 3.04117 2.39095]
even2 [2.36013 1.59358 7.01123 4.85114]
split 54 features abs max [2.94548017 2.87531084 3.50241039 2.9544036  2.06908429 2.26297384
 2.3407881  2.77191202]
frozen [1.85309 1.23362 5.62991 3.76976]
w1 [0.88348 0.60554 2.98959 2.2205 ]
w2 [3.39237 2.25915 8.48954 5.67023]
w11 [0.91766 1.08674 1.86228 2.95878]
w12 [1.26182 1.42434 4.60828 5.46745]
w13 [ 1.55305  2.55099  6.38475 10.37875]
w21 [0.70307 1.05981 2.79026 3.16797]
w22 [0.93677 1.21112 4.52992 4.16339]
w23 [1.87979 1.03684 9.68401 3.8826 ]
z1 [0.79708 1.286   1.52403 2.72933]
z2 [0.6799  0.53912 2.36449 1.37432]
z3 [1.0262  0.63606 3.72188 2.44168]
even1 [0.91439 0.64668 3.05376 2.42574]
even2 [2.37977 1.58996 8.21316 5.54305]
split 56 features abs max [2.98973364 2.91364639 3.37856215 3.01258899 2.11017682 2.27174018
 2.33574853 2.67777599]
frozen [1.63226 1.08547 3.55241 2.38572]
w1 [0.96846 0.66316 2.97218 2.207  ]
w2 [1.85994 1.23546 3.68584 2.45581]
w11 [0.97821 1.1411  1.89749 3.00463]
w12 [1.35429 1.42811 4.55362 5.31067]
w13 [ 1.47475  2.89659  6.17111 10.48053]
w21 [0.72707 1.14469 2.79559 3.21062]
w22 [0.97587 1.31914 4.5014  4.16121]
w23 [2.02145 1.13556 9.66273 3.93866]
z1 [0.78955 1.37259 1.51819 2.71857]
z2 [0.75122 0.5903  2.36262 1.36232]
z3 [1.15342 0.71096 3.71419 2.42466]
even1 [1.01248 0.70012 3.03705 2.38476]
even2 [ 2.80569  1.88286 10.72105  7.1775 ]
split 60 features abs max [3.09487261 2.76608599 3.25511457 2.92260014 1.99098853 2.32543019
 2.35106027 2.64256306]
frozen [0.81465 0.54413 2.83683 1.8566 ]
w1 [0.74387 0.50015 2.27386 1.5199 ]
w2 [1.00442 0.66702 4.27954 2.83937]
w11 [0.86253 1.07527 1.87238 2.76062]
w12 [0.85355 1.00487 1.42353 2.92552]
w13 [0.94161 1.99916 2.6075  6.4181 ]
w21 [0.6466  1.0335  2.27873 2.5002 ]
w22 [0.84605 1.05898 2.73298 2.74391]
w23 [0.91913 1.03006 2.82072 2.91238]
z1 [0.75269 1.50463 1.45424 2.80714]
z2 [0.63762 0.54386 2.62092 1.53713]
z3 [1.24492 0.63727 3.49042 1.99878]
even1 [0.79326 0.58581 2.61332 1.50377]
even2 [1.62689 1.0773  4.39491 2.99243]
```

### Turn 5 — `execute_in_remote_docker` (`24909 ms` LLM)
- **Rationale**: The errors could reflect either information lost upstream or an omitted nonlinear law. Calibration checks and nearby pairs in signed and sign-quotient coordinates help distinguish those cases.
- **Alternatives Weighed**: `Increase polynomial degree on standardized features` (rejected: Cubic fits already worsened extrapolation on several chronological splits.)
- **ATMS NOGOOD DECLARED**: Refuted `A low-degree polynomial in the frozen features is sufficient to forecast both outputs.` | Witness: `At the 48-row split the full linear basis gave MAEs 0.79818 and 0.54144; quadratic expansion of either feature block failed to approach 0.038.` | Blanket Diagnosis: `The frozen features contain only even transformations under simultaneous latent sign reversal, and their affine mixing can obscure low-dimensional structure.` | Sublated Pivot: `Audit calibration error, near-antipodal pairs, output coupling, and latent coordinate correlations before selecting a more structured model.`

```bash
sed -n '128,185p' /holdout_task/test_dag_candidate.py; cat /tmp/hegel_continual_archive/conf_m102_a04_batch_00_solver.py; cat > /tmp/probe03.py <<'PY'
exec(open('/tmp/explore03.py').read().split('for n in (48')[0])
np.set_printoptions(precision=3,suppress=True,linewidth=180)
P=s['panel_a']; Z=a._calibrate(P,P['v']); print('calibration errors',np.quantile(abs(Z-P['u']),[0,.5,.8,.9,.95,.99,1],axis=0));print('panelB');print(np.c_[s['panel_b']['u'],s['panel_b']['v']][:12]); print('samples z,y,traj');print(np.c_[z[:N],y[:,0],y[:,1],y[:,-1]]);print('time correlations',np.corrcoef(y.T)[0]);
d=a.run_node({'__input__':s});w=d['features'];print('feature singulars',np.linalg.svd(w[:N],compute_uv=False));print('interaction dim affine',np.linalg.lstsq(a._interactions(z),w[:,:4],rcond=None)[1]);
for v in (z[:N],abs(z[:N]),z[:N]**2,z[:N]*np.roll(z[:N],-1,axis=1),w[:N], np.c_[tr['x']]):
 print('corr',np.round(np.corrcoef(np.c_[v,y[:,0],y[:,1],y[:,-1]].T)[:-3,-3:],2).T)
from scipy.spatial.distance import cdist
for v in (z[:N],z[:N]*np.sign(z[:N,:1]),w[:N]):
 m=cdist(v,v);np.fill_diagonal(m,np.inf);k=m.argmin(1);i=np.argsort(m.min(1))[:10];print('nearest', list(zip(i,k[i],m[i,k[i]].round(3),y[i,0].round(3),y[k[i],0].round(3),y[i,-1].round(3),y[k[i],-1].round(3))))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe03.py
```
- **Container Exit Code**: `0` (`605 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
8]
 [-1.109 -0.427  0.541 -0.617 -1.119  0.154  1.162]
 [ 1.072 -1.139 -0.308 -0.872 -1.578 -0.674 -2.344]
 [ 1.033 -0.558  0.322  0.622  1.21   1.002  3.807]
 [-1.051  0.426  0.15  -0.739 -0.252  0.497  2.199]
 [-0.313 -1.132  0.545 -1.056  0.272 -0.563 -2.675]
 [ 1.11   1.059  0.696 -0.808  0.15  -0.816 -2.382]]
time correlations [1.    0.311 0.258 0.25  0.251 0.254 0.259 0.263 0.268]
feature singulars [10.248  9.996  9.454  8.356  8.012  7.376  7.27   6.355]
interaction dim affine [0. 0. 0. 0.]
corr [[ 0.2  -0.08  0.11  0.5 ]
 [ 0.08 -0.03  0.18  0.46]
 [ 0.09 -0.05  0.17  0.42]]
corr [[-0.04  0.36 -0.21 -0.24]
 [ 0.25 -0.22 -0.09 -0.31]
 [ 0.41 -0.34 -0.13 -0.48]]
corr [[-0.    0.36 -0.18 -0.23]
 [ 0.22 -0.3  -0.02 -0.33]
 [ 0.38 -0.4  -0.06 -0.5 ]]
corr [[-0.07  0.16 -0.32  0.21]
 [-0.25 -0.24  0.41  0.52]
 [-0.13 -0.22  0.22  0.38]]
corr [[ 0.1  -0.37  0.16  0.08 -0.16  0.26 -0.27  0.26]
 [ 0.07  0.39  0.36  0.51 -0.27 -0.15  0.28  0.24]
 [-0.02  0.22  0.23  0.39 -0.39 -0.22  0.44  0.4 ]]
corr [[ 0.2  -0.07 -0.06  0.46]
 [ 0.04 -0.12  0.12  0.5 ]
 [ 0.07 -0.13  0.1   0.44]]
nearest [(8, 32, 0.195, -2.957, -3.248, -1.224, -1.856), (32, 8, 0.195, -3.248, -2.957, -1.856, -1.224), (44, 42, 0.266, 0.336, 0.559, -0.419, -0.389), (42, 44, 0.266, 0.559, 0.336, -0.389, -0.419), (23, 30, 0.273, 0.778, 0.519, 1.85, 1.811), (30, 23, 0.273, 0.519, 0.778, 1.811, 1.85), (58, 47, 0.292, -0.761, -0.423, 1.041, 1.02), (47, 58, 0.292, -0.423, -0.761, 1.02, 1.041), (69, 59, 0.3, -0.252, -0.42, 2.199, 1.724), (59, 69, 0.3, -0.42, -0.252, 1.724, 2.199)]
nearest [(8, 32, 0.195, -2.957, -3.248, -1.224, -1.856), (32, 8, 0.195, -3.248, -2.957, -1.856, -1.224), (44, 42, 0.266, 0.336, 0.559, -0.419, -0.389), (42, 44, 0.266, 0.559, 0.336, -0.389, -0.419), (24, 2, 0.269, 0.642, -0.764, 3.204, 1.548), (2, 24, 0.269, -0.764, 0.642, 1.548, 3.204), (3, 18, 0.271, -0.828, 0.893, -1.386, 0.712), (18, 3, 0.271, 0.893, -0.828, 0.712, -1.386), (30, 23, 0.273, 0.519, 0.778, 1.811, 1.85), (23, 30, 0.273, 0.778, 0.519, 1.85, 1.811)]
nearest [(53, 1, 0.724, -0.714, -0.702, 0.548, 1.095), (1, 53, 0.724, -0.702, -0.714, 1.095, 0.548), (32, 8, 0.753, -3.248, -2.957, -1.856, -1.224), (8, 32, 0.753, -2.957, -3.248, -1.224, -1.856), (30, 23, 0.809, 0.519, 0.778, 1.811, 1.85), (23, 30, 0.809, 0.778, 0.519, 1.85, 1.811), (42, 44, 0.82, 0.559, 0.336, -0.389, -0.419), (44, 42, 0.82, 0.336, 0.559, -0.419, -0.389), (20, 61, 0.851, -0.39, 0.2, -1.282, -0.012), (61, 20, 0.851, 0.2, -0.39, -0.012, -1.282)]
```

### Turn 6 — `execute_in_remote_docker` (`27200 ms` LLM)
- **Rationale**: Inspect the complete frozen feature map and quantify the ambiguity between near-antipodal samples before fitting a predictor that can use only the allowed inputs.
- **Alternatives Weighed**: `Assume sign loss is the only missing term` (rejected: Even signed quadratic fits are far above tolerance; the next check separates sign ambiguity from higher-order structure.)

```bash
cd /holdout_task; ls -lh /tmp/hegel_continual_archive; cat _frozen_nodes/node_00.py; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'));print('batch records:',len(p));
for i,r in enumerate(p):
 s=r['study'];print(i,list(r),[(k,[(j,np.shape(v)) for j,v in d.items()] if isinstance(d,dict) else np.shape(d)) for k,d in s.items()]); print('observed time grid',s['eval_times']);print('observed training input quantiles',np.quantile(s['train']['x'],[0,.25,.5,.75,1],axis=0))
PY
cat > /tmp/odd03.py <<'PY'
exec(open('/tmp/explore03.py').read().split('for n in (48')[0]); from scipy.spatial.distance import cdist
z=z[:N]; n=52
for k in range(1,5):
 X=np.column_stack([np.ones(N)]+[np.prod(z[:,ix],axis=1) for h in range(1,k+1) for ix in combinations_with_replacement(range(4),h)])
 beta=np.linalg.lstsq(X[:n],y[:n],rcond=1e-9)[0];g=abs(X[n:]@beta-y[n:]); print('signed poly',k,g[:,0].mean(),g[:,1:].mean());
 for deg,h in enumerate([1,5,15,35,70][:k+1]):
  a0=[np.arange(1),np.arange(1,5),np.arange(5,15),np.arange(15,35),np.arange(35,70)][deg];print('deg',deg,'term std',np.std(X[:n,a0]@beta[a0],axis=0)[[0,1,8]])
print('z range and mean', z.min(0),z.max(0),z.mean(0)); print('features oddness algebraically exact:');
for ff in (a._interactions,lambda u:a._curvature(u,.25)): print(np.max(abs(ff(z)-ff(-z))))
m=cdist(z,-z); np.fill_diagonal(m,np.inf);i,j=np.unravel_index(np.argmin(m),m.shape); print('nearest antipodes',i,j,m[i,j],z[i],z[j],y[[i,j]][:,[0,1,8]]);
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/odd03.py
```
- **Container Exit Code**: `0` (`897 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
4))]), ('panel_c', [('u', (24, 4)), ('v1', (24, 4)), ('v2', (24, 4))]), ('train', [('x', (72, 4)), ('y', (72,)), ('traj', (72, 8)), ('summary', (4,))]), ('test', [('x', (36, 4))]), ('telemetry_vec', (4,))]
observed time grid [0.25       0.64285714 1.03571429 1.42857143 1.82142857 2.21428571
 2.60714286 3.        ]
observed training input quantiles [[-1.85790397 -1.7777296  -1.81786612 -2.08041966]
 [-0.47271133 -0.51203132 -0.50247522 -0.68065999]
 [ 0.3812207   0.04803183  0.11215434 -0.10320973]
 [ 0.81896765  0.4782218   0.93033355  0.507191  ]
 [ 1.41365354  1.19259153  1.61160473  1.47224456]]
z range [-1.14343411 -1.13933953 -1.1444233  -1.13171176] [1.14367481 1.13318496 1.14372859 1.14775612]
z abs maxima train rows [70 31 67  7 52  5 57  0 45 44]
targets range [-3.24811227 -0.81567234 -1.57623962 -1.96288923 -2.1662015  -2.27050573
 -2.3918173  -2.53683408 -2.67510551] [2.64097715 1.26596514 2.27401237 2.85632562 3.23259002 3.5082256
 3.71133647 3.87963098 4.03116973]
traj singular values [31.67329522  2.57575588  0.50076105  0.05211529  0.04538649  0.04081331
  0.03771812  0.03582071]
signed poly 1 0.9038572202911596 1.1955213970541931
deg 0 term std [3.46944695e-17 8.32667268e-17 3.33066907e-16]
deg 1 term std [0.68099505 0.18342587 0.76990358]
signed poly 2 0.6395415089720623 0.49641197561471334
deg 0 term std [4.16333634e-17 1.66533454e-16 4.44089210e-16]
deg 1 term std [0.58347284 0.17281454 0.60815912]
deg 2 term std [0.80795945 0.27608801 1.26252673]
signed poly 3 0.9736986576248977 0.6363658443696355
deg 0 term std [4.16333634e-17 0.00000000e+00 0.00000000e+00]
deg 1 term std [0.37769142 0.08105998 0.37418738]
deg 2 term std [0.90315964 0.30200839 1.42361297]
deg 3 term std [0.99619178 0.24731675 0.86290228]
signed poly 4 0.3493998799247768 0.5082060712480546
deg 0 term std [1.11022302e-16 2.77555756e-16 4.44089210e-16]
deg 1 term std [0.4110631  0.03955868 0.13601028]
deg 2 term std [0.84697466 0.24134492 1.19805271]
deg 3 term std [0.62734462 0.19042783 0.79436568]
deg 4 term std [0.56626913 0.18021476 0.72838891]
z range and mean [-1.12137893 -1.13933953 -1.1444233  -1.13171176] [1.14367481 1.059282   1.14372859 1.14775612] [ 0.09918516 -0.07198668  0.01784947 -0.03014263]
features oddness algebraically exact:
0.0
0.0
nearest antipodes 2 24 0.26905381091645497 [-0.90193695  0.07276215  0.87207124 -0.41767704] [ 0.97665935 -0.15118229 -0.63047566  0.46551399] [[-0.76435985  0.23434753  1.54841155]
 [ 0.64189781  0.69104901  3.20435514]]
```

### Turn 7 — `execute_in_remote_docker` (`26745 ms` LLM)
- **Rationale**: The first frozen edge is exactly invariant under global latent sign reversal. Inspect the remaining payloads and quantify evidence of signed label dependence using training rows before choosing the best contract-compliant predictor.
- **Alternatives Weighed**: `Recalibrate the raw test inputs inside the sink` (rejected: Raw test inputs are not part of the declared upstream payloads and cannot be read by the sink.)

```bash
cat /holdout_task/_frozen_nodes/node_01.py /holdout_task/_frozen_nodes/node_02.py; cat /tmp/hegel_continual_archive/conf_m102_a04_batch_00_solver.py; cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; cat > /tmp/sign_audit03.py <<'PY'
exec(open('/tmp/explore03.py').read().split('for n in (48')[0]); from scipy.spatial.distance import cdist
z=z[:N]; n=52
m=cdist(z[:n],-z[:n]); np.fill_diagonal(m,np.inf)
print('training-only near-antipode mismatch:')
for k in np.argsort(m.min(1))[:8]:
 j=m[k].argmin()
 if j>k: print(k,j,m[k,j],y[k,0]-y[j,0],np.mean(abs(y[k,1:]-y[j,1:])))
import itertools
for h in range(1,5):
 P=np.column_stack([np.prod(z[:,ix],axis=1) for ix in itertools.combinations_with_replacement(range(4),h)]); print('order',h,'shape',P.shape)
for f in [1,3,5,7,10,15]:
 inds=[ix for h in range(1,5) for ix in itertools.combinations_with_replacement(range(4),h)]; X=np.column_stack([np.ones(N)]+[np.prod(z[:,ix],axis=1) for ix in inds]); d=np.array([0]+[len(ix) for ix in inds]); B=np.linalg.solve(X[:n].T@X[:n]+np.diag(.1*f**d),X[:n].T@y[:n]); print('parity residual',f,[(X[:n,d%2==p]@B[d%2==p]).std(0)[[0,1,8]] for p in [0,1]],abs(X[n:]@B-y[n:]).mean(0)[[0,1,8]])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/sign_audit03.py
```
- **Container Exit Code**: `0` (`562 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
},
    "max_fanout": 2,
    "node_code_shas": {
      "node_00": "707822a36ae53a2cad8733482798bf85a2542b4507c759710f3b90ede4a16787",
      "node_01": "493aca178d528c16ec681d5e109ba4e4500ff58bbef4f1e9c17dd6aa7ee50743",
      "node_02": "9d9cf57d412595efc361c244c11a554b623747129b330a44ce7601a11d8fcffd",
      "node_03": "eeeaeecd9a19626365176f029b400d14a57213e504550a9bef596c3f286941ff"
    },
    "non_identity_checks": {
      "node_00": true,
      "node_01": true,
      "node_02": true
    },
    "shared_fanout_nodes": [
      "node_01"
    ],
    "topo_order": [
      "node_00",
      "node_01",
      "node_02",
      "node_03"
    ]
  },
  "kernel_version": "hegel_shared_math_handoff_kernel_v3_m961_three_layer",
  "wall_ms": 3141.92
}z range [-1.14343411 -1.13933953 -1.1444233  -1.13171176] [1.14367481 1.13318496 1.14372859 1.14775612]
z abs maxima train rows [70 31 67  7 52  5 57  0 45 44]
targets range [-3.24811227 -0.81567234 -1.57623962 -1.96288923 -2.1662015  -2.27050573
 -2.3918173  -2.53683408 -2.67510551] [2.64097715 1.26596514 2.27401237 2.85632562 3.23259002 3.5082256
 3.71133647 3.87963098 4.03116973]
traj singular values [31.67329522  2.57575588  0.50076105  0.05211529  0.04538649  0.04081331
  0.03771812  0.03582071]
training-only near-antipode mismatch:
2 24 0.26905381091645497 -1.4062576650710183 1.1823695133101209
3 18 0.2714942309622947 -1.721753370991943 1.4546248514439803
12 44 0.33870091757393417 -2.0150979899232597 0.493396957890751
33 42 0.3441642350380411 -1.5489748868669344 0.645742592088787
order 1 shape (72, 4)
order 2 shape (72, 10)
order 3 shape (72, 20)
order 4 shape (72, 35)
parity residual 1 [array([0.79175043, 0.31084498, 1.40758816]), array([0.63681186, 0.20119791, 0.75745047])] [0.40149159 0.09472038 0.42947972]
parity residual 3 [array([0.68470195, 0.26428935, 1.18857893]), array([0.63024316, 0.18305503, 0.66060817])] [0.47585276 0.18125673 0.62936272]
parity residual 5 [array([0.58922174, 0.22614151, 1.00928936]), array([0.62225662, 0.17705401, 0.64721306])] [0.53567528 0.22642635 0.80629238]
parity residual 7 [array([0.48855656, 0.1915531 , 0.84532954]), array([0.62482582, 0.17532418, 0.65305993])] [0.57197205 0.25712082 0.9349821 ]
parity residual 10 [array([0.35869364, 0.14738861, 0.63956713]), array([0.62818148, 0.17341887, 0.66388347])] [0.66213269 0.29159579 1.09178853]
parity residual 15 [array([0.21924714, 0.09567598, 0.4074407 ]), array([0.62562796, 0.17046763, 0.67486256])] [0.75287342 0.33685848 1.2946105 ]
```

### Turn 8 — `execute_in_remote_docker` (`33748 ms` LLM)
- **Rationale**: Fit a contract-compliant baseline with model weighting determined by training-only leave-one-out errors. This tests how much prediction accuracy the frozen representation supports without assuming access to the missing orientation.
- **Alternatives Weighed**: `Infer the missing global sign from frozen predictions` (rejected: The frozen regression output is itself a deterministic function of the sign-invariant features and training labels.)

```bash
cat > /tmp/invariant03.py <<'PY'
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve

def _models(w,n):
    for u in (w, w[:,:4],w[:,4:]):
        uc = u-u[:n].mean(0)
        _, sv, vt = np.linalg.svd(uc[:n], full_matrices=False)
        for frac in ((1., .18, .003) if u.shape[1]==8 else (1.,.15)):
            v = uc @ vt.T / np.sqrt((sv*sv/n+frac)**.65)
            v /= np.sqrt(np.mean(np.sum(v[:n]**2,axis=1)))+1e-12
            g=v @ v[:n].T
            d=cdist(v,v[:n],'sqeuclidean')
            yield g, 'linear'
            yield g+.35*g*g, 'quadratic'
            for t in (.2,.8,2.):
                yield np.exp(-t*d)+.45*g, 'rbf+linear'
                s=np.sqrt(t*d)
                yield (1+s)*np.exp(-s)+.45*g, 'matern+linear'

def _predict(w, target, n, groups):
    center=target.mean(0)
    Y=target-center
    eye=np.eye(n)
    records=[]
    for K,name in _models(w,n):
        for noise in (.02,.09,.28,.75,1.65):
            C=cho_factor(K[:n]+noise*eye+3.)
            ki=cho_solve(C,eye)
            a=ki@Y
            res=np.abs(a/ki.diagonal()[:,None])
            scores=np.array([res[:,j].mean() for j in groups])
            pred=(K[n:]+3.)@a+center
            records.append((scores,pred,name,noise))
    scores=np.stack([z[0] for z in records])
    ans=np.empty((len(w)-n,target.shape[1]))
    for i,g in enumerate(groups):
        idx=np.argsort(scores[:,i])[:18]; s=scores[idx,i]; scale=max(.025*s[0],1e-12)
        weights=np.exp(-(s-s[0])/scale); weights/=weights.sum()
        ans[:,g]=np.tensordot(weights,np.stack([records[j][1][:,g] for j in idx]),axes=1)
    return ans

def run_node(inputs: dict) -> dict:
    feature_data=inputs['node_00']
    w=np.asarray(feature_data['features'],dtype=float)
    n=int(feature_data['n_train'])
    temporal=inputs['node_01']
    y=np.asarray(temporal['targets'],dtype=float)
    basis=np.asarray(temporal['basis'],dtype=float)
    prior=np.asarray(inputs['node_02']['fit'],dtype=float)
    q=np.linalg.qr(basis,mode='reduced')[0]
    proj=q@q.T
    traj=y[:,1:]
    smooth=traj@proj
    tail=traj-smooth
    pred=_predict(w,np.column_stack((y[:,0],smooth,tail)),n,([0],slice(1,1+traj.shape[1]),slice(1+traj.shape[1],None)))
    yh=pred[:,0]
    sp=pred[:,1:1+traj.shape[1]]+pred[:,1+traj.shape[1]:]
    # Limit the influence of the extrapolating upstream fit when accounting for its disagreement.
    radius=np.maximum(np.std(y,axis=0),1e-12)
    delta=prior-np.column_stack((yh,sp))
    correction=.015*radius*np.tanh(delta/radius)
    yh=yh+correction[:,0]
    sp=sp+correction[:,1:]
    diag=np.array([yh.mean(),yh.std(),sp[:,-1].mean(),sp[:,-1].std()]) if len(yh) else np.zeros(4)
    return {'y_hat':yh,'state_pred':sp,'diagnostics_vec':diag}
PY
cp /tmp/invariant03.py /holdout_task/node_03.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
sed -n '17,120p' /tmp/hegel_continual_archive/conf_m102_a04_batch_00_solver.py
```
- **Container Exit Code**: `0` (`667 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_03.py']`
- **Stdout (tail)**:
```text
nd(slope)
    return np.asarray(offset), np.asarray(scale)


def _panel_maps(panel, z):
    u = np.asarray(panel['u'], dtype=float)

    # Both responses share their nonlinear shape parameters across coordinates.
    def softsign(w, q):
        return one(w / (1 + q[0]*abs(w)) + q[1]*w)

    def curvature(w, q):
        w2 = w*w
        return np.c_[one(w), w2, w2/(1+q[0]*w2)]

    def fit(features, target, initial, bounds, **options):
        def residual(q):
            f = features(u, q)
            return (f @ lin(f, target) - target).ravel()
        q = least_squares(residual, initial, bounds=bounds, **options).x
        return features(z, q) @ lin(features(u, q), target)

    c = fit(softsign, panel['v1'], [.85, -.01], ([.02, -1], [10, 1]))
    d = fit(curvature, panel['v2'], [.25], (.08, .65), gtol=1e-13)
    return np.c_[c, d]


def _time_basis(t, trajectory):
    def basis(r):
        return -np.expm1(-t[:, None]*r)

    def residual(r):
        f = basis(r)
        return (f @ lin(f, trajectory.T) - trajectory.T).ravel()

    rates = least_squares(residual, [.4, 1.8], bounds=([.05, .8], [.8, 4])).x
    return basis(rates)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    offset, scale = _align(study['panel_a'])
    z = (np.vstack((train['x'], study['test']['x'])) - offset)/scale
    w = _panel_maps(study['panel_c'], z)
    w = (w-w[:n].mean(0))/np.maximum(w[:n].std(0), .1)
    gram = w @ w.T / w.shape[1]
    norms = gram.diagonal()
    distance = np.maximum(norms[:, None] + norms - 2*gram, 0)
    quadratic = (1+.25*gram)**2
    targets = np.column_stack((train['y'], train['traj']))
    eye = np.eye(n)
    fit = np.mean([
        kernel[n:, :n] @ np.linalg.solve(kernel[:n, :n] + ridge*eye, targets)
        for kernel, ridge in ((quadratic, 3e-5),
                              (quadratic + np.exp(-.06*distance), 3e-4))
    ], axis=0)
    y = fit[:, 0]
    basis = _time_basis(np.asarray(study['eval_times']), train['traj'])
    state = (basis @ lin(basis, fit[:, 1:].T)).T
    terminal = state[:, -1]
    diagnostics = np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=99e7edcee464c94e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_99e7edce (directory).
```

### Turn 9 — `execute_in_remote_docker` (`30185 ms` LLM)
- **Rationale**: A paired kernel ablation on identical folds will measure whether signed information helps without conflating it with model choice. Exact feature symmetry does not by itself establish a numerical lower bound on this finite holdout, so report measured errors separately.
- **Alternatives Weighed**: `Load raw test coordinates inside the sink` (rejected: That would bypass the declared DAG inputs rather than solve the node synthesis task.)
- **ATMS NOGOOD DECLARED**: Refuted `The rewired edge adds enough information to distinguish latent inputs that differ only by global sign.` | Witness: `Both frozen feature blocks are exactly unchanged by global sign reversal. Training rows 3 and 18 are near antipodes at latent distance 0.27149, yet differ by 1.72175 in scalar target and 1.45462 in trajectory MAE.` | Blanket Diagnosis: `Per-column standardization preserves the symmetry, node_01 exposes no test-row information, and node_02 derives its test predictions from the same invariant features.` | Sublated Pivot: `Stress-test the allowed predictor and use signed latent coordinates only in an offline diagnostic to measure the cost of the missing information.`

```bash
cat > /tmp/stress03.py <<'PY'
import sys,pickle,numpy as np
sys.path[:0]=['/holdout_task','/holdout_task/_frozen_nodes']
import node_00 as a,node_01 as b,node_02 as c,node_03 as d
from scipy.spatial.distance import cdist
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
t=s['train']; y=np.c_[t['y'],t['traj']]; z=a._calibrate(s['panel_a'],t['x']); rng=np.random.default_rng(2026)
for train,test in [(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [rng.permutation(72) for _ in range(6)]]:
 st=dict(s,train={k:t[k][train] for k in ('x','y','traj')},test={'x':t['x'][test]}); f=a.run_node({'__input__':st}); g=b.run_node({'__input__':st}); h=c.run_node({'node_00':f,'node_01':g}); j=d.run_node({'node_00':f,'node_01':g,'node_02':h}); err=np.abs(np.c_[j['y_hat'],j['state_pred']]-y[test]); print('candidate',*(round(v,5) for v in (err[:,0].mean(),err[:,1:].mean(),err[:,0].max(),err[:,1:].mean(1).max())))
 # Diagnostic only: compare the same kernel with and without access to the sign.
 D=cdist(z,z,'sqeuclidean'); A=cdist(z,-z,'sqeuclidean'); K=np.exp(-.6*D); V=np.exp(-.6*A); E=(K+V)/2; O=(K-V)/2
 for name,M in [('even',E),('signed',E+O)]:
  M=M+1; u=solve(M[np.ix_(train,train)]+.07*np.eye(len(train)),y[train],assume_a='pos'); e=np.abs(M[np.ix_(test,train)]@u-y[test]); print(name,round(e[:,0].mean(),5),round(e[:,1:].mean(),5))
print('exact sign-collision deltas',np.max(np.abs(a._interactions(z)-a._interactions(-z))),np.max(np.abs(a._curvature(z,.25)-a._curvature(-z,.25))))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/stress03.py
```
- **Container Exit Code**: `0` (`1035 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
candidate 0.74605 0.59982 3.00103 2.35358
even 0.88352 0.82403
signed 0.36271 0.64458
candidate 0.73309 0.44962 2.14785 1.8588
even 0.69729 0.45094
signed 0.36293 0.37926
candidate 0.95357 0.61848 3.41652 2.68207
even 0.94797 0.62586
signed 0.56623 0.57044
candidate 1.08081 0.7377 4.23331 2.99003
even 0.93372 0.75698
signed 0.36319 0.50779
candidate 0.77051 0.55635 3.22682 2.19557
even 0.80983 0.57233
signed 0.35334 0.38413
candidate 0.7238 0.50966 2.88439 1.86805
even 0.71059 0.48332
signed 0.31019 0.3405
candidate 0.83738 0.56059 1.99767 1.41777
even 0.80707 0.54491
signed 0.50387 0.45354
exact sign-collision deltas 0.0 0.0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=99e7edcee464c94e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_99e7edce (directory).
```

### Turn 10 — `execute_in_remote_docker` (`28015 ms` LLM)
- **Rationale**: Verify the saved artifact in an isolated process and test whether the sign collision survives the full DAG. This distinguishes a structural limitation from the separately measured public threshold failures.
- **Alternatives Weighed**: `Fit a richer model on the same frozen payloads` (rejected: Model flexibility cannot separate two test rows whose feature payloads are identical under latent sign reversal.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I test_dag_candidate.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path[:0]=['/holdout_task','/holdout_task/_frozen_nodes']
import node_00 as a,node_01 as b,node_02 as c,node_03 as d
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v')); p=u.shape[1]; j=(np.arange(p)+1)%p; coef=[]
for i in range(p):
 x,w=u[:,i],u[:,j[i]]; f=np.column_stack((np.ones(len(x)),x,w,x**3,x*w)); yy=v[:,i]; beta=least_squares(lambda q:f@q-yy,a._linear(f,yy),jac=lambda q:f.copy(),loss='cauchy',f_scale=.04).x
 for _ in range(4):
  r=abs(f@beta-yy); keep=r<=max(.001,5*np.median(r))
  if keep.sum()>=f.shape[1]: beta=a._linear(f[keep],yy[keep])
 coef.append(beta)
o,g,h,k,l=np.array(coef).T
z=np.array([[.2,.4,-.7,.1],[.65,-.25,.5,-.4],[.8,.3,.3,-.4]]); z=np.r_[z,-z]; x=o+g*z+h*z[:,j]+k*z**3+l*z*z[:,j]; assert np.max(abs(a._calibrate(s['panel_a'],x)-z))<1e-8
s=dict(s,test={'x':x}); A=a.run_node({'__input__':s}); B=b.run_node({'__input__':s}); C=c.run_node({'node_00':A,'node_01':B}); D=d.run_node({'node_00':A,'node_01':B,'node_02':C}); n=A['n_train']
for name,val in [('features',A['features'][n:]),('upstream predictions',C['fit']),('scalar output',D['y_hat']),('trajectory output',D['state_pred'])]:
 delta=np.max(abs(val[:3]-val[3:])); print(name,'antipodal difference',delta); assert delta<1e-7
assert D['diagnostics_vec'].shape==(4,) and np.isfinite(D['diagnostics_vec']).all()
PY
```
- **Container Exit Code**: `1` (`627 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"accepted": false, "errors": ["public_cv_y_mae_exceeded:0.746054", "public_cv_traj_mae_exceeded:0.599819"], "cv_metrics": {"public_cv_y_mae": 0.746054, "public_cv_traj_mae": 0.599819}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1, "node_03": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=99e7edcee464c94e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_99e7edce (directory).
```

### Turn 11 — `execute_in_remote_docker` (`21231 ms` LLM)
- **Rationale**: The public harness returned a nonzero status and skipped the previous chained stress test. Run the checks independently, constructing antipodal pairs through the actual frozen calibration without changing the candidate.

```bash
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I test_dag_candidate.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path[:0]=['/holdout_task','/holdout_task/_frozen_nodes']
import node_00 as a,node_01 as b,node_02 as c,node_03 as d
with open('/holdout_task/public_batch.pkl','rb') as f:
 s=pickle.load(f)[0]['study']
p=s['panel_a']; u,v=p['u'],p['v']; z=np.array([[.2,.4,-.7,.1],[.65,-.25,.5,-.4],[.8,.3,.3,-.4]]); z=np.r_[z,-z]; lin=np.linalg.lstsq(np.c_[np.ones(len(u)),u],v,rcond=None)[0]
x=np.stack([least_squares(lambda x:a._calibrate(p,x[None,:])[0]-r,np.r_[1.,r]@lin,gtol=1e-12).x for r in z]); check=np.max(abs(a._calibrate(p,x)-z)); print('calibration inversion error',check); assert check<1e-7
s=dict(s,test={'x':x}); A=a.run_node({'__input__':s}); B=b.run_node({'__input__':s}); C=c.run_node({'node_00':A,'node_01':B}); D=d.run_node({'node_00':A,'node_01':B,'node_02':C}); n=A['n_train']
for name,val in [('features',A['features'][n:]),('upstream predictions',C['fit']),('scalar output',D['y_hat']),('trajectory output',D['state_pred'])]:
 delta=np.max(abs(val[:3]-val[3:])); print(name,'antipodal difference',delta); assert delta<1e-6
assert D['diagnostics_vec'].shape==(4,) and np.isfinite(D['diagnostics_vec']).all()
PY
```
- **Container Exit Code**: `0` (`2061 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"accepted": false, "errors": ["public_cv_y_mae_exceeded:0.746054", "public_cv_traj_mae_exceeded:0.599819"], "cv_metrics": {"public_cv_y_mae": 0.746054, "public_cv_traj_mae": 0.599819}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1, "node_03": 1}}
calibration inversion error 2.7158830739892892e-14
features antipodal difference 5.96189764223709e-14
upstream predictions antipodal difference 3.72952224658718e-11
scalar output antipodal difference 5.525579993559404e-13
trajectory output antipodal difference 2.120525977034049e-13

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=99e7edcee464c94e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_99e7edce (directory).
```

### Turn 12 — `submit_completed_task` (`16395 ms` LLM)
- **Rationale**: Isolated verification confirmed the artifact executes once and fails both error thresholds. The antipodal stress test confirms an upstream information loss, but does not establish a numerical lower bound on this finite public holdout.
- **Alternatives Weighed**: `Increase sink model capacity` (rejected: A richer model cannot distinguish antipodal rows with identical incoming payloads.)
- **Submitted Final Answer / Summary**: `Saved and stress-tested `/holdout_task/node_03.py`. The public harness rejected it: scalar MAE was 0.746054 and trajectory MAE was 0.599819, both above 0.038. The end-to-end test confirmed that the frozen inputs cannot distinguish latent vectors that differ only by global sign. To address that ambiguity, an upstream node would need to expose sign-sensitive features.`

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
