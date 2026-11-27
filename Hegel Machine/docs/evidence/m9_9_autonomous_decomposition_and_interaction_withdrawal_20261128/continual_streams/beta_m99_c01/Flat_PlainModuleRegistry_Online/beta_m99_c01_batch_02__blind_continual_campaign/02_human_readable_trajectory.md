# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m99_c01_batch_02__blind_continual_campaign__downstream_module`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M99-Autonomous-Decomposition-And-Interaction-Withdrawal-World` |
| **Domain** | `coupled_2d_transport_reaction_box_simplex_kkt` |
| **Wave / Wormhole Family** | `m9_9_autonomous_decomposition_and_interaction_withdrawal` / `m99_beta` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `390,718` (`379,648` / `178,048` / `11,070`) |
| **Total Episode Wall Time (s)** | `350.04s` |

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

### Turn 1 — `execute_in_remote_docker` (`14511 ms` LLM)
- **Rationale**: Inspect the wrapper, input shapes, and prefix log to distinguish boundary changes from coordinate or interface errors before implementing the downstream solve.
- **Alternatives Weighed**: `Reuse open-wall Robin unchanged` (rejected: The prior public gate failed and the task explicitly allows boundary changes; inspect the apparatus log before selecting an operator.)

```bash
cd /holdout_task && cat batch_solver.py _verified_upstream.py _upstream_param_payload.json validate_public_schema.py && cat public_asset_index.json && ls -l /tmp/hegel_continual_archive && cat /tmp/hegel_continual_archive/archive_apparatus_log.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'box simplex KKT ridge nuisance profiling'; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_batch.pkl','rb')); print('n',len(x)); a=x[0]; print(a.keys()); print('diagnostics',type(a.get('campaign_diagnostics'))); print('study:');
for k,v in a['study'].items():
 print(k, type(v).__name__, getattr(v,'shape',None), list(v) if isinstance(v,dict) else repr(v)[:160] if np.size(v)<12 else '')
PY
```
- **Container Exit Code**: `0` (`403 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
75142,
        2.49333,
        -0.7982
      ],
      [
        0.42018,
        0.69128,
        -0.85291,
        -0.34711,
        -0.44479,
        0.30481,
        2.64392,
        -0.59039
      ],
      [
        0.23794,
        0.33435,
        -0.73356,
        -0.37421,
        -0.78778,
        -0.34623,
        2.56828,
        -0.25972
      ],
      [
        0.0632,
        -0.10492,
        -0.222,
        -0.35157,
        -0.8618,
        -0.8199,
        2.67664,
        0.17168
      ],
      [
        -0.14956,
        -0.47224,
        0.51344,
        -0.37876,
        -0.71875,
        -0.76417,
        2.72058,
        0.57738
      ],
      [
        -0.46109,
        -0.76865,
        0.81524,
        -0.33953,
        -0.36562,
        -0.29391,
        2.59502,
        0.78914
      ],
      [
        -0.61547,
        -0.79466,
        0.72471,
        -0.3336,
        0.09282,
        0.32324,
        2.50005,
        0.81341
      ],
      [
        -0.69802,
        -0.6745,
        0.14356,
        -0.33688,
        0.39927,
        0.80021,
        2.63147,
        0.58265
      ],
      [
        -0.82058,
        -0.28369,
        -0.46445,
        -0.35487,
        0.75843,
        0.74584,
        2.58252,
        0.23576
      ]
    ]
  }
]autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
n 1
dict_keys(['study', 'campaign_diagnostics'])
diagnostics <class 'list'>
study:
Lx float None 2.4
Ly float None 2.0
nx int None 14
ny int None 12
kappa_x ndarray (14,) 
kappa_y ndarray (12,) 
vel_x ndarray (14,) 
vel_y ndarray (12,) 
robin_x float None 1.0893237660938055
robin_y float None 1.101249119959542
V_reaction_2d ndarray (14, 12) 
mu_shifts ndarray (3,) array([0.15, 0.55, 1.1 ])
source_modes ndarray (8, 14, 12) 
sensor_ij ndarray (32, 2) 
row_bank_id ndarray (96,) 
sensor_calibration_sweeps dict None ['ref_rosette', 'raw_rosette', 'single_bank_sweeps']
Z_nuisance ndarray (96, 3) 
y_obs_raw ndarray (96,) 
aux_manifold_telemetry ndarray (24, 4) 
nuisance_ridge float None 0.08
source_ridge float None 0.04
u_box float None 0.42
total_budget float None 1.4
```

### Turn 2 — `execute_in_remote_docker` (`17357 ms` LLM)
- **Rationale**: Inspect compact log summaries and prior solver structure to identify candidate boundary rows and whether the verified upstream calibration leaves residual channel mixing.
- **Alternatives Weighed**: `Fit the boundary from the study observations alone` (rejected: Known-weight tracer pulses are present and can distinguish boundary effects from source-weight errors.)

```bash
cd /holdout_task; python3 - <<'PY'
import json,pickle,numpy as np
from pathlib import Path
p=Path('/tmp/hegel_continual_archive'); print([(f.name,f.stat().st_size) for f in p.iterdir()]); print((p/'archive_manifest.json').read_text()[:3500]);
for entry in json.loads((p/'archive_apparatus_log.json').read_text()):
 print('log',[(k, np.shape(v) if isinstance(v,list) else str(v)[:700]) for k,v in entry.items()]);
a=pickle.load(open('public_batch.pkl','rb'))[0]; print('diagnostics',len(a['campaign_diagnostics']),[(k,np.shape(v),np.round(np.asarray(v).ravel()[:8],5)) for k,v in a['campaign_diagnostics'][0].items()]); print('sweep',[(k, list(v) if isinstance(v,dict) else np.shape(v)) for k,v in a['study']['sensor_calibration_sweeps'].items()]);
PY
for f in /tmp/hegel_continual_archive/*solver.py; do echo ==== "$f"; wc -l "$f"; grep -nEi 'bound|wall|robin|rosette|diagnos|def |calibr|period|telemet|archive' "$f" | head -100; done; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```
- **Container Exit Code**: `0` (`367 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
tte'])
37:    y = np.asarray(sweep['raw_rosette'])
43:def _box_simplex(hessian, linear, cap, budget):
90:def solve_batch(study: dict) -> dict:
92:    y = _calibrate(study)
109:                                'primal_objective': objective, 'dual_lower_bound': float(dual)}}
{
  "ok": true,
  "target": "autogenous_sparse_ridge_certification.solve_and_certify_l0",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_sparse_ridge_certification",
      "granularity_tier": "L2_domain_operator",
      "compiled_code_sha256": "770f6644bb1fd8eb49dbacf51ffa3405a9ca876974e0021d66415d9014d8dfb9",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "n_samples",
          "op": ">=",
          "value": 1
        },
        {
          "field": "n_features",
          "op": "<=",
          "value": 16
        },
        {
          "field": "lam0",
          "op": ">=",
          "value": 0
        },
        {
          "field": "lam2",
          "op": ">=",
          "value": 0
        },
        {
          "field": "M",
          "op": ">=",
          "value": 0
        },
        {
          "field": "eps",
          "op": ">",
          "value": 0
        },
        {
          "field": "max_leaves",
          "op": ">=",
          "value": 1
        }
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {},
      "functions": [
        {
          "function": "solve_and_certify_l0",
          "signature": "solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict",
          "doc_summary": "Certify boxed sparse ridge regression with at most 16 real predictors."
        }
      ],
      "index_cards": [
        "autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_sparse_ridge_certification",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`18248 ms` LLM)
- **Rationale**: Inspect the asset metadata, exact archived stencil and sensor layout before building pulse tests; this will distinguish relevant boundary alternatives from calibration or indexing errors.
- **Alternatives Weighed**: `Use the archived zero-loss wall unchanged` (rejected: It is a useful candidate, but the current campaign supplies tracer pulses that can test whether that operator still applies.); `Use the boxed sparse ridge skill` (rejected: The objective has no cardinality penalty; it requires a fixed-sum box-constrained quadratic program.)

```bash
cd /holdout_task && cat public_asset_index.json && cat /tmp/hegel_continual_archive/beta_m99_c01_batch_01_solver.py && python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
a=pickle.load(open('public_batch.pkl','rb'))[0]; s=a['study']; print('pulse weights', [p['w_pulse'].tolist() for p in a['campaign_diagnostics']]); print('axis endpoints',[(k,s[k][[0,-1]]) for k in ('kappa_x','kappa_y','vel_x','vel_y')]); print('sensors',s['sensor_ij'].tolist()); print('bank sequence',np.unique(s['row_bank_id'],return_counts=True),s['row_bank_id'][:16]); print('public telemetry first rows',s['aux_manifold_telemetry'][:4]);
for e in json.load(open(p)):
 t=np.asarray(e['facility_telemetry_trace']); print(e['completed_campaign_short_name'],'mean',t.mean(0).round(4),'std',t.std(0).round(4))
PY
```
- **Container Exit Code**: `0` (`390 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
y = _calibrate(study)
    z = np.asarray(study['Z_nuisance'])
    proj = np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    s -= z @ (proj @ s)
    y -= z @ (proj @ y)
    ridge = study['source_ridge']
    cap, budget = study['u_box'], study['total_budget']
    w, nu = _box_simplex(s.T @ s + ridge * np.eye(s.shape[1]), -s.T @ y, cap, budget)
    residual = s @ w - y
    objective = float((residual @ residual + ridge * (w @ w)) / 2)
    g = s.T @ residual + ridge * w
    reduced = g - nu
    dual = objective - g @ w + nu * budget + cap * np.minimum(reduced, 0).sum()
    return {'weights': w, 'terminal_field': np.tensordot(w, u_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(reduced, 0).tolist(),
                                'lam_upper': np.maximum(-reduced, 0).tolist(),
                                'primal_objective': objective, 'dual_lower_bound': float(dual)}}
pulse weights [[1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0]]
axis endpoints [('kappa_x', array([0.41682986, 0.41682986])), ('kappa_y', array([0.52791443, 0.28791443])), ('vel_x', array([ 0.35      , -0.20572484])), ('vel_y', array([-0.        , -0.22652476]))]
sensors [[0, 8], [8, 0], [10, 10], [8, 4], [4, 8], [12, 2], [6, 0], [12, 10], [8, 6], [10, 6], [10, 2], [12, 4], [2, 6], [6, 10], [12, 0], [2, 4], [0, 2], [0, 6], [2, 8], [6, 2], [6, 4], [2, 0], [10, 8], [8, 8], [4, 10], [4, 6], [10, 0], [12, 8], [2, 2], [12, 6], [0, 10], [8, 2]]
bank sequence (array([0, 1, 2, 3]), array([24, 24, 24, 24])) [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3]
public telemetry first rows [[-0.21695418  1.92840336 -1.03437628  1.13205374]
 [ 1.92510512 -1.76977495 -1.63564425  0.58652472]
 [-0.54267536 -0.07152287  0.29151261 -0.51545114]
 [ 1.9210943   0.70016692  1.20269052 -0.31461173]]
facility_init mean [ 3.600e-03 -2.400e-03  4.100e-03 -3.669e-01  1.000e-04  8.300e-03
 -4.141e-01  7.800e-03] std [0.6035 0.6056 0.6025 0.0751 0.6049 0.6004 0.0848 0.6003]
beta_m99_c01_s00 mean [ 5.9000e-03 -1.2000e-03 -2.0000e-03  2.4826e+00  1.4000e-03 -7.4000e-03
 -3.8920e-01  3.7000e-03] std [0.5973 0.5936 0.6121 0.0716 0.5997 0.606  0.071  0.6037]
beta_m99_c01_batch_01 mean [-8.9000e-03  1.1300e-02 -1.0500e-02 -3.4330e-01 -2.9000e-03 -6.2000e-03
  2.5674e+00  1.4000e-03] std [0.6048 0.5966 0.6111 0.0622 0.6015 0.5969 0.0596 0.5924]
```

### Turn 4 — `execute_in_remote_docker` (`36332 ms` LLM)
- **Rationale**: Compare exact stencils and fit nested boundary-row families with each pulse held out. A residual that persists after nuisance profiling would call for a different row structure rather than a more flexible nuisance fit.
- **Alternatives Weighed**: `Infer the wall setting from archive telemetry alone` (rejected: The current campaign has known-input tracer pulses that directly test the boundary operator.)

```bash
cat > /tmp/probe_wall.py <<'PY'
import sys,json,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
from scipy.sparse import diags,kron,eye,csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
root='/holdout_task/'
a=pickle.load(open(root+'public_batch.pkl','rb'))[0]; pay=json.load(open(root+'_upstream_param_payload.json')); st=calibrate_upstream(a['study'],pay)
pulses=a['campaign_diagnostics']; W=np.stack([p['w_pulse'] for p in pulses],axis=1)
Y=np.stack([calibrate_upstream(dict(a['study'],y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in pulses],axis=1)
z=st['Z_nuisance']; proj=np.linalg.lstsq(z,np.eye(len(z)),rcond=None)[0]
nx,ny=st['nx'],st['ny']; ix,iy=st['sensor_ij'].T; N=nx*ny
M=st['source_modes']; F=np.asfortranarray(M.transpose(2,1,0).reshape(N,len(M)))
B=F@W
I=eye(N,format='csc'); V=diags(st['V_reaction_2d'].ravel(order='F'),format='csc')

def axis(k,v,h,r,p):
 d0,d1,c0,c1,b0,b1=p
 kp=.5*(k[1:]+k[:-1])/h**2
 lo=-kp-np.maximum(v[1:],0)/h; up=-kp-np.maximum(-v[:-1],0)/h
 lo[-1]=-c1*k[-1]/h**2; up[0]=-c0*k[0]/h**2
 diag=np.r_[c0*k[0]/h**2+b0*k[0]/h**2+d0*r/h,-lo[:-1]-up[1:],c1*k[-1]/h**2+b1*k[-1]/h**2+d1*r/h]
 A=diags([lo,diag,up],[-1,0,1],format='lil'); A[0,-1]=-b0*k[0]/h**2; A[-1,0]=-b1*k[-1]/h**2
 return A.tocsc()

def forward(p,rhs=B):
 Ax,Ay=[axis(st['kappa_'+q],st['vel_'+q],st['L'+q]/(n-1),st['robin_'+q],p0) for q,n,p0 in zip('xy',(nx,ny),np.reshape(p,(2,6)))]; L=(kron(eye(ny),Ax)+kron(Ay,eye(nx))+V).tocsc()
 U=[splu(L+mu*I).solve(np.asfortranarray(rhs)) for mu in st['mu_shifts']]
 return np.concatenate([u[ix+nx*iy] for u in U]),U[-1]

def err(p, nuisance=False):
 r=forward(p)[0]-Y
 return r-z@(proj@r) if nuisance else r

def metrics(e):
 return np.round(np.sqrt(np.mean(e**2,axis=0)),9)

if __name__=='__main__':
 for name,p in [('Robin',[1,1,1,1,0,0]),('reflecting',[0,0,1,1,0,0]),('ghost mirror',[0,0,2,2,0,0]),('periodic',[0,0,1,1,1,1]),('one sided Dirichlet',[1,1,0,0,0,0])]:
  print(name,metrics(err(p*2)),metrics(err(p*2,True)))
 p0=np.array([1,1,1,1,0,0]*2,dtype=float)
 for free in ([0,1,6,7],[0,1,2,3,6,7,8,9],list(range(12))):
  for nuisance in (False,True):
   def unpack(x):
    p=p0.copy(); p[free]=x; return p
   lo=np.array([0,0,0,0,0,0]*2); hi=np.array([100,100,15,15,15,15]*2)
   for held in (-1,0,1,2):
    keep=np.arange(3)!=held
    fit=least_squares(lambda x:err(unpack(x),nuisance)[:,keep].ravel(),p0[free],bounds=(lo[free],hi[free]),max_nfev=100,gtol=1e-12,xtol=1e-12)
    p=unpack(fit.x); print('fit',len(free),'nuis',nuisance,'held',held,'p',p.round(5),'rmse',metrics(err(p,nuisance)))
 print('Y ranges',Y.min(0),Y.max(0))
 sweep=st['sensor_calibration_sweeps']; raw=sweep['raw_rosette']; ref=sweep['ref_rosette']; good=(abs(raw)<=8.5).all(axis=1); print('canonical sweep delta',np.max(abs(raw[good]-ref[good])), 'outlier rows',sum(~good))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_wall.py
```
- **Container Exit Code**: `0` (`14293 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
56  0.04386661 0.04486002]
fit 8 nuis False held 1 p [0.      0.02876 1.66397 0.62495 0.      0.      0.      0.      1.21751
 0.6657  0.      0.     ] rmse [0.04804894 0.04046924 0.05078232]
fit 8 nuis False held 2 p [ 0.       0.       0.57121 15.       0.       0.       0.       0.
  0.44585 15.       0.       0.     ] rmse [0.02281561 0.01999926 0.09717445]
fit 8 nuis True held -1 p [0.      0.03956 1.45207 0.75436 0.      0.      0.      0.      0.90182
 1.07956 0.      0.     ] rmse [0.04204975 0.03505327 0.05626568]
fit 8 nuis True held 0 p [0.00000e+00 8.61000e-03 1.50000e+01 3.61130e-01 0.00000e+00 0.00000e+00
 0.00000e+00 2.64200e-02 1.52577e+00 1.00077e+00 0.00000e+00 0.00000e+00] rmse [0.05541348 0.04345265 0.04430689]
fit 8 nuis True held 1 p [0.      0.02289 1.67798 0.5683  0.      0.      0.      0.      1.20016
 0.65775 0.      0.     ] rmse [0.0475265  0.04032179 0.04998683]
fit 8 nuis True held 2 p [ 0.       0.       0.57316 15.       0.       0.       0.       0.
  0.4405  15.       0.       0.     ] rmse [0.02256178 0.01988215 0.09667094]
fit 12 nuis False held -1 p [0.      0.      1.04098 1.03975 1.15563 1.09059 0.      0.      0.99536
 1.00818 0.77266 1.55949] rmse [4.492e-06 4.466e-06 4.557e-06]
fit 12 nuis False held 0 p [0.      0.      1.04088 1.03983 1.15557 1.09067 0.      0.      0.99541
 1.00803 0.77272 1.55935] rmse [4.682e-06 4.468e-06 4.534e-06]
fit 12 nuis False held 1 p [0.      0.      1.04084 1.03986 1.15552 1.09068 0.      0.      0.99558
 1.00792 0.77278 1.55915] rmse [4.471e-06 5.235e-06 4.527e-06]
fit 12 nuis False held 2 p [0.      0.      1.04086 1.03973 1.15541 1.09055 0.      0.      0.9954
 1.00798 0.77273 1.5594 ] rmse [4.473e-06 4.464e-06 8.545e-06]
fit 12 nuis True held -1 p [0.      0.      1.04111 1.03967 1.15573 1.09049 0.      0.      0.99532
 1.00819 0.77265 1.55952] rmse [4.415e-06 4.385e-06 4.500e-06]
fit 12 nuis True held 0 p [0.      0.      1.04101 1.03975 1.15568 1.09058 0.      0.      0.99539
 1.00805 0.7727  1.55939] rmse [4.596e-06 4.389e-06 4.477e-06]
fit 12 nuis True held 1 p [0.      0.      1.04096 1.03978 1.15562 1.09059 0.      0.      0.99553
 1.00795 0.77276 1.55922] rmse [4.400e-06 5.043e-06 4.472e-06]
fit 12 nuis True held 2 p [0.      0.      1.04096 1.03966 1.15546 1.09046 0.      0.      0.99539
 1.00793 0.77274 1.55942] rmse [4.3880e-06 4.3830e-06 1.0016e-05]
Y ranges [0.0584202  0.0462334  0.04532672] [0.36283274 0.37338512 0.45483882]
canonical sweep delta 0.0 outlier rows 0
```

### Turn 5 — `execute_in_remote_docker` (`25352 ms` LLM)
- **Rationale**: Express the fitted inward and wrap terms in physical flux units. Agreement with face averages and row-local upwinding would support a transferable stencil; disagreement would isolate the terms that need further pulse tests.
- **Alternatives Weighed**: `Freeze all fitted row multipliers` (rejected: They predict held-out pulses on this grid but may not transfer when endpoint transport coefficients or grid spacing change.)
- **ATMS NOGOOD DECLARED**: Refuted `Adjusting local wall loss and inward flux alone is sufficient.` | Witness: `The local-only eight-parameter family reaches 0.0972 RMSE on its worst held-out pulse; adding opposing-edge terms lowers that to 8.55e-6.` | Blanket Diagnosis: `The local basis omits opposing-edge transfer; nuisance projection does not remove its tracer signature.` | Sublated Pivot: `Identify the exact diffusion average, upwind direction, and one-way closure represented by the fitted coefficients.`

```bash
cd /tmp && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
from probe_wall import *
pfit=np.array([[0,0,1.04098,1.03975,1.15563,1.09059],[0,0,.99536,1.00818,.77266,1.55949]])
for j,q in enumerate('xy'):
 k=st['kappa_'+q]; v=st['vel_'+q]; h=st['L'+q]/(len(k)-1); e=np.array([0,-1]); adj=np.array([1,-2]); a=(k[e]+k[adj])/2; b=np.repeat(k[e].mean(),2)
 print(q,'endpoint kappa',k[e],'adjacent kappa',k[adj],'endpoint vel',v[e],'h',h)
 print('inward diff',a/k[e],'inward upwind', (a+h*np.maximum(v[e]*[-1,1],0))/k[e], 'wrap diff',b/k[e],'wrap upwind',(b+h*np.maximum(v[e]*[1,-1],0))/k[e])
 print('fit inward effective k',pfit[j,2:4]*k[e],'fit wrap effective k',pfit[j,4:]*k[e])
 print('implied inward velocity contribution',(pfit[j,2:4]*k[e]-a)/h,'implied wrap velocity contribution',(pfit[j,4:]*k[e]-b)/h)
 print('all velocities',v.round(6))
PY
```
- **Container Exit Code**: `0` (`503 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
x endpoint kappa [0.41682986 0.41682986] adjacent kappa [0.45033405 0.45033405] endpoint vel [ 0.35       -0.20572484] h 0.1846153846153846
inward diff [1.04018929 1.04018929] inward upwind [1.04018929 1.04018929] wrap diff [1. 1.] wrap upwind [1.15501621 1.09111624]
fit inward effective k [0.43391154 0.43339884] fit wrap effective k [0.48170109 0.45459047]
implied inward velocity contribution [ 0.00178528 -0.00099184] implied wrap velocity contribution [0.35138583 0.20453667]
all velocities [ 0.35      0.345004  0.330159  0.305889  0.272886  0.232093  0.184674
  0.131983  0.075524  0.01691  -0.042188 -0.100081 -0.155117 -0.205725]
y endpoint kappa [0.52791443 0.28791443] adjacent kappa [0.52305359 0.29277527] endpoint vel [-0.         -0.22652476] h 0.18181818181818182
inward diff [0.99539618 1.00844147] inward upwind [0.99539618 1.00844147] wrap diff [0.77269043 1.4167905 ] wrap upwind [0.77269043 1.55984106]
fit inward effective k [0.52546491 0.29026957] fit wrap effective k [0.40789836 0.44899968]
implied inward velocity contribution [-0.00010506 -0.00041405] implied wrap velocity contribution [-8.83672252e-05  2.25968848e-01]
all velocities [-0.       -0.055605 -0.108996 -0.158044 -0.200797 -0.235551 -0.260922
 -0.275899 -0.279886 -0.272723 -0.254697 -0.226525]
```

### Turn 6 — `execute_in_remote_docker` (`19448 ms` LLM)
- **Rationale**: If the parameter-free stencil matches the fitted error across all pulses, shifts, and banks, use it in the downstream solver. A structured residual would identify which flux or calibration term still needs work. Read the upstream and validator contracts before wiring the solver.
- **Alternatives Weighed**: `Deploy the fitted endpoint multipliers` (rejected: The physical stencil reproduces those coefficients nearly exactly and transfers with grid spacing and endpoint transport parameters; test its pulse predictions directly.)

```bash
cd /tmp && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
from probe_wall import *
p=[]
for q in 'xy':
 k=st['kappa_'+q]; v=st['vel_'+q]; h=st['L'+q]/(len(k)-1)
 ends=k[[0,-1]]
 inward=(0.5*(ends+k[[1,-2]])+h*np.maximum(v[[0,-1]]*[-1,1],0))/ends
 wrap=(ends.mean()+h*np.maximum(v[[0,-1]]*[1,-1],0))/ends
 p.extend([0,0,*inward,*wrap])
r=err(p)
print('periodic face-average upstream flux: rmse by pulse',metrics(r),'max abs',abs(r).max())
print('rmse by shift and pulse',np.sqrt((r.reshape(len(st['mu_shifts']),-1,3)**2).mean(axis=1)))
print('rmse by bank and pulse',np.sqrt((r.reshape(-1,4,3)**2).mean(axis=0)))
PY
cd /holdout_task && cat _verified_upstream.py validate_public_schema.py && sed -n '48,95p' /tmp/hegel_continual_archive/beta_m99_c01_batch_01_solver.py && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance projection profiling' | head -65
```
- **Container Exit Code**: `0` (`546 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
t(0 if len(errs) == 0 else 1)
def _box_simplex(hessian, linear, cap, budget):
    """Primal active set, using exact equality-constrained solves on each face."""
    n = len(linear)
    if budget == 0 or budget == n * cap:
        w = np.full(n, budget / n)
        g = hessian @ w + linear
        return w, float(g.min() if budget == 0 else g.max())
    w = np.full(n, budget / n)
    status = np.zeros(n, dtype=int)  # -1: lower; +1: upper
    tol = 1e-12 * max(1., np.max(np.abs(linear)), cap * np.max(np.abs(hessian)))
    for _ in range(20 * n + 20):
        free = np.flatnonzero(status == 0)
        fixed = np.flatnonzero(status)
        if not len(free):
            g = hessian @ w + linear
            lower, upper = np.flatnonzero(status == -1), np.flatnonzero(status == 1)
            a, b = lower[g[lower].argmin()], upper[g[upper].argmax()]
            if g[a] >= g[b] - tol:
                return w, float((g[a] + g[b]) / 2)
            status[[a, b]] = 0
            continue
        hff = hessian[np.ix_(free, free)]
        cf = linear[free] + hessian[np.ix_(free, fixed)] @ w[fixed]
        q, t = np.linalg.solve(hff, np.column_stack((-cf, np.ones(len(free))))).T
        nu = (budget - w[fixed].sum() - q.sum()) / t.sum()
        target = q + nu * t
        step = target - w[free]
        ratio = np.full(len(free), np.inf)
        lower, upper = (target < 0) & (step < 0), (target > cap) & (step > 0)
        ratio[lower] = -w[free[lower]] / step[lower]
        ratio[upper] = (cap - w[free[upper]]) / step[upper]
        k = int(ratio.argmin())
        if ratio[k] < 1:
            w[free] += ratio[k] * step
            status[free[k]] = -1 if lower[k] else 1
            w[free[k]] = 0. if lower[k] else cap
            continue
        w[free] = target
        g = hessian @ w + linear
        release = status * (g - nu)
        k = int(release.argmax())
        if release[k] <= tol:
            return w, float(nu)
        status[k] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 7 — `execute_in_remote_docker` (`34687 ms` LLM)
- **Rationale**: The exact periodic stencil matches all three tracer pulses without fitted wall coefficients. Implement it with the specified nuisance projection and box-simplex certificate, preserve the staged wrapper, and run the public schema check before independently validating the field and certificate.
- **Alternatives Weighed**: `Recalibrate the observations inside the downstream module` (rejected: The staged upstream wrapper already transforms the observations into canonical coordinates; downstream recalibration would apply that correction twice.)

```bash
set -eu
cd /holdout_task
cat > downstream_solver.py <<'PY'
"""Periodic transport-reaction inversion in calibrated sensor coordinates."""
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, vel, h):
    # Each endpoint is a regular periodic row, sharing a face with the other endpoint.
    face = (kappa + np.roll(kappa, -1)) / (2 * h**2)
    west = np.roll(face, 1) + np.maximum(vel, 0.) / h
    east = face + np.maximum(-vel, 0.) / h
    mat = diags((-west[1:], west + east, -east[:-1]), (-1, 0, 1), format='lil')
    mat[0, -1] -= west[0]
    mat[-1, 0] -= east[-1]
    return mat.tocsc()


def _forward(study):
    nx, ny = study['nx'], study['ny']
    axes = [_axis(np.asarray(study['kappa_' + d]), np.asarray(study['vel_' + d]),
                  study['L' + d] / (n - 1))
            for d, n in zip('xy', (nx, ny))]
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    op = (kron(eye(ny), axes[0]) + kron(axes[1], eye(nx)) + diags(reaction)).tocsc()
    ident = eye(nx * ny, format='csc')
    modes = np.asarray(study['source_modes'])
    rhs = np.asfortranarray(modes.transpose(0, 2, 1).reshape(len(modes), -1).T)
    ix, iy = np.asarray(study['sensor_ij']).T
    rows = ix + nx * iy
    blocks = []
    for shift in study['mu_shifts']:
        field = splu(op + shift * ident).solve(rhs)
        blocks.append(field[rows, :])
    terminal_modes = field.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)
    return np.vstack(blocks), terminal_modes


PY
sed -n '/^def _box_simplex(/,/^def solve_batch(/p' /tmp/hegel_continual_archive/beta_m99_c01_batch_01_solver.py | sed '$d' >> downstream_solver.py
cat >> downstream_solver.py <<'PY'
def solve_downstream(cal_item: dict) -> dict:
    study = cal_item
    sensors, terminal_modes = _forward(study)
    y = np.asarray(study['y_obs_raw'], dtype=float)
    z = np.asarray(study['Z_nuisance'], dtype=float)
    projector = np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    sensors = sensors - z @ (projector @ sensors)
    y = y - z @ (projector @ y)

    ridge = study['source_ridge']
    cap, budget = study['u_box'], study['total_budget']
    hessian = sensors.T @ sensors + ridge * np.eye(sensors.shape[1])
    w, nu = _box_simplex(hessian, -sensors.T @ y, cap, budget)
    residual = sensors @ w - y
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    gradient = sensors.T @ residual + ridge * w
    reduced = gradient - nu
    dual = objective - gradient @ w + nu * budget + cap * np.minimum(reduced, 0.).sum()
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(),
            'nu_sum': nu,
            'lam_lower': np.maximum(reduced, 0.).tolist(),
            'lam_upper': np.maximum(-reduced, 0.).tolist(),
            'primal_objective': objective,
            'dual_lower_bound': float(dual),
        },
    }
PY
OPENBLAS_NUM_THREADS=1 python3 validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`441 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=44caeec2be0fc139 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_44caeec2 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`40323 ms` LLM)
- **Rationale**: Use a clean process and an independently assembled dense operator to test tracer slices and terminal fields. Compare weights against an independent optimizer and stress the certificate at tight and fully saturated budgets; a failure will isolate an operator, indexing, or active-set defect.
- **Alternatives Weighed**: `Submit after schema validation alone` (rejected: The schema check cannot detect transposed field coordinates, wrong endpoint fluxes, or an invalid KKT certificate.)

```bash
set -eu
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys, pickle
sys.path.insert(0, '/holdout_task')
import numpy as np
from scipy.optimize import minimize
from batch_solver import solve_batch, _PAY
from _verified_upstream import calibrate_upstream
from downstream_solver import _box_simplex

items = pickle.load(open('public_batch.pkl', 'rb'))
pulse_worst = gap_worst = field_worst = 0.
for item in items:
    raw = item['study']
    st = calibrate_upstream(raw, _PAY)
    nx, ny = st['nx'], st['ny']
    n = nx * ny
    A = np.diag(st['V_reaction_2d'].flatten(order='F'))
    for x in range(nx):
        for y in range(ny):
            row = x + nx*y
            for axis, i, size, stride in [('x', x, nx, 1), ('y', y, ny, nx)]:
                k, v = st['kappa_'+axis], st['vel_'+axis][i]
                h = st['L'+axis]/(size-1)
                for step in (-1, 1):
                    j = (i+step) % size
                    rate = (k[i]+k[j])/(2*h*h) + max(-step*v, 0)/h
                    A[row, row] += rate
                    A[row, row+(j-i)*stride] -= rate
    rhs = np.column_stack([mode.flatten(order='F') for mode in st['source_modes']])
    ix, iy = st['sensor_ij'].T
    fields = [np.linalg.solve(A+mu*np.eye(n), rhs) for mu in st['mu_shifts']]
    S = np.vstack([field[ix+nx*iy] for field in fields])
    for p in item.get('campaign_diagnostics') or []:
        y = calibrate_upstream(dict(raw, y_obs_raw=p['y_pulse_raw']), _PAY)['y_obs_raw']
        e = (S@p['w_pulse']-y).reshape(len(fields), -1, 4)
        pulse_worst = max(pulse_worst, float(np.sqrt((e**2).mean(axis=1)).max()))
    z = st['Z_nuisance']
    P = np.eye(len(z))-z@np.linalg.solve(z.T@z+st['nuisance_ridge']*np.eye(z.shape[1]),z.T)
    S, y = P@S, P@st['y_obs_raw']
    r, cap, b = st['source_ridge'], st['u_box'], st['total_budget']
    f = lambda w: .5*(np.linalg.norm(S@w-y)**2+r*(w@w))
    g = lambda w: S.T@(S@w-y)+r*w
    out = solve_batch(raw)
    w, c = out['weights'], out['kkt_certificate']
    lower, upper = np.array(c['lam_lower']), np.array(c['lam_upper'])
    gap = f(w)-c['dual_lower_bound']
    gap_worst = max(gap_worst, abs(gap))
    assert abs(w.sum()-b)<1e-11 and w.min()>-1e-11 and w.max()<cap+1e-11
    assert np.array_equal(w, c['weights']) and abs(f(w)-c['primal_objective'])<1e-11
    assert min(lower.min(),upper.min())>=0 and np.max(np.abs(g(w)-c['nu_sum']-lower+upper))<1e-11
    assert max(np.max(np.abs(lower*w)),np.max(np.abs(upper*(cap-w))),abs(gap))<1e-11
    field_error = np.max(np.abs(out['terminal_field']-(fields[-1]@w).reshape(nx,ny,order='F')))
    field_worst = max(field_worst, float(field_error))
    assert field_error<1e-11
    opt = minimize(f,np.full(len(w),b/len(w)),jac=g,method='SLSQP',bounds=[(0,cap)]*len(w),constraints=[{'type':'eq','fun':lambda x:x.sum()-b,'jac':lambda x:np.ones_like(x)}],options={'ftol':1e-14})
    assert opt.success and np.max(np.abs(opt.x-w))<2e-6
assert pulse_worst < 1e-5
rng = np.random.default_rng(240)
stress_worst = 0.
for size in (3, 8, 20):
    for b in (0., .001, .42, .84, .42*size-.001, .42*size):
        for _ in range(12):
            X = rng.normal(size=(size,size)); H = X.T@X+0.04*np.eye(size)
            q = rng.normal(size=size)*2
            w,nu = _box_simplex(H,q,.42,b)
            d = H@w+q-nu
            gap = np.maximum(d,0)@w + np.maximum(-d,0)@(.42-w)
            stress_worst = max(stress_worst,abs(gap))
            assert abs(w.sum()-b)<1e-10 and w.min()>-1e-10 and w.max()<.42+1e-10 and abs(gap)<1e-10
print(f'public_studies={len(items)} worst_pulse_shift_bank_rmse={pulse_worst:.6g} terminal_max_error={field_worst:.3g} public_max_dual_gap={gap_worst:.3g} stress_cases=216 stress_max_dual_gap={stress_worst:.3g}')
PY
```
- **Container Exit Code**: `0` (`514 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public_studies=1 worst_pulse_shift_bank_rmse=8.48675e-06 terminal_max_error=2.22e-16 public_max_dual_gap=7.68e-16 stress_cases=216 stress_max_dual_gap=2.96e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=44caeec2be0fc139 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_44caeec2 (directory).
```

### Turn 9 — `submit_completed_task` (`12014 ms` LLM)
- **Rationale**: The final bundle passed public schema validation and clean-process checks against an independently assembled transport operator and an independent optimizer. The parameter-free periodic stencil reproduced the tracer measurements across all tested slices.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:408', '/holdout_task/downstream_solver.py': 'FILE:4790'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Completed and verified both solver modules. The public schema passed. Worst tracer-slice RMSE was 8.49e-6, terminal-field error against an independent solve was 2.22e-16, and the maximum duality gap across 216 stress cases was 2.96e-14.`

### Turn 10 — `execute_in_remote_docker` (`30258 ms` LLM)
- **Rationale**: Withdraw wrap flux on each axis separately to test whether both couplings are required after nuisance profiling. Compare the staged sensor calibration against a sweep refit on known-weight tracers; revisit calibration only if the difference is material.
- **Alternatives Weighed**: `Run relay, drift-equation, or classification checks` (rejected: This task has no relay state, stochastic drift equation, or class labels. The relevant confounding test is whether nuisance profiling masks an incorrect transport boundary.)

```bash
set -eu
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, sys
sys.path.insert(0, '/holdout_task')
import numpy as np
from batch_solver import _PAY, solve_batch
from _verified_upstream import calibrate_upstream
from downstream_solver import solve_downstream, _forward
sys.path.insert(0, '/tmp')
from probe_wall import st, forward, W, Y

# Withdraw wrap flux on each axis separately while retaining its local diffusion.
periodic, reflecting = [], []
for d in 'xy':
    k, v = st['kappa_'+d], st['vel_'+d]
    h = st['L'+d]/(len(k)-1)
    end = k[[0,-1]]
    inward = (0.5*(end+k[[1,-2]]) + h*np.maximum(v[[0,-1]]*[-1,1],0))/end
    wrap = (end.mean()+h*np.maximum(v[[0,-1]]*[1,-1],0))/end
    periodic.append([0,0,*inward,*wrap])
    reflecting.append([0,0,*inward,0,0])
z = st['Z_nuisance']
P = np.eye(len(z))-z@np.linalg.solve(z.T@z+st['nuisance_ridge']*np.eye(z.shape[1]),z.T)
def worst(e):
    return np.sqrt(np.mean(e.reshape(len(st['mu_shifts']),-1,4,e.shape[1])**2,axis=1)).max()
for label, rows in [('both_periodic',periodic),('without_x_wrap',[reflecting[0],periodic[1]]),('without_y_wrap',[periodic[0],reflecting[1]])]:
    e = forward(rows)[0]-Y
    print(f'{label}: worst_raw={worst(e):.8g} worst_profiled={worst(P@e):.8g}')

# Check whether refitting the sensor sweeps, instead of using the verified
# parameter payload, would materially change the fit or the certified solution.
for item in pickle.load(open('public_batch.pkl','rb')):
    raw = item['study']
    fixed = solve_batch(raw)
    fitted = solve_downstream(calibrate_upstream(raw))
    S, _ = _forward(calibrate_upstream(raw,_PAY))
    residuals = {name: [] for name in ('staged','sweep_refit')}
    for pulse in item.get('campaign_diagnostics') or []:
        pulse_study = dict(raw,y_obs_raw=pulse['y_pulse_raw'])
        for name,payload in [('staged',_PAY),('sweep_refit',None)]:
            obs = calibrate_upstream(pulse_study,payload)['y_obs_raw']
            residuals[name].append(S@pulse['w_pulse']-obs)
    for name, columns in residuals.items():
        if columns:
            print(f'{name}: worst_tracer_slice={worst(np.column_stack(columns)):.8g}')
    print('refit_max_weight_change', np.max(abs(fitted['weights']-fixed['weights'])), 'refit_max_field_change', np.max(abs(fitted['terminal_field']-fixed['terminal_field'])))
    for name,out in [('staged',fixed),('sweep_refit',fitted)]:
        c=out['kkt_certificate']
        print(name,'certificate_gap',c['primal_objective']-c['dual_lower_bound'])
PY
```
- **Container Exit Code**: `0` (`541 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
both_periodic: worst_raw=8.4867456e-06 worst_profiled=8.4882744e-06
without_x_wrap: worst_raw=0.078428187 worst_profiled=0.077573226
without_y_wrap: worst_raw=0.091313196 worst_profiled=0.092055344
staged: worst_tracer_slice=8.4867456e-06
sweep_refit: worst_tracer_slice=1.1749533e-06
refit_max_weight_change 3.858711422810579e-05 refit_max_field_change 4.214770314842076e-06
staged certificate_gap 7.563394355258879e-16
sweep_refit certificate_gap 1.124100812432971e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=44caeec2be0fc139 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_44caeec2 (directory).
```

### Turn 11 — `submit_completed_task` (`14841 ms` LLM)
- **Rationale**: Removing either periodic coupling raised the worst nuisance-profiled tracer RMSE above 0.077, confirming that both couplings are needed. The unchanged final bundle passed the public schema and clean-process field and KKT checks.
- **Alternatives Weighed**: `Replace the verified upstream calibration with a sweep refit` (rejected: The refit lowered tracer error but changed the terminal field by only 4.21e-6; the task specifies the staged upstream composition and a downstream-only repair.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:408', '/holdout_task/downstream_solver.py': 'FILE:4790'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Verified both solver modules. The public schema passed. Worst tracer RMSE across pulse, shift, and sensor-bank slices was 8.49e-6. The terminal field matched an independent solve to 2.22e-16, and the largest KKT duality gap across 216 stress cases was 2.96e-14.`

### Turn 12 — `execute_in_remote_docker` (`30073 ms` LLM)
- **Rationale**: Test the shipped forward operator against two simpler periodic stencils on the measured tracer slices, with nuisance profiling held fixed. If either simplification materially improves the worst slice, revisit that endpoint term; otherwise keep the verified bundle.
- **Alternatives Weighed**: `Repeat the axis-withdrawal test` (rejected: The previous test established that both axes need wrap coupling. It did not separately test the shared diffusion average and signed endpoint advection.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, sys
sys.path.insert(0, '/holdout_task')
import numpy as np
from scipy.sparse import diags
from batch_solver import _PAY, solve_batch
from _verified_upstream import calibrate_upstream
import downstream_solver as dm

original_axis = dm._axis

def ablated_axis(kind):
    def axis(k, v, h):
        face = (k + np.roll(k, -1)) / (2*h*h)
        west = np.roll(face, 1) + np.maximum(v, 0)/h
        east = face + np.maximum(-v, 0)/h
        if kind == 'no_wrap_upwind':
            west[0] -= max(v[0], 0)/h
            east[-1] -= max(-v[-1], 0)/h
        elif kind == 'endpoint_local_wrap_diffusion':
            west[0] += (k[0]-k[-1])/(2*h*h)
            east[-1] += (k[-1]-k[0])/(2*h*h)
        A = diags((-west[1:], west+east, -east[:-1]), (-1,0,1), format='lil')
        A[0,-1] -= west[0]
        A[-1,0] -= east[-1]
        return A.tocsc()
    return axis

for item in pickle.load(open('public_batch.pkl','rb')):
    raw = item['study']
    st = calibrate_upstream(raw, _PAY)
    pulses = item['campaign_diagnostics']
    if not pulses:
        continue
    W = np.column_stack([p['w_pulse'] for p in pulses])
    Y = np.column_stack([calibrate_upstream(dict(raw, y_obs_raw=p['y_pulse_raw']), _PAY)['y_obs_raw'] for p in pulses])
    z = st['Z_nuisance']
    P = np.eye(len(z))-z@np.linalg.solve(z.T@z+st['nuisance_ridge']*np.eye(z.shape[1]), z.T)
    def slices(e):
        return np.sqrt(np.mean(e.reshape(len(st['mu_shifts']),-1,4,len(pulses))**2,axis=1))
    S, _ = dm._forward(st)
    base = slices(P@(S@W-Y))
    print('final_artifact_profiled_worst_slice', base.max())
    assert base.max() < 1e-5
    for kind in ('no_wrap_upwind', 'endpoint_local_wrap_diffusion'):
        dm._axis = ablated_axis(kind)
        try:
            alternative, _ = dm._forward(st)
        finally:
            dm._axis = original_axis
        trial = slices(P@(alternative@W-Y))
        print(kind, 'profiled_worst_slice', trial.max(), 'improved_slices', np.count_nonzero(trial < base), 'of', base.size)
    out = solve_batch(raw)
    c = out['kkt_certificate']
    assert np.isfinite(out['terminal_field']).all() and abs(c['primal_objective']-c['dual_lower_bound']) < 1e-11
PY
```
- **Container Exit Code**: `0` (`407 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
final_artifact_profiled_worst_slice 8.488274353345981e-06
no_wrap_upwind profiled_worst_slice 0.004079689958057656 improved_slices 0 of 36
endpoint_local_wrap_diffusion profiled_worst_slice 0.042739983310573736 improved_slices 0 of 36

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=44caeec2be0fc139 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_44caeec2 (directory).
```

### Turn 13 — `submit_completed_task` (`13670 ms` LLM)
- **Rationale**: The unchanged bundle passed clean-process schema, field, and KKT verification. The tracer ablations confirm that both wrap couplings, shared-face diffusion, and signed endpoint upwinding are needed even after nuisance profiling.
- **Alternatives Weighed**: `Omit endpoint wrap advection or use endpoint-local wrap diffusion` (rejected: Neither simplification improved any of the 36 measured slices. Worst profiled RMSE rose to 0.00408 and 0.0427, respectively.)
- **Submitted Final Answer / Summary**: `Both solver modules passed public schema validation. Worst tracer-slice RMSE was 8.49e-6. The terminal field matched an independent solve to 2.22e-16, and the maximum KKT duality gap across 216 stress cases was 2.96e-14.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M99-Autonomous-Decomposition-And-Interaction-Withdrawal-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
