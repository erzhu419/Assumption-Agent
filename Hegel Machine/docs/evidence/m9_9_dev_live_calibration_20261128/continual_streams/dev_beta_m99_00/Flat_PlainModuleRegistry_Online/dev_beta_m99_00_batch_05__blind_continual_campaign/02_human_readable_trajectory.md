# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_beta_m99_00_batch_05__blind_continual_campaign__downstream_module`

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
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `352,629` (`340,594` / `156,160` / `12,035`) |
| **Total Episode Wall Time (s)** | `347.02s` |

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

### Turn 1 — `execute_in_remote_docker` (`15384 ms` LLM)
- **Rationale**: Inspect the transducer contract, public input shapes, and apparatus setting first: a mismatch would change which boundary rows or calibration steps the downstream solver implements. Check for a reusable ridge profiler before writing one.
- **Alternatives Weighed**: `Implement open-wall Robin immediately` (rejected: The active boundary may differ, and the apparatus log is the specified evidence when diagnostic pulses are absent.)

```bash
cd /holdout_task && cat batch_solver.py _verified_upstream.py _upstream_param_payload.json validate_public_schema.py && ls -l /tmp/hegel_continual_archive/ && cat /tmp/hegel_continual_archive/archive_apparatus_log.json && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling'; python3 - <<'PY'
import pickle
with open('/holdout_task/public_batch.pkl','rb') as f: items=pickle.load(f)
print('items',len(items))
for item in items[:3]:
 print('item keys',item.keys(),'diagnostics',None if item.get('campaign_diagnostics') is None else len(item['campaign_diagnostics']))
 for k,v in item['study'].items(): print(k,type(v).__name__,getattr(v,'shape',None),v if isinstance(v,(int,float,str,bool)) else (list(v) if isinstance(v,dict) else ''))
PY
```
- **Container Exit Code**: `0` (`362 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0493,
        0.30619,
        -0.25364,
        0.70826,
        0.86749
      ],
      [
        0.79766,
        0.38836,
        2.55698,
        0.54855,
        -0.07793,
        -0.33995,
        0.57479,
        0.69369
      ],
      [
        0.78004,
        -0.05658,
        2.61727,
        0.70501,
        -0.5147,
        -0.21968,
        0.40949,
        0.39319
      ],
      [
        0.76442,
        -0.42761,
        2.56728,
        0.87352,
        -0.86216,
        -0.36967,
        0.18026,
        -0.07478
      ],
      [
        0.60407,
        -0.74989,
        2.48018,
        0.83467,
        -0.84049,
        -0.2176,
        -0.05033,
        -0.45995
      ],
      [
        0.4325,
        -0.8174,
        2.55816,
        0.8479,
        -0.71665,
        -0.30366,
        -0.1845,
        -0.76311
      ],
      [
        0.21046,
        -0.70163,
        2.65786,
        0.73942,
        -0.32966,
        -0.42505,
        -0.42878,
        -0.8407
      ],
      [
        0.03456,
        -0.41989,
        2.55667,
        0.58386,
        0.16969,
        -0.26126,
        -0.62119,
        -0.73675
      ],
      [
        -0.20911,
        0.06628,
        2.58959,
        0.46324,
        0.58454,
        -0.34843,
        -0.7245,
        -0.35233
      ]
    ]
  }
]autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
items 1
item keys dict_keys(['study', 'campaign_diagnostics']) diagnostics 3
Lx float None 2.4
Ly float None 2.0
nx int None 14
ny int None 12
kappa_x ndarray (14,) 
kappa_y ndarray (12,) 
vel_x ndarray (14,) 
vel_y ndarray (12,) 
robin_x float None 1.2738792292989902
robin_y float None 1.270567880377232
V_reaction_2d ndarray (14, 12) 
mu_shifts ndarray (3,) 
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

### Turn 2 — `execute_in_remote_docker` (`19766 ms` LLM)
- **Rationale**: Inspect archived boundary models and the calibrated input contract before fitting. This will distinguish reusable exact stencils from cases requiring a new model, and establish how to leave one pulse out for validation.
- **Alternatives Weighed**: `Infer the current wall solely from prefix telemetry` (rejected: The current campaign has three direct diagnostic pulses; archive evidence can supply candidate stencils, but pulse prediction should decide between them.)

```bash
cd /holdout_task; tail -30 _verified_upstream.py; cat validate_public_schema.py; ls -lh /tmp/hegel_continual_archive; grep -inE 'robin|boundary|apparatus|pulse|wrap|wall|periodic' /tmp/hegel_continual_archive/*solver.py | cut -c1-235 | head -100; python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/'
a=json.load(open(p+'archive_apparatus_log.json')); print('log',type(a).__name__,len(a));
for x in (a if isinstance(a,list) else a.values()):
 if isinstance(x,dict): print({k:(v if isinstance(v,(str,int,float,bool,type(None))) else (type(v).__name__,len(v))) for k,v in x.items()})
i=pickle.load(open('public_batch.pkl','rb'))[0];print('pulses', [{k:(np.shape(v),np.min(v),np.max(v)) for k,v in q.items()} for q in i['campaign_diagnostics']]); print('sensor',i['study']['sensor_ij'][:8]);print('shifts',i['study']['mu_shifts'])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -25
```
- **Container Exit Code**: `0` (`384 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lver.py:1:"""Reconstruct the open-wall chamber and solve its box-simplex inverse problem."""
/tmp/hegel_continual_archive/dev_beta_m99_00_s00_solver.py:9:def _axis(kappa, vel, h, robin):
/tmp/hegel_continual_archive/dev_beta_m99_00_s00_solver.py:15:    diagonal = np.r_[-upper[0] + robin / h,
/tmp/hegel_continual_archive/dev_beta_m99_00_s00_solver.py:16:                     -lower[:-1] - upper[1:], -lower[-1] + robin / h]
/tmp/hegel_continual_archive/dev_beta_m99_00_s00_solver.py:23:                    study['L' + key] / (n - 1), study['robin_' + key])
log list 6
{'completed_campaign_index': 0, 'completed_campaign_short_name': 'facility_init', 'facility_telemetry_trace': ('list', 24)}
{'completed_campaign_index': 1, 'completed_campaign_short_name': 'dev_beta_m99_00_s00', 'facility_telemetry_trace': ('list', 24)}
{'completed_campaign_index': 2, 'completed_campaign_short_name': 'dev_beta_m99_00_batch_01', 'facility_telemetry_trace': ('list', 24)}
{'completed_campaign_index': 3, 'completed_campaign_short_name': 'dev_beta_m99_00_batch_02', 'facility_telemetry_trace': ('list', 24)}
{'completed_campaign_index': 4, 'completed_campaign_short_name': 'dev_beta_m99_00_batch_03', 'facility_telemetry_trace': ('list', 24)}
{'completed_campaign_index': 5, 'completed_campaign_short_name': 'dev_beta_m99_00_batch_04', 'facility_telemetry_trace': ('list', 24)}
pulses [{'w_pulse': ((8,), 0.0, 1.0), 'y_pulse_raw': ((96,), -0.05492766515519896, 0.8051492141461674)}, {'w_pulse': ((8,), 0.0, 1.0), 'y_pulse_raw': ((96,), -0.049861993503312836, 0.8295891686557957)}, {'w_pulse': ((8,), 0.0, 1.0), 'y_pulse_raw': ((96,), -0.037860864169945024, 0.7803123432262677)}]
sensor [[ 8  0]
 [ 4  4]
 [ 0 10]
 [12  2]
 [ 8  4]
 [12  8]
 [ 0  4]
 [10  6]]
shifts [0.15 0.55 1.1 ]
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
```

### Turn 3 — `execute_in_remote_docker` (`16168 ms` LLM)
- **Rationale**: Read only archived stencil definitions and compact telemetry summaries. The formulas establish the discrete candidates, and pulse excitation and edge coverage show which wall parameters the three measurements can distinguish.
- **Alternatives Weighed**: `Reuse the last verified solver unchanged` (rejected: Its periodic stencil may not match this campaign; extract the exact archived row formulas for comparison against the current pulses.)

```bash
cd /holdout_task; for p in /tmp/hegel_continual_archive/*solver.py; do echo "=== $p"; sed -n '1,45p' "$p"; done; cat /tmp/hegel_continual_archive/archive_manifest.json; python3 - <<'PY'
import json,numpy as np,pickle
for a in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 x=np.array(a['facility_telemetry_trace']); print(a['completed_campaign_short_name'], 'mean', x.mean(0).round(4), 'std',x.std(0).round(4))
i=pickle.load(open('public_batch.pkl','rb'))[0]; x=i['study']['aux_manifold_telemetry'];print('current aux',x.mean(0),x.std(0));print('pulse weights', [p['w_pulse'] for p in i['campaign_diagnostics']]); print('row sensor bank',i['study']['row_bank_id'][:40]); print('sensor edge counts',[(i['study']['sensor_ij'][:,k]==j).sum() for k,j in [(0,0),(0,13),(1,0),(1,11)]])
PY
```
- **Container Exit Code**: `0` (`339 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[
  {
    "episode_short_name": "dev_beta_m99_00_s00",
    "task_id": "dev_beta_m99_00_s00__blind_continual_campaign",
    "entrypoint": "solve_batch",
    "output_schema_keys": [
      "kkt_certificate",
      "terminal_field",
      "weights"
    ],
    "solver_path": "/tmp/hegel_continual_archive/dev_beta_m99_00_s00_solver.py"
  },
  {
    "episode_short_name": "dev_beta_m99_00_batch_01",
    "task_id": "dev_beta_m99_00_batch_01__blind_continual_campaign",
    "entrypoint": "solve_batch",
    "output_schema_keys": [
      "kkt_certificate",
      "terminal_field",
      "weights"
    ],
    "solver_path": "/tmp/hegel_continual_archive/dev_beta_m99_00_batch_01_solver.py"
  },
  {
    "episode_short_name": "dev_beta_m99_00_batch_02",
    "task_id": "dev_beta_m99_00_batch_02__blind_continual_campaign",
    "entrypoint": "solve_downstream",
    "output_schema_keys": [
      "kkt_certificate",
      "terminal_field",
      "weights"
    ],
    "solver_path": "/tmp/hegel_continual_archive/dev_beta_m99_00_batch_02_solver.py"
  }
]facility_init mean [-0.0099  0.0039 -0.4017 -0.0022  0.0055 -0.3396  0.0008 -0.0134] std [0.6068 0.5991 0.0596 0.5972 0.5993 0.0536 0.5994 0.6001]
dev_beta_m99_00_s00 mean [ 9.000e-04  3.200e-03 -3.870e-01  4.000e-04  9.000e-04  2.447e+00
  6.000e-03 -3.600e-03] std [0.6014 0.5974 0.0691 0.6103 0.6101 0.0657 0.6048 0.5966]
dev_beta_m99_00_batch_01 mean [ 1.0000e-04  1.8000e-03  2.5437e+00 -5.2000e-03  3.5000e-03 -3.3150e-01
  1.4000e-03 -2.6000e-03] std [0.6006 0.6029 0.0703 0.6045 0.6193 0.0599 0.6004 0.6088]
dev_beta_m99_00_batch_02 mean [-0.0061  0.003  -0.3974  0.0067 -0.0148 -0.3221 -0.0048  0.0083] std [0.6075 0.5955 0.0784 0.5919 0.6002 0.0668 0.604  0.5908]
dev_beta_m99_00_batch_03 mean [-2.2000e-03  4.2000e-03 -3.9200e-01  5.2000e-03  5.2000e-03  2.4499e+00
  7.7000e-03 -1.0000e-04] std [0.6005 0.6085 0.0591 0.6099 0.5972 0.0897 0.6035 0.6045]
dev_beta_m99_00_batch_04 mean [ 1.0000e-04  1.8000e-03  2.5437e+00 -5.2000e-03  3.5000e-03 -3.3150e-01
  1.4000e-03 -2.6000e-03] std [0.6006 0.6029 0.0703 0.6045 0.6193 0.0599 0.6004 0.6088]
current aux [-0.10378712 -0.4235833  -0.0143037   0.35344843] [1.07818827 0.92117887 0.82444969 1.05446353]
pulse weights [array([1. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. ]), array([0. , 1. , 0. , 0. , 0.5, 0. , 0. , 0. ]), array([0. , 0. , 1. , 0. , 0. , 0.5, 0. , 0. ])]
row sensor bank [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0
 1 2 3]
sensor edge counts [5, 0, 6, 0]
```

### Turn 4 — `execute_in_remote_docker` (`18799 ms` LLM)
- **Rationale**: Compare all three exact stencils against each calibrated pulse and shift, before and after nuisance projection. If none predicts at the measurement noise scale, use the residual structure to test mixed-axis or modified boundary rows.
- **Alternatives Weighed**: `Fit a new continuous wall model immediately` (rejected: The three archived stencils are parameter-free on this campaign and may already explain the pulses.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import importlib.util, glob, pickle, numpy as np
from _verified_upstream import calibrate_upstream
item=pickle.load(open('public_batch.pkl','rb'))[0]
st=calibrate_upstream(dict(item['study'],reference_tracer_pulses=item['campaign_diagnostics']))
pulses=st['reference_tracer_pulses']; W=np.array([p['w_pulse'] for p in pulses]).T; Y=np.array([p['y_pulse_raw'] for p in pulses]).T
Z=st['Z_nuisance']; P=np.linalg.solve(Z.T@Z+st['nuisance_ridge']*np.eye(Z.shape[1]), Z.T); nz=len(st['sensor_ij']); print('calibrated pulse RMS by shift',np.sqrt(np.mean(Y.reshape(-1,nz,3)**2,axis=1)).round(7))
for path in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 spec=importlib.util.spec_from_file_location('archive',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
 S,_=m._forward(st); E=S@W-Y; R=E-Z@(P@E)
 print(path.split('/')[-1], 'raw RMS by pulse',np.sqrt(np.mean(E**2,axis=0)).round(8),'profiled RMS',np.sqrt(np.mean(R**2,axis=0)).round(8),'raw max abs',np.max(abs(E)))
 print('RMS by shift',np.sqrt(np.mean(E.reshape(-1,nz,3)**2,axis=1)).round(8))
PY
```
- **Container Exit Code**: `0` (`454 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calibrated pulse RMS by shift [[0.3286479 0.3111083 0.3039068]
 [0.1705099 0.1620913 0.1614917]
 [0.1051963 0.1008904 0.1022562]]
dev_beta_m99_00_batch_01_solver.py raw RMS by pulse [0.04614634 0.03865226 0.04058586] profiled RMS [0.04480292 0.03777065 0.03981993] raw max abs 0.15987958585130913
RMS by shift [[0.06298088 0.05354766 0.0565252 ]
 [0.04099351 0.03385638 0.03531161]
 [0.02722859 0.02164221 0.02235237]]
dev_beta_m99_00_batch_02_solver.py raw RMS by pulse [0.04404511 0.03171357 0.02106219] profiled RMS [0.04343375 0.03132248 0.02089266] raw max abs 0.13079517270792757
RMS by shift [[0.06338189 0.04613718 0.03061363]
 [0.03581668 0.02532687 0.01678176]
 [0.02279951 0.01572139 0.01058422]]
dev_beta_m99_00_s00_solver.py raw RMS by pulse [0.15500379 0.14028639 0.13738573] profiled RMS [0.15473106 0.14010917 0.13717235] raw max abs 0.3365002058801423
RMS by shift [[0.24482632 0.22315623 0.21720247]
 [0.09983462 0.0878559  0.0881462 ]
 [0.04660089 0.03903139 0.04096165]]
```

### Turn 5 — `execute_in_remote_docker` (`35628 ms` LLM)
- **Rationale**: Compare all nine archived axis combinations, then fit inward, wrap, and leakage rates with each pulse held out in turn. The fitted rates are a diagnostic for missing discrete terms; held-out errors and Jacobian singular values will show which rates the data actually identify.
- **Alternatives Weighed**: `Choose periodic walls from the aggregate residual` (rejected: Periodic is the best shared stencil but still misses the first pulse by 0.04405 RMS; mixed axes and modified rows have not been tested.)
- **ATMS NOGOOD DECLARED**: Refuted `The shared open-wall Robin stencil explains the current tracer data.` | Witness: `Open-wall pulse RMS is 0.137–0.155 after calibration and changes negligibly after nuisance projection; sealed and periodic rows give substantially smaller errors on every pulse.` | Blanket Diagnosis: `The largest discrepancy is at the smallest reaction shift and is not explained by the measured nuisance subspace.` | Sublated Pivot: `Keep the specified interior stencil and compare exact mixed-axis boundaries before using fitted wall rates to identify missing terms.`

```bash
cat > /tmp/wall_fit.py <<'PY'
import sys
sys.path.insert(0,'/holdout_task')
import numpy as np, pickle
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from itertools import product
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=calibrate_upstream(dict(item['study'],reference_tracer_pulses=item['campaign_diagnostics']))
W=np.stack([p['w_pulse'] for p in s['reference_tracer_pulses']],axis=1)
Y=np.stack([p['y_pulse_raw'] for p in s['reference_tracer_pulses']],axis=1)
nx,ny=s['nx'],s['ny']; pos=s['sensor_ij']@np.array([1,nx]); V=diags(s['V_reaction_2d'].ravel(order='F')); I=eye(nx*ny,format='csc'); B=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T@W
base=[]; variants=[]
for q,n in zip('xy',(nx,ny)):
 k=s['kappa_'+q]; v=s['vel_'+q]; h=s['L'+q]/(n-1); r=s['robin_'+q]/h
 A=np.zeros((n,n)); j=np.arange(1,n-1); l=(k[j]+k[j-1])/(2*h*h)+np.maximum(v[j],0)/h; u=(k[j]+k[j+1])/(2*h*h)+np.maximum(-v[j],0)/h
 A[j,j-1]=-l; A[j,j+1]=-u; A[j,j]=l+u; base.append(A)
 b=k[[0,-1]]/h**2; seal=np.column_stack([b,[0,0],[0,0]]); robin=seal.copy(); robin[:,2]=r
 wrap=(k[0]+k[-1])/(2*h*h); per=np.array([[(k[0]+k[1])/(2*h*h)+max(-v[0],0)/h,wrap+max(v[0],0)/h,0],[(k[-1]+k[-2])/(2*h*h)+max(v[-1],0)/h,wrap+max(-v[-1],0)/h,0]])
 variants.append((seal,robin,per)); print(q,'sealed/Robin/periodic',variants[-1])
def pred(p):
 axes=[]
 for a,rates in zip(base,p.reshape(2,2,3)):
  a=a.copy(); n=len(a)
  for (j,near,far),(d,w,r) in zip([(0,1,n-1),(n-1,n-2,0)],rates): a[j,j],a[j,near],a[j,far]=d+w+r,-d,-w
  axes.append(csc_matrix(a))
 L=(kron(eye(ny),axes[0])+kron(axes[1],eye(nx))+V).tocsc()
 return np.vstack([splu(L+z*I).solve(B)[pos] for z in s['mu_shifts']])
def rms(x): return np.sqrt(np.mean(x*x,axis=0))
combos=[]
for x,y in product(range(3),repeat=2):
 p=np.stack([variants[0][x],variants[1][y]]).ravel(); e=pred(p)-Y; combos.append((np.mean(e**2),p)); print('stencil',x,y,rms(e))
p0=min(combos,key=lambda t:t[0])[1]
for i in [-1,0,1,2]:
 use=np.arange(3)!=i
 fit=least_squares(lambda p:(pred(p)-Y)[:,use].ravel(),p0,bounds=(-30,250),jac='3-point',ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=220)
 print('omit',i,'pulse RMS',rms(pred(fit.x)-Y),'rates',fit.x.reshape(2,2,3).round(6),'jac sv',np.linalg.svd(fit.jac,compute_uv=False).round(6))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wall_fit.py
```
- **Container Exit Code**: `0` (`1089 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
]]))
y sealed/Robin/periodic (array([[16.80278482,  0.        ,  0.        ],
       [ 9.54278482,  0.        ,  0.        ]]), array([[16.80278482,  0.        ,  6.98812334],
       [ 9.54278482,  0.        ,  6.98812334]]), array([[16.72926456, 13.17278482,  0.        ],
       [ 9.61630507, 14.41867099,  0.        ]]))
stencil 0 0 [0.04614634 0.03865226 0.04058586]
stencil 0 1 [0.12386841 0.11167642 0.11534693]
stencil 0 2 [0.05020399 0.04117163 0.04835535]
stencil 1 0 [0.12123379 0.10943283 0.1124235 ]
stencil 1 1 [0.15500379 0.14028639 0.13738573]
stencil 1 2 [0.1301683  0.11550661 0.11517828]
stencil 2 0 [3.33220395e-06 3.35175090e-06 3.22778367e-06]
stencil 2 1 [0.1279082  0.11495755 0.10681966]
stencil 2 2 [0.04404511 0.03171357 0.02106219]
omit -1 pulse RMS [3.07060283e-06 3.08771237e-06 2.95550483e-06] rates [[[ 1.4386199e+01  1.5790926e+01 -8.8000000e-05]
  [ 1.4387391e+01  1.5010516e+01  9.8000000e-05]]

 [[ 1.6803775e+01  3.8000000e-05  4.5000000e-05]
  [ 9.5426830e+00 -4.0000000e-06  2.3000000e-05]]] jac sv [1.530983e+00 4.114100e-01 1.554710e-01 1.115600e-01 6.122400e-02
 4.155700e-02 3.350800e-02 1.507200e-02 4.614000e-03 2.598000e-03
 1.733000e-03 1.033000e-03]
omit 0 pulse RMS [4.85190351e-06 3.07372696e-06 2.93328111e-06] rates [[[ 1.4390285e+01  1.5795741e+01  4.0000000e-05]
  [ 1.4382871e+01  1.5006656e+01 -3.4000000e-05]]

 [[ 1.6804188e+01  2.9000000e-05  4.9000000e-05]
  [ 9.5431020e+00  8.5000000e-05  4.0000000e-06]]] jac sv [1.15821e+00 3.17268e-01 1.14715e-01 7.98740e-02 3.72540e-02 3.01960e-02
 2.59160e-02 5.75000e-03 3.57500e-03 1.65000e-03 1.35100e-03 4.16000e-04]
omit 1 pulse RMS [3.06552634e-06 4.69367908e-06 2.93441215e-06] rates [[[ 1.4391552e+01  1.5796235e+01  1.6000000e-05]
  [ 1.4383473e+01  1.5007116e+01 -7.0000000e-06]]

 [[ 1.6804071e+01  2.1000000e-05  4.2000000e-05]
  [ 9.5428800e+00  3.5000000e-05  1.2000000e-05]]] jac sv [1.262562e+00 3.395750e-01 1.392240e-01 9.802800e-02 5.668900e-02
 3.201500e-02 2.492300e-02 7.622000e-03 3.717000e-03 1.962000e-03
 1.441000e-03 3.680000e-04]
omit 2 pulse RMS [3.06785298e-06 3.08052401e-06 5.26559526e-06] rates [[[ 1.4385819e+01  1.5789305e+01 -1.1000000e-05]
  [ 1.4385794e+01  1.5008783e+01  2.4000000e-05]]

 [[ 1.6803104e+01  7.1000000e-05  5.4000000e-05]
  [ 9.5428170e+00  4.0000000e-05  1.1000000e-05]]] jac sv [1.326661e+00 3.505650e-01 1.226190e-01 6.494500e-02 4.038400e-02
 3.237100e-02 1.588500e-02 6.712000e-03 1.759000e-03 1.061000e-03
 7.690000e-04 2.210000e-04]
```

### Turn 6 — `execute_in_remote_docker` (`41424 ms` LLM)
- **Rationale**: Implement the exact mixed-axis stencil and keep the staged upstream composition unchanged. Run the public validator to catch interface and shape errors before independently checking the field and certificate.
- **Alternatives Weighed**: `Use continuously fitted wall rates` (rejected: The exact discrete stencil already matches all pulses at the calibration residual scale; fitted rates add parameters and produce larger held-out errors.)

```bash
cd /holdout_task && cat > downstream_solver.py <<'PY'
"""Source inversion in the calibrated periodic-x, sealed-y chamber."""

import numpy as np
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, periodic):
    n = len(kappa)
    i = np.arange(n)
    left, right = (i - 1) % n, (i + 1) % n
    face = (kappa + kappa[right]) / (2 * h**2)
    lo = face[left] + np.maximum(velocity, 0) / h
    hi = face + np.maximum(-velocity, 0) / h
    if not periodic:
        lo[0], hi[0] = 0, kappa[0] / h**2
        lo[-1], hi[-1] = kappa[-1] / h**2, 0
    return coo_matrix((np.r_[lo + hi, -lo, -hi],
                       (np.tile(i, 3), np.r_[i, left, right])), shape=(n, n)).tocsc()


def _forward(s):
    nx, ny = s['nx'], s['ny']
    ax, ay = (_axis(s['kappa_' + key], s['vel_' + key], s['L' + key] / (n - 1), key == 'x')
              for key, n in zip('xy', (nx, ny)))
    L = (kron(eye(ny), ax) + kron(ay, eye(nx))
         + diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    identity = eye(nx * ny, format='csc')
    rhs = s['source_modes'].transpose(0, 2, 1).reshape(-1, nx * ny).T
    ij = s['sensor_ij']
    positions = ij[:, 0] + nx * ij[:, 1]
    blocks = []
    for mu in s['mu_shifts']:
        u = splu(L + mu * identity).solve(rhs)
        blocks.append(u[positions])
    return np.vstack(blocks), u.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _box_simplex(H, c, cap, total):
    n = len(c)
    if total == 0 or total == n * cap:
        w = np.full(n, total / n)
        g = H @ w + c
        return w, float(g.min() if total == 0 else g.max())
    w = np.full(n, total / n)
    state = np.zeros(n, dtype=int)  # -1 lower, 0 free, +1 upper
    for _ in range(20 * n + 20):
        free = state == 0
        g = H @ w + c
        if free.any():
            block = H[np.ix_(free, free)]
            ones = np.ones(free.sum())
            q, r = np.linalg.solve(block, np.column_stack((g[free], ones))).T
            nu = (q.sum() + total - w.sum()) / r.sum()
            p = nu * r - q
            wf = w[free]
            trial = wf + p
            hit = np.flatnonzero((trial < 0) | (trial > cap))
            if hit.size:
                targets = np.where(p[hit] < 0, 0, cap)
                steps = (targets - wf[hit]) / p[hit]
                k = np.argmin(steps)
                w[free] = np.clip(wf + steps[k] * p, 0, cap)
                j = np.flatnonzero(free)[hit[k]]
                w[j] = targets[k]
                state[j] = -1 if targets[k] == 0 else 1
                continue
            w[free] = trial
            g = H @ w + c
        else:
            nu = (g[state == -1].min() + g[state == 1].max()) / 2
        violation = state * (g - nu)
        j = np.argmax(violation)
        if violation[j] <= 1e-12:
            return w, float(nu)
        state[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    s = cal_item
    S, U = _forward(s)
    y = s['y_obs_raw']
    Z = s['Z_nuisance']
    P = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    S = S - Z @ (P @ S)
    y = y - Z @ (P @ y)
    ridge, cap, budget = s['source_ridge'], s['u_box'], s['total_budget']
    w, nu = _box_simplex(S.T @ S + ridge * np.eye(S.shape[1]), -S.T @ y, cap, budget)
    r = S @ w - y
    g = S.T @ r + ridge * w
    reduced = g - nu
    J = float(0.5 * (r @ r + ridge * (w @ w)))
    certificate = dict(weights=w.tolist(), nu_sum=nu,
                       lam_lower=np.maximum(reduced, 0).tolist(),
                       lam_upper=np.maximum(-reduced, 0).tolist(),
                       primal_objective=J,
                       dual_lower_bound=float(J - g @ w + nu * budget + cap * np.minimum(reduced, 0).sum()))
    return dict(weights=w, terminal_field=np.tensordot(w, U, axes=(0, 0)), kkt_certificate=certificate)
PY
python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`403 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9f7ef0e081f4a85b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9f7ef0e0 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`48718 ms` LLM)
- **Rationale**: Use archived axis formulas with a dense solve as an independent reference, select the stencil without each held-out pulse, and check the field and certificate from the composed entrypoint. Exact-cap budgets test whether the active set handles bound-only optima.
- **Alternatives Weighed**: `Submit on the schema pass alone` (rejected: The schema pass does not detect wrong axis ordering, incorrect terminal fields, or a numerically invalid KKT certificate.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys, pickle, inspect, json, importlib.util
from itertools import product
import numpy as np
sys.path.insert(0, '/holdout_task')
import batch_solver as batch
import downstream_solver as dn
from _verified_upstream import calibrate_upstream

def calibrate(study):
    return calibrate_upstream(study, batch._PAY) if len(inspect.signature(calibrate_upstream).parameters) >= 2 else calibrate_upstream(study)

def archived_axis(name, s, key):
    path = '/tmp/hegel_continual_archive/dev_beta_m99_00_' + name + '_solver.py'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    h = s['L'+key] / (s['n'+key]-1)
    return mod._axis(s['kappa_'+key], s['vel_'+key], h).toarray()

worst_pulse = worst_slice = worst_field = worst_gap = worst_kkt = 0.0
with open('public_batch.pkl','rb') as f:
    items = pickle.load(f)
for item in items:
    raw = item['study']
    s = calibrate(dict(raw, reference_tracer_pulses=item.get('campaign_diagnostics'))) if item.get('campaign_diagnostics') else calibrate(raw)
    nx, ny = s['nx'], s['ny']
    Z = s['Z_nuisance']
    P = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    modes = np.column_stack([mode.ravel(order='F') for mode in s['source_modes']])
    ii, jj = s['sensor_ij'].T
    pos = ii + nx*jj
    eye = np.eye(nx*ny)
    reaction = np.diag(s['V_reaction_2d'].ravel(order='F'))
    axes = {key: [archived_axis(name, s, key) for name in ('batch_01','batch_02')] for key in 'xy'}
    for key in 'xy':
        robin = axes[key][0].copy()
        robin[0,0] += s['robin_'+key] * (s['n'+key]-1) / s['L'+key]
        robin[-1,-1] += s['robin_'+key] * (s['n'+key]-1) / s['L'+key]
        axes[key].append(robin)
    def forward(x,y):
        L = np.kron(np.eye(ny),axes['x'][x]) + np.kron(axes['y'][y],np.eye(nx)) + reaction
        U = [np.linalg.solve(L+mu*eye,modes) for mu in s['mu_shifts']]
        return np.vstack([u[pos] for u in U]), U[-1]
    S, Ulast = forward(1,0)
    S_actual, _ = dn._forward(s)
    np.testing.assert_allclose(S_actual,S,rtol=0,atol=1e-12)
    if item.get('campaign_diagnostics'):
        pulses = s['reference_tracer_pulses']
        W = np.column_stack([p['w_pulse'] for p in pulses])
        Y = np.column_stack([p['y_pulse_raw'] for p in pulses])
        errors = {xy: forward(*xy)[0] @ W - Y for xy in product(range(3), repeat=2)}
        for j in range(W.shape[1]):
            train = np.arange(W.shape[1]) != j
            selected = min(errors, key=lambda xy: np.mean(errors[xy][:,train]**2))
            assert selected == (1,0), selected
            err = errors[selected][:,j]
            worst_pulse = max(worst_pulse, np.sqrt(np.mean(err**2)))
            worst_slice = max(worst_slice, np.sqrt(np.mean(err.reshape(len(s['mu_shifts']),-1)**2,axis=1)).max())
    result = batch.solve_batch(raw)
    w, cert = result['weights'], result['kkt_certificate']
    lo, hi = np.array(cert['lam_lower']), np.array(cert['lam_upper'])
    cap, budget = s['u_box'], s['total_budget']
    field = (Ulast @ w).reshape((nx,ny), order='F')
    worst_field = max(worst_field, np.max(abs(result['terminal_field'] - field)))
    np.testing.assert_allclose(cert['weights'],w,rtol=0,atol=0)
    Sd = S-Z@(P@S)
    residual = Sd@w - (s['y_obs_raw']-Z@(P@s['y_obs_raw']))
    g = Sd.T@residual + s['source_ridge']*w
    J = (residual@residual + s['source_ridge']*(w@w))/2
    nu = cert['nu_sum']
    dual = J-g@w+nu*budget+cap*np.minimum(g-nu,0).sum()
    np.testing.assert_allclose([cert['primal_objective'],cert['dual_lower_bound']],[J,dual],rtol=0,atol=1e-12)
    worst_gap = max(worst_gap, abs(J-dual))
    worst_kkt = max(worst_kkt, abs(w.sum()-budget), np.max(abs(g-nu-lo+hi)), np.max(abs(w*lo)), np.max(abs((cap-w)*hi)))
    assert w.min() >= -1e-12 and w.max() <= cap+1e-12 and lo.min() >= 0 and hi.min() >= 0
assert worst_slice < 1e-5 and worst_field < 1e-12 and worst_gap < 1e-10 and worst_kkt < 1e-10
print(f'public cases={len(items)}: LOPO worst-pulse RMS={worst_pulse:.4g}, worst pulse/shift RMS={worst_slice:.4g}, terminal max error={worst_field:.4g}, worst dual gap={worst_gap:.4g}, worst KKT error={worst_kkt:.4g}')

rng = np.random.default_rng(52)
worst = 0.0
for n in (3,8,16):
    for _ in range(50):
        A = rng.normal(size=(n,n))
        H = A.T@A+0.04*np.eye(n)
        c = rng.normal(size=n)*rng.choice([0.1,1,100])
        cap = 0.42
        for budget in (0, cap, cap*(n-1), n*cap, rng.uniform(0,n*cap)):
            w,nu = dn._box_simplex(H,c,cap,budget)
            g = H@w+c-nu
            gap = np.maximum(g,0)@w + np.maximum(-g,0)@(cap-w)
            worst = max(worst,gap,abs(w.sum()-budget),-w.min(),w.max()-cap)
assert worst < 1e-9, worst
print(f'750 QPs across interior, exact-cap budgets, and endpoints: worst feasibility/complementarity error={worst:.4g}')
PY
python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`818 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public cases=1: LOPO worst-pulse RMS=3.352e-06, worst pulse/shift RMS=3.419e-06, terminal max error=1.943e-16, worst dual gap=3.178e-15, worst KKT error=2.543e-15
750 QPs across interior, exact-cap budgets, and endpoints: worst feasibility/complementarity error=1.871e-14
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9f7ef0e081f4a85b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9f7ef0e0 (directory).
```

### Turn 8 — `submit_completed_task` (`12562 ms` LLM)
- **Rationale**: Both deliverables passed the schema and isolated numerical checks on the verified bundle. An independent dense solve matched the terminal field within 1.943e-16, and 750 box-simplex stress cases had worst feasibility or complementarity error of 1.871e-14.
- **Alternatives Weighed**: `Fit continuous boundary rates` (rejected: The exact mixed stencil predicted held-out pulses better without fitted wall parameters.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:408', '/holdout_task/downstream_solver.py': 'FILE:3867'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented the periodic-x, sealed-y downstream solver and verified it through the existing batch entrypoint. The public schema passed. Worst held-out pulse/shift RMS was 3.419e-6, and the worst public KKT residual was 2.543e-15.`

### Turn 9 — `execute_in_remote_docker` (`40193 ms` LLM)
- **Rationale**: Leave out each real sensor bank and choose among the nine exact wall combinations with and without nuisance profiling fitted only on the retained banks. A reversal would reopen boundary selection; excess held-out error would direct inspection to calibration or indexing. The isolated run leaves both verified artifacts unchanged.
- **Alternatives Weighed**: `Audit relay modes, drift operators, or classification accuracy` (rejected: This is a steady linear transport inverse problem with continuous measurements; none of those mechanisms or metrics is part of its contract.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys, inspect, pickle
from itertools import product
import numpy as np
sys.path.insert(0, '/holdout_task')
import batch_solver as batch
import downstream_solver as downstream
from _verified_upstream import calibrate_upstream

def axis(s, key, wall):
    k, v = s['kappa_'+key], s['vel_'+key]
    n = len(k)
    h = s['L'+key]/(n-1)
    a = np.zeros((n,n))
    for i in range(n):
        if i in (0,n-1) and wall != 'periodic':
            j = 1 if i == 0 else n-2
            a[i,j] = -k[i]/h**2
            a[i,i] = -a[i,j] + (s['robin_'+key]/h if wall == 'robin' else 0)
        else:
            left, right = (i-1)%n, (i+1)%n
            a[i,left] = -(k[i]+k[left])/(2*h*h)-max(v[i],0)/h
            a[i,right] = -(k[i]+k[right])/(2*h*h)-max(-v[i],0)/h
            a[i,i] = -a[i,left]-a[i,right]
    return a

worst_raw = worst_profiled = worst_gap = worst_stationarity = 0.
folds = 0
with open('public_batch.pkl','rb') as f:
    items = pickle.load(f)
for item in items:
    raw = item['study']
    pulses = item.get('campaign_diagnostics')
    if not pulses:
        continue
    inp = dict(raw, reference_tracer_pulses=pulses)
    s = calibrate_upstream(inp,batch._PAY) if len(inspect.signature(calibrate_upstream).parameters) >= 2 else calibrate_upstream(inp)
    nx, ny = s['nx'], s['ny']
    Z, banks = s['Z_nuisance'],np.asarray(s['row_bank_id'])
    W = np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']])
    Y = np.column_stack([p['y_pulse_raw'] for p in s['reference_tracer_pulses']])
    B = np.column_stack([m.ravel(order='F') for m in s['source_modes']])
    pos = s['sensor_ij'] @ np.array([1,nx])
    E = {}
    for x,y in product(('sealed','periodic','robin'),repeat=2):
        L = np.kron(np.eye(ny),axis(s,'x',x)) + np.kron(axis(s,'y',y),np.eye(nx)) + np.diag(s['V_reaction_2d'].ravel(order='F'))
        S = np.vstack([np.linalg.solve(L+mu*np.eye(nx*ny),B)[pos] for mu in s['mu_shifts']])
        E[x,y] = S @ W - Y
        if (x,y) == ('periodic','sealed'):
            np.testing.assert_allclose(S,downstream._forward(s)[0],rtol=0,atol=1e-12)
    reg = s['nuisance_ridge']*np.eye(Z.shape[1])
    for bank in np.unique(banks):
        train = banks != bank
        Zt = Z[train]
        P = np.linalg.solve(Zt.T@Zt+reg,Zt.T)
        nuisance = {key: P@e[train] for key,e in E.items()}
        raw_key = min(E,key=lambda key: np.mean(E[key][train]**2))
        prof_key = min(E,key=lambda key: np.mean((E[key][train]-Zt@nuisance[key])**2))
        assert raw_key == prof_key == ('periodic','sealed'), (bank,raw_key,prof_key)
        e = E[raw_key][~train]
        rms = lambda a: np.sqrt(np.mean(a**2,axis=0)).max()
        raw_error, prof_error = rms(e), rms(e-Z[~train]@nuisance[prof_key])
        worst_raw, worst_profiled = max(worst_raw,raw_error),max(worst_profiled,prof_error)
        print(f'held-out bank {bank}: {raw_key}, max pulse RMS raw={raw_error:.4g}, train-profiled={prof_error:.4g}')
        folds += 1
    out = batch.solve_batch(raw)
    cert = out['kkt_certificate']
    w = out['weights']
    F,U = downstream._forward(s)
    P = np.linalg.solve(Z.T@Z+reg,Z.T)
    F -= Z@(P@F)
    y = s['y_obs_raw']-Z@(P@s['y_obs_raw'])
    g = F.T@(F@w-y)+s['source_ridge']*w
    lo,hi = np.asarray(cert['lam_lower']), np.asarray(cert['lam_upper'])
    worst_stationarity = max(worst_stationarity,np.max(np.abs(g-cert['nu_sum']-lo+hi)))
    worst_gap = max(worst_gap,abs(cert['primal_objective']-cert['dual_lower_bound']))
assert folds and max(worst_raw,worst_profiled) < 1e-5
assert max(worst_gap,worst_stationarity) < 1e-10
print(f'{folds} bank folds: worst raw RMS={worst_raw:.4g}, profiled RMS={worst_profiled:.4g}, certificate gap={worst_gap:.4g}, stationarity={worst_stationarity:.4g}')
PY
```
- **Container Exit Code**: `0` (`440 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held-out bank 0: ('periodic', 'sealed'), max pulse RMS raw=4.26e-06, train-profiled=4.286e-06
held-out bank 1: ('periodic', 'sealed'), max pulse RMS raw=2.804e-06, train-profiled=2.984e-06
held-out bank 2: ('periodic', 'sealed'), max pulse RMS raw=2.639e-06, train-profiled=2.614e-06
held-out bank 3: ('periodic', 'sealed'), max pulse RMS raw=3.536e-06, train-profiled=3.633e-06
4 bank folds: worst raw RMS=4.26e-06, profiled RMS=4.286e-06, certificate gap=6.436e-16, stationarity=0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9f7ef0e081f4a85b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9f7ef0e0 (directory).
```

### Turn 10 — `submit_completed_task` (`13222 ms` LLM)
- **Rationale**: Both deliverables are saved in the verified bundle. The isolated bank holdouts selected the same stencil with and without nuisance profiling. An independent dense solve matched the terminal field within 1.943e-16, and 750 box-simplex stress cases had a worst feasibility or complementarity error of 1.871e-14.
- **Alternatives Weighed**: `Fit continuous boundary rates` (rejected: The exact stencil matched the public pulses at the calibration residual scale and predicted held-out pulses better than the fitted rates.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:408', '/holdout_task/downstream_solver.py': 'FILE:3867'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented the periodic-x, sealed-y downstream solver and verified it through the existing batch entrypoint. The public schema passed. Worst held-out RMS was 3.419e-6 by pulse and shift and 4.286e-6 by sensor bank after nuisance profiling. The worst public KKT residual was 2.543e-15.`

### Turn 11 — `execute_in_remote_docker` (`44248 ms` LLM)
- **Rationale**: Select the wall stencil after excluding both a pulse and a reaction shift, and compare the resulting test residual with and without nuisance profiling. A ranking reversal would reopen boundary selection; elevated test error would point to shift-dependent misspecification. Check the implemented wall coefficients and zero row sums against the independent stencil in the same isolated run.
- **Alternatives Weighed**: `Repeat the sensor-bank holdout` (rejected: The prior run already tested bank dependence; it did not test whether wall selection relies on a particular reaction shift and pulse together.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import itertools, inspect, json, pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import batch_solver as batch
import downstream_solver as dn
from _verified_upstream import calibrate_upstream

winner = ('periodic', 'sealed')
worst = np.zeros(2)
margin = np.full(2, np.inf)
folds = 0
with open('public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    if not item.get('campaign_diagnostics'):
        continue
    raw = dict(item['study'], reference_tracer_pulses=item['campaign_diagnostics'])
    s = calibrate_upstream(raw, batch._PAY) if len(inspect.signature(calibrate_upstream).parameters) >= 2 else calibrate_upstream(raw)
    nx, ny = s['nx'], s['ny']
    shifts = s['mu_shifts']
    pulses = s['reference_tracer_pulses']
    W = np.column_stack([p['w_pulse'] for p in pulses])
    Y = np.column_stack([p['y_pulse_raw'] for p in pulses])
    B = np.column_stack([m.ravel(order='F') for m in s['source_modes']]) @ W
    ij = s['sensor_ij']; pos = ij[:, 0] + nx * ij[:, 1]
    rows_shift = np.repeat(np.arange(len(shifts)), len(pos))
    Z = s['Z_nuisance']
    reg = s['nuisance_ridge'] * np.eye(Z.shape[1])
    axes = {}
    for key, n in zip('xy', (nx, ny)):
        k = s['kappa_' + key]; v = s['vel_' + key]; h = s['L' + key] / (n - 1)
        for wall in ('periodic', 'sealed', 'robin'):
            A = np.zeros((n, n))
            for i in range(n):
                if i in (0, n - 1) and wall != 'periodic':
                    j = 1 if i == 0 else n - 2
                    A[i, j] = -k[i] / h**2
                    A[i, i] = -A[i, j] + (s['robin_' + key] / h if wall == 'robin' else 0)
                else:
                    l, r = (i - 1) % n, (i + 1) % n
                    A[i, l] = -(k[i] + k[l]) / (2 * h*h) - max(v[i], 0) / h
                    A[i, r] = -(k[i] + k[r]) / (2 * h*h) - max(-v[i], 0) / h
                    A[i, i] = -A[i, l] - A[i, r]
            axes[key, wall] = A
        np.testing.assert_allclose(dn._axis(k, v, h, key == 'x').toarray(), axes[key, winner['xy'.index(key)]], atol=1e-12, rtol=0)
        np.testing.assert_allclose(axes[key, winner['xy'.index(key)]].sum(axis=1), 0, atol=1e-12, rtol=0)
    identity = np.eye(nx*ny)
    reaction = np.diag(s['V_reaction_2d'].ravel(order='F'))
    E = {}
    for x, y in itertools.product(('periodic', 'sealed', 'robin'), repeat=2):
        L = np.kron(np.eye(ny), axes['x', x]) + np.kron(axes['y', y], np.eye(nx)) + reaction
        E[x, y] = np.vstack([np.linalg.solve(L + mu * identity, B)[pos] for mu in shifts]) - Y
    for j, k in itertools.product(range(len(pulses)), range(len(shifts))):
        train = rows_shift != k
        cols = np.arange(len(pulses)) != j
        Zt = Z[train]
        P = np.linalg.solve(Zt.T @ Zt + reg, Zt.T)
        scores = {wall: np.array([np.sqrt(np.mean(e**2)), np.sqrt(np.mean((e - Zt @ (P @ e))**2))]) for wall, err in E.items() for e in [err[train][:, cols]]}
        assert all(min(scores, key=lambda wall: scores[wall][q]) == winner for q in (0, 1)), (j, k, scores)
        margin = np.minimum(margin, np.min([score for wall, score in scores.items() if wall != winner], axis=0) / scores[winner])
        # Fit the held-out pulse's nuisance only on its retained shifts, never on its test shift.
        e = E[winner][:, j]
        test = e[~train]
        prof = test - Z[~train] @ (P @ e[train])
        worst = np.maximum(worst, [np.sqrt(np.mean(test**2)), np.sqrt(np.mean(prof**2))])
        folds += 1
assert folds and max(worst) < 1e-5, (folds, worst)
print(json.dumps(dict(nogood_id='ng_t05', verified_stencil=list(winner), joint_pulse_shift_folds=folds, worst_test_rms_raw=worst[0], worst_test_rms_profiled=worst[1], minimum_runner_up_training_rms_ratio=margin.tolist())))
PY
python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`609 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"nogood_id": "ng_t05", "verified_stencil": ["periodic", "sealed"], "joint_pulse_shift_folds": 9, "worst_test_rms_raw": 3.4190375109689944e-06, "worst_test_rms_profiled": 3.430624292937085e-06, "minimum_runner_up_training_rms_ratio": [5358.763538493113, 5179.410127224392]}
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9f7ef0e081f4a85b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9f7ef0e0 (directory).
```

### Turn 12 — `submit_completed_task` (`13468 ms` LLM)
- **Rationale**: The unchanged artifact bundle passed isolated execution, the public schema, independent terminal-field and KKT checks, and 750 box-simplex stress cases. The joint holdout also confirmed that nuisance profiling did not change the wall selection.
- **Alternatives Weighed**: `Fit continuous boundary rates` (rejected: The exact stencil predicted held-out pulses better, and after nuisance profiling the nearest discrete alternative had at least 5,179 times its training RMS in every joint pulse-and-shift fold.)
- **Submitted Final Answer / Summary**: `Implemented and verified the periodic-x, sealed-y downstream solver through the existing calibrated batch entrypoint. The public schema passed. Worst held-out RMS was 3.431e-6 across pulse-and-shift folds and 4.286e-6 across sensor-bank folds after nuisance profiling. The terminal field matched an independent dense solve within 1.943e-16.`

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
