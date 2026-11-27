# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_beta_m99_00_batch_02__blind_continual_campaign__downstream_module`

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
| **Total Tokens (Prompt / Cached / Out)** | `397,727` (`385,273` / `136,960` / `12,454`) |
| **Total Episode Wall Time (s)** | `364.12s` |

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

### Turn 1 — `execute_in_remote_docker` (`15806 ms` LLM)
- **Rationale**: Check whether the wrapper passes the expected calibrated keys and whether the apparatus log identifies a boundary change. Either finding determines which component to repair before testing numerical accuracy.
- **Alternatives Weighed**: `Reuse the prior solver unchanged` (rejected: The public gate flagged the downstream solve; inspect the current apparatus record and calibrated interface before choosing what to reuse.)

```bash
cd /holdout_task && ls -l && cat batch_solver.py _verified_upstream.py validate_public_schema.py && ls -l /tmp/hegel_continual_archive && python3 - <<'PY'
import pickle, json
from pathlib import Path
from _verified_upstream import calibrate_upstream
p=pickle.load(open('public_batch.pkl','rb')); print('items:',len(p)); print('raw keys:',p[0].keys()); print('study keys:',p[0]['study'].keys())
a=calibrate_upstream(p[0]['study'],json.load(open('_upstream_param_payload.json'))); print('calibrated keys:',a.keys());
for k,v in a.items(): print(k,type(v).__name__,getattr(v,'shape',None),list(v) if isinstance(v,dict) else v if isinstance(v,(float,int,str)) else '')
for n in ('archive_manifest.json','archive_apparatus_log.json'):
 q=Path('/tmp/hegel_continual_archive')/n
 print(n,q.stat().st_size); print(q.read_text()[-16000:])
PY
```
- **Container Exit Code**: `0` (`308 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-0.5055
      ],
      [
        -0.40588,
        -0.88181,
        2.56281,
        -0.85036,
        -0.65644,
        -0.33575,
        0.248,
        -0.76123
      ],
      [
        -0.19308,
        -0.70015,
        2.66493,
        -0.76657,
        -0.32584,
        -0.41808,
        0.50209,
        -0.88357
      ],
      [
        -0.02033,
        -0.38805,
        2.598,
        -0.59931,
        0.13008,
        -0.39351,
        0.60862,
        -0.71704
      ],
      [
        0.21863,
        0.07575,
        2.5438,
        -0.43979,
        0.52955,
        -0.31081,
        0.7175,
        -0.41132
      ],
      [
        0.46436,
        0.50754,
        2.50193,
        -0.2414,
        0.84366,
        -0.30069,
        0.85626,
        0.0732
      ],
      [
        0.58756,
        0.74369,
        2.51284,
        -0.00499,
        0.8527,
        -0.284,
        0.83867,
        0.55779
      ],
      [
        0.73422,
        0.84887,
        2.38866,
        0.17944,
        0.67885,
        -0.34593,
        0.8125,
        0.70441
      ],
      [
        0.83402,
        0.72115,
        2.47508,
        0.40493,
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
]
```

### Turn 2 — `execute_in_remote_docker` (`19954 ms` LLM)
- **Rationale**: Determine whether the current study has direct tracer data or must inherit an apparatus setting from the archive. Check how the public validator loads the wrapper before changing the downstream solver.
- **Alternatives Weighed**: `Dump the full apparatus log again` (rejected: Large calibration arrays obscured the chamber settings in the first output; summarize the records and locate boundary code instead.)

```bash
cd /holdout_task && cat validate_public_schema.py _upstream_param_payload.json && ls -l /tmp/hegel_continual_archive && python3 - <<'PY'
import json,pickle
from pathlib import Path
p=pickle.load(open('public_batch.pkl','rb')); print('items',len(p)); print('diagnostics counts',[None if x.get('campaign_diagnostics') is None else len(x['campaign_diagnostics']) for x in p]);
for k,v in p[0]['study'].items(): print(k,type(v).__name__,getattr(v,'shape',None),v if isinstance(v,(int,float,str)) else list(v) if isinstance(v,dict) else '')
root=Path('/tmp/hegel_continual_archive'); a=json.loads((root/'archive_apparatus_log.json').read_text()); print('apparatus type',type(a).__name__)
if isinstance(a,dict): print('apparatus keys',list(a)); a=next((v for v in a.values() if isinstance(v,list) and v and isinstance(v[0],dict)),[a]);
for entry in a:
 print('entry', {k:(f'{type(v).__name__} len={len(v)}' if isinstance(v,(list,dict)) else str(v)[:500]) for k,v in entry.items()});
 for k,v in entry.items():
  if isinstance(v,dict): print(k,{q:r for q,r in v.items() if isinstance(r,(int,float,str,bool,type(None)))},list(v))
PY
for p in /tmp/hegel_continual_archive/*solver.py; do echo "--- $p"; wc -l "$p"; grep -nEi 'wall|robin|boundary|def |apparatus' "$p" | head -85; done
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling'
```
- **Container Exit Code**: `0` (`374 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2
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
apparatus type list
entry {'completed_campaign_index': '0', 'completed_campaign_short_name': 'facility_init', 'facility_telemetry_trace': 'list len=24'}
entry {'completed_campaign_index': '1', 'completed_campaign_short_name': 'dev_beta_m99_00_s00', 'facility_telemetry_trace': 'list len=24'}
entry {'completed_campaign_index': '2', 'completed_campaign_short_name': 'dev_beta_m99_00_batch_01', 'facility_telemetry_trace': 'list len=24'}
--- /tmp/hegel_continual_archive/dev_beta_m99_00_batch_01_solver.py
119 /tmp/hegel_continual_archive/dev_beta_m99_00_batch_01_solver.py
1:"""Reconstruct the sealed-wall chamber and solve its box-simplex inverse problem."""
9:def _axis(kappa, vel, h):
15:    # Tracer pulses identify zero wall leakage and unchanged wall-face diffusion.
20:def _forward(study):
39:def _calibrate(study):
48:def _box_simplex(H, c, cap, budget):
96:def solve_batch(study: dict) -> dict:
--- /tmp/hegel_continual_archive/dev_beta_m99_00_s00_solver.py
119 /tmp/hegel_continual_archive/dev_beta_m99_00_s00_solver.py
1:"""Reconstruct the open-wall chamber and solve its box-simplex inverse problem."""
9:def _axis(kappa, vel, h, robin):
15:    diagonal = np.r_[-upper[0] + robin / h,
16:                     -lower[:-1] - upper[1:], -lower[-1] + robin / h]
20:def _forward(study):
23:                    study['L' + key] / (n - 1), study['robin_' + key])
39:def _calibrate(study):
48:def _box_simplex(H, c, cap, budget):
96:def solve_batch(study: dict) -> dict:
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 3 — `execute_in_remote_docker` (`20282 ms` LLM)
- **Rationale**: Inspect the archived forward model and pulse dimensions before comparing boundary predictions. Check whether the staged upstream calibration already removes sensor mixing so calibration error is not mistaken for a boundary change.
- **Alternatives Weighed**: `Infer the boundary from archive telemetry alone` (rejected: Three current-campaign tracer pulses are available and can directly distinguish transport operators.)

```bash
cd /holdout_task && cat /tmp/hegel_continual_archive/dev_beta_m99_00_batch_01_solver.py && cat /tmp/hegel_continual_archive/archive_manifest.json && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0; cd /holdout_task && python3 - <<'PY'
import pickle, json, numpy as np
from _verified_upstream import calibrate_upstream
item=pickle.load(open('public_batch.pkl','rb'))[0]; s=item['study']; pay=json.load(open('_upstream_param_payload.json')); cal=calibrate_upstream(s,pay)
for k,v in s['sensor_calibration_sweeps'].items():
 print('sweep',k, {a:np.shape(b) for a,b in v.items()} if isinstance(v,dict) else np.shape(v))
for j,p in enumerate(item['campaign_diagnostics']):
 print('pulse',j, {k:(np.shape(v),np.min(v),np.max(v)) for k,v in p.items()})
r=s['sensor_calibration_sweeps']['ref_rosette']; y=cal['sensor_calibration_sweeps']['raw_rosette']; m=np.all(abs(y)<=8.5,axis=1); A=np.c_[np.ones(m.sum()),r[m]]; B=np.linalg.lstsq(A,y[m],rcond=None)[0]; print('calibrated ros. fit',B.round(6),'max error',abs(A@B-y[m]).max(),'inliers',m.sum())
print('row bank pattern',s['row_bank_id'][:32]); print('reaction and shifts',s['V_reaction_2d'].min(),s['V_reaction_2d'].max(),s['mu_shifts'])
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.array(e['facility_telemetry_trace']); print('log',e['completed_campaign_short_name'],'median',np.median(a,axis=0).round(4))
PY
```
- **Container Exit Code**: `0` (`343 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
sweep ref_rosette (80, 4)
sweep raw_rosette (80, 4)
sweep single_bank_sweeps {'bank_id': (24,), 'ref_signal': (24,), 'raw_readout': (24,)}
pulse 0 {'w_pulse': ((8,), 0.0, 1.0), 'y_pulse_raw': ((96,), -0.03741257474299828, 0.52857979051538)}
pulse 1 {'w_pulse': ((8,), 0.0, 1.0), 'y_pulse_raw': ((96,), -0.036135476260359144, 0.5130927003635409)}
pulse 2 {'w_pulse': ((8,), 0.0, 1.0), 'y_pulse_raw': ((96,), -0.03519802567241349, 0.4980733740654569)}
calibrated ros. fit [[ 0.  0.  0.  0.]
 [ 1.  0. -0.  0.]
 [ 0.  1.  0.  0.]
 [-0.  0.  1. -0.]
 [ 0. -0.  0.  1.]] max error 1.7763568394002505e-15 inliers 80
row bank pattern [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3]
reaction and shifts 0.25 0.42738476356834465 [0.15 0.55 1.1 ]
log facility_init median [-8.700e-03  9.500e-03 -3.865e-01  1.760e-02 -2.400e-03 -3.312e-01
  1.430e-02 -2.000e-04]
log dev_beta_m99_00_s00 median [-5.5000e-03 -1.5100e-02 -3.8960e-01  1.2400e-02  1.9000e-03  2.4551e+00
  3.3600e-02 -1.5000e-02]
log dev_beta_m99_00_batch_01 median [ 0.0071  0.0049  2.5568 -0.0059  0.0261 -0.3426 -0.0246 -0.0203]
```

### Turn 4 — `execute_in_remote_docker` (`32030 ms` LLM)
- **Rationale**: Compare calibrated pulse residuals for exact open and sealed variants, then profile edge leakage and wall-face diffusion. Comparing raw and nuisance-projected residuals tests whether nuisance structure is being mistaken for a boundary change.
- **Alternatives Weighed**: `Transfer the previous sealed-wall operator unchanged` (rejected: The current tracer pulses can distinguish uniform sealing from axis-specific or edge-specific changes.)

```bash
cd /holdout_task && cat > /tmp/boundary_probe.py <<'PY'
import json,pickle,sys
import numpy as np
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream

def forward(s, p, modes):
    nx,ny=s['nx'],s['ny']; axes=[]
    for j,(key,n) in enumerate(zip('xy',(nx,ny))):
        h=s['L'+key]/(n-1); k=s['kappa_'+key]; v=s['vel_'+key]; face=(k[:-1]+k[1:])/(2*h*h)
        lo=np.r_[-face[:-1]-np.maximum(v[1:-1],0)/h,-p[5+2*j]*k[-1]/h**2]
        up=np.r_[-p[4+2*j]*k[0]/h**2,-face[1:]-np.maximum(-v[1:-1],0)/h]
        d=np.r_[-up[0]+p[2*j]*s['robin_'+key]/h,-lo[:-1]-up[1:],-lo[-1]+p[2*j+1]*s['robin_'+key]/h]
        axes.append(diags((lo,d,up),(-1,0,1),format='csc'))
    L=(kron(eye(ny),axes[0])+kron(axes[1],eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    ij=s['sensor_ij']; pos=ij[:,0]+nx*ij[:,1]; I=eye(nx*ny,format='csc'); rhs=modes.transpose(0,2,1).reshape(-1,nx*ny).T
    return np.vstack([splu(L+mu*I).solve(rhs)[pos] for mu in s['mu_shifts']])

if __name__=='__main__':
    item=pickle.load(open('public_batch.pkl','rb'))[0]; s=item['study']; pay=json.load(open('_upstream_param_payload.json'))
    pulses=item['campaign_diagnostics']; w=np.stack([p['w_pulse'] for p in pulses]); modes=np.einsum('pm,mxy->pxy',w,s['source_modes']); print('weights of pulses',w)
    y=np.stack([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in pulses],axis=1)
    Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; print('pulse amplitude rms',np.sqrt(np.mean(y*y,axis=0)))
    def errs(p): return forward(s,p,modes)-y
    def report(name,p):
        e=errs(p); q=e-Q@(Q.T@e)
        print(name,np.round(p,6),'raw',np.sqrt(np.mean(e*e,axis=0)).round(8),'projected',np.sqrt(np.mean(q*q,axis=0)).round(8))
    for name,robs in [('open',[1]*4),('sealed',[0]*4),('sealed x',[0,0,1,1]),('sealed y',[1,1,0,0])]: report(name,np.r_[robs,[1]*4])
    for projected in (False,True):
        def residual(z, fitdiff):
            e=errs(z if fitdiff else np.r_[z,[1]*4]); return (e-Q@(Q.T@e) if projected else e).ravel()
        for fitdiff in (False,True):
            p=least_squares(residual,np.ones(8 if fitdiff else 4),bounds=(0,10),args=(fitdiff,),gtol=1e-13,ftol=1e-13,xtol=1e-13).x
            report(('projected' if projected else 'raw')+(' full' if fitdiff else ' leakage'),p if fitdiff else np.r_[p,[1]*4])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/boundary_probe.py
```
- **Container Exit Code**: `0` (`1441 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
weights of pulses [[1.  0.  0.  0.5 0.  0.  0.  0. ]
 [0.  1.  0.  0.  0.5 0.  0.  0. ]
 [0.  0.  1.  0.  0.  0.5 0.  0. ]]
pulse amplitude rms [0.19842718 0.18398696 0.18575218]
open [1 1 1 1 1 1 1 1] raw [0.12754906 0.11806465 0.11822215] projected [0.12554264 0.11612113 0.11631761]
sealed [0 0 0 0 1 1 1 1] raw [0.04509583 0.0390843  0.01099317] projected [0.04400479 0.03883009 0.0108971 ]
sealed x [0 0 1 1 1 1 1 1] raw [0.10814447 0.10230793 0.09484616] projected [0.10673872 0.1007951  0.09321397]
sealed y [1 1 0 0 1 1 1 1] raw [0.09852732 0.09262713 0.09134177] projected [0.0972266  0.09143891 0.09006328]
raw leakage [0.       0.024877 0.       0.       1.       1.       1.       1.      ] raw [0.04307345 0.04027195 0.01154172] projected [0.04216607 0.03999176 0.01141832]
raw full [ 0.        0.        0.        0.024071 10.        0.436788  1.726086
  1.157804] raw [0.03167409 0.03692971 0.02217342] projected [0.03096864 0.03645834 0.02198535]
projected leakage [0.       0.022444 0.       0.       1.       1.       1.       1.      ] raw [0.04322496 0.04014807 0.01143285] projected [0.04230133 0.03987103 0.01131446]
projected full [ 0.        0.        0.        0.020805 10.        0.433638  1.688775
  1.125382] raw [0.03176335 0.03686642 0.02216028] projected [0.03105033 0.03639338 0.02196874]
```

### Turn 5 — `execute_in_remote_docker` (`28388 ms` LLM)
- **Rationale**: Compare exact axis-wise wall stencils before adding more fitted coefficients. Inspect the original sweep fit as well: identity on the staged, rewritten sweeps does not independently rule out residual sensor mixing.
- **Alternatives Weighed**: `Adopt the fitted local-wall coefficients` (rejected: One diffusion coefficient hit the search bound while substantial pulse residuals remained. Test whether the boundary stencil is missing an opposite-edge term.)

```bash
cd /holdout_task && cp /tmp/boundary_probe.py /tmp/wrap_probe.py && python3 - <<'PY'
p='/tmp/wrap_probe.py'
s=open(p).read().split("if __name__")[0]; s=s.replace("axes.append(diags((lo,d,up),(-1,0,1),format='csc'))", """a=diags((lo,d,up),(-1,0,1),format='lil')
        if len(p)>8:
            wrap=np.array([p[8+2*j],p[9+2*j]])*k[[0,-1]]/h**2
            a[0,-1]=-wrap[0]; a[-1,0]=-wrap[1]
            a[0,0]+=wrap[0]; a[-1,-1]+=wrap[1]
        axes.append(a.tocsc())""");open(p,'w').write(s)
PY
cat >> /tmp/wrap_probe.py <<'PY'
from itertools import product
item=pickle.load(open('public_batch.pkl','rb'))[0]; s=item['study']; pay=json.load(open('_upstream_param_payload.json')); pulses=item['campaign_diagnostics']
modes=np.stack([np.tensordot(p['w_pulse'],s['source_modes'],axes=1) for p in pulses]); y=np.column_stack([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in pulses]);
print('sweep raw affine coefficients',np.linalg.lstsq(np.c_[np.ones(len(s['sensor_calibration_sweeps']['ref_rosette'])),s['sensor_calibration_sweeps']['ref_rosette']],s['sensor_calibration_sweeps']['raw_rosette'],rcond=None)[0].round(6)); print('local telemetry median',np.median(s['aux_manifold_telemetry'],axis=0))
def res(p): return forward(s,p,modes)-y
opts={'open':[1,1,1,1,0,0],'sealed':[0,0,1,1,0,0],'wrap':[0,0,1,1,1,1],'link':[0,0,0,0,1,1],'mirror':[0,0,2,2,0,0]}
rows=[]
for (a,x),(b,z) in product(opts.items(),repeat=2):
 p=np.array([x[:2],z[:2],x[2:4],z[2:4],x[4:],z[4:]]).ravel(); e=res(p); rows.append((np.mean(e*e)**.5,a,b,p))
for err,a,b,p in sorted(rows,key=lambda x:x[0])[:8]: print(a,b,err,np.mean(res(p)**2,axis=0)**.5)
for a,b in [('wrap','wrap'),('wrap','sealed'),('sealed','wrap')]:
 p=next(p for _,aa,bb,p in rows if (a,b)==(aa,bb)); fit=least_squares(lambda v:res(v).ravel(),p,bounds=(0,20),gtol=1e-13,ftol=1e-13,xtol=1e-13);print('fit',a,b,fit.x.round(7),np.mean(res(fit.x)**2,axis=0)**.5)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wrap_probe.py
```
- **Container Exit Code**: `0` (`2735 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
sweep raw affine coefficients [[ 8.000200e-02 -7.000000e-02  6.000000e-02 -8.000200e-02]
 [ 1.269996e+00  1.000000e-06  0.000000e+00 -1.000000e-06]
 [ 1.000000e-06  8.000030e-01 -0.000000e+00 -1.000000e-06]
 [ 1.000000e-06  0.000000e+00  1.250000e+00  3.000000e-06]
 [ 2.000000e-06  1.000000e-06 -1.000000e-06  8.200010e-01]]
local telemetry median [-0.23287832  0.24748223 -0.55048586  0.00407196]
wrap wrap 0.01435044200725487 [0.01704074 0.01804334 0.00136246]
sealed wrap 0.02649864410612806 [0.03620026 0.02738315 0.0067999 ]
mirror wrap 0.02754551585359148 [0.04028054 0.01936178 0.01669927]
wrap sealed 0.027678307222158346 [0.03530679 0.03224181 0.00348741]
wrap mirror 0.031804958629353126 [0.04409293 0.02934274 0.0151487 ]
sealed sealed 0.03503363092732675 [0.04509583 0.0390843  0.01099317]
mirror sealed 0.037184996455694105 [0.0495244  0.03620039 0.01962237]
sealed mirror 0.037830402952205903 [0.05045629 0.03602576 0.02120674]
fit wrap wrap [0.        0.        0.        0.        1.0355357 1.0349044 0.9957358
 1.0077161 1.1366049 1.0797104 0.7840226 1.5109291] [3.42781140e-06 3.32831841e-06 3.40702167e-06]
fit wrap sealed [0.        0.        0.        0.        1.0355357 1.0349044 0.9957358
 1.0077161 1.1366049 1.0797104 0.7840226 1.5109291] [3.42781139e-06 3.32831869e-06 3.40702140e-06]
fit sealed wrap [0.        0.        0.        0.        1.0355357 1.0349044 0.9957358
 1.0077161 1.1366049 1.0797104 0.7840226 1.5109291] [3.42781165e-06 3.32831872e-06 3.40702110e-06]
```

### Turn 6 — `execute_in_remote_docker` (`29177 ms` LLM)
- **Rationale**: Toggle face averaging and endpoint advection to determine which terms are needed. Check the fitted seam on omitted pulses before choosing whether to ship an exact stencil or investigate the remaining discrepancy.
- **Alternatives Weighed**: `Ship fitted edge multipliers` (rejected: They may encode the public grid spacing and diffusivity rather than a stencil that transfers to new studies.)
- **ATMS NOGOOD DECLARED**: Refuted `Nearest-neighbor wall rows with adjustable Robin leakage and inward diffusion suffice.` | Witness: `The best local stencil leaves worst-pulse RMSE 0.0369; adding conservative opposite-edge entries reduces it to 3.43e-6 across three calibrated tracer pulses.` | Blanket Diagnosis: `The local stencil omits opposite-edge flux; nuisance projection does not eliminate its residual.` | Sublated Pivot: `Preserve the interior upwind stencil and test the exact periodic seam with shared face diffusivity and row-local advection.`

```bash
cd /holdout_task && cat _verified_upstream.py batch_solver.py && cat > /tmp/periodic_probe.py <<'PY'
import sys,json,pickle
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
path='/tmp/wrap_probe.py'
exec(open(path).read().split('from itertools import product')[0])
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
ps=item['campaign_diagnostics']; modes=np.stack([np.tensordot(p['w_pulse'],s['source_modes'],axes=1) for p in ps]); y=np.column_stack([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in ps])
def coeff(inface, seam, adv):
    inward=[]; across=[]
    for axis in 'xy':
        k=s['kappa_'+axis]; v=s['vel_'+axis]; h=s['L'+axis]/(len(k)-1)
        e=k[[0,-1]]; near=k[[1,-2]]; t=v[[0,-1]]*h*np.array([1,-1]);
        inward.extend(((e+near)/2 if inface else e)/e + (np.maximum(-t,0)/e if adv else 0)); across.extend((e.mean() if seam else e)/e + (np.maximum(t,0)/e if adv else 0))
        if inface and seam and adv: print(axis,'endpoint k',e,'near',near,'v*h',t)
    return np.r_[np.zeros(4),inward,across]
for i in (False,True):
 for j in (False,True):
  for k in (False,True):
   p=coeff(i,j,k); e=forward(s,p,modes)-y; print('avg inward, avg seam, upwind',i,j,k,'params',p[4:].round(7),'pulse RMSE',np.sqrt(np.mean(e*e,axis=0)))
p=coeff(True,True,True)
for hold in range(len(ps)):
    keep=np.arange(len(ps))!=hold
    fit=least_squares(lambda x:(forward(s,np.r_[np.zeros(4),x],modes[:, :, :][keep])-y[:,keep]).ravel(),p[4:],bounds=(0,10),gtol=1e-13,ftol=1e-13,xtol=1e-13)
    e=forward(s,np.r_[np.zeros(4),fit.x],modes[hold:hold+1])[:,0]-y[:,hold]
    print('holdout',hold,'free conservative seam fit',fit.x.round(7),'RMSE',np.sqrt(np.mean(e*e)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/periodic_probe.py
```
- **Container Exit Code**: `0` (`904 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
s _dn

_PAY = json.load(open('/holdout_task/_upstream_param_payload.json', 'r'))

def solve_batch(item: dict) -> dict:
    fn_u = getattr(_up, 'calibrate_upstream')
    cal = fn_u(item, _PAY) if len(inspect.signature(fn_u).parameters) >= 2 else fn_u(item)
    return _dn.solve_downstream(cal)
avg inward, avg seam, upwind False False False params [1. 1. 1. 1. 1. 1. 1. 1.] pulse RMSE [0.01704074 0.01804334 0.00136246]
avg inward, avg seam, upwind False False True params [1.        1.        1.        1.        1.1364505 1.0802036 1.
 1.1305579] pulse RMSE [0.01496475 0.01414738 0.00133037]
avg inward, avg seam, upwind False True False params [1.        1.        1.        1.        1.        1.        0.7839644
 1.3803921] pulse RMSE [0.00093241 0.00362724 0.000684  ]
avg inward, avg seam, upwind False True True params [1.        1.        1.        1.        1.1364505 1.0802036 0.7839644
 1.51095  ] pulse RMSE [0.00110305 0.00055899 0.00080638]
avg inward, avg seam, upwind True False False params [1.035376  1.035376  0.9956245 1.0077043 1.        1.        1.
 1.       ] pulse RMSE [0.01825285 0.01758555 0.00121815]
avg inward, avg seam, upwind True False True params [1.035376  1.035376  0.9956245 1.0077043 1.1364505 1.0802036 1.
 1.1305579] pulse RMSE [0.01617216 0.01368552 0.00097286]
avg inward, avg seam, upwind True True False params [1.035376  1.035376  0.9956245 1.0077043 1.        1.        0.7839644
 1.3803921] pulse RMSE [0.00141119 0.00311147 0.00028264]
x endpoint k [0.47354451 0.47354451] near [0.50704871 0.50704871] v*h [0.06461538 0.03797997]
y endpoint k [0.55546396 0.31546396] near [0.55060312 0.3203248 ] v*h [-0.          0.04118632]
avg inward, avg seam, upwind True True True params [1.035376  1.035376  0.9956245 1.0077043 1.1364505 1.0802036 0.7839644
 1.51095  ] pulse RMSE [4.14056782e-06 4.05051606e-06 4.11760796e-06]
x endpoint k [0.47354451 0.47354451] near [0.50704871 0.50704871] v*h [0.06461538 0.03797997]
y endpoint k [0.55546396 0.31546396] near [0.55060312 0.3203248 ] v*h [-0.          0.04118632]
holdout 0 free conservative seam fit [1.0354787 1.0350457 0.9958094 1.0074691 1.1367128 1.0801782 0.7841233
 1.5108454] RMSE 7.738593500262892e-06
holdout 1 free conservative seam fit [1.0354415 1.0350113 0.9958033 1.0075844 1.1365695 1.0799025 0.7840244
 1.5106564] RMSE 3.433086450810427e-06
holdout 2 free conservative seam fit [1.036234  1.034183  0.9956994 1.0077648 1.13694   1.0785243 0.7839874
 1.5110007] RMSE 4.080750975094624e-06
```

### Turn 7 — `execute_in_remote_docker` (`40139 ms` LLM)
- **Rationale**: Implement the exact periodic stencil and reuse the archived feasible active-set solver. Run the public schema gate to catch interface or field-shape failures before independently checking the transport and certificate.
- **Alternatives Weighed**: `Ship numerically fitted seam multipliers` (rejected: The exact stencil is already at the calibration residual scale, while fitted multipliers gave a worse omitted-pulse result of 7.74e-6.)

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/hegel_continual_archive/dev_beta_m99_00_batch_01_solver.py').read_text()
qp = p[p.index('def _box_simplex('):p.index('def solve_batch(')]
head = '''"""Periodic chamber transport and certified box-simplex source inversion."""

import numpy as np
from scipy.linalg import solve
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, vel, h):
    n = len(kappa)
    row = np.arange(n)
    left, right = (row - 1) % n, (row + 1) % n
    face = (kappa + kappa[right]) / (2 * h**2)
    lower = -face[left] - np.maximum(vel, 0) / h
    upper = -face - np.maximum(-vel, 0) / h
    return coo_matrix((np.r_[lower, -lower-upper, upper],
                       (np.tile(row, 3), np.r_[left, row, right])),
                      shape=(n, n)).tocsc()


def _forward(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = (_axis(study['kappa_' + key], study['vel_' + key],
                    study['L' + key] / (n - 1))
              for key, n in zip('xy', (nx, ny)))
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx))
                + diags(study['V_reaction_2d'].ravel(order='F'), 0)).tocsc()
    modes = study['source_modes']
    rhs = modes.transpose(0, 2, 1).reshape(len(modes), nx * ny).T
    ij = np.asarray(study['sensor_ij'])
    positions = ij[:, 0] + nx * ij[:, 1]
    identity = eye(nx * ny, format='csc')
    blocks = []
    for shift in study['mu_shifts']:
        u = splu(operator + shift * identity).solve(rhs)
        blocks.append(u[positions])
    terminal_modes = u.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)
    return np.vstack(blocks), terminal_modes


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = map(np.asarray, (sweeps['ref_rosette'], sweeps['raw_rosette']))
    valid = np.all(np.abs(raw) <= 8.5, axis=1)
    design = np.column_stack((np.ones(valid.sum()), ref[valid]))
    coeff = np.linalg.lstsq(design, raw[valid], rcond=None)[0]
    groups = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return solve(coeff[1:].T, (groups - coeff[0]).T).T.ravel()


'''
tail = '''def solve_downstream(cal_item: dict) -> dict:
    S, terminal_modes = _forward(cal_item)
    y = _calibrate(cal_item)
    Z = np.asarray(cal_item['Z_nuisance'])
    ridge = cal_item['nuisance_ridge']
    if Z.shape[1]:
        Pz = solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T, assume_a='pos')
        S = S - Z @ (Pz @ S)
        y = y - Z @ (Pz @ y)
    alpha = cal_item['source_ridge']
    cap, budget = float(cal_item['u_box']), float(cal_item['total_budget'])
    H = S.T @ S + alpha * np.eye(S.shape[1])
    w, nu = _box_simplex(H, S.T @ y, cap, budget)
    residual = S @ w - y
    g = S.T @ residual + alpha * w
    reduced = g - nu
    J = 0.5 * (residual @ residual + alpha * (w @ w))
    dual = J - g @ w + nu * budget + cap * np.minimum(reduced, 0).sum()
    return {'weights': w,
            'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
            'kkt_certificate': {
                'weights': w.tolist(), 'nu_sum': float(nu),
                'lam_lower': np.maximum(reduced, 0).tolist(),
                'lam_upper': np.maximum(-reduced, 0).tolist(),
                'primal_objective': float(J), 'dual_lower_bound': float(dual)}}
'''
Path('downstream_solver.py').write_text(head + qp + tail)
PY
python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
sed -n '54,110p' downstream_solver.py
```
- **Container Exit Code**: `0` (`436 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py']`
- **Stdout (tail)**:
```text
s": []}
    tol = 2e-12
    if budget == 0:
        return np.zeros(n), float((-c).min())
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float((H @ w - c).max())
    w = np.full(n, budget / n)
    side = np.zeros(n, dtype=int)  # -1: lower, +1: upper, 0: free
    while True:
        free = np.flatnonzero(side == 0)
        g = H @ w - c
        if not len(free):
            lower, upper = np.flatnonzero(side == -1), np.flatnonzero(side == 1)
            i, j = lower[g[lower].argmin()], upper[g[upper].argmax()]
            if g[j] <= g[i] + tol:
                return w, float((g[i] + g[j]) / 2)
            side[[i, j]] = 0
            continue
        locked = np.flatnonzero(side)
        face = H[np.ix_(free, free)]
        rhs = c[free] - H[np.ix_(free, locked)] @ w[locked]
        v, q = solve(face, np.stack((rhs, np.ones(len(free))), axis=1),
                     assume_a='pos').T
        nu = (budget - w[locked].sum() - v.sum()) / q.sum()
        target = v + nu * q
        direction = target - w[free]
        step = np.full(len(free), np.inf)
        below, above = target < -tol, target > cap + tol
        step[below] = -w[free[below]] / direction[below]
        step[above] = (cap - w[free[above]]) / direction[above]
        hit = step.argmin()
        if step[hit] < 1:
            w[free] += step[hit] * direction
            idx = free[hit]
            side[idx] = -1 if below[hit] else 1
            w[idx] = 0 if below[hit] else cap
            continue
        w[free] = np.clip(target, 0, cap)
        violation = side * (H @ w - c - nu)
        release = violation.argmax()
        if violation[release] <= tol:
            return w, float(nu)
        side[release] = 0


def solve_downstream(cal_item: dict) -> dict:
    S, terminal_modes = _forward(cal_item)
    y = _calibrate(cal_item)
    Z = np.asarray(cal_item['Z_nuisance'])
    ridge = cal_item['nuisance_ridge']
    if Z.shape[1]:
        Pz = solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T, assume_a='pos')
        S = S - Z @ (Pz @ S)
        y = y - Z @ (Pz @ y)
    alpha = cal_item['source_ridge']
    cap, budget = float(cal_item['u_box']), float(cal_item['total_budget'])
    H = S.T @ S + alpha * np.eye(S.shape[1])

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=803b8e5a272c9be4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_803b8e5a (directory).
```

### Turn 8 — `execute_in_remote_docker` (`47280 ms` LLM)
- **Rationale**: Verify the exact staged artifacts in an isolated Python process against an independently assembled stencil. Per-shift tracer residuals can reveal a transport or ordering error; known-optimum interior, mixed, vertex, and saturated cases can reveal an active-set or certificate error.
- **Alternatives Weighed**: `Submit on the schema pass alone` (rejected: The schema checks shapes but cannot detect a transposed field, incorrect shift order, or invalid certificate.)

```bash
cd /holdout_task && cat > /tmp/verify_m99.py <<'PY'
import json, pickle, sys
import numpy as np
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import spsolve
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import solve_downstream
from _verified_upstream import calibrate_upstream

item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
pay = json.load(open('/holdout_task/_upstream_param_payload.json'))

def reference(st):
    nx, ny = st['nx'], st['ny']
    A = lil_matrix((nx*ny, nx*ny))
    for j in range(ny):
        for i in range(nx):
            row = i + nx*j
            A[row, row] = st['V_reaction_2d'][i,j]
            for axis, coord, n, stride in [('x',i,nx,1),('y',j,ny,nx)]:
                h = st['L'+axis]/(n-1)
                k, v = st['kappa_'+axis], st['vel_'+axis][coord]
                for direction in (-1,1):
                    neighbor = (coord+direction)%n
                    rate = (k[coord]+k[neighbor])/(2*h*h) + max(-direction*v,0)/h
                    A[row,row+(neighbor-coord)*stride] -= rate
                    A[row,row] += rate
    modes = st['source_modes']
    rhs = np.stack([m.flatten(order='F') for m in modes],axis=1)
    mats = [spsolve(A.tocsc() + shift*eye(nx*ny,format='csc'),rhs) for shift in st['mu_shifts']]
    ij = st['sensor_ij']
    S = np.vstack([u[ij[:,0]+nx*ij[:,1],:] for u in mats])
    U = np.stack([u.reshape(nx,ny,order='F') for u in mats[-1].T])
    return S,U

def verify(st,S,U,y,out):
    Z = st['Z_nuisance']
    whiten = lambda b: b - Z @ np.linalg.solve(Z.T @ Z + st['nuisance_ridge']*np.eye(Z.shape[1]), Z.T @ b)
    S,y = whiten(S),whiten(y)
    w, cert = out['weights'],out['kkt_certificate']
    cap, budget = st['u_box'],st['total_budget']
    nu = cert['nu_sum']
    ll, ul = np.asarray(cert['lam_lower']),np.asarray(cert['lam_upper'])
    r = S @ w - y
    g = S.T @ r + st['source_ridge']*w
    obj = (r@r + st['source_ridge']*(w@w))/2
    np.testing.assert_allclose(out['terminal_field'], np.tensordot(w,U,axes=1), rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(cert['weights'],w)
    assert np.all(ll >= 0) and np.all(ul >= 0)
    assert min(w) >= -1e-12 and max(w) <= cap+1e-12
    assert abs(sum(w)-budget)<1e-10
    dual = obj - g@w + nu*budget + cap*np.minimum(g-nu,0).sum()
    assert abs(obj-cert['primal_objective'])<1e-10 and abs(dual-cert['dual_lower_bound'])<1e-10
    errs = (abs(sum(w)-budget),np.max(abs(g-nu-ll+ul)),np.max(abs(np.r_[ll*w,ul*(cap-w)])),abs(obj-dual))
    assert max(errs)<1e-9, errs
    return errs

S,U = reference(s)
cal = calibrate_upstream(s,pay)
y = cal['y_obs_raw']
errs = [verify(s,S,U,y,solve_batch(s))]
res = []
for p in item['campaign_diagnostics']:
    yp = calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw']
    res.append(np.sqrt(np.mean((S@p['w_pulse']-yp).reshape(len(s['mu_shifts']),-1)**2,axis=1)))
print('independent periodic tracer RMSE by pulse and shift:', np.array(res).tolist())
assert np.max(res) < 1e-5

# Perturb the study on the same supported grid: opposite velocity signs,
# different reaction, shifts, and a full affine rosette in downstream coordinates.
t = dict(cal, vel_x=-cal['vel_x'], vel_y=0.4-cal['vel_y'], V_reaction_2d=cal['V_reaction_2d']*1.4, mu_shifts=np.asarray([0.23,0.72,1.34]))
S,U = reference(t)
ref = t['sensor_calibration_sweeps']['ref_rosette']
B = np.array([[1.2,.04,0,0],[-.08,1.02,.05,0],[0,.06,.83,.04],[.1,0,-.04,1.05]])
b = np.array([.11,-.06,.01,.2])
t['sensor_calibration_sweeps'] = {'ref_rosette':ref,'raw_rosette':ref@B+b}
Z=t['Z_nuisance']; P=np.eye(len(S))-Z@np.linalg.solve(Z.T@Z+t['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
T=P@S
n = S.shape[1]; cap = t['u_box']; rng=np.random.default_rng(912)
for kind in ('interior','mixed','vertex','zero','full'):
    if kind=='interior': w=cap*rng.uniform(.15,.85,n)
    if kind=='mixed': w=cap*np.resize([0,1,.4,.7],n)
    if kind=='vertex': w=cap*(np.arange(n)%2)
    if kind=='zero': w=np.zeros(n)
    if kind=='full': w=np.full(n,cap)
    grad=.025+np.where(w==0,.01,np.where(w==cap,-.02,0))
    y=S@w+np.linalg.lstsq(T.T@P,t['source_ridge']*w-grad,rcond=None)[0]
    st=dict(t,total_budget=w.sum(),y_obs_raw=(y.reshape(-1,4)@B+b).ravel())
    out=solve_downstream(st)
    np.testing.assert_allclose(out['weights'],w,rtol=0,atol=1e-8)
    errs.append(verify(st,S,U,y,out))
print('worst budget, stationarity, complementarity, dual gap:',np.max(errs,axis=0).tolist())
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_m99.py && python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`529 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
independent periodic tracer RMSE by pulse and shift: [[4.545465434460422e-06, 4.0164279167523205e-06, 3.826219604099568e-06], [4.402718507925653e-06, 3.9419822412189165e-06, 3.7811224392963754e-06], [4.497388993693224e-06, 4.0014941901139675e-06, 3.824346021622055e-06]]
worst budget, stationarity, complementarity, dual gap: [0.0, 3.3306690738754696e-15, 3.438915818776422e-16, 5.329070518200751e-15]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=803b8e5a272c9be4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_803b8e5a (directory).
```

### Turn 9 — `submit_completed_task` (`15009 ms` LLM)
- **Rationale**: Independent assembly verified the periodic seam and terminal field. The isolated run verified primal feasibility, stationarity, complementarity, and the dual bound for interior, mixed, vertex, and saturated budgets. The public schema passed on the same staged bundle.
- **Alternatives Weighed**: `Use fitted edge multipliers` (rejected: The parameter-free periodic stencil matched all public pulses at the calibration residual scale and had a lower worst-pulse error than the leave-one-pulse-out fitted model.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:408', '/holdout_task/downstream_solver.py': 'FILE:4971'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Verified both solvers. The periodic upwind stencil matched the public tracer data with a worst pulse-by-shift RMSE of 4.55e-6. The public schema passed, and the worst KKT duality gap across the tested cases was 5.33e-15.`

### Turn 10 — `execute_in_remote_docker` (`23897 ms` LLM)
- **Rationale**: A bank-specific calibration error could be hidden by the pooled tracer score. Compare staged and sweep-fitted calibration on each pulse-by-shift-by-bank slice, then test whether a nuisance correction learned on other shifts improves the held shift. A material improvement would justify revisiting calibration; otherwise retain the verified stencil and prescribed nuisance projection.
- **Alternatives Weighed**: `Test relay modes, classification balance, or stochastic drift` (rejected: This study has no relay state, class labels, or stochastic drift operator. Its relevant slices are tracer pulse, sensor bank, and transport shift.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import json, pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import _calibrate, _forward
item = pickle.load(open('public_batch.pkl', 'rb'))[0]
s = item['study']
pay = json.load(open('_upstream_param_payload.json'))
S, _ = _forward(s)
Z = s['Z_nuisance']
shifts = np.repeat(np.arange(len(s['mu_shifts'])), len(s['sensor_ij']))
banks = s['row_bank_id']
for mode, payload in [('staged', pay), ('sweep-fit', None)]:
    E = np.column_stack([S @ p['w_pulse'] - _calibrate(calibrate_upstream(dict(s, y_obs_raw=p['y_pulse_raw']), payload)) for p in item['campaign_diagnostics']])
    rmse = lambda x: float(np.sqrt(np.mean(x*x)))
    slices = [(rmse(E[(shifts == k) & (banks == b), p]), int(k), int(b), p) for k in np.unique(shifts) for b in np.unique(banks) if np.any((shifts == k) & (banks == b)) for p in range(E.shape[1])]
    cross = []
    for k in np.unique(shifts):
        train = shifts != k
        z = Z[train]
        gamma = np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ E[train])
        cross.append((int(k), rmse(E[~train]), rmse(E[~train] - Z[~train] @ gamma)))
    print(mode, 'overall RMSE', rmse(E), 'worst RMSE / shift / bank / pulse', max(slices), 'held-shift nuisance RMSE (shift, raw, corrected)', cross)
    assert max(slices)[0] < 1e-5
PY
```
- **Container Exit Code**: `0` (`394 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
staged overall RMSE 4.103075172607802e-06 worst RMSE / shift / bank / pulse (7.704055653785133e-06, 0, 2, 0) held-shift nuisance RMSE (shift, raw, corrected) [(0, 4.482249957182798e-06, 4.718365215501277e-06), (1, 3.98676447582567e-06, 3.782890648725222e-06), (2, 3.8106196283335472e-06, 4.021149760207461e-06)]
sweep-fit overall RMSE 1.076521969878297e-06 worst RMSE / shift / bank / pulse (1.7017809434713763e-06, 0, 1, 0) held-shift nuisance RMSE (shift, raw, corrected) [(0, 1.0894070106278006e-06, 1.1505918919079344e-06), (1, 1.0595276249748134e-06, 1.048755997676933e-06), (2, 1.0804129914075347e-06, 1.0910899179332685e-06)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=803b8e5a272c9be4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_803b8e5a (directory).
```

### Turn 11 — `submit_completed_task` (`16926 ms` LLM)
- **Rationale**: The independently assembled periodic stencil matches the public tracers, and held-shift nuisance fitting did not consistently improve predictions. The unchanged bundle passed isolated checks of terminal-field ordering, feasibility, stationarity, complementarity, and the dual bound.
- **Alternatives Weighed**: `Replace the staged upstream payload with a sweep-only fit` (rejected: The sweep-only fit reduced the worst slice to 1.70e-6, but the task specifies composition with the verified upstream payload. Retain that interface.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:408', '/holdout_task/downstream_solver.py': 'FILE:4971'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Both solvers passed the public schema and isolated numerical checks. With the required upstream calibration, the periodic stencil’s worst tracer RMSE across pulse, shift, and sensor-bank slices was 7.70e-6. The worst tested KKT duality gap was 5.33e-15.`

### Turn 12 — `execute_in_remote_docker` (`30428 ms` LLM)
- **Rationale**: Ablate only the seam connection, shared diffusivity, or endpoint upwinding, preserving each row sum. Compare on the same real tracer slices before and after nuisance projection. If an ablation is competitive, revisit that term; otherwise retain and verify the unchanged bundle.
- **Alternatives Weighed**: `Fit a more flexible boundary law` (rejected: First test whether each periodic seam term has measurable support on the real pulses. A flexible law is unwarranted if the exact stencil stays at the calibration residual scale and its ablations fail.)

```bash
cd /holdout_task && cat > /tmp/m99_seam_ablation.py <<'PY'
import json, pickle, sys
import numpy as np
from scipy.sparse import coo_matrix
sys.path.insert(0, '/holdout_task')
import downstream_solver as dn
from _verified_upstream import calibrate_upstream

item = pickle.load(open('public_batch.pkl', 'rb'))[0]
s = item['study']
pay = json.load(open('_upstream_param_payload.json'))
pulses = item['campaign_diagnostics']
W = np.column_stack([p['w_pulse'] for p in pulses])
Y = np.column_stack([dn._calibrate(calibrate_upstream(dict(s, y_obs_raw=p['y_pulse_raw']), pay)) for p in pulses])
Z = s['Z_nuisance']
Pz = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
shape = (len(s['mu_shifts']), len(s['sensor_ij']), len(pulses))
axis = dn._axis

def ablation(name):
    def assemble(kappa, vel, h):
        a = axis(kappa, vel, h).tolil()
        for i, j, sign in ((0, len(kappa)-1, 1), (len(kappa)-1, 0, -1)):
            original = -a[i,j]
            rate = {'remove_wrap': 0., 'no_seam_advection': (kappa[i]+kappa[j])/(2*h*h),
                    'local_seam_diffusivity': kappa[i]/(h*h)+max(sign*vel[i],0.)/h}[name]
            a[i,j] = -rate
            a[i,i] += rate-original
        return a.tocsc()
    return assemble

banks = np.asarray(s['row_bank_id']).reshape(shape[:2])
results = {}
for name in ('exact_periodic', 'remove_wrap', 'no_seam_advection', 'local_seam_diffusivity'):
    dn._axis = axis if name == 'exact_periodic' else ablation(name)
    S, _ = dn._forward(s)
    e = S @ W - Y
    projected = e - Z @ (Pz @ e)
    raw = e.reshape(shape)
    worst = max(float(np.sqrt(np.mean(raw[k, banks[k]==b, p]**2))) for k in range(shape[0]) for b in np.unique(banks[k]) for p in range(shape[2]))
    by_pulse = np.sqrt(np.mean(e**2, axis=0))
    results[name] = worst
    print(json.dumps({'model': name, 'pulse_rmse': by_pulse.tolist(), 'projected_pulse_rmse': np.sqrt(np.mean(projected**2,axis=0)).tolist(), 'worst_pulse_shift_bank_rmse': worst}))
dn._axis = axis
assert results['exact_periodic'] < 1e-5, results
assert all(results[a] > 10*results['exact_periodic'] for a in results if a != 'exact_periodic'), results
print('ng_t06: measured seam ablations verify the opposite-wall term, shared seam diffusion, and endpoint upwinding on the exact staged candidate')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m99_seam_ablation.py && OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_m99.py && python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`694 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"model": "exact_periodic", "pulse_rmse": [4.140567822806437e-06, 4.050516063968883e-06, 4.117607962941788e-06], "projected_pulse_rmse": [3.995354738030077e-06, 3.917397556105278e-06, 3.9891470925983625e-06], "worst_pulse_shift_bank_rmse": 7.704055653785133e-06}
{"model": "remove_wrap", "pulse_rmse": [0.045328274477937885, 0.03875020940783778, 0.011301046366217028], "projected_pulse_rmse": [0.04421862820669677, 0.038492873791623, 0.011200329945335478], "worst_pulse_shift_bank_rmse": 0.08454118303053244}
{"model": "no_seam_advection", "pulse_rmse": [0.0014111891378074935, 0.003111472211451126, 0.0002826386258072874], "projected_pulse_rmse": [0.0013611085084783876, 0.0030542952971603044, 0.0002812423441573793], "worst_pulse_shift_bank_rmse": 0.00492670865694915}
{"model": "local_seam_diffusivity", "pulse_rmse": [0.01617215835368765, 0.013685517760320627, 0.0009728609392552455], "projected_pulse_rmse": [0.015814122299877775, 0.013382351272636702, 0.0009515150368678649], "worst_pulse_shift_bank_rmse": 0.027570225954240257}
ng_t06: measured seam ablations verify the opposite-wall term, shared seam diffusion, and endpoint upwinding on the exact staged candidate
independent periodic tracer RMSE by pulse and shift: [[4.545465434460422e-06, 4.0164279167523205e-06, 3.826219604099568e-06], [4.402718507925653e-06, 3.9419822412189165e-06, 3.7811224392963754e-06], [4.497388993693224e-06, 4.0014941901139675e-06, 3.824346021622055e-06]]
worst budget, stationarity, complementarity, dual gap: [0.0, 3.3306690738754696e-15, 3.438915818776422e-16, 5.329070518200751e-15]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=803b8e5a272c9be4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_803b8e5a (directory).
```

### Turn 13 — `submit_completed_task` (`14194 ms` LLM)
- **Rationale**: The real tracer ablations support all three seam terms. Independent assembly and isolated execution verified terminal-field ordering and the KKT certificate across interior, mixed, vertex, and saturated budgets on the same bundle that passed the public schema.
- **Alternatives Weighed**: `Simplify the periodic seam` (rejected: Removing the wrap, omitting endpoint advection, or replacing shared face diffusivity each increased worst-slice tracer RMSE by more than 600-fold.)
- **Submitted Final Answer / Summary**: `Both solvers passed the public schema and isolated numerical checks. The periodic stencil’s worst tracer RMSE across pulse, shift, and sensor-bank slices was 7.70e-6. The worst tested KKT duality gap was 5.33e-15.`

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
