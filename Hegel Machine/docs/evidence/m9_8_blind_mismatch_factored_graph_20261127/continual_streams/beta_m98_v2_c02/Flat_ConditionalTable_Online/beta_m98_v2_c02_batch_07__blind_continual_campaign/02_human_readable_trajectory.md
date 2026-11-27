# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_v2_c02_batch_07__blind_continual_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M98-Blind-Mismatch-Factored-Lineage-World` |
| **Domain** | `2D Advection-Diffusion-Reaction Tomography & Blind Continual Campaign` |
| **Wave / Wormhole Family** | `wave_1` / `fam_elliptic_transport_and_factored_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `320,686` (`310,830` / `111,104` / `9,856`) |
| **Total Episode Wall Time (s)** | `352.06s` |

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

### Turn 1 — `execute_in_remote_docker` (`17004 ms` LLM)
- **Rationale**: Inspect the public interface and apparatus evidence before choosing a boundary operator; a missing or ambiguous match would require comparing archived candidates against calibration and observation residuals.
- **Alternatives Weighed**: `Start with the baseline boundary rows` (rejected: The apparatus log may identify a different regime that a schema-only test would miss.)

```bash
ls -lh /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; python3 - <<'PY'
import json,pickle
from pathlib import Path
with open('/holdout_task/public_batch.pkl','rb') as f: items=pickle.load(f)
print('public cases:',len(items))
for item in items:
 print('keys',list(item),'diagnostics',type(item.get('campaign_diagnostics')).__name__)
 for k,v in item['study'].items(): print(k,type(v).__name__,getattr(v,'shape',list(v) if isinstance(v,dict) else v if isinstance(v,(int,float,str)) else None))
p=Path('/tmp/hegel_continual_archive/archive_apparatus_log.json')
if p.exists(): print('apparatus log:',json.dumps(json.loads(p.read_text()),indent=1)[-12500:])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling' 2>&1
```
- **Container Exit Code**: `0` (`458 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-0.3803,
    -1.07374,
    0.2616,
    -0.25909,
    -0.85616,
    0.55851,
    2.29734,
    -1.10714
   ]
  ]
 },
 {
  "completed_batch_index": 6,
  "completed_short_name": "beta_m98_v2_c02_batch_06",
  "facility_telemetry_trace": [
   [
    2.55083,
    1.11136,
    0.72554,
    -0.95375,
    1.02971,
    0.77716,
    -0.30263,
    0.97169
   ],
   [
    2.52886,
    0.98617,
    0.85042,
    -0.92027,
    1.45909,
    0.81306,
    -0.417,
    0.87741
   ],
   [
    2.50647,
    1.24481,
    0.66271,
    -0.92724,
    1.08788,
    1.01832,
    -0.43744,
    1.28059
   ],
   [
    2.64647,
    1.28396,
    0.4207,
    -0.57202,
    1.37043,
    0.81322,
    -0.31658,
    1.21893
   ],
   [
    2.55811,
    1.11646,
    0.63861,
    -0.26265,
    1.07884,
    0.62627,
    -0.46026,
    0.99041
   ],
   [
    2.49465,
    1.18393,
    0.72616,
    0.0344,
    1.01312,
    0.74881,
    -0.29586,
    1.10981
   ],
   [
    2.59913,
    1.04859,
    0.99935,
    0.30604,
    1.09018,
    0.37082,
    -0.45869,
    0.87505
   ],
   [
    2.48687,
    1.47884,
    0.71611,
    0.51408,
    1.03641,
    0.74586,
    -0.3415,
    1.32957
   ],
   [
    2.60714,
    1.3064,
    0.75893,
    0.97726,
    1.03418,
    0.59218,
    -0.38576,
    0.73586
   ],
   [
    2.54068,
    1.3593,
    0.80334,
    0.87899,
    1.23874,
    0.84455,
    -0.40367,
    1.30199
   ],
   [
    2.65496,
    1.1515,
    0.69455,
    0.83467,
    1.13693,
    0.69514,
    -0.47476,
    0.97888
   ],
   [
    2.6058,
    1.16781,
    0.82131,
    0.51641,
    1.35621,
    0.54359,
    -0.36077,
    1.13604
   ],
   [
    2.51706,
    1.0531,
    0.81192,
    0.2228,
    1.29598,
    0.83742,
    -0.39993,
    1.01652
   ],
   [
    2.59455,
    1.06488,
    0.5482,
    -0.17835,
    1.14671,
    0.7751,
    -0.36242,
    1.12252
   ],
   [
    2.61026,
    1.07522,
    0.50507,
    -0.65929,
    1.1486,
    0.55082,
    -0.48239,
    1.03901
   ],
   [
    2.51345,
    1.08254,
    0.66759,
    -0.7482,
    1.34368,
    0.66208,
    -0.15924,
    1.1762
   ]
  ]
 }
]
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`18429 ms` LLM)
- **Rationale**: Summarize the telemetry and map prior solvers to distinct boundary laws. If telemetry does not identify one law, compare the plausible candidates on held-out observation blocks.
- **Alternatives Weighed**: `Reuse batch 04 as-is` (rejected: The public gate rejected that table row; compare the implemented wall laws and telemetry before deciding which parts to reuse.)

```bash
cd /tmp/hegel_continual_archive; cat archive_manifest.json; for f in *_solver.py; do echo ==== "$f"; grep -nE '^def |^[[:space:]]*(Ax|Ay|kind|regime|wall|family|cfg|boundary|kap|rob)' "$f"; done; cat /holdout_task/public_asset_index.json; python3 - <<'PY'
import json,pickle,numpy as np
x=pickle.load(open('/holdout_task/public_batch.pkl','rb')); print('items',len(x)); print('item keys',list(x[0]),'diagnostics',type(x[0].get('campaign_diagnostics')).__name__)
for k,v in x[0]['study'].items(): print(k, type(v).__name__, list(v) if isinstance(v,dict) else v if np.isscalar(v) else np.shape(v))
x=json.load(open('archive_apparatus_log.json'))
for a in x:
 t=np.array(a['facility_telemetry_trace']); print(a['completed_batch_index'],t.shape,'mean',t.mean(0).round(3),'std',t.std(0).round(3))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 2>&1 | head -25
```
- **Container Exit Code**: `0` (`492 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
}
          }
        },
        "len": 1,
        "type": "list"
      }
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 1970
    }
  },
  "schema": "hegel_public_asset_index_v1"
}items 1
item keys ['study', 'campaign_diagnostics'] diagnostics list
Lx float 2.4
Ly float 2.0
nx int 14
ny int 12
kappa_x ndarray (14,)
kappa_y ndarray (12,)
vel_x ndarray (14,)
vel_y ndarray (12,)
robin_x float 1.2080255131959075
robin_y float 1.2589843637401956
V_reaction_2d ndarray (14, 12)
mu_shifts ndarray (3,)
source_modes ndarray (8, 14, 12)
sensor_ij ndarray (32, 2)
row_bank_id ndarray (96,)
sensor_calibration_sweeps dict ['ref_rosette', 'raw_rosette', 'single_bank_sweeps']
Z_nuisance ndarray (96, 3)
y_obs_raw ndarray (96,)
aux_manifold_telemetry ndarray (24, 4)
nuisance_ridge float 0.08
source_ridge float 0.04
u_box float 0.42
total_budget float 1.4
0 (16, 8) mean [-0.378  1.163 -0.66  -0.03   1.146 -0.637  2.421  1.17 ] std [0.09  0.114 0.138 0.646 0.105 0.124 0.072 0.13 ]
1 (16, 8) mean [-0.367 -1.227 -0.392  0.026 -1.129 -0.459 -0.356 -1.225] std [0.057 0.112 0.131 0.58  0.157 0.156 0.065 0.153]
2 (16, 8) mean [ 2.551  1.158 -0.25   0.091  1.119 -0.239 -0.33   1.141] std [0.081 0.169 0.12  0.679 0.167 0.147 0.063 0.096]
3 (16, 8) mean [-0.404 -1.11   0.009  0.025 -1.111 -0.004 -0.375 -1.193] std [0.068 0.124 0.18  0.669 0.122 0.112 0.081 0.109]
4 (16, 8) mean [-0.44   1.156  0.291 -0.005  1.136  0.21   2.451  1.168] std [0.077 0.137 0.167 0.699 0.146 0.172 0.06  0.145]
5 (16, 8) mean [-0.418 -1.154  0.408  0.036 -1.119  0.471  2.426 -1.133] std [0.094 0.173 0.111 0.637 0.138 0.126 0.061 0.109]
6 (16, 8) mean [ 2.563  1.17   0.709 -0.059  1.179  0.713 -0.379  1.073] std [0.052 0.129 0.137 0.665 0.14  0.149 0.081 0.162]
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

### Turn 3 — `execute_in_remote_docker` (`19409 ms` LLM)
- **Rationale**: Read the archived implementations and pulse layout to identify which boundary and calibration differences need testing. If the archived laws fail on held-out pulses, fit the boundary rows directly instead of carrying forward the failed table choice.
- **Alternatives Weighed**: `Select the wall law from the apparatus log alone` (rejected: The public item has diagnostic pulses, which can distinguish candidate wall laws directly.)

```bash
cd /tmp/hegel_continual_archive; cat beta_m98_v2_c02_batch_06_solver.py; diff -u beta_m98_v2_c02_batch_06_solver.py beta_m98_v2_c02_batch_04_solver.py || true; python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=x['study']; d=x['campaign_diagnostics']; print('pulses',len(d))
for p in d:
 print({k:(np.shape(v),float(np.linalg.norm(v))) for k,v in p.items()})
for k,v in s['sensor_calibration_sweeps'].items():
 print('cal',k,type(v),np.shape(v) if not isinstance(v,dict) else {a:np.shape(b) for a,b in v.items()})
print('bank order',np.asarray(s['row_bank_id']).reshape(-1,4)[:8]); print('shifts',s['mu_shifts'])
PY
```
- **Container Exit Code**: `0` (`406 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(0, g - nu)
    upper = np.maximum(0, nu - g)
    bound = float(value - g @ w + nu * total - cap * upper.sum())
    return {'weights': w, 'terminal_field': np.tensordot(w, last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': lower, 'lam_upper': upper,
                                'primal_objective': value, 'dual_lower_bound': bound}}
--- beta_m98_v2_c02_batch_06_solver.py	2026-11-27 14:49:03.149397583 +0000
+++ beta_m98_v2_c02_batch_04_solver.py	2026-11-27 14:49:03.149186185 +0000
@@ -1,4 +1,4 @@
-"""Reflecting-wall transport tomography with sweep-fitted rosette calibration."""
+"""Steady transport tomography with empirical rosette calibration."""
 
 import numpy as np
 from scipy import sparse
@@ -6,7 +6,7 @@
 from scipy.sparse.linalg import splu
 
 
-def _axis(kappa, velocity, h):
+def _axis(kappa, velocity, h, robin):
     faces = (0.5 / h**2) * (kappa[:-1] + kappa[1:])
     lower = -faces - np.maximum(velocity[1:], 0) / h
     upper = -faces - np.maximum(-velocity[:-1], 0) / h
@@ -14,15 +14,15 @@
     upper[0] = -kappa[0] / h**2
     diag = np.empty_like(kappa, dtype=float)
     diag[1:-1] = -lower[:-1] - upper[1:]
-    diag[0] = -upper[0]
-    diag[-1] = -lower[-1]
+    diag[0] = -upper[0] + robin / h
+    diag[-1] = -lower[-1] + robin / h
     return sparse.diags((lower, diag, upper), (-1, 0, 1), format='csc')
 
 
 def _forward(study):
     nx, ny = study['nx'], study['ny']
-    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1))
-    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1))
+    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1), study['robin_x'])
+    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1), study['robin_y'])
     reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
     operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                 + sparse.kron(ay, sparse.eye(nx), format='csc')
pulses 2
{'w_pulse': ((8,), 1.118033988749895), 'y_pulse_raw': ((96,), 1.9420266776710036)}
{'w_pulse': ((8,), 1.118033988749895), 'y_pulse_raw': ((96,), 1.96300944537147)}
cal ref_rosette <class 'numpy.ndarray'> (80, 4)
cal raw_rosette <class 'numpy.ndarray'> (80, 4)
cal single_bank_sweeps <class 'dict'> {'bank_id': (24,), 'ref_signal': (24,), 'raw_readout': (24,)}
bank order [[0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]]
shifts [0.15 0.55 1.1 ]
```

### Turn 4 — `execute_in_remote_docker` (`18852 ms` LLM)
- **Rationale**: Compare known-source pulses by shift to detect a wall or calibration mismatch that a schema check cannot reveal. If no archived combination fits both pulses, separate calibration selection from boundary identification.
- **Alternatives Weighed**: `Fit continuous boundary coefficients immediately` (rejected: An exact archived law may explain both pulses; test the discrete candidates before introducing fitted coefficients.)

```bash
cd /tmp/hegel_continual_archive; for f in *_solver.py; do echo ==== "$f"; grep -nE '^def |diag\[0\]|diag\[-1\]|ax =|ay =' "$f"; done; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import glob, importlib.util, pickle, numpy as np
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']
for f in sorted(glob.glob('*_solver.py')):
 spec=importlib.util.spec_from_file_location('archived',f); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
 try:
  S,_=m._forward(s)
  scores=[]
  for p in pulses:
   y=m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])); e=(S@p['w_pulse']-y).reshape(len(s['mu_shifts']),-1)
   scores.append(np.sqrt(np.mean(e**2,axis=1)).round(7).tolist())
  print(f, scores)
 except Exception as e: print(f, type(e).__name__, str(e))
for p in pulses: print('pulse support:',p['w_pulse'])
PY
```
- **Container Exit Code**: `0` (`528 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_x'], study['vel_x'], study['Lx'] / (nx-1), study['robin_x'])
25:    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1), study['robin_y'])
41:def _rosette_features(x):
45:def _fit_rosette(sweeps):
55:def _calibrate(study):
71:def _box_budget_qp(H, b, cap, total):
113:def solve_batch(study: dict) -> dict:
==== beta_m98_v2_c02_batch_05_solver.py
8:def _axis(kappa, velocity, h):
16:    diag[0] = -upper[0]
17:    diag[-1] = -lower[-1]
21:def _forward(study):
23:    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1))
24:    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1))
40:def _calibrate(study):
49:def _box_budget_qp(H, b, cap, total):
91:def solve_batch(study: dict) -> dict:
==== beta_m98_v2_c02_batch_06_solver.py
9:def _axis(kappa, velocity, h):
17:    diag[0] = -upper[0]
18:    diag[-1] = -lower[-1]
22:def _forward(study):
24:    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1))
25:    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1))
41:def _rosette_features(x):
45:def _fit_rosette(sweeps):
55:def _calibrate(study):
71:def _box_budget_qp(H, b, cap, total):
113:def solve_batch(study: dict) -> dict:
==== beta_m98_v2_c02_s00_solver.py
8:def _axis(kappa, velocity, h, robin):
16:    diag[0] = -upper[0] + robin / h
17:    diag[-1] = -lower[-1] + robin / h
21:def _forward(study):
23:    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1), study['robin_x'])
24:    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1), study['robin_y'])
40:def _calibrate(study):
49:def _box_budget_qp(H, b, cap, total):
91:def solve_batch(study: dict) -> dict:
beta_m98_v2_c02_batch_01_solver.py [[0.4224141, 0.4373546, 0.4464228], [0.4311887, 0.4445247, 0.4515724]]
beta_m98_v2_c02_batch_02_solver.py [[0.4932899, 0.4696751, 0.4646203], [0.4939272, 0.4707359, 0.4655783]]
beta_m98_v2_c02_batch_03_solver.py [[0.4374541, 0.4500408, 0.4564531], [0.4363723, 0.4493007, 0.4558179]]
beta_m98_v2_c02_batch_04_solver.py [[0.203086, 0.0866203, 0.0424276], [0.200515, 0.084512, 0.0405948]]
beta_m98_v2_c02_batch_05_solver.py [[0.4224141, 0.4373546, 0.4464228], [0.4311887, 0.4445247, 0.4515724]]
beta_m98_v2_c02_batch_06_solver.py [[0.0774728, 0.0491317, 0.033362], [0.0498444, 0.0332372, 0.0234953]]
beta_m98_v2_c02_s00_solver.py [[0.4932899, 0.4696751, 0.4646203], [0.4939272, 0.4707359, 0.4655783]]
pulse support: [1.  0.  0.  0.5 0.  0.  0.  0. ]
pulse support: [0.  1.  0.  0.  0.5 0.  0.  0. ]
```

### Turn 5 — `execute_in_remote_docker` (`27253 ms` LLM)
- **Rationale**: Use held-out calibration rows to test whether the archive's cubic and adjacent-channel terms suffice. If they do, use the calibrated tracer pulses to identify the wall stencil; if not, repair calibration before fitting the walls.
- **Alternatives Weighed**: `Fit wall coefficients against the current calibration immediately` (rejected: Calibration error could look like a boundary mismatch; the independent sweeps can isolate it first.)

```bash
sed -n '1,72p' /tmp/hegel_continual_archive/beta_m98_v2_c02_batch_06_solver.py; sed -n '1,24p' /tmp/hegel_continual_archive/beta_m98_v2_c02_batch_03_solver.py; cat > /tmp/cal07.py <<'PY'
import pickle,numpy as np
from itertools import combinations_with_replacement
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; q=s['sensor_calibration_sweeps']; x,y=q['ref_rosette'],q['raw_rosette']
print('ranges',x.min(0),x.max(0),y.min(0),y.max(0)); print('single sweeps', {k:np.asarray(v).reshape(4,6).round(5) for k,v in q['single_bank_sweeps'].items()})
sets=[('affine',[()]+[(i,) for i in range(4)]),('archive',[()]+[(i,) for i in range(4)]+[(i,)*3 for i in range(4)]+[tuple(sorted((i,(i+1)%4))) for i in range(4)])]; sets+= [('degree '+str(n),[()] + [a for d in range(1,n+1) for a in combinations_with_replacement(range(4),d)]) for n in (2,3,4)]
for name,terms in sets:
 D=np.array([x[:,a].prod(1) for a in terms]).T; err=[]
 for test in np.array_split(np.random.default_rng(42).permutation(len(x)),5):
  tr=np.setdiff1d(np.arange(len(x)),test); c=np.linalg.lstsq(D[tr],y[tr],rcond=None)[0]; et=abs(D[tr]@c-y[tr]).max(1); keep=et<=max(5*np.median(et),1e-10); c=np.linalg.lstsq(D[tr][keep],y[tr][keep],rcond=None)[0]; err.extend(abs(D[test]@c-y[test]).max(1));
 print(name,'OOF row max percentiles',np.quantile(err,[0,.5,.9,.95,1]));
 if name=='archive':
  c=np.linalg.lstsq(D,y,rcond=None)[0]; e=abs(D@c-y).max(1); keep=e<=max(5*np.median(e),1e-10); c=np.linalg.lstsq(D[keep],y[keep],rcond=None)[0]; print('archive coef',c.round(7),'removed',np.flatnonzero(~keep),'inlier max',abs(D[keep]@c-y[keep]).max())
print('sensor locations',s['sensor_ij'].T)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/cal07.py
```
- **Container Exit Code**: `0` (`465 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
'], study['vel_x'], study['Lx'] / (nx-1))
ranges [-1.39340786 -1.37457167 -1.3579545  -1.30391549] [1.79572023 1.66618143 1.68399407 1.76691299] [ -2.31053492 -12.8800143   -2.19213368 -13.59333512] [13.82295584  2.42089813 12.61559051  2.65461935]
single sweeps {'bank_id': array([[0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1],
       [2, 2, 2, 2, 2, 2],
       [3, 3, 3, 3, 3, 3]]), 'ref_signal': array([[-1.1, -0.6, -0.1,  0.4,  0.9,  1.4],
       [-1.1, -0.6, -0.1,  0.4,  0.9,  1.4],
       [-1.1, -0.6, -0.1,  0.4,  0.9,  1.4],
       [-1.1, -0.6, -0.1,  0.4,  0.9,  1.4]]), 'raw_readout': array([[-1.62466, -0.72776, -0.03823,  0.61674,  1.40998, 17.01421],
       [-1.13826, -0.58474, -0.15914,  0.2451 ,  0.73464,  1.41621],
       [-1.67147, -0.76054, -0.06023,  0.60497,  1.41057,  2.5321 ],
       [-1.14506, -0.57749, -0.14114,  0.27332,  0.77528,  1.47407]])}
affine OOF row max percentiles [ 0.03927245  0.25381499  0.67854586  1.00852963 13.47477506]
archive OOF row max percentiles [3.01295250e-06 1.55278203e-05 2.63672274e-05 2.81037473e-05
 1.35000052e+01]
archive coef [[ 9.0001200e-02 -7.9999600e-02  7.0000200e-02 -6.0000700e-02]
 [ 1.2799969e+00  1.1060590e-01 -4.5000000e-06 -2.2680720e-01]
 [-3.5840050e-01  7.9000380e-01  1.8200470e-01  1.8000000e-06]
 [-8.0000000e-07 -2.2120190e-01  1.3000055e+00  1.1339860e-01]
 [ 1.7920150e-01 -7.2000000e-06 -3.6399960e-01  8.1000700e-01]
 [ 2.3040120e-01 -3.3000000e-06  1.6000000e-06  3.3000000e-06]
 [-3.0000000e-07  1.4219750e-01 -2.2000000e-06 -8.0000000e-07]
 [-6.0000000e-07  2.1000000e-06  2.3399760e-01 -6.0000000e-07]
 [ 1.1000000e-06  1.9000000e-06 -1.1000000e-06  1.4579500e-01]
 [-2.8160110e-01  7.0000000e-07  4.0000000e-07 -1.9000000e-06]
 [ 2.1000000e-06 -1.7380060e-01 -6.0000000e-07 -6.0000000e-07]
 [ 8.0000000e-07 -4.0000000e-07 -2.8600140e-01  2.2000000e-06]
 [-1.6000000e-06 -1.0000000e-06  4.0000000e-07 -1.7820020e-01]] removed [13 14 15] inlier max 2.3156610302832448e-05
degree 2 OOF row max percentiles [ 0.02931227  0.17650944  0.45030726  0.4797663  13.56459925]
degree 3 OOF row max percentiles [3.20241781e-06 2.36519095e-05 7.15504607e-05 9.50532756e-05
 1.35000104e+01]
degree 4 OOF row max percentiles [2.07431570e-01 2.96358975e+01 2.48788055e+02 3.00760061e+02
 7.81462394e+02]
sensor locations [[ 0  6  6  6  6  2  6 12  2 10  0  8  8  8  4 10  2 12  4  8 10  8  8 10
  12 12  0  6  0 10  2 10]
 [10  4  8 10  0  0  2  4  8  8  0 10  4  0  6  0  2  2  4  6  6  8  2  4
   6  0  2  6  4 10  4  2]]
```

### Turn 6 — `execute_in_remote_docker` (`38462 ms` LLM)
- **Rationale**: The edgewise search tests whether a mixed exact law explains the pulses. If it fails, fitting on each pulse separately will show which missing boundary terms transfer rather than merely interpolating the public measurements.
- **Alternatives Weighed**: `Expand the rosette feature basis` (rejected: Held-out sweeps already support the cubic and adjacent-channel terms; a larger polynomial worsens clean-sweep prediction.)
- **ATMS NOGOOD DECLARED**: Refuted `A uniform archived wall stencil with the sweep-supported cubic rosette explains both tracer pulses.` | Witness: `With the same cubic calibration, worst pulse-by-shift RMSE is 0.0774728 for reflecting walls and 0.203086 for Robin walls.` | Blanket Diagnosis: `Independent sweeps support the rosette basis; boundary transport remains unresolved.` | Sublated Pivot: `Test wall laws independently on each edge, then use cross-pulse fits to identify any missing stencil term.`

```bash
cat > /tmp/wall07.py <<'PY'
import sys,pickle,itertools,importlib.util
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares

def load(b):
 p=f'/tmp/hegel_continual_archive/beta_m98_v2_c02_batch_{b:02d}_solver.py'; sp=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
m=load(6)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']; nx,ny=s['nx'],s['ny']; N=nx*ny
W=np.stack([p['w_pulse'] for p in pulses],1)
Y=np.stack([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses],1).reshape(-1,len(s['sensor_ij']),len(pulses))
rows=s['sensor_ij']@[1,nx]; modes=s['source_modes'].reshape(-1,N,order='F').T; rhs=modes@W

def axis(a,p):
 k,v=s['kappa_'+a],s['vel_'+a]; h=s['L'+a]/(len(k)-1); r=s['robin_'+a]; n=len(k)
 M=m._axis(k,v,h).tolil()
 for j,(selfscale,robscale,seamscale) in enumerate(np.asarray(p).reshape(2,3)):
  i,q,t=(0,1,n-1) if j==0 else (n-1,n-2,0)
  inward=k[i]/h**2*selfscale
  seam=(.5*(k[i]+k[t])/h**2+max(v[i] if j==0 else -v[i],0)/h)*seamscale
  M[i,i]=inward+seam+r/h*robscale; M[i,q]=-inward; M[i,t]=-seam
 return M.tocsc()

def pred(p):
 ax,ay=axis('x',p[:6]),axis('y',p[6:]); L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=sparse.eye(N,format='csc')
 return np.array([splu(L+mu*I).solve(rhs)[rows] for mu in s['mu_shifts']])

def scores(e): return (np.sqrt(np.mean(e**2)), np.sqrt(np.mean(e**2,axis=1)).max())

if __name__=='__main__':
 defs=[]
 for a in 'xy':
  k,v=s['kappa_'+a],s['vel_'+a]; h=s['L'+a]/(len(k)-1)
  for j,(i,q) in enumerate([(0,1),(-1,-2)]):
   per=(.5*(k[i]+k[q])+h*max(-v[i] if j==0 else v[i],0))/k[i]
   defs.append({'N':(1,0,0),'R':(1,1,0),'P':(per,0,1),'D':(2,0,0),'B':(2,2,0)})
 fit=[]
 for labels in itertools.product(defs[0],repeat=4):
  p=np.concatenate([d[l] for d,l in zip(defs,labels)]); e=pred(p)-Y
  fit.append((scores(e),''.join(labels),p))
 for sr,lab,p in sorted(fit)[:12]: print('discrete',lab,sr,p.round(5))
 for init in [sorted(fit)[0][2], np.tile([1.,0.,0.],4)]:
  for train in [np.s_[:,:,:],np.s_[:,:,0],np.s_[:,:,1]]:
   sol=least_squares(lambda p:(pred(p)-Y)[train].ravel(),init,ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=200)
   print('continuous',sol.nfev,scores(pred(sol.x)-Y),sol.x.reshape(-1,3).round(7), 'singular',np.linalg.svd(sol.jac,compute_uv=False).round(6))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/wall07.py
```
- **Container Exit Code**: `0` (`3838 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(5.028342347237257e-07, 5.772861057603568e-07) [[ 1.0365127e+00  5.1000000e-06  9.9988150e-01]
 [ 1.0367557e+00 -5.2000000e-06  1.0000778e+00]
 [ 9.9511510e-01  3.3000000e-06  9.9978490e-01]
 [ 1.0089566e+00 -5.7000000e-06  1.0000560e+00]] singular [6.065033e+00 7.151990e-01 3.437670e-01 2.303830e-01 1.768600e-01
 1.230840e-01 4.232600e-02 2.343700e-02 1.674600e-02 7.847000e-03
 4.227000e-03 3.510000e-03]
continuous 3 (9.04622324863916e-07, 1.5999012206948496e-06) [[ 1.0366210e+00  5.7000000e-06  9.9994510e-01]
 [ 1.0365145e+00 -3.9000000e-06  9.9978190e-01]
 [ 9.9473230e-01  3.6000000e-06  9.9928420e-01]
 [ 1.0098872e+00 -1.2300000e-05  1.0004315e+00]] singular [4.33505e+00 5.07660e-01 2.23956e-01 1.53626e-01 1.26410e-01 5.56370e-02
 1.64130e-02 1.40700e-02 4.85200e-03 1.40300e-03 1.23700e-03 6.55000e-04]
continuous 3 (6.511498292755501e-07, 1.0748541756284307e-06) [[ 1.0363792e+00  2.9000000e-06  9.9984270e-01]
 [ 1.0368156e+00 -4.5000000e-06  1.0001677e+00]
 [ 9.9499930e-01  4.3000000e-06  9.9966440e-01]
 [ 1.0090487e+00 -6.5000000e-06  1.0001132e+00]] singular [4.248644e+00 5.007180e-01 1.923830e-01 1.363550e-01 1.116030e-01
 4.665400e-02 1.642600e-02 1.424900e-02 8.068000e-03 5.011000e-03
 1.752000e-03 1.320000e-03]
continuous 20 (5.028342347324881e-07, 5.772860579231377e-07) [[ 1.0365127e+00  5.1000000e-06  9.9988160e-01]
 [ 1.0367557e+00 -5.2000000e-06  1.0000778e+00]
 [ 9.9511510e-01  3.3000000e-06  9.9978490e-01]
 [ 1.0089566e+00 -5.7000000e-06  1.0000560e+00]] singular [6.065033e+00 7.151990e-01 3.437670e-01 2.303830e-01 1.768600e-01
 1.230840e-01 4.232600e-02 2.343700e-02 1.674600e-02 7.847000e-03
 4.227000e-03 3.510000e-03]
continuous 11 (9.046260326362087e-07, 1.5999127028066367e-06) [[ 1.0366210e+00  5.7000000e-06  9.9994500e-01]
 [ 1.0365146e+00 -3.9000000e-06  9.9978200e-01]
 [ 9.9473230e-01  3.6000000e-06  9.9928410e-01]
 [ 1.0098872e+00 -1.2300000e-05  1.0004315e+00]] singular [4.33505e+00 5.07660e-01 2.23956e-01 1.53626e-01 1.26410e-01 5.56370e-02
 1.64130e-02 1.40700e-02 4.85200e-03 1.40300e-03 1.23700e-03 6.55000e-04]
continuous 13 (6.511661919079966e-07, 1.0749021480672362e-06) [[ 1.0363792e+00  2.9000000e-06  9.9984270e-01]
 [ 1.0368156e+00 -4.5000000e-06  1.0001677e+00]
 [ 9.9499930e-01  4.3000000e-06  9.9966440e-01]
 [ 1.0090487e+00 -6.5000000e-06  1.0001132e+00]] singular [4.248644e+00 5.007180e-01 1.923830e-01 1.363550e-01 1.116030e-01
 4.665400e-02 1.642600e-02 1.424900e-02 8.068000e-03 5.012000e-03
 1.752000e-03 1.320000e-03]
```

### Turn 7 — `execute_in_remote_docker` (`45360 ms` LLM)
- **Rationale**: Combine the verified periodic stencil with sweep-fitted nonlinear calibration. Test the assembled artifact on both public pulses, apply the field equation independently to catch axis or ordering mistakes, and compare the optimizer against enumerated box faces to catch active-set or certificate errors.
- **Alternatives Weighed**: `Fit continuous wall coefficients` (rejected: The exact periodic stencil predicts both pulses to calibration precision; continuous coefficients reduce in-sample error but do not improve transfer to the held-out pulse.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
archive = Path('/tmp/hegel_continual_archive')
periodic = (archive / 'beta_m98_v2_c02_batch_03_solver.py').read_text()
nonlinear = (archive / 'beta_m98_v2_c02_batch_06_solver.py').read_text()
forward = periodic[periodic.index('def _axis('):periodic.index('def _calibrate(')]
rest = nonlinear[nonlinear.index('def _rosette_features('):]
Path('/holdout_task/batch_solver.py').write_text('"""Periodic transport tomography with sweep-fitted nonlinear rosette calibration."""\n\nimport numpy as np\nfrom scipy import sparse\nfrom scipy.optimize import root\nfrom scipy.sparse.linalg import splu\n\n\n' + forward + rest)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
cat > /tmp/verify_batch07.py <<'PY'
import importlib.util
import itertools
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
A, last = m._forward(s)
for p in item['campaign_diagnostics']:
    y = m._calibrate(dict(s, y_obs_raw=p['y_pulse_raw']))
    rmse = np.sqrt(np.mean((A @ p['w_pulse'] - y).reshape(len(s['mu_shifts']), -1)**2, axis=1))
    print('pulse RMSE by shift:', rmse)
    assert rmse.max() < 2e-6

r = m.solve_batch(s)
w, u, c = (r[k] for k in ('weights', 'terminal_field', 'kkt_certificate'))
cap, total, ridge = (s[k] for k in ('u_box', 'total_budget', 'source_ridge'))
Z = s['Z_nuisance']
y = m._calibrate(s)
project = lambda d: d-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@d)
D, b = project(A), project(y)
g = D.T@(D@w-b)+ridge*w
J = lambda v: .5*(np.linalg.norm(D@v-b)**2+ridge*(v@v))
lo, hi, nu = (np.asarray(c[k]) for k in ('lam_lower', 'lam_upper', 'nu_sum'))
errors = [max(0.,-w.min(),w.max()-cap), abs(w.sum()-total), np.max(abs(np.asarray(c['weights'])-w)), np.max(abs(g-nu-lo+hi)), np.max(abs(lo*w)), np.max(abs(hi*(cap-w))), abs(c['primal_objective']-J(w)), abs(J(w)-c['dual_lower_bound'])]
assert min(lo.min(), hi.min()) >= 0 and max(errors) < 1e-9, errors

# Independent stencil application, including seams and corners, checks field ordering as well.
Lu = s['V_reaction_2d']*u + s['mu_shifts'][-1]*u
for a, ax in enumerate('xy'):
    k, v = s['kappa_'+ax], s['vel_'+ax]
    h = s['L'+ax]/(len(k)-1)
    shape = (-1, 1) if a == 0 else (1, -1)
    for direction in (-1, 1):
        rate = ((k+np.roll(k,-direction))/(2*h*h)+np.maximum(-direction*v,0)/h).reshape(shape)
        Lu += rate*(u-np.roll(u,-direction,axis=a))
field_error = np.max(abs(Lu-np.tensordot(w,s['source_modes'],axes=1)))
assert field_error < 1e-8, field_error

# Brute-force box faces independently certify the optimizer on the actual design.
H, q = D.T@D+ridge*np.eye(w.size), D.T@b
worst = 0.
for budget in (total, 0., cap, cap*(w.size-.35), cap*w.size):
    wc, nc = m._box_budget_qp(H,q,cap,budget)
    best = np.inf
    for pattern in itertools.product((-1,0,1),repeat=w.size):
        state = np.asarray(pattern)
        v = np.where(state == 1, cap, 0.)
        free = np.flatnonzero(state == 0)
        left = budget-v.sum()
        if left < -1e-12 or left > cap*len(free)+1e-12:
            continue
        if free.size:
            n = free.size
            M = np.ones((n+1,n+1))
            M[:n,:n] = H[np.ix_(free,free)]
            M[-1,-1] = 0
            v[free] = np.linalg.solve(M,np.r_[q[free]-H[free]@v,left])[:-1]
        if v.min() >= -1e-10 and v.max() <= cap+1e-10:
            best = min(best, J(v))
    err = max(abs(J(wc)-best),abs(wc.sum()-budget),-wc.min(),wc.max()-cap)
    assert err < 1e-9, (budget,err)
    worst = max(worst,err)
print('max certificate error:', max(errors), 'max field equation error:', field_error, 'max enumerated QP error:', worst)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_batch07.py
```
- **Container Exit Code**: `0` (`1099 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
pulse RMSE by shift: [7.84129786e-07 6.45380963e-07 5.95989955e-07]
pulse RMSE by shift: [7.86148788e-07 6.50546698e-07 6.06322683e-07]
max certificate error: 3.3306690738754696e-16 max field equation error: 3.9968028886505635e-15 max enumerated QP error: 3.552713678800501e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6fe9c7376c73f67a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6fe9c737 (directory).
```

### Turn 8 — `submit_completed_task` (`15075 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema validation, an independent terminal-field equation check, and comparison against enumerated box faces at five budgets. The host also verified the field, weights, and certificate on the same artifact.
- **Alternatives Weighed**: `Fit continuous wall coefficients` (rejected: The exact periodic stencil predicted both public pulses more accurately than the continuous fits predicted a held-out pulse.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5616'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, and the worst tracer-pulse RMSE across shifts was 7.86e-7. Independent checks verified the terminal field and KKT certificate.`

### Turn 9 — `execute_in_remote_docker` (`46355 ms` LLM)
- **Rationale**: Check whether the current rosette fit hides a weak bank or unnecessary terms by comparing calibration challengers on held-out sweeps and known-source pulses. Inspect the projected source spectrum to detect nuisance–source overlap that would make the recovered weights unstable.
- **Alternatives Weighed**: `Run relay-mode or within-cohort classification checks` (rejected: This is a steady linear transport inverse problem with no relay state or class labels; the relevant nuisance risk is loss of source identifiability after projection.)

```bash
set -e
cat > /tmp/audit_batch07.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import root
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
S, _ = m._forward(s)
x = s['sensor_calibration_sweeps']['ref_rosette']
y = s['sensor_calibration_sweeps']['raw_rosette']
D = m._rosette_features(x)
# The observed sparse structure has only self cubic, cyclic-neighbor affine,
# and a product with the next channel. All coefficients are fitted in each fold.
cols = [[0, 1+j, 1+(j+1)%4, 1+(j-1)%4, 5+j, 9+j] for j in range(4)]
Dplus = np.column_stack((D, x*x, x[:,0]*x[:,2], x[:,1]*x[:,3]))
truth = [S @ p['w_pulse'] for p in item['campaign_diagnostics']]
for model in ('current', 'channel_sparse', 'extra_quadratic'):
    held = []
    pulse = []
    for test in np.array_split(np.random.default_rng(42).permutation(len(x)), 5):
        train = np.setdiff1d(np.arange(len(x)), test)
        # Detect contamination without looking at the held-out rows.
        e = abs(D[train] @ np.linalg.lstsq(D[train], y[train], rcond=None)[0]-y[train]).max(1)
        keep = train[e <= 5*np.median(e)]
        E = Dplus if model == 'extra_quadratic' else D
        c = np.zeros((E.shape[1], 4))
        for j in range(4):
            ix = cols[j] if model == 'channel_sparse' else np.arange(E.shape[1])
            c[ix,j] = np.linalg.lstsq(E[keep][:,ix], y[keep,j], rcond=None)[0]
        held.append(E[test] @ c - y[test])
        def f(z):
            a = m._rosette_features(z[None,:])[0]
            if model == 'extra_quadratic':
                a = np.r_[a, z*z, z[0]*z[2], z[1]*z[3]]
            return a @ c
        for p, target in zip(item['campaign_diagnostics'],truth):
            raw = p['y_pulse_raw'].reshape(-1,4)
            guesses = np.linalg.solve(c[1:5].T, (raw-c[0]).T).T
            inv = [root(lambda z: f(z)-r, g, tol=1e-10) for r,g in zip(raw,guesses)]
            assert max(np.max(abs(v.fun)) for v in inv) < 1e-9
            err = (np.array([v.x for v in inv]).ravel()-target).reshape(len(s['mu_shifts']),-1,4)
            pulse.append(np.sqrt(np.mean(err**2,axis=1)))
    err = abs(np.vstack(held))
    print(model,'worst bank OOF p95',np.quantile(err,.95,axis=0).max(),'worst pulse x shift x bank RMSE',np.max(pulse))
Z = s['Z_nuisance']
project = lambda a: a-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@a)
# Restrict sensitivity to perturbations that preserve the equality budget.
B = np.eye(S.shape[1])[:,:-1]; B[-1] = -1
before = np.linalg.svd(S@B,compute_uv=False)
after = np.linalg.svd(project(S)@B,compute_uv=False)
print('projected and unprojected source tangent singular values:',after,before)
print('pulse groups:',len(truth),'shift groups:',len(s['mu_shifts']),'nuisance columns:',Z.shape[1])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_batch07.py
```
- **Container Exit Code**: `0` (`763 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
current worst bank OOF p95 2.810374735044973e-05 worst pulse x shift x bank RMSE 2.7104190448119043e-06
channel_sparse worst bank OOF p95 2.561070212314663e-05 worst pulse x shift x bank RMSE 3.287254964643526e-06
extra_quadratic worst bank OOF p95 2.9161692047408038e-05 worst pulse x shift x bank RMSE 7.366464000400884e-06
projected and unprojected source tangent singular values: [0.73968842 0.50413032 0.43238957 0.21578174 0.09735158 0.05631995
 0.03639464] [0.76035915 0.5109409  0.4360769  0.21720921 0.09784817 0.05730135
 0.03677156]
pulse groups: 2 shift groups: 3 nuisance columns: 3

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6fe9c7376c73f67a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6fe9c737 (directory).
```

### Turn 10 — `submit_completed_task` (`19389 ms` LLM)
- **Rationale**: Keep the verified artifact unchanged. The focused audit found no calibration or source-identifiability bottleneck that warrants replacing it, and the host receipt covers the saved solver.
- **Alternatives Weighed**: `Replace the current rosette calibration with a sparse or expanded quadratic model` (rejected: Neither alternative improved the worst held-out tracer slice, and nuisance projection retained all seven budget-preserving source directions.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5616'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, the worst tracer-pulse RMSE across shifts was 7.86e-7, and independent checks verified the terminal field and KKT certificate.`

### Turn 11 — `execute_in_remote_docker` (`36636 ms` LLM)
- **Rationale**: Test whether rational terms in individual channels or neighboring-channel sums and differences explain the remaining calibration error. Revisit the calibration only if a challenger improves the worst tracer slice and remains stable on held-out sweeps; otherwise keep the verified artifact unchanged.
- **Alternatives Weighed**: `Repeat the boundary and nuisance audits` (rejected: The tracer pulses already resolve the periodic stencil, and the previous audit showed that nuisance projection preserves all seven budget-preserving source directions. There are no class labels or relay states.)

```bash
set -e
cat > /tmp/rational_batch07.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
cal = s['sensor_calibration_sweeps']
x, y = cal['ref_rosette'], cal['raw_rosette']
D = m._rosette_features(x)
S, _ = m._forward(s)
pulses = [(np.asarray(S @ p['w_pulse']).reshape(-1,4), p['y_pulse_raw'].reshape(-1,4)) for p in item['campaign_diagnostics']]

def q2(z, kind):
    if kind == 'self':
        return z*z
    sign = 1 if kind == 'neighbor_sum' else -1
    return (z + sign*np.roll(z,-1,axis=1))**2

folds = np.array_split(np.random.default_rng(42).permutation(len(x)),5)
for kind in ('polynomial','self','neighbor_sum','neighbor_difference'):
    held, slices, denominator_min = [], [], np.inf
    for test in folds:
        train = np.setdiff1d(np.arange(len(x)),test)
        c = np.linalg.lstsq(D[train],y[train],rcond=None)[0]
        e = np.max(abs(D[train]@c-y[train]),axis=1)
        keep = train[e <= 5*np.median(e)]
        a = np.zeros(4)
        if kind == 'polynomial':
            c = np.linalg.lstsq(D[keep],y[keep],rcond=None)[0]
        else:
            q = q2(x[keep],kind)
            for j in range(4):
                fit = np.linalg.lstsq(np.column_stack((D[keep],-y[keep,j]*q[:,j])),y[keep,j],rcond=None)[0]
                c[:,j], a[j] = fit[:-1], fit[-1]
        def predict(z):
            global denominator_min
            den = np.ones_like(z) if kind == 'polynomial' else 1+q2(z,kind)*a
            denominator_min = min(denominator_min,den.min())
            return (m._rosette_features(z)@c)/den
        held.append(abs(predict(x[test])-y[test]))
        for z, raw in pulses:
            residual = (predict(z)-raw).reshape(len(s['mu_shifts']),-1,4)
            slices.append(np.max(np.sqrt(np.mean(residual**2,axis=1))))
    print(kind, 'worst bank sweep p95:', np.quantile(np.vstack(held),.95,axis=0).max(), 'worst raw pulse/shift/bank RMSE:',max(slices),'min tested denominator:',denominator_min)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rational_batch07.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`757 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
polynomial worst bank sweep p95: 2.8103747347096855e-05 worst raw pulse/shift/bank RMSE: 3.0300972361842175e-06 min tested denominator: 1.0
self worst bank sweep p95: 3.1976688545309866e-05 worst raw pulse/shift/bank RMSE: 3.330518746413553e-06 min tested denominator: 0.9999841244710694
neighbor_sum worst bank sweep p95: 3.079267198980994e-05 worst raw pulse/shift/bank RMSE: 3.7941317993789723e-06 min tested denominator: 0.9999824244374524
neighbor_difference worst bank sweep p95: 3.0263252524620458e-05 worst raw pulse/shift/bank RMSE: 3.484977254224528e-06 min tested denominator: 0.9999962742601934
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6fe9c7376c73f67a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6fe9c737 (directory).
```

### Turn 12 — `submit_completed_task` (`19221 ms` LLM)
- **Rationale**: Retain the verified solver. Nuisance projection preserved all seven budget-preserving source directions, and the unchanged artifact passed isolated schema validation and independent transport and optimality checks.
- **Alternatives Weighed**: `Add rational corrections based on individual channels or adjacent-channel sums and differences` (rejected: All three challengers worsened both the held-out sweep error and the worst raw tracer-pulse slice.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, the worst tracer-pulse RMSE across shifts was 7.86e-7, and independent checks verified the terminal field and KKT certificate.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M98-Blind-Mismatch-Factored-Lineage-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
