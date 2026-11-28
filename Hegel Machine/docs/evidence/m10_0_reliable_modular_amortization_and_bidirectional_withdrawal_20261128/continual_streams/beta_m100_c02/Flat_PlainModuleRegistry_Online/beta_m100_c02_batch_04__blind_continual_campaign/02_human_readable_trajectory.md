# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m100_c02_batch_04__blind_continual_campaign__downstream_module`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World` |
| **Domain** | `coupled_2d_transport_reaction_box_simplex_kkt` |
| **Wave / Wormhole Family** | `m10_0_reliable_modular_amortization_and_bidirectional_withdrawal` / `m100_beta` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `11 / 16` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `286,414` (`275,309` / `102,784` / `11,105`) |
| **Total Episode Wall Time (s)** | `356.65s` |

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

### Turn 1 — `execute_in_remote_docker` (`15482 ms` LLM)
- **Rationale**: Inspect the actual calibrated interface and available boundary evidence before choosing an operator; a coordinate or schema mismatch would change both assembly and validation.

```bash
cd /holdout_task && ls -l && cat batch_solver.py _verified_upstream.py _upstream_param_payload.json validate_public_schema.py public_asset_index.json && ls -lh /tmp/hegel_continual_archive && python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_batch.pkl','rb')); print('items',len(x))
for k,v in x[0].items():
 print(k,type(v).__name__)
 if isinstance(v,dict):
  for a,b in v.items(): print(' ',a,type(b).__name__,getattr(b,'shape',None),b if np.isscalar(b) else (list(b) if isinstance(b,dict) else ''))
 elif isinstance(v,list): print(' length',len(v),'first keys',list(v[0]) if v and isinstance(v[0],dict) else '')
PY
```
- **Container Exit Code**: `0` (`409 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"bank_id": "ndarray",
                "raw_readout": "ndarray",
                "ref_signal": "ndarray"
              }
            },
            "sensor_ij": {
              "dtype": "int64",
              "shape": [
                32,
                2
              ],
              "type": "ndarray"
            },
            "source_modes": {
              "dtype": "float64",
              "shape": [
                8,
                14,
                12
              ],
              "type": "ndarray"
            },
            "source_ridge": 0.04,
            "total_budget": 1.4,
            "u_box": 0.42,
            "vel_x": {
              "dtype": "float64",
              "shape": [
                14
              ],
              "type": "ndarray"
            },
            "vel_y": {
              "dtype": "float64",
              "shape": [
                12
              ],
              "type": "ndarray"
            },
            "y_obs_raw": {
              "dtype": "float64",
              "shape": [
                96
              ],
              "type": "ndarray"
            }
          }
        },
        "len": 1,
        "type": "list"
      }
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 1091
    }
  },
  "schema": "hegel_public_asset_index_v1"
}total 44K
-r--r--r-- 1 root root  19K Nov 28 03:43 archive_apparatus_log.json
-rw-r--r-- 1 root root  675 Nov 28 03:43 archive_manifest.json
-rw-r--r-- 1 root root 4.7K Nov 28 03:43 beta_m100_c02_batch_01_solver.py
-rw-r--r-- 1 root root 4.7K Nov 28 03:43 beta_m100_c02_s00_solver.py
-rw-r--r-- 1 root root  127 Nov 28 03:43 pre_llm_gate_diagnostic.json
items 1
study dict
  Lx float None 2.4
  Ly float None 2.0
  nx int None 14
  ny int None 12
  kappa_x ndarray (14,) 
  kappa_y ndarray (12,) 
  vel_x ndarray (14,) 
  vel_y ndarray (12,) 
  robin_x float None 1.0141673808964704
  robin_y float None 1.1585219318384303
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
campaign_diagnostics list
 length 3 first keys ['w_pulse', 'y_pulse_raw']
```

### Turn 2 — `execute_in_remote_docker` (`14468 ms` LLM)
- **Rationale**: Read the calibrated interface and archived wall grammar before fitting the diagnostic pulses. If archival and current wall states disagree, use the current pulse evidence to choose the downstream operator.
- **Alternatives Weighed**: `Use the archived operator without testing it` (rejected: Prior campaigns can have different wall states; the current pulses can discriminate between them.)

```bash
cd /holdout_task && tail -32 _verified_upstream.py && cat validate_public_schema.py && cd /tmp/hegel_continual_archive && cat archive_manifest.json pre_llm_gate_diagnostic.json && cat beta_m100_c02_batch_01_solver.py && python3 - <<'PY'
import json
x=json.load(open('archive_apparatus_log.json')); print('log structure:',type(x), list(x) if isinstance(x,dict) else len(x));
for k,v in (x.items() if isinstance(x,dict) else enumerate(x)):
 print(k,str(v)[:950])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling box simplex quadratic KKT'
```
- **Container Exit Code**: `0` (`328 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.1303, -0.30011, 0.01625, -0.57997, -0.44758, -0.26644], [-0.69003, -0.37988, -0.72053, -0.06611, 0.43059, 0.04933, -0.2083, -0.36372], [-0.50708, -0.29458, -0.84685, 0
3 {'completed_campaign_index': 3, 'completed_campaign_short_name': 'beta_m100_c02_batch_02', 'facility_telemetry_trace': [[-0.7816, -0.39469, -0.64361, -0.82093, 0.83534, -0.6688, 0.58453, 2.66264], [-0.81723, -0.4221, -0.87148, -0.82507, 0.59782, -0.07715, 0.69471, 2.36737], [-0.85125, -0.29329, -0.64657, -0.81331, 0.28372, 0.58597, 0.80714, 2.37861], [-0.82267, -0.37433, -0.06001, -0.71287, -0.16628, 0.85769, 0.92183, 2.42998], [-0.70407, -0.39659, 0.56787, -0.62738, -0.63648, 0.64433, 0.83311, 2.51734], [-0.51512, -0.29556, 0.84729, -0.40337, -0.84855, 0.09224, 0.74263, 2.53696], [-0.30524, -0.30115, 0.61851, -0.2476, -0.76109, -0.48462, 0.60567, 2.50321], [-0.10271, -0.38401, -0.03753, 0.00612, -0.67509, -0.87859, 0.43203, 2.41987], [0.13789, -0.36886, -0.58558, 0.19803, -0.27683, -0.67447, 0.25286, 2.4443], [0.28174, -0.34729, -0.83202, 0.38186, 0.13199, -0.11567, -0.00318, 2.41272], [0.45171, -0.26547, -0.63949, 0.59663, 0.54316, 0.
4 {'completed_campaign_index': 4, 'completed_campaign_short_name': 'beta_m100_c02_batch_03', 'facility_telemetry_trace': [[-0.24626, 2.50359, -0.37308, -0.87533, 0.83152, -0.19964, -0.80421, -0.40314], [-0.05086, 2.59738, -0.77707, -0.7417, 0.63401, 0.47049, -0.86958, -0.40649], [0.16724, 2.51288, -0.76352, -0.57696, 0.37906, 0.81554, -0.86913, -0.45468], [0.40011, 2.49938, -0.31905, -0.43889, -0.14408, 0.70361, -0.77167, -0.39508], [0.54113, 2.58549, 0.31388, -0.23083, -0.52056, 0.2464, -0.67125, -0.43909], [0.74491, 2.63069, 0.76039, 0.03644, -0.7906, -0.46588, -0.52081, -0.35051], [0.82574, 2.48278, 0.78945, 0.25922, -0.84635, -0.7951, -0.32498, -0.38147], [0.82412, 2.52464, 0.33193, 0.47739, -0.63432, -0.71961, -0.13416, -0.42365], [0.84601, 2.46833, -0.36982, 0.56707, -0.30879, -0.19306, 0.12852, -0.39273], [0.76731, 2.69742, -0.82237, 0.77764, 0.14586, 0.42425, 0.3797, -0.30955], [0.64567, 2.60283, -0.78326, 0.77724, 0.55261, 0.8463
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 3 — `execute_in_remote_docker` (`18144 ms` LLM)
- **Rationale**: Recover the archived wall formulas from the truncated output and verify that the upstream transducer calibrates the pulse observations. Pulse excitation and sensor coverage determine which boundary parameters can be fit and tested separately.

```bash
cd /tmp/hegel_continual_archive && cat archive_manifest.json pre_llm_gate_diagnostic.json && sed -n '1,150p' beta_m100_c02_batch_01_solver.py && diff -u beta_m100_c02_s00_solver.py beta_m100_c02_batch_01_solver.py | head -75; cd /holdout_task && python3 - <<'PY'
import pickle,json,numpy as np
from _verified_upstream import calibrate_upstream
x=pickle.load(open('public_batch.pkl','rb'))[0]; s=calibrate_upstream(dict(x['study'],reference_tracer_pulses=x['campaign_diagnostics'])); print('sensors',s['sensor_ij'].T); print('nuisance column norms',np.linalg.norm(s['Z_nuisance'],axis=0)); print('pulses',[(p['w_pulse'].tolist(),np.linalg.norm(p['y_pulse_raw']),[np.linalg.norm(c) for c in np.split(p['y_pulse_raw'],3)]) for p in s['reference_tracer_pulses']]); print('telemetry means',s['aux_manifold_telemetry'].mean(0))
for t in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print({k:v for k,v in t.items() if k!='facility_telemetry_trace'}, np.asarray(t['facility_telemetry_trace']).mean(0).round(3))
PY
```
- **Container Exit Code**: `0` (`444 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
+17,15 @@
     center[1:-1] = -west[:-1] - east[1:]
     east[0] = -kappa[0] / h**2
     west[-1] = -kappa[-1] / h**2
-    center[0] = -east[0] + robin / h
-    center[-1] = -west[-1] + robin / h
+    center[0] = -east[0]
+    center[-1] = -west[-1]
     return sparse.diags((west, center, east), (-1, 0, 1), format='csc')
 
 
 def chamber_response(study):
     nx, ny = int(study['nx']), int(study['ny'])
-    ax = axis_operator(study['kappa_x'], study['vel_x'], study['Lx']/(nx-1), study['robin_x'])
-    ay = axis_operator(study['kappa_y'], study['vel_y'], study['Ly']/(ny-1), study['robin_y'])
+    ax = axis_operator(study['kappa_x'], study['vel_x'], study['Lx']/(nx-1))
+    ay = axis_operator(study['kappa_y'], study['vel_y'], study['Ly']/(ny-1))
     reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
     operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                 + sparse.kron(ay, sparse.eye(nx), format='csc')
sensors [[10  8  6  0 12  6  2  8  8 10  0 10  2  4  0  2  6  2  0 12  8 10  8 12
   6 10  6  0 12  0  4 12]
 [ 4  8 10  6  8  4  4  0  2  2  8  8  6 10  2 10  2  0  0  2 10  0  4 10
   8 10  0 10  0  4  2  6]]
nuisance column norms [10.68800665 10.01123258  9.41570978]
pulses [([1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], 1.9981630774058567, [1.70446732310925, 0.8876589386395008, 0.547273639977504]), ([0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], 1.7979883051453844, [1.4663672249188282, 0.864386177220083, 0.5791076270904036]), ([0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0], 1.9305583714299888, [1.6394870414098632, 0.8659487694706789, 0.5378388189781037])]
telemetry means [-0.01194178 -0.18928104 -0.10944447 -0.41620018]
{'completed_campaign_index': 0, 'completed_campaign_short_name': 'facility_init'} [-0.013 -0.42   0.002 -0.002 -0.012 -0.005  0.003 -0.354]
{'completed_campaign_index': 1, 'completed_campaign_short_name': 'beta_m100_c02_s00'} [-2.000e-03 -3.830e-01  1.000e-03  1.300e-02  2.000e-03 -3.000e-03
 -9.000e-03  2.433e+00]
{'completed_campaign_index': 2, 'completed_campaign_short_name': 'beta_m100_c02_batch_01'} [ 0.003 -0.414  0.003  0.011  0.004  0.002 -0.003 -0.343]
{'completed_campaign_index': 3, 'completed_campaign_short_name': 'beta_m100_c02_batch_02'} [ 3.000e-03 -3.640e-01 -1.200e-02  6.000e-03 -4.000e-03 -5.000e-03
 -1.000e-03  2.459e+00]
{'completed_campaign_index': 4, 'completed_campaign_short_name': 'beta_m100_c02_batch_03'} [ 6.000e-03  2.555e+00  3.000e-03  1.000e-03 -5.000e-03  2.000e-03
 -2.000e-03 -3.810e-01]
```

### Turn 4 — `execute_in_remote_docker` (`38545 ms` LLM)
- **Rationale**: Compare baseline and extended endpoint stencils on calibrated pulses, leaving each pulse out of the fit. If a larger stencil only improves training error, keep the simpler one; if it improves held-out error, inspect which endpoint terms account for the gain.
- **Alternatives Weighed**: `Fit wall coefficients to the inverse study observation` (rejected: Unknown source weights and nuisance effects can mask the wrong wall operator; known tracer inputs separate these effects.)

```bash
cat > /tmp/fit_walls.py <<'PY'
import sys,pickle,numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=calibrate_upstream(dict(item['study'],reference_tracer_pulses=item['campaign_diagnostics']))
nx,ny=s['nx'],s['ny']; ix=sp.eye(nx,format='csc'); iy=sp.eye(ny,format='csc'); eye=sp.eye(nx*ny,format='csc')
ijk=s['sensor_ij']; sensors=ijk[:,0]+nx*ijk[:,1]
modes=np.stack([m.ravel(order='F') for m in s['source_modes']],axis=1)
ps=s['reference_tracer_pulses']; rhs=modes@np.stack([p['w_pulse'] for p in ps],axis=1); target=np.column_stack([p['y_pulse_raw'] for p in ps]); z=s['Z_nuisance']; proj=z@np.linalg.pinv(z)

def axis(q,suffix):
    k=s['kappa_'+suffix]; v=s['vel_'+suffix]; h=s['L'+suffix]/(len(k)-1); rob=s['robin_'+suffix]
    faces=(k[:-1]+k[1:])/(2*h*h); lo=-faces[:-1]-np.maximum(v[1:-1],0)/h; hi=-faces[1:]-np.maximum(-v[1:-1],0)/h
    ds=q[:2]*k[[0,-1]]/h**2; leaks=q[2:4]*rob/h; wraps=q[4:]*k[[0,-1]]/h**2
    a=sp.diags((np.r_[lo,-ds[1]],np.r_[ds[0]+leaks[0]+wraps[0],-lo-hi,ds[1]+leaks[1]+wraps[1]],np.r_[-ds[0],hi]),(-1,0,1),format='lil'); a[0,-1]=-wraps[0]; a[-1,0]=-wraps[1]; return a.tocsc()

def predict(q,rhs=rhs):
    a=sp.kron(iy,axis(q[:6],'x'),format='csc')+sp.kron(axis(q[6:],'y'),ix,format='csc')+sp.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    return np.concatenate([splu(a+mu*eye).solve(rhs)[sensors] for mu in s['mu_shifts']])

def report(name,e):
    print(name,'pulse rms',np.sqrt(np.mean(e**2,axis=0)).round(9),'shift rms',np.sqrt(np.mean(e.reshape(3,-1)**2,axis=1)).round(9),'nuisance fraction',round(np.linalg.norm(proj@e)/max(np.linalg.norm(e),1e-15),5),flush=True)

if __name__=='__main__':
    for x in (0,1):
      for y in (0,1):
        report(f'Robin {x,y}',predict(np.array([1,1,x,x,0,0,1,1,y,y,0,0]))-target)
    b=np.array([1.,1.,0.4,0.4,0,0,1.,1.,0.4,0.4,0,0]); inds=[[2,3,8,9],[0,1,2,3,6,7,8,9],list(range(12))]
    for ids in inds:
      for nuisance in (False,True):
        for hold in (-1,0,1,2):
          keep=[j for j in range(3) if j!=hold]
          def expand(v):
            q=b.copy(); q[ids]=v; return q
          def err(v):
            e=predict(expand(v))-target
            return e-proj@e if nuisance else e
          f=least_squares(lambda v: err(v)[:,keep].ravel(),b[ids],bounds=(-0.25,8),ftol=1e-12,gtol=1e-12,xtol=1e-12)
          print('fit',len(ids),nuisance,hold,expand(f.x).round(7))
          report(' residual',err(f.x))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/fit_walls.py
```
- **Container Exit Code**: `0` (`8459 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
03e+00  1.0356192e+00 -2.1200000e-05  2.6100000e-05
  1.1371543e+00  1.0806790e+00  1.0000238e+00  1.0000389e+00
 -1.7000000e-06  2.0000000e-07  4.0000000e-07  2.0000000e-07]
 residual pulse rms [8.60e-07 8.15e-07 8.27e-07] shift rms [1.035e-06 7.520e-07 6.720e-07] nuisance fraction 0.17611
fit 12 False 0 [ 1.0357424e+00  1.0354846e+00 -2.0100000e-05  2.5000000e-05
  1.1372636e+00  1.0805076e+00  1.0000282e+00  1.0000220e+00
 -1.5000000e-06 -8.0000000e-07  4.0000000e-07  3.0000000e-07]
 residual pulse rms [1.369e-06 8.190e-07 7.820e-07] shift rms [1.386e-06 8.620e-07 7.020e-07] nuisance fraction 0.18416
fit 12 False 1 [ 1.0354736e+00  1.0356740e+00 -2.3300000e-05  2.8300000e-05
  1.1370961e+00  1.0807224e+00  1.0000268e+00  1.0000370e+00
 -2.3000000e-06  6.0000000e-07 -6.0000000e-07 -1.5000000e-06]
 residual pulse rms [8.460e-07 1.057e-06 8.290e-07] shift rms [1.16e-06 8.19e-07 7.10e-07] nuisance fraction 0.15746
fit 12 False 2 [ 1.0354637e+00  1.0356571e+00 -1.5200000e-05  1.9700000e-05
  1.1372960e+00  1.0808947e+00  1.0000279e+00  1.0000654e+00
 -5.0000000e-07  2.0000000e-07  1.0000000e-07 -5.0000000e-07]
 residual pulse rms [8.030e-07 8.070e-07 1.662e-06] shift rms [1.527e-06 1.031e-06 8.160e-07] nuisance fraction 0.21157
fit 12 True -1 [ 1.0355325e+00  1.0356186e+00 -2.1400000e-05  2.5700000e-05
  1.1371465e+00  1.0806798e+00  1.0000259e+00  1.0000403e+00
 -1.5000000e-06  2.0000000e-07  3.0000000e-07  2.0000000e-07]
 residual pulse rms [8.45e-07 8.02e-07 8.14e-07] shift rms [1.003e-06 7.500e-07 6.720e-07] nuisance fraction 0.0
fit 12 True 0 [ 1.0357884e+00  1.0354429e+00 -1.8400000e-05  2.1400000e-05
  1.1372960e+00  1.0804759e+00  1.0000334e+00  1.0000281e+00
 -8.0000000e-07 -6.0000000e-07  2.0000000e-07  0.0000000e+00]
 residual pulse rms [1.557e-06 8.040e-07 7.580e-07] shift rms [1.506e-06 9.060e-07 7.450e-07] nuisance fraction 0.0
fit 12 True 1 [ 1.0354586e+00  1.0356772e+00 -2.3800000e-05  2.8500000e-05
  1.1370804e+00  1.0807232e+00  1.0000292e+00  1.0000379e+00
 -2.1000000e-06  7.0000000e-07 -6.0000000e-07 -1.8000000e-06]
 residual pulse rms [8.280e-07 1.081e-06 8.170e-07] shift rms [1.147e-06 8.380e-07 7.100e-07] nuisance fraction 0.0
fit 12 True 2 [ 1.0354580e+00  1.0356480e+00 -1.4000000e-05  1.8000000e-05
  1.1373168e+00  1.0809057e+00  1.0000296e+00  1.0000633e+00
 -3.0000000e-07  0.0000000e+00  0.0000000e+00 -5.0000000e-07]
 residual pulse rms [7.810e-07 7.880e-07 1.832e-06] shift rms [1.625e-06 1.092e-06 8.690e-07] nuisance fraction 0.0
```

### Turn 5 — `execute_in_remote_docker` (`34074 ms` LLM)
- **Rationale**: Compare reduced boundary formulas against the calibrated pulse holdouts. This determines whether the two wrap rates follow a shared physical scaling or need to remain separate.
- **Alternatives Weighed**: `Deploy the unrestricted 12-parameter wall fit` (rejected: The fitted leakage and y-axis corrections are near zero; test whether removing them preserves held-out accuracy.)
- **ATMS NOGOOD DECLARED**: Refuted `Local endpoint leakage and adjacent-node diffusion suffice for the chamber boundary.` | Witness: `The closed-wall baseline reaches 0.0464 worst-pulse RMSE, and allowing local leakage and diffusion still leaves percent-scale errors. Adding opposite-end coupling reduces worst held-out RMSE to 1.66e-6.` | Blanket Diagnosis: `The local wall stencil omitted transport between the two x endpoints; fitted y-wrap and leakage terms are near zero.` | Sublated Pivot: `Test a conservative x-only wrap stencil and retain shared coefficients only if held-out pulse accuracy is preserved.`

```bash
cat > /tmp/compact_walls.py <<'PY'
from fit_walls import *
k=s['kappa_x']; v=s['vel_x']; h=s['Lx']/(len(k)-1)
print('x h, robin, kappa, velocity',h,s['robin_x'],k,v)
print('y h, robin, kappa, velocity',s['Ly']/(ny-1),s['robin_y'],s['kappa_y'],s['vel_y'])
p=np.array([1.03554,1.03562,0,0,1.1371543,1.080679,1,1,0,0,0,0]); print('endpoint rates',p[:2]*k[[0,-1]]/h**2,p[4:6]*k[[0,-1]]/h**2,'increments', (p[:2]-1)*k[[0,-1]]/h**2,'wrap diffusivity',p[4:6]*k[[0,-1]])
base=np.r_[1.,1.,0,0,0,0,1.,1.,0,0,0,0]
forms={
 'x-only independent':(4,lambda t: [t[0],t[1],0,0,t[2],t[3]]),
 'x shared neighbor multiplier':(3,lambda t: [t[0],t[0],0,0,t[1],t[2]]),
 'x shared neighbor and wrap multipliers':(2,lambda t: [t[0],t[0],0,0,t[1],t[1]]),
 'x shared neighbor multiplier, equal absolute wraps':(2,lambda t: [t[0],t[0],0,0,*(t[1]*k[[0,-1]].mean()/k[[0,-1]])]),
 'x shared multiplier, wrap excess proportional to robin':(2,lambda t: [t[0],t[0],0,0,*(1+t[1]*s['robin_x']*h/k[[0,-1]])]),
 'x shared multiplier, wrap scaled face neighbors':(2,lambda t: [t[0],t[0],0,0,*(t[1]*(k[[0,-1]]+k[[1,-2]])/(2*k[[0,-1]]))]),
}
for name,(n,form) in forms.items():
 print(name)
 for hold in [-1,0,1,2]:
  mask=np.arange(3)!=hold
  def err(t): return predict(np.r_[form(t),base[6:]])-target
  f=least_squares(lambda t:err(t)[:,mask].ravel(),np.ones(n),gtol=1e-12,ftol=1e-12,xtol=1e-12)
  e=err(f.x)
  print(hold,np.round(f.x,9),np.sqrt(np.mean(e**2,axis=0)).round(9))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/compact_walls.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -30
```
- **Container Exit Code**: `0` (`2295 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
92 1.08049298] [1.722e-06 8.700e-07 8.600e-07]
1 [1.03556638 1.03558623 1.1372274  1.08070171] [9.73e-07 9.13e-07 9.24e-07]
2 [1.03557752 1.03555274 1.13740821 1.08078725] [9.150e-07 8.630e-07 2.155e-06]
x shared neighbor multiplier
-1 [1.03558136 1.1372513  1.08070954] [9.81e-07 8.93e-07 9.26e-07]
0 [1.03559524 1.13724365 1.08070937] [1.086e-06 8.810e-07 9.070e-07]
1 [1.0355765  1.13723404 1.08069098] [9.75e-07 9.18e-07 9.23e-07]
2 [1.0355658  1.13740635 1.08081285] [9.160e-07 8.630e-07 2.016e-06]
x shared neighbor and wrap multipliers
-1 [1.06782782 1.249258  ] [0.00060684 0.00025191 0.00133424]
0 [1.09352669 1.23546108] [0.00120158 0.00069296 0.00100795]
1 [1.07246662 1.26213896] [0.00071766 0.00033873 0.00126977]
2 [1.05094449 1.17344151] [0.00025102 0.000175   0.00161975]
x shared neighbor multiplier, equal absolute wraps
-1 [1.06782782 1.249258  ] [0.00060684 0.00025191 0.00133424]
0 [1.09352669 1.23546108] [0.00120158 0.00069296 0.00100795]
1 [1.07246662 1.26213896] [0.00071766 0.00033873 0.00126977]
2 [1.05094449 1.17344151] [0.00025102 0.000175   0.00161975]
x shared multiplier, wrap excess proportional to robin
-1 [1.06782782 0.62717763] [0.00060684 0.00025191 0.00133424]
0 [1.09352665 0.59246294] [0.00120158 0.00069296 0.00100795]
1 [1.07246662 0.65958928] [0.00071766 0.00033873 0.00126977]
2 [1.05094447 0.43641331] [0.00025102 0.000175   0.00161974]
x shared multiplier, wrap scaled face neighbors
-1 [1.06782781 1.20636142] [0.00060684 0.00025191 0.00133424]
0 [1.09352666 1.19303791] [0.00120158 0.00069296 0.00100795]
1 [1.07246661 1.21879975] [0.00071766 0.00033873 0.00126977]
2 [1.05094449 1.13314793] [0.00025102 0.00017499 0.00161975]
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
```

### Turn 6 — `execute_in_remote_docker` (`54583 ms` LLM)
- **Rationale**: Implement the parameter-free stencil and verify it against the calibrated pulses. If it misses the pulse threshold, revisit the face or velocity convention before testing the inverse solver.
- **Alternatives Weighed**: `Embed fitted wrap multipliers` (rejected: The inferred inward rates match adjacent-face diffusion, and the asymmetric seam increments match the signed endpoint velocities divided by grid spacing; the stencil can use the supplied physical coefficients directly.)

```bash
set -e
cd /holdout_task
cat > downstream_solver.py <<'PY'
"""Conservative wrapped-x, closed-y transport and box-simplex inversion."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def axis_operator(kappa, velocity, h, wrapped=False):
    face = 0.5 * (kappa[:-1] + kappa[1:]) / h**2
    v = velocity[1:-1] / h
    west = -face[:-1] - np.maximum(v, 0)
    east = -face[1:] - np.maximum(-v, 0)
    if wrapped:
        near_left = face[0] + max(-velocity[0], 0) / h
        near_right = face[-1] + max(velocity[-1], 0) / h
        seam = 0.5 * (kappa[0] + kappa[-1]) / h**2
        wrap_left = seam + max(velocity[0], 0) / h
        wrap_right = seam + max(-velocity[-1], 0) / h
    else:
        near_left, near_right = kappa[[0, -1]] / h**2
        wrap_left = wrap_right = 0.0
    a = sparse.diags((np.r_[west, -near_right],
                      np.r_[near_left + wrap_left, -west - east, near_right + wrap_right],
                      np.r_[-near_left, east]), (-1, 0, 1), format='lil')
    if wrapped:
        a[0, -1] -= wrap_left
        a[-1, 0] -= wrap_right
    return a.tocsc()


def chamber_response(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = axis_operator(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1), wrapped=True)
    ay = axis_operator(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1))
    op = (sparse.kron(sparse.eye(ny), ax, format='csc')
          + sparse.kron(ay, sparse.eye(nx), format='csc')
          + sparse.diags(np.asarray(study['V_reaction_2d']).ravel(order='F'), format='csc'))
    eye = sparse.eye(nx * ny, format='csc')
    rhs = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    readings = []
    for mu in study['mu_shifts']:
        response = splu(op + mu * eye).solve(rhs)
        readings.append(response[i + nx * j])
    return np.concatenate(readings), response.T.reshape((-1, nx, ny), order='F')


def box_simplex(hess, linear, cap, budget):
    """Minimize 0.5*w@hess@w + linear@w, keeping iterates feasible."""
    n = len(linear)
    w = np.full(n, budget / n)
    if budget == 0 or budget == n * cap:
        g = hess @ w + linear
        return w, float(g.min() if budget == 0 else g.max())
    # -1 and +1 denote active lower and upper bounds.
    state = np.zeros(n, dtype=int)
    tol = 64 * np.finfo(float).eps * max(1.0, np.linalg.norm(hess, np.inf) * cap,
                                         np.linalg.norm(linear, np.inf))
    for _ in range(20 * n * n):
        f = np.flatnonzero(state == 0)
        a = np.flatnonzero(state != 0)
        if f.size:
            hf = hess[np.ix_(f, f)]
            c = linear[f] + hess[np.ix_(f, a)] @ w[a]
            v, q = np.linalg.solve(hf, np.column_stack((-c, np.ones(f.size)))).T
            nu = (budget - w[a].sum() - v.sum()) / q.sum()
            target = v + nu * q
            if f.size == 1:
                target[0] = np.clip(budget - w[a].sum(), 0, cap)
            d = target - w[f]
            alpha = np.full(f.size, np.inf)
            low, high = (target < 0) & (d < 0), (target > cap) & (d > 0)
            alpha[low] = -w[f[low]] / d[low]
            alpha[high] = (cap - w[f[high]]) / d[high]
            k = int(alpha.argmin())
            if alpha[k] < 1:
                w[f] += alpha[k] * d
                state[f[k]] = -1 if low[k] else 1
                w[f[k]] = 0.0 if low[k] else cap
                continue
            w[f] = np.clip(target, 0, cap)
        g = hess @ w + linear
        if not f.size:
            lower = np.flatnonzero(state < 0)
            upper = np.flatnonzero(state > 0)
            l, u = lower[g[lower].argmin()], upper[g[upper].argmax()]
            if g[u] - g[l] <= tol:
                return w, float((g[u] + g[l]) / 2)
            state[[l, u]] = 0
            continue
        violation = state * (g - nu)
        k = int(violation.argmax())
        if violation[k] <= tol:
            return w, float(nu)
        state[k] = 0
    raise RuntimeError('box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    """Solve using the observations in canonical coordinates."""
    s = cal_item
    response, terminal = chamber_response(s)
    y = np.asarray(s['y_obs_raw'], dtype=float)
    z = np.asarray(s['Z_nuisance'], dtype=float)
    p = np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    st = response - z @ (p @ response)
    ry = y - z @ (p @ y)
    ridge, cap, budget = s['source_ridge'], s['u_box'], s['total_budget']
    hess = st.T @ st + ridge * np.eye(st.shape[1])
    w, nu = box_simplex(hess, -(st.T @ ry), cap, budget)
    residual = st @ w - ry
    g = st.T @ residual + ridge * w
    lower, upper = np.maximum(0, g - nu), np.maximum(0, nu - g)
    obj = 0.5 * (residual @ residual + ridge * (w @ w))
    dual = obj - g @ w + nu * budget - cap * upper.sum()
    return {'weights': w, 'terminal_field': np.tensordot(w, terminal, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
                                'primal_objective': float(obj), 'dual_lower_bound': float(dual)}}
PY
python3 validate_public_schema.py /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp:/holdout_task python3 - <<'PY'
from fit_walls import s, target, ps, report
from downstream_solver import chamber_response
import numpy as np
S,_=chamber_response(s)
e=S @ np.column_stack([p['w_pulse'] for p in ps])-target
report('periodic upwind x and closed y', e)
assert np.max(np.sqrt(np.mean(e**2,axis=0))) < 3e-6
PY
```
- **Container Exit Code**: `0` (`627 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
periodic upwind x and closed y pulse rms [1.106e-06 1.125e-06 1.033e-06] shift rms [1.331e-06 1.025e-06 8.570e-07] nuisance fraction 0.23794

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=939e40bffc9ee888 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_939e40bf (directory).
```

### Turn 7 — `execute_in_remote_docker` (`38978 ms` LLM)
- **Rationale**: The pulse test supports the wall stencil, but the schema test does not verify field ordering or optimality. Check both against an independent dense reference in an isolated process, then stress the box-simplex solver at lower, upper, and mixed active bounds. Any failure will identify whether to fix assembly, reconstruction, or the active set.

```bash
set -e
cd /holdout_task
python3 validate_public_schema.py /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys,pickle
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.optimize import minimize
from batch_solver import solve_batch
from downstream_solver import solve_downstream, box_simplex
from _verified_upstream import calibrate_upstream

def dense_axis(s,key,periodic):
    k,v=s['kappa_'+key],s['vel_'+key]
    n=len(k); h=s['L'+key]/(n-1); a=np.zeros((n,n))
    for i in range(n):
        if not periodic and i in (0,n-1):
            j=1 if i==0 else n-2
            a[i,j]=-k[i]/h**2
        else:
            for direction in (-1,1):
                j=(i+direction)%n
                a[i,j]=-(k[i]+k[j])/(2*h*h)-max(0,-direction*v[i])/h
        a[i,i]=-sum(a[i])
    return a

metrics=np.zeros(7)
items=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
for item in items:
    study=item['study']; s=calibrate_upstream(study); nx,ny=s['nx'],s['ny']
    a=np.kron(np.eye(ny),dense_axis(s,'x',True))+np.kron(dense_axis(s,'y',False),np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
    i,j=s['sensor_ij'].T; ids=i+nx*j
    rhs=np.array([m.flatten(order='F') for m in s['source_modes']]).T
    us=[np.linalg.solve(a+mu*np.eye(nx*ny),rhs) for mu in s['mu_shifts']]; S=np.vstack([u[ids] for u in us])
    z=s['Z_nuisance']; R=np.eye(len(z))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
    B=R@S; y=R@s['y_obs_raw']; ridge,cap,budget=(s[k] for k in ('source_ridge','u_box','total_budget'))
    J=lambda w: 0.5*(np.linalg.norm(B@w-y)**2+ridge*(w@w))
    G=lambda w: B.T@(B@w-y)+ridge*w
    out=solve_batch(study); direct=solve_downstream(s); w=out['weights']; c=out['kkt_certificate']; nu=c['nu_sum']; lo=np.array(c['lam_lower']); hi=np.array(c['lam_upper']); g=G(w)
    ref=minimize(J,np.full(len(w),budget/len(w)),jac=G,bounds=[(0,cap)]*len(w),constraints={'type':'eq','fun':lambda x:x.sum()-budget,'jac':lambda x:np.ones_like(x)},method='SLSQP',options={'ftol':1e-13})
    assert ref.success
    m=[max(abs(w.sum()-budget),max(0,-w.min(),w.max()-cap)), np.max(np.abs(g-nu-lo+hi)), max(np.max(abs(w*lo)),np.max(abs((cap-w)*hi))), abs(J(w)-c['primal_objective']), abs(J(w)-c['dual_lower_bound']), np.max(abs(out['terminal_field']-(us[-1]@w).reshape(nx,ny,order='F'))),abs(J(w)-ref.fun)]
    metrics=np.maximum(metrics,m)
    np.testing.assert_allclose(w,direct['weights'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(c['weights'],w,rtol=0,atol=0)
    assert min(lo.min(),hi.min())>=0
print('Public cases:',len(items),'max feasibility, stationarity, complementarity, objective error, gap, field error, reference objective error:',metrics)
assert np.max(metrics)<1e-9
rng=np.random.default_rng(24); worst=0.; slices=set(); total=0
for n in (4,8,16):
    for budget in (0.,0.17,1.,n/2,n-0.13,float(n)):
        for _ in range(40):
            A=rng.normal(size=(n,n)); H=A.T@A+0.04*np.eye(n); c=rng.normal(size=n)*rng.choice([0.1,1.,50.]); w,nu=box_simplex(H,c,1.,budget)
            g=H@w+c; low=np.maximum(0,g-nu); high=np.maximum(0,nu-g)
            worst=max(worst,abs(w.sum()-budget),max(0,-w.min(),w.max()-1),np.max(abs(w*low)),np.max(abs((1-w)*high)))
            slices.add((bool(np.any(w<1e-10)),bool(np.any(w>1-1e-10)),bool(np.any((w>1e-10)&(w<1-1e-10))))
            total+=1
print('Quadratic stress cases:',total,'active patterns:',sorted(slices),'worst feasibility/complementarity:',worst)
assert worst<1e-9
PY
```
- **Container Exit Code**: `1` (`612 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=939e40bffc9ee888 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_939e40bf (directory).
```
- **Stderr (tail)**:
```text
File "<stdin>", line 52
    slices.add((bool(np.any(w<1e-10)),bool(np.any(w>1-1e-10)),bool(np.any((w>1e-10)&(w<1-1e-10))))
              ^
SyntaxError: '(' was never closed
```

### Turn 8 — `execute_in_remote_docker` (`43697 ms` LLM)
- **Rationale**: The prior verification script stopped at a syntax error before testing the solver. Rerun it without changing the artifacts. The dense reference checks field ordering and objective assembly; the active-bound stress cases check feasibility and complementary slackness.

```bash
set -e
cd /holdout_task
python3 validate_public_schema.py /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys, pickle
sys.path.insert(0, '/holdout_task')
import numpy as np
from scipy.optimize import minimize
from batch_solver import solve_batch
from downstream_solver import solve_downstream, box_simplex
from _verified_upstream import calibrate_upstream

def dense_axis(s, key, periodic):
    k, v = s['kappa_'+key], s['vel_'+key]
    n = len(k)
    h = s['L'+key]/(n-1)
    a = np.zeros((n,n))
    for i in range(n):
        if not periodic and i in (0,n-1):
            j = 1 if i == 0 else n-2
            a[i,j] = -k[i]/h**2
        else:
            for d in (-1,1):
                j = (i+d)%n
                a[i,j] += -(k[i]+k[j])/(2*h*h)-max(0,-d*v[i])/h
        a[i,i] = -sum(a[i])
    return a

items = pickle.load(open('public_batch.pkl','rb'))
worst = np.zeros(7)
pulse_worst = 0.
for item in items:
    study = item['study']
    pulses = item.get('campaign_diagnostics')
    s = calibrate_upstream(dict(study, reference_tracer_pulses=pulses)) if pulses else calibrate_upstream(study)
    nx,ny = s['nx'],s['ny']
    a = np.kron(np.eye(ny),dense_axis(s,'x',True))+np.kron(dense_axis(s,'y',False),np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
    i,j = s['sensor_ij'].T
    rhs = np.column_stack([m.ravel(order='F') for m in s['source_modes']])
    us = [np.linalg.solve(a+mu*np.eye(nx*ny),rhs) for mu in s['mu_shifts']]
    S = np.vstack([u[i+nx*j] for u in us])
    for pulse in s.get('reference_tracer_pulses', []):
        pulse_worst = max(pulse_worst, np.sqrt(np.mean((S@pulse['w_pulse']-pulse['y_pulse_raw'])**2)))
    z = s['Z_nuisance']
    R = np.eye(len(z))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
    B,y = R@S,R@s['y_obs_raw']
    ridge,cap,budget = (s[k] for k in ('source_ridge','u_box','total_budget'))
    J = lambda w: 0.5*(np.linalg.norm(B@w-y)**2+ridge*(w@w))
    G = lambda w: B.T@(B@w-y)+ridge*w
    out = solve_batch(study)
    w,c = out['weights'],out['kkt_certificate']
    nu,lo,hi = c['nu_sum'],np.array(c['lam_lower']),np.array(c['lam_upper'])
    g = G(w)
    ref = minimize(J,np.full(len(w),budget/len(w)),jac=G,bounds=[(0,cap)]*len(w),constraints={'type':'eq','fun':lambda x:x.sum()-budget,'jac':lambda x:np.ones_like(x)},method='SLSQP',options={'ftol':1e-13})
    assert ref.success, ref.message
    errs = [max(abs(w.sum()-budget),0,-w.min(),w.max()-cap),np.max(abs(g-nu-lo+hi)),max(np.max(abs(w*lo)),np.max(abs((cap-w)*hi))),abs(J(w)-c['primal_objective']),abs(J(w)-c['dual_lower_bound']),np.max(abs(out['terminal_field']-(us[-1]@w).reshape(nx,ny,order='F'))),abs(J(w)-ref.fun)]
    worst = np.maximum(worst,errs)
    np.testing.assert_allclose(w,solve_downstream(s)['weights'],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(c['weights'],w)
    assert min(lo.min(),hi.min()) >= 0
print('Public cases:',len(items),'max feasibility, stationarity, complementarity, objective error, gap, field error, reference objective error:',worst)
print('Worst calibrated pulse RMSE:',pulse_worst)
assert np.max(worst)<1e-9 and 0 < pulse_worst < 3e-6
rng=np.random.default_rng(24)
stress=0.
patterns=set()
for n in (4,8,16):
    for budget in (0.,0.17,1.,n/2,n-0.13,float(n)):
        for _ in range(40):
            A=rng.normal(size=(n,n)); H=A.T@A+0.04*np.eye(n)
            c=rng.normal(size=n)*rng.choice([0.1,1.,50.])
            w,nu=box_simplex(H,c,1.,budget)
            g=H@w+c
            low,high=np.maximum(0,g-nu),np.maximum(0,nu-g)
            stress=max(stress,abs(w.sum()-budget),-w.min(),w.max()-1,np.max(abs(w*low)),np.max(abs((1-w)*high)))
            patterns.add(tuple(bool(x) for x in (np.any(w<1e-10),np.any(w>1-1e-10),np.any((w>1e-10)&(w<1-1e-10)))))
print('720 quadratic stress cases; active patterns:',sorted(patterns),'worst feasibility/complementarity:',stress)
assert stress < 1e-9
PY
```
- **Container Exit Code**: `0` (`1161 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Public cases: 1 max feasibility, stationarity, complementarity, objective error, gap, field error, reference objective error: [0.00000000e+00 8.56953397e-16 1.47430581e-16 5.20417043e-18
 5.84601811e-16 1.38777878e-16 2.17915963e-14]
Worst calibrated pulse RMSE: 1.1250621385666948e-06
720 quadratic stress cases; active patterns: [(False, False, True), (False, True, False), (False, True, True), (True, False, False), (True, False, True), (True, True, False), (True, True, True)] worst feasibility/complementarity: 2.3590018827235325e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=939e40bffc9ee888 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_939e40bf (directory).
```

### Turn 9 — `submit_completed_task` (`13011 ms` LLM)
- **Rationale**: The parameter-free stencil matches the calibrated pulses at the measured noise scale. Isolated-process verification confirmed the terminal field against an independent dense solve and verified the KKT certificate across all observed active-bound patterns.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:408', '/holdout_task/downstream_solver.py': 'FILE:5242'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/downstream_solver.py` and verified it through the existing `/holdout_task/batch_solver.py` entry point. The public schema passed. Worst calibrated tracer-pulse RMSE was 1.13e-6, and the worst feasibility or complementary-slackness error across 720 stress cases was 2.36e-14.`

### Turn 10 — `execute_in_remote_docker` (`32171 ms` LLM)
- **Rationale**: The pooled pulse error could hide a bad sensor-bank/shift slice, or the boundary preference could disappear after removing nuisance directions. If either happens, revisit the wall or calibration mechanism; otherwise keep the verified artifacts unchanged. Relay modes and classification accuracy do not apply to this chamber.

```bash
cat > /tmp/audit_wall_slices.py <<'PY'
import sys
sys.path[:0] = ['/holdout_task', '/tmp']
import numpy as np
from fit_walls import s, ps, target, predict
from batch_solver import solve_batch
from downstream_solver import chamber_response
import pickle

S, fields = chamber_response(s)
W = np.column_stack([p['w_pulse'] for p in ps])
y = S @ W
err = y - target
k = s['kappa_x']; v = s['vel_x']; h = s['Lx']/(len(k)-1)
end = k[[0, -1]]
face = (end + k[[1, -2]]) / 2
near = face + h*np.maximum([-v[0], v[-1]], 0)
seam = end.mean() + h*np.maximum([v[0], -v[-1]], 0)
full = np.r_[near/end, 0., 0., seam/end, 1., 1., 0., 0., 0., 0.]
np.testing.assert_allclose(y, predict(full), atol=2e-13, rtol=0)
variants = {'no seam': (slice(4,6), [0.,0.]), 'diffusive seam only': (slice(4,6), end.mean()/end), 'endpoint kappa in place of face': (slice(0,2), [1.,1.]), 'open y': (slice(8,10), [1.,1.])}
Z = s['Z_nuisance']
P = Z @ np.linalg.pinv(Z)
R = np.eye(len(Z)) - P
rms = lambda a: np.sqrt(np.mean(a*a,axis=0))
blocks = np.array_split(np.arange(len(Z)), len(s['mu_shifts']))
bank = np.asarray(s['row_bank_id'])
print('Periodic x / closed y pulse RMSE:', rms(err))
print('Mean pulse residual by bank:', {str(b): err[bank==b].mean(axis=0).tolist() for b in np.unique(bank)})
print('Nuisance-orthogonal pulse RMSE:', rms(R @ err))
worst = max(float(rms(err[idx[bank[idx]==b]]).max()) for idx in blocks for b in np.unique(bank[idx]))
print('Worst pulse × shift × bank RMSE:', worst)
for name,(cols,val) in variants.items():
    q = full.copy(); q[cols] = val
    alt = predict(q) - target
    print(name, 'raw pulse RMSE', rms(alt), 'nuisance-orthogonal pulse RMSE', rms(R @ alt))
    assert rms(R @ alt).min() > 10 * rms(R @ err).max()
assert worst < 5e-6
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    out = solve_batch(pickle.load(f)[0]['study'])
c = out['kkt_certificate']
print('Isolated batch entry point: field shape', out['terminal_field'].shape, 'duality gap', c['primal_objective'] - c['dual_lower_bound'])
assert out['terminal_field'].shape == fields.shape[1:]
assert abs(c['primal_objective'] - c['dual_lower_bound']) < 1e-9
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_wall_slices.py
```
- **Container Exit Code**: `0` (`577 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Periodic x / closed y pulse RMSE: [1.10603266e-06 1.12506214e-06 1.03266248e-06]
Mean pulse residual by bank: {'0': [6.853863215485155e-08, -1.0827347754049227e-07, 1.3440074316898679e-07], '1': [-8.317907650905986e-07, -7.266848763481359e-07, -7.499778565632035e-07], '2': [-1.8609415420575124e-06, -1.9194767727959776e-06, -1.753656459234365e-06], '3': [2.282474149947177e-07, -1.9093786761172533e-08, 2.9099914010486533e-07]}
Nuisance-orthogonal pulse RMSE: [1.07206867e-06 1.09654158e-06 1.00121679e-06]
Worst pulse × shift × bank RMSE: 2.2713505484653114e-06
no seam raw pulse RMSE [0.01072296 0.01364515 0.04608607] nuisance-orthogonal pulse RMSE [0.01045593 0.01327968 0.04417723]
diffusive seam only raw pulse RMSE [0.00045403 0.00059529 0.00219893] nuisance-orthogonal pulse RMSE [0.00044257 0.00058049 0.00208839]
endpoint kappa in place of face raw pulse RMSE [0.00094198 0.00073807 0.00052273] nuisance-orthogonal pulse RMSE [0.00090573 0.00072159 0.00050964]
open y raw pulse RMSE [0.11820229 0.09656876 0.1064673 ] nuisance-orthogonal pulse RMSE [0.11503968 0.09500253 0.10381033]
Isolated batch entry point: field shape (14, 12) duality gap 5.898059818321144e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=939e40bffc9ee888 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_939e40bf (directory).
```

### Turn 11 — `submit_completed_task` (`15570 ms` LLM)
- **Rationale**: The sensor-bank and shift audit confirmed that the wall model's advantage persists after nuisance projection. The unchanged artifact passed isolated-process execution, dense field reconstruction, and active-bound KKT stress tests.
- **Alternatives Weighed**: `Use a diffusion-only x seam` (rejected: Even after nuisance projection, its best pulse RMSE was 4.43e-4, compared with 1.10e-6 for the worst pulse under the signed-upwind stencil.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/downstream_solver.py` and verified it through the existing `/holdout_task/batch_solver.py` entry point. The public schema passed, and the terminal field matched an independent dense solve to 1.39e-16. The worst calibrated tracer RMSE across pulse, shift, and sensor-bank slices was 2.27e-6. Across 720 optimization stress cases, the worst feasibility or complementary-slackness error was 2.36e-14.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
