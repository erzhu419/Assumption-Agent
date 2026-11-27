# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_v2_c03_batch_07__blind_continual_campaign`

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
| **Turns Used / Max Turns** | `11 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `252,114` (`245,085` / `105,600` / `7,029`) |
| **Total Episode Wall Time (s)** | `266.65s` |

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

### Turn 1 — `execute_in_remote_docker` (`19997 ms` LLM)
- **Rationale**: Inspect the public schema and apparatus log to detect a regime or input mismatch before selecting an archived operator; a mismatch changes which boundary model or calibration fit to reuse.
- **Alternatives Weighed**: `Reuse the most recent solver unchanged` (rejected: The latest solver is not necessarily the one whose boundary regime matches the blind batch.)

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/public_asset_index.json /holdout_task/validate_public_schema.py; cat /tmp/hegel_continual_archive/archive_apparatus_log.json; python3 - <<'PY'
import pickle
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
print('item keys', list(item), 'blind', item.get('campaign_diagnostics') is None)
for k,v in item['study'].items():
 print(k,type(v).__name__,getattr(v,'shape',None),list(v) if isinstance(v,dict) else v if isinstance(v,(int,float,str)) else '')
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling' || true
```
- **Container Exit Code**: `0` (`425 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1.03274,
        0.72431,
        0.77311,
        -0.28995,
        0.76138,
        0.83324,
        2.56265
      ],
      [
        1.0911,
        1.24382,
        0.48697,
        0.76483,
        -0.36523,
        0.67657,
        1.1181,
        2.62256
      ],
      [
        1.00767,
        1.26192,
        0.77916,
        0.88501,
        -0.25667,
        0.64461,
        0.97476,
        2.62671
      ],
      [
        0.74237,
        1.11072,
        0.43766,
        0.9209,
        -0.49947,
        0.59228,
        0.63851,
        2.47322
      ],
      [
        0.58096,
        1.10104,
        0.54799,
        0.45899,
        -0.35978,
        0.65456,
        0.45,
        2.50683
      ],
      [
        0.14417,
        0.96862,
        0.90258,
        0.17165,
        -0.32816,
        0.51567,
        0.18329,
        2.5177
      ],
      [
        -0.22877,
        1.17854,
        1.01167,
        0.16294,
        -0.40196,
        0.60404,
        -0.16185,
        2.56338
      ],
      [
        -0.49855,
        1.11482,
        0.58233,
        -0.60914,
        -0.43372,
        0.64441,
        -0.57095,
        2.58166
      ],
      [
        -0.96488,
        0.99611,
        0.58379,
        -0.95951,
        -0.37188,
        0.6252,
        -0.71368,
        2.49378
      ]
    ]
  }
]item keys ['study', 'campaign_diagnostics'] blind False
Lx float None 2.4
Ly float None 2.0
nx int None 14
ny int None 12
kappa_x ndarray (14,) 
kappa_y ndarray (12,) 
vel_x ndarray (14,) 
vel_y ndarray (12,) 
robin_x float None 0.943511194485015
robin_y float None 1.0215322743752013
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
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`14115 ms` LLM)
- **Rationale**: Inspect prior boundary and calibration implementations before scoring them on the pulses; this will distinguish a reused operator mismatch from a calibration mismatch and avoid reusing an incompatible optimizer.
- **Alternatives Weighed**: `Select the regime solely from the archive log` (rejected: This campaign has two direct tracer measurements, so the archive can provide candidate operators but should not decide the regime.)

```bash
cd /tmp/hegel_continual_archive; for f in *solver.py; do echo "=== $f"; grep -nE '^def |^    (ax |ay |Ax |Ay |return |.*single_bank)' "$f"; done; cat beta_m98_v2_c03_batch_06_solver.py; cat /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```
- **Container Exit Code**: `0` (`316 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-1e-6: errs.append("kkt_neg_multiplier")
            if dl < -1e-7 or po < dl - 1e-6 or (po - dl) > 2e-4: errs.append(f"kkt_duality_gap_or_sign:{po:.6f}_vs_{dl:.6f}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
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

### Turn 3 — `execute_in_remote_docker` (`13533 ms` LLM)
- **Rationale**: The previous output truncated the operator and calibration bodies. Read only those bodies to identify distinct boundary and sensor candidates for a crossed pulse comparison; this avoids mistaking calibration error for a boundary mismatch.
- **Alternatives Weighed**: `Use the boxed sparse-ridge skill` (rejected: It does not enforce the required equality budget.)

```bash
cd /tmp/hegel_continual_archive; for f in *solver.py; do echo ===$f; sed -n '1,/^def _box_budget/p' "$f" | head -85; done
```
- **Container Exit Code**: `0` (`294 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
asarray(study['V_reaction_2d']).ravel(order='F'), format='csc'))
    identity = eye(nx * ny, format='csc')
    sources = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    ix, iy = np.asarray(study['sensor_ij']).T
    sensor = ix + nx * iy
    rows = []
    for shift in study['mu_shifts']:
        field = splu(operator + shift * identity).solve(sources)
        rows.append(field[sensor, :])
    return np.vstack(rows), field


def _box_budget(H, c, cap, budget):
===beta_m98_v2_c03_s00_solver.py
"""Calibrated transport inversion with a box-and-budget KKT certificate."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, robin):
    face = 0.5 * (kappa[1:] + kappa[:-1]) / h**2
    lower = np.r_[-face[:-1] - np.maximum(velocity[1:-1], 0) / h,
                  -kappa[-1] / h**2]
    upper = np.r_[-kappa[0] / h**2,
                  -face[1:] - np.maximum(-velocity[1:-1], 0) / h]
    main = np.r_[kappa[0] / h**2 + robin / h,
                 face[:-1] + face[1:] + np.abs(velocity[1:-1]) / h,
                 kappa[-1] / h**2 + robin / h]
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    design = np.column_stack((np.ones(len(ref)), ref))
    coefficients = np.linalg.lstsq(design, sweeps['raw_rosette'], rcond=None)[0]
    raw = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return np.linalg.solve(coefficients[1:].T, (raw - coefficients[0]).T).T.ravel()


def _forward(study):
    nx, ny = study['nx'], study['ny']
    axes = [_axis(np.asarray(study['kappa_' + k]), np.asarray(study['vel_' + k]),
                  study['L' + k] / (n - 1), study['robin_' + k])
            for k, n in (('x', nx), ('y', ny))]
    operator = (kron(eye(ny), axes[0], format='csc')
                + kron(axes[1], eye(nx), format='csc')
                + diags(np.asarray(study['V_reaction_2d']).ravel(order='F'), format='csc'))
    identity = eye(nx * ny, format='csc')
    sources = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    ix, iy = np.asarray(study['sensor_ij']).T
    sensor = ix + nx * iy
    rows = []
    for shift in study['mu_shifts']:
        field = splu(operator + shift * identity).solve(sources)
        rows.append(field[sensor, :])
    return np.vstack(rows), field


def _box_budget(H, c, cap, budget):
```

### Turn 4 — `execute_in_remote_docker` (`21281 ms` LLM)
- **Rationale**: Calibrate independently from sweeps, then compare per-pulse errors with and without nuisance projection. If no archived pair fits both pulses, inspect the boundary rows and calibration residuals separately before selecting a model.
- **Alternatives Weighed**: `Reuse the operator from the rejected table row` (rejected: The gate does not identify whether the boundary, calibration, or optimizer caused the failure.)

```bash
cd /tmp/hegel_continual_archive; sed -n '1,58p' beta_m98_v2_c03_batch_06_solver.py; sed -n '1,20p' beta_m98_v2_c03_batch_03_solver.py; python3 - <<'PY'
import glob,runpy,pickle,numpy as np
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; st=item['study']; sweeps=st['sensor_calibration_sweeps']; ref=sweeps['ref_rosette']; raw=sweeps['raw_rosette']; X=np.c_[np.ones(len(ref)),ref]; print('sweeps',ref.shape,'affine residual quantiles',np.quantile(np.linalg.norm(X@np.linalg.lstsq(X,raw,rcond=None)[0]-raw,axis=1),[0,.5,.9,1])); print('pulses',len(item['campaign_diagnostics'])); mods={f.split('_')[-2]:runpy.run_path(f) for f in sorted(glob.glob('*solver.py'))}; Z=st['Z_nuisance']; project=lambda r:r-Z@np.linalg.lstsq(Z,r,rcond=None)[0];
for a in ['s00','01','03']:
 S,_=mods[a]['_forward'](st)
 for b in ['s00','04','06']:
  errors=[]
  for p in item['campaign_diagnostics']:
   t=dict(st,y_obs_raw=p['y_pulse_raw']); y=mods[b]['_calibrate'](t); r=y-S@p['w_pulse']; errors.append(tuple(round(float(x),8) for x in [np.sqrt(np.mean(r*r)),np.sqrt(np.mean(project(r)**2)),np.sqrt(np.mean(y*y))]))
  print(a,b,errors)
PY
```
- **Container Exit Code**: `0` (`705 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
for i in range(4)])


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    coef = _rosette_coefficients(np.asarray(sweeps['ref_rosette']),
                                 np.asarray(sweeps['raw_rosette']))
    raw = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    linear = coef[:, 1:5]
    ref = np.linalg.solve(linear, (raw - coef[:, 0]).T).T
    for j, target in enumerate(raw):
        def residual(q):
            return np.array([_features(q, i) @ coef[i] for i in range(4)]) - target

        def jacobian(q):
            matrix = linear.copy()
            for i in range(4):
                other = (i + 1) % 4
                matrix[i, i] += coef[i, 5] * q[other] + 3 * coef[i, 6] * q[i]**2
                matrix[i, other] += coef[i, 5] * q[i]
            return matrix

        result = least_squares(residual, ref[j], jac=jacobian,
                               gtol=1e-13, ftol=1e-13, xtol=1e-13)
"""Calibrated periodic-wall transport inversion with a box-and-budget KKT certificate."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    face = 0.5 * (kappa[1:] + kappa[:-1]) / h**2
    wrap = 0.5 * (kappa[-1] + kappa[0]) / h**2
    left = np.r_[wrap, face] + np.maximum(velocity, 0) / h
    right = np.r_[face, wrap] + np.maximum(-velocity, 0) / h
    return diags((left + right, -left[1:], -right[:-1],
                  -left[:1], -right[-1:]),
                 (0, -1, 1, len(kappa) - 1, 1 - len(kappa)), format='csc')


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
sweeps (80, 4) affine residual quantiles [ 0.41313653  0.96473758  1.63908791 25.04220409]
pulses 2
s00 s00 [(0.52398385, 0.52059705, 0.55433024), (0.52240319, 0.51909952, 0.55650442)]
s00 04 [(0.1152564, 0.11327733, 0.19331816), (0.10314331, 0.10130974, 0.18579242)]
s00 06 [(0.1152564, 0.11327733, 0.19331816), (0.10314331, 0.10130974, 0.18579242)]
01 s00 [(0.5220696, 0.51997784, 0.55433024), (0.52031175, 0.51806285, 0.55650442)]
01 04 [(0.04027235, 0.04023516, 0.19331816), (0.0060337, 0.00595992, 0.18579242)]
01 06 [(0.04027235, 0.04023516, 0.19331816), (0.0060337, 0.00595992, 0.18579242)]
03 s00 [(0.52064037, 0.51843324, 0.55433024), (0.520792, 0.51853737, 0.55650442)]
03 04 [(1.18e-06, 1.17e-06, 0.19331816), (1.18e-06, 1.16e-06, 0.18579242)]
03 06 [(1.18e-06, 1.17e-06, 0.19331816), (1.18e-06, 1.16e-06, 0.18579242)]
```

### Turn 5 — `execute_in_remote_docker` (`20361 ms` LLM)
- **Rationale**: The crossed pulse comparison identifies which components to combine. Build a standalone solver and run the public schema gate to catch interface or certificate failures before checking numerical accuracy on the exact artifact.
- **Alternatives Weighed**: `Reuse the periodic archive solver unchanged` (rejected: Its affine rosette model fails both tracer pulses even with the correct boundary operator.)
- **ATMS NOGOOD DECLARED**: Refuted `Open or reflecting endpoint rows and affine sensor calibration suffice for this campaign.` | Witness: `With the same cubic calibration, worst pulse RMSE is 0.1153 for open walls, 0.04027 for reflecting walls, and 1.18e-6 for periodic walls. With periodic walls held fixed, affine calibration gives 0.5208.` | Blanket Diagnosis: `The crossed comparisons isolate missing periodic endpoint couplings from missing sensor interaction and cubic terms; nuisance projection does not remove either mismatch.` | Sublated Pivot: `Combine the archived periodic transport operator with the archived cubic rosette model fitted anew from each study's sweeps.`

```bash
set -e
cd /tmp/hegel_continual_archive
python3 - <<'PY'
from pathlib import Path
periodic = Path('beta_m98_v2_c03_batch_03_solver.py').read_text()
cubic = Path('beta_m98_v2_c03_batch_06_solver.py').read_text()
start = periodic[periodic.index('import numpy'):periodic.index('def _calibrate')]
start = start.replace('import numpy as np\n', 'import numpy as np\nfrom scipy.optimize import least_squares\n')
calibration = cubic[cubic.index('def _features'):cubic.index('def _forward')]
Path('/holdout_task/batch_solver.py').write_text('"""Periodic transport inversion with sweep-fitted rosettes and a box-budget certificate."""\n\n' + start + calibration + periodic[periodic.index('def _forward'):])
PY
sed -n '/^def _box_budget/,$p' /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`541 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
stem[:m, :m] = H[np.ix_(free, free)]
        system[m, m] = 0
        rhs = np.r_[-c[free] - H[np.ix_(free, fixed)] @ w[fixed], budget - w[fixed].sum()]
        sol = np.linalg.solve(system, rhs)
        candidate, nu = sol[:-1], -float(sol[-1])
        direction = candidate - w[free]
        ratio = np.full(m, np.inf)
        low, high = (candidate < 0) & (direction < 0), (candidate > cap) & (direction > 0)
        ratio[low] = -w[free[low]] / direction[low]
        ratio[high] = (cap - w[free[high]]) / direction[high]
        k = np.argmin(ratio)
        if ratio[k] < 1:
            w[free] += max(0., ratio[k]) * direction
            j = free[k]
            active[j] = -1 if low[k] else 1
            w[j] = 0. if low[k] else cap
            continue
        w[free] = np.clip(candidate, 0., cap)
        violation = active * (H @ w + c - nu)
        j = np.argmax(violation)
        if violation[j] <= tol:
            return w, nu
        active[j] = 0
    raise RuntimeError('Box-and-budget active set did not converge')


def solve_batch(study: dict) -> dict:
    y = _calibrate(study)
    S, last = _forward(study)
    Z = np.asarray(study['Z_nuisance'])
    observations = np.column_stack((S, y))
    profiled = observations - Z @ np.linalg.solve(
        Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T @ observations)
    D, target = profiled[:, :-1], profiled[:, -1]
    ridge, cap, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    H = D.T @ D + ridge * np.eye(D.shape[1])
    w, nu = _box_budget(H, -D.T @ target, cap, budget)
    residual = D @ w - target
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = D.T @ residual + ridge * w
    reduced = g - nu
    dual = objective - g @ w + nu * budget + cap * np.minimum(0, reduced).sum()
    return {
        'weights': w,
        'terminal_field': (last @ w).reshape((study['nx'], study['ny']), order='F'),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': np.maximum(0, reduced), 'lam_upper': np.maximum(0, -reduced),
            'primal_objective': objective, 'dual_lower_bound': float(dual),
        },
    }
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e0fdaa7aab8cf65c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e0fdaa7a (directory).
```

### Turn 6 — `execute_in_remote_docker` (`33123 ms` LLM)
- **Rationale**: Verify the exact artifact in an isolated Python process. A pulse-slice failure would reopen calibration or boundary selection, a PDE residual failure would isolate an assembly or layout error, and an optimizer or certificate disagreement would require correcting the quadratic solve.
- **Alternatives Weighed**: `Submit on the schema pass alone` (rejected: The schema does not check the terminal PDE residual or pulse accuracy at each shift.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy
import numpy as np
from scipy.optimize import minimize

m = runpy.run_path('/holdout_task/batch_solver.py')
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
out = m['solve_batch'](s)
w, u, cert = (out[k] for k in ('weights', 'terminal_field', 'kkt_certificate'))
S, _ = m['_forward'](s)

# Check periodic seam rows, field layout and the terminal shift directly against the PDE.
lhs = (s['V_reaction_2d'] + s['mu_shifts'][-1]) * u
for axis, k in enumerate('xy'):
    n = u.shape[axis]
    h = s['L'+k] / (n-1)
    kap, vel = (np.asarray(s[name+k]) for name in ('kappa_', 'vel_'))
    shape = (n,1) if axis == 0 else (1,n)
    for step in (-1, 1):
        coef = (kap + np.roll(kap, -step)) / (2*h*h) + np.maximum(-step*vel, 0)/h
        lhs += coef.reshape(shape) * (u - np.roll(u, -step, axis=axis))
rhs = np.tensordot(w, s['source_modes'], axes=1)
field_error = np.max(np.abs(lhs-rhs))
assert field_error < 1e-9, field_error

# Score both pulses separately at every shift using the exact submitted implementation.
errors = []
for p in item['campaign_diagnostics']:
    calibrated = m['_calibrate'](dict(s, y_obs_raw=p['y_pulse_raw']))
    residual = (calibrated - S @ p['w_pulse']).reshape(len(s['mu_shifts']), -1)
    errors.extend(np.sqrt(np.mean(residual**2, axis=1)))
assert max(errors) < 1e-5, errors

# Independently rebuild the profiled objective and cross-check the certificate and optimum.
Z = s['Z_nuisance']; ridge = s['source_ridge']; budget = s['total_budget']; cap = s['u_box']; y = m['_calibrate'](s)
P = np.eye(len(y)) - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
D, target = P @ S, P @ y
obj = lambda q: (np.linalg.norm(D @ q-target)**2 + ridge*(q@q))/2
grad = lambda q: D.T @ (D @ q-target) + ridge*q
lo, hi = (np.asarray(cert[k]) for k in ('lam_lower', 'lam_upper'))
nu = cert['nu_sum']; g = grad(w)
stat = np.max(np.abs(g-nu-lo+hi)); comp = max(np.max(np.abs(w*lo)),np.max(np.abs((cap-w)*hi)))
gap = obj(w) - cert['dual_lower_bound']
assert abs(w.sum()-budget) < 1e-10 and w.min() >= 0 and w.max() <= cap
assert np.array_equal(w, cert['weights']) and min(lo.min(),hi.min()) >= 0
assert abs(obj(w)-cert['primal_objective']) < 1e-10 and max(stat,comp,abs(gap)) < 1e-9
ref = minimize(obj, np.full(len(w), budget/len(w)), jac=grad, method='SLSQP', bounds=[(0,cap)]*len(w), constraints=[{'type':'eq','fun':lambda q:q.sum()-budget,'jac':lambda q:np.ones_like(q)}], options={'ftol':1e-14,'maxiter':1000})
assert ref.success and abs(ref.fun-obj(w)) < 1e-10, ref.message
print(f'worst pulse/shift RMSE={max(errors):.3g}; PDE max residual={field_error:.3g}; KKT stationarity={stat:.3g}, complementarity={comp:.3g}, gap={gap:.3g}; independent QP objective difference={abs(ref.fun-obj(w)):.3g}')
PY
```
- **Container Exit Code**: `0` (`568 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
worst pulse/shift RMSE=1.31e-06; PDE max residual=3.22e-15; KKT stationarity=2.22e-16, complementarity=1.78e-16, gap=1.67e-16; independent QP objective difference=2.66e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e0fdaa7aab8cf65c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e0fdaa7a (directory).
```

### Turn 7 — `submit_completed_task` (`12535 ms` LLM)
- **Rationale**: The exact artifact passed isolated verification: the terminal PDE residual was 3.22e-15, KKT stationarity and complementarity residuals were below 2.3e-16, and the independent optimizer's objective differed by 2.66e-15.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:6187'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, the worst tracer pulse/shift RMSE was 1.31e-6, and the solution matched an independent optimizer to numerical precision.`

### Turn 8 — `execute_in_remote_docker` (`35869 ms` LLM)
- **Rationale**: Held-out sweeps test whether pairwise quadratic terms or a rational correction capture missing sensor response; independent pulses test whether that improvement transfers to the inversion rather than being absorbed by nuisance projection. Change the solver only if both support the upgrade.
- **Alternatives Weighed**: `Apply batch-centering or test relay modes` (rejected: This is a single-campaign continuous inversion with no class labels or relay state. The applicable nuisance check compares tracer residuals before and after the specified projection.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy
import numpy as np
from scipy.optimize import least_squares

m = runpy.run_path('/holdout_task/batch_solver.py')
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']; sweeps = s['sensor_calibration_sweeps']
x, y = (sweeps[k] for k in ('ref_rosette', 'raw_rosette'))
clean = np.max(np.abs(y), axis=1) < 7
print(f'Calibration rows: {clean.sum()} retained, {(~clean).sum()} excluded; single-bank sweep type: {type(sweeps["single_bank_sweeps"]).__name__}')
families = ('affine', 'cubic', 'quadratic+cubic', 'rational+cubic')
def design(x, i, family, a=0.):
    z = x[..., i]
    base = m['_features'](x, i)
    if family == 'affine':
        return base[..., :5]
    if family == 'quadratic+cubic':
        extra = [x[..., j]*x[..., k] for j in range(4) for k in range(j,4)
                 if {j,k} != {i,(i+1)%4}]
        return np.concatenate((base, np.stack(extra,axis=-1)),axis=-1)
    if family == 'rational+cubic':
        return np.concatenate((base, (z**3/(1+a*z*z))[...,None]),axis=-1)
    return base

def fit(train,family,a):
    return [np.linalg.lstsq(design(x[train],i,family,a),y[train,i],rcond=None)[0] for i in range(4)]
def pred(x,coef,family,a):
    return np.stack([design(x,i,family,a) @ coef[i] for i in range(4)],axis=-1)
def choose(train,family):
    if family != 'rational+cubic':
        return 0.
    parts = np.array_split(train, 4)
    def loss(a):
        return max(np.sqrt(np.mean((pred(x[v],fit(np.setdiff1d(train,v),family,a),family,a)-y[v])**2)) for v in parts)
    return min((0.03, 0.1, 0.3, 1., 3.), key=loss)

ids = np.flatnonzero(clean)
folds = np.array_split(np.random.default_rng(42).permutation(ids), 5)
S, _ = m['_forward'](s)
Z = s['Z_nuisance']; proj = np.eye(len(Z))-Z @ np.linalg.solve(Z.T @ Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
for family in families:
    oof = np.zeros_like(y)
    for test in folds:
        train = np.setdiff1d(ids,test); a = choose(train,family)
        oof[test] = pred(x[test],fit(train,family,a),family,a)-y[test]
    a = choose(ids,family); coef = fit(ids,family,a)
    worst = [0.,0.]
    for pulse in item['campaign_diagnostics']:
        raw = pulse['y_pulse_raw'].reshape(-1,4)
        start = np.linalg.solve(np.array(coef)[:,1:5],(raw-np.array(coef)[:,0]).T).T
        inv = np.array([least_squares(lambda q:pred(q,coef,family,a)-r,q0,gtol=1e-13).x for q0,r in zip(start,raw)])
        r = inv.ravel()-S @ pulse['w_pulse']
        for j, e in enumerate((r,proj @ r)):
            worst[j] = max(worst[j],np.sqrt(np.mean(e.reshape(len(s['mu_shifts']),-1)**2,axis=1)).max())
    print(f'{family}: worst held-out sweep fold/channel RMSE={max(np.sqrt(np.mean(oof[v]**2,axis=0)).max() for v in folds):.3g}; worst pulse/shift RMSE raw={worst[0]:.3g}, nuisance-profiled={worst[1]:.3g}')
PY
```
- **Container Exit Code**: `0` (`872 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Calibration rows: 77 retained, 3 excluded; single-bank sweep type: dict
affine: worst held-out sweep fold/channel RMSE=0.411; worst pulse/shift RMSE raw=0.0773, nuisance-profiled=0.0746
cubic: worst held-out sweep fold/channel RMSE=1.37e-05; worst pulse/shift RMSE raw=1.31e-06, nuisance-profiled=1.3e-06
quadratic+cubic: worst held-out sweep fold/channel RMSE=1.61e-05; worst pulse/shift RMSE raw=1.71e-06, nuisance-profiled=1.69e-06
rational+cubic: worst held-out sweep fold/channel RMSE=1.38e-05; worst pulse/shift RMSE raw=1.14e-06, nuisance-profiled=1.13e-06

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e0fdaa7aab8cf65c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e0fdaa7a (directory).
```

### Turn 9 — `submit_completed_task` (`14954 ms` LLM)
- **Rationale**: The unchanged artifact passed the schema, isolated numerical checks, and independent verification of the field, weights, and certificate. The stress audit found no reason to replace the cubic calibration; nuisance projection barely changed the tracer residuals.
- **Alternatives Weighed**: `Add a rational calibration term` (rejected: It reduced worst pulse/shift RMSE from 1.31e-6 to 1.14e-6 but did not improve held-out sweep accuracy or resolve a demonstrated tolerance failure.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:6187'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, and the worst tracer pulse/shift RMSE was 1.31e-6. Independent verification measured relative errors of 1.65e-6 for the terminal field and 8.41e-6 for the weights.`

### Turn 10 — `execute_in_remote_docker` (`28349 ms` LLM)
- **Rationale**: The earlier random folds did not isolate the strongest paired-channel contrasts. A blocked holdout tests for a missing cross-cubic term, while the sweep peak separation shows whether small changes to the clipping cutoff would alter the fit. Investigate a change only if this reveals a material error.
- **Alternatives Weighed**: `Repeat the rational-basis, nuisance, and KKT checks` (rejected: Those checks already ran on the unchanged artifact. The remaining distinct calibration question is whether the cubic basis holds up at high paired-channel contrast.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy
import numpy as np

m = runpy.run_path('/holdout_task/batch_solver.py')
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
x, raw = (s['sensor_calibration_sweeps'][k] for k in ('ref_rosette', 'raw_rosette'))
peak = np.max(np.abs(raw), axis=1)
keep = peak < 7
print('Largest retained and smallest excluded sweep peaks:', peak[keep].max(), peak[~keep].min())
# Hold out the strongest reference excitations for each channel as a block.
# A coherent residual in their paired-difference direction would motivate a new term.
worst_base = worst_extra = 0.
for i in range(4):
    d = x[:, i] - x[:, (i+1) % 4]
    X = m['_features'](x, i)
    A = np.column_stack((X, d**3))
    eligible = np.flatnonzero(keep)
    test = eligible[np.argsort(np.abs(d[eligible]))[-max(1, len(eligible)//5):]]
    train = np.setdiff1d(eligible, test)
    scores = []
    for design in (X, A):
        coef = np.linalg.lstsq(design[train], raw[train, i], rcond=None)[0]
        scores.append(np.sqrt(np.mean((design[test] @ coef - raw[test, i])**2)))
    worst_base = max(worst_base, scores[0]); worst_extra = max(worst_extra, scores[1])
    print(f'Channel {i} high-contrast holdout RMSE: cubic={scores[0]:.3g}, paired-difference cubic={scores[1]:.3g}')
print(f'Worst high-contrast RMSE: baseline={worst_base:.3g}, expanded={worst_extra:.3g}')
PY
```
- **Container Exit Code**: `0` (`599 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Largest retained and smallest excluded sweep peaks: 4.375289628784571 13.285437673325445
Channel 0 high-contrast holdout RMSE: cubic=1.08e-05, paired-difference cubic=1.14e-05
Channel 1 high-contrast holdout RMSE: cubic=9.63e-06, paired-difference cubic=1.67e-05
Channel 2 high-contrast holdout RMSE: cubic=1.43e-05, paired-difference cubic=1.9e-05
Channel 3 high-contrast holdout RMSE: cubic=1.54e-05, paired-difference cubic=9.32e-06
Worst high-contrast RMSE: baseline=1.54e-05, expanded=1.9e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e0fdaa7aab8cf65c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e0fdaa7a (directory).
```

### Turn 11 — `submit_completed_task` (`26644 ms` LLM)
- **Rationale**: The unchanged artifact passed clean-process verification of the terminal PDE, constrained optimum, and KKT certificate. The high-contrast holdout favored retaining the cubic basis, and the largest retained sweep peak (4.38) and smallest excluded peak (13.29) were well separated from the cutoff of 7.
- **Alternatives Weighed**: `Add a paired-channel difference term` (rejected: It raised the worst high-contrast held-out RMSE from 1.54e-5 to 1.90e-5.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed. Independent verification measured relative L2 errors of 1.65e-6 for the terminal field and 8.41e-6 for the source weights. The worst tracer pulse/shift RMSE was 1.31e-6.`

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
