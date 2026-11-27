# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_v2_c04_batch_04__blind_continual_campaign`

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
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `337,489` (`325,752` / `107,008` / `11,737`) |
| **Total Episode Wall Time (s)** | `380.87s` |

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

### Turn 1 — `execute_in_remote_docker` (`17865 ms` LLM)
- **Rationale**: The field and log summaries detect a wrong input-schema assumption or a boundary-regime change. Either would change which archived implementation to inspect and reuse.
- **Alternatives Weighed**: `Implement the stated open-wall baseline immediately` (rejected: A blind campaign may use a different boundary operator; inspect the apparatus record before choosing one.)

```bash
cd /holdout_task && ls -l /tmp/hegel_continual_archive && cat public_asset_index.json validate_public_schema.py && python3 - <<'PY'
import pickle, numpy as np, json
item=pickle.load(open('public_batch.pkl','rb'))[0]
print('item keys', list(item), 'diagnostics', None if item.get('campaign_diagnostics') is None else len(item['campaign_diagnostics']))
for k,v in item['study'].items():
 print(k, (v.shape, str(v.dtype)) if isinstance(v,np.ndarray) else list(v) if isinstance(v,dict) else (type(v).__name__,len(v)) if isinstance(v,(list,tuple)) else v)
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
with open(p) as f: print('apparatus log',json.dumps(json.load(f))[:14000])
PY
```
- **Container Exit Code**: `0` (`418 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.46844, 1.02957, -0.24534, 0.82096, 1.16164], [0.68865, -0.47067, -0.05205, 2.43675, 1.03425, -0.24843, 0.8803, 1.16854], [0.44084, -0.40902, -0.38359, 2.62599, 1.03368, 0.04918, 0.36004, 0.8026], [0.04303, -0.30678, -0.08788, 2.45931, 1.25955, -0.16333, 0.04979, 1.08877], [-0.48813, -0.29643, -0.3089, 2.60469, 1.42567, -0.13442, -0.2926, 1.32993], [-0.34461, -0.39056, -0.19544, 2.63263, 1.15964, -0.1031, -0.75947, 1.25482], [-0.95325, -0.41078, -0.2184, 2.46724, 0.98501, -0.11442, -0.91069, 1.17837], [-1.02651, -0.39194, -0.22023, 2.4927, 0.92018, -0.14334, -0.64577, 0.95356], [-1.00831, -0.2932, -0.27375, 2.34929, 1.10586, -0.24387, -0.75931, 1.0885], [-0.53718, -0.21667, -0.19107, 2.58718, 1.1542, -0.05801, -0.57505, 1.06765], [-0.33157, -0.34765, -0.36956, 2.45353, 1.42404, -0.34166, -0.26324, 1.39784], [0.19203, -0.37259, 0.00346, 2.63, 1.10718, -0.25965, 0.16073, 1.19583], [0.59052, -0.3615, -0.12915, 2.54946, 1.39581, -0.03266, 0.58751, 0.99916], [1.00216, -0.2123, -0.14781, 2.47212, 1.08049, -0.43137, 0.93937, 1.22784], [1.00329, -0.31847, -0.24661, 2.62077, 0.95942, -0.2005, 1.02741, 1.30136]]}, {"completed_batch_index": 3, "completed_short_name": "beta_m98_v2_c04_batch_03", "facility_telemetry_trace": [[0.7055, -0.33512, 0.24789, -0.41102, -1.15322, -0.18807, 0.85927, -1.04281], [0.64848, -0.24521, 0.31086, -0.30393, -1.27341, -0.24127, 0.54556, -1.08], [0.18459, -0.39284, 0.22619, -0.32843, -1.08541, 0.02193, 0.16827, -1.21797], [-0.18077, -0.26581, -0.11593, -0.38213, -1.10172, 0.15471, 0.00775, -1.15518], [-0.58251, -0.34685, -0.23082, -0.38136, -1.197, -0.03978, -0.41592, -1.26305], [-0.72692, -0.43541, -0.23542, -0.40475, -1.19012, 0.04662, -0.82594, -0.99422], [-0.88409, -0.3218, -0.03301, -0.53163, -1.22009, -0.08394, -1.11553, -1.0617], [-0.60583, -0.40567, -0.16927, -0.42556, -1.12834, -0.12648, -0.86856, -1.49371], [-1.00614, -0.38964, -0.26156, -0.3554, -1.0684, -0.08427, -0.68388, -1.16937], [-0.35511, -0.37793, -0.09219, -0.45192, -1.11256, 0.01201, -0.29205, -0.90868], [-0.1803, -0.33324, -0.09781, -0.28228, -1.00426, -0.17118, 0.12874, -1.29811], [0.57572, -0.47103, -0.06708, -0.40152, -1.14977, -0.09363, 0.0375, -1.21522], [0.78053, -0.26643, 0.26406, -0.34482, -1.17306, 0.08379, 0.62595, -1.19666], [0.81875, -0.31027, 0.07313, -0.49361, -1.11646, -0.2346, 0.66879, -1.40237], [1.1624, -0.38266, 0.08944, -0.37319, -1.19915, 0.09185, 0.84017, -1.25825], [0.84121, -0.26822, 0.1086, -0.37631, -1.32892, -0.03798, 0.57064, -1.25285]]}]
```

### Turn 2 — `execute_in_remote_docker` (`18691 ms` LLM)
- **Rationale**: Compact telemetry summaries test for an incorrect regime match; source diffs show which operator terms would change and whether calibration or optimization can be reused unchanged.
- **Alternatives Weighed**: `Reuse the latest archived solver unchanged` (rejected: The latest log entry is the signal to match against older entries; it does not establish that the latest solver has the right boundary rows.)

```bash
cd /holdout_task && cat validate_public_schema.py /tmp/hegel_continual_archive/archive_manifest.json && cat /tmp/hegel_continual_archive/beta_m98_v2_c04_batch_03_solver.py && for s in /tmp/hegel_continual_archive/*solver.py; do echo "=== $s vs batch 03 ==="; diff -u /tmp/hegel_continual_archive/beta_m98_v2_c04_batch_03_solver.py "$s" || true; done
python3 - <<'PY'
import json, numpy as np
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
a=json.load(open(p))
if isinstance(a,dict):
 print('log keys:',list(a)); a=next(v for v in a.values() if isinstance(v,list) and v and isinstance(v[0],dict) and 'facility_telemetry_trace' in v[0])
means=[]
for row in a:
 x=np.array(row['facility_telemetry_trace']); means.append(x.mean(axis=0)); print(row['completed_short_name'], x.shape, 'mean',means[-1].round(3),'sd',x.std(axis=0).round(3))
print('distance from latest:',np.linalg.norm(np.array(means[:-1])-means[-1],axis=1).round(3))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling' || true
```
- **Container Exit Code**: `0` (`460 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
array(velocity)
+def _axis(kappa, velocity, length, robin):
     n = len(kappa)
     h = length / (n - 1)
-    left = -(kappa + np.roll(kappa, 1)) / (2 * h**2) - np.maximum(velocity, 0) / h
-    right = -(kappa + np.roll(kappa, -1)) / (2 * h**2) - np.maximum(-velocity, 0) / h
-    seam = diags(([left[0]], [right[-1]]), (n - 1, 1 - n), shape=(n, n), format='csc')
-    return diags((left[1:], -left - right, right[:-1]), (-1, 0, 1), format='csc') + seam
+    faces = (kappa[:-1] + kappa[1:]) / (2 * h**2)
+    lower = -faces - np.maximum(velocity[1:], 0) / h
+    upper = -faces - np.maximum(-velocity[:-1], 0) / h
+    upper[0], lower[-1] = -kappa[0] / h**2, -kappa[-1] / h**2
+    main = np.r_[-upper[0] + robin / h,
+                 -lower[:-1] - upper[1:],
+                 -lower[-1] + robin / h]
+    return diags((lower, main, upper), (-1, 0, 1), format='csc')
 
 
 def _forward(study):
     nx, ny = study['nx'], study['ny']
-    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'])
-    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'])
+    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'], study['robin_x'])
+    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'], study['robin_y'])
     reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
     op = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
     op += diags(reaction, format='csc')
beta_m98_v2_c04_s00 (16, 8) mean [-0.007  2.435 -0.676 -0.381  1.136 -0.703 -0.018  1.178] sd [0.633 0.075 0.181 0.089 0.127 0.104 0.665 0.137]
beta_m98_v2_c04_batch_01 (16, 8) mean [ 0.093 -0.35  -0.479 -0.397 -1.203 -0.345 -0.024 -1.133] sd [0.648 0.055 0.116 0.079 0.102 0.13  0.61  0.15 ]
beta_m98_v2_c04_batch_02 (16, 8) mean [ 0.069 -0.344 -0.213  2.522  1.141 -0.184  0.093  1.15 ] sd [0.728 0.073 0.109 0.084 0.156 0.117 0.679 0.145]
beta_m98_v2_c04_batch_03 (16, 8) mean [ 7.500e-02 -3.470e-01  1.000e-03 -3.900e-01 -1.156e+00 -5.600e-02
  1.600e-02 -1.188e+00] sd [0.696 0.064 0.185 0.063 0.077 0.115 0.63  0.144]
distance from latest: [4.413 0.566 4.393]
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 3 — `execute_in_remote_docker` (`15182 ms` LLM)
- **Rationale**: Read the matched solver without output truncation to identify its exact boundary rows and check for campaign-specific calibration or optimization assumptions before reuse.
- **Alternatives Weighed**: `Copy the batch-01 solver without inspection` (rejected: The trace selects a boundary regime, but calibration and the equality-constrained optimizer must still work from the current study alone.)

```bash
cd /holdout_task && cat /tmp/hegel_continual_archive/beta_m98_v2_c04_batch_01_solver.py && for s in /tmp/hegel_continual_archive/*solver.py; do echo "=== $(basename "$s") ==="; sed -n '/^def _axis/,/^def _forward/p' "$s"; done && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```
- **Container Exit Code**: `0` (`371 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
upper[0], lower[-1] = -kappa[0] / h**2, -kappa[-1] / h**2
    main = np.r_[-upper[0] + robin / h,
                 -lower[:-1] - upper[1:],
                 -lower[-1] + robin / h]
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
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

### Turn 4 — `execute_in_remote_docker` (`25500 ms` LLM)
- **Rationale**: Save the matched sealed-wall solver. The schema check detects output or certificate failures, held-out sweeps can expose an inadequate affine calibration, and the archived objective comparison can flag a boundary match that warrants further testing.
- **Alternatives Weighed**: `Use the sparse L0 ridge primitive` (rejected: It does not impose the required fixed-sum constraint, and the objective has no L0 penalty.)

```bash
set -e
cd /holdout_task
cp /tmp/hegel_continual_archive/beta_m98_v2_c04_batch_01_solver.py batch_solver.py
sed -i 's/The tracer pulses identify zero wall leakage on both axes./The apparatus log matches batch 01: zero wall leakage on both axes./' batch_solver.py
sed -n '42,155p' batch_solver.py
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pathlib, pickle
import numpy as np
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']
a=s['sensor_calibration_sweeps']; x,y=a['ref_rosette'],a['raw_rosette']
for k,v in a.items():
 print('sweep',k,type(v).__name__,getattr(v,'shape',None),list(v) if isinstance(v,dict) else '')
 if k=='single_bank_sweeps':
  for z in (list(v.values()) if isinstance(v,dict) else list(v))[:4]: print(' bank', {k:(type(e).__name__,getattr(e,'shape',None)) for k,e in z.items()} if isinstance(z,dict) else np.shape(z))
print('banks',np.unique(s['row_bank_id'],return_counts=True),'sources',s['source_modes'].shape)
for inds in np.array_split(np.arange(len(x)),4):
 train=np.ones(len(x),bool); train[inds]=False
 d=np.c_[np.ones(len(x)),x]; coeff=np.linalg.lstsq(d[train],y[train],rcond=None)[0]; e=(d@coeff-y)[inds]; print('held calibration MAE',np.mean(abs(e),axis=0).round(8),'mix condition',round(np.linalg.cond(coeff[1:]),4))
for p in sorted(pathlib.Path('/tmp/hegel_continual_archive').glob('*solver.py')):
 spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
 o=m.solve_batch(s); k=o['kkt_certificate']; print(p.name,'objective',k['primal_objective'],'gap',k['primal_objective']-k['dual_lower_bound'],'weights',np.round(o['weights'],5))
PY
```
- **Container Exit Code**: `0` (`652 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
tudy)
    z = np.asarray(study['Z_nuisance'])
    values = np.column_stack((design, calibrated))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    projected = values - z @ np.linalg.solve(gram, z.T @ values)
    x, y = projected[:, :-1], projected[:, -1]
    ridge = float(study['source_ridge'])
    cap, budget = float(study['u_box']), float(study['total_budget'])
    w, nu = _box_budget(x.T @ x + ridge * np.eye(x.shape[1]), -x.T @ y, cap, budget)
    residual = x @ w - y
    g = x.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    delta = g - nu
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(),
            'nu_sum': nu,
            'lam_lower': np.maximum(delta, 0).tolist(),
            'lam_upper': np.maximum(-delta, 0).tolist(),
            'primal_objective': objective,
            'dual_lower_bound': float(objective - g @ w + nu * budget + cap * np.minimum(delta, 0).sum()),
        },
    }
{"schema_valid": true, "errors": []}
sweep ref_rosette ndarray (80, 4) 
sweep raw_rosette ndarray (80, 4) 
sweep single_bank_sweeps dict None ['bank_id', 'ref_signal', 'raw_readout']
 bank (24,)
 bank (24,)
 bank (24,)
banks (array([0, 1, 2, 3]), array([24, 24, 24, 24])) sources (8, 14, 12)
held calibration MAE [2.08052618 2.0008474  1.95398426 2.0276816 ] mix condition 2.044
held calibration MAE [0.70179085 0.64274622 0.69926308 0.63700101] mix condition 6.4339
held calibration MAE [0.67710544 0.76372606 0.65094177 0.65357731] mix condition 11.5265
held calibration MAE [0.70967389 0.6293293  0.72935499 0.75876196] mix condition 6.2593
beta_m98_v2_c04_batch_01_solver.py objective 41.98455294835281 gap 0.0 weights [0.37686 0.18314 0.      0.42    0.      0.42    0.      0.     ]
beta_m98_v2_c04_batch_02_solver.py objective 40.98566661131354 gap 0.0 weights [0.42 0.42 0.   0.   0.14 0.42 0.   0.  ]
beta_m98_v2_c04_batch_03_solver.py objective 42.277758048364504 gap 0.0 weights [0.42 0.42 0.   0.   0.14 0.42 0.   0.  ]
beta_m98_v2_c04_s00_solver.py objective 40.98566661131354 gap 0.0 weights [0.42 0.42 0.   0.   0.14 0.42 0.   0.  ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a880c8d10be867d0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a880c8d1 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`20873 ms` LLM)
- **Rationale**: Inspect the sweep design and residual pattern to distinguish missing calibration terms from a separate sweep regime. That distinction determines whether to expand the calibration model or fit it on the appropriate sweep subset.
- **Alternatives Weighed**: `Select the boundary operator by the current residual objective` (rejected: The held-out calibration errors make those residuals unreliable for comparing boundary operators.)

```bash
cd /holdout_task && python3 -I - <<'PY'
import pickle,numpy as np,json
np.set_printoptions(precision=4,suppress=True,linewidth=150)
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; print('keys',list(s)); a=s['sensor_calibration_sweeps']; x,y=a['ref_rosette'],a['raw_rosette']; print('ref/raw first 10',np.c_[x,y][:10]); print('ref/raw quantiles',np.quantile(np.c_[x,y],[0,.1,.5,.9,1],axis=0)); print('raw obs quantiles',np.quantile(s['y_obs_raw'].reshape(-1,4),[0,.1,.5,.9,1],axis=0)); print('bank sweep',np.column_stack(list(a['single_bank_sweeps'].values()))); print('bank layout',s['row_bank_id'].reshape(-1,4)[:8]); print('aux stats',s['aux_manifold_telemetry'].mean(0),s['aux_manifold_telemetry'].std(0)); print('log top keys',list(json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')).keys())); d=np.c_[np.ones(len(x)),x]; c=np.linalg.lstsq(d,y,rcond=None)[0]; e=y-d@c; print('row residual norms',np.linalg.norm(e,axis=1).reshape(-1,10)); print('coeff',c)
PY
```
- **Container Exit Code**: `1` (`390 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calibration_sweeps', 'Z_nuisance', 'y_obs_raw', 'aux_manifold_telemetry', 'nuisance_ridge', 'source_ridge', 'u_box', 'total_budget']
ref/raw first 10 [[-1.2     0.      0.      0.     -1.299  -0.1527  0.08    0.2015]
 [-0.4     0.      0.      0.     -0.3974 -0.0043  0.08    0.0205]
 [ 0.5     0.      0.      0.      0.3685  0.1628  0.08   -0.1831]
 [ 1.3     0.      0.      0.      1.3303  0.3113  0.08   -0.3641]
 [ 0.     -1.2     0.      0.      0.2254 -1.8641 -0.1497 -0.07  ]
 [ 0.     -0.4     0.      0.      0.0351 -0.4567  0.0035 -0.07  ]
 [ 0.      0.5     0.      0.     -0.1789  0.7388  0.1757 -0.07  ]
 [ 0.      1.3     0.      0.     -0.3691  2.2402  0.3288 -0.07  ]
 [ 0.      0.     -1.2     0.     -0.06    0.5154 -1.9146 -0.2057]
 [ 0.      0.     -0.4     0.     -0.06    0.2185 -0.4632 -0.1152]]
ref/raw quantiles [[ -1.3905  -1.3893  -1.396   -1.3644  -1.6487 -12.73    -2.2253 -13.591 ]
 [ -0.9834  -1.1285  -1.0708  -1.0522  -0.9892  -1.9049  -1.6589  -1.1102]
 [  0.4171   0.0339   0.       0.       0.2864   0.1183   0.0852  -0.1891]
 [  1.2788   1.2982   1.424    1.1372   1.5454   2.2323   3.082    0.8559]
 [  1.6737   1.6772   1.7145   1.7538  13.5946   4.1476  12.6331   2.2779]]
raw obs quantiles [[-0.2427 -0.0954 -0.2028 -0.2044]
 [-0.1527 -0.0062 -0.0285 -0.1548]
 [-0.0056  0.1631  0.1005 -0.0145]
 [ 0.1389  0.462   0.3997  0.0933]
 [ 0.2579  0.6179  0.7063  0.1657]]
bank sweep [[ 0.     -1.1    -1.1585]
 [ 0.     -0.6    -0.5839]
 [ 0.     -0.1    -0.1422]
 [ 0.      0.4     0.2774]
 [ 0.      0.9     0.7856]
 [ 0.      1.4    15.993 ]
 [ 1.     -1.1    -1.6446]
 [ 1.     -0.6    -0.7478]
 [ 1.     -0.1    -0.0582]
 [ 1.      0.4     0.5967]
 [ 1.      0.9     1.39  ]
 [ 1.      1.4     2.4942]
 [ 2.     -1.1    -1.6882]
 [ 2.     -0.6    -0.7633]
 [ 2.     -0.1    -0.0522]
 [ 2.      0.4     0.6232]
 [ 2.      0.9     1.4412]
 [ 2.      1.4     2.58  ]
 [ 3.     -1.1    -1.1149]
 [ 3.     -0.6    -0.5683]
 [ 3.     -0.1    -0.1481]
 [ 3.      0.4     0.251 ]
 [ 3.      0.9     0.7344]
 [ 3.      1.4     1.4072]]
bank layout [[0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]]
aux stats [ 0.1115 -0.0855 -0.1632 -0.2134] [0.6944 1.1478 0.8303 1.1456]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a880c8d10be867d0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a880c8d1 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'keys'
```

### Turn 6 — `execute_in_remote_docker` (`29442 ms` LLM)
- **Rationale**: Held-out residuals and independent bank-only fits test whether robust fitting alone suffices or whether nonlinear and cross-channel terms are needed before comparing wall operators.
- **Alternatives Weighed**: `Discard the single-axis sweeps as a separate regime` (rejected: The single-axis sweeps isolate curvature and cross-talk, helping distinguish missing terms from bad readings.)
- **ATMS NOGOOD DECLARED**: Refuted `A global affine rosette map explains the calibration sweeps.` | Witness: `Channel 0's isolated-input sweep has slopes of 1.127, 0.851, and 1.202 over consecutive intervals; a separate bank-only readout jumps to 15.993 at reference 1.4.` | Blanket Diagnosis: `Systematic curvature and gross sweep errors are both being absorbed into the estimated affine mixing matrix.` | Sublated Pivot: `Compare robust affine, diagonal cubic, and cross-channel polynomial fits on held-out sweeps.`

```bash
cd /holdout_task && cat >/tmp/rosette_cv.py <<'PY'
import pickle,numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['sensor_calibration_sweeps']; x,y=a['ref_rosette'],a['raw_rosette']; np.set_printoptions(precision=6,suppress=True,linewidth=160)
print('single-axis count',np.unique(np.count_nonzero(x,axis=1),return_counts=True))
def features(x,j,k):
 cols=[np.ones(len(x)),*x.T]
 if k=='diag3': cols.append(x[:,j]**3)
 if k=='all3': cols.extend((x**3).T)
 if k=='full3': cols.extend(np.prod(x[:,ids],axis=1) for d in (2,3) for ids in itertools.combinations_with_replacement(range(4),d))
 return np.column_stack(cols)
def fit(d,y,k,n):
 mask=np.ones(len(y),bool)
 for _ in range(9):
  c=np.linalg.lstsq(d[mask],y[mask],rcond=None)[0]; r=abs(y-d@c); sel=np.argsort(r)[:max(d.shape[1]+2,len(y)-n)]; m=np.zeros(len(y),bool); m[sel]=True
  if np.array_equal(m,mask): break
  mask=m
 return c
for k in ('affine','diag3','all3','full3'):
 print('model',k)
 for j in range(4):
  d=features(x,j,k); errors=np.zeros(len(y));
  for test in np.array_split(np.random.default_rng(42).permutation(len(y)),5):
   train=np.setdiff1d(np.arange(len(y)),test); c=fit(d[train],y[train,j],k,7); errors[test]=y[test,j]-d[test]@c
  c=fit(d,y[:,j],k,8); r=y[:,j]-d@c; out=np.argsort(abs(r))[-8:]; print(j,'OOF abs quantiles',np.quantile(abs(errors),[.5,.8,.9,.95,1]),'coef',c if k!='full3' else '', 'largest residuals',list(zip(out,r[out].round(5))))
b=a['single_bank_sweeps']; print('bank-only robust cubic fits');
for j in range(4):
 v=b['bank_id']==j; u=b['ref_signal'][v]; q=b['raw_readout'][v]; d=np.c_[np.ones(len(u)),u,u**3]; c=fit(d,q,'',1); print(j,c,q-d@c)
PY
python3 -I /tmp/rosette_cv.py
```
- **Container Exit Code**: `0` (`463 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.32744), (15, 13.4094), (14, 13.46069), (13, 13.50088)]
1 OOF abs quantiles [ 0.137383  0.397113  0.594326  0.709446 12.809201] coef [ 0.069906  0.139111  1.473714 -0.292131 -0.045209  0.012017  0.108056 -0.106366  0.028801] largest residuals [(45, 0.40254), (50, 0.45556), (57, 0.58918), (41, 0.62097), (26, 0.72165), (14, -12.78091), (15, -12.80443), (13, -12.81616)]
2 OOF abs quantiles [ 0.130873  0.386502  0.468563  0.704401 12.408889] coef [ 0.111302 -0.143221  0.226633  1.19267  -0.347408  0.063323 -0.030992  0.35738  -0.010051] largest residuals [(61, 0.40576), (33, -0.45125), (63, -0.46295), (16, 0.65335), (45, 0.75306), (15, 12.34479), (14, 12.35227), (13, 12.38221)]
3 OOF abs quantiles [ 0.062585  0.148691  0.249862  0.440917 13.219563] coef [-0.091442 -0.233865  0.04443   0.040986  0.739692  0.027243 -0.024487  0.041956  0.106683] largest residuals [(16, 0.24079), (39, -0.31455), (17, 0.32494), (52, -0.42992), (45, 0.59523), (15, -13.05208), (14, -13.1542), (13, -13.19684)]
model full3
0 OOF abs quantiles [ 0.000011  0.000024  0.00004   0.000063 13.500012] coef  largest residuals [(6, 2e-05), (10, -2e-05), (5, 2e-05), (53, -3e-05), (61, -3e-05), (14, 13.49999), (15, 13.5), (13, 13.50001)]
1 OOF abs quantiles [ 0.000012  0.000025  0.000031  0.000044 12.800029] coef  largest residuals [(10, -2e-05), (58, -2e-05), (77, -2e-05), (20, 2e-05), (1, -3e-05), (14, -12.80001), (13, -12.80002), (15, -12.80002)]
2 OOF abs quantiles [ 0.000011  0.000026  0.000035  0.000043 12.400002] coef  largest residuals [(22, -2e-05), (37, 2e-05), (7, -2e-05), (23, -3e-05), (76, 3e-05), (13, 12.39999), (14, 12.4), (15, 12.4)]
3 OOF abs quantiles [ 0.000013  0.000021  0.000042  0.000054 13.200016] coef  largest residuals [(70, 1e-05), (5, 1e-05), (1, 2e-05), (37, -2e-05), (43, 2e-05), (15, -13.2), (13, -13.2), (14, -13.2)]
bank-only robust cubic fits
0 [-0.446672 -3.1199    7.470838] [ 5.800015 -0.395453 -0.000001  1.493936 -1.406058  0.307575]
1 [0.070007 1.279994 0.230399] [ 0.000003 -0.000001 -0.00002  -0.000007  0.000006 -0.      ]
2 [0.080004 1.319983 0.237607] [-0.000004  0.000006 -0.000013 -0.000006  0.000006 -0.000003]
3 [-0.070002  0.780001  0.140397] [-0.000005  0.000006  0.000013  0.        0.000001 -0.000002]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a880c8d10be867d0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a880c8d1 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`42919 ms` LLM)
- **Rationale**: Fit the supported cubic and reject grossly corrupted sweep rows. Blocked sweeps test whether the fitted map transfers across sweep designs; the inverse round trip tests the observed range. Compare archived boundary objectives again only after fixing calibration.
- **Alternatives Weighed**: `Fit only independent bank cubics` (rejected: The diagonal and additive cubic fits leave held-out errors orders of magnitude larger than the full cubic.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('batch_solver.py')
s = p.read_text().replace('import numpy as np', 'from itertools import combinations_with_replacement\n\nimport numpy as np', 1)
a, b = s.index('def _calibrate('), s.index('def _box_budget(')
s = s[:a] + '''_TERMS = [()] + [(j,) for j in range(4)] + [
    ids for degree in (2, 3)
    for ids in combinations_with_replacement(range(4), degree)
]


def _rosette_basis(x):
    return np.column_stack([np.prod(x[:, ids], axis=1) for ids in _TERMS])


def _fit_rosette(ref, raw):
    basis = _rosette_basis(ref)
    keep = np.ones(len(ref), dtype=bool)
    for _ in range(10):
        coef = np.linalg.lstsq(basis[keep], raw[keep], rcond=None)[0]
        err = np.linalg.norm(raw - basis @ coef, axis=1)
        updated = err <= max(8 * np.median(err), 1e-10)
        if np.array_equal(keep, updated) or updated.sum() < basis.shape[1]:
            return coef
        keep = updated
    return np.linalg.lstsq(basis[keep], raw[keep], rcond=None)[0]


def _invert_rosette(raw, coef):
    x = np.linalg.solve(coef[1:5].T, (raw - coef[0]).T).T
    for _ in range(20):
        residual = _rosette_basis(x) @ coef - raw
        if np.max(np.abs(residual)) < 1e-12:
            return x
        jac = np.broadcast_to(coef[1:5].T, (len(x), 4, 4)).copy()
        for ids, c in zip(_TERMS[5:], coef[5:]):
            for k, j in enumerate(ids):
                factor = np.prod(x[:, ids[:k] + ids[k+1:]], axis=1)
                jac[:, :, j] += factor[:, None] * c
        x -= np.linalg.solve(jac, residual[..., None])[..., 0]
    raise RuntimeError('Rosette calibration inversion did not converge')


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    coef = _fit_rosette(np.asarray(sweeps['ref_rosette']),
                        np.asarray(sweeps['raw_rosette']))
    return _invert_rosette(np.asarray(study['y_obs_raw']).reshape(-1, 4), coef).ravel()


''' + s[b:]
p.write_text(s)
PY
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pathlib, pickle
import numpy as np

def load(p):
 spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
m=load('batch_solver.py'); s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; a=s['sensor_calibration_sweeps']; x,y=a['ref_rosette'],a['raw_rosette']; c=m._fit_rosette(x,y)
for ids,co in zip(m._TERMS,c):
 if np.max(abs(co))>0.001: print('term',ids,co.round(6))
e=y-m._rosette_basis(x)@c; print('row errors >0.01',[(int(i),round(np.linalg.norm(e[i]),4)) for i in np.flatnonzero(np.max(abs(e),axis=1)>.01)],'inlier max',np.max(abs(e[np.max(abs(e),axis=1)<.01])))
for name, groups in [('contiguous',np.array_split(np.arange(len(x)),4)),('random',np.array_split(np.random.default_rng(42).permutation(len(x)),5)),('all isolated',[np.flatnonzero(np.count_nonzero(x,axis=1)==1)])]:
 print(name)
 for ix in groups:
  tr=np.ones(len(x),bool); tr[ix]=False; cf=m._fit_rosette(x[tr],y[tr]); r=y[ix]-m._rosette_basis(x[ix])@cf; print('held max inlier',np.max(abs(r[np.max(abs(e[ix]),axis=1)<.01])))
obs=s['y_obs_raw'].reshape(-1,4); cal=m._invert_rosette(obs,c); print('obs inversion max',np.max(abs(m._rosette_basis(cal)@c-obs)),'cal range',cal.min(),cal.max());
for p in sorted(pathlib.Path('/tmp/hegel_continual_archive').glob('*solver.py')):
 a=load(p); a._calibrate=lambda _: cal.ravel(); o=a.solve_batch(s); k=o['kkt_certificate']; print(p.name,'objective',k['primal_objective'],'gap',k['primal_objective']-k['dual_lower_bound'],'weights',o['weights'].round(6))
PY
```
- **Container Exit Code**: `0` (`680 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
term () [-0.059994  0.070001  0.080005 -0.069997]
term (0,) [ 8.20008e-01  1.85602e-01 -1.00000e-06 -2.26198e-01]
term (1,) [-2.378100e-01  1.279994e+00  1.913930e-01  1.000000e-06]
term (2,) [-2.000000e-06 -3.711960e-01  1.319994e+00  1.130940e-01]
term (3,) [ 1.18891e-01 -9.00000e-06 -3.82796e-01  7.79995e-01]
term (0, 1) [-1.804e-01 -2.000e-06  3.000e-06 -5.000e-06]
term (0, 3) [-0.00000e+00 -0.00000e+00  6.00000e-06 -1.71602e-01]
term (1, 2) [ 2.00000e-06 -2.81596e-01  2.00000e-06  3.00000e-06]
term (2, 3) [-2.00000e-06  1.00000e-06 -2.90391e-01 -3.00000e-06]
term (0, 0, 0) [ 1.47595e-01  1.00000e-06  0.00000e+00 -5.00000e-06]
term (1, 1, 1) [ 6.00000e-06  2.30402e-01  3.00000e-06 -0.00000e+00]
term (2, 2, 2) [ 3.00000e-06 -1.00000e-06  2.37602e-01  2.00000e-06]
term (3, 3, 3) [6.00000e-06 5.00000e-06 2.00000e-06 1.40406e-01]
row errors >0.01 [(13, 25.9633), (14, 25.9632), (15, 25.9632)] inlier max 1.943988624973636e-05
contiguous
held max inlier 3.5618578088592034e-05
held max inlier 8.986602867971214
held max inlier 0.00010417093751469153
held max inlier 7.722606824289713e-05
random
held max inlier 3.8624213184873524e-05
held max inlier 5.805735386488209e-05
held max inlier 4.607258881761389e-05
held max inlier 4.099647536759399e-05
held max inlier 4.2450566954843616e-05
all isolated
held max inlier 3.310500315700706e-05
obs inversion max 1.1102230246251565e-16 cal range -0.1946089150011443 0.4274949221027177
beta_m98_v2_c04_batch_01_solver.py objective 0.42593668921604316 gap -2.220446049250313e-16 weights [0.       0.17465  0.106465 0.278885 0.       0.42     0.42     0.      ]
beta_m98_v2_c04_batch_02_solver.py objective 0.008249949080366458 gap 1.1102230246251565e-16 weights [0.159387 0.203236 0.2648   0.092717 0.151047 0.125668 0.106981 0.296164]
beta_m98_v2_c04_batch_03_solver.py objective 0.5497271654854053 gap 3.3306690738754696e-16 weights [0.       0.42     0.       0.107137 0.       0.42     0.42     0.032863]
beta_m98_v2_c04_s00_solver.py objective 0.008249949080366458 gap 1.1102230246251565e-16 weights [0.159387 0.203236 0.2648   0.092717 0.151047 0.125668 0.106981 0.296164]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=81cb7dec5ca60051 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_81cb7dec (directory).
```

### Turn 8 — `execute_in_remote_docker` (`35272 ms` LLM)
- **Rationale**: Identify the exact competing wall rows before changing the solver. Test the compact calibration on the failed block and the independent bank data; if it does not transfer, do not promote it.
- **Alternatives Weighed**: `Keep the unrestricted cubic because its random-fold errors are small` (rejected: The blocked-fold failure is large. The fitted coefficients suggest a compact cyclic model that can be tested on that same fold and the independent bank sweeps.)
- **ATMS NOGOOD DECLARED**: Refuted `Nearest apparatus-trace mean identifies the active sealed-wall stencil.` | Witness: `With the same calibrated observations and constrained objective, the sealed-wall fit is about 52 times worse than the shared batch-02/s00 stencil and puts two sources at the upper bound.` | Blanket Diagnosis: `Apparatus-trace proximity is a proxy, not a direct observation of the wall flux law; the calibrated chamber response distinguishes the candidate stencils.` | Sublated Pivot: `Inspect the shared batch-02/s00 stencil and test whether its advantage holds on omitted sensors.`

```bash
set -e
cd /holdout_task
for p in /tmp/hegel_continual_archive/*solver.py; do echo "=== $(basename "$p") ==="; sed -n '1,43p' "$p"; done
python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; a=s['sensor_calibration_sweeps']; x,y=a['ref_rosette'],a['raw_rosette']
def basis(x,j):
 u=x[:,j]; v=x[:,(j+1)%4]; p=x[:,(j-1)%4]
 return np.column_stack((np.ones(len(x)),u,v,p,u*v,u**3))
def fit(x,y):
 d=[basis(x,j) for j in range(4)]; keep=np.count_nonzero(x,axis=1)>1
 for _ in range(10):
  c=np.array([np.linalg.lstsq(b[keep],y[keep,j],rcond=None)[0] for j,b in enumerate(d)])
  r=np.linalg.norm(y-np.column_stack([b@c[j] for j,b in enumerate(d)]),axis=1)
  update=r<=max(8*np.median(r),1e-10)
  if np.array_equal(keep,update): return c
  keep=update
 return c
def predict(x,c): return np.column_stack([basis(x,j)@c[j] for j in range(4)])
c=fit(x,y); print('structured coefficients',c.round(7)); e=np.max(abs(y-predict(x,c)),axis=1); print('gross rows',np.flatnonzero(e>.01),'inlier max',max(e[e<.01]));
for name,groups in [('blocked',np.array_split(np.arange(len(x)),4)),('random',np.array_split(np.random.default_rng(42).permutation(len(x)),5)),('isolated',[np.flatnonzero(np.count_nonzero(x,axis=1)==1)])]:
 errors=[]
 for ix in groups:
  tr=np.ones(len(x),bool); tr[ix]=False; clean=ix[e[ix]<.01]; errors.append(float(np.max(abs(y[clean]-predict(x[clean],fit(x[tr],y[tr]))))))
 print(name,'clean held max',errors)
b=a['single_bank_sweeps']; j=b['bank_id']; u=b['ref_signal']; r=b['raw_readout']-(c[j,0]+c[j,1]*u+c[j,5]*u**3); print('independent bank residual max excluding gross',np.max(abs(r[abs(r)<.01])),'gross',r[abs(r)>=.01]); print('sensor layout',s['sensor_ij'].shape,'shifts',s['mu_shifts'],'ridge',s['source_ridge'])
PY
```
- **Container Exit Code**: `0` (`462 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ocity[1:], 0) / h
    upper = -faces - np.maximum(-velocity[:-1], 0) / h
    upper[0], lower[-1] = -kappa[0] / h**2, -kappa[-1] / h**2
    main = np.r_[-upper[0] + robin / h,
                 -lower[:-1] - upper[1:],
                 -lower[-1] + robin / h]
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'], study['robin_x'])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'], study['robin_y'])
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    op = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
    op += diags(reaction, format='csc')
    modes = np.asarray(study['source_modes'])
    rhs = modes.transpose(0, 2, 1).reshape(len(modes), nx * ny).T
    i, j = np.asarray(study['sensor_ij']).T
    idx = i + nx * j
    identity = eye(nx * ny, format='csc')
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(op + shift * identity).solve(rhs)
        blocks.append(fields[idx])
    return np.vstack(blocks), fields.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = np.asarray(sweeps['ref_rosette']), np.asarray(sweeps['raw_rosette'])
    offset, *mix = np.linalg.lstsq(np.column_stack((np.ones(len(ref)), ref)), raw, rcond=None)[0]
structured coefficients [[-0.0600003  0.8200044 -0.2377973  0.1188994 -0.180401   0.1475968]
 [ 0.0699989  1.2799998 -0.3712011  0.1855993 -0.2815994  0.2304004]
 [ 0.0800014  1.3199975 -0.3827987  0.1913999 -0.2903986  0.2376013]
 [-0.0700011  0.7799966 -0.2262026  0.1131008 -0.1716008  0.1404028]]
gross rows [13 14 15] inlier max 2.8994677389683476e-05
blocked clean held max [3.0452573068488753e-05, 2.6375438165748832e-05, 2.1898051490332193e-05, 2.8829687616305222e-05]
random clean held max [2.152683917983067e-05, 3.0181758629198308e-05, 3.0259736534699044e-05, 2.6594647460104204e-05, 2.7073918914766182e-05]
isolated clean held max [3.0461231541101252e-05]
independent bank residual max excluding gross 1.9648131296401417e-05 gross [14.50000965]
sensor layout (32, 2) shifts [0.15 0.55 1.1 ] ridge 0.04

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=81cb7dec5ca60051 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_81cb7dec (directory).
```

### Turn 9 — `execute_in_remote_docker` (`67475 ms` LLM)
- **Rationale**: Install the compact calibration and measured-response-supported wall stencil. Held-out sensor, rosette, and shift errors can disqualify the stencil; blocked sweeps test calibration transfer. An independent stencil and KKT calculation can catch ordering, terminal-field, or certificate errors before submission.
- **Alternatives Weighed**: `Keep sealed walls based on the apparatus trace alone` (rejected: The calibrated public measurements favor the Robin stencil strongly; held-out sensors and shifts will test whether that advantage generalizes.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('batch_solver.py')
s = p.read_text().replace('from itertools import combinations_with_replacement\n\n', '')
archive = Path('/tmp/hegel_continual_archive/beta_m98_v2_c04_batch_02_solver.py').read_text()
s = s[:s.index('def _axis(')] + archive[archive.index('def _axis('):archive.index('def _calibrate(')] + s[s.index('_TERMS ='):]
a, b = s.index('_TERMS ='), s.index('def _box_budget(')
s = s[:a] + '''def _rosette_basis(x):
    nxt, prev = np.roll(x, -1, axis=1), np.roll(x, 1, axis=1)
    return np.stack((np.ones_like(x), x, nxt, prev, x * nxt, x**3), axis=-1)


def _fit_rosette(ref, raw):
    # Mixed sweeps identify the cross-talk terms without relying on isolated sweeps.
    basis = _rosette_basis(ref)
    keep = np.count_nonzero(ref, axis=1) > 1

    def fit(mask):
        return np.array([np.linalg.lstsq(basis[mask, j], raw[mask, j], rcond=None)[0]
                         for j in range(4)])

    for _ in range(10):
        coef = fit(keep)
        err = np.linalg.norm(raw - np.einsum('njk,jk->nj', basis, coef), axis=1)
        updated = err <= max(8 * np.median(err), 1e-10)
        if np.array_equal(keep, updated):
            return coef
        keep = updated
    return fit(keep)


def _invert_rosette(raw, coef):
    j = np.arange(4)
    nxt, prev = (j + 1) % 4, (j - 1) % 4
    linear = np.zeros((4, 4))
    linear[j, j], linear[j, nxt], linear[j, prev] = coef[:, 1:4].T
    x = np.linalg.solve(linear, (raw - coef[:, 0]).T).T
    for _ in range(20):
        residual = np.einsum('njk,jk->nj', _rosette_basis(x), coef) - raw
        if np.max(np.abs(residual)) < 1e-12:
            return x
        jac = np.broadcast_to(linear, (len(x), 4, 4)).copy()
        jac[:, j, j] += coef[:, 4] * x[:, nxt] + 3 * coef[:, 5] * x**2
        jac[:, j, nxt] += coef[:, 4] * x
        x -= np.linalg.solve(jac, residual[..., None])[..., 0]
    raise RuntimeError('Rosette calibration inversion did not converge')


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    coef = _fit_rosette(np.asarray(sweeps['ref_rosette']),
                        np.asarray(sweeps['raw_rosette']))
    return _invert_rosette(np.asarray(study['y_obs_raw']).reshape(-1, 4), coef).ravel()


''' + s[b:]
p.write_text(s)
PY
python3 -I /holdout_task/validate_public_schema.py
cat >/tmp/verify_m98_c04.py <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
archive = Path('/tmp/hegel_continual_archive')
def load(p):
    spec = importlib.util.spec_from_file_location('solver', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m = load(root / 'batch_solver.py')
s = pickle.load((root / 'public_batch.pkl').open('rb'))[0]['study']
a = s['sensor_calibration_sweeps']
x, raw = a['ref_rosette'], a['raw_rosette']
coef = m._fit_rosette(x, raw)
def pred(x, c):
    return np.einsum('njk,jk->nj', m._rosette_basis(x), c)
errors = []
for test in np.array_split(np.arange(len(x)), 4) + [np.flatnonzero(np.count_nonzero(x, axis=1) == 1)]:
    tr = ~np.isin(np.arange(len(x)), test)
    r = abs(raw[test] - pred(x[test], m._fit_rosette(x[tr], raw[tr])))
    gross = np.max(r, axis=1) > 1
    assert not np.any((np.max(r, axis=1) > 1e-4) & ~gross)
    errors.append(np.max(r[~gross]))
b = a['single_bank_sweeps']; j, u = b['bank_id'], b['ref_signal']
r = abs(b['raw_readout'] - (coef[j, 0] + coef[j, 1] * u + coef[j, 5] * u**3))
assert np.count_nonzero(r > 1e-4) == 1 and r.max() > 10
obs = s['y_obs_raw'].reshape(-1, 4)
y = m._calibrate(s)
roundtrip = np.max(abs(pred(y.reshape(-1, 4), coef) - obs))
assert max(errors) < 1e-4 and roundtrip < 1e-11
print('blocked/isolated calibration worst clean abs error', max(errors), 'independent bank clean max', max(r[r<1]), 'inverse abs error', roundtrip)

Z = s['Z_nuisance']; ridge, sr = s['nuisance_ridge'], s['source_ridge']; cap, budget = s['u_box'], s['total_budget']
ns = len(s['sensor_ij']); nk = len(s['mu_shifts']); rows = np.arange(len(y)).reshape(nk, ns)
outputs = {}
for name, mm in [('Robin', m), ('sealed', load(archive/'beta_m98_v2_c04_batch_01_solver.py')), ('batch03', load(archive/'beta_m98_v2_c04_batch_03_solver.py'))]:
    S, _ = mm._forward(s)
    for split, folds in [('rosette', [rows[:,i:i+4].ravel() for i in range(0, ns, 4)]), ('sensor', list(rows.T)), ('shift', list(rows))]:
        rms = []
        for test in folds:
            tr = ~np.isin(np.arange(len(y)), test)
            z, G, t = Z[tr], S[tr], y[tr]
            p = np.linalg.solve(z.T@z + ridge*np.eye(z.shape[1]), z.T)
            D, q = G-z@(p@G), t-z@(p@t)
            w, _ = m._box_budget(D.T@D + sr*np.eye(S.shape[1]), -D.T@q, cap, budget)
            g = p@(t-G@w)
            rms.append(np.sqrt(np.mean((S[test]@w+Z[test]@g-y[test])**2)))
        outputs[name, split] = max(rms)
        print(name, split, 'mean / worst held-out RMSE', np.mean(rms), max(rms))
assert all(outputs['Robin',k] < min(outputs['sealed',k],outputs['batch03',k]) for k in ('rosette','sensor','shift'))

# Rebuild the stencil directly and test the delivered field against its independent residual.
def axis(k, v, L, rob):
    n = len(k); h = L/(n-1); A = np.zeros((n,n))
    for i in range(1,n-1):
        a = (k[i]+k[i-1])/(2*h*h); b = (k[i]+k[i+1])/(2*h*h)
        A[i,i-1] = -a-max(v[i],0)/h
        A[i,i+1] = -b-max(-v[i],0)/h
        A[i,i] = a+b+abs(v[i])/h
    for i,j in [(0,1),(n-1,n-2)]:
        A[i,i], A[i,j] = k[i]/h**2 + rob/h, -k[i]/h**2
    return A
nx,ny=s['nx'],s['ny']; modes=s['source_modes']
A,B=[axis(s['kappa_'+d],s['vel_'+d],s['L'+d],s['robin_'+d]) for d in 'xy']
L=np.kron(np.eye(ny),A)+np.kron(B,np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
idx=np.array(s['sensor_ij']) @ [1,nx]
rhs=np.stack([u.ravel(order='F') for u in modes],axis=1)
S=np.vstack([np.linalg.solve(L+shift*np.eye(nx*ny),rhs)[idx] for shift in s['mu_shifts']])
out=m.solve_batch(s); w=out['weights']; k=out['kkt_certificate']; nu=k['nu_sum']
p=np.linalg.solve(Z.T@Z+ridge*np.eye(Z.shape[1]),Z.T)
D=S-Z@(p@S); q=y-Z@(p@y); e=D@w-q
J=0.5*(e@e+sr*(w@w)); g=D.T@e+sr*w
lower,upper=np.asarray(k['lam_lower']),np.asarray(k['lam_upper'])
gap=J-k['dual_lower_bound']; stat=np.max(abs(g-nu-lower+upper))
res=np.max(abs((L+s['mu_shifts'][-1]*np.eye(nx*ny))@out['terminal_field'].ravel(order='F')-rhs@w))
assert np.allclose(k['weights'],w,atol=1e-13) and abs(w.sum()-budget)<1e-12
assert min(w)>=-1e-12 and max(w)<=cap+1e-12 and min(lower)>=0 and min(upper)>=0
assert max(np.max(abs(lower*w)),np.max(abs(upper*(cap-w))),stat,abs(gap),abs(J-k['primal_objective']),res)<1e-10
print('independent objective',J,'dual gap',gap,'KKT stationarity',stat,'terminal PDE residual',res)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_m98_c04.py
```
- **Container Exit Code**: `0` (`770 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
blocked/isolated calibration worst clean abs error 3.0461231541101252e-05 independent bank clean max 1.9648131296401417e-05 inverse abs error 1.1102230246251565e-16
Robin rosette mean / worst held-out RMSE 0.007851981056795918 0.010319387096278169
Robin sensor mean / worst held-out RMSE 0.007392881428232611 0.012456342285623995
Robin shift mean / worst held-out RMSE 0.007917490158153432 0.00877961718149097
sealed rosette mean / worst held-out RMSE 0.09618691033335988 0.12488664398605469
sealed sensor mean / worst held-out RMSE 0.09433671078113832 0.16228133070087886
sealed shift mean / worst held-out RMSE 0.09859477410397526 0.1609475978219687
batch03 rosette mean / worst held-out RMSE 0.11027729995569382 0.1245365233712046
batch03 sensor mean / worst held-out RMSE 0.11025641668210556 0.13271674090487687
batch03 shift mean / worst held-out RMSE 0.10321727563093348 0.17668324183699755
independent objective 0.008249643364771561 dual gap 7.45931094670027e-17 KKT stationarity 6.331740687315346e-16 terminal PDE residual 2.6645352591003757e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6840b3f53d4a8082 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_6840b3f5 (directory).
```

### Turn 10 — `submit_completed_task` (`16470 ms` LLM)
- **Rationale**: The saved solver passed isolated public schema validation and independent field and KKT checks. Blocked calibration sweeps and held-out chamber measurements support the selected model.
- **Alternatives Weighed**: `Retain sealed walls based on the apparatus trace` (rejected: Robin walls yielded lower held-out RMSE across sensor, rosette, and shift splits.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:6584'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, worst held-out sensor RMSE was 0.01246, and the terminal field satisfied the independently reconstructed PDE to a residual of 2.7e-15.`

### Turn 11 — `execute_in_remote_docker` (`37101 ms` LLM)
- **Rationale**: Compare the current calibration against two physically plausible challengers fitted inside each blocked fold and tested on independent bank sweeps. Promote a challenger only if the measurements show a meaningful improvement. Residuals grouped by bank and shift can flag systematic structure hidden by pooled error.
- **Alternatives Weighed**: `Test relay modes, HJB drift, or within-batch classification accuracy` (rejected: This study has a steady continuous inverse problem, not relay states, a time-dependent control law, or class labels. The relevant nuisance check is calibrated residual bias by bank and shift.)

```bash
set -e
cd /holdout_task
cat >/tmp/m98_calibration_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
a = s['sensor_calibration_sweeps']
x, y = a['ref_rosette'], a['raw_rosette']
basis = m._rosette_basis(x)
# Test the observed common gain-normalized cyclic law, and whether a rational
# interaction in the difference coordinate explains any residual structure.
def predict(p, d, rational):
    v = d[..., 1] + np.einsum('njk,k->nj', d[..., 2:], p[8:12])
    if rational:
        v /= 1 + p[12] * (d[..., 1] - d[..., 2])**2
    return p[:4] + v*p[4:8]

def fit(train, rational):
    c = m._fit_rosette(x[train], y[train])
    r = np.linalg.norm(y[train] - np.einsum('njk,jk->nj', basis[train], c), axis=1)
    clean = r <= max(8*np.median(r), 1e-10)
    p0 = np.r_[c[:,0], c[:,1], np.mean(c[:,2:]/c[:,1,None], axis=0)]
    if rational:
        p0 = np.r_[p0, 0.]
    p = least_squares(lambda p: (predict(p, basis[train][clean], rational)-y[train][clean]).ravel(), p0, xtol=1e-13, ftol=1e-13, gtol=1e-13).x
    return c, p, clean

folds = np.array_split(np.arange(len(x)),4) + [np.flatnonzero(np.count_nonzero(x,axis=1)==1)]
errors = [[], [], []]
ratios = []
for test in folds:
    train = ~np.isin(np.arange(len(x)), test)
    c, p, _ = fit(train, False)
    _, q, _ = fit(train, True)
    forecasts = [np.einsum('njk,jk->nj', basis[test], c), predict(p,basis[test],False), predict(q,basis[test],True)]
    for group, forecast in zip(errors, forecasts):
        e = np.max(abs(y[test]-forecast), axis=1)
        group.append((float(e[e<1].max()), int(np.sum(e>=1))))
    ratios.append(q[-1])
for name, result in zip(('cyclic', 'shared gain-normalized', 'shared rational difference'), errors):
    print(name, 'blocked (max clean, gross count)', result, 'worst clean', max(v[0] for v in result))
print('rational denominator coefficients by fold', ratios)
b = a['single_bank_sweeps']; j, u = b['bank_id'], b['ref_signal']
for rational in (False, True):
    c, p, _ = fit(np.ones(len(x), dtype=bool), rational)
    refs = np.zeros((len(u),4)); refs[np.arange(len(u)),j] = u
    e = abs(predict(p,m._rosette_basis(refs),rational)[np.arange(len(u)),j]-b['raw_readout'])
    print('independent bank, rational=',rational,'worst clean',e[e<1].max(),'gross count',np.sum(e>=1))

S, _ = m._forward(s)
ycal = m._calibrate(s); Z = s['Z_nuisance']; w = m.solve_batch(s)['weights']
gamma = np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@(ycal-S@w))
res = (ycal-S@w-Z@gamma).reshape(len(s['mu_shifts']),-1,4)
print('calibrated signed residual means by shift x bank',res.mean(axis=1).round(6).tolist())
print('calibrated residual RMSE by shift x bank',np.sqrt((res**2).mean(axis=1)).round(6).tolist())
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m98_calibration_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`900 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cyclic blocked (max clean, gross count) [(3.0452573068488753e-05, 3), (2.6375438165748832e-05, 0), (2.189805149011015e-05, 0), (2.8829687616305222e-05, 0), (3.0461231541101252e-05, 3)] worst clean 3.0461231541101252e-05
shared gain-normalized blocked (max clean, gross count) [(2.738152398635424e-05, 3), (2.6446487421449838e-05, 0), (2.5176451308395187e-05, 0), (2.8215460460501163e-05, 0), (2.7833637197702266e-05, 3)] worst clean 2.8215460460501163e-05
shared rational difference blocked (max clean, gross count) [(2.765987327466024e-05, 3), (2.6799674792687966e-05, 0), (2.5105048139462127e-05, 0), (2.726185075271892e-05, 0), (2.817691772936254e-05, 3)] worst clean 2.817691772936254e-05
rational denominator coefficients by fold [-3.840147553785932e-07, -5.472272149440973e-07, -2.705412895810798e-07, -2.574883769813192e-07, -3.8017747414731407e-07]
independent bank, rational= False worst clean 1.991492785125004e-05 gross count 1
independent bank, rational= True worst clean 1.9357875855163797e-05 gross count 1
calibrated signed residual means by shift x bank [[-0.003122, 0.000482, 0.001782, -0.002506], [-0.00191, -0.003909, -0.001666, 3.6e-05], [0.004044, 0.001045, -0.000726, -0.001828]]
calibrated residual RMSE by shift x bank [[0.006867, 0.005967, 0.006257, 0.006448], [0.005408, 0.007942, 0.008283, 0.006328], [0.010371, 0.004877, 0.006943, 0.01012]]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6840b3f53d4a8082 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_6840b3f5 (directory).
```

### Turn 12 — `submit_completed_task` (`16836 ms` LLM)
- **Rationale**: Robin walls had the lowest worst-case held-out error across sensor, rosette, and shift splits. The final artifact passed isolated schema validation and independent checks of the field and KKT certificate. The follow-up calibration audit found no change that meaningfully improved transfer.
- **Alternatives Weighed**: `Replace the cyclic fit with a shared gain-normalized or rational calibration.` (rejected: Both challengers reduced the worst blocked clean error by only about 2.3e-6 raw units. The independent bank sweeps showed no consistent improvement, and the fitted rational term was negligible.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed. Worst held-out sensor RMSE was 0.01246, and the independently checked terminal PDE residual was 2.7e-15.`

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
