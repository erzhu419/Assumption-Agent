# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_v2_c04_batch_06__blind_continual_campaign`

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
| **Total Tokens (Prompt / Cached / Out)** | `255,552` (`248,136` / `96,000` / `7,416`) |
| **Total Episode Wall Time (s)** | `274.61s` |

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

### Turn 1 — `execute_in_remote_docker` (`14815 ms` LLM)
- **Rationale**: Inspect the supported schema and compare apparatus entries first; a mismatch would change which archived boundary operator is used.
- **Alternatives Weighed**: `Assume the most recent solver is unchanged` (rejected: The blind campaign can reactivate a different archived regime.)

```bash
cd /holdout_task && ls -l /tmp/hegel_continual_archive && cat public_asset_index.json && cat validate_public_schema.py && python3 - <<'PY'
import pickle
x=pickle.load(open('public_batch.pkl','rb')); print('items',len(x))
for a in x:
 print('keys',list(a),'diagnostics',None if a.get('campaign_diagnostics') is None else len(a['campaign_diagnostics']))
 for k,v in a['study'].items(): print(k,type(v).__name__,getattr(v,'shape',None), list(v) if isinstance(v,dict) else v if isinstance(v,(int,float,str)) else '')
PY
cat /tmp/hegel_continual_archive/archive_apparatus_log.json
```
- **Container Exit Code**: `0` (`396 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
trace": [
      [
        -0.29913,
        2.39469,
        0.48273,
        -0.34435,
        -1.33268,
        0.38171,
        -0.52292,
        -1.168
      ],
      [
        -0.7756,
        2.46504,
        0.57901,
        -0.33559,
        -1.21669,
        0.38174,
        -0.67984,
        -1.17982
      ],
      [
        -0.73617,
        2.53082,
        0.32236,
        -0.29894,
        -1.27952,
        0.40781,
        -0.71846,
        -1.15225
      ],
      [
        -0.97042,
        2.47631,
        0.3288,
        -0.4187,
        -1.00215,
        0.16029,
        -0.90841,
        -0.96909
      ],
      [
        -0.92441,
        2.38492,
        0.29367,
        -0.36429,
        -1.29939,
        0.59088,
        -0.80638,
        -1.17572
      ],
      [
        -0.77011,
        2.4229,
        0.61794,
        -0.35363,
        -1.07768,
        0.62316,
        -0.66384,
        -1.2167
      ],
      [
        -0.30374,
        2.4523,
        0.43826,
        -0.26102,
        -1.26648,
        0.58822,
        -0.19765,
        -1.1509
      ],
      [
        0.28213,
        2.42717,
        0.32119,
        -0.30907,
        -1.0027,
        0.50106,
        -0.01857,
        -0.91425
      ],
      [
        0.4525,
        2.44495,
        0.58941,
        -0.24926,
        -0.84692,
        0.18561,
        0.43808,
        -1.02379
      ],
      [
        0.48487,
        2.53282,
        0.30706,
        -0.32826,
        -1.12877,
        0.7729,
        0.89099,
        -1.13294
      ],
      [
        0.96914,
        2.38501,
        0.4317,
        -0.33673,
        -1.22896,
        0.44035,
        0.77449,
        -1.38558
      ],
      [
        0.82686,
        2.32407,
        0.47519,
        -0.28104,
        -1.25747,
        0.40731,
        0.942,
        -1.08776
      ],
      [
        0.81017,
        2.45572,
        0.35912,
        -0.43527,
        -1.29816,
        0.60092,
        0.85477,
        -1.43195
      ],
      [
        0.40703,
        2.30297,
        0.37096,
        -0.41913,
        -1.30596,
        0.44421,
        0.46172,
        -1.12046
      ],
      [
        0.06687,
        2.39764,
        0.62501,
        -0.39158,
        -1.4066,
        0.64821,
        0.05333,
        -0.92009
      ],
      [
        -0.44876,
        2.50949,
        0.35074,
        -0.37633,
        -1.0042,
        0.35622,
        -0.49897,
        -0.93654
      ]
    ]
  }
]
```

### Turn 2 — `execute_in_remote_docker` (`18565 ms` LLM)
- **Rationale**: Read the actual pulse and calibration schemas and extract the archived constitutive differences. These checks determine whether to select the boundary from tracer measurements or fall back to the apparatus log.
- **Alternatives Weighed**: `Use the apparatus trace alone` (rejected: If the two pulses in the asset index are present in the pickle, they directly test the boundary operator.)

```bash
cd /holdout_task && cat validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,json,numpy as np
items=pickle.load(open('public_batch.pkl','rb'))
print('items',len(items))
for a in items[:4]:
 s=a['study']; d=a.get('campaign_diagnostics'); print('diagnostics',None if d is None else len(d),'keys',list(s)); print('cal',[(k,np.shape(v) if not isinstance(v,dict) else list(v)) for k,v in s['sensor_calibration_sweeps'].items()])
l=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); traces=np.array([a['facility_telemetry_trace'] for a in l]); print('means',traces.mean(1).round(3)); print('latest trace RMSE',np.sqrt(((traces-traces[-1])**2).mean((1,2))).round(4))
PY
cd /tmp/hegel_continual_archive && cat archive_manifest.json && cat beta_m98_v2_c04_batch_03_solver.py && for p in beta_m98_v2_c04_s00_solver.py beta_m98_v2_c04_batch_01_solver.py beta_m98_v2_c04_batch_04_solver.py; do echo "$p"; diff -u beta_m98_v2_c04_batch_03_solver.py "$p" | head -145; done
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling'
```
- **Container Exit Code**: `0` (`420 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
3), axis=-1)
+
+
+def _fit_rosette(ref, raw):
+    # Mixed sweeps identify the cross-talk terms without relying on isolated sweeps.
+    basis = _rosette_basis(ref)
+    keep = np.count_nonzero(ref, axis=1) > 1
+
+    def fit(mask):
+        return np.array([np.linalg.lstsq(basis[mask, j], raw[mask, j], rcond=None)[0]
+                         for j in range(4)])
+
+    for _ in range(10):
+        coef = fit(keep)
+        err = np.linalg.norm(raw - np.einsum('njk,jk->nj', basis, coef), axis=1)
+        updated = err <= max(8 * np.median(err), 1e-10)
+        if np.array_equal(keep, updated):
+            return coef
+        keep = updated
+    return fit(keep)
+
+
+def _invert_rosette(raw, coef):
+    j = np.arange(4)
+    nxt, prev = (j + 1) % 4, (j - 1) % 4
+    linear = np.zeros((4, 4))
+    linear[j, j], linear[j, nxt], linear[j, prev] = coef[:, 1:4].T
+    x = np.linalg.solve(linear, (raw - coef[:, 0]).T).T
+    for _ in range(20):
+        residual = np.einsum('njk,jk->nj', _rosette_basis(x), coef) - raw
+        if np.max(np.abs(residual)) < 1e-12:
+            return x
+        jac = np.broadcast_to(linear, (len(x), 4, 4)).copy()
+        jac[:, j, j] += coef[:, 4] * x[:, nxt] + 3 * coef[:, 5] * x**2
+        jac[:, j, nxt] += coef[:, 4] * x
+        x -= np.linalg.solve(jac, residual[..., None])[..., 0]
+    raise RuntimeError('Rosette calibration inversion did not converge')
+
+
 def _calibrate(study):
     sweeps = study['sensor_calibration_sweeps']
-    ref, raw = np.asarray(sweeps['ref_rosette']), np.asarray(sweeps['raw_rosette'])
-    offset, *mix = np.linalg.lstsq(np.column_stack((np.ones(len(ref)), ref)), raw, rcond=None)[0]
-    observed = np.asarray(study['y_obs_raw']).reshape(-1, 4)
-    return np.linalg.solve(np.asarray(mix).T, (observed - offset).T).T.ravel()
+    coef = _fit_rosette(np.asarray(sweeps['ref_rosette']),
+                        np.asarray(sweeps['raw_rosette']))
+    return _invert_rosette(np.asarray(study['y_obs_raw']).reshape(-1, 4), coef).ravel()
 
 
 def _box_budget(hessian, linear, cap, budget):
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 3 — `execute_in_remote_docker` (`15589 ms` LLM)
- **Rationale**: Extract the distinct boundary rows and calibration entry points without repeating the truncated archive dump. These are needed to compare the public pulses across boundary and calibration combinations; the discovered L0 primitive is for a different optimization problem.
- **Alternatives Weighed**: `Copy the previous solver unchanged` (rejected: The archived boundary can change independently of the live sensor calibration.)

```bash
cd /tmp/hegel_continual_archive; for p in *solver.py; do echo "--- $p"; sed -n '1,/^def _forward/p' "$p"; grep '^def ' "$p"; done; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -16
```
- **Container Exit Code**: `0` (`316 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
aw):
def _invert_rosette(raw, coef):
def _calibrate(study):
def _box_budget(hessian, linear, cap, budget):
def solve_batch(study: dict) -> dict:
--- beta_m98_v2_c04_batch_05_solver.py
"""Steady advection–diffusion–reaction tomography on a tensor grid."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, length):
    n = len(kappa)
    h = length / (n - 1)
    faces = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = -faces - np.maximum(velocity[1:], 0) / h
    upper = -faces - np.maximum(-velocity[:-1], 0) / h
    upper[0], lower[-1] = -kappa[0] / h**2, -kappa[-1] / h**2
    # The tracer pulses identify zero wall leakage on both axes.
    main = np.r_[-upper[0], -lower[:-1] - upper[1:], -lower[-1]]
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
def _axis(kappa, velocity, length):
def _forward(study):
def _calibrate(study):
def _box_budget(hessian, linear, cap, budget):
def solve_batch(study: dict) -> dict:
--- beta_m98_v2_c04_s00_solver.py
"""Steady advection–diffusion–reaction tomography on a tensor grid."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, length, robin):
    n = len(kappa)
    h = length / (n - 1)
    faces = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = -faces - np.maximum(velocity[1:], 0) / h
    upper = -faces - np.maximum(-velocity[:-1], 0) / h
    upper[0], lower[-1] = -kappa[0] / h**2, -kappa[-1] / h**2
    main = np.r_[-upper[0] + robin / h,
                 -lower[:-1] - upper[1:],
                 -lower[-1] + robin / h]
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
def _axis(kappa, velocity, length, robin):
def _forward(study):
def _calibrate(study):
def _box_budget(hessian, linear, cap, budget):
def solve_batch(study: dict) -> dict:
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
```

### Turn 4 — `execute_in_remote_docker` (`28653 ms` LLM)
- **Rationale**: Compare archived wall stencils on both pulses and calibration models on held-out sweeps. This separates boundary mismatch from sensor error and determines which components of the archived solvers to retain.
- **Alternatives Weighed**: `Choose the archived periodic stencil from the pre-LLM table reference` (rejected: The table row failed its gate; the two measured pulses can test the stencil directly.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import importlib.util, pathlib, pickle, numpy as np
root=pathlib.Path('/tmp/hegel_continual_archive'); item=pickle.load(open('public_batch.pkl','rb'))[0]; s=item['study']; sw=s['sensor_calibration_sweeps']; x,y=sw['ref_rosette'],sw['raw_rosette']; mods={}
for name in ('02','03','04','05'):
 p=root/f'beta_m98_v2_c04_batch_{name}_solver.py'; spec=importlib.util.spec_from_file_location('m'+name,p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); mods[name]=m
m=mods['04']; B=m._rosette_basis(x); A=np.c_[np.ones(len(x)),x]; preds={}
for model in ('linear','nonlinear'):
 e=[]
 for tr,te in [(np.arange(len(x))%5!=i,np.arange(len(x))%5==i) for i in range(5)]:
  if model=='linear': residual=A[te]@np.linalg.lstsq(A[tr],y[tr],rcond=None)[0]-y[te]
  else: residual=np.einsum('njk,jk->nj',B[te],m._fit_rosette(x[tr],y[tr]))-y[te]
  e.append((float(np.sqrt(np.mean(residual**2))),float(np.median(np.abs(residual)))))
 print(model,'OOF (RMSE, median absolute)',np.array(e))
 for d in item['campaign_diagnostics']:
  ss={**s,'y_obs_raw':d['y_pulse_raw']}; cal=(mods['02'] if model=='linear' else m)._calibrate(ss); preds.setdefault(model,[]).append(cal)
for name in ('02','03','05'):
 S,U=mods[name]._forward(s)
 for model, vals in preds.items():
  print('operator',name,model,'pulse RMSE',[np.sqrt(np.mean((S@d['w_pulse']-cal)**2)) for d,cal in zip(item['campaign_diagnostics'],vals)])
print('linear coefficients',np.linalg.lstsq(A,y,rcond=None)[0]); print('nonlinear coefficients',m._fit_rosette(x,y)); print('bank sweep schema',[(k,np.shape(v)) for k,v in sw['single_bank_sweeps'].items()])
PY
sed -n '/^def _box_budget/,$p' /tmp/hegel_continual_archive/beta_m98_v2_c04_batch_03_solver.py
```
- **Container Exit Code**: `0` (`441 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
loat((g[high] + g[low]) / 2)
            state[[low, high]] = 0
            continue
        inv_g, inv_one = np.linalg.solve(hessian[np.ix_(free, free)],
                                       np.column_stack((g[free], np.ones(free.sum())))).T
        nu = (inv_g.sum() + budget - w.sum()) / inv_one.sum()
        direction = nu * inv_one - inv_g
        current = w[free]
        trial = current + direction
        hits_lower = trial < 0
        hits_upper = trial > cap
        if (hits_lower | hits_upper).any():
            ratios = np.full(len(current), np.inf)
            ratios[hits_lower] = -current[hits_lower] / direction[hits_lower]
            ratios[hits_upper] = (cap - current[hits_upper]) / direction[hits_upper]
            hit = ratios.argmin()
            w[free] = np.clip(current + ratios[hit] * direction, 0, cap)
            idx = np.flatnonzero(free)[hit]
            state[idx] = 1 if hits_upper[hit] else -1
            w[idx] = cap if hits_upper[hit] else 0
            continue
        w[free] = trial
        g = hessian @ w + linear
        violation = state * (g - nu)
        idx = violation.argmax()
        if violation[idx] <= tol:
            return w, float(nu)
        state[idx] = 0
    raise RuntimeError('Box-budget active set did not converge')


def solve_batch(study: dict) -> dict:
    design, terminal_modes = _forward(study)
    calibrated = _calibrate(study)
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
```

### Turn 5 — `execute_in_remote_docker` (`23026 ms` LLM)
- **Rationale**: The two pulses select the sealed-wall stencil after independent sweep calibration. Compose the two archived components without requiring pulse data at runtime. The isolated schema check will catch output-contract failures before independently verifying the field and KKT certificate.
- **Alternatives Weighed**: `Use the periodic or open-wall stencil with the nonlinear calibration` (rejected: Worst pulse RMSE is 0.0528 for periodic and 0.147 for open walls, versus 2.42e-6 for sealed walls.)
- **ATMS NOGOOD DECLARED**: Refuted `The affine rosette basis suffices for this campaign.` | Witness: `Adding the neighbor product and self-cubic terms reduced the worst held-out fold median absolute residual from 0.644 to 8.38e-6. Three folds still contained gross sweep outliers, so their full-fold RMSE remained high.` | Blanket Diagnosis: `Fit the sensor mapping on independent calibration sweeps before comparing wall stencils, so sensor distortion does not masquerade as leakage.` | Sublated Pivot: `Combine robust nonlinear sweep calibration with the archived sealed-wall stencil and retain the prescribed nuisance projection and constrained solve.`

```bash
set -eu
python3 - <<'PY'
from pathlib import Path
archive = Path('/tmp/hegel_continual_archive')
sealed = (archive / 'beta_m98_v2_c04_batch_05_solver.py').read_text()
nonlinear = (archive / 'beta_m98_v2_c04_batch_04_solver.py').read_text()
start = sealed.index('def _calibrate(')
end = sealed.index('def _box_budget(', start)
cal_start = nonlinear.index('def _rosette_basis(')
cal_end = nonlinear.index('def _box_budget(', cal_start)
solver = sealed[:start] + nonlinear[cal_start:cal_end] + sealed[end:]
solver = solver.replace('Steady advection–diffusion–reaction tomography on a tensor grid.', 'Sealed-wall transport tomography with sweep-fitted nonlinear rosette calibration.')
Path('/holdout_task/batch_solver.py').write_text(solver)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`514 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=930bff3d31fb9a00 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_930bff3d (directory).
```

### Turn 6 — `execute_in_remote_docker` (`37121 ms` LLM)
- **Rationale**: Apply the sealed stencil directly to the returned terminal field and independently recompute KKT residuals and the duality gap. A failed field check would redirect work to grid ordering or boundary assembly; a failed KKT check would redirect it to nuisance projection or multiplier signs. Breakdown by shift checks whether the aggregate pulse error hides a bad shift.
- **Alternatives Weighed**: `Submit on schema validation alone` (rejected: The schema validator cannot detect a transposed terminal field or multipliers inconsistent with the projected objective.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
out = m.solve_batch(s)
w, u, c = (out[k] for k in ('weights', 'terminal_field', 'kkt_certificate'))

# Apply the documented stencil directly, without the candidate sparse matrices.
def axis(u, k, v, h):
    flux = (k[1:] + k[:-1]) / (2 * h**2) * (u[1:] - u[:-1])
    r = np.empty_like(u)
    r[1:-1] = flux[:-1] - flux[1:] + np.maximum(v[1:-1], 0) / h * (u[1:-1] - u[:-2]) + np.maximum(-v[1:-1], 0) / h * (u[1:-1] - u[2:])
    r[0] = k[0] * (u[0] - u[1]) / h**2
    r[-1] = k[-1] * (u[-1] - u[-2]) / h**2
    return r

def apply(u, mu):
    return (axis(u, s['kappa_x'][:, None], s['vel_x'][:, None], s['Lx'] / (s['nx'] - 1)) + axis(u.T, s['kappa_y'][:, None], s['vel_y'][:, None], s['Ly'] / (s['ny'] - 1)).T + (s['V_reaction_2d'] + mu) * u)

r = apply(u, s['mu_shifts'][-1]) - np.einsum('m,mij->ij', w, s['source_modes'])
wall = np.ones(r.shape, dtype=bool)
wall[1:-1, 1:-1] = False
field_errors = [np.max(np.abs(r[mask])) for mask in (wall, ~wall)]
assert max(field_errors) < 1e-8, field_errors

S, _ = m._forward(s)
y = m._calibrate(s)
z = s['Z_nuisance']
p = np.eye(len(y)) - z @ np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
X = p @ S
residual = X @ w - p @ y
ridge, cap, budget = (s[k] for k in ('source_ridge', 'u_box', 'total_budget'))
g = X.T @ residual + ridge * w
low, high, nu = np.asarray(c['lam_lower']), np.asarray(c['lam_upper']), c['nu_sum']
primal = (residual @ residual + ridge * (w @ w)) / 2
complementarity = low @ w + high @ (cap - w)
stationarity = np.max(np.abs(g - nu - low + high))
assert np.array_equal(w, c['weights'])
assert max(-w.min(), w.max() - cap, abs(w.sum() - budget), -low.min(), -high.min()) < 1e-9
np.testing.assert_allclose([c['primal_objective'], c['dual_lower_bound']], [primal, primal - complementarity], rtol=0, atol=1e-9)
assert max(stationarity, abs(complementarity)) < 1e-8

pulse_errors = []
for d in item['campaign_diagnostics']:
    e = S @ d['w_pulse'] - m._calibrate({**s, 'y_obs_raw': d['y_pulse_raw']})
    pulse_errors.extend(np.sqrt(np.mean(e.reshape(len(s['mu_shifts']), -1)**2, axis=1)))
assert max(pulse_errors) < 3e-5, pulse_errors
print('max field residual (walls, interior):', field_errors)
print('worst pulse-by-shift RMSE:', max(pulse_errors))
print('KKT stationarity, duality gap:', stationarity, c['primal_objective'] - c['dual_lower_bound'])
print('active weights (lower, upper):', np.sum(w < 1e-10), np.sum(w > cap - 1e-10))
PY
```
- **Container Exit Code**: `0` (`470 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
max field residual (walls, interior): [2.2065682614424986e-15, 3.524958103184872e-15]
worst pulse-by-shift RMSE: 2.4472216299574742e-06
KKT stationarity, duality gap: 2.220446049250313e-16 3.885780586188048e-16
active weights (lower, upper): 3 2

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=930bff3d31fb9a00 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_930bff3d (directory).
```

### Turn 7 — `submit_completed_task` (`12669 ms` LLM)
- **Rationale**: The final artifact passed isolated schema validation, direct stencil checks at the walls and interior, and independent verification of the weights, terminal field, and KKT certificate.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:6561'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. It returns source weights, the terminal field, and a KKT certificate. The public schema check passed, and the worst tracer-pulse RMSE across shifts was 2.45e-6.`

### Turn 8 — `execute_in_remote_docker` (`40164 ms` LLM)
- **Rationale**: Test whether added rational and interaction terms improve held-out calibration, whether each pulse independently selects the sealed walls, and whether the prescribed nuisance projection works across shifts. A material calibration improvement would warrant a pulse-level comparison before editing the verified solver.
- **Alternatives Weighed**: `Apply batch-classification, relay, or Fokker–Planck checks` (rejected: The study is a steady transport inverse problem with continuous observations and a specified nuisance design, not a classification, relay, or stochastic-control task.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np

def load(path):
    spec = importlib.util.spec_from_file_location('audit_solver', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m = load('/holdout_task/batch_solver.py')
with open('public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
x, y = (np.asarray(s['sensor_calibration_sweeps'][k]) for k in ('ref_rosette', 'raw_rosette'))
base = m._rosette_basis(x)
cols = [x[:, i] - sign*x[:, j] for i in range(4) for j in range(i+1,4) for sign in (-1,1)]
extra = np.column_stack([q**p for q in cols for p in (2,3)] + [x[:,i]*x[:,j]*x[:,k] for i in range(4) for j in range(i+1,4) for k in range(j+1,4)] + [x[:,i]/(1+a*q*q) for i in range(4) for q in [x[:,i],x[:,(i+1)%4],cols[2*i]] for a in (.25,1.)])
metrics = {k: [] for k in ('base', 'augmented')}
for fold in range(5):
    te = np.arange(len(x)) % 5 == fold
    tr = ~te
    coef = m._fit_rosette(x[tr], y[tr])
    pred = np.einsum('njk,jk->nj', base, coef)
    challenge = np.empty_like(y[te])
    for ch in range(4):
        e = np.abs(pred[tr,ch]-y[tr,ch]); keep = tr.copy(); keep[tr] = e <= max(8*np.median(e),1e-4)
        A = np.column_stack((base[:,ch,:],extra))
        b = np.linalg.lstsq(A[keep], y[keep,ch], rcond=1e-10)[0]
        challenge[:,ch] = A[te] @ b
    for key, est in [('base',pred[te]),('augmented',challenge)]:
        metrics[key].append(np.median(np.abs(est-y[te]),axis=0))
print('Worst fold/channel median absolute calibration errors:', {key:float(np.max(val)) for key,val in metrics.items()})
archive = Path('/tmp/hegel_continual_archive')
pulses = item['campaign_diagnostics']
cal = [m._calibrate({**s,'y_obs_raw':d['y_pulse_raw']}) for d in pulses]
errors = {}
for name in ('02','03','05'):
    fwd = load(archive/f'beta_m98_v2_c04_batch_{name}_solver.py')._forward(s)[0]
    errors[name] = [float(np.sqrt(np.mean((fwd @ d['w_pulse']-yy)**2))) for d,yy in zip(pulses,cal)]
print('Pulse RMSEs for archived wall models:',errors)
for i in range(len(pulses)):
    assert min(errors,key=lambda k:errors[k][i]) == '05'
S,_ = m._forward(s)
z = s['Z_nuisance']
yc = m._calibrate(s)
w = m.solve_batch(s)['weights']
r = S@w-yc
A = z.T@z + s['nuisance_ridge']*np.eye(z.shape[1])
gamma = np.linalg.solve(A,z.T@r)
r_profiled = r-z@gamma
print('Observed residual RMSE by shift, before and after prescribed projection:', [np.sqrt(np.mean(q.reshape(len(s['mu_shifts']),-1)**2,axis=1)).tolist() for q in (r,r_profiled)])
print('Nuisance ridge normal-equation residual:',float(np.linalg.norm(z.T@r_profiled-s['nuisance_ridge']*gamma,np.inf)))
PY
```
- **Container Exit Code**: `0` (`443 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Worst fold/channel median absolute calibration errors: {'base': 1.0212646265085612e-05, 'augmented': 4.455199639352703e-05}
Pulse RMSEs for archived wall models: {'02': [0.14666527150863923, 0.10607804487001331], '03': [0.052772637021662565, 0.021874844686698862], '05': [2.4114597586430086e-06, 2.4212184956934338e-06]}
Observed residual RMSE by shift, before and after prescribed projection: [[0.26112691061876464, 0.24412346675068178, 0.24070683125658204], [0.15204911345305686, 0.07133552749483114, 0.0369421889153621]]
Nuisance ridge normal-equation residual: 4.528495634037455e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=930bff3d31fb9a00 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_930bff3d (directory).
```

### Turn 9 — `submit_completed_task` (`15571 ms` LLM)
- **Rationale**: Both public pulses independently favored sealed walls. The targeted audit found no improvement from the expanded calibration basis and confirmed the prescribed nuisance projection. The unchanged artifact passed isolated schema, terminal-field, and KKT verification.
- **Alternatives Weighed**: `Add rational and higher-order interaction terms to the calibration model` (rejected: The added terms increased the worst held-out fold/channel median absolute error from 1.02e-5 to 4.46e-5.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:6561'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. It returns source weights, the terminal field, and a KKT certificate. Public schema validation passed; the worst tracer-pulse RMSE across shifts was 2.45e-6.`

### Turn 10 — `execute_in_remote_docker` (`27842 ms` LLM)
- **Rationale**: The prior interleaved folds did not test sensitivity to a contiguous block of sweep rows. Delete each block in turn and measure the effect on weights, field, and pulse predictions. A large change would warrant inspecting which calibration rows drive the fit before changing the solver.
- **Alternatives Weighed**: `Repeat the rational-basis and nuisance audits` (rejected: The expanded basis already worsened held-out calibration, and the prescribed nuisance projection was verified across shifts.)

```bash
cd /holdout_task && export OPENBLAS_NUM_THREADS=1 && python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
sweeps = s['sensor_calibration_sweeps']
ref, raw = (np.asarray(sweeps[k]) for k in ('ref_rosette', 'raw_rosette'))
reference = m.solve_batch(s)
S, _ = m._forward(s)
worst = np.zeros(3)
for omitted in np.array_split(np.arange(len(ref)), 5):
    keep = np.ones(len(ref), dtype=bool)
    keep[omitted] = False
    sf = {**s, 'sensor_calibration_sweeps': {**sweeps, 'ref_rosette': ref[keep], 'raw_rosette': raw[keep]}}
    out = m.solve_batch(sf)
    for j, key in enumerate(('weights', 'terminal_field')):
        worst[j] = max(worst[j], np.linalg.norm(out[key] - reference[key]) / np.linalg.norm(reference[key]))
    for p in item['campaign_diagnostics']:
        e = S @ p['w_pulse'] - m._calibrate({**sf, 'y_obs_raw': p['y_pulse_raw']})
        worst[2] = max(worst[2], np.sqrt(np.mean(e.reshape(len(s['mu_shifts']), -1)**2, axis=1)).max())
print('Worst contiguous calibration-block deletion: relative weight change, relative field change, pulse-by-shift RMSE:', worst.tolist())
assert np.all(worst < [1e-4, 1e-4, 3e-5]), worst
PY
status=$?; if [ "$status" -ne 0 ]; then exit "$status"; fi
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`653 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Worst contiguous calibration-block deletion: relative weight change, relative field change, pulse-by-shift RMSE: [7.235935596418125e-06, 2.5116191493067586e-06, 3.4207113277356962e-06]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=930bff3d31fb9a00 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_930bff3d (directory).
```

### Turn 11 — `submit_completed_task` (`15834 ms` LLM)
- **Rationale**: Both measured pulses favored sealed walls. The exact submitted artifact passed isolated schema validation and independent field and KKT verification. Omitting contiguous calibration blocks changed the inferred weights by at most 7.24e-6 relative, so the stress test did not warrant an edit.
- **Alternatives Weighed**: `Expand the rosette calibration basis` (rejected: Adding rational and higher-order interaction terms increased the worst held-out fold/channel median absolute error from 1.02e-5 to 4.46e-5.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. It returns source weights, the terminal field, and a KKT certificate. Public schema validation passed. The worst tracer-pulse RMSE across shifts was 2.45e-6, rising to 3.42e-6 when calibration blocks were omitted.`

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
