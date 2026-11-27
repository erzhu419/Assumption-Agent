# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_v2_c03_s00__blind_continual_campaign`

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
| **Turns Used / Max Turns** | `10 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `229,664` (`221,206` / `100,224` / `8,458`) |
| **Total Episode Wall Time (s)** | `284.85s` |

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

### Turn 1 — `execute_in_remote_docker` (`22268 ms` LLM)
- **Rationale**: Inspect the study schema and archive inventory to determine which calibration and boundary models are supported before implementing the inverse solver.

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np
b=pickle.load(open('public_batch.pkl','rb'));print('batch length',len(b)); i=b[0]; print(i.keys());
for k,v in i['study'].items():
 print(k,type(v).__name__,getattr(v,'shape',''), list(v) if isinstance(v,dict) else (repr(v)[:220] if not isinstance(v,np.ndarray) or v.size<12 else (float(v.min()),float(v.max()))))
d=i.get('campaign_diagnostics');print('diagnostics',None if d is None else [(k,getattr(v,'shape',None)) for k,v in d[0].items()]);
for k,v in i['study']['sensor_calibration_sweeps'].items(): print(k,type(v).__name__,getattr(v,'shape',''),repr(v)[:460]);
PY
if [ -d /tmp/hegel_continual_archive ]; then du -ah /tmp/hegel_continual_archive | sort -h | tail -40; fi
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling' 2>&1 | head -30
```
- **Container Exit Code**: `0` (`400 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cribe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`23033 ms` LLM)
- **Rationale**: A calibration residual above sweep noise would require a richer sensor model. An applicable archive record would change the boundary operator; otherwise the task prescribes the open-wall baseline.

```bash
cd /holdout_task && cat validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np
b=pickle.load(open('public_batch.pkl','rb')); print('batch',len(b))
for it in b:
 s=it['study']; c=s['sensor_calibration_sweeps']; X=c['ref_rosette']; Y=c['raw_rosette']; D=np.c_[np.ones(len(X)),X]; B=np.linalg.lstsq(D,Y,rcond=None)[0]; print('dimensions',[(k,getattr(v,'shape',v)) for k,v in s.items() if k not in ('sensor_calibration_sweeps',)]); print('calibration affine coefficients',B.round(6),'rms',np.sqrt(np.mean((Y-D@B)**2))); print('row banks',s['row_bank_id'][:36]); print('telemetry first rows',s['aux_manifold_telemetry'][:5]); print('diagnostics available',it.get('campaign_diagnostics') is not None)
PY
if [ -d /tmp/hegel_continual_archive ]; then find /tmp/hegel_continual_archive -maxdepth 3 -type f -printf '%s %p\n' | head -40; else echo 'No continual archive'; fi
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance 2>&1 | head -25
```
- **Container Exit Code**: `0` (`454 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
), ('kappa_y', (12,)), ('vel_x', (14,)), ('vel_y', (12,)), ('robin_x', 0.943511194485015), ('robin_y', 1.0215322743752013), ('V_reaction_2d', (14, 12)), ('mu_shifts', (3,)), ('source_modes', (8, 14, 12)), ('sensor_ij', (32, 2)), ('row_bank_id', (96,)), ('Z_nuisance', (96, 3)), ('y_obs_raw', (96,)), ('aux_manifold_telemetry', (24, 4)), ('nuisance_ridge', 0.08), ('source_ridge', 0.04), ('u_box', 0.42), ('total_budget', 1.4)]
calibration affine coefficients [[-2.000000e-06 -1.000000e-06  0.000000e+00 -1.000000e-06]
 [ 1.000000e+00  1.000000e-06 -1.000000e-06  0.000000e+00]
 [-1.000000e-06  1.000000e+00 -1.000000e-06 -2.000000e-06]
 [-0.000000e+00 -1.000000e-06  9.999990e-01  1.000000e-06]
 [-1.000000e-06  2.000000e-06 -1.000000e-06  1.000001e+00]] rms 9.140529600147025e-06
row banks [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3]
telemetry first rows [[ 0.76118799  1.36152393 -1.72329076  1.67894086]
 [ 1.47780006  2.32222049  0.71007619 -0.56618141]
 [-1.45826681  0.00760135  1.51577312  0.6195927 ]
 [ 0.20352498 -0.32609707 -0.42169338  0.23617143]
 [-0.32586185 -0.23509852 -1.45654445  0.4896492 ]]
diagnostics available False
No continual archive
{
  "ok": true,
  "target": "autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_prim_ridge_profiled_matching",
      "granularity_tier": "L1_compositional_primitive",
      "compiled_code_sha256": "dfdf7838f3618f600b4d7d7df948af526e9de4b41cbe0e429a9a76032e691078",
      "qualification_status": "pretraining_only_l1_primitive_sublation",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        "design (n, p) and values (n,) or (n, q) share leading axis n; ridge >= 0 finite scalar or (p,) vector.",
        "whitener (r, p) must match the SVD Gram whitener of the same weighted design matrix; candidate_energy >= 0."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "fit_ridge_nuisance": "APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)",
        "profile_candidate_amplitudes": "APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)"
      },
      "functions": [
        {
          "function": "fit_ridge_nuisance",
          "signature": "fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray]",
```

### Turn 3 — `execute_in_remote_docker` (`50357 ms` LLM)
- **Rationale**: Build the specified forward operator and constrained inverse, then run the public validator normally and in isolation to catch output-schema, certificate, and import failures.
- **Alternatives Weighed**: `Fit an alternative wall condition to the blind observations` (rejected: No pulse diagnostics or archive are available, and the contract prescribes open-wall Robin in that case.); `Use the retrieved ridge profiling primitive` (rejected: The required residual uses the specified ridge projection directly; a small multi-right-hand-side Gram solve implements it without a dependency.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
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
    n = len(c)
    if budget == 0 or cap == 0:
        return np.zeros(n), float(min(c))
    if budget >= n * cap:
        w = np.full(n, cap)
        return w, float(max(H @ w + c))
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=int)  # -1 lower, +1 upper
    tol = 5e-13 * max(1., np.max(np.abs(c)), cap * np.max(np.sum(abs(H), axis=1)))
    for _ in range(30 * n + 30):
        free = np.flatnonzero(active == 0)
        g = H @ w + c
        if not len(free):
            lo, hi = np.flatnonzero(active < 0), np.flatnonzero(active > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] <= g[i] + tol:
                return w, float((g[j] + g[i]) / 2)
            active[[i, j]] = 0
            continue
        fixed = np.flatnonzero(active)
        m = len(free)
        system = np.ones((m + 1, m + 1))
        system[:m, :m] = H[np.ix_(free, free)]
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
PY
python3 /holdout_task/validate_public_schema.py && python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`581 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=42772b83fc1b03fc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_42772b83 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`38872 ms` LLM)
- **Rationale**: The schema check cannot catch ordering errors, an incorrect terminal field, or incorrect optimality multipliers. Compare against an independent dense forward operator and test calibration and optimization on cases with known solutions.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('batch_solver',p+'batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; nx,ny=s['nx'],s['ny']; n=nx*ny
# Independent dense row assembly checks placement, upwind signs and wall rows.
def axis(k,n):
    h=s['L'+k]/(n-1); a=s['kappa_'+k]; v=s['vel_'+k]; A=np.zeros((n,n))
    for i in range(n):
        if i in (0,n-1):
            A[i,1 if i==0 else n-2]=-a[i]/h**2; A[i,i]=a[i]/h**2+s['robin_'+k]/h
        else:
            A[i,i-1]=-(a[i]+a[i-1])/(2*h*h)-max(v[i],0)/h
            A[i,i+1]=-(a[i]+a[i+1])/(2*h*h)-max(-v[i],0)/h
            A[i,i]=-A[i,i-1]-A[i,i+1]
    return A
L=np.kron(np.eye(ny),axis('x',nx))+np.kron(axis('y',ny),np.eye(nx))+np.diag(s['V_reaction_2d'].flatten(order='F'))
F=np.array([f.flatten(order='F') for f in s['source_modes']]).T
x,y=s['sensor_ij'].T; idx=x+nx*y
U=[np.linalg.solve(L+shift*np.eye(n),F) for shift in s['mu_shifts']]; S=np.vstack([u[idx] for u in U])
SS,last=m._forward(s); worst=max(np.max(abs(SS-S)),np.max(abs(last-U[-1]))); assert worst<1e-12
cal=s['sensor_calibration_sweeps']; ref,raw=cal['ref_rosette'],cal['raw_rosette']; groups=np.arange(len(ref))%5; cv=[]
for k in range(5):
    keep=groups!=k; cen=ref[keep].mean(0); rcen=raw[keep].mean(0); A=np.linalg.lstsq(ref[keep]-cen,raw[keep]-rcen,rcond=None)[0]; cv.append(np.sqrt(np.mean(((ref[~keep]-cen)@A+rcen-raw[~keep])**2)))
out=m.solve_batch(s); w=out['weights']; kc=out['kkt_certificate']; cap=s['u_box']; budget=s['total_budget']; ridge=s['source_ridge']; Z=s['Z_nuisance']; R=np.eye(len(Z))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
res=R@(S@w-m._calibrate(s)); g=(R@S).T@res+ridge*w; J=(res@res+ridge*(w@w))/2; lo=kc['lam_lower']; hi=kc['lam_upper']; nu=kc['nu_sum']; gap=J-kc['dual_lower_bound']; err=max(abs(J-kc['primal_objective']),abs(gap),np.max(abs(g-nu-lo+hi)),np.max(abs(w*lo)),np.max(abs((cap-w)*hi)),np.max(abs(w-kc['weights'])),abs(w.sum()-budget),np.max(abs(out['terminal_field']-(U[-1]@w).reshape(nx,ny,order='F')))); assert err<1e-10
# Exact non-identity affine sweep exposes channel-transpose or offset errors.
A=np.array([[1.2,.1,0,0],[0,.8,.15,0],[.04,0,1.1,0],[.01,0,.1,.95]]); b=np.array([.02,-.01,.1,-.1]); clean=(S@w).reshape(-1,4); t=dict(s,y_obs_raw=(clean@A+b).ravel(),sensor_calibration_sweeps=dict(ref_rosette=ref,raw_rosette=ref@A+b)); ce=np.max(abs(m._calibrate(t)-clean.ravel())); assert ce<1e-12
# Known solutions cover all-active vertices, mixed faces and both endpoint budgets.
rng=np.random.default_rng(512); qe=0.
for k in range(240):
    N=8; cap=.42; B=rng.normal(size=(N,N)); H=B.T@B+.04*np.eye(N)
    if k%6==0: expected=np.zeros(N)
    elif k%6==1: expected=np.full(N,cap)
    elif k%6==2: expected=np.array([0.]*4+[cap]*4)
    else:
        expected=rng.uniform(-.3,.72,N).clip(0,cap); expected[:2]=[0,cap]
    rng.shuffle(expected); g0=.3+np.where(expected==0,rng.uniform(.01,1,N),0)-np.where(expected==cap,rng.uniform(.01,1,N),0); c=g0-H@expected
    found,nu=m._box_budget(H,c,cap,expected.sum()); qe=max(qe,np.max(abs(found-expected)))
assert qe<1e-10
print(f'Public max dense forward delta={worst:.3g}; worst held-out rosette RMS={max(cv):.3g}; public KKT/field max error={err:.3g}; mixed-rosette inversion error={ce:.3g}; 240 known-QP worst weight error={qe:.3g}')
PY
```
- **Container Exit Code**: `0` (`464 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public max dense forward delta=3.33e-16; worst held-out rosette RMS=1.08e-05; public KKT/field max error=4.7e-16; mixed-rosette inversion error=6.25e-16; 240 known-QP worst weight error=5.83e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=42772b83fc1b03fc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_42772b83 (directory).
```

### Turn 5 — `submit_completed_task` (`13818 ms` LLM)
- **Rationale**: The unchanged candidate passed isolated schema validation, independent dense forward and KKT checks, non-identity rosette inversion, and known-solution tests spanning interior, mixed-bound, vertex, and endpoint budgets.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`13` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5047'}): Do NOT lock prematurely with `13` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py`. The public schema passed in an isolated Python process. Independent checks confirmed the forward operator, terminal field, and KKT certificate; across 240 known-solution cases, the maximum weight error was 5.83e-16.`

### Turn 6 — `execute_in_remote_docker` (`28550 ms` LLM)
- **Rationale**: Compare nonlinear and affine models on held-out real sweeps so added flexibility must improve prediction, not just fit noise. Inspect shift-by-bank residuals for structure a pooled metric could hide, then verify the exact artifact in isolated Python.
- **Alternatives Weighed**: `Apply classification balance or relay-mode tests` (rejected: This study has continuous measurements and no class labels or relay state. Shift-by-bank residuals are the relevant nuisance slices.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('batch_solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=map(np.asarray,(c['ref_rosette'],c['raw_rosette']))
print('calibration sweep count',len(x),'single-bank representation',type(c.get('single_bank_sweeps')).__name__)
order=np.random.default_rng(731).permutation(len(x)); folds=np.array_split(order,5)
# Test cross-channel curvature and saturation against the affine noise floor.
base=np.c_[np.ones(len(x)),x]; cross=np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]); triple=np.column_stack([x[:,i]*x[:,j]*x[:,k] for i in range(4) for j in range(i+1,4) for k in range(j+1,4)])
scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-12); q=x/scale
sat=np.column_stack([q[:,i]/(1+a*q[:,j]**2) for a in (.1,1.) for i in range(4) for j in range(4)])
for name,D in [('affine',base),('quadratic-pair-plus-triples',np.c_[base,cross,triple]),('rational',np.c_[base,sat])]:
    err=np.empty_like(y)
    for test in folds:
        train=np.ones(len(x),bool); train[test]=False
        err[test]=D[test]@np.linalg.lstsq(D[train],y[train],rcond=None)[0]-y[test]
    print(name,'pooled RMS',np.sqrt(np.mean(err**2)),'worst fold-channel RMS',max(np.sqrt(np.mean(err[test]**2,axis=0)).max() for test in folds))
S,_=m._forward(s); ycal=m._calibrate(s); out=m.solve_batch(s); w=out['weights']; Z=s['Z_nuisance']; r=ycal-S@w; r-=Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@r)
banks=np.asarray(s['row_bank_id']).reshape(len(s['mu_shifts']),-1); r=r.reshape(banks.shape)
for mu,ids,row in zip(s['mu_shifts'],banks,r):
    print('shift',mu,'bank mean / RMS',[(int(b),round(row[ids==b].mean(),7),round(np.sqrt(np.mean(row[ids==b]**2)),7)) for b in np.unique(ids)])
k=out['kkt_certificate']; print('public gap',k['primal_objective']-k['dual_lower_bound'])
PY
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`659 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calibration sweep count 80 single-bank representation dict
affine pooled RMS 9.744138232063739e-06 worst fold-channel RMS 1.3191021745520108e-05
quadratic-pair-plus-triples pooled RMS 1.2588594553610118e-05 worst fold-channel RMS 1.74515440949336e-05
rational pooled RMS 2.060721137420298e-05 worst fold-channel RMS 3.320888409046237e-05
shift 0.15 bank mean / RMS [(0, 0.0040109, 0.0078389), (1, 0.0019793, 0.0043908), (2, -0.0007972, 0.0098441), (3, -0.0014692, 0.0044662)]
shift 0.55 bank mean / RMS [(0, 0.0008129, 0.0082302), (1, 0.0020405, 0.004569), (2, -0.0031811, 0.0050197), (3, -0.0020835, 0.008157)]
shift 1.1 bank mean / RMS [(0, 0.0009159, 0.0070822), (1, -0.002766, 0.0110563), (2, 0.0035515, 0.0065459), (3, -0.0019248, 0.0059737)]
public gap 1.1796119636642288e-16
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=42772b83fc1b03fc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_42772b83 (directory).
```

### Turn 7 — `submit_completed_task` (`17258 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema validation, independent dense forward and terminal-field checks, and KKT stress tests spanning mixed bounds, vertices, and endpoint budgets. The follow-up sweep audit found no calibration upgrade supported by held-out measurements.
- **Alternatives Weighed**: `Quadratic and triple terms or rational rosette calibration` (rejected: Both worsened held-out sweep error relative to the affine model.); `Infer a different wall operator from blind observations` (rejected: The contract prescribes Robin when neither tracer pulses nor an applicable archive is available.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5047'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The isolated public schema check passed, and the worst weight error across 240 known-solution cases was 5.83e-16. The solver fits calibration from the supplied sweeps and uses the prescribed Robin fallback because no boundary diagnostics or archive were available.`

### Turn 8 — `execute_in_remote_docker` (`33296 ms` LLM)
- **Rationale**: The previous bank residuals were in-sample. Holding out whole rosettes tests whether extra bank offsets predict unseen measurements or merely absorb training residuals. A reproducible improvement would identify nuisance misspecification to report without silently changing the prescribed objective; single-bank sweep disagreement would instead motivate revisiting calibration.
- **Alternatives Weighed**: `Repeat the nonlinear calibration audit or apply relay and binary classification metrics` (rejected: The held-out sweep comparison is already settled, and this continuous inverse problem has neither class labels nor relay states.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('batch_solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; cal=s['sensor_calibration_sweeps']; ref,raw=cal['ref_rosette'],cal['raw_rosette']; B=np.linalg.lstsq(np.c_[np.ones(len(ref)),ref],raw,rcond=None)[0]
for bank,v in cal['single_bank_sweeps'].items():
    if isinstance(v,dict) and {'ref_rosette','raw_rosette'}<=v.keys():
        x,y=np.asarray(v['ref_rosette']),np.asarray(v['raw_rosette']); print('single-bank calibration',bank,'RMS',np.sqrt(np.mean((np.c_[np.ones(len(x)),x]@B-y)**2)))
    else:
        print('single-bank calibration layout',bank,type(v).__name__,getattr(v,'shape',None),list(v) if isinstance(v,dict) else '')
S,_=m._forward(s); y=m._calibrate(s); Z=s['Z_nuisance']; ids=np.asarray(s['row_bank_id']); cap,budget,rs,rz=(s[k] for k in ('u_box','total_budget','source_ridge','nuisance_ridge')); n=len(y); blocks=np.arange(n)//4
bank_design=(ids[:,None]==np.unique(ids)[1:]).astype(float)
for name,z in [('specified nuisance',Z),('additional bank offsets',np.c_[Z,bank_design])]:
    e=np.empty(n); wd=[]; full=m.solve_batch(s)['weights']
    for group in np.unique(blocks):
        test=blocks==group; train=~test; a=z[train]; t=S[train]; yy=y[train]; G=a.T@a+rz*np.eye(a.shape[1]); projected=np.c_[t,yy]-a@np.linalg.solve(G,a.T@np.c_[t,yy]); D,r=projected[:,:-1],projected[:,-1]; w,_=m._box_budget(D.T@D+rs*np.eye(D.shape[1]),-D.T@r,cap,budget); gamma=np.linalg.solve(G,a.T@(yy-t@w)); e[test]=y[test]-S[test]@w-z[test]@gamma; wd.append(np.max(abs(w-full)))
    rms=lambda r:np.sqrt(np.mean(r*r)); print(name,'OOF pooled RMS',rms(e),'worst rosette RMS',max(rms(e[blocks==j]) for j in np.unique(blocks)),'bank RMS',[rms(e[ids==j]) for j in np.unique(ids)],'max weight deviation',max(wd))
PY
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`551 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
single-bank calibration layout bank_id ndarray (24,) 
single-bank calibration layout ref_signal ndarray (24,) 
single-bank calibration layout raw_readout ndarray (24,) 
specified nuisance OOF pooled RMS 0.0077581015050804565 worst rosette RMS 0.013784810975415177 bank RMS [0.008038687427786635, 0.00798297864104011, 0.008204089987316348, 0.006715431466971032] max weight deviation 0.012553287294002333
additional bank offsets OOF pooled RMS 0.007952755812831749 worst rosette RMS 0.015205385472639207 bank RMS [0.008001677979066493, 0.008528200973461455, 0.008451606596271444, 0.006693174410662795] max weight deviation 0.02216375021092204
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=42772b83fc1b03fc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_42772b83 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`19528 ms` LLM)
- **Rationale**: The previous audit identified the single-bank data layout but did not test the readings. This check can detect a gain or offset mismatch under isolated excitation; agreement at the sweep noise scale would close the remaining calibration question without changing or revalidating the artifact.
- **Alternatives Weighed**: `Add bank offsets to the nuisance design` (rejected: They worsened both pooled and worst-rosette held-out prediction and would change the prescribed objective.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle
import numpy as np
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    c = pickle.load(f)[0]['study']['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
b = np.linalg.lstsq(np.c_[np.ones(len(x)), x], y, rcond=None)[0]
s = c['single_bank_sweeps']
bank, signal, raw = (np.asarray(s[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
for j in np.unique(bank):
    mask = bank == j
    pred = b[0,j] + b[j+1,j]*signal[mask]
    error = raw[mask]-pred
    slope, intercept = np.linalg.lstsq(np.c_[signal[mask], np.ones(mask.sum())], raw[mask], rcond=None)[0]
    print(f'bank {j}: n={mask.sum()}, transfer RMS={np.sqrt(np.mean(error**2)):.6g}, bias={error.mean():.6g}, slope delta={slope-b[j+1,j]:.6g}, intercept delta={intercept-b[0,j]:.6g}')
PY
```
- **Container Exit Code**: `0` (`371 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
bank 0: n=6, transfer RMS=1.31599e-05, bias=1.88512e-06, slope delta=5.04852e-06, intercept delta=1.12784e-06
bank 1: n=6, transfer RMS=7.00467e-06, bias=2.19943e-06, slope delta=4.33114e-06, intercept delta=1.54976e-06
bank 2: n=6, transfer RMS=9.198e-06, bias=-5.1484e-06, slope delta=2.98009e-06, intercept delta=-5.59542e-06
bank 3: n=6, transfer RMS=4.22329e-06, bias=3.90799e-06, slope delta=-7.36793e-07, intercept delta=4.01851e-06

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=42772b83fc1b03fc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_42772b83 (directory).
```

### Turn 10 — `submit_completed_task` (`15199 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema validation, independent forward and terminal-field checks, and KKT stress tests spanning mixed bounds, vertices, and endpoint budgets. The follow-up audits found no change supported by the measurements.
- **Alternatives Weighed**: `Add nonlinear calibration terms or extra bank offsets` (rejected: Both worsened held-out prediction, and the affine fit transferred to the independent single-bank sweeps at the calibration noise scale.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The isolated public schema check passed, and the maximum weight error across 240 known-solution cases was 5.83e-16. The solver fits calibration from the supplied sweeps and uses the prescribed Robin fallback because no boundary diagnostics or archive were available.`

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
