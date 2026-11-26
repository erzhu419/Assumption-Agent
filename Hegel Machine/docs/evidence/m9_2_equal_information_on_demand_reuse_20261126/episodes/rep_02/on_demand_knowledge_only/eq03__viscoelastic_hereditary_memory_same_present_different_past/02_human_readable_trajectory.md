# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `eq03__viscoelastic_hereditary_memory_same_present_different_past`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M92-EqualInfo-4` |
| **Domain** | `Multi-Scale Viscoelastic Hereditary Memory Rebinding & Coupon-Calibrated Creep Hazard Forecasting` |
| **Wave / Wormhole Family** | `m92_equal_info` / `fam_multiscale_memory_and_outlier_deconfounding` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`PASS (1.0)`** |
| **Failure / Outcome Taxonomy** | `PASSED_ALL_CHECKS` |
| **Turns Used / Max Turns** | `9 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `210,000` (`202,014` / `45,056` / `7,986`) |
| **Total Episode Wall Time (s)** | `272.48s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`16674 ms` LLM)
- **Rationale**: Inspect the public schema and available targets to identify viable validation slices and avoid shape mismatches; consult the reference catalog before implementing the specified estimators.

```bash
ls -lh /holdout_task; sed -n '1,230p' /holdout_task/validate_public_schema.py; python3 /tmp/hegel_knowledge_ref/describe_ref.py --search 'viscoelastic'; python3 /tmp/hegel_knowledge_ref/describe_ref.py --search 'Cauchy'; python3 /tmp/hegel_knowledge_ref/describe_ref.py --search 'alignment'; python3 - <<'PY'
import pickle
with open('/holdout_task/public_eq03_datasets.pkl','rb') as f: d=pickle.load(f)
def inspect(x,depth=0):
 if hasattr(x,'shape'): return (x.shape,str(x.dtype))
 if isinstance(x,dict): return {k:inspect(v,depth+1) for k,v in x.items()} if depth<4 else list(x)
 if isinstance(x,(list,tuple)): return (len(x),[inspect(v,depth+1) for v in x[:3]])
 return str(x)[:200]
print(inspect(d))
PY
```
- **Container Exit Code**: `1` (`618 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit
autogenous_prim_robust_affine_projection@abc35d3d | profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | For predictor, target and optional weights (n, m) and predictor sensitivity (n | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_robust_affine_projection.profile_weighted_affine
autogenous_prim_conservative_spd_solve@05cb4387 | solve_residual_checked_cg(operator: np.ndarray | sparse.spmatrix | sparse.sparray | LinearOperator, rhs: np.ndarray, *, preconditioner: LinearOperator | None=None, tol: float=1e-08, maxiter: int=450, refinements: int=4) -> tuple[np.ndarray, float] | Solve a real SPD operator of shape (n, n) against rhs of shape (n,), returning | REQ: grid op assumes isotropic scalar coeff>0 + zero Dirichlet ghosts; CG/AMG need square SPD | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_conservative_spd_solve.solve_residual_checked_cg
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_reference_panel_balancing.weighted_column_fill
No matching reference cards found.
```
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task': No such file or directory
sed: can't read /holdout_task/validate_public_schema.py: No such file or directory
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
FileNotFoundError: [Errno 2] No such file or directory: '/holdout_task/public_eq03_datasets.pkl'
```

### Turn 2 — `execute_in_remote_docker` (`14848 ms` LLM)
- **Rationale**: Check whether the public data and validator live under the visible pretrain mount. Their location determines whether validation can use real measurements or must fall back to synthetic tests.
- **Alternatives Weighed**: `Proceed directly to synthetic validation` (rejected: Inspect the visible task mount first so real measurements can be used if available.)

```bash
ls -la / /pretrain_task; find /pretrain_task /app /workspace /results -maxdepth 4 -type f \( -name '*.pkl' -o -name 'validate_public_schema.py' -o -name '*viscoelastic*' \) -printf '%p %s bytes\n' 2>/dev/null; R=/tmp/hegel_knowledge_ref; python3 "$R/describe_ref.py" --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit; python3 "$R/describe_ref.py" --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights; find "$R" -name '*.reference.txt' | grep -E 'robust_trend|reference_panel'
```
- **Container Exit Code**: `0` (`495 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
code_sha256": "a0bd4b961c67e2ccad723d94cee31eee33e031d5925b39c96b71ace643383209",
  "mathematical_preconditions": [
    "Alignment accepts real numeric arrays of shapes (n, p) and (n, q), with p, q > 0, and a finite real group vector of shape (n,). Nonfinite panel entries are treated as missing. Each group must have at least one control with two finite observations, a finite median control mean, and a finite, strictly positive median control population standard deviation. A constant control can contribute; a zero median gain raises ValueError.",
    "Interpreting alignment as removal of a nuisance transform assumes a shared additive offset and positive multiplicative gain for signals and controls within each group. The control panel must have comparable baseline location and spread across groups, independent of sample composition. These scientific assumptions cannot be verified by shape or moment checks.",
    "Cell weighting requires two nonempty, finite real vectors of matching shape (n,). Every combination of an observed group and an observed stratum must be present. If there are K such cells, each row in a cell of size m receives weight n/(K*m).",
    "Column fill requires a nonempty real matrix of shape (n, p) and finite, strictly positive row weights of shape (n,). Nonfinite entries are excluded from both numerator and denominator; a column with no finite observations returns zero. Numeric panel and fill arithmetic use float64 and assume intermediate sums, squares, and ratios are within usable float64 range."
  ],
  "function_level_applicability": {
    "align_reference_panel": "APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)",
    "balanced_cell_weights": "APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)",
    "weighted_column_fill": "APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)"
  },
  "functions": [
    {
      "function": "balanced_cell_weights",
      "signature": "balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray",
      "doc_summary": "Return positive mean-one weights (n,) from nonempty finite real vectors groups (n,) and strata (n,) with every observed group\u2013stratum combination present."
    }
  ],
  "reference_source_txt": "/tmp/hegel_knowledge_ref/autogenous_prim_reference_panel_balancing.reference.txt"
}
/tmp/hegel_knowledge_ref/autogenous_prim_robust_trend_decomposition.reference.txt
/tmp/hegel_knowledge_ref/autogenous_prim_reference_panel_balancing.reference.txt
```

### Turn 3 — `execute_in_remote_docker` (`45788 ms` LLM)
- **Rationale**: Implement the specified physical model and robust estimator without depending on the missing data mount. Next, use independent synthetic oracles to check alignment, history order, outlier resistance, and first crossings.

```bash
set -e
R=/tmp/hegel_knowledge_ref
sed -n '/^def cauchy_linear_fit(/,/^def fit_anchored_relaxation(/p' "$R/autogenous_prim_robust_trend_decomposition.reference.txt" | head -85
sed -n '/^def align_reference_panel(/,$p' "$R/autogenous_prim_reference_panel_balancing.reference.txt" | head -140
mkdir -p /holdout_task
cat > /holdout_task/viscoelastic_hereditary_forecast.py <<'PY'
"""Coupon-calibrated two-mode hereditary creep and threshold forecasts."""

import numpy as np


def _aligned_features(train, test):
    def pooled(key):
        return np.concatenate((train[key], test[key]), axis=0)

    labs = pooled('lab_id')
    strata = pooled('stratum')
    features = pooled('coupon_features_raw').astype(float, copy=False)
    controls = pooled('ref_coupons').astype(float, copy=False)
    for lab in np.unique(labs):
        rows = labs == lab
        panel = controls[rows]
        counts = np.isfinite(panel).sum(axis=0)
        offset = np.median(np.nanmean(panel[:, counts > 0], axis=0))
        gain = np.median(np.nanstd(panel[:, counts >= 2], axis=0, ddof=0))
        features[rows] = (features[rows] - offset) / gain

    _, cell, counts = np.unique(np.column_stack((labs, strata)), axis=0,
                                return_inverse=True, return_counts=True)
    weights = (len(labs) / (len(counts) * counts[cell]))[:, None]
    finite = np.isfinite(features)
    mass = np.sum(weights * finite, axis=0)
    sums = np.sum(weights * np.where(finite, features, 0.0), axis=0)
    means = np.divide(sums, mass, out=np.zeros_like(sums), where=mass > 0)
    features = np.where(finite, features, means)
    n_train = len(train['lab_id'])
    return features[:n_train], features[n_train:]


def _design(split, features, times, taus, dt):
    history = np.asarray(split['past_stress_history'], dtype=float)
    hold = np.asarray(split['stress_hold_0'], dtype=float)
    decay = np.exp(-dt / taus)
    injection = -np.expm1(-dt / taus)
    state = np.zeros((len(hold), len(taus)))
    for stress in history.T:
        state = decay * state + injection * stress[:, None]
    growth = -np.expm1(-times[:, None] / taus)

    design = np.empty((len(hold), len(times), 3 + features.shape[1]))
    design[:, :, :2] = (hold[:, None] - state)[:, None, :] * growth[None, :, :]
    design[:, :, 2] = times
    design[:, :, 3:] = features[:, None, :] * times[None, :, None]
    return design


def _fit_cauchy(design, target):
    theta = np.linalg.lstsq(design, target, rcond=None)[0]
    # Refresh the residual scale after robust warm-up so initial spikes do not
    # set the bandwidth of the final Cauchy fit.
    for iterations in (8, 32):
        residual = target - design @ theta
        mad = np.median(np.abs(residual - np.median(residual)))
        scale = max(2.385 * 1.4826 * mad, 1e-8)
        for _ in range(iterations):
            root_weight = 1.0 / np.hypot(1.0, residual / scale)
            update = np.linalg.lstsq(design * root_weight[:, None],
                                     target * root_weight, rcond=None)[0]
            delta = np.linalg.norm(update - theta)
            theta = update
            if delta <= 1e-10 * (1.0 + np.linalg.norm(theta)):
                break
            residual = target - design @ theta
    return theta


def _crossing_times(times, initial, predicted, critical):
    crossing = np.full(len(initial), times[-1], dtype=float)
    at_start = initial >= critical
    crossing[at_start] = 0.0
    above = predicted >= critical[:, None]
    rows = np.flatnonzero(~at_start & above.any(axis=1))
    right = above[rows].argmax(axis=1)
    knots = np.r_[0.0, times]
    t_a, t_b = knots[right], knots[right + 1]
    y_a = np.column_stack((initial, predicted))[rows, right]
    y_b = predicted[rows, right]
    crossing[rows] = t_a + (critical[rows] - y_a) / (y_b - y_a) * (t_b - t_a)
    return crossing


def solve_viscoelastic_hereditary_forecast(dataset: dict) -> dict:
    """Fit the causal Prony model on training curves and forecast test coupons."""
    train, test = dataset['train'], dataset['test']
    times = np.asarray(dataset['eval_times'], dtype=float)
    taus = np.asarray(dataset['memory_taus'], dtype=float)
    dt = float(dataset['history_dt'])
    x_train, x_test = _aligned_features(train, test)
    design = _design(train, x_train, times, taus, dt)
    strain0 = np.asarray(train['strain_0'], dtype=float)
    target = np.asarray(train['strain_curves_obs'], dtype=float) - strain0[:, None]
    theta = _fit_cauchy(design.reshape(-1, design.shape[-1]), target.ravel())

    initial = np.asarray(test['strain_0'], dtype=float)
    predicted = initial[:, None] + _design(test, x_test, times, taus, dt) @ theta
    critical = np.asarray(test['critical_strain'], dtype=float)
    return {
        'aligned_test_features': x_test,
        'hereditary_weights': theta,
        'predicted_strain': predicted,
        'crossing_times': _crossing_times(times, initial, predicted, critical),
    }
PY
```
- **Container Exit Code**: `0` (`398 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoelastic_hereditary_forecast.py']`
- **Stdout (tail)**:
```text
d_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Return positive mean-one weights (n,) from nonempty finite real vectors groups (n,) and strata (n,) with every observed group–stratum combination present."""
    groups = _real_array(groups, 'groups')
    strata = _real_array(strata, 'strata')
    if groups.ndim != 1 or not groups.size or strata.shape != groups.shape:
        raise ValueError('groups and strata must be matching nonempty vectors')
    if not np.isfinite(groups).all() or not np.isfinite(strata).all():
        raise ValueError('groups and strata must be finite')
    (group_levels, group_codes) = np.unique(groups, return_inverse=True)
    (stratum_levels, stratum_codes) = np.unique(strata, return_inverse=True)
    pairs = np.column_stack((group_codes, stratum_codes))
    (_, inverse, counts) = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    if counts.size != group_levels.size * stratum_levels.size:
        raise ValueError('every group must contain every observed stratum')
    return groups.size / counts.size / counts[inverse.reshape(-1)]


def weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return finite-entry weighted means (p,) for a nonempty real matrix values (n, p) with finite positive weights (n,), using zero for unobserved columns."""
    values = _real_array(values, 'values').astype(float, copy=False)
    weights = _real_array(weights, 'weights').astype(float, copy=False)
    if values.ndim != 2 or not values.shape[0] or (not values.shape[1]):
        raise ValueError('values must be a nonempty matrix')
    if weights.shape != (values.shape[0],):
        raise ValueError('weights must be a vector matching value rows')
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('weights must be finite and strictly positive')
    observed = np.isfinite(values)
    support = weights @ observed
    numerator = weights @ np.where(observed, values, 0.0)
    if not np.isfinite(support).all() or not np.isfinite(numerator).all():
        raise ValueError('weighted column sums must be finite')
    return np.divide(numerator, support, out=np.zeros(values.shape[1]), where=support > 0)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=36f47144e48cf935 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_36f47144 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`62848 ms` LLM)
- **Rationale**: With public measurements unavailable, verify the exact artifact in an isolated process against independent synthetic oracles. Lab-cell and horizon slices can expose localized forecast failures even when pooled error passes; direct crossing checks separate interpolation errors from fit errors.
- **Alternatives Weighed**: `Run an import-only smoke test` (rejected: It would miss history-order errors, outlier sensitivity, incorrect cell weighting, and off-by-one crossing interpolation.)

```bash
cat > /tmp/verify_eq03.py <<'PY'
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('forecast', '/holdout_task/viscoelastic_hereditary_forecast.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Check first crossings independently of fit quality: immediate, repeated,
# exact knot, censored, and interpolation on a later segment.
y0 = np.array([1., 0., 0., 0., 0.])
y = np.array([[0., 0., 0.], [2., 0., 2.], [0., 1., 0.], [.3, .5, .9], [.2, .6, 1.4]])
np.testing.assert_allclose(mod._crossing_times(np.array([1., 2., 4.]), y0, y, np.ones(5)), [0., .5, 2., 4., 3.])

def crossing(t, initial, curves, crit):
    result = []
    for start, curve, cap in zip(initial, curves, crit):
        hit = t[-1]
        if start >= cap:
            result.append(0.)
            continue
        prev_t, prev_y = 0., start
        for next_t, next_y in zip(t, curve):
            if next_y >= cap:
                hit = prev_t + (next_t - prev_t) * (cap - prev_y) / (next_y - prev_y)
                break
            prev_t, prev_y = next_t, next_y
        result.append(hit)
    return np.array(result)

worst = np.zeros(7)
min_twin_gap = np.inf
for seed in range(12):
    rng = np.random.default_rng(seed)
    n_train, n_test = 240, 98
    n = n_train + n_test
    cell = rng.choice(8, n, p=[.34, .04, .08, .07, .07, .15, .20, .05])
    cell[:8] = np.arange(8)
    lab, stratum = cell // 2, cell % 2
    gain = np.array([.55, 1.7, 2.4, .8])[lab]
    offset = np.array([-3., 1.3, 2., -1.7])[lab]
    raw = offset[:, None] + gain[:, None] * (rng.normal(size=(n, 4)) + 1.8 * stratum[:, None])
    refs = offset[:, None] + gain[:, None] * rng.normal(size=(n, 4))
    raw[rng.random(raw.shape) < .05] = np.nan
    refs[rng.random(refs.shape) < .08] = np.nan
    hold = rng.uniform(.8, 2.8, n)
    initial = rng.normal(.1, .04, n)
    rate = rng.normal(size=n)
    history = rng.uniform(.1, 3.3, (n, 40)) + rng.uniform(-.2, 1.2, (n, 1))
    twins = np.arange(n_train, n_train + 20, 2)
    for arr in (cell, lab, stratum, raw, hold, initial, rate):
        arr[twins + 1] = arr[twins]
    history[twins, -12:] += .9
    history[twins + 1] = history[twins]
    history[twins + 1, -12:] -= 1.7

    aligned = raw.copy()
    for group in range(4):
        ix = np.flatnonzero(lab == group)
        col = [[v for v in refs[ix, j] if np.isfinite(v)] for j in range(4)]
        mu = np.median([sum(a) / len(a) for a in col if a])
        sigma = np.median([np.std(a) for a in col if len(a) > 1])
        aligned[ix] = (raw[ix] - mu) / sigma
    for j in range(4):
        keep = np.isfinite(aligned[:, j])
        values = []
        weights = []
        for c in range(8):
            rows = cell == c
            values.extend(aligned[rows & keep, j])
            weights.extend([1 / rows.sum()] * np.count_nonzero(rows & keep))
        aligned[~keep, j] = np.average(values, weights=weights)

    dt = .25
    taus = np.array([1., 4.5])
    times = np.r_[0.07, np.sort(rng.uniform(.13, 7.9, size=37)), 8.]
    theta = np.array([.17, .31, .055, .025, -.014, .008, .018])
    # Closed-form convolution and sum of separately evaluated processes.
    z = np.column_stack([history @ (np.exp(-np.arange(39, -1, -1) * dt / tau) * (1 - np.exp(-dt / tau))) for tau in taus])
    clean = initial[:, None] + (theta[2] + aligned @ theta[3:])[:, None] * times
    for j, tau in enumerate(taus):
        clean += theta[j] * (hold - z[:, j])[:, None] * (1 - np.exp(-times / tau))
    obs = clean[:n_train].copy()
    if seed % 4:
        obs += rng.normal(0, [.0, .002, .008, .014][seed % 4], obs.shape)
    spike = rng.random(obs.shape) < .09
    obs[spike] += np.clip(rng.standard_cauchy(size=spike.sum()) * .65, -.95, .95)
    crit = initial[n_train:] + rng.uniform(.04, .53, n_test)
    crit[:2] = initial[n_train:n_train+2] - .01
    crit[-2:] = clean[-2:].max(axis=1) + np.abs(initial[-2:]) + 1.
    test = slice(n_train, None)
    def split(s):
        return dict(lab_id=lab[s], stratum=stratum[s], coupon_features_raw=raw[s], ref_coupons=refs[s], past_stress_history=history[s], stress_hold_0=hold[s], strain_0=initial[s], instant_rate_0=rate[s])
    d = dict(train=dict(split(slice(None, n_train)), strain_curves_obs=obs), test=dict(split(test), critical_strain=crit), eval_times=times, memory_taus=taus, history_dt=dt)
    o = mod.solve_viscoelastic_hereditary_forecast(d)
    shapes = dict(aligned_test_features=(n_test, 4), hereditary_weights=(7,), predicted_strain=(n_test, len(times)), crossing_times=(n_test,))
    assert set(o) == set(shapes)
    for k, shape in shapes.items():
        assert o[k].shape == shape and np.isfinite(o[k]).all(), (k, o[k].shape)
    np.testing.assert_allclose(o['aligned_test_features'], aligned[test], atol=1e-12, rtol=0)
    np.testing.assert_allclose(o['crossing_times'], crossing(times, initial[test], o['predicted_strain'], crit), atol=1e-12, rtol=0)
    e = np.abs(o['predicted_strain'] - clean[test])
    h = np.abs(o['crossing_times'] - crossing(times, initial[test], clean[test], crit))
    gap = clean[twins + 1] - clean[twins]
    gap_pred = o['predicted_strain'][twins + 1 - n_train] - o['predicted_strain'][twins - n_train]
    min_twin_gap = min(min_twin_gap, float(gap[:, -1].min()))
    metrics = [np.linalg.norm(o['hereditary_weights'] - theta) / np.linalg.norm(theta), e.mean(), e[:20].mean(), h.mean(), max(e[cell[test] == c].mean() for c in np.unique(cell[test])), max(e[:, j].mean() for j in range(len(times))), np.abs(gap - gap_pred).mean()]
    worst = np.maximum(worst, metrics)
    assert np.all(np.array(metrics) < [.10, .08, .08, .18, .08, .08, .08]), (seed, metrics)
print('12 synthetic held-out test sets; alignment and crossing algorithm agree with independent oracles to 1e-12')
print('Worst errors [rel theta, strain MAE, twin MAE, crossing MAE, cell strain MAE, horizon strain MAE, twin gap MAE]:', np.array2string(worst, precision=8))
print('Minimum terminal twin strain gap:', min_twin_gap)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/verify_eq03.py
```
- **Container Exit Code**: `0` (`690 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
12 synthetic held-out test sets; alignment and crossing algorithm agree with independent oracles to 1e-12
Worst errors [rel theta, strain MAE, twin MAE, crossing MAE, cell strain MAE, horizon strain MAE, twin gap MAE]: [0.003833   0.00028662 0.00030715 0.00215813 0.00060529 0.0004631
 0.00050982]
Minimum terminal twin strain gap: 0.4876086480173398

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=36f47144e48cf935 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_36f47144 (directory).
```

### Turn 5 — `submit_completed_task` (`14394 ms` LLM)
- **Rationale**: The exact artifact passed isolated stress tests for all four outputs. Alignment and first-crossing interpolation matched independent oracles to 1e-12, and synthetic forecast errors stayed below the specified tolerances.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`13` turns remaining; artifact status={'/holdout_task/viscoelastic_hereditary_forecast.py': 'FILE:4585'}): Do NOT lock prematurely with `13` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoelastic_hereditary_forecast.py`. Across 12 synthetic holdouts, worst-case relative weight error was 0.38%, twin trajectory MAE was 0.00031, and crossing-time MAE was 0.00216. The public dataset was not mounted, so public-data accuracy was not verified.`

### Turn 6 — `execute_in_remote_docker` (`30407 ms` LLM)
- **Rationale**: Check once whether the public measurements are now available. The additional isolated stress test distinguishes ordered hereditary memory from a shortcut based on instantaneous features, endpoints, or total historical stress: each twin pair preserves all three but changes the order of loading. A failure would direct debugging to the history-state calculation rather than calibration.
- **Alternatives Weighed**: `Add rational terms or test relay modes and classification balance` (rejected: The contract specifies a linear hereditary trajectory with continuous crossing times, and supplies neither relay modes nor classification labels. The applicable algebraic checks are the signed hold-minus-memory transients and the pooled lab calibration.)

```bash
set -e
cd /holdout_task
if [ -f validate_public_schema.py ] && [ -f public_eq03_datasets.pkl ]; then
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I validate_public_schema.py
elif [ -f public_eq03_datasets.pkl ]; then
  python3 -I - <<'PY'
import pickle
with open('public_eq03_datasets.pkl', 'rb') as f:
    obj = pickle.load(f)
print('Public data available:', type(obj).__name__, list(obj)[:12] if isinstance(obj, dict) else len(obj))
PY
else
  echo 'Public measurement file is still absent.'
fi
python3 - <<'PY'
from pathlib import Path
src = Path('/tmp/verify_eq03.py').read_text()
old = '''    history[twins, -12:] += .9
    history[twins + 1] = history[twins]
    history[twins + 1, -12:] -= 1.7'''
new = '''    # Identical history multisets, stress integrals, and endpoint values.
    history[twins, 1:17] -= .85
    history[twins, 23:39] += .85
    history[twins + 1] = history[twins]
    history[twins + 1, 1:17] = history[twins, 23:39]
    history[twins + 1, 23:39] = history[twins, 1:17]
    np.testing.assert_array_equal(np.sort(history[twins], axis=1), np.sort(history[twins + 1], axis=1))'''
assert old in src
src = src.replace(old, new).replace('min_twin_gap = np.inf', 'min_twin_gap = np.inf\nworst_twin_lower_bound = 0.')
src = src.replace('    min_twin_gap =', '    worst_twin_lower_bound = max(worst_twin_lower_bound, np.abs(gap).mean() / 2)\n    min_twin_gap =')
src += '\nprint("Worst seed lower bound on order-blind twin MAE:", worst_twin_lower_bound)\nassert worst_twin_lower_bound > .08\n'
Path('/tmp/verify_eq03_history_order.py').write_text(src)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/verify_eq03_history_order.py
```
- **Container Exit Code**: `0` (`1106 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public measurement file is still absent.
12 synthetic held-out test sets; alignment and crossing algorithm agree with independent oracles to 1e-12
Worst errors [rel theta, strain MAE, twin MAE, crossing MAE, cell strain MAE, horizon strain MAE, twin gap MAE]: [0.003833   0.00028514 0.00030409 0.00210812 0.00060529 0.00046224
 0.00045923]
Minimum terminal twin strain gap: 0.1622290913920128
Worst seed lower bound on order-blind twin MAE: 0.17830690629145154

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=36f47144e48cf935 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_36f47144 (directory).
```

### Turn 7 — `submit_completed_task` (`17343 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated tests with wide margins for all four outputs. Alignment and crossing interpolation matched independent oracles to 1e-12, and the reordered-history stress test confirmed that the recurrence preserves the information needed to distinguish twins.
- **Alternatives Weighed**: `Replace the ordered recurrence with stress-history summaries` (rejected: Twins with identical history values and endpoints but different loading order had an order-blind error lower bound above the 0.08 trajectory tolerance.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/viscoelastic_hereditary_forecast.py': 'FILE:4585'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/viscoelastic_hereditary_forecast.py` and verified all four outputs in isolated Python runs. Across 12 synthetic holdouts with identical-present twins whose loading histories differed only in order, worst-case relative weight error was 0.38%, twin trajectory MAE was 0.00030, and crossing-time MAE was 0.00211. The public dataset was not mounted, so public-data accuracy remains unverified.`

### Turn 8 — `execute_in_remote_docker` (`20365 ms` LLM)
- **Rationale**: Search for the named public fixture outside the paths already checked without reading or copying data. If found, run the public audit; otherwise retain the verified artifact and report that validation was synthetic only.
- **Alternatives Weighed**: `Repeat the completed synthetic checks or add relay and classification tests` (rejected: The exact artifact already passed isolated synthetic checks, and the forecast schema has neither relay states nor classification labels.)

```bash
find / -xdev \( -path /proc -o -path /sys -o -path /dev \) -prune -o -type f \( -name public_eq03_datasets.pkl -o -name validate_public_schema.py \) -print 2>/dev/null; if [ -f /holdout_task/public_eq03_datasets.pkl ]; then ls -lh /holdout_task/public_eq03_datasets.pkl; fi
```
- **Container Exit Code**: `0` (`545 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=36f47144e48cf935 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_36f47144 (directory).
```

### Turn 9 — `submit_completed_task` (`16772 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated stress tests with all four outputs within tolerance. Independent oracles verified alignment and first-crossing interpolation, and reordered-history twins confirmed that the forecast distinguishes loading order. Repeating those checks would not close the missing public-data validation gap.
- **Alternatives Weighed**: `Run the public-data audit` (rejected: The named dataset and validator were not found in the container.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/viscoelastic_hereditary_forecast.py` and verified all four outputs in isolated Python runs. Across 12 synthetic holdouts with identical-present twins, worst-case relative weight error was 0.38%, twin trajectory MAE was 0.00030, and crossing-time MAE was 0.00211. The public dataset was unavailable, so accuracy on public measurements remains unverified.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/viscoelastic_hereditary_forecast.py": "FILE:4585"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "all_valid": true,
    "max_aligned_feat_mae": 0.11361,
    "max_hereditary_w_rel_l2": 0.0615,
    "max_strain_mae": 0.0437,
    "max_twin_same_present_mae": 0.04027,
    "max_crossing_time_mae": 0.09649,
    "student_elapsed_sec": 0.0957,
    "instance_diagnostics": [
      {
        "seed": 99421,
        "feat_mae": 0.10318,
        "w_rel_l2": 0.0615,
        "strain_mae": 0.0437,
        "twin_mae": 0.04027,
        "cross_mae": 0.09649
      },
      {
        "seed": 99422,
        "feat_mae": 0.1073,
        "w_rel_l2": 0.02066,
        "strain_mae": 0.03362,
        "twin_mae": 0.03488,
        "cross_mae": 0.03316
      },
      {
        "seed": 99423,
        "feat_mae": 0.11361,
        "w_rel_l2": 0.00813,
        "strain_mae": 0.04319,
        "twin_mae": 0.03824,
        "cross_mae": 0.08603
      }
    ]
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_aligned_feat_mae\": 0.11361, \"max_hereditary_w_rel_l2\": 0.0615, \"max_strain_mae\": 0.0437, \"max_twin_same_present_mae\": 0.04027, \"max_crossing_time_mae\": 0.09649, \"student_elapsed_sec\": 0.0957, \"instance_diagnostics\": [{\"seed\": 99421, \"feat_mae\": 0.10318, \"w_rel_l2\": 0.0615, \"strain_mae\": 0.0437, \"twin_mae\": 0.04027, \"cross_mae\": 0.09649}, {\"seed\": 99422, \"feat_mae\": 0.1073, \"w_rel_l2\": 0.02066, \"strain_mae\": 0.03362, \"twin_mae\": 0.03488, \"cross_mae\": 0.03316}, {\"seed\": 99423, \"feat_mae\": 0.11361, \"w_rel_l2\": 0.00813, \"strain_mae\": 0.04319, \"twin_mae\": 0.03824, \"cross_mae\": 0.08603}]}\n__HEGEL_VERIFIER_AUTH_0a0f6db4f16e82ca8d4ccdd206488ea7__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_aligned_feat_mae\": 0.11361, \"max_hereditary_w_rel_l2\": 0.0615, \"max_strain_mae\": 0.0437, \"max_twin_same_present_mae\": 0.04027, \"max_crossing_time_mae\": 0.09649, \"student_elapsed_sec\": 0.0957, \"instance_diagnostics\": [{\"seed\": 99421, \"feat_mae\": 0.10318, \"w_rel_l2\": 0.0615, \"strain_mae\": 0.0437, \"twin_mae\": 0.04027, \"cross_mae\": 0.09649}, {\"seed\": 99422, \"feat_mae\": 0.1073, \"w_rel_l2\": 0.02066, \"strain_mae\": 0.03362, \"twin_mae\": 0.03488, \"cross_mae\": 0.03316}, {\"seed\": 99423, \"feat_mae\": 0.11361, \"w_rel_l2\": 0.00813, \"strain_mae\": 0.04319, \"twin_mae\": 0.03824, \"cross_mae\": 0.08603}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5908.5,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
