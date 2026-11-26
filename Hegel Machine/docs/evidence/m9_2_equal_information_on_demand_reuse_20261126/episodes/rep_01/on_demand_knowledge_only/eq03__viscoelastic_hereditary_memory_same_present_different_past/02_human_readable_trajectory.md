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
| **Turns Used / Max Turns** | `10 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `230,895` (`221,577` / `69,760` / `9,318`) |
| **Total Episode Wall Time (s)** | `323.56s` |

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

### Turn 1 — `execute_in_remote_docker` (`14485 ms` LLM)
- **Rationale**: Inspect the public validator to determine how to measure trajectory and hazard errors, and consult the mounted numerical references before implementing the fit. Keep all dataset inspection inside the container.
- **Alternatives Weighed**: `Fit directly from instantaneous observables` (rejected: The specified twins have identical instantaneous features, so those features cannot account for their divergent trajectories.)

```bash
cd /holdout_task && ls -lh && sed -n '1,200p' validate_public_schema.py; if [ -f /tmp/hegel_knowledge_ref/describe_ref.py ]; then python3 /tmp/hegel_knowledge_ref/describe_ref.py --search 'viscoelastic relaxation'; python3 /tmp/hegel_knowledge_ref/describe_ref.py --search 'Cauchy IRLS'; python3 /tmp/hegel_knowledge_ref/describe_ref.py --search 'reference panel alignment imputation'; fi
```
- **Container Exit Code**: `0` (`435 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
trol col with >=2 finite obs & spread>0 per group | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_reference_panel_balancing.weighted_column_fill
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```
- **Stderr (tail)**:
```text
bash: line 1: cd: /holdout_task: No such file or directory
```

### Turn 2 — `execute_in_remote_docker` (`16895 ms` LLM)
- **Rationale**: Locate the public data and validator without transferring them, and inspect the reference method preconditions before implementing alignment and robust fitting.
- **Alternatives Weighed**: `Switch immediately to synthetic validation` (rejected: The initial snapshot names /pretrain_task; a bounded search may locate public measurements and avoid replacing them with synthetic evidence.)

```bash
printf '%s\n' '--- task files ---'; ls -la /pretrain_task; find /pretrain_task /workspace /app /results -maxdepth 3 -type f \( -name '*eq03*' -o -name '*schema*' -o -name 'AGENTS.md' \) -print 2>/dev/null; printf '%s\n' '--- reference methods ---'; ref=/tmp/hegel_knowledge_ref/describe_ref.py; for method in autogenous_prim_robust_trend_decomposition.cauchy_linear_fit autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation autogenous_prim_reference_panel_balancing.align_reference_panel autogenous_prim_reference_panel_balancing.balanced_cell_weights autogenous_prim_reference_panel_balancing.weighted_column_fill; do python3 "$ref" --describe "$method"; done; find /tmp/hegel_knowledge_ref -maxdepth 4 -name '*.reference.txt' | grep -E 'robust_trend|reference_panel|viscoelastic'
```
- **Container Exit Code**: `0` (`656 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"compiled_code_sha256": "a0bd4b961c67e2ccad723d94cee31eee33e031d5925b39c96b71ace643383209",
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
      "function": "weighted_column_fill",
      "signature": "weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray",
      "doc_summary": "Return finite-entry weighted means (p,) for a nonempty real matrix values (n, p) with finite positive weights (n,), using zero for unobserved columns."
    }
  ],
  "reference_source_txt": "/tmp/hegel_knowledge_ref/autogenous_prim_reference_panel_balancing.reference.txt"
}
/tmp/hegel_knowledge_ref/autogenous_prim_robust_trend_decomposition.reference.txt
/tmp/hegel_knowledge_ref/autogenous_prim_reference_panel_balancing.reference.txt
```

### Turn 3 — `execute_in_remote_docker` (`15388 ms` LLM)
- **Rationale**: Search by exact filenames to determine whether public validation is possible. Read only the applicable reference implementations to confirm finite-entry balancing and Cauchy weighting before writing the standalone solver.
- **Alternatives Weighed**: `Reuse the anchored relaxation fitter` (rejected: It estimates a single relaxation from onset observables; this task supplies two fixed time constants and requires integrating each coupon's stress history.)

```bash
find / -xdev -type f \( -name 'public_eq03_datasets.pkl' -o -name 'validate_public_schema.py' \) -print 2>/dev/null; for f in /tmp/hegel_knowledge_ref/autogenous_prim_{robust_trend_decomposition,reference_panel_balancing}.reference.txt; do grep -A 40 -E '^def (cauchy_linear_fit|align_reference_panel|balanced_cell_weights|weighted_column_fill)\(' "$f"; done
```
- **Container Exit Code**: `0` (`994 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
gain <= 0:
            raise ValueError(f'group {level!r} requires a finite offset and positive finite gain')
        block = values[rows]
        result[rows] = (np.where(np.isfinite(block), block, np.nan) - offset) / gain
    return result


def balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray:
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
```

### Turn 4 — `execute_in_remote_docker` (`52547 ms` LLM)
- **Rationale**: Implement the specified linear-in-parameters law and use Cauchy weights to limit transducer-spike influence. Next, test coefficient recovery, twin divergence, and first-crossing interpolation independently in a clean process.
- **Alternatives Weighed**: `Fit the supplied relaxation times or add instantaneous strain rate as a predictor` (rejected: The contract fixes both relaxation times and specifies exactly seven coefficients; instantaneous strain rate does not distinguish the twins.)

```bash
set -eu
mkdir -p /holdout_task
cat > /holdout_task/viscoelastic_hereditary_forecast.py <<'PY'
"""Coupon-calibrated creep forecasts with two causal Prony memory modes."""

import numpy as np


def _aligned_features(train, test):
    def pooled(key, dtype=None):
        return np.concatenate((np.asarray(train[key], dtype=dtype),
                               np.asarray(test[key], dtype=dtype)))

    labs = pooled('lab_id')
    strata = pooled('stratum')
    controls = pooled('ref_coupons', float)
    aligned = pooled('coupon_features_raw', float)
    for lab in np.unique(labs):
        rows = labs == lab
        panel = controls[rows]
        observed = np.isfinite(panel)
        valid = observed.sum(axis=0) >= 2
        panel = np.where(observed[:, valid], panel[:, valid], np.nan)
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0, ddof=0))
        aligned[rows] = (aligned[rows] - offset) / gain

    _, cell, counts = np.unique(np.column_stack((labs, strata)), axis=0,
                                return_inverse=True, return_counts=True)
    weights = labs.size / (counts.size * counts[cell.ravel()])
    finite = np.isfinite(aligned)
    support = weights @ finite
    means = np.divide(weights @ np.where(finite, aligned, 0.0), support,
                      out=np.zeros(aligned.shape[1]), where=support > 0)
    return np.where(finite, aligned, means)


def _design(coupons, features, times, taus, dt):
    stress = np.asarray(coupons['stress_hold_0'], dtype=float)
    decay = np.exp(-dt / taus)
    step = -np.expm1(-dt / taus)
    memory = np.zeros((stress.size, taus.size))
    for load in np.asarray(coupons['past_stress_history'], dtype=float).T:
        memory = memory * decay + load[:, None] * step
    relaxation = -np.expm1(-times[:, None] / taus)
    drift = np.column_stack((np.ones(stress.size), features))
    return np.concatenate(((stress[:, None] - memory)[:, None, :] * relaxation,
                           drift[:, None, :] * times[None, :, None]), axis=2)


def _cauchy_fit(design, target):
    def scale_from(residual):
        mad = np.median(np.abs(residual - np.median(residual)))
        return max(1e-3, 2.0 * 1.4826 * mad)

    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ beta
    scale = scale_from(residual)
    # Re-estimate scale once after the pilot so large spikes cannot set the
    # final loss width through their influence on the initial least squares.
    for iterations in (8, 25):
        for _ in range(iterations):
            root_weight = scale / np.hypot(scale, residual)
            updated = np.linalg.lstsq(design * root_weight[:, None],
                                      target * root_weight, rcond=None)[0]
            change = np.linalg.norm(updated - beta)
            beta = updated
            residual = target - design @ beta
            if change <= 1e-9 * (1.0 + np.linalg.norm(beta)):
                break
        scale = scale_from(residual)
    return beta


def _first_crossing(times, initial, forecast, critical):
    knots = np.concatenate(([0.0], times))
    values = np.column_stack((initial, forecast))
    reached = values >= critical[:, None]
    first = np.argmax(reached, axis=1)
    result = np.full(initial.size, times[-1])
    result[reached[:, 0]] = 0.0
    rows = np.flatnonzero(first > 0)
    right = first[rows]
    left = right - 1
    before = values[rows, left]
    after = values[rows, right]
    result[rows] = (knots[left] + (critical[rows] - before) / (after - before)
                    * (knots[right] - knots[left]))
    return result


def solve_viscoelastic_hereditary_forecast(dataset: dict) -> dict:
    """Fit the seven hereditary coefficients and forecast test coupon hazards."""
    train, test = dataset['train'], dataset['test']
    times = np.asarray(dataset['eval_times'], dtype=float)
    taus = np.asarray(dataset['memory_taus'], dtype=float)
    dt = float(dataset['history_dt'])
    initial_train = np.asarray(train['strain_0'], dtype=float)
    initial_test = np.asarray(test['strain_0'], dtype=float)
    aligned = _aligned_features(train, test)
    test_features = aligned[initial_train.size:]
    train_design = _design(train, aligned[:initial_train.size], times, taus, dt)
    target = np.asarray(train['strain_curves_obs'], dtype=float) - initial_train[:, None]
    beta = _cauchy_fit(train_design.reshape(-1, train_design.shape[-1]), target.ravel())
    forecast = initial_test[:, None] + _design(test, test_features, times, taus, dt) @ beta
    critical = np.asarray(test['critical_strain'], dtype=float)
    return {'aligned_test_features': test_features,
            'hereditary_weights': beta,
            'predicted_strain': forecast,
            'crossing_times': _first_crossing(times, initial_test, forecast, critical)}
PY
```
- **Container Exit Code**: `0` (`414 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoelastic_hereditary_forecast.py']`
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2e95f72fae00ac96 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2e95f72f (directory).
```

### Turn 5 — `execute_in_remote_docker` (`64584 ms` LLM)
- **Rationale**: Independent alignment, convolution, and scalar crossing oracles isolate failures in calibration, memory, and hazard timing. Held-out synthetic coupons test recovery under spikes, with errors reported separately for twins, lab-stratum cells, and hazard outcomes.
- **Alternatives Weighed**: `Verify only import and output shapes` (rejected: That would miss reversed history weighting, biased imputation, and incorrect first-crossing interpolation.)

```bash
set -eu
export OPENBLAS_NUM_THREADS=1
python3 -I - <<'PY'
import importlib.util
import numpy as np

path = '/holdout_task/viscoelastic_hereditary_forecast.py'
spec = importlib.util.spec_from_file_location('forecast', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
solve = module.solve_viscoelastic_hereditary_forecast

# Independent oracle: geometric convolution instead of the solver recurrence,
# scalar hazard traversal instead of vectorized segment selection.
def crossings(times, start, curves, critical):
    result = []
    for y0, curve, c in zip(start, curves, critical):
        if y0 >= c:
            result.append(0.)
            continue
        ta, ya = 0., y0
        for tb, yb in zip(times, curve):
            if yb >= c:
                result.append(ta + (tb-ta)*(c-ya)/(yb-ya))
                break
            ta, ya = tb, yb
        else:
            result.append(times[-1])
    return np.array(result)

np.testing.assert_allclose(module._first_crossing(
    np.array([1., 2., 4.]), np.array([1., 0., 0., 0., 0.]),
    np.array([[0., 0., 0.], [2., 0., 3.], [0., 2., 3.], [0., 0., 1.], [0., 1., 2.]]),
    np.array([1., 1., 2., 2., 1.5])), [0., .5, 2., 4., 3.])

worst = dict(alignment=0., weights=0., curve=0., twins=0., crossing=0.)
worst_slices = dict(lab_curve=0., lab_crossing=0., event_crossing=0.)
for seed in range(12):
    rng = np.random.default_rng(seed)
    ntr, nte = 320, 112
    n = ntr + nte
    tau = np.array([1., 4.5])
    dt = .25
    times = np.r_[.08, np.linspace(.25, 7.5, 35), 8.]
    beta = np.array([.48, .63, .065, .026, -.021, .017, -.013])
    lab = rng.choice(4, n, p=[.10, .17, .29, .44])
    stratum = rng.binomial(1, np.array([.2, .8, .35, .65])[lab])
    lab[ntr::8] = np.arange(nte//8) % 4
    stratum[ntr::8] = np.arange(nte//8) // 4 % 2
    off = np.array([-.9, .3, 1.2, -.48])
    gain = np.array([.56, 1.6, .84, 1.27])
    refs = off[lab,None] + gain[lab,None] * rng.normal(size=(n,4))
    refs[rng.random(refs.shape) < .08] = np.nan
    raw = off[lab,None] + gain[lab,None] * (rng.normal(size=(n,4)) + .6*stratum[:,None])
    raw[rng.random(raw.shape) < .05] = np.nan
    stress = rng.uniform(.7, 2.1, size=n)
    initial = rng.uniform(.05, .3, size=n)
    rate = rng.normal(size=n)
    history = rng.normal(.05,.3,(n,40)) + rng.uniform(-.4,1.8,(n,1))
    a = np.arange(ntr,n,8)
    b = a+1
    for arr in (lab, stratum, raw, stress, initial, rate):
        arr[b] = arr[a]
    history[a] = -.5
    history[b] = 1.9
    aligned = raw.copy()
    for l in range(4):
        controls = [refs[lab==l,j] for j in range(4)]
        controls = [c[np.isfinite(c)] for c in controls]
        offsets = [np.mean(c) for c in controls if len(c)>=2]
        spreads = [np.std(c) for c in controls if len(c)>=2]
        aligned[lab==l] = (raw[lab==l] - np.median(offsets))/np.median(spreads)
    cells = [(lab==l)&(stratum==s) for l in range(4) for s in range(2)]
    for j in range(4):
        numer, denom = 0., 0.
        for cell in cells:
            finite = cell & np.isfinite(aligned[:,j])
            numer += aligned[finite,j].sum()/cell.sum()
            denom += finite.sum()/cell.sum()
        aligned[~np.isfinite(aligned[:,j]),j] = numer/denom
    kernel = np.exp(-dt*np.arange(39,-1,-1)[:,None]/tau)*(1-np.exp(-dt/tau))
    memory = history @ kernel
    curves = initial[:,None] + ((stress[:,None]-memory)*beta[:2]) @ (1-np.exp(-times[None,:]/tau[:,None]))
    curves += (beta[2] + aligned @ beta[3:])[:,None]*times
    observed = curves[:ntr] + rng.normal(0,.008,(ntr,len(times)))
    spike = rng.random(observed.shape) < .09
    observed[spike] += np.clip(.95*rng.standard_cauchy(spike.sum()), -.95, .95)
    crit = initial[ntr:] + .65*(curves[ntr:].max(axis=1)-initial[ntr:])
    crit[2::8] = initial[ntr:][2::8]
    crit[3::8] = np.maximum(initial[ntr:][3::8], curves[ntr:][3::8].max(axis=1)) + .1
    coupons = dict(lab_id=lab, stratum=stratum, ref_coupons=refs, coupon_features_raw=raw,
                   past_stress_history=history, stress_hold_0=stress, strain_0=initial, instant_rate_0=rate)
    train = {k:v[:ntr] for k,v in coupons.items()}
    train['strain_curves_obs'] = observed
    test = {k:v[ntr:] for k,v in coupons.items()}
    test['critical_strain'] = crit
    result = solve(dict(train=train,test=test,eval_times=times,memory_taus=tau,history_dt=dt))
    shape = dict(aligned_test_features=(nte,4), hereditary_weights=(7,), predicted_strain=(nte,len(times)), crossing_times=(nte,))
    assert result.keys() == shape.keys()
    assert all(result[k].shape == s and np.isfinite(result[k]).all() for k,s in shape.items())
    np.testing.assert_allclose(result['aligned_test_features'], aligned[ntr:], atol=1e-12, rtol=0)
    forecast = result['predicted_strain']
    np.testing.assert_allclose(result['crossing_times'], crossings(times, initial[ntr:], forecast, crit), atol=1e-12, rtol=0)
    assert np.min(forecast[a-ntr,-1] - forecast[b-ntr,-1]) > 1.
    err = np.abs(forecast - curves[ntr:])
    true_crossings = crossings(times, initial[ntr:], curves[ntr:], crit)
    cross_err = np.abs(result['crossing_times'] - true_crossings)
    metrics = dict(alignment=np.mean(np.abs(result['aligned_test_features']-aligned[ntr:])),
                   weights=np.linalg.norm(result['hereditary_weights']-beta)/np.linalg.norm(beta),
                   curve=err.mean(), twins=err[np.r_[a,b]-ntr].mean(), crossing=cross_err.mean())
    for k, v in metrics.items():
        worst[k] = max(worst[k],v)
    for l in range(4):
        for s in range(2):
            group = (lab[ntr:]==l) & (stratum[ntr:]==s)
            if group.any():
                worst_slices['lab_curve'] = max(worst_slices['lab_curve'],err[group].mean())
                worst_slices['lab_crossing'] = max(worst_slices['lab_crossing'],cross_err[group].mean())
    for group in (true_crossings == 0, (true_crossings > 0) & (true_crossings < times[-1]), true_crossings == times[-1]):
        if group.any():
            worst_slices['event_crossing'] = max(worst_slices['event_crossing'],cross_err[group].mean())
print('Worst synthetic seed of 12:', worst)
print('Worst synthetic lab-stratum and hazard slices:', worst_slices)
for k,limit in dict(alignment=.15, weights=.10, curve=.08, twins=.08, crossing=.18).items():
    assert worst[k] <= limit, (k,worst[k],limit)
assert worst_slices['lab_curve'] <= .08 and max(worst_slices['lab_crossing'],worst_slices['event_crossing']) <= .18
print('Isolated-process artifact verification passed:', path)
PY
```
- **Container Exit Code**: `0` (`1038 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Worst synthetic seed of 12: {'alignment': 1.466238232638544e-16, 'weights': 0.0019060965998222669, 'curve': 0.00021758119492591238, 'twins': 0.0002787498577069501, 'crossing': 0.001250692334876866}
Worst synthetic lab-stratum and hazard slices: {'lab_curve': 0.00030015163989945327, 'lab_crossing': 0.003813891779887113, 'event_crossing': 0.0020301092971914346}
Isolated-process artifact verification passed: /holdout_task/viscoelastic_hereditary_forecast.py

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2e95f72fae00ac96 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2e95f72f (directory).
```

### Turn 6 — `submit_completed_task` (`14411 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification: independent oracles confirmed balanced imputation and first-crossing interpolation, and held-out synthetic trials confirmed coefficient recovery and twin forecasts under transducer spikes.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/viscoelastic_hereditary_forecast.py': 'FILE:4770'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/viscoelastic_hereditary_forecast.py`. All contract thresholds passed across 12 synthetic trials, including identical-present twins. The public dataset was not available for validation.`

### Turn 7 — `execute_in_remote_docker` (`40105 ms` LLM)
- **Rationale**: Check whether the public measurements are now mounted. Independently test the exact two-mode response to moving a pulse in time and whether directed, bounded contamination can bias the fitted coefficients past tolerance. A failure would isolate either history indexing or robust-fit initialization.
- **Alternatives Weighed**: `Add rational or relay terms, or optimize within-batch classification accuracy` (rejected: The specified response is a continuous trajectory with a fixed seven-term basis and a derived crossing time. There is no relay state or classification target; lab offsets and gains are already estimated from reference coupons.)

```bash
set -eu
cd /holdout_task
if [ -f public_eq03_datasets.pkl ]; then
  ls -lh public_eq03_datasets.pkl
  if [ -f validate_public_schema.py ]; then python3 -I validate_public_schema.py; fi
else
  echo 'Public measurements are not mounted at the documented path.'
fi
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import numpy as np
p = '/holdout_task/viscoelastic_hereditary_forecast.py'
spec = importlib.util.spec_from_file_location('forecast', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
taus = np.array([1., 4.5])
times = np.array([.125, .5, 2., 4., 8.])
dt = .25
# Equal stress totals and identical presents. Moving a pulse to the latest
# interval must change each transient by its own exact recency factor.
hist = np.zeros((2, 40))
hist[0, 0] = hist[1, -1] = 2.
x = np.tile([.2, -.5, 1., .8], (2, 1))
g = m._design({'stress_hold_0': np.ones(2), 'past_stress_history': hist}, x, times, taus, dt)
expected = 2 * (-np.expm1(-dt/taus)) * (-np.expm1(-39*dt/taus)) * (-np.expm1(-times[:, None]/taus))
np.testing.assert_allclose(g[0, :, :2] - g[1, :, :2], expected, rtol=1e-13, atol=1e-15)
np.testing.assert_array_equal(g[0, :, 2:], g[1, :, 2:])
# Concentrate the allowed bounded spikes at the most influential observations
# for one memory coefficient, with no cancellation from opposite signs.
rng = np.random.default_rng(42)
n = 192
x = rng.normal(size=(n, 4))
hist = rng.normal(0, .3, (n, 40)) + rng.uniform(-1., 2., (n, 1))
design = m._design({'stress_hold_0': rng.uniform(.5, 2., n), 'past_stress_history': hist}, x, times, taus, dt).reshape(-1, 7)
truth = np.array([.48, .63, .065, .026, -.021, .017, -.013])
clean = design @ truth
leverage = np.linalg.pinv(design)[1]
spikes = np.argsort(np.abs(leverage))[-int(.09 * clean.size):]
obs = clean + rng.normal(0., .008, clean.size)
obs[spikes] += .95 * np.sign(leverage[spikes])
ols = np.linalg.lstsq(design, obs, rcond=None)[0]
robust = m._cauchy_fit(design, obs)
relative = lambda w: np.linalg.norm(w - truth) / np.linalg.norm(truth)
err = relative(robust)
print('Exact paired-history basis check passed; one-sided high-influence spike weight relative L2: robust=%.6f, OLS=%.6f' % (err, relative(ols)))
assert err <= .10 and err < relative(ols)
PY
```
- **Container Exit Code**: `0` (`651 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public measurements are not mounted at the documented path.
Exact paired-history basis check passed; one-sided high-influence spike weight relative L2: robust=0.011943, OLS=1.943940

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2e95f72fae00ac96 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2e95f72f (directory).
```

### Turn 8 — `submit_completed_task` (`16080 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification. Independent checks confirmed pooled calibration and imputation, the exact two-mode response to pulse timing, and first-crossing interpolation. Synthetic worst-slice errors stayed below the contract tolerances.
- **Alternatives Weighed**: `Replace Cauchy IRLS with ordinary least squares` (rejected: Under targeted spikes, relative coefficient error was 1.943940 for ordinary least squares versus 0.011943 for Cauchy IRLS.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/viscoelastic_hereditary_forecast.py': 'FILE:4770'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/viscoelastic_hereditary_forecast.py`. All contract tolerances passed across 12 synthetic trials, including identical-present twins with different loading histories. The public dataset was not mounted, so validation on public measurements remains pending.`

### Turn 9 — `execute_in_remote_docker` (`38694 ms` LLM)
- **Rationale**: If the public fixture is now mounted, run its validator. Independently test a nuisance property not covered by the prior audit: changing lab offsets and units consistently in controls and raw features must leave all four outputs unchanged despite cell imbalance and missingness. A failure would warrant tracing calibration or unintended predictor use; a pass requires no artifact edit.
- **Alternatives Weighed**: `Add rational, relay, or classification terms` (rejected: The specified seven-term Prony law and continuous hazard calculation contain no relay state or classification target. The paired-pulse audit already verified the two exact stress-minus-memory terms.)

```bash
set -eu
cd /holdout_task
if [ -f public_eq03_datasets.pkl ]; then
  ls -lh public_eq03_datasets.pkl
  if [ -f validate_public_schema.py ]; then python3 validate_public_schema.py; fi
else
  echo 'Real-measurement audit unavailable: public_eq03_datasets.pkl is not mounted.'
fi
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('forecast', '/holdout_task/viscoelastic_hereditary_forecast.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
rng = np.random.default_rng(713)
n, ntr = 240, 176
lab = np.repeat(np.arange(4), [16, 28, 68, 128])[rng.permutation(n)]
stratum = rng.binomial(1, [.2, .8, .35, .7][0] + np.array([0., .6, .15, .5])[lab])
refs = rng.normal(size=(n, 4))
raw = rng.normal(size=(n, 4)) + .7 * stratum[:, None]
refs[rng.random(refs.shape) < .08] = np.nan
raw[rng.random(raw.shape) < .05] = np.nan
history = rng.normal(size=(n, 40)) + rng.normal(size=(n, 1))
times = np.linspace(.25, 8., 32)
taus = np.array([1., 4.5])
dt = .25
coupons = dict(lab_id=lab, stratum=stratum, ref_coupons=refs, coupon_features_raw=raw, past_stress_history=history, stress_hold_0=rng.uniform(.7, 2., n), strain_0=rng.uniform(.1, .25, n), instant_rate_0=rng.normal(size=n))
train = {k: v[:ntr] for k, v in coupons.items()}
test = {k: v[ntr:] for k, v in coupons.items()}
# The test compares two descriptions of the same measurements, not model-generated truth.
# Arbitrary observations suffice to expose laboratory shortcuts in the fitted pipeline.
train['strain_curves_obs'] = train['strain_0'][:, None] + (.13 + .02 * stratum[:ntr, None]) * times + rng.normal(0., .01, (ntr, times.size))
test['critical_strain'] = rng.uniform(.25, 1.5, n-ntr)
d = dict(train=train, test=test, eval_times=times, memory_taus=taus, history_dt=dt)
base = m.solve_viscoelastic_hereditary_forecast(d)
shift = np.array([-3., .8, 4., -1.5])
scale = np.array([.25, 4., .8, 1.7])
changed = dict(d)
for split in ('train', 'test'):
    part = d[split].copy()
    l = part['lab_id']
    for key in ('ref_coupons', 'coupon_features_raw'):
        part[key] = part[key] * scale[l, None] + shift[l, None]
    part['instant_rate_0'] = -10. * part['instant_rate_0']
    changed[split] = part
other = m.solve_viscoelastic_hereditary_forecast(changed)
for name in base:
    delta = np.max(np.abs(base[name] - other[name]))
    print(f'Lab-unit invariance {name}: max absolute delta {delta:.3g}')
    np.testing.assert_allclose(other[name], base[name], atol=1e-8, rtol=0)
assert ((base['crossing_times'] > 0) & (base['crossing_times'] < times[-1])).any()
print('Isolated lab-unit and unused-instantaneous-rate audit passed.')
PY
```
- **Container Exit Code**: `0` (`699 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Real-measurement audit unavailable: public_eq03_datasets.pkl is not mounted.
Lab-unit invariance aligned_test_features: max absolute delta 1.61e-15
Lab-unit invariance hereditary_weights: max absolute delta 1.25e-15
Lab-unit invariance predicted_strain: max absolute delta 4.22e-15
Lab-unit invariance crossing_times: max absolute delta 1.69e-14
Isolated lab-unit and unused-instantaneous-rate audit passed.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2e95f72fae00ac96 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2e95f72f (directory).
```

### Turn 10 — `submit_completed_task` (`14793 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification. Independent checks confirmed the exact history-pulse response and first-crossing interpolation. Under targeted spikes, relative coefficient error was 0.011943 versus the 0.10 tolerance. Changing lab units shifted no output by more than 1.69e-14. The audit found no issue warranting a code change.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoelastic_hereditary_forecast.py`. Across 12 synthetic trials, all contract tolerances passed: worst twin trajectory MAE was 0.000279, and worst lab–stratum crossing-time MAE was 0.003814. The public dataset was not mounted, so validation on real measurements remains pending.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/viscoelastic_hereditary_forecast.py": "FILE:4770"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "all_valid": true,
    "max_aligned_feat_mae": 0.11361,
    "max_hereditary_w_rel_l2": 0.05884,
    "max_strain_mae": 0.04337,
    "max_twin_same_present_mae": 0.03994,
    "max_crossing_time_mae": 0.09606,
    "student_elapsed_sec": 0.0644,
    "instance_diagnostics": [
      {
        "seed": 99421,
        "feat_mae": 0.10318,
        "w_rel_l2": 0.05884,
        "strain_mae": 0.04337,
        "twin_mae": 0.03994,
        "cross_mae": 0.09606
      },
      {
        "seed": 99422,
        "feat_mae": 0.1073,
        "w_rel_l2": 0.02217,
        "strain_mae": 0.03436,
        "twin_mae": 0.03595,
        "cross_mae": 0.03364
      },
      {
        "seed": 99423,
        "feat_mae": 0.11361,
        "w_rel_l2": 0.00881,
        "strain_mae": 0.0431,
        "twin_mae": 0.03834,
        "cross_mae": 0.08649
      }
    ]
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_aligned_feat_mae\": 0.11361, \"max_hereditary_w_rel_l2\": 0.05884, \"max_strain_mae\": 0.04337, \"max_twin_same_present_mae\": 0.03994, \"max_crossing_time_mae\": 0.09606, \"student_elapsed_sec\": 0.0644, \"instance_diagnostics\": [{\"seed\": 99421, \"feat_mae\": 0.10318, \"w_rel_l2\": 0.05884, \"strain_mae\": 0.04337, \"twin_mae\": 0.03994, \"cross_mae\": 0.09606}, {\"seed\": 99422, \"feat_mae\": 0.1073, \"w_rel_l2\": 0.02217, \"strain_mae\": 0.03436, \"twin_mae\": 0.03595, \"cross_mae\": 0.03364}, {\"seed\": 99423, \"feat_mae\": 0.11361, \"w_rel_l2\": 0.00881, \"strain_mae\": 0.0431, \"twin_mae\": 0.03834, \"cross_mae\": 0.08649}]}\n__HEGEL_VERIFIER_AUTH_d8e3fed70fd043cac65643221bb6f18f__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_aligned_feat_mae\": 0.11361, \"max_hereditary_w_rel_l2\": 0.05884, \"max_strain_mae\": 0.04337, \"max_twin_same_present_mae\": 0.03994, \"max_crossing_time_mae\": 0.09606, \"student_elapsed_sec\": 0.0644, \"instance_diagnostics\": [{\"seed\": 99421, \"feat_mae\": 0.10318, \"w_rel_l2\": 0.05884, \"strain_mae\": 0.04337, \"twin_mae\": 0.03994, \"cross_mae\": 0.09606}, {\"seed\": 99422, \"feat_mae\": 0.1073, \"w_rel_l2\": 0.02217, \"strain_mae\": 0.03436, \"twin_mae\": 0.03595, \"cross_mae\": 0.03364}, {\"seed\": 99423, \"feat_mae\": 0.11361, \"w_rel_l2\": 0.00881, \"strain_mae\": 0.0431, \"twin_mae\": 0.03834, \"cross_mae\": 0.08649}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5951.94,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
