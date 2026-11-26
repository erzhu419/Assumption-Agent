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
| **Turns Used / Max Turns** | `8 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `170,419` (`163,446` / `24,704` / `6,973`) |
| **Total Episode Wall Time (s)** | `241.22s` |

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

### Turn 1 — `execute_in_remote_docker` (`14062 ms` LLM)
- **Rationale**: Inspect the public runner to identify the dataset layout and available validation targets before implementing; query the primitive library for compatible alignment, relaxation, and robust fitting routines.
- **Alternatives Weighed**: `Fit instantaneous observables alone` (rejected: Identical present-state inputs cannot distinguish the specified history twins.)

```bash
ls -lh /holdout_task; sed -n '1,220p' /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation Cauchy IRLS reference panel alignment'
```
- **Container Exit Code**: `0` (`357 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
le)
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
```
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task': No such file or directory
sed: can't read /holdout_task/validate_public_schema.py: No such file or directory
```

### Turn 2 — `execute_in_remote_docker` (`16487 ms` LLM)
- **Rationale**: Search the expected staging directories without copying data. Inspect the available primitives to determine whether their missing-value handling and robust fit match the requested contract.
- **Alternatives Weighed**: `Assume the public data are at the documented path` (rejected: The prior command found no /holdout_task mount; locate a staged copy before deciding whether real-data validation is available.)

```bash
pwd; ls -ld /pretrain_task /holdout_task /app /workspace; find /pretrain_task /holdout_task /app /workspace -maxdepth 3 -type f \( -iname '*eq03*' -o -name 'validate_public_schema.py' -o -iname '*viscoelastic*' \) -print 2>/dev/null; for fn in autogenous_prim_reference_panel_balancing.align_reference_panel autogenous_prim_reference_panel_balancing.balanced_cell_weights autogenous_prim_reference_panel_balancing.weighted_column_fill autogenous_prim_robust_trend_decomposition.cauchy_linear_fit; do PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe "$fn"; done
```
- **Container Exit Code**: `0` (`457 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
thin both segments. The scale and loss_clip are positive finite scalars. Interpretation as a level step assumes a shared local trend on both sides."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "cauchy_linear_fit": "APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)",
        "fit_anchored_relaxation": "APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)",
        "fit_local_level_step": "APPLICABLE_SUBROUTINE(real_1d_step_series)"
      },
      "functions": [
        {
          "function": "cauchy_linear_fit",
          "signature": "cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float]",
          "doc_summary": "Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank before and after weighting, and positive scale if supplied and positive floor, returning (coefficients (p,), floored residual MAD)."
        }
      ],
      "index_cards": [
        "autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit",
        "autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation",
        "autogenous_prim_robust_trend_decomposition@1f45421a | fit_local_level_step(predictor: np.ndarray, values: np.ndarray, splits: np.ndarray, *, scale: float, loss_clip: float) -> tuple[int, float] | Fit finite predictor and values (n,) and integer splits (k,) with k >= 1, at l | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_local_level_step"
      ]
    }
  ]
}
```
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task': No such file or directory
ls: cannot access '/app': No such file or directory
ls: cannot access '/workspace': No such file or directory
```

### Turn 3 — `execute_in_remote_docker` (`47028 ms` LLM)
- **Rationale**: Implement the specified recurrence, robust fit, and crossing rule. Inspect staging metadata without copying data to determine whether validation can use the public fixture or must use contract-based simulations.
- **Alternatives Weighed**: `Import the reference-panel and Cauchy-fit primitives directly` (rejected: The weighting primitive requires every lab–stratum combination, and the fit primitive requires full rank. The self-contained NumPy implementation uses observed cells and supports rank-deficient training folds.)

```bash
set -eu
mkdir -p /holdout_task
cat > /holdout_task/viscoelastic_hereditary_forecast.py <<'PY'
"""Coupon-calibrated two-mode hereditary creep and first-crossing forecasts."""

import numpy as np


def _align(train, test):
    def pool(key):
        return np.concatenate((train[key], test[key]), axis=0)

    labs, strata = pool('lab_id'), pool('stratum')
    raw = np.asarray(pool('coupon_features_raw'), dtype=float)
    reference = np.asarray(pool('ref_coupons'), dtype=float)
    aligned = np.empty_like(raw)
    for lab in np.unique(labs):
        rows = labs == lab
        controls = reference[rows]
        controls = controls[:, np.any(np.isfinite(controls), axis=0)]
        offset = np.median(np.nanmean(controls, axis=0))
        gain = np.median(np.nanstd(controls, axis=0, ddof=0))
        aligned[rows] = (raw[rows] - offset) / gain

    _, inverse, counts = np.unique(np.column_stack((labs, strata)), axis=0,
                                   return_inverse=True, return_counts=True)
    weights = (len(labs) / (len(counts) * counts[inverse]))[:, None]
    finite = np.isfinite(aligned)
    fill = np.sum(weights * np.where(finite, aligned, 0.0), axis=0)
    mass = np.sum(weights * finite, axis=0)
    np.divide(fill, mass, out=fill, where=mass > 0)
    return np.where(finite, aligned, fill)


def _design(split, x, dt, taus, times):
    history = np.asarray(split['past_stress_history'], dtype=float)
    decay = np.exp(-dt / taus)
    injection = -np.expm1(-dt / taus)
    state = np.zeros((len(x), len(taus)))
    for load in history.T:
        state = state * decay + load[:, None] * injection

    gaps = np.asarray(split['stress_hold_0'], dtype=float)[:, None] - state
    relaxation = -np.expm1(-times[:, None] / taus)
    rates = np.column_stack((np.ones(len(x)), x))
    return np.concatenate((gaps[:, None, :] * relaxation[None, :, :],
                           rates[:, None, :] * times[None, :, None]), axis=2)


def _cauchy_fit(design, observed):
    # Equilibrate columns so the fast and slow modes are not numerically
    # penalized by their smaller units relative to the linear creep terms.
    norms = np.linalg.norm(design, axis=0)
    norms[norms == 0] = 1.0
    a = design / norms
    beta = np.linalg.lstsq(a, observed, rcond=None)[0]
    floor = 1e-9 * max(1.0, np.median(np.abs(observed)))
    for _ in range(60):
        residual = observed - a @ beta
        scale = max(floor, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        root_weight = 1.0 / np.hypot(1.0, residual / (1.5 * scale))
        next_beta = np.linalg.lstsq(a * root_weight[:, None],
                                     observed * root_weight, rcond=None)[0]
        converged = np.linalg.norm(next_beta - beta) <= 1e-10 * (1 + np.linalg.norm(beta))
        beta = next_beta
        if converged:
            break
    return beta / norms


def _crossings(times, initial, forecast, critical):
    knots = np.concatenate(([0.0], times))
    values = np.column_stack((initial, forecast))
    reached = values >= critical[:, None]
    first = np.argmax(reached, axis=1)
    result = np.full(len(initial), times[-1])
    result[reached[:, 0]] = 0.0
    rows = np.flatnonzero(first > 0)
    end = first[rows]
    start = end - 1
    ya, yb = values[rows, start], values[rows, end]
    result[rows] = knots[start] + ((critical[rows] - ya) / (yb - ya)) * (knots[end] - knots[start])
    return result


def solve_viscoelastic_hereditary_forecast(dataset: dict) -> dict:
    """Fit on observed training curves and forecast every test coupon."""
    train, test = dataset['train'], dataset['test']
    times = np.asarray(dataset['eval_times'], dtype=float)
    taus = np.asarray(dataset['memory_taus'], dtype=float)
    dt = float(dataset['history_dt'])
    cut = len(train['lab_id'])
    aligned = _align(train, test)
    x_test = aligned[cut:]
    design = _design(train, aligned[:cut], dt, taus, times)
    residual_target = (np.asarray(train['strain_curves_obs'], dtype=float)
                       - np.asarray(train['strain_0'], dtype=float)[:, None])
    coeff = _cauchy_fit(design.reshape(-1, design.shape[-1]), residual_target.ravel())
    initial = np.asarray(test['strain_0'], dtype=float)
    forecast = initial[:, None] + _design(test, x_test, dt, taus, times) @ coeff
    crossing = _crossings(times, initial, forecast, np.asarray(test['critical_strain'], dtype=float))
    return {'aligned_test_features': x_test, 'hereditary_weights': coeff,
            'predicted_strain': forecast, 'crossing_times': crossing}
PY
ls -la /pretrain_task /holdout_task; ls -1 /; find /pretrain_task -maxdepth 3 -type f -print | head -40
```
- **Container Exit Code**: `0` (`357 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoelastic_hereditary_forecast.py']`
- **Stdout (tail)**:
```text
/holdout_task:
total 16
drwxr-xr-x 2 root root 4096 Nov 26 04:51 .
drwxr-xr-x 1 root root 4096 Nov 26 04:51 ..
-rw-r--r-- 1 root root 4456 Nov 26 04:51 viscoelastic_hereditary_forecast.py

/pretrain_task:
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 26 04:51 ..
bin
boot
dev
etc
holdout_task
home
lib
lib64
media
mnt
opt
pretrain_task
proc
root
run
sbin
srv
sys
tmp
usr
var

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=329119edfedd2631 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_329119ed (directory).
```

### Turn 4 — `execute_in_remote_docker` (`58262 ms` LLM)
- **Rationale**: Verify the saved artifact against an independent convolution and scalar crossing oracle. If a threshold fails, use the per-output and worst-slice metrics to locate the relevant stage; use the public runner if it is mounted.
- **Alternatives Weighed**: `Check output shapes alone` (rejected: Shapes would not detect reversed history, incorrect cell balancing, or a later crossing selected instead of first passage.)

```bash
set -eu
cat > /tmp/verify_viscoelastic.py <<'PY'
import importlib.util
import numpy as np

path = '/holdout_task/viscoelastic_hereditary_forecast.py'
spec = importlib.util.spec_from_file_location('forecast', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# Independent scalar check for initial contact, nonmonotone first passage,
# exact contact at a knot, and the no-contact convention.
np.testing.assert_allclose(m._crossings(np.array([1., 3., 8.]), np.array([1., 0., 0., 0., 0.]), np.array([[0, 0, 0], [2, -2, 4], [0, 1, 0], [-1, 0, .9], [-2, -1, 2]]), np.ones(5)), [0, .5, 3, 8, 3 + 10/3])

T = np.array([.12, .35, .7, 1.15, 1.8, 2.6, 3.8, 5.1, 6.5, 8.])
taus = np.array([1., 4.5])
dt = .25
truth = np.array([.53, .39, .029, .021, -.016, .013, .012])

def simulate(seed, severe_imbalance):
    rng = np.random.default_rng(seed)
    counts = [115, 27, 36, 13, 24, 8, 18, 5] if severe_imbalance else [35]*8
    train_cells = np.repeat(np.arange(8), counts)
    rng.shuffle(train_cells)
    ntr = len(train_cells)
    cells = np.r_[train_cells, np.repeat(np.arange(8), 14)]
    labs, strata = cells//2, cells%2
    n = len(cells)
    gains, offsets = np.array([.52, 1.8, .9, 2.2])[labs], np.array([-.9, .8, .2, -1.4])[labs]
    refs = offsets[:, None] + gains[:, None]*rng.normal(size=(n, 4))
    raw = offsets[:, None] + gains[:, None]*(rng.normal(size=(n, 4)) + .8*strata[:, None])
    refs[rng.random(refs.shape) < .08] = np.nan
    raw[rng.random(raw.shape) < .05] = np.nan
    if severe_imbalance:
        refs[labs == 3, 0] = np.nan
    history = rng.normal(.45, .65, size=(n, 40)) + rng.normal(0, .45, (n, 1))
    hold = rng.uniform(.3, 1.3, n)
    initial = rng.uniform(-.1, .25, n)
    rate = rng.uniform(0, 1, n)
    twin_left = ntr + np.arange(0, n-ntr, 2)
    twin_right = twin_left + 1
    for shared in (raw, hold, initial, rate):
        shared[twin_right] = shared[twin_left]
    history[twin_left, -15:] -= 1.3
    history[twin_right, -15:] += 1.3

    align = raw.copy()
    for lab in range(4):
        rows = labs == lab
        controls = [c[np.isfinite(c)] for c in refs[rows].T]
        means = [c.mean() for c in controls if len(c)]
        stds = [np.sqrt(np.mean((c-c.mean())**2)) for c in controls if len(c)]
        align[rows] = (raw[rows] - np.median(means))/np.median(stds)
    weights = np.array([1/np.count_nonzero(cells == c) for c in cells])
    for column in align.T:
        available = np.isfinite(column)
        column[~available] = np.average(column[available], weights=weights[available])

    # Closed-form convolution integral, independent of the solver recurrence.
    boundaries = dt*np.arange(40, -1, -1)
    mode = []
    for tau in taus:
        integrated = np.exp(-boundaries[1:]/tau) - np.exp(-boundaries[:-1]/tau)
        mode.append((hold - history @ integrated)[:, None]*(1-np.exp(-T/tau)))
    clean = initial[:, None] + truth[0]*mode[0] + truth[1]*mode[1] + (truth[2] + align@truth[3:])[:, None]*T
    noisy = clean[:ntr] + rng.normal(0, .008 if severe_imbalance else .003, (ntr, len(T)))
    spikes = rng.random(noisy.shape) < .09
    noisy[spikes] += np.clip(.95*rng.standard_cauchy(spikes.sum()), -.95, .95)
    fields = dict(lab_id=labs, stratum=strata, coupon_features_raw=raw, ref_coupons=refs, past_stress_history=history, stress_hold_0=hold, strain_0=initial, instant_rate_0=rate)
    train = {k:v[:ntr] for k,v in fields.items()}
    train['strain_curves_obs'] = noisy
    test = {k:v[ntr:] for k,v in fields.items()}
    crit = rng.uniform(.12, .85, n-ntr) + test['strain_0']
    crit[::13] = test['strain_0'][::13]
    test['critical_strain'] = crit
    result = m.solve_viscoelastic_hereditary_forecast(dict(train=train, test=test, history_dt=dt, memory_taus=taus, eval_times=T))
    exact_times = []
    for start, values, level in zip(test['strain_0'], clean[ntr:], crit):
        t_a, y_a = 0., start
        crossing = T[-1]
        if start >= level:
            crossing = 0.
        else:
            for t_b, y_b in zip(T, values):
                if y_b >= level:
                    crossing = t_a + (t_b-t_a)*(level-y_a)/(y_b-y_a)
                    break
                t_a, y_a = t_b, y_b
        exact_times.append(crossing)
    assert set(result) == {'aligned_test_features', 'hereditary_weights', 'predicted_strain', 'crossing_times'}
    for key, shape in zip(('aligned_test_features', 'hereditary_weights', 'predicted_strain', 'crossing_times'), ((n-ntr, 4), (7,), (n-ntr, len(T)), (n-ntr,))):
        assert result[key].shape == shape and np.isfinite(result[key]).all(), key
    pred = result['predicted_strain']
    e = np.abs(pred - clean[ntr:])
    h = np.abs(result['crossing_times'] - exact_times)
    worst_cell = max(e[cells[ntr:] == j].mean() for j in range(8))
    worst_cross_cell = max(h[cells[ntr:] == j].mean() for j in range(8))
    twin_delta_error = np.abs((pred[::2] - pred[1::2]) - (clean[twin_left] - clean[twin_right])).mean()
    np.testing.assert_allclose(result['aligned_test_features'], align[ntr:], atol=1e-12)
    return [np.abs(result['aligned_test_features'] - align[ntr:]).mean(), np.linalg.norm(result['hereditary_weights'] - truth)/np.linalg.norm(truth), e.mean(), e.mean(axis=0).max(), worst_cell, twin_delta_error, h.mean(), worst_cross_cell]

worst = np.max([simulate(i, bool(i % 2)) for i in range(24)], axis=0)
for name, value in zip(('alignment_mae', 'theta_rel_l2', 'trajectory_twin_mae', 'worst_horizon_trajectory_mae', 'worst_cell_trajectory_mae', 'twin_contrast_mae', 'crossing_mae', 'worst_cell_crossing_mae'), worst):
    print(f'{name}={value:.9g}')
assert np.all(worst <= [.15, .10, .08, .08, .08, .08, .18, .18]), worst
print('Isolated import and stress checks passed (24 synthetic datasets).')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_viscoelastic.py
if test -f /holdout_task/validate_public_schema.py && test -f /holdout_task/public_eq03_datasets.pkl; then python3 /holdout_task/validate_public_schema.py; else printf '%s\n' 'Public fixture and schema runner not mounted.'; fi
```
- **Container Exit Code**: `0` (`770 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
alignment_mae=2.39345374e-16
theta_rel_l2=0.00222520388
trajectory_twin_mae=0.00071453693
worst_horizon_trajectory_mae=0.00118124915
worst_cell_trajectory_mae=0.000787570652
twin_contrast_mae=0.0013401287
crossing_mae=0.00144334513
worst_cell_crossing_mae=0.00404636412
Isolated import and stress checks passed (24 synthetic datasets).
Public fixture and schema runner not mounted.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=329119edfedd2631 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_329119ed (directory).
```

### Turn 5 — `submit_completed_task` (`12511 ms` LLM)
- **Rationale**: The saved artifact passed an isolated Python import and independent checks for alignment, coefficient recovery, twin forecasts, and first crossings. All measured errors were below their tolerances; the available checks provide no reason to change the candidate.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`13` turns remaining; artifact status={'/holdout_task/viscoelastic_hereditary_forecast.py': 'FILE:4456'}): Do NOT lock prematurely with `13` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/viscoelastic_hereditary_forecast.py`. All contract checks passed across 24 synthetic datasets, including same-present twins and nonmonotone threshold crossings. The public dataset and validator were not mounted, so public-data validation remains unverified.`

### Turn 6 — `execute_in_remote_docker` (`29535 ms` LLM)
- **Rationale**: Look for the public fixture without copying data. If unavailable, test the saved candidate on early- and late-heavy time grids under lab imbalance; a failed coefficient or forecast margin would flag weak separation of relaxation and linear creep. Lab–stratum crossing error is the relevant nuisance slice for this continuous target.
- **Alternatives Weighed**: `Add generic rational, drift, or relay terms` (rejected: The specified seven-column creep equation already contains both required hold-minus-history differences and all drift terms. There is no relay state, drift PDE, or classification label in this contract.)

```bash
set -eu
find /holdout_task /pretrain_task /mnt /opt -maxdepth 4 -type f \( -name 'public_eq03_datasets.pkl' -o -name 'validate_public_schema.py' \) -print 2>/dev/null
if [ -f /holdout_task/public_eq03_datasets.pkl ] && [ -f /holdout_task/validate_public_schema.py ]; then
  python3 -I /holdout_task/validate_public_schema.py
else
  printf '%s\n' 'Real-data audit unavailable: the documented public fixture and validator are not mounted.'
  OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
from pathlib import Path
import numpy as np
p = Path('/tmp/verify_viscoelastic.py')
scope = {}
exec(compile(p.read_text().split('\nworst = np.max(', 1)[0], str(p), 'exec'), scope)
# Probe admissible early-heavy and late-heavy sampling: the latter makes
# fast-mode saturation and slow-mode/drift separation harder to estimate.
grids = [np.array([.002, .015, .05, .13, .3, .65, 1.3, 2.6, 5.2, 8.]),
         np.array([2., 2.3, 2.6, 3., 3.6, 4.2, 5., 6., 7., 8.])]
limits = np.array([.15, .10, .08, .08, .08, .08, .18, .18])
for name, grid in zip(('early-heavy', 'late-heavy'), grids):
    scope['T'] = grid
    worst = np.max([scope['simulate'](i, True) for i in range(801, 813)], axis=0)
    print(name, 'worst alignment, coefficient, twin, horizon, lab-cell, twin-contrast, crossing, lab-cell crossing:', np.round(worst, 8))
    assert np.all(worst <= limits), (name, worst)
PY
fi
```
- **Container Exit Code**: `0` (`1017 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Real-data audit unavailable: the documented public fixture and validator are not mounted.
early-heavy worst alignment, coefficient, twin, horizon, lab-cell, twin-contrast, crossing, lab-cell crossing: [0.         0.00732459 0.00087569 0.00176556 0.00098879 0.00158008
 0.0043833  0.00871381]
late-heavy worst alignment, coefficient, twin, horizon, lab-cell, twin-contrast, crossing, lab-cell crossing: [0.         0.00616182 0.00100551 0.00120032 0.00112325 0.0019965
 0.00201005 0.00480909]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=329119edfedd2631 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_329119ed (directory).
```

### Turn 7 — `submit_completed_task` (`15333 ms` LLM)
- **Rationale**: The saved artifact passed isolated execution and independent checks for alignment, coefficient recovery, twin forecasts, and first crossings. Early- and late-heavy sampling under lab imbalance also passed every tolerance, so no model change is warranted.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/viscoelastic_hereditary_forecast.py': 'FILE:4456'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/viscoelastic_hereditary_forecast.py`. All contract checks passed across 48 synthetic datasets, including same-present twins. The worst lab–stratum crossing-time MAE was 0.00871 against a tolerance of 0.18. Public-data validation remains pending because the dataset and validator were not mounted.`

### Turn 8 — `submit_completed_task` (`15989 ms` LLM)
- **Rationale**: The verified artifact passed independent checks for all four outputs, including imbalanced lab–stratum cells, early- and late-heavy sampling, and nonmonotone first crossings. No measured bottleneck warrants a model change.
- **Alternatives Weighed**: `Repeat the audit or add unrelated model terms` (rejected: The unchanged artifact already passed the focused audit, real measurements were unavailable, and the specified continuous model has neither relay modes nor classification labels.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/viscoelastic_hereditary_forecast.py` and verified it in an isolated Python process across 48 synthetic datasets. The worst same-present twin strain MAE was 0.00101 (limit 0.08), and the worst lab–stratum crossing-time MAE was 0.00871 (limit 0.18). The public dataset and validator were not mounted, so real-data validation remains pending.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/viscoelastic_hereditary_forecast.py": "FILE:4456"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "all_valid": true,
    "max_aligned_feat_mae": 0.11361,
    "max_hereditary_w_rel_l2": 0.05279,
    "max_strain_mae": 0.04292,
    "max_twin_same_present_mae": 0.03911,
    "max_crossing_time_mae": 0.09528,
    "student_elapsed_sec": 0.0763,
    "instance_diagnostics": [
      {
        "seed": 99421,
        "feat_mae": 0.10318,
        "w_rel_l2": 0.05279,
        "strain_mae": 0.04269,
        "twin_mae": 0.03911,
        "cross_mae": 0.09528
      },
      {
        "seed": 99422,
        "feat_mae": 0.1073,
        "w_rel_l2": 0.02519,
        "strain_mae": 0.03554,
        "twin_mae": 0.03763,
        "cross_mae": 0.03467
      },
      {
        "seed": 99423,
        "feat_mae": 0.11361,
        "w_rel_l2": 0.00963,
        "strain_mae": 0.04292,
        "twin_mae": 0.03824,
        "cross_mae": 0.08735
      }
    ]
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_aligned_feat_mae\": 0.11361, \"max_hereditary_w_rel_l2\": 0.05279, \"max_strain_mae\": 0.04292, \"max_twin_same_present_mae\": 0.03911, \"max_crossing_time_mae\": 0.09528, \"student_elapsed_sec\": 0.0763, \"instance_diagnostics\": [{\"seed\": 99421, \"feat_mae\": 0.10318, \"w_rel_l2\": 0.05279, \"strain_mae\": 0.04269, \"twin_mae\": 0.03911, \"cross_mae\": 0.09528}, {\"seed\": 99422, \"feat_mae\": 0.1073, \"w_rel_l2\": 0.02519, \"strain_mae\": 0.03554, \"twin_mae\": 0.03763, \"cross_mae\": 0.03467}, {\"seed\": 99423, \"feat_mae\": 0.11361, \"w_rel_l2\": 0.00963, \"strain_mae\": 0.04292, \"twin_mae\": 0.03824, \"cross_mae\": 0.08735}]}\n__HEGEL_VERIFIER_AUTH_13de047e27b9cc3c07a3463bea0867c7__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_aligned_feat_mae\": 0.11361, \"max_hereditary_w_rel_l2\": 0.05279, \"max_strain_mae\": 0.04292, \"max_twin_same_present_mae\": 0.03911, \"max_crossing_time_mae\": 0.09528, \"student_elapsed_sec\": 0.0763, \"instance_diagnostics\": [{\"seed\": 99421, \"feat_mae\": 0.10318, \"w_rel_l2\": 0.05279, \"strain_mae\": 0.04269, \"twin_mae\": 0.03911, \"cross_mae\": 0.09528}, {\"seed\": 99422, \"feat_mae\": 0.1073, \"w_rel_l2\": 0.02519, \"strain_mae\": 0.03554, \"twin_mae\": 0.03763, \"cross_mae\": 0.03467}, {\"seed\": 99423, \"feat_mae\": 0.11361, \"w_rel_l2\": 0.00963, \"strain_mae\": 0.04292, \"twin_mae\": 0.03824, \"cross_mae\": 0.08735}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 6143.53,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
