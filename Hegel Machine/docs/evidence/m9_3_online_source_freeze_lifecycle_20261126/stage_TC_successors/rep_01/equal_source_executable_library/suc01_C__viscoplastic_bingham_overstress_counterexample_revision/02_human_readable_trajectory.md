# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `suc01_C__viscoplastic_bingham_overstress_counterexample_revision`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M93-Lifecycle` |
| **Domain** | `Localized Premise Counterexample: Bingham-Perzyna Viscoplastic Overstress & Plastic-Work Softening Above Yield` |
| **Wave / Wormhole Family** | `m93_stage_C_counterexample` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`PASS (1.0)`** |
| **Failure / Outcome Taxonomy** | `PASSED_ALL_CHECKS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `1018` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `339,483` (`327,567` / `78,976` / `11,916`) |
| **Total Episode Wall Time (s)** | `438.96s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_10_hereditary_spectrum_identifi`**: No empirical nogoods were recorded; the supplied solver nevertheless addresses two clear failure modes: pooled raw features confound lab calibration with material drift, and ordinary L2 gives extreme strain glitches unbounded influence. Reference-coupon alignment and equal total weight per observed lab–stratum cell precede a weighted Cauchy fit, initialized by annealed IRLS over ordered relaxation-time pairs. The exact variable-step exponential operator uses the same hereditary state and coefficients for the creep curve and its analytic initial derivative, with all search times expressed relative to the median history step and all residuals expressed in observation-noise units. The refactor flags local identifiability only when the optimizer converges, the linear coefficient block has full numerical rank, and the smallest profiled log-time sensitivity is at least 0.01 by default; this threshold is a configurable diagnostic, not an empirically established regime boundary.
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`17489 ms` LLM)
- **Rationale**: Inspect the schema and array shapes to resolve the alignment and time-axis conventions before fitting. Look up reusable primitives without dumping dataset values or committing to an unverified implementation.
- **Alternatives Weighed**: `Implement from the equations alone` (rejected: The input layout, coupon references, and calibration fields determine how to construct and validate the design matrix.)

```bash
cd /holdout_task && ls -la && cat validate_public_schema.py && python3 - <<'PY'
import pickle
with open('public_suc01_C_datasets.pkl','rb') as f:
    data=pickle.load(f)
def show(x, indent='', depth=0):
    if isinstance(x, dict) and depth < 5:
        for k,v in list(x.items())[:30]:
            print(indent, repr(k), type(v).__name__, getattr(v,'shape',''), getattr(v,'dtype',''), repr(v)[:160] if isinstance(v,(str,int,float)) else '')
            show(v,indent+'  ',depth+1)
    elif isinstance(x,(list,tuple)) and depth < 5:
        print(indent,'length',len(x))
        if x: show(x[0],indent+'  ',depth+1)
show(data)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation variable-step Prony Cauchy robust IRLS reference-panel alignment imputation'
```
- **Container Exit Code**: `0` (`702 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights | STATUS: APPLICABLE_SUBROUTINE(real_residual_array_positive_scale)
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_hereditary_spectrum_identification@d7d5a0ae | align_reference_panel(raw_features: np.ndarray, reference_values: np.ndarray, groups: np.ndarray, strata: np.ndarray) -> tuple[np.ndarray, np.ndarray] | Align a common panel under groupwise scalar affine shifts | REQ: {'field': 'minimum_history_step', 'op': '>', 'value': 0}; {'field': 'minimum_evaluation_ | --describe autogenous_hereditary_spectrum_identification.align_reference_panel
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
```

### Turn 2 — `execute_in_remote_docker` (`27818 ms` LLM)
- **Rationale**: Inspect the available hereditary primitive for the inherited recurrence and feature conventions. Small panel and missingness summaries will distinguish fixed lab controls from row-varying references and determine how to fit the alignment.
- **Alternatives Weighed**: `Normalize coupon features independently by lab` (rejected: That could remove real stratum differences along with lab calibration shifts; the reference coupons should identify the shifts.)

```bash
export PYTHONPATH=/tmp
for fn in autogenous_hereditary_spectrum_identification.align_reference_panel autogenous_prim_reference_panel_balancing.align_reference_panel autogenous_prim_reference_panel_balancing.weighted_column_fill; do python3 -m hegel_wormhole_skills --describe "$fn"; done
ls -la /tmp/hegel_wormhole_skills
find /pretrain_task /tmp/hegel_wormhole_skills -maxdepth 3 -type f | head -100
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_suc01_C_datasets.pkl','rb'))[0]; d=p['dataset']; print('dt/time',d['past_dt'],d['eval_times']); print('GT parameters only',*(p['gt'][k] for k in ('taus','theta','vp_theta')))
for split in ('train','test'):
 s=d[split]; lab=s['lab_id']; print(split, np.unique(lab,return_counts=True),np.unique(s['stratum'],return_counts=True))
 for k,a in s.items():
  if a.ndim and k not in ('lab_id','stratum'):
   print(k,'nonfinite',np.sum(~np.isfinite(a)), 'quantile',np.nanquantile(a,[0,.1,.5,.9,1]).round(4))
 for l in np.unique(lab):
  r=s['ref_coupons'][lab==l]; print('ref',l,np.nanmedian(np.nanstd(r,axis=0)), np.nanmedian(r,axis=0).round(3))
PY
```
- **Container Exit Code**: `0` (`631 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.08  0.15 -0.12  0.11 -0.09] [0.24 0.16]
train (array([0, 1, 2, 3]), array([36, 38, 38, 36])) (array([0, 1, 2]), array([54, 47, 47]))
coupon_features_raw nonfinite 15 quantile [-5.1604 -3.7771 -1.1446  4.5485  6.8496]
ref_coupons nonfinite 112 quantile [-5.7836 -3.8961 -1.1247  4.6091  8.2399]
past_stress nonfinite 0 quantile [-3.1004 -1.426  -0.0181  1.426   3.9725]
strain_0 nonfinite 0 quantile [-1.0208 -0.8302 -0.0398  0.4883  1.0033]
stress_hold_0 nonfinite 0 quantile [-2.7774 -2.4954 -0.8269  2.4273  2.7726]
instant_rate_obs nonfinite 0 quantile [-8.009  -6.2983 -0.2387  5.8723  9.4914]
strain_curves_obs nonfinite 0 quantile [-10.9542  -4.0417  -0.166    4.1274  12.7602]
ref 0 1.306404245680194 [2.811 2.903 2.778 3.796 3.215 3.023 4.021 3.321 3.642 2.793 3.419 2.69
 3.414 2.763 2.915 3.058 3.401 3.022 3.099 3.02 ]
ref 1 0.7559130238665906 [-3.301 -3.388 -3.765 -3.303 -3.091 -3.248 -3.336 -3.364 -3.381 -3.387
 -3.159 -3.332 -3.367 -3.611 -3.54  -3.561 -3.358 -3.454 -3.329 -3.482]
ref 2 1.3864053602127493 [3.452 3.689 3.724 4.047 3.491 3.304 3.891 3.426 4.152 3.83  3.529 3.35
 3.708 3.222 3.864 3.496 3.106 3.537 3.748 3.646]
ref 3 0.8430457139895005 [-2.857 -2.572 -2.983 -3.024 -3.162 -2.838 -3.049 -3.105 -2.778 -3.076
 -2.664 -3.282 -2.909 -2.806 -2.795 -2.745 -2.996 -2.781 -2.991 -2.992]
test (array([0, 1, 2, 3]), array([22, 23, 23, 22])) (array([0, 1, 2]), array([33, 29, 28]))
coupon_features_raw nonfinite 9 quantile [-5.3876 -3.7871  0.2032  4.6007  6.5992]
ref_coupons nonfinite 72 quantile [-5.4371 -3.9251 -0.1183  4.554   7.3904]
past_stress nonfinite 0 quantile [-3.2552 -1.426   0.0068  1.426   3.5248]
strain_0 nonfinite 0 quantile [-0.9997 -0.3963  0.1676  0.8521  0.9854]
stress_hold_0 nonfinite 0 quantile [-2.7918 -2.0594  1.3175  2.4388  2.7748]
instant_rate_obs nonfinite 0 quantile [-7.5973 -2.6291  1.2208  5.4469  9.7118]
ref 0 1.317048567747967 [3.541 3.152 2.962 2.84  3.13  3.2   3.059 3.611 2.961 3.087 3.503 3.323
 3.793 2.711 3.404 2.804 2.484 3.038 3.162 3.13 ]
ref 1 0.8026467125074335 [-3.587 -3.409 -3.51  -3.225 -3.324 -3.749 -3.709 -3.545 -3.72  -3.332
 -3.39  -3.09  -3.159 -3.545 -3.32  -3.33  -3.22  -3.541 -3.125 -3.299]
ref 2 1.2318919086743065 [3.717 3.856 3.492 3.672 4.227 3.745 3.413 2.937 4.213 4.136 3.515 3.563
 3.439 3.653 4.141 3.689 3.489 3.841 3.249 3.511]
ref 3 0.8117715887075526 [-3.077 -3.134 -2.453 -2.601 -3.32  -2.606 -2.62  -2.888 -2.885 -2.737
 -2.614 -2.824 -2.599 -2.795 -2.967 -3.109 -2.872 -2.508 -3.402 -3.535]
```

### Turn 3 — `execute_in_remote_docker` (`31565 ms` LLM)
- **Rationale**: Read the hereditary implementation to catch a recurrence or design mismatch before extending it; check SciPy availability to choose the bounded fitting implementation.
- **Alternatives Weighed**: `Infer inherited design columns from array names` (rejected: The mounted hereditary primitive can resolve the baseline, stress factors, and rate derivative directly.)

```bash
sed -n '1,270p' /tmp/hegel_wormhole_skills/autogenous_hereditary_spectrum_identification.py; ls -la /pretrain_task; python3 -c 'import scipy; print("scipy", scipy.__version__)'
```
- **Container Exit Code**: `0` (`525 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
be finite and scales and weights positive')
    w = w / w.mean()
    weight = np.r_[np.repeat(w, k), w]
    root = np.sqrt(weight)
    target = target / scale

    def decode(p):
        fast = low + span * p[0]
        return np.array([fast, fast + span * (1.0 - p[0]) * p[1]])

    def matrix(tau):
        (curve, rate) = hereditary_design(past_stress, dt / unit, hold_stress, time / unit, features, tau)
        return np.vstack((curve.reshape(n * k, -1), rate)) / scale[:, None]

    def weighted_fit(a, v):
        r = np.sqrt(v)
        return np.linalg.lstsq(a * r[:, None], target * r, rcond=None)[0]

    def residual(p):
        r = matrix(decode(p)) @ p[2:] - target
        return root * np.copysign(np.sqrt(np.log1p(r * r)), r)
    margin = 1e-07
    starts = []
    grid = np.geomspace(low, high, 11)
    for (i, fast) in enumerate(grid[:-1]):
        for slow in grid[i + 1:]:
            p0 = np.clip((fast - low) / span, 0.0, 1.0 - margin)
            p1 = np.clip((slow - fast) / (high - fast), margin, 1.0)
            a = matrix(decode((p0, p1)))
            coef = weighted_fit(a, weight)
            for temperature in (5.0, 3.0, 2.0, 1.0, 1.0):
                r = a @ coef - target
                coef = weighted_fit(a, weight / (1.0 + (r / temperature) ** 2))
            p = np.r_[p0, p1, coef]
            r = residual(p)
            starts.append((r @ r, p))
    bound_lo = np.r_[0.0, margin, np.full(coef.size, -np.inf)]
    bound_hi = np.r_[1.0 - margin, 1.0, np.full(coef.size, np.inf)]
    best = min((least_squares(residual, p, bounds=(bound_lo, bound_hi), x_scale='jac', max_nfev=150) for (_, p) in sorted(starts, key=lambda item: item[0])[:4]), key=lambda result: result.cost)
    tau = decode(best.x)
    (p0, p1) = best.x[:2]
    log_time_map = np.array([[span / tau[0], 0.0], [span * (1.0 - p1) / tau[1], span * (1.0 - p0) / tau[1]]])
    j_time = np.linalg.solve(log_time_map.T, best.jac[:, :2].T).T
    j_linear = best.jac[:, 2:]
    (nuisance, _, rank, _) = np.linalg.lstsq(j_linear, j_time, rcond=None)
    profiled = (j_time - j_linear @ nuisance) / np.sqrt(weight.sum())
    sensitivity = float(np.linalg.svd(profiled, compute_uv=False)[-1])
    resolved = bool(best.success and rank == j_linear.shape[1] and (sensitivity >= min_sensitivity))
    coef = best.x[2:].copy()
    coef[2:] /= unit
    return (tau * unit, coef, sensitivity, resolved)
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 26 08:47 ..
scipy 1.13.0
```

### Turn 4 — `execute_in_remote_docker` (`53571 ms` LLM)
- **Rationale**: Use the calibrated coupon panel and exact Prony design, then fit the two thresholded terms jointly with a robust loss. The public check separates parameter recovery from prediction errors by yield regime, lab, and horizon so the next step can address whichever part fails.
- **Alternatives Weighed**: `Fit only strain curves` (rejected: The measured initial rates give an independent constraint on the fast branch and the damage surge.)

```bash
set -e
cd /holdout_task
sed -n '/^def hereditary_design/,/^def fit_/p' /tmp/hegel_wormhole_skills/autogenous_hereditary_spectrum_identification.py | head -60
cat > viscoplastic_overstress_hereditary_solver.py <<'PY'
"""Identify a two-branch Prony law with thresholded viscoplastic flow."""

import numpy as np
from scipy.optimize import least_squares
from hegel_wormhole_skills.autogenous_hereditary_spectrum_identification import (
    align_reference_panel,
    hereditary_design,
)


def _history(split, dt, yield_stress):
    past = np.asarray(split['past_stress'])
    hold = np.asarray(split['stress_hold_0'])
    over = np.sign(hold) * np.maximum(np.abs(hold) - yield_stress, 0.0)**2
    work = np.maximum(np.abs(past) - yield_stress, 0.0)**2 @ dt
    return past, hold, over, work * hold


def _design(history, dt, time, x, taus):
    past, hold, over, work_stress = history
    curve, rate = hereditary_design(past, dt, hold, time, x, taus)
    rise = -np.expm1(-time / taus[0])
    vp_curve = np.stack((over[:, None] * time, work_stress[:, None] * rise), axis=-1)
    vp_rate = np.column_stack((over, work_stress / taus[0]))
    return np.concatenate((curve, vp_curve), axis=-1), np.column_stack((rate, vp_rate))


def solve_viscoplastic_hereditary_revision(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt = np.asarray(dataset['past_dt'])
    time = np.asarray(dataset['eval_times'])
    threshold = float(dataset['yield_stress'])
    n = len(train['lab_id'])
    panel = [np.concatenate((train[key], test[key]), axis=0) for key in
             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum')]
    x, _ = align_reference_panel(*panel)
    train_history = _history(train, dt, threshold)
    test_history = _history(test, dt, threshold)

    target = np.r_[
        (train['strain_curves_obs'] - train['strain_0'][:, None]).ravel(),
        train['instant_rate_obs'],
    ]
    scale = np.r_[np.full(n * len(time), 0.03), np.full(n, 0.045)]
    target = target / scale
    log_low, log_high = np.log([0.2, 8.0])

    def decode(p):
        # Parameterize the second time in the part of the interval above the first.
        fast = p[0]
        return np.exp([fast, fast + p[1] * (log_high - fast)])

    def matrix(p):
        curve, rate = _design(train_history, dt, time, x[:n], decode(p))
        return np.vstack((curve.reshape(n * len(time), -1), rate)) / scale[:, None]

    def residual(p):
        return np.einsum('ij,j->i', matrix(p[:2]), p[2:]) - target

    starts = []
    for fast in np.geomspace(0.24, 1.7, 4):
        for slow in np.geomspace(0.95, 7.8, 5):
            if slow <= fast:
                continue
            p = np.array([np.log(fast), np.log(slow / fast) / np.log(8 / fast)])
            a = matrix(p)
            coef = np.linalg.lstsq(a, target, rcond=None)[0]
            for temperature in (5, 3, 2, 1, 1):
                r = np.einsum('ij,j->i', a, coef) - target
                root = 1 / np.hypot(1, r / temperature)
                coef = np.linalg.lstsq(a * root[:, None], target * root, rcond=None)[0]
            r = np.einsum('ij,j->i', a, coef) - target
            starts.append((np.log1p(r*r).sum(), np.r_[p, coef]))
    bounds = (np.r_[log_low, 1e-8, np.full(coef.size, -np.inf)],
              np.r_[log_high - 1e-8, 1.0, np.full(coef.size, np.inf)])
    best = min((least_squares(residual, p, bounds=bounds, loss='cauchy',
                              x_scale='jac', max_nfev=120)
                for _, p in sorted(starts, key=lambda row: row[0])[:2]),
               key=lambda fit: fit.cost)
    taus = decode(best.x)
    coef = best.x[2:]
    curve, _ = _design(test_history, dt, time, x[n:], taus)
    return {
        'estimated_taus': taus,
        'viscoplastic_weights': coef[-2:],
        'predicted_strain': test['strain_0'][:, None] + curve @ coef,
    }
PY
cat >/tmp/score_viscoplastic.py <<'PY'
import importlib.util, pickle, time
import numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('cand', root+'viscoplastic_overstress_hereditary_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p, = pickle.load(open(root+'public_suc01_C_datasets.pkl', 'rb'))
d, g = p['dataset'], p['gt']
t = time.perf_counter()
o = m.solve_viscoplastic_hereditary_revision(d)
print('seconds', time.perf_counter()-t)
print('tau', o['estimated_taus'], 'rel', abs(o['estimated_taus']/g['taus']-1))
print('vp', o['viscoplastic_weights'], 'rel L2', np.linalg.norm(o['viscoplastic_weights']-g['vp_theta'])/np.linalg.norm(g['vp_theta']))
s = d['test']; y = d['yield_stress']; over = (abs(s['stress_hold_0']) > y) | (abs(s['past_stress']).max(axis=1)>y)
e = abs(o['predicted_strain']-g['te_clean'])
print('MAE all / over / sub', e.mean(), e[over].mean(), e[~over].mean())
for name in ('lab_id', 'stratum'):
 print(name, [(int(i),float(e[s[name] == i].mean())) for i in np.unique(s[name])])
print('MAE by time', e.mean(axis=0))
PY
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp
python3 validate_public_schema.py
python3 /tmp/score_viscoplastic.py
```
- **Container Exit Code**: `0` (`1040 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoplastic_overstress_hereditary_solver.py']`
- **Stdout (tail)**:
```text
history, hold, feature or time shapes')
    if not all((np.isfinite(a).all() for a in (h, dt, hold, t, x, tau))) or np.any(dt <= 0) or np.any(t < 0) or (not 0 < tau[0] < tau[1]):
        raise ValueError('use finite arrays, positive steps and ordered taus')
    unit = np.median(dt)
    (dt, t, tau) = (dt / unit, t / unit, tau / unit)
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    memory = -np.expm1(-dt[:, None] / tau) * np.exp(-tail[:, None] / tau)
    gap = hold[:, None] - h @ memory
    creep = -np.expm1(-t[:, None] / tau)
    drift = np.column_stack((np.ones(len(x)), x))
    curve = np.concatenate((gap[:, None, :] * creep[None, :, :], drift[:, None, :] * (t * unit)[None, :, None]), axis=2)
    rate = np.column_stack((gap / tau / unit, drift))
    return (curve, rate)


@track_wormhole_skill("autogenous_hereditary_spectrum_identification")
def fit_hereditary_spectrum(past_stress: np.ndarray, past_dt: np.ndarray, hold_stress: np.ndarray, eval_times: np.ndarray, features: np.ndarray, initial_strain: np.ndarray, strain_obs: np.ndarray, rate_obs: np.ndarray, relaxation_bounds: tuple[float, float], strain_scale: float | np.ndarray, rate_scale: float | np.ndarray, specimen_weights: np.ndarray | None=None, *, min_sensitivity: float=0.01) -> tuple[np.ndarray, np.ndarray, float, bool]:
{"schema_valid": true, "errors": []}
seconds 0.1618125808890909
tau [0.5808017  3.88661156] rel [0.00138224 0.00343293]
vp [0.24038309 0.16045877] rel L2 0.0020720973627425384
MAE all / over / sub 0.03610824890222553 0.04005704197682255 0.03335154430297857
lab_id [(0, 0.057368687312517944), (1, 0.038355062048594214), (2, 0.016254082185043064), (3, 0.03325549831596572)]
stratum [(0, 0.032372868498344254), (1, 0.04438502840539245), (2, 0.03193828274994847)]
MAE by time [0.00237506 0.00533186 0.00947435 0.01477263 0.0217951  0.03056537
 0.04234118 0.05768016 0.07656652 0.10018027]

[HOST STATIC PACKAGER]: Bundled self-contained `hegel_wormhole_skills` closure into `['/holdout_task/viscoplastic_overstress_hereditary_solver.py']` (`pre_sha=37416b6abd6e` -> `post_sha=70f79f00d26d`). Before `submit_completed_task`, run a clean `python3 -I` verification on the packaged target artifact so `VerificationReceipt` binds to `post_sha`.
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=70f79f00d26de1a4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_70f79f00 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`32959 ms` LLM)
- **Rationale**: Verify that the packaged artifact imports in isolation. Then separate measurement noise from residual drift and check whether rate conditioning is promising for complete and incomplete coupons before testing any correction out of fold.
- **Alternatives Weighed**: `Submit the passing public baseline immediately` (rejected: The packaged solver still needs isolated verification, and the increase in error with time points to a drift bottleneck worth diagnosing before freezing the model.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/score_viscoplastic.py
python3 -I - <<'PY'
import runpy, numpy as np
v = runpy.run_path('/tmp/score_viscoplastic.py'); m,d,g = (v[k] for k in ('m','d','g')); t=d['eval_times']; calls=[]; original=m.least_squares
def record(*a,**kw):
    fit=original(*a,**kw); calls.append(fit); return fit
m.least_squares=record
out=m.solve_viscoplastic_hereditary_revision(d); coef=min(calls,key=lambda x:x.cost).x[2:]; print('fitted coefficients:',coef)
tr,te=d['train'],d['test']; n=len(tr['lab_id']); panel=[np.concatenate((tr[k],te[k])) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')]; x,_=m.align_reference_panel(*panel)
for s,z in ((tr,x[:n]),(te,x[n:])):
    c,r=m._design(m._history(s,d['past_dt'],d['yield_stress']),d['past_dt'],t,z,out['estimated_taus']); q=s['instant_rate_obs']-r@coef; missing=~np.isfinite(s['coupon_features_raw']).all(axis=1)
    print('rate residual quantiles complete/missing:',*[np.quantile(q[mask],[0,.1,.5,.9,1]).round(4) for mask in (~missing,missing)])
    if s is te:
        true_r=g['te_true_rate0']; p=out['predicted_strain']; e=p-g['te_clean']; over=(abs(s['stress_hold_0'])>d['yield_stress']) | (abs(s['past_stress']).max(axis=1)>d['yield_stress']); print('measured rate SD against clean:', np.std(s['instant_rate_obs']-true_r))
        for title,mask in [('all',np.ones(len(q),bool)),('complete',~missing),('missing',missing),('over',over),('sub',~over)]:
            print(title,'n',mask.sum(),'MAE across diagnostic gains 0,.25,.5,.75,1:',[round(np.abs(e[mask]+w*q[mask,None]*t).mean(),5) for w in (0,.25,.5,.75,1)],'rate correction correlation',np.corrcoef(q[mask],(true_r-r@coef)[mask])[0,1].round(4))
PY
```
- **Container Exit Code**: `0` (`1504 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
seconds 0.14742580009624362
tau [0.5808017  3.88661156] rel [0.00138224 0.00343293]
vp [0.24038309 0.16045877] rel L2 0.0020720973627425384
MAE all / over / sub 0.03610824890222553 0.04005704197682255 0.03335154430297857
lab_id [(0, 0.057368687312517944), (1, 0.038355062048594214), (2, 0.016254082185043064), (3, 0.03325549831596572)]
stratum [(0, 0.032372868498344254), (1, 0.04438502840539245), (2, 0.03193828274994847)]
MAE by time [0.00237506 0.00533186 0.00947435 0.01477263 0.0217951  0.03056537
 0.04234118 0.05768016 0.07656652 0.10018027]
seconds 0.15177316009067
tau [0.5808017  3.88661156] rel [0.00138224 0.00343293]
vp [0.24038309 0.16045877] rel L2 0.0020720973627425384
MAE all / over / sub 0.03610824890222553 0.04005704197682255 0.03335154430297857
lab_id [(0, 0.057368687312517944), (1, 0.038355062048594214), (2, 0.016254082185043064), (3, 0.03325549831596572)]
stratum [(0, 0.032372868498344254), (1, 0.04438502840539245), (2, 0.03193828274994847)]
MAE by time [0.00237506 0.00533186 0.00947435 0.01477263 0.0217951  0.03056537
 0.04234118 0.05768016 0.07656652 0.10018027]
fitted coefficients: [ 0.68074259  0.53762395  0.0805946   0.14594988 -0.11737221  0.10765437
 -0.08881886  0.24038309  0.16045877]
rate residual quantiles complete/missing: [-0.0339 -0.0162 -0.      0.0144  0.0394] [-0.3063 -0.2149 -0.0014  0.1183  0.1794]
rate residual quantiles complete/missing: [-0.0357 -0.012  -0.002   0.0184  0.037 ] [-0.0778 -0.0571  0.0098  0.1053  0.1148]
measured rate SD against clean: 0.009766502205689571
all n 90 MAE across diagnostic gains 0,.25,.5,.75,1: [0.03611, 0.02928, 0.02476, 0.0231, 0.0237] rate correction correlation 0.9218
complete n 81 MAE across diagnostic gains 0,.25,.5,.75,1: [0.01797, 0.01506, 0.01471, 0.01754, 0.02288] rate correction correlation 0.6513
missing n 9 MAE across diagnostic gains 0,.25,.5,.75,1: [0.19936, 0.1573, 0.11524, 0.07318, 0.03114] rate correction correlation 0.9932
over n 37 MAE across diagnostic gains 0,.25,.5,.75,1: [0.04006, 0.03305, 0.02919, 0.02794, 0.02781] rate correction correlation 0.937
sub n 53 MAE across diagnostic gains 0,.25,.5,.75,1: [0.03335, 0.02665, 0.02167, 0.01972, 0.02083] rate correction correlation 0.9027

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=70f79f00d26de1a4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_70f79f00 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`47826 ms` LLM)
- **Rationale**: Test the fixed missing-only correction on held-out labs and strata without refitting it against clean test targets. Promote it only if it reduces the observed drift error out of fold and improves clean-test margins without changing parameter recovery.
- **Alternatives Weighed**: `Apply the full rate correction to every coupon` (rejected: On complete coupons it raised public MAE from 0.0180 to 0.0229, consistent with injecting onset-rate noise where the panel already estimates drift.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
cat >/tmp/validate_rate_conditioning.py <<'PY'
import importlib.util
import pickle

import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('cand', root+'viscoplastic_overstress_hereditary_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p, = pickle.load(open(root+'public_suc01_C_datasets.pkl', 'rb'))
d, g = p['dataset'], p['gt']
time = d['eval_times']
fits = []
least_squares = m.least_squares

def capture(*args, **kwargs):
    fit = least_squares(*args, **kwargs)
    fits.append(fit)
    return fit

m.least_squares = capture

def compare(dataset):
    fits.clear()
    train, test = dataset['train'], dataset['test']
    out = m.solve_viscoplastic_hereditary_revision(dataset)
    coef = min(fits, key=lambda fit: fit.cost).x[2:]
    panel = [np.concatenate((train[k], test[k])) for k in
             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum')]
    x, _ = m.align_reference_panel(*panel)
    hist = m._history(test, d['past_dt'], d['yield_stress'])
    _, rate = m._design(hist, d['past_dt'], time, x[len(train['lab_id']):], out['estimated_taus'])
    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    drift = (test['instant_rate_obs'] - rate @ coef) * missing
    base = out['predicted_strain']
    return base, base + drift[:, None] * time, missing

def report(label, split, target, a, b, missing):
    over = (np.abs(split['past_stress']).max(axis=1) > d['yield_stress']) | (np.abs(split['stress_hold_0']) > d['yield_stress'])
    e0, e1 = np.abs(a-target), np.abs(b-target)
    print(label)
    for name, mask in [('all', np.ones(len(a), dtype=bool)), ('over', over), ('sub', ~over), ('missing', missing)]:
        if mask.any():
            print(name, mask.sum(), 'MAE baseline/conditioned', np.round([e0[mask].mean(), e1[mask].mean()], 5), 'last horizon', np.round([e0[mask, -1].mean(), e1[mask, -1].mean()], 5))
    return e0, e1

s = d['train']
test_keys = d['test'].keys()
for source in ('lab_id', 'stratum'):
    a = np.empty_like(s['strain_curves_obs'])
    b = np.empty_like(a)
    missing = np.zeros(len(a), dtype=bool)
    for label in np.unique(s[source]):
        mask = s[source] == label
        fold = dict(d, train={k:v[~mask] for k,v in s.items()}, test={k:s[k][mask] for k in test_keys})
        a[mask], b[mask], missing[mask] = compare(fold)
    e0, e1 = report('Out of fold by '+source+' (against observed strain)', s, s['strain_curves_obs'], a, b, missing)
    print('missing-only fold mean absolute errors', [(int(k), round(e0[(s[source]==k)&missing].mean(), 5), round(e1[(s[source]==k)&missing].mean(), 5)) for k in np.unique(s[source][missing])])
a, b, missing = compare(d)
report('Public clean test', d['test'], g['te_clean'], a, b, missing)
PY
python3 -I /tmp/validate_rate_conditioning.py
```
- **Container Exit Code**: `0` (`1530 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Out of fold by lab_id (against observed strain)
all 148 MAE baseline/conditioned [0.11531 0.08869] last horizon [0.18234 0.11104]
over 74 MAE baseline/conditioned [0.12681 0.09237] last horizon [0.18643 0.0973 ]
sub 74 MAE baseline/conditioned [0.10381 0.08501] last horizon [0.17826 0.12479]
missing 15 MAE baseline/conditioned [0.35262 0.08995] last horizon [0.86596 0.16245]
missing-only fold mean absolute errors [(0, 0.49236, 0.15369), (1, 0.05528, 0.04057), (2, 0.40177, 0.10249), (3, 0.26622, 0.0128)]
Out of fold by stratum (against observed strain)
all 148 MAE baseline/conditioned [0.11425 0.08764] last horizon [0.17881 0.10739]
over 74 MAE baseline/conditioned [0.12608 0.09171] last horizon [0.18344 0.09415]
sub 74 MAE baseline/conditioned [0.10242 0.08358] last horizon [0.17418 0.12063]
missing 15 MAE baseline/conditioned [0.35305 0.0905 ] last horizon [0.86691 0.16223]
missing-only fold mean absolute errors [(0, 0.53547, 0.07436), (1, 0.06375, 0.05679), (2, 0.36559, 0.13684)]
Public clean test
all 90 MAE baseline/conditioned [0.03611 0.01929] last horizon [0.10018 0.05352]
over 37 MAE baseline/conditioned [0.04006 0.01596] last horizon [0.11131 0.04448]
sub 53 MAE baseline/conditioned [0.03335 0.02161] last horizon [0.09241 0.05984]
missing 9 MAE baseline/conditioned [0.19936 0.03114] last horizon [0.55304 0.08649]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=70f79f00d26de1a4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_70f79f00 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`28152 ms` LLM)
- **Rationale**: Promote the correction supported by both held-out lab and held-out stratum results. Verify the edited artifact in isolated Python, checking the public contract and the incomplete-coupon and final-horizon slices where an integration error would erase the measured gain.
- **Alternatives Weighed**: `Keep the passing imputation-only solver` (rejected: The missing-only correction improved every held-out lab and stratum on incomplete coupons without altering parameter estimation.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('viscoplastic_overstress_hereditary_solver.py')
s = p.read_text()
old = "    curve, _ = _design(test_history, dt, time, x[n:], taus)\n    return {"
new = "    curve, rate = _design(test_history, dt, time, x[n:], taus)\n    prediction = test['strain_0'][:, None] + curve @ coef\n    # Coupon imputation affects only the steady drift. For incomplete panels,\n    # use the observed onset rate after subtracting the modeled contributions.\n    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)\n    drift_residual = test['instant_rate_obs'][missing] - rate[missing] @ coef\n    prediction[missing] += drift_residual[:, None] * time\n    return {"
assert s.count(old) == 1
s = s.replace(old, new).replace("'predicted_strain': test['strain_0'][:, None] + curve @ coef,", "'predicted_strain': prediction,")
p.write_text(s)
PY
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import runpy
import numpy as np
v = runpy.run_path('/tmp/score_viscoplastic.py')
o, g, e, over, s = (v[k] for k in ('o', 'g', 'e', 'over', 's'))
tau = o['estimated_taus']; vp = o['viscoplastic_weights']
assert 0.2 <= tau[0] < tau[1] <= 8.0
assert np.max(np.abs(tau / g['taus'] - 1)) <= 0.10
assert np.linalg.norm(vp-g['vp_theta']) <= 0.14 * np.linalg.norm(g['vp_theta'])
assert e.mean() <= 0.070 and e[over].mean() <= 0.080 and e[~over].mean() <= 0.065
missing = ~np.isfinite(s['coupon_features_raw']).all(axis=1)
print('Public missing-only MAE and final-horizon MAE:', e[missing].mean(), e[missing, -1].mean())
assert e[missing].mean() < 0.04 and e[:, -1].mean() < 0.06
PY
```
- **Container Exit Code**: `1` (`399 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=70f79f00d26de1a4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_70f79f00 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
AssertionError
```

### Turn 8 — `execute_in_remote_docker` (`30182 ms` LLM)
- **Rationale**: The failed edit left the verified baseline unchanged. Patch only the packaged solver’s prediction block, then run isolated schema and numerical checks to catch an integration error that would erase the validated gain or change parameter recovery.
- **Alternatives Weighed**: `Retry the original text replacement` (rejected: The packaged source did not match that exact text; locating the prediction block structurally does not depend on its formatting.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
cd /holdout_task
python3 - <<'PY'
import ast
from pathlib import Path

path = Path('viscoplastic_overstress_hereditary_solver.py')
source = path.read_text()
module = ast.parse(source)
solver, = (node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'solve_viscoplastic_hereditary_revision')
assignment, = (node for node in solver.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == '_design' and isinstance(node.value.args[0], ast.Name) and node.value.args[0].id == 'test_history')
result, = (node for node in solver.body if isinstance(node, ast.Return))
assert assignment.end_lineno + 1 == result.lineno
indent = ' ' * assignment.col_offset
replacement = '''curve, rate = _design(test_history, dt, time, x[n:], taus)
prediction = test['strain_0'][:, None] + curve @ coef
# Imputation affects only steady drift. Recover it from the measured onset
# rate on incomplete coupons after subtracting the modeled contributions.
missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
drift_residual = test['instant_rate_obs'][missing] - rate[missing] @ coef
prediction[missing] += drift_residual[:, None] * time
return {
    'estimated_taus': taus,
    'viscoplastic_weights': coef[-2:],
    'predicted_strain': prediction,
}'''
lines = source.splitlines(keepends=True)
lines[assignment.lineno - 1:result.end_lineno] = [indent + line + '\n' for line in replacement.splitlines()]
updated = ''.join(lines)
ast.parse(updated)
path.write_text(updated)
PY
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import runpy
import numpy as np
v = runpy.run_path('/tmp/score_viscoplastic.py')
o, g, e, over, s = (v[k] for k in ('o', 'g', 'e', 'over', 's'))
tau = o['estimated_taus']; vp = o['viscoplastic_weights']
assert 0.2 <= tau[0] < tau[1] <= 8.0
assert np.max(np.abs(tau / g['taus'] - 1)) <= 0.10
assert np.linalg.norm(vp-g['vp_theta']) <= 0.14 * np.linalg.norm(g['vp_theta'])
assert e.mean() <= 0.070 and e[over].mean() <= 0.080 and e[~over].mean() <= 0.065
missing = ~np.isfinite(s['coupon_features_raw']).all(axis=1)
print('Incomplete-coupon MAE and overall final-horizon MAE:', e[missing].mean(), e[:, -1].mean())
assert e[missing].mean() < 0.04 and e[:, -1].mean() < 0.06
PY
```
- **Container Exit Code**: `0` (`1189 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoplastic_overstress_hereditary_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
seconds 0.1679832860827446
tau [0.5808017  3.88661156] rel [0.00138224 0.00343293]
vp [0.24038309 0.16045877] rel L2 0.0020720973627425384
MAE all / over / sub 0.019286649700153032 0.015959644870644746 0.02160927571320599
lab_id [(0, 0.04203043501493479), (1, 0.009573197095407935), (2, 0.009337793165981467), (3, 0.01709891484878416)]
stratum [(0, 0.025953284884391255), (1, 0.010364540654339257), (2, 0.020670299744750834)]
MAE by time [0.00127837 0.00286512 0.00508886 0.0079174  0.0116424  0.01629434
 0.0225813  0.03078476 0.04088896 0.05352499]
Incomplete-coupon MAE and overall final-horizon MAE: 0.031141888326381033 0.05352499084633609

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5dba10373d18ed8c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_5dba1037 (directory).
```

### Turn 9 — `submit_completed_task` (`15376 ms` LLM)
- **Rationale**: The exact final artifact passed isolated schema and numerical verification. The correction improved the targeted worst slices out of fold, and all public contract metrics passed with margin.
- **Alternatives Weighed**: `Retain the imputation-only baseline` (rejected: The onset-rate correction improved incomplete-coupon predictions in every held-out lab and stratum without changing the fitted material parameters.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/viscoplastic_overstress_hereditary_solver.py': 'FILE:8669'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoplastic_overstress_hereditary_solver.py`. Public strain MAE was 0.0193 overall, 0.0160 above yield, and 0.0216 sub-yield. The maximum relaxation-time relative error was 0.34%, and the viscoplastic-weight relative L2 error was 0.21%. The schema and all numerical tolerances passed in isolated Python.`

### Turn 10 — `execute_in_remote_docker` (`34933 ms` LLM)
- **Rationale**: Separate hold-only, history-only, and signed loading cases so pooled MAE cannot hide an incorrect threshold or sign. Refit a Prony-only ablation with the same alignment and drift conditioning and compare within labs. A failed design identity warrants a code correction; a gain confined to pooled data warrants investigating lab calibration before submission.
- **Alternatives Weighed**: `Add rational terms or test relay modes and classification accuracy` (rejected: The contract specifies a continuous constitutive law with no relay state or classification target. A within-lab regression comparison tests the relevant confounding risk.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
cat >/tmp/audit_viscoplastic.py <<'PY'
import runpy
import numpy as np

v = runpy.run_path('/tmp/score_viscoplastic.py')
m, d, g, out = (v[k] for k in ('m', 'd', 'g', 'o'))
s, threshold = d['test'], d['yield_stress']
dt, time = d['past_dt'], d['eval_times']
held = abs(s['stress_hold_0']) > threshold
past = (abs(s['past_stress']) > threshold).any(axis=1)
history = m._history(s, dt, threshold)
np.testing.assert_allclose(history[2], np.sign(s['stress_hold_0']) * np.maximum(abs(s['stress_hold_0']) - threshold, 0)**2)
np.testing.assert_allclose(history[3], s['stress_hold_0'] * np.sum(dt * np.maximum(abs(s['past_stress']) - threshold, 0)**2, axis=1))
x = np.zeros((len(held), 4))
tau = out['estimated_taus']
eps = 1e-6
c, r = m._design(history, dt, np.array([0., eps]), x, tau)
np.testing.assert_allclose(c[:, 0], 0., atol=1e-12)
np.testing.assert_allclose(c[:, 1] / eps, r, rtol=5e-6, atol=1e-9)
assert not np.any(c[~held, :, -2]) and not np.any(c[~past, :, -1])

design = m._design

def ablated(history, dt, time, x, taus):
    curve, rate = design(history, dt, time, x, taus)
    return curve[..., :-2], rate[..., :-2]

m._design = ablated
try:
    baseline = m.solve_viscoplastic_hereditary_revision(d)['predicted_strain']
finally:
    m._design = design

a, b = (abs(pred - g['te_clean']) for pred in (baseline, out['predicted_strain']))
for name, mask in [('sub-yield', ~(held | past)), ('hold only', held & ~past), ('history only', past & ~held), ('both', held & past), ('positive hold', held & (s['stress_hold_0'] > 0)), ('negative hold', held & (s['stress_hold_0'] < 0))]:
    if mask.any():
        print(name, int(mask.sum()), 'MAE Prony-only / full', np.round([a[mask].mean(), b[mask].mean()], 5), 'full last', round(b[mask, -1].mean(), 5))
for lab in np.unique(s['lab_id']):
    for label, cohort in [('over', held | past), ('sub', ~(held | past))]:
        mask = (s['lab_id'] == lab) & cohort
        if mask.any():
            print('lab', int(lab), label, int(mask.sum()), 'MAE Prony-only / full', np.round([a[mask].mean(), b[mask].mean()], 5))
assert b.mean() <= .070 and b[held | past].mean() <= .080 and b[~(held | past)].mean() <= .065
assert np.max(abs(tau / g['taus'] - 1)) <= .10
assert np.linalg.norm(out['viscoplastic_weights'] - g['vp_theta']) <= .14 * np.linalg.norm(g['vp_theta'])
PY
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/audit_viscoplastic.py
```
- **Container Exit Code**: `0` (`1264 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
seconds 0.15201403689570725
tau [0.5808017  3.88661156] rel [0.00138224 0.00343293]
vp [0.24038309 0.16045877] rel L2 0.0020720973627425384
MAE all / over / sub 0.019286649700153032 0.015959644870644746 0.02160927571320599
lab_id [(0, 0.04203043501493479), (1, 0.009573197095407935), (2, 0.009337793165981467), (3, 0.01709891484878416)]
stratum [(0, 0.025953284884391255), (1, 0.010364540654339257), (2, 0.020670299744750834)]
MAE by time [0.00127837 0.00286512 0.00508886 0.0079174  0.0116424  0.01629434
 0.0225813  0.03078476 0.04088896 0.05352499]
sub-yield 53 MAE Prony-only / full [0.32754 0.02161] full last 0.05984
both 37 MAE Prony-only / full [0.80991 0.01596] full last 0.04448
positive hold 24 MAE Prony-only / full [0.94147 0.01435] full last 0.03988
negative hold 13 MAE Prony-only / full [0.56704 0.01893] full last 0.05298
lab 0 over 6 MAE Prony-only / full [0.63697 0.02968]
lab 0 sub 16 MAE Prony-only / full [0.342   0.04666]
lab 1 over 10 MAE Prony-only / full [1.18567 0.00788]
lab 1 sub 13 MAE Prony-only / full [0.29937 0.01087]
lab 2 over 7 MAE Prony-only / full [0.34688 0.00882]
lab 2 sub 16 MAE Prony-only / full [0.33271 0.00957]
lab 3 over 14 MAE Prony-only / full [0.84714 0.01942]
lab 3 sub 8 MAE Prony-only / full [0.33409 0.01304]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5dba10373d18ed8c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_5dba1037 (directory).
```

### Turn 11 — `submit_completed_task` (`20893 ms` LLM)
- **Rationale**: The audit verified the threshold, sign, and initial-rate identities before comparing mechanisms. The final artifact passed isolated schema and numerical checks, and the within-lab improvements support retaining the two specified overstress terms.
- **Alternatives Weighed**: `Retain the Prony-only ablation` (rejected: It failed the overstressed tolerance and performed worse in every observed lab-by-yield slice.)
- **ATMS NOGOOD DECLARED**: Refuted `The linear Prony law remains sufficient when stress exceeds yield.` | Witness: `The refitted Prony-only ablation had overstressed MAE 0.80991 versus 0.01596 for the full solver. The full solver improved both hold-stress signs and every observed lab-by-yield slice.` | Blanket Diagnosis: `The improvement persisted within each lab under the same coupon alignment and onset-rate conditioning, so it was not an artifact of pooling labs.` | Sublated Pivot: `Preserve exact sub-yield Prony memory and add the specified signed quadratic hold flow and history-dependent transient surge.`
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/viscoplastic_overstress_hereditary_solver.py': 'FILE:8669'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoplastic_overstress_hereditary_solver.py`. Public strain MAE was 0.0193 overall, 0.0160 for overstressed coupons, and 0.0216 for sub-yield coupons. The maximum relaxation-time relative error was 0.34%, and the viscoplastic-weight relative L2 error was 0.21%. The schema and all numerical tolerances passed in isolated Python.`

### Turn 12 — `execute_in_remote_docker` (`33315 ms` LLM)
- **Rationale**: The prior audit found no naturally isolated activation cases. Clipping one component of measured load paths tests for unintended coupling in the implementation; omitting one term at a time tests whether each improves predictions on the real public cohort. An activation failure calls for a design fix, while an ablation with no gain calls for an identifiability check.
- **Alternatives Weighed**: `Repeat the joint Prony-only ablation` (rejected: It already established a within-lab gain but cannot distinguish the two terms because both activate together in the public observations.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I - <<'PY'
import runpy
import numpy as np
v = runpy.run_path('/tmp/score_viscoplastic.py')
m, d, g, full = (v[k] for k in ('m','d','g','o'))
s = d['test']; dt = d['past_dt']; t = d['eval_times']; y = d['yield_stress']
raw = m._history(s, dt, y)
mask = (np.abs(raw[1]) > y) & (np.abs(raw[0]) > y).any(axis=1)
x = np.zeros((mask.sum(), 4)); tau = full['estimated_taus']
past, hold = raw[0][mask], raw[1][mask]
for hold_on, history_on in ((True, False), (False, True)):
    split = {'past_stress': past if history_on else np.clip(past, -y, y), 'stress_hold_0': hold if hold_on else np.clip(hold, -y, y)}
    h = m._history(split, dt, y)
    c, r = m._design(h, dt, t, x, tau)
    inactive = -1 if hold_on else -2
    assert not np.any(c[..., inactive]) and not np.any(r[:, inactive])
    if hold_on:
        expected = np.sign(hold) * (np.abs(hold) - y)**2
        np.testing.assert_allclose(c[..., -2], expected[:, None] * t)
        np.testing.assert_allclose(r[:, -2], expected)
    else:
        expected = ((np.maximum(np.abs(past) - y, 0)**2) @ dt) * split['stress_hold_0']
        np.testing.assert_allclose(c[..., -1], expected[:, None] * (-np.expm1(-t/tau[0])))
        np.testing.assert_allclose(r[:, -1], expected/tau[0])
print('Independent activation identities passed on clipped measured load paths')
base_design = m._design
clean = g['te_clean']
full_mae = np.abs(full['predicted_strain'] - clean)[mask].mean()
for drop, name in ((-2, 'hold creep'), (-1, 'plastic-work surge')):
    def ablated(*args):
        c, r = base_design(*args)
        return np.delete(c, drop, axis=-1), np.delete(r, drop, axis=-1)
    m._design = ablated
    try:
        pred = m.solve_viscoplastic_hereditary_revision(d)['predicted_strain']
    finally:
        m._design = base_design
    err = np.abs(pred - clean)
    print('Omit', name, 'overstressed MAE', round(err[mask].mean(), 5), 'full', round(full_mae, 5))
    assert err[mask].mean() > full_mae
PY
```
- **Container Exit Code**: `0` (`959 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
seconds 0.15887511405162513
tau [0.5808017  3.88661156] rel [0.00138224 0.00343293]
vp [0.24038309 0.16045877] rel L2 0.0020720973627425384
MAE all / over / sub 0.019286649700153032 0.015959644870644746 0.02160927571320599
lab_id [(0, 0.04203043501493479), (1, 0.009573197095407935), (2, 0.009337793165981467), (3, 0.01709891484878416)]
stratum [(0, 0.025953284884391255), (1, 0.010364540654339257), (2, 0.020670299744750834)]
MAE by time [0.00127837 0.00286512 0.00508886 0.0079174  0.0116424  0.01629434
 0.0225813  0.03078476 0.04088896 0.05352499]
Independent activation identities passed on clipped measured load paths
Omit hold creep overstressed MAE 0.37231 full 0.01596
Omit plastic-work surge overstressed MAE 0.61126 full 0.01596

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5dba10373d18ed8c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_5dba1037 (directory).
```

### Turn 13 — `submit_completed_task` (`15566 ms` LLM)
- **Rationale**: The verified artifact resolves the model deficiency recorded in ng_t11. Independent activation checks on clipped measured load paths confirmed the threshold and onset-rate identities, and the public ablations showed that both terms are needed. No further artifact changes are warranted.
- **Alternatives Weighed**: `Retain only one overstress term` (rejected: Omitting hold creep raised public overstressed MAE to 0.3723; omitting the plastic-work surge raised it to 0.6113. Both exceed the 0.080 tolerance.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoplastic_overstress_hereditary_solver.py`. Public strain MAE was 0.0193 overall, 0.0160 for overstressed coupons, and 0.0216 for sub-yield coupons. The maximum relaxation-time relative error was 0.34%, and the viscoplastic-weight relative L2 error was 0.21%. The final artifact passed all public tolerances in isolated Python.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M93-Lifecycle",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/viscoplastic_overstress_hereditary_solver.py": "FILE:8669"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "max_tau_rel_err": 0.026663,
    "max_vp_rel_l2": 0.008278,
    "max_all_mae": 0.023647,
    "max_overstress_mae": 0.021858,
    "max_subyield_mae": 0.024896,
    "student_elapsed_sec": 0.2942,
    "instance_diagnostics": [
      {
        "seed": 99521,
        "tau_rel": 0.026663,
        "vp_rel_l2": 0.005978,
        "all_mae": 0.023647,
        "over_mae": 0.021858,
        "sub_mae": 0.024896
      },
      {
        "seed": 99522,
        "tau_rel": 0.026215,
        "vp_rel_l2": 0.008278,
        "all_mae": 0.016431,
        "over_mae": 0.014731,
        "sub_mae": 0.017619
      }
    ],
    "candidate_timing_breakdown": {
      "import_wall_sec": 0.099724,
      "import_cpu_sec": 0.091589,
      "candidate_call_wall_sec": 0.247148,
      "candidate_call_cpu_sec": 0.227053,
      "ipc_and_codec_wall_sec": 0.127745,
      "rpc_call_count": 2,
      "verifier_and_gt_wall_sec": 0.008715,
      "verifier_and_gt_cpu_sec": 0.03073,
      "total_supervisor_wall_sec": 0.483332
    }
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"max_tau_rel_err\": 0.026663, \"max_vp_rel_l2\": 0.008278, \"max_all_mae\": 0.023647, \"max_overstress_mae\": 0.021858, \"max_subyield_mae\": 0.024896, \"student_elapsed_sec\": 0.2942, \"instance_diagnostics\": [{\"seed\": 99521, \"tau_rel\": 0.026663, \"vp_rel_l2\": 0.005978, \"all_mae\": 0.023647, \"over_mae\": 0.021858, \"sub_mae\": 0.024896}, {\"seed\": 99522, \"tau_rel\": 0.026215, \"vp_rel_l2\": 0.008278, \"all_mae\": 0.016431, \"over_mae\": 0.014731, \"sub_mae\": 0.017619}]}\n__HEGEL_VERIFIER_AUTH_56407057476345206733ff0dc88190ad__:{\"strict_pass\": true, \"reward\": 1.0, \"max_tau_rel_err\": 0.026663, \"max_vp_rel_l2\": 0.008278, \"max_all_mae\": 0.023647, \"max_overstress_mae\": 0.021858, \"max_subyield_mae\": 0.024896, \"student_elapsed_sec\": 0.2942, \"instance_diagnostics\": [{\"seed\": 99521, \"tau_rel\": 0.026663, \"vp_rel_l2\": 0.005978, \"all_mae\": 0.023647, \"over_mae\": 0.021858, \"sub_mae\": 0.024896}, {\"seed\": 99522, \"tau_rel\": 0.026215, \"vp_rel_l2\": 0.008278, \"all_mae\": 0.016431, \"over_mae\": 0.014731, \"sub_mae\": 0.017619}], \"candidate_timing_breakdown\": {\"import_wall_sec\": 0.099724, \"import_cpu_sec\": 0.091589, \"candidate_call_wall_sec\": 0.247148, \"candidate_call_cpu_sec\": 0.227053, \"ipc_and_codec_wall_sec\": 0.127745, \"rpc_call_count\": 2, \"verifier_and_gt_wall_sec\": 0.008715, \"verifier_and_gt_cpu_sec\": 0.03073, \"total_supervisor_wall_sec\": 0.483332}}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5361.25,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
