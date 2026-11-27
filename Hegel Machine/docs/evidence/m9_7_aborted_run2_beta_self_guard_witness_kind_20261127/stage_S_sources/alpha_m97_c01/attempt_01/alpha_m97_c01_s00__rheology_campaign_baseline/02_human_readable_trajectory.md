# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c01_s00__rheology_campaign_baseline`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c01_source` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `306,826` (`294,500` / `80,896` / `12,326`) |
| **Total Episode Wall Time (s)** | `394.24s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`17676 ms` LLM)
- **Rationale**: Inspect the public schema before choosing the fitting implementation: unexpected layouts would change data extraction, and the calibration rows determine whether archive fallback is needed.

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && python3 - <<'PY'
import pickle
import numpy as np
with open('public_alpha_m97_c01_s00.pkl','rb') as f: d=pickle.load(f)
def show(obj, depth=0):
    if isinstance(obj,dict):
        for k,v in obj.items():
            print('  '*depth,k,type(v).__name__,getattr(v,'shape',''))
            if depth<2: show(v,depth+1)
    elif isinstance(obj,(list,tuple)):
        print('  '*depth,'length',len(obj),'first',repr(obj[0])[:500] if len(obj) else '')
    elif isinstance(obj,np.ndarray):
        print('  '*depth,obj.dtype,repr(obj.ravel()[:8]))
    else: print('  '*depth,repr(obj)[:250])
show(d)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust Cauchy IRLS variable projection; reference-panel alignment imputation; viscoelastic relaxation' 2>&1 | head -35
```
- **Container Exit Code**: `0` (`477 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
le)
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
```

### Turn 2 — `execute_in_remote_docker` (`20009 ms` LLM)
- **Rationale**: Inspect the compact split layout, panel dimensions and calibration residuals: these determine reference normalization, missing-value weights and whether the archive is needed.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_alpha_m97_c01_s00.pkl','rb')); print('item keys',list(x[0])); d=x[0]['dataset']; print('dataset keys',list(d));
for name in ('train','test'):
 s=d[name];print(name)
 for k,v in s.items():
  if isinstance(v,dict):print(k,[(j,np.shape(w),np.asarray(w).ravel()[:4]) for j,w in v.items()]);continue
  a=np.asarray(v);print(k,a.shape,a.dtype,(a.ravel()[:5] if a.dtype.kind!='O' else str(v)[:120]), ('NaN',np.isnan(a).sum()) if a.dtype.kind=='f' else '')
 if 'stratum' in s: print('lab-stratum cells',np.unique(np.column_stack([s['lab_id'],s['stratum']]),axis=0,return_counts=True))
c=d['calibration_load_cells'];
for lab in np.unique(c['lab_id']):
 m=c['lab_id']==lab; a,b=c['cmd_stress'][m],c['true_stress'][m]; q=np.polyfit(a,b,1); print('cal',lab,len(a),q,np.round(b-np.polyval(q,a),6))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel | head -45
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | head -28
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | head -35
ls -l /tmp/hegel_continual_archive/archive_calibration_history.json
```
- **Container Exit Code**: `2` (`500 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
constant within both segments. The scale and loss_clip are positive finite scalars. Interpretation as a level step assumes a shared local trend on both sides."
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
      ],
```
- **Stderr (tail)**:
```text
ls: cannot access '/tmp/hegel_continual_archive/archive_calibration_history.json': No such file or directory
```

### Turn 3 — `execute_in_remote_docker` (`78223 ms` LLM)
- **Rationale**: Implement the exact memory and rate equations, then test the schema and hold out each lab. Unstable fitted parameters or concentrated missing-coupon error would motivate revisiting the optimizer or imputation.
- **Alternatives Weighed**: `Ordinary least squares on all observations` (rejected: The documented Cauchy glitches can distort both relaxation times and drift.); `Generic reference-panel primitive` (rejected: The task specifies median per-column reference statistics and equal-cell imputation; implementing these directly preserves that convention.)

```bash
cd /holdout_task && cat > alpha_m97_c01_s00_solver.py <<'PY'
"""Calibrated two-branch hereditary model for amorphous melt flow."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _calibrate(dataset):
    rows = dataset['calibration_load_cells']
    lab = np.asarray(rows['lab_id'])
    cmd = np.asarray(rows['cmd_stress'], float)
    true = np.asarray(rows['true_stress'], float)
    coeff = np.empty((4, 2))
    archive_path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
    for channel in range(4):
        m = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[m], true[m]
        i, j = np.triu_indices(x.size, 1)
        distinct = x[i] != x[j]
        i, j = i[distinct], j[distinct]
        clean = np.ones(x.size, dtype=bool)
        if len(i):
            slopes = (y[i] - y[j]) / (x[i] - x[j])
            # Use the line with greatest clean support to screen isolated EMI.
            lines = y[i, None] + slopes[:, None] * (x - x[i, None])
            err = np.abs(lines - y)
            best = min(range(len(i)), key=lambda k: (-np.sum(err[k] <= 3), np.median(err[k])))
            clean = err[best] <= 3.0
        if (len(x) < 2 or not clean.all()) and archive_path.is_file():
            with archive_path.open() as f:
                archive = json.load(f)
            historic = archive.get('calibration_load_cells', archive) if isinstance(archive, dict) else archive
            if isinstance(historic, dict) and 'lab_id' in historic:
                historic = [dict(zip(('lab_id', 'cmd_stress', 'true_stress'), r)) for r in zip(historic['lab_id'], historic['cmd_stress'], historic['true_stress'])]
            if isinstance(historic, list) and clean.sum() < 4:
                points = [(r['cmd_stress'], r['true_stress']) for r in historic
                          if {'lab_id', 'cmd_stress', 'true_stress'} <= r.keys() and int(r['lab_id']) == channel]
                if points:
                    xx, yy = np.asarray(points, float).T
                    x, y = np.r_[x[clean], xx], np.r_[y[clean], yy]
                    clean = np.ones(x.size, dtype=bool)
        x, y = x[clean], y[clean]
        if x.size >= 2 and np.ptp(x) > 0:
            coeff[channel] = np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]
        else:
            coeff[channel] = 1.0, np.median(y - x) if x.size else 0.0
    return coeff


def _inputs(dataset):
    splits = (dataset['train'], dataset['test'])

    def stack(key):
        return np.concatenate([s[key] for s in splits], axis=0)

    lab, strata = stack('lab_id'), stack('stratum')
    raw, ref = stack('coupon_features_raw'), stack('ref_coupons')
    aligned = np.empty_like(raw, dtype=float)
    for channel in np.unique(lab):
        m = lab == channel
        center = np.nanmedian(np.nanmean(ref[m], axis=0))
        scale = np.nanmedian(np.nanstd(ref[m], axis=0, ddof=0))
        aligned[m] = (raw[m] - center) / scale
    cells = np.unique(np.column_stack((lab, strata)), axis=0)
    means = []
    for c, s in cells:
        v = aligned[(lab == c) & (strata == s)]
        count = np.isfinite(v).sum(axis=0)
        means.append(np.divide(np.nansum(v, axis=0), count,
                               out=np.full(v.shape[1], np.nan), where=count > 0))
    fill = np.nanmean(means, axis=0)
    aligned = np.where(np.isfinite(aligned), aligned, fill)
    gain, bias = _calibrate(dataset)[lab].T
    past = gain[:, None] * stack('past_stress_cmd') + bias[:, None]
    hold = gain * stack('stress_hold_cmd') + bias
    return past, hold, aligned, stack('strain_0')


class _Model:
    def __init__(self, dt, times, past, hold, features):
        self.dt = np.asarray(dt)
        self.after = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.0]
        self.t = np.asarray(times)
        self.past = past
        self.hold = hold
        self.x = np.column_stack((np.ones(len(hold)), features))

    def design(self, log_taus):
        tau = np.exp(log_taus)
        u, a, t = (z[:, None] / tau for z in (self.dt, self.after, self.t))
        ed, ea, et = np.exp(-u), np.exp(-a), np.exp(-t)
        step, response = -np.expm1(-u), -np.expm1(-t)
        gap = self.hold[:, None] - self.past @ (ea * step)
        dgap = -self.past @ (ea * (a * step - u * ed))
        n, k = len(gap), len(self.t)
        design = np.empty((n, k + 1, 7))
        deriv = np.empty((n, k + 1, 2))
        design[:, :-1, :2] = gap[:, None, :] * response
        design[:, -1, :2] = gap / tau
        design[:, :-1, 2:] = self.x[:, None, :] * self.t[None, :, None]
        design[:, -1, 2:] = self.x
        deriv[:, :-1] = dgap[:, None, :] * response - gap[:, None, :] * (t * et)
        deriv[:, -1] = (dgap - gap) / tau
        return design, deriv


def _mad(values):
    return max(1.4826 * np.median(np.abs(values - np.median(values))), 1e-8)


def _fit(model, y):
    cache = [None, None, None]

    def evaluate(theta, scale):
        if cache[0] is None or not np.array_equal(theta[:2], cache[0]):
            cache[:] = [theta[:2].copy(), *model.design(theta[:2])]
        design, deriv = cache[1:]
        return (design @ theta[2:] - y) / scale, design, deriv

    def fun(theta, scale):
        return evaluate(theta, scale)[0].ravel()

    def jac(theta, scale):
        _, design, deriv = evaluate(theta, scale)
        j = np.concatenate((deriv * theta[2:4], design), axis=2)
        return (j / np.asarray(scale)[..., None]).reshape(-1, 9)

    t_min, t_max = model.t[model.t > 0].min(), model.t.max()
    low, high = np.log(t_min / 25), np.log(t_max * 20)
    bounds = (np.r_[low, low, np.full(7, -np.inf)],
              np.r_[high, high, np.full(7, np.inf)])
    seeds = np.log(t_max * np.array([[.07, .5], [.025, .3], [.14, 1.5], [.3, 2.5]]))
    p = seeds[0]
    design, _ = model.design(p)
    beta = np.linalg.lstsq(design.reshape(-1, 7), y.ravel(), rcond=None)[0]
    r = design @ beta - y
    scale = np.r_[np.full(y.shape[1] - 1, _mad(r[:, :-1])), _mad(r[:, -1])]
    warm = least_squares(fun, np.r_[p, beta], jac=jac, args=(scale,), bounds=bounds,
                         loss='soft_l1', max_nfev=120).x
    r = (fun(warm, scale).reshape(y.shape) * scale)
    scale = 1.25 * np.r_[np.full(y.shape[1] - 1, _mad(r[:, :-1])), _mad(r[:, -1])]
    candidates = [warm]
    for p in seeds:
        design, _ = model.design(p)
        weighted = design / scale[:, None]
        beta = np.linalg.lstsq(weighted.reshape(-1, 7), (y / scale).ravel(), rcond=None)[0]
        candidates.append(np.r_[p, beta])
    best = min((least_squares(fun, p, jac=jac, args=(scale,), bounds=bounds,
                              loss='cauchy', max_nfev=150) for p in candidates), key=lambda r: r.cost)
    taus, weights = np.exp(best.x[:2]), best.x[2:].copy()
    order = np.argsort(taus)
    weights[:2] = weights[:2][order]
    return taus[order], weights


def solve_alpha_m97_c01_s00(dataset: dict) -> dict:
    past, hold, x, strain0 = _inputs(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    t, dt = dataset['eval_times'], dataset['past_dt']
    y = np.column_stack((train['strain_curves_obs'] - strain0[:n, None], train['instant_rate_obs']))
    taus, weights = _fit(_Model(dt, t, past[:n], hold[:n], x[:n]), y)
    design, _ = _Model(dt, t, past[n:], hold[n:], x[n:]).design(np.log(taus))
    pred = design @ weights
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2),
            'predicted_strain': strain0[n:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle, numpy as np, time
from alpha_m97_c01_s00_solver import _inputs, _Model, _fit
with open('public_alpha_m97_c01_s00.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
p,h,x,b=_inputs(d); s=d['train']; n=len(s['lab_id']); p,h,x=p[:n],h[:n],x[:n]; y=np.column_stack((s['strain_curves_obs']-b[:n,None],s['instant_rate_obs'])); t=d['eval_times']; dt=d['past_dt']; start=time.monotonic()
for c in [-1,0,1,2,3]:
 m=s['lab_id']!=c; v=~m if c>=0 else m
 tau,w=_fit(_Model(dt,t,p[m],h[m],x[m]),y[m]); A,_=_Model(dt,t,p[v],h[v],x[v]).design(np.log(tau)); e=abs(A@w-y[v]); miss=np.isnan(s['coupon_features_raw'][v]).any(axis=1); print(c, 'tau',tau.round(5),'weights',w.round(5),'curve |e| 50/90',np.quantile(e[:,:-1],[.5,.9]),'rate |e| 50/90',np.quantile(e[:,-1],[.5,.9]),'missing curve median',np.median(e[miss,:-1]) if miss.any() else None)
print('elapsed',round(time.monotonic()-start,2),'max past',abs(p).max())
PY
```
- **Container Exit Code**: `0` (`887 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
-1 tau [0.54858 3.76877] weights [ 0.69545  0.54948  0.13887  0.16791 -0.13821  0.11766 -0.08904] curve |e| 50/90 [0.01819879 0.21675986] rate |e| 50/90 [0.0169357  0.04395305] missing curve median 0.21407556448769444
0 tau [0.54513 3.73665] weights [ 0.68934  0.55805  0.13901  0.16625 -0.1361   0.11608 -0.08714] curve |e| 50/90 [0.02491784 0.13906973] rate |e| 50/90 [0.01528622 0.03492897] missing curve median 0.09350659216363902
1 tau [0.54866 3.68488] weights [ 0.69428  0.54524  0.13916  0.16949 -0.14003  0.11905 -0.08999] curve |e| 50/90 [0.02475513 0.28353131] rate |e| 50/90 [0.01823437 0.04944786] missing curve median 0.2329564723679774
2 tau [0.551   3.81664] weights [ 0.69979  0.54612  0.13912  0.16587 -0.13666  0.11666 -0.08844] curve |e| 50/90 [0.01830641 0.31135751] rate |e| 50/90 [0.01755266 0.03417584] missing curve median 0.44047789876707566
3 tau [0.54777 3.72874] weights [ 0.69236  0.55431  0.13937  0.16908 -0.13955  0.11882 -0.09015] curve |e| 50/90 [0.02408239 0.19665287] rate |e| 50/90 [0.02058264 0.04410642] missing curve median 0.4172182464728453
elapsed 0.2 max past 2.06707077345052

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c7f1a22823e68ac8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c7f1a228 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`30042 ms` LLM)
- **Rationale**: Check both prediction outputs against the literal recurrence to catch indexing or derivative errors, and use public truth where available to determine whether the missing-coupon error reflects a solver mismatch or the noisy observation target.
- **Alternatives Weighed**: `Change the imputation based on noisy training residuals` (rejected: The missing-coupon residual alone cannot distinguish the cost of missing data from an error in the specified preprocessing.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'alpha_m97_c01_s00_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('public_alpha_m97_c01_s00.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, gt = item['dataset'], item['gt']
out = m.solve_alpha_m97_c01_s00(d)
print('public ground truth fields:', {k: np.shape(v) for k, v in gt.items()})
for k, estimate in out.items():
    if k in gt and np.shape(gt[k]) == estimate.shape:
        err = np.abs(estimate - gt[k]); print(k, 'MAE', err.mean(), 'max AE', err.max())
        if k in ('predicted_strain', 'initial_rate_pred'):
            missing = np.isnan(d['test']['coupon_features_raw']).any(axis=1)
            for c in range(4):
                for miss in (False, True):
                    sel = (d['test']['lab_id'] == c) & (missing == miss)
                    if sel.any(): print('lab', c, 'missing', miss, 'MAE', err[sel].mean())
past, hold, x, base = m._inputs(d)
n = len(d['train']['lab_id']); past, hold, x, base = past[n:], hold[n:], x[n:], base[n:]
tau, w = out['estimated_taus'], out['hereditary_weights']; t = d['eval_times']
z = np.zeros((len(hold), 2))
for s, dt in enumerate(d['past_dt']):
    decay = np.exp(-dt / tau)
    z = decay * z + (1 - decay) * past[:, s, None]
amplitude = (hold[:, None] - z) * w[:2]
flow = w[2] + x @ w[3:]
np.testing.assert_allclose(out['predicted_strain'], base[:, None] + amplitude @ (1 - np.exp(-t[None, :] / tau[:, None])) + flow[:, None] * t, atol=1e-12)
np.testing.assert_allclose(out['initial_rate_pred'], amplitude @ (1 / tau) + flow, atol=1e-12)
np.testing.assert_array_equal(out['regime_mode_weights'], [0., 0.])
print('stepwise recurrence, initial rate, and zero regime weights verified')
PY
```
- **Container Exit Code**: `0` (`542 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public ground truth fields: {'taus': (2,), 'theta': (7,), 'vp_theta': (2,), 'stress_gains': (4,), 'stress_biases': (4,), 'te_clean': (72, 10), 'te_true_rate0': (72,), 'te_hist_true': (72, 18), 'te_hold_true': (72,)}
stepwise recurrence, initial rate, and zero regime weights verified

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c7f1a22823e68ac8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c7f1a228 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`24742 ms` LLM)
- **Rationale**: The previous check verified the recurrence but skipped clean comparisons because the truth keys differ from the output names. Compare the correct fields and evaluate predictions with the public physical parameters and stresses to isolate any alignment or imputation mismatch.
- **Alternatives Weighed**: `Retune the robust optimizer now` (rejected: The observed missing-coupon errors could come from preprocessing rather than parameter fitting; the public clean targets distinguish the two.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
s = importlib.util.spec_from_file_location('solver', 'alpha_m97_c01_s00_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('public_alpha_m97_c01_s00.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, g = item['dataset'], item['gt']; o = m.solve_alpha_m97_c01_s00(d)
for key, target in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]:
    print(key, 'signed errors', (o[key] - g[target]).round(6))
past, hold, x, base = m._inputs(d); n = len(d['train']['lab_id']); lab = d['test']['lab_id']
print('stress max absolute errors', np.max(abs(past[n:] - g['te_hist_true'])), np.max(abs(hold[n:] - g['te_hold_true'])))
A, _ = m._Model(d['past_dt'], d['eval_times'], g['te_hist_true'], g['te_hold_true'], x[n:]).design(np.log(g['taus']))
ref = A @ g['theta']
missing = np.isnan(d['test']['coupon_features_raw']).any(axis=1)
for key, truth, fixed in [('predicted_strain',g['te_clean'],base[n:,None]+ref[:,:-1]),('initial_rate_pred',g['te_true_rate0'],ref[:,-1])]:
    e, ef = abs(o[key]-truth), abs(fixed-truth)
    print(key, 'overall MAE', e.mean(), 'preprocessing-only MAE', ef.mean())
    for c in range(4):
        for mv in (False, True):
            mask = (lab==c) & (missing==mv)
            if mask.any(): print('  lab',c,'missing',mv,'n',mask.sum(),'MAE',e[mask].mean(),'preprocessing-only MAE',ef[mask].mean())
PY
```
- **Container Exit Code**: `0` (`600 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
estimated_taus signed errors [0.001585 0.142767]
hereditary_weights signed errors [ 0.005449 -0.000522 -0.001131 -0.002094  0.001792 -0.002341  0.000961]
regime_mode_weights signed errors [0. 0.]
stress max absolute errors 7.100918360203323e-05 6.791443310127931e-05
predicted_strain overall MAE 0.04475267321817636 preprocessing-only MAE 0.044288136212486444
  lab 0 missing False n 16 MAE 0.017596827495481537 preprocessing-only MAE 0.009056648080997489
  lab 0 missing True n 2 MAE 0.33284041073847387 preprocessing-only MAE 0.32256530665098515
  lab 1 missing False n 15 MAE 0.011803414942604637 preprocessing-only MAE 0.0192293419242893
  lab 1 missing True n 1 MAE 0.3312965548548764 preprocessing-only MAE 0.3076575625256136
  lab 2 missing False n 18 MAE 0.007898072096420168 preprocessing-only MAE 0.0036354115835704735
  lab 2 missing True n 3 MAE 0.1560853356094804 preprocessing-only MAE 0.15675388581946456
  lab 3 missing False n 16 MAE 0.023608526668861696 preprocessing-only MAE 0.031026826887930115
  lab 3 missing True n 1 MAE 0.7784569000443075 preprocessing-only MAE 0.7704828371415967
initial_rate_pred overall MAE 0.01656533008740001 preprocessing-only MAE 0.01610477680454043
  lab 0 missing False n 16 MAE 0.007147911487209543 preprocessing-only MAE 0.003293326574907947
  lab 0 missing True n 2 MAE 0.1194840451763608 preprocessing-only MAE 0.11729647514581276
  lab 1 missing False n 15 MAE 0.004502694098273915 preprocessing-only MAE 0.006992487972468864
  lab 1 missing True n 1 MAE 0.11703985975584486 preprocessing-only MAE 0.11187547728204139
  lab 2 missing False n 18 MAE 0.003578806980535328 preprocessing-only MAE 0.001321967848570933
  lab 2 missing True n 3 MAE 0.05628876814999203 preprocessing-only MAE 0.05700141302525966
  lab 3 missing False n 16 MAE 0.00878830114111842 preprocessing-only MAE 0.011282482504701812
  lab 3 missing True n 1 MAE 0.28089117255726626 preprocessing-only MAE 0.28017557714239916

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c7f1a22823e68ac8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c7f1a228 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`28238 ms` LLM)
- **Rationale**: Compare all-row and complete-row fitting on held-out labs to test whether imputed training features degrade parameter estimates. Keep the verified baseline unless exclusion improves the held-out bottleneck without materially worsening the other slices.
- **Alternatives Weighed**: `Change imputation to reduce public clean-target errors` (rejected: The public truth diagnostic implicates missing-feature information, but does not justify replacing the specified cell-balanced imputation.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
sp = importlib.util.spec_from_file_location('solver', 'alpha_m97_c01_s00_solver.py')
s = importlib.util.module_from_spec(sp); sp.loader.exec_module(s)
with open('public_alpha_m97_c01_s00.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
a = d['train']; lab = a['lab_id']; n = len(lab)
complete = np.isfinite(a['coupon_features_raw']).all(axis=1)
p,h,x,b = s._inputs(d); p,h,x = p[:n],h[:n],x[:n]
y = np.column_stack((a['strain_curves_obs'] - b[:n,None], a['instant_rate_obs']))
def model(mask):
    return s._Model(d['past_dt'], d['eval_times'], p[mask], h[mask], x[mask])
for exclude_missing in (False, True):
    worst_complete = np.zeros(2); worst_missing = np.zeros(2); taus = []
    for c in range(4):
        fit = (lab != c) & (complete if exclude_missing else True)
        tau, w = s._fit(model(fit), y[fit]); taus.append(tau)
        for subset, worst in ((complete,worst_complete),(~complete,worst_missing)):
            val = (lab == c) & subset
            A,_ = model(val).design(np.log(tau)); r = A@w-y[val]
            worst[:] = np.maximum(worst, [np.median(abs(r[:,:-1])), np.median(abs(r[:,-1]))])
    print('fit complete only', exclude_missing, 'worst complete strain/rate medAE', worst_complete, 'worst missing', worst_missing, 'tau ranges', np.min(taus,axis=0), np.max(taus,axis=0))
print('past peak and critical stress', np.max(np.abs(p)), d['crit_stress'], 'banding', np.max(d['shear_banding_index']))
PY
```
- **Container Exit Code**: `0` (`806 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
fit complete only False worst complete strain/rate medAE [0.02235529 0.02002692] worst missing [0.4404779  0.17661908] tau ranges [0.54512898 3.68487558] [0.55099617 3.81664053]
fit complete only True worst complete strain/rate medAE [0.02317518 0.01987908] worst missing [0.44064313 0.17543531] tau ranges [0.54559357 3.69071672] [0.55117113 3.84984313]
past peak and critical stress 2.06707077345052 2.65 banding 0.12

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c7f1a22823e68ac8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c7f1a228 (directory).
```

### Turn 7 — `submit_completed_task` (`18477 ms` LLM)
- **Rationale**: The verified artifact matches the specified recurrence, initial-rate equation, and zero regime weights. Even the true physical parameters left nearly the same missing-coupon error under the prescribed imputation, so the data do not support further optimizer tuning.
- **Alternatives Weighed**: `Fit only on coupons with complete features` (rejected: Held-out labs showed no clear improvement: the worst complete-coupon strain error rose slightly and the worst missing-coupon strain error was effectively unchanged.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_s00_solver.py': 'FILE:7584'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_m97_c01_s00_solver.py`. The public schema and both prediction equations passed isolated verification. Public test MAE was 0.0448 for strain and 0.0166 for initial rate. Missing coupon features were the main source of error.`

### Turn 8 — `execute_in_remote_docker` (`32249 ms` LLM)
- **Rationale**: Test whether nonlinear corrections learned without the held-out lab reduce its residuals consistently. Separately, perturb the measured calibration rows with a supported EMI glitch; a material stress error would justify repairing the spike screen before final verification.
- **Alternatives Weighed**: `Apply relay-mode, PDE, or classification checks` (rejected: This is continuous rheology regression with no relay state, PDE drift, or class labels. Held-out lab-by-stratum residuals test for the relevant nuisance effects.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('solver','alpha_m97_c01_s00_solver.py'); s=importlib.util.module_from_spec(sp); sp.loader.exec_module(s)
with open('public_alpha_m97_c01_s00.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
a=d['train']; lab=a['lab_id']; st=a['stratum']; n=len(lab); complete=np.isfinite(a['coupon_features_raw']).all(axis=1)
p,h,x,b=s._inputs(d); p,h,x=p[:n],h[:n],x[:n]; dt=d['past_dt']; t=np.array(d['eval_times']); y=np.column_stack((a['strain_curves_obs']-b[:n,None],a['instant_rate_obs']))
for c in np.unique(lab):
    train=lab!=c; fit=train & complete; val=~train & complete
    tau,w=s._fit(s._Model(dt,t,p[train],h[train],x[train]),y[train]); A,_=s._Model(dt,t,p,h,x).design(np.log(tau)); base=A@w; r=y-base
    shape=(t/t.max())**2; drift=w[2]+x@w[3:]; q=h/float(d['crit_stress']); pair=x[:,0]-x[:,1]
    features=np.column_stack((np.ones(n),drift/(1+q*q),pair,pair*x[:,2]*x[:,3])); Z=features[:,None,:]*shape[None,:,None]
    opt=least_squares(lambda beta:(Z[fit]@beta-r[fit,:-1]).ravel(), np.zeros(Z.shape[-1]),loss='cauchy',f_scale=1.25*s._mad(r[fit,:-1].ravel()))
    rr=r[:,:-1]-Z@opt.x
    for g in np.unique(st[val]):
        m=val & (st==g); print('lab/stratum',c,g,'n',m.sum(),'strain medAE exact/extra',round(np.median(abs(r[m,:-1])),5),round(np.median(abs(rr[m])),5),'signed strain/rate medians',round(np.median(r[m,:-1]),5),round(np.median(r[m,-1]),5))
rows=d['calibration_load_cells']; labcal=np.asarray(rows['lab_id']); true=np.asarray(rows['true_stress']); ref=s._calibrate(d)
for c in range(4):
    inds=np.flatnonzero(labcal==c); err=[]
    for i in inds:
        for spike in (-6.,6.):
            corrupt=true.copy(); corrupt[i]+=spike; ds=dict(d,calibration_load_cells=dict(rows,true_stress=corrupt)); new=s._calibrate(ds)[c]; e=new-ref[c]; err.append(np.max(np.abs(e[0]*p[lab==c]+e[1])))
    print('calibration lab',c,'points',len(inds),'worst coefficient-induced deviation under one +/-6 EMI spike',max(err,default=0))
PY
```
- **Container Exit Code**: `0` (`853 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lab/stratum 0 0 n 14 strain medAE exact/extra 0.0228 0.02487 signed strain/rate medians -0.0073 0.00542
lab/stratum 0 1 n 18 strain medAE exact/extra 0.02102 0.02125 signed strain/rate medians 0.00266 0.00472
lab/stratum 0 2 n 10 strain medAE exact/extra 0.02581 0.02479 signed strain/rate medians -0.00528 -0.00379
lab/stratum 1 0 n 10 strain medAE exact/extra 0.01891 0.0189 signed strain/rate medians 0.00968 -0.00665
lab/stratum 1 1 n 18 strain medAE exact/extra 0.02338 0.02382 signed strain/rate medians -0.00163 0.0016
lab/stratum 1 2 n 10 strain medAE exact/extra 0.02084 0.02048 signed strain/rate medians -0.00111 -0.01013
lab/stratum 2 0 n 16 strain medAE exact/extra 0.01745 0.01934 signed strain/rate medians -0.00559 0.00482
lab/stratum 2 1 n 10 strain medAE exact/extra 0.01342 0.01556 signed strain/rate medians -0.00493 -0.01004
lab/stratum 2 2 n 17 strain medAE exact/extra 0.01679 0.0178 signed strain/rate medians 0.00495 -0.00563
lab/stratum 3 0 n 13 strain medAE exact/extra 0.0181 0.01766 signed strain/rate medians -0.00913 -0.00082
lab/stratum 3 1 n 14 strain medAE exact/extra 0.02365 0.02426 signed strain/rate medians -0.00761 0.00519
lab/stratum 3 2 n 10 strain medAE exact/extra 0.02032 0.02131 signed strain/rate medians 0.00671 0.01834
calibration lab 0 points 10 worst coefficient-induced deviation under one +/-6 EMI spike 4.7988941287287605e-05
calibration lab 1 points 10 worst coefficient-induced deviation under one +/-6 EMI spike 5.641617654017615e-05
calibration lab 2 points 10 worst coefficient-induced deviation under one +/-6 EMI spike 4.451623503425155e-05
calibration lab 3 points 10 worst coefficient-induced deviation under one +/-6 EMI spike 8.595458443658483e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c7f1a22823e68ac8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c7f1a228 (directory).
```

### Turn 9 — `submit_completed_task` (`15198 ms` LLM)
- **Rationale**: The audit found no consistent lab-by-stratum residual bias that justified changing the model. Injecting one EMI spike into the measured calibration rows changed fitted stresses by at most 8.6e-5, so no calibration change was warranted.
- **Alternatives Weighed**: `Add nonlinear residual terms` (rejected: They did not consistently improve held-out lab-by-stratum errors.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_s00_solver.py': 'FILE:7584'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_s00_solver.py`. The public schema, memory recurrence, and initial-rate equation passed isolated verification. Public test MAE was 0.0448 for strain and 0.0166 for initial rate. Missing coupon features accounted for most of the remaining error.`

### Turn 10 — `execute_in_remote_docker` (`24909 ms` LLM)
- **Rationale**: The prior EMI test used spikes twice the specified cutoff. A spike just above 3.0 could survive if the consensus fit shifts toward it. Test that boundary on the measured calibration rows and stress commands; a material deviation would warrant tightening the spike screen.
- **Alternatives Weighed**: `Repeat nonlinear residual and lab-stratum tests` (rejected: The previous audit found no consistent held-out improvement; repeating it would not test the unmeasured margin at the EMI threshold.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
sp = importlib.util.spec_from_file_location('solver', 'alpha_m97_c01_s00_solver.py')
s = importlib.util.module_from_spec(sp); sp.loader.exec_module(s)
with open('public_alpha_m97_c01_s00.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
rows = d['calibration_load_cells']; y = np.asarray(rows['true_stress']); labs = np.asarray(rows['lab_id']); baseline = s._calibrate(d)
for c in range(4):
    cmds = np.concatenate([np.r_[a['past_stress_cmd'][a['lab_id']==c].ravel(), a['stress_hold_cmd'][a['lab_id']==c]] for a in (d['train'],d['test'])])
    for magnitude in (3.0001, 3.001, 3.01, 3.1):
        worst = (0., -1, 0.)
        for idx in np.flatnonzero(labs == c):
            for direction in (-1., 1.):
                yy = y.copy(); yy[idx] += direction * magnitude
                ds = dict(d, calibration_load_cells=dict(rows, true_stress=yy))
                delta = s._calibrate(ds)[c] - baseline[c]
                worst = max(worst, (float(np.max(abs(delta[0]*cmds+delta[1]))), int(idx), direction))
        print('channel', c, 'spike magnitude', magnitude, 'worst stress deviation / row / sign', worst)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`1654 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
channel 0 spike magnitude 3.0001 worst stress deviation / row / sign (1.2346795508037318, 5, -1.0)
channel 0 spike magnitude 3.001 worst stress deviation / row / sign (1.2350499423225885, 5, -1.0)
channel 0 spike magnitude 3.01 worst stress deviation / row / sign (1.2387538575111599, 5, -1.0)
channel 0 spike magnitude 3.1 worst stress deviation / row / sign (0.8119395034065247, 8, 1.0)
channel 1 spike magnitude 3.0001 worst stress deviation / row / sign (1.224029885681658, 16, -1.0)
channel 1 spike magnitude 3.001 worst stress deviation / row / sign (1.2243970824074717, 16, -1.0)
channel 1 spike magnitude 3.01 worst stress deviation / row / sign (1.2280690496656077, 16, -1.0)
channel 1 spike magnitude 3.1 worst stress deviation / row / sign (1.264788722246972, 16, -1.0)
channel 2 spike magnitude 3.0001 worst stress deviation / row / sign (1.2471579990872645, 20, 1.0)
channel 2 spike magnitude 3.001 worst stress deviation / row / sign (1.247532134015826, 20, 1.0)
channel 2 spike magnitude 3.01 worst stress deviation / row / sign (1.251273483301445, 20, 1.0)
channel 2 spike magnitude 3.1 worst stress deviation / row / sign (1.2886869761576345, 20, 1.0)
channel 3 spike magnitude 3.0001 worst stress deviation / row / sign (1.1814787337291308, 30, 1.0)
channel 3 spike magnitude 3.001 worst stress deviation / row / sign (1.1818331655348562, 30, 1.0)
channel 3 spike magnitude 3.01 worst stress deviation / row / sign (1.1853774835921091, 30, 1.0)
channel 3 spike magnitude 3.1 worst stress deviation / row / sign (1.2208206641646302, 30, 1.0)
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c7f1a22823e68ac8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c7f1a228 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`54535 ms` LLM)
- **Rationale**: The supported EMI stress test exposed a large calibration error despite a passing schema. Change only the calibration screen, then verify on the exact artifact that clean fits are unchanged and near-cutoff spikes no longer materially shift physical stress.
- **Alternatives Weighed**: `Lower the inlier-count cutoff alone` (rejected: It moves the masking boundary without separating estimation of the clean line from spike detection.)
- **ATMS NOGOOD DECLARED**: Refuted `Maximizing the number of calibration rows within the 3.0 cutoff identifies the clean affine relation.` | Witness: `On the measured calibration grid, a single 3.0001 spike caused stress deviations above 1.18 in every channel, and a 3.1 spike caused a deviation of 1.289 in channel 2.` | Blanket Diagnosis: `A line tilted toward the corrupted row can keep all rows inside the cutoff, beating the clean line on inlier count and biasing converted stresses.` | Sublated Pivot: `Estimate the tight majority line first, then apply the EMI cutoff to residuals around that line.`

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_s00_solver.py')
s = p.read_text()
old = '''        clean = np.ones(x.size, dtype=bool)
        if len(i):
            slopes = (y[i] - y[j]) / (x[i] - x[j])
            # Use the line with greatest clean support to screen isolated EMI.
            lines = y[i, None] + slopes[:, None] * (x - x[i, None])
            err = np.abs(lines - y)
            best = min(range(len(i)), key=lambda k: (-np.sum(err[k] <= 3), np.median(err[k])))
            clean = err[best] <= 3.0
        if (len(x) < 2 or not clean.all()) and archive_path.is_file():'''
new = '''        clean = np.ones(x.size, dtype=bool)
        emi = False
        if len(i):
            slopes = (y[i] - y[j]) / (x[i] - x[j])
            lines = y[i, None] + slopes[:, None] * (x - x[i, None])
            err = np.abs(lines - y)
            # Estimate the affine relation from the tight majority first. A
            # count at the EMI cutoff alone favors lines tilted toward a spike.
            typical = np.median(err, axis=1)
            best = np.argmin(typical)
            core = err[best] <= 8 * typical[best] + 1e-12
            design = np.column_stack((x, np.ones(x.size)))
            provisional = np.linalg.lstsq(design[core], y[core], rcond=None)[0]
            residual = np.abs(design @ provisional - y)
            # Allow for calibration noise in deciding which near-cutoff rows
            # to exclude, without relaxing the trigger for archive lookup.
            margin = min(0.01, 8 * np.median(residual[core]))
            clean = residual <= 3.0 - margin
            emi = np.any(residual > 3.0)
        if (len(x) < 2 or emi) and clean.sum() < 4 and archive_path.is_file():'''
assert s.count(old) == 1
p.write_text(s.replace(old,new))
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
sp = importlib.util.spec_from_file_location('solver', 'alpha_m97_c01_s00_solver.py')
s = importlib.util.module_from_spec(sp); sp.loader.exec_module(s)
with open('public_alpha_m97_c01_s00.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, g = item['dataset'], item['gt']
rows = d['calibration_load_cells']; y = np.asarray(rows['true_stress']); labs = np.asarray(rows['lab_id']); x = np.asarray(rows['cmd_stress']); baseline = s._calibrate(d)
for c in range(4):
    m = labs == c
    np.testing.assert_allclose(baseline[c], np.polyfit(x[m], y[m], 1), atol=1e-12, rtol=0)
    cmds = np.concatenate([np.r_[a['past_stress_cmd'][a['lab_id']==c].ravel(), a['stress_hold_cmd'][a['lab_id']==c]] for a in (d['train'],d['test'])])
    for magnitude in (3.0001, 3.001, 3.01, 3.1, 6.):
        worst = 0.
        for idx in np.flatnonzero(m):
            for sign in (-1., 1.):
                yy = y.copy(); yy[idx] += sign * magnitude
                ds = dict(d, calibration_load_cells=dict(rows, true_stress=yy))
                delta = s._calibrate(ds)[c] - baseline[c]
                worst = max(worst, np.max(abs(delta[0]*cmds+delta[1])))
        print('channel', c, 'EMI', magnitude, 'worst stress deviation', worst)
        assert worst < 0.001
out = s.solve_alpha_m97_c01_s00(d)
for k, t in [('predicted_strain','te_clean'), ('initial_rate_pred','te_true_rate0')]:
    print(k, 'public MAE', np.mean(abs(out[k]-g[t])))
PY
```
- **Container Exit Code**: `0` (`1117 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
channel 0 EMI 3.0001 worst stress deviation 4.798940577466397e-05
channel 0 EMI 3.001 worst stress deviation 4.798940577466397e-05
channel 0 EMI 3.01 worst stress deviation 4.798940577466397e-05
channel 0 EMI 3.1 worst stress deviation 4.798940577466397e-05
channel 0 EMI 6.0 worst stress deviation 4.798940577466397e-05
channel 1 EMI 3.0001 worst stress deviation 5.641636027710594e-05
channel 1 EMI 3.001 worst stress deviation 5.641636027710594e-05
channel 1 EMI 3.01 worst stress deviation 5.641636027710594e-05
channel 1 EMI 3.1 worst stress deviation 5.641636027710594e-05
channel 1 EMI 6.0 worst stress deviation 5.641636027710594e-05
channel 2 EMI 3.0001 worst stress deviation 4.451649640737445e-05
channel 2 EMI 3.001 worst stress deviation 4.451649640737445e-05
channel 2 EMI 3.01 worst stress deviation 4.451649640737445e-05
channel 2 EMI 3.1 worst stress deviation 4.451649640737445e-05
channel 2 EMI 6.0 worst stress deviation 4.451649640737445e-05
channel 3 EMI 3.0001 worst stress deviation 8.59557080721055e-05
channel 3 EMI 3.001 worst stress deviation 8.59557080721055e-05
channel 3 EMI 3.01 worst stress deviation 8.59557080721055e-05
channel 3 EMI 3.1 worst stress deviation 8.59557080721055e-05
channel 3 EMI 6.0 worst stress deviation 8.59557080721055e-05
predicted_strain public MAE 0.04475267321817636
initial_rate_pred public MAE 0.01656533008740001

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3c71b630af7f3725 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_3c71b630 (directory).
```

### Turn 12 — `submit_completed_task` (`19331 ms` LLM)
- **Rationale**: The final artifact passed isolated schema verification and single-spike tests at and above the critical margin, starting at magnitude 3.0001. Clean calibration still agrees with the direct affine fit, and both public prediction errors are unchanged.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_s00_solver.py`. The public schema passed. The revised calibration reduced the worst stress deviation under a single EMI spike from 1.289 to 0.000086 without changing clean-data predictions. Public test MAE was 0.0448 for strain and 0.0166 for initial rate. Missing coupon features were the main source of remaining error.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M97-NonNested-Counterfactual-Lineage-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
