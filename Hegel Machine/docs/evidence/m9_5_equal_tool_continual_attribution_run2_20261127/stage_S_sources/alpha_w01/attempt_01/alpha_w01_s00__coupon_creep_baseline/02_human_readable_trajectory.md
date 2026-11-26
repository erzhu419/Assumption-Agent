# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_w01_s00__coupon_creep_baseline`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M95-Equal-Tool-Continual-World` |
| **Domain** | `Multi-Wing Polymer Coupon Hereditary Creep & Continual Calibration Stream` |
| **Wave / Wormhole Family** | `alpha_w01_source` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `361,275` (`346,780` / `94,464` / `14,495`) |
| **Total Episode Wall Time (s)** | `472.34s` |

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

### Turn 1 — `execute_in_remote_docker` (`15911 ms` LLM)
- **Rationale**: Inspect the public schema and small data summaries to catch field-layout assumptions before implementation; archive and primitive availability determine whether calibration or fitting can reuse existing evidence or code.

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && python3 - <<'PY'
import pickle, numpy as np
with open('public_alpha_w01_s00.pkl','rb') as f: d=pickle.load(f)
def show(o, path='dataset', depth=0):
 if isinstance(o,dict):
  for k,v in o.items(): show(v,path+'.'+str(k),depth+1)
 elif isinstance(o,np.ndarray):
  print(path,o.shape,o.dtype, 'sample',repr(o.reshape(-1)[:8]))
 elif isinstance(o,(list,tuple)):
  print(path,type(o).__name__,len(o));
  if o and depth<5: show(o[0],path+'[0]',depth+1)
 else: print(path,str(o)[:200])
show(d)
PY
printf '\nContinual archive:\n'; if [ -d /tmp/hegel_continual_archive ]; then find /tmp/hegel_continual_archive -maxdepth 3 -type f -printf '%p %s bytes\n' | head -40; fi
printf '\nPrimitive search:\n'; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation Cauchy robust IRLS reference-panel alignment imputation'
```
- **Container Exit Code**: `0` (`628 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
le)
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
```

### Turn 2 — `execute_in_remote_docker` (`18953 ms` LLM)
- **Rationale**: Inspect only solver-visible fields. Reference-panel shapes determine the alignment calculation, and the affine residuals distinguish exact checks from contaminated checks before choosing a calibration estimator.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle, numpy as np
rows=pickle.load(open('public_alpha_w01_s00.pkl','rb')); print('case count',len(rows)); d=rows[0]['dataset']; print('keys',list(d));
for part,v in d.items():
 if isinstance(v,dict):
  print(part,[(k,np.shape(x),str(getattr(x,'dtype',type(x).__name__))) for k,x in v.items()]);
  if 'lab_id' in v: print('labs',np.unique(v['lab_id'],return_counts=True))
  for k,x in v.items():
   a=np.asarray(x)
   if a.dtype.kind=='f': print(' ',k,'missing',np.isnan(a).sum(),'quantiles',np.nanquantile(a,[0,.01,.5,.99,1]))
 else: print(part,repr(v)[:320])
c=d['calibration_load_cells'];
for lab in np.unique(c['lab_id']):
 i=c['lab_id']==lab; x=c['cmd_stress'][i]; y=c['true_stress'][i]; p=np.polyfit(x,y,1); print('lab affine',lab,p,'residuals',np.round(y-np.polyval(p,x),7))
PY
for primitive in autogenous_prim_robust_affine_projection.robust_residual_weights autogenous_prim_reference_panel_balancing.align_reference_panel autogenous_prim_reference_panel_balancing.balanced_cell_weights autogenous_prim_reference_panel_balancing.weighted_column_fill autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation; do PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe "$primitive"; done
```
- **Container Exit Code**: `0` (`680 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
fit_anchored_relaxation": "APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)",
        "fit_local_level_step": "APPLICABLE_SUBROUTINE(real_1d_step_series)"
      },
      "functions": [
        {
          "function": "fit_anchored_relaxation",
          "signature": "fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None]",
          "doc_summary": "Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing nonnegative times, a predictor independent of the intercept, nonnegative rate and curvature ratio, positive noise scale, 2 <= bins <= n, and ordered positive tau bounds, returning (offset, positive slope,"
        }
      ],
      "index_cards": [
        "autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit",
        "autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation",
        "autogenous_prim_robust_trend_decomposition@1f45421a | fit_local_level_step(predictor: np.ndarray, values: np.ndarray, splits: np.ndarray, *, scale: float, loss_clip: float) -> tuple[int, float] | Fit finite predictor and values (n,) and integer splits (k,) with k >= 1, at l | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_local_level_step"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_robust_trend_decomposition",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`59225 ms` LLM)
- **Rationale**: Implement the exact recurrence and its analytic onset derivative, then check the public schema and rate residual scale. A large rate discrepancy would trigger a check of feature imputation and curve/rate weighting before held-out validation.
- **Alternatives Weighed**: `Use the single-series anchored-relaxation primitive` (rejected: It does not fit two shared relaxation times from signed, specimen-specific stress histories and joint curve/rate observations.)

```bash
cd /holdout_task && cat > alpha_w01_s00_solver.py <<'PY'
import numpy as np
from scipy.optimize import least_squares


def _load_cells(checks):
    """Fit a per-lab affine map, using pair consensus before a clean refit."""
    labs = np.asarray(checks['lab_id'])
    x_all = np.asarray(checks['cmd_stress'])
    y_all = np.asarray(checks['true_stress'])
    maps = np.tile([1., 0.], (4, 1))
    for lab in range(4):
        keep = (labs == lab) & np.isfinite(x_all) & np.isfinite(y_all)
        x, y = x_all[keep], y_all[keep]
        if not len(x):
            continue
        if np.ptp(x) < 1e-3:
            maps[lab, 1] = np.median(y - x)
            continue
        a, b = np.triu_indices(len(x), k=1)
        spread = np.abs(x[a] - x[b]) > .05 * np.ptp(x)
        a, b = a[spread], b[spread]
        slope = (y[b] - y[a]) / (x[b] - x[a])
        intercept = y[a] - slope * x[a]
        residual = np.abs(y - slope[:, None] * x - intercept[:, None])
        h = max(2, len(x) // 2 + 1)
        order = np.sort(residual, axis=1)[:, :h]
        best = np.argmin(np.sum(order**2, axis=1))
        err = residual[best]
        clean = err <= max(2e-4, 5 * np.median(err))
        A = np.column_stack((x[clean], np.ones(clean.sum())))
        maps[lab] = np.linalg.lstsq(A, y[clean], rcond=None)[0]
    return maps


def _alignment(train, test):
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    refs = np.concatenate((train['ref_coupons'], test['ref_coupons']))
    raw = np.concatenate((train['coupon_features_raw'], test['coupon_features_raw']))
    for lab in np.unique(labs):
        m = labs == lab
        panel = refs[m]
        cols = np.sum(np.isfinite(panel), axis=0) >= 2
        if np.any(cols):
            center = np.median(np.nanmean(panel[:, cols], axis=0))
            scale = np.median(np.nanstd(panel[:, cols], axis=0))
            raw[m] = (raw[m] - center) / (scale if scale > 0 else 1.)
    _, cell = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    finite = np.isfinite(raw)
    counts = np.zeros((int(cell.max()) + 1, raw.shape[1]))
    sums = np.zeros_like(counts)
    np.add.at(counts, cell, finite)
    np.add.at(sums, cell, np.where(finite, raw, 0.))
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    available = (counts > 0).sum(axis=0)
    fill = means.sum(axis=0) / np.maximum(1, available)
    x = np.where(finite, raw, fill)
    n = len(train['lab_id'])
    return x[:n], x[n:]


def _stress(block, maps):
    gain, bias = maps[np.asarray(block['lab_id'])].T
    return (block['past_stress_cmd'] * gain[:, None] + bias[:, None],
            block['stress_hold_cmd'] * gain + bias)


def _design(taus, hist, hold, cov, dt, time):
    # Integrate each constant-stress interval, then age its memory to hold onset.
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    q = dt[:, None] / taus
    memory = np.exp(-tail[:, None] / taus) * -np.expm1(-q)
    jump = hold[:, None] - hist @ memory
    drift = cov[:, None, :] * time[None, :, None]
    relax = jump[:, None, :] * -np.expm1(-time[:, None] / taus)
    return np.concatenate((relax, drift), axis=2), np.column_stack((jump / taus, cov))


def _mad(e):
    return max(1e-8, 1.4826 * np.median(np.abs(e - np.median(e))))


def _fit(train, x, maps, dt, time):
    hist, hold = _stress(train, maps)
    cov = np.column_stack((np.ones(len(x)), x))
    eps = train['strain_curves_obs'] - train['strain_0'][:, None]
    rate = train['instant_rate_obs']
    y = np.r_[eps.ravel(), rate]
    ncurve = eps.size

    def design(taus):
        a, b = _design(taus, hist, hold, cov, dt, time)
        return np.vstack((a.reshape(-1, 7), b))

    def unpack(p):
        return np.array([p[0], p[0] + (8. - p[0]) * p[1]])

    def residual(p):
        return design(unpack(p)) @ p[2:] - y

    bounds = (np.r_[.2, 1e-5, np.full(7, -np.inf)],
              np.r_[7.999, 1., np.full(7, np.inf)])
    starts = []
    for t1, t2 in ((.27, 1.2), (.43, 3.), (.65, 5.5), (1., 2.), (1.3, 6.), (2.5, 7.5)):
        A = design(np.array([t1, t2]))
        w = np.linalg.lstsq(A, y, rcond=None)[0]
        starts.append(np.r_[t1, (t2 - t1) / (8. - t1), w])
    prelim = [least_squares(residual, p, bounds=bounds, loss='soft_l1',
                             f_scale=.05, max_nfev=85) for p in starts]
    best = min(prelim, key=lambda opt: opt.cost)
    p = best.x
    for _ in range(2):
        e = residual(p)
        scale = np.r_[np.full(ncurve, 2.5 * _mad(e[:ncurve])),
                      np.full(len(rate), 2.5 * _mad(e[ncurve:]))]
        opt = least_squares(lambda q: residual(q) / scale, p, bounds=bounds,
                            loss='cauchy', max_nfev=100)
        p = opt.x
    return unpack(p), p[2:]


def solve_alpha_w01_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    maps = _load_cells(dataset['calibration_load_cells'])
    x, xt = _alignment(train, test)
    taus, w = _fit(train, x, maps, dt, time)
    hist, hold = _stress(test, maps)
    cov = np.column_stack((np.ones(len(xt)), xt))
    a, b = _design(taus, hist, hold, cov, dt, time)
    return {'estimated_taus': taus, 'hereditary_weights': w,
            'viscoplastic_weights': np.zeros(2),
            'predicted_strain': test['strain_0'][:, None] + a @ w,
            'initial_rate_pred': b @ w}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import alpha_w01_s00_solver as s
r=pickle.load(open('public_alpha_w01_s00.pkl','rb'))[0]; d=r['dataset']; o=s.solve_alpha_w01_s00(d); print('estimates',o['estimated_taus'],o['hereditary_weights']); print('calibration',s._load_cells(d['calibration_load_cells'])); print('reference evaluation fields',[(k,np.shape(v)) for k,v in r.get('gt',{}).items()]); e=o['initial_rate_pred']-d['test']['instant_rate_obs']; print('rate vs observed, abs error quantiles',np.quantile(abs(e),[.5,.8,.95,1.])); print('strata',np.unique(d['train']['stratum'],return_counts=True))
PY
```
- **Container Exit Code**: `0` (`1085 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
estimates [0.52096426 3.68735087] [ 0.66077145  0.53286559  0.08844201  0.15999403 -0.13034869  0.10952929
 -0.07999408]
calibration [[ 1.00001670e+00 -1.59482473e-05]
 [ 9.99989283e-01 -3.10852473e-05]
 [ 9.99979561e-01  2.19208943e-05]
 [ 1.00000541e+00  1.52358347e-05]]
reference evaluation fields [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (80, 10)), ('te_true_rate0', (80,)), ('te_cross', (80,)), ('te_secant', (80,)), ('te_hold_true', (80,)), ('te_hist_true', (80, 18))]
rate vs observed, abs error quantiles [0.00759842 0.01291702 0.06507329 0.12290318]
strata (array([0, 1, 2]), array([52, 77, 71]))

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e3d30e4acaa40ba9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e3d30e4a (directory).
```

### Turn 4 — `execute_in_remote_docker` (`26310 ms` LLM)
- **Rationale**: Use clean public targets for evaluation only and refit without each lab’s response labels to detect lab-specific transfer failure. Separate missing-feature and early/late errors to distinguish imputation problems from relaxation-model problems. Check archive availability to determine whether sparse-calibration fallback can be tested against real prior checks.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle
import numpy as np
import alpha_w01_s00_solver as s
row = pickle.load(open('public_alpha_w01_s00.pkl', 'rb'))[0]
d, gt = row['dataset'], row['gt']
tr, te = d['train'], d['test']
o = s.solve_alpha_w01_s00(d)
def metrics(a):
    z = np.abs(a)
    return f'MAE={z.mean():.5f} med={np.median(z):.5f} p95={np.quantile(z,.95):.5f}'
e = o['predicted_strain'] - gt['te_clean']
r = o['initial_rate_pred'] - gt['te_true_rate0']
print('public clean:', metrics(e), 'rate:', metrics(r), 'observed-rate error:', metrics(te['instant_rate_obs']-gt['te_true_rate0']))
print('parameter absolute error:', o['estimated_taus']-gt['taus'], o['hereditary_weights']-gt['theta'])
for key in ('lab_id', 'stratum'):
    for val in np.unique(te[key]):
        m = te[key] == val
        print(key,val,'curve',metrics(e[m]),'rate',metrics(r[m]))
for j in (0, len(d['eval_times'])-1):
    print('time',d['eval_times'][j],metrics(e[:,j]))
m = np.any(~np.isfinite(te['coupon_features_raw']), axis=1)
for flag in (False, True):
    print('features missing',flag,metrics(e[m==flag]),metrics(r[m==flag]))
for lab in np.unique(tr['lab_id']):
    m = tr['lab_id']==lab
    a = {k:v[~m] for k,v in tr.items()}
    b = {k:v[m] for k,v in tr.items() if k != 'strain_curves_obs'}
    p = s.solve_alpha_w01_s00(dict(d, train=a, test=b))
    f = p['predicted_strain'] - tr['strain_curves_obs'][m]
    g = p['initial_rate_pred'] - tr['instant_rate_obs'][m]
    print('leave lab out',lab,'taus',p['estimated_taus'],'curve',metrics(f),'rate',metrics(g),'early/late median',np.median(abs(f[:,0])),np.median(abs(f[:,-1])))
PY
if [ -d /tmp/hegel_continual_archive ]; then find /tmp/hegel_continual_archive -maxdepth 3 -printf '%y %p\n' | head -40; else echo 'Continual archive not mounted'; fi
```
- **Container Exit Code**: `0` (`1241 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public clean: MAE=0.02440 med=0.00417 p95=0.09625 rate: MAE=0.00887 med=0.00192 p95=0.07232 observed-rate error: MAE=0.00675 med=0.00548 p95=0.01691
parameter absolute error: [-0.00103574 -0.09364913] [-2.22855469e-03 -5.13440852e-03 -1.55799394e-03 -5.96609403e-06
 -3.48687496e-04 -4.70714562e-04  5.92318185e-06]
lab_id 0 curve MAE=0.03331 med=0.01236 p95=0.12188 rate MAE=0.01192 med=0.00531 p95=0.04290
lab_id 1 curve MAE=0.02925 med=0.00351 p95=0.23425 rate MAE=0.01033 med=0.00130 p95=0.07446
lab_id 2 curve MAE=0.02788 med=0.00244 p95=0.22095 rate MAE=0.01061 med=0.00123 p95=0.09396
lab_id 3 curve MAE=0.01036 med=0.00610 p95=0.03356 rate MAE=0.00366 med=0.00355 p95=0.00880
stratum 0 curve MAE=0.03035 med=0.00700 p95=0.16177 rate MAE=0.01102 med=0.00322 p95=0.06034
stratum 1 curve MAE=0.01560 med=0.00290 p95=0.03691 rate MAE=0.00554 med=0.00126 p95=0.00917
stratum 2 curve MAE=0.02903 med=0.00389 p95=0.20657 rate MAE=0.01072 med=0.00188 p95=0.08144
time 0.25 MAE=0.00221 med=0.00045 p95=0.01812
time 5.2 MAE=0.04686 med=0.00938 p95=0.37686
features missing False MAE=0.00773 med=0.00372 p95=0.02976 MAE=0.00274 med=0.00165 p95=0.00783
features missing True MAE=0.27447 med=0.26546 p95=0.54793 MAE=0.10083 med=0.10694 p95=0.11490
leave lab out 0 taus [0.52259252 3.77125058] curve MAE=0.11956 med=0.00930 p95=0.93299 rate MAE=0.01195 med=0.00714 p95=0.02061 early/late median 0.007271442072651152 0.012941464215392995
leave lab out 1 taus [0.52143167 3.70965233] curve MAE=0.10881 med=0.01444 p95=0.92494 rate MAE=0.01629 med=0.00902 p95=0.03251 early/late median 0.009668569955528165 0.028514575577858303
leave lab out 2 taus [0.52184838 3.75802567] curve MAE=0.09092 med=0.00799 p95=0.84331 rate MAE=0.01332 med=0.00860 p95=0.03058 early/late median 0.00838821190114572 0.0063705642836108245
leave lab out 3 taus [0.52221609 3.7990445 ] curve MAE=0.09630 med=0.01563 p95=0.87571 rate MAE=0.01429 med=0.00905 p95=0.03194 early/late median 0.006648213262717868 0.023798030418393762
Continual archive not mounted

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e3d30e4acaa40ba9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e3d30e4a (directory).
```

### Turn 5 — `execute_in_remote_docker` (`33084 ms` LLM)
- **Rationale**: Compare the algebraic correction and complete-case parameter fitting without editing the artifact. Use clean public targets to judge the observed-rate correction, and use leave-one-lab-out curve residuals to check that it transfers across labs; held-out rate residuals are not independent when the correction uses the observed rate.
- **Alternatives Weighed**: `Apply the observed onset rate to every specimen` (rejected: On complete specimens the physical model already has clean rate MAE 0.00274, lower than the pooled measurement MAE of 0.00675; the correction targets missing inputs.)
- **ATMS NOGOOD DECLARED**: Refuted `Cell-balanced mean imputation alone adequately recovers specimen-specific linear drift.` | Witness: `Missing-feature specimens have strain MAE 0.27447 versus 0.00773 on complete specimens, and rate MAE 0.10083 versus 0.00274.` | Blanket Diagnosis: `Replacing a missing coupon feature with a cell-balanced mean drops its specimen-specific contribution to drift. Under the specified law, that contribution enters the onset rate directly and the curve multiplied by elapsed time.` | Sublated Pivot: `Keep the prescribed feature alignment and imputation; test using the measured onset rate to recover missing drift and compare shared parameter fits with and without incomplete specimens.`

```bash
cd /holdout_task && cat > /tmp/alpha_w01_compare.py <<'PY'
import sys, pickle, numpy as np
sys.path.insert(0, '/holdout_task')
import alpha_w01_s00_solver as s
row = pickle.load(open('/holdout_task/public_alpha_w01_s00.pkl','rb'))[0]
d, gt = row['dataset'], row['gt']
def run(d, fit_complete, correction):
    tr, te = d['train'], d['test']
    dt,t=d['past_dt'],d['eval_times']
    maps=s._load_cells(d['calibration_load_cells'])
    x,xt=s._alignment(tr,te)
    m=np.isfinite(tr['coupon_features_raw']).all(axis=1) if fit_complete else np.ones(len(x),bool)
    taus,w=s._fit({k:v[m] for k,v in tr.items()},x[m],maps,dt,t)
    a,b=s._design(taus,*s._stress(te,maps),np.c_[np.ones(len(xt)),xt],dt,t)
    ep=te['strain_0'][:,None]+a@w
    rate=b@w
    missing=~np.isfinite(te['coupon_features_raw']).all(axis=1)
    delta=np.where(missing,te['instant_rate_obs']-rate,0)*correction
    return taus,w,ep+delta[:,None]*t,rate+delta

def report(label, te, e, r):
    miss=~np.isfinite(te['coupon_features_raw']).all(axis=1)
    mae=lambda z: np.mean(abs(z))
    print(label,'strain/rate MAE',[(round(mae(e[m]),5),round(mae(r[m]),5)) if m.any() else None for m in (np.ones(len(e),bool),miss,~miss)],'early/late',np.round(np.median(abs(e),axis=0)[[0,-1]],5))
for fit in (False,True):
    taus,w,p,q=run(d,fit,0.)
    print('complete fit',fit,'tau error',taus-gt['taus'],'weight error norm',np.linalg.norm(w-gt['theta']))
    for c in (0.,1.):
        _,_,p,q=run(d,fit,c)
        report(('clean',fit,c),d['test'],p-gt['te_clean'],q-gt['te_true_rate0'])
tr=d['train']
for lab in np.unique(tr['lab_id']):
    m=tr['lab_id']==lab
    dd=dict(d, train={k:v[~m] for k,v in tr.items()},test={k:v[m] for k,v in tr.items() if k!='strain_curves_obs'})
    for fit,c in ((False,0.),(False,1.),(True,1.)):
        ts,w,p,q=run(dd,fit,c)
        report(('LOLO',lab,fit,c),dd['test'],p-tr['strain_curves_obs'][m],q-tr['instant_rate_obs'][m])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/alpha_w01_compare.py
```
- **Container Exit Code**: `0` (`3035 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
complete fit False tau error [-0.00103574 -0.09364913] weight error norm 0.005839450130950983
('clean', False, 0.0) strain/rate MAE [(0.0244, 0.00887), (0.27447, 0.10083), (0.00773, 0.00274)] early/late [0.00045 0.00938]
('clean', False, 1.0) strain/rate MAE [(0.00857, 0.00305), (0.02109, 0.00759), (0.00773, 0.00274)] early/late [0.00045 0.00938]
complete fit True tau error [-0.00070606 -0.07076128] weight error norm 0.005023077708628229
('clean', True, 0.0) strain/rate MAE [(0.02432, 0.0088), (0.27416, 0.10067), (0.00767, 0.00267)] early/late [0.00041 0.00921]
('clean', True, 1.0) strain/rate MAE [(0.00849, 0.00298), (0.0208, 0.00759), (0.00767, 0.00267)] early/late [0.00041 0.00921]
('LOLO', 0, False, 0.0) strain/rate MAE [(0.11956, 0.01195), (0.33119, 0.07442), (0.10711, 0.00827)] early/late [0.00727 0.01294]
('LOLO', 0, False, 1.0) strain/rate MAE [(0.10897, 0.00781), (0.14059, 0.0), (0.10711, 0.00827)] early/late [0.00658 0.01294]
('LOLO', 0, True, 1.0) strain/rate MAE [(0.10892, 0.0077), (0.14163, 0.0), (0.107, 0.00815)] early/late [0.00672 0.01283]
('LOLO', 1, False, 0.0) strain/rate MAE [(0.10881, 0.01629), (0.30623, 0.10055), (0.0915, 0.0089)] early/late [0.00967 0.02851]
('LOLO', 1, False, 1.0) strain/rate MAE [(0.08864, 0.00818), (0.0561, 0.0), (0.0915, 0.0089)] early/late [0.009   0.02528]
('LOLO', 1, True, 1.0) strain/rate MAE [(0.08865, 0.00819), (0.056, 0.0), (0.09151, 0.00891)] early/late [0.00912 0.02615]
('LOLO', 2, False, 0.0) strain/rate MAE [(0.09092, 0.01332), (0.36476, 0.09595), (0.07541, 0.00864)] early/late [0.00839 0.00637]
('LOLO', 2, False, 1.0) strain/rate MAE [(0.07911, 0.00818), (0.14436, 0.0), (0.07541, 0.00864)] early/late [0.00839 0.00637]
('LOLO', 2, True, 1.0) strain/rate MAE [(0.07908, 0.0085), (0.14441, 0.0), (0.07538, 0.00898)] early/late [0.00849 0.00609]
('LOLO', 3, False, 0.0) strain/rate MAE [(0.0963, 0.01429), (0.23146, 0.07326), (0.08687, 0.01017)] early/late [0.00665 0.0238 ]
('LOLO', 3, False, 1.0) strain/rate MAE [(0.08931, 0.00951), (0.12434, 0.0), (0.08687, 0.01017)] early/late [0.00665 0.0238 ]
('LOLO', 3, True, 1.0) strain/rate MAE [(0.08913, 0.00943), (0.12323, 0.0), (0.08675, 0.01009)] early/late [0.00662 0.02369]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e3d30e4acaa40ba9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e3d30e4a (directory).
```

### Turn 6 — `execute_in_remote_docker` (`25928 ms` LLM)
- **Rationale**: The correction improved clean missing-feature predictions and missing-feature curve MAE in every leave-one-lab-out fold. Promote it, verify the edited output contract, and inspect the asset metadata for an archive format before implementing sparse-calibration fallback.
- **Alternatives Weighed**: `Use onset-rate correction on complete specimens` (rejected: Their fitted rates are already more accurate than the observed rates, so there is no missing drift to recover.)

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('alpha_w01_s00_solver.py')
s = p.read_text()
s = s.replace('    taus, w = _fit(train, x, maps, dt, time)\n', '''    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    # Estimate shared weights without treating imputed features as measurements.
    if complete.sum() >= 12:
        fit_train = {key: np.asarray(train[key])[complete] for key in (
            'lab_id', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0',
            'strain_curves_obs', 'instant_rate_obs')}
        taus, w = _fit(fit_train, x[complete], maps, dt, time)
    else:
        taus, w = _fit(train, x, maps, dt, time)
''')
s = s.replace("    return {'estimated_taus': taus, 'hereditary_weights': w,\n", '''    rate = b @ w
    # An onset-rate residual identifies the unobserved linear drift, not memory.
    observed = np.asarray(test['instant_rate_obs'])
    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    informed = missing & np.isfinite(observed)
    drift_delta = np.where(informed, observed - rate, 0.)
    return {'estimated_taus': taus, 'hereditary_weights': w,
''')
s = s.replace("'predicted_strain': test['strain_0'][:, None] + a @ w,\n            'initial_rate_pred': b @ w}", "'predicted_strain': test['strain_0'][:, None] + a @ w + drift_delta[:, None] * time,\n            'initial_rate_pred': rate + drift_delta}")
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
python3 - <<'PY'
import json
obj=json.load(open('public_asset_index.json'))
def walk(x,path=''):
    if isinstance(x,dict):
        for k,v in x.items(): walk(v,path+'/'+k)
    elif isinstance(x,list):
        for i,v in enumerate(x): walk(v,path+'/'+str(i))
    elif any(w in (path+' '+str(x)).lower() for w in ('archive','continual','sparse','calibration','spot check')):
        print(path,str(x)[:450])
walk(obj)
PY
if [ -d /tmp/hegel_continual_archive ]; then find /tmp/hegel_continual_archive -maxdepth 3 -type f -printf '%p %s bytes\n' | head -30; fi
```
- **Container Exit Code**: `0` (`766 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w01_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
/files//holdout_task/public_alpha_w01_s00.pkl/summary/first/dataset/calibration_load_cells/cmd_stress/dtype float64
/files//holdout_task/public_alpha_w01_s00.pkl/summary/first/dataset/calibration_load_cells/cmd_stress/shape/0 40
/files//holdout_task/public_alpha_w01_s00.pkl/summary/first/dataset/calibration_load_cells/cmd_stress/type ndarray
/files//holdout_task/public_alpha_w01_s00.pkl/summary/first/dataset/calibration_load_cells/lab_id/dtype int64
/files//holdout_task/public_alpha_w01_s00.pkl/summary/first/dataset/calibration_load_cells/lab_id/shape/0 40
/files//holdout_task/public_alpha_w01_s00.pkl/summary/first/dataset/calibration_load_cells/lab_id/type ndarray
/files//holdout_task/public_alpha_w01_s00.pkl/summary/first/dataset/calibration_load_cells/true_stress/dtype float64
/files//holdout_task/public_alpha_w01_s00.pkl/summary/first/dataset/calibration_load_cells/true_stress/shape/0 40
/files//holdout_task/public_alpha_w01_s00.pkl/summary/first/dataset/calibration_load_cells/true_stress/type ndarray

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9e9b454c98b801f9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_9e9b454c (directory).
```

### Turn 7 — `execute_in_remote_docker` (`73959 ms` LLM)
- **Rationale**: A single spot check does not identify both affine coefficients. Add a historical fallback for those labs, then test it with known gains, sparse checks and EMI because no real archive is mounted. Recheck the exact artifact in an isolated process and compare clean public slices to detect regression on dense calibration.
- **Alternatives Weighed**: `Pool calibration checks across labs` (rejected: A shared line would discard the lab-specific gains and offsets required by the model.)

```bash
cd /holdout_task && cat > /tmp/alpha_calibration_edit.py <<'PY'
from pathlib import Path
p = Path('alpha_w01_s00_solver.py')
s = p.read_text().replace('import numpy as np', 'import json\nimport pickle\nfrom pathlib import Path\n\nimport numpy as np', 1)
start, end = s.index('def _load_cells('), s.index('def _alignment(')
s = s[:start] + '''def _lab_checks(checks, lab):
    labs = np.asarray(checks['lab_id'])
    x = np.asarray(checks['cmd_stress'], dtype=float)
    y = np.asarray(checks['true_stress'], dtype=float)
    good = (labs == lab) & np.isfinite(x) & np.isfinite(y)
    return x[good], y[good]


def _affine(x, y, prior):
    """Return the best local affine fit and whether the checks identify it."""
    if len(x) < 2 or np.ptp(x) < .05:
        return np.array([prior[0], np.median(y - prior[0]*x) if len(x) else prior[1]]), False
    a, b = np.triu_indices(len(x), k=1)
    sep = np.abs(x[a] - x[b]) >= max(.05, .05 * np.ptp(x))
    a, b = a[sep], b[sep]
    gain = (y[b] - y[a]) / (x[b] - x[a])
    bias = y[a] - gain * x[a]
    residual = np.abs(y - gain[:, None] * x - bias[:, None])
    # Trim gross EMI. If only two points agree, prefer the line nearest the prior.
    h = max(2, len(x) // 2 + 1)
    score = (np.sort(residual, axis=1)[:, :h]**2).sum(axis=1)
    support = (residual <= .001).sum(axis=1)
    if support.max() < h or h == 2:
        candidates = np.flatnonzero(support == support.max())
        delta = (gain[candidates] - prior[0]) * max(1., np.ptp(x))
        best = candidates[np.argmin(delta**2 + (bias[candidates] - prior[1])**2)]
    else:
        best = np.argmin(score)
    err = residual[best]
    clean = err <= max(.001, min(.005, 5 * np.median(err)))
    A = np.column_stack((x[clean], np.ones(clean.sum())))
    fit = np.linalg.lstsq(A, y[clean], rcond=None)[0]
    identified = clean.sum() >= max(h, 3) and np.ptp(x[clean]) >= .1 and fit[0] > 0
    return fit, identified


def _check_panels(value):
    if isinstance(value, dict):
        if {'lab_id', 'cmd_stress', 'true_stress'} <= value.keys():
            yield value
        for key in ('dataset', 'calibration_load_cells'):
            if key in value:
                yield from _check_panels(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _check_panels(item)


def _archived_maps(labs, root):
    """Find the latest identifiable historical calibration per requested lab."""
    files = [p for p in Path(root).rglob('*')
             if p.suffix.lower() in ('.json', '.pkl', '.pickle', '.npz') and p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    found = {}
    for p in files:
        if len(found) == len(labs):
            break
        try:
            ext = p.suffix.lower()
            if ext == '.json':
                value = json.loads(p.read_text())
            elif ext == '.npz':
                with np.load(p, allow_pickle=False) as arrays:
                    if not {'lab_id', 'cmd_stress', 'true_stress'} <= set(arrays.files):
                        continue
                    value = {k: arrays[k] for k in ('lab_id', 'cmd_stress', 'true_stress')}
            else:
                with p.open('rb') as handle:
                    value = pickle.load(handle)
            for panel in reversed(list(_check_panels(value))):
                for lab in labs - found.keys():
                    fit, reliable = _affine(*_lab_checks(panel, lab), (1., 0.))
                    if reliable:
                        found[lab] = fit
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            continue
    return found


def _load_cells(checks, archive_root='/tmp/hegel_continual_archive'):
    nominal = np.array([1., 0.])
    data = [_lab_checks(checks, lab) for lab in range(4)]
    local = [_affine(x, y, nominal) for x, y in data]
    sparse = {lab for lab, (_, identified) in enumerate(local) if not identified}
    archive = _archived_maps(sparse, archive_root) if sparse else {}
    for lab, prior in archive.items():
        x, y = data[lab]
        # A spot check can update the offset but cannot identify a gain.
        offset = y - (prior[0] * x + prior[1])
        close = offset[np.abs(offset) <= .02]
        gain, bias = prior
        if len(close):
            bias += np.median(close)
        local[lab] = (np.array([gain, bias]), True)
    return np.array([fit for fit, _ in local])


''' + s[end:]
p.write_text(s)
PY
python3 /tmp/alpha_calibration_edit.py
/usr/bin/time -f 'isolated schema elapsed=%e s' timeout 60s python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import json, tempfile, pathlib, pickle
import numpy as np
import alpha_w01_s00_solver as s
row = pickle.load(open('public_alpha_w01_s00.pkl','rb'))[0]
d, g = row['dataset'], row['gt']
lab = np.repeat(np.arange(4), 7)
x = np.tile(np.linspace(-2,2,7), 4)
maps = np.array([[1.02,.03],[.98,.04],[1.12,-.11],[.88,.08]])
y = maps[lab,0]*x + maps[lab,1]
archive = dict(lab_id=lab.tolist(),cmd_stress=x.tolist(),true_stress=y.tolist())
# Wing A has full checks with EMI; Wing B has one valid spot check and one EMI-only check.
idx = np.r_[np.arange(14), 16, 24]; yy = y[idx].copy(); yy[[2,10,15]] += [.9,-1.2,1.3]; yy[14] += .003
checks = dict(lab_id=lab[idx],cmd_stress=x[idx],true_stress=yy)
with tempfile.TemporaryDirectory() as root:
    pathlib.Path(root,'prior.json').write_text(json.dumps({'dataset': {'calibration_load_cells': archive}}))
    inferred = s._load_cells(checks, root)
    expected = maps.copy(); expected[2,1] += .003
    error = np.max(np.abs(inferred-expected)); print('calibration max gain/offset error',error)
    assert error < 1e-10
# Public clean evaluation is separate from fitting; report the worst slices.
o = s.solve_alpha_w01_s00(d)
te = d['test']
e = np.abs(o['predicted_strain']-g['te_clean']); r = np.abs(o['initial_rate_pred']-g['te_true_rate0'])
print('public strain/rate MAE', e.mean(), r.mean())
print('worst lab strain/rate MAE', max(e[te['lab_id']==i].mean() for i in range(4)), max(r[te['lab_id']==i].mean() for i in range(4)))
m = ~np.isfinite(te['coupon_features_raw']).all(axis=1)
print('incomplete strain/rate MAE',e[m].mean(),r[m].mean(),'final-time strain MAE',e[:,-1].mean())
print('tau/weight max relative error',np.max(abs(o['estimated_taus']/g['taus']-1)),np.max(abs(o['hereditary_weights']/g['theta']-1)))
assert np.all(o['viscoplastic_weights']==0) and .2 <= o['estimated_taus'][0] < o['estimated_taus'][1] <= 8.
PY
```
- **Container Exit Code**: `0` (`654 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w01_s00_solver.py']`
- **Stdout (tail)**:
```text
calibration max gain/offset error 5.551115123125783e-16
public strain/rate MAE 0.008487279703040163 0.002980341423703378
worst lab strain/rate MAE 0.016439979221743208 0.0058293409672066565
incomplete strain/rate MAE 0.02080073826658581 0.0075902742261325425 final-time strain MAE 0.01640195287169894
tau/weight max relative error 0.018714756814455225 0.01721685221095748

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c8fb88b1108dc6ec (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_c8fb88b1 (directory).
```
- **Stderr (tail)**:
```text
bash: line 107: /usr/bin/time: No such file or directory
```

### Turn 8 — `execute_in_remote_docker` (`24500 ms` LLM)
- **Rationale**: The timing wrapper was unavailable, so the isolated schema check never ran on this revision. Run it directly with a timeout to catch import, runtime, or schema failures, and independently compare both prediction outputs with the stepwise recurrence to catch an indexing or onset-derivative error before submission.

```bash
set -e
cd /holdout_task
timeout 60s python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/alpha_w01_s00_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('public_alpha_w01_s00.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
te = d['test']
a = s.solve_alpha_w01_s00(d)
taus, w = a['estimated_taus'], a['hereditary_weights']
maps = s._load_cells(d['calibration_load_cells'])
hist, hold = s._stress(te, maps)
z = np.zeros((len(hold), 2))
for j, dt in enumerate(d['past_dt']):
    decay = np.exp(-dt / taus)
    z = decay * z + (1 - decay) * hist[:, j, None]
jump = hold[:, None] - z
x = s._alignment(d['train'], te)[1]
drift = w[2] + x @ w[3:]
relax_rate = (jump / taus) @ w[:2]
missing = ~np.isfinite(te['coupon_features_raw']).all(axis=1) & np.isfinite(te['instant_rate_obs'])
drift[missing] = te['instant_rate_obs'][missing] - relax_rate[missing]
t = d['eval_times']
curves = te['strain_0'][:, None] + (jump * w[:2]) @ (-np.expm1(-t[None, :] / taus[:, None])) + drift[:, None] * t
curve_err = np.max(np.abs(curves - a['predicted_strain']))
rate_err = np.max(np.abs(relax_rate + drift - a['initial_rate_pred']))
print(f'Direct recurrence: max strain discrepancy={curve_err:.3g}, max onset-rate discrepancy={rate_err:.3g}')
assert curve_err < 1e-10 and rate_err < 1e-10
PY
```
- **Container Exit Code**: `0` (`10804 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Direct recurrence: max strain discrepancy=1.33e-15, max onset-rate discrepancy=8.88e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c8fb88b1108dc6ec (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_c8fb88b1 (directory).
```

### Turn 9 — `submit_completed_task` (`14150 ms` LLM)
- **Rationale**: The final artifact passed isolated schema validation and direct recurrence checks, preserved the public clean-slice gains, and recovered known calibration coefficients in the sparse-EMI stress test.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_w01_s00_solver.py': 'FILE:9191'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w01_s00_solver.py`. The isolated schema check passed, and both prediction outputs matched the direct creep recurrence to floating-point precision. Public strain and onset-rate MAE were 0.00849 and 0.00298, respectively. The historical calibration fallback passed a synthetic sparse-check test with EMI outliers; no real archive was mounted.`

### Turn 10 — `execute_in_remote_docker` (`31454 ms` LLM)
- **Rationale**: A reflected-load mismatch would require fixing the hereditary sign convention; large wing-by-stratum signed residuals would guide a calibration or alignment investigation, and a large jackknife error would flag undue reliance on one load-cell check. The near-yield slice tests the edge of the supplied regime before isolated schema verification.
- **Alternatives Weighed**: `Test relay modes, stochastic drift operators, or classification balance` (rejected: The specified task is deterministic sub-yield creep with continuous targets; signed-load symmetry and wing-by-stratum residuals are the relevant checks.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/alpha_w01_s00_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('public_alpha_w01_s00.pkl', 'rb') as f:
    row = pickle.load(f)[0]
d, gt = row['dataset'], row['gt']
o = s.solve_alpha_w01_s00(d)
te = d['test']
lab, stratum = te['lab_id'], te['stratum']
missing = ~np.isfinite(te['coupon_features_raw']).all(axis=1)
maps = s._load_cells(d['calibration_load_cells'])
hist, hold = s._stress(te, maps)
e = o['predicted_strain'] - gt['te_clean']
r = o['initial_rate_pred'] - gt['te_true_rate0']
peak = np.maximum(np.abs(hold), np.max(np.abs(hist), axis=1))
print('minimum sub-yield margin', d['yield_stress'] - peak.max())
for name, m in [(f'wing={i} stratum={v}', (lab//2 == i) & (stratum == v)) for i in (0,1) for v in np.unique(stratum)] + [('near-yield quartile', peak >= np.quantile(peak,.75)), ('negative hold', hold < 0), ('nonnegative hold', hold >= 0), ('missing features', missing)]:
    if m.any():
        print(name, 'n', m.sum(), 'curve MAE', round(np.mean(abs(e[m])),6), 'onset MAE', round(np.mean(abs(r[m])),6), 'terminal/onset median signed error', np.round([np.median(e[m,-1]),np.median(r[m])],6))
# Reflect the calibrated loading about zero, holding coupon drift fixed.
x = s._alignment(d['train'], te)[1]
cov = np.c_[np.ones(len(x)), x]
taus, w = o['estimated_taus'], o['hereditary_weights']
a,b = s._design(taus,hist,hold,cov,d['past_dt'],d['eval_times'])
ar,br = s._design(taus,-hist,-hold,cov,d['past_dt'],d['eval_times'])
curve_reflection = np.max(np.abs((a+ar) @ w - 2*(cov @ w[2:])[:,None]*d['eval_times']))
rate_reflection = np.max(np.abs((b+br) @ w - 2*(cov @ w[2:])))
print('signed-load reflection errors',curve_reflection,rate_reflection)
assert max(curve_reflection,rate_reflection) < 1e-10
# Jackknife actual calibration checks to expose dependence on a single reference point.
c = d['calibration_load_cells']
errors = []
for j, labj in enumerate(c['lab_id']):
    m = np.arange(len(c['lab_id'])) != j
    g, b0 = s._load_cells({k:v[m] for k,v in c.items()})[labj]
    errors.append(abs(g*c['cmd_stress'][j]+b0-c['true_stress'][j]))
print('worst actual calibration jackknife stress error',max(errors))
PY
timeout 60s python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`18420 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
minimum sub-yield margin 0.4719380402489959
wing=0 stratum=0 n 13 curve MAE 0.016501 onset MAE 0.005868 terminal/onset median signed error [-0.025382 -0.004855]
wing=0 stratum=1 n 12 curve MAE 0.004433 onset MAE 0.001331 terminal/onset median signed error [-0.009784 -0.001428]
wing=0 stratum=2 n 9 curve MAE 0.00875 onset MAE 0.00313 terminal/onset median signed error [-0.008858 -0.001336]
wing=1 stratum=0 n 12 curve MAE 0.009062 onset MAE 0.003181 terminal/onset median signed error [0.008261 0.001637]
wing=1 stratum=1 n 18 curve MAE 0.006995 onset MAE 0.002482 terminal/onset median signed error [0.000855 0.000295]
wing=1 stratum=2 n 16 curve MAE 0.006117 onset MAE 0.002197 terminal/onset median signed error [0.004995 0.000681]
near-yield quartile n 25 curve MAE 0.013227 onset MAE 0.004672 terminal/onset median signed error [-0.010143 -0.001646]
negative hold n 33 curve MAE 0.009071 onset MAE 0.003206 terminal/onset median signed error [0.004517 0.000504]
nonnegative hold n 47 curve MAE 0.008078 onset MAE 0.002822 terminal/onset median signed error [-4.315e-03 -5.900e-05]
missing features n 5 curve MAE 0.020801 onset MAE 0.00759 terminal/onset median signed error [0.037015 0.006781]
signed-load reflection errors 8.881784197001252e-16 2.220446049250313e-16
worst actual calibration jackknife stress error 0.00024772850266507884
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c8fb88b1108dc6ec (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_c8fb88b1 (directory).
```

### Turn 11 — `submit_completed_task` (`14677 ms` LLM)
- **Rationale**: The saved candidate passed isolated schema validation, direct recurrence and signed-load symmetry checks, and the public wing, stratum, and near-yield audits. The worst leave-one-check-out calibration error was 0.000248 stress units, so the audit did not identify a reason to change the calibration fit.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_w01_s00_solver.py': 'FILE:9191'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w01_s00_solver.py`. The isolated schema check passed. Public strain and onset-rate MAE were 0.00849 and 0.00298 overall, and 0.02080 and 0.00759 on coupons with missing features. No real calibration archive was mounted; the fallback passed a synthetic sparse-check test with EMI outliers.`

### Turn 12 — `execute_in_remote_docker` (`33591 ms` LLM)
- **Rationale**: Thin actual calibration rows and inject EMI to test whether the historical fallback preserves stress over the real command range; a deviation above 0.001 would trigger inspection of the sparse-fit decision. For complete coupons, substitute public reference stresses and parameters only in the audit to distinguish an alignment discrepancy from a fitted-parameter discrepancy in the wing-by-stratum onset residuals.
- **Alternatives Weighed**: `Add rational terms, relay modes, or classification corrections` (rejected: The specified hold law is a deterministic two-branch recurrence with continuous outputs; none of those terms or modes is part of its input or output contract.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/alpha_w01_s00_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('public_alpha_w01_s00.pkl','rb') as f:
    row = pickle.load(f)[0]
d, gt = row['dataset'], row['gt']
c = d['calibration_load_cells']
baseline = s._load_cells(c)
lab = np.asarray(c['lab_id'])
x = np.asarray(c['cmd_stress'])
y = np.asarray(c['true_stress'])
te = d['test']
with tempfile.TemporaryDirectory(dir='/tmp') as root:
    Path(root,'prior.json').write_text(json.dumps({k:np.asarray(v).tolist() for k,v in c.items()}))
    worst = {'one_spot': 0., 'two_spots_one_emi': 0., 'three_spots_one_emi': 0.}
    for i in range(4):
        idx = np.flatnonzero(lab == i)
        idx = idx[np.argsort(x[idx])]
        cmds = np.r_[te['stress_hold_cmd'][te['lab_id']==i], te['past_stress_cmd'][te['lab_id']==i].ravel()]
        for j in range(len(idx)):
            for label, n in [('one_spot',1),('two_spots_one_emi',2),('three_spots_one_emi',3)]:
                take = idx[(j + np.arange(n)*(len(idx)//n)) % len(idx)]
                keep = (lab != i) | np.isin(np.arange(len(lab)),take)
                yy = y.copy()
                if n > 1:
                    yy[take[-1]] += .4 * (-1)**j
                a, b = s._load_cells({'lab_id':lab[keep], 'cmd_stress':x[keep], 'true_stress':yy[keep]},root)[i] - baseline[i]
                worst[label] = max(worst[label], np.max(np.abs(a*cmds+b)))
    print('Worst stress deviation under thinning and injected EMI on real check rows:', worst)
    assert max(worst.values()) < .001
# Diagnose whether calibration or reference-aligned drift explains the wing-by-stratum differences.
xtr, xt = s._alignment(d['train'], te)
o = s.solve_alpha_w01_s00(d)
complete = np.isfinite(te['coupon_features_raw']).all(axis=1)
a_true, b_true = s._design(gt['taus'], gt['te_hist_true'], gt['te_hold_true'], np.c_[np.ones(len(xt)),xt], d['past_dt'], d['eval_times'])
ref_rate = b_true @ gt['theta'] - gt['te_true_rate0']
fit_rate = o['initial_rate_pred'] - gt['te_true_rate0']
for wing in (0,1):
    for stratum in np.unique(te['stratum']):
        m = complete & (te['lab_id']//2==wing) & (te['stratum']==stratum)
        if m.any():
            print('complete-case wing/stratum',wing,int(stratum),'n',int(m.sum()),'reference and fitted signed onset medians',np.round([np.median(ref_rate[m]),np.median(fit_rate[m])],6))
PY
timeout 60s python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`14979 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Worst stress deviation under thinning and injected EMI on real check rows: {'one_spot': 0.00022196091964843398, 'two_spots_one_emi': 0.00022196091964843398, 'three_spots_one_emi': 0.00011502698224269414}
complete-case wing/stratum 0 0 n 12 reference and fitted signed onset medians [-0.002708 -0.003381]
complete-case wing/stratum 0 1 n 11 reference and fitted signed onset medians [ 0.000277 -0.001213]
complete-case wing/stratum 0 2 n 8 reference and fitted signed onset medians [ 0.000319 -0.001583]
complete-case wing/stratum 1 0 n 11 reference and fitted signed onset medians [0.002301 0.001634]
complete-case wing/stratum 1 1 n 18 reference and fitted signed onset medians [0.002025 0.000295]
complete-case wing/stratum 1 2 n 15 reference and fitted signed onset medians [0.002319 0.00058 ]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c8fb88b1108dc6ec (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_c8fb88b1 (directory).
```

### Turn 13 — `submit_completed_task` (`13698 ms` LLM)
- **Rationale**: The final artifact passed isolated schema and recurrence checks. The focused stress audit found a maximum calibration deviation of 0.000222 stress units under thinning and injected EMI, below the 0.001 audit threshold. The wing-by-stratum residuals did not justify a model change.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_w01_s00_solver.py`. The isolated schema check passed, and predictions matched the direct creep recurrence to floating-point precision. Public strain and onset-rate MAE were 0.00849 and 0.00298 overall, and 0.02080 and 0.00759 on coupons with missing features. No real calibration archive was mounted; the fallback was tested using public calibration rows with simulated sparse checks and EMI.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M95-Equal-Tool-Continual-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
