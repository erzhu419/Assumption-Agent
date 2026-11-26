# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_w02_s00__coupon_creep_baseline`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M95-Equal-Tool-Continual-World` |
| **Domain** | `Multi-Wing Polymer Coupon Hereditary Creep & Continual Calibration Stream` |
| **Wave / Wormhole Family** | `alpha_w02_source` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `16 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `479,592` (`460,017` / `90,112` / `19,575`) |
| **Total Episode Wall Time (s)** | `565.63s` |

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

### Turn 1 — `execute_in_remote_docker` (`15922 ms` LLM)
- **Rationale**: Inspect the public schema and archive inventory to determine how to align references and whether past checks can support a sparse lab fit, without transferring data files.
- **Alternatives Weighed**: `Implement before inspecting the dataset` (rejected: An incorrect split key or reference-panel layout would invalidate calibration and predictions.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && python3 - <<'PY'
import pickle,numpy as np
with open('public_alpha_w02_s00.pkl','rb') as f: d=pickle.load(f)
def show(obj,p=''):
 if isinstance(obj,dict):
  for k,v in obj.items(): show(v,p+'/'+str(k))
 elif isinstance(obj,np.ndarray):
  print(p,obj.shape,obj.dtype,repr(obj.reshape(-1)[:8]))
 elif isinstance(obj,(tuple,list)):
  print(p,type(obj).__name__,len(obj))
  if len(obj): show(obj[0],p+'/0')
 else: print(p,type(obj).__name__,repr(obj)[:180])
show(d)
PY
printf '\nArchive contents:\n'; if [ -d /tmp/hegel_continual_archive ]; then find /tmp/hegel_continual_archive -maxdepth 3 -type f -printf '%p %s bytes\n' | head -60; fi
printf '\nRelaxation skills:\n'; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation Cauchy robust IRLS reference-panel alignment imputation'
```
- **Container Exit Code**: `0` (`598 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
le)
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
```

### Turn 2 — `execute_in_remote_docker` (`22538 ms` LLM)
- **Rationale**: Inspect only solver inputs to establish the reference-panel axes, calibration coverage, and whether missing observations require masking. These determine the alignment and robust fit implementation.
- **Alternatives Weighed**: `Assume the reference-panel axes and observation layout` (rejected: A wrong axis would mix lab calibration with coupon variation; inspect the inputs before choosing the reductions.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle
import numpy as np
with open('public_alpha_w02_s00.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
for k,v in d.items():
    if isinstance(v,dict):
        print(k, {j:(np.shape(a),str(np.asarray(a).dtype)) for j,a in v.items()})
    else:
        print(k,np.shape(v),np.asarray(v).ravel()[:20])
c=d['calibration_load_cells']
for lab in range(4):
    m=c['lab_id']==lab
    x,y=c['cmd_stress'][m],c['true_stress'][m]
    print('cal lab',lab,'checks',list(zip(x.round(6),y.round(6))))
for k in ('train','test'):
    s=d[k]; X=s['coupon_features_raw']; labs=s['lab_id']; R=s['ref_coupons']
    print(k,'n by lab',np.bincount(labs),'missing by col',np.isnan(X).sum(0),'refs finite counts',np.isfinite(R).sum(axis=0).shape)
    print('ref rows',R[:2]); print('strata',[(j,np.unique(a,return_counts=True)) for j,a in s.items() if 'strat' in j]);
    for j in ('strain_curves_obs','instant_rate_obs','strain_0'):
        if j in s:
            a=np.asarray(s[j]); print(j,'nonfinite',np.sum(~np.isfinite(a)),'quantiles',np.nanquantile(a,[0,.05,.5,.95,1]))
try:
    import scipy; print('scipy',scipy.__version__)
except ImportError: print('no scipy')
PY
for fn in autogenous_prim_robust_affine_projection.robust_residual_weights autogenous_prim_reference_panel_balancing.balanced_cell_weights autogenous_prim_reference_panel_balancing.align_reference_panel autogenous_prim_reference_panel_balancing.weighted_column_fill autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation; do PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe "$fn"; done
```
- **Container Exit Code**: `0` (`636 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`61636 ms` LLM)
- **Rationale**: Implement the stated recurrence and its exact initial derivative. Run the public validator to catch shape, finiteness, or branch-ordering failures before testing held-out labs and degraded calibration.
- **Alternatives Weighed**: `Use the single-branch anchored-relaxation primitive` (rejected: It does not represent two independent memory states and signed branch amplitudes across coupons.)

```bash
cat > /holdout_task/alpha_w02_s00_solver.py <<'PY'
"""Robust calibration and two-branch sub-yield creep identification."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x, y, prior):
    """Fit an affine correction using consensus over pairs of spot checks."""
    if not len(x):
        return prior
    # Fit corrections rather than the dominant identity component.
    dy = y - x
    a, b = prior[0] - 1, prior[1]
    if len(x) == 1:
        return a + 1, dy[0] - a*x[0]
    i, j = np.triu_indices(len(x), 1)
    keep = np.abs(x[i] - x[j]) >= max(0.04, 0.12*np.ptp(x))
    i, j = i[keep], j[keep]
    if not len(i):
        return a + 1, np.median(dy - a*x)
    slopes = np.r_[a, (dy[j] - dy[i])/(x[j] - x[i])]
    offsets = np.r_[np.median(dy - a*x), dy[i] - slopes[1:]*x[i]]
    err = abs(dy - slopes[:, None]*x - offsets[:, None])
    consensus = err < 0.0007
    score = consensus.sum(axis=1)
    loss = np.minimum(err/0.0007, 1).sum(axis=1)
    win = np.lexsort((np.abs(slopes-a), loss, -score))[0]
    m = consensus[win]
    if m.sum() < 2 or np.ptp(x[m]) < 0.04:
        return a + 1, np.median(dy - a*x)
    if len(x) <= 3 and m.sum() < len(x):
        # Too few checks to identify a new slope and reject a glitch independently.
        return a + 1, np.median(dy - a*x)
    u = x[m] - x[m].mean()
    slope = np.dot(u, dy[m])/np.dot(u, u)
    return 1 + slope, dy[m].mean() - slope*x[m].mean()


def _calibration(d):
    c = d['calibration_load_cells']
    lab, x, y = map(np.asarray, (c['lab_id'], c['cmd_stress'], c['true_stress']))
    finite = np.isfinite(x) & np.isfinite(y)
    out = np.empty((4, 2))
    for l in range(4):
        m = finite & (lab == l)
        out[l] = _affine(x[m], y[m], (1., 0.))
    return out


def _align(d):
    tr, te = d['train'], d['test']
    features, ref, lab, stratum = [np.concatenate((tr[k], te[k])) for k in
        ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum')]
    x = features.astype(float, copy=True)
    observed = np.isfinite(x)
    for l in np.unique(lab):
        m = lab == l
        r = ref[m]
        center = np.nanmedian(np.nanmean(r, axis=0))
        scale = np.nanmedian(np.nanstd(r, axis=0, ddof=0))
        x[m] = (x[m] - center)/scale
    _, cells = np.unique(np.column_stack((lab, stratum)), axis=0, return_inverse=True)
    counts = np.bincount(cells)
    # Average available values within each cell, then weight cells equally.
    cell_means = np.full((len(counts), x.shape[1]), np.nan)
    for cell in range(len(counts)):
        v = x[cells == cell]
        n = np.isfinite(v).sum(axis=0)
        np.divide(np.nansum(v, axis=0), n, out=cell_means[cell], where=n>0)
    x = np.where(observed, x, np.nanmean(cell_means, axis=0))
    n = len(tr['lab_id'])
    return x[:n], x[n:], observed[:n].all(axis=1)


def _state_inputs(split, calibration):
    a, b = calibration[np.asarray(split['lab_id'])].T
    hist = a[:, None]*split['past_stress_cmd'] + b[:, None]
    hold = a*split['stress_hold_cmd'] + b
    return hist, hold


def _basis(tau, hist, hold, dt, times, x):
    dt = np.asarray(dt)
    after = np.cumsum(dt[::-1])[::-1] - dt
    coeff = -np.expm1(-dt[:, None]/tau)*np.exp(-after[:, None]/tau)
    diff = hold[:, None] - hist @ coeff
    t = np.asarray(times)
    drift = np.column_stack((np.ones(len(x)), x))
    curve = np.concatenate((diff[:, None, :] * (-np.expm1(-t[:, None]/tau))[None, :, :],
                           drift[:, None, :]*t[None, :, None]), axis=2)
    rate = np.column_stack((diff/tau, drift))
    return curve, rate


def _noise(r):
    return max(1e-9, 1.4826*np.median(np.abs(r - np.median(r))))


def _fit(hist, hold, dt, t, x, curve, rate):
    k = len(t)
    yc = curve.ravel()
    mc, mr = np.isfinite(yc), np.isfinite(rate)
    yc, yr = yc[mc], rate[mr]
    y = np.r_[yc, yr]
    n = len(yc)

    def design(tau):
        bc, br = _basis(tau, hist, hold, dt, t, x)
        return np.vstack((bc.reshape(-1, 7)[mc], br[mr]))

    def search(scale, starts):
        def residual(p):
            return (design(p[:2]) @ p[2:] - y)/scale
        best = None
        lo = np.r_[0.2, 0.2, np.full(7, -np.inf)]
        hi = np.r_[8., 8., np.full(7, np.inf)]
        for p in starts:
            fit = least_squares(residual, p, bounds=(lo, hi), loss='cauchy',
                                x_scale='jac', max_nfev=110, ftol=2e-9)
            if best is None or fit.cost < best.cost:
                best = fit
        return best.x

    starts = []
    for tau in ([0.35, 2.4], [0.7, 4.0], [1.3, 6.5]):
        b = design(np.asarray(tau))
        w = np.linalg.lstsq(b, y, rcond=None)[0]
        starts.append(np.r_[tau, w])
    r0 = design(starts[1][:2]) @ starts[1][2:] - y
    def scales(r, multiple):
        sc = _noise(r[:n])
        sr = _noise(r[n:]) if len(yr) else sc
        return multiple*np.r_[np.full(n, sc), np.full(len(yr), sr)]
    p = search(scales(r0, 1.7), starts)
    r = design(p[:2]) @ p[2:] - y
    p = search(scales(r, 2.0), [p])
    if p[0] > p[1]:
        p[:4] = p[[1, 0, 3, 2]]
    return p[:2], p[2:]


def solve_alpha_w02_s00(dataset: dict) -> dict:
    tr, te = dataset['train'], dataset['test']
    dt, t = dataset['past_dt'], dataset['eval_times']
    cal = _calibration(dataset)
    x, xt, complete = _align(dataset)
    hist, hold = _state_inputs(tr, cal)
    hist_t, hold_t = _state_inputs(te, cal)
    # Do not fit material parameters to proxies for missing coupon covariates.
    taus, weights = _fit(hist[complete], hold[complete], dt, t, x[complete],
                        (tr['strain_curves_obs'] - tr['strain_0'][:, None])[complete],
                        tr['instant_rate_obs'][complete])
    b, v = _basis(taus, hist_t, hold_t, dt, t, xt)
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'viscoplastic_weights': np.zeros(2),
            'predicted_strain': te['strain_0'][:, None] + b @ weights,
            'initial_rate_pred': v @ weights}
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
python3 - <<'PY'
import pickle, numpy as np
from alpha_w02_s00_solver import solve_alpha_w02_s00
with open('public_alpha_w02_s00.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
o=solve_alpha_w02_s00(d)
print({k:v for k,v in o.items() if v.ndim==1 and len(v)<10})
for k in ('train','test'):
 s=d[k]; print(k,'strata',np.unique(s['stratum'],return_counts=True),'complete',int(np.isfinite(s['coupon_features_raw']).all(1).sum()),'rate finite',np.isfinite(s['instant_rate_obs']).sum())
PY
```
- **Container Exit Code**: `0` (`4646 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w02_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
{'estimated_taus': array([0.53919292, 3.84094209]), 'hereditary_weights': array([ 0.66278946,  0.52201753,  0.09005913,  0.1594531 , -0.12937732,
        0.10957437, -0.07947444]), 'viscoplastic_weights': array([0., 0.])}
train strata (array([0, 1, 2]), array([61, 63, 76])) complete 173 rate finite 200
test strata (array([0, 1, 2]), array([21, 27, 32])) complete 75 rate finite 80

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d00e1232e7c7624a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d00e1232 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`37161 ms` LLM)
- **Rationale**: Check the state recurrence against an independent loop, measure held-out errors separately for complete and imputed coupons, and determine when current-batch calibration needs an archive prior.
- **Alternatives Weighed**: `Submit after the schema check` (rejected: The validator does not test the recurrence, transfer to a held-out lab, or calibration under sparse and noisy checks.)

```bash
cat > /tmp/alpha_w02_validate.py <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_w02_s00_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open(root + 'public_alpha_w02_s00.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr = d['train']
t = d['eval_times']
dt = d['past_dt']
x, _, complete = s._align(d)
cal = s._calibration(d)
h, p = s._state_inputs(tr, cal)
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']

# Independent recurrence check, including a nonzero calibration offset.
taus = np.array([0.20, 8.0])
weights = np.array([.6, .5, .09, .16, -.13, .11, -.08])
hh, pp = h[:12] + .16, p[:12] + .16
z = np.zeros((len(hh), 2))
for k, step in enumerate(dt):
    a = np.exp(-step/taus)
    z = a*z + (1-a)*hh[:, k, None]
amp = (pp[:, None]-z)*weights[:2]
drift = weights[2] + x[:12] @ weights[3:]
c, v = s._basis(taus, hh, pp, dt, np.r_[0., t], x[:12])
np.testing.assert_allclose(c @ weights, amp @ (1-np.exp(-np.r_[0., t][None, :]/taus[:, None])) + drift[:, None]*np.r_[0., t], atol=1e-12)
np.testing.assert_allclose(v @ weights, (amp/taus).sum(axis=1) + drift, atol=1e-12)
print('independent endpoint and rate checks passed')

worst = np.zeros(3)
for l in np.unique(tr['lab_id']):
    holdout = tr['lab_id'] == l
    fit = ~holdout & complete
    tau, w = s._fit(h[fit], p[fit], dt, t, x[fit], y[fit], r[fit])
    b, v = s._basis(tau, h[holdout], p[holdout], dt, t, x[holdout])
    ce = np.abs(b @ w - y[holdout]); re = np.abs(v @ w - r[holdout]); mc = complete[holdout]
    vals = [np.nanmedian(ce[mc], axis=0).max(), np.nanmedian(re[mc]), np.nanmedian(ce[~mc], axis=0).max()]
    worst = np.maximum(worst, vals)
    print('lab', l, 'n complete/missing', int(mc.sum()), int((~mc).sum()), 'tau', tau.round(5), 'worst horizon complete / rate complete / horizon missing median abs errors', np.round(vals, 6))
print('worst LOLO errors:', worst.round(6))

rng = np.random.default_rng(102)
for n, glitch in [(10, 2), (4, 1), (2, 0), (1, 0)]:
    errors = []
    for i in range(100):
        a, b = rng.uniform(.88, 1.12), rng.uniform(-.12, .12)
        xc = np.linspace(-2, 2, n) if n > 1 else np.array([1.2])
        yc = a*xc + b + rng.normal(0, 1e-4, n)
        yc[rng.choice(n, glitch, replace=False)] += rng.choice([-1., 1.], glitch)*.15
        aa, bb = s._affine(xc, yc, (1., 0.))
        errors.append(max(abs((aa-a)*q + bb-b) for q in (-2.5, 2.5)))
    print('cal stress checks/glitches', n, glitch, 'maximum extrapolation error p95/max', np.round([np.quantile(errors,.95),max(errors)], 6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/alpha_w02_validate.py
printf '\nArchive inventory:\n'; if [ -d /tmp/hegel_continual_archive ]; then find /tmp/hegel_continual_archive -maxdepth 3 -type f -printf '%P %s bytes\n' | head -40; else echo 'continual archive not mounted'; fi
```
- **Container Exit Code**: `0` (`800 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
independent endpoint and rate checks passed
lab 0 n complete/missing 37 8 tau [0.53927 3.85901] worst horizon complete / rate complete / horizon missing median abs errors [0.092133 0.021096 0.424628]
lab 1 n complete/missing 43 5 tau [0.54161 3.96504] worst horizon complete / rate complete / horizon missing median abs errors [0.052163 0.009405 0.634167]
lab 2 n complete/missing 48 5 tau [0.53825 3.78097] worst horizon complete / rate complete / horizon missing median abs errors [0.028018 0.008976 0.461997]
lab 3 n complete/missing 45 9 tau [0.53807 3.70596] worst horizon complete / rate complete / horizon missing median abs errors [0.029746 0.007914 0.512896]
worst LOLO errors: [0.092133 0.021096 0.634167]
cal stress checks/glitches 10 2 maximum extrapolation error p95/max [0.0002   0.000279]
cal stress checks/glitches 4 1 maximum extrapolation error p95/max [0.000233 0.000372]
cal stress checks/glitches 2 0 maximum extrapolation error p95/max [0.000249 0.000277]
cal stress checks/glitches 1 0 maximum extrapolation error p95/max [0.399886 0.441342]

Archive inventory:
continual archive not mounted

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d00e1232e7c7624a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d00e1232 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`29444 ms` LLM)
- **Rationale**: Look for the archive layout and establish observed noise scales before deciding how to supplement single-point calibration. A reference-coverage failure would instead point to an alignment problem.
- **Alternatives Weighed**: `Freeze the current candidate` (rejected: A one-point check leaves gain unidentified; the single-check stress test had 0.400 p95 extrapolation error, and the solver does not yet consult the requested archive.)

```bash
printf 'Continual calibration inventory:\n'; find /tmp /holdout_task /pretrain_task -maxdepth 3 \( -iname '*continual*' -o -iname '*calibrat*' -o -iname '*alpha_w02*' \) -print 2>/dev/null | head -60; cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p=importlib.util.spec_from_file_location('solver','alpha_w02_s00_solver.py'); s=importlib.util.module_from_spec(p); p.loader.exec_module(s)
a=pickle.load(open('public_alpha_w02_s00.pkl','rb'))[0]; d=a['dataset']; g=a['gt']; print('public evaluation fields:',[(k,np.shape(v)) for k,v in g.items()]); print('reference row coverage by lab:')
for l in range(4):
 r=np.concatenate([q['ref_coupons'][q['lab_id']==l] for q in (d['train'],d['test'])]); print(l,int(np.isfinite(r).sum(0).min()),float(np.nanmedian(np.nanmean(r,0))),float(np.nanmedian(np.nanstd(r,0))))
print('rate/curve residual scale by lab (complete coupons):')
x,_,m=s._align(d); q=d['train']; h,p=s._state_inputs(q,s._calibration(d)); o=s.solve_alpha_w02_s00(d); b,v=s._basis(o['estimated_taus'],h,p,d['past_dt'],d['eval_times'],x); w=o['hereditary_weights']; rc=b@w+q['strain_0'][:,None]-q['strain_curves_obs']; rr=v@w-q['instant_rate_obs'];
for l in range(4):
 j=m & (q['lab_id']==l); print(l,round(s._noise(rc[j]),6),round(s._noise(rr[j]),6))
PY
```
- **Container Exit Code**: `0` (`548 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Continual calibration inventory:
/tmp/alpha_w02_validate.py
/tmp/.hegel_candidate_archive/v01_d00e1232/slot_00__alpha_w02_s00_solver.py
/tmp/.hegel_candidate_archive/v01_d00e1232/alpha_w02_s00_solver.py
/holdout_task/public_alpha_w02_s00.pkl
/holdout_task/__pycache__/alpha_w02_s00_solver.cpython-312.pyc
/holdout_task/alpha_w02_s00_solver.py
public evaluation fields: [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (80, 10)), ('te_true_rate0', (80,)), ('te_cross', (80,)), ('te_secant', (80,)), ('te_hold_true', (80,)), ('te_hist_true', (80, 18))]
reference row coverage by lab:
0 53 -0.5782687139867693 0.9020490515011541
1 59 0.47974986940000097 1.245703510230955
2 76 -0.27456020396319536 0.9565516568856414
3 67 0.8072200998954855 1.4027271644149484
rate/curve residual scale by lab (complete coupons):
0 0.062328 0.026851
1 0.019367 0.014635
2 0.01622 0.01349
3 0.013091 0.008517

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d00e1232e7c7624a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d00e1232 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`27517 ms` LLM)
- **Rationale**: Use the public evaluation fields only for diagnostics, not fitting. If the lab or missing-feature gap persists with known parameters, audit alignment and imputation; if it disappears, focus on the fit. Sparse load-cell calibration remains a separate issue because the archive is not mounted.
- **Alternatives Weighed**: `Retune the robust fit immediately` (rejected: Tuning cannot fix a reference-alignment error that remains when the public case's known parameters and physical stresses are used.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p=importlib.util.spec_from_file_location('solver','alpha_w02_s00_solver.py'); s=importlib.util.module_from_spec(p); p.loader.exec_module(s)
with open('public_alpha_w02_s00.pkl','rb') as f: cases=pickle.load(f)
for i,a in enumerate(cases):
 d,g=a['dataset'],a['gt']; tr,te=d['train'],d['test']; out=s.solve_alpha_w02_s00(d); ca=s._calibration(d)
 print('case',i,'tau error',np.round(out['estimated_taus']-g['taus'],6),'weight error',np.round(out['hereditary_weights']-g['theta'],6),'calibration max error',np.max(abs(ca-np.column_stack((g['stress_gains'],g['stress_biases'])))))
 time=d['eval_times']; _,xt,_=s._align(d)
 b,v=s._basis(g['taus'],g['te_hist_true'],g['te_hold_true'],d['past_dt'],time,xt)
 oracle_curve=te['strain_0'][:,None]+b@g['theta']; oracle_rate=v@g['theta']
 for l in range(4):
  for complete in (True,False):
   m=(te['lab_id']==l)&(np.isfinite(te['coupon_features_raw']).all(1)==complete)
   if not m.any(): continue
   err=out['predicted_strain'][m]-g['te_clean'][m]; rate=out['initial_rate_pred'][m]-g['te_true_rate0'][m]; actual=te['instant_rate_obs'][m]-g['te_true_rate0'][m]; oc=(oracle_curve-g['te_clean'])[m]; ov=(oracle_rate-g['te_true_rate0'])[m]
   print('lab, complete, n',l,complete,m.sum(),'curve/rate MAE',np.round([abs(err).mean(),abs(rate).mean()],6),'known-parameter curve/rate MAE',np.round([abs(oc).mean(),abs(ov).mean()],6),'observed-rate MAE',round(abs(actual).mean(),6),'alignment-rate signed',round(ov.mean(),6))
PY
```
- **Container Exit Code**: `0` (`540 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case 0 tau error [-0.000807 -0.037058] weight error [-2.211e-03  1.018e-03  5.900e-05 -5.470e-04  6.230e-04 -4.260e-04
  5.260e-04] calibration max error 6.778088103920332e-05
lab, complete, n 0 True 13 curve/rate MAE [0.049398 0.018861] known-parameter curve/rate MAE [0.047902 0.017579] observed-rate MAE 0.007788 alignment-rate signed -0.014913
lab, complete, n 0 False 2 curve/rate MAE [0.204339 0.074344] known-parameter curve/rate MAE [0.203909 0.074829] observed-rate MAE 0.007008 alignment-rate signed 0.074829
lab, complete, n 1 True 16 curve/rate MAE [0.012496 0.004323] known-parameter curve/rate MAE [0.014865 0.005455] observed-rate MAE 0.006973 alignment-rate signed 0.000897
lab, complete, n 1 False 1 curve/rate MAE [0.019168 0.007621] known-parameter curve/rate MAE [0.025935 0.009517] observed-rate MAE 0.020969 alignment-rate signed -0.009517
lab, complete, n 2 True 29 curve/rate MAE [0.013092 0.00503 ] known-parameter curve/rate MAE [0.010805 0.003965] observed-rate MAE 0.009729 alignment-rate signed 0.000972
lab, complete, n 2 False 1 curve/rate MAE [0.182407 0.069333] known-parameter curve/rate MAE [0.179579 0.0659  ] observed-rate MAE 0.008427 alignment-rate signed 0.0659
lab, complete, n 3 True 17 curve/rate MAE [0.00599  0.002814] known-parameter curve/rate MAE [0.008531 0.003131] observed-rate MAE 0.007947 alignment-rate signed -0.00071
lab, complete, n 3 False 1 curve/rate MAE [0.269555 0.098688] known-parameter curve/rate MAE [0.272125 0.099863] observed-rate MAE 0.004119 alignment-rate signed 0.099863

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d00e1232e7c7624a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d00e1232 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`24507 ms` LLM)
- **Rationale**: Check whether lab 0 has a few anomalous reference columns or a broader shift after the prescribed pooling. The former warrants an alignment audit; the latter does not justify replacing the specified statistic. Read the public index for any documented archive layout before implementing sparse-check support.
- **Alternatives Weighed**: `Use test rate observations to overwrite the drift` (rejected: That would bypass the specified feature-based hold law and use the noisy counterpart of a scored output as its prediction.)

```bash
cd /holdout_task && cat public_asset_index.json && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p=importlib.util.spec_from_file_location('solver','alpha_w02_s00_solver.py'); s=importlib.util.module_from_spec(p); p.loader.exec_module(s)
with open('public_alpha_w02_s00.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
tr,te=d['train'],d['test']; c=d['calibration_load_cells']; x,_,m=s._align(d); o=s.solve_alpha_w02_s00(d); tau,w=o['estimated_taus'],o['hereditary_weights']; h,p=s._state_inputs(tr,s._calibration(d)); b,v=s._basis(tau,h,p,d['past_dt'],d['eval_times'],x); err=tr['instant_rate_obs']-v@w
for l in range(4):
 q=c['lab_id']==l; r=np.concatenate([z['ref_coupons'][z['lab_id']==l] for z in (tr,te)]); j=(tr['lab_id']==l)&m
 print('lab',l,'checks',q.sum(),'command span',np.ptp(c['cmd_stress'][q]),'check residuals',np.round(c['true_stress'][q]-np.polyval(s._calibration(d)[l],c['cmd_stress'][q]),6)); print('ref column means/std quantiles',np.round(np.quantile(np.nanmean(r,0),[0,.25,.5,.75,1]),4),np.round(np.quantile(np.nanstd(r,0),[0,.25,.5,.75,1]),4)); print('rate residual median/MAD',np.round([np.median(err[j]),s._noise(err[j])],5),'feature/rate residual correlations',np.round(np.corrcoef(np.column_stack((x[j],err[j])),rowvar=False)[-1,:4],3))
PY
```
- **Container Exit Code**: `0` (`634 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
},
            "te_true_rate0": {
              "dtype": "float64",
              "shape": [
                80
              ],
              "type": "ndarray"
            },
            "theta": {
              "dtype": "float64",
              "shape": [
                7
              ],
              "type": "ndarray"
            },
            "vp_theta": {
              "dtype": "float64",
              "shape": [
                2
              ],
              "type": "ndarray"
            }
          }
        },
        "len": 1,
        "type": "list"
      }
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 1277
    }
  },
  "schema": "hegel_public_asset_index_v1"
}lab 0 checks 10 command span 3.1148697708006523 check residuals [ 1.20e-05  2.70e-05  7.10e-05 -9.80e-05  1.00e-04 -5.20e-05 -4.90e-05
  1.09e-04 -1.16e-04 -4.00e-06]
ref column means/std quantiles [-0.8204 -0.6477 -0.5783 -0.5004 -0.3023] [0.709  0.8281 0.902  0.9343 1.0634]
rate residual median/MAD [0.00701 0.02685] feature/rate residual correlations [ 0.509 -0.509  0.622 -0.276]
lab 1 checks 10 command span 3.1934824895557394 check residuals [-5.90e-05  8.20e-05 -5.00e-06  7.00e-05 -1.84e-04  1.70e-04 -6.20e-05
 -5.50e-05  3.20e-05  1.00e-05]
ref column means/std quantiles [0.1604 0.4082 0.4797 0.5609 0.7586] [1.0881 1.1591 1.2457 1.3175 1.4414]
rate residual median/MAD [0.00314 0.01463] feature/rate residual correlations [-0.173  0.467 -0.292  0.122]
lab 2 checks 10 command span 3.1530242620878806 check residuals [ 1.70e-05 -2.00e-06 -1.17e-04  5.30e-05  9.00e-06 -5.80e-05 -5.20e-05
 -1.10e-05  1.15e-04  4.70e-05]
ref column means/std quantiles [-0.4705 -0.3472 -0.2746 -0.1541 -0.0871] [0.8215 0.9242 0.9566 0.984  1.1065]
rate residual median/MAD [-0.00089  0.01349] feature/rate residual correlations [ 0.25  -0.161  0.337  0.046]
lab 3 checks 10 command span 2.762774178854202 check residuals [-1.00e-04  1.40e-05 -1.70e-05 -9.00e-06 -8.80e-05  2.70e-05  1.26e-04
 -7.20e-05  4.90e-05  7.00e-05]
ref column means/std quantiles [0.3719 0.6472 0.8072 0.8829 1.1609] [1.2252 1.3197 1.4027 1.4939 1.6542]
rate residual median/MAD [-0.00255  0.00852] feature/rate residual correlations [-0.22   0.279 -0.006  0.084]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d00e1232e7c7624a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d00e1232 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`39352 ms` LLM)
- **Rationale**: With no archive mounted, hide all but one public check at a time and test whether the calibrated labs supply enough information to recover the remaining lab’s gain. If response-only bias is poorly identified or anchoring fails to improve worst-case extrapolation, do not promote that fallback.
- **Alternatives Weighed**: `Replace the specified balanced feature imputation with test-rate reconstruction` (rejected: This would change the requested predictor and use the measured counterpart of a scored output.)

```bash
cat > /tmp/alpha_sparse_probe.py <<'PY'
import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',root+'alpha_w02_s00_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
a=pickle.load(open(root+'public_alpha_w02_s00.pkl','rb'))[0]; d,g=a['dataset'],a['gt']; tr=d['train']; checks=d['calibration_load_cells']
lab=tr['lab_id']; dt,t=d['past_dt'],d['eval_times']; x,_,complete=s._align(d); hist,hold=s._state_inputs(tr,s._calibration(d)); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']
true=np.column_stack((g['stress_gains'],g['stress_biases']))
for l in range(4):
    m=(lab!=l)&complete; j=(lab==l)&complete
    tau,w=s._fit(hist[m],hold[m],dt,t,x[m],y[m],r[m]); h=tr['past_stress_cmd'][j]; p=tr['stress_hold_cmd'][j]
    bc,br=s._basis(tau,h,p,dt,t,x[j]); ac,ar=bc[:,:,:2]@w[:2],br[:,:2]@w[:2]
    bc,br=s._basis(tau,np.ones_like(h),np.ones_like(p),dt,t,x[j]); ec,er=bc[:,:,:2]@w[:2],br[:,:2]@w[:2]
    drift=w[2]+x[j]@w[3:]
    # Profile a lab drift offset/scale so finite-panel alignment error does not look like bias or gain.
    C=np.column_stack((ac.ravel(),ec.ravel(),np.tile(t,j.sum()),(drift[:,None]*t).ravel()))
    R=np.column_stack((ar,er,np.ones(j.sum()),drift))
    design=np.vstack((C/.025,R/.014)); yy=np.r_[((y[j]-drift[:,None]*t)/.025).ravel(),(r[j]-drift)/.014]
    z=least_squares(lambda z: design@z-yy,[1.,0.,0.,0.],loss='cauchy',f_scale=2).x
    mc=checks['lab_id']==l; q=checks['cmd_stress'][mc]; v=checks['true_stress'][mc]; err=[]; base=[]
    for qi,vi in zip(q,v):
        A=design[:,[0,2,3]].copy(); A[:,0]-=qi*design[:,1]; Y=yy-vi*design[:,1]
        p=least_squares(lambda p: A@p-Y,z[[0,2,3]],loss='cauchy',f_scale=2).x
        err.append(np.abs(p[0]-true[l,0])*2.5+abs(vi-qi*p[0]-true[l,1])); base.append(abs(1-true[l,0])*2.5+abs(vi-qi-true[l,1]))
    print(l,'true',true[l].round(5),'response-only gain,bias',z[:2].round(5),'response/check residual max',round(np.max(np.abs(v-(z[0]*q+z[1]))),5),'single-check max error baseline -> anchored',round(max(base),5),round(max(err),5),'two-stress design svals',np.linalg.svd(design,compute_uv=False).round(2))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/alpha_sparse_probe.py
```
- **Container Exit Code**: `0` (`817 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 true [1. 0.] response-only gain,bias [ 1.00045 -0.03283] response/check residual max 0.03334 single-check max error baseline -> anchored 0.00016 0.00188 two-stress design svals [2474.83 1252.6   552.15    3.87]
1 true [1. 0.] response-only gain,bias [0.99994 0.16077] response/check residual max 0.16097 single-check max error baseline -> anchored 0.00026 0.00034 two-stress design svals [2663.59 1299.63  691.12    4.3 ]
2 true [1. 0.] response-only gain,bias [ 0.9991  -0.04427] response/check residual max 0.04615 single-check max error baseline -> anchored 0.00012 0.0042 two-stress design svals [2843.44 1609.37  609.07    4.33]
3 true [1. 0.] response-only gain,bias [1.0006  0.09895] response/check residual max 0.09967 single-check max error baseline -> anchored 0.00025 0.0025 two-stress design svals [2747.2  1851.53  708.63    4.11]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d00e1232e7c7624a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d00e1232 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`61939 ms` LLM)
- **Rationale**: Preserve the verified dense-calibration path and use the archive or training response only when checks are sparse. Perturb both labs on each wing while preserving their physical responses; if either wing exceeds the stress-error threshold, revise the sparse fallback before submission.
- **Alternatives Weighed**: `Infer both calibration coefficients from creep alone` (rejected: The public lab jackknife showed response-only bias errors up to 0.161; a spot check or archived calibration provides a better intercept anchor.)

```bash
cd /holdout_task && cp alpha_w02_s00_solver.py /tmp/alpha_w02_before_sparse.py && python3 - <<'PY'
from pathlib import Path
p=Path('alpha_w02_s00_solver.py')
s=p.read_text().replace('import numpy as np','import json\nfrom pathlib import Path\n\nimport numpy as np',1)
start=s.index('def _calibration(d):')
end=s.index('\n\ndef _align(d):',start)
s=s[:start]+'''def _checks(table, lab):
    ids, x, y = map(np.asarray, (table['lab_id'], table['cmd_stress'], table['true_stress']))
    m = (ids == lab) & np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _reliable(x, y, line):
    m = abs(y - np.polyval(line, x)) < 0.0007
    return m.sum() >= max(3, len(x)//2 + 1) and np.ptp(x[m]) >= 0.12


def _archive_calibration(labs):
    """Recover per-lab priors from archived JSON load-cell tables, if mounted."""
    prior = {}
    for path in sorted(Path('/tmp/hegel_continual_archive').rglob('*.json')):
        try:
            entry = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue
        entry = entry.get('dataset', entry)
        if not isinstance(entry, dict):
            continue
        table = entry.get('calibration_load_cells', entry)
        if not isinstance(table, dict) or not {'lab_id', 'cmd_stress', 'true_stress'} <= table.keys():
            continue
        for l in labs:
            x, y = _checks(table, l)
            fit = _affine(x, y, (1., 0.))
            if _reliable(x, y, fit):
                prior.setdefault(l, []).append(fit)
    return {l: np.median(fits, axis=0) for l, fits in prior.items()}


def _calibration(d):
    c = d['calibration_load_cells']
    return np.array([_affine(*_checks(c, l), (1., 0.)) for l in range(4)])
'''+s[end:]
pos=s.index('\ndef solve_alpha_w02_s00(')
s=s[:pos]+'''
def _sparse_calibration(d, cal, x, complete):
    tr = d['train']
    labs, dt, t = tr['lab_id'], d['past_dt'], d['eval_times']
    checks = [_checks(d['calibration_load_cells'], l) for l in range(4)]
    trusted = np.array([_reliable(q, v, c) for (q, v), c in zip(checks, cal)])
    sparse = np.flatnonzero(~trusted)
    if not len(sparse):
        return cal
    priors = _archive_calibration(sparse)
    use = trusted[labs] & complete
    y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
    r = tr['instant_rate_obs']
    if use.sum() >= 12:
        h, p = _state_inputs(tr, cal)
        tau, w = _fit(h[use], p[use], dt, t, x[use], y[use], r[use])
        bc, br = _basis(tau, h[use], p[use], dt, t, x[use])
        sc = _noise((bc @ w - y[use])[np.isfinite(y[use])])
        sr = _noise((br @ w - r[use])[np.isfinite(r[use])])
    for l in sparse:
        q, v = checks[l]
        gain, bias = priors.get(l, (1., 0.))
        j = (labs == l) & complete
        if l not in priors and use.sum() >= 12 and j.sum() >= 8 and len(q):
            h, p = tr['past_stress_cmd'][j], tr['stress_hold_cmd'][j]
            bc, br = _basis(tau, h, p, dt, t, x[j])
            ec, er = _basis(tau, np.ones_like(h), np.ones_like(p), dt, t, x[j])
            drift = w[2] + x[j] @ w[3:]
            ac, ar = bc[:, :, :2] @ w[:2], br[:, :2] @ w[:2]
            ec, er = ec[:, :, :2] @ w[:2], er[:, :2] @ w[:2]
            # Anchor the offset to a check; separate lab feature drift from load gain.
            k = np.argmin(abs(v - gain*q - bias))
            C = np.column_stack(((ac-q[k]*ec).ravel(), np.tile(t, j.sum()),
                                 (drift[:, None]*t).ravel()))
            R = np.column_stack((ar-q[k]*er, np.ones(j.sum()), drift))
            A = np.vstack((C/sc, R/sr))
            target = np.r_[((y[j]-drift[:, None]*t-v[k]*ec)/sc).ravel(),
                           (r[j]-drift-v[k]*er)/sr]
            m = np.isfinite(target)
            fit = least_squares(lambda z: A[m] @ z-target[m], [gain, 0., 0.],
                                loss='cauchy', f_scale=2.)
            gain = fit.x[0]
        if len(q) == 2 and np.ptp(q) >= 0.12 and abs(cal[l, 0]-gain) < 0.015:
            continue
        if len(q):
            offsets = v-gain*q
            center = np.median(offsets)
            close = abs(offsets-center) < 0.003
            bias = (np.median(offsets[close]) if close.sum() > len(q)/2
                    else offsets[np.argmin(abs(offsets-bias))])
        cal[l] = gain, bias
    return cal

'''+s[pos:]
s=s.replace("    x, xt, complete = _align(dataset)\n", "    x, xt, complete = _align(dataset)\n    cal = _sparse_calibration(dataset, cal, x, complete)\n")
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','alpha_w02_s00_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open('public_alpha_w02_s00.pkl','rb'))[0]['dataset']; rng=np.random.default_rng(87)
for wing in ([0,1],[2,3]):
 for count in (1,2,4):
  b=copy.deepcopy(d); c=b['calibration_load_cells']; keep=np.ones(len(c['lab_id']),bool); truth=s._calibration(d)
  for l in wing:
   a,e=rng.uniform(.88,1.12),rng.uniform(-.12,.12); truth[l]=a,e
   for k in ('train','test'):
    z=b[k]; j=z['lab_id']==l
    for field in ('past_stress_cmd','stress_hold_cmd'): z[field][j]=(z[field][j]-e)/a
   j=np.flatnonzero(c['lab_id']==l); chosen=j[np.argsort(c['cmd_stress'][j])][np.round(np.linspace(0,len(j)-1,count)).astype(int)]; keep[j]=False; keep[chosen]=True
   c['cmd_stress'][j]=(c['cmd_stress'][j]-e)/a
   if count==4: c['true_stress'][chosen[1]]+=.15
  b['calibration_load_cells']={k:v[keep] for k,v in c.items()}; x,_,m=s._align(b); baseline=s._calibration(b); fit=s._sparse_calibration(b,baseline.copy(),x,m)
  def error(q):
   e=abs(q-truth); return np.max(2.5*e[:,0]+e[:,1])
  print('wing',wing,'checks/lab',count,'max stress error baseline -> candidate',round(error(baseline),6),round(error(fit),6)); assert error(fit)<.02
PY
```
- **Container Exit Code**: `0` (`1085 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w02_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
wing [0, 1] checks/lab 1 max stress error baseline -> candidate 0.372354 0.002479
wing [0, 1] checks/lab 2 max stress error baseline -> candidate 0.000171 0.000171
wing [0, 1] checks/lab 4 max stress error baseline -> candidate 0.000181 0.000181
wing [2, 3] checks/lab 1 max stress error baseline -> candidate 0.404295 0.005559
wing [2, 3] checks/lab 2 max stress error baseline -> candidate 0.000136 0.000136
wing [2, 3] checks/lab 4 max stress error baseline -> candidate 0.000118 0.000118

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5cb29d92c2f929e9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_5cb29d92 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`25908 ms` LLM)
- **Rationale**: Check whether the solver discards an identifiable two-point fit when neither wing is trusted, or lets an isolated EMI check overwrite reliable archive calibration. If either failure occurs, correct only that decision rule.
- **Alternatives Weighed**: `Submit after the per-wing stress tests` (rejected: Those tests always left a trusted wing and never exercised the requested archive path.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'alpha_w02_s00_solver.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
with open('public_alpha_w02_s00.pkl', 'rb') as f: d = pickle.load(f)[0]['dataset']
truth = np.array([[1.08,.06],[.92,-.05],[1.11,-.08],[.89,.09]])
def batch(count):
    b = copy.deepcopy(d)
    c = b['calibration_load_cells']; keep = np.zeros(len(c['lab_id']), bool)
    for l,(a,e) in enumerate(truth):
        for split in ('train','test'):
            q = b[split]; j = q['lab_id']==l
            for k in ('past_stress_cmd','stress_hold_cmd'): q[k][j] = (q[k][j]-e)/a
        j = np.flatnonzero(c['lab_id']==l); j = j[np.argsort(c['cmd_stress'][j])]
        keep[j[np.round(np.linspace(0,len(j)-1,count)).astype(int)]] = True
        c['cmd_stress'][j] = (c['cmd_stress'][j]-e)/a
    b['calibration_load_cells'] = {k:v[keep] for k,v in c.items()}
    return b

def error(b):
    x,_,m = s._align(b); cal = s._sparse_calibration(b,s._calibration(b),x,m)
    return float(np.max(np.abs(cal-truth) @ [2.5,1.]))

print('both wings, two clean checks per lab, maximum stress error:', error(batch(2)))
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp)
    a = batch(10)['calibration_load_cells']
    (path/'calibration.json').write_text(json.dumps({'calibration_load_cells': {k:v.tolist() for k,v in a.items()}}))
    s.Path = lambda _: path
    print('archive plus one clean check per lab, maximum stress error:', error(batch(1)))
    b = batch(1); b['calibration_load_cells']['true_stress'][0] += .15
    print('archive plus one EMI-corrupted check, maximum stress error:', error(b))
PY
```
- **Container Exit Code**: `0` (`520 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
both wings, two clean checks per lab, maximum stress error: 0.47745256621393645
archive plus one clean check per lab, maximum stress error: 0.00019968104382225849
archive plus one EMI-corrupted check, maximum stress error: 0.15003577502206483

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5cb29d92c2f929e9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_5cb29d92 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`39757 ms` LLM)
- **Rationale**: Preserve the measured two-point line when no independent gain estimate is available, and prevent an isolated conflicting check from overriding an archive prior. Verify the exact solver in a clean process, including how calibration changes propagate to both prediction outputs.
- **Alternatives Weighed**: `Change the hereditary fit` (rejected: Both regressions came from calibration decision rules; neither implicated the two-branch creep law.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_w02_s00_solver.py')
s = p.read_text()
needle = "        j = (labs == l) & complete\n"
replacement = """        j = (labs == l) & complete
        if l in priors:
            # A lone conflicting check cannot distinguish EMI from calibration drift.
            agrees = np.abs(v - (gain*q + bias)) < 0.003
            if agrees.sum() >= 2 and np.ptp(q[agrees]) >= 0.12:
                gain, bias = _affine(q[agrees], v[agrees], (gain, bias))
            elif agrees.any():
                bias = np.median(v[agrees] - gain*q[agrees])
            cal[l] = gain, bias
            continue
        if len(q) == 2 and np.ptp(q) >= 0.12 and (use.sum() < 12 or j.sum() < 8):
            # With no independent gain estimate, retain the identified two-point line.
            cal[l] = np.polyfit(q, v, 1)
            continue
"""
assert s.count(needle) == 1
s = s.replace(needle, replacement).replace('if l not in priors and use.sum() >= 12 and j.sum() >= 8 and len(q):', 'if use.sum() >= 12 and j.sum() >= 8 and len(q):')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
cat > /tmp/alpha_sparse_verify.py <<'PY'
import copy, importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np
root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root/'alpha_w02_s00_solver.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
with (root/'public_alpha_w02_s00.pkl').open('rb') as f: case = pickle.load(f)[0]
d, g = case['dataset'], case['gt']
o0 = s.solve_alpha_w02_s00(d)
truth = np.array([[1.08,.06],[.92,-.05],[1.11,-.08],[.89,.09]])
def batch(labs, count, glitch=False):
    b = copy.deepcopy(d)
    c = b['calibration_load_cells']; keep = np.ones(len(c['lab_id']), bool)
    target = s._calibration(d)
    for l in labs:
        a,e = truth[l]; target[l] = a,e
        for split in ('train','test'):
            q = b[split]; j = q['lab_id']==l
            for k in ('past_stress_cmd','stress_hold_cmd'): q[k][j] = (q[k][j]-e)/a
        j = np.flatnonzero(c['lab_id']==l); j = j[np.argsort(c['cmd_stress'][j])]
        chosen = j[np.round(np.linspace(0,len(j)-1,count)).astype(int)]
        keep[j] = False; keep[chosen] = True
        c['cmd_stress'][j] = (c['cmd_stress'][j]-e)/a
        if glitch: c['true_stress'][chosen[count//2]] += .15
    b['calibration_load_cells'] = {k:v[keep] for k,v in c.items()}
    return b,target

def check(label, b, target, tol):
    x,_,m = s._align(b); cal = s._sparse_calibration(b,s._calibration(b),x,m)
    stress = float(np.max(np.abs(cal-target) @ [2.5,1.]))
    out = s.solve_alpha_w02_s00(b)
    changes = [np.max(np.abs(out[k]-o0[k])) for k in ('predicted_strain','initial_rate_pred')]
    print(label, 'stress error', round(stress,6), 'max curve/rate changes', np.round(changes,6))
    assert stress < tol and max(changes) < .03
    assert .2 <= out['estimated_taus'][0] < out['estimated_taus'][1] <= 8
    np.testing.assert_array_equal(out['viscoplastic_weights'], 0)

check('both wings, two clean checks', *batch(range(4),2), .001)
for wing in ((0,1),(2,3)):
    for n, glitch in ((1,False),(2,False),(4,True)):
        check((wing,n,glitch), *batch(wing,n,glitch), .02)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp)
    a = batch(range(4),10)[0]['calibration_load_cells']
    (path/'calibration.json').write_text(json.dumps({'calibration_load_cells': {k:v.tolist() for k,v in a.items()}}))
    s.Path = lambda _: path
    for glitch in (False,True):
        check(('both wings with archive, one check',glitch), *batch(range(4),1,glitch), .001)
print('Public tau absolute errors:', np.round(np.abs(o0['estimated_taus']-g['taus']),6))
print('Public maximum hereditary coefficient error:', round(float(np.max(np.abs(o0['hereditary_weights']-g['theta']))),6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/alpha_sparse_verify.py
```
- **Container Exit Code**: `0` (`1863 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w02_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
both wings, two clean checks stress error 0.00018 max curve/rate changes [0.00011  0.000134]
((0, 1), 1, False) stress error 0.002453 max curve/rate changes [0.001539 0.002116]
((0, 1), 2, False) stress error 0.00018 max curve/rate changes [0.000193 0.000251]
((0, 1), 4, True) stress error 0.000157 max curve/rate changes [0.000197 0.000257]
((2, 3), 1, False) stress error 0.006511 max curve/rate changes [0.004483 0.005245]
((2, 3), 2, False) stress error 0.000143 max curve/rate changes [8.60e-05 1.23e-04]
((2, 3), 4, True) stress error 0.000154 max curve/rate changes [6.7e-05 9.1e-05]
('both wings with archive, one check', False) stress error 0.0002 max curve/rate changes [6.e-06 2.e-06]
('both wings with archive, one check', True) stress error 0.000196 max curve/rate changes [0. 0.]
Public tau absolute errors: [0.000807 0.037058]
Public maximum hereditary coefficient error: 0.002211

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7b726f0a7cea2e89 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_7b726f0a (directory).
```

### Turn 12 — `submit_completed_task` (`12307 ms` LLM)
- **Rationale**: The final artifact passed the public schema validator and both-wing calibration tests, including a corrupted spot check with an archive prior. The dense-check prediction path was unchanged from the candidate that passed the independent recurrence and initial-rate checks.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/alpha_w02_s00_solver.py': 'FILE:10298'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_w02_s00_solver.py`. The public schema validator passed in an isolated Python process. Across the sparse and noisy calibration stress tests, the largest changes from the dense-check baseline were 0.004483 for strain and 0.005245 for initial rate. Archive fallback was verified with a test fixture; no live archive was mounted.`

### Turn 13 — `execute_in_remote_docker` (`29859 ms` LLM)
- **Rationale**: Test whether the fallback loses a clean two-of-three calibration consensus when EMI hits a sparse batch. Compare the feature law with an intercept-only baseline within each lab to detect a pooled-only signal that would warrant changing the fit.
- **Alternatives Weighed**: `Add rational terms, relay modes, or classification checks` (rejected: The specified recurrence and initial derivative already match their independent checks, and the data have neither relay modes nor classification labels.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
from itertools import combinations
exec(open('/tmp/alpha_sparse_verify.py').read().split("check('both wings, two clean checks'")[0])
# Reuse measured coupon responses; change only the calibration coordinates and which checks survive.
for wing in ((0,1),(2,3)):
    b, target = batch(wing,10)
    checks = b['calibration_load_cells']; ids = checks['lab_id']
    x, _, m = s._align(b)
    worst = {2:(0,None),3:(0,None)}
    for n in worst:
        for mode in ('wide','mid','narrow'):
            keep = ~np.isin(ids,wing)
            picks = []
            for l in wing:
                j = np.flatnonzero(ids==l); j = j[np.argsort(checks['cmd_stress'][j])]
                ranks = {'wide':[0,5,9], 'mid':[2,5,7], 'narrow':[3,4,5]}[mode]
                if n==2: ranks = ranks[::2]
                idx = j[ranks]; keep[idx]=True; picks.append(idx)
            for ix in range(n):
                for sign in (-1,1):
                    c = {k:v[keep].copy() for k,v in checks.items()}
                    for j in picks: c['true_stress'][np.sum(keep[:j[ix]])] += sign*.15
                    q = {**b,'calibration_load_cells':c}
                    cal = s._sparse_calibration(q,s._calibration(q),x,m)
                    err = float(np.max(np.abs(cal-target) @ [2.5,1.]))
                    if err>worst[n][0]: worst[n]=(err,(mode,ix,sign))
    print('wing',wing,'worst sparse EMI stress errors and witnesses',worst)
# Check that a measured within-lab signal survives lab centering; no classification labels exist here.
tr=d['train']; x,_,m=s._align(d); h,p=s._state_inputs(tr,s._calibration(d)); bc,br=s._basis(o0['estimated_taus'],h,p,d['past_dt'],d['eval_times'],x)
w=o0['hereditary_weights']; actual=tr['instant_rate_obs']-br[:,:2]@w[:2]; drift=br[:,2:]@w[2:]
for l in range(4):
    j=m & (tr['lab_id']==l); a,f=actual[j],drift[j]; a=a-np.median(a); f=f-np.median(f)
    print('lab',l,'centered instantaneous drift median error, intercept-only vs feature law',np.round([np.median(abs(a)),np.median(abs(a-f))],6))
PY
```
- **Container Exit Code**: `0` (`3401 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
wing (0, 1) worst sparse EMI stress errors and witnesses {2: (0.0024526593135789226, ('wide', 1, -1)), 3: (0.49923577897804855, ('narrow', 2, -1))}
wing (2, 3) worst sparse EMI stress errors and witnesses {2: (0.15822545317945763, ('wide', 1, 1)), 3: (0.15822507119251633, ('wide', 2, 1))}
lab 0 centered instantaneous drift median error, intercept-only vs feature law [0.159046 0.021311]
lab 1 centered instantaneous drift median error, intercept-only vs feature law [0.18326  0.009447]
lab 2 centered instantaneous drift median error, intercept-only vs feature law [0.146998 0.011052]
lab 3 centered instantaneous drift median error, intercept-only vs feature law [0.20096  0.007213]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7b726f0a7cea2e89 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_7b726f0a (directory).
```

### Turn 14 — `execute_in_remote_docker` (`42524 ms` LLM)
- **Rationale**: Separate slope estimation from spot-check selection so one EMI hit cannot set both calibration coefficients. For incompatible pairs without stronger evidence, the midpoint bounds the offset ambiguity. Replay the failed EMI slices and rerun the earlier calibration tests to catch regressions.
- **Alternatives Weighed**: `Change the creep law or remove the feature term` (rejected: The recurrence and initial derivative passed independent checks, and the feature law reduced centered rate error in every lab.)

```bash
set -e
cd /holdout_task
cp alpha_w02_s00_solver.py /tmp/alpha_w02_v03.py
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_w02_s00_solver.py')
s = p.read_text()
a = s.index('            # Anchor the offset to a check;')
b = s.index('        cal[l] = gain, bias\n    return cal', a)
s = s[:a] + '''            # A constant stress offset contributes the same hereditary curve
            # to every coupon in this lab. Profile it separately from gain.
            C = np.column_stack((ac.ravel(), ec.ravel(), np.tile(t, j.sum()),
                                 (drift[:, None]*t).ravel()))
            R = np.column_stack((ar, er, np.ones(j.sum()), drift))
            A = np.vstack((C/sc, R/sr))
            target = np.r_[((y[j]-drift[:, None]*t)/sc).ravel(), (r[j]-drift)/sr]
            m = np.isfinite(target)
            fit = least_squares(lambda z: A[m] @ z-target[m], [gain, bias, 0., 0.],
                                loss='cauchy', f_scale=2.)
            gain = fit.x[0]
        if len(q) >= 2:
            i, k = np.triu_indices(len(q), 1)
            span = q[k] - q[i]
            # Confirm a measured slope with the independent gain estimate.
            # For nearly coincident checks the response estimate is better conditioned.
            wide = abs(span) >= 0.12
            i, k, span = i[wide], k[wide], span[wide]
            if len(span):
                slopes = (v[k]-v[i])/span
                agreement = abs(slopes-gain)
                best = np.argmin(agreement)
                if agreement[best] < 0.015:
                    a, b = i[best], k[best]
                    cal[l] = slopes[best], v[a]-slopes[best]*q[a]
                    continue
        if len(q):
            # For three checks this preserves a clean majority even when
            # no pair spans enough command range to estimate a precise slope.
            # For two incompatible checks, the midpoint bounds the ambiguity.
            bias = np.median(v-gain*q)
''' + s[b:]
p.write_text(s)
PY
cat > /tmp/alpha_sparse_emi.py <<'PY'
exec(open('/tmp/alpha_sparse_verify.py').read().split("check('both wings, two clean checks'")[0])
worst = {2: (0., None), 3: (0., None)}
for wing in ((0,1),(2,3)):
    b, target = batch(wing,10)
    checks = b['calibration_load_cells']; ids = checks['lab_id']
    x, _, m = s._align(b)
    for n in worst:
        for mode, ranks in [('wide',[0,5,9]), ('mid',[2,5,7]), ('narrow',[3,4,5])]:
            keep = ~np.isin(ids,wing); picks = []
            for l in wing:
                j = np.flatnonzero(ids==l); j = j[np.argsort(checks['cmd_stress'][j])]
                selected = j[ranks if n==3 else ranks[::2]]
                keep[selected] = True; picks.append(selected)
            for ix in range(n):
                for sign in (-1,1):
                    c = {k:v[keep].copy() for k,v in checks.items()}
                    for j in picks: c['true_stress'][np.sum(keep[:j[ix]])] += sign*.15
                    q = {**b,'calibration_load_cells':c}
                    cal = s._sparse_calibration(q,s._calibration(q),x,m)
                    err = float(np.max(np.abs(cal-target) @ [2.5,1.]))
                    if err>worst[n][0]: worst[n]=(err,(wing,mode,ix,sign))
print('Worst sparse EMI errors and witnesses:', worst)
assert worst[3][0] < .025 and worst[2][0] < .095
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/alpha_sparse_verify.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/alpha_sparse_emi.py
```
- **Container Exit Code**: `1` (`4784 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w02_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
both wings, two clean checks stress error 0.00018 max curve/rate changes [0.00011  0.000134]
((0, 1), 1, False) stress error 0.002389 max curve/rate changes [0.001491 0.002063]
((0, 1), 2, False) stress error 0.00018 max curve/rate changes [0.000193 0.000251]
((0, 1), 4, True) stress error 0.000157 max curve/rate changes [0.000197 0.000257]
((2, 3), 1, False) stress error 0.006364 max curve/rate changes [0.004424 0.005175]
((2, 3), 2, False) stress error 0.000143 max curve/rate changes [8.60e-05 1.23e-04]
((2, 3), 4, True) stress error 0.000154 max curve/rate changes [6.7e-05 9.1e-05]
('both wings with archive, one check', False) stress error 0.0002 max curve/rate changes [6.e-06 2.e-06]
('both wings with archive, one check', True) stress error 0.000196 max curve/rate changes [0. 0.]
Public tau absolute errors: [0.000807 0.037058]
Public maximum hereditary coefficient error: 0.002211
Worst sparse EMI errors and witnesses: {2: (0.0805031479352523, ((2, 3), 'mid', 0, 1)), 3: (0.49923577897804855, ((0, 1), 'narrow', 2, -1))}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bdb31269beb88375 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_bdb31269 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/alpha_sparse_emi.py", line 23, in <module>
    assert worst[3][0] < .025 and worst[2][0] < .095
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

### Turn 15 — `execute_in_remote_docker` (`21312 ms` LLM)
- **Rationale**: When denser checks identify the material response, route three-check labs through the independent gain check rather than trusting their residual consensus alone. Replay the narrow EMI witness and the earlier clean and archive cases to test the routing change.
- **Alternatives Weighed**: `Retune the sparse offset estimator` (rejected: The unchanged three-check failure after replacing that estimator suggests the affected lab bypassed the fallback.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_w02_s00_solver.py')
s = p.read_text()
old = '    trusted = np.array([_reliable(q, v, c) for (q, v), c in zip(checks, cal)])'
new = '''    # Three apparent inliers can still be a bad line when two commands
    # cluster together and a corrupted third check supplies the leverage.
    trusted = np.array([len(q) >= 4 and _reliable(q, v, c)
                        for (q, v), c in zip(checks, cal)])
    if not trusted.any():
        # If the entire batch is sparse, retain identifiable three-point fits.
        trusted = np.array([_reliable(q, v, c) for (q, v), c in zip(checks, cal)])'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
python3 -I /tmp/alpha_sparse_verify.py
python3 -I /tmp/alpha_sparse_emi.py
```
- **Container Exit Code**: `0` (`4615 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_w02_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
both wings, two clean checks stress error 0.00018 max curve/rate changes [0.00011  0.000134]
((0, 1), 1, False) stress error 0.002389 max curve/rate changes [0.001491 0.002063]
((0, 1), 2, False) stress error 0.00018 max curve/rate changes [0.000193 0.000251]
((0, 1), 4, True) stress error 0.000157 max curve/rate changes [0.000197 0.000257]
((2, 3), 1, False) stress error 0.006364 max curve/rate changes [0.004424 0.005175]
((2, 3), 2, False) stress error 0.000143 max curve/rate changes [8.60e-05 1.23e-04]
((2, 3), 4, True) stress error 0.000154 max curve/rate changes [6.7e-05 9.1e-05]
('both wings with archive, one check', False) stress error 0.0002 max curve/rate changes [6.e-06 2.e-06]
('both wings with archive, one check', True) stress error 0.000196 max curve/rate changes [0. 0.]
Public tau absolute errors: [0.000807 0.037058]
Public maximum hereditary coefficient error: 0.002211
Worst sparse EMI errors and witnesses: {2: (0.0805031479352523, ((2, 3), 'mid', 0, 1)), 3: (0.004624144137159461, ((2, 3), 'narrow', 2, 1))}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f9fb5c9502a5f6ea (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v05_f9fb5c95 (directory).
```

### Turn 16 — `submit_completed_task` (`12472 ms` LLM)
- **Rationale**: The verified candidate reduced worst three-check EMI error from 0.499236 to 0.004624 while retaining the two-check improvement and passing the public schema and prior calibration cases. The remaining two-check ambiguity cannot be resolved from the conflicting spot checks alone.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_w02_s00_solver.py` and verified it in an isolated Python process. The public schema and both-wing calibration stress tests passed. With one corrupted spot check, the worst stress error was 0.004624 for three checks and 0.080503 for two. Archive fallback was verified with a fixture; no live archive was mounted.`

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
