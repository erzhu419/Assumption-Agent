# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c03_s00__rheology_campaign_baseline`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c03_source` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `295,859` (`284,233` / `35,840` / `11,626`) |
| **Total Episode Wall Time (s)** | `373.54s` |

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

### Turn 1 — `execute_in_remote_docker` (`20897 ms` LLM)
- **Rationale**: Inspect the validator and a few field samples to catch schema assumptions that would change calibration, alignment, or the fitting design. Check whether reusable numerical primitives are mounted.

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && python3 - <<'PY'
import pickle, numpy as np
with open('public_alpha_m97_c03_s00.pkl','rb') as f: d=pickle.load(f)
for k,v in d.items():
 print(k, type(v).__name__, getattr(v,'shape',None), getattr(v,'dtype',None))
 if isinstance(v,dict):
  for a,b in v.items(): print(' ',a,type(b).__name__,getattr(b,'shape',None),repr(b)[:150])
 elif isinstance(v,np.ndarray): print(' ',repr(v[:3] if v.ndim else v)[:500])
 else: print(' ',repr(v)[:500])
import scipy; print('scipy',scipy.__version__)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation Cauchy robust IRLS viscoelastic relaxation'
```
- **Container Exit Code**: `0` (`630 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
le)
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`19670 ms` LLM)
- **Rationale**: The validator identifies the dataset as a record inside a list. Inspect its fields to determine panel axes and missing-value handling, and measure per-channel calibration residuals to see whether the archive fallback is warranted.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle, numpy as np, scipy
with open('public_alpha_m97_c03_s00.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
print('scipy',scipy.__version__)
for k,v in d.items():
 print(k,type(v).__name__,getattr(v,'shape',None),repr(v)[:200] if not isinstance(v,dict) else '')
 if isinstance(v,dict):
  for a,b in v.items():
   print(' ',a,type(b).__name__,getattr(b,'shape',None), np.sum(~np.isfinite(b)) if isinstance(b,np.ndarray) and np.issubdtype(b.dtype,np.number) else '',repr(b[:2] if isinstance(b,np.ndarray) else b)[:230])
c=d['calibration_load_cells']
for lab in np.unique(c['lab_id']):
 m=c['lab_id']==lab; x=c['cmd_stress'][m]; y=c['true_stress'][m]; p=np.polyfit(x,y,1); print('calibration',lab,len(x),p,np.round(y-np.polyval(p,x),5))
PY
for f in autogenous_prim_reference_panel_balancing.align_reference_panel autogenous_prim_reference_panel_balancing.weighted_column_fill autogenous_prim_robust_trend_decomposition.cauchy_linear_fit; do PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe "$f"; done
```
- **Container Exit Code**: `0` (`689 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_robust_trend_decomposition",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`60161 ms` LLM)
- **Rationale**: Build a runnable candidate using the exact memory recurrence and its analytical time-zero derivative. Run the public schema check and inspect calibration output and archive structure to determine whether the documented fallback applies.
- **Alternatives Weighed**: `Use the bundled panel-alignment helper unchanged` (rejected: The task specifies median column statistics and cell-specific imputation; implementing those directly keeps the convention explicit.)

```bash
cd /holdout_task && cat > /tmp/rheology_candidate.py <<'PY'
import numpy as np
from scipy.optimize import least_squares


def _calibration(cal):
    gains = np.empty(4)
    biases = np.empty(4)
    use_archive = []
    for c in range(4):
        take = (cal['lab_id'] == c) & np.isfinite(cal['cmd_stress']) & np.isfinite(cal['true_stress'])
        x, y = cal['cmd_stress'][take], cal['true_stress'][take]
        if x.size < 2:
            use_archive.append(c)
            continue
        i, j = np.triu_indices(x.size, 1)
        distinct = x[j] != x[i]
        i, j = i[distinct], j[distinct]
        g = np.median((y[j] - y[i]) / (x[j] - x[i]))
        b = np.median(y - g*x)
        r = y - (g*x+b)
        clean = np.abs(r) <= 3.0
        if not clean.all():
            use_archive.append(c)
        gains[c], biases[c] = np.linalg.lstsq(np.column_stack((x[clean], np.ones(clean.sum()))), y[clean], rcond=None)[0]
    return gains, biases, use_archive


def _align(train, test):
    labs, strata, raw, panel = (np.concatenate((train[k], test[k])) for k in ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        mask = labs == lab
        controls = panel[mask]
        center = np.nanmedian(np.nanmean(controls, axis=0))
        spread = np.nanmedian(np.nanstd(controls, axis=0, ddof=0))
        aligned[mask] = (raw[mask]-center)/spread
    # Compute each cell mean from observed entries before filling any row.
    cells = [(labs == lab) & (strata == s) for lab, s in np.unique(np.column_stack((labs, strata)), axis=0)]
    means = []
    for mask in cells:
        v = aligned[mask]
        n = np.isfinite(v).sum(axis=0)
        means.append(np.divide(np.nansum(v, axis=0), n, out=np.full(v.shape[1], np.nan), where=n > 0))
    means = np.asarray(means)
    for idx, mask in enumerate(cells):
        lab = labs[np.flatnonzero(mask)[0]]
        same_lab = np.array([labs[np.flatnonzero(m)[0]] == lab for m in cells])
        fill = means[idx].copy()
        for j in np.flatnonzero(~np.isfinite(fill)):
            v = means[same_lab, j]
            v = v[np.isfinite(v)]
            fill[j] = v.mean() if v.size else np.nanmean(means[:, j])
        aligned[mask] = np.where(np.isfinite(aligned[mask]), aligned[mask], fill)
    return aligned


def _design(logtau, stress, hold, dt, times, x):
    tau = np.exp(logtau)
    edges = np.r_[0., np.cumsum(dt)]
    remaining = edges[-1] - edges
    u = remaining[:, None]/tau
    e = np.exp(-u)
    memory = stress @ np.diff(e, axis=0)
    dmemory = stress @ np.diff(e*u, axis=0)
    gap = hold[:, None] - memory
    v = times[:, None]/tau
    ev = np.exp(-v)
    phi = -np.expm1(-v)
    n, k = len(x), len(times)
    a = np.empty((n,k+1,7))
    a[:,:k,:2] = gap[:,None,:]*phi
    a[:,k,:2] = gap/tau
    xd = np.column_stack((np.ones(n), x))
    a[:,:k,2:] = xd[:,None,:]*times[None,:,None]
    a[:,k,2:] = xd
    d = np.empty((n,k+1,2))
    d[:,:k] = -dmemory[:,None,:]*phi - gap[:,None,:]*v*ev
    d[:,k] = (-dmemory-gap)/tau
    return a, d


def _fit(stress, hold, dt, times, x, y):
    def get(p):
        return _design(p[:2], stress, hold, dt, times, x)

    def optimize(p, scale, loss):
        def fun(p):
            a, _ = get(p)
            return ((a @ p[2:] - y)/scale).ravel()

        def jac(p):
            a, d = get(p)
            j = np.concatenate((d*p[2:4], a), axis=2)
            return (j/scale[..., None]).reshape(-1, 9)

        low = np.r_[np.log(min(dt.min(), times.min())/30)*np.ones(2), np.full(7,-np.inf)]
        high = np.r_[np.log((dt.sum()+times.max())*15)*np.ones(2), np.full(7,np.inf)]
        return least_squares(fun, p, jac=jac, bounds=(low, high), loss=loss, x_scale='jac', max_nfev=150)

    best = None
    for pair in ((.35, 2.3), (.7, 3.8), (.18, 1.25)):
        logtau = np.log(pair)
        design, _ = _design(logtau, stress, hold, dt, times, x)
        w = np.linalg.lstsq(design.reshape(-1,7), y.ravel(), rcond=None)[0]
        fit = optimize(np.r_[logtau, w], np.asarray(.09), 'soft_l1')
        if best is None or fit.cost < best.cost:
            best = fit
    p = best.x
    for _ in range(2):
        a, _ = get(p)
        r = a @ p[2:] - y
        def mad(v):
            return max(1e-6, 1.4826*np.median(np.abs(v-np.median(v))))
        scale = 2.385*np.r_[np.full(len(times), mad(r[:,:-1])), mad(r[:,-1])]
        p = optimize(p, scale, 'cauchy').x
    order = np.argsort(p[:2])
    return p[:2][order], np.r_[p[2:4][order], p[4:]]


def solve_alpha_m97_c03_s00(dataset: dict) -> dict:
    tr, te = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    gains, biases, _ = _calibration(dataset['calibration_load_cells'])
    x = _align(tr, te)
    def stresses(part):
        g, b = gains[part['lab_id']], biases[part['lab_id']]
        return part['past_stress_cmd']*g[:,None]+b[:,None], part['stress_hold_cmd']*g+b
    n = len(tr['lab_id'])
    y = np.column_stack((tr['strain_curves_obs']-tr['strain_0'][:,None], tr['instant_rate_obs']))
    logtau, w = _fit(*stresses(tr), dt, times, x[:n], y)
    a, _ = _design(logtau, *stresses(te), dt, times, x[n:])
    pred = a @ w
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': w, 'regime_mode_weights': np.zeros(2), 'predicted_strain': pred[:,:-1]+te['strain_0'][:,None], 'initial_rate_pred': pred[:,-1]}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py /tmp/rheology_candidate.py
ls -l /tmp/hegel_continual_archive/ 2>/dev/null || true
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp python3 - <<'PY'
import pickle,numpy as np,json,pathlib
import rheology_candidate as m
d=pickle.load(open('public_alpha_m97_c03_s00.pkl','rb'))[0]['dataset']
print('calibration',m._calibration(d['calibration_load_cells']))
o=m.solve_alpha_m97_c03_s00(d)
print('taus',o['estimated_taus'],'weights',o['hereditary_weights'])
p=pathlib.Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
if p.exists():
 a=json.loads(p.read_text()); print('archive structure',type(a).__name__,str(a)[:1800])
PY
```
- **Container Exit Code**: `0` (`762 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
calibration (array([0.99999623, 0.99998781, 1.00004377, 1.00004233]), array([ 1.33276965e-05, -2.47202936e-05,  1.85681540e-05, -8.41290837e-05]), [])
taus [0.59954895 4.07257021] weights [ 0.68715836  0.55404065  0.1395066   0.16865993 -0.1394451   0.11899013
 -0.08975908]
```

### Turn 4 — `execute_in_remote_docker` (`26937 ms` LLM)
- **Rationale**: Leave each lab’s response measurements out while retaining the specified unlabeled reference panels. Separate curve and rate errors by lab, stratum, and feature missingness to distinguish a fitting problem from an alignment or imputation problem.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp python3 - <<'PY'
import pickle,numpy as np
import rheology_candidate as m
d=pickle.load(open('public_alpha_m97_c03_s00.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); dt,t=d['past_dt'],d['eval_times']; g,b,_=m._calibration(d['calibration_load_cells']); lab=tr['lab_id']; h=tr['stress_hold_cmd']*g[lab]+b[lab]; s=tr['past_stress_cmd']*g[lab,None]+b[lab,None]; x=m._align(tr,te)[:n]; y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; missing=np.isnan(tr['coupon_features_raw']).any(axis=1)
print('max physical past stress',np.max(np.abs(s)),'limit',d['crit_stress'])
print('panel center/spread:',[(i,round(float(np.nanmedian(np.nanmean(np.r_[tr['ref_coupons'],te['ref_coupons']][np.r_[lab,te['lab_id']]==i],axis=0))),4),round(float(np.nanmedian(np.nanstd(np.r_[tr['ref_coupons'],te['ref_coupons']][np.r_[lab,te['lab_id']]==i],axis=0))),4)) for i in range(4)])
r=np.empty_like(y)
def report(label,rr):
 q=np.abs(rr); print(label, 'n',len(rr),'curve med/p90',np.round(np.quantile(q[:,:-1],[.5,.9]),5),'rate med/p90',np.round(np.quantile(q[:,-1],[.5,.9]),5),'rate bias',round(float(np.median(rr[:,-1])),5))
for i in range(4):
 f=lab!=i; p,w=m._fit(s[f],h[f],dt,t,x[f],y[f]); a,_=m._design(p,s[~f],h[~f],dt,t,x[~f]); r[~f]=a@w-y[~f]; report('lab '+str(i),r[~f]); print('  taus',np.exp(p),'weights',np.round(w,5))
for k in np.unique(tr['stratum']): report('stratum '+str(k),r[tr['stratum']==k])
report('feature missing',r[missing]); report('feature complete',r[~missing]); print('curve med by time',np.round(np.median(np.abs(r[:,:-1]),axis=0),5))
print('rate residual MAD',np.median(np.abs(r[:,-1]-np.median(r[:,-1]))),'curve residual MAD',np.median(np.abs(r[:,:-1]-np.median(r[:,:-1]))))
PY
```
- **Container Exit Code**: `0` (`671 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
max physical past stress 2.0671716281578143 limit 2.65
panel center/spread: [(0, -0.6521, 0.8187), (1, 0.4536, 1.2717), (2, -0.2617, 0.9389), (3, 0.8106, 1.3952)]
lab 0 n 46 curve med/p90 [0.01156 0.0913 ] rate med/p90 [0.01122 0.02447] rate bias 0.00328
  taus [0.60033002 4.13702657] weights [ 0.68907  0.55677  0.13951  0.1684  -0.13909  0.11906 -0.0897 ]
lab 1 n 46 curve med/p90 [0.01544 0.40176] rate med/p90 [0.01559 0.12348] rate bias 0.00185
  taus [0.59999762 4.11271868] weights [ 0.68822  0.55482  0.13976  0.16855 -0.13953  0.11886 -0.08986]
lab 2 n 42 curve med/p90 [0.01388 0.23458] rate med/p90 [0.01423 0.05332] rate bias 0.00104
  taus [0.59831229 4.01304986] weights [ 0.68515  0.55184  0.13943  0.1684  -0.13925  0.11871 -0.08962]
lab 3 n 46 curve med/p90 [0.01731 0.25551] rate med/p90 [0.0108  0.04137] rate bias 0.00075
  taus [0.59962112 4.03314719] weights [ 0.68641  0.55283  0.13943  0.16924 -0.13985  0.11942 -0.08989]
stratum 0 n 60 curve med/p90 [0.01482 0.33306] rate med/p90 [0.0132  0.09596] rate bias 0.00363
stratum 1 n 63 curve med/p90 [0.01368 0.18679] rate med/p90 [0.01339 0.03658] rate bias -0.00175
stratum 2 n 57 curve med/p90 [0.0138  0.19879] rate med/p90 [0.01032 0.04031] rate bias 0.00113
feature missing n 20 curve med/p90 [0.26673 0.87098] rate med/p90 [0.11186 0.23858] rate bias -0.03328
feature complete n 160 curve med/p90 [0.01201 0.03494] rate med/p90 [0.01128 0.02657] rate bias 0.00175
curve med by time [0.01304 0.00993 0.0135  0.01197 0.01438 0.01581 0.01504 0.01633 0.01428
 0.01878]
rate residual MAD 0.012534621645595045 curve residual MAD 0.014267472477893672
```

### Turn 5 — `execute_in_remote_docker` (`43899 ms` LLM)
- **Rationale**: An independent forward simulator with nonidentity calibration can expose errors masked by the near-identity public gains. Check both parameter recovery and noiseless predictions by lab, and inspect the documented archive path before implementing its conditional fallback.
- **Alternatives Weighed**: `Replace prescribed cell-mean imputation with a learned missing-feature model` (rejected: The observed missing-feature error identifies a limitation, but changing imputation would violate the specified model.)

```bash
cd /holdout_task && ls -ld /tmp/hegel_continual_archive /tmp/hegel_continual_archive/archive_calibration_history.json /pretrain_task 2>/dev/null; if test -r /tmp/hegel_continual_archive/archive_calibration_history.json; then head -c 1800 /tmp/hegel_continual_archive/archive_calibration_history.json; fi
cat > /tmp/rheology_stress.py <<'PY'
import importlib.util, pickle, sys
import numpy as np
spec = importlib.util.spec_from_file_location('candidate',sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_alpha_m97_c03_s00.pkl','rb') as f:
    d=pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
cal = d['calibration_load_cells']
g=np.array([.81,1.17,.93,1.09]); b=np.array([-.21,.18,.09,-.04]);
cal['true_stress']=cal['cmd_stress']*g[cal['lab_id']]+b[cal['lab_id']]
n=len(tr['lab_id']); times=d['eval_times']
lab, groups, raw, panel = (np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons'))
x=raw.copy()
for j in range(4):
    i=lab==j; c=panel[i]; x[i]=(raw[i]-np.median(np.nanmean(c,axis=0)))/np.median(np.nanstd(c,axis=0))
    for s in np.unique(groups[i]):
        rows=i & (groups==s); z=x[rows]; x[rows]=np.where(np.isnan(z),np.nanmean(z,axis=0),z)
w = np.array([.74,.61,.18,-.15,.11,.19,-.12])
def analytic(p, a, taus):
    labs=p['lab_id']; z=np.zeros((len(labs),2)); h=p['stress_hold_cmd']*g[labs]+b[labs]
    for dt, cmd in zip(d['past_dt'],p['past_stress_cmd'].T):
        e=np.exp(-dt/taus); stress=cmd*g[labs]+b[labs]; z=e*z+(1-e)*stress[:,None]
    gap=h[:,None]-z; drift=w[2]+a@w[3:]
    return p['strain_0'][:,None] + (gap*w[:2])@(1-np.exp(-times[None,:]/taus[:,None])) + drift[:,None]*times, (gap*w[:2]/taus).sum(axis=1)+drift
rng=np.random.default_rng(92)
for taus in (np.array([.25,1.4]),np.array([1.1,6.2])):
    yc,yr=analytic(tr,x[:n],taus)
    mask=rng.random(yc.shape)<.07
    tr['strain_curves_obs']=yc+rng.normal(0,.015,yc.shape)+mask*rng.standard_cauchy(yc.shape)*2
    tr['instant_rate_obs']=yr+rng.normal(0,.018,yr.shape)
    o=m.solve_alpha_m97_c03_s00(d)
    yc,yr=analytic(te,x[n:],taus)
    ec=np.abs(o['predicted_strain']-yc); er=np.abs(o['initial_rate_pred']-yr)
    ts=np.max(np.abs(o['estimated_taus']/taus-1)); ws=np.max(np.abs(o['hereditary_weights']-w));
    worst_c=max(np.quantile(ec[te['lab_id']==j],.9) for j in range(4)); worst_r=max(np.quantile(er[te['lab_id']==j],.9) for j in range(4))
    print('injected taus',taus,'fit',o['estimated_taus'],'max relative tau error',ts,'max absolute weight error',ws,'worst lab p90 curve/rate error',worst_c,worst_r)
    assert ts<.12 and ws<.07 and worst_c<.035 and worst_r<.035 and np.all(o['regime_mode_weights']==0)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_stress.py /tmp/rheology_candidate.py
```
- **Container Exit Code**: `0` (`692 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
drwxr-xr-x 2 root root 4096 Nov 23 09:31 /pretrain_task
injected taus [0.25 1.4 ] fit [0.24931043 1.38750528] max relative tau error 0.008924802708680013 max absolute weight error 0.003317007221625645 worst lab p90 curve/rate error 0.0023944755586665866 0.0009658336836010585
injected taus [1.1 6.2] fit [1.10210834 6.28233681] max relative tau error 0.013280131181387267 max absolute weight error 0.0029024629330727247 worst lab p90 curve/rate error 0.001316892507095957 0.00048753380445580867
```

### Turn 6 — `execute_in_remote_docker` (`17077 ms` LLM)
- **Rationale**: Inspect only the available calibration metadata. A documented archive layout would determine the reader; otherwise, handle missing historical calibration explicitly without changing the validated in-batch fit.
- **Alternatives Weighed**: `Promote the temporary candidate unchanged` (rejected: A channel with fewer than two usable in-batch rows currently leaves its calibration uninitialized.)

```bash
ls -la /pretrain_task; grep -inE 'archive|calibrat|EMI|gain|bias' /holdout_task/public_asset_index.json; if test -r /tmp/hegel_continual_archive/archive_calibration_history.json; then head -c 2400 /tmp/hegel_continual_archive/archive_calibration_history.json; fi
```
- **Container Exit Code**: `0` (`282 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 27 06:38 ..
8:            "calibration_load_cells": {
189:            "stress_biases": {
196:            "stress_gains": {
```

### Turn 7 — `execute_in_remote_docker` (`42421 ms` LLM)
- **Rationale**: Write the required solver and verify that it preserves clean in-batch calibration, rejects an EMI spike, and uses history for a sparse channel. The archive is absent in this container, so exercise the fallback with a temporary fixture using the indexed gain and bias field names, then rerun the schema and synthetic recovery tests on the exact deliverable.
- **Alternatives Weighed**: `Substitute identity calibration when measurements are insufficient` (rejected: Identity calibration would silently bias predictions when channel gain or bias differs from nominal.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/rheology_candidate.py')
s = p.read_text()
s = 'import json\nfrom pathlib import Path\n\n' + s
start = s.index('def _calibration')
end = s.index('def _align', start)
s = s[:start] + '''_CALIBRATION_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _historical_calibration(history, channel):
    if history is None:
        return None
    if isinstance(history, list):
        for entry in reversed(history):
            estimate = _historical_calibration(entry, channel)
            if estimate is not None:
                return estimate
        return None
    if 'stress_gains' in history and 'stress_biases' in history:
        return float(history['stress_gains'][channel]), float(history['stress_biases'][channel])
    if history.get('lab_id') == channel and 'gain' in history and 'bias' in history:
        return float(history['gain']), float(history['bias'])
    return None


def _calibration(cal):
    gains = np.empty(4)
    biases = np.empty(4)
    history_loaded = False
    history = None
    for c in range(4):
        take = (cal['lab_id'] == c) & np.isfinite(cal['cmd_stress']) & np.isfinite(cal['true_stress'])
        x, y = cal['cmd_stress'][take], cal['true_stress'][take]
        clean = np.ones(len(x), dtype=bool)
        if np.unique(x).size >= 2:
            i, j = np.triu_indices(len(x), 1)
            distinct = x[j] != x[i]
            i, j = i[distinct], j[distinct]
            slope = np.median((y[j] - y[i]) / (x[j] - x[i]))
            intercept = np.median(y - slope*x)
            clean = np.abs(y - (slope*x + intercept)) <= 3.0
        fallback = len(x) < 2 or not clean.all()
        historical = None
        if fallback:
            if not history_loaded:
                try:
                    history = json.loads(_CALIBRATION_HISTORY.read_text())
                except FileNotFoundError:
                    pass
                history_loaded = True
            historical = _historical_calibration(history, c)
        if np.unique(x[clean]).size >= 2 and (clean.sum() >= 4 or historical is None):
            gains[c], biases[c] = np.linalg.lstsq(
                np.column_stack((x[clean], np.ones(clean.sum()))), y[clean], rcond=None
            )[0]
        elif historical is not None:
            gains[c], biases[c] = historical
        else:
            raise ValueError(f'No usable load-cell calibration for channel {c}')
    return gains, biases


''' + s[end:]
s = s.replace('gains, biases, _ = _calibration', 'gains, biases = _calibration')
Path('alpha_m97_c03_s00_solver.py').write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
python3 -I /tmp/rheology_stress.py /holdout_task/alpha_m97_c03_s00_solver.py
python3 -I - <<'PY'
import importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'alpha_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('public_alpha_m97_c03_s00.pkl', 'rb') as f:
    cal = pickle.load(f)[0]['dataset']['calibration_load_cells']
lab, x = cal['lab_id'], cal['cmd_stress']
g = np.array([.82, 1.13, .97, 1.06]); b = np.array([.11, -.06, .07, .04]); y = g[lab]*x+b[lab]
with tempfile.TemporaryDirectory() as tmp:
    m._CALIBRATION_HISTORY = Path(tmp)/'archive.json'
    m._CALIBRATION_HISTORY.write_text('not json')
    np.testing.assert_allclose(m._calibration(dict(lab_id=lab, cmd_stress=x, true_stress=y)), [g,b], atol=1e-12)
    archive_g, archive_b = g.copy(), b.copy()
    archive_g[1] += .4; archive_b[1] += .3
    m._CALIBRATION_HISTORY.write_text(json.dumps(dict(stress_gains=archive_g.tolist(), stress_biases=archive_b.tolist())))
    y[np.flatnonzero(lab==1)[0]] += 12
    keep = lab!=2; keep[np.flatnonzero(lab==2)[0]] = True
    np.testing.assert_allclose(m._calibration(dict(lab_id=lab[keep], cmd_stress=x[keep], true_stress=y[keep])), [g,b], atol=1e-12)
print('Calibration verified: no archive access for clean rows, in-batch fit despite an EMI spike, archive fallback for a sparse channel.')
PY
```
- **Container Exit Code**: `0` (`1211 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
injected taus [0.25 1.4 ] fit [0.24931043 1.38750528] max relative tau error 0.008924802708680013 max absolute weight error 0.003317007221625645 worst lab p90 curve/rate error 0.0023944755586665866 0.0009658336836010585
injected taus [1.1 6.2] fit [1.10210834 6.28233681] max relative tau error 0.013280131181387267 max absolute weight error 0.0029024629330727247 worst lab p90 curve/rate error 0.001316892507095957 0.00048753380445580867
Calibration verified: no archive access for clean rows, in-batch fit despite an EMI spike, archive fallback for a sparse channel.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=270d8dfc990808fa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_270d8dfc (directory).
```

### Turn 8 — `submit_completed_task` (`15106 ms` LLM)
- **Rationale**: The exact saved artifact passed isolated-Python schema and synthetic recovery tests, plus calibration checks for clean data, an EMI spike, and sparse-channel archive fallback using a fixture. The larger public missing-feature error does not justify departing from the prescribed cell-mean imputation.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_s00_solver.py': 'FILE:6846'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_s00_solver.py`. The public schema passed, and both synthetic stress tests recovered the relaxation times within 1.33%. Public leave-one-lab-out errors were highest for rows with missing coupon features.`

### Turn 9 — `execute_in_remote_docker` (`37420 ms` LLM)
- **Rationale**: Check that train and test actually stay in the stated linear regime and verify the memory-to-rate formula independently. A consistent held-out gain from stress saturation would warrant investigating a missing term; if curve-derived drift instead explains the missing-feature rate error, retain the prescribed imputation and treat that slice as a known limitation.
- **Alternatives Weighed**: `Test relay modes, Fokker–Planck/HJB operators, and classification metrics` (rejected: The documented inputs have no relay state, stochastic transport equation, or class labels; laboratory transfer is tested directly on strain and rate.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'alpha_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('public_alpha_m97_c03_s00.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr,te=d['train'],d['test']; lab=tr['lab_id']; dt,t=d['past_dt'],d['eval_times']; g,b=m._calibration(d['calibration_load_cells']); x=m._align(tr,te)[:len(lab)]
s=tr['past_stress_cmd']*g[lab,None]+b[lab,None]; h=tr['stress_hold_cmd']*g[lab]+b[lab]
testlab=te['lab_id']; teststress=te['past_stress_cmd']*g[testlab,None]+b[testlab,None]
print('regime: banding, max |past stress| train/test, critical',d['shear_banding_index'],np.max(abs(s)),np.max(abs(teststress)),d['crit_stress'])
assert d['shear_banding_index'] <= .50 and max(np.max(abs(s)),np.max(abs(teststress))) <= d['crit_stress']
y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; complete=np.isfinite(tr['coupon_features_raw']).all(1)
residual=np.empty_like(y); nuisance=np.empty_like(y); drift_removed=np.empty(len(lab))
for j in range(4):
    f=lab!=j; p,w=m._fit(s[f],h[f],dt,t,x[f],y[f]); a,_=m._design(p,s[~f],h[~f],dt,t,x[~f]); pred=a@w
    residual[~f]=y[~f]-pred
    # Finite branch memory at t=0 must agree with an independent time-stepping recurrence.
    tau=np.exp(p); z=np.zeros((sum(~f),2))
    for step,stress in zip(dt,s[~f].T):
        e=np.exp(-step/tau); z=z*e+stress[:,None]*(1-e)
    np.testing.assert_allclose(pred[:,-1],((h[~f,None]-z)/tau)@w[:2]+w[2]+x[~f]@w[3:],rtol=1e-12,atol=1e-12)
    # Diagnostic only: per-coupon residual drift is estimated from curves, not rate.
    drift_removed[~f]=residual[~f,-1]-np.median(residual[~f,:-1]/t,axis=1)
    # Prespecified stress-saturation correction, estimated solely in the outer fold's training labs.
    a_train,_=m._design(p,s[f],h[f],dt,t,x[f]); r=y[f]-a_train@w
    def sat(a,hh):
        return (a[:,:,:2]@w[:2])*(hh**2/(d['crit_stress']**2+hh**2))[:,None]
    q=sat(a_train,h[f]); clean=complete[f]; scale=.05
    corr=np.sum(q[clean]*np.clip(r[clean],-scale,scale))/np.sum(q[clean]**2)
    nuisance[~f]=residual[~f]-corr*sat(a,h[~f])
    for label,v in [('Prony',residual),('stress saturation',nuisance)]:
        z=v[(~f)&complete]; print(j,label,'complete-coupon median strain/rate',np.median(abs(z[:,:-1])),np.median(abs(z[:,-1])))
for mask in (complete,~complete):
    print('complete',bool(complete[mask][0]),'n',sum(mask),'median absolute raw rate residual / after independent curve drift removal',np.median(abs(residual[mask,-1])),np.median(abs(drift_removed[mask])))
PY
```
- **Container Exit Code**: `0` (`592 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
regime: banding, max |past stress| train/test, critical 0.12 2.0671716281578143 2.0671716281578143 2.65
0 Prony complete-coupon median strain/rate 0.010906552432266264 0.010136725141507208
0 stress saturation complete-coupon median strain/rate 0.010939579120304473 0.010137585478700424
1 Prony complete-coupon median strain/rate 0.012190274651132288 0.013563371194709739
1 stress saturation complete-coupon median strain/rate 0.01213782395939648 0.01367387442649413
2 Prony complete-coupon median strain/rate 0.011612915076573471 0.010244508668026509
2 stress saturation complete-coupon median strain/rate 0.011582933241228999 0.010117476692266503
3 Prony complete-coupon median strain/rate 0.014132440509088795 0.009272283430367581
3 stress saturation complete-coupon median strain/rate 0.0139786133064452 0.009310039056363673
complete True n 160 median absolute raw rate residual / after independent curve drift removal 0.011280901985689984 0.011704670353058732
complete False n 20 median absolute raw rate residual / after independent curve drift removal 0.11186165765603173 0.014594269351990728

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=270d8dfc990808fa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_270d8dfc (directory).
```

### Turn 10 — `submit_completed_task` (`16296 ms` LLM)
- **Rationale**: The saved artifact passed isolated-Python schema, parameter recovery, and calibration checks. The independent recurrence matched the initial-rate formula. Removing curve-estimated drift diagnostically reduced the missing-feature median rate error from 0.11186 to 0.01459, supporting an imputation limitation rather than a change to the specified flow law.
- **Alternatives Weighed**: `Add a stress-saturation term` (rejected: It produced no consistent improvement across held-out labs, and public stresses were within the specified linear regime.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_s00_solver.py': 'FILE:6846'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_s00_solver.py`. The public schema passed, and both noisy synthetic tests recovered the relaxation times within 1.33%. Missing coupon features remain the largest source of error in public validation.`

### Turn 11 — `execute_in_remote_docker` (`29024 ms` LLM)
- **Rationale**: The remaining plausible fitting failure is that rows with imputed coupon features bias the shared drift or relaxation parameters despite robust weighting. Compare complete-only fitting against the verified baseline on identical held-out labs, without changing prediction-time alignment. Inspect signed lab residuals and separate early strain, late strain, and initial rate so a pooled improvement cannot hide a worse slice.
- **Alternatives Weighed**: `Repeat the stress-saturation and memory-recurrence checks` (rejected: The recurrence already matched independently, and the saturation term did not improve held-out labs consistently.)

```bash
cat > /tmp/rheology_fit_slice_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root+'public_alpha_m97_c03_s00.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
tr,te = d['train'],d['test']
lab = tr['lab_id']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
g,b = m._calibration(d['calibration_load_cells'])
s = tr['past_stress_cmd']*g[lab,None]+b[lab,None]
h = tr['stress_hold_cmd']*g[lab]+b[lab]
x = m._align(tr,te)[:len(lab)]
t,dt = d['eval_times'],d['past_dt']
y = np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]
res = {k:np.empty_like(y) for k in ('all rows','complete fit')}
for j in np.unique(lab):
    val = lab==j
    for name,r in res.items():
        fit = ~val if name=='all rows' else ~val & complete
        p,w = m._fit(s[fit],h[fit],dt,t,x[fit],y[fit])
        a,_ = m._design(p,s[val],h[val],dt,t,x[val])
        r[val] = y[val] - a@w
        rc = r[val & complete]
        print(j,name,'tau',np.round(np.exp(p),5),'median signed strain/rate',np.round([np.median(rc[:,:-1]),np.median(rc[:,-1])],6),'median absolute early/late/rate',np.round([np.median(abs(rc[:,:3])),np.median(abs(rc[:,-4:-1])),np.median(abs(rc[:,-1]))],6))
for name,r in res.items():
    for status in (True,False):
        vals=[]
        for j in np.unique(lab):
            e=abs(r[(lab==j)&(complete==status)])
            if len(e): vals.append([np.median(e[:,:-1]),np.median(e[:,-1]),np.quantile(e[:,:-1],.9),np.quantile(e[:,-1],.9)])
        print(name,'complete',status,'worst lab medians and P90s',np.round(np.max(vals,axis=0),6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_fit_slice_audit.py
```
- **Container Exit Code**: `0` (`765 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 all rows tau [0.60033 4.13703] median signed strain/rate [-0.001718 -0.00328 ] median absolute early/late/rate [0.009876 0.01173  0.010137]
0 complete fit tau [0.60149 4.17306] median signed strain/rate [-0.001601 -0.00337 ] median absolute early/late/rate [0.010213 0.012179 0.010687]
1 all rows tau [0.6     4.11272] median signed strain/rate [-0.003646 -0.001853] median absolute early/late/rate [0.010683 0.013014 0.013563]
1 complete fit tau [0.60032 4.09882] median signed strain/rate [-0.003559 -0.000802] median absolute early/late/rate [0.010928 0.012269 0.013121]
2 all rows tau [0.59831 4.01305] median signed strain/rate [-0.000361 -0.001423] median absolute early/late/rate [0.010685 0.012855 0.010245]
2 complete fit tau [0.59981 4.08614] median signed strain/rate [-0.000713 -0.002836] median absolute early/late/rate [0.010564 0.011059 0.011704]
3 all rows tau [0.59962 4.03315] median signed strain/rate [ 0.003018 -0.000225] median absolute early/late/rate [0.010119 0.020163 0.009272]
3 complete fit tau [0.60051 4.06305] median signed strain/rate [ 0.002942 -0.00061 ] median absolute early/late/rate [0.010268 0.019677 0.008994]
all rows complete True worst lab medians and P90s [0.014132 0.013563 0.045116 0.030544]
all rows complete False worst lab medians and P90s [0.335433 0.140456 1.094798 0.246413]
complete fit complete True worst lab medians and P90s [0.014517 0.013121 0.045451 0.029974]
complete fit complete False worst lab medians and P90s [0.334734 0.139463 1.093815 0.250346]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=270d8dfc990808fa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_270d8dfc (directory).
```

### Turn 12 — `submit_completed_task` (`16342 ms` LLM)
- **Rationale**: The unchanged solver passed isolated Python checks for the public schema, synthetic parameter recovery, and calibration fallback using an archive fixture. Held-out lab tests found no consistent gain from stress saturation or complete-only fitting. The largest residuals track coupon drift where features are missing, so the solver retains the specified cell-mean imputation and linear flow law.
- **Alternatives Weighed**: `Fit only on coupons with complete features` (rejected: It did not resolve the missing-feature error: the worst-lab median strain error barely changed, and the worst-lab P90 rate error rose from 0.24641 to 0.25035.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_s00_solver.py`. The public schema passed, and both noisy synthetic tests recovered the relaxation times within 1.33%. Missing coupon features remain the largest source of public validation error.`

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
