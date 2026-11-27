# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c02_batch_03__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c02_step_03` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `406,110` (`391,907` / `90,112` / `14,203`) |
| **Total Episode Wall Time (s)** | `471.62s` |

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

### Turn 1 — `execute_in_remote_docker` (`18086 ms` LLM)
- **Rationale**: Inspect the public field layout, validator, and archived alignment conventions before implementing so a schema or normalization mismatch does not masquerade as a rheology fit error.
- **Alternatives Weighed**: `Reuse the previous solver unchanged` (rejected: The earlier branch binding failed its gate, and the target specifies a different terminal-flow law.)

```bash
cd /holdout_task && ls -l && cat validate_public_schema.py && python3 - <<'PY'
import pickle,json,os,numpy as np
p=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))
def show(o,k='',depth=0):
 if isinstance(o,dict) and depth<3:
  print(k, 'keys',list(o));
  for a,b in o.items(): show(b,k+'/'+str(a),depth+1)
 elif isinstance(o,np.ndarray):
  print(k,o.shape,o.dtype,repr(o.ravel()[:8]))
 elif isinstance(o,(list,tuple)):
  print(k,type(o).__name__,len(o),repr(o[:1])[:400])
 else: print(k,type(o).__name__,repr(o)[:300])
show(p)
PY
ls -l /tmp/hegel_continual_archive; grep -nE '^def |align|imput|least_squares' /tmp/hegel_continual_archive/*solver.py | head -100; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation variable projection'
```
- **Container Exit Code**: `0` (`859 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
enous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
```

### Turn 2 — `execute_in_remote_docker` (`18474 ms` LLM)
- **Rationale**: The initial inspection was truncated. Read only the relevant field summaries and archived implementation to resolve panel grouping and identify which measured training targets can constrain the two relaxation times.
- **Alternatives Weighed**: `Copy the previous solver as-is` (rejected: The archive can clarify alignment and calibration conventions, but the new regime requires its own flow and slip terms.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle, numpy as np
p=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]
print('item keys',list(p)); d=p['dataset']; print('dataset keys',list(d))
for k in ('train','test','ref_coupons','calibration_load_cells'):
 v=d.get(k); print(k,type(v).__name__)
 if isinstance(v,dict):
  for a,b in v.items():
   if isinstance(b,dict): print(' ',a,{q:np.shape(r) for q,r in b.items()}); continue
   x=np.asarray(b); print(' ',a,x.shape,x.dtype,x.ravel()[:5], 'nan',np.isnan(x).sum() if x.dtype.kind=='f' else '-')
 elif v is not None: print(' ',str(v)[:500])
PY
sed -n '1,180p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_02_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel | head -45
```
- **Container Exit Code**: `0` (`680 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
) and finite, strictly positive row weights of shape (n,). Nonfinite entries are excluded from both numerator and denominator; a column with no finite observations returns zero. Numeric panel and fill arithmetic use float64 and assume intermediate sums, squares, and ratios are within usable float64 range."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "align_reference_panel": "APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)",
        "balanced_cell_weights": "APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)",
        "weighted_column_fill": "APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)"
      },
      "functions": [
        {
          "function": "align_reference_panel",
          "signature": "align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray",
          "doc_summary": "Return aligned real values (n, p) using real controls (n, q) and finite real groups (n,), with nonempty columns and a supported positive control gain per group."
        }
      ],
      "index_cards": [
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel",
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights",
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_reference_panel_balancing",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`18727 ms` LLM)
- **Rationale**: Recover the exact archived imputation convention from the previously truncated output. Calibration residuals determine whether the public run exercises the archive fallback, and lab-by-stratum counts identify supported held-out validation slices.
- **Alternatives Weighed**: `Use archived calibration coefficients for all four channels` (rejected: The task requires current-batch calibration when clean sweeps are available.)

```bash
sed -n '13,116p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_02_solver.py; cd /holdout_task && python3 - <<'PY'
import pickle
import numpy as np
p=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]['dataset']; c=p['calibration_load_cells']
for lab in range(4):
 i=c['lab_id']==lab; x=c['cmd_stress'][i]; y=c['true_stress'][i]; a,b=np.polyfit(x,y,1); print('calibration',lab,len(x),a,b,'max residual',np.max(np.abs(y-a*x-b)))
for key in ('train','test'):
 d=p[key]; print(key,'lab-by-stratum counts',[[int(np.sum((d['lab_id']==l)&(d['stratum']==s))) for s in np.unique(d['stratum'])] for l in range(4)])
PY
```
- **Container Exit Code**: `0` (`1625 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
np.unique(lab):
        mask = lab == channel
        panel = ref[mask]
        panel = panel[:, np.any(np.isfinite(panel), axis=0)]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[mask] = (x[mask] - center) / scale

    # Each observed lab/stratum cell contributes one column mean, not one
    # vote per coupon; compute the means after panel alignment.
    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    means = np.full((len(cells), x.shape[1]), np.nan)
    for row, (channel, group) in enumerate(cells):
        values = x[(lab == channel) & (stratum == group)]
        counts = np.isfinite(values).sum(axis=0)
        np.divide(np.nansum(values, axis=0), counts, out=means[row], where=counts > 0)
    fill = np.nanmean(means, axis=0)
    x = np.where(np.isfinite(x), x, fill)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    return (gain[:, None]*past + bias[:, None], gain*hold + bias,
            np.column_stack((np.ones(len(x)), x)), strain0)


def _basis(log_taus, past, hold, features, dt, times):
    taus = np.exp(log_taus)
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    a, q, r = age[:, None]/taus, dt[:, None]/taus, times[:, None]/taus
    ea, eq, er = np.exp(-a), np.exp(-q), np.exp(-r)
    # Unroll the recurrence from z(0)=0; each history interval's
    # contribution decays through all subsequent intervals.
    memory_weights = ea * -np.expm1(-q)
    gap = hold[:, None] - past @ memory_weights
    dgap = -past @ (a*memory_weights - q*ea*eq)
    step = -np.expm1(-r)
    curve = np.concatenate((gap[:, None, :]*step,
                            times[None, :, None]*features[:, None, :]), axis=2)
    rate = np.column_stack((gap/taus, features))
    dcurve = dgap[:, None, :]*step - gap[:, None, :]*r*er
    drate = (dgap - gap)/taus
    return curve, rate, dcurve, drate


def _fit(past, hold, features, dt, times, curves, rates):
calibration 0 10 0.789974688683986 -0.13996010961137748 max residual 0.00026644408372386574
calibration 1 10 1.3099732469730951 0.1599621257048209 max residual 0.00014693766250709062
calibration 2 10 0.8499868792510629 0.10001623237093303 max residual 0.00018462014860752551
calibration 3 10 1.2399723212528375 -0.13003537314250055 max residual 0.0002188130456673154
train lab-by-stratum counts [[15, 13, 10], [12, 26, 19], [12, 8, 23], [8, 13, 21]]
test lab-by-stratum counts [[3, 7, 6], [4, 10, 3], [5, 2, 10], [7, 8, 7]]
```

### Turn 4 — `execute_in_remote_docker` (`42660 ms` LLM)
- **Rationale**: The four clean sweeps support direct in-batch calibration. Implement the specified slip and saturation basis, retaining the archived panel convention. Run the isolated schema gate to catch missing outputs or numerical failures before testing scientific accuracy by slice.
- **Alternatives Weighed**: `Optimize all eleven parameters jointly` (rejected: At fixed time scales the nine weights have an exact conditional least-squares solution, leaving only two nonlinear coordinates.)

```bash
set -e
cd /holdout_task
f=alpha_m97_c02_batch_03_solver.py
printf '%s\n\n' '"""Calibrated two-branch Prony model for jammed shear-banding slip."""' > "$f"
sed -n '3,/^def _basis/{ /^def _basis/!p; }' /tmp/hegel_continual_archive/alpha_m97_c02_batch_02_solver.py >> "$f"
cat >> "$f" <<'PY'
def _basis(taus, past, hold, features, dt, times):
    # Unroll the memory recurrence: each interval's contribution decays
    # through all later intervals before the start of the hold.
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    memory = np.exp(-age[:, None] / taus) * -np.expm1(-dt[:, None] / taus)
    gap = hold[:, None] - past @ memory
    step = -np.expm1(-times[:, None] / taus)
    phi = times / (1 + 0.85 * times)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    cross_gap = gap[:, 0] - gap[:, 1]
    curve = np.concatenate((gap[:, None, :] * step,
                            features[:, None, :] * phi[None, :, None],
                            (slip[:, None] * phi)[..., None],
                            (cross_gap[:, None] * step[:, 0])[..., None]), axis=2)
    rate = np.column_stack((gap / taus, features, slip, cross_gap / taus[0]))
    return curve, rate


def _fit(past, hold, features, dt, times, curves, rates):
    # The nine coefficients are linear conditional on the two time scales.
    # Use QR/SVD least squares rather than normal equations, since the
    # slip and Prony terms can have similar shapes on short holds.
    rate_weight = 1.0
    target = np.r_[curves.ravel(), rate_weight * rates]

    def project(params):
        taus = np.exp([params[0], params[0] + params[1]])
        curve, rate = _basis(taus, past, hold, features, dt, times)
        design = np.concatenate((curve.reshape(-1, 9), rate_weight * rate))
        weights = np.linalg.lstsq(design, target, rcond=None)[0]
        return design @ weights - target, weights, taus

    best = None
    bounds = (np.log([0.015, 1.015]), np.log([20., 250.]))
    for fast, slow in ((0.15, 1.), (0.4, 2.5), (0.8, 4.), (1.5, 8.)):
        fit = least_squares(lambda p: project(p)[0], np.log([fast, slow / fast]),
                            bounds=bounds, max_nfev=100,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if best is None or fit.cost < best.cost:
            best = fit
    _, weights, taus = project(best.x)
    return taus, weights


def solve_alpha_m97_c02_batch_03(dataset: dict) -> dict:
    past, hold, features, strain0 = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    taus, weights = _fit(past[:n], hold[:n], features[:n], dt, times,
                         train['strain_curves_obs'] - strain0[:n, None],
                         train['instant_rate_obs'])
    curve, rate = _basis(taus, past[n:], hold[n:], features[n:], dt, times)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
python3 - <<'PY'
import json,pickle,numpy as np
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
print('archive top keys:', list(h) if isinstance(h,dict) else type(h).__name__)
print('archive first-level summary:', [(k, type(v).__name__, list(v)[:6] if isinstance(v,dict) else '') for k,v in h.items()] if isinstance(h,dict) else '')
g=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]['gt']
print('public truth schema:', {k:np.shape(v) for k,v in g.items()})
PY
```
- **Container Exit Code**: `0` (`997 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
archive top keys: list
archive first-level summary: 
public truth schema: {'taus': (2,), 'theta': (7,), 'vp_theta': (2,), 'stress_gains': (4,), 'stress_biases': (4,), 'te_clean': (72, 10), 'te_true_rate0': (72,), 'te_hist_true': (72, 18), 'te_hold_true': (72,)}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7ff308aee496e9bc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7ff308ae (directory).
```

### Turn 5 — `execute_in_remote_docker` (`32332 ms` LLM)
- **Rationale**: Evaluate public truth only after fitting from measured inputs, and refit without each lab's response labels to expose lab-specific errors. Missing-coupon and lab-by-stratum slices distinguish panel issues from overall fit error. Check archive lookup separately because clean public calibrations do not exercise it.
- **Alternatives Weighed**: `Submit after the schema gate alone` (rejected: Shape and finiteness checks cannot detect wrong relaxation times, inaccurate initial rates, or an archive lookup that silently fails.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle, time
import numpy as np
s=importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; start=time.monotonic(); o=m.solve_alpha_m97_c02_batch_03(d); print('solve seconds',round(time.monotonic()-start,3))
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(k,'fit',o[k],'reference',g[j], 'max abs',np.max(np.abs(o[k]-g[j])))
def score(a,b): return [round(float(np.sqrt(np.mean(a*a))),7),round(float(np.sqrt(np.mean(b*b))),7)]
for split in ['test','train']:
    tbl=d[split]; lab=tbl['lab_id']; stratum=tbl['stratum']
    if split=='test':
        ec=o['predicted_strain']-g['te_clean']; er=o['initial_rate_pred']-g['te_true_rate0']
        print('public test RMSE',score(ec,er),'max absolute',np.max(np.abs(ec)),np.max(np.abs(er)))
    else:
        past,hold,f,e0=m._prepare(d); n=len(lab); past,hold,f,e0=past[:n],hold[:n],f[:n],e0[:n]; dt,t=d['past_dt'],d['eval_times']; c=tbl['strain_curves_obs']-e0[:,None]; r=tbl['instant_rate_obs']; ec=np.empty_like(c); er=np.empty_like(r)
        for l in np.unique(lab):
            v=lab==l; a=~v; tau,w=m._fit(past[a],hold[a],f[a],dt,t,c[a],r[a]); cb,rb=m._basis(tau,past[v],hold[v],f[v],dt,t); ec[v]=cb@w-c[v]; er[v]=rb@w-r[v]; print('held-out lab',l,'RMSE',score(ec[v],er[v]),'taus',tau)
    groups=[(l,s,score(ec[(lab==l)&(stratum==s)],er[(lab==l)&(stratum==s)])) for l,s in np.unique(np.column_stack([lab,stratum]),axis=0)]
    print(split,'worst lab-by-stratum strain and rate',max(groups,key=lambda a:a[2][0]),max(groups,key=lambda a:a[2][1]),'worst time RMSE',np.sqrt(np.mean(ec**2,axis=0)).max())
    miss=np.isnan(tbl['coupon_features_raw']).any(axis=1); print(split,'complete/missing coupon RMSE',score(ec[~miss],er[~miss]),score(ec[miss],er[miss]))
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('first archive entry shape',[(k,type(v).__name__) for k,v in h[0].items()]); print('archive lookup', [m._archived_calibration(i) for i in range(4)])
cal=d['calibration_load_cells']; print('public calibration gain max error, bias max error',np.abs(m._calibrate(cal)-np.column_stack([g['stress_gains'],g['stress_biases']])).max(axis=0))
PY
```
- **Container Exit Code**: `0` (`1009 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
solve seconds 0.085
estimated_taus fit [0.42596618 1.85879562] reference [0.576 4.235] max abs 2.3762043783503515
hereditary_weights fit [ 0.41150503  0.66467227  0.10633404  0.18171161 -0.06801608  0.13673726
 -0.07985637] reference [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09] max abs 0.26849497295211383
regime_mode_weights fit [ 0.27152185 -0.14530636] reference [ 0.31 -0.22] max abs 0.07469363939847842
public test RMSE [0.1037325, 0.1156895] max absolute 0.3160665096759585 0.2497879899560821
test worst lab-by-stratum strain and rate (0, 0, [0.1480005, 0.1985991]) (0, 0, [0.1480005, 0.1985991]) worst time RMSE 0.13749835760766482
test complete/missing coupon RMSE [0.1031228, 0.1144763] [0.1168809, 0.1407363]
held-out lab 0 RMSE [0.5063518, 0.1888982] taus [0.40739371 1.84767017]
held-out lab 1 RMSE [1.0042193, 0.1480121] taus [0.427084   1.81330569]
held-out lab 2 RMSE [0.4698243, 0.1501783] taus [0.38261443 1.67604917]
held-out lab 3 RMSE [2.9693269, 0.0826676] taus [0.55497208 3.07873221]
train worst lab-by-stratum strain and rate (3, 0, [6.7669732, 0.0866035]) (0, 0, [0.1696202, 0.204027]) worst time RMSE 4.490404721250942
train complete/missing coupon RMSE [0.7321297, 0.1384283] [4.4733293, 0.2050999]
first archive entry shape [('episode_short_name', 'str'), ('lab_id', 'list'), ('cmd_stress', 'list'), ('true_stress', 'list')]
archive lookup [None, None, None, None]
public calibration gain max error, bias max error [2.76787472e-05 3.98903886e-05]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7ff308aee496e9bc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7ff308ae (directory).
```

### Turn 6 — `execute_in_remote_docker` (`30525 ms` LLM)
- **Rationale**: Compare public-reference residuals on noisy training observations and clean test responses strictly as a diagnostic, not as fitting inputs. This separates sparse contamination from systematic preprocessing error. The archive lookup returned no coefficients despite available sweep rows; inspect sweep order and channel coverage before fixing the parser.
- **Alternatives Weighed**: `Change the rheology basis immediately` (rejected: The existing errors do not yet distinguish a basis error from response outliers or preprocessing errors.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle
import numpy as np
s=importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr=d['train']; n=len(tr['lab_id']); past,hold,f,e0=m._prepare(d); dt,t=d['past_dt'],d['eval_times']; print('dt and evaluation grid',dt,t)
w=np.r_[g['theta'],g['vp_theta']]; basis,rate=m._basis(g['taus'],past,hold,f,dt,t); ce=e0[:n,None]+basis[:n]@w-tr['strain_curves_obs']; re=rate[:n]@w-tr['instant_rate_obs']; te=e0[n:,None]+basis[n:]@w-g['te_clean']; rt=rate[n:]@w-g['te_true_rate0']; print('oracle residual quantiles',[(v.shape,np.quantile(np.abs(v),[0,.25,.5,.8,.9,.95,.98,.99,1])) for v in [ce,re,te,rt]]); print('oracle train curve residual median/MAD by time',np.median(ce,axis=0),np.median(np.abs(ce-np.median(ce,axis=0)),axis=0)); print('oracle residual train lab quantiles',[(l,np.quantile(np.abs(ce[tr['lab_id']==l]),[.5,.95,1]),np.quantile(np.abs(re[tr['lab_id']==l]),[.5,.95,1])) for l in range(4)]); print('oracle test stress max errors',np.max(np.abs(past[n:]-g['te_hist_true'])),np.max(np.abs(hold[n:]-g['te_hold_true']))
for label,x in [('train',ce),('test',te)]:
 tab=d[label]; print(label,'largest curve row residuals');
 for i in np.argsort(np.max(np.abs(x),axis=1))[-8:]: print(i,tab['lab_id'][i],tab['stratum'][i],np.isnan(tab['coupon_features_raw'][i]).sum(),np.round(x[i],6))
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('archive calibration sweeps')
for e in h:
 l,x,y=(np.asarray(e[k]) for k in ('lab_id','cmd_stress','true_stress')); print(e['episode_short_name'],[(c,int(np.sum(l==c)),np.round(np.polyfit(x[l==c],y[l==c],1),5).tolist() if np.unique(x[l==c]).size>=2 else None) for c in range(4)])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS variable projection' | head -25
```
- **Container Exit Code**: `1` (`317 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7ff308aee496e9bc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7ff308ae (directory).
```
- **Stderr (tail)**:
```text
p_theta']]; basis,rate=m._basis(g['taus'],past,hold,f,dt,t); ce=e0[:n,None]+basis[:n]@w-tr['strain_curves_obs']; re=rate[:n]@w-tr['instant_rate_obs']; te=e0[n:,None]+basis[n:]@w-g['te_clean']; rt=rate[n:]@w-g['te_true_rate0']; print('oracle residual quantiles',[(v.shape,np.quantile(np.abs(v),[0,.25,.5,.8,.9,.95,.98,.99,1])) for v in [ce,re,te,rt]]); print('oracle train curve residual median/MAD by time',np.median(ce,axis=0),np.median(np.abs(ce-np.median(ce,axis=0)),axis=0)); print('oracle residual train lab quantiles',[(l,np.quantile(np.abs(ce[tr['lab_id']==l]),[.5,.95,1]),np.quantile(np.abs(re[tr['lab_id']==l]),[.5,.95,1])) for l in range(4)]); print('oracle test stress max errors',np.max(np.abs(past[n:]-g['te_hist_true'])),np.max(np.abs(hold[n:]-g['te_hold_true']))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ^
SyntaxError: '(' was never closed
```

### Turn 7 — `execute_in_remote_docker` (`29674 ms` LLM)
- **Rationale**: Rerun the diagnostic after fixing its syntax. Residuals proportional to the drift shape would point to coupon preprocessing; isolated deviations in measured training curves would motivate robust fitting. The archive summary will identify which prefix sweeps can supply the missing-channel calibration fallback.
- **Alternatives Weighed**: `Switch to a robust loss before examining the residuals` (rejected: A robust loss can suppress training spikes but cannot correct a systematic mismatch in the clean prediction curves.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle
import numpy as np
s = importlib.util.spec_from_file_location('solver', 'alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
p = pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]
d, g = p['dataset'], p['gt']
tr = d['train']
n = len(tr['lab_id'])
past, hold, f, e0 = m._prepare(d)
t = d['eval_times']
print('evaluation grid', t)
w = np.r_[g['theta'], g['vp_theta']]
basis, rate = m._basis(g['taus'], past, hold, f, d['past_dt'], t)
ce = e0[:n,None] + basis[:n] @ w - tr['strain_curves_obs']
re = rate[:n] @ w - tr['instant_rate_obs']
te = e0[n:,None] + basis[n:] @ w - g['te_clean']
rt = rate[n:] @ w - g['te_true_rate0']
for label, v in [('train_curve',ce),('train_rate',re),('test_curve',te),('test_rate',rt)]:
 print(label, np.quantile(np.abs(v), [.25,.5,.8,.9,.95,.98,.99,1]))
print('train curve residual MAD by time', np.median(np.abs(ce-np.median(ce,axis=0)),axis=0))
print('test residual unmodeled by drift shape', np.max(np.abs(te - rt[:,None]*(t/(1+.85*t)))))
print('test stress max errors', np.max(np.abs(past[n:]-g['te_hist_true'])), np.max(np.abs(hold[n:]-g['te_hold_true'])))
for label, x in [('train',ce),('test',te)]:
 tab=d[label]
 miss=np.isnan(tab['coupon_features_raw']).any(axis=1)
 print(label, 'complete/missing RMSE', [np.sqrt(np.mean(x[mask]**2)) for mask in (~miss,miss)])
 for i in np.argsort(np.max(np.abs(x),axis=1))[-5:]:
  print(i,tab['lab_id'][i],tab['stratum'][i],int(miss[i]),np.round(x[i],5))
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
for e in h:
 labs,x,y=(np.asarray(e[k]) for k in ('lab_id','cmd_stress','true_stress'))
 print(e['episode_short_name'], [(c,int(np.sum(labs==c)),np.round(np.polyfit(x[labs==c],y[labs==c],1),5).tolist() if np.unique(x[labs==c]).size>=2 else None) for c in range(4)])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS variable projection' | head -22
```
- **Container Exit Code**: `0` (`585 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cribe autogenous_prim_robust_affine_projection.profile_weighted_affine | STATUS: APPLICABLE_SUBROUTINE(real_2d_predictor_3d_sensitivity_2d_target)
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
autogenous_prim_robust_affine_projection@abc35d3d | propagate_linear_basis(transition: np.ndarray, initial: np.ndarray, increments: np.ndarray) -> np.ndarray | Return a trajectory of shape (steps+1, n, k) from transition (n, n), initial ( | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.propagate_linear_basis | STATUS: APPLICABLE_SUBROUTINE(real_linear_transition_and_2d_initial_basis)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_local_level_step(predictor: np.ndarray, values: np.ndarray, splits: np.ndarray, *, scale: float, loss_clip: float) -> tuple[int, float] | Fit finite predictor and values (n,) and integer splits (k,) with k >= 1, at l | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_local_level_step | STATUS: APPLICABLE_SUBROUTINE(real_1d_step_series)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7ff308aee496e9bc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7ff308ae (directory).
```

### Turn 8 — `execute_in_remote_docker` (`42760 ms` LLM)
- **Rationale**: Measure whether suppressing sparse spikes recovers the relaxation parameters and lowers public slice errors without consulting truth during fitting. Read the inherited preprocessing to locate the remaining drift mismatch and prepare the archive parser fix.
- **Alternatives Weighed**: `Drop entire traces with a large residual` (rejected: The observed transient spikes leave neighboring measurements informative; pointwise weights preserve those constraints.)
- **ATMS NOGOOD DECLARED**: Refuted `All measured strain points can be treated as comparable deviations around the physical relaxation curve.` | Witness: `A single observed strain point deviates by 60.00245 while neighboring points on the same trace deviate by at most 0.153; per-time residual MAD is about 0.012–0.015.` | Blanket Diagnosis: `Sparse observation spikes distort the unweighted fit; separately, the reference-parameter test residual has the terminal-flow time shape to within 1.51e-5, isolating a drift-side preprocessing mismatch.` | Sublated Pivot: `Keep the specified constitutive basis and downweight individual strain spikes through robust variable projection before isolating the coupon alignment mismatch.`

```bash
set -e
cd /holdout_task
sed -n '1,105p' alpha_m97_c02_batch_03_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_affine_projection.robust_residual_weights | head -35
cp alpha_m97_c02_batch_03_solver.py /tmp/rheology_v01.py
python3 - <<'PY'
from pathlib import Path
p=Path('alpha_m97_c02_batch_03_solver.py'); s=p.read_text()
a=s.index('    # The nine coefficients', s.index('def _fit(')); b=s.index('\n\ndef solve_',a)
s=s[:a]+'''    target = np.r_[curves.ravel(), rates]
    bounds = (np.log([0.015, 1.015]), np.log([20., 250.]))

    def design_at(params):
        taus = np.exp([params[0], params[0] + params[1]])
        curve, rate = _basis(taus, past, hold, features, dt, times)
        return np.vstack((curve.reshape(-1, 9), rate)), taus

    def project(params, weight):
        design, taus = design_at(params)
        coef = np.linalg.lstsq(design * weight[:, None], target * weight, rcond=None)[0]
        return design @ coef - target, coef, taus

    # Reweight individual observations, so a transient spike cannot discard
    # the other time points in the same specimen's trace.
    best = None
    for fast, slow in ((0.2, 1.5), (0.55, 4.), (1.2, 8.)):
        params = np.log([fast, slow / fast])
        weight = np.ones_like(target)
        for scale in (0.20, 0.08, 0.04, 0.025, 0.025, 0.025, 0.025, 0.025):
            residual = project(params, weight)[0]
            weight = 1 / np.hypot(1, residual / scale)
            fit = least_squares(lambda p: weight * project(p, weight)[0], params,
                                bounds=bounds, max_nfev=45, ftol=1e-8, xtol=1e-8)
            params = fit.x
        residual, coef, taus = project(params, weight)
        cost = np.sum(np.log1p((residual / scale)**2))
        if best is None or cost < best[0]:
            best = cost, taus, coef
    return best[1:]
'''+s[b:];p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, time
import numpy as np
s=importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; start=time.monotonic(); o=m.solve_alpha_m97_c02_batch_03(d); print('seconds',time.monotonic()-start)
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(k,np.round(o[k],6),'max error',np.max(np.abs(o[k]-g[j])))
ec=o['predicted_strain']-g['te_clean']; er=o['initial_rate_pred']-g['te_true_rate0']; lab=d['test']['lab_id']; st=d['test']['stratum']; miss=np.isnan(d['test']['coupon_features_raw']).any(axis=1)
for name,mask in [('all',np.ones(len(er),bool)),('complete',~miss),('missing',miss)]+[(f'{l}/{s}',(lab==l)&(st==s)) for l,s in np.unique(np.c_[lab,st],axis=0)]: print(name,*(round(np.sqrt(np.mean(x[mask]**2)),7) for x in (ec,er)))
PY
```
- **Container Exit Code**: `0` (`1259 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
S and a positive scalar or scale array broadcastable to S, return Huber or Cauchy IRLS weights of shape S."
        }
      ],
      "index_cards": [
        "autogenous_prim_robust_affine_projection@abc35d3d | propagate_linear_basis(transition: np.ndarray, initial: np.ndarray, increments: np.ndarray) -> np.ndarray | Return a trajectory of shape (steps+1, n, k) from transition (n, n), initial ( | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.propagate_linear_basis",
        "autogenous_prim_robust_affine_projection@abc35d3d | profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | For predictor, target and optional weights (n, m) and predictor sensitivity (n | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.profile_weighted_affine",
        "autogenous_prim_robust_affine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar or scale array broadcastable to | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights"
      ],
      "execution_available": true,
{"schema_valid": true, "errors": []}
seconds 0.14784529502503574
estimated_taus [0.575307 4.244732] max error 0.009732383585787652
hereditary_weights [ 0.67858   0.554354  0.139661  0.15648  -0.127764  0.118556 -0.088496] max error 0.004353571356284114
regime_mode_weights [ 0.309212 -0.218303] max error 0.0016970052232618826
all 0.0169299 0.0218794
complete 0.0076542 0.0103065
missing 0.0743735 0.0951098
0/0 0.0055685 0.0079418
0/1 0.0111887 0.0149532
0/2 0.0063571 0.0076739
1/0 0.0031901 0.004795
1/1 0.0029473 0.0042564
1/2 0.0027665 0.0037921
2/0 0.0029414 0.0039385
2/1 0.0022887 0.0016875
2/2 0.0230244 0.0299882
3/0 0.0408443 0.0520808
3/1 0.0091288 0.0125376
3/2 0.0149997 0.0197209

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9ec1ba69e83089d7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_9ec1ba69 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`24552 ms` LLM)
- **Rationale**: Inspect the inherited alignment and compare complete versus missing rows at fixed reference parameters. An affine residual within labs points to centering or scaling; a residual confined to missing rows points to imputation. Keep the response-free preprocessing contract and fix the known list-of-sweeps archive parser for sparse or contaminated calibrations.
- **Alternatives Weighed**: `Tune robust loss against public reference outputs` (rejected: The fitted parameters now agree closely with the public reference. An affine residual on complete rows and a larger residual on missing rows call for auditing feature preparation, not tuning the loss against test labels.)

```bash
set -e
cd /holdout_task
sed -n '38,96p' alpha_m97_c02_batch_03_solver.py
ls -l /tmp/hegel_continual_archive
python3 - <<'PY'
import json
p=json.load(open('public_asset_index.json'))
print(str(p)[:2200])
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; n=len(d['train']['lab_id']); past,hold,f,e0=m._prepare(d); _,rb=m._basis(g['taus'],past,hold,f,d['past_dt'],d['eval_times']); e=(rb@np.r_[g['theta'],g['vp_theta']])[n:]-g['te_true_rate0']; test=d['test']; miss=np.isnan(test['coupon_features_raw']); print('test missing rows (index, lab, stratum, columns, reference-parameter rate residual):');
for i in np.flatnonzero(miss.any(axis=1)):
 print(i,test['lab_id'][i],test['stratum'][i],np.flatnonzero(miss[i]),round(e[i],6))
for l in range(4):
 a=(test['lab_id']==l)&~miss.any(axis=1); z=np.column_stack([np.ones(a.sum()),f[n:][a,1:]]); w=np.linalg.lstsq(z,e[a],rcond=None)[0]; print('complete cases lab',l,'affine residual coeffs',np.round(w,6),'unexplained RMSE',np.sqrt(np.mean((z@w-e[a])**2)))
PY
```
- **Container Exit Code**: `0` (`579 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
w': {'dtype': 'float64', 'shape': [72, 4], 'type': 'ndarray'}, 'lab_id': {'dtype': 'int64', 'shape': [72], 'type': 'ndarray'}, 'past_stress_cmd': {'dtype': 'float64', 'shape': [72, 18], 'type': 'ndarray'}, 'ref_coupons': {'dtype': 'float64', 'shape': [72, 32], 'type': 'ndarray'}, 'strain_0': {'dtype': 'float64', 'shape': [72], 'type': 'ndarray'}, 'stratum': {'dtype': 'int64', 'shape': [72], 'type': 'ndarray'}, 'stress_hold_cmd': {'dtype': 'float64', 'shape': [72], 'type': 'ndarray'}}, 'train': {'aux_chamber_telemetry': {'dtype': 'float64', 'shape': [180, 3], 'type': 'ndarray'}, 'coupon_features_raw': {'dtype': 'float64', 'shape': [180, 4], 'type': 'ndarray'}, 'instant_rate_obs': {'dtype': 'float64', 'shape': [180], 'type': 'ndarray'}, 'lab_id': {'dtype': 'int64', 'shape': [180], 'type': 'ndarray'}, 'past_stress_cmd': {'dtype': 'float64', 'shape': [180, 18], 'type': 'ndarray'}, 'ref_coupons': {'dtype': 'float64', 'shape': [180, 32], 'type': 'ndarray'}, 'strain_0': {'dtype': 'float64', 'shape': [180], 'type': 'ndarray'}, 'strain_curves_obs': {'dtype': 'float64', 'shape': [180, 10], 'type': 'ndarray'}, 'stratum': {'dtype': 'int64', 'shape': [180], 'type': 'ndarray'}, 'stress_hold_cmd': {'dtype': 'float64', 'shape': [180], 'type': 'ndarray'}}, 'yield_stress': 2.65}, 'gt': {'stress_biases': {'dtype': 'float64', 'shape': [4], 'type': 'ndarray'}, 'stress_gains': {'dtype': 'float64', 'shape': [4], 'type': 'ndarray'}, 'taus': {'dtype': 'float64', 'shape': [2], 'type': 'ndarray'}, 'te_clean': {'dtype': 'float64', 'shape': [72, 10], 'type'
test missing rows (index, lab, stratum, columns, reference-parameter rate residual):
10 2 2 [1] 0.090773
46 0 2 [1 3] 0.028716
65 3 0 [0] 0.143096
complete cases lab 0 affine residual coeffs [ 0.002111  0.00901  -0.007322  0.006742 -0.005075] unexplained RMSE 3.831497192373569e-05
complete cases lab 1 affine residual coeffs [-0.003314  0.003594 -0.002932  0.002713 -0.002041] unexplained RMSE 2.2336639622833723e-05
complete cases lab 2 affine residual coeffs [ 0.002066  0.001484 -0.001194  0.001112 -0.000823] unexplained RMSE 2.774384143598782e-05
complete cases lab 3 affine residual coeffs [-0.000179 -0.005505  0.004474 -0.004113  0.003101] unexplained RMSE 2.9717904863212774e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9ec1ba69e83089d7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_9ec1ba69 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`32721 ms` LLM)
- **Rationale**: Fix the confirmed archive-format bug and exercise the missing-channel path. Diagnose the remaining error by comparing the affine pattern implied by public reference outputs with response-free panel statistics and imputation conventions; do not use public truth in the solver.
- **Alternatives Weighed**: `Absorb lab-specific drift offsets into fitted weights` (rejected: That would depart from the required shared hereditary weights and conceal an affine preprocessing mismatch.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p=Path('alpha_m97_c02_batch_03_solver.py'); s=p.read_text(); a=s.index('    if not _ARCHIVE'); b=s.index('\n\ndef _calibrate',a)
s=s[:a]+'''    with _ARCHIVE.open() as stream:
        history = json.load(stream)
    for sweep in reversed(history):
        mask = np.asarray(sweep['lab_id']) == channel
        x, y = (np.asarray(sweep[key])[mask] for key in ('cmd_stress', 'true_stress'))
        if np.unique(x).size >= 2:
            return np.polyfit(x, y, 1)
    return None
'''+s[b:]; p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); raw,ref,lab,st=(np.concatenate([tr[k],te[k]]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')); past,hold,f,_=m._prepare(d); rate=m._basis(g['taus'],past,hold,f,d['past_dt'],d['eval_times'])[1]; beta=g['theta'][3:]; y=g['te_true_rate0']-(rate@np.r_[g['theta'],g['vp_theta']])[n:]+f[n:,1:]@beta
complete=np.isfinite(raw[n:]).all(axis=1); x=raw[n:]@beta
for l in range(4):
 v=(lab[n:]==l)&complete; inv,offset=np.polyfit(x[v],y[v],1); oracle=np.array([-offset/inv,1/inv]); oracle[0]/=beta.sum(); print('lab',l,'empirically implied center/scale',np.round(oracle,6));
 for label,idx in [('all',np.arange(len(lab))),('train',np.arange(n)),('test',np.arange(n,len(lab)))]:
  panel=ref[idx[lab[idx]==l]]; print(label,'median mean/std',np.round([np.median(np.nanmean(panel,axis=0)),np.median(np.nanstd(panel,axis=0)),np.median(np.nanstd(panel,axis=0,ddof=1))],6),'flat',np.round([np.nanmean(panel),np.nanstd(panel)],6))
print('implied missing drift and candidate fills (aligned):')
for j in np.flatnonzero(~complete):
 i=n+j; missing=~np.isfinite(raw[i]); cell=(lab==lab[i])&(st==st[i]); candidates=[f[i,1:][missing], np.nanmean(np.where(np.isfinite(raw[cell]),f[cell,1:],np.nan),axis=0)[missing], np.zeros(missing.sum())]; print(j,np.flatnonzero(missing),'required contribution',y[j]-f[i,1:][~missing]@beta[~missing], 'global, cell, zero', [float(a@beta[missing]) for a in candidates])
cal=d['calibration_load_cells']; mask=cal['lab_id']!=2; sparse={k:v[mask] for k,v in cal.items()}; print('sparse calibration',m._calibrate(sparse)[2], 'latest archive',m._archived_calibration(2)); assert np.allclose(m._calibrate(sparse)[2],m._archived_calibration(2))
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`899 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
lab 0 empirically implied center/scale [-0.650105  0.820023]
all median mean/std [-0.678883  0.773867  0.781417] flat [-0.671235  0.809764]
train median mean/std [-0.657348  0.761869  0.772677] flat [-0.682635  0.809714]
test median mean/std [-0.636007  0.764497  0.790454] flat [-0.644451  0.809251]
lab 1 empirically implied center/scale [0.479637 1.280092]
all median mean/std [0.550467 1.25127  1.260178] flat [0.521424 1.254891]
train median mean/std [0.541641 1.234707 1.246194] flat [0.505507 1.255336]
test median mean/std [0.616601 1.18776  1.224315] flat [0.573687 1.252005]
lab 2 empirically implied center/scale [-0.24987   0.939986]
all median mean/std [-0.282352  0.931278  0.939557] flat [-0.276062  0.939881]
train median mean/std [-0.286713  0.918776  0.930189] flat [-0.284568  0.931874]
test median mean/std [-0.269561  0.899549  0.928493] flat [-0.254632  0.959423]
lab 3 empirically implied center/scale [0.800102 1.419904]
all median mean/std [0.804273 1.468764 1.481056] flat [0.794938 1.452295]
train median mean/std [0.744781 1.405227 1.422684] flat [0.764176 1.442386]
test median mean/std [0.835242 1.399771 1.433503] flat [0.853194 1.469114]
implied missing drift and candidate fills (aligned):
10 [1] required contribution -0.08422714444774097 global, cell, zero [0.006545746775803127, -0.05761382282168436, 0.0]
46 [1 3] required contribution -0.024520573693997405 global, cell, zero [0.004195585629842279, -0.09607450309788507, 0.0]
65 [0] required contribution -0.14977963704618336 global, cell, zero [-0.006683674932643196, -0.06335982114813475, 0.0]
sparse calibration [0.84998575 0.09998359] latest archive [0.84998575 0.09998359]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=678afb84309a474f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_678afb84 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`48657 ms` LLM)
- **Rationale**: Refit without each lab’s response labels to test whether the robust fit transfers despite strain spikes. Test sparse and contaminated calibration on all four channels; a failure would require a calibration fix. Independently verify the memory recurrence and initial slope to catch basis sign or ordering errors.
- **Alternatives Weighed**: `Replace panel estimates with lab constants inferred from public truth` (rejected: That would violate the requested panel estimator and would not transfer to a new campaign.)

```bash
set -e
cd /holdout_task
cat >/tmp/verify_rheology03.py <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_03.pkl', 'rb'))[0]
d, g = p['dataset'], p['gt']
tr, te = d['train'], d['test']
n = len(tr['lab_id'])
past, hold, f, e0 = m._prepare(d)
dt, t = d['past_dt'], d['eval_times']
y, r = tr['strain_curves_obs'] - e0[:n,None], tr['instant_rate_obs']
rmse = lambda e: float(np.sqrt(np.mean(e**2)))
ce, re = np.empty_like(y), np.empty_like(r)
for l in range(4):
    v = tr['lab_id'] == l
    a = ~v
    tau, w = m._fit(past[:n][a], hold[:n][a], f[:n][a], dt, t, y[a], r[a])
    b, b0 = m._basis(tau, past[:n][v], hold[:n][v], f[:n][v], dt, t)
    ce[v], re[v] = b @ w - y[v], b0 @ w - r[v]
    print('held-out lab', l, 'strain MAE/median AE, rate RMSE, taus', np.mean(abs(ce[v])), np.median(abs(ce[v])), rmse(re[v]), tau)
lab, st = tr['lab_id'], tr['stratum']
cells = [(l,s,(lab==l)&(st==s)) for l,s in np.unique(np.c_[lab,st],axis=0)]
print('worst held-out lab/stratum median abs strain and rate RMSE', max((np.median(abs(ce[v])),l,s) for l,s,v in cells), max((rmse(re[v]),l,s) for l,s,v in cells))
o = m.solve_alpha_m97_c02_batch_03(d)
ec, er = o['predicted_strain']-g['te_clean'],o['initial_rate_pred']-g['te_true_rate0']
print('worst clean-test horizon RMSE', max(rmse(ec[:,i]) for i in range(len(t))))
for label, v in [('positive',hold[n:]>0),('negative',hold[n:]<0),('large hold',abs(hold[n:])>d['crit_stress']),('missing coupon',np.isnan(te['coupon_features_raw']).any(axis=1))]:
    print(label, int(v.sum()), rmse(ec[v]), rmse(er[v]))
cal = d['calibration_load_cells']
base = m._calibrate(cal)
for l in range(4):
    ch = np.flatnonzero(cal['lab_id']==l)
    i = ch[len(ch)//2]
    for spike in (-12.,12.):
        changed = {k:v.copy() for k,v in cal.items()}
        changed['true_stress'][i] += spike
        assert np.max(abs(m._calibrate(changed)-base)) < 3e-4
        keep = (cal['lab_id']!=l) | np.isin(np.arange(len(cal['lab_id'])),ch[[0,4,9]])
        changed = {k:v[keep].copy() for k,v in cal.items()}
        spike_row = np.flatnonzero(changed['lab_id']==l)[1]
        changed['true_stress'][spike_row] += spike
        assert np.max(abs(m._calibrate(changed)[l]-m._archived_calibration(l))) < 3e-4
    keep = (cal['lab_id']!=l) | (np.arange(len(cal['lab_id']))==i)
    assert np.allclose(m._calibrate({k:v[keep] for k,v in cal.items()})[l],m._archived_calibration(l))
print('all four channels pass single-row fallback and injected EMI tests')
# Independent recurrence and near-zero slope check for basis ordering and sign.
tau = o['estimated_taus']
w = np.r_[o['hereditary_weights'],o['regime_mode_weights']]
z = np.zeros((len(hold),2))
for j,h in enumerate(dt):
    decay = np.exp(-h/tau)
    z = z*decay + past[:,j,None]*(1-decay)
gap = hold[:,None]-z
v = f @ w[2:7] + w[7]*np.sign(hold)*np.sqrt(abs(hold))
direct = (gap/tau) @ w[:2] + v + w[8]*(gap[:,0]-gap[:,1])/tau[0]
np.testing.assert_allclose(o['initial_rate_pred'],direct[n:],atol=1e-12)
b,_ = m._basis(tau,past,hold,f,dt,np.array([0.,1e-7]))
np.testing.assert_allclose((b[:,1]@w-b[:,0]@w)/1e-7,direct,rtol=0,atol=1e-6)
print('direct memory recurrence and strain derivative agree')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_rheology03.py
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`1448 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held-out lab 0 strain MAE/median AE, rate RMSE, taus 0.07217409367207908 0.013072314670585505 0.04924134625858242 [0.57450653 4.23244389]
held-out lab 1 strain MAE/median AE, rate RMSE, taus 0.13343725733668352 0.013864678144212342 0.03791318709568322 [0.57419887 4.25952644]
held-out lab 2 strain MAE/median AE, rate RMSE, taus 0.10306885054355798 0.015246463907767738 0.04063325019386633 [0.57452342 4.25511835]
held-out lab 3 strain MAE/median AE, rate RMSE, taus 0.22016208098292447 0.016063649977419492 0.05349640811069991 [0.57781645 4.2278063 ]
worst held-out lab/stratum median abs strain and rate RMSE (0.020649524077929637, 0, 2) (0.08179003703136277, 0, 2)
worst clean-test horizon RMSE 0.020929600788224936
positive 33 0.019327632968637397 0.024729620274827755
negative 39 0.014596551774454418 0.019138983318212788
large hold 0 nan nan
missing coupon 3 0.074373518702071 0.09510977448617498
all four channels pass single-row fallback and injected EMI tests
direct memory recurrence and strain derivative agree
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=678afb84309a474f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_678afb84 (directory).
```
- **Stderr (tail)**:
```text
/usr/local/lib/python3.12/dist-packages/numpy/core/fromnumeric.py:3504: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/usr/local/lib/python3.12/dist-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
```

### Turn 12 — `submit_completed_task` (`22510 ms` LLM)
- **Rationale**: The final artifact passed the isolated schema check, leave-one-lab-out validation, sparse and spiked calibration tests on all four channels, and independent checks of the memory recurrence and initial strain derivative.
- **Alternatives Weighed**: `Infer lab normalization or missing coupon values from public test outcomes` (rejected: That would replace the prescribed response-free panel estimator with a fit to test labels.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_03_solver.py': 'FILE:6408'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_03_solver.py`. All five outputs passed schema validation. Public test RMSE was 0.01693 for strain and 0.02188 for initial rate. The three samples with missing coupon features remained the least accurate slice.`

### Turn 13 — `execute_in_remote_docker` (`37844 ms` LLM)
- **Rationale**: Compare response-free imputation conventions using only held-out training responses. A clear gain on the worst missing-coupon lab without hurting complete cases would justify revisiting imputation. Independently check all nine curve and slope columns to catch a missing or mis-signed term that a weighted aggregate could hide.
- **Alternatives Weighed**: `Test relay modes and within-batch classification accuracy` (rejected: The task has continuous strain and rate responses and no relay state or class labels. Lab-by-stratum residuals are the relevant nuisance diagnostic.)

```bash
set -e
cd /holdout_task
cat >/tmp/rheology03_margin.py <<'PY'
import importlib.util, pickle
import numpy as np
s = importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_03.pkl','rb'))[0]['dataset']
tr, te = d['train'], d['test']; n = len(tr['lab_id'])
past, hold, f, e0 = m._prepare(d)
lab, st, raw = (np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','coupon_features_raw'))
missing = ~np.isfinite(raw)
observed = np.where(missing, np.nan, f[:,1:])
cell = np.array([[np.nanmean(observed[(lab==l)&(st==g)],axis=0) for g in range(3)] for l in range(4)])
variants = {'global_cell_balanced': f[:,1:], 'local_cell': cell[lab,st], 'stratum_balanced': np.nanmean(cell,axis=0)[st], 'lab_balanced': np.nanmean(cell,axis=1)[lab]}
y, r = tr['strain_curves_obs']-e0[:n,None], tr['instant_rate_obs']; dt, t = d['past_dt'], d['eval_times']
for key, fill in variants.items():
    feat = np.column_stack((f[:,0], np.where(missing,fill,observed)))[:n]
    ce, re = np.empty_like(y), np.empty_like(r)
    for l in range(4):
        v = lab[:n]==l; a = ~v
        tau,w=m._fit(past[:n][a],hold[:n][a],feat[a],dt,t,y[a],r[a])
        cb,rb=m._basis(tau,past[:n][v],hold[:n][v],feat[v],dt,t)
        ce[v],re[v]=cb@w-y[v],rb@w-r[v]
    ms=missing[:n].any(axis=1)
    def metrics(v): return round(np.median(abs(ce[v])),6),round(np.sqrt(np.mean(re[v]**2)),6)
    print(key, 'missing n=',ms.sum(),'missing OOF',metrics(ms),'complete OOF',metrics(~ms),'worst missing-lab rate RMSE',max(metrics(ms&(lab[:n]==l))[1] for l in np.unique(lab[:n][ms])))
    if key=='global_cell_balanced':
        for l in range(4):
            for g in range(3):
                v=(lab[:n]==l)&(st[:n]==g)
                print('OOF lab/stratum',l,g,'rate median bias',round(np.median(re[v]),6),'count',v.sum())
# Verify each basis column separately, so a cancellation cannot hide a missing term.
tau,w=m._fit(past[:n],hold[:n],f[:n],dt,t,y,r)
z=np.zeros((len(hold),2))
for j,h in enumerate(dt):
    q=np.exp(-h/tau); z=z*q+past[:,j,None]*(1-q)
gap=hold[:,None]-z
b,br=m._basis(tau,past,hold,f,dt,t)
u0=1-np.exp(-t/tau[0]); u1=1-np.exp(-t/tau[1]); phi=t/(1+.85*t)
slip=np.sign(hold)*np.sqrt(abs(hold)); delta=gap[:,0]-gap[:,1]
expected=np.stack([gap[:,0,None]*u0,gap[:,1,None]*u1,*[f[:,i,None]*phi for i in range(5)],slip[:,None]*phi,delta[:,None]*u0],axis=-1)
np.testing.assert_allclose(b,expected,rtol=0,atol=1e-12)
np.testing.assert_allclose(br,np.column_stack((gap/tau,f,slip,delta/tau[0])),rtol=0,atol=1e-12)
print('Independent term-by-term constitutive audit passed')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology03_margin.py
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`2543 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
global_cell_balanced missing n= 18 missing OOF (0.078085, 0.126172) complete OOF (0.012862, 0.022043) worst missing-lab rate RMSE 0.139705
OOF lab/stratum 0 0 rate median bias 0.007524 count 15
OOF lab/stratum 0 1 rate median bias -0.003847 count 13
OOF lab/stratum 0 2 rate median bias 0.017138 count 10
OOF lab/stratum 1 0 rate median bias -0.010137 count 12
OOF lab/stratum 1 1 rate median bias 0.001039 count 26
OOF lab/stratum 1 2 rate median bias -0.00157 count 19
OOF lab/stratum 2 0 rate median bias 0.015245 count 12
OOF lab/stratum 2 1 rate median bias 0.022505 count 8
OOF lab/stratum 2 2 rate median bias 0.008995 count 23
OOF lab/stratum 3 0 rate median bias 0.004089 count 8
OOF lab/stratum 3 1 rate median bias -0.00752 count 13
OOF lab/stratum 3 2 rate median bias -0.016115 count 21
local_cell missing n= 18 missing OOF (0.062149, 0.12729) complete OOF (0.012841, 0.022002) worst missing-lab rate RMSE 0.150486
stratum_balanced missing n= 18 missing OOF (0.071741, 0.128987) complete OOF (0.012939, 0.022147) worst missing-lab rate RMSE 0.153133
lab_balanced missing n= 18 missing OOF (0.075595, 0.125369) complete OOF (0.012836, 0.021966) worst missing-lab rate RMSE 0.131247
Independent term-by-term constitutive audit passed
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=678afb84309a474f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_678afb84 (directory).
```

### Turn 14 — `submit_completed_task` (`17203 ms` LLM)
- **Rationale**: The final artifact passed clean-process schema validation, independent checks of all nine constitutive basis terms, and sparse and spiked calibration tests on all four channels. Leave-one-lab-out validation identified missing coupon values as the remaining weakness without establishing a better imputation rule within the specified estimator.
- **Alternatives Weighed**: `Use lab-specific imputation` (rejected: It modestly improved worst-lab missing-coupon rate RMSE on 18 training samples, but changed which cells contribute to the pooled cell-balanced estimator.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_03_solver.py`. All five outputs passed schema validation. Public test RMSE was 0.01693 for strain and 0.02188 for initial rate. The three samples with missing coupon features remained the least accurate slice, at 0.07437 and 0.09511 RMSE.`

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
