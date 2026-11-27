# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c02_s00__rheology_campaign_baseline`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c02_source` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `308,999` (`296,751` / `107,520` / `12,248`) |
| **Total Episode Wall Time (s)** | `388.33s` |

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

### Turn 1 — `execute_in_remote_docker` (`16087 ms` LLM)
- **Rationale**: Inspecting the validator and compact data summaries will expose schema or convention mismatches that would change the implementation.

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && python3 - <<'PY'
import pickle,numpy as np
with open('public_alpha_m97_c02_s00.pkl','rb') as f: d=pickle.load(f)
def describe(x,depth=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print('  '*depth,k,end=': ')
   if isinstance(v,dict): print(); describe(v,depth+1)
   elif isinstance(v,np.ndarray): print(v.shape,v.dtype,repr(v.ravel()[:7]))
   elif isinstance(v,(list,tuple)): print(type(v).__name__,len(v),repr(v[:1])[:250])
   else: print(repr(v)[:250])
describe(d)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS viscoelastic relaxation reference panel alignment'
```
- **Container Exit Code**: `0` (`393 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
le)
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
```

### Turn 2 — `execute_in_remote_docker` (`19851 ms` LLM)
- **Rationale**: The previous stdout omitted the input summary. A compact dataset-only inspection will establish the panel and time layouts and determine whether any channel needs archive calibration. The primitive descriptions will show whether their normalization and fitting conventions match the contract.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
with open('public_alpha_m97_c02_s00.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
def show(d,indent=''):
 for k,v in d.items():
  if isinstance(v,dict): print(indent+k+':'); show(v,indent+'  ')
  elif isinstance(v,np.ndarray):
   print(indent+k,v.shape,v.dtype, 'values', (v if v.size<=20 else v.ravel()[:5]), 'nonfinite', np.count_nonzero(~np.isfinite(v)) if v.dtype.kind in 'fiu' else '-')
  else: print(indent+k,type(v).__name__,str(v)[:150])
show(d)
c=d['calibration_load_cells']
for l in np.unique(c['lab_id']):
 m=c['lab_id']==l; a=c['cmd_stress'][m]; b=c['true_stress'][m]; p=np.polyfit(a,b,1); print('cal',l,len(a),p,'max abs residual',np.max(np.abs(b-np.polyval(p,a))))
PY
export PYTHONPATH=/tmp
python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit
python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel
python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.weighted_column_fill
```
- **Container Exit Code**: `0` (`529 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
nempty real matrix of shape (n, p) and finite, strictly positive row weights of shape (n,). Nonfinite entries are excluded from both numerator and denominator; a column with no finite observations returns zero. Numeric panel and fill arithmetic use float64 and assume intermediate sums, squares, and ratios are within usable float64 range."
      ],
      "counter_regime_guards": [],
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

### Turn 3 — `execute_in_remote_docker` (`80992 ms` LLM)
- **Rationale**: The isolated schema check will catch packaging or shape failures. Per-channel and per-time residuals will reveal whether noise scaling, panel alignment, or the rate model needs revision before held-out validation.
- **Alternatives Weighed**: `Use the single-relaxation primitive directly` (rejected: It does not fit two history-dependent branches with shared coefficients and both response types.)

```bash
cat > /holdout_task/alpha_m97_c02_s00_solver.py <<'PY'
"""Fit the two-branch linear-flow rheology model from calibrated observations."""

import numpy as np
from scipy.optimize import least_squares


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _calibration(data):
    c = data['calibration_load_cells']
    result = np.empty((4, 2))
    for lab in range(4):
        take = (c['lab_id'] == lab) & np.isfinite(c['cmd_stress']) & np.isfinite(c['true_stress'])
        x, y = c['cmd_stress'][take], c['true_stress'][take]
        clean = np.ones(len(x), dtype=bool)
        if len(x) >= 3:
            dx = x[:, None] - x
            dy = y[:, None] - y
            slopes = np.divide(dy, dx, out=np.full_like(dx, np.nan), where=dx != 0)
            gain = np.nanmedian(np.nanmedian(slopes, axis=1))
            clean = np.abs(y - gain*x - np.median(y - gain*x)) <= 3.0
        if clean.sum() >= 2 and (clean.all() or clean.sum() >= 4):
            result[lab] = _line(x[clean], y[clean])
        else:
            result[lab] = _archive_calibration(lab, x[clean], y[clean])
    return result


def _archive_calibration(lab, x, y):
    import json
    try:
        with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as file:
            records = json.load(file)
    except FileNotFoundError:
        return _line(x, y) if len(x) >= 2 else (1.0, y[0] - x[0] if len(x) else 0.0)

    def rows(node):
        if isinstance(node, dict):
            if {'lab_id', 'cmd_stress', 'true_stress'} <= node.keys():
                yield from zip(*(np.atleast_1d(node[key]) for key in ('lab_id', 'cmd_stress', 'true_stress')))
            else:
                for value in node.values():
                    yield from rows(value)
        elif isinstance(node, list):
            for value in node:
                yield from rows(value)

    pairs = np.array([(u, v) for c, u, v in rows(records) if int(c) == lab and np.isfinite(u) and np.isfinite(v)]).reshape(-1, 2)
    if len(pairs) >= 2:
        gain, bias = _line(*pairs.T)
        if len(x):
            bias = np.mean(y - gain*x)
        return gain, bias
    return _line(x, y) if len(x) >= 2 else (1.0, y[0] - x[0] if len(x) else 0.0)


def _align(data):
    train, test = data['train'], data['test']
    x, ref, lab, stratum = (np.concatenate((train[k], test[k]), axis=0) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    for c in np.unique(lab):
        sel = lab == c
        panel = ref[sel]
        shift = np.nanmedian(np.nanmean(panel, axis=0))
        gain = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[sel] = (x[sel] - shift)/gain
    means = []
    for c, s in np.unique(np.column_stack((lab, stratum)), axis=0):
        part = x[(lab == c) & (stratum == s)]
        count = np.isfinite(part).sum(axis=0)
        means.append(np.divide(np.nansum(part, axis=0), count,
                               out=np.full(x.shape[1], np.nan), where=count > 0))
    return np.where(np.isfinite(x), x, np.nanmean(means, axis=0))


def _physical(split, calibration):
    gain, bias = calibration[split['lab_id']].T
    return split['past_stress_cmd']*gain[:, None] + bias[:, None], split['stress_hold_cmd']*gain + bias


def _design(logtau, past, hold, dt, time, features, derivatives=False):
    tau = np.exp(logtau)
    u = dt[:, None]/tau
    decay = np.exp(-u)
    update = -np.expm1(-u)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for j in range(len(dt)):
        if derivatives:
            dz = dz*decay[j] + (z - past[:, j, None])*(u[j]*decay[j])
        z = z*decay[j] + past[:, j, None]*update[j]
    gap = hold[:, None] - z
    q = time[:, None]/tau
    accum = -np.expm1(-q)
    n, k = len(hold), len(time)
    curves = np.empty((n, k, 7))
    curves[:, :, :2] = gap[:, None, :]*accum
    curves[:, :, 2:] = features[:, None, :]*time[None, :, None]
    rates = np.column_stack((gap/tau, features))
    if not derivatives:
        return curves, rates
    dc = -dz[:, None, :]*accum - gap[:, None, :]*(q*np.exp(-q))
    dr = (-dz - gap)/tau
    return curves, rates, dc, dr


def _mad(x):
    return max(1.4826*np.median(np.abs(x - np.median(x))), 1e-7)


def _fit(past, hold, dt, time, features, strains, instant):
    y = np.r_[strains.ravel(), instant]
    n_curves = strains.size
    scale = np.ones_like(y)
    lower = np.r_[np.log(min(dt.min(), time[time > 0].min()) / 15), np.log(min(dt.min(), time[time > 0].min()) / 15), np.full(7, -np.inf)]
    upper = np.r_[np.log(40*max(time.max(), dt.sum())), np.log(40*max(time.max(), dt.sum())), np.full(7, np.inf)]

    def residual(p, jac=False):
        curves, rates, dc, dr = _design(p[:2], past, hold, dt, time, features, True)
        linear = np.concatenate((curves.reshape(-1, 7), rates), axis=0)
        if jac:
            derivative = np.concatenate((dc.reshape(-1, 2), dr), axis=0)
            return np.column_stack((derivative*p[2:4], linear))/scale[:, None]
        return (linear @ p[2:] - y)/scale

    starts = [(0.35, 1.8), (0.8, 4.0), (0.18, 0.8), (1.5, 9.0)]
    start = np.log(starts[0])
    curves, rates = _design(start, past, hold, dt, time, features)
    linear = np.concatenate((curves.reshape(-1, 7), rates), axis=0)
    pilot = np.r_[start, np.linalg.lstsq(linear, y, rcond=None)[0]]
    error = residual(pilot)

    def set_scale(err):
        scale[:n_curves] = 1.8*_mad(err[:n_curves])
        scale[n_curves:] = 1.8*_mad(err[n_curves:])

    set_scale(error)
    def optimize(p, loss):
        return least_squares(residual, p, jac=lambda p: residual(p, True),
                             bounds=(lower, upper), loss=loss, x_scale='jac', max_nfev=100)

    pilot = optimize(pilot, 'soft_l1').x
    set_scale(residual(pilot)*scale)
    best = optimize(pilot, 'cauchy')
    for pair in starts[1:]:
        start = np.log(pair)
        curves, rates = _design(start, past, hold, dt, time, features)
        linear = np.concatenate((curves.reshape(-1, 7), rates), axis=0)
        weights = 1 / np.sqrt(1 + residual(best.x)**2) / scale
        p = np.r_[start, np.linalg.lstsq(linear*weights[:, None], y*weights, rcond=None)[0]]
        fit = optimize(p, 'cauchy')
        if fit.cost < best.cost:
            best = fit
    for _ in range(2):
        set_scale(residual(best.x)*scale)
        best = optimize(best.x, 'cauchy')
    p = best.x.copy()
    if p[0] > p[1]:
        p[:2] = p[1::-1]
        p[2:4] = p[3:1:-1]
    return p[:2], p[2:]


def solve_alpha_m97_c02_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    cal = _calibration(dataset)
    aligned = _align(dataset)
    features = np.column_stack((np.ones(len(aligned)), aligned))
    n = len(train['lab_id'])
    past, hold = _physical(train, cal)
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    strains = train['strain_curves_obs'] - train['strain_0'][:, None]
    logtau, hw = _fit(past, hold, dt, time, features[:n], strains, train['instant_rate_obs'])
    past, hold = _physical(test, cal)
    curves, rates = _design(logtau, past, hold, dt, time, features[n:])
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': hw,
            'regime_mode_weights': np.zeros(2), 'predicted_strain': test['strain_0'][:, None] + curves @ hw,
            'initial_rate_pred': rates @ hw}
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np,alpha_m97_c02_s00_solver as s
D=pickle.load(open('public_alpha_m97_c02_s00.pkl','rb'))[0]['dataset']
print('calibration',s._calibration(D)); out=s.solve_alpha_m97_c02_s00(D); print('fit',out['estimated_taus'],out['hereditary_weights']); p,h=s._physical(D['train'],s._calibration(D)); print('max past stress',abs(p).max()); x=s._align(D)[:len(p)]; a,b=s._design(np.log(out['estimated_taus']),p,h,D['past_dt'],D['eval_times'],np.c_[np.ones(len(p)),x]); w=out['hereditary_weights']; tr=D['train'];
for label,r in [('curve',a@w+tr['strain_0'][:,None]-tr['strain_curves_obs']),('rate',b@w-tr['instant_rate_obs'])]:
 print(label,'abs residual quantiles',np.quantile(abs(r),[.25,.5,.75,.9,.95,.99,1])); print('median lab x time',np.array([np.median(abs(r[tr['lab_id']==l]),axis=0) for l in range(4)]))
PY
```
- **Container Exit Code**: `0` (`852 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
calibration [[ 9.99976805e-01  1.21877020e-06]
 [ 9.99993252e-01  3.53543932e-05]
 [ 1.00000751e+00  2.72105742e-05]
 [ 1.00000953e+00 -3.06336064e-05]]
fit [0.57190732 3.57479712] [ 0.68604784  0.55013119  0.1404703   0.16712926 -0.1379927   0.1178265
 -0.08823144]
max past stress 2.0670503266848597
curve abs residual quantiles [7.00449443e-03 1.50280075e-02 3.01852866e-02 9.65788647e-02
 3.06217984e-01 1.14632595e+00 6.82770120e+02]
median lab x time [[0.01172418 0.01169917 0.0124424  0.02008446 0.03092947 0.01852878
  0.03422454 0.04714414 0.04373909 0.04993356]
 [0.01169136 0.01502207 0.01058356 0.01213109 0.01176437 0.01523778
  0.01222821 0.01649132 0.01322084 0.01071885]
 [0.01110646 0.00994911 0.01398444 0.01314468 0.00971453 0.01398343
  0.01466299 0.01370009 0.01721811 0.01039154]
 [0.01098769 0.00945871 0.01842521 0.0121936  0.01654662 0.01719702
  0.01647239 0.0194914  0.01997186 0.0265377 ]]
rate abs residual quantiles [0.00711436 0.01468679 0.02481217 0.036007   0.04860575 0.17904395
 0.18658352]
median lab x time [0.01523777 0.01674597 0.0122932  0.01489121]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d7487a85d37a3147 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d7487a85 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`25661 ms` LLM)
- **Rationale**: Leave-one-lab-out predictions will show whether the late-time residuals generalize across labs. Comparing complete and incomplete coupons will test whether to revisit the imputation or the shared flow fit.

```bash
cat > /tmp/m97_validate.py <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_m97_c02_s00_solver as s
D=pickle.load(open('/holdout_task/public_alpha_m97_c02_s00.pkl','rb'))[0]['dataset']; T=D['train']; n=len(T['lab_id']); dt,t=D['past_dt'],D['eval_times']; p,h=s._physical(T,s._calibration(D)); a=s._align(D); X=np.c_[np.ones(len(a)),a][:n]; y=T['strain_curves_obs']-T['strain_0'][:,None]; v=T['instant_rate_obs']; labs=T['lab_id']; miss=~np.isfinite(T['coupon_features_raw']); missing=miss.any(axis=1)

def fit(mask):
 u,w=s._fit(p[mask],h[mask],dt,t,X[mask],y[mask],v[mask]); A,B=s._design(u,p,h,dt,t,X); return np.exp(u),w,A@w-y,B@w-v

def summarize(r,q,mask):
 return np.round(np.r_[np.median(abs(r[mask])),np.max(np.median(abs(r[mask]),axis=0)),np.median(abs(q[mask]))],6)
for l in range(4):
 mask=labs==l; u,w,r,q=fit(~mask); print('leave lab',l,'taus',np.round(u,5),'all',summarize(r,q,mask),'complete',summarize(r,q,mask&~missing),'missing n',sum(mask&missing));
 if (mask&missing).any():print('  missing',summarize(r,q,mask&missing))
u,w,r,q=fit(~missing); print('complete-case coefficients',u,w); print('complete in sample',summarize(r,q,~missing)); print('missing case detail: lab,stratum,missing cols,observed-minus-pred row drift and rate');
for i in np.flatnonzero(missing):print(labs[i],T['stratum'][i],np.flatnonzero(miss[i]),round(-np.median(r[i]/t),5),round(-q[i],5))
print('cell counts',np.unique(np.c_[labs,T['stratum']],return_counts=True,axis=0))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m97_validate.py
```
- **Container Exit Code**: `0` (`941 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
leave lab 0 taus [0.57332 3.71043] all [0.024057 0.056796 0.016444] complete [0.023286 0.056308 0.015983] missing n 3
  missing [0.044697 0.067981 0.017614]
leave lab 1 taus [0.57149 3.56418] all [0.013826 0.01671  0.016444] complete [0.013035 0.015232 0.015244] missing n 3
  missing [0.09505  0.269825 0.030871]
leave lab 2 taus [0.57176 3.49203] all [0.014102 0.021338 0.013256] complete [0.012417 0.018776 0.010942] missing n 6
  missing [0.151963 0.384411 0.078825]
leave lab 3 taus [0.57049 3.48265] all [0.017959 0.030703 0.015026] complete [0.017644 0.027582 0.014475] missing n 1
  missing [0.62408  1.134153 0.181569]
complete-case coefficients [0.5716012  3.56288735] [ 0.6848739   0.55074255  0.14055929  0.16716556 -0.1380351   0.11765022
 -0.08818995]
complete in sample [0.013662 0.018006 0.013526]
missing case detail: lab,stratum,missing cols,observed-minus-pred row drift and rate
1 2 [1] -0.05003 -0.02835
1 0 [0] 0.01714 0.03321
2 0 [3] 0.02696 0.02785
2 2 [3] -0.16627 -0.17695
1 1 [0] -0.08768 -0.06543
0 0 [0] -0.0057 -0.01788
2 2 [2] 0.07919 0.08998
0 2 [2] 0.17064 0.16785
2 0 [0] -0.0264 -0.0347
2 0 [2] -0.06582 -0.06494
0 0 [0] -0.01493 -0.0003
2 2 [1] -0.17034 -0.18621
3 0 [2] -0.21855 -0.18465
cell counts (array([[0, 0],
       [0, 1],
       [0, 2],
       [1, 0],
       [1, 1],
       [1, 2],
       [2, 0],
       [2, 1],
       [2, 2],
       [3, 0],
       [3, 1],
       [3, 2]]), array([18, 15,  9,  6, 20, 14, 22, 11, 27, 14, 10, 14]))

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d7487a85d37a3147 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d7487a85 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`35762 ms` LLM)
- **Rationale**: The largest missing-feature error comes from a one-coupon slice. Pooling held-out incomplete coupons will show whether the interpretation of cell-balanced fill materially changes prediction, while signed complete-coupon residuals distinguish that issue from lab-wide drift. Inspecting archive keys will reveal whether the conditional calibration reader matches the mounted format.
- **Alternatives Weighed**: `Add lab-specific drift coefficients` (rejected: The specified model shares drift coefficients, and a correction learned from a withheld lab would not validate transfer to it.)

```bash
cat > /tmp/m97_slices.py <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_m97_c02_s00_solver as s
with open('/holdout_task/public_alpha_m97_c02_s00.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
tr,te=d['train'],d['test']; n=len(tr['lab_id']); labs=tr['lab_id']; t=d['eval_times']; dt=d['past_dt']; cal=s._calibration(d); p,h=s._physical(tr,cal)
x,ref,g,b=[np.concatenate((tr[k],te[k])) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')]
missing=~np.isfinite(x[:n]).all(1)
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; v=tr['instant_rate_obs']
for l in range(4):
 m=g==l; shift=np.nanmedian(np.nanmean(ref[m],0)); scale=np.nanmedian(np.nanstd(ref[m],0)); print('panel',l,'shift',round(shift,5),'scale',round(scale,5)); x[m]=(x[m]-shift)/scale
cells=np.unique(np.c_[g,b],axis=0); means=np.array([np.nanmean(x[(g==l)&(b==j)],0) for l,j in cells]); default=np.nanmean(means,0)
for mode in ('global','local','stratum'):
 z=x.copy()
 for i,(l,j) in enumerate(cells):
  m=(g==l)&(b==j); pick={'global':default,'local':means[i],'stratum':np.nanmean(means[cells[:,1]==j],0)}[mode]; z[m]=np.where(np.isfinite(z[m]),z[m],pick)
 X=np.c_[np.ones(n),z[:n]]
 if mode=='global': np.testing.assert_allclose(z,s._align(d))
 misses=[]; scores=[]; drift=[]
 for l in range(4):
  m=labs==l; u,w=s._fit(p[~m],h[~m],dt,t,X[~m],y[~m],v[~m]); A,B=s._design(u,p[m],h[m],dt,t,X[m]); r=A@w-y[m]; q=B@w-v[m]; scores.append([np.max(np.median(abs(r),0)),np.median(abs(q))]); misses.append(np.c_[np.median(abs(r[missing[m]]),1),abs(q[missing[m]])]); drift.append(np.median((r/t)[~missing[m]],axis=0)[-1])
 a=np.concatenate(misses); print(mode,'worst lab time / rate',np.max(scores,0),'missing medians',np.median(a,0),'missing p90',np.quantile(a,.9,axis=0),'signed complete late drift',np.round(drift,6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m97_slices.py
python3 -I - <<'PY'
import json,os
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
if os.path.isfile(p):
 with open(p) as f: a=json.load(f)
 print('archive structure:',type(a).__name__,list(a)[:12] if isinstance(a,dict) else type(a[0]).__name__ if a else 'empty')
 if isinstance(a,list) and a and isinstance(a[0],dict): print('record keys:',list(a[0]))
 if isinstance(a,dict):
  for k,v in list(a.items())[:6]: print(k,type(v).__name__,list(v)[:10] if isinstance(v,dict) else list(v[0])[:10] if isinstance(v,list) and v and isinstance(v[0],dict) else '')
else: print('calibration archive not mounted')
PY
```
- **Container Exit Code**: `0` (`1653 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
panel 0 shift -0.62771 scale 0.76826
panel 1 shift 0.45269 scale 1.25386
panel 2 shift -0.24619 scale 0.92821
panel 3 shift 0.83914 scale 1.43027
global worst lab time / rate [0.05679564 0.01644387] missing medians [0.18410482 0.06542364] missing p90 [0.4708372  0.18078504] signed complete late drift [-2.082e-03  2.143e-03 -1.600e-05 -2.651e-03]
local worst lab time / rate [0.06161808 0.01803274] missing medians [0.17357503 0.066338  ] missing p90 [0.45183632 0.16609143] signed complete late drift [-0.001988  0.002542  0.000279 -0.002529]
stratum worst lab time / rate [0.06147772 0.01801151] missing medians [0.17591968 0.06791987] missing p90 [0.43170813 0.16353121] signed complete late drift [-0.002016  0.002459  0.000256 -0.002356]
calibration archive not mounted

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d7487a85d37a3147 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d7487a85 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`35520 ms` LLM)
- **Rationale**: Public folds measure transfer but cannot reveal a shared sign, calibration, or time-convention error. An independent closed-form test with nonidentity gains, two relaxation pairs, and 7% Cauchy glitches checks those failure modes; a mismatch would warrant correcting the affected equation or fit.
- **Alternatives Weighed**: `Use lab-local or stratum-local imputation` (rejected: Neither implements global cell balancing, and both worsened the worst held-out lab strain and rate medians.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c02_s00_solver.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m97_c02_s00.pkl','rb') as f: d = pickle.load(f)[0]['dataset']
rng = np.random.default_rng(920)
t = d['eval_times']; dt = d['past_dt']; c = d['calibration_load_cells']; g = np.array([.82, 1.16, 1.31, .91]); b = np.array([-.11, .09, .14, -.06]); c['true_stress'] = g[c['lab_id']]*c['cmd_stress'] + b[c['lab_id']]
tr, te = d['train'], d['test']; lab = np.r_[tr['lab_id'],te['lab_id']]; strata = np.r_[tr['stratum'],te['stratum']]; raw = np.r_[tr['coupon_features_raw'],te['coupon_features_raw']]; panel = np.r_[tr['ref_coupons'],te['ref_coupons']]; x = raw.copy()
for k in range(4):
 m = lab == k; x[m] = (raw[m]-np.median(np.nanmean(panel[m],axis=0)))/np.median(np.nanstd(panel[m],axis=0))
fill = np.nanmean(np.array([np.nanmean(x[(lab==a)&(strata==b0)],axis=0) for a,b0 in sorted(set(zip(lab,strata)))]),axis=0); x = np.where(np.isnan(x),fill,x)
w = np.array([.68,.55,.14,.17,-.14,.12,-.09]); n = len(tr['lab_id'])
for tau in (np.array([.29,1.65]),np.array([.87,6.1])):
 def truth(split, features):
  ids = split['lab_id']; past = split['past_stress_cmd']*g[ids,None] + b[ids,None]; hold = split['stress_hold_cmd']*g[ids]+b[ids]; gap = np.column_stack([hold - past @ ((1-np.exp(-dt/q))*np.exp(-(dt.sum()-np.cumsum(dt))/q)) for q in tau]); drift = w[2] + features@w[3:]; return (split['strain_0'][:,None] + (gap*w[:2]) @ (1-np.exp(-t[None,:]/tau[:,None])) + drift[:,None]*t, (gap*w[:2]) @ (1/tau) + drift)
 y, rate = truth(tr,x[:n]); target, target_rate = truth(te,x[n:]); ix = rng.choice(y.size,round(.07*y.size),replace=False); y.flat[ix] += .4*rng.standard_cauchy(len(ix)); tr['strain_curves_obs'] = y; tr['instant_rate_obs'] = rate
 out = s.solve_alpha_m97_c02_s00(d); err = np.abs(out['predicted_strain']-target); erate = np.abs(out['initial_rate_pred']-target_rate); print('taus',tau,'recovered',out['estimated_taus'],'max strain error by lab',[float(err[te['lab_id']==j].max()) for j in range(4)],'max rate error',erate.max()); np.testing.assert_allclose(out['estimated_taus'],tau,rtol=.02); np.testing.assert_allclose(out['hereditary_weights'],w,rtol=.04); np.testing.assert_array_equal(out['regime_mode_weights'],np.zeros(2)); assert err.max()<.01 and erate.max()<.01
PY
```
- **Container Exit Code**: `0` (`1180 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
taus [0.29 1.65] recovered [0.29 1.65] max strain error by lab [1.0462741784067475e-12, 1.5893952820533741e-12, 2.2377655284344655e-12, 1.3877787807814457e-12] max rate error 8.597567102697212e-13
taus [0.87 6.1 ] recovered [0.87 6.1 ] max strain error by lab [3.552713678800501e-14, 2.7533531010703882e-14, 3.1308289294429414e-14, 3.7969627442180354e-14] max rate error 1.865174681370263e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d7487a85d37a3147 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d7487a85 (directory).
```

### Turn 7 — `submit_completed_task` (`16280 ms` LLM)
- **Rationale**: The exact final artifact passed the isolated schema check, held-out lab evaluation, and independent tests of calibration, memory propagation, and initial rates. No further edit is warranted by the observed results.
- **Alternatives Weighed**: `Lab-local or stratum-local imputation` (rejected: Both worsened the worst held-out lab strain and rate medians and neither implements global cell balancing.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_s00_solver.py': 'FILE:7377'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_s00_solver.py`. The public schema passed, and both synthetic stress tests recovered the relaxation times under 7% Cauchy glitches. The optional calibration archive was not mounted, so its integration was not verified.`

### Turn 8 — `execute_in_remote_docker` (`36041 ms` LLM)
- **Rationale**: Negative stress margins would invalidate the assumed regime, and derivative mismatches would warrant fixing the optimizer Jacobian. Comparing held-out residuals across and within labs tests whether the worst slice reflects systematic drift or larger error spread before changing the fit.
- **Alternatives Weighed**: `Fit rational corrections, relay modes, or a classification model` (rejected: The task specifies continuous strain and rate targets in a linear-flow regime with zero mode weights. The relevant checks are regime margins, exact branch derivatives, and pooled versus within-lab residual patterns.)

```bash
cat > /tmp/m97_gate.py <<'PY'
import importlib.util, pickle
import numpy as np
from scipy.stats import spearmanr

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c02_s00_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open(root+'public_alpha_m97_c02_s00.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
cal = s._calibration(d)
p, h = s._physical(tr, cal)
pte, hte = s._physical(te, cal)
print('linear regime margins: banding', .50-d['shear_banding_index'], 'past stress train/test', [d['crit_stress']-np.abs(a).max() for a in (p,pte)])
n = len(p)
X = np.c_[np.ones(n), s._align(d)[:n]]
y = tr['strain_curves_obs']-tr['strain_0'][:,None]
v = tr['instant_rate_obs']
lab, stratum = tr['lab_id'], tr['stratum']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
r, q = np.empty_like(y), np.empty_like(v)
for l in np.unique(lab):
    held = lab == l
    u,w = s._fit(p[~held],h[~held],dt,t,X[~held],y[~held],v[~held])
    a,b = s._design(u,p[held],h[held],dt,t,X[held])
    r[held], q[held] = a@w-y[held], b@w-v[held]
row = np.median(r[:, -4:]/t[-4:], axis=1)
print('held-out complete coupons by lab: n, signed late slope, median absolute late slope, signed rate, median absolute rate')
for l in np.unique(lab):
    m = (lab == l) & complete
    print(l, m.sum(), np.round([np.median(row[m]),np.median(abs(row[m])),np.median(q[m]),np.median(abs(q[m]))],6))
print('held-out complete coupons by stratum: n, signed late slope and rate')
for j in np.unique(stratum):
    m = (stratum == j) & complete
    print(j, m.sum(), np.round([np.median(row[m]),np.median(q[m])],6))
for name,x in [('hold stress',h),('abs hold stress',abs(h)),('peak history stress',abs(p).max(1))]:
    def rho(m):
        return [spearmanr(x[m],z[m]).statistic for z in (row,q)]
    print(name, 'pooled then within lab late slope / rate rank correlations', np.round([rho(complete)]+[rho(complete & (lab == l)) for l in np.unique(lab)],3).tolist())
u = np.log(s.solve_alpha_m97_c02_s00(d)['estimated_taus'])
_,_,da,db = s._design(u,p,h,dt,t,X,True)
for j in range(2):
    step = np.eye(2)[j]*1e-5
    ap,bp = s._design(u+step,p,h,dt,t,X)
    am,bm = s._design(u-step,p,h,dt,t,X)
    np.testing.assert_allclose((ap-am)/(2e-5), np.eye(7)[j]*da[:,:,j,None], atol=1e-8)
    np.testing.assert_allclose((bp-bm)/(2e-5), np.eye(7)[j]*db[:,j,None], atol=1e-8)
print('Both branch Jacobians agree with finite differences on measured histories.')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m97_gate.py
```
- **Container Exit Code**: `0` (`1202 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
linear regime margins: banding 0.38 past stress train/test [0.5829496733151402, 0.5829496733151402]
held-out complete coupons by lab: n, signed late slope, median absolute late slope, signed rate, median absolute rate
0 39 [-0.001683  0.00821  -0.003176  0.015983]
1 37 [0.002473 0.002696 0.01285  0.015244]
2 54 [-0.000447  0.002508  0.002784  0.010942]
3 37 [-0.00272   0.004277  0.000223  0.014475]
held-out complete coupons by stratum: n, signed late slope and rate
0 53 [-0.000185  0.002478]
1 55 [0.000836 0.005796]
2 59 [-0.00034  -0.006487]
hold stress pooled then within lab late slope / rate rank correlations [[0.02, 0.065], [0.007, 0.053], [0.075, -0.007], [0.068, -0.061], [-0.329, 0.145]]
abs hold stress pooled then within lab late slope / rate rank correlations [[0.03, -0.025], [0.036, 0.006], [-0.188, -0.122], [0.024, 0.152], [-0.066, -0.369]]
peak history stress pooled then within lab late slope / rate rank correlations [[-0.049, -0.006], [-0.136, -0.126], [-0.075, 0.067], [-0.047, -0.132], [0.181, 0.2]]
Both branch Jacobians agree with finite differences on measured histories.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d7487a85d37a3147 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d7487a85 (directory).
```

### Turn 9 — `submit_completed_task` (`15702 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated schema validation, independent parameter recovery, and both branch Jacobian checks. The public held-out audit found no evidence that warrants changing the specified flow law.
- **Alternatives Weighed**: `Add lab-specific or nonlinear stress corrections` (rejected: Measured histories remained below the regime threshold, and held-out residuals did not show a consistent stress dependence across labs.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_s00_solver.py': 'FILE:7377'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_s00_solver.py`. The public schema passed, and both synthetic stress tests recovered the relaxation times with 7% Cauchy glitches. Calibration archive integration was not verified because the archive was not mounted.`

### Turn 10 — `execute_in_remote_docker` (`32587 ms` LLM)
- **Rationale**: An independent convolution on measured histories would expose a memory indexing or sign error, and the paired-history identities would expose inconsistent branch labeling. The default-environment run checks whether earlier validation depended on forcing one BLAS thread.
- **Alternatives Weighed**: `Add rational corrections, stochastic-control operators, or relay states` (rejected: The measured histories are inside the prescribed linear regime, which has continuous strain and rate targets and no relay state. The relevant algebraic checks are superposition and branch exchange.)

```bash
cd /holdout_task && /usr/bin/time -f 'default-environment schema check: %e s' python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
path = '/holdout_task/alpha_m97_c02_s00_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m97_c02_s00.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
out = s.solve_alpha_m97_c02_s00(d)
tau, w = out['estimated_taus'], out['hereditary_weights']
dt, t = d['past_dt'], d['eval_times']
x = s._align(d)
cal = s._calibration(d)
offset = 0
for key in ('train', 'test'):
    split = d[key]
    p, h = s._physical(split, cal)
    features = np.c_[np.ones(len(h)), x[offset:offset+len(h)]]
    offset += len(h)
    a, b = s._design(np.log(tau), p, h, dt, t, features)
    duration_after = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    kernel = -np.expm1(-dt[:,None]/tau) * np.exp(-duration_after[:,None]/tau)
    gap = h[:,None] - p @ kernel
    expected = a @ w
    expected_rate = b @ w
    np.testing.assert_allclose(expected, (gap*w[:2]) @ -np.expm1(-t[None,:]/tau[:,None]) + (features @ w[2:])[:,None]*t, atol=1e-12)
    np.testing.assert_allclose(expected_rate, (gap*w[:2]) @ (1/tau) + features @ w[2:], atol=1e-12)
    for sign in (-1,1):
        coeff = sign*.25
        conj = lambda q: q + coeff*q[::-1]
        ap,bp = s._design(np.log(tau[::-1]), conj(p), conj(h), dt, t, conj(features))
       /replacement = None
        v = np.r_[w[1::-1], w[2:]]
        np.testing.assert_allclose(ap @ v, conj(expected), atol=1e-12)
        np.testing.assert_allclose(bp @ v, conj(expected_rate), atol=1e-12)
    print(key, 'closed-form memory, branch exchange, and paired sums/differences agree to 1e-12')
PY
```
- **Container Exit Code**: `127` (`287 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d7487a85d37a3147 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d7487a85 (directory).
```
- **Stderr (tail)**:
```text
bash: line 1: /usr/bin/time: No such file or directory
```

### Turn 11 — `execute_in_remote_docker` (`28461 ms` LLM)
- **Rationale**: The previous command stopped before verification because `/usr/bin/time` was missing. Bash timing tests the isolated solver without a BLAS override; the closed-form comparison would identify a memory indexing, branch labeling, or output assembly error if it failed.

```bash
cd /holdout_task && time python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'alpha_m97_c02_s00_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('public_alpha_m97_c02_s00.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
out = s.solve_alpha_m97_c02_s00(d)
tau, w = out['estimated_taus'], out['hereditary_weights']
dt, t = d['past_dt'], d['eval_times']
x, cal = s._align(d), s._calibration(d)
remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
kernel = -np.expm1(-dt[:,None]/tau) * np.exp(-remaining[:,None]/tau)
step = -np.expm1(-t[None,:]/tau[:,None])
offset = 0
for key in ('train','test'):
    split = d[key]
    p, h = s._physical(split, cal)
    X = np.c_[np.ones(len(h)), x[offset:offset+len(h)]]
    offset += len(h)
    gap = h[:,None] - p @ kernel
    drift = X @ w[2:]
    curve = (gap*w[:2]) @ step + drift[:,None]*t
    rate = (gap*w[:2]) @ (1/tau) + drift
    a,b = s._design(np.log(tau), p, h, dt, t, X)
    np.testing.assert_allclose(a @ w, curve, rtol=0, atol=1e-12)
    np.testing.assert_allclose(b @ w, rate, rtol=0, atol=1e-12)
    if key == 'test':
        np.testing.assert_allclose(out['predicted_strain'], split['strain_0'][:,None]+curve, rtol=0, atol=1e-12)
        np.testing.assert_allclose(out['initial_rate_pred'], rate, rtol=0, atol=1e-12)
    for c in (-.25,.25):
        mix = lambda v: v + c*v[::-1]
        ap,bp = s._design(np.log(tau[::-1]), mix(p), mix(h), dt, t, mix(X))
        wp = np.r_[w[1::-1], w[2:]]
        np.testing.assert_allclose(ap @ wp, mix(curve), rtol=0, atol=1e-12)
        np.testing.assert_allclose(bp @ wp, mix(rate), rtol=0, atol=1e-12)
    print(key, 'closed-form memory, branch exchange, and paired sums/differences agree within 1e-12')
PY
```
- **Container Exit Code**: `0` (`932 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
train closed-form memory, branch exchange, and paired sums/differences agree within 1e-12
test closed-form memory, branch exchange, and paired sums/differences agree within 1e-12

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d7487a85d37a3147 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d7487a85 (directory).
```
- **Stderr (tail)**:
```text
real	0m0.298s
user	0m0.243s
sys	0m0.033s
```

### Turn 12 — `submit_completed_task` (`15346 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema validation in 0.30 seconds without a BLAS override. On measured train and test histories, closed-form memory, branch exchange, and linear superposition agreed within 1e-12. The audit found no reason to change the solver.
- **Alternatives Weighed**: `Add lab-specific or nonlinear stress terms` (rejected: The measured histories remained below the linear-regime thresholds, and held-out residuals showed no consistent stress dependence across labs.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_s00_solver.py`. The public schema passed, and both synthetic tests recovered the relaxation times with 7% Cauchy glitches. The calibration archive was not mounted, so that fallback was not verified.`

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
