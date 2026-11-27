# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c03_batch_03__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c03_step_03` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `398,732` (`385,404` / `142,848` / `13,328`) |
| **Total Episode Wall Time (s)** | `493.81s` |

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

### Turn 1 — `execute_in_remote_docker` (`15382 ms` LLM)
- **Rationale**: Inspect the public schema and array structure before fitting: the available labels determine how to build held-out checks, and the archive inventory identifies the calibration fallback without pulling bulk data.

```bash
cd /holdout_task && cat validate_public_schema.py && ls -l /tmp/hegel_continual_archive && python3 - <<'PY'
import pickle,numpy as np
with open('public_alpha_m97_c03_batch_03.pkl','rb') as f:d=pickle.load(f)
for k,v in d.items():
 print(k,type(v).__name__,getattr(v,'shape',''),getattr(v,'dtype',''),str(v)[:230] if not isinstance(v,dict) else list(v))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation variable projection' 2>&1 | head -40
ls /tmp/hegel_continual_archive/*solver.py | xargs -r wc -l
```
- **Container Exit Code**: `0` (`469 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
  190 /tmp/hegel_continual_archive/alpha_m97_c03_batch_01_solver.py
  188 /tmp/hegel_continual_archive/alpha_m97_c03_batch_02_solver.py
  188 /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py
  566 total
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`20611 ms` LLM)
- **Rationale**: Inspect the dataset inside the pickle list to identify the measured training targets and define feasible holdouts. Read the prior alignment code and the documented fill primitive to resolve the reference-panel conventions before implementing the new regime.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
items=pickle.load(open('public_alpha_m97_c03_batch_03.pkl','rb'))
print('items',len(items),[list(i) for i in items[:4]])
for k,v in items[0]['dataset'].items():
 print(k,type(v).__name__,getattr(v,'shape',''),str(v)[:200] if not isinstance(v,dict) else '')
 if isinstance(v,dict):
  for kk,a in v.items():
   a=np.asarray(a); print('  ',kk,a.shape,a.dtype,repr(a.ravel()[:6]))
PY
sed -n '1,215p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_02_solver.py
head -70 /tmp/hegel_continual_archive/archive_calibration_history.json
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | head -65
```
- **Container Exit Code**: `0` (`422 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`17132 ms` LLM)
- **Rationale**: Check per-channel calibration residuals to determine whether the archive fallback is warranted, and confirm the prior solver’s fill convention before implementing the new response law.

```bash
sed -n '1,125p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_02_solver.py; head -38 /tmp/hegel_continual_archive/archive_calibration_history.json; python3 - <<'PY'
import pickle,numpy as np,scipy
p='/holdout_task/public_alpha_m97_c03_batch_03.pkl'
d=pickle.load(open(p,'rb'))[0]['dataset']; c=d['calibration_load_cells']; print('scipy',scipy.__version__)
for i in range(4):
 m=c['lab_id']==i; x=c['cmd_stress'][m]; y=c['true_stress'][m]; a,b=np.polyfit(x,y,1); print('calibration',i,len(x),a,b,np.max(np.abs(y-a*x-b)))
for s in ['train','test']:
 p=d[s]; print(s,[(k,int(np.sum(~np.isfinite(v)))) for k,v in p.items()]); print('nan feature rows',np.unique(p['lab_id'][~np.isfinite(p['coupon_features_raw']).all(1)],return_counts=True))
PY
```
- **Container Exit Code**: `0` (`1236 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
['lab_id']], bias[split['lab_id']]
    hist = g[:, None]*split['past_stress_cmd'] + b[:, None]
    hold = g*split['stress_hold_cmd'] + b
    # Each past segment's surviving contribution to the terminal memory.
    age = np.cumsum(dt[::-1])[::-1] - dt
    flow = np.column_stack((np.ones(len(x)), x))
    curve = np.empty((len(x), len(t), 7))
    rate = np.empty((len(x), 7))
    curve[:, :, 2:] = flow[:, None, :]*t[None, :, None]
    rate[:, 2:] = flow

    def evaluate(logtau, deriv=False):
        tau = np.exp(logtau)
        h, a = dt[:, None]/tau, age[:, None]/tau
        survival = np.exp(-a)
        kernel = survival*(-np.expm1(-h))
        gap = hold[:, None] - hist @ kernel
        u = t[:, None]/tau
        envelope = -np.expm1(-u)
        curve[:, :, :2] = gap[:, None, :]*envelope
        rate[:, :2] = gap/tau
        if deriv:
            dgap = -hist @ (a*kernel - survival*h*np.exp(-h))
            dc = dgap[:, None, :]*envelope - gap[:, None, :]*u*np.exp(-u)
            dr = (dgap - gap)/tau
            return curve, rate, dc, dr
        return curve, rate

    return evaluate


def _fit(evaluate, curve_y, rate_y, dt, t):
    cm, rm = np.isfinite(curve_y), np.isfinite(rate_y)
    y = np.r_[curve_y[cm], rate_y[rm]]
    n = cm.sum()
[
  {
    "episode_short_name": "alpha_m97_c03_s00",
    "lab_id": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      3,
      3,
      3,
      3,
scipy 1.13.0
calibration 0 10 1.289962862014423 0.11997817371921288 0.00017199813083895732
calibration 1 10 0.8199772001152258 -0.14996294385453945 0.0002451248283934504
calibration 2 10 1.169997993831413 0.14000869015530928 0.0001723904152397815
calibration 3 10 0.7600032873540995 -0.11003611614552662 0.00015032266651655857
train [('lab_id', 0), ('stratum', 0), ('coupon_features_raw', 22), ('ref_coupons', 239), ('past_stress_cmd', 0), ('stress_hold_cmd', 0), ('strain_0', 0), ('aux_chamber_telemetry', 0), ('strain_curves_obs', 0), ('instant_rate_obs', 0)]
nan feature rows (array([0, 1, 2, 3]), array([9, 6, 4, 1]))
test [('lab_id', 0), ('stratum', 0), ('coupon_features_raw', 8), ('ref_coupons', 101), ('past_stress_cmd', 0), ('stress_hold_cmd', 0), ('strain_0', 0), ('aux_chamber_telemetry', 0)]
nan feature rows (array([0, 1, 2, 3]), array([5, 1, 1, 1]))
```

### Turn 4 — `execute_in_remote_docker` (`61813 ms` LLM)
- **Rationale**: All channels have ten clean points and calibration residuals below 0.00025, so the public fit uses no archive data. The schema smoke test detects interface failures, and reading the archive keys checks whether the sparse-channel fallback needs a parser correction before held-out validation.
- **Alternatives Weighed**: `Reuse the prior linear-flow response law` (rejected: The specified jammed regime requires rational saturation and both slip terms.)

```bash
cat > /holdout_task/alpha_m97_c03_batch_03_solver.py <<'PY'
"""Calibrated two-branch rheology with saturating terminal flow and slip."""
import json

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _prefix_calibration():
    with open(_ARCHIVE) as handle:
        sweeps = json.load(handle)
    fits = [[] for _ in range(4)]
    for sweep in sweeps:
        lab_ids = np.asarray(sweep['lab_id'])
        cmds = np.asarray(sweep['cmd_stress'])
        measured = np.asarray(sweep['true_stress'])
        for lab in range(4):
            rows = (lab_ids == lab) & np.isfinite(cmds) & np.isfinite(measured)
            if rows.sum() >= 2:
                fits[lab].append(_line(cmds[rows], measured[rows]))
    return np.asarray([np.median(v, axis=0) for v in fits])


def _calibrate(cal):
    params = np.empty((4, 2))
    archive = None
    for lab in range(4):
        rows = (cal['lab_id'] == lab) & np.isfinite(cal['cmd_stress']) & np.isfinite(cal['true_stress'])
        x, y = cal['cmd_stress'][rows], cal['true_stress'][rows]
        clean = np.ones(len(x), dtype=bool)
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            dx = x[j] - x[i]
            use = dx != 0
            if use.any():
                slope = np.median((y[j] - y[i])[use] / dx[use])
                offset = np.median(y - slope*x)
                clean = np.abs(y - slope*x - offset) <= 3.0
        if clean.sum() < 2 or not clean.all():
            if archive is None:
                archive = _prefix_calibration()
        if clean.sum() < 2 or (not clean.all() and clean.sum() < 4):
            params[lab] = archive[lab]
        else:
            params[lab] = _line(x[clean], y[clean])
    return params


def _align(train, test):
    raw, ref, labs, strata = (np.concatenate((train[k], test[k])) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    x = np.empty(raw.shape, dtype=float)
    cells, ids = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    for lab in np.unique(labs):
        mask = labs == lab
        panel = ref[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[mask] = (raw[mask] - center) / scale
    means = np.full((len(cells), x.shape[1]), np.nan)
    for i in range(len(cells)):
        block = x[ids == i]
        n = np.isfinite(block).sum(axis=0)
        np.divide(np.nansum(block, axis=0), n, out=means[i], where=n > 0)
    for i, (lab, _) in enumerate(cells):
        for k in np.flatnonzero(~np.isfinite(means[i])):
            pool = means[cells[:, 0] == lab, k]
            if not np.isfinite(pool).any():
                pool = means[:, k]
            means[i, k] = np.nanmean(pool) if np.isfinite(pool).any() else 0.0
    x = np.where(np.isfinite(x), x, means[ids])
    return np.split(x, [len(train['lab_id'])])


def _design(split, x, params, dt, t, jammed):
    g, b = params[split['lab_id']].T
    hist = g[:, None]*split['past_stress_cmd'] + b[:, None]
    hold = g*split['stress_hold_cmd'] + b
    age = np.cumsum(dt[::-1])[::-1] - dt
    # Use the exact impulse-response sum of the zero-initialized recurrence.
    flow = np.column_stack((np.ones(len(x)), x))
    phi = t/(1.0 + 0.85*t) if jammed else t
    slip = np.sign(hold)*np.sqrt(np.abs(hold))
    curve = np.zeros((len(x), len(t), 9))
    rate = np.zeros((len(x), 9))
    curve[:, :, 2:7] = flow[:, None, :]*phi[None, :, None]
    rate[:, 2:7] = flow
    if jammed:
        curve[:, :, 7] = slip[:, None]*phi
        rate[:, 7] = slip

    def evaluate(tau):
        kernel = np.exp(-age[:, None]/tau)*(-np.expm1(-dt[:, None]/tau))
        gap = hold[:, None] - hist @ kernel
        exp_term = -np.expm1(-t[:, None]/tau)
        curve[:, :, :2] = gap[:, None, :]*exp_term
        rate[:, :2] = gap/tau
        if jammed:
            gap_delta = gap[:, 0] - gap[:, 1]
            curve[:, :, 8] = gap_delta[:, None]*exp_term[:, 0]
            rate[:, 8] = gap_delta/tau[0]
        return curve, rate

    return evaluate


def _fit(evaluate, curve_y, rate_y, dt, t, rate_weight=1.0, row_mask=None):
    if row_mask is None:
        row_mask = np.ones(len(rate_y), dtype=bool)
    cm = np.isfinite(curve_y) & row_mask[:, None]
    rm = np.isfinite(rate_y) & row_mask
    y = np.r_[curve_y[cm], rate_weight*rate_y[rm]]

    def project(log_tau, coefficients=False):
        tau = np.sort(np.exp(log_tau))
        curve, rate = evaluate(tau)
        design = np.vstack((curve[cm], rate_weight*rate[rm]))
        w = np.linalg.lstsq(design, y, rcond=None)[0]
        return (tau, w) if coefficients else design @ w - y

    t0 = max(np.min(t[t > 0]), np.min(dt))
    horizon = max(np.sum(dt), np.max(t))
    bounds = np.log([t0/20.0, horizon*30.0])
    starts = [(t0, horizon/3), (2*t0, horizon/2), (t0/3, horizon),
              (horizon/5, horizon*2), (t0/2, 2*t0)]
    best = min((least_squares(project, np.log(start), bounds=tuple(bounds),
                              max_nfev=90, ftol=1e-9, gtol=1e-9) for start in starts),
               key=lambda fit: fit.cost)
    return project(best.x, coefficients=True)


def solve_alpha_m97_c03_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    params = _calibrate(dataset['calibration_load_cells'])
    x_train, x_test = _align(train, test)
    jammed = dataset['shear_banding_index'] > 0.50
    design = _design(train, x_train, params, dt, t, jammed)
    tau, w = _fit(design, train['strain_curves_obs'] - train['strain_0'][:, None],
                  train['instant_rate_obs'], dt, t)
    curve, rate = _design(test, x_test, params, dt, t, jammed)(tau)
    return {'estimated_taus': tau, 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:],
            'predicted_strain': test['strain_0'][:, None] + curve @ w,
            'initial_rate_pred': rate @ w}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
python3 - <<'PY'
import json
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
a=json.load(open(p)); print('sweep schema:', [(s.get('episode_short_name'),list(s)) for s in a])
PY
```
- **Container Exit Code**: `0` (`583 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
sweep schema: [('alpha_m97_c03_s00', ['episode_short_name', 'lab_id', 'cmd_stress', 'true_stress']), ('alpha_m97_c03_batch_01', ['episode_short_name', 'lab_id', 'cmd_stress', 'true_stress']), ('alpha_m97_c03_batch_02', ['episode_short_name', 'lab_id', 'cmd_stress', 'true_stress'])]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f8d183c5bfc5c068 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f8d183c5 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`30539 ms` LLM)
- **Rationale**: Leave each lab’s response labels out to check whether the fit transfers across calibrated channels. Compare curve and rate errors separately to see whether their relative weighting needs adjustment, and test the specified sparse and EMI calibration paths to catch fallback failures the clean public data cannot reveal.

```bash
cat > /tmp/rheology03_cv.py <<'PY'
import importlib.util
import pickle
import numpy as np

p = '/holdout_task/alpha_m97_c03_batch_03_solver.py'
spec = importlib.util.spec_from_file_location('solver', p)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
d = pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_03.pkl','rb'))[0]['dataset']
train, test = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
params = s._calibrate(d['calibration_load_cells'])
x, _ = s._align(train, test)
model = s._design(train, x, params, dt, t, True)
y = train['strain_curves_obs'] - train['strain_0'][:,None]
r = train['instant_rate_obs']
labs = train['lab_id']
missing = ~np.isfinite(train['coupon_features_raw']).all(axis=1)
hold = params[labs, 0]*train['stress_hold_cmd'] + params[labs, 1]
slices = {**{f'lab{i}': labs == i for i in range(4)}, **{f'stratum{i}':train['stratum'] == i for i in np.unique(train['stratum'])}, 'missing':missing, 'complete':~missing, 'negative':hold<0, 'positive':hold>=0, 'high_abs_stress':np.abs(hold)>d['crit_stress']}
rmse = lambda z: np.sqrt(np.mean(z*z))
for weight in (0.25, 0.5, 1.0, 2.0):
    ec, er, taus = np.empty_like(y), np.empty_like(r), []
    for lab in range(4):
        val = labs == lab
        tau, w = s._fit(model, y, r, dt, t, rate_weight=weight, row_mask=~val)
        c, rr = model(tau)
        ec[val], er[val] = (c @ w-y)[val], (rr @ w-r)[val]
        taus.append(tau)
    print('weight',weight,'taus',np.round(taus,5).tolist(), 'pooled', tuple(np.round([rmse(ec),rmse(er)],6)))
    print('slices', {k:tuple(np.round([rmse(ec[v]),rmse(er[v])],6)) for k,v in slices.items() if v.any()})
    print('per-horizon',np.round(np.sqrt(np.mean(ec**2,axis=0)),6).tolist())
cal = d['calibration_load_cells']
for lab in range(4):
    chosen = np.flatnonzero(cal['lab_id']==lab)
    for j in chosen:
        spike = {k:v.copy() for k,v in cal.items()}
        spike['true_stress'][j] += 3.2
        assert np.max(np.abs(s._calibrate(spike)-params)) < 0.00025, (lab,j)
    keep = (cal['lab_id']!=lab)
    keep[chosen[0]] = True
    sparse = {k:v[keep] for k,v in cal.items()}
    print('sparse',lab,'prefix delta',np.round(s._calibrate(sparse)[lab]-params[lab],7))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology03_cv.py
```
- **Container Exit Code**: `0` (`1530 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
atum2': (0.719247, 0.161274), 'missing': (0.528779, 0.25955), 'complete': (1.713279, 0.160726), 'negative': (0.690184, 0.16443), 'positive': (2.286124, 0.185805)}
per-horizon [0.167695, 0.367422, 0.545805, 0.415053, 4.822401, 0.383995, 0.462026, 0.19076, 0.909796, 1.135159]
weight 0.5 taus [[0.79376, 15.32878], [0.85828, 23.58478], [0.58393, 4.5177], [0.81091, 8.62805]] pooled (1.62455, 0.16214)
slices {'lab0': (0.796418, 0.161504), 'lab1': (0.629187, 0.185327), 'lab2': (3.044471, 0.131369), 'lab3': (0.4604, 0.165104), 'stratum0': (2.757764, 0.162238), 'stratum1': (0.444927, 0.173903), 'stratum2': (0.718664, 0.145617), 'missing': (0.527951, 0.252941), 'complete': (1.712955, 0.146895), 'negative': (0.689774, 0.148289), 'positive': (2.28575, 0.177278)}
per-horizon [0.167898, 0.367132, 0.545128, 0.413309, 4.821985, 0.382137, 0.461526, 0.18956, 0.90957, 1.134383]
weight 1.0 taus [[0.76761, 12.7817], [0.82096, 17.84914], [0.57016, 4.4831], [0.76603, 7.64706]] pooled (1.623773, 0.139795)
slices {'lab0': (0.795319, 0.147139), 'lab1': (0.625904, 0.163603), 'lab2': (3.044112, 0.102401), 'lab3': (0.458268, 0.136527), 'stratum0': (2.757299, 0.140203), 'stratum1': (0.44198, 0.1535), 'stratum2': (0.717017, 0.119521), 'missing': (0.525722, 0.241931), 'complete': (1.712212, 0.121116), 'negative': (0.688616, 0.122005), 'positive': (2.284956, 0.158456)}
per-horizon [0.167728, 0.365341, 0.54319, 0.408643, 4.821373, 0.377519, 0.460265, 0.187059, 0.908623, 1.132316]
weight 2.0 taus [[0.72767, 10.17394], [0.76648, 12.72146], [0.53977, 4.3537], [0.70788, 6.97131]] pooled (1.622365, 0.106291)
slices {'lab0': (0.792025, 0.124267), 'lab1': (0.619078, 0.127104), 'lab2': (3.043891, 0.068069), 'lab3': (0.455276, 0.089029), 'stratum0': (2.756483, 0.108539), 'stratum1': (0.43659, 0.122569), 'stratum2': (0.713915, 0.077638), 'missing': (0.520993, 0.221835), 'complete': (1.71089, 0.080985), 'negative': (0.686079, 0.08958), 'positive': (2.283672, 0.123327)}
per-horizon [0.166651, 0.360982, 0.540133, 0.401534, 4.820526, 0.369968, 0.457842, 0.183116, 0.905547, 1.127874]
sparse 0 prefix delta [5.0e-07 5.1e-06]
sparse 1 prefix delta [ 6.18e-05 -4.27e-05]
sparse 2 prefix delta [-1.71e-05 -3.15e-05]
sparse 3 prefix delta [-1.80e-06  4.81e-05]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f8d183c5bfc5c068 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f8d183c5 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`32410 ms` LLM)
- **Rationale**: The worst curve error is concentrated at one horizon, while relaxation times vary sharply across folds. Fit separate robust scales using only each fold’s training labels, then check whether the same jammed-slip equations yield more stable parameters and lower held-out bulk errors without hiding raw RMSE.
- **Alternatives Weighed**: `Increase the rate weight in ordinary least squares` (rejected: It improves rate RMSE but barely changes the curve bottleneck; it does not reduce the influence of large curve residuals.)

```bash
cat > /tmp/rheology03_robust.py <<'PY'
from pathlib import Path
exec(Path('/tmp/rheology03_cv.py').read_text().split('for weight in')[0])
from scipy.optimize import least_squares

def mad(a):
    return 1.4826*np.median(np.abs(a-np.median(a)))

def fit_robust(keep, k):
    tau, w = s._fit(model, y, r, dt, t, row_mask=keep)
    def errors(p):
        c, rr = model(np.sort(np.exp(p[:2])))
        return (c @ p[2:] - y)[keep], (rr @ p[2:] - r)[keep]
    p0 = np.r_[np.log(tau), w]
    for step in range(3):
        e, f = errors(p0)
        sc, sr = max(mad(e), 1e-6), max(mad(f), 1e-6)
        def residual(p):
            a,b=errors(p)
            return np.r_[a.ravel()/sc, b/sr]
        lo = [np.log(min(t)/20)]*2+[-np.inf]*9
        hi = [np.log(max(sum(dt),max(t))*30)]*2+[np.inf]*9
        out=least_squares(residual, p0, bounds=(lo,hi), loss='soft_l1', f_scale=k, x_scale='jac', max_nfev=160)
        p0=out.x
    return np.sort(np.exp(p0[:2])),p0[2:],(sc,sr)

def metrics(ec,er):
    for a in (ec,er):
        print('absolute error q50/90/95/98/99/100:',np.round(np.quantile(np.abs(a),[.5,.9,.95,.98,.99,1.]),5))
    for name, mask in slices.items():
        if mask.any():
            print(name,'RMSE',np.round([rmse(ec[mask]),rmse(er[mask])],6),'MAE',np.round([np.mean(np.abs(ec[mask])),np.mean(np.abs(er[mask]))],6))
    print('horizon MAE',np.round(np.mean(np.abs(ec),axis=0),6))

for k in (None, 1.5, 3.0):
    print('robust cutoff:',k)
    ec, er = np.empty_like(y), np.empty_like(r)
    for lab in range(4):
        val = labs == lab
        if k is None:
            tau,w=s._fit(model,y,r,dt,t,row_mask=~val)
            scales=()
        else:
            tau,w,scales=fit_robust(~val,k)
        c,rr=model(tau)
        ec[val],er[val]=(c@w-y)[val],(rr@w-r)[val]
        print('fold',lab,'tau',np.round(tau,5),'weights',np.round(w,5),'scales',scales)
    metrics(ec,er)
    print('largest curve errors (row, horizon), signed residual:',[(tuple(i),round(ec[tuple(i)],4)) for i in np.argwhere(np.abs(ec)>3)])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology03_robust.py
```
- **Container Exit Code**: `0` (`1720 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(row, horizon), signed residual: [((28, 4), 3.646), ((34, 1), 4.1042), ((59, 9), 4.2345), ((67, 4), 3.1947), ((82, 5), -4.4479), ((105, 2), -3.2091), ((127, 6), -4.8404), ((133, 2), 4.6295), ((136, 9), -14.138), ((158, 3), -4.9987), ((168, 8), 11.4264), ((176, 4), -64.3658), ((177, 2), 3.0952)]
robust cutoff: 3.0
fold 0 tau [0.53102 4.12528] weights [ 0.68205  0.56747  0.13746  0.1567  -0.12567  0.1191  -0.08551  0.29773
 -0.22108] scales (0.01850644804780312, 0.021091029886572552)
fold 1 tau [0.53444 4.167  ] weights [ 0.68664  0.56323  0.13842  0.15595 -0.12547  0.11877 -0.08774  0.29924
 -0.22372] scales (0.019235135932761487, 0.021136316860080884)
fold 2 tau [0.53451 4.18041] weights [ 0.68762  0.5647   0.13746  0.15792 -0.12795  0.11924 -0.08912  0.29737
 -0.22282] scales (0.01887779973801417, 0.01686124250968038)
fold 3 tau [0.53348 4.21686] weights [ 0.68982  0.57318  0.13588  0.15686 -0.12588  0.12042 -0.08607  0.28773
 -0.22977] scales (0.020029009172002527, 0.02178948072207807)
absolute error q50/90/95/98/99/100: [1.32000e-02 8.11500e-02 2.27180e-01 7.67390e-01 1.73864e+00 6.43672e+01]
absolute error q50/90/95/98/99/100: [0.01382 0.04625 0.1227  0.2227  0.26786 0.42162]
lab0 RMSE [0.774635 0.083248] MAE [0.125871 0.036264]
lab1 RMSE [0.597064 0.054565] MAE [0.087212 0.027339]
lab2 RMSE [3.042498 0.054593] MAE [0.19238  0.027945]
lab3 RMSE [0.457801 0.031752] MAE [0.082863 0.019853]
stratum0 RMSE [2.751844 0.058694] MAE [0.19472  0.030111]
stratum1 RMSE [0.42759  0.078956] MAE [0.092514 0.035009]
stratum2 RMSE [0.702257 0.022128] MAE [0.086535 0.01781 ]
missing RMSE [0.505252 0.172947] MAE [0.180707 0.135933]
complete RMSE [1.706205 0.018498] MAE [0.115883 0.014844]
negative RMSE [0.673276 0.040121] MAE [0.086931 0.021676]
positive RMSE [2.280521 0.077712] MAE [0.166295 0.036213]
horizon MAE [0.035316 0.065949 0.127345 0.069948 0.458253 0.064358 0.105471 0.049609
 0.118951 0.135656]
largest curve errors (row, horizon), signed residual: [((28, 4), 3.6441), ((34, 1), 4.1034), ((59, 9), 4.2332), ((67, 4), 3.1919), ((82, 5), -4.4476), ((105, 2), -3.2102), ((127, 6), -4.8409), ((133, 2), 4.6292), ((136, 9), -14.1419), ((158, 3), -5.0032), ((168, 8), 11.4246), ((176, 4), -64.3672), ((177, 2), 3.0956)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f8d183c5bfc5c068 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f8d183c5 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`28967 ms` LLM)
- **Rationale**: Check whether stronger outlier downweighting improves cross-lab parameter stability and typical held-out errors without worsening the raw worst slices. If it does not, retain the simpler soft-L1 fit.
- **Alternatives Weighed**: `Change the constitutive law to fit the largest curve residuals` (rejected: The same law yields stable relaxation times under robust fitting, and the largest errors are isolated observations rather than coherent curve departures.)

```bash
export PYTHONPATH=/tmp OPENBLAS_NUM_THREADS=1
python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS' | head -20
cat > /tmp/rheology03_loss_check.py <<'PY'
from pathlib import Path
exec(Path('/tmp/rheology03_robust.py').read_text().split('for k in (None,')[0])

def cauchy_refine(keep, tau, w, scales):
    p0 = np.r_[np.log(tau), w]
    sc, sr = scales
    def residual(p):
        c, rr = model(np.sort(np.exp(p[:2])))
        return np.r_[((c @ p[2:] - y)[keep]/sc).ravel(), (rr @ p[2:] - r)[keep]/sr]
    lo = [np.log(min(t)/20)]*2 + [-np.inf]*9
    hi = [np.log(max(sum(dt),max(t))*30)]*2 + [np.inf]*9
    out = least_squares(residual, p0, bounds=(lo,hi), loss='cauchy', f_scale=2.5, x_scale='jac', max_nfev=100)
    return np.sort(np.exp(out.x[:2])), out.x[2:]

results = {name:(np.empty_like(y),np.empty_like(r),[]) for name in ('soft_l1','cauchy')}
for lab in range(4):
    val = labs == lab
    tau, w, scales = fit_robust(~val, 1.5)
    for name, (a,b) in zip(results, [(tau,w),cauchy_refine(~val,tau,w,scales)]):
        ec, er, folds = results[name]
        c, rr = model(a)
        ec[val], er[val] = (c @ b-y)[val], (rr @ b-r)[val]
        folds.append(np.r_[a,b])
for name,(ec,er,folds) in results.items():
    print(name,'fold estimates',np.round(folds,5).tolist())
    for label, values in [('curve',ec),('rate',er)]:
        print(label,'worst slice RMSE',max((rmse(values[m]),k) for k,m in slices.items() if m.any()),'worst slice MedAE',max((np.median(np.abs(values[m])),k) for k,m in slices.items() if m.any()),'complete q50/q90',np.quantile(np.abs(values[~missing]),[.5,.9]))
    print('worst horizon MedAE', np.max(np.median(np.abs(ec),axis=0)))
PY
python3 -I /tmp/rheology03_loss_check.py
```
- **Container Exit Code**: `0` (`945 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ncreasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_local_level_step(predictor: np.ndarray, values: np.ndarray, splits: np.ndarray, *, scale: float, loss_clip: float) -> tuple[int, float] | Fit finite predictor and values (n,) and integer splits (k,) with k >= 1, at l | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_local_level_step | STATUS: APPLICABLE_SUBROUTINE(real_1d_step_series)
soft_l1 fold estimates [[0.53147, 4.0738, 0.68041, 0.56017, 0.13802, 0.1572, -0.12666, 0.11897, -0.08651, 0.30299, -0.21971], [0.53344, 4.09022, 0.6825, 0.557, 0.13892, 0.15604, -0.12645, 0.11862, -0.08807, 0.30489, -0.22095], [0.53377, 4.09368, 0.68354, 0.55702, 0.13796, 0.15771, -0.12791, 0.11884, -0.0892, 0.30337, -0.22089], [0.53328, 4.13619, 0.68601, 0.5638, 0.13704, 0.15676, -0.12681, 0.1197, -0.08688, 0.29591, -0.2263]]
curve worst slice RMSE (3.0424289919195524, 'lab2') worst slice MedAE (0.08275343819656952, 'missing') complete q50/q90 [0.01136968 0.03038149]
rate worst slice RMSE (0.17388336131501597, 'missing') worst slice MedAE (0.12172507945154643, 'missing') complete q50/q90 [0.01100256 0.02916438]
worst horizon MedAE 0.014711426382181425
cauchy fold estimates [[0.53148, 4.02338, 0.67826, 0.55455, 0.13844, 0.15784, -0.12768, 0.11872, -0.08734, 0.30747, -0.21817], [0.53222, 4.01875, 0.67816, 0.55233, 0.1394, 0.15624, -0.12751, 0.11824, -0.08831, 0.30995, -0.21828], [0.53261, 3.99971, 0.67858, 0.5503, 0.13829, 0.1575, -0.12812, 0.11834, -0.08932, 0.3094, -0.21882], [0.53293, 4.06282, 0.68223, 0.55605, 0.13793, 0.15667, -0.12781, 0.11902, -0.08765, 0.30321, -0.22298]]
curve worst slice RMSE (3.042381092784154, 'lab2') worst slice MedAE (0.0836414391506044, 'missing') complete q50/q90 [0.01111209 0.02912769]
rate worst slice RMSE (0.17474649854074478, 'missing') worst slice MedAE (0.1225105299124073, 'missing') complete q50/q90 [0.01075323 0.02874478]
worst horizon MedAE 0.013960690460202364

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f8d183c5bfc5c068 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f8d183c5 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`47610 ms` LLM)
- **Rationale**: Promote the measured robust improvement without changing the specified constitutive law or imputation. Verify that the actual artifact reproduces the held-out performance and that its curve and initial-rate outputs agree with an independent implementation of the memory recurrence.
- **Alternatives Weighed**: `Stop after soft-L1 fitting` (rejected: Cauchy refinement improves complete-coupon errors and the worst-horizon median curve error, with a rate RMSE increase below 0.001 in the missing-feature slice.)

```bash
set -e
export PYTHONPATH=/tmp OPENBLAS_NUM_THREADS=1
python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_affine_projection.robust_residual_weights > /tmp/rheology03_robust_skill.txt
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c03_batch_03_solver.py')
s = p.read_text()
marker = '\ndef solve_alpha_m97_c03_batch_03(dataset: dict) -> dict:\n'
assert s.count(marker) == 1
new = '''
def _fit_robust(evaluate, curve_y, rate_y, dt, t, row_mask=None):
    if row_mask is None:
        row_mask = np.ones(len(rate_y), dtype=bool)
    tau, w = _fit(evaluate, curve_y, rate_y, dt, t, row_mask=row_mask)
    cm = np.isfinite(curve_y) & row_mask[:, None]
    rm = np.isfinite(rate_y) & row_mask
    p = np.r_[np.log(tau), w]
    lo = np.r_[[np.log(max(min(t[t > 0]), min(dt))/20)]*2, [-np.inf]*len(w)]
    hi = np.r_[[np.log(max(sum(dt), max(t))*30)]*2, [np.inf]*len(w)]

    def errors(p):
        curve, rate = evaluate(np.sort(np.exp(p[:2])))
        return (curve @ p[2:] - curve_y)[cm], (rate @ p[2:] - rate_y)[rm]

    def scale(a):
        return max(1.4826*np.median(np.abs(a - np.median(a))), 1e-6)

    # Stabilize the fit before applying a loss that strongly downweights spikes.
    for loss, cutoff in [('soft_l1', 1.5)]*3 + [('cauchy', 2.5)]:
        if loss == 'soft_l1':
            sc, sr = map(scale, errors(p))

        def residual(p):
            ec, er = errors(p)
            return np.r_[ec/sc, er/sr]

        p = least_squares(residual, p, bounds=(lo, hi), loss=loss,
                          f_scale=cutoff, x_scale='jac', max_nfev=160).x
    return np.sort(np.exp(p[:2])), p[2:]

'''
s = s.replace(marker, '\n'+new+marker)
s = s.replace("tau, w = _fit(design, train['strain_curves_obs'] - train['strain_0'][:, None],\n                  train['instant_rate_obs'], dt, t)", "tau, w = _fit_robust(design, train['strain_curves_obs'] - train['strain_0'][:, None],\n                         train['instant_rate_obs'], dt, t)")
p.write_text(s)
PY
python3 -I /holdout_task/validate_public_schema.py
cat > /tmp/rheology03_verify_final.py <<'PY'
from pathlib import Path
exec(Path('/tmp/rheology03_cv.py').read_text().split('for weight in')[0])

ec, er = np.empty_like(y), np.empty_like(r)
fold_taus = []
for lab in range(4):
    val = labs == lab
    tau, w = s._fit_robust(model, y, r, dt, t, row_mask=~val)
    c, rr = model(tau)
    ec[val], er[val] = (c @ w-y)[val], (rr @ w-r)[val]
    fold_taus.append(tau)
print('Fold taus:', np.round(fold_taus, 5).tolist())
for name, err in [('curve',ec),('rate',er)]:
    print(name, 'worst slice RMSE:', max((rmse(err[m]), k) for k,m in slices.items() if m.any()), 'worst slice MedAE:', max((np.median(np.abs(err[m])), k) for k,m in slices.items() if m.any()))
print('Worst horizon MedAE:', np.max(np.median(np.abs(ec), axis=0)))
assert max(rmse(ec[m]) for m in slices.values() if m.any()) < 3.045
assert max(rmse(er[m]) for m in slices.values() if m.any()) < 0.176
assert np.max(np.median(np.abs(ec), axis=0)) < 0.016
out = s.solve_alpha_m97_c03_batch_03(d)
tau, w, rw = (out[k] for k in ('estimated_taus','hereditary_weights','regime_mode_weights'))
assert np.all((tau > 0) & (tau >= np.min(fold_taus, axis=0)*0.95) & (tau <= np.max(fold_taus, axis=0)*1.05))
_, x_test = s._align(train, test)
g, b = params[test['lab_id']].T
hold = g*test['stress_hold_cmd'] + b
z = np.zeros((len(hold),2))
for k, step in enumerate(dt):
    decay = np.exp(-step/tau)
    stress = g*test['past_stress_cmd'][:,k]+b
    z = z*decay + stress[:,None]*(1-decay)
a, c = (hold[:,None]-z).T
flow = w[2]+x_test@w[3:]+rw[0]*np.sign(hold)*np.sqrt(np.abs(hold))
prony = 1-np.exp(-t[:,None]/tau)
f1 = w[0]*a+rw[1]*(a-c)
expected = test['strain_0'][:,None] + f1[:,None]*prony[:,0] + (w[1]*c)[:,None]*prony[:,1] + flow[:,None]*t/(1+0.85*t)
np.testing.assert_allclose(out['predicted_strain'], expected, rtol=0, atol=1e-12)
np.testing.assert_allclose(out['initial_rate_pred'], f1/tau[0]+w[1]*c/tau[1]+flow, rtol=0, atol=1e-12)
print('Final taus:', tau, '; independent curve and initial-rate checks passed')
PY
python3 -I /tmp/rheology03_verify_final.py
```
- **Container Exit Code**: `0` (`1366 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Fold taus: [[0.53148, 4.02338], [0.53222, 4.01875], [0.53261, 3.99971], [0.53293, 4.06282]]
curve worst slice RMSE: (3.0423810927833927, 'lab2') worst slice MedAE: (0.0836414391506044, 'missing')
rate worst slice RMSE: (0.17474649855336177, 'missing') worst slice MedAE: (0.1225105299124073, 'missing')
Worst horizon MedAE: 0.013960690445901803
Final taus: [0.53232298 4.02431995] ; independent curve and initial-rate checks passed

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d069b1a02358088e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d069b1a0 (directory).
```

### Turn 9 — `submit_completed_task` (`14272 ms` LLM)
- **Rationale**: The saved artifact passed clean-process schema validation and matched the independent recurrence for both outputs. Sparse and EMI calibration checks passed. The raw worst-lab strain RMSE was 3.042, dominated by an isolated measurement spike.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_03_solver.py': 'FILE:7438'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_03_solver.py`. The public schema and independent checks of the strain curves and initial rates passed.`

### Turn 10 — `execute_in_remote_docker` (`29065 ms` LLM)
- **Rationale**: Check that the curve starts at the measured strain and differentiates to the reported initial rate; a mismatch would require correcting the response basis. Compare held-out errors with the slip and gap-difference terms removed, and inspect signed residuals by lab: a persistent offset on complete coupons would warrant revisiting alignment before submission.
- **Alternatives Weighed**: `Apply binary balanced-accuracy, relay-mode, and HJB checks` (rejected: This task has continuous strain and rate targets and no relay state or HJB operator; the relevant nuisance check is signed error by held-out lab.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
cat > /tmp/rheology03_gate.py <<'PY'
from pathlib import Path
exec(Path('/tmp/rheology03_cv.py').read_text().split('for weight in')[0])

# Check the rational terminal flow at t=0, where its slope contributes to the reported rate.
eps = 1e-5
tau, _ = s._fit_robust(model, y, r, dt, t)
c, rate = s._design(train, x, params, dt, np.array([0., eps, 2*eps]), True)(tau)
np.testing.assert_allclose(c[:, 0], 0., atol=1e-14)
np.testing.assert_allclose((4*c[:, 1]-c[:, 2])/(2*eps), rate, rtol=1e-7, atol=1e-8)

# Fit each ablation on fold-training labels only; compare response transfer, not pooled classification accuracy.
for name, removed in [('full', ()), ('no_gap_difference', (8,)), ('no_slip', (7,)), ('linear_flow', ())]:
    base = s._design(train, x, params, dt, t, name != 'linear_flow')
    def design(taus):
        a, b = base(taus)
        if removed:
            a, b = a.copy(), b.copy()
            a[..., removed] = 0
            b[..., removed] = 0
        return a, b
    ec, er = np.empty_like(y), np.empty_like(r)
    for lab in range(4):
        val = labs == lab
        ta, w = s._fit_robust(design, y, r, dt, t, row_mask=~val)
        c, rate = design(ta)
        ec[val], er[val] = (c @ w - y)[val], (rate @ w - r)[val]
    print(name, 'complete-coupon worst-lab MedAE: curve, rate', [round(max(np.median(np.abs(a[(labs==lab)&~missing])) for lab in range(4)), 6) for a in (ec, er)])
    if name == 'full':
        for lab in range(4):
            mask = (labs == lab) & ~missing
            print('lab', lab, 'complete-coupon signed median: curve, rate', np.round([np.median(ec[mask]), np.median(er[mask])], 6).tolist())
        print('full missing-coupon counts by lab', [int(np.sum((labs==lab)&missing)) for lab in range(4)])
PY
python3 -I /tmp/rheology03_gate.py
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`2891 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
full complete-coupon worst-lab MedAE: curve, rate [0.011321, 0.014815]
lab 0 complete-coupon signed median: curve, rate [-0.000871, -0.000459]
lab 1 complete-coupon signed median: curve, rate [0.000946, -0.001311]
lab 2 complete-coupon signed median: curve, rate [-0.002539, -0.002889]
lab 3 complete-coupon signed median: curve, rate [-0.000374, -0.002607]
full missing-coupon counts by lab [9, 6, 4, 1]
no_gap_difference complete-coupon worst-lab MedAE: curve, rate [0.041626, 0.20305]
no_slip complete-coupon worst-lab MedAE: curve, rate [0.045465, 0.067238]
linear_flow complete-coupon worst-lab MedAE: curve, rate [0.110004, 0.380781]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d069b1a02358088e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d069b1a0 (directory).
```

### Turn 11 — `submit_completed_task` (`17184 ms` LLM)
- **Rationale**: The final artifact passed clean-process verification. The focused audit confirmed agreement between the initial rate and the curve derivative and found complete-coupon signed median errors below 0.003 in every held-out lab, so the measurements do not support adding a lab correction.
- **Alternatives Weighed**: `Simplify the constitutive basis` (rejected: Removing slip or the gap-difference term, or replacing saturating flow with linear flow, worsened held-out strain and rate errors.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_03_solver.py': 'FILE:7438'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_03_solver.py`. The public schema, independent strain and initial-rate checks, and sparse and EMI calibration tests passed. An isolated measurement spike dominated the worst-lab strain RMSE of 3.042.`

### Turn 12 — `execute_in_remote_docker` (`34359 ms` LLM)
- **Rationale**: The audit already confirmed the constitutive terms and found no persistent lab offset. The calibration tests covered sparsity and EMI separately, but not together. Thin the real sweeps and inject a specified-size spike to see whether a corrupted affine fit materially shifts either output; if it does, revise only the affected calibration path.

```bash
set -e
export OPENBLAS_NUM_THREADS=1
cat > /tmp/rheology03_calibration_margin.py <<'PY'
from pathlib import Path
from itertools import combinations
exec(Path('/tmp/rheology03_cv.py').read_text().split('for weight in')[0])
cal = d['calibration_load_cells']
ref = s._prefix_calibration()
base = s.solve_alpha_m97_c03_batch_03(d)
for n_clean in (1, 2, 3, 4, 9):
    worst = (0., None)
    for lab in range(4):
        lab_rows = np.flatnonzero(cal['lab_id'] == lab)
        other = cal['lab_id'] != lab
        for rows in combinations(lab_rows, n_clean + 1):
            keep = other.copy()
            keep[list(rows)] = True
            selected = {k: v[keep] for k, v in cal.items()}
            for j in np.flatnonzero(selected['lab_id'] == lab):
                for shift in (-3.2, 3.2):
                    case = {k: v.copy() for k, v in selected.items()}
                    case['true_stress'][j] += shift
                    pred = s._calibrate(case)[lab]
                    cmd = np.r_[train['past_stress_cmd'][labs == lab].ravel(), train['stress_hold_cmd'][labs == lab]]
                    error = np.max(np.abs((pred[0]-params[lab,0])*cmd + pred[1]-params[lab,1]))
                    if error > worst[0]:
                        worst = (float(error), (lab, case, pred))
    err, (lab, case, pred) = worst
    out = s.solve_alpha_m97_c03_batch_03({**d, 'calibration_load_cells': case})
    print('clean', n_clean, 'worst lab', lab, 'stress error', round(err, 6), 'fit', pred.tolist(), 'prefix', ref[lab].tolist(), 'strain/rate max change', [round(float(np.max(np.abs(out[k]-base[k]))),6) for k in ('predicted_strain','initial_rate_pred')], 'cmd/true', [(float(a),float(b)) for a,b in zip(case['cmd_stress'][case['lab_id']==lab],case['true_stress'][case['lab_id']==lab])])
PY
python3 -I /tmp/rheology03_calibration_margin.py
```
- **Container Exit Code**: `0` (`16475 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
clean 1 worst lab 0 stress error 423.402906 fit [-219.95683272128034, -89.3546247348766] prefix [1.2899633414625333, 0.11998328836131866] strain/rate max change [2.885605, 3.736442] cmd/true [(-0.389947, -3.583117685711525), (-0.404411, -0.4016620572309279)]
clean 2 worst lab 1 stress error 131.356783 fit [-35.512093159978335, 46.269528810495906] prefix [0.8200389791210289, -0.1500056296012814] strain/rate max change [2.386986, 3.140516] cmd/true [(1.291089, 0.9086845517628155), (1.197564, 4.032051325930518), (1.256207, 0.8800333627179)]
clean 3 worst lab 1 stress error 62.122022 fit [-16.453216027579163, 21.59070392324732] prefix [0.8200389791210289, -0.1500056296012814] strain/rate max change [2.537672, 3.376184] cmd/true [(1.291089, 0.9086845517628155), (1.104415, 3.955877300439111), (1.197564, 0.8320513259305183), (1.256207, 0.8800333627179)]
clean 4 worst lab 1 stress error 0.00558 fit [0.8184198455378591, -0.14802408890506433] prefix [0.8200389791210289, -0.1500056296012814] strain/rate max change [0.003777, 0.005076] cmd/true [(1.291089, 0.9086845517628155), (0.982953, -2.5440082859854467), (1.104415, 0.755877300439111), (1.197564, 0.8320513259305183), (1.256207, 0.8800333627179)]
clean 9 worst lab 1 stress error 0.000575 fit [0.8198193247789866, -0.149757335129502] prefix [0.8200389791210289, -0.1500056296012814] strain/rate max change [0.000402, 0.00053] cmd/true [(1.291089, 0.9086845517628155), (0.982953, 0.6559917140145533), (1.507609, 1.0861707071239628), (1.104415, 0.755877300439111), (1.197564, 0.8320513259305183), (1.834972, 1.354561142136367), (1.565365, 1.1335993328451062), (1.256207, 0.8800333627179), (-1.07194, -4.229016859751257), (0.426374, 0.19976638833628743)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d069b1a02358088e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d069b1a0 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`45734 ms` LLM)
- **Rationale**: A spike in a two-point sweep can fit perfectly and still produce a grossly wrong gain. Check whether nominal-response deviations separate the clean sweep from the injected spikes before using them to trigger archive confirmation. Rerun the calibration stress test and verify that the clean-data fit and both prediction outputs still pass.
- **Alternatives Weighed**: `Use prefix calibration for every small sweep` (rejected: Clean two- and three-point sweeps should still fit directly, and four clean points take precedence over the archive.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
python3 - <<'PY'
from pathlib import Path
import pickle
import numpy as np
p = Path('/holdout_task/alpha_m97_c03_batch_03_solver.py')
with p.with_name('public_alpha_m97_c03_batch_03.pkl').open('rb') as f:
    c = pickle.load(f)[0]['dataset']['calibration_load_cells']
delta = c['true_stress'] - c['cmd_stress']
print('Nominal-response separation: clean max, injected-spike min:', np.abs(delta).max(), min(np.abs(delta+3.2).min(), np.abs(delta-3.2).min()))
assert np.abs(delta).max() < 1.5 < min(np.abs(delta+3.2).min(), np.abs(delta-3.2).min())
s = p.read_text()
mark = '        if clean.sum() < 2 or not clean.all():\n'
assert s.count(mark) == 1
s = s.replace(mark, '''        # With at most four rows, a spike can tilt even the median-slope
        # fit. Screen for a gross departure from the nominal unit-gain
        # response, then use the prefix line to apply the actual EMI cutoff.
        if 2 <= len(x) <= 4 and np.any(np.abs(y - x) > 1.5):
            if archive is None:
                archive = _prefix_calibration()
            gain, bias = archive[lab]
            clean &= np.abs(y - (gain*x + bias)) <= 3.0
''' + mark)
p.write_text(s)
PY
python3 -I /tmp/rheology03_calibration_margin.py
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/rheology03_verify_final.py
```
- **Container Exit Code**: `0` (`26203 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_03_solver.py']`
- **Stdout (tail)**:
```text
, injected-spike min: 0.7206141867941951 2.479385813205805
clean 1 worst lab 1 stress error 0.000187 fit [0.8200389791210289, -0.1500056296012814] prefix [0.8200389791210289, -0.1500056296012814] strain/rate max change [0.000148, 0.000202] cmd/true [(1.291089, -2.2913154482371847), (0.982953, 0.6559917140145533)]
clean 2 worst lab 1 stress error 0.000187 fit [0.8200389791210289, -0.1500056296012814] prefix [0.8200389791210289, -0.1500056296012814] strain/rate max change [0.000148, 0.000202] cmd/true [(1.291089, -2.2913154482371847), (0.982953, 0.6559917140145533), (1.507609, 1.0861707071239628)]
clean 3 worst lab 1 stress error 0.000187 fit [0.8200389791210289, -0.1500056296012814] prefix [0.8200389791210289, -0.1500056296012814] strain/rate max change [0.000148, 0.000202] cmd/true [(1.291089, -2.2913154482371847), (0.982953, 0.6559917140145533), (1.507609, 1.0861707071239628), (1.104415, 0.755877300439111)]
clean 4 worst lab 1 stress error 0.00558 fit [0.8184198455378591, -0.14802408890506433] prefix [0.8200389791210289, -0.1500056296012814] strain/rate max change [0.003777, 0.005076] cmd/true [(1.291089, 0.9086845517628155), (0.982953, -2.5440082859854467), (1.104415, 0.755877300439111), (1.197564, 0.8320513259305183), (1.256207, 0.8800333627179)]
clean 9 worst lab 1 stress error 0.000575 fit [0.8198193247789866, -0.149757335129502] prefix [0.8200389791210289, -0.1500056296012814] strain/rate max change [0.000402, 0.00053] cmd/true [(1.291089, 0.9086845517628155), (0.982953, 0.6559917140145533), (1.507609, 1.0861707071239628), (1.104415, 0.755877300439111), (1.197564, 0.8320513259305183), (1.834972, 1.354561142136367), (1.565365, 1.1335993328451062), (1.256207, 0.8800333627179), (-1.07194, -4.229016859751257), (0.426374, 0.19976638833628743)]
{"schema_valid": true, "errors": []}
Fold taus: [[0.53148, 4.02338], [0.53222, 4.01875], [0.53261, 3.99971], [0.53293, 4.06282]]
curve worst slice RMSE: (3.0423810927833927, 'lab2') worst slice MedAE: (0.0836414391506044, 'missing')
rate worst slice RMSE: (0.17474649855336177, 'missing') worst slice MedAE: (0.1225105299124073, 'missing')
Worst horizon MedAE: 0.013960690445901803
Final taus: [0.53232298 4.02431995] ; independent curve and initial-rate checks passed

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c7b438b9fd309957 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_c7b438b9 (directory).
```

### Turn 14 — `submit_completed_task` (`14807 ms` LLM)
- **Rationale**: The final artifact passed clean-process verification. The EMI fix substantially improved sparse calibration without changing clean-data results. An isolated observation dominated the raw worst-lab strain RMSE of 3.042.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_03_solver.py`. The public schema and independent checks of the strain curves and initial rates passed. In sparse calibration sweeps with one to three clean points, EMI handling reduced the worst tested stress error from 423.403 to 0.000187.`

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
