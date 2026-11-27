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
| **Turns Used / Max Turns** | `16 / 18` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `481,583` (`464,205` / `130,304` / `17,378`) |
| **Total Episode Wall Time (s)** | `681.9s` |

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

### Turn 1 — `execute_in_remote_docker` (`16725 ms` LLM)
- **Rationale**: Inspect the public schema and available methods to catch wrong field mappings or an incorrect regime dispatch before implementing the fit.

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py; ls -l /tmp/hegel_continual_archive; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_03.pkl','rb'))
for k,v in p.items():
 if isinstance(v,dict): print(k,{a:(type(b).__name__,getattr(b,'shape',None),len(b) if hasattr(b,'__len__') else b) for a,b in v.items()})
 else: print(k,type(v).__name__,getattr(v,'shape',None),repr(v)[:350])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation'
```
- **Container Exit Code**: `0` (`469 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
enous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`18026 ms` LLM)
- **Rationale**: Read the first case inside the list and inspect the archive’s panel layout and training targets. Calibration residuals will determine whether any channel needs the history fallback.

```bash
cd /holdout_task; cat validate_public_schema.py; cat /tmp/hegel_continual_archive/alpha_m97_c02_batch_02_solver.py; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))
print('cases',len(p))
for q in p[:1]:
 print('wrapper',q.keys()); d=q.get('dataset',q)
 for k,v in d.items():
  print(k,{a:(np.shape(b),str(np.asarray(b).dtype)) for a,b in v.items()} if isinstance(v,dict) else v)
 c=d['calibration_load_cells'];
 for lab in np.unique(c['lab_id']):
  m=c['lab_id']==lab; x=c['cmd_stress'][m]; y=c['true_stress'][m]; print('cal',lab,len(x),np.polyfit(x,y,1),np.round(y-np.polyval(np.polyfit(x,y,1),x),4))
 for s in ('train','test'):
  v=d[s]; print(s,'lab counts',np.unique(v['lab_id'],return_counts=True),'NaNs',np.isnan(v['coupon_features_raw']).sum(axis=0)); print('panel first row',v['ref_coupons'][0])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel
```
- **Container Exit Code**: `0` (`477 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`20097 ms` LLM)
- **Rationale**: The previous output truncated the field mapping and imputation code. A bounded inspection will identify which observations constrain the fit and whether any calibration channel needs spike handling before implementing the new law.
- **Alternatives Weighed**: `Reuse the previous solver wholesale` (rejected: The previous linear-flow law does not implement the specified saturating slip term or branch-gap coupling.)

```bash
cd /holdout_task && sed -n '26,200p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_02_solver.py && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb')); print('case count',len(p))
for q in p:
 d=q['dataset']; print('wrapper',list(q),'dataset',list(d),'times',d['eval_times'],'dt range',np.min(d['past_dt']),np.max(d['past_dt']))
 for s in ('train','test'):
  a=d[s]; print(s,{k:np.shape(v) for k,v in a.items()},'labs',np.unique(a['lab_id'],return_counts=True),'missing',np.isnan(a['coupon_features_raw']).sum(axis=0))
 c=d['calibration_load_cells'];
 for l in range(4):
  m=c['lab_id']==l; x,y=c['cmd_stress'][m],c['true_stress'][m]; print('calibration',l,len(x),np.max(np.abs(y-np.polyval(np.polyfit(x,y,1),x))) if len(x)>=2 else None)
PY
```
- **Container Exit Code**: `0` (`449 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ld, dt, time, features)
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
case count 1
wrapper ['dataset', 'gt'] dataset ['past_dt', 'eval_times', 'crit_stress', 'yield_stress', 'shear_banding_index', 'calibration_load_cells', 'train', 'test'] times [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ] dt range 0.2323 0.7147
train {'lab_id': (180,), 'stratum': (180,), 'coupon_features_raw': (180, 4), 'ref_coupons': (180, 32), 'past_stress_cmd': (180, 18), 'stress_hold_cmd': (180,), 'strain_0': (180,), 'aux_chamber_telemetry': (180, 3), 'strain_curves_obs': (180, 10), 'instant_rate_obs': (180,)} labs (array([0, 1, 2, 3]), array([38, 57, 43, 42])) missing [6 8 4 1]
test {'lab_id': (72,), 'stratum': (72,), 'coupon_features_raw': (72, 4), 'ref_coupons': (72, 32), 'past_stress_cmd': (72, 18), 'stress_hold_cmd': (72,), 'strain_0': (72,), 'aux_chamber_telemetry': (72, 3)} labs (array([0, 1, 2, 3]), array([16, 17, 17, 22])) missing [1 2 0 1]
calibration 0 10 0.0002664440837238935
calibration 1 10 0.00014693766250711837
calibration 2 10 0.0001846201486075394
calibration 3 10 0.0002188130456675097
```

### Turn 4 — `execute_in_remote_docker` (`68997 ms` LLM)
- **Rationale**: Implement the specified basis and its analytic onset rate with shared, ordered relaxation times. The isolated schema check catches interface and nonfinite-output failures; the history preview determines whether the documented sparse-calibration path can read the archived records.
- **Alternatives Weighed**: `Reuse the previous linear-flow fit` (rejected: It omits the specified slip saturation and branch-gap coupling.)

```bash
cd /holdout_task && cat > alpha_m97_c02_batch_03_solver.py <<'PY'
"""Calibrated two-branch rheology with saturating jammed slip."""

import json

import numpy as np
from scipy.optimize import least_squares


_HISTORY = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _history_rows(node):
    if isinstance(node, dict):
        keys = ('lab_id', 'cmd_stress', 'true_stress')
        if all(k in node for k in keys):
            yield from zip(*(np.atleast_1d(node[k]) for k in keys))
        else:
            for value in node.values():
                yield from _history_rows(value)
    elif isinstance(node, list):
        for value in node:
            yield from _history_rows(value)


def _calibration(data):
    c = data['calibration_load_cells']
    lab_ids = np.asarray(c['lab_id'])
    cmd, true = np.asarray(c['cmd_stress']), np.asarray(c['true_stress'])
    result = np.empty((4, 2))
    history = None
    for lab in range(4):
        keep = (lab_ids == lab) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[keep], true[keep]
        spike = False
        if len(x) >= 3:
            dx, dy = x[:, None] - x, y[:, None] - y
            slopes = np.divide(dy, dx, out=np.full(dx.shape, np.nan), where=dx != 0)
            gain = np.nanmedian(np.nanmedian(slopes, axis=1))
            offset = y - gain*x
            clean = np.abs(offset - np.median(offset)) <= 3.0
            spike = not clean.all()
            x, y = x[clean], y[clean]
        if len(x) >= 4 or (len(x) >= 2 and not spike):
            result[lab] = _line(x, y)
        else:
            if history is None:
                with open(_HISTORY) as file:
                    history = list(_history_rows(json.load(file)))
            pairs = np.array([(u, v) for c, u, v in history if int(c) == lab and np.isfinite(u) and np.isfinite(v)])
            gain, bias = _line(*pairs.T)
            if len(x):
                bias = np.mean(y - gain*x)
            result[lab] = gain, bias
    return result


def _align(data):
    train, test = data['train'], data['test']
    x, ref, labs, strata = (np.concatenate((train[k], test[k])) for k in
                            ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    for lab in np.unique(labs):
        take = labs == lab
        panel = ref[take]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[take] = (x[take] - center)/scale
    cells = []
    for lab, stratum in np.unique(np.column_stack((labs, strata)), axis=0):
        part = x[(labs == lab) & (strata == stratum)]
        counts = np.isfinite(part).sum(axis=0)
        cells.append(np.divide(np.nansum(part, axis=0), counts,
                               out=np.full(x.shape[1], np.nan), where=counts > 0))
    return np.where(np.isfinite(x), x, np.nanmean(cells, axis=0))


def _physical(split, calibration):
    gain, bias = calibration[split['lab_id']].T
    return (split['past_stress_cmd']*gain[:, None] + bias[:, None],
            split['stress_hold_cmd']*gain + bias)


def _design(tau, past, hold, dt, time, features, jammed):
    decay = np.exp(-dt[:, None]/tau)
    update = -np.expm1(-dt[:, None]/tau)
    memory = np.zeros((len(hold), 2))
    for cmd, a, b in zip(past.T, decay, update):
        memory = a*memory + b*cmd[:, None]
    g0, g1 = (hold[:, None] - memory).T
    a0, a1 = (-np.expm1(-time[:, None]/tau)).T
    phi = time/(1.0 + 0.85*time) if jammed else time
    curves = np.zeros((len(hold), len(time), 9))
    rates = np.zeros((len(hold), 9))
    curves[:, :, 0], curves[:, :, 1] = g0[:, None]*a0, g1[:, None]*a1
    rates[:, 0], rates[:, 1] = g0/tau[0], g1/tau[1]
    curves[:, :, 2:7] = phi[None, :, None]*features[:, None, :]
    rates[:, 2:7] = features
    if jammed:
        slip = np.sign(hold)*np.sqrt(np.abs(hold))
        change = g0 - g1
        curves[:, :, 7], curves[:, :, 8] = slip[:, None]*phi, change[:, None]*a0
        rates[:, 7], rates[:, 8] = slip, change/tau[0]
    return curves, rates


def _taus(p):
    return np.exp([p[0], p[0] + p[1]])


def _mad(residual):
    return max(1e-8, 1.4826*np.median(np.abs(residual - np.median(residual))))


def _fit(past, hold, dt, time, features, y_curve, y_rate, jammed):
    n_curve = y_curve.size
    n_weights = 9 if jammed else 7
    y = np.r_[y_curve.ravel(), y_rate]
    good = np.isfinite(y)
    y = y[good]
    is_curve = (np.arange(len(good)) < n_curve)[good]

    def matrix(p):
        curves, rates = _design(_taus(p), past, hold, dt, time, features, jammed)
        return np.concatenate((curves.reshape(-1, 9), rates))[good, :n_weights]

    def initialize(pair, weights):
        p = np.log([pair[0], pair[1]/pair[0]])
        a = matrix(p)
        return np.r_[p, np.linalg.lstsq(a*weights[:, None], y*weights, rcond=None)[0]]

    def unscaled(p):
        return matrix(p) @ p[2:] - y

    def thresholds(residual):
        return np.where(is_curve, 2.3*_mad(residual[is_curve]), 2.3*_mad(residual[~is_curve]))

    bounds = (np.r_[np.log(0.015), np.log(1.003), np.full(n_weights, -np.inf)],
              np.r_[np.log(30.), np.log(300.), np.full(n_weights, np.inf)])
    starts = [(0.35, 1.8), (0.18, 0.8), (0.8, 4.0), (1.5, 9.0)]
    pilot = initialize(starts[0], np.ones_like(y))
    scale = thresholds(unscaled(pilot))

    def fit(p, loss):
        return least_squares(lambda z: unscaled(z)/scale, p, bounds=bounds, loss=loss,
                             x_scale='jac', max_nfev=100)

    pilot = fit(pilot, 'soft_l1').x
    scale = thresholds(unscaled(pilot))
    best = fit(pilot, 'cauchy')
    for pair in starts[1:]:
        weights = 1/(scale*np.sqrt(1 + (unscaled(best.x)/scale)**2))
        other = fit(initialize(pair, weights), 'cauchy')
        if other.cost < best.cost:
            best = other
    scale = thresholds(unscaled(best.x))
    p = fit(best.x, 'cauchy').x
    weights = np.zeros(9)
    weights[:n_weights] = p[2:]
    return _taus(p), weights


def solve_alpha_m97_c02_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    calibration = _calibration(dataset)
    x = _align(dataset)
    features = np.column_stack((np.ones(len(x)), x))
    n_train = len(train['lab_id'])
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    jammed = dataset['shear_banding_index'] > 0.50
    past, hold = _physical(train, calibration)
    taus, weights = _fit(past, hold, dt, time, features[:n_train],
                         train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs'], jammed)
    past, hold = _physical(test, calibration)
    curves, rates = _design(taus, past, hold, dt, time, features[n_train:], jammed)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
python3 - <<'PY'
import json
p=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('history structure:',str(p)[:650])
PY
```
- **Container Exit Code**: `0` (`817 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
history structure: [{'episode_short_name': 'alpha_m97_c02_s00', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [1.619455, -1.307963, -0.646756, 0.95759, 2.133391, -0.419148, -0.546245, -1.403033, 1.416515, -0.474218, 0.56263, -1.431472, -1.102381, 0.100096, 0.332884, 0.978926, 1.410301, 2.014558, 2.037985, -1.520981, -1.132282, -0.400487, 1.428566, 2.184907, 2.114515, 2.024265, 0.373395, 1.839347, -1.383881, 0.252459, -1.650907, 2.152332, 0.673079, 0.633995, 2.133025, 0.157003, -0.816267, -1.031715, -0.029779, 1.574899], 'true_stress': [1.619593, -1.308033, -0.646

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=dfc9a1bb3228dd23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_dfc9a1bb (directory).
```

### Turn 5 — `execute_in_remote_docker` (`27019 ms` LLM)
- **Rationale**: Hold out each lab’s response labels while retaining its unlabeled reference controls as specified. Check trajectories and onset rates separately, including imputed rows and late horizons. Compare archived calibration sweeps to determine whether pooling them is defensible when the sparse-calibration fallback is needed.
- **Alternatives Weighed**: `Accept the schema pass alone` (rejected: It cannot reveal lab-specific prediction errors, unstable relaxation times, or a slow default execution path.)

```bash
cd /holdout_task && cat > /tmp/rheology_cv.py <<'PY'
import importlib.util, pickle
import numpy as np
path='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',path+'alpha_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open(path+'public_alpha_m97_c02_batch_03.pkl','rb'))[0]['dataset']
f=m.solve_alpha_m97_c02_batch_03
r=f(d)
print('full parameters',*[np.round(r[k],7) for k in ('estimated_taus','hereditary_weights','regime_mode_weights')]); print('calibration',m._calibration(d))
tr=d['train']; fields=d['test'].keys(); e=[]; er=[]; spread=[]
for lab in range(4):
 v=tr['lab_id']==lab
 fold=dict(d,train={k:x[~v] for k,x in tr.items()},test={k:np.concatenate((tr[k][v],d['test'][k])) for k in fields})
 p=f(fold); n=v.sum(); ec=p['predicted_strain'][:n]-tr['strain_curves_obs'][v]; rr=p['initial_rate_pred'][:n]-tr['instant_rate_obs'][v]
 e.append(ec); er.append(rr); spread.append(p['estimated_taus']); missing=np.isnan(tr['coupon_features_raw'][v]).any(axis=1)
 print('lab',lab,'taus',p['estimated_taus'],'strain MAE/p95/max',np.round(np.percentile(np.abs(ec),[95,100]),6),np.mean(np.abs(ec)),'rate MAE/p95/max',np.round(np.percentile(np.abs(rr),[95,100]),6),np.mean(np.abs(rr)), 'complete and imputed MAE',[(int(g.sum()),np.mean(np.abs(ec[g])),np.mean(np.abs(rr[g]))) for g in (~missing,missing) if g.any()]); print('strain abs by time',np.round(np.mean(np.abs(ec),axis=0),6))
print('tau range',np.min(spread,axis=0),np.max(spread,axis=0))
print('archived per-sweep calibration')
import json
for q in json.load(open(m._HISTORY)):
 l,x,y=map(np.asarray,(q['lab_id'],q['cmd_stress'],q['true_stress'])); print(q['episode_short_name'],[m._line(x[l==i],y[l==i]).round(5).tolist() for i in range(4)])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_cv.py
/usr/bin/time -f 'default elapsed %e s' python3 -I validate_public_schema.py
```
- **Container Exit Code**: `127` (`1401 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
full parameters [0.5765685 4.2682667] [ 0.6800215  0.5522831  0.138837   0.1562067 -0.1276988  0.1188692
 -0.0880395] [ 0.3098688 -0.2190671]
calibration [[ 0.78997469 -0.13996011]
 [ 1.30997325  0.15996213]
 [ 0.84998688  0.10001623]
 [ 1.23997232 -0.13003537]]
lab 0 taus [0.57577828 4.24400381] strain MAE/p95/max [0.139407 8.832072] 0.07210003645111025 rate MAE/p95/max [0.107626 0.172371] 0.029272353082505163 complete and imputed MAE [(33, 0.06835482857468438, 0.01597460336450901), (5, 0.09681840843552088, 0.1170375012212798)]
strain abs by time [0.024457 0.040211 0.12443  0.044072 0.025819 0.030888 0.053292 0.044835
 0.260524 0.072472]
lab 1 taus [0.57553304 4.29269692] strain MAE/p95/max [ 0.195235 13.946531] 0.13332307798093537 rate MAE/p95/max [0.053327 0.219952] 0.020481334910176497 complete and imputed MAE [(54, 0.13589782702485464, 0.015223826605324094), (3, 0.08697759519038885, 0.11511648439751981)]
strain abs by time [0.312544 0.037055 0.321499 0.022243 0.022316 0.102159 0.024484 0.099675
 0.247483 0.143773]
lab 2 taus [0.57526117 4.26667456] strain MAE/p95/max [0.329579 4.880343] 0.10300717312047673 rate MAE/p95/max [0.105488 0.128998] 0.02653303564809268 complete and imputed MAE [(39, 0.10430945110937452, 0.017574689344518914), (4, 0.09030996272872341, 0.11387691210793688)]
strain abs by time [0.096918 0.151579 0.14193  0.21842  0.064471 0.028309 0.032259 0.05693
 0.050678 0.188577]
lab 3 taus [0.57935736 4.26636376] strain MAE/p95/max [ 0.189983 60.001705] 0.22022361167772728 rate MAE/p95/max [0.129725 0.197914] 0.033148337289729705 complete and imputed MAE [(36, 0.07711855766148878, 0.021473323713242263), (6, 1.078853935775158, 0.10319841874865433)]
strain abs by time [0.020166 0.359846 0.038737 1.474989 0.058862 0.044078 0.060263 0.060848
 0.03893  0.045516]
tau range [0.57526117 4.24400381] [0.57935736 4.29269692]
archived per-sweep calibration
alpha_m97_c02_s00 [[0.99998, 0.0], [0.99999, 4e-05], [1.00001, 3e-05], [1.00001, -3e-05]]
alpha_m97_c02_batch_01 [[0.79005, -0.13999], [1.30996, 0.16006], [0.84996, 0.10004], [1.24, -0.12996]]
alpha_m97_c02_batch_02 [[0.79001, -0.14], [1.30998, 0.15996], [0.84999, 0.09998], [1.24004, -0.12998]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=dfc9a1bb3228dd23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_dfc9a1bb (directory).
```
- **Stderr (tail)**:
```text
bash: line 24: /usr/bin/time: No such file or directory
```

### Turn 6 — `execute_in_remote_docker` (`42447 ms` LLM)
- **Rationale**: The worst trajectory MAE includes a roughly 60-unit residual at a single observation, while fitted relaxation times remain stable across held-out labs. The actionable failure is the historical fallback: test missing calibration and isolated EMI on each channel, then verify the edited solver in an isolated process under the default runtime settings.
- **Alternatives Weighed**: `Pool every archived sweep` (rejected: The first sweep belongs to a different calibration regime and would move the fallback away from both recent sweeps and this batch.)
- **ATMS NOGOOD DECLARED**: Refuted `All archived sweeps share one physical gain and bias per channel.` | Witness: `The first sweep had gains near 1.00 for all channels, whereas the two recent sweeps and current batch agree near [0.79, 1.31, 0.85, 1.24].` | Blanket Diagnosis: `Pooling across the hardware calibration change biases the stress inputs and therefore both memory gaps and slip on sparse channels.` | Sublated Pivot: `Keep direct fits on clean in-batch data and use the latest usable sweep only in the documented fallback.`

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c02_batch_03_solver.py')
s = p.read_text()
s = s.replace('history = list(_history_rows(json.load(file)))', 'history = json.load(file)')
s = s.replace('            pairs = np.array([(u, v) for c, u, v in history if int(c) == lab and np.isfinite(u) and np.isfinite(v)])\n            gain, bias = _line(*pairs.T)', '''            # A hardware change separates the identity sweep from the recent prefix.
            # Use the latest sweep that can estimate this channel's gain.
            for sweep in reversed(history):
                pairs = np.array([(u, v) for c, u, v in _history_rows(sweep)
                                  if int(c) == lab and np.isfinite(u) and np.isfinite(v)])
                if len(pairs) >= 2 and np.ptp(pairs[:, 0]) > 0:
                    gain, bias = _line(*pairs.T)
                    break
            else:
                raise ValueError(f'No usable archived calibration for channel {lab}')''')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
s=importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; base=m._calibration(d)
for lab in range(4):
    ind=np.flatnonzero(c['lab_id']==lab)
    for n in (0,1,2,3,4):
        keep=np.ones(len(c['lab_id']),bool); keep[ind[n:]]=False
        few={k:v[keep] for k,v in c.items()}
        estimate=m._calibration(dict(d,calibration_load_cells=few))
        if n<2: np.testing.assert_allclose(estimate[lab],base[lab],atol=0.001,rtol=0)
        else: np.testing.assert_allclose(estimate[lab],m._line(c['cmd_stress'][ind[:n]],c['true_stress'][ind[:n]]),atol=1e-12,rtol=0)
    for j in ind:
        for amplitude in (-6.,6.):
            spike={k:v.copy() for k,v in c.items()}; spike['true_stress'][j]+=amplitude
            estimate=m._calibration(dict(d,calibration_load_cells=spike))
            clean=ind[ind!=j]
            np.testing.assert_allclose(estimate[lab],m._line(c['cmd_stress'][clean],c['true_stress'][clean]),atol=1e-12,rtol=0)
print('Calibration checks passed: 0–4 retained points and isolated positive/negative EMI on every public calibration row.')
PY
TIMEFORMAT='default elapsed %R s'; time timeout 90s python3 -I validate_public_schema.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS viscoelastic relaxation' | head -n 12
```
- **Container Exit Code**: `0` (`1157 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_local_level_step(predictor: np.ndarray, values: np.ndarray, splits: np.ndarray, *, scale: float, loss_clip: float) -> tuple[int, float] | Fit finite predictor and values (n,) and integer splits (k,) with k >= 1, at l | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_local_level_step | STATUS: APPLICABLE_SUBROUTINE(real_1d_step_series)
autogenous_prim_robust_affine_projection@abc35d3d | profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | For predictor, target and optional weights (n, m) and predictor sensitivity (n | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.profile_weighted_affine | STATUS: APPLICABLE_SUBROUTINE(real_2d_predictor_3d_sensitivity_2d_target)
autogenous_prim_robust_affine_projection@abc35d3d | propagate_linear_basis(transition: np.ndarray, initial: np.ndarray, increments: np.ndarray) -> np.ndarray | Return a trajectory of shape (steps+1, n, k) from transition (n, n), initial ( | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.propagate_linear_basis | STATUS: APPLICABLE_SUBROUTINE(real_linear_transition_and_2d_initial_basis)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=71c2e1268921b4fe (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_71c2e126 (directory).
```
- **Stderr (tail)**:
```text
default elapsed 0.390 s
```

### Turn 7 — `execute_in_remote_docker` (`23606 ms` LLM)
- **Rationale**: Score the unchanged solver against any matching public reference outputs. Unlike the schema check or noisy leave-one-lab-out observations, this can distinguish a parameter or prediction error from a large observation residual.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
s = importlib.util.spec_from_file_location('solver', 'alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
with open('public_alpha_m97_c02_batch_03.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    d, gt = item['dataset'], item['gt']
    p = m.solve_alpha_m97_c02_batch_03(d)
    print('Public reference fields:', {k: np.shape(v) for k,v in gt.items()})
    for k, v in p.items():
        if k not in gt or np.shape(gt[k]) != v.shape:
            continue
        e = np.abs(v - np.asarray(gt[k]))
        print(k, 'MAE', float(e.mean()), 'p95', float(np.percentile(e, 95)), 'max', float(e.max()))
        if v.ndim > 1 or k == 'initial_rate_pred':
            lab = d['test']['lab_id']
            print('  per-lab MAE', {int(i): float(e[lab == i].mean()) for i in np.unique(lab)})
        else:
            print('  absolute errors by coefficient', e)
PY
```
- **Container Exit Code**: `0` (`721 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public reference fields: {'taus': (2,), 'theta': (7,), 'vp_theta': (2,), 'stress_gains': (4,), 'stress_biases': (4,), 'te_clean': (72, 10), 'te_true_rate0': (72,), 'te_hist_true': (72, 18), 'te_hold_true': (72,)}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=71c2e1268921b4fe (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_71c2e126 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`25223 ms` LLM)
- **Rationale**: The previous comparison skipped all outputs because the reference keys differ from the solver keys. Map them explicitly to verify both coefficient vectors and clean onset rates; a discrepancy isolated to imputed rows would point to panel alignment, while a discrepancy across complete rows would point to calibration or the dynamics fit.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_alpha_m97_c02_batch_03.pkl', 'rb') as f:
    items = pickle.load(f)
for i, item in enumerate(items):
    d, gt = item['dataset'], item['gt']
    p = m.solve_alpha_m97_c02_batch_03(d)
    print('case', i)
    for k, target in [('estimated_taus','taus'), ('hereditary_weights','theta'), ('regime_mode_weights','vp_theta'), ('predicted_strain','te_clean'), ('initial_rate_pred','te_true_rate0')]:
        v, truth = np.asarray(p[k]), np.asarray(gt[target])
        assert v.shape == truth.shape and np.isfinite(v).all(), k
        e = np.abs(v - truth)
        if k in ('predicted_strain', 'initial_rate_pred'):
            lab = d['test']['lab_id']
            missing = np.isnan(d['test']['coupon_features_raw']).any(axis=1)
            print(k, 'MAE', e.mean(), 'p95', np.percentile(e, 95), 'max', e.max(), 'lab MAE', {int(j): float(e[lab == j].mean()) for j in np.unique(lab)}, 'complete/imputed MAE', {str(flag): float(e[missing == flag].mean()) for flag in (False, True) if np.any(missing == flag)})
            if e.ndim == 2:
                print('worst horizon MAE', e.mean(axis=0).max())
        else:
            print(k, 'coefficient absolute errors', e, 'max relative error', np.max(e/np.maximum(np.abs(truth), 1e-12)))
    hist, hold = m._physical(d['test'], m._calibration(d))
    print('physical stress max absolute errors', np.max(np.abs(hist-gt['te_hist_true'])), np.max(np.abs(hold-gt['te_hold_true'])))
PY
```
- **Container Exit Code**: `0` (`702 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case 0
estimated_taus coefficient absolute errors [0.00056853 0.03326675] max relative error 0.00785519368325855
hereditary_weights coefficient absolute errors [2.15275153e-05 2.28306937e-03 1.16295193e-03 3.79331055e-03
 2.30120943e-03 1.13076798e-03 1.96052818e-03] max relative error 0.023708190915294226
regime_mode_weights coefficient absolute errors [0.00013116 0.00093292] max relative error 0.004240536544997498
predicted_strain MAE 0.007683060795742825 p95 0.02173270131160758 max 0.12898668976719274 lab MAE {0: 0.006108585733857202, 1: 0.0032611499925433948, 2: 0.006080437394306178, 3: 0.013483455453423885} complete/imputed MAE {'False': 0.0053944255075338695, 'True': 0.06032167242454884}
worst horizon MAE 0.00980142715952172
initial_rate_pred MAE 0.010762067007612236 p95 0.026584327486606104 max 0.1320132544990149 lab MAE {0: 0.008323865538521392, 1: 0.004800382478547589, 2: 0.008767878547609177, 3: 0.018683024476685166} complete/imputed MAE {'False': 0.007744019987740955, 'True': 0.0801771484646517}
physical stress max absolute errors 0.00010163077591363034 8.012874888030375e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=71c2e1268921b4fe (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_71c2e126 (directory).
```

### Turn 9 — `submit_completed_task` (`16533 ms` LLM)
- **Rationale**: The saved solver passed isolated schema validation, sparse-calibration and EMI stress tests, and clean-reference checks for all five outputs. Leave-one-lab-out fits kept both relaxation times stable. The largest remaining errors are concentrated in rows with missing coupon features; the solver retains the specified cell-balanced imputation.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_03_solver.py': 'FILE:7502'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_03_solver.py`. The solver passed the public schema and calibration stress tests. Against the clean public reference, strain MAE was 0.00768 and initial-rate MAE was 0.01076. Rows with imputed coupon features had the largest errors.`

### Turn 10 — `execute_in_remote_docker` (`41664 ms` LLM)
- **Rationale**: An independent closed-form memory calculation and numerical onset derivative detect omitted terms or sign errors. Comparing fitted and reference parameters under the same panel alignment distinguishes missing-feature error from errors the fit can reduce. An independent affine shift in each lab tests whether nuisance units leak into the outputs; if it fails, revisit alignment before changing the dynamics.
- **Alternatives Weighed**: `Test relay modes, Fokker–Planck operators, or classification balanced accuracy` (rejected: The data contain continuous strain and rate measurements, not relay states, a stochastic drift operator, or class labels. The applicable checks are the rational saturation, branch-gap difference, and lab-specific alignment.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

s = importlib.util.spec_from_file_location('solver', 'alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
for item in pickle.load(open('public_alpha_m97_c02_batch_03.pkl', 'rb')):
    d, gt = item['dataset'], item['gt']
    tr, te = d['train'], d['test']
    p = m.solve_alpha_m97_c02_batch_03(d)
    assert d['shear_banding_index'] > 0.5
    x = m._align(d)[len(tr['lab_id']):]
    t = np.asarray(d['eval_times'])
    past, hold = m._physical(te, m._calibration(d))
    def law(tau, hw, rw, h, u, times):
        dt = np.asarray(d['past_dt'])
        remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
        z = h @ ((1 - np.exp(-dt[:, None]/tau)) * np.exp(-remaining[:, None]/tau))
        g = u[:, None] - z
        a = 1 - np.exp(-times[:, None]/tau)
        phi = times/(1 + .85*times)
        flow = hw[2] + x @ hw[3:] + rw[0]*np.sign(u)*np.sqrt(abs(u))
        return (g*hw[:2]) @ a.T + flow[:, None]*phi + rw[1]*(g[:, 0]-g[:, 1])[:, None]*a[:, 0]
    params = tuple(p[k] for k in ('estimated_taus', 'hereditary_weights', 'regime_mode_weights'))
    exact = te['strain_0'][:, None] + law(*params, past, hold, t)
    derivative = law(*params, past, hold, np.array([1e-5, -1e-5])) @ np.array([5e4, -5e4])
    print('independent closed-form curve and onset max differences', np.max(abs(exact-p['predicted_strain'])), np.max(abs(derivative-p['initial_rate_pred'])))
    np.testing.assert_allclose(exact, p['predicted_strain'], rtol=0, atol=1e-11)
    np.testing.assert_allclose(derivative, p['initial_rate_pred'], rtol=0, atol=1e-8)
    ref = (gt['taus'], gt['theta'], gt['vp_theta'])
    oracle_curve = te['strain_0'][:, None] + law(*ref, gt['te_hist_true'], gt['te_hold_true'], t)
    oracle_rate = law(*ref, gt['te_hist_true'], gt['te_hold_true'], np.array([1e-5, -1e-5])) @ np.array([5e4, -5e4])
    pred = (p['predicted_strain'], p['initial_rate_pred']); truth = (gt['te_clean'], gt['te_true_rate0'])
    missing = np.isnan(te['coupon_features_raw']).any(axis=1)
    for label, mask in [('complete', ~missing), ('imputed', missing)] + [(f'lab {i}', te['lab_id']==i) for i in range(4)]:
        if mask.any():
            print(label, int(mask.sum()), 'strain/rate fitted MAE', [round(np.mean(abs(a[mask]-b[mask])), 7) for a,b in zip(pred, truth)], 'reference-parameter MAE with same panel', [round(np.mean(abs(a[mask]-b[mask])), 7) for a,b in zip((oracle_curve, oracle_rate), truth)])
    shifted = dict(d)
    for part in ('train', 'test'):
        v = d[part]; lab = v['lab_id']; gain = np.array([.65, 1.3, .9, 1.6])[lab, None]; bias = np.array([1., -.7, 2., -.8])[lab, None]
        shifted[part] = dict(v, coupon_features_raw=v['coupon_features_raw']*gain+bias, ref_coupons=v['ref_coupons']*gain+bias)
    q = m.solve_alpha_m97_c02_batch_03(shifted)
    print('per-lab affine-gauge max output difference', max(np.max(abs(p[k]-q[k])) for k in p))
    for k in p:
        np.testing.assert_allclose(p[k], q[k], rtol=0, atol=1e-8)
PY
python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`1198 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
independent closed-form curve and onset max differences 8.881784197001252e-16 1.7710632960188377e-10
complete 69 strain/rate fitted MAE [0.0053944, 0.007744] reference-parameter MAE with same panel [0.0050957, 0.0068478]
imputed 3 strain/rate fitted MAE [0.0603217, 0.0801771] reference-parameter MAE with same panel [0.065152, 0.0875532]
lab 0 16 strain/rate fitted MAE [0.0061086, 0.0083239] reference-parameter MAE with same panel [0.0091324, 0.0122723]
lab 1 17 strain/rate fitted MAE [0.0032611, 0.0048004] reference-parameter MAE with same panel [0.0045941, 0.0061737]
lab 2 17 strain/rate fitted MAE [0.0060804, 0.0087679] reference-parameter MAE with same panel [0.0058758, 0.007896]
lab 3 22 strain/rate fitted MAE [0.0134835, 0.018683] reference-parameter MAE with same panel [0.0101343, 0.0136188]
per-lab affine-gauge max output difference 1.954325590247663e-09
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=71c2e1268921b4fe (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_71c2e126 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`29433 ms` LLM)
- **Rationale**: The exact law and lab alignment passed independent checks. Earlier calibration tests separated missing rows from EMI and used spikes well above the threshold. Combining those documented conditions on the real command grids can expose a missed spike; if it does, revise only the calibration decision.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, itertools, pickle
import numpy as np
s = importlib.util.spec_from_file_location('solver', 'alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
d = pickle.load(open('public_alpha_m97_c02_batch_03.pkl', 'rb'))[0]['dataset']
c = d['calibration_load_cells']; base = m._calibration(d)
for n, amplitude in [(10, 3.05), (5, 3.05), (4, 6.), (3, 6.)]:
    failures = total = 0; worst = 0.; witness = None
    for lab in range(4):
        ids = np.flatnonzero(c['lab_id'] == lab)
        for subset in itertools.combinations(ids, n):
            keep = (c['lab_id'] != lab); keep[list(subset)] = True
            selected = np.flatnonzero(c['lab_id'][keep] == lab)
            for ix in selected:
                for sign in (-1., 1.):
                    q = {k:v[keep] for k,v in c.items()}; q['true_stress'][ix] += sign*amplitude
                    ab = m._calibration(dict(d, calibration_load_cells=q))[lab]
                    stress_error = np.max(np.abs((ab[0]-base[lab, 0])*c['cmd_stress'][ids] + ab[1]-base[lab, 1]))
                    total += 1; failures += int(stress_error > .01)
                    if stress_error > worst:
                        worst = stress_error; witness = (lab, tuple(int(j) for j in subset), int(ix), sign)
    print('points', n, 'spike', amplitude, 'cases above 0.01 physical stress error', failures, '/', total, 'max', round(worst, 6), 'witness', witness)
PY
```
- **Container Exit Code**: `0` (`21348 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
points 10 spike 3.05 cases above 0.01 physical stress error 0 / 80 max 0.000113 witness (3, (30, 31, 32, 33, 34, 35, 36, 37, 38, 39), 38, -1.0)
points 5 spike 3.05 cases above 0.01 physical stress error 0 / 10080 max 0.000969 witness (3, (30, 31, 32, 35, 38), 31, -1.0)
points 4 spike 6.0 cases above 0.01 physical stress error 0 / 6720 max 0.000305 witness (3, (30, 31, 35, 37), 31, -1.0)
points 3 spike 6.0 cases above 0.01 physical stress error 1914 / 2880 max 166.861366 witness (0, (2, 5, 9), 2, 1.0)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=71c2e1268921b4fe (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_71c2e126 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`44613 ms` LLM)
- **Rationale**: Fix the demonstrated calibration failure without changing the verified rheology law. Re-run the failing combinations and near-threshold spikes, confirm clean triples do not consult history, and check that the normal public case still passes in an isolated process.
- **Alternatives Weighed**: `Use archived calibration for every three-row sweep` (rejected: Clean three-row sweeps can be fitted directly, and the task restricts historical fallback to sparse or EMI-affected channels.)
- **ATMS NOGOOD DECLARED**: Refuted `The in-batch repeated median can reliably separate one transient EMI spike from the physical calibration line when only three measurements remain.` | Witness: `On the real calibration grids with three retained rows and one injected ±6 stress spike, 1,914 of 2,880 cases exceeded 0.01 physical stress error; the worst error was 166.861366.` | Blanket Diagnosis: `With three observations, a high-leverage corrupted point can steer the robust line and mask its own residual. The independently checked rheology basis does not cause this error.` | Sublated Pivot: `Flag a leave-one-out discrepancy above 3.0 on three-row sweeps and use the latest usable archived slope to separate the offset from the transient spike.`

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c02_batch_03_solver.py')
s = p.read_text()
old = '''            spike = not clean.all()
            x, y = x[clean], y[clean]
'''
new = '''            spike = not clean.all()
            if len(x) == 3:
                # A high-leverage point can steer the repeated median with
                # only three observations. Flag a >3 leave-one-out discrepancy
                # and use the archived slope to identify which point is bad.
                for j in range(3):
                    other = np.arange(3) != j
                    if np.ptp(x[other]) > 0:
                        a, b = _line(x[other], y[other])
                        spike |= abs(y[j] - (a*x[j] + b)) > 3.0
            else:
                x, y = x[clean], y[clean]
'''
assert old in s
s = s.replace(old, new, 1)
old = '''            if len(x):
                bias = np.mean(y - gain*x)
'''
new = '''            if spike and len(x) == 3:
                offsets = y - gain*x
                clean = np.abs(offsets - np.median(offsets)) <= 3.0
                x, y = x[clean], y[clean]
            if len(x):
                bias = np.mean(y - gain*x)
'''
assert old in s
p.write_text(s.replace(old, new, 1))
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import itertools
import pickle
from unittest.mock import patch
import numpy as np
s = importlib.util.spec_from_file_location('solver', 'alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
d = pickle.load(open('public_alpha_m97_c02_batch_03.pkl', 'rb'))[0]['dataset']
c = d['calibration_load_cells']; base = m._calibration(d)
for n, amplitude in [(10, 3.05), (5, 3.05), (4, 6.), (3, 3.05), (3, 6.)]:
    worst = 0.; total = 0
    for lab in range(4):
        ids = np.flatnonzero(c['lab_id'] == lab)
        for subset in itertools.combinations(ids, n):
            keep = c['lab_id'] != lab; keep[list(subset)] = True
            q = {k:v[keep] for k,v in c.items()}
            selected = np.flatnonzero(q['lab_id'] == lab)
            if n == 3 and amplitude == 3.05:
                with patch.object(m, '_HISTORY', '/no_archive_needed_for_clean_triples'):
                    fit = m._calibration(dict(d, calibration_load_cells=q))[lab]
                np.testing.assert_allclose(fit, m._line(c['cmd_stress'][list(subset)], c['true_stress'][list(subset)]), atol=1e-12, rtol=0)
            for ix in selected:
                for sign in (-1., 1.):
                    y = q['true_stress'].copy(); y[ix] += sign*amplitude
                    ab = m._calibration(dict(d, calibration_load_cells=dict(q, true_stress=y)))[lab]
                    err = np.max(np.abs((ab[0]-base[lab,0])*c['cmd_stress'][ids] + ab[1]-base[lab,1]))
                    total += 1; worst = max(worst, err)
    print('rows', n, 'EMI amplitude', amplitude, 'cases', total, 'worst physical stress error', worst)
    assert worst < 0.01
p = m.solve_alpha_m97_c02_batch_03(d)
prev = importlib.util.spec_from_file_location('previous', '/tmp/.hegel_candidate_archive/v02_71c2e126/alpha_m97_c02_batch_03_solver.py')
if __import__('os').path.isfile(prev.origin):
    old = importlib.util.module_from_spec(prev); prev.loader.exec_module(old)
    for k,v in old.solve_alpha_m97_c02_batch_03(d).items():
        np.testing.assert_array_equal(v,p[k])
PY
python3 -I validate_public_schema.py
```
- **Container Exit Code**: `1` (`23485 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
rows 10 EMI amplitude 3.05 cases 80 worst physical stress error 0.00011261301675297153
rows 5 EMI amplitude 3.05 cases 10080 worst physical stress error 0.0009688815337219325
rows 4 EMI amplitude 6.0 cases 6720 worst physical stress error 0.0003047860224332233
rows 3 EMI amplitude 3.05 cases 2880 worst physical stress error 3.5576595641081905

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3e740d0f81edbac0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_3e740d0f (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 29, in <module>
AssertionError
```

### Turn 13 — `execute_in_remote_docker` (`30569 ms` LLM)
- **Rationale**: The new stress test stopped before verification. Compare the leave-one-out trigger with the archive-based offset check on failing triples to determine which decision needs to change without touching the verified rheology fit.

```bash
set -e
cd /holdout_task
sed -n '18,90p' alpha_m97_c02_batch_03_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, itertools, json, pickle
import numpy as np
s=importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('public_alpha_m97_c02_batch_03.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; base=m._calibration(d)
recent=json.load(open(m._HISTORY))[-1]
worst=(0,None); counts={}
for lab in range(4):
    ids=np.flatnonzero(c['lab_id']==lab)
    u,v=np.array([(u,v) for i,u,v in m._history_rows(recent) if i==lab]).T
    a,b=m._line(u,v)
    for subset in itertools.combinations(ids,3):
        keep=c['lab_id']!=lab; keep[list(subset)]=True
        q={k:v[keep] for k,v in c.items()}; idx=np.flatnonzero(q['lab_id']==lab); x=q['cmd_stress'][idx]
        for j in range(3):
            for sign in (-1,1):
                y=q['true_stress'].copy(); y[idx[j]]+=sign*3.05; yy=y[idx]
                est=m._calibration(dict(d,calibration_load_cells=dict(q,true_stress=y)))[lab]
                err=np.max(np.abs((est[0]-base[lab,0])*c['cmd_stress'][ids]+est[1]-base[lab,1]))
                if err>0.01:
                    loo=[]
                    for k in range(3):
                        mask=np.arange(3)!=k
                        gain,bias=m._line(x[mask],yy[mask]); loo.append(float(yy[k]-gain*x[k]-bias))
                    offsets=yy-a*x; key=(max(map(abs,loo))>3,np.max(np.abs(offsets-np.median(offsets)))>3)
                    counts[key]=counts.get(key,0)+1
                    if err>worst[0]: worst=(err,dict(lab=lab,subset=subset,spike=j,sign=sign,x=x.tolist(),y=yy.tolist(),loo=loo,archived_offsets=offsets.tolist(),estimate=est.tolist()))
print('failure counts by (LOO trigger, archive offset trigger):',counts)
print('worst witness:',worst)
PY
```
- **Container Exit Code**: `0` (`5014 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
e(3) != j
                    if np.ptp(x[other]) > 0:
                        a, b = _line(x[other], y[other])
                        spike |= abs(y[j] - (a*x[j] + b)) > 3.0
            else:
                x, y = x[clean], y[clean]
        if len(x) >= 4 or (len(x) >= 2 and not spike):
            result[lab] = _line(x, y)
        else:
            if history is None:
                with open(_HISTORY) as file:
                    history = json.load(file)
            # A hardware change separates the identity sweep from the recent prefix.
            # Use the latest sweep that can estimate this channel's gain.
            for sweep in reversed(history):
                pairs = np.array([(u, v) for c, u, v in _history_rows(sweep)
                                  if int(c) == lab and np.isfinite(u) and np.isfinite(v)])
                if len(pairs) >= 2 and np.ptp(pairs[:, 0]) > 0:
                    gain, bias = _line(*pairs.T)
                    break
            else:
                raise ValueError(f'No usable archived calibration for channel {lab}')
            if spike and len(x) == 3:
                offsets = y - gain*x
                clean = np.abs(offsets - np.median(offsets)) <= 3.0
                x, y = x[clean], y[clean]
            if len(x):
                bias = np.mean(y - gain*x)
            result[lab] = gain, bias
    return result


def _align(data):
    train, test = data['train'], data['test']
    x, ref, labs, strata = (np.concatenate((train[k], test[k])) for k in
                            ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    for lab in np.unique(labs):
        take = labs == lab
        panel = ref[take]
        center = np.nanmedian(np.nanmean(panel, axis=0))
failure counts by (LOO trigger, archive offset trigger): {(False, True): 2}
worst witness: (3.5576595641081905, {'lab': 0, 'subset': (2, 5, 8), 'spike': 2, 'sign': 1, 'x': [-1.868094, -1.876496, 1.000356], 'y': [-1.6156115083534845, -1.6223949132305664, 3.7002332021676487], 'loo': [-0.00876161496598682, 0.008787278682960586, 2.9999844725222626], 'archived_offsets': [-0.13978979091617205, -0.13993549229979574, 2.909937258832506], 'estimate': [1.8516753915935495, 1.8478857823859782]})

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3e740d0f81edbac0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_3e740d0f (directory).
```

### Turn 14 — `execute_in_remote_docker` (`44016 ms` LLM)
- **Rationale**: A 3.05 spike produced a leave-one-out residual of 2.999984 because the two clean commands were only 0.0084 units apart. Add a margin only to the screening step, verify clean triples still use direct fits, and test near-threshold spikes across sparse and full sweeps. Exact equality of all five public outputs checks that the calibration fix did not change the validated rheology fit.
- **Alternatives Weighed**: `Lower the final EMI threshold` (rejected: The observed failure was in the extrapolated screening residual, not the archive-based confirmation. The final threshold stays at 3.0.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c02_batch_03_solver.py')
s = p.read_text()
old = '''                # A high-leverage point can steer the repeated median with
                # only three observations. Flag a >3 leave-one-out discrepancy
                # and use the archived slope to identify which point is bad.'''
new = '''                # A close pair can amplify calibration noise when extrapolated.
                # Screen with a 0.10 stress-unit margin; confirm against the
                # recent archived slope at the actual EMI threshold below.'''
assert old in s
s = s.replace(old, new, 1).replace('spike |= abs(y[j] - (a*x[j] + b)) > 3.0', 'spike |= abs(y[j] - (a*x[j] + b)) > 2.90', 1)
old = '''                clean = np.abs(offsets - np.median(offsets)) <= 3.0
                x, y = x[clean], y[clean]'''
new = '''                clean = np.abs(offsets - np.median(offsets)) <= 3.0
                if clean.all():
                    result[lab] = _line(x, y)
                    continue
                x, y = x[clean], y[clean]'''
assert old in s
p.write_text(s.replace(old, new, 1))
PY
cat > /tmp/rheology_emi_check.py <<'PY'
import importlib.util
import itertools
import pickle
from unittest.mock import patch
import numpy as np
path = '/holdout_task/'
s = importlib.util.spec_from_file_location('solver', path+'alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open(path+'public_alpha_m97_c02_batch_03.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']; base = m._calibration(d)
for lab in range(4):
    ids = np.flatnonzero(c['lab_id']==lab)
    for rows in itertools.combinations(ids,3):
        keep = c['lab_id']!=lab; keep[list(rows)] = True
        q = {k:v[keep] for k,v in c.items()}
        with patch.object(m,'_HISTORY','/archive_not_needed_for_clean_triples'):
            got = m._calibration(dict(d, calibration_load_cells=q))[lab]
        np.testing.assert_allclose(got, m._line(c['cmd_stress'][list(rows)],c['true_stress'][list(rows)]),rtol=0,atol=1e-12)
print('Clean triples: all 480 used direct fits without archive access.')
failed = False
for n, amplitude in [(3,3.01),(3,3.05),(3,6.),(4,3.05),(5,3.05),(10,3.05)]:
    worst = 0.; count = failures = 0; witness = None
    for lab in range(4):
        ids = np.flatnonzero(c['lab_id']==lab)
        for rows in itertools.combinations(ids,n):
            keep = c['lab_id']!=lab; keep[list(rows)] = True
            q = {k:v[keep] for k,v in c.items()}
            for j, i in enumerate(np.flatnonzero(q['lab_id']==lab)):
                for sign in (-1,1):
                    y = q['true_stress'].copy(); y[i] += sign*amplitude
                    got = m._calibration(dict(d, calibration_load_cells=dict(q,true_stress=y)))[lab]
                    stress_error = np.max(np.abs(np.polyval(got-base[lab],c['cmd_stress'][ids])))
                    count += 1; failures += int(stress_error > .01)
                    if stress_error > worst:
                        worst = stress_error; witness = (lab,rows,j,sign)
    print('rows',n,'EMI',amplitude,'failures',failures,'/',count,'worst stress error',worst,'witness',witness)
    failed |= bool(failures)
reference = importlib.util.spec_from_file_location('reference','/tmp/.hegel_candidate_archive/v02_71c2e126/alpha_m97_c02_batch_03_solver.py')
old = importlib.util.module_from_spec(reference); reference.loader.exec_module(old)
actual = m.solve_alpha_m97_c02_batch_03(d)
for k,v in old.solve_alpha_m97_c02_batch_03(d).items():
    np.testing.assert_array_equal(actual[k],v)
print('All five public outputs unchanged from the validated baseline.')
assert not failed
PY
python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_emi_check.py
```
- **Container Exit Code**: `1` (`35051 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Clean triples: all 480 used direct fits without archive access.
rows 3 EMI 3.01 failures 0 / 2880 worst stress error 0.00033490055671917267 witness (3, (30, 33, 34), 0, -1)
rows 3 EMI 3.05 failures 0 / 2880 worst stress error 0.00033490055671917267 witness (3, (30, 33, 34), 0, -1)
rows 3 EMI 6.0 failures 0 / 2880 worst stress error 0.00033490055671917267 witness (3, (30, 33, 34), 0, -1)
rows 4 EMI 3.05 failures 14 / 6720 worst stress error 3.560364475553656 witness (0, (2, 5, 8, 9), 2, 1)
rows 5 EMI 3.05 failures 0 / 10080 worst stress error 0.0009688815337219218 witness (3, (30, 31, 32, 35, 38), 1, -1)
rows 10 EMI 3.05 failures 0 / 80 worst stress error 0.00011261301675296371 witness (3, (30, 31, 32, 33, 34, 35, 36, 37, 38, 39), 8, -1)
All five public outputs unchanged from the validated baseline.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=edb4fdc0265d72d9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_edb4fdc0 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/rheology_emi_check.py", line 45, in <module>
    assert not failed
           ^^^^^^^^^^
AssertionError
```

### Turn 15 — `execute_in_remote_docker` (`37948 ms` LLM)
- **Rationale**: Extend the existing sparse-EMI fix to the observed four-row failure without changing the rheology law or clean calibration fits. Re-run the failing cases and near-threshold variants, assert direct fits without archive access on clean sweeps, and require exact equality of all five public outputs before submission.
- **Alternatives Weighed**: `Use archived gains for every sparse sweep` (rejected: Clean sweeps with two or more points can be fitted directly; the archive is needed only for missing coverage or suspected EMI.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c02_batch_03_solver.py')
s = p.read_text().replace('import json\n', 'import json\nfrom itertools import combinations\n', 1)
a = s.index('        if len(x) >= 3:')
b = s.index('        if len(x) >= 4 or (len(x) >= 2 and not spike):', a)
s = s[:a] + '''        sparse = len(x) in (3, 4)
        if sparse:
            # Screen pair extrapolations with a margin for calibration noise.
            # A recent archived slope confirms suspected EMI below.
            for j, k in combinations(range(len(x)), 2):
                dx = x[k] - x[j]
                if dx != 0:
                    residual = y - y[j] - (y[k] - y[j]) / dx * (x - x[j])
                    if np.max(np.abs(residual)) > 2.90:
                        spike = True
                        break
        elif len(x) >= 5:
            dx, dy = x[:, None] - x, y[:, None] - y
            slopes = np.divide(dy, dx, out=np.full(dx.shape, np.nan), where=dx != 0)
            gain = np.nanmedian(np.nanmedian(slopes, axis=1))
            offset = y - gain*x
            clean = np.abs(offset - np.median(offset)) <= 3.0
            spike = not clean.all()
            x, y = x[clean], y[clean]
''' + s[b:]
s = s.replace('if len(x) >= 4 or (len(x) >= 2 and not spike):', 'if len(x) >= 2 and (not spike or (not sparse and len(x) >= 4)):', 1)
s = s.replace('if spike and len(x) == 3:', 'if spike and sparse:', 1)
p.write_text(s)
p = Path('/tmp/rheology_emi_check.py')
s = p.read_text()
a = s.index('for lab in range(4):')
b = s.index('failed = False', a)
s = s[:a] + '''for n in (2, 3, 4):
    count = 0
    for lab in range(4):
        ids = np.flatnonzero(c['lab_id']==lab)
        for rows in itertools.combinations(ids,n):
            keep = c['lab_id']!=lab; keep[list(rows)] = True
            q = {k:v[keep] for k,v in c.items()}
            with patch.object(m,'_HISTORY','/archive_not_needed_for_clean_calibration'):
                got = m._calibration(dict(d, calibration_load_cells=q))[lab]
            np.testing.assert_allclose(got, m._line(c['cmd_stress'][list(rows)],c['true_stress'][list(rows)]),rtol=0,atol=1e-12)
            count += 1
    print('Clean',n,'row sweeps:',count,'direct fits without archive access.')
for lab in range(4):
    ids = np.flatnonzero(c['lab_id']==lab)
    for n in (0, 1):
        keep = c['lab_id']!=lab; keep[ids[:n]] = True
        q = {k:v[keep] for k,v in c.items()}
        got = m._calibration(dict(d,calibration_load_cells=q))[lab]
        np.testing.assert_allclose(got, base[lab], rtol=0, atol=0.001)
''' + s[b:]
s = s.replace('(4,3.05),(5,3.05)', '(4,3.01),(4,3.05),(4,6.),(5,3.05)')
p.write_text(s)
PY
python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_emi_check.py
```
- **Container Exit Code**: `0` (`44792 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Clean 2 row sweeps: 180 direct fits without archive access.
Clean 3 row sweeps: 480 direct fits without archive access.
Clean 4 row sweeps: 840 direct fits without archive access.
rows 3 EMI 3.01 failures 0 / 2880 worst stress error 0.00033490055671917267 witness (3, (30, 33, 34), 0, -1)
rows 3 EMI 3.05 failures 0 / 2880 worst stress error 0.00033490055671917267 witness (3, (30, 33, 34), 0, -1)
rows 3 EMI 6.0 failures 0 / 2880 worst stress error 0.00033490055671917267 witness (3, (30, 33, 34), 0, -1)
rows 4 EMI 3.01 failures 0 / 6720 worst stress error 0.0003047860224332137 witness (3, (30, 31, 35, 37), 1, -1)
rows 4 EMI 3.05 failures 0 / 6720 worst stress error 0.0003047860224332137 witness (3, (30, 31, 35, 37), 1, -1)
rows 4 EMI 6.0 failures 0 / 6720 worst stress error 0.0003047860224332137 witness (3, (30, 31, 35, 37), 1, -1)
rows 5 EMI 3.05 failures 0 / 10080 worst stress error 0.0009688815337219218 witness (3, (30, 31, 32, 35, 38), 1, -1)
rows 10 EMI 3.05 failures 0 / 80 worst stress error 0.00011261301675296371 witness (3, (30, 31, 32, 33, 34, 35, 36, 37, 38, 39), 8, -1)
All five public outputs unchanged from the validated baseline.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=03d6ff603dbf6f6c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v05_03d6ff60 (directory).
```

### Turn 16 — `submit_completed_task` (`14879 ms` LLM)
- **Rationale**: The final artifact passed isolated validation. The calibration fix passed the previously failing sparse-EMI slices, preserved direct fits on clean sweeps, and left all five public outputs unchanged.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_03_solver.py`. The solver passed isolated schema validation and all 38,960 tested single-spike calibration cases. Against the clean public reference, strain MAE was 0.00768 and initial-rate MAE was 0.01076. Rows with imputed coupon features had the largest prediction errors.`

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
