# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c01_batch_01__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c01_step_01` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `368,184` (`355,463` / `105,600` / `12,721`) |
| **Total Episode Wall Time (s)** | `789.95s` |

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

### Turn 1 — `execute_in_remote_docker` (`17435 ms` LLM)
- **Rationale**: Inspect the public keys, shapes, and validator to catch schema mismatches before fitting. Inventory the archive and skill library so the implementation can use the prescribed references without copying bulk data.
- **Alternatives Weighed**: `Reuse the archived solver unchanged` (rejected: The prior branch failed the public gate, and the current batch requires a crystalline logarithmic plateau.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -l /tmp/hegel_continual_archive && python3 - <<'PY'
import pickle, numpy as np
with open('public_alpha_m97_c01_batch_01.pkl','rb') as f: d=pickle.load(f)
def show(x,p='',depth=0):
 if isinstance(x,dict):
  for k,v in x.items(): show(v,p+'/'+str(k),depth+1)
 elif isinstance(x,np.ndarray):
  print(p,x.shape,x.dtype,repr(x.reshape(-1)[:5]))
 elif isinstance(x,(list,tuple)):
  print(p,type(x).__name__,len(x),repr(x[0])[:500] if len(x) else '')
 else: print(p,repr(x)[:300])
show(d)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference-panel alignment imputation viscoelastic relaxation variable projection'
```
- **Container Exit Code**: `0` (`445 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
enous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
```

### Turn 2 — `execute_in_remote_docker` (`17216 ms` LLM)
- **Rationale**: The first inspection was truncated before the training labels and reference-panel layout. Inspect those fields and the archived preprocessing to distinguish row-aligned panels from separate controls and avoid fitting the wrong target or fallback calibration.

```bash
cd /holdout_task; ls -l /tmp/hegel_continual_archive; sed -n '1,220p' /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py; python3 - <<'PY'
import pickle,numpy as np,json
items=pickle.load(open('public_alpha_m97_c01_batch_01.pkl','rb')); item=items[0]; print('item',list(item)); d=item['dataset']; print('dataset',list(d));
for section in ('train','test','calibration_load_cells'):
 print(section)
 for k,v in d[section].items():
  a=np.asarray(v); print(k,a.shape,a.dtype, a.ravel()[:4], 'nan', np.isnan(a).sum() if a.dtype.kind in 'fi' else '-')
p='/tmp/hegel_continual_archive/archive_calibration_history.json'; h=json.load(open(p)); print('history',str(h)[:1100]);
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.weighted_column_fill
```
- **Container Exit Code**: `0` (`489 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`20714 ms` LLM)
- **Rationale**: Recover the archived preprocessing and training-target definitions omitted by the truncated output. Per-channel residuals determine whether to fit the clean batch directly or use the prescribed EMI fallback; cell counts and missingness determine how to apply the balanced fill.
- **Alternatives Weighed**: `Use the panel primitive's default fill directly` (rejected: The required cell-balanced fill and handling of missing reference columns must match the stream's established conventions.)

```bash
cd /holdout_task && sed -n '52,190p' /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py && python3 - <<'PY'
import json,pickle,numpy as np
with open('public_alpha_m97_c01_batch_01.pkl','rb') as f: items=pickle.load(f)
d=items[0]['dataset']; print('items',len(items),'dataset keys',list(d))
for split in ('train','test'):
 q=d[split]; print(split,[(k,np.shape(v)) for k,v in q.items()]); print('lab/stratum counts',np.unique(np.c_[q['lab_id'],q['stratum']],axis=0,return_counts=True)[1]); print('missing feature/ref',*[np.isnan(q[k]).sum(axis=0) for k in ('coupon_features_raw','ref_coupons')]);
 for k,v in q.items():
  if 'strain' in k or 'rate' in k: print(k,np.asarray(v).ravel()[:12])
c=d['calibration_load_cells'];
for lab in range(4):
 m=c['lab_id']==lab; x=c['cmd_stress'][m]; y=c['true_stress'][m]; b=np.polyfit(x,y,1); print('cal',lab,len(x),b,np.max(abs(y-np.polyval(b,x))))
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f: h=json.load(f)
print('archive structure',[(k,type(v).__name__, str(v)[:180]) for k,v in h.items()] if isinstance(h,dict) else str(h)[:500])
PY
```
- **Container Exit Code**: `0` (`391 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
reditary_weights=weights,
                regime_mode_weights=np.zeros(2), predicted_strain=test['strain_0'][:, None]+c@weights,
                initial_rate_pred=r@weights)
items 1 dataset keys ['past_dt', 'eval_times', 'crit_stress', 'yield_stress', 'shear_banding_index', 'calibration_load_cells', 'train', 'test']
train [('lab_id', (180,)), ('stratum', (180,)), ('coupon_features_raw', (180, 4)), ('ref_coupons', (180, 32)), ('past_stress_cmd', (180, 18)), ('stress_hold_cmd', (180,)), ('strain_0', (180,)), ('aux_chamber_telemetry', (180, 3)), ('strain_curves_obs', (180, 10)), ('instant_rate_obs', (180,))]
lab/stratum counts [13 10 13 12 23 15 13  7 24 19 13 18]
missing feature/ref [4 4 5 6] [ 3  4  6  3  4 12  5 10  9  9  7 11  5  7 12  4  4  3  8 11  6  3 16  7
  8  9  9  7  8 12  5 11]
strain_0 [-0.552188  0.406017 -0.145016 -0.423774  0.182002  0.009923 -0.103664
  0.686789  0.192855  0.47489  -0.125929 -0.220215]
strain_curves_obs [-0.30682153 -0.00513339  0.11082238  0.22859336  0.27228429  0.33917022
  0.37680614  0.41672633  0.44559225  0.45735521  0.23559912  0.12504199]
instant_rate_obs [ 1.11099833 -0.66347914 -0.3031002  -0.69647305 -1.55177841 -3.51394676
  1.71808258  0.22935342 -3.6850861  -1.27832215 -1.72895204  0.40633243]
test [('lab_id', (72,)), ('stratum', (72,)), ('coupon_features_raw', (72, 4)), ('ref_coupons', (72, 32)), ('past_stress_cmd', (72, 18)), ('stress_hold_cmd', (72,)), ('strain_0', (72,)), ('aux_chamber_telemetry', (72, 3))]
lab/stratum counts [8 7 5 5 9 4 7 9 5 4 6 3]
missing feature/ref [1 1 1 1] [6 1 1 2 1 6 3 5 5 2 7 1 4 3 0 1 5 4 3 1 3 3 2 4 2 5 1 2 3 2 3 5]
strain_0 [ 0.513179  0.68354  -0.21838   0.415848 -0.32671  -0.086465 -0.380754
  0.565439  0.11894   0.058164  0.173351  0.064834]
cal 0 10 [1.33999935 0.14999378] 0.00015647060919077305
cal 1 10 [ 0.77000406 -0.16999413] 0.00018822176288391113
cal 2 10 [ 1.21002713 -0.12000366] 0.0001037212625434325
cal 3 10 [0.83000992 0.10997266] 8.725097943051985e-05
archive structure [{'episode_short_name': 'alpha_m97_c01_s00', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [1.660283, -0.418768, 1.112723, -1.62604, -0.501169, -1.672067, 0.863389, 1.008755, 1.968856, 1.530083, 0.979468, 0.449112, 1.469886, -0.332491, -1.532413, 0.660575, -1.672999, -1.196407, 0.934354, 0.731875, 1.670914, -2.075727, 1.021513, -1.730233, -1.965428, 0.955855, 0.129182, -0.328527, -1.550117, -1.993
```

### Turn 4 — `execute_in_remote_docker` (`433900 ms` LLM)
- **Rationale**: Implement the specified response and its analytic initial-rate basis. Check the output schema and compare residuals for damaged and undamaged coupons; a systematic inactive-slice error would warrant revisiting the plateau gate before cross-validation.
- **Alternatives Weighed**: `Sort fitted relaxation times after optimization` (rejected: The crystalline terms distinguish branch zero, so swapping branches would change physical predictions.)

```bash
cat > /holdout_task/alpha_m97_c01_batch_01_solver.py <<'PY'
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _clean(cmd, true):
    if len(cmd) < 3:
        return np.ones(len(cmd), dtype=bool)
    i, j = np.triu_indices(len(cmd), 1)
    distinct = cmd[i] != cmd[j]
    slopes = (true[j[distinct]] - true[i[distinct]]) / (cmd[j[distinct]] - cmd[i[distinct]])
    offsets = true[:, None] - cmd[:, None] * slopes
    errors = np.abs(offsets - np.median(offsets, axis=0))
    return errors[:, np.argmin(np.median(errors, axis=0))] <= 3.0


def _calibration(cal, channel):
    m = cal['lab_id'] == channel
    cmd, true = cal['cmd_stress'][m], cal['true_stress'][m]
    clean = _clean(cmd, true)
    if len(cmd) < 2 or (not clean.all() and clean.sum() < 4):
        xs, ys = [], []
        for sweep in json.loads(_ARCHIVE.read_text()):
            take = np.asarray(sweep['lab_id']) == channel
            xs.extend(np.asarray(sweep['cmd_stress'])[take])
            ys.extend(np.asarray(sweep['true_stress'])[take])
        x, y = np.asarray(xs), np.asarray(ys)
        good = _clean(x, y)
        prior = np.polyfit(x[good], y[good], 1)
        if len(cmd) >= 2:
            clean = np.abs(true - np.polyval(prior, cmd)) <= 3.0
            if clean.sum() >= 2:
                return np.polyfit(cmd[clean], true[clean], 1)
        return prior
    return np.polyfit(cmd[clean], true[clean], 1)


def _prepare(d):
    train, test = d['train'], d['test']

    def join(k):
        return np.concatenate((train[k], test[k]), axis=0)

    lab, stratum = join('lab_id'), join('stratum')
    x, ref = join('coupon_features_raw'), join('ref_coupons')
    past, hold = join('past_stress_cmd'), join('stress_hold_cmd')
    cells = []
    for channel in np.unique(lab):
        m = lab == channel
        gain, bias = _calibration(d['calibration_load_cells'], channel)
        past[m] = gain * past[m] + bias
        hold[m] = gain * hold[m] + bias
        panel = ref[m]
        panel = panel[:, np.isfinite(panel).any(axis=0)]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[m] = (x[m] - center) / scale
        cells.extend(m & (stratum == level) for level in np.unique(stratum[m]))
    means = []
    for cell in cells:
        values = x[cell]
        count = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), count,
                               out=np.full(x.shape[1], np.nan), where=count > 0))
    balanced = np.nanmean(means, axis=0)
    for cell, mean in zip(cells, means):
        x[cell] = np.where(np.isnan(x[cell]), np.where(np.isnan(mean), balanced, mean), x[cell])
    dt = np.asarray(d['past_dt'])
    damage = np.maximum(np.abs(past) - d['crit_stress'], 0.)**2 @ dt
    plateau = (d['shear_banding_index'] <= .50) & (damage > 0.)
    act = np.where(plateau, np.tanh(damage), 0.)
    return past, hold, np.column_stack((np.ones(len(x)), x)), plateau, act


def _basis(logtau, dt, t, data):
    past, hold, x, plateau, act = data
    tau = np.exp(logtau)
    after = (np.cumsum(dt[::-1])[::-1] - dt)[:, None]
    kernel = np.exp(-after / tau) * -np.expm1(-dt[:, None] / tau)
    z = past @ kernel
    gap = hold[:, None] - z
    step = -np.expm1(-t[:, None] / tau)
    phi = np.where(plateau[:, None], np.log1p(t), t)
    stress_term, memory_term = act * hold, act * z[:, 0]
    curves = np.concatenate((gap[:, None, :] * step, phi[:, :, None] * x[:, None, :],
                             (stress_term[:, None] * step[:, 0])[:, :, None],
                             (memory_term[:, None] * phi)[:, :, None]), axis=2)
    rates = np.column_stack((gap / tau, x, stress_term / tau[0], memory_term))
    return curves, rates


def _fit(dt, t, data, yc, yr):
    yc, yr = yc.ravel(), yr.ravel()
    target = np.r_[yc, yr]

    def design(logtau):
        c, r = _basis(logtau, dt, t, data)
        return np.vstack((c.reshape(-1, 9), r))

    # Profile out the linear coefficients to find initial relaxation times;
    # refine jointly using a robust loss for contaminated strain/rate readings.
    scale = np.r_[np.full(yc.size, .04), np.full(yr.size, .05)]
    lo, hi = np.log([.025, 120.])

    def linear(logtau):
        a = design(logtau) / scale[:, None]
        w = np.linalg.lstsq(a, target / scale, rcond=None)[0]
        return w, a @ w - target / scale

    def obj(logtau):
        return np.sum(linear(logtau)[1]**2)

    def raw(p):
        return np.einsum('ij,j->i', design(p[:2]), p[2:]) - target

    def residual(p):
        return raw(p) / scale

    bounds = (np.r_[[lo, lo], np.full(9, -np.inf)], np.r_[[hi, hi], np.full(9, np.inf)])
    opts = dict(bounds=bounds, loss='cauchy', x_scale='jac', max_nfev=100)
    best = None
    for pair in ((.2, 2.), (.7, 5.), (1.5, 12.)):
        q = minimize(obj, np.log(pair), method='L-BFGS-B', bounds=[(lo, hi)]*2,
                     options={'maxiter': 45}).x
        # Branch zero is physically distinguished by both crystalline terms.
        p = np.r_[q, linear(q)[0]]
        fit = least_squares(residual, p, **opts)
        if best is None or fit.cost < best.cost:
            best = fit
    err = raw(best.x)
    for idx in (slice(0, yc.size), slice(yc.size, None)):
        e = err[idx]
        scale[idx] = max(1e-7, 2.5 * 1.4826 * np.median(np.abs(e - np.median(e))))
    p = least_squares(residual, best.x, **opts).x
    return p[:2], p[2:]


def solve_alpha_m97_c01_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    data = _prepare(dataset)
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    yc = train['strain_curves_obs'] - train['strain_0'][:, None]
    logtau, w = _fit(dt, t, tuple(a[:n] for a in data), yc, train['instant_rate_obs'])
    curves, rates = _basis(logtau, dt, t, tuple(a[n:] for a in data))
    return dict(estimated_taus=np.exp(logtau), hereditary_weights=w[:7],
                regime_mode_weights=w[7:], predicted_strain=test['strain_0'][:, None] + curves @ w,
                initial_rate_pred=rates @ w)
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np,time
import alpha_m97_c01_batch_01_solver as s
with open('public_alpha_m97_c01_batch_01.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
start=time.monotonic(); o=s.solve_alpha_m97_c01_batch_01(d); print('runtime',time.monotonic()-start)
for k in ('estimated_taus','hereditary_weights','regime_mode_weights'): print(k,o[k])
a=s._prepare(d); n=len(d['train']['lab_id']); tr=tuple(v[:n] for v in a); print('plateau counts',np.bincount(a[3].astype(int)), 'train activation quantiles',np.quantile(tr[4],[0,.1,.5,.9,1])); t=d['eval_times']; dt=d['past_dt']; C,R=s._basis(np.log(o['estimated_taus']),dt,t,tr); w=np.r_[o['hereditary_weights'],o['regime_mode_weights']]; yc=d['train']['strain_curves_obs']-d['train']['strain_0'][:,None]; yr=d['train']['instant_rate_obs'];
for label,mask in [('all',np.ones(n,bool)),('inactive',~tr[3]),('active',tr[3])]:
 if mask.any():
  for name,e in [('curve',(C@w-yc)[mask]),('rate',(R@w-yr)[mask])]: print(label,name,mask.sum(),np.quantile(abs(e),[.5,.9,.99,1]))
PY
```
- **Container Exit Code**: `0` (`2641 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
runtime 0.5759997081477195
estimated_taus [0.61509012 3.57106152]
hereditary_weights [ 0.68293038  0.54478508  0.13203607  0.15220025 -0.12274892  0.11167586
 -0.0859738 ]
regime_mode_weights [0.2769155 0.2556438]
plateau counts [ 84 168] train activation quantiles [0.         0.         0.99421417 0.99955568 0.99994875]
all curve 180 [ 0.02842067  0.50454832  2.3726564  23.02987785]
all rate 180 [0.0161226  0.04624459 0.19432433 0.2705711 ]
inactive curve 60 [ 0.19429381  0.90491503  2.41466023 15.78055873]
inactive rate 60 [0.01502852 0.04427504 0.10103866 0.1496055 ]
active curve 120 [1.89231585e-02 1.60682113e-01 1.92327637e+00 2.30298778e+01]
active rate 120 [0.01693691 0.04624459 0.19588693 0.2705711 ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5f04042ddfdce764 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5f04042d (directory).
```

### Turn 5 — `execute_in_remote_docker` (`28439 ms` LLM)
- **Rationale**: Compare gated, batch-wide logarithmic, and linear drift with the same crystalline correction and robust fitting. The per-lab, per-activation, and late-horizon errors show whether the inactive discrepancy calls for a different drift law or further investigation of the observations.
- **Alternatives Weighed**: `Treat the inactive residuals as measurement contamination` (rejected: A held-out-lab comparison can distinguish a systematic drift-law error from isolated strain outliers without changing the calibration or activation terms.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
import alpha_m97_c01_batch_01_solver as s
with open('public_alpha_m97_c01_batch_01.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
q=d['train']; n=len(q['lab_id']); a=tuple(v[:n] for v in s._prepare(d)); dt,t=d['past_dt'],d['eval_times']; y=q['strain_curves_obs']-q['strain_0'][:,None]; v=q['instant_rate_obs']; orig=s._basis
for mode in ('gated','global_log','global_linear'):
 def basis(logtau,dt,t,data):
  if mode=='gated': return orig(logtau,dt,t,data)
  p,h,x,g,b=data; return orig(logtau,dt,t,(p,h,x,np.full(len(p),mode=='global_log'),b))
 s._basis=basis
 P=np.zeros_like(y); V=np.zeros_like(v); ps=[]
 for lab in range(4):
  ix=q['lab_id']==lab; sub=lambda m:tuple(z[m] for z in a)
  l,w=s._fit(dt,t,sub(~ix),y[~ix],v[~ix]); C,R=basis(l,dt,t,sub(ix)); P[ix]=C@w; V[ix]=R@w; ps.append(np.r_[np.exp(l),w[7:]])
 print(mode,'fold parameters',np.round(ps,4)); e=P-y; er=V-v
 for lab in range(4):
  for active in (False,True):
   ix=(q['lab_id']==lab)&(a[3]==active); print(lab,active,ix.sum(),'curve |err| q50/80/95',np.round(np.quantile(np.abs(e[ix]),[.5,.8,.95]),4),'rate',np.round(np.quantile(np.abs(er[ix]),[.5,.8,.95]),4),'signed curve med at last 3',np.round(np.median(e[ix,-3:],axis=0),4))
 if mode=='gated':
  for active in (False,True):
   ix=a[3]==active; E=e[ix]; print('gated by horizon',active,'abs med',np.round(np.median(abs(E),axis=0),3),'bias',np.round(np.median(E,axis=0),3),'clean-row fractions',[(h,round(np.mean(np.median(abs(E),axis=1)<h),3)) for h in (.03,.08,.15,.35)])
 print('worst cell median',max(np.median(np.abs(e[(q['lab_id']==lab)&(a[3]==b)])) for lab in range(4) for b in (False,True)))
PY
```
- **Container Exit Code**: `0` (`2238 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ve |err| q50/80/95 [0.0167 0.1164 0.3266] rate [0.0152 0.0345 0.1883] signed curve med at last 3 [-0.0167  0.0053 -0.0092]
1 False 14 curve |err| q50/80/95 [0.0163 0.067  0.1645] rate [0.0183 0.0487 0.0669] signed curve med at last 3 [0.0102 0.005  0.0042]
1 True 36 curve |err| q50/80/95 [0.0146 0.0323 0.3635] rate [0.0184 0.0314 0.1307] signed curve med at last 3 [-0.0025 -0.008   0.0001]
2 False 16 curve |err| q50/80/95 [0.0106 0.0226 0.1821] rate [0.012  0.0237 0.0629] signed curve med at last 3 [-0.0025 -0.001  -0.0032]
2 True 28 curve |err| q50/80/95 [0.011  0.0278 0.7113] rate [0.0055 0.0238 0.0391] signed curve med at last 3 [-0.0022  0.0024 -0.0019]
3 False 13 curve |err| q50/80/95 [0.0097 0.0193 0.101 ] rate [0.0143 0.0293 0.0344] signed curve med at last 3 [ 0.0039  0.0036 -0.0047]
3 True 37 curve |err| q50/80/95 [0.0126 0.0271 0.2359] rate [0.0162 0.0235 0.0425] signed curve med at last 3 [-0.0033  0.0024  0.0032]
worst cell median 0.01669295396145254
global_linear fold parameters [[0.6338 4.4187 0.2937 0.093 ]
 [0.6231 4.3045 0.2857 0.0998]
 [0.6272 4.1129 0.2951 0.0989]
 [0.6263 4.3904 0.2859 0.1011]]
0 False 17 curve |err| q50/80/95 [0.043  0.1008 0.2951] rate [0.1083 0.2358 0.3792] signed curve med at last 3 [0.0054 0.0452 0.0589]
0 True 19 curve |err| q50/80/95 [0.0507 0.1418 0.3204] rate [0.1514 0.2149 0.3073] signed curve med at last 3 [-0.0276 -0.0084 -0.0281]
1 False 14 curve |err| q50/80/95 [0.0433 0.087  0.2069] rate [0.1153 0.163  0.1837] signed curve med at last 3 [0.0204 0.0103 0.0436]
1 True 36 curve |err| q50/80/95 [0.0564 0.1631 0.3619] rate [0.1555 0.3259 0.4853] signed curve med at last 3 [0.0074 0.0184 0.0405]
2 False 16 curve |err| q50/80/95 [0.0406 0.1116 0.2338] rate [0.1125 0.2744 0.3584] signed curve med at last 3 [0.0069 0.0472 0.0656]
2 True 28 curve |err| q50/80/95 [0.0543 0.1245 0.7145] rate [0.1279 0.2685 0.3478] signed curve med at last 3 [0.01   0.0505 0.069 ]
3 False 13 curve |err| q50/80/95 [0.037  0.0867 0.2026] rate [0.1032 0.2554 0.3663] signed curve med at last 3 [0.0205 0.0399 0.0624]
3 True 37 curve |err| q50/80/95 [0.0459 0.0972 0.2766] rate [0.1424 0.2394 0.2847] signed curve med at last 3 [-0.0027  0.0217  0.0539]
worst cell median 0.05638390757431133

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5f04042ddfdce764 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5f04042d (directory).
```

### Turn 6 — `execute_in_remote_docker` (`49178 ms` LLM)
- **Rationale**: Promote the drift law that removed the held-out bias. Verify the edited artifact in an isolated process: the direct recurrence detects branch or time-axis mistakes, an injected spike and a missing sweep exercise the calibration fallback, and the two-response held-out metrics detect regression after the edit.
- **Alternatives Weighed**: `Keep linear inactive drift and increase outlier resistance` (rejected: It cannot explain the same-sign late-horizon bias across all four held-out labs.)
- **ATMS NOGOOD DECLARED**: Refuted `Undamaged coupons in this batch exhibit linear terminal drift.` | Witness: `All four held-out labs had positive inactive-coupon late-horizon median residuals of 0.409–0.545 under linear inactive drift; using logarithmic drift reduced the worst lab-by-activation median absolute error from 0.2097 to 0.0167.` | Blanket Diagnosis: `The damage gate was applied to both the baseline drift law and the two damage-specific correction amplitudes, but the held-out data support gating only the corrections.` | Sublated Pivot: `Use log1p(t) for baseline drift throughout this batch; retain calibrated stress, both memory branches, and damage-gated crystalline corrections.`

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_batch_01_solver.py')
s = p.read_text()
s = s.replace('    return past, hold, np.column_stack((np.ones(len(x)), x)), plateau, act', '    return past, hold, np.column_stack((np.ones(len(x)), x)), act')
s = s.replace('    past, hold, x, plateau, act = data', '    past, hold, x, act = data')
s = s.replace('    phi = np.where(plateau[:, None], np.log1p(t), t)', '    # Held-out labs support logarithmic drift even at zero damage.\n    phi = np.log1p(t)\n    drift_terms = phi[None, :, None] * x[:, None, :]')
s = s.replace('gap[:, None, :] * step, phi[:, :, None] * x[:, None, :],', 'gap[:, None, :] * step, drift_terms,')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, pickle, time
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c01_batch_01_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('public_alpha_m97_c01_batch_01.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
q, te = d['train'], d['test']
n = len(q['lab_id'])
dt, t = d['past_dt'], d['eval_times']
start = time.monotonic()
o = s.solve_alpha_m97_c01_batch_01(d)
print('solve seconds', round(time.monotonic()-start,3), 'parameters', [o[k].tolist() for k in ('estimated_taus','hereditary_weights','regime_mode_weights')])
a = s._prepare(d)
w = np.r_[o['hereditary_weights'], o['regime_mode_weights']]
tau = o['estimated_taus']
# Check branch order, time-axis convention, gate, and rate against the direct recurrence.
p,h,x,act = [z[n:] for z in a]
mem = np.zeros((len(p), 2))
for j, step in enumerate(dt):
    decay = np.exp(-step/tau)
    mem = decay*mem + (1-decay)*p[:,j,None]
gap = h[:, None]-mem
amplitude = gap*w[:2]
amplitude[:,0] += w[7]*act*h
drift = x@w[2:7] + w[8]*act*mem[:,0]
np.testing.assert_allclose(o['predicted_strain'], te['strain_0'][:,None] + amplitude@(1-np.exp(-t[None,:]/tau[:,None])) + drift[:,None]*np.log1p(t), atol=1e-12)
np.testing.assert_allclose(o['initial_rate_pred'], amplitude@(1/tau)+drift, atol=1e-12)
# Stress actual calibration inputs with a transient and with a missing channel sweep.
cal = d['calibration_load_cells']
for j in range(4):
    idx = np.flatnonzero(cal['lab_id']==j)
    contaminated = copy.deepcopy(cal)
    contaminated['true_stress'][idx[0]] += 10.
    np.testing.assert_allclose(s._calibration(contaminated,j), s._calibration(cal,j), atol=1e-4)
    truncated = {k:v[np.arange(len(v))!=idx[0]] for k,v in cal.items()}
    keep = truncated['lab_id']!=j
    np.testing.assert_allclose(s._calibration({k:v[keep] for k,v in truncated.items()},j), s._calibration(cal,j), atol=5e-4)
# Held-out-lab metrics for both responses, with worst-regime and late-horizon slices.
y = q['strain_curves_obs']-q['strain_0'][:,None]
r = q['instant_rate_obs']
zz = tuple(z[:n] for z in a)
summary = []
for lab in np.unique(q['lab_id']):
    m = q['lab_id']==lab
    l, weights = s._fit(dt,t,tuple(z[~m] for z in zz),y[~m],r[~m])
    c,b = s._basis(l,dt,t,tuple(z[m] for z in zz))
    e, er = c@weights-y[m], b@weights-r[m]
    active = zz[-1][m]>0
    for state in (False,True):
        k = active==state
        summary.append([np.median(abs(e[k])), np.median(abs(er[k])), np.max(np.median(abs(e[k]),axis=0)), np.max(abs(np.median(e[k],axis=0)))])
        print('lab',lab,'active',state,'n',sum(k),'curve/rate median, worst-horizon median abs and signed:',np.round(summary[-1],5))
print('worst slices', np.max(summary,axis=0))
assert np.all(np.max(summary,axis=0) < [.025,.025,.05,.04])
print('recurrence, rate, calibration fallback and held-out-lab verification passed')
PY
```
- **Container Exit Code**: `1` (`1058 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
solve seconds 0.139 parameters [[0.613157637284054, 3.6876367523501], [0.6785424132869364, 0.5523036302657799, 0.1399203500479178, 0.1566477428033287, -0.12689315305682541, 0.11779808844189589, -0.08872415221951474], [0.2812735422475223, 0.24201832610150195]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f04ce10a47850cb1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_f04ce10a (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 38, in <module>
  File "/usr/local/lib/python3.12/dist-packages/numpy/testing/_private/utils.py", line 1504, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "/usr/lib/python3.12/contextlib.py", line 81, in inner
    return func(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/testing/_private/utils.py", line 797, in assert_array_compare
    raise AssertionError(msg)
AssertionError: 
Not equal to tolerance rtol=1e-07, atol=0.0005

Mismatched elements: 2 / 2 (100%)
Max absolute difference: 0.34002373
Max relative difference: 0.9998635
 x: array([9.999756e-01, 2.047400e-05])
 y: array([1.339999, 0.149994])
```

### Turn 7 — `execute_in_remote_docker` (`41035 ms` LLM)
- **Rationale**: The failed assertion used the current batch as the reference for archive fallback. Test fallback against an independent fit to the archived sweep, clean and EMI calibration against the relevant in-batch rows, and complete the interrupted held-out and direct-recurrence verification without changing the solver.
- **Alternatives Weighed**: `Change the solver to make sparse calibration match the current batch fit` (rejected: The archived sweep has a different calibration, and the contract calls for using that sweep when fewer than two in-batch points are available.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'alpha_m97_c01_batch_01_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('public_alpha_m97_c01_batch_01.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f:
    archive = json.load(f)
cal = d['calibration_load_cells']
for lab in range(4):
    mask = cal['lab_id'] == lab
    x, y = cal['cmd_stress'][mask], cal['true_stress'][mask]
    clean = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)[0]
    np.testing.assert_allclose(s._calibration(cal, lab), clean, atol=1e-10)
    spike = {k:v.copy() for k,v in cal.items()}
    j = np.flatnonzero(mask)[0]
    spike['true_stress'][j] += 10.
    keep = mask.copy(); keep[j] = False
    x, y = cal['cmd_stress'][keep], cal['true_stress'][keep]
    np.testing.assert_allclose(s._calibration(spike, lab), np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)[0], atol=1e-10)
    x = np.concatenate([np.asarray(a['cmd_stress'])[np.asarray(a['lab_id']) == lab] for a in archive])
    y = np.concatenate([np.asarray(a['true_stress'])[np.asarray(a['lab_id']) == lab] for a in archive])
    prior = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)[0]
    assert np.max(np.abs(y - np.c_[x,np.ones(len(x))] @ prior)) < 3.0
    for keep in (~mask, (~mask) | (np.arange(len(mask)) == j)):
        np.testing.assert_allclose(s._calibration({k:v[keep] for k,v in cal.items()}, lab), prior, atol=1e-10)
print('Clean, EMI, zero-row and one-row calibration checks passed')
q, te = d['train'], d['test']
n = len(q['lab_id'])
dt, t = d['past_dt'], d['eval_times']
a = s._prepare(d)
o = s.solve_alpha_m97_c01_batch_01(d)
tau = o['estimated_taus']
w = np.r_[o['hereditary_weights'],o['regime_mode_weights']]
p,h,x,act = [z[n:] for z in a]
mem = np.zeros((len(p),2))
for j,step in enumerate(dt):
    decay = np.exp(-step/tau)
    mem = decay*mem + (1-decay)*p[:,j,None]
amp = (h[:,None]-mem)*w[:2]
amp[:,0] += w[7]*act*h
drift = x@w[2:7] + w[8]*act*mem[:,0]
np.testing.assert_allclose(o['predicted_strain'], te['strain_0'][:,None] + amp@(-np.expm1(-t[None,:]/tau[:,None])) + drift[:,None]*np.log1p(t), atol=1e-12)
np.testing.assert_allclose(o['initial_rate_pred'], amp@(1/tau)+drift, atol=1e-12)
y = q['strain_curves_obs']-q['strain_0'][:,None]
r = q['instant_rate_obs']
base = tuple(z[:n] for z in a)
summary = []
for lab in np.unique(q['lab_id']):
    m = q['lab_id']==lab
    l,weights = s._fit(dt,t,tuple(z[~m] for z in base),y[~m],r[~m])
    c,b = s._basis(l,dt,t,tuple(z[m] for z in base))
    e,er = c@weights-y[m], b@weights-r[m]
    active = base[-1][m]>0
    for state in (False,True):
        k = active==state
        row = [np.median(abs(e[k])), np.median(abs(er[k])), np.max(np.median(abs(e[k]),axis=0)), np.max(abs(np.median(e[k],axis=0)))]
        summary.append(row)
        print('lab',lab,'active',state,'n',sum(k), 'metrics',np.round(row,5))
worst = np.max(summary,axis=0)
print('worst curve median AE, rate median AE, horizon median AE, horizon absolute bias:',worst)
assert np.all(worst < [.025,.025,.05,.04])
print('Direct recurrence, initial rate, and held-out-lab checks passed')
PY
```
- **Container Exit Code**: `0` (`2857 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Clean, EMI, zero-row and one-row calibration checks passed
lab 0 active False n 17 metrics [0.0128  0.00866 0.01999 0.01049]
lab 0 active True n 19 metrics [0.01669 0.01524 0.02361 0.01667]
lab 1 active False n 14 metrics [0.01629 0.01832 0.0277  0.01021]
lab 1 active True n 36 metrics [0.01456 0.01841 0.01982 0.00805]
lab 2 active False n 16 metrics [0.0106  0.012   0.01601 0.01043]
lab 2 active True n 28 metrics [0.01099 0.00546 0.01613 0.00454]
lab 3 active False n 13 metrics [0.00973 0.01435 0.01482 0.01248]
lab 3 active True n 37 metrics [0.01265 0.01621 0.01654 0.00548]
worst curve median AE, rate median AE, horizon median AE, horizon absolute bias: [0.01669295 0.01840714 0.02769815 0.01666551]
Direct recurrence, initial rate, and held-out-lab checks passed

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f04ce10a47850cb1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_f04ce10a (directory).
```

### Turn 8 — `submit_completed_task` (`13532 ms` LLM)
- **Rationale**: The final artifact passed isolated schema, calibration fallback, direct recurrence, initial-rate, and held-out-lab checks. The host receipt is bound to the same candidate.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_01_solver.py': 'FILE:6263'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_01_solver.py`. Across held-out labs and activation groups, the worst median absolute errors were 0.0167 for strain and 0.0184 for initial rate.`

### Turn 9 — `execute_in_remote_docker` (`37502 ms` LLM)
- **Rationale**: Pairwise sums and differences are already represented by the four linear drift features. Test omitted products and rational terms on real held-out labs, and inspect signed residuals for lab dependence. A material improvement in the worst slice would warrant a targeted follow-up; otherwise retain the verified artifact.
- **Alternatives Weighed**: `Fit relay modes, a Fokker–Planck/HJB operator, or a batch classifier` (rejected: The task specifies continuous strain and rate from a two-memory recurrence, with no relay state, stochastic drift operator, or classification label. Held-out lab residuals directly test the relevant nuisance effect.)

```bash
cd /holdout_task && cat > /tmp/rheology_batch01_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c01_batch_01_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m97_c01_batch_01.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
q = d['train']
lab = q['lab_id']
n = len(lab)
a = tuple(v[:n] for v in s._prepare(d))
dt, t = d['past_dt'], d['eval_times']
y = q['strain_curves_obs'] - q['strain_0'][:, None]
r = q['instant_rate_obs']
print('Dataset fields:', sorted(d), '; train fields:', sorted(q))
out = s.solve_alpha_m97_c01_batch_01(d)
assert all(np.isfinite(v).all() for v in out.values())
assert (out['estimated_taus'] > 0).all()

# A sum or difference of features is already in the linear drift span.
# Challenge that span with pairwise products, a three-way product basis, and
# a smooth rational hold-stress response, using only training labs for fits.
x = a[2][:, 1:]
h = a[1]
z = np.column_stack([x[:, i]*x[:, j] for i in range(4) for j in range(i+1, 4)] +
                    [x[:, i]*x[:, j]*x[:, k] for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)] +
                    [h/(1+h*h), h*h/(1+h*h)])
phi = np.log1p(t)
results = {'base': [], 'expanded': []}
for group in np.unique(lab):
    test = lab == group
    tr = ~test
    logtau, w = s._fit(dt, t, tuple(v[tr] for v in a), y[tr], r[tr])
    c, b = s._basis(logtau, dt, t, a)
    offsets = (c @ w - y, b @ w - r)
    scale = [max(1e-7, 2.5*1.4826*np.median(np.abs(e[tr]-np.median(e[tr])))) for e in offsets]
    ce = np.concatenate((c, phi[None, :, None]*z[:, None, :]), axis=-1)
    be = np.column_stack((b, z))
    design = np.vstack((ce[tr].reshape(-1, be.shape[1])/scale[0], be[tr]/scale[1]))
    target = np.r_[y[tr].ravel()/scale[0], r[tr]/scale[1]]
    ww = least_squares(lambda p: design@p-target, np.r_[w, np.zeros(z.shape[1])],
                       jac=lambda p: design.copy(), loss='cauchy', max_nfev=60).x
    for mode, (e, er) in [('base', offsets), ('expanded', (ce@ww-y, be@ww-r))]:
        for gate in (False, True):
            mask = test & ((a[-1] > 0) == gate)
            results[mode].append((np.median(abs(e[mask])), np.median(abs(er[mask])),
                                  np.max(np.median(abs(e[mask]), axis=0)),
                                  np.max(abs(np.median(e[mask], axis=0)))))
    print('held-out lab', group, 'signed strain/rate median residuals',
          [round(float(np.median(e[test])), 5) for e in offsets])
for mode, rows in results.items():
    print(mode, 'worst slice: strain AE, rate AE, horizon AE, horizon bias', np.round(np.max(rows, axis=0), 6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_batch01_audit.py && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`1891 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Dataset fields: ['calibration_load_cells', 'crit_stress', 'eval_times', 'past_dt', 'shear_banding_index', 'test', 'train', 'yield_stress'] ; train fields: ['aux_chamber_telemetry', 'coupon_features_raw', 'instant_rate_obs', 'lab_id', 'past_stress_cmd', 'ref_coupons', 'strain_0', 'strain_curves_obs', 'stratum', 'stress_hold_cmd']
held-out lab 0 signed strain/rate median residuals [-0.00385, -0.00622]
held-out lab 1 signed strain/rate median residuals [0.00076, 0.00121]
held-out lab 2 signed strain/rate median residuals [-9e-05, 0.00348]
held-out lab 3 signed strain/rate median residuals [0.00146, 0.00032]
base worst slice: strain AE, rate AE, horizon AE, horizon bias [0.016693 0.018407 0.027698 0.016666]
expanded worst slice: strain AE, rate AE, horizon AE, horizon bias [0.019601 0.019409 0.024816 0.018945]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f04ce10a47850cb1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_f04ce10a (directory).
```

### Turn 10 — `submit_completed_task` (`18063 ms` LLM)
- **Rationale**: The focused audit found no material reason to change the solver. The verified candidate already passed isolated execution, calibration stress tests, direct curve and rate checks, and held-out-lab validation.
- **Alternatives Weighed**: `Add nonlinear product and rational drift terms` (rejected: The added terms reduced the worst horizon median absolute error from 0.0277 to 0.0248, but worsened the worst strain and initial-rate errors and increased horizon bias.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_01_solver.py': 'FILE:6263'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_01_solver.py`. Across held-out labs and activation groups, the worst median absolute errors were 0.0167 for strain and 0.0184 for initial rate. The solver passed the public schema, calibration fallback, and direct recurrence checks.`

### Turn 11 — `execute_in_remote_docker` (`29443 ms` LLM)
- **Rationale**: Binary activation summaries can hide errors near the specified shear boundary or on imputed coupons. Slice the real held-out-lab residuals by those conditions and by stratum; a material concentration would identify whether to investigate the gate or panel fill. Leave the verified artifact unchanged during the audit.
- **Alternatives Weighed**: `Repeat the nonlinear basis audit` (rejected: The measured product and rational augmentation already worsened both worst-slice response medians. Relay modes and classification metrics do not apply to these measurements.)

```bash
cd /holdout_task && cat > /tmp/rheology_batch01_boundary_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c01_batch_01_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m97_c01_batch_01.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
q = d['train']
lab = q['lab_id']
n = len(lab)
a = tuple(v[:n] for v in s._prepare(d))
dt, t = d['past_dt'], d['eval_times']
y = q['strain_curves_obs'] - q['strain_0'][:, None]
r = q['instant_rate_obs']
e = np.empty_like(y)
er = np.empty_like(r)
for j in np.unique(lab):
    m = lab == j
    train = tuple(v[~m] for v in a)
    test = tuple(v[m] for v in a)
    p, w = s._fit(dt, t, train, y[~m], r[~m])
    c, b = s._basis(p, dt, t, test)
    e[m] = c @ w - y[m]
    er[m] = b @ w - r[m]
act = a[-1]
shear = np.asarray(d['shear_banding_index'])[:n]
missing = np.isnan(q['coupon_features_raw']).any(axis=1)
slices = {'no missing features': ~missing, 'imputed features': missing,
          'partial activation': (act > 0) & (act < .95), 'saturated activation': act >= .95,
          'near gate, lower side': (shear >= .4) & (shear <= .5),
          'near gate, upper side': (shear > .5) & (shear <= .6)}
slices.update({f'stratum {v}': q['stratum'] == v for v in np.unique(q['stratum'])})
for name, m in slices.items():
    if m.any():
        print(name, 'n=', m.sum(), 'curve/rate median abs, worst horizon median abs and bias=',
              np.round([np.median(abs(e[m])), np.median(abs(er[m])),
                        np.max(np.median(abs(e[m]), axis=0)),
                        np.max(abs(np.median(e[m], axis=0)) )], 5))
    else:
        print(name, 'n=0; unsupported by public labels')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_batch01_boundary_audit.py && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `1` (`1045 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f04ce10a47850cb1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_f04ce10a (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/rheology_batch01_boundary_audit.py", line 28, in <module>
    shear = np.asarray(d['shear_banding_index'])[:n]
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^
IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
```

### Turn 12 — `execute_in_remote_docker` (`16305 ms` LLM)
- **Rationale**: The audit incorrectly indexed a batch-wide scalar; the verified solver already handles it. Fix only the audit and report which slices the measurements support. If errors concentrate in imputed or partially activated coupons, investigate the corresponding panel fill or activation term; otherwise retain the verified artifact.

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/rheology_batch01_boundary_audit.py')
s = p.read_text().replace("shear = np.asarray(d['shear_banding_index'])[:n]", "shear_all = np.asarray(d['shear_banding_index'])\nshear = np.broadcast_to(shear_all, (n,)) if shear_all.ndim == 0 else shear_all[:n]\nif shear_all.ndim == 0:\n    print('Batch-wide shear index:', shear_all.item(), '; no within-batch shear contrast')")
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_batch01_boundary_audit.py
```
- **Container Exit Code**: `0` (`1039 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Batch-wide shear index: 0.12 ; no within-batch shear contrast
no missing features n= 161 curve/rate median abs, worst horizon median abs and bias= [0.0114  0.01366 0.01291 0.00375]
imputed features n= 19 curve/rate median abs, worst horizon median abs and bias= [0.08621 0.06998 0.14902 0.04288]
partial activation n= 1 curve/rate median abs, worst horizon median abs and bias= [0.01033 0.02655 0.02575 0.02575]
saturated activation n= 119 curve/rate median abs, worst horizon median abs and bias= [0.01355 0.01584 0.01647 0.00501]
near gate, lower side n=0; unsupported by public labels
near gate, upper side n=0; unsupported by public labels
stratum 0 n= 57 curve/rate median abs, worst horizon median abs and bias= [0.01352 0.01435 0.01845 0.00682]
stratum 1 n= 53 curve/rate median abs, worst horizon median abs and bias= [0.01259 0.01338 0.01654 0.00253]
stratum 2 n= 70 curve/rate median abs, worst horizon median abs and bias= [0.01264 0.01664 0.01498 0.00365]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f04ce10a47850cb1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_f04ce10a (directory).
```

### Turn 13 — `submit_completed_task` (`21339 ms` LLM)
- **Rationale**: The exact artifact passed isolated execution and the host-bound scientific checks. The final audit identified missing-feature coupons as the main limitation and found no measurements near the shear boundary; neither result warrants changing the verified solver.
- **Alternatives Weighed**: `Replace the prescribed cell-mean imputation` (rejected: The missing-feature slice has higher errors, but that comparison does not show that the required imputation is wrong or that a replacement would improve held-out predictions.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_01_solver.py`. It passed the public schema, calibration fallback, and direct recurrence checks. The worst held-out lab-by-activation median absolute errors were 0.0167 for strain and 0.0184 for initial rate. Coupons with missing features were the weakest slice: 0.0862 for strain and 0.0700 for initial rate.`

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
