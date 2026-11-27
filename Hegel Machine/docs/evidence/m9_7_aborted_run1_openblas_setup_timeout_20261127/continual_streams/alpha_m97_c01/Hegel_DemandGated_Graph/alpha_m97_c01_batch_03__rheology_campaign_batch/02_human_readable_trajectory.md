# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c01_batch_03__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c01_step_03` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `322,937` (`311,237` / `78,976` / `11,700`) |
| **Total Episode Wall Time (s)** | `597.78s` |

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

### Turn 1 — `execute_in_remote_docker` (`18910 ms` LLM)
- **Rationale**: Inspect field shapes and representative records to catch schema and split assumptions before implementing the fit. Look up applicable alignment and relaxation primitives without transferring raw data.
- **Alternatives Weighed**: `Reuse the prior solver unchanged` (rejected: The prior branch failed the public gate, and the current batch specifies a different terminal-flow basis.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && python3 - <<'PY'
import pickle
import numpy as np
with open('public_alpha_m97_c01_batch_03.pkl','rb') as f:
    d=pickle.load(f)
for k,v in d.items():
    print(k, type(v).__name__, getattr(v,'shape',None), getattr(v,'dtype',None))
    if isinstance(v,dict):
        print({a:(type(b).__name__,getattr(b,'shape',None)) for a,b in v.items()})
    elif isinstance(v,(list,tuple)):
        print('length',len(v),'sample',repr(v[:2])[:350])
    elif isinstance(v,np.ndarray):
        print('sample',repr(v[:2])[:350] if v.ndim else repr(v))
    else:
        print(repr(v)[:350])
PY
ls -l /tmp/hegel_continual_archive
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference-panel alignment imputation viscoelastic relaxation variable projection'
```
- **Container Exit Code**: `0` (`1431 ms`, timed_out=`False`)
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
  File "<stdin>", line 5, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`17195 ms` LLM)
- **Rationale**: Read the first record through the pickle's list envelope and inspect the prior alignment and calibration conventions. This detects field or reference-layout mismatches before they can bias the constitutive fit.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
items=pickle.load(open('public_alpha_m97_c01_batch_03.pkl','rb'))
print('items',len(items)); it=items[0]; print('item keys',list(it))
d=it['dataset']
for key,v in d.items():
 print(key)
 if isinstance(v,dict):
  for k,a in v.items():
   a=np.asarray(a); print(' ',k,a.shape,a.dtype,'nan',np.isnan(a).sum() if a.dtype.kind in 'fc' else '-',repr(a[:2])[:160] if a.ndim else repr(a.item())[:160])
 else: print(repr(v)[:500])
PY
sed -n '1,235p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_02_solver.py
head -85 /tmp/hegel_continual_archive/archive_calibration_history.json
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.weighted_column_fill
```
- **Container Exit Code**: `0` (`616 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`21497 ms` LLM)
- **Rationale**: Check whether any calibration channel needs archive fallback and whether each lab–stratum cell supports imputation. Read only the prior preprocessing code and compact summaries; leave evaluation labels out of model fitting.
- **Alternatives Weighed**: `Reuse the gate diagnostic's calibration coefficients` (rejected: The specification requires fitting directly from the current batch when it supplies enough clean points.)

```bash
sed -n '1,125p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_02_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_calibration_history.json'; a=json.load(open(p)); print('archive format',str(a)[:1400])
d=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_03.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']
for lab in range(4):
 s=c['lab_id']==lab; x=c['cmd_stress'][s]; y=c['true_stress'][s]; b=np.linalg.lstsq(np.c_[x,np.ones(len(x))],y,rcond=None)[0]; print('calibration',lab,len(x),b,'max deviation',max(abs(y-(b[0]*x+b[1]))))
for split in ('train','test'):
 q=d[split]; print(split,'fields',list(q),'counts by lab and stratum',np.unique(np.c_[q['lab_id'],q['stratum']],axis=0,return_counts=True)); print('rows with missing coupon',np.isnan(q['coupon_features_raw']).any(axis=1).sum())
PY
```
- **Container Exit Code**: `0` (`634 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [1.660283, -0.418768, 1.112723, -1.62604, -0.501169, -1.672067, 0.863389, 1.008755, 1.968856, 1.530083, 0.979468, 0.449112, 1.469886, -0.332491, -1.532413, 0.660575, -1.672999, -1.196407, 0.934354, 0.731875, 1.670914, -2.075727, 1.021513, -1.730233, -1.965428, 0.955855, 0.129182, -0.328527, -1.550117, -1.993809, 2.131528, -2.023748, 0.667305, -0.968757, -0.649251, -0.022422, -1.62562, -0.902597, 1.990079, -0.41266], 'true_stress': [1.660305, -0.418669, 1.112724, -1.626001, -0.501125, -1.671976, 0.863198, 1.008634, 1.968816, 1.530248, 0.979291, 0.449215, 1.470001, -0.332446, -1.532394, 0.660518, -1.672977, -1.196416, 0.934345, 0.731869, 1.670821, -2.075755, 1.021515, -1.730423, -1.965592, 0.95586, 0.129076, -0.328478, -1.550116, -1.993815, 2.131554, -2.023643, 0.667296, -0.968674, -0.649069, -0.022271, -1.625842, -0.902458, 1.990138, -0.412675]}, {'episode_short_name': 'alpha_m97_c01_batch_01', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [-1.457421, -1.418035, 1.887501, -2.004396, 1.699178, -1.899037, 1.288097, 1.856886, 1.825204, 1.801222, -0.795406, 1.120198, -1.084462, 0.640209, 1.524142,
calibration 0 10 [1.33998908 0.15000143] max deviation 0.00021677044913309684
calibration 1 10 [ 0.77003896 -0.17002605] max deviation 0.00014201764236365655
calibration 2 10 [ 1.20998093 -0.11999076] max deviation 0.0001616162574000768
calibration 3 10 [0.83000147 0.11001181] max deviation 0.0001720959915644249
train fields ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0', 'aux_chamber_telemetry', 'strain_curves_obs', 'instant_rate_obs'] counts by lab and stratum (array([[0, 0],
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
       [3, 2]]), array([12, 13, 10, 12, 25, 12, 16, 15, 26,  8, 12, 19]))
rows with missing coupon 23
test fields ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0', 'aux_chamber_telemetry'] counts by lab and stratum (array([[0, 0],
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
       [3, 2]]), array([ 5,  7,  5,  2, 10,  4,  6,  3, 12,  6,  8,  4]))
rows with missing coupon 7
```

### Turn 4 — `execute_in_remote_docker` (`60967 ms` LLM)
- **Rationale**: Write a first working solver using in-batch calibration, panel alignment, and a robust joint fit. The isolated schema check catches import, shape, and finiteness failures before lab-blocked scientific validation.
- **Alternatives Weighed**: `Fit the onset rate separately` (rejected: The rate is the analytical derivative of the same strain law; a separate fit would break that constraint.)

```bash
sed -n '65,107p' /tmp/hegel_continual_archive/alpha_m97_c01_batch_02_solver.py
cat > /holdout_task/alpha_m97_c01_batch_03_solver.py <<'PY'
"""Fit the calibrated two-branch jammed-slip rheology law."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _historical_line(lab):
    for sweep in reversed(json.loads(_ARCHIVE.read_text())):
        m = np.asarray(sweep['lab_id']) == lab
        x = np.asarray(sweep['cmd_stress'])[m]
        y = np.asarray(sweep['true_stress'])[m]
        if len(x) >= 2 and np.ptp(x) > 0:
            return np.polyfit(x, y, 1)
    raise ValueError(f'No calibration line for lab {lab}')


def _calibrate(cal):
    lab, x, y = (np.asarray(cal[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    lines = []
    for c in range(4):
        m = (lab == c) & np.isfinite(x) & np.isfinite(y)
        u, v = x[m], y[m]
        if len(u) >= 4 and np.ptp(u) > 0:
            slopes = []
            for ui, vi in zip(u, v):
                different = u != ui
                slopes.append(np.median((v[different] - vi) / (u[different] - ui)))
            g = np.median(slopes)
            b = np.median(v - g*u)
            keep = np.abs(v - (g*u+b)) <= 3.0
            if keep.sum() >= 4:
                lines.append(np.polyfit(u[keep], v[keep], 1))
                continue
        if len(u) >= 2 and np.ptp(u) > 0:
            g, b = np.polyfit(u, v, 1)
            old = _historical_line(c) if len(u) >= 4 else None
            lines.append((g, b) if old is None else old)
        else:
            lines.append(_historical_line(c))
    return np.asarray(lines)


def _prepare(dataset):
    a, b = dataset['train'], dataset['test']
    def joined(key):
        return np.concatenate((a[key], b[key]), axis=0)
    lab = joined('lab_id')
    panel = joined('ref_coupons')
    features = joined('coupon_features_raw').astype(float)
    for c in range(4):
        m = lab == c
        r = panel[m]
        features[m] = (features[m] - np.nanmedian(np.nanmean(r, axis=0))) / np.nanmedian(np.nanstd(r, axis=0))
    _, cell, counts = np.unique(np.c_[lab, joined('stratum')], axis=0, return_inverse=True, return_counts=True)
    weights = (1.0/counts[cell])[:, None]
    good = np.isfinite(features)
    fill = np.sum(np.where(good, features, 0.)*weights, axis=0) / np.sum(good*weights, axis=0)
    features = np.where(good, features, fill)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*joined('past_stress_cmd') + bias[:, None]
    hold = gain*joined('stress_hold_cmd') + bias
    return past, hold, features


def _design(data, dt, t, tau):
    past, hold, x = data
    memory = np.zeros((len(hold), 2))
    for step, stress in zip(dt, past.T):
        u = -step/tau
        memory = np.exp(u)*memory - np.expm1(u)*stress[:, None]
    gap = hold[:, None] - memory
    prony = -np.expm1(-t[:, None]/tau)
    phi = t/(1.0 + 0.85*t)
    slip = np.sign(hold)*np.sqrt(np.abs(hold))
    flow = np.column_stack((np.ones(len(hold)), x, slip))
    contrast = gap[:, 0] - gap[:, 1]
    curves = np.concatenate((gap[:, None, :]*prony,
                             flow[:, None, :]*phi[None, :, None],
                             (contrast[:, None]*prony[:, 0])[:, :, None]), axis=2)
    rates = np.column_stack((gap/tau, flow, contrast/tau[0]))
    return curves, rates


def _fit(data, dt, t, curves, rates):
    cm, rm = np.isfinite(curves), np.isfinite(rates)
    cy, ry = curves[cm], rates[rm]
    y = np.r_[cy, ry]
    def design(log_tau):
        c, r = _design(data, dt, t, np.exp(log_tau))
        return np.vstack((c[cm], r[rm]))
    def profile(log_tau, scale):
        a = design(log_tau)
        w = np.linalg.lstsq(a/scale[:, None], y/scale, rcond=None)[0]
        return (a@w-y)/scale, w
    def mad(r):
        return max(1e-5, 1.4826*np.median(np.abs(r-np.median(r))))
    lo = np.log(min(dt)*0.12)
    hi = np.log(max(t[-1], sum(dt))*12)
    grid = np.geomspace(min(dt)*0.7, max(t[-1], sum(dt))*1.25, 8)
    seeds = [np.log([a, b]) for i, a in enumerate(grid) for b in grid[i+1:]]
    scale = np.r_[np.ones(len(cy)), np.full(len(ry), 2.)]
    def score(seed):
        r, _ = profile(seed, scale)
        return np.dot(r, r)
    seeds.sort(key=score)
    pilot = least_squares(lambda z: profile(z, scale)[0], seeds[0], bounds=(lo, hi))
    err, _ = profile(pilot.x, np.ones_like(y))
    scale = np.r_[np.full(len(cy), mad(err[:len(cy)])), np.full(len(ry), mad(err[len(cy):]))]
    winners = []
    lower = np.r_[[lo, lo], np.full(9, -np.inf)]
    upper = np.r_[[hi, hi], np.full(9, np.inf)]
    for seed in [pilot.x]+seeds[:4]:
        _, w = profile(seed, scale)
        trial = least_squares(lambda z: (design(z[:2])@z[2:]-y)/scale,
                              np.r_[seed, w], bounds=(lower, upper),
                              loss='soft_l1', f_scale=1.3, max_nfev=100)
        if trial.x[0] < trial.x[1]:
            winners.append(trial)
    if not winners:
        return np.exp(seeds[0]), profile(seeds[0], scale)[1]
    fit = min(winners, key=lambda trial: trial.cost).x
    return np.exp(fit[:2]), fit[2:]


def solve_alpha_m97_c01_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    data = _prepare(dataset)
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    tau, w = _fit(tuple(v[:n] for v in data), dt, t,
                  train['strain_curves_obs'] - train['strain_0'][:, None],
                  train['instant_rate_obs'])
    curves, rates = _design(tuple(v[n:] for v in data), dt, t, tau)
    return {'estimated_taus': tau, 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:],
            'predicted_strain': test['strain_0'][:, None] + curves@w,
            'initial_rate_pred': rates@w}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`795 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_03_solver.py']`
- **Stdout (tail)**:
```text
train_size = len(train['lab_id'])
    def join(key):
        return np.concatenate((train[key], test[key]), axis=0)
    lab = join('lab_id')
    strata = join('stratum')
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*join('past_stress_cmd') + bias[:, None]
    hold = gain*join('stress_hold_cmd') + bias
    x = join('coupon_features_raw')
    ref = join('ref_coupons')
    for channel in np.unique(lab):
        sel = lab == channel
        panel = ref[sel]
        eligible = np.isfinite(panel).sum(axis=0) >= 2
        panel = panel[:, eligible]
        loc = np.median(np.nanmean(panel, axis=0))
        std = np.median(np.nanstd(panel, axis=0))
        x[sel] = (x[sel] - loc)/std
    # Each cell with an observation contributes one mean per feature.
    cells, cell = np.unique(np.column_stack((lab, strata)), axis=0, return_inverse=True)
    valid = np.isfinite(x)
    sums = np.zeros((len(cells), x.shape[1]))
    counts = np.zeros_like(sums)
    np.add.at(sums, cell, np.where(valid, x, 0.))
    np.add.at(counts, cell, valid)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    observed_cells = np.count_nonzero(counts, axis=0)
    fill = np.divide(means.sum(axis=0), observed_cells,
                     out=np.zeros(x.shape[1]), where=observed_cells > 0)
    x = np.where(valid, x, fill)
    return (past, hold, x), train_size


def _design(logtau, data, dt, time):
    past, hold, x = data
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for width, stress in zip(dt, past.T):
        u = width/tau
        a = np.exp(-u)
        dz = a*dz + (u*a)*(z-stress[:, None])
        z = a*z - np.expm1(-u)*stress[:, None]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2fb92834239d46cd (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2fb92834 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`38168 ms` LLM)
- **Rationale**: Correct the imputation rule and calibration fallback, then hold out each lab's response measurements to detect unstable relaxation times or systematic errors by lab, horizon, stress sign, and coupon missingness. Re-run the isolated schema validator on the edited artifact.
- **Alternatives Weighed**: `Keep fixed inverse cell-size row weights for imputation` (rejected: With missing entries, they underweight cells that have lower observation rates for a given feature; averaging the observed cell means gives each contributing cell equal weight.)

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p=Path('alpha_m97_c01_batch_03_solver.py')
s=p.read_text()
a=s.index('        if len(u) >= 4')
b=s.index('    return np.asarray(lines)',a)
s=s[:a]+'''        if len(u) >= 3 and np.ptp(u) > 0:
            slopes = []
            for ui, vi in zip(u, v):
                different = u != ui
                slopes.append(np.median((v[different] - vi) / (u[different] - ui)))
            g = np.median(slopes)
            b = np.median(v - g*u)
            keep = np.abs(v - (g*u+b)) <= 3.0
            if not np.all(keep):
                u, v = u[keep], v[keep]
                if len(u) < 4:
                    lines.append(_historical_line(c))
                    continue
        if len(u) >= 2 and np.ptp(u) > 0:
            lines.append(np.polyfit(u, v, 1))
        else:
            lines.append(_historical_line(c))
'''+s[b:]
a=s.index('        r = panel[m]')
b=s.index('    features = np.where',a)
s=s[:a]+'''        r = panel[m]
        r = r[:, np.isfinite(r).sum(axis=0) >= 2]
        features[m] = (features[m] - np.median(np.nanmean(r, axis=0))) / np.median(np.nanstd(r, axis=0))
    cells, cell = np.unique(np.c_[lab, joined('stratum')], axis=0, return_inverse=True)
    good = np.isfinite(features)
    sums = np.zeros((len(cells), features.shape[1]))
    counts = np.zeros_like(sums)
    np.add.at(sums, cell, np.where(good, features, 0.))
    np.add.at(counts, cell, good)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    fill = means.sum(axis=0) / (counts > 0).sum(axis=0)
'''+s[b:]
p.write_text(s)
PY
cat > /tmp/rheology_cv.py <<'PY'
import importlib.util,pickle,numpy as np
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',root+'alpha_m97_c01_batch_03_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open(root+'public_alpha_m97_c01_batch_03.pkl','rb'))[0]['dataset']; tr=d['train']; n=len(tr['lab_id']); data=tuple(x[:n] for x in m._prepare(d)); dt=d['past_dt']; t=d['eval_times']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; pc=np.empty_like(y); pr=np.empty_like(r)
for lab in range(4):
    sel=tr['lab_id']==lab
    tau, w=m._fit(tuple(x[~sel] for x in data),dt,t,y[~sel],r[~sel]); c,b=m._design(tuple(x[sel] for x in data),dt,t,tau); pc[sel]=c@w; pr[sel]=b@w
    print('lab',lab,'tau',tau,'curve/rate RMSE',np.mean((pc[sel]-y[sel])**2)**.5,np.mean((pr[sel]-r[sel])**2)**.5,'MAE',np.mean(abs(pc[sel]-y[sel])),np.mean(abs(pr[sel]-r[sel])))
print('horizon RMSE',np.mean((pc-y)**2,axis=0)**.5)
for name,sel in [('negative hold',data[1]<0),('positive hold',data[1]>=0),('missing coupon',np.isnan(tr['coupon_features_raw']).any(axis=1)),('above critical stress',np.abs(data[1])>d['crit_stress'])]:
    print(name,sel.sum(),'curve/rate MAE',np.mean(abs(pc[sel]-y[sel])),np.mean(abs(pr[sel]-r[sel])))
out=m.solve_alpha_m97_c01_batch_03(d)
for k in ('estimated_taus','hereditary_weights','regime_mode_weights'): print(k,out[k])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_cv.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1803 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_03_solver.py']`
- **Stdout (tail)**:
```text
lab 0 tau [0.5909745  3.85798221] curve/rate RMSE 0.874896589711355 0.05133704233046753 MAE 0.1225450927535344 0.029941337557536998
lab 1 tau [0.59601438 3.91767878] curve/rate RMSE 0.4177068939592096 0.03469380206898395 MAE 0.07292691252739035 0.022376097927062927
lab 2 tau [0.59178469 3.8864754 ] curve/rate RMSE 0.5471090315213231 0.06353246870759122 MAE 0.07722653744720469 0.02959517348855541
lab 3 tau [0.59359509 3.81037379] curve/rate RMSE 0.1715226715162246 0.044296916315837376 MAE 0.053841611461003025 0.030157269137873657
horizon RMSE [0.57216733 0.45790134 0.26970219 0.17481507 0.11267654 0.45879814
 0.1446021  0.18218912 0.89221297 1.12305986]
negative hold 87 curve/rate MAE 0.07713890750997074 0.02591517356952039
positive hold 93 curve/rate MAE 0.08229190847653939 0.029600149436734793
missing coupon 23 curve/rate MAE 0.1950675012139375 0.09804155637198367
above critical stress 0 curve/rate MAE nan nan
estimated_taus [0.59319618 3.8711895 ]
hereditary_weights [ 0.67263753  0.54660221  0.13821289  0.15417048 -0.12886609  0.11598556
 -0.08768944]
regime_mode_weights [ 0.32166984 -0.21244914]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ca7ea1c95ef61c17 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ca7ea1c9 (directory).
```
- **Stderr (tail)**:
```text
/usr/local/lib/python3.12/dist-packages/numpy/core/fromnumeric.py:3504: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/usr/local/lib/python3.12/dist-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
```

### Turn 6 — `execute_in_remote_docker` (`41911 ms` LLM)
- **Rationale**: Test sparse and EMI calibration against the prescribed fallback and clean fit; fix the detector if a spike changes the fitted line. Check the recurrence against a closed-form convolution and the rate against a numerical derivative; any mismatch identifies which design term needs correction. An independent noiseless fit checks optimizer recovery before submission.
- **Alternatives Weighed**: `Tune against the largest raw curve RMSE` (rejected: The gap between RMSE and MAE points to a heavy residual tail. First verify that the solver recovers known parameters and satisfies the physical law independently of that tail.)

```bash
cat > /tmp/rheology_contract.py <<'PY'
import copy, importlib.util, pickle
from time import perf_counter
import numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c01_batch_03_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d = pickle.load(open(root+'public_alpha_m97_c01_batch_03.pkl','rb'))[0]['dataset']
cal = d['calibration_load_cells']; nominal = m._calibrate(cal)
for c in range(4):
    idx = np.flatnonzero(cal['lab_id'] == c)
    keep = (cal['lab_id'] != c) | (np.arange(len(cal['lab_id'])) == idx[0])
    sparse = m._calibrate({k: v[keep] for k,v in cal.items()})
    np.testing.assert_allclose(sparse[c], m._historical_line(c), atol=1e-12)
    np.testing.assert_allclose(sparse[np.arange(4) != c], nominal[np.arange(4) != c], atol=1e-12)
    worst = 0.
    for i in idx:
        for spike in (-12., -3.01, 3.01, 12.):
            noisy = {k: v.copy() for k,v in cal.items()}; noisy['true_stress'][i] += spike
            worst = max(worst, float(np.max(np.abs(m._calibrate(noisy)-nominal))))
    print('channel', c, 'sparse archive versus full current max coefficient delta', np.max(abs(sparse[c]-nominal[c])), 'worst EMI coefficient error', worst)
    assert worst < 0.001

def forward(data, dt, times, tau, w):
    past, hold, x = data
    ends = np.cumsum(dt)
    age = ends[-1] - ends
    kernel = np.exp(-age[:,None]/tau) - np.exp(-(age+dt)[:,None]/tau)
    g = hold[:,None] - past@kernel
    drift = w[2] + x@w[3:7] + w[7]*np.sign(hold)*np.sqrt(abs(hold))
    f = 1 - np.exp(-times[:,None]/tau)
    return (g*w[:2])@f.T + drift[:,None]*times/(1+0.85*times) + w[8]*(g[:,0]-g[:,1])[:,None]*f[:,0]

start = perf_counter()
out = m.solve_alpha_m97_c01_batch_03(d)
print('default-process solve seconds', perf_counter()-start)
n = len(d['train']['lab_id']); data = m._prepare(d); dt = d['past_dt']; t = d['eval_times']
test = tuple(v[n:] for v in data)
tau = out['estimated_taus']; w = np.r_[out['hereditary_weights'],out['regime_mode_weights']]
np.testing.assert_allclose(out['predicted_strain'] - d['test']['strain_0'][:,None], forward(test,dt,t,tau,w), atol=2e-12)
h = 1e-5
v = forward(test,dt,np.array([-h,h]),tau,w); rate = (v[:,1]-v[:,0])/(2*h)
print('max independent onset derivative error', np.max(abs(rate-out['initial_rate_pred'])))
np.testing.assert_allclose(rate, out['initial_rate_pred'], atol=2e-8)
clean = copy.deepcopy(d); tau0 = np.array([0.52, 4.6]); w0 = np.array([0.72,0.46,0.18,0.13,-0.16,0.08,-0.1,0.28,-0.18]); train = tuple(v[:n] for v in data)
clean['train']['strain_curves_obs'] = d['train']['strain_0'][:,None] + forward(train,dt,t,tau0,w0)
v = forward(train,dt,np.array([-h,h]),tau0,w0)
clean['train']['instant_rate_obs'] = (v[:,1]-v[:,0])/(2*h)
ret = m.solve_alpha_m97_c01_batch_03(clean)
print('noiseless max tau/weight errors', np.max(abs(ret['estimated_taus']-tau0)), np.max(abs(np.r_[ret['hereditary_weights'],ret['regime_mode_weights']]-w0)))
np.testing.assert_allclose(ret['estimated_taus'],tau0,atol=1e-5)
np.testing.assert_allclose(np.r_[ret['hereditary_weights'],ret['regime_mode_weights']],w0,atol=1e-5)
PY
python3 -I /tmp/rheology_contract.py && python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`146238 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
channel 0 sparse archive versus full current max coefficient delta 4.078998669498901e-05 worst EMI coefficient error 3.818153686457881e-05
channel 1 sparse archive versus full current max coefficient delta 4.8948314750635324e-05 worst EMI coefficient error 1.8460903046024768e-05
channel 2 sparse archive versus full current max coefficient delta 2.218728711822493e-05 worst EMI coefficient error 1.9925208780274817e-05
channel 3 sparse archive versus full current max coefficient delta 7.735259867341915e-05 worst EMI coefficient error 2.4296443576946047e-05
default-process solve seconds 15.819631479913369
max independent onset derivative error 1.6498713506507556e-10
noiseless max tau/weight errors 4.956213217610639e-10 3.0666635897347305e-11
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ca7ea1c95ef61c17 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ca7ea1c9 (directory).
```

### Turn 7 — `submit_completed_task` (`16365 ms` LLM)
- **Rationale**: The exact submitted artifact passed isolated schema validation and stress checks on all four calibration channels. An independent convolution matched the strain predictions, the onset derivative agreed within 1.7e-10, and noiseless relaxation times and weights were recovered within 5e-10.
- **Alternatives Weighed**: `Add a separate model for rows with missing coupons` (rejected: That would depart from the prescribed cell-balanced imputation without evidence that a separate model generalizes.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_03_solver.py': 'FILE:5952'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_03_solver.py`. The solver passed the public schema, sparse-calibration, EMI-spike, and noiseless-recovery checks. Missing coupons were the hardest validation slice: strain MAE was 0.195 and onset-rate MAE was 0.098.`

### Turn 8 — `execute_in_remote_docker` (`27811 ms` LLM)
- **Rationale**: The independent convolution and derivative tests already verify the rational time factor and pairwise gap difference. Use real held-out residuals to distinguish cell-level bias from isolated measurement spikes. Combine allowed sparse sweeps and EMI spikes to test whether the fallback can detect contamination that a full sweep would reject.
- **Alternatives Weighed**: `Audit classification balance, relay modes, or Fokker–Planck drift` (rejected: This task has continuous strain and rate measurements, no relay state, and a specified deterministic memory recurrence. Signed errors by lab and stratum are the relevant confounding check.)

```bash
sed '/out=m.solve/,$d' /tmp/rheology_cv.py | sed 's/    print(name,sel.sum()/    if sel.any(): print(name,sel.sum()/' > /tmp/rheology_audit.py
cat >> /tmp/rheology_audit.py <<'PY'
e=pc-y; f=pr-r
for lab in range(4):
    for stratum in np.unique(tr['stratum']):
        s=(tr['lab_id']==lab)&(tr['stratum']==stratum)
        if s.any():
            print('cell',lab,stratum,'n',s.sum(),'median signed curve/rate error',np.median(e[s]),np.median(f[s]),'median absolute',np.median(abs(e[s])),np.median(abs(f[s])))
print('absolute curve residual quantiles',np.quantile(abs(e),[.5,.9,.95,.99,.995,1]))
print('absolute rate residual quantiles',np.quantile(abs(f),[.5,.9,.95,.99,1]))
print('curve residuals above 0.5 by time',np.sum(abs(e)>.5,axis=0),'affected coupons',np.sum(np.any(abs(e)>.5,axis=1)))
cal=d['calibration_load_cells']; base=m._calibrate(cal)
for lab in range(4):
    u=np.flatnonzero(cal['lab_id']==lab); z=np.argsort(cal['cmd_stress'][u]); picks=u[z[[0,len(z)//2,-1]]]; s=(cal['lab_id']!=lab)|np.isin(np.arange(len(cal['lab_id'])),picks)
    err=[]
    for i in picks:
        q={k:v[s].copy() for k,v in cal.items()}; q['true_stress'][np.flatnonzero(np.flatnonzero(s)==i)[0]]+=4.; err.append(np.max(abs(m._calibrate(q)[lab]-base[lab])))
    print('3-point sweep, one EMI spike, max errors',lab,err)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_audit.py
```
- **Container Exit Code**: `0` (`1193 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
702564 median absolute 0.011674197770208936 0.012954729064346582
cell 0 2 n 10 median signed curve/rate error 0.003977371420463116 0.0021834258309189614 median absolute 0.016883654244807977 0.010492855443639315
cell 1 0 n 12 median signed curve/rate error 0.0007564613485394611 0.014582264865730388 median absolute 0.015687997399939313 0.026224027825740626
cell 1 1 n 25 median signed curve/rate error -0.0011452731296411611 -0.000868384965658775 median absolute 0.011414055890517916 0.01857234249137618
cell 1 2 n 12 median signed curve/rate error 0.003058642488519947 -0.002749987333269477 median absolute 0.013350132700738593 0.009069269958769666
cell 2 0 n 16 median signed curve/rate error 0.001568769160839506 0.003521632942241948 median absolute 0.0134020804787493 0.010034254935993858
cell 2 1 n 15 median signed curve/rate error 0.0006005199960073537 0.004092553036445928 median absolute 0.01325578271120098 0.016963618824526572
cell 2 2 n 26 median signed curve/rate error -0.0019775580692394934 -0.00437046607728292 median absolute 0.01425765293827247 0.01870487390586184
cell 3 0 n 8 median signed curve/rate error -0.00840339119572564 -0.026091735699581342 median absolute 0.016620063733655988 0.03214875120080632
cell 3 1 n 12 median signed curve/rate error 0.0020802420979675468 -0.001690470375866976 median absolute 0.015801335422388374 0.013046313186217406
cell 3 2 n 19 median signed curve/rate error 5.874853842033345e-05 0.009254858285646206 median absolute 0.017304198415350247 0.023372993485827553
absolute curve residual quantiles [1.42518225e-02 6.30094106e-02 2.24821250e-01 1.19246057e+00
 2.89763126e+00 1.49850031e+01]
absolute rate residual quantiles [0.01670735 0.04584513 0.09794824 0.20871459 0.31140447]
curve residuals above 0.5 by time [7 5 4 4 1 5 4 4 7 5] affected coupons 41
3-point sweep, one EMI spike, max errors 0 [1.7534303704404004, 4.078998669498901e-05, 4.078998669498901e-05]
3-point sweep, one EMI spike, max errors 1 [1.2668995018713025, 4.8948314750635324e-05, 1.4125001073490886]
3-point sweep, one EMI spike, max errors 2 [1.3621629015413859, 2.218728711822493e-05, 1.306438030480895]
3-point sweep, one EMI spike, max errors 3 [1.4043857209483694, 7.735259867341915e-05, 1.2543585194856886]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ca7ea1c95ef61c17 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ca7ea1c9 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`44030 ms` LLM)
- **Rationale**: The audit found that endpoint leverage can hide a spike in a short sweep. Check omitted-point deviations and use the archive only when it corroborates an isolated spike. Test every three- and four-point subset of the real sweeps to catch missed spikes or unintended changes to clean fits; if either fails, revise the detector before submission.
- **Alternatives Weighed**: `Always use archive calibration for three- or four-point sweeps` (rejected: That would replace valid in-batch calibration even when there is no evidence of EMI.)

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_batch_03_solver.py')
s = p.read_text()
a = s.index('        if len(u) >= 3')
s = s[:a] + '''        if 3 <= len(u) <= 4 and np.ptp(u) > 0:
            # A high-leverage endpoint can hide a spike in a short sweep.
            # Check whether omitting a point exposes an EMI-sized deviation.
            for i in range(len(u)):
                other = np.arange(len(u)) != i
                if np.ptp(u[other]) == 0:
                    continue
                line = np.polyfit(u[other], v[other], 1)
                if abs(v[i] - np.polyval(line, u[i])) > 3.0:
                    prior = _historical_line(c)
                    delta = np.abs(v - np.polyval(prior, u))
                    if (delta <= 0.05).sum() >= len(u) - 1 and np.any(delta > 3.0):
                        u, v = u[delta <= 3.0], v[delta <= 3.0]
                        if len(u) < 4:
                            lines.append(prior)
                            break
                    # The archive is used only to corroborate an isolated spike.
                    break
            if len(u) < 3 or (len(u) == 3 and len(lines) == c + 1):
                continue
''' + s[a:]
s = s.replace('            for i in range(len(u)):\n', '            fallback = None\n            for i in range(len(u)):\n', 1)
s = s.replace('                        u, v = u[delta <= 3.0], v[delta <= 3.0]\n                        if len(u) < 4:\n                            lines.append(prior)\n                            break\n', '                        fallback = prior\n', 1)
s = s.replace('            if len(u) < 3 or (len(u) == 3 and len(lines) == c + 1):\n                continue\n        if len(u) >= 3', '            if fallback is not None:\n                lines.append(fallback)\n                continue\n        if len(u) >= 3', 1)
p.write_text(s)
PY
cat > /tmp/rheology_sparse_check.py <<'PY'
from itertools import combinations
import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c01_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_m97_c01_batch_03.pkl','rb'))[0]['dataset']; cal=d['calibration_load_cells']; base=m._calibrate(cal)
for size in (3,4):
    worst=0.; clean_error=0.; total=0
    for c in range(4):
        idx=np.flatnonzero(cal['lab_id']==c); others=np.flatnonzero(cal['lab_id']!=c)
        for picks in combinations(idx,size):
            subset=np.r_[list(picks),others]; q={k:v[subset].copy() for k,v in cal.items()}
            expected=np.polyfit(q['cmd_stress'][:size],q['true_stress'][:size],1)
            clean_error=max(clean_error,np.max(abs(m._calibrate(q)[c]-expected)))
            for i in range(size):
                old=q['true_stress'][i]
                for shift in (-4.,-3.01,3.01,4.):
                    q['true_stress'][i]=old+shift
                    worst=max(worst,np.max(abs(m._calibrate(q)[c]-base[c]))); total+=1
                q['true_stress'][i]=old
    print('sweep size',size,'EMI cases',total,'worst coefficient error',worst,'clean deviation from current fit',clean_error)
    assert clean_error < 1e-9 and worst < .001
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_sparse_check.py && OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_contract.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `1` (`6900 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_03_solver.py']`
- **Stdout (tail)**:
```text
sweep size 3 EMI cases 5760 worst coefficient error 4.3853637024271315 clean deviation from current fit 0.0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=184d388cee2ffefb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_184d388c (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/rheology_sparse_check.py", line 21, in <module>
    assert clean_error < 1e-9 and worst < .001
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

### Turn 10 — `execute_in_remote_docker` (`23084 ms` LLM)
- **Rationale**: Print one worst witness per sweep size, including the omitted-point and archival residuals. If the omitted clean pair is too narrowly spaced to predict the suspect point reliably, change the short-sweep trigger; if corroboration succeeds but the result still misses, fix the fallback control flow.
- **Alternatives Weighed**: `Lower the EMI threshold now` (rejected: The task specifies a 3.0 threshold, and the failing assertion does not yet show whether the observed deviation or the fallback logic caused the miss.)

```bash
sed -n '1,95p' /holdout_task/alpha_m97_c01_batch_03_solver.py; sed '/    assert clean_error/,$d' /tmp/rheology_sparse_check.py | sed 's/worst=max(worst,np.max(abs(m._calibrate(q)\[c\]-base\[c\]))); total+=1/err=np.max(abs(m._calibrate(q)[c]-base[c])); total+=1\n                    if err>worst:\n                        worst=err; witness=(c,q["cmd_stress"][:size].copy(),q["true_stress"][:size].copy(),i,shift)/' > /tmp/rheology_sparse_diagnose.py
cat >> /tmp/rheology_sparse_diagnose.py <<'PY'
    c,x,y,i,shift=witness
    print('witness:',c,'cmd',x,'true',y,'index',i,'shift',shift)
    for j in range(size):
        mask=np.arange(size)!=j
        est=np.polyfit(x[mask],y[mask],1)
        print('omit',j,'signed residual',y[j]-np.polyval(est,x[j]))
    print('archive residual',y-np.polyval(m._historical_line(c),x))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_sparse_diagnose.py
```
- **Container Exit Code**: `0` (`25713 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
pend(_historical_line(c))
    return np.asarray(lines)


def _prepare(dataset):
    a, b = dataset['train'], dataset['test']
    def joined(key):
        return np.concatenate((a[key], b[key]), axis=0)
    lab = joined('lab_id')
    panel = joined('ref_coupons')
    features = joined('coupon_features_raw').astype(float)
    for c in range(4):
        m = lab == c
        r = panel[m]
        r = r[:, np.isfinite(r).sum(axis=0) >= 2]
        features[m] = (features[m] - np.median(np.nanmean(r, axis=0))) / np.median(np.nanstd(r, axis=0))
    cells, cell = np.unique(np.c_[lab, joined('stratum')], axis=0, return_inverse=True)
    good = np.isfinite(features)
    sums = np.zeros((len(cells), features.shape[1]))
    counts = np.zeros_like(sums)
    np.add.at(sums, cell, np.where(good, features, 0.))
    np.add.at(counts, cell, good)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    fill = means.sum(axis=0) / (counts > 0).sum(axis=0)
    features = np.where(good, features, fill)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*joined('past_stress_cmd') + bias[:, None]
    hold = gain*joined('stress_hold_cmd') + bias
    return past, hold, features


def _design(data, dt, t, tau):
    past, hold, x = data
    memory = np.zeros((len(hold), 2))
sweep size 3 EMI cases 5760 worst coefficient error 4.3853637024271315 clean deviation from current fit 0.0
witness: 2 cmd [-0.646451 -2.057672 -2.064745] true [-3.91224093 -2.60982659 -2.61832196] index 0 shift -3.01
omit 0 signed residual -2.997430432563278
omit 1 signed residual 0.014948117562026741
omit 2 signed residual -0.015023037107240622
archive residual [-3.01002361e+00 -3.47716418e-05  2.81707356e-05]
sweep size 4 EMI cases 13440 worst coefficient error 7.735259867341915e-05 clean deviation from current fit 0.0
witness: 3 cmd [-1.355878 -1.273919 -1.02295   2.198106] true [-5.01532677 -0.94732626 -0.73907526  1.934429  ] index 0 shift -4.0
omit 0 signed residual -3.9999490169463874
omit 1 signed residual 2.149626006344911
omit 2 signed residual 1.851029165083248
omit 3 signed residual -30.062706285818933
archive residual [-4.00012010e+00 -1.40548564e-04 -1.78472495e-04  4.62667744e-05]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=184d388cee2ffefb (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_184d388c (directory).
```

### Turn 11 — `execute_in_remote_docker` (`24679 ms` LLM)
- **Rationale**: In the failing sweep, the two clean commands were only 0.0071 apart, so extrapolation reduced the apparent spike from 3.01 to 2.9974. Allow for that amplification in the preliminary check and retain the 3.0 threshold for archive corroboration. The real-sweep subset tests will show whether the fix catches missed spikes without replacing clean current fits; the isolated contract and schema tests verify the edited artifact.
- **Alternatives Weighed**: `Lower the EMI threshold itself` (rejected: The threshold remains 3.0. Only the preliminary check gets a noise allowance, scaled by how far it extrapolates from the other points.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_batch_03_solver.py')
s = p.read_text()
s = s.replace('                if abs(v[i] - np.polyval(line, u[i])) > 3.0:\n', '''                # Account for calibration noise amplified by extrapolation.
                # The archive check below still requires a residual above 3.0.
                x_other = u[other]
                centered = x_other - x_other.mean()
                leverage = 1.0/len(x_other) + (u[i] - x_other.mean())*centered/(centered@centered)
                allowance = 0.001*(1.0 + np.abs(leverage).sum())
                if abs(v[i] - np.polyval(line, u[i])) > 3.0 - allowance:
''', 1)
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /tmp/rheology_sparse_check.py
python3 -I /tmp/rheology_contract.py
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`26238 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_03_solver.py']`
- **Stdout (tail)**:
```text
sweep size 3 EMI cases 5760 worst coefficient error 7.735259867341915e-05 clean deviation from current fit 0.0
sweep size 4 EMI cases 13440 worst coefficient error 7.735259867341915e-05 clean deviation from current fit 0.0
channel 0 sparse archive versus full current max coefficient delta 4.078998669498901e-05 worst EMI coefficient error 3.818153686457881e-05
channel 1 sparse archive versus full current max coefficient delta 4.8948314750635324e-05 worst EMI coefficient error 1.8460903046024768e-05
channel 2 sparse archive versus full current max coefficient delta 2.218728711822493e-05 worst EMI coefficient error 1.9925208780274817e-05
channel 3 sparse archive versus full current max coefficient delta 7.735259867341915e-05 worst EMI coefficient error 2.4296443576946047e-05
default-process solve seconds 0.19256982719525695
max independent onset derivative error 1.6129941826648064e-10
noiseless max tau/weight errors 4.956204335826442e-10 3.0666635897347305e-11
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7f222abbed84596a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_7f222abb (directory).
```

### Turn 12 — `submit_completed_task` (`15059 ms` LLM)
- **Rationale**: The exact final artifact passed isolated schema, calibration, and physical-law checks. It fixed the short-sweep failure without changing clean fits; independent onset derivatives agreed within 1.7×10⁻¹⁰, and noiseless parameters were recovered within 5×10⁻¹⁰.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_03_solver.py`. The public schema passed, and the worst calibration coefficient error across 19,200 short-sweep EMI tests was 7.74×10⁻⁵, with no change to clean fits. Missing coupons remained the hardest validation slice: strain MAE was 0.195 and onset-rate MAE was 0.098.`

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
