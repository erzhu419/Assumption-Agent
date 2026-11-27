# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c03_batch_01__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c03_step_01` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `375,292` (`361,666` / `90,752` / `13,626`) |
| **Total Episode Wall Time (s)** | `435.12s` |

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

### Turn 1 — `execute_in_remote_docker` (`13212 ms` LLM)
- **Rationale**: Inspect the public schema and archived implementation to catch incorrect key, axis, or alignment assumptions before fitting; reuse only primitives that match the batch contract.
- **Alternatives Weighed**: `Reuse the prior solver as-is` (rejected: The batch plateau law and calibration rules must be checked before reusing any components.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -lh /tmp/hegel_continual_archive && sed -n '1,240p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel cell balanced imputation viscoelastic relaxation variable projection'
```
- **Container Exit Code**: `0` (`397 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
l_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
```

### Turn 2 — `execute_in_remote_docker` (`19919 ms` LLM)
- **Rationale**: Inspect calibration residuals for EMI and confirm reference-panel axes and missingness before implementing alignment and variable projection.
- **Alternatives Weighed**: `Reuse the archived alignment and calibration without inspection` (rejected: The batch specifies its own calibration precedence and reference-panel statistics; an inherited mismatch would bias the constitutive fit.)

```bash
sed -n '1,145p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.weighted_column_fill; python3 - <<'PY'
import numpy as np,pickle
items=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_01.pkl','rb')); print('items',len(items))
for item in items:
 d=item['dataset'];print('keys',list(item),list(d)); print('times',d['eval_times'],'dt',d['past_dt'],'regime',d['shear_banding_index'],d['crit_stress']);
 c=d['calibration_load_cells'];
 for l in range(4):
  m=c['lab_id']==l; x=c['cmd_stress'][m];y=c['true_stress'][m]; p=np.polyfit(x,y,1); print('calibration',l,len(x),p,'maxres',max(abs(y-np.polyval(p,x))))
 for part in ['train','test']:
  a=d[part]; print(part,[(k,np.shape(v)) for k,v in a.items()]); print('lab counts',np.bincount(a['lab_id'])); print('reference range',np.nanmin(a['ref_coupons']),np.nanmax(a['ref_coupons']));print('feature missing',np.isnan(a['coupon_features_raw']).sum(axis=0));
PY
```
- **Container Exit Code**: `0` (`754 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel",
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights",
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_reference_panel_balancing",
      "reference_source_txt": null
    }
  ]
}
items 1
keys ['dataset', 'gt'] ['past_dt', 'eval_times', 'crit_stress', 'yield_stress', 'shear_banding_index', 'calibration_load_cells', 'train', 'test']
times [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ] dt [0.4917 0.5023 0.5442 0.7304 0.4565 0.765  0.2965 0.6942 0.5869 0.3435
 0.3902 0.5082 0.3785 0.7877 0.3776 0.2105 0.2875 0.2881] regime 0.12 2.65
calibration 0 10 [1.28998908 0.12000143] maxres 0.00021677044913304133
calibration 1 10 [ 0.82003896 -0.15002605] maxres 0.0001420176423634345
calibration 2 10 [1.16998093 0.14000924] maxres 0.0001616162574000768
calibration 3 10 [ 0.76000147 -0.10998819] maxres 0.0001720959915643052
train [('lab_id', (180,)), ('stratum', (180,)), ('coupon_features_raw', (180, 4)), ('ref_coupons', (180, 32)), ('past_stress_cmd', (180, 18)), ('stress_hold_cmd', (180,)), ('strain_0', (180,)), ('aux_chamber_telemetry', (180, 3)), ('strain_curves_obs', (180, 10)), ('instant_rate_obs', (180,))]
lab counts [35 49 57 39]
reference range -5.334500795452639 5.556347685232906
feature missing [ 5  2  5 11]
test [('lab_id', (72,)), ('stratum', (72,)), ('coupon_features_raw', (72, 4)), ('ref_coupons', (72, 32)), ('past_stress_cmd', (72, 18)), ('stress_hold_cmd', (72,)), ('strain_0', (72,)), ('aux_chamber_telemetry', (72, 3))]
lab counts [17 16 21 18]
reference range -3.3126887763976267 4.480120931206276
feature missing [3 1 1 3]
```

### Turn 3 — `execute_in_remote_docker` (`19801 ms` LLM)
- **Rationale**: Recover the exact imputation convention and verify whether any observed cell lacks a feature column; that determines whether a fallback is needed. Inspect public scoring field names to plan verification without using targets for fitting.
- **Alternatives Weighed**: `Use pooled reference-panel statistics` (rejected: The specified medians of column statistics are not generally equal to pooled panel statistics.)

```bash
sed -n '/^def _align/,/^def solve/p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py; head -c 2200 /tmp/hegel_continual_archive/archive_calibration_history.json; printf '\n'; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d=p['dataset']; g=p['gt']; print('public scoring fields:',[(k,np.shape(v)) for k,v in g.items()] if isinstance(g,dict) else type(g)); t=d['train']; v=d['test']; ids=np.r_[t['lab_id'],v['lab_id']]; s=np.r_[t['stratum'],v['stratum']]; x=np.r_[t['coupon_features_raw'],v['coupon_features_raw']]; r=np.r_[t['ref_coupons'],v['ref_coupons']]; print('panel NaNs',np.isnan(r).sum()); print('cell sizes and finite features:');
for k in np.unique(np.c_[ids,s],axis=0):
 m=(ids==k[0])&(s==k[1]); print(k,m.sum(),np.isfinite(x[m]).sum(0))
PY
```
- **Container Exit Code**: `0` (`638 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
x
    order = np.argsort(p[:2])
    return p[:2][order], np.r_[p[2:4][order], p[4:]]


def solve_alpha_m97_c03_s00(dataset: dict) -> dict:
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
      3,
      3,
      3,
      3,
      3,
      3
    ],
    "cmd_stress": [
      0.893607,
      -0.483769,
      -1.39264,
      1.611445,
      -1.810815,
      -1.870654,
      1.48082,
      1.268821,
      -0.114529,
      0.252739,
      -1.204476,
      -0.664616,
      -0.154159,
      -0.613549,
      -1.259125,
      -2.011382,
      2.18392,
      -0.23172,
      1.98347,
      -2.000907,
      -0.486537,
      -0.139492,
      -0.631342,
      -1.874772,
      -0.708257,
      2.13352,
      0.386634,
      1.10423,
      -1.407517,
      1.70372,
      1.521295,
      1.987385,
      1.659545,
      -0.294044,
      0.359165,
      -0.017198,
      1.319848,
      0.959274,
      1.866895,
      -0.480321
    ],
    "true_stress": [
      0.893713,
      -0.483774,
      -1.392656,
      1.611564,
      -1.810667,
      -1.870622,
      1.480773,
      1.268837,
      -0.114695,
      0.252685,
      -1.204516,
      -0.664704,
      -0.154113,
      -0.613571,
      -1.259248,
      -2.011128,
      2.183777,
      -0.231673,
      1.983509,
      -2.001075,
      -0.486729,
      -0.139381,
      -0.631253,
      -1.874747,
      -0.708398,
      2.133538,
      0.386675,
      1.104402,
      -1.407596,
      1.703866,
      1.521337,
      1.987399,
      1.659449,
      -0.294029,
      0.359147,
      -0.017404,
      1.319703,
      0.959321,
      1.866928,
      -0.480472
    ]
  }
]
public scoring fields: [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (72, 10)), ('te_true_rate0', (72,)), ('te_hist_true', (72, 18)), ('te_hold_true', (72,))]
panel NaNs 333
cell sizes and finite features:
[0 0] 17 [16 17 17 15]
[0 1] 20 [19 20 20 18]
[0 2] 15 [15 15 14 12]
[1 0] 14 [13 13 14 13]
[1 1] 35 [34 33 35 35]
[1 2] 16 [16 16 16 16]
[2 0] 22 [22 22 22 22]
[2 1] 18 [17 18 17 18]
[2 2] 38 [37 38 36 37]
[3 0] 14 [13 14 14 12]
[3 1] 20 [20 20 18 20]
[3 2] 23 [22 23 23 20]
```

### Turn 4 — `execute_in_remote_docker` (`69014 ms` LLM)
- **Rationale**: Implement the stated curve and initial-rate laws as a shared design fitted only to training observations. Run the isolated schema validator now to catch import, shape, or nonfinite-output failures before evaluating each output on public reference targets and held-out labs.
- **Alternatives Weighed**: `Sort the branches after fitting` (rejected: The crystallization terms attach specifically to the fast branch, so swapping only the time constants and hereditary weights would change the predictions.)

```bash
cat > /holdout_task/alpha_m97_c03_batch_01_solver.py <<'PY'
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _calibration(cal):
    coefficients = np.empty((4, 2))
    history = None
    for lab in range(4):
        mask = (cal['lab_id'] == lab) & np.isfinite(cal['cmd_stress']) & np.isfinite(cal['true_stress'])
        x, y = cal['cmd_stress'][mask], cal['true_stress'][mask]
        good = np.ones(len(x), dtype=bool)
        if np.unique(x).size >= 2:
            i, j = np.triu_indices(len(x), 1)
            distinct = x[i] != x[j]
            slope = np.median((y[j[distinct]] - y[i[distinct]]) / (x[j[distinct]] - x[i[distinct]]))
            intercept = np.median(y - slope*x)
            good = np.abs(y - slope*x - intercept) <= 3.0
        if good.sum() >= 4 or (good.all() and np.unique(x).size >= 2):
            coefficients[lab] = _line(x[good], y[good])
            continue
        if history is None:
            history = json.loads(_HISTORY.read_text())
        for sweep in reversed(history):
            hist_lab = np.asarray(sweep['lab_id'])
            a = np.asarray(sweep['cmd_stress'])[hist_lab == lab]
            b = np.asarray(sweep['true_stress'])[hist_lab == lab]
            if np.unique(a).size >= 2:
                coefficients[lab] = _line(a, b)
                break
        else:
            raise ValueError(f'No usable calibration for lab {lab}')
    return coefficients


def _align(train, test):
    labs, strata, raw, panel = (np.concatenate((train[k], test[k])) for k in (
        'lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        mask = labs == lab
        ref = panel[mask]
        center = np.nanmedian(np.nanmean(ref, axis=0))
        scale = np.nanmedian(np.nanstd(ref, axis=0, ddof=0))
        aligned[mask] = (raw[mask] - center) / scale
    keys = np.unique(np.column_stack((labs, strata)), axis=0)
    cells = [(labs == lab) & (strata == stratum) for lab, stratum in keys]
    means = []
    for mask in cells:
        v = aligned[mask]
        n = np.isfinite(v).sum(axis=0)
        means.append(np.divide(np.nansum(v, axis=0), n, out=np.full(v.shape[1], np.nan), where=n > 0))
    means = np.asarray(means)
    for i, mask in enumerate(cells):
        fill = means[i].copy()
        for j in np.flatnonzero(~np.isfinite(fill)):
            local = means[keys[:, 0] == keys[i, 0], j]
            fill[j] = np.nanmean(local) if np.isfinite(local).any() else np.nanmean(means[:, j])
        aligned[mask] = np.where(np.isfinite(aligned[mask]), aligned[mask], fill)
    return aligned


def _inputs(dataset, part, aligned, coefficients):
    gain, bias = coefficients[part['lab_id']].T
    stress = gain[:, None]*part['past_stress_cmd'] + bias[:, None]
    hold = gain*part['stress_hold_cmd'] + bias
    dt = np.asarray(dataset['past_dt'])
    times = np.asarray(dataset['eval_times'])
    damage = np.maximum(0., np.abs(stress) - dataset['crit_stress'])**2 @ dt
    plateau = (dataset['shear_banding_index'] <= .50) & (damage > 0.)
    activation = np.where(plateau, np.tanh(damage), 0.)
    return stress, hold, aligned, dt, times, plateau, activation


def _design(logtau, stress, hold, x, dt, times, plateau, activation):
    tau = np.exp(logtau)
    remaining = np.r_[np.cumsum(dt[::-1])[::-1], 0.]
    u = remaining[:, None]/tau
    e = np.exp(-u)
    mem = stress @ np.diff(e, axis=0)
    dmem = stress @ np.diff(u*e, axis=0)
    gap = hold[:, None] - mem
    v = times[:, None]/tau
    phi = -np.expm1(-v)
    dphi = -v*np.exp(-v)
    terminal = np.where(plateau[:, None], np.log1p(times)[None, :], times[None, :])
    drift_basis = np.column_stack((np.ones(len(x)), x))
    primary = activation*hold
    secondary = activation*mem[:, 0]
    n, k = len(x), len(times)
    a = np.empty((n, k+1, 9))
    a[:, :k, :2] = gap[:, None, :]*phi
    a[:, k, :2] = gap/tau
    a[:, :k, 2:7] = drift_basis[:, None, :]*terminal[:, :, None]
    a[:, k, 2:7] = drift_basis
    a[:, :k, 7] = primary[:, None]*phi[:, 0]
    a[:, k, 7] = primary/tau[0]
    a[:, :k, 8] = secondary[:, None]*terminal
    a[:, k, 8] = secondary
    d = np.empty((n, k+1, 4))
    d[:, :k, :2] = -dmem[:, None, :]*phi + gap[:, None, :]*dphi
    d[:, k, :2] = (-dmem-gap)/tau
    d[:, :k, 2] = primary[:, None]*dphi[:, 0]
    d[:, k, 2] = -primary/tau[0]
    dsecondary = activation*dmem[:, 0]
    d[:, :k, 3] = dsecondary[:, None]*terminal
    d[:, k, 3] = dsecondary
    return a, d


def _fit(inputs, y):
    dt, times = inputs[3:5]
    low = np.r_[np.full(2, np.log(min(dt.min(), times[times > 0].min())/30)), np.full(9, -np.inf)]
    high = np.r_[np.full(2, np.log((dt.sum()+times.max())*15)), np.full(9, np.inf)]

    def optimize(p, scale, loss):
        def fun(p):
            a, _ = _design(p[:2], *inputs)
            return ((a @ p[2:] - y)/scale).ravel()

        def jac(p):
            a, d = _design(p[:2], *inputs)
            fast = d[..., 0]*p[2] + d[..., 2]*p[9] + d[..., 3]*p[10]
            slow = d[..., 1]*p[3]
            jacobian = np.concatenate((fast[..., None], slow[..., None], a), axis=2)
            return (jacobian/scale[..., None]).reshape(-1, 11)

        return least_squares(fun, p, jac=jac, bounds=(low, high), loss=loss, x_scale='jac', max_nfev=120)

    best = None
    for pair in ((.35, 2.3), (.7, 3.8), (.18, 1.25)):
        logtau = np.log(pair)
        a, _ = _design(logtau, *inputs)
        w = np.linalg.lstsq(a.reshape(-1, 9), y.ravel(), rcond=None)[0]
        fit = optimize(np.r_[logtau, w], np.asarray(.09), 'soft_l1')
        if fit.x[0] < fit.x[1] and (best is None or fit.cost < best.cost):
            best = fit
    if best is None:
        raise RuntimeError('No fit in fast-to-slow branch order')
    p = best.x

    def mad(v):
        return max(1e-6, 1.4826*np.median(np.abs(v-np.median(v))))

    for _ in range(2):
        a, _ = _design(p[:2], *inputs)
        r = a @ p[2:] - y
        scale = 2.385*np.r_[np.full(len(times), mad(r[:, :-1])), mad(r[:, -1])]
        refined = optimize(p, scale, 'cauchy').x
        if refined[0] < refined[1]:
            p = refined
    return p


def solve_alpha_m97_c03_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    x = _align(train, test)
    cal = _calibration(dataset['calibration_load_cells'])
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs']))
    p = _fit(_inputs(dataset, train, x[:n], cal), y)
    a, _ = _design(p[:2], *_inputs(dataset, test, x[n:], cal))
    pred = a @ p[2:]
    return {'estimated_taus': np.exp(p[:2]), 'hereditary_weights': p[2:9],
            'regime_mode_weights': p[9:11], 'predicted_strain': pred[:, :-1] + test['strain_0'][:, None],
            'initial_rate_pred': pred[:, -1]}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`569 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f5917f3b82824e25 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f5917f3b (directory).
```

### Turn 5 — `execute_in_remote_docker` (`29516 ms` LLM)
- **Rationale**: Compare all five outputs against public references only after fitting, then refit without each lab's strain and rate measurements. A lab-specific failure would direct attention to calibration or alignment; a regime- or horizon-specific failure would direct attention to the constitutive terms.
- **Alternatives Weighed**: `Submit after schema validation` (rejected: Correct shapes do not establish parameter recovery or accuracy on held-out labs and damage regimes.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',root+'alpha_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p=pickle.load(open(root+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; out=m.solve_alpha_m97_c03_batch_01(d)
for key,ref in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]:
 a,b=np.asarray(out[key]),np.asarray(g[ref]); print(key,'estimate',a,'reference',b,'maxabs',np.max(abs(a-b)),'relative L2',np.linalg.norm(a-b)/np.linalg.norm(b))
cal=m._calibration(d['calibration_load_cells']); x=m._align(tr,te); n=len(tr['lab_id']); y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; ti=m._inputs(d,tr,x[:n],cal); vi=m._inputs(d,te,x[n:],cal)
print('active train/test',ti[-2].sum(),vi[-2].sum(),'calibration max errors',np.max(abs(cal[:,0]-g['stress_gains'])),np.max(abs(cal[:,1]-g['stress_biases'])))
curve=out['predicted_strain']-g['te_clean']; rate=out['initial_rate_pred']-g['te_true_rate0']; rms=lambda z:np.sqrt(np.mean(z*z)); print('public clean curve/rate RMSE',rms(curve),rms(rate),'worst horizon',np.sqrt(np.mean(curve**2,axis=0)).max())
for label, mask in [(f'lab {i}',te['lab_id']==i) for i in range(4)]+[('plateau',vi[-2]),('linear',~vi[-2])]:
 if mask.any(): print('clean slice',label,mask.sum(),rms(curve[mask]),rms(rate[mask]))
for lab in range(4):
 keep=tr['lab_id']!=lab; a=tuple(v[keep] if j in (0,1,2,5,6) else v for j,v in enumerate(ti)); b=tuple(v[~keep] if j in (0,1,2,5,6) else v for j,v in enumerate(ti)); q=m._fit(a,y[keep]); residual=m._design(q[:2],*b)[0]@q[2:]-y[~keep]; print('LOLO',lab,'taus',np.exp(q[:2]),'curve/rate RMSE',rms(residual[:,:-1]),rms(residual[:,-1]),'curve/rate median abs',np.median(abs(residual[:,:-1])),np.median(abs(residual[:,-1])))
PY
```
- **Container Exit Code**: `0` (`803 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
estimated_taus estimate [0.57706476 3.97438892] reference [0.568 4.113] maxabs 0.13861107633569025 relative L2 0.033455202333712006
hereditary_weights estimate [ 0.68888961  0.53058566  0.08856024  0.10121914 -0.10457303  0.06961376
 -0.04894824] reference [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09] maxabs 0.0587808634489634 relative L2 0.11598406182276019
regime_mode_weights estimate [0.29010345 0.22962968] reference [0.28 0.24] maxabs 0.010370321330410964 relative L2 0.039259994141449164
active train/test 120 48 calibration max errors 3.896190412500111e-05 2.6047468645679084e-05
public clean curve/rate RMSE 0.19070720706415464 0.1065868724568904 worst horizon 0.3351933621499605
clean slice lab 0 17 0.14894944837690655 0.08218315802661362
clean slice lab 1 16 0.24963747471126221 0.1476452224080046
clean slice lab 2 21 0.19397643762840694 0.10572004468483835
clean slice lab 3 18 0.1588427455462714 0.08153340572541164
clean slice plateau 48 0.1367593212115572 0.10234403736703628
clean slice linear 24 0.2677713438778304 0.11460226948561546
LOLO 0 taus [0.58466646 3.72461036] curve/rate RMSE 2.4102115963761594 0.13452986891315813 curve/rate median abs 0.10001442055074888 0.0816968954560125
LOLO 1 taus [0.57452678 3.55941569] curve/rate RMSE 0.379981971510393 0.10356789230664726 curve/rate median abs 0.08752095784768729 0.07311441753711767
LOLO 2 taus [0.56813887 4.45306781] curve/rate RMSE 3.6269883056989873 0.11451610843923749 curve/rate median abs 0.08270413329137727 0.07255529919374304
LOLO 3 taus [0.57424979 3.7292538 ] curve/rate RMSE 1.3083948803994354 0.11112439111029045 curve/rate median abs 0.0910141081219384 0.06941999365444973

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f5917f3b82824e25 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f5917f3b (directory).
```

### Turn 6 — `execute_in_remote_docker` (`29494 ms` LLM)
- **Rationale**: Use reference parameters only for diagnosis, never fitting. Test whether clean residuals lie in the terminal-flow basis, which would implicate alignment, or have additional time structure, which would implicate the constitutive design. Inspect the largest measured training residuals to determine whether they are isolated points or whole-curve disturbances.
- **Alternatives Weighed**: `Retune the robust loss immediately` (rejected: Retuning cannot resolve a discrepancy that persists when the forward model uses the public reference parameters.)

```bash
cd /holdout_task && cat public_asset_index.json; OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
r='/holdout_task/'; s=importlib.util.spec_from_file_location('candidate',r+'alpha_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
p=pickle.load(open(r+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); x=m._align(tr,te); c=np.c_[g['stress_gains'],g['stress_biases']]; w=np.r_[g['theta'],g['vp_theta']]; lt=np.log(g['taus']); rms=lambda x:np.sqrt(np.mean(x*x))
for name,part,xx,curve,rate in [('train',tr,x[:n],tr['strain_curves_obs'],tr['instant_rate_obs']),('test',te,x[n:],g['te_clean'],g['te_true_rate0'])]:
 inp=m._inputs(d,part,xx,c); a=m._design(lt,*inp)[0]; y=np.c_[curve-part['strain_0'][:,None],rate]; e=y-a@w; t=inp[4]; print(name,'oracle stress check',np.max(abs(inp[0]-g['te_hist_true'])) if name=='test' else None)
 for lab in [-1,0,1,2,3]:
  for active in [False,True]:
   mask=(inp[-2]==active)&((part['lab_id']==lab) if lab>=0 else True); v=e[mask]; f=np.log1p(t) if active else t; h=v[:,:-1]@f/(f@f); print(lab,active,len(v),'curve RMSE/MdAE',rms(v[:,:-1]),np.median(abs(v[:,:-1])),'rate RMSE/MdAE',rms(v[:,-1]),np.median(abs(v[:,-1])),'median terminal residual',np.round(np.median(v,axis=0),4),'excess beyond terminal basis',rms(v[:,:-1]-h[:,None]*f),'rate minus terminal coeff',rms(v[:,-1]-h))
 for lab in range(4):
  mask=part['lab_id']==lab; print('panel lab',lab,'raw med',np.nanmedian(part['coupon_features_raw'][mask],axis=0),'ref mean/std med',np.nanmedian(np.nanmean(part['ref_coupons'][mask],axis=0)),np.nanmedian(np.nanstd(part['ref_coupons'][mask],axis=0)))
 if name=='train':
  mag=np.max(abs(e[:,:-1]),axis=1)
  for i in np.argsort(mag)[-8:]: print('largest train residual',i,'lab',part['lab_id'][i],'active',inp[-2][i],np.round(e[i],3))
PY
```
- **Container Exit Code**: `0` (`596 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
9162013746164 rate minus terminal coeff 0.2851901861068182
1 True 12 curve RMSE/MdAE 0.052719339414197125 0.0005039502522662942 rate RMSE/MdAE 0.0402567304949382 0.0004436600239688185 median terminal residual [-0.     -0.0001 -0.0001 -0.0002 -0.0002 -0.0002 -0.0002 -0.0003 -0.0003
 -0.0003 -0.0002] excess beyond terminal basis 1.7219488767498601e-16 rate minus terminal coeff 3.003836566871899e-16
2 False 9 curve RMSE/MdAE 0.5841870547539513 0.2678259677570948 rate RMSE/MdAE 0.004456756246664047 0.004354886230920757 median terminal residual [-0.0124 -0.0718 -0.1581 -0.26   -0.3722 -0.4916 -0.6165 -0.7456 -0.878
 -1.0132 -0.0044] excess beyond terminal basis 0.06553657641420212 rate minus terminal coeff 0.17904108011503908
2 True 12 curve RMSE/MdAE 0.06003310182639625 0.004779650978423266 rate RMSE/MdAE 0.04584155317298227 0.004174936972143217 median terminal residual [-0.0011 -0.0026 -0.0036 -0.0045 -0.0052 -0.0058 -0.0063 -0.0068 -0.0072
 -0.0076 -0.0042] excess beyond terminal basis 1.4418258433950068e-16 rate minus terminal coeff 3.4426148386305484e-16
3 False 4 curve RMSE/MdAE 0.48377143054931276 0.2088368251022391 rate RMSE/MdAE 0.013342445106582213 0.008160769530530465 median terminal residual [-0.0093 -0.0504 -0.109  -0.1778 -0.2532 -0.3334 -0.4172 -0.5037 -0.5924
 -0.6828 -0.0071] excess beyond terminal basis 0.051101636479393274 rate minus terminal coeff 0.13960589172512775
3 True 14 curve RMSE/MdAE 0.026723308548635537 0.01448083733282704 rate RMSE/MdAE 0.020406041542428498 0.014867261044792968 median terminal residual [-0.0008 -0.002  -0.0028 -0.0034 -0.004  -0.0045 -0.0049 -0.0052 -0.0055
 -0.0058 -0.0032] excess beyond terminal basis 1.551184838939811e-16 rate minus terminal coeff 2.1757079508399816e-16
panel lab 0 raw med [-0.38526421 -0.58233567 -0.92923654 -0.78655863] ref mean/std med -0.6355278208636703 0.7794880082856288
panel lab 1 raw med [-0.00488688  0.09689779  0.71733505  0.2278693 ] ref mean/std med 0.5971242557846763 1.1858141495088605
panel lab 2 raw med [ 0.30538536  0.03850366 -0.13400759 -0.16593985] ref mean/std med -0.24886536276956758 0.8469398029433292
panel lab 3 raw med [0.59393193 0.34610036 0.41664549 0.86755063] ref mean/std med 0.5945581513614469 1.2936324583128371

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f5917f3b82824e25 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f5917f3b (directory).
```

### Turn 7 — `execute_in_remote_docker` (`34184 ms` LLM)
- **Rationale**: Compare both terminal bases on measured training curves and clean public slices at fixed parameters. If the log basis removes the inactive time trend, change the gate; if it does not, inspect the remaining time structure before editing. Complete and missing-feature rate slices separate alignment and imputation errors from the clock because both clocks have derivative one at zero.
- **Alternatives Weighed**: `Retune robust fitting under the linear inactive law` (rejected: The residual remains when fitting uncertainty is removed by using the public reference parameters for diagnosis.)
- **ATMS NOGOOD DECLARED**: Refuted `The inactive coupons follow the linear terminal basis used in the first candidate.` | Witness: `With reference stresses and parameters, inactive lab 1 curve residual RMSE was 0.910 while initial-rate RMSE was 0.000669; projecting out a linear terminal term still left curve RMSE 0.104.` | Blanket Diagnosis: `A coupon alignment error can change terminal amplitude but cannot explain this time-dependent residual while leaving the initial rate nearly exact.` | Sublated Pivot: `Test whether the logarithmic clock is campaign-wide and only the two regime amplitudes are gated by damage; diagnose imputation separately using the initial rate.`

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
root='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',root+'alpha_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open(root+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); cal=m._calibration(d['calibration_load_cells']); x=m._align(tr,te); w=np.r_[g['theta'],g['vp_theta']]; lt=np.log(g['taus']); rms=lambda v:np.sqrt(np.mean(v*v)); t=d['eval_times']
for name,part,xx,curve,rate in [('train',tr,x[:n],tr['strain_curves_obs'],tr['instant_rate_obs']),('test',te,x[n:],g['te_clean'],g['te_true_rate0'])]:
    inp=m._inputs(d,part,xx,cal); a=m._design(lt,*inp)[0]; y=np.c_[curve-part['strain_0'][:,None],rate]; active=inp[-2]; beta=a[:,-1,2:7]@w[2:7]; full=np.isfinite(part['coupon_features_raw']).all(axis=1)
    for law,f in [('linear',t),('log',np.log1p(t))]:
        pred=a@w; pred[~active,:-1]+=(beta[~active,None]*(f-t))
        e=y-pred
        for label,mask in [('inactive',~active),('inactive complete',~active&full),('active complete',active&full),('missing',~full)]:
            v=e[mask]; print(name,law,label,len(v),'RMSE',rms(v[:,:-1]),rms(v[:,-1]),'MdAE',np.median(abs(v[:,:-1])),np.median(abs(v[:,-1])))
        if name=='train' and law=='log':
            print('abs residual quantiles by horizon (last is rate):', np.round(np.quantile(abs(e),[.5,.8,.95,.99],axis=0),4)); print('large curve residuals per coupon',np.unique((abs(e[:,:-1])>.7).sum(axis=1),return_counts=True))
    if name=='test':
        e=y-a@w
        for lab in range(4):
            keep=(part['lab_id']==lab)&full
            z=np.linalg.lstsq(np.c_[np.ones(keep.sum()),xx[keep]@w[3:7]],e[keep,-1],rcond=None)[0]
            print('complete rate alignment lab',lab,'offset/slope residual',z,'remaining max',np.max(abs(e[keep,-1]-z[0]-z[1]*(xx[keep]@w[3:7]))))
        for i in np.flatnonzero(~full): print('missing feature row',i,'lab',part['lab_id'][i],'stratum',part['stratum'][i],'cols',np.flatnonzero(~np.isfinite(part['coupon_features_raw'][i])),'rate residual',round(e[i,-1],5))
PY
```
- **Container Exit Code**: `0` (`472 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
4 ]
 [ 0.0303  0.0294  0.0359  0.0502  0.0366  0.0532  0.0591  0.0424  0.0482
   0.0432  0.0331]
 [ 1.1508  0.2703  0.1503  0.6344  0.2729  0.4206  0.4915  0.2766  0.3179
   0.3254  0.0867]
 [10.1356  5.9132  4.8819  3.9503  1.3645  1.2652  2.2193  0.524   1.0994
   1.7574  0.2188]]
large curve residuals per coupon (array([0, 1, 2]), array([135,  41,   4]))
test linear inactive 24 RMSE 0.6155105343401884 0.013577873333838454 MdAE 0.2408163801775569 0.004213473217962074
test linear inactive complete 23 RMSE 0.6239576530536433 0.00690631908410486 MdAE 0.2357104236757131 0.004066655329738822
test linear active complete 42 RMSE 0.01579400342436808 0.012061468926725346 MdAE 0.004254169344363068 0.004187303495191919
test linear missing 7 RMSE 0.1782228659077974 0.0865910139979473 MdAE 0.09088673404043923 0.066400788013798
test log inactive 24 RMSE 0.01778446628579409 0.013577873333838454 MdAE 0.003931853772737165 0.004213473217962074
test log inactive complete 23 RMSE 0.00904537825263059 0.00690631908410486 MdAE 0.003574151610072407 0.004066655329738822
test log active complete 42 RMSE 0.01579400342436808 0.012061468926725346 MdAE 0.004254169344363068 0.004187303495191919
test log missing 7 RMSE 0.11339442833473798 0.0865910139979473 MdAE 0.08138477692852872 0.066400788013798
complete rate alignment lab 0 offset/slope residual [-0.00145856 -0.02728705] remaining max 2.4536662706332217e-05
complete rate alignment lab 1 offset/slope residual [-0.00033717 -0.00110187] remaining max 0.00015402585632147132
complete rate alignment lab 2 offset/slope residual [-0.00217452 -0.01390836] remaining max 4.81683079838445e-05
complete rate alignment lab 3 offset/slope residual [-0.00269206 -0.07592686] remaining max 1.0851764251585284e-05
missing feature row 16 lab 2 stratum 1 cols [0] rate residual -0.08304
missing feature row 18 lab 1 stratum 1 cols [1] rate residual 0.13929
missing feature row 27 lab 0 stratum 2 cols [3] rate residual -0.0664
missing feature row 44 lab 1 stratum 1 cols [0] rate residual 0.00659
missing feature row 45 lab 3 stratum 2 cols [0 3] rate residual -0.01605
missing feature row 47 lab 2 stratum 2 cols [2] rate residual -0.13472
missing feature row 62 lab 0 stratum 1 cols [3] rate residual 0.05769

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f5917f3b82824e25 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f5917f3b (directory).
```

### Turn 8 — `execute_in_remote_docker` (`35778 ms` LLM)
- **Rationale**: Apply the logarithmic clock to all coupons in the crystallization campaign while retaining the damage gate on the two regime amplitudes. Refit from measured training data and verify in an isolated process. If error remains concentrated in missing-feature rows, assess imputation separately rather than retuning the clock.
- **Alternatives Weighed**: `Change the clock and imputation together` (rejected: The clock has direct evidence from complete coupons. Keeping the specified imputation fixed isolates the effect of the clock correction.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p=Path('/holdout_task/alpha_m97_c03_batch_01_solver.py')
s=p.read_text()
s=s.replace('    return stress, hold, aligned, dt, times, plateau, activation', '    return stress, hold, aligned, dt, times, plateau, activation, dataset[\'shear_banding_index\'] <= .50')
s=s.replace('def _design(logtau, stress, hold, x, dt, times, plateau, activation):', 'def _design(logtau, stress, hold, x, dt, times, plateau, activation, crystallization_campaign):')
s=s.replace('    terminal = np.where(plateau[:, None], np.log1p(times)[None, :], times[None, :])', '    # The clock is shared in this campaign; damage gates the added amplitudes.\n    terminal = np.log1p(times) if crystallization_campaign else times')
s=s.replace('terminal[:, :, None]', 'terminal[None, :, None]')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
r='/holdout_task/'; spec=importlib.util.spec_from_file_location('c',r+'alpha_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p=pickle.load(open(r+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); x=m._align(tr,te); cal=m._calibration(d['calibration_load_cells']); ti=m._inputs(d,tr,x[:n],cal); vi=m._inputs(d,te,x[n:],cal); out=m.solve_alpha_m97_c03_batch_01(d); rms=lambda z:np.sqrt(np.mean(z*z))
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]:
 a,b=out[k],g[j]; print(k,np.round(a,6),'relative L2',np.linalg.norm(a-b)/np.linalg.norm(b))
e=out['predicted_strain']-g['te_clean']; q=out['initial_rate_pred']-g['te_true_rate0']; full=np.isfinite(te['coupon_features_raw']).all(1); active=vi[5]
for label,keep in [('pooled',np.ones(len(e),bool)),('active',active),('inactive',~active),('complete',full),('missing',~full)]+[(f'lab {j}',te['lab_id']==j) for j in range(4)]:
 print('clean',label,keep.sum(),'curve/rate RMSE',rms(e[keep]),rms(q[keep]),'worst horizon',np.sqrt(np.mean(e[keep]**2,axis=0)).max())
y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; sub=lambda inp,mask:tuple(v[mask] if j in (0,1,2,5,6) else v for j,v in enumerate(inp))
for lab in range(4):
 keep=tr['lab_id']!=lab; z=m._fit(sub(ti,keep),y[keep]); er=m._design(z[:2],*sub(ti,~keep))[0]@z[2:]-y[~keep]; print('LOLO',lab,'taus',np.exp(z[:2]),'curve/rate MdAE',np.median(abs(er[:,:-1])),np.median(abs(er[:,-1])),'curve/rate RMSE',rms(er[:,:-1]),rms(er[:,-1]))
PY
```
- **Container Exit Code**: `0` (`952 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
estimated_taus [0.566771 3.987231] relative L2 0.03029234833300313
hereditary_weights [ 0.674995  0.543962  0.137695  0.156563 -0.128476  0.115799 -0.088275] relative L2 0.010934632323186307
regime_mode_weights [0.282176 0.238819] relative L2 0.0067130695922249925
clean pooled 72 curve/rate RMSE 0.03600524473046763 0.027811916166167492 worst horizon 0.049978948228215934
clean active 48 curve/rate RMSE 0.04257166458064671 0.03293415118685124 worst horizon 0.05905729509383801
clean inactive 24 curve/rate RMSE 0.016261601826647366 0.01229599177394964 worst horizon 0.022763075152420386
clean complete 65 curve/rate RMSE 0.011028689456653298 0.008771482204113917 worst horizon 0.015709502678053364
clean missing 7 curve/rate RMSE 0.11047502060241504 0.08509756125121684 worst horizon 0.15297395124494736
clean lab 0 17 curve/rate RMSE 0.026106383465221587 0.01977380113183881 worst horizon 0.03649595330298095
clean lab 1 16 curve/rate RMSE 0.04617016307004685 0.03571977721817651 worst horizon 0.06399986074240319
clean lab 2 21 curve/rate RMSE 0.0446934952368228 0.034914143117073665 worst horizon 0.06171971790267102
clean lab 3 18 curve/rate RMSE 0.01779252825501739 0.012978087292237992 worst horizon 0.025466449482140198
LOLO 0 taus [0.5668494  3.98197658] curve/rate MdAE 0.014877422664787948 0.015356455253913603 curve/rate RMSE 2.403689045853015 0.06010891245376855
LOLO 1 taus [0.56622707 3.94212518] curve/rate MdAE 0.01371841389209276 0.010866861502446479 curve/rate RMSE 0.3548056117518936 0.027811453735665796
LOLO 2 taus [0.56620188 3.9936014 ] curve/rate MdAE 0.013336083626021011 0.016798432973114785 curve/rate RMSE 3.6034816309511086 0.06511964758580213
LOLO 3 taus [0.56792031 4.03868742] curve/rate MdAE 0.020366214414389683 0.018710096911977114 curve/rate RMSE 1.2931294963169417 0.03932760827416853

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=456a88111013bf3f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_456a8811 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`35042 ms` LLM)
- **Rationale**: Test whether imputed training features materially bias parameter recovery without changing the required prediction-time imputation. Also exercise the specified one-point history fallback and EMI rejection: a failure would require fixing calibration precedence or outlier detection before submission.
- **Alternatives Weighed**: `Tune imputation against clean test residuals` (rejected: The task specifies cell-mean imputation; fitting an adjustment to test residuals would leak targets and would not recover the missing measurements.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
r='/holdout_task/'; s=importlib.util.spec_from_file_location('candidate',r+'alpha_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open(r+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); c=m._calibration(d['calibration_load_cells']); x=m._align(tr,te); ti=m._inputs(d,tr,x[:n],c); vi=m._inputs(d,te,x[n:],c)
y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; yref=np.c_[g['te_clean']-te['strain_0'][:,None],g['te_true_rate0']]; complete=np.isfinite(tr['coupon_features_raw']).all(1); test_complete=np.isfinite(te['coupon_features_raw']).all(1); sub=lambda inp,k:tuple(v[k] if j in (0,1,2,5,6) else v for j,v in enumerate(inp)); rms=lambda z:np.sqrt(np.mean(z*z))
for label,usable in [('all',np.ones(n,bool)),('complete-only',complete)]:
 q=m._fit(sub(ti,usable),y[usable]); e=m._design(q[:2],*vi)[0]@q[2:]-yref
 print(label,'tau/hw/rw relative L2',*[round(np.linalg.norm(a-b)/np.linalg.norm(b),6) for a,b in [(np.exp(q[:2]),g['taus']),(q[2:9],g['theta']),(q[9:11],g['vp_theta'])]])
 for tag,k in [('all',np.ones(len(e),bool)),('complete',test_complete),('missing',~test_complete)]+[(f'lab {i}',te['lab_id']==i) for i in range(4)]:
  print(label,tag,'clean curve/rate RMSE',rms(e[k,:-1]),rms(e[k,-1]),'worst horizon',max(np.sqrt(np.mean(e[k,:-1]**2,axis=0))))
 for lab in range(4):
  k=tr['lab_id']==lab; j=(~k)&usable; q=m._fit(sub(ti,j),y[j]); e=m._design(q[:2],*sub(ti,k))[0]@q[2:]-y[k]; v=complete[k]
  print(label,'LOLO',lab,'curve/rate MdAE all',np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])),'complete',np.median(abs(e[v,:-1])),np.median(abs(e[v,-1])))
cal=d['calibration_load_cells']; idx=np.flatnonzero(cal['lab_id']==0)
for tag,keep,spike in [('one-point',np.arange(len(cal['lab_id']))!= -1,False),('EMI',np.ones(len(cal['lab_id']),bool),True)]:
 if not spike: keep[idx[1:]]=False
 q={k:np.asarray(v)[keep].copy() for k,v in cal.items()}
 if spike: q['true_stress'][idx[-1]]+=10
 a=m._calibration(q); assert np.isfinite(a).all() and np.allclose(a[1:],c[1:]); print('calibration',tag,'lab 0 coefficient difference',a[0]-c[0]);
 if spike: assert np.max(abs(a[0]-c[0]))<.002
PY
```
- **Container Exit Code**: `0` (`1011 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
934952368228 0.034914143117073665 worst horizon 0.061719717902671015
all lab 3 clean curve/rate RMSE 0.01779252825501739 0.012978087292237992 worst horizon 0.02546644948214019
all LOLO 0 curve/rate MdAE all 0.014877422664787948 0.015356455253913603 complete 0.011611323480124891 0.012500471008602965
all LOLO 1 curve/rate MdAE all 0.01371841389209276 0.010866861502446479 complete 0.012844529296590426 0.01062807710617375
all LOLO 2 curve/rate MdAE all 0.013336083626021011 0.016798432973114785 complete 0.01203557931129709 0.014227138529512295
all LOLO 3 curve/rate MdAE all 0.020366214414389683 0.018710096911977114 complete 0.01791475737127335 0.013145991448332034
complete-only tau/hw/rw relative L2 0.015716 0.008221 0.005488
complete-only all clean curve/rate RMSE 0.03600008056161274 0.027664241942870208 worst horizon 0.05006590852039815
complete-only complete clean curve/rate RMSE 0.010778012389238457 0.008279328996781717 worst horizon 0.015241986917538617
complete-only missing clean curve/rate RMSE 0.11068724072345221 0.0850602866562481 worst horizon 0.15370387734323515
complete-only lab 0 clean curve/rate RMSE 0.025983290484593375 0.019455300289798 worst horizon 0.036278022353701346
complete-only lab 1 clean curve/rate RMSE 0.045772422179420666 0.035165833761758525 worst horizon 0.06361113630123988
complete-only lab 2 clean curve/rate RMSE 0.04493628265795272 0.034915454670679136 worst horizon 0.06231577707642709
complete-only lab 3 clean curve/rate RMSE 0.018117963002977966 0.013500308568198678 worst horizon 0.025615573171026976
complete-only LOLO 0 curve/rate MdAE all 0.015104723674809473 0.014558352542785258 complete 0.011197453948108116 0.012654182106365708
complete-only LOLO 1 curve/rate MdAE all 0.013708566430796654 0.010427523805731302 complete 0.012432935462950628 0.009964440416120546
complete-only LOLO 2 curve/rate MdAE all 0.012501675683333532 0.01529128421213688 complete 0.011739249968378163 0.01370106207005195
complete-only LOLO 3 curve/rate MdAE all 0.021455613702769494 0.019444631671326484 complete 0.01787012433800278 0.012234839667925629
calibration one-point lab 0 coefficient difference [-0.28999268 -0.11998819]
calibration EMI lab 0 coefficient difference [-2.16994531e-06 -3.13932559e-06]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=456a88111013bf3f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_456a8811 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`32922 ms` LLM)
- **Rationale**: Apply the tested fit-selection change without altering the prediction law or imputation. Verify the exact artifact in isolated Python; if any parameter or slice regresses beyond the observed comparison margins, inspect the integration before submitting.
- **Alternatives Weighed**: `Keep all coupons in the fit` (rejected: Complete-only fitting improves all three parameter estimates, while pooled and worst-slice prediction errors are essentially unchanged.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c03_batch_01_solver.py')
s = p.read_text()
old = '    p = _fit(_inputs(dataset, train, x[:n], cal), y)'
new = '''    # Estimate material parameters from measured features, not cell-mean substitutes.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    inputs = _inputs(dataset, train, x[:n], cal)
    fit_inputs = tuple(v[complete] if i in (0, 1, 2, 5, 6) else v
                       for i, v in enumerate(inputs))
    p = _fit(fit_inputs, y[complete])'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root+'alpha_m97_c03_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root+'public_alpha_m97_c03_batch_01.pkl','rb') as f:
    p = pickle.load(f)[0]
d, g = p['dataset'], p['gt']
out = m.solve_alpha_m97_c03_batch_01(d)
limits = [('estimated_taus','taus',.020), ('hereditary_weights','theta',.010), ('regime_mode_weights','vp_theta',.007)]
for key, ref, limit in limits:
    err = np.linalg.norm(out[key]-g[ref])/np.linalg.norm(g[ref])
    print(key, 'relative L2', round(err, 7))
    assert err < limit
tr, te = d['train'], d['test']
x = m._align(tr, te)
cal = m._calibration(d['calibration_load_cells'])
active = m._inputs(d, te, x[len(tr['lab_id']):], cal)[5]
complete = np.isfinite(te['coupon_features_raw']).all(axis=1)
e = out['predicted_strain'] - g['te_clean']
r = out['initial_rate_pred'] - g['te_true_rate0']
slices = [('pooled', np.ones(len(r),bool), .037,.029,.052), ('complete', complete,.012,.010,.017), ('missing',~complete,.112,.087,.155), ('active',active,.044,.034,.061), ('inactive',~active,.018,.014,.025)]
slices += [(f'lab {j}',te['lab_id']==j,.047,.037,.065) for j in range(4)]
for name, mask, cmax, rmax, hmax in slices:
    c = np.sqrt(np.mean(e[mask]**2)); rr = np.sqrt(np.mean(r[mask]**2)); h = np.sqrt(np.mean(e[mask]**2,axis=0)).max()
    print(name, 'curve/rate/worst-horizon RMSE', *(round(v,7) for v in (c,rr,h)))
    assert c<cmax and rr<rmax and h<hmax
PY
```
- **Container Exit Code**: `0` (`971 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
estimated_taus relative L2 0.015716
hereditary_weights relative L2 0.008221
regime_mode_weights relative L2 0.0054882
pooled curve/rate/worst-horizon RMSE 0.0360001 0.0276642 0.0500659
complete curve/rate/worst-horizon RMSE 0.010778 0.0082793 0.015242
missing curve/rate/worst-horizon RMSE 0.1106872 0.0850603 0.1537039
active curve/rate/worst-horizon RMSE 0.042656 0.0328351 0.0593031
inactive curve/rate/worst-horizon RMSE 0.0157779 0.0118172 0.0220468
lab 0 curve/rate/worst-horizon RMSE 0.0259833 0.0194553 0.036278
lab 1 curve/rate/worst-horizon RMSE 0.0457724 0.0351658 0.0636111
lab 2 curve/rate/worst-horizon RMSE 0.0449363 0.0349155 0.0623158
lab 3 curve/rate/worst-horizon RMSE 0.018118 0.0135003 0.0256156

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=330ff4761d9401f0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_330ff476 (directory).
```

### Turn 11 — `submit_completed_task` (`14393 ms` LLM)
- **Rationale**: The exact final artifact passed isolated schema and public slice checks. Complete-coupon fitting improved recovery of all three parameter groups without materially worsening prediction errors; leave-one-lab-out comparisons and calibration stress checks were already run.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_01_solver.py': 'FILE:7488'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_01_solver.py`. The isolated public schema check passed. Public strain and initial-rate RMSE were 0.0360 and 0.0277 overall; coupons with missing features had the highest errors, at 0.1107 and 0.0851.`

### Turn 12 — `execute_in_remote_docker` (`40556 ms` LLM)
- **Rationale**: An independent recurrence can catch a missing coefficient, sign, or rate term in either damage slice. Held-out signed residuals can distinguish a reproducible lab or time bias, which would justify a targeted challenger, from scattered measurement outliers, which would not justify adding constitutive terms.
- **Alternatives Weighed**: `Probe relay, classifier, and PDE operators` (rejected: The dataset has continuous rheology responses and no relay modes, class labels, or PDE observations. The relevant checks are the memory recurrence, the zero-time rate, and held-out residuals within lab and stratum.)

```bash
cat > /tmp/rheology_gate_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root+'alpha_m97_c03_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root+'public_alpha_m97_c03_batch_01.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
n = len(tr['lab_id'])
x = m._align(tr, te)
cal = m._calibration(d['calibration_load_cells'])
ti = m._inputs(d, tr, x[:n], cal)
vi = m._inputs(d, te, x[n:], cal)
y = np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None], tr['instant_rate_obs']]
complete = np.isfinite(tr['coupon_features_raw']).all(1)
def subset(inp, keep):
    return tuple(v[keep] if j in (0,1,2,5,6) else v for j,v in enumerate(inp))

out = m.solve_alpha_m97_c03_batch_01(d)
tau, w, rw = (out[k] for k in ('estimated_taus','hereditary_weights','regime_mode_weights'))
stress, hold, feat, dt, times, active, _, campaign = vi
memory = np.zeros((len(hold),2))
damage = np.zeros(len(hold))
for step, step_dt in enumerate(dt):
    a = np.exp(-step_dt/tau)
    memory = a*memory + (1-a)*stress[:,step,None]
    damage += np.maximum(0, abs(stress[:,step])-d['crit_stress'])**2*step_dt
activation = np.tanh(damage)*(damage>0)*campaign
weights = (hold[:,None]-memory)*w[:2]
weights[:,0] += rw[0]*activation*hold
terminal_weight = w[2]+feat@w[3:]+rw[1]*activation*memory[:,0]
clock = np.log1p(times) if campaign else times
curve = te['strain_0'][:,None] + weights@(-np.expm1(-times[None,:]/tau[:,None])) + terminal_weight[:,None]*clock
rate = weights@(1/tau) + terminal_weight
for label, keep in [('active',active),('inactive',~active)]:
    if keep.any():
        e = np.max(abs(curve[keep]-out['predicted_strain'][keep])); r = np.max(abs(rate[keep]-out['initial_rate_pred'][keep]))
        print('independent recurrence and rate',label,keep.sum(),e,r)
        assert max(e,r) < 1e-10

# Keep the mandated panels and calibrations available, but hold out all response labels from each lab.
e = np.empty_like(y)
for lab in np.unique(tr['lab_id']):
    test = tr['lab_id']==lab
    fit = ~test & complete
    q = m._fit(subset(ti,fit), y[fit])
    e[test] = m._design(q[:2],*subset(ti,test))[0]@q[2:]-y[test]
for lab, stratum in np.unique(np.c_[tr['lab_id'],tr['stratum']],axis=0):
    k = complete & (tr['lab_id']==lab) & (tr['stratum']==stratum)
    if k.any():
        print('OOF complete cell',lab,stratum,k.sum(),'signed curve-end/rate medians',np.round(np.median(e[k][:,[-2,-1]],axis=0),5),'absolute curve/rate medians',round(np.median(abs(e[k,:-1])),5),round(np.median(abs(e[k,-1])),5))
curve_e = e[complete,:-1]
print('OOF complete abs curve error quantiles by time',np.round(np.quantile(abs(curve_e),[.5,.9,.95,.99],axis=0),4))
print('OOF complete gross residual counts per coupon',tuple(v.tolist() for v in np.unique((abs(curve_e)>.25).sum(1),return_counts=True)))
print('OOF complete signed median by horizon and rate',np.round(np.median(e[complete],axis=0),5))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_gate_audit.py
```
- **Container Exit Code**: `0` (`930 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
independent recurrence and rate active 48 8.881784197001252e-16 1.3322676295501878e-15
independent recurrence and rate inactive 24 6.661338147750939e-16 4.440892098500626e-16
OOF complete cell 0 0 9 signed curve-end/rate medians [-0.00037 -0.00647] absolute curve/rate medians 0.00987 0.00936
OOF complete cell 0 1 11 signed curve-end/rate medians [ 0.0041  -0.00191] absolute curve/rate medians 0.0116 0.0107
OOF complete cell 0 2 7 signed curve-end/rate medians [0.00746 0.0184 ] absolute curve/rate medians 0.0112 0.03212
OOF complete cell 1 0 9 signed curve-end/rate medians [-0.00909  0.0004 ] absolute curve/rate medians 0.01098 0.01837
OOF complete cell 1 1 24 signed curve-end/rate medians [-0.00337 -0.00263] absolute curve/rate medians 0.01277 0.00994
OOF complete cell 1 2 12 signed curve-end/rate medians [-0.00176 -0.0028 ] absolute curve/rate medians 0.01248 0.009
OOF complete cell 2 0 16 signed curve-end/rate medians [ 5.88e-03 -8.00e-05] absolute curve/rate medians 0.01299 0.01317
OOF complete cell 2 1 14 signed curve-end/rate medians [-0.00013 -0.01181] absolute curve/rate medians 0.01206 0.01849
OOF complete cell 2 2 23 signed curve-end/rate medians [-0.00222 -0.01196] absolute curve/rate medians 0.01129 0.01363
OOF complete cell 3 0 5 signed curve-end/rate medians [ 0.00432 -0.00874] absolute curve/rate medians 0.01333 0.01953
OOF complete cell 3 1 10 signed curve-end/rate medians [-0.00415  0.00912] absolute curve/rate medians 0.01251 0.01214
OOF complete cell 3 2 17 signed curve-end/rate medians [ 0.00475 -0.00443] absolute curve/rate medians 0.02439 0.01145
OOF complete abs curve error quantiles by time [[0.0108 0.0113 0.0108 0.0129 0.0116 0.0144 0.0145 0.0125 0.016  0.013 ]
 [0.0694 0.034  0.0383 0.053  0.0346 0.0559 0.0608 0.0337 0.0464 0.0383]
 [1.1538 0.2919 0.0516 0.478  0.1299 0.472  0.5209 0.0634 0.066  0.0618]
 [6.9189 7.6684 8.8335 4.6941 1.9267 1.2837 2.3474 0.46   1.2419 0.4519]]
OOF complete gross residual counts per coupon ([0, 1, 2, 3], [86, 60, 10, 1])
OOF complete signed median by horizon and rate [-1.44e-03 -2.59e-03  7.20e-04 -1.72e-03 -2.23e-03  1.21e-03  1.78e-03
 -8.00e-05 -1.23e-03  1.01e-03 -4.24e-03]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=330ff4761d9401f0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_330ff476 (directory).
```

### Turn 13 — `submit_completed_task` (`15655 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema and public slice checks. The focused audit matched the independent memory recurrence and zero-time rate within 1.4e-15 in both damage slices and found no systematic residual pattern warranting another edit.
- **Alternatives Weighed**: `Add lab-by-stratum corrections or another constitutive term` (rejected: Held-out signed residuals showed no consistent lab-by-stratum or horizon pattern that would justify either change.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_01_solver.py`. The isolated schema check passed, and an independent calculation matched the predicted curves and initial rates to numerical precision in both damage regimes. Public strain and initial-rate RMSE were 0.0360 and 0.0277 overall; coupons with missing features were the worst slice at 0.1107 and 0.0851.`

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
