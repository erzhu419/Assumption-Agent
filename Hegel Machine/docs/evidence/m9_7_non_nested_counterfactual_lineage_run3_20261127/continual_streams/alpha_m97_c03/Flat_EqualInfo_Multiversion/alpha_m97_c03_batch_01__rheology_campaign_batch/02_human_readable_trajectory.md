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
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `329,461` (`317,388` / `96,384` / `12,073`) |
| **Total Episode Wall Time (s)** | `385.34s` |

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

### Turn 1 — `execute_in_remote_docker` (`21239 ms` LLM)
- **Rationale**: Inspect the schema, array shapes, and prior implementation to identify how references and regime metadata are stored and which alignment or fitting routines can be reused.
- **Alternatives Weighed**: `Reuse the archived solver unchanged` (rejected: The current plateau law and data contract must be checked before carrying forward any prior fitting logic.)

```bash
cd /holdout_task && ls -l && cat validate_public_schema.py && cat public_asset_index.json && ls -l /tmp/hegel_continual_archive && sed -n '1,250p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation variable projection' ; cd /holdout_task && python3 - <<'PY'
import pickle
import numpy as np
with open('public_alpha_m97_c03_batch_01.pkl','rb') as f: d=pickle.load(f)
def show(x,k='',dep=0):
 if isinstance(x,dict) and dep<4:
  for a,b in x.items(): show(b,k+'/'+str(a),dep+1)
 elif isinstance(x,(np.ndarray,list,tuple)):
  y=np.asarray(x)
  print(k,type(x).__name__,y.shape,y.dtype,repr(y.reshape(-1)[:6])[:500])
 else: print(k,type(x).__name__,repr(x)[:250])
show(d)
PY
```
- **Container Exit Code**: `0` (`513 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, t)
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2),
            'predicted_strain': baseline[n:, None] + a @ weights,
            'initial_rate_pred': b @ weights}
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
 list (1,) object array([{'dataset': {'past_dt': array([0.4917, 0.5023, 0.5442, 0.7304, 0.4565, 0.765 , 0.2965, 0.6942,
              0.5869, 0.3435, 0.3902, 0.5082, 0.3785, 0.7877, 0.3776, 0.2105,
              0.2875, 0.2881]), 'eval_times': array([0.3   , 0.8444, 1.3889, 1.9333, 2.4778, 3.0222, 3.5667, 4.1111,
              4.6556, 5.2   ]), 'crit_stress': 2.65, 'yield_stress': 2.65, 'shear_banding_index': 0.12, 'calibration_load_cells': {'lab_id': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1
```

### Turn 2 — `execute_in_remote_docker` (`20844 ms` LLM)
- **Rationale**: The first inspection summarized the pickle wrapper rather than the dataset. Inspect the nested panel and outcome fields, then measure per-channel calibration residuals to choose the required calibration path before fitting.
- **Alternatives Weighed**: `Use the prior gate's load-cell coefficients directly` (rejected: The task requires fitting the current batch when it has enough clean points; the residuals will show whether the archive fallback is needed.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
items=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb')); print('items:',len(items),'item keys:',list(items[0]))
d=items[0]['dataset']
for k,v in d.items():
 print(k)
 for name,a in (v.items() if isinstance(v,dict) else [(k,v)]):
  x=np.asarray(a); print(' ',name,x.shape,x.dtype,('nan='+str(np.isnan(x).sum()) if x.dtype.kind=='f' else ''), x.ravel()[:4] if x.dtype.kind!='O' else type(a).__name__)
c=d['calibration_load_cells']
for lab in range(4):
 m=c['lab_id']==lab;x=c['cmd_stress'][m];y=c['true_stress'][m];p=np.polyfit(x,y,1);print('calibration',lab,len(x),p,np.round(y-np.polyval(p,x),6))
PY
sed -n '1,150p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py
python3 - <<'PY'
import json
p='/tmp/hegel_continual_archive/archive_calibration_history.json'; d=json.load(open(p)); print('history structure:',str(d)[:2300])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -45
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy IRLS Prony variable projection' | head -12
```
- **Container Exit Code**: `0` (`499 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
e[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray",
          "doc_summary": "Return signal values in control-referenced coordinates, in input row order."
        }
      ],
      "index_cards": [
        "autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel",
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 3 — `execute_in_remote_docker` (`16886 ms` LLM)
- **Rationale**: The previous output truncated the archived alignment and history code. Recover those definitions to avoid using the wrong panel coordinates or fallback mapping, and identify public review fields and lab-by-stratum coverage for validation without using ground truth in the solver.
- **Alternatives Weighed**: `Reuse the prior constitutive law` (rejected: Only calibration and panel preparation are candidates for reuse; the current task specifies a different time dependence.)

```bash
sed -n '1,110p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
h=json.load(open(p)); print('CAL HISTORY',str(h)[:2100])
i=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d=i['dataset']; print('PUBLIC GT FIELDS',[(k,np.shape(v)) for k,v in i['gt'].items()]); print('CELL COUNTS',[(b,s,int(np.sum((d['train']['lab_id']==b)&(d['train']['stratum']==s)))) for b in range(4) for s in range(3)]); print('SCIENCE STACK'); import scipy; print(scipy.__version__)
PY
```
- **Container Exit Code**: `0` (`407 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ned('stratum')
    ref = joined('ref_coupons')
    aligned = joined('coupon_features_raw').astype(float)
    for c in np.unique(lab):
        rows = lab == c
        panel = ref[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        aligned[rows] = (aligned[rows] - center)/scale
    cells = np.column_stack((lab, stratum))
    means = []
    for cell in np.unique(cells, axis=0):
        values = aligned[np.all(cells == cell, axis=1)]
        count = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), count,
                               out=np.full(aligned.shape[1], np.nan), where=count > 0))
    aligned = np.where(np.isnan(aligned), np.nanmean(means, axis=0), aligned)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = joined('past_stress_cmd') * gain[:, None] + bias[:, None]
    hold = joined('stress_hold_cmd') * gain + bias
CAL HISTORY [{'episode_short_name': 'alpha_m97_c03_s00', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [0.893607, -0.483769, -1.39264, 1.611445, -1.810815, -1.870654, 1.48082, 1.268821, -0.114529, 0.252739, -1.204476, -0.664616, -0.154159, -0.613549, -1.259125, -2.011382, 2.18392, -0.23172, 1.98347, -2.000907, -0.486537, -0.139492, -0.631342, -1.874772, -0.708257, 2.13352, 0.386634, 1.10423, -1.407517, 1.70372, 1.521295, 1.987385, 1.659545, -0.294044, 0.359165, -0.017198, 1.319848, 0.959274, 1.866895, -0.480321], 'true_stress': [0.893713, -0.483774, -1.392656, 1.611564, -1.810667, -1.870622, 1.480773, 1.268837, -0.114695, 0.252685, -1.204516, -0.664704, -0.154113, -0.613571, -1.259248, -2.011128, 2.183777, -0.231673, 1.983509, -2.001075, -0.486729, -0.139381, -0.631253, -1.874747, -0.708398, 2.133538, 0.386675, 1.104402, -1.407596, 1.703866, 1.521337, 1.987399, 1.659449, -0.294029, 0.359147, -0.017404, 1.319703, 0.959321, 1.866928, -0.480472]}]
PUBLIC GT FIELDS [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (72, 10)), ('te_true_rate0', (72,)), ('te_hist_true', (72, 18)), ('te_hold_true', (72,))]
CELL COUNTS [(0, 0, 12), (0, 1, 13), (0, 2, 10), (1, 0, 12), (1, 1, 25), (1, 2, 12), (2, 0, 16), (2, 1, 15), (2, 2, 26), (3, 0, 8), (3, 1, 12), (3, 2, 19)]
SCIENCE STACK
1.13.0
```

### Turn 4 — `execute_in_remote_docker` (`81913 ms` LLM)
- **Rationale**: Implement the specified stress, memory, and derivative equations and fit only training observations. Measure curve and initial-rate errors separately against the public reference to catch an incorrect time basis or branch assignment.
- **Alternatives Weighed**: `Apply the logarithmic plateau to every row` (rejected: The instruction makes activation conditional. Compare the inactive rows before deciding whether the condition gates the time basis or only the regime terms.)

```bash
cat > /holdout_task/alpha_m97_c03_batch_01_solver.py <<'PY'
"""Fit a calibrated two-branch creep model to a rheology campaign."""
import json

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _prior_calibration(lab):
    with open(_ARCHIVE) as stream:
        history = json.load(stream)
    # The prefix is chronological; use the most recent usable sweep.
    for sweep in reversed(history):
        rows = np.asarray(sweep['lab_id']) == lab
        x = np.asarray(sweep['cmd_stress'])[rows]
        y = np.asarray(sweep['true_stress'])[rows]
        clean = np.isfinite(x) & np.isfinite(y)
        if clean.sum() >= 2 and np.ptp(x[clean]) > 0:
            return np.polyfit(x[clean], y[clean], 1)
    raise ValueError(f'No calibration sweep for lab {lab}')


def _calibration(table):
    fits = []
    for lab in range(4):
        rows = (table['lab_id'] == lab) & np.isfinite(table['cmd_stress']) & np.isfinite(table['true_stress'])
        x = table['cmd_stress'][rows]
        y = table['true_stress'][rows]
        if len(x) < 2:
            fits.append(_prior_calibration(lab))
            continue
        fit = np.polyfit(x, y, 1)
        if np.max(np.abs(y - np.polyval(fit, x))) > 0.01:
            # Find the tight affine consensus before testing the 3-unit EMI threshold.
            candidates = []
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    if x[i] == x[j]:
                        continue
                    gain = (y[j] - y[i]) / (x[j] - x[i])
                    bias = y[i] - gain * x[i]
                    e = np.abs(y - (gain * x + bias))
                    clean = e <= 0.01
                    if np.any(e > 3.0) and np.all(clean | (e > 3.0)):
                        candidates.append(clean)
            if candidates:
                largest = max(mask.sum() for mask in candidates)
                masks = np.unique([mask for mask in candidates if mask.sum() == largest], axis=0)
                if largest >= 4 or len(masks) == 1:
                    clean = masks[0]
                else:
                    prior = _prior_calibration(lab)
                    e = np.abs(y - np.polyval(prior, x))
                    clean = min(masks, key=lambda mask: np.mean(e[mask] ** 2))
                fit = np.polyfit(x[clean], y[clean], 1)
        fits.append(fit)
    return np.asarray(fits)


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']

    def join(key):
        return np.concatenate((train[key], test[key]))

    lab, stratum = join('lab_id'), join('stratum')
    ref, features = join('ref_coupons'), join('coupon_features_raw').astype(float)
    cells = []
    for c in np.unique(lab):
        rows = lab == c
        panel = ref[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        features[rows] = (features[rows] - center) / scale
        for s in np.unique(stratum[rows]):
            cell = features[rows & (stratum == s)]
            count = np.isfinite(cell).sum(axis=0)
            cells.append(np.divide(np.nansum(cell, axis=0), count,
                                   out=np.full(features.shape[1], np.nan), where=count > 0))
    features = np.where(np.isnan(features), np.nanmean(cells, axis=0), features)
    gain, bias = _calibration(dataset['calibration_load_cells'])[lab].T
    past = join('past_stress_cmd') * gain[:, None] + bias[:, None]
    hold = join('stress_hold_cmd') * gain + bias
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    damage = np.maximum(np.abs(past) - dataset['crit_stress'], 0.0) ** 2 @ dt
    active = (dataset['shear_banding_index'] <= 0.5) & (damage > 0)
    act = np.where(active, np.tanh(damage), 0.0)
    return past, hold, features, act, active, join('strain_0'), dt, t


def _basis(logtaus, past, hold, features, act, active, dt, t, plateau='gated'):
    tau = np.exp(logtaus)
    u, v = dt[:, None] / tau, t[:, None] / tau
    decay, relax = np.exp(-u), -np.expm1(-v)
    drelax = -v * np.exp(-v)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for step, (a, da) in enumerate(zip(decay, u * decay)):
        stress = past[:, step, None]
        dz = a * dz + da * (z - stress)
        z = a * z + (-np.expm1(-u[step])) * stress
    gap = hold[:, None] - z
    ramp = np.broadcast_to(t, (len(hold), len(t)))
    if plateau == 'all':
        ramp = np.broadcast_to(np.log1p(t), ramp.shape)
    elif plateau == 'gated':
        ramp = np.where(active[:, None], np.log1p(t), ramp)
    ramp = np.column_stack((ramp, np.ones(len(hold))))
    relax = np.vstack((relax, 1 / tau))
    drelax = np.vstack((drelax, -1 / tau))
    design = np.empty((len(hold), len(t) + 1, 9))
    design[:, :, :2] = gap[:, None, :] * relax
    design[:, :, 2:7] = ramp[:, :, None] * np.column_stack((np.ones(len(hold)), features))[:, None, :]
    design[:, :, 7] = (act * hold)[:, None] * relax[:, 0]
    design[:, :, 8] = (act * z[:, 0])[:, None] * ramp
    dhered = gap[:, None, :] * drelax - dz[:, None, :] * relax
    dreg0 = (act * hold)[:, None] * drelax[:, 0]
    dreg1 = (act * dz[:, 0])[:, None] * ramp
    return design, dhered, dreg0, dreg1


def _scale(error):
    return max(1.4826 * np.median(np.abs(error - np.median(error))), 1e-7)


def _fit(args, target, plateau='gated'):
    def model(p, jac=False):
        a, dh, d0, d1 = _basis(p[:2], *args, plateau)
        a = a.reshape(-1, 9)
        if not jac:
            return np.einsum('ij,j->i', a, p[2:]) - target
        h = dh.reshape(-1, 2) * p[2:4]
        h[:, 0] += d0.ravel() * p[9] + d1.ravel() * p[10]
        return np.column_stack((h, a))

    lower = np.r_[np.log(0.015), np.log(0.015), np.full(9, -np.inf)]
    upper = np.r_[np.log(120.0), np.log(120.0), np.full(9, np.inf)]
    starts = []
    for pair in ((0.25, 2.0), (0.65, 4.5), (1.5, 8.0), (3.0, 0.45), (7.0, 1.0)):
        logs = np.log(pair)
        a = _basis(logs, *args, plateau)[0].reshape(-1, 9)
        starts.append(np.r_[logs, np.linalg.lstsq(a, target, rcond=None)[0]])

    def fit(start, loss, scale):
        return least_squares(model, start, jac=lambda p: model(p, True), bounds=(lower, upper),
                             loss=loss, f_scale=scale, x_scale='jac', max_nfev=160)

    scale = min(_scale(model(p)) for p in starts)
    best = min((fit(p, 'soft_l1', scale) for p in starts), key=lambda result: result.cost)
    for _ in range(2):
        best = fit(best.x, 'cauchy', 2.0 * _scale(best.fun))
    return np.exp(best.x[:2]), best.x[2:]


def solve_alpha_m97_c03_batch_01(dataset: dict) -> dict:
    past, hold, features, act, active, offset, dt, t = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    grouped = (past, hold, features, act, active)
    target = np.column_stack((train['strain_curves_obs'] - offset[:n, None], train['instant_rate_obs']))
    taus, weights = _fit((*[x[:n] for x in grouped], dt, t), target.ravel())
    basis = _basis(np.log(taus), *[x[n:] for x in grouped], dt, t)[0]
    pred = basis @ weights
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': offset[n:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
PY
cat > /tmp/check_rheology01.py <<'PY'
import importlib.util, pickle, numpy as np
base='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',base+'alpha_m97_c03_batch_01_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
i=pickle.load(open(base+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d=i['dataset']; g=i['gt']; n=len(d['train']['lab_id'])
p,h,x,a,act,off,dt,t=m._prepare(d); args=(p,h,x,a,act); yy=np.c_[d['train']['strain_curves_obs']-off[:n,None],d['train']['instant_rate_obs']]
print('active',act[:n].sum(),act[n:].sum(),'stress errors',np.max(np.abs(p[n:]-g['te_hist_true'])),np.max(np.abs(h[n:]-g['te_hold_true'])))
for key in ('taus','theta','vp_theta'): print('reference',key,g[key])
for mode in ('gated','all','linear'):
 tau,w=m._fit((*[v[:n] for v in args],dt,t),yy.ravel(),mode)
 z=m._basis(np.log(tau),*args,dt,t,mode)[0]@w
 err=z[:n]-yy; ee=z[n:,:-1]+off[n:,None]-g['te_clean']; er=z[n:,-1]-g['te_true_rate0']
 print(mode,'tau',tau,'w',w.round(6),'train median abs',np.median(abs(err[:,:-1])),np.median(abs(err[:,-1])),'p95',np.quantile(abs(err),.95),'test rmse',np.sqrt(np.mean(ee**2)),np.sqrt(np.mean(er**2)))
 for b in (True,False):
  mask=act[n:]==b
  if mask.any(): print(' active',b,'max errors',abs(ee[mask]).max(),abs(er[mask]).max())
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check_rheology01.py
```
- **Container Exit Code**: `0` (`890 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
active 120 48 stress errors 0.00020877658787910036 0.00010832826309536969
reference taus [0.568 4.113]
reference theta [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09]
reference vp_theta [0.28 0.24]
gated tau [0.57467968 3.94997923] w [ 0.686359  0.532174  0.097456  0.114267 -0.115325  0.078843 -0.056423
  0.287702  0.233596] train median abs 0.07723221230745689 0.0674178736026414 p95 0.6620168271589923 test rmse 0.21684370754739055 0.08527056165588215
 active True max errors 0.46775408234372384 0.24654512145504048
 active False max errors 1.4906441624293052 0.24414018824504446
all tau [0.56699021 3.99366056] w [ 0.675642  0.544431  0.138228  0.15662  -0.128375  0.115736 -0.088606
  0.281907  0.23881 ] train median abs 0.013756695792404072 0.014889249124321768 p95 0.33390456622983355 test rmse 0.03714862043196379 0.028575607559410004
 active True max errors 0.2779851375757638 0.15367739116779996
 active False max errors 0.12108715406059956 0.06257405386157444
linear tau [0.575966   4.23565928] w [ 0.665988  0.559939  0.056216  0.063931 -0.052843  0.047462 -0.036228
  0.30232   0.093279] train median abs 0.04614045493410962 0.12591915924398991 p95 0.3842931904464925 test rmse 0.07884746825489228 0.18801992648826987
 active True max errors 0.3093278258527672 0.453883325638071
 active False max errors 0.2657385743548749 0.5124023744687969

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=51d153c0d3488a53 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_51d153c0 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`25615 ms` LLM)
- **Rationale**: Leave one lab out to test whether the time-basis improvement generalizes. Separately, evaluate the formula at published reference coefficients to distinguish remaining alignment error from parameter-estimation error; those coefficients will not enter the solver.
- **Alternatives Weighed**: `Tighten robust fitting without changing the time basis` (rejected: Fitting cannot repair the observed inactive-row time-basis mismatch.)
- **ATMS NOGOOD DECLARED**: Refuted `Zero damage switches the base drift from logarithmic to linear within the crystalline campaign.` | Witness: `Applying the logarithmic drift throughout the campaign reduced the maximum inactive-row curve error from 1.491 to 0.121 and reduced the training curve median absolute residual from 0.0772 to 0.0138.` | Blanket Diagnosis: `The sample-level damage gate was applied to the campaign-level time basis as well as to the damage-dependent correction.` | Sublated Pivot: `Use the logarithmic drift in the low-banding campaign and retain the damage gate on the two regime corrections; confirm on lab-held-out observations.`

```bash
cat > /tmp/diagnose_rheology01.py <<'PY'
exec(open('/tmp/check_rheology01.py').read().split("print('active'")[0])
wt=np.r_[g['theta'],g['vp_theta']]; tau=g['taus']; l=d['test']['lab_id']; miss=np.isnan(d['test']['coupon_features_raw']); b=m._basis(np.log(tau),*args,dt,t,'all')[0]@wt
err=b[n:,:-1]+off[n:,None]-g['te_clean']; er=b[n:,-1]-g['te_true_rate0']
for lab in range(4):
 for missing in (False,True):
  k=(l==lab)&(miss.any(1)==missing)
  if k.any(): print('oracle coefficient audit',lab,missing,k.sum(),'strain/rate max',abs(err[k]).max(),abs(er[k]).max(),'rate mean',er[k].mean())
print('rate vs curve drift residual',np.max(abs(err-er[:,None]*np.log1p(t))))
for j in np.flatnonzero(miss.any(1)):
 print('missing',j,'lab',l[j],'stratum',d['test']['stratum'][j],'columns',np.where(miss[j])[0], 'drift error',er[j])
for mode in ('gated','all'):
 for lab in range(4):
  keep=d['train']['lab_id']!=lab
  tau,w=m._fit((*[v[:n][keep] for v in args],dt,t),yy[keep].ravel(),mode)
  pred=m._basis(np.log(tau),*[v[:n][~keep] for v in args],dt,t,mode)[0]@w
  e=pred-yy[~keep]
  print('LOLO',mode,lab, 'taus',tau.round(4),'curve/rate MAE medians',np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/diagnose_rheology01.py
```
- **Container Exit Code**: `0` (`1323 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
28 0.012088449839527171 rate mean 0.0011851609138507691
oracle coefficient audit 0 True 2 strain/rate max 0.17890135568254384 0.09804146832607596 rate mean 0.012162319895605567
oracle coefficient audit 1 False 14 strain/rate max 0.0021369619054545375 0.0011890942078973676 rate mean 0.00034932678068946003
oracle coefficient audit 1 True 2 strain/rate max 0.28394718431114985 0.15561661471558652 rate mean -0.08240724997858634
oracle coefficient audit 2 False 19 strain/rate max 0.013393545448926847 0.007363767834206669 rate mean 0.0034259052331282545
oracle coefficient audit 2 True 2 strain/rate max 0.15953936182071354 0.08745038435058858 rate mean 0.08148933934972195
oracle coefficient audit 3 False 17 strain/rate max 0.07407833704281663 0.0405935111921405 rate mean 0.002322792243164923
oracle coefficient audit 3 True 1 strain/rate max 0.09291482573106899 0.05092305417062093 rate mean -0.05092305417062093
rate vs curve drift residual 0.00022473298032595198
missing 16 lab 2 stratum 1 columns [0] drift error 0.07552829434885533
missing 18 lab 1 stratum 1 columns [1] drift error -0.15561661471558652
missing 27 lab 0 stratum 2 columns [3] drift error 0.09804146832607596
missing 44 lab 1 stratum 1 columns [0] drift error -0.009197885241586157
missing 45 lab 3 stratum 2 columns [0 3] drift error -0.05092305417062093
missing 47 lab 2 stratum 2 columns [2] drift error 0.08745038435058858
missing 62 lab 0 stratum 1 columns [3] drift error -0.07371682853486483
LOLO gated 0 taus [0.5844 3.7803] curve/rate MAE medians 0.09150627169345804 0.0751980805628184
LOLO gated 1 taus [0.5742 3.6222] curve/rate MAE medians 0.08580722437861983 0.062462585151774164
LOLO gated 2 taus [0.5659 4.2225] curve/rate MAE medians 0.07164724052179462 0.055012596774300926
LOLO gated 3 taus [0.5748 3.7833] curve/rate MAE medians 0.08895538256736402 0.0620167174211228
LOLO all 0 taus [0.5672 3.9909] curve/rate MAE medians 0.014497548459315862 0.015466164692817586
LOLO all 1 taus [0.5663 3.9419] curve/rate MAE medians 0.01328796709060659 0.011701952644725067
LOLO all 2 taus [0.5666 3.99  ] curve/rate MAE medians 0.013036949837404888 0.01532972739163263
LOLO all 3 taus [0.5681 4.0482] curve/rate MAE medians 0.0217295510668281 0.019742802209875232

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=51d153c0d3488a53 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_51d153c0 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`32724 ms` LLM)
- **Rationale**: The logarithmic basis improved both outputs in every held-out lab. Apply it to the solver and run the isolated schema validator. Check whether training residuals cluster by specimen or by measurement type; that distinction determines whether the fit should downweight whole specimens or individual observations.
- **Alternatives Weighed**: `Replace cell-balanced imputation with a learned predictor` (rejected: The contract specifies cell-balanced means; first separate missing-feature prediction error from errors in fitting the shared parameters.)

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c03_batch_01_solver.py')
s = p.read_text()
s = s.replace("    taus, weights = _fit((*[x[:n] for x in grouped], dt, t), target.ravel())\n    basis = _basis(np.log(taus), *[x[n:] for x in grouped], dt, t)[0]", "    plateau = 'all' if dataset['shear_banding_index'] <= 0.5 else 'linear'\n    taus, weights = _fit((*[x[:n] for x in grouped], dt, t), target.ravel(), plateau)\n    basis = _basis(np.log(taus), *[x[n:] for x in grouped], dt, t, plateau)[0]")
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
exec(open('/tmp/check_rheology01.py').read().split("print('active'")[0])
out=m.solve_alpha_m97_c03_batch_01(d)
curve=out['predicted_strain']-g['te_clean']; rate=out['initial_rate_pred']-g['te_true_rate0']
miss=np.isnan(d['test']['coupon_features_raw']).any(1)
rmse=lambda e: np.sqrt(np.mean(e**2))
for label, mask in [('all',np.ones(len(miss),bool)),('complete',~miss),('missing',miss)]+[(f'lab {lab}',d['test']['lab_id']==lab) for lab in range(4)]:
 print(label,'curve/rate RMSE',rmse(curve[mask]),rmse(rate[mask]))
w=np.r_[out['hereditary_weights'],out['regime_mode_weights']]
e=m._basis(np.log(out['estimated_taus']),*[v[:n] for v in args],dt,t,'all')[0]@w-yy
for group,err in [('curves',e[:,:-1]),('rate',e[:,-1:])]:
 print(group,'abs residual quantiles',np.round(np.quantile(abs(err),[.25,.5,.75,.85,.90,.95,.99]),5),'median by horizon',np.round(np.median(abs(err),axis=0),5))
curve_bad=(abs(e[:,:-1])>.10).sum(1); rate_bad=abs(e[:,-1])>.10
print('curve outlier counts per row',np.bincount(curve_bad,minlength=11))
print('rate outliers with/without >=2 curve outliers',np.sum(rate_bad & (curve_bad>=2)),np.sum(rate_bad & (curve_bad<2)))
print('outliers in complete/missing training rows',[(int(mk.sum()),int(np.sum(rate_bad[mk])),int(np.sum(curve_bad[mk]>=2))) for mk in [~np.isnan(d['train']['coupon_features_raw']).any(1),np.isnan(d['train']['coupon_features_raw']).any(1)]])
PY
```
- **Container Exit Code**: `0` (`1028 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
all curve/rate RMSE 0.03714862043196379 0.028575607559410004
complete curve/rate RMSE 0.010856207490681788 0.008524980030695982
missing curve/rate RMSE 0.11445568033686145 0.0878869497203187
lab 0 curve/rate RMSE 0.037139757612113654 0.0281113350875832
lab 1 curve/rate RMSE 0.05131094965332034 0.03957921685681862
lab 2 curve/rate RMSE 0.03305038759197088 0.026038772769459585
lab 3 curve/rate RMSE 0.024549706365990406 0.018342272576715012
curves abs residual quantiles [0.0061  0.01376 0.02787 0.04787 0.10533 0.37168 3.30023] median by horizon [0.01166 0.01105 0.01233 0.01571 0.01224 0.0155  0.01734 0.01273 0.01556
 0.01313]
rate abs residual quantiles [0.0076  0.01489 0.02634 0.0389  0.04299 0.09019 0.21785] median by horizon [0.01489]
curve outlier counts per row [85 67 17  1  0  1  1  2  2  3  1]
rate outliers with/without >=2 curve outliers 8 0
outliers in complete/missing training rows [(157, 0, 17), (23, 8, 11)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d8415e456b3b959e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d8415e45 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`29650 ms` LLM)
- **Rationale**: All eight training rate residuals above 0.10 occurred on coupons with missing features. Compare complete-coupon fitting against the verified baseline in lab-held-out folds, and test sparse and EMI calibration paths that the clean public sweep does not exercise. Retain the baseline unless the challenger improves the worst slice.
- **Alternatives Weighed**: `Change the imputation rule to improve missing-coupon predictions` (rejected: The contract specifies cell-balanced means. Excluding imputed coupons from parameter estimation tests their influence without changing that rule.)

```bash
cat > /tmp/stress_rheology01.py <<'PY'
exec(open('/tmp/check_rheology01.py').read().split("print('active'")[0])
lab=d['train']['lab_id']; miss=np.isnan(d['train']['coupon_features_raw']).any(1)
# Fit on all observations or on complete coupons, leaving each lab out in turn.
for complete_only in (False, True):
    scores=[]
    for held in range(4):
        train=(lab!=held) & (~miss if complete_only else True)
        test=lab==held
        tau,w=m._fit((*[v[:n][train] for v in args],dt,t),yy[train].ravel(),'all')
        err=m._basis(np.log(tau),*[v[:n][test] for v in args],dt,t,'all')[0]@w-yy[test]
        scores.append([np.median(abs(err[:,:-1])), np.median(abs(err[:,-1])), np.median(abs(err[~miss[test],:-1])), np.median(abs(err[~miss[test],-1]))])
    print('complete_only',complete_only,'LOLO per lab: curve, rate, complete curve, complete rate',np.round(scores,5).tolist())
    print('worst per metric',np.round(np.max(scores,axis=0),5))
# Exercise the specified archive branch and in-batch cleaning independently.
cal=d['calibration_load_cells']; baseline=m._calibration(cal)
for held in range(4):
    rows=np.flatnonzero(cal['lab_id']==held)
    mask=(cal['lab_id']!=held); mask[rows[0]]=True
    assert np.allclose(m._calibration({k:v[mask] for k,v in cal.items()})[held],m._prior_calibration(held))
    for ix in rows:
        for sign in (-1,1):
            bad={k:v.copy() for k,v in cal.items()}; bad['true_stress'][ix]+=sign*3.1
            clean=rows[rows!=ix]; expected=np.polyfit(cal['cmd_stress'][clean],cal['true_stress'][clean],1)
            assert np.allclose(m._calibration(bad)[held],expected), (held,ix,sign)
print('Calibration: four sparse-channel archive paths and 80 single-point EMI injections passed')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/stress_rheology01.py
```
- **Container Exit Code**: `0` (`1380 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
complete_only False LOLO per lab: curve, rate, complete curve, complete rate [[0.0145, 0.01547, 0.01105, 0.01261], [0.01329, 0.0117, 0.01278, 0.01128], [0.01304, 0.01533, 0.01241, 0.01408], [0.02173, 0.01974, 0.01791, 0.01319]]
worst per metric [0.02173 0.01974 0.01791 0.01408]
complete_only True LOLO per lab: curve, rate, complete curve, complete rate [[0.01478, 0.01457, 0.01135, 0.01229], [0.01293, 0.01043, 0.0123, 0.01002], [0.01229, 0.01547, 0.01166, 0.01375], [0.02232, 0.01943, 0.01782, 0.01226]]
worst per metric [0.02232 0.01943 0.01782 0.01375]
Calibration: four sparse-channel archive paths and 80 single-point EMI injections passed

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d8415e456b3b959e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d8415e45 (directory).
```

### Turn 8 — `submit_completed_task` (`17105 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated schema validation, held-out lab evaluation, and the calibration stress checks. Retain the all-coupon robust fit and the specified cell-balanced imputation.
- **Alternatives Weighed**: `Fit only on complete coupons` (rejected: It slightly improved the worst held-out lab rate error but worsened the corresponding strain metric, with no clear overall gain.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_01_solver.py': 'FILE:7493'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_01_solver.py`. The public schema passed. Public test RMSE was 0.0371 for strain and 0.0286 for initial rate; missing coupon features were the largest source of error. Sparse calibration and all 80 EMI stress cases passed.`

### Turn 9 — `execute_in_remote_docker` (`37753 ms` LLM)
- **Rationale**: An independent convolution and zero-time derivative can detect a sign, time-index, or rate-handoff error in the two damage states. Held-out lab residuals can reveal a systematic offset worth tracing to calibration or panel alignment. Check coefficient margins and run the public validator in an isolated process.
- **Alternatives Weighed**: `Add rational terms, PDE operators, or relay states` (rejected: This continuous rheology task specifies an exponential memory convolution and logarithmic drift; it has no relay state or PDE. An independent calculation can test the actual algebra without adding unsupported terms.); `Apply classification-style batch balancing` (rejected: Both targets are continuous. Signed residuals in held-out labs test for systematic calibration or alignment bias without assuming class labels.)

```bash
cat > /tmp/audit_rheology01.py <<'PY'
exec(open('/tmp/check_rheology01.py').read().split("print('active'")[0])
out = m.solve_alpha_m97_c03_batch_01(d)
tau = out['estimated_taus']; w = np.r_[out['hereditary_weights'], out['regime_mode_weights']]
# Independent closed-form convolution, using the time elapsed after each past interval.
after = np.cumsum(dt[::-1])[::-1] - dt
z = p @ ((-np.expm1(-dt[:,None]/tau))*np.exp(-after[:,None]/tau))
gap = h[:,None] - z
a0 = w[7]*a*h
slope = w[2] + x @ w[3:7] + w[8]*a*z[:,0]
def curve(t):
    b = -np.expm1(-t[:,None]/tau)
    return (gap*w[:2]) @ b.T + a0[:,None]*b[:,0] + slope[:,None]*np.log1p(t)
rate = (gap*w[:2]) @ (1/tau) + a0/tau[0] + slope
for on in (False, True):
    mask = act[n:] == on
    ec = np.max(abs((curve(t)[n:]+off[n:,None]-out['predicted_strain'])[mask]))
    er = np.max(abs((rate[n:]-out['initial_rate_pred'])[mask]))
    print('identity, damage active',on,'coupon count',mask.sum(),'curve/rate max disagreement',ec,er)
    assert max(ec,er)<1e-11
h0=1e-5
fd = (4*curve(np.array([h0]))[:,0]-curve(np.array([2*h0]))[:,0])/(2*h0)
print('independent zero-time derivative max error',abs(fd-rate).max())
assert abs(fd-rate).max()<1e-8
# Cross-fit by lab so an offset cannot be learned from the lab being scored.
lab=d['train']['lab_id']; complete=~np.isnan(d['train']['coupon_features_raw']).any(1)
for held in np.unique(lab):
    train=lab!=held; obs=(lab==held)&complete
    tf,wf=m._fit((*[v[:n][train] for v in args],dt,t),yy[train].ravel(),'all')
    design=m._basis(np.log(tf),*[v[:n][obs] for v in args],dt,t,'all')[0]
    pred=design@wf; e=pred-yy[obs]
    base=pred-design[:,:,7:]@wf[7:]
    print('held lab',held,'complete n',obs.sum(),'signed median curve/rate',np.round([np.median(e[:,:-1]),np.median(e[:,-1])],6),'median abs full vs omit damage',np.round([np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])),np.median(abs(base[:,:-1]-yy[obs,:-1])),np.median(abs(base[:,-1]-yy[obs,-1]))],6))
print('max relative tau error',abs(tau/g['taus']-1).max(),'max abs hereditary/plateau coefficient error',abs(w[:7]-g['theta']).max(),abs(w[7:]-g['vp_theta']).max())
PY
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I /tmp/audit_rheology01.py
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import json
p='/holdout_task/public_asset_index.json'
def scan(x,path=''):
 if isinstance(x,dict):
  for k,v in x.items():
   if any(s in k.lower() for s in ('toleran','threshold','metric','limit')): print(path+'/'+k,repr(v)[:400])
   if isinstance(v,(dict,list)): scan(v,path+'/'+k)
 elif isinstance(x,list):
  for j,v in enumerate(x): scan(v,path+'/'+str(j))
scan(json.load(open(p)))
PY
```
- **Container Exit Code**: `0` (`1472 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
identity, damage active False coupon count 24 curve/rate max disagreement 8.881784197001252e-16 8.881784197001252e-16
identity, damage active True coupon count 48 curve/rate max disagreement 8.881784197001252e-16 1.7763568394002505e-15
independent zero-time derivative max error 4.6060666392122585e-10
held lab 0 complete n 27 signed median curve/rate [-0.001484 -0.001483] median abs full vs omit damage [0.011046 0.012612 0.16665  0.289597]
held lab 1 complete n 45 signed median curve/rate [-0.003749 -0.003854] median abs full vs omit damage [0.01278  0.011284 0.125963 0.301997]
held lab 2 complete n 53 signed median curve/rate [-0.000376 -0.01023 ] median abs full vs omit damage [0.012409 0.01408  0.112967 0.19677 ]
held lab 3 complete n 32 signed median curve/rate [ 0.004222 -0.005241] median abs full vs omit damage [0.017911 0.013185 0.191789 0.298087]
max relative tau error 0.029015181016623592 max abs hereditary/plateau coefficient error 0.005569357406693176 0.0019073313749609633
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d8415e456b3b959e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d8415e45 (directory).
```

### Turn 10 — `submit_completed_task` (`14595 ms` LLM)
- **Rationale**: The final artifact passed isolated schema validation. An independent convolution agreed with its strain and rate predictions to within 1.8e-15 in both damage states, and the host verified panel affine invariance.
- **Alternatives Weighed**: `Fit only on coupons with complete features` (rejected: It slightly improved the worst-lab rate error but worsened the worst-lab strain error.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_01_solver.py': 'FILE:7493'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_01_solver.py`. The public schema passed. Public test RMSE was 0.0371 for strain and 0.0286 for initial rate. Missing coupon features were the largest source of error. All four sparse-calibration checks and 80 EMI stress cases passed.`

### Turn 11 — `execute_in_remote_docker` (`37991 ms` LLM)
- **Rationale**: Each coupon contributes ten trajectory samples but only one initial-rate observation. Compare fixed residual weights on real lab-held-out measurements, including upper-quantile errors and slow-time stability. Change the solver only if the comparison shows a meaningful gain without sacrificing either output; the audit leaves the verified artifact untouched.
- **Alternatives Weighed**: `Repeat the convolution and panel-invariance audits` (rejected: The current artifact already has a bound receipt for those checks. The unresolved question is whether the joint fit gives adequate weight to the independently measured initial rate.); `Add batch intercepts or unrelated basis terms` (rejected: Held-out residuals have small signed medians, and the exact constitutive basis already agrees with the independent convolution in both damage states.)

```bash
cat > /tmp/weight_rheology01.py <<'PY'
exec(open('/tmp/check_rheology01.py').read().split("print('active'")[0])
from pathlib import Path
source=Path(base+'alpha_m97_c03_batch_01_solver.py').read_text()
lab=d['train']['lab_id']
complete=~np.isnan(d['train']['coupon_features_raw']).any(1)
for rate_weight in (0.5,1.,2.):
    ns={}
    s=source.replace("    def model(p, jac=False):", f"    obs_weight = np.tile(np.r_[np.ones(len(args[-1])), {rate_weight!r}], len(target)//(len(args[-1])+1))\n    def model(p, jac=False):")
    s=s.replace("return np.einsum('ij,j->i', a, p[2:]) - target", "return obs_weight * (np.einsum('ij,j->i', a, p[2:]) - target)")
    s=s.replace('return np.column_stack((h, a))','return obs_weight[:, None] * np.column_stack((h, a))')
    exec(compile(s,'<weight-probe>','exec'),ns)
    folds=[]; spectra=[]
    for held in range(4):
        keep=lab!=held; test=~keep; loc=complete[test]
        tau,w=ns['_fit']((*[v[:n][keep] for v in args],dt,t),yy[keep].ravel(),'all')
        spectra.append(tau)
        e=np.abs(m._basis(np.log(tau),*[v[:n][test] for v in args],dt,t,'all')[0]@w-yy[test])
        folds.append([np.median(e[:,:-1]),np.median(e[:,-1]),np.median(e[loc,:-1]),np.median(e[loc,-1]),np.quantile(e[loc,:-1],.8),np.quantile(e[loc,-1],.8)])
    print('rate residual weight',rate_weight,'worst fold medians all and complete, then complete p80 strain/rate:',np.round(np.max(folds,axis=0),6),'fold slow taus:',np.round(np.asarray(spectra)[:,1],5))
PY
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/weight_rheology01.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`2335 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rate residual weight 0.5 worst fold medians all and complete, then complete p80 strain/rate: [0.021758 0.020022 0.017957 0.014213 0.039605 0.035379] fold slow taus: [4.00208 3.94878 3.98351 4.05743]
rate residual weight 1.0 worst fold medians all and complete, then complete p80 strain/rate: [0.02173  0.019743 0.017911 0.01408  0.0394   0.038709] fold slow taus: [3.99091 3.94191 3.98998 4.04822]
rate residual weight 2.0 worst fold medians all and complete, then complete p80 strain/rate: [0.021555 0.019457 0.01786  0.014232 0.039019 0.040122] fold slow taus: [3.98146 3.93073 4.00183 4.03754]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d8415e456b3b959e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_d8415e45 (directory).
```

### Turn 12 — `submit_completed_task` (`16351 ms` LLM)
- **Rationale**: Retain the independently verified artifact. Neither rate reweighting nor complete-coupon-only fitting produced a clear improvement across strain and initial-rate slices. The isolated schema check passed, and the independent convolution matched both predictions in both damage states.
- **Alternatives Weighed**: `Double the initial-rate residual weight` (rejected: It lowered the worst held-out lab rate median from 0.01974 to 0.01946 but raised the complete-coupon rate 80th percentile from 0.03871 to 0.04012.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_01_solver.py`. The public schema passed. Public test RMSE was 0.0371 for strain and 0.0286 for initial rate. All four sparse-calibration checks and 80 EMI stress cases passed. Missing coupon features remained the largest source of prediction error.`

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
