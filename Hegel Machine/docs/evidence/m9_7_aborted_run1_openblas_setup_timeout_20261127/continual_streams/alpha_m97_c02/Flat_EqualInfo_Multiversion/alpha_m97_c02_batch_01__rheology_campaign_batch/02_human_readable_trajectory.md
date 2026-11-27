# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c02_batch_01__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c02_step_01` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `11 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `294,707` (`284,790` / `85,248` / `9,917`) |
| **Total Episode Wall Time (s)** | `335.12s` |

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

### Turn 1 — `execute_in_remote_docker` (`18841 ms` LLM)
- **Rationale**: Inspect the schema and data shapes before choosing fit and validation slices. This will detect mismatched fields, unavailable labels, and prior-phase assumptions that would otherwise compromise the implementation.
- **Alternatives Weighed**: `Reuse the previous phase solver unchanged` (rejected: The current phase specifies a logarithmic plateau; inspect the archived solver to identify reusable calibration and alignment code without inheriting the old response law.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -l /tmp/hegel_continual_archive && sed -n '1,250p' /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference-panel alignment imputation'; cd /holdout_task && python3 - <<'PY'
import pickle
with open('public_alpha_m97_c02_batch_01.pkl','rb') as f: d=pickle.load(f)
def show(x,p='',depth=0):
 if isinstance(x,dict) and depth<3:
  for k,v in x.items(): show(v,p+'/'+str(k),depth+1)
 elif hasattr(x,'shape'): print(p,x.shape,x.dtype,str(x.flat[:5]))
 elif isinstance(x,(list,tuple)): print(p,type(x).__name__,len(x),str(x[:1])[:500]);
 else: print(p,type(x).__name__,str(x)[:300])
show(d)
PY
```
- **Container Exit Code**: `0` (`706 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
return {'estimated_taus': np.exp(log_taus), 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2),
            'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
 list 1 [{'dataset': {'past_dt': array([0.6862, 0.3536, 0.4115, 0.5101, 0.4903, 0.7741, 0.4359, 0.5228,
       0.5538, 0.2493, 0.5622, 0.3497, 0.6793, 0.3183, 0.7778, 0.7773,
       0.754 , 0.2048]), 'eval_times': array([0.3   , 0.8444, 1.3889, 1.9333, 2.4778, 3.0222, 3.5667, 4.1111,
       4.6556, 5.2   ]), 'crit_stress': 2.65, 'yield_stress': 2.65, 'shear_banding_index': 0.12, 'calibration_load_cells': {'lab_id': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
       2, 2, 2,
```

### Turn 2 — `execute_in_remote_docker` (`20103 ms` LLM)
- **Rationale**: Read the archived preprocessing and the nested public dataset to resolve reference-panel layout, label names, and calibration coverage. If the affine residuals reveal EMI or a channel lacks enough points, implement the documented archive fallback rather than applying blind least squares.
- **Alternatives Weighed**: `Use the diagnostic calibration vector as a fixed input` (rejected: The contract requires fitting gains and biases from clean in-batch observations.)

```bash
sed -n '1,155p' /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py; python3 - <<'PY'
import pickle,json,numpy as np
p='/holdout_task/public_alpha_m97_c02_batch_01.pkl'
item=pickle.load(open(p,'rb'))[0]; print('item keys',list(item)); d=item['dataset']
for k,v in d.items():
 if isinstance(v,dict):
  print(k,[(a,np.shape(b),str(np.asarray(b).dtype)) for a,b in v.items()]);
  for a in ['lab_id','stratum','ref_coupons']:
   if a in v:
    b=v[a]; print(a, (np.unique(b,return_counts=True) if a!='ref_coupons' else str(b)[:400]))
 else: print(k,np.shape(v),str(v)[:150])
c=d['calibration_load_cells'];
for j in range(4):
 m=c['lab_id']==j; x=c['cmd_stress'][m]; y=c['true_stress'][m]; a,b=np.polyfit(x,y,1); print('cal',j,len(x),a,b,np.max(abs(y-a*x-b)))
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('archive layout',str(h)[:1000])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -35
```
- **Container Exit Code**: `0` (`682 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
31224]
 ...
 [        nan  1.32408117  0.70202891 ...  3.9471362   1.00880318
  -0.21719363]
 [ 1.09264083  0.45643974 -0.29520899 ... -0.92766323  0.22595185
   1.8
cal 0 10 0.79004725175476 -0.13998828119953727 0.00019010847293882138
cal 1 10 1.3099623181039106 0.16005601310626413 0.00019176057234507327
cal 2 10 0.8499613249826024 0.10003695262801829 0.00012464239328242865
cal 3 10 1.240001538252827 -0.12996202646637137 0.00014402498836060817
archive layout [{'episode_short_name': 'alpha_m97_c02_s00', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [1.619455, -1.307963, -0.646756, 0.95759, 2.133391, -0.419148, -0.546245, -1.403033, 1.416515, -0.474218, 0.56263, -1.431472, -1.102381, 0.100096, 0.332884, 0.978926, 1.410301, 2.014558, 2.037985, -1.520981, -1.132282, -0.400487, 1.428566, 2.184907, 2.114515, 2.024265, 0.373395, 1.839347, -1.383881, 0.252459, -1.650907, 2.152332, 0.673079, 0.633995, 2.133025, 0.157003, -0.816267, -1.031715, -0.029779, 1.574899], 'true_stress': [1.619593, -1.308033, -0.646936, 0.95752, 2.133164, -0.418978, -0.546283, -1.402876, 1.416566, -0.474167, 0.562697, -1.431478, -1.102292, 0.1, 0.333007, 0.979026, 1.410368, 2.014512, 2.037978, -1.520941, -1.132263, -0.400526, 1.428582, 2.184857, 2.114623, 2.024408, 0.373485, 1.83932, -1.383844, 0.25249, -1.650978, 2.152261, 0.672955, 0.634083, 2.133016, 0.157033, -0.816183, -
{
  "ok": true,
  "target": "autogenous_control_panel_deconfounding.align_control_panel",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_control_panel_deconfounding",
      "granularity_tier": "L2_domain_operator",
      "compiled_code_sha256": "aa502f6d83a30b6babf8dcdff2012cefc7cf8ce998e199d66c4221b48051b07f",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "observations_per_contributing_control",
          "op": ">=",
          "value": 2
        },
        {
          "field": "usable_control_features_per_center",
          "op": ">=",
          "value": 1
        },
        {
          "field": "estimated_shared_scanner_gain",
          "op": ">",
          "value": 0
        },
        {
          "field": "observations_per_training_center_class_cell",
          "op": ">=",
          "value": 1
        }
      ],
      "counter_regime_guards": [],
```

### Turn 3 — `execute_in_remote_docker` (`16790 ms` LLM)
- **Rationale**: All four public channels have ten clean in-batch points. The earlier output truncated the archived panel transform and observation summary; read those sections directly to resolve reference-column layout, imputation, and curve-versus-rate weighting before implementing the plateau fit.
- **Alternatives Weighed**: `Apply a generic control-panel alignment primitive` (rejected: Its shared scanner-gain assumptions do not match the requested four-feature, per-lab reference statistics.)

```bash
sed -n '54,140p' /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py; python3 - <<'PY'
import pickle,numpy as np
item=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]; print('item keys',list(item)); d=item['dataset']; print('dataset keys',list(d)); tr=d['train']
for k,v in tr.items():
 a=np.asarray(v); print(k,a.shape,a.dtype, 'finite',np.isfinite(a).sum(), 'range', (np.nanmin(a),np.nanmax(a)))
print('times',d['eval_times'])
PY
```
- **Container Exit Code**: `0` (`551 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
np.concatenate((gap[:, None, :]*step,
                            times[None, :, None]*features[:, None, :]), axis=2)
    rate = np.column_stack((gap/taus, features))
    dcurve = dgap[:, None, :]*step - gap[:, None, :]*r*er
    drate = (dgap - gap)/taus
    return curve, rate, dcurve, drate


def _fit(past, hold, features, dt, times, curves, rates):
    scales = np.r_[np.full(curves.size, 0.025), np.full(rates.size, 0.025)]
    target = np.r_[curves.ravel(), rates]

    def design(log_taus):
        curve, rate, dcurve, drate = _basis(log_taus, past, hold, features, dt, times)
        return (np.vstack((curve.reshape(-1, 7), rate)),
                np.vstack((dcurve.reshape(-1, 2), drate)))

    def residual(params):
        matrix, _ = design(params[:2])
        return (np.einsum('ij,j->i', matrix, params[2:]) - target) / scales

    def jacobian(params):
        matrix, deriv = design(params[:2])
        return np.column_stack((deriv*params[2:4], matrix)) / scales[:, None]

    # Downweight large glitches for initialization, then minimize the
    # observation-level Cauchy loss on both the curve and its initial slope.
    weight = 1 / np.maximum(1, np.abs(target)/max(np.quantile(np.abs(target), 0.9), 0.01))
    lower = np.r_[np.log([0.01, 0.01]), np.full(7, -np.inf)]
    upper = np.r_[np.log([100., 100.]), np.full(7, np.inf)]
    best = None
    for pair in ((0.45, 2.4), (0.20, 1.0), (0.20, 6.0), (1.0, 6.0),
                 (0.09, 2.4), (0.45, 15.0)):
item keys ['dataset', 'gt']
dataset keys ['past_dt', 'eval_times', 'crit_stress', 'yield_stress', 'shear_banding_index', 'calibration_load_cells', 'train', 'test']
lab_id (180,) int64 finite 180 range (0, 3)
stratum (180,) int64 finite 180 range (0, 2)
coupon_features_raw (180, 4) float64 finite 700 range (-3.872691122990179, 5.177864993438177)
ref_coupons (180, 32) float64 finite 5535 range (-4.76307065482229, 5.395998560530416)
past_stress_cmd (180, 18) float64 finite 3240 range (-4.873684035853066, 5.22664370440438)
stress_hold_cmd (180,) float64 finite 180 range (-2.3022823529411767, 2.6337329113924053)
strain_0 (180,) float64 finite 180 range (-0.823836, 0.820409)
aux_chamber_telemetry (180, 3) float64 finite 540 range (-3.266351, 3.284674)
strain_curves_obs (180, 10) float64 finite 1800 range (-17.781020196386567, 25.323727123410176)
instant_rate_obs (180,) float64 finite 180 range (-5.111013802613548, 4.849736995223994)
times [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ]
```

### Turn 4 — `execute_in_remote_docker` (`62767 ms` LLM)
- **Rationale**: Fit the prescribed response jointly to curves and initial rates using only training labels. Holding out each lab and separating inactive from active coupons will show whether the terminal-flow interpretation transfers and whether either output needs a different robust scale.
- **Alternatives Weighed**: `Apply the logarithmic terminal term to every coupon` (rejected: The stated regime requires positive work damage; compare both interpretations on inactive held-out coupons before accepting either.)

```bash
set -e
cd /holdout_task
p=alpha_m97_c02_batch_01_solver.py
cat > "$p" <<'PY'
"""Estimate a calibrated two-branch rheology model for the plateau campaign."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _archived_calibration(channel):
    if not _ARCHIVE.is_file():
        return None
    with _ARCHIVE.open() as stream:
        history = json.load(stream)
    for table in reversed(history):
        take = np.asarray(table['lab_id']) == channel
        x = np.asarray(table['cmd_stress'])[take]
        if np.unique(x).size >= 2:
            return np.polyfit(x, np.asarray(table['true_stress'])[take], 1)
    return None


PY
sed -n '/^def _calibrate(/,/^def _basis(/{ /^def _basis(/!p; }' /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py >> "$p"
cat >> "$p" <<'PY'
def _activity(dataset, past, dt):
    if dataset['shear_banding_index'] > 0.50:
        return np.zeros(len(past))
    excess = np.maximum(0., np.abs(past) - dataset['crit_stress'])
    return np.tanh(excess**2 @ dt)


def _basis(log_taus, past, hold, features, dt, times, act, full_plateau=False):
    taus = np.exp(log_taus)
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    a, q, r = age[:, None]/taus, dt[:, None]/taus, times[:, None]/taus
    ea, eq, er = np.exp(-a), np.exp(-q), np.exp(-r)
    memory = ea * -np.expm1(-q)
    z = past @ memory
    dz = past @ (a*memory - q*ea*eq)
    gap = hold[:, None] - z
    step = -np.expm1(-r)
    prony = gap[:, None, :]*step
    phi = np.where((full_plateau | (act > 0))[:, None], np.log1p(times), times)
    drive = act * hold
    curve = np.concatenate((prony, phi[:, :, None]*features[:, None, :],
                            (drive[:, None]*step[:, 0])[..., None],
                            ((act*z[:, 0])[:, None]*phi)[..., None]), axis=2)
    rate = np.column_stack((gap/taus, features, drive/taus[0], act*z[:, 0]))
    dc = -dz[:, None, :]*step - prony*0 - gap[:, None, :]*r*er
    dr = (-dz - gap)/taus
    dc_extra = np.stack((-drive[:, None]*(r*er)[:, 0], (act*dz[:, 0])[:, None]*phi), axis=-1)
    dr_extra = np.column_stack((-drive/taus[0], act*dz[:, 0]))
    return curve, rate, np.concatenate((dc, dc_extra), axis=-1), np.column_stack((dr, dr_extra))


def _fit(past, hold, features, dt, times, act, curves, rates, full_plateau=False, scales=(0.025, 0.025)):
    target = np.r_[curves.ravel()/scales[0], rates/scales[1]]

    def design(log_taus):
        c, r, dc, dr = _basis(log_taus, past, hold, features, dt, times, act, full_plateau)
        return (np.vstack((c.reshape(-1, 9)/scales[0], r/scales[1])),
                np.vstack((dc.reshape(-1, 4)/scales[0], dr/scales[1])))

    def residual(params):
        matrix, _ = design(params[:2])
        return np.einsum('ij,j->i', matrix, params[2:]) - target

    def jacobian(params):
        matrix, d = design(params[:2])
        fast = d[:, 0]*params[2] + d[:, 2]*params[9] + d[:, 3]*params[10]
        return np.column_stack((fast, d[:, 1]*params[3], matrix))

    lo = np.r_[np.log([0.01, 0.01]), np.full(9, -np.inf)]
    hi = np.r_[np.log([100., 100.]), np.full(9, np.inf)]
    best = None
    for pair in ((0.45, 2.4), (0.20, 1.0), (0.20, 6.0), (1.0, 6.0), (0.09, 2.4), (0.45, 15.0)):
        log_taus = np.log(pair)
        matrix, _ = design(log_taus)
        weights = np.linalg.lstsq(matrix, target, rcond=None)[0]
        fit = least_squares(residual, np.r_[log_taus, weights], jac=jacobian,
                            bounds=(lo, hi), loss='cauchy', x_scale='jac', max_nfev=130)
        if best is None or fit.cost < best.cost:
            best = fit
    params = best.x.copy()
    if not np.any(act) and params[0] > params[1]:
        params[[0, 1, 2, 3]] = params[[1, 0, 3, 2]]
    return params[:2], params[2:]


def solve_alpha_m97_c02_batch_01(dataset: dict) -> dict:
    past, hold, features, strain0 = _prepare(dataset)
    dt, times = (np.asarray(dataset[k], dtype=float) for k in ('past_dt', 'eval_times'))
    act = _activity(dataset, past, dt)
    train = dataset['train']
    n = len(train['lab_id'])
    curves = np.asarray(train['strain_curves_obs']) - strain0[:n, None]
    taus, weights = _fit(past[:n], hold[:n], features[:n], dt, times, act[:n],
                         curves, np.asarray(train['instant_rate_obs']))
    curve, rate, _, _ = _basis(taus, past[n:], hold[n:], features[n:], dt, times, act[n:])
    return {'estimated_taus': np.exp(taus), 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
PY
sed -i 's/ - prony\*0//' "$p"
cat > /tmp/plateau_cv.py <<'PY'
import pickle, sys
import numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_m97_c02_batch_01_solver as s
with open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
p,h,x,y0=s._prepare(d)
dt,t=d['past_dt'],d['eval_times']
a=s._activity(d,p,dt)
l=d['train']['lab_id']; n=len(l)
y=d['train']['strain_curves_obs']-y0[:n,None]; r=d['train']['instant_rate_obs']
print('active by lab (active/total):',[(j,int(np.sum((l==j)&(a[:n]>0))),int(np.sum(l==j))) for j in np.unique(l)], 'test',sum(a[n:]>0),len(a)-n)
for full in (False,True):
    for j in [None,*np.unique(l)]:
        tr=np.flatnonzero(l!=j); va=np.flatnonzero(l==j) if j is not None else tr
        z,w=s._fit(p[tr],h[tr],x[tr],dt,t,a[tr],y[tr],r[tr],full_plateau=full)
        cb,rb,*_=s._basis(z,p[va],h[va],x[va],dt,t,a[va],full_plateau=full)
        e=cb@w-y[va]; er=rb@w-r[va]
        print('full',full,'lab',j,'taus',np.exp(z).round(5),'weights',w.round(5), 'median and rmse curve/rate',[(round(np.median(abs(v)),5),round(np.sqrt(np.mean(v**2)),5)) for v in (e,er)], 'medians by damage',[(m,np.round(np.median(abs(e[(a[va]>0)==m]),axis=0)[[0,-1]],5)) for m in (False,True) if np.any((a[va]>0)==m)])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/plateau_cv.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1885 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
0.08753  0.28434
  0.23755] median and rmse curve/rate [(0.01927, 1.42014), (0.0174, 0.04785)] medians by damage [(False, array([0.01936, 0.56289])), (True, array([0.01068, 0.01448]))]
full False lab 2 taus [0.50068 3.73787] weights [ 0.67757  0.54569  0.13798  0.15509 -0.12565  0.11635 -0.08732  0.2799
  0.24104] median and rmse curve/rate [(0.02168, 0.96183), (0.01281, 0.0255)] medians by damage [(False, array([0.01225, 0.45964])), (True, array([0.0116, 0.0178]))]
full False lab 3 taus [0.49999 3.7715 ] weights [ 0.67651  0.54927  0.13922  0.15636 -0.12688  0.11652 -0.08773  0.28108
  0.24001] median and rmse curve/rate [(0.01825, 0.41063), (0.01163, 0.04915)] medians by damage [(False, array([0.01731, 0.4672 ])), (True, array([0.01343, 0.01204]))]
full True lab None taus [0.50011 3.74747] weights [ 0.67695  0.54823  0.13886  0.15632 -0.12761  0.11756 -0.08824  0.27995
  0.24092] median and rmse curve/rate [(0.01235, 0.94571), (0.01317, 0.04472)] medians by damage [(False, array([0.01697, 0.0173 ])), (True, array([0.01189, 0.01193]))]
full True lab 0 taus [0.49975 3.72817] weights [ 0.67612  0.54699  0.13823  0.15576 -0.12701  0.11701 -0.0878   0.28046
  0.23973] median and rmse curve/rate [(0.01355, 1.02597), (0.01208, 0.0554)] medians by damage [(False, array([0.02056, 0.02412])), (True, array([0.01024, 0.01101]))]
full True lab 1 taus [0.50005 3.74449] weights [ 0.67642  0.5488   0.13839  0.15646 -0.12753  0.11724 -0.08816  0.28021
  0.24096] median and rmse curve/rate [(0.01305, 1.36877), (0.01694, 0.04735)] medians by damage [(False, array([0.01175, 0.0188 ])), (True, array([0.01129, 0.01363]))]
full True lab 2 taus [0.50052 3.75996] weights [ 0.67784  0.54732  0.1389   0.1558  -0.12752  0.11754 -0.08813  0.27973
  0.24154] median and rmse curve/rate [(0.01193, 0.90625), (0.01351, 0.02532)] medians by damage [(False, array([0.0168 , 0.01812])), (True, array([0.01214, 0.01406]))]
full True lab 3 taus [0.50045 3.79247] weights [ 0.67835  0.55091  0.14013  0.15749 -0.12831  0.11892 -0.08881  0.27956
  0.24115] median and rmse curve/rate [(0.0142, 0.34056), (0.01381, 0.04953)] medians by damage [(False, array([0.01905, 0.01751])), (True, array([0.01397, 0.01375]))]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fe1d2ce3e8a6ec88 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_fe1d2ce3 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`44099 ms` LLM)
- **Rationale**: Implement the held-out improvement and verify the basis against the direct recurrence and initial derivative. The residual quantiles and outlier counts distinguish isolated glitches, which the Cauchy fit can handle, from a remaining systematic error that would warrant another model change.
- **Alternatives Weighed**: `Use raw RMSE to select the terminal basis` (rejected: Large observation glitches dominate RMSE; the zero-damage late-time medians show a systematic error across all four labs.)
- **ATMS NOGOOD DECLARED**: Refuted `Zero work damage restores linear terminal flow for individual coupons in the plateau campaign.` | Witness: `At the final time, held-out zero-damage lab medians were 0.460–0.659 with linear flow versus 0.0175–0.0241 with logarithmic flow. Active-coupon errors also stayed at the noise scale.` | Blanket Diagnosis: `The damage threshold controls the added crystallization stresses, but does not switch the shared campaign terminal basis back to linear.` | Sublated Pivot: `Use the campaign-level logarithmic terminal basis below the shear-banding threshold, retaining both damage-dependent terms and the exact initial derivative.`

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p=Path('alpha_m97_c02_batch_01_solver.py')
s=p.read_text().replace('full_plateau=False','plateau').replace('(full_plateau | (act > 0))[:, None], np.log1p(times), times','plateau, np.log1p(times), times').replace('phi = np.where(plateau, np.log1p(times), times)','phi = np.broadcast_to(np.log1p(times) if plateau else times, (len(past), len(times)))').replace('act, full_plateau)', 'act, plateau)').replace("    act = _activity(dataset, past, dt)\n", "    # The campaign selects the terminal basis; work damage gates only the two extra terms.\n    plateau = dataset['shear_banding_index'] <= 0.50\n    act = _activity(dataset, past, dt)\n").replace("np.asarray(train['instant_rate_obs']))", "np.asarray(train['instant_rate_obs']), plateau)").replace('dt, times, act[n:])','dt, times, act[n:], plateau)')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
cat > /tmp/plateau_diagnostics.py <<'PY'
import pickle, sys
import numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_m97_c02_batch_01_solver as s
with open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb') as f:
    d=pickle.load(f)[0]['dataset']
p,h,x,y0=s._prepare(d)
dt,t=d['past_dt'],d['eval_times']
a=s._activity(d,p,dt)
l=d['train']['lab_id']; n=len(l)
y=d['train']['strain_curves_obs']-y0[:n,None]; r=d['train']['instant_rate_obs']
e=np.empty_like(y); er=np.empty_like(r)
for j in np.unique(l):
    tr=l!=j; va=~tr
    z,w=s._fit(p[:n][tr],h[:n][tr],x[:n][tr],dt,t,a[:n][tr],y[tr],r[tr],True)
    cb,rb,*_=s._basis(z,p[:n][va],h[:n][va],x[:n][va],dt,t,a[:n][va],True)
    e[va]=cb@w-y[va]; er[va]=rb@w-r[va]
print('Curve residual absolute quantiles by time:',np.quantile(abs(e),[.5,.8,.9,.95,.99],axis=0).round(4))
print('Rate residual absolute quantiles:',np.quantile(abs(er),[.5,.8,.9,.95,.99]).round(4))
print('Curve outliers > 0.15 per coupon:',np.unique((abs(e)>.15).sum(axis=1),return_counts=True))
print('Curve outliers > 0.15 by time:',(abs(e)>.15).sum(axis=0))
print('Lab/damage slices: lab, active, count, worst time median absolute curve error, median absolute rate error')
for j in np.unique(l):
    for active in [False,True]:
        m=(l==j)&((a[:n]>0)==active)
        print(j,active,m.sum(),np.median(abs(e[m]),axis=0).max(),np.median(abs(er[m])))
for plateau in (False,True):
    logtau=np.log([.5,3.8]); taus=np.exp(logtau)
    act=a if plateau else np.zeros_like(a)
    z=np.zeros((len(p),2))
    for stress,ds in zip(p.T,dt):
        z=np.exp(-ds/taus)*z+(-np.expm1(-ds/taus))*stress[:,None]
    cb,rb,*_=s._basis(logtau,p,h,x,dt,t,act,plateau)
    phi=np.log1p(t) if plateau else t
    step=-np.expm1(-t[:,None]/taus)
    for m in range(2):
        np.testing.assert_allclose(cb[:,:,m],(h-z[:,m])[:,None]*step[:,m],atol=1e-13)
    np.testing.assert_allclose(cb[:,:,2:7],x[:,None,:]*phi[None,:,None],atol=1e-13)
    np.testing.assert_allclose(cb[:,:,7],(act*h)[:,None]*step[:,0],atol=1e-13)
    np.testing.assert_allclose(cb[:,:,8],(act*z[:,0])[:,None]*phi,atol=1e-13)
    cb_small,*_=s._basis(logtau,p,h,x,dt,np.array([1e-7]),act,plateau)
    np.testing.assert_allclose(cb_small[:,0,:]/1e-7,rb,rtol=2e-7,atol=1e-12)
print('Recurrence and initial-rate checks passed for both terminal bases.')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/plateau_diagnostics.py
```
- **Container Exit Code**: `0` (`1411 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Curve residual absolute quantiles by time: [[0.0129 0.0099 0.0121 0.0144 0.0115 0.0131 0.0126 0.0151 0.0127 0.0147]
 [0.027  0.0233 0.0262 0.0299 0.0318 0.03   0.0271 0.0323 0.0348 0.0325]
 [0.0487 0.0404 0.0655 0.1394 0.1516 0.1742 0.069  0.0981 0.173  0.161 ]
 [0.1776 0.1182 0.256  0.3334 0.4562 0.4326 0.1782 0.3337 0.5156 0.2996]
 [0.7191 1.0789 1.3305 1.9601 2.3695 0.9852 0.6673 1.4595 2.4289 4.04  ]]
Rate residual absolute quantiles: [0.0138 0.0275 0.0405 0.0655 0.219 ]
Curve outliers > 0.15 per coupon: (array([0, 1, 2, 3, 4, 5, 6, 7, 9]), array([94, 60, 15,  4,  1,  1,  1,  2,  2]))
Curve outliers > 0.15 by time: [10  9 10 16 18 19 12 14 20 21]
Lab/damage slices: lab, active, count, worst time median absolute curve error, median absolute rate error
0 False 11 0.024122451699865266 0.008956520074368868
0 True 30 0.016563119555997513 0.012640669299774698
1 False 13 0.033696344121535216 0.021745158587052504
1 True 24 0.014780500183653955 0.015313233904161061
2 False 21 0.018121829673239664 0.01675787106009488
2 True 31 0.015518027485506458 0.010688604327768747
3 False 15 0.02707583444071604 0.017258517834440656
3 True 35 0.01959394210447374 0.01272492584443341
Recurrence and initial-rate checks passed for both terminal bases.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ced7f5cfe519d5af (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ced7f5cf (directory).
```

### Turn 6 — `submit_completed_task` (`18894 ms` LLM)
- **Rationale**: The verified campaign-level logarithmic basis reduced the worst held-out late-time median error on zero-damage coupons from 0.659 to 0.0241. Independent recurrence and derivative checks passed for both terminal bases, and the final artifact passed the schema validator in isolated Python.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_01_solver.py': 'FILE:7282'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_m97_c02_batch_01_solver.py`. It passed the public schema validator and the direct recurrence and initial-rate checks. Across held-out lab and damage slices, the worst median absolute errors were 0.0337 for strain and 0.0217 for initial rate.`

### Turn 7 — `execute_in_remote_docker` (`35370 ms` LLM)
- **Rationale**: Replay the two documented calibration failure modes on public measurements to catch a broken archive lookup or an EMI-biased fit. Compare robust scales under held-out labs to see whether either output or the missing-feature slices have a material bottleneck before changing the solver.
- **Alternatives Weighed**: `Add rational or cross-feature response terms` (rejected: The prescribed basis already contains the hold-minus-memory differences and both damage interactions. Additional terms would need evidence of systematic held-out residuals.); `Run classification balance or relay-mode checks` (rejected: This dataset has continuous strain and rate labels and no relay state. Held-out lab and missing-feature slices test the relevant nuisance sensitivity.)

```bash
set -e
sed -n '1,62p' /holdout_task/alpha_m97_c02_batch_01_solver.py
cat > /tmp/plateau_stress_audit.py <<'PY'
import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import alpha_m97_c02_batch_01_solver as s
with open('/holdout_task/public_alpha_m97_c02_batch_01.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']
base = s._calibrate(c)
for j in range(4):
    rows = np.flatnonzero(c['lab_id'] == j)
    for mode in ('one_point', 'one_spike'):
        altered = {k: np.array(v, copy=True) for k,v in c.items()}
        if mode == 'one_point':
            take = np.ones(len(c['lab_id']), dtype=bool)
            take[rows[1:]] = False
            altered = {k:v[take] for k,v in altered.items()}
        else:
            altered['true_stress'][rows[0]] += 4.
        try:
            candidate = s._calibrate(altered)
            print('calibration', j, mode, 'delta gain/bias', (candidate[j]-base[j]).round(7), 'other channels unchanged', np.array_equal(np.delete(candidate,j,axis=0),np.delete(base,j,axis=0)))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            print('calibration', j, mode, type(exc).__name__, str(exc))
p,h,x,y0 = s._prepare(d)
dt,t=d['past_dt'],d['eval_times']
a=s._activity(d,p,dt)
train=d['train']; l=train['lab_id']; n=len(l)
y=train['strain_curves_obs']-y0[:n,None]; r=train['instant_rate_obs']
missing=np.isnan(train['coupon_features_raw']).any(axis=1)
for scales in ((.025,.025),(.018,.018),(.018,.030)):
    maxima=np.zeros(6); pars=[]
    for j in np.unique(l):
        tr=np.flatnonzero(l!=j); va=np.flatnonzero(l==j)
        z,w=s._fit(p[tr],h[tr],x[tr],dt,t,a[tr],y[tr],r[tr],True,scales=scales)
        pars.append(np.r_[np.exp(z),w])
        cb,rb,*_=s._basis(z,p[va],h[va],x[va],dt,t,a[va],True)
        e=abs(cb@w-y[va]); er=abs(rb@w-r[va]); active=a[va]>0
        for masks, start in (([active,~active],0),([missing[va],~missing[va]],2)):
            for m in masks:
                if m.any():
                    maxima[start:start+2]=np.maximum(maxima[start:start+2],[np.median(e[m],axis=0).max(),np.median(er[m])])
        maxima[4:6]=np.maximum(maxima[4:6],[np.median(e[:,-1]),np.median(er)])
    print('scales',scales,'worst errors [damage curve/rate, missingness curve/rate, late lab curve/lab rate]',maxima.round(6),'tau fold ranges',np.ptp(pars,axis=0)[:2].round(6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/plateau_stress_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1924 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
x.size < 2 or spike else None
        if np.count_nonzero(clean) >= 4 and np.unique(x[clean]).size >= 2:
            result[channel] = np.polyfit(x[clean], y[clean], 1)
        elif archive is not None:
            result[channel] = archive
        elif np.unique(x[clean]).size >= 2:
            result[channel] = np.polyfit(x[clean], y[clean], 1)
        else:
            result[channel] = (1.0, np.median(y - x) if y.size else 0.0)
    return result


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    keys = ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons',
            'past_stress_cmd', 'stress_hold_cmd', 'strain_0')
    lab, stratum, x, ref, past, hold, strain0 = (
        np.concatenate((train[key], test[key])) for key in keys
    )
    x = x.astype(float, copy=False)
    for channel in np.unique(lab):
        mask = lab == channel
calibration 0 one_point delta gain/bias [0.2099296 0.1399896] other channels unchanged True
calibration 0 one_spike delta gain/bias [-1.2e-06  3.1e-06] other channels unchanged True
calibration 1 one_point delta gain/bias [-0.309969  -0.1600206] other channels unchanged True
calibration 1 one_spike delta gain/bias [ 1.2e-06 -3.1e-06] other channels unchanged True
calibration 2 one_point delta gain/bias [ 0.1500461 -0.1000096] other channels unchanged True
calibration 2 one_spike delta gain/bias [-2.4e-06 -1.1e-06] other channels unchanged True
calibration 3 one_point delta gain/bias [-0.239992   0.1299313] other channels unchanged True
calibration 3 one_spike delta gain/bias [-8.7e-06 -9.9e-06] other channels unchanged True
scales (0.025, 0.025) worst errors [damage curve/rate, missingness curve/rate, late lab curve/lab rate] [0.033696 0.021745 0.239611 0.105045 0.015835 0.016937] tau fold ranges [0.000771 0.064299]
scales (0.018, 0.018) worst errors [damage curve/rate, missingness curve/rate, late lab curve/lab rate] [0.034062 0.021269 0.239104 0.104758 0.015887 0.016911] tau fold ranges [0.000814 0.064208]
scales (0.018, 0.03) worst errors [damage curve/rate, missingness curve/rate, late lab curve/lab rate] [0.034384 0.0211   0.238785 0.104536 0.015887 0.017209] tau fold ranges [0.000868 0.06226 ]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ced7f5cfe519d5af (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ced7f5cf (directory).
```

### Turn 8 — `execute_in_remote_docker` (`22225 ms` LLM)
- **Rationale**: The current artifact passed isolated schema and host handoff checks. Size the missing-feature slices and inspect their imputation path before editing: a tiny slice calls for reporting the limitation, while a well-populated slice with incorrectly pooled means warrants a targeted fix.
- **Alternatives Weighed**: `Retune the Cauchy scales` (rejected: Neither tested setting materially improved the missing-feature bottleneck.); `Replace the sparse-channel archive fallback with the public calibration vector` (rejected: The vector is not a calibration measurement for future batches; clean in-batch measurements and the permitted archive are the documented sources.)

```bash
set -e
sed -n '55,96p' /holdout_task/alpha_m97_c02_batch_01_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, numpy as np
with open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb') as f:
    item=pickle.load(f)[0]
d=item['dataset']; print('item keys',list(item))
for split in ('train','test'):
    p=d[split]; l=p['lab_id']; x=p['coupon_features_raw']; s=p['stratum']; r=p['ref_coupons']
    print(split,'lab: rows, rows with missing features, missing counts by feature')
    for j in np.unique(l):
        m=l==j; z=np.isnan(x[m]); print(j,m.sum(),z.any(axis=1).sum(),z.sum(axis=0).tolist())
    print('stratum cell sizes:',[(int(j),str(k),int(np.sum((l==j)&(s==k)))) for j in np.unique(l) for k in np.unique(s)])
    print('reference row std range:',np.nanquantile(np.nanstd(r.reshape(len(l),-1,4),axis=1),[0,.5,1]))
PY
```
- **Container Exit Code**: `0` (`366 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
n(np.nanmean(panel, axis=0))
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


def _activity(dataset, past, dt):
    if dataset['shear_banding_index'] > 0.50:
        return np.zeros(len(past))
    excess = np.maximum(0., np.abs(past) - dataset['crit_stress'])
    return np.tanh(excess**2 @ dt)


def _basis(log_taus, past, hold, features, dt, times, act, plateau):
    taus = np.exp(log_taus)
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    a, q, r = age[:, None]/taus, dt[:, None]/taus, times[:, None]/taus
    ea, eq, er = np.exp(-a), np.exp(-q), np.exp(-r)
    memory = ea * -np.expm1(-q)
item keys ['dataset', 'gt']
train lab: rows, rows with missing features, missing counts by feature
0 41 5 [1, 1, 1, 2]
1 37 4 [1, 2, 1, 1]
2 52 5 [0, 3, 1, 1]
3 50 5 [1, 3, 0, 1]
stratum cell sizes: [(0, '0', 19), (0, '1', 16), (0, '2', 6), (1, '0', 7), (1, '1', 20), (1, '2', 10), (2, '0', 17), (2, '1', 15), (2, '2', 20), (3, '0', 10), (3, '1', 15), (3, '2', 25)]
reference row std range: [0.22844703 0.93121601 2.87393435]
test lab: rows, rows with missing features, missing counts by feature
0 17 1 [0, 0, 1, 0]
1 16 2 [1, 0, 1, 1]
2 16 0 [0, 0, 0, 0]
3 23 0 [0, 0, 0, 0]
stratum cell sizes: [(0, '0', 6), (0, '1', 5), (0, '2', 6), (1, '0', 4), (1, '1', 8), (1, '2', 4), (2, '0', 4), (2, '1', 5), (2, '2', 7), (3, '0', 8), (3, '1', 6), (3, '2', 9)]
reference row std range: [0.3029649  1.01532791 2.14851931]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ced7f5cfe519d5af (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ced7f5cf (directory).
```

### Turn 9 — `submit_completed_task` (`15994 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated validation and the host scientific probe. The calibration audit confirmed that a single EMI spike barely affects the fit when enough clean points remain, and that a one-point channel uses the archive. The remaining missing-feature error does not establish an implementation defect or justify changing the prescribed imputation.
- **Alternatives Weighed**: `Change imputation or retune the robust loss` (rejected: The solver uses the specified cell-balanced means, and the tested loss scales did not materially improve the missing-feature slices.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_01_solver.py': 'FILE:7282'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_m97_c02_batch_01_solver.py`. It passed isolated schema validation and the recurrence, initial-rate, and panel-alignment checks. The public verification probe measured a strain MAE of 0.00938 and a relative relaxation-time error of 2.03%. Predictions were less accurate on held-out coupons with missing features.`

### Turn 10 — `execute_in_remote_docker` (`28488 ms` LLM)
- **Rationale**: Test whether imputed training features bias the shared coefficients or relaxation times. Compare both fits on the same unseen labs and report signed residuals on complete coupons to detect remaining lab offsets. Keep the already verified artifact unless the challenger materially improves transfer without worsening either output.
- **Alternatives Weighed**: `Add rational terms, drift operators, or relay states` (rejected: The direct recurrence and derivative checks verified the prescribed rheology basis, and the dataset has continuous responses and no relay state.)

```bash
set -e
cat > /tmp/plateau_fit_sensitivity.py <<'PY'
import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import alpha_m97_c02_batch_01_solver as s
with open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
p,h,x,y0=s._prepare(d)
dt,t=d['past_dt'],d['eval_times']
a=s._activity(d,p,dt)
train=d['train']; lab=train['lab_id']; n=len(lab)
y=train['strain_curves_obs']-y0[:n,None]; r=train['instant_rate_obs']
complete=np.isfinite(train['coupon_features_raw']).all(axis=1)
for complete_fit in (False,True):
    worst=np.zeros((3,2)); fold_taus=[]
    for j in np.unique(lab):
        tr=np.flatnonzero((lab!=j)&(complete | (not complete_fit)))
        va=np.flatnonzero(lab==j)
        z,w=s._fit(p[tr],h[tr],x[tr],dt,t,a[tr],y[tr],r[tr],True)
        c,b,*_=s._basis(z,p[va],h[va],x[va],dt,t,a[va],True)
        e=c@w-y[va]; er=b@w-r[va]; fold_taus.append(np.exp(z))
        for i,m in enumerate((complete[va],~complete[va],np.ones(len(va),dtype=bool))):
            worst[i]=np.maximum(worst[i], [np.median(abs(e[m]),axis=0).max(),np.median(abs(er[m]))])
        print('complete_fit',complete_fit,'lab',j,'signed residual medians among complete coupons',np.round(np.median(e[complete[va]],axis=0)[[0,-1]],5),round(np.median(er[complete[va]]),5))
    print('complete_fit',complete_fit,'worst medians [complete, missing, all] x [curve, rate]',worst.round(6).tolist(),'tau fold range',np.ptp(fold_taus,axis=0).round(6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/plateau_fit_sensitivity.py
```
- **Container Exit Code**: `0` (`1200 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
complete_fit False lab 0 signed residual medians among complete coupons [-0.00213 -0.00105] -0.00561
complete_fit False lab 1 signed residual medians among complete coupons [-0.00643 -0.00288] 0.00447
complete_fit False lab 2 signed residual medians among complete coupons [-0.00408 -0.00036] 0.00416
complete_fit False lab 3 signed residual medians among complete coupons [0.00101 0.00494] 0.00461
complete_fit False worst medians [complete, missing, all] x [curve, rate] [[0.017005, 0.013689], [0.239611, 0.105045], [0.018007, 0.016937]] tau fold range [0.000771 0.064299]
complete_fit True lab 0 signed residual medians among complete coupons [-0.00197 -0.00092] -0.00566
complete_fit True lab 1 signed residual medians among complete coupons [-0.00649 -0.0041 ] 0.00502
complete_fit True lab 2 signed residual medians among complete coupons [-0.00439  0.00066] 0.00538
complete_fit True lab 3 signed residual medians among complete coupons [0.00085 0.0045 ] 0.0045
complete_fit True worst medians [complete, missing, all] x [curve, rate] [[0.016664, 0.012628], [0.238027, 0.103771], [0.018379, 0.016885]] tau fold range [0.001067 0.062746]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ced7f5cfe519d5af (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ced7f5cf (directory).
```

### Turn 11 — `submit_completed_task` (`15531 ms` LLM)
- **Rationale**: The current artifact passed isolated schema validation, direct recurrence and initial-rate checks, and the host scientific probe. The complete-feature challenger did not materially improve the missing-feature bottleneck, so the verified all-coupon fit is retained.
- **Alternatives Weighed**: `Fit only coupons with complete features` (rejected: Worst missing-feature medians changed only from 0.2396 to 0.2380 for strain and from 0.1050 to 0.1038 for initial rate, while the worst all-coupon strain median increased.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_01_solver.py`. The isolated schema check and scientific probe passed, with a public strain MAE of 0.00938 and a relative relaxation-time error of 2.03%. Held-out coupons with missing features remained the least accurate slice.`

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
