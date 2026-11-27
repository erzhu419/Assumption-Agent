# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c03_batch_07__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c03_step_07` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `338,123` (`325,472` / `101,760` / `12,651`) |
| **Total Episode Wall Time (s)** | `387.49s` |

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

### Turn 1 — `execute_in_remote_docker` (`17211 ms` LLM)
- **Rationale**: Inspect the public schema and sampled values first to identify which fields determine calibration fallback, panel imputation, and regime gating before fitting the model.
- **Alternatives Weighed**: `Reuse the previous solver unchanged` (rejected: The current contract specifies a logarithmic plateau; the prior solver's regime law and schema must be checked before reuse.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -l /tmp/hegel_continual_archive && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation variable projection' ; python3 - <<'PY'
import pickle,numpy as np
p='/holdout_task/public_alpha_m97_c03_batch_07.pkl'
d=pickle.load(open(p,'rb'))
def show(x,depth=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print(' '*depth,k,end=': ');show(v,depth+2)
 elif isinstance(x,np.ndarray):
  print(x.shape,x.dtype,repr(x.reshape(-1)[:12]));
 elif isinstance(x,(list,tuple)):
  print(type(x).__name__,len(x));
  if len(x) and depth<6: show(x[0],depth+2)
 else: print(repr(x)[:300])
show(d)
PY
```
- **Container Exit Code**: `0` (`383 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
,) int64 array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
       stratum: (72,) int64 array([2, 0, 2, 1, 0, 0, 2, 2, 2, 1, 1, 0])
       coupon_features_raw: (72, 4) float64 array([-1.67830773, -1.53220366, -0.36993473,  0.72296123, -0.47746914,
        1.39802973,  1.91569094, -1.38415462, -0.15090601, -1.06020224,
       -0.96094493,  0.3837808 ])
       ref_coupons: (72, 32) float64 array([ 0.11459974, -0.67503284,  0.91850224, -0.80665586, -0.56053786,
       -1.25627655, -0.19606127, -1.40008823,  0.46219092, -1.16578214,
        1.11622003,  0.32808578])
       past_stress_cmd: (72, 18) float64 array([ 2.13138308, -0.62916602,  2.8387094 ,  4.76069416,  5.48052363,
        4.95638195,  4.52646993,  5.35934466,  4.73096143,  5.30207539,
        5.51847544, -1.71118359])
       stress_hold_cmd: (72,) float64 array([ 2.31129178, -1.23952683, -1.60588718, -1.36121912,  1.04286986,
       -0.24317073,  0.57345983,  0.05450956, -1.4176863 , -0.11988415,
        0.2928265 ,  1.16068529])
       strain_0: (72,) float64 array([-0.574576,  0.222915, -0.262396, -0.176916, -0.181844, -0.0388  ,
        0.712147, -0.102512,  0.038206, -0.310468, -0.107969,  0.316038])
       aux_chamber_telemetry: (72, 3) float64 array([ 0.460869, -0.680016,  4.62959 ,  4.403524,  4.47216 ,  2.855778,
        2.13737 , -1.15078 , 10.224212, -1.894437, -0.854841,  3.112868])
   gt:      taus: (2,) float64 array([0.535, 3.734])
     theta: (7,) float64 array([ 0.68,  0.55,  0.14,  0.16, -0.13,  0.12, -0.09])
     vp_theta: (2,) float64 array([0.27, 0.25])
     stress_gains: (4,) float64 array([0.73, 0.82, 1.17, 1.36])
     stress_biases: (4,) float64 array([-0.18, -0.15,  0.14,  0.17])
     te_clean: (72, 10) float64 array([-0.0465801 ,  0.42555194,  0.62121583,  0.71332682,  0.76508353,
        0.79980472,  0.82634459,  0.84820934,  0.86693755,  0.88328792,
       -0.22151046, -0.64739899])
     te_true_rate0: (72,) float64 array([ 2.27226608, -1.87938039, -3.22121916, -3.36292126,  1.15452723,
       -0.70393062,  2.20173556,  0.82024007, -3.45982718, -0.74858562,
        1.20447767,  3.7259431 ])
     te_hist_true: (72, 18) float64 array([ 1.37590965, -0.6392912 ,  1.89225786,  3.29530674,  3.82078225,
        3.43815882,  3.12432305,  3.7323216 ,  3.27360185,  3.69051503,
        3.84848707, -1.42916402])
     te_hold_true: (72,) float64 array([ 1.507243, -1.166412, -1.738888, -1.681258,  0.581295, -0.3494  ,
        0.810948,  0.244133, -1.214911, -0.248305,  0.482607,  1.748532])
```

### Turn 2 — `execute_in_remote_docker` (`19173 ms` LLM)
- **Rationale**: Check row counts and prior alignment code to determine where history is required and how the 32 reference values map to four features. Neither public truth nor unrelated telemetry is needed for fitting.
- **Alternatives Weighed**: `Fit all load-cell maps from current rows alone` (rejected: The shown batch supplies only two calibration rows for four channels, so at least some maps require the authorized history.)

```bash
ls -l /tmp/hegel_continual_archive; for f in /tmp/hegel_continual_archive/*batch_0{5,6}_solver.py; do echo "--- $f"; grep -nE '^def |ref_coupons|nanmean|nanmedian|calibration_history|EMI' "$f"; done; python3 - <<'PY'
import json, pickle, numpy as np
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
h=json.load(open(p)); print('history layout',type(h),list(h) if isinstance(h,dict) else len(h)); print('history tail',str(h)[-3300:]);
a=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_07.pkl','rb')); print('public items',len(a));
for i,item in enumerate(a):
 d=item['dataset']; c=d['calibration_load_cells']; print(i,'calibration rows',list(zip(c['lab_id'],c['cmd_stress'],c['true_stress'])),'times',d['eval_times'],'band',d['shear_banding_index']);
 for k in ('train','test'):
  s=d[k]; print(k,'keys',list(s),'labs',np.unique(s['lab_id'],return_counts=True),'panel shape',s['ref_coupons'].shape,'missing features',np.isnan(s['coupon_features_raw']).sum(axis=0),'missing references',np.isnan(s['ref_coupons']).sum());
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference-panel alignment imputation' 2>&1 | head -35
```
- **Container Exit Code**: `0` (`379 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
s 1
0 calibration rows [(0, -1.3, -1.1290274421217048), (0, 1.4, 0.841941898503713)] times [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ] band 0.12
train keys ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0', 'aux_chamber_telemetry', 'strain_curves_obs', 'instant_rate_obs'] labs (array([0, 1, 2, 3]), array([39, 59, 36, 46])) panel shape (180, 32) missing features [5 3 7 3] missing references 239
test keys ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0', 'aux_chamber_telemetry'] labs (array([0, 1, 2, 3]), array([11, 25, 16, 20])) panel shape (72, 32) missing features [1 1 1 3] missing references 86
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 3 — `execute_in_remote_docker` (`16847 ms` LLM)
- **Rationale**: Read the prior panel and imputation conventions before implementation. Quantify historical residuals by channel and campaign to decide whether pooling clean rows is justified or a campaign-specific fallback is needed.
- **Alternatives Weighed**: `Pool historical calibration rows by ordinary least squares` (rejected: The archive includes known impulse-contaminated sweeps; a per-sweep residual summary will show which rows need robust rejection.)

```bash
sed -n '1,125p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_05_solver.py; sed -n '83,155p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_06_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel 2>&1 | head -40; python3 - <<'PY'
import json,numpy as np
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
for e in h:
 print(e['episode_short_name'],end=' ')
 for k in range(4):
  m=np.asarray(e['lab_id'])==k; x=np.asarray(e['cmd_stress'])[m]; y=np.asarray(e['true_stress'])[m]
  r=y-(np.array([.73,.82,1.17,1.36])[k]*x+np.array([-.18,-.15,.14,.17])[k]); print((len(x),round(float(np.max(np.abs(r),initial=0)),4)),end=' ')
 print()
PY
```
- **Container Exit Code**: `0` (`394 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
finite(train['coupon_features_raw']).all(axis=1)
    fit_rows = {k: v[complete] for k, v in train.items()}
    taus, weights = _fit(_design(fit_rows, a_train[complete], calibration, dt, times),
                         fit_rows['strain_curves_obs'] - fit_rows['strain_0'][:, None],
                         fit_rows['instant_rate_obs'], dt, times)
    curves, rates = _design(test, a_test, calibration, dt, times)(taus)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
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
      "function_level_applicability": {},
      "functions": [
        {
          "function": "align_control_panel",
          "signature": "align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray",
alpha_m97_c03_s00 (10, 0.6152) (10, 0.543) (10, 0.5027) (10, 0.8854) 
alpha_m97_c03_batch_01 (10, 1.5047) (10, 0.0002) (10, 0.0002) (10, 1.5989) 
alpha_m97_c03_batch_02 (10, 1.2236) (10, 0.0002) (10, 0.0002) (10, 1.5409) 
alpha_m97_c03_batch_03 (10, 1.4598) (10, 0.0003) (10, 0.0002) (10, 1.5753) 
alpha_m97_c03_batch_04 (2, 1.084) (0, 0.0) (0, 0.0) (0, 0.0) 
alpha_m97_c03_batch_05 (10, 1.2236) (10, 0.0002) (10, 0.0002) (10, 1.5409) 
alpha_m97_c03_batch_06 (12, 0.0002) (3, 4.75) (3, 4.9) (12, 0.0002)
```

### Turn 4 — `execute_in_remote_docker` (`78241 ms` LLM)
- **Rationale**: Implement the measured-input model and use public truth only to verify calibration, each output, and the damage-gate coverage. The panel skill targets a different control-panel geometry, so the solver uses the supplied median-of-column statistics directly.
- **Alternatives Weighed**: `Use pooled historical calibration` (rejected: Historical sweeps show both drift and EMI; use only the newest supported clean sweep when current rows require history.); `Fit a separate curve and slope model` (rejected: The exact time-zero derivative must share the same parameters as the strain curves.)

```bash
cat > /holdout_task/alpha_m97_c03_batch_07_solver.py <<'PY'
"""Fit the damage-gated Prony/log-creep model to the current rheology batch."""
import json
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _align(train, test):
    raw, ref, labs, strata = (np.concatenate((train[k], test[k])) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        mask = labs == lab
        panel = ref[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        aligned[mask] = (raw[mask] - center) / scale
    cells, member = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = np.full((len(cells), aligned.shape[1]), np.nan)
    for i in range(len(cells)):
        block = aligned[member == i]
        count = np.isfinite(block).sum(axis=0)
        np.divide(np.nansum(block, axis=0), count, out=means[i], where=count > 0)
    for i, (lab, _) in enumerate(cells):
        rows = member == i
        fill = means[i].copy()
        for j in np.flatnonzero(~np.isfinite(fill)):
            pool = means[cells[:, 0] == lab, j]
            if not np.isfinite(pool).any():
                pool = means[:, j]
            fill[j] = np.nanmean(pool) if np.isfinite(pool).any() else 0.0
        aligned[rows] = np.where(np.isfinite(aligned[rows]), aligned[rows], fill)
    return np.split(aligned, [len(train['lab_id'])])


def _sweep(rows, lab):
    x, y = (np.asarray(rows[k], dtype=float)[np.asarray(rows['lab_id']) == lab]
            for k in ('cmd_stress', 'true_stress'))
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _consensus(x, y):
    """Identify the clean affine subset of a precision calibration sweep."""
    best = np.zeros(len(x), dtype=bool)
    best_key = (-1, -np.inf)
    for i, j in combinations(range(len(x)), 2):
        if x[i] == x[j]:
            continue
        gain = (y[j] - y[i]) / (x[j] - x[i])
        if gain <= 0:
            continue
        bias = y[i] - gain * x[i]
        keep = np.abs(y - (gain * x + bias)) < 0.001
        key = (int(keep.sum()), -abs(np.log(gain)) - abs(bias))
        if key > best_key:
            best, best_key = keep, key
    return best


def _calibrate(rows):
    coeff = np.empty((4, 2))
    history = None
    for lab in range(4):
        x, y = _sweep(rows, lab)
        keep = _consensus(x, y)
        if keep.sum() >= 4:
            coeff[lab] = np.polyfit(x[keep], y[keep], 1)
            continue
        # A nominal-map deviation also detects sparse EMI sweeps in which
        # two readings alone cannot distinguish a spike from an affine line.
        spike = np.any(np.abs(y - x) > 3.0)
        if keep.sum() >= 2:
            line = np.polyfit(x[keep], y[keep], 1)
            spike |= np.any(np.abs(y - np.polyval(line, x)) > 3.0)
        if len(x) >= 2 and not spike:
            coeff[lab] = np.polyfit(x, y, 1)
            continue
        if history is None:
            with open(_ARCHIVE) as stream:
                history = json.load(stream)
        for sweep in reversed(history):
            hx, hy = _sweep(sweep, lab)
            clean = _consensus(hx, hy)
            if clean.sum() >= 4:
                prior = np.polyfit(hx[clean], hy[clean], 1)
                break
        else:
            raise ValueError(f'No clean calibration sweep for channel {lab}')
        residual = y - np.polyval(prior, x)
        usable = np.abs(residual) <= 3.0
        if usable.sum() >= 2 and np.ptp(x[usable]) > 0:
            coeff[lab] = np.polyfit(x[usable], y[usable], 1)
        else:
            coeff[lab] = prior
    return coeff


def _design(sample, aligned, calibration, dt, time, crit, band):
    gain, bias = calibration[sample['lab_id']].T
    past = gain[:, None] * sample['past_stress_cmd'] + bias[:, None]
    hold = gain * sample['stress_hold_cmd'] + bias
    damage = np.maximum(np.abs(past) - crit, 0.0)**2 @ dt
    active = (band <= 0.50) & (damage > 0)
    act = np.where(active, np.tanh(damage), 0.0)
    drift = np.column_stack((np.ones(len(hold)), aligned))
    terminal = np.where(active[:, None], np.log1p(time), time)
    after = np.cumsum(dt[::-1])[::-1] - dt
    curves_static = terminal[:, :, None] * drift[:, None, :]

    def evaluate(tau):
        # The closed-form convolution is the stated zero-initial-state recurrence.
        kernel = np.exp(-after[:, None] / tau) * -np.expm1(-dt[:, None] / tau)
        z = past @ kernel
        gap = hold[:, None] - z
        q = -np.expm1(-time[:, None] / tau)
        surge = act * hold
        plateau = act * z[:, 0]
        curves = np.concatenate((gap[:, None, :] * q[None, :, :], curves_static,
                                 (surge[:, None] * q[None, :, 0])[:, :, None],
                                 (plateau[:, None] * terminal)[:, :, None]), axis=2)
        rates = np.column_stack((gap / tau, drift, surge / tau[0], plateau))
        return curves, rates

    return evaluate


def _fit(evaluate, curve_y, rate_y, dt, time):
    cm, rm = np.isfinite(curve_y), np.isfinite(rate_y)
    y = np.r_[curve_y[cm], rate_y[rm]]
    groups = np.r_[np.nonzero(cm)[1], np.full(rm.sum(), len(time))]
    ratio = np.r_[np.ones(cm.sum()), np.full(rm.sum(), np.sqrt(len(time)))]
    tau_min = max(1e-4, min(dt.min(), time[time > 0].min()) / 20)
    tau_max = 20 * max(dt.sum(), time.max())
    bounds = (np.log(tau_min), np.log(tau_max))

    def project(logt, w):
        curves, rates = evaluate(np.exp(np.sort(logt)))
        m = np.concatenate((curves[cm], rates[rm]), axis=0)
        coef = np.linalg.lstsq(m * w[:, None], y * w, rcond=None)[0]
        return m @ coef - y, coef

    def fit(start, w):
        return least_squares(lambda p: project(p, w)[0] * w, start,
                             bounds=bounds, max_nfev=80)

    step = max(np.median(dt), time.max() / 10)
    starts = [(0.5 * step, 4 * step), (step, 8 * step), (0.3 * step, time.max())]
    sol = min((fit(np.log(pair), ratio) for pair in starts), key=lambda s: s.cost)
    p, w = sol.x, ratio
    indices = [groups == i for i in np.unique(groups)]
    for _ in range(7):
        residual = project(p, w)[0]
        scale = np.empty_like(y)
        for ix in indices:
            r = residual[ix]
            scale[ix] = max(1e-5, 1.4826 * np.median(np.abs(r - np.median(r))))
        w = ratio / (scale * np.sqrt(1 + (residual / (2.5 * scale))**2))**0.5
        p = fit(p, w).x
    return np.exp(np.sort(p)), project(p, w)[1]


def solve_alpha_m97_c03_batch_07(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_train, x_test = _align(train, test)
    calibration = _calibrate(dataset['calibration_load_cells'])
    dt, time = (np.asarray(dataset[k]) for k in ('past_dt', 'eval_times'))
    crit, band = (dataset[k] for k in ('crit_stress', 'shear_banding_index'))
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    measured = {k: train[k][complete] for k in ('lab_id', 'past_stress_cmd', 'stress_hold_cmd')}
    taus, weights = _fit(_design(measured, x_train[complete], calibration, dt, time, crit, band),
                         train['strain_curves_obs'][complete] - train['strain_0'][complete, None],
                         train['instant_rate_obs'][complete], dt, time)
    curves, rates = _design(test, x_test, calibration, dt, time, crit, band)(taus)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/alpha_m97_c03_batch_07_solver.py'; spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_07.pkl','rb'))[0]; d=a['dataset']; g=a['gt']; o=m.solve_alpha_m97_c03_batch_07(d)
print('calibration',m._calibrate(d['calibration_load_cells']))
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(k,o[k], 'max abs error',np.max(np.abs(o[k]-g[j])))
for k,j in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
 e=o[k]-g[j]; print(k,'RMSE',np.sqrt(np.mean(e**2)), 'worst lab RMSE',max(np.sqrt(np.mean(e[d['test']['lab_id']==lab]**2)) for lab in range(4)), 'max abs',np.max(np.abs(e)))
for split in ['train','test']:
 s=d[split]; co=m._calibrate(d['calibration_load_cells'])[s['lab_id']]; past=co[:,0,None]*s['past_stress_cmd']+co[:,1,None]; D=np.maximum(abs(past)-d['crit_stress'],0)**2@d['past_dt']; print(split,'damage zero',np.sum(D==0),'damage range',D.min(),D.max())
PY
```
- **Container Exit Code**: `0` (`772 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_07_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
calibration [[ 0.72998864 -0.1800422 ]
 [ 0.82003341 -0.15000563]
 [ 1.16998653  0.13997717]
 [ 1.36003672  0.16995116]]
estimated_taus [0.53629568 3.58914961] max abs error 0.14485039237794428
hereditary_weights [ 0.67951999  0.55513151  0.11892167  0.12753151 -0.10810912  0.09398019
 -0.07413369] max abs error 0.032468490599237565
regime_mode_weights [0.26801344 0.24056423] max abs error 0.009435772568807538
predicted_strain RMSE 0.23579825910452532 worst lab RMSE 0.24822609541125637 max abs 1.380205729051685
initial_rate_pred RMSE 0.0755253611634875 worst lab RMSE 0.09587283177001177 max abs 0.29199255417807946
train damage zero 60 damage range 0.0 5.818202358819864
test damage zero 24 damage range 0.0 6.072463513302097

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=05f90d47e3174f33 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_05f90d47 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`32592 ms` LLM)
- **Rationale**: Subtract the exact Prony and surge terms on the public test set and compare the remaining time dependence by damage and missingness slice. A shape mismatch calls for a basis correction; an amplitude-only mismatch calls for checking panel alignment or imputation. Public truth is used only for diagnostics.
- **Alternatives Weighed**: `Retune robust fit weights immediately` (rejected: The curve error is large relative to the rate error despite accurate calibration; first test whether the exact-parameter model itself reproduces the curve shape.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/alpha_m97_c03_batch_07_solver.py'; spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_07.pkl','rb'))[0]; d=a['dataset']; g=a['gt']; tr, te=d['train'],d['test']; x,y=m._align(tr,te); t=d['eval_times']; dt=d['past_dt']; tau=g['taus']; w=np.r_[g['theta'],g['vp_theta']]; cal=np.column_stack([g['stress_gains'],g['stress_biases']]); C,R=m._design(te,y,cal,dt,t,d['crit_stress'],d['shear_banding_index'])(tau)
past=g['te_hist_true']; D=np.maximum(abs(past)-d['crit_stress'],0)**2@dt; active=D>0; z=past@(np.exp(-(np.cumsum(dt[::-1])[::-1]-dt)[:,None]/tau)*(-np.expm1(-dt[:,None]/tau))); hold=g['te_hold_true']; q=-np.expm1(-t[:,None]/tau); B=(hold[:,None]-z)*w[:2]; act=np.tanh(D); Base=B@q.T+(w[7]*act*hold)[:,None]*q[:,0]; curve=g['te_clean']-te['strain_0'][:,None]-Base; rate=g['te_true_rate0']-(B/tau).sum(1)-w[7]*act*hold/tau[0]; nominal=np.c_[np.ones(len(y)),y]@w[2:7]+w[8]*act*z[:,0]; complete=np.isfinite(te['coupon_features_raw']).all(1)
for label,mask in [('active complete',active&complete),('inactive complete',~active&complete),('active imputed',active&~complete),('inactive imputed',~active&~complete)]:
 if not mask.any(): continue
 print(label, mask.sum(), 'exact parameter model curve/rate',np.sqrt(np.mean((te['strain_0'][mask,None]+C[mask]@w-g['te_clean'][mask])**2)),np.sqrt(np.mean((R[mask]@w-g['te_true_rate0'][mask])**2)))
 for name,terminal in [('linear',t),('log',np.log1p(t))]:
  implied=curve[mask]@terminal/(terminal@terminal); print(name,'shape residual',np.sqrt(np.mean((curve[mask]-implied[:,None]*terminal)**2)), 'rate identity max',np.max(abs(implied-rate[mask])), 'alignment residual',np.sqrt(np.mean((implied-nominal[mask])**2)))
 print('rate regression',np.linalg.lstsq(np.c_[np.ones(mask.sum()),y[mask]], (rate-w[8]*act*z[:,0])[mask],rcond=None)[0])
for lab in range(4):
 panel=np.concatenate([tr['ref_coupons'][tr['lab_id']==lab],te['ref_coupons'][te['lab_id']==lab]]); center=np.nanmean(panel,0); scale=np.nanstd(panel,0); print('panel',lab,'median center scale',np.median(center),np.median(scale),'ranges',np.percentile(center,[0,25,75,100]),np.percentile(scale,[0,25,75,100]))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust IRLS variable projection relaxation' 2>&1 | head -8
```
- **Container Exit Code**: `0` (`534 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
96 0.43866044 0.63958903 0.72717989] [1.05793132 1.22722017 1.38128426 1.51588019]
panel 2 median center scale -0.26154455925718845 0.9461353425208083 ranges [-0.7519177  -0.32376089 -0.14306485  0.12553826] [0.7531788  0.86038992 0.99665952 1.1780364 ]
panel 3 median center scale 0.8340474203075848 1.3831985739961703 ranges [0.51297863 0.73712497 0.99169123 1.18363687] [1.13914846 1.30932648 1.4447492  1.65953834]
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=05f90d47e3174f33 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_05f90d47 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`39735 ms` LLM)
- **Rationale**: Correct the falsified basis before tuning estimation. Compare all five outputs and report public errors by lab, damage, missingness, and horizon. Refit while holding each lab out to detect calibration or panel errors hidden by pooled fitting, and check that the predicted initial rate is the derivative of the strain model on both sides of the banding threshold.
- **Alternatives Weighed**: `Refit robust weights on the old basis` (rejected: Even independently fitted amplitudes cannot reproduce the zero-damage curve shape with the linear basis.)
- **ATMS NOGOOD DECLARED**: Refuted `Within the low-banding regime, zero-damage coupons use a linear terminal term.` | Witness: `For 23 complete zero-damage test coupons, the isolated creep residual has 0.0630 shape RMSE under the best per-coupon linear amplitude but 1.15e-16 under log(1+t), whose fitted amplitude also matches the initial-rate remainder to 4.44e-16.` | Blanket Diagnosis: `The previous model conflated the banding regime with damage activation; neither load-cell calibration nor panel alignment can account for the isolated time-shape mismatch.` | Sublated Pivot: `Use the logarithmic terminal basis throughout the low-banding regime and retain the damage gate on the surge and crystallization corrections.`

```bash
set -e
p=/holdout_task/alpha_m97_c03_batch_07_solver.py
sed -i 's/    terminal = np.where(active\[:, None\], np.log1p(time), time)/    # The low-banding regime sets the creep basis; damage gates only its corrections.\n    terminal = np.log1p(time) if band <= 0.50 else time/; s/    curves_static = terminal\[:, :, None\] \* drift\[:, None, :\]/    curves_static = terminal[None, :, None] * drift[:, None, :]/' "$p"
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/alpha_m97_c03_batch_07_solver.py'; spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_07.pkl','rb'))[0]; d=a['dataset']; g=a['gt']; tr,te=d['train'],d['test']; t=d['eval_times']; dt=d['past_dt']; crit=d['crit_stress']; band=d['shear_banding_index']; o=m.solve_alpha_m97_c03_batch_07(d)
cal=m._calibrate(d['calibration_load_cells']); print('max calibration coefficient error',np.max(abs(cal-np.c_[g['stress_gains'],g['stress_biases']])))
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(k,o[k], 'max abs error',np.max(abs(o[k]-g[j])))
D=np.maximum(abs(g['te_hist_true'])-crit,0)**2@dt
complete_te=np.isfinite(te['coupon_features_raw']).all(1)
E=o['predicted_strain']-g['te_clean']; R=o['initial_rate_pred']-g['te_true_rate0']; rms=lambda z:np.sqrt(np.mean(z**2))
for name, mask in [('all',np.ones(len(R),bool)), *[(f'lab {i}',te['lab_id']==i) for i in range(4)], ('active complete',(D>0)&complete_te),('inactive complete',(D==0)&complete_te),('imputed',~complete_te)]:
 print(name, 'count',mask.sum(),'curve/rate RMSE',rms(E[mask]),rms(R[mask]))
print('per-horizon RMSE',np.sqrt(np.mean(E**2,axis=0)))
xt,xe=m._align(tr,te); complete=np.isfinite(tr['coupon_features_raw']).all(1)
for lab in range(4):
 fit=complete&(tr['lab_id']!=lab); hold=complete&(tr['lab_id']==lab)
 def sub(mask): return {k:tr[k][mask] for k in ('lab_id','past_stress_cmd','stress_hold_cmd')}
 def design(mask): return m._design(sub(mask),xt[mask],cal,dt,t,crit,band)
 cy=tr['strain_curves_obs']-tr['strain_0'][:,None]; ry=tr['instant_rate_obs']; tau,w=m._fit(design(fit),cy[fit],ry[fit],dt,t); C,V=design(hold)(tau)
 e=C@w-cy[hold]; r=V@w-ry[hold]; print('leave-lab-out',lab,'median abs strain/rate residual',np.nanmedian(abs(e)),np.nanmedian(abs(r)),'tau',tau)
w=np.r_[o['hereditary_weights'],o['regime_mode_weights']]; tau=o['estimated_taus']; eps=1e-7
for b in (0.50,0.500001):
 C,V=m._design(te,xe,cal,dt,np.array([0.,eps]),crit,b)(tau)
 assert np.max(abs(C[:,0]@w))<1e-12
 assert np.max(abs((C[:,1]@w)/eps-V@w))<2e-6
print('zero strain increment and analytic derivative verified on both sides of band threshold')
PY
```
- **Container Exit Code**: `0` (`974 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_07_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
max calibration coefficient error 4.8837256654832295e-05
estimated_taus [0.53408946 3.73259622] max abs error 0.0014037843815373208
hereditary_weights [ 0.67972455  0.55263806  0.14240606  0.1592183  -0.12948952  0.11865458
 -0.08851772] max abs error 0.0026380557350680345
regime_mode_weights [0.26950318 0.25247618] max abs error 0.002476183731290771
all count 72 curve/rate RMSE 0.056601197119679925 0.04302128519241159
lab 0 count 11 curve/rate RMSE 0.06395326540947373 0.048822046811605395
lab 1 count 25 curve/rate RMSE 0.008611411320239562 0.006438039107596123
lab 2 count 16 curve/rate RMSE 0.09043069902760718 0.06856694918838084
lab 3 count 20 curve/rate RMSE 0.0514677394279349 0.03923090790643562
active complete count 43 curve/rate RMSE 0.011086573494023333 0.008825181988111198
inactive complete count 23 curve/rate RMSE 0.007965490948864375 0.00622827047158379
imputed count 6 curve/rate RMSE 0.19318450948901741 0.14663936739205005
per-horizon RMSE [0.01130897 0.0264445  0.03765069 0.04653408 0.05389193 0.0601703
 0.06564844 0.07050655 0.07487266 0.07883618]
leave-lab-out 0 median abs strain/rate residual 0.015587274382723892 0.011630920006963474 tau [0.5345843  3.76385147]
leave-lab-out 1 median abs strain/rate residual 0.015547236931136188 0.014276635180670905 tau [0.53467321 3.81144613]
leave-lab-out 2 median abs strain/rate residual 0.012871780576861225 0.009423005964795034 tau [0.53409291 3.71194258]
leave-lab-out 3 median abs strain/rate residual 0.014081703809395163 0.017581200516875783 tau [0.53364327 3.69299981]
zero strain increment and analytic derivative verified on both sides of band threshold

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c7aaae5f672d4f71 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c7aaae5f (directory).
```

### Turn 7 — `execute_in_remote_docker` (`32024 ms` LLM)
- **Rationale**: The verified candidate fixes the zero-damage time basis. The public split mainly exercises archive fallback; a mixed sweep checks whether the solver incorrectly overrides clean current data or propagates a sparse EMI spike. A failure would call for a calibration change; otherwise the verified model is ready to submit.
- **Alternatives Weighed**: `Replace cell-mean imputation to lower the public missing-feature error` (rejected: Cell-balanced means are specified by the task, and the exact-parameter diagnostic already showed that this slice's error mostly comes from unobserved features rather than the fitted rheology.)

```bash
set -e
python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
base='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',base+'alpha_m97_c03_batch_07_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
items=pickle.load(open(base+'public_alpha_m97_c03_batch_07.pkl','rb'))
for i,item in enumerate(items):
 d,g=item['dataset'],item['gt']; o=m.solve_alpha_m97_c03_batch_07(d)
 for k in o: assert np.isfinite(o[k]).all(),k
 assert np.all(np.diff(o['estimated_taus'])>0)
 D=np.maximum(np.abs(g['te_hist_true'])-d['crit_stress'],0)**2@d['past_dt']; complete=np.isfinite(d['test']['coupon_features_raw']).all(1)
 e=o['predicted_strain']-g['te_clean']; r=o['initial_rate_pred']-g['te_true_rate0']
 rms=lambda x: np.sqrt(np.mean(x*x))
 for label,mask in [('all',np.ones(len(r),bool)),('zero-damage complete',(D==0)&complete),('positive-damage complete',(D>0)&complete),('imputed',~complete)]:
  if mask.any(): print(i,label,int(mask.sum()),'strain/rate RMSE',rms(e[mask]),rms(r[mask]))
 print('max parameter errors',*[np.max(abs(o[k]-g[j])) for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]])
# Separate the four-clean direct fit, two-point direct fit, sparse EMI fallback,
# and absent-channel fallback in one mixed sweep on the measured stress range.
c=items[0]['dataset']['calibration_load_cells']; ref=m._calibrate(c)
x=np.array([-2.,-1.,0.,1.,2.]); y=0.91*x+0.21; y[-1]+=4.9
x1=np.array([-2.,2.]); y1=0.86*x1-0.12
x2=np.array([-1.5,0.,1.5]); y2=np.polyval(ref[2],x2); y2[1]+=4.9
mixed={'lab_id':np.repeat([0,1,2],[5,2,3]),'cmd_stress':np.r_[x,x1,x2],'true_stress':np.r_[y,y1,y2]}
z=m._calibrate(mixed)
np.testing.assert_allclose(z[:2],[[0.91,0.21],[0.86,-0.12]],atol=1e-12)
np.testing.assert_allclose(z[2:],ref[2:],atol=2e-4)
print('Calibration passed direct clean, direct sparse, sparse EMI, and absent-channel cases.')
PY
```
- **Container Exit Code**: `0` (`838 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
0 all 72 strain/rate RMSE 0.056601197119679925 0.04302128519241159
0 zero-damage complete 23 strain/rate RMSE 0.007965490948864375 0.00622827047158379
0 positive-damage complete 43 strain/rate RMSE 0.011086573494023333 0.008825181988111198
0 imputed 6 strain/rate RMSE 0.19318450948901741 0.14663936739205005
max parameter errors 0.0014037843815373208 0.0026380557350680345 0.002476183731290771
Calibration passed direct clean, direct sparse, sparse EMI, and absent-channel cases.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c7aaae5f672d4f71 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c7aaae5f (directory).
```

### Turn 8 — `submit_completed_task` (`13898 ms` LLM)
- **Rationale**: The exact final artifact passed isolated schema validation, calibration stress tests, and the host scientific probe. The zero-damage basis correction is verified, and further tuning would not recover the missing coupon measurements.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_07_solver.py': 'FILE:7945'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_07_solver.py`. The public schema passed. Public RMSE was 0.0566 for strain and 0.0430 for initial rate; the six coupons requiring imputation had the largest errors.`

### Turn 9 — `execute_in_remote_docker` (`38971 ms` LLM)
- **Rationale**: A recurrence or derivative mismatch calls for a basis fix. A material signed bias on complete coupons confined to one lab calls for an alignment or calibration investigation. The audit separates those failure modes from the already measured loss due to missing features.
- **Alternatives Weighed**: `Add rational, pairwise, or relay terms and compare pooled fit` (rejected: The stated model is a deterministic two-branch recurrence with a scalar regime gate, not a relay, PDE, or classification model. An independent algebraic check can detect a missing or mis-signed term without fitting unsupported operators.)

```bash
set -e
cat > /tmp/rheology_batch07_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root + 'alpha_m97_c03_batch_07_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_alpha_m97_c03_batch_07.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    d, gt = item['dataset'], item['gt']
    tr, te = d['train'], d['test']
    print('dataset fields', sorted(d), 'coupon fields', sorted(te))
    c = d['calibration_load_cells']
    print('current calibration rows by lab', np.bincount(c['lab_id'], minlength=4))
    print('calibration command range', np.min(c['cmd_stress']), np.max(c['cmd_stress']))
    out = m.solve_alpha_m97_c03_batch_07(d)
    tau = out['estimated_taus']
    w = np.r_[out['hereditary_weights'], out['regime_mode_weights']]
    _, x = m._align(tr, te)
    cal = m._calibrate(c)
    gain, bias = cal[te['lab_id']].T
    past = te['past_stress_cmd'] * gain[:, None] + bias[:, None]
    hold = te['stress_hold_cmd'] * gain + bias
    z = np.zeros((len(hold), 2))
    damage = np.zeros(len(hold))
    for s, dt in enumerate(d['past_dt']):
        decay = np.exp(-dt / tau)
        z = decay * z + (1 - decay) * past[:, s, None]
        damage += np.maximum(0, np.abs(past[:, s]) - d['crit_stress'])**2 * dt
    gap = hold[:, None] - z
    t = d['eval_times']
    for band in (d['shear_banding_index'], 0.50, 0.500001):
        act = np.tanh(damage) if band <= 0.5 else np.zeros_like(damage)
        drift = w[2] + x @ w[3:7] + w[8] * act * z[:, 0]
        prony = (gap * w[:2]) @ (1 - np.exp(-t[:, None] / tau)).T
        surge = (w[7] * act * hold)[:, None] * (1 - np.exp(-t / tau[0]))
        explicit = prony + surge + drift[:, None] * (np.log1p(t) if band <= 0.5 else t)
        initial = (gap * w[:2] / tau).sum(1) + drift + w[7] * act * hold / tau[0]
        C, V = m._design(te, x, cal, d['past_dt'], t, d['crit_stress'], band)(tau)
        np.testing.assert_allclose(C @ w, explicit, atol=1e-12, rtol=0)
        np.testing.assert_allclose(V @ w, initial, atol=1e-12, rtol=0)
    # Look for a signed lab effect that pooled RMSE could hide. Missing-feature
    # coupons are a separate slice, not evidence of a lab offset.
    observed = np.isfinite(te['coupon_features_raw']).all(1)
    ce = out['predicted_strain'] - gt['te_clean']
    re = out['initial_rate_pred'] - gt['te_true_rate0']
    for lab in np.unique(te['lab_id']):
        k = observed & (te['lab_id'] == lab)
        print('complete lab', lab, int(k.sum()), 'signed terminal/rate bias', np.mean(ce[k, -1]), np.mean(re[k]), 'terminal/rate RMSE', np.sqrt(np.mean(ce[k, -1]**2)), np.sqrt(np.mean(re[k]**2)))
    print('explicit memory and curve/rate identities passed at the measured band and both threshold sides')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_batch07_audit.py
```
- **Container Exit Code**: `0` (`574 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
dataset fields ['calibration_load_cells', 'crit_stress', 'eval_times', 'past_dt', 'shear_banding_index', 'test', 'train', 'yield_stress'] coupon fields ['aux_chamber_telemetry', 'coupon_features_raw', 'lab_id', 'past_stress_cmd', 'ref_coupons', 'strain_0', 'stratum', 'stress_hold_cmd']
current calibration rows by lab [2 0 0 0]
calibration command range -1.3 1.4
complete lab 0 10 signed terminal/rate bias 0.004615438519234708 0.0021546785943770573 terminal/rate RMSE 0.025468220593773388 0.01410850332191974
complete lab 1 25 signed terminal/rate bias -0.0008896073513633496 -0.0006465819459840488 terminal/rate RMSE 0.011956051919997247 0.006438039107596123
complete lab 2 14 signed terminal/rate bias 0.007577423459293373 0.004570761906066469 terminal/rate RMSE 0.010397641843388542 0.007054824573312604
complete lab 3 17 signed terminal/rate bias -0.0004851486653679697 -0.0007953024008453089 terminal/rate RMSE 0.009397611559995899 0.005518477366640787
explicit memory and curve/rate identities passed at the measured band and both threshold sides

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c7aaae5f672d4f71 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c7aaae5f (directory).
```

### Turn 10 — `submit_completed_task` (`15868 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated schema validation, calibration stress cases, and the independent recurrence and derivative audit at the measured banding value and on both sides of the threshold. The final audit found no additional discrepancy that warrants changing the verified solver.
- **Alternatives Weighed**: `Add lab-specific residual corrections` (rejected: Complete-coupon signed initial-rate bias was at most 0.0046 across labs, and lab corrections would not recover the missing features that dominate the worst slice.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_07_solver.py': 'FILE:7945'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_07_solver.py`. The public schema passed. Public RMSE was 0.0566 for strain and 0.0430 for initial rate. The six coupons requiring feature imputation had the largest errors: 0.1932 and 0.1466, respectively.`

### Turn 11 — `execute_in_remote_docker` (`39646 ms` LLM)
- **Rationale**: Test whether excluding incomplete training coupons harms transfer across labs. If including them materially improves the worst held-out lab without hurting parameter recovery, reconsider the fit. Exact public parameters are used only for diagnosis: if the imputed-slice error persists, retuning the physical fit will not address the bottleneck.
- **Alternatives Weighed**: `Repeat the recurrence audit or add rational, PDE, or relay terms` (rejected: The independent recurrence and derivative checks already passed, and those operators do not occur in the supplied rheology model.)

```bash
set -e
cat > /tmp/rheology_batch07_missingness_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root + 'alpha_m97_c03_batch_07_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_alpha_m97_c03_batch_07.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    d, g = item['dataset'], item['gt']
    tr, te = d['train'], d['test']
    xt, xe = m._align(tr, te)
    cal = m._calibrate(d['calibration_load_cells'])
    dt, t = d['past_dt'], d['eval_times']
    args = cal, dt, t, d['crit_stress'], d['shear_banding_index']
    complete = np.isfinite(tr['coupon_features_raw']).all(1)
    missing = ~np.isfinite(te['coupon_features_raw']).all(1)
    lab = tr['lab_id']
    cy = tr['strain_curves_obs'] - tr['strain_0'][:, None]
    ry = tr['instant_rate_obs']

    def design(mask):
        subset = {k: tr[k][mask] for k in ('lab_id', 'past_stress_cmd', 'stress_hold_cmd')}
        return m._design(subset, xt[mask], *args)

    print('Incomplete training coupons by lab:', np.bincount(lab[~complete], minlength=4))
    for name, eligible in [('complete-only', complete), ('include-imputed', np.ones(len(lab), bool))]:
        folds = []
        for held in np.unique(lab):
            fit, val = eligible & (lab != held), complete & (lab == held)
            taus, weights = m._fit(design(fit), cy[fit], ry[fit], dt, t)
            C, R = design(val)(taus)
            folds.append([np.nanmedian(abs(C @ weights - cy[val])), np.nanmedian(abs(R @ weights - ry[val]))])
        print(name, 'leave-lab-out median absolute strain/rate errors', np.round(folds, 6).tolist(), 'worst', np.round(np.max(folds, axis=0), 6).tolist())
        taus, weights = m._fit(design(eligible), cy[eligible], ry[eligible], dt, t)
        C, R = m._design(te, xe, *args)(taus)
        e = te['strain_0'][:, None] + C @ weights - g['te_clean']
        r = R @ weights - g['te_true_rate0']
        for label, mask in [('complete', ~missing), ('imputed', missing)]:
            if mask.any():
                print(name, label, 'test strain/rate RMSE', np.sqrt(np.mean(e[mask]**2)), np.sqrt(np.mean(r[mask]**2)))
        print(name, 'max absolute parameter errors (tau, hereditary, regime)', np.max(abs(taus - g['taus'])), np.max(abs(weights[:7] - g['theta'])), np.max(abs(weights[7:] - g['vp_theta'])))
    exact_args = (np.c_[g['stress_gains'], g['stress_biases']], *args[1:])
    C, R = m._design(te, xe, *exact_args)(g['taus'])
    w = np.r_[g['theta'], g['vp_theta']]
    if missing.any():
        e = te['strain_0'][:, None] + C @ w - g['te_clean']
        r = R @ w - g['te_true_rate0']
        print('Imputed test strain/rate RMSE with exact physical parameters and unchanged panel imputation', np.sqrt(np.mean(e[missing]**2)), np.sqrt(np.mean(r[missing]**2)))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_batch07_missingness_audit.py
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1211 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Incomplete training coupons by lab: [3 6 3 6]
complete-only leave-lab-out median absolute strain/rate errors [[0.015587, 0.011631], [0.015547, 0.014277], [0.012872, 0.009423], [0.014082, 0.017581]] worst [0.015587, 0.017581]
complete-only complete test strain/rate RMSE 0.010108913609726647 0.008016277112232524
complete-only imputed test strain/rate RMSE 0.19318450948901741 0.14663936739205005
complete-only max absolute parameter errors (tau, hereditary, regime) 0.0014037843815373208 0.0026380557350680345 0.002476183731290771
include-imputed leave-lab-out median absolute strain/rate errors [[0.017123, 0.012893], [0.016336, 0.019442], [0.013578, 0.012808], [0.01506, 0.018879]] worst [0.017123, 0.019442]
include-imputed complete test strain/rate RMSE 0.010278072256914867 0.008605660555283457
include-imputed imputed test strain/rate RMSE 0.1939197943751437 0.1486420226928309
include-imputed max absolute parameter errors (tau, hereditary, regime) 0.1954179087146466 0.00691396778354092 0.003879892746220359
Imputed test strain/rate RMSE with exact physical parameters and unchanged panel imputation 0.19292689778553063 0.14731986811058298
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c7aaae5f672d4f71 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c7aaae5f (directory).
```

### Turn 12 — `submit_completed_task` (`14347 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated schema validation, calibration stress cases, and independent recurrence and derivative checks. The latest held-out-lab comparison supports retaining the complete-coupon fit and the specified cell-mean imputation.
- **Alternatives Weighed**: `Include imputed training coupons in the physical fit` (rejected: Worst leave-one-lab-out median absolute strain/rate errors rose from 0.01559/0.01758 to 0.01712/0.01944, with no improvement on the imputed test slice.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_07_solver.py`. The public schema passed. Public strain and initial-rate RMSE were 0.0566 and 0.0430, respectively. The six coupons requiring feature imputation had the largest errors, which persisted even with exact physical parameters.`

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
