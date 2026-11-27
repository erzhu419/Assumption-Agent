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
| **Turns Used / Max Turns** | `15 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `466,901` (`452,465` / `101,248` / `14,436`) |
| **Total Episode Wall Time (s)** | `482.53s` |

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

### Turn 1 — `execute_in_remote_docker` (`18312 ms` LLM)
- **Rationale**: Inspect the input schema, calibration coverage, and prior alignment conventions before fitting; a mismatch would change which fields or calibration fallback the solver uses.
- **Alternatives Weighed**: `Reuse the prior solver unchanged` (rejected: The current batch specifies a new activated plateau term, so prior code is a reference for shared conventions rather than a drop-in solver.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -l /tmp/hegel_continual_archive && sed -n '1,260p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py && python3 - <<'PY'
import pickle,numpy as np
with open('public_alpha_m97_c03_batch_01.pkl','rb') as f: d=pickle.load(f)
def describe(x,depth=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print(' '*depth,k,end=': '); describe(v,depth+2)
 elif isinstance(x,np.ndarray):
  print(x.shape,x.dtype,x.ravel()[:6])
 elif isinstance(x,(list,tuple)):
  print(type(x).__name__,len(x),repr(x[:1])[:350])
 else: print(type(x).__name__,repr(x)[:350])
describe(d)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation variable projection'
```
- **Container Exit Code**: `0` (`457 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(build, train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs'], dt, t)
    curve, rate = _design(test, xte, gain, bias, dt, t)(logtau)
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': beta,
            'regime_mode_weights': np.zeros(2), 'predicted_strain': test['strain_0'][:, None] + curve @ beta,
            'initial_rate_pred': rate @ beta}
list 1 [{'dataset': {'past_dt': array([0.4917, 0.5023, 0.5442, 0.7304, 0.4565, 0.765 , 0.2965, 0.6942,
       0.5869, 0.3435, 0.3902, 0.5082, 0.3785, 0.7877, 0.3776, 0.2105,
       0.2875, 0.2881]), 'eval_times': array([0.3   , 0.8444, 1.3889, 1.9333, 2.4778, 3.0222, 3.5667, 4.1111,
       4.6556, 5.2   ]), 'crit_stress': 2.65, 'yield_stress': 2.65, 'shea
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 2 — `execute_in_remote_docker` (`18593 ms` LLM)
- **Rationale**: Recover the omitted reference-panel and response schemas, and check per-channel calibration residuals. The results determine how to pool references and whether any channel requires archive calibration before fitting the specified strain and initial-rate model.
- **Alternatives Weighed**: `Carry forward the archived calibration parameters` (rejected: Current-batch clean measurements take priority; per-channel residuals determine whether the archive fallback is needed.)

```bash
cd /holdout_task && sed -n '1,125p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py && python3 - <<'PY'
import pickle,numpy as np
item=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d=item['dataset']; print('item keys',list(item),'dataset keys',list(d));
for k in ('train','test','calibration_load_cells'):
 print(k)
 for n,v in d[k].items():
  if isinstance(v,dict): print(n,{j:(np.shape(a),np.asarray(a).ravel()[:5]) for j,a in v.items()}); continue
  a=np.asarray(v); print(n,a.shape,a.dtype,a.ravel()[:5], 'nan', int(np.isnan(a).sum()) if a.dtype.kind=='f' else '')
c=d['calibration_load_cells']
for l in np.unique(c['lab_id']):
 m=c['lab_id']==l; x=c['cmd_stress'][m]; y=c['true_stress'][m]; p=np.polyfit(x,y,1); print('cal',l,len(x),p,np.round(y-np.polyval(p,x),5))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic'
```
- **Container Exit Code**: `0` (`482 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-0.593888 -0.376375 -0.415761  0.396469] nan 0
aux_chamber_telemetry (180, 3) float64 [-0.297735  0.10502  -0.26042  -1.220394  0.080705] nan 0
strain_curves_obs (180, 10) float64 [0.07152707 0.2694247  0.38389393 0.41258096 0.49431274] nan 0
instant_rate_obs (180,) float64 [ 0.76519135 -1.90563337 -1.66500632  3.09979376  3.47047517] nan 0
test
lab_id (72,) int64 [0 1 2 3 0] nan 
stratum (72,) int64 [0 1 0 1 2] nan 
coupon_features_raw (72, 4) float64 [-0.79389545 -2.29779507 -0.39013796 -0.84175003 -0.55163686] nan 8
ref_coupons (72, 32) float64 [-0.8255135  -0.00757547 -0.77430999 -0.20369185  1.04878858] nan 95
past_stress_cmd (72, 18) float64 [ 0.31217379 -1.28492709  0.829346   -2.50153204 -2.69981006] nan 0
stress_hold_cmd (72,) float64 [-1.18647364  2.21660854 -1.43158889 -0.22731053  0.39403178] nan 0
strain_0 (72,) float64 [ 0.052036 -0.500013 -0.481813  0.101502  0.322033] nan 0
aux_chamber_telemetry (72, 3) float64 [-0.175019  0.782023  1.124864  1.075463  1.483753] nan 0
calibration_load_cells
lab_id (40,) int64 [0 0 0 0 0] nan 
cmd_stress (40,) float64 [ 2.15143   1.121894 -0.424697 -1.698186  1.597457] nan 0
true_stress (40,) float64 [ 2.89522328  1.56727542 -0.42798096 -2.07055034  2.18078372] nan 0
cal 0 10 [1.28998908 0.12000143] [-1.0e-04  4.0e-05 -1.3e-04  9.0e-05  8.0e-05  2.2e-04  1.0e-05 -9.0e-05
 -1.5e-04  3.0e-05]
cal 1 10 [ 0.82003896 -0.15002605] [ 0.0e+00  3.0e-05 -1.0e-05 -7.0e-05 -1.0e-04  3.0e-05  1.1e-04  1.0e-04
  5.0e-05 -1.4e-04]
cal 2 10 [1.16998093 0.14000924] [-6.0e-05 -1.2e-04  7.0e-05  3.0e-05  4.0e-05 -9.0e-05 -1.0e-05  1.0e-05
  1.6e-04 -3.0e-05]
cal 3 10 [ 0.76000147 -0.10998819] [ 4.0e-05  2.0e-05 -4.0e-05 -1.0e-05 -1.0e-05  1.0e-05 -2.0e-05  4.0e-05
 -1.7e-04  1.5e-04]
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 3 — `execute_in_remote_docker` (`17120 ms` LLM)
- **Rationale**: Read only the previously truncated alignment and fit code and the archive schema. A different reference-panel layout or archive format would change the implementation; the skill description determines whether the available alignment primitive actually matches this panel.
- **Alternatives Weighed**: `Use the archived calibration for the public batch` (rejected: All four channels have ten clean in-batch measurements, so the requested direct fits take precedence.)

```bash
sed -n '58,106p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py; sed -n '121,188p' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py; head -c 1800 /tmp/hegel_continual_archive/archive_calibration_history.json; printf '\n'; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -c 2300
```
- **Container Exit Code**: `0` (`324 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
          "doc_summary": "Return signal values in control-referenced coordinates, in input row order."
        }
      ],
      "index_cards": [
        "autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel",
        "autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespeci
```

### Turn 4 — `execute_in_remote_docker` (`57453 ms` LLM)
- **Rationale**: Build the physical model with a two-parameter profiled fit and group-scaled robust refinement. The schema check detects malformed output; inspecting the archive structure determines whether the sparse-calibration path needs a correction before it is exercised.
- **Alternatives Weighed**: `Use the control-panel alignment skill` (rejected: Its shared scanner-gain model does not match the specified per-lab median reference statistics and cell-balanced imputation.); `Sort fitted relaxation times after optimization` (rejected: The crystallization terms specifically use branch zero, so swapping branches would change the predictions.)

```bash
set -e
cd /holdout_task
f=alpha_m97_c03_batch_01_solver.py
{ printf '%s\n' '"""Calibrated two-branch hereditary model with crystallization-induced arrest."""' 'import json' '' 'import numpy as np' 'from scipy.optimize import least_squares' '' '_ARCHIVE = "/tmp/hegel_continual_archive/archive_calibration_history.json"' '' ''; sed -n '/^def _align(/,/^def _design(/{ /^def _design(/!p; }' /tmp/hegel_continual_archive/alpha_m97_c03_s00_solver.py; } > "$f"
cat >> "$f" <<'PY'
def _calibrate(cal):
    gain, bias = np.ones(4), np.zeros(4)
    history = None
    for lab in range(4):
        x, y = (np.asarray(cal[key])[cal['lab_id'] == lab]
                for key in ('cmd_stress', 'true_stress'))
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        clean = np.ones(len(x), dtype=bool)
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            dx = x[j] - x[i]
            use = dx != 0
            if use.any():
                slope = np.median((y[j] - y[i])[use] / dx[use])
                r = y - slope*x
                clean = np.abs(r - np.median(r)) <= 3.0
        if len(x) < 2 or not clean.all():
            if history is None:
                with open(_ARCHIVE) as f:
                    history = json.load(f)['calibration_load_cells']
            if clean.sum() < 4:
                ref_lab, ref_x, ref_y = (np.asarray(history[key])
                                         for key in ('lab_id', 'cmd_stress', 'true_stress'))
                x, y = ref_x[ref_lab == lab], ref_y[ref_lab == lab]
                clean = np.ones(len(x), dtype=bool)
        gain[lab], bias[lab] = np.linalg.lstsq(
            np.column_stack((x[clean], np.ones(clean.sum()))), y[clean], rcond=None)[0]
    return gain, bias


def _design(split, x, gain, bias, dt, t, crit, banding):
    g, b = gain[split['lab_id']], bias[split['lab_id']]
    hist = g[:, None]*split['past_stress_cmd'] + b[:, None]
    hold = g*split['stress_hold_cmd'] + b
    damage = np.maximum(np.abs(hist) - crit, 0)**2 @ dt
    arrest = (banding <= .50) & (damage > 0)
    act = np.where(arrest, np.tanh(damage), 0.)
    terminal = np.where(arrest[:, None], np.log1p(t), t)
    flow = np.column_stack((np.ones(len(x)), x))
    age = np.cumsum(dt[::-1])[::-1] - dt
    curve = np.zeros((len(x), len(t), 9))
    rate = np.zeros((len(x), 9))
    curve[:, :, 2:7] = terminal[:, :, None]*flow[:, None, :]
    rate[:, 2:7] = flow

    def evaluate(log_tau):
        tau = np.exp(log_tau)
        kernel = np.exp(-age[:, None]/tau)*(-np.expm1(-dt[:, None]/tau))
        z = hist @ kernel
        gap = hold[:, None] - z
        rise = -np.expm1(-t[:, None]/tau)
        curve[:, :, :2] = gap[:, None, :]*rise
        rate[:, :2] = gap/tau
        curve[:, :, 7] = (act*hold)[:, None]*rise[:, 0]
        curve[:, :, 8] = (act*z[:, 0])[:, None]*terminal
        rate[:, 7] = act*hold/tau[0]
        rate[:, 8] = act*z[:, 0]
        return curve, rate

    return evaluate, arrest


def _mad(r):
    return 1.4826*np.median(np.abs(r - np.median(r)))


def _fit(evaluate, curve_y, rate_y, dt, t):
    cm, rm = np.isfinite(curve_y), np.isfinite(rate_y)
    y = np.r_[curve_y[cm], rate_y[rm]]
    n = cm.sum()
    lo = np.log(min(dt.min(), t[t > 0].min())/20)
    hi = np.log(20*max(dt.sum(), t.max()))

    def matrix(log_tau):
        curve, rate = evaluate(log_tau)
        return np.concatenate((curve[cm], rate[rm]))

    w = np.r_[np.ones(n), np.full(rm.sum(), .5)]

    def project(log_tau):
        M = matrix(log_tau)*w[:, None]
        return np.linalg.lstsq(M, y*w, rcond=None)[0], M

    def profile(log_tau):
        beta, M = project(log_tau)
        return M @ beta - y*w

    starts = np.array([[.25, 2.], [.65, 4.], [1.4, 7.], [3., .4], [7., 1.4]])
    sols = [least_squares(profile, np.clip(np.log(start), lo, hi),
                          bounds=(lo, hi), max_nfev=60) for start in starts]
    best = min(sols, key=lambda r: r.cost)
    beta, _ = project(best.x)
    r = matrix(best.x) @ beta - y
    group_scale = [max(_mad(r[:n]), 1.e-6), max(_mad(r[n:]), 1.e-6)]
    scales = np.repeat(group_scale, [n, rm.sum()])

    def residual(p):
        return (matrix(p[:2]) @ p[2:] - y)/scales

    best = least_squares(residual, np.r_[best.x, beta], loss='soft_l1', f_scale=2.,
                         bounds=(np.r_[[lo]*2, [-np.inf]*9],
                                 np.r_[[hi]*2, [np.inf]*9]), max_nfev=100)
    return best.x[:2], best.x[2:]


def solve_alpha_m97_c03_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = dataset['past_dt'], dataset['eval_times']
    gain, bias = _calibrate(dataset['calibration_load_cells'])
    xtr, xte = _align(train, test)
    args = gain, bias, dt, t, dataset['crit_stress'], dataset['shear_banding_index']
    build, _ = _design(train, xtr, *args)
    log_tau, beta = _fit(build, train['strain_curves_obs'] - train['strain_0'][:, None],
                        train['instant_rate_obs'], dt, t)
    build, _ = _design(test, xte, *args)
    curve, rate = build(log_tau)
    return {'estimated_taus': np.exp(log_tau), 'hereditary_weights': beta[:7],
            'regime_mode_weights': beta[7:],
            'predicted_strain': test['strain_0'][:, None] + curve @ beta,
            'initial_rate_pred': rate @ beta}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
python3 - <<'PY'
import json
p=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
print('archive fields', list(p))
for k,v in p.items():
 if isinstance(v,dict): print(k,[(j,len(a) if isinstance(a,list) else type(a).__name__) for j,a in v.items()])
PY
```
- **Container Exit Code**: `1` (`536 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
archive fields [{'episode_short_name': 'alpha_m97_c03_s00', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [0.893607, -0.483769, -1.39264, 1.611445, -1.810815, -1.870654, 1.48082, 1.268821, -0.114529, 0.252739, -1.204476, -0.664616, -0.154159, -0.613549, -1.259125, -2.011382, 2.18392, -0.23172, 1.98347, -2.000907, -0.486537, -0.139492, -0.631342, -1.874772, -0.708257, 2.13352, 0.386634, 1.10423, -1.407517, 1.70372, 1.521295, 1.987385, 1.659545, -0.294044, 0.359165, -0.017198, 1.319848, 0.959274, 1.866895, -0.480321], 'true_stress': [0.893713, -0.483774, -1.392656, 1.611564, -1.810667, -1.870622, 1.480773, 1.268837, -0.114695, 0.252685, -1.204516, -0.664704, -0.154113, -0.613571, -1.259248, -2.011128, 2.183777, -0.231673, 1.983509, -2.001075, -0.486729, -0.139381, -0.631253, -1.874747, -0.708398, 2.133538, 0.386675, 1.104402, -1.407596, 1.703866, 1.521337, 1.987399, 1.659449, -0.294029, 0.359147, -0.017404, 1.319703, 0.959321, 1.866928, -0.480472]}]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=403d933a707802c9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_403d933a (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 5 — `execute_in_remote_docker` (`42833 ms` LLM)
- **Rationale**: Correct the observed archive layout mismatch, then exercise sparse and spiked calibration paths. Holding out each lab's response labels while retaining the specified reference panel tests whether a fit transfers across labs and exposes errors concentrated in inactive, weakly activated, or missing-feature slices.
- **Alternatives Weighed**: `Replace two or three clean current measurements with historical calibration` (rejected: The public batch has shifted gains and biases relative to the prior sweep. Two distinct clean points identify the current affine map even after an EMI spike.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c03_batch_01_solver.py')
s = p.read_text().replace("history = json.load(f)['calibration_load_cells']", "history = json.load(f)").replace("if clean.sum() < 4:\n                ref_lab, ref_x, ref_y = (np.asarray(history[key])\n                                         for key in ('lab_id', 'cmd_stress', 'true_stress'))\n                x, y = ref_x[ref_lab == lab], ref_y[ref_lab == lab]\n                clean = np.ones(len(x), dtype=bool)", "if clean.sum() < 2:\n                sweep = next(row for row in reversed(history) if lab in row['lab_id'])\n                ref_lab, ref_x, ref_y = (np.asarray(sweep[key])\n                                         for key in ('lab_id', 'cmd_stress', 'true_stress'))\n                x, y = ref_x[ref_lab == lab], ref_y[ref_lab == lab]\n                clean = np.ones(len(x), dtype=bool)")
p.write_text(s)
PY
cat > /tmp/validate_rheology_01.py <<'PY'
import importlib.util, pickle, time
import numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c03_batch_01_solver.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
item = pickle.load(open(root+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]
d = item['dataset']; tr, te = d['train'], d['test']
start=time.monotonic(); out=s.solve_alpha_m97_c03_batch_01(d)
print('fit secs',round(time.monotonic()-start,3), 'fit', {k:np.round(out[k],7).tolist() for k in ('estimated_taus','hereditary_weights','regime_mode_weights')})
gt=item['gt']; print('public evaluation fields', {k:np.shape(v) for k,v in gt.items()})
for k in out.keys() & gt.keys():
    err = out[k]-gt[k]
    print('public',k,'RMSE',np.sqrt(np.mean(err**2)),'max abs',np.max(np.abs(err)))
cal=d['calibration_load_cells']; g,b=s._calibrate(cal)
spike={k:v.copy() for k,v in cal.items()}; spike['true_stress'][2] += 7
sg,sb=s._calibrate(spike); assert max(np.max(abs(sg-g)),np.max(abs(sb-b))) < .001
sparse={k:v[(cal['lab_id'] != 1) | (np.arange(len(cal['lab_id'])) == 10)] for k,v in cal.items()}
ag,ab=s._calibrate(sparse); assert abs(ag[1]-1) < .001 and abs(ab[1]) < .001
assert np.allclose(ag[[0,2,3]],g[[0,2,3]]) and np.allclose(ab[[0,2,3]],b[[0,2,3]])
print('EMI and sparse archive paths pass')
x,_=s._align(tr,te)
args = g,b,d['past_dt'],d['eval_times'],d['crit_stress'],d['shear_banding_index']
hist = g[tr['lab_id'],None]*tr['past_stress_cmd']+b[tr['lab_id'],None]
damage=np.maximum(abs(hist)-d['crit_stress'],0)**2 @ d['past_dt']
cv=[]
for lab in range(4):
    m=tr['lab_id']==lab
    train={k:v[~m] for k,v in tr.items()}; test={k:v[m] for k,v in tr.items()}
    build,_=s._design(train,x[~m],*args)
    logtau,beta=s._fit(build,train['strain_curves_obs']-train['strain_0'][:,None],train['instant_rate_obs'],args[2],args[3])
    build,arrest=s._design(test,x[m],*args); c,r=build(logtau)
    ce=c@beta+test['strain_0'][:,None]-test['strain_curves_obs']; re=r@beta-test['instant_rate_obs']
    for label,mask in [('all',np.ones(m.sum(),bool)),('inactive',~arrest),('weak',arrest&(damage[m]<1)),('strong',arrest&(damage[m]>=1)),('missing',np.isnan(test['coupon_features_raw']).any(axis=1))]:
        if mask.any():
            cv.append((lab,label,int(mask.sum()),round(np.sqrt(np.mean(ce[mask]**2)),6),round(np.sqrt(np.mean(re[mask]**2)),6)))
    print('held lab',lab,'taus',np.round(np.exp(logtau),6),'regime weights',np.round(beta[7:],6))
print('LOLO (lab,slice,n,strain RMSE,rate RMSE)',cv)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/validate_rheology_01.py
```
- **Container Exit Code**: `0` (`1079 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
fit secs 0.053 fit {'estimated_taus': [0.5766025, 3.4749639], 'hereditary_weights': [0.6838682, 0.5122266, 0.0778207, 0.0858768, -0.0894467, 0.0615089, -0.0442108], 'regime_mode_weights': [0.2816089, 0.2345809]}
public evaluation fields {'taus': (2,), 'theta': (7,), 'vp_theta': (2,), 'stress_gains': (4,), 'stress_biases': (4,), 'te_clean': (72, 10), 'te_true_rate0': (72,), 'te_hist_true': (72, 18), 'te_hold_true': (72,)}
EMI and sparse archive paths pass
held lab 0 taus [0.578407 2.943324] regime weights [0.306882 0.217195]
held lab 1 taus [0.566408 3.077355] regime weights [0.260358 0.254936]
held lab 2 taus [0.572563 4.864315] regime weights [0.274758 0.235972]
held lab 3 taus [0.565673 3.023024] regime weights [0.27046  0.230185]
LOLO (lab,slice,n,strain RMSE,rate RMSE) [(0, 'all', 35, 2.410759, 0.152233), (0, 'inactive', 11, 1.906809, 0.112992), (0, 'strong', 24, 2.60941, 0.167167), (0, 'missing', 8, 2.041795, 0.106836), (1, 'all', 49, 0.384648, 0.132937), (1, 'inactive', 17, 0.558293, 0.11726), (1, 'strong', 32, 0.246919, 0.140556), (1, 'missing', 4, 0.176985, 0.117889), (2, 'all', 57, 3.614888, 0.176445), (2, 'inactive', 21, 1.902927, 0.213459), (2, 'strong', 36, 4.31019, 0.150713), (2, 'missing', 4, 0.382183, 0.332371), (3, 'all', 39, 1.305712, 0.151118), (3, 'inactive', 11, 1.209402, 0.16508), (3, 'strong', 28, 1.341657, 0.145266), (3, 'missing', 7, 2.507603, 0.187051)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=182f917e3d9c8ca6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_182f917e (directory).
```

### Turn 6 — `execute_in_remote_docker` (`26858 ms` LLM)
- **Rationale**: Compare held-out residual quantiles and time profiles to determine whether the worst slice calls for a stronger robust fit or a model correction. Evaluate all five outputs against their correctly mapped public reference fields; the previous name intersection matched none.
- **Alternatives Weighed**: `Change the creep law based on the worst raw RMSE` (rejected: Raw RMSE alone cannot distinguish a systematic model error from a few large corrupted strain measurements.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','alpha_m97_c03_batch_01_solver.py'); s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
i=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb'))[0];d=i['dataset'];tr,te=d['train'],d['test'];gt=i['gt'];out=s.solve_alpha_m97_c03_batch_01(d)
for p,q in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta'),('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
 er=out[p]-gt[q]; print(p,'RMSE',np.sqrt(np.mean(er**2)),'max',abs(er).max());
 if er.ndim==1 and er.size<10: print('reference',gt[q])
g,b=s._calibrate(d['calibration_load_cells']);x,_=s._align(tr,te);a=g,b,d['past_dt'],d['eval_times'],d['crit_stress'],d['shear_banding_index']; print('eval times',a[3],'past duration',sum(a[2]),'crit',a[4],'banding',a[5]); m=tr['lab_id']; obs=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; qs=[.5,.75,.9,.95,.98,1.]
for lab in range(4):
 take=m!=lab; train={k:v[take] for k,v in tr.items()}; test={k:v[~take] for k,v in tr.items()};f,_=s._design(train,x[take],*a); tau,be=s._fit(f,obs[take],rate[take],a[2],a[3]); f,act=s._design(test,x[~take],*a);c,r=f(tau); ce=c@be-obs[~take];re=r@be-rate[~take];print('held lab',lab,'strain absolute quantiles',np.round(np.quantile(abs(ce),qs),4),'rate',np.round(np.quantile(abs(re),qs),4),'largest curve residuals by time',np.round(abs(ce).max(axis=0),2),'curves with >= 1 point above 1',np.sum(np.max(abs(ce),axis=1)>1))
PY
```
- **Container Exit Code**: `0` (`787 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
estimated_taus RMSE 0.4512006870850122 max 0.6380361405281918
reference [0.568 4.113]
hereditary_weights RMSE 0.050659541577574574 max 0.07412315151428082
reference [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09]
regime_mode_weights RMSE 0.003997180953769142 max 0.005419066991070448
reference [0.28 0.24]
predicted_strain RMSE 0.17359748312807277 max 0.815193301494272
initial_rate_pred RMSE 0.12986833929844097 max 0.36757422519195515
eval times [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ] past duration 8.6395 crit 2.65 banding 0.12
held lab 0 strain absolute quantiles [ 0.1059  0.2673  0.4678  0.6379  1.2414 36.9776] rate [0.0873 0.1714 0.2477 0.3136 0.3591 0.3655] largest curve residuals by time [ 0.11 14.62  0.34  2.03 36.98  0.58  8.08  0.68  1.23 18.12] curves with >= 1 point above 1 9
held lab 1 strain absolute quantiles [0.0908 0.1936 0.2995 0.3831 0.5523 5.8335] rate [0.0932 0.1627 0.1915 0.2379 0.2806 0.3502] largest curve residuals by time [0.36 0.52 1.21 3.22 0.44 1.44 2.59 0.53 0.56 5.83] curves with >= 1 point above 1 6
held lab 2 strain absolute quantiles [ 0.0969  0.2324  0.4397  0.6859  1.7007 76.1325] rate [0.1216 0.1864 0.2965 0.3283 0.3521 0.5377] largest curve residuals by time [21.14  1.02 76.13 23.97  0.43  1.65  2.79  7.95  0.93  1.14] curves with >= 1 point above 1 16
held lab 3 strain absolute quantiles [ 0.0976  0.2168  0.4021  0.5592  1.2245 20.8567] rate [0.1083 0.1812 0.2419 0.2742 0.3011 0.3735] largest curve residuals by time [20.86 11.92  0.25  0.88  0.77  0.52  0.52  0.58  1.89  0.79] curves with >= 1 point above 1 9

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=182f917e3d9c8ca6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_182f917e (directory).
```

### Turn 7 — `execute_in_remote_docker` (`28867 ms` LLM)
- **Rationale**: Use public references only to diagnose whether the error follows the drift term across time and initial rate. A shared drift discrepancy points to alignment or imputation; a time-dependent discrepancy points to memory fitting. The diagnostics do not enter the solver.
- **Alternatives Weighed**: `Increase robustness immediately` (rejected: Isolated strain spikes explain the extreme raw RMSE, but do not yet explain why the fitted drift coefficients are all smaller in magnitude than the public reference.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p=importlib.util.spec_from_file_location('solver','alpha_m97_c03_batch_01_solver.py');s=importlib.util.module_from_spec(p);p.loader.exec_module(s)
i=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb'))[0];d=i['dataset'];gt=i['gt'];tr,te=d['train'],d['test'];g,b=s._calibrate(d['calibration_load_cells']);dt,t=d['past_dt'],d['eval_times'];xs=s._align(tr,te);w=np.r_[gt['theta'],gt['vp_theta']];lt=np.log(gt['taus'])
for split,x in zip((tr,te),xs):
 f,a=s._design(split,x,g,b,dt,t,d['crit_stress'],d['shear_banding_index']);c,r=f(lt); ref=split['ref_coupons']; labs=split['lab_id']; print('raw/ref shapes',split['coupon_features_raw'].shape,ref.shape)
 target=gt['te_clean'] if split is te else tr['strain_curves_obs']; rates=gt['te_true_rate0'] if split is te else tr['instant_rate_obs']; ce=target-split['strain_0'][:,None]-c@w; re=rates-r@w; T=c[:,:,2]; dc=np.median(ce/T,axis=1)
 for lab in range(4):
  m=labs==lab;rp=ref[m]; raw=split['coupon_features_raw'][m]; print('lab',lab,'ref means',np.round(np.nanmean(rp,axis=0),3),'ref stds',np.round(np.nanstd(rp,axis=0),3),'raw means/stds',np.round(np.nanmean(raw,axis=0),3),np.round(np.nanstd(raw,axis=0),3),'drift error median/std',np.round([np.median(dc[m]),np.std(dc[m]),np.median(re[m]),np.std(re[m])],4))
 print('drift across times |error minus median| q',np.quantile(abs(ce/T-dc[:,None]),[.5,.9,.99,1]),'curve-rate drift agreement',np.quantile(abs(dc-re),[.5,.9,1]))
 print('drift correction intercept/slopes',np.linalg.lstsq(np.c_[np.ones(len(x)),x],dc,rcond=None)[0]);print('hold/hist cal max', np.max(abs(g[labs,None]*split['past_stress_cmd']+b[labs,None]-gt['te_hist_true'])) if split is te else '-')
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS' | grep -i -m 6 'cauchy\|irls' | cut -c1-320
```
- **Container Exit Code**: `0` (`671 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
37 0.766 0.807 0.709 0.665
 0.877 0.714 0.98  0.844 0.816 1.012 0.976 0.699] raw means/stds [-0.499 -0.702 -0.988 -0.587] [0.745 0.789 0.815 0.684] drift error median/std [-0.0069  0.0838 -0.0006  0.022 ]
lab 1 ref means [ 0.45   0.427  0.183  0.684  0.643  0.812  0.29   1.135  0.225  0.841
  0.586  0.31   0.914  0.814 -0.172  0.481  0.215  0.645  0.431  0.81
  0.691  0.862  0.77   0.663  0.608  0.379  0.323  0.561  0.951  0.583
  0.505  0.643] ref stds [1.352 1.185 1.557 1.131 1.344 1.18  0.952 1.078 0.88  1.414 1.177 1.007
 1.492 1.046 1.124 1.59  1.067 1.462 1.242 1.246 0.942 1.096 1.231 1.285
 1.113 0.976 1.306 1.245 1.013 1.31  1.186 1.524] raw means/stds [0.275 0.574 0.729 0.318] [1.713 1.331 1.199 1.804] drift error median/std [-0.0003  0.1287 -0.0002  0.0337]
lab 2 ref means [ 0.023 -0.288 -0.271 -0.249 -0.405 -0.01  -0.14  -0.296 -0.043 -0.241
 -0.341 -0.006 -0.248  0.116  0.035 -0.19  -0.154 -0.477 -0.196  0.008
 -0.109 -0.4   -0.408 -0.407 -0.45  -0.431 -0.456 -0.785  0.056 -0.314
 -0.063 -0.544] ref stds [0.78  1.134 0.734 0.826 0.92  0.931 0.792 1.124 0.834 0.68  0.783 0.87
 1.003 0.847 0.735 0.882 0.634 0.812 0.718 0.855 0.927 1.122 0.938 0.877
 0.91  0.897 0.903 0.847 0.777 0.841 0.684 0.692] raw means/stds [ 0.288 -0.16  -0.221 -0.114] [0.834 0.853 0.992 0.876] drift error median/std [-0.0073  0.0848 -0.0044  0.0321]
lab 3 ref means [ 0.341  0.637  1.022  0.793  0.5    0.546  0.434 -0.015  0.18   0.738
  0.963  0.358  0.523  0.204  0.002  1.008  0.919  0.306  0.332  0.924
  0.762  1.177  1.38   0.552  0.36   0.707  1.19   0.308  1.088  1.187
  0.665  0.47 ] ref stds [1.209 1.179 1.516 1.355 1.546 0.977 1.311 1.521 1.072 1.271 1.084 0.887
 1.204 1.055 1.67  1.274 1.336 1.367 1.31  1.239 1.039 1.708 1.519 1.298
 1.398 1.289 0.984 1.384 1.679 1.39  1.004 1.176] raw means/stds [0.616 0.329 0.288 0.482] [1.488 0.931 1.324 1.506] drift error median/std [-0.0073  0.061  -0.0066  0.0188]
drift across times |error minus median| q [4.40726227e-06 3.54112545e-02 1.40090393e-01 3.23074230e-01] curve-rate drift agreement [2.69106362e-05 1.80631550e-01 4.26265393e-01]
drift correction intercept/slopes [-0.03342401 -0.04727405  0.00781217 -0.02155879  0.01120955]
hold/hist cal max 0.00020877658788087672

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=182f917e3d9c8ca6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_182f917e (directory).
```

### Turn 8 — `execute_in_remote_docker` (`29109 ms` LLM)
- **Rationale**: Separate drift-only error from time-shape error on complete and imputed coupons in both regimes. A discrepancy on complete coupons confined to one regime points to the regime equation; a shared strain-and-rate drift discrepancy confined to missing coupons points to imputation.
- **Alternatives Weighed**: `Strengthen the robust loss before diagnosing the residuals` (rejected: The clean public references still show time-dependent residuals at the reference coefficients. Reweighting noisy training observations cannot explain those residuals.)

```bash
cd /holdout_task && sed -n '8,42p' alpha_m97_c03_batch_01_solver.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p=importlib.util.spec_from_file_location('solver','alpha_m97_c03_batch_01_solver.py');s=importlib.util.module_from_spec(p);p.loader.exec_module(s)
i=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb'))[0];d=i['dataset'];gt=i['gt'];tr,te=d['train'],d['test'];g,b=s._calibrate(d['calibration_load_cells']);dt,t=d['past_dt'],d['eval_times'];xs=s._align(tr,te);w=np.r_[gt['theta'],gt['vp_theta']];lt=np.log(gt['taus']);print('test keys',list(te),'train keys',list(tr))
for split,x in zip((tr,te),xs):
 f,a=s._design(split,x,g,b,dt,t,d['crit_stress'],d['shear_banding_index']);c,r=f(lt); y=gt['te_clean'] if split is te else split['strain_curves_obs']; v=gt['te_true_rate0'] if split is te else split['instant_rate_obs']; res=y-split['strain_0'][:,None]-c@w; rr=v-r@w; u=res/c[:,:,2]; drift=np.median(u,axis=1); shape=np.max(abs(u-drift[:,None]),axis=1); miss=np.isnan(split['coupon_features_raw']).any(axis=1)
 for regime in (False,True):
  for missing in (False,True):
   m=(a==regime)&(miss==missing)
   if not m.any(): continue
   print('split/arrest/missing/n', 'test' if split is te else 'train',regime,missing,m.sum(),'median/q90 by [drift, shape, rate, agreement]',np.round(np.quantile(np.abs(np.c_[drift,shape,rr,drift-rr][m]),[.5,.9],axis=0),5))
   print('time residual medians',np.round(np.median(res[m],axis=0),5))
 if split is te:
  inds=np.flatnonzero(shape>.02); print('shape failures (index,lab,stratum,active,missing,drift,rate,shape)',[(int(j),int(split['lab_id'][j]),int(split['stratum'][j]),bool(a[j]),bool(miss[j]),*np.round([drift[j],rr[j],shape[j]],4)) for j in inds]);print('example residuals',np.round(res[inds[:5]],4))
PY
```
- **Container Exit Code**: `0` (`557 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.03953 0.05769 0.05216]
 [0.10984 0.03953 0.05769 0.05216]]
time residual medians [0.02109 0.07209 0.13226 0.1978  0.26687 0.33843 0.41187 0.48673 0.56275
 0.63969]
split/arrest/missing/n test True False 42 median/q90 by [drift, shape, rate, agreement] [[4.180e-03 1.000e-05 4.190e-03 1.000e-05]
 [1.954e-02 3.000e-05 1.954e-02 4.000e-05]]
time residual medians [-0.0001  -0.00025 -0.00036 -0.00045 -0.00053 -0.00059 -0.00065 -0.0007
 -0.00074 -0.00078]
split/arrest/missing/n test True True 6 median/q90 by [drift, shape, rate, agreement] [[7.4710e-02 1.0000e-05 7.4720e-02 1.0000e-05]
 [1.3701e-01 3.0000e-05 1.3701e-01 4.0000e-05]]
time residual medians [-0.01082 -0.02524 -0.0359  -0.04437 -0.05139 -0.05738 -0.06262 -0.06726
 -0.07144 -0.07523]
shape failures (index,lab,stratum,active,missing,drift,rate,shape) [(48, 0, 1, False, False, -0.0563, -0.0006, 0.0422), (49, 2, 2, False, False, -0.1564, -0.0044, 0.1153), (50, 2, 2, False, False, -0.2276, -0.0062, 0.1678), (51, 1, 0, False, False, 0.183, 0.0002, 0.1385), (52, 2, 0, False, False, -0.0466, -0.0015, 0.0342), (53, 1, 1, False, False, -0.186, -0.0006, 0.1405), (55, 0, 1, False, False, -0.1674, -0.0063, 0.1221), (56, 2, 2, False, False, -0.1438, -0.0041, 0.1059), (58, 0, 2, False, False, -0.2803, -0.0121, 0.2033), (59, 1, 1, False, False, -0.4275, -0.0012, 0.3231), (60, 2, 0, False, False, -0.0821, -0.0024, 0.0604), (61, 3, 1, False, False, -0.1107, -0.0077, 0.0781), (62, 0, 1, False, True, 0.1098, 0.0577, 0.0395), (63, 3, 0, False, False, -0.2234, -0.0232, 0.1518), (64, 2, 2, False, False, -0.2485, -0.0068, 0.1832), (65, 2, 2, False, False, -0.1652, -0.0046, 0.1217), (67, 2, 2, False, False, -0.1976, -0.0055, 0.1456), (68, 3, 0, False, False, -0.1018, -0.0065, 0.0722), (69, 0, 0, False, False, -0.1198, -0.0039, 0.0879)]
example residuals [[-0.0042 -0.0255 -0.0565 -0.0932 -0.1338 -0.177  -0.2222 -0.2689 -0.3168
  -0.3658]
 [-0.0124 -0.0718 -0.1581 -0.26   -0.3722 -0.4916 -0.6165 -0.7456 -0.878
  -1.0132]
 [-0.0179 -0.1045 -0.2299 -0.3782 -0.5414 -0.7152 -0.897  -1.0848 -1.2776
  -1.4742]
 [ 0.0133  0.0821  0.1831  0.3028  0.435   0.5758  0.7233  0.8757  1.0322
   1.192 ]
 [-0.0037 -0.0215 -0.0471 -0.0774 -0.1108 -0.1463 -0.1835 -0.2218 -0.2612
  -0.3014]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=182f917e3d9c8ca6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_182f917e (directory).
```

### Turn 9 — `execute_in_remote_docker` (`48120 ms` LLM)
- **Rationale**: Compare the two gate structures with each lab's labels held out, then test whether missing-feature rows bias coefficient estimation. If decoupling improves the inactive slice without degrading the others, promote it; otherwise retain the current gate and investigate the remaining shape error.
- **Alternatives Weighed**: `Fit an arbitrary time-shape correction` (rejected: The observed endpoint ratios are consistent with the specified t minus log1p(t) basis, so test that exact mechanism first.)
- **ATMS NOGOOD DECLARED**: Refuted `At low banding, a zero work-damage activation implies an exactly linear terminal basis.` | Witness: `On 23 complete inactive public coupons at reference parameters, the median normalized time-shape error is 0.10594 and the median final strain residual is -0.77538, while the median absolute initial-rate residual is only 0.00407.` | Blanket Diagnosis: `The current model shares a single gate between terms that change finite-time creep shape and terms that vanish with work-damage; a drift or alignment correction alone cannot reproduce the residual's time dependence and small initial-rate error.` | Sublated Pivot: `Test the exact pair of bases t and log1p(t) with the terminal-flow gate separated from work-damage activation.`

```bash
set -e
cd /holdout_task
cat >/tmp/rheo_gate.py <<'PY'
import importlib.util,pickle,numpy as np
from pathlib import Path
p=Path('/holdout_task/alpha_m97_c03_batch_01_solver.py'); s=importlib.util.module_from_spec(importlib.util.spec_from_file_location('solver',p)); exec(compile(p.read_text(),str(p),'exec'),s.__dict__)
i=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_01.pkl','rb'))[0];d=i['dataset'];gt=i['gt'];tr,te=d['train'],d['test'];g,b=s._calibrate(d['calibration_load_cells']);dt,t=d['past_dt'],d['eval_times'];x,xt=s._align(tr,te); args=(g,b,dt,t,d['crit_stress'],d['shear_banding_index']);lt=np.log(gt['taus']);w=np.r_[gt['theta'],gt['vp_theta']]; labs=tr['lab_id'];y=tr['strain_curves_obs']-tr['strain_0'][:,None];v=tr['instant_rate_obs'];mtr=np.isnan(tr['coupon_features_raw']).any(axis=1);mte=np.isnan(te['coupon_features_raw']).any(axis=1)
raw=p.read_text(); alt={};
for mode in ('linear','log'):
 if mode=='linear': alt[mode]=s._design
 else:
  env={};exec(compile(raw.replace('np.where(arrest[:, None], np.log1p(t), t)','np.log1p(t) if banding <= .50 else t').replace('terminal = np.log1p(t) if banding <= .50 else t','terminal = np.broadcast_to(np.log1p(t) if banding <= .50 else t, (len(x), len(t)))'),str(p),'exec'),env);alt[mode]=env['_design']
for mode, design in alt.items():
 for complete in (False,True):
  yy=np.where((complete&mtr)[:,None],np.nan,y);vv=np.where(complete&mtr,np.nan,v);cv=[]; build,ar=design(tr,x,*args)
  for lab in range(4):
   train=labs!=lab; lt1,w1=s._fit(build,np.where(train[:,None],yy,np.nan),np.where(train,vv,np.nan),dt,t);c,r=build(lt1);e=np.abs(c@w1-y); f=np.abs(r@w1-v)
   for a in (False,True):
    for miss in (False,True):
     m=~train&(ar==a)&(mtr==miss)
     if m.any():cv.append((lab,a,miss,m.sum(),np.median(e[m]),np.quantile(e[m],.95),np.sqrt(np.mean(f[m]**2))))
  lt1,w1=s._fit(build,yy,vv,dt,t);build,ar=design(te,xt,*args);c,r=build(lt1);e=c@w1+te['strain_0'][:,None]-gt['te_clean'];f=r@w1-gt['te_true_rate0'];rmse=lambda v:np.sqrt(np.mean(v*v));print(mode,'fit complete only',complete,'worst complete inactive CV median/p95/rate',np.round(np.max([z[-3:] for z in cv if not z[1] and not z[2]],axis=0),5),'worst CV p95/rate',np.round(np.max([z[-2:] for z in cv],axis=0),5),'public RMSE tau/hered/regime/strain/rate',np.round([rmse(np.exp(lt1)-gt['taus']),rmse(w1[:7]-w[:7]),rmse(w1[7:]-w[7:]),rmse(e),rmse(f)],6),'fit taus',np.exp(lt1));print('public RMSE by active / missing:',[(a,m,round(rmse(e[(ar==a)&(mte==m)]),5),round(rmse(f[(ar==a)&(mte==m)]),5)) for a in (False,True) for m in (False,True)])
 if mode=='log':
  c,r=build(lt); er=gt['te_clean']-te['strain_0'][:,None]-c@w; rr=gt['te_true_rate0']-r@w; dr=np.median(er/c[:,:,2],axis=1);print('reference log complete inactive shape max:',np.max(abs(er/c[:,:,2]-dr[:,None])[~ar&~mte]),'missing rows inferred drift / cell-aligned drift',np.round(np.c_[rr[mte]+xt[mte]@w[3:7],xt[mte]@w[3:7]],4)); print('missing patterns',np.isnan(te['coupon_features_raw'][mte]).astype(int).tolist())
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheo_gate.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust IRLS' | grep -i -m 4 'irls\|cauchy' | cut -c1-240 || true
```
- **Container Exit Code**: `0` (`1757 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
linear fit complete only False worst complete inactive CV median/p95/rate [0.12483 0.88096 0.18371] worst CV p95/rate [0.88096 0.39724] public RMSE tau/hered/regime/strain/rate [0.451201 0.05066  0.003997 0.173597 0.129868] fit taus [0.57660252 3.47496386]
public RMSE by active / missing: [(False, False, 0.18336, 0.14309), (False, True, 0.18349, 0.00676), (True, False, 0.17353, 0.12715), (True, True, 0.12788, 0.10363)]
linear fit complete only True worst complete inactive CV median/p95/rate [0.18164 0.92619 0.25708] worst CV p95/rate [0.92619 0.4102 ] public RMSE tau/hered/regime/strain/rate [0.037366 0.050811 0.008678 0.17416  0.129079] fit taus [0.58388391 4.06260077]
public RMSE by active / missing: [(False, False, 0.18104, 0.14069), (False, True, 0.20115, 0.02862), (True, False, 0.17588, 0.12829), (True, True, 0.12275, 0.09338)]
log fit complete only False worst complete inactive CV median/p95/rate [0.01986 0.68331 0.03618] worst CV p95/rate [0.68331 0.26404] public RMSE tau/hered/regime/strain/rate [0.450354 0.012563 0.003946 0.037254 0.033829] fit taus [0.56690818 3.47610481]
public RMSE by active / missing: [(False, False, 0.01495, 0.01932), (False, True, 0.06434, 0.02214), (True, False, 0.01554, 0.01862), (True, True, 0.11584, 0.09896)]
log fit complete only True worst complete inactive CV median/p95/rate [0.02091 0.67624 0.02668] worst CV p95/rate [0.67624 0.24573] public RMSE tau/hered/regime/strain/rate [0.141769 0.006785 0.002417 0.035959 0.028324] fit taus [0.57102473 3.91253081]
public RMSE by active / missing: [(False, False, 0.01112, 0.00866), (False, True, 0.07499, 0.05426), (True, False, 0.01403, 0.01045), (True, True, 0.11282, 0.08991)]
reference log complete inactive shape max: 1.7588331479958955e-05 missing rows inferred drift / cell-aligned drift [[-0.169  -0.0859]
 [-0.113  -0.2522]
 [ 0.0838  0.1502]
 [ 0.1786  0.172 ]
 [ 0.1827  0.1988]
 [-0.2534 -0.1187]
 [-0.2406 -0.2983]]
missing patterns [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=182f917e3d9c8ca6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_182f917e (directory).
```

### Turn 10 — `execute_in_remote_docker` (`42884 ms` LLM)
- **Rationale**: Promote the gate and fit demonstrated by held-lab validation. Verify the exact saved solver in an isolated process, using an independent stepwise recurrence to catch branch-order, rate, or threshold mistakes that would warrant correcting the implementation.
- **Alternatives Weighed**: `Keep all coupons in the material fit` (rejected: With the corrected gate, fitting complete coupons reduced the public RMSE on all five outputs and reduced the worst held-lab rate RMSE from 0.264 to 0.246.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c03_batch_01_solver.py')
s = p.read_text()
s = s.replace('    terminal = np.where(arrest[:, None], np.log1p(t), t)', '    # Banding determines the flow basis; damage activates the network terms.\n    terminal = np.log1p(t) if banding <= .50 else t')
s = s.replace('terminal[:, :, None]*flow[:, None, :]', 'terminal[None, :, None]*flow[:, None, :]')
s = s.replace('(act*z[:, 0])[:, None]*terminal', '(act*z[:, 0])[:, None]*terminal[None, :]')
s = s.replace("    log_tau, beta = _fit(build, train['strain_curves_obs'] - train['strain_0'][:, None],\n                        train['instant_rate_obs'], dt, t)", "    # Imputed coupons remain predictable, but do not inform the material fit\n    # when complete coupons identify all nine linear coefficients.\n    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)\n    if np.linalg.matrix_rank(np.concatenate([build(np.log([.65, 4.]))[1][complete],\n                           build(np.log([.65, 4.]))[0][complete].reshape(-1, 9)], axis=0)) < 9:\n        complete[:] = True\n    log_tau, beta = _fit(\n        build, np.where(complete[:, None], train['strain_curves_obs'] - train['strain_0'][:, None], np.nan),\n        np.where(complete, train['instant_rate_obs'], np.nan), dt, t)")
s = s.replace('    if np.linalg.matrix_rank(np.concatenate([build(np.log([.65, 4.]))[1][complete],\n                           build(np.log([.65, 4.]))[0][complete].reshape(-1, 9)], axis=0)) < 9:\n        complete[:] = True', '    c0, r0 = build(np.log([.65, 4.]))\n    # With no active network, its two coefficients are unidentifiable by design.\n    columns = slice(None) if np.any(c0[:, :, 7:]) else slice(0, 7)\n    design = np.vstack((c0[complete].reshape(-1, 9), r0[complete]))[:, columns]\n    if not len(design) or np.linalg.matrix_rank(design) < design.shape[1]:\n        complete[:] = True')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/validate_rheology_01.py | head -3
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p=importlib.util.spec_from_file_location('solver','alpha_m97_c03_batch_01_solver.py');s=importlib.util.module_from_spec(p);p.loader.exec_module(s)
i=pickle.load(open('public_alpha_m97_c03_batch_01.pkl','rb'))[0];d=i['dataset'];gt=i['gt'];tr,te=d['train'],d['test'];o=s.solve_alpha_m97_c03_batch_01(d);rmse=lambda e:np.sqrt(np.mean(e*e));print('verified output RMSE',[(k,round(rmse(o[k]-gt[v]),6)) for k,v in zip(o,('taus','theta','vp_theta','te_clean','te_true_rate0'))]);g,b=s._calibrate(d['calibration_load_cells']);_,x=s._align(tr,te);dt=d['past_dt'];t=d['eval_times'];crit=d['crit_stress'];lab=te['lab_id'];h=g[lab,None]*te['past_stress_cmd']+b[lab,None];h0=g[lab]*te['stress_hold_cmd']+b[lab];tau=o['estimated_taus'];w=o['hereditary_weights'];rw=o['regime_mode_weights'];z=np.zeros((len(lab),2));
for j,step in enumerate(dt):
    dec=np.exp(-step/tau);z=dec*z+(1-dec)*h[:,j,None]
damage=np.maximum(abs(h)-crit,0)**2@dt; gaps=h0[:,None]-z; drift=w[2]+x@w[3:]; rise=-np.expm1(-t[:,None]/tau)
for band in (.50,.51):
    f,ar=s._design(te,x,g,b,dt,t,crit,band);c,r=f(np.log(tau));act=np.tanh(damage)*(band<=.50);time=np.log1p(t) if band<=.50 else t;beta=np.r_[w,rw];base=(gaps*w[:2])@rise.T;expected=base+(rw[0]*act*h0)[:,None]*rise[:,0]+(drift+rw[1]*act*z[:,0])[:,None]*time;rate=gaps@(w[:2]/tau)+rw[0]*act*h0/tau[0]+drift+rw[1]*act*z[:,0];np.testing.assert_allclose(c@beta,expected,atol=1e-12);np.testing.assert_allclose(r@beta,rate,atol=1e-12);assert np.array_equal(ar,(damage>0)&(band<=.50));print('recurrence and rate pass at banding',band,'active/inactive',ar.sum(),(~ar).sum())
PY
```
- **Container Exit Code**: `0` (`1547 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
fit secs 0.067 fit {'estimated_taus': [0.5710247, 3.9125308], 'hereditary_weights': [0.682767, 0.533853, 0.1381136, 0.1557423, -0.1300779, 0.1144858, -0.0886698], 'regime_mode_weights': [0.279199, 0.2433231]}
public evaluation fields {'taus': (2,), 'theta': (7,), 'vp_theta': (2,), 'stress_gains': (4,), 'stress_biases': (4,), 'te_clean': (72, 10), 'te_true_rate0': (72,), 'te_hist_true': (72, 18), 'te_hold_true': (72,)}
EMI and sparse archive paths pass
verified output RMSE [('estimated_taus', 0.141769), ('hereditary_weights', 0.006785), ('regime_mode_weights', 0.002417), ('predicted_strain', 0.035959), ('initial_rate_pred', 0.028324)]
recurrence and rate pass at banding 0.5 active/inactive 48 24
recurrence and rate pass at banding 0.51 active/inactive 0 72

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3174766638c648c0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_31747666 (directory).
```

### Turn 11 — `submit_completed_task` (`13175 ms` LLM)
- **Rationale**: The saved solver passed isolated verification. Separating the two gates improved held-out performance, and an independent recurrence confirmed the strain and initial-rate formulas on both sides of the banding threshold.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_01_solver.py': 'FILE:7294'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_01_solver.py`. The public schema, sparse and EMI calibration checks, and recurrence and initial-rate checks passed. Public RMSE was 0.03596 for strain and 0.02832 for initial rate.`

### Turn 12 — `execute_in_remote_docker` (`32412 ms` LLM)
- **Rationale**: Test whether sparse strain spikes still bias the estimated spectrum and whether signed prediction errors persist within individual labs. Use held-lab labels to compare robust losses and report complete and imputed coupons separately; change the saved solver only if the challenger improves the relevant margins.
- **Alternatives Weighed**: `Test relay modes, classification accuracy, or Fokker–Planck operators` (rejected: This dataset has continuous rheology responses, no relay state, and no stochastic drift equation. Held-lab signed residuals test the relevant nuisance effect.)

```bash
set -e
cd /holdout_task
cat >/tmp/rheo_margin.py <<'PY'
import pickle
from pathlib import Path
import numpy as np
p=Path('/holdout_task/alpha_m97_c03_batch_01_solver.py')
source=p.read_text()
models={}
for name, code in [('current',source),('cauchy',source.replace("loss='soft_l1', f_scale=2.","loss='cauchy', f_scale=2.5")),('cauchy4',source.replace("loss='soft_l1', f_scale=2.","loss='cauchy', f_scale=4."))]:
    env={};exec(compile(code,str(p),'exec'),env);models[name]=env
items=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_01.pkl','rb')); print('public cases',len(items))
for item in items:
    d=item['dataset']; tr,te=d['train'],d['test']; lab=tr['lab_id']; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1)
    s=models['current']; g,b=s['_calibrate'](d['calibration_load_cells']); x,_=s['_align'](tr,te); dt,t=d['past_dt'],d['eval_times']; args=g,b,dt,t,d['crit_stress'],d['shear_banding_index']; f,active=s['_design'](tr,x,*args)
    y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; rmse=lambda e:np.sqrt(np.nanmean(e**2)); gt=item['gt']; cols=('taus','theta','vp_theta','te_clean','te_true_rate0')
    for name,s in models.items():
        slices=[]; bias=[]
        for c in np.unique(lab):
            keep=(lab!=c)&complete
            lt,w=s['_fit'](f,np.where(keep[:,None],y,np.nan),np.where(keep,rate,np.nan),dt,t)
            C,R=f(lt); e=C@w-y; re=R@w-rate
            for a in (False,True):
                for full in (True,False):
                    m=(lab==c)&(active==a)&(complete==full)
                    if m.any(): slices.append((c,a,full,np.nanmedian(abs(e[m])),np.nanquantile(abs(e[m]),.90),np.nanquantile(abs(e[m]),.95),rmse(re[m])))
            m=(lab==c)&complete; bias.append((int(c),round(float(np.nanmedian(re[m])),5),round(float(np.nanmedian(e[m])),5)))
        out=s['solve_alpha_m97_c03_batch_01'](d)
        print(name,'public RMSE',np.round([rmse(z-gt[k]) for z,k in zip(out.values(),cols)],6),'held-lab complete and missing maxima [median,p90,p95,rateRMSE]',*[np.round(np.max([z[3:] for z in slices if z[2]==v],axis=0),5) for v in (True,False)],'complete held-lab signed median [lab,rate,strain]',bias)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheo_margin.py
```
- **Container Exit Code**: `0` (`1467 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public cases 1
current public RMSE [0.141769 0.006785 0.002417 0.035959 0.028324] held-lab complete and missing maxima [median,p90,p95,rateRMSE] [0.02091 0.07077 0.67624 0.02822] [0.29704 0.53598 0.63589 0.24573] complete held-lab signed median [lab,rate,strain] [(0, 0.0029, -0.00105), (1, -0.00509, -0.005), (2, -0.00978, -0.00188), (3, -0.00622, 0.00287)]
cauchy public RMSE [0.097658 0.004959 0.001955 0.035765 0.027939] held-lab complete and missing maxima [median,p90,p95,rateRMSE] [0.02251 0.06819 0.68323 0.02815] [0.29104 0.5337  0.63177 0.24275] complete held-lab signed median [lab,rate,strain] [(0, 0.00053, -0.00127), (1, -0.00369, -0.00488), (2, -0.0105, 0.0002), (3, -0.00698, 0.0024)]
cauchy4 public RMSE [0.118201 0.006068 0.002342 0.035927 0.028199] held-lab complete and missing maxima [median,p90,p95,rateRMSE] [0.02088 0.07271 0.68145 0.02891] [0.29351 0.53401 0.63269 0.24323] complete held-lab signed median [lab,rate,strain] [(0, 0.00274, -0.00123), (1, -0.00463, -0.00546), (2, -0.0106, -0.00025), (3, -0.00667, 0.00132)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3174766638c648c0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_31747666 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`20461 ms` LLM)
- **Rationale**: Promote Cauchy for the 31% reduction in relaxation-time RMSE. Run the saved file in an isolated process to confirm the schema, calibration fallback, and held-lab margins; revert if the saved candidate fails those checks or does not reproduce the measured gains.
- **Alternatives Weighed**: `Retain soft-L1 refinement` (rejected: It gives a slightly better complete-coupon strain p95, but Cauchy improves all five public output RMSEs and lowers the worst held-lab rate RMSE.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c03_batch_01_solver.py')
s = p.read_text()
old = "loss='soft_l1', f_scale=2."
assert s.count(old) == 1
p.write_text(s.replace(old, "loss='cauchy', f_scale=2.5"))
PY
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
python3 -I /tmp/validate_rheology_01.py
sed '/^for name, code in /c\for name, code in [("saved_cauchy", source)]:' /tmp/rheo_margin.py > /tmp/rheo_final_margin.py
python3 -I /tmp/rheo_final_margin.py
```
- **Container Exit Code**: `1` (`1423 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
fit secs 0.058 fit {'estimated_taus': [0.5695402, 3.9749], 'hereditary_weights': [0.6788319, 0.5388105, 0.1379212, 0.1560455, -0.1288885, 0.1152309, -0.0887137], 'regime_mode_weights': [0.2827071, 0.2405644]}
public evaluation fields {'taus': (2,), 'theta': (7,), 'vp_theta': (2,), 'stress_gains': (4,), 'stress_biases': (4,), 'te_clean': (72, 10), 'te_true_rate0': (72,), 'te_hist_true': (72, 18), 'te_hold_true': (72,)}
EMI and sparse archive paths pass
held lab 0 taus [0.567325 3.496491] regime weights [0.293443 0.234601]
held lab 1 taus [0.561813 3.400985] regime weights [0.28859  0.240963]
held lab 2 taus [0.565467 3.95031 ] regime weights [0.278928 0.243925]
held lab 3 taus [0.565408 3.433185] regime weights [0.289645 0.235937]
LOLO (lab,slice,n,strain RMSE,rate RMSE) [(0, 'all', 35, 2.404593, 0.064399), (0, 'inactive', 11, 1.901881, 0.087068), (0, 'strong', 24, 2.602753, 0.05073), (0, 'missing', 8, 2.030006, 0.124577), (1, 'all', 49, 0.354375, 0.032487), (1, 'inactive', 17, 0.559261, 0.040191), (1, 'strong', 32, 0.161668, 0.027532), (1, 'missing', 4, 0.143441, 0.08869), (2, 'all', 57, 3.603282, 0.067204), (2, 'inactive', 21, 1.864551, 0.08245), (2, 'strong', 36, 4.304584, 0.056439), (2, 'missing', 4, 0.319508, 0.236709), (3, 'all', 39, 1.29307, 0.041502), (3, 'inactive', 11, 1.196556, 0.02913), (3, 'strong', 28, 1.32907, 0.045451), (3, 'missing', 7, 2.49603, 0.074911)]
public cases 1

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=584232b883214157 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_584232b8 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/rheo_final_margin.py", line 12, in <module>
    s=models['current']; g,b=s['_calibrate'](d['calibration_load_cells']); x,_=s['_align'](tr,te); dt,t=d['past_dt'],d['eval_times']; args=g,b,dt,t,d['crit_stress'],d['shear_banding_index']; f,active=s['_design'](tr,x,*args)
      ~~~~~~^^^^^^^^^^^
KeyError: 'current'
```

### Turn 14 — `execute_in_remote_docker` (`17299 ms` LLM)
- **Rationale**: Fix only the audit script and verify the unchanged artifact in isolated Python processes. If the saved solver does not reproduce the measured public and held-lab margins, investigate the discrepancy before submitting.
- **Alternatives Weighed**: `Revert the Cauchy refinement` (rejected: The schema and calibration checks passed. The remaining failure came from a mismatched model key in the temporary audit script, not from the solver.)

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
sed '/^for name, code in /c\for name, code in [("current", source)]:' /tmp/rheo_margin.py > /tmp/rheo_final_margin.py
python3 -I validate_public_schema.py
python3 -I /tmp/rheo_final_margin.py
```
- **Container Exit Code**: `0` (`1120 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
public cases 1
current public RMSE [0.097658 0.004959 0.001955 0.035765 0.027939] held-lab complete and missing maxima [median,p90,p95,rateRMSE] [0.02251 0.06819 0.68323 0.02815] [0.29104 0.5337  0.63177 0.24275] complete held-lab signed median [lab,rate,strain] [(0, 0.00053, -0.00127), (1, -0.00369, -0.00488), (2, -0.0105, 0.0002), (3, -0.00698, 0.0024)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=584232b883214157 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_584232b8 (directory).
```

### Turn 15 — `submit_completed_task` (`15174 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema and held-lab verification, reproduced the Cauchy improvement, and retained the calibration and creep equations already checked against an independent recurrence.
- **Alternatives Weighed**: `Retain soft-L1 refinement` (rejected: It slightly lowered complete-coupon strain p95, but Cauchy reduced public relaxation-time RMSE by 31% and lowered the worst held-lab rate RMSE.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_01_solver.py`. Public RMSE was 0.03577 for strain, 0.02794 for initial rate, and 0.09766 for relaxation times. Missing coupon features remained the largest source of held-out rate error.`

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
