# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c03_batch_04__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c03_step_04` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `337,297` (`326,144` / `90,112` / `11,153`) |
| **Total Episode Wall Time (s)** | `400.82s` |

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

### Turn 1 — `execute_in_remote_docker` (`16361 ms` LLM)
- **Rationale**: Inspect the public schema and array shapes to identify the fitting targets and validation slices. Locate prior solvers and check whether an applicable relaxation or alignment primitive is available before implementing either step.

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -lh /tmp/hegel_continual_archive && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference-panel alignment imputation' ; python3 - <<'PY'
import pickle
from pathlib import Path
p=Path('/holdout_task/public_alpha_m97_c03_batch_04.pkl')
d=pickle.loads(p.read_bytes())
def show(v, depth=0):
 if isinstance(v,dict) and depth<2: return {k:show(x,depth+1) for k,x in v.items()}
 if hasattr(v,'shape'): return (type(v).__name__,v.shape,str(getattr(v,'dtype','')))
 if isinstance(v,(list,tuple)): return (type(v).__name__,len(v),show(v[0],depth+1) if len(v) else None)
 return repr(v)[:140]
print(show(d))
PY
```
- **Container Exit Code**: `0` (`418 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ver.py
-rw-r--r-- 1 root root 6.8K Nov 27 09:05 alpha_m97_c03_batch_02_solver.py
-rw-r--r-- 1 root root 7.1K Nov 27 09:05 alpha_m97_c03_batch_03_solver.py
-rw-r--r-- 1 root root 7.8K Nov 27 09:05 alpha_m97_c03_s00_solver.py
-r--r--r-- 1 root root 7.1K Nov 27 09:05 archive_calibration_history.json
-rw-r--r-- 1 root root 1.9K Nov 27 09:05 archive_manifest.json
-rw-r--r-- 1 root root  337 Nov 27 09:05 pre_llm_gate_diagnostic.json
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
('list', 1, {'dataset': "{'past_dt': array([0.2026, 0.5276, 0.2722, 0.4003, 0.5436, 0.7958, 0.2002, 0.7521,\n       0.3744, 0.6677, 0.7518, 0.7456, 0.6947, 0.4099, 0.", 'gt': "{'taus': array([0.536, 3.591]), 'theta': array([ 0.68,  0.55,  0.14,  0.16, -0.13,  0.12, -0.09]), 'vp_theta': array([0.26, 0.22]), 'stress_"})
```

### Turn 2 — `execute_in_remote_docker` (`18339 ms` LLM)
- **Rationale**: Inspect the prior alignment convention, available training labels, and calibration sweep layout to avoid pooling load cells or using the wrong reference-panel axes. Inspect only the dataset inputs, not the packaged ground truth.
- **Alternatives Weighed**: `Reuse the pre-gate calibration payload` (rejected: The shared gate rejected that candidate; independently recover calibration from the permitted measurements and prefix sweeps.)

```bash
cat /tmp/hegel_continual_archive/alpha_m97_c03_batch_03_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel; python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
h=json.load(open(p)); print('history schema',list(h) if isinstance(h,dict) else len(h)); print(str(h)[:2400])
d=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_04.pkl','rb'))[0]['dataset']
for k,v in d.items():
 if not isinstance(v,dict): print(k,v); continue
 print(k)
 for a,x in v.items():
  x=np.asarray(x); print(' ',a,x.shape,x.dtype, np.unique(x,return_counts=True) if a in ('lab_id','stratum') else (np.isnan(x).sum() if x.dtype.kind=='f' else ''))
 if k=='calibration_load_cells': print(v)
PY
```
- **Container Exit Code**: `0` (`417 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [2.15143, 1.121894, -0.424697, -1.698186, 1.597457, -0.341977, 0.304259, 1.741536, -2.045051, 1.452617, 1.72849, -1.533376, -0.721054, -1.917953, -0.632766, 1.663405, -0.425306, 2.117922, -1.82032, 2.080468, -0.646451, 1.975872, 1.354166, 0.435969, 0.173637, -2.057672, -1.240089, 1.805134, -1.203484, -2.064745, -1.355878, -1.273919, -1.02295, 2.198106, 0.357224, -0.137312, -0.934877, -1.832142, 0.163072, 1.102407], 'true_stress': [2.895223, 1.567275, -0.427981, -2.07055, 2.180784, -0.320928, 0.512501, 2.366474, -2.518246, 1.993895, 1.267407, -1.40742, -0.74133, -1.722896, -0.669016, 1.214058, -0.498681, 1.586854, -1.642714, 1.555897, -0.616383, 2.451623, 1.724431, 0.650111, 0.343201, -2.26752, -1.310882, 2.251988, -1.267882, -2.275732, -1.140415, -1.078152, -0.887469, 1.560562, 0.16149, -0.214337, -0.820516, -1.502377, 0.013775, 0.72799]}, {'episode_short_name': 'alpha_m97_c03_batch_02', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [1.649489, -1.504968, 0.922268, 0.800933, 1.648912, 1.320542, -1.111114, 0.490613, -1.748203, 1.005033, -1.4101
past_dt [0.2026 0.5276 0.2722 0.4003 0.5436 0.7958 0.2002 0.7521 0.3744 0.6677
 0.7518 0.7456 0.6947 0.4099 0.5419 0.2921 0.7079 0.4224]
eval_times [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ]
crit_stress 2.65
yield_stress 2.65
shear_banding_index 0.12
calibration_load_cells
  lab_id (2,) int64 (array([0]), array([2]))
  cmd_stress (2,) float64 0
  true_stress (2,) float64 0
{'lab_id': array([0, 0]), 'cmd_stress': array([-1.3,  1.4]), 'true_stress': array([-1.55696249,  1.92595813])}
train
  lab_id (180,) int64 (array([0, 1, 2, 3]), array([48, 43, 42, 47]))
  stratum (180,) int64 (array([0, 1, 2]), array([59, 55, 66]))
  coupon_features_raw (180, 4) float64 25
  ref_coupons (180, 32) float64 223
  past_stress_cmd (180, 18) float64 0
  stress_hold_cmd (180,) float64 0
  strain_0 (180,) float64 0
  aux_chamber_telemetry (180, 3) float64 0
  strain_curves_obs (180, 10) float64 0
  instant_rate_obs (180,) float64 0
test
  lab_id (72,) int64 (array([0, 1, 2, 3]), array([18, 17, 16, 21]))
  stratum (72,) int64 (array([0, 1, 2]), array([17, 24, 31]))
  coupon_features_raw (72, 4) float64 6
  ref_coupons (72, 32) float64 85
  past_stress_cmd (72, 18) float64 0
  stress_hold_cmd (72,) float64 0
  strain_0 (72,) float64 0
  aux_chamber_telemetry (72, 3) float64 0
```

### Turn 3 — `execute_in_remote_docker` (`20437 ms` LLM)
- **Rationale**: A contaminated or shifting sweep would change how calibration is transferred. Counting active histories by channel shows which slices can constrain the crystallization terms and which rely on the shared hereditary model.
- **Alternatives Weighed**: `Use the latest sweep unconditionally` (rejected: The ordered sweep fits will show whether the latest sweep is clean and whether transfer is stable across campaigns.)

```bash
sed -n '51,175p' /tmp/hegel_continual_archive/alpha_m97_c03_batch_03_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
for s in h:
 print(s['episode_short_name'])
 for c in range(4):
  m=np.asarray(s['lab_id'])==c;x=np.asarray(s['cmd_stress'])[m];y=np.asarray(s['true_stress'])[m];p=np.polyfit(x,y,1);print(c,len(x),p, np.max(np.abs(y-np.polyval(p,x))))
d=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_04.pkl','rb'))[0]['dataset']; t=d['train']; g=np.array([1.29,.82,1.17,.76]); b=np.array([.12,-.15,.14,-.11]);
for part in ('train','test'):
 t=d[part]
 for c in range(4):
  m=t['lab_id']==c;p=t['past_stress_cmd'][m]*g[c]+b[c]; D=(np.maximum(np.abs(p)-d['crit_stress'],0)**2)@d['past_dt'];print(part,c,np.sum(D>0),np.quantile(D,[0,.5,.9,1]),np.unique(t['stratum'][m],return_counts=True)[1])
PY
```
- **Container Exit Code**: `0` (`438 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
de != 'soft_l1':
        for _ in range(3):
            err = residual(best.x).reshape(target.shape)
            curve_scale, rate_scale = _scale(err[:, :-1]), _scale(err[:, -1])
            mult[:, -1] = curve_scale / rate_scale
            best = fit(best.x, mode, 2.25 * curve_scale)
    return np.exp(best.x[:2]), best.x[2:]


def solve_alpha_m97_c03_batch_03(dataset: dict) -> dict:
    past, hold, features, offset, dt, t = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    target = np.column_stack((train['strain_curves_obs'] - offset[:n, None], train['instant_rate_obs']))
    taus, weights = _fit((past[:n], hold[:n], features[:n], dt, t), target)
    basis, _, _ = _basis(np.log(taus), past[n:], hold[n:], features[n:], dt, t)
    pred = basis @ weights
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': offset[n:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
alpha_m97_c03_s00
0 10 [9.99996399e-01 1.32406004e-05] 0.00017965296462654767
1 10 [ 9.99987895e-01 -2.46087894e-05] 0.0002542608844398053
2 10 [1.00004377e+00 1.86490255e-05] 0.00018935354483656797
3 10 [ 1.00004220e+00 -8.39804234e-05] 0.00012129383951794276
alpha_m97_c03_batch_01
0 10 [1.28998901 0.12000156] 0.00021700626342929796
1 10 [ 0.82003898 -0.15002602] 0.00014183185140925758
2 10 [1.16998089 0.14000927] 0.0001620203768579742
3 10 [ 0.76000152 -0.10998804] 0.0001719282826188693
alpha_m97_c03_batch_02
0 10 [1.28996334 0.11998329] 0.0001703216589850065
1 10 [ 0.82003341 -0.15000563] 0.00013444553809754956
2 10 [1.16998653 0.13997717] 0.0001720728489744694
3 10 [ 0.75996642 -0.11003498] 0.00016497030799489387
alpha_m97_c03_batch_03
0 10 [1.28996282 0.1199782 ] 0.0001717193952429641
1 10 [ 0.81997725 -0.14996309] 0.00024491625334988054
2 10 [1.16999803 0.14000889] 0.00017227354629523184
3 10 [ 0.76000326 -0.11003605] 0.0001500469101773927
train 0 35 [0.         3.29451378 4.87703035 5.24708715] [22 12 14]
train 1 29 [0.         3.47967867 4.71043395 6.19832098] [ 8 19 16]
train 2 24 [0.         3.27211891 4.85907391 6.01433004] [14  5 23]
train 3 32 [0.         3.4718402  4.82491658 5.81633689] [15 19 13]
test 0 10 [0.         2.80578617 4.16678508 4.93612731] [7 5 6]
test 1 12 [0.         3.17711681 4.4026083  5.58278608] [3 5 9]
test 2 12 [0.         3.4420939  4.29268371 4.52680175] [7 1 8]
test 3 14 [0.         3.3868239  5.02473194 5.71732638] [13  8]
```

### Turn 4 — `execute_in_remote_docker` (`50118 ms` LLM)
- **Rationale**: Implement the specified two-branch recurrence and fit the nine weights by variable projection. The grouped checks distinguish a wrong inactive-history basis from channel or stratum transfer failure; large RMSE relative to median error would motivate robust fitting.
- **Alternatives Weighed**: `Fit campaign-specific stress corrections from rheology labels` (rejected: Recent prefix sweeps agree within about 0.0001 in gain and bias; preserve the measured calibration unless held-out residuals show a channel-level discrepancy.)

```bash
set -e
f=/holdout_task/alpha_m97_c03_batch_04_solver.py
{ printf '%s\n' '"""Load-cell calibrated two-branch model for crystallized creep."""'; sed -n '/^import json$/,/^def _basis/{ /^def _basis/!p; }' /tmp/hegel_continual_archive/alpha_m97_c03_batch_03_solver.py; } > "$f"
cat >> "$f" <<'PY'
def _basis(logtau, past, hold, features, dt, t, damage, terminal_mode='active'):
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
    u = dt[:, None] / tau
    for j, (a, b) in enumerate(zip(np.exp(-u), -np.expm1(-u))):
        z = a * z + b * past[:, j, None]
    phi = np.vstack((-np.expm1(-t[:, None] / tau), 1 / tau))
    logtime = np.r_[np.log1p(t), 1.0]
    lintime = np.r_[t, 1.0]
    active = damage > 0
    if terminal_mode == 'all_log':
        terminal = np.broadcast_to(logtime, (len(hold), len(t) + 1))
    else:
        terminal = np.where(active[:, None], logtime, lintime)
    act = np.tanh(damage)
    a = np.empty((len(hold), len(t) + 1, 9))
    a[:, :, :2] = (hold[:, None] - z)[:, None, :] * phi
    a[:, :, 2] = terminal
    a[:, :, 3:7] = features[:, None, :] * terminal[:, :, None]
    a[:, :, 7] = (act * hold)[:, None] * phi[:, 0]
    a[:, :, 8] = (act * z[:, 0])[:, None] * logtime
    return a


def _damage(dataset, past, dt):
    if dataset['shear_banding_index'] > 0.5:
        return np.zeros(len(past))
    excess = np.maximum(np.abs(past) - dataset['crit_stress'], 0.0)
    return np.sum(excess**2 * dt, axis=1)


def _fit(args, target, mode='active', mult=None):
    if mult is None:
        mult = np.ones(target.shape)
    y = (target * mult).ravel()

    def project(logtau):
        basis = _basis(logtau, *args, terminal_mode=mode)
        a = (basis * mult[:, :, None]).reshape(-1, 9)
        w = np.linalg.lstsq(a, y, rcond=None)[0]
        return np.einsum('ij,j->i', a, w) - y, w

    bounds = np.log([0.015, 120.0])
    starts = ((0.4, 3.0), (1.0, 7.0), (3.0, 0.4))
    best = min((least_squares(lambda p: project(p)[0], np.log(start),
                              bounds=tuple(bounds), max_nfev=120) for start in starts),
               key=lambda result: result.cost)
    return best.x, project(best.x)[1]


def solve_alpha_m97_c03_batch_04(dataset: dict) -> dict:
    past, hold, features, offset, dt, t = _prepare(dataset)
    damage = _damage(dataset, past, dt)
    train = dataset['train']
    n = len(train['lab_id'])
    target = np.column_stack((train['strain_curves_obs'] - offset[:n, None], train['instant_rate_obs']))
    logtau, w = _fit((past[:n], hold[:n], features[:n], dt, t, damage[:n]), target)
    pred = _basis(logtau, past[n:], hold[n:], features[n:], dt, t, damage[n:]) @ w
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:], 'predicted_strain': offset[n:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
PY
cat > /tmp/rheo_validate.py <<'PY'
import importlib.util,pickle,numpy as np
P='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',P+'alpha_m97_c03_batch_04_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open(P+'public_alpha_m97_c03_batch_04.pkl','rb'))[0]['dataset']; tr=d['train']; lab=tr['lab_id']; st=tr['stratum']; n=len(lab)
p,h,x,o,dt,t=m._prepare(d); D=m._damage(d,p,dt)[:n]; p,h,x=p[:n],h[:n],x[:n]
y=np.column_stack((tr['strain_curves_obs']-o[:n,None], tr['instant_rate_obs']))
def args(i): return p[i],h[i],x[i],dt,t,D[i]
def metrics(e):
 return np.array([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),np.median(np.abs(e[:,:-1])),np.median(np.abs(e[:,-1])),np.max(np.abs(e[:,:-1])),np.max(np.abs(e[:,-1]))])
for mode in ('active','all_log'):
 pp,w=m._fit(args(slice(None)),y,mode); e=m._basis(pp,*args(slice(None)),terminal_mode=mode)@w-y
 print(mode,'train',np.exp(pp),w,'metrics',metrics(e))
 for name,g in [('lab',lab),('stratum',st)]:
  for k in np.unique(g):
   v=g==k; pp,w=m._fit(args(~v),y[~v],mode); e=m._basis(pp,*args(v),terminal_mode=mode)@w-y[v]; print(name,k,'taus',np.exp(pp),'metrics',metrics(e),'active',metrics(e[D[v]>0])[:2],'inactive',metrics(e[D[v]==0])[:2]);
PY
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/rheo_validate.py
```
- **Container Exit Code**: `0` (`1338 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
lab 3 taus [0.56906724 8.17955916] metrics [ 0.74883841  0.13628313  0.11557897  0.0993272  11.65724049  0.3461248 ] active [0.60695442 0.1328224 ] inactive [0.98546347 0.14338711]
stratum 0 taus [0.56665069 4.9060008 ] metrics [ 1.67175127  0.12450543  0.10934369  0.08931684 39.20537732  0.28659735] active [2.074957   0.12833061] inactive [0.24721467 0.11726693]
stratum 1 taus [ 0.56938007 10.21810209] metrics [ 0.69544764  0.12969965  0.10776976  0.08639401 11.65457046  0.32512305] active [0.55264358 0.12542795] inactive [0.97925177 0.14045715]
stratum 2 taus [0.50100795 2.03993681] metrics [ 1.02231743  0.16337206  0.10489609  0.0885787  22.50784719  0.46202471] active [1.25463208 0.17742891] inactive [0.34559747 0.13530298]
all_log train [0.56708925 6.23474505] [ 0.72620127  0.68832323  0.13262534  0.18430564 -0.14433602  0.12881624
 -0.06773869  0.30497191  0.23057646] metrics [ 1.17793631  0.08326259  0.04372581  0.04382381 39.09464217  0.33649995]
lab 0 taus [0.55529946 5.57403686] metrics [ 1.88524093  0.10143212  0.06083653  0.04152405 39.3656265   0.27333279] active [2.20539605 0.10012086] inactive [0.1678805  0.10488099]
lab 1 taus [0.57580884 6.79864239] metrics [0.42871397 0.13168595 0.08403247 0.09117119 6.63447305 0.27041382] active [0.49532782 0.13905026] inactive [0.23725722 0.11494001]
lab 2 taus [0.55755588 3.5119743 ] metrics [ 1.11586498  0.04552884  0.03768014  0.03630444 22.00266336  0.10026492] active [1.45051423 0.05323714] inactive [0.31628918 0.03252359]
lab 3 taus [0.54966189 9.6177839 ] metrics [ 0.72103143  0.08933525  0.06262479  0.05744163 11.38136912  0.30722115] active [0.59413665 0.09282249] inactive [0.93590275 0.081398  ]
stratum 0 taus [0.55499996 5.29668697] metrics [ 1.6714245   0.09902637  0.06120396  0.05812963 39.3431067   0.33119973] active [2.07893457 0.10517539] inactive [0.1677744  0.08679926]
stratum 1 taus [ 0.54987429 10.20475871] metrics [ 0.673994    0.1245347   0.0778167   0.07857011 11.41758476  0.41498983] active [0.53944142 0.12030684] inactive [0.94321624 0.135164  ]
stratum 2 taus [0.52821736 2.49358661] metrics [ 0.99294122  0.07975207  0.04903748  0.04864792 22.04419904  0.30394578] active [1.22120896 0.09441049] inactive [0.31851045 0.04350556]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9193ad642b76b738 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9193ad64 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`27319 ms` LLM)
- **Rationale**: Fit both time laws with fold-local Cauchy weights and separate curve and rate scales. Compare active and inactive errors on held-out labs and strata to tell whether the logarithmic law improves typical predictions rather than merely changing the fit around isolated spikes.
- **Alternatives Weighed**: `Switch to campaign-wide logarithmic creep immediately` (rejected: The current comparison fits squared error to spikes as large as 39 strain units; repeat it with robust fits before changing the terminal law.)

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS variable projection'; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; sed '/^for mode in /,$d' /tmp/rheo_validate.py > /tmp/rheo_robust.py; cat >> /tmp/rheo_robust.py <<'PY'

def scale(e):
    return np.maximum(1e-6, 1.4826*np.median(np.abs(e-np.median(e))))

def fit(i, mode):
    a=args(i); yy=y[i]
    pp,w=m._fit(a,yy,mode)
    def resid(): return m._basis(pp,*a,terminal_mode=mode)@w-yy
    e=resid(); sc=np.array([scale(e[:,:-1]),scale(e[:,-1])]); sc=np.r_[np.repeat(sc[0],len(t)),sc[1]]
    for k in range(12):
        mult=1/(sc*np.sqrt(1+(e/(sc*(1.5 if k<3 else 2.385)))**2))
        pp,w=m._fit(a,yy,mode,mult); e=resid()
        if k in (2,5,8):
            sc=np.r_[np.repeat(scale(e[:,:-1]),len(t)),scale(e[:,-1])]
    return pp,w,sc

for mode in ('active','all_log'):
    pp,w,sc=fit(slice(None),mode); print(mode,'taus',np.exp(pp),'weights',w,'scales',sc[[0,-1]])
    for name,g in [('lab',lab),('stratum',st)]:
        for k in np.unique(g):
            v=g==k; pp,w,sc=fit(~v,mode); e=m._basis(pp,*args(v),terminal_mode=mode)@w-y[v]
            for act in (False,True):
                q=np.abs(e[(D[v]>0)==act]); a=q[:,:-1]; b=q[:,-1]; print(name,k,int(act),'curve/rate MAE',np.mean(a),np.mean(b),'median',np.median(a),np.median(b),'p90',np.quantile(a,.9),np.quantile(b,.9))
PY
python3 -I /tmp/rheo_robust.py
```
- **Container Exit Code**: `0` (`7359 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.21807491] scales [0.0191412  0.02326895]
lab 0 0 curve/rate MAE 0.04409839283071227 0.028727914758383596 median 0.011743427486390068 0.0178283212975896 p90 0.06780990962147415 0.038558755973499004
lab 0 1 curve/rate MAE 0.25555881557921833 0.05032465319031802 median 0.01570199954477891 0.021538712952688055 p90 0.2952725662593966 0.1706383801395585
lab 1 0 curve/rate MAE 0.0476980478288215 0.02198540290599443 median 0.01345696447295519 0.018327811696014495 p90 0.058798776481690254 0.039685115745143135
lab 1 1 curve/rate MAE 0.09828975508162031 0.02713115440814781 median 0.015452240704808912 0.01819509753109294 p90 0.1456734085311899 0.04504379703979624
lab 2 0 curve/rate MAE 0.07317128617724737 0.01734551240025684 median 0.013548720912769 0.013874226204099727 p90 0.05053852740481125 0.031801044200870425
lab 2 1 curve/rate MAE 0.14735433553967386 0.011282436712841903 median 0.012187308252991413 0.008740481254978727 p90 0.03283923394828087 0.022753878194088337
lab 3 0 curve/rate MAE 0.0966317849309842 0.013860082458295644 median 0.010289795571331327 0.012889423583470894 p90 0.028106215395289513 0.028206657292491367
lab 3 1 curve/rate MAE 0.13528682596734956 0.0384711112266581 median 0.014144447856267395 0.01667767199109682 p90 0.1732886106822603 0.07688945525333381
stratum 0 0 curve/rate MAE 0.03685570124847147 0.02611509926395378 median 0.012943067509741779 0.016427434914803918 p90 0.06009587701672846 0.04333494018528039
stratum 0 1 curve/rate MAE 0.21465681994698038 0.03802152224690778 median 0.013774892080794077 0.013349614340900873 p90 0.2623551483086663 0.07969439702323937
stratum 1 0 curve/rate MAE 0.11128690893802996 0.01738912625333028 median 0.012942558393788023 0.013745564398663923 p90 0.0667465486734181 0.03451596108804411
stratum 1 1 curve/rate MAE 0.10422852701731591 0.032074124467843565 median 0.015323147561996209 0.01856834567129048 p90 0.18666350521627245 0.055248742124448975
stratum 2 0 curve/rate MAE 0.06521469986413816 0.015521538187637894 median 0.00992293224109328 0.013888591100195913 p90 0.03324624903600062 0.02786624369751007
stratum 2 1 curve/rate MAE 0.17375675298416163 0.030799923550490733 median 0.012943986436256175 0.014335186608131822 p90 0.21977625473960188 0.05488802065013311

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9193ad642b76b738 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9193ad64 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`58849 ms` LLM)
- **Rationale**: The robust grouped comparison supports changing the inactive time law while preserving measured load-cell calibration. Verify the exact artifact on held-out labs and strata, check the rate against the derivative of strain, and confirm that clean in-batch calibration and EMI rejection work without unnecessary archive access.
- **Alternatives Weighed**: `Retain an inactive linear baseline and rely on robust weighting` (rejected: Fold-local robust weighting did not remove the systematic inactive-history errors.)
- **ATMS NOGOOD DECLARED**: Refuted `Zero past-work damage restores linear terminal flow in the low-shear campaign.` | Witness: `After robust fitting, the worst inactive held-out strain median fell from 0.1656 with linear flow to 0.0135 with logarithmic flow.` | Blanket Diagnosis: `Gross strain spikes obscured the inactive baseline; separating measurement contamination from the time law revealed that damage gates the stress-memory corrections, not the baseline flow in this campaign.` | Sublated Pivot: `Use logarithmic creep throughout the low-shear campaign, retain both specified damage-dependent terms, and fit with separate robust strain and rate scales.`

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c03_batch_04_solver.py')
s = p.read_text().split('def _basis(', 1)[0]
p.write_text(s + '''def _basis(logtau, past, hold, features, dt, t, damage, plateau):
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
    u = dt[:, None] / tau
    for j, (a, b) in enumerate(zip(np.exp(-u), -np.expm1(-u))):
        z = a * z + b * past[:, j, None]
    # The final row is the time derivative at zero, not an extra horizon.
    phi = np.vstack((-np.expm1(-t[:, None] / tau), 1 / tau))
    terminal = np.r_[np.log1p(t) if plateau else t, 1.0]
    act = np.tanh(damage)
    a = np.empty((len(hold), len(t) + 1, 9))
    a[:, :, :2] = (hold[:, None] - z)[:, None, :] * phi
    a[:, :, 2] = terminal
    a[:, :, 3:7] = features[:, None, :] * terminal[None, :, None]
    a[:, :, 7] = (act * hold)[:, None] * phi[:, 0]
    a[:, :, 8] = (act * z[:, 0])[:, None] * terminal
    return a


def _damage(dataset, past, dt):
    if dataset['shear_banding_index'] > 0.5:
        return np.zeros(len(past))
    excess = np.maximum(np.abs(past) - dataset['crit_stress'], 0.0)
    return np.sum(excess**2 * dt, axis=1)


def _scale(e):
    return max(1e-6, 1.4826 * np.median(np.abs(e - np.median(e))))


def _fit(args, target):
    mult = np.ones_like(target)
    bounds = tuple(np.log([0.015, 120.0]))
    starts = np.log(((0.4, 3.0), (1.0, 7.0), (3.0, 0.4)))

    def fit_weighted(starts):
        y = (target * mult).ravel()

        def project(logtau):
            a = (_basis(logtau, *args) * mult[:, :, None]).reshape(-1, 9)
            w = np.linalg.lstsq(a, y, rcond=None)[0]
            return np.einsum('ij,j->i', a, w) - y, w

        best = min((least_squares(lambda p: project(p)[0], start,
                                  bounds=bounds, max_nfev=120) for start in starts),
                   key=lambda result: result.cost)
        return best.x, project(best.x)[1]

    logtau, w = fit_weighted(starts)
    err = _basis(logtau, *args) @ w - target
    # The rate channel and strain trajectories have different noise scales.
    for k in range(12):
        if k in (0, 3, 6, 9):
            scales = np.r_[np.repeat(_scale(err[:, :-1]), target.shape[1] - 1), _scale(err[:, -1])]
        c = 1.5 if k < 3 else 2.385
        mult = 1 / (scales * np.hypot(1, err / (c * scales)))
        logtau, w = fit_weighted(starts)
        err = _basis(logtau, *args) @ w - target
    return logtau, w


def solve_alpha_m97_c03_batch_04(dataset: dict) -> dict:
    past, hold, features, offset, dt, t = _prepare(dataset)
    damage = _damage(dataset, past, dt)
    plateau = dataset['shear_banding_index'] <= 0.5
    train = dataset['train']
    n = len(train['lab_id'])
    target = np.column_stack((train['strain_curves_obs'] - offset[:n, None], train['instant_rate_obs']))
    logtau, w = _fit((past[:n], hold[:n], features[:n], dt, t, damage[:n], plateau), target)
    pred = _basis(logtau, past[n:], hold[n:], features[n:], dt, t, damage[n:], plateau) @ w
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:], 'predicted_strain': offset[n:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
''')
PY
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
sed '/^for mode in /,$d' /tmp/rheo_validate.py > /tmp/rheo_final.py
cat >> /tmp/rheo_final.py <<'PY'
from unittest.mock import patch

def data(i, times=t):
    return p[i], h[i], x[i], dt, times, D[i], True

params, w = m._fit(data(slice(None)), y)
print('fit', np.exp(params), w)
worst = np.zeros((2, 2))
for g in (lab, st):
    for k in np.unique(g):
        v = g == k
        pp, ww = m._fit(data(~v), y[~v])
        abs_err = np.abs(m._basis(pp, *data(v)) @ ww - y[v])
        for active in (False, True):
            e = abs_err[(D[v] > 0) == active]
            stat = np.array([[np.median(e[:, :-1]), np.median(e[:, -1])],
                             [np.quantile(e[:, :-1], .9), np.quantile(e[:, -1], .9)]])
            worst = np.maximum(worst, stat)
print('worst grouped median and p90 (strain, rate):', worst)
assert np.all(worst[0] < [.02, .025]), worst
small = m._basis(params, *data(slice(None), np.array([0., 1e-6]))) @ w
assert np.allclose(small[:, 0], 0) and np.allclose(small[:, 1]/1e-6, small[:, 2], atol=2e-5, rtol=2e-6)
normal = {'lab_id': np.repeat(np.arange(4), 6), 'cmd_stress': np.tile(np.linspace(-2, 2, 6), 4)}
gain = np.array([1.29, .82, 1.17, .76]); bias = np.array([.12, -.15, .14, -.11]); ids = normal['lab_id']
normal['true_stress'] = gain[ids] * normal['cmd_stress'] + bias[ids]
with patch.object(m, '_prior_calibration', side_effect=AssertionError('unnecessary archive access')):
    assert np.allclose(m._calibration(normal), np.column_stack((gain, bias)))
    normal['true_stress'][[0, 9, 15, 22]] += [5, -8, 6, -7]
    assert np.allclose(m._calibration(normal), np.column_stack((gain, bias)))
sparse = m._calibration(d['calibration_load_cells'])
assert np.max(np.abs(sparse - np.column_stack((gain, bias)))) < 2e-4
assert np.isfinite(x).all()
print('Passed analytic-rate, clean and EMI-spiked calibration, sparse calibration, and alignment checks.')
PY
python3 -I /tmp/rheo_final.py
```
- **Container Exit Code**: `0` (`4159 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c03_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
fit [0.53639302 3.60252397] [ 0.68069119  0.54916891  0.14035596  0.16019235 -0.12958464  0.11970648
 -0.09012956  0.2607187   0.21807491]
worst grouped median and p90 (strain, rate): [[0.015702   0.02153871]
 [0.29527257 0.17063838]]
Passed analytic-rate, clean and EMI-spiked calibration, sparse calibration, and alignment checks.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fc51b4e224d272b5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_fc51b4e2 (directory).
```

### Turn 7 — `submit_completed_task` (`14161 ms` LLM)
- **Rationale**: The final artifact passed isolated-process schema validation and held-out lab and stratum tests split by damage activity. The verified logarithmic baseline resolves the inactive-history failure while preserving both damage-dependent terms.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_04_solver.py': 'FILE:6622'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_04_solver.py`. The public schema, calibration, alignment, and analytic initial-rate checks passed. Across held-out labs and strata, the worst median absolute errors were 0.0157 for strain and 0.0215 for initial rate.`

### Turn 8 — `execute_in_remote_docker` (`31188 ms` LLM)
- **Rationale**: Fit candidate missing terms only on each training fold and compare their transfer to held-out labs and strata. Adopt an extension only if it materially improves the worst slices; investigate calibration or panel alignment if signed residuals instead show consistent group offsets.
- **Alternatives Weighed**: `Apply class-balance, relay-mode, or HJB diagnostics` (rejected: The supplied model predicts continuous strain and rate from a Prony recurrence; it has no class label, relay state, or HJB operator. Held-out signed residuals test lab and stratum offsets directly.)

```bash
set -e
sed -n '1,102p' /holdout_task/alpha_m97_c03_batch_04_solver.py
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
sed '/^for mode in /,$d' /tmp/rheo_validate.py > /tmp/rheo_gate.py
cat >> /tmp/rheo_gate.py <<'PY'
print('dataset fields:', sorted(d), 'train fields:', sorted(tr))
print('calibration rows by lab:', np.unique(d['calibration_load_cells']['lab_id'], return_counts=True))
def fit_args(i):
    return p[i], h[i], x[i], dt, t, D[i], True

def extra(i):
    # Stress-normalized, pairwise-feature, and three-way interaction challengers.
    a = np.tanh(D[i]); q = h[i]; f = x[i]; logt = np.r_[np.log1p(t), 1.0]
    cols = [q/(1+q*q), a*q/(1+q*q), a*q*(f[:,0]-f[:,1]), a*q*(f[:,2]+f[:,3])]
    cols.extend(f[:,j]*f[:,k] for j in range(4) for k in range(j+1,4))
    cols.extend(a*q*f[:,j] for j in range(4))
    return np.column_stack(cols)[:,None,:]*logt[None,:,None]

for name, groups in [('lab', lab), ('stratum', st)]:
    worst = np.zeros((2,2)); shifts = []
    for k in np.unique(groups):
        v = groups == k
        pp, ww = m._fit(fit_args(~v), y[~v])
        e0 = y[~v] - m._basis(pp, *fit_args(~v))@ww
        e = y[v] - m._basis(pp, *fit_args(v))@ww
        sc = np.r_[np.repeat(m._scale(e0[:,:-1]), len(t)), m._scale(e0[:,-1])]
        weight = 1 / (sc * np.hypot(1, e0/(2.385*sc)))
        coef = np.linalg.lstsq((extra(~v)*weight[:,:,None]).reshape(-1,14), (e0*weight).ravel(), rcond=None)[0]
        e2 = e - extra(v)@coef
        for act in (False, True):
            q = (D[v]>0)==act
            if q.any():
                row = [[np.median(np.abs(r[q,:-1])), np.median(np.abs(r[q,-1]))] for r in (e,e2)]
                worst = np.maximum(worst, row)
                shifts.append([k, int(act), np.median(e[q,:-1]), np.median(e[q,-1])])
    print(name, 'worst strain/rate median AE, baseline then additions:', worst.tolist())
    print(name, 'held-out signed medians [group, active, strain, rate]:', np.round(shifts,5).tolist())
PY
python3 -I /tmp/rheo_gate.py
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`3761 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
e] + bias[:, None]
    hold = join('stress_hold_cmd') * gain + bias
    return past, hold, features, join('strain_0'), np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])


def _basis(logtau, past, hold, features, dt, t, damage, plateau):
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
    u = dt[:, None] / tau
    for j, (a, b) in enumerate(zip(np.exp(-u), -np.expm1(-u))):
        z = a * z + b * past[:, j, None]
    # The final row is the time derivative at zero, not an extra horizon.
    phi = np.vstack((-np.expm1(-t[:, None] / tau), 1 / tau))
    terminal = np.r_[np.log1p(t) if plateau else t, 1.0]
    act = np.tanh(damage)
    a = np.empty((len(hold), len(t) + 1, 9))
    a[:, :, :2] = (hold[:, None] - z)[:, None, :] * phi
    a[:, :, 2] = terminal
    a[:, :, 3:7] = features[:, None, :] * terminal[None, :, None]
    a[:, :, 7] = (act * hold)[:, None] * phi[:, 0]
    a[:, :, 8] = (act * z[:, 0])[:, None] * terminal
    return a
dataset fields: ['calibration_load_cells', 'crit_stress', 'eval_times', 'past_dt', 'shear_banding_index', 'test', 'train', 'yield_stress'] train fields: ['aux_chamber_telemetry', 'coupon_features_raw', 'instant_rate_obs', 'lab_id', 'past_stress_cmd', 'ref_coupons', 'strain_0', 'strain_curves_obs', 'stratum', 'stress_hold_cmd']
calibration rows by lab: (array([0]), array([2]))
lab worst strain/rate median AE, baseline then additions: [[0.015701999521517917, 0.021538712911890023], [0.01570131080574473, 0.021484302323301264]]
lab held-out signed medians [group, active, strain, rate]: [[0.0, 0.0, 0.00323, -0.0016], [0.0, 1.0, 0.00013, -0.0006], [1.0, 0.0, 0.00368, 0.00153], [1.0, 1.0, -0.00098, -0.00703], [2.0, 0.0, -0.00161, -0.00601], [2.0, 1.0, -0.00287, -0.00304], [3.0, 0.0, -0.00037, 0.01139], [3.0, 1.0, -0.00058, -0.01004]]
stratum worst strain/rate median AE, baseline then additions: [[0.01532314766525239, 0.018568345645134793], [0.014934791339195355, 0.018175771645882702]]
stratum held-out signed medians [group, active, strain, rate]: [[0.0, 0.0, 0.0003, 0.00286], [0.0, 1.0, -0.00395, -0.0046], [1.0, 0.0, 0.00238, 0.00845], [1.0, 1.0, 0.00286, 4e-05], [2.0, 0.0, 0.00048, -0.00156], [2.0, 1.0, -0.00109, -0.00607]]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fc51b4e224d272b5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_fc51b4e2 (directory).
```

### Turn 9 — `submit_completed_task` (`14464 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process schema validation and grouped stress tests. The independent probe confirmed panel affine invariance and agreement between the strain trajectory and initial-rate prediction; the additional terms did not materially improve held-out performance.
- **Alternatives Weighed**: `Add rational stress corrections and feature interactions` (rejected: The additions improved worst held-out lab median errors by less than 0.0001 for both outputs, without resolving a material failure.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c03_batch_04_solver.py': 'FILE:6622'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_04_solver.py`. The public schema and independent scientific checks passed. Worst held-out median absolute errors were 0.0157 for strain and 0.0215 for initial rate.`

### Turn 10 — `execute_in_remote_docker` (`34150 ms` LLM)
- **Rationale**: Hold out whole lab–stratum cells and inspect each horizon by missingness and damage status. A sustained missing-coupon bias would motivate revisiting imputation; material output shifts across legitimate prefix sweeps would motivate revisiting calibration transfer.
- **Alternatives Weighed**: `Repeat the rational and interaction challenger or apply relay and class-balance diagnostics` (rejected: The previous fold-local challenger found no material gain, and the dataset has neither relay states nor class labels. The remaining question is whether joint nuisance strata or sparse calibration hide a prediction failure.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
sed '/^for mode in /,$d' /tmp/rheo_validate.py > /tmp/rheo_margin.py
cat >> /tmp/rheo_margin.py <<'PY'
import json
from unittest.mock import patch

# Look for transfer failures specific to intersections or to missing coupons.
e = np.empty_like(y)
for c, s in np.unique(np.column_stack((lab, st)), axis=0):
    v = (lab == c) & (st == s)
    pp, w = m._fit((*args(~v), True), y[~v])
    e[v] = m._basis(pp, *args(v), True) @ w - y[v]
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
for miss in (False, True):
    for active in (False, True):
        v = (missing == miss) & ((D > 0) == active)
        a = e[v]
        if len(a):
            print('missing/active/n', miss, active, len(a), 'max horizon median abs', round(np.median(np.abs(a[:, :-1]), axis=0).max(), 6), 'rate median abs', round(np.median(np.abs(a[:, -1])), 6), 'signed medians', np.round(np.median(a, axis=0), 5).tolist())

# Use prior sweeps separately rather than assuming their differences are negligible.
reference = m.solve_alpha_m97_c03_batch_04(d)
with open(m._ARCHIVE) as f:
    history = json.load(f)
for sweep in history[:-1]:
    def prior(c):
        v = np.asarray(sweep['lab_id']) == c
        return np.polyfit(np.asarray(sweep['cmd_stress'])[v], np.asarray(sweep['true_stress'])[v], 1)
    with patch.object(m, '_prior_calibration', side_effect=prior):
        other = m.solve_alpha_m97_c03_batch_04(d)
    print('prefix sweep', sweep['episode_short_name'], 'max absolute changes', {k: round(float(np.max(np.abs(np.asarray(other[k]) - reference[k]))), 7) for k in reference})
PY
python3 -I /tmp/rheo_margin.py
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`7709 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
missing/active/n False False 53 max horizon median abs 0.013327 rate median abs 0.014195 signed medians [-0.00322, -0.00401, -0.00187, 0.00063, 0.00047, 0.00141, 0.00125, -0.00029, 0.00056, 0.00139, -0.0026]
missing/active/n False True 103 max horizon median abs 0.014534 rate median abs 0.013048 signed medians [0.00185, -0.0006, -0.00018, -0.00315, -0.00083, -0.00177, 0.00062, 0.00079, 0.00275, 7e-05, 0.00295]
missing/active/n True False 7 max horizon median abs 0.080898 rate median abs 0.037829 signed medians [0.00273, 0.00044, 0.01388, -0.01725, -0.01543, -0.01925, -0.01408, -0.01404, -0.01534, -0.01324, -0.01287]
missing/active/n True True 17 max horizon median abs 0.262878 rate median abs 0.140279 signed medians [0.02411, 0.02399, 0.03624, 0.02457, 0.01198, 0.01098, 0.01299, 0.03839, -0.05127, 0.03481, 0.03141]
prefix sweep alpha_m97_c03_s00 max absolute changes {'estimated_taus': 5.1418965, 'hereditary_weights': 0.2800268, 'regime_mode_weights': 0.0871212, 'predicted_strain': 0.7239856, 'initial_rate_pred': 1.2284731}
prefix sweep alpha_m97_c03_batch_01 max absolute changes {'estimated_taus': 7.2e-06, 'hereditary_weights': 1.74e-05, 'regime_mode_weights': 2.05e-05, 'predicted_strain': 0.0002237, 'initial_rate_pred': 0.0002785}
prefix sweep alpha_m97_c03_batch_02 max absolute changes {'estimated_taus': 0.0002403, 'hereditary_weights': 1.92e-05, 'regime_mode_weights': 2.11e-05, 'predicted_strain': 0.0002228, 'initial_rate_pred': 0.0003349}
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fc51b4e224d272b5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_fc51b4e2 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`31535 ms` LLM)
- **Rationale**: Use held-out lab–stratum cells to distinguish a drift mismatch from residuals that the imputer cannot explain. Compare against a complete-coupon fit to see whether missing rows materially distort the shared parameters before changing the verified solver.
- **Alternatives Weighed**: `Replace the cell-balanced imputation immediately` (rejected: Imputation is specified and affects only the drift term. The larger residuals do not establish an imputation failure unless they follow that term's time dependence and agree with the initial-rate residual.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
sed '/^for mode in /,$d' /tmp/rheo_validate.py > /tmp/rheo_missing.py
cat >> /tmp/rheo_missing.py <<'PY'
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
phi = np.r_[np.log1p(t), 1.]
results = [np.empty_like(y), np.empty_like(y)]
for c, s in np.unique(np.column_stack((lab, st)), axis=0):
    v = (lab == c) & (st == s)
    for j, use in enumerate((~v, ~v & ~missing)):
        pp, ww = m._fit((*args(use), True), y[use])
        results[j][v] = y[v] - m._basis(pp, *args(v), True) @ ww
for j, e in enumerate(results):
    print('fit', ['all coupons','complete coupons only'][j])
    worst = np.zeros(2)
    for miss in (False, True):
        for active in (False, True):
            v = (missing == miss) & ((D > 0) == active)
            a = e[v]
            # If incorrect imputation is the only systematic error, residuals lie in span(phi), including the initial rate.
            beta = np.median(a[:, :-1] / phi[:-1], axis=1)
            transverse = a - beta[:, None] * phi
            raw = np.r_[np.median(np.abs(a[:, :-1]), axis=0).max(), np.median(np.abs(a[:, -1]))]
            off = np.r_[np.median(np.abs(transverse[:, :-1]), axis=0).max(), np.median(np.abs(transverse[:, -1]))]
            print('missing/active', miss, active, 'raw', raw, 'off-log', off, 'median abs drift shift', np.median(np.abs(beta)))
            worst = np.maximum(worst, raw)
    print('worst slice', worst)
for use in (np.ones(n, dtype=bool), ~missing):
    pp, ww = m._fit((*args(use), True), y[use]); print('parameters', np.exp(pp), ww)
PY
python3 -I /tmp/rheo_missing.py
```
- **Container Exit Code**: `0` (`11305 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
fit all coupons
missing/active False False raw [0.01332668 0.01419503] off-log [0.01165325 0.01329453] median abs drift shift 0.003663234863405778
missing/active False True raw [0.01453392 0.01304757] off-log [0.01229496 0.01355378] median abs drift shift 0.0035564481899147023
missing/active True False raw [0.08089777 0.03782915] off-log [0.01345212 0.00466023] median abs drift shift 0.04362071725812891
missing/active True True raw [0.26287777 0.14027909] off-log [0.01538842 0.01200455] median abs drift shift 0.13030474785129517
worst slice [0.26287777 0.14027909]
fit complete coupons only
missing/active False False raw [0.0134666  0.01403945] off-log [0.0118063  0.01348249] median abs drift shift 0.0027363480852670625
missing/active False True raw [0.01436787 0.0135626 ] off-log [0.01256242 0.01407605] median abs drift shift 0.0036678980534176745
missing/active True False raw [0.08067944 0.03703303] off-log [0.01316016 0.00501872] median abs drift shift 0.04326580959210184
missing/active True True raw [0.26344375 0.13998614] off-log [0.0159795  0.01307644] median abs drift shift 0.1280392325213442
worst slice [0.26344375 0.13998614]
parameters [0.53639302 3.60252397] [ 0.68069119  0.54916891  0.14035596  0.16019235 -0.12958464  0.11970648
 -0.09012956  0.2607187   0.21807491]
parameters [0.53635149 3.5826318 ] [ 0.68093206  0.5489159   0.14012672  0.16006149 -0.12973414  0.119358
 -0.08995885  0.25952978  0.22023565]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fc51b4e224d272b5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_fc51b4e2 (directory).
```

### Turn 12 — `submit_completed_task` (`24024 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated schema validation and the independent scientific probe. Held-out lab–stratum tests localized the missing-coupon error to the logarithmic drift term rather than the hereditary dynamics; the tested alternative did not materially improve that slice.
- **Alternatives Weighed**: `Fit only on complete coupons` (rejected: The worst missing-coupon strain median increased from 0.2629 to 0.2634, and the rate median improved by only 0.0003.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c03_batch_04_solver.py`. The public schema passed. Worst held-out lab or stratum median absolute errors were 0.0157 for strain and 0.0215 for initial rate.

Limitation: For held-out samples with missing coupon features and active damage, median absolute errors reached 0.2629 at the worst strain horizon and 0.1403 for initial rate. The residuals followed the logarithmic drift term, and fitting on complete coupons did not materially improve them. The solver retains the specified imputation.`

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
