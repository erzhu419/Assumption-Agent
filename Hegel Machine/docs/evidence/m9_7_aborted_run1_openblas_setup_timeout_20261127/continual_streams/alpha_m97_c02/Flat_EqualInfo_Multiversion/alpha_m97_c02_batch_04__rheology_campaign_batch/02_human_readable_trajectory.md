# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c02_batch_04__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c02_step_04` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `335,697` (`324,059` / `93,184` / `11,638`) |
| **Total Episode Wall Time (s)** | `376.37s` |

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

### Turn 1 — `execute_in_remote_docker` (`19959 ms` LLM)
- **Rationale**: Inspect field names, shapes, and validator expectations before fitting. Missing calibration coverage would trigger the prescribed archive fallback, and the available training targets determine which prediction metrics can be validated.
- **Alternatives Weighed**: `Reuse the prior solver unchanged` (rejected: The present batch specifies a logarithmic plateau; prior code can inform shared preprocessing but should not determine the current regime law.)

```bash
cd /holdout_task && cat validate_public_schema.py && ls -lh /tmp/hegel_continual_archive && cat public_asset_index.json && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference-panel alignment imputation variable projection' ; cd /holdout_task && python3 - <<'PY'
import pickle, numpy as np
from collections.abc import Mapping
with open('public_alpha_m97_c02_batch_04.pkl','rb') as f: d=pickle.load(f)
def show(x, name='', level=0):
 if isinstance(x,Mapping):
  print(name,'dict',list(x)[:24]);
  if level<3:
   for k,v in list(x.items())[:24]: show(v, name+'.'+str(k),level+1)
 elif isinstance(x,np.ndarray): print(name,x.shape,x.dtype,str(x.reshape(-1)[:8]))
 elif isinstance(x,(list,tuple)):
  print(name,type(x).__name__,len(x));
  if x and level<3: show(x[0],name+'[0]',level+1)
 else: print(name,type(x).__name__,str(x)[:180])
show(d)
PY
```
- **Container Exit Code**: `0` (`586 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
 list 1
[0] dict ['dataset', 'gt']
[0].dataset dict ['past_dt', 'eval_times', 'crit_stress', 'yield_stress', 'shear_banding_index', 'calibration_load_cells', 'train', 'test']
[0].dataset.past_dt (18,) float64 [0.7303 0.4761 0.699  0.3755 0.2141 0.3417 0.556  0.323 ]
[0].dataset.eval_times (10,) float64 [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111]
[0].dataset.crit_stress float 2.65
[0].dataset.yield_stress float 2.65
[0].dataset.shear_banding_index float 0.12
[0].dataset.calibration_load_cells dict ['lab_id', 'cmd_stress', 'true_stress']
[0].dataset.train dict ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0', 'aux_chamber_telemetry', 'strain_curves_obs', 'instant_rate_obs']
[0].dataset.test dict ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0', 'aux_chamber_telemetry']
[0].gt dict ['taus', 'theta', 'vp_theta', 'stress_gains', 'stress_biases', 'te_clean', 'te_true_rate0', 'te_hist_true', 'te_hold_true']
[0].gt.taus (2,) float64 [0.566 4.25 ]
[0].gt.theta (7,) float64 [ 0.68  0.55  0.14  0.16 -0.13  0.12 -0.09]
[0].gt.vp_theta (2,) float64 [0.26 0.22]
[0].gt.stress_gains (4,) float64 [0.79 1.31 0.85 1.24]
[0].gt.stress_biases (4,) float64 [-0.14  0.16  0.1  -0.13]
[0].gt.te_clean (72, 10) float64 [-0.80018704 -1.81493196 -2.29830392 -2.56693708 -2.7434487  -2.87582718
 -2.98357606 -3.07503803]
[0].gt.te_true_rate0 (72,) float64 [-4.40087488  2.68666935  0.56796129  2.18560354 -0.43275744  0.97825611
 -0.47898115  1.57981321]
[0].gt.te_hist_true (72, 18) float64 [ 1.40693381 -0.51481438  1.76921348  3.26659112  3.20091926  3.75558886
  3.30649425  3.44779333]
[0].gt.te_hold_true (72,) float64 [-1.732648  1.630115  0.232425  1.372623 -0.092051 -0.007354 -0.828837
  0.483612]
```

### Turn 2 — `execute_in_remote_docker` (`19353 ms` LLM)
- **Rationale**: Inspect the prefix sweep structure and check lab–stratum coverage before implementing calibration and imputation. Check training observation availability to determine how to fit and validate curves and initial rates separately; do not use public ground truth for fitting.
- **Alternatives Weighed**: `Estimate all four calibration lines from the current sweep` (rejected: The current sweep contains only two points total; channels with fewer than two points require the permitted prefix calibration history.)

```bash
cd /holdout_task && sed -n '1,260p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_03_solver.py && head -85 /tmp/hegel_continual_archive/archive_calibration_history.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference panel imputation' > /tmp/m97_skills.txt; grep -Ei 'viscoelastic|imput' /tmp/m97_skills.txt | cut -c1-330 | head -18; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
D=pickle.load(open('public_alpha_m97_c02_batch_04.pkl','rb'))[0]['dataset']; print('cal',D['calibration_load_cells']); print('time',D['eval_times']);
for s in ('train','test'):
 for k,v in D[s].items():
  if k=='ref_coupons' and isinstance(v,dict): print(s,k,[(a,np.asarray(b).shape) for a,b in v.items()]);continue
  a=np.asarray(v); print(s,k,a.shape,a.dtype, 'nonfinite',np.sum(~np.isfinite(a)) if a.dtype.kind in 'fiu' else '', 'preview',a.ravel()[:4]);
 g=D[s];print('cells',np.unique(np.c_[g['lab_id'],g['stratum']],axis=0,return_counts=True))
PY
```
- **Container Exit Code**: `0` (`624 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.673079,
      0.633995,
      2.133025,
      0.157003,
      -0.816267,
      -1.031715,
      -0.029779,
cal {'lab_id': array([1, 1]), 'cmd_stress': array([-1.3,  1.4]), 'true_stress': array([-1.54314271,  1.99396734])}
time [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ]
train lab_id (180,) int64 nonfinite 0 preview [0 1 2 3]
train stratum (180,) int64 nonfinite 0 preview [0 1 2 1]
train coupon_features_raw (180, 4) float64 nonfinite 22 preview [ 0.20927482 -1.32888332 -2.08882567 -2.33904064]
train ref_coupons (180, 32) float64 nonfinite 238 preview [-0.41104459 -2.36046946 -0.37993796 -1.05563557]
train past_stress_cmd (180, 18) float64 nonfinite 0 preview [ 0.08166482 -1.86487808 -0.35468902  4.51697596]
train stress_hold_cmd (180,) float64 nonfinite 0 preview [ 1.68720759  0.86210229 -1.27090471 -0.12899194]
train strain_0 (180,) float64 nonfinite 0 preview [-0.044554  0.013663 -0.266441 -0.603854]
train aux_chamber_telemetry (180, 3) float64 nonfinite 0 preview [ 0.206079  0.162665 -0.531659 -0.026741]
train strain_curves_obs (180, 10) float64 nonfinite 0 preview [0.73510347 1.4778566  1.82512958 1.97045598]
train instant_rate_obs (180,) float64 nonfinite 0 preview [ 3.32531914  2.48433019 -1.89117201 -0.26488422]
cells (array([[0, 0],
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
       [3, 2]]), array([24, 16,  7, 11, 16, 10, 15,  7, 18, 15, 21, 20]))
test lab_id (72,) int64 nonfinite 0 preview [0 1 2 3]
test stratum (72,) int64 nonfinite 0 preview [0 2 2 0]
test coupon_features_raw (72, 4) float64 nonfinite 9 preview [-0.87994183  0.26379751         nan -0.88123122]
test ref_coupons (72, 32) float64 nonfinite 99 preview [ 0.77565306  0.18284838 -0.88660253 -1.23250686]
test past_stress_cmd (72, 18) float64 nonfinite 0 preview [ 1.95814406 -0.47444859  2.41672592  4.31214066]
test stress_hold_cmd (72,) float64 nonfinite 0 preview [-2.01601013  1.12222519  0.15579412  1.21179274]
test strain_0 (72,) float64 nonfinite 0 preview [ 0.247488  0.765664 -0.077795 -0.259985]
test aux_chamber_telemetry (72, 3) float64 nonfinite 0 preview [ 0.469126 -0.280347  0.350479 -0.643394]
cells (array([[0, 0],
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
       [3, 2]]), array([13,  7,  2,  3,  7,  4,  4,  4,  8, 10,  5,  5]))
```

### Turn 3 — `execute_in_remote_docker` (`16788 ms` LLM)
- **Rationale**: The previous output truncated the preprocessing code. Read the exact 32-column panel mapping and check archive line residuals to determine whether missing channels can use pooled clean rows or need sweep-specific selection.
- **Alternatives Weighed**: `Pool all archived calibration measurements immediately` (rejected: First check whether prefix sweeps agree and whether EMI points would distort a pooled fit.)

```bash
sed -n '51,230p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_03_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -45; python3 - <<'PY'
import json,numpy as np
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f: h=json.load(f)
for a in h:
 print(a['episode_short_name'],end=': ')
 for j in range(4):
  m=np.asarray(a['lab_id'])==j;x=np.asarray(a['cmd_stress'])[m];y=np.asarray(a['true_stress'])[m]
  if len(x)>1 and np.ptp(x)>0:
   p=np.polyfit(x,y,1);print(j,len(x),np.round(p,5),round(np.max(np.abs(y-np.polyval(p,x))),4),end='; ')
  else: print(j,len(x),end='; ')
 print()
PY
```
- **Container Exit Code**: `0` (`1328 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
econfounding.align_control_panel",
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
alpha_m97_c02_s00: 0 10 [0.99998 0.     ] 0.0002; 1 10 [9.9999e-01 4.0000e-05] 0.0001; 2 10 [1.00001e+00 3.00000e-05] 0.0001; 3 10 [ 1.00001e+00 -3.00000e-05] 0.0001; 
alpha_m97_c02_batch_01: 0 10 [ 0.79005 -0.13999] 0.0002; 1 10 [1.30996 0.16006] 0.0002; 2 10 [0.84996 0.10004] 0.0001; 3 10 [ 1.24    -0.12996] 0.0001; 
alpha_m97_c02_batch_02: 0 10 [ 0.79001 -0.14   ] 0.0002; 1 10 [1.30998 0.15996] 0.0002; 2 10 [0.84999 0.09998] 0.0002; 3 10 [ 1.24004 -0.12998] 0.0002; 
alpha_m97_c02_batch_03: 0 10 [ 0.78997 -0.13996] 0.0003; 1 10 [1.30997 0.15996] 0.0001; 2 10 [0.84999 0.10002] 0.0002; 3 10 [ 1.23997 -0.13004] 0.0002;
```

### Turn 4 — `execute_in_remote_docker` (`52210 ms` LLM)
- **Rationale**: Implement the stated response and its analytic initial rate with shared weights, fitting only training observations. Leave-one-lab-out errors by horizon, activation, and missingness will show whether any slice warrants a change to calibration, alignment, or observation weighting.
- **Alternatives Weighed**: `Use the earlier control-panel skill` (rejected: It does not implement the specified median-of-column normalization and stratum-balanced imputation; the archived solver already has those operations.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
prior = Path('/tmp/hegel_continual_archive/alpha_m97_c02_batch_03_solver.py').read_text()
shared = prior[prior.index('import json'):prior.index('def _basis(')]
Path('alpha_m97_c02_batch_04_solver.py').write_text('"""Calibrated two-branch creep with a strain-crystallization plateau."""\n' + shared)
PY
cat >> alpha_m97_c02_batch_04_solver.py <<'PY'
def _design(log_tau, past, hold, features, dt, times, act):
    tau = np.exp(log_tau)
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    z = past @ (np.exp(-age[:, None] / tau) * -np.expm1(-dt[:, None] / tau))
    gap = hold[:, None] - z
    step = -np.expm1(-times[:, None] / tau)
    phi = np.log1p(times)
    curve = np.concatenate((gap[:, None, :] * step,
                            features[:, None, :] * phi[None, :, None],
                            (act * hold)[:, None, None] * step[None, :, :1],
                            (act * z[:, 0])[:, None, None] * phi[None, :, None]), axis=2)
    rate = np.column_stack((gap / tau, features, act * hold / tau[0], act * z[:, 0]))
    return curve, rate


def _activation(dataset, past, dt):
    if dataset['shear_banding_index'] > 0.50:
        return np.zeros(len(past))
    excess = np.maximum(np.abs(past) - dataset['crit_stress'], 0.)
    return np.tanh(excess**2 @ dt)


def _fit(past, hold, features, dt, times, act, curves, rates, rate_weight=1.):
    curve_ok, rate_ok = np.isfinite(curves), np.isfinite(rates)
    target = np.r_[curves[curve_ok], rate_weight * rates[rate_ok]]

    def profile(log_tau, weights_only=False):
        curve, rate = _design(log_tau, past, hold, features, dt, times, act)
        design = np.concatenate((curve[curve_ok], rate_weight * rate[rate_ok]))
        weights = np.linalg.lstsq(design, target, rcond=None)[0]
        return weights if weights_only else design @ weights - target

    best = min((least_squares(profile, np.log(start),
                              bounds=(np.log([.02, .04]), np.log([40., 100.])),
                              diff_step=2e-5, max_nfev=100)
                for start in ((.5, 4.), (.2, 2.), (1., 10.))), key=lambda fit: fit.cost)
    log_tau = best.x
    # The crystallization terms use the fast branch. If the gate is closed,
    # the two Prony branches are interchangeable and are reported fast first.
    if not np.any(act):
        log_tau = np.sort(log_tau)
    return log_tau, profile(log_tau, weights_only=True)


def solve_alpha_m97_c02_batch_04(dataset: dict) -> dict:
    past, hold, features, strain0 = _prepare(dataset)
    dt, times = (np.asarray(dataset[k], dtype=float) for k in ('past_dt', 'eval_times'))
    act = _activation(dataset, past, dt)
    train = dataset['train']
    n = len(train['lab_id'])
    curves = np.asarray(train['strain_curves_obs']) - strain0[:n, None]
    rates = np.asarray(train['instant_rate_obs'])
    log_tau, weights = _fit(past[:n], hold[:n], features[:n], dt, times, act[:n], curves, rates)
    curve, rate = _design(log_tau, past[n:], hold[n:], features[n:], dt, times, act[n:])
    return {'estimated_taus': np.exp(log_tau), 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
cat > /tmp/m97_b04_cv.py <<'PY'
import importlib.util
import pickle
import numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c02_batch_04_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open(root+'public_alpha_m97_c02_batch_04.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
p, h, x, s0 = m._prepare(d)
tr = d['train']; n = len(tr['lab_id']); dt = d['past_dt']; t = d['eval_times']
p,h,x,s0 = p[:n],h[:n],x[:n],s0[:n]
a = m._activation(d,p,dt); y = tr['strain_curves_obs']-s0[:,None]; r = tr['instant_rate_obs']
def rms(x): return np.sqrt(np.mean(x*x))
def report(v,w): return [round(rms(v),7),round(rms(w),7)]
v, w = np.zeros_like(y), np.zeros_like(r)
for lab in np.unique(tr['lab_id']):
    q = tr['lab_id']==lab; f = ~q
    lt,b=m._fit(p[f],h[f],x[f],dt,t,a[f],y[f],r[f]); cy,cr=m._design(lt,p[q],h[q],x[q],dt,t,a[q]); v[q],w[q]=cy@b-y[q],cr@b-r[q]
    print('held lab',lab,'curve/rate RMSE',report(v[q],w[q]),'taus',np.exp(lt),'active',sum(a[q]>0))
for j in np.unique(tr['stratum']):
    q=tr['stratum']==j;print('OOF stratum',j,report(v[q],w[q]))
for label,q in [('zero',a==0),('nonzero',a>0),('strong',a>.8),('imputed',np.isnan(tr['coupon_features_raw']).any(axis=1))]:
    if q.any(): print('OOF',label,sum(q),report(v[q],w[q]))
lt,b=m._fit(p,h,x,dt,t,a,y,r);cy,cr=m._design(lt,p,h,x,dt,t,a)
print('full fit',np.exp(lt),b,'curve/rate RMSE',report(cy@b-y,cr@b-r)); print('OOF each horizon',np.sqrt(np.mean(v*v,axis=0)))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m97_b04_cv.py
```
- **Container Exit Code**: `0` (`1100 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_04_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
held lab 0 curve/rate RMSE [1.600401, 0.0832705] taus [0.59901569 4.40883744] active 26
held lab 1 curve/rate RMSE [0.4875273, 0.1073134] taus [0.577759   4.62226252] active 24
held lab 2 curve/rate RMSE [0.833015, 0.0875261] taus [0.59481587 4.77044517] active 33
held lab 3 curve/rate RMSE [1.3101061, 0.0834966] taus [0.55911643 3.80265499] active 37
OOF stratum 0 [1.3949708, 0.0848593]
OOF stratum 1 [0.5568303, 0.0947446]
OOF stratum 2 [1.4010281, 0.0896637]
OOF zero 60 [1.4211053, 0.0739303]
OOF nonzero 120 [1.0483157, 0.096648]
OOF strong 120 [1.0483157, 0.096648]
OOF imputed 20 [2.460504, 0.1298212]
full fit [0.58160237 4.35165384] [ 0.70203126  0.50847697  0.12398546  0.14697882 -0.1313902   0.1468359
 -0.07075309  0.30723354  0.21496484] curve/rate RMSE [1.1800343, 0.0820347]
OOF each horizon [0.60833645 0.24809518 0.11059102 1.20432607 3.31981326 0.41426044
 0.30854934 0.37573646 0.82841411 0.22036655]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=594a03a9b04f5b3d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_594a03a9 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`45652 ms` LLM)
- **Rationale**: The strain RMSE varies sharply by horizon while initial-rate errors are much smaller. Compare a robust challenger on the same held-out labs, reporting all-observation RMSE, median error, and shared-core RMSE separately. If the large errors are widespread rather than concentrated, inspect the response basis and alignment instead of promoting robust fitting.
- **Alternatives Weighed**: `Add a horizon-specific response term` (rejected: One large horizon RMSE does not distinguish a missing response term from a few contaminated readings. Compare robust and ordinary fits before changing the physical model.)

```bash
set -e
cp /holdout_task/alpha_m97_c02_batch_04_solver.py /tmp/m97_b04_robust.py
cat >> /tmp/m97_b04_robust.py <<'PY'

_fit_linear = _fit


def _fit(past, hold, features, dt, times, act, curves, rates):
    log_tau, weights = _fit_linear(past, hold, features, dt, times, act, curves, rates)
    curve_ok, rate_ok = np.isfinite(curves), np.isfinite(rates)

    def residual(p):
        curve, rate = _design(p[:2], past, hold, features, dt, times, act)
        return ((np.einsum('nkp,p->nk', curve, p[2:]) - curves)[curve_ok],
                (np.einsum('np,p->n', rate, p[2:]) - rates)[rate_ok])

    def mad(r):
        return max(1e-7, 1.4826 * np.median(np.abs(r - np.median(r)))) if r.size else 1.

    start = np.r_[log_tau, weights]
    bounds = (np.r_[np.log([.02, .04]), np.full(9, -np.inf)],
              np.r_[np.log([40., 100.]), np.full(9, np.inf)])
    for _ in range(2):
        scales = [mad(r) for r in residual(start)]
        def normalized(p):
            return np.concatenate([r / s for r, s in zip(residual(p), scales)])
        fit = least_squares(normalized, start, bounds=bounds, loss='soft_l1',
                            f_scale=1.5, x_scale='jac', max_nfev=120)
        start = fit.x
    return start[:2], start[2:]
PY
cat > /tmp/m97_b04_compare.py <<'PY'
exec(open('/tmp/m97_b04_cv.py').read().split('v, w =')[0])
spec = importlib.util.spec_from_file_location('robust','/tmp/m97_b04_robust.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
fits = [m._fit,b._fit]
res = [[np.zeros_like(y),np.zeros_like(r)] for _ in fits]
core_y = np.zeros_like(y,dtype=bool); core_r = np.zeros_like(r,dtype=bool)
for lab in np.unique(tr['lab_id']):
    q = tr['lab_id']==lab; f = ~q
    for i,fit in enumerate(fits):
        lt,w=fit(p[f],h[f],x[f],dt,t,a[f],y[f],r[f]); cy,cr=m._design(lt,p[q],h[q],x[q],dt,t,a[q]); res[i][0][q]=cy@w-y[q];res[i][1][q]=cr@w-r[q]
    cy,cr=m._design(lt,p[f],h[f],x[f],dt,t,a[f]); ey,er=cy@w-y[f],cr@w-r[f]
    def fence(e): return 5*1.4826*np.median(np.abs(e-np.median(e)))
    core_y[q]=np.abs(res[1][0][q]) < fence(ey); core_r[q]=np.abs(res[1][1][q]) < fence(er)
    print('fold',lab,'robust taus',np.exp(lt),'fences',fence(ey),fence(er),'curve outside',sum(~core_y[q].ravel()))
for label,q in [(f'lab {u}',tr['lab_id']==u) for u in np.unique(tr['lab_id'])]+[(f'stratum {u}',tr['stratum']==u) for u in np.unique(tr['stratum'])]+[('zero damage',a==0),('positive damage',a>0),('imputed',np.isnan(tr['coupon_features_raw']).any(axis=1)),('all',np.ones(n,dtype=bool))]:
    print(label,'baseline -> robust: raw RMSE',[report(v[q],w[q]) for v,w in res], 'median AE',[[round(np.median(np.abs(v[q])),6),round(np.median(np.abs(w[q])),6)] for v,w in res], 'same core RMSE',[report(v[q][core_y[q]],w[q][core_r[q]]) for v,w in res])
for fit in fits:
    lt,w=fit(p,h,x,dt,t,a,y,r); print('full',np.exp(lt),w)
print('curve residual abs percentiles', [np.percentile(np.abs(v),[50,90,95,98,99,100]).round(5) for v,w in res]); print('robust core horizon RMSE',[rms(res[1][0][:,i][core_y[:,i]]) for i in range(len(t))])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m97_b04_compare.py
```
- **Container Exit Code**: `0` (`878 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
15244, 0.012813]] same core RMSE [[0.0748563, 0.0734669], [0.0202582, 0.0210362]]
stratum 0 baseline -> robust: raw RMSE [[1.3949708, 0.0848593], [1.391286, 0.0518073]] median AE [[0.06837, 0.058525], [0.013996, 0.014181]] same core RMSE [[0.0850167, 0.0787763], [0.0208609, 0.0218678]]
stratum 1 baseline -> robust: raw RMSE [[0.5568303, 0.0947446], [0.5503463, 0.0557041]] median AE [[0.054721, 0.061179], [0.014493, 0.014478]] same core RMSE [[0.0772039, 0.0840009], [0.0203654, 0.0199701]]
stratum 2 baseline -> robust: raw RMSE [[1.4010281, 0.0896637], [1.398382, 0.0545532]] median AE [[0.057321, 0.06195], [0.013861, 0.016308]] same core RMSE [[0.0806915, 0.0825866], [0.0208718, 0.0257824]]
zero damage baseline -> robust: raw RMSE [[1.4211053, 0.0739303], [1.4186473, 0.0495989]] median AE [[0.049102, 0.052322], [0.013845, 0.013882]] same core RMSE [[0.0747281, 0.064947], [0.0215316, 0.0180088]]
positive damage baseline -> robust: raw RMSE [[1.0483157, 0.096648], [1.0439876, 0.0560295]] median AE [[0.06604, 0.073344], [0.014309, 0.015074]] same core RMSE [[0.0841499, 0.088911], [0.0202622, 0.0245057]]
imputed baseline -> robust: raw RMSE [[2.460504, 0.1298212], [2.460842, 0.1525952]] median AE [[0.101391, 0.076257], [0.102859, 0.080779]] same core RMSE [[0.0653078, 0.0520938], [0.0469889, 0.0507515]]
all baseline -> robust: raw RMSE [[1.1856745, 0.0897169], [1.1821422, 0.0539711]] median AE [[0.060339, 0.061179], [0.014135, 0.014823]] same core RMSE [[0.0811137, 0.0817077], [0.0206964, 0.022549]]
full [0.58160237 4.35165384] [ 0.70203126  0.50847697  0.12398546  0.14697882 -0.1313902   0.1468359
 -0.07075309  0.30723354  0.21496484]
full [0.56477121 4.17961853] [ 0.675888    0.54745834  0.13919405  0.15648273 -0.12703362  0.11751106
 -0.08758604  0.26422605  0.21732936]
curve residual abs percentiles [array([ 0.06034,  0.18819,  0.3341 ,  0.71305,  1.10001, 34.27227]), array([1.414000e-02, 1.258000e-01, 3.545200e-01, 6.334100e-01,
       1.119260e+00, 3.424426e+01])]
robust core horizon RMSE [0.019660726483776758, 0.018261389045744654, 0.021504627563151208, 0.021781567365170434, 0.018921780698620744, 0.02037120417737907, 0.019169680079512273, 0.02364264902334288, 0.020802786668787362, 0.022407041600174114]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=594a03a9b04f5b3d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_594a03a9 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`37794 ms` LLM)
- **Rationale**: Compare Cauchy reweighting and excluding imputed rows from fitting, while still predicting those rows with the required cell-balanced imputation. The held-out lab, damage, and missingness slices will determine whether either change improves the soft-L1 candidate without altering the specified response law.
- **Alternatives Weighed**: `Promote soft-L1 unchanged` (rejected: It improves typical errors across labs, but the imputed-row rate regression warrants one targeted comparison before selecting the final fit.); `Fit horizon-specific corrections` (rejected: The same physical basis already achieves core horizon RMSE below 0.024; a horizon correction would risk fitting isolated large errors.)

```bash
set -e
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust residual scale IRLS nonlinear least squares' > /tmp/m97_b04_robust_skills.json
python3 - <<'PY'
import json
p='/tmp/m97_b04_robust_skills.json'
try:
 d=json.load(open(p)); print(str(d)[:1700])
except ValueError:
 print(open(p).read()[:1700])
PY
cat > /tmp/m97_b04_cauchy_compare.py <<'PY'
exec(open('/tmp/m97_b04_cv.py').read().split('v, w =')[0])
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('robust','/tmp/m97_b04_robust.py')
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
missing=np.isnan(tr['coupon_features_raw']).any(axis=1)
def refit(f, initial, loss, omit_imputed=False):
    if omit_imputed: f=f & ~missing
    def residual(pv):
        cy,cr=m._design(pv[:2],p[f],h[f],x[f],dt,t,a[f])
        return np.r_[(cy@pv[2:]-y[f]).ravel(),cr@pv[2:]-r[f]]
    e=residual(initial); cut=f.sum()*len(t)
    def mad(v): return max(1.e-7,1.4826*np.median(np.abs(v-np.median(v))))
    scale=np.r_[np.full(cut,mad(e[:cut])),np.full(f.sum(),mad(e[cut:]))]
    fit=least_squares(lambda pv: residual(pv)/scale,initial,loss=loss,f_scale=2.5,x_scale='jac',max_nfev=100,bounds=(np.r_[np.log([.02,.04]),np.full(9,-np.inf)],np.r_[np.log([40.,100.]),np.full(9,np.inf)]))
    return fit.x
names=['soft_l1','cauchy','cauchy_observed_features']
res={j:[np.zeros_like(y),np.zeros_like(r)] for j in names}
for lab in np.unique(tr['lab_id']):
    q=tr['lab_id']==lab; f=~q
    lt,w=b._fit(p[f],h[f],x[f],dt,t,a[f],y[f],r[f]); start=np.r_[lt,w]
    for j,pv in zip(names,(start,refit(f,start,'cauchy'),refit(f,start,'cauchy',True))):
        cy,cr=m._design(pv[:2],p[q],h[q],x[q],dt,t,a[q]);res[j][0][q]=cy@pv[2:]-y[q];res[j][1][q]=cr@pv[2:]-r[q]
    e,z=res['cauchy']; print('held lab',lab,'curve errors > 1 per horizon',np.sum(np.abs(e[q])>1,axis=0),'rate errors > .1',np.sum(np.abs(z[q])>.1))
for j,(e,z) in res.items():
    print(j)
    for name,q in [(f'lab {k}',tr['lab_id']==k) for k in range(4)]+[('imputed',missing),('observed features',~missing),('zero damage',a==0),('positive damage',a>0)]:
        print(name,'raw rmse',report(e[q],z[q]),'median ae',np.round([np.median(np.abs(e[q])),np.median(np.abs(z[q]))],6),'90% ae',np.round([np.quantile(np.abs(e[q]),.9),np.quantile(np.abs(z[q]),.9)],6))
    print('maximum horizon median ae',np.max(np.median(np.abs(e),axis=0)))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m97_b04_cauchy_compare.py
```
- **Container Exit Code**: `0` (`848 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
3 raw rmse [1.3061352, 0.0672333] median ae [0.015244 0.012813] 90% ae [0.246755 0.063323]
imputed raw rmse [2.460842, 0.1525952] median ae [0.102859 0.080779] 90% ae [0.432699 0.251147]
observed features raw rmse [0.9028703, 0.0191398] median ae [0.012714 0.013477] 90% ae [0.035853 0.031431]
zero damage raw rmse [1.4186473, 0.0495989] median ae [0.013845 0.013882] 90% ae [0.134688 0.03029 ]
positive damage raw rmse [1.0439876, 0.0560295] median ae [0.014309 0.015074] 90% ae [0.120139 0.046289]
maximum horizon median ae 0.016343938142367564
cauchy
lab 0 raw rmse [1.5971538, 0.0626713] median ae [0.015311 0.015245] 90% ae [0.159254 0.031187]
lab 1 raw rmse [0.4794906, 0.037802] median ae [0.011816 0.018204] 90% ae [0.039025 0.030684]
lab 2 raw rmse [0.8310213, 0.0275718] median ae [0.012576 0.015076] 90% ae [0.08641 0.0352 ]
lab 3 raw rmse [1.3061754, 0.0678] median ae [0.015384 0.012608] 90% ae [0.24738  0.062559]
imputed raw rmse [2.4609383, 0.1536274] median ae [0.103791 0.084255] 90% ae [0.431662 0.252799]
observed features raw rmse [0.9029113, 0.0182479] median ae [0.012313 0.012864] 90% ae [0.035873 0.029973]
zero damage raw rmse [1.4186729, 0.0496671] median ae [0.013919 0.012819] 90% ae [0.134898 0.030162]
positive damage raw rmse [1.0440554, 0.0560725] median ae [0.013685 0.015818] 90% ae [0.121372 0.043878]
maximum horizon median ae 0.015606685327253678
cauchy_observed_features
lab 0 raw rmse [1.5971458, 0.0627553] median ae [0.015472 0.015446] 90% ae [0.159495 0.028187]
lab 1 raw rmse [0.479422, 0.0375752] median ae [0.011497 0.017829] 90% ae [0.03813  0.029776]
lab 2 raw rmse [0.831083, 0.0273651] median ae [0.012861 0.014027] 90% ae [0.086791 0.034652]
lab 3 raw rmse [1.3062193, 0.0677018] median ae [0.015173 0.01327 ] 90% ae [0.246808 0.062523]
imputed raw rmse [2.4609027, 0.1537736] median ae [0.103622 0.085587] 90% ae [0.432301 0.252193]
observed features raw rmse [0.9029472, 0.017861] median ae [0.012297 0.013552] 90% ae [0.035359 0.027607]
zero damage raw rmse [1.4186368, 0.0496445] median ae [0.013991 0.014048] 90% ae [0.135201 0.029445]
positive damage raw rmse [1.0441074, 0.0559832] median ae [0.013491 0.016001] 90% ae [0.122341 0.042902]
maximum horizon median ae 0.01574394944268742

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=594a03a9b04f5b3d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_594a03a9 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`38625 ms` LLM)
- **Rationale**: Promote soft-L1 based on the held-lab comparison. Verify the exact deliverable in an isolated Python process, then use public reference outputs only for evaluation to separate prediction error from contaminated observations. Calibration, telemetry invariance, and the strain derivative check can catch independent contract failures.
- **Alternatives Weighed**: `Use Cauchy loss or exclude imputed rows from fitting` (rejected: Both slightly improve rate RMSE on complete rows but worsen it on imputed rows; neither improves the worst held-lab rate RMSE over soft-L1.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
src = Path('/tmp/m97_b04_robust.py').read_text()
src = src.replace('def _fit(past, hold, features, dt, times, act, curves, rates, rate_weight=1.):', 'def _profile_fit(past, hold, features, dt, times, act, curves, rates, rate_weight=1.):', 1)
a = src.index('def solve_alpha_m97_c02_batch_04')
b = src.index('_fit_linear = _fit', a)
solve = src[a:b].rstrip()
robust = src[b:].split('def _fit(', 1)[1].replace('_fit_linear(', '_profile_fit(', 1)
src = src[:a] + 'def _fit(' + robust.rstrip() + '\n\n\n' + solve + '\n'
Path('alpha_m97_c02_batch_04_solver.py').write_text(src)
PY
sed -n '1,56p' alpha_m97_c02_batch_04_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m97_b04_cv.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','alpha_m97_c02_batch_04_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('public_alpha_m97_c02_batch_04.pkl','rb') as f: items=pickle.load(f)
for i,item in enumerate(items):
    d,g=item['dataset'],item['gt']; out=m.solve_alpha_m97_c02_batch_04(d)
    print('public item',i,'gt fields',list(g),'estimated taus',out['estimated_taus'])
    cal=m._calibrate(d['calibration_load_cells']); print('max calibration gain/bias abs errors',np.max(np.abs(cal[:,0]-g['stress_gains'])),np.max(np.abs(cal[:,1]-g['stress_biases'])))
    a=m._activation(d,m._prepare(d)[0],d['past_dt'])[-len(d['test']['lab_id']):]
    te=d['test']; es=out['predicted_strain']-g['te_clean']; er=out['initial_rate_pred']-g['te_true_rate0']; missing=np.isnan(te['coupon_features_raw']).any(axis=1)
    slices=[('all',np.ones(len(a),dtype=bool)),('zero damage',a==0),('positive damage',a>0),('imputed',missing),('observed',~missing)] + [(f'lab {lab}',te['lab_id']==lab) for lab in range(4)]
    for label,q in slices:
        if q.any(): print(label,int(q.sum()),'clean RMSE strain/rate',np.sqrt(np.mean(es[q]**2)),np.sqrt(np.mean(er[q]**2)))
    for key in ('train','test'):
        d[key]['aux_chamber_telemetry']=np.full_like(d[key]['aux_chamber_telemetry'],123.)
    repeat=m.solve_alpha_m97_c02_batch_04(d)
    assert all(np.array_equal(out[k],repeat[k]) for k in out), 'uncoupled telemetry changed prediction'
    p,h,x,_=m._prepare(d); dt=d['past_dt']; a=m._activation(d,p,dt); lt=np.log(out['estimated_taus']); b=np.r_[out['hereditary_weights'],out['regime_mode_weights']]
    t=np.array([0.,1e-7]); c,r=m._design(lt,p,h,x,dt,t,a)
    assert np.allclose(c[:,0]@b,0.,atol=1e-14)
    assert np.allclose((c[:,1]@b)/t[1],r@b,rtol=2e-6,atol=2e-7), 'initial rate inconsistent with curve derivative'
PY
```
- **Container Exit Code**: `0` (`1334 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_04_solver.py']`
- **Stdout (tail)**:
```text
e
        if np.count_nonzero(clean) >= 4 and np.unique(x[clean]).size >= 2:
            result[channel] = np.polyfit(x[clean], y[clean], 1)
        elif archive is not None:
            result[channel] = archive
        elif np.unique(x[clean]).size >= 2:
            result[channel] = np.polyfit(x[clean], y[clean], 1)
        else:
{"schema_valid": true, "errors": []}
held lab 0 curve/rate RMSE [1.5971037, 0.0626006] taus [0.56475723 4.22745368] active 26
held lab 1 curve/rate RMSE [0.4796023, 0.0384962] taus [0.56428466 4.11752766] active 24
held lab 2 curve/rate RMSE [0.8308423, 0.0283567] taus [0.56444573 4.1798863 ] active 33
held lab 3 curve/rate RMSE [1.3061352, 0.0672333] taus [0.5655164  4.19964313] active 37
OOF stratum 0 [1.391286, 0.0518073]
OOF stratum 1 [0.5503463, 0.0557041]
OOF stratum 2 [1.398382, 0.0545532]
OOF zero 60 [1.4186473, 0.0495989]
OOF nonzero 120 [1.0439876, 0.0560295]
OOF strong 120 [1.0439876, 0.0560295]
OOF imputed 20 [2.460842, 0.1525952]
full fit [0.56477121 4.17961853] [ 0.675888    0.54745834  0.13919405  0.15648273 -0.12703362  0.11751106
 -0.08758604  0.26422605  0.21732936] curve/rate RMSE [1.1820486, 0.0531655]
OOF each horizon [0.60922449 0.24535191 0.09588135 1.20380871 3.31541526 0.41189024
 0.28545981 0.37443889 0.8149065  0.19365951]
public item 0 gt fields ['taus', 'theta', 'vp_theta', 'stress_gains', 'stress_biases', 'te_clean', 'te_true_rate0', 'te_hist_true', 'te_hold_true'] estimated taus [0.56477121 4.17961853]
max calibration gain/bias abs errors 4.076033455180372e-05 8.972029976170304e-05
all 72 clean RMSE strain/rate 0.050620972326458895 0.03864796115853858
zero damage 24 clean RMSE strain/rate 0.056650743473507424 0.043365428127579785
positive damage 48 clean RMSE strain/rate 0.04731882174001804 0.03605852429348317
imputed 9 clean RMSE strain/rate 0.14235326191911776 0.10789677833654356
observed 63 clean RMSE strain/rate 0.005799156776093747 0.0066289886999671005
lab 0 22 clean RMSE strain/rate 0.053016224957910356 0.03907103975139475
lab 1 14 clean RMSE strain/rate 0.06412216076561494 0.04972614513831616
lab 2 16 clean RMSE strain/rate 0.061889305078556375 0.047592311752740125
lab 3 20 clean RMSE strain/rate 0.013811632212241723 0.012453421882537532

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ca9bd5bb4a5dfa7d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ca9bd5bb (directory).
```

### Turn 8 — `submit_completed_task` (`15632 ms` LLM)
- **Rationale**: The saved solver passed isolated schema validation, held-out lab and regime checks, and independent tests for calibration accuracy, telemetry invariance, and consistency between the strain curve and its initial derivative.
- **Alternatives Weighed**: `Replace soft-L1 with Cauchy fitting` (rejected: Cauchy slightly improved complete-row rate RMSE but worsened imputed-row and worst held-lab rate RMSE.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_04_solver.py': 'FILE:8181'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_04_solver.py`. The public schema validator passed. Against the public clean reference, strain and initial-rate RMSE were 0.0506 and 0.0386 overall, and 0.1424 and 0.1079 on rows with imputed features.`

### Turn 9 — `execute_in_remote_docker` (`36298 ms` LLM)
- **Rationale**: Compare the specified basis with plausible omissions on held-out labs. A consistent advantage for an alternative would trigger a review of the regime interpretation; disagreement with the direct memory recurrence or analytic initial rate would identify an implementation error. Check the measured distance to the damage boundary to assess whether calibration uncertainty can change activation.
- **Alternatives Weighed**: `Test relay modes, classification thresholds, or unrelated drift operators` (rejected: This is a single-batch continuous rheology regression with no relay state or class labels. The relevant checks are held-lab error within damage strata and exact recurrence at the shear gate.)

```bash
set -e
cd /holdout_task
sed -n '57,105p' alpha_m97_c02_batch_04_solver.py
cat >/tmp/m97_b04_gate_audit.py <<'PY'
exec(open('/tmp/m97_b04_cv.py').read().split('v, w =')[0])
from scipy.optimize import least_squares
print('gate settings',d['shear_banding_index'],d['crit_stress'])
print('in-batch calibration',d['calibration_load_cells'])
cp,ch,_,_=m._prepare(d)
excess=np.max(np.abs(cp),axis=1)-d['crit_stress']
print('minimum absolute history distance to damage boundary',np.min(np.abs(excess)))
actual=m._design

def alternative(kind):
    def design(lt,p,h,x,dt,t,a):
        c,r=actual(lt,p,h,x,dt,t,a)
        if kind=='linear':
            c[:,:,2:7]=x[:,None,:]*t[None,:,None]
        elif kind=='linear_zero_damage':
            zero=a==0
            c[zero,:,2:7]=x[zero,None,:]*t[None,:,None]
        elif kind=='no_crystallization':
            c[:,:,7:]=0; r[:,7:]=0
        return c,r
    return design

try:
    for kind in ('specified','linear','linear_zero_damage','no_crystallization'):
        m._design=actual if kind=='specified' else alternative(kind)
        stats=[]
        for lab in np.unique(tr['lab_id']):
            q=tr['lab_id']==lab; f=~q
            lt,b=m._fit(p[f],h[f],x[f],dt,t,a[f],y[f],r[f]); c,g=m._design(lt,p[q],h[q],x[q],dt,t,a[q])
            for sub in (a[q]==0,a[q]>0):
                if sub.any(): stats.append((np.median(np.abs((c@b-y[q])[sub])),np.median(np.abs((g@b-r[q])[sub]))))
        print(kind,'worst held-lab/damage median AE strain/rate',np.round(np.max(stats,axis=0),6))
finally:
    m._design=actual

out=m.solve_alpha_m97_c02_batch_04(d)
lt=np.log(out['estimated_taus']); b=np.r_[out['hereditary_weights'],out['regime_mode_weights']]; tau=np.exp(lt)
for shear in (.50,np.nextafter(.50,1.)):
    a0=m._activation(dict(d,shear_banding_index=shear),cp,dt)
    z=np.zeros((len(cp),2))
    for s,step in enumerate(dt):
        alpha=np.exp(-step/tau);z=alpha*z+(1-alpha)*cp[:,s,None]
    _,_,xx,_=m._prepare(d)
    gap=ch[:,None]-z
    drift=xx@b[2:7]+b[8]*a0*z[:,0]
    prony=gap*b[:2]; prony[:,0]+=b[7]*a0*ch
    expected=prony@(-np.expm1(-t[:,None]/tau)).T+drift[:,None]*np.log1p(t)
    c,g=m._design(lt,cp,ch,xx,dt,t,a0)
    assert np.allclose(c@b,expected,rtol=1e-12,atol=1e-12)
    assert np.allclose(g@b,prony@(1/tau)+drift,rtol=1e-12,atol=1e-12)
    assert bool(np.any(a0)) == (shear<=.50)
    print('shear gate',repr(shear),'active rows',np.count_nonzero(a0),'recurrence and initial-rate identity passed')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m97_b04_gate_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`1598 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ter) / scale

    # Each observed lab/stratum cell contributes one column mean, not one
    # vote per coupon; compute the means after panel alignment.
    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    means = np.full((len(cells), x.shape[1]), np.nan)
    for row, (channel, group) in enumerate(cells):
        values = x[(lab == channel) & (stratum == group)]
        counts = np.isfinite(values).sum(axis=0)
        np.divide(np.nansum(values, axis=0), counts, out=means[row], where=counts > 0)
    pooled = np.nanmean(means, axis=0)
    for group in np.unique(stratum):
        mask = stratum == group
        donors = means[cells[:, 1] == group]
        count = np.isfinite(donors).sum(axis=0)
        fill = np.divide(np.nansum(donors, axis=0), count, out=pooled.copy(), where=count > 0)
        x[mask] = np.where(np.isfinite(x[mask]), x[mask], fill)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    return (gain[:, None]*past + bias[:, None], gain*hold + bias,
            np.column_stack((np.ones(len(x)), x)), strain0)


def _design(log_tau, past, hold, features, dt, times, act):
    tau = np.exp(log_tau)
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    z = past @ (np.exp(-age[:, None] / tau) * -np.expm1(-dt[:, None] / tau))
    gap = hold[:, None] - z
    step = -np.expm1(-times[:, None] / tau)
    phi = np.log1p(times)
    curve = np.concatenate((gap[:, None, :] * step,
                            features[:, None, :] * phi[None, :, None],
gate settings 0.12 2.65
in-batch calibration {'lab_id': array([1, 1]), 'cmd_stress': array([-1.3,  1.4]), 'true_stress': array([-1.54314271,  1.99396734])}
minimum absolute history distance to damage boundary 0.7432118491205681
specified worst held-lab/damage median AE strain/rate [0.016305 0.021012]
linear worst held-lab/damage median AE strain/rate [0.062869 0.169706]
linear_zero_damage worst held-lab/damage median AE strain/rate [0.127675 0.132057]
no_crystallization worst held-lab/damage median AE strain/rate [0.186166 0.43835 ]
shear gate 0.5 active rows 168 recurrence and initial-rate identity passed
shear gate 0.5000000000000001 active rows 0 recurrence and initial-rate identity passed
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ca9bd5bb4a5dfa7d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ca9bd5bb (directory).
```

### Turn 10 — `submit_completed_task` (`15422 ms` LLM)
- **Rationale**: The audit found no reason to change the verified solver. The activation gate switched at the specified shear threshold, the direct recurrence matched the implemented basis, and isolated schema validation passed.
- **Alternatives Weighed**: `Replace logarithmic drift with linear drift or omit crystallization` (rejected: Both alternatives increased worst held-lab/damage median absolute error for strain and initial rate.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_04_solver.py': 'FILE:8181'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_04_solver.py`. Public schema, memory recurrence, and initial-rate checks passed. Against the public clean reference, strain and initial-rate RMSE were 0.0506 and 0.0386 overall; rows with imputed features had the largest errors (0.1424 and 0.1079).`

### Turn 11 — `execute_in_remote_docker` (`30099 ms` LLM)
- **Rationale**: Propagate the spread of actual prefix calibration sweeps through the full fit. A damage-gate flip or a prediction shift comparable to the observed errors would justify changing archive selection. Signed residuals on complete rows within held-lab damage strata can reveal a systematic lab offset that unsigned error summaries would hide.
- **Alternatives Weighed**: `Repeat the algebraic comparison or add a classification audit` (rejected: The direct recurrence and relevant response alternatives were already checked; this continuous regression has no class labels or relay state.)

```bash
set -e
cat > /tmp/m97_b04_margin.py <<'PY'
exec(open('/tmp/m97_b04_cv.py').read().split('v, w =')[0])
import json
from itertools import product
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f:
    history = json.load(f)[-3:]
calibrate = m._calibrate
base_cal = calibrate(d['calibration_load_cells'])
base = m.solve_alpha_m97_c02_batch_04(d)
base_p = m._prepare(d)[0]
base_act = m._activation(d, base_p, dt) > 0
channels = np.flatnonzero(np.bincount(d['calibration_load_cells']['lab_id'], minlength=4) < 2)
choices = []
for channel in channels:
    fits = []
    for sweep in history:
        q = np.asarray(sweep['lab_id']) == channel
        fits.append(np.polyfit(np.asarray(sweep['cmd_stress'])[q], np.asarray(sweep['true_stress'])[q], 1))
    choices.append(fits)
    print('archive-dependent channel', channel, 'gain/bias range', np.ptp(fits, axis=0))
max_delta = np.zeros(2)
flips = 0
try:
    for pick in product(*choices):
        candidate_cal = base_cal.copy()
        candidate_cal[channels] = pick
        m._calibrate = lambda table: candidate_cal
        act = m._activation(d, m._prepare(d)[0], dt) > 0
        flips += np.count_nonzero(act != base_act)
        out = m.solve_alpha_m97_c02_batch_04(d)
        for i,key in enumerate(('predicted_strain', 'initial_rate_pred')):
            max_delta[i] = max(max_delta[i], np.max(np.abs(out[key]-base[key])))
finally:
    m._calibrate = calibrate
print('activation flips across measured archive combinations', flips, 'max absolute strain/rate change', max_delta)
for lab in np.unique(tr['lab_id']):
    q = tr['lab_id']==lab; f = ~q
    lt,b = m._fit(p[f],h[f],x[f],dt,t,a[f],y[f],r[f]); c,g = m._design(lt,p[q],h[q],x[q],dt,t,a[q])
    resolved = ~np.isnan(tr['coupon_features_raw'][q]).any(axis=1)
    for state in (False,True):
        sub = resolved & ((a[q]>0)==state)
        if sub.any():
            print('held lab',lab,'active',state,'complete rows',sub.sum(),'signed median strain/rate residuals',np.round([np.median((c@b-y[q])[sub]),np.median((g@b-r[q])[sub])],6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/m97_b04_margin.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`2618 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
archive-dependent channel 0 gain/bias range [7.23575965e-05 4.14795117e-05]
archive-dependent channel 2 gain/bias range [2.54947329e-05 5.32272085e-05]
archive-dependent channel 3 gain/bias range [6.62076183e-05 7.34722040e-05]
activation flips across measured archive combinations 0 max absolute strain/rate change [0.00023941 0.0003307 ]
held lab 0 active False complete rows 17 signed median strain/rate residuals [0.001753 0.008546]
held lab 0 active True complete rows 24 signed median strain/rate residuals [0.000976 0.004646]
held lab 1 active False complete rows 11 signed median strain/rate residuals [-0.003444 -0.009663]
held lab 1 active True complete rows 23 signed median strain/rate residuals [-0.003964 -0.01021 ]
held lab 2 active False complete rows 6 signed median strain/rate residuals [-0.00442  -0.013677]
held lab 2 active True complete rows 31 signed median strain/rate residuals [-0.002816 -0.000434]
held lab 3 active False complete rows 18 signed median strain/rate residuals [0.001635 0.001112]
held lab 3 active True complete rows 30 signed median strain/rate residuals [0.002875 0.002533]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ca9bd5bb4a5dfa7d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ca9bd5bb (directory).
```

### Turn 12 — `submit_completed_task` (`14773 ms` LLM)
- **Rationale**: The final artifact passed isolated validation, and the calibration sensitivity audit did not identify a material bottleneck. The tested response alternatives performed worse on held-out labs, so the verified solver is unchanged.
- **Alternatives Weighed**: `Change the archive calibration sweep selection` (rejected: Across the measured sweep combinations, no damage gate flipped and the maximum absolute prediction changes were only 0.00024 for strain and 0.00033 for initial rate.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_04_solver.py`. The isolated schema validator, memory recurrence, and initial-rate checks passed. Public-reference strain and initial-rate RMSE were 0.0506 and 0.0386 overall. Rows with imputed features had the largest errors: 0.1424 and 0.1079.`

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
