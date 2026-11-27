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
| **Turns Used / Max Turns** | `16 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `495,280` (`479,708` / `114,816` / `15,572`) |
| **Total Episode Wall Time (s)** | `515.35s` |

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

### Turn 1 — `execute_in_remote_docker` (`17858 ms` LLM)
- **Rationale**: Inspect the public schema and prior solver before fitting. This will establish which alignment logic can be reused, whether any channel needs archived calibration, and how to define independent validation slices.

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && sed -n '1,260p' /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment cell balanced imputation variable projection' 2>&1; python3 - <<'PY'
import pickle, numpy as np
p='/holdout_task/public_alpha_m97_c01_batch_01.pkl'
d=pickle.load(open(p,'rb'))
def describe(x,depth=0):
 if isinstance(x,dict) and depth<2:
  for k,v in x.items(): print('  '*depth,k,end=': '); describe(v,depth+1)
 elif isinstance(x,np.ndarray): print(x.shape,x.dtype,x.ravel()[:5])
 elif isinstance(x,(list,tuple)): print(type(x).__name__,len(x),repr(x[:1])[:280])
 else: print(repr(x)[:350])
describe(d)
PY
```
- **Container Exit Code**: `0` (`715 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_obs'] - train['strain_0'][:, None]
    logs, weights = _fit(tuple(v[:n] for v in data), dt, times, delta,
                         train.get('instant_rate_obs'))
    strain, rate = _predict(logs, weights, tuple(v[n:] for v in data), dt, times)
    return {'estimated_taus': np.exp(logs), 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2), 'predicted_strain': test['strain_0'][:, None] + strain,
            'initial_rate_pred': rate}
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
list 1 [{'dataset': {'past_dt': array([0.6483, 0.2653, 0.7395, 0.6054, 0.3605, 0.7336, 0.6854, 0.2348,
       0.3829, 0.2905, 0.5424, 0.2461, 0.5108, 0.4695, 0.3015, 0.3905,
       0.2275, 0.2772]), 'eval_times': array([0.3   , 0.8444, 1.3889, 1.9333, 2.4778, 3.0222, 3.5667, 4.1111,
```

### Turn 2 — `execute_in_remote_docker` (`18055 ms` LLM)
- **Rationale**: Inspect the panel shapes and calibration residuals to determine the alignment implementation and whether any channel qualifies for archived calibration. Identify the available training targets before fitting the plateau model.

```bash
sed -n '1,150p' /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py; head -c 2300 /tmp/hegel_continual_archive/archive_calibration_history.json; printf '\n'; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation Prony'; python3 - <<'PY'
import pickle,numpy as np
obj=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0]; print('item keys',list(obj)); d=obj['dataset']; print('dataset keys',list(d));
for label in ('train','test','calibration_load_cells'):
 print(label)
 for k,v in d[label].items():
  a=np.asarray(v); print(k,a.shape,a.dtype,a.ravel()[:4], 'missing',np.sum(~np.isfinite(a)) if a.dtype.kind=='f' else '')
 if 'lab_id' in d[label]: print('lab counts',np.unique(d[label]['lab_id'],return_counts=True))
print('shear',d['shear_banding_index'],'times',d['eval_times']); c=d['calibration_load_cells'];
for lab in np.unique(c['lab_id']):
 m=c['lab_id']==lab; x=c['cmd_stress'][m]; y=c['true_stress'][m]; p=np.polyfit(x,y,1); print('cal',lab,p,'max abs residual',np.max(abs(y-np.polyval(p,x))))
PY
```
- **Container Exit Code**: `0` (`635 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
item keys ['dataset', 'gt']
dataset keys ['past_dt', 'eval_times', 'crit_stress', 'yield_stress', 'shear_banding_index', 'calibration_load_cells', 'train', 'test']
train
lab_id (180,) int64 [0 1 2 3] missing 
stratum (180,) int64 [2 1 2 2] missing 
coupon_features_raw (180, 4) float64 [-1.41345577 -3.01022004  0.23115262 -1.27150237] missing 19
ref_coupons (180, 32) float64 [-0.70667737 -0.36521757 -1.47123501 -2.24526949] missing 238
past_stress_cmd (180, 18) float64 [-0.19080749 -1.34885709 -0.84168836  2.76498738] missing 0
stress_hold_cmd (180,) float64 [ 0.07958731 -0.11365195 -0.21990909 -0.68871687] missing 0
strain_0 (180,) float64 [-0.552188  0.406017 -0.145016 -0.423774] missing 0
aux_chamber_telemetry (180, 3) float64 [-0.94468  -0.248683 -0.856825  0.67546 ] missing 0
strain_curves_obs (180, 10) float64 [-0.30682153 -0.00513339  0.11082238  0.22859336] missing 0
instant_rate_obs (180,) float64 [ 1.11099833 -0.66347914 -0.3031002  -0.69647305] missing 0
lab counts (array([0, 1, 2, 3]), array([36, 50, 44, 50]))
test
lab_id (72,) int64 [0 1 2 3] missing 
stratum (72,) int64 [0 0 1 0] missing 
coupon_features_raw (72, 4) float64 [-1.34597117 -1.27212031  1.0461071  -1.57610825] missing 4
ref_coupons (72, 32) float64 [ 0.07822747 -0.22232973 -1.10442901 -0.49520557] missing 96
past_stress_cmd (72, 18) float64 [-1.0782928  -0.36112687 -1.45929886 -3.06396519] missing 0
stress_hold_cmd (72,) float64 [-0.29222313  2.18481558 -1.35448264  2.28210482] missing 0
strain_0 (72,) float64 [ 0.513179  0.68354  -0.21838   0.415848] missing 0
aux_chamber_telemetry (72, 3) float64 [ 1.223187 -0.042873 -0.05421  -1.122217] missing 0
lab counts (array([0, 1, 2, 3]), array([20, 18, 21, 13]))
calibration_load_cells
lab_id (40,) int64 [0 0 0 0] missing 
cmd_stress (40,) float64 [-1.457421 -1.418035  1.887501 -2.004396] missing 0
true_stress (40,) float64 [-1.80293462 -1.75023439  2.67922472 -2.53590638] missing 0
lab counts (array([0, 1, 2, 3]), array([10, 10, 10, 10]))
shear 0.12 times [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ]
cal 0 [1.33999935 0.14999378] max abs residual 0.00015647060919077305
cal 1 [ 0.77000406 -0.16999413] max abs residual 0.00018822176288391113
cal 2 [ 1.21002713 -0.12000366] max abs residual 0.0001037212625434325
cal 3 [0.83000992 0.10997266] max abs residual 8.725097943051985e-05
```

### Turn 3 — `execute_in_remote_docker` (`23739 ms` LLM)
- **Rationale**: Check the prior alignment for a reusable cell-mean implementation and count active, inactive, and missing-feature slices. An empty feature cell would require a balanced fallback; a sparse damage slice would affect how validation is grouped.
- **Alternatives Weighed**: `Use archived calibration coefficients` (rejected: All four channels have ten clean in-batch rows, so their affine maps should be fitted directly.)

```bash
sed -n '/^def _prepare/,/^def _fit/p' /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel; python3 - <<'PY'
import pickle,numpy as np
p='/holdout_task/public_alpha_m97_c01_batch_01.pkl'
item=pickle.load(open(p,'rb'))[0]; d=item['dataset']; g=item['gt']; print('public validation keys:', [(k,np.shape(v)) for k,v in g.items()] if isinstance(g,dict) else type(g))
for split in ('train','test'):
 s=d[split]; c=d['calibration_load_cells']; lab=s['lab_id']; st=s['stratum']; x=s['coupon_features_raw']; params=np.array([np.polyfit(c['cmd_stress'][c['lab_id']==i],c['true_stress'][c['lab_id']==i],1) for i in range(4)])[lab]; past=params[:,0,None]*s['past_stress_cmd']+params[:,1,None]; damage=np.maximum(abs(past)-d['crit_stress'],0)**2@d['past_dt']; print(split,'inactive/active:',sum(damage==0),sum(damage>0),'damage quantiles:',np.quantile(damage,[0,.25,.5,.75,1])); print('counts and minimum finite feature count by cell:',[(i,int(j),sum(m),int(np.isfinite(x[m]).sum(axis=0).min())) for i in range(4) for j in np.unique(st) if np.any(m:=(lab==i)&(st==j))]);
PY
```
- **Container Exit Code**: `0` (`623 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
icability": {},
      "functions": [
        {
          "function": "align_control_panel",
          "signature": "align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray",
          "doc_summary": "Return signal values in control-referenced coordinates, in input row order."
        }
      ],
      "index_cards": [
        "autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel",
        "autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_control_panel_deconfounding",
      "reference_source_txt": null
    }
  ]
}
public validation keys: [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (72, 10)), ('te_true_rate0', (72,)), ('te_hist_true', (72, 18)), ('te_hold_true', (72,))]
train inactive/active: 60 120 damage quantiles: [0.         0.         2.92135341 3.60866767 5.28598181]
counts and minimum finite feature count by cell: [(0, 0, 13, 11), (0, 1, 10, 10), (0, 2, 13, 11), (1, 0, 12, 10), (1, 1, 23, 21), (1, 2, 15, 13), (2, 0, 13, 12), (2, 1, 7, 7), (2, 2, 24, 24), (3, 0, 19, 18), (3, 1, 13, 12), (3, 2, 18, 16)]
test inactive/active: 24 48 damage quantiles: [0.         0.         2.66043706 3.51930565 4.81411533]
counts and minimum finite feature count by cell: [(0, 0, 8, 8), (0, 1, 7, 7), (0, 2, 5, 4), (1, 0, 5, 5), (1, 1, 9, 8), (1, 2, 4, 4), (2, 0, 7, 6), (2, 1, 9, 9), (2, 2, 5, 5), (3, 0, 4, 4), (3, 1, 6, 6), (3, 2, 3, 3)]
```

### Turn 4 — `execute_in_remote_docker` (`52900 ms` LLM)
- **Rationale**: Profile the nine linear coefficients at each pair of relaxation times. Hold public ground truth out of fitting and use it to check whether prediction errors concentrate in inactive, missing-feature, or individual lab slices.
- **Alternatives Weighed**: `Force the two relaxation times into ascending order` (rejected: The crystallization terms single out branch zero, so the two orders must be tested rather than treated as interchangeable.)

```bash
set -e
cd /holdout_task
p=alpha_m97_c01_batch_01_solver.py
{ printf '"""Fit calibrated two-branch polymer rheology with a crystallized plateau."""\n'; sed -n '2,/^def _design/{ /^def _design/!p; }' /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py; } > "$p"
cat >> "$p" <<'PY'
def _state(data, dt, crit, plateau):
    past, hold, x = data
    excess = np.maximum(np.abs(past) - crit, 0.)
    activation = np.tanh((excess**2) @ dt) if plateau else np.zeros_like(hold)
    return past, hold, x, activation


def _design(logtaus, state, dt, times, plateau):
    past, hold, x, activation = state
    tau = np.exp(logtaus)
    z = np.zeros((len(hold), 2))
    for width, stress in zip(dt, past.T):
        u = -width/tau
        z = np.exp(u)*z - np.expm1(u)*stress[:, None]
    gap = hold[:, None] - z
    transient = -np.expm1(-times[:, None]/tau)
    phi = np.log1p(times) if plateau else times
    drift = np.column_stack((np.ones(len(hold)), x))
    a = np.empty((len(hold), len(times), 9))
    a[:, :, :2] = gap[:, None, :]*transient
    a[:, :, 2:7] = drift[:, None, :]*phi[None, :, None]
    a[:, :, 7] = (activation*hold)[:, None]*transient[:, 0]
    a[:, :, 8] = (activation*z[:, 0])[:, None]*phi
    rate = np.column_stack((gap/tau, drift, activation*hold/tau[0], activation*z[:, 0]))
    return a, rate


def _fit(state, dt, times, plateau, delta, rate_obs=None):
    curve_good = np.isfinite(delta)
    rate_good = np.isfinite(rate_obs) if rate_obs is not None else None
    use_rates = rate_good is not None and rate_good.any()
    def matrices(logtaus):
        a, rate = _design(logtaus, state, dt, times, plateau)
        return a[curve_good], rate[rate_good] if use_rates else None

    def profile(logtaus, rate_weight):
        a, rate = matrices(logtaus)
        y = delta[curve_good]
        if rate_weight > 0:
            a = np.vstack((a, rate_weight*rate))
            y = np.r_[y, rate_weight*rate_obs[rate_good]]
        weights = np.linalg.lstsq(a, y, rcond=None)[0]
        return np.einsum('ij,j->i', a, weights) - y, weights

    bounds = np.log([max(np.min(dt)*0.05, 1e-4), 20*(dt.sum()+times.max())])
    starts = ([0.35, 2.5], [0.75, 4.0], [1.5, 7.0])
    if plateau and np.any(state[3]):
        starts += tuple(pair[::-1] for pair in starts)
    def fit_from(start, rate_weight):
        return least_squares(lambda p: profile(p, rate_weight)[0], start,
                             bounds=(bounds[0], bounds[1]), max_nfev=90)

    fits = [fit_from(np.clip(np.log(pair), *bounds), 0.) for pair in starts]
    best = min(fits, key=lambda fit: fit.cost)
    rate_weight = 0.
    if use_rates:
        residual, w = profile(best.x, 0.)
        _, rate = matrices(best.x)
        def scale(r):
            return max(1.4826*np.median(np.abs(r-np.median(r))), 1e-8)
        rate_weight = scale(residual)/scale(rate @ w-rate_obs[rate_good])
        best = min((fit_from(fit.x, rate_weight) for fit in fits), key=lambda fit: fit.cost)
    logs = best.x.copy()
    weights = profile(logs, rate_weight)[1]
    if not np.any(state[3]) and logs[0] > logs[1]:
        logs = logs[::-1]
        weights[:2] = weights[1::-1]
    return logs, weights


def solve_alpha_m97_c01_batch_01(dataset: dict) -> dict:
    dt, times = dataset['past_dt'], dataset['eval_times']
    train, test = dataset['train'], dataset['test']
    data, n = _prepare(dataset)
    plateau = dataset['shear_banding_index'] <= 0.50
    state = _state(data, dt, dataset['crit_stress'], plateau)
    delta = train['strain_curves_obs'] - train['strain_0'][:, None]
    logs, weights = _fit(tuple(s[:n] for s in state), dt, times, plateau, delta,
                         train.get('instant_rate_obs'))
    a, rate = _design(logs, tuple(s[n:] for s in state), dt, times, plateau)
    return {'estimated_taus': np.exp(logs), 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + a @ weights,
            'initial_rate_pred': rate @ weights}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import runpy,pickle,numpy as np,time
p='alpha_m97_c01_batch_01_solver.py'; m=runpy.run_path(p); obj=pickle.load(open('public_alpha_m97_c01_batch_01.pkl','rb'))[0]; d,g=obj['dataset'],obj['gt']; st=time.monotonic(); out=m['solve_alpha_m97_c01_batch_01'](d); print('runtime',time.monotonic()-st)
for k,l in zip(out,('taus','theta','vp_theta','te_clean','te_true_rate0')):
 e=out[k]-g[l]; print(k,'RMSE',np.sqrt(np.mean(e**2)),'max',np.max(abs(e))); print((out[k],g[l]) if out[k].ndim==1 and len(out[k])<10 else '')
data,n=m['_prepare'](d); a=m['_state'](data,d['past_dt'],d['crit_stress'],True)[3][n:]; t=d['test']; axes={'active':a>0,'inactive':a==0,'missing':np.isnan(t['coupon_features_raw']).any(axis=1)}; axes.update({f'lab{i}':t['lab_id']==i for i in range(4)})
for name,sel in axes.items(): print(name,sel.sum(),*(np.sqrt(np.mean((out[k][sel]-g[l][sel])**2)) for k,l in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]))
PY
```
- **Container Exit Code**: `0` (`938 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
runtime 0.13033879292197526
estimated_taus RMSE 0.702751115822842 max 0.9925238478863703
(array([0.56186613, 2.67947615]), array([0.613, 3.672]))
hereditary_weights RMSE 0.02326998965320482 max 0.05583326573987135
(array([ 0.62416673,  0.56094946,  0.13544674,  0.15761817, -0.14208664,
        0.10693754, -0.07550578]), array([ 0.68,  0.55,  0.14,  0.16, -0.13,  0.12, -0.09]))
regime_mode_weights RMSE 0.06142250698385331 max 0.08032319927410167
(array([0.1996768 , 0.27307011]), array([0.28, 0.24]))
predicted_strain RMSE 0.08102553133416378 max 0.3700990190381457

initial_rate_pred RMSE 0.08593268995872821 max 0.22174254562774554

active 48 0.09126985285007491 0.07447250445298885
inactive 24 0.05509117992384998 0.10517116416407987
missing 4 0.1538155235088276 0.14330758158474136
lab0 20 0.07744183937011305 0.09546685383355984
lab1 18 0.09556007133434842 0.0871010648163685
lab2 21 0.06992405630336877 0.07701672873878768
lab3 13 0.08119167154350733 0.0824054374423313

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a2b790476505c297 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a2b79047 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`25105 ms` LLM)
- **Rationale**: The current design applies log-time drift even when past work-damage is zero. Compare it with specimen-level gating in leave-one-lab-out fits using only training measurements; retain the change only if it improves held-lab errors.
- **Alternatives Weighed**: `Retune relaxation times before checking the plateau gate` (rejected: Changing time constants can mask an incorrect terminal-flow basis without fixing it.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle,runpy,numpy as np
m=runpy.run_path('alpha_m97_c01_batch_01_solver.py'); d=pickle.load(open('public_alpha_m97_c01_batch_01.pkl','rb'))[0]['dataset']; data,n=m['_prepare'](d); dt,t=d['past_dt'],d['eval_times']; state=m['_state'](data,dt,d['crit_stress'],True); train=d['train']; y=train['strain_curves_obs']-train['strain_0'][:,None]; r=train['instant_rate_obs']; fit=m['_fit']; env=fit.__globals__; original=env['_design'];
def gated(logs,s,dt,t,plateau):
 a,b=original(logs,s,dt,t,plateau)
 if plateau:
  a[s[3]==0,:,2:7]=b[s[3]==0,None,2:7]*t[None,:,None]
 return a,b
for name,design in [('batch_log',original),('damage_gated',gated)]:
 env['_design']=design; p=np.empty_like(y); q=np.empty_like(r)
 for lab in range(4):
  test=train['lab_id']==lab; sel=~test
  logs,w=fit(tuple(v[:n][sel] for v in state),dt,t,True,y[sel],r[sel]); a,b=design(logs,tuple(v[:n][test] for v in state),dt,t,True); p[test]=a@w; q[test]=b@w
  print(name,'held lab',lab,'tau',np.exp(logs),'curve RMSE',np.mean((p[test]-y[test])**2)**.5,'rate RMSE',np.mean((q[test]-r[test])**2)**.5)
 for label,sel in [('active',state[3][:n]>0),('inactive',state[3][:n]==0),('missing',np.isnan(train['coupon_features_raw']).any(axis=1)),('complete',np.isfinite(train['coupon_features_raw']).all(axis=1))]:
  e=p[sel]-y[sel]; f=q[sel]-r[sel]; print(name,label,'n',sel.sum(),'curve RMSE',np.mean(e**2)**.5,'curve MAD',np.median(abs(e)),'rate RMSE',np.mean(f**2)**.5,'rate MAD',np.median(abs(f)),'late bias',e[:,-1].mean())
PY
```
- **Container Exit Code**: `0` (`1536 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
98906 2.01436439] curve RMSE 0.670553506577639 rate RMSE 0.11170202539807171
batch_log held lab 1 tau [0.60575752 3.10621219] curve RMSE 1.5796808035415209 rate RMSE 0.08868410963744673
batch_log held lab 2 tau [0.56783415 3.45319592] curve RMSE 0.6410427302268509 rate RMSE 0.08692086399518298
batch_log held lab 3 tau [0.51919983 2.63641759] curve RMSE 0.41853923153681866 rate RMSE 0.09461694463624408
batch_log active n 120 curve RMSE 1.0443556429156335 curve MAD 0.07698223701169499 rate RMSE 0.09888027033084114 rate MAD 0.06530589416221676 late bias 0.0059871427466873794
batch_log inactive n 60 curve RMSE 0.7843805198739812 curve MAD 0.03875519889011576 rate RMSE 0.08652549450371869 rate MAD 0.07286854090687109 late bias 0.0926358253139254
batch_log missing n 19 curve RMSE 1.1703772773450611 curve MAD 0.14001333937401483 rate RMSE 0.17132838137626613 rate MAD 0.13440595683895795 late bias 0.1281513650162503
batch_log complete n 161 curve RMSE 0.9383834937927296 curve MAD 0.05552983801408765 rate RMSE 0.0813229794251375 rate MAD 0.06468483000114711 late bias 0.02386168144800781
damage_gated held lab 0 tau [0.48744834 1.659381  ] curve RMSE 0.6960674262469916 rate RMSE 0.1589782934606804
damage_gated held lab 1 tau [0.60632053 2.62402553] curve RMSE 1.5914188414590198 rate RMSE 0.16099043258485585
damage_gated held lab 2 tau [0.50847499 1.83156978] curve RMSE 0.6783320188985676 rate RMSE 0.15672919872105978
damage_gated held lab 3 tau [0.51043998 2.38756013] curve RMSE 0.4464586622746804 rate RMSE 0.1618064972865045
damage_gated active n 120 curve RMSE 1.0588857296496368 curve MAD 0.15196110054027856 rate RMSE 0.16534084751707837 rate MAD 0.1289323609277997 late bias -0.10469105313503668
damage_gated inactive n 60 curve RMSE 0.8138033151259004 curve MAD 0.07499730569193733 rate RMSE 0.14805049539354057 rate MAD 0.0893804229760694 late bias 0.24618713845092588
damage_gated missing n 19 curve RMSE 1.1765469445151076 curve MAD 0.16893495447988355 rate RMSE 0.18878814478755931 rate MAD 0.14944996522610277 late bias 0.06811162120525065
damage_gated complete n 161 curve RMSE 0.9587266309539907 curve MAD 0.12130879623011237 rate RMSE 0.15600744817109613 rate MAD 0.11833645246673274 late bias 0.005678143651871977

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a2b790476505c297 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a2b79047 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`40673 ms` LLM)
- **Rationale**: Held-lab curve RMSE is much larger than median absolute error, suggesting a heavy residual tail. Compare ordinary and robust fits on the same held-lab folds, estimating residual scales only from each fitting fold. Keep ordinary least squares if robustness does not improve held-out errors or stabilize relaxation times.
- **Alternatives Weighed**: `Add lab-specific flow coefficients` (rejected: That would not transfer to a held-out lab and would obscure whether sparse large residuals are distorting the shared fit.)
- **ATMS NOGOOD DECLARED**: Refuted `Zero work-damage should switch the baseline terminal-flow basis from log-time to linear time within this batch.` | Witness: `Specimen-level gating raised curve and initial-rate RMSE in all four held-out labs; inactive-slice median absolute curve error rose from 0.0388 to 0.0750.` | Blanket Diagnosis: `Work-damage controls the two activation-weighted terms; using it to also switch the baseline drift basis worsens generalization.` | Sublated Pivot: `Keep the batch-wide log-time basis and test whether robust fitting improves held-lab performance without changing the physical model.`

```bash
cd /holdout_task && cat > /tmp/rheology_robust_probe.py <<'PY'
import pickle, runpy
import numpy as np
from scipy.optimize import least_squares
m = runpy.run_path('/holdout_task/alpha_m97_c01_batch_01_solver.py')
d = pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0]['dataset']
s = d['train']; dt,t = d['past_dt'],d['eval_times']; data,n = m['_prepare'](d)
state = tuple(v[:n] for v in m['_state'](data,dt,d['crit_stress'],True))
y = s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; design=m['_design']

def mad(e):
    return max(1.4826*np.median(np.abs(e-np.median(e))),1e-8)

def fit(mask, kind):
    ss=tuple(v[mask] for v in state); yy=y[mask]; rr=r[mask]
    p0,w0=m['_fit'](ss,dt,t,True,yy,rr)
    A,B=design(p0,ss,dt,t,True)
    sc,sr=mad(A@w0-yy),mad(B@w0-rr)
    print('scales',kind,round(sc,4),round(sr,4),'beyond 5 MAD',np.sum(abs(A@w0-yy)>5*sc),np.sum(abs(B@w0-rr)>5*sr),flush=True)
    if kind=='ols': return p0,w0
    target=np.r_[yy.ravel()/sc,rr/sr]
    def profile(p, ret=False):
        a,b=design(p,ss,dt,t,True)
        x=np.vstack((a.reshape(-1,9)/sc,b/sr))
        w=np.linalg.lstsq(x,target,rcond=None)[0]
        for _ in range(10):
            e=(x@w-target)/2.0
            q=np.power(1+e**2,-0.5 if kind=='soft_l1' else -1.)
            v=np.sqrt(q)
            w=np.linalg.lstsq(x*v[:,None],target*v,rcond=None)[0]
        if ret: return w
        e=x@w-target
        if kind=='soft_l1': return e*np.sqrt(2/(np.sqrt(1+(e/2)**2)+1))
        return np.sign(e)*2*np.sqrt(np.log1p((e/2)**2))
    fits=[least_squares(profile,p,bounds=(-5.,5.5),max_nfev=75) for p in (p0,p0[::-1],np.log([0.6,3.5]))]
    best=min(fits,key=lambda f:f.cost)
    return best.x,profile(best.x,True)

for kind in ('ols','soft_l1','cauchy'):
    pc=np.empty_like(y); pr=np.empty_like(r); pars=[]
    for lab in range(4):
        ix=s['lab_id']==lab; p,w=fit(~ix,kind); pars.append(np.exp(p))
        a,b=design(p,tuple(v[ix] for v in state),dt,t,True); pc[ix]=a@w; pr[ix]=b@w
    print(kind,'fold taus',np.round(pars,4))
    for name,ix in [(f'lab{i}',s['lab_id']==i) for i in range(4)]+[('active',state[3]>0),('inactive',state[3]==0),('missing',np.isnan(s['coupon_features_raw']).any(axis=1))]:
        a,b=abs(pc[ix]-y[ix]),abs(pr[ix]-r[ix]); print(kind,name,'MAE/median/90p curves',np.mean(a),np.median(a),np.quantile(a,.90),'rate RMSE/median',np.mean(b*b)**.5,np.median(b))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS' | head -8
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_robust_probe.py
```
- **Container Exit Code**: `0` (`2985 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
an/90p curves 0.19707658954481547 0.015825333475125936 0.16018366973855294 rate RMSE/median 0.06424241035116537 0.021824140730086317
soft_l1 lab2 MAE/median/90p curves 0.1295462728888691 0.013145658233939761 0.1257626775237691 rate RMSE/median 0.02854704550978576 0.012063465900203463
soft_l1 lab3 MAE/median/90p curves 0.07996781068491877 0.016609326605818597 0.06917039713144815 rate RMSE/median 0.025627508941267845 0.014603981347234396
soft_l1 active MAE/median/90p curves 0.1486919214726463 0.01778012357619152 0.17963955611486845 rate RMSE/median 0.05526294274502567 0.017469696470048035
soft_l1 inactive MAE/median/90p curves 0.10392660006439523 0.013291620523358916 0.10192508682375202 rate RMSE/median 0.03534656228035596 0.013627967490724813
soft_l1 missing MAE/median/90p curves 0.2411452125556831 0.1101308354973577 0.3636130661049632 rate RMSE/median 0.14028380986188219 0.09280682352736003
scales cauchy 0.0848 0.1085 beyond 5 MAD 66 0
scales cauchy 0.0559 0.0492 beyond 5 MAD 76 1
scales cauchy 0.0892 0.0959 beyond 5 MAD 54 0
scales cauchy 0.1067 0.102 beyond 5 MAD 58 0
cauchy fold taus [[0.6122 3.6795]
 [0.6116 3.6567]
 [0.6122 3.638 ]
 [0.6098 3.6811]]
cauchy lab0 MAE/median/90p curves 0.1246860134089044 0.018696438796597437 0.21250234702660636 rate RMSE/median 0.06816341987424956 0.01592047051060949
cauchy lab1 MAE/median/90p curves 0.19610887004197133 0.016166145987430935 0.161995668809742 rate RMSE/median 0.06437303402511085 0.021240130581978267
cauchy lab2 MAE/median/90p curves 0.12860328003117455 0.012157360608901824 0.12612378205559574 rate RMSE/median 0.027920051646163315 0.0103065707016948
cauchy lab3 MAE/median/90p curves 0.07794252215111704 0.014345427592184337 0.06823939606023303 rate RMSE/median 0.024147842537513488 0.01480031498730483
cauchy active MAE/median/90p curves 0.14700440489053104 0.01603810145043108 0.1811804172738382 rate RMSE/median 0.05517022124117583 0.015393438814332167
cauchy inactive MAE/median/90p curves 0.10348803044804882 0.012617658006851551 0.10152625977407675 rate RMSE/median 0.03480044220018095 0.013611044243480752
cauchy missing MAE/median/90p curves 0.24119942534215202 0.10914074612005764 0.36103736548630144 rate RMSE/median 0.14145091986131242 0.08631033375327535

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a2b790476505c297 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a2b79047 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`37922 ms` LLM)
- **Rationale**: Promote the robust fit because it reduced held-lab errors without changing the physical model. Verify every output against the public reference in an isolated process, and separate missing-feature errors from complete-coupon errors to determine whether alignment still limits predictions.
- **Alternatives Weighed**: `Use soft-L1 profiling` (rejected: It gave similar missing-feature errors, but Cauchy profiling yielded a higher and more stable slow relaxation time across held-lab folds.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_batch_01_solver.py')
s = p.read_text()
a = s.index('    fits = [fit_from(')
b = s.index('    if not np.any(state[3])', a)
s = s[:a] + '''    fits = [fit_from(np.clip(np.log(pair), *bounds), 0.) for pair in starts]
    best = min(fits, key=lambda fit: fit.cost)
    initial, w = profile(best.x, 0.)

    def scale(residual):
        return max(1.4826*np.median(np.abs(residual-np.median(residual))), 1e-8)

    curve_scale = scale(initial)
    _, rate = matrices(best.x)
    if use_rates:
        rate_scale = scale(rate @ w-rate_obs[rate_good])
        rate_weight = curve_scale/rate_scale
        best = min((fit_from(fit.x, rate_weight) for fit in fits), key=lambda fit: fit.cost)
        initial, w = profile(best.x, rate_weight)
        curve_scale = scale(initial[:curve_good.sum()])
        _, rate = matrices(best.x)
        rate_scale = scale(rate @ w-rate_obs[rate_good])
        target = np.r_[delta[curve_good]/curve_scale, rate_obs[rate_good]/rate_scale]
    else:
        target = delta[curve_good]/curve_scale

    def robust(logtaus, return_weights=False):
        a, rate = matrices(logtaus)
        a = a/curve_scale
        if use_rates:
            a = np.vstack((a, rate/rate_scale))
        weights = np.linalg.lstsq(a, target, rcond=None)[0]
        for _ in range(10):
            e = np.einsum('ij,j->i', a, weights)-target
            v = 1/np.hypot(1., e/2.)
            weights = np.linalg.lstsq(a*v[:, None], target*v, rcond=None)[0]
        if return_weights:
            return weights
        e = np.einsum('ij,j->i', a, weights)-target
        return np.sign(e)*2*np.sqrt(np.log1p((e/2.)**2))

    robust_starts = [best.x, np.clip(np.log([0.75, 4.0]), *bounds)]
    if plateau and np.any(state[3]):
        robust_starts.append(best.x[::-1])
    best = min((least_squares(robust, start, bounds=tuple(bounds), max_nfev=90)
                for start in robust_starts), key=lambda fit: fit.cost)
    logs = best.x.copy()
    weights = robust(logs, return_weights=True)
''' + s[b:]
p.write_text(s)
PY
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import pickle,runpy,time,numpy as np
p='alpha_m97_c01_batch_01_solver.py'; obj=pickle.load(open('public_alpha_m97_c01_batch_01.pkl','rb'))[0]; d,g=obj['dataset'],obj['gt']; m=runpy.run_path(p); t0=time.monotonic(); out=m['solve_alpha_m97_c01_batch_01'](d); print('runtime',time.monotonic()-t0)
for k,l in zip(out,('taus','theta','vp_theta','te_clean','te_true_rate0')):
 e=out[k]-g[l]; print(k,'RMSE',np.sqrt(np.mean(e**2)),'max',np.max(abs(e)), 'estimate' if out[k].size<10 else '',out[k] if out[k].size<10 else '')
data,n=m['_prepare'](d); h=m['_state'](data,d['past_dt'],d['crit_stress'],True)[3][n:]; test=d['test']; missing=np.isnan(test['coupon_features_raw']).any(axis=1)
axes={'active':h>0,'inactive':h==0,'missing':missing,'complete':~missing}; axes.update({f'lab{i}':test['lab_id']==i for i in range(4)})
for name,sel in axes.items():
 e=out['predicted_strain'][sel]-g['te_clean'][sel]; r=out['initial_rate_pred'][sel]-g['te_true_rate0'][sel]; print(name,sel.sum(),'strain/rate RMSE',np.mean(e**2)**.5,np.mean(r**2)**.5,'early/late strain RMSE',np.mean(e[:,0]**2)**.5,np.mean(e[:,-1]**2)**.5)
PY
```
- **Container Exit Code**: `0` (`1932 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
runtime 0.3289577781688422
estimated_taus RMSE 0.0029174988131229966 max 0.003934398064320277 estimate [0.61175738 3.6680656 ]
hereditary_weights RMSE 0.0035301035943197476 max 0.005439686539387423 estimate [ 0.67456031  0.55418266  0.1384529   0.15601024 -0.12572538  0.11823794
 -0.08932349]
regime_mode_weights RMSE 0.004621065259712971 max 0.005648150624190301 estimate [0.28328738 0.24564815]
predicted_strain RMSE 0.05438634605772322 max 0.5709688576600527  
initial_rate_pred RMSE 0.041719706160113494 max 0.3119144688275366  
active 48 strain/rate RMSE 0.060452718471232514 0.04618409761135344 early/late strain RMSE 0.012083999810766542 0.08438712048581705
inactive 24 strain/rate RMSE 0.03955453920104597 0.030913749399071454 early/late strain RMSE 0.008088860262874012 0.05497070545524482
missing 4 strain/rate RMSE 0.22860363641349837 0.17417554622787787 early/late strain RMSE 0.04560172428078186 0.31916883341180946
complete 68 strain/rate RMSE 0.007600777412687305 0.007640792127753564 early/late strain RMSE 0.0019604519564846096 0.010047416333629967
lab0 20 strain/rate RMSE 0.045992543207713094 0.035161628610927025 early/late strain RMSE 0.009222116422862914 0.06410942568192625
lab1 18 strain/rate RMSE 0.09676938489030278 0.07398191389652171 early/late strain RMSE 0.019353220398238998 0.13508013554534617
lab2 21 strain/rate RMSE 0.009049510090729716 0.008523022730847896 early/late strain RMSE 0.002194769497158682 0.012177936898710402
lab3 13 strain/rate RMSE 0.005429815545109578 0.006482712066649654 early/late strain RMSE 0.0016516634609271366 0.006710080412500295

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ea99f886db4e4969 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ea99f886 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`27269 ms` LLM)
- **Rationale**: Use reference coefficients only for diagnosis. If the residual is confined to missing coupons and follows the log-time drift basis, audit the prescribed cell-mean alignment rather than changing the fitted spectrum. If it persists on complete coupons or has a different time dependence, inspect calibration and the memory basis.
- **Alternatives Weighed**: `Retune robust fitting to reduce missing-coupon error` (rejected: The complete-coupon errors are already small; first determine whether the missing-coupon error persists with the reference coefficients.)

```bash
set -e
cd /holdout_task
sed -n '1,/^def _state/p' alpha_m97_c01_batch_01_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy, numpy as np
m = runpy.run_path('alpha_m97_c01_batch_01_solver.py')
item = pickle.load(open('public_alpha_m97_c01_batch_01.pkl','rb'))[0]
d,g = item['dataset'],item['gt']; (past,hold,x),n = m['_prepare'](d)
test = d['test']; missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
s = m['_state']((past[n:],hold[n:],x[n:]),d['past_dt'],d['crit_stress'],True)
a,b = m['_design'](np.log(g['taus']),s,d['past_dt'],d['eval_times'],True)
w = np.r_[g['theta'],g['vp_theta']]
e = test['strain_0'][:,None]+a@w-g['te_clean']; r=b@w-g['te_true_rate0']
phi = np.log1p(d['eval_times']); drift_error = e@phi/(phi@phi)
print('public reference keys:',list(g))
for name,sel in [('complete',~missing),('missing',missing)]:
 print(name,'reference-coefficient strain/rate RMSE',np.mean(e[sel]**2)**.5,np.mean(r[sel]**2)**.5,'max after removing drift',np.max(np.abs((e-drift_error[:,None]*phi)[sel])))
for i in np.flatnonzero(missing):
 print('missing row',i,'lab',test['lab_id'][i],'stratum',test['stratum'][i],'missing columns',np.flatnonzero(~np.isfinite(test['coupon_features_raw'][i])).tolist(),'drift/rate errors',drift_error[i],r[i])
print('max alignment errors past/hold',np.max(abs(past[n:]-g['te_past_true'])),np.max(abs(hold[n:]-g['te_hold_true']))) if 'te_past_true' in g else None
PY
```
- **Container Exit Code**: `0` (`428 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
np.concatenate((train[key], test[key]), axis=0)
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


def _state(data, dt, crit, plateau):
public reference keys: ['taus', 'theta', 'vp_theta', 'stress_gains', 'stress_biases', 'te_clean', 'te_true_rate0', 'te_hist_true', 'te_hold_true']
complete reference-coefficient strain/rate RMSE 0.008900654218951663 0.006796941681474485 max after removing drift 1.607429234433671e-05
missing reference-coefficient strain/rate RMSE 0.23514401515917646 0.17956214324654624 max after removing drift 4.1949746862335324e-06
missing row 12 lab 0 stratum 2 missing columns [3] drift/rate errors 0.07148900597998087 0.07148723435657534
missing row 40 lab 2 stratum 0 missing columns [1] drift/rate errors 0.024912849239261115 0.02491085154336936
missing row 43 lab 1 stratum 1 missing columns [0] drift/rate errors 0.3197531582685397 0.319765173569722
missing row 68 lab 0 stratum 2 missing columns [2] drift/rate errors -0.14487764332013223 -0.1448775749401218

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ea99f886db4e4969 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ea99f886 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`30076 ms` LLM)
- **Rationale**: The reference coefficients reproduce the observed residual as a drift error, so another spectrum fit will not fix it. Compare cell-mean interpretations without using public test labels. Also check that archived calibration resolves the four channel IDs; if it does not, fix the fallback before testing sparse and spiked calibration.
- **Alternatives Weighed**: `Choose an imputer by its error on the four public missing coupons` (rejected: Four rows are too few to distinguish a general improvement from chance; mask observed training entries instead.)

```bash
set -e
cd /holdout_task
python3 -I - <<'PY'
import json, pickle, runpy
import numpy as np
m = runpy.run_path('alpha_m97_c01_batch_01_solver.py')
d = pickle.load(open('public_alpha_m97_c01_batch_01.pkl', 'rb'))[0]['dataset']
(past, hold, x), n = m['_prepare'](d)
train, test = d['train'], d['test']
raw = np.concatenate((train['coupon_features_raw'], test['coupon_features_raw']))
lab = np.r_[train['lab_id'], test['lab_id']]
stratum = np.r_[train['stratum'], test['stratum']]
print('reference panel shapes', np.shape(train['ref_coupons']), np.shape(test['ref_coupons']))
for key, value in json.load(open('public_asset_index.json')).items():
    if any(word in key.lower() for word in ('panel', 'align', 'imput', 'reference')):
        print(key, str(value)[:1000])
print('archive fallback by lab', [m['_historical_calibration'](i) for i in range(4)])
# Mask observed training entries one at a time; use only donor rows for each estimate.
errors = {'balanced_global': [], 'matched_cell': [], 'balanced_within_stratum': []}
for i, j in zip(*np.where(np.isfinite(raw[:n]))):
    donor = np.isfinite(raw[:, j]); donor[i] = False
    means = {(l, s): x[sel, j].mean() for l, s in set(zip(lab, stratum))
             if (sel := (lab == l) & (stratum == s) & donor).any()}
    global_mean = np.mean(list(means.values()))
    estimates = [global_mean, means.get((lab[i], stratum[i]), global_mean),
                 np.mean([v for (l, s), v in means.items() if s == stratum[i]])]
    for rows, estimate in zip(errors.values(), estimates):
        rows.append((i, j, estimate-x[i, j]))
for name, rows in errors.items():
    i, j, e = np.asarray(rows).T; i, j = i.astype(int), j.astype(int)
    print(name, 'feature RMSE', round(np.mean(e**2)**.5, 5), 'worst lab/stratum/column RMSE', round(max(np.mean(e[mask]**2)**.5 for l,s,c in set(zip(lab[i],stratum[i],j)) if (mask := (lab[i]==l)&(stratum[i]==s)&(j==c)).sum() >= 5), 5))
PY
```
- **Container Exit Code**: `1` (`776 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
reference panel shapes (180, 32) (72, 32)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ea99f886db4e4969 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ea99f886 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 14, in <module>
  File "alpha_m97_c01_batch_01_solver.py", line 23, in _historical_calibration
    rows = [r for r in history if int(r.get('lab_id', lab)) == lab]
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: int() argument must be a string, a bytes-like object or a real number, not 'list'
```

### Turn 10 — `execute_in_remote_docker` (`28834 ms` LLM)
- **Rationale**: Inspect only the archive layout needed to repair the parser. Run the interrupted masked-entry comparison independently: retain the balanced global mean unless a contract-compatible interpretation improves both overall and worst-cell error.
- **Alternatives Weighed**: `Keep the inherited archive reader unchanged` (rejected: It raises TypeError on the supplied history, so the required sparse-channel fallback cannot run.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import json, pickle, runpy
import numpy as np

def outline(obj, level=0):
    if isinstance(obj, dict) and level < 4:
        return {k: outline(v, level+1) for k, v in list(obj.items())[:12]}
    if isinstance(obj, list):
        return {'length': len(obj), 'first': outline(obj[0], level+1) if obj and level < 4 else None}
    return obj if isinstance(obj, (str, int, float, type(None))) else type(obj).__name__
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f:
    print('archive layout:', json.dumps(outline(json.load(f)))[:2400])
m = runpy.run_path('alpha_m97_c01_batch_01_solver.py')
with open('public_alpha_m97_c01_batch_01.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
(_, _, x), n = m['_prepare'](d)
train, test = d['train'], d['test']
raw = np.concatenate((train['coupon_features_raw'], test['coupon_features_raw']))
lab = np.r_[train['lab_id'], test['lab_id']]
stratum = np.r_[train['stratum'], test['stratum']]
cells = np.unique(np.column_stack((lab,stratum)), axis=0)
errors = {'balanced_global': [], 'matched_cell': [], 'balanced_within_stratum': []}
for i, j in zip(*np.where(np.isfinite(raw[:n]))):
    donor = np.isfinite(raw[:,j]); donor[i] = False
    means = {(l,s): x[ix,j].mean() for l,s in cells if (ix := (lab==l)&(stratum==s)&donor).any()}
    overall = np.mean(list(means.values()))
    estimates = [overall, means.get((lab[i],stratum[i]),overall), np.mean([v for (l,s),v in means.items() if s==stratum[i]])]
    for rows, estimate in zip(errors.values(),estimates):
        rows.append((lab[i], stratum[i], j, estimate-x[i,j]))
for name, rows in errors.items():
    rows = np.asarray(rows); e = rows[:,3]
    worst = max(np.mean(e[ix]**2)**.5 for group in np.unique(rows[:,:3],axis=0) if (ix := np.all(rows[:,:3]==group,axis=1)).sum() >= 5)
    print(name, 'feature RMSE', round(np.mean(e**2)**.5,5), 'worst lab/stratum/column RMSE', round(worst,5))
PY
```
- **Container Exit Code**: `0` (`539 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
archive layout: {"length": 1, "first": {"episode_short_name": "alpha_m97_c01_s00", "lab_id": {"length": 40, "first": 0}, "cmd_stress": {"length": 40, "first": 1.660283}, "true_stress": {"length": 40, "first": 1.660305}}}
balanced_global feature RMSE 1.10477 worst lab/stratum/column RMSE 1.43262
matched_cell feature RMSE 1.11144 worst lab/stratum/column RMSE 1.56463
balanced_within_stratum feature RMSE 1.09032 worst lab/stratum/column RMSE 1.48849

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ea99f886db4e4969 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ea99f886 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`47224 ms` LLM)
- **Rationale**: The archive stores column arrays per sweep, which the inherited parser mistook for individual rows. Read the latest usable sweep by channel. In a clean process, verify that sparse channels use the archive, clean channels bypass it, and rejecting EMI spikes leaves all five outputs close to the clean-data result. Any failure would identify whether to revise archive selection, spike rejection, or the calibration handoff.
- **Alternatives Weighed**: `Use stratum-conditioned imputation` (rejected: Masked-entry overall RMSE improved from 1.105 to 1.090, but worst-cell RMSE worsened from 1.433 to 1.488; equal weighting across lab–stratum cells remains the better fit to the contract.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('alpha_m97_c01_batch_01_solver.py')
s = p.read_text()
a = s.index('    history = json.loads(')
b = s.index('\n\ndef _calibrate', a)
s = s[:a] + '''    history = json.loads(_ARCHIVE.read_text())
    for sweep in reversed(history):
        mask = np.asarray(sweep['lab_id']) == lab
        x = np.asarray(sweep['cmd_stress'])[mask]
        y = np.asarray(sweep['true_stress'])[mask]
        if len(x) >= 2 and np.ptp(x) > 0:
            return np.polyfit(x, y, 1)
    return None
''' + s[b:]
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import json, pickle, runpy
from unittest.mock import patch
import numpy as np
m = runpy.run_path('alpha_m97_c01_batch_01_solver.py')
with open('public_alpha_m97_c01_batch_01.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, g = item['dataset'], item['gt']
c = d['calibration_load_cells']; lab = c['lab_id']
calibrate = m['_calibrate']; solve = m['solve_alpha_m97_c01_batch_01']
with patch.dict(calibrate.__globals__, {'_historical_calibration': lambda _: (_ for _ in ()).throw(AssertionError('clean data must not read the archive'))}):
    clean = calibrate(c)
    out = solve(d)
    for count in (2, 3):
        rows = np.flatnonzero(lab == 0)[:count]
        keep = (lab != 0); keep[rows] = True
        sub = {k: v[keep] for k,v in c.items()}
        np.testing.assert_allclose(calibrate(sub)[0], np.polyfit(c['cmd_stress'][rows], c['true_stress'][rows], 1))
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f:
    archived = json.load(f)[-1]
a_lab, a_x, a_y = (np.asarray(archived[k]) for k in ('lab_id','cmd_stress','true_stress'))
for channel in range(4):
    old = a_lab == channel
    expected = np.polyfit(a_x[old], a_y[old], 1)
    for count in (0, 1):
        keep = lab != channel
        keep[np.flatnonzero(~keep)[:count]] = True
        got = calibrate({k: v[keep] for k, v in c.items()})
        np.testing.assert_allclose(got[channel], expected)
        np.testing.assert_allclose(got[np.arange(4) != channel], clean[np.arange(4) != channel])
    row = np.flatnonzero(lab == channel)[0]
    spiked = {k:v.copy() for k,v in c.items()}; spiked['true_stress'][row] += 8
    keep = (lab == channel); keep[row] = False
    np.testing.assert_allclose(calibrate(spiked)[channel], np.polyfit(c['cmd_stress'][keep], c['true_stress'][keep], 1))
print('Calibration stress cases passed: 0–3 points and isolated EMI on each channel (2–3 clean points tested on channel 0).')
spiked = {k:v.copy() for k,v in c.items()}
for channel in range(4):
    row = np.flatnonzero(lab == channel)[0]; spiked['true_stress'][row] += 8
challenger = solve(dict(d, calibration_load_cells=spiked))
for key in out:
    np.testing.assert_allclose(challenger[key], out[key], rtol=0, atol=1e-3)
keys = [('estimated_taus','taus',.004),('hereditary_weights','theta',.005),('regime_mode_weights','vp_theta',.006),('predicted_strain','te_clean',.056),('initial_rate_pred','te_true_rate0',.043)]
for key, reference, limit in keys:
    rmse = np.sqrt(np.mean((out[key]-g[reference])**2)); print(key, 'RMSE', round(rmse,6)); assert rmse < limit
missing = np.isnan(d['test']['coupon_features_raw']).any(axis=1)
(data, n) = m['_prepare'](d)
active = m['_state'](data,d['past_dt'],d['crit_stress'],True)[3][n:] > 0
for label, mask in [('missing',missing),('complete',~missing),('active',active),('inactive',~active)]+[(f'lab{i}',d['test']['lab_id']==i) for i in range(4)]:
    e = out['predicted_strain'][mask]-g['te_clean'][mask]; r = out['initial_rate_pred'][mask]-g['te_true_rate0'][mask]
    print(label, 'strain/rate/late-strain RMSE', *(round(float(np.sqrt(np.mean(v*v))),6) for v in (e,r,e[:,-1])))
PY
```
- **Container Exit Code**: `0` (`1948 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Calibration stress cases passed: 0–3 points and isolated EMI on each channel (2–3 clean points tested on channel 0).
estimated_taus RMSE 0.002917
hereditary_weights RMSE 0.00353
regime_mode_weights RMSE 0.004621
predicted_strain RMSE 0.054386
initial_rate_pred RMSE 0.04172
missing strain/rate/late-strain RMSE 0.228604 0.174176 0.319169
complete strain/rate/late-strain RMSE 0.007601 0.007641 0.010047
active strain/rate/late-strain RMSE 0.060453 0.046184 0.084387
inactive strain/rate/late-strain RMSE 0.039555 0.030914 0.054971
lab0 strain/rate/late-strain RMSE 0.045993 0.035162 0.064109
lab1 strain/rate/late-strain RMSE 0.096769 0.073982 0.13508
lab2 strain/rate/late-strain RMSE 0.00905 0.008523 0.012178
lab3 strain/rate/late-strain RMSE 0.00543 0.006483 0.00671

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a4911d4eeb17943a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_a4911d4e (directory).
```

### Turn 12 — `submit_completed_task` (`17004 ms` LLM)
- **Rationale**: The final artifact passed isolated verification for all five outputs and the calibration stress cases. The remaining worst-slice error is concentrated in missing coupon features and persists when reference coefficients are used, so further relaxation-parameter tuning would not address it.
- **Alternatives Weighed**: `Replace globally cell-balanced imputation with stratum-conditioned means` (rejected: Masked-entry overall RMSE improved slightly, but worst-cell RMSE rose from 1.433 to 1.488.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_01_solver.py': 'FILE:8422'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_01_solver.py`. All five outputs passed schema validation, and sparse-calibration and EMI-spike checks passed. Public strain and initial-rate RMSE were 0.0544 and 0.0417 overall; the missing-feature coupons were the hardest slice at 0.229 and 0.174.`

### Turn 13 — `execute_in_remote_docker` (`34738 ms` LLM)
- **Rationale**: Independently reconstruct the branch states from the closed-form convolution. A disagreement would identify a memory or output-handoff error; signed residuals and physical stress errors by lab will reveal whether a calibration mismatch is being hidden by pooling. Check the exact artifact in an isolated process without changing the fit.
- **Alternatives Weighed**: `Add rational, pairwise, or relay terms to the rheology model` (rejected: The specified model has two exponential memory branches and logarithmic drift, with no relay state or stochastic drift operator. The independent reconstruction checks that every prescribed term reaches both outputs.); `Audit within-batch balanced accuracy` (rejected: Both targets are continuous. Signed errors within lab and damage strata can expose biases that cancel in pooled regression metrics.)

```bash
set -e
cd /holdout_task
sed -n '/^def _calibrate/,/^def _prepare/p' alpha_m97_c01_batch_01_solver.py
cat > /tmp/rheology_handoff_audit.py <<'PY'
import pickle, runpy
import numpy as np
from pathlib import Path
root = Path('/holdout_task')
m = runpy.run_path(str(root/'alpha_m97_c01_batch_01_solver.py'))
with (root/'public_alpha_m97_c01_batch_01.pkl').open('rb') as f:
    item = pickle.load(f)[0]
d, g = item['dataset'], item['gt']
out = m['solve_alpha_m97_c01_batch_01'](d)
(past, hold, x), n = m['_prepare'](d)
dt, t = d['past_dt'], d['eval_times']
tau, hw, rw = (out[k] for k in ('estimated_taus','hereditary_weights','regime_mode_weights'))
tail = dt.sum() - np.cumsum(dt)
z = past[n:] @ (np.exp(-tail[:,None]/tau) - np.exp(-(tail+dt)[:,None]/tau))
gap = hold[n:,None] - z
D = np.maximum(np.abs(past[n:])-d['crit_stress'],0)**2 @ dt
act = np.tanh(D)
amplitudes = gap*hw[:2]
amplitudes[:,0] += rw[0]*act*hold[n:]
drift = hw[2] + x[n:]@hw[3:7] + rw[1]*act*z[:,0]
eps = d['test']['strain_0'][:,None] + amplitudes @ (1-np.exp(-t[None,:]/tau[:,None])) + drift[:,None]*np.log1p(t)
rate = amplitudes @ (1/tau) + drift
for label, pred, actual in [('strain',eps,out['predicted_strain']),('rate',rate,out['initial_rate_pred'])]:
    for name, sel in [('zero damage',D==0),('positive damage',D>0)]:
        e = np.max(np.abs(pred[sel]-actual[sel])); print(label,name,'max independent handoff difference',e); assert e < 1e-10
s = d['test']; mask = np.isfinite(s['coupon_features_raw']).all(axis=1)
curve_e = eps-g['te_clean']; rate_e = rate-g['te_true_rate0']
print('Complete coupons only, mean signed strain/rate residual by lab and damage group:')
for lab in np.unique(s['lab_id']):
    for damage in (False, True):
        q = mask & (s['lab_id']==lab) & ((D>0)==damage)
        if q.any(): print(int(lab),damage,int(q.sum()),round(curve_e[q].mean(),6),round(rate_e[q].mean(),6))
print('Reference alignment comparison by lab, including cases with missing coupons:')
for lab in np.unique(s['lab_id']):
    q = s['lab_id']==lab
    print(int(lab),'max past/hold stress error',np.max(abs(past[n:][q]-g['te_past_true'][q])),np.max(abs(hold[n:][q]-g['te_hold_true'][q])))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_handoff_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `1` (`1282 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
def _calibrate(c):
    result = np.tile([1., 0.], (4, 1))
    for lab in range(4):
        mask = (c['lab_id'] == lab) & np.isfinite(c['cmd_stress']) & np.isfinite(c['true_stress'])
        x, y = c['cmd_stress'][mask], c['true_stress'][mask]
        spikes = False
        if len(x) >= 3 and np.ptp(x) > 0:
            # A repeated-median line detects gross calibration errors before OLS.
            dx = x[:, None] - x
            dy = y[:, None] - y
            with np.errstate(divide='ignore', invalid='ignore'):
                slopes = [np.median(yi[xi != 0] / xi[xi != 0]) for xi, yi in zip(dx, dy)]
            gain = np.median(slopes)
            bias = np.median(y - gain*x)
            good = np.abs(y - gain*x - bias) <= 3.0
            spikes = not np.all(good)
            if spikes:
                x, y = x[good], y[good]
        historical = _historical_calibration(lab) if len(x) < 2 or spikes else None
        if len(x) >= 2 and np.ptp(x) > 0 and (len(x) >= 4 or historical is None):
            result[lab] = np.polyfit(x, y, 1)
        elif historical is not None:
            result[lab] = historical
        elif len(x):
            result[lab, 1] = np.median(y - x)
    return result


def _prepare(dataset):
strain zero damage max independent handoff difference 8.881784197001252e-16
strain positive damage max independent handoff difference 8.881784197001252e-16
rate zero damage max independent handoff difference 6.661338147750939e-16
rate positive damage max independent handoff difference 8.881784197001252e-16
Complete coupons only, mean signed strain/rate residual by lab and damage group:
0 False 6 -0.002165 0.000794
0 True 12 -0.001365 -0.001428
1 False 4 -0.007043 -0.009184
1 True 13 -0.005488 -0.004662
2 False 8 -0.002956 -0.002722
2 True 12 -0.001361 -0.000613
3 False 5 -0.00387 -0.004246
3 True 8 -0.001123 -0.001111
Reference alignment comparison by lab, including cases with missing coupons:

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a4911d4eeb17943a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_a4911d4e (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/rheology_handoff_audit.py", line 36, in <module>
    print(int(lab),'max past/hold stress error',np.max(abs(past[n:][q]-g['te_past_true'][q])),np.max(abs(hold[n:][q]-g['te_hold_true'][q])))
                                                                       ~^^^^^^^^^^^^^^^^
KeyError: 'te_past_true'
```

### Turn 14 — `execute_in_remote_docker` (`21862 ms` LLM)
- **Rationale**: Complete the interrupted calibration audit using the measured reference under its actual key. A lab exceeding the audit margin would warrant checking its gain, bias, or channel indexing; otherwise the solver remains unchanged.
- **Alternatives Weighed**: `Change the solver after the audit exception` (rejected: The exception came from a reference key in the temporary audit. Independent reconstruction already agreed with the solver in both damage slices.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
from pathlib import Path
p = Path('/tmp/rheology_handoff_audit.py')
s = p.read_text().split("print('Reference alignment comparison by lab", 1)[0]
exec(compile(s, str(p), 'exec'))
print('Public reference fields:', [(k, np.shape(v)) for k,v in g.items()])
traces = [(k, np.asarray(v)) for k,v in g.items() if 'past' in k and np.shape(v) == past[n:].shape]
assert len(traces) == 1, 'Cannot uniquely identify the public test stress-history reference'
key, reference_past = traces[0]
print('Stress-history reference:', key)
for lab in np.unique(d['test']['lab_id']):
    sel = d['test']['lab_id'] == lab
    ep = np.max(np.abs(past[n:][sel]-reference_past[sel]))
    eh = np.max(np.abs(hold[n:][sel]-g['te_hold_true'][sel]))
    print(f'lab {lab}: maximum history/hold stress error {ep:.6g}/{eh:.6g}')
    assert max(ep, eh) < 1e-3, 'Calibration handoff exceeds the public audit margin'
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `1` (`1183 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
strain zero damage max independent handoff difference 8.881784197001252e-16
strain positive damage max independent handoff difference 8.881784197001252e-16
rate zero damage max independent handoff difference 6.661338147750939e-16
rate positive damage max independent handoff difference 8.881784197001252e-16
Complete coupons only, mean signed strain/rate residual by lab and damage group:
0 False 6 -0.002165 0.000794
0 True 12 -0.001365 -0.001428
1 False 4 -0.007043 -0.009184
1 True 13 -0.005488 -0.004662
2 False 8 -0.002956 -0.002722
2 True 12 -0.001361 -0.000613
3 False 5 -0.00387 -0.004246
3 True 8 -0.001123 -0.001111
Public reference fields: [('taus', (2,)), ('theta', (7,)), ('vp_theta', (2,)), ('stress_gains', (4,)), ('stress_biases', (4,)), ('te_clean', (72, 10)), ('te_true_rate0', (72,)), ('te_hist_true', (72, 18)), ('te_hold_true', (72,))]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a4911d4eeb17943a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_a4911d4e (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
AssertionError: Cannot uniquely identify the public test stress-history reference
```

### Turn 15 — `execute_in_remote_docker` (`20024 ms` LLM)
- **Rationale**: Use the confirmed te_hist_true field to finish the interrupted per-lab calibration audit. A threshold failure would identify the channel whose gain, bias, or indexing needs inspection. If all channels pass, verify the unchanged artifact in an isolated process and submit.
- **Alternatives Weighed**: `Change the calibration or plateau fit now` (rejected: The last audit stopped because it did not select the known reference field, not because it measured a solver error.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy
import numpy as np
m = runpy.run_path('alpha_m97_c01_batch_01_solver.py')
with open('public_alpha_m97_c01_batch_01.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, g = item['dataset'], item['gt']
(past, hold, _), n = m['_prepare'](d)
for lab in np.unique(d['test']['lab_id']):
    mask = d['test']['lab_id'] == lab
    history_error = np.max(np.abs(past[n:][mask] - g['te_hist_true'][mask]))
    hold_error = np.max(np.abs(hold[n:][mask] - g['te_hold_true'][mask]))
    print(f'lab {lab}: maximum history/hold stress error {history_error:.6g}/{hold_error:.6g}')
    assert max(history_error, hold_error) < 1e-3, f'lab {lab}: calibration handoff exceeds the public audit margin'
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`1521 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lab 0: maximum history/hold stress error 8.08935e-06/7.10968e-06
lab 1: maximum history/hold stress error 2.76239e-05/1.6444e-05
lab 2: maximum history/hold stress error 9.06358e-05/4.40259e-05
lab 3: maximum history/hold stress error 7.60654e-05/5.22783e-05
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a4911d4eeb17943a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_a4911d4e (directory).
```

### Turn 16 — `submit_completed_task` (`15511 ms` LLM)
- **Rationale**: The final artifact passed isolated validation. Independent strain and initial-rate reconstruction agreed within 8.9e-16 in both damage slices, and maximum per-lab stress alignment error was 9.1e-5, below the 1e-3 audit margin. Missing-feature error persisted with reference coefficients, so further relaxation-parameter tuning would not address it.
- **Alternatives Weighed**: `Use stratum-conditioned imputation` (rejected: Masked-entry overall RMSE fell slightly, but worst-cell RMSE rose from 1.433 to 1.488.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_01_solver.py`. All five outputs passed schema validation, and sparse-calibration and EMI-spike checks passed. Public strain and initial-rate RMSE were 0.0544 and 0.0417 overall. Coupons with missing features were the hardest slice, at 0.229 and 0.174.`

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
