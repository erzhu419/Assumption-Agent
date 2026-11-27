# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c02_batch_07__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c02_step_07` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `415,226` (`401,178` / `108,032` / `14,048`) |
| **Total Episode Wall Time (s)** | `458.03s` |

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

### Turn 1 — `execute_in_remote_docker` (`17313 ms` LLM)
- **Rationale**: Inspect actual array shapes, available labels, and reference panel layout before coding; a schema or alignment mismatch would change which estimator can be fit and how it must be validated.
- **Alternatives Weighed**: `Reuse the rejected gain and bias payload` (rejected: The public gate rejected it, and the contract prioritizes clean in-batch calibration.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -l /tmp/hegel_continual_archive; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference-panel alignment imputation variable projection'; python3 - <<'PY'
import pickle,glob,numpy as np
p=pickle.load(open(glob.glob('/holdout_task/*.pkl')[0],'rb'))
def view(x,d=0):
 if isinstance(x,dict) and d<3:
  for k,v in x.items():
   print(' '*d,k,type(v).__name__,getattr(v,'shape',''),end=' ')
   if isinstance(v,dict):print();view(v,d+1)
   elif isinstance(v,list):print('len',len(v),str(v[:1])[:500])
   else:print(str(v if np.size(v)<20 else np.asarray(v).flat[:0])[:350])
view(p)
PY
```
- **Container Exit Code**: `0` (`436 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_index_v1"
}total 76
-rw-r--r-- 1 root root  7679 Nov 27 09:20 alpha_m97_c02_batch_01_solver.py
-rw-r--r-- 1 root root  6161 Nov 27 09:20 alpha_m97_c02_batch_02_solver.py
-rw-r--r-- 1 root root  7444 Nov 27 09:20 alpha_m97_c02_batch_03_solver.py
-rw-r--r-- 1 root root  6786 Nov 27 09:20 alpha_m97_c02_batch_04_solver.py
-rw-r--r-- 1 root root  6786 Nov 27 09:20 alpha_m97_c02_batch_05_solver.py
-rw-r--r-- 1 root root  6563 Nov 27 09:20 alpha_m97_c02_batch_06_solver.py
-rw-r--r-- 1 root root  7113 Nov 27 09:20 alpha_m97_c02_s00_solver.py
-r--r--r-- 1 root root 10572 Nov 27 09:20 archive_calibration_history.json
-rw-r--r-- 1 root root  3324 Nov 27 09:20 archive_manifest.json
-rw-r--r-- 1 root root   340 Nov 27 09:20 pre_llm_gate_diagnostic.json
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 2 — `execute_in_remote_docker` (`17765 ms` LLM)
- **Rationale**: Inspect the calibration history and predecessor alignment code to establish which sweeps are usable and how the reference coupons and training measurements are encoded. Those details determine the calibration fallback and the fitting design.
- **Alternatives Weighed**: `Fit all four channels from current calibration alone` (rejected: The public dataset contains only two calibration rows across four channels, so some channels necessarily require historical sweeps.)

```bash
python3 - <<'PY'
import json,pickle,numpy as np
p=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_07.pkl','rb'))[0]['dataset']
for k,v in p.items():
 print(k,[(a, np.shape(b)) for a,b in v.items()] if isinstance(v,dict) else v)
print('calibration',p['calibration_load_cells'])
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('history:',str(h)[:1300]); print('history tail:',str(h)[-1900:])
for k in ('train','test'):
 v=p[k]; print(k,'lab counts',np.unique(v['lab_id'],return_counts=True)); print(k,'reference keys',[(a, np.shape(b)) for a,b in v.items() if 'ref' in a])
PY
sed -n '1,245p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_06_solver.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel | head -35
```
- **Container Exit Code**: `0` (`407 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
fit.cost).fun)
    robust = []
    for fit in fits:
        matrix = design(fit.x)
        weights = np.linalg.lstsq(matrix / scale[:, None], target / scale, rcond=None)[0]
        robust.append(least_squares(residual, np.r_[fit.x, weights], args=(scale,),
                                   bounds=joint_bounds, loss='cauchy', f_scale=2.0,
                                   x_scale='jac', max_nfev=100))
    best = min(robust, key=lambda fit: fit.cost)
    scale = scales(best.fun * scale)
    best = least_squares(residual, best.x, args=(scale,), bounds=joint_bounds,
                         loss='cauchy', f_scale=2.0, x_scale='jac', max_nfev=100)
    return np.exp(best.x[:2]), best.x[2:]


def solve_alpha_m97_c02_batch_06(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    past, hold, terminal = _prepare(dataset)
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    curves = train['strain_curves_obs'] - np.asarray(train['strain_0'])[:, None]
    tau, weights = _fit(past[:n], hold[:n], terminal[:n], dt, times, curves, train['instant_rate_obs'])
    a, b = _design(past[n:], hold[n:], terminal[n:], dt, times, tau)
    return {'estimated_taus': tau, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': np.asarray(test['strain_0'])[:, None] + a @ weights,
            'initial_rate_pred': b @ weights}
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

### Turn 3 — `execute_in_remote_docker` (`24596 ms` LLM)
- **Rationale**: Summarize each archived sweep by channel to distinguish repeated calibration lines from episode shifts. If there are multiple plausible lines, retain them as candidates rather than averaging them. Read the predecessor’s reference-panel transform and check the training measurements for missing values before building the fit.
- **Alternatives Weighed**: `Pool all prior calibration rows unconditionally` (rejected: The earlier output already shows different channel-1 calibration lines across episodes; averaging distinct lines could assign the wrong stress scale.)

```bash
sed -n '1,107p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_06_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
p=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_07.pkl','rb'))[0]['dataset']
for h in json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))+[dict(p['calibration_load_cells'],episode_short_name='current')]:
 print(h['episode_short_name'],end=' ')
 for c in range(4):
  mask=np.array(h['lab_id'])==c; x=np.array(h['cmd_stress'])[mask]; y=np.array(h['true_stress'])[mask]; n=len(x)
  if n>=2:
   a=np.column_stack((x,np.ones(n))); w=np.linalg.lstsq(a,y,rcond=None)[0]; good=abs(y-a@w)<0.02; print((c,n,np.round(w,5).tolist(),int(good.sum())),end=' ')
  else: print((c,n),end=' ')
 print()
for s in ('train','test'):
 a=p[s]; print(s,'strata/counts',[(c,np.unique(a['stratum'][a['lab_id']==c],return_counts=True)) for c in range(4)],'NaNs coupons/reference',np.isnan(a['coupon_features_raw']).sum(axis=0),np.isnan(a['ref_coupons']).sum()); print('stress range per channel',[(c,np.round(np.quantile(abs(a['past_stress_cmd'][a['lab_id']==c]),[.5,.95,1]),3)) for c in range(4)])
print('bad training labels',[(k,int(np.sum(~np.isfinite(p['train'][k])))) for k in ('strain_curves_obs','instant_rate_obs')]); print('time grids',p['eval_times'])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation' | grep -Ei 'viscoelastic|Prony|rheolog' | head -15
```
- **Container Exit Code**: `0` (`456 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
curve = np.concatenate((gap[:, None, :] * relax,
                            terminal[:, None, :] * phi[None, :, None],
                            (contrast[:, None] * relax[None, :, 0])[:, :, None]), axis=2)
    rate = np.column_stack((gap / tau, terminal, contrast / tau[0]))
    return curve, rate


def _fit(past, hold, terminal, dt, times, curves, rates):
    nc = curves.size
    target = np.r_[curves.ravel(), rates]

    def design(logtau):
        a, b = _design(past, hold, terminal, dt, times, np.exp(logtau))
alpha_m97_c02_s00 (0, 10, [0.99998, 0.0], 10) (1, 10, [0.99999, 4e-05], 10) (2, 10, [1.00001, 3e-05], 10) (3, 10, [1.00001, -3e-05], 10) 
alpha_m97_c02_batch_01 (0, 10, [0.79005, -0.13999], 10) (1, 10, [1.30996, 0.16006], 10) (2, 10, [0.84996, 0.10004], 10) (3, 10, [1.24, -0.12996], 10) 
alpha_m97_c02_batch_02 (0, 10, [0.79001, -0.14], 10) (1, 10, [1.30998, 0.15996], 10) (2, 10, [0.84999, 0.09998], 10) (3, 10, [1.24004, -0.12998], 10) 
alpha_m97_c02_batch_03 (0, 10, [0.78997, -0.13996], 10) (1, 10, [1.30997, 0.15996], 10) (2, 10, [0.84999, 0.10002], 10) (3, 10, [1.23997, -0.13004], 10) 
alpha_m97_c02_batch_04 (0, 0) (1, 2, [1.31004, 0.15991], 2) (2, 0) (3, 0) 
alpha_m97_c02_batch_05 (0, 10, [0.79001, -0.14], 10) (1, 10, [1.30998, 0.15996], 10) (2, 10, [0.84999, 0.09998], 10) (3, 10, [1.24004, -0.12998], 10) 
alpha_m97_c02_batch_06 (0, 3, [1.40483, -3.50642], 0) (1, 12, [0.74, 0.20002], 12) (2, 12, [1.37002, -0.17003], 12) (3, 3, [1.66308, 3.22], 0) 
current (0, 0) (1, 2, [0.73991, 0.20004], 2) (2, 0) (3, 0) 
train strata/counts [(0, (array([0, 1, 2]), array([20, 18,  5]))), (1, (array([0, 1, 2]), array([16, 22, 15]))), (2, (array([0, 1, 2]), array([11, 14, 13]))), (3, (array([0, 1, 2]), array([ 8, 20, 18])))] NaNs coupons/reference [3 4 5 4] 229
stress range per channel [(0, array([1.746, 4.844, 5.238])), (1, array([1.931, 5.385, 5.668])), (2, array([0.956, 2.872, 3.03 ])), (3, array([1.03 , 3.104, 3.329]))]
test strata/counts [(0, (array([0, 1]), array([7, 9]))), (1, (array([0, 1, 2]), array([3, 9, 6]))), (2, (array([0, 1, 2]), array([4, 5, 6]))), (3, (array([0, 1, 2]), array([ 5,  7, 11])))] NaNs coupons/reference [1 2 5 4] 89
stress range per channel [(0, array([1.812, 4.839, 5.188])), (1, array([1.969, 5.104, 5.666])), (2, array([0.95 , 2.841, 3.031])), (3, array([1.116, 3.112, 3.317]))]
bad training labels [('strain_curves_obs', 0), ('instant_rate_obs', 0)]
time grids [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ]
```

### Turn 4 — `execute_in_remote_docker` (`73564 ms` LLM)
- **Rationale**: Build the solver from the specified constitutive equation without using public truth for fitting. The schema check catches output-contract failures; activation coverage and residual quantiles show whether rate, strain, or the plateau transition needs separate validation.
- **Alternatives Weighed**: `Take the latest archived sweep for every missing channel` (rejected: The latest channel-0 and channel-3 sweeps have three points each and no residual within 0.02 under the fitted lines. Channel 2 has a consistent 12-point sweep and should use it.)

```bash
cat > /holdout_task/alpha_m97_c02_batch_07_solver.py <<'PY'
"""Calibrate load cells and fit a two-branch crystallization plateau."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _channel(sweep, lab):
    x = np.asarray(sweep['cmd_stress'], dtype=float)
    y = np.asarray(sweep['true_stress'], dtype=float)
    mask = (np.asarray(sweep['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _line(x, y, archived=False):
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    design = np.column_stack((x, np.ones(len(x))))
    coeff = np.linalg.lstsq(design, y, rcond=None)[0]
    # Archived sweeps resolve a static calibration to much better than 0.02.
    # Use that redundancy to reject sparse sweeps made entirely of EMI points.
    tol = 0.02 if archived else 3.0
    if np.max(np.abs(y - design @ coeff)) <= tol:
        return coeff
    best = None
    for i, j in zip(*np.triu_indices(len(x), k=1)):
        if x[i] == x[j]:
            continue
        gain = (y[j] - y[i]) / (x[j] - x[i])
        error = np.abs(y - y[i] - gain * (x - x[i]))
        keep = error <= tol
        count = np.count_nonzero(keep)
        key = (count, -np.dot(error[keep], error[keep]))
        if best is None or key > best[0]:
            best = key, keep
    if best is not None and best[0][0] >= 3:
        keep = best[1]
        return np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
    return None


def _calibrate(sweep):
    coeffs = []
    history = None
    for lab in range(4):
        coeff = _line(*_channel(sweep, lab))
        if coeff is None:
            if history is None:
                history = json.loads(_HISTORY.read_text())
            for past_sweep in reversed(history):
                coeff = _line(*_channel(past_sweep, lab), archived=True)
                if coeff is not None:
                    break
            else:
                raise ValueError(f'No clean calibration for load cell {lab}')
        coeffs.append(coeff)
    return np.asarray(coeffs)


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']

    def join(key):
        return np.concatenate((train[key], test[key]), axis=0)

    lab, stratum = join('lab_id'), join('stratum')
    refs, raw = join('ref_coupons'), join('coupon_features_raw')
    aligned = np.empty_like(raw, dtype=float)
    cells = []
    for channel in np.unique(lab):
        rows = lab == channel
        panel = refs[rows]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        aligned[rows] = (raw[rows] - center) / scale
        for s in np.unique(stratum[rows]):
            values = aligned[rows & (stratum == s)]
            count = np.isfinite(values).sum(axis=0)
            cells.append(np.divide(np.nansum(values, axis=0), count,
                                   out=np.full(raw.shape[1], np.nan), where=count > 0))
    aligned = np.where(np.isfinite(aligned), aligned, np.nanmean(cells, axis=0))
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None] * join('past_stress_cmd') + bias[:, None]
    hold = gain * join('stress_hold_cmd') + bias
    dt = np.asarray(dataset['past_dt'])
    damage = np.maximum(0, np.abs(past) - dataset['crit_stress'])**2 @ dt
    act = np.tanh(damage)
    plateau = (damage > 0) & (dataset['shear_banding_index'] <= 0.50)
    return past, hold, np.column_stack((np.ones(len(hold)), aligned)), act * plateau, plateau


def _design(past, hold, drift, act, plateau, dt, times, tau):
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    memory = past @ (-np.expm1(-dt[:, None] / tau) * np.exp(-remaining[:, None] / tau))
    gap = hold[:, None] - memory
    relax = -np.expm1(-times[:, None] / tau)
    phi = np.where(plateau[:, None], np.log1p(times), times)
    transient, terminal = act * hold, act * memory[:, 0]
    curves = np.concatenate((gap[:, None, :] * relax,
                             drift[:, None, :] * phi[:, :, None],
                             (transient[:, None] * relax[:, 0])[:, :, None],
                             (terminal[:, None] * phi)[:, :, None]), axis=2)
    rates = np.column_stack((gap / tau, drift, transient / tau[0], terminal))
    return curves, rates


def _fit(inputs, dt, times, curves, rates):
    nc = curves.size
    target = np.r_[curves.ravel(), rates]
    measured = np.isfinite(target)
    target = target[measured]
    rate_rows = (np.arange(nc + len(rates)) >= nc)[measured]

    def matrix(logtau):
        a, b = _design(*inputs, dt, times, np.exp(logtau))
        return np.concatenate((a.reshape(-1, 9), b))[measured]

    def profile(logtau):
        a = matrix(logtau)
        return a @ np.linalg.lstsq(a, target, rcond=None)[0] - target

    lo, hi = np.log([0.015, 80.0])
    starts = ([0.25, 2.5], [0.7, 5.0], [2.5, 0.25], [5.0, 0.7])
    fits = [least_squares(profile, np.log(start), bounds=(lo, hi), max_nfev=80)
            for start in starts]

    def scales(residual):
        scale = np.empty_like(residual)
        for rows in (rate_rows, ~rate_rows):
            if rows.any():
                block = residual[rows]
                scale[rows] = max(1e-9, 1.4826 * np.median(np.abs(block - np.median(block))))
        return scale

    scale = scales(min(fits, key=lambda fit: fit.cost).fun)
    bounds = (np.r_[lo, lo, np.full(9, -np.inf)], np.r_[hi, hi, np.full(9, np.inf)])

    def refine(start, scale):
        def residual(params):
            return (matrix(params[:2]) @ params[2:] - target) / scale
        return least_squares(residual, start, bounds=bounds, loss='cauchy',
                             f_scale=2, x_scale='jac', max_nfev=90)

    candidates = []
    for logtau in (fit.x for fit in fits):
        a = matrix(logtau)
        weights = np.linalg.lstsq(a / scale[:, None], target / scale, rcond=None)[0]
        candidates.append(refine(np.r_[logtau, weights], scale))
    best = min(candidates, key=lambda fit: fit.cost)
    best = refine(best.x, scales(best.fun * scale))
    return np.exp(best.x[:2]), best.x[2:]


def solve_alpha_m97_c02_batch_07(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    inputs = _prepare(dataset)
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    curves = np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None]
    tau, weights = _fit(tuple(column[:n] for column in inputs), dt, times, curves,
                        np.asarray(train['instant_rate_obs']))
    a, b = _design(*(column[n:] for column in inputs), dt, times, tau)
    return {'estimated_taus': tau, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': np.asarray(test['strain_0'])[:, None] + a @ weights,
            'initial_rate_pred': b @ weights}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',p+'alpha_m97_c02_batch_07_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
a=pickle.load(open(p+'public_alpha_m97_c02_batch_07.pkl','rb'))[0]; d=a['dataset']; n=len(d['train']['lab_id']); out=m.solve_alpha_m97_c02_batch_07(d)
print('calibration',m._calibrate(d['calibration_load_cells'])); print('fit',[(k,np.round(v,8)) for k,v in out.items() if not k.endswith('pred') and k!='predicted_strain'])
inputs=m._prepare(d); print('damage activation per lab (train/test)',[(c,int(np.sum((d['train']['lab_id']==c)&inputs[-1][:n])),int(np.sum((d['test']['lab_id']==c)&inputs[-1][n:]))) for c in range(4)])
weights=np.r_[out['hereditary_weights'],out['regime_mode_weights']]; A,B=m._design(*(x[:n] for x in inputs),d['past_dt'],d['eval_times'],out['estimated_taus']); dy=A@weights+d['train']['strain_0'][:,None]-d['train']['strain_curves_obs']; dr=B@weights-d['train']['instant_rate_obs']; print('train absolute error quantiles (curve, rate)',np.quantile(abs(dy),[.5,.95,.99,1]),np.quantile(abs(dr),[.5,.95,.99,1])); print('public evaluation fields',[(k,np.shape(v) if not isinstance(v,dict) else list(v)) for k,v in a.items() if k!='dataset'])
PY
```
- **Container Exit Code**: `0` (`1336 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_07_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
calibration [[ 0.7900147  -0.14000156]
 [ 0.73991367  0.20004232]
 [ 1.37001768 -0.17003273]
 [ 1.24003845 -0.12998439]]
fit [('estimated_taus', array([0.56299656, 4.11923106])), ('hereditary_weights', array([ 0.67205586,  0.54137326,  0.11704534,  0.14187171, -0.11458396,
        0.11366398, -0.08594463])), ('regime_mode_weights', array([0.27227389, 0.24697075]))]
damage activation per lab (train/test) [(0, 28, 11), (1, 41, 13), (2, 24, 9), (3, 27, 15)]
train absolute error quantiles (curve, rate) [ 0.04126294  0.84333741  2.42279769 34.25725295] [0.0238982  0.08995763 0.22733584 0.35552244]
public evaluation fields [('gt', ['taus', 'theta', 'vp_theta', 'stress_gains', 'stress_biases', 'te_clean', 'te_true_rate0', 'te_hist_true', 'te_hold_true'])]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b72669af6db50b67 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b72669af (directory).
```

### Turn 5 — `execute_in_remote_docker` (`29688 ms` LLM)
- **Rationale**: Score the fixed solver by lab, plateau activation, and coupon missingness. Signed residuals across time distinguish isolated contamination from a systematic shape error, while the public reference values are used only to evaluate predictions and calibration.
- **Alternatives Weighed**: `Increase robust downweighting immediately` (rejected: A structured mismatch across a lab or activation state would require correcting the design, not downweighting those observations.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/'; s=importlib.util.spec_from_file_location('candidate',p+'alpha_m97_c02_batch_07_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open(p+'public_alpha_m97_c02_batch_07.pkl','rb'))[0]; d=item['dataset']; g=item['gt']; tr=d['train']; te=d['test']; n=len(tr['lab_id']); v=m._prepare(d); t=d['eval_times']; dt=d['past_dt']; out=m.solve_alpha_m97_c02_batch_07(d); w=np.r_[out['hereditary_weights'],out['regime_mode_weights']]; A,B=m._design(*v,dt,t,out['estimated_taus']); C=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=A[:n]@w-C; q=B[:n]@w-tr['instant_rate_obs'];
def stats(a): return np.round([np.median(np.abs(a)),np.quantile(abs(a),.95),np.sqrt(np.mean(a*a)),np.max(abs(a))],5)
print('parameter deltas',[(k,np.round(out[k]-g[j],5)) for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]])
print('stress errors',stats(v[0][n:]-g['te_hist_true']),stats(v[1][n:]-g['te_hold_true'])); er=out['predicted_strain']-g['te_clean']; eq=out['initial_rate_pred']-g['te_true_rate0'];
for c in range(4):
 for mode in (False,True):
  i=(tr['lab_id']==c)&(v[-1][:n]==mode); j=(te['lab_id']==c)&(v[-1][n:]==mode)
  if i.any() and j.any(): print('lab/active',c,mode,'n',sum(i),sum(j),'train curve/rate',stats(r[i]),stats(q[i]),'test curve/rate',stats(er[j]),stats(eq[j]))
print('test missing coupon',stats(er[np.isnan(te['coupon_features_raw']).any(axis=1)]),'fully observed',stats(er[np.isfinite(te['coupon_features_raw']).all(axis=1)]))
for i in np.argsort(np.median(abs(r),axis=1))[-12:][::-1]: print('worst train row',i,tr['lab_id'][i],v[-1][i],np.isnan(tr['coupon_features_raw'][i]).any(),'curve',np.round(r[i],3),'rate',round(q[i],3))
for i in np.argsort(np.median(abs(er),axis=1))[-7:][::-1]: print('worst test row',i,te['lab_id'][i],v[-1][n+i],np.isnan(te['coupon_features_raw'][i]).any(),'curve',np.round(er[i],3),'rate',round(eq[i],3))
PY
```
- **Container Exit Code**: `0` (`748 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
933 1.85014]
worst train row 160 3 False False curve [0.003 0.127 0.303 0.518 0.773 0.988 1.29  1.564 1.616 2.161] rate -0.042
worst train row 156 3 False False curve [0.008 0.085 0.229 0.356 0.57  0.778 0.956 1.332 1.365 1.61 ] rate -0.013
worst train row 171 3 False True curve [0.032 0.149 0.235 0.358 0.51  0.755 0.93  1.093 1.276 1.473] rate 0.056
worst train row 155 0 False False curve [-0.009  0.084  0.601  0.331  0.471  0.596  0.776  0.989  1.144  1.342] rate -0.052
worst train row 152 1 False False curve [-0.012  0.037  0.18   0.309  0.464  0.632  0.812  0.952  1.134  1.298] rate -0.039
worst train row 141 1 False False curve [-0.006  0.048  0.173  0.311  0.444  0.578  0.731  0.884  1.062  1.219] rate -0.032
worst train row 27 3 True True curve [-0.073 -0.206 -0.317 -0.368 -1.54  -0.476 -0.54  -0.576 -0.606 -0.626] rate -0.356
worst train row 161 0 False False curve [-0.011  0.072  0.141  0.297  0.396  0.576  0.719  0.877  1.061  1.217] rate -0.077
worst train row 143 3 False False curve [ 0.025 -0.056 -0.173  0.131 -0.392 -0.579 -0.721 -0.86  -1.011 -1.178] rate 0.03
worst train row 124 3 False False curve [0.131 0.073 0.167 0.266 0.432 0.534 0.647 0.813 0.986 1.146] rate -0.022
worst train row 173 1 False False curve [ 0.012 -0.29   0.131  0.131  0.374  0.507  0.674  0.787  0.937  1.111] rate -0.022
worst train row 178 2 False False curve [ 0.016  0.076  0.142  0.275  0.369  0.479 -1.226  0.743  0.853  1.012] rate -0.001
worst test row 70 2 False False curve [0.008 0.103 0.254 0.439 0.646 0.868 1.102 1.345 1.594 1.85 ] rate -0.052
worst test row 63 3 False False curve [0.006 0.092 0.231 0.402 0.593 0.799 1.016 1.242 1.474 1.712] rate -0.053
worst test row 50 2 False False curve [0.011 0.097 0.232 0.394 0.575 0.769 0.973 1.184 1.402 1.624] rate -0.031
worst test row 64 2 False True curve [0.046 0.155 0.283 0.423 0.572 0.725 0.883 1.044 1.208 1.373] rate 0.129
worst test row 51 3 False False curve [0.004 0.081 0.209 0.367 0.545 0.736 0.937 1.147 1.362 1.583] rate -0.054
worst test row 62 0 False False curve [-0.002  0.064  0.178  0.32   0.482  0.656  0.841  1.034  1.232  1.436] rate -0.074
worst test row 71 0 False True curve [0.03  0.103 0.191 0.286 0.387 0.492 0.6   0.71  0.822 0.935] rate 0.079

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b72669af6db50b67 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b72669af (directory).
```

### Turn 6 — `execute_in_remote_docker` (`32384 ms` LLM)
- **Rationale**: The logarithmic and linear kernels have the same derivative at zero but diverge at later horizons. A lab-held-out comparison tests whether changing that kernel removes the late-time bias without hurting active rows or instantaneous rates; if it does not, retain the baseline and investigate a different zero-damage mechanism.
- **Alternatives Weighed**: `Retune stress calibration or discard zero-damage rows as outliers` (rejected: Stress calibration is already accurate, and the signed late-time discrepancy repeats across channels and in clean public references.)
- **ATMS NOGOOD DECLARED**: Refuted `Zero measured damage implies a return to linear terminal flow in this low-shear batch.` | Witness: `Zero-damage rows show coherent late-time strain errors across all labs, up to 1.85 on public references, despite small initial-rate errors and test stress calibration errors below 0.0006.` | Blanket Diagnosis: `The transient and the derivative at zero are approximately right; the extrapolating terminal time kernel is wrong specifically where the solver switches to linear flow.` | Sublated Pivot: `Compare the baseline against a logarithmic terminal kernel with damage gating only the two additional coefficients, using leave-one-lab-out measurements before promoting it.`

```bash
cat > /tmp/rheology_check.py <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',p+'alpha_m97_c02_batch_07_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open(p+'public_alpha_m97_c02_batch_07.pkl','rb'))[0]; d=item['dataset']; g=item['gt']; tr=d['train']; te=d['test']; n=len(tr['lab_id']); v=m._prepare(d); t=d['eval_times']; dt=d['past_dt']; C=tr['strain_curves_obs']-tr['strain_0'][:,None]; R=tr['instant_rate_obs']; design=m._design

def stats(x):
    return np.round([np.median(abs(x)),np.quantile(abs(x),.95),np.sqrt(np.mean(x*x))],5)

def log_for_network(past, hold, drift, act, plateau, dt, times, tau):
    return design(past,hold,drift,act,np.ones_like(plateau),dt,times,tau)

for name, fn in [('linear_at_zero_damage',design),('log_at_zero_damage',log_for_network)]:
    print(name,flush=True); m._design=fn
    residual=np.empty_like(C); rate=np.empty_like(R)
    for lab in range(4):
        use=tr['lab_id']!=lab; x=[col[:n] for col in v]
        tau,w=m._fit(tuple(col[use] for col in x),dt,t,C[use],R[use]); a,b=fn(*(col[~use] for col in x),dt,t,tau)
        residual[~use]=a@w-C[~use]; rate[~use]=b@w-R[~use]
        print('LOLO fold',lab,'tau',np.round(tau,4),'strain late by activation',[stats(residual[(~use)&(v[-1][:n]==k),-3:]).tolist() for k in (False,True)],'rate',stats(rate[~use]))
    out=m.solve_alpha_m97_c02_batch_07(d); e=out['predicted_strain']-g['te_clean']; q=out['initial_rate_pred']-g['te_true_rate0'];
    print('full fit',out['estimated_taus'],out['hereditary_weights'],out['regime_mode_weights']); print('public error overall curve/rate',stats(e),stats(q));
    for lab in range(4):
        for mode in (False,True):
            mask=(te['lab_id']==lab)&(v[-1][n:]==mode)
            print('public',lab,mode,'strain',stats(e[mask]),'late signed median',round(float(np.median(e[mask,-3:])),5),'rate',stats(q[mask]))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_check.py
```
- **Container Exit Code**: `0` (`2249 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
86 rate [0.04158 0.06715 0.04578]
public 1 True strain [0.02178 0.06659 0.03352] late signed median -0.04069 rate [0.01909 0.04248 0.02622]
public 2 False strain [0.21961 1.41135 0.65949] late signed median 0.64336 rate [0.02653 0.10947 0.0593 ]
public 2 True strain [0.01485 0.24414 0.08655] late signed median -0.01995 rate [0.02289 0.12124 0.06198]
public 3 False strain [0.20701 1.24789 0.54407] late signed median 0.6422 rate [0.02711 0.18513 0.09574]
public 3 True strain [0.01751 0.24178 0.11253] late signed median 0.00072 rate [0.01511 0.14316 0.08877]
log_at_zero_damage
LOLO fold 0 tau [0.5669 4.1963] strain late by activation [[0.01629, 0.48437, 0.15582], [0.01803, 0.13943, 1.53225]] rate [0.01501 0.06237 0.04944]
LOLO fold 1 tau [0.5665 4.1001] strain late by activation [[0.01303, 0.03599, 0.01854], [0.01311, 0.39551, 0.80459]] rate [0.01155 0.04132 0.02067]
LOLO fold 2 tau [0.5659 4.0782] strain late by activation [[0.0149, 3.96813, 2.02956], [0.01279, 0.33027, 0.51077]] rate [0.01567 0.05086 0.03629]
LOLO fold 3 tau [0.5664 4.1317] strain late by activation [[0.01843, 0.31279, 0.26589], [0.01385, 0.58207, 3.80572]] rate [0.01218 0.08677 0.06078]
full fit [0.56632267 4.12344645] [ 0.68029019  0.54910693  0.13881786  0.15744033 -0.12896418  0.11746662
 -0.08828514] [0.26828293 0.25089685]
public error overall curve/rate [0.00306 0.137   0.07423] [0.00298 0.11586 0.05713]
public 0 False strain [0.00544 0.17125 0.06358] late signed median 0.00594 rate [0.00697 0.08798 0.04846]
public 0 True strain [0.00506 0.07795 0.02722] late signed median 0.00616 rate [0.0052  0.03936 0.02106]
public 1 False strain [0.00185 0.08355 0.03096] late signed median 4e-05 rate [0.00427 0.04153 0.02295]
public 1 True strain [0.00154 0.01313 0.00555] late signed median -0.00023 rate [0.00165 0.00756 0.00396]
public 2 False strain [0.0019  0.24191 0.0849 ] late signed median 0.00256 rate [0.00266 0.12315 0.0662 ]
public 2 True strain [0.00232 0.16851 0.05591] late signed median 0.00224 rate [0.00173 0.08155 0.04232]
public 3 False strain [0.00274 0.30495 0.10115] late signed median 0.00175 rate [0.00249 0.14578 0.07754]
public 3 True strain [0.00444 0.25994 0.11829] late signed median -0.00409 rate [0.00372 0.14165 0.09129]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=b72669af6db50b67 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_b72669af (directory).
```

### Turn 7 — `execute_in_remote_docker` (`43877 ms` LLM)
- **Rationale**: Promote the kernel supported by held-out measurements. Verify the saved artifact against an independent memory recurrence, the initial derivative, and public lab-by-activation slices; a disagreement would distinguish an implementation error from a remaining model error.
- **Alternatives Weighed**: `Retain linear terminal flow at zero damage` (rejected: The logarithmic kernel reduced the worst zero-damage late-horizon median absolute error across held-out labs from 0.74940 to 0.01843.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c02_batch_07_solver.py')
s = p.read_text()
old = '''    act = np.tanh(damage)
    plateau = (damage > 0) & (dataset['shear_banding_index'] <= 0.50)
    return past, hold, np.column_stack((np.ones(len(hold)), aligned)), act * plateau, plateau'''
new = '''    # The low-shear network sets the time kernel; damage gates only the
    # two additive regime terms, which vanish continuously at zero damage.
    low_shear = dataset['shear_banding_index'] <= 0.50
    act = np.tanh(damage) * low_shear
    return past, hold, np.column_stack((np.ones(len(hold)), aligned)), act, low_shear'''
assert old in s
s = s.replace(old, new).replace('drift, act, plateau, dt', 'drift, act, low_shear, dt').replace('np.where(plateau[:, None], np.log1p(times), times)', '(np.log1p(times) if low_shear else times)[None, :]')
s = s.replace('tuple(column[:n] for column in inputs)', '(*[column[:n] for column in inputs[:-1]], inputs[-1])').replace('*(column[n:] for column in inputs), dt, times, tau', '*(column[n:] for column in inputs[:-1]), inputs[-1], dt, times, tau')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',p+'alpha_m97_c02_batch_07_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open(p+'public_alpha_m97_c02_batch_07.pkl','rb'))[0]; d=item['dataset']; g=item['gt']; n=len(d['train']['lab_id']); out=m.solve_alpha_m97_c02_batch_07(d)
past,hold,features,act,low=m._prepare(d); dt=d['past_dt']; tau=out['estimated_taus']; w=np.r_[out['hereditary_weights'],out['regime_mode_weights']]
z=np.zeros((len(hold),2))
for x,step in zip(past.T,dt):
    a=np.exp(-step/tau)
    z=z*a+x[:,None]*(1-a)
drift=features@w[2:7]+w[8]*act*z[:,0]
amplitude=(hold[:,None]-z)*w[:2]; amplitude[:,0]+=w[7]*act*hold
relax=-np.expm1(-d['eval_times'][:,None]/tau)
expected=d['test']['strain_0'][:,None]+amplitude[n:]@relax.T+drift[n:,None]*np.log1p(d['eval_times'])
np.testing.assert_allclose(out['predicted_strain'],expected,atol=1e-10)
np.testing.assert_allclose(out['initial_rate_pred'],(amplitude/tau).sum(axis=1)[n:]+drift[n:],atol=1e-10)
for mode in (False,True):
    a,b=m._design(past,hold,features,act*mode,mode,dt,np.array([0.,1e-7]),tau)
    np.testing.assert_allclose(a[:,0]@w,0,atol=1e-12)
    np.testing.assert_allclose((a[:,1]@w)/1e-7,b@w,atol=2e-6,rtol=2e-6)
E=out['predicted_strain']-g['te_clean']; R=out['initial_rate_pred']-g['te_true_rate0']; T=d['test']; rmse=lambda x: np.sqrt(np.mean(x*x))
metrics=[]
for lab in range(4):
    for activated in (False,True):
        use=(T['lab_id']==lab)&((act[n:]>0)==activated)
        metrics.append((rmse(E[use]),rmse(R[use])))
        print('lab, active, n, strain RMSE, rate RMSE:',lab,activated,use.sum(),*metrics[-1])
assert max(x[0] for x in metrics)<.12
assert max(x[1] for x in metrics)<.10
for missing in (False,True):
    use=np.isnan(T['coupon_features_raw']).any(axis=1)==missing
    print('missing coupon, n, strain RMSE, rate RMSE:',missing,use.sum(),rmse(E[use]),rmse(R[use]))
print('overall strain/rate RMSE',rmse(E),rmse(R)); print('max stress calibration error',np.max(abs(past[n:]-g['te_hist_true'])),np.max(abs(hold[n:]-g['te_hold_true'])))
for key,reference in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]:
    print(key,'maximum absolute error',np.max(abs(out[key]-g[reference])))
PY
```
- **Container Exit Code**: `0` (`1181 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_07_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
lab, active, n, strain RMSE, rate RMSE: 0 False 5 0.0635820699958504 0.048457843821649124
lab, active, n, strain RMSE, rate RMSE: 0 True 11 0.027220354861113454 0.021061151734202112
lab, active, n, strain RMSE, rate RMSE: 1 False 5 0.03095741858323254 0.022952565334817543
lab, active, n, strain RMSE, rate RMSE: 1 True 13 0.0055503712499122525 0.003960627673247437
lab, active, n, strain RMSE, rate RMSE: 2 False 6 0.08490072327768482 0.06619730104845716
lab, active, n, strain RMSE, rate RMSE: 2 True 9 0.05591111965443005 0.04231620287301022
lab, active, n, strain RMSE, rate RMSE: 3 False 8 0.10114786099291215 0.07753736768226963
lab, active, n, strain RMSE, rate RMSE: 3 True 15 0.11829204454390961 0.09129280994951505
missing coupon, n, strain RMSE, rate RMSE: False 61 0.0049954326744041525 0.004511271649596095
missing coupon, n, strain RMSE, rate RMSE: True 11 0.18953517654273078 0.14576978635321017
overall strain/rate RMSE 0.07422578447452298 0.05712785157406797
max stress calibration error 0.0005314588289615507 0.00026815713367622784
estimated_taus maximum absolute error 0.020553545050191246
hereditary_weights maximum absolute error 0.002559666796768806
regime_mode_weights maximum absolute error 0.001717069350032463

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=34e42e97dce08721 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_34e42e97 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`28095 ms` LLM)
- **Rationale**: Public stress calibration is already accurate. This check looks for a distinct contract failure: a documented EMI spike being absorbed into the direct calibration line. If real archived sweeps expose it, fix spike rejection without changing the verified public fit; otherwise retain the current solver.
- **Alternatives Weighed**: `Tune imputation against the public test targets` (rejected: Missing-coupon rows dominate the remaining error, but the contract specifies cell-balanced mean imputation and test targets should not tune it.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, json, numpy as np
p='/holdout_task/alpha_m97_c02_batch_07_solver.py'; spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for sweep in json.loads(m._HISTORY.read_text()):
 for lab in range(4):
  x,y=m._channel(sweep,lab)
  if len(x)<3: continue
  clean=m._line(x,y,archived=True)
  if clean is None: continue
  A=np.column_stack((x,np.ones_like(x)))
  ref_error=np.abs(y-A@clean)
  if ref_error.max()<=3: continue
  local=m._line(x,y)
  print(sweep.get('episode_short_name'),lab,'rows',len(x),'clean',sum(ref_error<=.02),'EMI',np.round(ref_error[ref_error>3],3),'local-vs-clean max shift',None if local is None else round(float(np.max(abs(A@(local-clean)))),6))
PY
sed -n '/^def .*align/,/^def /p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_06_solver.py | head -55
```
- **Container Exit Code**: `0` (`585 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=34e42e97dce08721 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_34e42e97 (directory).
```

### Turn 9 — `submit_completed_task` (`18717 ms` LLM)
- **Rationale**: Leave-one-lab-out measurements support the logarithmic kernel, and the saved artifact passed clean-process schema and equation checks. The host receipt binds scientific and gauge verification to that artifact.
- **Alternatives Weighed**: `Revise calibration or imputation before submitting` (rejected: The archived sweep check did not establish a calibration defect, and the remaining missing-coupon error does not justify departing from the specified cell-balanced imputation.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_07_solver.py': 'FILE:7167'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_07_solver.py`. The public schema, independent memory recurrence, and initial-rate checks passed. Public strain and initial-rate RMSEs were 0.0742 and 0.0571 overall, rising to 0.1895 and 0.1458 for coupons with missing features.`

### Turn 10 — `execute_in_remote_docker` (`31975 ms` LLM)
- **Rationale**: If the basis loses rank or its derivative disagrees with the rate design, correct the equation before changing the fit. If excluding incomplete coupons during fitting materially improves held-out lab errors, test that challenger while keeping the required prediction-time imputation. Otherwise retain the current artifact.
- **Alternatives Weighed**: `Add rational, relay, or stochastic drift terms` (rejected: This batch defines a deterministic two-branch rheology equation and has no relay state or classification labels. The applicable checks are basis identifiability, derivative consistency, and signed residuals within held-out labs.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',p+'alpha_m97_c02_batch_07_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open(p+'public_alpha_m97_c02_batch_07.pkl','rb') as f: item=pickle.load(f)[0]
d=item['dataset']; tr=d['train']; te=d['test']; n=len(tr['lab_id']); dt=d['past_dt']; t=d['eval_times']; data=m._prepare(d); x=data[:-1]; low=data[-1]; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; ids=tr['lab_id']; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1)
for mode in (False,True):
    tau=np.array([.57,4.12]); a,b=m._design(*x[:-1],x[-1]*mode,mode,dt,t,tau); at,bt=m._design(*x[:-1],x[-1]*mode,mode,dt,np.array([0.,1e-7]),tau)
    np.testing.assert_allclose(at[:,0],0,atol=1e-12); np.testing.assert_allclose(at[:,1]/1e-7,b,atol=2e-6,rtol=2e-6); np.testing.assert_allclose(bt,b,atol=1e-12)
    if mode: assert np.linalg.matrix_rank(np.vstack((a[:n].reshape(-1,9),b[:n])))==9
print('All nine low-shear basis terms identifiable at nominal times; initial derivative verified for both kernel branches.')
for name in ('all_rows','complete_fit'):
    ep=np.empty_like(y); er=np.empty_like(r)
    for lab in np.unique(ids):
        held=ids==lab; fit=(~held)&(complete if name=='complete_fit' else True)
        tau,w=m._fit((*[v[:n][fit] for v in x],low),dt,t,y[fit],r[fit]); a,b=m._design(*(v[:n][held] for v in x),low,dt,t,tau); ep[held]=a@w-y[held]; er[held]=b@w-r[held]
    print(name,'LOLO complete lab slices (lab, late bias, late median absolute error, rate bias, rate median absolute error)')
    for lab in np.unique(ids):
        j=(ids==lab)&complete; late=ep[j,-3:]; rate=er[j]; print(lab,*np.round([np.median(late),np.median(abs(late)),np.median(rate),np.median(abs(rate))],6))
    print(name,'LOLO incomplete late/rate median absolute error',np.median(abs(ep[~complete,-3:])),np.median(abs(er[~complete])))
PY
```
- **Container Exit Code**: `0` (`2035 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
All nine low-shear basis terms identifiable at nominal times; initial derivative verified for both kernel branches.
all_rows LOLO complete lab slices (lab, late bias, late median absolute error, rate bias, rate median absolute error)
0 0.004285 0.016287 -0.001653 0.014025
1 0.001967 0.013009 0.00016 0.011297
2 -0.000836 0.011912 0.004483 0.0118
3 -0.002073 0.013658 0.006637 0.011713
all_rows LOLO incomplete late/rate median absolute error 0.11778871794939222 0.06459787223721902
complete_fit LOLO complete lab slices (lab, late bias, late median absolute error, rate bias, rate median absolute error)
0 0.003166 0.015152 -0.003737 0.014417
1 -0.000436 0.012251 -0.001122 0.012498
2 -0.000812 0.01249 0.00675 0.011975
3 -0.002356 0.013262 0.004206 0.011075
complete_fit LOLO incomplete late/rate median absolute error 0.1167996206786206 0.06658817191049315

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=34e42e97dce08721 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_34e42e97 (directory).
```

### Turn 11 — `submit_completed_task` (`15944 ms` LLM)
- **Rationale**: The saved artifact passed isolated Python verification, all nine low-shear basis terms were identifiable, and the initial derivative matched both kernel branches. The held-out lab audit found no material improvement from excluding incomplete coupons, so the verified solver is unchanged.
- **Alternatives Weighed**: `Fit only on coupons with complete features` (rejected: Leave-one-lab-out strain gains were small and mixed, and the median absolute rate error on incomplete coupons increased from 0.06460 to 0.06659.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_07_solver.py': 'FILE:7167'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_07_solver.py`. The public schema, memory recurrence, and initial-rate checks passed. Public strain and initial-rate RMSEs were 0.0742 and 0.0571 overall, and 0.1895 and 0.1458 on coupons with missing features.`

### Turn 12 — `execute_in_remote_docker` (`26852 ms` LLM)
- **Rationale**: The earlier EMI audit skipped sweeps with no internally identifiable clean line. This replay checks those real sparse sweeps against earlier clean calibrations and measures whether direct fitting absorbs spikes. Change the calibration only if the replay identifies a material shift supported by clean observations, not merely disagreement with an older episode.
- **Alternatives Weighed**: `Add rational terms or batch classification corrections` (rejected: The nine prescribed basis terms are identifiable, held-out lab residuals show little systematic bias, and the dataset has no classification or relay targets.)

```bash
set -e
cat > /tmp/rheology_calibration_replay.py <<'PY'
import importlib.util
import json
import numpy as np

path = '/holdout_task/alpha_m97_c02_batch_07_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
history = json.loads(m._HISTORY.read_text())
for i, sweep in enumerate(history):
    for lab in range(4):
        x, y = m._channel(sweep, lab)
        if len(x) < 2:
            continue
        A = np.column_stack((x, np.ones_like(x)))
        local = m._line(x, y)
        strict = m._line(x, y, archived=True) if len(x) >= 3 else None
        if strict is not None:
            clean = np.abs(y - A @ strict) <= .02
            if clean.sum() >= 4:
                shift = np.inf if local is None else np.max(np.abs(A @ (local - strict)))
                if shift > .02:
                    print('independently identified clean line differs from direct fit', i, lab, 'points', len(x), 'clean', clean.sum(), 'stress shift', shift)
        if strict is None:
            previous = next((p for h in reversed(history[:i]) if len(m._channel(h, lab)[0]) >= 4 and (p := m._line(*m._channel(h, lab), archived=True)) is not None), None)
            if previous is not None:
                residual = y - A @ previous
                if np.max(abs(residual)) > 3:
                    print('sparse EMI replay', i, lab, 'residuals vs previous clean sweep', np.round(residual, 4), 'direct fit accepted', local is not None)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheology_calibration_replay.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`864 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
sparse EMI replay 6 0 residuals vs previous clean sweep [-4.75 -4.55 -0.  ] direct fit accepted True
sparse EMI replay 6 3 residuals vs previous clean sweep [4.9    4.5999 0.    ] direct fit accepted True
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=34e42e97dce08721 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_34e42e97 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`41688 ms` LLM)
- **Rationale**: Two real three-point sweeps passed the direct fit despite spikes above 4.5. Replay both to verify that the change recovers the anchored line, then compare every public output against the verified baseline to catch any unintended effect on clean calibration or rheology predictions.
- **Alternatives Weighed**: `Replace any sparse calibration that disagrees with history` (rejected: A disagreement can reflect a real calibration shift; the fallback requires a point that agrees with the prior line and another that differs by more than 3.0.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c02_batch_07_solver.py')
s = p.read_text()
start = s.index('def _calibrate(sweep):')
end = s.index('\n\ndef _prepare', start)
s = s[:start] + '''def _calibrate(sweep):
    coeffs = []
    history = None
    for lab in range(4):
        x, y = _channel(sweep, lab)
        coeff = _line(x, y, archived=True)
        if coeff is None:
            # With two spikes in three rows, an in-batch fit can absorb both.
            # A prior line can identify that pattern if a clean point anchors it.
            if history is None:
                history = json.loads(_HISTORY.read_text())
            direct = _line(x, y)
            for past_sweep in reversed(history):
                px, py = _channel(past_sweep, lab)
                prior = _line(px, py, archived=True)
                if prior is None:
                    continue
                if direct is None:
                    direct = prior
                    break
                residual = np.abs(y - (prior[0] * x + prior[1]))
                clean = residual <= 0.02
                if len(px) >= 4 and clean.any() and (residual > 3.0).any():
                    direct = _line(x[clean], y[clean], archived=True)
                    if direct is None:
                        direct = prior
                    break
            if direct is None:
                raise ValueError(f'No clean calibration for load cell {lab}')
            coeff = direct
        coeffs.append(coeff)
    return np.asarray(coeffs)
''' + s[end:]
s = s.replace('def _line(x, y, archived=False):','def _line(x, y, strict=False):').replace('if archived else 3.0','if strict else 3.0').replace('archived=True','strict=True')
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, json, pickle
from pathlib import Path
import numpy as np
root = Path('/holdout_task')
def load(path):
    spec=importlib.util.spec_from_file_location('solver',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
path=root/'alpha_m97_c02_batch_07_solver.py'
m=load(path)
old=load(next(Path('/tmp/.hegel_candidate_archive/v02_34e42e97').rglob(path.name)))
history=json.loads(m._HISTORY.read_text())
item=pickle.loads((root/'public_alpha_m97_c02_batch_07.pkl').read_bytes())[0]
d=item['dataset']; g=item['gt']
for lab in (0,3):
    x,y=m._channel(history[6],lab)
    prior=next(c for h in reversed(history[:6]) if len(m._channel(h,lab)[0])>=4 and (c:=m._line(*m._channel(h,lab),strict=True)) is not None)
    sweep=dict(lab_id=np.full(len(x),lab),cmd_stress=x,true_stress=y)
    assert np.count_nonzero(np.abs(y-(prior[0]*x+prior[1]))>3)==2
    np.testing.assert_allclose(m._calibrate(sweep)[lab],prior,atol=1e-10,rtol=0)
    print('EMI replay lab',lab,'maximum line shift before/after',np.max(np.abs((old._calibrate(sweep)[lab]-prior)@[x,np.ones_like(x)])),np.max(np.abs((m._calibrate(sweep)[lab]-prior)@[x,np.ones_like(x)])))
a=m.solve_alpha_m97_c02_batch_07(d); b=old.solve_alpha_m97_c02_batch_07(d)
for k in a: np.testing.assert_allclose(a[k],b[k],rtol=0,atol=1e-9)
missing=np.isnan(d['test']['coupon_features_raw']).any(axis=1)
for k,t in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
    e=a[k]-g[t]; print(k,'overall/missing RMSE',np.sqrt(np.mean(e**2)),np.sqrt(np.mean(e[missing]**2)))
PY
```
- **Container Exit Code**: `0` (`1286 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_07_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
EMI replay lab 0 maximum line shift before/after 4.165677005168425 0.0
EMI replay lab 3 maximum line shift before/after 3.8576305528248596 0.0
predicted_strain overall/missing RMSE 0.07422578447452298 0.18953517654273078
initial_rate_pred overall/missing RMSE 0.05712785157406797 0.14576978635321017

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=42594982c0e6b99c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_42594982 (directory).
```

### Turn 14 — `submit_completed_task` (`16052 ms` LLM)
- **Rationale**: The final artifact passed the schema check in isolated Python, recovered the clean line in both real EMI replays, and matched every public output of the previously verified model within 1e-9.
- **Alternatives Weighed**: `Keep the previous calibration logic` (rejected: It accepted both archived sparse sweeps with EMI, shifting fitted stress by up to 4.17 and 3.86.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_07_solver.py`. The public schema passed, and the calibration fix corrected both archived EMI cases without changing public predictions. Public strain and initial-rate RMSEs were 0.0742 and 0.0571 overall, and 0.1895 and 0.1458 on coupons with missing features.`

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
