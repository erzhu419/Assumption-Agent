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
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `379,718` (`366,036` / `99,456` / `13,682`) |
| **Total Episode Wall Time (s)** | `447.05s` |

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

### Turn 1 — `execute_in_remote_docker` (`19977 ms` LLM)
- **Rationale**: Inspect the public array layout, validator, and archive conventions to detect mismatches that would change the calibration or design matrix before fitting.
- **Alternatives Weighed**: `Reuse the archived solver unchanged` (rejected: The archive may clarify alignment conventions, but the prior branch failed the public gate and the current task specifies a crystallization plateau.)

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && ls -l /tmp/hegel_continual_archive && sed -n '1,230p' /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference-panel alignment imputation'; python3 - <<'PY'
import pickle,numpy as np
p='/holdout_task/public_alpha_m97_c02_batch_01.pkl'
d=pickle.load(open(p,'rb'))
def show(x,depth=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print('  '*depth,k,end=': ')
   if isinstance(v,dict) and depth<2: print();show(v,depth+1)
   elif isinstance(v,np.ndarray): print(v.shape,v.dtype,repr(v.ravel()[:7]))
   elif isinstance(v,(list,tuple)): print(type(v).__name__,len(v),repr(v[:1])[:350])
   else: print(repr(v)[:350])
show(d)
PY
```
- **Container Exit Code**: `0` (`439 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
e weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 2 — `execute_in_remote_docker` (`20856 ms` LLM)
- **Rationale**: The first inspection was truncated. Check the exact target fields, archived panel convention, and per-channel calibration residuals; these determine the fit equations, missing-value treatment, and whether history is needed.

```bash
sed -n '1,190p' /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | head -25; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation variable projection' | head -8; python3 - <<'PY'
import pickle,numpy as np
items=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb')); print('items',len(items))
for item in items:
 d=item['dataset']; print('item keys',list(item),'dataset keys',list(d)); print('times',d['eval_times'],'dt',np.round(d['past_dt'],3),'index',d['shear_banding_index']);
 for key in ('train','test','calibration_load_cells'):
  s=d[key]; print(key,{k:(np.shape(v),str(np.asarray(v).dtype)) for k,v in s.items()}); L=s['lab_id']; print('counts',np.unique(L,return_counts=True))
  if key=='calibration_load_cells':
   for l in np.unique(L):
    x=s['cmd_stress'][L==l]; y=s['true_stress'][L==l]; a,b=np.polyfit(x,y,1); print(l,a,b,'max residual',max(abs(y-a*x-b)))
  else:
   X=s['coupon_features_raw']; print('nan',np.isnan(X).sum(0),'refs shape',np.shape(s['ref_coupons'])); print('targets',[(k,float(np.nanmin(v)),float(np.nanmax(v))) for k,v in s.items() if 'rate' in k or 'strain' in k]);
PY
```
- **Container Exit Code**: `0` (`544 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
BLE_SUBROUTINE(real_1d_step_series)
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
items 1
item keys ['dataset', 'gt'] dataset keys ['past_dt', 'eval_times', 'crit_stress', 'yield_stress', 'shear_banding_index', 'calibration_load_cells', 'train', 'test']
times [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ] dt [0.686 0.354 0.412 0.51  0.49  0.774 0.436 0.523 0.554 0.249 0.562 0.35
 0.679 0.318 0.778 0.777 0.754 0.205] index 0.12
train {'lab_id': ((180,), 'int64'), 'stratum': ((180,), 'int64'), 'coupon_features_raw': ((180, 4), 'float64'), 'ref_coupons': ((180, 32), 'float64'), 'past_stress_cmd': ((180, 18), 'float64'), 'stress_hold_cmd': ((180,), 'float64'), 'strain_0': ((180,), 'float64'), 'aux_chamber_telemetry': ((180, 3), 'float64'), 'strain_curves_obs': ((180, 10), 'float64'), 'instant_rate_obs': ((180,), 'float64')}
counts (array([0, 1, 2, 3]), array([41, 37, 52, 50]))
nan [3 9 3 5] refs shape (180, 32)
targets [('strain_0', -0.823836, 0.820409), ('strain_curves_obs', -17.781020196386567, 25.323727123410176), ('instant_rate_obs', -5.111013802613548, 4.849736995223994)]
test {'lab_id': ((72,), 'int64'), 'stratum': ((72,), 'int64'), 'coupon_features_raw': ((72, 4), 'float64'), 'ref_coupons': ((72, 32), 'float64'), 'past_stress_cmd': ((72, 18), 'float64'), 'stress_hold_cmd': ((72,), 'float64'), 'strain_0': ((72,), 'float64'), 'aux_chamber_telemetry': ((72, 3), 'float64')}
counts (array([0, 1, 2, 3]), array([17, 16, 16, 23]))
nan [1 0 2 1] refs shape (72, 32)
targets [('strain_0', -1.061798, 0.518879)]
calibration_load_cells {'lab_id': ((40,), 'int64'), 'cmd_stress': ((40,), 'float64'), 'true_stress': ((40,), 'float64')}
counts (array([0, 1, 2, 3]), array([10, 10, 10, 10]))
0 0.79004725175476 -0.13998828119953727 max residual 0.00019010847293882138
1 1.3099623181039106 0.16005601310626413 max residual 0.00019176057234507327
2 0.8499613249826024 0.10003695262801829 max residual 0.00012464239328242865
3 1.240001538252827 -0.12996202646637137 max residual 0.00014402498836060817
```

### Turn 3 — `execute_in_remote_docker` (`20602 ms` LLM)
- **Rationale**: Recover the exact panel convention before constructing the regression. Check whether the public measurements cover both damage regimes; an absent regime would change the validation slices and limit which coefficients can be identified.
- **Alternatives Weighed**: `Reuse the pre-gate calibration payload` (rejected: All four public channels have ten clean in-batch points, so the solver should estimate their calibrations directly.)

```bash
sed -n '58,175p' /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py; python3 - <<'PY'
import json, pickle, numpy as np
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
h=json.load(open(p)); print('history layout:',type(h).__name__,list(h)[:12] if isinstance(h,dict) else len(h)); print(str(h)[:650])
d=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]['dataset']; x=d['train']; c=d['calibration_load_cells']; ab=np.array([np.polyfit(c['cmd_stress'][c['lab_id']==l],c['true_stress'][c['lab_id']==l],1) for l in range(4)]); g,b=ab[x['lab_id']].T; s=x['past_stress_cmd']*g[:,None]+b[:,None]; D=np.maximum(np.abs(s)-d['crit_stress'],0)**2@d['past_dt']; print('damage quantiles:',np.quantile(D,[0,.1,.25,.5,.75,.9,1])); print('active by lab:',[(l,int(np.sum((D>0)&(x['lab_id']==l)))) for l in range(4)]); print('strata:',np.unique(x['stratum'],return_counts=True))
PY
```
- **Container Exit Code**: `0` (`436 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.9, 5.]])
    scale = np.r_[np.full(n*k, .05), np.full(n, .05)]
    trials = []
    for start in starts:
        d = design(start)
        w = np.linalg.lstsq(d, target, rcond=None)[0]
        trials.append(fit(np.r_[start, w], scale, 'soft_l1'))
    best = min(trials, key=lambda r: r.cost)
    err = residual(best.x)
    mad = lambda e: 1.4826*np.median(np.abs(e - np.median(e)))
    scale = np.r_[np.full(n*k, max(mad(err[:n*k]), 1.e-6)),
                  np.full(n, max(mad(err[n*k:]), 1.e-6))]
    result = fit(best.x, 2*scale, 'cauchy').x
    order = np.argsort(result[:2])
    return np.exp(result[:2][order]), np.r_[result[2:4][order], result[4:]]


def solve_alpha_m97_c02_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    times, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    calibration = _calibrate(dataset['calibration_load_cells'])
    aligned = _align(train, test)
    train_past, train_hold = _physical(train, calibration)
    taus, weights = _fit(train_past, train_hold, aligned[:n], dt, times,
                         train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs'])
    a, b = _basis(*_physical(test, calibration), dt, times, taus)
    drift = weights[2] + aligned[n:] @ weights[3:]
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2),
            'predicted_strain': test['strain_0'][:, None] + a @ weights[:2] + drift[:, None]*times,
            'initial_rate_pred': b @ weights[:2] + drift}
history layout: list 1
[{'episode_short_name': 'alpha_m97_c02_s00', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [1.619455, -1.307963, -0.646756, 0.95759, 2.133391, -0.419148, -0.546245, -1.403033, 1.416515, -0.474218, 0.56263, -1.431472, -1.102381, 0.100096, 0.332884, 0.978926, 1.410301, 2.014558, 2.037985, -1.520981, -1.132282, -0.400487, 1.428566, 2.184907, 2.114515, 2.024265, 0.373395, 1.839347, -1.383881, 0.252459, -1.650907, 2.152332, 0.673079, 0.633995, 2.133025, 0.157003, -0.816267, -1.031715, -0.029779, 1.574899], 'true_stress': [1.619593, -1.308033, -0.646
damage quantiles: [0.         0.         0.         3.07026187 3.72308391 4.16344929
 5.0070849 ]
active by lab: [(0, 30), (1, 24), (2, 31), (3, 35)]
strata: (array([0, 1, 2]), array([53, 66, 61]))
```

### Turn 4 — `execute_in_remote_docker` (`72029 ms` LLM)
- **Rationale**: Build the specified basis using the verified reference-panel convention. Check every public output for shape and finiteness, then compare any matching public reference fields to detect numerical or model mismatches before held-out lab validation.
- **Alternatives Weighed**: `Sort both fitted branches after optimization` (rejected: The crystallization terms use branch zero specifically, so swapping branch labels is not a symmetry of the active model.)

```bash
cat > /holdout_task/alpha_m97_c02_batch_01_solver.py <<'PY'
"""Load-cell calibration and two-branch strain-crystallization regression."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _historical_fit(lab):
    fits = []
    for sweep in json.loads(_HISTORY.read_text()):
        take = np.asarray(sweep['lab_id']) == lab
        x = np.asarray(sweep['cmd_stress'])[take]
        y = np.asarray(sweep['true_stress'])[take]
        if len(x) >= 2 and np.ptp(x) > 0:
            fits.append(_affine(x, y))
    return np.median(fits, axis=0)


def _calibrate(cal):
    fits = np.empty((4, 2))
    for lab in range(4):
        take = (cal['lab_id'] == lab) & np.isfinite(cal['cmd_stress']) & np.isfinite(cal['true_stress'])
        x, y = cal['cmd_stress'][take], cal['true_stress'][take]
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            use = x[i] != x[j]
            i, j = i[use], j[use]
            if len(i):
                slope = np.median((y[j] - y[i]) / (x[j] - x[i]))
                intercept = np.median(y - slope*x)
                clean = np.abs(y - (slope*x + intercept)) <= 3.
                x, y = x[clean], y[clean]
        if len(x) >= 2 and np.ptp(x) > 0:
            fits[lab] = _affine(x, y)
        else:
            fits[lab] = _historical_fit(lab)
            if len(x):
                fits[lab, 1] = np.median(y - fits[lab, 0]*x)
    return fits


def _align(train, test):
    labs, strata, raw, refs = (np.concatenate((train[k], test[k])) for k in
                              ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        take = labs == lab
        ref = refs[take]
        center = np.nanmedian(np.nanmean(ref, axis=0))
        scale = np.nanmedian(np.nanstd(ref, axis=0, ddof=0))
        aligned[take] = (raw[take] - center) / scale
    cell_means = []
    for lab, stratum in np.unique(np.column_stack((labs, strata)), axis=0):
        values = aligned[(labs == lab) & (strata == stratum)]
        counts = np.isfinite(values).sum(axis=0)
        cell_means.append(np.divide(np.nansum(values, axis=0), counts,
                                    out=np.full(raw.shape[1], np.nan), where=counts > 0))
    return np.where(np.isfinite(aligned), aligned, np.nanmean(cell_means, axis=0))


def _physical(split, calibration):
    gain, bias = calibration[split['lab_id']].T
    return (gain[:, None]*split['past_stress_cmd'] + bias[:, None],
            gain*split['stress_hold_cmd'] + bias)


class _Design:
    def __init__(self, past, hold, x, dt, times, crit, shear):
        self.past, self.hold, self.dt, self.times = past, hold, dt, times
        self.later = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
        damage = np.sum(np.maximum(np.abs(past)-crit, 0.)**2 * dt, axis=1)
        self.active = (damage > 0.) & (shear <= .50)
        self.act = np.where(self.active, np.tanh(damage), 0.)
        self.phi = np.where(self.active[:, None], np.log1p(times), times)
        self.drift = np.column_stack((np.ones(len(x)), x))

    def matrices(self, tau):
        n, k = self.phi.shape
        kernel = -np.expm1(-self.dt[:, None]/tau) * np.exp(-self.later[:, None]/tau)
        z = self.past @ kernel
        gap = self.hold[:, None] - z
        resp = -np.expm1(-self.times[:, None]/tau)
        u, v = self.act*self.hold, self.act*z[:, 0]
        a = np.empty((n, k, 9))
        a[:, :, :2] = gap[:, None, :] * resp
        a[:, :, 2:7] = self.drift[:, None, :] * self.phi[:, :, None]
        a[:, :, 7] = u[:, None] * resp[:, 0]
        a[:, :, 8] = v[:, None] * self.phi
        b = np.column_stack((gap/tau, self.drift, u/tau[0], v))
        return a, b


def _fit(design, curves, rates):
    target = np.r_[curves.ravel(), rates]
    valid = np.isfinite(target)
    is_rate = np.arange(len(target))[valid] >= curves.size
    target = target[valid]
    regime = np.any(design.active)
    p = 9 if regime else 7
    lo = np.log(min(design.dt.min(), design.times[design.times > 0].min()) / 30.)
    hi = np.log(design.times.max()*30.)

    def matrix(logtau):
        a, b = design.matrices(np.exp(logtau))
        return np.vstack((a.reshape(-1, 9), b))[valid, :p]

    def residual(params):
        return np.einsum('ij,j->i', matrix(params[:2]), params[2:]) - target

    def jac(params):
        d = matrix(params[:2])
        diff = []
        for step in ([1e-4, 0.], [0., 1e-4]):
            deriv = (matrix(params[:2]+step) - matrix(params[:2]-step)) / 2e-4
            diff.append(np.einsum('ij,j->i', deriv, params[2:]))
        return np.column_stack((*diff, d))

    def optimize(params, scale, loss):
        return least_squares(lambda q: residual(q)/scale, params,
                             jac=lambda q: jac(q)/scale[:, None], loss=loss,
                             bounds=(np.r_[[lo, lo], np.full(p, -np.inf)],
                                     np.r_[[hi, hi], np.full(p, np.inf)]),
                             x_scale='jac', max_nfev=100)

    scale = np.full(len(target), .05)
    trials = []
    for tau in ([.16, 1.1], [.38, 2.7], [.85, 5.5]):
        logtau = np.clip(np.log(tau), lo+1e-6, hi-1e-6)
        weights = np.linalg.lstsq(matrix(logtau), target, rcond=None)[0]
        trials.append(optimize(np.r_[logtau, weights], scale, 'soft_l1'))
    best = min(trials, key=lambda r: r.cost).x
    errors = residual(best)
    for mask in (is_rate, ~is_rate):
        e = errors[mask]
        if e.size:
            scale[mask] = 2.*max(1e-5, 1.4826*np.median(np.abs(e - np.median(e))))
    best = optimize(best, scale, 'cauchy').x
    taus = np.exp(best[:2])
    weights = np.zeros(9)
    weights[:p] = best[2:]
    if not regime and taus[0] > taus[1]:
        taus = taus[::-1]
        weights[:2] = weights[1::-1]
    return taus, weights


def solve_alpha_m97_c02_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    alignment = _align(train, test)
    calibration = _calibrate(dataset['calibration_load_cells'])
    common = (dataset['past_dt'], dataset['eval_times'], dataset['crit_stress'],
              dataset['shear_banding_index'])
    train_design = _Design(*_physical(train, calibration), alignment[:n], *common)
    taus, weights = _fit(train_design, train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs'])
    test_design = _Design(*_physical(test, calibration), alignment[n:], *common)
    curves, rates = test_design.matrices(taus)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('model','/holdout_task/alpha_m97_c02_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]; out=m.solve_alpha_m97_c02_batch_01(item['dataset']); gt=item['gt']
print('parameters', {k:v for k,v in out.items() if np.size(v)<=7}); print('public evaluation fields', {k:np.shape(v) for k,v in gt.items()})
for k,v in out.items():
 if k in gt and np.shape(v)==np.shape(gt[k]):
  e=v-np.asarray(gt[k]); print(k, 'rmse',np.sqrt(np.mean(e**2)), 'max abs',np.max(np.abs(e)))
PY
```
- **Container Exit Code**: `0` (`892 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
parameters {'estimated_taus': array([0.49877778, 3.60545009]), 'hereditary_weights': array([ 0.66961591,  0.54055112,  0.12256782,  0.13483478, -0.11310404,
        0.10607928, -0.07537934]), 'regime_mode_weights': array([0.2861248 , 0.23359868])}
public evaluation fields {'taus': (2,), 'theta': (7,), 'vp_theta': (2,), 'stress_gains': (4,), 'stress_biases': (4,), 'te_clean': (72, 10), 'te_true_rate0': (72,), 'te_hist_true': (72, 18), 'te_hold_true': (72,)}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=423976fc99829ff3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_423976fc (directory).
```

### Turn 5 — `execute_in_remote_docker` (`42339 ms` LLM)
- **Rationale**: Compare all five outputs against public references without fitting to them. Refit by held-out lab to expose cross-lab errors, and test the recurrence, shear boundary, and sparse or spiked calibration. A failure will identify whether to revisit the physical basis, panel alignment, or calibration.
- **Alternatives Weighed**: `Submit after schema validation` (rejected: Shape and finiteness checks cannot detect lab-specific bias, a rate derivative error, or a calibration failure after an EMI spike.)

```bash
cat > /tmp/verify_rheology_batch01.py <<'PY'
import importlib.util
import pickle
import numpy as np

path = '/holdout_task/alpha_m97_c02_batch_01_solver.py'
spec = importlib.util.spec_from_file_location('candidate', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_alpha_m97_c02_batch_01.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, gt = item['dataset'], item['gt']
out = m.solve_alpha_m97_c02_batch_01(d)
rmse = lambda e: np.sqrt(np.mean(e**2))
cal = m._calibrate(d['calibration_load_cells'])
common = (d['past_dt'], d['eval_times'], d['crit_stress'], d['shear_banding_index'])
for key, ref in [('estimated_taus','taus'), ('hereditary_weights','theta'), ('regime_mode_weights','vp_theta'), ('predicted_strain','te_clean'), ('initial_rate_pred','te_true_rate0')]:
    err = out[key] - gt[ref]
    print(key, 'rmse', rmse(err), 'max_abs', np.max(np.abs(err)))
    if err.size < 10:
        print('  signed errors', err)
test = d['test']
past, hold = m._physical(test, cal)
print('calibrated stress max abs', np.max(np.abs(past-gt['te_hist_true'])), np.max(np.abs(hold-gt['te_hold_true'])))
z = np.zeros((len(hold), 2))
tau = out['estimated_taus']
for step, dt in enumerate(d['past_dt']):
    decay = np.exp(-dt/tau)
    z = decay*z + (1-decay)*past[:, step, None]
active = ((np.maximum(abs(past)-d['crit_stress'],0)**2)@d['past_dt']) > 0
for lab in range(4):
    for mode in (False, True):
        take = (test['lab_id'] == lab) & (active == mode)
        if take.any():
            print('public lab/active', lab, int(mode), 'n', take.sum(), 'curve/rate RMSE', rmse((out['predicted_strain']-gt['te_clean'])[take]), rmse((out['initial_rate_pred']-gt['te_true_rate0'])[take]))
x = m._align(d['train'], test)[-len(hold):]
a,b = m._Design(past, hold, x, *common).matrices(tau)
assert np.allclose(b[:, :2], (hold[:,None]-z)/tau, atol=1e-12, rtol=0)
for shear in (.50, .50001):
    curve0, rate = m._Design(past, hold, x, common[0], np.array([0., 1.e-7]), common[2], shear).matrices(tau)
    assert np.all(curve0[:, 0] == 0.)
    assert np.allclose(curve0[:, 1]/1.e-7, rate, atol=8e-6, rtol=1e-6)
    if shear > .50:
        assert np.all(curve0[:, :, 7:] == 0) and np.all(rate[:, 7:] == 0)
train = d['train']
for lab in range(4):
    keep = train['lab_id'] != lab
    ns = {k:v[keep] for k,v in train.items()}
    cv = dict(d, train=ns, test={k:np.concatenate((v[~keep],test[k])) for k,v in train.items() if k in test})
    pred = m.solve_alpha_m97_c02_batch_01(cv)
    n = int((~keep).sum())
    ec = pred['predicted_strain'][:n]-train['strain_curves_obs'][~keep]
    er = pred['initial_rate_pred'][:n]-train['instant_rate_obs'][~keep]
    print('held-out lab', lab, 'curve/rate RMSE', rmse(ec), rmse(er), 'curve/rate median abs', np.median(abs(ec)), np.median(abs(er)), 'worst horizon RMSE', np.max(np.sqrt(np.mean(ec**2, axis=0))))
rows = d['calibration_load_cells']
spiked = {k:v.copy() for k,v in rows.items()}
for lab in range(4):
    idx = np.flatnonzero(rows['lab_id'] == lab)
    spiked['true_stress'][idx[np.argmin(rows['cmd_stress'][idx])]] += 5.
spike_err = np.max(abs(m._calibrate(spiked)-cal))
print('single EMI per channel: max coefficient shift', spike_err)
assert spike_err < .001
for num in (0, 1):
    keep = rows['lab_id'] != 0
    keep[np.flatnonzero(~keep)[:num]] = True
    fit = m._calibrate({k:v[keep] for k,v in rows.items()})
    assert np.all(np.isfinite(fit)) and np.allclose(fit[1:], cal[1:])
    print('history fallback, lab 0 in-batch rows', num, 'coefficients', fit[0])
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_rheology_batch01.py
```
- **Container Exit Code**: `0` (`1078 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
estimated_taus rmse 0.15525318485430983 max_abs 0.21954991357298947
  signed errors [-0.00222222 -0.21954991]
hereditary_weights rmse 0.016156836738786165 max_abs 0.02516522358881429
  signed errors [-0.01038409 -0.00944888 -0.01743218 -0.02516522  0.01689596 -0.01392072
  0.01462066]
regime_mode_weights rmse 0.006264585895150012 max_abs 0.0064013244451777696
  signed errors [ 0.0061248  -0.00640132]
predicted_strain rmse 0.206397909992626 max_abs 1.2470759761493362
initial_rate_pred rmse 0.04700419142256654 max_abs 0.2630901070707481
calibrated stress max abs 0.0002544414505800674 0.00013367516081164155
public lab/active 0 0 n 7 curve/rate RMSE 0.3536285536525965 0.029451542782213777
public lab/active 0 1 n 10 curve/rate RMSE 0.053582678112173494 0.04437771137440398
public lab/active 1 0 n 3 curve/rate RMSE 0.34978744678734863 0.01833306275196692
public lab/active 1 1 n 13 curve/rate RMSE 0.10320977653149496 0.07862821028586428
public lab/active 2 0 n 2 curve/rate RMSE 0.2460611578171322 0.025631642210923518
public lab/active 2 1 n 14 curve/rate RMSE 0.05167995578873414 0.04025157457701313
public lab/active 3 0 n 12 curve/rate RMSE 0.34989143960864977 0.0309659077683714
public lab/active 3 1 n 11 curve/rate RMSE 0.0522397378783579 0.038645353723325226
held-out lab 0 curve/rate RMSE 1.020580778794555 0.08131949708296365 curve/rate median abs 0.056310322853926675 0.04539323266522688 worst horizon RMSE 2.9913440580649597
held-out lab 1 curve/rate RMSE 1.409346593132807 0.04924450018525309 curve/rate median abs 0.040112925554714796 0.029657567719988398 worst horizon RMSE 4.205371858084186
held-out lab 2 curve/rate RMSE 0.9534514630828842 0.030473125334305528 curve/rate median abs 0.031709269228123005 0.015002035030778238 worst horizon RMSE 2.5967246736950886
held-out lab 3 curve/rate RMSE 0.3828833795500343 0.09882879204163054 curve/rate median abs 0.08375800468653949 0.07105957320180961 worst horizon RMSE 0.8327197576933167
single EMI per channel: max coefficient shift 1.1380833699087267e-05
history fallback, lab 0 in-batch rows 0 coefficients [9.99976886e-01 1.27323341e-06]
history fallback, lab 0 in-batch rows 1 coefficients [0.99997689 0.0712195 ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=423976fc99829ff3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_423976fc (directory).
```

### Turn 6 — `execute_in_remote_docker` (`28976 ms` LLM)
- **Rationale**: The largest clean-reference errors occur on inactive coupons, while initial-rate errors remain smaller. Hold the public reference parameters and physical test stresses fixed for diagnosis only, then compare linear and logarithmic drift by damage and missing-feature status. This separates a wrong time basis from panel alignment and observation outliers before editing the solver.
- **Alternatives Weighed**: `Retune the robust loss immediately` (rejected: A loss change could downweight systematic errors from the wrong time basis without correcting them.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/alpha_m97_c02_batch_01_solver.py'; s=importlib.util.spec_from_file_location('m',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]; d,g=i['dataset'],i['gt']; w=np.r_[g['theta'],g['vp_theta']]; tau=g['taus']; x=m._align(d['train'],d['test']); n=len(d['train']['lab_id']); common=(d['past_dt'],d['eval_times'],d['crit_stress'],d['shear_banding_index']); t=common[1]; cal=m._calibrate(d['calibration_load_cells']); rmse=lambda e: np.sqrt(np.mean(e**2))
for k, features in [('train',x[:n]),('test',x[n:])]:
    r=d[k]; past,hold=(g['te_hist_true'],g['te_hold_true']) if k=='test' else m._physical(r,cal); model=m._Design(past,hold,features,*common); y=(g['te_clean'] if k=='test' else r['strain_curves_obs'])-r['strain_0'][:,None]; rate=g['te_true_rate0'] if k=='test' else r['instant_rate_obs']; has_nan=np.isnan(r['coupon_features_raw']).any(axis=1); a,b=model.matrices(tau)
    for active in (False,True):
        for missing in (False,True):
            q=(model.active==active)&(has_nan==missing)
            if not q.any(): continue
            print(k,'active',active,'missing',missing,'n',q.sum(),'rate RMSE',rmse((b@w-rate)[q]))
            for name,phi in [('current',model.phi),('log',np.broadcast_to(np.log1p(t),model.phi.shape)),('linear',np.broadcast_to(t,model.phi.shape))]:
                e=(a@w+(phi-model.phi)*(model.drift@w[2:7])[:,None]-y)[q]
                print(name,'rmse',rmse(e),'median abs',np.median(abs(e)),'horizon median',np.round(np.median(e,axis=0),5))
    base=(a[:,:,:2]@w[:2]+a[:,:,7]*w[7]+a[:,:,8]*w[8]); q=(~model.active)&(~has_nan); drift=(model.drift@w[2:7]); use=q&(abs(drift)>.05)
    if use.any(): print(k,'inactive implied time basis',np.round(np.median(((y-base)/np.where(abs(drift)>.05,drift,np.nan)[:,None])[use],axis=0),6),'log',np.round(np.log1p(t),6))
PY
```
- **Container Exit Code**: `0` (`477 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[-0.01898 -0.0197  -0.03133 -0.03863 -0.03639  0.04367 -0.04955 -0.04623
 -0.03813 -0.04237]
train inactive implied time basis [0.282932 0.586842 0.850739 1.060358 1.266114 1.350693 1.47722  1.601366
 1.714104 1.780958] log [0.262364 0.612154 0.870833 1.076128 1.2464   1.391829 1.518791 1.631415
 1.732646 1.824549]
test active False missing False n 24 rate RMSE 0.006293145249535681
current rmse 0.4409151701162223 median abs 0.16418459070210864 horizon median [0.00048 0.00424 0.01009 0.01717 0.02505 0.0335  0.04237 0.05158 0.06104
 0.07072]
log rmse 0.008241366258862614 median abs 0.0035953561104842408 horizon median [-0.00041 -0.00095 -0.00135 -0.00167 -0.00193 -0.00215 -0.00235 -0.00253
 -0.00268 -0.00282]
linear rmse 0.4409151701162223 median abs 0.16418459070210864 horizon median [0.00048 0.00424 0.01009 0.01717 0.02505 0.0335  0.04237 0.05158 0.06104
 0.07072]
test active True missing False n 45 rate RMSE 0.007854821995047351
current rmse 0.010286504187096573 median abs 0.004179612364053509 horizon median [-0.00038 -0.00088 -0.00126 -0.00155 -0.0018  -0.00201 -0.00219 -0.00236
 -0.0025  -0.00264]
log rmse 0.010286504187096573 median abs 0.004179612364053509 horizon median [-0.00038 -0.00088 -0.00126 -0.00155 -0.0018  -0.00201 -0.00219 -0.00236
 -0.0025  -0.00264]
linear rmse 0.5765745033835705 median abs 0.23192635987807542 horizon median [0.00207 0.01285 0.0287  0.04751 0.06827 0.09042 0.11359 0.13755 0.16215
 0.18727]
test active True missing True n 3 rate RMSE 0.1501886924490657
current rmse 0.19668384779514958 median abs 0.09928141446111938 horizon median [-0.01361 -0.03176 -0.04518 -0.05583 -0.06467 -0.07221 -0.0788  -0.08464
 -0.08989 -0.09466]
log rmse 0.19668384779514958 median abs 0.09928141446111938 horizon median [-0.01361 -0.03176 -0.04518 -0.05583 -0.06467 -0.07221 -0.0788  -0.08464
 -0.08989 -0.09466]
linear rmse 0.3099267256836605 median abs 0.14151618474039898 horizon median [-0.0188  -0.06378 -0.11662 -0.11592 -0.09438 -0.06271 -0.02349  0.02157
  0.07129  0.12481]
test inactive implied time basis [0.255704 0.596614 0.848727 1.04881  1.21476  1.356497 1.480236 1.590001
 1.688663 1.778233] log [0.262364 0.612154 0.870833 1.076128 1.2464   1.391829 1.518791 1.631415
 1.732646 1.824549]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=423976fc99829ff3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_423976fc (directory).
```

### Turn 7 — `execute_in_remote_docker` (`23697 ms` LLM)
- **Rationale**: Apply the smallest change supported by the train and public reference diagnostics. Rerun the lab holdouts and public slices to measure the improvement after refitting and see whether missing covariates remain the limiting slice.
- **Alternatives Weighed**: `Change the robust-loss tuning without changing the basis` (rejected: The inactive-coupon error persists at fixed reference parameters and is removed by changing the time basis, not by changing fit weights.)
- **ATMS NOGOOD DECLARED**: Refuted `Zero work damage restores linear background drift in a low-shear campaign.` | Witness: `On 24 complete inactive public test coupons, holding reference parameters and stresses fixed, logarithmic drift reduced curve RMSE from 0.4409 to 0.00824. Complete inactive training coupons independently favored logarithmic drift, with median absolute error falling from 0.1738 to 0.0120.` | Blanket Diagnosis: `The implementation used a single joint shear-and-damage gate for both the background time basis and the two activation-dependent correction terms.` | Sublated Pivot: `Use logarithmic background drift throughout the low-shear campaign while retaining the damage gate on the two correction terms. Keep linear drift above the shear threshold.`

```bash
sed -i 's/self.phi = np.where(self.active\[:, None\], np.log1p(times), times)/# Shear selects the campaign drift law; damage gates only its nonlinear corrections.\n        phi = np.log1p(times) if shear <= .50 else times\n        self.phi = np.broadcast_to(phi, (len(x), len(times)))/' /holdout_task/alpha_m97_c02_batch_01_solver.py
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_rheology_batch01.py
```
- **Container Exit Code**: `0` (`1170 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
alid": true, "errors": []}
estimated_taus rmse 0.06270515765219316 max_abs 0.08867096557684251
  signed errors [-0.00115475 -0.08867097]
hereditary_weights rmse 0.002570316144011305 max_abs 0.0036486884426056854
  signed errors [-0.00364869 -0.00188171 -0.00103407 -0.00364396  0.00236276 -0.0024921
  0.00180305]
regime_mode_weights rmse 0.0004655786660673418 max_abs 0.0006554009213024514
  signed errors [-6.30604547e-05  6.55400921e-04]
predicted_strain rmse 0.040488371743374336 max_abs 0.4426807563317808
initial_rate_pred rmse 0.03082131901576521 max_abs 0.2394800518624629
calibrated stress max abs 0.0002544414505800674 0.00013367516081164155
public lab/active 0 0 n 7 curve/rate RMSE 0.0036940638021969023 0.0030471472881563348
public lab/active 0 1 n 10 curve/rate RMSE 0.03549995527081091 0.02796303727225444
public lab/active 1 0 n 3 curve/rate RMSE 0.0038887014060209335 0.004384924551931334
public lab/active 1 1 n 13 curve/rate RMSE 0.08964431278088734 0.06785764100791679
public lab/active 2 0 n 2 curve/rate RMSE 0.0013254730439578887 0.0020932477832883885
public lab/active 2 1 n 14 curve/rate RMSE 0.0013982787735033555 0.002058779037627718
public lab/active 3 0 n 12 curve/rate RMSE 0.00463945728603265 0.003821682232331882
public lab/active 3 1 n 11 curve/rate RMSE 0.006930296319313148 0.005646677714246789
held-out lab 0 curve/rate RMSE 1.0259799111975558 0.05552303989067738 curve/rate median abs 0.01346865439352779 0.012210494230401103 worst horizon RMSE 3.058198991724767
held-out lab 1 curve/rate RMSE 1.3687561368960597 0.04745645352364808 curve/rate median abs 0.013082634189797587 0.01694926760861115 worst horizon RMSE 4.166416117348355
held-out lab 2 curve/rate RMSE 0.9062702841591183 0.02538134242459139 curve/rate median abs 0.011823958655991285 0.013336126367484824 worst horizon RMSE 2.558466072370326
held-out lab 3 curve/rate RMSE 0.3405774628893469 0.04956954534370801 curve/rate median abs 0.014302156335461197 0.013350824172821296 worst horizon RMSE 0.7849418512773313
single EMI per channel: max coefficient shift 1.1380833699087267e-05
history fallback, lab 0 in-batch rows 0 coefficients [9.99976886e-01 1.27323341e-06]
history fallback, lab 0 in-batch rows 1 coefficients [0.99997689 0.0712195 ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=df7d651f8637e1ca (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_df7d651f (directory).
```

### Turn 8 — `execute_in_remote_docker` (`31190 ms` LLM)
- **Rationale**: The revised drift law removes the inactive-coupon error. Check whether the remaining held-out horizon RMSE is driven by sparse bad readings or a systematic residual across many coupons. This determines whether to adjust fit weighting or revisit the physical basis.
- **Alternatives Weighed**: `Replace cell-balanced imputation with a learned predictor` (rejected: The task specifies cell-balanced column means; first confirm the archived convention and determine whether missing-feature rows should influence physical parameter estimation.)

```bash
sed -n '/def _align/,/^def _/p' /tmp/hegel_continual_archive/alpha_m97_c02_s00_solver.py; OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/alpha_m97_c02_batch_01_solver.py'; s=importlib.util.spec_from_file_location('m',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]; d,g=i['dataset'],i['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); x=m._align(tr,te); cal=m._calibrate(d['calibration_load_cells']); common=tuple(d[k] for k in ('past_dt','eval_times','crit_stress','shear_banding_index')); des=m._Design(*m._physical(tr,cal),x[:n],*common); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; tau,w=m._fit(des,y,r); a,b=des.matrices(tau); ec=a@w-y; er=b@w-r; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1)
print('complete train curve residual abs quantiles by horizon:',np.quantile(abs(ec[complete]),[.5,.8,.9,.95,1],axis=0).round(4)); print('complete train rate residual abs quantiles:',np.quantile(abs(er[complete]),[.5,.8,.9,.95,1]).round(4)); print('curve vs rate MAD by labs',[(l,np.median(abs(ec[complete&(tr['lab_id']==l)])),np.median(abs(er[complete&(tr['lab_id']==l)]))) for l in range(4)]); print('largest curve residual coordinates',np.dstack(np.unravel_index(np.argsort(abs(ec).ravel())[-15:],ec.shape))[0]); print('taus',tau,'relative error',tau/g['taus']-1)
for name,sp,start in [('train',tr,0),('test',te,n)]:
 miss=~np.isfinite(sp['coupon_features_raw']); print(name,'missing by column',miss.sum(axis=0)); print(name,'missing lab/stratum/columns',[(int(sp['lab_id'][j]),int(sp['stratum'][j]),np.flatnonzero(miss[j]).tolist()) for j in np.flatnonzero(miss.any(axis=1))]); print(name,'complete aligned feature correlation',np.corrcoef(x[start:start+len(miss)][~miss.any(axis=1)].T).round(2))
PY
```
- **Container Exit Code**: `0` (`1495 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ansum(values, axis=0), counts,
                                    out=np.full(raw.shape[1], np.nan), where=counts > 0))
    fill = np.nanmean(cell_means, axis=0)
    return np.where(np.isfinite(aligned), aligned, fill)


def _physical(split, calibration):
complete train curve residual abs quantiles by horizon: [[1.15000e-02 9.00000e-03 1.09000e-02 1.23000e-02 1.10000e-02 1.06000e-02
  1.08000e-02 1.34000e-02 1.12000e-02 1.15000e-02]
 [2.52000e-02 1.96000e-02 2.23000e-02 2.38000e-02 2.27000e-02 2.25000e-02
  2.07000e-02 2.64000e-02 2.16000e-02 2.44000e-02]
 [3.51000e-02 2.98000e-02 3.13000e-02 4.03000e-02 4.12000e-02 2.92000e-02
  2.68000e-02 3.41000e-02 3.37000e-02 3.13000e-02]
 [1.34100e-01 4.18000e-02 1.06800e-01 4.16300e-01 5.32000e-01 2.48000e-01
  3.95000e-02 7.49000e-02 2.45200e-01 1.92900e-01]
 [7.08600e-01 7.69220e+00 1.61340e+00 3.03100e+00 4.01550e+00 2.52760e+01
  8.39300e-01 2.96160e+00 1.84234e+01 1.77352e+01]]
complete train rate residual abs quantiles: [0.0122 0.0224 0.0293 0.0316 0.0577]
curve vs rate MAD by labs [(0, 0.011416742219555381, 0.009376506387113737), (1, 0.011312715263787132, 0.013471237314934159), (2, 0.01060525782033106, 0.012230988164723441), (3, 0.011503148731324409, 0.012312673704242605)]
largest curve residual coordinates [[113   9]
 [  2   3]
 [143   9]
 [123   9]
 [103   7]
 [127   3]
 [ 35   4]
 [ 53   4]
 [  0   6]
 [ 26   8]
 [ 48   1]
 [142   9]
 [ 51   9]
 [ 45   8]
 [139   5]]
taus [0.49984525 3.73632903] relative error [-0.0023049  -0.02318195]
train missing by column [3 9 3 5]
train missing lab/stratum/columns [(0, 1, [2]), (3, 0, [1]), (0, 1, [0]), (2, 0, [1]), (2, 2, [1]), (3, 0, [3]), (0, 0, [1]), (1, 1, [3]), (3, 1, [0]), (3, 1, [1]), (2, 1, [2]), (1, 1, [1]), (2, 0, [1]), (0, 0, [3]), (3, 1, [1]), (0, 1, [3]), (1, 1, [1]), (2, 2, [3]), (1, 1, [0, 2])]
train complete aligned feature correlation [[ 1.    0.05 -0.06 -0.14]
 [ 0.05  1.    0.13 -0.05]
 [-0.06  0.13  1.    0.07]
 [-0.14 -0.05  0.07  1.  ]]
test missing by column [1 0 2 1]
test missing lab/stratum/columns [(0, 0, [2]), (1, 0, [0]), (1, 1, [2, 3])]
test complete aligned feature correlation [[ 1.   -0.07  0.07  0.06]
 [-0.07  1.   -0.08  0.16]
 [ 0.07 -0.08  1.    0.01]
 [ 0.06  0.16  0.01  1.  ]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=df7d651f8637e1ca (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_df7d651f (directory).
```

### Turn 9 — `execute_in_remote_docker` (`37679 ms` LLM)
- **Rationale**: Test whether imputed training covariates or gross strain outliers distort the estimated spectrum. Compare three fits without editing the deliverable, using held-out lab metrics and the public clean references for evaluation only. Keep the existing fit unless a challenger improves the limiting metric without materially degrading other outputs.
- **Alternatives Weighed**: `Replace cell-balanced imputation` (rejected: The specified imputation is correct as implemented, and the observed feature correlations offer little support for a learned replacement.); `Select a fit by raw held-out curve RMSE` (rejected: Isolated strain readings exceed 25 units of residual while median errors are near 0.01; raw RMSE mostly measures those readings rather than fit quality.)

```bash
cat > /tmp/compare_rheology_fits.py <<'PY'
import importlib.util
import pickle
import numpy as np

p = '/holdout_task/alpha_m97_c02_batch_01_solver.py'
s = importlib.util.spec_from_file_location('model', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
with open('/holdout_task/public_alpha_m97_c02_batch_01.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, gt = item['dataset'], item['gt']
tr, te = d['train'], d['test']
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
cal = m._calibrate(d['calibration_load_cells'])
x = m._align(tr, te)
common = tuple(d[k] for k in ('past_dt', 'eval_times', 'crit_stress', 'shear_banding_index'))
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
test_complete = np.isfinite(te['coupon_features_raw']).all(axis=1)

def design(split, features, mask):
    return m._Design(*(v[mask] for v in m._physical(split, cal)), features[mask], *common)

n = len(y)
train_design = design(tr, x[:n], np.ones(n, bool))
test_design = design(te, x[n:], np.ones(len(te['lab_id']), bool))
rmse = lambda z: np.sqrt(np.mean(np.square(z)))

for name in ('baseline','complete','complete_refit'):
    errors = []
    for lab in (-1, 0, 1, 2, 3):
        take = (tr['lab_id'] != lab) if lab >= 0 else np.ones(n, bool)
        if name != 'baseline':
            take &= complete
        model = design(tr, x[:n], take)
        yy, rr = y[take], r[take]
        tau, w = m._fit(model, yy, rr)
        if name == 'complete_refit':
            a, b = model.matrices(tau)
            def retain(obs, fit):
                e = obs-fit
                center = np.nanmedian(e)
                sigma = 1.4826*np.nanmedian(abs(e-center))
                return np.where(abs(e-center) <= 4*max(sigma, 1e-5), obs, np.nan)
            a_target, b_target = retain(yy, a@w), retain(rr, b@w)
            tau, w = m._fit(model, a_target, b_target)
        if lab < 0:
            a,b = test_design.matrices(tau)
            ec = te['strain_0'][:,None] + a@w - gt['te_clean']
            er = b@w - gt['te_true_rate0']
            print(name, 'public tau max rel err', np.max(abs(tau/gt['taus']-1)), 'hw/rw max abs', np.max(abs(w[:7]-gt['theta'])), np.max(abs(w[7:]-gt['vp_theta'])), 'strain/rate RMSE', rmse(ec),rmse(er), 'complete', rmse(ec[test_complete]),rmse(er[test_complete]))
        else:
            q = (tr['lab_id'] == lab) & complete
            a,b = train_design.matrices(tau)
            ec, er = (a@w-y)[q], (b@w-r)[q]
            errors.append([np.median(abs(ec)),np.median(abs(er)),rmse(np.clip(ec,-.06,.06)),rmse(er)])
    print(name, 'worst held-out lab complete rows: curve/rate median abs, capped curve RMSE, rate RMSE', np.max(errors,axis=0))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/compare_rheology_fits.py
```
- **Container Exit Code**: `0` (`1971 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
baseline public tau max rel err 0.02318195178479543 hw/rw max abs 0.0036486884426056854 0.0006554009213024514 strain/rate RMSE 0.040488371743374336 0.03082131901576521 complete 0.004525432107175521 0.003990288383749685
baseline worst held-out lab complete rows: curve/rate median abs, capped curve RMSE, rate RMSE [0.01264359 0.01426821 0.02343482 0.01966831]
complete public tau max rel err 0.014484112857174991 hw/rw max abs 0.003695331930499368 0.0008514479486774196 strain/rate RMSE 0.0404962840418289 0.030883370391337504 complete 0.004487288496527988 0.0037834844087777063
complete worst held-out lab complete rows: curve/rate median abs, capped curve RMSE, rate RMSE [0.01247964 0.01268068 0.02340516 0.01955569]
complete_refit public tau max rel err 0.013909121324842566 hw/rw max abs 0.003791192170552321 0.0008300995379121878 strain/rate RMSE 0.04049503008146135 0.030884645770418422 complete 0.00448258480184669 0.003766688329245364
complete_refit worst held-out lab complete rows: curve/rate median abs, capped curve RMSE, rate RMSE [0.01243983 0.01245253 0.02340564 0.01953465]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=df7d651f8637e1ca (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_df7d651f (directory).
```

### Turn 10 — `execute_in_remote_docker` (`28353 ms` LLM)
- **Rationale**: Use complete coupons to estimate physical parameters while retaining the required cell-balanced imputation for predictions. The challenger improves spectrum recovery and held-out rate error with essentially unchanged overall strain RMSE. Verify that the installed solver reproduces those gains and preserves the schema, lab-by-regime performance, shear boundary, and calibration fallback before submitting.
- **Alternatives Weighed**: `Refit after clipping gross residuals` (rejected: The additional pass yielded only small gains over the complete-coupon robust fit and did not improve the primary strain error materially.)

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/alpha_m97_c02_batch_01_solver.py')
s = p.read_text()
old = '''    train_design = _Design(*_physical(train, calibration), alignment[:n], *common)
    taus, weights = _fit(train_design, train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs'])
'''
new = '''    # Do not let imputed covariates bias the physical parameter estimates.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    train_past, train_hold = _physical(train, calibration)
    train_design = _Design(train_past[complete], train_hold[complete], alignment[:n][complete], *common)
    curves = train['strain_curves_obs'][complete] - train['strain_0'][complete, None]
    taus, weights = _fit(train_design, curves, train['instant_rate_obs'][complete])
'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
cd /holdout_task && export OPENBLAS_NUM_THREADS=1 && python3 -I validate_public_schema.py && python3 -I /tmp/verify_rheology_batch01.py
```
- **Container Exit Code**: `0` (`1269 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_01_solver.py']`
- **Stdout (tail)**:
```text
_valid": true, "errors": []}
estimated_taus rmse 0.03917651199773598 max_abs 0.05540173167869433
  signed errors [-0.0004963  -0.05540173]
hereditary_weights rmse 0.002317160707610781 max_abs 0.003695331930499368
  signed errors [-0.0019996  -0.0018497  -0.0012164  -0.00369533  0.00243911 -0.00247177
  0.00172361]
regime_mode_weights rmse 0.0006167112644354518 max_abs 0.0008514479486774196
  signed errors [0.00018895 0.00085145]
predicted_strain rmse 0.0404962840418289 max_abs 0.4429971109033835
initial_rate_pred rmse 0.030883370391337504 max_abs 0.2408195458507647
calibrated stress max abs 0.0002544414505800674 0.00013367516081164155
public lab/active 0 0 n 7 curve/rate RMSE 0.0037526809329872786 0.0029262923005723258
public lab/active 0 1 n 10 curve/rate RMSE 0.03498508186864972 0.02748532761862675
public lab/active 1 0 n 3 curve/rate RMSE 0.003897734430884325 0.003972791155696242
public lab/active 1 1 n 13 curve/rate RMSE 0.08982518760640273 0.06819907788344524
public lab/active 2 0 n 2 curve/rate RMSE 0.0011607809942482496 0.0017900283003877287
public lab/active 2 1 n 14 curve/rate RMSE 0.0012840276910415604 0.0017514433276345272
public lab/active 3 0 n 12 curve/rate RMSE 0.004483261497827073 0.0036426695332053946
public lab/active 3 1 n 11 curve/rate RMSE 0.006969451608093518 0.005539555131101155
held-out lab 0 curve/rate RMSE 1.0259823576801832 0.055267742923600295 curve/rate median abs 0.013400171726308918 0.012128114698592096 worst horizon RMSE 3.0581866895302596
held-out lab 1 curve/rate RMSE 1.368814369723163 0.047211937619074905 curve/rate median abs 0.012652375403082283 0.016592760414311236 worst horizon RMSE 4.1665932358651085
held-out lab 2 curve/rate RMSE 0.9061776771687121 0.025040267273827195 curve/rate median abs 0.011816014833689048 0.013643062091380997 worst horizon RMSE 2.558329077047375
held-out lab 3 curve/rate RMSE 0.3405120021783379 0.049433820439388786 curve/rate median abs 0.014142300644283921 0.013870448779037203 worst horizon RMSE 0.7848362824149202
single EMI per channel: max coefficient shift 1.1380833699087267e-05
history fallback, lab 0 in-batch rows 0 coefficients [9.99976886e-01 1.27323341e-06]
history fallback, lab 0 in-batch rows 1 coefficients [0.99997689 0.0712195 ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9085e865b5f17a23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_9085e865 (directory).
```

### Turn 11 — `submit_completed_task` (`14619 ms` LLM)
- **Rationale**: The final artifact passed isolated-process schema validation, held-out lab tests, and checks of the memory recurrence, shear boundary, and calibration fallback. Restricting the fit to complete training coupons improved relaxation-time recovery without materially changing overall prediction error.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_01_solver.py': 'FILE:7377'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_01_solver.py`. The public schema passed, and the maximum relative error in the relaxation times was 1.45%. Public RMSE was 0.0405 for strain and 0.0309 for initial rate. Coupons with missing features remained the largest source of prediction error.`

### Turn 12 — `execute_in_remote_docker` (`37349 ms` LLM)
- **Rationale**: An independent expansion can detect a missing or mis-gated physical term. Signed residuals within held-out labs and damage regimes can expose bias hidden by pooled scores. Four clean calibration points plus a near-threshold EMI spike tests whether the documented in-batch rule holds when the sweep is sparse; material coefficient errors would warrant a targeted calibration fix.
- **Alternatives Weighed**: `Add rational, relay, or Fokker–Planck terms and score balanced accuracy` (rejected: The task specifies continuous strain and rate from a two-branch recurrence. There are no relay states, class labels, or diffusion operators to validate.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('candidate','/holdout_task/alpha_m97_c02_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); x=m._align(tr,te)[:n]; dt,t,crit,shear=(d[k] for k in ('past_dt','eval_times','crit_stress','shear_banding_index')); rows=d['calibration_load_cells']; cal=m._calibrate(rows); p,h=m._physical(tr,cal); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1); active=(np.maximum(abs(p)-crit,0)**2@dt)>0
for lab in range(4):
    fit=complete&(tr['lab_id']!=lab); test=complete&(tr['lab_id']==lab); args=(dt,t,crit,shear)
    tau,w=m._fit(m._Design(p[fit],h[fit],x[fit],*args),y[fit],r[fit]); z=np.zeros((n,2))
    for j,delta in enumerate(dt):
        a=np.exp(-delta/tau); z=a*z+(1-a)*p[:,j,None]
    gap=h[:,None]-z; act=np.tanh(np.maximum(abs(p)-crit,0)**2@dt) if shear<=.50 else np.zeros(n); phi=np.log1p(t) if shear<=.50 else t; response=1-np.exp(-t[:,None]/tau); drift=w[2]+x@w[3:7]+w[8]*act*z[:,0]; curve=(gap*w[:2])@response.T+w[7]*np.outer(act*h,response[:,0])+np.outer(drift,phi); rate=(gap*w[:2]/tau).sum(axis=1)+w[7]*act*h/tau[0]+drift
    A,B=m._Design(p,h,x,*args).matrices(tau); assert np.allclose(A@w,curve,atol=1e-12,rtol=0) and np.allclose(B@w,rate,atol=1e-12,rtol=0)
    for mode in (False,True):
        q=test&(active==mode); ec=curve[q]-y[q]; er=rate[q]-r[q]; print('held-out lab/damage',lab,int(mode),q.sum(),'max abs horizon median',round(np.max(abs(np.median(ec,axis=0))),5),'signed rate median',round(np.median(er),5),'rate RMSE',round(np.sqrt(np.mean(er**2)),5))
    print('held-out lab',lab,'max abs rate-residual/feature correlation',round(np.max(abs(np.corrcoef(np.column_stack((rate[test]-r[test],x[test])).T)[0,1:])),4))
# Four clean in-batch points take priority over stale calibration when a fifth has EMI.
from itertools import combinations
worst=(0.,None)
for lab in range(4):
    inds=np.flatnonzero(rows['lab_id']==lab); base=rows['lab_id']!=lab
    for ids in combinations(inds,5):
        sel=base.copy(); sel[list(ids)]=True; c={k:v[sel].copy() for k,v in rows.items()}; local=np.flatnonzero(c['lab_id']==lab)
        for j in local:
            for spike in (-3.1,3.1):
                c['true_stress'][j]+=spike; err=np.max(abs(m._calibrate(c)[lab]-cal[lab])); c['true_stress'][j]-=spike
                if err>worst[0]: worst=(float(err),(lab,tuple(map(int,ids)),int(j-local[0]),spike))
print('worst four-clean plus one EMI coefficient error',worst)
PY
```
- **Container Exit Code**: `0` (`5383 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held-out lab/damage 0 0 9 max abs horizon median 0.01098 signed rate median -0.00468 rate RMSE 0.01291
held-out lab/damage 0 1 27 max abs horizon median 0.00806 signed rate median -0.00566 rate RMSE 0.01711
held-out lab 0 max abs rate-residual/feature correlation 0.4557
held-out lab/damage 1 0 10 max abs horizon median 0.01521 signed rate median 0.00718 rate RMSE 0.01735
held-out lab/damage 1 1 23 max abs horizon median 0.00975 signed rate median -0.00162 rate RMSE 0.01947
held-out lab 1 max abs rate-residual/feature correlation 0.3659
held-out lab/damage 2 0 18 max abs horizon median 0.00475 signed rate median 0.00446 rate RMSE 0.02313
held-out lab/damage 2 1 29 max abs horizon median 0.00452 signed rate median 0.00608 rate RMSE 0.01696
held-out lab 2 max abs rate-residual/feature correlation 0.2572
held-out lab/damage 3 0 12 max abs horizon median 0.0097 signed rate median -0.00361 rate RMSE 0.01797
held-out lab/damage 3 1 33 max abs horizon median 0.01257 signed rate median 0.00737 rate RMSE 0.01632
held-out lab 3 max abs rate-residual/feature correlation 0.2084
worst four-clean plus one EMI coefficient error (0.0010312528377833252, (0, (0, 1, 2, 5, 9), 0, -3.1))

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9085e865b5f17a23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_9085e865 (directory).
```

### Turn 13 — `submit_completed_task` (`19569 ms` LLM)
- **Rationale**: The verified artifact passed isolated-process schema validation and matched the independently reconstructed strain and rate equations. The largest held-out lab-by-damage median horizon bias was 0.0152 strain units, and the largest calibration coefficient shift with four clean points and one near-threshold EMI spike was 0.00103. Neither check identified a material defect warranting another edit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_01_solver.py`. The public schema passed. Public RMSE was 0.0405 for strain and 0.0309 for initial rate, and the maximum relative relaxation-time error was 1.45%. Missing coupon features remained the main source of prediction error.`

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
