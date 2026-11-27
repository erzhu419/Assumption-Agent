# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c02_batch_03__rheology_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c02_step_03` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `328,201` (`316,850` / `107,520` / `11,351`) |
| **Total Episode Wall Time (s)** | `371.97s` |

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

### Turn 1 — `execute_in_remote_docker` (`17014 ms` LLM)
- **Rationale**: Inspect the public schema and archive interfaces to identify the available fit targets, calibration coverage, and whether an existing alignment or relaxation primitive applies before implementing the solver.
- **Alternatives Weighed**: `Reuse a prior solver without inspecting the new batch` (rejected: The prior gate failed; the new batch's schema and calibration coverage determine which components can be reused.)

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; cat /tmp/hegel_continual_archive/archive_manifest.json | head -100; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation viscoelastic relaxation variable projection' 2>&1 | head -30; python3 - <<'PY'
import pickle,numpy as np
with open('/holdout_task/public_alpha_m97_c02_batch_03.pkl','rb') as f: d=pickle.load(f)
def dump(d,level=0):
 for k,v in d.items():
  print(' '*level,k,type(v).__name__,getattr(v,'shape',''),repr(v)[:230] if not isinstance(v,dict) else '')
  if isinstance(v,dict) and level<3: dump(v,level+1)
dump(d)
PY
```
- **Container Exit Code**: `1` (`412 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
enous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
  File "<stdin>", line 4, in dump
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`19460 ms` LLM)
- **Rationale**: Unwrap the public episode as the validator does, then inspect calibration coverage, training targets, and reference-panel layout. These determine when archive fallback is warranted and which archived preprocessing can be reused without changing the physical coordinates.
- **Alternatives Weighed**: `Reuse the archived solver unchanged` (rejected: The prior gate failed, and the current batch explicitly requires a different strain and initial-rate basis.)

```bash
sed -n '1,235p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_02_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel 2>&1 | head -28; python3 - <<'PY'
import pickle, collections, json, numpy as np
p=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_03.pkl','rb'))
print('episode count',len(p),'item keys',list(p[0]))
d=p[0]['dataset']
def show(d, level=0):
 for k,v in d.items():
  a=getattr(v,'shape',None); print(' '*level,k,type(v).__name__,a if a is not None else len(v) if isinstance(v,(list,dict,tuple)) else repr(v), '' if isinstance(v,dict) else repr(v)[:160])
  if isinstance(v,dict): show(v,level+1)
show(d)
c=d['calibration_load_cells']; print('calibration',repr(c)[:3000]); print('archive structure',repr(json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')))[:1100])
PY
```
- **Container Exit Code**: `0` (`430 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
7 ],
       [-0.775507,  0.128107,  0.736435],
       [-0.586829, -0.99547 ,  1.383784],
       [-2.331334, -0.572272,  2.5
calibration {'lab_id': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), 'cmd_stress': array([ 0.946041,  0.026539, -1.868094,  0.186519,  1.478584, -1.876496,
        0.722244, -0.491277,  1.000356, -1.752239, -0.257067, -0.430637,
        0.442373,  1.855268,  0.321991, -1.993656, -2.094652, -1.028239,
       -1.643449,  0.129373,  2.036062,  2.193845, -1.436971, -1.129126,
        1.519563,  0.48526 , -1.347118, -1.835224, -0.864378, -0.591004,
       -1.143398,  0.664015, -0.898756,  1.989517,  1.564437, -1.108175,
       -0.501381,  0.364932, -1.903257,  0.817037]), 'true_stress': array([ 0.60765478, -0.11919811, -1.61561151,  0.00731804,  1.02816572,
       -1.62239491,  0.43045557, -0.52803385,  0.6502332 , -1.52412899,
       -0.17684157, -0.40410696,  0.73944538,  2.59021954,  0.58181761,
       -2.45166172, -2.58400505, -1.18702633, -1.99296862,  0.32958423,
        1.83052771,  1.96485337, -1.12133496, -0.85991067,  1.39159056,
        0.51252736, -1.04501925, -1.45992564, -0.63456421, -0.40229658,
       -1.54776417,  0.69340999, -1.24446418,  2.33684661,  1.80970421,
       -1.5039791 , -0.75179943,  0.3226494 , -2.49024019,  0.88305147])}
archive structure [{'episode_short_name': 'alpha_m97_c02_s00', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [1.619455, -1.307963, -0.646756, 0.95759, 2.133391, -0.419148, -0.546245, -1.403033, 1.416515, -0.474218, 0.56263, -1.431472, -1.102381, 0.100096, 0.332884, 0.978926, 1.410301, 2.014558, 2.037985, -1.520981, -1.132282, -0.400487, 1.428566, 2.184907, 2.114515, 2.024265, 0.373395, 1.839347, -1.383881, 0.252459, -1.650907, 2.152332, 0.673079, 0.633995, 2.133025, 0.157003, -0.816267, -1.031715, -0.029779, 1.574899], 'true_stress': [1.619593, -1.308033, -0.646936, 0.95752, 2.133164, -0.418978, -0.546283, -1.402876, 1.416566, -0.474167, 0.562697, -1.431478, -1.102292, 0.1, 0.333007, 0.979026, 1.410368, 2.014512, 2.037978, -1.520941, -1.132263, -0.400526, 1.428582, 2.184857, 2.114623, 2.024408, 0.373485, 1.83932, -1.383844, 0.25249, -1.650978, 2.152261, 0.672955, 0.634083, 2.133016, 0.157033, -0.816183, -1.031889, -0.02981, 1.574906]}, {'episode_short_name': 'alpha_m97_c02_batch_01', 'lab_id': [0, 0, 0,
```

### Turn 3 — `execute_in_remote_docker` (`18746 ms` LLM)
- **Rationale**: The previous listing was truncated. A compact inspection will recover the exact training targets and reference-panel transformation before implementation. Comparing historical channel fits will also show whether fallback may pool prior sweeps or must account for calibration changes across phases.
- **Alternatives Weighed**: `Copy the archived solver unchanged` (rejected: The archived terminal-flow model lacks the specified saturated flow and jammed-slip terms.)

```bash
sed -n '56,230p' /tmp/hegel_continual_archive/alpha_m97_c02_batch_02_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
p=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_03.pkl','rb'))
print('episode count:',len(p))
d=p[0]['dataset']
for k,v in d.items():
 if isinstance(v,dict):
  print(k,{j:np.shape(a) for j,a in v.items()})
 else:
  a=np.asarray(v); print(k,a.shape,a.ravel()[:22])
print('missing coupons',[(k,np.isnan(d[k]['coupon_features_raw']).sum(axis=0).tolist()) for k in ('train','test')])
for e in json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')):
 c,x,y=map(np.asarray,(e['lab_id'],e['cmd_stress'],e['true_stress'])); print(e['episode_short_name'],[np.polyfit(x[c==l],y[c==l],1).round(5).tolist() if (c==l).sum()>1 else None for l in range(4)])
PY
```
- **Container Exit Code**: `0` (`428 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
))
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
episode count: 1
past_dt (18,) [0.6662 0.6332 0.3214 0.709  0.3215 0.3399 0.3466 0.2323 0.6797 0.7147
 0.5853 0.3089 0.5838 0.6262 0.6828 0.2639 0.6961 0.5385]
eval_times (10,) [0.3    0.8444 1.3889 1.9333 2.4778 3.0222 3.5667 4.1111 4.6556 5.2   ]
crit_stress () [2.65]
yield_stress () [2.65]
shear_banding_index () [0.82]
calibration_load_cells {'lab_id': (40,), 'cmd_stress': (40,), 'true_stress': (40,)}
train {'lab_id': (180,), 'stratum': (180,), 'coupon_features_raw': (180, 4), 'ref_coupons': (180, 32), 'past_stress_cmd': (180, 18), 'stress_hold_cmd': (180,), 'strain_0': (180,), 'aux_chamber_telemetry': (180, 3), 'strain_curves_obs': (180, 10), 'instant_rate_obs': (180,)}
test {'lab_id': (72,), 'stratum': (72,), 'coupon_features_raw': (72, 4), 'ref_coupons': (72, 32), 'past_stress_cmd': (72, 18), 'stress_hold_cmd': (72,), 'strain_0': (72,), 'aux_chamber_telemetry': (72, 3)}
missing coupons [('train', [6, 8, 4, 1]), ('test', [1, 2, 0, 1])]
alpha_m97_c02_s00 [[0.99998, 0.0], [0.99999, 4e-05], [1.00001, 3e-05], [1.00001, -3e-05]]
alpha_m97_c02_batch_01 [[0.79005, -0.13999], [1.30996, 0.16006], [0.84996, 0.10004], [1.24, -0.12996]]
alpha_m97_c02_batch_02 [[0.79001, -0.14], [1.30998, 0.15996], [0.84999, 0.09998], [1.24004, -0.12998]]
```

### Turn 4 — `execute_in_remote_docker` (`69035 ms` LLM)
- **Rationale**: Implement the exact strain and initial-rate equations with shared coefficients. Hold out each lab's responses while retaining the reference panel and calibration inputs allowed by the task. Lab, missing-coupon, high-stress, and time-slice errors will show whether robust fitting or an alignment correction is warranted.
- **Alternatives Weighed**: `Use the archived fit and append a slip correction` (rejected: Saturated flow changes the drift basis itself, so the inherited weights and the slip weights must be fitted together.); `Pool all historical calibrations if fallback is needed` (rejected: The baseline sweep has nearly unit gains while the two later sweeps agree with each other; pooling across that shift would bias fallback.)

```bash
cat > /holdout_task/alpha_m97_c02_batch_03_solver.py <<'PY'
"""Two-branch hereditary creep with saturated jammed slip."""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _points(sweep, lab):
    ids, x, y = (np.asarray(sweep[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    take = (ids == lab) & np.isfinite(x) & np.isfinite(y)
    return x[take], y[take]


def _clean(x, y):
    i, j = np.triu_indices(len(x), 1)
    distinct = x[i] != x[j]
    if len(x) < 3 or not distinct.any():
        return np.ones(len(x), dtype=bool)
    i, j = i[distinct], j[distinct]
    gain = np.median((y[j] - y[i]) / (x[j] - x[i]))
    intercepts = y - gain * x
    return np.abs(intercepts - np.median(intercepts)) <= 3.0


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _calibrate(sweep):
    out = np.empty((4, 2))
    history = None
    for lab in range(4):
        x, y = _points(sweep, lab)
        clean = _clean(x, y)
        if clean.sum() >= 4 or (len(x) >= 2 and clean.all()):
            out[lab] = _affine(x[clean], y[clean])
        else:
            if history is None:
                history = json.loads(_ARCHIVE.read_text())
            for previous in reversed(history):
                hx, hy = _points(previous, lab)
                keep = _clean(hx, hy)
                if keep.sum() >= 2 and np.ptp(hx[keep]) > 0:
                    out[lab] = _affine(hx[keep], hy[keep])
                    break
            else:
                raise ValueError(f'No calibration sweep with stress variation for lab {lab}')
    return out


def _align(train, test):
    labs, strata, raw, refs = (np.concatenate((train[k], test[k])) for k in
                               ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        take = labs == lab
        ref = refs[take]
        aligned[take] = ((raw[take] - np.nanmedian(np.nanmean(ref, axis=0))) /
                         np.nanmedian(np.nanstd(ref, axis=0)))
    _, cells = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = []
    for c in np.unique(cells):
        values = aligned[cells == c]
        counts = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), counts,
                               out=np.full(raw.shape[1], np.nan), where=counts > 0))
    return np.where(np.isfinite(aligned), aligned, np.nanmean(means, axis=0))


def _physical(split, calibration):
    gain, bias = calibration[split['lab_id']].T
    return (gain[:, None] * split['past_stress_cmd'] + bias[:, None],
            gain * split['stress_hold_cmd'] + bias)


def _design(past, hold, x, dt, times, taus):
    n = len(hold)
    later = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = -np.expm1(-dt[:, None] / taus) * np.exp(-later[:, None] / taus)
    gap = hold[:, None] - past @ memory
    response = -np.expm1(-times[:, None] / taus)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    phi = times / (1.0 + 0.85 * times)
    drift = np.column_stack((np.ones(n), x, slip))
    diff = gap[:, 0] - gap[:, 1]
    curves = np.concatenate((gap[:, None, :] * response,
                             drift[:, None, :] * phi[None, :, None],
                             (diff[:, None] * response[:, 0])[..., None]), axis=2)
    rate = np.column_stack((gap / taus, drift, diff / taus[0]))
    return curves, rate


def _fit(past, hold, x, dt, times, curves, rates):
    target = np.r_[curves.ravel(), rates]

    def project(logtaus, parameters=False):
        a, b = _design(past, hold, x, dt, times, np.exp(logtaus))
        design = np.concatenate((a.reshape(-1, 9), b))
        weights = lstsq(design, target, lapack_driver='gelsy')[0]
        if parameters:
            return np.exp(logtaus), weights
        return np.einsum('ij,j->i', design, weights) - target

    bounds = (np.log(dt.min() / 30), np.log(times.max() * 30))
    starts = [(0.16, 1.0), (0.4, 2.5), (0.9, 5.0)]
    starts += [(b, a) for a, b in starts]
    fits = [least_squares(project, np.clip(np.log(start), *bounds), bounds=bounds,
                          max_nfev=90, ftol=1e-10, gtol=1e-10) for start in starts]
    return project(min(fits, key=lambda r: r.cost).x, parameters=True)


def solve_alpha_m97_c02_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    times, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    cal = _calibrate(dataset['calibration_load_cells'])
    aligned = _align(train, test)
    taus, weights = _fit(*_physical(train, cal), aligned[:n], dt, times,
                         train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs'])
    a, b = _design(*_physical(test, cal), aligned[n:], dt, times, taus)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + a @ weights,
            'initial_rate_pred': b @ weights}
PY
cat > /tmp/hegel_batch03_cv.py <<'PY'
import importlib.util, pickle
import numpy as np
base = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', base + 'alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d = pickle.load(open(base+'public_alpha_m97_c02_batch_03.pkl','rb'))[0]['dataset']
tr, te = d['train'], d['test']; dt, times = d['past_dt'], d['eval_times']
lab = tr['lab_id']; cal = m._calibrate(d['calibration_load_cells']); print('calibration',cal.tolist())
x = m._align(tr,te)[:len(lab)]; p,h = m._physical(tr,cal)
y = tr['strain_curves_obs'] - tr['strain_0'][:,None]; r = tr['instant_rate_obs']; er_y = np.empty_like(y); er_r = np.empty_like(r)
rmse = lambda v: float(np.sqrt(np.mean(v*v)))
for l in [-1,0,1,2,3]:
    fit = lab != l; pred = ~fit if l >= 0 else fit
    taus,w = m._fit(p[fit],h[fit],x[fit],dt,times,y[fit],r[fit]); a,b = m._design(p[pred],h[pred],x[pred],dt,times,taus)
    dy,dr = a@w-y[pred], b@w-r[pred]
    if l >= 0: er_y[pred],er_r[pred] = dy,dr
    print(l, 'taus',taus,'weights',w,'rmse',rmse(dy),rmse(dr),'abs residual 50/95/100%',np.quantile(abs(dy),[.5,.95,1]),np.quantile(abs(dr),[.5,.95,1]))
for name,mask in [('missing',np.isnan(tr['coupon_features_raw']).any(1)),('large_stress',abs(h)>=2),('negative',h<0),('positive',h>=0)]:
    print(name,mask.sum(),rmse(er_y[mask]),rmse(er_r[mask]))
print('time-wise OOF strain RMSE',np.sqrt(np.mean(er_y**2,axis=0)))
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/hegel_batch03_cv.py
```
- **Container Exit Code**: `0` (`1080 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
calibration [[0.7899746886839856, -0.13996010961137748], [1.3099732469730958, 0.1599621257048209], [0.8499868792510624, 0.10001623237093304], [1.2399723212528373, -0.13003537314250024]]
-1 taus [0.42603181 1.85906495] weights [ 0.41161289  0.66458906  0.10633312  0.18171319 -0.06801652  0.13673791
 -0.07985628  0.2715066  -0.14533785] rmse 1.5688468892376684 0.1271080494567588 abs residual 50/95/100% [ 0.07092738  0.27456446 59.75133655] [0.09415212 0.24414453 0.30947417]
0 taus [0.40740139 1.84772141] weights [ 0.39167601  0.72988182  0.0833256   0.1806642  -0.05413166  0.1441519
 -0.07476015  0.2208793  -0.12161863] rmse 0.50635180794668 0.18889787527637253 abs residual 50/95/100% [0.10250213 0.31833711 8.68162857] [0.15724755 0.33530062 0.40158938]
1 taus [0.42707662 1.81327776] weights [ 0.41632201  0.68555588  0.07931461  0.18945542 -0.03780831  0.10575728
 -0.06721607  0.21485718 -0.15125407] rmse 1.004219261198345 0.14801228981880665 abs residual 50/95/100% [ 0.0997615   0.33206166 13.89709679] [0.10166862 0.2551249  0.35789509]
2 taus [0.38264279 1.67611153] weights [ 0.34313785  0.72646465  0.10868433  0.1904005  -0.06188429  0.14602508
 -0.08687833  0.2450736  -0.11176448] rmse 0.46982370922563776 0.150174627367714 abs residual 50/95/100% [0.07155072 0.37713403 4.81656022] [0.12026749 0.27653235 0.34911404]
3 taus [3.22606724 0.63402504] weights [ 0.35270276  0.71079143  0.15342639  0.16898213 -0.11926014  0.16059753
 -0.09838881  0.3623825   0.47984348] rmse 2.9695882418149186 0.10160184812105795 abs residual 50/95/100% [4.00201154e-02 2.30717414e-01 5.99805363e+01] [0.07071316 0.18911782 0.21762331]
missing 18 4.473417415789008 0.19677705795489261
large_stress 0 nan nan
negative 87 0.7039070627527462 0.14116763272324012
positive 93 2.084206770982244 0.1563777474437869
time-wise OOF strain RMSE [1.05605453 0.7886619  1.13750178 4.49047833 0.16255352 0.26167191
 0.17026583 0.30484573 1.06601162 0.53821733]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7c7367410b52b30b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7c736741 (directory).
```
- **Stderr (tail)**:
```text
/usr/local/lib/python3.12/dist-packages/numpy/core/fromnumeric.py:3504: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/usr/local/lib/python3.12/dist-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
```

### Turn 5 — `execute_in_remote_docker` (`36016 ms` LLM)
- **Rationale**: Keep the exact shared constitutive basis, but reduce the influence of extreme observations with a soft-L1 initialization and a Cauchy refinement scaled separately for strain and rate. Compare held-out lab and upper-stress-slice errors, along with median residuals, so isolated measurement spikes do not obscure predictive performance.
- **Alternatives Weighed**: `Add a new physical response term` (rejected: A strain residual near 60 alongside a 95th percentile near 0.27 points to sparse contamination; changing the prescribed basis would not be justified by this evidence.); `Discard whole specimens with an extreme strain residual` (rejected: A single bad time point need not invalidate the remaining observations or the measured initial rate.)

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS scale weighted least squares' 2>&1 | head -10; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation 2>&1 | head -15
python3 - <<'PY'
from pathlib import Path
p=Path('/holdout_task/alpha_m97_c02_batch_03_solver.py')
s=p.read_text()
start=s.index('    def project(', s.index('def _fit(')); end=s.index('\n\ndef solve_',start)
s=s[:start]+'''    def design(logtaus):
        a, b = _design(past, hold, x, dt, times, np.exp(logtaus))
        return np.concatenate((a.reshape(-1, 9), b))

    def predict(d, w):
        return np.einsum('ij,j->i', d, w)

    def residual(params):
        return predict(design(params[:2]), params[2:]) - target

    def jacobian(params):
        d = design(params[:2])
        h = 1.e-5
        # Only the two memory coordinates require numerical derivatives.
        columns = [predict((design(params[:2] + step) - design(params[:2] - step)) / (2*h),
                           params[2:]) for step in ([h, 0.], [0., h])]
        return np.column_stack((*columns, d))

    lo, hi = np.log(dt.min() / 30), np.log(times.max() * 30)
    bounds = (np.r_[lo, lo, np.full(9, -np.inf)], np.r_[hi, hi, np.full(9, np.inf)])
    starts = [(0.16, 1.0), (0.4, 2.5), (0.9, 5.0)]
    starts += [(b, a) for a, b in starts]

    def optimize(params, scale, loss):
        return least_squares(lambda q: residual(q) / scale, params,
                             jac=lambda q: jacobian(q) / np.broadcast_to(scale, target.shape)[:, None],
                             bounds=bounds, x_scale='jac', loss=loss, max_nfev=120,
                             ftol=1.e-9)

    fits = []
    for start in starts:
        logtaus = np.clip(np.log(start), lo + 1.e-8, hi - 1.e-8)
        weights = lstsq(design(logtaus), target, lapack_driver='gelsy')[0]
        fits.append(optimize(np.r_[logtaus, weights], .05, 'soft_l1'))
    seed = min(fits, key=lambda r: r.cost).x
    err = residual(seed)
    def mad(values):
        return max(1.e-6, 1.4826 * np.median(np.abs(values - np.median(values))))
    scale = 2.5 * np.r_[np.full(curves.size, mad(err[:curves.size])),
                       np.full(rates.size, mad(err[curves.size:]))]
    best = optimize(seed, scale, 'cauchy').x
    return np.exp(best[:2]), best[2:]
'''+s[end:]
p.write_text(s)
p=Path('/tmp/hegel_batch03_cv.py');s=p.read_text().replace('abs(h)>=2','abs(h)>=np.quantile(abs(h),.9)').replace("print('time-wise", "print('OOF curve and rate median absolute errors',np.median(abs(er_y)),np.median(abs(er_r)))\nprint('time-wise");p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/hegel_batch03_cv.py
```
- **Container Exit Code**: `0` (`1547 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_batch_03_solver.py']`
- **Stdout (tail)**:
```text
positive onset rate, a numerically full-rank fitted Jacobian, and a positive slope not active at its lower bound.",
{"schema_valid": true, "errors": []}
calibration [[0.7899746886839856, -0.13996010961137748], [1.3099732469730958, 0.1599621257048209], [0.8499868792510624, 0.10001623237093304], [1.2399723212528373, -0.13003537314250024]]
-1 taus [0.57678499 4.27307036] weights [ 0.68023674  0.5519697   0.1386321   0.15613458 -0.12763699  0.11890251
 -0.08794411  0.30999915 -0.21916516] rmse 1.5731440699025978 0.043748877390359826 abs residual 50/95/100% [1.34958101e-02 1.93175157e-01 6.00016768e+01] [0.01589896 0.0997413  0.21880427]
0 taus [0.57578727 4.24411256] weights [ 0.67761032  0.55182701  0.13934791  0.1574239  -0.12959013  0.11998516
 -0.08767789  0.31221864 -0.21798238] rmse 0.49810345809939494 0.04871669736350539 abs residual 50/95/100% [0.01293251 0.13921007 8.83204644] [0.01657656 0.10767417 0.17207473]
1 taus [0.57578796 4.29954757] weights [ 0.67930605  0.55599014  0.13634817  0.15797393 -0.12860177  0.12020621
 -0.0884189   0.30996271 -0.21758504] rmse 0.9972855645900599 0.03692093532207583 abs residual 50/95/100% [1.34239409e-02 1.94936871e-01 1.39473223e+01] [0.01094563 0.05315236 0.21916988]
2 taus [0.57535106 4.2680547 ] weights [ 0.67879577  0.55686646  0.13997724  0.15576586 -0.12619699  0.11832941
 -0.08889715  0.30742092 -0.21964218] rmse 0.453514750583214 0.04027501627670124 abs residual 50/95/100% [0.015481   0.32960297 4.88008004] [0.01805857 0.10567599 0.12883463]
3 taus [0.5799883  4.28204533] weights [ 0.68455894  0.54368042  0.13849615  0.15404556 -0.12665182  0.11729157
 -0.08703162  0.31104908 -0.22073768] rmse 2.9700826671945117 0.05337869711313584 abs residual 50/95/100% [1.60818810e-02 1.89513793e-01 6.00021955e+01] [0.02301695 0.13049449 0.19697023]
missing 18 4.47369821583856 0.12453628881580192
large_stress 18 0.4424749767620498 0.02131948274322011
negative 87 0.7001116152893542 0.03424055203251987
positive 93 2.081195693228562 0.05239480432291314
OOF curve and rate median absolute errors 0.014643169115761467 0.017165883094720114
time-wise OOF strain RMSE [1.05116404 0.78414725 1.13794265 4.49079681 0.12627415 0.24863557
 0.1092352  0.27351268 1.06083732 0.52533585]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bebd97489bcba8fe (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_bebd9748 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`26275 ms` LLM)
- **Rationale**: Check whether imputed training features are pulling the shared coefficients away from the measured complete-coupon response. Compare both estimators on the same held-out labs and retain all rows unless complete-only fitting meaningfully improves the worst slices without harming missing-coupon predictions.
- **Alternatives Weighed**: `Change the constitutive basis to reduce raw strain RMSE` (rejected: The largest residual is near 60 while the OOF median is 0.0146; these measurements do not justify departing from the specified equations.)

```bash
cat > /tmp/hegel_batch03_compare.py <<'PY'
exec(open('/tmp/hegel_batch03_cv.py').read().split('for l in [-1,0,1,2,3]:')[0])
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
for name, usable in [('robust_all', np.ones(len(lab), bool)), ('robust_complete', ~missing)]:
    ey, er = np.empty_like(y), np.empty_like(r)
    coeff = []
    for l in range(4):
        f, q = (lab != l) & usable, lab == l
        taus, w = m._fit(p[f], h[f], x[f], dt, times, y[f], r[f])
        coeff.append(np.r_[taus,w]); a,b = m._design(p[q],h[q],x[q],dt,times,taus)
        ey[q], er[q] = a@w-y[q], b@w-r[q]
    print(name, 'fold parameter range', np.ptp(coeff,axis=0).round(5))
    for label,q in [('complete',~missing),('missing',missing),('high_stress',abs(h) >= np.quantile(abs(h),.9)), *[(f'lab{i}',lab==i) for i in range(4)]]:
        yy,rr=abs(ey[q]),abs(er[q]); print(label, 'strain median/p90, rate median/p90/RMSE', np.r_[np.quantile(yy,[.5,.9]),np.quantile(rr,[.5,.9]),rmse(rr)].round(5))
    print('curve residuals > 0.5, complete/missing:', np.count_nonzero(abs(ey[~missing])>.5), np.count_nonzero(abs(ey[missing])>.5), 'out of', ey.size)
    print('high residual counts by time',np.sum(abs(ey)>.5,axis=0))
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/hegel_batch03_compare.py
```
- **Container Exit Code**: `0` (`1708 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calibration [[0.7899746886839856, -0.13996010961137748], [1.3099732469730958, 0.1599621257048209], [0.8499868792510624, 0.10001623237093304], [1.2399723212528373, -0.13003537314250024]]
robust_all fold parameter range [0.00464 0.05544 0.00695 0.01319 0.00363 0.00393 0.00339 0.00291 0.00187
 0.0048  0.00315]
complete strain median/p90, rate median/p90/RMSE [0.01277 0.037   0.01526 0.03495 0.02196]
missing strain median/p90, rate median/p90/RMSE [0.07719 0.17328 0.10276 0.18739 0.12454]
high_stress strain median/p90, rate median/p90/RMSE [0.01146 0.028   0.01702 0.03291 0.02132]
lab0 strain median/p90, rate median/p90/RMSE [0.01293 0.08264 0.01658 0.07297 0.04872]
lab1 strain median/p90, rate median/p90/RMSE [0.01342 0.04744 0.01095 0.04112 0.03692]
lab2 strain median/p90, rate median/p90/RMSE [0.01548 0.09847 0.01806 0.04133 0.04028]
lab3 strain median/p90, rate median/p90/RMSE [0.01608 0.11954 0.02302 0.05474 0.05338]
curve residuals > 0.5, complete/missing: 52 1 out of 1800
high residual counts by time [5 8 8 5 2 2 4 6 5 8]
robust_complete fold parameter range [0.00178 0.08422 0.00233 0.01365 0.00349 0.00435 0.0032  0.00296 0.002
 0.00644 0.00316]
complete strain median/p90, rate median/p90/RMSE [0.01272 0.03677 0.01451 0.03596 0.02202]
missing strain median/p90, rate median/p90/RMSE [0.07862 0.17588 0.10222 0.19322 0.12674]
high_stress strain median/p90, rate median/p90/RMSE [0.01134 0.02734 0.01036 0.03196 0.02033]
lab0 strain median/p90, rate median/p90/RMSE [0.01296 0.08176 0.019   0.0751  0.04929]
lab1 strain median/p90, rate median/p90/RMSE [0.01357 0.04762 0.01262 0.03962 0.03789]
lab2 strain median/p90, rate median/p90/RMSE [0.01464 0.10219 0.01712 0.04386 0.04088]
lab3 strain median/p90, rate median/p90/RMSE [0.01575 0.12023 0.01999 0.05275 0.05383]
curve residuals > 0.5, complete/missing: 52 1 out of 1800
high residual counts by time [5 8 8 5 2 2 4 6 5 8]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bebd97489bcba8fe (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_bebd9748 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`40781 ms` LLM)
- **Rationale**: Keep the measured-data winner. Check the untested sparse and EMI calibration paths, compare both prediction outputs against a sequential recurrence, and test coefficient recovery in both branch orders. A failure would identify a concrete calibration, equation, or optimization defect to fix before submission.
- **Alternatives Weighed**: `Fit only complete-coupon specimens` (rejected: It worsens held-out missing-coupon rate RMSE and provides no material gain on complete coupons.)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

base = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', base + 'alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(base + 'public_alpha_m97_c02_batch_03.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
cal = m._calibrate(d['calibration_load_cells'])
x = m._align(tr, te)
n = len(tr['lab_id'])

def forward(split, features, taus, hw, rw):
    g, b = cal[split['lab_id']].T
    hold = g * split['stress_hold_cmd'] + b
    z = np.zeros((len(hold), 2))
    for j, step in enumerate(dt):
        q = np.exp(-step / taus)
        z = q * z + (1 - q) * (g * split['past_stress_cmd'][:, j] + b)[:, None]
    gap = hold[:, None] - z
    slip = np.sign(hold) * np.sqrt(abs(hold))
    drift = hw[2] + features @ hw[3:] + rw[0] * slip
    a = hw[0] * gap[:, 0] + rw[1] * (gap[:, 0] - gap[:, 1])
    b = hw[1] * gap[:, 1]
    curves = (split['strain_0'][:, None] + a[:, None] * (1 - np.exp(-t / taus[0]))
              + b[:, None] * (1 - np.exp(-t / taus[1])) + drift[:, None] * t / (1 + .85 * t))
    return curves, a / taus[0] + b / taus[1] + drift

out = m.solve_alpha_m97_c02_batch_03(d)
curves, rates = forward(te, x[n:], out['estimated_taus'], out['hereditary_weights'], out['regime_mode_weights'])
for key, expected, shape in [('predicted_strain', curves, (len(te['lab_id']), len(t))), ('initial_rate_pred', rates, (len(te['lab_id']),))]:
    assert out[key].shape == shape and np.isfinite(out[key]).all(), key
    np.testing.assert_allclose(out[key], expected, atol=1e-12, rtol=0)
print('Public outputs match independent sequential recurrence and analytic initial rate')

s = {k: np.array(v, copy=True) for k, v in d['calibration_load_cells'].items()}
ids = s['lab_id']; s['true_stress'][np.flatnonzero(ids == 1)[0]] += 7.
keep = (ids != 0); keep[np.flatnonzero(ids == 0)[0]] = True
stressed = m._calibrate({k:v[keep] for k,v in s.items()})
np.testing.assert_allclose(stressed, cal, atol=2e-4, rtol=0)
print('Sparse-channel archive fallback and in-batch EMI rejection pass')

hw = np.array([.67, .55, .14, .16, -.13, .12, -.088]); rw = np.array([.31, -.22])
for taus in (np.array([.58, 4.2]), np.array([4.2, .58])):
    y, r = forward(tr, x[:n], taus, hw, rw)
    result = m.solve_alpha_m97_c02_batch_03(dict(d, train=dict(tr, strain_curves_obs=y, instant_rate_obs=r)))
    for key, truth in zip(('estimated_taus', 'hereditary_weights', 'regime_mode_weights'), (taus, hw, rw)):
        np.testing.assert_allclose(result[key], truth, atol=1e-5, rtol=0)
print('Noise-free coefficient recovery passes in both branch orders')
PY
```
- **Container Exit Code**: `0` (`912 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public outputs match independent sequential recurrence and analytic initial rate
Sparse-channel archive fallback and in-batch EMI rejection pass
Noise-free coefficient recovery passes in both branch orders

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bebd97489bcba8fe (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_bebd9748 (directory).
```

### Turn 8 — `submit_completed_task` (`18673 ms` LLM)
- **Rationale**: The final artifact passed isolated-process verification, including agreement with the sequential memory recurrence and noise-free coefficient recovery in both branch orders. Held-out lab results favored retaining all training rows with robust weighting.
- **Alternatives Weighed**: `Fit only specimens with complete coupon features` (rejected: It worsened missing-coupon rate predictions without a material gain on complete coupons.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_03_solver.py': 'FILE:6390'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_03_solver.py`. The public schema, independent strain and initial-rate calculations, sparse calibration fallback, and EMI rejection all passed. Leave-one-lab-out median absolute strain error was 0.0146, and worst-lab initial-rate RMSE was 0.0534. Isolated strain spikes still dominate raw strain RMSE.`

### Turn 9 — `execute_in_remote_docker` (`28539 ms` LLM)
- **Rationale**: Check whether one calibration point materially changes a channel fit and whether pooled errors hide a systematic lab or stress-sign offset. A large influence would justify investigating calibration; a sustained signed offset would justify investigating alignment before changing the constitutive model.
- **Alternatives Weighed**: `Add quadratic denominators, stochastic drift operators, or relay states` (rejected: The stated equations require the exact linear-denominator saturation and gap difference already verified against an independent recurrence; the dataset specifies neither stochastic drift dynamics nor relay states.); `Compute within-batch balanced accuracy` (rejected: The task predicts continuous strain and rate, not classes. Signed residuals within held-out labs and stress directions test for cancellation hidden by pooled regression metrics.)

```bash
cat > /tmp/hegel_batch03_margin.py <<'PY'
exec(open('/tmp/hegel_batch03_cv.py').read().split('for l in [-1,0,1,2,3]:')[0])
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
ey, er = np.empty_like(y), np.empty_like(r)
for l in np.unique(lab):
    f = lab != l
    taus, w = m._fit(p[f], h[f], x[f], dt, times, y[f], r[f])
    a, b = m._design(p[~f], h[~f], x[~f], dt, times, taus)
    ey[~f], er[~f] = a @ w - y[~f], b @ w - r[~f]
print('regime index:', d['shear_banding_index'])
print('complete coupons: signed held-out median residual, by lab and stress sign')
for l in np.unique(lab):
    for sign in (-1, 1):
        q = (~missing) & (lab == l) & (np.sign(h) == sign)
        if q.any():
            print(l, sign, int(q.sum()), 'strain', round(float(np.median(ey[q])), 5), 'rate', round(float(np.median(er[q])), 5))
print('complete-coupon held-out strain median absolute error by horizon:', np.round(np.median(abs(ey[~missing]), axis=0), 5))
sweep = d['calibration_load_cells']
for l in range(4):
    cx, cy = m._points(sweep, l)
    g, b = cal[l]
    deviation = [m._affine(np.delete(cx,i), np.delete(cy,i)) - cal[l] for i in range(len(cx))]
    print('calibration lab', l, 'max stress residual', float(np.max(abs(cy-g*cx-b))), 'max single-point gain/bias influence', np.max(abs(np.asarray(deviation)), axis=0))
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/hegel_batch03_margin.py && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1407 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calibration [[0.7899746886839856, -0.13996010961137748], [1.3099732469730958, 0.1599621257048209], [0.8499868792510624, 0.10001623237093304], [1.2399723212528373, -0.13003537314250024]]
regime index: 0.82
complete coupons: signed held-out median residual, by lab and stress sign
0 -1 16 strain 0.00232 rate -0.00086
0 1 17 strain 0.00036 rate 0.00559
1 -1 27 strain -0.00689 rate -0.00696
1 1 27 strain -0.00616 rate -0.00334
2 -1 18 strain 0.00564 rate 0.01734
2 1 21 strain -0.00096 rate 0.00886
3 -1 20 strain -0.00133 rate -0.00319
3 1 16 strain -0.0011 rate -0.01187
complete-coupon held-out strain median absolute error by horizon: [0.0112  0.01285 0.01166 0.01365 0.0139  0.01157 0.01217 0.01187 0.01631
 0.01215]
calibration lab 0 max stress residual 0.00026644408372430983 max single-point gain/bias influence [2.46248628e-05 3.66472368e-05]
calibration lab 1 max stress residual 0.00014693766250700735 max single-point gain/bias influence [3.06921153e-05 3.27989289e-05]
calibration lab 2 max stress residual 0.0001846201486080945 max single-point gain/bias influence [1.69891374e-05 2.07030834e-05]
calibration lab 3 max stress residual 0.0002188130456680648 max single-point gain/bias influence [4.20978614e-05 3.24899673e-05]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bebd97489bcba8fe (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_bebd9748 (directory).
```

### Turn 10 — `submit_completed_task` (`17304 ms` LLM)
- **Rationale**: The same verified artifact passed isolated-process checks, matched an independent sequential recurrence, and recovered coefficients in both branch orders. The real-measurement audit found no calibration sensitivity or systematic residual pattern that warranted changing the specified model.
- **Alternatives Weighed**: `Revise load-cell calibration or add lab-specific drift` (rejected: Deleting any single calibration point changed gains and biases by at most 4.3e-5. Held-out complete-coupon errors did not show a consistent offset across labs and stress directions.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_batch_03_solver.py': 'FILE:6390'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_03_solver.py`. The public schema, independent strain and initial-rate calculations, sparse calibration fallback, and EMI-spike rejection passed. Leave-one-lab-out median absolute strain error was 0.0146, and worst-lab initial-rate RMSE was 0.0534. Missing coupons remained the hardest slice for initial-rate prediction (RMSE 0.1245), while isolated strain spikes dominated raw strain RMSE.`

### Turn 11 — `execute_in_remote_docker` (`31034 ms` LLM)
- **Rationale**: Use actual missing coupons in held-out labs to isolate the effect of donor weighting without changing the fit or using held-out responses in preprocessing. A large lab-level shift would warrant investigating panel support or reporting a limitation; a small shift would leave the specified imputation and verified artifact unchanged.
- **Alternatives Weighed**: `Repeat the calibration and constitutive-equation checks` (rejected: Independent recurrence checks passed and deleting a calibration point changed coefficients by at most 4.3e-5. The unresolved measured bottleneck is missing-coupon prediction.)

```bash
cat > /tmp/hegel_batch03_imputation_audit.py <<'PY'
exec(open('/tmp/hegel_batch03_cv.py').read().split('for l in [-1,0,1,2,3]:')[0])
raw = np.concatenate([tr['coupon_features_raw'], te['coupon_features_raw']])
labs = np.concatenate([lab, te['lab_id']])
strata = np.concatenate([tr['stratum'], te['stratum']])
missing = np.isnan(raw[:len(lab)])
scores = {k: [] for k in ('cell_balanced', 'row_weighted')}
for l in np.unique(lab):
    f = lab != l
    q = (lab == l) & missing.any(axis=1)
    if not q.any():
        continue
    taus, w = m._fit(p[f], h[f], x[f], dt, times, y[f], r[f])
    observed = np.where(np.isfinite(raw), m._align(tr, te), np.nan)
    row_average = np.nanmean(observed, axis=0)
    for name, xx in [('cell_balanced', x[q]), ('row_weighted', np.where(missing[q], row_average, x[q]))]:
        a, b = m._design(p[q], h[q], xx, dt, times, taus)
        dy, dr = a @ w-y[q], b @ w-r[q]
        scores[name].append((dy, dr))
        print('lab', l, name, 'n', q.sum(), 'strain median AE', round(float(np.median(abs(dy))), 5), 'rate RMSE', round(rmse(dr), 5), 'signed rate median', round(float(np.median(dr)), 5))
print('Missing-feature donors per column by lab and stratum:')
for l, s in np.unique(np.column_stack((labs, strata)), axis=0):
    donor = np.isfinite(raw[(labs == l) & (strata == s)])
    print(l, s, len(donor), donor.sum(axis=0).tolist())
for name, pairs in scores.items():
    dy, dr = [np.concatenate(v) for v in zip(*pairs)]
    print(name, 'OOF missing-coupon strain median AE', round(float(np.median(abs(dy))), 5), 'rate RMSE', round(rmse(dr), 5))
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/hegel_batch03_imputation_audit.py
```
- **Container Exit Code**: `0` (`1236 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calibration [[0.7899746886839856, -0.13996010961137748], [1.3099732469730958, 0.1599621257048209], [0.8499868792510624, 0.10001623237093304], [1.2399723212528373, -0.13003537314250024]]
lab 0 cell_balanced n 5 strain median AE 0.07892 rate RMSE 0.12392 signed rate median 0.06653
lab 0 row_weighted n 5 strain median AE 0.08398 rate RMSE 0.12193 signed rate median 0.06101
lab 1 cell_balanced n 3 strain median AE 0.07141 rate RMSE 0.13651 signed rate median 0.07189
lab 1 row_weighted n 3 strain median AE 0.06645 rate RMSE 0.13221 signed rate median 0.06672
lab 2 cell_balanced n 4 strain median AE 0.0834 rate RMSE 0.11456 signed rate median 0.10959
lab 2 row_weighted n 4 strain median AE 0.08131 rate RMSE 0.11169 signed rate median 0.11019
lab 3 cell_balanced n 6 strain median AE 0.04064 rate RMSE 0.12516 signed rate median 0.08579
lab 3 row_weighted n 6 strain median AE 0.04179 rate RMSE 0.12685 signed rate median 0.09216
Missing-feature donors per column by lab and stratum:
0 0 18 [18, 18, 17, 18]
0 1 20 [20, 20, 19, 20]
0 2 16 [15, 14, 15, 14]
1 0 16 [16, 15, 16, 16]
1 1 36 [36, 34, 36, 36]
1 2 22 [22, 22, 22, 22]
2 0 17 [17, 17, 17, 17]
2 1 10 [10, 9, 10, 10]
2 2 33 [31, 31, 33, 33]
3 0 15 [14, 15, 14, 15]
3 1 21 [21, 20, 21, 21]
3 2 28 [25, 27, 28, 28]
cell_balanced OOF missing-coupon strain median AE 0.07719 rate RMSE 0.12454
row_weighted OOF missing-coupon strain median AE 0.07867 rate RMSE 0.1232

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bebd97489bcba8fe (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_bebd9748 (directory).
```

### Turn 12 — `submit_completed_task` (`16468 ms` LLM)
- **Rationale**: Every observed lab-stratum cell had donors for all four coupon features. Changing donor weights did not resolve the positive median rate residuals on incomplete coupons. The unchanged artifact passed isolated-process verification, including an independent sequential recurrence and coefficient recovery in both branch orders.
- **Alternatives Weighed**: `Replace cell-balanced imputation with specimen-weighted donor means` (rejected: It departs from the specified weighting, worsens median strain error from 0.0772 to 0.0787, and only lowers rate RMSE from 0.1245 to 0.1232.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c02_batch_03_solver.py`. The public schema, independent strain and initial-rate calculations, sparse calibration fallback, and EMI rejection passed. Leave-one-lab-out median absolute strain error was 0.0146, and worst-lab initial-rate RMSE was 0.0534. Missing coupons remained the hardest slice for rate prediction (RMSE 0.1245).`

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
