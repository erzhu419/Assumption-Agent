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
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `354,499` (`341,346` / `82,176` / `13,153`) |
| **Total Episode Wall Time (s)** | `417.91s` |

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

### Turn 1 — `execute_in_remote_docker` (`17695 ms` LLM)
- **Rationale**: Inspect the public schema and prior solver to identify the usable strain labels and exact panel conventions before fitting; reusing the prior regime equation without checking it could bias the inferred relaxation times.

```bash
cd /holdout_task && cat validate_public_schema.py public_asset_index.json && ls -l /tmp/hegel_continual_archive && sed -n '1,240p' /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation variable projection reference-panel alignment imputation'; python3 - <<'PY'
import pickle, numpy as np
p='/holdout_task/public_alpha_m97_c01_batch_01.pkl'
d=pickle.load(open(p,'rb'))
def inspect(o, level=0):
 if isinstance(o,dict) and level<3:
  for k,v in o.items():
   print(' '*level,k,end=': '); inspect(v,level+1)
 elif isinstance(o,np.ndarray): print(o.shape,o.dtype,o.ravel()[:8])
 elif isinstance(o,(list,tuple)): print(type(o).__name__,len(o),repr(o[:2])[:350])
 else: print(repr(o)[:250])
inspect(d)
PY
```
- **Container Exit Code**: `0` (`699 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.align_control_panel
autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral
list 1 [{'dataset': {'past_dt': array([0.6483, 0.2653, 0.7395, 0.6054, 0.3605, 0.7336, 0.6854, 0.2348,
       0.3829, 0.2905, 0.5424, 0.2461, 0.5108, 0.4695, 0.3015, 0.3905,
       0.2275, 0.2772]), 'eval_times': array([0.3   , 0.8444, 1.3889, 1.9333, 2.4778, 3.0222, 3.5667, 4.1111,
       4.6556, 5.2   ]), 'crit_stress': 2.65, 'yield_stress': 2.65, 'shea
```

### Turn 2 — `execute_in_remote_docker` (`20986 ms` LLM)
- **Rationale**: The first inspection truncated the label and panel details. This targeted check identifies which measurements can constrain the fit, whether the panel primitive matches the actual layout, and whether any calibration channel needs the archival fallback.
- **Alternatives Weighed**: `Reuse the previous solver unchanged` (rejected: Its zero regime weights cannot represent the specified crystallization terms.)

```bash
cat /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py; ls -lh /tmp/hegel_continual_archive; head -c 2000 /tmp/hegel_continual_archive/archive_calibration_history.json; printf '\n'; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel; python3 - <<'PY'
import pickle, numpy as np, scipy
it=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0]; print('item keys:', list(it)); d=it['dataset']; print('scipy:',scipy.__version__)
for k,v in d.items():
 if isinstance(v,dict):
  print(k)
  for p,a in v.items():
   a=np.asarray(a); print(' ',p,a.shape,a.dtype,'missing',np.isnan(a).sum() if a.dtype.kind=='f' else '-', 'example',a.ravel()[:5]);
   if p in ('lab_id','stratum'): print(' ',np.unique(a,return_counts=True))
 else: print(k,np.asarray(v).shape,repr(v)[:260])
c=d['calibration_load_cells'];
for i in range(4):
 m=c['lab_id']==i; x=c['cmd_stress'][m]; y=c['true_stress'][m]; A=np.column_stack((x,np.ones(len(x)))); b=np.linalg.lstsq(A,y,rcond=None)[0]; print('calibration',i,len(x),b,np.max(np.abs(y-A@b)))
PY
```
- **Container Exit Code**: `0` (`713 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
- example [0 0 0 0 0]
  (array([0, 1, 2, 3]), array([10, 10, 10, 10]))
  cmd_stress (40,) float64 missing 0 example [-1.457421 -1.418035  1.887501 -2.004396  1.699178]
  true_stress (40,) float64 missing 0 example [-1.80293462 -1.75023439  2.67922472 -2.53590638  2.42703202]
train
  lab_id (180,) int64 missing - example [0 1 2 3 0]
  (array([0, 1, 2, 3]), array([36, 50, 44, 50]))
  stratum (180,) int64 missing - example [2 1 2 2 0]
  (array([0, 1, 2]), array([57, 53, 70]))
  coupon_features_raw (180, 4) float64 missing 19 example [-1.41345577 -3.01022004  0.23115262 -1.27150237  0.34352064]
  ref_coupons (180, 32) float64 missing 238 example [-0.70667737 -0.36521757 -1.47123501 -2.24526949  0.03498016]
  past_stress_cmd (180, 18) float64 missing 0 example [-0.19080749 -1.34885709 -0.84168836  2.76498738  2.55992532]
  stress_hold_cmd (180,) float64 missing 0 example [ 0.07958731 -0.11365195 -0.21990909 -0.68871687 -0.76236343]
  strain_0 (180,) float64 missing 0 example [-0.552188  0.406017 -0.145016 -0.423774  0.182002]
  aux_chamber_telemetry (180, 3) float64 missing 0 example [-0.94468  -0.248683 -0.856825  0.67546  -1.895343]
  strain_curves_obs (180, 10) float64 missing 0 example [-0.30682153 -0.00513339  0.11082238  0.22859336  0.27228429]
  instant_rate_obs (180,) float64 missing 0 example [ 1.11099833 -0.66347914 -0.3031002  -0.69647305 -1.55177841]
test
  lab_id (72,) int64 missing - example [0 1 2 3 0]
  (array([0, 1, 2, 3]), array([20, 18, 21, 13]))
  stratum (72,) int64 missing - example [0 0 1 0 0]
  (array([0, 1, 2]), array([24, 31, 17]))
  coupon_features_raw (72, 4) float64 missing 4 example [-1.34597117 -1.27212031  1.0461071  -1.57610825  0.79412606]
  ref_coupons (72, 32) float64 missing 96 example [ 0.07822747 -0.22232973 -1.10442901 -0.49520557 -1.61718567]
  past_stress_cmd (72, 18) float64 missing 0 example [-1.0782928  -0.36112687 -1.45929886 -3.06396519 -3.06912584]
  stress_hold_cmd (72,) float64 missing 0 example [-0.29222313  2.18481558 -1.35448264  2.28210482 -1.12242761]
  strain_0 (72,) float64 missing 0 example [ 0.513179  0.68354  -0.21838   0.415848 -0.32671 ]
  aux_chamber_telemetry (72, 3) float64 missing 0 example [ 1.223187 -0.042873 -0.05421  -1.122217 -0.209209]
calibration 0 10 [1.33999935 0.14999378] 0.00015647060919121714
calibration 1 10 [ 0.77000406 -0.16999413] 0.00018822176288391113
calibration 2 10 [ 1.21002713 -0.12000366] 0.00010372126254387659
calibration 3 10 [0.83000992 0.10997266] 8.725097943007576e-05
```

### Turn 3 — `execute_in_remote_docker` (`21756 ms` LLM)
- **Rationale**: Recover the exact panel-imputation convention and archive layout before coding. Count activated specimens by lab to see whether laboratory holdouts can separately evaluate the active and inactive regimes, and identify which public reference fields can verify each output independently.
- **Alternatives Weighed**: `Fit the nonlinear timescales and all weights jointly from a single start` (rejected: The nine weights are linear at fixed timescales, so profiling them leaves only a two-dimensional nonlinear search.)

```bash
sed -n '57,155p' /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_calibration_history.json'
h=json.load(open(p)); print('archive structure:',type(h).__name__,str(h)[:850]); it=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0]; print('gt field names and shapes:',{k:np.shape(v) for k,v in it['gt'].items()}); d=it['dataset']; c=d['calibration_load_cells']; print('calibration command ranges:',[(i,np.min(c['cmd_stress'][c['lab_id']==i]),np.max(c['cmd_stress'][c['lab_id']==i])) for i in range(4)])
for part in ('train','test'):
 s=d[part]; a=np.array([np.polyfit(c['cmd_stress'][c['lab_id']==i],c['true_stress'][c['lab_id']==i],1) for i in range(4)])[s['lab_id']]; stress=s['past_stress_cmd']*a[:,0,None]+a[:,1,None]; act=np.tanh(np.maximum(np.abs(stress)-d['crit_stress'],0)**2@d['past_dt']); print(part,'activation by lab [zero, weak, strong]',[(i,*np.bincount(np.digitize(act[s['lab_id']==i],[1e-12,.3]),minlength=3)) for i in range(4)])
PY
```
- **Container Exit Code**: `0` (`603 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
resid/(scale_y*factor))
                b = np.linalg.lstsq(a*w[:, None], y*w, rcond=None)[0]
            score = np.median(np.abs(y-a@b))
            seeds.append((score, np.r_[logs, b]))
    seeds.sort(key=lambda item: item[0])
    best, runner = seeds[0], next((s for s in seeds[1:] if np.linalg.norm(s[1][:2]-seeds[0][1][:2]) > 0.9), seeds[1])
    def residual(p):
        return _design(p[:2], data, dt, time)[0][good]@p[2:] - y
    def jacobian(p):
        a, jac, _ = _design(p[:2], data, dt, time)
        return np.column_stack((jac[good]*p[2:4], a[good]))
    bounds = (np.r_[lo, lo, np.full(7, -np.inf)], np.r_[hi, hi, np.full(7, np.inf)])
    scale = max(1.2*best[0], 1e-7*scale_y)
    solutions = []
    for _, initial in (best, runner):
        p = initial
        for factor in (4., 1.):
            opt = least_squares(residual, p, jac=jacobian, bounds=bounds, x_scale='jac', loss='cauchy', f_scale=scale*factor, max_nfev=120)
            p = opt.x
        solutions.append((opt.cost, p))
    params = min(solutions, key=lambda item: item[0])[1]
archive structure: list [{'episode_short_name': 'alpha_m97_c01_s00', 'lab_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'cmd_stress': [1.660283, -0.418768, 1.112723, -1.62604, -0.501169, -1.672067, 0.863389, 1.008755, 1.968856, 1.530083, 0.979468, 0.449112, 1.469886, -0.332491, -1.532413, 0.660575, -1.672999, -1.196407, 0.934354, 0.731875, 1.670914, -2.075727, 1.021513, -1.730233, -1.965428, 0.955855, 0.129182, -0.328527, -1.550117, -1.993809, 2.131528, -2.023748, 0.667305, -0.968757, -0.649251, -0.022422, -1.62562, -0.902597, 1.990079, -0.41266], 'true_stress': [1.660305, -0.418669, 1.112724, -1.626001, -0.501125, -1.671976, 0.863198, 1.008634, 1.968816, 1.530248, 0.979291, 0.449215, 1.470001, -0.332446, -1.532394, 0.660518, -1.672977, -1.196416, 0.934345, 0.731869, 1.670821, -2.07
gt field names and shapes: {'taus': (2,), 'theta': (7,), 'vp_theta': (2,), 'stress_gains': (4,), 'stress_biases': (4,), 'te_clean': (72, 10), 'te_true_rate0': (72,), 'te_hist_true': (72, 18), 'te_hold_true': (72,)}
calibration command ranges: [(0, -2.004396, 1.887501), (1, -1.773961, 1.524142), (2, -0.837586, 2.150495), (3, -1.655924, 1.717156)]
train activation by lab [zero, weak, strong] [(0, 17, 0, 19), (1, 14, 0, 36), (2, 16, 0, 28), (3, 13, 0, 37)]
test activation by lab [zero, weak, strong] [(0, 7, 0, 13), (1, 4, 0, 14), (2, 8, 0, 13), (3, 5, 0, 8)]
```

### Turn 4 — `execute_in_remote_docker` (`90484 ms` LLM)
- **Rationale**: Implement the forward model and robust fit, then test whether the logarithmic terminal basis applies to the whole campaign or only to activated specimens. The public reference reconstruction distinguishes a basis error from an estimation error; the solver itself reads only the dataset.
- **Alternatives Weighed**: `Use a single nonlinear fit from an arbitrary timescale seed` (rejected: Profiling the linear weights on a two-dimensional timescale grid gives physically distinct starts and reduces the chance of selecting a spurious minimum.)

```bash
cat > /holdout_task/alpha_m97_c01_batch_01_solver.py <<'PY'
"""Fit calibrated two-branch creep and predict strain and onset rate."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y):
    dx = x[:, None] - x
    dy = y[:, None] - y
    # The repeated-median line screens EMI without allowing spikes to pull OLS.
    center = np.median([np.median(v[u != 0] / u[u != 0])
                        for u, v in zip(dx, dy)])
    bias = np.median(y - center*x)
    return np.abs(y - center*x - bias) <= 3.0


def _calibrate(c):
    lab, cmd, true = (np.asarray(c[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    result = np.zeros((4, 2))
    history = None
    for channel in range(4):
        take = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[take], true[take]
        spikes = False
        if len(x) >= 3 and np.ptp(x) > 0:
            clean = _line(x, y)
            spikes = not clean.all()
            x, y = x[clean], y[clean]
        if len(x) >= 4 and np.ptp(x) > 0:
            result[channel] = np.polyfit(x, y, 1)
            continue
        if len(x) < 2 or spikes:
            if history is None:
                history = json.loads(_ARCHIVE.read_text())
            # Use the latest usable prefix sweep if the batch cannot resolve a line.
            for sweep in reversed(history):
                take = np.asarray(sweep['lab_id']) == channel
                p = np.asarray(sweep['cmd_stress'])[take]
                q = np.asarray(sweep['true_stress'])[take]
                if len(p) >= 3 and np.ptp(p) > 0:
                    clean = _line(p, q)
                    p, q = p[clean], q[clean]
                if len(p) >= 2 and np.ptp(p) > 0:
                    result[channel] = np.polyfit(p, q, 1)
                    break
            else:
                raise ValueError(f'No usable calibration for load cell {channel}')
        else:
            result[channel] = np.polyfit(x, y, 1)
    return result


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']

    def join(key):
        return np.concatenate((train[key], test[key]), axis=0)

    lab, strata = join('lab_id'), join('stratum')
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*join('past_stress_cmd') + bias[:, None]
    hold = gain*join('stress_hold_cmd') + bias
    x = join('coupon_features_raw').astype(float)
    ref = join('ref_coupons')
    for channel in np.unique(lab):
        take = lab == channel
        panel = ref[take]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        loc = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[take] = (x[take] - loc) / scale
    means = np.array([np.nanmean(x[(lab == i) & (strata == j)], axis=0)
                      for i, j in np.unique(np.column_stack((lab, strata)), axis=0)])
    x = np.where(np.isfinite(x), x, np.nanmean(means, axis=0))
    act = np.tanh(np.maximum(np.abs(past) - dataset['crit_stress'], 0)**2 @ dataset['past_dt'])
    if dataset['shear_banding_index'] > .5:
        act[:] = 0.
    return past, hold, x, act


def _design(logtau, data, dt, time, mode):
    past, hold, x, act = data
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for width, stress in zip(dt, past.T):
        u = width/tau
        a = np.exp(-u)
        dz = a*dz + u*a*(z - stress[:, None])
        z = a*z - np.expm1(-u)*stress[:, None]
    gap = hold[:, None] - z
    u = time[:, None]/tau
    f = -np.expm1(-u)
    df = -u*np.exp(-u)
    plateau = np.log1p(time)
    if mode == 'row':
        terminal = np.where(act[:, None] > 0, plateau, time)
    elif mode == 'linear':
        terminal = time
    else:
        terminal = plateau
    a = np.empty((len(hold), len(time), 9))
    a[:, :, :2] = gap[:, None, :]*f
    a[:, :, 2] = terminal
    a[:, :, 3:7] = a[:, :, 2, None]*x[:, None, :]
    a[:, :, 7] = (act*hold)[:, None]*f[:, 0]
    a[:, :, 8] = (act*z[:, 0])[:, None]*plateau
    j = -dz[:, None, :]*f + gap[:, None, :]*df
    j0 = (act*hold)[:, None]*df[:, 0]
    j1 = (act*dz[:, 0])[:, None]*plateau
    r = np.column_stack((gap/tau, np.ones(len(hold)), x, act*hold/tau[0], act*z[:, 0]))
    jr = (-dz - gap)/tau
    return a, r, j, j0, j1, jr, act*dz[:, 0]


def _scale(v):
    return max(1.4826*np.median(np.abs(v - np.median(v))), 1e-7)


def _dot(a, b):
    return np.einsum('...j,j->...', a, b)


def _fit(data, dt, time, delta, rate, mode):
    ys, yr = np.isfinite(delta), np.isfinite(rate)
    y = delta[ys]
    lower = np.log(max(np.min(dt)*.1, 1e-4))
    upper = np.log(max(np.max(time), np.sum(dt))*12)
    grid = np.geomspace(max(np.min(dt)*.55, 1e-4), max(np.max(time), np.sum(dt))*1.25, 9)
    seeds = []
    scale_y = _scale(y)
    for i, fast in enumerate(grid[:-1]):
        for slow in grid[i+1:]:
            logs = np.log([fast, slow])
            a = _design(logs, data, dt, time, mode)[0][ys]
            b = np.linalg.lstsq(a, y, rcond=None)[0]
            for width in (.2, .06, .02):
                rootw = 1./np.sqrt(np.hypot(1., (_dot(a, b)-y)/(width*scale_y)))
                b = np.linalg.lstsq(a*rootw[:, None], y*rootw, rcond=None)[0]
            seeds.append((np.median(np.abs(_dot(a,b) - y)), np.r_[logs, b]))
    seeds.sort(key=lambda s: s[0])
    initial = seeds[0][1]
    runner = next(s[1] for s in seeds[1:] if np.linalg.norm(s[1][:2]-initial[:2]) > .8)
    bnd = (np.r_[lower, lower, np.full(9, -np.inf)], np.r_[upper, upper, np.full(9, np.inf)])

    def components(p):
        return _design(p[:2], data, dt, time, mode)

    def residual(p, sc, sr):
        a, r, *_ = components(p)
        return np.r_[(_dot(a[ys], p[2:]) - y)/sc, (_dot(r[yr], p[2:]) - rate[yr])/sr]

    def jacobian(p, sc, sr):
        a, r, j, j0, j1, jr, jr1 = components(p)
        jc = j*p[2:4]
        jc[:, :, 0] += j0*p[9] + j1*p[10]
        jrate = jr*p[2:4]
        jrate[:, 0] += -r[:, 7]*p[9] + jr1*p[10]
        return np.vstack((np.column_stack((jc[ys], a[ys]))/sc,
                          np.column_stack((jrate[yr], r[yr]))/sr))

    a, r, *_ = components(initial)
    sc = _scale(_dot(a[ys], initial[2:]) - y)
    sr = _scale(_dot(r[yr], initial[2:]) - rate[yr]) if yr.any() else 1.
    opts = []
    for seed in (initial, runner):
        for width in (3., 1.5):
            opt = least_squares(residual, seed, jac=jacobian, args=(sc, sr), bounds=bnd,
                                loss='cauchy', f_scale=width, x_scale='jac', max_nfev=120)
            seed = opt.x
        opts.append(opt)
    best = min(opts, key=lambda opt: opt.cost).x
    a, r, *_ = components(best)
    sc = _scale(_dot(a[ys], best[2:]) - y)
    sr = _scale(_dot(r[yr], best[2:]) - rate[yr]) if yr.any() else 1.
    return least_squares(residual, best, jac=jacobian, args=(sc, sr), bounds=bnd,
                         loss='cauchy', f_scale=2., x_scale='jac', max_nfev=100).x


def solve_alpha_m97_c01_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = dataset['past_dt'], dataset['eval_times']
    data = _prepare(dataset)
    n = len(train['lab_id'])
    delta = train['strain_curves_obs'] - train['strain_0'][:, None]
    mode = 'row' if dataset['shear_banding_index'] <= .5 else 'linear'
    p = _fit(tuple(a[:n] for a in data), dt, time, delta, train['instant_rate_obs'], mode)
    a, r, *_ = _design(p[:2], tuple(a[n:] for a in data), dt, time, mode)
    return dict(estimated_taus=np.exp(p[:2]), hereditary_weights=p[2:9], regime_mode_weights=p[9:],
                predicted_strain=test['strain_0'][:, None] + _dot(a, p[2:]),
                initial_rate_pred=_dot(r, p[2:]))
PY
cat >/tmp/rheo_compare.py <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/alpha_m97_c01_batch_01_solver.py'; sp=importlib.util.spec_from_file_location('cand',p); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0];d,g=it['dataset'],it['gt'];tr,te=d['train'],d['test'];data=m._prepare(d);n=len(tr['lab_id']);dt,t=d['past_dt'],d['eval_times'];xtr=tuple(a[:n] for a in data);xte=tuple(a[n:] for a in data);y=tr['strain_curves_obs']-tr['strain_0'][:,None];r=tr['instant_rate_obs']; truth=np.r_[g['taus'],g['theta'],g['vp_theta']];print('public reference parameters:',truth)
for mode in ('row','log','linear'):
 p=m._fit(xtr,dt,t,y,r,mode);a,b,*_=m._design(p[:2],xtr,dt,t,mode);u=m._dot(a,p[2:])-y;v=m._dot(b,p[2:])-r
 print(mode,'parameters',np.r_[np.exp(p[:2]),p[2:]])
 for active in (0,1):
  sel=(xtr[-1]>0)==active; print('training',active,'abs residual q50,q90,q99, max',np.quantile(abs(u[sel]),[.5,.9,.99,1]),np.quantile(abs(v[sel]),[.5,.9,.99,1]))
 a,b,*_=m._design(p[:2],xte,dt,t,mode);u=m._dot(a,p[2:])+te['strain_0'][:,None]-g['te_clean'];v=m._dot(b,p[2:])-g['te_true_rate0']
 print('test RMSE strain/rate',np.sqrt(np.mean(u*u)),np.sqrt(np.mean(v*v)))
 q=np.r_[np.log(g['taus']),truth[2:]];a,b,*_=m._design(q[:2],xte,dt,t,mode);print('reference-parameter reconstruction max',np.max(abs(m._dot(a,q[2:])+te['strain_0'][:,None]-g['te_clean'])),np.max(abs(m._dot(b,q[2:])-g['te_true_rate0'])))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rheo_compare.py
```
- **Container Exit Code**: `0` (`801 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
public reference parameters: [ 0.613  3.672  0.68   0.55   0.14   0.16  -0.13   0.12  -0.09   0.28
  0.24 ]
row parameters [ 0.61498286  3.25579156  0.67579792  0.52496193  0.11324814  0.13696157
 -0.11545408  0.09965965 -0.07769776  0.28091343  0.27981174]
training 0 abs residual q50,q90,q99, max [ 0.1557373   0.74015381  2.02503764 15.77285267] [0.03722772 0.08521842 0.1147257  0.13834085]
training 1 abs residual q50,q90,q99, max [ 0.03797059  0.18697548  1.93894094 23.07542475] [0.03720404 0.07961233 0.25456784 0.28983328]
test RMSE strain/rate 0.2825293392022826 0.056492778244921306
reference-parameter reconstruction max 2.2816921395448526 0.319765173569722
log parameters [ 0.61306803  3.69117524  0.67906978  0.55228462  0.13997322  0.15669885
 -0.12708235  0.11753194 -0.08865097  0.28065607  0.24166707]
training 0 abs residual q50,q90,q99, max [1.16248692e-02 9.98969259e-02 1.52715789e+00 1.56464778e+01] [0.0125638  0.04160829 0.12636685 0.13834188]
training 1 abs residual q50,q90,q99, max [1.32841933e-02 1.78203873e-01 1.91528323e+00 2.30122140e+01] [0.01458813 0.03717873 0.26778598 0.35191548]
test RMSE strain/rate 0.054866343065982 0.04173357451873751
reference-parameter reconstruction max 0.5834023019017318 0.319765173569722
linear parameters [ 0.61806417  3.89134453  0.67514823  0.56963048  0.05682498  0.06392667
 -0.05135194  0.04795052 -0.03638641  0.28641243  0.24722523]
training 0 abs residual q50,q90,q99, max [ 0.04062871  0.14823703  1.51280364 15.62953879] [0.11093047 0.30275486 0.45999456 0.47406783]
training 1 abs residual q50,q90,q99, max [ 0.04563583  0.21116704  1.92425796 23.10220303] [0.13265022 0.27015106 0.4354881  0.46303684]
test RMSE strain/rate 0.08762096032673149 0.19166952502212942
reference-parameter reconstruction max 2.392257760229276 0.319765173569722

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7c5f8c9085c0b48c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7c5f8c90 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`29551 ms` LLM)
- **Rationale**: Separate residuals by missingness and activation, then substitute public reference stresses in the diagnostic only. If the discrepancy persists as a pure logarithmic drift error, focus on coupon alignment or imputation rather than changing the calibrated stress or memory model.
- **Alternatives Weighed**: `Retune the optimizer before diagnosing the remaining error` (rejected: Substituting public reference parameters still leaves a 0.31977 maximum rate discrepancy, so optimizer tuning alone cannot explain it.)
- **ATMS NOGOOD DECLARED**: Refuted `Zero work damage switches the terminal drift basis from log(1+t) to t within this crystallization campaign.` | Witness: `On inactive training specimens, the median absolute strain residual drops from 0.15574 to 0.01162 with the logarithmic basis; public test strain RMSE drops from 0.28253 to 0.05487.` | Blanket Diagnosis: `The specimen work-damage gate controls the two additive regime corrections; applying it to the terminal basis also changes the drift of inactive specimens.` | Sublated Pivot: `Use logarithmic terminal drift throughout the crystallization campaign while retaining specimen-level activation on both regime corrections.`

```bash
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/alpha_m97_c01_batch_01_solver.py'; s=importlib.util.spec_from_file_location('cand',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0]; d,g=it['dataset'],it['gt']; n=len(d['train']['lab_id']); te=d['test']; data=m._prepare(d); dt,t=d['past_dt'],d['eval_times']; w=np.r_[g['theta'],g['vp_theta']]; missing=np.isnan(te['coupon_features_raw']); act=data[-1][n:]; print('gain/bias errors:',m._calibrate(d['calibration_load_cells'])-np.column_stack((g['stress_gains'],g['stress_biases'])))
for label,past,hold in [('calibrated',data[0][n:],data[1][n:]),('public reference stress',g['te_hist_true'],g['te_hold_true'])]:
 a,r,*_=m._design(np.log(g['taus']),(past,hold,data[2][n:],np.tanh(np.maximum(np.abs(past)-d['crit_stress'],0)**2@dt)),dt,t,'log'); e=m._dot(a,w)+te['strain_0'][:,None]-g['te_clean']; er=m._dot(r,w)-g['te_true_rate0']; shape=e-er[:,None]*np.log1p(t); print(label)
 for name,sel in [('complete',~missing.any(axis=1)),('missing',missing.any(axis=1)),('inactive',act==0),('active',act>0)]:
  print(name,sel.sum(),'strain/rate RMSE',np.sqrt(np.mean(e[sel]**2)),np.sqrt(np.mean(er[sel]**2)),'max non-drift residual',np.max(abs(shape[sel])))
 if label=='public reference stress':
  for k in np.flatnonzero((abs(er)>0.005)|missing.any(axis=1)):
   print('row',k,'cell',(te['lab_id'][k],te['stratum'][k]),'mask',missing[k].astype(int),'rate residual',round(er[k],6),'max shape error',round(np.max(abs(shape[k])),6))
PY
```
- **Container Exit Code**: `0` (`531 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
E 0.23513860172944206 0.17955291974370985 max non-drift residual 1.5543122344752192e-15
inactive 24 strain/rate RMSE 0.04021130704433558 0.030705539343254686 max non-drift residual 1.0412677664550785e-15
active 48 strain/rate RMSE 0.06254012462820213 0.04775592733620971 max non-drift residual 1.7338561142388187e-15
row 0 cell (0, 0) mask [0 0 0 0] rate residual 0.011065 max shape error 0.0
row 3 cell (3, 0) mask [0 0 0 0] rate residual 0.010389 max shape error 0.0
row 4 cell (0, 0) mask [0 0 0 0] rate residual 0.005404 max shape error 0.0
row 8 cell (0, 2) mask [0 0 0 0] rate residual -0.013181 max shape error 0.0
row 12 cell (0, 2) mask [0 0 0 1] rate residual 0.071492 max shape error 0.0
row 17 cell (0, 0) mask [0 0 0 0] rate residual 0.007928 max shape error 0.0
row 19 cell (0, 0) mask [0 0 0 0] rate residual 0.01123 max shape error 0.0
row 20 cell (0, 2) mask [0 0 0 0] rate residual -0.007011 max shape error 0.0
row 22 cell (3, 2) mask [0 0 0 0] rate residual 0.009638 max shape error 0.0
row 30 cell (0, 1) mask [0 0 0 0] rate residual 0.010224 max shape error 0.0
row 34 cell (0, 1) mask [0 0 0 0] rate residual 0.014977 max shape error 0.0
row 37 cell (0, 0) mask [0 0 0 0] rate residual 0.008543 max shape error 0.0
row 40 cell (2, 0) mask [0 1 0 0] rate residual 0.024919 max shape error 0.0
row 41 cell (0, 0) mask [0 0 0 0] rate residual 0.009444 max shape error 0.0
row 43 cell (1, 1) mask [1 0 0 0] rate residual 0.319743 max shape error 0.0
row 49 cell (0, 0) mask [0 0 0 0] rate residual -0.018554 max shape error 0.0
row 50 cell (2, 0) mask [0 0 0 0] rate residual 0.00521 max shape error 0.0
row 51 cell (0, 0) mask [0 0 0 0] rate residual -0.017204 max shape error 0.0
row 52 cell (2, 1) mask [0 0 0 0] rate residual -0.012407 max shape error 0.0
row 55 cell (0, 2) mask [0 0 0 0] rate residual -0.015065 max shape error 0.0
row 58 cell (0, 1) mask [0 0 0 0] rate residual -0.007831 max shape error 0.0
row 61 cell (2, 0) mask [0 0 0 0] rate residual 0.006306 max shape error 0.0
row 67 cell (3, 1) mask [0 0 0 0] rate residual 0.012843 max shape error 0.0
row 68 cell (0, 2) mask [0 0 1 0] rate residual -0.144878 max shape error 0.0
row 69 cell (0, 1) mask [0 0 0 0] rate residual 0.014728 max shape error 0.0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7c5f8c9085c0b48c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7c5f8c90 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`33752 ms` LLM)
- **Rationale**: Commit the supported terminal basis and run the isolated schema check. Diagnose whether missing-coupon errors in measured training data are consistent with lost feature information or show a reproducible lab or stratum effect that warrants an out-of-fold comparison.
- **Alternatives Weighed**: `Change the stress calibration or Prony recurrence` (rejected: Using reference stresses eliminates the non-drift residual to numerical precision but leaves the missing-coupon error essentially unchanged.)

```bash
python3 - <<'PY'
from pathlib import Path
p=Path('/holdout_task/alpha_m97_c01_batch_01_solver.py')
s=p.read_text(); s=s.replace("    if mode == 'row':\n        terminal = np.where(act[:, None] > 0, plateau, time)\n    elif mode == 'linear':\n        terminal = time\n    else:\n        terminal = plateau\n", "    terminal = time if mode == 'linear' else plateau\n"); s=s.replace("mode = 'row' if", "mode = 'log' if"); p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/alpha_m97_c01_batch_01_solver.py';s=importlib.util.spec_from_file_location('cand',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0]['dataset']; tr=d['train'];te=d['test']; n=len(tr['lab_id']); data=m._prepare(d); dt,t=d['past_dt'],d['eval_times']; xtr=tuple(a[:n] for a in data); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; p=m._fit(xtr,dt,t,y,r,'log'); a,b,*_=m._design(p[:2],xtr,dt,t,'log'); e=np.median((y-m._dot(a,p[2:]))/np.log1p(t),axis=1); er=r-m._dot(b,p[2:]); raw=np.r_[tr['coupon_features_raw'],te['coupon_features_raw']]; missing=np.isnan(raw[:n]); labs=tr['lab_id']; strata=tr['stratum']; print('complete versus missing strain-inferred drift / rate error quantiles:');
for take in (~missing.any(1),missing.any(1)):
 print(take.sum(),np.quantile(abs(e[take]),[.25,.5,.75,.9,1]),np.quantile(abs(er[take]),[.25,.5,.75,.9,1]),np.corrcoef(e[take],er[take])[0,1])
print('aligned complete-case feature correlation:',np.round(np.corrcoef(data[2][np.isfinite(raw).all(1)].T),3));
for i in range(4):
 take=(labs==i)&(~missing.any(1)); print('lab',i,'complete drift bias/MAD',np.median(e[take]),np.median(abs(e[take]-np.median(e[take]))));
 for j in range(3):
  w=(labs==i)&(strata==j); print('cell',i,j,'n',sum(w),'x means',np.round(np.nanmean(np.where(missing[w],np.nan,xtr[2][w]),axis=0),3))
print('train missing rows: lab,stratum,feature,strain-inferred drift error,rate error')
for i,j in zip(*np.where(missing)): print(labs[i],strata[i],j,round(e[i],4),round(er[i],4))
PY
```
- **Container Exit Code**: `0` (`799 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c01_batch_01_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
complete versus missing strain-inferred drift / rate error quantiles:
161 [0.00189659 0.00333967 0.00616949 0.00950256 0.01704269] [0.00499379 0.01248035 0.02101752 0.02821154 0.04918883] 0.2032929871152694
19 [0.05546791 0.07829357 0.13422537 0.24629778 0.336162  ] [0.04329806 0.08706478 0.13134214 0.25093299 0.35191548] 0.9893685444764885
aligned complete-case feature correlation: [[ 1.     0.049 -0.015  0.055]
 [ 0.049  1.    -0.006  0.042]
 [-0.015 -0.006  1.     0.08 ]
 [ 0.055  0.042  0.08   1.   ]]
lab 0 complete drift bias/MAD 0.0012540803006832655 0.004714452712534534
cell 0 0 n 13 x means [-0.264 -0.53  -0.715 -0.267]
cell 0 1 n 10 x means [-0.045  0.72  -0.015 -0.204]
cell 0 2 n 13 x means [0.501 0.166 0.648 0.021]
lab 1 complete drift bias/MAD 0.00018918598373547757 0.003065132761274968
cell 1 0 n 12 x means [-0.438 -0.12  -0.237 -0.439]
cell 1 1 n 23 x means [-0.131  0.344  0.483  0.191]
cell 1 2 n 15 x means [ 0.491  0.409 -0.482  0.574]
lab 2 complete drift bias/MAD 0.00046027237694728064 0.0028531603232532002
cell 2 0 n 13 x means [-0.152 -0.023  0.183 -0.31 ]
cell 2 1 n 7 x means [ 0.492 -0.23  -0.213  0.019]
cell 2 2 n 24 x means [0.264 0.203 0.414 0.324]
lab 3 complete drift bias/MAD 0.00011427625222971342 0.004357348635142067
cell 3 0 n 19 x means [-0.222  0.211 -0.266 -0.421]
cell 3 1 n 13 x means [ 0.144 -0.03   0.531 -0.032]
cell 3 2 n 18 x means [-0.042  0.075  0.825  0.098]
train missing rows: lab,stratum,feature,strain-inferred drift error,rate error
0 0 1 0.243 0.2454
1 2 2 0.1544 0.1243
3 1 0 -0.0571 -0.0261
0 2 3 0.026 0.0028
3 0 2 -0.0668 -0.0371
0 0 1 0.1536 0.1486
1 1 1 0.1046 0.1106
1 0 0 -0.3362 -0.3519
0 0 2 -0.2595 -0.273
0 2 3 -0.0292 0.0021
1 0 0 -0.0988 -0.0871
3 2 3 -0.0355 -0.0436
3 2 0 -0.068 -0.0736
1 2 2 -0.1148 -0.118
3 2 3 -0.0456 -0.0479
2 0 3 -0.1117 -0.1383
1 2 3 -0.0761 -0.0939
1 1 1 0.0783 0.043
1 0 2 -0.0538 -0.0806

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3b67a1ab2d0b7bab (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_3b67a1ab (directory).
```

### Turn 7 — `execute_in_remote_docker` (`27744 ms` LLM)
- **Rationale**: Compare the two fits on measured leave-one-lab-out errors without using public reference labels to choose. If training on complete coupons reduces worst-lab error without degrading other slices, use it to estimate the shared parameters while keeping the required imputation for prediction.
- **Alternatives Weighed**: `Predict missing features from the observed coupon columns` (rejected: Observed feature correlations are weak, and the requested prediction path uses cell-balanced means.)

```bash
cat >/tmp/rheo_cv.py <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c01_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0]; d=it['dataset']; tr=d['train']; te=d['test']; n=len(tr['lab_id']); lab=tr['lab_id']; dt,t=d['past_dt'],d['eval_times']; xx=m._prepare(d); x=tuple(a[:n] for a in xx); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; cc=np.isfinite(tr['coupon_features_raw']).all(1)
def report(y,v,r,w,sl):
    return np.round([np.median(abs(y[sl]-v[sl])),np.median(abs(r[sl]-w[sl]))],6)
for fit_cc in (False,True):
    print('complete-only fit',fit_cc)
    for i in np.unique(lab):
        hold=lab==i; fit=~hold & (cc if fit_cc else True)
        p=m._fit(tuple(a[fit] for a in x),dt,t,y[fit],rate[fit],'log')
        a,b,*_=m._design(p[:2],x,dt,t,'log'); print('lab',i,'taus',np.round(np.exp(p[:2]),5),'strain/rate median absolute errors',[(tag,report(y,m._dot(a,p[2:]),rate,m._dot(b,p[2:]),hold&sl).tolist()) for tag,sl in [('active-complete',cc&(x[-1]>0)),('inactive-complete',cc&(x[-1]==0)),('missing',~cc)]])
    fit=cc if fit_cc else np.ones(n,dtype=bool)
    p=m._fit(tuple(a[fit] for a in x),dt,t,y[fit],rate[fit],'log'); print('full refit parameters',np.round(np.r_[np.exp(p[:2]),p[2:]],6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheo_cv.py
```
- **Container Exit Code**: `0` (`1260 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
complete-only fit False
lab 0 taus [0.61346 3.7206 ] strain/rate median absolute errors [('active-complete', [0.012442, 0.013986]), ('inactive-complete', [0.013045, 0.008321]), ('missing', [0.169996, 0.153226])]
lab 1 taus [0.61223 3.65414] strain/rate median absolute errors [('active-complete', [0.01282, 0.017894]), ('inactive-complete', [0.011052, 0.015308]), ('missing', [0.125961, 0.106124])]
lab 2 taus [0.61373 3.69676] strain/rate median absolute errors [('active-complete', [0.011024, 0.006435]), ('inactive-complete', [0.010382, 0.01085]), ('missing', [0.143426, 0.141469])]
lab 3 taus [0.61289 3.69311] strain/rate median absolute errors [('active-complete', [0.011097, 0.01578]), ('inactive-complete', [0.009324, 0.014787]), ('missing', [0.069014, 0.044973])]
full refit parameters [ 0.613068  3.691175  0.67907   0.552285  0.139973  0.156699 -0.127082
  0.117532 -0.088651  0.280656  0.241667]
complete-only fit True
lab 0 taus [0.61364 3.7122 ] strain/rate median absolute errors [('active-complete', [0.01203, 0.013347]), ('inactive-complete', [0.012795, 0.007896]), ('missing', [0.170095, 0.151947])]
lab 1 taus [0.61271 3.66898] strain/rate median absolute errors [('active-complete', [0.0124, 0.017182]), ('inactive-complete', [0.010875, 0.014785]), ('missing', [0.127209, 0.106227])]
lab 2 taus [0.61422 3.71004] strain/rate median absolute errors [('active-complete', [0.011378, 0.007474]), ('inactive-complete', [0.009674, 0.011012]), ('missing', [0.143605, 0.139666])]
lab 3 taus [0.61393 3.71965] strain/rate median absolute errors [('active-complete', [0.011165, 0.015087]), ('inactive-complete', [0.009328, 0.014812]), ('missing', [0.068756, 0.044087])]
full refit parameters [ 0.613578  3.702149  0.681083  0.551864  0.140705  0.156611 -0.127563
  0.1174   -0.088553  0.279806  0.24186 ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3b67a1ab2d0b7bab (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_3b67a1ab (directory).
```

### Turn 8 — `execute_in_remote_docker` (`30145 ms` LLM)
- **Rationale**: Check whether the small complete-coupon drift residual is fully explained by a per-lab affine panel difference. If it is not, retain the specified pooled panel rather than adding a nuisance correction. Confirm the archive layout before testing the documented sparse-calibration branch.
- **Alternatives Weighed**: `Switch to complete-only fitting` (rejected: It slightly improves worst complete-coupon medians but does not improve worst missing-coupon strain and moves the slow timescale farther from the public reference. Keep the all-coupon robust fit for now.)

```bash
set -e
sed -n '60,105p' /tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py
python3 - <<'PY'
import json
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('archive layout:',type(h).__name__,len(h),list(h[0]) if isinstance(h,list) else list(h))
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/alpha_m97_c01_batch_01_solver.py'; s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0];d,g=it['dataset'],it['gt'];tr,te=d['train'],d['test'];x=te['coupon_features_raw']; n=len(tr['lab_id']);p,h,xx,act=m._prepare(d);p=g['te_hist_true'];h=g['te_hold_true'];dt,t=d['past_dt'],d['eval_times'];a,b,*_=m._design(np.log(g['taus']),(p,h,xx[n:],act[n:]),dt,t,'log');w=np.r_[g['theta'],g['vp_theta']]; raw_drift=g['te_true_rate0']-m._dot(b,w)+xx[n:]@w[3:7];print('Reference panel train/test shapes:',tr['ref_coupons'].shape,te['ref_coupons'].shape)
for i in range(4):
 take=(te['lab_id']==i)&np.isfinite(x).all(1); v=x[take]@w[3:7]; y=raw_drift[take]; A=np.column_stack((v,np.ones(len(v)))); coef=np.linalg.lstsq(A,y,rcond=None)[0]; print('lab',i,'complete counts',sum(take),'implied scalar center/scale',[-coef[1]/coef[0]/sum(w[3:7]),1/coef[0]],'unexplained max',max(abs(A@coef-y)))
 for label,rows in [('train',tr['ref_coupons'][tr['lab_id']==i]),('test',te['ref_coupons'][te['lab_id']==i]),('both',np.r_[tr['ref_coupons'][tr['lab_id']==i],te['ref_coupons'][te['lab_id']==i]])]:
  print(' ',label,'counts',np.isfinite(rows).sum(0).tolist(),'median mean/std',np.round([np.nanmedian(np.nanmean(rows,axis=0)),np.nanmedian(np.nanstd(rows,axis=0))],6))
PY
```
- **Container Exit Code**: `0` (`465 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
9, 18] median mean/std [-0.654269  0.774925]
  both counts [53, 55, 54, 55, 56, 55, 56, 52, 52, 53, 52, 54, 54, 55, 54, 56, 55, 55, 52, 53, 53, 54, 51, 55, 54, 52, 55, 52, 53, 54, 54, 52] median mean/std [-0.631707  0.788584]
lab 1 complete counts 17 implied scalar center/scale [0.4800052325458616, 1.280002247807904] unexplained max 4.6483562685661894e-06
  train counts [49, 47, 49, 49, 49, 47, 49, 46, 45, 49, 48, 48, 48, 46, 47, 48, 50, 48, 49, 49, 47, 50, 48, 44, 49, 47, 46, 48, 49, 49, 49, 45] median mean/std [0.499049 1.233156]
  test counts [16, 17, 17, 18, 18, 16, 18, 18, 16, 17, 16, 18, 18, 18, 18, 18, 17, 17, 17, 18, 18, 17, 18, 17, 18, 17, 18, 18, 16, 18, 17, 16] median mean/std [0.526872 1.199304]
  both counts [65, 64, 66, 67, 67, 63, 67, 64, 61, 66, 64, 66, 66, 64, 65, 66, 67, 65, 66, 67, 65, 67, 66, 61, 67, 64, 64, 66, 65, 67, 66, 61] median mean/std [0.511879 1.265833]
lab 2 complete counts 20 implied scalar center/scale [-0.25000433493065377, 0.9400039196731671] unexplained max 6.981096847824553e-06
  train counts [44, 44, 42, 43, 42, 40, 43, 40, 43, 43, 43, 42, 44, 43, 40, 43, 43, 44, 44, 40, 44, 42, 42, 44, 41, 42, 44, 44, 42, 41, 44, 43] median mean/std [-0.259243  0.92716 ]
  test counts [21, 21, 21, 21, 20, 19, 18, 20, 21, 20, 19, 21, 19, 19, 21, 20, 20, 20, 20, 20, 19, 20, 21, 20, 21, 20, 20, 21, 21, 20, 21, 20] median mean/std [-0.286825  0.898675]
  both counts [65, 65, 63, 64, 62, 59, 61, 60, 64, 63, 62, 63, 63, 62, 61, 63, 63, 64, 64, 60, 63, 62, 63, 64, 62, 62, 64, 65, 63, 61, 65, 63] median mean/std [-0.246415  0.924491]
lab 3 complete counts 13 implied scalar center/scale [0.8000662299219053, 1.419990648754431] unexplained max 2.0119823212899623e-05
  train counts [49, 50, 49, 49, 49, 46, 47, 49, 49, 46, 49, 44, 48, 49, 47, 49, 47, 49, 47, 47, 50, 50, 42, 49, 48, 48, 46, 49, 47, 44, 47, 47] median mean/std [0.819356 1.363165]
  test counts [11, 13, 13, 12, 13, 11, 13, 12, 12, 13, 11, 13, 12, 12, 13, 13, 11, 12, 12, 13, 12, 13, 12, 12, 11, 12, 13, 11, 13, 12, 12, 13] median mean/std [0.848449 1.352774]
  both counts [60, 63, 62, 61, 62, 57, 60, 61, 61, 59, 60, 57, 60, 61, 60, 62, 58, 61, 59, 60, 62, 63, 54, 61, 59, 60, 59, 60, 60, 56, 59, 60] median mean/std [0.830968 1.382014]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3b67a1ab2d0b7bab (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_3b67a1ab (directory).
```

### Turn 9 — `execute_in_remote_docker` (`41700 ms` LLM)
- **Rationale**: Verify the exact candidate in isolation. Sparse and EMI checks detect incorrect calibration routing; independent noiseless recovery can expose a forward-model or Jacobian error that robust fitting on noisy measurements could hide.
- **Alternatives Weighed**: `Replace the prescribed panel statistics with inferred lab constants` (rejected: The reference curves reflect latent panel constants, but the solver contract calls for estimating alignment from the observed pooled panels; deriving constants from the evaluation targets would leak labels.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import copy, importlib.util, json, pickle
import numpy as np
p='/holdout_task/alpha_m97_c01_batch_01_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0]; d,g=it['dataset'],it['gt']; c=d['calibration_load_cells']; ref=np.column_stack((g['stress_gains'],g['stress_biases']))
original_path=m._ARCHIVE
m._ARCHIVE=original_path.with_name('not_present_for_direct_fit.json')
np.testing.assert_allclose(m._calibrate(c),ref,atol=5e-5,rtol=0)
for lab in range(4):
    idx=np.flatnonzero(c['lab_id']==lab)
    for j in (idx[0],idx[np.argmax(c['cmd_stress'][idx])]):
        emi={k:np.array(v,copy=True) for k,v in c.items()}; emi['true_stress'][j]+=4.5
        np.testing.assert_allclose(m._calibrate(emi),ref,atol=7e-5,rtol=0)
m._ARCHIVE=original_path
archive=json.loads(original_path.read_text())[-1]
archive_cal=m._calibrate(archive)
for lab in range(4):
    idx=np.flatnonzero(c['lab_id']==lab)
    others=c['lab_id']!=lab
    for num in (0,1,2,3):
        take=others.copy(); take[idx[:num]]=True
        sparse={k:np.asarray(v)[take] for k,v in c.items()}
        expect=archive_cal[lab] if num<2 else np.polyfit(sparse['cmd_stress'][~others[take]],sparse['true_stress'][~others[take]],1)
        np.testing.assert_allclose(m._calibrate(sparse)[lab],expect,atol=1e-12,rtol=0)
print('Calibration: all four channels pass direct, EMI, and sparse routing checks')
out=m.solve_alpha_m97_c01_batch_01(d); te=d['test']; e=out['predicted_strain']-g['te_clean']; r=out['initial_rate_pred']-g['te_true_rate0']; n=len(d['train']['lab_id']); active=m._prepare(d)[-1][n:]>0; complete=np.isfinite(te['coupon_features_raw']).all(1)
for tag,sel in [('active',active),('inactive',~active),('complete',complete),('missing',~complete)]+[(f'lab {i}',te['lab_id']==i) for i in range(4)]:
    print(tag, int(sel.sum()),'strain/rate MAE',np.mean(abs(e[sel])),np.mean(abs(r[sel])))
print('parameter max absolute errors:',*[np.max(abs(out[k]-g[v])) for k,v in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]])
assert np.max(abs(out['estimated_taus']/g['taus']-1)) < .015
assert np.max(abs(out['hereditary_weights']-g['theta'])) < .012
assert np.max(abs(out['regime_mode_weights']-g['vp_theta'])) < .012
assert np.mean(abs(e[complete])) < .02 and np.mean(abs(r[complete])) < .015
# Generate exact labels from the prescribed design, independently of the public ground truth.
clean=copy.deepcopy(d); dt, t=clean['past_dt'],clean['eval_times']; tr=clean['train']; past,hold,x,act=(a[:n] for a in m._prepare(clean)); tau=np.array([.51,3.08]); hw=np.array([.7,.42,.11,.09,-.12,.16,-.07]); rw=np.array([.24,.17]); z=np.zeros((n,2))
for j,s in enumerate(dt):
    f=np.exp(-s/tau); z=z*f+past[:,j,None]*(1-f)
gap=hold[:,None]-z; amp=gap*hw[:2]; amp[:,0]+=rw[0]*act*hold; drift=hw[2]+x@hw[3:]+rw[1]*act*z[:,0]
tr['strain_curves_obs']=tr['strain_0'][:,None]+amp@(1-np.exp(-t[None,:]/tau[:,None]))+drift[:,None]*np.log1p(t)
tr['instant_rate_obs']=amp@(1/tau)+drift
exact=m.solve_alpha_m97_c01_batch_01(clean)
for k,v in [('estimated_taus',tau),('hereditary_weights',hw),('regime_mode_weights',rw)]: np.testing.assert_allclose(exact[k],v,atol=1e-5,rtol=0)
print('Exact synthetic recovery: both timescales and all nine weights')
PY
```
- **Container Exit Code**: `0` (`1042 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Calibration: all four channels pass direct, EMI, and sparse routing checks
active 48 strain/rate MAE 0.01330569638257108 0.011321014173391028
inactive 24 strain/rate MAE 0.010445722625616449 0.009180363529536347
complete 68 strain/rate MAE 0.0033236383162881396 0.0032474241560480513
missing 4 strain/rate MAE 0.16584084096765328 0.13572814060509353
lab 0 20 strain/rate MAE 0.01761062987945282 0.014414859628171845
lab 1 18 strain/rate MAE 0.0249349064964639 0.02082633490430924
lab 2 21 strain/rate MAE 0.0030308595998515026 0.0032759905683456443
lab 3 13 strain/rate MAE 0.0018986002504552512 0.0024438755581061436
parameter max absolute errors: 0.019175236596449263 0.0033011502228878498 0.0016670686843609783
Exact synthetic recovery: both timescales and all nine weights

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3b67a1ab2d0b7bab (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_3b67a1ab (directory).
```

### Turn 10 — `submit_completed_task` (`14318 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema and recovery checks. Leave-one-lab-out validation identifies missing coupon features as the remaining error bottleneck; the solver retains the prescribed cell-balanced imputation.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/alpha_m97_c01_batch_01_solver.py': 'FILE:7683'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_m97_c01_batch_01_solver.py`. The public schema, sparse and EMI calibration checks, and exact recovery of both relaxation times and all nine weights passed. Public strain and initial-rate MAE were 0.00332 and 0.00325 on complete coupons, versus 0.16584 and 0.13573 on the four coupons with missing features.`

### Turn 11 — `execute_in_remote_docker` (`30930 ms` LLM)
- **Rationale**: Fit residual corrections using only non-held labs and compare worst held-lab slices. If new terms improve those slices consistently, investigate omitted structure; otherwise retain the specified basis. Signed residuals by lab test whether pooled performance is masking a laboratory offset.
- **Alternatives Weighed**: `Run balanced-accuracy, relay-mode, and Fokker–Planck checks` (rejected: This dataset has continuous strain and rate measurements, no classification labels or relay state, and a specified deterministic memory recurrence. Held-lab residuals directly test the applicable nuisance and model-basis risks.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
cat > /tmp/rheo_stress_audit.py <<'PY'
import importlib.util, pickle
import numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c01_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0]; d=it['dataset']; tr=d['train']; lab=tr['lab_id']; n=len(lab); dt,t=d['past_dt'],d['eval_times']; data=tuple(a[:n] for a in m._prepare(d)); past,hold,x,act=data; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; cc=np.isfinite(tr['coupon_features_raw']).all(1); L=np.log1p(t)
# Pairwise sums and differences are already in the four-feature linear span.
# Probe genuinely new terms: pairwise interactions, curvatures and saturated damage response.
inter=np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)])
damage=np.maximum(np.abs(past)-d['crit_stress'],0)**2@dt
sat=np.column_stack((act*hold/(1+damage**2),act*hold/(1+hold**2)))
rows=[]
for k in np.unique(lab):
    keep=lab!=k; p=m._fit(tuple(a[keep] for a in data),dt,t,y[keep],rate[keep],'log'); a,b,*_=m._design(p[:2],data,dt,t,'log'); ey=y-m._dot(a,p[2:]); er=rate-m._dot(b,p[2:]); drift=np.median(ey/L,axis=1); fit=keep & cc
    print('held lab', k, 'signed complete implied drift/rate medians', np.round([np.median(drift[~keep&cc]),np.median(er[~keep&cc])],6))
    for name,u in [('baseline',np.zeros((n,1))),('coupon interactions',inter),('rational damage',sat)]:
        coef=np.linalg.lstsq(u[fit],drift[fit],rcond=None)[0]; corr=u@coef
        for sl,v in [('active complete',cc&(act>0)),('inactive complete',cc&(act==0)),('missing',~cc)]:
            sel=~keep&v
            rows.append((name,sl,np.median(abs(ey[sel]-corr[sel,None]*L)),np.median(abs(er[sel]-corr[sel]))))
for name in ('baseline','coupon interactions','rational damage'):
    for sl in ('active complete','inactive complete','missing'):
        z=np.array([(a,b) for nm,ss,a,b in rows if nm==name and ss==sl]); print(name,sl,'worst lab median strain/rate',np.round(z.max(axis=0),6))
PY
python3 -I /tmp/rheo_stress_audit.py
python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1394 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held lab 0 signed complete implied drift/rate medians [0.001147 0.003009]
held lab 1 signed complete implied drift/rate medians [-0.000351  0.001292]
held lab 2 signed complete implied drift/rate medians [ 0.000817 -0.002211]
held lab 3 signed complete implied drift/rate medians [0.000287 0.000651]
baseline active complete worst lab median strain/rate [0.01282  0.017894]
baseline inactive complete worst lab median strain/rate [0.013045 0.015308]
baseline missing worst lab median strain/rate [0.169996 0.153226]
coupon interactions active complete worst lab median strain/rate [0.012743 0.01781 ]
coupon interactions inactive complete worst lab median strain/rate [0.012579 0.015931]
coupon interactions missing worst lab median strain/rate [0.169761 0.153698]
rational damage active complete worst lab median strain/rate [0.012762 0.017626]
rational damage inactive complete worst lab median strain/rate [0.013045 0.015308]
rational damage missing worst lab median strain/rate [0.170032 0.153339]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3b67a1ab2d0b7bab (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_3b67a1ab (directory).
```

### Turn 12 — `submit_completed_task` (`16145 ms` LLM)
- **Rationale**: The saved solver passed isolated verification. The held-lab audit found no consistent benefit from extra terms and no material signed drift offset on complete coupons, so the solver retains the specified cell-balanced imputation.
- **Alternatives Weighed**: `Add coupon interactions or rational damage corrections` (rejected: Neither improved worst-lab strain and rate errors together on the missing-coupon slice.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/alpha_m97_c01_batch_01_solver.py`. The isolated schema, calibration, and parameter-recovery checks passed. Public mean absolute strain and initial-rate errors were 0.00332 and 0.00325 on complete coupons, versus 0.16584 and 0.13573 on the four coupons with missing features.`

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
