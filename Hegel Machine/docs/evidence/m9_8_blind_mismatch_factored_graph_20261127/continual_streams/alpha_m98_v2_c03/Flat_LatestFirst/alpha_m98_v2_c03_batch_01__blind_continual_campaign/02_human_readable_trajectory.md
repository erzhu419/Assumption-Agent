# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m98_v2_c03_batch_01__blind_continual_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M98-Blind-Mismatch-Factored-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Blind Continual Campaign` |
| **Wave / Wormhole Family** | `wave_1` / `fam_hereditary_memory_and_factored_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `18 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `659,706` (`637,291` / `200,448` / `22,415`) |
| **Total Episode Wall Time (s)** | `717.17s` |

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

### Turn 1 — `execute_in_remote_docker` (`20374 ms` LLM)
- **Rationale**: Inspect the public schema and archived laws first. A changed apparatus signature would alter chamber-law selection, while changed bridge sweeps require refitting calibration regardless of the archive match.
- **Alternatives Weighed**: `Reuse the prior solver unchanged` (rejected: A blind campaign can retain the material law while switching chamber behavior or bridge calibration.)

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; cat /tmp/hegel_continual_archive/archive_manifest.json /tmp/hegel_continual_archive/archive_apparatus_log.json; sed -n '1,260p' /tmp/hegel_continual_archive/alpha_m98_v2_c03_s00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation robust IRLS reference-panel alignment'; python3 - <<'PY'
import pickle
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
def show(d,depth=0):
 for k,v in d.items():
  print(' '*depth,k,type(v).__name__,getattr(v,'shape',None), (len(v) if isinstance(v,(list,tuple,dict)) else ''),v if v is None else '')
  if isinstance(v,dict) and depth<3: show(v,depth+1)
print('items:',len(p)); show(p[0])
PY
```
- **Container Exit Code**: `0` (`388 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
items: 1
 dataset dict None 5 
  past_dt ndarray (6,)  
  eval_times ndarray (12,)  
  calibration_load_cells dict None 3 
   bridge_raw ndarray (80, 4)  
   true_channel_loads ndarray (80, 4)  
   single_channel_sweeps dict None 3 
    lab_id ndarray (24,)  
    cmd_stress ndarray (24,)  
    true_stress ndarray (24,)  
  train dict None 10 
   lab_id ndarray (72,)  
   stratum ndarray (72,)  
   coupon_features_raw ndarray (72, 4)  
   ref_coupons ndarray (72, 8)  
   past_channel_raw ndarray (72, 6, 4)  
   hold_channel_raw ndarray (72, 4)  
   strain_0 ndarray (72,)  
   strain_curves_obs ndarray (72, 12)  
   instant_rate_obs ndarray (72,)  
   aux_chamber_telemetry ndarray (72, 4)  
  test dict None 8 
   lab_id ndarray (36,)  
   stratum ndarray (36,)  
   coupon_features_raw ndarray (36, 4)  
   ref_coupons ndarray (36, 8)  
   past_channel_raw ndarray (36, 6, 4)  
   hold_channel_raw ndarray (36, 4)  
   strain_0 ndarray (36,)  
   aux_chamber_telemetry ndarray (36, 4)  
 campaign_diagnostics dict None 9 
  lab_id ndarray (18,)  
  stratum ndarray (18,)  
  coupon_features_raw ndarray (18, 4)  
  ref_coupons ndarray (18, 8)  
  past_channel_raw ndarray (18, 6, 4)  
  hold_channel_raw ndarray (18, 4)  
  strain_0 ndarray (18,)  
  strain_curves_obs ndarray (18, 12)  
  instant_rate_obs ndarray (18,)
```

### Turn 2 — `execute_in_remote_docker` (`20295 ms` LLM)
- **Rationale**: Read the archived implementation without truncation to recover its reference-panel and chamber formulas. Check lab coverage because a diagnostic lab absent from the train/test reference panels would require a separate standardization estimate during validation.
- **Alternatives Weighed**: `Select the archived chamber law from telemetry alone` (rejected: The current public batch contains measured chamber curves and rates, which can directly test the archived law.)

```bash
cat /tmp/hegel_continual_archive/alpha_m98_v2_c03_s00_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']
for r in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print({k:v for k,v in r.items() if k!='facility_telemetry_trace'}); print('trace',np.array(r['facility_telemetry_trace']))
for name,s in [('train',d['train']),('test',d['test']),('diag',p['campaign_diagnostics'])]:
 print(name,'labs',np.unique(s['lab_id'],return_counts=True),'strata',np.unique(s['stratum'],return_counts=True),'refs',s['ref_coupons'][:4]);
 if 'aux_chamber_telemetry' in s: print('telemetry',np.quantile(s['aux_chamber_telemetry'],[0,.5,1],axis=0))
print('dt',d['past_dt'],'times',d['eval_times'])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic' | head -16
```
- **Container Exit Code**: `0` (`435 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-0.65303733]
 [ 2.14784702 -2.43216958  0.98373446  2.10859825 -1.42530322  2.08300792
  -2.04470336 -1.42868314]
 [-0.8346009   0.20693647 -0.96494883  1.14616193  0.08818466  0.31267706
   0.15479391  1.69918647]]
telemetry [[-1.87077072 -2.33715908 -2.71360408 -1.79003928]
 [ 0.16339109 -0.10463832 -0.11283681  0.13610162]
 [ 2.86934228  1.72499155  2.07306701  2.3723951 ]]
test labs (array([0, 1, 2, 3]), array([ 8,  8,  9, 11])) strata (array([0, 1, 2]), array([16, 10, 10])) refs [[ 1.04931461  0.51475627 -0.55572073 -0.6352028   1.39928238  1.00507729
   1.84966577 -0.22689756]
 [-0.02819212  0.20269438  0.57794195  0.55291913  0.36991015 -0.17674049
   0.43020523  0.69392621]
 [-1.94338697 -1.76961783 -1.76470474 -0.15080516  1.61334474  0.31949566
  -2.12938841 -0.08652535]
 [-0.81475175 -0.15421315  0.32835673 -0.27414357 -0.60705148  0.3486297
  -0.67185026  1.2571077 ]]
telemetry [[-1.85993386 -1.63952896 -1.88520269 -2.00100192]
 [ 0.2206361   0.23017739 -0.35875491  0.12516123]
 [ 1.44813682  2.17371331  1.7358121   2.25113542]]
diag labs (array([0, 1, 2, 3]), array([7, 4, 5, 2])) strata (array([0, 1, 2]), array([ 2,  6, 10])) refs [[ 0.94617826 -0.2500637  -1.60804669  0.1809598   0.57894801 -0.72756008
  -0.75775754 -0.29218669]
 [ 0.27918349 -1.35530444 -1.54967932 -0.42086854 -0.98425842 -2.25627479
  -1.8445672  -1.46153732]
 [-0.22293364  2.16201258  0.68602358 -0.6970266   0.18231814 -1.84347905
  -0.48851841 -0.40124975]
 [ 1.52885717  0.94259553  0.47542085 -1.0419334   0.30288376  1.20805606
   0.21491524  1.3342075 ]]
dt [0.35 0.35 0.4  0.45 0.5  0.55] times [0.25       0.59090909 0.93181818 1.27272727 1.61363636 1.95454545
 2.29545455 2.63636364 2.97727273 3.31818182 3.65909091 4.        ]
autogenous_dispersive_wave_tomography@fbf620db | localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
```

### Turn 3 — `execute_in_remote_docker` (`17215 ms` LLM)
- **Rationale**: Recover the untruncated archived fit and prediction functions. If the archived predictor is already the baseline law, compare measured chamber curves and rates against that one model rather than treating them as distinct candidates.
- **Alternatives Weighed**: `Rewrite the material estimator before testing the chamber law` (rejected: The archived preprocessing matches the stated contract; reusing it initially isolates the chamber mismatch from changes to material estimation.)

```bash
sed -n '55,$p' /tmp/hegel_continual_archive/alpha_m98_v2_c03_s00_solver.py; python3 - <<'PY'
import json,numpy as np
r=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('apparatus:',[(x['completed_batch_index'],np.round(np.mean(x['facility_telemetry_trace'],axis=0),3).tolist()) for x in r]); print('asset index keys:',list(json.load(open('/holdout_task/public_asset_index.json'))))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_dispersive_wave_tomography.localize_pitch_catch | head -8
```
- **Container Exit Code**: `0` (`395 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0001, 24.999999])
    bounds = (np.r_[lo, lo, np.full(7, -np.inf)],
              np.r_[hi, hi, np.full(7, np.inf)])
    def linear_seed(taus):
        a = _design(coupon, dt, times, taus).reshape(-1, 7)
        coef = np.linalg.lstsq(a, y.ravel(), rcond=None)[0]
        return np.sum((a @ coef - y.ravel())**2), np.r_[np.log(taus), coef]
    seeds = sorted((linear_seed(np.array(pair)) for pair in combinations(
        (0.075, 0.15, 0.3, 0.55, 0.9, 1.6, 3., 6., 12., 22.), 2)), key=lambda p: p[0])
    def residual(p):
        return _design(coupon, dt, times, np.exp(p[:2])) @ p[2:] - y
    def optimize(seed, scale, kind):
        return least_squares(lambda p: (residual(p) / scale).ravel(), seed,
                             method='trf', bounds=bounds, loss=kind, x_scale='jac',
                             max_nfev=130, ftol=1e-9)
    seed = seeds[0][1]
    def scales(p):
        r = residual(p)
        curve, rate = r[:, :-1].ravel(), r[:, -1]
        return np.maximum([1.4826 * np.median(abs(v - np.median(v)))
                           for v in (curve, rate)], 1e-8)
    def weights(s):
        return np.r_[np.repeat(s[0], k - 1), s[1]]
    if loss == 'linear':
        p = optimize(seed, np.ones(k), 'linear').x
    else:
        p = optimize(seed, 1.5 * weights(scales(seed)), 'soft_l1').x
        p = optimize(p, tuning * weights(scales(p)), loss).x
    order = np.argsort(p[:2])
    return np.exp(p[:2][order]), np.r_[p[2:4][order], p[4:]]


def solve_batch(dataset: dict) -> dict:
    """Predict deployment strain and its initial derivative from benchtop data."""
    train, test = _prepare(dataset)
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    tr = dataset['train']
    y = np.column_stack((tr['strain_curves_obs'] - tr['strain_0'][:, None],
                         tr['instant_rate_obs']))
    taus, theta = _fit(train, dt, times, y)
    pred = _design(test, dt, times, taus) @ theta
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': dataset['test']['strain_0'][:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
apparatus: [(0, [2.416, 1.128, -0.633, -0.047, -0.389, -0.692, 0.015, 1.134])]
asset index keys: ['files', 'schema']
{
  "ok": true,
  "target": "autogenous_dispersive_wave_tomography.localize_pitch_catch",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_dispersive_wave_tomography",
```

### Turn 4 — `execute_in_remote_docker` (`33582 ms` LLM)
- **Rationale**: Measure curve and initial-rate errors separately. If the baseline fails despite a good benchtop fit, compare exact history reorderings and branch or drift gains before using the free coefficients to identify a missing term.
- **Alternatives Weighed**: `Use a flexible chamber regression directly` (rejected: The free history coefficients serve as a diagnostic; with only 18 labeled coupons, a deployment model needs a simpler law and held-out validation.)

```bash
cat > /tmp/m98_probe.py <<'PY'
import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/alpha_m98_v2_c03_s00_solver.py'); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr=d['train']; te=d['test']; ch=p['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']
train,test=old._prepare(d)
def obs(s): return np.column_stack([s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']])
y=obs(tr); tau,b=old._fit(train,dt,t,y); print('train tau,theta',tau,b)
def metrics(r):
 return np.round([[np.median(abs(a)),np.sqrt(np.mean(a*a)),np.quantile(abs(a),.9)] for a in [r[:,:-1],r[:,-1]]],7).tolist()
print('train fit',metrics(old._design(train,dt,t,tau)@b-y))
if ch is None: raise SystemExit
# Keep the public train/test panel in the feature estimate for the chamber probe.
merged={k:np.concatenate([te[k],ch[k]]) for k in ['lab_id','stratum','coupon_features_raw','ref_coupons']}
x=old._features(tr,merged)[-len(ch['lab_id']):]; offset,total=old._calibration(d['calibration_load_cells']); past=(ch['past_channel_raw']-offset)@total; hold=(ch['hold_channel_raw']-offset)@total; drift=np.c_[np.ones(len(x)),x]; cu=(past,hold,drift); Y=obs(ch); A=old._design(cu,dt,t,tau)
print('baseline chamber',metrics(A@b-Y),'signed residual mean',np.round(np.mean(Y-A@b,axis=0),6)); print('chamber signal',metrics(Y))
S=np.r_[np.repeat(.008,len(t)),.02]
def assess(name,design,base=None):
 if base is None: base=np.zeros_like(Y)
 z=(design/S[None,:,None]).reshape(-1,design.shape[-1]); yy=((Y-base)/S).ravel(); coef=np.linalg.lstsq(z,yy,rcond=None)[0]; coef=least_squares(lambda q:z@q-yy,coef,loss='soft_l1').x
 print(name, 'coefs',np.round(coef,6), 'err',metrics(design@coef+base-Y))
assess('free theta',A)
assess('branch and drift gains',np.dstack([A[:,:,0]*b[0],A[:,:,1]*b[1],A[:,:,2:]@b[2:]]))
for name,hist,delta in [('rev stress',past[:,::-1],dt),('rev pairs',past[:,::-1],dt[::-1]),('zero',np.zeros_like(past),dt),('last',np.repeat(past[:,-1:],past.shape[1],axis=1),dt),('mean',np.repeat(past.mean(axis=1,keepdims=True),past.shape[1],axis=1),dt)]:
 Q=old._design((hist,hold,drift),delta,t,tau); print(name,'fixed',metrics(Q@b-Y)); assess(name+' free theta',Q)
# Separate hold and individual history contributions for both branches.
gram=np.column_stack((hold,past)); kernels=np.vstack([-np.expm1(-t[:,None]/tau),1/tau]); B=(gram[:,None,:,None]*kernels[None,:,None,:]).reshape(len(hold),len(t)+1,-1)
assess('free history (hold,step 0..5; fast,slow)',np.dstack([B,A[:,:,2:]]))
print('baseline linear coefficients in that layout:'); q=np.exp(-dt[:,None]/tau); print(np.vstack([b[:2],-(1-q)*np.vstack([np.cumprod(q[:0:-1],axis=0)[::-1],np.ones(2)])*b[:2]]));print('drift',b[2:]); print('cal affine raw RMSE',np.sqrt(np.mean((((d['calibration_load_cells']['bridge_raw']-offset)@total)-d['calibration_load_cells']['true_channel_loads'].sum(1))**2)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/m98_probe.py
```
- **Container Exit Code**: `0` (`563 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
90683, 1.4811751, 2.1848886]] signed residual mean [ 0.239669  0.404926  0.464176  0.469573  0.442876  0.396009  0.340622
  0.276534  0.205648  0.12842   0.048227 -0.036505  1.258463]
chamber signal [[3.5928025, 3.5410253, 4.5850582], [5.9602279, 5.8615897, 6.7687734]]
free theta coefs [ 0.924489  0.319502  0.29645   0.221089 -0.120726  0.055017 -0.049933] err [[0.2068652, 0.3400764, 0.5564371], [0.6050786, 1.003736, 1.5306175]]
branch and drift gains coefs [1.114621 1.444631 0.622436] err [[0.2680712, 0.5101715, 0.7690707], [0.8472204, 1.2037084, 1.6880985]]
rev stress fixed [[0.4012693, 0.5394402, 0.8602753], [1.3503001, 1.6077718, 2.183873]]
rev stress free theta coefs [ 0.94906   0.437479  0.310891  0.150367 -0.117573  0.074873 -0.115595] err [[0.2572226, 0.400553, 0.5931634], [0.5846787, 0.9364168, 1.4420841]]
rev pairs fixed [[0.4294691, 0.5579835, 0.9135744], [1.5338815, 1.6851905, 2.2513623]]
rev pairs free theta coefs [ 1.001879  0.253463  0.343667  0.17715  -0.15209   0.099261 -0.104525] err [[0.2053968, 0.3996464, 0.6643794], [0.5840367, 0.9777179, 1.6092495]]
zero fixed [[0.2166621, 0.4837969, 0.8056337], [0.0496736, 0.1383705, 0.1660653]]
zero free theta coefs [ 0.726804  0.568864  0.108124  0.115518 -0.115479  0.097467 -0.093788] err [[0.0402351, 0.1045501, 0.1572093], [0.1553577, 0.2120175, 0.3001855]]
last fixed [[0.3807998, 0.6448076, 1.0784965], [0.87459, 1.4757838, 2.5424733]]
last free theta coefs [ 0.788325  0.099996  0.456093  0.235488 -0.135286  0.041052 -0.054384] err [[0.2379172, 0.463419, 0.8632899], [0.5537715, 1.2942944, 2.1878464]]
mean fixed [[0.4770958, 0.5596345, 0.8278381], [1.6527595, 1.6905623, 2.0842848]]
mean free theta coefs [ 1.076091  0.032457  0.406917  0.181588 -0.110702  0.124707 -0.075379] err [[0.2096647, 0.3110972, 0.4972033], [0.5122369, 0.9010676, 1.6327277]]
free history (hold,step 0..5; fast,slow) coefs [ 0.689896  0.636492 -0.030931  0.217184  0.040179 -0.05902   0.045848
 -0.089357  0.098645 -0.226637 -0.010504  0.07351  -0.091258  0.248577
  0.064564  0.112724 -0.123389  0.104078 -0.092532] err [[0.0260172, 0.0943734, 0.1537666], [0.0598568, 0.2139193, 0.4000651]]
baseline linear coefficients in that layout:
[[ 0.68403615  0.61526337]
 [-0.00316511 -0.03068933]
 [-0.0065982  -0.03388327]
 [-0.0166802  -0.04306265]
 [-0.04614229 -0.05464199]
 [-0.14013933 -0.06945557]
 [-0.46839297 -0.08864875]]
drift [ 0.27314496  0.17179557 -0.19091267  0.13527499 -0.14959927]
cal affine raw RMSE 0.00020183852071958445
```

### Turn 5 — `execute_in_remote_docker` (`26193 ms` LLM)
- **Rationale**: Coupon-level residuals can distinguish a wrong initial state from a missing post-hold drift term. In particular, a curve residual whose extrapolated derivative disagrees with the observed rate would rule out simply refitting the existing two branches and linear drift.
- **Alternatives Weighed**: `Fit all history coefficients freely` (rejected: The in-sample curve fit improves, but the initial-rate RMSE is worse than the reset alone and 18 coupons cannot reliably identify the unrestricted design.)
- **ATMS NOGOOD DECLARED**: Refuted `Deployment uses the same chronological hereditary state and drift law as benchtop training.` | Witness: `The baseline chamber median absolute errors are 0.4266 strain and 1.3391 initial rate, compared with 0.0079 and 0.0064 on train. Forward and reverse histories both fail; zeroing the state lowers chamber rate median error to 0.0497.` | Blanket Diagnosis: `The mismatch is downstream of the bridge calibration and is specific to the chamber; gain changes and simple history reorderings do not account for it.` | Sublated Pivot: `Keep the benchtop material fit, start from a chamber reset, and inspect the remaining residual for a reproducible load, feature, or time-dependent mechanism.`

```bash
cat > /tmp/m98_structure.py <<'PY'
exec(open('/tmp/m98_probe.py').read().split("print('baseline chamber'")[0])
Z=old._design((np.zeros_like(past),hold,drift),dt,t,tau)@b
R=Y-Z
np.set_printoptions(precision=3,suppress=True,linewidth=150)
print('time',t); print('past, hold, standardized features, reset residual per coupon:')
for i in range(len(hold)): print(i,'lab/str',ch['lab_id'][i],ch['stratum'][i],past[i],hold[i].round(3),x[i],R[i])
K=np.column_stack([-np.expm1(-t[:,None]/tau),t]); print('SVD reset residual',np.linalg.svd(R[:,:-1],compute_uv=False)); print('SVD right:',np.linalg.svd(R[:,:-1],full_matrices=False)[2][:3]); print('per coupon curve residual fit to intrinsic kernels and drift, actual rate residual and implied rate:')
c=np.linalg.lstsq(K,R[:,:-1].T,rcond=None)[0].T
print(np.c_[c,R[:,-1],c@np.r_[1/tau,1],np.sqrt(np.mean((c@K.T-R[:,:-1])**2,axis=1))]); print('reset rates residual vs candidate features'); q=np.column_stack((hold,past,drift,hold-past[:,-1],past[:,-1]-past[:,-2])); print(np.corrcoef(np.c_[R[:,-1],R[:,0],R[:,-2],q].T)[:3,3:]); print('cal sweeps',d['calibration_load_cells']['single_channel_sweeps'])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/m98_structure.py
python3 - <<'PY'
import json
j=json.load(open('/holdout_task/public_asset_index.json')); print('files:',list(j['files'])); print('schema:',str(j['schema'])[:1500]); j=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('archive metadata:',[{k:v for k,v in r.items() if k!='facility_telemetry_trace'} for r in j])
PY
```
- **Container Exit Code**: `0` (`548 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
3 0.469 1.032 1.867 1.002 1.507] 3.68 [0.54  0.185 0.167 0.507] [-0.011 -0.031 -0.075 -0.122 -0.179 -0.244 -0.307 -0.369 -0.441 -0.511 -0.59  -0.666 -0.001]
SVD reset residual [7.09  0.536 0.024 0.023 0.022 0.016 0.014 0.011 0.01  0.007 0.006 0.004]
SVD right: [[-0.003  0.012  0.041  0.079  0.125  0.176  0.23   0.287  0.348  0.411  0.476  0.542]
 [ 0.19   0.329  0.396  0.422  0.388  0.34   0.272  0.166  0.062 -0.06  -0.189 -0.33 ]
 [-0.245 -0.619 -0.218  0.215  0.302 -0.203  0.146  0.203  0.041  0.29   0.006 -0.426]]
per coupon curve residual fit to intrinsic kernels and drift, actual rate residual and implied rate:
[[ 0.167  1.707 -0.718  0.106  0.116  0.003]
 [-0.006  0.099 -0.073 -0.053 -0.057  0.004]
 [ 0.227  2.202 -0.855  0.214  0.245  0.002]
 [ 0.238  2.507 -0.718  0.473  0.49   0.004]
 [ 0.003  0.072 -0.072 -0.047 -0.045  0.003]
 [ 0.044  0.931 -0.365  0.017 -0.009  0.003]
 [ 0.158  1.526 -0.665  0.081  0.099  0.004]
 [ 0.082  0.921 -0.427  0.013  0.006  0.002]
 [ 0.026  0.581 -0.25  -0.02  -0.03   0.005]
 [ 0.115  0.934 -0.374  0.105  0.133  0.003]
 [-0.042 -0.625  0.126 -0.145 -0.14   0.003]
 [ 0.059  0.522 -0.265 -0.002  0.006  0.003]
 [ 0.039  0.453 -0.227 -0.008 -0.017  0.003]
 [ 0.042  0.482 -0.231 -0.011 -0.007  0.003]
 [ 0.019  0.327 -0.161 -0.015 -0.029  0.002]
 [ 0.148  1.124 -0.541  0.057  0.088  0.005]
 [ 0.118  1.131 -0.426  0.125  0.142  0.003]
 [ 0.055  0.617 -0.285 -0.001  0.006  0.002]]
reset rates residual vs candidate features
[[-0.111 -0.021  0.49   0.073  0.181 -0.183  0.295    nan  0.553 -0.058  0.49  -0.114 -0.272  0.292]
 [ 0.003  0.034  0.474  0.035  0.115 -0.131  0.314    nan  0.511  0.002  0.431 -0.098 -0.219  0.266]
 [ 0.07   0.071 -0.345  0.06  -0.3    0.232 -0.059    nan -0.379  0.342 -0.256  0.456  0.082 -0.194]]
cal sweeps {'lab_id': array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]), 'cmd_stress': array([-1.81 , -0.886, -0.094,  0.698,  1.49 ,  2.282, -1.305, -0.738, -0.252,  0.234,  0.72 ,  1.206, -1.31 , -0.764, -0.296,  0.172,  0.64 ,
        1.108, -1.77 , -0.888, -0.132,  0.624,  1.38 ,  2.136]), 'true_stress': array([-1.5, -0.8, -0.2,  0.4,  1. ,  1.6, -1.5, -0.8, -0.2,  0.4,  1. ,  1.6, -1.5, -0.8, -0.2,  0.4,  1. ,  1.6, -1.5, -0.8, -0.2,  0.4,  1. ,
        1.6])}
files: ['/holdout_task/public_batch.pkl', '/holdout_task/validate_public_schema.py']
schema: hegel_public_asset_index_v1
archive metadata: [{'completed_batch_index': 0, 'completed_short_name': 'alpha_m98_v2_c03_s00'}]
```
- **Stderr (tail)**:
```text
/usr/local/lib/python3.12/dist-packages/numpy/lib/function_base.py:2897: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/function_base.py:2898: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
```

### Turn 6 — `execute_in_remote_docker` (`43057 ms` LLM)
- **Rationale**: The per-coupon linear correction is close to the negative of the predicted benchtop drift. Subtract the reset branch response and test whether a saturating drift law explains both the residual curves and their initial derivatives. If an amplitude mismatch remains, its association with coupon features or load history will guide the next comparison.
- **Alternatives Weighed**: `Fit unrelated corrections for each observation time` (rejected: This would not explain the measured initial derivative or establish which correction transfers to new coupons.)

```bash
cat > /tmp/m98_drift.py <<'PY'
exec(open('/tmp/m98_probe.py').read().split("print('baseline chamber'")[0])
np.set_printoptions(precision=5,suppress=True,linewidth=145)
def kern(v): return np.vstack([-np.expm1(-t[:,None]/v),1/v])
K=kern(tau); r=drift@b[2:]; D=Y-(hold[:,None]*b[:2])@K.T
C=D@np.linalg.pinv(K.T)
print('r, measured rate minus reset branches, fitted two branch amplitudes, amp/r:',np.c_[r,D[:,-1],C,C/r[:,None]])
print('intrinsic kernel only',metrics(C@K.T-D), 'SVD',np.linalg.svd(D,compute_uv=False)[:6]); print('end ratios',np.median(D[r>.1]/r[r>.1,None],axis=0))
scale=np.r_[np.repeat(.004,len(t)),.008]; delta=np.r_[t,1.]
def reg(name,B):
 coef=np.linalg.lstsq((B/scale[None,:,None]).reshape(-1,B.shape[-1]),(D/scale).ravel(),rcond=None)[0]; print(name, np.round(coef,5),metrics(B@coef-D))
reg('intrinsic kernels X drift',r[:,None,None]*K[None,:,:]); reg('intrinsic kernels X feature columns',(drift[:,None,:,None]*K[None,:,None,:]).reshape(len(r),len(t)+1,-1))
reg('intrinsic kernels X r and r^2', (np.c_[r,r*r][:,None,:,None]*K[None,:,None,:]).reshape(len(r),len(t)+1,-1))
reg('intrinsic kernels X r and signed square', (np.c_[r,r*abs(r)][:,None,:,None]*K[None,:,None,:]).reshape(len(r),len(t)+1,-1))
for kind in ('exp','log','rational','sqrt'):
 def F(s):
  a=np.exp(s); u=t/a
  vals={'exp':-np.expm1(-u),'log':np.log1p(u),'rational':u/(1+u),'sqrt':2*(np.sqrt(1+u)-1)}[kind]
  return np.r_[a*vals,1.]
 for free in (False,True):
  opt=least_squares(lambda p: ((r[:,None]*F(p[0])*(p[1] if free else 1.)-D)/scale).ravel(),[0.,1.] if free else [0.],bounds=([-5.,-10.],[5.,10.]) if free else ([-5.],[5.]))
  print(kind,free,np.exp(opt.x[0]),opt.x[1:],metrics(r[:,None]*F(opt.x[0])*(opt.x[1] if free else 1.)-D))
print('residual amplitude after r regression, correlations against stress/features:')
err=C-r[:,None]*np.linalg.lstsq(r[:,None],C,rcond=None)[0]; q=np.c_[hold,past,x,r*r,r*abs(r),r**3,hold*r]; print(np.round(np.corrcoef(np.c_[err,q].T)[:2,2:],3))
print('raw calibrated channels hold sample',((ch['hold_channel_raw']-offset)@np.linalg.inv(np.linalg.lstsq(np.c_[np.ones(len(d['calibration_load_cells']['bridge_raw'])),d['calibration_load_cells']['true_channel_loads']],d['calibration_load_cells']['bridge_raw'],rcond=None)[0][1:]))[:4])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/m98_drift.py
```
- **Container Exit Code**: `0` (`524 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.04131  0.47139  0.1788   2.04041]
 [ 0.2249   0.21406  0.04302  0.44548  0.19128  1.98083]
 [ 0.15272  0.13739  0.02866  0.26684  0.18763  1.74725]
 [ 0.56815  0.62486  0.12128  1.31644  0.21346  2.31705]
 [ 0.42802  0.55351  0.10925  1.15955  0.25523  2.70909]
 [ 0.27726  0.27618  0.05529  0.57795  0.1994   2.08454]]
intrinsic kernel only [[0.0026109, 0.0038997, 0.0061664], [0.0021596, 0.0032048, 0.0047168]] SVD [9.49325 0.02628 0.02384 0.02349 0.01793 0.01389]
end ratios [0.22691 0.478   0.67214 0.86371 1.01106 1.12827 1.24144 1.35207 1.45088 1.52295 1.60829 1.65481 1.05089]
intrinsic kernels X drift [0.24552 2.55136] [[0.0673515, 0.1111337, 0.1172803], [0.06522, 0.0969037, 0.0975812]]
intrinsic kernels X feature columns [ 0.05481  0.57961  0.05072  0.52272 -0.04926 -0.49754  0.04405  0.45579 -0.03849 -0.39812] [[0.0312888, 0.0768447, 0.1413698], [0.0301337, 0.066316, 0.1193619]]
intrinsic kernels X r and r^2 [0.1915  2.02325 0.08711 0.85151] [[0.0441258, 0.1030808, 0.1333199], [0.0525009, 0.0900114, 0.1111999]]
intrinsic kernels X r and signed square [0.18715 1.98288 0.09399 0.91546] [[0.0420653, 0.1019919, 0.134284], [0.0510693, 0.0890507, 0.1100355]]
exp False 2.409881174896224 [] [[0.0707754, 0.1115756, 0.1213677], [0.0496736, 0.1383705, 0.1660653]]
exp True 2.192590267620388 [1.05011] [[0.0750954, 0.1116507, 0.1204684], [0.0394498, 0.1235353, 0.1492794]]
log False 1.6375763514756274 [] [[0.061468, 0.1128252, 0.1316419], [0.0496736, 0.1383705, 0.1660653]]
log True 1.0114077156130135 [1.22061] [[0.068534, 0.1111287, 0.116676], [0.0617214, 0.0967527, 0.0932211]]
rational False 3.9671724407626514 [] [[0.0631601, 0.1118862, 0.1235614], [0.0496736, 0.1383705, 0.1660653]]
rational True 3.017077502026455 [1.13448] [[0.0729499, 0.1112824, 0.1179595], [0.0428173, 0.1045325, 0.1082987]]
sqrt False 0.5589746394140129 [] [[0.0593897, 0.1154636, 0.1507198], [0.0496736, 0.1383705, 0.1660653]]
sqrt True 0.2331521459489817 [1.33828] [[0.059192, 0.1121009, 0.1322599], [0.0943741, 0.1090372, 0.1390699]]
residual amplitude after r regression, correlations against stress/features:
[[-0.107 -0.005  0.44   0.147  0.091 -0.095  0.32   0.482  0.06   0.478  0.085  0.495  0.499  0.473  0.428]
 [-0.084  0.013  0.417  0.094  0.063 -0.105  0.335  0.484  0.107  0.454  0.071  0.475  0.478  0.456  0.407]]
raw calibrated channels hold sample [[1.00653 0.6361  0.89951 0.47753]
 [1.02578 1.00307 1.22392 0.43448]
 [1.26264 0.43917 1.23627 0.91063]
 [0.79867 0.35666 0.69376 0.79024]]
```

### Turn 7 — `execute_in_remote_docker` (`42611 ms` LLM)
- **Rationale**: The chamber residual after subtracting the reset hold response is nearly rank one, and the two intrinsic kernels also reproduce its measured initial derivative. Extract the shared kernel and compare reference-panel conventions and sparse algebraic amplitude terms to determine which correction is worth validating.
- **Alternatives Weighed**: `Predict the lowest-error coupon amplitude from a free regression` (rejected: The 18 chamber coupons can support a compact law, not an unrestricted expansion; this search diagnoses missing terms before independent validation.)

```bash
cat > /tmp/m98_amplitude.py <<'PY'
exec(open('/tmp/m98_probe.py').read().split("print('baseline chamber'")[0])
from itertools import combinations_with_replacement,combinations
np.set_printoptions(precision=5,suppress=True,linewidth=150)
K=np.vstack([-np.expm1(-t[:,None]/tau),1/tau]); D=Y-(hold[:,None]*b[:2])@K.T
C=D@np.linalg.pinv(K.T); _,_,vh=np.linalg.svd(C,full_matrices=False); alpha=vh[0]/(vh[0]@(1/tau)); F=K@alpha; target=D@F/(F@F)
print('rank-one kernel coefficients and rate fractions',alpha,alpha/tau,'floor',metrics(target[:,None]*F-D))
r=drift@b[2:]; print('x, missing mask, r, inferred rate, delta',np.c_[x,np.isnan(ch['coupon_features_raw']).astype(int),r,target,target-r]); print('archived feature procedure:')

def score(A,y):
 U,s,V=np.linalg.svd(A,full_matrices=False); use=s>s[0]*1.e-9; U=U[:,use]; beta=V[use].T@((U.T@y)/s[use]); res=A@beta-y; loo=res/np.maximum(1-(U*U).sum(1),1.e-6); return np.sqrt(np.mean(loo**2)),np.max(abs(loo)),np.sqrt(np.mean(res**2)),beta

def report(name,A,y):
 a,m,e,beta=score(A,y); print(name,'LOO/worst/fit',np.round([a,m,e],6),'coef',np.round(beta,5))
for name, xx in [('joint',x),('standalone',old._features(ch,ch)[:len(r)]),('train+diag',old._features(tr,ch)[-len(r):])]:
 dd=np.c_[np.ones(len(r)),xx]; rr=dd@b[2:]; report(name+' plain',rr[:,None],target); report(name+' affine features',dd,target); report(name+' r powers',np.c_[rr,rr*abs(rr),rr**3],target)
 print('feature shifts from joint',np.max(abs(xx-x),axis=0))
q=[]; names=[]
def add(name,v):
 if np.std(v)>1.e-7: names.append(name); q.append(v)
for j,k in combinations_with_replacement(range(4),2): add(f'x{j}*x{k}',x[:,j]*x[:,k])
for j in range(4):
 add(f'abs(x{j})',abs(x[:,j])); add(f'x{j}^3',x[:,j]**3); add(f'x{j}*abs(x{j})',x[:,j]*abs(x[:,j])); add(f'missing x{j}',np.isnan(ch['coupon_features_raw'][:,j]).astype(float))
for n,v in [('r^2',r*r),('r*abs(r)',r*abs(r)),('r^3',r**3),('hold',hold),('r*hold',r*hold),('hold^2',hold**2),('mean past',past.mean(1)),('r*mean past',r*past.mean(1)),('range past',np.ptp(past,axis=1)),('last gap',hold-past[:,-1])]: add(n,v)
for j in range(past.shape[1]): add(f'past{j}',past[:,j]); add(f'past{j}*r',past[:,j]*r)
for j in range(3): add(f'stratum{j}',ch['stratum']==j)
for j in range(4): add(f'lab{j}',ch['lab_id']==j)
q=np.array(q).T
for label,base,yy in [('fixed drift',np.ones((len(r),1)),target-r),('drift gain',np.c_[np.ones(len(r)),r],target),('refit features',drift,target)]:
 for size in (1,2,3):
  scores=[]
  for cols in combinations(range(q.shape[1]),size):
   a,m,e,c=score(np.c_[base,q[:,cols]],yy); scores.append((a,m,e,cols,c))
  for a,m,e,cols,c in sorted(scores,key=lambda v:v[0])[:3]: print(label,[names[i] for i in cols],np.round([a,m,e],6),np.round(c,5))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/m98_amplitude.py
sed -n '12,55p' /tmp/hegel_continual_archive/alpha_m98_v2_c03_s00_solver.py
```
- **Container Exit Code**: `0` (`4435 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
26  ] [ 0.27669  0.1929  -0.25388  0.18004 -0.20096 -0.0186   0.01662]
refit features ['x0^3', 'past0*r'] [0.04402 0.09261 0.02456] [ 0.24103  0.17226 -0.22302  0.17109 -0.16615  0.01248  0.17307]
refit features ['x0^3', 'past2*r', 'past3*r'] [0.03105 0.0488  0.01756] [ 0.32522  0.21622 -0.26681  0.19894 -0.21762  0.01466 -0.05383 -0.08698]
refit features ['x0^3', 'hold', 'lab1'] [0.03533 0.06207 0.02108] [ 0.19863  0.18708 -0.24868  0.17515 -0.20346  0.01443  0.02826 -0.03625]
refit features ['x0^3', 'hold^2', 'past0*r'] [0.03563 0.08428 0.02143] [ 0.20495  0.1775  -0.22599  0.17297 -0.17067  0.01289  0.00367  0.14363]
    # Only the sum of the four corrected channels enters the constitutive law.
    return fit[0], np.linalg.solve(fit[1:], np.ones(loads.shape[1]))


def _features(train, test):
    lab = np.concatenate((train['lab_id'], test['lab_id']))
    stratum = np.concatenate((train['stratum'], test['stratum']))
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    for group in np.unique(lab):
        rows = lab == group
        controls = ref[rows]
        center = np.median(np.mean(controls, axis=0))
        scale = np.median(np.std(controls, axis=0, ddof=0))
        x[rows] = (x[rows] - center) / scale
    _, cell, count = np.unique(np.column_stack((lab, stratum)), axis=0,
                               return_inverse=True, return_counts=True)
    weights = 1.0 / count[cell].reshape(-1, 1)
    observed = ~np.isnan(x)
    filled = np.where(observed, x, 0.0)
    total = (weights * observed).sum(axis=0)
    means = np.divide((weights * filled).sum(axis=0), total,
                      out=np.zeros(x.shape[1]), where=total > 0)
    return np.where(observed, x, means)


def _prepare(dataset):
    tr, te = dataset['train'], dataset['test']
    x = _features(tr, te)
    offset, total = _calibration(dataset['calibration_load_cells'])
    n = len(tr['lab_id'])
    def coupon(data, features):
        return ((data['past_channel_raw'] - offset) @ total,
                (data['hold_channel_raw'] - offset) @ total,
                np.column_stack((np.ones(len(features)), features)))
    return coupon(tr, x[:n]), coupon(te, x[n:])


def _design(coupon, dt, times, taus):
    past, hold, drift = coupon
    z = np.zeros((len(hold), 2))
    for step, duration in enumerate(dt):
        w = -np.expm1(-duration / taus)
        z += w * (past[:, step, None] - z)
```

### Turn 8 — `execute_in_remote_docker` (`54693 ms` LLM)
- **Rationale**: Compare shared feature transforms, pairwise combinations, and alternate reference pools. If a compact law lowers both coupon- and lab-held-out errors, prefer it over an unconstrained amplitude fit.
- **Alternatives Weighed**: `Deploy the lowest-error subset from the first feature search` (rejected: The selected terms have not been validated across labs, and several unrelated load and feature terms gave similar coupon-LOO scores.)

```bash
cat > /tmp/m98_grammar.py <<'PY'
exec(open('/tmp/m98_amplitude.py').read().split("print('rank-one")[0])
from scipy.optimize import least_squares
lab=ch['lab_id']; raw=ch['coupon_features_raw']; r=drift@b[2:]; lin=r-b[2]; w=b[3:]; N=len(r)
print('refs ranges and missing',[(int(j),np.round(ch['ref_coupons'][lab==j].mean(),4),np.round(ch['ref_coupons'][lab==j].std(),4)) for j in np.unique(lab)],np.isnan(raw).sum(0)); print('joint-standalone',np.round(x-old._features(ch,ch)[:N],3)); print('fit y =',np.round(target,4))
def evaluate(A,y=target):
 c=np.linalg.lstsq(A,y,rcond=1e-9)[0]; e=A@c-y; es=np.empty(N); labs=np.empty(N)
 for i in range(N):
  k=np.arange(N)!=i; es[i]=A[i]@np.linalg.lstsq(A[k],y[k],rcond=1e-9)[0]-y[i]
 for g in np.unique(lab):
  k=lab!=g; labs[~k]=A[~k]@np.linalg.lstsq(A[k],y[k],rcond=1e-9)[0]-y[~k]
 return np.sqrt(np.mean(e*e)),np.sqrt(np.mean(es*es)),np.max(abs(es)),np.sqrt(np.mean(labs*labs)),c

def pr(name,A,y=target):
 a,l,m,ll,c=evaluate(A,y); print(name,np.round([a,l,m,ll],5),np.round(c,5))
for tag,xx in [('joint',x),('diag',old._features(ch,ch)[:N]),('train+diag',old._features(tr,ch)[-N:])]:
 rr=b[2]+xx@w; dd=np.c_[np.ones(N),xx]; print('\n',tag)
 for name,phi in [('linear',xx),('cubic',xx**3),('signed2',xx*abs(xx)),('tanh',np.tanh(xx)),('sinh',np.sinh(xx)),('symmetric cubic',xx**3+3*xx),('signed saturation',xx/(1+.25*abs(xx))),('x0 cubic',xx[:,0,None]**3)]:
  pr('1,r, transform weighted '+name,np.c_[np.ones(N),rr,phi@w] if phi.shape[1]==4 and name!='linear' else np.c_[np.ones(N),rr,phi] if name=='x0 cubic' else np.c_[np.ones(N),rr]);
  if name!='linear': pr('free linear + '+name,np.c_[dd,phi]);
 for name,phi in [('sum-cubes',np.sum(xx**3,1)),('prod-all',np.prod(xx,1)),('r cubed',rr**3),('lin cubed',(rr-b[2])**3),('lin norm squared',(rr-b[2])*np.sum(xx*xx,1)),('r norm squared',rr*np.sum(xx*xx,1))]: pr(name,np.c_[np.ones(N),rr,phi])
# Search one additive term at a time, including cube of sums/differences and powers of individual coordinates.
lib=[]
def add(name,z):
 if np.std(z)>1e-9: lib.append((name,z))
for i in range(4):
 for j in range(i+1,4):
  for s in (-2,-1,-.5,.5,1,2):
   z=x[:,i]+s*x[:,j]; nm=f'(x{i}{s:+}x{j})'
   for p in (2,3): add(f'{nm}^{p}',z**p)
   add(f'{nm}abs',z*abs(z)); add(f'{nm}rational',z/(1+z*z)); add(f'{nm}tanh',np.tanh(z))
for ids in combinations_with_replacement(range(4),3): add(str(ids),np.prod(x[:,ids],1))
for j in range(4):
 for power in [2,3,4,5]: add(f'x{j}^{power}',x[:,j]**power)
 for f,n in [(np.tanh,'tanh'),(np.sinh,'sinh'),(np.sin,'sin'),(np.abs,'abs')]: add(f'{n}x{j}',f(x[:,j]))
for base in [np.c_[np.ones(N),r],drift]:
 vals=[]
 for n,z in lib:
  a,l,m,ll,c=evaluate(np.c_[base,z]); vals.append((l,ll,a,m,n,c))
 for v in sorted(vals,key=lambda v:v[0])[:10]: print('single term',np.round(v[:4],5),v[4],np.round(v[5],5))
# Smooth shared coordinate transform with fitted mixing of linear and nonlinear terms.
for tag in ['sinh','rational','asinh','signed_power','cubic rational','weighted cube shifted']:
 def trans(xx,a):
  if tag=='sinh': return np.sinh(a*xx)/a
  if tag=='rational': return xx/(1+a*xx**2)
  if tag=='asinh': return np.arcsinh(a*xx)/a
  if tag=='signed_power': return np.sign(xx)*abs(xx)**(1+a)
  if tag=='cubic rational': return xx**3/(1+a*xx**2)
  return (xx+a)**3
 def f(p,xx): return p[0]+p[1]*(xx@w)+p[2]*(trans(xx,p[3])@w)
 lo=-3 if tag=='weighted cube shifted' else .001; hi=3
 def fit(ok): return least_squares(lambda p:f(p,x[ok])-target[ok],[.27,1,.1,.4],bounds=([-np.inf]*3+[lo],[np.inf]*3+[hi]),max_nfev=500).x
 es=np.array([f(fit(np.arange(N)!=i),x[i:i+1])[0]-target[i] for i in range(N)]); p=fit(np.ones(N,bool)); print('nonlinear',tag,np.round(p,5),np.sqrt(np.mean((f(p,x)-target)**2)),np.sqrt(np.mean(es**2)),max(abs(es)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/m98_grammar.py
```
- **Container Exit Code**: `0` (`2486 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
9387 0.05692 0.29061] (0, 1, 3) [-0.07144  1.3617   0.04875]
single term [0.09126 0.11194 0.05132 0.30725] (x1+2x2)^3 [-0.08142  1.31361  0.00361]
single term [0.09218 0.11249 0.05552 0.30411] (0, 1, 2) [-0.10067  1.38523  0.05165]
single term [0.09229 0.10993 0.05688 0.30597] (x1+2x2)^2 [-0.10795  1.30314  0.01456]
single term [0.09284 0.09712 0.0758  0.32893] sinx0 [-0.10585  1.4434  -0.04111]
single term [0.09308 0.09256 0.04403 0.33626] x0^2 [-0.11137  1.32268  0.03068]
single term [0.09315 0.09784 0.0781  0.33616] (x0-0.5x1)rational [-0.11151  1.46807 -0.0689 ]
single term [0.0939  0.09917 0.07757 0.33341] (x1-1x2)rational [-0.09911  1.38033 -0.06923]
single term [0.05355 0.06103 0.02858 0.13364] (x0+0.5x3)^3 [ 0.27706  0.18047 -0.24508  0.16653 -0.21435  0.00847]
single term [0.05382 0.12309 0.02887 0.12577] sinhx0 [ 0.2826   0.14478 -0.25292  0.17457 -0.20027  0.04864]
single term [0.07004 0.12137 0.02951 0.22193] (0, 0, 0) [ 0.28428  0.18439 -0.25093  0.17514 -0.20045  0.01418]
single term [0.07004 0.12137 0.02951 0.22193] x0^3 [ 0.28428  0.18439 -0.25093  0.17514 -0.20045  0.01418]
single term [0.07212 0.08463 0.03029 0.23674] (x0+0.5x3)abs [ 0.27622  0.12856 -0.23333  0.16406 -0.23745  0.0457 ]
single term [0.08226 0.11483 0.0287  0.29516] x0^5 [ 0.28063  0.20506 -0.25517  0.17432 -0.19996  0.00121]
single term [0.08486 0.1452  0.03068 0.29769] (x0+0.5x1)^3 [ 0.2813   0.18949 -0.26858  0.1806  -0.19791  0.01022]
single term [0.08534 0.12101 0.03155 0.29125] sinx0 [ 0.28688  0.32309 -0.24745  0.17682 -0.20065 -0.15331]
single term [0.08947 0.11952 0.03331 0.31645] (x0+1x1)^3 [ 0.2746   0.18645 -0.30038  0.18306 -0.19871  0.00827]
single term [0.09005 0.10173 0.03321 0.31071] (x0-0.5x1)^3 [ 0.28101  0.15857 -0.23123  0.16344 -0.20904  0.02344]
nonlinear sinh [0.28044 1.27514 0.00079 3.     ] 0.03279986105673386 0.07533498810433797 0.26879421404718595
nonlinear rational [  0.28632  60.8239  -59.686     0.001  ] 0.04513758795180267 0.08765020931035637 0.29037875985106165
nonlinear asinh [    0.28629  5592.85024 -5591.71154     0.00797] 0.04504218057889453 0.08753237297101407 0.29019699653398856
nonlinear signed_power [0.28444 1.19993 0.01827 3.     ] 0.04055428861592118 0.08265433142836492 0.28579643896049334
nonlinear cubic rational [0.28632 1.13789 0.05969 0.001  ] 0.04513758795097497 0.08764925260900337 0.29037356064117503
nonlinear weighted cube shifted [0.28683 1.17273 0.05067 0.14545] 0.043861906760064316 0.08934633066281844 0.2902154253687582
```

### Turn 9 — `execute_in_remote_docker` (`39739 ms` LLM)
- **Rationale**: Apply the deployment reference panel unchanged to diagnostics, matching how new coupons are handled. Then test compact two-term factorizations and inspect telemetry to decide whether the public chamber law can be transferred across campaigns.
- **Alternatives Weighed**: `Adopt the best single cubic term immediately` (rejected: Its error remains well above the chamber measurement floor, and the previous diagnostic transform let diagnostic coupons change the reference panel.)

```bash
cat > /tmp/m98_factor.py <<'PY'
exec(open('/tmp/m98_amplitude.py').read().split("print('rank-one")[0])
from itertools import combinations
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=135)
# Transform diagnostics with the deployment panel frozen, rather than allowing held-out diagnostics to change it.
labs=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); xp=old._features(tr,te); rawp=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); imp=np.array([xp[np.isnan(rawp[:,j]),j][0] if np.isnan(rawp[:,j]).any() else np.mean(xp[:,j]) for j in range(4)]); raw=ch['coupon_features_raw']; xx=np.empty_like(raw)
for g in np.unique(ch['lab_id']):
 s=ch['lab_id']==g; rr=refs[labs==g]; xx[s]=(raw[s]-np.median(rr.mean(0)))/np.median(rr.std(0))
xx=np.where(np.isnan(xx),imp,xx); print('frozen panel delta max',np.max(abs(xx-x),axis=0)); print('test feature quantiles',np.quantile(xp[len(tr['lab_id']):],[0,.5,1],axis=0))
for tag,s in [('train',tr),('test',te)]:
 a=s['aux_chamber_telemetry']; print(tag,'telemetry quantiles',np.quantile(a,[0,.1,.5,.9,1],axis=0),'first rows',a[:4]); print('by lab means',np.array([a[s['lab_id']==g].mean(0) for g in np.unique(s['lab_id'])]))
print('archive trace summary',np.array(__import__('json').load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))[-1]['facility_telemetry_trace'])[:3]); print('beta',b,'kernel alpha',alpha)
for tag,x in [('joint',x),('frozen',xx)]:
 N=len(x); r=b[2]+x@b[3:]; labels=ch['lab_id']; groups=[labels==g for g in np.unique(labels)]; base=np.c_[np.ones(N),x]; lib={}
 def add(k,v):
  if np.std(v)>1e-8: lib[k]=v
 z={f'x{j}':x[:,j] for j in range(4)}; z.update({'r':r,'lin':r-b[2]})
 for i,j in combinations(range(4),2):
  for a in [-2,-1,-.5,.5,1,2]: z[f'x{i}{a:+g}x{j}']=x[:,i]+a*x[:,j]
 for k,v in z.items():
  for p in [2,3,4]: add(f'{k}^{p}',v**p)
  add(f'tanh {k}',np.tanh(v)); add(f'sinh {k}',np.sinh(v)); add(f'abs {k}',abs(v)); add(f'r*{k}',r*v)
  for a in [.1,.3,1,3]:
   add(f'{k}/(1+{a}*{k}^2)',v/(1+a*v*v)); add(f'r/(1+{a}*{k}^2)',r/(1+a*v*v))
 for ids in combinations_with_replacement(range(4),3): add(f'product {ids}',np.prod(x[:,ids],axis=1))
 for j in range(4):
  add(f'x{j} missing',np.isnan(raw[:,j])); add(f'x{j}*r^2',x[:,j]*r*r)
 def stats(A,y):
  u,s,v=np.linalg.svd(A,full_matrices=False); ok=s>s[0]*1e-8; u=u[:,ok]; p=u@(u.T@y); e=p-y; loo=e/np.maximum(1-(u*u).sum(1),1e-6); return np.linalg.norm(loo)/N**.5,np.linalg.norm(e)/N**.5,np.max(abs(loo))
 for basetag,A,y in [('fixed',np.ones((N,1)),target-r),('free',base,target)]:
  win=[]; names=list(lib)
  for i,k in enumerate(names):
   for m in [None]+names[i+1:]:
    design=np.c_[A,lib[k]] if m is None else np.c_[A,lib[k],lib[m]]; win.append((*stats(design,y),k,m))
  for row in sorted(win)[:4]:
   l,f,m,k,j=row; a=np.c_[A,lib[k]] if j is None else np.c_[A,lib[k],lib[j]]; e=np.concatenate([a[g]@np.linalg.lstsq(a[~g],y[~g],rcond=1e-8)[0]-y[g] for g in groups]); print(tag,basetag,k,j,'LOO, fit, worst, lab',np.round([l,f,m,np.sqrt(np.mean(e*e))],5),'coef',np.round(np.linalg.lstsq(a,y,rcond=1e-8)[0],5))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/m98_factor.py
```
- **Container Exit Code**: `1` (`8878 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
train tau,theta [0.47644388 3.53512686] [ 0.68403615  0.61526337  0.27314496  0.17179557 -0.19091267  0.13527499
 -0.14959927]
train fit [[0.007916, 0.0453063, 0.0208024], [0.0063703, 0.008807, 0.013742]]
frozen panel delta max [0.07841 0.0789  0.0601  0.09072]
test feature quantiles [[-3.4085  -2.35298 -1.48068 -1.16452]
 [-0.27846 -0.20866 -0.03269  0.35637]
 [ 2.26471  1.03122  2.03844  3.99053]]
train telemetry quantiles [[-1.87077 -2.33716 -2.7136  -1.79004]
 [-1.09622 -1.44998 -1.29746 -1.0188 ]
 [ 0.16339 -0.10464 -0.11284  0.1361 ]
 [ 1.29746  0.88216  1.23874  1.39092]
 [ 2.86934  1.72499  2.07307  2.3724 ]] first rows [[-0.54573  0.02196 -1.54089 -0.54939]
 [-0.50286  1.02096 -0.75194 -0.85927]
 [-0.44598  0.4287   0.20885  0.26602]
 [ 0.36935 -1.45094 -0.46304 -1.25894]]
by lab means [[ 0.52138 -0.0863  -0.00336  0.28661]
 [ 0.18579 -0.2882   0.18709  0.10519]
 [-0.06703  0.02639 -0.22257 -0.11836]
 [-0.05712 -0.54687 -0.3983   0.21936]]
test telemetry quantiles [[-1.85993 -1.63953 -1.8852  -2.001  ]
 [-0.9952  -1.1351  -1.5201  -0.93752]
 [ 0.22064  0.23018 -0.35875  0.12516]
 [ 1.183    1.62003  0.69489  1.04559]
 [ 1.44814  2.17371  1.73581  2.25114]] first rows [[ 0.87697  1.52963  0.03225 -0.93368]
 [ 1.31368  0.94855  0.55083 -0.76543]
 [ 1.44814 -0.29771  0.24926  0.53776]
 [ 0.15295  1.43885  0.41257  1.26115]]
by lab means [[ 0.33238 -0.05676 -0.57923  0.11608]
 [ 0.33715  0.28567 -0.44101  0.36279]
 [ 0.08012  0.13368 -0.20208  0.31116]
 [-0.1857   0.53016 -0.10295 -0.2996 ]]
archive trace summary [[ 2.41191  1.1538  -0.72998 -0.03021 -0.22763 -0.65908 -0.06875  0.97529]
 [ 2.35995  1.2767  -0.45929  0.2234  -0.33592 -0.93423  0.28419  1.31677]
 [ 2.44428  0.93888 -0.74625  0.68549 -0.46652 -0.27535  0.83827  1.2074 ]]
beta [ 0.68404  0.61526  0.27314  0.1718  -0.19091  0.13527 -0.1496 ] kernel alpha [0.19591 2.08154]
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/m98_factor.py", line 35, in <module>
    for row in sorted(win)[:4]:
               ^^^^^^^^^^^
TypeError: '<' not supported between instances of 'str' and 'NoneType'
```

### Turn 10 — `execute_in_remote_docker` (`26934 ms` LLM)
- **Rationale**: Repair the interrupted two-term comparison and test whether affine lab alignment can explain the amplitude residual. If substantial errors persist within labs after affine alignment, prioritize a shared chamber nonlinearity over lab-specific corrections.
- **Alternatives Weighed**: `Treat lab identity as a deployment correction` (rejected: Use lab gains and offsets only to diagnose reference alignment; diagnostic lab fits do not establish transfer to private coupons.)

```bash
sed -i 's/sorted(win)\[:4\]/sorted(win,key=lambda v:v[:3])[:4]/' /tmp/m98_factor.py
OPENBLAS_NUM_THREADS=1 python3 /tmp/m98_factor.py | tail -16
cat >/tmp/m98_labprobe.py <<'PY'
exec(open('/tmp/m98_factor.py').read().split('for tag,x in')[0].replace("print(tag,'telemetry quantiles',np.quantile(a,[0,.1,.5,.9,1],axis=0),'first rows',a[:4]); print('by lab means',np.array([a[s['lab_id']==g].mean(0) for g in np.unique(s['lab_id'])]))",'pass'))
l=ch['lab_id']; H=np.eye(4)[l]; r=xx@b[3:]; N=len(r)
for name,A in [('lab gain and offset',np.c_[H,H*r[:,None]]),('lab offsets, common gain',np.c_[H,r]),('lab offsets, free features',np.c_[H,xx]),('lab offsets and gain, cubic features',np.c_[H,H*r[:,None],xx**3@b[3:]]),('quartic scalar drift',np.column_stack([r**i for i in range(5)]))]:
 c=np.linalg.lstsq(A,target,rcond=None)[0]; e=A@c-target; print(name, np.round(c,5),np.round(np.sqrt(np.mean(e**2)),5), np.round(e,4))
for i in np.argsort(r): print(int(i),l[i],np.round(np.r_[r[i]+b[2],target[i],xx[i]],4))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/m98_labprobe.py | tail -23
```
- **Container Exit Code**: `0` (`36222 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-0.5x2^2) LOO, fit, worst, lab [0.04196 0.03804 0.10666 0.04506] coef [-0.08579  0.00208  0.34575]
frozen fixed r/(1+0.3*x1^2) x0+1x1^4 LOO, fit, worst, lab [0.04212 0.03754 0.10976 0.03882] coef [-0.0943   0.38333  0.0015 ]
frozen fixed x0+0.5x1^4 r/(1+0.1*x1-0.5x3^2) LOO, fit, worst, lab [0.04317 0.03616 0.10668 0.04033] coef [-0.08415  0.00204  0.35094]
frozen free x0+0.5x1^3 product (1, 1, 3) LOO, fit, worst, lab [0.03394 0.02003 0.07818 0.04646] coef [ 0.2982   0.18509 -0.26309  0.17816 -0.23787  0.01086  0.06127]
frozen free product (0, 0, 0) product (1, 1, 3) LOO, fit, worst, lab [0.03507 0.0221  0.07269 0.07936] coef [ 0.29987  0.18044 -0.2446   0.17372 -0.23603  0.01495  0.05472]
frozen free x0^3 product (1, 1, 3) LOO, fit, worst, lab [0.03507 0.0221  0.07269 0.07936] coef [ 0.29987  0.18044 -0.2446   0.17372 -0.23603  0.01495  0.05472]
frozen free sinh x0+1x1 product (1, 1, 3) LOO, fit, worst, lab [0.03623 0.02091 0.08156 0.05757] coef [ 0.29119  0.17014 -0.30162  0.17775 -0.23656  0.02319  0.05733]
 -0.0734 -0.0153]
lab offsets and gain, cubic features [0.29654 0.27246 0.2618  0.27954 1.24047 1.10826 1.31784 1.8767  0.01772] 0.02283 [ 0.0314  0.0099 -0.0086  0.      0.0138  0.0204 -0.0037  0.0367 -0.0116  0.      0.0035 -0.0201  0.0141 -0.0178 -0.0005  0.0048
 -0.0718 -0.0006]
quartic scalar drift [ 0.28173  1.28866  0.38299  0.94396 -2.16864] 0.08453 [ 0.0922  0.0035  0.0192 -0.2828  0.0032  0.0097  0.1279  0.0598  0.0112 -0.0669 -0.0029  0.0062  0.0032  0.0045 -0.0063  0.0908
 -0.0819  0.0094]
10 2 [-0.111  -0.2546 -0.0675  0.4314 -0.7431  1.2675]
1 0 [ 0.065   0.0139  0.9199  1.6843 -0.6757 -0.3126]
4 0 [ 0.0718  0.0234 -1.502  -0.176   0.7528  0.5263]
14 1 [ 0.1487  0.1313 -1.0811 -0.323   0.0416  0.0404]
13 0 [ 0.2242  0.215   0.9161 -0.8296  0.1016  2.5299]
8 2 [ 0.2288  0.214  -0.1757 -0.4598  0.1095  0.7805]
12 0 [ 0.2302  0.2238 -1.6881 -0.5983  1.552   0.5154]
11 2 [ 0.2716  0.2735  1.318  -0.8368 -0.995   1.6923]
17 1 [0.2779 0.2784 0.5456 0.1771 0.158  0.5116]
5 0 [ 0.3327  0.3502 -0.1456 -0.1584  0.478   0.0692]
9 3 [ 0.3828  0.4955 -0.901  -0.3449  1.7217  0.2292]
16 0 [ 0.4159  0.5572 -0.7193 -1.5616 -0.0327  0.1832]
7 2 [ 0.4315  0.438   1.1204  0.031  -0.0282  0.1634]
15 1 [ 0.5798  0.6302  0.8656  1.0562  0.3363 -2.0995]
6 1 [ 0.6827  0.7497 -0.1838 -1.7748  0.0234 -0.6627]
3 3 [0.7048 1.1927 3.2093 0.6813 1.7333 1.4982]
0 0 [ 0.7065  0.8203  1.5795 -0.7851  1.7631  1.5133]
2 2 [ 0.8565  1.0809  0.8121 -0.6143 -0.3039 -2.4578]
```

### Turn 11 — `execute_in_remote_docker` (`42292 ms` LLM)
- **Rationale**: Save a working candidate now. The inferred chamber amplitude nearly equals the benchtop drift near its intercept, but deviates in both directions away from it. Test whether a feature-dependent gain on the centered drift explains that pattern and lowers held-out error.
- **Alternatives Weighed**: `Use lab-specific gains` (rejected: Lab fits still leave residuals above measurement noise and offer no prediction rule for a held-out lab.)

```bash
sed '/^def solve_batch/,$d' /tmp/hegel_continual_archive/alpha_m98_v2_c03_s00_solver.py > /holdout_task/batch_solver.py
cat >> /holdout_task/batch_solver.py <<'PY'
def _chamber(coupon, dt, times, taus, theta):
    _, hold, drift = coupon
    kernel = np.vstack((-np.expm1(-times[:, None] / taus), 1.0 / taus))
    x = drift[:, 1:]
    amp = (0.2982 + x @ np.array([0.18509, -0.26309, 0.17816, -0.23787])
           + 0.01086 * (x[:, 0] + 0.5*x[:, 1])**3 + 0.06127*x[:, 1]**2*x[:, 3])
    return (hold[:, None] * theta[:2] + amp[:, None] * taus * [0.41118, 0.58882]) @ kernel.T


def solve_batch(dataset: dict) -> dict:
    train, test = _prepare(dataset)
    dt, times = dataset['past_dt'], dataset['eval_times']
    tr = dataset['train']
    y = np.column_stack((tr['strain_curves_obs'] - tr['strain_0'][:, None], tr['instant_rate_obs']))
    taus, theta = _fit(train, dt, times, y)
    pred = _chamber(test, dt, times, taus, theta)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': dataset['test']['strain_0'][:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
PY
cat >/tmp/m98_product.py <<'PY'
exec(open('/tmp/m98_factor.py').read().split('for tag,x in')[0])
from itertools import combinations
for tag,X in [('joint',x),('frozen',xx)]:
 y=target-b[2]; r=X@b[3:]; N=len(r); lab=ch['lab_id']; L=[lab==g for g in np.unique(lab)]; E=np.eye(N,dtype=bool); Z=np.c_[X,r]; Q={f'x{i}':X[:,i] for i in range(4)}
 for i,j in combinations(range(4),2):
  for s in [-2,-1,-.5,.5,1,2]: Q[f'x{i}{s:+g}x{j}']=X[:,i]+s*X[:,j]
 Q['norm']=np.linalg.norm(X,axis=1); Q['lin']=r
 def score(A, fixed=False):
  Y=y-r if fixed else y; c=np.linalg.lstsq(A,Y,rcond=1e-8)[0]; resid=A@c-Y
  loo=np.array([A[i]@np.linalg.lstsq(A[~e],Y[~e],rcond=1e-8)[0]-Y[i] for i,e in enumerate(E)])
  laberr=np.concatenate([A[e]@np.linalg.lstsq(A[~e],Y[~e],rcond=1e-8)[0]-Y[e] for e in L]); return (np.linalg.norm(laberr)/N**.5,np.linalg.norm(loo)/N**.5,np.linalg.norm(resid)/N**.5,np.max(abs(loo)),c)
 out=[]
 for n,q in Q.items():
  for f,fn in [(q,'linear'),(q*q,'square'),(abs(q),'abs'),(np.tanh(q),'tanh'),(q**3,'cube'),(q*q/(1+.5*q*q),'sat square')]:
   for label,A,fixed in [('fixed',np.c_[r*f],True),('gain',np.c_[r,r*f],False),('offset gain',np.c_[np.ones(N),r,r*f],False),('affine base',np.c_[np.ones(N),X,r*f],False),('unshared',np.c_[np.ones(N),X,X*f[:,None]],False)]:
    out.append((*score(A,fixed),label,n,fn))
 print('FACTORED',tag)
 for v in sorted(out,key=lambda v:v[:4])[:12]: print(v[5:],np.round(v[:4],5),np.round(v[4],5))
 for name,A in [('quadratic gate',np.c_[r,r[:,None]*np.array([X[:,i]*X[:,j] for i,j in combinations_with_replacement(range(4),2)]).T]),('diagonal gate',np.c_[r,r[:,None]*X**2]),('abs gate',np.c_[r,r[:,None]*abs(X)]),('linear gate',np.c_[r,r[:,None]*X])]:
  v=score(A); print(name,np.round(v[:4],5),np.round(v[4],5))
 print('ratios vs features:',np.round(np.c_[Z,y/r],3))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/m98_product.py | grep -E 'FACTORED|fixed|gain|affine base|unshared| gate' | tail -34
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`2041 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
3043 -0.00365  0.00576]
('fixed', 'x0+1x3', 'square') [0.0698  0.06954 0.06721 0.14205] [0.04904]
('fixed', 'x0+1x1', 'square') [0.07255 0.0723  0.07211 0.22151] [0.07299]
('fixed', 'x1+2x2', 'abs') [0.07257 0.07074 0.05297 0.19682] [0.2204]
quadratic gate [0.96983 0.85149 0.01738 2.55705] [ 1.46123 -0.23802  0.45234 -0.03134  0.12285 -0.1488   0.30705  0.08544  0.26387  0.20932  0.0285 ]
diagonal gate [0.17629 0.19261 0.03856 0.69299] [ 1.26762  0.09201 -0.01139 -0.05754  0.00716]
abs gate [0.15577 0.13921 0.05143 0.41424] [ 1.20872  0.28305 -0.03524 -0.09062 -0.00467]
linear gate [0.12721 0.13632 0.05932 0.41398] [ 1.31027  0.15397  0.03768 -0.01447  0.02771]
FACTORED frozen
('fixed', 'norm', 'cube') [0.05428 0.0541  0.05286 0.12852] [0.0174]
('unshared', 'x0-1x1', 'square') [0.06363 0.05797 0.02557 0.10396] [ 0.01612  0.19315 -0.29115  0.17537 -0.21056  0.01727  0.03209 -0.00341  0.00381]
('fixed', 'x0-0.5x2', 'square') [0.06464 0.065   0.06349 0.13626] [0.20769]
('fixed', 'x1+2x2', 'square') [0.06597 0.0651  0.06282 0.16968] [0.06487]
('fixed', 'x0-0.5x2', 'abs') [0.06614 0.06314 0.05295 0.15017] [0.4175]
('fixed', 'norm', 'square') [0.06702 0.06615 0.05397 0.16115] [0.06]
('fixed', 'x0', 'abs') [0.06985 0.06779 0.05977 0.13525] [0.32375]
('fixed', 'x0+1x3', 'square') [0.07058 0.07031 0.06799 0.14813] [0.05034]
('fixed', 'x0+0.5x1', 'abs') [0.07058 0.06798 0.06169 0.14105] [0.29494]
('unshared', 'x1+2x2', 'square') [0.07096 0.06094 0.03019 0.10765] [ 0.01046  0.21608 -0.29267  0.11804 -0.21742 -0.00364  0.01696  0.00788  0.01173]
('fixed', 'x0+2x3', 'square') [0.07457 0.07443 0.06176 0.17142] [0.025]
('unshared', 'x1+1x2', 'abs') [0.07465 0.06617 0.02539 0.16413] [ 0.02369  0.21362 -0.308    0.06746 -0.26823 -0.00461  0.0415   0.09573  0.07379]
quadratic gate [1.04606 1.00178 0.02181 3.06451] [ 1.58229 -0.33737  0.6072  -0.01826  0.1437  -0.22695  0.35378  0.10838  0.2557   0.32455  0.03286]
diagonal gate [0.18611 0.20479 0.04242 0.74369] [ 1.27804  0.09409 -0.02723 -0.04955  0.00414]
abs gate [0.15734 0.14017 0.05484 0.41348] [ 1.18571  0.28028 -0.03876 -0.05203 -0.0058 ]
linear gate [0.12807 0.13665 0.06205 0.39895] [ 1.29687  0.16746  0.02943 -0.01761  0.03651]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=71ab731ca69626aa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_71ab731c (directory).
```

### Turn 12 — `execute_in_remote_docker` (`42933 ms` LLM)
- **Rationale**: Compare bounded corrections with the existing cubic on lab and feature-edge holdouts. Separately check the shared kernel split against the observed curves and rates so an amplitude correction does not conceal a kernel error.
- **Alternatives Weighed**: `Freeze the schema-valid cubic model` (rejected: Deployment includes feature extremes beyond the diagnostic range, where an unconstrained cubic can amplify error.)

```bash
cat > /tmp/m98_bounded.py <<'PY'
exec(open('/tmp/m98_factor.py').read().split('for tag,x in')[0])
from itertools import combinations, combinations_with_replacement
from scipy.optimize import least_squares
X=xx; W=b[3:]; N=len(X); r=X@W; g=ch['lab_id']; y=target; base=np.c_[np.ones(N),X]; fixed=b[2]+r
# Compare compact feature warps to the selected cubic on lab folds and on edge coupons.
E=[g==i for i in np.unique(g)]; E += [abs(X[:,j])>=np.sort(abs(X[:,j]))[-2] for j in range(4)]
def cv(A, offset=0., ridge=0.):
 Y=y-offset; pred=[]
 for v in E:
  a=A[~v]; reg=np.eye(a.shape[1])*ridge; c=np.linalg.solve(a.T@a+reg+1e-10*np.eye(a.shape[1]),a.T@Y[~v]); pred.append(A[v]@c-Y[v]);
 c=np.linalg.lstsq(A,Y,rcond=1e-8)[0]
 return np.array([np.sqrt(np.mean(np.concatenate(pred[:4])**2)),np.sqrt(np.mean(np.concatenate(pred[4:])**2)),max(np.sqrt(np.mean(z*z)) for z in pred),np.sqrt(np.mean((A@c-Y)**2))]),c
A=np.c_[base,(X[:,0]+.5*X[:,1])**3,X[:,1]**2*X[:,3]]; print('candidate lab, edge, worst, fit',cv(A)[0]);
for a in [0.,.1,.25,.5,1.]:
 for name,Q in [('diagonal-cube',X**3),('positive/negative cubic',np.c_[np.maximum(X,0)**3,np.minimum(X,0)**3]),('cubic-compaction',np.c_[(X[:,0]+.5*X[:,1])**3,X[:,1]**2*X[:,3]])]:
  den=np.c_[1+a*X**2,1+a*X**2] if Q.shape[1]==8 else 1+a*X**2 if Q.shape[1]==4 else np.c_[1+a*(X[:,0]+.5*X[:,1])**2,1+a*X[:,1]**2]
  for ridge in [.01,.1,1.]:
   m,c=cv(np.c_[base,Q/den],ridge=ridge); print('warp',name,a,ridge,np.round(m,5))
# Screen bounded rational corrections to known drift, and rank both edge and lab transfer rather than pooled fit.
L={f'x{j}':X[:,j] for j in range(4)}; L['r']=r
for i,j in combinations(range(4),2):
 for s in (-2,-1,-.5,.5,1,2): L[f'x{i}{s:+}x{j}']=X[:,i]+s*X[:,j]
P={'r':r,'r2':r*r,'r3':r**3}; P.update(L)
for ids in combinations_with_replacement(range(4),3): P[str(ids)]=np.prod(X[:,ids],axis=1)
for k,v in L.items(): P[k+'^3']=v**3; P[k+'abs']=v*abs(v)
res=[]
for pn,pv in P.items():
 for qn,qv in L.items():
  for a in (.1,.3,1.):
   z=pv/(1+a*qv*qv)
   for label,A,off in [('fixed',z[:,None],fixed),('affine',np.c_[base,z],0.)]:
    m,c=cv(A,off); res.append((max(m[0],m[1]),m,pn,qn,a,label,c))
for strength,m,pn,qn,a,label,c in sorted(res,key=lambda u:u[0])[:10]: print('rational',label,pn,qn,a,np.round(m,5),np.round(c,5))
# Independently test whether the empirical rate split or a simple rounded split is supported.
for f in [.38,.4,.41118,.42,.45]:
 F=K@(tau*np.array([f,1-f])); tar=D@F/(F@F); print('kernel split',f,metrics(tar[:,None]*F-D))
print('candidate cubic amplitudes public diagnostic/test ranges',np.quantile(A[:,0],[0,1]) if False else np.round(np.quantile((xp[len(tr['lab_id']):,0]+.5*xp[len(tr['lab_id']):,1])**3,[0,.05,.5,.95,1]),3))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/m98_bounded.py | grep -E 'candidate|rational|kernel' ; OPENBLAS_NUM_THREADS=1 python3 /tmp/m98_bounded.py | grep '^warp' | sort -g -k5 | head -12
```
- **Container Exit Code**: `0` (`16022 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
3  0.21496 -0.29459  0.14257 -0.20362  0.05656]
rational affine x1+2x3^3 x0-2x3 1.0 [0.04798 0.05255 0.09096 0.02789] [ 0.28134  0.20617 -0.28105  0.16961 -0.22102  0.01006]
rational affine x0+1x1^3 x1-2x3 0.3 [0.05406 0.04982 0.11646 0.02818] [ 0.28837  0.17197 -0.32093  0.18476 -0.20543  0.02444]
rational affine x2+1x3 x1+1x2 0.1 [0.05546 0.05984 0.08737 0.0275 ] [ 0.2888   0.21126 -0.25121  0.56084  0.18642 -0.42172]
rational affine x1+1x2^3 x0-2x3 1.0 [0.0581  0.06109 0.12028 0.03254] [ 0.28373  0.20919 -0.28138  0.15804 -0.2006   0.03381]
rational affine (0, 3, 3) x1-1x3 1.0 [0.05655 0.06308 0.10917 0.03125] [ 0.2672   0.18114 -0.24023  0.17481 -0.18621  0.11086]
rational affine x0+2x2^3 x0-2x1 1.0 [0.05237 0.06391 0.11373 0.02988] [ 0.27568  0.19864 -0.25664  0.13742 -0.19771  0.00759]
rational affine x0+1x1 x0-1x1 0.1 [0.06057 0.0641  0.12557 0.02846] [ 0.29018  0.54745  0.10076  0.16541 -0.20916 -0.38199]
kernel split 0.38 [[0.0037662, 0.0057565, 0.0096373], [0.0102841, 0.014418, 0.0250698]]
kernel split 0.4 [[0.0030442, 0.0044198, 0.0072741], [0.0031162, 0.0046403, 0.0069171]]
kernel split 0.41118 [[0.0027252, 0.0039357, 0.0060032], [0.0024316, 0.0039997, 0.0065897]]
kernel split 0.42 [[0.002425, 0.0037791, 0.0060909], [0.0050032, 0.00808, 0.0129294]]
kernel split 0.45 [[0.0032108, 0.0049954, 0.0075655], [0.0135125, 0.0249183, 0.0376886]]
candidate cubic amplitudes public diagnostic/test ranges [-43.955 -16.372  -0.184   1.508   4.793]
warp cubic-compaction 0.0 0.01 [0.04605 0.09676 0.14441 0.02003]
warp cubic-compaction 0.0 0.1 [0.05797 0.06809 0.15112 0.02003]
warp cubic-compaction 0.0 1.0 [0.1723  0.21922 0.39816 0.02003]
warp cubic-compaction 0.1 0.01 [0.07716 0.08766 0.20587 0.02141]
warp cubic-compaction 0.1 0.1 [0.08619 0.11121 0.23595 0.02141]
warp cubic-compaction 0.1 1.0 [0.1701  0.2566  0.37242 0.02141]
warp cubic-compaction 0.25 0.01 [0.09377 0.14209 0.25164 0.02641]
warp cubic-compaction 0.25 0.1 [0.10387 0.16343 0.27508 0.02641]
warp cubic-compaction 0.25 1.0 [0.16799 0.27125 0.38954 0.02641]
warp cubic-compaction 0.5 0.01 [0.1072  0.17741 0.28172 0.03377]
warp cubic-compaction 0.5 0.1 [0.11992 0.19691 0.30012 0.03377]
warp cubic-compaction 0.5 1.0 [0.16116 0.27728 0.39756 0.03377]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=71ab731ca69626aa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_71ab731c (directory).
```

### Turn 13 — `execute_in_remote_docker` (`50353 ms` LLM)
- **Rationale**: Use inner lab folds to choose among amplitude corrections before scoring outer lab and feature-edge holdouts. If the apparent gain disappears, retain the simpler law rather than promote a term selected on the stress slices.
- **Alternatives Weighed**: `Promote the best rational term from the full-data search now` (rejected: Its apparent edge improvement was measured on the same folds used to select the term.)

```bash
cat >/tmp/m98_gate_cv.py <<'PY'
exec(open('/tmp/m98_factor.py').read().split('for tag,x in')[0])
X=xx; r=X@b[3:]; y=target; n=len(y); lab=ch['lab_id']; R=y-(b[2]+r)
A=np.c_[np.ones(n),X]; u=X[:,0]+.5*X[:,1]; v=X[:,1]; z=X[:,3]
F=[('lab'+str(g),lab==g) for g in np.unique(lab)]+[('edge'+str(j),abs(X[:,j])>=np.sort(abs(X[:,j]))[-2]) for j in range(4)]
for i in np.argsort(r): print('amp',i,np.round(np.r_[X[i],r[i],R[i],R[i]/r[i]],4))
def fit(B,y,k,pen=0.):
 M=B[k]; return np.linalg.lstsq(np.r_[M,np.sqrt(pen)*np.eye(M.shape[1])],np.r_[y[k],np.zeros(M.shape[1])],rcond=1e-9)[0]
lib={}
def add(name,Q,base=A,offset=0.,ridge=0.): lib[name]=(np.column_stack((base,Q)),np.broadcast_to(offset,y.shape),ridge)
empty=np.empty((n,0)); add('linear',empty); add('cubic',np.c_[u**3,v*v*z]); add('rational winner',((X[:,1]+X[:,2])**3/(1+(v-z)**2))[:,None]); add('rational alternate',((X[:,0]+2*v)**3/(1+(v-z)**2))[:,None]);
for c in (0.,.5,1.,1.5):
 for p in (2,3):
  P=np.maximum(abs(X)-c,0)**p; S=P*np.sign(X); W=b[3:];
  add(f'shared signed {c} {p}',S@W); add(f'shared one sided {c} {p}',np.maximum(np.sign(W)*X-c,0)**p@abs(W)); add(f'ratio norm {c} {p}',r*P.sum(1),empty,b[2]+r); add(f'component shrink {c} {p}',S,A,0.,.04)
  add(f'ratio diagonals {c} {p}',r[:,None]*P,empty,b[2]+r,.04)
for c in (.025,.05,.1,.2):
 for p in (2,3):
  for mode in ('norm','weighted norm','max','signed max'):
   S={'norm':np.linalg.norm(X,axis=1),'weighted norm':np.linalg.norm(X*b[3:],axis=1)/np.linalg.norm(b[3:]),'max':abs(X).max(1),'signed max':(np.sign(b[3:])*X).max(1)}[mode]; add(f'{mode} sat {p} {c}', r*S**p/(1+c*S**p),empty,b[2]+r)
for name,Q in [('calibrated hold',hold),('strain0',ch['strain_0']),('mean history',past.mean(1)),('history spread',past.std(1)),('last gap',hold-past[:,-1])]:
 add(name,Q); add(name+' gate',r*Q,empty,b[2]+r)
def err(name,v):
 B,o,p=lib[name]; return B[v]@fit(B,y-o,~v,p)+o[v]-y[v]
def stats(es): return np.round([np.sqrt(np.mean(np.concatenate(es[:4])**2)),np.sqrt(np.mean(np.concatenate(es[4:])**2)),max(np.sqrt(np.mean(e**2)) for e in es)],5)
rank=sorted((tuple(stats([err(name,v) for _,v in F])),name) for name in lib); print('best fixed choices',sorted(rank,key=lambda t:max(t[0][:2]))[:14]); print('baselines',[(name,stats([err(name,v) for _,v in F])) for name in list(lib)[:4]])
for label, names in [('sparse',list(lib)[:4]),('shared',list(lib)[4:])]:
 es=[]; sel=[]
 for _,v in F:
  losses=[]
  for name in names:
   B,o,p=lib[name]; chunks=[]
   for g in np.unique(lab[~v]):
    m=(lab==g)&~v; k=(lab!=g)&~v; chunks.extend(B[m]@fit(B,y-o,k,p)+o[m]-y[m]);
   losses.append(np.sqrt(np.mean(np.square(chunks))))
  name=names[np.argmin(losses)]; sel.append(name); es.append(err(name,v))
 print('nested',label,stats(es),sel)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/m98_gate_cv.py | grep -E '^amp|^best|^baselines|^nested'
```
- **Container Exit Code**: `0` (`719 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.1016  2.5299 -0.049  -0.0092  0.1883]
amp 8 [-0.1757 -0.4598  0.1095  0.7805 -0.0444 -0.0148  0.3328]
amp 12 [-1.6881 -0.5983  1.552   0.5154 -0.0429 -0.0064  0.1487]
amp 11 [ 1.318  -0.8368 -0.995   1.6923 -0.0016  0.0019 -1.2271]
amp 17 [0.5456 0.1771 0.158  0.5116 0.0048 0.0005 0.1093]
amp 5 [-0.1456 -0.1584  0.478   0.0692  0.0595  0.0176  0.2949]
amp 9 [-0.901  -0.3449  1.7217  0.2292  0.1097  0.1127  1.0272]
amp 16 [-0.7193 -1.5616 -0.0327  0.1832  0.1427  0.1413  0.99  ]
amp 7 [ 1.1204  0.031  -0.0282  0.1634  0.1583  0.0065  0.0412]
amp 15 [ 0.8656  1.0562  0.3363 -2.0995  0.3067  0.0504  0.1643]
amp 6 [-0.1838 -1.7748  0.0234 -0.6627  0.4096  0.067   0.1637]
amp 3 [3.2093 0.6813 1.7333 1.4982 0.4316 0.488  1.1305]
amp 0 [ 1.5795 -0.7851  1.7631  1.5133  0.4333  0.1138  0.2626]
amp 2 [ 0.8121 -0.6143 -0.3039 -2.4578  0.5834  0.2244  0.3846]
best fixed choices [((0.04578, 0.04103, 0.09186), 'rational alternate'), ((0.05094, 0.02254, 0.07607), 'ratio norm 0.0 3'), ((0.03848, 0.05245, 0.0849), 'rational winner'), ((0.05314, 0.04505, 0.08196), 'weighted norm sat 3 0.025'), ((0.054, 0.01932, 0.07282), 'ratio norm 1.0 2'), ((0.0543, 0.03079, 0.07963), 'ratio norm 0.5 3'), ((0.056, 0.04299, 0.08907), 'weighted norm sat 3 0.05'), ((0.05641, 0.04315, 0.09668), 'ratio norm 0.5 2'), ((0.05541, 0.06146, 0.11256), 'shared one sided 1.0 3'), ((0.06332, 0.0616, 0.11851), 'weighted norm sat 3 0.1'), ((0.06555, 0.05613, 0.09338), 'ratio norm 1.5 2'), ((0.06, 0.07431, 0.11838), 'shared one sided 1.5 2'), ((0.06702, 0.0764, 0.13573), 'ratio norm 0.0 2'), ((0.06642, 0.07761, 0.13483), 'weighted norm sat 2 0.025')]
baselines [('linear', array([0.13405, 0.20653, 0.30177])), ('cubic', array([0.04646, 0.10084, 0.15199])), ('rational winner', array([0.03848, 0.05245, 0.0849 ])), ('rational alternate', array([0.04578, 0.04103, 0.09186]))]
nested sparse [0.04941 0.06135 0.10243] ['rational winner', 'cubic', 'rational alternate', 'cubic', 'rational winner', 'cubic', 'rational winner', 'rational alternate']
nested shared [0.10441 0.20383 0.27881] ['ratio norm 0.0 3', 'ratio norm 0.0 3', 'ratio norm 1.0 2', 'calibrated hold gate', 'calibrated hold gate', 'component shrink 1.0 3', 'calibrated hold gate', 'ratio norm 0.0 3']

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=71ab731ca69626aa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_71ab731c (directory).
```

### Turn 14 — `execute_in_remote_docker` (`37810 ms` LLM)
- **Rationale**: The amplitude correction generally has the sign of the centered intrinsic drift. Test low-rank and regularized gains using lab and feature-edge holdouts; if they do not improve transfer, use the evidence from nested sparse selection to regularize the existing candidate.
- **Alternatives Weighed**: `Deploy the lowest-error rational term from the full search` (rejected: Its apparent advantage is selection-sensitive, and its cubic numerator can extrapolate along directions the denominator does not constrain.)

```bash
cat >/tmp/m98_gainprobe.py <<'PY'
exec(open('/tmp/m98_factor.py').read().split('for tag,x in')[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
for tag,X in [('joint',x),('frozen',xx)]:
 r=X@b[3:]; y=target-b[2]-r; N=len(y); lab=ch['lab_id']; folds=[lab==g for g in np.unique(lab)]+[abs(X[:,j])>=np.sort(abs(X[:,j]))[-2] for j in range(4)]; allk=np.ones(N,bool)
 Q=np.c_[X, hold, past, ch['strain_0'],past.mean(1),past.std(1),past.max(1)-past.min(1)]; Q[:,4:]=(Q[:,4:]-Q[:,4:].mean(0))/Q[:,4:].std(0)
 Qnames=['x0','x1','x2','x3','hold',*['p'+str(j) for j in range(6)],'strain0','past mean','past sd','past range']; print(tag,'ratio correlations', sorted(zip(Qnames, np.round(np.corrcoef(np.c_[Q,(y/r)].T)[-1,:-1],3)),key=lambda z:-abs(z[1]))[:8]);
 for grp in ([0,1,2,3],list(range(4,12)),[12,13,14],list(range(15))):
  Z=Q[:,grp]
  for mode in ('linear','diagonal square','diagonal cube','quadratic'):
   W=Z if mode=='linear' else Z**2 if mode=='diagonal square' else abs(Z)**3 if mode=='diagonal cube' else np.array([Z[:,i]*Z[:,j] for i,j in combinations_with_replacement(range(Z.shape[1]),2)]).T
   A=r[:,None]*np.c_[np.ones(N),W]
   for p in (.003,.03):
    mu=np.linalg.norm(A,axis=0); B=A/np.maximum(mu,1e-8); P=p*np.eye(B.shape[1]); P[0,0]=0
    def fit(k): return np.linalg.solve(B[k].T@B[k]+P,B[k].T@y[k]);
    e=[B[v]@fit(~v)-y[v] for v in folds]; print('ridge',grp[0],grp[-1],mode,p,np.round([np.sqrt(np.mean((B@fit(allk)-y)**2)), np.sqrt(np.mean(np.concatenate(e[:4])**2)),np.sqrt(np.mean(np.concatenate(e[4:])**2)),max(np.sqrt(np.mean(a*a)) for a in e)],5))
 rng=np.random.default_rng(13); starts=rng.normal(size=(12,5))*.25
 for typ in ('square','cube','bounded','shifted'):
  def predict(p):
   a=X@p[:4]; c=p[4]; u=a*a if typ=='square' else abs(a)**3 if typ=='cube' else a*a/(1+a*a) if typ=='bounded' else (a+c)**2; return r*(u+(0 if typ=='shifted' else c))
  def fit(k):
   def fun(p): return np.r_[ (predict(p)-y)[k], .04*p ]
   z=[least_squares(fun,p,max_nfev=250) for p in starts]; return min(z,key=lambda w:w.cost).x
  p=fit(allk); e=[(predict(fit(~v))-y)[v] for v in folds]; print('projection',typ,np.round(p,4),np.round([np.sqrt(np.mean((predict(p)-y)**2)), np.sqrt(np.mean(np.concatenate(e[:4])**2)),np.sqrt(np.mean(np.concatenate(e[4:])**2)),max(np.sqrt(np.mean(a*a)) for a in e)],5))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/m98_gainprobe.py | awk '/ratio correlations|projection|ridge 0 3|ridge 4 11/'
```
- **Container Exit Code**: `0` (`3022 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.11204 0.11944 0.24524]
ridge 4 11 diagonal square 0.03 [0.02398 0.09905 0.15008 0.25283]
ridge 4 11 diagonal cube 0.003 [0.02577 0.12182 0.09126 0.25894]
ridge 4 11 diagonal cube 0.03 [0.02623 0.1034  0.15685 0.26935]
ridge 4 11 quadratic 0.003 [0.00086 0.0783  0.04796 0.18361]
ridge 4 11 quadratic 0.03 [0.00438 0.07904 0.06935 0.18764]
projection square [ 0.2638  0.1178 -0.2075  0.2874  0.1002] [0.02979 0.147   0.78859 1.53746]
projection cube [ 0.2607  0.129  -0.2357  0.3033  0.1898] [0.02944 0.15816 3.06026 6.10958]
projection bounded [ 0.3957  0.1792 -0.2113  0.3265  0.1242] [0.0461  0.13806 0.16644 0.22885]
projection shifted [ 0.3056  0.1058 -0.2621  0.3356 -0.0438] [0.03128 0.20414 0.23946 0.33963]
frozen ratio correlations [('x2', 0.527), ('p3', -0.411), ('p4', 0.358), ('p5', -0.301), ('p2', 0.228), ('past sd', -0.213), ('past range', -0.2), ('x3', -0.115)]
ridge 0 3 linear 0.003 [0.06205 0.12777 0.2631  0.42763]
ridge 0 3 linear 0.03 [0.06215 0.1259  0.22656 0.336  ]
ridge 0 3 diagonal square 0.003 [0.04242 0.17217 0.27941 0.46609]
ridge 0 3 diagonal square 0.03 [0.04279 0.13571 0.20608 0.32375]
ridge 0 3 diagonal cube 0.003 [0.03884 0.18588 0.37328 0.62778]
ridge 0 3 diagonal cube 0.03 [0.03906 0.12937 0.20792 0.31312]
ridge 0 3 quadratic 0.003 [0.02498 0.28886 0.4255  0.47966]
ridge 0 3 quadratic 0.03 [0.03023 0.12715 0.18435 0.23449]
ridge 4 11 linear 0.003 [0.05635 0.48459 0.53053 0.83209]
ridge 4 11 linear 0.03 [0.0565  0.27778 0.36013 0.42102]
ridge 4 11 diagonal square 0.003 [0.0283  0.1385  0.14133 0.27391]
ridge 4 11 diagonal square 0.03 [0.02889 0.11547 0.16622 0.27469]
ridge 4 11 diagonal cube 0.003 [0.03087 0.14836 0.11339 0.28843]
ridge 4 11 diagonal cube 0.03 [0.0312  0.11801 0.17084 0.28558]
ridge 4 11 quadratic 0.003 [0.00141 0.0792  0.04255 0.17643]
ridge 4 11 quadratic 0.03 [0.00544 0.08017 0.06332 0.18069]
projection square [ 0.2683  0.1015 -0.2066  0.2976  0.0863] [0.0363  0.16307 0.52166 0.95059]
projection cube [ 0.2582  0.1051 -0.2214  0.3092  0.186 ] [0.03595 0.18369 0.84809 1.61429]
projection bounded [ 0.3997  0.145  -0.2099  0.3308  0.1268] [0.05267 0.14618 0.20453 0.27222]
projection shifted [ 0.3074  0.0925 -0.2537  0.3377 -0.0446] [0.03689 0.22109 0.23431 0.35614]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=71ab731ca69626aa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_71ab731c (directory).
```

### Turn 15 — `execute_in_remote_docker` (`33961 ms` LLM)
- **Rationale**: Compare fixed blends on lab and feature-edge slices and inspect their deployment amplitude ranges. Promote a blend only if it reduces the worst held-out error without producing unsupported extremes. Test simple load contrasts separately; retain the reset law if they do not improve transfer.
- **Alternatives Weighed**: `Promote a dense load-history gain` (rejected: Its near-zero fit error did not transfer to held-out labs or feature extremes.)

```bash
cat >/tmp/m98_finalprobe.py <<'PY'
exec(open('/tmp/m98_gate_cv.py').read().split('rank=sorted')[0])
from itertools import combinations
for g in (0.,.25,.5,.75,1.):
 names=['cubic','rational winner','rational alternate','ratio norm 1.0 2','ratio norm 0.0 3']; w=np.array([g,(1-g)*.25,(1-g)*.25,(1-g)*.35,(1-g)*.15]); es=[sum(a*err(name,v) for a,name in zip(w,names)) for _,v in F]; print('ENSEMBLE',g,stats(es))
for name in ('cubic','rational winner','rational alternate','ratio norm 1.0 2','ratio norm 0.0 3'):
 B,o,p=lib[name]; c=fit(B,y-o,np.ones(n,bool),p); print('COEF',name,c)
# Test whether a simple load-history contrast, as opposed to a dense quadratic, explains the drift gain.
load=np.c_[hold,past]; dv={f'p{j}':load[:,j] for j in range(load.shape[1])}; dv.update({'mean':past.mean(1),'std':past.std(1),'range':np.ptp(past,axis=1),'roughness':np.sqrt(np.mean(np.diff(past,axis=1)**2,axis=1))});
for i,j in combinations(range(load.shape[1]),2):
 for s in (-1.,-.5,.5,1.): dv[f'p{i}{s:+}p{j}']=load[:,i]+s*load[:,j]
for name,q in dv.items():
 for z,zz in [('sq',q*q),('abs',abs(q)),('cube',q**3),('centered_sq',(q-np.mean(q))**2)]: add('load '+name+' '+z,r*zz,np.c_[r],b[2]+r)
loadout=[]
for name in lib:
 if name.startswith('load '):
  e=[err(name,v) for _,v in F]; z=stats(e); loadout.append((max(z[:2]),tuple(z),name))
print('LOAD',sorted(loadout)[:6]); print('FALLBACK',stats([err('linear',v) for _,v in F]));
Xtest=xp[len(tr['lab_id']):]; u=Xtest[:,0]+.5*Xtest[:,1]; v=Xtest[:,1]; z=Xtest[:,3]; rt=Xtest@b[3:]; Btest=[np.c_[np.ones(len(rt)),Xtest,u**3,v*v*z],np.c_[np.ones(len(rt)),Xtest,(v+Xtest[:,2])**3/(1+(v-z)**2)],np.c_[np.ones(len(rt)),Xtest,(Xtest[:,0]+2*v)**3/(1+(v-z)**2)],(rt*np.maximum(abs(Xtest)-1,0).__pow__(2).sum(1))[:,None],(rt*(abs(Xtest)**3).sum(1))[:,None]]
for name,Bt in zip(names,Btest):
 B,o,p=lib[name]; c=fit(B,y-o,np.ones(n,bool),p); a=Bt@c+(b[2]+rt if name.startswith('ratio') else 0.); print('DEPLOY',name,np.round(np.quantile(a,[0,.05,.5,.95,1]),4))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/m98_finalprobe.py | grep -E '^(ENSEMBLE|COEF|LOAD|FALLBACK|DEPLOY)'; cat /tmp/hegel_continual_archive/archive_manifest.json; head -12 /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`674 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ENSEMBLE 0.0 [0.03951 0.02282 0.06061]
ENSEMBLE 0.25 [0.03783 0.02197 0.06864]
ENSEMBLE 0.5 [0.03856 0.04559 0.07833]
ENSEMBLE 0.75 [0.04159 0.07285 0.10669]
ENSEMBLE 1.0 [0.04646 0.10084 0.15199]
COEF cubic [ 0.2982   0.18509 -0.26309  0.17816 -0.23787  0.01086  0.06127]
COEF rational winner [ 0.28133  0.21496 -0.29459  0.14257 -0.20362  0.05656]
COEF rational alternate [ 0.28469  0.18539 -0.32837  0.17639 -0.20921  0.0094 ]
COEF ratio norm 1.0 2 [0.19849]
COEF ratio norm 0.0 3 [0.02632]
LOAD [(0.07276, (0.07276, 0.0617, 0.17683), 'load p6 centered_sq'), (0.12756, (0.07719, 0.12756, 0.19039), 'load p0-1.0p6 centered_sq'), (0.13897, (0.09508, 0.13897, 0.2124), 'load p4+1.0p6 centered_sq'), (0.14053, (0.07612, 0.14053, 0.19101), 'load p1+1.0p6 centered_sq'), (0.14395, (0.05605, 0.14395, 0.21565), 'load p6 cube'), (0.1462, (0.07904, 0.1462, 0.20049), 'load p3+1.0p6 centered_sq')]
FALLBACK [0.13405 0.20653 0.30177]
DEPLOY cubic [-1.3061 -0.4008  0.2752  0.9591  1.2039]
DEPLOY rational winner [-1.4908 -0.5478  0.5253  1.6851  2.0728]
DEPLOY rational alternate [-1.4643 -0.57    0.4879  1.6883  2.0983]
DEPLOY ratio norm 1.0 2 [-1.9866 -0.5252  0.2542  0.8124  1.2334]
DEPLOY ratio norm 0.0 3 [-1.8701 -0.5407  0.2534  0.8391  1.2939]
[
  {
    "episode_short_name": "alpha_m98_v2_c03_s00",
    "task_id": "alpha_m98_v2_c03_s00__blind_continual_campaign",
    "entrypoint": "solve_batch",
    "output_schema_keys": [
      "estimated_taus",
      "hereditary_weights",
      "initial_rate_pred",
      "predicted_strain"
    ],
    "solver_path": "/tmp/hegel_continual_archive/alpha_m98_v2_c03_s00_solver.py"
  }
]import importlib.util, json, os, pickle, sys
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_batch.pkl"), "rb"))[0]
ds = item["dataset"]
out = mod.solve_batch(ds)
n_te = len(ds["test"]["lab_id"]); k_t = len(ds["eval_times"])
errs = []
if not isinstance(out, dict): errs.append("not_dict")

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=71ab731ca69626aa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_71ab731c (directory).
```

### Turn 16 — `execute_in_remote_docker` (`32547 ms` LLM)
- **Rationale**: Promote the fixed blend and verify the exact artifact in isolated Python. The derivative and history-perturbation checks would catch an inconsistent rate calculation or accidental retention of benchtop memory in the chamber.
- **Alternatives Weighed**: `Keep the cubic-only candidate` (rejected: The fixed blend substantially lowers the worst diagnostic slice error without changing the benchtop fit or the validated chamber kernel.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p=Path('/holdout_task/batch_solver.py')
s=p.read_text()
a=s.index('def _chamber(')
z=s.index('def solve_batch(',a)
s=s[:a]+'''def _chamber_amplitude(x, theta):
    # Fixed chamber correction inferred from the public verification coupons.
    # Average the sparse rational fits with gains on the intrinsic drift.
    x0, x1, x2, x3 = x.T
    divisor = 1.0 + (x1 - x3)**2
    rational_a = (0.28133 + x @ np.array([0.21496, -0.29459, 0.14257, -0.20362])
                  + 0.05656 * (x1 + x2)**3 / divisor)
    rational_b = (0.28469 + x @ np.array([0.18539, -0.32837, 0.17639, -0.20921])
                  + 0.00940 * (x0 + 2.0*x1)**3 / divisor)
    centered_drift = x @ theta[3:]
    magnitude = np.abs(x)
    gain = 1.0 + 0.35*0.19849*np.sum(np.maximum(magnitude - 1.0, 0.0)**2, axis=1)
    gain += 0.15*0.02632*np.sum(magnitude**3, axis=1)
    return 0.25*(rational_a + rational_b) + 0.5*theta[2] + (gain - 0.5)*centered_drift


def _chamber(coupon, times, taus, theta):
    _, hold, drift = coupon
    # The chamber resets history; its coupon correction relaxes in the same two branches.
    kernel = np.vstack((-np.expm1(-times[:, None] / taus), 1.0 / taus))
    amp = _chamber_amplitude(drift[:, 1:], theta)
    coefficients = hold[:, None] * theta[:2] + amp[:, None] * taus * [0.41118, 0.58882]
    return coefficients @ kernel.T


'''+s[z:]
s=s.replace('_chamber(test, dt, times, taus, theta)', '_chamber(test, times, taus, theta)')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; out=m.solve_batch(d)
assert all(np.isfinite(v).all() for v in out.values()); t=out['estimated_taus']; assert .05<t[0]<t[1]<25
_,cu=m._prepare(d); theta=out['hereditary_weights']; h=1.e-6
v=m._chamber(cu,np.array([0.,h]),t,theta)
np.testing.assert_allclose(v[:,0],0.,atol=1e-12)
np.testing.assert_allclose(v[:,1]/h,out['initial_rate_pred'],rtol=3e-6,atol=1e-5)
np.testing.assert_allclose(m._chamber((cu[0]+3.,*cu[1:]),d['eval_times'],t,theta)[:,-1],out['initial_rate_pred'])
print('Final artifact: finite outputs, ordered spectrum, exact zero-time boundary, initial derivative, and chamber reset verified.')
PY
```
- **Container Exit Code**: `0` (`993 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Final artifact: finite outputs, ordered spectrum, exact zero-time boundary, initial derivative, and chamber reset verified.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8dfd625f01b067db (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_8dfd625f (directory).
```

### Turn 17 — `execute_in_remote_docker` (`21025 ms` LLM)
- **Rationale**: Compare the exact saved solver with chronological memory on measured chamber coupons, holding calibration and material estimates fixed. If either output regresses in any lab, inspect that slice before submission.
- **Alternatives Weighed**: `Submit based on the schema receipt alone` (rejected: The receipt does not show that the saved solver improves both measured chamber outputs across labs.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import contextlib, importlib.util, io, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/m98_factor.py').read().split('for tag,x in')[0])
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
out = m.solve_batch(d)
taus, theta = out['estimated_taus'], out['hereditary_weights']
cu = (past, hold, np.c_[np.ones(len(hold)), xx])
groups = ch['lab_id']
a = m._chamber(cu, t, taus, theta) - Y
b = m._design(cu, dt, t, taus) @ theta - Y
def lab_metrics(e):
    return np.array([[np.sqrt(np.mean(e[groups == g, :-1] ** 2)), np.sqrt(np.mean(e[groups == g, -1] ** 2))] for g in np.unique(groups)])
base, candidate = lab_metrics(b), lab_metrics(a)
print('Worst diagnostic lab RMSE [strain, initial rate], chronological baseline:', base.max(axis=0), 'final chamber model:', candidate.max(axis=0))
assert np.all(candidate < base), (base, candidate)
PY
```
- **Container Exit Code**: `0` (`722 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Worst diagnostic lab RMSE [strain, initial rate], chronological baseline: [0.76864 2.23365] final chamber model: [0.04736 0.03804]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8dfd625f01b067db (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_8dfd625f (directory).
```

### Turn 18 — `submit_completed_task` (`15072 ms` LLM)
- **Rationale**: The unchanged saved artifact passed isolated schema, boundary, and derivative checks and improved both measured chamber outputs in every public diagnostic lab. Private-coupon accuracy was not measured.
- **Alternatives Weighed**: `Retain the cubic-only chamber correction` (rejected: Its worst held-out diagnostic slice amplitude RMSE was 0.1520, versus 0.0606 for the blend.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema and isolated numerical checks passed. On the public chamber diagnostics used for model development, worst-lab RMSE fell from 0.769 to 0.047 for strain and from 2.234 to 0.038 for initial rate.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M98-Blind-Mismatch-Factored-Lineage-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
