# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m98_v2_c04_batch_04__blind_continual_campaign`

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
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `349,061` (`338,701` / `114,176` / `10,360`) |
| **Total Episode Wall Time (s)** | `378.77s` |

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

### Turn 1 — `execute_in_remote_docker` (`17712 ms` LLM)
- **Rationale**: Inspect the public schema and diagnostics status before choosing a chamber law. This detects missing diagnostics or incompatible array layouts, which would change the source of regime evidence or the solver implementation.

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation robust IRLS reference panel alignment' || true; python3 - <<'PY'
import pickle
with open('/holdout_task/public_batch.pkl','rb') as f: item=pickle.load(f)[0]
def show(x,p='',n=0):
 if isinstance(x,dict) and n<3:
  for k,v in x.items(): show(v,p+'/'+str(k),n+1)
 else: print(p,type(x).__name__,getattr(x,'shape',None),len(x) if isinstance(x,(list,tuple)) else '')
show(item)
PY
```
- **Container Exit Code**: `0` (`478 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
one=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict] | Locate one localized flaw in a rectangular plate; return (xy, diagnostics) | REQ: {'field': 'samples_per_carrier_period', 'op': '>', 'value': 2}; {'field': 'unique_unorde | --describe autogenous_dispersive_wave_tomography.localize_pitch_catch
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
/dataset/past_dt ndarray (6,) 
/dataset/eval_times ndarray (12,) 
/dataset/calibration_load_cells/bridge_raw ndarray (80, 4) 
/dataset/calibration_load_cells/true_channel_loads ndarray (80, 4) 
/dataset/calibration_load_cells/single_channel_sweeps dict None 
/dataset/train/lab_id ndarray (72,) 
/dataset/train/stratum ndarray (72,) 
/dataset/train/coupon_features_raw ndarray (72, 4) 
/dataset/train/ref_coupons ndarray (72, 8) 
/dataset/train/past_channel_raw ndarray (72, 6, 4) 
/dataset/train/hold_channel_raw ndarray (72, 4) 
/dataset/train/strain_0 ndarray (72,) 
/dataset/train/strain_curves_obs ndarray (72, 12) 
/dataset/train/instant_rate_obs ndarray (72,) 
/dataset/train/aux_chamber_telemetry ndarray (72, 4) 
/dataset/test/lab_id ndarray (36,) 
/dataset/test/stratum ndarray (36,) 
/dataset/test/coupon_features_raw ndarray (36, 4) 
/dataset/test/ref_coupons ndarray (36, 8) 
/dataset/test/past_channel_raw ndarray (36, 6, 4) 
/dataset/test/hold_channel_raw ndarray (36, 4) 
/dataset/test/strain_0 ndarray (36,) 
/dataset/test/aux_chamber_telemetry ndarray (36, 4) 
/campaign_diagnostics NoneType None
```

### Turn 2 — `execute_in_remote_docker` (`19565 ms` LLM)
- **Rationale**: Compare apparatus signatures and archived equations to detect a regime mismatch before reusing any solver. Check whether the panel primitive implements the specified reference statistics; if it does not, implement that transformation directly.
- **Alternatives Weighed**: `Copy the latest archived solver unchanged` (rejected: Archive recency does not establish that its chamber law matches the current blind campaign.)

```bash
cd /tmp/hegel_continual_archive && cat archive_manifest.json archive_apparatus_log.json && diff -u alpha_m98_v2_c04_batch_02_solver.py alpha_m98_v2_c04_batch_01_solver.py; cat /tmp/hegel_continual_archive/alpha_m98_v2_c04_batch_02_solver.py; cat /holdout_task/public_asset_index.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_control_panel_deconfounding.align_control_panel
```
- **Container Exit Code**: `0` (`376 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[
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
        "autogenous_control_panel_deconfounding@aa502f6d | fit_predict_neutral(X_train: np.ndarray, y_train: np.ndarray, center_train: np.ndarray, X_test: np.ndarray, center_test: np.ndarray, signal_columns: tuple[int, int, int, int, int], control_columns: tuple[int, ...] | None=None, *, C: float=3.0) -> np.ndarray | Fit equal-prior logistic probabilities for a prespecified signal basis | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | --describe autogenous_control_panel_deconfounding.fit_predict_neutral"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_control_panel_deconfounding",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`25084 ms` LLM)
- **Rationale**: The previous output truncated the chamber equations and apparatus comparison. Read the solver differences and compact telemetry summaries to identify which archived law applies; if the summaries and full traces disagree, inspect the discriminating telemetry channels before choosing.
- **Alternatives Weighed**: `Reuse the most recent solver unchanged` (rejected: The blind campaign may match an older chamber regime; archive order alone cannot establish the match.)

```bash
cd /tmp/hegel_continual_archive && diff -u alpha_m98_v2_c04_batch_02_solver.py alpha_m98_v2_c04_batch_01_solver.py; cat /tmp/hegel_continual_archive/alpha_m98_v2_c04_batch_02_solver.py; python3 - <<'PY'
import json, numpy as np, pickle
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
log=json.load(open(p)); traces=[]
for row in log:
 a=np.array(row['facility_telemetry_trace']); traces.append(a)
 print({k:v for k,v in row.items() if k!='facility_telemetry_trace'}, a.shape, 'mean', np.round(a.mean(0),4), 'std', np.round(a.std(0),4))
for i,a in enumerate(traces[:-1]):
 b=traces[-1]; print('latest vs',i,'mean distance', np.linalg.norm(a.mean(0)-b.mean(0)), 'trace RMSE', np.sqrt(np.mean((a-b)**2)) if a.shape==b.shape else None)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
for part in ('train','test'):
 z=d[part]; print(part, 'labs',np.unique(z['lab_id'],return_counts=True),'missing',np.isnan(z['coupon_features_raw']).sum(0),'telemetry',np.round(z['aux_chamber_telemetry'].mean(0),4),np.round(z['aux_chamber_telemetry'].std(0),4))
PY
```
- **Container Exit Code**: `0` (`461 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
p.r_[np.full(len(yc), sc), np.full(len(yr), sr)]
        p = least_squares(residual, p, bounds=bounds, loss='cauchy',
                          x_scale='jac', max_nfev=180).x
    perm = np.argsort(p[:2])
    return np.exp(p[:2][perm]), np.r_[p[2:4][perm], p[4:]]


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = dataset['past_dt'], dataset['eval_times']
    bridge = _bridge(dataset['calibration_load_cells'])
    x = _features(train, test)
    n = len(train['lab_id'])
    taus, theta = _fit(x[:n], *_loads(train, bridge), dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    # The blind campaign provides no chamber labels or prior lineage archive.
    curve, rate = _design(x[n:], *_loads(test, bridge), dt, times, taus)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + (curve @ theta).reshape(-1, len(times)),
            'initial_rate_pred': rate @ theta}
{'completed_batch_index': 0, 'completed_short_name': 'alpha_m98_v2_c04_s00'} (16, 8) mean [-0.0378  1.1815 -0.3903 -0.026   1.1071 -0.6551 -0.0374  2.4689] std [0.5809 0.1387 0.0906 0.6017 0.1216 0.1673 0.6758 0.0879]
{'completed_batch_index': 1, 'completed_short_name': 'alpha_m98_v2_c04_batch_01'} (16, 8) mean [ 0.0775 -1.164  -0.3893  0.0475 -1.1953 -0.4108  0.0355 -0.3529] std [0.6323 0.1291 0.0845 0.651  0.1478 0.1683 0.6245 0.076 ]
{'completed_batch_index': 2, 'completed_short_name': 'alpha_m98_v2_c04_batch_02'} (16, 8) mean [ 0.0678  1.1389  2.56    0.0634  1.1293 -0.2206  0.0521 -0.3623] std [0.6989 0.1288 0.0651 0.6037 0.1494 0.1331 0.7079 0.0453]
{'completed_batch_index': 3, 'completed_short_name': 'alpha_m98_v2_c04_batch_03'} (16, 8) mean [ 0.084  -1.1095 -0.421   0.0369 -1.153  -0.0039  0.0344 -0.3147] std [0.5643 0.1716 0.0856 0.6347 0.1676 0.1253 0.6762 0.0999]
latest vs 0 mean distance 4.307431234655198 trace RMSE 1.6634555189924396
latest vs 1 mean distance 0.41589932007760694 trace RMSE 0.5084460352748547
latest vs 2 mean distance 4.3819354128065156 trace RMSE 1.578995007339518
train labs (array([0, 1, 2, 3]), array([21, 19, 17, 15])) missing [1 3 3 3] telemetry [ 0.1223  0.3043 -0.2727  0.0506] [0.8019 0.9796 1.0443 0.9351]
test labs (array([0, 1, 2, 3]), array([12,  8,  6, 10])) missing [2 1 1 1] telemetry [-0.0453 -0.0161  0.3538 -0.1017] [1.192  1.152  1.0943 1.1244]
```

### Turn 4 — `execute_in_remote_docker` (`25501 ms` LLM)
- **Rationale**: Install the matching archived law with current-batch calibration and material fitting. The clean-process schema test detects contract errors; leave-one-lab-out residuals detect unstable material fits that would warrant changing the estimator before deployment.
- **Alternatives Weighed**: `Use the archived benchtop law for deployment` (rejected: Both the apparatus mean and full-trace distance favor batch 01, the archived non-benchtop chamber regime.)

```bash
cp /tmp/hegel_continual_archive/alpha_m98_v2_c04_batch_01_solver.py /holdout_task/batch_solver.py
sed -n '1,94p' /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; out=m.solve_batch(d); print('fit',out['estimated_taus'],out['hereditary_weights'])
cal=d['calibration_load_cells']; M=m._bridge(cal); b=cal['bridge_raw']; truth=cal['true_channel_loads'].sum(1); print('bridge shape',np.shape(M),'stress range',truth.min(),truth.max()); print('sweeps',[(k, type(v).__name__,getattr(v,'shape',None)) for k,v in cal['single_channel_sweeps'].items()]);
past,hold=m._loads(tr,M); x=m._features(tr,te)[:len(hold)]; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; labs=tr['lab_id'];
for lab in np.unique(labs):
 test=labs==lab; fit=~test; taus,th=m._fit(x[fit],past[fit],hold[fit],dt,t,y[fit],r[fit]); A,B=m._design(x[test],past[test],hold[test],dt,t,taus); ec=(A@th).reshape(y[test].shape)-y[test]; er=B@th-r[test]; print('lab',lab,'tau',taus,'curve/rate absolute median',np.median(abs(ec)),np.median(abs(er)),'curve/rate abs p90',np.quantile(abs(ec),.9),np.quantile(abs(er),.9))
print('dt',dt,'times',t)
PY
```
- **Container Exit Code**: `0` (`1693 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
.01, 0.015 * np.std(yc))
    trials = []
    for guess in ((0.18, 1.6), (0.45, 4.0), (0.9, 12.0), (0.12, 8.0)):
        q = np.log(guess)
        coeff = np.linalg.lstsq(design(q), target, rcond=None)[0]
        trials.append(least_squares(residual, np.r_[q, coeff], bounds=bounds,
                                    loss='soft_l1', x_scale='jac', max_nfev=180))
    p = min(trials, key=lambda r: r.cost).x
    for _ in range(2):
        e = design(p[:2]) @ p[2:] - target
        sc = max(_mad(e[:len(yc)]), 1e-6)
        sr = max(_mad(e[len(yc):]), 1e-6)
        # Cauchy downweighting preserves the Gaussian center of each sensor.
        scale = 2.5 * np.r_[np.full(len(yc), sc), np.full(len(yr), sr)]
        p = least_squares(residual, p, bounds=bounds, loss='cauchy',
                          x_scale='jac', max_nfev=180).x
    perm = np.argsort(p[:2])
    return np.exp(p[:2][perm]), np.r_[p[2:4][perm], p[4:]]



# Identified from chamber verification coupons, separate from the
# intrinsic material fit on benchtop observations.
_CHAMBER_TIMES = np.array([0.4718099191421826, 2.4756939307917163])
{"schema_valid": true, "errors": []}
fit [0.49708282 2.89138754] [ 1.04854436  0.08048524  0.53282288  0.16785717 -0.10006991  0.15786543
 -0.11551303]
bridge shape (5, 4) stress range -4.465616879031037 3.5772325544737096
sweeps [('lab_id', 'ndarray', (24,)), ('cmd_stress', 'ndarray', (24,)), ('true_stress', 'ndarray', (24,))]
lab 0 tau [0.4619039  1.03867259] curve/rate absolute median 0.0766654517223 0.14953225000951553 curve/rate abs p90 0.20936806370551084 0.5160257086216413
lab 1 tau [0.48811122 5.62998322] curve/rate absolute median 0.07704459706231803 0.3068945945978525 curve/rate abs p90 0.19857785705980843 0.597899333675272
lab 2 tau [0.48905616 3.36183261] curve/rate absolute median 0.07605206481977445 0.19757499961025005 curve/rate abs p90 0.21216229523512656 0.402868796431403
lab 3 tau [0.60971075 0.62232027] curve/rate absolute median 0.07222237954845862 0.08791592039169505 curve/rate abs p90 0.19931448324721296 0.2649238312870635
dt [0.35 0.35 0.4  0.45 0.5  0.55] times [0.25       0.59090909 0.93181818 1.27272727 1.61363636 1.95454545
 2.29545455 2.63636364 2.97727273 3.31818182 3.65909091 4.        ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c23d2298437ace73 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c23d2298 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`28346 ms` LLM)
- **Rationale**: Inspect bridge residuals, sweep behavior, and signed residuals by lab and time. Structured calibration error would motivate changing the bridge fit; systematic sensor residuals would motivate testing a different weighting scheme. If neither appears, avoid adding nuisance terms.
- **Alternatives Weighed**: `Submit the schema-valid solver now` (rejected: The second relaxation time ranges from 0.62 to 5.63 across held-out labs; calibration residuals and signed sensor residuals can distinguish a correctable bias from weak identification.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p='/holdout_task/batch_solver.py'; s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; cal=d['calibration_load_cells']; a=np.c_[np.ones(len(cal['bridge_raw'])),cal['bridge_raw']]; y=cal['true_channel_loads']; M=m._bridge(cal); e=a@M-y
print('bridge condition',np.linalg.cond(a),'raw min/max',a[:,1:].min(0),a[:,1:].max(0)); print('bridge residual quantiles per channel',np.quantile(abs(e),[.5,.9,1],axis=0)); print('total residual quantiles',np.quantile(abs(e.sum(1)),[.5,.9,1])); print('bridge matrix',M)
for j in np.unique(cal['single_channel_sweeps']['lab_id']):
 q=cal['single_channel_sweeps']; idx=q['lab_id']==j; print('sweep lab',j,np.c_[q['cmd_stress'][idx],q['true_stress'][idx]].round(4).tolist())
for k,b in [('train',tr),('test',te)]:
 r=np.concatenate((b['hold_channel_raw'][:,None,:],b['past_channel_raw']),axis=1); print(k,'raw min/max',r.min((0,1)),r.max((0,1)))
x=m._features(tr,te)[:len(tr['lab_id'])]; past,hold=m._loads(tr,M); tau,th=m._fit(x,past,hold,d['past_dt'],d['eval_times'],tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); A,B=m._design(x,past,hold,d['past_dt'],d['eval_times'],tau); e=(A@th).reshape(len(x),-1)+tr['strain_0'][:,None]-tr['strain_curves_obs']; r=B@th-tr['instant_rate_obs']; print('rate quantiles',np.quantile(r,[0,.1,.25,.5,.75,.9,1])); print('curve median by time',np.median(e,axis=0).round(4));
for lab in np.unique(tr['lab_id']):
 i=tr['lab_id']==lab; print('lab',lab,'curve/rate signed median',np.median(e[i]),np.median(r[i]),'rate MAD',m._mad(r[i]))
F=np.c_[hold,past,x,tr['aux_chamber_telemetry']]; print('correlation with clipped rate residual',np.corrcoef(np.c_[np.clip(r,-.6,.6),F].T)[0,1:].round(3)); print('correlation with clipped curve residual',np.corrcoef(np.c_[np.mean(np.clip(e,-.3,.3),axis=1),F].T)[0,1:].round(3))
PY
```
- **Container Exit Code**: `0` (`746 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ridge condition 6.862556512468368 raw min/max [ -2.00324069 -13.06998886  -2.9385744  -14.72486092] [14.59664506  3.83832287 13.2094582   2.25201036]
bridge residual quantiles per channel [[0.43641882 0.26864167 0.18717083 0.37314481]
 [0.89212447 0.65328085 0.39582733 0.86112163]
 [1.69671939 0.83787001 0.98653371 1.20549648]]
total residual quantiles [0.19438033 0.46902963 1.1470284 ]
bridge matrix [[ 0.09025386 -0.14092762 -0.05322772  0.09035839]
 [ 0.57820493  0.08949858 -0.08859829  0.43593959]
 [ 0.23674977  0.50253143 -0.02827103 -0.06080406]
 [-0.09763962  0.18415286  0.48226467  0.06530449]
 [ 0.25835904 -0.19783276  0.36116781  0.59202488]]
sweep lab 0 [[-1.817, -1.5], [-0.8326, -0.8], [-0.2734, -0.2], [0.2234, 0.4], [0.8458, 1.0], [17.2831, 1.6]]
sweep lab 1 [[-2.5256, -1.5], [-0.9941, -0.8], [-0.1238, -0.2], [0.6485, 0.4], [1.6169, 1.0], [3.0751, 1.6]]
sweep lab 2 [[-2.6329, -1.5], [-1.0166, -0.8], [-0.0979, -0.2], [0.7173, 0.4], [1.7394, 1.0], [3.2784, 1.6]]
sweep lab 3 [[-1.785, -1.5], [-0.8248, -0.8], [-0.2791, -0.2], [0.205, 0.4], [0.8122, 1.0], [1.7265, 1.6]]
train raw min/max [-0.80595627 -0.98336525 -0.97512849 -0.81234839] [1.27376345 2.30080027 2.48090161 1.26629671]
test raw min/max [-0.77414181 -0.95110932 -0.93869555 -0.81258947] [1.27620218 2.19657886 2.56321518 1.33916081]
rate quantiles [-0.75142208 -0.38190083 -0.22740136 -0.01677886  0.10089842  0.22218145
  0.48378168]
curve median by time [ 0.0011 -0.0112 -0.0011 -0.0093 -0.0124 -0.0202 -0.016  -0.0111 -0.0077
  0.0041  0.0169  0.0359]
lab 0 curve/rate signed median -0.013240995743784456 -0.047431236088771644 rate MAD 0.1758421219619909
lab 1 curve/rate signed median -0.01165271042331284 0.01772386986631158 rate MAD 0.39490841763738915
lab 2 curve/rate signed median -0.012601715671691105 -0.03644179123788627 rate MAD 0.2783598046666725
lab 3 curve/rate signed median 0.03135681038762389 0.04036232862277056 rate MAD 0.09012728619377958
correlation with clipped rate residual [ 0.594 -0.051 -0.15   0.001 -0.182 -0.053 -0.274 -0.018 -0.02   0.046
  0.111 -0.001  0.001  0.148 -0.219]
correlation with clipped curve residual [ 0.157  0.016 -0.099 -0.    -0.24  -0.086 -0.067 -0.     0.012 -0.003
  0.016 -0.036 -0.012  0.067 -0.27 ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c23d2298437ace73 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c23d2298 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`26720 ms` LLM)
- **Rationale**: Calibration-only folds distinguish a reproducible nonlinear bridge from a few bad sweeps. If a compact forward map generalizes better, invert it and check whether it also stabilizes benchtop fits across labs.
- **Alternatives Weighed**: `Add a lab-specific rate correction` (rejected: It would absorb stress-correlated error downstream without addressing the independently measured bridge mismatch.)
- **ATMS NOGOOD DECLARED**: Refuted `A single affine raw-to-load map is adequate for this bridge across its calibration range.` | Witness: `The well-conditioned affine fit leaves p90 total-stress error 0.469 and maximum error 1.147; its inferred hold load correlates 0.594 with clipped benchtop rate residuals.` | Blanket Diagnosis: `The bridge is upstream of both the hereditary excitation and initial-rate predictions; stress-dependent calibration error can masquerade as a change in material relaxation.` | Sublated Pivot: `Fit and cross-validate nonlinear bridge terms using only the paired empirical calibration, then retest the unchanged material law.`

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS calibration'; OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.stats import skew
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; x=C['true_channel_loads']; y=C['bridge_raw']; print('calibration rows',len(x),'true range',x.min(0),x.max(0),'std',x.std(0),'skew',skew(x)); print(np.c_[x,y][:8].round(4)); print('quantiles true',np.quantile(x,[0,.1,.5,.9,1],axis=0).round(3))
def basis(x,d,kind):
 cols=[np.ones(len(x))]; names=[()];
 for k in range(1,d+1):
  for ix in combinations_with_replacement(range(4),k):
   if kind=='diag' and len(set(ix))>1: continue
   cols.append(np.prod(x[:,ix],axis=1)); names.append(ix)
 return np.array(cols).T,names
for u,v in [(x,y),(y,x)]:
 print('direction', 'forward' if u is x else 'inverse'); f=np.arange(len(u))%5
 for j,kind in [(1,'all'),(2,'diag'),(2,'all'),(3,'diag'),(3,'all'),(4,'all')]:
  a,n=basis(u,j,kind); e=np.zeros_like(v)
  for i in range(5):
   b=f!=i; e[~b]=a[~b]@np.linalg.lstsq(a[b],v[b],rcond=None)[0]-v[~b]
  print(j,kind,'p50/p90',np.quantile(abs(e),[.5,.9]).round(5),'channel med',np.median(abs(e),0).round(5),'sum p90',np.quantile(abs(e.sum(1)),.9).round(5));
  if j==3 and kind=='all':
   coef=np.linalg.lstsq(a,v,rcond=None)[0]; print('terms',[(k,c.round(4).tolist()) for k,c in zip(n,coef) if np.max(abs(c))>.015])
PY
```
- **Container Exit Code**: `0` (`691 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, 0.2302]), ((0, 2, 3), [0.0453, -0.0413, 0.04, -0.044]), ((0, 3, 3), [-0.1863, 0.1696, -0.1645, 0.1812]), ((1, 1, 1), [-0.0616, 0.2829, -0.0544, 0.0599]), ((1, 1, 2), [0.5008, -0.4559, 0.4421, -0.487]), ((1, 1, 3), [-0.6622, 0.6028, -0.5846, 0.6439]), ((1, 2, 2), [-0.1721, 0.1567, -0.1518, 0.1674]), ((1, 2, 3), [-0.2671, 0.2431, -0.2358, 0.2597]), ((1, 3, 3), [0.4197, -0.3821, 0.3705, -0.4081]), ((2, 2, 2), [0.1667, -0.1517, 0.3866, -0.162]), ((2, 2, 3), [-0.7729, 0.7036, -0.6823, 0.7516]), ((2, 3, 3), [-0.2284, 0.208, -0.2017, 0.2222]), ((3, 3, 3), [1.127, -1.0259, 0.9949, -0.9537])]
4 all p50/p90 [ 36.06766 144.00649] channel med [39.21225 35.72854 34.64932 38.15131] sum p90 0.31534
direction inverse
1 all p50/p90 [0.34417 0.84553] channel med [0.46797 0.28054 0.1936  0.42812] sum p90 0.50027
2 diag p50/p90 [0.10179 0.28387] channel med [0.10846 0.097   0.08753 0.11609] sum p90 0.55339
2 all p50/p90 [0.08807 0.20635] channel med [0.07813 0.09568 0.07467 0.09867] sum p90 0.5539
3 diag p50/p90 [0.08915 0.2829 ] channel med [0.09394 0.07893 0.07458 0.11552] sum p90 0.51198
3 all p50/p90 [0.04834 0.24635] channel med [0.06585 0.03773 0.04499 0.05137] sum p90 0.8383
terms [((), [0.0591, -0.1153, -0.061, 0.1619]), ((0,), [0.9845, -0.0978, 0.0702, 0.2124]), ((1,), [0.1874, 0.6767, -0.0888, 0.0115]), ((2,), [0.0323, 0.167, 0.6627, -0.0533]), ((3,), [-0.1225, 0.0732, 0.2852, 1.0232]), ((0, 0), [-0.0222, 0.0016, 0.0112, 0.0263]), ((0, 1), [0.0696, -0.0015, 0.0093, 0.0435]), ((0, 2), [0.031, -0.0083, 0.0293, 0.0244]), ((0, 3), [0.0049, -0.0028, 0.0465, 0.1632]), ((1, 2), [0.0186, 0.0541, -0.0019, -0.0113]), ((1, 3), [0.0129, 0.0142, -0.0134, 0.0518]), ((2, 3), [0.0131, 0.0353, 0.0807, 0.0251]), ((3, 3), [0.0194, -0.0103, 0.0262, -0.0123]), ((0, 0, 0), [-0.0506, -0.0016, 0.0086, 0.0088]), ((0, 0, 1), [-0.0302, 0.002, -0.0082, -0.0048]), ((0, 0, 3), [0.0072, -0.003, 0.0183, 0.0186]), ((0, 1, 3), [0.0145, -0.0107, 0.0197, 0.0469]), ((0, 2, 3), [0.0081, -0.0105, 0.0211, 0.0302]), ((1, 1, 1), [-0.0029, -0.0157, 0.0025, 0.0039]), ((2, 2, 3), [0.0019, 0.0113, -0.0241, 0.0054]), ((3, 3, 3), [0.0157, -0.0097, -0.0136, -0.0254])]
4 all p50/p90 [0.14695 1.3594 ] channel med [0.14976 0.08292 0.13711 0.31866] sum p90 3.55507

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c23d2298437ace73 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c23d2298 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`23390 ms` LLM)
- **Rationale**: Test whether the unexplained calibration signal lies in one direction and inspect which cubic terms survive projection. If it does, recover a compact bridge from those terms and use held-out coupon responses to test which inversion transfers to specimen data.
- **Alternatives Weighed**: `Invert the unconstrained cubic fit` (rejected: Its large individual-channel errors nearly cancel in the sum, so its four-channel inverse would use an unstable direction.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle,numpy as np
from itertools import combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; x=C['true_channel_loads']; y=C['bridge_raw']; ix=[()] + [j for n in range(1,4) for j in combinations_with_replacement(range(4),n)]; A=np.array([np.prod(x[:,j],axis=1) for j in ix]).T; B=np.linalg.lstsq(A,y,rcond=None)[0]; E=y-A@B; U,S,V=np.linalg.svd(E,full_matrices=False); v=V[0]; P=np.eye(4)-np.outer(v,v); print('singular values',S,'nuisance',v,'sum',v.sum()); G=B@P
for j,g in zip(ix,G):
 if np.max(abs(g))>.002: print(j,np.round(g,5))
print('paired rows: index, nonzero inputs, projected residual size, nuisance coordinate'); print(np.c_[np.arange(len(x)),np.count_nonzero(x,axis=1),np.linalg.norm(E@P,axis=1),y@v].round(4).tolist()); print('coupon nuisance percentiles');
for n in ('train','test'):
 for k in ('past_channel_raw','hold_channel_raw'): print(n,k,np.quantile(D[n][k]@v,[0,.1,.5,.9,1]).round(4))
PY
```
- **Container Exit Code**: `0` (`451 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
001, -27.6881], [14.0, 1.0, 0.0001, -26.9581], [15.0, 1.0, 0.0001, -26.093], [16.0, 4.0, 0.0001, -0.1601], [17.0, 4.0, 0.0001, 2.1787], [18.0, 4.0, 0.0001, 0.9383], [19.0, 4.0, 0.0002, 0.5036], [20.0, 4.0, 0.0001, -0.7795], [21.0, 4.0, 0.0001, 0.6412], [22.0, 4.0, 0.0001, 0.0032], [23.0, 4.0, 0.0001, -0.4685], [24.0, 4.0, 0.0002, 1.7563], [25.0, 4.0, 0.0001, 0.2948], [26.0, 4.0, 0.0001, -1.088], [27.0, 4.0, 0.0001, 3.2267], [28.0, 4.0, 0.0001, 1.1553], [29.0, 4.0, 0.0, 2.651], [30.0, 4.0, 0.0002, -0.0946], [31.0, 4.0, 0.0, -0.4281], [32.0, 4.0, 0.0001, -1.9324], [33.0, 4.0, 0.0001, -0.7753], [34.0, 4.0, 0.0002, 1.2541], [35.0, 4.0, 0.0001, -0.0821], [36.0, 4.0, 0.0001, -2.0217], [37.0, 4.0, 0.0001, -1.2492], [38.0, 4.0, 0.0001, -1.9383], [39.0, 4.0, 0.0001, -0.9676], [40.0, 4.0, 0.0001, -0.7923], [41.0, 4.0, 0.0002, 1.4127], [42.0, 4.0, 0.0001, 2.0207], [43.0, 4.0, 0.0001, 0.0856], [44.0, 4.0, 0.0001, 2.1147], [45.0, 4.0, 0.0001, -2.5743], [46.0, 4.0, 0.0001, 2.1612], [47.0, 4.0, 0.0002, -0.3648], [48.0, 4.0, 0.0001, 0.0939], [49.0, 4.0, 0.0002, 0.5829], [50.0, 4.0, 0.0001, -0.3697], [51.0, 4.0, 0.0001, -0.2877], [52.0, 4.0, 0.0002, -1.4388], [53.0, 4.0, 0.0001, -2.7033], [54.0, 4.0, 0.0001, 1.2253], [55.0, 4.0, 0.0001, -0.0529], [56.0, 4.0, 0.0002, -1.4171], [57.0, 4.0, 0.0001, 0.446], [58.0, 4.0, 0.0001, -0.8322], [59.0, 4.0, 0.0001, -0.9128], [60.0, 4.0, 0.0001, 0.8196], [61.0, 4.0, 0.0001, -2.5739], [62.0, 4.0, 0.0001, -0.068], [63.0, 4.0, 0.0002, 1.4476], [64.0, 4.0, 0.0001, -1.2394], [65.0, 4.0, 0.0003, 0.9093], [66.0, 4.0, 0.0001, -1.8161], [67.0, 4.0, 0.0001, 1.4056], [68.0, 4.0, 0.0001, 1.632], [69.0, 4.0, 0.0001, -0.1068], [70.0, 4.0, 0.0002, 0.7143], [71.0, 4.0, 0.0001, -1.3579], [72.0, 4.0, 0.0001, -1.3577], [73.0, 4.0, 0.0001, -0.7875], [74.0, 4.0, 0.0001, -1.8295], [75.0, 4.0, 0.0001, -1.8708], [76.0, 4.0, 0.0001, -0.9861], [77.0, 4.0, 0.0002, -1.5754], [78.0, 4.0, 0.0001, 0.5453], [79.0, 4.0, 0.0001, 0.3966]]
coupon nuisance percentiles
train past_channel_raw [-2.1735 -0.9921 -0.0078  0.9602  2.0336]
train hold_channel_raw [-0.9338 -0.5835  0.0972  0.5647  1.0109]
test past_channel_raw [-1.8199 -0.9611 -0.0321  0.8398  1.7927]
test hold_channel_raw [-0.67   -0.495  -0.1357  0.44    1.0591]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c23d2298437ace73 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c23d2298 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`40917 ms` LLM)
- **Rationale**: The projected coefficients identify four cyclic pair products and four diagonal cubics. Test whether that compact map predicts held-out sweeps, reconstructs coupon readings, and improves the unchanged material fit across labs before replacing the affine bridge.
- **Alternatives Weighed**: `Discard the contaminated direction at prediction time` (rejected: Projection leaves three measurements for four unknown loads; refitting clean sweeps can recover the fourth direction.)

```bash
cat > /tmp/bridge_trial.py <<'PY'
import importlib.util,pickle,numpy as np
from itertools import combinations_with_replacement
p='/holdout_task/batch_solver.py'; s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; dt,t=d['past_dt'],d['eval_times']
full=[()] + [j for n in range(1,4) for j in combinations_with_replacement(range(4),n)]
aidx=np.arange(4); prev=(aidx-1)%4

def basis(z):
 return np.c_[np.ones(len(z)),z,z**3,z*z[:,prev]]

def fit(xx,yy):
 a=np.array([np.prod(xx[:,j],axis=1) for j in full]).T
 e=yy-a@np.linalg.lstsq(a,yy,rcond=None)[0]; _,_,v=np.linalg.svd(e,full_matrices=False); n=v[0]; proj=np.eye(4)-np.outer(n,n)
 b=basis(xx); delta=yy-b@np.linalg.lstsq(b,yy@proj,rcond=None)[0]; score=delta@n
 # Seed the physical fit with the compact nonlinear model on the majority of rows.
 keep=np.ones(len(xx),bool)
 for _ in range(5):
  k=np.linalg.lstsq(b[keep],score[keep],rcond=None)[0]; r=score-b@k
  idx=np.argsort(abs(r))[:int(.9*len(r))]; keep[:]=False; keep[idx]=True
 k=np.linalg.lstsq(b[keep],yy[keep],rcond=None)[0]; e=yy-b@k
 keep=np.linalg.norm(e,axis=1)<max(.002,8*np.median(np.linalg.norm(e,axis=1)))
 return np.linalg.lstsq(b[keep],yy[keep],rcond=None)[0],keep

def invert(raw,k):
 shape=raw.shape; y=raw.reshape(-1,4); z=np.linalg.solve(k[1:5].T,(y-k[0]).T).T
 for _ in range(20):
  r=basis(z)@k-y
  J=np.broadcast_to(k[1:5].T,(len(z),4,4)).copy()
  for j in range(4):
   J[:,:,j]+=3*z[:,j,None]**2*k[5+j]+z[:,prev[j],None]*k[9+j]
   J[:,:,prev[j]]+=z[:,j,None]*k[9+j]
  step=np.linalg.solve(J,r[...,None])[...,0]; z-=step/np.maximum(1,np.max(abs(step),axis=1,keepdims=True))
  if np.max(abs(r))<1e-10: break
 return z.reshape(shape),np.linalg.norm(basis(z)@k-y,axis=1)

if __name__=='__main__':
 k,good=fit(x,y); print('excluded',np.flatnonzero(~good),'coefficients',k.round(5)); print('clean forward p90',np.quantile(abs(basis(x[good])@k-y[good]),.9));
 for split in (np.arange(len(x))%5, np.digitize(np.linalg.norm(x,axis=1),np.quantile(np.linalg.norm(x,axis=1),[.2,.4,.6,.8]))):
  for g in np.unique(split):
   i=split==g; kf,ok=fit(x[~i],y[~i]); z,_=invert(y[i & good],kf); e=z-x[i & good]; print('cal fold',g,'clean load/sum max',np.max(abs(e)),np.max(abs(e.sum(1))),'retained',ok.sum())
 loads=[]
 for b in (tr,te):
  for key in ('past_channel_raw','hold_channel_raw'):
   z,r=invert(b[key],k); print(key,'max reconstruction',max(r),'load range',z.min(),z.max()); loads.append(z.sum(-1))
 past,hold=loads[:2]; xx=m._features(tr,te)[:len(hold)]; yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; labs=tr['lab_id'];
 for lab in [*np.unique(labs),None]:
  i=labs==lab if lab is not None else np.ones(len(labs),bool); j=~i if lab is not None else i; tau,th=m._fit(xx[j],past[j],hold[j],dt,t,yc[j],yr[j]); A,B=m._design(xx[i],past[i],hold[i],dt,t,tau); print('lab',lab,'tau',tau,'theta',th.round(4),'curve/rate p90',np.quantile(abs(A@th-yc[i].ravel()),.9),np.quantile(abs(B@th-yr[i]),.9))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/bridge_trial.py
```
- **Container Exit Code**: `0` (`1042 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
00000e+00  1.00000e-05]
 [-1.00000e-05 -2.77200e-01 -3.00000e-05 -0.00000e+00]
 [ 1.00000e-05  2.00000e-05 -2.92600e-01  0.00000e+00]]
clean forward p90 0.00014640305001348283
cal fold 0 clean load/sum max 0.00020825478265373842 0.000455040538878404 retained 62
cal fold 1 clean load/sum max 0.0002542569844163688 0.0004821377246079783 retained 61
cal fold 2 clean load/sum max 0.0002547828045817435 0.0002956415460572026 retained 61
cal fold 3 clean load/sum max 0.00019768985168622244 0.0003712683401368988 retained 62
cal fold 4 clean load/sum max 0.00015893515142706427 0.0003158047167280964 retained 62
cal fold 0 clean load/sum max 0.0002567654851202703 0.00038014819498169704 retained 63
cal fold 1 clean load/sum max 0.0002085104371035376 0.0004720991516660722 retained 61
cal fold 2 clean load/sum max 0.00022145757682789924 0.0003093835423736424 retained 62
cal fold 3 clean load/sum max 0.00023921188039877084 0.0005734199068119783 retained 61
cal fold 4 clean load/sum max 0.00020140157663806058 0.00036385848320452663 retained 61
past_channel_raw max reconstruction 5.003707553108401e-16 load range -0.6498779407957269 1.1489350949587924
hold_channel_raw max reconstruction 7.065416064076988e-16 load range 0.3525960649183144 1.347352387720859
past_channel_raw max reconstruction 5.087681048627601e-16 load range -0.6465367171789886 1.1489928283929538
hold_channel_raw max reconstruction 7.021666937153402e-16 load range 0.3528749478151803 1.3496253136384322
lab 0 tau [0.47721566 4.47541829] theta [ 0.6353  0.5617  0.3315  0.1681 -0.0963  0.1582 -0.1106] curve/rate p90 0.02313253826607975 0.015807196135736667
lab 1 tau [0.47599955 4.41459215] theta [ 0.6327  0.5624  0.3308  0.1681 -0.0965  0.1577 -0.1105] curve/rate p90 0.02238522614243293 0.009546102221695918
lab 2 tau [0.47559263 4.33475007] theta [ 0.6316  0.5581  0.3313  0.1675 -0.0967  0.158  -0.1103] curve/rate p90 0.022320785767493592 0.01219901413459672
lab 3 tau [0.47507187 4.27668801] theta [ 0.6303  0.5549  0.3316  0.168  -0.0964  0.1578 -0.1104] curve/rate p90 0.02020448364855731 0.0100799898913964
lab None tau [0.47590431 4.37072905] theta [ 0.6323  0.5594  0.3312  0.168  -0.0964  0.1579 -0.1104] curve/rate p90 0.021325142104055252 0.013018921373828318

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c23d2298437ace73 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c23d2298 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`25996 ms` LLM)
- **Rationale**: Promote the independently validated bridge and test the exact integrated artifact. The clean-process check detects integration or output failures; inspect the retained chamber equation to confirm that only the bridge changed.
- **Alternatives Weighed**: `Retain affine calibration` (rejected: It loses on both clean calibration holdouts and both held-out benchtop outputs.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p=Path('/holdout_task/batch_solver.py')
s=p.read_text(); trial=Path('/tmp/bridge_trial.py').read_text()
a=trial.index('full='); b=trial.index("if __name__")
bridge=trial[a:b].replace('def basis(', 'def _bridge_basis(').replace('basis(', '_bridge_basis(').replace('def _bridge__bridge_basis(', 'def _bridge_basis(')
bridge=bridge.replace('def fit(xx,yy):','def _fit_bridge(xx, yy):').replace('def invert(raw,k):','def _invert_bridge(raw, k):')
bridge += "def _bridge(cal):\n    return _fit_bridge(cal['true_channel_loads'], cal['bridge_raw'])[0]\n\n\ndef _loads(part, bridge):\n    past = _invert_bridge(part['past_channel_raw'], bridge)[0].sum(axis=-1)\n    hold = _invert_bridge(part['hold_channel_raw'], bridge)[0].sum(axis=-1)\n    return past, hold\n\n\n"
start=s.index('def _bridge('); end=s.index('\ndef ',s.index('def _loads(',start)+1)
s=s[:start]+bridge+s[end+1:]; s=s.replace('import numpy as np','from itertools import combinations_with_replacement\n\nimport numpy as np',1); p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
tail -47 /holdout_task/batch_solver.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; out=m.solve_batch(d)
for k,v in out.items():
 assert np.isfinite(v).all(); print(k,v.shape, np.min(v),np.max(v))
print(out['estimated_taus'],out['hereditary_weights'])
assert 0.05 < out['estimated_taus'][0] < out['estimated_taus'][1] < 25
PY
```
- **Container Exit Code**: `0` (`1030 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
ials, key=lambda r: r.cost).x
    for _ in range(2):
        e = design(p[:2]) @ p[2:] - target
        sc = max(_mad(e[:len(yc)]), 1e-6)
        sr = max(_mad(e[len(yc):]), 1e-6)
        # Cauchy downweighting preserves the Gaussian center of each sensor.
        scale = 2.5 * np.r_[np.full(len(yc), sc), np.full(len(yr), sr)]
        p = least_squares(residual, p, bounds=bounds, loss='cauchy',
                          x_scale='jac', max_nfev=180).x
    perm = np.argsort(p[:2])
    return np.exp(p[:2][perm]), np.r_[p[2:4][perm], p[4:]]



# Identified from chamber verification coupons, separate from the
# intrinsic material fit on benchtop observations.
_CHAMBER_TIMES = np.array([0.4718099191421826, 2.4756939307917163])
_CHAMBER_COEFFICIENTS = np.array([[0.6193595560932273, 0.16764713872971412], [0.28119327738345373, 1.3920963008647889], [0.03080586637439311, 0.08122038107343733]])


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = dataset['past_dt'], dataset['eval_times']
    bridge = _bridge(dataset['calibration_load_cells'])
    x = _features(train, test)
    n = len(train['lab_id'])
    taus, theta = _fit(x[:n], *_loads(train, bridge), dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    _, hold = _loads(test, bridge)
    drift = theta[2] + x[n:] @ theta[3:]
    components = np.column_stack((hold, drift)) @ _CHAMBER_COEFFICIENTS.T
    creep = -np.expm1(-times[:, None] / _CHAMBER_TIMES)
    strain = (test['strain_0'][:, None] + components[:, :2] @ creep.T
              + components[:, 2, None] * times)
    rate = components[:, :2] @ (1.0 / _CHAMBER_TIMES) + components[:, 2]
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': strain, 'initial_rate_pred': rate}
estimated_taus (2,) 0.47590430603286343 4.3707290535836245
hereditary_weights (7,) -0.11044075884192416 0.6323357179431066
predicted_strain (36, 12) 0.3766174029939044 5.4292891166995
initial_rate_pred (36,) 2.6472534462561086 7.329870115973162
[0.47590431 4.37072905] [ 0.63233572  0.55936883  0.33124259  0.1679731  -0.09644894  0.15789559
 -0.11044076]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c5d7d044affb16d1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c5d7d044 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`33132 ms` LLM)
- **Rationale**: The exact artifact has passed schema validation. Run the scientific holdouts against that artifact to catch integration regressions, and check that deployment strain and initial rate agree at the chamber boundary. A holdout failure would reopen calibration or material fitting; a boundary failure would require correcting the chamber output equation.

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; c=d['calibration_load_cells']; x,y=c['true_channel_loads'],c['bridge_raw']; k,clean=m._fit_bridge(x,y)
worst=0.
for groups in (np.arange(len(x))%5, np.digitize(np.linalg.norm(x,axis=1),np.quantile(np.linalg.norm(x,axis=1),[.2,.4,.6,.8]))):
 for g in np.unique(groups):
  j=groups!=g; kf,_=m._fit_bridge(x[j],y[j]); i=~j & clean; z,_=m._invert_bridge(y[i],kf); worst=max(worst,np.max(abs((z-x[i]).sum(1))))
print('worst held-out clean calibration stress error',worst,'excluded',np.flatnonzero(~clean)); assert worst<.001
X=m._features(tr,te); n=len(tr['lab_id']); xp,xt=X[:n],X[n:]; p,h=m._loads(tr,k); _,ht=m._loads(te,k); dt,t=d['past_dt'],d['eval_times']; yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; out=m.solve_batch(d); wc=wr=0.
for lab in np.unique(tr['lab_id']):
 i=tr['lab_id']==lab; j=~i; tau,th=m._fit(xp[j],p[j],h[j],dt,t,yc[j],yr[j]); A,B=m._design(xp[i],p[i],h[i],dt,t,tau); ec=np.quantile(abs(A@th-yc[i].ravel()),.9); er=np.quantile(abs(B@th-yr[i]),.9); wc=max(wc,ec); wr=max(wr,er)
 comp=np.c_[ht,th[2]+xt@th[3:]]@m._CHAMBER_COEFFICIENTS.T; strain=te['strain_0'][:,None]+comp[:,:2]@(-np.expm1(-t[:,None]/m._CHAMBER_TIMES)).T+comp[:,2,None]*t; rate=comp[:,:2]@(1/m._CHAMBER_TIMES)+comp[:,2]; print('held-out lab',lab,'p90 strain/rate',ec,er,'max deployment changes',np.max(abs(strain-out['predicted_strain'])),np.max(abs(rate-out['initial_rate_pred'])))
assert wc<.03 and wr<.025
# Exercise the actual entrypoint at t=0 without changing the benchtop fit grid.
fit=m._fit
m._fit=lambda xp,p,h,dt,unused,y,r: fit(xp,p,h,dt,t,y,r)
eps=1e-6; b=m.solve_batch({**d,'eval_times':np.array([0.,eps])}); np.testing.assert_allclose(b['predicted_strain'][:,0],te['strain_0'],rtol=0,atol=1e-12); np.testing.assert_allclose((b['predicted_strain'][:,1]-b['predicted_strain'][:,0])/eps,b['initial_rate_pred'],rtol=2e-6,atol=1e-8); print('chamber boundary and initial derivative agree')
PY
```
- **Container Exit Code**: `0` (`1376 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
worst held-out clean calibration stress error 0.0005734199068119783 excluded [13 14 15]
held-out lab 0 p90 strain/rate 0.02313253826607975 0.015807196135736667 max deployment changes 0.002180644291025935 0.001354642158769792
held-out lab 1 p90 strain/rate 0.02238522614243293 0.009546102221695918 max deployment changes 0.0014416831327519475 0.0008955907019094766
held-out lab 2 p90 strain/rate 0.022320785767493592 0.01219901413459672 max deployment changes 0.003096944032477733 0.0019238584518390667
held-out lab 3 p90 strain/rate 0.02020448364855731 0.0100799898913964 max deployment changes 0.000927502241616418 0.0005761754193556179
chamber boundary and initial derivative agree

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c5d7d044affb16d1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c5d7d044 (directory).
```

### Turn 11 — `submit_completed_task` (`16893 ms` LLM)
- **Rationale**: The exact submitted artifact passed isolated schema validation, calibration and benchtop holdout checks, and the chamber boundary and initial-derivative checks. Public data support the bridge correction but do not independently validate the blind chamber predictions.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:6361'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema check passed. Worst leave-one-lab-out 90th-percentile absolute errors were 0.0231 for strain and 0.0158 for initial rate. The blind deployment law was selected from the apparatus log; deployment accuracy could not be measured on the public batch.`

### Turn 12 — `execute_in_remote_docker` (`35115 ms` LLM)
- **Rationale**: Challenge the cyclic bridge with omitted interactions and rational pair terms on held-out sweeps. Promote a challenger only if it materially improves calibration and transfers to coupon predictions. Dropping each apparatus channel tests whether the chamber match depends on one channel; a change in the nearest regime would reopen that choice.
- **Alternatives Weighed**: `Run relay-mode, classification, or Fokker–Planck checks` (rejected: This task has continuous rheology measurements, no relay states or classification labels, and an explicit hereditary constitutive law.)

```bash
set -e
cat >/tmp/audit_batch04.py <<'PY'
import importlib.util, json, pickle
import numpy as np

s = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
c = d['calibration_load_cells']
x, y = c['true_channel_loads'], c['bridge_raw']
# Challenge the cyclic forward law with omitted quadratic/cubic interactions
# and bounded pairwise terms. Freeze all choices before evaluating each fold.
pairs = [(i,j) for i in range(4) for j in range(i+1,4)]
triples = [(i,j,k) for i in range(4) for j in range(i+1,4) for k in range(j+1,4)]
prev = (np.arange(4)-1)%4

def terms(z, mode):
    base = m._bridge_basis(z)
    if mode == 'cyclic':
        return base
    if mode == 'cubic':
        return np.column_stack([base, z[:,0]*z[:,2], z[:,1]*z[:,3],
            *[z[:,i]**2*z[:,j] for i in range(4) for j in range(4) if i != j],
            *[z[:,i]*z[:,j]*z[:,k] for i,j,k in triples]])
    return np.column_stack([base, *[(z[:,i]*z[:,j])/(1+a*(z[:,i]+sgn*z[:,j])**2)
        for i,j in pairs for sgn in (-1,1) for a in (0.5,)]])

results = {mode:[[],[]] for mode in ('cyclic','cubic','rational')}
for grouping in (np.arange(len(x))%5, np.argsort(np.argsort(np.linalg.norm(x,axis=1)))//16):
    for g in np.unique(grouping):
        train = grouping != g
        k, clean = m._fit_bridge(x[train], y[train])
        # Separate sum and decontaminated channel scores: the sum uses every held-out
        # observation, while channel scores exclude large within-row discrepancies.
        e0 = y[~train] - terms(x[~train], 'cyclic')@k
        clean_test = np.linalg.norm(e0,axis=1) < .002
        for mode,(sums,channels) in results.items():
            a = terms(x[train][clean],mode)
            coef = np.linalg.lstsq(a,y[train][clean],rcond=None)[0]
            e = terms(x[~train],mode)@coef-y[~train]
            sums.extend(abs(e.sum(1)))
            channels.extend(np.max(abs(e[clean_test]),axis=1))
for mode,(sums,channels) in results.items():
    print(mode, 'all-row raw-sum p90/max',np.quantile(sums,.9),max(sums), 'clean-row channel max',max(channels))

log = json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
traces = np.asarray([row['facility_telemetry_trace'] for row in log]); means = traces.mean(1)
for omit in range(traces.shape[2]):
    distance = np.linalg.norm(np.delete(means[:-1]-means[-1],omit,axis=1),axis=1)
    j = np.argsort(distance)
    print('omit apparatus channel',omit,'nearest archive row',j[0],'runner-up gap',distance[j[1]]-distance[j[0]])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_batch04.py
```
- **Container Exit Code**: `0` (`564 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cyclic all-row raw-sum p90/max 0.00033760835022924733 0.000601707987425204 clean-row channel max 0.0003310671215081551
cubic all-row raw-sum p90/max 0.00044741143014220266 0.0013827032631018588 clean-row channel max 0.0012531251851837233
rational all-row raw-sum p90/max 0.0004212224234374215 0.0007267777823211918 clean-row channel max 0.0005221713243233062
omit apparatus channel 0 nearest archive row 1 runner-up gap 3.8898606501055006
omit apparatus channel 1 nearest archive row 1 runner-up gap 3.235373152486237
omit apparatus channel 2 nearest archive row 1 runner-up gap 2.7969379149680176
omit apparatus channel 3 nearest archive row 1 runner-up gap 3.8912086577328164
omit apparatus channel 4 nearest archive row 1 runner-up gap 3.2531068852836644
omit apparatus channel 5 nearest archive row 1 runner-up gap 4.171954535181397
omit apparatus channel 6 nearest archive row 1 runner-up gap 3.8909362272290346
omit apparatus channel 7 nearest archive row 1 runner-up gap 2.8730344843920066

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c5d7d044affb16d1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c5d7d044 (directory).
```

### Turn 13 — `submit_completed_task` (`17570 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated schema, calibration, leave-one-lab-out, and chamber boundary checks. The final audit found no improvement from expanded bridge terms, and dropping any individual apparatus channel preserved the batch-01 match.
- **Alternatives Weighed**: `Expand the bridge with additional cubic or rational pair terms` (rejected: Both challengers increased held-out raw-sum p90 error and worst clean-channel error.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed. Worst leave-one-lab-out p90 absolute errors were 0.0231 for strain and 0.0158 for initial rate. The apparatus log consistently selected the batch-01 chamber law; the blind batch had no deployment labels to verify its accuracy.`

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
