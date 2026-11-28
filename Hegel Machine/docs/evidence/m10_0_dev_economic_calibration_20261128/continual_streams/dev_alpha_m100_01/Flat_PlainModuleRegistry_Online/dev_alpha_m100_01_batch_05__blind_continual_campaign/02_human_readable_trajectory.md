# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_alpha_m100_01_batch_05__blind_continual_campaign__upstream_transducer`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World` |
| **Domain** | `coupled_viscoelastic_chamber_hereditary_mechanics` |
| **Wave / Wormhole Family** | `m10_0_reliable_modular_amortization_and_bidirectional_withdrawal` / `m100_alpha` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `11 / 16` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `291,502` (`279,555` / `111,488` / `11,947`) |
| **Total Episode Wall Time (s)** | `372.91s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`15658 ms` LLM)
- **Rationale**: Inspect the composition and array shapes to detect schema mismatches, and compare operating and calibration ranges to determine whether the inverse must handle extrapolation.

```bash
cd /holdout_task && cat batch_solver.py _verified_downstream.py validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_batch.pkl','rb'))
def show(v,p='',depth=0):
 if isinstance(v,dict) and depth<4:
  for k,w in v.items(): show(w,p+'/'+str(k),depth+1)
 elif isinstance(v,(list,tuple)):
  print(p,type(v).__name__,len(v))
  if v and depth<4: show(v[0],p+'/[0]',depth+1)
 elif isinstance(v,np.ndarray):
  print(p,v.shape,v.dtype, np.round(np.nanmin(v,axis=0),4) if v.ndim==2 and v.shape[1]<=4 and v.size else '',np.round(np.nanmax(v,axis=0),4) if v.ndim==2 and v.shape[1]<=4 and v.size else '')
 else: print(p,type(v).__name__,repr(v)[:100])
show(x)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust IRLS Cauchy outlier regression'
```
- **Container Exit Code**: `0` (`370 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
gle_channel_sweeps dict {'lab_id': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 2, 
/[0]/dataset/train/lab_id (72,) int64  
/[0]/dataset/train/stratum (72,) int64  
/[0]/dataset/train/coupon_features_raw (72, 4) float64 [-4.3394 -2.9006 -3.1405 -3.2706] [3.7874 2.135  3.5839 2.1272]
/[0]/dataset/train/ref_coupons (72, 8) float64  
/[0]/dataset/train/past_channel_raw (72, 6, 4) float64  
/[0]/dataset/train/hold_channel_raw (72, 4) float64 [1.0162 0.4214 1.0291 0.438 ] [1.9763 1.0219 2.0249 1.0323]
/[0]/dataset/train/strain_0 (72,) float64  
/[0]/dataset/train/strain_curves_obs (72, 12) float64  
/[0]/dataset/train/instant_rate_obs (72,) float64  
/[0]/dataset/train/aux_chamber_telemetry (72, 4) float64 [-4.0327 -2.2887 -1.874  -2.3813] [2.4492 1.6919 3.0095 2.3682]
/[0]/dataset/test/lab_id (36,) int64  
/[0]/dataset/test/stratum (36,) int64  
/[0]/dataset/test/coupon_features_raw (36, 4) float64 [-2.4634 -2.6009 -3.0657 -3.6287] [2.6786 2.1823 3.3829 2.2898]
/[0]/dataset/test/ref_coupons (36, 8) float64  
/[0]/dataset/test/past_channel_raw (36, 6, 4) float64  
/[0]/dataset/test/hold_channel_raw (36, 4) float64 [0.9894 0.4611 1.0226 0.4527] [1.8762 1.0391 1.939  1.0385]
/[0]/dataset/test/strain_0 (36,) float64  
/[0]/dataset/test/aux_chamber_telemetry (36, 4) float64 [-2.3253 -2.6654 -1.4054 -2.71  ] [2.3415 1.2834 2.5927 2.7655]
/[0]/campaign_diagnostics NoneType None
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`19541 ms` LLM)
- **Rationale**: Check whether mixed-load calibrations identify interaction terms and whether operating loads extend beyond the calibration grid; those findings determine the model basis and validation slices.

```bash
cd /holdout_task && cat public_asset_index.json && tail -43 _verified_downstream.py && python3 - <<'PY'
import numpy as np,pickle,json,os
np.set_printoptions(precision=4,suppress=True)
d=pickle.load(open('public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; print('dataset keys',list(d),'verification',type(d.get('chamber_verification_trials'))); print('sweeps',[(k,np.asarray(v).shape,np.asarray(v).ravel()[:14]) for k,v in c['single_channel_sweeps'].items()]); print('calibration first rows x | y',np.c_[x,y][:28]); print('nonzeros/row',np.unique(np.count_nonzero(x,axis=1),return_counts=True));
for s in ['train','test']:
 for k in ['past_channel_raw','hold_channel_raw']:
  a=d[s][k].reshape(-1,4); print(s,k,np.quantile(a,[0,.05,.5,.95,1],axis=0))
for f in ('pre_llm_gate_diagnostic.json','archive_manifest.json','archive_apparatus_log.json'):
 p='/tmp/hegel_continual_archive/'+f
 if os.path.isfile(p):
  a=json.load(open(p)); print(f, str(a if f.startswith('pre_llm') else {k:(len(v) if isinstance(v,(dict,list)) else v) for k,v in a.items()} if isinstance(a,dict) else a[-1:])[:2500])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
```
- **Container Exit Code**: `0` (`432 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lable": true,
  "matches": [
    {
      "module_stem": "autogenous_robust_network_assimilation",
      "granularity_tier": "L2_domain_operator",
      "compiled_code_sha256": "2262dea7fb83d907b5e4ba036cab3459cc080447cbb668b784227f746f8023d6",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "dt",
          "op": ">",
          "value": 0
        },
        {
          "field": "observation_sample_count",
          "op": ">=",
          "value": 3
        },
        {
          "field": "forecast_sample_count",
          "op": ">=",
          "value": 1
        },
        {
          "field": "harmonic_count",
          "op": ">=",
          "value": 1
        },
        {
          "field": "twice_harmonic_count_over_horizon",
          "op": "<",
          "value": 1
        },
        {
          "field": "gain_floor",
          "op": ">",
          "value": 0
        }
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {},
      "functions": [
        {
          "function": "assimilate_diffusive_network",
          "signature": "assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]",
          "doc_summary": "Infer constant transmission and forecast a forward-Euler network."
        }
      ],
      "index_cards": [
        "autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_robust_network_assimilation",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`20776 ms` LLM)
- **Rationale**: Measure calibration geometry, gross-spike patterns, and the operating envelope. If operational loads exceed the calibration range, prioritize a structured forward model over an unconstrained global inverse.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np,json
np.set_printoptions(precision=4,suppress=True)
d=pickle.load(open('public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; print('keys',list(d),'trials',type(d.get('chamber_verification_trials'))); print('sweep',[(k,np.shape(v),np.ravel(v)[:6]) for k,v in c['single_channel_sweeps'].items()]); print('nonzeros',np.unique(np.count_nonzero(x,axis=1),return_counts=True)); print('first rows',np.c_[x,y][:22]); A=np.c_[np.ones(len(x)),x]; w=np.ones(len(x));
for _ in range(25):
 a=np.linalg.lstsq(A*w[:,None],y*w[:,None],rcond=None)[0]; r=y-A@a; e=np.linalg.norm(r,axis=1); w=1/np.sqrt(1+(e/max(np.median(e),1e-5))**2)
print('affine coeffs',a,'residual percentiles',np.quantile(abs(r),[.5,.8,.9,.95,1],axis=0)); print('largest rows',np.c_[np.arange(len(x)),x,r][np.argsort(e)[-12:]]);
for part in ['train','test']:
 for k in ['past_channel_raw','hold_channel_raw']:
  z=(d[part][k].reshape(-1,4)-a[0])@np.linalg.inv(a[1:]); print(part,k,'affine extent',np.quantile(z,[0,.5,1],axis=0))
for f in ['pre_llm_gate_diagnostic.json','archive_manifest.json','archive_apparatus_log.json']:
 a=json.load(open('/tmp/hegel_continual_archive/'+f)); print(f, list(a)[:18] if isinstance(a,dict) else len(a));
 if f.startswith('pre_llm'): print(str(a)[:900])
PY
sed -n '/^def solve_downstream/,$p' _verified_downstream.py
```
- **Container Exit Code**: `0` (`336 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.2057
   -0.1616]
 [ 45.      -0.6271   0.3175  -0.5413   1.1272  -0.2764   0.0874  -0.2955
    0.1143]
 [ 61.       1.4759   1.1142   1.2144   0.8684  -0.3965   0.0491  -0.1505
    0.1164]
 [ 75.       1.5674  -0.178   -1.572    0.7609  -0.327   -0.1275   0.1732
    0.2187]
 [ 71.       0.9838   0.5767  -0.6043  -0.5077   0.186    0.1493  -0.3545
   -0.1859]
 [ 43.      -1.504   -1.2738  -0.4012  -0.902    0.37     0.0309  -0.3035
   -0.1167]
 [ 37.      -1.5454   0.3291  -0.3008   0.8756   0.3137   0.1176  -0.2961
    0.2262]
 [ 26.      -0.7532  -0.2021   1.5654   0.0761  -0.4058   0.0129  -0.2908
    0.1375]
 [ 67.      -0.7483   1.3646  -1.551   -0.8682  -0.3323  -0.2202   0.2856
   -0.2359]
 [ 14.       0.       0.       0.       0.7     14.5538 -13.1875  12.8536
  -13.8369]
 [ 15.       0.       0.       0.       1.7     14.5353 -13.2232  12.8389
  -14.2079]
 [ 13.       0.       0.       0.      -0.6     14.5781 -13.2028  12.8724
  -14.2282]]
train past_channel_raw affine extent [[-0.9922 -0.8689 -0.9343 -0.9707]
 [ 0.2315  0.4116  0.4255  0.443 ]
 [ 1.286   1.2388  1.2568  1.3559]]
train hold_channel_raw affine extent [[0.4771 0.5434 0.4737 0.4894]
 [0.9752 1.0275 0.9968 1.0422]
 [1.1822 1.2437 1.2091 1.2847]]
test past_channel_raw affine extent [[-0.939  -0.8351 -0.9419 -0.9366]
 [ 0.5134  0.4043  0.39    0.3737]
 [ 1.2528  1.2388  1.2498  1.3265]]
test hold_channel_raw affine extent [[0.4242 0.5597 0.4658 0.4849]
 [0.8528 0.9379 0.9584 1.0373]
 [1.1646 1.2189 1.2117 1.2658]]
pre_llm_gate_diagnostic.json ['pre_llm_gate_accepted', 'repair_target', 'failure_kind']
{'pre_llm_gate_accepted': False, 'repair_target': 'upstream_module_solve', 'failure_kind': 'scientific_or_numerical'}
archive_manifest.json 3
archive_apparatus_log.json 6
def solve_downstream(cal_dataset: dict) -> dict:
    train, test = cal_dataset['train'], cal_dataset['test']
    x = _features(train, test)
    n = len(train['lab_id'])
    dt, t = (np.asarray(cal_dataset[key]) for key in ('past_dt', 'eval_times'))
    taus, theta = _fit(*_total_loads(train), x[:n], dt, t,
                       np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None],
                       np.asarray(train['instant_rate_obs']))
    c, r = _design(taus, *_total_loads(test), x[n:], dt, t)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': np.asarray(test['strain_0'])[:, None] + c @ theta,
            'initial_rate_pred': r @ theta}
```

### Turn 4 — `execute_in_remote_docker` (`25599 ms` LLM)
- **Rationale**: Use measured slopes and held-out residuals to distinguish smooth bridge laws before choosing an inverse.
- **Alternatives Weighed**: `Fit a direct raw-to-load polynomial` (rejected: It obscures the paired forward structure and leaves gross spikes in the predictor coordinates.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine bridge explains clean calibration measurements.` | Witness: `Channel 0 rises by 0.713, 2.280, and 0.673 over successive single-channel intervals of length 1.0, 1.3, and 1.0; other true loads are held at zero.` | Blanket Diagnosis: `The changing slope is visible at fixed cross-channel load, so linear coupling or the three gross spike rows cannot explain it.` | Sublated Pivot: `Retain the observed (0,2) and (1,3) couplings while testing saturating bases on held-out calibration rows.`

```bash
cat > /tmp/bridge_probe.py <<'PY'
import numpy as np,pickle
from scipy.optimize import least_squares
x,y=(lambda c:(c['true_channel_loads'],c['bridge_raw']))(pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']); good=np.max(abs(y),axis=1)<5
np.set_printoptions(precision=5,suppress=True)
for j in range(4):
 u,v=x[:,j],x[:,j^2]; z=y[:,j]
 for name in ['lin','tanh','tanh+lin','rational','atan','poly','shifted','full']:
  def pred(p):
   b,a,k,c,h,d,e=p
   if name=='lin': return b+a*u+c*v
   if name=='tanh': return b+a*np.tanh(k*u)+c*np.tanh(h*v)
   if name=='tanh+lin': return b+a*np.tanh(k*u)+c*np.tanh(h*v)+d*u+e*v
   if name=='rational': return b+a*u/(1+k*k*u*u)+c*v/(1+h*h*v*v)+d*u+e*v
   if name=='atan': return b+a*np.arctan(k*u)+c*np.arctan(h*v)+d*u+e*v
   if name=='poly': return b+a*u+k*u**3+c*v+h*v**3+d*u**5+e*v**5
   if name=='shifted': return b+a*np.tanh(k*u+d)+c*np.tanh(h*v+e)
   if name=='full': return b+a*np.tanh(k*u)+c*np.tanh(h*v)+d*u*v+e*u*u*v
  err=[]
  for fold in range(5):
   fit=good & (np.arange(len(x))%5!=fold); val=good & ~fit
   p=least_squares(lambda p:(pred(p)-z)[fit],[0,2,1,.6,.5,0,0],max_nfev=800).x; err.extend((pred(p)-z)[val]);
  print(j,name,'rmse/max',np.sqrt(np.mean(np.square(err))),np.max(abs(np.array(err))),'p',p)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/bridge_probe.py
ls -la /tmp/hegel_continual_archive
```
- **Container Exit Code**: `0` (`5753 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
989041472298559 0.3933610626487958 p [0.06763 1.34321 1.00013 0.28331 0.5     0.      0.     ]
2 tanh rmse/max 0.06623064935158189 0.21423056653700545 p [0.05616 1.88951 1.11185 2.76544 0.09825 0.      0.     ]
2 tanh+lin rmse/max 0.06425001078832483 0.17642185118129028 p [ 0.05697  1.2421   1.47837 15.61854  0.06718  0.37677 -0.77927]
2 rational rmse/max 0.06415420343833336 0.17381333152659684 p [  0.05698   1.57208   0.91187  15.56864  -0.01031   0.63784 -15.29842]
2 atan rmse/max 0.06482701838741806 0.1879740152792433 p [ 0.05694  1.45745  1.49006 12.57952  0.06898  0.06781 -0.59897]
2 poly rmse/max 0.06318596979371574 0.12244934793195084 p [ 0.05644  2.10203 -0.71495  0.23184  0.07123  0.13423 -0.02334]
2 shifted rmse/max 0.03437964236048788 0.16364185180405877 p [-0.32603  1.87526  1.11202  0.79285  0.51857  0.001    0.63976]
2 full rmse/max 0.06802236507312326 0.2124178899966953 p [ 0.06012  1.87897  1.11787  4.03045  0.07106  0.00658 -0.01978]
3 lin rmse/max 0.12429781783285043 0.2635197913864121 p [-0.16269  0.80107  2.99445  0.18977  0.5      0.       0.     ]
3 tanh rmse/max 0.0378363776350828 0.07587713700490939 p [-0.13377  1.11596  1.18373  0.58245  0.29256  0.       0.     ]
3 tanh+lin rmse/max 0.03865252360468429 0.09837813515577892 p [-0.13421  0.85787  1.42853 34.45378  0.08131  0.15099 -2.62788]
3 rational rmse/max 0.038322893138639125 0.09861174832876257 p [ -0.13416   1.0467    0.8824   39.97218  -0.01236   0.32716 -39.79859]
3 atan rmse/max 0.03848851505669052 0.0891667887611293 p [-0.13451  1.0312   1.42143 65.87147  0.06623 -0.07158 -4.1889 ]
3 poly rmse/max 0.04005850277687929 0.12246455180505131 p [-0.13241  1.28433 -0.44498  0.15402  0.02355  0.084   -0.0089 ]
3 shifted rmse/max 0.013833492367346429 0.05788682464630229 p [-0.339    1.13263  1.13227  0.411    0.60428  0.0037   0.63757]
3 full rmse/max 0.040258870607967887 0.12229510537940569 p [-0.12768  1.10224  1.21988  0.40871  0.35523 -0.01814  0.02512]
total 56
drwxr-xr-x 2 root root  4096 Nov 28 02:47 .
drwxrwxrwt 1 root root  4096 Nov 28 02:48 ..
-r--r--r-- 1 root root 23306 Nov 28 02:47 archive_apparatus_log.json
-rw-r--r-- 1 root root  1177 Nov 28 02:47 archive_manifest.json
-rw-r--r-- 1 root root  4726 Nov 28 02:47 dev_alpha_m100_01_batch_01_solver.py
-rw-r--r-- 1 root root  4074 Nov 28 02:47 dev_alpha_m100_01_batch_02_solver.py
-rw-r--r-- 1 root root  4094 Nov 28 02:47 dev_alpha_m100_01_s00_solver.py
-rw-r--r-- 1 root root   125 Nov 28 02:47 pre_llm_gate_diagnostic.json
```

### Turn 5 — `execute_in_remote_docker` (`31917 ms` LLM)
- **Rationale**: Compare compact asymmetric coupling laws and mixed-load terms on the same held-out rows. If an interaction consistently lowers error, retain it; otherwise prefer the simpler separable law.
- **Alternatives Weighed**: `Use shifted tanh immediately` (rejected: Its worst held-out bridge error is 0.164, large enough to warrant checking whether a missing coupling term explains it.)

```bash
grep -inE 'bridge|tanh|calibrat|arctan|coupl|saturat' /tmp/hegel_continual_archive/*solver.py; python3 - <<'PY'
import json
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
a=json.load(open(p)); print('telemetry entries:',[(i,list(r)) for i,r in enumerate(a)])
PY
cat > /tmp/bridge_terms.py <<'PY'
import numpy as np,pickle
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; clean=np.max(abs(y),axis=1)<5
terms={
 'tanh_shift': (lambda v,k,s:np.tanh(k*v+s)-np.tanh(s), [.5,.6]),
 'atan_shift': (lambda v,k,s:np.arctan(k*v+s)-np.arctan(s), [.5,.6]),
 'exp': (lambda v,k:-np.expm1(-k*v), [.3]),
 'rat':(lambda v,k:v/(1+k*v), [.15]),
 'quadratic':(lambda v,k:v+k*v*v, [-.1]),
 'sin_cos':(lambda v,k,s:np.sin(k*v)+s*(np.cos(k*v)-1), [.6,.5]),
 'log':(lambda v,k:np.log1p(k*v), [.2]),
 'linear_tanh':(lambda v,k,s:v+s*(np.tanh(k*v+.6)-np.tanh(.6)), [.6,1]),
}
for j in range(4):
 u,v=x[:,j],x[:,j^2]; z=y[:,j]
 for name,(term,start) in terms.items():
  for interaction in ('none','uv','sum_inside','sum_difference'):
   def pred(p):
    b,a,k,d=p[:4]; m=0 if interaction=='none' else p[-1]; w=p[4:4+len(start)]
    if interaction=='sum_difference': return b+a*np.tanh(k*u)+d*(term(v+m*u,*w)-term(m*u,*w))
    return b+a*np.tanh(k*(u+(m*v if interaction=='sum_inside' else 0)))+d*term(v,*w)+(m*u*v if interaction=='uv' else 0)
   ini=[0,1.8 if j%2==0 else 1.1,1,.35]+start+([] if interaction=='none' else [0.]); lo=np.full(len(ini),-np.inf); hi=-lo
   if name in ('rat','log'):lo[4]=-.53;hi[4]=.53
   errs=[]
   for fold in range(5):
    val=clean&(np.arange(len(x))%5==fold); tr=clean&~val
    p=least_squares(lambda p:(pred(p)-z)[tr],ini,bounds=(lo,hi),max_nfev=180).x; errs.extend((pred(p)-z)[val])
   p=least_squares(lambda p:(pred(p)-z)[clean],ini,bounds=(lo,hi),max_nfev=260).x
   print(j,name,interaction,'%.6f %.6f %.6f'%(np.mean(np.square(errs))**.5,max(abs(np.array(errs))),np.mean((pred(p)-z)[clean]**2)**.5),np.round(p,4))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/bridge_terms.py
```
- **Container Exit Code**: `0` (`4080 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
9995 0.014476 [-0.1107  1.1252  1.1488  0.3165  0.4913]
3 exp uv 0.016461 0.049387 0.014470 [-1.1070e-01  1.1255e+00  1.1483e+00  3.1560e-01  4.9260e-01  6.0000e-04]
3 exp sum_inside 0.016393 0.054834 0.014401 [-0.1106  1.1253  1.1485  0.3014  0.5052  0.0042]
3 exp sum_difference 0.016606 0.048842 0.014438 [-0.1106  1.1262  1.1475  0.3143  0.4947 -0.0154]
3 rat none 0.017149 0.052531 0.015519 [-0.1125  1.1236  1.1526  0.1533  0.22  ]
3 rat uv 0.017540 0.052063 0.015519 [-0.1125  1.1236  1.1526  0.1533  0.22    0.    ]
3 rat sum_inside 0.017485 0.053414 0.015500 [-0.1125  1.1236  1.1526  0.1517  0.2225  0.0021]
3 rat sum_difference 0.017721 0.051241 0.015482 [-0.1124  1.1247  1.1512  0.1532  0.2215 -0.0153]
3 quadratic none 0.014946 0.047971 0.013686 [-0.1093  1.1267  1.1462  0.1649 -0.2539]
3 quadratic uv 0.016207 0.054517 0.013647 [-0.1092  1.1274  1.1451  0.165  -0.2557  0.0014]
3 quadratic sum_inside 0.015212 0.046670 0.013522 [-0.109   1.127   1.1455  0.1607 -0.2651  0.0061]
3 quadratic sum_difference 0.016207 0.054517 0.013647 [-0.1092  1.1274  1.1451  0.165  -0.2557 -0.0163]
3 sin_cos none 0.015999 0.068554 0.013666 [-0.1092  1.1269  1.146   0.7886  0.2116  2.3961]
3 sin_cos uv 0.017808 0.079979 0.013622 [-1.0910e-01  1.1276e+00  1.1448e+00  7.4380e-01  2.2470e-01  2.2732e+00
  1.5000e-03]
3 sin_cos sum_inside 0.016270 0.067509 0.013499 [-0.1089  1.1272  1.1453  0.7403  0.2197  2.4094  0.0062]
3 sin_cos sum_difference 0.018302 0.085944 0.013628 [-0.1091  1.1275  1.1449  0.793   0.2104  2.4271 -0.0164]
3 log none 0.018545 0.055159 0.016760 [-0.1146  1.1218  1.1572  0.3999  0.3816]
3 log uv 0.018833 0.054814 0.016755 [-1.1460e-01  1.1216e+00  1.1575e+00  4.0060e-01  3.8100e-01 -5.0000e-04]
3 log sum_inside 0.018652 0.055288 0.016760 [-1.1460e-01  1.1218e+00  1.1572e+00  4.0090e-01  3.8120e-01 -3.0000e-04]
3 log sum_difference 0.018998 0.053695 0.016721 [-0.1145  1.1231  1.1556  0.3969  0.3843 -0.0161]
3 linear_tanh none 0.013150 0.053872 0.011959 [-1.05900e-01  1.13200e+00  1.13240e+00  1.16000e-02  7.15300e-01
  2.78964e+01]
3 linear_tanh uv 0.013839 0.055002 0.011904 [-1.05800e-01  1.13280e+00  1.13110e+00  1.02000e-02  7.14200e-01
  3.18231e+01  1.50000e-03]
3 linear_tanh sum_inside 0.013209 0.053937 0.011834 [-1.05700e-01  1.13220e+00  1.13210e+00  4.80000e-03  7.03800e-01
  6.96641e+01  5.10000e-03]
3 linear_tanh sum_difference 0.014021 0.057300 0.011894 [-1.05700e-01  1.13180e+00  1.13190e+00  1.12000e-02  7.23300e-01
  2.85668e+01 -2.37000e-02]
```

### Turn 6 — `execute_in_remote_docker` (`37561 ms` LLM)
- **Rationale**: Compare held-out axis and mixed-load errors to distinguish a missing asymmetric basis from missing cross-pair interactions. Use the result to select the smallest invertible forward law.
- **Alternatives Weighed**: `Increase polynomial degree without splitting the response at zero` (rejected: A high-degree fit can hide a slope break in-sample while producing larger errors on the held-out axis and boundary rows.)

```bash
cat >/tmp/bridge_basis.py <<'PY'
import pickle,json,numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; clean=np.max(abs(y),axis=1)<5; axis=np.count_nonzero(x,axis=1)==1
np.set_printoptions(precision=5,suppress=True)
for j in range(4):
 u,v=x[:,j],x[:,j^2]; z=y[:,j]
 print('\nCH',j)
 for n,m,mode in [(5,5,'pair'),(8,8,'pair'),(11,11,'pair'),(8,8,'all'),(8,8,'interact'),(8,8,'hinges'),(8,8,'split'),(8,8,'warp')]:
  U=chebvander(u/1.75,n); V=chebvander(v/1.75,m); A=np.c_[U,V[:,1:]]
  if mode=='all': A=np.c_[A,*[chebvander(x[:,k]/1.75,3)[:,1:] for k in range(4) if k not in (j,j^2)]]
  if mode=='interact': A=np.c_[A,*[U[:,k]*V[:,l] for k in range(1,4) for l in range(1,4) if k+l<=4]]
  if mode=='hinges': A=np.c_[U,np.maximum(v,0),np.minimum(v,0),np.maximum(v,0)**2,np.minimum(v,0)**2,np.maximum(v,0)**3,np.minimum(v,0)**3]
  err=np.zeros(len(x))
  def fit(tr,te):
   if mode not in ('warp','split'): return A[te]@np.linalg.lstsq(A[tr],z[tr],rcond=None)[0]
   def pred(p):
    b,a,k,c,d,h,l=p
    if mode=='split': return b+a*np.tanh(k*u)+c*np.tanh(d*np.minimum(v,0))+h*np.tanh(l*np.maximum(v,0))
    return b+a*np.tanh(k*u+c*u**3)+d*v+h*abs(v)+l*v**3
   p=least_squares(lambda p:(pred(p)-z)[tr],[0,1.8,1.1,.3,1,.2,1],max_nfev=300).x
   return pred(p)[te]
  for fold in range(5):
   te=clean&(np.arange(len(x))%5==fold); err[te]=fit(clean&~te,te)-z[te]
  print(n,m,mode,'rmse',*[round(np.mean(err[s]**2)**.5,6) for s in (clean,clean&axis,clean&~axis)],'max',round(max(abs(err)),6),'in',round(np.mean((fit(clean,clean)-z[clean])**2)**.5,6))
  if mode=='pair' and n==8:
   ix=np.argsort(abs(err))[-6:];print('worst i,x,e',np.c_[ix,x[ix],err[ix]])
print('Telemetry scalar fields:')
a=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for r in a[-2:]:
 t=r['facility_telemetry_trace']; print(r['completed_campaign_short_name'],list(t) if isinstance(t,dict) else type(t).__name__)
 if isinstance(t,dict):print({k:v for k,v in t.items() if isinstance(v,(str,int,float,type(None)))})
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/bridge_basis.py
```
- **Container Exit Code**: `0` (`1990 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
077]
 [ 4.       0.      -1.6      0.       0.       0.00085]
 [61.       1.47593  1.11421  1.21436  0.86839 -0.00089]
 [38.       0.30349  1.45058  1.38591  0.58673  0.00102]
 [ 7.       0.       1.7      0.       0.      -0.02287]]
11 11 pair rmse 0.000151 0.000109 0.000159 max 0.000742 in 8.8e-05
8 8 all rmse 0.002749 0.006634 0.000389 max 0.023891 in 0.00027
8 8 interact rmse 0.002708 0.006525 0.000417 max 0.02347 in 0.000269
8 8 hinges rmse 0.001937 0.004645 0.00036 max 0.016692 in 0.000293
8 8 split rmse 0.014661 0.025967 0.01103 max 0.076618 in 0.013955
8 8 warp rmse 0.01406 0.024693 0.010676 max 0.071858 in 0.013183

CH 2
5 5 pair rmse 0.01337 0.02427 0.009768 max 0.082357 in 0.008272
8 8 pair rmse 0.00134 0.002997 0.000579 max 0.010165 in 0.000494
worst i,x,e [[57.      -1.19108 -0.00437 -0.17887  0.35761  0.00107]
 [ 0.      -1.6      0.       0.       0.      -0.00112]
 [34.      -1.58444 -0.43856  1.57016 -1.1084   0.00129]
 [ 3.       1.7      0.       0.       0.       0.00217]
 [ 8.       0.       0.      -1.6      0.       0.00245]
 [11.       0.       0.       1.7      0.      -0.01017]]
11 11 pair rmse 0.000225 0.00045 0.00014 max 0.001435 in 9.2e-05
8 8 all rmse 0.001394 0.003073 0.000649 max 0.010095 in 0.000478
8 8 interact rmse 0.002055 0.004768 0.000682 max 0.015992 in 0.000463
8 8 hinges rmse 0.001471 0.003264 0.000662 max 0.011188 in 0.00057
8 8 split rmse 0.026172 0.042764 0.021275 max 0.127078 in 0.024024
8 8 warp rmse 0.025509 0.041894 0.020648 max 0.125554 in 0.023135

CH 3
5 5 pair rmse 0.006425 0.011449 0.004801 max 0.035609 in 0.003933
8 8 pair rmse 0.000473 0.000748 0.000394 max 0.00217 in 0.00022
worst i,x,e [[50.       1.56055  0.18677  0.21324  1.03936 -0.00075]
 [55.      -1.38613 -0.32033  0.64521  1.07037 -0.00089]
 [45.      -0.62707  0.31747 -0.5413   1.12723 -0.00095]
 [ 7.       0.       1.7      0.       0.      -0.00152]
 [36.      -0.5999   0.06378 -0.17903  1.56794 -0.00164]
 [12.       0.       0.       0.      -1.6      0.00217]]
11 11 pair rmse 0.001086 0.002626 0.000139 max 0.009457 in 8.4e-05
8 8 all rmse 0.000626 0.001252 0.000391 max 0.00412 in 0.000208
8 8 interact rmse 0.000487 0.00076 0.00041 max 0.002546 in 0.000207
8 8 hinges rmse 0.000556 0.000957 0.000431 max 0.002939 in 0.000288
8 8 split rmse 0.011447 0.016312 0.010179 max 0.053057 in 0.01037
8 8 warp rmse 0.011141 0.016152 0.009815 max 0.050487 in 0.00989
Telemetry scalar fields:
dev_alpha_m100_01_batch_03 list
dev_alpha_m100_01_batch_04 list
```

### Turn 7 — `execute_in_remote_docker` (`41527 ms` LLM)
- **Rationale**: Measure held-out mixed-load error and single-axis boundary error separately. If a compact parametric law matches both, use it; otherwise use a regularized paired basis and verify inversion in load coordinates.
- **Alternatives Weighed**: `Fix both polynomial degrees at 11` (rejected: Channel 3's held-out boundary error increased relative to degree 8; test separate self and coupling degrees and mild regularization before choosing the inverse.)

```bash
cat >/tmp/select_bridge.py <<'PY'
import numpy as np,pickle
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; x,y=c['true_channel_loads'],c['bridge_raw']; ix=np.arange(len(x)); ax=np.count_nonzero(x,axis=1)==1
A=np.c_[np.ones(len(x)),x]; w=np.ones(len(x))
for _ in range(12):
 p=np.linalg.lstsq(A*w[:,None],y*w[:,None],rcond=None)[0]; e=np.linalg.norm(y-A@p,axis=1); w=1/np.sqrt(1+(e/np.median(e))**2)
clean=e<6*np.median(e); print('outliers',ix[~clean], 'residual gap',np.sort(e)[-6:]); np.set_printoptions(precision=6,suppress=True)
for j in range(4):
 u,v=x[:,j],x[:,j^2]; z=y[:,j]; U,V=chebvander(u/1.75,13),chebvander(v/1.75,13)
 p=np.linalg.lstsq(np.c_[U[:,:12],V[:,1:12]][clean],z[clean],rcond=None)[0]; print('CH',j,'Cheb own',p[:12],'cross',p[12:]);
 def score(fit):
  err=np.zeros(len(x)); edge=np.zeros(len(x))
  for fold in range(5):
   te=clean&(ix%5==fold); err[te]=fit(clean&~te,te)-z[te]
  for k in ix[ax&clean]:
   te=ix==k; edge[k]=(fit(clean&~te,te)-z[te])[0]
  return np.array([np.linalg.norm(err[clean&~ax])/np.sqrt(sum(clean&~ax)),max(abs(err)),max(abs(edge))])
 scores=[]
 for n in (7,9,11,13):
  for m in (5,7,9,11,13):
   D=np.c_[U[:,:n+1],V[:,1:m+1]]
   for lam in (0,1e-5,1e-4,1e-3):
    pen=lam*np.diag(np.r_[np.maximum(np.arange(n+1)-4,0)**2,np.maximum(np.arange(1,m+1)-3,0)**2]);
    def fit(tr,te): return D[te]@np.linalg.lstsq(np.r_[D[tr],pen],np.r_[z[tr],np.zeros(D.shape[1])],rcond=None)[0]
    a=score(fit); scores.append((max(a),n,m,lam,a))
 print('best polynomial',sorted(scores,key=lambda s:s[0])[:5]); print('degree 11,7 (unpenalized)',next(s[-1] for s in scores if s[1:4]==(11,7,0)))
 for name in ('rat','tanh_square','shift_linear','rat_shift','sin','asym'):
  def f(p):
   b,a,k,d,g,h,l,m,q=p; own=a*np.tanh(k*u)+d*u+g*u**3
   if name=='rat': cross=(h*v+l*v*v)/(1+(m*v)**2)+q*v
   if name=='tanh_square': cross=h*np.tanh(m*v)+l*np.tanh(m*v)**2+q*v
   if name=='shift_linear': cross=h*(np.tanh(m*v+l)-np.tanh(l))+q*v
   if name=='rat_shift': cross=h*(v+l)/(1+(m*(v+l))**2)-h*l/(1+(m*l)**2)+q*v
   if name=='sin': cross=h*np.sin(m*v)+l*(np.cos(m*v)-1)+q*v
   if name=='asym': cross=h*v+l*v*v/(1+(m*v)**2)+q*np.tanh(m*v)
   return b+own+cross
  def fit(tr,te):
   p=least_squares(lambda p:(f(p)-z)[tr],[0,1.6 if j%2==0 else 1.,1.2,.1,0,.4,-.25,.65,.05],max_nfev=200).x
   return f(p)[te]
  print(name,score(fit))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/select_bridge.py
```
- **Container Exit Code**: `0` (`11892 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
62, 11, 7, 0.001, array([0.000125, 0.000343, 0.000522])), (0.000829434402255913, 11, 7, 0.0001, array([0.000125, 0.000343, 0.000829])), (0.000832572249414909, 11, 7, 1e-05, array([0.000125, 0.000343, 0.000833])), (0.0008326039512434047, 11, 7, 0, array([0.000125, 0.000343, 0.000833])), (0.0016142497314566295, 11, 11, 0, array([0.000159, 0.000742, 0.001614]))]
degree 11,7 (unpenalized) [0.000125 0.000343 0.000833]
rat [0.001792 0.006667 0.006924]
tanh_square [0.001436 0.005497 0.00593 ]
shift_linear [0.0039   0.013108 0.003471]
rat_shift [0.003757 0.014901 0.014866]
sin [0.000313 0.003745 0.003902]
asym [0.0018   0.006723 0.007032]
CH 2 Cheb own [ 0.032353  2.147986 -0.000009 -0.280313  0.000013  0.115849  0.000023
 -0.016696  0.000066  0.00119   0.000055 -0.000104] cross [ 0.477674 -0.071487 -0.000013  0.031964 -0.000007 -0.003936 -0.000037
  0.000264 -0.000026  0.000003 -0.000016]
best polynomial [(0.0005804127692887739, 11, 9, 0.001, array([0.000134, 0.00058 , 0.000276])), (0.0006152365046858677, 11, 9, 0.0001, array([0.000134, 0.000615, 0.000278])), (0.0006155861469188117, 11, 9, 1e-05, array([0.000134, 0.000616, 0.000278])), (0.0006155896788300197, 11, 9, 0, array([0.000134, 0.000616, 0.000278])), (0.0010277943294685343, 9, 11, 0, array([0.000142, 0.001028, 0.000903]))]
degree 11,7 (unpenalized) [0.000174 0.001241 0.001207]
rat [0.00355  0.015153 0.015239]
tanh_square [0.002823 0.012868 0.013224]
shift_linear [0.008477 0.034964 0.036589]
rat_shift [0.007872 0.032085 0.033695]
sin [0.000533 0.006138 0.006118]
asym [0.003556 0.015228 0.015349]
CH 3 Cheb own [-0.165273  1.304801  0.000127 -0.170804  0.000094  0.070049  0.000071
 -0.010344  0.000073  0.000625 -0.000005 -0.00011 ] cross [ 0.29018  -0.043341 -0.000129  0.01944  -0.000142 -0.002368 -0.000071
  0.000152 -0.000054  0.000014  0.000005]
best polynomial [(0.0006176507936176273, 9, 9, 0.001, array([0.000138, 0.000375, 0.000618])), (0.0006283108095018752, 9, 9, 0.0001, array([0.000138, 0.000374, 0.000628])), (0.0006284176471488545, 9, 9, 1e-05, array([0.000138, 0.000374, 0.000628])), (0.0006284187263372043, 9, 9, 0, array([0.000138, 0.000374, 0.000628])), (0.000908252333929202, 13, 9, 0, array([0.000138, 0.000908, 0.00033 ]))]
degree 11,7 (unpenalized) [0.000133 0.000652 0.001429]
rat [0.002003 0.014818 0.015123]
tanh_square [0.001639 0.012757 0.013044]
shift_linear [0.00321  0.031508 0.032296]
rat_shift [0.00357  0.029663 0.030599]
sin [0.000246 0.001255 0.001348]
asym [0.002005 0.015742 0.016031]
```

### Turn 8 — `execute_in_remote_docker` (`35261 ms` LLM)
- **Rationale**: Test whether the alternating odd coefficients and even coupling coefficients come from a shared Fourier frequency. Compare held-out boundary errors and derivative margins before choosing how to invert the bridge.
- **Alternatives Weighed**: `Use the Chebyshev fit directly` (rejected: Keep it as a fallback: the repeated coefficient pattern suggests a lower-parameter law that may extrapolate more reliably.)

```bash
cat > /tmp/fourier_bridge.py <<'PY'
import pickle
import numpy as np
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']
x,y=c['true_channel_loads'],c['bridge_raw']; idx=np.arange(len(x)); axis=np.count_nonzero(x,axis=1)==1
A=np.c_[np.ones(len(x)),x]; w=np.ones(len(x))
for _ in range(12):
    p=np.linalg.lstsq(A*w[:,None],y*w[:,None],rcond=None)[0]
    e=np.linalg.norm(y-A@p,axis=1); w=1/np.sqrt(1+(e/np.median(e))**2)
clean=e<6*np.median(e)
for j in range(4):
    u,v=x[:,j],x[:,j^2]; z=y[:,j]
    for shared in (True,False):
        def design(freq,u,v):
            k,h=(freq[0],freq[-1]); return np.c_[np.ones(len(u)),u,np.sin(k*u),v,np.sin(h*v),np.cos(h*v)-1]
        def fit(tr,te):
            def profile(f):
                D=design(f,u[tr],v[tr]); p=np.linalg.lstsq(D,z[tr],rcond=None)[0]; return D,p
            def residual(f):
                D,p=profile(f); return D@p-z[tr]
            f=least_squares(residual,np.full(1 if shared else 2,2.4),bounds=(0.6,5.)).x
            _,p=profile(f); return design(f,u[te],v[te])@p, f,p
        err=np.zeros(len(x)); edge=[]
        for fold in range(5):
            te=clean&(idx%5==fold); err[te]=fit(clean&~te,te)[0]-z[te]
        for k in idx[axis&clean]:
            te=idx==k; edge.append((fit(clean&~te,te)[0]-z[te])[0])
        pred,f,p=fit(clean,clean)
        print(j,'shared',shared,'frequencies',f,'coef',p,'fit rms',np.std(pred-z[clean]),'mixed rms',np.std(err[clean&~axis]),'max fivefold/axis',max(abs(err)),max(abs(np.array(edge))))
        zz=np.linspace(-2.5,2.5,1001); print('derivative ranges',np.min(p[1]+f[0]*p[2]*np.cos(f[0]*zz)),np.max(abs(p[3]+f[-1]*(p[4]*np.cos(f[-1]*zz)-p[5]*np.sin(f[-1]*zz)))))
for part in ('train','test'):
    d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset'][part]
    for key in ('past_channel_raw','hold_channel_raw'):
        a=d[key].reshape(-1,4); print(part,key,a.min(0),a.max(0))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/fourier_bridge.py
```
- **Container Exit Code**: `0` (`738 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
39968e-05 max fivefold/axis 0.00023811673669249522 0.00019134803579534676
derivative ranges 0.4762413997500945 0.4190876265230277
1 shared True frequencies [2.41216061] coef [-0.10799616  0.78630132  0.20625012  0.1670894  -0.00510841  0.04544839] fit rms 0.007322842628221548 mixed rms 0.00854097713002539 max fivefold/axis 0.01954554496216654 0.01896230309071223
derivative ranges 0.28880121624185046 0.27740700802684637
1 shared False frequencies [2.5007934  2.00073948] coef [-1.10010146e-01  8.00126936e-01  1.99945966e-01  1.67996606e-01
 -4.48204386e-05  4.79912418e-02] fit rms 0.00010157516383192103 mixed rms 0.00011249379236790501 max fivefold/axis 0.0003190440918016124 0.0002496976251573635
derivative ranges 0.3001057800888227 0.26401461650820646
2 shared True frequencies [2.37553629] coef [1.45544405e-01 1.26652518e+00 3.40348720e-01 2.72431075e-01
 1.02536470e-03 7.47240463e-02] fit rms 0.01441974045554292 mixed rms 0.015019120723135371 max fivefold/axis 0.041546645995231835 0.0418640389463944
derivative ranges 0.458028445891848 0.4499573625266595
2 shared False frequencies [2.49994494 2.00051239] coef [ 1.40009131e-01  1.29998962e+00  3.25001127e-01  2.72970617e-01
 -1.61101243e-06  7.80132742e-02] fit rms 0.00010797634751548237 mixed rms 0.00011665990123915437 max fivefold/axis 0.00030476689148484226 0.00031077174427096566
derivative ranges 0.4875117363214435 0.42903712487953927
3 shared True frequencies [2.36308824] coef [-0.09766757  0.76606405  0.2062031   0.16665052 -0.00127091  0.04489179] fit rms 0.006120692874660783 mixed rms 0.0062269334867635 max fivefold/axis 0.03442110578573217 0.0315097077533506
derivative ranges 0.278788358245353 0.2727754243503976
3 shared False frequencies [2.49978762 1.99993937] coef [-1.00014658e-01  7.89967134e-01  1.97534006e-01  1.65913084e-01
 -3.88422996e-05  4.73693380e-02] fit rms 9.449340474261286e-05 mixed rms 0.00011373413716372545 max fivefold/axis 0.0002905224440367532 0.0001955974594788218
derivative ranges 0.2961787627546002 0.2606487890037832
train past_channel_raw [-1.22215782 -0.9385742  -1.232478   -0.91511029] [1.83656595 0.94440853 1.88119777 0.92536363]
train hold_channel_raw [1.01624979 0.42142669 1.02911658 0.43795524] [1.97633611 1.02191459 2.02492607 1.03227813]
test past_channel_raw [-1.1071219  -0.88787561 -1.16296403 -0.86652444] [1.82695572 0.95184563 1.8747623  0.9428324 ]
test hold_channel_raw [0.98941441 0.46110585 1.02258465 0.45266919] [1.87621591 1.03909859 1.93898567 1.03848333]
```

### Turn 9 — `execute_in_remote_docker` (`51202 ms` LLM)
- **Rationale**: The calibration fits identify separate frequencies near 2.5 and 2.0 and no measurable cross-channel sine term. Fit both frequencies from each dataset, reject spike rows, and use diagonal dominance to safeguard inversion. Run the public composition check to catch interface or downstream failures before load-space stress testing.
- **Alternatives Weighed**: `Use one Fourier frequency for both terms` (rejected: It raised the worst held-out bridge error from 0.000319 to 0.0508.)

```bash
cd /holdout_task && cat > upstream_transducer.py <<'PY'
"""Calibrate the paired four-channel bridge before constitutive fitting."""

import numpy as np
from scipy.optimize import least_squares

_PAIR = np.array([2, 3, 0, 1])
_LOAD_KEYS = ('past_channel_raw', 'hold_channel_raw')


def _clean_rows(loads, raw):
    """Use row residuals so a spike on one bridge cannot bias its pair."""
    design = np.column_stack((np.ones(len(loads)), loads))
    weight = np.ones(len(loads))
    for _ in range(12):
        coeff = np.linalg.lstsq(design * weight[:, None], raw * weight[:, None], rcond=None)[0]
        error = np.linalg.norm(design @ coeff - raw, axis=1)
        scale = max(np.median(error), 1e-10)
        weight = 1.0 / np.hypot(1.0, error / scale)
    return error <= 6.0 * scale


def _design(loads, freq):
    """One design matrix per channel; self and paired frequencies are distinct."""
    paired = loads[:, _PAIR]
    return np.stack((np.ones_like(loads), loads, np.sin(freq[0] * loads),
                     paired, -2.0 * np.sin(0.5 * freq[1] * paired)**2), axis=-1)


def _profile(loads, raw):
    def linear_fit(freq):
        design = _design(loads, freq)
        coeff = np.stack([np.linalg.lstsq(design[:, j], raw[:, j], rcond=None)[0]
                          for j in range(4)])
        return np.einsum('ncp,cp->nc', design, coeff) - raw, coeff

    freq = least_squares(lambda f: linear_fit(f)[0].ravel(), (2.4, 2.1),
                         bounds=(0.6, 5.0)).x
    return freq, linear_fit(freq)[1]


def _fit_bridge(loads, raw):
    clean = _clean_rows(loads, raw)
    freq, coeff = _profile(loads[clean], raw[clean])
    error = np.linalg.norm(_forward(loads[clean], freq, coeff) - raw[clean], axis=1)
    keep = error <= 6.0 * max(np.median(error), 1e-10)
    if not np.all(keep):
        freq, coeff = _profile(loads[clean][keep], raw[clean][keep])
    return freq, coeff


def _forward(loads, freq, coeff):
    b, a, s, c, d = coeff.T
    paired = loads[:, _PAIR]
    return b + a * loads + s * np.sin(freq[0] * loads) + c * paired - 2.0 * d * np.sin(0.5 * freq[1] * paired)**2


def _invert_bridge(raw, freq, coeff):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    b, a, s, c, d = coeff.T
    z = (y - b) / a
    # A Newton step is fast in the operating range. Use the contractive
    # fixed-point step whenever Newton does not improve the residual.
    contraction = (abs(s * freq[0]) + abs(c) + abs(d * freq[1])) / a
    if np.any(a <= 0) or np.any(contraction >= 1):
        raise ValueError('Fitted bridge is not diagonally dominant')
    for _ in range(600):
        residual = (_forward(z, freq, coeff) - y) / a
        error = np.max(np.abs(residual), axis=1)
        if np.all(error <= 1e-12):
            return z.reshape(raw.shape)
        diagonal = 1 + s * freq[0] / a * np.cos(freq[0] * z)
        cross = (c - d * freq[1] * np.sin(freq[1] * z[:, _PAIR])) / a
        det = diagonal * diagonal[:, _PAIR] - cross * cross[:, _PAIR]
        step = (diagonal[:, _PAIR] * residual - cross * residual[:, _PAIR]) / det
        trial = z - step
        trial_error = np.max(np.abs((_forward(trial, freq, coeff) - y) / a), axis=1)
        z = np.where((trial_error < 0.5 * error)[:, None], trial, z - residual)
    raise RuntimeError('Bridge inverse did not converge')


def calibrate_upstream(dataset: dict) -> dict:
    """Return canonical load coordinates without changing the input dataset."""
    cal = dataset['calibration_load_cells']
    loads = np.asarray(cal['true_channel_loads'], dtype=float)
    raw = np.asarray(cal['bridge_raw'], dtype=float)
    freq, coeff = _fit_bridge(loads, raw)
    result = dict(dataset)
    for name in ('train', 'test', 'chamber_verification_trials'):
        part = dataset.get(name)
        if part is not None:
            result[name] = dict(part)
            for key in _LOAD_KEYS:
                result[name][key] = _invert_bridge(part[key], freq, coeff)

    sweeps = cal['single_channel_sweeps']
    result['calibration_load_cells'] = dict(
        cal, bridge_raw=np.asarray(cal['true_channel_loads']).copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=np.asarray(sweeps['true_stress']).copy()),
    )
    return result
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
sed -n '1,125p' /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`595 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/upstream_transducer.py']`
- **Stdout (tail)**:
```text
sfinite(after)):
            return float(np.max(np.abs(after - before)))
        return None

    strain_diff = difference("predicted_strain")
    rate_diff = (difference("initial_rate_pred") if "initial_rate_pred" in out0 and "initial_rate_pred" in out1
                 else (None if "initial_rate_pred" in out0 or "initial_rate_pred" in out1 else 0.0))
    return {
        "valid": bool(swapped_rows and strain_diff is not None and rate_diff is not None
                      and max(strain_diff, rate_diff) <= float(tol)),
        "swapped_ref_rows_per_split": int(swapped_rows),
        "ref_swap_strain_max_diff": round(strain_diff, 10) if strain_diff is not None else None,
        "ref_swap_rate_max_diff": round(rate_diff, 10) if rate_diff is not None else None,
        "tol": float(tol),
    }

_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open(os.path.join(_d, "public_batch.pkl"), "rb") as fh:
    ds = pickle.load(fh)[0]["dataset"]
out = mod.solve_batch(copy.deepcopy(ds))
n_te = len(ds["test"]["lab_id"])
k_t = len(ds["eval_times"])
errs = []
coupon = None
if not isinstance(out, dict):
    errs.append("not_dict")
else:
    for k, shp in (("estimated_taus", (2,)), ("hereditary_weights", (7,)), ("predicted_strain", (n_te, k_t)), ("initial_rate_pred", (n_te,))):
        if k not in out:
            errs.append(f"missing:{k}")
            continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)):
            errs.append(f"bad:{k}:{a.shape}")
    if not errs:
        try:
            coupon = verify_alpha_pooled_ref_coupon_invariance(mod.solve_batch, ds, base_out=out)
            if not coupon["valid"]:
                errs.append("pooled_reference_coupon_invariance_failed")
        except (KeyError, TypeError, ValueError, IndexError):
            errs.append("invalid_output_on_reference_exchange")
print(json.dumps({"schema_valid": not errs, "errors": errs, "pooled_reference_check": coupon}, allow_nan=False))
sys.exit(1 if errs else 0)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a3e9737cb3031bc5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a3e9737c (directory).
```

### Turn 10 — `execute_in_remote_docker` (`35825 ms` LLM)
- **Rationale**: Test the exact artifact with fold-local spike rejection on real mixed-load and single-axis rows. Load-space errors will reveal whether inversion amplifies the calibration residuals; extended-range round trips and an explicit verification-trial fixture will catch convergence or routing failures the public schema check cannot detect.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, pickle, runpy, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up

ds = pickle.load(open('public_batch.pkl', 'rb'))[0]['dataset']
cal = ds['calibration_load_cells']
x, y = (cal[k] for k in ('true_channel_loads', 'bridge_raw'))
i = np.arange(len(x))
axis = np.count_nonzero(x, axis=1) == 1
# Identify the gross spikes independently of the candidate and use this mask only for scoring.
amp = np.linalg.norm(y, axis=1)
clean = amp < 5 * np.median(amp)
print('independent gross-spike rows:', i[~clean].tolist())
assert np.any(~clean)
for label, tests, group in [('mixed', [i % 5 == k for k in range(5)], ~axis),
                             ('single-axis leave-one-out', [i == k for k in i[axis]], axis)]:
    forward, inverse = [], []
    for test in tests:
        score = test & clean & group
        if not np.any(score):
            continue
        f, c = up._fit_bridge(x[~test], y[~test])
        forward.extend(up._forward(x[score], f, c) - y[score])
        inverse.extend(up._invert_bridge(y[score], f, c) - x[score])
    for name, error, limit in [('raw', forward, 1e-3), ('load', inverse, 3e-3)]:
        e = np.asarray(error)
        mx = np.max(np.abs(e))
        print(label, name, 'RMSE by channel', np.sqrt(np.mean(e**2, axis=0)), 'max abs', mx)
        assert mx < limit

f, c = up._fit_bridge(x, y)
print('fitted frequencies', f)
q = (abs(c[:, 2] * f[0]) + abs(c[:, 3]) + abs(c[:, 4] * f[1])) / c[:, 1]
print('global contraction bound', max(q))
assert np.all(q < 1)
rng = np.random.default_rng(27)
z = np.r_[rng.uniform(-3., 3., (10000, 4)), np.array(np.meshgrid(*[[-3., 0., 3.]] * 4)).reshape(4, -1).T]
r = up._forward(z, f, c)
err = np.max(np.abs(up._invert_bridge(r, f, c) - z))
print('extended-range inverse max abs', err)
assert err < 1e-10

original = copy.deepcopy(ds)
for verification in (None, {k: ds['test'][k][:2] for k in up._LOAD_KEYS}):
    item = dict(ds, chamber_verification_trials=verification)
    out = up.calibrate_upstream(item)
    if verification is None:
        assert out['chamber_verification_trials'] is None
    for name in ('train', 'test', 'chamber_verification_trials'):
        if item[name] is None:
            continue
        for key in up._LOAD_KEYS:
            raw, loads = item[name][key], out[name][key]
            assert raw.shape == loads.shape and not np.shares_memory(raw, loads)
            e = np.max(np.abs(up._forward(loads.reshape(-1, 4), f, c) - raw.reshape(-1, 4)))
            print(name, key, 'raw reconstruction max abs', e)
            assert e < 1e-10
            if name in original:
                np.testing.assert_array_equal(raw, original[name][key])
    for source, converted, target, canonical in [
        (cal, out['calibration_load_cells'], 'bridge_raw', 'true_channel_loads'),
        (cal['single_channel_sweeps'], out['calibration_load_cells']['single_channel_sweeps'], 'cmd_stress', 'true_stress')]:
        np.testing.assert_array_equal(converted[target], source[canonical])
        assert not np.shares_memory(converted[target], source[canonical])
np.testing.assert_array_equal(cal['bridge_raw'], original['calibration_load_cells']['bridge_raw'])
np.testing.assert_array_equal(cal['single_channel_sweeps']['cmd_stress'], original['calibration_load_cells']['single_channel_sweeps']['cmd_stress'])
sys.argv = ['/holdout_task/validate_public_schema.py', '/holdout_task/batch_solver.py']
runpy.run_path(sys.argv[0], run_name='__main__')
PY
```
- **Container Exit Code**: `0` (`716 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
independent gross-spike rows: [13, 14, 15]
mixed raw RMSE by channel [8.71327555e-05 1.10242817e-04 1.12042374e-04 1.07952514e-04] max abs 0.0003053469968561373
mixed load RMSE by channel [0.00020227 0.0003912  0.00022628 0.00032754] max abs 0.0015328307837667587
single-axis leave-one-out raw RMSE by channel [1.18005051e-04 1.57506751e-04 1.35922547e-04 9.03954325e-05] max abs 0.00035286948653778794
single-axis leave-one-out load RMSE by channel [1.20746314e-04 2.72677166e-04 1.33229918e-04 9.33394923e-05] max abs 0.0007726637530567348
fitted frequencies [2.49990923 2.00017663]
global contraction bound 0.9550399135676393
extended-range inverse max abs 3.1086244689504383e-15
train past_channel_raw raw reconstruction max abs 8.637535131583718e-14
train hold_channel_raw raw reconstruction max abs 1.2012613126444194e-12
test past_channel_raw raw reconstruction max abs 8.903988657493755e-14
test hold_channel_raw raw reconstruction max abs 4.440892098500626e-16
train past_channel_raw raw reconstruction max abs 8.637535131583718e-14
train hold_channel_raw raw reconstruction max abs 1.2012613126444194e-12
test past_channel_raw raw reconstruction max abs 8.903988657493755e-14
test hold_channel_raw raw reconstruction max abs 4.440892098500626e-16
chamber_verification_trials past_channel_raw raw reconstruction max abs 1.7763568394002505e-15
chamber_verification_trials hold_channel_raw raw reconstruction max abs 7.116529587847253e-14
{"schema_valid": true, "errors": [], "pooled_reference_check": {"valid": true, "swapped_ref_rows_per_split": 24, "ref_swap_strain_max_diff": 3e-10, "ref_swap_rate_max_diff": 2e-10, "tol": 1e-05}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a3e9737cb3031bc5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a3e9737c (directory).
```

### Turn 11 — `submit_completed_task` (`12585 ms` LLM)
- **Rationale**: The exact submitted bundle passed isolated verification: fold-local spike rejection and load-space validation on both measured slices, extended-range inversion, trial routing, identity normalization, and the composed public checks. No further edits are needed.
- **Submitted Final Answer / Summary**: `Implemented robust four-channel calibration in `/holdout_task/upstream_transducer.py` and verified its integration with `batch_solver.py`. The public schema and reference-swap checks passed. The maximum absolute load error on held-out calibration rows was 0.00153.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
