# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_beta_m99_00_batch_03__blind_continual_campaign__upstream_transducer`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M99-Autonomous-Decomposition-And-Interaction-Withdrawal-World` |
| **Domain** | `coupled_2d_transport_reaction_box_simplex_kkt` |
| **Wave / Wormhole Family** | `m9_9_autonomous_decomposition_and_interaction_withdrawal` / `m99_beta` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `10 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `271,828` (`262,892` / `101,376` / `8,936`) |
| **Total Episode Wall Time (s)** | `263.14s` |

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

### Turn 1 — `execute_in_remote_docker` (`17160 ms` LLM)
- **Rationale**: Inspect actual array shapes, the downstream input contract, and the apparatus log before choosing the calibration model. These checks determine which fields to transform and which sweep slices can test inversion.

```bash
cd /holdout_task && cat batch_solver.py validate_public_schema.py public_asset_index.json && sed -n '1,220p' _verified_downstream.py && python3 - <<'PY'
import pickle
import numpy as np
with open('public_batch.pkl','rb') as f: b=pickle.load(f)
def describe(x,d=0):
 if isinstance(x,dict): return {k:describe(v,d+1) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return (type(x).__name__,len(x),describe(x[0],d+1) if x and d<4 else None)
 if isinstance(x,np.ndarray): return (x.shape,str(x.dtype), (float(np.nanmin(x)),float(np.nanmax(x))) if x.size and x.dtype.kind in 'fiu' else None)
 return x if isinstance(x,(int,float,str)) else type(x).__name__
print(describe(b))
PY
ls -lh /tmp/hegel_continual_archive; tail -c 7000 /tmp/hegel_continual_archive/archive_apparatus_log.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust Cauchy IRLS nonlinear calibration outlier regression'
```
- **Container Exit Code**: `0` (`398 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
],
      [
        0.39506,
        0.64935,
        -0.35511,
        0.75063,
        -0.93078,
        -0.33535,
        -0.25026,
        0.00755
      ],
      [
        0.14009,
        0.26814,
        -0.4356,
        0.76993,
        -0.81877,
        -0.29051,
        -0.51927,
        -0.43007
      ],
      [
        -0.02245,
        -0.2061,
        -0.38403,
        0.6821,
        -0.55314,
        -0.28427,
        -0.59136,
        -0.71118
      ],
      [
        -0.23452,
        -0.54973,
        -0.3688,
        0.52744,
        -0.09918,
        -0.13127,
        -0.70406,
        -0.80535
      ],
      [
        -0.4586,
        -0.82612,
        -0.19234,
        0.34565,
        0.33914,
        -0.27629,
        -0.85164,
        -0.66976
      ],
      [
        -0.67919,
        -0.81361,
        -0.49343,
        0.16122,
        0.62032,
        -0.33256,
        -0.85841,
        -0.42074
      ],
      [
        -0.77302,
        -0.66301,
        -0.35702,
        -0.07286,
        0.80704,
        -0.40027,
        -0.7787,
        0.02752
      ],
      [
        -0.84656,
        -0.291,
        -0.40566,
        -0.35331,
        0.80349,
        -0.37849,
        -0.74986,
        0.47069
      ],
      [
        -0.84367,
        0.18257,
        -0.35915,
        -0.47602,
        0.51133,
        -0.3519,
        -0.61243,
        0.73489
      ]
    ]
  }
]autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`19101 ms` LLM)
- **Rationale**: The first output was truncated. Recover the field layout and compare sweep coverage with observation coverage to see whether inversion requires extrapolation. Affine residual quantiles will set the baseline for separating saturation from EMI spikes.

```bash
cd /holdout_task && cat _verified_downstream.py && python3 - <<'PY'
import json,pickle,numpy as np
b=pickle.load(open('public_batch.pkl','rb')); print('items',len(b))
for item in b:
 s=item['study']; c=s['sensor_calibration_sweeps']; print('study keys',list(s),'calibration keys',list(c)); print('single bank',type(c.get('single_bank_sweeps')),str(c.get('single_bank_sweeps'))[:400]); x=c['ref_rosette']; y=c['raw_rosette']; print('ref/raw shapes',x.shape,y.shape); print('ranges',np.min(x,0),np.max(x,0),np.min(y,0),np.max(y,0)); print('first refs',x[:8]); print('obs',s['y_obs_raw'].shape,np.percentile(s['y_obs_raw'],[0,5,50,95,100])); p=s.get('reference_tracer_pulses'); print('pulses',None if p is None else [(list(v), np.shape(v.get('y_pulse_raw'))) for v in p]); a=np.c_[np.ones(len(x)),x]; w=np.linalg.lstsq(a,y,rcond=None)[0]; print('affine',w); print('row residual percentiles',np.percentile(np.linalg.norm(a@w-y,axis=1),[0,25,50,75,90,95,100]))
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('apparatus summaries');
for r in log:
 print({k:v for k,v in r.items() if len(str(v))<500},'other keys',[k for k,v in r.items() if len(str(v))>=500])
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_robust_network_assimilation.assimilate_diffusive_network | head -30
```
- **Container Exit Code**: `0` (`414 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.4]), 'raw_readout': array([-1.62127838, -0.73136368, -0.04721868,  0.60262904,  1.38964756,
       16.98528235, -1.14167355,
ref/raw shapes (80, 4) (80, 4)
ranges [-1.28100358 -1.37940096 -1.2        -1.3791615 ] [1.78321946 1.76593043 1.6093786  1.76450291] [ -2.25388534 -12.87000811  -1.97209973 -13.61745535] [13.66128281  2.58698568 12.97998919  2.97950579]
first refs [[-1.2  0.   0.   0. ]
 [-0.4  0.   0.   0. ]
 [ 0.5  0.   0.   0. ]
 [ 1.3  0.   0.   0. ]
 [ 0.  -1.2  0.   0. ]
 [ 0.  -0.4  0.   0. ]
 [ 0.   0.5  0.   0. ]
 [ 0.   1.3  0.   0. ]]
obs (96,) [-0.09222058 -0.0672597   0.07701422  0.24960514  0.33641092]
pulses None
affine [[ 6.57669751e-01 -5.58925739e-01  5.67999781e-01 -5.83568962e-01]
 [ 1.70078792e+00 -2.70421897e-02 -6.98864638e-02  3.56018132e-01]
 [ 3.67414140e-01  1.17150169e+00 -2.98528437e-01  1.13059620e-01]
 [-6.81523490e-02  4.29836574e-01  1.47167786e+00  5.04139727e-03]
 [-8.77553877e-04 -1.36514864e-01  6.49044603e-01  8.84271212e-01]]
row residual percentiles [ 0.28229139  0.73918323  1.06399739  1.41328169  1.748981    2.16138902
 25.03160258]
apparatus summaries
{'completed_campaign_index': 0, 'completed_campaign_short_name': 'facility_init'} other keys ['facility_telemetry_trace']
{'completed_campaign_index': 1, 'completed_campaign_short_name': 'dev_beta_m99_00_s00'} other keys ['facility_telemetry_trace']
{'completed_campaign_index': 2, 'completed_campaign_short_name': 'dev_beta_m99_00_batch_01'} other keys ['facility_telemetry_trace']
{'completed_campaign_index': 3, 'completed_campaign_short_name': 'dev_beta_m99_00_batch_02'} other keys ['facility_telemetry_trace']
{
  "ok": true,
  "target": "autogenous_robust_network_assimilation.assimilate_diffusive_network",
  "access_mode": "executable",
  "execution_available": true,
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
```

### Turn 3 — `execute_in_remote_docker` (`27646 ms` LLM)
- **Rationale**: Compare robust fits on identical held-out sweeps to separate reproducible nonlinear structure from EMI. Check single-bank residuals separately before deciding whether those measurements can improve the rosette fit.
- **Alternatives Weighed**: `Fit a high-degree inverse directly to raw sweeps` (rejected: EMI spikes would become extreme predictor values; forward regression keeps the reference predictors clean.)

```bash
cd /holdout_task; grep -nEi 'rosette|satur|tanh|calibrat|cubic' /tmp/hegel_continual_archive/*solver.py | cut -c1-220 | head -60; cat > /tmp/rosette_probe.py <<'PY'
import pickle,itertools,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; b=c['single_bank_sweeps']; xb=np.eye(4)[b['bank_id']]*b['ref_signal'][:,None]
def fit(a,y):
 w=np.linalg.lstsq(a,y,rcond=None)[0]
 for _ in range(20):
  e=np.linalg.norm(y-a@w,axis=1); q=np.minimum(1,(max(np.median(e),1e-12)*1.8/np.maximum(e,1e-15))**2)**.5
  w=np.linalg.lstsq(a*q[:,None],y*q[:,None],rcond=None)[0]
 return w
def poly(x,maxdeg):
 return np.column_stack([np.ones(len(x))]+[x[:,idx].prod(axis=1) for d in range(1,maxdeg+1) for idx in itertools.combinations_with_replacement(range(4),d)])
models={'linear':lambda x:poly(x,1),'quadratic':lambda x:poly(x,2),'diag23':lambda x:np.c_[poly(x,1),x*x,x**3],'quad+cubics':lambda x:np.c_[poly(x,2),x**3],'full3':lambda x:poly(x,3),'quad+tanh':lambda x:np.c_[poly(x,2),np.tanh(x)],'quad+rational':lambda x:np.c_[poly(x,2),x/(1+x*x)],'quad+sin':lambda x:np.c_[poly(x,2),np.sin(x)],'quad+cubics+5':lambda x:np.c_[poly(x,2),x**3,x**5]}
np.set_printoptions(precision=5,suppress=True)
for name,f in models.items():
 a=f(x); w=fit(a,y); e=np.linalg.norm(a@w-y,axis=1); pred=np.empty_like(y)
 for te in np.array_split(np.random.default_rng(12).permutation(len(y)),5):
  tr=np.setdiff1d(np.arange(len(y)),te); pred[te]=a[te]@fit(a[tr],y[tr]);
 cv=np.linalg.norm(pred-y,axis=1); eb=(f(xb)@w)[np.arange(len(xb)),b['bank_id']]-b['raw_readout']; print(name, 'train',np.percentile(e,[25,50,90,97,100]), 'cv',np.percentile(cv,[25,50,90,97,100]),'bank',np.percentile(abs(eb),[50,90,100]));
 if name in ('quad+cubics','quad+tanh','full3'): print('coeffs',w.round(5)); print('top rows',np.argsort(e)[-8:])
print('axes ref/raw',np.c_[x[:16],y[:16]]); print('bank raw',b['raw_readout'].reshape(4,-1))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rosette_probe.py
```
- **Container Exit Code**: `0` (`484 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0246  0.0047   0.18256]
 [ 0.00124  0.0136  -0.00244 -0.00075]
 [ 0.00827  0.17487 -0.00327 -0.00408]
 [ 0.00124  0.00608  0.0003   0.0037 ]
 [ 0.00212 -0.0029   0.02216  0.00612]
 [-0.00024  0.00078  0.27286 -0.00553]
 [-0.00465 -0.00136  0.00125  0.01665]
 [-1.46804  0.03347  0.004    0.01287]
 [-0.00566 -0.93217  0.03587  0.00532]
 [ 0.0201   0.01197 -1.4087  -0.00396]
 [-0.02434 -0.04567  0.00325 -1.03234]]
top rows [60 72 70 45 17 14 13 15]
quad+rational train [ 0.04328  0.05546  0.10486 16.44155 26.01056] cv [ 0.05539  0.0833   0.18048 16.4728  26.01031] bank [ 0.02165  0.0507  14.48183]
quad+sin train [ 0.00587  0.00772  0.0186  16.3698  25.9689 ] cv [ 0.0074   0.01086  0.02884 16.37666 25.96843] bank [ 0.00274  0.00736 14.49313]
quad+cubics+5 train [ 0.00001  0.00002  0.00003 16.35686 25.96326] cv [ 0.00002  0.00003  0.00005 16.35686 25.96326] bank [ 0.00001  0.00002 14.50001]
axes ref/raw [[ -1.2       0.        0.        0.       -1.83902   0.08358   0.06
   -0.39488]
 [ -0.4       0.        0.        0.       -0.44265  -0.01879   0.06001
   -0.18498]
 [  0.5       0.        0.        0.        0.74358  -0.134     0.05999
    0.05123]
 [  1.3       0.        0.        0.        2.23324  -0.23641   0.05998
    0.26114]
 [  0.       -1.2       0.        0.       -0.40768  -1.27883   0.3
   -0.08   ]
 [  0.       -0.4       0.        0.       -0.08256  -0.39923   0.13999
   -0.08001]
 [  0.        0.5       0.        0.        0.2832    0.34801  -0.04
   -0.08002]
 [  0.        1.3       0.        0.        0.60831   1.28635  -0.19999
   -0.07999]
 [  0.        0.       -1.2       0.        0.08001  -0.3772   -1.82879
    0.07743]
 [  0.        0.       -0.4       0.        0.07999  -0.1724   -0.45439
   -0.02751]
 [  0.        0.        0.5       0.        0.07999   0.05799   0.71316
   -0.14561]
 [  0.        0.        1.3       0.        0.08      0.2628    2.17932
   -0.25054]
 [  0.        0.        0.       -1.2       0.32383  -0.07002  -0.41998
   -1.31905]
 [  0.        0.        0.       -0.4      13.66128 -12.87001  12.30001
  -13.61746]
 [  0.        0.        0.        0.5      13.4784  -12.86999  12.66001
  -12.85156]
 [  0.        0.        0.        1.3      13.31584 -12.87001  12.97999
  -11.88973]]
bank raw [[-1.62128 -0.73136 -0.04722  0.60263  1.38965 16.98528]
 [-1.14167 -0.5811  -0.15014  0.25922  0.75498  1.44514]
 [-1.61447 -0.73859 -0.06521  0.5744   1.34904  2.42739]
 [-1.17844 -0.60389 -0.16214  0.25744  0.7656   1.47302]]
```

### Turn 4 — `execute_in_remote_docker` (`39408 ms` LLM)
- **Rationale**: Implement the smallest tested basis that reaches the sweep noise floor. Run the public schema gate now to detect integration failures before checking inverse error on held-out sweeps and the pulse path.
- **Alternatives Weighed**: `Use a full cubic polynomial` (rejected: Mixed cubic terms were near the measurement noise floor; pure cubics already explain the independent bank readings at that floor.); `Fit a smooth saturating function` (rejected: Tested tanh and rational bases left held-out residuals orders of magnitude larger.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine or quadratic forward map accounts for rosette coupling once EMI rows are downweighted.` | Witness: `Under the same robust fit and folds, adding pure channel cubics lowered the 90th-percentile held-out forward error from 0.412 to about 4e-5; independent single-bank median error also dropped from 0.076 to about 1e-5.` | Blanket Diagnosis: `Cross-channel quadratic interactions and repeatable single-channel cubic curvature explain the clean readings; the remaining large joint residuals are isolated EMI rows.` | Sublated Pivot: `Retain all quadratic coupling terms, add the four pure cubics, reject joint sweep outliers, and numerically invert the fitted forward map.`

```bash
cd /holdout_task && cat > upstream_transducer.py <<'PY'
"""Robust four-channel calibration for the chamber rosette."""

import numpy as np


_I, _J = np.triu_indices(4)
_EYE = np.eye(4)


def _basis(x):
    return np.column_stack((np.ones(len(x)), x, x[:, _I] * x[:, _J], x**3))


def _fit(ref, raw):
    a = _basis(ref)
    coef = np.linalg.lstsq(a, raw, rcond=None)[0]
    # Weight whole rows: the EMI bursts affect multiple channels together.
    for _ in range(25):
        error = np.linalg.norm(a @ coef - raw, axis=1)
        scale = max(1.8 * np.median(error), 1e-12)
        weight = 1 / np.hypot(1, error / scale)
        coef = np.linalg.lstsq(a * weight[:, None], raw * weight[:, None], rcond=None)[0]
    error = np.linalg.norm(a @ coef - raw, axis=1)
    clean = error <= max(8 * np.median(error), 1e-12)
    if np.linalg.matrix_rank(a[clean]) == a.shape[1]:
        coef = np.linalg.lstsq(a[clean], raw[clean], rcond=None)[0]
    return coef


def _jacobian(x, coef):
    q_derivative = (x[:, _J, None] * _EYE[_I] + x[:, _I, None] * _EYE[_J])
    q = np.einsum('nki,kj->nji', q_derivative, coef[5:15])
    return coef[1:5].T + q + 3 * x[:, None, :]**2 * coef[15:].T


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    if not len(y):
        return raw.copy()
    x = np.linalg.solve(coef[1:5].T, (y - coef[0]).T).T
    residual = _basis(x) @ coef - y
    for _ in range(30):
        error = np.linalg.norm(residual, axis=1)
        active = error > 2e-13 * (1 + np.linalg.norm(y, axis=1))
        if not active.any():
            break
        xa, ya = x[active], y[active]
        delta = np.linalg.solve(_jacobian(xa, coef), residual[active, :, None])[..., 0]
        step = np.ones((len(xa), 1))
        for _ in range(16):
            trial = xa - step * delta
            change = _basis(trial) @ coef - ya
            worse = np.linalg.norm(change, axis=1) > error[active]
            if not worse.any():
                break
            step[worse] *= 0.5
        x[active] = np.where(worse[:, None], xa, trial)
        residual[active] = np.where(worse[:, None], residual[active], change)
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a study in reference coordinates without changing the input."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    coef = _fit(ref, np.asarray(sweeps['raw_rosette']))
    out = dict(study, y_obs_raw=_invert(study['y_obs_raw'], coef))
    normalized = dict(sweeps, raw_rosette=ref.copy())
    bank = sweeps['single_bank_sweeps']
    normalized['single_bank_sweeps'] = dict(bank, raw_readout=bank['ref_signal'].copy())
    out['sensor_calibration_sweeps'] = normalized
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef))
                                          for pulse in pulses]
    return out
PY
python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
sed -n '79,155p' _verified_downstream.py
```
- **Container Exit Code**: `0` (`370 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/upstream_transducer.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
        violation = side * (H @ w - c - nu)
        release = violation.argmax()
        if violation[release] <= tol:
            return w, float(nu)
        side[release] = 0


def solve_downstream(cal_study: dict) -> dict:
    S, terminal_modes = _forward(cal_study)
    y_cal = cal_study['y_obs_raw']
    Z = np.asarray(cal_study['Z_nuisance'])
    Pz = solve(Z.T @ Z + cal_study['nuisance_ridge'] * np.eye(Z.shape[1]),
               Z.T, assume_a='pos')
    S = S - Z @ (Pz @ S)
    y = y_cal - Z @ (Pz @ y_cal)
    ridge, cap, budget = (cal_study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(S.T @ S + ridge * np.eye(S.shape[1]), S.T @ y, cap, budget)
    residual = S @ w - y
    objective = float((residual @ residual + ridge * (w @ w)) / 2)
    g = S.T @ residual + ridge * w
    lower, upper = np.maximum(0, g - nu), np.maximum(0, nu - g)
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
            'primal_objective': objective,
            'dual_lower_bound': float(objective - g @ w + nu * budget - cap * upper.sum()),
        },
    }

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c9b5361fdd197f52 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c9b5361f (directory).
```

### Turn 5 — `execute_in_remote_docker` (`34742 ms` LLM)
- **Rationale**: Hold out the entire axis sweep and contiguous mixed-sweep blocks to detect inverse errors hidden by random folds. Exercise the pulse path, which the public item omits, and verify that normalization leaves the input intact and the composed solver preserves budget feasibility and the KKT gap.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up
from batch_solver import solve_batch

with open('public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    s = item['study']
    c = s['sensor_calibration_sweeps']
    x, y = c['ref_rosette'], c['raw_rosette']
    full = up._fit(x, y)
    base_error = np.linalg.norm(up._basis(x) @ full - y, axis=1)
    clean = base_error < 0.01
    assert clean.sum() >= x.shape[0] * .8
    print('EMI rows:', np.flatnonzero(~clean).tolist())
    folds = [('axis sweeps', np.flatnonzero(np.count_nonzero(x, axis=1) == 1))]
    mixed = np.flatnonzero(np.count_nonzero(x, axis=1) != 1)
    folds.extend((f'mixed fold {i}', rows) for i, rows in enumerate(np.array_split(mixed, 5), 1))
    worst = 0.
    for name, test in folds:
        train = np.setdiff1d(np.arange(len(x)), test)
        coef = up._fit(x[train], y[train])
        keep = test[clean[test]]
        err = np.linalg.norm(up._invert(y[keep], coef) - x[keep], axis=1)
        maximum = float(err.max())
        worst = max(worst, maximum)
        print(name, 'n=', len(keep), 'max reference-space error=', maximum)
    assert worst < 2e-4, worst
    first = np.flatnonzero(clean)[:8]
    bank = c['single_bank_sweeps']
    bank_x = np.eye(4)[bank['bank_id']] * bank['ref_signal'][:, None]
    bank_error = np.abs((up._basis(bank_x) @ full)[np.arange(len(bank_x)), bank['bank_id']] - bank['raw_readout'])
    print('independent bank clean max forward error:', bank_error[bank_error < 0.01].max(), 'spike count:', (bank_error >= .01).sum())
    snapshots = [a.copy() for a in (s['y_obs_raw'], y, bank['raw_readout'])]
    pulse = y[first].reshape(2, 4, 4).copy()
    extra = dict(s, reference_tracer_pulses=[{'y_pulse_raw': pulse, 'label': 'shape test'}])
    cal = up.calibrate_upstream(extra)
    np.testing.assert_allclose(cal['reference_tracer_pulses'][0]['y_pulse_raw'], x[first].reshape(pulse.shape), atol=2e-4, rtol=0)
    assert cal['reference_tracer_pulses'][0]['label'] == 'shape test'
    normalized = cal['sensor_calibration_sweeps']
    np.testing.assert_array_equal(normalized['raw_rosette'], x)
    np.testing.assert_array_equal(normalized['single_bank_sweeps']['raw_readout'], bank['ref_signal'])
    assert not np.shares_memory(normalized['raw_rosette'], x)
    for current, old in zip((s['y_obs_raw'], y, bank['raw_readout']), snapshots):
        np.testing.assert_array_equal(current, old)
    reconstructed = up._basis(cal['y_obs_raw'].reshape(-1, 4)) @ full
    np.testing.assert_allclose(reconstructed.ravel(), s['y_obs_raw'], atol=1e-11, rtol=0)
    absent = up.calibrate_upstream(s)
    assert absent.get('reference_tracer_pulses') is None
    out = solve_batch(s)
    w = out['weights']
    kkt = out['kkt_certificate']
    assert np.all(np.isfinite(w)) and np.all(np.isfinite(out['terminal_field']))
    assert np.all(w >= -1e-10) and np.all(w <= s['u_box'] + 1e-10)
    gap = kkt['primal_objective'] - kkt['dual_lower_bound']
    assert abs(w.sum() - s['total_budget']) < 1e-10 and abs(gap) < 1e-9
    print('budget residual:', abs(w.sum() - s['total_budget']), 'KKT gap:', gap)
PY
python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`571 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
EMI rows: [13, 14, 15]
axis sweeps n= 13 max reference-space error= 4.343243009217791e-05
mixed fold 1 n= 13 max reference-space error= 3.206183387653559e-05
mixed fold 2 n= 13 max reference-space error= 3.237003185556103e-05
mixed fold 3 n= 13 max reference-space error= 5.95425386814992e-05
mixed fold 4 n= 13 max reference-space error= 2.9403593072405625e-05
mixed fold 5 n= 12 max reference-space error= 2.9145941965962435e-05
independent bank clean max forward error: 1.733899773703751e-05 spike count: 1
budget residual: 0.0 KKT gap: 9.367506770274758e-17
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c9b5361fdd197f52 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c9b5361f (directory).
```

### Turn 6 — `submit_completed_task` (`12152 ms` LLM)
- **Rationale**: The final bundle passed isolated execution, grouped sweep holdouts, pulse conversion, and non-mutating normalization. The composed solver had zero budget residual and a KKT gap of 9.37 × 10⁻¹⁷.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:204', '/holdout_task/upstream_transducer.py': 'FILE:2964'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented robust four-channel calibration in `/holdout_task/upstream_transducer.py` and verified it through `/holdout_task/batch_solver.py`. The public schema check passed, and the maximum held-out inverse error on clean sweeps was 5.95 × 10⁻⁵ in reference coordinates.`

### Turn 7 — `execute_in_remote_docker` (`32832 ms` LLM)
- **Rationale**: Hold out the highest-amplitude measured sweeps to detect extrapolation bias. Revisit the basis only if a rational alternative improves a meaningful error margin; check conditioning and round-trip accuracy on actual observations to distinguish calibration bias from numerical inversion failure.
- **Alternatives Weighed**: `Add relay-mode, drift-operator, or batch-classification tests` (rejected: The interface is a continuous rosette calibration and exposes no relay modes or classification labels; the verified downstream operator is unchanged.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up
from batch_solver import solve_batch

with open('public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    s = item['study']
    sweeps = s['sensor_calibration_sweeps']
    x, y = (sweeps[k] for k in ('ref_rosette', 'raw_rosette'))
    a = up._basis(x)
    # Both signs of a pairwise quadratic contrast must be spanned exactly.
    for i, j in zip(*np.triu_indices(4, 1)):
        for c in (-0.6, 0.6):
            diff = (x[:, i] + c*x[:, j])**2 - (x[:, i]**2 + 2*c*x[:, i]*x[:, j] + c*c*x[:, j]**2)
            assert np.max(abs(diff)) < 1e-14
    def robust(a, y):
        w = np.linalg.lstsq(a, y, rcond=None)[0]
        for _ in range(25):
            e = np.linalg.norm(a@w-y, axis=1)
            q = 1/np.hypot(1, e/max(1.8*np.median(e), 1e-12))
            w = np.linalg.lstsq(a*q[:, None], y*q[:, None], rcond=None)[0]
        return w
    print('study keys:', sorted(s))
    # An amplitude-tail holdout can reveal extrapolation bias hidden by random folds.
    for label, rank in [('joint amplitude', np.linalg.norm(x, axis=1)), ('channel peak', np.max(abs(x), axis=1))]:
        test = np.argsort(rank)[-16:]
        train = np.setdiff1d(np.arange(len(x)), test)
        coef = up._fit(x[train], y[train])
        forward = np.linalg.norm(a[test]@coef-y[test], axis=1)
        clean = forward < .01
        assert clean.sum() >= 12
        inverse = np.linalg.norm(up._invert(y[test][clean], coef)-x[test][clean], axis=1).max()
        rational = {}
        for scale in (.01, .1, 1.):
            basis = np.c_[a[:, :15], x**3/(1+scale*x*x)]
            w = robust(basis[train], y[train])
            rational[scale] = np.linalg.norm(basis[test][clean]@w-y[test][clean], axis=1).max()
        print(label, 'clean n', int(clean.sum()), 'cubic forward max', forward[clean].max(), 'cubic inverse max', inverse, 'rational forward maxima by scale', rational)
        assert inverse < 2e-4, (label, inverse)
    cal = up.calibrate_upstream(s)
    z = cal['y_obs_raw'].reshape(-1, 4)
    coef = up._fit(x, y)
    j = up._jacobian(z, coef)
    print('observation reference ranges:', z.min(axis=0), z.max(axis=0), 'min Jacobian singular value:', np.linalg.svd(j, compute_uv=False)[:, -1].min())
    np.testing.assert_allclose(up._basis(z)@coef, s['y_obs_raw'].reshape(-1, 4), rtol=0, atol=1e-11)
    out = solve_batch(s)
    w = out['weights']
    k = out['kkt_certificate']
    gap = abs(k['primal_objective'] - k['dual_lower_bound'])
    assert np.all(np.isfinite(out['terminal_field'])) and np.all(w >= -1e-10) and np.all(w <= s['u_box'] + 1e-10)
    assert abs(w.sum()-s['total_budget']) < 1e-10 and gap < 1e-9
    print('KKT gap:', gap)
PY
python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`611 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
study keys: ['Lx', 'Ly', 'V_reaction_2d', 'Z_nuisance', 'aux_manifold_telemetry', 'kappa_x', 'kappa_y', 'mu_shifts', 'nuisance_ridge', 'nx', 'ny', 'robin_x', 'robin_y', 'row_bank_id', 'sensor_calibration_sweeps', 'sensor_ij', 'source_modes', 'source_ridge', 'total_budget', 'u_box', 'vel_x', 'vel_y', 'y_obs_raw']
joint amplitude clean n 16 cubic forward max 5.1221663220288736e-05 cubic inverse max 3.595658372044112e-05 rational forward maxima by scale {0.01: 0.018505520622313195, 0.1: 0.15298249265759814, 1.0: 0.5916902177177038}
channel peak clean n 16 cubic forward max 5.2082588673735145e-05 cubic inverse max 4.090943226890216e-05 rational forward maxima by scale {0.01: 0.016650176401561346, 0.1: 0.1374260805590491, 1.0: 0.553796450357691}
observation reference ranges: [-0.05968197 -0.02013955  0.00718792 -0.02301393] [0.14487658 0.19688435 0.17623335 0.23610576] min Jacobian singular value: 0.7562376626219548
KKT gap: 9.367506770274758e-17
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c9b5361fdd197f52 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c9b5361f (directory).
```

### Turn 8 — `submit_completed_task` (`13609 ms` LLM)
- **Rationale**: The unchanged bundle passed isolated execution, grouped and high-amplitude sweep holdouts, pulse conversion, and non-mutating normalization. The composed solver had zero budget residual and a KKT gap of 9.37 × 10⁻¹⁷.
- **Alternatives Weighed**: `Replace the cubic terms with rational saturation terms` (rejected: The best tested rational basis had a maximum forward error above 0.016 on each amplitude-tail slice, versus about 5.2 × 10⁻⁵ for the cubic basis.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:204', '/holdout_task/upstream_transducer.py': 'FILE:2964'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented robust four-channel rosette calibration in `/holdout_task/upstream_transducer.py` and verified it through `/holdout_task/batch_solver.py`. The public schema passed. The maximum held-out inverse error on clean sweeps was 5.95 × 10⁻⁵ in reference coordinates.`

### Turn 9 — `execute_in_remote_docker` (`30515 ms` LLM)
- **Rationale**: The remaining relevant failure is an EMI-sensitive fit or a channel-specific error hidden by pooled metrics. Compare the delivered estimate with an EMI-deleted fit on real observations and report each independent bank separately. A material shift would warrant revisiting row rejection; otherwise keep the verified artifacts unchanged.
- **Alternatives Weighed**: `Repeat the rational-basis, relay, and classification checks` (rejected: The amplitude-tail comparison already ruled out the tested rational terms. The calibration interface has neither relay modes nor class labels.)

```bash
cd /holdout_task && cat > /tmp/rosette_margin_audit.py <<'PY'
import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up
from batch_solver import solve_batch

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    study = item['study']
    sweeps = study['sensor_calibration_sweeps']
    x, y = (sweeps[k] for k in ('ref_rosette', 'raw_rosette'))
    full = up._fit(x, y)
    clean = np.linalg.norm(up._basis(x) @ full - y, axis=1) < 0.01
    bank = sweeps['single_bank_sweeps']
    ids = np.asarray(bank['bank_id'])
    bx = np.eye(4)[ids] * np.asarray(bank['ref_signal'])[:, None]
    predicted = (up._basis(bx) @ full)[np.arange(len(bx)), ids]
    err = abs(predicted - bank['raw_readout'])
    maxima = [float(err[(ids == i) & (err < .01)].max()) for i in range(4)]
    print('Independent clean single-bank max error by channel:', maxima)
    assert max(maxima) < 2e-4
    # Compare the actual delivered inverse with the same fit after removing the observed EMI rows.
    clean_fit = up._fit(x[clean], y[clean])
    obs = study['y_obs_raw']
    shift = float(np.max(abs(up._invert(obs, full) - up._invert(obs, clean_fit))))
    print('Observation coordinate shift after removing measured EMI rows:', shift)
    assert shift < 2e-5
    # Signed channel bias and curvature omission are evaluated on the same held-out rows.
    rows = np.flatnonzero(clean)
    folds = np.array_split(np.random.default_rng(18).permutation(rows), 5)
    errors, quad_errors = [], []
    a = up._basis(x)
    for test in folds:
        train = np.setdiff1d(rows, test)
        coef = up._fit(x[train], y[train])
        errors.append(up._invert(y[test], coef) - x[test])
        q = np.linalg.lstsq(a[train, :15], y[train], rcond=None)[0]
        quad_errors.extend(np.linalg.norm(a[test, :15] @ q - y[test], axis=1))
    e = np.concatenate(errors)
    worst = float(np.linalg.norm(e, axis=1).max())
    q90 = float(np.percentile(quad_errors, 90))
    print('ng_t04 comparison: quadratic p90 forward error:', q90, 'cubic worst inverse error:', worst, 'mean signed error by channel:', e.mean(axis=0).tolist())
    assert worst < 2e-4 and q90 > 10 * np.percentile(np.linalg.norm(e, axis=1), 90)
    out = solve_batch(study)
    cert = out['kkt_certificate']
    gap = abs(cert['primal_objective'] - cert['dual_lower_bound'])
    assert np.isfinite(out['terminal_field']).all() and gap < 1e-9
    print('Isolated composed solver KKT gap:', gap)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rosette_margin_audit.py && python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `0` (`568 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Independent clean single-bank max error by channel: [1.733899773703751e-05, 8.537949711007187e-06, 1.453181808182269e-05, 1.56929932129124e-05]
Observation coordinate shift after removing measured EMI rows: 0.0
ng_t04 comparison: quadratic p90 forward error: 0.4161062701318886 cubic worst inverse error: 6.655751606243713e-05 mean signed error by channel: [-3.8332746432717414e-07, -7.702748802503144e-08, -6.625563773956318e-07, -5.300727406008374e-07]
Isolated composed solver KKT gap: 9.367506770274758e-17
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c9b5361fdd197f52 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c9b5361f (directory).
```

### Turn 10 — `submit_completed_task` (`14466 ms` LLM)
- **Rationale**: The unchanged artifacts passed isolated execution, grouped and high-amplitude holdouts, pulse conversion, and non-mutating normalization. Independent bank measurements agreed within 1.74 × 10⁻⁵ after excluding the EMI spike. The composed solver had zero budget residual and a KKT gap of 9.37 × 10⁻¹⁷.
- **Alternatives Weighed**: `Replace the pure cubics with rational saturation terms` (rejected: The best tested rational basis exceeded 0.016 maximum forward error on both high-amplitude slices, versus about 5.2 × 10⁻⁵ for the cubic basis.)
- **Submitted Final Answer / Summary**: `Implemented robust four-channel rosette calibration in `/holdout_task/upstream_transducer.py` and verified it through `/holdout_task/batch_solver.py`. The public schema passed. Across the held-out clean sweeps, the largest inverse error was 6.66 × 10⁻⁵ in reference coordinates. Removing the identified EMI rows did not change the calibrated observations.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M99-Autonomous-Decomposition-And-Interaction-Withdrawal-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
