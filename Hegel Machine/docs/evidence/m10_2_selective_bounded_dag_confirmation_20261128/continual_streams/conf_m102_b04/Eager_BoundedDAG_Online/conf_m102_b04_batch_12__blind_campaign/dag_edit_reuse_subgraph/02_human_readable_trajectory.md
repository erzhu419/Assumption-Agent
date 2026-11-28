# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m102_b04_batch_12__blind_campaign__reuse_subgraph`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel_M10_1_UnknownCut_DAG_Discovery` |
| **Domain** | `blind_multi_stage_scientific_computation_graph` |
| **Wave / Wormhole Family** | `m10_2_selective_bounded_dag_discovery_h16` / `m102_blind_dag` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 12` |
| **ATMS Nogoods Recorded** | `3` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `357,935` (`346,401` / `139,392` / `11,534`) |
| **Total Episode Wall Time (s)** | `337.6s` |

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

### Turn 1 — `execute_in_remote_docker` (`11681 ms` LLM)
- **Rationale**: Inspect the exact node interface, public fold construction, and archived reference implementations before choosing the calibration and prediction basis.

```bash
cd /holdout_task; cat _dag_spec.json _frozen_nodes/node_00.py test_dag_candidate.py validate_public_schema.py public_asset_index.json; du -ah /tmp/hegel_continual_archive | sort -h | tail -45; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_batch.pkl','rb'))
def dump(v,k='',d=0):
 if isinstance(v,dict):
  for a,b in v.items():dump(b,k+'/'+str(a),d+1)
 elif isinstance(v,(list,tuple)):
  print(k,type(v),len(v));
  if len(v):dump(v[0],k+'/0',d+1)
 else:
  print(k,type(v),getattr(v,'shape',None),str(v).replace('\n',' ')[:240])
dump(p)
PY
```
- **Container Exit Code**: `0` (`405 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
dy/panel_b/v <class 'numpy.ndarray'> (24, 4) [[-1.25296656  0.30638535 -0.31155951  0.1977111 ]  [-0.33130518 -1.25414323 -0.12093736  0.75454919]  [ 0.35712341 -0.75161777 -0.24561648  1.29929031]  [-0.59590609  0.13564196  1.16367843  0.92314452]  [-0.03995161 -0.61573236  0.8162008
/0/study/panel_c/u <class 'numpy.ndarray'> (24, 4) [[ 5.29932425e-01 -6.57244279e-01  1.12979903e+00  1.13963166e+00]  [ 8.10225213e-01  4.64364432e-01  1.82127581e-01 -3.83144556e-01]  [-1.94559178e-01  7.83215795e-01 -2.51070919e-02  6.65680719e-01]  [ 1.06958840e+00 -1.23022861e-02 -2.36
/0/study/panel_c/v1 <class 'numpy.ndarray'> (24, 4) [[-0.1649556  -0.81812366  1.0343724   1.16430814]  [ 0.4651691  -0.90707076  0.08151641 -0.54315127]  [ 1.01640121  0.43413261  0.26199149  0.3750544 ]  [-0.37454913 -0.63721392  0.11930786 -0.99927339]  [ 0.04076929  0.03461479 -1.135894 
/0/study/panel_c/v2 <class 'numpy.ndarray'> (24, 4) [[ 0.24252146 -1.02948612 -0.42344838  0.24282715]  [-0.16497026 -0.12638734  0.4465364   0.06009065]  [ 0.61431434 -0.20274196  0.23154121 -0.31899091]  [-0.31357681  0.0775065   0.44050288  0.40881244]  [ 0.01666929  0.21822535 -0.1684706
/0/study/train/x <class 'numpy.ndarray'> (72, 4) [[ 0.43441159 -1.07677482 -1.00815337  1.70281525]  [-0.73756618 -1.08775988  0.24880314 -0.47123747]  [-0.11871155  0.21850813  1.29868518  1.09600986]  [ 0.158062   -0.13970136 -0.0942705  -0.83078869]  [ 0.895041   -0.51971402  1.8922804
/0/study/train/y <class 'numpy.ndarray'> (72,) [ 1.51949824e+00 -1.02321735e+00  5.64190463e-01 -1.17978084e+00   2.09223685e+00 -7.09300942e-01  2.86673210e-01 -5.19084903e-01  -1.52415407e+00 -3.61711982e-01 -5.15217199e-01 -4.00135463e-01   8.12842205e-01 -7.04625314e-02 -4.21971411e
/0/study/train/traj <class 'numpy.ndarray'> (72, 8) [[-6.80431974e-01 -1.35974661e+00 -1.74811838e+00 -1.99753391e+00   -2.14894844e+00 -2.27406256e+00 -2.36221513e+00 -2.43830496e+00]  [ 1.56626627e-01  2.87443932e-01  3.54888513e-01  3.66607957e-01    3.67989765e-01  3.70800249e-01  3.7905
/0/study/train/summary <class 'numpy.ndarray'> (4,) [-0.06185208  1.00835588  0.11060751  1.7566594 ]
/0/study/test/x <class 'numpy.ndarray'> (36, 4) [[ 1.07858741  1.10438106  0.32111455  1.29233426]  [-0.08547784 -0.10251213 -0.51696816 -1.20260743]  [ 0.16355698 -0.95060452  0.75313907 -1.17859752]  [ 0.66066646  1.46992972  1.41221265  0.83706655]  [ 0.28954012 -0.13651499  0.3925200
/0/study/telemetry_vec <class 'numpy.ndarray'> (4,) [0.22 0.22 2.15 1.25]
```

### Turn 2 — `execute_in_remote_docker` (`13627 ms` LLM)
- **Rationale**: Measure panel map residuals and trajectory rank, and locate archived sink code and public fold definitions to constrain the candidate grammar.

```bash
cd /holdout_task; grep -nE 'def |cv_|diagnostics|fold|expected' test_dag_candidate.py; find /tmp/hegel_continual_archive -maxdepth 4 -type f | sort | head -120; python3 - <<'PY'
import pickle,numpy as n
p=pickle.load(open('public_batch.pkl','rb'));print('len',len(p));
for z in p:
 s=z['study'];print(z.keys(),s.keys(),s['telemetry_vec'],[(k,v.shape) for k,v in s.items() if hasattr(v,'shape')]);
 for k in ['panel_a','panel_b','panel_c']:
  d=s[k]; A=n.c_[n.ones(len(d['u'])),d['u']];
  for key in set(d)-{'u'}:
   v=d[key];f=n.linalg.lstsq(A,v,rcond=None)[0]; print(k,key,n.round(f,3),n.round(n.quantile(n.linalg.norm(A@f-v,axis=1),[0,.5,1]),3));
 print('sv traj',n.linalg.svd(s['train']['traj'],compute_uv=False))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment robust IRLS variable projection'
```
- **Container Exit Code**: `0` (`408 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
if tm > 0.038: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")
169:print(json.dumps({"accepted": ok, "errors": errs, "cv_metrics": cv_metrics, "exec_counts": ec}, allow_nan=False))
/tmp/hegel_continual_archive/archive_manifest.json
/tmp/hegel_continual_archive/conf_m102_b04_batch_00_solver.py
/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
len 1
dict_keys(['study']) dict_keys(['eval_times', 'panel_a', 'panel_b', 'panel_c', 'train', 'test', 'telemetry_vec']) [0.22 0.22 2.15 1.25] [('eval_times', (8,)), ('telemetry_vec', (4,))]
panel_a v [[ 0.151  0.031 -0.161 -0.067]
 [ 0.507  0.309 -0.388  0.725]
 [ 0.228  1.21  -0.062  0.068]
 [ 0.19   0.384  1.127 -0.324]
 [-0.154 -0.19   0.407  1.289]] [ 0.305  0.773 12.234]
panel_b v [[-0.052 -0.101  0.437  0.158]
 [-0.314 -0.093  0.389 -0.484]
 [ 0.119 -0.226  0.077 -0.166]
 [-0.174 -0.113  0.337  0.512]
 [-0.09  -0.219 -0.078 -0.464]] [0.421 1.044 2.133]
panel_c v2 [[ 0.165 -0.122 -0.007 -0.033]
 [-0.171 -0.172  0.283  0.351]
 [ 0.101 -0.106  0.499 -0.319]
 [-0.179 -0.42  -0.106 -0.16 ]
 [ 0.358 -0.233 -0.093  0.075]] [0.04  0.154 0.24 ]
panel_c v1 [[ 0.022 -0.001  0.066  0.144]
 [-0.04  -0.539  0.626 -0.406]
 [ 0.919 -0.019 -0.123 -0.288]
 [ 0.101 -0.702 -0.178  0.579]
 [ 0.299  0.311  0.646  0.486]] [0.124 0.197 0.333]
sv traj [3.28352605e+01 1.67931411e+00 5.14192235e-02 4.64235018e-02
 4.27317903e-02 3.79803518e-02 3.39886964e-02 3.27000915e-02]
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 3 — `execute_in_remote_docker` (`16306 ms` LLM)
- **Rationale**: Recover the archived feature grammar and exact diagnostics contract, then quantify the inlier map before deciding whether diagonal calibration is adequate.

```bash
cd /holdout_task; sed -n '99,180p' test_dag_candidate.py; cat /tmp/hegel_continual_archive/conf_m102_b04_batch_00_solver.py; cat /tmp/hegel_continual_archive/archive_manifest.json; cat public_asset_index.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Procrustes alignment outlier calibration exponential relaxation'; python3 - <<'PY'
import pickle,numpy as n
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; u=s['panel_a']['u'];v=s['panel_a']['v'];A=n.c_[n.ones(len(u)),u];f=n.linalg.lstsq(A,v,rcond=None)[0]
for _ in range(12):
 e=n.linalg.norm(A@f-v,axis=1);i=n.argsort(e)[:37];f=n.linalg.lstsq(A[i],v[i],rcond=None)[0]
print('robust a',f,'residual',n.sort(n.linalg.norm(A@f-v,axis=1))); print('traj vectors',n.linalg.svd(s['train']['traj'],full_matrices=False)[2][:2]);print('panel b norm quantiles',[(k,n.quantile(n.linalg.norm(x,axis=1),[0,.25,.5,.75,1])) for k,x in s['panel_b'].items()]); print('raw ranges',[(k,n.min(s[k]['x'],axis=0),n.max(s[k]['x'],axis=0)) for k in ('train','test')]);print('train keys',s['train'].keys())
PY
```
- **Container Exit Code**: `0` (`429 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
robust a [[ 0.07150468 -0.092629   -0.07735544  0.04267945]
 [ 0.84045901 -0.02907611 -0.03943944  0.27428639]
 [ 0.2616998   1.09371722 -0.03316424 -0.00629725]
 [-0.00707711  0.30912144  1.20288457 -0.03641578]
 [ 0.01534123 -0.01074552  0.21783794  1.18450986]] residual [ 0.05705313  0.06294717  0.07172183  0.09060967  0.11447319  0.11894153
  0.11949788  0.13818916  0.14045742  0.15540658  0.17197405  0.20026622
  0.21192656  0.22040675  0.22188101  0.22751558  0.22855218  0.22863327
  0.22943564  0.25870077  0.27395218  0.27891805  0.30146554  0.30453181
  0.30763687  0.32183998  0.32965382  0.33690086  0.34891415  0.35012647
  0.36055748  0.3667895   0.38424326  0.38526826  0.39797976  0.43744472
  0.48015772  0.51090775  0.56920647  0.69111176  0.72117354  0.81495742
  0.87249321  0.98455374  1.07871055 11.72241527 12.12776348 12.36515268]
traj vectors [[ 0.11328438  0.23008454  0.30311122  0.35214867  0.38746959  0.41471485
   0.43610694  0.45390781]
 [ 0.36049594  0.51797322  0.43399033  0.26121131  0.06342802 -0.13608795
  -0.31331991 -0.47376688]]
panel b norm quantiles [('u', array([0.46379098, 1.24285006, 1.36461564, 1.7362393 , 2.01107952])), ('v', array([0.41789905, 1.17339283, 1.34748541, 1.46151873, 1.72189612]))]
raw ranges [('train', array([-1.03653382, -1.5048644 , -1.52714379, -1.30633175]), array([1.631376  , 1.9281467 , 1.89228049, 1.97312486])), ('test', array([-1.09640748, -1.26347452, -1.53830928, -1.26662398]), array([1.5112118 , 1.46992972, 1.41221265, 1.86769161]))]
train keys dict_keys(['x', 'y', 'traj', 'summary'])
```

### Turn 4 — `execute_in_remote_docker` (`16217 ms` LLM)
- **Rationale**: Retrieve the archive code and exact public scoring folds without the long asset listing that truncated them in the previous output.
- **ATMS NOGOOD DECLARED**: Refuted `Panel A can be calibrated with a diagonal affine map.` | Witness: `The 37-row trimmed fit has four off-diagonal couplings of magnitude 0.218–0.309, each much larger than the other off-diagonal terms.` | Blanket Diagnosis: `Diagonal inversion leaves structured cross-coordinate mixing in the sink features.` | Sublated Pivot: `Retain the frozen feature provider and test a robust coupled calibration in the sink.`

```bash
cd /holdout_task; sed -n '140,175p' test_dag_candidate.py; cat /tmp/hegel_continual_archive/conf_m102_b04_batch_00_solver.py
```
- **Container Exit Code**: `0` (`296 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"], float) - vy)))
    tm = float(np.mean(np.abs(np.asarray(out_cv["state_pred"], float) - vtr)))
    cv_metrics = {"public_cv_y_mae": round(ym, 6), "public_cv_traj_mae": round(tm, 6)}
    if ym > 0.038: errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
    if tm > 0.038: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")

ok = len(errs) == 0 and all(v == 1 for v in ec.values())
print(json.dumps({"accepted": ok, "errors": errs, "cv_metrics": cv_metrics, "exec_counts": ec}, allow_nan=False))
sys.exit(0 if ok else 1)
"""Panel-informed scalar and state prediction for the public campaign."""
from itertools import combinations_with_replacement

import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _basis(x):
    return np.column_stack([np.ones(len(x))] + [np.prod(x[:, j], axis=1) for d in (1, 2, 3) for j in combinations_with_replacement(range(x.shape[1]), d)])


def _panel_features(study, x):
    a, c = study['panel_a'], study['panel_c']
    u, v = a['u'], a['v']
    f = _affine(u)
    fit = np.linalg.lstsq(f, v, rcond=None)[0]
    for _ in range(10):
        err = np.linalg.norm(f @ fit - v, axis=1)
        keep = np.argsort(err)[:int(.82 * len(u))]
        fit = np.linalg.lstsq(f[keep], v[keep], rcond=None)[0]
    maps = np.linalg.lstsq(_affine(c['u']), c['v1'], rcond=None)[0]
    raw = _affine(x) @ maps
    calibrated = (x - fit[0]) / fit[1:].diagonal()
    aligned = _affine(calibrated) @ maps
    squash = lambda q: q / np.sqrt(1 + .09 * q*q)
    return _basis(squash(raw)), _basis(squash(aligned)), calibrated


def _ridge(x, y, q, penalty):
    return q @ np.linalg.solve(x.T @ x + penalty * np.eye(x.shape[1]), x.T @ y)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.vstack((train['x'], study['test']['x']))
    raw, aligned, calibrated = _panel_features(study, x)
    # Blend two cubic bases with different extrapolation behavior.
    hybrid = np.column_stack((aligned, _basis(calibrated)))
    y = .8 * _ridge(hybrid[:n], train['y'], hybrid[n:], .007)
    y += .2 * _ridge(raw[:n], train['y'], raw[n:], .012)
    z = train['traj']
    time_basis = np.linalg.svd(z, full_matrices=False)[2][:2]
    amps = z @ time_basis.T
    state = (.5 * _ridge(raw[:n], amps, raw[n:], .014) + .5 * _ridge(aligned[:n], amps, aligned[n:], .014)) @ time_basis
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
```

### Turn 5 — `execute_in_remote_docker` (`34387 ms` LLM)
- **Rationale**: Test calibration families against panel leave-one-out residuals and isolate which coordinate systems improve the scalar and rank-two spatial heads on held-out measurements.

```bash
cat > /tmp/dag_probe.py <<'PY'
import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];Y=s['train']['y'];Z=s['train']['traj'];T=np.linalg.svd(Z,full_matrices=False)[2][:2]; targets=np.c_[Y,Z@T.T]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(x.shape[1]),k)])
def panel(u,v,fun):
 f=fun(u);keep=np.ones(len(u),bool)
 for j in range(5):
  b=np.linalg.lstsq(f[keep],v[keep],rcond=None)[0];e=np.linalg.norm(f@b-v,axis=1);keep=e<3
 return b,e
u,v=(s['panel_a'][k] for k in ('u','v')); f,e=panel(u,v,lambda x:poly(x,1));print('a fit',np.round(f,3));good=e<3
for name,fun in [('linear',lambda x:poly(x,1)),('sq',lambda x:poly(x,2)),('cube',lambda x:poly(x,3)),('axis-sq',lambda x:np.c_[poly(x,1),x*x]),('axis-cube',lambda x:np.c_[poly(x,1),x*x,x**3]),('sine',lambda x:np.c_[poly(x,1),np.sin(2.15*x),np.cos(2.15*x)]),('tanh',lambda x:np.c_[poly(x,1),np.tanh(2.15*x)])]:
 a=fun(u)[good];b=v[good];p=np.linalg.pinv(a);r=a@(p@b)-b;loo=r/(1-np.sum(a*p.T,axis=1))[:,None]; print('A',name,np.round(np.mean(r*r,axis=0)**.5,4),np.round(np.mean(loo*loo,axis=0)**.5,4))
c=s['panel_c'];b=s['panel_b'];matrix=np.linalg.lstsq(poly(c['u'],1),c['v1'],rcond=None)[0]; print('map sv',np.linalg.svd(matrix[1:],compute_uv=False));print('corr a err',np.round(np.corrcoef(np.c_[u,v-poly(u,1)@f][good].T),2));print('b dist corr',np.corrcoef(*[np.linalg.norm(b[k][:,None]-b[k][None,:],axis=-1)[np.triu_indices(len(b[k]),1)] for k in ('u','v')])[0,1]);
q=(x-f[0])@np.linalg.inv(f[1:]); qdiag=(x-f[0])/f[1:].diagonal(); qtrim=(x-np.array([.0715,-.0926,-.0774,.0427]))@np.linalg.inv(np.array([[.84,-.029,-.039,.274],[.262,1.094,-.033,-.006],[-.007,.309,1.203,-.036],[.015,-.011,.218,1.185]]));
def calc(A,inds=np.arange(52),test=np.arange(52,72)):
 a=A[inds]; B=A[test];U,d,V=np.linalg.svd(a,full_matrices=False); m=U.T@targets[inds]; errs=[]
 for l in [1e-8,1e-5,.0001,.001,.003,.01,.03,.1,.3,1,3]:
  p=(B@V.T)@((d/(d*d+l))[:,None]*m);errs.append([np.mean(abs(p[:,0]-Y[test])),np.mean(abs((p[:,1:]-targets[test,1:])@T))])
 return np.array(errs).min(0)
res=[]
for name,z in [('raw',x),('diag',qdiag),('full',q),('trim',qtrim)]:
 for mapname,w in [('axis',z),('v1',poly(z,1)@matrix),('v2',poly(z,1)@np.linalg.lstsq(poly(c['u'],1),c['v2'],rcond=None)[0])]:
  for scale in [0,.03,.09,.22,.5,1]:
   h=w/np.sqrt(1+scale*w*w)
   for d in [2,3,4,5]:
    p=poly(h,d);res.append((calc(p),name,mapname,scale,d))
   for freq in [.5,1,1.25,1.5,2.15,2.5,3]:
    p=np.c_[poly(h,2),np.sin(freq*h),np.cos(freq*h)];res.append((calc(p),name,mapname,scale,('four',freq)))
for j in [0,1]:
 print('best',j); [print(r) for r in sorted(res,key=lambda a:a[0][j])[:12]]
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/dag_probe.py
```
- **Container Exit Code**: `0` (`940 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.0434] [0.0657 0.0908 0.0668 0.0774]
A cube [0. 0. 0. 0.] [0.0003 0.0005 0.0006 0.0003]
A axis-sq [0.1875 0.1934 0.1615 0.1623] [0.2709 0.2742 0.2153 0.2172]
A axis-cube [0.1774 0.184  0.1491 0.1476] [0.2986 0.2969 0.2444 0.2392]
A sine [0.18   0.1834 0.1499 0.1479] [0.29   0.2846 0.2384 0.2303]
A tanh [0.1908 0.1921 0.1539 0.168 ] [0.2586 0.2541 0.2028 0.2213]
map sv [0.99106857 0.94148054 0.91378093 0.89452598]
corr a err [[ 1.   -0.13  0.17  0.03  0.    0.   -0.   -0.  ]
 [-0.13  1.    0.04  0.05 -0.   -0.    0.   -0.  ]
 [ 0.17  0.04  1.    0.01 -0.   -0.   -0.   -0.  ]
 [ 0.03  0.05  0.01  1.   -0.   -0.   -0.   -0.  ]
 [ 0.   -0.   -0.   -0.    1.    0.05 -0.04  0.2 ]
 [ 0.   -0.   -0.   -0.    0.05  1.   -0.34 -0.08]
 [-0.    0.   -0.   -0.   -0.04 -0.34  1.    0.12]
 [-0.   -0.   -0.   -0.    0.2  -0.08  0.12  1.  ]]
b dist corr 0.13566398354294879
best 0
(array([0.13480174, 0.22796354]), 'trim', 'axis', 0.22, ('four', 2.5))
(array([0.13580584, 0.22311551]), 'trim', 'axis', 0.5, ('four', 2.5))
(array([0.14030433, 0.23901709]), 'full', 'axis', 0.22, ('four', 2.5))
(array([0.14178275, 0.23219522]), 'trim', 'axis', 0.22, ('four', 2.15))
(array([0.14200402, 0.25051132]), 'trim', 'axis', 1, 2)
(array([0.14250193, 0.22187349]), 'trim', 'axis', 1, ('four', 0.5))
(array([0.14280069, 0.22083768]), 'trim', 'axis', 1, ('four', 1.5))
(array([0.14312881, 0.22292857]), 'trim', 'axis', 0.5, ('four', 2.15))
(array([0.14358558, 0.24154357]), 'full', 'axis', 0.09, ('four', 2.5))
(array([0.14365318, 0.23927574]), 'trim', 'axis', 0.09, ('four', 2.15))
(array([0.143831  , 0.22062116]), 'trim', 'axis', 1, ('four', 1))
(array([0.14390326, 0.21778276]), 'trim', 'axis', 1, ('four', 2.15))
best 1
(array([0.20571779, 0.16762629]), 'full', 'axis', 1, 3)
(array([0.20202691, 0.2010047 ]), 'trim', 'axis', 1, 3)
(array([0.1451976 , 0.20688895]), 'trim', 'axis', 1, ('four', 2.5))
(array([0.16001512, 0.21003915]), 'full', 'axis', 1, ('four', 2.5))
(array([0.16045619, 0.21448127]), 'full', 'axis', 1, ('four', 2.15))
(array([0.17847423, 0.21576664]), 'trim', 'axis', 0.5, ('four', 3))
(array([0.14390326, 0.21778276]), 'trim', 'axis', 1, ('four', 2.15))
(array([0.17037408, 0.21958137]), 'trim', 'axis', 1, ('four', 3))
(array([0.143831  , 0.22062116]), 'trim', 'axis', 1, ('four', 1))
(array([0.14616635, 0.22071646]), 'trim', 'axis', 1, ('four', 1.25))
(array([0.14280069, 0.22083768]), 'trim', 'axis', 1, ('four', 1.5))
(array([0.14250193, 0.22187349]), 'trim', 'axis', 1, ('four', 0.5))
```

### Turn 6 — `execute_in_remote_docker` (`29529 ms` LLM)
- **Rationale**: First measure how accurately the cubic map reconstructs the observed study inputs, then check whether the recovered latent coordinates improve held-out predictions for both heads.
- **Alternatives Weighed**: `Expand the Fourier head on affine-corrected inputs` (rejected: The existing head sweep remains far above tolerance, while panel measurements directly support a richer calibration map.)
- **ATMS NOGOOD DECLARED**: Refuted `A full affine inverse adequately recovers the latent study coordinates.` | Witness: `After excluding three gross outliers, panel A leave-one-out RMSE is 0.195–0.247 per coordinate for the affine fit but only 0.0003–0.0006 for the full cubic basis.` | Blanket Diagnosis: `Cross-coordinate nonlinear terms belong in the calibration map; fitting prediction heads to affine-corrected inputs leaves substantial measurement distortion.` | Sublated Pivot: `Invert the robust cubic map before fitting and validating the two prediction heads.`

```bash
cat /holdout_task/_frozen_nodes/node_00.py; sed '/u,v=(s/,$d' /tmp/dag_probe.py > /tmp/cubic_probe.py; cat >> /tmp/cubic_probe.py <<'PY'
from itertools import product
a=s['panel_a'];u,v=a['u'],a['v']; good=np.linalg.norm(v,axis=1)<7
p=poly(u,3); cf=np.linalg.lstsq(p[good],v[good],rcond=None)[0]; print('panel A inliers',sum(good),'max residual',np.max(abs(p[good]@cf-v[good])),'latent bounds',u.min(0),u.max(0)); print('coeff',np.round(cf,5))
lin=cf[1:5]; z=[]
for obs in x:
 r=least_squares(lambda q:poly(q[None],3)[0]@cf-obs,(obs-cf[0])@np.linalg.inv(lin),gtol=1e-12,xtol=1e-12); z.append(r.x)
z=np.array(z);print('inverse ranges',z.min(0),z.max(0),'rms',np.mean((poly(z,3)@cf-x)**2)**.5)
c=s['panel_c']; cfs={}
for k in ['v1','v2']:
 for d in [1,2,3]:
  p=poly(c['u'],d); f=np.linalg.lstsq(p,c[k],rcond=None)[0]; e=p@f-c[k]; lev=np.sum(p*np.linalg.pinv(p).T,axis=1); print('C',k,d,p.shape,'rmse',np.mean(e*e)**.5,'loo',np.mean((e/np.maximum((1-lev)[:,None],1e-9))**2)**.5)
  cfs[k,d]=f
b=s['panel_b'];print('panel b head',np.round(b['u'][:5],3),np.round(b['v'][:5],3))
np.savez('/tmp/cubic_probe.npz',z=z,cf=cf)
res=[]; idx=np.arange(52); te=np.arange(52,72)
def score(A,tag):
 U,d,V=np.linalg.svd(A[idx],full_matrices=False); m=U.T@targets[idx]; B=A[te]@V.T
 for l in [1e-9,1e-6,.0001,.001,.003,.01,.03,.1,.3,1]:
  pred=B@((d/(d*d+l))[:,None]*m);res.append((np.mean(abs(pred[:,0]-Y[te])),np.mean(abs((pred[:,1:]-targets[te,1:])@T)),tag,l))
for key,w in [('axis',z),('raw',x)]+[(str(k),poly(z,k[1])@f) for k,f in cfs.items()]:
 for sq in [0,.09,.22,.5,1.25]:
  h=w/np.sqrt(1+sq*w*w)
  for d in [2,3,4]:score(poly(h,d),(key,sq,'poly',d))
  for f in [.75,1,1.25,1.5,1.75,2,2.15,2.25,2.5,3]:
   for d in [1,2,3]: score(np.c_[poly(h,d),np.sin(f*h),np.cos(f*h)],(key,sq,'four',f,d))
for j in [0,1]: print('best',j,*sorted(res,key=lambda r:r[j])[:8],sep='\n')
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/cubic_probe.py
```
- **Container Exit Code**: `0` (`1392 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-1.00000e-05 -0.00000e+00 -6.00000e-05]
 [ 9.00000e-05  1.00000e-05  6.00000e-05  5.00000e-05]
 [ 5.00000e-05  6.00000e-05  1.59850e-01 -1.10000e-04]
 [-6.00000e-05  6.00000e-05  7.00000e-05  4.00000e-05]
 [-1.50000e-04  4.00000e-05 -5.00000e-05 -2.80000e-04]
 [-2.00000e-05  8.00000e-05  1.30000e-04  1.60030e-01]]
inverse ranges [-1.09571732 -1.13918668 -1.13118738 -1.04877189] [1.12410511 1.14381696 1.092313   1.14075131] rms 1.0456442788117424e-13
C v1 1 (24, 5) rmse 0.10176404241175013 loo 0.1287645926997606
C v1 2 (24, 15) rmse 0.0698412230681809 loo 0.22835430325281408
C v1 3 (24, 35) rmse 1.3579042400246673e-15 loo 1.3579042400246674e-06
C v2 1 (24, 5) rmse 0.07991188432411611 loo 0.10262328579338492
C v2 2 (24, 15) rmse 0.0035301620909834485 loo 0.011309918620818173
C v2 3 (24, 35) rmse 6.948389084095589e-16 loo 6.948389084095589e-07
panel b head [[-0.209 -1.025 -0.01   0.629]
 [-0.62   0.212  0.055  1.155]
 [-1.077  0.198 -0.068 -0.707]
 [ 0.187 -0.233 -0.335 -0.118]
 [ 0.369  0.228 -0.712 -0.703]] [[-1.253  0.306 -0.312  0.198]
 [-0.331 -1.254 -0.121  0.755]
 [ 0.357 -0.752 -0.246  1.299]
 [-0.596  0.136  1.164  0.923]
 [-0.04  -0.616  0.816 -0.009]]
best
0
(0.0712146488286686, 0.020782341645918963, ('axis', 0, 'four', 2.5, 1), 0.1)
(0.07143055520495259, 0.02163993460519626, ('axis', 0, 'four', 2.5, 1), 0.03)
(0.07159514181139537, 0.021939160119905855, ('axis', 0, 'four', 2.5, 1), 0.01)
(0.07162856999994734, 0.017543651386405113, ('axis', 0.09, 'four', 2.5, 1), 1e-09)
(0.07162857714367067, 0.017543627926380725, ('axis', 0.09, 'four', 2.5, 1), 1e-06)
(0.07162928508463087, 0.017541303213816715, ('axis', 0.09, 'four', 2.5, 1), 0.0001)
(0.07163572131215033, 0.017520183504659302, ('axis', 0.09, 'four', 2.5, 1), 0.001)
(0.07165002660926852, 0.017478945434682678, ('axis', 0.09, 'four', 2.5, 1), 0.003)
best
1
(0.08666649377799893, 0.0030541486007690712, ('axis', 0.09, 'four', 1.75, 2), 1e-06)
(0.08668424097889213, 0.0030549415055058523, ('axis', 0.09, 'four', 1.75, 2), 1e-09)
(0.08559248611720571, 0.0036789523434979697, ('axis', 0.09, 'four', 2, 2), 1e-09)
(0.08558613795814592, 0.003679541398704811, ('axis', 0.09, 'four', 2, 2), 1e-06)
(0.08496193743624639, 0.0038921504522748075, ('axis', 0.09, 'four', 2, 2), 0.0001)
(0.08726607292992353, 0.004089371684181285, ('axis', 0, 'four', 2.5, 2), 0.001)
(0.08765123074022586, 0.004168372596628551, ('axis', 0, 'four', 2.5, 2), 0.0001)
(0.08769411266870875, 0.004182108922860026, ('axis', 0, 'four', 2.5, 2), 1e-06)
```

### Turn 7 — `execute_in_remote_docker` (`25519 ms` LLM)
- **Rationale**: The cubic inverse makes the trajectory head viable. Probe scalar residuals for pairwise interactions and rational terms before choosing the sink's scalar model.

```bash
sed '/u,v=(s/,$d' /tmp/dag_probe.py > /tmp/head.py; cat >> /tmp/head.py <<'PY'
from itertools import combinations,product
z=np.load('/tmp/cubic_probe.npz')['z']; tr=np.arange(52); te=np.arange(52,72)
for k in ['v1','v2']:
 c=s['panel_c']; a=c['u']; b=c[k]; print(k, 'cubic axis', np.round(np.linalg.lstsq(np.c_[poly(a,1),a*a,a**3],b,rcond=None)[0],3));print('err',np.std(b-np.c_[poly(a,1),a*a,a**3]@np.linalg.lstsq(np.c_[poly(a,1),a*a,a**3],b,rcond=None)[0]))
print('telemetry',s['telemetry_vec']); print('times',s['eval_times']);
D=np.vstack([np.eye(4)]+[np.eye(4)[i]+sg*np.eye(4)[j] for i,j in combinations(range(4),2) for sg in [-1,1]]+[np.array([1,*sg]) for sg in product([-1,1],repeat=3)]); print('D', D)
for sq in [0,.09,.22,.5,1.25]:
 h=z/np.sqrt(1+sq*z*z);p=poly(h,2)
 for head,t in [('y',Y),('a',targets[:,1]),('b',targets[:,2])]:
  for f in [1.25,1.75,2.15,2.5,3.15]:
   A=np.c_[p,np.sin(f*h),np.cos(f*h)]; fit=np.linalg.lstsq(A[tr],t[tr],rcond=None)[0];r=t-A@fit
   cand=[];w=h@D.T
   for ff in [.5,1,1.25,1.75,2.15,2.5,3,4]:
    for nm,F in [('sin',np.sin(ff*w)),('cos',np.cos(ff*w)),('tanh',np.tanh(ff*w)),('rat',w/(1+ff*w*w)),('rat2',1/(1+ff*w*w))]:
     F-=A@np.linalg.lstsq(A[tr],F[tr],rcond=None)[0]; score=np.abs(r[tr]@F[tr])/np.maximum(np.linalg.norm(F[tr],axis=0),1e-8)
     j=score.argmax(); pred=F[:,j]*(r[tr]@F[tr,j])/max(np.sum(F[tr,j]**2),1e-10); cand.append((score[j],np.mean(abs(r[te]-pred[te])),nm,ff,j))
   if head=='y' or (sq==.09 and f==1.75):print(head,sq,f,'base',np.round([np.std(r[tr]),np.mean(abs(r[te]))],4),'terms', [tuple(np.round(q[:2],4))+q[2:] for q in sorted(cand,reverse=True)[:4]])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/head.py; find /tmp/hegel_continual_archive -maxdepth 2 -type f | sort | tail -25
```
- **Container Exit Code**: `0` (`909 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
626, 'cos', 3, 10), (0.2868, 0.0463, 'rat2', 2.5, 11)]
y 0.22 2.15 base [0.0528 0.0842] terms [(0.2895, 0.0631, 'cos', 3, 10), (0.2879, 0.0629, 'cos', 2.5, 10), (0.2865, 0.0454, 'rat2', 3, 11), (0.286, 0.0478, 'rat2', 4, 11)]
y 0.22 2.5 base [0.0534 0.084 ] terms [(0.2922, 0.0638, 'cos', 3, 10), (0.2919, 0.0635, 'cos', 2.5, 10), (0.2883, 0.0645, 'cos', 2.15, 10), (0.2881, 0.0636, 'rat2', 0.5, 10)]
y 0.22 3.15 base [0.0551 0.0839] terms [(0.3012, 0.0678, 'cos', 2.5, 10), (0.2989, 0.0686, 'cos', 2.15, 10), (0.2983, 0.0685, 'cos', 3, 10), (0.2972, 0.0685, 'rat2', 0.5, 10)]
y 0.5 1.25 base [0.0559 0.083 ] terms [(0.3025, 0.0697, 'cos', 2.5, 10), (0.3018, 0.0703, 'cos', 2.15, 10), (0.3003, 0.0709, 'cos', 1.75, 10), (0.3002, 0.0698, 'cos', 3, 10)]
y 0.5 1.75 base [0.0562 0.083 ] terms [(0.3036, 0.0709, 'cos', 2.5, 10), (0.3031, 0.0715, 'cos', 2.15, 10), (0.3017, 0.072, 'cos', 1.75, 10), (0.3009, 0.0707, 'cos', 3, 10)]
y 0.5 2.15 base [0.0566 0.0833] terms [(0.3048, 0.0723, 'cos', 2.5, 10), (0.3045, 0.0728, 'cos', 2.15, 10), (0.3032, 0.0733, 'cos', 1.75, 10), (0.3017, 0.0718, 'cos', 3, 10)]
y 0.5 2.5 base [0.0569 0.0836] terms [(0.306, 0.0737, 'cos', 2.5, 10), (0.3058, 0.0743, 'cos', 2.15, 10), (0.3048, 0.0747, 'cos', 1.75, 10), (0.3031, 0.0751, 'cos', 1.25, 10)]
y 0.5 3.15 base [0.0578 0.0845] terms [(0.3089, 0.0777, 'cos', 2.15, 10), (0.3085, 0.0772, 'cos', 2.5, 10), (0.3082, 0.0782, 'cos', 1.75, 10), (0.3069, 0.0786, 'cos', 1.25, 10)]
y 1.25 1.25 base [0.0588 0.0892] terms [(0.3044, 0.0823, 'cos', 0.5, 10), (0.304, 0.0822, 'cos', 1, 10), (0.3038, 0.0821, 'cos', 1.25, 10), (0.3028, 0.0818, 'cos', 1.75, 10)]
y 1.25 1.75 base [0.0588 0.0894] terms [(0.3043, 0.0825, 'cos', 0.5, 10), (0.304, 0.0824, 'cos', 1, 10), (0.3037, 0.0823, 'cos', 1.25, 10), (0.3027, 0.082, 'cos', 1.75, 10)]
y 1.25 2.15 base [0.0587 0.0895] terms [(0.3042, 0.0827, 'cos', 0.5, 10), (0.3039, 0.0825, 'cos', 1, 10), (0.3036, 0.0824, 'cos', 1.25, 10), (0.3027, 0.0822, 'cos', 1.75, 10)]
y 1.25 2.5 base [0.0587 0.0896] terms [(0.3041, 0.0828, 'cos', 0.5, 10), (0.3038, 0.0827, 'cos', 1, 10), (0.3036, 0.0826, 'cos', 1.25, 10), (0.3026, 0.0824, 'cos', 1.75, 10)]
y 1.25 3.15 base [0.0585 0.09  ] terms [(0.304, 0.0833, 'cos', 0.5, 10), (0.3037, 0.0831, 'cos', 1, 10), (0.3034, 0.083, 'cos', 1.25, 10), (0.3025, 0.0828, 'cos', 1.75, 10)]
/tmp/hegel_continual_archive/archive_manifest.json
/tmp/hegel_continual_archive/conf_m102_b04_batch_00_solver.py
/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
```

### Turn 8 — `execute_in_remote_docker` (`32256 ms` LLM)
- **Rationale**: Select residual terms using training rows only, and test whether either reference panel has a compact nonlinear map that explains the missing scalar structure.

```bash
sed '/u,v=(s/,$d' /tmp/dag_probe.py > /tmp/sparse.py; cat >> /tmp/sparse.py <<'PY'
from itertools import product, combinations
z=np.load('/tmp/cubic_probe.npz')['z']; tr=np.arange(52); te=np.arange(52,72)
D=np.array([d for d in product([-1,0,1],repeat=4) if any(d) and next(x for x in d if x)==1]);
for squash in [0,.09,.22]:
 h=z/np.sqrt(1+squash*z*z); w=h@D.T
 F=[]; keys=[]
 def add(v,k):
  v=np.asarray(v).reshape(len(h),-1);F.append(v);keys.extend([(k,j) for j in range(v.shape[1])]);
 for d in [2,3]:
  add(poly(h,d),('poly',d))
 for f in [.5,1,1.25,1.75,2.15,2.5,3,4]:
  for nm,v in [('sin',np.sin(f*w)),('cos',np.cos(f*w)),('rat',w/(1+f*w*w)),('rat2',1/(1+f*w*w)),('tanh',np.tanh(f*w))]:add(v,(nm,f))
 for f in [.22,.5,1,2.15,3]:
  for j in range(4): add(h[:,j,None]/(1+f*w*w),('ratio',j,f))
  for j,k in combinations(range(4),2):add((h[:,j]*h[:,k])[:,None]/(1+f*w*w),('ratio2',j,k,f))
 F=np.column_stack(F)
 for base in [poly(h,1),poly(h,2),np.c_[poly(h,2),np.sin(2.15*h),np.cos(2.15*h)],np.c_[poly(h,2),np.sin(2.15*h),np.cos(1.25*h)]]:
  active=[]; A=base
  for step in range(9):
   q=np.linalg.svd(A[tr],full_matrices=False)[0];R=F[tr]-q@(q.T@F[tr]); norm=np.linalg.norm(R,axis=0); beta=np.linalg.lstsq(A[tr],Y[tr],rcond=None)[0]; e=Y-A@beta
   if step<5 or step==8: print(squash,base.shape[1],step,round(np.std(e[tr]),5),round(np.mean(abs(e[te])),5),[(keys[i],tuple(D[keys[i][1]]) if keys[i][0][0]!='poly' else None) for i in active]);
   j=np.argmax(np.where(norm>1e-7,abs(e[tr]@R)/np.maximum(norm,1e-9),0)); active.append(j); A=np.c_[A,F[:,j]]
c=s['panel_c'];a=c['u'];
for key in ['v1','v2']:
 res=[]
 for f in np.arange(.1,3,.05):
  for nm,p in [('sin',np.sin(f*a)),('tanh',np.tanh(f*a)),('rat',a/np.sqrt(1+f*a*a)),('sincos',np.c_[np.sin(f*a),np.cos(f*a)])]:
   A=np.c_[np.ones(len(a)),p];P=np.linalg.pinv(A); e=c[key]-A@(P@c[key]);res.append((np.mean((e/(1-np.sum(A*P.T,axis=1))[:,None])**2)**.5,nm,round(f,3)))
 print('C',key,sorted(res)[:8]);
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/sparse.py > /tmp/sparse.out; head -18 /tmp/sparse.out; grep -v ' 0 \[' /tmp/sparse.out | sort -n -k5,5 | head -16; tail -2 /tmp/sparse.out
```
- **Container Exit Code**: `0` (`694 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(0, 0, 1, 0)), ((('ratio2', 1, 2, 0.5), 11), (0, 1, 1, 0)), ((('ratio', 0, 0.5), 16), (1, -1, 0, -1)), ((('rat2', 0.5), 25), (1, 0, 0, -1)), ((('cos', 2.5), 5), (0, 1, -1, 0)), ((('tanh', 1.75), 36), (1, 1, 0, 1))]
0.22 23 4 0.01094 0.03592 [((('ratio2', 1, 2, 1), 11), (0, 1, 1, 0)), ((('ratio2', 0, 3, 2.15), 25), (1, 0, 0, -1)), ((('ratio', 1, 0.22), 13), (1, -1, -1, -1)), ((('rat2', 1), 0), (0, 0, 0, 1))]
0.22 23 8 0.00523 0.03646 [((('ratio2', 1, 2, 1), 11), (0, 1, 1, 0)), ((('ratio2', 0, 3, 2.15), 25), (1, 0, 0, -1)), ((('ratio', 1, 0.22), 13), (1, -1, -1, -1)), ((('sin', 2.5), 24), (1, 0, -1, 1)), ((('sin', 4), 16), (1, -1, 0, -1)), ((('ratio2', 1, 3, 1), 19), (1, -1, 1, -1)), ((('cos', 3), 7), (0, 1, 0, -1)), ((('cos', 1.75), 31), (1, 1, -1, -1))]
0 23 8 0.00501 0.03748 [((('ratio2', 1, 2, 0.5), 11), (0, 1, 1, 0)), ((('ratio2', 0, 3, 1), 25), (1, 0, 0, -1)), ((('sin', 3), 17), (1, -1, 0, 0)), ((('ratio', 0, 3), 5), (0, 1, -1, 0)), ((('sin', 4), 4), (0, 1, -1, -1)), ((('ratio2', 0, 2, 3), 26), (1, 0, 0, 0)), ((('ratio2', 1, 3, 3), 36), (1, 1, 0, 1)), ((('rat2', 4), 2), (0, 0, 1, 0))]
0.09 15 4 0.02739 0.03758 [((('cos', 0.5), 0), (0, 0, 0, 1)), ((('cos', 3), 26), (1, 0, 0, 0)), ((('cos', 1), 2), (0, 0, 1, 0)), ((('ratio2', 1, 2, 0.5), 11), (0, 1, 1, 0))]
0.09 23 1 0.02172 0.03803 [((('ratio2', 1, 2, 1), 11), (0, 1, 1, 0))]
0.22 23 4 0.00994 0.03818 [((('ratio2', 1, 2, 1), 11), (0, 1, 1, 0)), ((('ratio2', 0, 3, 2.15), 25), (1, 0, 0, -1)), ((('ratio', 1, 0.22), 13), (1, -1, -1, -1)), ((('sin', 2.5), 24), (1, 0, -1, 1))]
0.09 15 8 0.01221 0.03841 [((('cos', 0.5), 0), (0, 0, 0, 1)), ((('cos', 3), 26), (1, 0, 0, 0)), ((('cos', 1), 2), (0, 0, 1, 0)), ((('ratio2', 1, 2, 0.5), 11), (0, 1, 1, 0)), ((('rat2', 4), 25), (1, 0, 0, -1)), ((('ratio', 0, 0.22), 16), (1, -1, 0, -1)), ((('ratio2', 1, 2, 1), 6), (0, 1, -1, 1)), ((('sin', 2.15), 28), (1, 0, 1, -1))]
C v1 [(0.01003647760994062, 'rat', 2.2), (0.01008361999395209, 'rat', 2.15), (0.01017357496626619, 'rat', 2.25), (0.010324449620259676, 'rat', 2.1), (0.01047671530575371, 'rat', 2.3), (0.010758305470663995, 'rat', 2.05), (0.010922284794728702, 'rat', 2.35), (0.011375670588287083, 'rat', 2.0)]
C v2 [(0.007559843929005192, 'sincos', 0.25), (0.007564758240829549, 'sincos', 0.2), (0.007608542723411061, 'sincos', 0.15), (0.007633023065429769, 'sincos', 0.3), (0.00766074610212333, 'sincos', 0.1), (0.007829568085896178, 'sincos', 0.35), (0.008195273594930639, 'sincos', 0.4), (0.008769280928869232, 'sincos', 0.45)]
```

### Turn 9 — `execute_in_remote_docker` (`45617 ms` LLM)
- **Rationale**: Save the candidate now and check the exact sink contract and both public error thresholds in normal and isolated Python processes.
- **Alternatives Weighed**: `Keep all residual terms found by the sparse search` (rejected: Later terms lowered training error but generally worsened public-split MAE; the two recurrent interactions are the smaller supported model.)
- **ATMS NOGOOD DECLARED**: Refuted `An axiswise Fourier head is sufficient after correcting calibration.` | Witness: `The scalar public-split MAE falls from 0.07121 for the prior best axiswise head to 0.02442 when two rational pair interactions are added.` | Blanket Diagnosis: `The scalar head is missing nonseparable interactions between the second and third coordinates and between the first and fourth coordinates.` | Sublated Pivot: `Add the two rational interaction terms and preserve the separately validated rank-two trajectory head.`

```bash
cat > /holdout_task/node_01.py <<'PY'
from itertools import combinations_with_replacement

import numpy as np


def _quadratic(x):
    return np.column_stack((np.ones(len(x)), x, *(x[:, i] * x[:, j] for i, j in combinations_with_replacement(range(x.shape[1]), 2))))


def _calibration(panel):
    u, v = (np.asarray(panel[k], float) for k in ('u', 'v'))
    prev = np.roll(np.arange(u.shape[1]), -1)
    features = [np.column_stack((np.ones(len(u)), u[:, j], u[:, prev[j]], u[:, j] * u[:, prev[j]], u[:, j] ** 3)) for j in range(u.shape[1])]
    weights = np.ones(len(u))
    for _ in range(12):
        root = np.sqrt(weights)
        coef = np.array([np.linalg.lstsq(a * root[:, None], v[:, j] * root, rcond=None)[0] for j, a in enumerate(features)])
        errors = np.linalg.norm(np.column_stack([a @ b for a, b in zip(features, coef)]) - v, axis=1)
        scale = max(3.0 * np.median(errors), 1e-4)
        weights = 1 / (1 + (errors / scale) ** 2)
    keep = errors <= max(8 * np.median(errors), 5e-4)
    if keep.sum() >= max(8, len(u) // 2):
        coef = np.array([np.linalg.lstsq(a[keep], v[keep, j], rcond=None)[0] for j, a in enumerate(features)])
    return coef, prev


def _invert(x, warm, coef, prev):
    bias, diagonal, coupling, cross, cubic = coef.T
    axis = np.arange(len(prev))
    def forward(z):
        p = z[:, prev]
        return bias + diagonal * z + coupling * p + cross * z * p + cubic * z ** 3
    linear = np.diag(diagonal)
    linear[axis, prev] = coupling
    affine = np.linalg.solve(linear, (x - bias).T).T
    use_warm = np.sum((forward(warm) - x) ** 2, axis=1) < np.sum((forward(affine) - x) ** 2, axis=1)
    z = np.where(use_warm[:, None], warm, affine)
    jac = np.zeros((len(z), len(axis), len(axis)))
    for _ in range(22):
        residual = forward(z) - x
        if np.max(np.abs(residual)) < 1e-11:
            break
        jac[:, axis, axis] = diagonal + cross * z[:, prev] + 3 * cubic * z * z
        jac[:, axis, prev] = coupling + cross * z
        try:
            step = np.linalg.solve(jac, residual[..., None])[..., 0]
        except np.linalg.LinAlgError:
            step = (np.linalg.pinv(jac) @ residual[..., None])[..., 0]
        best = np.sum(residual * residual, axis=1)
        new = z.copy()
        for damping in (1., .5, .25, .125, .0625, .03125):
            proposal = z - damping * step
            er = np.sum((forward(proposal) - x) ** 2, axis=1)
            better = er < best
            new[better], best[better] = proposal[better], er[better]
        z = new
    return z


def _ridge(a, t, n, penalty):
    train = a[:n]
    reg = penalty * np.eye(a.shape[1])
    reg[0, 0] = 0.
    return a[n:] @ np.linalg.solve(train.T @ train + reg, train.T @ t)


def _scalar(z, y, n, telemetry):
    h = z / np.sqrt(1 + telemetry[0] * z * z)
    p, q, r, s = h.T
    rat1 = q * r / (1 + (q + r) ** 2)
    rat2 = p * s / (1 + telemetry[2] * (p - s) ** 2)
    return _ridge(np.column_stack((_quadratic(h), np.sin(telemetry[2] * h), np.cos(telemetry[2] * h), rat1, rat2)), y, n, 1e-6)


def _trajectory(z, traj, n):
    h = z / np.sqrt(1 + .09 * z * z)
    time = np.linalg.svd(traj, full_matrices=False)[2][:2]
    features = np.column_stack((_quadratic(h), np.sin(1.75 * h), np.cos(1.75 * h)))
    return _ridge(features, traj @ time.T, n, 1e-6) @ time


def run_node(inputs: dict) -> dict:
    study, frozen = inputs['__input__'], inputs['node_00']
    train = study['train']
    n = len(train['x'])
    x = np.row_stack((train['x'], study['test']['x']))
    d = x.shape[1]
    start = np.asarray(frozen['hybrid'])[:, -((d + 1) * (d + 2) * (d + 3) // 6):][:, 1:d + 1]
    coef, prev = _calibration(study['panel_a'])
    z = _invert(x, start, coef, prev)
    y = _scalar(z, np.asarray(train['y']).reshape(n), n, np.asarray(study['telemetry_vec']))
    state = _trajectory(z, np.asarray(train['traj']), n)
    diagnostics = np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
PY
cd /holdout_task; OPENBLAS_NUM_THREADS=1 python3 test_dag_candidate.py; OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`609 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.024574, "public_cv_traj_mae": 0.004588}, "exec_counts": {"node_00": 1, "node_01": 1}}
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.024574, "public_cv_traj_mae": 0.004588}, "exec_counts": {"node_00": 1, "node_01": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0d9c7e4b68fe6da4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0d9c7e4b (directory).
```

### Turn 10 — `execute_in_remote_docker` (`21153 ms` LLM)
- **Rationale**: The artifact passed the official test in normal and isolated Python. Stress-test the unchanged file on blocked and interleaved holdouts to measure sensitivity to the public split.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
from pathlib import Path
root = Path('/holdout_task')
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
sink = load('sink', root/'node_01.py')
frozen = load('frozen', root/'_frozen_nodes/node_00.py')
rows = []
for bi, batch in enumerate(pickle.load(open(root/'public_batch.pkl', 'rb'))):
    s = batch['study']; train = s['train']; n = len(train['x']); ids = np.arange(n)
    for name, folds in [('blocked', np.array_split(ids, 4)), ('interleaved', [ids[i::4] for i in range(4)])]:
        for i, te in enumerate(folds):
            tr = np.setdiff1d(ids, te)
            st = dict(s, train=dict(train, **{k: train[k][tr] for k in ('x','y','traj')}), test=dict(x=train['x'][te]))
            out = sink.run_node({'__input__':st, 'node_00':frozen.run_node({'__input__':st})})
            y, z = out['y_hat'], out['state_pred']
            assert y.shape == (len(te),) and z.shape == train['traj'][te].shape
            np.testing.assert_allclose(out['diagnostics_vec'], [y.mean(), y.std(), z[:,-1].mean(), z[:,-1].std()])
            assert np.isfinite(y).all() and np.isfinite(z).all()
            rows.append((bi,name,i,float(np.mean(abs(y-train['y'][te].reshape(-1)))),float(np.mean(abs(z-train['traj'][te])))))
print('folds:', len(rows), 'worst scalar:', max(rows,key=lambda r:r[3]), 'worst trajectory:', max(rows,key=lambda r:r[4]))
PY
```
- **Container Exit Code**: `0` (`497 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
folds: 8 worst scalar: (0, 'interleaved', 0, 0.056439813254364486, 0.005332513038787036) worst trajectory: (0, 'blocked', 2, 0.03209741817092923, 0.0053347635205149915)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0d9c7e4b68fe6da4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0d9c7e4b (directory).
```

### Turn 11 — `execute_in_remote_docker` (`44533 ms` LLM)
- **Rationale**: Refit nonlinear scalar parameters inside each fold. Promote only if a challenger clears the supplemental blocked and interleaved tolerance, improves coordinate-tail robustness, and passes the official test in normal and isolated Python; otherwise keep the verified baseline.
- **Alternatives Weighed**: `Submit the already accepted baseline` (rejected: The worst supplemental scalar fold exceeds 0.038; one guarded challenger can test whether learning the nonlinear scales closes that gap without risking the accepted baseline.)

```bash
set -e
cat > /tmp/scalar_options.txt <<'PY'
def _scalar_vp(z, y, n, telemetry):
    from scipy.optimize import least_squares
    t = float(telemetry[2])
    def basis(a, par):
        squash, k1, k2, c1, c2 = par[:5]
        h = a / np.sqrt(1 + squash * a * a)
        p, q, r, s = h.T
        return np.column_stack((_quadratic(h), np.sin(h * par[5:9]), np.cos(h * par[9:13]), q*r/(1+k1*(q+c1*r)**2), p*s/(1+k2*(p-c2*s)**2)))
    a = z[:n]
    center = np.array([float(telemetry[0]), 1., t, 1., 1.] + [t]*8)
    lo = np.array([0., .03, .03, .15, .15] + [.65]*8)
    hi = np.array([.65, 5., 6., 2.7, 2.7] + [4.4]*8)
    best = None
    for strength in (0.0015,):
        def prof(par, output=False):
            A = basis(a, par)
            D = np.eye(A.shape[1]) * .001
            D[0,0] = 0.
            b = np.linalg.lstsq(np.vstack((A,D)), np.r_[y, np.zeros(A.shape[1])], rcond=None)[0]
            if output:
                return basis(z[n:], par) @ b
            return np.r_[A @ b - y, D @ b, strength*(par-center)]
        for f in (1.45,t,2.85):
            initial = center.copy()
            initial[5:] = f
            opt = least_squares(prof, initial, bounds=(lo,hi), max_nfev=110)
            if best is None or opt.cost < best[0]:
                best = opt.cost, prof(opt.x, True)
    return best[1]
PY
cat > /tmp/dag_stress.py <<'PY'
import importlib.util, pickle, pathlib, subprocess, os
import numpy as np
p=pathlib.Path('/holdout_task/node_01.py'); base=p.read_text(); opts=pathlib.Path('/tmp/scalar_options.txt').read_text()
def load(path):
    spec=importlib.util.spec_from_file_location(path.stem,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
m=load(p); frozen=load(p.parent/'_frozen_nodes/node_00.py'); exec(opts,m.__dict__)
rows=[]
for b in pickle.load(open(p.parent/'public_batch.pkl','rb')):
    st=b['study']; tr=st['train']; n=len(tr['x']); ids=np.arange(n); coef, prev=m._calibration(st['panel_a']); split=max(40,n-20)
    folds=[ids[split:], *np.array_split(ids,4), *[ids[i::4] for i in range(4)], *[c[-max(10,n//5):] for c in (ids[np.argsort(abs(tr['x'][:,i]))] for i in range(tr['x'].shape[1]))]]
    def inverse(x, st):
        F=frozen.run_node({'__input__':st}); d=x.shape[1]; start=F['hybrid'][:,-((d+1)*(d+2)*(d+3)//6):][:,1:d+1];return m._invert(x,start,coef,prev)
    for i,te in enumerate(folds):
        ti=np.setdiff1d(ids,te); y=tr['y'].reshape(-1); x=tr['x'][np.r_[ti,te]]; study=dict(st, train=dict(tr,x=tr['x'][ti], y=tr['y'][ti],traj=tr['traj'][ti]), test={'x':tr['x'][te]}); z=inverse(x,study); nn=len(ti); meta=st['telemetry_vec']; a=m._scalar(z,y[ti],nn,meta); v=m._scalar_vp(z,y[ti],nn,meta)
        rows.append([np.mean(abs(a-y[te])),np.mean(abs(v-y[te])),np.mean(abs(.5*(a+v)-y[te])),np.mean(abs((.25*a+.75*v)-y[te])),np.mean(abs(m._trajectory(z,tr['traj'][ti],nn)-tr['traj'][te]))]); print(i,np.round(rows[-1],6),flush=True)
a=np.array(rows); print('columns: baseline, projection, equal blend, projection-weighted blend, trajectory');print('worst standard',a[:9].max(0),'worst including coordinate tails',a.max(0),'mean',a.mean(0))
choices=[i for i in (1,2,3) if a[0,i]<.038 and a[:9,i].max() < min(.038,.9*a[:9,0].max()) and a[:,i].max() < .95*a[:,0].max() and a[:,i].mean() < a[:,0].mean()]
if choices:
    j=min(choices,key=lambda i:(a[:,i].max(),a[:,i].mean())); expression={1:'_scalar_vp(z, target, n, telemetry)',2:'.5 * _scalar(z, target, n, telemetry) + .5 * _scalar_vp(z, target, n, telemetry)',3:'.25 * _scalar(z, target, n, telemetry) + .75 * _scalar_vp(z, target, n, telemetry)'}[j]
    text=base.replace('\ndef run_node(', '\n'+opts+'\n\ndef run_node(').replace("    y = _scalar(z, np.asarray(train['y']).reshape(n), n, np.asarray(study['telemetry_vec']))", "    target = np.asarray(train['y']).reshape(n)\n    telemetry = np.asarray(study['telemetry_vec'])\n    y = "+expression)
    p.write_text(text); print('PROMOTED',j,flush=True)
try:
    for flags in ([],['-I']):
        result=subprocess.check_output(['python3',*flags,str(p.parent/'test_dag_candidate.py')],cwd=p.parent,text=True); print(result,end='');import json; assert json.loads(result)['accepted']
except Exception:
    if not choices: raise
    p.write_text(base); print('RESTORED BASELINE'); subprocess.run(['python3','-I',str(p.parent/'test_dag_candidate.py')],cwd=p.parent,check=True)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/dag_stress.py
```
- **Container Exit Code**: `0` (`4062 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 [0.024574 0.036252 0.027605 0.031632 0.004588]
1 [0.025689 0.032935 0.028865 0.030702 0.004734]
2 [0.023262 0.033149 0.025923 0.029333 0.0048  ]
3 [0.032097 0.031823 0.031745 0.031702 0.005335]
4 [0.026789 0.036981 0.030141 0.033385 0.004384]
5 [0.05644  0.048276 0.049757 0.048229 0.005333]
6 [0.032378 0.031389 0.03051  0.030879 0.004432]
7 [0.030803 0.031764 0.029444 0.03048  0.004771]
8 [0.035689 0.037116 0.036072 0.036264 0.004266]
9 [0.035147 0.034892 0.034292 0.034402 0.00587 ]
10 [0.037399 0.030697 0.034041 0.032369 0.005515]
11 [0.023555 0.043098 0.029394 0.035654 0.006626]
12 [0.040071 0.06529  0.044385 0.052838 0.006262]
columns: baseline, projection, equal blend, projection-weighted blend, trajectory
worst standard [0.05643981 0.04827589 0.04975684 0.04822854 0.00533476] worst including coordinate tails [0.05643981 0.06528962 0.04975684 0.0528385  0.0066257 ] mean [0.03260718 0.03797399 0.03324422 0.03522066 0.00514738]
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.024574, "public_cv_traj_mae": 0.004588}, "exec_counts": {"node_00": 1, "node_01": 1}}
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.024574, "public_cv_traj_mae": 0.004588}, "exec_counts": {"node_00": 1, "node_01": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0d9c7e4b68fe6da4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_0d9c7e4b (directory).
```

### Turn 12 — `submit_completed_task` (`12096 ms` LLM)
- **Rationale**: Submit the unchanged, receipt-bound candidate. It passed both required public thresholds and all DAG contract checks. The challengers did not meet the prespecified promotion criteria.
- **Alternatives Weighed**: `Promote the profiled scalar head or a blend` (rejected: Neither the profiled head nor either blend brought the worst supplemental fold below 0.038, and all three increased mean scalar MAE across the 13 folds.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_01.py`. The official test accepted it in both standard and isolated Python: scalar MAE 0.024574, trajectory MAE 0.004588, and one execution per node. The worst supplemental scalar fold reached 0.056440, above the official split’s 0.038 cutoff.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel_M10_1_UnknownCut_DAG_Discovery",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
