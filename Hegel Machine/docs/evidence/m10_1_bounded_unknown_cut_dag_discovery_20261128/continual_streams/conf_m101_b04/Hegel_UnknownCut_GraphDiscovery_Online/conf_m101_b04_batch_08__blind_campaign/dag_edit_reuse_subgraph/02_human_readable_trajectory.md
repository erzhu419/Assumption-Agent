# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m101_b04_batch_08__blind_campaign__reuse_subgraph`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel_M10_1_UnknownCut_DAG_Discovery` |
| **Domain** | `blind_multi_stage_scientific_computation_graph` |
| **Wave / Wormhole Family** | `m10_1_bounded_unknown_cut_dag_discovery` / `m101_blind_dag` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `10 / 12` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `260,779` (`250,791` / `70,400` / `9,988`) |
| **Total Episode Wall Time (s)** | `410.07s` |

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

### Turn 1 — `execute_in_remote_docker` (`15652 ms` LLM)
- **Rationale**: Inspect the frozen interfaces and CV split before fitting either branch; a key, shape, or split mismatch would change the implementation and invalidate the reported errors.

```bash
cd /holdout_task && cat _dag_spec.json _frozen_nodes/node_00.py _frozen_nodes/node_03.py test_dag_candidate.py && python3 - <<'PY'
import pickle,numpy as np
with open('public_batch.pkl','rb') as f: d=pickle.load(f)
def show(x,k=''):
 if isinstance(x,dict):
  for a,b in x.items(): show(b,k+'/'+str(a))
 else:
  a=np.asarray(x); print(k,a.shape,a.dtype,repr(a.reshape(-1)[:10]))
show(d)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling Cauchy robust IRLS' 2>&1 | head -20
```
- **Container Exit Code**: `0` (`508 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-0.31108192, -1.02962921, -0.77371308, -0.79481763],
              [ 0.53259523, -1.1328076 ,  0.15054943, -0.44027736],
              [-0.79174968, -0.67789983, -0.50191507,  0.7855083 ],
              [ 1.11007916, -0.84288379, -0.52097912,  0.98915108],
              [-0.62062225, -0.61778596,  0.61028531,  0.01460956],
              [ 0.73894107, -0.7712393 , -0.32312672, -0.91412907],
              [ 0.47702272,  0.28738446, -0.4718904 ,  0.25106501],
              [-0.67843134,  0.32350271, -0.73997354,  0.16815547],
              [ 0.57259508,  0.99780904, -0.31590679, -0.04045835],
              [-0.87542137, -0.30919744, -0.40402264, -0.84160158],
              [-0.55170623,  0.0684792 , -0.65986443, -0.21173084],
              [ 0.24411369,  0.98953628, -0.27590321, -0.23580167],
              [ 1.02704811,  0.70547077, -0.94579846,  1.02835105],
              [-0.4185212 ,  0.53412553, -0.74690281, -0.59941545],
              [ 1.04171492,  0.79937657,  0.91135248, -0.47920531],
              [-0.91652875,  0.27611128,  0.02348722, -0.53355488],
              [ 1.0823997 , -0.95228217, -0.77495006, -0.9541187 ],
              [-0.40031814, -0.02938951, -0.94994587,  0.04226933],
              [-0.82612106, -0.87323803, -0.22167739, -0.4783706 ]])}, 'telemetry_vec': array([0.2, 0.2, 0.1, 1.6])}}                                                                                  ],
      dtype=object)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
```

### Turn 2 — `execute_in_remote_docker` (`16198 ms` LLM)
- **Rationale**: The first inspection printed the batch as an object and hid its dimensions and fold logic. A bounded schema summary will establish which targets reach each node and how to measure held-out errors without leakage.

```bash
cd /holdout_task; sed -n '43,170p' test_dag_candidate.py; cat public_asset_index.json; python3 - <<'PY'
import pickle,numpy as np
obj=pickle.load(open('public_batch.pkl','rb'))
def show(x,k='root',depth=0):
 if isinstance(x,dict):
  for a,b in x.items(): show(b,k+'/'+str(a),depth+1)
 elif isinstance(x,(list,tuple)) or isinstance(x,np.ndarray) and x.dtype==object:
  print(k,type(x).__name__,len(x))
  if len(x) and depth<4: show(x[0],k+'/0',depth+1)
 else:
  a=np.asarray(x); print(k,a.shape,a.dtype,str(a.ravel()[:8])[:110])
show(obj)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -24
```
- **Container Exit Code**: `0` (`463 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1"
}root list 1
root/0/study/eval_times (8,) float64 [0.25       0.64285714 1.03571429 1.42857143 1.82142857 2.21428571
 2.60714286 3.        ]
root/0/study/panel_a/u (48, 4) float64 [ 0.15488598 -0.99892935 -1.09106382 -0.23236593  0.22066113  0.01023507
 -0.11108656  1.23036232]
root/0/study/panel_a/v (48, 4) float64 [ 0.28415071 -1.0561401  -1.18365697 -0.15072029  0.34953654 -0.12565667
 -0.22091317  1.0995279 ]
root/0/study/panel_b/u (24, 4) float64 [-0.44510524 -0.41889611  0.5284348  -0.30694162  0.62039514  1.16018894
  1.00524379 -0.7316741 ]
root/0/study/panel_b/v (24, 4) float64 [ 0.31866687 -0.04076389 -0.4130763  -0.62680811 -0.84497576  0.43731505
 -1.16647999  0.36115017]
root/0/study/panel_c/u (24, 4) float64 [ 0.96836128  0.06123008 -0.381964    0.08062366  0.04178271  0.71680202
  0.61238606 -0.86331184]
root/0/study/panel_c/v1 (24, 4) float64 [-0.01139891 -0.00562623  0.93299684 -0.11653464 -0.18202094 -1.02553409
 -0.39829532 -0.91921512]
root/0/study/panel_c/v2 (24, 4) float64 [-0.29218631  0.26562784  0.39829195  0.54571683 -0.4270658  -0.09406961
 -0.19665755 -0.15758576]
root/0/study/train/x (72, 4) float64 [-0.93700706  1.08166934 -0.59948945 -0.34857135  0.04946606 -0.19564352
  0.35615746 -0.29075095]
root/0/study/train/y (72,) float64 [-1.69941003  0.17080994 -1.14715889 -0.25428602  0.0240093   0.1944763
 -0.84877167  0.07664995]
root/0/study/train/traj (72, 8) float64 [0.00750734 0.04506147 0.05277743 0.08010125 0.09839057 0.09935443
 0.1192449  0.11728101]
root/0/study/train/summary (4,) float64 [-0.3055232   0.61238197  0.06370743  0.3145892 ]
root/0/study/test/x (36, 4) float64 [ 0.81018732 -0.67547387  0.58160654 -0.17418217 -0.93463287 -0.79625278
  0.07367069  1.08820523]
root/0/study/telemetry_vec (4,) float64 [0.2 0.2 0.1 1.6]
{
  "ok": true,
  "target": "autogenous_sparse_ridge_certification.solve_and_certify_l0",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_sparse_ridge_certification",
      "granularity_tier": "L2_domain_operator",
      "compiled_code_sha256": "770f6644bb1fd8eb49dbacf51ffa3405a9ca876974e0021d66415d9014d8dfb9",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "n_samples",
          "op": ">=",
          "value": 1
        },
        {
          "field": "n_features",
          "op": "<=",
          "value": 16
        },
        {
```

### Turn 3 — `execute_in_remote_docker` (`31604 ms` LLM)
- **Rationale**: Compare nested polynomial bases and smooth kernels on the official split and four additional folds. If error remains above tolerance, coefficient structure and trajectory singular values will guide the next readout grammar.
- **Alternatives Weighed**: `Use panel C targets as direct labels` (rejected: They are four-dimensional calibration outputs, not scalar or trajectory labels, and node_02 does not receive them.)

```bash
cat >/tmp/probe.py <<'PY'
import numpy as np,pickle,runpy,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; f=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); d=f['run_node']({'__input__':s}); x=s['train']['x']; y=np.c_[s['train']['y'],d['traj']]; z=d['z']; a=f['_panel'](s['panel_a'],3); b=f['_panel'](s['panel_b']); w=f['_affine'](f['_affine'](x)@a)@b
print('affine',a.round(3),b.round(3),sep='\n'); print('std targets',y.std(0).round(4)); print('singular trajectory',np.linalg.svd(y[:,1:],compute_uv=False).round(4)); print('w mean/std',w.mean(0),w.std(0));
def poly(t,deg):
 p=[np.ones(len(t))]; names=['1'];
 for k in range(1,deg+1):
  for i in itertools.combinations_with_replacement(range(t.shape[1]),k): p.append(np.prod(t[:,i],axis=1)); names.append(''.join(map(str,i)))
 return np.array(p).T,names
spl=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),v),v) for v in np.array_split(np.random.default_rng(123).permutation(72),4)]
def test(K,lams=(1e-6,1e-4,1e-3,.01,.1,1)):
 results=[]
 for l in lams:
  err=[]
  for t,v in spl:
   p=K[np.ix_(v,t)]@np.linalg.solve(K[np.ix_(t,t)]+l*np.eye(len(t)),y[t]); r=np.abs(p-y[v]); err.append([r[:,0].mean(),r[:,1:].mean()])
  results.append(np.array(err))
 return np.array(results)
for name,t in [('x',x),('w',w),('z',z),('a',f['_affine'](x)@a)]:
 print(name)
 for deg in range(1,5):
  p,n=poly(t,deg); q=np.linalg.lstsq(p,y,rcond=None)[0]; g=test(p@p.T); print('poly',deg,'fit',np.mean(np.abs(p@q-y),axis=0)[[0,-1]].round(4),'official',g[0,0].round(4),'worst',g.max(1).min(0).round(4));
  if deg in (2,3) and name in ('x','w'): print('tops', [[(n[i],round(q[i,j],4)) for i in np.argsort(-np.abs(q[:,j]))[:10]] for j in (0,-1)])
 dist=((t[:,None,:]-t[None,:,:])**2).sum(2); p,_=poly(t,2)
 for e in (.04,.1,.25,.5,1):
  g=test(np.exp(-e*dist)+p@p.T); print('rbf',e,'official',g[:,0].min(0).round(4),'worst',g.max(1).min(0).round(4))
 u=s['panel_c']['u']; out=np.c_[s['panel_c']['v1'],s['panel_c']['v2']]; p,n=poly(u,2); q=np.linalg.lstsq(p,out,rcond=None)[0]; print('panel_c quad rmse',np.sqrt(((out-p@q)**2).mean(0)).round(4)) if name=='x' else None
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe.py
```
- **Container Exit Code**: `0` (`605 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cial [0.0389 0.0151] worst [0.0498 0.0183]
panel_c quad rmse [0.0566 0.0702 0.0681 0.0617 0.0019 0.0018 0.0017 0.0027]
w
poly 1 fit [0.2417 0.1186] official [0.3153 0.0792] worst [0.3153 0.1059]
poly 2 fit [0.0493 0.0274] official [0.057  0.0347] worst [0.0867 0.0328]
tops [[('1', 0.8635), ('02', 0.7606), ('01', -0.5336), ('12', -0.4435), ('33', -0.3875), ('03', 0.2736), ('11', -0.2727), ('3', 0.2313), ('23', 0.1935), ('2', -0.1619)], [('0', -0.5132), ('1', 0.3579), ('23', -0.2589), ('03', 0.2527), ('2', -0.1959), ('02', 0.1919), ('12', -0.1879), ('33', 0.1719), ('3', -0.1636), ('1', 0.1346)]]
poly 3 fit [0.0284 0.0119] official [0.1096 0.0306] worst [0.1095 0.0237]
tops [[('1', 1.0388), ('02', 0.7796), ('01', -0.5504), ('33', -0.4128), ('12', -0.3603), ('3', 0.3272), ('2', -0.3218), ('11', -0.2816), ('122', -0.2545), ('013', -0.232)], [('0', -0.543), ('1', 0.426), ('23', -0.286), ('2', -0.2733), ('03', 0.2186), ('12', -0.1968), ('02', 0.1794), ('33', 0.1576), ('1', 0.1481), ('13', 0.1465)]]
poly 4 fit [0.0004 0.0002] official [0.0738 0.0368] worst [0.0767 0.0244]
rbf 0.04 official [0.0521 0.0184] worst [0.0787 0.0206]
rbf 0.1 official [0.0348 0.0185] worst [0.0572 0.0189]
rbf 0.25 official [0.0321 0.0153] worst [0.0467 0.0153]
rbf 0.5 official [0.0316 0.0147] worst [0.0359 0.0147]
rbf 1 official [0.0422 0.0149] worst [0.0481 0.0219]
z
poly 1 fit [0.2417 0.1186] official [0.3153 0.0792] worst [0.3153 0.1059]
poly 2 fit [0.0493 0.0274] official [0.057  0.0347] worst [0.087  0.0298]
poly 3 fit [0.0284 0.0119] official [0.1096 0.0306] worst [0.1248 0.0291]
poly 4 fit [0.0004 0.0002] official [0.2331 0.1627] worst [0.2331 0.1612]
rbf 0.04 official [0.0361 0.0171] worst [0.0538 0.0179]
rbf 0.1 official [0.0328 0.0167] worst [0.0481 0.0167]
rbf 0.25 official [0.0335 0.0168] worst [0.0467 0.0168]
rbf 0.5 official [0.0394 0.0175] worst [0.0529 0.0185]
rbf 1 official [0.0481 0.0204] worst [0.068  0.0217]
a
poly 1 fit [0.2417 0.1186] official [0.3153 0.0792] worst [0.3153 0.1059]
poly 2 fit [0.0493 0.0274] official [0.057  0.0347] worst [0.0859 0.0305]
poly 3 fit [0.0284 0.0119] official [0.1096 0.0306] worst [0.1054 0.0252]
poly 4 fit [0.0004 0.0002] official [0.0725 0.0425] worst [0.0914 0.039 ]
rbf 0.04 official [0.0426 0.0171] worst [0.068 0.019]
rbf 0.1 official [0.0355 0.0176] worst [0.0534 0.0179]
rbf 0.25 official [0.0318 0.0156] worst [0.0453 0.0156]
rbf 0.5 official [0.0319 0.0138] worst [0.0353 0.0141]
rbf 1 official [0.0416 0.0149] worst [0.0526 0.02  ]
```

### Turn 4 — `execute_in_remote_docker` (`37199 ms` LLM)
- **Rationale**: Test whether panel-C readouts transfer to the labeled branches and whether the trajectory’s dominant time factor has a compact state law. Failed transfer would rule out using panel C as a direct predictor rather than just a structural clue.
- **Alternatives Weighed**: `Promote the passing mixed kernel immediately` (rejected: It gives a useful baseline, but compact structural fits could improve held-out margins or extrapolation.)
- **ATMS NOGOOD DECLARED**: Refuted `An unrestricted degree-two polynomial is sufficient for both branches.` | Witness: `Scalar MAE 0.0570 on the official split exceeds 0.038, and the degree-three polynomial worsens it to 0.1096.` | Blanket Diagnosis: `Adding monomials does not capture the missing scalar structure stably; the trajectory matrix also has a dominant singular mode that independent horizon fits do not exploit.` | Sublated Pivot: `Test compact nonlinear projections for the scalar and a shared-amplitude basis for the trajectory, with the passing mixed kernel as a comparator.`

```bash
cat >/tmp/structure.py <<'PY'
import itertools,pickle,runpy,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; f=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; c=s['panel_c']; u=c['u']; w=f['_affine'](f['_affine'](x)@f['_panel'](s['panel_a'],3))@f['_panel'](s['panel_b']); a=f['_affine'](x)@f['_panel'](s['panel_a'],3)
ix=list(itertools.combinations_with_replacement(range(4),2))
def Q(x): return np.column_stack([np.ones(len(x)),x,*[x[:,i]*x[:,j] for i,j in ix]])
R=np.random.default_rng(42); splits=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),v),v) for v in np.array_split(R.permutation(72),4)]
def check(p,Y):
 return np.array([np.abs(p[v]@np.linalg.lstsq(p[t],Y[t],rcond=None)[0]-Y[v]).mean() for t,v in splits]).round(4)
U,S,V=np.linalg.svd(T,full_matrices=False); print('time factors',V[:2].round(4),'amplitude min/max', (T@V[0]).min(),(T@V[0]).max())
for raw in [x,a,w]:
 print('curve test',[(k,check(Q(raw),np.arctanh(np.clip((T@V[0])/k,-.999,.999)))) for k in [1,1.5,2,3,4]])
co=np.linalg.lstsq(Q(u),c['v2'],rcond=None)[0]; print('v2 quad coeffs',co.round(3));
for name,raw in [('x',x),('a',a),('w',w)]:
 P=Q(raw)@co; print('v2 transfer',name,check(np.c_[np.ones(len(raw)),P],T),check(np.c_[np.ones(len(raw)),np.tanh(P)],T));
forms={'linear+sine':np.sin,'linear+tanh':np.tanh,'linear+softsign':lambda t:t/np.sqrt(1+t*t),'linear+cos':np.cos}
A=Q(u)[:,:5]; B=Q(u)
for name,phi in forms.items():
 for linear in [True,False]:
  pars=[]; errors=[]
  for j,Y in enumerate(c['v1'].T):
   base=A if linear else A[:,:1]
   models=[]
   for _ in range(16):
    p0=np.r_[np.linalg.lstsq(base,Y,rcond=None)[0],R.uniform(-.5,.5),R.normal(0,.8,5)]
    def pred(p,b,l):return b@p[:b.shape[1]]+p[-6]*phi(l@p[-5:])
    r=least_squares(lambda p:pred(p,base,A)-Y,p0,max_nfev=350); models.append((np.linalg.norm(r.fun),r.x))
   err,p=min(models,key=lambda x:x[0]); pars.append(p); errors.append(err/np.sqrt(len(u)))
  out=[]
  for raw in [x,a,w]:
   l=Q(raw)[:,:5]; b=l if linear else l[:,:1]; P=np.column_stack([pred(p,b,l) for p in pars]); out.append(check(np.c_[np.ones(len(raw)),P],y))
  print(name,linear,np.round(errors,4),out)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/structure.py
```
- **Container Exit Code**: `0` (`10784 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1 , 0.2624, 0.3224])), (2, array([0.253 , 0.2013, 0.2155, 0.1504, 0.1526])), (3, array([0.0363, 0.0441, 0.0351, 0.0307, 0.0271])), (4, array([0.0242, 0.0275, 0.0221, 0.0214, 0.0203]))]
v2 quad coeffs [[ 0.078  0.001  0.084  0.12 ]
 [-0.289  0.155  0.114  0.328]
 [-0.334 -0.093  0.137 -0.308]
 [-0.169 -0.255 -0.365  0.097]
 [ 0.076 -0.366  0.261  0.144]
 [-0.132  0.073  0.055  0.152]
 [-0.005 -0.     0.003 -0.002]
 [ 0.015 -0.001 -0.013 -0.001]
 [-0.007  0.002  0.002 -0.001]
 [-0.148 -0.042  0.061 -0.138]
 [ 0.002 -0.001 -0.005  0.001]
 [ 0.002  0.003 -0.001  0.002]
 [-0.08  -0.116 -0.171  0.046]
 [ 0.009 -0.    -0.005  0.007]
 [ 0.034 -0.171  0.133  0.07 ]]
v2 transfer x [0.1057 0.1362 0.1129 0.1452 0.1121] [0.1037 0.1395 0.1152 0.1432 0.1079]
v2 transfer a [0.1    0.1329 0.11   0.1374 0.1093] [0.0986 0.1341 0.1128 0.1385 0.1097]
v2 transfer w [0.0914 0.1187 0.1024 0.1218 0.1176] [0.0821 0.1142 0.102  0.119  0.1139]
linear+sine True [0.0469 0.0408 0.0427 0.046 ] [array([0.3236, 0.2617, 0.2687, 0.2141, 0.3451]), array([0.3229, 0.2478, 0.249 , 0.2175, 0.3043]), array([0.3102, 0.2465, 0.2323, 0.1902, 0.3208])]
linear+sine False [0.0937 0.0834 0.0885 0.0786] [array([0.3166, 0.2545, 0.2548, 0.2093, 0.338 ]), array([0.3055, 0.2504, 0.2537, 0.1992, 0.3323]), array([0.3051, 0.2353, 0.2625, 0.2066, 0.3206])]
linear+tanh True [0.0515 0.0407 0.0414 0.0461] [array([0.3256, 0.2514, 0.2749, 0.2087, 0.345 ]), array([0.3253, 0.241 , 0.2563, 0.2093, 0.3177]), array([0.3176, 0.2523, 0.2472, 0.1822, 0.3304])]
linear+tanh False [0.0936 0.0835 0.0885 0.0787] [array([0.3146, 0.2514, 0.2557, 0.2089, 0.3364]), array([0.3057, 0.2468, 0.2543, 0.2021, 0.3326]), array([0.305 , 0.2355, 0.2619, 0.2062, 0.3207])]
linear+softsign True [0.0515 0.0407 0.0412 0.0461] [array([0.3253, 0.2506, 0.2749, 0.2079, 0.3451]), array([0.3247, 0.2399, 0.2561, 0.2094, 0.3174]), array([0.3156, 0.2516, 0.2462, 0.179 , 0.3301])]
linear+softsign False [0.0936 0.0836 0.0886 0.0788] [array([0.3142, 0.2506, 0.2559, 0.2087, 0.336 ]), array([0.3058, 0.246 , 0.2544, 0.2025, 0.3326]), array([0.3051, 0.2354, 0.2619, 0.2061, 0.3207])]
linear+cos True [0.0469 0.0408 0.0427 0.046 ] [array([0.3236, 0.2617, 0.2687, 0.2141, 0.3451]), array([0.3229, 0.2478, 0.249 , 0.2175, 0.3043]), array([0.3102, 0.2465, 0.2323, 0.1902, 0.3208])]
linear+cos False [0.0937 0.0834 0.0885 0.0786] [array([0.3166, 0.2545, 0.2548, 0.2093, 0.338 ]), array([0.3055, 0.2504, 0.2537, 0.1992, 0.3323]), array([0.3051, 0.2353, 0.2625, 0.2066, 0.3206])]
```

### Turn 5 — `execute_in_remote_docker` (`41821 ms` LLM)
- **Rationale**: Install a working comparator and test compact saturating and oscillatory links with coefficients fitted anew in each fold. Promote a structural model only if it improves held-out prediction rather than just fitting the training rows more closely.
- **Alternatives Weighed**: `Transfer the fitted panel-C readouts directly` (rejected: Even the best probed transfer gave official-split MAE above 0.30 for scalar and 0.08 for trajectory.)

```bash
cat >/holdout_task/node_01.py <<'PY'
import numpy as np


def _kernel(x, u):
    dot = x @ u.T
    distance = np.maximum((x*x).sum(1)[:, None] + (u*u).sum(1) - 2*dot, 0)
    # Affine and quadratic trend with a smooth residual.
    quad = (dot*dot + (x*x) @ (u*u).T) / 2
    return 1 + dot + quad + np.exp(-0.5*distance)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x, xt = study['train']['x'], study['test']['x']
    gram = _kernel(x, x)
    gram.flat[::len(x)+1] += 0.001
    weights = np.linalg.solve(gram, study['train']['y'])
    return {'y_hat': _kernel(xt, x) @ weights}
PY
cat >/holdout_task/node_02.py <<'PY'
import numpy as np


def _kernel(x, u):
    dot = x @ u.T
    distance = np.maximum((x*x).sum(1)[:, None] + (u*u).sum(1) - 2*dot, 0)
    quad = (dot*dot + (x*x) @ (u*u).T) / 2
    return 1 + dot + quad + np.exp(-0.1*distance)


def run_node(inputs: dict) -> dict:
    state = inputs['node_00']
    z, zt, traj = state['z'], state['zt'], state['traj']
    gram = _kernel(z, z)
    gram.flat[::len(z)+1] += 0.003
    weights = np.linalg.solve(gram, traj)
    return {'trajectories': (_kernel(zt, z) @ weights)[None, :, :]}
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/test_dag_candidate.py
cat >/tmp/links.py <<'PY'
import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; ix=list(itertools.combinations_with_replacement(range(4),2))
Q=np.column_stack([np.ones(len(x)),x,*[x[:,i]*x[:,j] for i,j in ix]])
A=Q[:,:5]; r=np.random.default_rng(293); N=np.arange(len(x)); splits=[(N[:52],N[52:])]+[(np.setdiff1d(N,v),v) for v in np.array_split(r.permutation(len(x)),4)]
links={'tanh':np.tanh,'sin':np.sin,'rational':lambda a:a/(1+a*a),'softsign':lambda a:a/np.sqrt(1+a*a),'atan':np.arctan}
for target in ('scalar','trajectory'):
 for label,phi in links.items():
  for bn,ln in ((5,15),(15,5),(15,15),(1,15)):
   errors=[]; fits=[]
   for i,j in splits:
    if target=='scalar': Y=y[i]
    else:
     _,_,V=np.linalg.svd(T[i],full_matrices=False); Y=T[i]@V[0]
    B=Q[i,:bn]; L=Q[i,:ln]; Bp=np.linalg.pinv(B); Z=Y-B@(Bp@Y); slope=np.linalg.lstsq(L,Y,rcond=None)[0]; pairs=[]
    for k in range(8):
     p=np.r_[1.5,np.zeros(ln)]; p[1:]=slope/(1+np.std(Y)) if not k else r.normal(0,.55,ln)
     def res(p):
      g=p[0]*phi(L@p[1:]); return np.r_[g-B@(Bp@g)-Z, .002*p]
     opt=least_squares(res,p,max_nfev=200); pairs.append((np.linalg.norm(opt.fun),opt.x))
    err,p=min(pairs,key=lambda x:x[0]); b=Bp@(Y-p[0]*phi(L@p[1:])); pred=Q[j,:bn]@b+p[0]*phi(Q[j,:ln]@p[1:]); fits.append(err/len(i)**.5)
    if target=='trajectory': pred=pred[:,None]*V[0]+ Q[j]@np.linalg.lstsq(Q[i],T[i]-Y[:,None]*V[0],rcond=None)[0]
    errors.append(abs(pred-(y[j] if target=='scalar' else T[j])).mean())
   print(target,label,bn,ln,'fit',round(np.mean(fits),4),'errors',np.round(errors,4),flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/links.py
```
- **Container Exit Code**: `0` (`39317 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py', '/holdout_task/node_02.py']`
- **Stdout (tail)**:
```text
6 0.0573 0.0393 0.0326]
scalar rational 1 15 fit 0.0509 errors [0.0435 0.056  0.0567 0.0456 0.0525]
scalar softsign 5 15 fit 0.0269 errors [0.0382 0.0362 0.0393 0.0258 0.0492]
scalar softsign 15 5 fit 0.0429 errors [0.0783 0.0703 0.0868 0.0589 0.067 ]
scalar softsign 15 15 fit 0.0144 errors [0.0453 0.0296 0.0579 0.0382 0.0306]
scalar softsign 1 15 fit 0.0508 errors [0.0419 0.0553 0.0702 0.0444 0.0515]
scalar atan 5 15 fit 0.0269 errors [0.0371 0.035  0.039  0.0256 0.05  ]
scalar atan 15 5 fit 0.0441 errors [0.0776 0.0703 0.1202 0.0639 0.0687]
scalar atan 15 15 fit 0.0146 errors [0.0485 0.0297 0.0578 0.0381 0.0812]
scalar atan 1 15 fit 0.0505 errors [0.0409 0.055  0.0684 0.0438 0.0511]
trajectory tanh 5 15 fit 0.0415 errors [0.034  0.0202 0.031  0.0166 0.0216]
trajectory tanh 15 5 fit 0.0449 errors [0.0344 0.0217 0.0365 0.0232 0.0349]
trajectory tanh 15 15 fit 0.0226 errors [0.0449 0.0207 0.0227 0.0366 0.0408]
trajectory tanh 1 15 fit 0.0479 errors [0.015  0.0163 0.0198 0.0169 0.0208]
trajectory sin 5 15 fit 0.0438 errors [0.037  0.0198 0.0216 0.0164 0.0208]
trajectory sin 15 5 fit 0.0453 errors [0.0346 0.021  0.035  0.0204 0.0336]
trajectory sin 15 15 fit 0.0238 errors [0.0464 0.0187 0.0354 0.0238 0.0338]
trajectory sin 1 15 fit 0.048 errors [0.0222 0.0165 0.0213 0.0169 0.0205]
trajectory rational 5 15 fit 0.042 errors [0.034  0.02   0.0446 0.0166 0.0209]
trajectory rational 15 5 fit 0.0443 errors [0.0344 0.022  0.037  0.0242 0.0415]
trajectory rational 15 15 fit 0.0256 errors [0.0475 0.0251 0.0433 0.031  0.036 ]
trajectory rational 1 15 fit 0.0479 errors [0.0161 0.0164 0.0201 0.0168 0.0207]
trajectory softsign 5 15 fit 0.0416 errors [0.0341 0.0202 0.0307 0.0167 0.022 ]
trajectory softsign 15 5 fit 0.0443 errors [0.0344 0.0216 0.0364 0.0278 0.0271]
trajectory softsign 15 15 fit 0.0208 errors [0.0319 0.0215 0.0423 0.0287 0.0339]
trajectory softsign 1 15 fit 0.048 errors [0.0145 0.0162 0.0194 0.0171 0.0208]
trajectory atan 5 15 fit 0.0418 errors [0.0341 0.0202 0.0299 0.0168 0.0223]
trajectory atan 15 5 fit 0.0438 errors [0.0344 0.0216 0.0363 0.0274 0.0407]
trajectory atan 15 15 fit 0.0239 errors [0.0407 0.0253 0.0216 0.0321 0.0419]
trajectory atan 1 15 fit 0.0482 errors [0.0143 0.0161 0.0192 0.0174 0.0208]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5b410e722de61a9a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5b410e72 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`26035 ms` LLM)
- **Rationale**: Run the exact installed nodes in isolated Python on fresh folds and on the largest-radius inputs. If scalar error rises materially at the edge, use the frozen coordinate conventions to test a more extrapolative basis; measure trajectory error separately by horizon.
- **Alternatives Weighed**: `Replace the scalar kernel with a single saturating, oscillatory, or rational projection` (rejected: The strongest such scalar alternatives had worse worst-fold error than the mixed kernel.)

```bash
cat /holdout_task/_frozen_nodes/node_00.py /holdout_task/_frozen_nodes/node_03.py; cat >/tmp/stress.py <<'PY'
import copy,pickle,runpy,numpy as np
root='/holdout_task/'
fns=[runpy.run_path(root+('_frozen_nodes/' if i in (0,3) else '')+f'node_{i:02d}.py')['run_node'] for i in range(4)]
s=pickle.load(open(root+'public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; n=len(x); N=np.arange(n)
rng=np.random.default_rng(427); splits=[(f'random-{j}',v) for j,v in enumerate(np.array_split(rng.permutation(n),4))]; splits += [('outer-radius',np.argsort(np.linalg.norm(x,axis=1))[-12:])]
for label,v in splits:
 t=np.setdiff1d(N,v); study=copy.deepcopy(s)
 for key in ('x','y','traj'): study['train'][key]=s['train'][key][t]
 study['train']['summary']=np.array([study['train']['y'].mean(),study['train']['y'].std(),study['train']['traj'][:,-1].mean(),study['train']['traj'][:,-1].std()]); study['test']['x']=x[v]
 shared=fns[0]({'__input__':study}); a=fns[1]({'__input__':study,'node_00':shared}); b=fns[2]({'node_00':shared}); out=fns[3]({'node_01':a,'node_02':b})
 err=abs(out['state_pred']-s['train']['traj'][v]); print(label,round(abs(out['y_hat']-s['train']['y'][v]).mean(),6),round(err.mean(),6),round(err.mean(0).max(),6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/stress.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`557 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _panel(panel, trim=0):
    u = _affine(panel['u'])
    v = panel['v']
    keep = np.arange(len(u))
    for _ in range(5 if trim else 0):
        c = np.linalg.lstsq(u[keep], v[keep], rcond=None)[0]
        keep = np.argsort(np.linalg.norm(u @ c - v, axis=1))[:len(u)-trim]
    return np.linalg.lstsq(u[keep], v[keep], rcond=None)[0]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    x, xt = train['x'], study['test']['x']
    a, b = _panel(study['panel_a'], 3), _panel(study['panel_b'])

    def transform(u):
        return _affine(_affine(u) @ a) @ b

    w, wt = transform(x), transform(xt)
    z, zt = (w-w.mean(0))/w.std(0), (wt-w.mean(0))/w.std(0)
    return {'z': z, 'zt': zt, 'traj': train['traj']}
import numpy as np


def run_node(inputs: dict) -> dict:
    y_hat = inputs['node_01']['y_hat']
    trajectories = inputs['node_02']['trajectories']
    state_pred = trajectories.mean(axis=0)
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}
random-0 0.037307 0.014439 0.01832
random-1 0.035861 0.013388 0.017643
random-2 0.0306 0.013475 0.019068
random-3 0.0376 0.017583 0.022985
outer-radius 0.060711 0.018859 0.026931
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.034247, "public_cv_traj_mae": 0.018353}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1, "node_03": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5b410e722de61a9a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5b410e72 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`75734 ms` LLM)
- **Rationale**: Compare additive and pairwise residual structure in the raw and calibrated coordinates, plus a regularized cubic trend. Directional tail splits will show whether an apparent edge improvement generalizes across directions; retain the trajectory node unchanged.
- **Alternatives Weighed**: `Retain the verified scalar baseline without further comparison` (rejected: It passes official CV, but its outer-radius error exposes a material extrapolation weakness. The trajectory branch does not show the same weakness.)

```bash
cat >/tmp/edge.py <<'PY'
import pickle,runpy,numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; N=np.arange(len(x)); f=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); w=f['_affine'](f['_affine'](x)@f['_panel'](s['panel_a'],3))@f['_panel'](s['panel_b']); c=s['panel_c']; u=c['u']
for k in ('v1','v2'):
 for basis in ([u,u*u],[u,u*u,u**3],[u,u*u,u**3,u**4],[u,u*u,u**3,np.sin(3*u),np.cos(3*u)]):
  P=np.column_stack([np.ones(len(u)),*basis]); q=np.linalg.lstsq(P,c[k],rcond=None)[0]; print(k,P.shape[1],np.sqrt(np.mean((P@q-c[k])**2)),np.round(q[1:],2).tolist())
splits=[('public',N[52:]),('edge',np.argsort(np.linalg.norm(x,axis=1))[-12:])]+[(str(i),v) for i,v in enumerate(np.array_split(np.random.default_rng(427).permutation(len(x)),4))]
splits += [('edge-'+str(j)+str(d),np.argsort(x[:,j]*d)[-12:]) for j in range(4) for d in [-1,1]]
models=[]
x1=x-x.mean(0); w1=w-w.mean(0)
dot=x@x.T; quad=1+dot+(dot**2+(x*x)@(x*x).T)/2
for name,b in [('x',x1),('w',w1),('white',(x1@np.linalg.inv(np.linalg.cholesky(np.cov(x1.T))).T)*.55)]:
 D=(b[:,None,:]-b[None,:,:])**2; g=b@b.T
 for e in (.2,.4,.6,1.):
  for a in (0,.1,.4,1):
   for label,k in [('rbf',np.exp(-e*D.sum(2))),('add',np.exp(-e*D).sum(2)),('pair',sum(np.exp(-e*(D[:,:,i]+D[:,:,j])) for i,j in itertools.combinations(range(4),2))),('matern',(1+np.sqrt(e*D.sum(2)))*np.exp(-np.sqrt(e*D.sum(2))))]:
    for reg in (.0003,.002,.01): models.append(((name,label,e,a,reg),quad+k+a*g**3,reg))
for label,K,reg in models:
 errors=[]
 for _,v in splits:
  t=np.setdiff1d(N,v); p=K[np.ix_(v,t)]@np.linalg.solve(K[np.ix_(t,t)]+reg*np.eye(len(t)),y[t]); errors.append(np.abs(y[v]-p).mean())
 scores=np.array(errors)
 if 'results' not in globals():results=[]
 results.append((label,scores))
for label,e in sorted(results,key=lambda a:max(a[1][:6]))[:12]: print(label,e.round(4))
print('all-slice robustness')
for label,e in sorted(results,key=lambda a:max(a[1]))[:10]: print(label,e.round(4))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/edge.py
```
- **Container Exit Code**: `0` (`1525 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
3  0.0694 0.0454]
('white', 'rbf', 1.0, 0, 0.002) [0.0407 0.0508 0.0455 0.0381 0.0369 0.038  0.0458 0.064  0.0814 0.0614
 0.0324 0.0426 0.0696 0.043 ]
('x', 'rbf', 1.0, 0, 0.0003) [0.0399 0.0428 0.0528 0.0384 0.0355 0.0321 0.0404 0.0617 0.0813 0.0666
 0.0349 0.0538 0.058  0.0623]
('x', 'rbf', 1.0, 0, 0.002) [0.0396 0.0426 0.0528 0.0386 0.0354 0.0322 0.04   0.0615 0.0817 0.066
 0.0354 0.053  0.057  0.0621]
('x', 'rbf', 1.0, 0, 0.01) [0.0389 0.0421 0.0536 0.0395 0.035  0.0331 0.0389 0.0604 0.0835 0.0643
 0.0376 0.0514 0.0542 0.0625]
('x', 'rbf', 0.6, 0, 0.0003) [0.0355 0.0539 0.0412 0.0351 0.0315 0.0351 0.0549 0.0732 0.0686 0.0641
 0.0366 0.0356 0.0649 0.0477]
('x', 'rbf', 0.6, 0, 0.01) [0.034  0.054  0.0442 0.0378 0.0346 0.0331 0.0496 0.0676 0.0751 0.0629
 0.0357 0.0346 0.0521 0.0467]
('x', 'rbf', 0.6, 0, 0.002) [0.0336 0.054  0.0411 0.0355 0.032  0.0341 0.0518 0.0718 0.0693 0.0644
 0.0348 0.0333 0.0583 0.044 ]
all-slice robustness
('w', 'rbf', 1.0, 0, 0.002) [0.0346 0.0487 0.0386 0.039  0.0324 0.0336 0.0459 0.0704 0.0683 0.0602
 0.0334 0.0365 0.0567 0.0461]
('w', 'rbf', 1.0, 0.1, 0.002) [0.0379 0.0596 0.0341 0.0449 0.0335 0.0324 0.0562 0.0626 0.0711 0.066
 0.0314 0.0362 0.0545 0.0485]
('w', 'rbf', 1.0, 0, 0.0003) [0.0351 0.0475 0.0387 0.0388 0.0322 0.0342 0.0474 0.0712 0.0669 0.0612
 0.0332 0.0396 0.0608 0.0469]
('w', 'rbf', 1.0, 0.1, 0.0003) [0.0381 0.0594 0.0335 0.0446 0.0333 0.0329 0.0582 0.0615 0.0714 0.066
 0.0305 0.0387 0.0599 0.0502]
('x', 'rbf', 0.6, 0, 0.002) [0.0336 0.054  0.0411 0.0355 0.032  0.0341 0.0518 0.0718 0.0693 0.0644
 0.0348 0.0333 0.0583 0.044 ]
('x', 'rbf', 0.6, 0, 0.0003) [0.0355 0.0539 0.0412 0.0351 0.0315 0.0351 0.0549 0.0732 0.0686 0.0641
 0.0366 0.0356 0.0649 0.0477]
('w', 'rbf', 1.0, 0, 0.01) [0.0341 0.0499 0.0408 0.0402 0.0345 0.0315 0.0435 0.0681 0.0736 0.0584
 0.0351 0.0337 0.0509 0.0471]
('w', 'rbf', 1.0, 0.1, 0.01) [0.0382 0.0588 0.037  0.0464 0.034  0.0314 0.0524 0.0631 0.0744 0.0628
 0.034  0.037  0.0465 0.0483]
('x', 'rbf', 0.6, 0, 0.01) [0.034  0.054  0.0442 0.0378 0.0346 0.0331 0.0496 0.0676 0.0751 0.0629
 0.0357 0.0346 0.0521 0.0467]
('white', 'rbf', 0.6, 0, 0.0003) [0.0395 0.0559 0.0356 0.0382 0.0366 0.0348 0.0565 0.0713 0.0694 0.0767
 0.0479 0.0383 0.0745 0.0393]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5b410e722de61a9a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5b410e72 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`36203 ms` LLM)
- **Rationale**: Fit the coordinatewise basis using panel C only, then fit scalar weights within each training split. If the resulting basis does not improve the tails while preserving official acceptance, prefer the simpler calibrated RBF challenger.
- **Alternatives Weighed**: `Promote the calibrated RBF immediately` (rejected: It improves the radial slice, but the panel data suggest a coordinatewise oscillatory basis that has not yet been tested in combination with the scalar trend.)

```bash
cat >/tmp/panel_basis.py <<'PY'
import pickle,runpy,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; c=s['panel_c']; u=c['u']; N=np.arange(len(x)); f=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); af=f['_affine']; a=af(x)@f['_panel'](s['panel_a'],3); w=af(a)@f['_panel'](s['panel_b']); Y=c['v1']
def basis(X,freq,mode):
    args=X*freq
    if mode==0: return np.c_[np.ones(len(X)), np.sin(args),np.cos(args)]
    if mode==1: return np.c_[np.ones(len(X)), X, np.sin(args),np.cos(args)]
    return np.c_[np.ones(len(X)), X, np.sin(args)]
def fit(freq,mode):
    F=basis(u,freq,mode); return np.linalg.lstsq(F,Y,rcond=None)[0]
models=[]
for mode in range(3):
    def residual(freq): return (basis(u,freq,mode)@fit(freq,mode)-Y).ravel()
    opt=min((least_squares(residual,np.full(4,k),bounds=(.15,8),max_nfev=150) for k in (1.5,2.5,3.5,4.5,5.5)),key=lambda q:np.linalg.norm(q.fun)); freq=opt.x; co=fit(freq,mode); print('panel',mode,freq.round(4),np.std(opt.fun)); models.append((mode,freq,co))
vs=[('official',N[52:]),('radius',np.argsort(np.linalg.norm(x,axis=1))[-12:])]+[(str(i),v) for i,v in enumerate(np.array_split(np.random.default_rng(427).permutation(len(x)),4))]+[(f'{i}:{d}',np.argsort(d*x[:,i])[-12:]) for i in range(4) for d in (-1,1)]
def trend(X):
    D=X@X.T; return 1+D+(D**2+(X*X)@(X*X).T)/2
def radial(X,k):return np.exp(-k*np.sum((X[:,None]-X[None,:])**2,axis=-1))
def evaluate(K,r):
    e=[]
    for _,v in vs:
     t=np.setdiff1d(N,v); p=K[np.ix_(v,t)]@np.linalg.solve(K[np.ix_(t,t)]+r*np.eye(len(t)),y[t]); e.append(np.mean(abs(p-y[v])))
    return np.array(e)
Q=trend(x); old=evaluate(Q+radial(x,.5),.001); print('installed',old.round(4)); print('calibrated',evaluate(Q+radial(w,1),.002).round(4)); results=[]
for coord,X in [('x',x),('a',a),('w',w)]:
 for mode,ff,co in models:
    Z=basis(X,ff,mode)@co; S=trend(Z); J=Z@Z.T
    for weight in (.1,1,10):
     for tag,K in [('replace',S),('combine',Q+weight*S),('rbf',Q+weight*radial(Z,1)),('interaction',Q+weight*J*(1+x@x.T)),('additive',Q+weight*(basis(X,ff,mode)@basis(X,ff,mode).T))]:
      for mix in (0,.3,1.):
       r=.002; e=evaluate(K+mix*radial(w,1),r); results.append(((coord,mode,weight,tag,mix),e))
for lab,e in sorted((r for r in results if r[1][0]<=.038),key=lambda r:r[1].max())[:10]: print(lab,e.round(4))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/panel_basis.py
```
- **Container Exit Code**: `0` (`1354 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
panel 0 [1.6122 1.4839 1.6992 1.6991] 0.022631910062957408
panel 1 [3.6835 3.6638 3.9374 3.6822] 0.00764085139890835
panel 2 [3.7333 3.5864 3.9245 3.7693] 0.008998626344961399
installed [0.0342 0.0607 0.0373 0.0359 0.0306 0.0376 0.0561 0.075  0.0693 0.0723
 0.0355 0.0374 0.0604 0.0432]
calibrated [0.0346 0.0487 0.0386 0.039  0.0324 0.0336 0.0459 0.0704 0.0683 0.0602
 0.0334 0.0365 0.0567 0.0461]
('w', 2, 1, 'additive', 1.0) [0.034  0.047  0.0356 0.0369 0.0363 0.0352 0.0507 0.0503 0.0564 0.0568
 0.0346 0.0476 0.0524 0.0477]
('w', 1, 0.1, 'additive', 1.0) [0.0333 0.0439 0.0333 0.032  0.032  0.0338 0.0393 0.0469 0.0466 0.0474
 0.0337 0.0571 0.0572 0.0526]
('w', 2, 0.1, 'additive', 1.0) [0.0329 0.0469 0.0363 0.0363 0.0358 0.0342 0.0447 0.0642 0.0642 0.0545
 0.0335 0.0446 0.0542 0.0486]
('w', 2, 10, 'additive', 1.0) [0.0362 0.0526 0.0354 0.0383 0.0357 0.0359 0.0619 0.0321 0.0454 0.0643
 0.0374 0.0488 0.0504 0.0475]
('w', 0, 0.1, 'additive', 1.0) [0.0347 0.0497 0.0369 0.0378 0.0322 0.0343 0.0427 0.0646 0.0637 0.0628
 0.0333 0.0352 0.053  0.0471]
('w', 0, 0.1, 'interaction', 1.0) [0.0349 0.0521 0.0374 0.0379 0.0314 0.0349 0.0434 0.066  0.0675 0.0646
 0.0343 0.0349 0.0519 0.046 ]
('a', 0, 0.1, 'combine', 1.0) [0.0341 0.0476 0.0372 0.0388 0.0305 0.033  0.0387 0.0711 0.0676 0.0603
 0.0314 0.0321 0.0469 0.0465]
('w', 2, 0.1, 'interaction', 1.0) [0.0337 0.0511 0.0398 0.0383 0.0325 0.0363 0.0466 0.0675 0.0711 0.0658
 0.0342 0.0381 0.0554 0.0476]
('w', 1, 0.1, 'interaction', 1.0) [0.0338 0.0513 0.0398 0.0386 0.0325 0.0362 0.0464 0.0677 0.0713 0.0659
 0.0342 0.0382 0.0551 0.0478]
('w', 0, 0.1, 'combine', 1.0) [0.0369 0.0536 0.0385 0.0388 0.0308 0.0339 0.0439 0.0659 0.0722 0.0626
 0.0338 0.0377 0.0542 0.0463]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5b410e722de61a9a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5b410e72 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`37186 ms` LLM)
- **Rationale**: Promote the scalar challenger because it improves both radial and directional tails without sacrificing official CV. Leave the trajectory branch unchanged and verify the installed DAG in isolated Python, including directional tails and the worst trajectory horizon.
- **Alternatives Weighed**: `Use the calibrated Gaussian residual without harmonics` (rejected: Its worst-slice scalar MAE was 0.0704 versus 0.0572 with harmonics, with no offsetting official-split improvement.)

```bash
cat >/holdout_task/node_01.py <<'PY'
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel, trim=0):
    design, values = _affine(panel['u']), panel['v']
    keep = np.arange(len(design))
    for _ in range(5 if trim else 0):
        coef = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
        keep = np.argsort(np.linalg.norm(design @ coef - values, axis=1))[:-trim]
    return np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]


def _harmonics(x, frequency):
    phase = x * frequency
    return np.column_stack((_affine(x), np.sin(phase), np.cos(phase)))


def _panel_frequencies(panel):
    u, values = panel['u'], panel['v1']

    def residual(frequency):
        design = _harmonics(u, frequency)
        coef = np.linalg.lstsq(design, values, rcond=None)[0]
        return (design @ coef - values).ravel()

    fits = (least_squares(residual, np.full(4, start), bounds=(0.15, 8),
                          max_nfev=150) for start in (1.5, 2.5, 3.5, 4.5, 5.5))
    return min(fits, key=lambda fit: fit.cost).x


def _kernel(x, u, w, v, harmonics, reference):
    dot = x @ u.T
    quadratic = (dot**2 + (x*x) @ (u*u).T) / 2
    distance = np.sum((w[:, None] - v[None, :])**2, axis=-1)
    return 1 + dot + quadratic + np.exp(-distance) + 0.1 * (harmonics @ reference.T)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    shared = inputs['node_00']
    x, xt = study['train']['x'], study['test']['x']
    # Restore panel coordinates: the harmonic frequencies are in panel units.
    a, b = _calibration(study['panel_a'], 3), _calibration(study['panel_b'])
    calibrated = _affine(_affine(x) @ a) @ b
    scale, center = calibrated.std(0), calibrated.mean(0)
    w = shared['z'] * scale + center
    wt = shared['zt'] * scale + center
    frequency = _panel_frequencies(study['panel_c'])
    h, ht = _harmonics(w, frequency), _harmonics(wt, frequency)
    gram = _kernel(x, x, w, w, h, h)
    gram.flat[::len(x)+1] += 0.002
    weights = np.linalg.solve(gram, study['train']['y'])
    return {'y_hat': _kernel(xt, x, wt, w, ht, h) @ weights}
PY
sed -e '/^for label,v in splits:/i splits += [(f"edge-{j}:{d}", np.argsort(d*x[:,j])[-12:]) for j in range(x.shape[1]) for d in (-1, 1)]' /tmp/stress.py >/tmp/stress_final.py
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/test_dag_candidate.py && python3 -I /tmp/stress_final.py
```
- **Container Exit Code**: `0` (`1106 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.03332, "public_cv_traj_mae": 0.018353}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1, "node_03": 1}}
random-0 0.033302 0.014439 0.01832
random-1 0.032005 0.013388 0.017643
random-2 0.032015 0.013475 0.019068
random-3 0.033803 0.017583 0.022985
outer-radius 0.043912 0.018859 0.026931
edge-0:-1 0.039332 0.021817 0.033818
edge-0:1 0.046873 0.019472 0.028404
edge-1:-1 0.046602 0.017852 0.024034
edge-1:1 0.047422 0.0255 0.039206
edge-2:-1 0.033652 0.017291 0.027202
edge-2:1 0.057123 0.027158 0.037999
edge-3:-1 0.057154 0.020256 0.029772
edge-3:1 0.052591 0.027579 0.043945

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=dfc3bc2327b03519 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_dfc3bc23 (directory).
```

### Turn 10 — `submit_completed_task` (`14712 ms` LLM)
- **Rationale**: The exact saved bundle passed isolated Python verification and the public DAG contract. Stress tests confirmed the scalar improvement and quantified the remaining tail errors; the trajectory branch remained unchanged from the accepted baseline.
- **Alternatives Weighed**: `Keep the isotropic scalar baseline` (rejected: The calibrated harmonic model lowered public CV error and worst directional and radial tail errors.)
- **Submitted Final Answer / Summary**: `Implemented and verified both editable nodes; the frozen nodes were unchanged. The public DAG test passed with scalar MAE 0.033320 and trajectory MAE 0.018353, both below 0.038. On additional stress splits, the worst scalar MAE was 0.057154 and the worst trajectory MAE was 0.027579.`

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
