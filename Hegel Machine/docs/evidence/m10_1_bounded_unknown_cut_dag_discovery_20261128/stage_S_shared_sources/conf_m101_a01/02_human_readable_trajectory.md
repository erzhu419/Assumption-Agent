# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m101_a01_batch_00__blind_campaign`

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
| **Turns Used / Max Turns** | `12 / 14` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `367,396` (`354,016` / `130,560` / `13,380`) |
| **Total Episode Wall Time (s)** | `448.64s` |

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

### Turn 1 — `execute_in_remote_docker` (`18244 ms` LLM)
- **Rationale**: Inspect the public contract and panel residuals to distinguish linear operators with sparse contamination from genuinely nonlinear operators; that distinction determines which features to test on the prescribed split.
- **Alternatives Weighed**: `Fit a flexible predictor immediately` (rejected: With 52 fitting rows, discovering low-dimensional panel operators first may improve extrapolation.)

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as n
s=pickle.load(open('public_batch.pkl','rb'))[0]['study'];n.set_printoptions(precision=5,suppress=True,linewidth=130)
for k,v in s.items():
 for j,a in (v.items() if isinstance(v,dict) else [('',v)]):
  a=n.asarray(a);print(k,j,a.shape,'mean',a.mean(0),'std',a.std(0),'examples',a if a.size<=8 else a[:4])
for k in ['panel_a','panel_b','panel_c']:
 p=s[k];u=p['u'];X=n.c_[n.ones(len(u)),u];
 for j,v in p.items():
  if j=='u':continue
  b=n.linalg.lstsq(X,v,rcond=None)[0];r=n.linalg.norm(X@b-v,axis=1);print(k,j,'affine coefficients',b,'sorted residual',n.sort(r))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust IRLS reference panel alignment'
```
- **Container Exit Code**: `0` (`432 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.90382  0.90427  0.91926  0.95323  0.96433  1.01459  1.06726
  1.09109  1.47785  1.5058  10.23153 11.98473 12.27492]
panel_b v affine coefficients [[-0.07652  0.0021  -0.03878 -0.09595]
 [-0.64322 -0.12347 -0.0474  -0.58584]
 [ 0.5774  -0.24697  0.1613  -0.59495]
 [ 0.15969  0.1806  -0.83367 -0.14594]
 [-0.04209 -0.81581 -0.22615  0.23645]] sorted residual [0.00005 0.00007 0.00007 0.0001  0.0001  0.0001  0.00012 0.00012 0.00013 0.00014 0.00015 0.00015 0.00015 0.00015 0.00015 0.00017
 0.00017 0.00018 0.0002  0.00021 0.00023 0.00025 0.00025 0.00025]
panel_c v1 affine coefficients [[ 0.13756  0.02456  0.09894 -0.15267]
 [-0.6323  -0.45161  0.32574 -0.56824]
 [ 0.29581 -0.35642 -0.76833 -0.39325]
 [ 0.18988  0.76538  0.06349 -0.61616]
 [-0.68667  0.33021 -0.43064  0.2778 ]] sorted residual [0.10615 0.11813 0.11899 0.1347  0.14686 0.1513  0.16315 0.17128 0.17544 0.17548 0.19462 0.2021  0.20241 0.22372 0.22614 0.24224
 0.24412 0.25168 0.25337 0.25738 0.26454 0.28492 0.28765 0.28903]
panel_c v2 affine coefficients [[-0.17786 -0.04148 -0.083   -0.01301]
 [-0.36749  0.34338 -0.29556  0.03767]
 [-0.1894  -0.10381  0.26131 -0.32793]
 [ 0.07855 -0.12731 -0.28432 -0.27797]
 [-0.19901 -0.26377 -0.08581  0.2188 ]] sorted residual [0.04994 0.07465 0.07583 0.08598 0.1077  0.11061 0.11134 0.11547 0.11682 0.12489 0.14752 0.15352 0.15516 0.15959 0.16087 0.16561
 0.16654 0.16702 0.16781 0.17137 0.18226 0.18284 0.22045 0.25611]
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`28390 ms` LLM)
- **Rationale**: Separate panel A contamination from panel C nonlinearity, and measure trajectory rank and split errors to determine whether to model the scalar response through the trajectory or as a separate output.

```bash
cd /holdout_task && cat validate_public_schema.py && cat >/tmp/probe.py <<'PY'
import pickle,numpy as np
from itertools import combinations_with_replacement
np.set_printoptions(precision=5,suppress=True,linewidth=145)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times'];tr=s['train'];x=tr['x']; j=tr['traj']; y=tr['y']; ls=lambda X,Y:np.linalg.lstsq(X,Y,rcond=None)[0]; aff=lambda x:np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,k],axis=1) for i in range(1,d+1) for k in combinations_with_replacement(range(x.shape[1]),i)])
ops={}
for k in ['panel_a','panel_b','panel_c']:
 p=s[k];u=p['u']; ii=np.arange(len(u)); A=aff(u)
 for name,v in p.items():
  if name=='u':continue
  if k=='panel_a':
   for _ in range(4):ii=np.argsort(np.linalg.norm(A@ls(A[ii],v[ii])-v,axis=1))[:45]
  b=ls(A[ii],v[ii]); ops[k+name]=b;print(k,name,'affine residual quantiles',np.quantile(np.linalg.norm(A@b-v,axis=1),[0,.5,.9,.94,1]),'coef',b)
  Q=poly(u,2);coef=ls(Q[ii],v[ii]);print('quadratic RMS',np.sqrt(np.mean((Q[ii]@coef-v[ii])**2)), 'diag/factor',coef[5:])
print('times',t,'summary',tr['summary'],'y distributions',np.quantile(y,[0,.1,.5,.9,1]));print('ranges x',x.min(0),x.max(0),'test',s['test']['x'].min(0),s['test']['x'].max(0));print('first rows x/y/trajectory',np.c_[x,y,j][:10]);print('sv raw center diff',*[np.linalg.svd(a,compute_uv=False) for a in (j,j-j.mean(0),np.diff(j,axis=1))]);print('y corr',np.corrcoef(y,j.T)[0]);print('y from traj MAE',np.mean(abs(aff(j)@ls(aff(j)[:52],y[:52])-y)),ls(aff(j)[:52],y[:52]));
for name,z in [('raw',x)]+[(k,aff(x)@v) for k,v in ops.items()]+[('A->B',aff(aff(x)@ops['panel_av'])@ops['panel_bv'])]:
 print(name,'stats',z.mean(0),z.std(0));
 for d in [1,2,3]:
  X=poly(z,d); P=X[52:]@ls(X[:52],np.c_[y,j][:52]);print('degree',d,'mae',np.mean(abs(P-np.c_[y,j][52:]),axis=0))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe.py
```
- **Container Exit Code**: `0` (`389 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
iff [16.97331  0.70106  0.05136  0.0485   0.04385  0.03526  0.03371  0.03206] [16.81202  0.70048  0.05135  0.04827  0.04384  0.03519  0.03367  0.03125] [2.70099 0.31072 0.07846 0.0723  0.0635  0.05808 0.04193]
y corr [1.      0.10432 0.0998  0.09298 0.0839  0.07859 0.07108 0.06889 0.06565]
y from traj MAE 0.31198228634303465 [  0.03723  -4.11349  22.49646   8.12172 -17.84802   4.42907 -34.77031  13.96303  12.80231]
raw stats [-0.05518  0.00997 -0.06967  0.04301] [0.70656 0.68058 0.62302 0.65257]
degree 1 mae [0.14933 0.0473  0.09809 0.12876 0.14948 0.16483 0.17734 0.18514 0.1924 ]
degree 2 mae [0.02398 0.00658 0.01437 0.01641 0.01794 0.01917 0.02236 0.02114 0.02216]
degree 3 mae [0.01123 0.00789 0.01471 0.02177 0.02182 0.01657 0.02669 0.03498 0.03932]
panel_av stats [ 0.07801 -0.07223 -0.18388 -0.05552] [0.78702 0.6913  0.77169 0.72125]
degree 1 mae [0.14933 0.0473  0.09809 0.12876 0.14948 0.16483 0.17734 0.18514 0.1924 ]
degree 2 mae [0.02398 0.00658 0.01437 0.01641 0.01794 0.01917 0.02236 0.02114 0.02216]
degree 3 mae [0.01123 0.00789 0.01471 0.02177 0.02182 0.01657 0.02669 0.03498 0.03932]
panel_bv stats [-0.04821 -0.04122  0.0138  -0.04922] [0.56391 0.5839  0.55781 0.63685]
degree 1 mae [0.14933 0.0473  0.09809 0.12876 0.14948 0.16483 0.17734 0.18514 0.1924 ]
degree 2 mae [0.02398 0.00658 0.01437 0.01641 0.01794 0.01917 0.02236 0.02114 0.02216]
degree 3 mae [0.01123 0.00789 0.01471 0.02177 0.02182 0.01657 0.02669 0.03498 0.03932]
panel_cv1 stats [ 0.13264  0.0068   0.05036 -0.07037] [0.62579 0.65095 0.6395  0.67806]
degree 1 mae [0.14933 0.0473  0.09809 0.12876 0.14948 0.16483 0.17734 0.18514 0.1924 ]
degree 2 mae [0.02398 0.00658 0.01437 0.01641 0.01794 0.01917 0.02236 0.02114 0.02216]
degree 3 mae [0.01123 0.00789 0.01471 0.02177 0.02182 0.01657 0.02669 0.03498 0.03932]
panel_cv2 stats [-0.1735  -0.06394 -0.04797  0.01041] [0.33023 0.31215 0.32716 0.30113]
degree 1 mae [0.14933 0.0473  0.09809 0.12876 0.14948 0.16483 0.17734 0.18514 0.1924 ]
degree 2 mae [0.02398 0.00658 0.01437 0.01641 0.01794 0.01917 0.02236 0.02114 0.02216]
degree 3 mae [0.01123 0.00789 0.01471 0.02177 0.02182 0.01657 0.02669 0.03498 0.03932]
A->B stats [-0.19544  0.02239  0.11173 -0.08497] [0.60715 0.64195 0.6813  0.68472]
degree 1 mae [0.14933 0.0473  0.09809 0.12876 0.14948 0.16483 0.17734 0.18514 0.1924 ]
degree 2 mae [0.02398 0.00658 0.01437 0.01641 0.01794 0.01917 0.02236 0.02114 0.02216]
degree 3 mae [0.01123 0.00789 0.01471 0.02177 0.02182 0.01657 0.02669 0.03498 0.03932]
```

### Turn 3 — `execute_in_remote_docker` (`38284 ms` LLM)
- **Rationale**: The quadratic baseline clears the prescribed split, but the final-state cubic error rises. The radius holdout tests whether regularization or a more specific coordinate basis is needed to control extrapolation.
- **Alternatives Weighed**: `Fit y from trajectories alone` (rejected: The observed trajectories explain little of y; the tested linear readout had MAE about 0.312.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine law in the input or in any tested affine panel coordinate system is sufficient.` | Witness: `Public-split affine MAE is 0.14933 for y and 0.19240 at the final timestamp, versus 0.02398 and 0.02216 with quadratic terms.` | Blanket Diagnosis: `Affine coordinate changes preserve the same function space; changing panel coordinates alone cannot supply the missing curvature.` | Sublated Pivot: `Compare complete polynomial interactions and coordinate-specific higher powers across random and high-radius holdouts.`

```bash
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe.py | sed -n '1,/first rows/p'; cat >/tmp/cvprobe.py <<'PY'
from contextlib import redirect_stdout
from io import StringIO
with redirect_stdout(StringIO()):
 from probe import *
from itertools import permutations
Z={'x':x,'A':aff(x)@ops['panel_av'],'B':aff(x)@ops['panel_bv']}; Z['AB']=aff(Z['A'])@ops['panel_bv']; Z['BA']=aff(Z['B'])@ops['panel_av']
Y=np.c_[y,j]; rng=np.random.default_rng(37); folds=[(np.arange(52),np.arange(52,72))]; folds += [(np.setdiff1d(np.arange(72),q),q) for q in np.array_split(rng.permutation(72),4)]; q=np.argsort(np.linalg.norm(x,axis=1)); folds += [(q[:-15],q[-15:])]
def cv(F,pen):
 out=[]
 for it,iv in folds:
  sd=F[it].std(0); sd[sd<1e-12]=1; g=F/sd; a=g[it]; coef=np.linalg.lstsq(a,Y[it],rcond=None)[0] if np.max(pen)==0 else np.linalg.solve(a.T@a+np.diag(np.broadcast_to(pen,F.shape[1])),a.T@Y[it]); e=abs(g[iv]@coef-Y[iv]);out.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]);
 return np.array(out)
ps=[0,.01,.1,1,5,20]; results=[]
for key,z in Z.items():
 for d in (2,3,4,5):
  F=poly(z,d); n=F.shape[1]; w=np.array([0]+[10.**max(0,k-3) for k in range(1,d+1) for _ in combinations_with_replacement(range(4),k)]); w[:5]*=.01
  for p in ps:
   if p==0 and n>52:continue
   a=cv(F,w*p);results.append((key,d,p,a));
 for d in (3,4,5,6):
  F=np.c_[poly(z,2),*[z**k for k in range(3,d+1)]];w=np.r_[0,np.repeat(.01,4),np.ones(10),np.repeat([10.**max(0,k-3) for k in range(3,d+1)],4)];
  for p in ps:
   a=cv(F,w*p);results.append((key,-d,p,a));
for col in range(3):
 print('metric',col)
 for key,d,p,a in sorted(results,key=lambda m:m[3][:,col].max())[:7]: print(key,d,p, 'public + 4 folds + high radius',a[:,col])
print('plain poly2',cv(poly(x,2),0));
U,S,V=np.linalg.svd(j,full_matrices=False);print('temporal vectors',V[:2]);
for key,z in Z.items():
 for d in (1,2,3):
  f=poly(z,d); b=ls(f,Y); print('in-sample',key,d,np.sqrt(np.mean((f@b-Y)**2,axis=0))[[0,1,8]]) if key=='x' else None
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/cvprobe.py
```
- **Container Exit Code**: `0` (`665 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ows x/y/trajectory [[ 0.94223  0.72203 -0.76306 -0.32801  0.19807  0.46676  0.99757  1.35273  1.62374  1.83086  2.01966  2.16914  2.28873]
metric 0
x 5 0.1 public + 4 folds + high radius [0.01648 0.01448 0.02439 0.01232 0.02291 0.0255 ]
x 4 0.1 public + 4 folds + high radius [0.017   0.01351 0.02489 0.01178 0.02303 0.02627]
A 4 0.1 public + 4 folds + high radius [0.0164  0.01074 0.02039 0.01296 0.02345 0.0263 ]
AB 5 0.1 public + 4 folds + high radius [0.0141  0.01295 0.01894 0.01474 0.0258  0.02664]
AB 4 0.1 public + 4 folds + high radius [0.01369 0.01242 0.01832 0.01395 0.02705 0.02364]
B -4 0.1 public + 4 folds + high radius [0.01968 0.01948 0.0183  0.02127 0.02253 0.02741]
BA -4 0.1 public + 4 folds + high radius [0.01979 0.01938 0.01923 0.02223 0.02251 0.02754]
metric 1
BA -3 1 public + 4 folds + high radius [0.01742 0.01772 0.01781 0.01455 0.01536 0.01857]
BA -4 1 public + 4 folds + high radius [0.01735 0.01791 0.01818 0.01449 0.01616 0.01864]
BA -6 1 public + 4 folds + high radius [0.01732 0.01792 0.01818 0.0145  0.01614 0.01867]
BA -5 1 public + 4 folds + high radius [0.01733 0.01792 0.01817 0.01451 0.01613 0.01867]
B -3 1 public + 4 folds + high radius [0.01714 0.01736 0.01848 0.01429 0.01542 0.01889]
B -4 1 public + 4 folds + high radius [0.01705 0.01762 0.01923 0.01428 0.0164  0.01907]
B -5 1 public + 4 folds + high radius [0.01704 0.01765 0.01924 0.01434 0.01638 0.01917]
metric 2
AB -3 1 public + 4 folds + high radius [0.02326 0.02091 0.02372 0.01931 0.02045 0.02552]
AB -4 1 public + 4 folds + high radius [0.02321 0.02101 0.0247  0.01907 0.02241 0.0256 ]
AB -6 1 public + 4 folds + high radius [0.02321 0.02106 0.02481 0.01914 0.02241 0.0257 ]
AB -5 1 public + 4 folds + high radius [0.02321 0.02106 0.02477 0.01914 0.02236 0.02572]
BA -3 1 public + 4 folds + high radius [0.02322 0.02129 0.02159 0.02005 0.02062 0.0259 ]
BA -4 1 public + 4 folds + high radius [0.02311 0.02135 0.022   0.01977 0.02216 0.02607]
BA -6 1 public + 4 folds + high radius [0.02307 0.02136 0.02196 0.01978 0.02213 0.02609]
plain poly2 [[0.02398 0.01752 0.02216]
 [0.02893 0.01815 0.02248]
 [0.03529 0.01265 0.01579]
 [0.03465 0.01327 0.01745]
 [0.02746 0.01804 0.02241]
 [0.03873 0.02642 0.03442]]
temporal vectors [[-0.10165 -0.21266 -0.28802 -0.34272 -0.38413 -0.41793 -0.44662 -0.47153]
 [-0.32522 -0.48282 -0.45339 -0.30492 -0.11144  0.10303  0.30654  0.49556]]
in-sample x 1 [0.17341 0.06256 0.28304]
in-sample x 2 [0.02886 0.00643 0.01989]
in-sample x 3 [0.00672 0.00404 0.01343]
```

### Turn 4 — `execute_in_remote_docker` (`42005 ms` LLM)
- **Rationale**: Held-out panel errors will distinguish usable nonlinear operators from in-sample fits. The temporal-mode and coefficient tests will determine whether a structured response model is supported or whether to retain the polynomial baseline.
- **Alternatives Weighed**: `Finalize the regularized polynomial` (rejected: It passes the prescribed split, but the two dominant trajectory modes and the measured nonlinearity in panel C justify testing more specific laws before freezing it.)

```bash
cat >/tmp/structure.py <<'PY'
from contextlib import redirect_stdout
from io import StringIO
with redirect_stdout(StringIO()):
 from cvprobe import *
from scipy.optimize import least_squares
fns={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'softsign':lambda q:q/np.sqrt(1+q*q),'exp':lambda q:np.exp(np.clip(q,-15,15))}
p=s['panel_c']; u=p['u']; h=aff(u); folds_c=np.array_split(np.random.default_rng(34).permutation(len(u)),4)
def bases(h,v,it,key,fn):
 a=h[it]; b=v[it]; b0=ls(a,b); n=len(a); q=h[:,1:]; Q=np.c_[h,q*q];
 if key=='diag+sin':
  def pred(w):return Q@w[:9]+w[9]*np.sin(h@w[10:]);
  w=np.r_[ls(Q[it],b),.15,[0.,.5,.5,.5,.5]]
 elif key=='separable':
  def pred(w):return Q@w[:9]+(np.sin(q*w[9:13])*w[13:17]).sum(1)
  w=np.r_[ls(Q[it],b),[1.]*4,[.15]*4]
 elif key=='aff+link':
  def pred(w):return h@w[:5]+w[5]*fn(h@w[6:]);
  w=np.r_[b0*.2,1,b0*.8]
 else:
  def pred(w):return w[0]+w[1]*fn(h@w[2:]);
  w=np.r_[0.,1.,b0]
 fit=least_squares(lambda w:pred(w)[it]-b,w,max_nfev=220);return fit.x,pred(fit.x),float(np.sqrt(np.mean(fit.fun**2)))
for k in ('v1','v2'):
 v=p[k]; print(k)
 for key,fn in list(fns.items())+[(mode+'_'+name,g) for mode in ('aff+link',) for name,g in fns.items() if name!='exp']+[(g,None) for g in ('diag+sin','separable')]:
  mode=key.split('_')[0]; fits=[bases(h,v[:,i],np.arange(24),mode,fn) for i in range(4)]; errs=[]
  for test in folds_c:
   it=np.setdiff1d(np.arange(24),test); P=np.column_stack([bases(h,v[:,i],it,mode,fn)[1] for i in range(4)]);errs.append(np.mean((P[test]-v[test])**2)**.5)
  print(key,'fit',np.round([f[2] for f in fits],6),'cv',np.round(errs,5));
  if max(f[2] for f in fits)<.006:print('params',np.round([f[0] for f in fits],4))
U,sv,V=np.linalg.svd(j,full_matrices=False); Y=np.c_[y,j@V[:2].T];
for name,z in Z.items():
 F=poly(z,3); b=ls(F,Y); print(name,'linear',np.round(b[1:5].T,4),'quadratic',np.round(b[5:15].T,4),'cubic',np.round(b[15:].T,4));
print('trajectory time fits');
for forms in [('log','t'),('exp','t'),('log','exp'),('exp','exp'),('log','log')]:
 def htime(p): return np.column_stack([{'log':lambda:np.log1p(tt*t),'exp':lambda:-np.expm1(-tt*t),'t':lambda:t}[f]() for tt,f in zip(p,forms)])
 def res(p):
  H=htime(p);return (H@ls(H,j.T)-j.T).ravel()
 opt=least_squares(res,[1.,2.],bounds=([.01,.01],[25,25]));print(forms,opt.x,np.sqrt(np.mean(opt.fun**2)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/structure.py
```
- **Container Exit Code**: `0` (`12371 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
7  -0.0072  0.022   0.0023  0.0362 -0.0336 -0.0206 -0.0674 -0.003   0.0318  0.004   0.0337 -0.0414
  -0.0088  0.0359 -0.078 ]
 [ 0.0539  0.0001  0.0258  0.0279 -0.0064  0.0419  0.0196  0.0203 -0.0837 -0.0363 -0.0534  0.0665  0.0328 -0.0907 -0.013   0.0151  0.0379
  -0.0006  0.0544 -0.0797]
 [-0.0006  0.003   0.0136 -0.003  -0.0014  0.0007 -0.0005 -0.0038  0.0014 -0.0121 -0.0112 -0.018   0.0129  0.0065 -0.0057  0.0038 -0.004
  -0.0009  0.0085 -0.0126]]
AB linear [[ 0.1641  0.1401  0.3626  0.4926]
 [-0.2319  0.9621 -1.8831  1.7521]
 [-0.0883  0.0727 -0.0107 -0.0237]] quadratic [[ 0.0184 -0.1494 -0.1026  0.0617  0.0319 -0.0166 -0.2897 -0.0123 -0.1946  0.0147]
 [-0.0636  0.4167  0.3616 -0.3143  0.0152 -0.5095  1.055  -0.1744  0.6166 -0.1704]
 [-0.0042  0.0122  0.0003  0.0019 -0.0047  0.0389  0.0055  0.0164  0.0149 -0.0091]] cubic [[-0.0204 -0.014  -0.0187 -0.0132  0.0288 -0.0011  0.018  -0.0026  0.0254 -0.019  -0.0168 -0.0446 -0.0024  0.0164 -0.0005  0.0233 -0.0223
  -0.0039  0.0299 -0.0631]
 [ 0.0434 -0.0014  0.0242  0.0167 -0.0013  0.0221  0.0097  0.0191 -0.0624 -0.0204 -0.0384  0.0395  0.0218 -0.0521 -0.0018  0.0101  0.0205
  -0.0079  0.046  -0.0651]
 [-0.0001  0.0027  0.0095 -0.0019 -0.0014  0.0011 -0.0017 -0.0021  0.0003 -0.0084 -0.0083 -0.0128  0.0099  0.0036 -0.0044  0.0033 -0.0022
  -0.0006  0.0066 -0.01  ]]
BA linear [[ 0.1177  0.1626  0.3661  0.3935]
 [ 0.0804  0.8885 -2.0162  1.9589]
 [-0.0818  0.0802  0.0034 -0.025 ]] quadratic [[ 0.0292 -0.1472 -0.0912  0.0725  0.0127 -0.0269 -0.3094 -0.0241 -0.1909  0.0183]
 [-0.1009  0.4965  0.3496 -0.3298  0.0071 -0.5425  1.1031 -0.1557  0.6032 -0.1698]
 [-0.0014  0.0097 -0.0051  0.0004 -0.0125  0.0377  0.011   0.0157  0.0136 -0.0062]] cubic [[-0.0186 -0.0128 -0.0149 -0.0162  0.0322 -0.0052  0.0176  0.0013  0.0238 -0.0247 -0.0196 -0.0528 -0.0026  0.0204  0.0029  0.0272 -0.0218
  -0.0052  0.0238 -0.0578]
 [ 0.039   0.      0.0168  0.0203 -0.0056  0.0299  0.0157  0.0119 -0.0549 -0.0267 -0.0509  0.0521  0.0287 -0.0582 -0.0093  0.0121  0.0199
  -0.0003  0.0359 -0.059 ]
 [-0.0004  0.0024  0.0088 -0.0022 -0.0012  0.0005 -0.0004 -0.0022  0.0009 -0.0089 -0.0106 -0.0141  0.0113  0.0042 -0.0041  0.0031 -0.0021
  -0.0005  0.0056 -0.0093]]
trajectory time fits
('log', 't') [1.72105 2.     ] 0.0059431236623538105
('exp', 't') [1.33433 2.     ] 0.005456637365639813
('log', 'exp') [0.76273 1.57375] 0.004266508597966043
('exp', 'exp') [0.24696 1.56974] 0.004268185547474635
('log', 'log') [0.81425 0.81415] 0.004525149033567174
```

### Turn 5 — `execute_in_remote_docker` (`41579 ms` LLM)
- **Rationale**: Save and validate the working nonlinear baseline first. Then use held-out panel rows to determine whether a shared separable operator is accurate enough to test as a forward-model feature.
- **Alternatives Weighed**: `Use an independently fitted nonlinear ridge per panel C output` (rejected: Those fits had large held-out errors; sharing coordinate-wise nonlinear features across outputs tests a simpler explanation.)

```bash
cat >/holdout_task/batch_solver.py <<'PY'
from itertools import combinations_with_replacement
import numpy as np


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _poly(x, degree):
    return np.column_stack([np.ones(len(x))] + [np.prod(x[:, ix], axis=1)
        for k in range(1, degree + 1)
        for ix in combinations_with_replacement(range(x.shape[1]), k)])


def _ridge(f, q, target, penalty):
    scale = f.std(axis=0)
    scale[scale == 0] = 1
    a = f / scale
    return (q / scale) @ np.linalg.solve(a.T @ a + np.diag(penalty), a.T @ target)


def solve_batch(study: dict) -> dict:
    tr = study['train']
    x, q = tr['x'], study['test']['x']
    degree = x.shape[1]
    penalty = np.array([0.] + [.001]*degree + [.1]*10 + [.1]*20 + [1.]*35 + [10.]*56)
    y = _ridge(_poly(x, 5), _poly(q, 5), tr['y'], penalty)
    a, b = study['panel_a'], study['panel_b']
    ua = _aff(a['u'])
    keep = np.arange(len(ua))
    for _ in range(4):
        fit = np.linalg.lstsq(ua[keep], a['v'][keep], rcond=None)[0]
        keep = np.argsort(np.linalg.norm(ua @ fit - a['v'], axis=1))[:45]
    ca = np.linalg.lstsq(ua[keep], a['v'][keep], rcond=None)[0]
    cb = np.linalg.lstsq(_aff(b['u']), b['v'], rcond=None)[0]
    def features(z):
        z = _aff(_aff(z) @ cb) @ ca
        return np.column_stack((_poly(z, 2), z**3))
    state = _ridge(features(x), features(q), tr['traj'], np.r_[0., [.01]*4, [1.]*14])
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py
cat >/tmp/panelc.py <<'PY'
from contextlib import redirect_stdout
from io import StringIO
with redirect_stdout(StringIO()):
 from cvprobe import *
from scipy.optimize import least_squares
p=s['panel_c']; u=p['u']; perm=np.random.default_rng(19).permutation(len(u)); splits=[(np.arange(len(u)),np.arange(len(u)))]+[(np.setdiff1d(np.arange(len(u)),v),v) for v in np.array_split(perm,4)]
P=poly(u,2); links={'sin':np.sin,'tanh':np.tanh,'cos':np.cos,'atan':np.arctan,'sqrat':lambda q:q*q/(1+q*q),'sincos':lambda q:np.c_[np.sin(q),np.cos(q)]}
for k in ('v1','v2'):
 res=[]; target=p[k]
 for base in ('1','aff','diag'):
  for name,g in links.items():
   for npar in (1,4):
    if g==links['sincos'] and base=='diag': continue
    def features(z,f):
     a={'1':np.ones((len(z),1)),'aff':aff(z),'diag':np.c_[aff(z),z*z]}[base]
     return np.c_[a,g(z*f)]
    errors=[]; params=[]
    for it,iv in splits:
     def err(f):
      a=features(u[it],f); return (a@ls(a,target[it])-target[it]).ravel()
     opt=min([least_squares(err,np.full(npar,h),bounds=(.12,8),max_nfev=100) for h in (.65,1.8,3.,5.)],key=lambda o:o.cost)
     a=features(u,opt.x);errors.append(np.mean((a[iv]@ls(a[it],target[it])-target[iv])**2)**.5);params.append(opt.x)
    res.append((max(errors[1:]),base,name,npar,np.round(errors,6),np.round(params[0],5)))
 print(k)
 for r in sorted(res)[:12]:print(r)
print('panel inputs ranges',*[np.quantile(s[k]['u'],[0,.5,1],axis=0) for k in ('panel_a','panel_b','panel_c')]);print('panel C v1 / v2 covariance',np.corrcoef(np.c_[p['v1'],p['v2']].T)[:4,4:])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/panelc.py
```
- **Container Exit Code**: `0` (`8609 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
4, 2.1227 ]))
(0.024102703970111612, '1', 'tanh', 4, array([0.01186, 0.0241 , 0.01742, 0.01385, 0.01796]), array([1.62154, 1.61544, 1.70078, 1.58318]))
v2
(0.0004263592686080448, 'aff', 'sqrat', 1, array([0.00019, 0.00043, 0.00034, 0.00034, 0.00041]), array([0.40456]))
(0.0006688331595351041, 'diag', 'sqrat', 1, array([0.00008, 0.00067, 0.00054, 0.00027, 0.00027]), array([0.53908]))
(0.0007295850569500741, 'diag', 'cos', 1, array([0.00008, 0.00073, 0.00055, 0.00028, 0.00029]), array([2.52265]))
(0.0007741472269477821, 'diag', 'sqrat', 4, array([0.00008, 0.00077, 0.00061, 0.00029, 0.00026]), array([0.52186, 0.53055, 0.54519, 0.57614]))
(0.0008023468525162915, 'aff', 'cos', 1, array([0.00035, 0.0008 , 0.0006 , 0.00062, 0.00077]), array([1.2749]))
(0.0010715725188209422, 'diag', 'cos', 4, array([0.00008, 0.00107, 0.00064, 0.00029, 0.00029]), array([2.46155, 2.48368, 2.58005, 2.63636]))
(0.001107709056031985, 'aff', 'sqrat', 4, array([0.00019, 0.00111, 0.00038, 0.00034, 0.00051]), array([0.40411, 0.40588, 0.40939, 0.40067]))
(0.0013303723048935566, 'aff', 'sincos', 1, array([0.00029, 0.00133, 0.00105, 0.00067, 0.00103]), array([1.28452]))
(0.0022479565338370915, 'aff', 'cos', 4, array([0.00033, 0.00225, 0.00081, 0.00064, 0.0011 ]), array([1.27358, 1.27875, 1.3032 , 1.25696]))
(0.0024391972761351717, 'aff', 'sincos', 4, array([0.00026, 0.00244, 0.00137, 0.00079, 0.00123]), array([1.2165 , 1.29803, 1.32555, 1.27192]))
(0.009453220198340852, '1', 'sincos', 4, array([0.00372, 0.00945, 0.00758, 0.00551, 0.00781]), array([0.60733, 0.12   , 0.12   , 0.12   ]))
(0.011290937539996769, '1', 'sincos', 1, array([0.0043 , 0.01129, 0.01005, 0.00615, 0.01014]), array([0.12]))
panel inputs ranges [[-1.26658 -1.2418  -1.22192 -1.08616]
 [-0.07875 -0.02695  0.16287  0.49489]
 [ 1.27725  1.19407  1.20701  1.26726]] [[-0.84255 -1.11619 -0.75808 -1.12798]
 [ 0.13471  0.22833  0.147    0.12457]
 [ 1.15963  1.19058  1.18364  1.20476]] [[-0.7081  -1.23318 -1.15805 -1.23924]
 [ 0.36561  0.13315 -0.01799 -0.36653]
 [ 1.22381  1.01427  0.91155  1.17074]]
panel C v1 / v2 covariance [[ 0.58904  0.17845  0.26323 -0.72425]
 [ 0.55822 -0.53925 -0.53397  0.12807]
 [ 0.50729  0.62487 -0.65386  0.0326 ]
 [ 0.13468 -0.53154  0.35506  0.77899]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ba9f753d82585b48 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ba9f753d (directory).
```

### Turn 6 — `execute_in_remote_docker` (`35447 ms` LLM)
- **Rationale**: Test whether rational and saturating panel features improve the scalar and trajectory outputs separately; retain the verified baseline if they do not improve the worst slices.
- **Alternatives Weighed**: `Include panel C in the solver now` (rejected: Accurate reconstruction of an auxiliary operator does not establish that it improves response prediction.)

```bash
cat >/tmp/forward.py <<'PY'
from contextlib import redirect_stdout
from io import StringIO
with redirect_stdout(StringIO()):
 from cvprobe import *
from scipy.optimize import minimize_scalar, least_squares
p=s['panel_c']; u=p['u'];
def fit2(z,k):return np.c_[aff(z),z*z/(1+k*z*z)]
f=lambda k: np.mean((fit2(u,k)@ls(fit2(u,k),p['v2'])-p['v2'])**2)
k=minimize_scalar(f,bounds=(.01,2),method='bounded').x
print('v2 rational',k,f(k)**.5)
B=ops['panel_bv']; A=ops['panel_av'].copy(); A[1:]=np.diag(np.diag(A[1:])); V2=ls(fit2(u,k),p['v2']);
for form in ('atan','atan+id','atan+sin','atan+qrat','atan+atan','soft','odd_rat','atan+odd_rat'):
 def feat(z,w):
  d=np.arctan(w[0]*z); out={'atan':lambda:d,'atan+id':lambda:np.c_[d,z],'atan+sin':lambda:np.c_[d,np.sin(w[1]*z)],'atan+qrat':lambda:np.c_[d,z*z/(1+w[1]*z*z)],'atan+atan':lambda:np.c_[d,np.arctan(w[1]*z)],'soft':lambda:z/np.sqrt(1+w[0]*z*z),'odd_rat':lambda:z/(1+w[0]*z*z),'atan+odd_rat':lambda:np.c_[d,z/(1+w[1]*z*z)]}[form]();return aff(out)
 def r(w):
  h=feat(u,w); return (h@ls(h,p['v1'])-p['v1']).ravel()
 opts=[least_squares(r,start,bounds=(.02,10.),max_nfev=350) for start in ([2.2,.7],[2.2,2],[2.2,4.5])]; opt=min(opts,key=lambda a:a.cost)
 print(form,opt.x,np.sqrt(np.mean(opt.fun**2)))
 if form=='atan+id':
  a1=opt.x; H1=lambda z:aff(np.c_[np.arctan(a1[0]*z),z]); V1=ls(H1(u),p['v1'])
fun={'A':lambda z:aff(z)@A,'a':lambda z:(z-A[0])/np.diag(A[1:]),'B':lambda z:aff(z)@B,'b':lambda z:(z-B[0])@np.linalg.inv(B[1:]),'C':lambda z:H1(z)@V1,'D':lambda z:fit2(z,k)@V2}
from itertools import product
res=[]
for pre in ('','A','a','B','b','AB','aB','BA','Ba','Ab','ab','bA','ba'):
 z=x
 for v in pre:z=fun[v](z)
 c,d=fun['C'](z),fun['D'](z)
 feats={'Q':poly(z,2),'Q+CD':np.c_[poly(z,2),c,d],'Q+zC':np.c_[poly(z,2),c,z*c],'Q+zD':np.c_[poly(z,2),d,z*d],'CD':aff(np.c_[c,d]),'CDQ':poly(np.c_[c,d],2),'Dq':poly(d,2),'Cq':poly(c,2),'Q+rat':np.c_[poly(z,2),z**3/(1+k*z*z),z*z/(1+k*z*z)],'Q+cross':np.c_[poly(z,2), c,d,c*d,z*c,z*d]}
 for name,F in feats.items():
  for p1 in (0.,.02,.2,1.):
   if p1==0 and F.shape[1]>=35:continue
   w=np.repeat(p1,F.shape[1]);w[0]=0;w[1:5]*=.01
   a=cv(F,w);res.append((pre,name,p1,a))
for i in range(3):
 print('metric',i)
 for pre,name,p1,a in sorted(res,key=lambda q:q[3][:,i].max())[:8]: print(pre,name,p1,np.round(a[:,i],5))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/forward.py
```
- **Container Exit Code**: `0` (`975 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
v2 rational 0.16366373066465403 0.00019395471647961178
atan [2.10657 2.     ] 0.0043825289447411726
atan+id [2.3307 4.5   ] 0.0026622659418016064
atan+sin [3.07195 1.11817] 0.001390851758407771
atan+qrat [2.11909 0.02   ] 0.003667228970272084
atan+atan [9.74077 1.84581] 0.000645136790309366
soft [2.27699 4.5    ] 0.007800493152247833
odd_rat [0.67028 2.     ] 0.017697511030311582
atan+odd_rat [4.24822 7.49321] 0.00047607958524339034
metric 0
aB Q+rat 0.0 [0.02181 0.02078 0.01901 0.02145 0.02227 0.02674]
AB Q+rat 0.02 [0.0204  0.01957 0.01805 0.02186 0.02278 0.02677]
B Q+rat 0.02 [0.01973 0.01921 0.0188  0.02067 0.02226 0.02769]
BA Q+rat 0.02 [0.01961 0.01903 0.01868 0.02107 0.02223 0.02785]
Ba Q+rat 0.02 [0.02045 0.01928 0.0192  0.02046 0.02235 0.02791]
BA Q+rat 0.0 [0.02119 0.02048 0.0189  0.02232 0.02233 0.0286 ]
B Q+rat 0.0 [0.0213  0.02059 0.01897 0.02258 0.02234 0.02869]
Ba Q+rat 0.0 [0.02139 0.0207  0.01896 0.02281 0.02234 0.02888]
metric 1
BA Q+rat 1.0 [0.01788 0.01684 0.01782 0.01463 0.01489 0.01754]
Ba Q+rat 1.0 [0.0176  0.0167  0.0178  0.0141  0.01494 0.01789]
B Q+rat 1.0 [0.0176  0.01655 0.01786 0.01416 0.0149  0.018  ]
AB Q+rat 1.0 [0.01785 0.01645 0.01835 0.01414 0.01448 0.01741]
aB Q+rat 1.0 [0.01755 0.01702 0.01702 0.01456 0.01542 0.01838]
AB Q+zC 0.2 [0.01574 0.01482 0.01898 0.01375 0.01584 0.01845]
Ba Q+zC 0.2 [0.01527 0.01608 0.01882 0.01318 0.01603 0.01917]
AB Q+CD 1.0 [0.01919 0.01696 0.0132  0.01327 0.01513 0.01968]
metric 2
AB Q+rat 1.0 [0.02396 0.0198  0.02238 0.01951 0.01941 0.02381]
BA Q+rat 1.0 [0.02387 0.02041 0.02165 0.0202  0.01986 0.0243 ]
A Q+zC 1.0 [0.01913 0.02265 0.01437 0.02036 0.01869 0.02453]
Ba Q+rat 1.0 [0.02344 0.01992 0.02203 0.01958 0.01996 0.02456]
B Q+rat 1.0 [0.02348 0.01987 0.02184 0.01975 0.01983 0.02494]
AB Q+zC 0.2 [0.01928 0.01887 0.024   0.01748 0.01967 0.02495]
Ba Q+zC 0.2 [0.01803 0.02097 0.02318 0.01696 0.02008 0.02511]
aB Q+rat 1.0 [0.02312 0.02046 0.02123 0.02023 0.0205  0.02532]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ba9f753d82585b48 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ba9f753d (directory).
```

### Turn 7 — `execute_in_remote_docker` (`32854 ms` LLM)
- **Rationale**: Test whether polynomial residuals have learnable directional structure without using outer validation labels to select the direction or smoothing parameters. Keep the existing solver where the correction does not improve held-out error.
- **Alternatives Weighed**: `Add panel C features to both outputs` (rejected: The tested compositions did not improve the worst scalar slice, although rational features modestly improved the terminal-state slice.)

```bash
cat >/tmp/active_cv.py <<'PY'
from contextlib import redirect_stdout
from io import StringIO
with redirect_stdout(StringIO()):
 from cvprobe import *
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement
Q=poly(x,2)

def active(it,T,alpha):
 b=ls(Q[it],T)
 H=np.zeros((4,4,T.shape[1]))
 for (i,j),v in zip(combinations_with_replacement(range(4),2),b[5:]):
  H[i,j]+=v; H[j,i]+=v
 G=b[1:5][None,:,:]+np.einsum('ni,ijk->njk',x[it],H)
 G=G.reshape(-1,4) if T.shape[1]==1 else G.transpose(0,2,1).reshape(-1,4)
 M=G.T@G/len(G); return (1-alpha)*np.eye(4)+alpha*4*M/np.trace(M)

def run(it,iv,T,conf):
 alpha,l,noise,kind=conf
 M=active(it,T,alpha); eig,V=np.linalg.eigh(M); z=x@(V*np.sqrt(np.maximum(eig,0)))
 D=cdist(z,z,'sqeuclidean'); r=np.sqrt(D)/l
 K=np.exp(-r*r/2) if kind=='rbf' else (1+np.sqrt(5)*r+5*r*r/3)*np.exp(-np.sqrt(5)*r)
 A=Q[it]; K0=K[np.ix_(it,it)]+noise*np.eye(len(it)); Ka=K[np.ix_(iv,it)]
 Wi=np.linalg.solve(K0,np.c_[A,T]); p=ls(A.T@Wi[:,:15],A.T@Wi[:,15:]);return Q[iv]@p+Ka@(Wi[:,15:]-Wi[:,:15]@p)

from itertools import product
configs=list(product((0,.6,.95),(0.6,1.2,2.4),(.00001,.001,.03),('rbf','m52')))
for label,T in [('scalar',Y[:,:1]),('state',Y[:,1:])]:
 scores=[]
 for c in configs:
  allerr=[abs(run(it,iv,T[it],c)-T[iv]) for it,iv in folds]
  a=np.array([[e.mean(),e[:,-1].mean()] for e in allerr]); scores.append((c,a))
 print(label,'strongest fixed candidates')
 for c,a in sorted(scores,key=lambda m:m[1][:,1].max())[:5]:print(c,np.round(a.T,5))
 nested=[]
 for it,iv in folds:
  inner=[(np.setdiff1d(it,v),v) for v in np.array_split(np.random.default_rng(183).permutation(it),3)]
  w=min(configs,key=lambda c:np.mean([np.mean(abs(run(a,b,T[a],c)-T[b])) for a,b in inner]));e=abs(run(it,iv,T[it],w)-T[iv]);nested.append([e.mean(),e[:,-1].mean()]);
 print('nested',np.round(np.array(nested).T,5))
print('trajectory singular values',np.linalg.svd(j,compute_uv=False))
print('scalar target summary mean/std',y.mean(),y.std(),'state final',j[:,-1].mean(),j[:,-1].std())
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/active_cv.py
```
- **Container Exit Code**: `0` (`1347 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
scalar strongest fixed candidates
(0, 2.4, 0.001, 'm52') [[0.00885 0.00843 0.01811 0.01265 0.01112 0.01812]
 [0.00885 0.00843 0.01811 0.01265 0.01112 0.01812]]
(0, 2.4, 1e-05, 'm52') [[0.00819 0.00778 0.01788 0.0125  0.0117  0.01907]
 [0.00819 0.00778 0.01788 0.0125  0.0117  0.01907]]
(0.6, 2.4, 1e-05, 'm52') [[0.01239 0.01209 0.01576 0.0118  0.01642 0.01912]
 [0.01239 0.01209 0.01576 0.0118  0.01642 0.01912]]
(0.6, 2.4, 0.001, 'm52') [[0.01369 0.01296 0.01708 0.01346 0.01603 0.01932]
 [0.01369 0.01296 0.01708 0.01346 0.01603 0.01932]]
(0, 1.2, 1e-05, 'm52') [[0.01012 0.01084 0.01989 0.01416 0.01113 0.0192 ]
 [0.01012 0.01084 0.01989 0.01416 0.01113 0.0192 ]]
nested [[0.01032 0.00709 0.01708 0.0116  0.01335 0.02688]
 [0.01032 0.00709 0.01708 0.0116  0.01335 0.02688]]
state strongest fixed candidates
(0.6, 2.4, 1e-05, 'rbf') [[0.01198 0.01419 0.01416 0.01332 0.01808 0.02141]
 [0.01352 0.01054 0.01403 0.01608 0.02562 0.02384]]
(0.6, 1.2, 0.001, 'rbf') [[0.0115  0.01225 0.01185 0.01366 0.01651 0.01978]
 [0.01372 0.01324 0.01141 0.01752 0.02082 0.02646]]
(0.6, 0.6, 1e-05, 'rbf') [[0.01323 0.01339 0.00948 0.01207 0.01528 0.02081]
 [0.01605 0.01685 0.00986 0.01644 0.01826 0.02883]]
(0.6, 0.6, 0.001, 'rbf') [[0.01323 0.01337 0.00946 0.01206 0.01526 0.02078]
 [0.01603 0.01676 0.00987 0.01638 0.01823 0.02885]]
(0.6, 0.6, 0.03, 'rbf') [[0.01354 0.01311 0.00906 0.01184 0.01501 0.02064]
 [0.01651 0.01576 0.01023 0.01575 0.01783 0.02911]]
nested [[0.01372 0.01244 0.0118  0.01088 0.01523 0.02106]
 [0.01854 0.01546 0.01538 0.01527 0.01818 0.03017]]
trajectory singular values [16.97331  0.70106  0.05136  0.0485   0.04385  0.03526  0.03371  0.03206]
scalar target summary mean/std 0.07129632211332433 0.4613732553464125 state final 0.12833688193941942 0.9353335114844403

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ba9f753d82585b48 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ba9f753d (directory).
```

### Turn 8 — `execute_in_remote_docker` (`46252 ms` LLM)
- **Rationale**: Compare the exact implementations against the saved solver on the prescribed split and directional holdouts. Use the scalar correction only if its gain survives extrapolation, and change the trajectory model only if it improves the terminal and worst-horizon errors.
- **Alternatives Weighed**: `Use the supervised active-direction kernel for both outputs` (rejected: Nested tuning did not improve the worst trajectory slice; the isotropic scalar correction showed a clearer benefit.)

```bash
cat >/tmp/solver_challenger.py <<'PY'
import numpy as np


def scalar_predict(x, q, y, poly):
    a, b = poly(x, 2), poly(q, 2)
    def kernel(u, v):
        r = (np.sqrt(5.) / 2.4) * np.linalg.norm(u[:, None, :] - v[None, :, :], axis=2)
        return (1. + r + r*r/3.) * np.exp(-r)
    k = kernel(x, x) + .001 * np.eye(len(x))
    w = np.linalg.solve(k, np.column_stack((a, y)))
    coef = np.linalg.solve(a.T @ w[:, :-1], a.T @ w[:, -1])
    return b @ coef + kernel(q, x) @ (w[:, -1] - w[:, :-1] @ coef)


def state_predict(study, poly, ridge, aff):
    a, b = study['panel_a'], study['panel_b']
    ua = aff(a['u'])
    keep = np.arange(len(ua))
    for _ in range(4):
        fit = np.linalg.lstsq(ua[keep], a['v'][keep], rcond=None)[0]
        keep = np.argsort(np.linalg.norm(ua @ fit - a['v'], axis=1))[:45]
    ca = np.linalg.lstsq(ua[keep], a['v'][keep], rcond=None)[0]
    cb = np.linalg.lstsq(aff(b['u']), b['v'], rcond=None)[0]
    def features(x):
        z = aff(aff(x) @ ca) @ cb
        d = 1. + .16*z*z
        return np.column_stack((poly(z, 2), z**3/d, z*z/d))
    tr = study['train']
    return ridge(features(tr['x']), features(study['test']['x']), tr['traj'], np.r_[0., [.01]*4, [1.]*18])
PY
cat >/tmp/solver_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np
from solver_challenger import scalar_predict, state_predict

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', root+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open(root+'public_batch.pkl', 'rb'))[0]['study']
t = s['train']; x, y, j = (t[k] for k in ('x','y','traj'))
n = len(x); all_rows = np.arange(n)
radius = np.linalg.norm(x, axis=1)
folds = [('public', np.arange(52,n))]
folds += [(f'heldout_{i}', v) for i,v in enumerate(np.array_split(np.random.default_rng(37).permutation(n),4))]
folds += [('radius', np.argsort(radius)[-15:])]
folds += [(f'axis_{i}_{sign:+d}', np.argsort(sign*x[:, i])[-12:]) for i in range(x.shape[1]) for sign in (-1,1)]
rows = []
for name,v in folds:
    it = np.arange(52) if name == 'public' else np.setdiff1d(all_rows, v)
    st = dict(s, train=dict(t, x=x[it], y=y[it], traj=j[it]), test={'x':x[v]})
    o = m.solve_batch(st)
    yp = scalar_predict(x[it], x[v], y[it], m._poly)
    sp = state_predict(st, m._poly, m._ridge, m._aff)
    old = o['state_pred']
    _, _, vt = np.linalg.svd(j[it], full_matrices=False)
    blend = (old+sp)/2
    sm = [np.mean(abs(z-j[v]), axis=0) for z in (old, sp, blend, (blend@vt[:2].T)@vt[:2])]
    row = [np.mean(abs(z-y[v])) for z in (o['y_hat'], yp, .2*o['y_hat']+.8*yp)] + [b for z in sm for b in (z.mean(), z[-1], z.max())]
    rows.append(row)
    print(name, np.round(row,5))
print('columns: scalar baseline/Matern/blend; state baseline/rational/blend/projected blend, each avg/terminal/worst horizon')
print('worst original slices', np.round(np.max(rows[:6], axis=0),5))
print('worst directional slices', np.round(np.max(rows[6:], axis=0),5))
PY
OPENBLAS_NUM_THREADS=1 python3 -I -c 'import sys; sys.path.insert(0,"/tmp"); import solver_audit'
```
- **Container Exit Code**: `0` (`397 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
885 0.00974 0.01742 0.02322 0.02322 0.01784 0.02395 0.02395
 0.01761 0.02359 0.02359 0.01755 0.02357 0.02357]
heldout_0 [0.01448 0.00843 0.00896 0.01772 0.02129 0.02129 0.01646 0.0198  0.0198
 0.01704 0.02037 0.02037 0.01702 0.02015 0.0212 ]
heldout_1 [0.02439 0.01811 0.01902 0.01781 0.02159 0.02452 0.01838 0.02242 0.02472
 0.01808 0.02201 0.02453 0.01818 0.02365 0.02443]
heldout_2 [0.01232 0.01265 0.01141 0.01455 0.02005 0.02005 0.01414 0.01951 0.01951
 0.01432 0.01978 0.01978 0.01403 0.02006 0.02006]
heldout_3 [0.02291 0.01112 0.01248 0.01536 0.02062 0.02063 0.01449 0.01942 0.01992
 0.01488 0.01995 0.02028 0.01474 0.02018 0.02018]
radius [0.0255  0.01812 0.019   0.01857 0.0259  0.02594 0.01743 0.02384 0.02483
 0.0179  0.02487 0.02524 0.01767 0.02386 0.02502]
axis_0_-1 [0.02047 0.0186  0.0186  0.02198 0.02549 0.0264  0.02164 0.02554 0.02643
 0.02169 0.02546 0.02641 0.02133 0.0252  0.02529]
axis_0_+1 [0.03381 0.01811 0.01791 0.01822 0.02487 0.02487 0.01727 0.02297 0.02297
 0.01769 0.02392 0.02392 0.01742 0.02262 0.02262]
axis_1_-1 [0.04297 0.02583 0.02903 0.02509 0.03302 0.03302 0.02526 0.03275 0.03275
 0.02502 0.03263 0.03263 0.02454 0.03114 0.03114]
axis_1_+1 [0.0225  0.01287 0.01395 0.01913 0.02485 0.02485 0.01758 0.02359 0.02359
 0.01831 0.02422 0.02422 0.01807 0.02427 0.02427]
axis_2_-1 [0.01967 0.01733 0.0173  0.02388 0.03154 0.03266 0.02325 0.03084 0.03265
 0.02349 0.03119 0.03266 0.0231  0.03214 0.03214]
axis_2_+1 [0.04053 0.0321  0.03329 0.05476 0.08949 0.08949 0.0527  0.0831  0.0831
 0.05369 0.0863  0.0863  0.05272 0.08027 0.08027]
axis_3_-1 [0.03073 0.02239 0.02199 0.03235 0.04396 0.04396 0.03031 0.03983 0.03983
 0.03097 0.04144 0.04144 0.0307  0.04087 0.04087]
axis_3_+1 [0.03537 0.0257  0.02449 0.02118 0.02997 0.03397 0.01962 0.02725 0.03193
 0.02038 0.02849 0.03295 0.02019 0.02961 0.02961]
columns: scalar baseline/Matern/blend; state baseline/rational/blend/projected blend, each avg/terminal/worst horizon
worst original slices [0.0255  0.01812 0.01902 0.01857 0.0259  0.02594 0.01838 0.02395 0.02483
 0.01808 0.02487 0.02524 0.01818 0.02386 0.02502]
worst directional slices [0.04297 0.0321  0.03329 0.05476 0.08949 0.08949 0.0527  0.0831  0.0831
 0.05369 0.0863  0.0863  0.05272 0.08027 0.08027]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ba9f753d82585b48 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ba9f753d (directory).
```

### Turn 9 — `execute_in_remote_docker` (`28865 ms` LLM)
- **Rationale**: The scalar challenger improves the prescribed and directional splits. For trajectories, profile amplitude out of three one-rate laws and check the strong-signal rows: if temporal residuals approach the noise floor and rate is predictable, validate a joint forward model; otherwise retain the independently validated trajectory features.
- **Alternatives Weighed**: `Promote the projected trajectory blend immediately` (rejected: It improves the worst directional terminal error, but that error remains 0.08027. Testing whether amplitude and rate explain the two temporal modes could materially improve extrapolation.)

```bash
cat >/tmp/time_structure.py <<'PY'
from contextlib import redirect_stdout
from io import StringIO
with redirect_stdout(StringIO()):
 from cvprobe import *
from scipy.optimize import least_squares
for name in ('exp','log','rational'):
 def h(rate):
  z=rate*t
  return {'exp':lambda:-np.expm1(-z),'log':lambda:np.log1p(z),'rational':lambda:z/(1+z)}[name]()
 params=[]; residual=[]
 for row in j:
  def fun(r):
   a=h(r[0]);return row-a*(a@row)/(a@a)
  r=least_squares(fun,[1.],bounds=(.005,20.)).x[0]; a=h(r)
  params.append([a@row/(a@a),r]);residual.append(fun([r]))
 p=np.array(params); w=abs(j[:,-1])>.35
 print(name,'temporal RMSE',np.mean(np.array(residual)**2)**.5,'rates on strong rows',np.quantile(p[w,1],[0,.1,.5,.9,1]))
 print('rate r2 strong rows',[(i, round(1-np.sum((poly(x[w],i)@ls(poly(x[w],i),p[w,1])-p[w,1])**2)/np.sum((p[w,1]-p[w,1].mean())**2),4)) for i in [1,2]])
print('times',t)
U,S,vt=np.linalg.svd(j,full_matrices=False)
for label,T in [('dominant',j@vt[0]),('secondary',j@vt[1])]:
 print(label,'MAE per mode')
 for key,z in Z.items():
  for deg in (2,3):
   F=poly(z,deg); a=[]
   for it,iv in folds:
    a.append(np.mean(abs(F[iv]@ls(F[it],T[it])-T[iv])))
   if key=='x': print(deg,np.round(a,4))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/time_structure.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'variable projection viscoelastic relaxation' | head -12
```
- **Container Exit Code**: `0` (`908 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
exp temporal RMSE 0.014408232509206238 rates on strong rows [0.08863 0.49397 0.80874 1.32569 1.71656]
rate r2 strong rows [(1, 0.323), (2, 0.7464)]
log temporal RMSE 0.015400888036507963 rates on strong rows [ 0.09696  1.01353  2.55961  7.44632 14.28255]
rate r2 strong rows [(1, 0.2534), (2, 0.7575)]
rational temporal RMSE 0.010652145122332764 rates on strong rows [0.04634 0.34375 0.67029 1.33853 1.91925]
rate r2 strong rows [(1, 0.3017), (2, 0.7613)]
times [0.25    0.64286 1.03571 1.42857 1.82143 2.21429 2.60714 3.     ]
dominant MAE per mode
2 [0.0501 0.0514 0.0347 0.0358 0.0524 0.0768]
3 [0.0647 0.0431 0.0634 0.0515 0.1071 0.078 ]
secondary MAE per mode
2 [0.0044 0.0077 0.0076 0.0061 0.0069 0.0077]
3 [0.0117 0.0061 0.0068 0.0057 0.0141 0.0134]
No matching skill cards found.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ba9f753d82585b48 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ba9f753d (directory).
```

### Turn 10 — `execute_in_remote_docker` (`31661 ms` LLM)
- **Rationale**: Test whether a saturating amplitude or a different smooth trend reduces the positive-third-coordinate extrapolation error without worsening ordinary holdouts. If neither does, retain the validated projected blend and promote the Matérn scalar model.
- **Alternatives Weighed**: `Use the fitted rational relaxation law` (rejected: Even with each row’s entire measured trajectory available, its RMSE is 0.01065 versus roughly 0.0043 for two shared temporal modes.)

```bash
cat >/tmp/modal.py <<'PY'
from contextlib import redirect_stdout
from io import StringIO
with redirect_stdout(StringIO()):
 from solver_audit import *
from scipy.spatial.distance import cdist
F=m._poly(x,2); D=cdist(x,x,'sqeuclidean'); specs=[]
for power in [0,1,2]:
 for l in [.6,1.2,2.4,4.8]:
  for noise in [.0001,.003,.03]:
   specs.append(('gp',power,l,noise))
for mode in ['sinh','atanh','identity']:
 for scale in [1.05,1.4,2.0,3.0]:
  specs.append((mode,scale))
def pred(it,iv,c):
 if c[0]=='gp':
  _,p,l,nug=c; a=F[:,:{0:1,1:5,2:15}[p]]; r=(5*D)**.5/l; k=(1+r+r*r/3)*np.exp(-r); h=a[it]; w=np.linalg.solve(k[np.ix_(it,it)]+nug*np.eye(len(it)),np.c_[h,j[it]]); c0=np.linalg.solve(h.T@w[:,:h.shape[1]],h.T@w[:,h.shape[1]:]); return a[iv]@c0+k[np.ix_(iv,it)]@(w[:,h.shape[1]:]-w[:,:h.shape[1]]@c0)
 _,_,vt=np.linalg.svd(j[it],full_matrices=False); V=vt[:2]; u=j[it]@V.T; kind,scale=c; scale*=abs(u[:,0]).max(); f, inv={'sinh':(np.sinh,np.arcsinh),'atanh':(np.arctanh,np.tanh),'identity':(lambda z:z,lambda z:z)}[kind]; u[:,0]=scale*f(u[:,0]/scale); z=m._ridge(F[it],F[iv],u,np.r_[0.,[.001]*4,[.1]*10]); z[:,0]=scale*inv(z[:,0]/scale); return z@V

def get(it,iv,c):
 e=abs(pred(it,iv,c)-j[iv]); return e.mean(),e[:,-1].mean(),e.mean(axis=0).max()
sc=[]
for c in specs:
 a=np.array([get(np.setdiff1d(all_rows,iv),iv,c) for _,iv in folds]); sc.append((c,a))
for c,a in sorted(sc,key=lambda z:.65*z[1][:6,2].max()+.35*z[1][6:,2].max())[:12]: print(c,'public',a[0].round(5),'original worst',a[:6].max(0).round(5),'axis worst',a[6:].max(0).round(5),'axis 2+',a[11].round(5))
print('test extent vs train:',x.min(0),x.max(0),s['test']['x'].min(0),s['test']['x'].max(0)); print('public train summary',t['summary'])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/modal.py
```
- **Container Exit Code**: `0` (`567 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
286 0.01752 0.01752] original worst [0.02035 0.02977 0.02977] axis worst [0.02527 0.03587 0.03587] axis 2+ [0.02093 0.02891 0.02891]
('gp', 2, 4.8, 0.0001) public [0.01258 0.01792 0.01792] original worst [0.02044 0.03003 0.03003] axis worst [0.02408 0.03585 0.03585] axis 2+ [0.022   0.03056 0.03056]
('gp', 2, 1.2, 0.03) public [0.01361 0.01812 0.01812] original worst [0.02084 0.03005 0.03005] axis worst [0.02644 0.03511 0.03681] axis 2+ [0.02023 0.02729 0.02729]
('gp', 2, 1.2, 0.003) public [0.01366 0.01839 0.01839] original worst [0.0207  0.03    0.03009] axis worst [0.02576 0.03296 0.0371 ] axis 2+ [0.02022 0.02675 0.02675]
('gp', 2, 1.2, 0.0001) public [0.01372 0.01853 0.01853] original worst [0.02082 0.03014 0.03039] axis worst [0.02567 0.03242 0.03726] axis 2+ [0.02062 0.02672 0.02773]
('gp', 2, 2.4, 0.0001) public [0.01338 0.01862 0.01862] original worst [0.02099 0.03052 0.03052] axis worst [0.02485 0.03301 0.03829] axis 2+ [0.02485 0.03081 0.03638]
('sinh', 3.0) public [0.01662 0.02082 0.02082] original worst [0.02287 0.02973 0.02973] axis worst [0.03253 0.04028 0.04028] axis 2+ [0.02301 0.03021 0.03021]
('gp', 2, 2.4, 0.03) public [0.01414 0.01847 0.01847] original worst [0.02219 0.03113 0.03113] axis worst [0.02876 0.0391  0.0391 ] axis 2+ [0.02172 0.02755 0.02755]
('gp', 2, 0.6, 0.03) public [0.01472 0.01907 0.01918] original worst [0.02216 0.03056 0.03181] axis worst [0.02916 0.03623 0.03841] axis 2+ [0.02006 0.0259  0.0259 ]
('gp', 2, 0.6, 0.003) public [0.01469 0.01906 0.01912] original worst [0.02207 0.03049 0.03184] axis worst [0.02893 0.03583 0.03849] axis 2+ [0.02005 0.02594 0.02594]
('gp', 2, 0.6, 0.0001) public [0.01469 0.01906 0.01912] original worst [0.02206 0.03048 0.03184] axis worst [0.0289  0.03578 0.0385 ] axis 2+ [0.02005 0.02595 0.02595]
('gp', 2, 4.8, 0.003) public [0.01447 0.01904 0.01904] original worst [0.02252 0.03149 0.03149] axis worst [0.02959 0.04045 0.04045] axis 2+ [0.02229 0.02783 0.02793]
test extent vs train: [-1.13107893 -1.14769813 -1.11590378 -1.13452624] [1.06173166 1.07989355 1.14575252 1.13242102] [-1.03673431 -1.02881378 -1.07676103 -0.97635195] [1.05701819 0.97838215 1.13845297 1.10102549]
public train summary [0.07088742 0.46063947 0.12747991 0.93480004]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ba9f753d82585b48 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ba9f753d (directory).
```

### Turn 11 — `execute_in_remote_docker` (`36996 ms` LLM)
- **Rationale**: Promote the model that improves the prescribed split and directional stress slices. Refit the exact deliverable on each split in an isolated process to catch implementation errors or regression in either output, and run the required public schema validator.
- **Alternatives Weighed**: `Retain the panel-based trajectory blend` (rejected: Its worst directional horizon MAE was 0.08027, versus 0.03587 for the quadratic Matérn model.)

```bash
cat >/holdout_task/batch_solver.py <<'PY'
from itertools import combinations_with_replacement

import numpy as np


def _quadratic(x):
    pairs = combinations_with_replacement(range(x.shape[1]), 2)
    return np.column_stack([np.ones(len(x)), x] + [x[:, i] * x[:, j] for i, j in pairs])


def _matern52(u, v):
    r = (np.sqrt(5.0) / 2.4) * np.linalg.norm(u[:, None, :] - v[None, :, :], axis=2)
    return (1.0 + r + r * r / 3.0) * np.exp(-r)


def _predict(a, b, kernel, cross_kernel, target, regularization):
    # Fit the quadratic trend in the kernel metric, then predict its residual.
    w = np.linalg.solve(kernel + regularization * np.eye(len(a)), np.column_stack((a, target)))
    wa, wy = w[:, :a.shape[1]], w[:, a.shape[1]:]
    trend = np.linalg.solve(a.T @ wa, a.T @ wy)
    return b @ trend + cross_kernel @ (wy - wa @ trend)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, q = train['x'], study['test']['x']
    a, b = _quadratic(x), _quadratic(q)
    kernel, cross_kernel = _matern52(x, x), _matern52(q, x)
    y = _predict(a, b, kernel, cross_kernel, train['y'], 0.001)[:, 0]
    state = _predict(a, b, kernel, cross_kernel, train['traj'], 0.003)
    final = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), final.mean(), final.std()])}
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', root+'batch_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open(root+'public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
t = s['train']; x = t['x']; n = len(x); rows = np.arange(n)
folds = [('prescribed', rows[52:])]
folds += [(f'random_{i}', v) for i, v in enumerate(np.array_split(np.random.default_rng(37).permutation(n), 4))]
folds += [('high_radius', np.argsort(np.linalg.norm(x, axis=1))[-15:])]
folds += [(f'axis_{i}_{sign:+d}', np.argsort(sign*x[:, i])[-12:]) for i in range(x.shape[1]) for sign in (-1, 1)]
scores = []
for name, v in folds:
    fit = rows[:52] if name == 'prescribed' else np.setdiff1d(rows, v)
    out = mod.solve_batch(dict(s, train={k: t[k][fit] for k in ('x', 'y', 'traj')}, test={'x': x[v]}))
    yp, sp = out['y_hat'], out['state_pred']
    assert yp.shape == (len(v),) and sp.shape == (len(v), len(s['eval_times']))
    assert all(np.isfinite(a).all() for a in out.values())
    np.testing.assert_allclose(out['diagnostics_vec'], [yp.mean(), yp.std(), sp[:, -1].mean(), sp[:, -1].std()])
    e = abs(sp - t['traj'][v]).mean(0)
    scores.append((abs(yp-t['y'][v]).mean(), e.mean(), e[-1], e.max()))
    print(name, np.round(scores[-1], 5))
print('columns: y MAE, trajectory MAE, terminal MAE, worst-horizon MAE')
worst = np.max(scores, axis=0)
print('worst all slices:', np.round(worst, 5))
assert worst[0] <= 0.038 and worst[3] < 0.04
PY
```
- **Container Exit Code**: `0` (`482 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
prescribed [0.00885 0.01286 0.01752 0.01752]
random_0 [0.00843 0.01085 0.01349 0.01367]
random_1 [0.01811 0.01176 0.01562 0.01562]
random_2 [0.01265 0.01102 0.01561 0.01561]
random_3 [0.01112 0.0129  0.01623 0.01645]
high_radius [0.01812 0.02035 0.02977 0.02977]
axis_0_-1 [0.0186  0.01612 0.02447 0.02447]
axis_0_+1 [0.01811 0.01768 0.02411 0.02411]
axis_1_-1 [0.02583 0.01934 0.02467 0.02467]
axis_1_+1 [0.01287 0.01351 0.01725 0.01845]
axis_2_-1 [0.01733 0.0185  0.02167 0.02769]
axis_2_+1 [0.0321  0.02093 0.02891 0.02891]
axis_3_-1 [0.02239 0.02527 0.03587 0.03587]
axis_3_+1 [0.0257  0.02075 0.02914 0.02914]
columns: y MAE, trajectory MAE, terminal MAE, worst-horizon MAE
worst all slices: [0.0321  0.02527 0.03587 0.03587]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=77d4313f4b709bcd (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_77d4313f (directory).
```

### Turn 12 — `submit_completed_task` (`13837 ms` LLM)
- **Rationale**: The saved candidate passed the required schema validator and isolated-process testing on 14 held-out slices. Its nonlinear basis addresses the measured affine-model failure without claiming recovery of the physical mechanism.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema validator passed. On the prescribed split, scalar MAE was 0.00885, below the 0.038 limit, and trajectory MAE was 0.01286. Across all stress slices, the worst scalar MAE was 0.03210 and the worst trajectory-horizon MAE was 0.03587.`

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
