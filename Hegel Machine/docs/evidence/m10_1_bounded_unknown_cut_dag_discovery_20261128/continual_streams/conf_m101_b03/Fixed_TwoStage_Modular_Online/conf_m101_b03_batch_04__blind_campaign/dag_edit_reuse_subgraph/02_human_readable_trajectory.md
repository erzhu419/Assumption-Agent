# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m101_b03_batch_04__blind_campaign__reuse_subgraph`

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
| **Turns Used / Max Turns** | `11 / 12` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `309,171` (`297,638` / `98,944` / `11,533`) |
| **Total Episode Wall Time (s)** | `364.24s` |

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

### Turn 1 — `execute_in_remote_docker` (`15173 ms` LLM)
- **Rationale**: Inspect the frozen node, evaluator, and a small numerical preview to identify the actual fit inputs and output shapes before selecting a basis.

```bash
cd /holdout_task && cat _dag_spec.json _frozen_nodes/node_00.py test_dag_candidate.py validate_public_schema.py && python3 - <<'PY'
import pickle
import numpy as np
with open('public_batch.pkl','rb') as f: batch=pickle.load(f)
def show(x,p='',d=0):
 if isinstance(x,dict):
  for k,v in x.items(): show(v,p+'/'+str(k),d+1)
 elif isinstance(x,np.ndarray):
  print(p,x.shape,x.dtype,repr(x[:min(len(x),3)]) if x.ndim else repr(x))
 elif isinstance(x,(list,tuple)):
  print(p,type(x).__name__,len(x))
  for i,v in enumerate(x[:2]): show(v,p+'/'+str(i),d+1)
 else: print(p,repr(x)[:200])
show(batch)
PY
```
- **Container Exit Code**: `0` (`487 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
47774],
       [-0.43917815,  0.59034978,  0.96601709,  0.56875456],
       [ 0.94231712, -0.54115598, -0.34156726,  0.53314618]])
/0/study/panel_a/v (48, 4) float64 array([[ 1.73530943,  0.57828247,  0.07475772,  1.06849072],
       [-0.97075493,  0.440096  ,  1.00588464,  0.58581524],
       [ 1.15089307, -0.27866792, -0.70736207,  0.3427241 ]])
/0/study/panel_b/u (24, 4) float64 array([[-0.93899989, -0.04969768, -0.65586835, -0.02211051],
       [-1.22350214, -0.22661674, -0.93731687,  1.11035744],
       [-0.81292343, -1.16876736, -0.22592094, -0.8165147 ]])
/0/study/panel_b/v (24, 4) float64 array([[ 0.35553325, -0.28106387,  0.73153908,  0.35417606],
       [ 0.13649827,  0.1276991 ,  0.83011399,  1.31180894],
       [ 0.57842958,  0.40860834,  1.10151755, -0.6214239 ]])
/0/study/panel_c/u (24, 4) float64 array([[ 0.89739205,  0.24198475,  1.1800114 ,  0.29768402],
       [-0.30295471, -0.69028297,  0.60860289,  0.15983785],
       [-0.02353531, -0.41786703, -0.39764073, -0.80614839]])
/0/study/panel_c/v1 (24, 4) float64 array([[-0.30597758, -0.11031408,  0.56672133,  0.6006792 ],
       [-0.53024676, -0.00910207,  0.25572114, -0.30751916],
       [-0.36676232, -0.27226875,  0.21973281,  0.43577125]])
/0/study/panel_c/v2 (24, 4) float64 array([[-0.30745066,  0.15629751,  0.16116068, -0.86464241],
       [ 0.08260982,  0.3248528 , -0.23838475, -0.10855695],
       [ 0.30134628,  0.00230048,  0.09761446,  0.26782645]])
/0/study/train/x (72, 4) float64 array([[-0.61046641, -0.72592054, -0.64168413,  0.89893719],
       [ 0.94398197, -1.11042694,  0.79782163, -1.35776829],
       [-0.93786616,  0.65374453,  0.16798327, -1.12733279]])
/0/study/train/y (72,) float64 array([-1.64898216, -0.26715757,  0.58426279])
/0/study/train/traj (72, 8) float64 array([[-0.27744393, -0.56030534, -0.73366415, -0.8506829 , -0.93271209,
        -1.00441121, -1.04828283, -1.09469102],
       [-0.08636905, -0.14932455, -0.15032723, -0.12598801, -0.11359959,
        -0.08138625, -0.07400441, -0.05154698],
       [ 0.05985712,  0.12249267,  0.13959086,  0.15283165,  0.15770392,
         0.15836859,  0.16130594,  0.16063607]])
/0/study/train/summary (4,) float64 array([-0.26269763,  0.723851  ,  0.21412718])
/0/study/test/x (36, 4) float64 array([[-1.08978625, -0.0366479 , -0.90924932,  0.17162967],
       [ 1.37435704,  0.89619477, -1.44188711,  1.23045586],
       [-0.93792264,  1.22793799, -0.15099354, -0.99588794]])
/0/study/telemetry_vec (4,) float64 array([0.1, 0.2, 0.2])
```

### Turn 2 — `execute_in_remote_docker` (`24198 ms` LLM)
- **Rationale**: Compare extrapolation of candidate spatial bases and inspect trajectory rank. If the trajectory is low-rank, fit its temporal modes jointly rather than eight unrelated regressions.

```bash
cd /holdout_task && sed -n '40,180p' test_dag_candidate.py && cat public_asset_index.json && cat > /tmp/p.py <<'PY'
import numpy as np,pickle,sys
from itertools import combinations,combinations_with_replacement
sys.path.insert(0,'/holdout_task/_frozen_nodes')
from node_00 import run_node
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=run_node({'__input__':s}); z=d['z']; x=z[:d['n_train']]; t=d['eval_times']; y=d['y']; T=d['traj']; Y=np.c_[y,T]; n=len(x)
ones=np.ones(n)
def poly(deg):return np.column_stack([ones]+[np.prod(x[:,c],axis=1) for k in range(1,deg+1) for c in combinations_with_replacement(range(4),k)])
def report(k,A):
 P=np.linalg.lstsq(A,Y,rcond=None)[0]; E=Y-A@P; H=(np.linalg.svd(A,full_matrices=False)[0][:,:np.linalg.matrix_rank(A)]**2).sum(1); res=np.abs(E/(1-H[:,None])); print(k,A.shape[1], 'train',np.mean(abs(E),0).round(4),'loo',res.mean(0).round(4),'tail',np.mean(abs(Y[-18:]-A[-18:]@np.linalg.lstsq(A[:-18],Y[:-18],rcond=None)[0]),0).round(4))
if __name__=='__main__':
 print('ranges',x.min(0),x.max(0),'t',t,'T svd',np.linalg.svd(T)[1],'T temporal modes',np.linalg.svd(T)[2][:4].round(4));print('T rows',T[:8].round(4)); print('x y',np.c_[x,y][:8].round(4))
 for k in (1,2,3):report('poly'+str(k),poly(k))
 for f in (np.sin,np.tanh,np.cos):report(f.__name__,np.c_[poly(2),f(x)])
 print('coef quadratic',np.linalg.lstsq(poly(2),Y,rcond=None)[0].round(3))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/p.py
```
- **Container Exit Code**: `0` (`780 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
23 0.0953 0.1239 0.1387 0.1467 0.151  0.1588 0.1606]
poly3 35 train [0.0155 0.0061 0.0099 0.0122 0.0145 0.0159 0.0154 0.0167 0.017 ] loo [0.0453 0.0176 0.0279 0.0344 0.041  0.0455 0.0445 0.0481 0.0501] tail [0.0434 0.0175 0.0304 0.0322 0.0403 0.0455 0.049  0.0511 0.0544]
sin 19 train [0.0779 0.0244 0.0468 0.0585 0.0641 0.0685 0.0711 0.0732 0.0747] loo [0.1169 0.0366 0.0699 0.0878 0.0962 0.1028 0.1069 0.1101 0.1126] tail [0.1344 0.0484 0.092  0.1192 0.1345 0.145  0.1516 0.1584 0.1638]
tanh 19 train [0.0777 0.0245 0.0468 0.0586 0.0641 0.0685 0.0713 0.0732 0.0748] loo [0.1162 0.0366 0.0698 0.0877 0.0961 0.1026 0.107  0.1099 0.1126] tail [0.1335 0.0484 0.0921 0.1195 0.1348 0.1454 0.1521 0.1588 0.1641]
cos 19 train [0.0796 0.0256 0.0473 0.06   0.0643 0.0693 0.0726 0.0737 0.0748] loo [0.1208 0.0386 0.0718 0.0908 0.0978 0.1051 0.1102 0.1122 0.114 ] tail [0.1694 0.0568 0.1015 0.1317 0.1457 0.1534 0.1576 0.1644 0.1646]
coef quadratic [[-5.900e-02 -1.000e-02 -2.300e-02 -4.000e-02 -5.300e-02 -6.100e-02
  -7.200e-02 -8.100e-02 -8.700e-02]
 [ 7.000e-02  1.190e-01  2.630e-01  3.730e-01  4.560e-01  5.250e-01
   5.810e-01  6.300e-01  6.700e-01]
 [ 3.600e-02  9.400e-02  2.040e-01  2.860e-01  3.510e-01  4.010e-01
   4.420e-01  4.780e-01  5.080e-01]
 [-4.290e-01  1.000e-02  2.800e-02  4.200e-02  5.600e-02  6.700e-02
   7.900e-02  8.700e-02  9.500e-02]
 [-4.210e-01  2.000e-03  1.900e-02  3.300e-02  5.000e-02  6.200e-02
   7.500e-02  8.800e-02  9.700e-02]
 [-1.700e-02  4.600e-02  1.010e-01  1.510e-01  1.910e-01  2.200e-01
   2.480e-01  2.710e-01  2.890e-01]
 [-9.270e-01  2.720e-01  5.100e-01  6.270e-01  6.840e-01  7.140e-01
   7.230e-01  7.290e-01  7.330e-01]
 [-7.000e-03 -1.000e-03 -2.000e-03 -4.000e-03 -3.000e-03 -7.000e-03
  -7.000e-03 -7.000e-03 -5.000e-03]
 [ 8.430e-01  2.860e-01  5.410e-01  6.630e-01  7.180e-01  7.460e-01
   7.540e-01  7.590e-01  7.600e-01]
 [ 5.000e-02  4.300e-02  9.200e-02  1.190e-01  1.470e-01  1.640e-01
   1.780e-01  1.930e-01  2.010e-01]
 [ 4.250e-01 -4.570e-01 -8.620e-01 -1.058e+00 -1.151e+00 -1.189e+00
  -1.211e+00 -1.219e+00 -1.220e+00]
 [-5.700e-02 -1.000e-02 -1.400e-02 -1.500e-02 -1.200e-02 -1.000e-02
  -5.000e-03 -6.000e-03 -1.000e-03]
 [-2.060e-01 -5.000e-03 -1.200e-02 -1.100e-02 -1.300e-02 -1.500e-02
  -1.500e-02 -1.700e-02 -1.700e-02]
 [-4.580e-01  1.290e-01  2.460e-01  2.990e-01  3.270e-01  3.400e-01
   3.480e-01  3.500e-01  3.560e-01]
 [-2.150e-01 -5.000e-03 -7.000e-03 -1.000e-03  5.000e-03  8.000e-03
   1.400e-02  1.700e-02  2.300e-02]]
```

### Turn 3 — `execute_in_remote_docker` (`25603 ms` LLM)
- **Rationale**: Compare coefficient support and held-out error to distinguish stable cross-coordinate structure from terms that only improve the training fit.
- **Alternatives Weighed**: `Use the dense cubic fit` (rejected: Its response leave-one-out MAE and late-horizon trajectory errors exceed tolerance.)
- **ATMS NOGOOD DECLARED**: Refuted `A complete quadratic basis, optionally with separable sine, cosine, or tanh terms, is sufficient.` | Witness: `Quadratic leave-one-out response MAE was 0.1139; adding all cubic monomials reduced it to 0.0453, still above tolerance.` | Blanket Diagnosis: `The quadratic basis omits higher-order interactions, while the dense cubic fit may retain unsupported terms.` | Sublated Pivot: `Identify cubic terms stable across fits, with term selection repeated inside every validation fold.`

```bash
cat > /tmp/readout_probe.py <<'PY'
from p import *
idx=[c for k in range(4) for c in combinations_with_replacement(range(4),k)]
A=poly(3)
P=np.linalg.lstsq(A,Y,rcond=None)[0]
Q=np.linalg.lstsq(A[:52],Y[:52],rcond=None)[0]
print('monomial: response full/prefix, trajectory first/last full/prefix')
for c,p,q in zip(idx,P,Q): print(''.join(map(str,c)) or '1',np.round([p[0],q[0],p[1],q[1],p[-1],q[-1]],4))
print('trajectory singular values',np.linalg.svd(T,compute_uv=False).round(4))
print('coefficient singular values',np.linalg.svd(P[:,1:],compute_uv=False).round(4))
def fit(ii,j,cut):
 a=A[ii]; b=Y[ii,j]; keep=np.arange(a.shape[1]); sc=np.sqrt((a*a).mean(0)); a=a/sc
 while True:
  co=np.linalg.lstsq(a[:,keep],b,rcond=None)[0]; valid=np.max(abs(co),axis=1)>cut if co.ndim==2 else abs(co)>cut
  if np.all(valid):break
  keep=keep[valid]
 return keep,co/sc[keep,None] if co.ndim==2 else co/sc[keep]
for j in (0,slice(1,None)):
 for cut in (.006,.012,.02,.03,.045,.06,.08):
  k,c=fit(np.arange(52),j,cut); ee=[]
  for ii in np.array_split(np.random.default_rng(39).permutation(72),6):
   kk,cc=fit(np.setdiff1d(np.arange(72),ii),j,cut); ee.append(np.mean(abs(Y[ii,j]-A[ii][:,kk]@cc)))
  print(j,cut,len(k),'tail',round(np.mean(abs(Y[52:,j]-A[52:][:,k]@c)),5),'folds',np.round(ee,4),'terms',[idx[i] for i in k])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/readout_probe.py
```
- **Container Exit Code**: `0` (`646 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2, 3), (3, 3), (0, 0, 1), (0, 1, 3), (1, 2, 2), (2, 2, 3)]
0 0.045 13 tail 0.06363 folds [0.0803 0.1026 0.1045 0.0772 0.0618 0.0476] terms [(), (2,), (3,), (0, 1), (0, 3), (1, 2), (2, 2), (2, 3), (3, 3), (0, 0, 1), (0, 1, 3), (1, 2, 2), (2, 2, 3)]
0 0.06 12 tail 0.05837 folds [0.0955 0.1012 0.0859 0.0984 0.0509 0.0476] terms [(2,), (3,), (0, 1), (0, 3), (1, 2), (2, 2), (2, 3), (3, 3), (0, 0, 1), (0, 1, 3), (1, 2, 2), (2, 2, 3)]
0 0.08 8 tail 0.12166 folds [0.0848 0.1326 0.0971 0.0984 0.0509 0.0476] terms [(2,), (3,), (0, 1), (0, 3), (1, 2), (2, 2), (2, 3), (0, 1, 3)]
slice(1, None, None) 0.006 35 tail 0.05004 folds [0.0489 0.0581 0.0412 0.0395 0.035  0.0307] terms [(), (0,), (1,), (2,), (3,), (0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3), (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 2, 2), (0, 2, 3), (0, 3, 3), (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 2), (1, 2, 3), (1, 3, 3), (2, 2, 2), (2, 2, 3), (2, 3, 3), (3, 3, 3)]
slice(1, None, None) 0.012 24 tail 0.04224 folds [0.0539 0.0508 0.0474 0.0335 0.033  0.0231] terms [(), (0,), (1,), (2,), (3,), (0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (2, 3), (3, 3), (0, 0, 1), (0, 0, 2), (0, 1, 2), (0, 1, 3), (0, 2, 2), (1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 2, 3), (2, 2, 3), (2, 3, 3)]
slice(1, None, None) 0.02 23 tail 0.0422 folds [0.0416 0.0494 0.0496 0.0324 0.0231 0.0176] terms [(), (0,), (1,), (2,), (3,), (0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (2, 3), (3, 3), (0, 0, 1), (0, 1, 2), (0, 1, 3), (0, 2, 2), (1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 2, 3), (2, 2, 3), (2, 3, 3)]
slice(1, None, None) 0.03 20 tail 0.0404 folds [0.0501 0.0556 0.0487 0.0499 0.023  0.0402] terms [(), (0,), (1,), (2,), (3,), (0, 0), (0, 1), (0, 3), (1, 1), (1, 2), (2, 3), (3, 3), (0, 0, 1), (0, 1, 2), (0, 1, 3), (1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 2, 3), (2, 3, 3)]
slice(1, None, None) 0.045 17 tail 0.03663 folds [0.0501 0.0969 0.0388 0.0546 0.0316 0.042 ] terms [(), (0,), (1,), (3,), (0, 0), (0, 1), (0, 3), (1, 1), (1, 2), (2, 3), (0, 0, 1), (0, 1, 2), (0, 1, 3), (1, 1, 2), (1, 2, 2), (1, 2, 3), (2, 3, 3)]
slice(1, None, None) 0.06 14 tail 0.06389 folds [0.0428 0.1048 0.055  0.0546 0.0743 0.0433] terms [(), (0,), (1,), (0, 0), (0, 1), (0, 3), (1, 1), (1, 2), (2, 3), (0, 0, 1), (0, 1, 3), (1, 1, 2), (1, 2, 2), (1, 2, 3)]
slice(1, None, None) 0.08 7 tail 0.06839 folds [0.1007 0.1523 0.1127 0.1104 0.0723 0.056 ] terms [(0,), (1,), (0, 0), (0, 1), (0, 3), (1, 2), (2, 3)]
```

### Turn 4 — `execute_in_remote_docker` (`34640 ms` LLM)
- **Rationale**: Test whether higher-order terms improve held-out errors when regularization is selected inside each fold. Inspect panel C and the joint residual modes for a lower-dimensional algebraic explanation before editing the sink.
- **Alternatives Weighed**: `Promote coefficient thresholding` (rejected: Its best response tail error misses tolerance, and its worst random-fold errors are substantially larger.)

```bash
cat > /tmp/probe4.py <<'PY'
from p import *
from scipy.linalg import solve
from scipy.optimize import minimize
from math import factorial

def kernels(a,b):
 g=a@b.T
 return [np.ones_like(g),g,g*g,g**3,g**4,g**5,g**6]

def search(ii,j,kind):
 a=x[ii]; b=Y[ii,j]; b=b.reshape(len(ii),-1); K=kernels(a,a); sc=np.std(b,axis=0)
 def model(h,K):
  if kind==0: return sum(q*np.exp(w) for q,w in zip(K,h[:-1]))
  if kind==1: return (1+np.exp(h[0])*K[1])**int(h[1])
 def fun(h):
  M=model(h,K)+np.eye(len(ii))*np.exp(h[-1]); inv=solve(M,np.eye(len(ii)),assume_a='pos'); e=(inv@b)/np.diag(inv)[:,None]/sc
  return np.mean(e**2)
 if kind==0: h=minimize(fun,[1,0,0,-2,-4,-5,-6,-7],bounds=[(-15,8)]*8,method='L-BFGS-B').x
 else: h=min((minimize(fun,[0,o,-7],bounds=[(-7,3),(o,o),(-15,4)],method='L-BFGS-B').x for o in range(2,8)),key=fun)
 return h,lambda jj:model(h,kernels(x[jj],a))@solve(model(h,K)+np.eye(len(ii))*np.exp(h[-1]),b,assume_a='pos')
for j in (0,slice(1,None)):
 for kind in (0,1):
  folds=[np.arange(52,72),*np.array_split(np.random.default_rng(39).permutation(72),6)]
  err=[]
  for jj in folds:
   h,pred=search(np.setdiff1d(np.arange(72),jj),j,kind); err.append(np.mean(abs(Y[jj,j].reshape(len(jj),-1)-pred(jj))))
  print(j,kind,'errors',np.round(err,5),'params',h.round(2))

a=s['panel_c']['u']; V=s['panel_c'];
for key in ('v1','v2'):
 print('panel C',key)
 for j in range(4):
  b=V[key][:,j]; ids=[c for k in range(3) for c in combinations_with_replacement(range(4),k)]; A=np.column_stack([np.prod(a[:,c],axis=1) for c in ids]); co=np.linalg.lstsq(A,b,rcond=None)[0]; print(j,'range',np.round([min(b),max(b)],3),'quad rmse',np.std(b-A@co).round(4),[(i,round(co[i2],3)) for i2,i in enumerate(ids) if abs(co[i2])>0.12])
# Check whether nonlinear terms cancel between response and trajectory modes.
U,S,R=np.linalg.svd(T,full_matrices=False); W=U[:,:2]*S[:2]; Z=np.c_[y,W];
for k in (1,2,3):
 F=poly(k); e=Z-F@np.linalg.lstsq(F,Z,rcond=None)[0]; _,D,C=np.linalg.svd(e,full_matrices=False); print('target residual singular',k,D.round(5),C.round(3))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe4.py
```
- **Container Exit Code**: `0` (`2289 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 0 errors [0.03559 0.02857 0.05358 0.02877 0.03702 0.01568 0.02752] params [  1.01   4.61   3.56  -1.57  -3.85  -7.33 -14.31  -5.66]
0 1 errors [0.04312 0.03239 0.05572 0.03501 0.03774 0.02342 0.02857] params [-2.11  5.   -8.8 ]
slice(1, None, None) 0 errors [0.02995 0.02864 0.05942 0.02444 0.01926 0.01625 0.01751] params [  1.28   4.79   5.41  -2.06  -4.53  -7.18 -13.67  -6.77]
slice(1, None, None) 1 errors [0.03116 0.03451 0.06244 0.0238  0.02002 0.01671 0.01661] params [-2.82  7.   -9.81]
panel C v1
0 range [-1.962  0.826] quad rmse 0.0001 [((), -0.144), ((1, 2), 0.637), ((2, 3), -1.029)]
1 range [-1.278  1.098] quad rmse 0.0 [((), -0.135), ((0, 1), -0.146), ((0, 3), 1.061), ((1, 2), -0.542), ((2, 3), -0.207)]
2 range [-0.374  1.484] quad rmse 0.0001 [((), 0.123), ((0, 1), 1.146), ((0, 3), 0.308), ((1, 2), 0.247), ((2, 3), 0.121)]
3 range [-0.632  0.984] quad rmse 0.0001 [((0, 1), -0.381), ((0, 3), 0.5), ((1, 2), 0.85), ((2, 3), 0.605)]
panel C v2
0 range [-0.915  0.321] quad rmse 0.0019 [((0,), -0.125), ((1,), -0.188), ((3,), -0.424), ((3, 3), -0.19)]
1 range [-0.77   0.446] quad rmse 0.0033 [((1,), -0.43), ((2,), 0.132), ((3,), 0.191), ((1, 1), -0.194)]
2 range [-0.591  0.765] quad rmse 0.003 [((0,), 0.434), ((2,), -0.177), ((0, 0), 0.203)]
3 range [-0.865  0.475] quad rmse 0.0028 [((), 0.125), ((0,), -0.178), ((2,), -0.431), ((2, 2), -0.193)]
target residual singular 1 [12.91876  4.91379  0.49059] [[ 0.161 -0.984  0.072]
 [-0.987 -0.159  0.026]
 [-0.014 -0.076 -0.997]]
target residual singular 2 [2.40565 0.40694 0.03701] [[ 0.381  0.923 -0.046]
 [-0.851  0.37   0.372]
 [-0.361  0.103 -0.927]]
target residual singular 3 [0.42484 0.07835 0.03118] [[ 0.362  0.932 -0.028]
 [-0.844  0.341  0.415]
 [-0.396  0.127 -0.91 ]]
```

### Turn 5 — `execute_in_remote_docker` (`39947 ms` LLM)
- **Rationale**: Compare temporal laws against the rank-two noise floor and test whether a common rational denominator actually improves held-out predictions. If it does not, focus on finding the shared spatial features rather than expanding the polynomial basis further.
- **Alternatives Weighed**: `Ship the regularized polynomial baseline` (rejected: It passes the public tail split, but other folds exceed tolerance and the trajectory has clear low-rank structure worth exploiting.)

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'variable projection exponential relaxation' || true
cat > /tmp/probe5.py <<'PY'
from p import *
from scipy.optimize import minimize, minimize_scalar
from scipy.linalg import qr
np.set_printoptions(precision=5,suppress=True,linewidth=160)
U,S,V=np.linalg.svd(T,full_matrices=False)
for name,gen,start in [('step+t',lambda p:np.c_[-np.expm1(-p[0]*t),t],[1.]),('step+te',lambda p:np.c_[-np.expm1(-p[0]*t),t*np.exp(-p[0]*t)],[1.]),('two steps',lambda p:np.column_stack([-np.expm1(-q*t) for q in p]),[.3,1.4]),('step+sine',lambda p:np.c_[-np.expm1(-p[0]*t),np.sin(p[1]*t)],[1.,.7])]:
 def loss(p):
  A=gen(p); return np.mean((T-T@np.linalg.pinv(A).T@A.T)**2)
 opt=minimize(loss,start,method='Nelder-Mead',options={'xatol':1e-10}); A=gen(opt.x); print(name,opt.x,np.sqrt(opt.fun),A.T)
 print('amplitudes ranges',np.ptp(T@np.linalg.pinv(A).T,axis=0))
print('SVD basis',V[:2]); print('noise floor',np.sqrt(np.sum(S[2:]**2)/T.size))
Z=np.c_[y,U[:,:2]*S[:2]]
for k in (1,2,3):
 A=poly(k); E=Z-A@np.linalg.lstsq(A,Z,rcond=None)[0]; _,D,W=np.linalg.svd(E,full_matrices=False); print('residual basis',k,D,W)
 if k==2:
  C=np.linalg.lstsq(A,Z@W.T,rcond=None)[0]; print('poly2 target rotated coefficients',C)
for name in ('panel_b','panel_c'):
 a=s[name]['u']; ids=[c for k in range(3) for c in combinations_with_replacement(range(4),k)]; A=np.column_stack([np.prod(a[:,c],axis=1) for c in ids]); print(name)
 for key,b in s[name].items():
  if key=='u': continue
  C=np.linalg.lstsq(A,b,rcond=None)[0]; print(key,'quadratic coefficients\n',C,'rmse',np.sqrt(np.mean((b-A@C)**2,axis=0)))
# See if a common rational divisor reduces cubic residuals without a large model.
for kind in ('squares','quadratic','diagonal_affine'):
 A=poly(3); aa=poly(2); B=x*x if kind=='squares' else aa[:,5:] if kind=='quadratic' else np.c_[x,x*x];
 for scale in (0.,.01,.1):
  errs=[]
  for jj in [np.arange(52,72),*np.array_split(np.random.default_rng(39).permutation(72),6)]:
   ii=np.setdiff1d(np.arange(72),jj); F=A[ii]; q=np.linalg.qr(F,mode='reduced')[0]; yi=Z[ii]; H=(yi[:,:,None]*B[ii,None,:]).reshape(len(ii),-1); E=yi-q@(q.T@yi); H=(H-q@(q.T@H)).reshape(len(ii),3,-1); C=np.linalg.lstsq(np.vstack([H.reshape(-1,B.shape[1]),np.eye(B.shape[1])*scale]),np.r_[-E.ravel(),np.zeros(B.shape[1])],rcond=None)[0]; den=1+B@C; P=np.linalg.lstsq(F,yi*den[ii,None],rcond=None)[0]; pred=(A[jj]@P)/den[jj,None]; errs.append([np.mean(abs(pred[:,0]-y[jj])),np.mean(abs(pred[:,1:]@V[:2]-T[jj])),den.min()])
  print('rational',kind,scale,'tail',np.round(errs[0],4),'worst',np.max(errs,0).round(4))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe5.py
```
- **Container Exit Code**: `0` (`784 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
244 -0.23689  0.7429 ]
 [-0.00004  0.00002  0.00009  0.00012]
 [ 0.00005  0.00008  0.00009  0.00002]
 [ 0.00001  0.00009  0.00001  0.00004]
 [-0.00009  0.00001  0.00001  0.00016]
 [-0.00005 -0.00003  0.00002  0.00001]
 [ 0.0001   0.00005 -0.00009 -0.00022]
 [-0.       0.00001 -0.00009 -0.00009]
 [ 0.00001 -0.00007  0.00001 -0.00002]
 [ 0.00004 -0.00013  0.00004  0.00003]
 [-0.00003 -0.00007  0.00013  0.00024]] rmse [0.00007 0.00006 0.00004 0.00006]
panel_c
v1 quadratic coefficients
 [[-0.14415 -0.1345   0.12282  0.09457]
 [ 0.00001 -0.00006  0.0001   0.00006]
 [-0.00008 -0.00002 -0.00011 -0.00012]
 [ 0.00004  0.00002  0.00009  0.00002]
 [ 0.00009  0.00001 -0.00005 -0.00002]
 [-0.00022 -0.00016  0.00001  0.00016]
 [-0.06025 -0.14642  1.14587 -0.38113]
 [ 0.       0.00003 -0.0001  -0.00004]
 [ 0.11628  1.061    0.30785  0.49955]
 [-0.00022 -0.00016  0.00006  0.00008]
 [ 0.63744 -0.54171  0.24696  0.85022]
 [-0.00027 -0.00012 -0.       0.00002]
 [-0.00003  0.00008 -0.00002 -0.00001]
 [-1.02948 -0.2071   0.12051  0.60525]
 [-0.00002  0.00006  0.00007  0.00006]] rmse [0.00006 0.00004 0.00005 0.00006]
v2 quadratic coefficients
 [[ 0.01782 -0.00809 -0.01888  0.12531]
 [-0.1254  -0.00732  0.43358 -0.17778]
 [-0.1881  -0.42981 -0.10233 -0.08827]
 [ 0.01091  0.13183 -0.17656 -0.43077]
 [-0.42424  0.19074 -0.08419  0.07603]
 [-0.06083  0.00325  0.20262 -0.07531]
 [-0.00144  0.00471 -0.00572 -0.00572]
 [ 0.00035 -0.0046   0.00734  0.00085]
 [-0.00063 -0.00227  0.00275 -0.00218]
 [-0.0763  -0.19383 -0.04282 -0.03474]
 [ 0.00082  0.01005 -0.00471 -0.00312]
 [-0.00091 -0.004    0.00042  0.00369]
 [ 0.01132  0.04821 -0.08151 -0.19258]
 [ 0.00168 -0.0064  -0.00026  0.00251]
 [-0.19045  0.09157 -0.04494  0.03691]] rmse [0.00191 0.00326 0.00299 0.00278]
rational squares 0.0 tail [0.057  0.0392 0.8851] worst [0.0736 0.0715 0.9928]
rational squares 0.01 tail [0.057  0.0392 0.8851] worst [0.0736 0.0715 0.9928]
rational squares 0.1 tail [0.057  0.0392 0.8857] worst [0.0736 0.0714 0.9928]
rational quadratic 0.0 tail [0.0377 0.0721 0.796 ] worst [0.0735 0.0721 0.9057]
rational quadratic 0.01 tail [0.0377 0.0721 0.796 ] worst [0.0735 0.0721 0.9057]
rational quadratic 0.1 tail [0.0377 0.071  0.7985] worst [0.0731 0.071  0.9073]
rational diagonal_affine 0.0 tail [0.034  0.0192 0.5215] worst [0.0721 0.0521 0.5265]
rational diagonal_affine 0.01 tail [0.034  0.0192 0.522 ] worst [0.072  0.0521 0.5267]
rational diagonal_affine 0.1 tail [0.0348 0.0199 0.5611] worst [0.0666 0.0525 0.5611]
```

### Turn 6 — `execute_in_remote_docker` (`36960 ms` LLM)
- **Rationale**: Panel C v1 isolates four cyclic pair products. Test whether modulating those products with calibrated coordinates or panel-derived transforms explains the higher-order terms, and check which transforms the frozen handoff actually makes available.
- **Alternatives Weighed**: `Use a common rational denominator` (rejected: The tested denominators improve some tail scores but worsen the worst-fold errors relative to the regularized baseline.)

```bash
cat /holdout_task/_frozen_nodes/node_00.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling' || true
cat > /tmp/probe6.py <<'PY'
from p import *
from scipy.optimize import minimize_scalar
from itertools import product
U,S,V=np.linalg.svd(T,full_matrices=False); Z=np.c_[y,U[:,:2]*S[:2]]
B=s['panel_b']; C=s['panel_c'];
def aff(a):return np.c_[np.ones(len(a)),a]
def cy(a):return a*a[:,[1,2,3,0]]
def CB(a):return np.c_[np.ones(len(a)),cy(a)]
b=lambda a:aff(a)@np.linalg.lstsq(aff(B['u']),B['v'],rcond=None)[0]
f=lambda a:CB(a)@np.linalg.lstsq(CB(C['u']),C['v1'],rcond=None)[0]
u=C['u']; v=C['v2']; n2=len(u)
def d2(a):return np.c_[aff(a),a*a]
def d3(a):return np.c_[d2(a),a**3]
def d4(a):return np.c_[d3(a),a**4]
for name,dsg in [('d2',d2),('d3',d3),('d4',d4)]:
 A=dsg(u); Q=np.linalg.qr(A,mode='reduced')[0]; e=v-Q@(Q.T@v); print(name,np.std(e,0),np.linalg.lstsq(A,v,rcond=None)[0].round(4).T)
# Test separable quadratic deformation including saturation of the linear arguments.
for fun in (np.tanh,np.sin,np.arctan):
 def ff(a,k):return np.c_[np.ones(len(a)),fun(k*a)/k,a*a]
 def loss(k):
  A=ff(u,k); return np.mean((v-A@np.linalg.lstsq(A,v,rcond=None)[0])**2)
 r=minimize_scalar(loss,bounds=(.001,2),method='bounded');print(fun.__name__,r.x,np.sqrt(r.fun))
g=lambda a:d3(a)@np.linalg.lstsq(d3(u),v,rcond=None)[0]
folds=[np.arange(52,72),*np.array_split(np.random.default_rng(39).permutation(72),6)]
def score(A):
 err=[]
 for j in folds:
  i=np.setdiff1d(np.arange(len(x)),j); c=np.linalg.lstsq(A[i],Z[i],rcond=1e-8)[0]; p=A[j]@c; err.append([np.mean(abs(p[:,0]-y[j])),np.mean(abs(p[:,1:]@V[:2]-T[j]))])
 return np.array(err)
base=np.c_[d2(x),cy(x)]
D={'x':x,'b':b(x),'f':f(x),'g':g(x),'gb':g(b(x)),'gf':g(f(x)),'fg':f(g(x)),'fb':f(b(x))}
candidates=[('base',base)]
for k,a in D.items():
 candidates.extend([(k,np.c_[base,a]),(k+' product',np.c_[base,a,(a[:,:,None]*x[:,None,:]).reshape(len(x),-1)]),(k+' squares',np.c_[base,a,a*a])])
for k1,k2 in [('f','g'),('f','b'),('f','gb'),('f','gf'),('f','fg'),('f','fb'),('f','x')]:
 a,c=D[k1],D[k2]; candidates.extend([(k1+k2,np.c_[base,a,c,a*c]),(k1+k2+' allprod',np.c_[base,a,c,(a[:,:,None]*c[:,None,:]).reshape(len(x),-1)])])
for k,A in candidates:
 E=score(A); print(k,np.linalg.matrix_rank(A), 'tail',E[0].round(5),'worst',E.max(0).round(5))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe6.py
```
- **Container Exit Code**: `0` (`822 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2.299e-01  4.220e-02 -1.400e-03 -6.000e-04 -1.000e-04
   5.000e-04  1.090e-02  4.500e-03  2.840e-02 -5.500e-03]]
tanh 0.05966644496774586 0.004099673896533439
sin 0.08435436476158066 0.004099671609575809
arctan 0.059676217277772246 0.0040996754234542
base 13 tail [0.13025 0.11737] worst [0.15417 0.12753]
x 13 tail [0.13025 0.11737] worst [0.15417 0.12753]
x product 15 tail [0.15119 0.12405] worst [0.15119 0.12563]
x squares 13 tail [0.13025 0.11737] worst [0.15417 0.12753]
b 13 tail [0.13025 0.11737] worst [0.15417 0.12753]
b product 15 tail [0.15119 0.12405] worst [0.15119 0.12563]
b squares 15 tail [0.15119 0.12405] worst [0.15119 0.12563]
f 13 tail [0.13025 0.11737] worst [0.15417 0.12753]
f product 25 tail [0.03663 0.02347] worst [0.05426 0.04509]
f squares 17 tail [0.15102 0.1207 ] worst [0.15102 0.13049]
g 17 tail [0.12152 0.12072] worst [0.14524 0.12072]
g product 33 tail [0.06999 0.08974] worst [0.14328 0.13296]
g squares 21 tail [0.15765 0.12531] worst [0.15765 0.12531]
gb 17 tail [0.1074  0.07002] worst [0.14475 0.08323]
gb product 33 tail [0.13112 0.07549] worst [0.13112 0.07927]
gb squares 21 tail [0.12687 0.09388] worst [0.19465 0.14427]
gf 17 tail [0.15112 0.12091] worst [0.15112 0.13139]
gf product 33 tail [0.0557  0.05576] worst [0.11014 0.10455]
gf squares 21 tail [0.19359 0.14596] worst [0.19412 0.14596]
fg 17 tail [0.12946 0.13149] worst [0.13012 0.13149]
fg product 33 tail [0.05728 0.05167] worst [0.12019 0.085  ]
fg squares 21 tail [0.10197 0.1162 ] worst [0.20946 0.14119]
fb 15 tail [0.15119 0.12405] worst [0.15119 0.12563]
fb product 27 tail [0.10477 0.0788 ] worst [0.10477 0.0788 ]
fb squares 19 tail [0.10645 0.09512] worst [0.16626 0.12314]
fg 21 tail [0.01029 0.05127] worst [0.01152 0.05553]
fg allprod 33 tail [0.01516 0.00978] worst [0.01516 0.01309]
fb 17 tail [0.08501 0.09495] worst [0.11507 0.10891]
fb allprod 25 tail [0.03663 0.02347] worst [0.05426 0.04509]
fgb 21 tail [0.09027 0.05851] worst [0.11126 0.06608]
fgb allprod 33 tail [0.05974 0.04501] worst [0.11085 0.0663 ]
fgf 21 tail [0.17447 0.13798] worst [0.17447 0.14308]
fgf allprod 33 tail [0.13451 0.11111] worst [0.19992 0.19327]
ffg 21 tail [0.11422 0.13269] worst [0.12935 0.13269]
ffg allprod 33 tail [0.15248 0.11997] worst [0.16464 0.19888]
ffb 19 tail [0.17129 0.1382 ] worst [0.19428 0.16611]
ffb allprod 31 tail [0.1657  0.12522] worst [0.17483 0.13284]
fx 17 tail [0.07766 0.07238] worst [0.10929 0.08703]
fx allprod 25 tail [0.03663 0.02347] worst [0.05426 0.04509]
```

### Turn 7 — `execute_in_remote_docker` (`36063 ms` LLM)
- **Rationale**: Expand the successful panel-informed composition into cyclic pair products modulated by linear and squared coordinates. Fit regularization inside each fold to determine whether this basis transfers without panel coefficients; use the result to choose the sink implementation.
- **Alternatives Weighed**: `Use the fitted Panel C maps in the sink` (rejected: The frozen node does not pass the panels or their coefficients to the sink.)

```bash
sed -n '55,180p' /holdout_task/test_dag_candidate.py
cat > /tmp/probe7.py <<'PY'
from p import *
from scipy.linalg import solve
from scipy.optimize import minimize
from itertools import product
edges=[(0,1),(1,2),(2,3),(3,0)]
base=[(),*( (i,) for i in range(4)),*((i,i) for i in range(4)),*edges]
third=sorted({tuple(sorted((*e,i))) for e,i in product(edges,range(4))})
fourth=[tuple(sorted((*e,i,i))) for e,i in product(edges,range(4))]
full2=[c for k in range(3) for c in combinations_with_replacement(range(4),k)]

def features(a,terms):return np.column_stack([np.prod(a[:,k],axis=1) for k in terms])

def fit(F,Y,Fp,groups,opt=True):
 sc=np.sqrt(np.mean(F**2,axis=0)); F=F/sc; Fp=Fp/sc
 size=len(F); I=np.eye(size); mats=[F[:,g]@F[:,g].T for g in groups]; cross=[Fp[:,g]@F[:,g].T for g in groups]
 def objective(h):
  w=np.exp(h); M=sum(j*k for j,k in zip(w,mats))+I; Q=solve(M,I,assume_a='pos'); a=Q@Y; D=Q.diagonal()[:,None]; E=a/D
  J=[]
  for v,K in zip(w,mats):
   qk=Q@K; d=-(qk@a)/D+E*(np.sum(qk*Q,axis=1)[:,None]/D); J.append(2*v*np.mean(d*E))
  return np.mean(E*E),np.array(J)
 h=minimize(objective,np.full(len(groups),4.),jac=True,method='L-BFGS-B',bounds=[(-5,12)]*len(groups),options={'ftol':1e-11}).x if opt else np.full(len(groups),16.)
 w=np.exp(h); a=solve(sum(j*k for j,k in zip(w,mats))+I,Y,assume_a='pos'); C=np.concatenate([]) if False else (F.T@a)/sc[:,None];
 for g,v in zip(groups,w): C[g]*=v
 return sum(j*k for j,k in zip(w,cross))@a,C,h

if __name__=='__main__':
 folds=[np.arange(52,72),*np.array_split(np.random.default_rng(39).permutation(72),6)]
 for terms,ids in [(base+third+fourth,[1,5,9,13,25]),(full2+third+fourth,[1,5,15,27]),(full2+third+fourth,[1,5,15,27,31,35,39]),(base+third,[1,5,9,13])]:
  A=features(x,terms); G=np.split(np.arange(len(terms)),ids); errs=[]
  for jj in folds:
   ii=np.setdiff1d(np.arange(len(x)),jj); pred=[]
   for sl in (slice(0,1),slice(1,None)):
    yh,_,h=fit(A[ii],Y[ii,sl],A[jj],G);pred.append(yh)
   errs.append([np.mean(abs(pred[0]-Y[jj,:1])),np.mean(abs(pred[1]-T[jj]))]);
  print('groups',ids,'folds',np.round(errs,5));
  if len(terms)==41:
   _,C,h=fit(A,Y,A,G); print('coefficients y, first, last',list(zip(terms,np.round(C[:,[0,1,-1]],3).tolist())))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe7.py
```
- **Container Exit Code**: `0` (`1135 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
xceeded:{ym:.6f}")
    if tm > 0.038: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")

ok = len(errs) == 0 and all(v == 1 for v in ec.values())
print(json.dumps({"accepted": ok, "errors": errs, "cv_metrics": cv_metrics, "exec_counts": ec}, allow_nan=False))
sys.exit(0 if ok else 1)
groups [1, 5, 9, 13, 25] folds [[0.0132  0.01509]
 [0.01307 0.01672]
 [0.01768 0.01321]
 [0.02056 0.0131 ]
 [0.01242 0.01399]
 [0.01132 0.01284]
 [0.01305 0.00813]]
coefficients y, first, last [((), [-0.066, -0.017, -0.123]), ((0,), [0.052, 0.118, 0.689]), ((1,), [0.011, 0.086, 0.5]), ((2,), [-0.432, 0.003, 0.036]), ((3,), [-0.337, 0.027, 0.142]), ((0, 0), [0.017, 0.055, 0.314]), ((1, 1), [0.01, 0.039, 0.232]), ((2, 2), [-0.205, 0.003, 0.024]), ((3, 3), [-0.157, 0.015, 0.061]), ((0, 1), [-0.95, 0.255, 0.726]), ((1, 2), [0.61, -0.415, -1.111]), ((2, 3), [-0.415, 0.156, 0.411]), ((3, 0), [0.805, 0.296, 0.783]), ((0, 0, 1), [0.236, 0.081, 0.197]), ((0, 0, 3), [0.009, -0.001, -0.001]), ((0, 1, 1), [-0.003, 0.002, -0.0]), ((0, 1, 2), [-0.054, -0.018, -0.054]), ((0, 1, 3), [-0.28, -0.093, -0.215]), ((0, 2, 3), [-0.038, -0.012, -0.042]), ((0, 3, 3), [0.059, 0.016, 0.041]), ((1, 1, 2), [0.005, 0.012, 0.091]), ((1, 2, 2), [-0.193, -0.072, -0.177]), ((1, 2, 3), [-0.051, -0.038, -0.239]), ((2, 2, 3), [-0.146, -0.035, -0.078]), ((2, 3, 3), [0.173, 0.065, 0.22]), ((0, 0, 0, 1), [0.093, 0.034, 0.074]), ((0, 1, 1, 1), [-0.004, 0.019, 0.012]), ((0, 1, 2, 2), [-0.008, -0.01, -0.007]), ((0, 1, 3, 3), [-0.022, -0.016, -0.043]), ((0, 0, 1, 2), [-0.015, -0.017, -0.019]), ((1, 1, 1, 2), [0.019, 0.016, 0.063]), ((1, 2, 2, 2), [-0.08, -0.017, -0.098]), ((1, 2, 3, 3), [-0.077, -0.013, -0.09]), ((0, 0, 2, 3), [0.014, 0.003, 0.026]), ((1, 1, 2, 3), [0.042, -0.012, -0.064]), ((2, 2, 2, 3), [-0.058, -0.01, -0.027]), ((2, 3, 3, 3), [0.075, 0.017, 0.074]), ((0, 0, 0, 3), [0.009, -0.003, 0.002]), ((0, 1, 1, 3), [-0.11, -0.046, -0.094]), ((0, 2, 2, 3), [-0.036, -0.001, -0.013]), ((0, 3, 3, 3), [0.036, -0.004, 0.018])]
groups [1, 5, 15, 27] folds [[0.01657 0.01623]
 [0.015   0.0218 ]
 [0.01648 0.02098]
 [0.02    0.01296]
 [0.01553 0.01588]
 [0.01177 0.01278]
 [0.01632 0.00892]]
groups [1, 5, 15, 27, 31, 35, 39] folds [[0.01679 0.02619]
 [0.01591 0.02006]
 [0.02418 0.02969]
 [0.02146 0.0156 ]
 [0.0135  0.01921]
 [0.01125 0.01414]
 [0.0185  0.01367]]
groups [1, 5, 9, 13] folds [[0.03676 0.02326]
 [0.04047 0.03409]
 [0.0494  0.04555]
 [0.0465  0.03057]
 [0.05229 0.03097]
 [0.04839 0.0287 ]
 [0.03463 0.01982]]
```

### Turn 8 — `execute_in_remote_docker` (`43170 ms` LLM)
- **Rationale**: Implement the best supported basis without changing the frozen node. Verify the public contract, then measure whether the margin holds across additional splits, individual trajectory horizons, and the highest-radius inputs.
- **Alternatives Weighed**: `Use all quadratic cross terms` (rejected: They gave higher public-tail and worst-fold trajectory error than the cyclic basis.)
- **ATMS NOGOOD DECLARED**: Refuted `Cyclic pair products modulated only by linear coordinates suffice.` | Witness: `Adding squared-coordinate modulation reduced worst-fold response and trajectory MAEs from 0.05229 and 0.04555 to 0.02056 and 0.01672 on the same folds.` | Blanket Diagnosis: `The cubic basis omits the even modulation of cyclic pair products.` | Sublated Pivot: `Fit the cyclic quartic interaction block directly from calibrated coordinates, with shrinkage selected inside each fold.`

```bash
cat > /holdout_task/node_01.py <<'PY'
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
_CUBIC = sorted({tuple(sorted((i, j, k)))
                 for i, j in _EDGES for k in range(4)})


def _features(z):
    pairs = np.column_stack([z[:, i] * z[:, j] for i, j in _EDGES])
    cubic = np.column_stack([z[:, i] * z[:, j] * z[:, k]
                             for i, j, k in _CUBIC])
    squares = z**2
    quartic = (pairs[:, :, None] * squares[:, None, :]).reshape(len(z), 16)
    return np.column_stack((np.ones(len(z)), z, squares, pairs, cubic, quartic))


def _predict(train, targets, test):
    # Give each interaction order its own shrinkage, selected by exact
    # leave-one-out residuals on the training examples only.
    scale = np.sqrt(np.mean(train**2, axis=0))
    blocks = np.split(train / scale, [1, 5, 9, 13, 25], axis=1)
    test_blocks = np.split(test / scale, [1, 5, 9, 13, 25], axis=1)
    grams = [b @ b.T for b in blocks]
    identity = np.eye(len(train))

    def factor(weights):
        return cho_factor(identity + sum(w * k for w, k in zip(weights, grams)))

    def objective(log_weights):
        weights = np.exp(log_weights)
        inverse = cho_solve(factor(weights), identity)
        dual = inverse @ targets
        diagonal = inverse.diagonal()[:, None]
        residual = dual / diagonal
        gradient = []
        for weight, gram in zip(weights, grams):
            product = inverse @ gram
            deriv = (-product @ dual + residual * np.sum(product * inverse, axis=1)[:, None]) / diagonal
            gradient.append(2 * weight * np.mean(residual * deriv))
        return np.mean(residual**2), np.array(gradient)

    result = minimize(objective, np.full(len(blocks), 4.0), jac=True,
                      method='L-BFGS-B', bounds=[(-5, 12)] * len(blocks),
                      options={'ftol': 1e-11})
    weights = np.exp(result.x)
    dual = cho_solve(factor(weights), targets)
    return sum(w * (q @ b.T) for w, q, b in zip(weights, test_blocks, blocks)) @ dual


def run_node(inputs: dict) -> dict:
    data = inputs['node_00']
    n = data['n_train']
    features = _features(data['z'])
    train, test = features[:n], features[n:]
    response = _predict(train, data['y'].reshape(n, 1), test)[:, 0]
    trajectory = _predict(train, data['traj'], test)
    final_state = trajectory[:, -1]
    return {'y_hat': response, 'state_pred': trajectory,
            'diagnostics_vec': np.array([response.mean(), response.std(),
                                         final_state.mean(), final_state.std()])}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
cat > /tmp/stress_readout.py <<'PY'
import runpy
import pickle
import numpy as np
base = '/holdout_task/'
calibrate = runpy.run_path(base + '_frozen_nodes/node_00.py')['run_node']
run = runpy.run_path(base + 'node_01.py')['run_node']
with open(base + 'public_batch.pkl', 'rb') as file:
    study = pickle.load(file)[0]['study']
d = calibrate({'__input__': study})
n = d['n_train']; x = d['z'][:n]; y = d['y']; traj = d['traj']
rad = np.linalg.norm(x, axis=1)
slices = {'tail': [np.arange(n-20, n)],
          'random': [j for seed in (39, 101, 211) for j in np.array_split(np.random.default_rng(seed).permutation(n), 6)],
          'outer-radius': [np.argsort(rad)[-12:]]}
for name, folds in slices.items():
    errors = []
    for j in folds:
        i = np.setdiff1d(np.arange(n), j)
        out = run({'node_00': dict(d, z=x[np.r_[i,j]], n_train=len(i), y=y[i], traj=traj[i])})
        delta = np.abs(out['state_pred'] - traj[j]); p = out['y_hat']; q = out['state_pred'][:, -1]
        np.testing.assert_allclose(out['diagnostics_vec'], [p.mean(), p.std(), q.mean(), q.std()])
        errors.append([np.mean(np.abs(p-y[j])), delta.mean(), *delta.mean(0)])
    print(name, 'worst response / aggregate trajectory / each horizon:', np.max(errors, axis=0).round(6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/stress_readout.py
```
- **Container Exit Code**: `0` (`1235 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.0132, "public_cv_traj_mae": 0.015095}, "exec_counts": {"node_00": 1, "node_01": 1}}
tail worst response / aggregate trajectory / each horizon: [0.0132   0.015095 0.010262 0.008987 0.013728 0.01628  0.01594  0.017202
 0.018742 0.019616]
random worst response / aggregate trajectory / each horizon: [0.03216  0.020821 0.014447 0.01636  0.020932 0.02604  0.023938 0.025571
 0.028737 0.029746]
outer-radius worst response / aggregate trajectory / each horizon: [0.037807 0.035815 0.014595 0.020948 0.030852 0.034192 0.036429 0.046794
 0.051591 0.051123]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1ab656eb975cf1ca (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1ab656eb (directory).
```

### Turn 9 — `execute_in_remote_docker` (`31006 ms` LLM)
- **Rationale**: Compare simple alternatives on the same tail, random, and outer-radius splits without editing the verified artifact. Keep the current model unless an alternative materially improves the outer-radius horizon error while preserving the response and trajectory margins.
- **Alternatives Weighed**: `Submit the verified candidate now` (rejected: The official test passes, but one targeted comparison may improve the late-horizon error on the highest-radius rows without sacrificing the public margin.)

```bash
cat > /tmp/edge_stress.py <<'PY'
import pickle, runpy
import numpy as np
from scipy.linalg import solve
root='/holdout_task/'
model=runpy.run_path(root+'node_01.py'); features=model['_features']; predict=model['_predict']
with open(root+'public_batch.pkl','rb') as f: study=pickle.load(f)[0]['study']
d=runpy.run_path(root+'_frozen_nodes/node_00.py')['run_node']({'__input__':study})
n=d['n_train']; x=d['z'][:n]; y=d['y'][:,None]; T=d['traj']; A=features(x)
folds=[np.arange(n-20,n), *[j for seed in (39,101,211) for j in np.array_split(np.random.default_rng(seed).permutation(n),6)], np.argsort(np.linalg.norm(x,axis=1))[-12:]]
def trial(tag, fit):
    E=[]
    for j in folds:
        i=np.setdiff1d(np.arange(n),j)
        p=fit(i,j,y); q=fit(i,j,T); e=np.abs(q-T[j])
        E.append([np.mean(np.abs(p-y[j])),e.mean(),e.mean(0).max()])
    E=np.array(E); print(tag,'tail',E[0].round(5),'random',E[1:-1].max(0).round(5),'outer',E[-1].round(5))
trial('current',lambda i,j,v:predict(A[i],v[i],A[j]))
for add_name,extras in [('none',None),('diagonal quartics',x**4)]:
    B=A if extras is None else np.c_[A,extras]
    for strength in (0,0.0001,0.001,0.01):
        def fit(i,j,v):
            scale=np.sqrt(np.mean(B[i]**2,axis=0)); P=B[i]/scale
            if strength==0: co=np.linalg.lstsq(P,v[i],rcond=None)[0]
            else: co=solve(P.T@P+strength*len(i)*np.eye(len(scale)),P.T@v[i],assume_a='pos')
            return B[j]/scale@co
        trial((add_name,strength),fit)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/edge_stress.py
```
- **Container Exit Code**: `0` (`850 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
current tail [0.0132  0.01509 0.01962] random [0.03216 0.02082 0.02975] outer [0.03781 0.03582 0.05159]
('none', 0) tail [0.01485 0.02127 0.03159] random [0.02866 0.02272 0.03519] outer [0.04185 0.03488 0.05064]
('none', 0.0001) tail [0.01546 0.02173 0.03164] random [0.0269  0.02219 0.03549] outer [0.04115 0.03243 0.04725]
('none', 0.001) tail [0.02759 0.02697 0.03472] random [0.0386  0.03063 0.04127] outer [0.04637 0.04153 0.06148]
('none', 0.01) tail [0.06086 0.05981 0.07728] random [0.13147 0.10025 0.12929] outer [0.21258 0.16036 0.18463]
('diagonal quartics', 0) tail [0.01875 0.01629 0.0278 ] random [0.03017 0.01901 0.02765] outer [0.03345 0.02725 0.04164]
('diagonal quartics', 0.0001) tail [0.01684 0.01738 0.02439] random [0.02472 0.01836 0.02508] outer [0.03105 0.02707 0.04313]
('diagonal quartics', 0.001) tail [0.03238 0.03958 0.05361] random [0.04342 0.04289 0.05641] outer [0.04452 0.04536 0.06531]
('diagonal quartics', 0.01) tail [0.05925 0.07378 0.09653] random [0.12848 0.12037 0.15344] outer [0.2258  0.17778 0.20988]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1ab656eb975cf1ca (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1ab656eb (directory).
```

### Turn 10 — `execute_in_remote_docker` (`27459 ms` LLM)
- **Rationale**: Promote the simpler challenger because it improves both worst-slice aggregate metrics while keeping the public tail well below tolerance in the probe. Verify the exact saved artifact in isolated Python to catch any difference between the probe and the deployed implementation.
- **Alternatives Weighed**: `Keep grouped cyclic quartic shrinkage` (rejected: It scores slightly better on the public tail but leaves materially worse response and trajectory errors on the outer-radius slice.); `Use diagonal quartics without ridge` (rejected: Its outer-radius worst horizon is slightly better, but mild ridge improves outer-radius response and the worst random-fold errors for both outputs.)

```bash
cat > /holdout_task/node_01.py <<'PY'
import numpy as np
from scipy.linalg import solve


_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
_CUBIC = sorted({tuple(sorted((i, j, k)))
                 for i, j in _EDGES for k in range(4)})


def _features(z):
    pairs = np.column_stack([z[:, i] * z[:, j] for i, j in _EDGES])
    cubic = np.column_stack([z[:, i] * z[:, j] * z[:, k]
                             for i, j, k in _CUBIC])
    squares = z**2
    modulated_pairs = (pairs[:, :, None] * squares[:, None, :]).reshape(len(z), 16)
    return np.column_stack((np.ones(len(z)), z, squares, pairs, cubic,
                            modulated_pairs, squares**2))


def run_node(inputs: dict) -> dict:
    data = inputs['node_00']
    n = data['n_train']
    features = _features(data['z'])
    scale = np.sqrt(np.mean(features[:n]**2, axis=0))
    train = features[:n] / scale
    targets = np.column_stack((data['y'], data['traj']))
    # Fit all horizons on the same normalized basis with mild shrinkage.
    gram = train.T @ train + (1e-4 * n) * np.eye(train.shape[1])
    coefficients = solve(gram, train.T @ targets, assume_a='pos')
    prediction = (features[n:] / scale) @ coefficients
    response, trajectory = prediction[:, 0], prediction[:, 1:]
    final_state = trajectory[:, -1]
    return {'y_hat': response, 'state_pred': trajectory,
            'diagnostics_vec': np.array([response.mean(), response.std(),
                                         final_state.mean(), final_state.std()])}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py && OPENBLAS_NUM_THREADS=1 python3 -I /tmp/stress_readout.py
```
- **Container Exit Code**: `0` (`1036 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.016841, "public_cv_traj_mae": 0.017378}, "exec_counts": {"node_00": 1, "node_01": 1}}
tail worst response / aggregate trajectory / each horizon: [0.016841 0.017378 0.012367 0.020106 0.01045  0.010085 0.014892 0.024112
 0.02439  0.02262 ]
random worst response / aggregate trajectory / each horizon: [0.024724 0.01836  0.015149 0.023175 0.019013 0.022344 0.022802 0.020141
 0.025077 0.024692]
outer-radius worst response / aggregate trajectory / each horizon: [0.03105  0.027072 0.013283 0.043126 0.022382 0.019331 0.023411 0.02938
 0.041626 0.024041]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=91c8346cc04d31ba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_91c8346c (directory).
```

### Turn 11 — `submit_completed_task` (`15003 ms` LLM)
- **Rationale**: The saved artifact passed isolated Python verification, including the output contract and one execution per node. Both aggregate errors stayed below 0.038 across the stress slices; the higher individual-horizon error is not an acceptance criterion.
- **Alternatives Weighed**: `Retain grouped cyclic quartic shrinkage` (rejected: It scored slightly better on the public split but had higher worst-slice errors for both required outputs.)
- **Submitted Final Answer / Summary**: `Verified `/holdout_task/node_01.py`; the frozen node was unchanged. The public test passed with response MAE 0.016841 and trajectory MAE 0.017378, both below 0.038.`

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
