# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m101_b03_batch_06__blind_campaign__merge_nodes`

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
| **Turns Used / Max Turns** | `7 / 12` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `170,422` (`162,765` / `38,528` / `7,657`) |
| **Total Episode Wall Time (s)** | `267.26s` |

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

### Turn 1 — `execute_in_remote_docker` (`15089 ms` LLM)
- **Rationale**: Inspect the frozen consumer, fold scoring, and observed arrays before fitting so that key mismatches, leakage, and unsupported basis terms can be avoided.

```bash
cd /holdout_task && cat _dag_spec.json _frozen_nodes/node_02.py test_dag_candidate.py validate_public_schema.py public_asset_index.json && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust IRLS affine calibration'; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_batch.pkl','rb')); print(type(x));
def p(x,n=''):
 if isinstance(x,dict):
  for k,v in x.items():p(v,n+'/'+str(k))
 elif isinstance(x,np.ndarray):print(n,x.shape,x.dtype,np.round(x[:3],5))
 elif isinstance(x,(list,tuple)):print(n,type(x),len(x)); [p(a,n+'/'+str(i)) for i,a in enumerate(x[:2])]
 else:print(n,str(x)[:120])
p(x)
PY
```
- **Container Exit Code**: `0` (`480 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
<class 'list'>
 <class 'list'> 1
/0/study/eval_times (8,) float64 [0.25    0.64286 1.03571]
/0/study/panel_a/u (48, 4) float64 [[ 0.36914 -0.87519  1.0679  -0.80911]
 [ 0.99624  0.92564  0.83713 -0.74846]
 [-1.07516  1.16808 -0.66874 -1.29331]]
/0/study/panel_a/v (48, 4) float64 [[ 0.30483 -0.70387  0.97188 -0.91781]
 [ 1.07303  0.81693  0.73559 -0.85551]
 [-1.46405  1.02177 -0.80719 -1.41483]]
/0/study/panel_b/u (24, 4) float64 [[ 0.76941 -0.45441 -0.55885  1.06859]
 [-0.91762 -1.19567  0.90537  0.83829]
 [-0.27325 -0.30866  0.97059  0.58558]]
/0/study/panel_b/v (24, 4) float64 [[-0.95621  0.57063 -0.19816  0.42812]
 [ 0.92927  1.38757  0.13188  0.42712]
 [ 0.67601  0.72903 -0.52608  0.19617]]
/0/study/panel_c/u (24, 4) float64 [[ 0.43048  0.83364 -0.89161 -0.05867]
 [ 0.51828 -0.58448 -1.10339 -0.54919]
 [-0.46343  0.43539  0.11961 -0.24192]]
/0/study/panel_c/v1 (24, 4) float64 [[-0.69676  0.17776  0.34908 -0.65499]
 [-0.37177 -0.86704 -0.07961  0.98277]
 [-0.05619 -0.00819 -0.06446  0.2541 ]]
/0/study/panel_c/v2 (24, 4) float64 [[-0.87315  0.63098  0.566    0.69661]
 [-0.37569 -0.30458  0.52507  0.85352]
 [-1.04472  0.13038 -0.23853 -0.88711]]
/0/study/train/x (72, 4) float64 [[-0.73478  0.08921 -0.17639  1.0583 ]
 [ 0.15735  0.34992  0.55408  0.12187]
 [ 0.1095  -0.83254  0.52697  0.62621]]
/0/study/train/y (72,) float64 [-0.25539 -0.53572 -0.28151]
/0/study/train/traj (72, 8) float64 [[0.19367 0.43145 0.6426  0.80676 0.94496 1.06407 1.16456 1.25891]
 [0.32969 0.72833 1.04621 1.29412 1.50178 1.67822 1.8226  1.95076]
 [0.38136 0.7468  0.96263 1.08958 1.17573 1.23304 1.27878 1.30974]]
/0/study/train/summary (4,) float64 [-0.25798  0.92342  0.3471 ]
/0/study/test/x (36, 4) float64 [[-1.51891 -0.82777  0.30539 -0.15694]
 [ 1.15138 -0.88694 -0.61003 -0.19796]
 [-0.88849 -0.52155 -1.20562 -1.18507]]
/0/study/telemetry_vec (4,) float64 [0.1 0.2 0.2]
```

### Turn 2 — `execute_in_remote_docker` (`28416 ms` LLM)
- **Rationale**: Panel residuals will identify the v1 basis. Temporal fit residuals will show whether a shared relaxation law is worth modeling before selecting features inside the public folds.
- **Alternatives Weighed**: `Fit an unrestricted polynomial across panels` (rejected: Compare compact nonlinear bases first; the panels have only 24 rows.)

```bash
cd /holdout_task && sed -n '54,180p' test_dag_candidate.py && cat validate_public_schema.py; python3 - <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; aff=lambda x:np.c_[np.ones(len(x)),x]
for p,k in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[p]['u']; v=s[p][k]; adj=u*np.roll(u,-1,axis=1); pair=np.column_stack([u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]); specs={'u':u,'adj':adj,'u+adj':np.c_[u,adj],'pairs':pair,'u+pairs':np.c_[u,pair],'cos':np.cos(2.4*u),'sin':np.sin(2.4*u),'quadratic':np.c_[u,pair,u*u]}; print(p,k,{n:round(np.mean(abs(aff(a)@np.linalg.lstsq(aff(a),v,rcond=None)[0]-v)),6) for n,a in specs.items()})
u=s['panel_a']['u']; v=s['panel_a']['v']; X=aff(u); b=np.linalg.lstsq(X,v,rcond=None)[0]; print('A residual norms',np.round(np.sort(np.linalg.norm(v-X@b,axis=1)),4));
t=s['eval_times']; Y=s['train']['traj']; print('times',t,'traj range',Y.min(0),Y.max(0),'summary',s['train']['summary'],'telemetry',s['telemetry_vec']); print('trajectories, y, successive difference ratios'); print(np.round(np.c_[Y[:10],s['train']['y'][:10],(np.diff(Y,axis=1)[:,1:]/np.diff(Y,axis=1)[:,:-1])[:10]],4));
for name,fun,q in [('relax',lambda p:p[0]*(-np.expm1(-np.exp(p[1])*t)),[1,0]),('relax+linear',lambda p:p[0]*(-np.expm1(-np.exp(p[1])*t))+p[2]*t,[1,0,0]),('relax+offset',lambda p:p[0]*(-np.expm1(-np.exp(p[1])*t))+p[2],[1,0,0]),('relax+sin',lambda p:p[0]*(-np.expm1(-np.exp(p[1])*t))+p[2]*np.sin(t),[1,0,0]),('tanh',lambda p:p[0]*np.tanh(np.exp(p[1])*t),[1,0]),('rational',lambda p:p[0]*t/(1+np.exp(p[1])*t),[1,0])]:
 fits=[least_squares(lambda p:fun(p)-r,q,max_nfev=150,bounds=([-100,-8]+[-100]*(len(q)-2),[100,4]+[100]*(len(q)-2))) for r in Y]; ea=np.array([np.abs(f.fun).mean() for f in fits]); print(name,ea.mean(),ea.max(),np.round([f.x for f in fits[:6]],4))
PY
```
- **Container Exit Code**: `0` (`2318 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
e ratios
[[ 0.1937  0.4315  0.6426  0.8068  0.945   1.0641  1.1646  1.2589 -0.2554
   0.888   0.7775  0.8418  0.8619  0.8436  0.9389]
 [ 0.3297  0.7283  1.0462  1.2941  1.5018  1.6782  1.8226  1.9508 -0.5357
   0.7974  0.7799  0.8377  0.8496  0.8184  0.8876]
 [ 0.3814  0.7468  0.9626  1.0896  1.1757  1.233   1.2788  1.3097 -0.2815
   0.5906  0.5882  0.6786  0.6652  0.7983  0.6766]
 [-0.2084 -0.4471 -0.5908 -0.6974 -0.7841 -0.8491 -0.8975 -0.9455 -0.0584
   0.6022  0.7414  0.8134  0.7505  0.7434  0.9928]
 [ 0.3446  0.7604  1.0544  1.2864  1.46    1.604   1.7335  1.839  -0.9876
   0.7071  0.7892  0.7483  0.8296  0.8992  0.8147]
 [-0.1351 -0.3568 -0.5966 -0.8076 -0.9843 -1.1443 -1.2923 -1.4061 -0.3197
   1.0823  0.8797  0.8372  0.9057  0.9251  0.7685]
 [-0.2544 -0.5899 -0.8797 -1.1061 -1.2953 -1.4527 -1.597  -1.7175 -2.3084
   0.8639  0.7814  0.8355  0.832   0.9166  0.8354]
 [ 0.478   1.0365  1.4286  1.7461  1.9947  2.1924  2.3567  2.489   0.3556
   0.702   0.8097  0.7832  0.7952  0.8307  0.8054]
 [ 0.1193  0.334   0.568   0.7968  0.99    1.1554  1.3065  1.425   0.4902
   1.0906  0.9773  0.8447  0.8563  0.9131  0.7844]
 [ 0.5259  1.0472  1.3813  1.5937  1.7439  1.8618  1.9435  2.0102 -0.8656
   0.6409  0.6357  0.7073  0.785   0.6927  0.8173]]
relax 0.022294119720485957 0.15755171337251223 [[ 1.6555 -0.7572]
 [ 2.3902 -0.5934]
 [ 1.3194  0.2422]
 [-1.0032 -0.1453]
 [ 2.0893 -0.3901]
 [-2.9662 -1.5236]]
relax+linear 0.004175681399578145 0.021331435822125336 [[  0.9723  -0.3624   0.1353]
 [  1.3701  -0.1468   0.2285]
 [  1.1062   0.4506   0.0722]
 [ -0.658    0.2792  -0.1003]
 [  1.2575   0.0592   0.2117]
 [-22.2761  -2.6197   0.9843]]
relax+offset 0.01192673290674008 0.05208791783286549 [[ 1.6834 -0.8198  0.019 ]
 [ 2.4269 -0.681   0.0475]
 [ 1.2718  0.1535  0.0653]
 [-0.9892 -0.2586 -0.0435]
 [ 2.0969 -0.4815  0.056 ]
 [-2.5541 -1.257   0.0518]]
relax+sin 0.01025380325971991 0.07854151245172289 [[ 1.56470e+00 -5.99500e-01 -4.57000e-02]
 [ 2.27900e+00 -4.13500e-01 -9.72000e-02]
 [ 1.34430e+00  3.11500e-01 -6.06000e-02]
 [-1.00390e+00  1.50000e-02  6.67000e-02]
 [ 2.03520e+00 -2.11200e-01 -1.12700e-01]
 [-4.91081e+01 -4.64290e+00 -1.29300e-01]]
tanh 0.0386188105910861 0.16497554969212175 [[ 1.3892 -0.7524]
 [ 2.0775 -0.6442]
 [ 1.2712  0.0021]
 [-0.9343 -0.3174]
 [ 1.8816 -0.5007]
 [-1.9608 -1.1969]]
rational 0.016536585581996357 0.1575592095331966 [[ 0.8214 -1.1391]
 [ 1.4234 -0.9234]
 [ 2.1192  0.2369]
 [-1.0068 -0.3078]
 [ 1.569  -0.6497]
 [-0.6533 -2.0826]]
```

### Turn 3 — `execute_in_remote_docker` (`44305 ms` LLM)
- **Rationale**: Check the three calibration outliers, recover compact panel bases, and score coordinate/readout interactions on the published split and three other row blocks. This separates panel misspecification from trajectory prediction error.
- **Alternatives Weighed**: `Predict independently fitted relaxation parameters` (rejected: Some rows have poorly identified amplitude and rate; compare direct trajectory prediction before relying on those parameters.)

```bash
cat > /tmp/probe.py <<'PY'
import pickle
import numpy as np
from itertools import combinations
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']

def aff(x): return np.column_stack((np.ones(len(x)),x))
def ls(a,b): return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def adj(z): return z*np.roll(z,-1,axis=1)
def pair(z): return np.column_stack([z[:,i]*z[:,j] for i,j in combinations(range(z.shape[1]),2)])

def panels(s,x):
    u,v=(s['panel_a'][k] for k in ('u','v')); A=aff(u); b=ls(A,v)
    for _ in range(12):
        r=np.linalg.norm(v-A@b,axis=1); w=(1+(r/max(3*np.median(r),1e-6))**2)**-.5
        b=ls(w[:,None]*A,w[:,None]*v)
    r=np.linalg.norm(v-A@b,axis=1); mask=r<max(8*np.median(r),1e-4); b=ls(A[mask],v[mask]); z=np.linalg.solve(b[1:].T,(x-b[0]).T).T
    u=s['panel_c']['u']; c=aff(adj(z))@ls(aff(adj(u)),s['panel_c']['v1']); d=aff(np.cos(2.4*z))@ls(aff(np.cos(2.4*u)),s['panel_c']['v2']); return z,c,d,r

def bases(z):
    return {'linear':z,'adj':adj(z),'linear+adj':np.c_[z,adj(z)],'pair':pair(z),'quadratic':np.c_[z,pair(z),z*z],'cos':np.cos(2.4*z),'sin':np.sin(2.4*z),'tanh':np.tanh(z),'cube':z**3,'triple':np.column_stack([np.prod(z[:,ix],axis=1) for ix in combinations(range(4),3)])}

def scan_panel(s):
    for p,k in [('panel_b','v'),('panel_c','v1')]:
        u=s[p]['u']; y=s[p][k]; bb=bases(u); bb.update({f'lin+{n}':np.c_[u,f] for n,f in list(bb.items()) if n in ('cos','sin','tanh','cube','triple')}); print(p,k,[(n,round(np.mean(abs(aff(f)@ls(aff(f),y)-y)),7)) for n,f in bb.items()]);
        for freq in [0.75,1,1.2,1.5,1.7,1.8,2,2.2,2.4,2.6,3]:
            print(freq,end=':');
            for f in [np.sin(freq*u), np.cos(freq*u),np.tanh(freq*u),np.c_[np.sin(freq*u),np.cos(freq*u)]]:
                a=aff(f); print('%.5f'%np.mean(abs(a@ls(a,y)-y)),end=' ')
            print()

def cv(f,y,lam=0):
    f=np.asarray(f); errs=[]
    for i in [np.arange(52,72),np.arange(0,20),np.arange(20,40),np.arange(40,52)]:
        j=np.setdiff1d(np.arange(72),i); a=f[j]; m=a.mean(0); sd=np.maximum(a.std(0),1e-8); a=aff((a-m)/sd); at=aff((f[i]-m)/sd)
        if lam:
            d=np.eye(a.shape[1])*lam; d[0,0]=0; v=np.linalg.solve(a.T@a+d,a.T@y[j]);
        else: v=ls(a,y[j]);
        errs.append(np.mean(abs(at@v-y[i]),axis=0))
    return np.array(errs)

if __name__=='__main__':
    scan_panel(S)
    X=S['train']['x']; Y=S['train']['y']; T=S['train']['traj']; z,c,d,r=panels(S,X); print('A robust norms',np.sort(r)[[0,22,44,45,46,47]]); print('T svd',np.linalg.svd(T,compute_uv=False));
    fd=bases(z); fd.update({'raw':X,'c':c,'d':d,'cd':np.c_[c,d], 'zcd':np.c_[z,c,d], 'zcd+z*c':np.c_[z,c,d,z*c], 'zcd+z*d':np.c_[z,c,d,z*d], 'zcd+c*d':np.c_[z,c,d,c*d], 'zcd+all3':np.c_[z,c,d,z*c,z*d,c*d,z*c*d], 'zcd+outerzc':np.c_[z,c,d,(z[:,:,None]*c[:,None,:]).reshape(len(z),-1)], 'zcd+pairz':np.c_[z,c,d,pair(z),z*z], 'zcd+polycd':np.c_[z,c,d,z*z,c*c,d*d,pair(c),pair(d)], 'z+triple':np.c_[z,adj(z),np.roll(z,1,axis=1)*adj(z)]});
    for name,f in fd.items():
        ey=cv(f,Y); et=cv(f,T); print(name,f.shape[1], 'y',np.round(ey,4),'traj',np.round(et.mean(1),4),'traj last',np.round(et[:,-1],4));
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'exponential relaxation variable projection' | head -5; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_robust_network_assimilation.assimilate_diffusive_network | head -15
```
- **Container Exit Code**: `0` (`526 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.8777 1.037  0.7841] traj last [1.1841 1.2435 1.4719 1.0725]
tanh 4 y [0.6997 0.5614 0.8155 1.0868] traj [0.8806 0.9266 1.0443 0.7519] traj last [1.222  1.3172 1.4922 1.0513]
cube 4 y [0.6725 0.5915 0.8114 1.0435] traj [0.9009 1.0226 1.03   0.7794] traj last [1.2563 1.4492 1.4917 1.1198]
triple 4 y [0.5052 0.5729 0.7917 1.1084] traj [0.9604 1.0606 1.3336 0.733 ] traj last [1.3426 1.4908 1.8887 1.0389]
raw 4 y [0.7025 0.5656 0.8127 1.0798] traj [0.8878 0.9413 1.0453 0.7509] traj last [1.2342 1.3381 1.4972 1.0644]
c 4 y [0.3617 0.5992 0.8729 0.8235] traj [0.7369 0.8066 0.7568 0.6888] traj last [1.0839 1.198  1.1056 0.9918]
d 4 y [0.3603 0.345  0.5472 0.5477] traj [0.462  0.4101 0.5016 0.3669] traj last [0.5526 0.4975 0.6019 0.4297]
cd 8 y [0.188  0.1248 0.2905 0.2506] traj [0.1332 0.1405 0.2237 0.204 ] traj last [0.1592 0.1826 0.27   0.2501]
zcd 12 y [0.1895 0.1196 0.2586 0.2123] traj [0.1403 0.1459 0.2006 0.1745] traj last [0.1721 0.1926 0.2442 0.2154]
zcd+z*c 16 y [0.1956 0.1826 0.3042 0.2627] traj [0.1601 0.1802 0.2412 0.2257] traj last [0.2002 0.2361 0.2952 0.2791]
zcd+z*d 16 y [0.2001 0.1511 0.2993 0.1802] traj [0.1571 0.1477 0.2525 0.1656] traj last [0.1937 0.1931 0.3137 0.2093]
zcd+c*d 16 y [0.0048 0.005  0.0056 0.0045] traj [0.0524 0.0865 0.0742 0.0417] traj last [0.0785 0.1282 0.1094 0.0597]
zcd+all3 28 y [0.006  0.0067 0.0053 0.0079] traj [0.0913 0.1184 0.1255 0.0609] traj last [0.1375 0.1742 0.1858 0.0877]
zcd+outerzc 28 y [0.2341 0.2792 0.4676 0.217 ] traj [0.2244 0.2063 0.4808 0.2086] traj last [0.2856 0.2558 0.6207 0.2662]
zcd+pairz 22 y [0.173  0.1578 0.2684 0.185 ] traj [0.1278 0.1451 0.2217 0.1581] traj last [0.1568 0.1881 0.2768 0.2006]
zcd+polycd 36 y [0.1643 0.2731 0.314  0.3061] traj [0.1714 0.2089 0.2832 0.2379] traj last [0.232  0.2645 0.3551 0.2863]
z+triple 12 y [0.5879 0.5815 0.8645 0.9003] traj [0.7972 0.7    1.0555 0.7054] traj last [1.18   1.0678 1.6033 1.0332]
No matching skill cards found.
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
```

### Turn 4 — `execute_in_remote_docker` (`45177 ms` LLM)
- **Rationale**: Compare same-coordinate and cross-coordinate products, including those in Panel B's affine frame. Fit interaction selection inside each held-out block to distinguish a reproducible missing term from an in-sample fit.
- **Alternatives Weighed**: `Fit a separate relaxation rate for each row` (rejected: The trajectory matrix has only two singular values above the noise floor, so test shared time shapes and distinct spatial drivers first.)
- **ATMS NOGOOD DECLARED**: Refuted `An additive sink on calibrated coordinates and panel readouts is sufficient.` | Witness: `Adding the elementwise v1×v2 interaction reduced scalar MAE from 0.1895 to 0.0048 on the public split; the same basis still gave trajectory MAE 0.0524.` | Blanket Diagnosis: `The scalar output requires a cross-readout term, and the trajectory retains structure not explained by the scalar basis.` | Sublated Pivot: `Keep v1×v2 for y and search for the trajectory interaction against its measured rank-two time structure.`

```bash
cat > /tmp/traj_probe.py <<'PY'
from probe import *
from scipy.linalg import qr
from scipy.optimize import minimize_scalar
X=S['train']['x']; y=S['train']['y']; T=S['train']['traj']; t=S['eval_times']; z,c,d,r=panels(S,X)
u=S['panel_b']['u']; b=aff(z)@ls(aff(u),S['panel_b']['v'])
_,sing,V=np.linalg.svd(T,full_matrices=False)
print('temporal vectors',np.round(V[:2],5)); print('rank-2 MAE',np.mean(abs(T-T@V[:2].T@V[:2])))
for second in ['t','exp2','exp_ratio2','t_exp','sin','one']:
 def cols(k):
  g=-np.expm1(-k*t)
  h={'t':t,'exp2':-np.expm1(-2*k*t),'exp_ratio2':-np.expm1(-.5*k*t),'t_exp':t*np.exp(-k*t),'sin':np.sin(t),'one':np.ones_like(t)}[second]
  return np.c_[g,h]
 def cost(k):
  A=cols(k); return np.mean((A@ls(A,T.T)-T.T)**2)**.5
 opt=minimize_scalar(cost,bounds=(.02,4),method='bounded'); print('time',second,round(opt.x,6),round(opt.fun,7))
base=np.c_[z,c,d,c*d]
for q,w in [('z',z),('b',b),('c',c),('d',d),('cd',c*d),('x',X)]:
 for p,v in [('z',z),('b',b),('c',c),('d',d),('cd',c*d),('x',X)]:
  if q>p: continue
  for typ,f in [('same',w*v),('outer',(w[:,:,None]*v[:,None,:]).reshape(len(z),-1))]:
   A=np.c_[base,f]; err=cv(A,T); e=np.round(err.mean(1),4); tr=np.mean(abs(aff(A)@ls(aff(A),T)-T));
   print(q,p,typ,'rank',np.linalg.matrix_rank(aff(A),tol=1.e-8),'cv',e,'fit',round(tr,5))
# Pursuit on residualized candidate interactions, with selection repeated independently inside each fold.
F={}
VARS={'z':z,'b':b,'c':c,'d':d,'x':X,'cd':c*d}
for a,A in VARS.items():
 for bname,B in VARS.items():
  if a>bname:continue
  for i in range(4):
   for j in range(4):
    if a!=bname or j>=i:F[f'{a}{i}*{bname}{j}']=A[:,i]*B[:,j]
for n,A in VARS.items():
 for kind,W in [('tanh',np.tanh(A)),('sat',A/(1+A*A)),('cube',A**3),('sin',np.sin(A)),('cos',np.cos(A))]:
  for j in range(4):F[f'{kind}({n}{j})']=W[:,j]
for A,n in [(base,'base'),(np.c_[base,z*c],'base+zc')]:
 fm=np.column_stack(list(F.values())); keys=list(F)
 for i in [np.arange(52,72),np.arange(0,20),np.arange(20,40),np.arange(40,52),np.array([],dtype=int)]:
  j=np.setdiff1d(np.arange(len(T)),i); W=aff(A); R=fm[j]-W[j]@ls(W[j],fm[j]); scales=np.linalg.norm(R,axis=0); valid=scales>1e-7; active=[]; scores=[]
  for step in range(10):
   res=T[j]-W[j]@ls(W[j],T[j]); cur=fm[j]-W[j]@ls(W[j],fm[j]); norms=np.linalg.norm(cur,axis=0); good=valid & (norms>1e-7); score=np.where(good,np.linalg.norm(cur.T@res,axis=1)/np.maximum(norms,1e-7),-1); k=int(np.argmax(score)); active.append(keys[k]); valid[k]=False; W=np.c_[W,fm[:,k]]; pred=W@ls(W[j],T[j]); scores.append(round(np.mean(abs(pred[i if len(i) else j]-T[i if len(i) else j])),5))
  print('pursuit',n,'fold',i[0] if len(i) else 'fit',scores,active)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/traj_probe.py
```
- **Container Exit Code**: `0` (`698 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
same rank 21 cv [0.0664 0.0931 0.0922 0.0547] fit 0.04642
cd x outer rank 33 cv [0.1275 0.1085 0.1997 0.0823] fit 0.04214
x z same rank 21 cv [0.0593 0.1129 0.0987 0.0519] fit 0.04825
x z outer rank 23 cv [0.0585 0.1148 0.1005 0.0571] fit 0.04819
x x same rank 21 cv [0.0593 0.1129 0.0987 0.0519] fit 0.04825
x x outer rank 23 cv [0.0585 0.1148 0.1005 0.0571] fit 0.04819
pursuit base fold 52 [0.00472, 0.00471, 0.00497, 0.00495, 0.00528, 0.00558, 0.00598, 0.00592, 0.00621, 0.00645] ['c0*d1', 'cos(x0)', 'b3*cd2', 'cube(cd3)', 'cd3*d3', 'b0*c3', 'b0*cd0', 'b0*c0', 'cd2*d3', 'c2*cd3']
pursuit base fold 0 [0.00507, 0.00534, 0.00591, 0.00593, 0.00579, 0.00626, 0.00657, 0.00691, 0.00743, 0.00757] ['c0*d1', 'z0*z0', 'cd0*x2', 'b0*d1', 'b1*cd3', 'c3*d2', 'cd0*cd2', 'z3*z3', 'cos(z3)', 'cd0*d2']
pursuit base fold 20 [0.00644, 0.00649, 0.0072, 0.00742, 0.00771, 0.00865, 0.00902, 0.01025, 0.01033, 0.01048] ['c0*d1', 'z0*z0', 'cd0*z3', 'cd3*d0', 'b1*c0', 'c3*cd0', 'cd0*cd0', 'cd3*d3', 'x1*x1', 'sat(z2)']
pursuit base fold 40 [0.00493, 0.00487, 0.00504, 0.00515, 0.00512, 0.00544, 0.00541, 0.0054, 0.00554, 0.00561] ['c0*d1', 'b3*cd2', 'd3*x3', 'cos(z1)', 'd1*x3', 'cd2*cd2', 'c2*cd0', 'c2*x2', 'cd3*z3', 'b0*cd0']
pursuit base fold fit [0.00356, 0.00346, 0.00337, 0.00327, 0.00319, 0.00314, 0.00305, 0.00301, 0.00294, 0.00287] ['c0*d1', 'z0*z0', 'b3*cd2', 'cd1*cd3', 'cos(z1)', 'b1*cd0', 'cd3*z0', 'b0*c3', 'tanh(cd0)', 'd1*x3']
pursuit base+zc fold 52 [0.00527, 0.00573, 0.00561, 0.00567, 0.00575, 0.00605, 0.0062, 0.00658, 0.00693, 0.00728] ['c0*d1', 'cos(x0)', 'cd1*cd3', 'b1*cd0', 'cd0*d1', 'cd2*d3', 'b0*z2', 'b1*cd1', 'c1*d2', 'c2*cd0']
pursuit base+zc fold 0 [0.00574, 0.00609, 0.0067, 0.00715, 0.00758, 0.00793, 0.00821, 0.00867, 0.00876, 0.0091] ['c0*d1', 'cd2*d1', 'cd0*x2', 'cos(cd1)', 'b3*cd2', 'cd1*d0', 'c2*cd1', 'd0*z0', 'cd1*d2', 'c2*cd0']
pursuit base+zc fold 20 [0.0068, 0.00746, 0.00961, 0.01009, 0.01087, 0.01172, 0.01238, 0.02113, 0.02119, 0.02127] ['c0*d1', 'd2*d2', 'c2*cd3', 'b2*d1', 'cos(cd0)', 'cd1*cd3', 'cd3*x3', 'cd0*cd0', 'cos(d2)', 'sat(x0)']
pursuit base+zc fold 40 [0.00534, 0.00567, 0.00577, 0.00558, 0.00594, 0.00684, 0.00763, 0.00814, 0.00855, 0.00856] ['c0*d1', 'c1*d3', 'cd3*x3', 'b3*cd2', 'd1*x1', 'c1*d2', 'c1*z2', 'b1*cd1', 'cd0*x1', 'sat(b2)']
pursuit base+zc fold fit [0.0034, 0.00331, 0.0032, 0.0031, 0.00301, 0.00293, 0.00284, 0.00276, 0.00267, 0.00257] ['c0*d1', 'x0*x0', 'c1*d3', 'b2*d1', 'b0*cd2', 'b1*cd1', 'c3*cd1', 'b1*cd0', 'cd3*z3', 'tanh(b2)']
```

### Turn 5 — `execute_in_remote_docker` (`73123 ms` LLM)
- **Rationale**: Implement the sparse interaction selected consistently in all four probe folds, without fixing its index from validation rows. Verify the exact artifacts on the public contract, then check whether the interaction remains predictive across row blocks, extrapolation to larger input norms, and every trajectory horizon.
- **Alternatives Weighed**: `Fit all cross-coordinate products` (rejected: The full outer-product basis had worse worst-block trajectory MAE (0.0084 versus 0.00644) than selecting one term.)

```bash
set -e
cd /holdout_task
cat > node_00.py <<'PY'
import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, y):
    return np.linalg.lstsq(a, y, rcond=1e-10)[0]


def _adjacent(x):
    return x * np.roll(x, -1, axis=1)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    panel_a, panel_c = study['panel_a'], study['panel_c']
    a = _affine(panel_a['u'])
    v = panel_a['v']
    beta = _fit(a, v)
    # Estimate the forward map before inversion so response outliers do not
    # become high-leverage calibration inputs.
    for _ in range(12):
        residual = np.linalg.norm(v - a @ beta, axis=1)
        scale = max(3 * np.median(residual), 1e-6)
        root_weight = (1 + (residual / scale) ** 2) ** -0.5
        beta = _fit(a * root_weight[:, None], v * root_weight[:, None])
    residual = np.linalg.norm(v - a @ beta, axis=1)
    inlier = residual < max(8 * np.median(residual), 1e-4)
    beta = _fit(a[inlier], v[inlier])

    x = np.concatenate((study['train']['x'], study['test']['x']))
    z = np.linalg.solve(beta[1:].T, (x - beta[0]).T).T
    u = panel_c['u']
    c = _affine(_adjacent(z)) @ _fit(_affine(_adjacent(u)), panel_c['v1'])
    return {'u': u, 'z': z, 'v2': panel_c['v2'], 'c': c}
PY
cat > node_03.py <<'PY'
import numpy as np


def _fit(a, y):
    return np.linalg.lstsq(a, y, rcond=1e-10)[0]


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    data = inputs['node_00']
    z, c, d = data['z'], data['c'], inputs['node_02']['d']
    n = len(train['x'])
    base = np.column_stack((np.ones(len(z)), z, c, d, c * d))
    design = base[:n]
    y_hat = base[n:] @ _fit(design, train['y'])

    # The trajectory needs a cross-coordinate readout product as well.
    # Select it from training residuals after removing the baseline span.
    i, j = np.where(~np.eye(c.shape[1], dtype=bool))
    products = c[:, i] * d[:, j]
    orth = products[:n] - design @ _fit(design, products[:n])
    traj = train['traj']
    residual = traj - design @ _fit(design, traj)
    norms = np.linalg.norm(orth, axis=0)
    gain = np.linalg.norm(orth.T @ residual, axis=1) / np.maximum(norms, 1e-12)
    trajectory_design = np.column_stack((base, products[:, np.argmax(gain)]))
    state_pred = trajectory_design[n:] @ _fit(trajectory_design[:n], traj)
    final = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, runpy
import numpy as np
h = runpy.run_path('/holdout_task/test_dag_candidate.py')
s, run = h['st'], h['run_dag']
train = s['train']
x, y, tr = (train[k] for k in ('x', 'y', 'traj'))
# Row blocks test interaction selection with changed training rows; the
# radial tail checks extrapolation where cross-products are largest.
folds = {'public': np.arange(52,72), 'first': np.arange(20), 'middle': np.arange(20,40), 'late': np.arange(40,52), 'radial_tail': np.argsort(np.linalg.norm(x, axis=1))[-16:]}
for label, val in folds.items():
    keep = np.setdiff1d(np.arange(len(x)), val)
    case = copy.deepcopy(s)
    case['train'] = dict(x=x[keep], y=y[keep], traj=tr[keep], summary=np.array([y[keep].mean(), y[keep].std(), tr[keep,-1].mean(), tr[keep,-1].std()]))
    case['test'] = {'x': x[val]}
    pred, counts = run(case)
    p, q = pred['y_hat'], pred['state_pred']
    a = float(np.abs(p-y[val]).mean())
    b = float(np.abs(q-tr[val]).mean())
    c = float(np.abs(q-tr[val]).mean(axis=0).max())
    np.testing.assert_allclose(pred['diagnostics_vec'], [p.mean(),p.std(),q[:,-1].mean(),q[:,-1].std()])
    print(f'{label}: y={a:.6f} trajectory={b:.6f} worst_horizon={c:.6f}')
    assert set(counts.values()) == {1} and max(a,b,c) < 0.038
PY
```
- **Container Exit Code**: `0` (`615 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py', '/holdout_task/node_03.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.004821, "public_cv_traj_mae": 0.004723}, "exec_counts": {"node_00": 1, "node_02": 1, "node_03": 1}}
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.004821, "public_cv_traj_mae": 0.004723}, "exec_counts": {"node_00": 1, "node_02": 1, "node_03": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=786de900a04608f0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_786de900 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`26155 ms` LLM)
- **Rationale**: Run only the missing stress checks. A failure would distinguish sensitivity to training rows from high-norm extrapolation or a weak trajectory horizon and guide any targeted revision.
- **Alternatives Weighed**: `Submit on the public acceptance receipt alone` (rejected: The prior output did not show the intended row-block, radial-tail, or per-horizon checks on the exact artifacts.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy
import pickle
from pathlib import Path
import numpy as np
p = Path('/holdout_task/test_dag_candidate.py')
ns = {'__file__': str(p), '__name__': '__stress__'}
# Load the DAG runner without invoking the public script's top-level exit.
exec(compile(p.read_text().split('\nst = pickle.load', 1)[0], str(p), 'exec'), ns)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
x, y, tr = (s['train'][k] for k in ('x', 'y', 'traj'))
rows = np.arange(len(x))
folds = {'public': rows[52:], 'first': rows[:20], 'middle': rows[20:40], 'late': rows[40:52], 'radial_tail': np.argsort(np.linalg.norm(x, axis=1))[-16:]}
for label, val in folds.items():
    keep = np.setdiff1d(rows, val)
    case = copy.deepcopy(s)
    last = tr[keep, -1]
    case['train'] = dict(x=x[keep], y=y[keep], traj=tr[keep], summary=np.array([y[keep].mean(), y[keep].std(), last.mean(), last.std()]))
    case['test'] = {'x': x[val]}
    pred, counts = ns['run_dag'](case)
    a, b, d = (pred[k] for k in ('y_hat', 'state_pred', 'diagnostics_vec'))
    assert a.shape == y[val].shape and b.shape == tr[val].shape and d.shape == (4,)
    assert all(np.isfinite(v).all() for v in (a, b, d))
    np.testing.assert_allclose(d, [a.mean(), a.std(), b[:, -1].mean(), b[:, -1].std()])
    ey, et = np.abs(a-y[val]).mean(), np.abs(b-tr[val]).mean(axis=0)
    print(f'{label}: y_mae={ey:.6f} traj_mae={et.mean():.6f} worst_horizon_mae={et.max():.6f}', flush=True)
    assert set(counts.values()) == {1} and max(ey, et.max()) < 0.038
PY
```
- **Container Exit Code**: `0` (`463 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public: y_mae=0.004821 traj_mae=0.004723 worst_horizon_mae=0.006048
first: y_mae=0.004972 traj_mae=0.005066 worst_horizon_mae=0.006804
middle: y_mae=0.005561 traj_mae=0.006441 worst_horizon_mae=0.008170
late: y_mae=0.004518 traj_mae=0.004932 worst_horizon_mae=0.006605
radial_tail: y_mae=0.006347 traj_mae=0.008450 worst_horizon_mae=0.012758

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=786de900a04608f0 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_786de900 (directory).
```

### Turn 7 — `submit_completed_task` (`13826 ms` LLM)
- **Rationale**: The unchanged, receipt-bound bundle passed isolated public verification and all row-block, radial-tail, and per-horizon stress checks. The cross-coordinate term resolves the observed trajectory error without expanding to the full outer-product basis.
- **Submitted Final Answer / Summary**: `Implemented and verified `node_00.py` and `node_03.py`, leaving the frozen node unchanged. The public test accepted the DAG, with each node executing once. Public MAE was 0.004821 for the scalar output and 0.004723 for the trajectory. The worst stress-test horizon MAE was 0.012758.`

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
