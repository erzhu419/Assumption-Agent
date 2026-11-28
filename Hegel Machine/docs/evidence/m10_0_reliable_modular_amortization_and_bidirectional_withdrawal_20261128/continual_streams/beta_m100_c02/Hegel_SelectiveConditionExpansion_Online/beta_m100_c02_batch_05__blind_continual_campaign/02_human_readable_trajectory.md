# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m100_c02_batch_05__blind_continual_campaign__upstream_transducer`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World` |
| **Domain** | `coupled_2d_transport_reaction_box_simplex_kkt` |
| **Wave / Wormhole Family** | `m10_0_reliable_modular_amortization_and_bidirectional_withdrawal` / `m100_beta` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `11 / 16` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `310,590` (`297,133` / `101,504` / `13,457`) |
| **Total Episode Wall Time (s)** | `406.34s` |

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

### Turn 1 — `execute_in_remote_docker` (`16548 ms` LLM)
- **Rationale**: Inspect the downstream contract and calibration ranges first: field or shape mismatches change the transformation, while coverage of the observation range determines how to test inverse extrapolation.

```bash
cd /holdout_task && ls -l && cat batch_solver.py validate_public_schema.py _verified_downstream.py && ls -lh /tmp/hegel_continual_archive && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS multichannel calibration' ; cd /holdout_task && python3 - <<'PY'
import pickle, numpy as np
s=pickle.load(open('public_batch.pkl','rb'))
def show(x,p='',d=0):
 if isinstance(x,dict):
  print(p,list(x));
  if d<4:
   for k,v in x.items():show(v,p+'/'+str(k),d+1)
 elif isinstance(x,np.ndarray):
  print(p,x.shape,x.dtype, (np.nanmin(x),np.nanmax(x)) if x.size and x.dtype.kind in 'fiu' else '')
 elif isinstance(x,(list,tuple)):
  print(p,type(x).__name__,len(x));
  if len(x) and d<4:show(x[0],p+'[0]',d+1)
 else: print(p,repr(x)[:150])
show(s)
PY
```
- **Container Exit Code**: `0` (`456 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
darray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
 list 1
[0] ['study', 'campaign_diagnostics']
[0]/study ['Lx', 'Ly', 'nx', 'ny', 'kappa_x', 'kappa_y', 'vel_x', 'vel_y', 'robin_x', 'robin_y', 'V_reaction_2d', 'mu_shifts', 'source_modes', 'sensor_ij', 'row_bank_id', 'sensor_calibration_sweeps', 'Z_nuisance', 'y_obs_raw', 'aux_manifold_telemetry', 'nuisance_ridge', 'source_ridge', 'u_box', 'total_budget']
[0]/study/Lx 2.4
[0]/study/Ly 2.0
[0]/study/nx 14
[0]/study/ny 12
[0]/study/kappa_x (14,) float64 (0.47110686395408286, 0.6100861063278105)
[0]/study/kappa_y (12,) float64 (0.33938047029228036, 0.5793804702922803)
[0]/study/vel_x (14,) float64 (-0.20572483830236554, 0.35)
[0]/study/vel_y (12,) float64 (-0.279885813991893, -0.0)
[0]/study/robin_x 1.0141673808964704
[0]/study/robin_y 1.1585219318384303
[0]/study/V_reaction_2d (14, 12) float64 (0.25, 0.42738476356834465)
[0]/study/mu_shifts (3,) float64 (0.15, 1.1)
[0]/study/source_modes (8, 14, 12) float64 (2.706275892017887e-16, 0.9973180100514493)
[0]/study/sensor_ij (32, 2) int64 (0, 12)
[0]/study/row_bank_id (96,) int64 (0, 3)
[0]/study/sensor_calibration_sweeps ['ref_rosette', 'raw_rosette', 'single_bank_sweeps']
[0]/study/sensor_calibration_sweeps/ref_rosette (80, 4) float64 (-1.3758364485901176, 1.7625417061689732)
[0]/study/sensor_calibration_sweeps/raw_rosette (80, 4) float64 (-13.931281701244536, 13.420011422118368)
[0]/study/sensor_calibration_sweeps/single_bank_sweeps ['bank_id', 'ref_signal', 'raw_readout']
[0]/study/Z_nuisance (96, 3) float64 (-3.109177905602093, 2.9556966152831534)
[0]/study/y_obs_raw (96,) float64 (-0.5454049079874831, 0.8987605778053133)
[0]/study/aux_manifold_telemetry (24, 4) float64 (-2.740265872183093, 2.3753586078734883)
[0]/study/nuisance_ridge 0.08
[0]/study/source_ridge 0.04
[0]/study/u_box 0.42
[0]/study/total_budget 1.4
[0]/campaign_diagnostics None
```

### Turn 2 — `execute_in_remote_docker` (`34180 ms` LLM)
- **Rationale**: Compare affine, pairwise, and saturation terms on held-out sweep rows. Inspect the coefficient patterns to determine whether the nonlinearity is channel-local or coupled before choosing an inverse.
- **Alternatives Weighed**: `Fit a flexible inverse directly on raw sweep readings` (rejected: Gross spikes corrupt the inverse predictors; fitting the forward map keeps the reference coordinates clean and isolates spikes in the residuals.)

```bash
cd /holdout_task; cat public_asset_index.json; cat /tmp/hegel_continual_archive/{archive_manifest,pre_llm_gate_diagnostic}.json; grep -nEi 'rosette|saturat|transduc|emi' /tmp/hegel_continual_archive/archive_apparatus_log.json | cut -c1-350 | head -35; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_robust_network_assimilation.assimilate_diffusive_network | head -30; cat >/tmp/rosette_probe.py <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; n=len(x)
np.set_printoptions(precision=6,suppress=True)
for k,v in c['single_bank_sweeps'].items(): print(k,np.shape(v),np.asarray(v).ravel()[:16])
def design(z,key):
    a=[np.ones(len(z)),z]; t=z*z
    terms={'affine':[], 'square':[t], 'cube':[z*t], 'diagpoly':[t,z*t], 'poly2':[np.column_stack([z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)])], 'tanh':[np.tanh(z)-z], 'rat3':[z*t/(1+t)], 'rat2':[t/(1+t)], 'sin':[np.sin(z)-z], 'softsign':[z/(1+abs(z))-z], 'absquad':[z*abs(z)]}
    return np.column_stack(a+terms[key])
def robust(a,b):
    m=np.linalg.lstsq(a,b,rcond=None)[0]
    for scale in [0.3,.1,.03,.01,.003,.001]:
        for _ in range(15):
            r=np.linalg.norm(a@m-b,axis=1); w=1/np.hypot(1,r/scale); m=np.linalg.lstsq(a*w[:,None],b*w[:,None],rcond=None)[0]
    return m
base=design(x,'affine'); m=robust(base,y); r=y-base@m; rr=np.linalg.norm(r,axis=1); print('affine',m,'res quantiles',np.quantile(rr,[0,.5,.8,.9,.95,1])); print('largest',np.sort(rr)[-14:]); print('example clean x,y,res',np.c_[x,y,r][np.argsort(rr)[[5,10,20,30,40,50,60,68,69,70]]]); print('ref correlation',np.corrcoef(x.T)); print('raw obs quantiles',np.quantile(s['y_obs_raw'].reshape(-1,4),[0,.5,1],axis=0)); print('rows',s['row_bank_id'][:32]); folds=np.random.default_rng(20).permutation(n)%5
for key in ['affine','square','cube','diagpoly','poly2','tanh','rat3','rat2','sin','softsign','absquad']:
    a=design(x,key); o=np.zeros(n); m=robust(a,y)
    for j in range(5):
        v=folds==j; o[v]=np.linalg.norm(a[v]@robust(a[~v],y[~v])-y[v],axis=1)
    print(key,np.quantile(o,[.25,.5,.75,.85,.9]),'clean fold max med',max(np.median(o[(folds==j)&(rr<1)]) for j in range(5))); print(m if key not in ['affine','poly2','diagpoly'] else '')
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rosette_probe.py
```
- **Container Exit Code**: `0` (`899 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
5  0.091501]
 [-0.323887  0.108964  1.291388  0.110239]
 [ 0.041708 -0.181472 -0.132032  1.791532]]
rat3 [0.079714 0.102567 0.149584 0.160096 0.173021] clean fold max med 0.11955404218239268
[[-0.0469    0.126202 -0.030082  0.086291]
 [ 1.372895 -0.020122 -0.210454 -0.009143]
 [-0.036244  2.168671 -0.009869 -0.328017]
 [-0.224126 -0.021636  1.341901 -0.029048]
 [-0.042874 -0.302435 -0.031101  2.148382]
 [-0.917782  0.026692  0.092903  0.043562]
 [ 0.034227 -1.401525  0.049965  0.155975]
 [ 0.133042  0.047848 -0.824755  0.056439]
 [ 0.058437  0.105168  0.026191 -1.387312]]
rat2 [0.292144 0.388865 0.477573 0.52674  0.584544] clean fold max med 0.45728236068341155
[[-0.02951   0.065357 -0.140428 -0.004467]
 [ 0.856692  0.036785 -0.125827 -0.024888]
 [ 0.001272  1.295144 -0.002032 -0.26994 ]
 [-0.201595  0.04513   0.852571 -0.11794 ]
 [ 0.001507 -0.312801 -0.029222  1.37187 ]
 [-0.232119 -0.068739  0.180058  0.138164]
 [-0.082989 -0.058784  0.132218  0.316066]
 [ 0.130984 -0.085994  0.005048  0.330739]
 [ 0.180356  0.090294  0.010757 -0.581919]]
sin [0.12361  0.16785  0.237404 0.307975 0.351746] clean fold max med 0.1971446890405266
[[-0.053583  0.115113 -0.033392  0.112651]
 [ 1.12025   0.067491 -0.205321  0.112944]
 [-0.063131  1.905822 -0.059987 -0.337192]
 [-0.293466  0.067394  1.141638  0.013399]
 [-0.014093 -0.304677 -0.052509  1.768196]
 [ 1.079809  0.231831 -0.155495  0.274879]
 [-0.252116  2.283525 -0.164585 -0.335126]
 [-0.526593  0.086397  1.183278  0.167152]
 [-0.04607  -0.234656 -0.102096  1.525357]]
softsign [0.088183 0.117709 0.152827 0.179526 0.206557] clean fold max med 0.12156369251745386
[[-0.051624  0.122356 -0.034683  0.085487]
 [ 1.676887 -0.102309 -0.219312 -0.035991]
 [-0.067001  2.571056 -0.05222  -0.31823 ]
 [-0.345595 -0.106642  1.698265 -0.049144]
 [-0.092504 -0.439205 -0.025065  2.793353]
 [ 1.549349 -0.171275 -0.121624 -0.092543]
 [-0.097373  2.268095 -0.143504 -0.156484]
 [-0.342408 -0.159917  1.535054 -0.107903]
 [-0.156549 -0.342525 -0.014002  2.66785 ]]
absquad [0.088379 0.1268   0.188536 0.243263 0.278746] clean fold max med 0.16699013037993038
[[-0.043064  0.126211 -0.018622  0.105804]
 [ 1.354358  0.073994 -0.224808  0.061898]
 [-0.029346  2.270058 -0.046619 -0.212242]
 [-0.231767  0.047279  1.363     0.062401]
 [ 0.015037 -0.298645 -0.045793  2.025782]
 [-0.409883 -0.056101  0.047783 -0.038788]
 [ 0.021203 -0.758855  0.045226 -0.032807]
 [ 0.075701 -0.009147 -0.412834 -0.040416]
 [-0.017474  0.048547  0.020228 -0.544374]]
```

### Turn 3 — `execute_in_remote_docker` (`46094 ms` LLM)
- **Rationale**: Compare physically distinct layouts on held-out sweep rows. The result determines which map to invert; an error floor across layouts would instead motivate testing missing coupled terms.
- **Alternatives Weighed**: `Use the fixed-scale rational correction as the final model` (rejected: Its residual is still large, and a good forward fit alone does not establish that the inverse is unique on the observation range.)
- **ATMS NOGOOD DECLARED**: Refuted `A low-degree polynomial or affine mixing model adequately captures the calibration response.` | Witness: `The fixed rational cubic correction reduces the worst-fold median forward error from 0.21343 for cubic terms to 0.11955.` | Blanket Diagnosis: `The calibration residual is structured rather than confined to gross EMI rows; polynomial terms do not capture the observed saturation.` | Sublated Pivot: `Fit the saturation scale inside each fold and compare whether saturation acts before or after linear mixing.`

```bash
cd /holdout_task; python3 - <<'PY'
import pickle,numpy as np,json
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; d=c['single_bank_sweeps']
for k,v in d.items(): print(k,np.shape(v),np.asarray(v).ravel()[:28])
a=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('apparatus keys',list(a)[:12]); print('gate',open('/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json').read())
PY
cat > /tmp/rosette_struct.py <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; n=len(x)
np.set_printoptions(precision=5,suppress=True)
folds=np.random.default_rng(20).permutation(n)%5

def lin(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def robust(a,b):
 m=lin(a,b)
 for scale in [.3,.1,.03,.01]:
  for _ in range(12):
   w=1/np.hypot(1,np.linalg.norm(a@m-b,axis=1)/scale); m=lin(a*w[:,None],b*w[:,None])
 return m

def sat(z,k,kind):
 if kind=='rat': return z/(1+k*z*z)
 if kind=='root': return z/np.sqrt(1+k*z*z)
 if kind=='tanh': return np.tanh(z*np.sqrt(k))/np.sqrt(k)
 if kind=='atan': return np.arctan(z*np.sqrt(k))/np.sqrt(k)
 if kind=='exp': return z*np.exp(-k*z*z)

def fit(x,y,kind,layout):
 af=lambda z: np.column_stack((np.ones(len(z)),z))
 a=af(x); base=robust(a,y); e=np.linalg.norm(a@base-y,axis=1); good=e<max(2,4*np.median(e))
 if layout in ('pre','preplus','prequad'):
  def des(z,k):
   q=sat(z,k,kind)
   return af(q) if layout=='pre' else np.column_stack((af(q),z) if layout=='preplus' else (af(q),z,z*z))
  p=np.full(4,.7)
  for _ in range(3):
   def err(k):
    a=des(x[good],k); return (a@lin(a,y[good])-y[good]).ravel()
   p=least_squares(err,p,bounds=(.003,15),max_nfev=100).x
   a=des(x,p); m=lin(a[good],y[good]); e=np.linalg.norm(a@m-y,axis=1); good=e<max(.025,5*np.median(e))
  return lambda z:des(z,p)@m, p, m
 # Each channel is a nonlinear function of a normalized row of the mixing map.
 p=np.r_[base[0],(base[1:]/np.diag(base[1:]))[~np.eye(4,dtype=bool)],np.diag(base[1:]),np.full(4,.7),np.full(4,0.) if layout=='postplus' else []]
 def pred(z,p):
  mix=np.eye(4); mix[~np.eye(4,dtype=bool)]=p[4:16]; h=z@mix
  return p[:4]+p[16:20]*sat(h,p[20:24],kind)+(p[24:28]*h if layout=='postplus' else 0)
 for _ in range(3):
  b=np.full(len(p),np.inf); low=-b.copy(); low[20:24]=.003; b[20:24]=15
  p=least_squares(lambda p:(pred(x[good],p)-y[good]).ravel(),p,bounds=(low,b),max_nfev=120).x
  e=np.linalg.norm(pred(x,p)-y,axis=1); good=e<max(.025,5*np.median(e))
 return lambda z:pred(z,p),p,None

a=np.column_stack((np.ones(n),x)); base=robust(a,y); e=np.linalg.norm(a@base-y,axis=1); good=e<2
print('EMI rows:',np.flatnonzero(~good), 'affine worst fold med',max(np.median(np.linalg.norm(a[(folds==j)&good]@robust(a[folds!=j],y[folds!=j])-y[(folds==j)&good],axis=1)) for j in range(5)))
for layout in ('pre','preplus','prequad','post','postplus'):
 for kind in ('rat','root','tanh','atan','exp'):
  f,p,m=fit(x,y,kind,layout); r=np.linalg.norm(f(x)-y,axis=1)
  vals=[]
  for j in range(5):
   q=folds==j; ff,_,_=fit(x[~q],y[~q],kind,layout); v=np.linalg.norm(ff(x[q&good])-y[q&good],axis=1); vals.append([np.median(v),np.max(v)])
  print(layout,kind,'fit med',round(np.median(r[good]),7),'CV worst med/max',np.max(vals,axis=0),'params',p if m is None else p.tolist(),flush=True)
  if np.median(r[good])<.03:
   print('coef',m,'clean max',max(r[good]),'residual rows',np.c_[x,y-f(x)][np.argsort(r)[-len(x[~good])-5:-len(x[~good])]],flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rosette_struct.py
```
- **Container Exit Code**: `0` (`2578 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.1807338 CV worst med/max [0.2162  0.56289] params [-0.05418  0.12411 -0.01376  0.10991 -0.01167 -0.17871  0.00963 -0.01025
  0.00206 -0.191   -0.19229 -0.01958  0.00356  0.01713 -0.19975 -0.02315
  1.17105  1.75898  1.1548   1.86917  0.8711   0.62725  0.71568  0.78694]
post atan fit med 0.1708382 CV worst med/max [0.22001 0.54849] params [-0.05523  0.12664 -0.01342  0.10784 -0.01229 -0.17731  0.00992 -0.00917
  0.00459 -0.19214 -0.19126 -0.02098  0.00506  0.01718 -0.19527 -0.02164
  1.22212  1.86738  1.20589  1.94857  1.30877  1.00788  1.07468  1.18068]
post exp fit med 0.189168 CV worst med/max [0.22143 0.56784] params [-0.0542   0.11755 -0.01441  0.1117  -0.01215 -0.18036  0.00946 -0.01246
 -0.00547 -0.18943 -0.19781 -0.01786  0.00008  0.01827 -0.20695 -0.02782
  1.09043  1.62692  1.08176  1.75199  0.16855  0.11693  0.13999  0.15554]
postplus rat fit med 0.1711823 CV worst med/max [0.23758 0.50048] params [-0.05666  0.12265 -0.01309  0.10476 -0.01664 -0.17295  0.01086 -0.0084
  0.0069  -0.19383 -0.19083 -0.0235   0.0062   0.01739 -0.18804 -0.02055
  0.7896   1.2448   0.73727  1.24555  0.82427  1.90338  1.31007  0.78051
  0.46916  0.98854  0.59276  0.76663]
postplus root fit med 0.1688314 CV worst med/max [0.2348  0.50175] params [-0.05661  0.12373 -0.01286  0.10481 -0.01765 -0.17335  0.01057 -0.00826
  0.00602 -0.19365 -0.19081 -0.02402  0.00634  0.01735 -0.18958 -0.02081
  1.00851  1.51475  0.91149  1.57585  1.44411  4.38882  2.41659  1.47051
  0.25699  0.76697  0.4211   0.46111]
postplus tanh fit med 0.1690674 CV worst med/max [0.23641 0.50135] params [-0.05663  0.12323 -0.01304  0.10479 -0.01694 -0.17321  0.01078 -0.00837
  0.0066  -0.19379 -0.19084 -0.02379  0.00622  0.01738 -0.18888 -0.02064
  0.92155  1.42042  0.8499   1.45055  2.16516  5.36975  3.44728  2.0862
  0.33841  0.8175   0.47704  0.56699]
postplus atan fit med 0.1694468 CV worst med/max [0.23581 0.50221] params [-0.05659  0.12399 -0.01276  0.10485 -0.0181  -0.17343  0.01043 -0.0082
  0.00561 -0.19356 -0.1908  -0.02408  0.00642  0.01734 -0.18999 -0.02092
  1.09046  1.60293  0.97073  1.69259  2.14866  7.74821  3.74864  2.28313
  0.17905  0.71415  0.36646  0.35822]
postplus exp fit med 0.1732176 CV worst med/max [0.23009 0.50348] params [-0.05674  0.11861 -0.01412  0.10478 -0.01376 -0.17233  0.01145 -0.00906
  0.00838 -0.19398 -0.19114 -0.02062  0.00589  0.01753 -0.18358 -0.02042
  0.57638  0.96711  0.56874  0.91523  0.85484  1.38012  1.20279  0.75518
  0.65921  1.17398  0.7382   1.04551]
```

### Turn 4 — `execute_in_remote_docker` (`24480 ms` LLM)
- **Rationale**: Compare the best pre-mixing variants and isolate the one-channel response on the axis sweeps. If the axis fits are much tighter than the rosette fits, use the axis law to test which coupling terms account for the remaining error.
- **Alternatives Weighed**: `Add unconstrained four-channel interactions immediately` (rejected: The single-bank sweeps can first distinguish saturation shapes without confounding them with cross-channel coupling.)

```bash
OPENBLAS_NUM_THREADS=1 python3 /tmp/rosette_struct.py | grep -E '^(EMI|pre)' ; cat >/tmp/rosette_axes.py <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps']; d=c['single_bank_sweeps']; np.set_printoptions(precision=5,suppress=True)
def f(z,k,kind):
 if kind=='tanh': return np.tanh(k*z)
 if kind=='root': return k*z/np.sqrt(1+(k*z)**2)
 if kind=='atan': return np.arctan(k*z)
 if kind=='asinh': return np.arcsinh(k*z)
 if kind=='erf':
  from scipy.special import erf
  return erf(k*z)
for kind in ('tanh','root','atan','asinh','erf'):
 for extra in ('none','linear','shift','both'):
  rows=[]
  for j in range(4):
   q=d['bank_id']==j; z=d['ref_signal'][q]; y=d['raw_readout'][q]; ok=np.abs(y)<4
   def pred(p): return p[0]+p[1]*f(z+(p[3] if extra in ('shift','both') else 0),p[2],kind)+(p[-1]*z if extra in ('linear','both') else 0)
   p=least_squares(lambda p:(pred(p)-y)[ok], [0,1,1]+[0]*(extra!='none')+[0]*(extra=='both'),max_nfev=500).x
   rows.append([np.sqrt(np.mean((pred(p)-y)[ok]**2)),*p])
  print(kind,extra,np.array(rows).round(6).tolist())
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rosette_axes.py; grep -nE 'calibr|y_obs|pulse' /holdout_task/_verified_downstream.py
```
- **Container Exit Code**: `0` (`2726 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1.126217, 1.276958, -0.101405], [0.009321, 0.071565, 1.815537, 1.277145, -0.163206], [0.005849, -0.059015, 1.140004, 1.277138, -0.102491], [0.009248, 0.051554, 1.801661, 1.277069, -0.162056]]
atan shift [[0.006944, -0.077837, 0.955058, 1.423787, -0.002779], [0.010787, 0.072439, 1.540653, 1.422712, -0.001499], [0.00677, -0.058464, 0.96738, 1.422725, -0.001502], [0.010705, 0.052411, 1.528713, 1.422721, -0.001491]]
atan both [[0.005813, -0.075152, 1.143604, 1.266358, -0.004948, -0.112373], [0.009064, 0.076629, 1.843673, 1.265953, -0.003729, -0.180548], [0.005688, -0.055833, 1.157687, 1.265937, -0.003732, -0.11339], [0.008994, 0.056569, 1.829525, 1.265901, -0.003721, -0.17923]]
asinh none [[0.016309, -0.081625, 0.509947, 2.865114], [0.025311, 0.068187, 0.822657, 2.862278], [0.015892, -0.061136, 0.516548, 2.862306], [0.025118, 0.048202, 0.81628, 2.862309]]
asinh linear [[0.006775, -0.078913, 1.092646, 1.675546, -0.487751], [0.010464, 0.071765, 1.761501, 1.675736, -0.786167], [0.006567, -0.058889, 1.10608, 1.675721, -0.493661], [0.010382, 0.051753, 1.748065, 1.675621, -0.780242]]
asinh shift [[0.016305, -0.082577, 0.510223, 2.862543, 0.001422], [0.025219, 0.063603, 0.823641, 2.857024, 0.003865], [0.015834, -0.064012, 0.517166, 2.857053, 0.003862], [0.025026, 0.043643, 0.817258, 2.857042, 0.003874]]
asinh both [[0.006503, -0.072113, 1.115534, 1.656848, -0.005781, -0.506339], [0.010154, 0.080367, 1.799463, 1.655534, -0.004318, -0.816067], [0.006372, -0.053485, 1.129933, 1.655507, -0.004321, -0.512447], [0.010076, 0.060273, 1.785682, 1.65545, -0.004311, -0.809867]]
erf none [[0.011403, -0.077366, 1.087291, 1.034213], [0.017684, 0.073776, 1.753394, 1.034163], [0.011102, -0.057627, 1.100963, 1.034168], [0.017544, 0.053747, 1.739803, 1.034173]]
erf linear [[0.003758, -0.07939, 0.7216, 1.32514, 0.240842], [0.005805, 0.070974, 1.163265, 1.325352, 0.388598], [0.003641, -0.059386, 0.73043, 1.325347, 0.243995], [0.005759, 0.050968, 1.154331, 1.325302, 0.38553]]
erf shift [[0.010816, -0.070894, 1.084751, 1.038382, -0.00992], [0.016741, 0.084112, 1.749489, 1.03798, -0.009091], [0.01051, -0.051136, 1.098511, 1.037987, -0.009093], [0.01661, 0.063994, 1.735933, 1.037987, -0.009083]]
erf both [[0.00365, -0.078196, 0.725569, 1.321163, -0.002721, 0.237787], [0.005674, 0.072663, 1.169527, 1.321229, -0.002115, 0.383838], [0.003558, -0.058324, 0.734368, 1.321217, -0.002119, 0.241001], [0.00563, 0.052638, 1.160522, 1.321194, -0.002108, 0.380824]]
92:    y_cal = cal_study['y_obs_raw']
```

### Turn 5 — `execute_in_remote_docker` (`34192 ms` LLM)
- **Rationale**: The nearly identical single-bank shape estimates support testing a shared saturation scale. Compare bounded even terms and pairwise products on held-out rosette rows; the residuals and coefficient patterns will determine which terms belong in the inverse map.

```bash
cat >/tmp/rosette_terms.py <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from itertools import combinations
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; d=c['single_bank_sweeps']; z=d['ref_signal'].reshape(4,-1).T; w=d['raw_readout'].reshape(4,-1).T; qa=abs(w)<4; good=np.max(abs(y),axis=1)<4
pairs=list(combinations(range(4),2)); folds=np.random.default_rng(20).permutation(len(x))%5
np.set_printoptions(precision=4,suppress=True)
print('axis normalized (banks 1-3):', np.c_[z[:,1],(w[:,1:]-w[5,1:])/(w[11,1:]-w[5,1:])]);
func={'tanh':lambda h,k:np.tanh(k*h),'root':lambda h,k:h/np.sqrt(1+k*h*h),'exp':lambda h,k:h*np.exp(-k*h*h),'rat':lambda h,k:h/(1+k*h*h),'erf':lambda h,k:erf(k*h)}
even={'sq':lambda h:h*h,'ratsq':lambda h:h*h/(1+h*h),'expsq':lambda h:-np.expm1(-h*h),'zero':lambda h:np.zeros_like(h)}
for kind,f in func.items():
 for ek,e in even.items():
  def resid(p):
   h=f(z,p[0]); out=[]
   for j in range(4):
    q=qa[:,j]; a=np.column_stack([np.ones(len(z)),h[:,j],z[:,j],e(z[:,j])]); t=np.linalg.lstsq(a[q],w[q,j],rcond=None)[0]; out.extend((a@t-w[:,j])[q]);
   return np.array(out)
  p=least_squares(resid,[1.],bounds=(.05,8)).x; print('axis',kind,ek,round(p[0],5),round(np.sqrt(np.mean(resid(p)**2)),8))

def design(v,k,kind,ek,interaction):
 h=func[kind](v,k); a=[np.ones(len(v)),v,h,even[ek](v)]
 if interaction=='lin': a += [np.column_stack([v[:,i]*v[:,j] for i,j in pairs])]
 if interaction=='sat': a += [np.column_stack([h[:,i]*h[:,j] for i,j in pairs])]
 if interaction=='rational': a += [np.column_stack([v[:,i]*v[:,j]/(1+v[:,i]**2+v[:,j]**2) for i,j in pairs])]
 return np.column_stack(a)
def fit(q,kind,ek,interaction):
 def f(k):
  a=design(x[q],k,kind,ek,interaction); return (a@np.linalg.lstsq(a,y[q],rcond=None)[0]-y[q]).ravel()
 k=least_squares(f,[1.],bounds=(.05,8)).x; a=design(x[q],k,kind,ek,interaction); return k,np.linalg.lstsq(a,y[q],rcond=None)[0]
scores=[]
for kind in func:
 for ek in ['sq','ratsq','expsq']:
  for interaction in ['none','lin','sat','rational']:
   o=[]
   for j in range(5):
    q=good&(folds==j); k,b=fit(good&~q,kind,ek,interaction); r=np.linalg.norm(design(x[q],k,kind,ek,interaction)@b-y[q],axis=1); o.append((np.median(r),max(r)))
   scores.append((max(t[0] for t in o),max(t[1] for t in o),kind,ek,interaction))
for row in sorted(scores)[:14]: print('cv',row)
for _,_,kind,ek,interaction in sorted(scores)[:3]:
 k,b=fit(good,kind,ek,interaction);print('coeff',kind,ek,interaction,k,b)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rosette_terms.py
```
- **Container Exit Code**: `0` (`1452 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, 0.037192941356267745, 'exp', 'ratsq', 'none')
cv (0.016729612598222657, 0.04162409994286581, 'exp', 'ratsq', 'lin')
cv (0.016939907794965153, 0.03847455474963761, 'exp', 'expsq', 'sat')
cv (0.017592695588488153, 0.03783197387944251, 'exp', 'expsq', 'rational')
cv (0.01831882053677806, 0.04139797017982332, 'exp', 'ratsq', 'rational')
cv (0.019479134839254862, 0.04126835939560001, 'exp', 'ratsq', 'sat')
cv (0.029443496799621226, 0.06792946452520615, 'erf', 'expsq', 'lin')
cv (0.029741252025657007, 0.062230619029899396, 'erf', 'expsq', 'none')
cv (0.030471966615145403, 0.07027287207818507, 'erf', 'ratsq', 'lin')
cv (0.03136622215809267, 0.06549022635033258, 'erf', 'ratsq', 'none')
cv (0.031393605951604044, 0.06804632193521991, 'erf', 'expsq', 'sat')
cv (0.03191939363991538, 0.06650632846257114, 'erf', 'expsq', 'rational')
coeff exp expsq none [0.9259] [[-0.0807  0.0709 -0.0602  0.0478]
 [ 0.6442  0.0018 -0.1727 -0.002 ]
 [ 0.0012  1.0379  0.0007 -0.273 ]
 [-0.1697  0.      0.6511 -0.0018]
 [-0.0003 -0.2774 -0.0016  1.0356]
 [ 0.6806 -0.0086  0.0087  0.0039]
 [-0.0025  1.0938  0.0018  0.0143]
 [ 0.0026 -0.0052  0.6941  0.0059]
 [ 0.002   0.013   0.0041  1.0839]
 [-0.0005  0.0039  0.1121  0.005 ]
 [ 0.0012 -0.0025 -0.0016  0.1712]
 [ 0.1056 -0.0064  0.001   0.0006]
 [ 0.0001  0.1745  0.0001  0.0009]]
coeff exp expsq lin [0.9222] [[-0.0797  0.0719 -0.0609  0.0487]
 [ 0.645   0.0017 -0.1726 -0.0006]
 [ 0.0005  1.0359  0.0008 -0.2739]
 [-0.1701  0.0001  0.6503 -0.004 ]
 [-0.0002 -0.2759 -0.0021  1.0332]
 [ 0.6755 -0.0081  0.0078 -0.0005]
 [-0.0021  1.0951  0.0011  0.0148]
 [ 0.0029 -0.004   0.6934  0.009 ]
 [ 0.0003  0.0082  0.0055  1.0843]
 [-0.0016  0.0034  0.1118  0.0037]
 [ 0.0005 -0.0021 -0.0014  0.1684]
 [ 0.1048 -0.0081  0.0022  0.0011]
 [-0.0012  0.1722  0.0012  0.0012]
 [-0.0005  0.0019 -0.0003  0.0006]
 [-0.0016 -0.0012  0.0006 -0.0001]
 [ 0.0031  0.0033 -0.0011  0.0014]
 [ 0.0006  0.0003  0.0002  0.0001]
 [ 0.001  -0.0021  0.0009  0.0027]
 [ 0.0012  0.0007 -0.0005  0.003 ]]
coeff exp ratsq none [0.9255] [[-0.0824  0.0697 -0.0618  0.046 ]
 [ 0.644   0.0017 -0.1733 -0.0018]
 [ 0.0012  1.0377  0.0009 -0.2738]
 [-0.1696  0.001   0.6513 -0.0024]
 [-0.0007 -0.2773 -0.0025  1.035 ]
 [ 0.6815 -0.0096  0.0116  0.0041]
 [-0.0024  1.0931  0.0022  0.0167]
 [ 0.0019 -0.0083  0.6939  0.0062]
 [ 0.0035  0.0125  0.006   1.0851]
 [-0.001   0.0051  0.1467  0.0038]
 [ 0.0012 -0.0042 -0.0027  0.2234]
 [ 0.1377 -0.0117  0.0015 -0.0003]
 [ 0.0004  0.226  -0.0005  0.0025]]
```

### Turn 6 — `execute_in_remote_docker` (`37753 ms` LLM)
- **Rationale**: Fit axis and rosette sweeps jointly with a sparse paired-bank model. If the sparse fit loses held-out accuracy, retain the better-supported dense terms; if the residual persists across odd basis choices, inspect the paired-bank term before adding complexity.
- **Alternatives Weighed**: `Add generic pairwise interactions` (rejected: All three tested interaction bases raised the worst-fold median error, while the fitted coefficients point to coupling between opposite banks.)

```bash
cat >/tmp/rosette_sparse.py <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; d=c['single_bank_sweeps']; z=d['ref_signal'].reshape(4,-1).T; w=d['raw_readout'].reshape(4,-1).T; good=np.max(abs(y),axis=1)<4; qa=abs(w)<4; folds=np.random.default_rng(20).permutation(len(x))%5
odd={'exp':lambda x,k:x*np.exp(-k*x*x), 'rat':lambda x,k:x/(1+k*x*x), 'rat2':lambda x,k:x/(1+k*x*x)**2,'rat3':lambda x,k:x/(1+k*x*x)**3,'tanh':lambda x,k:np.tanh(k*x), 'sin':lambda x,k:np.sin(k*x),'exp4':lambda x,k:x*np.exp(-k*x**4),'root':lambda x,k:x/np.sqrt(1+k*x*x)}
even={'exp':lambda x,k:-np.expm1(-k*x*x),'rat':lambda x,k:x*x/(1+k*x*x),'cos':lambda x,k:1-np.cos(k*x),'tanh':lambda x,k:np.tanh(k*x*x),'sq':lambda x,k:x*x,'soft':lambda x,k:np.log1p(k*x*x)}
def fit(q,od,ev,extra=None):
 def des(v,j,k,axis=False):
  a=v[:,j]; b=v[:,(j+2)%4]; cols=[np.ones(len(v)),a,odd[od](a,k[0])]
  if extra: cols += [odd[extra](a,k[2])]
  cols += [b,even[ev](b,k[1])]; out=np.column_stack(cols)
  if axis:out[:,-2:]=0
  return out
 def solve(k,ret=False):
  rr=[]; ms=[]
  for j in range(4):
   a=np.vstack((des(x[q],j,k),des(z[qa[:,j]],j,k,True))); t=np.r_[y[q,j],w[qa[:,j],j]]; m=np.linalg.lstsq(a,t,rcond=None)[0]; rr.extend(a@m-t); ms.append(m)
  return ms if ret else np.array(rr)
 starts=[(.8,.8,.3),(1.6,1.6,2.0)] if extra else [(.8,.8)]
 sols=[least_squares(solve,p,bounds=(.04,8),max_nfev=160) for p in starts]; k=min(sols,key=lambda r:r.cost).x; ms=solve(k,True)
 return lambda v:np.column_stack([des(v,j,k)@ms[j] for j in range(4)]),k,np.array(ms),np.sqrt(np.mean(solve(k)**2))
rows=[]
for od in odd:
 for ev in even:
  f,k,m,r=fit(good,od,ev); rows.append((r,od,ev,None))
for od,e in [('exp','exp'),('exp','rat'),('exp','tanh'),('exp','sin'),('exp','exp4'),('rat','tanh'),('rat','rat'),('tanh','tanh'),('tanh','sin')]:
 for ev in ('exp','rat','cos'):
  f,k,m,r=fit(good,od,ev,e); rows.append((r,od,ev,e))
for r,od,ev,e in sorted(rows,key=lambda r:r[0])[:12]:
 f,k,m,r=fit(good,od,ev,e); rs=[]
 for j in range(5):
  q=folds==j; g,*_=fit(good&~q,od,ev,e); v=np.linalg.norm(g(x[q&good])-y[q&good],axis=1); rs.append([np.median(v),v.max()])
 print(od,ev,e,'rmse',r,'cv',np.max(rs,axis=0),'k',k,'coeff',m.round(5).tolist(),flush=True)
print('pulses:',[(p.keys(),np.shape(p.get('y_pulse_raw'))) for p in (s.get('reference_tracer_pulses') or [])])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rosette_sparse.py; cat /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`6093 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
124299.58899, -0.26883, 0.07678]]
rat cos rat rmse 0.0003220549979603387 cv [0.00066867 0.0020113 ] k [0.11812087 1.99979773 0.11805642] coeff [[-0.08, 5.09045, 15169.58036, -15173.36909, -0.16797, 0.04799], [0.07008, 8.18236, 24400.74274, -24406.8263, -0.27093, 0.07748], [-0.05994, 5.15432, 15360.35483, -15364.19114, -0.17008, 0.04859], [0.05006, 8.1293, 24234.71474, -24240.76146, -0.26883, 0.07678]]
exp cos exp4 rmse 0.0008418422794920377 cv [0.00187288 0.0057232 ] k [1.27703975 2.00298455 0.43720435] coeff [[-0.08016, 0.67913, 0.44331, 0.18241, -0.16804, 0.04805], [0.06994, 1.09561, 0.71254, 0.29479, -0.2709, 0.07736], [-0.06003, 0.68773, 0.44913, 0.18439, -0.1701, 0.04846], [0.04994, 1.0867, 0.71079, 0.29065, -0.26842, 0.07677]]
sin tanh None rmse 0.0011653340154693533 cv [0.00231305 0.00719326] k [2.50154338 0.88077187] coeff [[-0.07975, 0.80015, 0.20005, -0.16819, 0.09754], [0.07035, 1.29007, 0.32211, -0.27137, 0.15595], [-0.05975, 0.81032, 0.2024, -0.17013, 0.09854], [0.05029, 1.28038, 0.31969, -0.26914, 0.15601]]
rat exp tanh rmse 0.0022733153001335167 cv [0.00492023 0.01310111] k [0.34713673 1.04557575 0.9328488 ] coeff [[-0.08083, 13.91794, 73.26678, -92.06637, -0.16852, 0.10633], [0.06928, 22.28912, 117.33785, -147.43205, -0.27272, 0.16849], [-0.06071, 14.13051, 74.38155, -93.47195, -0.17064, 0.10691], [0.04879, 22.36778, 117.73248, -147.95565, -0.27038, 0.16885]]
pulses: []
import copy, importlib.util, json, os, pickle, sys
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_batch.pkl"), "rb"))[0]
st = item["study"]
out = mod.solve_batch(copy.deepcopy(st))
m_m = st["source_modes"].shape[0]; nx = int(st["nx"]); ny = int(st["ny"])
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights", (m_m,)), ("terminal_field", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}"); continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    if not isinstance(out.get("kkt_certificate"), dict): errs.append("missing_or_bad:kkt_certificate")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
```

### Turn 7 — `execute_in_remote_docker` (`59109 ms` LLM)
- **Rationale**: Write the transducer using the recovered basis, then check that the composed solver loads and that the implemented fit reaches the observed sweep noise floor before testing inverse and pulse slices.
- **Alternatives Weighed**: `Keep extra tanh terms` (rejected: Their fitted coefficients are near zero and the simpler sine-cosine model reaches the same residual scale.)
- **ATMS NOGOOD DECLARED**: Refuted `The bounded exponential basis captures the sweep response to the noise floor.` | Witness: `Replacing the exponential basis with diagonal sine and paired cosine terms drops worst-fold median forward error from 0.01504 to 2.17e-5.` | Blanket Diagnosis: `The missing odd and even terms account for the apparent saturation and opposite-bank residuals without adding generic interactions.` | Sublated Pivot: `Fit shared trigonometric frequencies and channel coefficients from robustly screened sweeps, then invert with damped Newton steps.`

```bash
cat > /holdout_task/upstream_transducer.py <<'PY'
"""Calibrate the four-channel rosette into the solver's reference coordinates."""

import numpy as np
from scipy.optimize import least_squares


def _design(x, frequency, channel):
    odd, even = frequency
    other = np.arange(4) != channel
    return np.column_stack((np.ones(len(x)), x,
                            np.sin(odd * x[:, channel]),
                            1 - np.cos(even * x[:, other])))


def _fit(ref, raw, single):
    # Estimate an initial residual scale without letting gross EMI rows set it.
    a = np.column_stack((np.ones(len(ref)), ref))
    coef = np.linalg.lstsq(a, raw, rcond=None)[0]
    scale = max(np.median(np.linalg.norm(a @ coef - raw, axis=1)), 1e-10)
    for _ in range(12):
        r = np.linalg.norm(a @ coef - raw, axis=1)
        w = 1 / np.hypot(1, r / scale)
        coef = np.linalg.lstsq(a * w[:, None], raw * w[:, None], rcond=None)[0]
    r = np.linalg.norm(a @ coef - raw, axis=1)
    good = r <= 7 * max(np.median(r), 1e-10)

    axes = []
    bank = np.asarray(single['bank_id'])
    for j in range(4):
        q = bank == j
        signal = np.asarray(single['ref_signal'])[q]
        readout = np.asarray(single['raw_readout'])[q]
        x = np.zeros((len(signal), 4))
        x[:, j] = signal
        dev = np.abs(readout - np.median(readout))
        accepted = dev <= 6 * max(np.median(dev), 1e-10)
        axes.append((x[accepted], readout[accepted]))

    frequency = np.array([2.5, 2.0])
    for _ in range(3):
        samples = [(np.vstack((ref[good], x)), np.r_[raw[good, j], y])
                   for j, (x, y) in enumerate(axes)]

        def profile(freq, coefficients=False):
            result = []
            for j, (x, y) in enumerate(samples):
                a = _design(x, freq, j)
                beta = np.linalg.lstsq(a, y, rcond=None)[0]
                result.append(beta if coefficients else a @ beta - y)
            return np.array(result) if coefficients else np.concatenate(result)

        frequency = least_squares(profile, frequency, bounds=(0.3, 6.0),
                                  ftol=1e-11, xtol=1e-11, gtol=1e-11).x
        coef = profile(frequency, coefficients=True)
        r = np.linalg.norm(_forward(ref, frequency, coef) - raw, axis=1)
        updated = r <= 8 * max(np.median(r), 1e-10)
        if np.array_equal(updated, good):
            break
        good = updated
    return frequency, coef


def _forward(x, frequency, coef):
    return np.column_stack([_design(x, frequency, j) @ coef[j] for j in range(4)])


def _inverse(raw, frequency, coef):
    values = np.asarray(raw, dtype=float)
    target = values.reshape(-1, 4)
    odd, even = frequency
    intercept = coef[:, 0]
    linear = coef[:, 1:5]
    sinusoid = coef[:, 5]
    coupling = np.zeros((4, 4))
    off = ~np.eye(4, dtype=bool)
    coupling[off] = coef[:, 6:].ravel()
    x = np.linalg.solve(linear + np.diag(odd * sinusoid), (target - intercept).T).T
    diag = np.arange(4)
    for _ in range(30):
        error = _forward(x, frequency, coef) - target
        norms = np.linalg.norm(error, axis=1)
        active = np.flatnonzero(norms > 1e-11)
        if not len(active):
            break
        xa = x[active]
        jac = linear + coupling * (even * np.sin(even * xa[:, None, :]))
        jac[:, diag, diag] += odd * sinusoid * np.cos(odd * xa)
        delta = np.linalg.solve(jac, error[active, :, None])[..., 0]
        # A full Newton step is fast near zero; backtrack near saturation.
        for power in range(16):
            trial = xa - delta * (0.5 ** power)
            new_norm = np.linalg.norm(_forward(trial, frequency, coef) - target[active], axis=1)
            improved = new_norm < norms[active]
            x[active[improved]] = trial[improved]
            remains = ~improved
            active, xa, delta = active[remains], xa[remains], delta[remains]
            if not len(active):
                break
    return x.reshape(values.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return calibrated observations and pulses, leaving the input unchanged."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    single = sweeps['single_bank_sweeps']
    model = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float), single)
    result = dict(study)
    result['y_obs_raw'] = _inverse(study['y_obs_raw'], *model)
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        result['reference_tracer_pulses'] = [
            dict(p, y_pulse_raw=_inverse(p['y_pulse_raw'], *model))
            if p.get('y_pulse_raw') is not None else dict(p) for p in pulses]
    result['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=ref.copy(),
        single_bank_sweeps=dict(single, raw_readout=np.asarray(single['ref_signal']).copy()))
    return result
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 validate_public_schema.py /holdout_task/batch_solver.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np,upstream_transducer as u
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; d=c['single_bank_sweeps']; f,b=u._fit(x,y,d)
print('frequencies',f,'coefficients',b.round(6)); q=np.max(np.abs(y),axis=1)<4
print('forward errors',np.quantile(np.linalg.norm(u._forward(x,f,b)-y,axis=1)[q],[.5,1])); print('range',np.min(u._inverse(s['y_obs_raw'],f,b)),np.max(u._inverse(s['y_obs_raw'],f,b)))
PY
```
- **Container Exit Code**: `0` (`883 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/upstream_transducer.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
frequencies [2.50000365 2.00000593] coefficients [[-8.000400e-02  7.999990e-01  1.000000e-06 -1.679990e-01  1.000000e-06
   2.000000e-01  3.000000e-06  4.800200e-02 -1.000000e-06]
 [ 7.000000e-02 -1.000000e-06  1.290001e+00 -3.000000e-06 -2.708980e-01
   3.225000e-01 -1.000000e-06  1.000000e-06  7.740100e-02]
 [-5.999900e-02 -1.701010e-01 -1.000000e-06  8.099970e-01 -0.000000e+00
   2.025030e-01  4.859800e-02  2.000000e-06 -0.000000e+00]
 [ 4.999900e-02 -0.000000e+00 -2.688000e-01  1.000000e-06  1.279999e+00
   3.200020e-01 -3.000000e-06  7.680100e-02  3.000000e-06]]
forward errors [1.75094135e-05 3.80842601e-05]
range -0.3475608823589185 0.4411064715872927

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=33affdb0fd59ba2e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_33affdb0 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`39586 ms` LLM)
- **Rationale**: Test the shipped fit on held-out and boundary sweeps, check Jacobian dominance for uniqueness, and exercise pulse and identity-normalization paths under added EMI. A failed calibration threshold would trigger a fit or inversion revision; a pulse or schema failure would trigger a contract fix.
- **Alternatives Weighed**: `Submit after the public schema check` (rejected: The public study contains no pulses, and the schema check cannot detect inaccurate or nonunique inversion.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, runpy, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as u
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']
c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; a=c['single_bank_sweeps']
clean=np.max(np.abs(y),axis=1)<4  # Score physical readings separately from the known spike stratum; do not filter training data here.
fold=np.random.default_rng(20).permutation(len(x))%5
edges=np.argsort(np.max(abs(x),axis=1))[-16:]
metrics=[]
for held in [np.flatnonzero(fold==j) for j in range(5)]+[edges, np.flatnonzero(~clean)]:
    keep=~np.isin(np.arange(len(x)),held)
    f,b=u._fit(x[keep],y[keep],a)
    q=held[clean[held]]
    if not len(q):
        q=np.flatnonzero(clean)
    fe=np.linalg.norm(u._forward(x[q],f,b)-y[q],axis=1)
    ie=np.max(abs(u._inverse(y[q],f,b)-x[q]),axis=1)
    metrics.append((np.median(fe),fe.max(),ie.max()))
print('exact artifact worst slices: forward median, forward max, inverse max',np.max(metrics,axis=0))
assert np.max(metrics,axis=0)[1]<1e-4 and np.max(metrics,axis=0)[2]<3e-4
f,b=u._fit(x,y,a)
linear=b[:,1:5]; diag=np.diag(linear)
margin=diag-abs(f[0]*b[:,5])-(np.sum(abs(linear),axis=1)-abs(diag))-f[1]*np.sum(abs(b[:,6:]),axis=1)
print('global Jacobian row dominance margins',margin)
assert np.min(margin)>0
rng=np.random.default_rng(47)
z=np.r_[rng.uniform(-2,2,(768,4)), np.array(np.meshgrid(*([[-1.4,0,1.4]]*4))).reshape(4,-1).T]
gain=np.array([.8,1.29,.81,1.28]); offset=np.array([-.08,.07,-.06,.05]); p=z[:,[2,3,0,1]]
truth_raw=offset+gain*(z+.25*np.sin(2.5*z)-.21*p+.06*(1-np.cos(2*p)))
corrupt=y.copy(); rows=rng.choice(np.flatnonzero(clean),7,replace=False); corrupt[rows]+=rng.choice([-1.,1.],(len(rows),4))*rng.uniform(7,18,(len(rows),4))
trial=dict(s,y_obs_raw=truth_raw[:12].reshape(3,4,4),sensor_calibration_sweeps=dict(c,raw_rosette=corrupt),reference_tracer_pulses=[{'y_pulse_raw':truth_raw[12:].ravel(),'marker':2},{'y_pulse_raw':None}]); cal=u.calibrate_upstream(trial)
obs_error=np.max(abs(cal['y_obs_raw'].reshape(-1,4)-z[:12])); pulse_error=np.max(abs(cal['reference_tracer_pulses'][0]['y_pulse_raw'].reshape(-1,4)-z[12:]))
print('extra-EMI observation / boundary pulse inverse max',obs_error,pulse_error)
assert max(obs_error,pulse_error)<3e-4 and cal['y_obs_raw'].shape==trial['y_obs_raw'].shape
assert cal['reference_tracer_pulses'][0]['marker']==2 and cal['reference_tracer_pulses'][1]['y_pulse_raw'] is None
for original,normalized in [(x,cal['sensor_calibration_sweeps']['raw_rosette']),(a['ref_signal'],cal['sensor_calibration_sweeps']['single_bank_sweeps']['raw_readout'])]:
    assert np.array_equal(original,normalized) and not np.shares_memory(original,normalized)
assert not np.array_equal(trial['sensor_calibration_sweeps']['raw_rosette'],x)
assert u.calibrate_upstream(dict(s,reference_tracer_pulses=None))['reference_tracer_pulses'] is None
sys.argv=['validate_public_schema.py','/holdout_task/batch_solver.py']
runpy.run_path('/holdout_task/validate_public_schema.py',run_name='__main__')
PY
```
- **Container Exit Code**: `0` (`581 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
exact artifact worst slices: forward median, forward max, inverse max [2.22797456e-05 4.22474778e-05 1.38839404e-04]
global Jacobian row dominance margins [0.03598348 0.05803976 0.03643542 0.0575781 ]
extra-EMI observation / boundary pulse inverse max 3.936894883960207e-05 8.185517104042361e-05
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=33affdb0fd59ba2e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_33affdb0 (directory).
```

### Turn 9 — `submit_completed_task` (`13106 ms` LLM)
- **Rationale**: The exact artifact passed isolated-process checks on held-out sweeps, calibration boundaries, synthetic pulses under added EMI, identity normalization, and the composed schema. The fitted map also retained positive Jacobian dominance margins in all four channels.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:204', '/holdout_task/upstream_transducer.py': 'FILE:4861'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified the upstream rosette calibration without changing the downstream solver. The composed pipeline passed public schema validation. Maximum held-out inverse error was 1.39e-4; maximum synthetic pulse inverse error under added EMI was 8.19e-5.`

### Turn 10 — `execute_in_remote_docker` (`45374 ms` LLM)
- **Rationale**: Hold out real rosette and single-bank readings together to detect an error hidden by reusing axis sweeps. Compare rational, pair-difference, and compact paired models on the same splits; change the shipped fit only if a challenger fixes a meaningful error or simplifies the model without sacrificing accuracy. Inspect the public fields for actual cohort, label, or relay variables before applying diagnostics that require them.

```bash
cat >/tmp/rosette_gate_audit.py <<'PY'
import pickle, runpy, sys
from collections import Counter
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
import upstream_transducer as u

items = pickle.load(open('/holdout_task/public_batch.pkl','rb'))
print('Public study count:', len(items))
print('Fields relevant to applicability:', sorted({k for item in items for k in item['study'] if any(t in k.lower() for t in ('batch','cohort','mode','relay','label','bank','pulse','time'))}))
for it in items:
    s=it['study']; c=s['sensor_calibration_sweeps']; a=c['single_bank_sweeps']
    x=c['ref_rosette']; y=c['raw_rosette']; bank=np.asarray(a['bank_id']); sig=np.asarray(a['ref_signal']); read=np.asarray(a['raw_readout'])
    f,b=u._fit(x,y,a)
    scoring=np.linalg.norm(u._forward(x,f,b)-y,axis=1)<1e-3
    folds=np.random.default_rng(431).permutation(len(x)) % 5
    edge=np.argsort(np.max(abs(x),axis=1))[-16:]
    held=[np.flatnonzero(folds==j) for j in range(5)]+[edge]
    err={k:[] for k in ('dense trig','paired trig','paired rational','pair difference trig')}; bets=[]; inv=[]; axis_scores=[]; resid=np.full_like(y,np.nan)
    def fit_alternative(train, axis, kind):
        xx=x[train]; yy=y[train]
        f0,b0=u._fit(xx,yy,axis)
        ok=np.linalg.norm(u._forward(xx,f0,b0)-yy,axis=1)<1e-3
        def design(v,j,p):
            q=v[:,j]; h=v[:,(j+2)%4]
            if kind=='paired trig': local=np.sin(p[0]*q); cross=1-np.cos(p[1]*h)
            if kind=='paired rational': local=q/(1+p[0]*q*q); cross=h*h/(1+p[1]*h*h)
            if kind=='pair difference trig': local=np.sin(p[0]*(q-p[2]*h)); cross=1-np.cos(p[1]*h)
            return np.column_stack((np.ones(len(v)),q,h,local,cross))
        samples=[]
        for j in range(4):
            q=np.asarray(axis['bank_id'])==j; z=np.asarray(axis['ref_signal'])[q]; w=np.asarray(axis['raw_readout'])[q]
            v=np.zeros((len(z),4)); v[:,j]=z
            r=abs(w-u._forward(v,f0,b0)[:,j]); clean=r<1e-3
            samples.append((np.vstack((xx[ok],v[clean])),np.r_[yy[ok,j],w[clean]]))
        def prof(p, coeff=False):
            out=[]
            for j,(v,w) in enumerate(samples):
                d=design(v,j,p); m=np.linalg.lstsq(d,w,rcond=None)[0]; out.append(m if coeff else d@m-w)
            return np.array(out) if coeff else np.concatenate(out)
        p0,lo,hi=([2.5,2.,.1],[.3,.3,-1.],[6.,6.,1.]) if kind=='pair difference trig' else (([2.5,2.],[.3,.3],[6.,6.]) if kind=='paired trig' else ([1.,1.],[.001,.001],[15.,15.]))
        p=least_squares(prof,p0,bounds=(lo,hi)).x; m=prof(p,True)
        return lambda v:np.column_stack([design(v,j,p)@m[j] for j in range(4)]), p
    for j,h in enumerate(held):
        tr=~np.isin(np.arange(len(x)),h); q=h[scoring[h]]
        av=(np.arange(len(sig)) % 5)==(j % 5)
        ax={k:np.asarray(a[k])[~av] for k in ('ref_signal','raw_readout','bank_id')}
        freq,beta=u._fit(x[tr],y[tr],ax)
        e=u._forward(x[q],freq,beta)-y[q]
        if j<5: resid[q]=e
        err['dense trig'].append(np.max(np.linalg.norm(e,axis=1)))
        inv.append(np.max(abs(u._inverse(y[q],freq,beta)-x[q])))
        jj=bank[av]; v=np.zeros((sum(av),4)); v[np.arange(len(v)),jj]=sig[av]
        e=u._forward(v,freq,beta)[np.arange(len(v)),jj]-read[av]; clean=abs(read[av]-np.median(read))<4
        axis_scores.append(np.max(abs(e[clean])))
        for kind in tuple(err)[1:]:
            fun,p=fit_alternative(tr,ax,kind)
            err[kind].append(np.max(np.linalg.norm(fun(x[q])-y[q],axis=1)))
            if kind=='pair difference trig': bets.append(p[2])
    print('Rosette clean/EMI rows:', Counter(scoring), 'outer-fold/boundary forward maxima:', {k:max(v) for k,v in err.items()})
    print('Pair-difference fitted shifts:', bets)
    print('Outer-fold/boundary inverse maximum:', max(inv), '; held-out axis maximum:', max(axis_scores), '; OOF per-bank signed mean:', np.nanmean(resid,axis=0))
    assert max(err['dense trig'])<1e-4 and max(inv)<3e-4 and max(axis_scores)<1e-4
sys.argv=['validate_public_schema.py','/holdout_task/batch_solver.py']
runpy.run_path('/holdout_task/validate_public_schema.py',run_name='__main__')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rosette_gate_audit.py
```
- **Container Exit Code**: `0` (`746 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public study count: 1
Fields relevant to applicability: ['row_bank_id', 'source_modes']
Rosette clean/EMI rows: Counter({True: 77, False: 3}) outer-fold/boundary forward maxima: {'dense trig': 4.414082972917989e-05, 'paired trig': 4.167533268499673e-05, 'paired rational': 0.13278582260238095, 'pair difference trig': 4.164231842486939e-05}
Pair-difference fitted shifts: [-1.1193402799030658e-07, -2.7059443466193924e-06, -3.455841742589045e-06, -2.135279461656086e-06, -2.1445977097190266e-06, -2.6596386507848185e-06]
Outer-fold/boundary inverse maximum: 0.00017543833317201774 ; held-out axis maximum: 3.719319472761384e-05 ; OOF per-bank signed mean: [-2.57642206e-07 -2.21701726e-07  1.29160616e-07  1.56613018e-07]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=33affdb0fd59ba2e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_33affdb0 (directory).
```

### Turn 11 — `submit_completed_task` (`15863 ms` LLM)
- **Rationale**: The saved artifacts passed the real-sweep boundary and joint holdouts, pulse and EMI stress tests, and isolated schema validation. The rational alternative was substantially worse, the fitted pair-difference shifts were near zero, and the public study has no cohort labels or relay states to audit.
- **Alternatives Weighed**: `Replace the shipped model with the compact opposite-bank fit` (rejected: It improved the worst forward residual by only 2.47e-6 and would rule out other cross-bank coupling.)
- **Submitted Final Answer / Summary**: `Implemented robust rosette calibration for observations and tracer pulses without changing the downstream solver. The composed pipeline passed isolated public schema validation. Maximum held-out inverse error was 1.75e-4 on real sweeps; maximum synthetic pulse error under added EMI was 8.19e-5.`

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
