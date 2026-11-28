# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m101_b01_batch_03__blind_campaign`

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
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `331,081` (`317,467` / `112,128` / `13,614`) |
| **Total Episode Wall Time (s)** | `426.62s` |

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

### Turn 1 — `execute_in_remote_docker` (`13216 ms` LLM)
- **Rationale**: The archive and diagnostics can identify reusable operators and avoid assuming every panel contributes to the forward law; the schema validator establishes the exact output contract.

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/public_asset_index.json; cat /holdout_task/validate_public_schema.py; for f in /tmp/hegel_continual_archive/{archive_manifest,pre_llm_gate_diagnostic}.json; do echo "=== $f"; head -c 14000 "$f"; echo; done; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust affine reference panel alignment and ridge nuisance profiling'
```
- **Container Exit Code**: `0` (`323 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ry": {
      "obs_00_pa_lin_r2": 1.0,
      "obs_01_pa_cross_corr": 6.2e-05,
      "obs_02_pa_cubic_gain": 0.0,
      "obs_03_pa_rms": 0.765902,
      "obs_04_pb_lin_r2": 1.0,
      "obs_05_pb_quad_gain": 0.0,
      "obs_06_pb_harmonic_gain": 0.0,
      "obs_07_pb_rms": 0.606207,
      "obs_08_pc_v1_lin_r2": 0.129279,
      "obs_09_pc_v1_nl_gain": 0.377415,
      "obs_10_pc_v2_lin_r2": 0.925932,
      "obs_11_pc_v2_nl_gain": 0.073963,
      "obs_12_tr_y_std": 0.510949,
      "obs_13_tr_traj_curv": 0.338323,
      "obs_14_tel_ch0": 0.2,
      "obs_15_tel_ch1": 0.1,
      "obs_16_tel_ch2": 0.2,
      "obs_17_tel_ch3": 0.2
    }
  }
]
=== /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
{
  "accepted": false,
  "schema_valid": true,
  "execution_status": "completed",
  "failure_kind": "scientific_or_numerical",
  "errors": [
    "public_cv_y_mae_exceeded:0.109309",
    "public_cv_traj_mae_exceeded:0.248051"
  ],
  "public_input_sha256": "b2bb9070afb483075325ac53a0a53455c428f960db533d17b8a855c8f3164254",
  "candidate_raw_sha256": "0cbb42d85432832ef1e9dcf0cd540b499b5190d632aff141f5b011996b2f855c",
  "setup_source_sha256": "df547f0e4d05c2cfba86c8a17418b8be609293731c0536ef89006cafbac33215",
  "cv_metrics": {
    "public_cv_y_mae": 0.109309,
    "public_cv_traj_mae": 0.248051
  },
  "dag_runtime_trace": null,
  "kernel_version": "hegel_shared_math_handoff_kernel_v3_m961_three_layer",
  "wall_ms": 2472.46
}
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
```

### Turn 2 — `execute_in_remote_docker` (`19461 ms` LLM)
- **Rationale**: Panel residuals distinguish affine calibration from nonlinear operators, while the trajectory spectrum and archived solvers indicate which low-dimensional dynamics are worth testing on held-out rows.

```bash
cat /tmp/hegel_continual_archive/*_solver.py; cat /holdout_task/validate_public_schema.py; python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
print('time',s['eval_times'],'telemetry',s['telemetry_vec']); t=s['train']; print('summary',t['summary']); print('x y traj',np.c_[t['x'],t['y'],t['traj']][:10]); print('range', np.min(t['traj'],0),np.max(t['traj'],0)); print('traj singulars',np.linalg.svd(t['traj'],compute_uv=False));
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; A=np.c_[np.ones(len(u)),u]; Q=np.c_[A,*[u[:,i]*u[:,j] for i in range(4) for j in range(i,4)]]
 for q,v in p.items():
  if q=='u': continue
  print(k,q,'sample',np.c_[u,v][:4]);
  for X in (A,Q):
   c=np.linalg.lstsq(X,v,rcond=None)[0]; r=v-X@c; print('RMS',np.sqrt(np.mean(r*r,0)),'row error percentiles',np.percentile(np.linalg.norm(r,axis=1),[0,50,80,95,100]));
   if X is A: print('affine',c)
PY
```
- **Container Exit Code**: `0` (`355 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
7.80171 12.01888]
affine [[ 0.06734 -0.00641 -0.3191   0.0719 ]
 [ 0.78414  0.12611 -0.05805 -0.18318]
 [ 0.04518  1.25883  0.123   -0.35795]
 [-0.00185 -0.4731   1.53015  0.10137]
 [ 0.27219  0.01238 -0.35825  0.78467]]
RMS [1.38137 1.35431 1.35724 1.37324] row error percentiles [0.10398 0.95921 1.72766 7.35301 9.9912 ]
panel_b v sample [[ 0.03948  1.08795 -0.75079 -0.97506 -0.07183  0.4319  -1.22582 -0.76626]
 [-0.069   -0.62049 -0.26561  0.45031  0.05265 -0.32663  0.46361 -0.00468]
 [-0.42448 -0.51152  0.40205 -0.77265  0.85278  0.08489 -0.32201  0.4238 ]
 [ 0.92791 -1.07473  1.0595   0.32913  0.42722  0.78871  1.16487  0.64707]]
RMS [0.00008 0.00012 0.00007 0.0001 ] row error percentiles [0.00007 0.00015 0.00022 0.00034 0.00036]
affine [[-0.006    0.13116 -0.08063  0.061  ]
 [-0.06773  0.66808  0.42545 -0.37748]
 [-0.58787  0.28359 -0.5877  -0.05493]
 [ 0.07293  0.4105   0.05291  0.77314]
 [-0.64724 -0.28119  0.4952   0.17647]]
RMS [0.00005 0.00009 0.00006 0.00009] row error percentiles [0.00004 0.00012 0.00018 0.00024 0.00031]
panel_c v1 sample [[ 1.05785 -0.55767 -1.11232 -0.07014 -0.94725  0.7016   0.76929 -0.40974]
 [-0.56978 -0.42648 -0.0117  -0.70363  0.41808 -0.42447  0.19753 -0.84425]
 [-0.10402 -0.01569  0.31666  0.57302  0.38748  0.38143 -0.39139  0.49921]
 [-0.23374 -0.36484  0.25528 -0.5501   0.1802  -0.39568 -0.23917 -0.67474]]
RMS [0.11701 0.08597 0.10967 0.13332] row error percentiles [0.11561 0.22405 0.25648 0.28407 0.30516]
affine [[-0.0275  -0.00477  0.10628  0.14992]
 [-0.85765  0.19772 -0.07441  0.14835]
 [-0.08537 -0.45939  0.16239  0.81824]
 [ 0.06454 -0.22106 -0.86669  0.0526 ]
 [ 0.25799  0.69395 -0.08979  0.51338]]
RMS [0.09457 0.05781 0.07573 0.10546] row error percentiles [0.04603 0.16373 0.18544 0.28864 0.31029]
panel_c v2 sample [[ 1.05785 -0.55767 -1.11232 -0.07014 -0.11939  0.14125 -0.43038  0.75869]
 [-0.56978 -0.42648 -0.0117  -0.70363 -0.13691  0.15615  0.16905 -0.04891]
 [-0.10402 -0.01569  0.31666  0.57302  0.47468  0.07749  0.06947  0.10328]
 [-0.23374 -0.36484  0.25528 -0.5501  -0.05201  0.1498   0.01874 -0.09118]]
RMS [0.10387 0.08515 0.12548 0.07517] row error percentiles [0.08056 0.18542 0.24781 0.26851 0.3166 ]
affine [[ 0.24702 -0.07155 -0.08906  0.16447]
 [-0.04155 -0.02116 -0.42459  0.22821]
 [ 0.05106 -0.49483  0.02874 -0.00711]
 [ 0.20083  0.06308 -0.31393 -0.40815]
 [ 0.4622   0.05823  0.10578  0.167  ]]
RMS [0.00181 0.00317 0.00321 0.0035 ] row error percentiles [0.00114 0.00509 0.00811 0.00939 0.01122]
```

### Turn 3 — `execute_in_remote_docker` (`39731 ms` LLM)
- **Rationale**: The calibration residual percentiles test whether sparse outliers explain the affine fit failure. Held-out temporal projection tests the basis independently of the spatial regressors, and spatial comparisons then test which panel coordinates improve prediction.
- **Alternatives Weighed**: `Predict each timestamp independently` (rejected: The first two singular values are 25.90 and 2.16, while the third is 0.050; a shared temporal basis is a stronger starting point.)

```bash
cat /tmp/hegel_continual_archive/*02_solver.py; cat /holdout_task/validate_public_schema.py; cat >/tmp/discover03.py <<'PY'
import pickle,itertools as it
import numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=tr['x'],tr['y'],tr['traj']; t=s['eval_times']; L=np.linalg.lstsq

def D(x): return np.c_[np.ones(len(x)),x]
def F(x,p): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,a],1) for k in range(1,p+1) for a in it.combinations_with_replacement(range(4),k)])
def robust(A,v):
 n,p=A.shape; rg=np.random.default_rng(8); best=None
 for _ in range(400):
  i=rg.choice(n,p,replace=False); w=L(A[i],v[i],rcond=None)[0]; e=np.linalg.norm(A@w-v,axis=1); q=np.median(e)
  if best is None or q<best[0]: best=(q,e)
 q,e=best; mask=e<max(0.001,5*q)
 for _ in range(3):
  w=L(A[mask],v[mask],rcond=None)[0]; e=np.linalg.norm(A@w-v,axis=1); mask=e<max(.001,5*np.median(e))
 return w,e
pa,pb,pc=[s['panel_'+k] for k in 'abc']; u,v=pa['u'],pa['v']; A,e=robust(D(u),v); B=L(D(pb['u']),pb['v'],rcond=None)[0]; C=L(F(pc['u'],2),pc['v2'],rcond=None)[0];
if __name__=='__main__':
 np.set_printoptions(precision=5,suppress=True); print('times',t,'telemetry',s['telemetry_vec'],'summary',tr['summary']); print('A',A,'err percentiles',np.percentile(e,[0,25,50,75,85,90,95,100])); print('robust quad pct',np.percentile(robust(F(u,2),v)[1],[25,50,75,85])); print('C2 coefficient map',C)
z=(x-A[0])@np.linalg.inv(A[1:]); b=D(z)@B; cs={'raw':x,'cal':z,'b':b,'braw':D(x)@B};
for k in ['cal','b']:
 q=cs[k]; cs[k+'c1']=D(q)@L(D(pc['u']),pc['v1'],rcond=None)[0]; cs[k+'c2']=F(q,2)@C

def projected(q,T): return (q @ L(q,T.T,rcond=None)[0]-T.T).ravel()
def timefit(T,case):
 def make(p):
  r=np.exp(p); h=-np.expm1(-t[:,None]*r)
  if case=='2exp':return h
  if case=='linear-exp':return np.c_[t,h]
  if case=='exp-decay':return np.c_[h[:,0],np.exp(-t*r[1])]
  if case=='two-rational':return t[:,None]/(1+t[:,None]*r)
  if case=='poly-exp':return np.c_[h,t] 
  if case=='two-damped':return np.exp(-t[:,None]*r)
 n=1 if case=='linear-exp' else 2; f=least_squares(lambda p:projected(make(p),T),np.linspace(-1,1,n),bounds=(-8,5)); return np.exp(f.x),make(f.x),np.std(f.fun)
if __name__=='__main__':
 for k in ['2exp','linear-exp','exp-decay','two-rational','two-damped']:
  r,H,err=timefit(T[:52],k); print('timefit',k,r,err,np.std(projected(H,T[52:])))
 r,H,_=timefit(T[:52],'2exp'); W=T@np.linalg.pinv(H.T); V=np.c_[y,W]; print('std targets',V.std(0),'corrcoef',np.corrcoef(V.T)); print('std x cal b',x.std(0),z.std(0),b.std(0));
 for k,q in cs.items():
  for typ,G in [('lin',D(q)),('quad',F(q,2)),('cub',F(q,3)),('tanh',F(np.tanh(q),2)),('sin',F(np.sin(q),2))]:
   p=G@L(G[:52],V[:52],rcond=None)[0]; print(k,typ,'fit',np.round(np.mean(abs(p[:52]-V[:52]),0),4),'hold',np.round(np.mean(abs(p[52:]-V[52:]),0),4),'traj',round(np.mean(abs(p[52:,1:]@H.T-T[52:])),4))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/discover03.py
```
- **Container Exit Code**: `0` (`618 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
al cub fit [0.0437 0.0572 0.0402] hold [0.402  0.4607 0.4275] traj 0.4628
cal tanh fit [0.1081 0.1601 0.0988] hold [0.2024 0.3327 0.2529] traj 0.3427
cal sin fit [0.1081 0.1566 0.104 ] hold [0.2052 0.3391 0.2678] traj 0.3509
b lin fit [0.2564 0.2367 0.1906] hold [0.2287 0.3988 0.2555] traj 0.3393
b quad fit [0.1234 0.1619 0.1168] hold [0.2403 0.3932 0.361 ] traj 0.4302
b cub fit [0.0437 0.0572 0.0402] hold [0.402  0.4607 0.4275] traj 0.4628
b tanh fit [0.076  0.1044 0.0739] hold [0.154  0.1968 0.1817] traj 0.2206
b sin fit [0.0847 0.1196 0.0877] hold [0.1818 0.2309 0.2203] traj 0.2655
braw lin fit [0.2564 0.2367 0.1906] hold [0.2287 0.3988 0.2555] traj 0.3393
braw quad fit [0.1234 0.1619 0.1168] hold [0.2403 0.3932 0.361 ] traj 0.4302
braw cub fit [0.0437 0.0572 0.0402] hold [0.402  0.4607 0.4275] traj 0.4628
braw tanh fit [0.0787 0.1215 0.1   ] hold [0.1133 0.2747 0.1941] traj 0.2226
braw sin fit [0.0693 0.1208 0.098 ] hold [0.129  0.262  0.2327] traj 0.2558
calc1 lin fit [0.2564 0.2367 0.1906] hold [0.2287 0.3988 0.2555] traj 0.3393
calc1 quad fit [0.1234 0.1619 0.1168] hold [0.2403 0.3932 0.361 ] traj 0.4302
calc1 cub fit [0.0437 0.0572 0.0402] hold [0.402  0.4607 0.4275] traj 0.4628
calc1 tanh fit [0.1026 0.1468 0.108 ] hold [0.2173 0.2699 0.2691] traj 0.3142
calc1 sin fit [0.105  0.1471 0.1089] hold [0.2277 0.2857 0.2899] traj 0.33
calc2 lin fit [0.3072 0.3938 0.2024] hold [0.3534 0.5268 0.4129] traj 0.4504
calc2 quad fit [0.1182 0.1469 0.1157] hold [0.5263 0.4523 0.2635] traj 0.4088
calc2 cub fit [0.036  0.0827 0.0439] hold [1.5303 0.8137 0.7365] traj 0.8854
calc2 tanh fit [0.1143 0.1433 0.1065] hold [0.3998 0.3482 0.2226] traj 0.3192
calc2 sin fit [0.1148 0.1425 0.1089] hold [0.4201 0.3812 0.2296] traj 0.3383
bc1 lin fit [0.2564 0.2367 0.1906] hold [0.2287 0.3988 0.2555] traj 0.3393
bc1 quad fit [0.1234 0.1619 0.1168] hold [0.2403 0.3932 0.361 ] traj 0.4302
bc1 cub fit [0.0437 0.0572 0.0402] hold [0.402  0.4607 0.4275] traj 0.4628
bc1 tanh fit [0.1295 0.1296 0.102 ] hold [0.2635 0.3542 0.2744] traj 0.3467
bc1 sin fit [0.1265 0.1394 0.1066] hold [0.2598 0.357  0.2788] traj 0.3575
bc2 lin fit [0.3185 0.4365 0.2263] hold [0.258  0.6238 0.337 ] traj 0.4426
bc2 quad fit [0.0732 0.1681 0.0769] hold [0.1912 0.2922 0.1018] traj 0.1494
bc2 cub fit [0.0362 0.0759 0.0397] hold [0.3249 0.9991 0.4266] traj 0.4387
bc2 tanh fit [0.0607 0.153  0.0738] hold [0.1678 0.2361 0.0784] traj 0.1003
bc2 sin fit [0.0634 0.1569 0.0748] hold [0.1655 0.2484 0.0834] traj 0.1126
```

### Turn 4 — `execute_in_remote_docker` (`16423 ms` LLM)
- **Rationale**: The previous log truncated the calibration residuals and C2 coefficient map. Those values will determine whether to extend the calibration operator or focus on a sparse C2 basis; the archived solver provides candidate nonlinear features to test rather than assume.
- **Alternatives Weighed**: `Increase the degree of the unrestricted polynomial` (rejected: After composing B and C2, moving from a quadratic to a cubic raises held-out trajectory MAE from 0.1494 to 0.4388 despite lowering training error.)

```bash
OPENBLAS_NUM_THREADS=1 python3 /tmp/discover03.py | head -37; sed -n '1,180p' /tmp/hegel_continual_archive/*01_solver.py; cat /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`557 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
pickle, sys
import numpy as np

_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open(os.path.join(_d, "public_batch.pkl"), "rb") as fh:
    st = pickle.load(fh)[0]["study"]
out = mod.solve_batch(copy.deepcopy(st))
n_te = len(st["test"]["x"])
k_t = len(st["eval_times"])
errs = []
if not isinstance(out, dict):
    errs.append("not_dict")
else:
    for k, shp in (("y_hat", (n_te,)), ("state_pred", (n_te, k_t)), ("diagnostics_vec", (4,))):
        if k not in out:
            errs.append(f"missing:{k}")
            continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)):
            errs.append(f"bad:{k}:{a.shape}")
    if not errs:
        yh = np.asarray(out["y_hat"], dtype=float)
        sp = np.asarray(out["state_pred"], dtype=float)
        dv = np.asarray(out["diagnostics_vec"], dtype=float)
        exp_dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
        if float(np.max(np.abs(dv - exp_dv))) > 1e-3:
            errs.append("diagnostics_vec_handoff_mismatch")
if not errs:
    st_cv = copy.deepcopy(st)
    sp_idx = max(40, len(st["train"]["x"]) - 20)
    vx, vy, vtr = st["train"]["x"][sp_idx:].copy(), st["train"]["y"][sp_idx:].copy(), st["train"]["traj"][sp_idx:].copy()
    st_cv["train"] = {
        "x": st["train"]["x"][:sp_idx].copy(),
        "y": st["train"]["y"][:sp_idx].copy(),
        "traj": st["train"]["traj"][:sp_idx].copy(),
        "summary": np.array([
            float(np.mean(st["train"]["y"][:sp_idx])),
            float(np.std(st["train"]["y"][:sp_idx])),
            float(np.mean(st["train"]["traj"][:sp_idx, -1])),
            float(np.std(st["train"]["traj"][:sp_idx, -1])),
        ], dtype=float),
    }
    st_cv["test"] = {"x": vx}
    out_cv = mod.solve_batch(st_cv)
    ym = float(np.mean(np.abs(np.asarray(out_cv["y_hat"], dtype=float) - vy)))
    tm = float(np.mean(np.abs(np.asarray(out_cv["state_pred"], dtype=float) - vtr)))
    if ym > 0.038:
        errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
    if tm > 0.038:
        errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")
print(json.dumps({"schema_valid": not errs, "errors": errs}, allow_nan=False))
sys.exit(1 if errs else 0)
```

### Turn 5 — `execute_in_remote_docker` (`30594 ms` LLM)
- **Rationale**: If compact nonlinear bases fail on omitted panel rows, downstream predictions should not rely on their extrapolation. Comparing separable and full bases distinguishes missing cross terms from coordinate-wise distortion.
- **Alternatives Weighed**: `Fit a larger supervised readout directly` (rejected: The cubic readout already lowered training error while worsening held-out error; panel measurements can identify missing structure independently.)
- **ATMS NOGOOD DECLARED**: Refuted `Sparse outlier removal alone makes panel A adequately affine.` | Witness: `After robust fitting, the median affine residual is 0.31035, whereas the robust quadratic median is 0.08309.` | Blanket Diagnosis: `A systematic calibration distortion remains in the inliers and propagates through subsequent coordinate transforms.` | Sublated Pivot: `Retain robust inlier selection and compare compact nonlinear calibration bases with leave-one-out panel residuals.`

```bash
cat >/tmp/panel03.py <<'PY'
from discover03 import *
from scipy.special import erf
np.set_printoptions(precision=4,suppress=True,linewidth=120)
idx=np.flatnonzero(e<1); print('cal inliers',len(idx));
for name,uu,vv in [('A',u[idx],v[idx]),('C1',pc['u'],pc['v1']),('C2',pc['u'],pc['v2'])]:
 print(name)
 for name2,f in [('powers2',lambda x:np.c_[D(x),x*x]),('powers3',lambda x:np.c_[D(x),x*x,x**3]),('powers4',lambda x:np.c_[D(x),x*x,x**3,x**4]),('full2',lambda x:F(x,2)),('full2+diag3',lambda x:np.c_[F(x,2),x**3]),('full3',lambda x:F(x,3))]:
  X=f(uu); w=L(X,vv,rcond=None)[0]; err=X@w-vv; h=np.sum(X*np.linalg.pinv(X).T,1); print(name2, 'rms',np.sqrt(np.mean(err**2)),'loo',np.sqrt(np.mean((err/np.maximum(1-h[:,None],1e-8))**2)) if X.shape[1]<len(X) else None)
  if name2 in ['powers3','full2'] and name!='C2': print(w)
 for label,phi in [('tanh',np.tanh),('sin',np.sin),('erf',erf),('asinh',np.arcsinh)]:
  def mat(p,x): return D(phi(x*np.exp(p[0])+p[1:]))
  def fun(p):
   m=mat(p,uu); return (m@L(m,vv,rcond=None)[0]-vv).ravel()
  q=least_squares(fun,np.r_[-.4,np.zeros(4)],bounds=(np.r_[-4,np.full(4,-3)],np.r_[2,np.full(4,3)]),max_nfev=100); print(label,'in',np.sqrt(np.mean(q.fun**2)),q.x)
  def fout(p): return (phi(D(uu)@p.reshape(5,4))-vv).ravel()
  q=least_squares(fout,L(D(uu),vv,rcond=None)[0].ravel(),max_nfev=80);print(label,'out',np.sqrt(np.mean(q.fun**2)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/panel03.py
```
- **Container Exit Code**: `0` (`814 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
16 -0.1445 -0.0881 -2.8693]
asinh out 0.19477151158049116
C1
powers2 rms 0.10278037502866254 loo 0.19288838201390776
powers3 rms 0.029028417441554755 loo 0.08326073986744467
[[-0.0096 -0.0279  0.1101  0.1395]
 [-1.2545  0.2829 -0.0832  0.1669]
 [-0.0874 -0.6585  0.1512  1.0736]
 [ 0.0762 -0.2408 -1.2536  0.0933]
 [ 0.3732  0.9544 -0.1615  0.7786]
 [-0.0004 -0.0139 -0.0045  0.0381]
 [-0.0037  0.02   -0.0096 -0.0143]
 [-0.0064  0.0207 -0.0107 -0.0388]
 [ 0.0059  0.0164 -0.0031 -0.0106]
 [ 0.3934 -0.0598 -0.0147 -0.0482]
 [ 0.0295  0.1531 -0.0028 -0.3525]
 [-0.012   0.0409  0.3869 -0.0192]
 [-0.1241 -0.2661  0.0446 -0.2326]]
powers4 rms 0.022957435899107346 loo 0.10852612193927674
full2 rms 0.08535589709757632 loo 0.3569060953377118
[[-0.0326 -0.0291 -0.0269  0.1583]
 [-0.9304  0.2051 -0.0768  0.2194]
 [-0.0802 -0.4377  0.2223  0.8643]
 [-0.0304 -0.2274 -0.9082  0.1338]
 [ 0.2464  0.7136 -0.1277  0.5538]
 [-0.0931 -0.1071  0.0501  0.1226]
 [ 0.0153 -0.0397  0.0241 -0.0882]
 [ 0.0149 -0.0468  0.0154 -0.0005]
 [ 0.0279  0.014  -0.1132 -0.1063]
 [-0.0278  0.0055 -0.0397 -0.0223]
 [-0.195  -0.0156 -0.1686  0.0058]
 [-0.0011  0.0375 -0.0056  0.1066]
 [ 0.2116  0.1099  0.1647 -0.1794]
 [ 0.1709 -0.079   0.0742 -0.1805]
 [ 0.0439  0.0394  0.1651  0.0279]]
full2+diag3 rms 0.017503406215179915 loo 0.13748932189660903
full3 rms 9.114523257659078e-16 loo None
tanh in 0.012658699147726393 [ 0.4639 -0.012   0.017  -0.0052 -0.0093]
tanh out 0.1354594800172825
sin in 0.028979486751195205 [ 0.3904 -0.0096  0.0419 -0.0113 -0.017 ]
sin out 0.1307705472799848
erf in 0.016326815458704652 [ 0.3464 -0.0119  0.019  -0.0055 -0.0097]
erf out 0.13032007993273786
asinh in 0.007180405223846834 [ 1.5602  0.0148 -0.022   0.0126  0.0054]
asinh out 0.09811913990252012
C2
powers2 rms 0.004876773647631561 loo 0.009587548263835833
powers3 rms 0.0038969492865771516 loo 0.013955439212204093
powers4 rms 0.0002981658835758554 loo 0.001573816049321721
full2 rms 0.002994681487868866 loo 0.012358465160765088
full2+diag3 rms 0.0017926345670336644 loo 0.018789300597980015
full3 rms 5.274593537375872e-16 loo None
tanh in 0.00260172676681227 [-0.1544 -0.8153 -0.8009 -0.8184 -0.8238]
tanh out 0.11547228519723385
sin in 0.005083599470252507 [-1.42   -1.3025 -1.3043 -1.3086 -1.3014]
sin out 0.1083847695173602
erf in 0.0008789077066574452 [-0.344  -0.7715 -0.7676 -0.7744 -0.7749]
erf out 0.11276897206294688
asinh in 0.014135212051232198 [ 0.988  -1.9893 -1.8577 -2.0821 -2.1261]
asinh out 0.10493752576103893
```

### Turn 6 — `execute_in_remote_docker` (`53116 ms` LLM)
- **Rationale**: The forward calibration can fit closely yet still invert poorly, so the probe measures inversion residuals before attributing downstream errors to the readouts. With the corrected coordinates, held-out readout and trajectory MAEs distinguish active panel chains from panels that merely fit their own observations.
- **Alternatives Weighed**: `Invert an unrestricted full cubic calibration` (rejected: The compact quadratic-plus-coordinate-cubic basis has lower panel leave-one-out error; the coefficient pattern supports testing an even smaller cyclic basis first.)

```bash
cat >/tmp/chain03.py <<'PY'
from discover03 import *
from scipy.special import hyp2f1,erf

def a_design(q,j):
 a=q[:,j]; b=q[:,(j+1)%4]; return np.c_[np.ones(len(q)),a,b,a*b,a**3]
idx=e<1
AA=np.column_stack([L(a_design(u[idx],j),v[idx,j],rcond=None)[0] for j in range(4)])
def calibrate(q):
 cur=(q-A[0])@np.linalg.inv(A[1:]); j=np.arange(4); k=(j+1)%4
 for _ in range(20):
  J=np.zeros((len(q),4,4)); J[:,j,j]=AA[1]+AA[3]*cur[:,k]+3*AA[4]*cur**2; J[:,j,k]=AA[2]+AA[3]*cur
  err=np.column_stack([a_design(cur,i)@AA[:,i] for i in j])-q
  step=np.linalg.solve(J,err[:,:,None])[:,:,0];cur-=step/np.maximum(1,np.max(abs(step),axis=1,keepdims=True)/.3)
 return cur,np.max(abs(err),axis=1)
uc=pc['u']; c1=pc['v1']; c2=pc['v2']
phis={'atan':lambda q,p:np.arctan(q*np.exp(p[0])), 'asinh':lambda q,p:np.arcsinh(q*np.exp(p[0])), 'tanh':lambda q,p:np.tanh(q*np.exp(p[0])), 'erf':lambda q,p:erf(q*np.exp(p[0])), 'rational':lambda q,p:q/(1+np.exp(p[0])*q*q)**np.exp(p[1]), 'integral':lambda q,p:q*hyp2f1(.5,np.exp(p[1]),1.5,-np.exp(p[0])*q*q), 'softsign':lambda q,p:q/(1+np.exp(p[0])*abs(q))}
fits={}
for n,phi in phis.items():
 def make(p,q):return D(phi(q,p))
 def fun(p):
  m=make(p,uc);return (m@L(m,c1,rcond=None)[0]-c1).ravel()
 f=least_squares(fun,[0.,0.] if n in ('rational','integral') else [0.],bounds=(-4,4),max_nfev=100)
 fits[n]=(f.x,L(make(f.x,uc),c1,rcond=None)[0]); print('C1',n,f.x,np.std(f.fun))

def p4(q):return np.c_[D(q),q*q,q**3,q**4]
C4=L(p4(uc),c2,rcond=None)[0]; print('A coefs',AA,'A inlier rms',np.sqrt(np.mean((np.column_stack([a_design(u[idx],i)@AA[:,i] for i in range(4)])-v[idx])**2))); print('C2 powers',np.round(C4,4));
def C1(q,n):
 p,w=fits[n];return D(phis[n](q,p))@w

def c2fit(name,phi,p0):
 def fun(p):
  m=D(phi(uc,p));return (m@L(m,c2,rcond=None)[0]-c2).ravel()
 f=least_squares(fun,p0,bounds=(-5,5),max_nfev=100)
 print('C2 sep',name,f.x,np.std(f.fun));return lambda q:D(phi(q,f.x))@L(D(phi(uc,f.x)),c2,rcond=None)[0]
C2s={
 'poly':c2fit('poly',lambda q,p:q+p[0]*q*q+p[1]*q**3+p[2]*q**4,[.45,0,0]),
 'tanh':c2fit('tanh',lambda q,p:np.tanh(np.exp(p[0])*(q+p[1])),[0,-.8]),
 'erf':c2fit('erf',lambda q,p:erf(np.exp(p[0])*(q+p[1])),[0,-.8]),
 'sqrat':c2fit('sqrat',lambda q,p:(q+p[1])/(1+np.exp(p[0])*(q+p[1])**2)**np.exp(p[2]),[0,-.8,-.6]),
 'squared':c2fit('squared',lambda q,p:q+p[0]*q**2+p[1]*(q+p[0]*q**2)**2,[.45,.02]),
 'ratquad':c2fit('ratquad',lambda q,p:(q+p[0]*q*q)/(1+np.exp(p[1])*q*q),[.45,-4]),
}
zz,ee=calibrate(x); print('inverse errors',np.quantile(ee,[0,.9,1]),'bounds',zz.min(0),zz.max(0)); bb=D(zz)@B
H=timefit(T[:52],'2exp')[1]; V=np.c_[y,T@np.linalg.pinv(H.T)]
coords={'a':zz,'ab':bb}
for tag,q in list(coords.items()):
 for n in ('rational','integral','asinh'):coords[tag+'1'+n]=C1(q,n)
 coords[tag+'2']=p4(q)@C4;coords[tag+'2sep']=C2s['poly'](q)
 if tag=='ab':
  for n,fn in C2s.items():coords[tag+'2'+n]=fn(q)
for key in [k for k in coords if k.startswith('ab1') or k=='ab2']:
 q=coords[key];coords[key+'2']=p4(q)@C4;coords[key+'1']=C1(q,'rational')
res=[]
for k,q in coords.items():
 for typ,G in [('lin',D(q)),('quad',F(q,2)),('cub',F(q,3)),('tanh',F(np.tanh(q),2)),('sin',F(np.sin(q),2)),('sep3',np.c_[D(q),q**2,q**3]),('pair',np.c_[D(q),q[:,:2]*q[:,2:]]),('quart',p4(q))]:
  p=G@L(G[:52],V[:52],rcond=None)[0]; res.append((np.mean(abs(p[52:,0]-y[52:])),np.mean(abs(p[52:,1:]@H.T-T[52:])), k, typ,np.mean(abs(p[:52]-V[:52]),0)))
print('best y',*sorted(res,key=lambda a:a[0])[:12],sep='\n');print('best traj',*sorted(res,key=lambda a:a[1])[:12],sep='\n')
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/chain03.py
```
- **Container Exit Code**: `0` (`520 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
array([0.00810262, 0.01346712, 0.01034531]))
(0.010009101820943219, 0.007402847186742394, 'ab1integral', 'quad', array([0.00539362, 0.00865972, 0.00784533]))
(0.010261942867026873, 0.018488528003073306, 'ab1asinh', 'pair', array([0.01221324, 0.01872044, 0.01375695]))
(0.011499025966256153, 0.008210796264624677, 'ab1rational', 'quad', array([0.00644894, 0.00960677, 0.0085429 ]))
(0.011782156763770339, 0.010741119214895806, 'ab1integral', 'cub', array([0.00377029, 0.00564398, 0.00543847]))
(0.013065751864478763, 0.011244797335015749, 'ab1rational', 'cub', array([0.0044202 , 0.00604219, 0.00615759]))
(0.01631035361108419, 0.04531787599053176, 'ab1asinh', 'cub', array([0.00793462, 0.01091381, 0.00619579]))
(0.018750773634097502, 0.016022133347744983, 'ab1asinh', 'quad', array([0.00938989, 0.0154011 , 0.01269935]))
(0.06208098800202553, 0.18248100930788885, 'ab21', 'cub', array([0.01833482, 0.03138556, 0.0140961 ]))
(0.0716749218881664, 0.00675838473280848, 'ab1integral', 'lin', array([0.05410936, 0.0130677 , 0.00951376]))
(0.07176226904120118, 0.007515265914754113, 'ab1rational', 'lin', array([0.05369257, 0.01415906, 0.01032216]))
best traj
(0.0716749218881664, 0.00675838473280848, 'ab1integral', 'lin', array([0.05410936, 0.0130677 , 0.00951376]))
(0.008333689205241015, 0.007268991558540097, 'ab1integral', 'pair', array([0.00687218, 0.01255226, 0.0094944 ]))
(0.010009101820943219, 0.007402847186742394, 'ab1integral', 'quad', array([0.00539362, 0.00865972, 0.00784533]))
(0.07176226904120118, 0.007515265914754113, 'ab1rational', 'lin', array([0.05369257, 0.01415906, 0.01032216]))
(0.08448680020355746, 0.00816550682621119, 'ab1integral', 'sep3', array([0.04260395, 0.01071261, 0.00850867]))
(0.009697615348200909, 0.008187360321629287, 'ab1rational', 'pair', array([0.00810262, 0.01346712, 0.01034531]))
(0.011499025966256153, 0.008210796264624677, 'ab1rational', 'quad', array([0.00644894, 0.00960677, 0.0085429 ]))
(0.08407091781816568, 0.009043643376855561, 'ab1rational', 'sep3', array([0.04243707, 0.01137921, 0.00911733]))
(0.011782156763770339, 0.010741119214895806, 'ab1integral', 'cub', array([0.00377029, 0.00564398, 0.00543847]))
(0.08601052534040986, 0.010992278959253976, 'ab1integral', 'quart', array([0.04075299, 0.01073379, 0.00764987]))
(0.013065751864478763, 0.011244797335015749, 'ab1rational', 'cub', array([0.0044202 , 0.00604219, 0.00615759]))
(0.08583660385603459, 0.012936145682124364, 'ab1rational', 'quart', array([0.04086085, 0.01132939, 0.00814653]))
```

### Turn 7 — `execute_in_remote_docker` (`53791 ms` LLM)
- **Rationale**: The independent calibration measurements support the cyclic cubic inverse. The public validator checks both required outputs; isolated random folds and a high-radius holdout check whether the selected composition loses its margin away from the official split.
- **Alternatives Weighed**: `Compose the C2 branch or add unrestricted readout terms` (rejected: C2 compositions performed substantially worse, and the compact paired-product scalar and affine state designs each beat their unrestricted quadratic counterpart on the 52/20 split.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
"""Predict the readout and two-mode trajectory from calibrated panel coordinates."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import hyp2f1


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _calibration_basis(x, j):
    a, b = x[:, j], x[:, (j + 1) % x.shape[1]]
    return np.column_stack((np.ones(len(x)), a, b, a*b, a**3))


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    # Use robust affine fits only to initialize outlier rejection and inversion.
    design = _affine(u)
    rng = np.random.default_rng(8)
    best_score = np.inf
    for _ in range(400):
        rows = rng.choice(len(u), design.shape[1], replace=False)
        err = np.linalg.norm(design @ _fit(design[rows], v[rows]) - v, axis=1)
        score = np.median(err)
        if score < best_score:
            best_score, best_err = score, err
    keep = best_err < max(0.001, 5 * best_score)
    for _ in range(3):
        pilot = _fit(design[keep], v[keep])
        err = np.linalg.norm(design @ pilot - v, axis=1)
        keep = err < max(0.001, 5 * np.median(err))
    j = np.arange(u.shape[1])
    k = (j + 1) % len(j)
    features = [_calibration_basis(u, i) for i in j]
    for _ in range(3):
        w = np.column_stack([_fit(f[keep], v[keep, i]) for i, f in enumerate(features)])
        err = np.linalg.norm(np.column_stack([f @ w[:, i] for i, f in enumerate(features)]) - v, axis=1)
        keep = err < max(0.001, 5 * np.median(err))
    z = np.linalg.solve(pilot[1:].T, (x - pilot[0]).T).T
    jac = np.zeros((len(x), len(j), len(j)))
    for _ in range(20):
        residual = w[0] + w[1]*z + w[2]*z[:, k] + w[3]*z*z[:, k] + w[4]*z**3 - x
        if np.max(np.abs(residual), initial=0.) < 1e-12:
            break
        jac[:, j, j] = w[1] + w[3]*z[:, k] + 3*w[4]*z*z
        jac[:, j, k] = w[2] + w[3]*z
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        z -= step / np.maximum(1., np.max(np.abs(step), axis=1, keepdims=True) / .3)
    return z


def _c1_coordinates(panel, x):
    u, v = panel['u'], panel['v1']

    def design(q, log_params):
        scale, power = np.exp(log_params)
        return _affine(q * hyp2f1(.5, power, 1.5, -scale*q*q))

    def residual(log_params):
        g = design(u, log_params)
        return (g @ _fit(g, v) - v).ravel()

    params = least_squares(residual, [0., 0.], bounds=(-4., 4.)).x
    return design(x, params) @ _fit(design(u, params), v)


def _state(design, traj, prediction_design, times):
    def modes(log_rates):
        return -np.expm1(-times[:, None] * np.exp(log_rates))

    def residual(log_rates):
        h = modes(log_rates)
        return (h @ _fit(h, traj.T) - traj.T).ravel()

    rates = least_squares(residual, np.log([.4, 1.8]), bounds=(-8., 5.)).x
    h = modes(rates)
    return prediction_design @ _fit(design, _fit(h, traj.T).T) @ h.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']))
    calibrated = _calibrate(study['panel_a'], x)
    b = study['panel_b']
    operator_b = _affine(calibrated) @ _fit(_affine(b['u']), b['v'])
    q = _c1_coordinates(study['panel_c'], operator_b)
    state_design = _affine(q)
    readout_design = np.column_stack((state_design, q[:, :2] * q[:, 2:]))
    y = readout_design[n:] @ _fit(readout_design[:n], train['y'])
    state = _state(state_design[:n], train['traj'], state_design[n:], np.asarray(study['eval_times']))
    terminal = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
spec = importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
t=s['train']; x=t['x']; n=len(x); ids=np.arange(n)
z=m._calibrate(s['panel_a'], x)
radial=np.linalg.norm(z,axis=1)
folds={'52/20':ids[52:], '56/16':ids[56:], 'high radius':np.argsort(radial)[-16:]}
for i,f in enumerate(np.array_split(np.random.default_rng(103).permutation(n),4)):
    folds[f'random {i+1}']=f
for label,te in folds.items():
    tr=np.setdiff1d(ids,te)
    train={k:t[k][tr] for k in ('x','y','traj')}
    train['summary']=np.array([train['y'].mean(),train['y'].std(),train['traj'][:,-1].mean(),train['traj'][:,-1].std()])
    o=m.solve_batch(dict(s,train=train,test={'x':x[te]}))
    dy=np.abs(o['y_hat']-t['y'][te]); ds=np.abs(o['state_pred']-t['traj'][te]);
    print(f'{label}: readout MAE {dy.mean():.6f}; trajectory MAE {ds.mean():.6f}; worst timestamp MAE {ds.mean(0).max():.6f}')
    assert dy.mean() <= .038 and ds.mean(0).max() <= .038
PY
```
- **Container Exit Code**: `0` (`887 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
52/20: readout MAE 0.008334; trajectory MAE 0.006758; worst timestamp MAE 0.008025
56/16: readout MAE 0.008511; trajectory MAE 0.006766; worst timestamp MAE 0.008326
high radius: readout MAE 0.009142; trajectory MAE 0.011105; worst timestamp MAE 0.013851
random 1: readout MAE 0.007794; trajectory MAE 0.008212; worst timestamp MAE 0.010044
random 2: readout MAE 0.008434; trajectory MAE 0.008489; worst timestamp MAE 0.010299
random 3: readout MAE 0.007985; trajectory MAE 0.006294; worst timestamp MAE 0.007651
random 4: readout MAE 0.007470; trajectory MAE 0.006740; worst timestamp MAE 0.008573

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=331760124bfb30ff (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_33176012 (directory).
```

### Turn 8 — `submit_completed_task` (`16576 ms` LLM)
- **Rationale**: The saved artifact passed the public validator and isolated-process stress tests, including the high-radius holdout. Both outputs have substantial margin to the required tolerance.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`6` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:3764'}): Do NOT lock prematurely with `6` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public validator passed: 52/20 MAE was 0.00833 for the scalar readout and 0.00676 for the trajectory, both below 0.038.`

### Turn 9 — `execute_in_remote_docker` (`55606 ms` LLM)
- **Rationale**: Leaving each reference row out of calibration and outlier selection tests whether the cyclic terms generalize beyond the fitted rows. Panel and downstream folds then test whether rational or pairwise alternatives improve the worst slice enough to change the delivered solver.
- **Alternatives Weighed**: `Run subgroup classification, relay, or drift-operator probes` (rejected: The observed task has continuous readouts, one apparatus telemetry vector, and no batch labels, relay modes, or field measurements to support those tests.)

```bash
cat >/tmp/audit03.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.special import hyp2f1
from itertools import combinations_with_replacement

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
tr = s['train']
print('Study fields:', sorted(s), '; train fields:', sorted(tr))
print('Telemetry shape:', np.shape(s['telemetry_vec']), '; distinct evaluation times:', len(np.unique(s['eval_times'])))

# Use the same robust pilot for the affine and cyclic fits, without the omitted row.
u, v = (s['panel_a'][key] for key in ('u', 'v'))
a = m._affine(u)
ref_errors = []
for i in range(len(u)):
    eligible = np.flatnonzero(np.arange(len(u)) != i)
    rng = np.random.default_rng(8)
    best_score = np.inf
    for _ in range(400):
        sample = rng.choice(eligible, a.shape[1], replace=False)
        err = np.linalg.norm(a @ m._fit(a[sample], v[sample]) - v, axis=1)
        score = np.median(err[eligible])
        if score < best_score:
            best_score, best_err = score, err
    keep = eligible[best_err[eligible] < max(.001, 5*best_score)]
    for _ in range(3):
        pilot = m._fit(a[keep], v[keep])
        err = np.linalg.norm(a @ pilot - v, axis=1)
        keep = eligible[err[eligible] < max(.001, 5*np.median(err[eligible]))]
    forward = np.array([m._calibration_basis(u[i:i+1], j)[0] @ m._fit(m._calibration_basis(u[keep], j), v[keep,j]) for j in range(u.shape[1])])
    ref_errors.append([np.linalg.norm(a[i] @ pilot - v[i]), np.linalg.norm(forward - v[i])])
ref_errors = np.array(ref_errors)
print('A leave-one-row-out row error median / 90th pct (affine, cyclic):', np.quantile(ref_errors,[.5,.9],axis=0))
print('A held-out residuals > 1 (affine, cyclic):', (ref_errors > 1).sum(0))

z = m._calibrate(s['panel_a'], tr['x'])
b = s['panel_b']
bz = m._affine(z) @ m._fit(m._affine(b['u']), b['v'])
cu, cv = (s['panel_c'][key] for key in ('u', 'v1'))
forms = {
    'integral': (lambda q,p: q*hyp2f1(.5,np.exp(p[1]),1.5,-np.exp(p[0])*q*q), [0.,0.]),
    'rational': (lambda q,p: q*(1+p[1]*q*q)/(1+np.exp(p[0])*q*q), [0.,.2]),
    'integral+linear': (lambda q,p: q*hyp2f1(.5,np.exp(p[1]),1.5,-np.exp(p[0])*q*q)+p[2]*q, [0.,0.,0.]),
    'asinh+linear': (lambda q,p: np.arcsinh(np.exp(p[0])*q)+p[1]*q, [.4,0.]),
}
allc = np.arange(len(cu))
coords = {}
for name, (phi, start) in forms.items():
    def calibrate(rows, x):
        def design(u,p): return m._affine(phi(u,p))
        def residual(p):
            g = design(cu[rows], p)
            return (g @ m._fit(g, cv[rows]) - cv[rows]).ravel()
        p = least_squares(residual,start,bounds=(-4,4),max_nfev=200).x
        return design(x,p) @ m._fit(design(cu[rows],p),cv[rows])
    loo = np.vstack([calibrate(allc[allc != i],cu[i:i+1]) for i in allc])
    coords[name] = calibrate(allc,bz)
    print(f'C1 {name} leave-one-row-out RMSE {np.mean((loo-cv)**2)**.5:.6g}')

ids = np.arange(len(z))
folds = [ids[52:], ids[56:], np.argsort(np.linalg.norm(z,axis=1))[-16:]]
folds += list(np.array_split(np.random.default_rng(207).permutation(len(z)),4))
for name,q in coords.items():
    g = m._affine(q)
    pair = np.column_stack((g,q[:,:2]*q[:,2:]))
    quad = np.column_stack((g,*[q[:,i]*q[:,j] for i,j in combinations_with_replacement(range(q.shape[1]),2)]))
    # The full quadratic includes every squared linear sum/difference. Separately
    # test whether a paired sum or difference acts in a rational denominator.
    rows = []
    for te in folds:
        ti = np.setdiff1d(ids,te)
        y = tr['y'][ti]
        def predict_rational(sign):
            d = (q[:,:2]+sign*q[:,2:])**2
            def numerator(p): return m._fit(pair[ti]/(1+d[ti]@p)[:,None],y)
            def residual(p): return (pair[ti] @ numerator(p))/(1+d[ti]@p)-y
            p = least_squares(residual,[.01,.01],bounds=(0,10)).x
            return (pair[te] @ numerator(p))/(1+d[te]@p)
        yh = [d[te] @ m._fit(d[ti],y) for d in (pair,quad)]
        yh.extend(predict_rational(sign) for sign in (1,-1))
        ds = np.abs(m._state(g[ti],tr['traj'][ti],g[te],s['eval_times'])-tr['traj'][te])
        rows.append([*(np.mean(np.abs(yj-tr['y'][te])) for yj in yh),ds.mean(),ds.mean(0).max()])
    print(f'{name}: 52/20 and worst fold MAEs [paired, quadratic, rational-sum, rational-diff, traj, worst timestamp]',np.round(rows[0],6),np.round(np.max(rows,axis=0),6))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit03.py && OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`2683 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Study fields: ['eval_times', 'panel_a', 'panel_b', 'panel_c', 'telemetry_vec', 'test', 'train'] ; train fields: ['summary', 'traj', 'x', 'y']
Telemetry shape: (4,) ; distinct evaluation times: 8
A leave-one-row-out row error median / 90th pct (affine, cyclic): [[3.42141322e-01 2.17960566e-04]
 [6.17926735e-01 3.51274003e-04]]
A held-out residuals > 1 (affine, cyclic): [3 3]
C1 integral leave-one-row-out RMSE 0.0036386
C1 rational leave-one-row-out RMSE 0.00725741
C1 integral+linear leave-one-row-out RMSE 0.00113146
C1 asinh+linear leave-one-row-out RMSE 0.00280865
integral: 52/20 and worst fold MAEs [paired, quadratic, rational-sum, rational-diff, traj, worst timestamp] [0.008334 0.010009 0.008434 0.008215 0.006758 0.008025] [0.009433 0.010009 0.009433 0.009304 0.011105 0.013851]
rational: 52/20 and worst fold MAEs [paired, quadratic, rational-sum, rational-diff, traj, worst timestamp] [0.0152   0.015944 0.015    0.015212 0.01026  0.012916] [0.01999  0.022472 0.021243 0.021597 0.024171 0.031654]
integral+linear: 52/20 and worst fold MAEs [paired, quadratic, rational-sum, rational-diff, traj, worst timestamp] [0.004234 0.004942 0.004658 0.00424  0.005294 0.006999] [0.007112 0.008074 0.007509 0.007056 0.008049 0.011992]
asinh+linear: 52/20 and worst fold MAEs [paired, quadratic, rational-sum, rational-diff, traj, worst timestamp] [0.006865 0.008352 0.007122 0.006922 0.005958 0.007103] [0.007293 0.008352 0.007444 0.007266 0.00746  0.009621]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=331760124bfb30ff (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_33176012 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`37608 ms` LLM)
- **Rationale**: The cyclic calibration passed the independent panel audit, reducing median held-out row error from 0.342 to 0.000218 without masking the three outliers. Promote the C1 extension because it improves independent panel prediction and both downstream MAEs, then verify the exact saved solver against the public contract and stress slices.
- **Alternatives Weighed**: `Use asinh plus linear for C1` (rejected: It slightly improves the worst trajectory slice but has more than twice the panel holdout RMSE and a worse scalar readout on the required split.); `Add a rational denominator to the readout` (rejected: Once C1 is corrected, neither paired denominator materially improves the readout over the simpler paired-product fit.)
- **ATMS NOGOOD DECLARED**: Refuted `The integral-only coordinate transform is sufficient to capture the reproducible structure in C1.` | Witness: `Adding a linear term reduces C1 leave-one-row-out RMSE from 0.003639 to 0.001131 and reduces worst-fold readout and trajectory MAEs from 0.009433 and 0.011105 to 0.007112 and 0.008049.` | Blanket Diagnosis: `A systematic error in the shared C1 coordinate transform propagates into both output regressions.` | Sublated Pivot: `Retain the cyclic calibration, B operator, and separate output fits; add the linear term to C1 and verify the packaged solver.`

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/batch_solver.py')
s = p.read_text()
old = '''        scale, power = np.exp(log_params)
        return _affine(q * hyp2f1(.5, power, 1.5, -scale*q*q))'''
new = '''        scale, power = np.exp(log_params[:2])
        linear = log_params[2]
        return _affine(q * (hyp2f1(.5, power, 1.5, -scale*q*q) + linear))'''
assert s.count(old) == 1
s = s.replace(old, new)
old = 'least_squares(residual, [0., 0.], bounds=(-4., 4.)).x'
assert s.count(old) == 1
p.write_text(s.replace(old, 'least_squares(residual, [0., 0., 0.], bounds=(-4., 4.), max_nfev=200).x'))
PY
export OPENBLAS_NUM_THREADS=1
python3 /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f:
    s = pickle.load(f)[0]['study']
t = s['train']; x = t['x']; ids = np.arange(len(x))
z = m._calibrate(s['panel_a'], x)
folds = {'52/20': ids[52:], '56/16': ids[56:], 'high radius': np.argsort(np.linalg.norm(z,axis=1))[-16:]}
for seed in (103, 207):
    for i, te in enumerate(np.array_split(np.random.default_rng(seed).permutation(ids), 4)):
        folds[f'{seed}/{i+1}'] = te

def diagnostics(y, state):
    return np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])

o = m.solve_batch(s)
np.testing.assert_allclose(o['diagnostics_vec'], diagnostics(o['y_hat'], o['state_pred']), rtol=0, atol=1e-14)
results = []
for label, te in folds.items():
    ti = np.setdiff1d(ids, te)
    train = {k: t[k][ti] for k in ('x', 'y', 'traj')}
    train['summary'] = diagnostics(train['y'], train['traj'])
    o = m.solve_batch(dict(s, train=train, test={'x': x[te]}))
    dy = np.abs(o['y_hat'] - t['y'][te]); ds = np.abs(o['state_pred'] - t['traj'][te])
    row = np.array([dy.mean(), ds.mean(), ds.mean(axis=0).max()])
    assert np.all(row <= .038), (label, row)
    results.append(row)
    print(f'{label}: readout {row[0]:.6f}, trajectory {row[1]:.6f}, worst timestamp {row[2]:.6f}')
print('Worst across all folds:', np.max(results,axis=0))
PY
```
- **Container Exit Code**: `0` (`1389 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
52/20: readout 0.004234, trajectory 0.005294, worst timestamp 0.006999
56/16: readout 0.004108, trajectory 0.005218, worst timestamp 0.006670
high radius: readout 0.007112, trajectory 0.008049, worst timestamp 0.011992
103/1: readout 0.005760, trajectory 0.006648, worst timestamp 0.007833
103/2: readout 0.004498, trajectory 0.005219, worst timestamp 0.006918
103/3: readout 0.003829, trajectory 0.004991, worst timestamp 0.005652
103/4: readout 0.005400, trajectory 0.005086, worst timestamp 0.007080
207/1: readout 0.005872, trajectory 0.005625, worst timestamp 0.007610
207/2: readout 0.006031, trajectory 0.005843, worst timestamp 0.008050
207/3: readout 0.003849, trajectory 0.004967, worst timestamp 0.005498
207/4: readout 0.003844, trajectory 0.005091, worst timestamp 0.006880
Worst across all folds: [0.00711171 0.00804876 0.01199181]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=719ceec507da46a1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_719ceec5 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`33437 ms` LLM)
- **Rationale**: The outstanding evidence concerns the calibration stages, not the interface. Test the saved solver functions on omitted reference rows to check whether the cyclic inverse recovers known coordinates and whether the added C1 term improves panel predictions without access to the omitted row.
- **Alternatives Weighed**: `Repeat the public schema and stress checks` (rejected: The exact saved artifact already passed both; repeating them would not test whether either calibration stage generalizes on its own.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.special import hyp2f1

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']

a = s['panel_a']; u, v = a['u'], a['v']; ids = np.arange(len(u))
# Evaluate forward residuals at each omitted row's known reference coordinates.
# Use the fitted inverse only on omitted inliers; the three corrupted outputs
# are not measurements of the forward law.
affine_errors, inverse_errors = [], []
for i in ids:
    eligible = ids[ids != i]
    keep = eligible.copy()
    design = m._affine(u)
    for _ in range(5):
        w = m._fit(design[keep], v[keep])
        r = np.linalg.norm(design @ w - v, axis=1)
        keep = eligible[r[eligible] < 5 * np.median(r[eligible])]
    pred = np.array([m._calibration_basis(u[i:i+1], j)[0] @ m._fit(m._calibration_basis(u[keep], j), v[keep, j]) for j in range(u.shape[1])])
    if np.linalg.norm(pred - v[i]) > 1:
        continue
    panel = {k: values[eligible] for k, values in a.items()}
    est = m._calibrate(panel, v[i:i+1])[0]
    affine = np.linalg.solve(w[1:].T, v[i] - w[0])
    affine_errors.append(np.linalg.norm(affine - u[i]))
    inverse_errors.append(np.linalg.norm(est - u[i]))
print(f'Panel A omitted inlier rows: {len(inverse_errors)}; inverse median error affine={np.median(affine_errors):.8f}, delivered={np.median(inverse_errors):.8f}; delivered max={max(inverse_errors):.8f}')
assert len(inverse_errors) == 45 and max(inverse_errors) < .002

c = s['panel_c']; u, v = c['u'], c['v1']; ids = np.arange(len(u))
old, new = [], []
for i in ids:
    keep = ids != i
    panel = {'u': u[keep], 'v1': v[keep]}
    new.append(m._c1_coordinates(panel, u[i:i+1])[0])
    def design(q, p):
        scale, power = np.exp(p)
        return m._affine(q * hyp2f1(.5, power, 1.5, -scale*q*q))
    def residual(p):
        g = design(u[keep], p)
        return (g @ m._fit(g, v[keep]) - v[keep]).ravel()
    p = least_squares(residual, [0., 0.], bounds=(-4., 4.)).x
    old.append((design(u[i:i+1], p) @ m._fit(design(u[keep], p), v[keep]))[0])
r0 = np.mean((np.asarray(old)-v)**2)**.5
r1 = np.mean((np.asarray(new)-v)**2)**.5
print(f'Panel C1 leave-one-row-out RMSE: integral-only={r0:.8f}, delivered={r1:.8f}')
assert r1 < .5*r0
PY
```
- **Container Exit Code**: `0` (`2013 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Panel A omitted inlier rows: 45; inverse median error affine=0.28095911, delivered=0.00017172; delivered max=0.00044054
Panel C1 leave-one-row-out RMSE: integral-only=0.00363860, delivered=0.00113146

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=719ceec507da46a1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_719ceec5 (directory).
```

### Turn 12 — `submit_completed_task` (`14761 ms` LLM)
- **Rationale**: The saved solver passed the public validator and isolated-process stress tests, including the high-radius holdout. Tests on omitted reference rows independently support both calibration changes.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public validator passed. On the required 52/20 split, mean absolute error was 0.00423 for the scalar readout and 0.00529 for the trajectory, both below 0.038.`

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
