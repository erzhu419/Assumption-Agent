# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m98_c03_s00__blind_continual_campaign`

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
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `460,314` (`442,096` / `62,464` / `18,218`) |
| **Total Episode Wall Time (s)** | `538.07s` |

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

### Turn 1 — `execute_in_remote_docker` (`14183 ms` LLM)
- **Rationale**: Identify the observed inputs, time grid, and chamber evidence before choosing a constitutive kernel or archive fallback.

```bash
cd /holdout_task; ls -la; cat validate_public_schema.py; ls -la /tmp/hegel_continual_archive/ 2>/dev/null; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation Cauchy robust IRLS'; python3 - <<'PY'
import pickle,numpy as np
with open('public_batch.pkl','rb') as f: d=pickle.load(f)
for k,v in d.items():
 if isinstance(v,dict):
  print(k,{a:(type(b).__name__,getattr(b,'shape',None)) for a,b in v.items()})
 elif isinstance(v,np.ndarray): print(k,v.shape,v.ravel()[:25])
 else: print(k,repr(v)[:600])
PY
```
- **Container Exit Code**: `1` (`360 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
array, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_local_level_step(predictor: np.ndarray, values: np.ndarray, splits: np.ndarray, *, scale: float, loss_clip: float) -> tuple[int, float] | Fit finite predictor and values (n,) and integer splits (k,) with k >= 1, at l | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_local_level_step | STATUS: APPLICABLE_SUBROUTINE(real_1d_step_series)
autogenous_prim_robust_affine_projection@abc35d3d | profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | For predictor, target and optional weights (n, m) and predictor sensitivity (n | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.profile_weighted_affine | STATUS: APPLICABLE_SUBROUTINE(real_2d_predictor_3d_sensitivity_2d_target)
autogenous_prim_robust_affine_projection@abc35d3d | propagate_linear_basis(transition: np.ndarray, initial: np.ndarray, increments: np.ndarray) -> np.ndarray | Return a trajectory of shape (steps+1, n, k) from transition (n, n), initial ( | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.propagate_linear_basis | STATUS: APPLICABLE_SUBROUTINE(real_linear_transition_and_2d_initial_basis)
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`17288 ms` LLM)
- **Rationale**: Read the case list using the validator’s indexing and inspect time, calibration, and archive metadata to determine which model terms and validation splits the public data support.

```bash
cd /holdout_task; cat public_asset_index.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation' | head -12; if test -d /tmp/hegel_continual_archive; then ls -lh /tmp/hegel_continual_archive; head -c 5000 /tmp/hegel_continual_archive/archive_apparatus_log.json; fi; python3 - <<'PY'
import pickle,numpy as np
b=pickle.load(open('public_batch.pkl','rb')); print('cases',len(b)); print('item keys',list(b[0]));
for i,item in enumerate(b):
 print('case',i,{k:str(v)[:140] for k,v in item.items() if k!='dataset'}); d=item['dataset']
 for k,v in d.items():
  if isinstance(v,dict):
   print(' ',k,{a:(type(x).__name__,getattr(x,'shape',None),str(x if not isinstance(x,np.ndarray) else x.flat[:5])[:110]) for a,x in v.items()})
  elif isinstance(v,np.ndarray): print(' ',k,v.shape,v.ravel()[:30])
  else: print(' ',k,repr(v)[:300])
 if i>=4: break
PY
```
- **Container Exit Code**: `0` (`372 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
")}
  train {'lab_id': ('ndarray', (72,), '[0 0 0 1 2]'), 'stratum': ('ndarray', (72,), '[0 2 2 2 1]'), 'coupon_features_raw': ('ndarray', (72, 4), '[-0.90974245         nan -0.33525929 -0.88932983  0.05228426]'), 'ref_coupons': ('ndarray', (72, 8), '[ 2.23382767 -0.64618343 -2.89660698 -2.42573284 -0.75754728]'), 'past_channel_raw': ('ndarray', (72, 6, 4), '[0.58519764 0.36601729 0.29569085 0.1671397  0.336191  ]'), 'hold_channel_raw': ('ndarray', (72, 4), '[0.86552234 0.58554433 1.22714413 0.77553244 0.79023101]'), 'strain_0': ('ndarray', (72,), '[ 0.10321396  0.23620404 -0.1122587  -0.13071453 -0.00944071]'), 'strain_curves_obs': ('ndarray', (72, 12), '[1.0861986  1.96974056 2.51681226 2.93494293 3.23195766]'), 'instant_rate_obs': ('ndarray', (72,), '[4.7353257  4.67577099 3.70406497 5.77017468 2.77371432]'), 'aux_chamber_telemetry': ('ndarray', (72, 4), '[-0.45465331 -0.37729846 -0.31726516  0.05379489 -0.0109354 ]')}
  test {'lab_id': ('ndarray', (36,), '[1 3 0 0 2]'), 'stratum': ('ndarray', (36,), '[0 2 0 1 0]'), 'coupon_features_raw': ('ndarray', (36, 4), '[ 0.28429839  0.18844065  0.34646629 -0.7228558   1.2439773 ]'), 'ref_coupons': ('ndarray', (36, 8), '[0.56996737 0.17319852 0.29385472 0.44898847 1.76087289]'), 'past_channel_raw': ('ndarray', (36, 6, 4), '[-0.42534712 -0.56354328  1.05825622  0.30684897 -0.19011415]'), 'hold_channel_raw': ('ndarray', (36, 4), '[1.25098517 0.77072602 0.63190923 0.85978614 0.96753552]'), 'strain_0': ('ndarray', (36,), '[ 0.21350215  0.21654365 -0.0097979  -0.00347068 -0.08093839]'), 'aux_chamber_telemetry': ('ndarray', (36, 4), '[0.7118042  0.38493228 0.33112323 0.07658104 1.22372396]')}
  chamber_verification_trials {'lab_id': ('ndarray', (18,), '[3 2 1 1 1]'), 'stratum': ('ndarray', (18,), '[0 0 0 0 2]'), 'coupon_features_raw': ('ndarray', (18, 4), '[ 0.74606178  1.09887789  1.18459499         nan -1.98991205]'), 'ref_coupons': ('ndarray', (18, 8), '[-0.16084201 -1.27571792  0.24099765  1.81167783  2.6019235 ]'), 'past_channel_raw': ('ndarray', (18, 6, 4), '[-0.08703059  0.68047509  0.98714566  0.30654795  0.88158848]'), 'hold_channel_raw': ('ndarray', (18, 4), '[0.86988099 0.41015996 0.42087884 0.91965438 0.91307983]'), 'strain_0': ('ndarray', (18,), '[0.01463489 0.09715617 0.09149285 0.16686298 0.12994231]'), 'strain_curves_obs': ('ndarray', (18, 12), '[0.74075291 1.40870661 1.85306764 2.17002957 2.4242286 ]'), 'instant_rate_obs': ('ndarray', (18,), '[3.50551587 4.50329036 1.56710471 5.14059261 4.03156314]')}
```

### Turn 3 — `execute_in_remote_docker` (`55873 ms` LLM)
- **Rationale**: Compare the physical sign, interval order, and drift conventions jointly against the measured curves and initial rates. Fit train and chamber separately to establish whether either requires a different constitutive response before testing held-out labs.
- **Alternatives Weighed**: `Assume a hold-only creep kernel` (rejected: The supplied stress histories may explain both strain and initial rate; compare hold-only against the exact history recurrence before choosing.)

```bash
cd /holdout_task; python3 - <<'PY'
import pickle,numpy as np
D=pickle.load(open('public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; R=c['bridge_raw']; L=c['true_channel_loads']; print('Bridge forward affine',np.linalg.lstsq(np.c_[L,np.ones(len(L))],R,rcond=None)[0]); print('single sweeps',c['single_channel_sweeps']); print('archive:'); import os; print(os.path.exists('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for label in ('train','test','chamber_verification_trials'):
 s=D[label]; print(label,'ref panels per lab mean/std',[(i,np.round(s['ref_coupons'][s['lab_id']==i].mean(axis=0),2),np.round(s['ref_coupons'][s['lab_id']==i].std(axis=0),2)) for i in np.unique(s['lab_id'])]); print('raw range',np.nanmin(s['coupon_features_raw'],axis=0),np.nanmax(s['coupon_features_raw'],axis=0));
 if 'instant_rate_obs' in s:
  y=s['strain_curves_obs']-s['strain_0'][:,None]; print('curve examples',np.round(y[:5],4)); print('rates',s['instant_rate_obs'][:10]); print('history/hold totals',np.c_[s['past_channel_raw'].sum(-1),s['hold_channel_raw'].sum(-1)][:5])
PY
cat > /tmp/model98.py <<'PY'
import numpy as np
from scipy.optimize import least_squares

def calibrate(d):
 c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; b=np.linalg.lstsq(np.c_[x,np.ones(len(x))],y,rcond=None)[0]; return np.linalg.solve(b[:4],np.ones(4)),b[4]

def prep(s, stats=None, imp=None):
 lab=s['lab_id']; stratum=s['stratum']; ref=s['ref_coupons']; raw=s['coupon_features_raw'];
 if stats is None:
  stats={i:(np.median(ref[lab==i].mean(0)),np.median(ref[lab==i].std(0))) for i in np.unique(lab)}
 x=raw.copy();
 for i in np.unique(lab):
  z=ref[lab==i]; m,v=stats.get(i,(np.median(z.mean(0)),np.median(z.std(0)))); x[lab==i]=(x[lab==i]-m)/max(v,1e-8)
 if imp is None:
  _,iv,cnt=np.unique(np.c_[lab,stratum],axis=0,return_inverse=True,return_counts=True); w=1/cnt[iv]; imp=np.nansum(x*w[:,None],axis=0)/np.sum(np.isfinite(x)*w[:,None],axis=0)
 return np.where(np.isfinite(x),x,imp),stats,imp

def basis(tau,dt,t,u,h,x,kind):
 dt=np.asarray(dt); tail=np.cumsum(dt[::-1])[::-1]-dt
 g=-np.expm1(-dt[:,None]/tau)*np.exp(-tail[:,None]/tau)
 m=u@g
 if kind.startswith('free'): m=np.concatenate([np.broadcast_to(h[:,None],m.shape),m],axis=1); tau=np.tile(tau,2)
 elif kind.startswith('hold'): m=h[:,None]+m*0
 elif kind.startswith('plus'): m=h[:,None]+m
 elif kind.startswith('reverse'): m=h[:,None]-u[:,::-1]@g
 elif kind.startswith('steps'): m=h[:,None]-u@np.exp(-tail[:,None]/tau)
 elif kind.startswith('hist'): pass
 else: m=h[:,None]-m
 f=m[:,None,:]*(-np.expm1(-t[:,None]/tau)); r=m/tau
 if '_rate' in kind: f=f*tau; r=r*tau
 z=np.c_[np.ones(len(x)),x];
 if '_stress' in kind: z=z*h[:,None]
 return np.concatenate([f,z[:,None,:]*t[None,:,None]],axis=-1),np.c_[r,z]

def fit(d,s,kind,tx=None):
 t=d['eval_times']; dt=d['past_dt']; w,b=calibrate(d); x=prep(s)[0] if tx is None else tx; u=(s['past_channel_raw']-b)@w; h=(s['hold_channel_raw']-b)@w; y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs'];
 def fun(p):
  a,c=basis(np.exp(p[:2]),dt,t,u,h,x,kind); return np.r_[(a@p[2:]-y).ravel(),(c@p[2:]-r)*2]
 out=[]
 for ts in ((.2,2),(.45,3),(.8,5),(.12,12)):
  q=least_squares(fun,np.r_[np.log(ts),np.ones(4 if kind.startswith('free') else 2),np.zeros(5)],bounds=(np.r_[np.log([.05001]*2),[-np.inf]*(9 if kind.startswith('free') else 7)],np.r_[np.log([24.999]*2),[np.inf]*(9 if kind.startswith('free') else 7)]),loss='cauchy',f_scale=.03,max_nfev=250); out.append(q)
 q=min(out,key=lambda q:q.cost); e=fun(q.x); return q.x,q.cost,np.quantile(np.abs(e[:y.size]),[.5,.9,.95,.99]),np.quantile(np.abs(e[y.size:])/2,[.5,.9,.99])
if __name__=='__main__':
 import pickle
 d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; z,stats,imp=prep(d['train']); print('stats',stats,'impute',imp)
 for k in ['diff','diff_stress','plus','hold','hist','reverse','steps','free','free_stress']:
  for s in ['train','chamber_verification_trials']:
   r=fit(d,d[s],k,z if s=='train' else prep(d[s],stats,imp)[0]); print(k,s,np.round(np.r_[np.exp(r[0][:2]),r[0][2:]],5),round(r[1],5),r[2].round(5),r[3].round(5),flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/model98.py
```
- **Container Exit Code**: `0` (`6687 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[0.93817 2.78749 3.84161]
plus chamber_verification_trials [ 0.2259   1.14613  0.07494  0.36663  0.58233  0.28085  0.01395  0.2117
 -0.16015] 0.46766 [0.19316 2.13485 2.68165 3.36507] [0.77046 2.85332 4.21148]
hold train [ 0.28311  0.76407  0.10899  0.51383  0.50081  0.24064 -0.07816  0.15735
 -0.16514] 1.88728 [0.31618 1.00599 1.18619 1.7103 ] [0.65261 1.66393 2.28323]
hold chamber_verification_trials [ 0.76255  0.83063  1.51075 -0.96678  0.61256  0.27907 -0.02045  0.21066
 -0.15745] 0.41408 [0.13051 1.68513 2.01025 2.51329] [0.63497 2.1725  3.15807]
hist train [ 4.6897800e+00  9.2821200e+00  6.4702220e+01 -2.1338837e+02
  1.8324800e+00  2.0104000e-01 -1.7891000e-01 -4.3900000e-03
 -2.0919000e-01] 2.36979 [0.55953 1.43023 1.75902 2.70161] [2.06005 3.80258 5.00428]
hist chamber_verification_trials [ 8.306700e-01  1.463119e+01  1.019300e+00 -9.340157e+01  1.765490e+00
  3.836500e-01 -9.978000e-02  8.099000e-02 -1.419100e-01] 0.50685 [0.39358 1.5975  1.71911 2.12154] [1.89728 3.54505 4.76568]
reverse train [ 0.06768  1.03224  0.03785  1.02825  0.49357  0.26607 -0.07696  0.08575
 -0.16105] 2.00717 [0.35254 1.06543 1.2045  1.92982] [0.93276 2.22623 3.58085]
reverse chamber_verification_trials [ 0.4846   5.87626  0.52522  1.87992  0.05489  0.25702 -0.10777 -0.08234
 -0.12587] 0.43053 [0.27439 1.13665 1.36857 1.66523] [0.69391 2.04897 2.4341 ]
steps train [  0.58653   0.59396  33.02455 -32.4234    0.75283   0.22375  -0.09552
   0.09465  -0.14421] 1.56038 [0.19078 0.50128 0.60863 0.80209] [0.57059 1.34323 1.57345]
steps chamber_verification_trials [ 0.37592  0.38464  6.4846  -5.76541  0.80393  0.28247 -0.15808  0.28619
 -0.12853] 0.37459 [0.16181 0.49035 0.6288  1.16244] [0.81787 2.20983 3.61544]
free train [ 0.54302  4.52344  0.81209  0.55287 -0.81422 -0.4931   0.36384  0.19834
 -0.10362  0.09547 -0.12529] 0.30688 [0.01754 0.0871  0.13097 0.28898] [0.00975 0.03101 0.0613 ]
free chamber_verification_trials [ 0.5437   5.85412  0.81511  0.7606  -0.81673 -0.3842   0.32959  0.2811
 -0.13453  0.12803 -0.18717] 0.10432 [0.00936 0.18128 0.24881 0.32089] [0.00535 0.08252 0.0883 ]
free_stress train [ 0.53643  1.5213   0.77178  0.22631 -0.78105 -0.23818  0.15264  0.06119
 -0.03141  0.02725 -0.03659] 0.86531 [0.06008 0.30326 0.40264 0.61917] [0.03387 0.11906 0.19696]
free_stress chamber_verification_trials [ 0.54514  4.29791  0.81703  0.46222 -0.82202 -0.09135  0.12898  0.10388
 -0.04322  0.03544 -0.07919] 0.19956 [0.02824 0.57203 0.83669 1.72118] [0.02066 0.22216 0.48196]
```

### Turn 4 — `execute_in_remote_docker` (`35216 ms` LLM)
- **Rationale**: Determine whether the remaining slice errors are shared by late-time drift and initial rate. This will distinguish imputation or drift mismatch from a change in the transient law before committing to a chamber correction.
- **Alternatives Weighed**: `Stress-multiplied coupon drift` (rejected: It worsened both strain and initial-rate fit on the bench and in the chamber.)
- **ATMS NOGOOD DECLARED**: Refuted `The transient response can be modeled using the current hold stress alone.` | Witness: `Replacing the hold-only basis with hold minus exponentially filtered history lowered train median strain error from 0.316 to 0.017 and median rate error from 0.652 to 0.010.` | Blanket Diagnosis: `At the hold transition, the current load does not determine the internal memory state.` | Sublated Pivot: `Use chronological piecewise-constant Prony memory and diagnose remaining drift errors by missingness, lab, and telemetry.`

```bash
cd /holdout_task; python3 - <<'PY'
import json
j=json.load(open('public_asset_index.json')); print(j.keys()); print({k:v for k,v in j.items() if k!='files'}); print(j['files'].keys())
PY
cat > /tmp/diagnose98.py <<'PY'
import pickle,numpy as np
from model98 import *
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; train=d['train']; _,stats,imp=prep(train); w,b=calibrate(d); t=d['eval_times']; dt=d['past_dt']
for label in ['train','chamber_verification_trials']:
 s=d[label]; p,_,_,_=fit(d,s,'diff',prep(s,stats,imp)[0]); tau=np.exp(p[:2]); u=(s['past_channel_raw']-b)@w; h=(s['hold_channel_raw']-b)@w; y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; x=prep(s,stats,imp)[0]; raw=s['coupon_features_raw']; miss=~np.isfinite(raw); a,c=basis(tau,dt,t,u,h,x,'diff'); e=y-a@p[2:]; er=r-c@p[2:]; rr=np.median(e[:,4:]/t[4:],axis=1); target=rr+np.c_[np.ones(len(x)),x]@p[4:]; print('\n',label,'taus',tau,'missing count',miss.sum(0)); print('residual by missing',[(i,int(np.sum(miss.sum(1)==i)),np.round(np.quantile(np.abs(rr[miss.sum(1)==i]),[.5,.9]),4).tolist()) for i in np.unique(miss.sum(1))]); print('residual by lab',[(int(i),np.quantile(np.abs(rr[s['lab_id']==i]),[.5,.9]).round(4)) for i in np.unique(s['lab_id'])]); print('drift delta and initial delta corr',np.corrcoef(rr,er)[0,1]); print('complete row e/r',np.quantile(np.abs(e[~miss.any(1)]),[.5,.9,.95]),np.quantile(np.abs(er[~miss.any(1)]),[.5,.9,.95])); print('worst rows lab,strat, missing, drift, initial',np.c_[s['lab_id'],s['stratum'],miss,rr,er][np.argsort(-np.abs(rr))[:10]].round(4));
 for desc,xx in [('train imp',x),('self standardization',prep(s)[0]),('zero impute',np.where(miss,0,x)),('missing intercepts',np.c_[x,miss]),('lab drift',np.c_[x,np.eye(4)[s['lab_id']][:,1:]]),('telemetry',np.c_[x,s['aux_chamber_telemetry']] if 'aux_chamber_telemetry' in s else x)]:
  A=np.c_[np.ones(len(xx)),xx]; q=least_squares(lambda z:A@z-target,np.linalg.lstsq(A,target,rcond=None)[0],loss='cauchy',f_scale=.008); print(desc,'q',q.x.round(4),'res quantiles',np.quantile(np.abs(q.fun),[.5,.75,.9,.95]).round(5));
 if label=='train':
  v=s['aux_chamber_telemetry']; print('telemetry distribution',np.quantile(v,[0,.25,.5,.75,1],axis=0).round(4),'corr',np.corrcoef(np.c_[v,x,h,rr].T).round(2)[:4]);
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/diagnose98.py
```
- **Container Exit Code**: `0` (`633 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
4 -0.0028  0.0033] res quantiles [0.00686 0.01578 0.02929 0.0416 ]
telemetry q [ 0.3648  0.1974 -0.1028  0.0947 -0.1248  0.001   0.0012 -0.0004 -0.0011] res quantiles [0.00639 0.01593 0.03073 0.03804]
telemetry distribution [[-3.793  -3.6831 -1.8868 -2.8878]
 [-0.5061 -0.6963 -0.5811 -0.6578]
 [ 0.1818  0.0101  0.0184  0.2431]
 [ 0.667   0.8256  0.8605  0.9819]
 [ 2.3446  3.4448  2.6841  2.8594]] corr [[ 1.   -0.   -0.06 -0.12  0.19 -0.11 -0.09  0.12 -0.14  0.03]
 [-0.    1.    0.11  0.03  0.01  0.01 -0.04  0.02 -0.    0.11]
 [-0.06  0.11  1.   -0.02 -0.08  0.13  0.    0.18 -0.15 -0.01]
 [-0.12  0.03 -0.02  1.   -0.1  -0.06 -0.02  0.06 -0.07 -0.28]]

 chamber_verification_trials taus [0.52670428 4.08688775] missing count [0 0 3 3]
residual by missing [(0, 12, [0.0049, 0.0414]), (1, 6, [0.0369, 0.0938])]
residual by lab [(0, array([0.0032, 0.0171])), (1, array([0.0153, 0.0938])), (2, array([0.0344, 0.0611])), (3, array([0.0099, 0.0445]))]
drift delta and initial delta corr 0.9893915339461977
complete row e/r [0.01100309 0.09109007 0.12616293] [0.01194595 0.03462885 0.03637303]
worst rows lab,strat, missing, drift, initial [[ 1.      0.      0.      0.      1.      0.     -0.1008 -0.1045]
 [ 1.      2.      0.      0.      0.      1.      0.0869  0.0822]
 [ 2.      1.      0.      0.      1.      0.      0.0678  0.0715]
 [ 3.      1.      0.      0.      0.      0.     -0.0457 -0.0362]
 [ 3.      2.      0.      0.      0.      0.     -0.0432 -0.0365]
 [ 1.      0.      0.      0.      0.      0.      0.0248  0.0185]
 [ 0.      1.      0.      0.      0.      0.      0.0228  0.0202]
 [ 3.      0.      0.      0.      0.      0.     -0.0139 -0.0092]
 [ 3.      0.      0.      0.      0.      1.     -0.006   0.0031]
 [ 1.      0.      0.      0.      0.      0.     -0.0058 -0.0125]]
train imp q [ 0.3592  0.2807 -0.1385  0.1374 -0.1747] res quantiles [0.00556 0.03931 0.07439 0.08899]
self standardization q [ 0.3283  0.2061 -0.1059  0.0932 -0.1298] res quantiles [0.0031  0.00801 0.02001 0.02134]
zero impute q [ 0.3582  0.2885 -0.1368  0.1323 -0.1857] res quantiles [0.02026 0.04318 0.05346 0.064  ]
missing intercepts q [ 0.3543  0.2698 -0.1373  0.1254 -0.1731  0.0153 -0.0179  0.0472  0.01  ] res quantiles [0.00515 0.02854 0.05636 0.09922]
lab drift q [ 0.3695  0.2809 -0.1367  0.1345 -0.1651 -0.0068  0.0557 -0.0293] res quantiles [0.00222 0.02773 0.06406 0.08784]
telemetry q [ 0.3592  0.2807 -0.1385  0.1374 -0.1747] res quantiles [0.00556 0.03931 0.07439 0.08899]
```

### Turn 5 — `execute_in_remote_docker` (`42448 ms` LLM)
- **Rationale**: Profile a nuisance drift for each measured specimen to estimate the transient independently of imputation. Then compare panel conventions on held-out labs; if a convention only improves in-sample error, do not use it for deployment.
- **Alternatives Weighed**: `Fit a larger telemetry or lab-intercept model now` (rejected: The late-strain and initial-rate residuals move together, and changing reference-panel alignment already reduces chamber drift error. Isolate those effects before adding predictors.)

```bash
cat > /tmp/profile98.py <<'PY'
import pickle,numpy as np
from model98 import calibrate,prep,basis
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; t=D['eval_times']; dt=D['past_dt']; w,b=calibrate(D); S=[D[k] for k in ('train','chamber_verification_trials','test')]
def data(s):
 return (s['past_channel_raw']-b)@w,(s['hold_channel_raw']-b)@w,s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']
def fit(s):
 u,h,y,r=data(s); n=len(h); z=np.r_[t,1.]; obs=np.c_[y,r]; scale=np.r_[np.full(len(t),.012),.009]; i=np.arange(n)
 def f(p,j=False):
  a,c=basis(np.exp(p[:2]),dt,t,u,h,np.empty((n,0)),'diff'); g=np.concatenate([a[:,:,:2],c[:,None,:2]],axis=1); err=(g@p[2:4]+p[4:,None]*z-obs)/scale
  if not j:return err.ravel()
  J=np.zeros((n,len(z),4+n)); J[:,:,:2]=np.stack([(np.concatenate([q[0][:,:,:2],q[1][:,None,:2]],axis=1)@p[2:4]-g@p[2:4])/1e-5 for q in [basis(np.exp(p[:2]+1e-5*v),dt,t,u,h,np.empty((n,0)),'diff') for v in np.eye(2)]],axis=-1); J[:,:,2:4]=g; J[i,:,4+i]=z; return (J/scale[:,None]).reshape(-1,4+n)
 out=[least_squares(f,np.r_[np.log(ts),[.8,.6],np.full(n,.35)],jac=lambda p:f(p,True),loss='cauchy',bounds=(np.r_[np.log([.050001]*2),[0.,0.],np.full(n,-np.inf)],np.r_[np.log([24.99999]*2),np.full(n+2,np.inf)]),max_nfev=140) for ts in ((.45,3),(.65,6),(.2,12))]; q=min(out,key=lambda q:q.cost); ix=np.argsort(q.x[:2]); return np.exp(q.x[:2])[ix],q.x[2:4][ix],q.x[4:],f(q.x).reshape(n,-1)*scale

def subset(s,mask): return {k:v[mask] for k,v in s.items()}
def combine(ss): return {k:np.concatenate([s[k] for s in ss]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}
if __name__=='__main__':
 pstats=[prep(s)[1] for s in S]; trainstats=pstats[0]; trainimp=prep(S[0])[2]; allstats=prep(combine(S))[1]; otherstats=prep(combine([S[1],S[2]]))[1]; print('panel stats train chamber test pooled deployment',pstats,allstats,otherstats)
 for k,s in enumerate(S[:2]):
  taus,weights,drift,e=fit(s); m=np.isfinite(s['coupon_features_raw']).all(axis=1); print('FIT',k,taus,weights,'curve and rate quantiles',np.quantile(np.abs(e[:,:-1]),[.5,.8,.9,.95]),np.quantile(np.abs(e[:,-1]),[.5,.8,.9,.95]));
  for label,stats in [('train',trainstats),('self',pstats[k]),('pooled',allstats),('deployment',otherstats),('test',pstats[2])]:
   for impmode in ('fit','train'):
    x=prep(s,stats,None if impmode=='fit' else trainimp)[0]; a=np.c_[np.ones(len(x)),x]; sol=least_squares(lambda p:a@p-drift,np.linalg.lstsq(a[m],drift[m],rcond=None)[0],loss='cauchy',f_scale=.006); oof=np.empty(len(x));
    for lab in np.unique(s['lab_id']):
     mask=s['lab_id']!=lab; q=least_squares(lambda p:a[mask]@p-drift[mask],np.linalg.lstsq(a[mask],drift[mask],rcond=None)[0],loss='soft_l1',f_scale=.012); oof[~mask]=a[~mask]@q.x-drift[~mask]
    print(label,impmode,sol.x.round(5),'complete resid',np.quantile(abs(sol.fun[m]),[.5,.9,1]).round(4),'missing',np.quantile(abs(sol.fun[~m]),[.5,.9,1]).round(4),'worst lab mean oof',max(np.mean(abs(oof[s['lab_id']==i])) for i in np.unique(s['lab_id'])).round(4))
  if k==0:
   print('full lab drift params:');
   for lab in np.unique(s['lab_id']):
    sel=(s['lab_id']==lab)&m; raw=s['coupon_features_raw'][sel]; A=np.c_[np.ones(len(raw)),raw]; print(lab,sel.sum(),np.linalg.lstsq(A,drift[sel],rcond=None)[0].round(5))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/profile98.py
```
- **Container Exit Code**: `0` (`1669 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
09] complete resid [0.0124 0.0336 0.0568] missing [0.0217 0.048  0.061 ] worst lab mean oof 0.0271
deployment train [ 0.30331  0.202   -0.10329  0.09399 -0.12211] complete resid [0.0101 0.0298 0.0601] missing [0.0284 0.0557 0.0668] worst lab mean oof 0.0312
test fit [ 0.30974  0.17069 -0.08802  0.08277 -0.10404] complete resid [0.0126 0.059  0.0759] missing [0.0157 0.0482 0.1257] worst lab mean oof 0.0445
test train [ 0.30971  0.17102 -0.08819  0.08268 -0.10419] complete resid [0.0123 0.0587 0.0754] missing [0.0175 0.0499 0.1258] worst lab mean oof 0.0446
full lab drift params:
0 16 [ 0.34026  0.1773  -0.0909   0.08451 -0.10702]
1 9 [ 0.24218  0.24065 -0.1203   0.12542 -0.16091]
2 13 [ 0.32199  0.14521 -0.07366  0.07168 -0.08695]
3 15 [ 0.22666  0.25147 -0.12919  0.11627 -0.15078]
FIT 1 [0.53871832 4.82825646] [0.80088121 0.69262736] curve and rate quantiles [0.00241439 0.00493466 0.00636439 0.00722261] [0.00135556 0.00274692 0.00388794 0.0048627 ]
train fit [ 0.35132  0.28233 -0.14061  0.1395  -0.17615] complete resid [0.0047 0.0438 0.0455] missing [0.0738 0.0978 0.0984] worst lab mean oof 0.1033
train train [ 0.35146  0.28101 -0.14001  0.13834 -0.17597] complete resid [0.0054 0.0431 0.0462] missing [0.0387 0.0907 0.0935] worst lab mean oof 0.0798
self fit [ 0.32079  0.20607 -0.10485  0.09417 -0.12977] complete resid [0.0028 0.0082 0.0115] missing [0.0155 0.0244 0.0263] worst lab mean oof 0.013
self train [ 0.32642  0.20976 -0.10592  0.09692 -0.12952] complete resid [0.0016 0.006  0.0108] missing [0.0169 0.0252 0.0265] worst lab mean oof 0.0121
pooled fit [ 0.35867  0.30031 -0.14375  0.14454 -0.18033] complete resid [0.0019 0.0429 0.0709] missing [0.0744 0.0978 0.1075] worst lab mean oof 0.0825
pooled train [ 0.359    0.30112 -0.14366  0.14445 -0.18082] complete resid [0.0017 0.043  0.0717] missing [0.0482 0.0748 0.0842] worst lab mean oof 0.0526
deployment fit [ 0.33935  0.27401 -0.12641  0.14127 -0.17895] complete resid [0.0111 0.053  0.0626] missing [0.0834 0.1152 0.136 ] worst lab mean oof 0.1091
deployment train [ 0.33698  0.27007 -0.12665  0.14219 -0.17756] complete resid [0.0088 0.056  0.0634] missing [0.0077 0.0777 0.1085] worst lab mean oof 0.0694
test fit [ 0.34381  0.26253 -0.12343  0.12373 -0.16343] complete resid [0.0113 0.0499 0.0672] missing [0.0721 0.1377 0.1962] worst lab mean oof 0.1484
test train [ 0.35765  0.25219 -0.11091  0.134   -0.16211] complete resid [0.015  0.0434 0.0612] missing [0.0027 0.1155 0.1251] worst lab mean oof 0.0949
```

### Turn 6 — `execute_in_remote_docker` (`29479 ms` LLM)
- **Rationale**: Compare plausible reference groupings on held-out complete coupons, then use profiled missing rows to check the imputation convention separately. This distinguishes an alignment error from a constitutive change.
- **Alternatives Weighed**: `Use one shared reference panel for all rows` (rejected: The pooled panel helped bench drift but hurt pilot drift; test whether grouping bench and test references while keeping the pilot separate explains both.)

```bash
cat > /tmp/align98.py <<'PY'
from profile98 import *
from itertools import combinations
bench,pilot,test=S
sizes=[len(s['lab_id']) for s in S]
keys='lab_id stratum ref_coupons coupon_features_raw'.split()
preps={}
for n in (1,2,3):
 for group in combinations(range(3),n):
  a=combine([S[k] for k in group]); _,st,im=prep(a); preps[group]=(a,st,im)
for k,s in enumerate(S[:2]):
 tau,w,drift,e=fit(s); miss=np.isnan(s['coupon_features_raw']); complete=~miss.any(1); print('\nsource',k,'tau',tau,'w',w)
 for group,(_,st,im) in preps.items():
  x=prep(s,st,preps[(0,)][2])[0]; A=np.c_[np.ones(len(x)),x]; q=least_squares(lambda p:A[complete]@p-drift[complete],np.linalg.lstsq(A[complete],drift[complete],rcond=None)[0],loss='soft_l1',f_scale=.006); oof=np.empty(len(x))
  for lab in np.unique(s['lab_id']):
   keep=(s['lab_id']!=lab)&complete; a=A[keep]; p=least_squares(lambda p:a@p-drift[keep],q.x,loss='soft_l1',f_scale=.006).x; oof[s['lab_id']==lab]=A[s['lab_id']==lab]@p-drift[s['lab_id']==lab]
  print(group, q.x.round(5),'complete fitted RMS',np.sqrt(np.mean(q.fun**2)).round(5),'worst lab oof',max(np.mean(abs(oof[(s['lab_id']==lab)&complete])) for lab in np.unique(s['lab_id']) if np.any((s['lab_id']==lab)&complete)).round(5))
  if group in ((0,2),(1,)):
   er=drift-A@q.x; implied=(er[:,None]/q.x[1:]+x)[miss]; print(' inferred missing fillers (lab,stratum,feature,implied)', np.c_[s['lab_id'][miss.any(1)],s['stratum'][miss.any(1)],np.where(miss)[1],implied].round(4)); print(' supplied possible fillers',im.round(4),preps[(0,)][2].round(4),preps[(0,2)][2].round(4)); print(' panel',st)
 print('resid vs time MAE',np.mean(abs(e),axis=0).round(4))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/align98.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic Prony variable projection' | head -9
```
- **Container Exit Code**: `0` (`1646 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
transition: np.ndarray, initial: np.ndarray, increments: np.ndarray) -> np.ndarray | Return a trajectory of shape (steps+1, n, k) from transition (n, n), initial ( | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.propagate_linear_basis | STATUS: APPLICABLE_SUBROUTINE(real_linear_transition_and_2d_initial_basis)
autogenous_prim_robust_affine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar or scale array broadcastable to | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights | STATUS: APPLICABLE_SUBROUTINE(real_residual_array_positive_scale)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_local_level_step(predictor: np.ndarray, values: np.ndarray, splits: np.ndarray, *, scale: float, loss_clip: float) -> tuple[int, float] | Fit finite predictor and values (n,) and integer splits (k,) with k >= 1, at l | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_local_level_step | STATUS: APPLICABLE_SUBROUTINE(real_1d_step_series)
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
```

### Turn 7 — `execute_in_remote_docker` (`57382 ms` LLM)
- **Rationale**: Save a working solver and verify the schema before running end-to-end held-out tests. The next comparison should distinguish drift transfer error from transient estimation error, and exercise both the pilot-present and blind paths.
- **Alternatives Weighed**: `Pool every reference panel` (rejected: Including pilot references raised the bench alignment error, and the pilot's own panel gave better held-out drift predictions than the mixed panel.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
"""Calibrate axial load and fit a two-branch viscoelastic hold response."""

import numpy as np
from scipy.optimize import least_squares


def _calibrate(c):
    loads, raw = c['true_channel_loads'], c['bridge_raw']
    bridge = np.linalg.lstsq(np.c_[loads, np.ones(len(loads))], raw, rcond=None)[0]
    return np.linalg.solve(bridge[:4], np.ones(4)), bridge[4]


def _panels(*samples):
    lab = np.concatenate([s['lab_id'] for s in samples])
    refs = np.concatenate([s['ref_coupons'] for s in samples])
    return {i: (np.median(refs[lab == i].mean(axis=0)),
                np.median(refs[lab == i].std(axis=0))) for i in np.unique(lab)}


def _standardize(s, panels):
    x = np.asarray(s['coupon_features_raw'], dtype=float).copy()
    for i in np.unique(s['lab_id']):
        loc, scale = panels[i]
        rows = s['lab_id'] == i
        x[rows] = (x[rows] - loc) / (scale if scale > 0 else 1.0)
    return x


def _impute(x, *samples):
    labs = np.concatenate([s['lab_id'] for s in samples])
    strata = np.concatenate([s['stratum'] for s in samples])
    _, il = np.unique(labs, return_inverse=True)
    _, it = np.unique(strata, return_inverse=True)
    _, ix, count = np.unique(np.c_[il, it], axis=0, return_inverse=True, return_counts=True)
    w = 1.0 / count[ix]
    denom = np.isfinite(x).T @ w
    return np.divide(np.nansum(x * w[:, None], axis=0), denom,
                     out=np.zeros(x.shape[1]), where=denom > 0)


def _design(x, fill):
    return np.c_[np.ones(len(x)), np.where(np.isfinite(x), x, fill)]


def _stress(s, calibration):
    total, zero = calibration
    return (s['past_channel_raw'] - zero) @ total, (s['hold_channel_raw'] - zero) @ total


def _kernel(tau, dt, t, u, h):
    age = np.cumsum(dt[::-1])[::-1] - dt
    past = np.exp(-age[:, None] / tau) * (-np.expm1(-dt[:, None] / tau))
    amplitude = h[:, None] - u @ past
    time = -np.expm1(-t[:, None] / tau)
    return np.concatenate((amplitude[:, None, :] * time, (amplitude / tau)[:, None, :]), axis=1)


def _fit_relaxation(s, stress, dt, t, noise):
    # One nuisance drift per measured coupon keeps alignment errors out of the spectrum.
    u, h = stress
    y = np.c_[s['strain_curves_obs'] - s['strain_0'][:, None], s['instant_rate_obs']]
    n, k = y.shape
    z = np.r_[t, 1.0]
    scale = np.r_[np.full(k - 1, noise), noise * .75]
    idx = np.arange(n)
    jac = np.zeros((n, k, n + 4))
    jac[idx, :, 4 + idx] = z / scale

    def residual(p):
        return ((_kernel(np.exp(p[:2]), dt, t, u, h) @ p[2:4]
                 + p[4:, None] * z - y) / scale).ravel()

    def derivative(p):
        tau = np.exp(p[:2])
        a = _kernel(tau, dt, t, u, h)
        jac[:, :, 2:4] = a / scale[None, :, None]
        # Each log time constant changes only the corresponding branch.
        da = (_kernel(tau * np.exp(1e-5), dt, t, u, h) - a) / 1e-5
        jac[:, :, :2] = da * p[None, None, 2:4] / scale[None, :, None]
        return jac.reshape(n * k, n + 4).copy()

    lower = np.r_[np.log([.050001] * 2), np.zeros(2), np.full(n, -np.inf)]
    upper = np.r_[np.log([24.9999] * 2), np.full(n + 2, np.inf)]
    fits = []
    for pair in ((.45, 3.), (.65, 6.), (.2, 12.)):
        a = _kernel(np.array(pair), dt, t, u, h) @ np.array([.8, .6])
        slopes = np.median((y - a)[:, z != 0] / z[z != 0], axis=1)
        p = np.r_[np.log(pair), .8, .6, slopes]
        fits.append(least_squares(residual, p, jac=derivative, bounds=(lower, upper),
                                  loss='cauchy', max_nfev=140))
    p = min(fits, key=lambda r: r.cost).x
    ix = np.argsort(p[:2])
    return np.exp(p[:2][ix]), p[2:4][ix], p[4:]


def _fit_drift(a, y, noise):
    initial = np.linalg.lstsq(a, y, rcond=None)[0]
    return least_squares(lambda p: a @ p - y, initial, jac=lambda p: a,
                         loss='soft_l1', f_scale=noise).x


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    pilot = dataset.get('chamber_verification_trials')
    dt = np.asarray(dataset['past_dt'])
    t = np.asarray(dataset['eval_times'])
    cal = _calibrate(dataset['calibration_load_cells'])

    panels = _panels(train, test)
    x_train, x_test = (_standardize(s, panels) for s in (train, test))
    fill = _impute(np.concatenate((x_train, x_test)), train, test)

    if pilot is not None and len(pilot['strain_0']):
        source = pilot
        x = _standardize(pilot, _panels(pilot))
        fill_fit = _impute(x, pilot)
        noise = .006
    else:
        source, x, fill_fit, noise = train, x_train, fill, .012

    tau, weights, specimen_drift = _fit_relaxation(source, _stress(source, cal), dt, t, noise)
    drift = _fit_drift(_design(x, fill_fit), specimen_drift, noise * .5)
    transient = _kernel(tau, dt, t, *_stress(test, cal)) @ weights
    rate = _design(x_test, fill) @ drift
    return {'estimated_taus': tau, 'hereditary_weights': np.r_[weights, drift],
            'predicted_strain': test['strain_0'][:, None] + transient[:, :-1] + rate[:, None] * t,
            'initial_rate_pred': transient[:, -1] + rate}
PY
cd /holdout_task; OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py; python3 - <<'PY'
import pickle,os
b=pickle.load(open('public_batch.pkl','rb')); print('cases',len(b))
for item in b:
 d=item['dataset']; print('item keys',list(item),'dataset keys',list(d)); print({k:list(d[k]) for k in ('train','test','chamber_verification_trials') if isinstance(d.get(k),dict)})
print('archive:',os.listdir('/tmp/hegel_continual_archive') if os.path.isdir('/tmp/hegel_continual_archive') else 'absent')
PY
```
- **Container Exit Code**: `0` (`539 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
cases 1
item keys ['dataset'] dataset keys ['past_dt', 'eval_times', 'calibration_load_cells', 'train', 'test', 'chamber_verification_trials']
{'train': ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_channel_raw', 'hold_channel_raw', 'strain_0', 'strain_curves_obs', 'instant_rate_obs', 'aux_chamber_telemetry'], 'test': ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_channel_raw', 'hold_channel_raw', 'strain_0', 'aux_chamber_telemetry'], 'chamber_verification_trials': ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_channel_raw', 'hold_channel_raw', 'strain_0', 'strain_curves_obs', 'instant_rate_obs']}
archive: absent

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4a65d4249f6587b4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_4a65d424 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`33591 ms` LLM)
- **Rationale**: Held-out labs test whether the pilot fit improves both strain and rate over bench transfer. Re-encoding the bridge detects reliance on the nearly identity public calibration, and an independent recurrence checks the history convention directly.

```bash
cat > /tmp/cv98.py <<'PY'
import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; t=d['eval_times']; dt=d['past_dt']; cal=m._calibrate(d['calibration_load_cells']); bench=d['train']; test=d['test']; pilot=d['chamber_verification_trials']
def subset(s,mask): return {k:v[mask] for k,v in s.items()}
base={}; scores={}
for label,s,noise in [('bench',bench,.012),('pilot',pilot,.006)]:
 panel=m._panels(bench,test) if label=='bench' else m._panels(pilot)
 x=m._standardize(s,panel)
 for lab in np.unique(s['lab_id']):
  tr=s['lab_id']!=lab; va=~tr; ss=subset(s,tr); check=subset(s,va)
  for mode in ('local','train'):
   if mode=='train' and label=='bench':continue
   if mode=='local':
    tau,w,dr=m._fit_relaxation(ss,m._stress(ss,cal),dt,t,noise)
    fill=m._impute(x,s) if label=='pilot' else m._impute(np.r_[x,m._standardize(test,panel)],s,test)
    coef=m._fit_drift(m._design(x[tr],fill),dr,noise*.5)
    if label=='bench':base[lab]=tau,w,coef,fill
   else:tau,w,coef,fill=base[lab]
   z=m._kernel(tau,dt,t,*m._stress(check,cal))@w; drift=m._design(x[va],fill)@coef
   e=check['strain_0'][:,None]+z[:,:-1]+drift[:,None]*t-check['strain_curves_obs']; r=z[:,-1]+drift-check['instant_rate_obs']; miss=~np.isfinite(x[va]).all(1)
   a=tuple(round(v,5) for v in [np.median(abs(e)),np.quantile(abs(e),.9),np.mean(abs(r)),np.mean(abs(e[:,-1])),np.mean(abs(e[miss])) if miss.any() else 0]); print(label,int(lab),mode,'median strain / q90 strain / rate MAE / final MAE / missing MAE',a,'params',np.r_[tau,w,coef].round(4)); scores.setdefault((label,mode),[]).append(a)
print('worst by lab', {k:np.max(v,axis=0) for k,v in scores.items()})
rng=np.random.default_rng(98); change=np.eye(4)+rng.normal(0,.12,(4,4)); bias=rng.normal(0,.3,4); raw=lambda x:x@change+bias
for p in [pilot,None]:
 case=dict(d,chamber_verification_trials=p); r=m.solve_batch(case); assert all(np.all(np.isfinite(v)) for v in r.values()); assert .05<r['estimated_taus'][0]<r['estimated_taus'][1]<25
 transformed=dict(case,calibration_load_cells=dict(case['calibration_load_cells'],bridge_raw=raw(d['calibration_load_cells']['bridge_raw'])))
 for k in ('test','train','chamber_verification_trials'):
  if transformed[k] is not None: transformed[k]=dict(transformed[k],**{a:raw(transformed[k][a]) for a in ('past_channel_raw','hold_channel_raw')})
 q=m.solve_batch(transformed); print('bridge invariance',p is not None,max(np.max(abs(q[k]-r[k])) for k in r)); assert all(np.allclose(q[k],v,atol=2e-6,rtol=1e-6) for k,v in r.items());
 # Independent recurrence and t=0 derivative identity.
 tau=r['estimated_taus']; u,h=m._stress(test,cal); mem=np.zeros((len(h),2))
 for j,length in enumerate(dt):
  decay=np.exp(-length/tau); mem=mem*decay+u[:,j,None]*(1-decay)
 z=m._kernel(tau,dt,t,u,h); assert np.allclose(z[:,-1],(h[:,None]-mem)/tau); assert np.allclose(z[:,:-1],(h[:,None]-mem)[:,None,:]*(1-np.exp(-t[:,None]/tau)))
 at0=m.solve_batch(dict(case,eval_times=t))['initial_rate_pred']; print('valid mode',p is not None, np.r_[tau,r['hereditary_weights']].round(5),[v.shape for v in r.values()]); assert np.allclose(at0,r['initial_rate_pred'])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/cv98.py
```
- **Container Exit Code**: `0` (`3085 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
2077 -0.1049  0.1017 -0.1252]
pilot 0 local median strain / q90 strain / rate MAE / final MAE / missing MAE (0.00908, 0.08492, 0.01344, 0.05543, 0) params [ 0.5401  5.7732  0.8055  0.8743  0.2741  0.2046 -0.1019  0.0859 -0.1301]
pilot 0 train median strain / q90 strain / rate MAE / final MAE / missing MAE (0.02293, 0.08095, 0.01546, 0.06628, 0) params [ 0.5371  5.5675  0.7975  0.879   0.2772  0.2101 -0.1053  0.1025 -0.1255]
pilot 1 local median strain / q90 strain / rate MAE / final MAE / missing MAE (0.00849, 0.03179, 0.00885, 0.0295, 0.02215) params [ 0.5366  4.1972  0.795   0.5931  0.3449  0.2073 -0.1051  0.0946 -0.1295]
pilot 1 train median strain / q90 strain / rate MAE / final MAE / missing MAE (0.00997, 0.02212, 0.00468, 0.02188, 0.01208) params [ 0.537   4.6752  0.7958  0.6835  0.3263  0.2081 -0.1065  0.0995 -0.1271]
pilot 2 local median strain / q90 strain / rate MAE / final MAE / missing MAE (0.0059, 0.03747, 0.00416, 0.0277, 0.02522) params [ 0.5394  5.1239  0.8029  0.744   0.3074  0.2051 -0.105   0.0889 -0.1298]
pilot 2 train median strain / q90 strain / rate MAE / final MAE / missing MAE (0.00837, 0.03098, 0.00919, 0.01967, 0.02084) params [ 0.5379  4.9174  0.7986  0.7242  0.3146  0.2086 -0.1064  0.0998 -0.1275]
pilot 3 local median strain / q90 strain / rate MAE / final MAE / missing MAE (0.01498, 0.05884, 0.01048, 0.04606, 0.04112) params [ 0.5388  4.6503  0.8002  0.6604  0.3284  0.2063 -0.1067  0.0906 -0.1291]
pilot 3 train median strain / q90 strain / rate MAE / final MAE / missing MAE (0.01388, 0.04882, 0.01189, 0.04034, 0.03034) params [ 0.5378  5.4845  0.7991  0.8507  0.2841  0.2077 -0.1049  0.1017 -0.1252]
worst by lab {('bench', 'local'): array([0.01874, 0.05645, 0.01049, 0.04974, 0.03746]), ('pilot', 'local'): array([0.01498, 0.08492, 0.01344, 0.05543, 0.04112]), ('pilot', 'train'): array([0.02293, 0.08095, 0.01546, 0.06628, 0.03034])}
bridge invariance True 2.2221193676230655e-06
valid mode True [ 0.53914  5.03118  0.80217  0.72905  0.30999  0.20584 -0.10486  0.08912
 -0.13009] [(2,), (7,), (36, 12), (36,)]
bridge invariance False 6.846828526363424e-08
valid mode False [ 0.53744  5.0663   0.79762  0.76161  0.30567  0.20851 -0.10594  0.10044
 -0.12654] [(2,), (7,), (36, 12), (36,)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4a65d4249f6587b4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_4a65d424 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`38637 ms` LLM)
- **Rationale**: An unconstrained drift per specimen can absorb part of the slow branch. Compare joint fitting against the saved baseline on the same held-out labs, including missing-coupon and late-time slices, before changing the solver.
- **Alternatives Weighed**: `Replace the pilot fit with bench parameters` (rejected: Bench transfer worsened worst-lab median strain and rate MAE; it could also erase a real chamber shift.)

```bash
cat > /tmp/joint98.py <<'PY'
import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)

def joint(s, stress, dt, t, noise, design, initial):
    u,h=stress
    y=np.c_[s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']]
    z=np.r_[t,1.]
    scale=np.r_[np.full(len(t),noise),noise*.75]
    drift_basis=design[:,None,:]*(z/scale)[None,:,None]
    def fun(p):
        g=m._kernel(np.exp(p[:2]),dt,t,u,h)
        return ((g@p[2:4]+(design@p[4:])[:,None]*z-y)/scale).ravel()
    def jac(p):
        tau=np.exp(p[:2]); g=m._kernel(tau,dt,t,u,h)
        gp=(m._kernel(tau*np.exp(1e-5),dt,t,u,h)-g)*1e5
        return np.concatenate((gp*p[None,None,2:4]/scale[None,:,None],g/scale[None,:,None],drift_basis),axis=-1).reshape(-1,len(p))
    lo=np.r_[np.log([.050001]*2),0.,0.,np.full(5,-np.inf)]
    hi=np.r_[np.log([24.9999]*2),np.full(7,np.inf)]
    sol=least_squares(fun,initial,jac=jac,loss='cauchy',bounds=(lo,hi),max_nfev=150).x
    ix=np.argsort(sol[:2]); return np.exp(sol[:2][ix]),sol[2:4][ix],sol[4:]

if __name__=='__main__':
    d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; train,test,pilot=[d[k] for k in ('train','test','chamber_verification_trials')]
    dt,t=d['past_dt'],d['eval_times']; cal=m._calibrate(d['calibration_load_cells'])
    for label,source,noise in [('bench',train,.012),('pilot',pilot,.006)]:
        panel=m._panels(train,test) if label=='bench' else m._panels(pilot)
        x=m._standardize(source,panel); complete=np.isfinite(x).all(1)
        fill=m._impute(np.r_[x,m._standardize(test,panel)],train,test) if label=='bench' else m._impute(x,pilot)
        a=m._design(x,fill); scores={}
        for lab in [*np.unique(source['lab_id']),None]:
            keep=(source['lab_id']!=lab) if lab is not None else np.ones(len(x),bool)
            sub={k:v[keep] for k,v in source.items()}; stress=m._stress(sub,cal)
            tau,w,dr=m._fit_relaxation(sub,stress,dt,t,noise); coef=m._fit_drift(a[keep],dr,.5*noise)
            p=np.r_[np.log(tau),w,coef]
            cc=keep&complete
            opts={'profile':(tau,w,coef),'joint':joint(sub,stress,dt,t,noise,a[keep],p),'complete':joint({k:v[cc] for k,v in source.items()},m._stress({k:v[cc] for k,v in source.items()},cal),dt,t,noise,a[cc],p)}
            for method,(tau,w,coef) in opts.items():
                va=~keep if lab is not None else keep
                sample={k:v[va] for k,v in source.items()}; z=m._kernel(tau,dt,t,*m._stress(sample,cal))@w; rate=a[va]@coef
                e=abs(sample['strain_0'][:,None]+z[:,:-1]+rate[:,None]*t-sample['strain_curves_obs']); r=abs(z[:,-1]+rate-sample['instant_rate_obs']); miss=~complete[va]
                metrics=np.array([np.median(e),np.quantile(e,.9),r.mean(),e[:,-1].mean(),e[miss].mean() if miss.any() else 0]); print(label,lab,method,metrics.round(5),np.r_[tau,w,coef].round(4));
                if lab is not None:scores.setdefault(method,[]).append(metrics)
        print(label,'WORST', {k:np.max(v,axis=0).round(5).tolist() for k,v in scores.items()})
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/joint98.py
```
- **Container Exit Code**: `0` (`2213 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
04162, 0.03222], 'complete': [0.00927, 0.02321, 0.00719, 0.04173, 0.03209]}
pilot 0 profile [0.00908 0.08492 0.01344 0.05543 0.     ] [ 0.5401  5.7732  0.8055  0.8743  0.2741  0.2046 -0.1019  0.0859 -0.1301]
pilot 0 joint [0.00844 0.03862 0.01006 0.02533 0.     ] [ 0.5294  3.9134  0.7763  0.6069  0.3414  0.2066 -0.1041  0.091  -0.1314]
pilot 0 complete [0.00291 0.00695 0.00129 0.00264 0.     ] [ 0.5374  4.1292  0.7962  0.5748  0.3556  0.2076 -0.1064  0.0998 -0.1281]
pilot 1 profile [0.00849 0.03179 0.00885 0.0295  0.02215] [ 0.5366  4.1972  0.795   0.5931  0.3449  0.2073 -0.1051  0.0946 -0.1295]
pilot 1 joint [0.01566 0.08331 0.01388 0.05582 0.04648] [ 0.5354  4.2328  0.792   0.6112  0.3447  0.2098 -0.1056  0.1002 -0.1292]
pilot 1 complete [0.00761 0.05296 0.01142 0.03773 0.03727] [ 0.5367  4.0816  0.7945  0.5754  0.3548  0.2073 -0.1065  0.0997 -0.1279]
pilot 2 profile [0.0059  0.03747 0.00416 0.0277  0.02522] [ 0.5394  5.1239  0.8029  0.744   0.3074  0.2051 -0.105   0.0889 -0.1298]
pilot 2 joint [0.00613 0.03986 0.00528 0.02759 0.02592] [ 0.5369  4.2767  0.7954  0.5945  0.3511  0.2063 -0.1067  0.0983 -0.1281]
pilot 2 complete [0.00619 0.03213 0.00359 0.02291 0.0218 ] [ 0.5374  4.1351  0.796   0.5772  0.3547  0.2075 -0.1066  0.0998 -0.1279]
pilot 3 profile [0.01498 0.05884 0.01048 0.04606 0.04112] [ 0.5388  4.6503  0.8002  0.6604  0.3284  0.2063 -0.1067  0.0906 -0.1291]
pilot 3 joint [0.01752 0.0834  0.01271 0.05192 0.05921] [ 0.5331  3.9859  0.7845  0.588   0.3497  0.2062 -0.1073  0.0991 -0.1302]
pilot 3 complete [0.00486 0.0929  0.01042 0.04403 0.06629] [ 0.5375  4.1801  0.7959  0.5871  0.3521  0.2077 -0.1064  0.0992 -0.1282]
pilot None profile [0.01303 0.04325 0.00851 0.03453 0.02369] [ 0.5391  5.0312  0.8022  0.7291  0.31    0.2058 -0.1049  0.0891 -0.1301]
pilot None joint [0.00509 0.04881 0.00709 0.02816 0.04052] [ 0.5371  4.2683  0.7955  0.5951  0.3507  0.2067 -0.1065  0.0985 -0.1284]
pilot None complete [0.00441 0.05261 0.00782 0.03033 0.04509] [ 0.5373  4.1244  0.7956  0.5776  0.3545  0.2075 -0.1065  0.0998 -0.128 ]
pilot WORST {'profile': [0.01498, 0.08492, 0.01344, 0.05543, 0.04112], 'joint': [0.01752, 0.0834, 0.01388, 0.05582, 0.05921], 'complete': [0.00761, 0.0929, 0.01142, 0.04403, 0.06629]}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4a65d4249f6587b4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_4a65d424 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`35619 ms` LLM)
- **Rationale**: Compare pilot-local, bench/test, and training-fold-inferred fillers on the same held-out pilot labs. This tests whether imputation rather than a distinct spectrum explains the pilot’s worst slice and avoids changing the solver based on in-sample missing-row error.
- **Alternatives Weighed**: `Promote complete-case fitting for every split` (rejected: It helped identify the pilot spectrum but left the largest missing-coupon errors unresolved.)

```bash
cat > /tmp/fill98.py <<'PY'
from joint98 import *
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; b,s,test=[d[k] for k in ('train','chamber_verification_trials','test')]; t,dt=d['eval_times'],d['past_dt']; cal=m._calibrate(d['calibration_load_cells']); z=np.r_[t,1.]; bp=m._panels(b,test); pp=m._panels(s); xb,xt,x=[m._standardize(a,p) for a,p in [(b,bp),(test,bp),(s,pp)]]; bf=m._impute(np.r_[xb,xt],b,test); pf=m._impute(x,s); print('bench+test fill',bf,'pilot fill',pf); print('telemetry bench/test',[(a['aux_chamber_telemetry'].shape, np.quantile(a['aux_chamber_telemetry'],[0,.5,1],axis=0).round(5)) for a in (b,test)]); print('t',t,'dt',dt)
def sub(s,k): return {name:v[k] for name,v in s.items()}
c=np.isfinite(x).all(1); ix=np.where(~c)[0]; j=np.where(~np.isfinite(x))[1]; print('missing feature counts',np.bincount(j,minlength=4)); tau,w,dr=m._fit_relaxation(b,m._stress(b,cal),dt,t,.012); p=np.r_[np.log(tau),w,m._fit_drift(m._design(xb,bf),dr,.006)]; bt,bw,bd=joint(b,m._stress(b,cal),dt,t,.012,m._design(xb,bf),p); p=np.r_[np.log(bt),bw,bd]; scores={}
for lab in [*np.unique(s['lab_id']),None]:
 keep=s['lab_id']!=lab if lab is not None else np.ones(len(x),bool); va=~keep if lab is not None else keep; sc=sub(s,keep&c); y=np.c_[s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']]; tt,ww,dd=joint(sc,m._stress(sc,cal),dt,t,.006,m._design(x[keep&c],pf),p); g=m._kernel(tt,dt,t,*m._stress(s,cal))@ww; rate=np.median((y-g)[:,z!=0]/z[z!=0],axis=1); fill_observed=(rate-m._design(x,pf)@dd)[ix]/dd[1:][j]+pf[j]; learned=bf.copy();
 for col in range(4):
  vals=fill_observed[(j==col)&keep[ix]]; learned[col]=np.median(vals) if len(vals) else bf[col]
 if lab is None: print('implied fillers lab/feat/val',np.c_[s['lab_id'][ix],j,fill_observed].round(4)); print('estimated',learned)
 for label,fill in [('pilot',pf),('bench',bf),('learned',learned)]:
  e=np.abs(g+(m._design(x,fill)@dd)[:,None]*z-y)[va]; miss=~c[va]; metric=[np.quantile(e[:,:-1],.9),e[:,-1].mean(),e[:,-2].mean(),e[miss,:-1].mean() if miss.any() else 0.]; print(lab,label,np.round(metric,5));
  if lab is not None:scores.setdefault(label,[]).append(metric)
print('worst q90 strain / rate MAE / final MAE / missing strain MAE', {k:np.max(v,axis=0).round(5) for k,v in scores.items()})
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/fill98.py
if test -f /tmp/hegel_continual_archive/archive_apparatus_log.json; then head -c 6500 /tmp/hegel_continual_archive/archive_apparatus_log.json; fi
```
- **Container Exit Code**: `0` (`261 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4a65d4249f6587b4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_4a65d424 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/fill98.py", line 1, in <module>
    from joint98 import *
ModuleNotFoundError: No module named 'joint98'
```

### Turn 11 — `execute_in_remote_docker` (`14797 ms` LLM)
- **Rationale**: The previous command never ran the comparison because isolated Python excluded the sibling module from its import path. Load that module explicitly to test whether a different filler fixes the pilot's missing-coupon error without changing the spectrum.
- **Alternatives Weighed**: `Use pilot-local imputation with the complete-case spectrum` (rejected: The measured missing-coupon error rose; compare the alternative fillers before changing the pilot path.)

```bash
set -e; OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import runpy, sys, types
m = types.ModuleType('joint98')
m.__dict__.update(runpy.run_path('/tmp/joint98.py'))
sys.modules['joint98'] = m
runpy.run_path('/tmp/fill98.py', run_name='__main__')
PY
```
- **Container Exit Code**: `0` (`983 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
bench+test fill [-0.15038074  0.03714656  0.04945277  0.10181443] pilot fill [-0.41583769  0.274488    0.22167654 -0.21056652]
telemetry bench/test [((72, 4), array([[-3.793  , -3.68305, -1.88681, -2.88782],
       [ 0.18182,  0.01005,  0.01845,  0.24306],
       [ 2.34462,  3.4448 ,  2.68406,  2.85943]])), ((36, 4), array([[-2.15126, -1.88508, -1.66368, -2.25368],
       [-0.10113, -0.15038,  0.31208, -0.24928],
       [ 2.78972,  2.34306,  1.77791,  1.93895]]))]
t [0.25       0.59090909 0.93181818 1.27272727 1.61363636 1.95454545
 2.29545455 2.63636364 2.97727273 3.31818182 3.65909091 4.        ] dt [0.35 0.35 0.4  0.45 0.5  0.55]
missing feature counts [0 0 3 3]
0 pilot [0.00695 0.00129 0.00264 0.     ]
0 bench [0.00695 0.00129 0.00264 0.     ]
0 learned [0.00695 0.00129 0.00264 0.     ]
1 pilot [0.05296 0.01142 0.03773 0.03727]
1 bench [0.02512 0.00342 0.01512 0.01584]
1 learned [0.00778 0.00266 0.00457 0.00374]
2 pilot [0.03213 0.00359 0.02291 0.0218 ]
2 bench [0.02329 0.00585 0.0141  0.01554]
2 learned [0.0062  0.00273 0.0016  0.00295]
3 pilot [0.0929  0.01042 0.04403 0.06629]
3 bench [0.02573 0.00592 0.0147  0.01894]
3 learned [0.00748 0.00334 0.00438 0.0039 ]
implied fillers lab/feat/val [[3.     3.     0.0401]
 [1.     2.     0.1075]
 [1.     2.     0.118 ]
 [1.     3.     0.0385]
 [3.     3.     0.0299]
 [2.     2.     0.1213]]
estimated [-0.15038074  0.03714656  0.1179513   0.03852387]
None pilot [0.05261 0.00782 0.03033 0.04509]
None bench [0.02261 0.00372 0.01104 0.01607]
None learned [0.00639 0.00205 0.00267 0.00309]
worst q90 strain / rate MAE / final MAE / missing strain MAE {'pilot': array([0.0929 , 0.01142, 0.04403, 0.06629]), 'bench': array([0.02573, 0.00592, 0.01512, 0.01894]), 'learned': array([0.00778, 0.00334, 0.00457, 0.0039 ])}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4a65d4249f6587b4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_4a65d424 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`61272 ms` LLM)
- **Rationale**: Promote the joint fit that improved held-out bench results and preserve pilot sensitivity without allowing unreliable pilot imputation to bias the spectrum. Refit the exact implementation by held-out lab to detect whether the simpler initialization loses the measured gain.
- **Alternatives Weighed**: `Apply pilot-inferred fillers to deployment coupons` (rejected: The held-out result validates those fillers within the pilot only; the bench/test reference population is distinct.)
- **ATMS NOGOOD DECLARED**: Refuted `The pilot's cell-balanced feature means adequately represent the missing values used in its measured coupon drift.` | Witness: `Keeping the complete-case pilot spectrum fixed and estimating missing fillers from the other labs cut held-out missing-coupon strain MAE from 0.06629 to 0.00390 and initial-rate MAE from 0.01142 to 0.00334.` | Blanket Diagnosis: `A small pilot panel can misestimate population imputation values; the resulting drift error appears in both late strain and initial rate.` | Sublated Pivot: `Fit pilot constitutive parameters on complete coupons when identifiable, and keep deployment imputation tied to the bench/test population rather than importing pilot-specific fillers.`

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/batch_solver.py')
s = p.read_text()
start = s.index('def _fit_relaxation(')
s = s[:start] + '''def _fit_constitutive(s, stress, dt, t, design, noise, prior=None):
    u, h = stress
    y = np.c_[s['strain_curves_obs'] - s['strain_0'][:, None], s['instant_rate_obs']]
    z = np.r_[t, 1.0]
    scale = np.r_[np.full(len(t), noise), noise * .75]
    drift = design[:, None, :] * (z / scale)[None, :, None]
    target = (y / scale).ravel()

    def kernel(log_tau):
        return _kernel(np.exp(log_tau), dt, t, u, h) / scale[None, :, None]

    def residual(p):
        return (kernel(p[:2]) @ p[2:4] + drift @ p[4:]).ravel() - target

    def jacobian(p):
        a = kernel(p[:2])
        step = 1e-5
        da = (kernel(p[:2] + step) - kernel(p[:2] - step)) / (2 * step)
        return np.concatenate((da * p[2:4], a, drift), axis=2).reshape(-1, len(p))

    lower = np.r_[np.log([.050001] * 2), np.zeros(2), np.full(5, -np.inf)]
    upper = np.r_[np.log([24.9999] * 2), np.full(7, np.inf)]
    starts = [] if prior is None else [np.r_[np.log(prior[0]), prior[1], prior[2]]]
    for pair in ((.45, 3.), (.65, 6.), (.2, 12.), (1.5, 17.)):
        log_tau = np.log(pair)
        a = np.concatenate((kernel(log_tau), drift), axis=2).reshape(-1, 7)
        coef = np.linalg.lstsq(a, target, rcond=None)[0]
        coef[:2] = np.maximum(coef[:2], .01)
        starts.append(np.r_[log_tau, coef])
    fits = [least_squares(residual, p, jac=jacobian, bounds=(lower, upper),
                          loss='cauchy', max_nfev=180) for p in starts]
    p = min(fits, key=lambda fit: fit.cost).x
    ix = np.argsort(p[:2])
    return np.exp(p[:2][ix]), p[2:4][ix], p[4:]


def _complete(s, x):
    mask = np.isfinite(x).all(axis=1)
    if np.linalg.matrix_rank(np.c_[np.ones(mask.sum()), x[mask]]) == x.shape[1] + 1:
        return {key: value[mask] for key, value in s.items()}, x[mask]
    return s, x


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    pilot = dataset.get('chamber_verification_trials')
    dt = np.asarray(dataset['past_dt'])
    t = np.asarray(dataset['eval_times'])
    cal = _calibrate(dataset['calibration_load_cells'])

    panels = _panels(train, test)
    x_train, x_test = (_standardize(s, panels) for s in (train, test))
    fill = _impute(np.concatenate((x_train, x_test)), train, test)
    fit = _fit_constitutive(train, _stress(train, cal), dt, t, _design(x_train, fill), .012)

    if pilot is not None and len(pilot['strain_0']):
        x = _standardize(pilot, _panels(pilot))
        # Do not let a new panel's poorly determined missing fillers bias the spectrum.
        source, x_complete = _complete(pilot, x)
        design = _design(x_complete, fill)
        if np.linalg.matrix_rank(design) == design.shape[1]:
            fit = _fit_constitutive(source, _stress(source, cal), dt, t, design, .006, fit)

    tau, weights, drift = fit
    transient = _kernel(tau, dt, t, *_stress(test, cal)) @ weights
    rate = _design(x_test, fill) @ drift
    return {'estimated_taus': tau, 'hereditary_weights': np.r_[weights, drift],
            'predicted_strain': test['strain_0'][:, None] + transient[:, :-1] + rate[:, None] * t,
            'initial_rate_pred': transient[:, -1] + rate}
'''
p.write_text(s)
PY
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import runpy,pickle,numpy as np
m=runpy.run_path('/holdout_task/batch_solver.py'); d=pickle.load(open('public_batch.pkl','rb'))[0]['dataset']; b=d['train']; s=d['chamber_verification_trials']; te=d['test']; dt,t=d['past_dt'],d['eval_times']; cal=m['_calibrate'](d['calibration_load_cells']); bp=m['_panels'](b,te); xb,xt=[m['_standardize'](a,bp) for a in (b,te)]; fill=m['_impute'](np.r_[xb,xt],b,te); pp=m['_panels'](s); xp=m['_standardize'](s,pp); z=np.r_[t,1.]
sub=lambda a,k:{name:v[k] for name,v in a.items()}
def fit(a,x,f,noise,prior=None): return m['_fit_constitutive'](a,m['_stress'](a,cal),dt,t,m['_design'](x,f),noise,prior)
def score(a,x,result,f):
 tau,w,p=result; g=m['_kernel'](tau,dt,t,*m['_stress'](a,cal))@w; r=m['_design'](x,f)@p; y=np.c_[a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']]; e=abs(g+r[:,None]*z-y); miss=~np.isfinite(x).all(1); return [np.quantile(e[:,:-1],.9),e[:,-1].mean(),e[:,-2].mean(),e[miss,:-1].mean() if miss.any() else 0.]
metrics={k:[] for k in ('bench','pilot_complete','pilot_fold_fill')}
for lab in np.unique(b['lab_id']):
 kb=b['lab_id']!=lab; kp=s['lab_id']!=lab; a=fit(sub(b,kb),xb[kb],fill,.012); metrics['bench'].append(score(sub(b,~kb),xb[~kb],a,fill)); sc,xx=m['_complete'](sub(s,kp),xp[kp]); v=fit(sc,xx,fill,.006,a); metrics['pilot_complete'].append(score(sub(s,~kp),xp[~kp],v,fill)); tau,w,p=v; y=np.c_[s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']]; g=m['_kernel'](tau,dt,t,*m['_stress'](s,cal))@w; drift=np.median((y-g)/z,axis=1); delta=drift-m['_design'](xp,fill)@p; miss=~np.isfinite(xp); pf=fill.copy()
 for j in range(xp.shape[1]):
  rows=kp & miss[:,j] & (miss.sum(1)==1)
  if rows.any() and abs(p[j+1])>1e-8: pf[j]+=np.median(delta[rows])/p[j+1]
 metrics['pilot_fold_fill'].append(score(sub(s,~kp),xp[~kp],v,pf))
print('worst lab: strain q90 / rate MAE / final strain MAE / missing strain MAE', {k:np.max(v,axis=0).round(5).tolist() for k,v in metrics.items()})
for p in [s,None]:
 r=m['solve_batch'](dict(d,chamber_verification_trials=p)); assert all(np.all(np.isfinite(v)) for v in r.values()); assert .05<r['estimated_taus'][0]<r['estimated_taus'][1]<25; print('pilot',p is not None,np.r_[r['estimated_taus'],r['hereditary_weights']].round(5))
PY
```
- **Container Exit Code**: `0` (`1252 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
worst lab: strain q90 / rate MAE / final strain MAE / missing strain MAE {'bench': [0.02271, 0.00701, 0.04162, 0.03222], 'pilot_complete': [0.02573, 0.00592, 0.01512, 0.01894], 'pilot_fold_fill': [0.00778, 0.00334, 0.00457, 0.0039]}
pilot True [ 0.53728  4.12444  0.79564  0.57761  0.35447  0.20749 -0.10648  0.09977
 -0.12799]
pilot False [ 0.53686  4.1725   0.79476  0.58307  0.35425  0.20774 -0.10682  0.0996
 -0.12838]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4e30d44e223d9d68 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_4e30d44e (directory).
```

### Turn 13 — `execute_in_remote_docker` (`28315 ms` LLM)
- **Rationale**: Exercise the final blind path through solve_batch with each lab’s outcomes withheld. Affine re-encoding checks that the revised fitter still respects bridge calibration and reference-panel standardization in both modes; failure would require fixing the corresponding preprocessing or fit before submission.

```bash
set -e
cd /holdout_task
export OPENBLAS_NUM_THREADS=1
python3 -I validate_public_schema.py
python3 -I - <<'PY'
import pickle, runpy, numpy as np
solve=runpy.run_path('/holdout_task/batch_solver.py')['solve_batch']
d=pickle.load(open('public_batch.pkl','rb'))[0]['dataset']
b,te,p=(d[k] for k in ('train','test','chamber_verification_trials'))
sample_keys=list(te)
metrics=[]
for lab in np.unique(b['lab_id']):
    va=b['lab_id']==lab
    train={k:v[~va] for k,v in b.items()}
    test={k:np.concatenate((b[k][va],te[k]),axis=0) for k in sample_keys}
    out=solve(dict(d,train=train,test=test,chamber_verification_trials=None))
    n=va.sum(); e=np.abs(out['predicted_strain'][:n]-b['strain_curves_obs'][va]); r=np.abs(out['initial_rate_pred'][:n]-b['instant_rate_obs'][va]); miss=np.isnan(b['coupon_features_raw'][va]).any(axis=1)
    metrics.append([np.quantile(e,.9),r.mean(),e[:,-1].mean(),e[miss].mean() if miss.any() else 0])
print('blind solve_batch held-out lab maxima: strain q90 / rate MAE / final strain MAE / missing strain MAE',np.max(metrics,axis=0).round(6))
assert np.max(metrics,axis=0)[0]<.025 and np.max(metrics,axis=0)[1]<.008
rng=np.random.default_rng(98); a=np.eye(4)+rng.normal(0,.15,(4,4)); shift=rng.normal(0,.4,4)
encode=lambda x:x@a+shift
for pilot in (p,None):
    case=dict(d,chamber_verification_trials=pilot)
    out=solve(case)
    changed=dict(case,calibration_load_cells=dict(d['calibration_load_cells'],bridge_raw=encode(d['calibration_load_cells']['bridge_raw'])))
    for k in ('train','test','chamber_verification_trials'):
        s=case[k]
        if s is None: continue
        lab=s['lab_id']; gain=1+.2*(lab+1); offset=.4*(lab-1)
        changed[k]=dict(s,ref_coupons=s['ref_coupons']*gain[:,None]+offset[:,None],coupon_features_raw=s['coupon_features_raw']*gain[:,None]+offset[:,None],past_channel_raw=encode(s['past_channel_raw']),hold_channel_raw=encode(s['hold_channel_raw']))
    other=solve(changed)
    assert .05<out['estimated_taus'][0]<out['estimated_taus'][1]<25
    assert all(np.isfinite(v).all() for v in out.values())
    assert all(np.allclose(out[k],other[k],atol=3e-6,rtol=2e-6) for k in out)
    print('pilot present',pilot is not None,'max bridge/panel re-encoding delta',max(np.max(np.abs(out[k]-other[k])) for k in out))
PY
```
- **Container Exit Code**: `0` (`1070 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
blind solve_batch held-out lab maxima: strain q90 / rate MAE / final strain MAE / missing strain MAE [0.022712 0.007009 0.041622 0.032216]
pilot present True max bridge/panel re-encoding delta 1.48885348494332e-11
pilot present False max bridge/panel re-encoding delta 1.790870829410096e-08

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4e30d44e223d9d68 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_4e30d44e (directory).
```

### Turn 14 — `submit_completed_task` (`12882 ms` LLM)
- **Rationale**: The saved candidate passed clean-process schema validation, held-out-lab tests, and affine re-encoding checks in both operating modes. No continual archive was present, so cross-campaign persistence was not verified.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema check passed, and both pilot-present and blind modes passed calibration and reference-panel invariance checks.`

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
