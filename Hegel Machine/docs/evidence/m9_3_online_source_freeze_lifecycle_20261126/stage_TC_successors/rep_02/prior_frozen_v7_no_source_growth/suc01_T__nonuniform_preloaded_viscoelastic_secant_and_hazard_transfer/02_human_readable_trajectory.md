# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `suc01_T__nonuniform_preloaded_viscoelastic_secant_and_hazard_transfer`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M93-Lifecycle` |
| **Domain** | `Pre-Loaded Non-Uniform Hereditary Viscoelasticity, Secant Compliance & First-Passage Hazard Crossing` |
| **Wave / Wormhole Family** | `m93_stage_T_transfer` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`PASS (1.0)`** |
| **Failure / Outcome Taxonomy** | `PASSED_ALL_CHECKS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `368,440` (`353,689` / `98,816` / `14,751`) |
| **Total Episode Wall Time (s)** | `499.76s` |

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

### Turn 1 — `execute_in_remote_docker` (`15566 ms` LLM)
- **Rationale**: Inspect split keys, response availability, and reference missingness to identify the constitutive fit and validation design; look up existing primitives before implementing them.

```bash
cd /holdout_task && cat validate_public_schema.py && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference panel alignment imputation robust IRLS' ; ls -la /pretrain_task /holdout_task; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_suc01_T_datasets.pkl','rb'))
def show(x,d=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print(' '*d,k,end=': ')
   if isinstance(v,dict): print();show(v,d+2)
   else:
    a=np.asarray(v); print(type(v).__name__,a.shape,a.dtype,repr(a.ravel()[:8]), ('nan='+str(np.isnan(a).sum())) if a.dtype.kind=='f' else '')
 elif isinstance(x,(tuple,list)):
  print(type(x),len(x));show(x[0],d+2)
 else: print(type(x))
show(p)
PY
```
- **Container Exit Code**: `0` (`695 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ndarray (90,) int64 array([2, 1, 3, 2, 2, 2, 3, 0]) 
       stratum: ndarray (90,) int64 array([2, 1, 1, 2, 1, 2, 0, 2]) 
       coupon_features_raw: ndarray (90, 4) float64 array([ 5.80670276,  3.21953915,  5.87440098,  2.30760519, -3.31586622,
       -3.07486611, -3.02838551, -3.80182155]) nan=9
       ref_coupons: ndarray (90, 20) float64 array([4.3984623 , 4.81373099, 3.52230739, 3.02044767, 4.29253891,
       3.01654852, 4.23250921, 6.04814587]) nan=71
       past_stress: ndarray (90, 22) float64 array([0.17514512, 1.11947631, 0.2836669 , 1.14246153, 0.97534767,
       0.67840891, 1.10217839, 1.21806182]) nan=0
       strain_0: ndarray (90,) float64 array([ 0.35702858,  0.51890486,  0.22478204, -0.48160431, -0.23762893,
       -0.565055  , -0.25472099,  0.30800075]) nan=0
       stress_hold_0: ndarray (90,) float64 array([ 0.55739059,  1.85838836,  1.2401115 , -0.90114279, -2.14202929,
       -0.98528186, -1.31289842,  1.72475   ]) nan=0
       instant_rate_obs: ndarray (90,) float64 array([-0.76026931,  2.89067114,  1.93731702,  1.6678929 , -4.29674337,
       -0.19329587, -2.62131066,  4.57946935]) nan=0
       initial_memory_states: ndarray (90, 2) float64 array([ 0.22517017,  0.50178775, -1.18377314,  0.63178951,  1.24105695,
       -0.86807283,  0.329819  ,  0.59031428]) nan=0
       critical_strain: ndarray (90,) float64 array([ 0.63559661,  1.06767827,  0.73497544, -0.06533551,  0.01804436,
        0.00780452, -0.04492213,  0.86482742]) nan=0
   gt: 
     taus: ndarray (2,) float64 array([0.46, 3.35]) nan=0
     theta: ndarray (7,) float64 array([ 0.74,  0.52,  0.08,  0.15, -0.12,  0.1 , -0.08]) nan=0
     vp_theta: ndarray (2,) float64 array([0., 0.]) nan=0
     te_Z: ndarray (90, 4) float64 array([ 1.56503742, -0.26983039,  1.61305034, -0.91659206,  0.10649845,
        0.41156189,  0.47039808, -0.50863487]) nan=0
     te_clean: ndarray (90, 12) float64 array([0.27117114, 0.22721742, 0.2570771 , 0.37213523, 0.57921606,
       0.86860451, 1.22837667, 1.67984171]) nan=0
     te_true_rate0: ndarray (90,) float64 array([-0.76516176,  2.87812647,  1.953571  ,  1.68407312, -4.29498085,
       -0.19313767, -2.62942418,  4.59392597]) nan=0
     te_cross: ndarray (90,) float64 array([1.5571546 , 0.24597169, 0.37389148, 0.30653728, 7.9       ,
       1.49316729, 7.9       , 0.13964904]) nan=0
     te_secant: ndarray (90,) float64 array([ 6.95668168,  1.52928602,  0.65570041,  5.7305678 , -1.13980353,
        3.76738188, -4.54465623,  2.628842  ]) nan=0
```

### Turn 2 — `execute_in_remote_docker` (`25531 ms` LLM)
- **Rationale**: Check whether the reference columns vary mostly by lab or specimen; that distinction determines whether alignment should use lab-level calibration or specimen-level predictors. Inspect available case and grid metadata before choosing the constitutive basis.

```bash
cd /holdout_task; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_affine_projection.robust_residual_weights; find /pretrain_task -maxdepth 2 -type f | head -20; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_suc01_T_datasets.pkl','rb'))
print('cases',len(p))
for c in p:
 d=c['dataset']; print('case keys',list(c),'dataset keys',list(d)); print('n',*[len(d[s]['lab_id']) for s in ('train','test')]); print('truth metadata',[(k,c['gt'][k]) for k in ('taus','theta','vp_theta') if k in c['gt']]); print('grid',[(k,v) for k,v in d.items() if k not in ('train','test')]);
d=p[0]['dataset']; q=d['train']; print('train keys',list(q));
for l in np.unique(q['lab_id']):
 for s in np.unique(q['stratum']):
  ix=(q['lab_id']==l)&(q['stratum']==s); r=q['ref_coupons'][ix]; f=q['coupon_features_raw'][ix]; print(l,s,len(r),np.round(np.nanmean(r,axis=0)[:8],2), np.round(np.nanstd(r,axis=0)[:8],2),np.round(np.nanmean(f,axis=0),2))
PY
```
- **Container Exit Code**: `0` (`616 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_robust_affine_projection",
      "reference_source_txt": null
    }
  ]
}
cases 1
case keys ['dataset', 'gt'] dataset keys ['past_dt', 'eval_times', 'train', 'test']
n 148 90
truth metadata [('taus', array([0.46, 3.35])), ('theta', array([ 0.74,  0.52,  0.08,  0.15, -0.12,  0.1 , -0.08])), ('vp_theta', array([0., 0.]))]
grid [('past_dt', array([0.0273, 0.0295, 0.0299, 0.0331, 0.0361, 0.0386, 0.0395, 0.0414,
       0.0425, 0.0456, 0.0497, 0.0609, 0.0551, 0.0604, 0.0636, 0.0808,
       0.0786, 0.0836, 0.0893, 0.0986, 0.1056, 0.1066])), ('eval_times', array([0.15, 0.35, 0.65, 1.  , 1.45, 2.  , 2.65, 3.45, 4.35, 5.45, 6.65,
       7.9 ]))]
train keys ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress', 'strain_0', 'stress_hold_0', 'instant_rate_obs', 'strain_curves_obs', 'initial_memory_states']
0 0 20 [3.   3.27 3.29 3.22 3.39 3.05 3.16 3.46] [1.45 1.74 1.56 0.75 1.44 1.21 1.48 1.2 ] [1.9  3.97 2.2  3.5 ]
0 1 8 [3.45 3.09 2.8  3.33 2.96 2.64 3.29 2.9 ] [1.21 1.51 1.41 1.04 1.26 1.7  0.89 1.28] [3.64 3.44 3.85 3.32]
0 2 8 [3.08 3.28 3.35 3.41 3.48 2.88 4.01 3.31] [0.57 0.76 1.47 0.97 1.16 1.04 1.17 0.93] [4.86 2.18 4.14 2.89]
1 0 8 [-3.66 -3.26 -3.7  -3.5  -3.15 -3.35 -3.01 -3.53] [0.94 0.74 0.34 0.77 0.69 0.56 0.81 0.91] [-4.   -2.67 -4.37 -3.1 ]
1 1 22 [-3.48 -3.38 -3.55 -3.44 -3.14 -3.36 -3.75 -3.19] [0.79 0.7  0.67 0.89 0.79 0.78 0.74 0.62] [-3.4  -3.48 -3.46 -3.28]
1 2 8 [-3.62 -3.78 -2.91 -3.79 -3.87 -3.44 -3.5  -3.47] [1.27 0.47 0.34 1.32 0.85 0.62 0.59 0.91] [-2.94 -3.74 -2.84 -4.07]
2 0 8 [3.66 3.9  3.78 4.21 3.02 3.45 3.95 3.98] [1.06 1.21 1.68 0.86 1.76 2.   1.95 1.56] [2.02 5.36 3.1  4.48]
2 1 8 [3.59 3.45 3.72 3.38 3.42 3.02 3.99 3.52] [1.48 1.58 1.34 0.69 1.49 1.18 0.99 0.91] [3.6  3.39 3.89 3.49]
2 2 22 [3.24 3.5  3.5  3.44 3.03 3.34 2.98 3.36] [1.34 1.22 1.33 1.34 1.24 1.52 1.1  1.49] [5.4  2.54 4.39 2.98]
3 0 18 [-2.98 -2.9  -2.88 -3.06 -2.94 -3.28 -2.83 -2.78] [0.6  0.92 1.13 0.85 0.84 1.02 0.83 0.54] [-3.82 -2.42 -3.59 -2.25]
3 1 9 [-2.79 -2.69 -2.46 -3.21 -3.17 -2.94 -2.67 -3.59] [0.77 1.25 0.76 1.07 0.8  0.57 0.58 0.44] [-2.86 -3.18 -2.58 -2.79]
3 2 9 [-2.36 -2.96 -2.6  -2.51 -3.03 -3.14 -2.6  -3.18] [0.72 0.63 1.09 0.5  0.88 0.68 0.84 0.94] [-1.75 -3.8  -2.53 -3.63]
```

### Turn 3 — `execute_in_remote_docker` (`57400 ms` LLM)
- **Rationale**: Fit small candidate constitutive bases to determine whether linear drift and specimen-dependent branch amplitudes explain the robust residuals. If late residuals persist, test the unresolved hold-law basis before building the solver.
- **Alternatives Weighed**: `Independent time-point regressions` (rejected: They do not enforce a shared memory recurrence or the supplied physical initial derivative.)

```bash
cat > /tmp/probe_suc.py <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
p=pickle.load(open('/holdout_task/public_suc01_T_datasets.pkl','rb'))[0];d=p['dataset'];q=d['train']; t=d['eval_times']; dt=d['past_dt']; n=len(q['lab_id']); l=q['lab_id']; s=q['stratum']; h=q['stress_hold_0']; y=q['strain_curves_obs']-q['strain_0'][:,None]; r=q['instant_rate_obs']; f=q['coupon_features_raw']; ref=q['ref_coupons']; u=np.empty_like(f)
for a in np.unique(l):
 ix=l==a; rr=ref[ix]; u[ix]=(f[ix]-np.nanmedian(np.nanmean(rr,axis=0)))/np.nanmedian(np.nanstd(rr,axis=0))
w=np.array([1/sum((l==a)&(s==b)) for a,b in zip(l,s)]); fill=np.nansum(w[:,None]*u,axis=0)/np.sum(w[:,None]*np.isfinite(u),axis=0); u=np.where(np.isfinite(u),u,fill); Z=np.c_[np.ones(n),u];
def gaps(tau):
 z=q['initial_memory_states'].copy()
 for j,step in enumerate(dt):
  a=np.exp(-step/tau); z=a*z+(1-a)*q['past_stress'][:,j,None]
 return h[:,None]-z

def design(tau,mode):
 g=gaps(tau); e=-np.expm1(-t[:,None]/tau); T=np.r_[0,t]; A=np.empty((n,len(T),2)); A[:,0,:]=g/tau; A[:,1:,:]=g[:,None,:]*e
 U=np.empty((n,len(T),Z.shape[1])); U[:,0,:]=Z*h[:,None]; U[:,1:,:]=T[None,1:,None]*U[:,0:1,:]
 if mode==0:return np.concatenate((A,U),axis=2)
 if mode==1:return np.concatenate((A,U,np.tile(U[:,:,[0]],(1,1,4))*Z[:,None,1:]**2),axis=2)
 if mode==2:return np.concatenate(((A[:,:,:,None]*Z[:,None,None,:]).reshape(n,len(T),-1),U),axis=2)
 if mode==3:return np.concatenate(((A[:,:,:,None]*Z[:,None,None,:]).reshape(n,len(T),-1),U,np.ones_like(U[:,:,:1])*T[None,:,None]),axis=2)
 if mode==4:return np.concatenate((A,U,np.ones_like(U)*Z[:,None,:]*T[None,:,None],A*tau*T[None,:,None]),axis=2)
Y=np.c_[r,y];
for mode in range(5):
 def fun(v):return ((design(v[:2],mode)@v[2:]-Y)*np.r_[1,np.ones(len(t))]).ravel()
 A=design(np.array([.55,3.]),mode); fit=least_squares(fun,np.r_[.55,3.,np.linalg.lstsq(A.reshape(-1,A.shape[-1]),Y.ravel(),rcond=None)[0]],loss='cauchy',f_scale=.04,bounds=(np.r_[.2,.22,np.full(A.shape[-1],-np.inf)],np.r_[8.9,9.,np.full(A.shape[-1],np.inf)]),max_nfev=180); E=fun(fit.x).reshape(Y.shape); print(mode,np.round(fit.x,4),np.round(np.quantile(np.abs(E),[.5,.8,.9],axis=0),3),flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_suc.py
```
- **Container Exit Code**: `0` (`1557 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 [ 0.4779  5.5534  0.7845  0.8539 -0.083  -0.0469 -0.0491  0.0315  0.1425] [[0.216 0.038 0.083 0.159 0.254 0.345 0.479 0.617 0.768 0.997 1.139 1.403
  1.739]
 [0.46  0.087 0.191 0.328 0.569 0.715 0.948 1.199 1.548 1.969 2.492 3.032
  3.614]
 [0.592 0.13  0.269 0.479 0.794 0.926 1.331 1.572 2.013 2.521 3.266 3.97
  4.719]]
1 [ 4.6440e-01  4.2897e+00  7.5200e-01  6.6320e-01 -3.6400e-02 -5.5100e-02
  2.6000e-03  3.8900e-02  7.6200e-02 -4.4400e-02 -9.0400e-02  4.1700e-02
  3.7800e-02] [[0.188 0.035 0.074 0.134 0.23  0.301 0.458 0.505 0.684 0.846 1.004 1.286
  1.515]
 [0.471 0.102 0.215 0.367 0.658 0.78  1.072 1.304 1.576 1.987 2.674 3.151
  3.779]
 [0.715 0.16  0.352 0.623 0.992 1.159 1.613 1.975 2.775 3.042 3.884 4.75
  5.707]]
2 [ 0.5073  9.      0.8023  0.0257  0.0117 -0.0343  0.0214  1.3231 -0.2586
  0.2019  0.3872  0.2713 -0.0658 -0.0148 -0.0841  0.0104  0.0924] [[0.212 0.036 0.072 0.141 0.251 0.355 0.448 0.582 0.756 0.966 1.161 1.418
  1.749]
 [0.507 0.095 0.201 0.332 0.593 0.701 0.913 1.15  1.449 1.851 2.301 2.885
  3.537]
 [0.645 0.156 0.281 0.504 0.818 0.998 1.347 1.588 2.082 2.495 3.116 3.801
  4.656]]
3 [ 0.4564  4.5641  0.721  -0.026  -0.04   -0.0164  0.0503  0.8271  0.0825
  0.2617  0.1359 -0.2111 -0.0382  0.0742 -0.0807  0.0454 -0.0279  0.1065] [[0.208 0.033 0.071 0.134 0.189 0.273 0.389 0.441 0.591 0.64  0.863 1.083
  1.329]
 [0.484 0.095 0.229 0.429 0.636 0.89  1.217 1.478 1.983 2.451 3.074 3.746
  4.33 ]
 [0.69  0.158 0.362 0.622 0.955 1.135 1.621 2.108 2.535 3.306 4.152 5.08
  5.979]]
4 [ 4.6240e-01  3.3714e+00  7.4270e-01  5.1970e-01 -8.0000e-04  2.0000e-04
  0.0000e+00 -1.0000e-04  2.0000e-04  7.9900e-02  1.4630e-01 -1.1770e-01
  9.7600e-02 -7.9000e-02  1.1000e-03  0.0000e+00] [[0.279 0.006 0.007 0.006 0.007 0.008 0.01  0.009 0.012 0.013 0.013 0.016
  0.017]
 [0.452 0.013 0.014 0.015 0.016 0.017 0.025 0.02  0.035 0.028 0.043 0.081
  0.077]
 [0.571 0.038 0.05  0.067 0.367 0.082 0.818 0.12  0.542 0.096 0.273 0.887
  0.811]]
```

### Turn 4 — `execute_in_remote_docker` (`50476 ms` LLM)
- **Rationale**: The previous probe used the additive drift only at positive times, omitting its initial derivative. Correct that basis and compare coupon- and rate-based drift estimates on complete and missing-coupon rows; rate noise is amplified at the terminal readout.
- **ATMS NOGOOD DECLARED**: Refuted `A drift basis made only from hold stress times aligned coupon features is sufficient.` | Witness: `Its terminal training median error was 1.739; adding stress-independent coupon drift reduced it to 0.017 and recovered two distinct relaxation times.` | Blanket Diagnosis: `The original basis forced all secular drift through the applied hold stress and omitted additive material drift.` | Sublated Pivot: `Retain both preloaded recurrence branches and fit the additive coupon drift with its matching contribution to the initial rate.`

```bash
sed '/^for mode in range(5):/,$d' /tmp/probe_suc.py > /tmp/suc_rate.py; cat >> /tmp/suc_rate.py <<'PY'
def prep(q, cal=None):
 l=q['lab_id']; s=q['stratum']; f=q['coupon_features_raw']; ref=q['ref_coupons']; u=np.empty_like(f)
 if cal is None:
  cal={}
  for a in np.unique(l):
   rr=ref[l==a]; cal[a]=(np.nanmedian(np.nanmean(rr,axis=0)),np.nanmedian(np.nanstd(rr,axis=0)))
 for a in np.unique(l):
  ix=l==a; mu,sigma=cal[a]; u[ix]=(f[ix]-mu)/sigma
 return u,cal

def fit_predict(tr,te,wt,lam=1):
 u,cal=prep(tr); v,_=prep(te,cal); l=tr['lab_id']; s=tr['stratum']; _,idx,counts=np.unique(np.c_[l,s],axis=0,return_inverse=True,return_counts=True); w=1/counts[idx]; w/=w.mean(); fill=np.nansum(w[:,None]*u,axis=0)/(np.isfinite(u).T@w); X=np.c_[np.ones(len(u)),np.where(np.isfinite(u),u,fill)]; V=np.c_[np.ones(len(v)),np.where(np.isfinite(v),v,fill)]; Y=np.c_[tr['instant_rate_obs'],tr['strain_curves_obs']-tr['strain_0'][:,None]]; good=np.isfinite(u).all(1)
 def basis(q,tau):
  z=q['initial_memory_states'].copy()
  for j,step in enumerate(dt):
   a=np.exp(-step/tau); z=a*z+(1-a)*q['past_stress'][:,j,None]
  g=q['stress_hold_0'][:,None]-z
  return np.concatenate(((g/tau)[:,None,:],g[:,None,:]*-np.expm1(-t[:,None]/tau)),1)
 T=np.r_[1,t]; W=(0.35+0.65*good)[:,None]**lam*np.r_[wt,np.ones(len(t))]
 def fun(b):return ((basis(tr,b[:2])@b[2:4]+(X@b[4:])[:,None]*T-Y)*W).ravel()
 A=basis(tr,np.array([.55,3.])); A=np.concatenate((A,X[:,None,:]*T[None,:,None]),2); b0=np.linalg.lstsq((A*W[:,:,None]).reshape(-1,7),(Y*W).ravel(),rcond=None)[0]
 opt=least_squares(fun,np.r_[.55,3.,b0],loss='cauchy',f_scale=.025,bounds=(np.r_[.2,.22,np.full(7,-np.inf)],np.r_[8.9,9.,np.full(7,np.inf)]),max_nfev=100); b=opt.x; B=basis(te,b[:2])@b[2:4]; D=V@b[4:]; Dr=te['instant_rate_obs']-B[:,0];
 print('fit',wt, np.round(b,5), 'train med',np.round(np.median(np.abs(fun(b).reshape(Y.shape)/W),0),4),'D vs Dr',np.round(np.quantile(np.abs(D-Dr),[.25,.5,.75,.9]),4)); return b,B,D,Dr

def hazards(te,P):
 P=np.c_[te['strain_0'],P]; T=np.r_[0,t]; crit=te['critical_strain']; reach=P>=crit[:,None]; j=reach.argmax(1); rr=np.arange(len(j)); ans=np.full(len(j),t[-1]); ans[reach[:,0]]=0.; ix=reach.any(1)&(j>0); rr=rr[ix]; k=j[ix]; ans[ix]=T[k-1]+(T[k]-T[k-1])*(crit[ix]-P[rr,k-1])/(P[rr,k]-P[rr,k-1]); return ans
te=d['test']; gt=p['gt']; good=np.isfinite(te['coupon_features_raw']).all(1); groups=[np.ones(len(good),bool),good,~good]; den=np.maximum(abs(te['stress_hold_0']),.5)
for wt in [.5,1,2,4]:
 b,B,D,Dr=fit_predict(q,te,wt)
 Dg=np.median((gt['te_clean']-te['strain_0'][:,None]-B[:,1:])/t,axis=1); print('drift errors quantiles',np.round(np.quantile(np.abs(D-Dg),[.5,.8,.95]),4),np.round(np.quantile(np.abs(Dr-Dg),[.5,.8,.95]),4))
 for mix in [0,.25,.5,.75,1]:
  m=np.where(good,mix,1); P=te['strain_0'][:,None]+B[:,1:]+((1-m)*D+m*Dr)[:,None]*t; E=np.c_[np.abs(P-gt['te_clean']).mean(1),np.abs((P[:,-1]-te['strain_0'])/den-gt['te_secant']),np.abs(hazards(te,P)-gt['te_cross'])];print(mix, np.round([E[g].mean(0) for g in groups],4).tolist())
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/suc_rate.py
```
- **Container Exit Code**: `0` (`771 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, 0.0178, 0.0035], [0.0202, 0.0426, 0.1235]]
0.25 [[0.0116, 0.0248, 0.0147], [0.0106, 0.0228, 0.0026], [0.0202, 0.0426, 0.1235]]
0.5 [[0.0174, 0.0365, 0.0158], [0.017, 0.0359, 0.0039], [0.0202, 0.0426, 0.1235]]
0.75 [[0.0244, 0.0507, 0.0177], [0.0249, 0.0516, 0.006], [0.0202, 0.0426, 0.1235]]
1 [[0.0322, 0.0661, 0.02], [0.0335, 0.0687, 0.0085], [0.0202, 0.0426, 0.1235]]
fit 1 [ 0.46103  3.36985  0.74182  0.51999  0.07964  0.14652 -0.1176   0.09755
 -0.07904] train med [0.0074 0.0064 0.007  0.0061 0.0064 0.0078 0.0106 0.0087 0.0112 0.0111
 0.0122 0.0157 0.017 ] D vs Dr [0.0043 0.0103 0.0186 0.0402]
drift errors quantiles [0.0026 0.0066 0.0949] [0.0074 0.0154 0.0293]
0 [[0.0102, 0.0203, 0.0154], [0.0091, 0.0178, 0.0034], [0.0197, 0.0421, 0.1238]]
0.25 [[0.0112, 0.024, 0.0146], [0.0103, 0.022, 0.0025], [0.0197, 0.0421, 0.1238]]
0.5 [[0.0165, 0.0349, 0.0158], [0.0162, 0.0341, 0.0039], [0.0197, 0.0421, 0.1238]]
0.75 [[0.0231, 0.0481, 0.0178], [0.0235, 0.0487, 0.006], [0.0197, 0.0421, 0.1238]]
1 [[0.0305, 0.0627, 0.0201], [0.0317, 0.0649, 0.0085], [0.0197, 0.0421, 0.1238]]
fit 2 [ 0.46096  3.36768  0.74165  0.52002  0.07965  0.14651 -0.11762  0.09754
 -0.07901] train med [0.0073 0.0064 0.0071 0.0062 0.0065 0.0078 0.0104 0.0086 0.0112 0.011
 0.0123 0.0154 0.017 ] D vs Dr [0.0042 0.0103 0.0182 0.0386]
drift errors quantiles [0.0025 0.0065 0.0949] [0.0073 0.0154 0.0286]
0 [[0.0101, 0.0203, 0.0156], [0.009, 0.0179, 0.0034], [0.0198, 0.0423, 0.1262]]
0.25 [[0.0111, 0.0237, 0.0148], [0.0101, 0.0216, 0.0025], [0.0198, 0.0423, 0.1262]]
0.5 [[0.0162, 0.0341, 0.0161], [0.0158, 0.0332, 0.0038], [0.0198, 0.0423, 0.1262]]
0.75 [[0.0226, 0.0469, 0.018], [0.0229, 0.0474, 0.006], [0.0198, 0.0423, 0.1262]]
1 [[0.0297, 0.061, 0.0202], [0.0307, 0.0631, 0.0085], [0.0198, 0.0423, 0.1262]]
fit 4 [ 0.46093  3.36308  0.7414   0.52002  0.07966  0.14649 -0.11763  0.09755
 -0.07896] train med [0.0074 0.0064 0.007  0.0063 0.0065 0.0078 0.0104 0.0087 0.0114 0.0112
 0.0123 0.0155 0.0169] D vs Dr [0.0043 0.0098 0.0183 0.0387]
drift errors quantiles [0.0025 0.0067 0.0949] [0.0072 0.015  0.0287]
0 [[0.0102, 0.0203, 0.016], [0.0091, 0.0178, 0.0033], [0.0203, 0.0428, 0.1303]]
0.25 [[0.011, 0.0236, 0.0153], [0.01, 0.0215, 0.0025], [0.0203, 0.0428, 0.1303]]
0.5 [[0.0162, 0.034, 0.0165], [0.0157, 0.033, 0.0039], [0.0203, 0.0428, 0.1303]]
0.75 [[0.0225, 0.0467, 0.0184], [0.0228, 0.0471, 0.006], [0.0203, 0.0428, 0.1303]]
1 [[0.0295, 0.0607, 0.0205], [0.0305, 0.0626, 0.0083], [0.0203, 0.0428, 0.1303]]
```

### Turn 5 — `execute_in_remote_docker` (`52506 ms` LLM)
- **Rationale**: Implement the observed additive-drift law with its initial derivative and ordered time constants. Run the isolated schema validator to catch import, output shape, and finiteness failures before testing quantitative transfer slices.
- **Alternatives Weighed**: `Use the measured rate to infer drift for every test specimen` (rejected: On complete-coupon rows, rate noise amplified over the hold and raised public secant MAE from 0.0178 to 0.0649.)

```bash
cat > /holdout_task/preloaded_hereditary_secant_hazard_solver.py <<'PY'
"""Fit two preloaded hereditary branches and predict hold-phase readouts."""

import numpy as np
from scipy.optimize import least_squares


def _reference_location_scale(panel):
    available = np.isfinite(panel).sum(axis=0) >= 2
    panel = panel[:, available]
    return np.nanmedian(np.nanmean(panel, axis=0)), np.nanmedian(np.nanstd(panel, axis=0))


def _coupon_design(train, test):
    calibration = {lab: _reference_location_scale(train['ref_coupons'][train['lab_id'] == lab])
                   for lab in np.unique(train['lab_id'])}

    def align(split):
        x = np.empty_like(split['coupon_features_raw'], dtype=float)
        for lab in np.unique(split['lab_id']):
            rows = split['lab_id'] == lab
            if lab in calibration:
                location, scale = calibration[lab]
            else:
                location, scale = _reference_location_scale(split['ref_coupons'][rows])
            x[rows] = (split['coupon_features_raw'][rows] - location) / scale
        return x

    x, v = align(train), align(test)
    _, cell, count = np.unique(np.column_stack((train['lab_id'], train['stratum'])),
                               axis=0, return_inverse=True, return_counts=True)
    weights = 1.0 / count[cell]
    finite = np.isfinite(x)
    fill = (weights @ np.where(finite, x, 0.0)) / (weights @ finite)

    def design(raw):
        valid = np.isfinite(raw)
        return np.column_stack((np.ones(len(raw)), np.where(valid, raw, fill))), valid.all(axis=1)

    return design(x), design(v)


def _branches(split, dt, times, taus):
    z = np.asarray(split['initial_memory_states'], dtype=float).copy()
    increments = -np.expm1(-dt[:, None] / taus)
    for load, fraction in zip(split['past_stress'].T, increments):
        z += fraction * (load[:, None] - z)
    gap = split['stress_hold_0'][:, None] - z
    hold = -np.expm1(-times[:, None] / taus)
    return np.concatenate(((gap / taus)[:, None, :], gap[:, None, :] * hold), axis=1)


def _first_crossings(initial, strain, critical, times):
    grid = np.r_[0.0, times]
    path = np.column_stack((initial, strain))
    reached = path >= critical[:, None]
    crossing = np.full(len(initial), times[-1], dtype=float)
    crossing[reached[:, 0]] = 0.0
    rows = np.flatnonzero(reached.any(axis=1) & ~reached[:, 0])
    right = reached[rows].argmax(axis=1)
    left = right - 1
    a, b = path[rows, left], path[rows, right]
    crossing[rows] = grid[left] + (grid[right] - grid[left]) * ((critical[rows] - a) / (b - a))
    return crossing


def solve_preloaded_hereditary_transfer(dataset: dict) -> dict:
    """Estimate shared relaxation times from training strains and initial rates."""
    train, test = dataset['train'], dataset['test']
    times = np.asarray(dataset['eval_times'], dtype=float)
    dt = np.asarray(dataset['past_dt'], dtype=float)
    (x, complete), (v, test_complete) = _coupon_design(train, test)
    # Column zero models the physical derivative; all others model strain increments.
    time_basis = np.r_[1.0, times]
    drift_basis = x[:, None, :] * time_basis[None, :, None]
    target = np.column_stack((train['instant_rate_obs'],
                              train['strain_curves_obs'] - train['strain_0'][:, None]))
    weights = np.where(complete, 1.0, 0.35)[:, None]
    separation = 1e-4

    def unpack(p):
        fast, fraction = p[:2]
        return np.array([fast, fast + separation + fraction * (9.0 - fast - separation)])

    def residual(p):
        branch = _branches(train, dt, times, unpack(p)) @ p[2:4]
        drift = (x @ p[4:])[:, None] * time_basis
        return ((branch + drift - target) * weights).ravel()

    start = np.array([0.55, (3.0 - 0.55 - separation) / (9.0 - 0.55 - separation)])
    basis = np.concatenate((_branches(train, dt, times, unpack(start)), drift_basis), axis=2)
    coefficients = np.linalg.lstsq((basis * weights[:, :, None]).reshape(-1, basis.shape[-1]),
                                    (target * weights).ravel(), rcond=None)[0]
    p = least_squares(residual, np.r_[start, coefficients],
                      bounds=(np.r_[0.20, 0.0, np.full(len(coefficients), -np.inf)],
                              np.r_[9.0 - separation, 1.0, np.full(len(coefficients), np.inf)]),
                      loss='cauchy', f_scale=0.025, max_nfev=100).x
    taus = unpack(p)
    branch = _branches(test, dt, times, taus) @ p[2:4]
    # When a coupon is missing, the measured derivative identifies the net drift.
    drift = np.where(test_complete, v @ p[4:], test['instant_rate_obs'] - branch[:, 0])
    initial = test['strain_0']
    strain = initial[:, None] + branch[:, 1:] + drift[:, None] * times
    return {
        'estimated_taus': taus,
        'predicted_strain': strain,
        'secant_compliance': (strain[:, -1] - initial) / np.maximum(np.abs(test['stress_hold_0']), 0.5),
        'hazard_crossing_times': _first_crossings(initial, strain, test['critical_strain'], times),
    }
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`509 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/preloaded_hereditary_secant_hazard_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f91c69222a05e84b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f91c6922 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`31514 ms` LLM)
- **Rationale**: The saved solver passed the schema check. An independent crossing calculation and lab, stratum, and missing-coupon slices will detect readout mistakes or errors hidden by pooled means. Leave-one-lab-out results will show whether reference alignment transfers without fitted calibration from that lab.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'preloaded_hereditary_secant_hazard_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def slice_split(split, mask):
    return {k: v[mask] for k, v in split.items()}

for case in pickle.load(open(root+'public_suc01_T_datasets.pkl','rb')):
    d, gt = case['dataset'], case['gt']
    tr, te, t = d['train'], d['test'], d['eval_times']
    print('ground truth keys:', list(gt))
    out = m.solve_preloaded_hereditary_transfer(d)
    p, c, h = (out[k] for k in ('predicted_strain','secant_compliance','hazard_crossing_times'))
    np.testing.assert_allclose(c, (p[:, -1]-te['strain_0'])/np.maximum(abs(te['stress_hold_0']),.5))
    ref = []
    for initial, path, crit in zip(te['strain_0'], p, te['critical_strain']):
        if initial >= crit:
            ref.append(0.); continue
        ta, ya = 0., initial
        for tb, yb in zip(t, path):
            if yb >= crit:
                ref.append(ta + (tb-ta)*(crit-ya)/(yb-ya)); break
            ta, ya = tb, yb
        else:
            ref.append(t[-1])
    np.testing.assert_allclose(h, ref)
    print('tau relative errors:', abs(out['estimated_taus']/gt['taus']-1))
    e = np.c_[abs(p-gt['te_clean']).mean(1), abs(c-gt['te_secant']), abs(h-gt['te_cross'])]
    missed = ~np.isfinite(te['coupon_features_raw']).all(axis=1)
    groups = {'overall': np.ones(len(p),bool),'complete': ~missed,'missing': missed,'initial':gt['te_cross']==0,'interior':(gt['te_cross']>0)&(gt['te_cross']<t[-1]),'censored':gt['te_cross']==t[-1]}
    for name in ('lab_id','stratum'):
        for value in np.unique(te[name]): groups[f'{name}={value}'] = te[name]==value
    for key, mask in groups.items():
        if mask.any(): print(key, int(mask.sum()), e[mask].mean(0).round(5))
    for lab in np.unique(tr['lab_id']):
        keep = tr['lab_id'] != lab
        fold = m.solve_preloaded_hereditary_transfer(dict(d, train=slice_split(tr,keep), test=slice_split(tr,~keep))) if 'critical_strain' in tr else m.solve_preloaded_hereditary_transfer(dict(d, train=slice_split(tr,keep), test=slice_split(te,te['lab_id']==lab)))
        held = te['lab_id']==lab
        print('LOLO', lab, 'tau err', abs(fold['estimated_taus']/gt['taus']-1).max().round(5), 'strain/secant/cross', np.round([abs(fold['predicted_strain']-gt['te_clean'][held]).mean(),abs(fold['secant_compliance']-gt['te_secant'][held]).mean(),abs(fold['hazard_crossing_times']-gt['te_cross'][held]).mean()],5))
PY
```
- **Container Exit Code**: `0` (`631 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ground truth keys: ['taus', 'theta', 'vp_theta', 'te_Z', 'te_clean', 'te_true_rate0', 'te_cross', 'te_secant']
tau relative errors: [0.00223843 0.00592479]
overall 90 [0.01016 0.02027 0.01544]
complete 81 [0.00909 0.01784 0.0034 ]
missing 9 [0.01974 0.04213 0.12378]
interior 63 [0.00983 0.01872 0.02206]
censored 27 [0.01092 0.0239  0.     ]
lab_id=0 22 [0.01674 0.03687 0.04951]
lab_id=1 23 [0.00688 0.01019 0.00114]
lab_id=2 23 [0.00801 0.01751 0.00863]
lab_id=3 22 [0.00925 0.01711 0.00345]
stratum=0 33 [0.01516 0.03281 0.03205]
stratum=1 29 [0.00627 0.01088 0.00206]
stratum=2 28 [0.00829 0.01522 0.00972]
LOLO 0 tau err 0.00195 strain/secant/cross [0.01003 0.02193 0.05754]
LOLO 1 tau err 0.00569 strain/secant/cross [0.03598 0.06146 0.01436]
LOLO 2 tau err 0.00584 strain/secant/cross [0.02945 0.07064 0.03056]
LOLO 3 tau err 0.00537 strain/secant/cross [0.02066 0.03958 0.00585]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f91c69222a05e84b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f91c6922 (directory).
```

### Turn 7 — `submit_completed_task` (`15317 ms` LLM)
- **Rationale**: The archived candidate passed isolated schema validation, independent checks of both derived readouts, and public slice and leave-one-lab-out tests. No tested metric exceeded its tolerance.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/preloaded_hereditary_secant_hazard_solver.py': 'FILE:4959'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/preloaded_hereditary_secant_hazard_solver.py`. The isolated schema check passed. On the public dataset, the maximum relative relaxation-time error was 0.59%; strain, secant compliance, and hazard crossing MAEs were 0.0102, 0.0203, and 0.0154, respectively. All evaluated slices met the requested tolerances.`

### Turn 8 — `execute_in_remote_docker` (`42878 ms` LLM)
- **Rationale**: Fit nonlinear coupon challengers without the held lab to test whether omitted interactions improve transfer rather than absorb lab offsets. Inspect lab-by-stratum errors and missing-coupon crossing misses to determine whether the rate fallback needs revision.
- **Alternatives Weighed**: `Apply relay-mode, Fokker–Planck/HJB, or balanced-accuracy tests` (rejected: The contract specifies a continuous hereditary recurrence and deterministic crossing readout, not a relay, stochastic control model, or classification endpoint.)

```bash
cat > /tmp/audit_secant.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'preloaded_hereditary_secant_hazard_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def split(q, rows):
    return {k:v[rows] for k,v in q.items()}

def features(x):
    a = x[:,1:]
    pairs = [(i,j) for i in range(4) for j in range(i+1,4)]
    differences = np.column_stack([a[:,i]-a[:,j] for i,j in pairs])
    sums = np.column_stack([a[:,i]+a[:,j] for i,j in pairs])
    products = np.column_stack([a[:,i]*a[:,j] for i,j in pairs])
    triples = np.column_stack([np.prod(a[:,np.arange(4)!=i],axis=1) for i in range(4)])
    bounded = np.column_stack([a[:,i]/(1+a[:,j]**2) for i in range(4) for j in range(4)])
    return {'linear':x,'polynomial':np.c_[x,a*a,products,triples], 'rational':np.c_[x,bounded,differences/(1+sums**2)]}

for case in pickle.load(open(root+'public_suc01_T_datasets.pkl','rb')):
    d, gt = case['dataset'], case['gt']
    tr, te, t = d['train'], d['test'], d['eval_times']
    print('observed fields:', sorted(d), sorted(tr))
    base = m.solve_preloaded_hereditary_transfer(d)
    print('baseline tau max rel error:', np.max(np.abs(base['estimated_taus']/gt['taus']-1)))
    for lab in np.unique(tr['lab_id']):
        fit = split(tr,tr['lab_id']!=lab)
        take = te['lab_id']==lab
        val = split(te,take)
        fd = dict(d,train=fit,test=val)
        out = m.solve_preloaded_hereditary_transfer(fd)
        (x,g),(v,h) = m._coupon_design(fit,val)
        train_b = m._branches(fit,d['past_dt'],t,out['estimated_taus'])
        val_b = m._branches(val,d['past_dt'],t,out['estimated_taus'])
        g = g & np.isfinite(fit['instant_rate_obs'])
        y = np.c_[fit['instant_rate_obs'],fit['strain_curves_obs']-fit['strain_0'][:,None]][g]
        taug = np.r_[1.,t]
        errors = {}
        for name,a in features(x).items():
            b = features(v)[name]
            A = np.concatenate((train_b[g], a[g,None,:]*taug[None,:,None]),axis=2)
            A = A.reshape(-1,A.shape[-1]); yy = y.ravel()
            p = least_squares(lambda p:A@p-yy,np.linalg.lstsq(A,yy,rcond=None)[0],jac=lambda p:A,loss='cauchy',f_scale=.025,max_nfev=100).x
            curve = val['strain_0'][:,None]+val_b[:,1:]@p[:2]+(b@p[2:])[:,None]*t
            errors[name] = round(np.abs(curve[h]-gt['te_clean'][take][h]).mean(),5)
        print('held lab',lab,'complete-coupon curve MAE',errors)
    for lab in np.unique(te['lab_id']):
        for s in np.unique(te['stratum']):
            rows = (te['lab_id']==lab)&(te['stratum']==s)
            if rows.any():
                print('cell',lab,s,int(rows.sum()),np.round([np.abs(base['predicted_strain'][rows]-gt['te_clean'][rows]).mean(),np.abs(base['secant_compliance'][rows]-gt['te_secant'][rows]).mean(),np.abs(base['hazard_crossing_times'][rows]-gt['te_cross'][rows]).mean()],5))
    miss = ~np.isfinite(te['coupon_features_raw']).all(axis=1)
    print('missing coupons [observed rate error, true crossing, predicted crossing]:')
    print(np.round(np.c_[(te['instant_rate_obs']-gt['te_true_rate0'])[miss],gt['te_cross'][miss],base['hazard_crossing_times'][miss]],5))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_secant.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`918 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
observed fields: ['eval_times', 'past_dt', 'test', 'train'] ['coupon_features_raw', 'initial_memory_states', 'instant_rate_obs', 'lab_id', 'past_stress', 'ref_coupons', 'strain_0', 'strain_curves_obs', 'stratum', 'stress_hold_0']
baseline tau max rel error: 0.00592479343371477
held lab 0 complete-coupon curve MAE {'linear': 0.01754, 'polynomial': 0.03749, 'rational': 0.05004}
held lab 1 complete-coupon curve MAE {'linear': 0.04389, 'polynomial': 0.04661, 'rational': 0.03959}
held lab 2 complete-coupon curve MAE {'linear': 0.0313, 'polynomial': 0.03184, 'rational': 0.03828}
held lab 3 complete-coupon curve MAE {'linear': 0.03543, 'polynomial': 0.04138, 'rational': 0.04143}
cell 0 0 12 [0.02302 0.05445 0.08524]
cell 0 1 5 [0.01115 0.01902 0.00971]
cell 0 2 5 [0.00727 0.01253 0.00356]
cell 1 0 5 [0.00832 0.01085 0.00011]
cell 1 1 13 [0.00442 0.00826 0.00025]
cell 1 2 5 [0.01186 0.01454 0.00447]
cell 2 0 5 [0.0116  0.02348 0.00433]
cell 2 1 5 [3.96e-03 7.04e-03 2.00e-05]
cell 2 2 13 [0.00818 0.01924 0.01359]
cell 3 0 11 [0.01132 0.02344 0.00114]
cell 3 1 6 [0.00815 0.01298 0.00131]
cell 3 2 5 [0.00603 0.00815 0.01109]
missing coupons [observed rate error, true crossing, predicted crossing]:
[[-8.31000e-03  7.90000e+00  7.90000e+00]
 [ 6.44000e-03  4.96129e+00  4.85818e+00]
 [ 7.48000e-03  7.90000e+00  7.90000e+00]
 [-8.21000e-03  4.29158e+00  5.30138e+00]
 [ 2.08000e-03  7.82900e-02  7.81900e-02]
 [ 1.20800e-02  8.89800e-02  8.87400e-02]
 [ 3.50000e-03  2.93950e-01  2.93150e-01]
 [-1.85000e-03  7.90000e+00  7.90000e+00]
 [-8.86000e-03  7.90000e+00  7.90000e+00]]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f91c69222a05e84b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f91c6922 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`40790 ms` LLM)
- **Rationale**: A shared lab calibration error can accumulate over the hold even when individual rate observations are noisy. Test whether estimating only a lab offset or affine correction from the provided rates improves held-lab transfer without degrading public cell or missing-coupon performance. Leave the verified artifact unchanged until the comparison is available.
- **Alternatives Weighed**: `Add polynomial or rational coupon interactions` (rejected: Neither improved worst held-lab curve error over the linear basis.)

```bash
cat > /tmp/suc_calibration_challenge.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'preloaded_hereditary_secant_hazard_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def subset(q, rows):
    return {k:v[rows] for k,v in q.items()}

def err(curve, te, t, gt):
    initial = te['strain_0']
    secant = (curve[:,-1]-initial)/np.maximum(abs(te['stress_hold_0']),.5)
    crossing = m._first_crossings(initial,curve,te['critical_strain'],t)
    return np.c_[abs(curve-gt['te_clean']).mean(axis=1),abs(secant-gt['te_secant']),abs(crossing-gt['te_cross'])]

for case in pickle.load(open(root+'public_suc01_T_datasets.pkl','rb')):
    data, gt = case['dataset'], case['gt']
    train, test, t = data['train'], data['test'], data['eval_times']
    for lab in [None,*np.unique(train['lab_id'])]:
        take = np.ones(len(test['lab_id']),dtype=bool) if lab is None else test['lab_id']==lab
        tr = train if lab is None else subset(train, train['lab_id']!=lab)
        te = subset(test,take)
        truth = {k:gt[k][take] for k in ('te_clean','te_secant','te_cross')}
        # Recover the original fitted parameters without changing the artifact.
        params = []
        original = m.least_squares
        def capture(*args,**kw):
            fit = original(*args,**kw)
            params.append(fit.x)
            return fit
        m.least_squares = capture
        try:
            out = m.solve_preloaded_hereditary_transfer(dict(data,train=tr,test=te))
        finally:
            m.least_squares = original
        p, tau = params[0], out['estimated_taus']
        (_, _), (v, complete) = m._coupon_design(tr,te)
        branch = m._branches(te,data['past_dt'],t,tau) @ p[2:4]
        drift = v @ p[4:]
        rate_drift = te['instant_rate_obs']-branch[:,0]
        corrections = {}
        for mode in ('base','offset','affine'):
            corrected = drift.copy()
            for a in np.unique(te['lab_id']):
                observed = (te['lab_id']==a)&complete
                rows = te['lab_id']==a
                if mode=='base' or not observed.any():
                    continue
                center = np.median(drift[observed])
                x = drift[observed]-center
                X = np.ones((len(x),1)) if mode=='offset' else np.c_[np.ones(len(x)),x]
                r = rate_drift[observed]-drift[observed]
                coef = least_squares(lambda z:X@z-r,np.zeros(X.shape[1]),loss='soft_l1',f_scale=.012).x
                corrected[rows] += coef[0]
                if len(coef)==2:
                    corrected[rows] += coef[1]*(drift[rows]-center)
            final = np.where(complete,corrected,rate_drift)
            curve = te['strain_0'][:,None]+branch[:,1:]+final[:,None]*t
            if mode=='base': np.testing.assert_allclose(curve,out['predicted_strain'])
            e = err(curve,te,t,truth)
            corrections[mode] = e.mean(0).round(5).tolist()
            if lab is None:
                groups = [(te['lab_id']==a)&(te['stratum']==s) for a in np.unique(te['lab_id']) for s in np.unique(te['stratum'])]
                groups.extend((complete,~complete))
                print('public',mode,'worst cell or missingness',np.max([e[g].mean(0) for g in groups if g.any()],axis=0).round(5))
        print('held lab',lab,'strain/secant/crossing',corrections)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/suc_calibration_challenge.py
```
- **Container Exit Code**: `0` (`722 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public base worst cell or missingness [0.02302 0.05445 0.12378]
public offset worst cell or missingness [0.01974 0.04213 0.12378]
public affine worst cell or missingness [0.02221 0.04213 0.12378]
held lab None strain/secant/crossing {'base': [0.01016, 0.02027, 0.01544], 'offset': [0.01017, 0.02014, 0.01522], 'affine': [0.00956, 0.01924, 0.01633]}
held lab 0 strain/secant/crossing {'base': [0.01003, 0.02193, 0.05754], 'offset': [0.00543, 0.01161, 0.0558], 'affine': [0.00581, 0.01245, 0.05558]}
held lab 1 strain/secant/crossing {'base': [0.03598, 0.06146, 0.01436], 'offset': [0.03594, 0.06156, 0.01351], 'affine': [0.01035, 0.01755, 0.00343]}
held lab 2 strain/secant/crossing {'base': [0.02945, 0.07064, 0.03056], 'offset': [0.02259, 0.05356, 0.02421], 'affine': [0.01257, 0.02819, 0.01427]}
held lab 3 strain/secant/crossing {'base': [0.02066, 0.03958, 0.00585], 'offset': [0.02055, 0.03892, 0.00622], 'affine': [0.00846, 0.01633, 0.00147]}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f91c69222a05e84b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f91c6922 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`48185 ms` LLM)
- **Rationale**: Promote the affine correction because it materially improves held-lab transfer while all observed readout errors remain below tolerance. Verify the saved solver in an isolated process against the schema, independent readout calculations, public lab and stratum slices, and held-lab folds; a regression would block submission.
- **Alternatives Weighed**: `Use only a lab-wise offset` (rejected: It left held-lab strain MAE at 0.0359 for lab 1, versus 0.0104 with the affine correction.)

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/preloaded_hereditary_secant_hazard_solver.py')
s = p.read_text()
anchor = 'def _first_crossings(initial, strain, critical, times):'
assert anchor in s
s = s.replace(anchor, '''def _calibrate_drift(labs, coupon_drift, rate_drift, complete):
    # Reference location and scale errors act affinely on the linear coupon score.
    # Estimate the shared correction from paired measurements, not row-wise rates.
    result = rate_drift.copy()
    for lab in np.unique(labs):
        rows = (labs == lab) & complete
        score = coupon_drift[rows]
        if score.size == 0:
            continue
        centered = score - np.median(score)
        design = np.ones((score.size, 1))
        if np.ptp(score) > 0.0:
            design = np.column_stack((design, centered))
        discrepancy = rate_drift[rows] - score
        correction = least_squares(lambda c: design @ c - discrepancy,
                                   np.zeros(design.shape[1]), loss='soft_l1',
                                   f_scale=0.012).x
        result[rows] = score + design @ correction
    return result


''' + anchor)
old = "    drift = np.where(test_complete, v @ p[4:], test['instant_rate_obs'] - branch[:, 0])"
assert old in s
p.write_text(s.replace(old, "    drift = _calibrate_drift(test['lab_id'], v @ p[4:],\n                             test['instant_rate_obs'] - branch[:, 0], test_complete)"))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'preloaded_hereditary_secant_hazard_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
limits = np.array([.075,.095,.28]); worst = np.zeros(3); worst_tau = 0.

def subset(split, rows):
    return {k:v[rows] for k,v in split.items()}

np.testing.assert_allclose(m._first_crossings(np.array([1.,0.,0.]), np.array([[0.,0.,0.],[2.,0.,2.],[0.,0.,0.]]), np.ones(3), np.array([1.,2.,4.])), [0.,.5,4.])
for case in pickle.load(open(root+'public_suc01_T_datasets.pkl','rb')):
    d, gt = case['dataset'], case['gt']; tr, te, t = d['train'], d['test'], d['eval_times']
    for held in [None, *np.unique(tr['lab_id'])]:
        take = np.ones(len(te['lab_id']),bool) if held is None else te['lab_id']==held
        train = tr if held is None else subset(tr,tr['lab_id']!=held)
        test = subset(te,take)
        out = m.solve_preloaded_hereditary_transfer(dict(d,train=train,test=test))
        tau = out['estimated_taus']; tau_error = np.max(np.abs(tau/gt['taus']-1)); worst_tau=max(worst_tau,tau_error)
        assert .20 <= tau[0] < tau[1] <= 9 and tau_error <= .1
        p, c, h = (out[k] for k in ('predicted_strain','secant_compliance','hazard_crossing_times'))
        np.testing.assert_allclose(c,(p[:,-1]-test['strain_0'])/np.maximum(abs(test['stress_hold_0']),.5))
        expected = []
        for initial,path,critical in zip(test['strain_0'],p,test['critical_strain']):
            if initial >= critical:
                expected.append(0.); continue
            last_t,last_y=0.,initial
            for time,value in zip(t,path):
                if value >= critical:
                    expected.append(last_t+(time-last_t)*(critical-last_y)/(value-last_y)); break
                last_t,last_y=time,value
            else:
                expected.append(t[-1])
        np.testing.assert_allclose(h, expected)
        truth = gt['te_cross'][take]
        e=np.c_[np.abs(p-gt['te_clean'][take]).mean(1),np.abs(c-gt['te_secant'][take]),np.abs(h-truth)]
        groups={'overall':np.ones(len(p),bool)}
        if held is None:
            missing=~np.isfinite(test['coupon_features_raw']).all(1)
            groups.update(complete=~missing,missing=missing,initial=truth==0,interior=(truth>0)&(truth<t[-1]),censored=truth==t[-1])
            for a in np.unique(test['lab_id']):
                lab=test['lab_id']==a; groups[f'lab {a}']=lab
                for b in np.unique(test['stratum']):
                    stratum=test['stratum']==b; groups[f'stratum {b}']=stratum; groups[f'cell {a}/{b}']=lab&stratum
        for name, rows in groups.items():
            if rows.any():
                mae=e[rows].mean(0); worst=np.maximum(worst,mae)
                assert np.all(mae<=limits), (held,name,mae)
        print('held lab',held,'overall MAEs',e.mean(0).round(5))
print('worst evaluated MAEs',worst.round(5),'worst tau relative error',round(worst_tau,6))
PY
```
- **Container Exit Code**: `0` (`851 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/preloaded_hereditary_secant_hazard_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
held lab None overall MAEs [0.00956 0.01924 0.01633]
held lab 0 overall MAEs [0.00581 0.01245 0.05558]
held lab 1 overall MAEs [0.01035 0.01755 0.00343]
held lab 2 overall MAEs [0.01257 0.02819 0.01427]
held lab 3 overall MAEs [0.00846 0.01633 0.00147]
worst evaluated MAEs [0.02221 0.04213 0.12378] worst tau relative error 0.005925

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=794a1b021a5a2455 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_794a1b02 (directory).
```

### Turn 11 — `submit_completed_task` (`15667 ms` LLM)
- **Rationale**: The saved candidate passed isolated schema validation, independent secant and crossing checks, and all evaluated slice tolerances. The calibration improved held-lab transfer without compromising any requested tolerance.
- **Alternatives Weighed**: `Retain reference-only alignment` (rejected: Affine calibration reduced worst held-lab strain and secant MAEs from 0.0360 and 0.0706 to 0.0126 and 0.0282, with no increase in worst-slice crossing MAE.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/preloaded_hereditary_secant_hazard_solver.py': 'FILE:5923'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/preloaded_hereditary_secant_hazard_solver.py`. The isolated schema check passed. Across public slices and leave-one-lab-out folds, the maximum relative relaxation-time error was 0.59%. Worst MAEs were 0.0222 for strain, 0.0421 for secant compliance, and 0.1238 for hazard crossing time. All were below the requested tolerances.`

### Turn 12 — `execute_in_remote_docker` (`36352 ms` LLM)
- **Rationale**: Compare the recurrence with its closed-form expansion on the real nonuniform histories; a discrepancy would identify a history-indexing or preload error. A lab-wise unit change tests whether the solver removes the nuisance it is meant to remove. Replay observed coupon dropouts to see whether the rate fallback remains within tolerance on more measured trajectories; a failure would motivate testing a partial-coupon estimate.
- **Alternatives Weighed**: `Repeat the nonlinear basis and held-lab audit` (rejected: The prior audit already found no worst-lab improvement from nonlinear terms and verified the affine calibration; relay and classification checks do not apply to these measurements.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'preloaded_hereditary_secant_hazard_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for case in pickle.load(open(root+'public_suc01_T_datasets.pkl','rb')):
    d, gt = case['dataset'], case['gt']
    te, tr, t, dt = d['test'], d['train'], d['eval_times'], d['past_dt']
    out = m.solve_preloaded_hereditary_transfer(d)
    tau = out['estimated_taus']
    retain = np.exp(-np.r_[np.cumsum(dt[::-1])[::-1][1:], 0.][:,None]/tau)
    z = te['initial_memory_states'] * np.exp(-dt.sum()/tau) + te['past_stress'] @ (retain * -np.expm1(-dt[:,None]/tau))
    gap = te['stress_hold_0'][:,None] - z
    rhs = np.concatenate(((gap/tau)[:,None,:],gap[:,None,:]*-np.expm1(-t[:,None]/tau)),axis=1)
    print('closed-form branch max discrepancy',abs(rhs-m._branches(te,dt,t,tau)).max())
    np.testing.assert_allclose(rhs,m._branches(te,dt,t,tau),atol=1e-12)
    moved = copy.deepcopy(d)
    for split in ('train','test'):
        q = moved[split]
        for lab, shift, scale in zip(np.unique(q['lab_id']), [2.,-7.,4.,9.], [.7,1.4,.85,1.2]):
            rows = q['lab_id']==lab
            for k in ('ref_coupons','coupon_features_raw'):
                q[k][rows] = shift+scale*q[k][rows]
    shifted = m.solve_preloaded_hereditary_transfer(moved)
    for k in out:
        print('lab-unit change max discrepancy',k,abs(out[k]-shifted[k]).max())
        np.testing.assert_allclose(out[k],shifted[k],atol=1e-5)
    # Replay each measured single-feature dropout at the same lab/stratum on the complete test rows.
    complete = np.isfinite(te['coupon_features_raw']).all(1)
    miss = ~np.isfinite(te['coupon_features_raw'])
    masks = sorted({(int(te['lab_id'][i]),int(te['stratum'][i]),int(j)) for i,j in zip(*np.where(miss))})
    perturbed = copy.deepcopy(d)
    replay = np.zeros(len(complete),bool)
    for lab,stratum,col in masks:
        rows=complete & (te['lab_id']==lab) & (te['stratum']==stratum) & ~replay
        perturbed['test']['coupon_features_raw'][rows,col]=np.nan
        replay |= rows
    drop = m.solve_preloaded_hereditary_transfer(perturbed)
    for name, rows, p in [('replay-before',replay,out),('replay-after',replay,drop),('observed-missing',~complete,out)]:
        if rows.any():
            print(name,int(rows.sum()),np.round([abs(p[k][rows]-gt[g][rows]).mean() for k,g in [('predicted_strain','te_clean'),('secant_compliance','te_secant'),('hazard_crossing_times','te_cross')]],5))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`833 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
closed-form branch max discrepancy 3.552713678800501e-15
lab-unit change max discrepancy estimated_taus 2.7189686058193274e-09
lab-unit change max discrepancy predicted_strain 2.0520802834766982e-09
lab-unit change max discrepancy secant_compliance 2.363298001739622e-09
lab-unit change max discrepancy hazard_crossing_times 1.7768954307939566e-08
replay-before 55 [0.00849 0.01774 0.00512]
replay-after 55 [0.03439 0.07235 0.00834]
observed-missing 9 [0.01974 0.04213 0.12378]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=794a1b021a5a2455 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_794a1b02 (directory).
```

### Turn 13 — `submit_completed_task` (`16531 ms` LLM)
- **Rationale**: The verified artifact agrees with the closed-form history propagation to numerical precision, is invariant to lab-specific affine unit changes, and meets all tolerances on the evaluated public slices, held-lab folds, and dropout replay.
- **Alternatives Weighed**: `Replace the initial-rate fallback with a partial-coupon estimator` (rejected: Replaying observed dropout patterns kept all three readout errors within tolerance; the audit did not establish a need for an additional estimator.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/preloaded_hereditary_secant_hazard_solver.py`. The isolated schema check passed. Across public slices, leave-one-lab-out folds, and replayed coupon dropouts, the maximum relative relaxation-time error was 0.59%. Worst MAEs were 0.0344 for strain, 0.0724 for secant compliance, and 0.1238 for hazard crossing time. All were below the requested tolerances.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M93-Lifecycle",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/preloaded_hereditary_secant_hazard_solver.py": "FILE:5923"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "max_tau_rel_err": 0.005737,
    "max_strain_mae": 0.013277,
    "max_secant_mae": 0.028425,
    "max_crossing_mae": 0.009699,
    "student_elapsed_sec": 0.1061,
    "instance_diagnostics": [
      {
        "seed": 99511,
        "tau_rel": 0.00558,
        "strain_mae": 0.011089,
        "secant_mae": 0.023624,
        "cross_mae": 0.006114
      },
      {
        "seed": 99512,
        "tau_rel": 0.005737,
        "strain_mae": 0.013277,
        "secant_mae": 0.028425,
        "cross_mae": 0.009699
      }
    ],
    "candidate_timing_breakdown": {
      "import_wall_sec": 0.093235,
      "import_cpu_sec": 0.084757,
      "candidate_call_wall_sec": 0.054032,
      "candidate_call_cpu_sec": 0.049123,
      "ipc_and_codec_wall_sec": 0.128262,
      "rpc_call_count": 2,
      "verifier_and_gt_wall_sec": 0.007911,
      "verifier_and_gt_cpu_sec": 0.033509,
      "total_supervisor_wall_sec": 0.28344
    }
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"max_tau_rel_err\": 0.005737, \"max_strain_mae\": 0.013277, \"max_secant_mae\": 0.028425, \"max_crossing_mae\": 0.009699, \"student_elapsed_sec\": 0.1061, \"instance_diagnostics\": [{\"seed\": 99511, \"tau_rel\": 0.00558, \"strain_mae\": 0.011089, \"secant_mae\": 0.023624, \"cross_mae\": 0.006114}, {\"seed\": 99512, \"tau_rel\": 0.005737, \"strain_mae\": 0.013277, \"secant_mae\": 0.028425, \"cross_mae\": 0.009699}]}\n__HEGEL_VERIFIER_AUTH_e0023116fca3b4f5a13b9840acd865cc__:{\"strict_pass\": true, \"reward\": 1.0, \"max_tau_rel_err\": 0.005737, \"max_strain_mae\": 0.013277, \"max_secant_mae\": 0.028425, \"max_crossing_mae\": 0.009699, \"student_elapsed_sec\": 0.1061, \"instance_diagnostics\": [{\"seed\": 99511, \"tau_rel\": 0.00558, \"strain_mae\": 0.011089, \"secant_mae\": 0.023624, \"cross_mae\": 0.006114}, {\"seed\": 99512, \"tau_rel\": 0.005737, \"strain_mae\": 0.013277, \"secant_mae\": 0.028425, \"cross_mae\": 0.009699}], \"candidate_timing_breakdown\": {\"import_wall_sec\": 0.093235, \"import_cpu_sec\": 0.084757, \"candidate_call_wall_sec\": 0.054032, \"candidate_call_cpu_sec\": 0.049123, \"ipc_and_codec_wall_sec\": 0.128262, \"rpc_call_count\": 2, \"verifier_and_gt_wall_sec\": 0.007911, \"verifier_and_gt_cpu_sec\": 0.033509, \"total_supervisor_wall_sec\": 0.28344}}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5343.67,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
