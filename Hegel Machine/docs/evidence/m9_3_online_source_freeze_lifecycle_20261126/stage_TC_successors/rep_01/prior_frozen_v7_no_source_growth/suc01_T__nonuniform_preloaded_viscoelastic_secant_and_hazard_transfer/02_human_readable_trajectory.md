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
| **Turns Used / Max Turns** | `11 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `282,505` (`270,947` / `69,760` / `11,558`) |
| **Total Episode Wall Time (s)** | `442.81s` |

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

### Turn 1 — `execute_in_remote_docker` (`18381 ms` LLM)
- **Rationale**: Inspect the validator and compact data summaries first: the reference layout, observed channels, and time-grid shapes determine the alignment and constitutive fit.

```bash
cd /holdout_task && cat validate_public_schema.py && ls -la; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation robust Cauchy IRLS viscoelastic relaxation'; python3 - <<'PY'
import pickle, numpy as np
with open('/holdout_task/public_suc01_T_datasets.pkl','rb') as f: d=pickle.load(f)
def show(x,p='',depth=0):
 if isinstance(x,dict):
  print(p,'keys',list(x)[:40]);
  if depth<4:
   for k,v in list(x.items())[:15]: show(v,p+'/'+str(k),depth+1)
 elif isinstance(x,(tuple,list)):
  print(p,type(x).__name__,len(x));
  if depth<4:
   for i,v in enumerate(x[:4]): show(v,p+'/'+str(i),depth+1)
 elif isinstance(x,np.ndarray):
  print(p,x.shape,x.dtype,'first',x[:2] if x.ndim==1 else x[:1].tolist());
  if x.dtype.kind in 'fiu': print('    min/max',np.nanmin(x),np.nanmax(x),'missing',np.sum(~np.isfinite(x)))
 else: print(p,repr(x)[:250])
show(d)
PY
```
- **Container Exit Code**: `0` (`702 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
in/max -5.732496399363159 7.857921168132616 missing 71
/0/dataset/test/past_stress (90, 22) float64 first [[0.17514512115823427, 1.1194763089445978, 0.283666904629292, 1.142461527740406, 0.9753476692790153, 0.6784089091884651, 1.1021783892360868, 1.2180618207619187, 0.6696769049154038, 2.0194480715210394, 1.1100209996012014, 1.7960711374157958, 1.2413765883292394, 1.3847464814274288, 1.8556526043290846, 2.518800644313414, 1.6533721781102357, 2.0326596023193746, 1.5688019078319166, 1.7319266844688699, 1.72415488854746, 0.5841502330318054]]
    min/max -24.129888739093715 24.56033975881988 missing 0
/0/dataset/test/strain_0 (90,) float64 first [0.35702858 0.51890486]
    min/max -0.6160693084677578 0.7595241116233458 missing 0
/0/dataset/test/stress_hold_0 (90,) float64 first [0.55739059 1.85838836]
    min/max -2.1420292872374787 2.1387064633418142 missing 0
/0/dataset/test/instant_rate_obs (90,) float64 first [-0.76026931  2.89067114]
    min/max -6.651433900375634 5.609172373348098 missing 0
/0/dataset/test/initial_memory_states (90, 2) float64 first [[0.22517016686352298, 0.5017877524334343]]
    min/max -6.350492406642847 5.320941753938318 missing 0
/0/dataset/test/critical_strain (90,) float64 first [0.63559661 1.06767827]
    min/max -0.3585496700507308 1.2866756479567243 missing 0
/0/gt keys ['taus', 'theta', 'vp_theta', 'te_Z', 'te_clean', 'te_true_rate0', 'te_cross', 'te_secant']
/0/gt/taus (2,) float64 first [0.46 3.35]
    min/max 0.46 3.35 missing 0
/0/gt/theta (7,) float64 first [0.74 0.52]
    min/max -0.12 0.74 missing 0
/0/gt/vp_theta (2,) float64 first [0. 0.]
    min/max 0.0 0.0 missing 0
/0/gt/te_Z (90, 4) float64 first [[1.5650374201382022, -0.2698303912421538, 1.6130503372859517, -0.9165920645688288]]
    min/max -3.0595818418082628 2.702223682280967 missing 0
/0/gt/te_clean (90, 12) float64 first [[0.2711711403542002, 0.22721741708630522, 0.25707710229561687, 0.3721352277872483, 0.5792160588322893, 0.8686045069098748, 1.2283766657449706, 1.6798417145237021, 2.1923275524621944, 2.822403853275034, 3.5129104080362925, 4.234617461858734]]
    min/max -8.079833792301418 9.103935670791884 missing 0
/0/gt/te_true_rate0 (90,) float64 first [-0.76516176  2.87812647]
    min/max -6.657124876392119 5.597091742454995 missing 0
/0/gt/te_cross (90,) float64 first [1.5571546  0.24597169]
    min/max 0.04237827912635525 7.9 missing 0
/0/gt/te_secant (90,) float64 first [6.95668168 1.52928602]
    min/max -4.54465622738818 7.1362868112813365 missing 0
```

### Turn 2 — `execute_in_remote_docker` (`22654 ms` LLM)
- **Rationale**: Inspect the complete training schema and lab-wise reference scales to detect mismatched alignment assumptions and determine which hold-law terms can be estimated from the supplied channels.

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_affine_projection.robust_residual_weights; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_suc01_T_datasets.pkl','rb'))
print(type(p),len(p))
for i,b in enumerate(p):
 d=b['dataset']; print(i,list(b),list(d),[(k,np.shape(v)) for k,v in d['train'].items()]); print('times',d['eval_times'],'dt',d['past_dt']); print('ground truth fields',list(b['gt']))
 if i==0:
  a=d['train']; print('strain/rate/hold/initial (first 5)'); print(np.column_stack([a[k][:5] for k in ['strain_0','stress_hold_0','instant_rate_obs','strain_curves_obs']]));
  for g in range(4):
   m=a['lab_id']==g
   for k in ['coupon_features_raw','ref_coupons']:
    x=a[k][m]; print(g,k,'median',np.round(np.nanmedian(x,axis=0),2),'sd',np.round(np.nanstd(x,axis=0),2))
PY
```
- **Container Exit Code**: `0` (`621 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
95 0.0299 0.0331 0.0361 0.0386 0.0395 0.0414 0.0425 0.0456
 0.0497 0.0609 0.0551 0.0604 0.0636 0.0808 0.0786 0.0836 0.0893 0.0986
 0.1056 0.1066]
ground truth fields ['taus', 'theta', 'vp_theta', 'te_Z', 'te_clean', 'te_true_rate0', 'te_cross', 'te_secant']
strain/rate/hold/initial (first 5)
[[-0.58411281 -1.05445389  0.05427442 -0.58279199 -0.58814551 -0.62709746
  -0.67825268 -0.76481398 -0.86935752 -2.03226591 -1.2085483  -1.42829304
  -1.70663533 -2.03975271 -2.33511209]
 [ 0.36648253  0.69399762  0.17210542  0.37580452  0.35918097  1.10984224
   0.14324833 -0.04650164 -0.34690568 -0.61854434 -0.97137387 -1.36865902
  -1.82048702 -2.29867358 -2.79964903]
 [-0.51440045 -2.06215889 -3.95171555 -1.0230837  -1.54666769 -2.07713654
  -2.44061886 -2.80033703 -3.10756419 -3.42310939 -4.74989748 -4.19548529
  -4.6429648  -4.34762606 -5.62920922]
 [ 0.41382183  1.90060773  2.37951012  0.69469523  0.9943055   1.20852317
   1.33182916  1.38668973  1.33051867  1.25381318  1.09746572  1.63884065
   0.58685985  0.27359643 -0.08869159]
 [ 0.05461523  1.27384925  4.10254055  0.6073849   1.15153594  0.74933955
   2.23064516  2.73229766  3.2117906   3.71536623  4.3115392   4.95206999
   5.68385384  6.48218083  7.26759574]]
0 coupon_features_raw median [2.53 3.45 2.96 3.25] sd [1.52 1.23 1.39 0.88]
0 ref_coupons median [3.   3.26 2.97 3.27 3.37 3.   3.37 3.37 2.77 2.88 2.94 3.39 2.9  3.34
 3.19 2.75 3.09 3.72 3.15 3.44] sd [1.27 1.51 1.52 0.88 1.36 1.3  1.34 1.18 1.48 1.25 1.39 1.26 1.35 1.25
 0.86 1.18 1.06 1.76 1.15 1.28]
1 coupon_features_raw median [-3.38 -3.32 -3.58 -3.39] sd [0.7  0.58 0.64 0.64]
1 ref_coupons median [-3.63 -3.44 -3.38 -3.68 -3.34 -3.36 -3.64 -3.33 -3.21 -3.35 -3.36 -3.42
 -3.13 -3.04 -3.38 -3.24 -3.19 -3.5  -3.34 -3.49] sd [0.95 0.69 0.62 0.97 0.83 0.71 0.78 0.77 0.85 0.68 0.89 0.76 0.68 0.79
 0.54 0.74 0.82 0.82 0.62 0.81]
2 coupon_features_raw median [4.5  3.3  4.22 3.38] sd [1.74 1.51 1.08 1.09]
2 ref_coupons median [3.36 3.54 3.55 3.45 3.14 3.5  3.6  3.63 3.63 3.8  3.86 3.45 3.56 3.52
 3.72 3.75 4.24 3.26 3.68 3.74] sd [1.32 1.32 1.42 1.2  1.42 1.57 1.39 1.43 1.66 1.3  1.43 1.3  1.61 1.5
 1.35 1.23 1.37 1.54 1.03 1.37]
3 coupon_features_raw median [-3.17 -2.83 -3.12 -2.85] sd [0.96 0.82 0.85 0.83]
3 ref_coupons median [-2.82 -2.8  -2.58 -2.9  -3.22 -2.94 -2.82 -3.12 -2.86 -3.16 -2.76 -3.05
 -2.88 -2.5  -2.64 -2.85 -2.93 -2.62 -2.67 -2.76] sd [0.72 0.96 1.06 0.89 0.85 0.86 0.79 0.72 0.7  0.71 0.83 0.79 0.8  0.78
 0.99 0.86 0.77 1.   0.87 0.87]
```

### Turn 3 — `execute_in_remote_docker` (`52871 ms` LLM)
- **Rationale**: A training-only shape fit tests whether the candidate basis explains the observed curves and rates. Regressing the fitted amplitudes identifies which terms should be retained in the shared constitutive fit.
- **Alternatives Weighed**: `Fit a generic regression at each evaluation time` (rejected: It does not enforce the measured initial derivative and cannot identify physical relaxation times.)

```bash
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.stats import median_abs_deviation as mad
D=pickle.load(open('/holdout_task/public_suc01_T_datasets.pkl','rb'))[0]['dataset']; a=D['train']; t=D['eval_times']; dt=D['past_dt']; h=a['stress_hold_0']; y=a['strain_curves_obs']-a['strain_0'][:,None]; r=a['instant_rate_obs']; P=a['past_stress']; Z=a['coupon_features_raw'].copy(); R=a['ref_coupons']; g=a['lab_id']; s=a['stratum'];
for j in np.unique(g):
 m=g==j; Z[m]=(Z[m]-np.median(np.nanmean(R[m],axis=0)))/np.median(np.nanstd(R[m],axis=0)); print('cells',j,np.bincount(s[m]))

def basis(taus):
 tau=np.asarray(taus); z=a['initial_memory_states'].copy();
 for n,d in enumerate(dt):
  e=np.exp(-d/tau); z=e*z+(1-e)*P[:,n,None]
 B=np.column_stack([-np.expm1(-t[:,None]/tau),t]); return B,h[:,None]-z

def fit(taus):
 B,G=basis(taus); X=np.vstack([B,np.r_[1/np.asarray(taus),1][None,:]*2]); Y=np.column_stack([y,2*r]); coef=Y@np.linalg.pinv(X).T
 for _ in range(14):
  v=coef@X.T-Y; w=1/(1+(v/.035)**2); coef=np.linalg.solve(np.einsum('nk,ki,kj->nij',w,X,X), (w*Y)@X[...,None] if False else ((w*Y)@X)[...,None])[...,0]
 v=coef@X.T-Y
 return v,coef,G
q=least_squares(lambda u:fit(u)[0].ravel(),[.5,3.2],bounds=([.2,1.2],[1.19,9]),loss='cauchy',f_scale=.035,max_nfev=60); v,c,G=fit(q.x); print('shape-only tau',q.x,'median abs residual',np.median(abs(v),axis=0).round(4)); print('ratios percentiles')
W=np.column_stack([c[:,:2]/G,c[:,2]/h,c[:,2],c[:,:2]]); print(np.nanpercentile(W,[5,25,50,75,95],axis=0).round(3)); print('feature correlations and affine slopes on centrally trimmed response')
for j,w in enumerate(W.T):
 m=np.isfinite(Z).all(1)&(w>=np.percentile(w,10))&(w<=np.percentile(w,90)); x=np.column_stack([np.ones(m.sum()),Z[m],h[m],G[m]]); beta=np.linalg.lstsq(x,w[m],rcond=None)[0]; print(j,'beta',beta.round(3),'MAD',mad(w[m]-x@beta).round(3))
print('typical sample',np.column_stack([Z,c,G,h])[:12].round(3))
PY
```
- **Container Exit Code**: `0` (`800 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cells 0 [20  8  8]
cells 1 [ 8 22  8]
cells 2 [ 8  8 22]
cells 3 [18  9  9]
shape-only tau [0.46015221 3.32559015] median abs residual [0.0064 0.0058 0.005  0.0057 0.0056 0.0065 0.0048 0.0051 0.0046 0.0052
 0.0041 0.0029 0.0012]
ratios percentiles
[[ 0.72   0.425 -0.626 -0.4   -1.666 -1.775]
 [ 0.737  0.502 -0.198 -0.224 -1.007 -0.729]
 [ 0.739  0.517 -0.017  0.086  0.025 -0.112]
 [ 0.743  0.533  0.2    0.36   1.018  0.653]
 [ 0.766  0.626  0.488  0.616  1.842  1.683]]
feature correlations and affine slopes on centrally trimmed response
0 beta [ 0.74 -0.   -0.   -0.   -0.   -0.    0.    0.  ] MAD 0.003
1 beta [ 0.516  0.    -0.001  0.003  0.006 -0.001  0.004 -0.002] MAD 0.01
2 beta [-0.027 -0.029 -0.04   0.03   0.099  0.011 -0.025  0.006] MAD 0.126
3 beta [ 0.081  0.147 -0.117  0.096 -0.079 -0.002  0.001  0.001] MAD 0.003
4 beta [ 0.001  0.    -0.001 -0.001  0.     0.     0.74  -0.   ] MAD 0.004
5 beta [-0.004 -0.007  0.     0.005  0.001  0.004 -0.002  0.517] MAD 0.023
typical sample [[-1.75   0.138  0.139  1.166  0.105  0.339 -0.275  0.141  0.636 -1.054]
 [-1.405  2.242  0.009 -0.068  0.334 -0.593 -0.375  0.454 -1.179  0.694]
 [-1.065  1.136 -1.303  0.345 -1.549 -0.728 -0.367 -2.087 -1.445 -2.062]
 [   nan  0.804  0.022  2.117  1.09   1.137 -0.333  1.466  2.203  1.901]
 [ 2.195 -0.532  0.993 -0.432  1.452  1.166  0.595  1.964  2.278  1.274]
 [ 0.936 -0.809  0.985 -1.502 -0.393 -1.012  0.531 -0.547 -1.85  -1.245]
 [ 1.248 -0.824  1.679 -0.996 -1.244 -0.396  0.609 -1.686 -0.696 -0.775]
 [ 0.482 -0.327  0.43  -0.605 -1.001  0.079  0.281 -1.359  0.221 -0.996]
 [ 0.408  0.718  0.486  0.293 -1.285 -1.285  0.078 -1.743 -2.491 -2.02 ]
 [   nan  0.93  -0.385 -0.276 -0.897 -0.621 -0.037 -1.23  -1.178 -0.83 ]
 [ 1.675 -0.084  1.319 -1.116 -1.031  0.116  0.556 -1.392  0.242 -1.23 ]
 [-0.593  1.058 -1.742  2.539 -1.42  -1.167 -0.502 -1.927 -2.193 -1.977]]
```

### Turn 4 — `execute_in_remote_docker` (`47551 ms` LLM)
- **Rationale**: The fitted branch amplitudes already scale with their propagated stress gaps, while drift tracks the aligned coupons. This fit tests whether that shared constraint leaves systematic residuals and whether reference alignment alone is accurate enough for drift transfer.
- **Alternatives Weighed**: `Use the observed initial rate as an exact constraint` (rejected: Its measurement error would propagate linearly to late strain; estimate the drift from curves and rates jointly before deciding how to combine rate and coupon evidence at prediction time.)

```bash
cat > /tmp/profile_hold.py <<'PY'
import pickle
import numpy as np
from scipy.optimize import least_squares


def branch(taus, a, dt, t):
    z = a['initial_memory_states'].copy()
    for j, d in enumerate(dt):
        u = -np.expm1(-d / taus)
        z += u * (a['past_stress'][:, j, None] - z)
    gap = a['stress_hold_0'][:, None] - z
    return gap[:, None, :] * (-np.expm1(-t[:, None] / taus)), gap / taus


def fit(a, dt, t):
    y = np.column_stack([a['strain_curves_obs'] - a['strain_0'][:, None], a['instant_rate_obs']])
    v = np.r_[t, 1.]
    scale = np.r_[np.full(t.size, .014), .017]
    vs = v / scale
    def profile(p):
        b, r = branch(p[:2], a, dt, t)
        m = (y - np.column_stack([b @ p[2:], r @ p[2:]])) / scale
        # A median seed prevents isolated Cauchy glitches from capturing the drift.
        c = np.median(m / vs, axis=1)
        for _ in range(18):
            res = m - c[:, None] * vs
            w = 1 / (1 + res**2)
            c = (w*m) @ vs / (w @ (vs**2))
        return (m - c[:, None]*vs).ravel(), c
    starts = ([.48, 3.4, .7, .5], [.27, 1.4, .6, .5], [.95, 6., .7, .6])
    fits = [least_squares(lambda p: profile(p)[0], p0, loss='cauchy', f_scale=1, bounds=([.2,.200001,-4,-4],[8.999999,9,4,4]), max_nfev=120) for p0 in starts]
    sol = min((q for q in fits if q.x[0] < q.x[1]), key=lambda q: q.cost)
    return sol.x, profile(sol.x)[1], [(q.x,q.cost) for q in fits], profile(sol.x)[0].reshape(y.shape)*scale

if __name__ == '__main__':
    d=pickle.load(open('/holdout_task/public_suc01_T_datasets.pkl','rb'))[0]['dataset']; a=d['train']; t=d['eval_times']; dt=d['past_dt']; p,c,q,res=fit(a,dt,t); print('optima:',q); print('residual absolute quantiles:',np.quantile(abs(res),[.50,.80,.95],axis=0).round(5))
    g=a['lab_id']; s=a['stratum']; x=a['coupon_features_raw'].copy(); refs=a['ref_coupons']; J=np.eye(4)[g]; S=np.eye(3)[s]; good=np.isfinite(x).all(1); cs=[]
    print('lab affine slopes, intercepts and drift residual spread')
    for j in range(4):
        m=(g==j)&good; R=refs[g==j]; offset=np.nanmean(R); gain=np.nanstd(R); x[g==j]=(x[g==j]-offset)/gain; F=np.column_stack([np.ones(m.sum()),x[m]]); w=np.linalg.lstsq(F,c[m],rcond=None)[0]; cs.append(w); print(j,'loc/scale',offset,gain,'coefs',w.round(5),'residual',np.quantile(c[m]-F@w,[.1,.5,.9]).round(5))
    for F in (np.column_stack([np.ones(len(c)),x]),np.column_stack([J,x]),np.column_stack([J,x,S[:,1:]])):
        f=F[good]; w=np.linalg.lstsq(f,c[good],rcond=None)[0]; print('reference slope',w.round(5),'residual sd',np.std(c[good]-f@w)); print('by cell mean',[(j,k,round(np.mean(c[m]-F[m]@w),5)) for j in range(4) for k in range(3) if np.any(m:=good&(g==j)&(s==k))])
    print('rate-based drift noise',np.quantile(abs(res[:,-1]),[.5,.8,.95,.99]))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/profile_hold.py
```
- **Container Exit Code**: `0` (`631 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
optima: [(array([0.46082209, 3.36648273, 0.74128286, 0.52043266]), 787.4474520367176), (array([0.46082209, 3.36648233, 0.74128285, 0.52043256]), 787.4474520372272), (array([0.4608221 , 3.36648309, 0.74128286, 0.52043273]), 787.4474520367419)]
residual absolute quantiles: [[0.00625 0.00602 0.00569 0.00569 0.00559 0.00661 0.00555 0.00593 0.00485
  0.00459 0.00387 0.00356 0.00612]
 [0.01181 0.01249 0.01206 0.011   0.0109  0.01305 0.01014 0.01059 0.00938
  0.01044 0.00959 0.01032 0.01148]
 [0.95237 0.82786 0.87241 0.89859 0.50079 0.99771 0.02173 0.94569 0.02108
  0.8419  0.9734  0.95482 0.01786]]
lab affine slopes, intercepts and drift residual spread
0 loc/scale 3.1639152548999063 1.3073091540371653 coefs [ 0.08256  0.14841 -0.11904  0.09894 -0.07942] residual [-7.10e-04 -6.00e-05  1.03e-03]
1 loc/scale -3.397816989749739 0.7806490090221504 coefs [ 0.08006  0.14825 -0.11858  0.09894 -0.07919] residual [-8.4e-04  8.0e-05  6.7e-04]
2 loc/scale 3.5204021859253385 1.4091227485486215 coefs [ 0.07734  0.14967 -0.12011  0.09979 -0.08023] residual [-8.6e-04  8.0e-05  8.6e-04]
3 loc/scale -2.8865877800115354 0.8602732926330319 coefs [ 0.08121  0.15191 -0.1216   0.10122 -0.08071] residual [-9.0e-04  4.0e-05  8.8e-04]
reference slope [ 0.08027  0.14952 -0.1201   0.0992  -0.08061] residual sd 0.003644888037073922
by cell mean [(0, 0, 0.00453), (0, 1, 0.00176), (0, 2, -0.00024), (1, 0, 0.00299), (1, 1, -0.00016), (1, 2, -0.00283), (2, 0, -0.00294), (2, 1, -0.00269), (2, 2, -0.00243), (3, 0, -0.00414), (3, 1, 0.00204), (3, 2, 0.00693)]
reference slope [ 0.08293  0.08025  0.07741  0.08046  0.14981 -0.12007  0.09947 -0.08047] residual sd 0.0031300384451826892
by cell mean [(0, 0, 0.00229), (0, 1, -0.00117), (0, 2, -0.00342), (1, 0, 0.00345), (1, 1, -0.00014), (1, 2, -0.00302), (2, 0, 0.00015), (2, 1, 2e-05), (2, 2, -6e-05), (3, 0, -0.00391), (3, 1, 0.00173), (3, 2, 0.00636)]
reference slope [ 0.08263  0.07982  0.07704  0.08015  0.1497  -0.12     0.09939 -0.08044
  0.00047  0.00064] residual sd 0.003127242528435729
by cell mean [(0, 0, 0.00238), (0, 1, -0.00128), (0, 2, -0.00349), (1, 0, 0.00363), (1, 1, -0.00018), (1, 2, -0.00305), (2, 0, 0.00025), (2, 1, -5e-05), (2, 2, -7e-05), (3, 0, -0.00384), (3, 1, 0.00163), (3, 2, 0.00633)]
rate-based drift noise [0.00612253 0.01148387 0.01785666 0.02213281]
```

### Turn 5 — `execute_in_remote_docker` (`70449 ms` LLM)
- **Rationale**: The training data support shared branch amplitudes and coupon-dependent linear drift. Fit only from training responses, then check public held-out errors by lab, stratum, missingness, and crossing status to detect a weak transfer slice.
- **Alternatives Weighed**: `Estimate test drift entirely from the initial-rate measurement` (rejected: The complete-coupon fit resolved drift much more precisely than that measurement; give the rate more weight only when coupon information is missing.)

```bash
cat > /holdout_task/preloaded_hereditary_secant_hazard_solver.py <<'PY'
"""Two-branch hereditary transfer from nonuniform, preloaded histories."""
import numpy as np
from scipy.optimize import least_squares


def _branch(tau, rows, dt, times):
    memory = np.array(rows['initial_memory_states'], dtype=float, copy=True)
    past = np.asarray(rows['past_stress'])
    for j, step in enumerate(dt):
        memory += -np.expm1(-step / tau) * (past[:, j, None] - memory)
    gap = np.asarray(rows['stress_hold_0'])[:, None] - memory
    return gap[:, None, :] * -np.expm1(-times[:, None] / tau), gap / tau


def _fit_physics(rows, dt, times):
    target = np.column_stack((np.asarray(rows['strain_curves_obs']) - np.asarray(rows['strain_0'])[:, None], rows['instant_rate_obs']))
    scale = np.r_[np.full(times.size, .014), .017]
    t = np.r_[times, 1.] / scale

    def decode(p):
        return np.array([p[0], p[0] + (9. - p[0]) * p[1]])

    def profile(p):
        curves, rate = _branch(decode(p), rows, dt, times)
        centered = (target - np.column_stack((curves @ p[2:], rate @ p[2:]))) / scale
        drift = np.median(centered / t, axis=1)
        for _ in range(18):
            residual = centered - drift[:, None] * t
            weights = 1. / (1. + residual**2)
            drift = np.sum(weights * centered * t, axis=1) / np.sum(weights * t**2, axis=1)
        return centered - drift[:, None] * t, drift

    best = None
    for lo, hi in ((.48, 3.4), (.27, 1.4), (.95, 6.), (2.2, 7.5)):
        start = [lo, (hi - lo) / (9. - lo), .7, .5]
        fit = least_squares(lambda p: profile(p)[0].ravel(), start,
                            bounds=([.2, 1e-6, -np.inf, -np.inf], [9. - 1e-6, 1., np.inf, np.inf]),
                            loss='cauchy', x_scale='jac', max_nfev=100)
        if best is None or fit.cost < best.cost:
            best = fit
    return decode(best.x), best.x[2:], profile(best.x)[1]


def _coupons(train, test):
    labs = np.r_[train['lab_id'], test['lab_id']]
    refs = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    for lab in np.unique(labs):
        sel = labs == lab
        x[sel] = (x[sel] - np.nanmean(refs[sel])) / np.nanstd(refs[sel])
    n = len(train['lab_id'])
    cells = np.column_stack((train['lab_id'], train['stratum']))
    _, ids, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    w = 1. / counts[ids]
    observed = np.isfinite(x)
    fill = np.sum(np.where(observed[:n], x[:n], 0.) * w[:, None], axis=0) / np.sum(observed[:n] * w[:, None], axis=0)
    return np.where(observed, x, fill), observed, w


def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _predict_drift(train, test, fitted, train_rate, test_rate):
    x, observed, w = _coupons(train, test)
    n = len(fitted)
    design = np.column_stack((np.ones(len(x)), x))
    complete = observed[:n].all(axis=1)

    def regress(sel):
        root = np.sqrt(w[sel])
        return np.linalg.lstsq(design[:n][sel] * root[:, None], fitted[sel] * root, rcond=None)[0]

    common = regress(complete)
    prior = design @ common
    # Use labeled complete specimens to remove small residual calibration errors
    # after panel normalization; keep the common calibration for an unseen lab.
    tlab = np.asarray(train['lab_id'])
    lab = np.r_[tlab, test['lab_id']]
    for j in np.unique(tlab):
        sel = (tlab == j) & complete
        if np.count_nonzero(sel) >= 2 * design.shape[1]:
            prior[lab == j] = design[lab == j] @ regress(sel)

    rate_sd = max(_mad(train_rate - fitted), .002)
    coupon_sd = max(_mad((fitted - prior[:n])[complete]), .0015)
    test_prior = prior[n:]
    missing = ~observed[n:]
    # Missing features make the balanced-fill prediction less informative.
    missing_scale = np.std(x[:n], axis=0) * common[1:]
    coupon_variance = coupon_sd**2 + missing @ (missing_scale**2)
    rate_weight = coupon_variance / (coupon_variance + rate_sd**2)
    return test_prior + rate_weight * (test_rate - test_prior)


def _crossings(times, initial, curves, critical):
    time = np.r_[0., times]
    strain = np.column_stack((initial, curves))
    reached = strain >= np.asarray(critical)[:, None]
    first = np.argmax(reached, axis=1)
    out = np.full(len(strain), time[-1])
    out[reached[:, 0]] = 0.
    rows = np.flatnonzero(first > 0)
    j = first[rows]
    low = strain[rows, j - 1]
    fraction = (np.asarray(critical)[rows] - low) / (strain[rows, j] - low)
    out[rows] = time[j - 1] + fraction * (time[j] - time[j - 1])
    return out


def solve_preloaded_hereditary_transfer(dataset: dict) -> dict:
    """Estimate shared relaxation times, then transfer strain and terminal readouts."""
    train, test = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    tau, amplitudes, fitted = _fit_physics(train, dt, times)
    _, train_rate = _branch(tau, train, dt, times)
    curves, test_rate = _branch(tau, test, dt, times)
    drift = _predict_drift(train, test, fitted,
                           np.asarray(train['instant_rate_obs']) - train_rate @ amplitudes,
                           np.asarray(test['instant_rate_obs']) - test_rate @ amplitudes)
    initial = np.asarray(test['strain_0'])
    prediction = initial[:, None] + curves @ amplitudes + drift[:, None] * times
    return {
        'estimated_taus': tau,
        'predicted_strain': prediction,
        'secant_compliance': (prediction[:, -1] - initial) / np.maximum(np.abs(test['stress_hold_0']), .5),
        'hazard_crossing_times': _crossings(times, initial, prediction, test['critical_strain']),
    }
PY
cd /holdout_task && python3 -I validate_public_schema.py && python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p='preloaded_hereditary_secant_hazard_solver.py'; sp=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
b=pickle.load(open('public_suc01_T_datasets.pkl','rb'))[0]; d,gt=b['dataset'],b['gt']; a=d['test']; o=m.solve_preloaded_hereditary_transfer(d)
print('taus',o['estimated_taus'],'relative errors',abs(o['estimated_taus']/gt['taus']-1))
err=np.column_stack([abs(o['predicted_strain']-gt['te_clean']).mean(1),abs(o['secant_compliance']-gt['te_secant']),abs(o['hazard_crossing_times']-gt['te_cross'])]); print('overall',err.mean(0));
for name,g in [('lab',a['lab_id']),('stratum',a['stratum']),('cell',3*a['lab_id']+a['stratum']),('missing',~np.isfinite(a['coupon_features_raw']).all(1)),('crossing',gt['te_cross']<d['eval_times'][-1])]:
 print(name,[(int(j),int(sum(g==j)),np.round(err[g==j].mean(0),5).tolist()) for j in np.unique(g)])
print('worst strain horizon',abs(o['predicted_strain']-gt['te_clean']).mean(0).max()); print('worst hazard rows',sorted(enumerate(err[:,2]),key=lambda p:p[1])[-5:])
PY
```
- **Container Exit Code**: `0` (`2024 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/preloaded_hereditary_secant_hazard_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
taus [0.4608221 3.3664828] relative errors [0.00178716 0.00492024]
overall [0.00375829 0.00625757 0.01228892]
lab [(0, 22, [0.0027, 0.00473, 0.04485]), (1, 23, [0.00546, 0.00729, 0.00048]), (2, 23, [0.00388, 0.0073, 0.00385]), (3, 22, [0.00291, 0.00562, 0.0009])]
stratum [(0, 33, [0.00468, 0.00867, 0.02991]), (1, 29, [0.00224, 0.0036, 0.00047]), (2, 28, [0.00424, 0.00616, 0.00376])]
cell [(0, 12, [0.00357, 0.00697, 0.08156]), (1, 5, [0.00121, 0.00186, 0.00103]), (2, 5, [0.0021, 0.00223, 0.00054]), (3, 5, [0.00738, 0.00932, 5e-05]), (4, 13, [0.00301, 0.00519, 0.0003]), (5, 5, [0.00992, 0.01072, 0.00138]), (6, 5, [0.00781, 0.01491, 0.0011]), (7, 5, [0.00169, 0.0022, 1e-05]), (8, 13, [0.00321, 0.00633, 0.00639]), (9, 11, [0.00325, 0.00741, 0.00024]), (10, 6, [0.00188, 0.00276, 0.00074]), (11, 5, [0.00341, 0.00512, 0.00253])]
missing [(0, 81, [0.00201, 0.00249, 0.00078]), (1, 9, [0.01945, 0.04015, 0.11583])]
crossing [(0, 27, [0.00412, 0.00794, 0.0]), (1, 63, [0.0036, 0.00554, 0.01756])]
worst strain horizon 0.008078462235273738
worst hazard rows [(18, 0.0049444006895544845), (17, 0.005745415294996725), (29, 0.010846251985856092), (31, 0.06740129419152652), (43, 0.973935404914827)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=698f7a25d90577b3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_698f7a25 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`26858 ms` LLM)
- **Rationale**: Leaving each lab out of the fit can reveal dependence on lab-specific labeled calibration; a failure would call for revising shared calibration. A targeted crossing check can catch returning a later crossing or mishandling an exact touch.
- **Alternatives Weighed**: `Tune to the largest public crossing error` (rejected: One labeled holdout trajectory does not justify changing the required first-passage rule or fitting to held-out labels.)

```bash
cd /holdout_task && python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('solver','preloaded_hereditary_secant_hazard_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
b=pickle.load(open('public_suc01_T_datasets.pkl','rb'))[0]; d=b['dataset']; gt=b['gt']; train=d['train']; test=d['test']; limits=np.array([.10,.075,.095,.28]); worst=np.zeros(4)
for lab in np.unique(test['lab_id']):
    tr=train['lab_id']!=lab; te=test['lab_id']==lab
    ds={**d,'train':{k:v[tr] for k,v in train.items()},'test':{k:v[te] for k,v in test.items()}}
    o=m.solve_preloaded_hereditary_transfer(ds)
    errors=np.array([np.max(np.abs(o['estimated_taus']/gt['taus']-1)), np.mean(np.abs(o['predicted_strain']-gt['te_clean'][te])), np.mean(np.abs(o['secant_compliance']-gt['te_secant'][te])), np.mean(np.abs(o['hazard_crossing_times']-gt['te_cross'][te]))]); worst=np.maximum(worst,errors)
    print('unseen lab',lab,'tau max rel / strain / secant / hazard:',errors)
assert np.all(worst<=limits), worst
# Initial hit, no hit, exact endpoint touch, and first of two upcrossings.
t=np.array([1.,2.,4.]); y=np.array([[0.,0.,0.],[.2,.4,.6],[0.,1.,0.],[2.,0.,2.]])
np.testing.assert_allclose(m._crossings(t,np.array([1.,0.,0.,0.]),y,np.ones(4)),[0.,4.,2.,.5]); print('Readout semantics passed; worst unseen-lab errors:',worst)
PY
```
- **Container Exit Code**: `0` (`1804 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
unseen lab 0 tau max rel / strain / secant / hazard: [0.00131499 0.00816379 0.01647977 0.05201718]
unseen lab 1 tau max rel / strain / secant / hazard: [0.0115707  0.0336171  0.05794717 0.01368881]
unseen lab 2 tau max rel / strain / secant / hazard: [0.00511068 0.02455995 0.05913956 0.02476734]
unseen lab 3 tau max rel / strain / secant / hazard: [0.00339679 0.01753886 0.03362533 0.00699447]
Readout semantics passed; worst unseen-lab errors: [0.0115707  0.0336171  0.05913956 0.05201718]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=698f7a25d90577b3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_698f7a25 (directory).
```

### Turn 7 — `submit_completed_task` (`15921 ms` LLM)
- **Rationale**: The saved artifact passed isolated Python schema validation, public holdout and leave-one-lab-out thresholds, and direct tests of the required first-crossing semantics.
- **Alternatives Weighed**: `Tune against the largest individual public crossing error` (rejected: All tested slices passed, and tuning to a public holdout row would compromise the validation.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/preloaded_hereditary_secant_hazard_solver.py': 'FILE:5697'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/preloaded_hereditary_secant_hazard_solver.py`. On the public holdout, the maximum relative error in the relaxation times was 0.49%, and the strain, secant compliance, and hazard crossing MAEs were 0.00376, 0.00626, and 0.01229. All four metrics also passed when each lab was held out of training.`

### Turn 8 — `execute_in_remote_docker` (`40767 ms` LLM)
- **Rationale**: An independent closed-form memory calculation and a derivative check can expose time-index or sign errors. Held-out-cell residuals test whether lab calibration or extra algebraic terms materially improve drift transfer, and preload quartiles check whether the largest surviving initial memory erodes the output margins.
- **Alternatives Weighed**: `Test relay modes or within-batch classification accuracy` (rejected: The input contains continuous internal stresses and continuous prediction targets, not relay states or class labels. Lab-stratum holdouts test the relevant calibration confounder.)

```bash
cat > /tmp/audit_preloaded.py <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'preloaded_hereditary_secant_hazard_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
b = pickle.load(open(root + 'public_suc01_T_datasets.pkl', 'rb'))[0]
d, gt = b['dataset'], b['gt']
a, test, dt, t = d['train'], d['test'], d['past_dt'], d['eval_times']
tau, amp, drift = m._fit_physics(a, dt, t)
curves, rate = m._branch(tau, a, dt, t)
tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
z = a['initial_memory_states'] * np.exp(-np.sum(dt) / tau) + a['past_stress'] @ (-np.expm1(-dt[:, None] / tau) * np.exp(-tail[:, None] / tau))
gap = a['stress_hold_0'][:, None] - z
np.testing.assert_allclose(rate, gap / tau, atol=1e-12)
np.testing.assert_allclose(curves, gap[:, None, :] * (1 - np.exp(-t[:, None] / tau)), atol=1e-12)
eps = 1e-6
short, _ = m._branch(tau, a, dt, np.array([eps]))
np.testing.assert_allclose(short[:, 0] @ amp / eps + drift, rate @ amp + drift, rtol=1e-5, atol=1e-5)
print('Train worst-horizon median absolute curve residual, measured versus zero preload (same fitted parameters):')
obs = a['strain_curves_obs'] - a['strain_0'][:, None]
for c in (curves, m._branch(tau, {**a, 'initial_memory_states': np.zeros_like(a['initial_memory_states'])}, dt, t)[0]):
    print(np.median(np.abs(obs - c @ amp - drift[:, None]*t), axis=0).max())
x, known, weights = m._coupons(a, test)
n = len(drift)
x, known = x[:n], known[:n].all(axis=1)
lab, stratum = a['lab_id'], a['stratum']
base = np.column_stack([np.ones(n), x])
quadratic = np.column_stack([x[:, i]*x[:, j] for i in range(4) for j in range(i, 4)])
rat = np.column_stack([x[:, i]/(1 + x[:, j]**2) for i in range(4) for j in range(4)])
# Holding out one cell per lab permits calibration on other strata in that lab.
lab_design = (base[:, :, None] * np.eye(4)[lab, None, :]).reshape(n, -1)
for label, F in [('shared affine',base), ('lab calibrated affine',lab_design), ('lab calibrated + quadratic/rational', np.column_stack([lab_design,quadratic,rat]))]:
    scores = []
    for l in np.unique(lab):
        for s in np.unique(stratum):
            held = known & (lab==l) & (stratum==s)
            keep = known & ~held
            sw = np.sqrt(weights[keep]); coef = np.linalg.lstsq(F[keep]*sw[:, None], drift[keep]*sw, rcond=None)[0]
            scores.append(np.mean(np.abs(drift[held] - F[held]@coef)))
    print(label, 'mean/worst held-cell drift MAE', np.mean(scores), np.max(scores))
o = m.solve_preloaded_hereditary_transfer(d)
mae = np.column_stack([np.abs(o['predicted_strain'] - gt['te_clean']).mean(1), np.abs(o['secant_compliance'] - gt['te_secant']), np.abs(o['hazard_crossing_times'] - gt['te_cross'])])
preload = np.linalg.norm(test['initial_memory_states'] * np.exp(-sum(dt)/o['estimated_taus']), axis=1)
for q in np.array_split(np.argsort(preload), 4):
    print('preload quartile n/max memory / strain secant crossing MAE', len(q), preload[q].max(), mae[q].mean(0))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_preloaded.py && OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`1706 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Train worst-horizon median absolute curve residual, measured versus zero preload (same fitted parameters):
0.006605647668208037
0.5075733194526358
shared affine mean/worst held-cell drift MAE 0.00527629954056813 0.012756972537903222
lab calibrated affine mean/worst held-cell drift MAE 0.0007115878331584378 0.0010013674637718078
lab calibrated + quadratic/rational mean/worst held-cell drift MAE 0.0009967978958251857 0.0017971963715911002
preload quartile n/max memory / strain secant crossing MAE 23 0.43431339252089024 [0.00294753 0.00538364 0.00114991]
preload quartile n/max memory / strain secant crossing MAE 23 1.0951729025332428 [0.00381589 0.00415309 0.000615  ]
preload quartile n/max memory / strain secant crossing MAE 22 1.6707174929115802 [0.00537302 0.00955699 0.04465348]
preload quartile n/max memory / strain secant crossing MAE 22 4.195482851568971 [0.00293096 0.00607192 0.00377424]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=698f7a25d90577b3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_698f7a25 (directory).
```

### Turn 9 — `submit_completed_task` (`16808 ms` LLM)
- **Rationale**: The final artifact passed isolated schema validation, direct first-passage tests, and all leave-one-lab-out thresholds. The closed-form audit confirmed the nonuniform memory recurrence and initial derivative, and every preload quartile remained within tolerance.
- **Alternatives Weighed**: `Add quadratic and rational coupon terms` (rejected: They increased worst held-cell drift MAE from 0.00100 to 0.00180.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/preloaded_hereditary_secant_hazard_solver.py': 'FILE:5697'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/preloaded_hereditary_secant_hazard_solver.py`. On the public holdout, the maximum relative error in relaxation times was 0.49%; strain, secant compliance, and hazard crossing MAEs were 0.00376, 0.00626, and 0.01229. All four metrics also passed in every leave-one-lab-out fold.`

### Turn 10 — `execute_in_remote_docker` (`41304 ms` LLM)
- **Rationale**: Compare the fused estimator with its rate-only and coupon-only endpoints on real held-out cells, using late strain curves and excluding the initial-rate measurement from the reference. If fusion loses systematically to either endpoint in the affected slice, revise the weighting without tuning to the public test labels.
- **Alternatives Weighed**: `Repeat the recurrence and nonlinear-basis checks` (rejected: The independent closed-form recurrence agreed, and nonlinear terms worsened held-cell error. Neither check isolates the remaining missing-coupon bottleneck.)

```bash
cat > /tmp/audit_coupon_missingness.py <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'preloaded_hereditary_secant_hazard_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root + 'public_suc01_T_datasets.pkl', 'rb'))[0]['dataset']
a, t, dt = d['train'], d['eval_times'], d['past_dt']
labs, strata = a['lab_id'], a['stratum']
missing = ~np.isfinite(a['coupon_features_raw']).all(axis=1)
errors = []
for lab, stratum in np.unique(np.column_stack((labs, strata)), axis=0):
    hold = (labs == lab) & (strata == stratum)
    fit = {k: v[~hold] for k, v in a.items()}
    test = {k: v[hold] for k, v in a.items()}
    tau, amp, drift = m._fit_physics(fit, dt, t)
    _, fit_rate = m._branch(tau, fit, dt, t)
    branch, test_rate = m._branch(tau, test, dt, t)
    train_rate = fit['instant_rate_obs'] - fit_rate @ amp
    rate = test['instant_rate_obs'] - test_rate @ amp
    fused = m._predict_drift(fit, test, drift, train_rate, rate)
    # The prediction is affine in the measured rate; recover the coupon-only endpoint.
    p0 = m._predict_drift(fit, test, drift, train_rate, np.zeros_like(rate))
    p1 = m._predict_drift(fit, test, drift, train_rate, np.ones_like(rate))
    prior = p0 / (1. - (p1 - p0))
    # An independent held-out drift proxy uses only curves, not the initial-rate observation.
    observed = test['strain_curves_obs'] - test['strain_0'][:, None] - branch @ amp
    reference = np.median(observed[:, -5:] / t[-5:], axis=1)
    errors.append(np.column_stack((np.full(sum(hold), lab), np.full(sum(hold), stratum), missing[hold], np.abs(reference[:, None] - np.column_stack((fused, rate, prior))))))
e = np.vstack(errors)
print('Held-cell drift MAE: fused / rate-only / coupon-only')
for flag in (0, 1):
    subset = e[e[:, 2] == flag]
    print('missing', flag, 'n', len(subset), 'MAE', subset[:, 3:].mean(0))
    for lab in np.unique(subset[:, 0]):
        rows = subset[subset[:, 0] == lab, 3:]
        print('  lab', int(lab), 'n', len(rows), 'MAE', rows.mean(0))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_coupon_missingness.py
```
- **Container Exit Code**: `0` (`3692 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Held-cell drift MAE: fused / rate-only / coupon-only
missing 0 n 135 MAE [0.00199763 0.00835597 0.00198532]
  lab 0 n 32 MAE [0.00105947 0.00813743 0.00101846]
  lab 1 n 36 MAE [0.00084838 0.0080312  0.00082106]
  lab 2 n 33 MAE [0.00523219 0.01095842 0.00523054]
  lab 3 n 34 MAE [0.00095802 0.00637963 0.00097829]
missing 1 n 13 MAE [0.00705803 0.00764707 0.10466319]
  lab 0 n 4 MAE [0.00462647 0.0056497  0.11577877]
  lab 1 n 2 MAE [0.00674453 0.00805417 0.12200539]
  lab 2 n 5 MAE [0.00852977 0.0088264  0.09136361]
  lab 3 n 2 MAE [0.00855531 0.00828639 0.09833875]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=698f7a25d90577b3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_698f7a25 (directory).
```

### Turn 11 — `submit_completed_task` (`40885 ms` LLM)
- **Rationale**: The saved artifact passed isolated schema validation, first-passage semantics tests, and all leave-one-lab-out thresholds. The final incomplete-coupon audit supported retaining rate fusion without changing the verified artifact.
- **Alternatives Weighed**: `Use the initial-rate measurement alone when coupons are incomplete` (rejected: Its held-cell drift MAE was higher than the fused estimate across the incomplete-coupon slice.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/preloaded_hereditary_secant_hazard_solver.py`. On the public holdout, the maximum relaxation-time relative error was 0.49%. The MAEs for strain, secant compliance, and hazard crossing time were 0.00376, 0.00626, and 0.01229. All four metrics passed in every leave-one-lab-out fold.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M93-Lifecycle",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/preloaded_hereditary_secant_hazard_solver.py": "FILE:5697"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "max_tau_rel_err": 0.007751,
    "max_strain_mae": 0.005108,
    "max_secant_mae": 0.010508,
    "max_crossing_mae": 0.001755,
    "student_elapsed_sec": 0.578,
    "instance_diagnostics": [
      {
        "seed": 99511,
        "tau_rel": 0.007751,
        "strain_mae": 0.005108,
        "secant_mae": 0.010508,
        "cross_mae": 0.001755
      },
      {
        "seed": 99512,
        "tau_rel": 0.00445,
        "strain_mae": 0.004196,
        "secant_mae": 0.008583,
        "cross_mae": 0.00157
      }
    ],
    "candidate_timing_breakdown": {
      "import_wall_sec": 0.083222,
      "import_cpu_sec": 0.076446,
      "candidate_call_wall_sec": 0.529835,
      "candidate_call_cpu_sec": 0.486753,
      "ipc_and_codec_wall_sec": 0.121056,
      "rpc_call_count": 2,
      "verifier_and_gt_wall_sec": 0.007874,
      "verifier_and_gt_cpu_sec": 0.031301,
      "total_supervisor_wall_sec": 0.741987
    }
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"max_tau_rel_err\": 0.007751, \"max_strain_mae\": 0.005108, \"max_secant_mae\": 0.010508, \"max_crossing_mae\": 0.001755, \"student_elapsed_sec\": 0.578, \"instance_diagnostics\": [{\"seed\": 99511, \"tau_rel\": 0.007751, \"strain_mae\": 0.005108, \"secant_mae\": 0.010508, \"cross_mae\": 0.001755}, {\"seed\": 99512, \"tau_rel\": 0.00445, \"strain_mae\": 0.004196, \"secant_mae\": 0.008583, \"cross_mae\": 0.00157}]}\n__HEGEL_VERIFIER_AUTH_56ffea8db24951314f3966707cd2c877__:{\"strict_pass\": true, \"reward\": 1.0, \"max_tau_rel_err\": 0.007751, \"max_strain_mae\": 0.005108, \"max_secant_mae\": 0.010508, \"max_crossing_mae\": 0.001755, \"student_elapsed_sec\": 0.578, \"instance_diagnostics\": [{\"seed\": 99511, \"tau_rel\": 0.007751, \"strain_mae\": 0.005108, \"secant_mae\": 0.010508, \"cross_mae\": 0.001755}, {\"seed\": 99512, \"tau_rel\": 0.00445, \"strain_mae\": 0.004196, \"secant_mae\": 0.008583, \"cross_mae\": 0.00157}], \"candidate_timing_breakdown\": {\"import_wall_sec\": 0.083222, \"import_cpu_sec\": 0.076446, \"candidate_call_wall_sec\": 0.529835, \"candidate_call_cpu_sec\": 0.486753, \"ipc_and_codec_wall_sec\": 0.121056, \"rpc_call_count\": 2, \"verifier_and_gt_wall_sec\": 0.007874, \"verifier_and_gt_cpu_sec\": 0.031301, \"total_supervisor_wall_sec\": 0.741987}}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5576.22,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
