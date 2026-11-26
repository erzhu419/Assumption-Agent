# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `suc01_C__viscoplastic_bingham_overstress_counterexample_revision`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M93-Lifecycle` |
| **Domain** | `Localized Premise Counterexample: Bingham-Perzyna Viscoplastic Overstress & Plastic-Work Softening Above Yield` |
| **Wave / Wormhole Family** | `m93_stage_C_counterexample` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`PASS (1.0)`** |
| **Failure / Outcome Taxonomy** | `PASSED_ALL_CHECKS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `387,228` (`371,769` / `116,224` / `15,459`) |
| **Total Episode Wall Time (s)** | `556.4s` |

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

### Turn 1 — `execute_in_remote_docker` (`19952 ms` LLM)
- **Rationale**: Inspect the schema and prior files to identify the coupon alignment and Prony conventions before fitting; use those conventions to compare linear and yield-gated models on held-out loading cohorts.
- **Alternatives Weighed**: `Fit strain curves directly with a generic regressor` (rejected: It would not preserve the specified sub-yield recurrence or identify the two physical weights.)

```bash
cd /holdout_task && cat validate_public_schema.py && python3 validate_public_schema.py; printf '\n--- prior task files ---\n'; ls -la /pretrain_task; find /pretrain_task -maxdepth 2 -type f -printf '%p %s bytes\n' | head -40; printf '\n--- skill lookup ---\n'; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation Prony variable step robust Cauchy IRLS reference panel alignment imputation' || true; printf '\n--- public data structure ---\n'; python3 - <<'PY'
import pickle, numpy as np
p=pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl','rb'))
def view(x,k='',d=0):
 if isinstance(x,dict):
  print(k,'dict',list(x)[:30]);
  if d<4:
   for a,b in list(x.items())[:18]: view(b,str(a),d+1)
 elif isinstance(x,(tuple,list)):
  print(k,type(x).__name__,len(x));
  if d<3:
   for j,a in enumerate(x[:2]): view(a,str(j),d+1)
 elif isinstance(x,np.ndarray): print(k,x.shape,x.dtype, x.ravel()[:min(4,x.size)])
 else: print(k,str(x)[:200])
view(p)
PY
```
- **Container Exit Code**: `0` (`1015 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Q: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)

--- public data structure ---
 list 1
0 dict ['dataset', 'gt']
dataset dict ['past_dt', 'eval_times', 'train', 'test', 'yield_stress']
past_dt (28,) float64 [0.2379 0.2773 0.2044 0.3101]
eval_times (10,) float64 [0.2  0.45 0.8  1.25]
train dict ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress', 'strain_0', 'stress_hold_0', 'instant_rate_obs', 'strain_curves_obs']
lab_id (148,) int64 [2 2 1 3]
stratum (148,) int64 [0 2 1 0]
coupon_features_raw (148, 4) float64 [-0.01002824  5.33934753  3.48412251  4.28436465]
ref_coupons (148, 20) float64 [4.87372906 3.08029807 6.82929884 3.28782367]
past_stress (148, 28) float64 [-1.426      -1.426      -1.426      -1.17049905]
strain_0 (148,) float64 [-0.53882384  0.21913343  0.49137244 -0.91798528]
stress_hold_0 (148,) float64 [-1.426       2.68460746  1.426      -2.4983265 ]
instant_rate_obs (148,) float64 [-0.84442339  6.95311845  0.52196025 -1.37230854]
strain_curves_obs (148, 10) float64 [-0.69433845 -0.88219714 -1.11568464 -1.43312041]
test dict ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress', 'strain_0', 'stress_hold_0', 'instant_rate_obs']
lab_id (90,) int64 [2 3 1 1]
stratum (90,) int64 [0 0 1 2]
coupon_features_raw (90, 4) float64 [2.13680554 4.83227042 2.80032932 3.40141876]
ref_coupons (90, 20) float64 [0.70260801 5.84584835 4.63223518 4.61166442]
past_stress (90, 28) float64 [1.426      1.426      1.426      1.21028511]
strain_0 (90,) float64 [-0.16391183  0.97618752  0.34255033  0.9853829 ]
stress_hold_0 (90,) float64 [-1.426       2.5857758   1.0508914   2.67838064]
instant_rate_obs (90,) float64 [-3.11361369  1.31699513  0.52964761  2.54490308]
yield_stress 1.55
gt dict ['taus', 'theta', 'vp_theta', 'te_Z', 'te_clean', 'te_true_rate0', 'te_cross', 'te_secant']
taus (2,) float64 [0.58 3.9 ]
theta (7,) float64 [0.68 0.54 0.08 0.15]
vp_theta (2,) float64 [0.24 0.16]
te_Z (90, 4) float64 [-1.03772657  0.87395065 -0.56714233 -0.14083776]
te_clean (90, 10) float64 [-0.70352378 -1.19422652 -1.66087148 -2.04766055]
te_true_rate0 (90,) float64 [-3.11614886  1.32986451  0.51334633  2.5517431 ]
te_cross (90,) float64 [8.5        0.39342466 0.53255253 0.22262001]
te_secant (90,) float64 [-2.96721262  0.44112716  3.0069437   1.97937193]
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/holdout_task/validate_public_schema.py", line 6, in <module>
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 991, in exec_module
  File "<frozen importlib._bootstrap_external>", line 1128, in get_code
  File "<frozen importlib._bootstrap_external>", line 1186, in get_data
FileNotFoundError: [Errno 2] No such file or directory: '/holdout_task/viscoplastic_overstress_hereditary_solver.py'
```

### Turn 2 — `execute_in_remote_docker` (`29224 ms` LLM)
- **Rationale**: Identify the reference-panel layout and any reusable recurrence before fitting. Count history-only, hold-only, and below-yield cases so validation can distinguish failures of the two mechanisms.
- **Alternatives Weighed**: `Standardize coupon columns independently within each lab` (rejected: This could remove genuine composition differences between labs; inspect the reference controls before choosing the calibration.)

```bash
export PYTHONPATH=/tmp; for f in autogenous_prim_reference_panel_balancing.align_reference_panel autogenous_prim_reference_panel_balancing.balanced_cell_weights autogenous_prim_reference_panel_balancing.weighted_column_fill autogenous_prim_robust_affine_projection.robust_residual_weights; do python3 -m hegel_wormhole_skills --describe "$f"; done; python3 -m hegel_wormhole_skills --search 'Prony hereditary exponential state variable step'; python3 - <<'PY'
import pickle,numpy as np
D=pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl','rb'))[0]['dataset']; print('times',D['eval_times'],'dt',D['past_dt']);
for k in ('train','test'):
 d=D[k]; y=D['yield_stress']; h=d['past_stress']; s=d['stress_hold_0']; a=d['coupon_features_raw']; r=d['ref_coupons']; g=d['lab_id']; print(k,'yield counts (past, hold, both, neither):',sum(np.max(abs(h),1)>y),sum(abs(s)>y),sum((np.max(abs(h),1)>y)&(abs(s)>y)),sum((np.max(abs(h),1)<=y)&(abs(s)<=y))); print('missing coupon/ref',np.isnan(a).sum(0),np.isnan(r).sum(0));
 for j in np.unique(g):
  print('lab',j,'strata',np.bincount(d['stratum'][g==j]),'coupon mean, std',np.nanmean(a[g==j],0).round(3),np.nanstd(a[g==j],0).round(3),'ref mean, std',np.nanmean(r[g==j]).round(3),np.nanstd(r[g==j]).round(3),'ref first row',r[g==j][0].round(3));
PY
```
- **Container Exit Code**: `0` (`777 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.2958 0.2504
 0.357  0.3062 0.3099 0.1891 0.2075 0.3623 0.1611 0.3081]
train yield counts (past, hold, both, neither): 74 74 74 74
missing coupon/ref [4 4 4 3] [ 5  7  8  7  5  4  9  4  0  9  5  4  9  7  3  3  3  7  3 10]
lab 0 strata [20  8  8] coupon mean, std [2.757 3.327 3.13  3.419] [1.21  1.414 1.1   1.143] ref mean, std 3.139 1.393 ref first row [2.329 4.029 3.757 1.801 1.171 3.564 3.834   nan 4.323 1.124 2.198 4.575
 4.679 4.437 2.801 3.007 2.995 2.027 2.499 3.02 ]
lab 1 strata [ 8 22  8] coupon mean, std [-3.468 -3.661 -3.358 -3.374] [0.579 0.539 0.617 0.736] ref mean, std -3.397 0.771 ref first row [-3.325 -3.948 -4.563 -2.733 -3.752 -1.882 -1.754 -3.245 -4.952 -3.546
 -3.084 -5.784 -3.528 -3.566 -3.462 -4.19  -1.417 -2.89  -4.89  -3.256]
lab 2 strata [ 8  8 22] coupon mean, std [4.078 3.111 4.135 3.421] [1.478 1.09  1.085 1.006] ref mean, std 3.632 1.414 ref first row [4.874 3.08  6.829 3.288 3.399 4.602 3.696 4.215 8.24    nan 4.986 3.012
 2.888 3.648 6.061 3.927 3.521 5.465 2.859 4.69 ]
lab 3 strata [18  9  9] coupon mean, std [-2.852 -2.72  -2.936 -2.794] [0.876 0.753 0.615 0.72 ] ref mean, std -2.898 0.856 ref first row [   nan -1.297    nan -3.912 -3.575 -2.812 -3.837 -2.61  -2.11  -2.819
 -1.875 -3.312 -4.74  -4.163 -2.187 -3.02  -2.476 -2.215 -2.094 -1.336]
test yield counts (past, hold, both, neither): 37 37 37 53
missing coupon/ref [2 2 3 2] [7 7 4 3 3 4 2 2 5 5 3 2 3 4 4 1 3 3 7 0]
lab 0 strata [12  5  5] coupon mean, std [3.126 3.146 3.267 2.999] [1.637 0.944 1.053 1.009] ref mean, std 3.16 1.405 ref first row [1.605 0.83  2.095 2.331 1.476 2.719 2.537 3.302 2.148 3.519 1.534 4.115
 2.395 1.785 3.144 2.615 5.229 1.81  2.571 3.603]
lab 1 strata [ 5 13  5] coupon mean, std [-3.244 -3.644 -3.118 -3.288] [0.738 0.61  0.572 0.769] ref mean, std -3.414 0.792 ref first row [-2.553 -3.558 -2.833 -4.868 -3.824 -2.942 -3.072 -4.04  -3.226 -4.034
 -3.527 -2.489 -3.651 -3.329 -3.057 -2.869 -4.082 -4.791 -4.275 -2.817]
lab 2 strata [ 5  5 13] coupon mean, std [4.671 2.699 4.315 2.972] [1.257 1.01  0.921 1.192] ref mean, std 3.651 1.325 ref first row [0.703 5.846 4.632 4.612 4.123 3.776 4.052 2.937 5.004   nan 1.136 5.738
 2.072 2.545 4.141 5.122 3.587 3.34  4.512 4.859]
lab 3 strata [11  6  5] coupon mean, std [-3.066 -2.49  -3.009 -2.953] [0.879 0.821 0.828 0.647] ref mean, std -2.903 0.872 ref first row [-3.074    nan -1.044 -4.204 -2.6   -2.076 -3.23  -2.021 -3.94  -2.051
 -2.508 -2.467 -2.19     nan -2.894 -1.271 -2.723 -2.889    nan -4.281]
```

### Turn 3 — `execute_in_remote_docker` (`45812 ms` LLM)
- **Rationale**: Use the public truth only as a diagnostic witness to identify the missing baseline terms and check the onset convention. The submitted solver will estimate its coefficients and relaxation times from the dataset input alone.
- **Alternatives Weighed**: `Start with an unconstrained coupon-by-branch regression` (rejected: First isolate the clean temporal components to see whether the seven-coefficient law has a simpler exact form.)

```bash
python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl','rb'))[0]; d=p['dataset']; g=p['gt']; q=d['test']; t=d['eval_times']; dt=d['past_dt']; a,b=g['taus']; y=d['yield_stress']; s=q['stress_hold_0']; h=q['past_stress']; X=g['te_Z']; print('public constitutive coefficient ordering witness',g['theta']); print('feature covariance',np.round(np.cov(X.T),3)); ov=np.maximum(0,abs(s)-y)**2*np.sign(s); w=(np.maximum(0,abs(h)-y)**2@dt)*s; ex=lambda tau:-np.expm1(-t/tau); L=np.column_stack((t,ex(a),ex(b))); Q=(g['te_clean']-q['strain_0'][:,None])-g['vp_theta'][0]*ov[:,None]*t-g['vp_theta'][1]*w[:,None]*ex(a); C=np.linalg.lstsq(L,Q.T,rcond=None)[0].T; print('three component exactness',np.max(abs(Q-C@L.T)),'rate exactness',np.max(abs(g['te_true_rate0']-C@np.array([1,1/a,1/b])-g['vp_theta'][0]*ov-g['vp_theta'][1]*w/a))); Z=[]; right=np.cumsum(dt[::-1])[::-1]-dt
for tau in (a,b): Z.append(h@(np.exp(-right/tau)*-np.expm1(-dt/tau)))
F=np.column_stack([np.ones_like(s),X,s,s[:,None]*X,Z[0],Z[1],q['strain_0'],s-Z[0],s-Z[1]]); names=['1']+['X'+str(i) for i in range(4)]+['s']+['sX'+str(i) for i in range(4)]+['z1','z2','strain0','s-z1','s-z2']; print('linear projections:')
for i,c in enumerate(C.T):
 for j,M in [('direct',F),('times hold',s[:,None]*F),('times branch gap 1',(s-Z[0])[:,None]*F),('times branch gap 2',(s-Z[1])[:,None]*F)]:
  co=np.linalg.lstsq(M,c,rcond=None)[0]; r=c-M@co; print(i,j,round(np.mean(abs(r)),8),round(np.max(abs(r)),8),[(n,round(v,5)) for n,v in zip(names,co) if abs(v)>0.0001]);
print('reference alignment comparison: normalized features versus public diagnostic only'); raw=q['coupon_features_raw']; ids=q['lab_id']; tr=d['train']; R=np.row_stack([tr['ref_coupons'],q['ref_coupons']]); G=np.r_[tr['lab_id'],ids];
for i in np.unique(ids):
 m=G==i; j=ids==i; r=R[m]; v=raw[j]; z=X[j]; print(i,[(round(np.polyfit(z[np.isfinite(v[:,k]),k],v[np.isfinite(v[:,k]),k],1)[0],4),round(np.polyfit(z[np.isfinite(v[:,k]),k],v[np.isfinite(v[:,k]),k],1)[1],4)) for k in range(4)], 'control pooled',np.nanmean(r),np.nanstd(r),'median',np.median(np.nanmean(r,0)),np.median(np.nanstd(r,0)))
PY
```
- **Container Exit Code**: `0` (`584 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
754478 [('1', 0.01572), ('X0', 0.01635), ('X1', 0.00448), ('X2', 0.02431), ('X3', 0.02103), ('s', 0.10557), ('sX0', 0.03654), ('sX1', -0.01905), ('sX2', 0.01542), ('sX3', -0.03505), ('z1', 0.08307), ('z2', 0.10145), ('strain0', -0.4644), ('s-z1', 0.0225), ('s-z2', 0.00412)]
1 direct 0.0 0.0 [('s', 0.17), ('z1', -0.255), ('z2', 0.085), ('s-z1', 0.425), ('s-z2', 0.085)]
1 times hold 0.6651194 2.31488011 [('1', 0.68569), ('X0', -0.03057), ('X1', 0.19853), ('X2', 0.08051), ('X3', -0.02054), ('s', 0.04942), ('sX0', 0.06096), ('sX1', 0.06488), ('sX2', -0.05256), ('sX3', -0.0387), ('z1', -0.0294), ('z2', 0.15498), ('strain0', -0.18688), ('s-z1', 0.07883), ('s-z2', -0.10556)]
1 times branch gap 1 0.0 0.0 [('1', 0.68)]
1 times branch gap 2 0.43880423 1.44170813 [('1', 0.77379), ('X0', -0.03236), ('X1', 0.13895), ('X2', 0.06775), ('X3', -0.0104), ('s', 0.23501), ('sX0', 0.05506), ('sX1', 0.03617), ('sX2', -0.02931), ('sX3', -0.01488), ('z1', 0.0826), ('z2', 0.42266), ('strain0', -1.08195), ('s-z1', 0.15241), ('s-z2', -0.18766)]
2 direct 0.0 0.0 [('s', 0.135), ('z1', 0.0675), ('z2', -0.2025), ('s-z1', 0.0675), ('s-z2', 0.3375)]
2 times hold 0.19602716 0.73039764 [('1', 0.53812), ('X0', 0.00702), ('X1', 0.03973), ('X2', 0.00593), ('X3', 0.00258), ('s', -0.11434), ('sX0', 0.01177), ('sX1', 0.0278), ('sX2', -0.00597), ('sX3', -0.01309), ('z1', -0.05411), ('z2', -0.19067), ('strain0', 0.43473), ('s-z1', -0.06023), ('s-z2', 0.07633)]
2 times branch gap 1 0.31529407 1.06578682 [('1', 0.40526), ('X0', 0.00119), ('X1', -0.03672), ('X2', -0.00426), ('X3', 0.01276), ('s', -0.08837), ('sX0', -0.00985), ('sX1', -0.01227), ('sX2', -0.00414), ('sX3', 0.0033), ('z1', 0.00158), ('z2', -0.2092), ('strain0', 0.67066), ('s-z1', -0.08995), ('s-z2', 0.12083)]
2 times branch gap 2 0.0 0.0 [('1', 0.54)]
reference alignment comparison: normalized features versus public diagnostic only
0 [(1.32, 3.1), (1.32, 3.1), (1.32, 3.1), (1.32, 3.1)] control pooled 3.147287620020758 1.3975718912143238 median 3.159827948112895 1.353295398245372
1 [(0.79, -3.4), (0.79, -3.4), (0.79, -3.4), (0.79, -3.4)] control pooled -3.403352663530347 0.7791726692834493 median -3.417450272497813 0.7835809763525032
2 [(1.41, 3.6), (1.41, 3.6), (1.41, 3.6), (1.41, 3.6)] control pooled 3.639255799701584 1.3813156096219013 median 3.6310831255803544 1.367297572559076
3 [(0.85, -2.9), (0.85, -2.9), (0.85, -2.9), (0.85, -2.9)] control pooled -2.8997025452278944 0.8622479923070544 median -2.8546627940462335 0.8197763756749052
```

### Turn 4 — `execute_in_remote_docker` (`63617 ms` LLM)
- **Rationale**: Implement the recovered basis and fit with a Cauchy loss on the training observations only. Check the isolated import, parameter recovery, and yield-cohort errors to determine whether onset noise or constitutive estimation limits accuracy.
- **Alternatives Weighed**: `Fit absolute curve residuals alone` (rejected: Onset differencing removes coupon drift from the curvature fit, so imputation errors cannot masquerade as relaxation.)

```bash
cat > /holdout_task/viscoplastic_overstress_hereditary_solver.py <<'PY'
"""Two-branch hereditary strain with yield-gated creep and history work."""

import numpy as np
from scipy.optimize import least_squares


def _align(train, test):
    """Estimate lab location and gain from reference coupons, not study coupons."""
    n = len(train['lab_id'])
    labs = np.r_[train['lab_id'], test['lab_id']]
    refs = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    for lab in np.unique(labs):
        mask = labs == lab
        panel = refs[mask]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0))
        x[mask] = (x[mask] - offset) / gain
    cells = np.column_stack((train['lab_id'], train['stratum']))
    _, cell, count = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    weights = 1.0 / count[cell]
    observed = np.isfinite(x[:n])
    weight_sum = weights @ observed
    fill = np.divide(weights @ np.where(observed, x[:n], 0.0), weight_sum,
                     out=np.zeros(x.shape[1]), where=weight_sum > 0)
    return tuple(np.where(np.isfinite(x), x, fill)[sl] for sl in (slice(None, n), slice(n, None)))


class _Design:
    def __init__(self, split, x, dt, time, yield_stress):
        self.time = time
        self.stress = split['stress_hold_0']
        self.history = split['past_stress']
        self.dt = dt
        self.remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
        s = self.stress
        self.work = (np.maximum(np.abs(self.history) - yield_stress, 0.0)**2 @ dt) * s
        overstress = np.sign(s) * np.maximum(np.abs(s) - yield_stress, 0.0)**2
        self.drift = np.column_stack((np.ones(len(s)), x, overstress))

    def matrices(self, taus):
        # Exact zero-initialized Prony state under piecewise constant history.
        kernels = np.exp(-self.remaining[:, None] / taus) * -np.expm1(-self.dt[:, None] / taus)
        gaps = self.stress[:, None] - self.history @ kernels
        fast = np.column_stack((gaps[:, 0], self.work))
        slow = gaps[:, 1:2]
        rate = np.column_stack((self.drift, fast / taus[0], slow / taus[1]))
        f = -np.expm1(-self.time[:, None] / taus)
        curves = np.concatenate((self.drift[:, None, :] * self.time[None, :, None],
                                 fast[:, None, :] * f[None, :, :1],
                                 slow[:, None, :] * f[None, :, 1:]), axis=2)
        return curves, rate


def _fit(design, train):
    target = train['strain_curves_obs'] - train['strain_0'][:, None]
    onset = train['instant_rate_obs']
    time = design.time
    # Differences against the onset extrapolation isolate curvature and cancel
    # the coupon drift exactly, even when a coupon entry had to be imputed.
    scale = 0.035 + 0.016 * time
    def system(taus):
        curves, rate = design.matrices(taus)
        a = ((curves - time[None, :, None] * rate[:, None, :]) / scale[None, :, None]).reshape(-1, 9)
        return np.vstack((a, rate / 0.04))

    b = np.r_[((target - onset[:, None] * time) / scale).ravel(), onset / 0.04]
    candidates = []
    for a in (0.28, 0.58, 1.1, 2.1):
        for c in (0.95, 2.0, 4.0, 7.0):
            if c <= a:
                continue
            m = system(np.array([a, c]))
            coeff = np.linalg.lstsq(m, b, rcond=None)[0]
            r = m @ coeff - b
            candidates.append((np.log1p(r*r).sum(), np.r_[a, (c-a)/(8-a), coeff]))

    def unpack(p):
        return np.array((p[0], p[0] + (8.0 - p[0]) * p[1]))

    def residual(p):
        return system(unpack(p)) @ p[2:] - b

    bounds = (np.r_[0.20, 1e-6, np.full(9, -np.inf)],
              np.r_[8.0-1e-6, 1.0, np.full(9, np.inf)])
    results = []
    for _, start in sorted(candidates, key=lambda p: p[0])[:2]:
        warm = least_squares(residual, start, bounds=bounds, loss='soft_l1', f_scale=2.0, x_scale='jac', max_nfev=80)
        fit = least_squares(residual, warm.x, bounds=bounds, loss='cauchy', f_scale=2.0, x_scale='jac', max_nfev=100)
        results.append(fit)
    fit = min(results, key=lambda r: r.cost)
    return unpack(fit.x), fit.x[2:]


def solve_viscoplastic_hereditary_revision(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_train, x_test = _align(train, test)
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    y = float(dataset['yield_stress'])
    taus, coef = _fit(_Design(train, x_train, dt, time, y), train)
    design = _Design(test, x_test, dt, time, y)
    curves, rate = design.matrices(taus)
    # The measured onset constrains the individual coupon's steady drift.
    correction = test['instant_rate_obs'] - rate @ coef
    prediction = test['strain_0'][:, None] + curves @ coef + correction[:, None] * time
    return {'estimated_taus': taus, 'viscoplastic_weights': coef[[5, 7]],
            'predicted_strain': prediction}
PY
cat > /tmp/validate_vp.py <<'PY'
import importlib.util
import pickle
import time
import numpy as np
p='/holdout_task/viscoplastic_overstress_hereditary_solver.py'
s=importlib.util.spec_from_file_location('candidate',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
for i,row in enumerate(pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl','rb'))):
    d,g=row['dataset'],row['gt']; start=time.monotonic(); out=m.solve_viscoplastic_hereditary_revision(d); print('runtime',time.monotonic()-start)
    print('taus',out['estimated_taus'],'rel',abs(out['estimated_taus']/g['taus']-1), 'vp',out['viscoplastic_weights'],'rel',np.linalg.norm(out['viscoplastic_weights']-g['vp_theta'])/np.linalg.norm(g['vp_theta']))
    q=d['test']; h=np.max(abs(q['past_stress']),axis=1)>d['yield_stress']; s=abs(q['stress_hold_0'])>d['yield_stress']; err=abs(out['predicted_strain']-g['te_clean']); missing=np.any(~np.isfinite(q['coupon_features_raw']),axis=1)
    for label,mask in [('all',h|~h),('over',h|s),('sub',~(h|s)),('over+', (h|s)&(q['stress_hold_0']>0)),('over-',(h|s)&(q['stress_hold_0']<0)),('missing coupon',missing)]+[(f'lab {j}',q['lab_id']==j) for j in np.unique(q['lab_id'])]:
        if mask.any(): print(label,mask.sum(),float(err[mask].mean()),'last',float(err[mask,-1].mean()))
PY
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; python3 -I /holdout_task/validate_public_schema.py && python3 -I /tmp/validate_vp.py
```
- **Container Exit Code**: `0` (`977 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoplastic_overstress_hereditary_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
runtime 0.0708593490999192
taus [0.57729618 3.57179933] rel [0.00466177 0.08415402] vp [0.24539855 0.15896813] rel 0.01905492675713419
all 90 0.023864699899010865 last 0.06686043142841269
over 37 0.026936269059470313 last 0.07560556043719488
sub 53 0.021720396900199553 last 0.06075534136567795
over+ 24 0.024828169361885016 last 0.07057141764506918
over- 13 0.030828145424243166 last 0.08489936251496541
missing coupon 9 0.03166343503660118 last 0.08736303271400464
lab 0 22 0.02692869471126072 last 0.07515278252704327
lab 1 23 0.02246393084182597 last 0.06335242656706923
lab 2 23 0.020073756760601226 last 0.05497424022578155
lab 3 22 0.02622840420033711 last 0.07466201257848283

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d6b441a10c2c1403 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d6b441a1 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`49738 ms` LLM)
- **Rationale**: Compare the unchanged baseline against a joint fit that treats onset noise once per coupon, and against a linear-only ablation. Promote a challenger only if it improves slow-time recovery without a material regression across held-out yield cohorts.
- **Alternatives Weighed**: `Submit the first passing solver` (rejected: The slow relaxation time is only 1.58 percentage points inside tolerance; a joint fit can test whether the current treatment of shared onset noise is limiting that estimate.)

```bash
cat > /tmp/compare_vp.py <<'PY'
import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, hstack
p='/holdout_task/viscoplastic_overstress_hereditary_solver.py'
s=importlib.util.spec_from_file_location('baseline',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)

def joint(design, train, curve_sigma=0.025, onset_sigma=0.016, null=False):
    taus, coeff = m._fit(design, train)
    t = design.time
    y = train['strain_curves_obs'] - train['strain_0'][:,None]
    onset = train['instant_rate_obs']
    n,k = y.shape
    idx = np.array([6,8] if null else [5,6,7,8])
    def un(q):
        return np.array([q[0],q[0]+(8-q[0])*q[1]])
    def evaluate(q):
        c,r = design.matrices(un(q))
        return c[:,:,idx]@q[2:],r[:,idx]@q[2:]
    def physical(q):
        c,r = evaluate(q)
        return np.r_[(c/curve_sigma).ravel(),r/onset_sigma]
    a = csr_matrix((np.r_[np.tile(t/curve_sigma,n), np.full(n,1/onset_sigma)], (np.arange(n*(k+1)),np.r_[np.repeat(np.arange(n),k),np.arange(n)])), shape=(n*(k+1),n))
    obs = np.r_[(y/curve_sigma).ravel(),onset/onset_sigma]
    q0 = np.r_[taus[0],(taus[1]-taus[0])/(8-taus[0]),coeff[idx]]
    _,r = evaluate(q0)
    nd = len(q0)
    def fun(q): return physical(q[:nd])+a@q[nd:]-obs
    def jac(q):
        q = q[:nd]
        c,r = design.matrices(un(q)); b = np.empty((len(obs),nd)); b[:,2:] = np.vstack(((c[:,:,idx]/curve_sigma).reshape(-1,len(idx)),r[:,idx]/onset_sigma))
        for j in range(2):
            delta = np.zeros(nd); delta[j]=1e-5
            b[:,j]=(physical(q+delta)-physical(q-delta))/(2e-5)
        return hstack((csr_matrix(b),a),format='csr')
    def loss(z):
        result = np.empty((3,len(z))); c = z[:n*k]/9
        result[:, :n*k] = np.stack((9*np.log1p(c),1/(1+c),-1/(9*(1+c)**2)))
        result[:,n*k:] = np.stack((z[n*k:],np.ones(n),np.zeros(n)))
        return result
    fit = least_squares(fun,np.r_[q0,onset-r],jac=jac,loss=loss,bounds=(np.r_[.2,1e-6,np.full(nd+n-2,-np.inf)],np.r_[8-1e-6,1,np.full(nd+n-2,np.inf)]),x_scale='jac',max_nfev=100,tr_options={'atol':1e-9,'btol':1e-9})
    coef = np.zeros(9); coef[idx] = fit.x[2:nd]
    rr=fun(fit.x); print('noise',np.median(np.abs(rr[:n*k]))*curve_sigma,np.median(np.abs(rr[n*k:]))*onset_sigma,end='; ')
    return un(fit.x),coef

def subset(d, mask): return {k:v[mask] for k,v in d.items()}
pub=pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl','rb'))[0]; d,g=pub['dataset'],pub['gt']; t,dt,y=d['eval_times'],d['past_dt'],d['yield_stress']
rng=np.random.default_rng(902); tr=d['train']; n=len(tr['lab_id']); fold=np.zeros(n,int)
for l in range(4):
    for j in range(3):
        inds=np.flatnonzero((tr['lab_id']==l)&(tr['stratum']==j)); fold[rng.permutation(inds)]=np.arange(len(inds))%4
for split in range(-1,4):
    train,test=(tr,d['test']) if split<0 else (subset(tr,fold!=split),subset(tr,fold==split)); x,xt=m._align(train,test)
    D=m._Design(train,x,dt,t,y); T=m._Design(test,xt,dt,t,y)
    truth=g['te_clean'] if split<0 else test['strain_curves_obs']; over=(np.max(abs(test['past_stress']),1)>y)|(abs(test['stress_hold_0'])>y)
    for name,fit in [('base',m._fit),('joint',joint),('joint2',lambda a,b:joint(a,b,.019,.019)),('null',lambda a,b:joint(a,b,null=True))]:
        taus,coef=fit(D,train); c,r=T.matrices(taus); pred=test['strain_0'][:,None]+c@coef+(test['instant_rate_obs']-r@coef)[:,None]*t
        err=abs(pred-truth); metric=np.mean if split<0 else lambda r: np.mean(np.minimum(r,.15)); print(split,name,taus.round(5),coef[[5,7]].round(5), 'slice err',*[round(metric(err[mask]),5) for mask in (over|~over,over,~over)],'early/late',np.round([metric(err[:,:3]),metric(err[:,-3:])],5))
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/compare_vp.py
```
- **Container Exit Code**: `0` (`3945 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ase [0.57544 3.5774 ] [0.24136 0.15825] slice err 0.03452 0.03675 0.0325 early/late [0.01673 0.05837]
noise 0.0054398434541466315 0.0058040943779091095; 0 joint [0.57657 3.68298] [0.24285 0.15879] slice err 0.03425 0.03654 0.03218 early/late [0.01673 0.05744]
noise 0.005342339862873917 0.006154347658305269; 0 joint2 [0.57679 3.701  ] [0.24269 0.1589 ] slice err 0.03443 0.0369 0.03219 early/late [0.01674 0.05795]
noise 0.03147267386574839 0.05373295052712627; 0 null [0.54321 3.71856] [0. 0.] slice err 0.10647 0.13027 0.08493 early/late [0.05634 0.13641]
1 base [0.5794  3.72676] [0.24119 0.15989] slice err 0.03681 0.03793 0.03526 early/late [0.0191  0.05932]
noise 0.005664457107931065 0.005603949713221482; 1 joint [0.57899 3.77188] [0.24214 0.15966] slice err 0.036 0.03704 0.03458 early/late [0.01921 0.05646]
noise 0.005618305030011783 0.005974314549185564; 1 joint2 [0.57898 3.77044] [0.24226 0.15954] slice err 0.03587 0.0368 0.03458 early/late [0.01921 0.05619]
noise 0.02179723037546424 0.043256284890198914; 1 null [0.55663 3.95095] [0. 0.] slice err 0.11425 0.13131 0.09079 early/late [0.06487 0.14235]
2 base [0.57759 3.52398] [0.24863 0.15919] slice err 0.03461 0.03669 0.03304 early/late [0.01329 0.06188]
noise 0.005458641973300882 0.0055987750446414; 2 joint [0.57744 3.56387] [0.24943 0.15922] slice err 0.03465 0.03658 0.03321 early/late [0.01328 0.06218]
noise 0.005500432016836358 0.0055720311116172925; 2 joint2 [0.57739 3.55942] [0.24946 0.15917] slice err 0.03459 0.03663 0.03307 early/late [0.01328 0.06204]
noise 0.03711012073558315 0.0496383634431586; 2 null [0.54689 3.93006] [0. 0.] slice err 0.09522 0.12561 0.07243 early/late [0.04527 0.12545]
3 base [0.57669 3.52727] [0.24422 0.15853] slice err 0.03604 0.03838 0.03356 early/late [0.02991 0.05842]
noise 0.00543738337152897 0.0066275422243195405; 3 joint [0.57655 3.58121] [0.24492 0.15868] slice err 0.03558 0.03788 0.03315 early/late [0.02986 0.05738]
noise 0.0053003538601560825 0.006698380632753299; 3 joint2 [0.57659 3.59215] [0.24487 0.15884] slice err 0.03548 0.03768 0.03315 early/late [0.02983 0.0571 ]
noise 0.022929475499822494 0.040289232789077006; 3 null [0.55349 3.67166] [0. 0.] slice err 0.11295 0.12476 0.10044 early/late [0.05969 0.14534]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d6b441a10c2c1403 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d6b441a1 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`57483 ms` LLM)
- **Rationale**: The ablation falsifies the linear-only basis. Compare joint fits with and without predictive drift shrinkage on held-out cells and labs; retain the independently observed onset for missing coupons and promote shrinkage only if it improves the measured slices.
- **Alternatives Weighed**: `Promote the unregularized joint fit immediately` (rejected: It improves the closest parameter margin, but late predictions still amplify onset noise. Test whether a soft coupon-drift constraint improves that margin across labs without biasing missing-coupon predictions.)
- **ATMS NOGOOD DECLARED**: Refuted `The complete two-branch linear Prony law, with aligned coupon drift but no yield-gated terms, suffices above yield.` | Witness: `On the clean public overstressed test cohort, refitting with both viscoplastic weights fixed to zero raises MAE from 0.02720 to 2.33015; worst clipped out-of-fold overstressed MAE rises from 0.03788 to 0.13131.` | Blanket Diagnosis: `After accounting for aligned coupon drift, exact history integration, and measured onset, the linear basis still lacks the signed hold-overstress slope and accumulated-work transient.` | Sublated Pivot: `Preserve the exact Prony basis and add signed quadratic hold overstress and the history-work fast transient; both vanish on fully sub-yield trajectories.`

```bash
cat > /tmp/compare_prior_vp.py <<'PY'
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, hstack
exec(open('/tmp/compare_vp.py').read().split('pub=pickle')[0])


def fit_prior(design, train, prior_sigma, curve_sigma, onset_sigma=.016):
    taus, coef = m._fit(design, train)
    t = design.time
    y = train['strain_curves_obs'] - train['strain_0'][:, None]
    onset = train['instant_rate_obs']
    n, k = y.shape
    # Standardized independent curve and onset observations; the last block
    # estimates coupon drift deviations not explained by the panel model.
    obs = np.r_[(y/curve_sigma).ravel(), onset/onset_sigma, np.zeros(n)]
    a = csr_matrix((np.r_[np.tile(t/curve_sigma,n), np.full(n,1/onset_sigma), np.full(n,1/prior_sigma)], (np.arange(n*(k+2)), np.r_[np.repeat(np.arange(n),k), np.arange(n), np.arange(n)])),shape=(n*(k+2),n))
    def un(q): return np.array([q[0], q[0]+(8-q[0])*q[1]])
    def regular(q):
        c,r = design.matrices(un(q))
        return np.vstack(((c/curve_sigma).reshape(-1,9), r/onset_sigma, np.zeros((n,9))))
    q = np.r_[taus[0], (taus[1]-taus[0])/(8-taus[0]), coef]
    def fun(p): return regular(p)@p[2:11] + a@p[11:] - obs
    def jac(p):
        b = np.zeros((len(obs),11)); b[:,2:] = regular(p)
        for j in range(2):
            delta = np.zeros(11); delta[j] = 1e-5
            b[:,j] = ((regular(p[:11]+delta)-regular(p[:11]-delta))@p[2:11])/(2e-5)
        return hstack((csr_matrix(b),a),format='csr')
    def loss(z):
        result = np.stack((z,np.ones_like(z),np.zeros_like(z)))
        for sl,c in [(slice(None,n*k),1.0),(slice(n*(k+1),None),4.0)]:
            u = z[sl]/c
            result[:,sl] = np.stack((c*np.log1p(u),1/(1+u),-1/(c*(1+u)**2)))
        return result
    res=least_squares(fun,np.r_[q,onset-design.matrices(taus)[1]@coef],jac=jac,loss=loss,bounds=(np.r_[.2,1e-6,np.full(9+n,-np.inf)],np.r_[8-1e-6,1,np.full(9+n,np.inf)]),x_scale='jac',max_nfev=60,tr_options={'atol':1e-9,'btol':1e-9})
    return un(res.x), res.x[2:11], np.median(abs(res.x[11:]))

pub=pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl','rb'))[0]; d,g=pub['dataset'],pub['gt']; t,dt,y=d['eval_times'],d['past_dt'],d['yield_stress']
tr=d['train']; n=len(tr['lab_id']); fold=np.zeros(n,int); rng=np.random.default_rng(902)
for l in range(4):
    for j in range(3):
        inds=np.flatnonzero((tr['lab_id']==l)&(tr['stratum']==j)); fold[rng.permutation(inds)]=np.arange(len(inds))%4
for split in range(-1,8):
    mask=(fold==split) if split<4 else (tr['lab_id']==split-4)
    train,test=(tr,d['test']) if split<0 else (subset(tr,~mask),subset(tr,mask)); x,xt=m._align(train,test)
    D=m._Design(train,x,dt,t,y); T=m._Design(test,xt,dt,t,y)
    truth=g['te_clean'] if split<0 else test['strain_curves_obs']; over=(np.max(abs(test['past_stress']),1)>y)|(abs(test['stress_hold_0'])>y); miss=np.isnan(test['coupon_features_raw']).any(1)
    for ps,cs in [(0.018,.025),(.032,.025),(.032,.045),(.065,.045)]:
        taus,coef,drift=fit_prior(D,train,ps,cs); c,r=T.matrices(taus)
        residual=test['instant_rate_obs']-r@coef
        # Redescending shrinkage: large onset discrepancies indicate inadequate coupon prior, not necessarily noisy onset.
        adjusted=residual*(1-(.016**2/(ps**2+.016**2))/(1+(residual/(2*ps))**2)); adjusted[miss]=residual[miss]
        no=c@coef+test['strain_0'][:,None]+residual[:,None]*t; pred=no+(adjusted-residual)[:,None]*t
        metric=np.mean if split<0 else lambda r: np.mean(np.minimum(r,.15))
        err=abs(pred-truth); raw=abs(no-truth)
        print(split,ps,cs, np.round(taus,4),np.round(coef[[5,7]],4),'drift',round(drift,4),'onset MAD',round(np.median(abs(residual[~miss])),4),'MAE',*[round(metric(err[mask]),5) for mask in (over|~over,over,~over)],'raw',round(metric(raw),5),'late',round(metric(err[:,-3:]),5),flush=True)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/compare_prior_vp.py
```
- **Container Exit Code**: `0` (`6523 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
t MAD 0.0069 MAE 0.03336 0.03526 0.03135 raw 0.03567 late 0.0525
3 0.065 0.045 [0.5764 3.5563] [0.245  0.1585] drift 0.0045 onset MAD 0.0072 MAE 0.03502 0.03724 0.03267 raw 0.03574 late 0.05623
4 0.018 0.025 [0.5769 3.5981] [0.2453 0.1591] drift 0.0045 onset MAD 0.006 MAE 0.03037 0.03845 0.02229 raw 0.03491 late 0.04297
4 0.032 0.025 [0.5769 3.6043] [0.2452 0.1591] drift 0.0046 onset MAD 0.006 MAE 0.03243 0.04015 0.02471 raw 0.03495 late 0.04775
4 0.032 0.045 [0.5769 3.597 ] [0.2455 0.1591] drift 0.0046 onset MAD 0.0062 MAE 0.03242 0.04003 0.02482 raw 0.03491 late 0.0478
4 0.065 0.045 [0.5769 3.5999] [0.2454 0.1591] drift 0.0046 onset MAD 0.0063 MAE 0.03415 0.04147 0.02684 raw 0.03492 late 0.05175
5 0.018 0.025 [0.5774 3.7199] [0.2421 0.1591] drift 0.0029 onset MAD 0.0131 MAE 0.0286 0.0286 0.02861 raw 0.03241 late 0.04656
5 0.032 0.025 [0.5774 3.7172] [0.242 0.159] drift 0.0031 onset MAD 0.0135 MAE 0.02955 0.03023 0.02919 raw 0.03241 late 0.04872
5 0.032 0.045 [0.5775 3.6892] [0.2423 0.159 ] drift 0.0031 onset MAD 0.0134 MAE 0.02955 0.03006 0.02928 raw 0.03239 late 0.04865
5 0.065 0.045 [0.5774 3.7094] [0.242 0.159] drift 0.0031 onset MAD 0.0133 MAE 0.0314 0.03248 0.03084 raw 0.03238 late 0.05287
6 0.018 0.025 [0.5779 3.6043] [0.2479 0.1591] drift 0.0044 onset MAD 0.0125 MAE 0.03112 0.03309 0.02841 raw 0.03932 late 0.05108
6 0.032 0.025 [0.5781 3.5843] [0.2479 0.1592] drift 0.0045 onset MAD 0.0124 MAE 0.035 0.03713 0.03208 raw 0.03921 late 0.05978
6 0.032 0.045 [0.5779 3.6023] [0.248  0.1591] drift 0.0043 onset MAD 0.0127 MAE 0.03513 0.03723 0.03224 raw 0.03933 late 0.06013
6 0.065 0.045 [0.578  3.6007] [0.2501 0.1592] drift 0.0054 onset MAD 0.0106 MAE 0.03813 0.0403 0.03513 raw 0.03936 late 0.06662
7 0.018 0.025 [0.5753 3.5726] [0.2446 0.1582] drift 0.0036 onset MAD 0.0068 MAE 0.02695 0.02538 0.02915 raw 0.03414 late 0.04055
7 0.032 0.025 [0.5753 3.5561] [0.2444 0.1581] drift 0.0036 onset MAD 0.0064 MAE 0.03055 0.02914 0.03254 raw 0.03433 late 0.0489
7 0.032 0.045 [0.5753 3.5698] [0.2448 0.1582] drift 0.0037 onset MAD 0.0065 MAE 0.03025 0.02857 0.03261 raw 0.03403 late 0.04806
7 0.065 0.045 [0.5753 3.5691] [0.2448 0.1581] drift 0.0038 onset MAD 0.0062 MAE 0.03296 0.03104 0.03564 raw 0.03407 late 0.05418

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d6b441a10c2c1403 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d6b441a1 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`40571 ms` LLM)
- **Rationale**: Promote the tested joint fit and verify the exact deliverable in an isolated process. Check history-only and hold-only cases separately because aggregate overstressed MAE could conceal a miswired yield gate.
- **Alternatives Weighed**: `Use the measured onset without shrinkage` (rejected: It preserves the physical basis but amplifies onset noise at late times; conservative shrinkage improved public and held-out lab errors while retaining raw onset for incomplete coupons.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/viscoplastic_overstress_hereditary_solver.py')
s = p.read_text().replace('from scipy.optimize import least_squares', 'from scipy.optimize import least_squares\nfrom scipy.sparse import csr_matrix, hstack')
start = s.index('\ndef solve_viscoplastic_hereditary_revision')
prior = Path('/tmp/compare_prior_vp.py').read_text().split('def fit_prior', 1)[1].split('\npub=', 1)[0]
prior = ('\n\ndef _fit_joint' + prior).replace('prior_sigma, curve_sigma, onset_sigma=.016', 'prior_sigma=0.018, curve_sigma=0.025, onset_sigma=0.016').replace('    taus, coef = m._fit(design, train)', '    """Fit onset and curves jointly, allowing a small coupon-specific drift."""\n    taus, coef = _fit(design, train)').replace(', np.median(abs(res.x[11:]))', '')
s = s[:start] + prior.rstrip() + '\n' + s[start:]
s = s.replace('taus, coef = _fit(_Design(train, x_train, dt, time, y), train)', 'taus, coef = _fit_joint(_Design(train, x_train, dt, time, y), train)')
s = s.replace('    # The measured onset constrains the individual coupon\'s steady drift.\n    correction = test[\'instant_rate_obs\'] - rate @ coef', '''    # Shrink small onset discrepancies toward the calibrated coupon drift.
    # Keep the measured onset when the coupon is incomplete or disagrees strongly.
    correction = test['instant_rate_obs'] - rate @ coef
    complete = np.isfinite(test['coupon_features_raw']).all(axis=1)
    shrinkage = (0.016**2 / (0.018**2 + 0.016**2)) / (1 + (correction / 0.036)**2)
    correction = correction * (1 - np.where(complete, shrinkage, 0.0))''')
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/validate_vp.py
python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
p = '/holdout_task/viscoplastic_overstress_hereditary_solver.py'
spec = importlib.util.spec_from_file_location('solver', p)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for row in pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl', 'rb')):
    d, gt = row['dataset'], row['gt']; a, b = d['train'], d['test']; y = d['yield_stress']
    out = m.solve_viscoplastic_hereditary_revision(d)
    tau = out['estimated_taus']; vp = out['viscoplastic_weights']; err = abs(out['predicted_strain'] - gt['te_clean'])
    h = (abs(b['past_stress']) > y).any(1); hold = abs(b['stress_hold_0']) > y
    assert 0.2 <= tau[0] < tau[1] <= 8 and np.max(abs(tau / gt['taus'] - 1)) <= 0.10
    assert np.linalg.norm(vp - gt['vp_theta']) / np.linalg.norm(gt['vp_theta']) <= 0.14
    assert err.mean() <= .070 and err[h | hold].mean() <= .080 and err[~(h | hold)].mean() <= .065
    _, x = m._align(a, b); design = m._Design(b, x, d['past_dt'], d['eval_times'], y)
    curves, rate = design.matrices(tau)
    np.testing.assert_array_equal(curves[~h,:,7], 0.0)
    np.testing.assert_array_equal(curves[~hold,:,5], 0.0)
    for name, mask in [('history only',h & ~hold), ('hold only',hold & ~h), ('both',h & hold)]:
        if mask.any():
            print(name, int(mask.sum()), 'MAE', round(err[mask].mean(), 6))
            assert err[mask].mean() <= .080
PY
```
- **Container Exit Code**: `0` (`1756 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoplastic_overstress_hereditary_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
runtime 0.23894828488118947
taus [0.57700153 3.61625434] rel [0.00516978 0.0727553 ] vp [0.24634871 0.1589206 ] rel 0.02232603622477567
all 90 0.019647100166075262 last 0.05436373601550313
over 37 0.02182047057809049 last 0.05988181596626769
sub 53 0.018129841576555197 last 0.050511491521573165
over+ 24 0.021290553497483808 last 0.05851055994317004
over- 13 0.02279877903459514 last 0.06241336554737106
missing coupon 9 0.030929414697673508 last 0.08504057687121198
lab 0 22 0.023370808135350873 last 0.06521221659773754
lab 1 23 0.015486400840459942 last 0.043041031637712036
lab 2 23 0.014869422590038943 last 0.041117462588824115
lab 3 22 0.025268058957617273 last 0.06920100495612386
both 37 MAE 0.02182

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=92aba501ba26ffe2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_92aba501 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`35045 ms` LLM)
- **Rationale**: Keep the verified artifact and test the two missing loading regimes at the exact yield boundary against an independently stepped reference law. A failing slice would warrant checking the corresponding gate or signed contribution before submission.
- **Alternatives Weighed**: `Submit on the public split alone` (rejected: Every overstressed public test coupon activates both gates, so that split cannot show whether predictions stay accurate when only one mechanism is active.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
path = '/holdout_task/viscoplastic_overstress_hereditary_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('/holdout_task/public_suc01_C_datasets.pkl', 'rb') as f:
    rows = pickle.load(f)
for row in rows:
    d, g = row['dataset'], row['gt']
    q, y = d['test'], float(d['yield_stress'])
    h, s = q['past_stress'], q['stress_hold_0']
    both = (np.abs(h) > y).any(1) & (np.abs(s) > y)
    for regime in ('history only', 'hold only'):
        test = {k: np.asarray(v)[both].copy() for k, v in q.items()}
        a, b = test['past_stress'], test['stress_hold_0']
        if regime == 'history only':
            b[:] = np.clip(b, -y, y)
        else:
            a[:] = np.clip(a, -y, y)
        ts, dt, tau = d['eval_times'], d['past_dt'], g['taus']
        z = np.zeros((len(b), 2))
        for level, step in zip(a.T, dt):
            z = level[:, None] + (z - level[:, None]) * np.exp(-step / tau)
        theta = g['theta']
        drift = theta[2] + g['te_Z'][both] @ theta[3:]
        work = np.sum(np.maximum(np.abs(a) - y, 0)**2 * dt, axis=1)
        slope = drift + g['vp_theta'][0] * np.sign(b) * np.maximum(np.abs(b) - y, 0)**2
        amp = (b[:, None] - z) * theta[:2]
        amp[:, 0] += g['vp_theta'][1] * work * b
        clean = test['strain_0'][:, None] + slope[:, None] * ts + amp @ (1 - np.exp(-ts[None, :] / tau[:, None]))
        # Match the public onset measurement error; leave real training data unchanged.
        test['instant_rate_obs'] = slope + amp @ (1 / tau) + q['instant_rate_obs'][both] - g['te_true_rate0'][both]
        pred = solver.solve_viscoplastic_hereditary_revision(dict(d, test=test))['predicted_strain']
        err = np.abs(pred - clean)
        for sign, mask in [('positive', b > 0), ('negative', b < 0)]:
            mae = np.mean(err[mask])
            print(regime, sign, int(mask.sum()), 'MAE', round(float(mae), 6))
            assert mae <= 0.080
PY
```
- **Container Exit Code**: `0` (`872 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
history only positive 24 MAE 0.018453
history only negative 13 MAE 0.022094
hold only positive 24 MAE 0.018389
hold only negative 13 MAE 0.022517

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=92aba501ba26ffe2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_92aba501 (directory).
```

### Turn 9 — `submit_completed_task` (`43004 ms` LLM)
- **Rationale**: The final artifact passed isolated-process schema and accuracy checks, and separate history-only and hold-only tests passed for both stress signs. No further edits are warranted.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/viscoplastic_overstress_hereditary_solver.py': 'FILE:7233'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoplastic_overstress_hereditary_solver.py`. Public MAE was 0.01965 overall, 0.02182 above yield, and 0.01813 below yield. Maximum relaxation-time error was 7.28%, and relative viscoplastic-weight error was 2.23%. All specified tolerances passed.`

### Turn 10 — `execute_in_remote_docker` (`40394 ms` LLM)
- **Rationale**: Compare the implemented pairwise branch gaps and yield terms with an independent recurrence on real histories. Then rerun the delivered estimator with each lab's outcomes held out to see whether pooled accuracy hides calibration bias or unstable parameter recovery. A recurrence mismatch would require a basis fix; a lab-specific failure would motivate a calibration change.
- **Alternatives Weighed**: `Probe relay modes, classification accuracy, or unrelated drift operators` (rejected: The solver has no relay state or classification target; signed residuals by held-out lab and stratum test the relevant calibration failure.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
p = '/holdout_task/viscoplastic_overstress_hereditary_solver.py'
s = importlib.util.spec_from_file_location('solver', p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('/holdout_task/public_suc01_C_datasets.pkl', 'rb') as f:
    rows = pickle.load(f)

def subset(split, mask):
    return {k: np.asarray(v)[mask] for k, v in split.items()}

for row in rows:
    d, g = row['dataset'], row['gt']
    tr, te = d['train'], d['test']
    dt, t, y = d['past_dt'], d['eval_times'], d['yield_stress']
    out = m.solve_viscoplastic_hereditary_revision(d)
    tau, vp = out['estimated_taus'], out['viscoplastic_weights']
    _, x = m._align(tr, te)
    c, r = m._Design(te, x, dt, t, y).matrices(tau)
    hold, hist = te['stress_hold_0'], te['past_stress']
    z = np.zeros((len(hold), 2))
    for load, step in zip(hist.T, dt):
        z += (load[:, None] - z) * -np.expm1(-step / tau)
    work = np.sum(np.maximum(np.abs(hist) - y, 0)**2 * dt, axis=1) * hold
    over = np.sign(hold) * np.maximum(np.abs(hold) - y, 0)**2
    transient = np.column_stack((hold-z[:,0], work, hold-z[:,1]))
    ix = [0,0,1]
    np.testing.assert_allclose(c[:,:,6:], transient[:,None,:] * -np.expm1(-t[None,:,None]/tau[ix]), atol=1e-12)
    np.testing.assert_allclose(r[:,6:], transient/tau[ix], atol=1e-12)
    np.testing.assert_allclose(c[:,:,5], over[:,None]*t, atol=1e-12)
    np.testing.assert_allclose(r[:,5], over, atol=1e-12)
    public = out['predicted_strain'] - g['te_clean']
    active = (np.abs(hist)>y).any(axis=1) | (np.abs(hold)>y)
    limits = np.array([.10,.14,.070,.080,.065])
    metrics = np.array([np.max(abs(tau/g['taus']-1)), np.linalg.norm(vp-g['vp_theta'])/np.linalg.norm(g['vp_theta']), np.abs(public).mean(), np.abs(public[active]).mean(), np.abs(public[~active]).mean()])
    print('public errors', metrics.round(6), 'fraction of tolerance', (metrics/limits).round(3))
    assert np.all(metrics <= limits)
    fold_metrics = []
    for lab in np.unique(tr['lab_id']):
        mask = tr['lab_id'] == lab
        fit, val = subset(tr, ~mask), subset(tr, mask)
        pred = m.solve_viscoplastic_hereditary_revision(dict(d, train=fit, test=val))
        e = pred['predicted_strain'] - val['strain_curves_obs']
        active = (np.abs(val['past_stress']) > y).any(axis=1) | (np.abs(val['stress_hold_0']) > y)
        errs = [np.minimum(np.abs(e[k]), .15).mean() for k in (active, ~active) if k.any()]
        rate = abs(pred['estimated_taus']/g['taus']-1).max()
        wp = np.linalg.norm(pred['viscoplastic_weights']-g['vp_theta'])/np.linalg.norm(g['vp_theta'])
        fold_metrics.append([max(errs), rate, wp])
        print('held-out lab', lab, 'clipped MAE by yield', np.round(errs,6), 'median signed late error by stratum', [round(float(np.median(e[val['stratum']==j,-3:])),6) for j in np.unique(val['stratum'])], 'tau/vp relative error', round(rate,6), round(wp,6))
    print('worst held-out lab: clipped yield-slice MAE, tau error, vp error', np.max(fold_metrics,axis=0).round(6))
PY
```
- **Container Exit Code**: `0` (`1759 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
public errors [0.072755 0.022326 0.019647 0.02182  0.01813 ] fraction of tolerance [0.728 0.159 0.281 0.273 0.279]
held-out lab 0 clipped MAE by yield [0.038447 0.022286] median signed late error by stratum [0.00592, 0.006871, -0.009932] tau/vp relative error 0.077417 0.018545
held-out lab 1 clipped MAE by yield [0.0286   0.028606] median signed late error by stratum [-0.028976, -0.004519, -0.003216] tau/vp relative error 0.046189 0.007819
held-out lab 2 clipped MAE by yield [0.033092 0.028414] median signed late error by stratum [-0.004428, 0.013993, -0.003226] tau/vp relative error 0.075823 0.027485
held-out lab 3 clipped MAE by yield [0.025377 0.029149] median signed late error by stratum [-0.000204, -0.038973, 0.003282] tau/vp relative error 0.083943 0.017004
worst held-out lab: clipped yield-slice MAE, tau error, vp error [0.038447 0.083943 0.027485]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=92aba501ba26ffe2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_92aba501 (directory).
```

### Turn 11 — `submit_completed_task` (`17424 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process schema and accuracy checks. An independent recurrence matched the implemented basis, both gates passed separately for both stress signs, and the worst held-out-lab relaxation-time error was 8.39% against the 10% limit. The audit did not identify a failure that warrants another edit.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/viscoplastic_overstress_hereditary_solver.py': 'FILE:7233'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoplastic_overstress_hereditary_solver.py`. All public tolerances passed: strain MAE was 0.01965 overall, 0.02182 for overstressed coupons, and 0.01813 for sub-yield coupons. Maximum relaxation-time error was 7.28%, and relative viscoplastic-weight error was 2.23%.`

### Turn 12 — `execute_in_remote_docker` (`32632 ms` LLM)
- **Rationale**: Check whether moderate curve glitches still bias the slow-time estimate by comparing two tighter robust-loss scales on the same held-out labs. Change the solver only if the challenger improves the slow-time margin without a material regression in weight recovery or yield-cohort predictions.
- **Alternatives Weighed**: `Repeat the basis and lab-confounding audit` (rejected: The independent recurrence and held-out-lab residual checks already passed; the remaining tight margin is relaxation-time recovery.)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
p = '/holdout_task/viscoplastic_overstress_hereditary_solver.py'
s = importlib.util.spec_from_file_location('solver', p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('/holdout_task/public_suc01_C_datasets.pkl', 'rb') as f:
    rows = pickle.load(f)
for row in rows:
    d,g = row['dataset'],row['gt']; tr = d['train']; t,dt,y = d['eval_times'],d['past_dt'],d['yield_stress']
    def take(q, mask):
        return {k: np.asarray(v)[mask] for k,v in q.items()}
    for scale in (.025,.016,.012):
        scores = []
        for lab in [None,*np.unique(tr['lab_id'])]:
            a,b = (tr,d['test']) if lab is None else (take(tr,tr['lab_id']!=lab),take(tr,tr['lab_id']==lab))
            x,xt = m._align(a,b)
            taus,coef = m._fit_joint(m._Design(a,x,dt,t,y),a,curve_sigma=scale)
            c,r = m._Design(b,xt,dt,t,y).matrices(taus)
            delta = b['instant_rate_obs'] - r@coef
            complete = np.isfinite(b['coupon_features_raw']).all(1)
            delta *= 1-complete*(.016**2/(.018**2+.016**2))/(1+(delta/.036)**2)
            pred = b['strain_0'][:,None]+c@coef+delta[:,None]*t
            active = (np.abs(b['past_stress'])>y).any(1)|(np.abs(b['stress_hold_0'])>y)
            err = abs(pred-(g['te_clean'] if lab is None else b['strain_curves_obs']))
            if lab is not None: err = np.minimum(err,.15)
            score = [np.max(abs(taus/g['taus']-1)),np.linalg.norm(coef[[5,7]]-g['vp_theta'])/np.linalg.norm(g['vp_theta']),err.mean(),err[active].mean(),err[~active].mean(),err[:,-3:].mean()]
            scores.append(score)
        print('Cauchy scale',scale,'[tau rel max, weight rel L2, all MAE, over MAE, sub MAE, late MAE]')
        print('  clean public:',np.round(scores[0],6),'worst held-out lab (clipped strain):',np.round(np.max(scores[1:],axis=0),6))
PY
```
- **Container Exit Code**: `0` (`3203 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Cauchy scale 0.025 [tau rel max, weight rel L2, all MAE, over MAE, sub MAE, late MAE]
  clean public: [0.072755 0.022326 0.019647 0.02182  0.01813  0.042295] worst held-out lab (clipped strain): [0.083943 0.027485 0.031122 0.038447 0.029149 0.051077]
Cauchy scale 0.016 [tau rel max, weight rel L2, all MAE, over MAE, sub MAE, late MAE]
  clean public: [0.072311 0.021413 0.019852 0.022109 0.018277 0.042727] worst held-out lab (clipped strain): [0.085498 0.02577  0.031149 0.038737 0.029228 0.050912]
Cauchy scale 0.012 [tau rel max, weight rel L2, all MAE, over MAE, sub MAE, late MAE]
  clean public: [0.073141 0.021625 0.019863 0.022101 0.018301 0.042806] worst held-out lab (clipped strain): [0.087438 0.027098 0.031103 0.038914 0.029208 0.050895]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=92aba501ba26ffe2 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_92aba501 (directory).
```

### Turn 13 — `submit_completed_task` (`17226 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process schema and accuracy checks, matched an independent Prony recurrence, and passed separate tests for each yield gate under both stress signs. The final loss-scale comparison did not improve the worst-case parameter margin.
- **Alternatives Weighed**: `Use a tighter Cauchy loss scale` (rejected: Neither tighter scale improved the limiting margin: worst held-out-lab relaxation-time error increased from 8.39% to 8.55% and 8.74%.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoplastic_overstress_hereditary_solver.py`. All public tolerances passed: strain MAE was 0.01965 overall, 0.02182 for overstressed coupons, and 0.01813 for sub-yield coupons. Maximum relaxation-time error was 7.28%, and relative viscoplastic-weight error was 2.23%.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M93-Lifecycle",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/viscoplastic_overstress_hereditary_solver.py": "FILE:7233"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "max_tau_rel_err": 0.031344,
    "max_vp_rel_l2": 0.01461,
    "max_all_mae": 0.019255,
    "max_overstress_mae": 0.017931,
    "max_subyield_mae": 0.020179,
    "student_elapsed_sec": 0.4897,
    "instance_diagnostics": [
      {
        "seed": 99521,
        "tau_rel": 0.023957,
        "vp_rel_l2": 0.0061,
        "all_mae": 0.019255,
        "over_mae": 0.017931,
        "sub_mae": 0.020179
      },
      {
        "seed": 99522,
        "tau_rel": 0.031344,
        "vp_rel_l2": 0.01461,
        "all_mae": 0.018439,
        "over_mae": 0.017851,
        "sub_mae": 0.018849
      }
    ],
    "candidate_timing_breakdown": {
      "import_wall_sec": 0.088862,
      "import_cpu_sec": 0.081265,
      "candidate_call_wall_sec": 0.440461,
      "candidate_call_cpu_sec": 0.402781,
      "ipc_and_codec_wall_sec": 0.126835,
      "rpc_call_count": 2,
      "verifier_and_gt_wall_sec": 0.008738,
      "verifier_and_gt_cpu_sec": 0.031531,
      "total_supervisor_wall_sec": 0.664896
    }
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"max_tau_rel_err\": 0.031344, \"max_vp_rel_l2\": 0.01461, \"max_all_mae\": 0.019255, \"max_overstress_mae\": 0.017931, \"max_subyield_mae\": 0.020179, \"student_elapsed_sec\": 0.4897, \"instance_diagnostics\": [{\"seed\": 99521, \"tau_rel\": 0.023957, \"vp_rel_l2\": 0.0061, \"all_mae\": 0.019255, \"over_mae\": 0.017931, \"sub_mae\": 0.020179}, {\"seed\": 99522, \"tau_rel\": 0.031344, \"vp_rel_l2\": 0.01461, \"all_mae\": 0.018439, \"over_mae\": 0.017851, \"sub_mae\": 0.018849}]}\n__HEGEL_VERIFIER_AUTH_9026d84b3f10dcc714d93248d5f2ee88__:{\"strict_pass\": true, \"reward\": 1.0, \"max_tau_rel_err\": 0.031344, \"max_vp_rel_l2\": 0.01461, \"max_all_mae\": 0.019255, \"max_overstress_mae\": 0.017931, \"max_subyield_mae\": 0.020179, \"student_elapsed_sec\": 0.4897, \"instance_diagnostics\": [{\"seed\": 99521, \"tau_rel\": 0.023957, \"vp_rel_l2\": 0.0061, \"all_mae\": 0.019255, \"over_mae\": 0.017931, \"sub_mae\": 0.020179}, {\"seed\": 99522, \"tau_rel\": 0.031344, \"vp_rel_l2\": 0.01461, \"all_mae\": 0.018439, \"over_mae\": 0.017851, \"sub_mae\": 0.018849}], \"candidate_timing_breakdown\": {\"import_wall_sec\": 0.088862, \"import_cpu_sec\": 0.081265, \"candidate_call_wall_sec\": 0.440461, \"candidate_call_cpu_sec\": 0.402781, \"ipc_and_codec_wall_sec\": 0.126835, \"rpc_call_count\": 2, \"verifier_and_gt_wall_sec\": 0.008738, \"verifier_and_gt_cpu_sec\": 0.031531, \"total_supervisor_wall_sec\": 0.664896}}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5570.98,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
