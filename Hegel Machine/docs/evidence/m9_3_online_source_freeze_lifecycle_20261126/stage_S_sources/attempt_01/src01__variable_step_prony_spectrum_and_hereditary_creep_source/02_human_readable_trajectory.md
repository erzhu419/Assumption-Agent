# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `src01__variable_step_prony_spectrum_and_hereditary_creep_source`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M93-Lifecycle` |
| **Domain** | `Multi-Timescale Variable-Step Hereditary Viscoelastic Spectrum Identification & Twin Creep Forecasting` |
| **Wave / Wormhole Family** | `m93_stage_S_source` / `fam_hereditary_viscoelastic_memory` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`PASS (1.0)`** |
| **Failure / Outcome Taxonomy** | `PASSED_ALL_CHECKS` |
| **Turns Used / Max Turns** | `9 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `215,776` (`206,236` / `29,056` / `9,540`) |
| **Total Episode Wall Time (s)** | `310.26s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`17646 ms` LLM)
- **Rationale**: Inspect the validator and summarize arrays in place to detect schema or noise assumptions that would change the fit or validation design, without transferring the dataset.

```bash
cd /holdout_task && cat validate_public_schema.py && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment imputation Cauchy robust IRLS viscoelastic relaxation'; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_src01_datasets.pkl','rb'))
def show(v,p=''):
 if isinstance(v,dict):
  for k,w in v.items(): show(w,p+'/'+str(k))
 elif isinstance(v,(list,tuple)) and v and isinstance(v[0],dict):
  print(p,len(v)); [show(w,p+'/'+str(i)) for i,w in enumerate(v[:8])]
 elif isinstance(v,np.ndarray):
  print(p,v.shape,v.dtype, np.round(np.nanpercentile(v,[0,10,50,90,100]),5) if v.size and v.dtype.kind in 'fiu' else '', np.isnan(v).sum() if v.dtype.kind=='f' else '')
 else: print(p,str(v)[:250])
show(x)
PY
```
- **Container Exit Code**: `0` (`640 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
 1
/0/dataset/past_dt (28,) float64 [0.1262  0.1741  0.27655 0.39212 0.4199 ] 0
/0/dataset/eval_times (10,) float64 [0.2   0.425 2.225 6.7   8.5  ] 0
/0/dataset/train/lab_id (148,) int64 [0.  0.  1.5 3.  3. ] 
/0/dataset/train/stratum (148,) int64 [0. 0. 1. 2. 2.] 
/0/dataset/train/coupon_features_raw (148, 4) float64 [-5.06203 -3.74939 -0.51996  4.59116  6.85576] 15
/0/dataset/train/ref_coupons (148, 20) float64 [-5.73559 -3.9266  -0.21686  4.59833  7.75776] 115
/0/dataset/train/past_stress (148, 28) float64 [-3.13315 -1.27816  0.01971  1.29623  3.14275] 0
/0/dataset/train/strain_0 (148,) float64 [-0.59385 -0.40294  0.09429  0.42846  0.60189] 0
/0/dataset/train/stress_hold_0 (148,) float64 [-2.14031 -1.80816  0.69834  1.82209  2.14417] 0
/0/dataset/train/instant_rate_obs (148,) float64 [-4.21296 -2.73961  0.65475  2.91169  5.02605] 0
/0/dataset/train/strain_curves_obs (148, 10) float64 [-7.74291 -2.43585  0.55708  3.20527  9.75294] 0
/0/dataset/test/lab_id (90,) int64 [0.  0.  1.5 3.  3. ] 
/0/dataset/test/stratum (90,) int64 [0. 0. 1. 2. 2.] 
/0/dataset/test/coupon_features_raw (90, 4) float64 [-5.04921 -3.82659  0.49125  4.49791  6.50862] 8
/0/dataset/test/ref_coupons (90, 20) float64 [-5.33779 -3.83738 -0.52829  4.52049  7.69783] 74
/0/dataset/test/past_stress (90, 28) float64 [-6.3108  -1.71343  0.01775  1.72872  5.82443] 0
/0/dataset/test/strain_0 (90,) float64 [-0.64315 -0.42504  0.07641  0.39561  0.59027] 0
/0/dataset/test/stress_hold_0 (90,) float64 [-2.14965 -1.73564  0.75139  1.58718  2.02812] 0
/0/dataset/test/instant_rate_obs (90,) float64 [-3.96398 -2.51525  0.61665  2.56864  3.71499] 0
/0/gt/taus (2,) float64 [0.6  0.94 2.3  3.66 4.  ] 0
/0/gt/theta (7,) float64 [-0.13  -0.106  0.11   0.596  0.68 ] 0
/0/gt/vp_theta (2,) float64 [0. 0. 0. 0. 0.] 0
/0/gt/te_Z (90, 4) float64 [-2.52848 -1.14978 -0.02285  1.25069  2.39454] 0
/0/gt/te_clean (90, 10) float64 [-7.3135  -2.32419  0.45708  3.06603  7.93484] 0
/0/gt/te_true_rate0 (90,) float64 [-3.95721 -2.51871  0.60802  2.56642  3.71705] 0
/0/gt/te_cross (90,) float64 [0.06636 0.12699 1.22732 8.5     8.5    ] 0
/0/gt/te_secant (90,) float64 [-7.12182 -4.00757  0.75935  3.76595  9.65925] 0
```

### Turn 2 — `execute_in_remote_docker` (`37585 ms` LLM)
- **Rationale**: The public oracle check distinguishes an implementation error in the exact recurrence from prediction error introduced by the mandated feature preprocessing. The residual diagnostics inform robust weighting without using oracle labels in the solver.
- **Alternatives Weighed**: `Fit strain and rate at a common residual scale` (rejected: The two channels have different units; their noise and shared drift should be measured before weighting them.)

```bash
cd /holdout_task
export PYTHONPATH=/tmp
for fn in align_reference_panel balanced_cell_weights weighted_column_fill; do python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.$fn | head -35; done
python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_affine_projection.robust_residual_weights | head -35
python3 - <<'PY'
import pickle,numpy as np,scipy
p=pickle.load(open('public_src01_datasets.pkl','rb'))[0]; d=p['dataset']; g=p['gt']; a,b=[d[k] for k in ('train','test')]; n=len(a['lab_id']); c=np.r_[a['lab_id'],b['lab_id']]; s=np.r_[a['stratum'],b['stratum']]; R=np.r_[a['ref_coupons'],b['ref_coupons']]; X=np.r_[a['coupon_features_raw'],b['coupon_features_raw']]; X0=X.copy(); _,j,k=np.unique(np.c_[c,s],axis=0,return_inverse=True,return_counts=True); w=1/k[j];
for i in np.unique(c):
 m=c==i; X[m]=(X[m]-np.median(np.nanmean(R[m],axis=0)))/np.median(np.nanstd(R[m],axis=0))
X=np.where(np.isnan(X),np.nansum(w[:,None]*X,axis=0)/np.sum(w[:,None]*np.isfinite(X),axis=0),X)
def pred(split,x):
 T=d['eval_times']; dt=d['past_dt']; tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; tau=g['taus']; h=split['stress_hold_0'][:,None]-split['past_stress']@(-np.expm1(-dt[:,None]/tau)*np.exp(-tail[:,None]/tau)); q=g['theta']; L=q[2]+x@q[3:]; return split['strain_0'][:,None]+(h*q[:2])@(-np.expm1(-T[:,None]/tau)).T+L[:,None]*T, (h*q[:2])@(1/tau)+L
pt,rt=pred(b,X[n:]); pa,ra=pred(a,X[:n]); pc,rc=pred(b,g['te_Z']);
print('scipy',scipy.__version__,'theta',g['theta']); print('gt features reconstruct MAE',np.abs(pc-g['te_clean']).mean(),np.abs(rc-g['te_true_rate0']).mean()); print('required aligned features at true theta MAE',np.abs(pt-g['te_clean']).mean(),np.abs(rt-g['te_true_rate0']).mean()); print('complete/missing feature MAE',[(m.sum(),np.abs(pt[m]-g['te_clean'][m]).mean(),np.abs(rt[m]-g['te_true_rate0'][m]).mean()) for m in (np.isfinite(X0[n:]).all(1),np.isnan(X0[n:]).any(1))]); print('rate observation error quantiles',np.quantile(np.abs(b['instant_rate_obs']-g['te_true_rate0']),[.25,.5,.75,.9,.99,1])); rr=a['strain_curves_obs']-pa; re=a['instant_rate_obs']-ra; zz=rr-re[:,None]*d['eval_times']; print('train strain abs med per time',np.median(np.abs(rr),axis=0)); print('train rate-cancelled strain abs med per time',np.median(np.abs(zz),axis=0)); print('train rate error quantiles',np.quantile(np.abs(re),[.25,.5,.75,.9,.99,1])); print('adjacent test feature distances',np.quantile(np.linalg.norm(g['te_Z'][::2]-g['te_Z'][1::2],axis=1),[0,.25,.5,.75,1])); print('adjacent test stress-hold distances',np.quantile(np.abs(b['stress_hold_0'][::2]-b['stress_hold_0'][1::2]),[0,.5,1])); print('all test/train keys',list(b),list(a))
PY
```
- **Container Exit Code**: `0` (`696 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
hape (steps+1, n, k) from transition (n, n), initial ( | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.propagate_linear_basis",
        "autogenous_prim_robust_affine_projection@abc35d3d | profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | For predictor, target and optional weights (n, m) and predictor sensitivity (n | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.profile_weighted_affine",
        "autogenous_prim_robust_affine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar or scale array broadcastable to | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights"
      ],
      "execution_available": true,
scipy 1.13.0 theta [ 0.68  0.54  0.09  0.16 -0.13  0.11 -0.09]
gt features reconstruct MAE 1.1233298242214258e-16 1.5358085173981332e-16
required aligned features at true theta MAE 0.05641788625638722 0.018407140703552125
complete/missing feature MAE [(82, 0.026191804654181784, 0.008545450131870117), (8, 0.3662352226789928, 0.11948946906329272)]
rate observation error quantiles [0.00309351 0.00685593 0.01210201 0.016579   0.02270579 0.02734658]
train strain abs med per time [0.00641729 0.00684079 0.00836714 0.01122835 0.01583449 0.01836717
 0.02697672 0.0332139  0.03867508 0.05651311]
train rate-cancelled strain abs med per time [0.00554054 0.00604731 0.00766094 0.00954209 0.01618997 0.01895945
 0.02385127 0.03444425 0.0446429  0.0540878 ]
train rate error quantiles [0.00522951 0.01103121 0.01834395 0.03460011 0.29927872 0.41599125]
adjacent test feature distances [0.         1.25615817 1.87830667 3.02276598 4.78337557]
adjacent test stress-hold distances [0.         1.74570336 3.71430684]
all test/train keys ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress', 'strain_0', 'stress_hold_0', 'instant_rate_obs'] ['lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons', 'past_stress', 'strain_0', 'stress_hold_0', 'instant_rate_obs', 'strain_curves_obs']
```

### Turn 3 — `execute_in_remote_docker` (`59622 ms` LLM)
- **Rationale**: The true-feature reconstruction validates the recurrence to machine precision. The candidate uses the required reference alignment and weighted imputation, then fits both observation channels with separate robust scales. Public lab, stratum, horizon, missingness, and twin diagnostics will distinguish spectral bias from errors concentrated in coupon preprocessing.
- **Alternatives Weighed**: `Substitute the observed test rate for the model rate` (rejected: This would bypass the required shared-weight physical derivative prediction.)

```bash
cd /holdout_task && cat > hereditary_prony_source_solver.py <<'PY'
"""Robust identification of two variable-step hereditary creep branches."""

import numpy as np
from scipy.optimize import least_squares


def _panel(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    refs = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    _, cell, count = np.unique(np.column_stack((lab, stratum)), axis=0,
                               return_inverse=True, return_counts=True)
    weight = len(lab) / (len(count) * count[cell])
    for group in np.unique(lab):
        mask = lab == group
        panel = refs[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[mask] = (x[mask] - offset) / gain
    valid = np.isfinite(x)
    mass = np.sum(weight[:, None] * valid, axis=0)
    weighted_sum = np.sum(weight[:, None] * np.where(valid, x, 0.0), axis=0)
    fill = np.divide(weighted_sum, mass, out=np.zeros_like(mass), where=mass > 0)
    return np.where(valid, x, fill), weight


def _design(tau, dt, tail, time, split, x):
    memory_weights = -np.expm1(-dt[:, None] / tau) * np.exp(-tail[:, None] / tau)
    stress_gap = split['stress_hold_0'][:, None] - split['past_stress'] @ memory_weights
    creep = -np.expm1(-time[:, None] / tau)
    drift = np.column_stack((np.ones(len(x)), x))
    curve = np.concatenate((stress_gap[:, None, :] * creep[None, :, :],
                            drift[:, None, :] * time[None, :, None]), axis=2)
    rate = np.column_stack((stress_gap / tau, drift))
    return curve, rate


def solve_hereditary_prony_source(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    x, weights = _panel(train, test)
    n = len(train['lab_id'])
    target = np.concatenate(((train['strain_curves_obs'] - train['strain_0'][:, None]).ravel(),
                             train['instant_rate_obs']))
    scale = np.r_[np.tile(0.012 + 0.008 * time, n), np.full(n, 0.020)]
    weight = np.r_[np.repeat(weights[:n], len(time)), weights[:n]]
    target = target / scale

    def taus(p):
        return np.array([0.20 + 7.8 * p[0], 0.20 + 7.8 * (p[0] + (1 - p[0]) * p[1])])

    def matrix(tau):
        curve, rate = _design(tau, dt, tail, time, train, x[:n])
        return np.vstack((curve.reshape(-1, 7), rate)) / scale[:, None]

    def weighted_fit(a, w):
        root = np.sqrt(w)
        return np.linalg.lstsq(a * root[:, None], target * root, rcond=None)[0]

    def residual(p):
        r = np.einsum('ij,j->i', matrix(taus(p)), p[2:]) - target
        # Square of this residual is weighted Cauchy loss. Applying the
        # cell weights outside the logarithm preserves the noise scales.
        return np.sqrt(weight) * np.copysign(np.sqrt(np.log1p(r * r)), r)

    starts = []
    grid = np.geomspace(0.22, 7.95, 11)
    for i, fast in enumerate(grid[:-1]):
        for slow in grid[i + 1:]:
            a = matrix(np.array([fast, slow]))
            coef = weighted_fit(a, weight)
            for anneal in (5., 3., 2., 1., 1.):
                r = np.einsum('ij,j->i', a, coef) - target
                coef = weighted_fit(a, weight / (1 + (r / anneal) ** 2))
            p = np.r_[(fast - 0.20) / 7.8, (slow - fast) / (8 - fast), coef]
            r = residual(p)
            starts.append((r @ r, p))

    lo = np.r_[0.0, 1e-7, np.full(7, -np.inf)]
    hi = np.r_[1 - 1e-7, 1.0, np.full(7, np.inf)]
    best = min((least_squares(residual, p, bounds=(lo, hi), x_scale='jac', max_nfev=150)
                for _, p in sorted(starts, key=lambda item: item[0])[:4]),
               key=lambda result: result.cost)
    tau, theta = taus(best.x), best.x[2:]
    curve, rate = _design(tau, dt, tail, time, test, x[n:])
    return {'estimated_taus': tau, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve @ theta,
            'initial_rate_pred': rate @ theta}
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,time,numpy as np
from hereditary_prony_source_solver import solve_hereditary_prony_source
p=pickle.load(open('public_src01_datasets.pkl','rb'))[0]; d=p['dataset']; g=p['gt']; t=time.monotonic(); o=solve_hereditary_prony_source(d)
e=np.abs(o['predicted_strain']-g['te_clean']); r=np.abs(o['initial_rate_pred']-g['te_true_rate0']); b=d['test']; lab=b['lab_id']; st=b['stratum']; missing=np.isnan(b['coupon_features_raw']).any(axis=1)
twins=(np.isclose(b['stress_hold_0'][:,None],b['stress_hold_0'][None,:],rtol=0,atol=1e-12)&np.isclose(b['strain_0'][:,None],b['strain_0'][None,:],rtol=0,atol=1e-12)); i,j=np.where(np.triu(twins,1));
print('runtime',time.monotonic()-t,'tau',o['estimated_taus'],'theta',o['hereditary_weights'],'tau relative error',np.abs(o['estimated_taus']/g['taus']-1)); print('overall',e.mean(),r.mean(),'by horizon',e.mean(0)); print('lab,stratum,count,strain,rate',[(int(c),int(s),int(m.sum()),round(e[m].mean(),5),round(r[m].mean(),5)) for c,s in np.unique(np.c_[lab,st],axis=0) for m in [(lab==c)&(st==s)]]); print('complete/missing',[(m.sum(),e[m].mean(),r[m].mean()) for m in (~missing,missing)]); print('twin count',len(i),'twin MAE',e[np.r_[i,j]].mean() if len(i) else None,'twin contrast MAE',np.abs((o['predicted_strain'][i]-o['predicted_strain'][j])-(g['te_clean'][i]-g['te_clean'][j])).mean() if len(i) else None); print('twins missing',missing[i].sum(),missing[j].sum())
PY
```
- **Container Exit Code**: `0` (`906 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/hereditary_prony_source_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
runtime 0.1249207800719887 tau [0.60055216 4.00345244] theta [ 0.68056581  0.54693984  0.08823557  0.16208076 -0.13190986  0.10977917
 -0.08953777] tau relative error [0.00092027 0.00086311]
overall 0.05736184475430716 0.018939661932113265 by horizon [0.00378928 0.00852156 0.01512623 0.0235929  0.0348509  0.04887846
 0.06751481 0.09166809 0.12132673 0.15834948]
lab,stratum,count,strain,rate [(0, 0, 12, 0.07524, 0.02459), (0, 1, 5, 0.02178, 0.00732), (0, 2, 5, 0.11368, 0.03626), (1, 0, 5, 0.05944, 0.01923), (1, 1, 13, 0.17283, 0.05636), (1, 2, 5, 0.07018, 0.02334), (2, 0, 5, 0.02275, 0.00739), (2, 1, 5, 0.00655, 0.00302), (2, 2, 13, 0.00921, 0.00411), (3, 0, 11, 0.02629, 0.00833), (3, 1, 6, 0.01208, 0.00456), (3, 2, 5, 0.01192, 0.00433)]
complete/missing [(82, 0.02752627128436577, 0.009214646284370638), (8, 0.36317647282120635, 0.11862107232147517)]
twin count 8 twin MAE 0.03440677691729771 twin contrast MAE 0.06205773294964575
twins missing 1 0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4033fc4a57c48b7d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_4033fc4a (directory).
```

### Turn 4 — `execute_in_remote_docker` (`41980 ms` LLM)
- **Rationale**: The public errors are concentrated in missing-feature rows even at the true parameters, so further optimizer tuning is unlikely to fix that slice under the prescribed imputation. An independent forward simulation on the public histories checks whether tighter or near-boundary spectra and altered step durations cause recovery failures; if they do, the nonlinear fit needs revision before submission.
- **Alternatives Weighed**: `Use test rate observations to correct strain forecasts` (rejected: That would replace the forecast from the required shared spectrum and coupon model with a correction based on a test outcome.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/hereditary_prony_source_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
p = pickle.load(open('public_src01_datasets.pkl', 'rb'))[0]
base = p['dataset']
train, test = (base[k] for k in ('train', 'test'))
n = len(train['lab_id'])
lab = np.r_[train['lab_id'], test['lab_id']]
st = np.r_[train['stratum'], test['stratum']]
refs = np.r_[train['ref_coupons'], test['ref_coupons']]
x = np.r_[train['coupon_features_raw'], test['coupon_features_raw']]
for c in range(4):
    m = lab == c
    x[m] = (x[m] - np.median(np.nanmean(refs[m], axis=0))) / np.median(np.nanstd(refs[m], axis=0))
w = np.array([1 / np.count_nonzero((lab == c) & (st == s)) for c, s in zip(lab, st)])
for j in range(x.shape[1]):
    ok = np.isfinite(x[:, j])
    x[~ok, j] = np.average(x[ok, j], weights=w[ok])
theta = p['gt']['theta']
twins = np.triu((test['stress_hold_0'][:, None] == test['stress_hold_0']) & (test['strain_0'][:, None] == test['strain_0']), 1)
i, j = np.where(twins)
def simulate(split, coupon, dt, t, tau):
    z = np.zeros((len(coupon), 2))
    for step, duration in enumerate(dt):
        decay = np.exp(-duration / tau)
        z = decay * z + (1 - decay) * split['past_stress'][:, step, None]
    b = (split['stress_hold_0'][:, None] - z) * theta[:2]
    drift = theta[2] + coupon @ theta[3:]
    return split['strain_0'][:, None] + b @ (1 - np.exp(-t / tau[:, None])) + drift[:, None] * t, b @ (1/tau) + drift
rng = np.random.default_rng(813)
for pair, altered in [((0.21, 7.9), True), ((0.44, 5.8), True), ((0.9, 2.3), False), ((1.25, 2.25), False)]:
    d = copy.deepcopy(base)
    tau = np.array(pair)
    dt = d['past_dt']
    if altered:
        dt[:] *= np.resize([0.45, 1.65, 0.8], dt.size)
    t = d['eval_times']
    a, b = d['train'], d['test']
    y, r = simulate(a, x[:n], dt, t, tau)
    good_y, good_r = simulate(b, x[n:], dt, t, tau)
    corrupt = rng.random(y.shape) < .08
    impulses = np.where(corrupt, np.clip(rng.standard_cauchy(y.shape), -.95, .95), 0.)
    a['strain_curves_obs'] = y + rng.normal(0, .006 + .004*t, y.shape) + impulses
    a['instant_rate_obs'] = r + rng.normal(0, .01, r.shape)
    del b['instant_rate_obs']
    out = mod.solve_hereditary_prony_source(d)
    pred = out['predicted_strain']
    e = np.abs(pred - good_y)
    er = np.abs(out['initial_rate_pred'] - good_r)
    rel = np.max(np.abs(out['estimated_taus']/tau-1))
    contrast = np.abs((pred[i]-pred[j])-(good_y[i]-good_y[j])).mean()
    cell = max(e[(b['lab_id']==c)&(b['stratum']==s)].mean() for c,s in zip(b['lab_id'],b['stratum']))
    print(f'{pair}: recovered={out["estimated_taus"].round(5)} max tau rel={rel:.5f} strain={e.mean():.5f} rate={er.mean():.5f} twin contrast={contrast:.5f} worst cell={cell:.5f}', flush=True)
    assert rel <= .10 and e.mean() <= .065 and contrast <= .065 and er.mean() <= .040
PY
```
- **Container Exit Code**: `0` (`1014 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
(0.21, 7.9): recovered=[0.20993 7.9954 ] max tau rel=0.01208 strain=0.00142 rate=0.00087 twin contrast=0.00204 worst cell=0.00203
(0.44, 5.8): recovered=[0.44009 5.84966] max tau rel=0.00856 strain=0.00108 rate=0.00049 twin contrast=0.00079 worst cell=0.00144
(0.9, 2.3): recovered=[0.90373 2.3123 ] max tau rel=0.00535 strain=0.00195 rate=0.00103 twin contrast=0.00124 worst cell=0.00357
(1.25, 2.25): recovered=[1.24803 2.26465] max tau rel=0.00651 strain=0.00204 rate=0.00061 twin contrast=0.00345 worst cell=0.00370

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4033fc4a57c48b7d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_4033fc4a (directory).
```

### Turn 5 — `submit_completed_task` (`14758 ms` LLM)
- **Rationale**: The exact saved artifact passed isolated-process schema and stress checks. The public aggregate thresholds passed, and the missing-coupon error persisted even at the true parameters under the required weighted-mean imputation, so further optimizer tuning is not warranted.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`13` turns remaining; artifact status={'/holdout_task/hereditary_prony_source_solver.py': 'FILE:4209'}): Do NOT lock prematurely with `13` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/hereditary_prony_source_solver.py`. Public validation passed: maximum relative tau error 0.092%, strain MAE 0.05736, twin-pair MAE 0.03441, and initial-rate MAE 0.01894. All four spectrum stress tests passed. The main limitation is missing coupon features: strain MAE on those public rows was 0.36318.`

### Turn 6 — `execute_in_remote_docker` (`37712 ms` LLM)
- **Rationale**: Leave one lab's outcomes out of each fit while retaining its unlabeled reference panel as the contract allows. Compare training-weight variants without changing prescribed imputation or the artifact. A variant merits promotion only if it reduces held-lab error without spending the narrow public twin margin.
- **Alternatives Weighed**: `Add rational drift terms or relay modes` (rejected: The specified ODE has neither mechanism. The strain-minus-rate-times-time contrast tests the relevant algebraic separation of coupon drift and hereditary memory.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/hereditary_prony_source_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = pickle.load(open('public_src01_datasets.pkl', 'rb'))[0]
d, truth = p['dataset'], p['gt']
a, b = d['train'], d['test']
orig_panel = m._panel

def panel_variant(mode):
    def panel(train, test):
        x, w = orig_panel(train, test)
        if mode == 'unweighted_fit':
            w = np.ones_like(w)
        elif mode == 'downweight_imputed_fit':
            incomplete = np.isnan(np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))).any(axis=1)
            w = w * np.where(incomplete, .25, 1.)
        return x, w
    return panel

for mode in ('baseline', 'unweighted_fit', 'downweight_imputed_fit'):
    m._panel = panel_variant(mode)
    fold = []
    for held in np.unique(a['lab_id']):
        h = a['lab_id'] == held
        train = {k: v[~h] for k, v in a.items()}
        test = {k: np.concatenate((a[k][h], v), axis=0) for k, v in b.items()}
        out = m.solve_hereditary_prony_source(dict(d, train=train, test=test))
        strain = out['predicted_strain'][:h.sum()]
        rate = out['initial_rate_pred'][:h.sum()]
        e = strain - a['strain_curves_obs'][h]
        er = rate - a['instant_rate_obs'][h]
        complete = np.isfinite(a['coupon_features_raw'][h]).all(axis=1)
        # Removing the linear drift gives a real-measurement check on the branch shape independent of coupon offsets.
        shape_e = e - er[:, None] * d['eval_times']
        fold.append([np.median(abs(e)), np.median(abs(er)), np.median(abs(er[complete])), np.median(er[complete]), np.median(abs(shape_e)), *out['estimated_taus']])
    out = m.solve_hereditary_prony_source(d)
    e = abs(out['predicted_strain'] - truth['te_clean'])
    er = abs(out['initial_rate_pred'] - truth['te_true_rate0'])
    i, j = np.where(np.triu((b['stress_hold_0'][:,None] == b['stress_hold_0']) & (b['strain_0'][:,None] == b['strain_0']), 1))
    resid = out['predicted_strain'] - truth['te_clean']
    print(mode, 'LOLO by lab [strain MedAE, rate MedAE, complete rate MedAE, complete signed median rate error, drift-cancelled strain MedAE, taus]:', np.round(fold, 5).tolist())
    print(mode, 'public [max tau relative error, strain MAE, rate MAE, twin member MAE, twin contrast MAE]:', np.round([np.max(abs(out['estimated_taus']/truth['taus']-1)), e.mean(), er.mean(), e[np.r_[i,j]].mean(), abs(resid[i]-resid[j]).mean()], 6).tolist())
m._panel = orig_panel
PY
```
- **Container Exit Code**: `0` (`2082 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
baseline LOLO by lab [strain MedAE, rate MedAE, complete rate MedAE, complete signed median rate error, drift-cancelled strain MedAE, taus]: [[0.03222, 0.01537, 0.01341, 0.00961, 0.01311, 0.60311, 4.16075], [0.02318, 0.01285, 0.01164, 0.00429, 0.01702, 0.59975, 3.96261], [0.01604, 0.00929, 0.00799, -0.00045, 0.01409, 0.59987, 3.94733], [0.0246, 0.01252, 0.01046, -0.00391, 0.0157, 0.59818, 3.8607]]
baseline public [max tau relative error, strain MAE, rate MAE, twin member MAE, twin contrast MAE]: [0.00092, 0.057362, 0.01894, 0.034407, 0.062058]
unweighted_fit LOLO by lab [strain MedAE, rate MedAE, complete rate MedAE, complete signed median rate error, drift-cancelled strain MedAE, taus]: [[0.03098, 0.01596, 0.01424, 0.00995, 0.0139, 0.60263, 4.12505], [0.02301, 0.0127, 0.01122, 0.00349, 0.01586, 0.59897, 3.93508], [0.01409, 0.00817, 0.00767, -0.00209, 0.01396, 0.59896, 3.90405], [0.02923, 0.01312, 0.01113, -0.0051, 0.01628, 0.59897, 3.92242]]
unweighted_fit public [max tau relative error, strain MAE, rate MAE, twin member MAE, twin contrast MAE]: [0.000592, 0.057204, 0.018846, 0.034879, 0.058103]
downweight_imputed_fit LOLO by lab [strain MedAE, rate MedAE, complete rate MedAE, complete signed median rate error, drift-cancelled strain MedAE, taus]: [[0.03279, 0.01534, 0.01335, 0.00991, 0.01324, 0.60319, 4.1933], [0.0221, 0.01271, 0.01138, 0.0041, 0.01758, 0.59954, 3.98637], [0.01659, 0.00985, 0.00798, 0.00045, 0.01342, 0.59986, 3.99754], [0.02465, 0.01244, 0.01021, -0.0038, 0.01537, 0.59825, 3.87765]]
downweight_imputed_fit public [max tau relative error, strain MAE, rate MAE, twin member MAE, twin contrast MAE]: [0.010686, 0.057314, 0.018733, 0.03423, 0.061202]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4033fc4a57c48b7d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_4033fc4a (directory).
```

### Turn 7 — `submit_completed_task` (`18333 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process schema and spectrum stress tests. The held-lab audit found no consistent improvement across strain and rate from reweighting, and the public true-parameter diagnostic showed that the prescribed imputation accounts for most of the missing-coupon error.
- **Alternatives Weighed**: `Use equal training weights while retaining cell-balanced imputation.` (rejected: This improved the secondary twin-contrast metric from 0.06206 to 0.05810, but slightly worsened twin-pair MAE and worst held-lab median rate error. It did not address the missing-coupon bottleneck.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/hereditary_prony_source_solver.py': 'FILE:4209'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/hereditary_prony_source_solver.py`. All public thresholds passed: maximum relative timescale error 0.092%, strain MAE 0.05736, twin-pair MAE 0.03441, and initial-rate MAE 0.01894. All four spectrum stress tests passed. The main limitation was missing coupon features: those public rows had strain MAE 0.36318 under the required imputation.`

### Turn 8 — `execute_in_remote_docker` (`33485 ms` LLM)
- **Rationale**: The held-lab audit already tested whether nuisance reweighting improves this regression; classification balanced accuracy does not apply. This final check can detect a different failure: a recurrence or derivative inconsistency that aggregate MAE would miss. If independent recurrence agrees and drift cancellation removes the missingness gap, retain the prescribed imputation and submit.
- **Alternatives Weighed**: `Add rational, diffusion, or relay terms` (rejected: The contract specifies a continuous two-branch ODE. Subtracting the initial rate times elapsed time cancels every linear drift term exactly and directly tests the relevant algebra.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/hereditary_prony_source_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('public_src01_datasets.pkl', 'rb') as f:
    records = pickle.load(f)
for row in records:
    d, g = row['dataset'], row['gt']
    a, b = d['train'], d['test']
    t, dt = d['eval_times'], d['past_dt']
    out = mod.solve_hereditary_prony_source(d)
    tau, theta = out['estimated_taus'], out['hereditary_weights']
    z = np.zeros((len(b['lab_id']), 2))
    for k, h in enumerate(dt):
        u = -np.expm1(-h/tau)
        z += u * (b['past_stress'][:, k, None] - z)
    amp = (b['stress_hold_0'][:, None] - z) * theta[:2]
    physical = amp @ (-np.expm1(-t[:, None]/tau) - t[:, None]/tau).T
    observed = out['predicted_strain'] - b['strain_0'][:, None] - out['initial_rate_pred'][:, None] * t
    np.testing.assert_allclose(observed, physical, atol=1e-12, rtol=1e-12)
    e = out['predicted_strain'] - g['te_clean']
    er = out['initial_rate_pred'] - g['te_true_rate0']
    shape = e - er[:, None]*t
    i, j = np.where(np.triu((b['stress_hold_0'][:, None] == b['stress_hold_0']) & (b['strain_0'][:, None] == b['strain_0']), 1))
    assert i.size and np.all(np.any(b['past_stress'][i] != b['past_stress'][j], axis=1))
    missing = np.isnan(b['coupon_features_raw']).any(axis=1)
    for name, mask in [('complete', ~missing), ('imputed', missing)]:
        print(name, 'count', mask.sum(), 'strain MAE', np.abs(e[mask]).mean(), 'drift-cancelled MAE', np.abs(shape[mask]).mean())
    metrics = np.array([np.max(np.abs(tau/g['taus']-1)), np.abs(e).mean(), np.abs(e[np.r_[i,j]]).mean(), np.abs(e[i]-e[j]).mean(), np.abs(er).mean()])
    print('tau relative error, strain MAE, twin member MAE, twin contrast MAE, rate MAE:', metrics)
    print('twin drift-cancelled contrast MAE', np.abs(shape[i]-shape[j]).mean(), 'independent recurrence identity max error', np.max(np.abs(observed-physical)))
    assert np.all(metrics <= [.10,.065,.065,.065,.040])
PY
```
- **Container Exit Code**: `0` (`906 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
complete count 82 strain MAE 0.02752627128436577 drift-cancelled MAE 0.0038492166469261506
imputed count 8 strain MAE 0.36317647282120635 drift-cancelled MAE 0.003471962678336757
tau relative error, strain MAE, twin member MAE, twin contrast MAE, rate MAE: [0.00092027 0.05736184 0.03440678 0.06205773 0.01893966]
twin drift-cancelled contrast MAE 0.0228496079412667 independent recurrence identity max error 7.105427357601002e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4033fc4a57c48b7d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_4033fc4a (directory).
```

### Turn 9 — `submit_completed_task` (`17982 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process validation, and the independent recurrence agreed with the forecast and derivative to within 7.11e-15. The held-lab audit found no consistent benefit from reweighting, and drift cancellation isolates the remaining error to the coupon term. Changing the prescribed imputation is outside the contract.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/hereditary_prony_source_solver.py`. All public thresholds passed: maximum relative timescale error 0.092%, strain MAE 0.05736, twin-pair MAE 0.03441, and initial-rate MAE 0.01894. All four spectrum stress tests passed. The main limitation was missing coupon features: strain MAE on those rows was 0.36318 under the required imputation.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M93-Lifecycle",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/hereditary_prony_source_solver.py": "FILE:4209"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "max_tau_rel_err": 0.031454,
    "max_strain_mae": 0.055167,
    "max_twin_mae": 0.02891,
    "max_rate_mae": 0.018355,
    "student_elapsed_sec": 0.2784,
    "instance_diagnostics": [
      {
        "seed": 99501,
        "tau_rel": 0.031454,
        "strain_mae": 0.055167,
        "twin_mae": 0.02891,
        "rate_mae": 0.018355
      },
      {
        "seed": 99502,
        "tau_rel": 0.000558,
        "strain_mae": 0.035442,
        "twin_mae": 0.008203,
        "rate_mae": 0.011535
      }
    ],
    "candidate_timing_breakdown": {
      "import_wall_sec": 0.085759,
      "import_cpu_sec": 0.079583,
      "candidate_call_wall_sec": 0.231634,
      "candidate_call_cpu_sec": 0.214988,
      "ipc_and_codec_wall_sec": 0.121664,
      "rpc_call_count": 2,
      "verifier_and_gt_wall_sec": 0.007886,
      "verifier_and_gt_cpu_sec": 0.030126,
      "total_supervisor_wall_sec": 0.446943
    }
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"max_tau_rel_err\": 0.031454, \"max_strain_mae\": 0.055167, \"max_twin_mae\": 0.02891, \"max_rate_mae\": 0.018355, \"student_elapsed_sec\": 0.2784, \"instance_diagnostics\": [{\"seed\": 99501, \"tau_rel\": 0.031454, \"strain_mae\": 0.055167, \"twin_mae\": 0.02891, \"rate_mae\": 0.018355}, {\"seed\": 99502, \"tau_rel\": 0.000558, \"strain_mae\": 0.035442, \"twin_mae\": 0.008203, \"rate_mae\": 0.011535}]}\n__HEGEL_VERIFIER_AUTH_d5e87c7b002979d8803a42465a48cbcb__:{\"strict_pass\": true, \"reward\": 1.0, \"max_tau_rel_err\": 0.031454, \"max_strain_mae\": 0.055167, \"max_twin_mae\": 0.02891, \"max_rate_mae\": 0.018355, \"student_elapsed_sec\": 0.2784, \"instance_diagnostics\": [{\"seed\": 99501, \"tau_rel\": 0.031454, \"strain_mae\": 0.055167, \"twin_mae\": 0.02891, \"rate_mae\": 0.018355}, {\"seed\": 99502, \"tau_rel\": 0.000558, \"strain_mae\": 0.035442, \"twin_mae\": 0.008203, \"rate_mae\": 0.011535}], \"candidate_timing_breakdown\": {\"import_wall_sec\": 0.085759, \"import_cpu_sec\": 0.079583, \"candidate_call_wall_sec\": 0.231634, \"candidate_call_cpu_sec\": 0.214988, \"ipc_and_codec_wall_sec\": 0.121664, \"rpc_call_count\": 2, \"verifier_and_gt_wall_sec\": 0.007886, \"verifier_and_gt_cpu_sec\": 0.030126, \"total_supervisor_wall_sec\": 0.446943}}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 4946.02,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
