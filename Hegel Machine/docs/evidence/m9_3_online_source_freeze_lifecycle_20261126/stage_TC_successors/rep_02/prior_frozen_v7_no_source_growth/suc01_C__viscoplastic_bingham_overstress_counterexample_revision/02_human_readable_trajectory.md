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
| **Turns Used / Max Turns** | `11 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `289,301` (`276,792` / `58,624` / `12,509`) |
| **Total Episode Wall Time (s)** | `440.96s` |

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

### Turn 1 — `execute_in_remote_docker` (`18500 ms` LLM)
- **Rationale**: Inspect the public data and prior-task schema for the exact history convention, alignment fields, and available labels before implementing the model; mismatched dimensions or missing calibration fields would change the implementation.

```bash
ls -la /holdout_task /pretrain_task; cat /holdout_task/validate_public_schema.py; python3 /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'viscoelastic relaxation reference-panel alignment cell-balanced imputation Cauchy robust IRLS'; python3 - <<'PY'
import pickle, numpy as np
p=pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl','rb'))
def show(x,d=0):
 if d>5:return
 if isinstance(x,dict):
  for k,v in x.items():
   print(' '*d,k,end=': '); show(v,d+1)
 elif isinstance(x,(list,tuple)):
  print(type(x),len(x));
  if len(x):show(x[0],d+1)
 elif isinstance(x,np.ndarray):
  print(x.shape,x.dtype,x.ravel()[:8], ('nan',np.isnan(x).sum()) if x.dtype.kind=='f' else '')
 else:print(repr(x)[:500])
show(p)
PY
find /pretrain_task -maxdepth 2 -type f -printf '%p %s\n' | head -40
```
- **Container Exit Code**: `0` (`976 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-1.426      -1.17049905 -0.72786527 -0.74905622
  1.27886419  1.426     ] ('nan', 0)
    strain_0: (148,) float64 [-0.53882384  0.21913343  0.49137244 -0.91798528 -0.29262777  0.21590306
  0.36369065 -0.09243466] ('nan', 0)
    stress_hold_0: (148,) float64 [-1.426       2.68460746  1.426      -2.4983265  -1.426       2.57620905
  1.426      -2.02603729] ('nan', 0)
    instant_rate_obs: (148,) float64 [-0.84442339  6.95311845  0.52196025 -1.37230854 -1.86280658  6.94843569
  1.06179546 -6.25836111] ('nan', 0)
    strain_curves_obs: (148, 10) float64 [-0.69433845 -0.88219714 -1.11568464 -1.43312041 -1.82003372 -2.26177321
 -2.86226285 -3.58188927] ('nan', 0)
   test:     lab_id: (90,) int64 [2 3 1 1 0 3 3 1] 
    stratum: (90,) int64 [0 0 1 2 0 2 0 2] 
    coupon_features_raw: (90, 4) float64 [ 2.13680554  4.83227042  2.80032932  3.40141876 -4.18727962 -2.23466216
 -2.74478594 -2.06179318] ('nan', 9)
    ref_coupons: (90, 20) float64 [0.70260801 5.84584835 4.63223518 4.61166442 4.12286171 3.7757354
 4.05167297 2.93688024] ('nan', 72)
    past_stress: (90, 28) float64 [ 1.426       1.426       1.426       1.21028511  0.62791057 -0.70784951
 -1.426      -1.426     ] ('nan', 0)
    strain_0: (90,) float64 [-0.16391183  0.97618752  0.34255033  0.9853829   0.08622585  0.07444828
 -0.43108293  0.90726158] ('nan', 0)
    stress_hold_0: (90,) float64 [-1.426       2.5857758   1.0508914   2.67838064  1.426       1.96184454
 -1.426       2.20051003] ('nan', 0)
    instant_rate_obs: (90,) float64 [-3.11361369  1.31699513  0.52964761  2.54490308  2.77331664  6.06654478
 -1.28305143  2.02014309] ('nan', 0)
   yield_stress: 1.55
  gt:    taus: (2,) float64 [0.58 3.9 ] ('nan', 0)
   theta: (7,) float64 [ 0.68  0.54  0.08  0.15 -0.12  0.11 -0.09] ('nan', 0)
   vp_theta: (2,) float64 [0.24 0.16] ('nan', 0)
   te_Z: (90, 4) float64 [-1.03772657  0.87395065 -0.56714233 -0.14083776 -1.51444661  0.7827504
  0.18260478  0.98612567] ('nan', 0)
   te_clean: (90, 10) float64 [-0.70352378 -1.19422652 -1.66087148 -2.04766055 -2.38597107 -2.6935991
 -3.03270408 -3.42641005] ('nan', 0)
   te_true_rate0: (90,) float64 [-3.11614886  1.32986451  0.51334633  2.5517431   2.76600838  6.083801
 -1.28322401  2.0260598 ] ('nan', 0)
   te_cross: (90,) float64 [8.5        0.39342466 0.53255253 0.22262001 0.13013045 0.09656052
 8.5        0.2639237 ] ('nan', 0)
   te_secant: (90,) float64 [-2.96721262  0.44112716  3.0069437   1.97937193 -0.55350513  3.30400206
 -2.2654106   2.86086919] ('nan', 0)
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

### Turn 2 — `execute_in_remote_docker` (`44728 ms` LLM)
- **Rationale**: Use the public clean curves only to diagnose the omitted coupon term and history convention. If the two exponentials and linear drift do not reconstruct those curves, revise the basis before fitting the noisy training data.
- **Alternatives Weighed**: `Fit each time point independently` (rejected: It would not enforce the shared relaxation times or recover the two plastic weights.)

```bash
export PYTHONPATH=/tmp; python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel; python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights; python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_affine_projection.robust_residual_weights; python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation; python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl','rb'))[0];d,g=p['dataset'],p['gt']; q=d['test'];s=q['stress_hold_0'];h=q['past_stress'];dt=d['past_dt'];t=d['eval_times'];u=g['taus'];x=g['te_Z'];y=d['yield_stress'];o=np.sign(s)*np.maximum(abs(s)-y,0)**2;w=np.maximum(abs(h)-y,0)**2@dt*s
F=-np.expm1(-t[:,None]/u);A=np.c_[F,t];b=np.linalg.lstsq(A,(g['te_clean']-q['strain_0'][:,None]).T,rcond=None)[0].T;print('times',t,'decomposition RMS',np.sqrt(np.mean(((b@A.T)-(g['te_clean']-q['strain_0'][:,None]))**2)));print('rate RMS',np.sqrt(np.mean((b@np.r_[1/u,1]-g['te_true_rate0'])**2)))
z=np.zeros((len(s),2));
for j,step in enumerate(dt):z=z*np.exp(-step/u)+h[:,j,None]*(-np.expm1(-step/u))
print('basis coefficient samples: [observed, branch+plastic, slope plastic]',np.c_[b, (s[:,None]-z)*g['theta'][:2]+np.c_[w*g['vp_theta'][1], np.zeros_like(w)],o*g['vp_theta'][0]][:10]);
base=np.c_[s,z,s[:,None]-z,o,w,x,x*s[:,None]]; names=['s','z1','z2','s-z1','s-z2','ov','work*s','x1','x2','x3','x4','s*x1','s*x2','s*x3','s*x4']; v=np.c_[b[:,0]-.68*(s-z[:,0])-.16*w,b[:,1]-.54*(s-z[:,1]),b[:,2]-.24*o];
for k in range(3):
 ids=[0,1,2,5,6,7,8,9,10,11,12,13,14];M=np.c_[np.ones(len(s)),base[:,ids]];c=np.linalg.lstsq(M,v[:,k],rcond=None)[0];print('remainder',k,'rms',np.std(M@c-v[:,k]),'coef',list(zip(['one']+[names[i] for i in ids],c.round(5))))
for k in range(4):
 j=q['lab_id']==k;print('lab',k,'cells',np.bincount(q['stratum'][j]),'raw mean/std',np.nanmean(q['coupon_features_raw'][j],0).round(2),np.nanstd(q['coupon_features_raw'][j],0).round(2),'reference mean/std',np.nanmean(q['ref_coupons'][j]).round(2),np.nanstd(q['ref_coupons'][j]).round(2),'latent mean/std',x[j].mean(0).round(2),x[j].std(0).round(2))
PY
```
- **Container Exit Code**: `0` (`709 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ormhole_skills.autogenous_prim_robust_trend_decomposition",
      "reference_source_txt": null
    }
  ]
}
times [0.2  0.45 0.8  1.25 1.85 2.6  3.6  4.9  6.5  8.5 ] decomposition RMS 1.2627635453739816e-15
rate RMS 3.4745280527003084e-15
basis coefficient samples: [observed, branch+plastic, slope plastic] [[-1.55287521 -0.81328452 -0.23024332 -1.55287521 -0.81328452 -0.        ]
 [ 0.64381537  1.06126908 -0.05228226  0.64381537  1.06126908  0.25747956]
 [ 0.03190008  0.59405793  0.30602364  0.03190008  0.59405793  0.        ]
 [ 1.11558966  0.89335234  0.39924796  1.11558966  0.89335234  0.30557829]
 [ 1.69288628  0.91577202 -0.38757441  1.69288628  0.91577202  0.        ]
 [ 3.16566781  1.54931896  0.22849179  3.16566781  1.54931896  0.04070782]
 [-0.49614711 -0.69779001 -0.24887745 -0.49614711 -0.69779001 -0.        ]
 [ 0.73682248  0.66896763  0.58414606  0.73682248  0.66896763  0.10155919]
 [ 1.31176535  0.85055891  0.71531631  1.31176535  0.85055891  0.        ]
 [ 1.61769519  1.01684732 -0.16217814  1.61769519  1.01684732  0.18839054]]
remainder 0 rms 7.380788516406272e-16 coef [('one', -0.0), ('s', -0.0), ('z1', 0.0), ('z2', 0.0), ('ov', -0.0), ('work*s', -0.0), ('x1', -0.0), ('x2', 0.0), ('x3', -0.0), ('x4', 0.0), ('s*x1', 0.0), ('s*x2', 0.0), ('s*x3', -0.0), ('s*x4', 0.0)]
remainder 1 rms 2.365387610333057e-15 coef [('one', 0.0), ('s', 0.0), ('z1', -0.0), ('z2', -0.0), ('ov', 0.0), ('work*s', 0.0), ('x1', 0.0), ('x2', -0.0), ('x3', -0.0), ('x4', -0.0), ('s*x1', -0.0), ('s*x2', -0.0), ('s*x3', 0.0), ('s*x4', -0.0)]
remainder 2 rms 3.7011531421005973e-16 coef [('one', 0.08), ('s', 0.0), ('z1', 0.0), ('z2', 0.0), ('ov', -0.0), ('work*s', 0.0), ('x1', 0.15), ('x2', -0.12), ('x3', 0.11), ('x4', -0.09), ('s*x1', -0.0), ('s*x2', -0.0), ('s*x3', -0.0), ('s*x4', -0.0)]
lab 0 cells [12  5  5] raw mean/std [3.13 3.15 3.27 3.  ] [1.64 0.94 1.05 1.01] reference mean/std 3.16 1.4 latent mean/std [ 0.02  0.03  0.15 -0.02] [1.24 0.72 0.79 0.79]
lab 1 cells [ 5 13  5] raw mean/std [-3.24 -3.64 -3.12 -3.29] [0.74 0.61 0.57 0.77] reference mean/std -3.41 0.79 latent mean/std [ 0.21 -0.34  0.33  0.14] [0.92 0.77 0.72 0.97]
lab 2 cells [ 5  5 13] raw mean/std [4.67 2.7  4.31 2.97] [1.26 1.01 0.92 1.19] reference mean/std 3.65 1.33 latent mean/std [ 0.76 -0.64  0.51 -0.4 ] [0.89 0.72 0.65 0.86]
lab 3 cells [11  6  5] raw mean/std [-3.07 -2.49 -3.01 -2.95] [0.88 0.82 0.83 0.65] reference mean/std -2.9 0.87 latent mean/std [-0.18  0.41 -0.1  -0.06] [1.01 1.   0.96 0.76]
```

### Turn 3 — `execute_in_remote_docker` (`77131 ms` LLM)
- **Rationale**: The public decomposition identifies two branch amplitudes and a linear drift. Fit the branches robustly while profiling out each specimen’s drift, then use the aligned coupons to separate hold overstress from ordinary drift. Test the exact artifact on clean public labels and separate hold-only, history-only, and sub-yield slices to detect localized model error.
- **Alternatives Weighed**: `Predict drift from coupon features alone` (rejected: The observed onset rate identifies drift directly once both transient derivatives are subtracted; relying only on coupons would expose predictions to lab calibration and missing-feature errors.)

```bash
cat > /holdout_task/viscoplastic_overstress_hereditary_solver.py <<'PY'
"""Two-branch hereditary response with signed overstress and work softening."""

import numpy as np
from scipy.optimize import minimize


def _aligned_coupons(train, test):
    """Calibrate shared lab transforms on the reference panel, then fill by cell balance."""
    lab = np.concatenate([train['lab_id'], test['lab_id']])
    stratum = np.concatenate([train['stratum'], test['stratum']])
    x = np.vstack([train['coupon_features_raw'], test['coupon_features_raw']])
    ref = np.vstack([train['ref_coupons'], test['ref_coupons']])
    for group in np.unique(lab):
        mask = lab == group
        panel = ref[mask]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0))
        x[mask] = (x[mask] - center) / scale
    n = len(train['lab_id'])
    _, cell, count = np.unique(np.column_stack((lab[:n], stratum[:n])), axis=0,
                               return_inverse=True, return_counts=True)
    weights = (1.0 / count[cell])[:, None]
    observed = np.isfinite(x[:n])
    means = np.sum(weights * np.where(observed, x[:n], 0), axis=0) / np.sum(weights * observed, axis=0)
    x = np.where(np.isfinite(x), x, means)
    return x[:n], x[n:]


def _history(split, dt, threshold):
    h = np.asarray(split['past_stress'], dtype=float)
    s = np.asarray(split['stress_hold_0'], dtype=float)
    ov = np.sign(s) * np.maximum(np.abs(s) - threshold, 0.0)**2
    work = (np.maximum(np.abs(h) - threshold, 0.0)**2 @ dt) * s
    return h, s, ov, work


def _branches(history, dt, t, tau):
    h, s, _, work = history
    decay = np.exp(-dt[:, None] / tau)
    arrival = -np.expm1(-dt[:, None] / tau)
    z = np.zeros((len(s), 2))
    for j in range(len(dt)):
        z = z * decay[j] + h[:, j, None] * arrival[j]
    a = np.column_stack((s[:, None] - z, work))
    phi = -np.expm1(-t[:, None] / tau)
    phi = np.vstack((phi, 1.0 / tau))
    return a[:, None, :] * phi[None, :, [0, 1, 0]]


def _project(A, y, b, w):
    """Weighted least squares with a separate linear drift for every sample."""
    wb = w * b
    norm = wb @ b
    a_slopes = np.einsum('nkp,nk->np', A, wb) / norm[:, None]
    y_slopes = np.sum(y * wb, axis=1) / norm
    Ap = A - b[None, :, None] * a_slopes[:, None, :]
    yp = y - b * y_slopes[:, None]
    gram = np.einsum('nki,nkj,nk->ij', Ap, Ap, w)
    rhs = np.einsum('nki,nk,nk->i', Ap, yp, w)
    coeff = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    drift = y_slopes - a_slopes @ coeff
    return coeff, drift, yp - Ap @ coeff


def _fit_memory(history, dt, t, train):
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs']))
    scale = np.r_[np.full(len(t), .026), .016]
    b = np.r_[t, 1.] / scale
    y = y / scale

    def profile(log_tau, full=False):
        A = _branches(history, dt, t, np.exp(log_tau)) / scale[None, :, None]
        w = np.ones_like(y)
        for _ in range(24):
            coeff, drift, resid = _project(A, y, b, w)
            new_w = 1. / (1. + resid**2)
            if np.max(np.abs(w - new_w)) < 1e-5:
                break
            w = new_w
        if full:
            return coeff, drift
        return np.mean(np.log1p(resid**2))

    # Enforce branch ordering while allowing the complete physical time range.
    bounds = [(np.log(.2), np.log(8.0))] * 2
    ordered = {'type': 'ineq', 'fun': lambda u: u[1] - u[0] - 1e-6}
    fits = [minimize(profile, np.log(start), method='SLSQP', bounds=bounds,
                     constraints=ordered, options={'ftol': 1e-10, 'maxiter': 70})
            for start in ((.45, 3.0), (1.0, 5.5))]
    log_tau = min(fits, key=lambda f: f.fun).x
    coeff, drift = profile(log_tau, full=True)
    return np.exp(log_tau), coeff, drift


def _fit_drift(x, ov, slopes):
    A = np.column_stack((np.ones(len(x)), x, ov))
    w = np.ones(len(x))
    for step in range(20):
        root = np.sqrt(w)
        coef = np.linalg.lstsq(A * root[:, None], slopes * root, rcond=None)[0]
        r = slopes - A @ coef
        if step == 0:
            scale = max(.014, 1.4826 * np.median(np.abs(r - np.median(r))))
        w = 1. / (1. + (r / (2 * scale))**2)
    return coef, scale


def solve_viscoplastic_hereditary_revision(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt = np.asarray(dataset['past_dt'], dtype=float)
    t = np.asarray(dataset['eval_times'], dtype=float)
    yld = float(dataset['yield_stress'])
    train_h, test_h = (_history(split, dt, yld) for split in (train, test))
    train_x, test_x = _aligned_coupons(train, test)
    tau, coeff, slopes = _fit_memory(train_h, dt, t, train)
    drift_coeff, scatter = _fit_drift(train_x, train_h[2], slopes)
    response = _branches(test_h, dt, t, tau) @ coeff
    coupon_drift = np.column_stack((np.ones(len(test_x)), test_x, test_h[2])) @ drift_coeff
    rate_drift = test['instant_rate_obs'] - response[:, -1]
    # The observed onset rate directly identifies drift after subtracting the
    # Prony and damage derivatives; coupons supply a noisy independent estimate.
    correction = coupon_drift - rate_drift
    fraction = .010**2 / (.010**2 + scatter**2)
    drift = rate_drift + fraction * np.clip(correction, -2.5*scatter, 2.5*scatter)
    return {'estimated_taus': tau,
            'viscoplastic_weights': np.array([drift_coeff[-1], coeff[-1]]),
            'predicted_strain': test['strain_0'][:, None] + response[:, :-1] + drift[:, None]*t}
PY
cat > /tmp/verify_vp.py <<'PY'
import importlib.util, pickle, numpy as np, time
f='/holdout_task/viscoplastic_overstress_hereditary_solver.py'
spec=importlib.util.spec_from_file_location('candidate',f); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for p in pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl','rb')):
 d,g=p['dataset'],p['gt']; start=time.perf_counter(); q=m.solve_viscoplastic_hereditary_revision(d); print('seconds',time.perf_counter()-start)
 e=np.abs(q['predicted_strain']-g['te_clean']); a=d['test']; ov=np.abs(a['stress_hold_0'])>d['yield_stress']; work=np.any(np.abs(a['past_stress'])>d['yield_stress'],axis=1)
 print('tau',q['estimated_taus'],'relative',abs(q['estimated_taus']/g['taus']-1),'vp',q['viscoplastic_weights'],'relative L2',np.linalg.norm(q['viscoplastic_weights']-g['vp_theta'])/np.linalg.norm(g['vp_theta']))
 print('MAE',e.mean(),{(int(i),int(j)):(int(k.sum()),float(e[k].mean())) for i in [False,True] for j in [False,True] if (k:=(ov==i)&(work==j)).any()}); print('sub',e[~(ov|work)].mean(),'above',e[ov|work].mean()); print('lab/stratum MAE',[(i,j,round(e[k].mean(),5)) for i in range(4) for j in range(3) if (k:=(a['lab_id']==i)&(a['stratum']==j)).any()]); print('clean rate noise RMS',np.std(a['instant_rate_obs']-g['te_true_rate0']))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_vp.py
```
- **Container Exit Code**: `0` (`1212 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoplastic_overstress_hereditary_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
seconds 0.24165732902474701
tau [0.57939927 3.83150679] relative [0.00103574 0.01756236] vp [0.24259594 0.15959664] relative L2 0.009107805907685505
MAE 0.020858833641032002 {(0, 0): (53, 0.018999288749643187), (1, 1): (37, 0.023522506053021402)}
sub 0.018999288749643187 above 0.023522506053021402
lab/stratum MAE [(0, 0, 0.02764), (0, 1, 0.02595), (0, 2, 0.01643), (1, 0, 0.00691), (1, 1, 0.02132), (1, 2, 0.02869), (2, 0, 0.01434), (2, 1, 0.02426), (2, 2, 0.01459), (3, 0, 0.02147), (3, 1, 0.02638), (3, 2, 0.02026)]
clean rate noise RMS 0.009766502205689571

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=632d7574e9eb560e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_632d7574 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`39661 ms` LLM)
- **Rationale**: The public data combine hold and history activation. Check hold-only and history-only trajectories independently, with changed relaxation times and contaminated training curves. A failing slice would reveal which mechanism or parameter fit needs revision.

```bash
cat > /tmp/stress_vp.py <<'PY'
import importlib.util
import pickle
import numpy as np

path = '/holdout_task/viscoplastic_overstress_hereditary_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl', 'rb'))[0]

def check(d, clean, tau, vp, label):
    out = m.solve_viscoplastic_hereditary_revision(d)
    e = np.abs(out['predicted_strain'] - clean)
    q, y = d['test'], d['yield_stress']
    hold = np.abs(q['stress_hold_0']) > y
    hist = np.any(np.abs(q['past_stress']) > y, axis=1)
    terr = np.max(np.abs(out['estimated_taus'] / tau - 1))
    verr = np.linalg.norm(out['viscoplastic_weights'] - vp) / np.linalg.norm(vp)
    slices = {(a,b): round(float(e[k].mean()), 6) for a in (0,1) for b in (0,1) if (k := ((hold == a) & (hist == b))).any()}
    print(label, 'tau error', round(terr,6), 'weight error', round(verr,6), 'MAE', round(e.mean(),6), 'hold/history MAE', slices, 'late-time MAE', round(e[:,-1].mean(),6), flush=True)
    assert np.isfinite(e).all() and .20 <= out['estimated_taus'][0] < out['estimated_taus'][1] <= 8.
    assert terr <= .10 and verr <= .14 and e.mean() <= .070
    assert e[~(hold|hist)].mean() <= .065 and all(v <= .080 for k,v in slices.items() if k != (0,0))

check(p['dataset'], p['gt']['te_clean'], p['gt']['taus'], p['gt']['vp_theta'], 'public')
for seed, tau, vp in [(19, [.27, 7.2], [.18,.21]), (37, [1.05, 2.6], [.34,.11])]:
    rng = np.random.default_rng(seed)
    src = p['dataset']
    d = dict(src, train=dict(src['train']), test=dict(src['test']))
    x = m._aligned_coupons(d['train'], d['test'])
    dt, t, y = d['past_dt'], d['eval_times'], d['yield_stress']
    for mode, coupon in zip(('train', 'test'), x):
        q = d[mode]
        n = len(coupon)
        c = rng.permutation(np.arange(n) % 4)
        h = q['past_stress'].copy()
        h = np.clip(h, -.96*y, .96*y)
        h[c >= 2, :6] = rng.choice([-1.,1.], ((c>=2).sum(),6)) * rng.uniform(1.05*y, 2*y, ((c>=2).sum(),6))
        s = rng.choice([-1.,1.], n) * rng.uniform(.55*y, .97*y, n)
        s[c%2==1] *= rng.uniform(1.9, 2.05, (c%2==1).sum())
        q['past_stress'], q['stress_hold_0'] = h, s
        z = (h * (-np.expm1(-dt[:,None]/tau)).T) if False else None
        remaining = np.cumsum(dt[::-1])[::-1] - dt
        w = -np.expm1(-dt[:,None]/tau) * np.exp(-remaining[:,None]/tau)
        amp = (s[:,None]-h@w) * [0.68,0.54]
        amp[:,0] += vp[1] * (np.maximum(np.abs(h)-y,0)**2@dt)*s
        slope = .08 + coupon @ [.15,-.12,.11,-.09] + vp[0]*np.sign(s)*np.maximum(np.abs(s)-y,0)**2
        clean = q['strain_0'][:,None] + amp@(-np.expm1(-t[:,None]/tau)).T + slope[:,None]*t
        q['instant_rate_obs'] = (amp/np.array(tau)).sum(axis=1) + slope + rng.normal(0,.010,n)
        if mode == 'train':
            mask = rng.random(clean.shape) < .08
            q['strain_curves_obs'] = clean + rng.normal(0,.023,clean.shape) + mask*np.clip(.35*rng.standard_cauchy(clean.shape),-.95,.95)
    check(d, clean, np.array(tau), np.array(vp), f'stress {seed}')
PY
sed -i '/if False else None/d' /tmp/stress_vp.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/stress_vp.py
```
- **Container Exit Code**: `1` (`2061 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public tau error 0.017562 weight error 0.009108 MAE 0.020859 hold/history MAE {(0, 0): 0.018999, (1, 1): 0.023523} late-time MAE 0.057782
stress 19 tau error 0.083109 weight error 0.028217 MAE 0.013606 hold/history MAE {(0, 0): 0.01453, (0, 1): 0.011395, (1, 0): 0.018272, (1, 1): 0.009973} late-time MAE 0.037556
stress 37 tau error 0.111709 weight error 0.015749 MAE 0.015407 hold/history MAE {(0, 0): 0.016426, (0, 1): 0.014738, (1, 0): 0.015593, (1, 1): 0.014817} late-time MAE 0.043103

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=632d7574e9eb560e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_632d7574 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/stress_vp.py", line 52, in <module>
    check(d, clean, np.array(tau), np.array(vp), f'stress {seed}')
  File "/tmp/stress_vp.py", line 22, in check
    assert terr <= .10 and verr <= .14 and e.mean() <= .070
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

### Turn 5 — `execute_in_remote_docker` (`43747 ms` LLM)
- **Rationale**: The stress case misses only the parameter tolerance, not the prediction limits. Profiling out unconstrained specimen drift can discard information that separates the two relaxation branches. Test a joint fit that uses aligned coupons to constrain drift without forcing imputed or noisy coupons to match the measured curves.
- **Alternatives Weighed**: `Constrain the branch times around their public values` (rejected: That would hide the observed weak identification and exclude allowed relaxation times.)

```bash
cp /holdout_task/viscoplastic_overstress_hereditary_solver.py /tmp/vp_baseline.py
python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/vp_baseline.py'); s=p.read_text()
pos=s.index('\ndef solve_viscoplastic_hereditary_revision')
s=s[:pos]+'''
def _refine_memory(history, dt, t, train, x, tau, slopes, drift_coeff):
    """Jointly fit the hereditary branches and a robust coupon drift prior."""
    D = np.column_stack((np.ones(len(x)), x, history[2]))
    missing = ~np.isfinite(train['coupon_features_raw']).all(axis=1)
    r = (slopes - D @ drift_coeff)[~missing]
    spread = max(.008, 1.4826 * np.median(np.abs(r - np.median(r))))
    drift_scale = np.where(missing, max(.12, spread), spread)
    scale = np.r_[np.full(len(t), .026), .012]
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs'])) / scale
    b = np.r_[t, 1.0] / scale
    # The last pseudo-observation expresses drift = D @ beta + residual.
    # Project out each row's drift as in the unregularized fit.
    Y = np.column_stack((y, np.zeros(len(y))))
    B = np.column_stack((np.broadcast_to(b, y.shape), 1.0 / drift_scale))

    def profile(log_tau, full=False):
        A = np.zeros((len(y), len(b)+1, 9))
        A[:, :-1, :3] = _branches(history, dt, t, np.exp(log_tau)) / scale[None, :, None]
        A[:, -1, 3:] = -D / drift_scale[:, None]
        w = np.ones_like(Y)
        for _ in range(40):
            WB = w * B
            norm = np.sum(WB * B, axis=1)
            a = np.einsum('nki,nk->ni', A, WB) / norm[:, None]
            y0 = np.sum(Y * WB, axis=1) / norm
            Ap = A - B[:, :, None] * a[:, None, :]
            Yp = Y - B * y0[:, None]
            WA = Ap * w[:, :, None]
            gram = Ap.reshape(-1, 9).T @ WA.reshape(-1, 9)
            rhs = WA.reshape(-1, 9).T @ Yp.ravel()
            pars = np.linalg.lstsq(gram, rhs, rcond=None)[0]
            resid = Yp - Ap @ pars
            # A Huber loss for the coupon prior avoids redescending away
            # its influence; direct curve and rate measurements use Cauchy.
            w_new = 1 / (1 + resid**2)
            w_new[:, -1] = 1 / np.maximum(1, np.abs(resid[:, -1]) / 1.5)
            if np.max(np.abs(w_new - w)) < 1e-5:
                break
            w = w_new
        if full:
            return pars[:3], pars[3:], spread
        a = np.abs(resid[:, -1])
        return (np.log1p(resid[:, :-1]**2).sum() +
                np.where(a <= 1.5, a*a, 3*a-2.25).sum()) / Y.size

    fit = minimize(profile, np.log(tau), method='SLSQP',
                   bounds=[(np.log(.2), np.log(8.0))]*2,
                   constraints={'type': 'ineq', 'fun': lambda u: u[1]-u[0]-1e-6},
                   options={'ftol': 1e-10, 'maxiter': 70})
    coeff, drift_coeff, spread = profile(fit.x, full=True)
    return np.exp(fit.x), coeff, drift_coeff, spread

''' + s[pos:]
s=s.replace('    response = _branches(test_h', '    tau, coeff, drift_coeff, scatter = _refine_memory(train_h, dt, t, train, train_x, tau, slopes, drift_coeff)\n    response = _branches(test_h')
s=s.replace('    drift = rate_drift + fraction * np.clip(correction, -2.5*scatter, 2.5*scatter)', '    available = np.isfinite(test[\'coupon_features_raw\']).all(axis=1)\n    drift = rate_drift + available * fraction * np.clip(correction, -2.5*scatter, 2.5*scatter)')
Path('/tmp/vp_joint.py').write_text(s)
PY
for test in verify stress; do sed "s@/holdout_task/viscoplastic_overstress_hereditary_solver.py@/tmp/vp_joint.py@" /tmp/${test}_vp.py >/tmp/${test}_joint.py; done
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_joint.py && OPENBLAS_NUM_THREADS=1 python3 -I /tmp/stress_joint.py
```
- **Container Exit Code**: `0` (`2910 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
seconds 0.31854495010338724
tau [0.57953394 3.84025314] relative [0.00080355 0.01531971] vp [0.24169771 0.15966212] relative L2 0.006001198421265738
MAE 0.016766281844193084 {(0, 0): (53, 0.016248739513238287), (1, 1): (37, 0.01750762626420942)}
sub 0.016248739513238287 above 0.01750762626420942
lab/stratum MAE [(0, 0, 0.0248), (0, 1, 0.01627), (0, 2, 0.0201), (1, 0, 0.00604), (1, 1, 0.00975), (1, 2, 0.01414), (2, 0, 0.01655), (2, 1, 0.01468), (2, 2, 0.01363), (3, 0, 0.02624), (3, 1, 0.01538), (3, 2, 0.01754)]
clean rate noise RMS 0.009766502205689571
public tau error 0.01532 weight error 0.006001 MAE 0.016766 hold/history MAE {(0, 0): 0.016249, (1, 1): 0.017508} late-time MAE 0.046425
stress 19 tau error 0.016367 weight error 0.002455 MAE 0.008991 hold/history MAE {(0, 0): 0.009029, (0, 1): 0.007691, (1, 0): 0.012391, (1, 1): 0.006698} late-time MAE 0.024894
stress 37 tau error 0.028069 weight error 0.006316 MAE 0.010596 hold/history MAE {(0, 0): 0.011935, (0, 1): 0.011073, (1, 0): 0.009829, (1, 1): 0.009521} late-time MAE 0.029381

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=632d7574e9eb560e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_632d7574 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`16875 ms` LLM)
- **Rationale**: Promote the joint fit because it passes the previously failing parameter slice and improves public prediction. Run the schema and numerical checks in isolated Python against the exact target path; if the installed artifact fails, resolve the discrepancy before submission.
- **Alternatives Weighed**: `Keep the unconstrained specimen-drift fit` (rejected: It exceeded the 10% relaxation-time tolerance on the closer-branch stress case.)

```bash
set -e
cp /tmp/vp_joint.py /holdout_task/viscoplastic_overstress_hereditary_solver.py
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/verify_vp.py
python3 -I /tmp/stress_vp.py
```
- **Container Exit Code**: `0` (`3926 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoplastic_overstress_hereditary_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
seconds 0.3883025289978832
tau [0.57953394 3.84025314] relative [0.00080355 0.01531971] vp [0.24169771 0.15966212] relative L2 0.006001198421265738
MAE 0.016766281844193084 {(0, 0): (53, 0.016248739513238287), (1, 1): (37, 0.01750762626420942)}
sub 0.016248739513238287 above 0.01750762626420942
lab/stratum MAE [(0, 0, 0.0248), (0, 1, 0.01627), (0, 2, 0.0201), (1, 0, 0.00604), (1, 1, 0.00975), (1, 2, 0.01414), (2, 0, 0.01655), (2, 1, 0.01468), (2, 2, 0.01363), (3, 0, 0.02624), (3, 1, 0.01538), (3, 2, 0.01754)]
clean rate noise RMS 0.009766502205689571
public tau error 0.01532 weight error 0.006001 MAE 0.016766 hold/history MAE {(0, 0): 0.016249, (1, 1): 0.017508} late-time MAE 0.046425
stress 19 tau error 0.016367 weight error 0.002455 MAE 0.008991 hold/history MAE {(0, 0): 0.009029, (0, 1): 0.007691, (1, 0): 0.012391, (1, 1): 0.006698} late-time MAE 0.024894
stress 37 tau error 0.028069 weight error 0.006316 MAE 0.010596 hold/history MAE {(0, 0): 0.011935, (0, 1): 0.011073, (1, 0): 0.009829, (1, 1): 0.009521} late-time MAE 0.029381

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2d8aaf67ccf5d1fc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_2d8aaf67 (directory).
```

### Turn 7 — `submit_completed_task` (`15515 ms` LLM)
- **Rationale**: The installed solver passed isolated Python schema validation and all numerical limits on the public data and both stress cases, including separate hold-only and history-only activation slices.
- **Alternatives Weighed**: `Retain the unconstrained specimen-drift fit` (rejected: Its 11.17% relaxation-time error exceeded the 10% limit in the closer-branch stress test; the joint fit reduced it to 2.81%.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/viscoplastic_overstress_hereditary_solver.py': 'FILE:8331'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoplastic_overstress_hereditary_solver.py`. Public strain MAE was 0.01677 overall, 0.01751 above yield, and 0.01625 below yield. Public relaxation-time and viscoplastic-weight errors were 1.53% and 0.60%, respectively. Both mixed-activation stress tests passed.`

### Turn 8 — `execute_in_remote_docker` (`47636 ms` LLM)
- **Rationale**: Audit the exact variable-step history difference and drift basis on the public clean curves; a nonzero structural residual would warrant investigating missing algebraic terms. Measure the plastic-term ablation within each lab and refit without that lab’s response measurements. A tolerance breach would identify which lab calibration or physical parameter needs revision.
- **Alternatives Weighed**: `Test relay modes, stochastic drift operators, or within-lab classification accuracy` (rejected: The supported interface supplies continuous strain and stress histories, not relay states, stochastic dynamics, or class labels. The relevant nuisance check is prediction and parameter recovery when each lab is excluded from fitting.)

```bash
set -e
cat > /tmp/audit_vp.py <<'PY'
import importlib.util
import pickle
import numpy as np

path = '/holdout_task/viscoplastic_overstress_hereditary_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

for p in pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl', 'rb')):
    d, g = p['dataset'], p['gt']
    tr, te = d['train'], d['test']
    t, dt, tau, sy = d['eval_times'], d['past_dt'], g['taus'], d['yield_stress']
    h, s = te['past_stress'], te['stress_hold_0']
    hold, hist = abs(s) > sy, (abs(h) > sy).any(axis=1)
    F = -np.expm1(-t[:, None] / tau)
    lag = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    z = h @ (-np.expm1(-dt[:, None] / tau) * np.exp(-lag[:, None] / tau))
    ov = s * 0 + np.sign(s) * np.maximum(abs(s)-sy, 0.)**2
    work = s * (np.maximum(abs(h)-sy, 0.)**2 @ dt)
    creep = g['vp_theta'][0] * ov[:, None] * t
    surge = g['vp_theta'][1] * work[:, None] * F[:, 0]
    base = te['strain_0'][:, None] + ((s[:, None]-z)*g['theta'][:2]) @ F.T
    remainder = g['te_clean'] - base - creep - surge
    drift = (remainder @ t) / (t @ t)
    residual = np.max(abs(remainder - drift[:, None]*t))
    print('Clean-law maximum residual:', residual)
    assert residual < 1e-9
    print('MAE if plastic terms are omitted, with all other physical terms unchanged:')
    for lab in np.unique(te['lab_id']):
        for mode, mask in [('sub', ~(hold|hist)), ('above', hold|hist)]:
            mask = mask & (te['lab_id'] == lab)
            if mask.any():
                print(int(lab), mode, mask.sum(), round(abs((creep+surge)[mask]).mean(), 6))
    def subset(split, mask):
        return {key: val[mask] if isinstance(val, np.ndarray) and val.ndim and len(val) == len(mask) else val for key, val in split.items()}
    residuals = np.zeros_like(g['te_clean'])
    for lab in np.unique(te['lab_id']):
        mask = te['lab_id'] == lab
        q = m.solve_viscoplastic_hereditary_revision(dict(d, train=subset(tr, tr['lab_id'] != lab), test=subset(te, mask)))
        e = abs(q['predicted_strain'] - g['te_clean'][mask])
        residuals[mask] = e
        err_t = max(abs(q['estimated_taus']/tau-1))
        err_v = np.linalg.norm(q['viscoplastic_weights']-g['vp_theta'])/np.linalg.norm(g['vp_theta'])
        print('held-out lab', int(lab), 'tau error', round(err_t,6), 'weight error', round(err_v,6), 'MAE', round(e.mean(),6), 'sub/above MAE', [round(e[cohort[mask]].mean(),6) for cohort in (~(hold|hist), hold|hist) if cohort[mask].any()])
        assert err_t <= .10 and err_v <= .14 and e.mean() <= .070
        for cohort, tolerance in [(~(hold|hist), .065), (hold|hist, .080)]:
            if cohort[mask].any():
                assert e[cohort[mask]].mean() <= tolerance
    complete = np.isfinite(te['coupon_features_raw']).all(axis=1)
    print('held-out lab MAE, complete / incomplete coupons:', [round(residuals[mask].mean(), 6) for mask in (complete, ~complete) if mask.any()])
    print('held-out lab MAE by horizon:', residuals.mean(axis=0).round(6))
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/validate_public_schema.py
python3 -I /tmp/audit_vp.py
```
- **Container Exit Code**: `0` (`2101 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Clean-law maximum residual: 1.7763568394002505e-15
MAE if plastic terms are omitted, with all other physical terms unchanged:
0 sub 16 0.0
0 above 6 0.892209
1 sub 13 0.0
1 above 10 1.009617
2 sub 16 0.0
2 above 7 0.632663
3 sub 8 0.0
3 above 14 1.083139
held-out lab 0 tau error 0.006502 weight error 0.003089 MAE 0.015511 sub/above MAE [0.016292, 0.013428]
held-out lab 1 tau error 0.008461 weight error 0.00144 MAE 0.014494 sub/above MAE [0.014546, 0.014427]
held-out lab 2 tau error 0.01317 weight error 0.002154 MAE 0.041544 sub/above MAE [0.038167, 0.049265]
held-out lab 3 tau error 0.012402 weight error 0.002297 MAE 0.022206 sub/above MAE [0.017674, 0.024796]
held-out lab MAE, complete / incomplete coupons: [0.022887, 0.029421]
held-out lab MAE by horizon: [0.001575 0.003535 0.006257 0.009722 0.014302 0.020005 0.027611 0.037523
 0.049767 0.065108]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2d8aaf67ccf5d1fc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_2d8aaf67 (directory).
```

### Turn 9 — `submit_completed_task` (`25567 ms` LLM)
- **Rationale**: The final artifact passed isolated Python schema validation and all prediction and parameter tolerances on the public data, held-out lab folds, and mixed-activation stress tests.
- **Alternatives Weighed**: `Retain the unconstrained specimen-drift fit` (rejected: It exceeded the 10% relaxation-time error limit in the closer-branch stress case; the joint drift fit reduced the error from 11.17% to 2.81%.)
- **ATMS NOGOOD DECLARED**: Refuted `The exact sub-yield Prony law extends unchanged to above-yield loading.` | Witness: `Removing the two plastic terms while holding all other terms fixed changed sub-yield curves by zero but produced above-yield MAE of 0.633–1.083 across the four public labs.` | Blanket Diagnosis: `Hold overstress contributes linear creep, while squared history overstress contributes a fast transient. The omission error appears above yield within every lab.` | Sublated Pivot: `Retain both Prony branches and add the signed hold-overstress and cumulative plastic-work terms.`
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/viscoplastic_overstress_hereditary_solver.py': 'FILE:8331'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoplastic_overstress_hereditary_solver.py`. Public strain MAE was 0.01677 overall, 0.01751 above yield, and 0.01625 below yield. The largest public relaxation-time error was 1.53%, and the relative viscoplastic-weight error was 0.60%. All four held-out labs and both mixed-activation stress tests passed.`

### Turn 10 — `execute_in_remote_docker` (`34174 ms` LLM)
- **Rationale**: Refit the linear-only comparison on the actual public training measurements, keeping the onset-rate input and coupon calibration available to both models. If the refitted model closes the gap, narrow the omission claim; if the full model fails for one loading sign, inspect the signed-overstress fit before submitting.
- **Alternatives Weighed**: `Repeat the fixed-coefficient ablation or the held-out-lab audit` (rejected: Both were already verified; neither tests whether refitting lets the linear model absorb the omitted terms.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
path = '/holdout_task/viscoplastic_overstress_hereditary_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_suc01_C_datasets.pkl', 'rb') as f:
    panels = pickle.load(f)
original_history = m._history
def linear_history(*args):
    h, s, ov, work = original_history(*args)
    return h, s, np.zeros_like(ov), np.zeros_like(work)
for panel in panels:
    d, g = panel['dataset'], panel['gt']
    test, sy = d['test'], d['yield_stress']
    declared = (np.abs(test['stress_hold_0']) > sy) | np.any(np.abs(test['past_stress']) > sy, axis=1)
    real = m.solve_viscoplastic_hereditary_revision(d)
    try:
        m._history = linear_history
        ablated = m.solve_viscoplastic_hereditary_revision(d)
    finally:
        m._history = original_history
    e = np.abs(real['predicted_strain'] - g['te_clean'])
    a = np.abs(ablated['predicted_strain'] - g['te_clean'])
    print('Refitted ablation, full vs Prony only, sub-yield / any yield:', [(round(e[k].mean(), 6), round(a[k].mean(), 6)) for k in (~declared, declared)])
    for sign in (-1, 1):
        for above in (False, True):
            k = (declared == above) & (np.sign(test['stress_hold_0']) == sign)
            if k.any():
                print('hold sign', sign, 'above yield', above, 'n', k.sum(), 'full MAE', round(e[k].mean(), 6), 'ablated MAE', round(a[k].mean(), 6))
                assert e[k].mean() <= (.080 if above else .065)
PY
```
- **Container Exit Code**: `0` (`1888 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Refitted ablation, full vs Prony only, sub-yield / any yield: [(0.016249, 0.464664), (0.017508, 1.871309)]
hold sign -1 above yield False n 21 full MAE 0.017749 ablated MAE 0.501744
hold sign -1 above yield True n 13 full MAE 0.018824 ablated MAE 1.746921
hold sign 1 above yield False n 32 full MAE 0.015264 ablated MAE 0.440331
hold sign 1 above yield True n 24 full MAE 0.016794 ablated MAE 1.938686

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2d8aaf67ccf5d1fc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_2d8aaf67 (directory).
```

### Turn 11 — `submit_completed_task` (`23108 ms` LLM)
- **Rationale**: The verified artifact retains the exact sub-yield recurrence and includes both terms that resolve ng_t09. It passed isolated Python schema validation and all tested tolerances, including both hold-stress signs, held-out labs, and separate hold-only and history-only activation.
- **Alternatives Weighed**: `Refit the linear-only model without the above-yield terms` (rejected: After refitting, its public above-yield MAE was 1.87131 versus 0.01751 for the full model.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoplastic_overstress_hereditary_solver.py`. Public strain MAE was 0.01677 overall, 0.01751 above yield, and 0.01625 below yield. The largest public relaxation-time error was 1.53%, and the relative viscoplastic-weight error was 0.60%. All four held-out lab folds and both mixed-activation stress tests passed.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M93-Lifecycle",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/viscoplastic_overstress_hereditary_solver.py": "FILE:8331"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "max_tau_rel_err": 0.014206,
    "max_vp_rel_l2": 0.006455,
    "max_all_mae": 0.018202,
    "max_overstress_mae": 0.015872,
    "max_subyield_mae": 0.019828,
    "student_elapsed_sec": 0.6156,
    "instance_diagnostics": [
      {
        "seed": 99521,
        "tau_rel": 0.010608,
        "vp_rel_l2": 0.001501,
        "all_mae": 0.018202,
        "over_mae": 0.015872,
        "sub_mae": 0.019828
      },
      {
        "seed": 99522,
        "tau_rel": 0.014206,
        "vp_rel_l2": 0.006455,
        "all_mae": 0.014878,
        "over_mae": 0.013937,
        "sub_mae": 0.015535
      }
    ],
    "candidate_timing_breakdown": {
      "import_wall_sec": 0.088201,
      "import_cpu_sec": 0.08103,
      "candidate_call_wall_sec": 0.567156,
      "candidate_call_cpu_sec": 0.521,
      "ipc_and_codec_wall_sec": 0.125033,
      "rpc_call_count": 2,
      "verifier_and_gt_wall_sec": 0.008933,
      "verifier_and_gt_cpu_sec": 0.030853,
      "total_supervisor_wall_sec": 0.789323
    }
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"max_tau_rel_err\": 0.014206, \"max_vp_rel_l2\": 0.006455, \"max_all_mae\": 0.018202, \"max_overstress_mae\": 0.015872, \"max_subyield_mae\": 0.019828, \"student_elapsed_sec\": 0.6156, \"instance_diagnostics\": [{\"seed\": 99521, \"tau_rel\": 0.010608, \"vp_rel_l2\": 0.001501, \"all_mae\": 0.018202, \"over_mae\": 0.015872, \"sub_mae\": 0.019828}, {\"seed\": 99522, \"tau_rel\": 0.014206, \"vp_rel_l2\": 0.006455, \"all_mae\": 0.014878, \"over_mae\": 0.013937, \"sub_mae\": 0.015535}]}\n__HEGEL_VERIFIER_AUTH_83be0b3dcd95f5a6ddb4ac1be90e5eae__:{\"strict_pass\": true, \"reward\": 1.0, \"max_tau_rel_err\": 0.014206, \"max_vp_rel_l2\": 0.006455, \"max_all_mae\": 0.018202, \"max_overstress_mae\": 0.015872, \"max_subyield_mae\": 0.019828, \"student_elapsed_sec\": 0.6156, \"instance_diagnostics\": [{\"seed\": 99521, \"tau_rel\": 0.010608, \"vp_rel_l2\": 0.001501, \"all_mae\": 0.018202, \"over_mae\": 0.015872, \"sub_mae\": 0.019828}, {\"seed\": 99522, \"tau_rel\": 0.014206, \"vp_rel_l2\": 0.006455, \"all_mae\": 0.014878, \"over_mae\": 0.013937, \"sub_mae\": 0.015535}], \"candidate_timing_breakdown\": {\"import_wall_sec\": 0.088201, \"import_cpu_sec\": 0.08103, \"candidate_call_wall_sec\": 0.567156, \"candidate_call_cpu_sec\": 0.521, \"ipc_and_codec_wall_sec\": 0.125033, \"rpc_call_count\": 2, \"verifier_and_gt_wall_sec\": 0.008933, \"verifier_and_gt_cpu_sec\": 0.030853, \"total_supervisor_wall_sec\": 0.789323}}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5572.57,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
