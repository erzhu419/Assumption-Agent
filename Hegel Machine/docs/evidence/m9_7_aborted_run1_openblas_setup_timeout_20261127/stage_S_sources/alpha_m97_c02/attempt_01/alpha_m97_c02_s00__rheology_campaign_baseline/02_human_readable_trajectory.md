# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `alpha_m97_c02_s00__rheology_campaign_baseline`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M97-NonNested-Counterfactual-Lineage-World` |
| **Domain** | `Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream` |
| **Wave / Wormhole Family** | `alpha_m97_c02_source` / `fam_hereditary_memory_and_non_nested_rheology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `352,276` (`337,794` / `27,136` / `14,482`) |
| **Total Episode Wall Time (s)** | `502.55s` |

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

### Turn 1 — `execute_in_remote_docker` (`17409 ms` LLM)
- **Rationale**: Inspect public field shapes, split conventions, and validator expectations before implementing the solver; discrepancies would change calibration, alignment, or output assembly. Check for reusable primitives for the requested robust fit and panel alignment.

```bash
cd /holdout_task && cat validate_public_schema.py && cat public_asset_index.json && python3 - <<'PY'
import pickle, numpy as np
with open('public_alpha_m97_c02_s00.pkl','rb') as f: d=pickle.load(f)
def show(d,depth=0):
 for k,v in d.items():
  print(' '*depth,k,type(v).__name__,getattr(v,'shape',''),getattr(v,'dtype',''),end=' ')
  if isinstance(v,dict):
   print();show(v,depth+2)
  elif isinstance(v,np.ndarray): print(repr(v.reshape(-1)[:8]))
  elif isinstance(v,(list,tuple)): print(len(v),repr(v[:1])[:400])
  else: print(repr(v)[:400])
show(d)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'Cauchy robust IRLS reference-panel alignment imputation viscoelastic relaxation'
```
- **Container Exit Code**: `0` (`677 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
le)
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 11, in <module>
  File "<stdin>", line 4, in show
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`18655 ms` LLM)
- **Rationale**: The pickle is a list of records, so inspect its dataset field directly. Per-channel affine residuals determine whether archive fallback is needed; panel dimensions and missingness determine how to apply the reference statistics and impute features.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
item=pickle.load(open('public_alpha_m97_c02_s00.pkl','rb'))[0]
d=item['dataset']
def show(d,ind=''):
 for k,v in d.items():
  if isinstance(v,dict): print(ind,k); show(v,ind+'  ')
  elif isinstance(v,np.ndarray): print(ind,k,v.shape,v.dtype,repr(v.ravel()[:6]), ('nonfinite='+str((~np.isfinite(v)).sum())) if v.dtype.kind in 'f' else '')
  else: print(ind,k,str(v)[:200])
show(d)
print('dt and t',d['past_dt'],d['eval_times'])
a=d['calibration_load_cells'];
for c in np.unique(a['lab_id']):
 m=a['lab_id']==c; x=a['cmd_stress'][m]; y=a['true_stress'][m]; p=np.polyfit(x,y,1);print('channel',c,len(x),p,'max residual',max(abs(y-np.polyval(p,x))))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.weighted_column_fill
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit
```
- **Container Exit Code**: `0` (`623 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"counter_regime_guards": [],
      "function_level_applicability": {
        "cauchy_linear_fit": "APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)",
        "fit_anchored_relaxation": "APPLICABLE_SUBROUTINE(real_1d_increasing_time_relaxation)",
        "fit_local_level_step": "APPLICABLE_SUBROUTINE(real_1d_step_series)"
      },
      "functions": [
        {
          "function": "cauchy_linear_fit",
          "signature": "cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float]",
          "doc_summary": "Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank before and after weighting, and positive scale if supplied and positive floor, returning (coefficients (p,), floored residual MAD)."
        }
      ],
      "index_cards": [
        "autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit",
        "autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation",
        "autogenous_prim_robust_trend_decomposition@1f45421a | fit_local_level_step(predictor: np.ndarray, values: np.ndarray, splits: np.ndarray, *, scale: float, loss_clip: float) -> tuple[int, float] | Fit finite predictor and values (n,) and integer splits (k,) with k >= 1, at l | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_local_level_step"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_robust_trend_decomposition",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`32488 ms` LLM)
- **Rationale**: Check whether the public rows stay in the stated regime and whether onset-rate observations have a different noise scale from curve measurements. The results determine the fit weights and whether any calibration fallback is needed.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
with open('public_alpha_m97_c02_s00.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
c=d['calibration_load_cells']; tr=d['train']; te=d['test']; t=d['eval_times']; x=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); r=np.vstack([tr['ref_coupons'],te['ref_coupons']]); labs=np.r_[tr['lab_id'],te['lab_id']]; strata=np.r_[tr['stratum'],te['stratum']]; cal=[]
for lab in range(4):
 m=c['lab_id']==lab; u,y=c['cmd_stress'][m],c['true_stress'][m]; p=np.polyfit(u,y,1); cal.append(p); a=labs==lab; mu=np.median(np.nanmean(r[a],axis=0)); sd=np.median(np.nanstd(r[a],axis=0)); x[a]=(x[a]-mu)/sd
 print('lab',lab,'cal rows',m.sum(),'gain/bias',p,'max affine residual',np.max(abs(y-np.polyval(p,u))),'ref center/scale',mu,sd,'cell counts',np.unique(strata[a],return_counts=True)[1])
means=np.array([np.nanmean(x[(labs==l)&(strata==s)],axis=0) for l,s in np.unique(np.c_[labs,strata],axis=0)]); x=np.where(np.isfinite(x),x,np.nanmean(means,axis=0)); print('range',t[[0,-1]],'history duration',sum(d['past_dt']),'curve and rate shape',tr['strain_curves_obs'].shape,tr['instant_rate_obs'].shape)
cal=np.array(cal); g,b=cal[tr['lab_id']].T; past=g[:,None]*tr['past_stress_cmd']+b[:,None]; hold=g*tr['stress_hold_cmd']+b; print('physical max',abs(past).max(),'over threshold rows',np.any(abs(past)>d['crit_stress'],axis=1).sum()); f=np.c_[np.ones(len(past)),x[:len(past)]]; y=(tr['strain_curves_obs']-tr['strain_0'][:,None]).ravel(); rate=tr['instant_rate_obs']; dt=d['past_dt']; age=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; drift=t[None,:,None]*f[:,None,:]
from scipy.optimize import least_squares
from itertools import combinations
q=np.geomspace(.12,16,12); best=None
for a,s in combinations(q,2):
 ts=np.array([a,s]); gaps=hold[:,None]-past@(np.exp(-age[:,None]/ts)*-np.expm1(-dt[:,None]/ts)); basis=np.concatenate([gaps[:,None,:]*-np.expm1(-t[:,None]/ts),drift],axis=2).reshape(-1,7); h=np.linalg.lstsq(basis,y,rcond=None)[0]; score=np.median(abs(basis@h-y));
 if best is None or score<best[0]: best=(score,ts,basis,np.c_[gaps/ts,f],h)
s,ts,A,R,h=best; h=least_squares(lambda h:A@h-y,h,loss='cauchy',f_scale=max(s*2,1e-5)).x; print('coarse taus, weights',ts,h); print('curve/rate absolute residual quantiles',*[np.quantile(abs(z),[.25,.5,.75,.9,.95,.99,1.]) for z in (A@h-y,R@h-rate)])
PY
p=/tmp/hegel_continual_archive/archive_calibration_history.json; if test -f "$p"; then head -c 1800 "$p"; printf '\n'; else echo 'No calibration archive mounted'; fi
```
- **Container Exit Code**: `0` (`4402 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lab 0 cal rows 10 gain/bias [9.99976805e-01 1.21877020e-06] max affine residual 0.00019668910391446026 ref center/scale -0.6277087010318103 0.7682641555060028 cell counts [24 17 13]
lab 1 cal rows 10 gain/bias [9.99993252e-01 3.53543932e-05] max affine residual 0.00013029032594710777 ref center/scale 0.45268751509577454 1.253864859647711 cell counts [10 29 19]
lab 2 cal rows 10 gain/bias [1.00000751e+00 2.72105742e-05] max affine residual 0.00010014096341892653 ref center/scale -0.24618661470902037 0.9282085445762174 cell counts [30 16 32]
lab 3 cal rows 10 gain/bias [ 1.00000953e+00 -3.06336064e-05] max affine residual 0.00013389717760170505 ref center/scale 0.8391382876398509 1.4302658014559029 cell counts [19 20 23]
range [0.3 5.2] history duration 7.713900000000001 curve and rate shape (180, 10) (180,)
physical max 2.067050326684861 over threshold rows 0
coarse taus, weights [0.18722226 1.73076753] [ 0.32095637  0.82124269  0.13654517  0.16507904 -0.13441582  0.11782915
 -0.08675197]
curve/rate absolute residual quantiles [2.21001988e-02 4.72259082e-02 8.54383819e-02 1.57868392e-01
 3.19687625e-01 1.16635647e+00 6.82713151e+02] [0.45846518 0.95977816 1.48232954 2.18628766 2.56186194 2.88936692
 3.13931502]
No calibration archive mounted
```

### Turn 4 — `execute_in_remote_docker` (`45925 ms` LLM)
- **Rationale**: Fit the prescribed Prony and drift basis with a Cauchy loss, multiple time-scale starts, and an analytic Jacobian. Comparing curve-only and joint residuals will decide whether onset rates should enter the fit; comparing the two imputation readings will identify which needs held-out validation.
- **Alternatives Weighed**: `Discard the onset-rate observations on the basis of the coarse fit` (rejected: The first strain sample is at 0.3, so the curve grid may underresolve a fast branch that is visible in the derivative.)

```bash
cat > /tmp/rheo_explore.py <<'PY'
import pickle, numpy as np
from scipy.optimize import least_squares
from itertools import combinations

def data(d, fill='balanced'):
    tr,te=d['train'],d['test']; keys=('lab_id','stratum','coupon_features_raw','ref_coupons','past_stress_cmd','stress_hold_cmd','strain_0')
    lab,st,x,r,p,u,s0=(np.concatenate((tr[k],te[k])) for k in keys)
    c=d['calibration_load_cells']; cal=[]
    for l in range(4):
        idx=c['lab_id']==l; cal.append(np.polyfit(c['cmd_stress'][idx],c['true_stress'][idx],1))
        idx=lab==l; x[idx]=(x[idx]-np.nanmedian(np.nanmean(r[idx],axis=0)))/np.nanmedian(np.nanstd(r[idx],axis=0))
    cells=[(lab==l)&(st==s) for l,s in np.unique(np.column_stack((lab,st)),axis=0)]
    means=np.array([np.nanmean(x[m],axis=0) for m in cells]); balanced=np.nanmean(means,axis=0)
    for m,mu in zip(cells,means): x[m]=np.where(np.isfinite(x[m]),x[m],np.where(np.isfinite(mu),mu,balanced) if fill=='local' else balanced)
    gain,bias=np.asarray(cal)[lab].T
    return gain[:,None]*p+bias[:,None],gain*u+bias,np.column_stack((np.ones(len(x)),x)),s0

def basis(lt,p,u,f,dt,t):
    tau=np.exp(lt); age=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; a=age[:,None]/tau; q=dt[:,None]/tau; r=t[:,None]/tau
    ea=np.exp(-a); eq=np.exp(-q); er=np.exp(-r); w=ea*-np.expm1(-q); dw=a*w-q*ea*eq
    g=u[:,None]-p@w; dg=-p@dw; e=-np.expm1(-r)
    curve=np.concatenate((g[:,None,:]*e,t[None,:,None]*f[:,None,:]),axis=2)
    rates=np.column_stack((g/tau,f)); dc=dg[:,None,:]*e-g[:,None,:]*r*er; dr=(dg-g)/tau
    return curve,rates,dc,dr

def fit(p,u,f,dt,t,y,rate,sc=.025,sr=None,starts=None):
    ns=len(t)
    def ev(b,jac):
        C,R,dc,dr=basis(b[:2],p,u,f,dt,t); w=b[2:]; A=C.reshape(-1,7); D=dc.reshape(-1,2)
        if jac:
            a=np.column_stack((D*w[:2],A))/sc
            return a if sr is None else np.vstack((a,np.column_stack((dr*w[:2],R))/sr))
        a=(A@w-y.ravel())/sc
        return a if sr is None else np.r_[a,(R@w-rate)/sr]
    if starts is None: starts=list(combinations([.035,.09,.20,.45,1.,2.4,6.,15.],2))
    out=[]
    for pair in starts:
        lt=np.log(pair); C,R,_,_=basis(lt,p,u,f,dt,t); A=C.reshape(-1,7)
        # Least-squares initialization with bounded response leverage, then Cauchy refinement.
        z=y.ravel(); wt=1/np.maximum(1,np.abs(z)/max(np.quantile(abs(z),.90),.01))
        h=np.linalg.lstsq(A*wt[:,None],z*wt,rcond=None)[0]
        sol=least_squares(lambda b:ev(b,False),np.r_[lt,h],jac=lambda b:ev(b,True),loss='cauchy',bounds=(np.r_[np.log([.01,.01]),np.full(7,-np.inf)],np.r_[np.log([100.,100.]),np.full(7,np.inf)]),max_nfev=130,x_scale='jac')
        b=sol.x; ts=np.exp(b[:2]); order=np.argsort(ts); out.append((sol.cost,ts[order],np.r_[b[2:4][order],b[4:]]))
    return sorted(out,key=lambda z:z[0])

if __name__=='__main__':
    with open('/holdout_task/public_alpha_m97_c02_s00.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
    tr=d['train']; rate=tr['instant_rate_obs']; n=len(rate); t=d['eval_times']; dt=d['past_dt']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]
    for fill in ('balanced','local'):
        p,u,f,_=data(d,fill); p,u,f=p[:n],u[:n],f[:n]
        for sr in (None,.1,.025,.006):
            a=fit(p,u,f,dt,t,y,rate,sr=sr); cost,tau,h=a[0]; C,R,*_=basis(np.log(tau),p,u,f,dt,t); ce=C@h-y; re=R@h-rate
            print(fill,'rate scale',sr,'best',round(cost,3),tau,h,'basins',[(round(c,2),q.round(4).tolist()) for c,q,_ in a[::6]],flush=True)
            print('resid quantiles .5/.8/.95: curve',np.quantile(abs(ce),[.5,.8,.95]),'rate',np.quantile(abs(re),[.5,.8,.95]),'rate corr',np.corrcoef(R@h,rate)[0,1],flush=True)
            if sr is None: print('rate obs vs pred',np.c_[rate,R@h][:10].round(4),'curve horizon medians',np.median(abs(ce),axis=0).round(4),flush=True)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/rheo_explore.py
```
- **Container Exit Code**: `0` (`8521 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ate corr 0.9998209176863093
balanced rate scale 0.006 best 1056.763 [0.57134639 3.58034115] [ 0.68588155  0.55075099  0.14037094  0.16739949 -0.13809952  0.11803485
 -0.08845808] basins [(1056.76, [0.5713, 3.5803]), (1056.76, [0.5713, 3.5804]), (1056.76, [0.5713, 3.5804]), (1056.76, [0.5713, 3.5804]), (2405.03, [0.7386, 0.7393])]
resid quantiles .5/.8/.95: curve [0.01487746 0.03773121 0.30458941] rate [0.01446654 0.02862578 0.04995184] rate corr 0.9998211701688894
local rate scale None best 880.13 [0.5728411  3.60501194] [ 0.68629873  0.55219783  0.14055696  0.16726637 -0.13820673  0.11782418
 -0.08826041] basins [(880.13, [0.5728, 3.605]), (880.13, [0.5728, 3.605]), (880.13, [0.5728, 3.605]), (880.13, [0.5728, 3.605]), (1942.32, [0.8267, 0.8284])]
resid quantiles .5/.8/.95: curve [0.01510783 0.03792648 0.30151701] rate [0.01514543 0.02790902 0.05089693] rate corr 0.9998319288475742
rate obs vs pred [[-0.3579 -0.3516]
 [ 1.8454  1.8373]
 [-2.987  -2.9743]
 [ 0.9647  0.944 ]
 [-0.6152 -0.6127]
 [-1.0628 -1.0778]
 [ 1.0996  1.0844]
 [-1.9828 -1.9982]
 [-2.3877 -2.4107]
 [ 2.2897  2.3159]] curve horizon medians [0.011  0.012  0.0124 0.0134 0.0153 0.0162 0.0165 0.0175 0.0189 0.0179]
local rate scale 0.1 best 886.982 [0.57206308 3.59633884] [ 0.68555407  0.55256311  0.1405535   0.16727406 -0.13819697  0.11782716
 -0.08826285] basins [(886.98, [0.5721, 3.5963]), (886.98, [0.5721, 3.5963]), (886.98, [0.5721, 3.5963]), (886.98, [0.5721, 3.5963]), (2047.77, [0.7847, 0.7861])]
resid quantiles .5/.8/.95: curve [0.01504964 0.03778845 0.30127303] rate [0.01466776 0.02824437 0.05154339] rate corr 0.9998323811552615
local rate scale 0.025 best 927.383 [0.57142848 3.59230523] [ 0.68516598  0.55273677  0.14053669  0.16729893 -0.13817781  0.11782468
 -0.08828195] basins [(927.38, [0.5714, 3.5923]), (927.38, [0.5714, 3.5923]), (927.38, [0.5714, 3.5923]), (927.38, [0.5714, 3.5923]), (2207.03, [0.7573, 0.7582])]
resid quantiles .5/.8/.95: curve [0.01499116 0.03778645 0.30094831] rate [0.01433393 0.02882328 0.05103781] rate corr 0.9998327321493624
local rate scale 0.006 best 1062.592 [0.57091303 3.58044608] [ 0.6850088   0.55215129  0.14052154  0.16731673 -0.13818131  0.11781267
 -0.08834277] basins [(1062.59, [0.5709, 3.5804]), (1062.59, [0.5709, 3.5804]), (1062.59, [0.5709, 3.5804]), (1062.59, [0.5709, 3.5805]), (2426.51, [0.7352, 0.7365])]
resid quantiles .5/.8/.95: curve [0.01478849 0.0375836  0.30043676] rate [0.01449725 0.02861672 0.05078344] rate corr 0.9998329470832618
```

### Turn 5 — `execute_in_remote_docker` (`71532 ms` LLM)
- **Rationale**: Write the solver and run the isolated schema validator to catch import failures, missing keys, nonfinite values, or incorrect output shapes before testing held-out labs and horizons.
- **Alternatives Weighed**: `Fill each missing feature with its own lab/stratum mean` (rejected: Equal weighting of cell means matches the cell-balanced specification and gave a lower public robust objective (922.45 versus 927.38).); `Fit curves without onset-rate observations` (rejected: Joint fitting gives nearly identical curve error while directly constraining the derivative used for onset-rate prediction.)

```bash
cat > /holdout_task/alpha_m97_c02_s00_solver.py <<'PY'
"""Calibrated two-branch Prony fit for the linear-flow campaign."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _archived_calibration(channel):
    if not _ARCHIVE.is_file():
        return None
    with _ARCHIVE.open() as stream:
        history = json.load(stream)
    if isinstance(history, dict):
        history = history.get('channels', history.get('calibration_load_cells', history))
    if isinstance(history, dict) and 'lab_id' in history:
        if 'cmd_stress' in history and 'true_stress' in history:
            mask = np.asarray(history['lab_id']) == channel
            x = np.asarray(history['cmd_stress'])[mask]
            y = np.asarray(history['true_stress'])[mask]
            return np.polyfit(x, y, 1) if np.unique(x).size >= 2 else None
    if isinstance(history, dict):
        entry = history.get(str(channel))
    else:
        entry = next((row for row in reversed(history)
                      if isinstance(row, dict) and row.get('lab_id') == channel
                      and 'gain' in row and 'bias' in row), None)
    if entry is not None:
        return (entry['gain'], entry['bias']) if isinstance(entry, dict) else entry
    return None


def _calibrate(table):
    lab, cmd, true = (np.asarray(table[key]) for key in ('lab_id', 'cmd_stress', 'true_stress'))
    result = np.empty((4, 2))
    for channel in range(4):
        take = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[take], true[take]
        i, j = np.triu_indices(x.size, k=1)
        distinct = x[i] != x[j]
        i, j = i[distinct], j[distinct]
        spike = False
        if i.size:
            gain = np.median((y[i] - y[j]) / (x[i] - x[j]))
            offsets = y - gain*x
            clean = np.abs(offsets - np.median(offsets)) <= 3.0
            spike = not np.all(clean)
        else:
            clean = np.ones(x.size, dtype=bool)
        archive = _archived_calibration(channel) if x.size < 2 or spike else None
        if np.count_nonzero(clean) >= 4 and np.unique(x[clean]).size >= 2:
            result[channel] = np.polyfit(x[clean], y[clean], 1)
        elif archive is not None:
            result[channel] = archive
        elif np.unique(x[clean]).size >= 2:
            result[channel] = np.polyfit(x[clean], y[clean], 1)
        else:
            result[channel] = (1.0, np.median(y - x) if y.size else 0.0)
    return result


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    keys = ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons',
            'past_stress_cmd', 'stress_hold_cmd', 'strain_0')
    lab, stratum, x, ref, past, hold, strain0 = (
        np.concatenate((train[key], test[key])) for key in keys
    )
    x = x.astype(float, copy=False)
    for channel in np.unique(lab):
        mask = lab == channel
        panel = ref[mask]
        panel = panel[:, np.any(np.isfinite(panel), axis=0)]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[mask] = (x[mask] - center) / scale

    # Each observed lab/stratum cell contributes one column mean, not one
    # vote per coupon; compute the means after panel alignment.
    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    means = np.full((len(cells), x.shape[1]), np.nan)
    for row, (channel, group) in enumerate(cells):
        values = x[(lab == channel) & (stratum == group)]
        counts = np.isfinite(values).sum(axis=0)
        np.divide(np.nansum(values, axis=0), counts, out=means[row], where=counts > 0)
    fill = np.nanmean(means, axis=0)
    x = np.where(np.isfinite(x), x, fill)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    return (gain[:, None]*past + bias[:, None], gain*hold + bias,
            np.column_stack((np.ones(len(x)), x)), strain0)


def _basis(log_taus, past, hold, features, dt, times):
    taus = np.exp(log_taus)
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    a, q, r = age[:, None]/taus, dt[:, None]/taus, times[:, None]/taus
    ea, eq, er = np.exp(-a), np.exp(-q), np.exp(-r)
    # Unroll the recurrence from z(0)=0; each history interval's
    # contribution decays through all subsequent intervals.
    memory_weights = ea * -np.expm1(-q)
    gap = hold[:, None] - past @ memory_weights
    dgap = -past @ (a*memory_weights - q*ea*eq)
    step = -np.expm1(-r)
    curve = np.concatenate((gap[:, None, :]*step,
                            times[None, :, None]*features[:, None, :]), axis=2)
    rate = np.column_stack((gap/taus, features))
    dcurve = dgap[:, None, :]*step - gap[:, None, :]*r*er
    drate = (dgap - gap)/taus
    return curve, rate, dcurve, drate


def _fit(past, hold, features, dt, times, curves, rates):
    scales = np.r_[np.full(curves.size, 0.025), np.full(rates.size, 0.025)]
    target = np.r_[curves.ravel(), rates]

    def design(log_taus):
        curve, rate, dcurve, drate = _basis(log_taus, past, hold, features, dt, times)
        return (np.vstack((curve.reshape(-1, 7), rate)),
                np.vstack((dcurve.reshape(-1, 2), drate)))

    def residual(params):
        matrix, _ = design(params[:2])
        return (np.einsum('ij,j->i', matrix, params[2:]) - target) / scales

    def jacobian(params):
        matrix, deriv = design(params[:2])
        return np.column_stack((deriv*params[2:4], matrix)) / scales[:, None]

    # Downweight large glitches for initialization, then minimize the
    # observation-level Cauchy loss on both the curve and its initial slope.
    weight = 1 / np.maximum(1, np.abs(target)/max(np.quantile(np.abs(target), 0.9), 0.01))
    lower = np.r_[np.log([0.01, 0.01]), np.full(7, -np.inf)]
    upper = np.r_[np.log([100., 100.]), np.full(7, np.inf)]
    best = None
    for pair in ((0.45, 2.4), (0.20, 1.0), (0.20, 6.0), (1.0, 6.0),
                 (0.09, 2.4), (0.45, 15.0)):
        log_taus = np.log(pair)
        matrix, _ = design(log_taus)
        initial = np.linalg.lstsq(matrix*weight[:, None], target*weight, rcond=None)[0]
        fit = least_squares(residual, np.r_[log_taus, initial], jac=jacobian,
                            bounds=(lower, upper), loss='cauchy', x_scale='jac', max_nfev=130)
        if best is None or fit.cost < best.cost:
            best = fit
    params = best.x
    order = np.argsort(params[:2])
    return params[:2][order], np.r_[params[2:4][order], params[4:]]


def solve_alpha_m97_c02_s00(dataset: dict) -> dict:
    """Estimate the shared rheology and predict every test coupon."""
    past, hold, features, strain0 = _prepare(dataset)
    dt, times = (np.asarray(dataset[key], dtype=float) for key in ('past_dt', 'eval_times'))
    train = dataset['train']
    n = len(train['lab_id'])
    curves = np.asarray(train['strain_curves_obs']) - strain0[:n, None]
    log_taus, weights = _fit(past[:n], hold[:n], features[:n], dt, times,
                             curves, np.asarray(train['instant_rate_obs']))
    curve, rate, _, _ = _basis(log_taus, past[n:], hold[n:], features[n:], dt, times)
    return {'estimated_taus': np.exp(log_taus), 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2),
            'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`578 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/alpha_m97_c02_s00_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c4caea57c8e0695d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c4caea57 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`29784 ms` LLM)
- **Rationale**: Withhold each lab’s response measurements while retaining the permitted unlabeled reference panel. A lab- or horizon-specific error increase would motivate revisiting alignment or the fit; an independent recurrence mismatch would require correcting the prediction basis.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
from pathlib import Path
p=Path('/holdout_task')
spec=importlib.util.spec_from_file_location('solver',p/'alpha_m97_c02_s00_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with (p/'public_alpha_m97_c02_s00.pkl').open('rb') as f:
    d=pickle.load(f)[0]['dataset']
tr,te=d['train'],d['test']
n=len(tr['lab_id']); dt=d['past_dt']; t=d['eval_times']; y=tr['strain_curves_obs']; rate=tr['instant_rate_obs']; lab=tr['lab_id']
curve_errors=np.empty_like(y); rate_errors=np.empty_like(rate)
for held in np.unique(lab):
    keep=lab!=held; val=~keep
    fold=dict(d, train={k:v[keep] for k,v in tr.items()}, test={k:np.concatenate((tr[k][val],v)) for k,v in te.items()})
    out=m.solve_alpha_m97_c02_s00(fold); nv=int(val.sum())
    curve_errors[val]=abs(out['predicted_strain'][:nv]-y[val]); rate_errors[val]=abs(out['initial_rate_pred'][:nv]-rate[val]);
    assert np.all(np.diff(out['estimated_taus'])>0) and np.array_equal(out['regime_mode_weights'],np.zeros(2))
    print('held lab',held,'curve medians by horizon',np.round(np.median(curve_errors[val],axis=0),5),'rate med/p90',np.round(np.quantile(rate_errors[val],[.5,.9]),5),'taus',np.round(out['estimated_taus'],5))
missing=np.isnan(tr['coupon_features_raw']).any(axis=1)
for name, sel in [('missing features',missing),('complete features',~missing),('high-stress histories',abs(tr['past_stress_cmd']).max(axis=1)>=np.quantile(abs(tr['past_stress_cmd']).max(axis=1),.9))]:
    print(name,sel.sum(),'curve med/p90',np.round(np.quantile(curve_errors[sel],[.5,.9]),5),'rate med/p90',np.round(np.quantile(rate_errors[sel],[.5,.9]),5))
past,hold,F,strain0=m._prepare(d); out=m.solve_alpha_m97_c02_s00(d); tau=out['estimated_taus']; h=out['hereditary_weights']; z=np.zeros((len(past)-n,2));
for j,step in enumerate(dt):
    decay=np.exp(-step/tau); z=z*decay+past[n:,j,None]*(1-decay)
gap=hold[n:,None]-z; drift=F[n:]@h[2:]; expected=strain0[n:,None]+(gap*h[:2])@(1-np.exp(-t[None,:]/tau[:,None]))+drift[:,None]*t
np.testing.assert_allclose(out['predicted_strain'],expected,rtol=1e-12,atol=1e-12)
np.testing.assert_allclose(out['initial_rate_pred'],(gap*h[:2])@(1/tau)+drift,rtol=1e-12,atol=1e-12)
print('Explicit memory recurrence and onset derivative match solver predictions; full-fit taus',tau)
PY
```
- **Container Exit Code**: `0` (`1020 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held lab 0 curve medians by horizon [0.01177 0.01313 0.01422 0.02295 0.03112 0.02498 0.04558 0.05218 0.0494
 0.05657] rate med/p90 [0.01644 0.04001] taus [0.57331 3.70412]
held lab 1 curve medians by horizon [0.01137 0.01494 0.00973 0.01185 0.01389 0.01489 0.01482 0.017   0.0142
 0.01313] rate med/p90 [0.0159  0.03584] taus [0.57067 3.58338]
held lab 2 curve medians by horizon [0.01143 0.01038 0.01334 0.0131  0.01057 0.01359 0.01701 0.01391 0.01919
 0.01405] rate med/p90 [0.01343 0.03922] taus [0.57206 3.52394]
held lab 3 curve medians by horizon [0.01059 0.01063 0.01921 0.01384 0.01928 0.02006 0.01951 0.02165 0.02545
 0.02955] rate med/p90 [0.01514 0.03063] taus [0.57006 3.49592]
missing features 13 curve med/p90 [0.12479 0.70218] rate med/p90 [0.06571 0.18062]
complete features 167 curve med/p90 [0.01512 0.06946] rate med/p90 [0.01394 0.03439]
high-stress histories 100 curve med/p90 [0.01698 0.09649] rate med/p90 [0.0166  0.04023]
Explicit memory recurrence and onset derivative match solver predictions; full-fit taus [0.57157768 3.58315422]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c4caea57c8e0695d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c4caea57 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`35062 ms` LLM)
- **Rationale**: Compare imputation on held-out responses rather than relying on the training objective. Separately test a documented EMI spike: if one corrupt calibration point moves the affine fit materially despite nine clean points, fix spike rejection before submitting.
- **Alternatives Weighed**: `Submit the already verified solver` (rejected: The missing-feature slice is materially weaker; a focused held-out comparison can determine whether a plausible reading of the imputation instruction improves it.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'alpha_m97_c02_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_alpha_m97_c02_s00.pkl', 'rb') as f:
    records = pickle.load(f)
print('Public records:', len(records))
for record in records:
    d = record['dataset']; tr = d['train']
    lab = tr['lab_id']; missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
    if not missing.any():
        continue
    past, hold, F, s0 = m._prepare(d)
    r = np.concatenate((tr['ref_coupons'], d['test']['ref_coupons']))
    x = np.concatenate((tr['coupon_features_raw'], d['test']['coupon_features_raw']))
    labs = np.concatenate((lab, d['test']['lab_id']))
    strata = np.concatenate((tr['stratum'], d['test']['stratum']))
    F_local = F.copy()
    for c in np.unique(labs):
        lm = labs == c; panel = r[lm]
        center = np.nanmedian(np.nanmean(panel, axis=0)); scale = np.nanmedian(np.nanstd(panel, axis=0))
        for s in np.unique(strata[lm]):
            cell = lm & (strata == s); z = (x[cell] - center)/scale
            mu = np.nanmean(z, axis=0)
            F_local[cell, 1:] = np.where(np.isfinite(z), z, np.where(np.isfinite(mu), mu, F[cell, 1:]))
    n = len(lab); dt, t = d['past_dt'], d['eval_times']
    y = tr['strain_curves_obs'] - s0[:n, None]; rate = tr['instant_rate_obs']
    for name, features in [('equal-cell pooled', F[:n]), ('within-cell', F_local[:n])]:
        ce = np.empty_like(y); re = np.empty_like(rate)
        for held in np.unique(lab):
            a = lab != held; b = ~a
            lt, w = m._fit(past[:n][a], hold[:n][a], features[a], dt, t, y[a], rate[a])
            C, R, *_ = m._basis(lt, past[:n][b], hold[:n][b], features[b], dt, t)
            ce[b], re[b] = np.abs(C@w-y[b]), np.abs(R@w-rate[b])
        for label, mask in [('missing', missing), ('complete', ~missing)]:
            print(name, label, mask.sum(), 'curve median/p90', np.quantile(ce[mask], [.5,.9]).round(5), 'rate median/p90', np.quantile(re[mask], [.5,.9]).round(5))
    table = {k: v.copy() for k, v in d['calibration_load_cells'].items()}
    baseline = m._calibrate(table)
    idx = np.flatnonzero(table['lab_id'] == 2)[0]
    table['true_stress'][idx] += 8.0
    np.testing.assert_allclose(m._calibrate(table), baseline, atol=2e-4, rtol=0)
    print('Single +8 stress EMI removal passed')
PY
```
- **Container Exit Code**: `0` (`1233 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public records: 1
equal-cell pooled missing 13 curve median/p90 [0.12479 0.70218] rate median/p90 [0.06571 0.18062]
equal-cell pooled complete 167 curve median/p90 [0.01512 0.06946] rate median/p90 [0.01394 0.03439]
within-cell missing 13 curve median/p90 [0.152   0.64442] rate median/p90 [0.06637 0.16603]
within-cell complete 167 curve median/p90 [0.01535 0.06826] rate median/p90 [0.01388 0.03475]
Single +8 stress EMI removal passed

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c4caea57c8e0695d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c4caea57 (directory).
```

### Turn 8 — `submit_completed_task` (`16884 ms` LLM)
- **Rationale**: The saved solver passed the isolated schema check, independent recurrence and derivative checks, and a calibration test with an injected EMI spike. Held-out results did not justify changing the specified imputation.
- **Alternatives Weighed**: `Impute from each coupon’s own lab/stratum cell` (rejected: It improved 90th-percentile errors but worsened median strain error on the 13 missing-feature coupons and replaces the specified equal-cell average.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_s00_solver.py': 'FILE:7423'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_m97_c02_s00_solver.py`. The isolated schema validator passed, and independent calculations confirmed the memory recurrence and onset-rate predictions. Coupons with missing features remained the hardest held-out slice.`

### Turn 9 — `execute_in_remote_docker` (`41147 ms` LLM)
- **Rationale**: Re-express the same measured stresses with different gains and offsets in all four channels: prediction changes would reveal a calibration or channel-index error. Check held-out lab biases and the genuinely highest-stress complete coupons; a systematic error there would motivate revisiting alignment or fit weighting. Re-run the schema validator in an isolated process.
- **Alternatives Weighed**: `Add rational flow terms or relay modes` (rejected: The specified regime has an exact linear envelope and zero mode weights; feature sums and differences are already in the linear span. There is no relay or stochastic-control state in this task.); `Use within-batch balanced accuracy to check confounding` (rejected: Both responses are continuous. Signed residuals for each held-out lab test the relevant failure: pooled error obscuring a lab-specific drift or rate bias.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, pickle
import numpy as np
from scipy.stats import spearmanr
spec = importlib.util.spec_from_file_location('solver', 'alpha_m97_c02_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_alpha_m97_c02_s00.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr = d['train']; n = len(tr['lab_id']); lab = tr['lab_id']
past, hold, F, s0 = m._prepare(d)
dt, t = d['past_dt'], d['eval_times']
y = tr['strain_curves_obs'] - s0[:n, None]; rate = tr['instant_rate_obs']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
curve_res = np.empty_like(y); rate_res = np.empty_like(rate)
for c in np.unique(lab):
    a = np.flatnonzero(lab != c); b = np.flatnonzero(lab == c)
    lt, w = m._fit(past[a], hold[a], F[a], dt, t, y[a], rate[a])
    C, R, *_ = m._basis(lt, past[b], hold[b], F[b], dt, t)
    curve_res[b] = y[b] - C @ w; rate_res[b] = rate[b] - R @ w
    for j in range(1, 5):
        for k in range(j+1, 5):
            pair = np.column_stack((F[b,j]+F[b,k], F[b,j]-F[b,k]))
            np.testing.assert_allclose(pair @ (np.array([w[j+2]+w[k+2], w[j+2]-w[k+2]])/2), F[b,j]*w[j+2]+F[b,k]*w[k+2], rtol=0, atol=1e-12)
    idx = b[complete[b]]
    drift_res = np.median(curve_res[idx]/t, axis=1)
    print('held lab', c, 'complete n', len(idx), 'median signed drift/rate residual', np.round([np.median(drift_res), np.median(rate_res[idx])], 6), 'drift/rate residual Spearman', round(spearmanr(drift_res,rate_res[idx]).statistic,3))
stress = abs(past[:n]).max(axis=1)
idx = complete & (stress >= np.sort(stress[complete])[-17]); assert idx.sum() == 17
print('Top 17 complete physical-stress histories, max and threshold:', stress[idx].max(), d['crit_stress'], 'curve/rate OOF absolute p90:', np.quantile(abs(curve_res[idx]),.9),np.quantile(abs(rate_res[idx]),.9))
base = m.solve_alpha_m97_c02_s00(d)
converted = copy.deepcopy(d)
g = np.array([.74,1.29,.87,1.18]); b = np.array([-.30,.26,.14,-.21])
cal = converted['calibration_load_cells']; c = cal['lab_id']
cal['cmd_stress'] = (cal['cmd_stress']-b[c])/g[c]
for split in ('train','test'):
    section = converted[split]; c = section['lab_id']
    section['past_stress_cmd'] = (section['past_stress_cmd']-b[c,None])/g[c,None]
    section['stress_hold_cmd'] = (section['stress_hold_cmd']-b[c])/g[c]
    if 'aux_chamber_telemetry' in section:
        section['aux_chamber_telemetry'] = np.full_like(section['aux_chamber_telemetry'], 1e6)
if 'aux_chamber_telemetry' in converted:
    converted['aux_chamber_telemetry'] = None
out = m.solve_alpha_m97_c02_s00(converted)
for k in base:
    np.testing.assert_allclose(out[k], base[k], rtol=2e-7, atol=2e-7)
    print('calibration/telemetry invariance', k, 'max absolute change:', np.max(np.abs(out[k]-base[k])))
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`1366 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held lab 0 complete n 39 median signed drift/rate residual [0.002927 0.003159] drift/rate residual Spearman 0.579
held lab 1 complete n 37 median signed drift/rate residual [-0.001246 -0.012791] drift/rate residual Spearman 0.109
held lab 2 complete n 54 median signed drift/rate residual [ 0.000641 -0.003352] drift/rate residual Spearman 0.362
held lab 3 complete n 37 median signed drift/rate residual [ 0.002923 -0.000288] drift/rate residual Spearman 0.392
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c4caea57c8e0695d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c4caea57 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 28, in <module>
AssertionError
```

### Turn 10 — `execute_in_remote_docker` (`30786 ms` LLM)
- **Rationale**: The previous audit stopped at a slice-size assertion, before testing calibration invariance. Select the highest-stress rows by rank to handle cutoff ties. A material held-out error increase would warrant examining the stress regime; a change under equivalent command units would warrant fixing calibration.

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', 'alpha_m97_c02_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_alpha_m97_c02_s00.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr = d['train']; lab = tr['lab_id']; n = len(lab)
p, u, F, s0 = m._prepare(d)
complete = np.flatnonzero(np.isfinite(tr['coupon_features_raw']).all(axis=1))
stress = np.max(np.abs(p[:n]), axis=1)
# Select a fixed number of rows even when measured stresses tie at the cutoff.
high = complete[np.argsort(-stress[complete], kind='stable')[:17]]
dt, t = d['past_dt'], d['eval_times']
y = tr['strain_curves_obs'] - s0[:n, None]; rate = tr['instant_rate_obs']
ce, re = [], []
for c in np.unique(lab[high]):
    keep = np.flatnonzero(lab != c); val = high[lab[high] == c]
    lt, w = m._fit(p[keep], u[keep], F[keep], dt, t, y[keep], rate[keep])
    C, R, *_ = m._basis(lt, p[val], u[val], F[val], dt, t)
    ce.append(np.abs(C @ w - y[val]).ravel()); re.append(np.abs(R @ w - rate[val]))
print('Top complete stress rows:', len(high), 'stress min/max:', stress[high].min(), stress[high].max(), 'threshold:', d['crit_stress'], 'OOF strain/rate p90:', np.quantile(np.concatenate(ce), .9), np.quantile(np.concatenate(re), .9))
base = m.solve_alpha_m97_c02_s00(d)
changed = copy.deepcopy(d)
g, b = np.array([.74, 1.29, .87, 1.18]), np.array([-.30, .26, .14, -.21])
cal = changed['calibration_load_cells']; lab = cal['lab_id']
cal['cmd_stress'] = (cal['cmd_stress'] - b[lab]) / g[lab]
for split in ('train', 'test'):
    s = changed[split]; lab = s['lab_id']
    s['past_stress_cmd'] = (s['past_stress_cmd'] - b[lab, None]) / g[lab, None]
    s['stress_hold_cmd'] = (s['stress_hold_cmd'] - b[lab]) / g[lab]
    if 'aux_chamber_telemetry' in s:
        s['aux_chamber_telemetry'] = None
changed['aux_chamber_telemetry'] = None
out = m.solve_alpha_m97_c02_s00(changed)
for k in base:
    np.testing.assert_allclose(out[k], base[k], rtol=2e-7, atol=2e-7)
    print(k, 'max absolute difference:', np.max(np.abs(out[k]-base[k])))
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`1299 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Top complete stress rows: 17 stress min/max: 2.0670427401859626 2.067050326684861 threshold: 2.65 OOF strain/rate p90: 0.057317693989442815 0.03661235924080843
estimated_taus max absolute difference: 3.2596236820836566e-09
hereditary_weights max absolute difference: 2.1837676111857718e-10
regime_mode_weights max absolute difference: 0.0
predicted_strain max absolute difference: 1.5434675759706806e-10
initial_rate_pred max absolute difference: 1.3136780552258642e-10
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c4caea57c8e0695d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c4caea57 (directory).
```

### Turn 11 — `submit_completed_task` (`16518 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated schema verification, independent recurrence and derivative checks, EMI spike rejection, and all-channel calibration invariance. The 17 highest-stress complete coupons remained below the regime threshold and had held-out 90th-percentile absolute errors of 0.05732 for strain and 0.03661 for onset rate.
- **Alternatives Weighed**: `Replace equal-cell imputation with within-cell imputation` (rejected: It improved tail errors but worsened median strain error on the 13 missing-feature coupons and does not preserve equal weighting across cells.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/alpha_m97_c02_s00_solver.py': 'FILE:7423'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_m97_c02_s00_solver.py`. The isolated schema validator passed. Independent calculations confirmed the memory recurrence and onset-rate prediction, and equivalent load-cell calibrations changed predictions by less than 1.6×10⁻¹⁰. Coupons with missing features remained the hardest held-out slice.`

### Turn 12 — `execute_in_remote_docker` (`80029 ms` LLM)
- **Rationale**: Fit with and without imputed rows in each held-out lab fold while retaining the prescribed imputation for prediction. A consistent reduction in worst-slice errors would support changing the training fit. Test the documented sparse-calibration branch separately: any failure to recover archived coefficients, or any archive read with clean calibration, calls for a calibration fix.
- **Alternatives Weighed**: `Expand the flow basis with nonlinear or relay terms` (rejected: The specified envelope is linear, pairwise linear sums and differences were verified to lie in the existing span, and there is no relay or stochastic-control state.)

```bash
set -e
cat > /tmp/rheo_final_audit.py <<'PY'
import importlib.util
import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'alpha_m97_c02_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root / 'public_alpha_m97_c02_s00.pkl').open('rb') as stream:
    d = pickle.load(stream)[0]['dataset']
tr = d['train']
lab = tr['lab_id']
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
n = len(lab)
p, u, f, s0 = m._prepare(d)
dt, t = d['past_dt'], d['eval_times']
y = tr['strain_curves_obs'] - s0[:n, None]
r = tr['instant_rate_obs']
results = {}
for complete_only in (False, True):
    ec, er = np.empty_like(y), np.empty_like(r)
    taus = []
    for c in np.unique(lab):
        keep = lab != c
        if complete_only:
            keep &= ~missing
        a = np.flatnonzero(keep)
        b = np.flatnonzero(lab == c)
        lt, w = m._fit(p[a], u[a], f[a], dt, t, y[a], r[a])
        C, R, *_ = m._basis(lt, p[b], u[b], f[b], dt, t)
        ec[b], er[b] = y[b] - C @ w, r[b] - R @ w
        taus.append(np.exp(lt))
    results[complete_only] = ec, er
    print('complete-only training:', complete_only, 'fold tau span:', np.ptp(taus, axis=0))
    for name, sel in [('missing', missing), ('complete', ~missing)]:
        print(name, 'strain and rate median/p90:', np.quantile(abs(ec[sel]), [.5,.9]), np.quantile(abs(er[sel]), [.5,.9]))
    print('largest lab/horizon absolute strain median:', max(np.median(abs(ec[lab == c]),axis=0).max() for c in np.unique(lab)))
a, b = results[False][0], results[True][0]
print('missing coupon horizon-median absolute error decreased on', int(np.count_nonzero(np.median(abs(b[missing]), axis=1) < np.median(abs(a[missing]), axis=1))), 'of', missing.sum())

cal = d['calibration_load_cells']
base = m._calibrate(cal)
with TemporaryDirectory() as tmp:
    m._ARCHIVE = Path(tmp) / 'archive_calibration_history.json'
    m._ARCHIVE.write_text(json.dumps({'channels': {str(c): {'gain': float(g), 'bias': float(b)} for c, (g,b) in enumerate(base)}}))
    for count in (0, 1):
        mask = cal['lab_id'] != 2
        mask[np.flatnonzero(~mask)[:count]] = True
        np.testing.assert_allclose(m._calibrate({k: v[mask] for k, v in cal.items()}), base, rtol=0, atol=1e-12)
    m._ARCHIVE.write_text('{}invalid')
    np.testing.assert_allclose(m._calibrate(cal), base, rtol=0, atol=1e-12)
print('Sparse channel uses calibration archive; clean channels do not read it.')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rheo_final_audit.py
```
- **Container Exit Code**: `0` (`1211 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
complete-only training: False fold tau span: [0.0032511  0.20820631]
missing strain and rate median/p90: [0.12479211 0.70218357] [0.06570893 0.18062131]
complete strain and rate median/p90: [0.01511691 0.06946371] [0.01394186 0.03439176]
largest lab/horizon absolute strain median: 0.05656924461196855
complete-only training: True fold tau span: [0.00341684 0.19967534]
missing strain and rate median/p90: [0.12425649 0.70167542] [0.06726803 0.18143396]
complete strain and rate median/p90: [0.01523811 0.0682237 ] [0.01416436 0.03462121]
largest lab/horizon absolute strain median: 0.05585917332555029
missing coupon horizon-median absolute error decreased on 5 of 13
Sparse channel uses calibration archive; clean channels do not read it.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c4caea57c8e0695d (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c4caea57 (directory).
```

### Turn 13 — `submit_completed_task` (`17942 ms` LLM)
- **Rationale**: The verified artifact matches the specified linear-flow model. Stress-slice, calibration-invariance, EMI-rejection, and sparse-channel checks passed; the final held-out comparison did not support changing the fit.
- **Alternatives Weighed**: `Train only on coupons with complete features` (rejected: On missing-feature coupons, strain p90 improved by only 0.00051, onset-rate p90 worsened by 0.00081, and median absolute strain error across horizons improved on only 5 of 13 coupons.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/alpha_m97_c02_s00_solver.py`. The isolated schema validator passed, and independent checks confirmed the memory recurrence, onset-rate predictions, and calibration fallback. Coupons with missing features remained the hardest held-out group.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M97-NonNested-Counterfactual-Lineage-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
