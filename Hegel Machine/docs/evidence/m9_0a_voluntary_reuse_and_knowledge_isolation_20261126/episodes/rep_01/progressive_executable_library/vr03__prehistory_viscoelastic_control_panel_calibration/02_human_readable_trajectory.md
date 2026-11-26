# Dialectical Synthesis Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_2_Selective_LiveGPT` | `vr03__prehistory_viscoelastic_control_panel_calibration`

- **Official Strict Pass**: **`PASS (1.0)`** (`PASSED_ALL_CHECKS`)
- **Blind Public Synthesis Choice**: `antithesis` via `selective_router_confident_single_branch`
- **Synthesis Rationale**: Primary Antithesis branch satisfied public checks under policy; second branch skipped (`1.00u` nominal units).
- **Post-Lock Branch Verdicts**: Thesis (`Bare`) = `NOT_SPAWNED (0T)` | Antithesis (`Wormhole`) = `True` (10T)

## Selected Branch Trajectory Record
# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `vr03__prehistory_viscoelastic_control_panel_calibration`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M90a-VoluntaryReuse-4` |
| **Domain** | `Viscoelastic Hereditary Memory Extraction, Reference Control Panel Alignment & Simpson-Deconfounded Risk` |
| **Wave / Wormhole Family** | `m90a_voluntary_reuse` / `fam_cohort_deconfounding_and_reference_balancing` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `10 / 28` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `7` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `314,244` (`303,671` / `27,136` / `10,573`) |
| **Total Episode Wall Time (s)** | `346.06s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`16029 ms` LLM)
- **Rationale**: Inspect the public schema and available truth fields to establish the input contract and feasible validation slices; check operator contracts before deciding whether they match this problem.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle
import numpy as np
with open('public_viscoelastic_cohort.pkl','rb') as f: data=pickle.load(f)
def inspect(x,path='data',depth=0):
    if isinstance(x,dict):
        print(path, 'dict keys', list(x)[:30]);
        if depth<5:
            for k,v in list(x.items())[:20]: inspect(v,path+'.'+str(k),depth+1)
    elif isinstance(x,(list,tuple)):
        print(path,type(x).__name__,len(x))
        if x and depth<5: inspect(x[0],path+'[0]',depth+1)
    elif isinstance(x,np.ndarray):
        print(path,x.shape,x.dtype, x.ravel()[:8] if x.dtype!=object else '')
    else: print(path,type(x).__name__,str(x)[:100])
inspect(data)
PY
python3 -m hegel_wormhole_skills --describe autogenous_prim_reference_panel_balancing.align_reference_panel
python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit
```
- **Container Exit Code**: `0` (`692 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
its lower bound.",
        "The step fit requires aligned length-n vectors and a nonempty integer vector of candidate split indices, each leaving at least three samples on either side. Each candidate design must have full column rank: the predictor cannot be constant within both segments. The scale and loss_clip are positive finite scalars. Interpretation as a level step assumes a shared local trend on both sides."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {},
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
      ]
    }
  ]
}
```

### Turn 2 — `execute_in_remote_docker` (`44850 ms` LLM)
- **Rationale**: Measure extraction error and held-out calibration separately to decide whether the next change belongs in the trace fit, panel alignment, or risk model. Public truth is used only for diagnostics.
- **Alternatives Weighed**: `OLS extraction` (rejected: Retained as a diagnostic comparator; large telemetry glitches can bias all three fitted moduli.); `Include center intercepts` (rejected: Not adopted by default; compare against the specified shared intercept on held-out centers before adding a nuisance term that cannot be measured directly at query time.)

```bash
cd /holdout_task && cat > /tmp/visco_probe.py <<'PY'
import pickle
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def design(tr, taus):
    p,u = np.asarray(tr['past_timestamps']), np.asarray(tr['past_strain_history'])
    t,x = np.asarray(tr['timestamps']), np.asarray(tr['strain_u'])
    m = np.full(len(taus),u[0],dtype=float)
    out = np.empty((len(t),len(taus)))
    dec = -np.expm1(-np.diff(np.r_[p,t])[:,None]/taus)
    for d,v in zip(dec[:len(p)-1],u[1:]): m += d*(v-m)
    for k,(d,v) in enumerate(zip(dec[len(p)-1:],x)):
        m += d*(v-m)
        out[k] = m
    return np.column_stack((np.ones(len(t)),x,out))

def fit(A,y):
    b = np.linalg.lstsq(A,y,rcond=None)[0]
    for _ in range(14):
        r = y-A@b
        s = max(.015, 1.4826*np.median(abs(r-np.median(r))))
        w = 1/np.hypot(1,r/(2*s))
        b = np.linalg.lstsq(w[:,None]*A,w*y,rcond=None)[0]
    return b

def extract(part,taus):
    raw, ols, details = [],[],[]
    for tr in part['traces']:
        A = design(tr,taus); y=np.asarray(tr['stress_obs']); b=fit(A,y)
        raw.append(b[1:]); ols.append(np.linalg.lstsq(A,y,rcond=None)[0][1:]); r=y-A@b
        details.append([np.linalg.cond(A),np.median(abs(r)),np.sum(abs(r)>1)])
    return np.array(raw),np.array(ols),np.array(details)

def aligned(raw,part):
    z=raw.copy()
    z[part['missing_channel_mask']]=np.nan
    for c in np.unique(part['center_id']):
        i=part['center_id']==c; ctl=part['controls'][i]
        z[i]=(z[i]-np.nanmedian(np.nanmean(ctl,axis=0)))/np.nanmedian(np.nanstd(ctl,axis=0))
    return z

def features(x,s,c,nuisance):
    S=np.eye(3)[s,1:]
    return np.column_stack((x,S,np.eye(4)[c,1:] if nuisance and c is not None else np.zeros((len(x),3)) if nuisance else np.empty((len(x),0))))

def predict(z,s,c,y,q,sq,nuisance=False):
    _,inv,cnt=np.unique(np.c_[c,s],axis=0,return_inverse=True,return_counts=True)
    w=1/cnt[inv];w/=w.mean()
    f=np.nansum(z*w[:,None],axis=0)/np.sum(np.isfinite(z)*w[:,None],axis=0)
    x=np.where(np.isfinite(z),z,f); q=np.where(np.isfinite(q),q,f)
    # Symmetric penalty on center offsets via all four one-hot columns.
    a=features(x,s,None,False); b=features(q,sq,None,False)
    if nuisance:
        G=np.eye(4)[c];g=np.unique(c);a=np.c_[a,G];b=np.c_[b,np.tile(np.isin(np.arange(4),g)/len(g),(len(q),1))]
    m=LogisticRegression(C=10,max_iter=500).fit(a,y,sample_weight=w)
    return m.predict_proba(b)[:,1],q,m.coef_[0],m.intercept_[0]

def scores(y,p): return [round(roc_auc_score(y,p),4),round(np.mean((y-p)**2),4)]
if __name__=='__main__':
    for k,d in enumerate(pickle.load(open('/holdout_task/public_viscoelastic_cohort.pkl','rb'))):
        co,gt=d['cohort'],d['gt']; tr,qu=co['train'],co['query']; t=np.asarray(co['taus'])
        rt,ot,dt=extract(tr,t); rq,oq,dq=extract(qu,t)
        z,q=aligned(rt,tr),aligned(rq,qu); ol=aligned(oq,qu)
        c,s,y=tr['center_id'],tr['stratum'],tr['y']; cq,sq=qu['center_id'],qu['stratum']; truth=gt['Z_query']; yq=gt['y_query']
        print('cohort',k,'trace diagnostic cond, median abs residual, glitch count',np.percentile(np.r_[dt,dq],[50,95,100],axis=0).round(4).tolist())
        print('per center stats means, prevalence, strata fraction')
        for zi,si,ci,yi in [(z,s,c,y),(q,sq,cq,yq)]:
            for j in np.unique(ci):
                i=ci==j; print(j,np.nanmean(zi[i],axis=0).round(3),round(yi[i].mean(),3),np.bincount(si[i])/i.sum())
        logit=np.log(gt['prob_query'])-np.log1p(-gt['prob_query']); print('query true nonmodulus logit by center/stratum',[[round(np.mean((logit-truth@gt['beta'])[(cq==j)&(sq==h)]),3) for h in range(3)] for j in np.unique(cq)])
        print('nonmissing aligned robust / OLS MAE',np.nanmean(abs(q-truth)),np.nanmean(abs(ol-truth)))
        for nuis in [False,True]:
            p,qf,b,a=predict(z,s,c,y,q,sq,nuis)
            print('center nuisance',nuis,'coef/intercept',b.round(3),round(a,3),'MAE',np.mean(abs(qf-truth)),'query pooled',scores(yq,p),'query centers',[scores(yq[cq==j],p[cq==j]) for j in np.unique(cq)],'loco',[scores(y[c==j],predict(z[c!=j],s[c!=j],c[c!=j],y[c!=j],z[c==j],s[c==j],nuis)[0]) for j in np.unique(c)])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/visco_probe.py
```
- **Container Exit Code**: `0` (`1657 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
06, 4.0], [12.5835, 0.0143, 4.0], [13.5296, 0.0163, 4.0]]
per center stats means, prevalence, strata fraction
0 [0.453 0.421 0.263] 0.528 [0.48611111 0.30555556 0.20833333]
1 [0.344 0.64  0.335] 0.458 [0.56944444 0.23611111 0.19444444]
2 [-0.454 -0.525 -0.614] 0.347 [0.48611111 0.25       0.26388889]
3 [-0.348 -0.419 -0.441] 0.389 [0.48611111 0.33333333 0.18055556]
4 [0.277 0.267 0.224] 0.489 [0.58888889 0.26666667 0.14444444]
5 [-0.618 -0.32  -0.471] 0.278 [0.63333333 0.23333333 0.13333333]
query true nonmodulus logit by center/stratum [[-0.9, -0.0, 0.85], [-0.9, -0.0, 0.85]]
nonmissing aligned robust / OLS MAE 0.03486363146152215 1.2607119610749535
center nuisance False coef/intercept [ 2.133 -1.757  1.758  0.075  1.509] -0.711 MAE 0.07206148674923443 query pooled [0.9462, 0.0922] query centers [[0.9575, 0.0874], [0.9274, 0.097]] loco [[0.9404, 0.1001], [0.9378, 0.1087], [0.9115, 0.1119], [0.9099, 0.1184]]
center nuisance True coef/intercept [ 2.271 -1.737  1.861  0.072  1.593 -0.049 -0.785  0.017  0.253] -0.616 MAE 0.07206148674923443 query pooled [0.9462, 0.0919] query centers [[0.9555, 0.0867], [0.9274, 0.0972]] loco [[0.942, 0.0971], [0.9378, 0.1095], [0.9123, 0.1124], [0.9091, 0.1196]]
cohort 1 trace diagnostic cond, median abs residual, glitch count [[10.6422, 0.0105, 4.0], [12.7642, 0.0146, 4.0], [14.1938, 0.0179, 4.0]]
per center stats means, prevalence, strata fraction
0 [0.674 0.859 0.643] 0.569 [0.58333333 0.26388889 0.15277778]
1 [0.505 0.461 0.921] 0.625 [0.5        0.33333333 0.16666667]
2 [-0.622 -0.53  -0.337] 0.389 [0.45833333 0.375      0.16666667]
3 [-0.55  -0.398 -0.341] 0.361 [0.54166667 0.30555556 0.15277778]
4 [0.429 0.601 0.32 ] 0.444 [0.57777778 0.28888889 0.13333333]
5 [-0.328 -0.297 -0.402] 0.4 [0.55555556 0.24444444 0.2       ]
query true nonmodulus logit by center/stratum [[-0.9, 0.0, 0.85], [-0.9, 0.0, 0.85]]
nonmissing aligned robust / OLS MAE 0.030313533017751703 1.4034934620217423
center nuisance False coef/intercept [ 1.664 -1.362  1.415  1.285  1.054] -0.847 MAE 0.08726911400750094 query pooled [0.9118, 0.1155] query centers [[0.8835, 0.1375], [0.93, 0.0934]] loco [[0.9079, 0.1253], [0.972, 0.0741], [0.8912, 0.1288], [0.8679, 0.1452]]
center nuisance True coef/intercept [ 1.763 -1.319  1.49   1.302  1.089 -0.558 -0.208  0.261 -0.158] -0.727 MAE 0.08726911400750094 query pooled [0.9116, 0.1152] query centers [[0.8815, 0.1378], [0.9316, 0.0927]] loco [[0.9079, 0.125], [0.9728, 0.0747], [0.8912, 0.1302], [0.8637, 0.1483]]
```

### Turn 3 — `execute_in_remote_docker` (`33226 ms` LLM)
- **Rationale**: Test whether refitting after removing clearly separated glitches improves extraction, and use raw-coordinate and oracle comparisons to determine whether the narrow ranking margin comes from scanner confounding, stratum estimation, or label variability.
- **Alternatives Weighed**: `Unadjusted OLS trace extraction` (rejected: Observed-channel MAE is 1.26–1.40 versus 0.030–0.035 for the robust fit on the same ZOH basis.); `Center intercepts` (rejected: They slightly improve pooled Brier but reduce worst query-center and LOCO AUROC; the specified common biological intercept transfers without estimating a new center term.)

```bash
cat > /tmp/visco_compare.py <<'PY'
from visco_probe import *

def extract_refit(part, taus):
    out=[]
    for trace in part['traces']:
        a=design(trace,taus); y=np.asarray(trace['stress_obs']); b=fit(a,y)
        r=y-a@b; med=np.median(r); mad=1.4826*np.median(abs(r-med))
        keep=abs(r-med) <= max(.12, 7*mad)
        out.append(np.linalg.lstsq(a[keep],y[keep],rcond=None)[0][1:])
    return np.asarray(out)

def eval_fit(z,s,c,y,q,sq,penalty):
    _,inv,n=np.unique(np.c_[c,s],axis=0,return_inverse=True,return_counts=True)
    w=1/n[inv]; w/=w.mean()
    f=np.nansum(z*w[:,None],axis=0)/np.sum(np.isfinite(z)*w[:,None],axis=0)
    x=np.where(np.isfinite(z),z,f); q=np.where(np.isfinite(q),q,f)
    a=np.c_[x,np.eye(3)[s]*penalty]; b=np.c_[q,np.eye(3)[sq]*penalty]
    m=LogisticRegression(C=10,max_iter=500,tol=1e-8).fit(a,y,sample_weight=w)
    return m.predict_proba(b)[:,1],q

for k,d in enumerate(pickle.load(open('/holdout_task/public_viscoelastic_cohort.pkl','rb'))):
    co,gt=d['cohort'],d['gt']; tr,qu=co['train'],co['query']; taus=np.asarray(co['taus'])
    rt=extract_refit(tr,taus); rq=extract_refit(qu,taus)
    z,q=aligned(rt,tr),aligned(rq,qu)
    c,s,y=tr['center_id'],tr['stratum'],tr['y']; cq,sq=qu['center_id'],qu['stratum']; t=gt['Z_query']; yq=gt['y_query']
    print('cohort',k,'refit observed MAE per center/column',[[round(v,5) for v in np.nanmean(abs(q[cq==j]-t[cq==j]),axis=0)] for j in np.unique(cq)])
    print('oracle test centers',[scores(yq[cq==j],gt['prob_query'][cq==j]) for j in np.unique(cq)])
    for name,zt,qq,scale in [('aligned',z,q,1.),('stratum-shrink',z,q,.25),('raw',np.where(tr['missing_channel_mask'],np.nan,rt),np.where(qu['missing_channel_mask'],np.nan,rq),1.)]:
        p,qf=eval_fit(zt,s,c,y,qq,sq,scale)
        print(name,'query',scores(yq,p),[scores(yq[cq==j],p[cq==j]) for j in np.unique(cq)],'LOCO',[scores(y[c==j],eval_fit(zt[c!=j],s[c!=j],c[c!=j],y[c!=j],zt[c==j],s[c==j],scale)[0]) for j in np.unique(c)],'max center MAE',max(np.mean(abs(qf[cq==j]-t[cq==j])) for j in np.unique(cq)))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/visco_compare.py
```
- **Container Exit Code**: `0` (`2000 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cohort 0 refit observed MAE per center/column [[0.04804, 0.05482, 0.04849], [0.01512, 0.01903, 0.02321]]
oracle test centers [[0.9605, 0.0851], [0.9489, 0.0829]]
aligned query [0.9465, 0.0921] [[0.9575, 0.0872], [0.928, 0.0971]] LOCO [[0.9404, 0.1002], [0.9378, 0.1085], [0.9123, 0.1119], [0.9091, 0.1181]] max center MAE 0.07510277951686695
stratum-shrink query [0.947, 0.092] [[0.957, 0.0884], [0.9311, 0.0957]] LOCO [[0.9404, 0.1004], [0.9394, 0.1078], [0.9123, 0.1129], [0.9099, 0.1191]] max center MAE 0.07510277951686695
raw query [0.7765, 0.1882] [[0.8785, 0.1721], [0.8228, 0.2042]] LOCO [[0.7353, 0.2175], [0.7257, 0.2348], [0.8162, 0.1691], [0.75, 0.214]] max center MAE 3.855563831722351
cohort 1 refit observed MAE per center/column [[0.02323, 0.0248, 0.02828], [0.03395, 0.03338, 0.0368]]
oracle test centers [[0.909, 0.1195], [0.945, 0.0765]]
aligned query [0.9117, 0.1153] [[0.8835, 0.1373], [0.9295, 0.0934]] LOCO [[0.9072, 0.1258], [0.972, 0.0743], [0.892, 0.1286], [0.8687, 0.1451]] max center MAE 0.08815550208857735
stratum-shrink query [0.9103, 0.1163] [[0.8795, 0.1388], [0.93, 0.0939]] LOCO [[0.9087, 0.1223], [0.9753, 0.0734], [0.8888, 0.1296], [0.8629, 0.1462]] max center MAE 0.08815550208857735
raw query [0.7406, 0.2056] [[0.8005, 0.1962], [0.8009, 0.2151]] LOCO [[0.7309, 0.21], [0.7671, 0.1959], [0.7865, 0.1834], [0.7508, 0.2077]] max center MAE 3.782678455141412
```

### Turn 4 — `execute_in_remote_docker` (`46174 ms` LLM)
- **Rationale**: Write the tested estimator with the specified missing-channel treatment and output contract. The next step is to import the exact artifact in an isolated process and verify each public query center, the training-only imputation, and the prehistory rollout.
- **Alternatives Weighed**: `Stronger stratum shrinkage` (rejected: It lowered worst query-center AUROC from 0.8835 to 0.8795 and worsened the worst training LOCO fold.); `Center-specific risk intercepts` (rejected: They worsened worst-center ranking and cannot be estimated for an unlabeled query center.)
- **ATMS NOGOOD DECLARED**: Refuted `Raw scanner coordinates support risk transfer between centers.` | Witness: `On public cohort 1, with the same additive basis and weighting, raw coordinates gave worst query-center AUROC 0.8005 and Brier 0.2151, versus 0.8835 and 0.1373 after control alignment.` | Blanket Diagnosis: `The scanner shifts and scales the modulus channels in a setting where centers also differ in prevalence; fitting in raw coordinates mixes scanner differences into the risk predictor.` | Sublated Pivot: `Use companion controls to estimate scanner offset and gain, then fit shared modulus slopes and biological stratum offsets with equal total training weight per center–stratum cell.`

```bash
cat > /holdout_task/viscoelastic_panel_calibration.py <<'PY'
"""Extract hereditary moduli and fit a scanner-aligned failure model."""

import numpy as np
from sklearn.linear_model import LogisticRegression


def _stress_design(trace, taus):
    past_t = np.asarray(trace['past_timestamps'], dtype=float)
    past_u = np.asarray(trace['past_strain_history'], dtype=float)
    t = np.asarray(trace['timestamps'], dtype=float)
    u = np.asarray(trace['strain_u'], dtype=float)
    # This includes the interval from the last prehistory sample to t[0].
    innovation = -np.expm1(-np.diff(np.r_[past_t, t])[:, None] / taus)
    state = np.full(taus.size, past_u[0])
    split = past_t.size - 1
    for rate, strain in zip(innovation[:split], past_u[1:]):
        state += rate * (strain - state)
    memory = np.empty((t.size, taus.size))
    for k, (rate, strain) in enumerate(zip(innovation[split:], u)):
        state += rate * (strain - state)
        memory[k] = state
    return np.column_stack((np.ones(t.size), u, memory))


def _trace_moduli(trace, taus):
    design = _stress_design(trace, taus)
    stress = np.asarray(trace['stress_obs'], dtype=float)
    coef = np.linalg.lstsq(design, stress, rcond=None)[0]
    # Adaptive Cauchy weights acquire the inlier fit before hard rejection.
    for _ in range(14):
        residual = stress - design @ coef
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        root_weight = 1.0 / np.hypot(1.0, residual / (2.0 * max(0.015, mad)))
        coef = np.linalg.lstsq(design * root_weight[:, None], stress * root_weight, rcond=None)[0]
    residual = stress - design @ coef
    deviation = np.abs(residual - np.median(residual))
    keep = deviation <= max(0.12, 7.0 * 1.4826 * np.median(deviation))
    refit, _, rank, _ = np.linalg.lstsq(design[keep], stress[keep], rcond=None)
    if rank == design.shape[1]:
        coef = refit
    return coef[1:]


def _aligned_moduli(partition, taus):
    moduli = np.array([_trace_moduli(trace, taus) for trace in partition['traces']])
    moduli[np.asarray(partition['missing_channel_mask'], dtype=bool)] = np.nan
    centers = np.asarray(partition['center_id'])
    controls = np.asarray(partition['controls'], dtype=float)
    for center in np.unique(centers):
        rows = centers == center
        panel = controls[rows]
        panel = panel[:, np.sum(np.isfinite(panel), axis=0) >= 2]
        means = np.nanmean(panel, axis=0)
        spreads = np.nanstd(panel, axis=0, ddof=0)
        offset = np.median(means)
        gain = np.median(spreads)
        moduli[rows] = (moduli[rows] - offset) / gain
    return moduli


def _cell_weights(center, stratum):
    _, cell, counts = np.unique(np.column_stack((center, stratum)), axis=0,
                                return_inverse=True, return_counts=True)
    weights = 1.0 / counts[cell.ravel()]
    return weights / weights.mean()


def calibrate_viscoelastic_cohort(cohort: dict) -> dict:
    """Return aligned query moduli, failure probabilities and biomarker slopes."""
    taus = np.asarray(cohort['taus'], dtype=float)
    train, query = cohort['train'], cohort['query']
    train_z = _aligned_moduli(train, taus)
    query_z = _aligned_moduli(query, taus)
    weights = _cell_weights(train['center_id'], train['stratum'])
    observed = np.isfinite(train_z)
    fill = (np.sum(np.where(observed, train_z, 0.0) * weights[:, None], axis=0)
            / np.sum(observed * weights[:, None], axis=0))
    train_z = np.where(observed, train_z, fill)
    query_z = np.where(np.isfinite(query_z), query_z, fill)

    # Keep the biological stratum offsets, but do not use scanner ID or balance
    # the labels themselves: the output is risk under the fitted class prior.
    strata = np.eye(3)
    train_features = np.column_stack((train_z, strata[np.asarray(train['stratum'])]))
    query_features = np.column_stack((query_z, strata[np.asarray(query['stratum'])]))
    model = LogisticRegression(C=10.0, max_iter=500, tol=1e-8)
    model.fit(train_features, train['y'], sample_weight=weights)
    probabilities = model.predict_proba(query_features)[:, 1]
    eps = np.finfo(float).eps
    return {
        'aligned_query_moduli': query_z,
        'query_probabilities': np.clip(probabilities, eps, 1.0 - eps),
        'modulus_weights': model.coef_[0, :3].copy(),
    }
PY
```
- **Container Exit Code**: `0` (`369 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoelastic_panel_calibration.py']`
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f2aa636eca504e70 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f2aa636e (directory).
```

### Turn 5 — `execute_in_remote_docker` (`41478 ms` LLM)
- **Rationale**: Verify the exact artifact rather than the development probe. A closed-form strain step detects a missed prehistory gap; independent cell aggregation detects incorrect imputation; query-center and leave-one-training-center-out results distinguish transfer failures from pooled performance.

```bash
cat > /tmp/verify_viscoelastic.py <<'PY'
import importlib.util
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

path = '/holdout_task/viscoelastic_panel_calibration.py'
spec = importlib.util.spec_from_file_location('viscoelastic_panel_calibration', path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
run = solver.calibrate_viscoelastic_cohort

# A constant history followed by a step gives a closed-form memory trajectory,
# including the gap between the prehistory and observation windows.
taus = np.array([0.41, 1.67])
t = np.cumsum(np.linspace(0.06, 0.17, 36)) + 0.29
u = np.full(t.size, 2.8)
tr = dict(past_timestamps=np.array([-1.52, -0.64, 0.0]),
          past_strain_history=np.full(3, -1.2), timestamps=t, strain_u=u)
expected = 2.8 - 4.0*np.exp(-t[:, None]/taus)
np.testing.assert_allclose(solver._stress_design(tr, taus)[:, 2:], expected, atol=2e-14, rtol=0)

with open('/holdout_task/public_viscoelastic_cohort.pkl', 'rb') as handle:
    data = pickle.load(handle)
worst = dict(auc=1.0, brier=0.0, mae=0.0, cell_mae=0.0, loco_auc=1.0, loco_brier=0.0)
for k, item in enumerate(data):
    cohort, gt = item['cohort'], item['gt']
    train, query = cohort['train'], cohort['query']
    out = run(cohort)
    assert set(out) == {'aligned_query_moduli', 'query_probabilities', 'modulus_weights'}
    z, p, beta = (out[key] for key in ('aligned_query_moduli', 'query_probabilities', 'modulus_weights'))
    n = len(query['center_id'])
    for arr, shape in [(z, (n, 3)), (p, (n,)), (beta, (3,))]:
        assert isinstance(arr, np.ndarray) and arr.shape == shape and arr.dtype.kind == 'f'
        assert np.isfinite(arr).all()
    assert np.all((p > 0) & (p < 1))
    mask = query['missing_channel_mask']
    train_z = solver._aligned_moduli(train, np.asarray(cohort['taus']))
    by_cell = [(train['center_id'] == c) & (train['stratum'] == s)
               for c in np.unique(train['center_id']) for s in np.unique(train['stratum'])]
    means_of_sums = np.sum([np.nansum(train_z[i], axis=0)/i.sum() for i in by_cell], axis=0)
    means_of_counts = np.sum([np.isfinite(train_z[i]).mean(axis=0) for i in by_cell], axis=0)
    expected_fill = means_of_sums/means_of_counts
    np.testing.assert_allclose(z[mask], np.broadcast_to(expected_fill, z.shape)[mask], atol=1e-12, rtol=0)
    for c in np.unique(query['center_id']):
        rows = query['center_id'] == c
        y = gt['y_query'][rows]
        auc = roc_auc_score(y, p[rows])
        brier = np.mean((p[rows]-y)**2)
        mae = np.mean(np.abs(z[rows]-gt['Z_query'][rows]))
        cell_mae = max(np.mean(np.abs(z[i]-gt['Z_query'][i])) for s in np.unique(query['stratum'][rows]) for i in [rows & (query['stratum'] == s)])
        print(f'public={k} center={c} AUROC={auc:.5f} Brier={brier:.5f} MAE={mae:.5f} max_stratum_MAE={cell_mae:.5f}')
        worst['auc'] = min(worst['auc'], auc)
        for name, value in [('brier', brier), ('mae', mae), ('cell_mae', cell_mae)]:
            worst[name] = max(worst[name], value)
    print(f'public={k} fitted slopes={beta.round(4)} reference={gt["beta"]}')
    def take(part, rows):
        return {key: [val[i] for i in np.flatnonzero(rows)] if key == 'traces' else val[rows] for key, val in part.items()}
    for c in np.unique(train['center_id']):
        test = train['center_id'] == c
        fold_query = take(train, test)
        y = fold_query.pop('y')
        pred = run(dict(taus=cohort['taus'], train=take(train, ~test), query=fold_query))['query_probabilities']
        auc, brier = roc_auc_score(y, pred), np.mean((pred-y)**2)
        worst['loco_auc'] = min(worst['loco_auc'], auc)
        worst['loco_brier'] = max(worst['loco_brier'], brier)
        print(f'public={k} training LOCO center={c} AUROC={auc:.5f} Brier={brier:.5f}')
print('Worst slices:', {k: round(v, 6) for k, v in worst.items()})
assert worst['auc'] >= 0.88 and worst['brier'] <= 0.160 and worst['mae'] <= 0.22
print('Verified target artifact with python3 -I')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_viscoelastic.py
```
- **Container Exit Code**: `0` (`4374 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public=0 center=4 AUROC=0.95751 Brier=0.08719 MAE=0.07510 max_stratum_MAE=0.08321
public=0 center=5 AUROC=0.92800 Brier=0.09711 MAE=0.06902 max_stratum_MAE=0.20366
public=0 fitted slopes=[ 2.1364 -1.7601  1.761 ] reference=[ 2.15 -1.95  1.8 ]
public=0 training LOCO center=0 AUROC=0.94040 Brier=0.10020
public=0 training LOCO center=1 AUROC=0.93784 Brier=0.10852
public=0 training LOCO center=2 AUROC=0.91234 Brier=0.11188
public=0 training LOCO center=3 AUROC=0.90909 Brier=0.11812
public=1 center=4 AUROC=0.88350 Brier=0.13732 MAE=0.08603 max_stratum_MAE=0.11478
public=1 center=5 AUROC=0.92953 Brier=0.09338 MAE=0.08816 max_stratum_MAE=0.11749
public=1 fitted slopes=[ 1.6659 -1.3592  1.4127] reference=[ 2.15 -1.95  1.8 ]
public=1 training LOCO center=0 AUROC=0.90716 Brier=0.12578
public=1 training LOCO center=1 AUROC=0.97202 Brier=0.07426
public=1 training LOCO center=2 AUROC=0.89205 Brier=0.12865
public=1 training LOCO center=3 AUROC=0.86873 Brier=0.14509
Worst slices: {'auc': 0.8835, 'brier': 0.137318, 'mae': 0.088156, 'cell_mae': 0.20366, 'loco_auc': 0.868729, 'loco_brier': 0.145085}
Verified target artifact with python3 -I

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f2aa636eca504e70 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f2aa636e (directory).
```

### Turn 6 — `submit_completed_task` (`13667 ms` LLM)
- **Rationale**: The exact artifact passed isolated-process verification of the output schema, prehistory rollout, and training-only imputation. Every public query center met all three thresholds.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`22` turns remaining; artifact status={'/holdout_task/viscoelastic_panel_calibration.py': 'FILE:4285'}): Do NOT lock prematurely with `22` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoelastic_panel_calibration.py`. Across the public query centers, the worst AUROC was 0.8835, the highest Brier score was 0.1373, and the highest aligned-modulus MAE was 0.0882. All three met the specified targets. The worst training leave-one-center-out AUROC was 0.8687.`

### Turn 7 — `execute_in_remote_docker` (`41825 ms` LLM)
- **Rationale**: Test whether the narrow AUROC margin reflects insufficient regularization or missing nonlinear terms, selecting regularization inside held-out-center folds. Compare raw and aligned fits on within-center balanced accuracy and both class recalls so pooled metrics cannot hide scanner confounding. Leave the verified artifact unchanged unless a challenger improves transfer.
- **Alternatives Weighed**: `Add explicit pairwise contrasts to the additive basis` (rejected: Sums and differences are already in the span of the three independent modulus terms. The audit tests products and bounded rational terms, which do add independent directions.); `Test drift operators or initial relay modes` (rejected: The specified system has two continuous memory states and no stochastic drift operator or relay state.)

```bash
cat > /tmp/audit_visco.py <<'PY'
import importlib.util
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location('calibration', '/holdout_task/viscoelastic_panel_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def features(z, s, basis):
    pair = np.column_stack([z[:, i]*z[:, j] for i, j in ((0,1),(0,2),(1,2))])
    extra = {'linear': (), 'quadratic': (z*z, pair),
             'rational': (z/(1+0.25*z*z), pair/(1+0.25*np.sum(z*z,axis=1,keepdims=True)))}[basis]
    return np.column_stack((z, np.eye(3)[s], *extra))

def predict(z, c, s, y, q, sq, basis, C):
    w = m._cell_weights(c, s)
    f = np.nansum(z*w[:,None],axis=0)/np.sum(np.isfinite(z)*w[:,None],axis=0)
    a = features(np.where(np.isfinite(z), z, f), s, basis)
    b = features(np.where(np.isfinite(q), q, f), sq, basis)
    return LogisticRegression(C=C,max_iter=500,tol=1e-8).fit(a,y,sample_weight=w).predict_proba(b)[:,1]

def scores(y,p):
    sensitivity = np.mean(p[y==1]>=.5)
    specificity = np.mean(p[y==0]<.5)
    return np.array([roc_auc_score(y,p), np.mean((y-p)**2), .5*(sensitivity+specificity), sensitivity, specificity])

for k, item in enumerate(pickle.load(open('/holdout_task/public_viscoelastic_cohort.pkl','rb'))):
    co, gt = item['cohort'], item['gt']
    tr, qu = co['train'], co['query']
    taus = np.asarray(co['taus'])
    z, q = (m._aligned_moduli(p,taus) for p in (tr,qu))
    raw, rawq = (np.where(p['missing_channel_mask'], np.nan, np.array([m._trace_moduli(t,taus) for t in p['traces']])) for p in (tr,qu))
    c,s,y = (tr[a] for a in ('center_id','stratum','y'))
    cq,sq,yq = qu['center_id'],qu['stratum'],gt['y_query']
    def loco(z, c, s, y, cfg):
        return np.array([scores(y[c==j],predict(z[c!=j],c[c!=j],s[c!=j],y[c!=j],z[c==j],s[c==j],*cfg)) for j in np.unique(c)])
    for label, a, b in [('raw',raw,rawq), ('aligned',z,q)]:
        p = predict(a,c,s,y,b,sq,'linear',10.)
        print(k,label,'pooled',scores(yq,p).round(4),'query center', [scores(yq[cq==j],p[cq==j]).round(4).tolist() for j in np.unique(cq)])
    cfgs = [(basis,C) for basis in ('linear','quadratic','rational') for C in (0.3,1.,3.,10.,30.)]
    for basis in ('linear','quadratic','rational'):
        cfg = [x for x in cfgs if x[0]==basis]
        outer, choices = [], []
        for j in np.unique(c):
            a = c!=j
            def key(x):
                h = loco(z[a],c[a],s[a],y[a],x)
                return (np.min(h[:,0]), -np.max(h[:,1]))
            best = max(cfg,key=key)
            choices.append(best[1]); outer.append(scores(y[~a],predict(z[a],c[a],s[a],y[a],z[~a],s[~a],*best)))
        outer = np.asarray(outer)
        print(k,basis,'nested C',choices,'worst AUC, Brier, BA, sensitivity, specificity', np.array([outer[:,0].min(),outer[:,1].max(),*outer[:,2:].min(axis=0)]).round(4))
    for C in (0.3,1.,3.,10.,30.):
        h = loco(z,c,s,y,('linear',C))
        p = predict(z,c,s,y,q,sq,'linear',C)
        g = np.array([scores(yq[cq==j],p[cq==j]) for j in np.unique(cq)])
        print(k,'linear',C,'LOCO + query worst AUC/Brier',*[round(v,5) for v in (h[:,0].min(),h[:,1].max(),g[:,0].min(),g[:,1].max())])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_visco.py
```
- **Container Exit Code**: `0` (`4813 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 raw pooled [0.7765 0.1882 0.726  0.6232 0.8288] query center [[0.8785, 0.1721, 0.7386, 0.4773, 1.0], [0.8228, 0.2042, 0.7938, 0.88, 0.7077]]
0 aligned pooled [0.9465 0.0921 0.8852 0.8696 0.9009] query center [[0.9575, 0.0872, 0.8997, 0.8864, 0.913], [0.928, 0.0971, 0.8662, 0.84, 0.8923]]
0 linear nested C [1.0, 3.0, 0.3, 0.3] worst AUC, Brier, BA, sensitivity, specificity [0.9099 0.1197 0.8174 0.7143 0.7949]
0 quadratic nested C [30.0, 0.3, 30.0, 30.0] worst AUC, Brier, BA, sensitivity, specificity [0.9002 0.1291 0.7962 0.7143 0.7949]
0 rational nested C [3.0, 3.0, 3.0, 30.0] worst AUC, Brier, BA, sensitivity, specificity [0.8937 0.1264 0.8068 0.7143 0.7949]
0 linear 0.3 LOCO + query worst AUC/Brier 0.9099 0.11974 0.93108 0.09775
0 linear 1.0 LOCO + query worst AUC/Brier 0.91153 0.11802 0.928 0.09681
0 linear 3.0 LOCO + query worst AUC/Brier 0.9099 0.11799 0.928 0.09695
0 linear 10.0 LOCO + query worst AUC/Brier 0.90909 0.11812 0.928 0.09711
0 linear 30.0 LOCO + query worst AUC/Brier 0.90909 0.11818 0.92738 0.09717
1 raw pooled [0.7406 0.2056 0.6817 0.6711 0.6923] query center [[0.8005, 0.1962, 0.685, 0.45, 0.92], [0.8009, 0.2151, 0.6991, 0.9167, 0.4815]]
1 aligned pooled [0.9117 0.1153 0.8376 0.8289 0.8462] query center [[0.8835, 0.1373, 0.78, 0.8, 0.76], [0.9295, 0.0934, 0.8935, 0.8611, 0.9259]]
1 linear nested C [3.0, 30.0, 0.3, 1.0] worst AUC, Brier, BA, sensitivity, specificity [0.8671 0.143  0.74   0.6538 0.8065]
1 quadratic nested C [0.3, 30.0, 30.0, 1.0] worst AUC, Brier, BA, sensitivity, specificity [0.8704 0.1445 0.74   0.6071 0.8261]
1 rational nested C [3.0, 0.3, 30.0, 0.3] worst AUC, Brier, BA, sensitivity, specificity [0.8704 0.1419 0.7581 0.6071 0.8065]
1 linear 0.3 LOCO + query worst AUC/Brier 0.86538 0.14178 0.8815 0.13863
1 linear 1.0 LOCO + query worst AUC/Brier 0.86706 0.14296 0.8825 0.13742
1 linear 3.0 LOCO + query worst AUC/Brier 0.8704 0.14433 0.884 0.13731
1 linear 10.0 LOCO + query worst AUC/Brier 0.86873 0.14509 0.8835 0.13732
1 linear 30.0 LOCO + query worst AUC/Brier 0.86873 0.14534 0.8835 0.13733

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f2aa636eca504e70 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f2aa636e (directory).
```

### Turn 8 — `submit_completed_task` (`15464 ms` LLM)
- **Rationale**: The follow-up audit confirmed that control alignment improved balanced accuracy in every public query center. No challenger materially improved transfer, and the unchanged deliverable passed isolated-process verification.
- **Alternatives Weighed**: `Quadratic or rational risk terms` (rejected: Neither gave consistent gains across the two public cohorts in nested leave-one-center-out validation.); `Stronger logistic regularization` (rejected: The best observed worst-query AUROC gain was 0.0005, with no material improvement in calibration.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`20` turns remaining; artifact status={'/holdout_task/viscoelastic_panel_calibration.py': 'FILE:4285'}): Do NOT lock prematurely with `20` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoelastic_panel_calibration.py`. Both public cohorts met all targets in every query center: worst AUROC 0.8835, highest Brier score 0.1373, and highest aligned-modulus MAE 0.0882. The worst training leave-one-center-out AUROC was 0.8687.`

### Turn 9 — `execute_in_remote_docker` (`25475 ms` LLM)
- **Rationale**: The open question is whether the narrow ranking margin or class recall depends on a single reference channel. Omit each control column from the real panels and check the resulting center metrics. If one omission causes a material failure, inspect that column's influence and test a physically justified calibration alternative before changing the estimator.
- **Alternatives Weighed**: `Repeat the nonlinear basis sweep` (rejected: Nested held-out-center validation already compared additive, quadratic, and rational terms; explicit pairwise linear contrasts add no span.)

```bash
cat > /tmp/visco_panel_jackknife.py <<'PY'
import importlib.util
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location('calibration', '/holdout_task/viscoelastic_panel_calibration.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
run = solver.calibrate_viscoelastic_cohort
with open('/holdout_task/public_viscoelastic_cohort.pkl', 'rb') as handle:
    data = pickle.load(handle)

for k, item in enumerate(data):
    co, gt = item['cohort'], item['gt']
    runs = []
    for omitted in (-1, 0, 1, 2, 3, 4):
        stress = dict(co)
        if omitted >= 0:
            for split in ('train', 'query'):
                part = co[split]
                stress[split] = dict(part, controls=np.delete(part['controls'], omitted, axis=1))
        out = run(stress)
        p, z = out['query_probabilities'], out['aligned_query_moduli']
        for c in np.unique(co['query']['center_id']):
            rows = co['query']['center_id'] == c
            y = gt['y_query'][rows]
            sens = np.mean(p[rows][y == 1] >= .5)
            specf = np.mean(p[rows][y == 0] < .5)
            runs.append((omitted, c, roc_auc_score(y, p[rows]), np.mean((p[rows] - y)**2), np.mean(abs(z[rows] - gt['Z_query'][rows])), (sens + specf)/2, sens, specf))
    print('cohort', k, 'rows: omitted control (-1 = full), center, AUROC, Brier, MAE, BA, sensitivity, specificity')
    for row in runs:
        print(tuple(round(float(v), 5) for v in row))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/visco_panel_jackknife.py && OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_viscoelastic.py
```
- **Container Exit Code**: `0` (`9635 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
7903, 0.86615, 0.84, 0.89231)
(3.0, 4.0, 0.95751, 0.08628, 0.06634, 0.88834, 0.86364, 0.91304)
(3.0, 5.0, 0.92923, 0.09654, 0.10693, 0.86615, 0.84, 0.89231)
(4.0, 4.0, 0.95603, 0.08694, 0.06605, 0.87698, 0.84091, 0.91304)
(4.0, 5.0, 0.92738, 0.09747, 0.07788, 0.88615, 0.88, 0.89231)
cohort 1 rows: omitted control (-1 = full), center, AUROC, Brier, MAE, BA, sensitivity, specificity
(-1.0, 4.0, 0.8835, 0.13732, 0.08603, 0.78, 0.8, 0.76)
(-1.0, 5.0, 0.92953, 0.09338, 0.08816, 0.89352, 0.86111, 0.92593)
(0.0, 4.0, 0.8835, 0.13759, 0.09788, 0.78, 0.8, 0.76)
(0.0, 5.0, 0.93056, 0.09311, 0.11797, 0.89352, 0.86111, 0.92593)
(1.0, 4.0, 0.882, 0.13863, 0.10525, 0.7925, 0.825, 0.76)
(1.0, 5.0, 0.93004, 0.09208, 0.07633, 0.89352, 0.86111, 0.92593)
(2.0, 4.0, 0.882, 0.13754, 0.08214, 0.79, 0.8, 0.78)
(2.0, 5.0, 0.93004, 0.09291, 0.11159, 0.89352, 0.86111, 0.92593)
(3.0, 4.0, 0.8825, 0.13805, 0.08223, 0.79, 0.8, 0.78)
(3.0, 5.0, 0.92953, 0.09251, 0.0752, 0.89352, 0.86111, 0.92593)
(4.0, 4.0, 0.8815, 0.1392, 0.11056, 0.7925, 0.825, 0.76)
(4.0, 5.0, 0.93056, 0.09279, 0.11503, 0.89352, 0.86111, 0.92593)
public=0 center=4 AUROC=0.95751 Brier=0.08719 MAE=0.07510 max_stratum_MAE=0.08321
public=0 center=5 AUROC=0.92800 Brier=0.09711 MAE=0.06902 max_stratum_MAE=0.20366
public=0 fitted slopes=[ 2.1364 -1.7601  1.761 ] reference=[ 2.15 -1.95  1.8 ]
public=0 training LOCO center=0 AUROC=0.94040 Brier=0.10020
public=0 training LOCO center=1 AUROC=0.93784 Brier=0.10852
public=0 training LOCO center=2 AUROC=0.91234 Brier=0.11188
public=0 training LOCO center=3 AUROC=0.90909 Brier=0.11812
public=1 center=4 AUROC=0.88350 Brier=0.13732 MAE=0.08603 max_stratum_MAE=0.11478
public=1 center=5 AUROC=0.92953 Brier=0.09338 MAE=0.08816 max_stratum_MAE=0.11749
public=1 fitted slopes=[ 1.6659 -1.3592  1.4127] reference=[ 2.15 -1.95  1.8 ]
public=1 training LOCO center=0 AUROC=0.90716 Brier=0.12578
public=1 training LOCO center=1 AUROC=0.97202 Brier=0.07426
public=1 training LOCO center=2 AUROC=0.89205 Brier=0.12865
public=1 training LOCO center=3 AUROC=0.86873 Brier=0.14509
Worst slices: {'auc': 0.8835, 'brier': 0.137318, 'mae': 0.088156, 'cell_mae': 0.20366, 'loco_auc': 0.868729, 'loco_brier': 0.145085}
Verified target artifact with python3 -I

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f2aa636eca504e70 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f2aa636e (directory).
```

### Turn 10 — `submit_completed_task` (`16050 ms` LLM)
- **Rationale**: Panel alignment improved within-center balanced accuracy in every public query center. The reference-channel jackknife found no failure requiring a model change, and the exact saved artifact passed isolated-process verification.
- **Alternatives Weighed**: `Change the control-panel estimator` (rejected: Omitting any one reference channel preserved all three query-center targets; no single channel drove the result.); `Expand the additive risk basis` (rejected: Linear contrasts are already in the basis, and quadratic and rational terms did not consistently improve nested leave-one-center-out performance.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoelastic_panel_calibration.py`. Every public query center met all three targets: worst AUROC 0.8835, highest Brier score 0.1373, and highest aligned-modulus MAE 0.0882. All targets also held when omitting any one reference channel. The worst training leave-one-center-out AUROC was 0.8687.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-M90a-VoluntaryReuse-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/viscoelastic_panel_calibration.py": "FILE:4285"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "all_valid": true,
    "max_aligned_moduli_mae": 0.07203,
    "worst_within_center_auroc": 0.89609,
    "max_brier_score": 0.11673,
    "elapsed_sec": 1.7876
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"all_valid\": true, \"max_aligned_moduli_mae\": 0.07203, \"worst_within_center_auroc\": 0.89609, \"max_brier_score\": 0.11673, \"elapsed_sec\": 1.7876}\n__HEGEL_VERIFIER_AUTH_bccfe2f729f7cf26ffac0bb1c431bb2c__:{\"strict_pass\": true, \"all_valid\": true, \"max_aligned_moduli_mae\": 0.07203, \"worst_within_center_auroc\": 0.89609, \"max_brier_score\": 0.11673, \"elapsed_sec\": 1.7876}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 6995.68,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
