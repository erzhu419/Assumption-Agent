# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m102_a04_batch_06__blind_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel_M10_1_UnknownCut_DAG_Discovery` |
| **Domain** | `blind_multi_stage_scientific_computation_graph` |
| **Wave / Wormhole Family** | `m10_2_selective_bounded_dag_discovery_h16` / `m102_blind_dag` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `14 / 14` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `True` |
| **Total Tokens (Prompt / Cached / Out)** | `493,646` (`476,985` / `164,736` / `16,661`) |
| **Total Episode Wall Time (s)** | `487.46s` |

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

### Turn 1 — `execute_in_remote_docker` (`10889 ms` LLM)
- **Rationale**: Inspect the schema and permitted archive before choosing candidate graphs for the prescribed 52/20 split.

```bash
ls -la /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/public_asset_index.json /tmp/hegel_continual_archive/archive_manifest.json /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; cat /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`350 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
e_from_spec(spec)
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
            errs.append(f"missing:{k}"); continue
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
        n_tr = len(st["train"]["x"])
        s_idx = max(40, n_tr - 20)
        vx = st["train"]["x"][s_idx:].copy()
        vy = st["train"]["y"][s_idx:].copy()
        vt = st["train"]["traj"][s_idx:].copy()
        st_cv["train"] = {
            "x": st["train"]["x"][:s_idx].copy(),
            "y": st["train"]["y"][:s_idx].copy(),
            "traj": st["train"]["traj"][:s_idx].copy(),
            "summary": np.array([float(np.mean(st["train"]["y"][:s_idx])), float(np.std(st["train"]["y"][:s_idx])), float(np.mean(st["train"]["traj"][:s_idx, -1])), float(np.std(st["train"]["traj"][:s_idx, -1]))], dtype=float),
        }
        st_cv["test"] = {"x": vx}
        ocv = mod.solve_batch(st_cv)
        cy = np.asarray(ocv["y_hat"], dtype=float)
        ct = np.asarray(ocv["state_pred"], dtype=float)
        if cy.shape != vy.shape or ct.shape != vt.shape or not (np.all(np.isfinite(cy)) and np.all(np.isfinite(ct))):
            errs.append("cv_shape_or_nonfinite")
        else:
            ym = float(np.mean(np.abs(cy - vy)))
            tm = float(np.mean(np.abs(ct - vt)))
            if ym > 0.038:
                errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
            if tm > 0.038:
                errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")
print(json.dumps({"schema_valid": not errs, "errors": errs}, allow_nan=False))
sys.exit(1 if errs else 0)
```

### Turn 2 — `execute_in_remote_docker` (`11834 ms` LLM)
- **Rationale**: Inspect the archived equations, panel ranges, and trajectory spectrum before fitting candidate models.

```bash
cat /tmp/hegel_continual_archive/*solver.py; cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; python3 - <<'PY'
import numpy as np,pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
print('times',s['eval_times'],'telemetry',s['telemetry_vec']);
for k,d in s.items():
 if isinstance(d,dict):
  for n,a in d.items(): print(k,n,np.shape(a),np.min(a,axis=0),np.max(a,axis=0),'sample',a[:4]);
print('trajectory sv',np.linalg.svd(s['train']['traj'],compute_uv=False))
PY
```
- **Container Exit Code**: `0` (`376 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1 -5.3178 -7.6085 -5.9939] [7.4921 7.3659 5.7972 7.7737] sample [[-1.5575  0.2429 -0.5605  0.08  ]
 [ 1.6205  0.1076  0.4289 -0.3012]
 [-0.528  -1.1251 -0.3874  0.5531]
 [-0.4315  0.5912 -2.1051  0.779 ]]
panel_b u (24, 4) [-1.2365 -1.2162 -1.0941 -1.2037] [1.1814 1.2014 1.1392 1.0568] sample [[ 0.0331  0.3354 -0.6475  0.0841]
 [-0.5595 -0.2078 -0.7476  0.2368]
 [-1.2301 -1.0207  1.1392  0.6799]
 [ 1.1037  1.044  -0.3614  0.7499]]
panel_b v (24, 4) [-0.9028 -1.4772 -1.2405 -0.9409] [0.8315 1.8465 1.0442 1.117 ] sample [[ 0.1447 -0.4735 -0.0517  0.0235]
 [ 0.5755 -0.0254 -0.3801 -0.1549]
 [ 0.2823  1.8465 -0.0355  0.0904]
 [-0.0254 -0.9538  1.0442  0.4954]]
panel_c u (24, 4) [-1.1447 -1.2278 -1.1899 -1.0979] [1.2239 1.171  1.185  1.2102] sample [[-1.1447 -0.2658 -0.2874  1.1491]
 [ 0.6148  0.4123  0.9089  0.3663]
 [-0.5595  0.1434 -0.2508  0.618 ]
 [ 0.8099 -0.6204 -0.0911  0.2479]]
panel_c v1 (24, 4) [-1.3488 -1.4801 -0.8061 -1.3641] [1.5231 1.3743 1.0986 1.8609] sample [[-0.3465 -0.1007 -0.8061 -1.3641]
 [ 0.6803  0.2222 -0.242   0.3189]
 [-0.1101 -0.0024  0.0155 -0.3958]
 [ 0.2206  0.0818  0.6898  0.0231]]
panel_c v2 (24, 4) [-1.3194 -1.1794 -0.9021 -1.8285] [1.2706 1.3238 1.3748 1.4177] sample [[ 0.9207 -1.1794  0.5132  0.5033]
 [-0.8621  0.2029  0.8238 -0.3469]
 [-0.5563 -1.1207  0.0163  0.0752]
 [-0.7769 -0.2921 -0.0474  1.1189]]
train x (72, 4) [-1.8579 -1.7777 -1.8179 -2.0804] [1.4137 1.1926 1.6116 1.4722] sample [[-0.6824 -0.5963  1.5909 -0.0167]
 [ 0.4185  0.1377  0.2744  1.0572]
 [-1.0126 -0.1209  1.1681 -0.213 ]
 [ 0.8465 -0.5266  0.2345 -2.0804]]
train y (72,) -3.248112269304403 2.6409771521288836 sample [-0.9588 -0.7021 -0.7644 -0.8283]
train traj (72, 8) [-0.8157 -1.5762 -1.9629 -2.1662 -2.2705 -2.3918 -2.5368 -2.6751] [1.266  2.274  2.8563 3.2326 3.5082 3.7113 3.8796 4.0312] sample [[ 0.1911  0.5041  0.7357  0.9416  1.1026  1.2271  1.344   1.4313]
 [ 0.3182  0.5808  0.7609  0.864   0.9356  0.9931  1.0618  1.0954]
 [ 0.2343  0.5878  0.8735  1.0704  1.2338  1.3635  1.4638  1.5484]
 [-0.3614 -0.6467 -0.8345 -0.9918 -1.1093 -1.2148 -1.3009 -1.3863]]
train summary (4,) 0.010837304916756262 1.6695644844261461 sample [0.0108 1.0733 0.4954 1.6696]
test x (36, 4) [-1.405  -1.7948 -1.8465 -1.9521] [1.2901 1.319  1.5652 1.1993] sample [[ 0.8028 -0.0716  0.3139 -0.9491]
 [ 0.3527 -0.796   0.0445  0.8244]
 [-0.12    0.0223  0.0373 -0.5853]
 [ 0.8065  0.3099  0.274   0.7552]]
trajectory sv [31.6733  2.5758  0.5008  0.0521  0.0454  0.0408  0.0377  0.0358]
```

### Turn 3 — `execute_in_remote_docker` (`37836 ms` LLM)
- **Rationale**: Estimate robust calibration, compare compact temporal laws, and use multiple 52/20 splits to see which spatial feature families warrant mechanistic investigation.
- **Alternatives Weighed**: `Reuse the previous campaign solver unchanged` (rejected: The current panels and trajectory spectrum need to be checked before assuming the same graph.)

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust reference panel alignment' || true; cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; cat > /tmp/probe.py <<'PY'
import numpy as np,pickle,sys
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from itertools import combinations_with_replacement
sys.path.insert(0,'/tmp/hegel_continual_archive')
from conf_m102_a04_batch_00_solver import _align, _panel_maps
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; A=s['panel_a']; t=s['eval_times']; T=tr['traj']; Y=np.c_[tr['y'],T]; n=len(Y)
a,b=_align(A); z=(tr['x']-a)/b
np.set_printoptions(precision=5,suppress=True)
print('align',a,b,'resid quantiles',np.quantile(abs(A['v']-a-A['u']*b),[.2,.5,.7,.8,.9,.95,1],axis=0),'z ranges',z.min(0),z.max(0)); print('sv centered',np.linalg.svd(T-T.mean(0),compute_uv=False))
for p in ['panel_b','panel_c']:
 d=s[p]; u=d['u'];
 for k in d:
  if k=='u':continue
  v=d[k]; print(p,k,'corr',np.corrcoef(u.T,v.T)[:4,4:]);
  for deg in [1,2,3]:
   F=np.column_stack([np.ones(len(u))]+[np.prod(u[:,j],axis=1) for m in range(1,deg+1) for j in combinations_with_replacement(range(4),m)]); print('poly',deg,np.sqrt(np.mean((F@np.linalg.lstsq(F,v,rcond=None)[0]-v)**2,axis=0)))
for name,F in [('exp2',lambda r:-np.expm1(-t[:,None]*r)),('exp2lin',lambda r:np.c_[t,-np.expm1(-t[:,None]*r)]),('exp3',lambda r:-np.expm1(-t[:,None]*r)),('exp2const',lambda r:np.c_[np.ones(len(t)),-np.expm1(-t[:,None]*r)]),('expsin',lambda r:np.c_[-np.expm1(-t[:,None]*r[:2]),np.sin(t*r[2])])]:
 def fun(r):
  f=F(r);return (f@np.linalg.lstsq(f,T.T,rcond=None)[0]-T.T).ravel()
 r=least_squares(fun,[.4,1.8,1.1] if name in ['exp3','expsin'] else [.4,1.8],bounds=(.01,10)).x;print(name,r,np.linalg.norm(fun(r))/np.sqrt(T.size))
folds=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [np.random.default_rng(i).permutation(n) for i in range(5)]]
def cv(K,l):
 er=[]
 for a,b in folds:
  P=K[np.ix_(b,a)]@solve(K[np.ix_(a,a)]+l*np.eye(len(a)),Y[a],assume_a='pos');d=abs(P-Y[b]);er.append([d[:,0].mean(),d[:,1:].mean()])
 return np.array(er)
for name,X in [('z',z),('tanh',np.tanh(z)),('soft',z/(1+abs(z))),('z+soft',np.c_[z,z/(1+abs(z))]),('archC',_panel_maps(s['panel_c'],z))]:
 X=(X-X.mean(0))/np.maximum(.06,X.std(0)); G=X@X.T/X.shape[1]; D=cdist(X,X,'sqeuclidean')/X.shape[1]; out=[]
 for p in [1,2,3,4]:
  for amp in [0,.1,1]:
   for sc in ([.25,1,4] if amp else [1]):
    K=(1+.3*G)**p+amp*np.exp(-sc*D)
    for l in [1e-6,1e-4,.002,.02]:
     e=cv(K,l);out.append((e[1:].mean(),p,amp,sc,l,e[0],e[1:].max(0)))
 print(name, sorted(out,key=lambda row:row[0])[:3])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe.py
```
- **Container Exit Code**: `0` (`872 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[-0.39825 -0.59788  0.71198 -0.33516]
 [-0.17984 -0.65952  0.0229   0.79318]
 [-0.29589  0.77807  0.25646  0.17373]
 [ 0.69921  0.23965  0.49667  0.51803]]
poly 1 [0.00006 0.00008 0.00008 0.00009]
poly 2 [0.00005 0.00006 0.00005 0.00007]
poly 3 [0. 0. 0. 0.]
panel_c v1 corr [[-0.10239 -0.26576  0.24784  0.56099]
 [ 0.43359 -0.23588 -0.19822  0.01232]
 [ 0.39441  0.61316 -0.12283 -0.22438]
 [-0.01399 -0.029   -0.17624  0.18093]]
poly 1 [0.53658 0.42834 0.44498 0.59475]
poly 2 [0.00005 0.00003 0.00007 0.00007]
poly 3 [0. 0. 0. 0.]
panel_c v2 corr [[ 0.23527  0.20943  0.12661  0.03847]
 [ 0.38724  0.0665   0.04516 -0.1643 ]
 [-0.66331  0.36572 -0.13605 -0.17067]
 [ 0.36467 -0.15811  0.3352  -0.01753]]
poly 1 [0.50156 0.58925 0.60914 0.73947]
poly 2 [0.09147 0.13486 0.08763 0.09304]
poly 3 [0. 0. 0. 0.]
exp2 [0.32108 1.95895] 0.021512573852316252
exp2lin [1.20478 3.95024] 0.010218557238230579
exp3 [0.37633 5.53515 1.78968] 0.004451580777506369
exp2const [0.34602 1.77606] 0.0058053812419564
expsin [0.41409 3.11474 1.46355] 0.008909290241946467
z [(0.3984284186733388, 1, 0.1, 1, 1e-06, array([0.34165, 0.48046]), array([0.42225, 0.48486])), (0.39976582280299605, 1, 0.1, 1, 0.0001, array([0.33846, 0.48146]), array([0.42204, 0.48825])), (0.4110576423823834, 1, 1, 1, 0.0001, array([0.36129, 0.52052]), array([0.42538, 0.48968]))]
tanh [(0.3401913997156236, 4, 0.1, 1, 1e-06, array([0.32786, 0.43744]), array([0.52093, 0.41526])), (0.34596487508748525, 4, 0.1, 1, 0.0001, array([0.32848, 0.44122]), array([0.53319, 0.42457])), (0.34794928385350155, 4, 1, 1, 1e-06, array([0.35455, 0.45948]), array([0.48456, 0.37979]))]
soft [(0.34529294891581064, 2, 0.1, 1, 1e-06, array([0.38956, 0.40142]), array([0.4625 , 0.36062])), (0.3538238511085028, 4, 1, 1, 1e-06, array([0.39309, 0.41609]), array([0.4664 , 0.38656])), (0.3543423658068014, 4, 1, 1, 0.0001, array([0.38982, 0.41793]), array([0.46846, 0.38751]))]
z+soft [(0.353405098359458, 1, 1, 0.25, 0.0001, array([0.39978, 0.46223]), array([0.54221, 0.38437])), (0.3541065892707142, 1, 0.1, 0.25, 1e-06, array([0.38296, 0.41344]), array([0.52474, 0.3861 ])), (0.3550120184159441, 4, 1, 1, 1e-06, array([0.3865 , 0.42172]), array([0.47501, 0.38411]))]
archC [(0.5154298085959884, 2, 0.1, 1, 0.002, array([0.63889, 0.56733]), array([0.68993, 0.5685 ])), (0.5156708083804619, 1, 0.1, 0.25, 0.0001, array([0.6805 , 0.54611]), array([0.72614, 0.64304])), (0.5157460971102898, 2, 1, 0.25, 0.002, array([0.65695, 0.58059]), array([0.7825, 0.5864]))]
```

### Turn 4 — `execute_in_remote_docker` (`36635 ms` LLM)
- **Rationale**: Test the low-dimensional operator grammar against the calibration data directly before tuning downstream predictors.
- **ATMS NOGOOD DECLARED**: Refuted `The archive's coordinatewise calibration and panel-C basis transfer unchanged.` | Witness: `Panel-A median coordinate residuals are 0.128–0.209; panel-C v1 fits a full quadratic to about 0.00005 RMS.` | Blanket Diagnosis: `Omitted cross-coordinate calibration or operator terms can corrupt downstream latent coordinates.` | Sublated Pivot: `Fit a robust full affine map and test recovered polynomial panel maps separately and in composition.`

```bash
cat > /tmp/struct.py <<'PY'
import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
np.set_printoptions(precision=4,suppress=True,linewidth=160)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; T=tr['traj']; Y=np.c_[tr['y'],T]; n=len(Y)
def lin(a,b):return np.linalg.lstsq(a,b,rcond=1e-9)[0]
def poly(x,k):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for m in range(1,k+1) for j in itertools.combinations_with_replacement(range(x.shape[1]),m)])
u=s['panel_a']['u']; v=s['panel_a']['v']; f=poly(u,1); best=None
for p in np.random.default_rng(1).integers(len(u),size=(500,5)):
 c=lin(f[p],v[p]); r=np.linalg.norm(f@c-v,axis=1); score=np.median(r)
 if best is None or score<best[0]:best=(score,c)
c=best[1]
for _ in range(5):
 r=np.linalg.norm(f@c-v,axis=1); m=r<max(5*np.median(r),.0006);c=lin(f[m],v[m]);
print('A',c,'error',np.sort(r)); z=np.linalg.solve(c[1:].T,(tr['x']-c[0]).T).T
b=s['panel_b']; cb=lin(poly(b['u'],1),b['v']);print('B',cb); bx=poly(z,1)@cb
p=s['panel_c']; c1=lin(poly(p['u'],2),p['v1']);print('C1',c1); cx=poly(z,2)@c1
print('C2 quadratic',lin(poly(p['u'],2),p['v2']));
for name,f in [('cube',lambda x:np.c_[poly(x,2),x**3]),('sparse3',lambda x:np.c_[poly(x,2),np.prod(x[:,list(itertools.combinations(range(4),3))],axis=2)]),('trig',lambda x:np.c_[poly(x,2),np.sin(x),np.cos(x)])]:
 F=f(p['u']); C=lin(F,p['v2']); resid=(F@C-p['v2'])/(1-np.diag(F@np.linalg.pinv(F)))[:,None]; print(name,'loocv',np.mean(abs(resid),axis=0),C[-4:]);
Y0=Y.copy(); U,S,V=np.linalg.svd(T,full_matrices=False); Y=np.c_[Y[:,0],U[:,:3]*S[:3]]
folds=[(np.arange(52),np.arange(52,72))]+[(o[:52],o[52:]) for o in (np.random.default_rng(i).permutation(n) for i in range(5))]
def cv(K,l):
 e=[]
 for a,b in folds:
  P=K[np.ix_(b,a)]@solve(K[np.ix_(a,a)]+l*np.eye(len(a)),Y[a],assume_a='pos'); E=P-Y[b];e.append([np.mean(abs(E[:,0])),np.mean(abs(E[:,1:]@V[:3]))]);
 return np.array(e)
for name,X in [('z',z),('b',bx),('c1',cx),('c1(b)',poly(bx,2)@c1),('c1(x)',poly(tr['x'],2)@c1),('raw',tr['x']),('z+c1',np.c_[z,cx]),('b+c1(b)',np.c_[bx,poly(bx,2)@c1])]:
 print(name,'linear MAE',np.mean(abs(poly(X,1)@lin(poly(X,1),Y)-Y),0)); out=[]
 for tf in [lambda x:x,np.tanh,lambda x:x/(1+abs(x)),lambda x:np.c_[x,np.tanh(x)]]:
  X0=tf(X);X0=(X0-X0.mean(0))/np.maximum(.06,X0.std(0)); G=X0@X0.T/X0.shape[1];D=cdist(X0,X0,'sqeuclidean')/X0.shape[1]
  for j in [1,2,3]:
   for q,g in [(0,1),(.1,.25),(.1,1),(1,.25),(1,1)]:
    for l in [1e-6,1e-4,.002,.02]:
     e=cv((1+.4*G)**j+q*np.exp(-g*D),l);out.append((e[1:].mean(),j,q,g,l,e[0],e[1:].max(0)))
 print('best',sorted(out,key=lambda row:row[0])[:2])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/struct.py
```
- **Container Exit Code**: `1` (`382 ms`, timed_out=`False`)
- **Stderr (tail)**:
```text
/dist-packages/numpy/core/__init__.py", line 72, in <module>
    from . import numerictypes as nt
  File "/usr/local/lib/python3.12/dist-packages/numpy/core/numerictypes.py", line 105, in <module>
    from ._type_aliases import (
  File "/usr/local/lib/python3.12/dist-packages/numpy/core/_type_aliases.py", line 20, in <module>
    from numpy.compat import unicode
  File "/usr/local/lib/python3.12/dist-packages/numpy/compat/__init__.py", line 14, in <module>
    from . import py3k
  File "/usr/local/lib/python3.12/dist-packages/numpy/compat/py3k.py", line 26, in <module>
    import pickle
  File "/usr/lib/python3.12/pickle.py", line 33, in <module>
    from struct import pack, unpack
  File "/tmp/struct.py", line 2, in <module>
    from scipy.optimize import least_squares
  File "/usr/local/lib/python3.12/dist-packages/scipy/__init__.py", line 67, in <module>
    from scipy._lib import _pep440
  File "/usr/local/lib/python3.12/dist-packages/scipy/_lib/__init__.py", line 12, in <module>
    from scipy._lib._testutils import PytestTester
  File "/usr/local/lib/python3.12/dist-packages/scipy/_lib/_testutils.py", line 113, in <module>
    class _TestPythranFunc:
  File "/usr/local/lib/python3.12/dist-packages/scipy/_lib/_testutils.py", line 126, in _TestPythranFunc
    ALL_INTEGER = [np.int8, np.int16, np.int32, np.int64, np.intc, np.intp]
                   ^^^^^^^
AttributeError: partially initialized module 'numpy' has no attribute 'int8' (most likely due to a circular import)
```

### Turn 5 — `execute_in_remote_docker` (`12352 ms` LLM)
- **Rationale**: Rename the probe to stop it shadowing Python's standard-library struct module, then rerun the calibration and operator tests. The failed import produced no new scientific evidence.

```bash
mv /tmp/struct.py /tmp/panel_structure_probe.py; OPENBLAS_NUM_THREADS=1 python3 -I /tmp/panel_structure_probe.py
```
- **Container Exit Code**: `0` (`1474 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.8456 -0.3358]
 [ 0.1796 -0.3996 -0.1718  0.4459]
 [ 1.1635 -0.4092 -0.48   -0.8311]]
cube loocv [0.2299 0.7327 0.3945 0.2219] [[-0.2171  0.0997  0.0076 -0.2261]
 [ 0.0204  0.1995 -0.1502  0.0168]
 [ 0.1583 -0.3968  0.0739  0.4236]
 [-0.9038  0.0968  0.2861  0.0038]]
sparse3 loocv [0.558  0.8726 0.422  0.2808] [[ 0.0442 -0.0615  0.2168 -0.3377]
 [-0.0573  0.2639  0.6047 -0.3069]
 [ 0.1974  0.1765 -0.2126  0.7691]
 [ 0.6301  0.1067 -0.5807  0.3414]]
trig loocv [0.0791 0.0304 0.101  0.0342] [[-21.8146  -5.6047  -1.2837 -22.4087]
 [ -9.9421 -21.5547  18.8167  -9.7321]
 [  5.7593 -13.3345 -20.3993  17.0672]
 [-24.5157   4.2416   8.0734   8.5537]]
z linear MAE [0.7236 2.4354 0.2308 0.0327]
best [(0.29459473282196835, 3, 1, 1, 1e-06, array([0.2937, 0.3417]), array([0.4047, 0.3272])), (0.2947204907688588, 3, 0.1, 1, 1e-06, array([0.39  , 0.3287]), array([0.4602, 0.3309]))]
b linear MAE [0.7236 2.4354 0.2308 0.0327]
best [(0.37232785964659804, 1, 0.1, 1, 0.0001, array([0.2908, 0.4503]), array([0.4671, 0.4788])), (0.3751057231878725, 1, 0.1, 1, 1e-06, array([0.2781, 0.4469]), array([0.477 , 0.4729]))]
c1 linear MAE [0.7082 2.399  0.1482 0.0443]
best [(0.8302084434744404, 1, 0.1, 1, 0.02, array([0.8519, 1.1929]), array([0.7613, 1.0445])), (0.8487102944185333, 1, 0.1, 1, 0.02, array([0.8557, 1.2183]), array([0.8264, 1.0442]))]
c1(b) linear MAE [0.7311 2.5716 0.1881 0.0417]
best [(0.7700895752267114, 1, 0.1, 1, 0.02, array([0.8216, 0.9276]), array([0.796 , 0.9621])), (0.7759351003019224, 1, 0.1, 1, 0.02, array([0.7778, 0.9298]), array([0.808 , 0.9413]))]
c1(x) linear MAE [0.7509 2.2575 0.1539 0.0424]
best [(0.7360000112339723, 1, 0.1, 1, 0.02, array([0.7671, 1.0429]), array([0.7507, 0.9478])), (0.747665986723217, 1, 0.1, 1, 0.02, array([0.7784, 1.0287]), array([0.7905, 0.8971]))]
raw linear MAE [0.7236 2.4354 0.2308 0.0327]
best [(0.36975457816070245, 1, 0.1, 1, 1e-06, array([0.3565, 0.4876]), array([0.423 , 0.4494])), (0.3725977848093626, 2, 1, 1, 1e-06, array([0.3451, 0.4704]), array([0.4491, 0.4061]))]
z+c1 linear MAE [0.6282 2.0662 0.1475 0.03  ]
best [(0.280488304139871, 3, 0, 1, 0.0001, array([0.3019, 0.4228]), array([0.3602, 0.3712])), (0.2805007304423009, 3, 0.1, 0.25, 0.0001, array([0.3086, 0.4209]), array([0.3624, 0.3717]))]
b+c1(b) linear MAE [0.6614 2.2213 0.1808 0.0302]
best [(0.4790609687002762, 1, 0.1, 1, 1e-06, array([0.4096, 0.5739]), array([0.5076, 0.5982])), (0.4793435511052507, 1, 0.1, 1, 0.0001, array([0.4089, 0.5814]), array([0.5073, 0.6059]))]
```

### Turn 6 — `execute_in_remote_docker` (`33947 ms` LLM)
- **Rationale**: Use calibration-only leave-one-out errors to distinguish competing nonlinear maps and examine whether the scalar output shares the trajectory's recovered amplitudes.
- **Alternatives Weighed**: `Tune the existing spatial kernel` (rejected: The gap to the MAE gate is too large to justify tuning before checking missing calibration and operator terms.)

```bash
cat > /tmp/discover.py <<'PY'
import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.linalg import solve
np.set_printoptions(precision=5,suppress=True,linewidth=145)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; T=tr['traj']; x=tr['x']; y=tr['y']; n=len(y)
def lin(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def one(x):return np.c_[np.ones(len(x)),x]
def mon(x,d,rep=True):return np.array([np.prod(x[:,j],axis=1) for j in (itertools.combinations_with_replacement if rep else itertools.combinations)(range(4),d)]).T
def fit(F,V):
 C=lin(F,V);E=F@C-V; h=np.sum(F*np.linalg.pinv(F).T,axis=1);return np.sqrt(np.mean(E**2)),np.mean(abs(E)/np.maximum(1-h[:,None],1e-8)),C
A=s['panel_a']; u=A['u']; v=A['v']; keep=abs(v).max(1)<4;u=u[keep];v=v[keep]
print('A inliers',len(u));print('A linear residual',v-one(u)@lin(one(u),v))
for name,fun in [('poly2',lambda u:np.c_[one(u),mon(u,2)]),('cube',lambda u:np.c_[one(u),mon(u,2),u**3]),('poly3',lambda u:np.c_[one(u),mon(u,2),mon(u,3)]),('pair',lambda u:np.c_[one(u),mon(u,2,False)]),('trig',lambda u:np.c_[one(u),np.sin(1.25*u),np.cos(1.25*u)]),('soft',lambda u:np.c_[one(u),u/(1+.22*abs(u)),u*u]),('rational',lambda u:np.c_[one(u),u/(1+.22*u*u),u*u/(1+.22*u*u)]),('linear and all pairs''',lambda u:np.c_[one(u),mon(u,2,False),np.tanh(u),np.tanh(u)**2])]:
 F=fun(u);e,cv,C=fit(F,v); print('A',name,e,cv);print(C if cv<.03 else '')
p=s['panel_c'];u=p['u'];v=p['v2'];
for name,fun in [('sin',lambda u,k:np.sin(k*u)),('cos',lambda u,k:np.cos(k*u)),('sincos',lambda u,k:np.c_[np.sin(k*u),np.cos(k*u)]),('tanh',lambda u,k:np.tanh(k*u)),('expquad',lambda u,k:np.exp(-k*u*u)),('rational',lambda u,k:1/(1+k*u*u)),('exp',lambda u,k:np.exp(k*u)),('soft',lambda u,k:u/(1+k*abs(u)))]:
 for base in [lambda u:np.ones((len(u),1)),one,lambda u:np.c_[one(u),u*u],lambda u:np.c_[one(u),mon(u,2,False)]]:
  def feat(k):return np.c_[base(u),fun(u,k[0])];
  def res(k):
   F=feat(k);return (F@lin(F,v)-v).ravel()
  o=min([least_squares(res,[k],bounds=(.02,7)) for k in [.3,1.25,2.7,4.2]],key=lambda o:o.cost);e,cv,C=fit(feat(o.x),v); print('C2',name,base(u).shape[1],o.x,e,cv)
  if cv<.001: print(C)
def tm(r):return -np.expm1(-t[:,None]*r)
def res(r):
 F=tm(r);return (F@lin(F,T.T)-T.T).ravel()
r=least_squares(res,[.38,1.8,5.5],bounds=(.03,15)).x; H=lin(tm(r),T.T).T; print('time rates',r,'H ranges/std',H.min(0),H.max(0),H.std(0));print('y vs H',fit(one(H),y[:,None])[:2],fit(np.c_[one(H),H[:,[0,0,0,1,1,2]]*H[:,[0,1,2,1,2,2]]],y[:,None])[:2]);print(np.c_[x,y,H][:30])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/discover.py
```
- **Container Exit Code**: `0` (`771 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
148439063 0.9187192089362667
time rates [0.37633 1.78968 5.5352 ] H ranges/std [-3.93022 -2.3097  -0.61276] [4.19792 2.49446 0.51842] [1.81121 0.87803 0.21207]
y vs H (0.8272011058658565, 0.6782737588152308) (0.7756760070018909, 0.7713015268443879)
[[-0.68241 -0.59634  1.5909  -0.01669 -0.95879  1.8072   0.33594 -0.12215]
 [ 0.41854  0.13772  0.27438  1.05719 -0.70214  0.76351  0.48359  0.09716]
 [-1.01263 -0.12085  1.16815 -0.21304 -0.76436  1.66554  0.61969 -0.19114]
 [ 0.84648 -0.52662  0.23446 -2.08042 -0.82834 -1.47346 -0.15492 -0.23094]
 [ 1.04975 -0.46453  0.31625 -1.0392  -0.17207  2.02664  0.25725 -0.1558 ]
 [ 1.3496  -0.23759 -1.62659  0.24013  1.99385 -0.16399  0.65135 -0.0277 ]
 [ 0.82355 -1.55236  0.91873 -0.15235  2.16208  1.69087  0.9598   0.27772]
 [ 1.39891  0.3618  -0.61485 -0.13681  0.12331  4.19792  0.73496  0.06316]
 [-0.49107  0.64113 -0.44253 -1.30182 -2.95662 -2.8989   1.00637 -0.26032]
 [-0.05427  0.58259 -0.58508 -0.22983  0.17522  0.73694  0.37828 -0.04204]
 [ 0.65688 -0.73496  1.28739  0.66226 -0.2594   1.06387  1.02594  0.32584]
 [-0.57605  0.45704 -1.30846  0.22029 -0.19063  1.67122  0.32116 -0.02496]
 [ 1.20842 -0.09841  0.91037 -2.00147 -1.67938  0.36131 -1.08606 -0.23253]
 [-0.35539 -0.02656 -0.79653 -0.70404 -1.73752 -2.32111  1.16803 -0.27504]
 [-0.68443  1.00941  0.45488 -1.45746  0.29374 -2.42461 -0.02278 -0.08163]
 [ 0.50399  1.06058  1.20549  1.42958  1.65741 -1.51308  1.35605  0.46165]
 [ 0.47719  0.9662  -1.15342 -0.49086 -0.56514 -0.25707  0.52403 -0.17154]
 [-0.29546  0.80272  0.84815 -1.5467   0.82248 -2.76492 -0.74642 -0.03273]
 [-1.08638  0.72985  0.01582  0.99544  0.89341  1.00198 -0.16711  0.19489]
 [-0.07608  0.4617  -1.55241  0.88432  0.29781 -0.37665 -0.65942  0.13312]
 [ 0.4901   0.01951  0.86535 -1.89791 -0.39048 -0.80966 -0.63615 -0.1014 ]
 [ 0.51323 -0.26549 -0.45175  0.35842  0.68134  1.09384  0.18954  0.02964]
 [ 1.07472  0.17308  0.06107 -1.99793 -2.1929   1.03408 -0.3651  -0.12965]
 [ 0.73233 -0.87522  1.04979 -0.40293  0.77761  1.62457  0.70331  0.04818]
 [ 1.14211  0.17621 -0.97725  0.29861  0.6419   3.40121  0.75709  0.14848]
 [ 0.75091 -0.28548 -0.2911  -1.9483  -0.9722  -3.64464  1.13753 -0.37992]
 [ 1.15269 -0.05305 -0.84479 -0.67287 -0.21262  1.33644  0.80443 -0.21914]
 [ 0.75224 -0.35133 -0.87255  0.33955  1.12829  0.24818  0.1509  -0.01301]
 [ 0.82651  1.07187 -0.32368 -0.2347   1.14896  1.76183  0.35663  0.01396]
 [-1.8579   1.09684 -0.24406 -0.61464 -1.59585 -2.63297  2.49446 -0.56172]]
```

### Turn 7 — `execute_in_remote_docker` (`38994 ms` LLM)
- **Rationale**: Recover the latent coordinates from the verified calibration and compare panel compositions on identical held-out splits.
- **ATMS NOGOOD DECLARED**: Refuted `A full affine or quadratic basis is sufficient for panel A.` | Witness: `Adding coordinate cubes to the quadratic basis reduces inlier leave-one-out MAE from 0.06307 to 0.000111.` | Blanket Diagnosis: `The affine inverse omitted the cyclic pair terms and coordinate cubes present in the calibration law.` | Sublated Pivot: `Invert the sparse cyclic polynomial map before reevaluating downstream operators.`

```bash
cat /tmp/hegel_continual_archive/*solver.py; OPENBLAS_NUM_THREADS=1 python3 -I /tmp/discover.py | grep -E '^C2|^time rates'; cat > /tmp/map_probe.py <<'PY'
import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; T=tr['traj']; y=tr['y']; Y=np.c_[y,T]; n=len(y)
np.set_printoptions(precision=4,suppress=True,linewidth=155)
def lin(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def one(a):return np.c_[np.ones(len(a)),a]
def mon(a,idx):return np.column_stack([np.prod(a[:,j],axis=1) for j in idx])
pairs=list(itertools.combinations(range(4),2))
cyc=[(0,1),(1,2),(2,3),(3,0)]
def ca(a):return np.c_[one(a),mon(a,cyc),a**3]
A=s['panel_a']; u,v=A['u'],A['v']; keep=abs(v).max(1)<4; c=lin(ca(u[keep]),v[keep]); c[abs(c)<.001]=0
print('t/telemetry',t,s['telemetry_vec'],'calib',c)
def inv(x):
 return np.array([least_squares(lambda a:(ca(a[None])@c).ravel()-q,np.linalg.solve(c[1:5].T,q-c[0]),gtol=1e-11).x for q in x])
z=inv(tr['x']); b=s['panel_b']; cb=lin(one(b['u']),b['v']); B=lambda x:one(x)@cb
p=s['panel_c']; C1=lambda x:np.c_[np.ones(len(x)),mon(x,cyc)]; c1=lin(C1(p['u']),p['v1']); C=lambda x:C1(x)@c1

def c2fit(u,v):
 for name,fn in [('expquad',lambda a,k:np.exp(-k*a*a)),('cos',lambda a,k:np.cos(k*a)),('rational',lambda a,k:1/(1+k*a*a)),('tanh',lambda a,k:np.tanh(k*a))]:
  def f(a,k):return np.c_[one(a),fn(a,k)]
  opt=least_squares(lambda k:(f(u,k)@lin(f(u,k),v)-v).ravel(),[1],bounds=(.02,6)); k=opt.x; F=f(u,k); co=lin(F,v); er=np.sqrt(np.mean((F@co-v)**2)); print('C2 compact',name,k,er); yield er,lambda a,f=f,k=k,co=co:f(a,k)@co
C2=min(c2fit(p['u'],p['v2']),key=lambda a:a[0])[1]
folds=[(np.arange(52),np.arange(52,72))]+[(i[:52],i[52:]) for i in [np.random.default_rng(j).permutation(n) for j in range(4)]]
def cv(K,l):
 E=[]
 for a,b in folds:
  d=abs(K[np.ix_(b,a)]@solve(K[np.ix_(a,a)]+l*np.eye(len(a)),Y[a],assume_a='pos')-Y[b]);E.append([d[:,0].mean(),d[:,1:].mean()])
 return np.array(E)
def kern(X):
 S=np.maximum(.1,X.std(0)); X=(X-X.mean(0))/S; G=X@X.T/X.shape[1];D=cdist(X,X,'sqeuclidean')/X.shape[1];return G,D
zz=B(z); dd={'z':z,'b':zz,'c':C(z),'d':C2(z),'cb':C(zz),'db':C2(zz),'xyz':tr['x'],'z+c+d':np.c_[z,C(z),C2(z)],'b+cb+db':np.c_[zz,C(zz),C2(zz)],'c+d':np.c_[C(z),C2(z)],'cb+db':np.c_[C(zz),C2(zz)]}
for name,X in dd.items():
 out=[]
 for i,F in enumerate([X,np.tanh(X),np.sin(X),X/(1+abs(X)),np.c_[X,np.sin(2*X),np.cos(2*X)],np.c_[X,np.tanh(2*X)]]):
  G,D=kern(F)
  for p in (1,2,3,4):
   for amp,k in [(0,1),(.1,.25),(.1,1),(1,.25),(1,1)]:
    K=(1+.45*G)**p+amp*np.exp(-k*D)
    for l in [1e-6,1e-4,.002,.02]:
     e=cv(K,l);out.append((e.mean(0),e[0],e.max(0),(i,p,amp,k,l)))
 print(name,'best-y',min(out,key=lambda a:a[0][0]),'best-T',min(out,key=lambda a:a[0][1]));
print('z and outputs sample',np.c_[z,y,T[:,0],T[:,-1]][:15]); print('correlations by source');
for name,X in dd.items(): print(name,np.corrcoef(np.c_[X,y,T[:,0],T[:,-1]].T)[:-3,-3:].T)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/map_probe.py
```
- **Container Exit Code**: `0` (`3535 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.7001 -0.7391 -0.1813 -1.0837 -0.8283 -0.3614 -1.3863]
 [ 0.8998 -0.5194  0.1416 -0.4233 -0.1721  0.1614  1.4672]
 [ 1.1422 -0.9497 -1.0325  0.4475  1.9939  0.2055  0.5114]
 [ 0.6334 -1.0887  0.7202  0.1136  2.1621  0.7065  2.3838]
 [ 1.1407  0.1929 -0.4616  0.2146  0.1233  0.6904  3.6362]
 [-0.3265  0.469  -0.8529 -1.026  -2.9566 -0.0954 -1.2241]
 [ 0.0524  0.46   -0.5497 -0.0852  0.1752  0.1681  0.8367]
 [ 0.5467 -0.4181  0.9642  0.6555 -0.2594  0.7087  2.0642]
 [-0.5297  0.1358 -0.9565  0.1547 -0.1906  0.2436  1.4279]
 [ 1.0227  0.0249  0.6247 -0.938  -1.6794 -0.5368 -1.0693]
 [-0.7218 -0.6879 -1.0077 -0.7751 -1.7375  0.0046 -0.6888]
 [-0.3236  0.9337  0.0764 -1.1192  0.2937 -0.2866 -1.7473]]
correlations by source
z [[ 0.2008 -0.0809  0.1083  0.4964]
 [ 0.0835 -0.0348  0.1756  0.4608]
 [ 0.0885 -0.0526  0.1675  0.4209]]
b [[0.2426 0.113  0.4509 0.0884]
 [0.2048 0.1951 0.3624 0.1733]
 [0.1846 0.1884 0.3434 0.1398]]
c [[ 0.1034 -0.3675  0.1565  0.0821]
 [ 0.0656  0.3922  0.3573  0.512 ]
 [-0.0247  0.2195  0.2272  0.3883]]
d [[-0.4777  0.1029 -0.1759  0.1363]
 [-0.4071  0.0822 -0.1228  0.2334]
 [-0.361   0.0919 -0.1272  0.2057]]
cb [[-0.2796 -0.0747  0.0504 -0.3497]
 [ 0.0389 -0.2223  0.0927 -0.3477]
 [-0.1565 -0.2455  0.1049 -0.2703]]
db [[-0.2354 -0.3644 -0.2785 -0.0015]
 [-0.1621 -0.3505 -0.0956  0.0478]
 [-0.1314 -0.3553 -0.1033  0.0366]]
xyz [[ 0.2023 -0.0746 -0.0644  0.4564]
 [ 0.0419 -0.1218  0.1242  0.498 ]
 [ 0.074  -0.1328  0.0982  0.4385]]
z+c+d [[ 0.2008 -0.0809  0.1083  0.4964  0.1034 -0.3675  0.1565  0.0821 -0.4777  0.1029 -0.1759  0.1363]
 [ 0.0835 -0.0348  0.1756  0.4608  0.0656  0.3922  0.3573  0.512  -0.4071  0.0822 -0.1228  0.2334]
 [ 0.0885 -0.0526  0.1675  0.4209 -0.0247  0.2195  0.2272  0.3883 -0.361   0.0919 -0.1272  0.2057]]
b+cb+db [[ 0.2426  0.113   0.4509  0.0884 -0.2796 -0.0747  0.0504 -0.3497 -0.2354 -0.3644 -0.2785 -0.0015]
 [ 0.2048  0.1951  0.3624  0.1733  0.0389 -0.2223  0.0927 -0.3477 -0.1621 -0.3505 -0.0956  0.0478]
 [ 0.1846  0.1884  0.3434  0.1398 -0.1565 -0.2455  0.1049 -0.2703 -0.1314 -0.3553 -0.1033  0.0366]]
c+d [[ 0.1034 -0.3675  0.1565  0.0821 -0.4777  0.1029 -0.1759  0.1363]
 [ 0.0656  0.3922  0.3573  0.512  -0.4071  0.0822 -0.1228  0.2334]
 [-0.0247  0.2195  0.2272  0.3883 -0.361   0.0919 -0.1272  0.2057]]
cb+db [[-0.2796 -0.0747  0.0504 -0.3497 -0.2354 -0.3644 -0.2785 -0.0015]
 [ 0.0389 -0.2223  0.0927 -0.3477 -0.1621 -0.3505 -0.0956  0.0478]
 [-0.1565 -0.2455  0.1049 -0.2703 -0.1314 -0.3553 -0.1033  0.0366]]
```

### Turn 8 — `execute_in_remote_docker` (`32932 ms` LLM)
- **Rationale**: Separate the three temporal amplitudes and the scalar readout, then test whether sparse trigonometric, quotient, or interaction terms explain them better than a shared kernel.
- **Alternatives Weighed**: `Choose the best generic kernel from the last sweep` (rejected: Its validation error remains several times the required tolerance.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/discover.py | grep -E '^C2|^time rates'; sed '/^folds=/,$d' /tmp/map_probe.py > /tmp/grammar_probe.py; cat >> /tmp/grammar_probe.py <<'PY'
F=-np.expm1(-t[:,None]*np.array([.37633,1.78968,5.5352])); H=lin(F,T.T).T; target=np.c_[y,H]; train=np.arange(52); test=np.arange(52,72)
print('basis error',np.mean(abs(H@F.T-T),axis=0)); print('latent ranges',z.min(0),z.max(0),'B',B(z).min(0),B(z).max(0));
for name,a in {'z':z,'b':B(z),'c':C(z),'d':C2(z),'cb':C(B(z)),'db':C2(B(z))}.items():
 cols={'1':np.ones(n)}
 def push(k,g):cols[k]=g
 for i in range(4):
  for j in range(i,4):
   for k in range(j,4):push(f'{i}{j}{k}',a[:,i]*a[:,j]*a[:,k])
 for i in range(4):
  push(f'{i}',a[:,i]); push(f'abs{i}',abs(a[:,i]));
  for j in range(i,4):
   push(f'{i}{j}',a[:,i]*a[:,j]);
 for i,w in enumerate(a.T):
  for j in range(i,4):
   for sg in ([0] if i==j else [-1,1]):
    d=w+sg*a[:,j];g=f'{i},{sg},{j}'
    for k in [.5,1,1.5,2,2.5,3,3.5,4,5,6,7,8]:
     for label,f in [('s',np.sin),('c',np.cos),('t',np.tanh),('a',lambda u:1/(1+u*u)),('r',lambda u:u/(1+u*u))]:push(f'{label}{k}:{g}',f(k*d))
 for k,(i,j,l) in enumerate(itertools.combinations(range(4),3)):
  for s1,s2 in itertools.product([-1,1],repeat=2):
   d=a[:,i]+s1*a[:,j]+s2*a[:,l]
   for f in (np.sin,np.cos,np.tanh):
    for r in [1,2,3,4,5,6]:push(f'{f.__name__}{r}:{i},{s1},{j},{s2},{l}',f(r*d))
 M=np.column_stack(list(cols.values())); names=np.array(list(cols)); M=(M-M[train].mean(0))/np.maximum(M[train].std(0),1e-8); M[:,0]=1
 print('\n',name)
 for q in range(4):
  yy=target[:,q]; chosen=[0]; rec=[]
  for step in range(18):
   U=np.linalg.qr(M[train][:,chosen],mode='reduced')[0]; resid=yy[train]-U@(U.T@yy[train]); A=M[train]-U@(U.T@M[train]); corr=np.abs(A.T@resid)/np.maximum(np.linalg.norm(A,axis=0),1e-6); corr[chosen]=0;chosen.append(np.argmax(corr)); cc=lin(M[train][:,chosen],yy[train]); p=M[:,chosen]@cc; rec.append((round(np.mean(abs(p[train]-yy[train])),4),round(np.mean(abs(p[test]-yy[test])),4),names[chosen[-1]]))
  print(q,rec[:12],min(rec,key=lambda e:e[1]))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/grammar_probe.py
```
- **Container Exit Code**: `0` (`1249 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
471, 0.843, 's7:1,-1,2'), (0.333, 0.8347, 'cos2:1,-1,2,1,3'), (0.3056, 0.8078, 'cos5:1,1,2,1,3'), (0.2756, 0.7219, 's0.5:0,-1,1'), (0.2458, 0.6563, 'cos4:0,1,2,1,3'), (0.2311, 0.6648, 's8:0,-1,2'), (0.2084, 0.6098, 'c5:0,1,1'), (0.1887, 0.6497, 's6:3,0,3'), (0.177, 0.6626, 'c3.5:1,-1,3'), (0.1691, 0.6179, 'sin4:0,1,1,1,3')] (0.2084, 0.6098, 'c5:0,1,1')
3 [(0.1424, 0.2164, 's6:1,-1,2'), (0.1288, 0.2405, '012'), (0.1159, 0.2472, 'sin3:0,1,1,-1,2'), (0.0951, 0.2341, 'c5:0,1,3'), (0.0878, 0.2639, 's6:0,-1,1'), (0.0829, 0.2634, 'c4:1,-1,3'), (0.0727, 0.2577, 'cos2:0,-1,2,1,3'), (0.0698, 0.259, 'sin6:0,-1,2,1,3'), (0.0633, 0.2632, 'cos5:0,1,2,-1,3'), (0.0599, 0.2503, 's5:1,1,2'), (0.0554, 0.2475, 'cos6:1,1,2,1,3'), (0.0493, 0.2414, 's8:2,-1,3')] (0.1424, 0.2164, 's6:1,-1,2')

 db
0 [(0.6329, 0.8771, 'c0.5:0,1,1'), (0.5363, 0.8562, '033'), (0.4735, 0.9298, 'c7:1,-1,3'), (0.4058, 0.9059, 's8:3,0,3'), (0.4038, 1.0071, 's7:1,1,3'), (0.3478, 1.0395, 'sin4:0,1,1,-1,3'), (0.3359, 0.9888, 'tanh3:0,-1,1,-1,3'), (0.3069, 0.9704, 'c6:2,0,2'), (0.2691, 0.93, 'c8:0,0,0'), (0.2473, 0.9263, 'c3:2,1,3'), (0.2218, 0.9294, 's7:2,1,3'), (0.1852, 0.9023, 'a0.5:0,-1,1')] (0.1826, 0.8549, 's7:1,1,2')
1 [(1.0922, 1.5588, 'cos1:0,1,1,1,2'), (0.959, 1.8405, 's3:0,-1,2'), (0.8421, 1.6691, 'c0.5:0,-1,3'), (0.7409, 1.6497, 'a5:1,1,3'), (0.6839, 1.8966, 'cos5:0,1,1,-1,3'), (0.5897, 1.7949, 'sin3:1,-1,2,1,3'), (0.5664, 1.6625, 'cos3:0,1,2,1,3'), (0.4764, 1.7297, 'sin4:0,-1,1,-1,3'), (0.4572, 1.7427, 's7:0,-1,1'), (0.4046, 1.788, 's6:0,0,0'), (0.3563, 1.8151, 'cos6:1,1,2,1,3'), (0.3603, 1.8488, 'cos5:0,1,2,-1,3')] (1.0922, 1.5588, 'cos1:0,1,1,1,2')
2 [(0.5107, 0.7113, 'cos1:0,1,2,1,3'), (0.421, 0.6918, '11'), (0.3851, 0.6677, 'cos4:0,-1,2,-1,3'), (0.331, 0.7035, 's4:2,0,2'), (0.3047, 0.6771, 'cos1:0,-1,2,1,3'), (0.2629, 0.6315, 'a3.5:0,1,2'), (0.2435, 0.5918, 'tanh6:1,1,2,1,3'), (0.2389, 0.5763, 'c2:1,1,3'), (0.2086, 0.5777, 's1.5:0,-1,3'), (0.1928, 0.5793, 's5:1,1,3'), (0.1734, 0.5881, 'sin6:1,1,2,1,3'), (0.1447, 0.6254, 'cos4:0,1,1,1,2')] (0.2389, 0.5763, 'c2:1,1,3')
3 [(0.0961, 0.1313, 'tanh2:0,1,1,1,2'), (0.0911, 0.1344, '111'), (0.0834, 0.1366, 's1:0,-1,1'), (0.08, 0.1321, 's4:0,1,3'), (0.0701, 0.1375, 'sin5:1,-1,2,1,3'), (0.0638, 0.1433, 'cos3:0,-1,1,1,2'), (0.0577, 0.1428, 'a8:1,-1,2'), (0.0503, 0.1456, 'c7:0,-1,2'), (0.0459, 0.1665, 's4:3,0,3'), (0.0401, 0.1722, 'sin3:0,-1,1,-1,3'), (0.0379, 0.1739, 's8:1,1,2'), (0.0317, 0.1784, 'cos1:0,1,2,-1,3')] (0.0961, 0.1313, 'tanh2:0,1,1,1,2')
```

### Turn 9 — `execute_in_remote_docker` (`37014 ms` LLM)
- **Rationale**: Use the measured cosine law without surplus terms, check whether panel B is truly affine, and test sparse cross-operator products and rational quotients separately for the scalar and trajectory.

```bash
sed '/^folds=/,$d' /tmp/map_probe.py | sed '/^C2=min/c\C2=lambda a:one(np.cos(2.4*a))@lin(one(np.cos(2.4*p["u"])),p["v2"])' > /tmp/joint_probe.py
cat >> /tmp/joint_probe.py <<'PY'
print('B errors',np.sqrt(np.mean((B(b['u'])-b['v'])**2,0)));print('C2 errors',np.sqrt(np.mean((C2(p['u'])-p['v2'])**2,0)))
print('B',cb);print('C1',c1);print('C2',lin(one(np.cos(2.4*p['u'])),p['v2']));
a=np.arange(52);h=np.arange(52,72); folds=[(a,h)]+[(ix[:52],ix[52:]) for ix in [np.random.default_rng(i).permutation(n) for i in range(4)]]
F=-np.expm1(-t[:,None]*np.array([.37633,1.78968,5.5352])); H=lin(F,T.T).T; L=np.c_[y,H,T]
D={'z':z,'b':B(z),'c':C(z),'d':C2(z),'cb':C(B(z)),'db':C2(B(z)),'p':mon(z,cyc),'q':np.cos(2.4*z),'qb':np.cos(2.4*B(z))};
for k,X in D.items(): print(k,np.round(np.corrcoef(np.c_[X,L[:,:4]].T)[:4,4:],2))
cols={};groups={}
def add(name,X):
 X=np.reshape(X,(n,-1)); groups[name]=X
 for i,g in enumerate(X.T):cols[name+':'+str(i)]=g
for name,X in D.items():
 add(name,X)
 for d in [2,3]:add(name+'P'+str(d),mon(X,list(itertools.combinations_with_replacement(range(4),d))))
 for k in [.5,1.25,2,2.4,3,4,5,6,8]:
  for fun in [np.sin,np.cos,np.tanh]:add(name+fun.__name__+str(k),fun(X*k))
 for k in [.22,.5,1,2]:
  add(name+'rat'+str(k),X/(1+k*X*X));add(name+'rat2'+str(k),1/(1+k*X*X))
for n1,n2 in [('z','c'),('z','d'),('c','d'),('b','c'),('b','d'),('b','cb'),('b','db'),('cb','db'),('z','b'),('z','cb'),('z','db'),('p','q')]:
 W,V=D[n1],D[n2];add(n1+'*'+n2,(W[:,:,None]*V[:,None,:]));add(n1+'/'+n2,(W[:,:,None]/(1+.22*V[:,None,:]**2)));add(n2+'/'+n1,(V[:,:,None]/(1+.22*W[:,None,:]**2)))
for k,X in D.items():
 for name,V in [('sum',X[:,[i for i,j in pairs]]+X[:,[j for i,j in pairs]]),('dif',X[:,[i for i,j in pairs]]-X[:,[j for i,j in pairs]])]:
  for f in [np.sin,np.cos,np.tanh]:
   for r in [1,1.25,2.4,3.5,5,6]:add(k+name+f.__name__+str(r),f(r*V))
from sklearn.linear_model import Lars
M=np.column_stack(list(cols.values())); N=np.array(list(cols)); print('M',M.shape)
for j in [0,1,2,3,4,11]:
 target=L[:,j]; print('TARGET',j); opt=[]
 for ks in [(k,) for k in D]+[('z','p','q','c','d'),('b','cb','db','qb'),('z','b','c','d'),tuple(D)]:
  mask=np.array([any(k in ks for k in name.split(':')[0].replace('*','/').split('/')) or any(name.startswith(k+p) for k in ks for p in ['P','sin','cos','tanh','rat','sum','dif']) for name in N]); X=M[:,mask]; names=N[mask]; E=[]; traces=[]
  for aa,hh in folds:
   m=X[aa].mean(0); sd=np.maximum(X[aa].std(0),1e-9); XX=(X-m)/sd; mdl=Lars(n_nonzero_coefs=28,fit_intercept=True).fit(XX[aa],target[aa]); pred=XX@mdl.coef_path_+mdl.intercept_; E.append(np.mean(abs(pred[hh]-target[hh,None]),0)); traces.append(mdl.active_)
  e=np.stack([np.pad(v,(0,29-len(v)),mode='edge') for v in E]); i=e.mean(0).argmin();opt.append((e.mean(0)[i],e[0,i],max(e[:,i]),str(ks),i,names[traces[0][:min(i,12)]]))
 for r in sorted(opt,key=lambda v:v[0])[:4]:print(r)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/joint_probe.py
```
- **Container Exit Code**: `0` (`9120 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, 'b/d:10', 'z*d:9', 'b*d:9', 'c*d:8', 'z*d:14', 'dsin2:3', 'dsin2.4:3', 'z/d:10', 'b*d:2', 'dcos6:1', 'dsin4:1'], dtype='<U15'))
(0.09430246149134801, 0.08233482684897711, 0.11900425175462123, "('b', 'cb', 'db', 'qb')", 16, array(['z/b:12', 'b*c:11', 'bP3:16', 'b*c:5', 'b*db:13', 'b*c:12', 'z*cb:5', 'cbP3:5', 'z/db:12', 'z/cb:14', 'cbcos8:0', 'z/db:14'], dtype='<U15'))
(0.10355982966968122, 0.09556443075956253, 0.1884574264176368, "('z',)", 9, array(['z/b:12', 'z*c:3', 'z*c:9', 'zsumtanh1:5', 'b/z:8', 'zP3:11', 'zsumsin1:5', 'zsumtanh1.25:5', 'zP3:17'], dtype='<U15'))
TARGET 4
(0.24685753862811555, 0.3426363920848782, 0.3426363920848782, "('z',)", 18, array(['z*b:12', 'z*c:9', 'z*b:2', 'db/z:7', 'ztanh8:3', 'zdifsin3.5:3', 'db/z:4', 'zsumcos1.25:3', 'zsumcos1:3', 'zP3:17', 'zsumsin1:2', 'zsumcos2.4:3'],
      dtype='<U15'))
(0.25243762240258594, 0.3112341934296528, 0.3112341934296528, "('c',)", 14, array(['csumsin1.25:4', 'csumsin1:5', 'z/c:15', 'z*c:9', 'csumcos2.4:0', 'd/c:0', 'd/c:11', 'd/c:14', 'd/c:2', 'd/c:1', 'csumcos2.4:2', 'csumsin1.25:5'],
      dtype='<U15'))
(0.26873712942445604, 0.3228506452345229, 0.3228506452345229, "('b',)", 11, array(['z*b:12', 'bcos1.25:0', 'bsumtanh3.5:1', 'z*b:2', 'db/b:5', 'b*c:11', 'bsumtanh5:1', 'bP3:11', 'd/b:8', 'bsumcos1:0', 'b*c:5'], dtype='<U15'))
(0.26973711815419754, 0.31446416117230963, 0.31446416117230963, "('cb',)", 10, array(['cbsumcos2.4:5', 'db/cb:13', 'db/cb:4', 'cbcos5:0', 'z*cb:8', 'z/cb:15', 'b*cb:4', 'cb*db:8', 'cbdifsin2.4:3', 'cbdifsin1.25:3'], dtype='<U15'))
TARGET 11
(1.1024554776197444, 1.3205072408329444, 1.3205072408329444, "('b',)", 15, array(['z*b:12', 'd/b:8', 'd/b:14', 'd/b:0', 'bsumtanh2.4:1', 'bsumtanh3.5:1', 'bdifcos1:2', 'db/b:0', 'b*c:5', 'z/b:12', 'bsumtanh1.25:1',
       'bsumsin1.25:1'], dtype='<U15'))
(1.1228680192534963, 1.0453096450216817, 1.2484637868223278, "('c',)", 11, array(['d/c:0', 'd/c:11', 'd/c:14', 'z*c:9', 'z/c:15', 'csumcos2.4:0', 'z/c:14', 'csumsin1.25:4', 'csumsin1.25:5', 'ccos3:0', 'b*c:11'], dtype='<U15'))
(1.1522666510546091, 1.1598369882647381, 1.6571697944152597, "('cb',)", 15, array(['cbsumcos2.4:5', 'z*cb:8', 'db/cb:4', 'cbsin3:2', 'cbsumsin1.25:0', 'z/cb:13', 'cb*db:8', 'cbcos5:0', 'b*cb:4', 'db/cb:13', 'cbsumsin2.4:5',
       'db/cb:3'], dtype='<U15'))
(1.1568296033675745, 1.2196242380812161, 1.3290134437729733, "('q',)", 8, array(['qdiftanh1:2', 'qsumsin1:4', 'qsumtanh1.25:5', 'p/q:14', 'qP3:0', 'qdifcos2.4:4', 'qsumtanh2.4:5', 'p*q:7'], dtype='<U15'))
```
- **Stderr (tail)**:
```text
9e-08. Reduce max_iter or increase eps parameters.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_least_angle.py:725: ConvergenceWarning: Regressors in active set degenerate. Dropping a regressor, after 7 iterations, i.e. alpha=4.736e-02, with an active set of 7 regressors, and the smallest cholesky pivot element being 2.220e-16. Reduce max_iter or increase eps parameters.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_least_angle.py:725: ConvergenceWarning: Regressors in active set degenerate. Dropping a regressor, after 20 iterations, i.e. alpha=3.360e-02, with an active set of 18 regressors, and the smallest cholesky pivot element being 8.429e-08. Reduce max_iter or increase eps parameters.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_least_angle.py:725: ConvergenceWarning: Regressors in active set degenerate. Dropping a regressor, after 8 iterations, i.e. alpha=5.274e-02, with an active set of 8 regressors, and the smallest cholesky pivot element being 8.429e-08. Reduce max_iter or increase eps parameters.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_least_angle.py:725: ConvergenceWarning: Regressors in active set degenerate. Dropping a regressor, after 22 iterations, i.e. alpha=8.297e-01, with an active set of 19 regressors, and the smallest cholesky pivot element being 2.220e-16. Reduce max_iter or increase eps parameters.
  warnings.warn(
```

### Turn 10 — `execute_in_remote_docker` (`35323 ms` LLM)
- **Rationale**: Search off-axis frequencies within the fitting split, then refine them by variable projection and check whether the resulting expressions generalize to the 20 unseen rows.

```bash
sed '/^print('\''B errors/,$d' /tmp/joint_probe.py > /tmp/ridge_probe.py
cat >> /tmp/ridge_probe.py <<'PY'
F=-np.expm1(-t[:,None]*np.array([.37633,1.78968,5.5352])); H=lin(F,T.T).T; L=np.c_[y,H]; a=np.arange(52); h=np.arange(52,72)
from scipy.linalg import qr
W=B(z); P=mon(z,cyc); Q=np.cos(2.4*z); R=np.cos(2.4*W); Cx=C(z); Dx=C2(z)
print('ranges',z.min(0),z.max(0),'inverse errors', np.max(abs(ca(z)@c-tr['x'])));print('rates fits baseline')
def jd(m):
 E=m@lin(m[a],L[a])-L;return np.round(np.mean(abs(E[a]),0),4),np.round(np.mean(abs(E[h]),0),4)
for k,m in [('z',one(z)),('z2',one(np.c_[z,mon(z,list(itertools.combinations_with_replacement(range(4),2)))])),('z3',one(np.c_[z,mon(z,list(itertools.combinations_with_replacement(range(4),2))),mon(z,list(itertools.combinations_with_replacement(range(4),3)))])),('zPQ',one(np.c_[z,P,Q])),('zPQR',one(np.c_[z,P,Q,R])),('known',one(np.c_[z,P,Q,R,mon(W,cyc),np.sin(1.25*W),np.sin(1.25*z)]))]: print(k,jd(m))
# Scan free Fourier ridges after removing low-complexity measured operators; jointly refine multiple peaks.
rng=np.random.default_rng(313); D=np.r_[rng.normal(size=(22000,4))*rng.uniform(.4,6,(22000,1)),cb[1:].T*1.25,cb[1:].T*2.4,np.eye(4)*2.4]
SW=np.sin(z@D.T); CW=np.cos(z@D.T)
base={ 'aff':one(z),'phys':one(np.c_[z,P,Q,R]),'quad':one(np.c_[z,mon(z,list(itertools.combinations_with_replacement(range(4),2))),Q,R]) }
for k,G in base.items():
 for j in range(4):
  best=[]; yy=L[:,j]; Ws=np.empty((0,4))
  for count in range(1,5):
   def design(w):
    P0=z@w.reshape(-1,4).T;return np.c_[G,np.sin(P0),np.cos(P0)]
   Q0=qr(design(Ws)[a],mode='economic')[0]; res=yy[a]-Q0@(Q0.T@yy[a]); s=SW[a]-Q0@(Q0.T@SW[a]); c0=CW[a]-Q0@(Q0.T@CW[a]); vs=(s*s).sum(0); vc=(c0*c0).sum(0); cs=(s*c0).sum(0); u=s.T@res; v=c0.T@res; scores=(u*u*vc+v*v*vs-2*u*v*cs)/np.maximum(vs*vc-cs*cs,1e-12); idx=np.argsort(scores)[-100:][::-1]; seeds=[]
   for ii in idx:
    if all(min(np.linalg.norm(D[ii]-vv),np.linalg.norm(D[ii]+vv))>.65 for vv in seeds):seeds.append(D[ii]);
    if len(seeds)>=5:break
   def fun(w):
    M=design(w)[a];return M@lin(M,yy[a])-yy[a]
   opt=min((least_squares(fun,np.r_[Ws.ravel(),w],max_nfev=85) for w in seeds),key=lambda o:o.cost); Ws=opt.x.reshape(-1,4); M=design(Ws); coef=lin(M[a],yy[a]); err=abs(M@coef-yy);best.append((np.round(err[a].mean(),4),np.round(err[h].mean(),4)))
   if err[h].mean()<.05 or count==2: print('ridge',k,j,count,best[-1],np.round(Ws,3),np.round(coef[-2*count:],3))
  print('ridge summary',k,j,best)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/ridge_probe.py
```
- **Container Exit Code**: `0` (`9388 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ridge aff 1 2 (0.6451, 1.1052) [[-0.001 -0.004 -0.004 -0.016]
 [-0.057 -2.151 -0.202  0.86 ]] [941366.814     -0.393  16007.081      1.428]
ridge summary aff 1 [(0.8479, 1.4066), (0.6451, 1.1052), (0.3339, 0.7914), (0.1998, 0.801)]
ridge aff 2 2 (0.1505, 0.3773) [[-0.428 -0.596 -0.949  1.531]
 [-0.002  0.001 -0.    -0.002]] [     -0.066 2364316.029       0.879 -163225.943]
ridge summary aff 2 [(0.2992, 0.5547), (0.1505, 0.3773), (0.1034, 0.4516), (0.0654, 0.499)]
ridge aff 3 2 (0.0406, 0.1416) [[ 0.2   -2.751  1.578  0.33 ]
 [ 1.171  1.05  -4.726 -1.692]] [-0.127  0.085 -0.009  0.015]
ridge summary aff 3 [(0.0518, 0.1031), (0.0406, 0.1416), (0.0269, 0.1462), (0.0159, 0.1748)]
ridge phys 0 2 (0.1215, 0.6604) [[ 1.696 -0.071  0.712  1.168]
 [ 7.893  0.642 -9.719  7.008]] [-2.998 -0.241 10.755 -0.229]
ridge summary phys 0 [(0.1769, 0.536), (0.1215, 0.6604), (0.0621, 0.9778), (0.0294, 0.8681)]
ridge phys 1 2 (0.1074, 0.5948) [[ 0.427 -0.801  0.859  0.225]
 [-0.005 -0.001  0.001 -0.004]] [      -2.215 19286964.784        0.683    73055.313]
ridge summary phys 1 [(0.1886, 0.4183), (0.1074, 0.5948), (0.0605, 0.6716), (0.0258, 0.8159)]
ridge phys 2 2 (0.0548, 0.3458) [[-0.538  1.069 -0.4    1.191]
 [ 0.043 -0.129 -1.56  -0.476]] [-0.07  -0.251 -0.917  1.474]
ridge summary phys 2 [(0.0921, 0.3553), (0.0548, 0.3458), (0.0253, 0.4169), (0.0086, 0.4125)]
ridge phys 3 2 (0.0223, 0.1467) [[ 0.009 -0.013  0.014  0.007]
 [-0.005 -0.001  0.001 -0.004]] [ -72432.12  5831723.17     -755.143   10146.8  ]
ridge summary phys 3 [(0.0456, 0.0956), (0.0223, 0.1467), (0.0145, 0.1576), (0.0058, 0.1747)]
ridge quad 0 2 (0.06, 1.2405) [[  6.681   3.9   -16.561   6.097]
 [ -1.216  -0.624   0.693   0.939]] [-0.28   2.409  0.531 -3.637]
ridge summary quad 0 [(0.1295, 0.7989), (0.06, 1.2405), (0.0262, 1.1901), (0.0064, 1.0562)]
ridge quad 1 2 (0.0727, 0.7197) [[-4.319 -3.946 -1.228 -4.011]
 [ 5.737 -9.571  2.666  7.03 ]] [ 0.329 -0.235 -0.049  0.019]
ridge summary quad 1 [(0.1175, 0.7686), (0.0727, 0.7197), (0.0312, 0.7557), (0.0105, 0.8086)]
ridge quad 2 2 (0.0349, 0.3976) [[-0.07   0.065 -0.015  0.043]
 [-0.023  0.414  2.789  0.982]] [   224.842      0.166 -78362.208      0.143]
ridge summary quad 2 [(0.0561, 0.3153), (0.0349, 0.3976), (0.0148, 0.4928), (0.0059, 0.5401)]
ridge quad 3 2 (0.0187, 0.1606) [[-4.368 -3.543 -1.11  -3.992]
 [-3.016  2.671 -6.061 -3.891]] [ 0.089 -0.026 -0.037 -0.065]
ridge summary quad 3 [(0.0285, 0.1567), (0.0187, 0.1606), (0.0075, 0.1761), (0.002, 0.1676)]
```

### Turn 11 — `execute_in_remote_docker` (`40564 ms` LLM)
- **Rationale**: Save a runnable calibrated baseline now, then test whether the nearly cubic fast mode identifies reusable structure in the remaining outputs.

```bash
cat > /holdout_task/batch_solver.py <<'PY'
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve
from itertools import combinations_with_replacement

def ls(a,b): return np.linalg.lstsq(a,b,rcond=1e-9)[0]
def one(a): return np.c_[np.ones(len(a)),a]
def pair(a): return a*np.roll(a,-1,axis=1)
def align(panel,x):
    u,v=panel['u'],panel['v']
    def f(a): return one(np.c_[a,pair(a),a**3])
    F=f(u); mask=np.max(abs(v-np.median(v,axis=0)),axis=1)<6
    for _ in range(3):
        c=ls(F[mask],v[mask]); r=np.linalg.norm(F@c-v,axis=1); mask=r<max(7*np.median(r),.001)
    c[abs(c)<.001]=0
    return np.array([least_squares(lambda a:(f(a[None])@c)[0]-q,np.linalg.solve(c[1:5].T,q-c[0]),gtol=1e-11).x for q in x])
def features(s,z):
    b=s['panel_b']; w=one(z)@ls(one(b['u']),b['v']); p=s['panel_c']; u=p['u']; cp=ls(one(pair(u)),p['v1']); cq=ls(one(np.cos(2.4*u)),p['v2']); c=one(pair(z))@cp; d=one(np.cos(2.4*z))@cq
    return w,c,d

def solve_batch(study):
    tr=study['train']; n=len(tr['y']); z=align(study['panel_a'],np.vstack([tr['x'],study['test']['x']])); w,c,d=features(study,z)
    X=np.c_[z,pair(z),np.cos(2.4*z)]
    X=X/np.maximum(X[:n].std(0),.1); G=X@X.T/X.shape[1]; D=np.maximum(G.diagonal()[:,None]+G.diagonal()-2*G,0)
    K=(1+.45*G)**3+.1*np.exp(-.25*D)
    Y=np.c_[tr['y'],tr['traj']]; pred=K[n:,:n]@solve(K[:n,:n]+.002*np.eye(n),Y,assume_a='pos'); y,T=pred[:,0],pred[:,1:]
    return dict(y_hat=y,state_pred=T,diagnostics_vec=np.array([y.mean(),y.std(),T[:,-1].mean(),T[:,-1].std()]))
PY
cat > /tmp/residual_probe.py <<'PY'
import sys,pickle,numpy as np,itertools
sys.path.insert(0,'/holdout_task'); from batch_solver import *
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=align(s['panel_a'],tr['x']); w,c,d=features(s,z); t=s['eval_times']; T=tr['traj']; y=tr['y']; a=slice(0,52); h=slice(52,None); np.set_printoptions(precision=3,suppress=True)
F=-np.expm1(-t[:,None]*[.37633,1.78968,5.5352]); H=ls(F,T.T).T; Y=np.c_[y,H]; print('H3 corrs',np.corrcoef(Y.T));
inds=[()]+[j for m in range(1,4) for j in itertools.combinations_with_replacement(range(4),m)]; P=np.column_stack([z[:,j].prod(1) for j in inds]); co=ls(P[a],Y[a]); print('cubic H3',[(j,round(k,3)) for j,k in zip(inds,co[:,3]) if abs(k)>.02]);
# Sweep groupwise additive rather than high-dimensional full interactions.
blocks={'z':z,'p':pair(z),'q':np.cos(2.4*z),'w':w,'c':c,'d':d,'cw':pair(w),'dw':np.cos(2.4*w)}
for k in range(4):
 print('out',k)
 for label,X in blocks.items():
  for typ,fun in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('r',lambda q:q/(1+q*q))]:
   full=[]
   for f in [.5,1,1.25,1.5,2,2.4,3,4,5,6]:
    cols=one(np.c_[z,pair(z),np.cos(2.4*z),fun(f*X)]); p=cols@ls(cols[a],Y[a,k]); full.append((abs(p[h]-Y[h,k]).mean(),f,abs(p[a]-Y[a,k]).mean()))
   b=min(full)
   if b[0]<[.43,.36,.20,.085][k]:print(label,typ,b)
# See whether residuals of the three laws collapse into shared directions after linear physical terms.
for X in [one(z),one(np.c_[z,pair(z),np.cos(2.4*z)]),P[:,:15],P]:
 E=Y-X@ls(X,Y); print('residual sv',np.linalg.svd(E,full_matrices=False)[1:]);
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/residual_probe.py; OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `1` (`990 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
H3 corrs [[ 1.     0.298 -0.062  0.637]
 [ 0.298  1.     0.145  0.41 ]
 [-0.062  0.145  1.    -0.129]
 [ 0.637  0.41  -0.129  1.   ]]
cubic H3 [((0, 0), -0.029), ((1, 1), 0.028), ((2, 3), 0.026), ((3, 3), 0.039), ((0, 0, 1), 0.078), ((0, 0, 3), 0.192), ((0, 1, 1), -0.029), ((0, 1, 2), -0.085), ((0, 1, 3), -0.155), ((0, 3, 3), 0.064), ((1, 1, 2), 0.236), ((1, 1, 3), 0.035), ((1, 2, 2), -0.07), ((1, 3, 3), -0.02), ((2, 2, 3), 0.233), ((2, 3, 3), -0.022)]
out 0
out 1
out 2
out 3
residual sv (array([13.444,  7.811,  7.101,  0.883]), array([[ 0.033,  0.991,  0.129,  0.   ],
       [ 0.862,  0.037, -0.501,  0.06 ],
       [ 0.502, -0.128,  0.855,  0.004],
       [ 0.054,  0.002, -0.027, -0.998]]))
residual sv (array([5.423, 2.526, 0.609, 0.059]), array([[ 0.732,  0.655,  0.108,  0.153],
       [-0.395,  0.536, -0.737,  0.121],
       [-0.511,  0.533,  0.616, -0.274],
       [-0.217, -0.02 ,  0.256,  0.942]]))
residual sv (array([6.023, 2.663, 1.274, 0.2  ]), array([[ 0.662,  0.73 ,  0.105,  0.13 ],
       [-0.551,  0.576, -0.602,  0.059],
       [-0.478,  0.365,  0.763, -0.237],
       [-0.173, -0.044,  0.21 ,  0.961]]))
residual sv (array([2.148, 1.718, 0.428, 0.1  ]), array([[ 0.695,  0.276,  0.664,  0.001],
       [-0.261,  0.957, -0.124, -0.003],
       [-0.67 , -0.088,  0.737, -0.002],
       [-0.003,  0.002,  0.   ,  1.   ]]))
{"schema_valid": false, "errors": ["public_cv_y_mae_exceeded:0.229140", "public_cv_traj_mae_exceeded:0.244225"]}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=849a2da4efdd5266 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_849a2da4 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`34042 ms` LLM)
- **Rationale**: Compare low-order interaction kernels across multiple splits and inspect shared cubic coefficients for an identifiable structural correction while retaining the runnable baseline.
- **Alternatives Weighed**: `Tune free Fourier ridges further` (rejected: The fitting error improved without a corresponding held-out improvement.)

```bash
cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; cat > /tmp/anova_probe.py <<'PY'
import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/holdout_task'); from batch_solver import *
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=align(s['panel_a'],tr['x']); w,c,d=features(s,z); T=tr['traj']; y=tr['y']; Y=np.c_[y,T]; n=len(y); a=np.arange(52); h=np.arange(52,72)
def nrm(x):return x/np.maximum(np.std(x,axis=0),.1)
def gram(x):
 x=nrm(x); return x@x.T/x.shape[1]
q=np.cos(2.4*z); p=pair(z)
P=np.column_stack([np.prod(z[:,j],axis=1) for j in itertools.combinations_with_replacement(range(4),3)]); print('coefficients fit cubic joint'); cf=ls(one(np.c_[z,p,q,P])[:52],Y[:52]); print(np.round(cf,2));
folds=[(a,h)]+[(i[:52],i[52:]) for i in [np.random.default_rng(k).permutation(n) for k in range(6)]]
def cv(K,lam,fs=folds):
 es=[]
 for i,j in fs:
  D=abs(K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+lam*np.eye(len(i)),Y[i],assume_a='pos')-Y[j]);es.append((D[:,0].mean(),D[:,1:].mean()))
 return np.array(es)
groups={'z':z,'z+w':np.c_[z,w],'z+c+d':np.c_[z,c,d],'z+p+q':np.c_[z,p,q],'w':w,'c+d':np.c_[c,d]}; base=1+gram(z)+gram(p)+gram(q)
mods=[]
for name,X in groups.items():
 X=nrm(X); G=X@X.T/X.shape[1]; ncol=X.shape[1]; ds=(X[:,None]-X[None,:])**2
 for k in [.07,.2,.55,1.5,4,10]:
  U=np.exp(-k*ds); K=U.mean(2); K2=(U.sum(2)**2-(U**2).sum(2))/max(1,ncol*(ncol-1)); K3=(U.sum(2)**3-3*U.sum(2)*(U**2).sum(2)+2*(U**3).sum(2))/max(1,ncol*(ncol-1)*(ncol-2)); R=np.exp(-k*ds.mean(2))
  for form,M in enumerate([K,K+.1*K2,K+K2,K+.5*K2+.15*K3,K2,K2+K3,K3,R,K+.5*K2+.5*R]):
   for pk,N in enumerate([np.ones_like(K),base,(1+.4*G)**3,base+.25*(1+.4*G)**3]):
    for l in [1e-6,3e-5,.001,.025]:
     e=cv(N+M,l); mods.append((e[1:].mean(0),e[0],e.max(0),(name,k,form,pk,l)))
for q0 in [0,1]:
 print('outer-robust best',q0, sorted(mods,key=lambda a:a[0][q0])[:4]); print('fixed best',q0,sorted(mods,key=lambda a:a[1][q0])[:4])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/anova_probe.py
```
- **Container Exit Code**: `0` (`4029 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.04  0.05  0.04]
 [ 0.3   0.08  0.15  0.18  0.2   0.2   0.2   0.2   0.2 ]]
outer-robust best 0 [(array([0.2120078 , 0.14587414]), array([0.21688916, 0.15817159]), array([0.25510698, 0.18874602]), ('z+c+d', 0.07, 7, 1, 1e-06)), (array([0.21268279, 0.1482731 ]), array([0.21910693, 0.16029241]), array([0.26065353, 0.18535341]), ('z+p+q', 0.07, 7, 1, 1e-06)), (array([0.2179253 , 0.14914392]), array([0.24430885, 0.18551911]), array([0.27077867, 0.19051428]), ('z+c+d', 0.07, 7, 1, 3e-05)), (array([0.21876178, 0.15133263]), array([0.24651553, 0.18783665]), array([0.27584988, 0.1900163 ]), ('z+p+q', 0.07, 7, 1, 3e-05))]
fixed best 0 [(array([0.22096828, 0.15316112]), array([0.20427258, 0.18316871]), array([0.27463792, 0.20338259]), ('z+c+d', 0.07, 7, 0, 1e-06)), (array([0.22152293, 0.15655638]), array([0.2072838 , 0.18659004]), array([0.27804766, 0.19984809]), ('z+p+q', 0.07, 7, 0, 1e-06)), (array([0.2120078 , 0.14587414]), array([0.21688916, 0.15817159]), array([0.25510698, 0.18874602]), ('z+c+d', 0.07, 7, 1, 1e-06)), (array([0.21268279, 0.1482731 ]), array([0.21910693, 0.16029241]), array([0.26065353, 0.18535341]), ('z+p+q', 0.07, 7, 1, 1e-06))]
outer-robust best 1 [(array([0.2120078 , 0.14587414]), array([0.21688916, 0.15817159]), array([0.25510698, 0.18874602]), ('z+c+d', 0.07, 7, 1, 1e-06)), (array([0.21268279, 0.1482731 ]), array([0.21910693, 0.16029241]), array([0.26065353, 0.18535341]), ('z+p+q', 0.07, 7, 1, 1e-06)), (array([0.2179253 , 0.14914392]), array([0.24430885, 0.18551911]), array([0.27077867, 0.19051428]), ('z+c+d', 0.07, 7, 1, 3e-05)), (array([0.21876178, 0.15133263]), array([0.24651553, 0.18783665]), array([0.27584988, 0.1900163 ]), ('z+p+q', 0.07, 7, 1, 3e-05))]
fixed best 1 [(array([0.2120078 , 0.14587414]), array([0.21688916, 0.15817159]), array([0.25510698, 0.18874602]), ('z+c+d', 0.07, 7, 1, 1e-06)), (array([0.21268279, 0.1482731 ]), array([0.21910693, 0.16029241]), array([0.26065353, 0.18535341]), ('z+p+q', 0.07, 7, 1, 1e-06)), (array([0.23319382, 0.16340247]), array([0.2465385 , 0.17847265]), array([0.29934717, 0.19571447]), ('z+p+q', 0.07, 7, 3, 1e-06)), (array([0.2312981 , 0.16138191]), array([0.2451828 , 0.17892754]), array([0.29540839, 0.1956182 ]), ('z+c+d', 0.07, 7, 3, 1e-06))]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=849a2da4efdd5266 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_849a2da4 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`53041 ms` LLM)
- **Rationale**: Promote the operator-informed baseline and compare structured challengers with tuning confined to the fitting rows of each split. Verify the saved artifact in an isolated process and report the schema validator's numerical result separately.
- **Alternatives Weighed**: `Retain the original polynomial kernel` (rejected: The operator-informed candidate improves both prescribed-split outputs, though neither yet passes the gate.)

```bash
cp /holdout_task/batch_solver.py /tmp/baseline_06.py
python3 - <<'PY'
p='/holdout_task/batch_solver.py'
s=open(p).read().split('def solve_batch')[0]
open(p,'w').write(s)
PY
cat >> /holdout_task/batch_solver.py <<'PY'
def kernel_bank(z,w,c,d,n,telemetry):
    p=pair(z); q=np.cos(2.4*z)
    def gram(x):
        x=x/np.maximum(x[:n].std(0),.1)
        return x@x.T/x.shape[1]
    def dist(g):
        return np.maximum(g.diagonal()[:,None]+g.diagonal()-2*g,0)
    Z,P,Q,W,C,D=map(gram,(z,p,q,w,c,d))
    base=1+Z+P+Q
    mix=Z*(P+Q)
    pp=P*P+P*Q+Q*Q
    extra=[]
    for x in (z,w,c,d,z+np.roll(z,1,axis=1),z-np.roll(z,1,axis=1), c+d,c-d):
        extra.append(gram(np.sin(1.25*x)))
        extra.append(gram(np.tanh(1.25*x)))
        extra.append(gram(x/(1+abs(float(telemetry[0]))*x*x)))
    e0=sum(extra)/len(extra)
    out=[]
    def add(k):
        for l in (1e-6,1e-5,1e-4,.001): out.append((k,l))
    for g in ((Z+C+D)/3,(Z+P+Q)/3,(Z+W+C+D)/4):
        for rate in (.025,.05,.07,.11):
            rr=np.exp(-rate*dist(g)); add(base+rr); add(base+rr+.05*mix)
    for a in (.01,.08,.5,2):
        for b in (.002,.03,.25):
            for e in (0,.03,.3):
                add(base+a*mix+b*pp+.02*Z**3+e*e0)
    for cc in (c,d,z,w,p,q):
        for j in range(5):
            zz=cc[:,j] if j<4 else np.linalg.norm(cc,axis=1)
            norm=1/(1+abs(float(telemetry[0]))*zz**2)
            cov=norm[:,None]*norm[None,:]
            out.append((base+.3*mix+.05*pp+.03*Z**3+cov*(base+mix),1e-4))
    return out


def predict(study, mode='safe'):
    tr=study['train']; n=len(tr['y'])
    z=align(study['panel_a'],np.vstack([tr['x'],study['test']['x']]))
    w,c,d=features(study,z)
    Y=np.c_[tr['y'],tr['traj']]
    # A broad squared-exponential correction on top of the measured operator basis.
    bank=kernel_bank(z,w,c,d,n,study['telemetry_vec'])
    if mode=='safe':
        K,lam=bank[8*2]
        return K[n:,:n]@solve(K[:n,:n]+lam*np.eye(n),Y,assume_a='pos')
    from scipy.linalg import cho_factor,cho_solve
    ranked=[]
    eye=np.eye(n)
    for K,lam in bank:
        cf=cho_factor(K[:n,:n]+lam*eye)
        alpha=cho_solve(cf,Y)
        diag=np.diag(cho_solve(cf,eye))
        err=np.abs(alpha/diag[:,None]).mean(0)
        ranked.append((err[0],err[1:].mean(),K[n:,:n]@alpha))
    pred=ranked[16][2].copy()
    for j,col in enumerate((slice(0,1),slice(1,None))):
        r=sorted(ranked,key=lambda x:x[j])[:8]
        p=np.mean([x[2][:,col] for x in r],axis=0)
        pred[:,col]=(.35*pred[:,col]+.65*p) if mode=='blend' else p
    return pred


def solve_batch(study):
    pred=predict(study,MODE)
    y,T=pred[:,0],pred[:,1:]
    return dict(y_hat=y,state_pred=T,diagnostics_vec=np.array([y.mean(),y.std(),T[:,-1].mean(),T[:,-1].std()]))

MODE='safe'
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
path='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('solver',path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=np.c_[tr['y'],tr['traj']]; order=[np.arange(72)]+[np.random.default_rng(i).permutation(72) for i in range(6)]
metrics={}
for mode in ('safe','blend','adaptive'):
    es=[]; last=[]
    for o in order:
        a,b=o[:52],o[52:]; st=dict(s,train={**tr,**{k:tr[k][a] for k in ('x','y','traj')}},test={'x':tr['x'][b]})
        e=abs(m.predict(st,mode)-y[b]); es.append([e[:,0].mean(),e[:,1:].mean()]); last.append(e[:,-1].mean())
    es=np.asarray(es); metrics[mode]=es
    print(mode,'prescribed',es[0],'random mean',es[1:].mean(0),'worst',es.max(0),'last-time worst',max(last))
base=metrics['safe']; mode=min(metrics,key=lambda k: metrics[k][1:].mean() if (metrics[k].max(0)<=base.max(0)*1.015).all() and (metrics[k][0]<=base[0]*1.015).all() else 100)
print('promoted',mode)
code=open(path).read().replace("MODE='safe'",f'MODE={mode!r}');open(path,'w').write(code)
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py; rc=$?; OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import runpy,pickle,numpy as np
m=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; o=m['solve_batch'](s); y,T=o['y_hat'],o['state_pred']; assert y.shape==(36,) and T.shape==(36,8); assert all(np.isfinite(v).all() for v in o.values()); np.testing.assert_array_equal(o['diagnostics_vec'],[y.mean(),y.std(),T[:,-1].mean(),T[:,-1].std()]); print('Isolated-process shape, finiteness and diagnostics verified.')
PY
exit "$rc"
```
- **Container Exit Code**: `1` (`4103 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
safe prescribed [0.21411347 0.15990836] random mean [0.21359305 0.1473374 ] worst [0.26087539 0.18916504] last-time worst 0.25587427046339917
blend prescribed [0.22777209 0.17296178] random mean [0.21953062 0.15239117] worst [0.32046614 0.19014946] last-time worst 0.2561152415734525
adaptive prescribed [0.25977412 0.18138603] random mean [0.23337462 0.15594552] worst [0.3824988 0.1907358] last-time worst 0.25668534554246725
promoted safe
{"schema_valid": false, "errors": ["public_cv_y_mae_exceeded:0.214113", "public_cv_traj_mae_exceeded:0.159908"]}
Isolated-process shape, finiteness and diagnostics verified.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e959cd5fa6ab2cc8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_e959cd5f (directory).
```

### Turn 14 — `submit_completed_task` (`14145 ms` LLM)
- **Rationale**: The saved solver improved both prescribed-split errors relative to the initial baseline and passed the output-contract checks, but it did not meet the numerical acceptance gate.
- **Alternatives Weighed**: `Promote the adaptive or blended kernel` (rejected: Both increased MAE for both outputs on the prescribed split.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6 CANDIDATE-HASH PRE-COMMIT GATE: Before locking `submit_completed_task`, emit ONE `execute_in_remote_docker` Python/test verification command against your current target artifact(s) to resolve: Declared artifacts (`sha256=e959cd5fa6ab`) lack a candidate-hash-bound post-modification `VerificationReceipt` (`true`/`ls`/`print` or double-submit do NOT qualify!).. Verify clean subprocess import/execution, no banned modules, finite outputs, and exact schema/shape, then call `submit_completed_task`.
- **Submitted Final Answer / Summary**: `Saved `/holdout_task/batch_solver.py`. Output shapes, finiteness, and diagnostics passed isolated-process checks. The public 52/20 validation did not pass: scalar MAE was 0.214113 and trajectory MAE was 0.159908, both above the 0.038 limit.`

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
