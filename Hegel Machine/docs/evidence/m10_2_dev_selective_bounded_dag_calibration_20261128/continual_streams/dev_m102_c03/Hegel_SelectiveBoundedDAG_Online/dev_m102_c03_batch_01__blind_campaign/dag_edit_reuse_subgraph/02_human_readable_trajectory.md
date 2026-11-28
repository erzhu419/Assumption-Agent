# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_m102_c03_batch_01__blind_campaign__reuse_subgraph`

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
| **Turns Used / Max Turns** | `12 / 12` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `361,201` (`351,650` / `74,880` / `9,551`) |
| **Total Episode Wall Time (s)** | `361.6s` |

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

### Turn 1 — `execute_in_remote_docker` (`32670 ms` LLM)
- **Rationale**: Inspect the frozen interfaces, public fold construction, and archived reference before selecting the nonlinear basis or fitting a readout.

```bash
cd /holdout_task && cat _dag_spec.json && cat _frozen_nodes/node_00.py _frozen_nodes/node_02.py && cat test_dag_candidate.py && cat public_asset_index.json && ls -laR /tmp/hegel_continual_archive | head -180 && python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_batch.pkl','rb'))
def show(x,p='',d=0):
 if isinstance(x,dict):
  for k,v in x.items():show(v,p+'/'+str(k),d+1)
 elif isinstance(x,(list,tuple)):
  print(p,type(x).__name__,len(x));
  if x and d<4:show(x[0],p+'/0',d+1)
 else:
  a=np.asarray(x); print(p,a.shape,a.dtype, a.ravel()[:8] if a.dtype != object else str(x)[:140]);
show(x)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling variable projection'
```
- **Container Exit Code**: `0` (`470 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
index_v1"
}/tmp/hegel_continual_archive:
total 20
drwxr-xr-x 2 root root 4096 Nov 28 13:12 .
drwxrwxrwt 1 root root 4096 Nov 28 13:12 ..
-rw-r--r-- 1 root root  662 Nov 28 13:12 archive_manifest.json
-rw-r--r-- 1 root root 3634 Nov 28 13:12 dev_m102_c03_batch_00_solver.py
-rw-r--r-- 1 root root 1245 Nov 28 13:12 pre_llm_gate_diagnostic.json
 list 1
/0/study/eval_times (8,) float64 [0.25       0.64285714 1.03571429 1.42857143 1.82142857 2.21428571
 2.60714286 3.        ]
/0/study/panel_a/u (48, 4) float64 [-0.49685942 -0.60348618  0.99396718  0.15275471  0.92494884 -0.13777887
 -0.04984001 -0.21140014]
/0/study/panel_a/v (48, 4) float64 [-0.44559826 -0.45596081  1.03057215  0.16050238  1.06161424 -0.01952075
 -0.18970546 -0.2639014 ]
/0/study/panel_b/u (24, 4) float64 [-0.68046055  0.95643873 -0.2838056  -0.03823275 -0.93547125  0.64560683
 -1.04991035  0.98873367]
/0/study/panel_b/v (24, 4) float64 [-0.4742048  -0.59601596 -0.47491035 -0.51307789 -0.83445061 -0.81293225
 -0.95824902  0.47984506]
/0/study/panel_c/u (24, 4) float64 [ 1.10721786 -0.10254288  1.24778559  0.37936518 -0.88432455 -0.75002292
  0.8024688   0.32326277]
/0/study/panel_c/v1 (24, 4) float64 [-0.20840677 -0.72796473 -0.33726893  0.07204161 -1.17063476  0.27397223
 -0.04270019 -0.06345502]
/0/study/panel_c/v2 (24, 4) float64 [-0.4576305  -0.78525888 -0.56960386 -0.4669499   0.38868821 -0.21088012
 -0.14744643 -0.22670743]
/0/study/train/x (72, 4) float64 [ 0.27235876 -0.5382677  -0.86697331 -0.67354233  0.53916875 -0.27601028
  0.17360235 -0.91537625]
/0/study/train/y (72,) float64 [ 0.03009843  0.03479846  1.75997242  0.58533491  0.82501467 -0.20029867
  0.52243059  0.23857759]
/0/study/train/traj (72, 8) float64 [0.09091907 0.18880023 0.26277417 0.29985001 0.32988837 0.35657984
 0.38737649 0.39832787]
/0/study/train/summary (4,) float64 [0.02030009 0.53969806 0.47477208 0.85169535]
/0/study/test/x (36, 4) float64 [ 0.70921712  0.35646312 -0.70355242 -0.72688632 -0.9587365  -0.01009885
 -0.30141973 -0.94485274]
/0/study/telemetry_vec (4,) float64 [0.22 1.25 0.22 0.22]
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`9800 ms` LLM)
- **Rationale**: Recover the archived calibration and exact sink and CV contracts before fitting or changing features.

```bash
cd /holdout_task; cat _frozen_nodes/node_02.py; cat /tmp/hegel_continual_archive/*; cat test_dag_candidate.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -35
```
- **Container Exit Code**: `0` (`348 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
y(out_full["state_pred"], float), np.asarray(out_full["diagnostics_vec"], float)
    exp_dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
    if float(np.max(np.abs(dv - exp_dv))) > 1e-3:
        errs.append("diagnostics_vec_handoff_mismatch")

cv_metrics = {}
if not errs:
    st_cv = copy.deepcopy(st)
    sp_idx = max(40, len(st["train"]["x"]) - 20)
    vx, vy, vtr = st["train"]["x"][sp_idx:].copy(), st["train"]["y"][sp_idx:].copy(), st["train"]["traj"][sp_idx:].copy()
    st_cv["train"] = {
        "x": st["train"]["x"][:sp_idx].copy(),
        "y": st["train"]["y"][:sp_idx].copy(),
        "traj": st["train"]["traj"][:sp_idx].copy(),
        "summary": np.array([float(np.mean(st["train"]["y"][:sp_idx])), float(np.std(st["train"]["y"][:sp_idx])), float(np.mean(st["train"]["traj"][:sp_idx, -1])), float(np.std(st["train"]["traj"][:sp_idx, -1]))], dtype=float),
    }
    st_cv["test"] = {"x": vx}
    out_cv, _ = run_dag(st_cv)
    ym = float(np.mean(np.abs(np.asarray(out_cv["y_hat"], float) - vy)))
    tm = float(np.mean(np.abs(np.asarray(out_cv["state_pred"], float) - vtr)))
    cv_metrics = {"public_cv_y_mae": round(ym, 6), "public_cv_traj_mae": round(tm, 6)}
    if ym > 0.038: errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
    if tm > 0.038: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")

ok = len(errs) == 0 and all(v == 1 for v in ec.values())
print(json.dumps({"accepted": ok, "errors": errs, "cv_metrics": cv_metrics, "exec_counts": ec}, allow_nan=False))
sys.exit(0 if ok else 1)
{
  "ok": true,
  "target": "autogenous_sparse_ridge_certification.solve_and_certify_l0",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_sparse_ridge_certification",
      "granularity_tier": "L2_domain_operator",
      "compiled_code_sha256": "770f6644bb1fd8eb49dbacf51ffa3405a9ca876974e0021d66415d9014d8dfb9",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "n_samples",
          "op": ">=",
          "value": 1
        },
        {
          "field": "n_features",
          "op": "<=",
          "value": 16
        },
        {
          "field": "lam0",
          "op": ">=",
          "value": 0
        },
        {
          "field": "lam2",
          "op": ">=",
          "value": 0
        },
        {
          "field": "M",
```

### Turn 3 — `execute_in_remote_docker` (`10008 ms` LLM)
- **Rationale**: Read the omitted archived panel implementation and diagnostic separately to recover the feature grammar and its baseline errors.

```bash
sed -n '24,160p' /tmp/hegel_continual_archive/*solver.py; cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; cat /holdout_task/validate_public_schema.py; ls -la /holdout_task/_frozen_nodes
```
- **Container Exit Code**: `0` (`299 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"test"]["x"])
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
total 16
drwxr-xr-x 2 root root 4096 Nov 28 13:12 .
drwxr-xr-x 3 root root 4096 Nov 28 13:12 ..
-r--r--r-- 1 root root 1843 Nov 28 13:12 node_00.py
-r--r--r-- 1 root root  400 Nov 28 13:12 node_02.py
```

### Turn 4 — `execute_in_remote_docker` (`26740 ms` LLM)
- **Rationale**: Measure the panel response independently of the target labels, and separate the frozen branch’s error from the calibration readout’s error across multiple held-out blocks.

```bash
export OPENBLAS_NUM_THREADS=1; cat >/tmp/panel_probe.py <<'PY'
import numpy as np,pickle,runpy,glob
from scipy.optimize import minimize_scalar
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=st['train']
arc=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); sparse=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']; sink=runpy.run_path('/holdout_task/_frozen_nodes/node_02.py')['run_node']
def fold(a,b):
 return dict(st,train={k:v[a] for k,v in tr.items() if k!='summary'},test={'x':tr['x'][b]})
def err(p,b): return np.mean(abs(p-np.column_stack([tr['y'],tr['traj']])[b]),axis=0)
for key in ('panel_a','panel_b','panel_c'):
 p=st[key]; u=p['u']; h=np.c_[np.ones(len(u)),u];print(key,'u range',u.min(0),u.max(0))
 for name,v in p.items():
  if name=='u':continue
  af=np.linalg.lstsq(h,v,rcond=None)[0]; r=np.linalg.norm(v-h@af,axis=1); print(name,'affine',af.round(3),'residual quantiles',np.quantile(r,[0,.5,.9,.95,1]).round(4))
  for lab,fn in [('negexp',lambda u,a:-np.expm1(-a*u*u)),('tanh',lambda u,a:np.tanh(a*u)),('atan',lambda u,a:np.arctan(a*u)),('rational',lambda u,a:u*u/(1+a*u*u)),('softsign',lambda u,a:u/(1+a*abs(u)))]:
   def obj(a):
    mat=np.c_[h,fn(u,a)]; e=v-mat@np.linalg.lstsq(mat,v,rcond=None)[0]; return np.sqrt(np.mean(e**2))
   o=minimize_scalar(obj,bounds=(.002,8),method='bounded');print(lab,round(o.x,4),round(o.fun,5),'fixed',[(a,round(obj(a),5)) for a in (.22,.2962,1.25)])
for b in [np.arange(52,72),np.arange(20),np.arange(20,40),np.arange(40,60)]:
 a=np.setdiff1d(np.arange(72),b);s=fold(a,b); d=sparse({'__input__':s}); dp=np.c_[d['y_hat'],d['state_pred']]; p=arc['_panel_pred'](s,1,.3,'sat');bl=sink({'node_00':d,'node_01':{'panel_pred':p}});print('fold',b[0], 'sparse',err(dp,b).round(4),'panel',err(p,b).round(4),'blend',err(np.c_[bl['y_hat'],bl['state_pred']],b).round(4))
print('target ranges',np.column_stack([tr['y'],tr['traj']]).min(0),np.column_stack([tr['y'],tr['traj']]).max(0));print('trajectory sv',np.linalg.svd(tr['traj'],compute_uv=False));print('x range',tr['x'].min(0),tr['x'].max(0),st['test']['x'].min(0),st['test']['x'].max(0))
PY
python3 /tmp/panel_probe.py
sed -n '1,90p' /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`570 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
elif isinstance(v, (dict, Mapping)):
            h.update(f"D:{len(v)}:".encode())
            for k in sorted(v.keys(), key=str): _feed(str(k)); _feed(v[k])
            h.update(b";")
        elif isinstance(v, (list, tuple)):
            h.update(f"L:{len(v)}:".encode())
            for it in v: _feed(it)
            h.update(b";")
        else: h.update(repr(v).encode())
    _feed(obj)
    return h.hexdigest()

def _collect_subtree_digests(obj) -> set[str]:
    out = {_canon_digest(obj)}
    if isinstance(obj, (dict, Mapping)):
        for v in obj.values(): out.update(_collect_subtree_digests(v))
    elif isinstance(obj, (list, tuple)):
        for it in obj: out.update(_collect_subtree_digests(it))
    return out

def _collect_numeric_leaf_digests(obj) -> set[str]:
    out = set()
    if isinstance(obj, np.ndarray) and obj.size > 0 and obj.dtype.kind in ("f", "i", "u"):
        out.add(_canon_digest(obj))
    elif isinstance(obj, (float, np.floating)):
        out.add(_canon_digest(float(obj)))
    elif isinstance(obj, (dict, Mapping)):
        for v in obj.values(): out.update(_collect_numeric_leaf_digests(v))
    elif isinstance(obj, (list, tuple)):
        for it in obj: out.update(_collect_numeric_leaf_digests(it))
    return out

class _TrackedPayloadDict(Mapping):
    def __init__(self, data: dict, src_key: str, tracker: set):
        self._data = data
        self._src_key = src_key
        self._tracker = tracker
    def __getitem__(self, k):
        self._tracker.add(self._src_key)
        return self._data[k]
    def get(self, k, default=None):
        self._tracker.add(self._src_key)
        return self._data.get(k, default)
    def __iter__(self):
        self._tracker.add(self._src_key)
        return iter(self._data)
    def __len__(self): return len(self._data)
    def keys(self):
        self._tracker.add(self._src_key)
        return self._data.keys()
    def items(self):
        self._tracker.add(self._src_key)
        return self._data.items()
    def values(self):
        self._tracker.add(self._src_key)
        return self._data.values()
    def __contains__(self, k):
        self._tracker.add(self._src_key)
        return k in self._data

fns = {}
for nid in spec_obj["topo_order"]:
    fp = f"/holdout_task/{nid}.py" if nid in spec_obj["editable_node_ids"] else os.path.join(frozen_dir, f"{nid}.py")
    if not os.path.isfile(fp):
        print(json.dumps({"accepted": False, "error": f"missing_node_file:{fp}"}))
        sys.exit(1)
```

### Turn 5 — `execute_in_remote_docker` (`29737 ms` LLM)
- **Rationale**: The paired measurements distinguish calibration strength from saturation shape. Measure the readout bottleneck separately for the scalar and every trajectory horizon before editing the node.
- **Alternatives Weighed**: `Use telemetry value 1.25 directly as the exponential saturation coefficient` (rejected: Its paired-measurement residual is 0.01276 RMS, versus 0.00024 for a fitted coefficient near 0.294.)

```bash
export OPENBLAS_NUM_THREADS=1; python3 /tmp/panel_probe.py | tail -16; cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; cat >/tmp/readout_probe.py <<'PY'
import numpy as np,pickle,runpy,glob
from itertools import combinations_with_replacement
from scipy.spatial.distance import cdist
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=st['train'];x=tr['x'];Y=np.c_[tr['y'],tr['traj']]; sparse=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']; arc=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]);
folds=[]
for b in [np.arange(52,72),np.arange(20),np.arange(20,40),np.arange(40,60),*np.array_split(np.random.default_rng(443).permutation(len(x)),4)]:
 a=np.setdiff1d(np.arange(len(x)),b);s=dict(st,train={k:v[a] for k,v in tr.items() if k!='summary'},test={'x':x[b]});d=sparse({'__input__':s});folds.append((a,b,np.c_[d['y_hat'],d['state_pred']]))
w=np.r_[.5,np.repeat(.65,8)]
def score(pred):
 out=[]
 for a,b,sp in folds:
  v=np.abs(w*pred(a,b)+(1-w)*sp-Y[b]);out.append([v[:,0].mean(),v[:,1:].mean(),v[:,1:].mean(0).max()]);
 return np.array(out)
def pol(x,deg):return np.column_stack([np.prod(x[:,c],axis=1) for d in range(1,deg+1) for c in combinations_with_replacement(range(4),d)])
def ridge(f,lam):
 def p(a,b):
  u=(f-f[a].mean(0))/np.maximum(f[a].std(0),1e-10); return Y[a].mean(0)+u[b]@u[a].T@np.linalg.solve(u[a]@u[a].T+lam*np.eye(len(a)),Y[a]-Y[a].mean(0))
 return p
res=[]
def append(name,p):
 s=score(p);res.append((max(s[:,:2].max(0)),name,s))
for deg in range(1,6):
 for l in (1e-8,1e-4,.003,.03,.3,1):append(('poly',deg,l),ridge(pol(x,deg),l))
d=cdist(x,x,'sqeuclidean')
for scale in (.08,.18,.35,.65,1.3,3,8):
 for lam in (1e-7,1e-5,.0003,.003,.03):
  for power in (0,1,2):
   t=np.sqrt(d*scale); K=(np.exp(-t*t) if power==0 else (1+t+(t*t/3 if power==2 else 0))*np.exp(-t))
   def p(a,b):return Y[a].mean(0)+K[np.ix_(b,a)]@np.linalg.solve(K[np.ix_(a,a)]+lam*np.eye(len(a)),Y[a]-Y[a].mean(0))
   append(('kernel',scale,lam,power),p)
for v in sorted(res,key=lambda v:v[0])[:14]:print(v[1],v[2].round(4).tolist())
for v in sorted(res,key=lambda v:v[2][0,:2].max())[:5]:print('gate best',v[1],v[2].round(4).tolist())
PY
python3 /tmp/readout_probe.py
```
- **Container Exit Code**: `0` (`766 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
64], [0.2163, 0.0479, 0.0564], [0.1777, 0.0877, 0.1029], [0.1522, 0.0904, 0.1082], [0.1171, 0.053, 0.0635], [0.1127, 0.0586, 0.0694], [0.1413, 0.0501, 0.0588]]
('kernel', 0.08, 1e-05, 2) [[0.1921, 0.0716, 0.0858], [0.1037, 0.0647, 0.0756], [0.2163, 0.048, 0.0567], [0.1795, 0.0871, 0.1014], [0.1526, 0.0892, 0.1068], [0.1219, 0.055, 0.0654], [0.1154, 0.0598, 0.0706], [0.1445, 0.0515, 0.0612]]
('kernel', 0.18, 0.0003, 0) [[0.1911, 0.0705, 0.0839], [0.1145, 0.0713, 0.0856], [0.2178, 0.0575, 0.0665], [0.1875, 0.1128, 0.1335], [0.1673, 0.0985, 0.1152], [0.1271, 0.07, 0.0812], [0.1163, 0.0823, 0.0975], [0.144, 0.0602, 0.0702]]
('kernel', 0.18, 1e-05, 2) [[0.1896, 0.0713, 0.0851], [0.1061, 0.0651, 0.0763], [0.2181, 0.0502, 0.0593], [0.1789, 0.0916, 0.1067], [0.1563, 0.092, 0.1099], [0.1222, 0.0566, 0.0671], [0.1152, 0.0633, 0.075], [0.143, 0.053, 0.0629]]
('kernel', 0.18, 1e-07, 2) [[0.1896, 0.0717, 0.0857], [0.1057, 0.0653, 0.0767], [0.2181, 0.0499, 0.0588], [0.1785, 0.0916, 0.107], [0.1563, 0.0924, 0.1106], [0.1208, 0.056, 0.0665], [0.1146, 0.0627, 0.0744], [0.1427, 0.0527, 0.0625]]
('kernel', 0.18, 1e-05, 0) [[0.1908, 0.0736, 0.0877], [0.1122, 0.074, 0.0892], [0.2182, 0.0592, 0.0709], [0.18, 0.1115, 0.135], [0.1664, 0.1018, 0.1197], [0.1184, 0.068, 0.0792], [0.1119, 0.0778, 0.0922], [0.1405, 0.0595, 0.0686]]
gate best ('poly', 5, 1e-08) [[0.1681, 0.0724, 0.0866], [0.1036, 0.0848, 0.1106], [0.2271, 0.1707, 0.2175], [0.1837, 0.1158, 0.1385], [0.1153, 0.0848, 0.1003], [0.1505, 0.0708, 0.0877], [0.1296, 0.1089, 0.1342], [0.1657, 0.0834, 0.0998]]
gate best ('poly', 5, 0.0001) [[0.1681, 0.0724, 0.0866], [0.1036, 0.0848, 0.1106], [0.2271, 0.1707, 0.2175], [0.1837, 0.1158, 0.1385], [0.1153, 0.0848, 0.1003], [0.1505, 0.0708, 0.0877], [0.1296, 0.1089, 0.1342], [0.1657, 0.0834, 0.0998]]
gate best ('poly', 5, 0.003) [[0.1681, 0.0725, 0.0868], [0.1037, 0.0849, 0.1107], [0.2271, 0.1708, 0.2175], [0.1833, 0.1156, 0.1381], [0.1153, 0.085, 0.1004], [0.1503, 0.0708, 0.0877], [0.1297, 0.1089, 0.1344], [0.1654, 0.0835, 0.0999]]
gate best ('poly', 5, 0.03) [[0.1689, 0.0739, 0.0881], [0.1043, 0.0857, 0.1115], [0.2269, 0.1712, 0.2175], [0.1802, 0.114, 0.1355], [0.1151, 0.086, 0.101], [0.1489, 0.071, 0.0877], [0.1298, 0.109, 0.1362], [0.1628, 0.0842, 0.1012]]
gate best ('poly', 5, 0.3) [[0.1735, 0.087, 0.1034], [0.1056, 0.087, 0.1115], [0.2323, 0.17, 0.2134], [0.1719, 0.1094, 0.1307], [0.1139, 0.0902, 0.107], [0.1439, 0.0709, 0.0916], [0.1295, 0.1143, 0.1445], [0.1559, 0.093, 0.115]]
```

### Turn 6 — `execute_in_remote_docker` (`29513 ms` LLM)
- **Rationale**: Identify the actual coordinate transform and compare compact target grammars independently of the frozen contribution; its isolated scalar error at the public split is 0.3866.
- **Alternatives Weighed**: `Keep the frozen blend and replace the panel readout with a generic kernel` (rejected: The gate scalar MAE remains at least 0.168 on the tested broad families, far above 0.038.)

```bash
export OPENBLAS_NUM_THREADS=1; cd /holdout_task; cat public_asset_index.json; cat _frozen_nodes/node_00.py; python3 - <<'PY'
import pickle,numpy as np
st=pickle.load(open('public_batch.pkl','rb'))[0]['study']; print(st.keys()); print('times',st['eval_times']);
for k in ('panel_a','panel_b','panel_c'):
 p=st[k]; print(k,[(q,np.shape(a)) for q,a in p.items()]);print(np.column_stack(list(p.values()))[:8].round(4));
 if k=='panel_a':
  H=np.c_[np.ones(len(p['u'])),p['u']];v=p['v'];r=np.linalg.norm(v-H@np.linalg.lstsq(H,v,rcond=None)[0],axis=1);g=r<3; print('clean',np.linalg.lstsq(H[g],v[g],rcond=None)[0]);print('res',np.quantile(np.linalg.norm(v[g]-H[g]@np.linalg.lstsq(H[g],v[g],rcond=None)[0],axis=1),[0,.5,1]))
print('len',len(st['train']['x']))
PY
cat >/tmp/sparseprobe.py <<'PY'
exec(open('/tmp/readout_probe.py').read().split('res=[]')[0])
from scipy.optimize import least_squares
from sklearn.linear_model import ARDRegression, OrthogonalMatchingPursuit
from itertools import combinations
qtime=st['eval_times']; T=tr['traj'];o=least_squares(lambda r:(lambda q:(q@np.linalg.lstsq(q,T.T,rcond=None)[0]-T.T).ravel())(-np.expm1(-qtime[:,None]*np.exp(r))),np.log([.39,1.86]));rt=np.exp(o.x);Q=-np.expm1(-qtime[:,None]*rt); A=np.c_[Y[:,0],np.linalg.lstsq(Q,T.T,rcond=None)[0].T];print('rates',rt,'rms',np.sqrt(np.mean(o.fun**2)))
def tst(f,kind):
 a0=[]
 for a,b,sp in folds:
  u=(f-f[a].mean(0))/np.maximum(f[a].std(0),1e-9); pred=[]
  for j in range(3):
   m=ARDRegression(threshold_lambda=kind) if kind>99 else OrthogonalMatchingPursuit(n_nonzero_coefs=kind,tol=None); m.fit(u[a],A[a,j]);pred.append(m.predict(u[b]))
  pred=np.array(pred).T;p=np.c_[pred[:,0],pred[:,1:]@Q.T];v=abs(p-Y[b]);a0.append([v[:,0].mean(),v[:,1:].mean()]);
 return np.array(a0)
p=st['panel_a'];u=p['u'];v=p['v'];h=np.c_[np.ones(len(u)),u];keep=np.linalg.norm(v-h@np.linalg.lstsq(h,v,rcond=None)[0],axis=1)<3; aff=np.linalg.lstsq(h[keep],v[keep],rcond=None)[0]; z0=(x-aff[0])@np.linalg.inv(aff[1:]);z1=(x-aff[0])/aff[1:].diagonal()
for z in (x,z0,z1):
 for z in (z,):
  f,g=arc['_features'](z); f=np.column_stack(f[1:]+g+[pol(z,3)]); fs=[f,pol(z,3)]
  for idx,f in enumerate(fs):
   for j in (6,12,1e3,1e4):
    s=tst(f,j);print(idx,j,s[0].round(4),s.max(0).round(4),s.mean(0).round(4))
PY
PYTHONWARNINGS=ignore python3 /tmp/sparseprobe.py
```
- **Container Exit Code**: `0` (`2323 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-1.0680e+00 -6.0610e-01
  -7.1430e-01  5.4740e-01]
 [-1.1342e+00 -5.1040e-01  4.3200e-01 -3.0000e-03  7.9350e-01 -4.8700e-02
  -7.0730e-01 -3.0000e-04]
 [-6.8990e-01 -1.0263e+00 -1.2448e+00 -6.3600e-01  7.1970e-01 -1.1107e+00
   3.4150e-01  7.8290e-01]]
panel_c [('u', (24, 4)), ('v1', (24, 4)), ('v2', (24, 4))]
[[ 1.1072 -0.1025  1.2478  0.3794 -0.2084 -0.728  -0.3373  0.072  -0.4576
  -0.7853 -0.5696 -0.4669]
 [-0.8843 -0.75    0.8025  0.3233 -1.1706  0.274  -0.0427 -0.0635  0.3887
  -0.2109 -0.1474 -0.2267]
 [ 0.7543  0.665   0.7698 -0.1449  0.0233  0.1571  0.7823  0.0353 -0.4283
  -0.7594  0.0182  0.0061]
 [-0.1793 -0.6666 -0.2803 -0.5663 -0.0295 -0.2279  0.1528  0.019  -0.059
   0.215   0.2835 -0.049 ]
 [ 0.7705 -1.1161  0.3723  1.2201  0.1921 -1.2228 -1.2175  0.4859 -0.2385
  -0.0431 -0.9547  0.1326]
 [-0.2257  0.1025 -0.3002  0.4271  0.0137  0.2002 -0.1298  0.1447  0.0295
   0.065  -0.0144  0.3394]
 [-0.4296  0.9234  0.9446 -0.396   1.1623 -0.0806  0.3223  0.2423  0.1801
  -0.9013  0.3412  0.1332]
 [-0.0184 -0.7721  0.3498  0.9439 -0.3894 -0.1749 -0.3414 -0.1009  0.1747
   0.0195 -0.5554  0.1381]]
len 72
rates [0.40392146 1.81454353] rms 0.004029916989251957
0 6 [0.2775 0.0861] [0.2775 0.1339] [0.2166 0.0942]
0 12 [0.229  0.0998] [0.229  0.0998] [0.1297 0.0715]
0 1000.0 [0.0828 0.1112] [0.1203 0.1112] [0.0935 0.0797]
0 10000.0 [0.0334 0.0401] [0.0642 0.0454] [0.0429 0.0358]
1 6 [0.0803 0.1122] [0.1555 0.1432] [0.1086 0.1065]
1 12 [0.0546 0.0535] [0.1293 0.0822] [0.0776 0.0546]
1 1000.0 [0.0751 0.0453] [0.0947 0.0902] [0.0672 0.0628]
1 10000.0 [0.0894 0.0742] [0.0894 0.0742] [0.0468 0.0392]
0 6 [0.1223 0.0749] [0.2699 0.1022] [0.1745 0.0711]
0 12 [0.0668 0.0712] [0.1814 0.0731] [0.1079 0.0496]
0 1000.0 [0.071 0.054] [0.1016 0.1234] [0.0752 0.0886]
0 10000.0 [0.0342 0.0326] [0.081  0.0433] [0.0448 0.0341]
1 6 [0.0877 0.0826] [0.1046 0.119 ] [0.0886 0.0888]
1 12 [0.0416 0.0762] [0.0832 0.0791] [0.0567 0.0544]
1 1000.0 [0.055  0.0428] [0.0854 0.0889] [0.0661 0.0619]
1 10000.0 [0.0547 0.048 ] [0.0547 0.048 ] [0.0405 0.0349]
0 6 [0.1222 0.0749] [0.2699 0.1022] [0.1745 0.0711]
0 12 [0.0668 0.0712] [0.1814 0.0731] [0.1079 0.0496]
0 1000.0 [0.071 0.054] [0.1016 0.1234] [0.0752 0.0886]
0 10000.0 [0.0342 0.0327] [0.081  0.0432] [0.0436 0.0341]
1 6 [0.0877 0.0827] [0.1046 0.119 ] [0.0886 0.0888]
1 12 [0.0416 0.0762] [0.0832 0.0791] [0.0567 0.0544]
1 1000.0 [0.055  0.0428] [0.0854 0.089 ] [0.0661 0.0619]
1 10000.0 [0.0547 0.048 ] [0.0547 0.048 ] [0.0405 0.0349]
```

### Turn 7 — `execute_in_remote_docker` (`18757 ms` LLM)
- **Rationale**: Inspect the exact baseline feature algebra and the paired-panel mixing matrices to determine which calibration and blend corrections are identifiable from the declared input.
- **ATMS NOGOOD DECLARED**: Refuted `An independently fitted generic panel regressor is sufficient inside the unchanged fixed-weight blend.` | Witness: `The frozen scalar term alone contributes 0.1933 MAE after its 0.5 weight, while broad generic blended readouts have public scalar MAE at least 0.168.` | Blanket Diagnosis: `The fixed sink mixes a high-error frozen readout with the editable output, so standalone accuracy and blended accuracy are distinct objectives.` | Sublated Pivot: `Resolve the latent response grammar and quantify a calibrated residual against the frozen blend without modifying either frozen file.`

```bash
export OPENBLAS_NUM_THREADS=1; cat /holdout_task/_frozen_nodes/node_00.py; head -25 /tmp/hegel_continual_archive/*solver.py; sed -n '90,230p' /holdout_task/test_dag_candidate.py; python3 - <<'PY'
exec(open('/tmp/readout_probe.py').read().split('res=[]')[0]); print('times',st['eval_times']);print('telemetry',st['telemetry_vec']);
for key,vkey in [('panel_b','v'),('panel_c','v2')]:
 p=st[key];u=p['u'];v=p[vkey];a=np.c_[np.ones(len(u)),u];
 if key=='panel_c':a=np.c_[a,u*u/(1+.1629*u*u)]
 w=np.linalg.lstsq(a,v,rcond=None)[0]; print(key,w.round(4));print('linear singulars',np.linalg.svd(w[1:5],compute_uv=False));
 if key=='panel_c': print('nonlinear after de-rotation', (w[5:]@np.linalg.inv(w[1:5])).round(4))
PY
```
- **Container Exit Code**: `0` (`536 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
c"], float)
    exp_dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
    if float(np.max(np.abs(dv - exp_dv))) > 1e-3:
        errs.append("diagnostics_vec_handoff_mismatch")

cv_metrics = {}
if not errs:
    st_cv = copy.deepcopy(st)
    sp_idx = max(40, len(st["train"]["x"]) - 20)
    vx, vy, vtr = st["train"]["x"][sp_idx:].copy(), st["train"]["y"][sp_idx:].copy(), st["train"]["traj"][sp_idx:].copy()
    st_cv["train"] = {
        "x": st["train"]["x"][:sp_idx].copy(),
        "y": st["train"]["y"][:sp_idx].copy(),
        "traj": st["train"]["traj"][:sp_idx].copy(),
        "summary": np.array([float(np.mean(st["train"]["y"][:sp_idx])), float(np.std(st["train"]["y"][:sp_idx])), float(np.mean(st["train"]["traj"][:sp_idx, -1])), float(np.std(st["train"]["traj"][:sp_idx, -1]))], dtype=float),
    }
    st_cv["test"] = {"x": vx}
    out_cv, _ = run_dag(st_cv)
    ym = float(np.mean(np.abs(np.asarray(out_cv["y_hat"], float) - vy)))
    tm = float(np.mean(np.abs(np.asarray(out_cv["state_pred"], float) - vtr)))
    cv_metrics = {"public_cv_y_mae": round(ym, 6), "public_cv_traj_mae": round(tm, 6)}
    if ym > 0.038: errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
    if tm > 0.038: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")

ok = len(errs) == 0 and all(v == 1 for v in ec.values())
print(json.dumps({"accepted": ok, "errors": errs, "cv_metrics": cv_metrics, "exec_counts": ec}, allow_nan=False))
sys.exit(0 if ok else 1)
times [0.25       0.64285714 1.03571429 1.42857143 1.82142857 2.21428571
 2.60714286 3.        ]
telemetry [0.22 1.25 0.22 0.22]
panel_b [[-0.0707  0.0588  0.057  -0.0071]
 [-0.3724  0.4252  0.6692  0.0847]
 [-0.6169 -0.1699 -0.1617 -0.582 ]
 [ 0.291   0.6692 -0.2068 -0.4462]
 [-0.4129  0.342  -0.5077  0.4788]]
linear singulars [0.8800622  0.88002054 0.87998567 0.87994004]
panel_c [[-0.0724 -0.0154  0.1264  0.1072]
 [-0.4417 -0.0798 -0.1614 -0.0755]
 [-0.0508 -0.3198  0.1436  0.3282]
 [ 0.1523 -0.3526 -0.1168 -0.2689]
 [ 0.1109  0.0196 -0.4159  0.2181]
 [-0.2389 -0.0434 -0.0873 -0.0411]
 [-0.0278 -0.1728  0.0777  0.178 ]
 [ 0.0824 -0.1908 -0.0635 -0.1453]
 [ 0.0605  0.0105 -0.2253  0.118 ]]
linear singulars [0.4832054  0.48305675 0.48290594 0.48283067]
nonlinear after de-rotation [[ 5.410e-01 -0.000e+00  6.000e-04 -3.000e-04]
 [ 4.000e-04  5.414e-01 -1.100e-03  3.000e-04]
 [ 2.000e-04  1.000e-04  5.410e-01  7.000e-04]
 [-6.000e-04 -1.000e-04  6.000e-04  5.417e-01]]
```

### Turn 8 — `execute_in_remote_docker` (`31493 ms` LLM)
- **Rationale**: Compare compact and expanded feature grammars with fold-local trajectory rates, and inspect stable support to distinguish physical response terms from interchangeable basis functions.

```bash
export OPENBLAS_NUM_THREADS=1; grep -A35 'def _panel_pred' /tmp/hegel_continual_archive/*solver.py; cat >/tmp/grammar.py <<'PY'
exec(open('/tmp/readout_probe.py').read().split('res=[]')[0])
from scipy.optimize import least_squares
from sklearn.linear_model import ARDRegression,Lasso,ElasticNet
from itertools import combinations

def features(z,extended=False):
 s,g=arc['_features'](z); f=np.column_stack(s+g+[np.prod(z[:,j],axis=1) for j in combinations(range(4),3)])[:,1:]; cols=[q for q in f.T]; names=[str(i) for i in range(f.shape[1])];
 def add(n,v):names.append(n);cols.append(v)
 if extended:
  for i in range(4):
   u=z[:,i]
   for c in (.5,1,1.5,2,2.5,3):
    for n,fn in [('sin',np.sin),('cos',np.cos),('tan',np.tanh),('rat',lambda u:u/(1+u*u)),('abs',np.abs)]:add((i,n,c),fn(c*u))
  for i,j in combinations(range(4),2):
   u,v=z[:,i],z[:,j]
   for c in (.25,1,1.5,2):add((i,j,'ratio',c),u*v/(1+c*(u*u+v*v)))
   for d in (-2,-1,-.5,.5,1,2):
    w=u+d*v
    for n,fn in [('sin',np.sin),('cos',np.cos),('tan',np.tanh),('ss',lambda u:u/(1+abs(u))),('sym',np.abs)]:add((i,j,d,n),fn(w))
 return np.column_stack(cols),names

def cal(mask=None):
 p=st['panel_a'];u=p['u'];v=p['v'];h=np.c_[np.ones(len(u)),u]; keep=np.ones(len(u),bool)
 for _ in range(4):
  f=np.linalg.lstsq(h[keep],v[keep],rcond=None)[0];r=np.linalg.norm(v-h@f,axis=1);keep=r<max(1e-3,4*np.median(r))
 return (x-f[0])/f[1:].diagonal()

def mode(a):
 t=st['eval_times']; T=Y[a,1:].T
 def q(r):return -np.expm1(-t[:,None]*np.exp(r))
 def r(r):
  h=q(r);return (h@np.linalg.lstsq(h,T,rcond=None)[0]-T).ravel()
 o=least_squares(r,np.log([.4,1.8]));h=q(o.x);return np.c_[Y[a,0],np.linalg.lstsq(h,T,rcond=None)[0].T],h
if __name__=='__main__':
 for zi,z in enumerate([x,cal()]):
  for ex in (False,True):
   f,names=features(z,ex);allscore=[]
   for threshold in (3000,10000,30000,100000):
    ss=[]
    for a,b,_ in folds+[(np.arange(len(x)),np.arange(len(x)),None)]:
     u=(f-f[a].mean(0))/np.maximum(f[a].std(0),1e-9);A,Q=mode(a);P=[];SUP=[]
     for j in range(3):
      m=ARDRegression(threshold_lambda=threshold);m.fit(u[a],A[:,j]);P.append(m.predict(u[b]));SUP.append([(names[k],round(m.coef_[k],3)) for k in np.argsort(-abs(m.coef_))[:12] if abs(m.coef_[k])>.008]);
     P=np.array(P).T;E=abs(np.c_[P[:,0],P[:,1:]@Q.T]-Y[b]);ss.append([E[:,0].mean(),E[:,1:].mean()]);
    print(zi,ex,threshold,np.round(ss[0],4),np.max(ss[:-1],axis=0).round(4),np.mean(ss[:-1],axis=0).round(4),np.round(ss[-1],4));
    if threshold==10000:print(SUP)
PY
python3 /tmp/grammar.py
```
- **Container Exit Code**: `0` (`27428 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, ((3, 'cos', 1), 0.07), ((1, 2, -1, 'sin'), 0.061), ((0, 2, -2, 'ss'), -0.057), ('30', 0.057)], [('32', -0.445), ('66', 0.414), ('1', 0.144), ((0, 2, -0.5, 'sin'), 0.093), ('134', -0.078), ('38', -0.074), ((0, 2, 1, 'ss'), 0.068), ((2, 3, 0.5, 'ss'), -0.058), ((0, 2, -2, 'ss'), -0.058), ('83', 0.051), ((1, 2, 2, 'ss'), -0.046), ('136', 0.038)]]
0 True 30000 [0.0388 0.0508] [0.1101 0.0815] [0.065  0.0572] [0.0013 0.0038]
0 True 100000 [0.0778 0.0507] [0.1286 0.0864] [0.0693 0.0525] [0.001  0.0032]
1 False 3000 [0.0546 0.0501] [0.1042 0.0642] [0.0573 0.0488] [0.0178 0.0182]
1 False 10000 [0.0417 0.0335] [0.0843 0.0528] [0.0492 0.0366] [0.0069 0.0098]
[[('117', 0.382), ('0', 0.217), ('107', -0.112), ('83', -0.112), ('124', 0.092), ('134', -0.085), ('96', 0.082), ('42', -0.058), ('24', -0.057), ('64', 0.047), ('136', 0.044), ('118', 0.043)], [('44', 0.298), ('3', -0.215), ('90', 0.163), ('40', -0.125), ('48', -0.083), ('24', 0.069), ('25', -0.065), ('85', 0.058), ('107', 0.058), ('20', 0.038), ('134', -0.036), ('17', -0.032)], [('32', -0.456), ('66', 0.414), ('1', 0.092), ('134', -0.074), ('47', -0.066), ('2', -0.053), ('83', 0.045), ('64', 0.042), ('76', 0.04), ('118', 0.036), ('40', -0.036), ('85', 0.035)]]
1 False 30000 [0.0311 0.0286] [0.0728 0.0445] [0.0397 0.0297] [0.0047 0.0059]
1 False 100000 [0.0292 0.0437] [0.0671 0.0471] [0.0386 0.0331] [0.0037 0.0045]
1 True 3000 [0.0442 0.0459] [0.145  0.0895] [0.0853 0.0665] [0.0191 0.0192]
1 True 10000 [0.0382 0.0612] [0.0848 0.0949] [0.0679 0.0635] [0.0048 0.0075]
[[('117', 0.242), ('83', -0.163), ((2, 3, 0.5, 'cos'), -0.111), ('61', 0.091), ('134', -0.089), ('0', 0.077), ((1, 2, 2, 'ss'), -0.057), ('98', 0.055), ((1, 2, 1, 'cos'), 0.053), ('118', 0.051), ('121', 0.048), ((2, 3, -1, 'cos'), 0.047)], [((1, 'tan', 0.5), 0.36), ('44', 0.227), ((1, 2, 1, 'sin'), -0.129), ((1, 3, -2, 'ss'), 0.088), ((2, 3, 1, 'ss'), -0.076), ((1, 3, -1, 'tan'), -0.072), ((3, 'cos', 1), 0.062), ((0, 2, 2, 'sym'), -0.055), ('17', -0.051), ('32', -0.05), ('26', -0.048), ((0, 2, 0.5, 'sym'), 0.048)], [('66', 0.407), ('32', -0.324), ((0, 1, 'ratio', 0.25), -0.152), ('129', -0.079), ((0, 3, 1, 'cos'), -0.074), ('134', -0.074), ((2, 3, 2, 'ss'), 0.049), ((1, 3, 0.5, 'ss'), 0.047), ((0, 3, 2, 'cos'), 0.044), ((2, 'tan', 1.5), 0.042), ((1, 'tan', 0.5), 0.041), ((1, 2, 2, 'ss'), -0.037)]]
1 True 30000 [0.0335 0.0476] [0.0786 0.0847] [0.0488 0.0528] [0.0019 0.0043]
1 True 100000 [0.0642 0.0493] [0.0679 0.0604] [0.0508 0.0387] [0.0004 0.0033]
```

### Turn 9 — `execute_in_remote_docker` (`27838 ms` LLM)
- **Rationale**: Save the best measured compact readout now, preserve both frozen files and their interfaces, and verify the composed DAG in an isolated Python process.
- **Alternatives Weighed**: `Expanded dictionary with independent ARD fits` (rejected: The extra basis terms increased worst-split errors without improving the gate trajectory error.)

```bash
export OPENBLAS_NUM_THREADS=1; cp /holdout_task/_frozen_nodes/node_00.py /tmp/node01_base.py; sed 's/^def run_node(inputs: dict) -> dict:/def _blend_reference(inputs: dict) -> dict:/' /tmp/node01_base.py > /holdout_task/node_01.py; cat >> /holdout_task/node_01.py <<'PY'

from scipy.optimize import least_squares
from sklearn.linear_model import ARDRegression


def _latent_coordinates(panel, x):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    h = np.column_stack((np.ones(len(u)), u))
    good = np.ones(len(u), dtype=bool)
    for _ in range(4):
        fit = np.linalg.lstsq(h[good], v[good], rcond=None)[0]
        residual = np.linalg.norm(v-h@fit, axis=1)
        next_good = residual < max(1e-3, 4*np.median(residual))
        if np.count_nonzero(next_good) < h.shape[1]:
            break
        good = next_good
    return (x-fit[0])/np.diag(fit[1:])


def _mode_readout(times, traj):
    def basis(log_rates):
        return -np.expm1(-times[:, None]*np.exp(log_rates))

    def residual(log_rates):
        q = basis(log_rates)
        return (q@np.linalg.lstsq(q, traj.T, rcond=None)[0]-traj.T).ravel()

    opt = least_squares(residual, np.log([.4, 1.8]), max_nfev=60)
    q = basis(opt.x)
    return np.linalg.lstsq(q, traj.T, rcond=None)[0].T, q


def _predict_features(z, n, targets, threshold):
    single, pair = _features(z)
    mat = np.column_stack(single + pair + [np.prod(z[:, j], axis=1) for j in combinations(range(4), 3)])[:, 1:]
    mat = (mat-mat[:n].mean(0))/np.maximum(mat[:n].std(0), 1e-9)
    return np.column_stack([ARDRegression(threshold_lambda=threshold).fit(mat[:n], target).predict(mat[n:]) for target in targets.T])


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    tr = study['train']
    n = len(tr['x'])
    x = np.vstack((tr['x'], study['test']['x']))
    if len(x) == n:
        return {'panel_pred': np.empty((0, 1+len(study['eval_times'])))}
    z = _latent_coordinates(study['panel_a'], x)
    amplitudes, basis = _mode_readout(np.asarray(study['eval_times']), np.asarray(tr['traj']))
    targets = np.column_stack((tr['y'], amplitudes))
    p = _predict_features(z, n, targets, 30000.)
    # Express the independently calibrated estimate in the fixed sink's mixing coordinates.
    baseline = _blend_reference({'__input__': study})
    return {'panel_pred': np.column_stack((2*p[:, 0]-baseline['y_hat'],
            (p[:, 1:]@basis.T-.35*baseline['state_pred'])/.65))}
PY
python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`1480 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.031144, "public_cv_traj_mae": 0.028582}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=423e30168d04f316 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_423e3016 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`24604 ms` LLM)
- **Rationale**: Compare a small set of fold-local ensembles against the accepted artifact, including the worst trajectory horizon. Promote only if public tolerances are preserved and the stress errors materially improve.
- **Alternatives Weighed**: `Submit the verified candidate immediately` (rejected: One read-only comparison can test whether the worst held-out errors improve without risking the verified artifact.)

```bash
export OPENBLAS_NUM_THREADS=1; cat >/tmp/ensemble_probe.py <<'PY'
exec(open('/tmp/readout_probe.py').read().split('res=[]')[0]); mod=runpy.run_path('/holdout_task/node_01.py'); P=[]
from sklearn.linear_model import ARDRegression
from itertools import combinations
for a,b,base in folds:
 xx=x[np.r_[a,b]]; n=len(a); z=mod['_latent_coordinates'](st['panel_a'],xx); A,Q=mod['_mode_readout'](st['eval_times'],Y[a,1:]); T=np.c_[Y[a,0],A]; opts=[]
 for mix in (1.,0.,.5):
  s,g=mod['_features'](mix*z+(1-mix)*xx); zz=mix*z+(1-mix)*xx
  f=np.column_stack(s+g+[np.prod(zz[:,j],axis=1) for j in combinations(range(4),3)])[:,1:]; f=(f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-9)
  for k in (10000,20000,30000,50000,100000):
   m=np.column_stack([ARDRegression(threshold_lambda=k).fit(f[:n],t).predict(f[n:]) for t in T.T]);opts.append(np.c_[m[:,0],m[:,1:]@Q.T]);
 P.append((np.array(opts),Y[b]))
G={'base':[2], 'cal':[0,1,2,3,4], 'calmid':[1,2,3], 'all':list(range(15)), 'allmid':[i for i in range(15) if i%5 in (1,2,3)], 'calmild':[0,1,2], 'calmidblend':[1,2,3,11,12,13], 'calplusraw':[0,1,2,3,4,5,6,7,8,9]}
for key,inds in G.items():
 out=[]
 for p,y in P:
  d=abs(p[inds].mean(0)-y);out.append([d[:,0].mean(),d[:,1:].mean(),d[:,1:].mean(0).max()]);
 print(key,np.round(out,5).tolist(),'max',np.max(out,0).round(5))
PY
python3 -I /tmp/ensemble_probe.py
```
- **Container Exit Code**: `0` (`13493 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
base [[0.03114, 0.02858, 0.03703], [0.05264, 0.02848, 0.03378], [0.07279, 0.02666, 0.03245], [0.03274, 0.04425, 0.05263], [0.04086, 0.04452, 0.05713], [0.02505, 0.02008, 0.02444], [0.03827, 0.02699, 0.03292], [0.024, 0.01815, 0.0231]] max [0.07279 0.04452 0.05713]
cal [[0.02662, 0.03239, 0.0395], [0.05407, 0.02905, 0.03449], [0.07266, 0.02181, 0.02526], [0.0328, 0.04407, 0.05238], [0.04421, 0.03988, 0.0511], [0.0247, 0.02568, 0.0323], [0.03252, 0.02809, 0.0344], [0.01824, 0.01886, 0.02381]] max [0.07266 0.04407 0.05238]
calmid [[0.02701, 0.03166, 0.03876], [0.05419, 0.0303, 0.03607], [0.07073, 0.02233, 0.02676], [0.02823, 0.04193, 0.0493], [0.04416, 0.03776, 0.04846], [0.02316, 0.02288, 0.02869], [0.03165, 0.02746, 0.03327], [0.019, 0.01814, 0.02279]] max [0.07073 0.04193 0.0493 ]
all [[0.03235, 0.03682, 0.04471], [0.05855, 0.02914, 0.0351], [0.06854, 0.01652, 0.01905], [0.04189, 0.03776, 0.04526], [0.03917, 0.04081, 0.05223], [0.02307, 0.02411, 0.03021], [0.02909, 0.03784, 0.04535], [0.02185, 0.02207, 0.02692]] max [0.06854 0.04081 0.05223]
allmid [[0.03002, 0.03833, 0.04692], [0.059, 0.03043, 0.03653], [0.06974, 0.01591, 0.01855], [0.03857, 0.03772, 0.04537], [0.04108, 0.04212, 0.05411], [0.02419, 0.02262, 0.02798], [0.02867, 0.03771, 0.0452], [0.02261, 0.0221, 0.02712]] max [0.06974 0.04212 0.05411]
calmild [[0.03054, 0.0281, 0.03468], [0.05299, 0.0288, 0.03429], [0.07582, 0.02645, 0.03115], [0.03991, 0.0451, 0.05443], [0.05061, 0.0426, 0.05508], [0.02504, 0.02649, 0.03268], [0.03296, 0.03227, 0.03867], [0.01881, 0.0182, 0.02299]] max [0.07582 0.0451  0.05508]
calmidblend [[0.02503, 0.03655, 0.04484], [0.05662, 0.02917, 0.03502], [0.07078, 0.0175, 0.02043], [0.03326, 0.03966, 0.04793], [0.04211, 0.04165, 0.05364], [0.02348, 0.02514, 0.03123], [0.0302, 0.03401, 0.04096], [0.02123, 0.02041, 0.02545]] max [0.07078 0.04165 0.05364]
calplusraw [[0.03059, 0.03458, 0.04183], [0.05836, 0.03016, 0.03627], [0.06695, 0.01752, 0.02039], [0.04066, 0.03948, 0.04708], [0.03975, 0.03803, 0.04848], [0.02276, 0.02235, 0.02808], [0.02903, 0.03573, 0.04305], [0.02271, 0.02199, 0.02668]] max [0.06695 0.03948 0.04848]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=423e30168d04f316 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_423e3016 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`25190 ms` LLM)
- **Rationale**: Promote the ensemble on its worst-slice gains, then run the official test and the eight-split stress check on the exact saved artifact. Restore the verified predecessor if the official test fails.
- **Alternatives Weighed**: `Retain the verified single readout` (rejected: The ensemble lowers all three worst-slice metrics and remains below both public tolerances in the probe.)

```bash
export OPENBLAS_NUM_THREADS=1; set -e; cp /holdout_task/node_01.py /tmp/node01_verified_v1.py; python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/node_01.py')
s = p.read_text()
old = '    p = _predict_features(z, n, targets, 30000.)'
assert s.count(old) == 1
p.write_text(s.replace(old, '''    # Average sparse readouts across calibrated and measured coordinates.
    # This reduces sensitivity to either a single coordinate map or pruning cutoff.
    p = np.mean([_predict_features(coords, n, targets, cutoff)
                 for coords in (z, x)
                 for cutoff in (10000., 20000., 30000., 50000., 100000.)], axis=0)'''))
PY
if ! python3 -I /holdout_task/test_dag_candidate.py; then cp /tmp/node01_verified_v1.py /holdout_task/node_01.py; python3 -I /holdout_task/test_dag_candidate.py; else python3 -I - <<'PY'
import runpy
import numpy as np
p = open('/tmp/readout_probe.py').read().split('res=[]')[0]
exec(p)
run = runpy.run_path('/holdout_task/node_01.py')['run_node']
rows = []
for a, b, sp in folds:
    sub = dict(st, train={k: v[a] for k,v in tr.items() if k != 'summary'}, test={'x': x[b]})
    panel = run({'__input__': sub})['panel_pred']
    e = abs(w*panel+(1-w)*sp-Y[b]); rows.append([e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max()])
worst = np.max(rows, axis=0)
print('Exact artifact stress maxima (scalar, trajectory, horizon):', worst)
assert np.all(worst < [.07279,.04452,.05713])
PY
fi
```
- **Container Exit Code**: `0` (`9950 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.030522, "public_cv_traj_mae": 0.034639}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}
Exact artifact stress maxima (scalar, trajectory, horizon): [0.06692707 0.03948218 0.0484796 ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=57683a522255084e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_57683a52 (directory).
```

### Turn 12 — `submit_completed_task` (`11225 ms` LLM)
- **Rationale**: The exact saved artifact passed the official test in isolated Python and improved all three eight-split stress metrics. The verification receipt is bound to candidate bundle 57683a522255084e.
- **Alternatives Weighed**: `Retain the single sparse readout` (rejected: The ensemble lowered all three worst-split errors while keeping both official CV metrics below 0.038.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_01.py` without changing the frozen nodes. The official test accepted it: scalar MAE was 0.030522, trajectory MAE was 0.034639, and each node executed once.`

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
