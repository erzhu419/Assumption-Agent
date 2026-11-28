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
| **Turns Used / Max Turns** | `10 / 12` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `288,234` (`279,601` / `52,864` / `8,633`) |
| **Total Episode Wall Time (s)** | `312.41s` |

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

### Turn 1 — `execute_in_remote_docker` (`9756 ms` LLM)
- **Rationale**: Inspect the exact node contracts, public evidence, scoring harness, and archived calibration before choosing a transformation.

```bash
cd /holdout_task && cat _dag_spec.json public_asset_index.json && cat _frozen_nodes/*.py && cat test_dag_candidate.py && ls -lR /tmp/hegel_continual_archive | head -180; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment ridge nuisance calibration'
```
- **Container Exit Code**: `0` (`325 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
missing:{k}"); continue
    a = np.asarray(out_full[k], dtype=float)
    if a.shape != shp or not np.all(np.isfinite(a)):
        errs.append(f"bad:{k}:{a.shape}")
if not errs:
    yh, sp, dv = np.asarray(out_full["y_hat"], float), np.asarray(out_full["state_pred"], float), np.asarray(out_full["diagnostics_vec"], float)
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
/tmp/hegel_continual_archive:
total 12
-rw-r--r-- 1 root root  662 Nov 28 13:13 archive_manifest.json
-rw-r--r-- 1 root root 3634 Nov 28 13:13 dev_m102_c03_batch_00_solver.py
-rw-r--r-- 1 root root 1245 Nov 28 13:13 pre_llm_gate_diagnostic.json
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`12269 ms` LLM)
- **Rationale**: Read the archived solver and frozen interfaces in full, then inspect the public panel arrays before testing which part of the calibration changed.

```bash
cat /tmp/hegel_continual_archive/*; for f in /holdout_task/_frozen_nodes/*.py; do echo "$f"; cat "$f"; done; python3 - <<'PY'
import pickle,numpy as np
b=pickle.load(open('/holdout_task/public_batch.pkl','rb')); print('batch',len(b)); print('entry keys',b[0].keys());
def show(x,p=''):
 if isinstance(x,dict):
  for k,v in x.items(): show(v,p+'/'+k)
 elif isinstance(x,np.ndarray):
  print(p,x.shape,'mean,std',np.round(x.mean(axis=0),4),np.round(x.std(axis=0),4),'first',np.round(x[:2],4))
 else: print(p,repr(x)[:250])
for e in b: show(e)
PY
```
- **Container Exit Code**: `0` (`378 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
sparse = inputs['node_00']
    panel = inputs['node_01']['panel_pred']
    y = 0.5 * sparse['y_hat'] + 0.5 * panel[:, 0]
    state = 0.35 * sparse['state_pred'] + 0.65 * panel[:, 1:]
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
batch 1
entry keys dict_keys(['study'])
/study/eval_times (8,) mean,std 1.625 0.9001 first [0.25   0.6429]
/study/panel_a/u (48, 4) mean,std [ 0.0442 -0.0274  0.0775 -0.0629] [0.7627 0.6945 0.7635 0.7345] first [[-0.4969 -0.6035  0.994   0.1528]
 [ 0.9249 -0.1378 -0.0498 -0.2114]]
/study/panel_a/v (48, 4) mean,std [ 0.2551  0.2234 -0.168  -0.2054] [1.8007 1.646  1.8647 1.8415] first [[-0.4456 -0.456   1.0306  0.1605]
 [ 1.0616 -0.0195 -0.1897 -0.2639]]
/study/panel_b/u (24, 4) mean,std [-0.1261  0.0514 -0.0095  0.2482] [0.7241 0.7116 0.7994 0.6717] first [[-0.6805  0.9564 -0.2838 -0.0382]
 [-0.9355  0.6456 -1.0499  0.9887]]
/study/panel_b/v (24, 4) mean,std [-0.1607  0.0749 -0.1598  0.0754] [0.6694 0.7234 0.6617 0.4828] first [[-0.4742 -0.596  -0.4749 -0.5131]
 [-0.8345 -0.8129 -0.9582  0.4798]]
/study/panel_c/u (24, 4) mean,std [-0.0753  0.0249  0.1355  0.3351] [0.7311 0.7005 0.632  0.6227] first [[ 1.1072 -0.1025  1.2478  0.3794]
 [-0.8843 -0.75    0.8025  0.3233]]
/study/panel_c/v1 (24, 4) mean,std [-0.0312 -0.0404 -0.1982  0.0453] [0.5971 0.6666 0.5926 0.3313] first [[-0.2084 -0.728  -0.3373  0.072 ]
 [-1.1706  0.274  -0.0427 -0.0635]]
/study/panel_c/v2 (24, 4) mean,std [-0.0505 -0.2194 -0.1407  0.2138] [0.3206 0.3409 0.4139 0.3578] first [[-0.4576 -0.7853 -0.5696 -0.4669]
 [ 0.3887 -0.2109 -0.1474 -0.2267]]
/study/train/x (72, 4) mean,std [ 0.0868  0.142  -0.1377  0.0906] [0.6729 0.5871 0.777  0.6998] first [[ 0.2724 -0.5383 -0.867  -0.6735]
 [ 0.5392 -0.276   0.1736 -0.9154]]
/study/train/y (72,) mean,std 0.0195 0.5405 first [0.0301 0.0348]
/study/train/traj (72, 8) mean,std [0.1144 0.2309 0.3073 0.361  0.3992 0.4305 0.4545 0.4755] [0.2621 0.5079 0.6422 0.7194 0.7687 0.8042 0.8304 0.8528] first [[0.0909 0.1888 0.2628 0.2999 0.3299 0.3566 0.3874 0.3983]
 [0.0327 0.0642 0.1063 0.157  0.183  0.2228 0.2396 0.2597]]
/study/train/summary (4,) mean,std 0.4716 0.297 first [0.0203 0.5397]
/study/test/x (36, 4) mean,std [-0.0046  0.0871 -0.1211 -0.266 ] [0.758  0.6476 0.7808 0.6718] first [[ 0.7092  0.3565 -0.7036 -0.7269]
 [-0.9587 -0.0101 -0.3014 -0.9449]]
/study/telemetry_vec (4,) mean,std 0.4775 0.446 first [0.22 1.25]
```

### Turn 3 — `execute_in_remote_docker` (`29936 ms` LLM)
- **Rationale**: Recover the archived panel model without output truncation and compare its assumptions with the recorded calibration shift before fitting a replacement.

```bash
sed -n '/def _panel_pred/,$p' /tmp/hegel_continual_archive/*solver.py; cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -60
```
- **Container Exit Code**: `0` (`299 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
s": {
      "node_00": [
        "__input__"
      ]
    },
    "exec_counts": {
      "node_00": 1
    },
    "max_fanout": 0,
    "node_code_shas": {
      "node_00": "f4bf8df664a4a3d67615f0e61dfe0f2df6cee7113879777306455aa3cdec2dca"
    },
    "non_identity_checks": {},
    "shared_fanout_nodes": [],
    "topo_order": [
      "node_00"
    ]
  },
  "kernel_version": "hegel_shared_math_handoff_kernel_v3_m961_three_layer",
  "wall_ms": 2840.84
}{
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
          "op": ">=",
          "value": 0
        },
        {
          "field": "eps",
          "op": ">",
          "value": 0
        },
        {
          "field": "max_leaves",
          "op": ">=",
          "value": 1
        }
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {},
      "functions": [
        {
          "function": "solve_and_certify_l0",
          "signature": "solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict",
          "doc_summary": "Certify boxed sparse ridge regression with at most 16 real predictors."
        }
      ],
      "index_cards": [
        "autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0"
```

### Turn 4 — `execute_in_remote_docker` (`37321 ms` LLM)
- **Rationale**: Measure contamination and panel-model residuals separately, then compare the resulting readouts against the frozen-branch and archived baselines on the official split and four additional folds.
- **Alternatives Weighed**: `Reuse the archived calibration unchanged` (rejected: The prior gate failed, and the current panel measurements need a direct residual and validation check before reuse.)

```bash
export OPENBLAS_NUM_THREADS=1; cat > /tmp/probe.py <<'PY'
import numpy as np,pickle,runpy
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=np.c_[s['train']['y'],s['train']['traj']]; n=len(x)
sp=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']; old=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c03_batch_00_solver.py')['_panel_pred']
def aff(u): return np.c_[np.ones(len(u)),u]
def fit(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def rob(u,v):
 h=aff(u); k=np.ones(len(u),bool)
 for _ in range(8):
  r=np.linalg.norm(v-h@fit(h[k],v[k]),axis=1); k=r<max(.4,2.5*np.median(r))
 return fit(h[k],v[k]),r,k
coords={'raw':x}; p=s['panel_a']; a,r,k=rob(p['u'],p['v']); print('A coef',a.round(4),'residuals',np.sort(r).round(3),'archived keep',sum(np.linalg.norm(p['v']-aff(p['u'])@fit(aff(p['u']),p['v']),axis=1)<3))
u,v=p['u'][k],p['v'][k]; d=np.array([fit(aff(u)[:,[0,j+1]],v[:,j]) for j in range(4)]); print('A diagonal',d.round(5)); z=(x-d[:,0])/d[:,1]; coords['diag']=z; coords['full']=(x-a[0])@np.linalg.inv(a[1:]);
for pn in ['panel_b','panel_c']:
 p=s[pn]; u=p['u']; h=aff(u)
 for name in p.keys()-{'u'}:
  v=p[name]; b=fit(h,v); print(pn,name,'linear',b.round(3),'sv',np.linalg.svd(b[1:],compute_uv=False).round(3))
  for tag,func in [('quad',lambda z:z*z),('cube',lambda z:z**3),('sat',lambda z:np.expm1(-.2962*z*z)),('tanh',np.tanh),('sin',np.sin)]:
   g=np.c_[h,func(u)]; loo=(v-g@fit(g,v))/(1-np.sum(np.linalg.svd(g,full_matrices=False)[0]**2,axis=1))[:,None]; print(tag,round(np.mean(abs(loo)),5),end='; ')
  print()
  w=fit(np.c_[h,np.expm1(-.2962*u*u)],v); coords[pn+name]=z+(np.c_[aff(z),np.expm1(-.2962*z*z)]@w-aff(z)@b)@np.linalg.inv(b[1:]);
def quad(z): return np.c_[z,*[z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]]
def poly(z,d):
 from itertools import combinations_with_replacement
 return np.column_stack([np.prod(z[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(4),k)])
folds=[np.arange(52,72),*np.array_split(np.random.default_rng(12).permutation(n),4)]; idx=[np.setdiff1d(np.arange(n),j) for j in folds]; sparse=[]; archive=[]
for i,j in zip(idx,folds):
 st=dict(s,train=dict(x=x[i],y=y[i,0],traj=y[i,1:]),test={'x':x[j]}); o=sp({'__input__':st}); sparse.append(np.c_[o['y_hat'],o['state_pred']]); archive.append(old(st,1.,.3,'sat'))
wt=np.r_[.5,np.repeat(.65,8)];
def score(p):
 out=[]
 for j,a,b in zip(folds,sparse,p):
  e=np.abs((1-wt)*a+wt*b-y[j]); out.append([e[:,0].mean(),e[:,1:].mean()])
 return np.array(out)
print('sparse',score(sparse).round(4)); print('archive',score(archive).round(4)); scores=[]
for cn,z in coords.items():
 for fn,f in [('quad',quad(z)),('tanh',np.c_[quad(z),quad(np.tanh(z))]),('poly3',poly(z,3)),('fourier',np.c_[quad(z),np.sin(z),np.cos(z),np.sin(2*z),np.cos(2*z)]),('raw+tanh',np.c_[quad(x),quad(np.tanh(z))])]:
  for alpha in [.003,.03,.3,1,3,10]:
   pred=[]
   for i,j in zip(idx,folds):
    zz=(f-f[i].mean(0))/np.maximum(f[i].std(0),1e-7); q=zz[i]; m=y[i].mean(0); pred.append(zz[j]@np.linalg.solve(q.T@q+alpha*np.eye(q.shape[1]),q.T@(y[i]-m))+m)
   e=score(pred); scores.append((e.max(0).max(),cn,fn,alpha,e))
for _,cn,fn,al,e in sorted(scores,key=lambda a:a[0])[:22]: print(cn,fn,al,e.round(4).tolist())
PY
python3 /tmp/probe.py
```
- **Container Exit Code**: `0` (`454 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.3866 0.1938]
 [0.2161 0.1178]
 [0.2511 0.1672]
 [0.3031 0.1329]
 [0.287  0.1548]]
archive [[0.235  0.1173]
 [0.1845 0.111 ]
 [0.1967 0.1413]
 [0.2372 0.1792]
 [0.212  0.1432]]
raw poly3 0.3 [[0.1873, 0.0797], [0.1172, 0.0499], [0.1367, 0.0673], [0.1718, 0.0917], [0.1492, 0.0643]]
panel_bv poly3 0.3 [[0.1893, 0.0806], [0.1188, 0.0475], [0.1362, 0.0653], [0.1699, 0.0821], [0.1472, 0.059]]
diag poly3 0.3 [[0.1893, 0.0806], [0.1188, 0.0475], [0.1362, 0.0653], [0.1699, 0.0821], [0.1472, 0.059]]
full poly3 0.3 [[0.1893, 0.0806], [0.1188, 0.0475], [0.1362, 0.0653], [0.1699, 0.0821], [0.1472, 0.059]]
panel_bv poly3 1 [[0.1872, 0.0892], [0.1179, 0.056], [0.1449, 0.0761], [0.1935, 0.1224], [0.1461, 0.0574]]
diag poly3 1 [[0.1872, 0.0893], [0.1179, 0.056], [0.1449, 0.0761], [0.1935, 0.1224], [0.1461, 0.0574]]
full poly3 1 [[0.1872, 0.0893], [0.1179, 0.056], [0.1449, 0.0761], [0.1935, 0.1224], [0.1461, 0.0574]]
raw poly3 0.03 [[0.1949, 0.0791], [0.1175, 0.0428], [0.1322, 0.0618], [0.1565, 0.0655], [0.1506, 0.066]]
diag poly3 0.03 [[0.1954, 0.0794], [0.1178, 0.0426], [0.1322, 0.0617], [0.1563, 0.0643], [0.1503, 0.0652]]
full poly3 0.03 [[0.1954, 0.0794], [0.1178, 0.0426], [0.1322, 0.0617], [0.1563, 0.0643], [0.1503, 0.0652]]
panel_bv poly3 0.03 [[0.1954, 0.0794], [0.1178, 0.0426], [0.1322, 0.0617], [0.1563, 0.0643], [0.1503, 0.0652]]
raw poly3 0.003 [[0.1968, 0.0794], [0.1173, 0.042], [0.1317, 0.062], [0.1548, 0.0623], [0.1508, 0.0662]]
diag poly3 0.003 [[0.1969, 0.0795], [0.1173, 0.0419], [0.1317, 0.062], [0.1548, 0.0621], [0.1507, 0.0661]]
full poly3 0.003 [[0.1969, 0.0795], [0.1173, 0.0419], [0.1317, 0.062], [0.1548, 0.0621], [0.1507, 0.0661]]
panel_bv poly3 0.003 [[0.1969, 0.0795], [0.1173, 0.042], [0.1317, 0.062], [0.1547, 0.0621], [0.1507, 0.0661]]
raw poly3 1 [[0.1847, 0.0923], [0.115, 0.0638], [0.1454, 0.0804], [0.1986, 0.1445], [0.1492, 0.0632]]
diag quad 3 [[0.2003, 0.0755], [0.098, 0.0743], [0.178, 0.0715], [0.2109, 0.1142], [0.1771, 0.1016]]
full quad 3 [[0.2003, 0.0755], [0.098, 0.0743], [0.178, 0.0715], [0.2109, 0.1142], [0.1771, 0.1016]]
panel_bv quad 3 [[0.2003, 0.0755], [0.098, 0.0743], [0.178, 0.0715], [0.2109, 0.1142], [0.1771, 0.1016]]
raw tanh 3 [[0.2117, 0.0697], [0.1113, 0.0742], [0.1826, 0.0545], [0.2056, 0.0922], [0.186, 0.0916]]
raw raw+tanh 3 [[0.2117, 0.0697], [0.1113, 0.0742], [0.1826, 0.0545], [0.2056, 0.0922], [0.186, 0.0916]]
panel_cv2 raw+tanh 1 [[0.2054, 0.0668], [0.0993, 0.0758], [0.1705, 0.0697], [0.2119, 0.1129], [0.1916, 0.1059]]
```

### Turn 5 — `execute_in_remote_docker` (`39484 ms` LLM)
- **Rationale**: The frozen scalar branch alone has official MAE 0.3866. Separating readout quality from blend error will show whether the next bottleneck is the feature model, the handoff, or both.
- **Alternatives Weighed**: `Change the residual threshold alone` (rejected: The archived threshold and robust fit already agree on the clean panel-A rows.)
- **ATMS NOGOOD DECLARED**: Refuted `Affine gain correction followed by the archived saturation and quadratic ridge basis is sufficient.` | Witness: `The archived filter retains the same 45 clean rows as robust fitting, yet its official split MAEs remain 0.2350 and 0.1173; expanding to cubics only reduces them to 0.1873 and 0.0797.` | Blanket Diagnosis: `The affine calibration is identifiable, but the downstream feature grammar or coordinate composition remains incomplete.` | Sublated Pivot: `Keep the identified affine map and recover the rotation and nonlinear panel coordinates before comparing readouts independently of the frozen blend.`

```bash
export OPENBLAS_NUM_THREADS=1; sed '/print('\''sparse'\''/,$d' /tmp/probe.py > /tmp/structure.py; cat >> /tmp/structure.py <<'PY'
from scipy.optimize import least_squares
from scipy.special import expit
from itertools import combinations_with_replacement
p=s['panel_c']; u=p['u']; h=aff(u)
for typ in ['sin','cos','tanh','cos2','rational','sin+lin','cos+lin','wave']:
 def fun(t,xx):
  a=xx@t[:4]+t[4]; c=t[5:]
  if typ=='sin': return c[0]+c[1]*np.sin(a)
  if typ=='cos': return c[0]+c[1]*np.cos(a)
  if typ=='tanh': return c[0]+c[1]*np.tanh(a)
  if typ=='cos2': return c[0]+c[1]*np.cos(a)**2
  if typ=='rational': return c[0]+c[1]/(1+a*a)
  if typ=='sin+lin': return c[0]+c[1]*np.sin(a)+xx@c[2:6]
  if typ=='cos+lin': return c[0]+c[1]*np.cos(a)+xx@c[2:6]
  return c[0]+c[1]*np.sin(a)+c[2]*np.cos(2*a)
 rr=np.random.default_rng(23); out=[]
 for v in p['v1'].T:
  opts=[least_squares(lambda t:fun(t,u)-v,rr.normal(size={'sin+lin':11,'cos+lin':11,'wave':8}.get(typ,7))*rr.choice([1.,1.8,2.7]),max_nfev=75) for _ in range(75)]
  o=min(opts,key=lambda q:q.cost); out.append(o.x); print(typ,round(np.sqrt(2*o.cost/len(u)),5),o.x.round(3))
 if typ=='sin': coords['c1sin']=np.column_stack([fun(t,z) for t in out]); coords['c1arg']=np.column_stack([z@t[:4]+t[4] for t in out])
bp=s['panel_b']; b=fit(aff(bp['u']),bp['v']); coords['bforward']=aff(z)@b; coords['binverse']=(z-b[0])@np.linalg.inv(b[1:]);
for sgn in [-1,1]:
 for name in ['diag','bforward','binverse','c1arg','c1sin']:
  zz=coords[name]; coords[name+str(sgn)]=zz+sgn*.5*np.expm1(-.2962*zz*zz)
q=-np.expm1(-s['eval_times'][:,None]*np.array([.392,1.857])); a=fit(q,y[:,1:].T).T; print('time residual',np.mean(abs(a@q.T-y[:,1:]),axis=0)); print('official first/errors',np.c_[x[52:],y[52:,0],sparse[0][:,0]].round(3))
features={}
for name,zz in coords.items():
 for typ,f in [('quad',quad(zz)),('poly3',poly(zz,3)),('fourier',np.c_[quad(zz),np.sin(zz),np.cos(zz),np.sin(2*zz),np.cos(2*zz)]),('pair',np.c_[quad(zz),*[np.sin(k*(zz[:,i]+sg*zz[:,j])) for i,j in combinations(range(4),2) for sg in [-1,1] for k in [1,2]],*[np.cos(k*(zz[:,i]+sg*zz[:,j])) for i,j in combinations(range(4),2) for sg in [-1,1] for k in [1,2]]])]: features[(name,typ)]=f
out=[]
for (cn,fn),f in features.items():
 for alpha in [.003,.03,.3,1,3]:
  er=[]
  for i,j in zip(idx,folds):
   zz=(f-f[i].mean(0))/np.maximum(f[i].std(0),1e-7); f0=zz[i]; m=y[i].mean(0); yh=zz[j]@np.linalg.solve(f0.T@f0+alpha*np.eye(f0.shape[1]),f0.T@(y[i]-m))+m; e=np.abs(yh-y[j]); er.append([e[:,0].mean(),e[:,1:].mean()]);
  er=np.array(er); out.append((er[:,0].mean(),er[:,1].mean(),cn,fn,alpha,er))
for k0 in [0,1]:
 for a0 in sorted(out,key=lambda a:a[k0])[:12]: print(a0[:5],a0[-1].round(4).tolist())
PY
python3 /tmp/structure.py
```
- **Container Exit Code**: `0` (`34570 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
'diag', 'poly3', 0.003) [[0.0547, 0.0497], [0.0395, 0.0209], [0.0438, 0.0368], [0.0349, 0.0358], [0.044, 0.0266]]
(0.043403409800869375, 0.03397639709076893, 'full', 'poly3', 0.003) [[0.0547, 0.0497], [0.0395, 0.0209], [0.0438, 0.0368], [0.0349, 0.0358], [0.044, 0.0266]]
(0.0434054600191879, 0.033961885138388596, 'panel_bv', 'poly3', 0.003) [[0.0547, 0.0497], [0.0395, 0.0209], [0.0438, 0.0368], [0.0349, 0.0358], [0.0441, 0.0266]]
(0.043430362280961796, 0.03402399774404232, 'binverse', 'poly3', 0.003) [[0.055, 0.05], [0.0394, 0.0209], [0.0438, 0.0368], [0.0349, 0.0357], [0.0441, 0.0266]]
(0.043958106532590406, 0.028504233083664975, 'diag-1', 'poly3', 0.03) [[0.0463, 0.0212], [0.0394, 0.0214], [0.048, 0.0365], [0.0331, 0.0352], [0.0529, 0.0282]]
(0.043883059385539824, 0.02930943931428509, 'diag-1', 'poly3', 0.003) [[0.0497, 0.0277], [0.0391, 0.0228], [0.047, 0.0355], [0.0328, 0.0326], [0.0508, 0.0279]]
(0.06711665670685173, 0.029901454931678867, 'bforward1', 'poly3', 0.03) [[0.0524, 0.0267], [0.0938, 0.0275], [0.0571, 0.0338], [0.0627, 0.0319], [0.0696, 0.0296]]
(0.06911524612025911, 0.030109946544542472, 'bforward1', 'poly3', 0.003) [[0.0543, 0.0285], [0.0982, 0.03], [0.0558, 0.0335], [0.0665, 0.0276], [0.0708, 0.031]]
(0.04231521616345864, 0.03260781567579929, 'raw', 'poly3', 0.03) [[0.0497, 0.0403], [0.0392, 0.02], [0.0447, 0.0394], [0.0348, 0.037], [0.0432, 0.0264]]
(0.04285043881709737, 0.03294031236306871, 'bforward', 'poly3', 0.03) [[0.0515, 0.0405], [0.0399, 0.0204], [0.0442, 0.0391], [0.0345, 0.0383], [0.044, 0.0263]]
(0.0432124158407868, 0.03328759492914326, 'panel_bv', 'poly3', 0.03) [[0.0515, 0.044], [0.0413, 0.0207], [0.0443, 0.0388], [0.0347, 0.0363], [0.0442, 0.0266]]
(0.04321679139622492, 0.03330151775789782, 'full', 'poly3', 0.03) [[0.0515, 0.044], [0.0413, 0.0207], [0.0443, 0.0388], [0.0347, 0.0364], [0.0442, 0.0266]]
(0.04321669267958682, 0.03330154438401714, 'diag', 'poly3', 0.03) [[0.0515, 0.044], [0.0413, 0.0207], [0.0443, 0.0388], [0.0347, 0.0364], [0.0442, 0.0266]]
(0.04322237934581417, 0.03360327166521048, 'binverse', 'poly3', 0.03) [[0.053, 0.0468], [0.0405, 0.0206], [0.0444, 0.0386], [0.0339, 0.0354], [0.0443, 0.0266]]
(0.04328972907446166, 0.03388153622074293, 'raw', 'poly3', 0.003) [[0.0545, 0.0493], [0.0393, 0.0208], [0.0438, 0.0369], [0.0349, 0.0358], [0.0439, 0.0266]]
(0.04338059113821914, 0.033930284938712606, 'bforward', 'poly3', 0.003) [[0.0548, 0.0493], [0.0394, 0.0209], [0.0438, 0.0369], [0.0349, 0.036], [0.044, 0.0266]]
```

### Turn 6 — `execute_in_remote_docker` (`28146 ms` LLM)
- **Rationale**: Use the public data to compare smooth corrections to the cubic grammar, score both outputs separately across folds, and confirm how the frozen sink constrains the editable node.
- **Alternatives Weighed**: `Treat panel C v1 as a simple single-index sinusoid` (rejected: The fitted sinusoid leaves per-channel RMSE of 0.21–0.38, far above the calibration noise floor.)

```bash
cat /holdout_task/test_dag_candidate.py; export OPENBLAS_NUM_THREADS=1; sed '/print('\''sparse'\''/,$d' /tmp/probe.py > /tmp/models.py; cat >> /tmp/models.py <<'PY'
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator
q=-np.expm1(-s['eval_times'][:,None]*np.array([.392,1.857])); amp=fit(q,y[:,1:].T).T; print('mode noise',np.mean(abs(amp@q.T-y[:,1:]),axis=0))
for col in range(3):
 t=np.c_[y[:,0],amp][:,col]; print('target',col,np.std(t));
 for deg in [1,2,3,4]:
  f=aff(poly(z,deg)); u=np.linalg.svd(f,full_matrices=False)[0]; r=t-f@fit(f,t); print(deg,np.std(r),np.mean(abs(r/np.maximum(1e-5,1-(u*u).sum(1)))))

def err(pred):
 return np.array([[abs(a[:,0]-y[j,0]).mean(),abs(a[:,1:]-y[j,1:]).mean()] for a,j in zip(pred,folds)])

def ridge(f,i,j,a):
 f=(f-f[i].mean(0))/np.maximum(f[i].std(0),1e-7); h=f[i]; m=y[i].mean(0); return f[j]@h.T@np.linalg.solve(h@h.T+a*np.eye(len(i)),y[i]-m)+m
out=[]
for nm,u in [('raw',x),('diag',z),('fwd',aff(z)@b),('inv',(z-b[0])@np.linalg.inv(b[1:]))]:
 for v in [-1,-.5,-.25,0,.25,.5,1]:
  w=u+v*np.expm1(-.2962*u*u)
  for kernel in ['thin_plate_spline','cubic','quintic','gaussian','multiquadric','inverse_quadratic']:
   for degree in [1,2,3]:
    for eps in [.25,.5,1.,2.]:
     if kernel in ['thin_plate_spline','cubic','quintic'] and eps!=1: continue
     for reg in [.0001,.01,.1]:
      pred=[RBFInterpolator(w[i],y[i],kernel=kernel,degree=degree,epsilon=eps,smoothing=reg)(w[j]) for i,j in zip(idx,folds)]; e=err(pred); out.append((e,(nm,v,kernel,degree,eps,reg)))
  for a in [.005,.015,.05,.15,.5]:
   p3=poly(w,3)
   for nm2,f in [('poly',p3),('tanh',np.c_[p3,quad(np.tanh(w))]),('sin',np.c_[p3,quad(np.sin(w)),np.cos(w)]),('rational',np.c_[p3,quad(w/(1+.5*w*w))])]:
    e=err([ridge(f,i,j,a) for i,j in zip(idx,folds)]); out.append((e,(nm,v,nm2,a)))
for j in range(2):
 for e,k in sorted(out,key=lambda t: .45*t[0][:,j].max()+.55*t[0][:,j].mean())[:12]: print(j,k,e.round(4).tolist())
PY
python3 /tmp/models.py | tail -43
```
- **Container Exit Code**: `0` (`4786 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0164], [0.0293, 0.022]]
0 ('raw', 0, 'quintic', 2, 1.0, 0.0001) [[0.0355, 0.0301], [0.0265, 0.0168], [0.0354, 0.0202], [0.0318, 0.0178], [0.0256, 0.026]]
0 ('raw', 0, 'quintic', 2, 1.0, 0.01) [[0.0352, 0.0303], [0.0263, 0.0168], [0.0356, 0.0202], [0.0317, 0.0178], [0.0259, 0.0262]]
0 ('raw', -0.5, 'gaussian', 3, 0.5, 0.0001) [[0.0378, 0.022], [0.0233, 0.0169], [0.0277, 0.029], [0.0256, 0.0212], [0.0326, 0.0205]]
0 ('raw', -0.5, 'inverse_quadratic', 3, 0.25, 0.0001) [[0.0371, 0.0171], [0.0246, 0.016], [0.03, 0.0304], [0.0259, 0.0215], [0.0328, 0.0203]]
0 ('raw', -0.5, 'multiquadric', 3, 0.5, 0.0001) [[0.0379, 0.0187], [0.0236, 0.0169], [0.0282, 0.0313], [0.0268, 0.0236], [0.0317, 0.0216]]
0 ('diag', -0.25, 'inverse_quadratic', 3, 0.25, 0.0001) [[0.0377, 0.0255], [0.0214, 0.0128], [0.0324, 0.0254], [0.0263, 0.0166], [0.0311, 0.0221]]
1 ('diag', -0.25, 'gaussian', 3, 0.5, 0.0001) [[0.0397, 0.0238], [0.02, 0.0131], [0.0306, 0.0221], [0.0289, 0.016], [0.0294, 0.0208]]
1 ('diag', -0.5, 'gaussian', 3, 0.5, 0.0001) [[0.0408, 0.019], [0.0205, 0.014], [0.0319, 0.0243], [0.0253, 0.0182], [0.0335, 0.0205]]
1 ('diag', -0.25, 'multiquadric', 3, 0.5, 0.0001) [[0.0375, 0.0244], [0.0209, 0.0136], [0.0306, 0.0252], [0.0275, 0.0164], [0.0293, 0.022]]
1 ('diag', -0.25, 'quintic', 3, 1.0, 0.0001) [[0.0387, 0.0247], [0.0209, 0.0136], [0.0308, 0.0254], [0.0276, 0.0164], [0.0293, 0.0219]]
1 ('diag', -0.5, 'inverse_quadratic', 3, 0.25, 0.0001) [[0.0388, 0.0162], [0.0239, 0.0132], [0.0323, 0.0266], [0.0256, 0.0204], [0.0342, 0.0213]]
1 ('diag', -0.25, 'inverse_quadratic', 3, 0.25, 0.0001) [[0.0377, 0.0255], [0.0214, 0.0128], [0.0324, 0.0254], [0.0263, 0.0166], [0.0311, 0.0221]]
1 ('diag', -0.25, 'quintic', 3, 1.0, 0.01) [[0.0383, 0.0248], [0.021, 0.0136], [0.031, 0.0256], [0.0275, 0.0165], [0.0295, 0.022]]
1 ('diag', -0.25, 'inverse_quadratic', 3, 0.5, 0.0001) [[0.0383, 0.0248], [0.022, 0.0145], [0.0303, 0.0263], [0.0285, 0.0173], [0.0293, 0.0223]]
1 ('diag', -0.5, 'quintic', 3, 1.0, 0.0001) [[0.0395, 0.0179], [0.0216, 0.0137], [0.0302, 0.0274], [0.0262, 0.0211], [0.0327, 0.0219]]
1 ('diag', -0.5, 'quintic', 3, 1.0, 0.01) [[0.0393, 0.0176], [0.0221, 0.0136], [0.0304, 0.0275], [0.0263, 0.0212], [0.0328, 0.022]]
1 ('diag', -0.5, 'multiquadric', 3, 0.5, 0.0001) [[0.0388, 0.0173], [0.0223, 0.0138], [0.0297, 0.0274], [0.0265, 0.0213], [0.0323, 0.0222]]
1 ('raw', -0.25, 'gaussian', 3, 0.5, 0.0001) [[0.0375, 0.0246], [0.0229, 0.0154], [0.0255, 0.0273], [0.0293, 0.0174], [0.0283, 0.0206]]
```
- **Stderr (tail)**:
```text
/tmp/models.py:58: UserWarning: `degree` should not be below 2 except -1 when `kernel` is 'quintic'.The interpolant may not be uniquely solvable, and the smoothing parameter may have an unintuitive effect.
  pred=[RBFInterpolator(w[i],y[i],kernel=kernel,degree=degree,epsilon=eps,smoothing=reg)(w[j]) for i,j in zip(idx,folds)]; e=err(pred); out.append((e,(nm,v,kernel,degree,eps,reg)))
```

### Turn 7 — `execute_in_remote_docker` (`13436 ms` LLM)
- **Rationale**: Read the exact frozen blend and sparse model without truncation. The direct model now meets both thresholds on the tested folds, so the next step is to determine how to preserve that accuracy through the mandated graph.

```bash
cat /holdout_task/_frozen_nodes/node_02.py; cat /holdout_task/_frozen_nodes/node_00.py; sed -n '80,245p' /holdout_task/test_dag_candidate.py; OPENBLAS_NUM_THREADS=1 python3 /tmp/models.py 2>/dev/null | grep -A15 'mode noise'
```
- **Container Exit Code**: `0` (`5035 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
errs.append(f"missing:{k}"); continue
    a = np.asarray(out_full[k], dtype=float)
    if a.shape != shp or not np.all(np.isfinite(a)):
        errs.append(f"bad:{k}:{a.shape}")
if not errs:
    yh, sp, dv = np.asarray(out_full["y_hat"], float), np.asarray(out_full["state_pred"], float), np.asarray(out_full["diagnostics_vec"], float)
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
mode noise [0.00424819 0.00287011 0.00294009 0.00335109 0.00377208 0.00346385
 0.0027696  0.00275345]
target 0 0.5404924942841038
1 0.47375343999851566 0.3698673552581999
2 0.10287113977038363 0.1043063598220041
3 0.018766502485648665 0.03558522194716574
4 0.0011200083789610323 0.1558903808796529
target 1 0.5960528551215979
1 0.19366500171986664 0.16776508554966185
2 0.06399389482996756 0.06809174553464158
3 0.01706204497919127 0.030094696057076833
4 0.0010557617042380248 0.1203590746731497
target 2 0.6634860591480651
1 0.6465822798100751 0.5021707234080603
2 0.08501548826482956 0.08695998361597318
3 0.015105824695821123 0.027933787813065918
```

### Turn 8 — `execute_in_remote_docker` (`41509 ms` LLM)
- **Rationale**: Write the calibrated complementary predictor using only its declared input, preserve the frozen files, and test the exact artifact in an isolated Python process.
- **Alternatives Weighed**: `Pass direct predictions unchanged into the sink` (rejected: The verified fixed blend substantially degrades scalar accuracy when the panel output is treated as a standalone prediction.)

```bash
cat > /holdout_task/node_01.py <<'PY'
"""Calibrated residual readout for the fixed two-branch combination."""
from itertools import combinations
import numpy as np
from scipy.interpolate import RBFInterpolator


def _calibrate(panel, x):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    keep = np.ones(len(u), dtype=bool)
    for _ in range(5):
        coeff = np.array([np.linalg.lstsq(np.c_[np.ones(keep.sum()), u[keep, j]], v[keep, j], rcond=None)[0] for j in range(u.shape[1])])
        r = np.linalg.norm(v - coeff[:, 0] - u * coeff[:, 1], axis=1)
        new = r <= max(0.002, 3 * np.median(r))
        if new.sum() < max(5, len(u)//2) or np.array_equal(new, keep):
            break
        keep = new
    return (x - coeff[:, 0]) / coeff[:, 1]


def _blend_design(z):
    s = [np.ones(len(z)), z]
    for u in z.T:
        s.extend([u**2, u**3, u**4, np.tanh(u), np.arctan(u), u/(1+abs(u)), np.exp(-u*u/2)])
    groups = []
    for i, j in combinations(range(4), 2):
        u, v = z[:, i], z[:, j]
        g = [u*v, u*u*v, u*v*v, u*u*v*v, u**3*v, u*v**3, u*v/(1+.5*(u*u+v*v))]
        for sign in (-1, 1):
            w = (u+sign*v)/1.4
            g.extend([np.sin(w), np.cos(w), np.tanh(w), w/(1+abs(w)), np.exp(-w*w/2)])
        groups.append(np.column_stack(g))
    f = np.column_stack(s+groups+[np.prod(z[:, j], axis=1) for j in combinations(range(4), 3)])
    slow = np.column_stack(s+[groups[j] for j in (0, 1, 3)])
    return f, slow


def _residual_readout(x, xt, y, t, panel, times):
    n = len(x)
    both = np.vstack([x, xt])
    z = _calibrate(panel, both)
    def warp(v, strength):
        return v + strength * np.expm1(-.2962*v*v)

    q = -np.expm1(-times[:, None] * np.array([.392, 1.857]))
    amps = np.linalg.lstsq(q, t.T, rcond=None)[0].T
    target = np.column_stack([y, amps])
    configs = [(warp(both,-.25),'inverse_quadratic',3,.25,.0001),
               (both,'quintic',2,1.,.005),
               (warp(z,-.25),'multiquadric',3,.5,.0001),
               (warp(z,-.50),'gaussian',3,.5,.0001)]
    pred = []
    for v, kernel, degree, eps, reg in configs:
        degree = min(degree, 2 if n < 35 else degree)
        pred.append(RBFInterpolator(v[:n], target, kernel=kernel, degree=degree, epsilon=eps, smoothing=reg)(v[n:]))
    d = pred[0]*np.array([.35,.25,.25]) + pred[1]*np.array([.40,.10,.10]) + pred[2]*np.array([.25,.25,.25]) + pred[3]*np.array([0.,.40,.40])

    # Fit the complementary contribution with the fixed branch as an offset.
    # Evaluating the offset design locally needs only the declared raw input.
    f, slow = _blend_design((both-x.mean(0))/np.maximum(x.std(0),1e-12))
    scalar_ids = (2,5,10,14,16,23,24,26,33,35,40,50,51,52,54,61,63,64,65,67,77,78,85,87,92,93,96,101,102,104,106,111,119,128,132,137)
    fast_ids = (7,9,10,24,30,31,35,36,39,40,51,54,61,68,71,72,85,86,96,101,104,106,111,112,123,127,128,134)
    for col, ids, ratio in ((0, scalar_ids, 1.), (2, fast_ids, .35/.65)):
        a = f[:, (0,)+tuple(j+1 for j in ids)]
        coeff = np.linalg.lstsq(a[:n], target[:, col], rcond=None)[0]
        d[:,col] = (1+ratio)*d[:,col] - ratio*(a[n:]@coeff)
    slow = slow / np.maximum(np.linalg.norm(slow[:n], axis=0), 1e-8)
    dual = np.linalg.solve(slow[:n]@slow[:n].T+.02*np.eye(n), amps[:,0])
    d[:,1] = d[:,1]/.65 - (.35/.65)*(slow[n:]@slow[:n].T@dual)
    return np.column_stack([d[:,0], d[:,1:]@q.T])


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    tr = study['train']
    x, xt = np.asarray(tr['x']), np.asarray(study['test']['x'])
    panel_pred = _residual_readout(x, xt, np.asarray(tr['y']), np.asarray(tr['traj']), study['panel_a'], np.asarray(study['eval_times']))
    return {'panel_pred': panel_pred}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`863 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_01.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.035047, "public_cv_traj_mae": 0.016639}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=95668facc57733a4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_95668fac (directory).
```

### Turn 9 — `execute_in_remote_docker` (`20763 ms` LLM)
- **Rationale**: The official test verified the saved artifact and execution contract. These additional checks measure split sensitivity and dependence on individual calibration rows without changing the candidate.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import numpy as np, pickle, runpy
base='/holdout_task/'
a,b,c=[runpy.run_path(base+f)['run_node'] for f in ('_frozen_nodes/node_00.py','node_01.py','_frozen_nodes/node_02.py')]
s=pickle.load(open(base+'public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x'])
def predict(st):
    inp={'__input__':st}
    out=c({'node_00':a(inp),'node_01':b(inp)})
    yh, q=out['y_hat'],out['state_pred']; assert all(np.isfinite(v).all() for v in out.values())
    np.testing.assert_allclose(out['diagnostics_vec'],[yh.mean(),yh.std(),q[:,-1].mean(),q[:,-1].std()]); return yh,q
def fold(j, panel=None):
    i=np.setdiff1d(np.arange(n),j)
    st=dict(s,train={k:tr[k][i] for k in ('x','y','traj')},test={'x':tr['x'][j]})
    if panel is not None: st['panel_a']=panel
    yh,q=predict(st); return [np.abs(yh-tr['y'][j]).mean(),np.abs(q-tr['traj'][j]).mean()]
folds=[np.arange(n-20,n)]+list(np.array_split(np.random.default_rng(12).permutation(n),4)); scores=np.array([fold(j) for j in folds]); print('official and four additional folds:',scores.round(6).tolist()); print('worst fold:',scores.max(0).round(6).tolist())
p=s['panel_a']; jk=np.array([fold(folds[0],{k:np.delete(p[k],j,0) for k in ('u','v')}) for j in range(len(p['u']))]); print('worst leave-one-calibration-row-out:',jk.max(0).round(6).tolist()); assert (scores[0]<=.038).all() and (jk.max(0)<=.038).all()
yp,qp=predict(s); assert yp.shape==(len(s['test']['x']),) and qp.shape==(len(yp),len(s['eval_times']))
PY
```
- **Container Exit Code**: `0` (`1085 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
official and four additional folds: [[0.035047, 0.016639], [0.022232, 0.010805], [0.031237, 0.02286], [0.028749, 0.011863], [0.026078, 0.019188]]
worst fold: [0.035047, 0.02286]
worst leave-one-calibration-row-out: [0.035047, 0.016639]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=95668facc57733a4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_95668fac (directory).
```

### Turn 10 — `submit_completed_task` (`10891 ms` LLM)
- **Rationale**: The saved candidate passed the isolated public test, all five evaluated splits stayed below both MAE limits, and leaving out any one calibration row did not worsen the official split scores at the reported precision.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_01.py`; the frozen nodes were unchanged. The public DAG test passed with scalar MAE 0.035047 and trajectory MAE 0.016639, both below 0.038. Each node executed exactly once.`

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
