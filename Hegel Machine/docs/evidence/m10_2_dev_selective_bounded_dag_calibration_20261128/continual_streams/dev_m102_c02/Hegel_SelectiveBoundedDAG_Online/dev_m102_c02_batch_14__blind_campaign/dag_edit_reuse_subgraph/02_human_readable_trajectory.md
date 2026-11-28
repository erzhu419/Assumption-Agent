# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_m102_c02_batch_14__blind_campaign__reuse_subgraph`

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
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `348,402` (`332,989` / `137,728` / `15,413`) |
| **Total Episode Wall Time (s)** | `437.81s` |

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

### Turn 1 — `execute_in_remote_docker` (`14833 ms` LLM)
- **Rationale**: Inspect the measurements, archived candidates, and exact frozen-node contract before selecting a calibration model.

```bash
ls -la /holdout_task; find /tmp/hegel_continual_archive -maxdepth 3 -type f | sort | head -120; cat /holdout_task/_dag_spec.json /holdout_task/_frozen_nodes/node_01.py; cat /holdout_task/test_dag_candidate.py; cat /holdout_task/public_asset_index.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel nonlinear alignment calibration imputation' || true; python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
def show(x,p='',d=0):
 if isinstance(x,dict):
  for k,v in x.items():show(v,p+'/'+str(k),d+1)
 elif isinstance(x,(list,tuple)):
  print(p,type(x),len(x));
  if d<4:
   for i,v in enumerate(x[:2]):show(v,p+'/'+str(i),d+1)
 elif isinstance(x,np.ndarray):
  print(p,x.shape,x.dtype,repr(x.flat[:10]))
 else:print(p,type(x),str(x)[:200])
show(x)
PY
```
- **Container Exit Code**: `0` (`423 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
g_candidate.py": {
      "bytes": 8018
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 2825
    }
  },
  "schema": "hegel_public_asset_index_v1"
}No matching skill cards found.
 <class 'list'> 1
/0/study/eval_times (8,) float64 array([0.25      , 0.64285714, 1.03571429, 1.42857143, 1.82142857,
       2.21428571, 2.60714286, 3.        ])
/0/study/panel_a/u (48, 4) float64 array([ 0.74039306, -0.27304113,  0.63673058, -1.25207299,  0.11725165,
        1.2596387 ,  0.02419325, -0.55435477, -0.53620942,  0.75312059])
/0/study/panel_a/v (48, 4) float64 array([ 0.47122764, -0.19559824,  0.22547492, -1.25653141,  0.11248229,
        1.7027739 , -0.23796283, -0.76517814, -0.29279824,  1.15110585])
/0/study/panel_b/u (24, 4) float64 array([-0.21958156, -0.47520124,  1.14405023,  0.47331887,  0.20582542,
        0.72577255,  0.50156753, -1.07293627,  0.63065777,  0.79987289])
/0/study/panel_b/v (24, 4) float64 array([-0.45327019, -0.13854649, -0.3887508 , -0.75282675,  0.1750348 ,
       -0.74356502,  0.13069209, -0.22335847,  0.37832134, -0.84639659])
/0/study/panel_c/u (24, 4) float64 array([-0.71872626,  0.39040372,  0.22014769,  0.96776634,  0.52871625,
       -0.21500673, -0.2438462 ,  1.08931988, -0.57873291, -0.20634793])
/0/study/panel_c/v1 (24, 4) float64 array([-0.43310645, -0.65057528, -0.80521827, -0.98435947,  0.28182216,
       -0.80277564, -0.57174809,  0.64539822, -0.25828139,  0.546182  ])
/0/study/panel_c/v2 (24, 4) float64 array([-0.38449164, -0.0672155 , -0.64051643,  0.26873752, -0.63468306,
       -0.40921144, -0.64481846, -0.26329045,  0.51296781, -0.13481555])
/0/study/train/x (72, 4) float64 array([-0.45519572,  0.38309722,  1.20183388, -0.79261217, -0.36883934,
       -0.32447693, -0.99704466, -0.62819428,  1.19605893,  0.00170249])
/0/study/train/y (72,) float64 array([ 0.35782904, -0.36257915, -0.55541021, -2.48682013, -0.33833048,
       -0.11468259, -0.58305875, -0.34629536,  1.17025408,  0.69416359])
/0/study/train/traj (72, 8) float64 array([ 0.1903159 ,  0.37081171,  0.53521058,  0.65985903,  0.75458479,
        0.84560898,  0.92443038,  0.99911944, -0.27728633, -0.50662836])
/0/study/train/summary (4,) float64 array([-0.3235892 ,  0.95419612, -0.32556109,  0.80575739])
/0/study/test/x (36, 4) float64 array([-0.76093254,  1.48497178,  0.60377762, -0.76662807, -0.47466137,
        0.99355756, -0.17907381, -0.86632704,  0.82510432,  0.93148892])
/0/study/telemetry_vec (4,) float64 array([0.22, 2.15, 1.25, 1.25])
```

### Turn 2 — `execute_in_remote_docker` (`10940 ms` LLM)
- **Rationale**: Recover the archived calibration basis and the frozen sink’s exact input contract before fitting a candidate.

```bash
cat /holdout_task/_frozen_nodes/node_01.py; ls -lh /tmp/hegel_continual_archive; grep -nE '^def |panel_b|dag_11|dag_12' /tmp/hegel_continual_archive/*solver.py; grep -nE 'def |fold|public_cv|accepted' /holdout_task/test_dag_candidate.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling' || true
```
- **Container Exit Code**: `0` (`320 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
r.py:62:def solve_batch(study: dict) -> dict:
/tmp/hegel_continual_archive/dev_m102_c02_batch_00_solver.py:64:    maps = _panel_maps(study['panel_a'], study['panel_b'])
/tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py:7:def _lin(z):
/tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py:11:def _ls(x, y):
/tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py:15:def _cal(p, points):
/tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py:28:def _op(p,k,z,mode):
/tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py:42:def _cross(a,b):
/tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py:46:def _ridge(a,y,q,r):
/tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py:53:def _kernel(f, b, y, n, ell, reg):
/tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py:62:def solve_batch(study: dict) -> dict:
/tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py:64:    z=_cal(study['panel_a'],np.r_[x,q]); b=_op(study['panel_b'],'v',z,'b'); c=_op(study['panel_c'],'v1',z,'c')
9:def _canon_digest(obj) -> str:
11:    def _feed(v):
37:def _collect_subtree_digests(obj) -> set[str]:
45:def _collect_numeric_leaf_digests(obj) -> set[str]:
58:    def __init__(self, data: dict, src_key: str, tracker: set):
62:    def __getitem__(self, k):
65:    def get(self, k, default=None):
68:    def __iter__(self):
71:    def __len__(self): return len(self._data)
72:    def keys(self):
75:    def items(self):
78:    def values(self):
81:    def __contains__(self, k):
89:        print(json.dumps({"accepted": False, "error": f"missing_node_file:{fp}"}))
95:        print(json.dumps({"accepted": False, "error": f"missing_run_node:{nid}"}))
99:def run_dag(study: dict):
164:    cv_metrics = {"public_cv_y_mae": round(ym, 6), "public_cv_traj_mae": round(tm, 6)}
165:    if ym > 0.038: errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
166:    if tm > 0.038: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")
169:print(json.dumps({"accepted": ok, "errors": errs, "cv_metrics": cv_metrics, "exec_counts": ec}, allow_nan=False))
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 3 — `execute_in_remote_docker` (`11845 ms` LLM)
- **Rationale**: Read the archived calibration equations and fold construction before testing whether a nonlinear panel-B map improves both frozen readouts.

```bash
cat /tmp/hegel_continual_archive/*solver.py /tmp/hegel_continual_archive/archive_manifest.json; sed -n '125,175p' /holdout_task/test_dag_candidate.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -30
```
- **Container Exit Code**: `0` (`328 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
a.shape}")
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
```

### Turn 4 — `execute_in_remote_docker` (`20149 ms` LLM)
- **Rationale**: Isolate the archive’s reusable calibration equations and quantify how far each measured panel departs from affine structure.

```bash
sed -n '1,47p' /tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py; cat /tmp/hegel_continual_archive/archive_manifest.json; python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];
for k in ('panel_a','panel_b','panel_c'):
 p=s[k];u=p['u']; a=np.c_[np.ones(len(u)),u];
 for l in p.keys()-{'u'}:
  v=p[l]; print(k,l,'affine RMSE',np.sqrt(np.mean((a@np.linalg.lstsq(a,v,rcond=None)[0]-v)**2,axis=0)));print('affine coefficients',np.linalg.lstsq(a,v,rcond=None)[0].round(4)); print('range',u.min(0),u.max(0))
PY
```
- **Container Exit Code**: `0` (`405 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
at_04": 1.0,
      "stat_05": 0.0,
      "stat_06": 0.0,
      "stat_07": 0.647492,
      "stat_08": 0.961774,
      "stat_09": 0.038039,
      "stat_10": 0.892032,
      "stat_11": 0.107833,
      "stat_12": 0.91225,
      "stat_13": 0.343018,
      "stat_14": 0.22,
      "stat_15": 0.22,
      "stat_16": 0.22,
      "stat_17": 0.22
    }
  },
  {
    "short_name": "dev_m102_c02_batch_08",
    "task_id": "dev_m102_c02_batch_08__blind_campaign",
    "solver_path": "/tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py",
    "stat_summary": {
      "stat_00": 0.944684,
      "stat_01": 0.337303,
      "stat_02": 0.020904,
      "stat_03": 0.918507,
      "stat_04": 0.80951,
      "stat_05": 0.190331,
      "stat_06": 0.081196,
      "stat_07": 0.430419,
      "stat_08": 0.946192,
      "stat_09": 0.053588,
      "stat_10": 0.882032,
      "stat_11": 0.117889,
      "stat_12": 0.859951,
      "stat_13": 0.197897,
      "stat_14": 0.22,
      "stat_15": 1.25,
      "stat_16": 1.25,
      "stat_17": 1.25
    }
  }
]panel_a v affine RMSE [1.48713338 1.45269006 1.49529364 1.5129521 ]
affine coefficients [[ 0.1158  0.2652 -0.3535 -0.1789]
 [ 1.2081  0.0454 -0.3122 -0.0657]
 [ 0.0306  0.7844  0.339  -0.2115]
 [-0.1905  0.433   0.6159 -0.1598]
 [ 0.2891  0.0438 -0.3374  0.8352]]
range [-1.26861938 -1.29267111 -1.28851669 -1.25207299] [1.27249953 1.29638061 0.98471436 1.19090474]
panel_b v affine RMSE [0.16636118 0.12768899 0.11770594 0.15685281]
affine coefficients [[-0.1476 -0.0644 -0.0111 -0.1567]
 [-0.0248  0.0997  0.4474 -0.3746]
 [-0.3522 -0.4123  0.1269  0.1773]
 [-0.1557 -0.1123 -0.3235 -0.4772]
 [-0.4152  0.3791  0.015   0.1184]]
range [-1.21540121 -1.13589757 -1.09768145 -1.19965149] [1.05434465 1.16280434 1.23509007 1.19480422]
panel_c v1 affine RMSE [0.12834933 0.08808347 0.09881052 0.10026397]
affine coefficients [[-0.0886 -0.1127 -0.0288 -0.1309]
 [-0.221  -0.1731  0.1047  0.9213]
 [-0.5982 -0.4565  0.43   -0.272 ]
 [-0.5485  0.4612 -0.498  -0.0633]
 [ 0.1114 -0.5826 -0.7436  0.0655]]
range [-1.24452427 -1.23917962 -1.23318527 -0.82640275] [0.98666294 1.12115344 1.22967892 1.18183872]
panel_c v2 affine RMSE [0.11338262 0.08158511 0.07128605 0.07734237]
affine coefficients [[-0.0398 -0.3097 -0.1638  0.0011]
 [-0.0179 -0.4151  0.0478 -0.1518]
 [-0.1393 -0.085   0.1212  0.3624]
 [ 0.3798 -0.1028 -0.2062  0.1163]
 [-0.3341 -0.0008 -0.4812 -0.0351]]
range [-1.24452427 -1.23917962 -1.23318527 -0.82640275] [0.98666294 1.12115344 1.22967892 1.18183872]
```

### Turn 5 — `execute_in_remote_docker` (`27166 ms` LLM)
- **Rationale**: Compare leave-one-out calibration errors and held-out errors for both frozen readouts, separating panel-A outlier handling from panel-B model selection.
- **Alternatives Weighed**: `Transfer the archived nonlinear solver wholesale` (rejected: Only the calibration node is editable; the frozen sink requires the existing c/query contract.)

```bash
cat >/tmp/probe.py <<'PY'
import pickle,numpy as np,importlib.util
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def load(p):
 spec=importlib.util.spec_from_file_location('m',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=load('/tmp/hegel_continual_archive/dev_m102_c02_batch_00_solver.py'); new=load('/tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py'); sink=load('/holdout_task/_frozen_nodes/node_01.py')
a,b=s['panel_a'],s['panel_b']; u,v=a['u'],a['v']; x=s['train']['x']; lin=new._lin; ls=new._ls
m=old._panel_maps(a,b); z0=(x-m[0])/m[1];z1=new._cal(a,x)
print('A diagonal parameters:',m[:2]);print('A diagonal residual sorted:',np.sort(np.linalg.norm(v-m[0]-u*m[1],axis=1)).round(4))
w=np.roll(u,-1,axis=1); d=np.c_[lin(u),u*w,u**3]; e=np.linalg.norm(v-d@ls(d,v),axis=1);keep=np.argsort(e)[:-3]
for name,ds in [('diag',lambda j:lin(u[:,j,None])),('cyclic',lambda j:np.c_[np.ones(len(u)),u[:,j],w[:,j],u[:,j]*w[:,j],u[:,j]**3]),('full',lambda j:d)]:
 pred=np.column_stack([ds(j)@ls(ds(j)[keep],v[keep,j]) for j in range(4)]); print('A',name,np.sort(np.linalg.norm(v-pred,axis=1)).round(4))
print('calibrated extremes:',z0.min(0),z0.max(0),z1.min(0),z1.max(0))
u,v=b['u'],b['v']
for name,D in [('affine',lambda z,a:lin(z)),('rational',lambda z,a:lin(z/(1+a*z*z))),('rat+linear',lambda z,a:np.c_[lin(z/(1+a*z*z)),z]),('atan',lambda z,a:lin(np.arctan(a*z))),('tanh',lambda z,a:lin(np.tanh(a*z))),('poly3',lambda z,a:np.c_[lin(z),z**3])]:
 def obj(a):
  d=D(u,a); h=np.sum(np.linalg.qr(d,mode='reduced')[0]**2,axis=1);return np.mean(((v-d@ls(d,v))/(1-h[:,None]))**2)
 a=minimize_scalar(obj,bounds=(.03,10),method='bounded').x; d=D(u,a); beta=ls(d,v);print('B',name,'a',round(a,5),'fit, loo',np.sqrt(np.mean((v-d@beta)**2)),np.sqrt(obj(a)))
 for tag,z in [('A diagonal',z0),('A cyclic',z1)]:
  c=D(z,a)@beta;n=52;y=s['train']['y'];t=s['train']['traj'];times=s['eval_times']; yp=sink._scalar_readout(c[:n],c[n:],y[:n]);tp=sink._trajectory(c[:n],c[n:],t[:n],times);print(' ',tag,np.mean(abs(yp-y[n:])),np.mean(abs(tp-t[n:])))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe.py
```
- **Container Exit Code**: `0` (`986 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.5352  0.549   0.5574  0.5996  0.6116  0.6204
 11.8581 12.2511 12.5309]
A diag [ 0.2432  0.2738  0.2828  0.2863  0.2919  0.3053  0.3204  0.3392  0.3412
  0.3435  0.3452  0.3482  0.3492  0.3995  0.4037  0.4165  0.4182  0.4202
  0.4282  0.4295  0.4299  0.4303  0.4326  0.4327  0.433   0.4532  0.4535
  0.4558  0.4665  0.478   0.4847  0.4877  0.493   0.4973  0.4995  0.502
  0.5029  0.5069  0.5292  0.5352  0.549   0.5574  0.5996  0.6116  0.6204
 11.8581 12.2511 12.5309]
A cyclic [ 0.2058  0.2168  0.2312  0.2428  0.2547  0.2555  0.2698  0.2699  0.3009
  0.311   0.3204  0.3211  0.3302  0.334   0.3389  0.3429  0.355   0.3797
  0.3814  0.389   0.3986  0.4071  0.4107  0.4124  0.4331  0.4367  0.4392
  0.443   0.4833  0.4869  0.4876  0.4886  0.4955  0.5019  0.5049  0.51
  0.5114  0.5228  0.5256  0.5271  0.5476  0.551   0.5652  0.5712  0.5789
 11.7374 12.2009 12.611 ]
A full [ 0.0136  0.0138  0.014   0.0142  0.0147  0.0149  0.0158  0.0169  0.019
  0.0192  0.0193  0.0197  0.0201  0.0217  0.0219  0.022   0.0249  0.026
  0.0275  0.0276  0.0292  0.0297  0.0302  0.0321  0.0327  0.0341  0.0346
  0.035   0.0357  0.0359  0.0361  0.0367  0.037   0.0372  0.038   0.0409
  0.0412  0.0444  0.0467  0.0492  0.0505  0.0508  0.0525  0.0684  0.0767
 12.0371 12.2496 12.3612]
calibrated extremes: [-1.27228424 -1.46780454 -1.31475753 -1.27400857] [1.37998925 1.32315524 1.29395521 1.3935774 ] [-1.34055781 -1.54500941 -1.67981952 -1.51541644] [1.44769573 1.55971587 1.82989114 1.77424161]
B affine a 9.99999 fit, loo 0.14356010093498583 0.1825267157452572
  A diagonal 0.23739573866155284 0.21024313553434068
  A cyclic 0.23007212297757001 0.19692805121485274
B rational a 0.03001 fit, loo 0.1481531482896278 0.18850776100708885
  A diagonal 0.23639315730339416 0.2105517850628525
  A cyclic 0.21845426534991447 0.19504644500599982
B rat+linear a 0.77717 fit, loo 0.005587296772497112 0.009187535441825694
  A diagonal 0.4590246283147222 0.2740231884104079
  A cyclic 0.4238678977177332 0.28581411563907805
B atan a 0.03001 fit, loo 0.1436069282055795 0.18258772371597334
  A diagonal 0.23738602304958198 0.21024573324170776
  A cyclic 0.22995407955227906 0.19690159771920338
B tanh a 0.03001 fit, loo 0.14360694182236283 0.18258774160918229
  A diagonal 0.23738598893367885 0.21024573255224038
  A cyclic 0.2299540210312505 0.19690158259729523
B poly3 a 9.99999 fit, loo 0.02620405861240504 0.04364869818132863
  A diagonal 0.4365260429867712 0.2888353385408849
  A cyclic 0.44693298404738935 0.3188231757139787
```

### Turn 6 — `execute_in_remote_docker` (`29145 ms` LLM)
- **Rationale**: Fit calibration using only measured panel pairs, then test whether the coupled inverse improves both held-out readouts without modifying the frozen sink.
- **Alternatives Weighed**: `Transfer the archived local cyclic inverse unchanged` (rejected: It leaves structured panel-A residuals and misses both public prediction tolerances.)
- **ATMS NOGOOD DECLARED**: Refuted `A diagonal or locally coupled cyclic basis adequately describes panel A, and an affine map adequately describes panel B.` | Witness: `Panel A's median inlier residual norm fell from above 0.38 for the local cyclic fit to about 0.028 with the full basis; panel B leave-one-out RMSE fell from 0.1825 to 0.0092 with rational-plus-linear terms.` | Blanket Diagnosis: `Both frozen readouts share an upstream coordinate estimate; nonlinear panel B alone does not repair an incomplete panel-A inverse.` | Sublated Pivot: `Invert a coupled panel-A calibration and compare basis ablations with and without the measured nonlinear panel-B map.`

```bash
cat >/tmp/probe_full.py <<'PY'
exec(open('/tmp/probe.py').read().split('m=old._panel_maps')[0])
from scipy.optimize import root
u,v=a['u'],a['v']
def design(z): return np.c_[lin(z),z*np.roll(z,-1,axis=1),z**3]
d=design(u); e=np.linalg.norm(v-d@ls(d,v),axis=1); keep=np.argsort(e)[:-3]; ba=ls(d[keep],v[keep]); print('A coefficient rows 1, linear, cyclic product, cubic:\n',ba.round(5));
ul,vl=b['u'],b['v']; y=s['train']['y'];t=s['train']['traj']
for mode,cols in [('full',np.arange(13)),('affine',np.arange(5)),('affine+pair',np.arange(9)),('affine+cubic',np.r_[0:5,9:13])]:
 aa=ls(d[keep][:,cols],v[keep]); z=np.stack([root(lambda zz:design(zz[None])[:,cols][0]@aa-xx,xx).x for xx in x]); print('A',mode,'fit RMSE',np.sqrt(np.mean((d[keep][:,cols]@aa-v[keep])**2)),'latent max',abs(z).max())
 for tag,D in [('affine',lambda z,a:lin(z)),('raw',lambda z,a:lin(z)[:,1:]),('rational+linear',lambda z,a:np.c_[lin(z/(1+a*z*z)),z])]:
  def obj(a):
   dd=D(ul,a); h=np.sum(np.linalg.qr(dd,mode='reduced')[0]**2,axis=1);return np.mean(((vl-dd@ls(dd,vl))/(1-h[:,None]))**2)
  alpha=minimize_scalar(obj,bounds=(.03,5),method='bounded').x; dd=D(ul,alpha); bb=ls(dd,vl); c=D(z,alpha)@bb if tag!='raw' else z; n=52; yp=sink._scalar_readout(c[:n],c[n:],y[:n]);tp=sink._trajectory(c[:n],c[n:],t[:n],s['eval_times']); print(tag,alpha,np.mean(abs(yp-y[n:])),np.mean(abs(tp-t[n:]))); 
  if mode=='full' and tag=='rational+linear': print('B coefficient rows 1, rational, linear:\n',bb.round(5))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_full.py
```
- **Container Exit Code**: `0` (`1309 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
A coefficient rows 1, linear, cyclic product, cubic:
 [[-1.28500e-02  8.84400e-02 -1.73090e-01 -4.47200e-02]
 [ 1.14972e+00  1.24000e-02 -6.74430e-01  5.89000e-03]
 [ 1.75900e-02  1.15623e+00 -7.38000e-03 -6.46700e-01]
 [-6.41670e-01 -3.69000e-03  1.12415e+00  8.56000e-03]
 [ 1.10100e-02 -6.76750e-01 -4.54000e-03  1.26223e+00]
 [-4.30000e-03  1.17000e-03  2.66000e-03 -1.09000e-03]
 [ 1.94000e-02  9.23000e-03 -1.33300e-02 -2.29600e-02]
 [-5.95000e-03 -8.15000e-03  1.00600e-02  6.53000e-03]
 [ 7.10000e-03 -3.04000e-03 -1.51000e-03  5.54000e-03]
 [-1.04210e-01 -8.78000e-03  4.30040e-01 -8.48000e-03]
 [-7.52000e-03 -7.91500e-02 -2.83000e-03  3.96940e-01]
 [ 3.98920e-01  7.07000e-03 -5.54700e-02 -8.35000e-03]
 [-1.10600e-02  4.32890e-01 -8.80000e-04 -9.49100e-02]]
A full fit RMSE 0.01732347597580159 latent max 1.5804575366737357
affine 4.99999641386828 0.18396165849880125 0.24865711835547813
raw 4.99999641386828 0.47110802256582013 0.2486571183680215
rational+linear 0.7771662658887638 0.4227945030936954 0.30647195727671483
B coefficient rows 1, rational, linear:
 [[-0.12546 -0.02832  0.07059 -0.13701]
 [ 0.19613 -0.1685  -1.35881  0.97775]
 [ 1.04175  1.13318 -0.39654 -0.58534]
 [ 0.6386   0.39265  0.90008  1.22964]
 [ 1.14636 -1.22085  0.04937 -0.30229]
 [-0.17351  0.15768  1.26928 -0.90255]
 [-0.96694 -1.04948  0.38106  0.54637]
 [-0.59993 -0.35744 -0.83427 -1.14056]
 [-1.06756  1.12641 -0.05302  0.28375]]
A affine fit RMSE 0.13160430976060014 latent max 1.3432078590631529
affine 4.99999641386828 0.23395100320739964 0.21024313553574508
raw 4.99999641386828 0.33395467367752785 0.21024313551677304
rational+linear 0.7771662658887638 0.36098564464762 0.2088515747342244
A affine+pair fit RMSE 0.12218644474602766 latent max 1.4354483137725353
affine 4.99999641386828 0.24146101204485 0.20979632631622236
raw 4.99999641386828 0.34238087989861465 0.2097963263175207
rational+linear 0.7771662658887638 0.34756317685911603 0.18892808730098257
A affine+cubic fit RMSE 0.01931369332967329 latent max 1.5741941495087441
affine 4.99999641386828 0.18689175142629727 0.24313597884834204
raw 4.99999641386828 0.4901209084106929 0.24313597886079555
rational+linear 0.7771662658887638 0.2808790764738728 0.2908839460120046
```

### Turn 7 — `execute_in_remote_docker` (`36210 ms` LLM)
- **Rationale**: The dominant panel-A terms couple opposite channels and have large opposing linear and cubic coefficients, consistent with a saturating law rather than the archived adjacent-channel map. Fit candidate laws on calibration pairs only and test each inverse through both frozen readouts.
- **Alternatives Weighed**: `Keep the polynomial inverse` (rejected: Its small panel-A fit residual does not translate into accurate held-out predictions, and the fitted cubic may have incorrect inverse branches.)

```bash
cat >/tmp/probe_exact.py <<'PY'
exec(open('/tmp/probe.py').read().split('m=old._panel_maps')[0])
from scipy.optimize import least_squares,root
from itertools import product
u,v=a['u'],a['v']; kl=np.arange(4); opp=(kl+2)%4; si=np.linalg.lstsq(lin(u),v,rcond=None)[0]; err=np.linalg.norm(v-lin(u)@si,axis=1); keep=err<3; U,V=u[keep],v[keep]; print('A rows',len(U)); print('input keys',s.keys()); print('B/C rows',[(k,len(s[k]['u'])) for k in ['panel_b','panel_c']]);
F={'lin':lambda z,h:z,'poly':lambda z,h:z**3,'rat':lambda z,h:z/(1+h*z*z),'sqrt':lambda z,h:z/np.sqrt(1+h*z*z),'tanh':lambda z,h:np.tanh(h*z),'atan':lambda z,h:np.arctan(h*z),'sin':lambda z,h:np.sin(h*z),'gauss':lambda z,h:z*np.exp(-h*z*z),'soft':lambda z,h:z/(1+h*abs(z))};
results=[]
for own,cross in product(F,F):
 def dd(z,j,p):
  w=z[:,j]; q=z[:,opp[j]]; return np.column_stack([np.ones(len(z)),w]+([F[own](w,p[0])] if own!='lin' else [])+[F[cross](q,p[1])]);
 def fit(p,ret=False):
  ds=[dd(U,j,p) for j in kl]; co=[ls(d,V[:,j]) for j,d in enumerate(ds)]; rr=np.column_stack([d@c-V[:,j] for j,(d,c) in enumerate(zip(ds,co))]);return co if ret else rr.ravel();
 ini=[0.8,1.5]; p=least_squares(fit,ini,bounds=([.025,.025],[5.,5.]),max_nfev=55).x; results.append((np.sqrt(np.mean(fit(p)**2)),own,cross,p,fit(p,True)))
U2,V2=b['u'],b['v']; D=lambda z,h:np.c_[lin(z/(1+h*z*z)),z]; ob=lambda h:np.mean((D(U2,h)@ls(D(U2,h),V2)-V2)**2); h=minimize_scalar(ob,bounds=(.1,3),method='bounded').x; bb=ls(D(U2,h),V2); print('B fitted shape',h,ob(h)**.5)
y=s['train']['y']; t=s['train']['traj']; n=52
for rm,own,cross,p,beta in sorted(results,key=lambda r:r[0])[:18]:
 def forward(z):return np.array([beta[j]@np.array([1,z[j]]+([F[own](z[j],p[0])] if own!='lin' else [])+[F[cross](z[opp[j]],p[1])]) for j in kl]);
 inv=[root(lambda z:forward(z)-xx,xx) for xx in x]; z=np.stack([i.x for i in inv]); c=D(z,h)@bb; yp=sink._scalar_readout(c[:n],c[n:],y[:n]); tp=sink._trajectory(c[:n],c[n:],t[:n],s['eval_times']); print('A',rm,own,cross,p.round(5),'coef',beta[0].round(5),'inverse',sum(i.success for i in inv),np.max(abs(z)),'CV',np.mean(abs(yp-y[n:])),np.mean(abs(tp-t[n:])));
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_exact.py; sed -n '30,115p' /tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py; tail -35 /holdout_task/_frozen_nodes/node_01.py
```
- **Container Exit Code**: `0` (`4083 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.asarray(tr['y'],float)[:,None]
    y_hat=(.75*_kernel(f,b,y,n,2.,.001)+.25*_kernel(f,b,y,n,1.,.03)).ravel()
    sf=_lin(np.c_[z,b,_cross(z,b)])
    state=_ridge(sf[:n],np.asarray(tr['traj'],float),sf[n:],.08)
    # The terminal state carries much of the scalar readout's interaction.
    # Fit the difference on measured training states, then add it to the
    # predicted terminal state. Keep the independent readout as a hedge.
    diff = y[:,0] - np.asarray(tr['traj'], float)[:,-1]
    z5 = z**5
    gf = _lin(np.c_[b, z5, _cross(b,z5)])
    u = np.asarray(study['panel_c']['u'],float)
    poly = lambda x: np.c_[_lin(x),x*x,x**3]
    d = poly(z) @ _ls(poly(u),np.asarray(study['panel_c']['v2'],float))
    df = _lin(np.c_[b,d])
    correction = (.65*_ridge(gf[:n],diff,gf[n:],.12)
                  +.35*_ridge(df[:n],diff,df[n:],.4))
    y_hat = .40*y_hat + .60*(state[:,-1]+correction)
    return {'y_hat':y_hat,'state_pred':state,'diagnostics_vec':np.array([y_hat.mean(),y_hat.std(),state[:,-1].mean(),state[:,-1].std()])}
    design, new_design = (_affine(c), _affine(query)) if intercept else (c, query)
    beta = np.linalg.lstsq(design, traj, rcond=None)[0]
    fitted = design @ beta
    scale = max(float(np.max(np.abs(times))), np.finfo(float).tiny)
    t = times / scale

    def basis(log_rates):
        return -np.expm1(-np.exp(log_rates[:, None]) * t)

    def residual(log_rates):
        modes = basis(log_rates)
        return (fitted @ np.linalg.pinv(modes) @ modes - fitted).ravel()

    log_rates = least_squares(residual, np.log([0.6, 2.7]),
                              bounds=(np.log([0.03, 0.25]), np.log([4., 20.])),
                              max_nfev=100).x
    modes = basis(log_rates)
    return new_design @ (beta @ np.linalg.pinv(modes)) @ modes




def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    coordinates = inputs['node_00']
    c = np.asarray(coordinates['c'], dtype=float)
    query = np.asarray(coordinates['query'], dtype=float)
    train = study['train']
    y_hat = _scalar_readout(c, query, np.asarray(train['y'], dtype=float))
    state_pred = _trajectory(c, query, np.asarray(train['traj'], dtype=float),
                             np.asarray(study['eval_times'], dtype=float))
    final_state = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(),
                                         final_state.mean(), final_state.std()])}
```

### Turn 8 — `execute_in_remote_docker` (`34477 ms` LLM)
- **Rationale**: Use trajectory linearity as a diagnostic for the coordinate convention and independently compare the measured panel maps and odd basis terms before writing the candidate.
- **Alternatives Weighed**: `Deploy the best panel-only fit immediately` (rejected: The fit repairs the measured inverse but still misses both downstream thresholds.)

```bash
cat >/tmp/diagnose.py <<'PY'
exec(open('/tmp/probe.py').read().split('m=old._panel_maps')[0])
from scipy.optimize import root,least_squares
from itertools import combinations
j=np.arange(4); op=(j+2)%4; u,v=a['u'],a['v']; good=np.linalg.norm(v-lin(u)@ls(lin(u),v),axis=1)<3
u,v=u[good],v[good]
def ad(z,p):return np.stack([np.ones_like(z),z,np.tanh(p[0]*z),np.sin(p[1]*z[:,op])],axis=-1)
def af(p,ret=False):
 d=ad(u,p); m=np.stack([ls(d[:,k],v[:,k]) for k in j]); return m if ret else (np.sum(d*m,axis=-1)-v).ravel()
p=least_squares(af,[1.8,2.4]).x; m=af(p,True)
def inv(q):return np.array([root(lambda zz:np.sum(ad(zz[None],p)[0]*m,axis=-1)-xx,xx).x for xx in q]);
z=inv(x); y=s['train']['y']; t=s['train']['traj']; n=52
print('trajectory singular values:',np.linalg.svd(t,compute_uv=False)); print('trajectory mean, endpoint range:',t.mean(0),t[:,-1].min(),t[:,-1].max())
D={ 'affine':lambda z,h:lin(z), 'poly3':lambda z,h:np.c_[lin(z),z**3], 'poly235':lambda z,h:np.c_[lin(z),z*z,z**3,z**5], 'ratlin':lambda z,h:np.c_[lin(z/(1+h*z*z)),z], 'atanlin':lambda z,h:np.c_[lin(np.arctan(h*z)),z], 'tanhlin':lambda z,h:np.c_[lin(np.tanh(h*z)),z], 'sinlin':lambda z,h:np.c_[lin(np.sin(h*z)),z], 'ratpure':lambda z,h:lin(z**3/(1+h*z*z)), 'ratmix':lambda z,h:lin((z**3-.1*z)/(1+h*z*z)), 'atancube':lambda z,h:np.c_[lin(np.arctan(h*z)),z,z**3], 'ratquad':lambda z,h:np.c_[lin(z/(1+h*z*z)),z,z*z] }
def calerr(q):
 d=lin(q); h=np.mean(abs(d[n:]@ls(d[:n],t[:n])-t[n:])); yy=np.mean(abs(sink._scalar_readout(q[:n],q[n:],y[:n])-y[n:]));return round(yy,5),round(h,5)
for k,q in [('x',x),('z',z),('z**3',z**3),('z**5',z**5),('A nonlinear',np.tanh(1.8*z)),('z rational',z/(1+1.25*z*z))]:print(k,calerr(q))
for panel,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[panel]['u'],s[panel][key];
 for name,d in D.items():
  def ob(h):
   a=d(u,h); e=1-np.sum(np.linalg.qr(a,mode='reduced')[0]**2,axis=1); return np.mean(((v-a@ls(a,v))/e[:,None])**2)
  h=minimize_scalar(ob,bounds=(.05,4.),method='bounded').x; a=d(u,h); b=ls(a,v); q=d(z,h)@b
  print(panel,key,name,round(h,4),round(ob(h)**.5,6),calerr(q))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/diagnose.py; sed -n '1,110p' /holdout_task/_frozen_nodes/node_01.py; sed -n '125,170p' /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`3385 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
== _canon_digest(outputs[s]) for s in srcs) or not out_num or not (out_num - in_num):
                raise RuntimeError(f"DAG node {nid} failed anti-wrapper-shell / novel numeric transformation check")
        outputs[nid] = {k: (dict(v) if isinstance(v, _TrackedPayloadDict) else v) for k, v in out_v.items()}
    return outputs[spec_obj["sink_node_id"]], ec

st = pickle.load(open("/holdout_task/public_batch.pkl", "rb"))[0]["study"]
out_full, ec = run_dag(st)
n_te, k_t = len(st["test"]["x"]), len(st["eval_times"])
errs = []
for k, shp in (("y_hat", (n_te,)), ("state_pred", (n_te, k_t)), ("diagnostics_vec", (4,))):
    if k not in out_full:
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
```

### Turn 9 — `execute_in_remote_docker` (`47580 ms` LLM)
- **Rationale**: Save the measured calibration model, run the official test, and use trajectory regression and compact panel-B basis tests to distinguish an operator mismatch from an incomplete calibration basis.

```bash
cat > /holdout_task/node_00.py <<'PY'
"""Recover latent coordinates from panel A and apply the panel-B operator."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar, root


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _latent(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    opposite = (np.arange(u.shape[1]) + 2) % u.shape[1]
    residual = np.linalg.norm(v - _affine(u) @ _ls(_affine(u), v), axis=1)
    keep = residual < max(6 * np.median(residual), 0.1)
    if keep.sum() >= 8:
        u, v = u[keep], v[keep]

    def basis(z, p):
        return np.stack((np.ones_like(z), z, np.tanh(p[0]*z), np.sin(p[1]*z[:, opposite])), axis=-1)

    def project(p, return_coef=False):
        a = basis(u, p)
        coef = np.stack([_ls(a[:, j], v[:, j]) for j in range(u.shape[1])])
        return coef if return_coef else (np.sum(a*coef, axis=-1)-v).ravel()

    p = least_squares(project, [1.8, 2.4], bounds=([0.3, 0.8], [4., 4.])).x
    coef = project(p, True)
    return np.stack([root(lambda z: np.sum(basis(z[None], p)[0]*coef, axis=-1)-q, q).x for q in x])


def _panel_b(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    def basis(q, a):
        return np.c_[_affine(q / (1 + a*q*q)), q]
    def loss(a):
        d = basis(u, a)
        return np.mean((d @ _ls(d, v) - v)**2)
    a = minimize_scalar(loss, bounds=(.05, 4.), method='bounded').x
    return basis(z, a) @ _ls(basis(u, a), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.asarray(study['train']['x'], dtype=float)
    q = np.asarray(study['test']['x'], dtype=float)
    c = _panel_b(study['panel_b'], _latent(study['panel_a'], np.concatenate((x, q))))
    return {'c': c[:len(x)], 'query': c[len(x):]}
PY
export OPENBLAS_NUM_THREADS=1; python3 /holdout_task/test_dag_candidate.py; sed -n '27,56p' /tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py; sed -n '20,55p' /tmp/hegel_continual_archive/dev_m102_c02_batch_00_solver.py; cat /tmp/hegel_continual_archive/archive_manifest.json | cut -c1-180 | head -40; grep -n 'stat_\|shape\|panel' /holdout_task/public_asset_index.json | tail -45
cat > /tmp/structure.py <<'PY'
exec(open('/tmp/probe.py').read().split('m=old._panel_maps')[0])
from scipy.optimize import least_squares
m=load('/holdout_task/node_00.py'); z=m._latent(a,x); t=s['train']['traj']; y=s['train']['y']; n=52
u,v=b['u'],b['v']; print('panels stats',s['telemetry_vec'],'B range',u.min(0),u.max(0),'latent',z.min(0),z.max(0));
from itertools import combinations
for name,g in [('separate polynomial',lambda z:np.c_[z,z*z,z**3,z**4,z**5]),('pairs',lambda z:np.c_[z,z**3,*[z[:,i]*z[:,j] for i,j in combinations(range(4),2)]]),('sine',lambda z:np.c_[z,np.sin(2.15*z)]),('panels',lambda z:np.c_[m._panel_b(b,z),z]),('rational',lambda z:np.c_[z,z**3/(1+1.25*z*z)]),('all',lambda z:np.c_[z,z**2,z**3,np.sin(2.15*z),z**3/(1+1.25*z*z),*[z[:,i]*z[:,j] for i,j in combinations(range(4),2)]])]:
 d=lin(g(z)); pred=d[n:]@ls(d[:n],t[:n]); print('traj',name, np.mean(abs(pred-t[n:])),np.sqrt(np.mean((d@ls(d,t)-t)**2)))
for label,g,p in [('cube_rat',lambda u,h:u**3/(1+h[0]*u*u),[1.25]),('z_rat',lambda u,h:u/(1+h[0]*u*u),[1.25]),('offset_rat',lambda u,h:(u**3-h[1]*u)/(1+h[0]*u*u),[1.25,.1]),('linear_sine',lambda u,h:u-h[1]*np.sin(h[0]*u),[2.15,.3]),('linear_tanh',lambda u,h:u-h[1]*np.tanh(h[0]*u),[2.15,.3]),('rat_plus_square',lambda u,h:(u**3+h[1]*u*u+h[2]*u)/(1+h[0]*u*u),[1.25,0,.1]),('cube_plus_sine',lambda u,h:u**3-h[1]*np.sin(h[0]*u),[2.15,.3])]:
 def f(h,ret=False):
  d=lin(g(u,h)); c=ls(d,v); return c if ret else (d@c-v).ravel()
 h=least_squares(f,p,max_nfev=100).x;c=f(h,True);zz=lin(g(z,h))@c; d=lin(zz); print('B',label,h,np.sqrt(np.mean(f(h)**2)),np.mean(abs(d[n:]@ls(d[:n],t[:n])-t[n:])))
PY
python3 /tmp/structure.py
```
- **Container Exit Code**: `0` (`1108 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
032,
      "stat_11": 0.107833,
      "stat_12": 0.91225,
      "stat_13": 0.343018,
      "stat_14": 0.22,
      "stat_15": 0.22,
      "stat_16": 0.22,
      "stat_17": 0.22
    }
  },
  {
    "short_name": "dev_m102_c02_batch_08",
    "task_id": "dev_m102_c02_batch_08__blind_campaign",
    "solver_path": "/tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py",
    "stat_summary": {
      "stat_00": 0.944684,
      "stat_01": 0.337303,
      "stat_02": 0.020904,
      "stat_03": 0.918507,
      "stat_04": 0.80951,
      "stat_05": 0.190331,
      "stat_06": 0.081196,
      "stat_07": 0.430419,
      "stat_08": 0.946192,
53:              "shape": [
58:            "panel_a": {
61:                "shape": [
69:                "shape": [
76:            "panel_b": {
79:                "shape": [
87:                "shape": [
94:            "panel_c": {
97:                "shape": [
105:                "shape": [
113:                "shape": [
122:              "shape": [
130:                "shape": [
140:                "shape": [
147:                "shape": [
155:                "shape": [
163:                "shape": [
panels stats [0.22 2.15 1.25 1.25] B range [-1.21540121 -1.13589757 -1.09768145 -1.19965149] [1.05434465 1.16280434 1.23509007 1.19480422] latent [-1.07374152 -1.14309699 -1.11248138 -1.13116589] [1.09411537 1.130101   1.13143849 1.13061654]
traj separate polynomial 0.25094576841748434 0.1964897493772689
traj pairs 0.13521254780042585 0.13013467167781237
traj sine 0.22244627114852483 0.24678212649989562
traj panels 0.2207964891221343 0.2472333252992177
traj rational 0.22007606164076599 0.2480856514245281
traj all 0.08949129648336969 0.06705163541148387
B cube_rat [0.37984563] 0.01525598526078226 0.2238324408840798
B z_rat [-0.43749068] 0.06656685475106712 0.21670498453639925
B offset_rat [0.77254059 0.09761518] 0.006173213007324568 0.22734891249047617
B linear_sine [2.84874006 0.39750793] 0.005842611330551398 0.22644666647816675
B linear_tanh [1.40686247 0.75605367] 0.00623079974912004 0.22737442328968852
B rat_plus_square [ 0.77221022 -0.0022401  -0.09744722] 0.006147321881644452 0.22757807031448482
B cube_plus_sine [ 0.09378384 -1.27927178] 0.02813444600109898 0.22447742134321552

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1ca13d2ec145a380 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1ca13d2e (directory).
```

### Turn 10 — `execute_in_remote_docker` (`45015 ms` LLM)
- **Rationale**: Use training-only orthogonal residual search and calibration-derived features to locate the missing terms without fitting to the official holdout.
- **ATMS NOGOOD DECLARED**: Refuted `An accurate panel-A inverse and a separable panel-B map suffice for the frozen trajectory readout.` | Witness: `The panel-A inverse fits to 0.000094 RMSE, yet the official trajectory MAE remains 0.227899; adding mixed features lowers a separate linear trajectory diagnostic to 0.089491.` | Blanket Diagnosis: `Calibration accuracy and downstream coordinate sufficiency are distinct; the frozen linear trajectory readout cannot reconstruct missing mixed terms.` | Sublated Pivot: `Identify a compact interaction basis from training-only trajectory and scalar diagnostics.`

```bash
cat >/tmp/search_latent.py <<'PY'
exec(open('/tmp/probe.py').read().split('m=old._panel_maps')[0])
from itertools import combinations,product
from scipy.linalg import qr
m=load('/holdout_task/node_00.py'); z=m._latent(a,x); y=s['train']['y']; T=s['train']['traj']; n=52
B=m._panel_b(b,z); norm=lambda f: f/np.maximum(f[:n].std(0),1e-9)
U,S,V=np.linalg.svd(T[:n],full_matrices=False); A=T@V[:3].T
C=[]
for key in ['v1','v2']:
 p=s['panel_c'];C.append(new._op(p,key,z,'c') if key=='v1' else np.c_[np.ones(len(z)),z,z*z,z**3]@ls(np.c_[np.ones(len(p['u'])),p['u'],p['u']**2,p['u']**3],p[key]))
F={'z':z,'x':x,'b':B,'p':C[0],'q':C[1],'s':np.sin(2.15*z),'t':np.tanh(1.25*z),'r':z/(1+1.25*z*z),'h':z*z/(1+1.25*z*z),'g':z**3/(1+1.25*z*z),'z2':z*z,'z3':z**3,'z4':z**4,'z5':z**5}; D={}
for k,f in F.items():
 for j in range(4): D[k+str(j)]=f[:,j]
for k,l in combinations(['z','b','p','q','s','t','r','h','g','z2','z3'],2):
 for j,i in product(range(4),repeat=2):D[f'{k}{j}*{l}{i}']=F[k][:,j]*F[l][:,i]
for k in ['z','b','p','q','s','r','g']:
 for j,i in combinations(range(4),2):D[f'{k}{j}*{k}{i}']=F[k][:,j]*F[k][:,i]
for a1,b1,c1 in combinations(range(4),3):D[f'z{a1}*z{b1}*z{c1}']=z[:,a1]*z[:,b1]*z[:,c1]
for j,i in combinations(range(4),2):
 for k in [-1,1]:
  w=z[:,j]+k*z[:,i]
  for tag,f in [('sin',np.sin(2.15*w)),('rat',w/(1+1.25*w*w)),('mix',w**2/(1+1.25*w*w)),('cub',w**3/(1+1.25*w*w))]:D[f'{tag}(z{j}{k:+}z{i})']=f
names=list(D); d=np.column_stack(list(D.values())); d=(d-d[:n].mean(0));d=norm(d)
for base in [['z','b'],['z','b','p','q'],['z','z2','z3','z4','z5','r','h'],['z','b','p','q','s','r','h','g']]:
 basef=np.column_stack([F[k] for k in base]); print('BASE',base)
 for ia,Y in enumerate([A[:,0],y,y-A[:,0]*np.dot(y[:n],A[:n,0])/np.dot(A[:n,0],A[:n,0])]):
  src=lin(basef); inds=[]; print('output',ia)
  for step in range(12):
   df=np.c_[src,d[:,inds]]; co=ls(df[:n],Y[:n]); res=Y[:n]-df[:n]@co; ort=d[:n]-df[:n]@ls(df[:n],d[:n]); os=np.linalg.norm(ort,axis=0); merit=np.abs(ort.T@res)/np.maximum(os,1e-7);merit[os<1e-6]=0; ix=np.argmax(merit); print(step,round(np.mean(np.abs(df[n:]@co-Y[n:])),4),round(np.linalg.norm(res)/np.sqrt(n),4),names[ix],end='; '); inds.append(ix)
  print()
for base in [['z'],['z','b'],['z','b','p','q'],['z','z2','b','p','q']]:
 Z=norm(np.column_stack([F[k] for k in base])); Y=np.c_[y,A/np.maximum(A[:n].std(0),0.05)];
 from scipy.spatial.distance import cdist
 from scipy.linalg import solve
 for kind in ['rbf','matern','poly']:
  L=[];ds=cdist(Z,Z[:n],'sqeuclidean')/Z.shape[1]
  for ell,reg in product([.4,.7,1,1.6,2.5,4,7],[1e-6,1e-4,1e-3,.01,.05]):
   H=np.exp(-ds/ell**2) if kind=='rbf' else ((1+np.sqrt(ds)*ell)*np.exp(-np.sqrt(ds)*ell) if kind=='matern' else (1+Z@Z[:n].T/Z.shape[1])**int(ell+2)); K=H[:n]+reg*np.eye(n); inv=solve(K,np.eye(n),assume_a='pos'); w=inv@(Y[:n]-Y[:n].mean(0)); loo=w/np.diag(inv)[:,None]; E=Y[n:]-(H[n:]@w+Y[:n].mean(0));L.append((np.mean(abs(loo),axis=0)[:2],ell,reg,np.mean(abs(E),axis=0)[:2]))
  print('kernel',base,kind,sorted(L,key=lambda q:sum(q[0]))[:2])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/search_latent.py | tee /tmp/latent_stats; head -55 /holdout_task/_frozen_nodes/node_01.py
```
- **Container Exit Code**: `0` (`722 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.14507215, 0.1553335 ]), 0.4, 1e-06, array([0.10831915, 0.14840812])), (array([0.14569244, 0.15643461]), 0.4, 0.0001, array([0.10857952, 0.14933372]))]
kernel ['z', 'z2', 'b', 'p', 'q'] poly [(array([0.09821162, 0.10728862]), 0.4, 1e-06, array([0.11697215, 0.11222707])), (array([0.09821162, 0.10728862]), 0.7, 1e-06, array([0.11697215, 0.11222707]))]
"""Scalar and trajectory readouts on the frozen calibrated coordinates."""

import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _scalar_inputs(c):
    paired = np.sum(c[:, :2] * c[:, 2:], axis=1)
    return np.column_stack((c, paired))


def _scalar_readout(c, query, y):
    u, uq = _scalar_inputs(c), _scalar_inputs(query)
    x, xq = _affine(u), _affine(uq)

    def design(base, z, w):
        s = z @ w
        return np.column_stack((base, s / (1.0 + s * s)))

    def projected_residual(w):
        d = design(x, u, w)
        return d @ np.linalg.lstsq(d, y, rcond=None)[0] - y

    linear = np.linalg.lstsq(x, y, rcond=None)[0][1:]
    if np.linalg.norm(linear) < 1e-10:
        return xq @ np.linalg.lstsq(x, y, rcond=None)[0]
    direction = linear / np.linalg.norm(linear)
    optimum = min((least_squares(projected_residual, scale * direction,
                                 bounds=(-5., 5.), max_nfev=160, gtol=2e-10)
                   for scale in (0.6, 1.2, 2.0)), key=lambda fit: fit.cost)
    d = design(x, u, optimum.x)
    beta = np.linalg.lstsq(d, y, rcond=None)[0]
    return design(xq, uq, optimum.x) @ beta


def _trajectory(c, query, traj, times, intercept=True):
    design, new_design = (_affine(c), _affine(query)) if intercept else (c, query)
    beta = np.linalg.lstsq(design, traj, rcond=None)[0]
    fitted = design @ beta
    scale = max(float(np.max(np.abs(times))), np.finfo(float).tiny)
    t = times / scale

    def basis(log_rates):
        return -np.expm1(-np.exp(log_rates[:, None]) * t)

    def residual(log_rates):
        modes = basis(log_rates)
        return (fitted @ np.linalg.pinv(modes) @ modes - fitted).ravel()

    log_rates = least_squares(residual, np.log([0.6, 2.7]),
                              bounds=(np.log([0.03, 0.25]), np.log([4., 20.])),

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1ca13d2ec145a380 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1ca13d2e (directory).
```

### Turn 11 — `execute_in_remote_docker` (`68637 ms` LLM)
- **Rationale**: The quadratic latent/panel-B kernel improves both readouts over the separable calibration. Build a compact regularized coordinate map, select smoothness using training-only predictive diagnostics, then check the exact saved artifact in the official split and three additional held-out blocks.
- **Alternatives Weighed**: `Pass the entire interaction dictionary to the frozen sink` (rejected: Its nonlinear optimizer and unregularized least squares would have too many degrees of freedom for the available training rows.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cp /holdout_task/node_00.py /tmp/calibration_baseline.py
python3 - <<'PY'
p='/holdout_task/node_00.py'
s=open(p).read().split('def run_node')[0];open(p,'w').write(s)
PY
cat >> /holdout_task/node_00.py <<'PY'
def _gram_blocks(study, x, n):
    z = _latent(study['panel_a'], x)
    b = _panel_b(study['panel_b'], z)
    # Separate first-order, mixed second-order and smooth remainder priors.
    def gram(f):
        f = (f-f[:n].mean(0))/np.maximum(f[:n].std(0),1.e-7)
        return (f@f.T)/f.shape[1]
    h = float(np.asarray(study['telemetry_vec'])[3])
    zc = gram(z)
    bc = gram(b)
    rc = gram(z/(1+h*z*z))
    gc = gram(z**3/(1+h*z*z))
    ec = gram(z*z)
    sc = gram(np.sin(float(np.asarray(study['telemetry_vec'])[1])*z))
    pc = []
    p=study['panel_c'];u=np.asarray(p['u'],float)
    for key, basis in [('v1',lambda a:np.c_[_affine(a),np.arctan(2.4*a),a**3]),('v2',lambda a:np.c_[_affine(a),a*a,a**3])]:
        pc.append(gram(basis(z)@_ls(basis(u),np.asarray(p[key],float))))
    fc=(zc+bc)/2
    dc=np.maximum(fc.diagonal()[:,None]+fc.diagonal()[None,:]-2*fc,0.)
    blocks=[zc,bc,rc,gc,ec,sc,*pc,zc*zc,zc*bc,bc*bc,zc*gc,gc*gc,zc*ec,bc*ec,bc*pc[0],bc*pc[1],fc**3,fc**4,fc**5,np.exp(-.3*dc),np.exp(-.1*dc)]
    return np.stack([k/max(np.mean(k.diagonal()[:n]),1.e-10) for k in blocks])


def _smooth_models(grams, response):
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize
    n=len(response); eye=np.eye(n); ones=np.ones((n,1));
    avg=response.mean(); sd=max(response.std(),1.e-7); y=(response-avg)/sd
    gs=np.ascontiguousarray(grams[:,:n,:n]); flat=gs.reshape(len(gs),-1)
    def solve(w, full=False):
        k=(w[:-1]@flat).reshape(n,n)+w[-1]*eye+1.
        fact=cho_factor(k,check_finite=False); ki=cho_solve(fact,eye,check_finite=False); a=ki@y
        loo=y-a/np.diag(ki)
        if full:
            pred=(np.einsum('k,kij->ij',w[:-1],grams[:,:,:n])+1.)@a
            return np.mean(np.abs(y-loo)),pred*sd+avg,loo*sd+avg
        v=ki-np.outer(a,a)
        val=.5*(y@a)+np.log(np.diag(fact[0])).sum()
        grad=np.r_[.5*(flat@v.ravel()),.5*np.trace(v)]*w
        return val,grad
    def objective(a): return solve(np.exp(a))
    results=[]
    for initial in (.003,.03):
        w0=np.full(len(gs)+1,.04);w0[-1]=initial
        opt=minimize(objective,np.log(w0),jac=True,method='L-BFGS-B',bounds=[(-13,4)]*len(gs)+[(-11,-.65)],options={'maxiter':145,'ftol':2e-8})
        results.append(solve(np.exp(opt.x),True))
    for ii in ([0,1,8,9,10],[0,1,8,9,10,17],[0,1,3,8,9,10,11,12,17],[0,1,2,3,8,9,10,11,12],[0,1,8,9,10,17,18],[0,1,8,9,10,17,18,19],[0,1,8,9,10,20,21],list(range(len(gs)))):
        for order in (.1,1.):
            for noise in (2.e-5,.0002,.002,.01,.04):
                w=np.full(len(gs)+1,0.);w[ii]=1./len(ii);w[np.array(ii)[np.array(ii)>=17]]*=order;w[-1]=noise
                results.append(solve(w,True))
    results.sort(key=lambda q:q[0]); best=results[0][0]
    sel=[q for q in results[:12] if q[0]<best*1.10+1.e-8]
    wt=np.exp(-35*(np.array([q[0] for q in sel])/max(best,1.e-8)-1)); wt/=wt.sum()
    return sum(q[1]*w for q,w in zip(sel,wt)),sum(q[2]*w for q,w in zip(sel,wt)),best*sd


def _coordinates(study, x, q):
    n=len(x); g=_gram_blocks(study,np.r_[x,q],n)
    traj=np.asarray(study['train']['traj'],float); y=np.asarray(study['train']['y'],float)
    _,_,v=np.linalg.svd(traj,full_matrices=False)
    amp=traj@v[:2].T
    fits=[_smooth_models(g,amp[:,j]) for j in range(2)]
    ys=_smooth_models(g,y)
    # Estimate the shared component separately from the scalar-specific response.
    # Only out-of-fold model estimates are used to pick the decomposition strength.
    yt=[]
    for coupling in (.5,1.):
        aa=_affine(amp)
        beta=_ls(aa,y)*coupling
        res=_smooth_models(g,y-aa@beta)
        full=res[0]+_affine(np.column_stack([f[0] for f in fits]))@beta
        loo=res[1]+_affine(np.column_stack([f[1] for f in fits]))@beta
        yt.append((full,loo,float(np.mean(abs(loo-y)))))
    yy=[ys,*yt]; err=np.array([k[2] for k in yy]); wt=np.exp(-12*(err/err.min()-1));wt/=wt.sum()
    scalar=sum(k[0]*w for k,w in zip(yy,wt))
    c=np.column_stack((scalar, fits[0][0], fits[1][0], np.zeros(len(scalar))))
    return c, (err, [f[2] for f in fits])


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; x=np.asarray(study['train']['x'],float); q=np.asarray(study['test']['x'],float)
    c,_=_coordinates(study,x,q)
    return {'c':c[:len(x)],'query':c[len(x):]}
PY
python3 -I /holdout_task/test_dag_candidate.py || true
cat >/tmp/check11.py <<'PY'
exec(open('/tmp/probe.py').read().split('m=old._panel_maps')[0])
m=load('/holdout_task/node_00.py'); Y=s['train']['y']; T=s['train']['traj'];
for held in (np.arange(52,72),np.arange(0,20),np.arange(20,40),np.arange(40,60)):
 tr=np.setdiff1d(np.arange(len(x)),held); sub=dict(s);sub['train']=dict(x=x[tr],y=Y[tr],traj=T[tr]);sub['test']={'x':x[held]}
 c,e=m._coordinates(sub,x[tr],x[held]);o=sink.run_node({'__input__':sub,'node_00':{'c':c[:len(tr)],'query':c[len(tr):]}});print(held[0],e,'raw scalar',np.mean(abs(c[len(tr):,0]-Y[held])),'DAG',np.mean(abs(o['y_hat']-Y[held])),np.mean(abs(o['state_pred']-T[held])))
PY
python3 -I /tmp/check11.py
```
- **Container Exit Code**: `0` (`2292 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
{"accepted": false, "errors": ["public_cv_y_mae_exceeded:0.040069"], "cv_metrics": {"public_cv_y_mae": 0.040069, "public_cv_traj_mae": 0.013041}, "exec_counts": {"node_00": 1, "node_01": 1}}
52 (array([0.03466699, 0.03766912, 0.0434888 ]), [0.03371183025786741, 0.005101054805583417]) raw scalar 0.04046070285841998 DAG 0.04006926139511222 0.013041077377514332
0 (array([0.02458487, 0.04515286, 0.04575139]), [0.02890381532227678, 0.004549303399092422]) raw scalar 0.05717582575674131 DAG 0.05748730008693161 0.012050571606388002
20 (array([0.03650207, 0.0397531 , 0.03618292]), [0.024002807079369984, 0.004319269827535014]) raw scalar 0.052113789999327066 DAG 0.0498438567357332 0.015513472858459656
40 (array([0.0393269 , 0.03937128, 0.03988241]), [0.020746075389245052, 0.004356187521746836]) raw scalar 0.051891690211679206 DAG 0.05097210952456834 0.014444949933808907

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f1703d03ab4326b1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_f1703d03 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`53691 ms` LLM)
- **Rationale**: Preserve the verified trajectory map. Fit the shared scalar component and its smooth residual jointly, accounting for leave-one-out deletion when weighting scalar candidates, then rerun the blocked checks and official harness in isolated Python.
- **Alternatives Weighed**: `Retune the trajectory coordinates` (rejected: Trajectory passed on every measured slice; the remaining bottleneck is scalar generalization.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cp /holdout_task/node_00.py /tmp/node00_v02.py
python3 - <<'PY'
p='/holdout_task/node_00.py'
s=open(p).read(); idx=s.index('def _coordinates(')
s=s[:idx]+'''def _scalar_profile(grams, y, amplitude, fits):
    """Fit the scalar-specific residual while profiling the shared modes."""
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize

    n = len(y)
    ymean, yscale = y.mean(), max(y.std(), 1.e-10)
    target = (y-ymean)/yscale
    center = amplitude.mean(0)
    scale = np.maximum(amplitude.std(0), 1.e-8)
    observed = (amplitude-center)/scale
    smooth = (np.column_stack([f[0] for f in fits])-center)/scale
    loo_amp = (np.column_stack([f[1] for f in fits])-center)/scale
    eye = np.eye(n)
    candidates = []
    base = (grams[0]+grams[1])/2
    for nonlinear in (False, True):
        def shared(a):
            m = _affine(a)
            if nonlinear:
                u = a[:,0]
                m = np.c_[m, u/(1.+.7*u*u), u*a[:,1]]
            return m
        design = shared(observed)
        pred_design = shared(smooth)
        loo_design = shared(loo_amp)
        delta = loo_design-design
        # Broad prior on the linear shared terms; shrink optional nonlinear terms.
        prior = np.array([1., 2., 1.] + ([.1, .02] if nonlinear else []))
        cov_shared = (design*prior)@design.T
        blocks = np.stack([grams[i] for i in (0,1,2,3,4,5,8,9,10,11,13,14,16,17,18)] + [base**3, base**4])
        flat = np.ascontiguousarray(blocks[:,:n,:n]).reshape(len(blocks),-1)
        mat = np.ascontiguousarray(blocks[:,:,:n])
        def solve(w, predict=False):
            k = cov_shared+(w[:-1]@flat).reshape(n,n)+w[-1]*eye
            factor = cho_factor(k,check_finite=False)
            inv = cho_solve(factor,eye,check_finite=False)
            coef = inv@target
            if predict:
                diag = inv.diagonal()
                # Sherman-Morrison deletion for the profiled shared coefficients.
                beta = prior*(design.T@coef)
                deletion = ((inv@design)*prior)*(coef/diag)[:,None]
                out = target-coef/diag+delta@beta-np.sum(delta*deletion,axis=1)
                full = pred_design@beta+np.einsum('k,kij,j->i',w[:-1],mat,coef)
                return np.mean(abs(out-target))*yscale, full*yscale+ymean, out*yscale+ymean
            score = .5*(target@coef)+np.log(np.diag(factor[0])).sum()
            v = inv-np.outer(coef,coef)
            gradient = .5*np.r_[flat@v.ravel(),np.trace(v)]*w
            return score,gradient
        for start in (.002,.015):
            init = np.full(len(blocks)+1,.015); init[-1]=start
            opt = minimize(lambda p: solve(np.exp(p)),np.log(init),jac=True,method='L-BFGS-B',bounds=[(-13.,2.)]*len(blocks)+[(-10.,-3.)],options={'maxiter':120,'ftol':1.e-8})
            candidates.append(solve(np.exp(opt.x),True))
        for width in (.03,.15,.6):
            for nugget in (.0001,.001,.006):
                w = np.zeros(len(blocks)+1)
                w[[0,1,6,7,8,13,14]] = width*np.array([1,1,1,1,1,.15,.025])
                w[-1] = nugget
                candidates.append(solve(w,True))
    candidates.sort(key=lambda m:m[0]); top=candidates[:8]
    error=np.array([m[0] for m in top]); w=np.exp(-19.*(error/error.min()-1));w/=w.sum()
    full=sum(v[1]*a for v,a in zip(top,w)); loo=sum(v[2]*a for v,a in zip(top,w))
    return full,loo,float(np.mean(abs(loo-y)))


'''+s[idx:]
s=s.replace('from scipy.optimize import least_squares, minimize_scalar, root','from scipy.optimize import least_squares, minimize_scalar, root, minimize')
s=s.replace('yy=[ys,*yt]; err=np.array([k[2] for k in yy]); wt=np.exp(-12*(err/err.min()-1));wt/=wt.sum()', '''profile = _scalar_profile(g,y,amp,fits)
    yy=[ys,*yt,profile]; err=np.array([k[2] for k in yy]); wt=np.exp(-13*(err/err.min()-1));wt/=wt.sum()''')
open(p,'w').write(s)
PY
python3 -I /tmp/check11.py
python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`2471 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
52 (array([0.03466699, 0.03766912, 0.0434888 , 0.03147088]), [0.03371183025786741, 0.005101054805583417]) raw scalar 0.02830196826620911 DAG 0.027879319076971075 0.013040646669569172
0 (array([0.02458487, 0.04515286, 0.04575139, 0.03620991]), [0.02890381532227678, 0.004549303399092422]) raw scalar 0.05709311637136636 DAG 0.05740815054147204 0.012050562738912104
20 (array([0.03650207, 0.0397531 , 0.03618292, 0.035914  ]), [0.024002807079369984, 0.004319269827535014]) raw scalar 0.048072671937897854 DAG 0.04572798573831908 0.01551435846044975
40 (array([0.0393269 , 0.03937128, 0.03988241, 0.02644928]), [0.020746075389245052, 0.004356187521746836]) raw scalar 0.033751272694113925 DAG 0.03222110172481678 0.014442716719593707
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.027879, "public_cv_traj_mae": 0.013041}, "exec_counts": {"node_00": 1, "node_01": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=83b9868a011f2163 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_83b9868a (directory).
```

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
