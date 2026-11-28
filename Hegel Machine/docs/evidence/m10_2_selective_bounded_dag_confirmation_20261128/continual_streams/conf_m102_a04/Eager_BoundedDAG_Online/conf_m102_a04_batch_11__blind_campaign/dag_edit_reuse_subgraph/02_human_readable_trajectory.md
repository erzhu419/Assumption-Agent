# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m102_a04_batch_11__blind_campaign__reuse_subgraph`

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
| **Turns Used / Max Turns** | `8 / 12` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `206,178` (`200,627` / `96,256` / `5,551`) |
| **Total Episode Wall Time (s)** | `177.01s` |

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

### Turn 1 — `execute_in_remote_docker` (`14381 ms` LLM)
- **Rationale**: Read the frozen consumers and public evaluator first so that calibration can be tested without changing the downstream feature contract.

```bash
cd /holdout_task; cat _dag_spec.json public_asset_index.json; for f in _frozen_nodes/*.py; do echo ==== $f; cat "$f"; done; cat test_dag_candidate.py; echo ==== archive; ls -la /tmp/hegel_continual_archive; find /tmp/hegel_continual_archive -maxdepth 4 -type f | head -100; python3 - <<'PY'
import pickle, numpy as np
x=pickle.load(open('public_batch.pkl','rb'))
def walk(x, depth=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print(' '*depth,k,end=' '); walk(v,depth+2)
 elif isinstance(x,(tuple,list)):
  print(type(x),len(x));
  if x and depth<4: walk(x[0],depth+2)
 elif isinstance(x,np.ndarray):
  print(x.shape,x.dtype, [np.nanmin(x),np.nanmax(x),np.nanmean(x),np.nanstd(x)] if x.size and np.issubdtype(x.dtype,np.number) else '')
 else: print(type(x),repr(x)[:200])
walk(x)
PY
```
- **Container Exit Code**: `0` (`430 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
t(np.mean(np.abs(np.asarray(out_cv["state_pred"], float) - vtr)))
    cv_metrics = {"public_cv_y_mae": round(ym, 6), "public_cv_traj_mae": round(tm, 6)}
    if ym > 0.038: errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
    if tm > 0.038: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")

ok = len(errs) == 0 and all(v == 1 for v in ec.values())
print(json.dumps({"accepted": ok, "errors": errs, "cv_metrics": cv_metrics, "exec_counts": ec}, allow_nan=False))
sys.exit(0 if ok else 1)
==== archive
total 20
drwxr-xr-x 2 root root 4096 Nov 28 15:44 .
drwxrwxrwt 1 root root 4096 Nov 28 15:44 ..
-rw-r--r-- 1 root root  665 Nov 28 15:44 archive_manifest.json
-rw-r--r-- 1 root root 3047 Nov 28 15:44 conf_m102_a04_batch_00_solver.py
-rw-r--r-- 1 root root 1882 Nov 28 15:44 pre_llm_gate_diagnostic.json
/tmp/hegel_continual_archive/archive_manifest.json
/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
/tmp/hegel_continual_archive/conf_m102_a04_batch_00_solver.py
<class 'list'> 1
   study      eval_times (8,) float64 [0.25, 3.0, 1.625, 0.9001487972234686]
     panel_a        u (48, 4) float64 [-1.246654400755808, 1.295259844328324, 0.03817409420745264, 0.7366320576722347]
       v (48, 4) float64 [-7.319523313840926, 8.102465229361833, 0.06902731859414897, 1.8274238308689497]
     panel_b        u (24, 4) float64 [-1.2377684943726133, 1.2406215496457378, -0.02556736720036824, 0.7930210587631974]
       v (24, 4) float64 [-1.439548882014906, 1.4663595966607004, 0.061457227263735544, 0.6944574897301048]
     panel_c        u (24, 4) float64 [-1.2251881634363615, 1.240949855187245, 0.04246866432954657, 0.7537456604512681]
       v1 (24, 4) float64 [-1.1790277831184683, 1.3975096378467804, 0.06402694256171128, 0.6882067822446776]
       v2 (24, 4) float64 [-1.0646806595724698, 0.993891789127385, -0.1072472675884587, 0.3890249231105622]
     train        x (72, 4) float64 [-1.8612785221279886, 1.8340898187172723, -0.04116688382252874, 0.8509295249630229]
       y (72,) float64 [-0.7593151723304444, 1.234168915677607, 0.13111513908245354, 0.4703317894703821]
       traj (72, 8) float64 [-2.1798091876375127, 3.0461121457406866, 0.17999233058984668, 1.030398044241445]
       summary (4,) float64 [0.13141408746113947, 1.2719167325922471, 0.534166327426818, 0.4426858895411759]
     test        x (36, 4) float64 [-1.8982944791030096, 1.6797220204061465, 0.002049785763788056, 0.9022496169275416]
     telemetry_vec (4,) float64 [0.22, 2.15, 0.7025, 0.8357145146519832]
```

### Turn 2 — `execute_in_remote_docker` (`9872 ms` LLM)
- **Rationale**: The first inspection confirmed the input shapes, but its output truncated the inherited implementation. Read the compact archive and frozen consumers before reconstructing the feature map.

```bash
cd /holdout_task; for f in _frozen_nodes/*.py /tmp/hegel_continual_archive/*; do echo ==== "$f"; cat "$f"; done
```
- **Container Exit Code**: `0` (`342 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
it = np.mean([
        kernel[n:, :n] @ np.linalg.solve(kernel[:n, :n] + ridge*eye, targets)
        for kernel, ridge in ((quadratic, 3e-5),
                              (quadratic + np.exp(-.06*distance), 3e-4))
    ], axis=0)
    y = fit[:, 0]
    basis = _time_basis(np.asarray(study['eval_times']), train['traj'])
    state = (basis @ lin(basis, fit[:, 1:].T)).T
    terminal = state[:, -1]
    diagnostics = np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
==== /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
{
  "accepted": false,
  "schema_valid": true,
  "execution_status": "completed",
  "failure_kind": "scientific_or_numerical",
  "errors": [
    "public_cv_y_mae_exceeded:0.211680",
    "public_cv_traj_mae_exceeded:0.290252"
  ],
  "public_input_sha256": "23cdbab2f866001cfaf6fd1f52ba2c6a226770703fe386a00eb9af1be9307dea",
  "candidate_raw_sha256": "4b1abc4f0184ab71775404a3e713813c25cb40a4210428271a80797ae21f1241",
  "setup_source_sha256": "bfdf28220f4612128c856180267af87d656d48b328d7eca585e55573f70f0d2b",
  "cv_metrics": {
    "public_cv_y_mae": 0.21168,
    "public_cv_traj_mae": 0.290252
  },
  "dag_runtime_trace": {
    "active_node_count": 4,
    "all_declared_edges_consumed": true,
    "all_exec_count_eq_1": true,
    "dag_id": "dag_03",
    "depth": 3,
    "edge_consumption_checks": {
      "node_00": [
        "__input__"
      ],
      "node_01": [
        "__input__"
      ],
      "node_02": [
        "node_00",
        "node_01"
      ],
      "node_03": [
        "node_01",
        "node_02"
      ]
    },
    "exec_counts": {
      "node_00": 1,
      "node_01": 1,
      "node_02": 1,
      "node_03": 1
    },
    "max_fanout": 2,
    "node_code_shas": {
      "node_00": "fd29ad70706555ad7f0e5e8d531fa297cb5f6bf3d213ea395f2dc1d949816a48",
      "node_01": "493aca178d528c16ec681d5e109ba4e4500ff58bbef4f1e9c17dd6aa7ee50743",
      "node_02": "9d9cf57d412595efc361c244c11a554b623747129b330a44ce7601a11d8fcffd",
      "node_03": "eeeaeecd9a19626365176f029b400d14a57213e504550a9bef596c3f286941ff"
    },
    "non_identity_checks": {
      "node_00": true,
      "node_01": true,
      "node_02": true
    },
    "shared_fanout_nodes": [
      "node_01"
    ],
    "topo_order": [
      "node_00",
      "node_01",
      "node_02",
      "node_03"
    ]
  },
  "kernel_version": "hegel_shared_math_handoff_kernel_v3_m961_three_layer",
  "wall_ms": 2394.73
}
```

### Turn 3 — `execute_in_remote_docker` (`13443 ms` LLM)
- **Rationale**: Inspect the complete inherited calibration and compare it with channelwise response curves before fitting a replacement.

```bash
cat /tmp/hegel_continual_archive/*solver.py; cd /holdout_task; sed -n '1,115p' test_dag_candidate.py; python3 - <<'PY'
import numpy as np,pickle
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; print('telemetry',s['telemetry_vec']);
for name in ('panel_a','panel_b','panel_c'):
 p=s[name]; u=p['u']; print(name)
 for k,v in p.items():
  if k=='u': continue
  for d in range(4):
   x=u[:,d]; y=v[:,d]; o=np.argsort(x); co=np.polynomial.polynomial.polyfit(x,y,5); print(k,d,'poly',np.round(co,4),'max res',abs(np.polynomial.polynomial.polyval(x,co)-y).max(),'pairs',np.round(np.c_[x[o],y[o]][::max(1,len(x)//10)],3).tolist())
PY
if [ -d /tmp/hegel_wormhole_skills ]; then PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference-panel alignment'; fi
```
- **Container Exit Code**: `0` (`444 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
8], [0.494, -0.256], [0.635, -1.261], [0.873, 0.117], [1.161, 1.466], [1.237, 1.356]]
panel_c
v1 0 poly [ 0.2381  0.4813 -1.5325 -1.0209  1.352   0.3186] max res 0.9977601855480249 pairs [[-1.154, 0.789], [-1.004, 0.581], [-0.787, 0.082], [-0.663, -0.55], [-0.061, 0.728], [0.11, 1.194], [0.18, 0.08], [0.365, -0.812], [0.535, -0.625], [0.867, 0.729], [0.93, 0.518], [1.133, -0.055]]
v1 1 poly [ 0.2369  0.2357  0.337  -2.0344 -0.2633  1.3951] max res 1.488465750737824 pairs [[-1.219, 1.291], [-1.129, -0.357], [-1.055, 0.405], [-0.845, 1.398], [-0.316, -0.773], [-0.029, 0.641], [0.103, -0.427], [0.255, -0.002], [0.481, 0.454], [0.628, 0.592], [0.941, -0.921], [1.068, -0.807]]
v1 2 poly [-0.2271 -1.2258 -0.7802  4.2849  1.0423 -2.5946] max res 1.2667914630765693 pairs [[-1.123, 0.767], [-0.891, 0.073], [-0.749, -0.571], [-0.476, -0.471], [-0.081, 0.074], [0.133, -0.749], [0.441, -1.147], [0.578, 0.391], [0.72, -0.362], [0.988, 0.388], [1.143, -0.024], [1.199, 0.044]]
v1 3 poly [ 0.1592  1.5202 -0.5301 -1.2354  0.3625  0.4898] max res 0.7530959258048967 pairs [[-1.225, -0.698], [-1.056, -0.813], [-0.86, -0.544], [-0.363, -0.121], [-0.278, -0.008], [-0.093, 0.156], [-0.027, -0.014], [0.012, -0.179], [0.107, 0.409], [0.409, 0.692], [0.881, 0.546], [1.063, 0.856]]
v2 0 poly [-0.1528 -0.3366 -0.0021  0.504  -0.2164 -0.2133] max res 0.6920255574167973 pairs [[-1.154, -0.577], [-1.004, -0.091], [-0.787, -0.681], [-0.663, -0.199], [-0.061, -0.16], [0.11, -0.357], [0.18, -0.127], [0.365, -0.949], [0.535, -0.236], [0.867, -0.273], [0.93, -0.476], [1.133, -0.672]]
v2 1 poly [-0.0167 -0.5456 -0.0697  0.5947 -0.087  -0.4002] max res 0.5445282461160104 pairs [[-1.219, -0.193], [-1.129, 0.412], [-1.055, 0.401], [-0.845, -0.142], [-0.316, 0.377], [-0.029, -0.069], [0.103, 0.126], [0.255, -0.032], [0.481, -0.27], [0.628, -0.493], [0.941, -0.322], [1.068, -0.311]]
v2 2 poly [ 0.0513 -0.0757 -0.5426  0.3771  0.4798 -0.4363] max res 0.48949151129653445 pairs [[-1.123, 0.816], [-0.891, 0.149], [-0.749, -0.271], [-0.476, 0.143], [-0.081, 0.347], [0.133, -0.227], [0.441, -0.185], [0.578, 0.414], [0.72, -0.247], [0.988, -0.326], [1.143, -0.267], [1.199, -0.525]]
v2 3 poly [-0.1981 -0.5115  0.5806  2.903  -0.1188 -1.7575] max res 0.892834498973711 pairs [[-1.225, 0.482], [-1.056, -0.305], [-0.86, -0.497], [-0.363, -0.091], [-0.278, 0.543], [-0.093, 0.175], [-0.027, 0.422], [0.012, -0.135], [0.107, -0.446], [0.409, -0.634], [0.881, 0.745], [1.063, 0.896]]
No matching skill cards found.
```

### Turn 4 — `execute_in_remote_docker` (`21760 ms` LLM)
- **Rationale**: Compare robust and ordinary panel fits, and check held-out panel residuals before selecting the calibration family.
- **Alternatives Weighed**: `Increase the inherited channelwise scale` (rejected: The reported response scatter does not distinguish gross outliers from cross-channel effects; robust residuals can separate them.)

```bash
cd /holdout_task; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import numpy as np,pickle,sys,glob,runpy
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; a=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); print('telemetry',s['telemetry_vec']); rng=np.random.default_rng(44)
def f(x,d):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(4),k)])
for name in ('panel_a','panel_b','panel_c'):
 p=s[name]; u=p['u']; print(name)
 for k,v in p.items():
  if k=='u':continue
  print(k,'std',v.std(0)); print('affine co',np.round(np.linalg.lstsq(f(u,1),v,rcond=None)[0],4));
  for d in (1,2):
   F=f(u,d); R=v-F@np.linalg.lstsq(F,v,rcond=None)[0]; H=np.sum(np.linalg.qr(F)[0]**2,axis=1); print(d,'LOO rms',np.sqrt(np.mean((R/(1-H[:,None]))**2)), 'rms',np.sqrt(np.mean(R**2)))
  if name=='panel_a':
   F=f(u,1); best=None
   for _ in range(2500):
    i=rng.choice(len(u),5,False); B=np.linalg.lstsq(F[i],v[i],rcond=None)[0]; r=np.linalg.norm(F@B-v,axis=1); score=np.median(r)
    if best is None or score<best[0]:best=(score,B,r)
   print('robust affine',best[0],np.round(best[1],5),'sorted residuals',np.round(np.sort(best[2]),5)); print('u/v first 15',np.round(np.c_[u,v][:15],4)); print('inherited',a['_align'](p));
 if name=='panel_c':
  w=a['_panel_maps'](p,u); print('inherited rms',np.sqrt(np.mean((w-np.c_[p['v1'],p['v2']])**2,axis=0)))
PY
```
- **Container Exit Code**: `0` (`731 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.51685  0.53198  0.53503  0.55486  0.60747  0.63672  0.66577  0.69467
  0.69488  0.70555  0.72801  0.73746  0.82297 12.06091 12.30802 12.5047 ]
u/v first 15 [[-0.0929 -0.8209 -0.494   0.92    0.2094 -1.0528 -0.5936  1.4572]
 [ 1.101  -0.5007  0.8866 -1.1616  1.0158 -0.3752  1.0744 -1.3795]
 [ 0.1468  0.5896 -0.6327 -0.0921  0.544   0.7956 -0.9537 -0.5656]
 [ 0.9985  0.8427 -0.2381 -0.7017  1.3566  1.2838 -0.5298 -1.3928]
 [ 1.0744 -1.0867 -0.5146  0.3373  1.5624 -1.2673 -0.8621  0.5559]
 [-0.5123 -0.408   1.2008  1.1281  5.8182 -6.3245  8.1025 -4.3155]
 [ 0.0539  1.0621  0.8302  1.2413 -0.1879  1.1124  1.118   1.378 ]
 [-0.8908 -0.7115 -0.0171  0.755  -0.9926 -1.0039  0.2801  1.2717]
 [ 0.9061  0.082   0.1222 -1.0317  0.9967  0.3698 -0.0475 -1.5816]
 [-0.6637  1.2736 -0.3978  0.1137 -0.5058  1.2641 -0.2157  0.0366]
 [ 1.0425  0.4506 -0.3341 -0.7362  1.4582  0.8984 -0.639  -1.4315]
 [-0.8355  1.1171 -0.9607 -0.5715 -0.7115  1.5263 -0.9552 -1.0766]
 [-1.1131  0.7926 -0.446   0.0108 -0.9508  0.9109 -0.4557 -0.3953]
 [ 0.6072  0.3784  0.5795 -0.1597  0.4583  0.6278  0.5286 -0.6036]
 [-0.5843 -0.3138 -0.2034  0.7394 -0.5302 -0.601   0.0524  1.1535]]
inherited (array([ 0.03329458,  0.10820992,  0.01356679, -0.1101529 ]), array([1.17116242, 1.14811738, 1.34515439, 1.43597866]))
panel_b
v std [0.67531984 0.69754171 0.6390219  0.74338759]
affine co [[-0.1433  0.0805  0.086  -0.0722]
 [-0.2616 -0.4019  0.6715 -0.3058]
 [-0.2376 -0.4588 -0.0433  0.7111]
 [-0.4993  0.6168  0.2879  0.2486]
 [ 0.6326  0.1482  0.4887  0.3368]]
1 LOO rms 0.00012577086983881478 rms 9.748604712732751e-05
2 LOO rms 0.00018495533634153806 rms 5.7673341429557066e-05
panel_c
v1 std [0.64331816 0.74185063 0.68695808 0.62963862]
affine co [[ 0.1021  0.0561  0.0047  0.0188]
 [ 0.0032 -0.1823 -0.8519  0.2746]
 [ 0.8076 -0.1916 -0.1157 -0.2507]
 [ 0.2839  0.8252 -0.1072  0.1267]
 [ 0.1699 -0.1942  0.2961  0.7696]]
1 LOO rms 0.13523529985952745 rms 0.10739386415008391
2 LOO rms 0.25016449074571523 rms 0.07160677080076625
v2 std [0.27706892 0.3963185  0.28600925 0.49484997]
affine co [[-0.3065 -0.0414  0.0419 -0.0345]
 [-0.1338  0.0097 -0.2833 -0.3552]
 [-0.083  -0.3457  0.2151 -0.1441]
 [-0.0014 -0.269  -0.2809  0.3227]
 [-0.4038  0.1048  0.0216  0.1724]]
1 LOO rms 0.1317585633841168 rms 0.10046270406920098
2 LOO rms 0.014626692618395587 rms 0.0033063066052070166
inherited rms [7.69486628e-04 5.37261795e-04 7.60489246e-04 8.38299017e-04
 5.21383440e-05 7.84669029e-05 1.04556344e-04 6.20703617e-05]
```

### Turn 5 — `execute_in_remote_docker` (`29129 ms` LLM)
- **Rationale**: Compare forward-law residuals and inverse-map CV together; a flexible forward fit can interpolate calibration points while producing a poor inverse.
- **ATMS NOGOOD DECLARED**: Refuted `Panel A is adequately described by independent channelwise affine fits, with poor residuals attributable to gross outliers alone.` | Witness: `Robust multivariate affine calibration still has median row residual 0.292, with residuals up to 0.823 outside three isolated rows above 12; the inherited panel C maps already fit at sub-milliscale RMS.` | Blanket Diagnosis: `The panel A calibration misses structured signal; downstream target fitting and the panel C maps need not be replaced to isolate it.` | Sublated Pivot: `Reject the gross panel A outliers and compare nested multivariate polynomial and nonlinear response bases using held-out calibration residuals and frozen-kernel CV.`

```bash
cat >/tmp/probe.py <<'PY'
import pickle, runpy, glob
import numpy as np
from itertools import combinations_with_replacement as cwr
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); lin=a['lin']; one=a['one']; p=s['panel_a']; U=p['u']; V=p['v']; mask=np.max(abs(V),axis=1)<4; u=U[mask]; v=V[mask]; x=s['train']['x']; y=np.column_stack([s['train']['y'],s['train']['traj']]); c=s['panel_c'];
def score(z, split=52):
 w=a['_panel_maps'](c,z); w=(w-w[:split].mean(0))/np.maximum(w[:split].std(0),.1); g=w@w.T/w.shape[1]; d=g.diagonal(); dist=np.maximum(d[:,None]+d-2*g,0); q=(1+.25*g)**2; m=np.arange(len(x))<split; p=np.mean([k[~m][:,m]@np.linalg.solve(k[m][:,m]+r*np.eye(m.sum()),y[m]) for k,r in [(q,3e-5),(q+np.exp(-.06*dist),3e-4)]],0); e=abs(p-y[~m]).mean(0); return np.round([e[0],e[1:].mean()],6)
def fit(name,f,init=None):
 if init is not None:
  q=least_squares(lambda q:(f(u,q)@lin(f(u,q),v)-v).ravel(),init,max_nfev=200).x; F=lambda z:f(z,q)
 else:q=[]; F=f
 A=F(u); C=lin(A,v); r=A@C-v; h=np.sum(np.linalg.qr(A)[0]**2,axis=1); rm=lambda e:np.sqrt(np.mean(e**2)); print(name,'q',np.round(q,5),'rms',rm(r),'loo',rm(r/np.maximum(1-h[:,None],1e-8)),'max',abs(r).max());
 if name in ('poly2','add3','poly3','affine','trig','tanh','rational','soft','add23'): print('co',np.round(C,4).T)
 base=lin(one(u),v); z0=(x-base[0])@np.linalg.inv(base[1:]); sol=least_squares(lambda z:(F(z.reshape(-1,4))@C-x).ravel(),z0.ravel(),max_nfev=35).x.reshape(-1,4); print('  CV',score(sol),'Fwd RMS',rm(F(sol)@C-x),'zmax',abs(sol).max()); return F,C
if __name__=='__main__':
 print('row mask',np.where(~mask)[0]); print('raw',score(x)); o,t=a['_align'](p); print('diagonal',score((x-o)/t));
 fit('affine',one)
 for d in (2,3):fit('poly'+str(d),lambda z:np.column_stack([np.ones(len(z))]+[np.prod(z[:,j],axis=1) for k in range(1,d+1) for j in cwr(range(4),k)]))
 fit('add3',lambda z:np.c_[one(z),z**3]); fit('add23',lambda z:np.c_[one(z),z**2,z**3]);
 for name,g in [('trig',np.sin),('tanh',np.tanh),('rational',lambda z:z/(1+z*z)),('soft',lambda z:z/(1+abs(z)))]:fit(name,lambda z,q:np.c_[one(z),g(z*q[0])],[1.4]);
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe.py
```
- **Container Exit Code**: `0` (`2343 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rms 0.0035482623148068407 loo 0.004483420685152309 max 0.010392410090288795
co [[ 3.4300e-02  1.1081e+00  1.4000e-03 -1.2500e-02 -1.3000e-03  6.8100e-02
  -2.6000e-03 -3.1180e-01 -7.0000e-04]
 [ 7.5400e-02  2.0000e-03  9.9340e-01  1.0000e-03 -7.5000e-03 -2.1000e-03
   6.4800e-02 -1.0000e-03 -3.1570e-01]
 [ 3.4800e-02 -1.1800e-02  2.0000e-04  1.2843e+00  1.6000e-03 -3.1340e-01
  -1.0000e-03  6.8900e-02 -3.2000e-03]
 [-1.0940e-01  2.4000e-03 -9.8000e-03  7.0000e-04  1.3339e+00 -2.1000e-03
  -3.1440e-01 -5.0000e-04  6.2800e-02]]
  CV [0.013836 0.012662] Fwd RMS 1.733934643297791e-13 zmax 1.1405967129844614
tanh q [1.10074] rms 0.0038281609427565566 loo 0.004872764315062135 max 0.010796384657594693
co [[ 0.0339  0.8185  0.0141  1.2835  0.0026  0.4176 -0.0182 -1.8753 -0.006 ]
 [ 0.0752  0.0082  0.7264  0.0119  1.3071 -0.0098  0.3863 -0.0149 -1.9037]
 [ 0.0351  1.3008  0.0055  0.9982  0.0155 -1.8968 -0.0079  0.4137 -0.0205]
 [-0.1093  0.0147  1.2941 -0.0086  1.0727 -0.0168 -1.8878  0.0108  0.3792]]
  CV [0.014048 0.015489] Fwd RMS 1.5548710090517734e-13 zmax 1.1398420399006146
rational q [0.69054] rms 0.0038071083619505334 loo 0.004843562730351346 max 0.010779359965852775
co [[ 3.3900e-02  8.9050e-01  1.0900e-02  9.6100e-01  1.6000e-03  5.6120e-01
  -2.4300e-02 -2.5218e+00 -8.0000e-03]
 [ 7.5200e-02  6.6000e-03  7.9270e-01  9.2000e-03  9.7980e-01 -1.3400e-02
   5.1960e-01 -1.9600e-02 -2.5599e+00]
 [ 3.5100e-02  9.7440e-01  4.2000e-03  1.0693e+00  1.1900e-02 -2.5502e+00
  -1.0700e-02  5.5640e-01 -2.7500e-02]
 [-1.0930e-01  1.1800e-02  9.6960e-01 -6.4000e-03  1.1379e+00 -2.2400e-02
  -2.5389e+00  1.3800e-02  5.0980e-01]]
  CV [0.014046 0.015372] Fwd RMS 1.5175874549851265e-13 zmax 1.1398367688138036
soft q [9.e-05] rms 0.007948455395309898 loo 0.009987142610847745 max 0.019385809652586072
co [[ 3.27000000e-02 -1.93332560e+03  9.20466000e+01  8.54846770e+03
   8.36306000e+01  2.27029605e+07 -1.08027616e+06 -1.00327041e+08
  -9.81513719e+05]
 [ 7.54000000e-02  3.67877000e+01 -1.68191790e+03  2.01283500e+02
   8.80311530e+03 -4.31751098e+05  1.97510790e+07 -2.36224489e+06
  -1.03315591e+08]
 [ 3.68000000e-02  8.71937850e+03  8.90817000e+01 -1.92067090e+03
   8.63184000e+01 -1.02332841e+08 -1.04546853e+06  2.25565047e+07
  -1.01307140e+06]
 [-1.08800000e-01  1.49327000e+02  8.62274660e+03 -1.77078900e+02
  -1.75744920e+03 -1.75246952e+06 -1.01198772e+08  2.07818448e+06
   2.06415344e+07]]
  CV [0.019831 0.025348] Fwd RMS 1.2203107394600074e-11 zmax 1.1426359921278282
```

### Turn 6 — `execute_in_remote_docker` (`28623 ms` LLM)
- **Rationale**: The harmonic model improves calibration and downstream prediction without changing panel C or any frozen node. Run the isolated public evaluator on the saved artifact before promoting it.
- **Alternatives Weighed**: `Cubic polynomial calibration` (rejected: The full cubic has roughly ten times the harmonic model's held-out panel RMS, while the additive cubic split trajectory MAE exceeds 0.038.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
src = next(Path('/tmp/hegel_continual_archive').glob('*solver.py')).read_text()
head = src[:src.index('def _align')]
maps = src[src.index('def _panel_maps'):src.index('def _time_basis')]
implementation = '''def _calibrate(panel, measured):
    """Fit the panel's forward response, then invert it row by row."""
    u = np.asarray(panel['u'], dtype=float)
    v = np.asarray(panel['v'], dtype=float)
    # Grossly corrupted rows have large joint residuals even to an affine fit.
    affine = one(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(4):
        beta = lin(affine[keep], v[keep])
        error = np.linalg.norm(affine @ beta - v, axis=1)
        med = np.median(error)
        spread = 1.4826 * np.median(np.abs(error - med))
        good = error <= med + 6 * max(spread, .01)
        if good.sum() < 2 * u.shape[1] + 2 or np.array_equal(good, keep):
            break
        keep = good
    u, v = u[keep], v[keep]

    def design(z, frequency):
        return np.c_[one(z), np.sin(frequency * z)]

    def residual(freq):
        f = design(u, freq[0])
        return (f @ lin(f, v) - v).ravel()

    frequency = least_squares(residual, [2.2], bounds=([1.2], [3.5]),
                              gtol=1e-12).x[0]
    intercept, *weights = lin(design(u, frequency), v)
    linear, harmonic = np.split(np.asarray(weights), 2, axis=0)
    z = np.linalg.solve(linear.T, (measured - intercept).T).T
    for _ in range(20):
        angle = frequency * z
        error = z @ linear + np.sin(angle) @ harmonic + intercept - measured
        if np.max(np.abs(error), initial=0.) < 1e-11:
            break
        jacobian = linear + (frequency * np.cos(angle))[:, :, None] * harmonic
        step = np.linalg.solve(jacobian.transpose(0, 2, 1), error[..., None])[..., 0]
        z -= step / np.maximum(1., np.max(np.abs(step), axis=1, keepdims=True) / .35)
    return z


'''
run = '''def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train_x = np.asarray(study['train']['x'], dtype=float)
    measured = np.vstack((train_x, study['test']['x']))
    z = _calibrate(study['panel_a'], measured)
    w = _panel_maps(study['panel_c'], z)
    n = len(train_x)
    features = (w - w[:n].mean(axis=0)) / np.maximum(w[:n].std(axis=0), .1)
    return {'features': features, 'n_train': np.asarray(n, dtype=np.intp)}
'''
Path('node_00.py').write_text(head + implementation + maps + run)
PY
sed -n '1,230p' test_dag_candidate.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`471 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
t_full, ec = run_dag(st)
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
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.013836, "public_cv_traj_mae": 0.010472}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1, "node_03": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=93897a87e1d1217a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_93897a87 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`27421 ms` LLM)
- **Rationale**: Keep the accepted artifact unchanged and test whether its margin holds across disjoint folds, high-amplitude inputs, and every leave-one-calibration-row-out split.
- **Alternatives Weighed**: `Submit on the single public split` (rejected: It does not test sensitivity to individual calibration rows or prediction at the edge of the measured input range.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I test_dag_candidate.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy, pickle, pathlib
import numpy as np
source = pathlib.Path('test_dag_candidate.py').read_text()
ns = {}
exec(compile(source.split('\nst = pickle.load(', 1)[0], 'test_dag_candidate.py', 'exec'), ns)
s = pickle.load(open('public_batch.pkl', 'rb'))[0]['study']
t = s['train']; x = t['x']; n = len(x); ids = np.arange(n)
def evaluate(label, test, omit=None):
    train = np.setdiff1d(ids, test)
    st = copy.deepcopy(s)
    st['train'] = {k:t[k][train] for k in ('x','y','traj')}
    y, tr = st['train']['y'], st['train']['traj']
    st['train']['summary'] = np.array([y.mean(),y.std(),tr[:,-1].mean(),tr[:,-1].std()])
    st['test'] = {'x':x[test]}
    if omit is not None:
        st['panel_a'] = {k:np.delete(v, omit, axis=0) for k,v in st['panel_a'].items()}
    out, counts = ns['run_dag'](st)
    ye = np.abs(out['y_hat'] - t['y'][test]); te = np.abs(out['state_pred'] - t['traj'][test]); d = out['diagnostics_vec']
    pred = out['state_pred'][:,-1]; yh = out['y_hat']
    assert all(c == 1 for c in counts.values()) and np.all(np.isfinite(ye)) and np.all(np.isfinite(te))
    assert np.allclose(d, [yh.mean(),yh.std(),pred.mean(),pred.std()], atol=1e-3)
    return np.array([ye.mean(),te.mean(),te.mean(axis=0).max()])
folds = np.array([evaluate(i, idx) for i,idx in enumerate(np.array_split(np.random.default_rng(871).permutation(n),4))])
edge = evaluate('high-amplitude', np.argsort(np.max(abs(x),axis=1))[-14:])
jack = np.array([evaluate(i, ids[-20:], i) for i in range(len(s['panel_a']['u']))])
print('MAE columns: y / trajectory / worst horizon')
print('Worst of four disjoint validation folds:', folds.max(axis=0))
print('Highest-amplitude validation cases:', edge)
print('Worst leave-one-calibration-row-out:', jack.max(axis=0))
assert np.max(folds[:,:2]) <= .038 and np.max(edge[:2]) <= .038 and np.max(jack[:,:2]) <= .038
PY
```
- **Container Exit Code**: `0` (`1141 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.013836, "public_cv_traj_mae": 0.010472}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1, "node_03": 1}}
MAE columns: y / trajectory / worst horizon
Worst of four disjoint validation folds: [0.02954275 0.02094631 0.02377075]
Highest-amplitude validation cases: [0.01804927 0.00902238 0.01328696]
Worst leave-one-calibration-row-out: [0.01405047 0.01074702 0.01315928]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=93897a87e1d1217a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_93897a87 (directory).
```

### Turn 8 — `submit_completed_task` (`11403 ms` LLM)
- **Rationale**: The same archived artifact passed the isolated public evaluator, four disjoint validation folds, the high-amplitude slice, and every leave-one-calibration-row-out split. Each DAG node executed exactly once.
- **Submitted Final Answer / Summary**: `Updated and verified `/holdout_task/node_00.py`; all frozen nodes were unchanged. The public evaluator accepted the DAG with y MAE 0.013836 and trajectory MAE 0.010472. All stress slices stayed below 0.038.`

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
