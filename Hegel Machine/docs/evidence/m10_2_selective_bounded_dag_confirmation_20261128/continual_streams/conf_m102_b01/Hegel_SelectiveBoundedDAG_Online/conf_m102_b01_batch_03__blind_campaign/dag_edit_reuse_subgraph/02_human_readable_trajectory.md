# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m102_b01_batch_03__blind_campaign__reuse_subgraph`

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
| **Turns Used / Max Turns** | `7 / 12` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `171,663` (`166,903` / `74,368` / `4,760`) |
| **Total Episode Wall Time (s)** | `163.38s` |

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

### Turn 1 — `execute_in_remote_docker` (`14368 ms` LLM)
- **Rationale**: Inspect the frozen interface, available measurements, archived implementations, and validator before selecting a calibration model.

```bash
cd /holdout_task && cat _dag_spec.json _frozen_nodes/node_01.py test_dag_candidate.py public_asset_index.json; echo '--- archive ---'; find /tmp/hegel_continual_archive -maxdepth 4 -type f | sort | head -160; echo '--- batch ---'; python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
def show(x,p='root',n=0):
 if isinstance(x,dict):
  for k,v in x.items(): show(v,p+'.'+str(k),n+1)
 elif isinstance(x,np.ndarray): print(p,x.shape,x.dtype,x.reshape(-1)[:12])
 elif isinstance(x,(list,tuple)) and n<2:
  print(p,type(x),len(x));
  for i,v in enumerate(x[:2]): show(v,p+'.'+str(i),n+1)
 else: print(p,type(x),str(x)[:180])
show(x)
PY
```
- **Container Exit Code**: `0` (`384 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rchive/archive_manifest.json
/tmp/hegel_continual_archive/conf_m102_b01_batch_00_solver.py
/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
--- batch ---
root <class 'list'> 1
root.0.study.eval_times (8,) float64 [0.25       0.64285714 1.03571429 1.42857143 1.82142857 2.21428571
 2.60714286 3.        ]
root.0.study.panel_a.u (48, 4) float64 [ 0.57081864  0.37583775 -0.04671603 -1.26748563  0.40372932  0.53041923
  0.64697231 -0.67231861  0.34624324  0.6029179   0.73016946 -0.26110571]
root.0.study.panel_a.v (48, 4) float64 [ 0.83644754  0.3673593  -0.41743554 -1.70551856  0.70939113  0.87955893
  0.26398951 -0.73971712  0.67293713  1.02316766  0.56516242 -0.24190528]
root.0.study.panel_b.u (24, 4) float64 [ 0.85873314  0.06970115 -0.44567881 -1.07250634  0.37588779  1.12294693
 -1.19311536  0.08012066 -0.90387807 -0.42731599 -0.45839712  0.32714168]
root.0.study.panel_b.v (24, 4) float64 [ 0.21183364  0.52149501 -1.19844286 -0.1460647   0.14056391  1.17538618
  0.08802521 -0.7884907   0.75157361 -0.0835272   0.58133996  0.39174815]
root.0.study.panel_c.u (24, 4) float64 [ 0.89744404  0.7476798   1.03932283 -1.1679412  -0.39066877  0.95261148
  0.07883634 -0.52002078  1.22556916 -0.23112058 -1.00777482 -0.39719348]
root.0.study.panel_c.v1 (24, 4) float64 [-1.46820389 -0.73585618  0.00665529 -0.56681211 -0.88575383 -0.51579196
 -0.32441003  0.85894564  0.61627708 -1.04852913  0.33712197 -0.44403915]
root.0.study.panel_c.v2 (24, 4) float64 [-1.01574589 -0.06810866 -0.37503195 -0.09397149 -0.02080606 -0.2238138
 -0.55934719 -0.36610002 -0.32968051 -0.24854116  0.13803377  0.73411387]
root.0.study.train.x (72, 4) float64 [ 0.81998332  1.63778674  0.16253597 -0.53281403  1.47692159  0.40520222
 -0.04820001 -0.35009362  0.49623959 -1.08036355  0.09836754 -0.57025863]
root.0.study.train.y (72,) float64 [-1.00848293 -1.41358789 -1.12942412  1.31939833  0.41193543 -0.60403231
  0.15606009  0.36686962  0.89987894  0.97809289  1.07674487  1.12848335]
root.0.study.train.traj (72, 8) float64 [-0.49095045 -0.90288972 -1.08264843 -1.16201104 -1.18269522 -1.17697175
 -1.16929921 -1.15464159 -0.336199   -0.60155251 -0.71201026 -0.73636163]
root.0.study.train.summary (4,) float64 [0.0484676  0.86063571 0.26714155 1.28290099]
root.0.study.test.x (36, 4) float64 [ 0.24147795  0.5248694  -0.01103468  0.31640603 -0.8169361  -0.58147572
  0.92227758  0.04297618 -0.84989374 -0.50348078 -0.13108044  0.26039972]
root.0.study.telemetry_vec (4,) float64 [1.25 0.22 0.22 0.22]
```

### Turn 2 — `execute_in_remote_docker` (`13995 ms` LLM)
- **Rationale**: Recover the archived calibration basis and the exact validation split before testing which algebraic terms changed.

```bash
wc -l /tmp/hegel_continual_archive/*; cat /tmp/hegel_continual_archive/archive_manifest.json; cat /tmp/hegel_continual_archive/conf_m102_b01_batch_00_solver.py; sed -n '100,240p' /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`282 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`26524 ms` LLM)
- **Rationale**: Compare per-coordinate and joint calibration residuals before fitting the inverse, keeping prediction labels out of model identification.

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel calibration robust alignment' 2>&1 | head -25; python3 - <<'PY'
import numpy as np,pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
for key in ('panel_a','panel_b','panel_c'):
 p=s[key];u=p['u'];A=np.c_[np.ones(len(u)),u]
 for k,v in p.items():
  if k=='u':continue
  q=ls(A,v); R=v-A@q
  print(key,k,'range u,v',np.stack([u.min(0),u.max(0),v.min(0),v.max(0)]),'affine',q,'err rms',np.sqrt(np.mean(R**2,0)))
  for j in range(u.shape[1]):
   t=u[:,j];z=v[:,j];ix=np.argsort(t);X=np.column_stack([t**d for d in range(6)]);errs=[]
   for d in (1,2,3,4,5):
    D=X[:,:d+1]; qq=ls(D,z)
    for _ in range(12):
     keep=np.argsort(abs(D@qq-z))[:int(.875*len(t))];qq=ls(D[keep],z[keep]);
    errs.append(np.quantile(abs(D@qq-z),[.5,.85,1]))
   print(j,'trimmed residual q50/85/max by degree',np.array(errs).T)
   if key=='panel_a':print(np.c_[t[ix],z[ix]][::2].T)
PY
```
- **Container Exit Code**: `0` (`460 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
421 0.32299 0.29421 0.31736 0.32137]
 [0.58071 0.51857 0.51336 0.47584 0.4615 ]
 [1.11972 1.30105 1.2305  1.30421 1.28513]]
3 trimmed residual q50/85/max by degree [[0.41581 0.45329 0.44725 0.45236 0.45611]
 [0.93876 0.78513 0.82983 0.74367 0.75635]
 [1.22154 1.26241 1.03321 1.31432 1.40888]]
panel_c v1 range u,v [[-1.00961 -1.10883 -1.00777 -1.16794]
 [ 1.22557  1.10852  1.14579  1.18106]
 [-1.4682  -1.19395 -1.11606 -1.3747 ]
 [ 0.86678  1.23174  1.15727  1.14131]] affine [[-0.08439 -0.02805 -0.13612  0.11936]
 [-0.03042 -0.26717  0.55435 -0.74824]
 [-0.64287 -0.16097  0.46743  0.4826 ]
 [-0.69436  0.47902 -0.34808 -0.43364]
 [ 0.19409  0.78571  0.43589  0.00626]] err rms [0.09763 0.07419 0.13699 0.10045]
0 trimmed residual q50/85/max by degree [[0.52529 0.51372 0.48348 0.47539 0.39594]
 [0.80583 0.80128 0.78667 0.78092 0.74969]
 [1.68328 1.69718 1.71365 1.769   1.4048 ]]
1 trimmed residual q50/85/max by degree [[0.38083 0.38273 0.42318 0.37396 0.36436]
 [0.9077  0.90458 0.84719 0.82267 0.87054]
 [1.82893 1.82932 1.70117 1.6369  1.64375]]
2 trimmed residual q50/85/max by degree [[0.35848 0.37489 0.26969 0.28065 0.27683]
 [0.79046 0.79211 0.74919 0.61732 0.62635]
 [1.35685 1.35043 1.60942 1.76583 1.77571]]
3 trimmed residual q50/85/max by degree [[0.60735 0.64454 0.64006 0.5587  0.60487]
 [1.03225 1.03886 0.94601 0.96552 0.9165 ]
 [1.40702 1.6334  1.42475 1.8036  1.79469]]
panel_c v2 range u,v [[-1.00961 -1.10883 -1.00777 -1.16794]
 [ 1.22557  1.10852  1.14579  1.18106]
 [-1.1215  -0.30595 -0.85427 -0.48791]
 [ 0.28329  0.82486  0.34941  0.73411]] affine [[-0.07997  0.13174 -0.09273 -0.05532]
 [-0.32481 -0.014   -0.00695  0.39369]
 [ 0.00166 -0.06362 -0.47865 -0.05282]
 [-0.39005  0.19379  0.01284 -0.22626]
 [ 0.12847  0.4224  -0.07956  0.10662]] err rms [0.08301 0.09186 0.09187 0.07063]
0 trimmed residual q50/85/max by degree [[0.13545 0.15029 0.13637 0.13924 0.1256 ]
 [0.2823  0.24784 0.24229 0.24235 0.25234]
 [0.80061 0.76538 0.78598 0.78391 0.73979]]
1 trimmed residual q50/85/max by degree [[0.1507  0.15104 0.1513  0.12337 0.13517]
 [0.38851 0.38857 0.38749 0.35319 0.33695]
 [0.92642 0.92607 0.92582 1.00119 1.01783]]
2 trimmed residual q50/85/max by degree [[0.27333 0.2687  0.26569 0.20178 0.21641]
 [0.55799 0.49797 0.49648 0.51639 0.53154]
 [0.87249 0.81778 0.81678 0.87766 0.86214]]
3 trimmed residual q50/85/max by degree [[0.19516 0.23063 0.22281 0.21383 0.21599]
 [0.4291  0.4015  0.37942 0.38509 0.41212]
 [0.85069 0.86591 0.85368 0.85861 0.64103]]
```

### Turn 4 — `execute_in_remote_docker` (`21798 ms` LLM)
- **Rationale**: Separate gross corruption from cross-coordinate structure, then compare affine and interaction terms on held-out calibration rows before implementing the inverse.
- **Alternatives Weighed**: `Invert independent high-degree polynomials` (rejected: Increasing the degree in each coordinate did not materially reduce the trimmed panel residuals.)

```bash
cat > /tmp/panel_probe.py <<'PY'
import itertools as it,pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_a']['u'];v=s['panel_a']['v']; np.set_printoptions(precision=5,suppress=True,linewidth=145)
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
keep=np.linalg.norm(v,axis=1)<4
print('excluded',np.where(~keep)[0]);print('excluded values',v[~keep]); print('retained',sum(keep))
U=u[keep]; V=v[keep]
a=np.c_[np.ones(len(U)),U]; q=ls(a,V); r=V-a@q;print('joint affine',q,'rms',np.sqrt(np.mean(r*r,axis=0)))
for j in range(4):
 print('output',j)
 for i in range(4):
  for k in range(i,4):
   t=U[:,i]*U[:,k]; d=t-a@ls(a,t); co=d@r[:,j]/(d@d); rms=np.std(r[:,j]-co*d)
   if rms<.97*np.std(r[:,j]): print('pair',i,k,co,'rms',rms)
 for p in (2,3,4):
  powers=[inds for k in range(2,p+1) for inds in it.combinations_with_replacement(range(4),k)]
  d=np.c_[a,*[U[:,inds].prod(1) for inds in powers]]; q=ls(d,V[:,j]); res=d@q-V[:,j]; lev=np.linalg.norm(np.linalg.qr(d,mode='reduced')[0],axis=1)**2
  print('degree',p,'rms',np.std(res),'LOO MAE',np.mean(abs(res/np.maximum(1-lev,1e-12))),'strong terms',[(inds,round(float(z),5)) for inds,z in zip([(),*( (k,) for k in range(4)),*powers],q) if abs(z)>.024])
 print('correlations residual',np.corrcoef(np.c_[r[:,j],U**3,np.sin(1.25*U), U/(1+1.25*U**2)].T)[0,1:])
PY
python3 /tmp/panel_probe.py
```
- **Container Exit Code**: `0` (`419 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, 1, 3), 0.02674), ((0, 2, 2), -0.04879), ((0, 2, 3), -0.03887), ((0, 3, 3), -0.04179), ((1, 1, 2), 0.10086), ((1, 1, 3), 0.03272), ((1, 3, 3), 0.04188), ((2, 2, 2), 0.20719), ((2, 2, 3), 0.08527), ((2, 3, 3), 0.10022), ((3, 3, 3), 0.04677), ((0, 0, 0, 0), 0.03129), ((0, 0, 0, 2), -0.02733), ((0, 0, 1, 1), -0.08114), ((0, 0, 1, 2), -0.05638), ((0, 0, 2, 2), -0.08411), ((0, 0, 2, 3), 0.03267), ((0, 1, 1, 1), 0.02618), ((0, 1, 3, 3), 0.02934), ((0, 2, 2, 3), -0.0272), ((0, 2, 3, 3), -0.03547), ((1, 1, 2, 3), 0.04958), ((1, 1, 3, 3), 0.02563), ((1, 2, 2, 2), 0.07848), ((1, 2, 3, 3), -0.03281), ((2, 2, 2, 3), 0.04626), ((2, 2, 3, 3), 0.08121), ((2, 3, 3, 3), 0.06244), ((3, 3, 3, 3), 0.03369)]
correlations residual [ 0.15671 -0.02342  0.09824 -0.12438 -0.0422   0.0091  -0.0273   0.03735 -0.09392  0.04691 -0.06867  0.07635]
output 3
pair 0 2 -0.12074786097097628 rms 0.14840959266876094
pair 0 3 0.330585778467846 rms 0.05391986332939591
pair 1 3 0.12335924732702594 rms 0.14957263498675066
degree 2 rms 0.04034714944367792 LOO MAE 0.05130805158872295 strong terms [((), -0.09245), ((0,), 0.30493), ((3,), 1.1438), ((0, 3), 0.33452), ((1, 1), 0.03146), ((1, 3), 0.06197), ((2, 2), 0.03251), ((2, 3), -0.03479)]
degree 3 rms 5.244059015990018e-05 LOO MAE 0.0002961955966455464 strong terms [((), -0.05906), ((0,), 0.32029), ((3,), 1.00296), ((0, 3), 0.32003), ((3, 3, 3), 0.16012)]
degree 4 rms 1.8468943674039817e-15 LOO MAE 0.0013365851635348412 strong terms [((), -0.02454), ((0,), 0.36612), ((2,), -0.02787), ((3,), 0.90742), ((0, 0), 0.02622), ((0, 2), -0.0828), ((0, 3), 0.23094), ((1, 1), -0.05357), ((1, 2), -0.04731), ((1, 3), 0.06125), ((2, 2), -0.04066), ((0, 0, 1), -0.04593), ((0, 0, 2), -0.02797), ((0, 0, 3), 0.03662), ((0, 1, 1), -0.04479), ((0, 1, 2), 0.04466), ((0, 2, 2), -0.06272), ((0, 2, 3), -0.02444), ((1, 1, 2), 0.02418), ((1, 1, 3), 0.06373), ((1, 2, 2), 0.04566), ((2, 2, 2), 0.04021), ((2, 2, 3), 0.04604), ((3, 3, 3), 0.19553), ((0, 0, 0, 0), 0.02505), ((0, 0, 0, 2), 0.02506), ((0, 0, 0, 3), 0.05566), ((0, 0, 1, 3), -0.03595), ((0, 0, 2, 2), -0.06143), ((0, 1, 1, 3), 0.04487), ((0, 1, 2, 2), -0.04191), ((0, 1, 2, 3), 0.07426), ((0, 2, 2, 2), 0.05231), ((1, 1, 1, 3), -0.02669), ((1, 1, 2, 2), 0.05409), ((1, 2, 2, 2), 0.03619), ((1, 3, 3, 3), -0.03038), ((2, 2, 2, 2), 0.03754), ((2, 2, 2, 3), 0.05883), ((2, 3, 3, 3), -0.05441)]
correlations residual [ 0.1365  -0.06954  0.02827  0.19858 -0.03644  0.02329 -0.00794 -0.05776 -0.07656  0.04214 -0.0208  -0.1014 ]
```

### Turn 5 — `execute_in_remote_docker` (`33497 ms` LLM)
- **Rationale**: Fit the calibration from paired panel measurements alone, then test both frozen prediction heads without using prediction labels to tune the editable node.
- **Alternatives Weighed**: `Invert an unrestricted cubic model` (rejected: The fitted coefficients identify a much smaller shared cyclic basis, which reduces estimation error and improves extrapolation.)
- **ATMS NOGOOD DECLARED**: Refuted `Panel A is a separable affine calibration, or can be repaired with separate univariate nonlinearities.` | Witness: `After removing three grossly corrupted rows, even joint affine residual RMS was 0.147–0.168; cubic fits recovered cyclic neighbor products near 0.32 and own-coordinate cubics near 0.16, with displayed LOO MAEs below 0.00030.` | Blanket Diagnosis: `Each observed channel depends on its own latent coordinate and the next coordinate in a cycle.` | Sublated Pivot: `Fit the compact cyclic forward map, invert it jointly, and preserve the archived panel B transform and frozen sink.`

```bash
cat > /holdout_task/node_00.py <<'PY'
import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit_panel(panel):
    u = np.asarray(panel['u'], dtype=float)
    v = np.asarray(panel['v'], dtype=float)
    n, d = u.shape
    adj = np.roll(u, -1, axis=1)
    eye = np.broadcast_to(np.eye(d), (n, d, d))
    design = np.concatenate((eye, eye * u[..., None], adj[..., None],
                             (u * adj)[..., None], (u ** 3)[..., None]), axis=2)

    def fit(mask):
        return np.linalg.lstsq(design[mask].reshape(-1, 2*d+3), v[mask].ravel(), rcond=None)[0]

    # Gross contamination affects entire rows. Start with a trimmed fit,
    # then recover all rows consistent with the fitted calibration.
    norm = np.linalg.norm(v, axis=1)
    keep = norm <= max(4 * np.median(norm), 1e-3)
    q = fit(keep)
    for _ in range(4):
        err = np.max(abs(design @ q - v), axis=1)
        keep = err <= max(8 * np.median(err), 0.002)
        q = fit(keep)
    return q[:d], q[d:2*d], *q[2*d:]


def _invert(x, params):
    offset, scale, neighbor, pair, cubic = params
    d = x.shape[1]
    j = np.arange(d)
    nxt = (j + 1) % d
    rhs = x - offset
    lin = np.diag(scale)
    lin[j, nxt] += neighbor
    u = np.linalg.solve(lin, rhs.T).T

    def residual(z):
        return z * scale + (neighbor + pair * z) * z[:, nxt] + cubic * z**3 - rhs

    for _ in range(32):
        r = residual(u)
        err = np.max(abs(r), axis=1)
        if np.max(err, initial=0.0) < 1e-12:
            break
        jac = np.zeros((len(u), d, d))
        jac[:, j, j] = scale + pair * u[:, nxt] + 3*cubic*u*u
        jac[:, j, nxt] += neighbor + pair * u
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        rates = np.ones(len(u))
        for _ in range(18):
            trial = u - rates[:, None] * step
            bad = (np.max(abs(residual(trial)), axis=1) >= err) & (err > 1e-12)
            if not bad.any():
                break
            rates[bad] *= 0.5
        u = np.where((~bad & (err > 1e-12))[:, None], trial, u)
    return u


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.concatenate((study['train']['x'], study['test']['x']), axis=0)
    u = _invert(x, _fit_panel(study['panel_a']))
    panel = study['panel_b']
    mapping = np.linalg.lstsq(_affine(panel['u']), panel['v'], rcond=None)[0]
    return {'w': _affine(u) @ mapping}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`463 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.007026, "public_cv_traj_mae": 0.005892}, "exec_counts": {"node_00": 1, "node_01": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c8a6598dece53342 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c8a6598d (directory).
```

### Turn 6 — `execute_in_remote_docker` (`26091 ms` LLM)
- **Rationale**: Stress-test the accepted artifact without changing it: check held-out calibration rows, inversion across the observed coordinate range, and both prediction heads on blocked and high-latent-norm slices.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
node=load('calibrator','/holdout_task/node_00.py')
sink=load('heads','/holdout_task/_frozen_nodes/node_01.py')
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a=st['panel_a']; u=a['u']; v=a['v']; n=len(u)
fit=node._fit_panel(a)
def forward(z,p):
    b,s,c,k,h=p; adj=np.roll(z,-1,axis=1)
    return b+s*z+(c+k*z)*adj+h*z**3
inlier=np.max(abs(forward(u,fit)-v),axis=1)<0.002
loo=[]; inverse=[]
for i in range(n):
    keep=np.arange(n)!=i
    p=node._fit_panel({k:z[keep] for k,z in a.items()})
    if inlier[i]:
        loo.append(np.max(abs(forward(u[i:i+1],p)-v[i:i+1])))
        inverse.append(np.max(abs(node._invert(v[i:i+1],p)-u[i:i+1])))
grid=np.stack(np.meshgrid(*[np.linspace(l,h,7) for l,h in zip(u.min(0),u.max(0))]),axis=-1).reshape(-1,u.shape[1]); roundtrip=np.max(abs(node._invert(forward(grid,fit),fit)-grid))
print('panel inliers',sum(inlier),'max held-out forward error',max(loo),'max held-out inverse error',max(inverse),'max grid round-trip error',roundtrip)
assert max(loo)<0.002 and max(inverse)<0.004 and roundtrip<1e-8
x=st['train']['x']; y=st['train']['y']; traj=st['train']['traj']; reports=[]
idx=np.arange(len(x)); calibrated=node._invert(x,fit)
splits=[(f'block{i}',ix) for i,ix in enumerate(np.array_split(idx,4))]
splits += [('large_latent_norm',np.argsort(np.linalg.norm(calibrated,axis=1))[-18:])]
for label,ix in splits:
    keep=~np.isin(idx,ix); yt=y[keep]; tr=traj[keep]
    sub={**st,'train':{'x':x[keep],'y':yt,'traj':tr,'summary':np.array([yt.mean(),yt.std(),tr[:,-1].mean(),tr[:,-1].std()])},'test':{'x':x[ix]}}
    w=node.run_node({'__input__':sub}); out=sink.run_node({'__input__':sub,'node_00':w})
    ym=np.mean(abs(out['y_hat']-y[ix])); tm=np.mean(abs(out['state_pred']-traj[ix]),axis=0)
    reports.append((ym,tm.mean(),tm.max())); print(label,'y MAE',ym,'trajectory MAE',tm.mean(),'worst horizon MAE',tm.max())
worst=np.max(reports,axis=0); print('worst y/trajectory/horizon',worst); assert np.all(worst<0.038)
PY
```
- **Container Exit Code**: `0` (`579 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
panel inliers 45 max held-out forward error 0.0003184665601460024 max held-out inverse error 0.0003740595137313729 max grid round-trip error 1.3982747251728433e-12
block0 y MAE 0.009576473736777133 trajectory MAE 0.005591088702778477 worst horizon MAE 0.007070277234718468
block1 y MAE 0.005608447373143503 trajectory MAE 0.005329046451680944 worst horizon MAE 0.007819670360288158
block2 y MAE 0.007169371415755517 trajectory MAE 0.005832756096234953 worst horizon MAE 0.00811053899790355
block3 y MAE 0.007289786784644415 trajectory MAE 0.0059287802692502695 worst horizon MAE 0.007631092614840095
large_latent_norm y MAE 0.008632967721613078 trajectory MAE 0.007604621216674882 worst horizon MAE 0.009281161270547388
worst y/trajectory/horizon [0.00957647 0.00760462 0.00928116]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c8a6598dece53342 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c8a6598d (directory).
```

### Turn 7 — `submit_completed_task` (`11127 ms` LLM)
- **Rationale**: The verified artifact passed the public DAG contract with one execution per node. Blocked and high-latent-norm slices stayed below tolerance, and the maximum held-out calibration inverse error was 0.000374.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_00.py` without changing the frozen downstream node. The public validator accepted it: scalar MAE 0.007026 and trajectory MAE 0.005892, both below 0.038.`

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
