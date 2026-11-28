# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_m102_c02_batch_09__blind_campaign__reuse_subgraph`

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
| **Turns Used / Max Turns** | `9 / 12` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `245,390` (`238,428` / `114,048` / `6,962`) |
| **Total Episode Wall Time (s)** | `208.46s` |

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

### Turn 1 — `execute_in_remote_docker` (`9047 ms` LLM)
- **Rationale**: Inspect frozen consumers and the public scoring protocol before inferring the editable transform from archived and current measurements.

```bash
cd /holdout_task && cat _dag_spec.json && for f in _frozen_nodes/*.py; do echo "==== $f"; cat "$f"; done && cat test_dag_candidate.py && ls -la /tmp/hegel_continual_archive && find /tmp/hegel_continual_archive -maxdepth 3 -type f | sort | head -150
```
- **Container Exit Code**: `0` (`297 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
t = len(st["test"]["x"]), len(st["eval_times"])
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
total 20
drwxr-xr-x 2 root root 4096 Nov 28 13:58 .
drwxrwxrwt 1 root root 4096 Nov 28 13:58 ..
-rw-r--r-- 1 root root  663 Nov 28 13:58 archive_manifest.json
-rw-r--r-- 1 root root 2950 Nov 28 13:58 dev_m102_c02_batch_00_solver.py
-rw-r--r-- 1 root root 1881 Nov 28 13:58 pre_llm_gate_diagnostic.json
/tmp/hegel_continual_archive/archive_manifest.json
/tmp/hegel_continual_archive/dev_m102_c02_batch_00_solver.py
/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
```

### Turn 2 — `execute_in_remote_docker` (`12836 ms` LLM)
- **Rationale**: Inspect the archived coordinate formula and public calibration arrays before choosing which parts of the transform need to change.

```bash
cd /holdout_task; cat /tmp/hegel_continual_archive/{archive_manifest.json,pre_llm_gate_diagnostic.json,dev_m102_c02_batch_00_solver.py}; cat public_asset_index.json; cat validate_public_schema.py; python3 - <<'PY'
import numpy as np,pickle
np.set_printoptions(precision=7,linewidth=140)
p=pickle.load(open('public_batch.pkl','rb'))
print(type(p),len(p))
def show(d,indent=''):
 for k,v in d.items():
  if isinstance(v,dict): print(indent,k); show(v,indent+'  ')
  else:
   a=np.asarray(v); print(indent,k,type(v).__name__,a.shape,a.dtype,a if a.ndim==0 or a.size<80 else a[:4])
show(p[0])
PY
```
- **Container Exit Code**: `0` (`348 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
]]
   panel_c
     u ndarray (24, 4) float64 [[ 0.2556307  0.1536091 -0.7607245 -0.8598178]
 [ 0.6997908 -0.569109  -0.3342251  1.1953858]
 [ 0.2924535 -0.436484   0.8275472 -0.9899503]
 [-0.7075808 -0.0030561  0.9358798  1.1842353]]
     v1 ndarray (24, 4) float64 [[ 0.1852742 -0.2082042  1.1360036  0.1042272]
 [ 0.552772  -0.7129869 -0.6526617  0.8632637]
 [-0.3578965  1.1289938  0.1020938  0.3880582]
 [-0.4190991 -0.1257257 -1.299324  -0.8044026]]
     v2 ndarray (24, 4) float64 [[-0.1510132 -0.2147613  0.2822671 -0.1758341]
 [-0.7113093 -0.4927053 -0.7203961 -0.4009909]
 [ 0.5195733 -0.3813193 -0.1695824 -0.2181324]
 [-0.0318619 -0.1554424 -1.1281277  0.2076941]]
   train
     x ndarray (72, 4) float64 [[-0.3661061 -0.6096204 -0.9215192  0.9743149]
 [-0.2900984  0.1924035  0.2533542  0.3347788]
 [ 0.8675915 -0.4919501  0.7335846  0.1984901]
 [-0.5210518 -0.9756265  0.9743919  1.113905 ]]
     y ndarray (72,) float64 [ 0.2141084 -0.500113   0.6157096  0.9057031 -0.0109937 -1.1128867  0.7400883  1.2836316  0.7613798  1.0705802 -1.3185429  0.8854063
 -1.2224653  0.674622   0.416645   0.0791666 -0.5257722  0.8990743 -0.1931946  0.4328623  0.4912759  1.3537026 -0.2951941 -1.8480477
 -0.3086673  1.5048635 -0.9157753 -0.6501539  0.720349   0.1446362  0.5192278 -0.1793409 -0.7905282 -1.5144008 -0.7335808  1.0420702
 -0.7042807 -0.2201266  0.7663432 -1.4224988 -1.0380151 -0.4664131  0.366541  -1.0050437 -1.0135145  0.0463556 -0.1405798  1.3505738
 -0.7897913 -0.9766209  1.1289499  0.6840888  1.1376898  0.7833231 -1.6342014  1.487713  -1.5526623 -1.5364935 -1.4764175 -1.0002061
 -1.0346276 -0.4029399 -1.0201835  0.8867606 -1.2895671  0.5131721  1.0517169  0.8076786  1.4774751  0.0082349 -1.0644665 -0.8119088]
     traj ndarray (72, 8) float64 [[-0.0824058 -0.166234  -0.2143035 -0.2356758 -0.2589674 -0.2689515 -0.2834264 -0.2899428]
 [ 0.0353832 -0.002133  -0.0599881 -0.1319954 -0.20744   -0.2825468 -0.3455072 -0.4048516]
 [ 0.1116216  0.1022425 -0.0130449 -0.162675  -0.3179313 -0.4574061 -0.5863167 -0.6988128]
 [ 0.3492307  0.5930085  0.650289   0.6064404  0.5518463  0.4712185  0.3990665  0.3494024]]
     summary ndarray (4,) float64 [-0.0768091  0.9444762 -0.027983   1.0124467]
   test
     x ndarray (36, 4) float64 [[-0.704429   0.6955117 -0.9463571 -0.4025035]
 [-0.9201917 -0.3661589  0.4070664  1.3651692]
 [-0.056353  -0.1244108 -0.5991233  0.0289724]
 [-1.2215448 -0.552783   0.5707275 -0.3399058]]
   telemetry_vec ndarray (4,) float64 [0.22 2.15 0.22 0.22]
```

### Turn 3 — `execute_in_remote_docker` (`15586 ms` LLM)
- **Rationale**: Recover the archived formula without truncated output and inspect forward and inverse panel residuals for evidence of outliers, nonlinear calibration, or a direction mismatch.

```bash
sed -n '1,180p' /tmp/hegel_continual_archive/dev_m102_c02_batch_00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment robust IRLS' 2>&1 | head -25; cd /holdout_task; python3 - <<'PY'
import numpy as np,pickle
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; print(list(s))
np.set_printoptions(precision=5,suppress=True)
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; A=np.c_[np.ones(len(u)),u];
 for name,v in p.items():
  if name=='u': continue
  for z,w in [(u,v),(v,u)]:
   D=np.c_[np.ones(len(z)),z]; B=np.linalg.lstsq(D,w,rcond=None)[0]; e=np.linalg.norm(D@B-w,axis=1); print(k,name,'forward',z is u,'beta',B,'sorted errors',np.sort(e))
x=s['train']['x']; t=s['train']['traj']; y=s['train']['y']; A=np.c_[np.ones(len(x)),x]; print('raw affine MAE',np.mean(abs(A@np.linalg.lstsq(A,t,rcond=None)[0]-t)),np.mean(abs(A@np.linalg.lstsq(A,y,rcond=None)[0]-y)))
PY
```
- **Container Exit Code**: `0` (`447 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.464   -0.63757]
 [-0.59732  0.62516 -0.03613  0.15946]] sorted errors [0.00005 0.00006 0.00009 0.0001  0.0001  0.00011 0.00012 0.00013 0.00014
 0.00014 0.00015 0.00016 0.00016 0.00016 0.00017 0.00017 0.00018 0.00019
 0.00019 0.0002  0.00021 0.00022 0.00025 0.00034]
panel_b v forward False beta [[-0.16189 -0.07521 -0.13122 -0.04234]
 [-0.11075 -0.69995 -0.44059 -0.77134]
 [ 0.11204 -0.75279 -0.24558  0.80738]
 [ 0.92206  0.28266 -0.59915 -0.04658]
 [-0.6452   0.39342 -0.82337  0.20591]] sorted errors [0.00005 0.00007 0.0001  0.00011 0.00012 0.00013 0.00013 0.00015 0.00015
 0.00016 0.00017 0.00018 0.00018 0.00018 0.00019 0.00019 0.0002  0.00022
 0.00022 0.00023 0.00024 0.00025 0.00029 0.00039]
panel_c v1 forward True beta [[-0.04242 -0.08604 -0.02751 -0.10832]
 [-0.26215 -0.11508  0.28602  0.96451]
 [-0.70639 -0.47966  0.37098 -0.28745]
 [-0.56808  0.5182  -0.43801 -0.02409]
 [-0.00667 -0.55383 -0.70422  0.03578]] sorted errors [0.12813 0.13842 0.14331 0.14455 0.15254 0.15275 0.15455 0.15576 0.15855
 0.15968 0.16726 0.17958 0.18698 0.1982  0.20433 0.21436 0.2176  0.21848
 0.22702 0.24176 0.24731 0.24995 0.26718 0.30847]
panel_c v1 forward False beta [[ 0.07391 -0.10496  0.01098 -0.07384]
 [-0.21541 -0.71028 -0.721    0.00252]
 [-0.08633 -0.48656  0.67737 -0.69696]
 [ 0.14065  0.37062 -0.51166 -0.8227 ]
 [ 0.8771  -0.35589  0.02214  0.1328 ]] sorted errors [0.11837 0.1228  0.13459 0.13656 0.13676 0.13921 0.14069 0.14676 0.15522
 0.17166 0.18351 0.19314 0.19804 0.19837 0.20762 0.21757 0.22668 0.23047
 0.24538 0.25538 0.25922 0.29213 0.30512 0.33961]
panel_c v2 forward True beta [[-0.08585 -0.28809 -0.2228  -0.00727]
 [-0.12328 -0.42415  0.0385  -0.11883]
 [-0.15387 -0.17319  0.1393   0.43035]
 [ 0.37124 -0.10527 -0.25898  0.10134]
 [-0.31971 -0.05211 -0.39944  0.06243]] sorted errors [0.04822 0.05315 0.09366 0.10815 0.12107 0.12969 0.13855 0.14369 0.15354
 0.15515 0.16009 0.16226 0.1657  0.16653 0.17095 0.17157 0.18401 0.20876
 0.21096 0.21204 0.2212  0.228   0.25905 0.26326]
panel_c v2 forward False beta [[-0.51464 -0.00246 -0.28491 -0.38862]
 [-0.12963 -0.22439  1.46417 -1.04875]
 [-1.83924 -0.30065 -0.62593  0.11297]
 [ 0.24227  0.50621 -1.01579 -1.52033]
 [-0.95136  1.7767   0.58807  0.20327]] sorted errors [0.10802 0.11977 0.18994 0.24078 0.26527 0.26768 0.2797  0.29048 0.29278
 0.29771 0.30143 0.31784 0.31884 0.32453 0.35619 0.35705 0.37664 0.39075
 0.3987  0.42689 0.44902 0.4647  0.47255 0.5556 ]
raw affine MAE 0.08742379929905698 0.11019640126928174
```

### Turn 4 — `execute_in_remote_docker` (`18542 ms` LLM)
- **Rationale**: Inspect sensor residuals by channel and compare leave-one-out polynomial errors to distinguish sparse corruption from systematic curvature before choosing an invertible calibration law.
- **Alternatives Weighed**: `Replace the panel B affine operator` (rejected: Its largest forward residual is only 0.00034, so panel A is the more plausible source of the calibration mismatch.)

```bash
cd /holdout_task; cat public_asset_index.json; python3 - <<'PY'
import numpy as np,pickle
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v')); np.set_printoptions(precision=5,suppress=True,linewidth=140)
for j in range(4):
 a,b=u[:,j],v[:,j]; i,k=np.triu_indices(len(a),1); dx=a[k]-a[i]; ok=abs(dx)>.1*np.ptp(a); m=np.median((b[k[ok]]-b[i[ok]])/dx[ok]); q=np.median(b-a*m); e=b-q-a*m; order=np.argsort(a); print('\nCHANNEL',j,m,q,'raw vs residual',np.c_[a[order],b[order],e[order]].T); ix=np.argsort(abs(e))[:-3];
 for deg in range(1,8):
  D=np.polynomial.polynomial.polyvander(a[ix],deg); P=np.linalg.pinv(D); r=b[ix]-D@(P@b[ix]); h=np.sum(D*P.T,axis=1); print('poly',deg,'loo',np.median(abs(r/(1-h))),np.sqrt(np.mean((r/(1-h))**2)),P@b[ix]);
 print('large rows',np.argsort(abs(e))[-5:]);
print('top level',list(s)); print('ranges',[(k,s[k]['x'].min(axis=0),s[k]['x'].max(axis=0)) for k in ('train','test')]); print('panel A ranges',u.min(axis=0),u.max(axis=0),v.min(axis=0),v.max(axis=0))
PY
```
- **Container Exit Code**: `0` (`378 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[[-1.28955 -1.26712 -1.16179 -1.12927 -1.11483 -0.89707 -0.81914 -0.76639 -0.7354  -0.64542 -0.5758  -0.53638 -0.45941 -0.34388 -0.33007
  -0.31044 -0.18581 -0.17576 -0.14893 -0.12229  0.12076  0.12359  0.20884  0.27469  0.34731  0.40969  0.41771  0.52647  0.55021  0.56931
   0.5712   0.57337  0.58873  0.61855  0.62081  0.64649  0.65952  0.73637  0.78804  0.84795  0.84847  0.89107  0.9267   1.01331  1.09976
   1.15198  1.16229  1.24019]
 [-1.80714 -1.78559 -1.67598 -7.32173 -1.02487 -1.04939 -1.3278  -1.10422 -0.68458 -0.97034 -0.95532 -0.86365 -0.67907 -0.27316 -6.36986
  -0.52353 -0.39335 -0.47462 -0.12484 -0.28833  0.29908  6.15797  0.41858 -0.00836  0.69966  0.64162  0.59924  0.32149  0.77184  0.82616
   0.51751  0.85144  0.36136  1.00419  0.48264  1.02741  0.85377  1.14999  0.57019  1.20982  0.77017  1.31426  0.81558  0.81011  1.0184
   1.56742  1.04261  1.51254]
 [-0.05291 -0.05953 -0.08227 -5.76887  0.50985  0.21174 -0.16457 -0.00726  0.37344 -0.02536 -0.09781 -0.05567  0.03222  0.29298 -5.82108
   0.00059 -0.0258  -0.11969  0.19637 -0.00059  0.28146  6.1368   0.29031 -0.21936  0.39743  0.26101  0.20856 -0.20583  0.21469  0.24501
  -0.06601  0.2652  -0.24418  0.36118 -0.1632   0.3493   0.15929  0.35896 -0.28575  0.2786  -0.16169  0.32887 -0.21457 -0.32886 -0.22917
   0.25424 -0.28352  0.08854]]
poly 1 loo 0.221759613436811 0.23693620721632452 [-0.06755  1.23758]
poly 2 loo 0.19751008773595866 0.2379854446549545 [-0.0192   1.23567 -0.08701]
poly 3 loo 0.21302890904333105 0.24217547599357792 [-0.02362  1.31866 -0.09072 -0.08496]
poly 4 loo 0.21249087268954064 0.24868660475485835 [-0.02994  1.31775 -0.05895 -0.0858  -0.02088]
poly 5 loo 0.16574623130172175 0.23889660747283553 [-0.02986  1.6372  -0.08928 -1.02081  0.02885  0.50367]
poly 6 loo 0.1807596567185388 0.24737885106078417 [-0.02094  1.64183 -0.1816  -1.02498  0.19683  0.50164 -0.07433]
poly 7 loo 0.1844970198035632 0.25353077732206336 [-0.02976  1.43313 -0.11057  0.05021  0.03957 -0.91918  0.0134   0.53168]
large rows [39  7  5 31 17]
top level ['eval_times', 'panel_a', 'panel_b', 'panel_c', 'train', 'test', 'telemetry_vec']
ranges [('train', array([-1.46395, -1.39083, -1.6448 , -1.53204]), array([1.37504, 1.56523, 1.25173, 1.3994 ])), ('test', array([-1.43158, -0.97737, -1.38174, -1.3516 ]), array([1.31661, 1.54427, 1.07104, 1.39345]))]
panel A ranges [-1.29481 -1.29966 -1.29782 -1.28955] [1.00697 1.23079 1.29389 1.24019] [-5.49025 -6.91985 -6.52242 -7.32173] [6.87368 7.63924 5.80988 6.15797]
```

### Turn 5 — `execute_in_remote_docker` (`28564 ms` LLM)
- **Rationale**: Remove grossly corrupted panel rows, then compare cross-channel products, sums, differences, and bounded nonlinear terms on the remaining calibration pairs. The probe uses no prediction labels and retains the well-supported panel B affine map.
- **Alternatives Weighed**: `Increase the degree of independent channel polynomials` (rejected: Near-identical reference readings have substantially different residuals, and higher-degree fits did not consistently improve leave-one-out error.)

```bash
cat > /tmp/panel_probe.py <<'PY'
import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_a'][k] for k in ('u','v')]; keep=np.max(abs(v),axis=1)<3; u,v=u[keep],v[keep]; print('clean rows',len(u),'telemetry',s['telemetry_vec']); np.set_printoptions(precision=4,suppress=True)
base=np.c_[np.ones(len(u)),u]; B=np.linalg.lstsq(base,v,rcond=None)[0]; E=v-base@B; print('affine',B,'rms',np.std(E,axis=0),'residual svd',np.linalg.svd(E,compute_uv=False)); print('rows [u, residual]',np.c_[u,E][:12]);
L=[]; names=[]
def add(n,q):
 if np.std(q)>1e-10: L.append(q); names.append(n)
for j in range(4):
 for k in range(j,4):
  for l in range(k,4): add(f'u{j}*u{k}*u{l}',u[:,j]*u[:,k]*u[:,l]);
  add(f'u{j}*u{k}',u[:,j]*u[:,k]);
for z in itertools.product([0,1,-1],repeat=4):
 if not any(z) or next(t for t in z if t)!=1: continue
 a=u@z
 for f in [.22,.6,1.,1.25,1.5,2.,2.15,2.5,3.,3.5,4.]:
  add(f'sin({f}*{z})',np.sin(f*a)); add(f'cos({f}*{z})',np.cos(f*a)); add(f'tanh({f}*{z})',np.tanh(f*a)); add(f'rat({f}*{z})',1/(1+f*a*a))
D=np.array(L).T; D-=base@np.linalg.lstsq(base,D,rcond=None)[0]; D/=np.maximum(1e-13,np.linalg.norm(D,axis=0));
for j in range(4):
 for mode in ['diag','full']:
  A=base if mode=='full' else base[:,[0,1+j]]; D0=np.array(L).T; D0-=A@np.linalg.lstsq(A,D0,rcond=None)[0]; D0/=np.maximum(1e-13,np.linalg.norm(D0,axis=0)); R=v[:,j]-A@np.linalg.lstsq(A,v[:,j],rcond=None)[0]; corr=D0.T@R/np.linalg.norm(R); ix=np.argsort(-abs(corr))[:6]; print(j,mode,[(names[k],round(corr[k],5)) for k in ix]); sel=[]; e=R
  for _ in range(3):
   k=np.argmax(abs(D0.T@e)); sel.append(k); M=D0[:,sel]; e=R-M@np.linalg.lstsq(M,R,rcond=None)[0]; print('   ',names[k], 'rms',np.sqrt(np.mean(e**2)))
for name in ('panel_b','panel_c'):
 p=s[name]; A=np.c_[np.ones(len(p['u'])),p['u']]; print(name,[(k,np.linalg.svd(np.linalg.lstsq(A,v,rcond=None)[0][1:],compute_uv=False)) for k,v in p.items() if k!='u'])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/panel_probe.py
```
- **Container Exit Code**: `0` (`529 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.0*(1, 0, 1, 0)) rms 0.0020826228137087784
1 diag [('sin(2.5*(0, 0, 0, 1))', -0.9882), ('sin(2.15*(0, 0, 0, 1))', -0.98293), ('sin(2.0*(0, 0, 0, 1))', -0.9674), ('tanh(4.0*(0, 0, 0, 1))', -0.93434), ('tanh(3.5*(0, 0, 0, 1))', -0.93328), ('tanh(3.0*(0, 0, 0, 1))', -0.93021)]
    sin(2.5*(0, 0, 0, 1)) rms 0.03633514109831563
    sin(2.5*(0, 1, 0, 0)) rms 0.016155201023625734
    u3*u3*u3 rms 0.004146723420531206
1 full [('sin(2.5*(0, 0, 0, 1))', -0.98191), ('sin(2.15*(0, 0, 0, 1))', -0.98143), ('tanh(1.0*(0, 0, 0, 1))', -0.98103), ('sin(2.0*(0, 0, 0, 1))', -0.98083), ('tanh(1.25*(0, 0, 0, 1))', -0.98018), ('tanh(0.6*(0, 0, 0, 1))', -0.97833)]
    sin(2.5*(0, 0, 0, 1)) rms 0.027385972728077797
    tanh(2.0*(0, 1, 0, 0)) rms 0.0021885516311718127
    cos(4.0*(1, 1, 0, 0)) rms 0.0019234564549412104
2 diag [('sin(2.5*(1, 0, 0, 0))', -0.98973), ('sin(2.15*(1, 0, 0, 0))', -0.98403), ('sin(2.0*(1, 0, 0, 0))', -0.97203), ('tanh(4.0*(1, 0, 0, 0))', -0.94085), ('tanh(3.5*(1, 0, 0, 0))', -0.94074), ('tanh(3.0*(1, 0, 0, 0))', -0.93922)]
    sin(2.5*(1, 0, 0, 0)) rms 0.0350230966857713
    tanh(1.5*(0, 0, 1, 0)) rms 0.015167851258600734
    u0*u0*u0 rms 0.0036008857125235955
2 full [('sin(2.15*(1, 0, 0, 0))', -0.97598), ('sin(2.0*(1, 0, 0, 0))', -0.97559), ('tanh(1.0*(1, 0, 0, 0))', -0.97557), ('sin(2.5*(1, 0, 0, 0))', -0.97532), ('tanh(1.25*(1, 0, 0, 0))', -0.97368), ('tanh(0.6*(1, 0, 0, 0))', -0.97327)]
    sin(2.15*(1, 0, 0, 0)) rms 0.02954352990272795
    tanh(2.0*(0, 0, 1, 0)) rms 0.004533936596834491
    sin(4.0*(1, 0, 0, 0)) rms 0.0011861697359192797
3 diag [('sin(2.5*(0, 1, 0, 0))', -0.98601), ('sin(2.15*(0, 1, 0, 0))', -0.98278), ('sin(2.0*(0, 1, 0, 0))', -0.96927), ('tanh(4.0*(0, 1, 0, 0))', -0.96063), ('tanh(3.5*(0, 1, 0, 0))', -0.95875), ('tanh(3.0*(0, 1, 0, 0))', -0.95451)]
    sin(2.5*(0, 1, 0, 0)) rms 0.03766102867865297
    tanh(3.0*(0, 0, 0, 1)) rms 0.015081560995643093
    sin(4.0*(0, 1, 0, 0)) rms 0.006033674724335171
3 full [('tanh(1.5*(0, 1, 0, 0))', -0.976), ('sin(3.0*(0, 1, 0, 0))', -0.97501), ('tanh(1.25*(0, 1, 0, 0))', -0.97483), ('sin(2.5*(0, 1, 0, 0))', -0.97292), ('tanh(2.0*(0, 1, 0, 0))', -0.97232), ('tanh(1.0*(0, 1, 0, 0))', -0.97164)]
    tanh(1.5*(0, 1, 0, 0)) rms 0.02725182747918541
    sin(3.0*(0, 0, 0, 1)) rms 0.010151536346385995
    rat(4.0*(1, 0, -1, -1)) rms 0.008947304396896184
panel_b [('v', array([0.88, 0.88, 0.88, 0.88]))]
panel_c [('v1', array([1.0763, 0.9651, 0.8804, 0.8754])), ('v2', array([0.5786, 0.498 , 0.4519, 0.4203]))]
```

### Turn 6 — `execute_in_remote_docker` (`24460 ms` LLM)
- **Rationale**: Compare low-dimensional nonlinear bases on calibration pairs alone. Variable projection fits the linear coefficients exactly while testing whether shared frequencies and pair symmetry explain the remaining residuals.
- **Alternatives Weighed**: `Use an unrestricted four-dimensional polynomial` (rejected: The residual coupling follows two specific pairs; an unrestricted fit would add unsupported terms and could extrapolate poorly.)
- **ATMS NOGOOD DECLARED**: Refuted `Independent per-channel affine calibration is sufficient after removing gross outliers.` | Witness: `After removing three grossly corrupted rows, the opposite-channel sine feature explains the diagonal-affine residual with absolute correlation 0.986–0.990 across all four channels.` | Blanket Diagnosis: `Panel A has systematic coupling between channels 0 and 2 and between channels 1 and 3; diagonal calibration omits that coupling.` | Sublated Pivot: `Keep the supported panel B affine operator and fit a compact, pair-symmetric forward law for panel A before inverting it.`

```bash
cd /holdout_task; cat _frozen_nodes/node_01.py; cat _frozen_nodes/node_02.py; cat test_dag_candidate.py | tail -105; cat > /tmp/panel_basis.py <<'PY'
import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_a'][k] for k in ('u','v')]; keep=np.max(abs(v),axis=1)<3; u,v=u[keep],v[keep]
q=u[:,[2,3,0,1]]
fns={'sin':np.sin,'tanh':np.tanh,'cubic':lambda a:a**3,'rat':lambda a:a/(1+a*a),'atan':np.arctan,'sinh':np.sinh}
np.set_printoptions(precision=6,suppress=True)
for lin_cross in (False,True):
 ans=[]
 for own,other in itertools.product(fns,repeat=2):
  def calc(f,ids,details=False):
   E=[]; B=[]
   for j in range(4):
    a=u[ids,j]; b=q[ids,j]; A=np.column_stack([np.ones(len(a)),a]+([b] if lin_cross else [])+[fns[own](f[0]*a),fns[other](f[1]*b)]); y=v[ids,j]; co=np.linalg.lstsq(A,y,rcond=None)[0]; E.append(A@co-y); B.append(co)
   return (np.array(E),np.array(B)) if details else np.ravel(E)
  ids=np.arange(len(u)); starts=([1,2.15],[2,2.5],[3,3],[1,1]); fit=min((least_squares(calc,start,args=(ids,),bounds=(.15,5),max_nfev=55) for start in starts),key=lambda r:r.cost)
  E,B=calc(fit.x,ids,True); ans.append((np.sqrt(np.mean(E**2)),own,other,fit.x,B))
 print('linear cross',lin_cross)
 for rms,own,other,f,B in sorted(ans,key=lambda a:a[0])[:6]: print(rms,own,other,f,'coef',B)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/panel_basis.py
```
- **Container Exit Code**: `0` (`2152 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
7628  0.217829 -0.319951]
 [ 0.087427  0.835722  0.217986 -0.320003]
 [-0.178729  0.836409  0.217551 -0.320014]
 [-0.046083  0.922195  0.21785  -0.320127]]
0.0014686961241385124 sin sin [3.090928 2.39973 ] coef [[-0.007625  1.033957  0.045989 -0.320092]
 [ 0.087172  1.062934  0.045953 -0.320174]
 [-0.178431  1.062663  0.046515 -0.319856]
 [-0.046117  1.148218  0.046071 -0.319596]]
0.007667226762510215 cubic sin [1.136501 2.395429] coef [[-0.007193  1.127433 -0.055021 -0.320805]
 [ 0.087795  1.167692 -0.063836 -0.323071]
 [-0.179687  1.160494 -0.057401 -0.3194  ]
 [-0.046765  1.247056 -0.057692 -0.322485]]
0.007679462702429518 sinh sin [0.15    2.39542] coef [[  -0.007191   22.615218 -143.252421   -0.320806]
 [   0.087796   26.106586 -166.25972    -0.323076]
 [  -0.17969    23.57851  -149.453978   -0.3194  ]
 [  -0.046765   23.780759 -150.225154   -0.322491]]
linear cross True
8.30932940770816e-05 tanh sin [1.79843  2.399929] coef [[-0.008057  0.867248  0.000057  0.180264 -0.320026]
 [ 0.087383  0.895601  0.000063  0.180227 -0.320049]
 [-0.178652  0.895693  0.000017  0.180262 -0.32001 ]
 [-0.046103  0.981703  0.000107  0.180281 -0.320095]]
0.00010305748778751479 rat sin [1.109561 2.400688] coef [[-0.008032  0.913624 -0.000249  0.249692 -0.319827]
 [ 0.087372  0.941975 -0.000172  0.249647 -0.319901]
 [-0.178636  0.94199  -0.000266  0.249832 -0.319824]
 [-0.046099  1.02808  -0.000228  0.249674 -0.31984 ]]
0.0003538998128613659 atan sin [1.785802 2.396367] coef [[-0.008179  0.808127  0.001451  0.21707  -0.320919]
 [ 0.087428  0.83653   0.001219  0.217    -0.320781]
 [-0.178737  0.837113  0.001339  0.216635 -0.320874]
 [-0.046106  0.922511  0.001635  0.217278 -0.321238]]
0.0013921229748164404 sin sin [3.081096 2.414629] coef [[-0.007586  1.03372  -0.006134  0.046045 -0.315937]
 [ 0.087188  1.062296 -0.004208  0.046329 -0.317582]
 [-0.178407  1.062197 -0.005366  0.04673  -0.316429]
 [-0.046     1.147908 -0.00674   0.045961 -0.315015]]
0.001660154368179231 tanh rat [1.749888 0.681748] coef [[-0.007764  0.862547  0.990048  0.186585 -2.594699]
 [ 0.087562  0.886099  0.994578  0.191213 -2.60561 ]
 [-0.178212  0.888907  0.993245  0.188356 -2.60255 ]
 [-0.046333  0.980531  0.995507  0.183329 -2.605619]]
0.0016644828957668866 atan rat [1.726816 0.680349] coef [[-0.007883  0.801152  0.994039  0.227    -2.605289]
 [ 0.087605  0.823291  0.997769  0.232531 -2.614761]
 [-0.178295  0.827502  0.997053  0.228677 -2.612867]
 [-0.046335  0.920136  1.000326  0.223221 -2.61808 ]]
```

### Turn 7 — `execute_in_remote_docker` (`34511 ms` LLM)
- **Rationale**: Fit the shared nonlinear calibration terms from the reference panel alone, invert the paired response, and leave the validated panel B map and frozen consumers unchanged. Run the official DAG test in an isolated Python process.
- **Alternatives Weighed**: `Add an affine cross-channel term` (rejected: It barely improves the panel fit and yields coefficients near zero.)

```bash
cat > /holdout_task/node_00.py <<'PY'
"""Recover the shared coordinates from the two calibration panels."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit_sensor(u, v):
    # The gross faults affect entire paired calibration rows. Screen them
    # against a robust diagonal fit before fitting the coupled response.
    i, j = np.triu_indices(len(u), 1)
    dx = u[j] - u[i]
    dy = v[j] - v[i]
    slopes = np.full_like(dx, np.nan)
    np.divide(dy, dx, out=slopes, where=np.abs(dx) > 0.1 * np.ptp(u, axis=0))
    slope = np.nanmedian(slopes, axis=0)
    bias = np.median(v - u * slope, axis=0)
    err = np.abs(v - (bias + u * slope))
    keep = np.all(err <= np.maximum(8 * np.median(err, axis=0), 1e-4), axis=1)
    if np.count_nonzero(keep) >= 8:
        u, v = u[keep], v[keep]
    paired = np.roll(u, 2, axis=1)
    d = np.zeros((len(u), 4, 10))
    k = np.arange(4)
    d[:, k, k] = 1
    d[:, k, 4 + k] = u

    def fit(freq, return_coeff=False):
        d[:, :, 8] = np.tanh(freq[0] * u)
        d[:, :, 9] = np.sin(freq[1] * paired)
        design = d.reshape(-1, 10)
        coeff = np.linalg.lstsq(design, v.ravel(), rcond=None)[0]
        if return_coeff:
            return coeff
        return design @ coeff - v.ravel()

    freq = least_squares(fit, [1.8, 2.4], bounds=([0.5, 0.8], [4.0, 4.0])).x
    return freq, fit(freq, return_coeff=True)


def _invert(x, freq, coeff):
    f, g = freq
    offset, gain, a, b = coeff[:4], coeff[4:8], coeff[8], coeff[9]
    shifted = x - offset
    z = shifted / gain
    # Solve the strictly increasing self-response exactly within each
    # paired update. The fitted cross-response is a contraction.
    for _ in range(240):
        rhs = shifted - b * np.sin(g * np.roll(z, 2, axis=1))
        q = rhs / gain
        for _ in range(8):
            t = np.tanh(f * q)
            q -= (gain * q + a * t - rhs) / (gain + a * f * (1 - t * t))
        delta = np.max(np.abs(q - z), initial=0.0)
        z = q
        if delta < 2e-12:
            break
    return z


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a, b = study['panel_a'], study['panel_b']
    freq, coeff = _fit_sensor(np.asarray(a['u'], dtype=float), np.asarray(a['v'], dtype=float))
    operator = np.linalg.lstsq(_affine(np.asarray(b['u'], dtype=float)),
                              np.asarray(b['v'], dtype=float), rcond=None)[0]
    train = np.asarray(study['train']['x'], dtype=float)
    query = np.asarray(study['test']['x'], dtype=float)
    coords = _affine(_invert(np.concatenate((train, query)), freq, coeff)) @ operator
    return {'c': coords[:len(train)], 'query': coords[len(train):]}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`516 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.005066, "public_cv_traj_mae": 0.004048}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1, "node_03": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3cc1df23ecd42378 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3cc1df23 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`30913 ms` LLM)
- **Rationale**: Keep the accepted artifact unchanged. Jackknife the calibration panel, check inversion on all supplied coordinates, and measure both prediction errors on held-out channel tails.
- **Alternatives Weighed**: `Submit on the official public split alone` (rejected: The acceptance test passes with margin, but pooled MAE does not show whether calibration remains accurate at channel extremes.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
def load(i):
    name = f'node_{i:02}'
    path = root + ('_frozen_nodes/' if i else '') + name + '.py'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mods = [load(i) for i in range(4)]
s = pickle.load(open(root + 'public_batch.pkl', 'rb'))[0]['study']
u, v = (s['panel_a'][k] for k in ('u', 'v'))
jack = []
for i in range(len(u)):
    mask = np.arange(len(u)) != i
    f, b = mods[0]._fit_sensor(u[mask], v[mask])
    if np.max(abs(v[i])) < 3:
        inv = mods[0]._invert(v[i:i+1], f, b)
        jack.append(float(np.max(abs(inv-u[i]))))
print('panel A censored-gross-fault jackknife: clean held-out rows', len(jack), 'worst coordinate absolute error', max(jack))
assert len(jack) >= 40 and max(jack) < 0.003

f, b = mods[0]._fit_sensor(u, v)
xa, xb = (s[k]['x'] for k in ('train', 'test'))
xx = np.vstack((xa, xb))
z = mods[0]._invert(xx, f, b)
back = b[:4] + b[4:8]*z + b[8]*np.tanh(f[0]*z) + b[9]*np.sin(f[1]*np.roll(z, 2, axis=1))
print('train/test inversion worst forward residual', np.max(abs(back-xx)))
assert np.max(abs(back-xx)) < 1e-9

def evaluate(tr, te):
    case = dict(s, train={k:s['train'][k][tr] for k in ('x','y','traj')}, test={'x':xa[te]})
    c = mods[0].run_node({'__input__':case})
    p = mods[1].run_node({'__input__':case, 'node_00':c})
    q = mods[2].run_node({'node_01':p})
    return [np.mean(abs(p['y_hat']-s['train']['y'][te])), np.mean(abs(q['state_pred']-s['train']['traj'][te]))]

rng = np.random.default_rng(761)
n = len(xa)
k = max(n//4,1)
scores = []
for _ in range(20):
    ids = rng.permutation(n)
    scores.append(evaluate(ids[k:], ids[:k]))
tails = []
for col in range(xa.shape[1]):
    ids = np.argsort(xa[:,col])
    tails.extend([evaluate(ids[k:], ids[:k]), evaluate(ids[:-k], ids[-k:])])
print('worst across 20 random holdouts (y, traj):', np.max(scores, axis=0))
print('worst across 8 channel-tail holdouts (y, traj):', np.max(tails, axis=0))
assert np.max(scores) < 0.038 and np.max(tails) < 0.038
PY
```
- **Container Exit Code**: `0` (`943 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
panel A censored-gross-fault jackknife: clean held-out rows 45 worst coordinate absolute error 0.0003993086813118829
train/test inversion worst forward residual 1.2185807918285718e-12
worst across 20 random holdouts (y, traj): [0.00559134 0.00489743]
worst across 8 channel-tail holdouts (y, traj): [0.00610104 0.00423417]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3cc1df23ecd42378 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3cc1df23 (directory).
```

### Turn 9 — `submit_completed_task` (`10408 ms` LLM)
- **Rationale**: The unchanged, hash-verified candidate passed the isolated public DAG test with one execution per node. The calibration jackknife had a worst absolute coordinate error of 0.0003993 across 45 clean held-out rows, and both prediction metrics stayed below threshold on every stress split.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_00.py`; all frozen nodes were left unchanged. The public DAG test accepted the candidate: MAE was 0.005066 for y and 0.004048 for trajectory, both below 0.038.`

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
