# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m102_b02_batch_09__blind_campaign__reuse_subgraph`

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
| **Total Tokens (Prompt / Cached / Out)** | `170,987` (`165,162` / `83,584` / `5,825`) |
| **Total Episode Wall Time (s)** | `192.3s` |

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

### Turn 1 — `execute_in_remote_docker` (`12488 ms` LLM)
- **Rationale**: Inspect the frozen consumers, public measurements, and archived transforms before testing which panel-A change explains the two prediction errors.

```bash
cd /holdout_task && cat _dag_spec.json && cat _frozen_nodes/node_01.py _frozen_nodes/node_02.py && cat test_dag_candidate.py && cat public_asset_index.json && ls -la /tmp/hegel_continual_archive && find /tmp/hegel_continual_archive -maxdepth 4 -type f | head -100 && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_batch.pkl','rb'))
def walk(x, pre=''):
 if isinstance(x,dict):
  for k,v in x.items(): walk(v,pre+'/'+str(k))
 elif isinstance(x,np.ndarray): print(pre,x.shape,x.dtype, np.ravel(x)[:12])
 elif isinstance(x,(list,tuple)): print(pre,type(x),len(x)); walk(x[0],pre+'/0') if len(x) else None
 else: print(pre,type(x),str(x)[:100])
walk(p)
PY
```
- **Container Exit Code**: `0` (`361 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
py
/tmp/hegel_continual_archive/archive_manifest.json
/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
/tmp/hegel_continual_archive/conf_m102_b02_batch_00_solver.py
/tmp/hegel_continual_archive/conf_m102_b02_batch_05_solver.py
 <class 'list'> 1
/0/study/eval_times (8,) float64 [0.25       0.64285714 1.03571429 1.42857143 1.82142857 2.21428571
 2.60714286 3.        ]
/0/study/panel_a/u (48, 4) float64 [-0.0929379   0.93474646 -1.10112701  0.59012595  0.8076013  -0.2178566
 -0.08845315  0.00972941  0.26326614 -0.34145026 -0.585733    0.24608551]
/0/study/panel_a/v (48, 4) float64 [ 0.16464207  0.7133445  -0.98493766  0.52534174  1.09862689 -0.41404209
 -0.33636391  0.31486529  0.76409603 -0.74473026 -0.76813241  0.6568999 ]
/0/study/panel_b/u (24, 4) float64 [-0.32041821  0.55329232  1.01314396  0.34029578 -0.69749506  0.78274965
 -0.53359781 -0.38840083  0.57223192 -0.51194198 -0.98106719  0.55430464]
/0/study/panel_b/v (24, 4) float64 [-0.33187357 -0.0816506  -0.72235737 -0.79739661 -0.11671956 -1.1669628
  0.09889653 -0.0788175   0.68527092  0.28922549  0.47883297  0.61541308]
/0/study/panel_c/u (24, 4) float64 [ 1.01700663  0.35091951  0.65445961  0.66913538 -0.36282135 -0.4098593
 -0.66107908 -0.82897195  0.23415078 -0.7725393  -0.06106955  0.90180057]
/0/study/panel_c/v1 (24, 4) float64 [-0.94712391  0.02947252 -0.89492422  0.52400113  0.79867257  0.50621429
  1.01123625 -0.26017238  0.28787604  0.20288275 -0.14761226  1.2364362 ]
/0/study/panel_c/v2 (24, 4) float64 [-0.59665687  0.43042765 -0.24602707 -0.14042501  0.23962963 -0.3035656
  0.30900226  0.2820076  -0.01681291  0.26907925 -0.33450735  0.49933034]
/0/study/train/x (72, 4) float64 [ 1.24890274 -0.98983182 -1.35218381  0.71185239  0.8472765  -0.0819612
 -1.25016093 -0.06366273  1.25329771 -1.15640202 -1.22779524  1.32880862]
/0/study/train/y (72,) float64 [-0.28925278  0.00407875 -0.21447571  0.283805   -0.09306817  1.1285924
 -0.8438298   0.84093165  1.25318998  1.13434032  0.7164087   0.79403591]
/0/study/train/traj (72, 8) float64 [0.34475345 0.74372406 1.0313769  1.24158405 1.41174805 1.55991815
 1.67581557 1.77391868 0.50108547 1.01131743 1.36965418 1.61785932]
/0/study/train/summary (4,) float64 [ 0.32751034  0.78638652 -0.20656248  1.81616409]
/0/study/test/x (36, 4) float64 [-0.96238008 -0.12820033  0.19017411 -0.26584656 -0.81725846  0.31166505
 -0.76868336 -0.347034   -0.2499865  -0.58346294 -0.75498081  0.65390901]
/0/study/telemetry_vec (4,) float64 [0.22 2.15 0.22 0.22]
```

### Turn 2 — `execute_in_remote_docker` (`10664 ms` LLM)
- **Rationale**: Read the archived model equations and exact fold construction before fitting and validating the panel-A replacement.

```bash
cat /tmp/hegel_continual_archive/*solver.py; sed -n '95,235p' /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`263 ms`, timed_out=`False`)
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

### Turn 3 — `execute_in_remote_docker` (`26854 ms` LLM)
- **Rationale**: Compare affine and nonlinear forward laws using only paired panel measurements, so selecting the calibration model cannot leak prediction-fold labels.
- **Alternatives Weighed**: `Fit a flexible inverse directly from observed panel-A coordinates` (rejected: The test inputs may extend beyond calibration support; first test low-dimensional forward laws whose inverses can extrapolate.)

```bash
cat /tmp/hegel_continual_archive/conf_m102_b02_batch_05_solver.py; cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust IRLS reference panel alignment' 2>&1 | head -30; cat > /tmp/panel_probe.py <<'PY'
import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
u=s['panel_a']['u']; v=s['panel_a']['v']; a=np.c_[np.ones(len(u)),u]; fit=np.linalg.lstsq(a,v,rcond=None)[0]; k=s['telemetry_vec'][1]
print('u/v/train/test ranges',[(x.min(0),x.max(0)) for x in (u,v,s['train']['x'],s['test']['x'])]); print('affine',fit); print('affine errors',np.round(np.linalg.norm(a@fit-v,axis=1),4)); print('telemetry',s['telemetry_vec'])
funcs={'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'sin':np.sin,'cos':np.cos,'erf':__import__('scipy.special',fromlist=['erf']).erf,'square':lambda x:x*x,'cube':lambda x:x**3,'abs':np.abs,'softsign':lambda x:x/(1+abs(x)),'rational':lambda x:x/(1+x*x),'sqrt':lambda x:x/np.sqrt(1+x*x),'softplus':lambda x:np.logaddexp(x,0)}
for name,f in funcs.items():
 h=f(k*u); errs=[]
 for j in range(4):
  opts=[]
  for i in range(4):
   q=np.c_[a,h[:,i]]; c=np.linalg.lstsq(q,v[:,j],rcond=None)[0]; opts.append((np.linalg.norm(q@c-v[:,j]),i,round(c[-1],5)))
  errs.append(min(opts))
 q=np.c_[a,h]; c=np.linalg.lstsq(q,v,rcond=None)[0]; print(name,'single', [(round(e,5),i,c) for e,i,c in errs], 'mixed',round(np.linalg.norm(q@c-v),5))
for name,f in funcs.items():
 scores=[]
 for j in range(4):
  l=fit[:,j]; y=v[:,j];
  def r(p):
   t=a@p[:5]; return t+p[5]*f(k*t)-y
  best=min((least_squares(r,np.r_[l,z],max_nfev=250) for z in [-.5,0,.5]),key=lambda t:t.cost)
  scores.append((round(np.linalg.norm(best.fun),6),round(best.x[-1],5)))
 print('after affine',name,scores)
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/panel_probe.py
```
- **Container Exit Code**: `0` (`1075 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
41798), (10.0671, 3, 0.2921), (9.97562, 3, -0.26162), (10.18393, 2, 0.37143)] mixed 20.00165
erf single [(10.14967, 0, -1.22362), (9.93135, 3, -1.32652), (9.95885, 1, -0.76553), (10.08469, 0, 1.38394)] mixed 19.90023
square single [(10.22557, 3, 0.11184), (10.00098, 3, -0.11609), (9.92211, 3, 0.10398), (10.24066, 2, -0.08113)] mixed 20.03019
cube single [(10.23874, 0, 0.07446), (9.71852, 3, 0.14392), (9.83435, 3, -0.09917), (10.19641, 0, -0.08503)] mixed 19.74775
abs single [(10.22427, 2, 0.26428), (10.081, 3, -0.21622), (9.98919, 3, 0.18883), (10.17485, 2, -0.30413)] mixed 20.07667
softsign single [(10.08467, 0, -2.62457), (9.96451, 3, -2.39543), (9.9414, 0, -1.5294), (10.01466, 0, 2.90929)] mixed 19.84261
rational single [(10.0997, 0, -1.50427), (10.00839, 3, -1.24771), (9.94475, 1, -0.97068), (10.02883, 0, 1.67914)] mixed 19.88692
sqrt single [(10.13726, 0, -1.60915), (9.93784, 3, -1.70204), (9.96031, 1, -0.98077), (10.07148, 0, 1.81092)] mixed 19.89607
softplus single [(10.21974, 3, 1.13047), (10.02184, 3, -1.05763), (9.93905, 3, 0.947), (10.22521, 2, -0.87284)] mixed 20.02678
after affine tanh [(10.325637, 0.60471), (10.017787, -0.72634), (9.934515, 1.30871), (9.920365, -1.17813)]
after affine atan [(10.330083, 0.61479), (10.040395, -0.63026), (9.945784, 1.08392), (9.918368, -1.08132)]
after affine asinh [(10.33896, 0.52321), (10.06721, -0.50622), (9.972628, 0.76158), (9.916906, -0.86832)]
after affine sin [(10.320499, 0.33846), (10.077074, -0.30541), (9.990406, 0.23197), (9.832603, -0.83848)]
after affine cos [(10.269983, 0.34969), (9.798019, -0.58984), (9.715268, -0.93852), (10.161024, 0.66925)]
after affine erf [(10.323109, 0.54974), (10.002088, -0.70696), (9.929405, 1.32667), (9.91811, -1.11247)]
after affine square [(10.301341, -0.04219), (9.995948, 0.04001), (10.032904, -0.00318), (10.195372, -0.09527)]
after affine cube [(10.317125, -0.00977), (10.137232, 0.00217), (9.873672, -0.00436), (10.090901, 17.23394)]
after affine abs [(10.290045, -0.15293), (9.92138, 0.20242), (10.031326, 0.02343), (10.004814, -0.37705)]
after affine softsign [(9.878125, -2.32835), (10.024998, -1.01908), (9.936372, 2.49499), (9.927094, -1.76082)]
after affine rational [(10.260941, -1.65263), (9.950558, -1.2687), (9.960405, 0.68963), (9.937562, -1.77522)]
after affine sqrt [(10.323422, 0.69002), (10.01704, -0.76767), (9.935575, 1.53682), (9.922987, -1.26624)]
after affine softplus [(10.29586, -0.29296), (10.001308, 1.07192), (10.033922, 0.00222), (10.041952, -0.43051)]
```

### Turn 4 — `execute_in_remote_docker` (`26527 ms` LLM)
- **Rationale**: Remove gross contamination before testing the archived polynomial and competing pairwise, rational, and saturating bases. Independently verify that panel B remains affine.
- **Alternatives Weighed**: `Choose a nonlinear function from the untrimmed fits` (rejected: Three residuals exceed 10 and dominate the relative scores of those fits.)

```bash
cat > /tmp/robust_probe.py <<'PY'
import numpy as np,pickle
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_a'][i] for i in ('u','v')]; k=s['telemetry_vec'][1]; d=u.shape[1]; dsg=lambda x:np.c_[np.ones(len(x)),x]
def f(D,v):return np.linalg.lstsq(D,v,rcond=None)[0]
a=dsg(u); e=np.linalg.norm(a@f(a,v)-v,axis=1); g=e<5
for _ in range(6):
 c=f(a[g],v[g]); e=np.linalg.norm(a@c-v,axis=1); g=e<max(.002,4*np.median(e))
print('affine coef',c,'errors',np.round(e,4),'exclude',np.where(~g)[0]);print('ranges',[(x.min(0),x.max(0)) for x in (u,v,s['train']['x'],s['test']['x'])]); print('panel A values after excluding gross outliers:');print(np.round(np.c_[u,v,e][g][:14],4))
funcs={'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'sin':np.sin,'cos':np.cos,'erf':erf,'square':lambda x:x*x,'cube':lambda x:x**3,'abs':np.abs,'softsign':lambda x:x/(1+abs(x)),'rational':lambda x:x/(1+x*x),'sqrt':lambda x:x/np.sqrt(1+x*x),'softplus':lambda x:np.logaddexp(x,0),'sinh':np.sinh}
g=e<5
import itertools
pairs=np.array(list(itertools.combinations(range(d),2))); i,j=pairs.T;p=u[:,i]*u[:,j]; plus=u[:,i]+u[:,j]; minus=u[:,i]-u[:,j]; cyc=u*np.roll(u,-1,axis=1)
Ds={'affine':a,'archive poly':np.c_[a,cyc,u**3],'quadratic':np.c_[a,u**2,p],'odd poly':np.c_[a,u**3]}
for name,func in funcs.items():
 for lab,t in [('u',u),('sum/diff',np.c_[plus,minus]),('prod',p),('post affine',a@c)]: Ds[name+' '+lab]=np.c_[a,func(k*t)]
res=[]
for name,D in Ds.items():
 co=f(D[g],v[g]); e0=np.linalg.norm(D@co-v,axis=1); hat=np.sum(np.linalg.qr(D[g])[0]**2,axis=1); loo=e0[g]/(1-hat);res.append((np.mean(loo),np.max(loo),np.mean(e0[g]),name))
print('best panel A calibration loo:');print(*sorted(res)[:35],sep='\n')
b=s['panel_b']; aa=dsg(b['u']); print('panel B residual',np.max(abs(aa@f(aa,b['v'])-b['v'])))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/robust_probe.py; cat /tmp/hegel_continual_archive/archive_manifest.json
```
- **Container Exit Code**: `0` (`411 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.22014191187208468, 'sin prod')
(0.3041884558200919, 0.854552060253039, 0.21610890238834138, 'softplus prod')
(0.3093193577335281, 0.6263549660841496, 0.22740094988415357, 'rational prod')
(0.30967453209318463, 0.5313084049323717, 0.24293264801222933, 'abs post affine')
(0.3101776287937594, 0.8406302739994054, 0.16042816055897796, 'sinh sum/diff')
(0.31018278753790474, 0.536889678615195, 0.24372102801068785, 'cos post affine')
panel B residual 0.0002505786936450455
[
  {
    "short_name": "conf_m102_b02_batch_00",
    "task_id": "conf_m102_b02_batch_00__blind_campaign",
    "solver_path": "/tmp/hegel_continual_archive/conf_m102_b02_batch_00_solver.py",
    "stat_summary": {
      "stat_00": 1.0,
      "stat_01": 4.1e-05,
      "stat_02": 0.0,
      "stat_03": 0.701652,
      "stat_04": 1.0,
      "stat_05": 0.0,
      "stat_06": 0.0,
      "stat_07": 0.630562,
      "stat_08": 0.961049,
      "stat_09": 0.038689,
      "stat_10": 0.866191,
      "stat_11": 0.133646,
      "stat_12": 0.833161,
      "stat_13": 0.685808,
      "stat_14": 0.22,
      "stat_15": 0.22,
      "stat_16": 0.22,
      "stat_17": 0.22
    }
  },
  {
    "short_name": "conf_m102_b02_batch_01",
    "task_id": "conf_m102_b02_batch_01__blind_campaign",
    "solver_path": "/tmp/hegel_continual_archive/conf_m102_b02_batch_01_solver.py",
    "stat_summary": {
      "stat_00": 1.0,
      "stat_01": 4.5e-05,
      "stat_02": 0.0,
      "stat_03": 0.660101,
      "stat_04": 0.86104,
      "stat_05": 0.13881,
      "stat_06": 0.069176,
      "stat_07": 0.489409,
      "stat_08": 0.961996,
      "stat_09": 0.037709,
      "stat_10": 0.887945,
      "stat_11": 0.111893,
      "stat_12": 0.818752,
      "stat_13": 0.575795,
      "stat_14": 0.22,
      "stat_15": 0.22,
      "stat_16": 0.22,
      "stat_17": 1.25
    }
  },
  {
    "short_name": "conf_m102_b02_batch_05",
    "task_id": "conf_m102_b02_batch_05__blind_campaign",
    "solver_path": "/tmp/hegel_continual_archive/conf_m102_b02_batch_05_solver.py",
    "stat_summary": {
      "stat_00": 0.94302,
      "stat_01": 0.41256,
      "stat_02": 0.025337,
      "stat_03": 0.920942,
      "stat_04": 0.85379,
      "stat_05": 0.14608,
      "stat_06": 0.097845,
      "stat_07": 0.45247,
      "stat_08": 0.019995,
      "stat_09": 0.512117,
      "stat_10": 0.9257,
      "stat_11": 0.074219,
      "stat_12": 0.607733,
      "stat_13": 0.47729,
      "stat_14": 1.25,
      "stat_15": 1.25,
      "stat_16": 0.22,
      "stat_17": 1.25
    }
  }
]
```

### Turn 5 — `execute_in_remote_docker` (`58810 ms` LLM)
- **Rationale**: Test the sine map against both public prediction thresholds and check its numerical inverse. Separately profile the sine frequency using only calibration pairs to see whether the remaining error is structural.
- **Alternatives Weighed**: `Use the archived polynomial calibration unchanged` (rejected: It did not match the sine basis on leave-one-out calibration error.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine basis suffices for panel A once gross outliers are removed.` | Witness: `After excluding three gross outliers, mean leave-one-out panel-A residual drops from 0.29136 to 0.01436 with sine terms; panel B retains a maximum absolute affine residual of 0.000251.` | Blanket Diagnosis: `The missing nonlinear terms affect the shared coordinate reconstruction upstream of both responses.` | Sublated Pivot: `Add sine terms only to panel A, invert the fitted forward map, and retain the affine panel-B map and both frozen consumers.`

```bash
cat > /holdout_task/node_00.py <<'PY'
"""Recover the shared coordinates from independent calibration panels."""
import numpy as np


def _affine_features(u):
    return np.column_stack((np.ones(len(u)), u))


def _panel_a_features(u, frequency):
    return np.column_stack((_affine_features(u), np.sin(frequency * u)))


def _fit_panel_a(u, v, frequency):
    affine = _affine_features(u)
    initial = np.linalg.lstsq(affine, v, rcond=None)[0]
    residual = np.linalg.norm(affine @ initial - v, axis=1)
    keep = residual < max(0.002, 4.0 * np.median(residual))
    features = _panel_a_features(u, frequency)
    for _ in range(8):
        coef = np.linalg.lstsq(features[keep], v[keep], rcond=None)[0]
        residual = np.linalg.norm(features @ coef - v, axis=1)
        update = residual < max(0.002, 4.0 * np.median(residual))
        if update.sum() < features.shape[1] or np.array_equal(update, keep):
            break
        keep = update
    return coef, np.linalg.lstsq(affine[keep], v[keep], rcond=None)[0]


def _invert_panel_a(x, coef, affine, frequency):
    d = x.shape[1]
    u = np.linalg.solve(affine[1:].T, (x - affine[0]).T).T
    for _ in range(25):
        residual = _panel_a_features(u, frequency) @ coef - x
        error = np.linalg.norm(residual, axis=1)
        if np.max(error, initial=0.0) < 1e-12:
            break
        jac = coef[None, 1:d+1, :] + (frequency * np.cos(frequency * u))[..., None] * coef[None, d+1:, :]
        step = np.linalg.solve(jac.transpose(0, 2, 1), residual[..., None])[..., 0]
        step /= np.maximum(1.0, np.max(np.abs(step), axis=1, keepdims=True) / 0.5)
        for _ in range(12):
            trial = u - step
            improve = np.linalg.norm(_panel_a_features(trial, frequency) @ coef - x, axis=1) < error
            if np.all(improve | (error < 1e-12)):
                break
            step[~improve] *= 0.5
        u = np.where(improve[:, None], trial, u)
    return u


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = np.asarray(study['train']['x'], dtype=float)
    test = np.asarray(study['test']['x'], dtype=float)
    a = study['panel_a']
    frequency = float(study['telemetry_vec'][1])
    coef, affine = _fit_panel_a(np.asarray(a['u'], float), np.asarray(a['v'], float), frequency)
    u = _invert_panel_a(np.vstack((train, test)), coef, affine, frequency)
    b = study['panel_b']
    mapping = np.linalg.lstsq(_affine_features(b['u']), b['v'], rcond=None)[0]
    coordinates = _affine_features(u) @ mapping
    return {'z': coordinates[:len(train)].copy(), 'zt': coordinates[len(train):].copy()}
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/test_dag_candidate.py
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task'); import node_00 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v')); k=s['telemetry_vec'][1]
c,a=m._fit_panel_a(u,v,k); e=np.linalg.norm(m._panel_a_features(u,k)@c-v,axis=1); g=e<5
print('fixed-frequency coefficients',np.round(c,5), 'sorted residuals',np.round(np.sort(e),5))
for tag,F,w in [('common',lambda p: np.sin(u*p),[k]),('separate',lambda p: np.sin(u*p),[k]*4),('with cos',lambda p:np.c_[np.sin(u*p),np.cos(u*p)],[k]*4)]:
 def res(p):
  A=np.c_[m._affine_features(u),F(p)]; return (A[g]@np.linalg.lstsq(A[g],v[g],rcond=None)[0]-v[g]).ravel()
 fit=least_squares(res,w,ftol=1e-13,gtol=1e-13,xtol=1e-13); print(tag, fit.x, np.linalg.norm(fit.fun),np.max(np.abs(fit.fun)))
x=np.vstack((s['train']['x'],s['test']['x'])); ur=m._invert_panel_a(x,c,a,k); print('roundtrip max',np.max(np.abs(m._panel_a_features(ur,k)@c-x)))
PY
```
- **Container Exit Code**: `0` (`572 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.014912, "public_cv_traj_mae": 0.018447}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}
fixed-frequency coefficients [[ 1.26930e-01 -1.11300e-01  6.68400e-02  1.42130e-01]
 [ 1.02151e+00 -1.38000e-03  1.14760e-01  2.43000e-03]
 [ 4.46000e-03  1.14815e+00 -3.56000e-03  1.09340e-01]
 [ 1.09560e-01  1.50000e-04  9.69090e-01  6.20000e-04]
 [-1.01000e-03  1.14330e-01  2.64000e-03  9.33470e-01]
 [ 8.58800e-02  2.03000e-03 -4.00940e-01 -4.15000e-03]
 [-3.55000e-03  8.29700e-02  2.58000e-03 -3.98170e-01]
 [-3.96090e-01  2.60000e-04  8.16300e-02 -1.04000e-03]
 [-2.20000e-04 -3.98980e-01 -1.24000e-03  8.41800e-02]] sorted residuals [1.880000e-03 2.910000e-03 3.630000e-03 3.940000e-03 4.310000e-03
 4.670000e-03 4.870000e-03 5.850000e-03 7.010000e-03 7.210000e-03
 7.980000e-03 8.200000e-03 8.220000e-03 8.660000e-03 8.870000e-03
 9.140000e-03 9.340000e-03 9.800000e-03 1.032000e-02 1.033000e-02
 1.043000e-02 1.082000e-02 1.097000e-02 1.128000e-02 1.149000e-02
 1.149000e-02 1.200000e-02 1.209000e-02 1.246000e-02 1.260000e-02
 1.305000e-02 1.306000e-02 1.317000e-02 1.407000e-02 1.460000e-02
 1.513000e-02 1.550000e-02 1.568000e-02 1.849000e-02 1.856000e-02
 1.901000e-02 1.956000e-02 2.154000e-02 2.194000e-02 2.554000e-02
 1.199708e+01 1.225930e+01 1.235572e+01]
common [2.43782132] 0.04867955238169977 0.00874162979359805
separate [2.454586   2.41948834 2.44671296 2.41277807] 0.04847855717678807 0.00872719849785153
with cos [2.45430845 2.40559967 2.44366888 2.41511574] 0.04733420423233405 0.008043068083663285
roundtrip max 4.440892098500626e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=aa085b332651f727 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_aa085b33 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`28314 ms` LLM)
- **Rationale**: The current node passes both public gates. Verify it in an isolated process, then stress the inverse by leaving out each calibration row and measure out-of-fold errors at large latent radii and across trajectory horizons.
- **Alternatives Weighed**: `Refit a separate sine frequency for each coordinate` (rejected: It reduces the calibration residual norm only from 0.04868 to 0.04848 relative to a fitted shared frequency; test the current candidate's outer-slice errors before adding parameters.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/test_dag_candidate.py
python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, '/holdout_task/' + path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
m = load('candidate', 'node_00.py')
n1 = load('frozen1', '_frozen_nodes/node_01.py')
n2 = load('frozen2', '_frozen_nodes/node_02.py')
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
a = s['panel_a']; u, v = a['u'], a['v']; k = float(s['telemetry_vec'][1]); x = np.vstack((s['train']['x'], s['test']['x']))
c, affine = m._fit_panel_a(u,v,k)
r = m._invert_panel_a(x,c,affine,k)
errs = []
for i in range(len(u)):
    mask = np.arange(len(u)) != i
    ci, ai = m._fit_panel_a(u[mask],v[mask],k)
    ri = m._invert_panel_a(x,ci,ai,k)
    errs.append((np.max(np.abs(m._panel_a_features(ri,k)@ci-x)), np.max(np.abs(ri-r))))
print('calibration jackknife worst (observed-space round trip, latent displacement):', np.max(errs,axis=0))
assert np.max(np.asarray(errs)[:,0]) < 1e-9
tr = s['train']; n = len(tr['x']); pred = np.zeros_like(tr['y']); traj = np.zeros_like(tr['traj'])
for va in np.array_split(np.random.default_rng(31).permutation(n), 4):
    fit = np.setdiff1d(np.arange(n),va); ys = tr['y'][fit]; ts = tr['traj'][fit]
    sub = dict(s, train=dict(x=tr['x'][fit],y=ys,traj=ts,summary=np.array([ys.mean(),ys.std(),ts[:,-1].mean(),ts[:,-1].std()])), test={'x':tr['x'][va]})
    inp = {'__input__':sub}
    out = n2.run_node({'node_00':m.run_node(inp),'node_01':n1.run_node(inp)})
    pred[va] = out['y_hat']; traj[va] = out['state_pred']
    print('fold MAE', np.mean(abs(pred[va]-tr['y'][va])), np.mean(abs(traj[va]-tr['traj'][va])))
for label,mask in [('all',np.ones(n,bool)),('outer latent-radius quartile',np.linalg.norm(r[:n],axis=1)>=np.quantile(np.linalg.norm(r[:n],axis=1),.75))]:
    ey = np.mean(abs(pred[mask]-tr['y'][mask])); et = abs(traj[mask]-tr['traj'][mask]).mean(0)
    print(label,'OOF y and trajectory MAE',ey,et.mean(),'worst horizon',et.max())
    assert np.isfinite(ey) and np.all(np.isfinite(et))
PY
```
- **Container Exit Code**: `0` (`494 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.014912, "public_cv_traj_mae": 0.018447}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}
calibration jackknife worst (observed-space round trip, latent displacement): [5.99520433e-15 1.62684445e-02]
fold MAE 0.015161621397094927 0.019092621799554746
fold MAE 0.013546562838609076 0.014380177274978471
fold MAE 0.018267494979288423 0.022765790698729464
fold MAE 0.010837583011545027 0.016712125288892075
all OOF y and trajectory MAE 0.014453315556634363 0.018237678765538688 worst horizon 0.025689983575581506
outer latent-radius quartile OOF y and trajectory MAE 0.02017848761081394 0.021729217708472748 worst horizon 0.03065848708123971

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=aa085b332651f727 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_aa085b33 (directory).
```

### Turn 7 — `submit_completed_task` (`11871 ms` LLM)
- **Rationale**: The hash-bound artifact passed isolated-process verification without changing either frozen node. Across all calibration leave-one-out fits, the worst observed-space round-trip error was 6.0×10⁻¹⁵; the outer-radius out-of-fold check also stayed below 0.038 at every trajectory horizon.
- **Alternatives Weighed**: `Fit separate sine frequencies for each coordinate` (rejected: Separate frequencies offered only a marginal calibration improvement over a fitted common frequency, and the verified candidate passed both public gates with margin.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_00.py`. The public DAG test accepted it: scalar MAE was 0.014912 and trajectory MAE was 0.018447, both below 0.038. Each node executed exactly once.`

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
