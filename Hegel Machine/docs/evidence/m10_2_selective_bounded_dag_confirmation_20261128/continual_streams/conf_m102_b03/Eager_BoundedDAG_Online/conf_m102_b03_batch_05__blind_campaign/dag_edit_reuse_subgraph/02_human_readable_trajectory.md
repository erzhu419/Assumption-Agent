# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m102_b03_batch_05__blind_campaign__reuse_subgraph`

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
| **Total Tokens (Prompt / Cached / Out)** | `201,324` (`195,106` / `101,376` / `6,218`) |
| **Total Episode Wall Time (s)** | `193.4s` |

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

### Turn 1 — `execute_in_remote_docker` (`8803 ms` LLM)
- **Rationale**: Inspect the frozen contracts, public scoring definition, and archived candidates before changing the editable node.

```bash
cd /holdout_task; cat _dag_spec.json public_asset_index.json; cat _frozen_nodes/*.py; cat test_dag_candidate.py; ls -la /tmp/hegel_continual_archive; find /tmp/hegel_continual_archive -maxdepth 3 -type f | head -100
```
- **Container Exit Code**: `0` (`336 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
= len(st["test"]["x"]), len(st["eval_times"])
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
drwxr-xr-x 2 root root 4096 Nov 28 15:10 .
drwxrwxrwt 1 root root 4096 Nov 28 15:10 ..
-rw-r--r-- 1 root root  667 Nov 28 15:10 archive_manifest.json
-rw-r--r-- 1 root root 2517 Nov 28 15:10 conf_m102_b03_batch_00_solver.py
-rw-r--r-- 1 root root 1651 Nov 28 15:10 pre_llm_gate_diagnostic.json
/tmp/hegel_continual_archive/archive_manifest.json
/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
/tmp/hegel_continual_archive/conf_m102_b03_batch_00_solver.py
```

### Turn 2 — `execute_in_remote_docker` (`13839 ms` LLM)
- **Rationale**: The first inspection was truncated. Read the frozen handoff, archived solver, gate diagnostics, and public arrays in a compact pass before choosing a model.

```bash
cd /holdout_task; cat _frozen_nodes/node_01.py _frozen_nodes/node_02.py; cat /tmp/hegel_continual_archive/*; python3 - <<'PY'
import pickle,numpy as np
b=pickle.load(open('public_batch.pkl','rb'));print('batch',len(b))
def show(x,p=''):
 if isinstance(x,dict):
  for k,v in x.items():show(v,p+'/'+k)
 elif isinstance(x,np.ndarray):
  print(p,x.shape,x.dtype,'range',np.nanmin(x),np.nanmax(x),'first',x[:min(len(x),3)])
 else:print(p,repr(x)[:500])
show(b[0])
PY
```
- **Container Exit Code**: `0` (`397 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
71127 -0.66070089  0.15669352 -2.14971943]
 [-1.3119954   0.67734357 -0.33269193  0.71887409]
 [-2.32649877  0.61176621  1.04938601 -0.80864248]]
/study/panel_b/u (24, 4) float64 range -1.2362636301822425 1.1827453379279982 first [[ 0.14413493 -0.29402077 -0.8014562  -0.15324534]
 [ 0.38744535  0.8441116  -0.69211191 -0.52597976]
 [-1.01842671 -0.47247326 -0.52972133 -0.2800914 ]]
/study/panel_b/v (24, 4) float64 range -1.3571434840275896 1.4120053721252774 first [[-0.02173335  0.53179758 -0.30288823 -0.08274296]
 [-0.28971594  0.34387124  0.44996407 -0.78449158]
 [ 0.44564653 -0.16833347 -0.92635252 -0.29154021]]
/study/panel_c/u (24, 4) float64 range -1.2244050340401715 1.244976225249716 first [[ 0.41474094  0.73284424 -1.22440503  1.04585734]
 [-1.02310357 -0.77407634 -0.97754817  0.26356452]
 [ 0.60192158  1.05289899 -1.02348584 -0.29773532]]
/study/panel_c/v1 (24, 4) float64 range -1.9363686653193755 1.804389796609644 first [[-1.08536557  0.45932886  0.78782694  1.49842685]
 [-0.21630143 -1.38655313 -0.36456362  0.39897501]
 [-0.95176097 -0.14077653  1.26488708 -0.47407243]]
/study/panel_c/v2 (24, 4) float64 range -1.4440222939456897 1.8274278788017304 first [[ 0.30881164 -0.98291414  0.22662637 -1.08684788]
 [ 1.21342617 -0.29887134 -0.64393027  0.5976329 ]
 [ 1.44736429 -0.1580949   0.29608172  0.09075326]]
/study/train/x (72, 4) float64 range -2.155334691185456 1.604165832165529 first [[ 1.47151319  0.25407703  0.09039272 -0.51464257]
 [-0.23313925 -0.81712935 -0.75027837  0.60686038]
 [-0.62874621 -1.49412807  0.31708906  0.07455845]]
/study/train/y (72,) float64 range -2.052765709883216 2.312552882654539 first [0.72830498 0.14364139 1.01362423]
/study/train/traj (72, 8) float64 range -2.99338549610697 3.0528226533966816 first [[-0.42526591 -0.94862436 -1.32226292 -1.59502408 -1.81521196 -2.00239323
  -2.15537692 -2.28104253]
 [-0.20390347 -0.38188675 -0.47463464 -0.5057192  -0.53781735 -0.53390416
  -0.53864516 -0.53003556]
 [-0.04467542 -0.12704864 -0.17829791 -0.22719625 -0.28683335 -0.30788212
  -0.34717341 -0.3743193 ]]
/study/train/summary (4,) float64 range -0.15065124742701339 1.236540372081598 first [ 0.1857965   0.96183767 -0.15065125]
/study/test/x (36, 4) float64 range -2.317492060984384 1.5907376411167926 first [[ 0.70014677 -0.63032833  0.49812877 -0.52852606]
 [ 0.14477231 -0.05085386 -1.2135372   0.12557398]
 [-0.69778358 -0.05946471  0.03943973  1.04658018]]
/study/telemetry_vec (4,) float64 range 0.22 1.25 first [1.25 1.25 1.25]
```

### Turn 3 — `execute_in_remote_docker` (`15563 ms` LLM)
- **Rationale**: Recover the archived branch and readout formulas without truncation, then distinguish calibration curvature from outliers and check the trajectory rank numerically.

```bash
cat /tmp/hegel_continual_archive/*solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust reference panel alignment nonlinear calibration variable projection' || true; cd /holdout_task; python3 - <<'PY'
import pickle,numpy as np
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']
for key in ['panel_a','panel_b','panel_c']:
 p=s[key]; u=p['u']; print(key)
 for name,v in p.items():
  if name=='u':continue
  for j in range(4):
   print(name,j,'corr',np.round(np.corrcoef(np.c_[u,v[:,j]].T)[-1,:4],3))
   if key=='panel_a':
    i=np.argsort(u[:,j]); print(np.round(np.c_[u[i,j],v[i,j]][::3],4).tolist())
print('trajectory singular values',np.linalg.svd(s['train']['traj'],compute_uv=False))
PY
```
- **Container Exit Code**: `0` (`492 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
panel_a
v 0 corr [ 0.572 -0.119 -0.073 -0.131]
[[-1.2886, -2.7304], [-1.1052, -0.7547], [-0.6972, -1.312], [-0.6474, -1.0209], [-0.5025, -1.026], [-0.4127, -0.2507], [-0.2845, -0.8267], [-0.1874, -0.5902], [-0.0726, -0.093], [0.1528, -0.1891], [0.1945, 0.2899], [0.4974, 0.6417], [0.5932, 0.6147], [0.6322, 0.6291], [0.7537, 0.9127], [0.86, 1.0574]]
v 1 corr [-0.161  0.487 -0.201 -0.07 ]
[[-1.2752, -1.4739], [-0.9672, -0.8722], [-0.8023, -1.0984], [-0.6115, -0.6607], [-0.5797, -1.1451], [-0.5359, -0.5027], [-0.3939, -0.6415], [-0.3285, -0.1389], [-0.1698, -0.1076], [0.0934, 0.0703], [0.3844, 0.2935], [0.6862, 0.6547], [0.7894, 0.7157], [0.8404, 0.711], [0.9366, 6.9612], [1.135, 1.15]]
v 2 corr [ 0.072 -0.043  0.635 -0.183]
[[-1.2143, -1.3867], [-1.1505, -2.1951], [-0.8922, -1.6985], [-0.7722, -0.6689], [-0.7375, -1.5727], [-0.5395, -0.6156], [-0.5082, -0.9801], [-0.3585, -0.897], [-0.1945, 0.1567], [-0.0584, -0.3327], [-0.0052, -0.0957], [0.1193, -0.1339], [0.2455, 0.3586], [0.4193, 0.333], [0.8293, 0.9408], [1.1987, 1.6087]]
v 3 corr [-0.277 -0.089 -0.102  0.584]
[[-1.2818, -2.1497], [-1.1135, -1.8779], [-0.8235, -6.8094], [-0.7014, -0.5037], [-0.645, -0.7926], [-0.3398, -0.1955], [-0.2565, -0.3444], [-0.07, -0.3081], [0.1027, 0.3527], [0.3939, 0.3686], [0.5118, 0.8284], [0.6256, 0.9378], [0.6739, 0.8212], [0.7711, 0.9708], [0.8275, 1.0947], [1.0806, 1.4888]]
panel_b
v 0 corr [-0.563  0.084  0.167  0.705]
v 1 corr [ 0.713  0.108 -0.85   0.21 ]
v 2 corr [0.653 0.695 0.072 0.467]
v 3 corr [ 0.255 -0.628  0.519  0.511]
panel_c
v1 0 corr [-0.192  0.074  0.073 -0.104]
v1 1 corr [-0.17   0.048 -0.359  0.289]
v1 2 corr [-0.2   -0.199 -0.183 -0.013]
v1 3 corr [ 0.101  0.158 -0.112  0.303]
v2 0 corr [-0.114  0.023  0.072 -0.047]
v2 1 corr [0.063 0.317 0.373 0.06 ]
v2 2 corr [-0.189  0.078 -0.111  0.061]
v2 3 corr [ 0.248 -0.22   0.232 -0.025]
trajectory singular values [22.84299262  2.53048472  0.04671815  0.04536954  0.04030377  0.03778416
  0.03478323  0.03306584]
```

### Turn 4 — `execute_in_remote_docker` (`25554 ms` LLM)
- **Rationale**: Measure archived transfer and use robust panel residuals to distinguish cross-channel coupling, nonlinear calibration, and gross contamination.

```bash
cat > /tmp/dag_probe.py <<'PY'
import pickle, numpy as np, importlib.util
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
p='/tmp/hegel_continual_archive/conf_m102_b03_batch_00_solver.py'; spec=importlib.util.spec_from_file_location('old',p); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def ls(A,y):return np.linalg.lstsq(A,y,rcond=None)[0]
def ones(x):return np.c_[np.ones(len(x)),x]
def poly(u,d):return np.column_stack([np.ones(len(u))]+[np.prod(u[:,c],axis=1) for k in range(1,d+1) for c in combinations_with_replacement(range(u.shape[1]),k)])
def robust(A,y):
 p=least_squares(lambda z:A@z-y,ls(A,y),loss='soft_l1',f_scale=.06).x
 r=np.abs(A@p-y); good=r<min(1.5,max(.035,5*np.median(r))); return ls(A[good],y[good]),good
if __name__=='__main__':
 for name in ['panel_a','panel_b','panel_c']:
  p=s[name];u=p['u'];print(name)
  for k,v in p.items():
   if k=='u':continue
   for d in [1,2,3]:
    A=poly(u,d); rs=[]
    for j in range(4):
     if name=='panel_a': z,g=robust(A,v[:,j]); r=np.abs(A@z-v[:,j]);rs.append(np.round(np.quantile(r,[.5,.75,.85]),5));
     else:r=np.abs(A@ls(A,v[:,j])-v[:,j]);rs.append(np.round(np.quantile(r,[.5,.75,.85]),5))
    print(k,d,rs)
   if name=='panel_a':
    for name2,A in [('linear',ones(u)),('diag3',np.c_[ones(u),u**2,u**3]),('diag5',np.c_[ones(u),u**2,u**3,u**4,u**5])]:
     for j in range(4):
      z,g=robust(A,v[:,j]);r=A@z-v[:,j]; print(name2,j,'coef',np.round(z,3),'inliers',sum(g),'abs residual percentiles',np.round(np.quantile(abs(r),[.25,.5,.7,.8,.85,.9,.95]),4))
 if True:
  p=s['panel_c'];u=p['u']
  for j in [1,2]:
   v=p['v'+str(j)]; r=old._branch(p,u,j)-v; print('archive branch',j,np.max(abs(r)),np.sqrt(np.mean(r*r)))
 ts=s['train']; y,t=old._predict(s,slice(52),ts['x'][52:]);print('archive cv',np.mean(abs(y-ts['y'][52:])),np.mean(abs(t-ts['traj'][52:])))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/dag_probe.py
```
- **Container Exit Code**: `0` (`834 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
9 -0.08   0.083  1.172 -0.071 -0.008  0.013 -0.121  0.084
  0.066 -0.074  0.201] inliers 42 abs residual percentiles [0.0248 0.0498 0.0975 0.1496 0.1681 0.2752 4.0481]
diag5 0 coef [-0.004  1.176 -0.276 -0.097  0.323 -0.625  0.404  0.099 -0.24  -0.247
 -0.529 -0.044 -0.907  0.583 -0.286 -0.039  0.158  0.45   0.291  0.075
  0.494] inliers 40 abs residual percentiles [0.0131 0.0395 0.0788 0.1222 0.2988 0.3374 3.7904]
diag5 1 coef [-0.143 -0.137  0.748 -0.393 -0.2    0.519 -0.239 -0.271 -0.016  0.835
  0.727  0.455  0.329 -0.492  0.203  0.196 -0.058 -0.706 -0.336 -0.371
 -0.168] inliers 42 abs residual percentiles [0.0168 0.0534 0.0792 0.1282 0.1735 0.4558 4.1031]
diag5 2 coef [-0.097 -0.081 -0.015  0.962 -0.543  0.159 -0.02  -0.271  0.019  0.359
 -0.021  0.918  0.3   -0.18   0.089  0.123 -0.085 -0.28   0.03  -0.487
 -0.163] inliers 40 abs residual percentiles [0.0187 0.0465 0.0827 0.1129 0.2117 0.3336 4.0594]
diag5 3 coef [ 0.018 -0.119 -0.074 -0.317  1.084 -0.077  0.301 -0.168 -0.036 -0.107
  0.138  1.077  0.455 -0.048 -0.148  0.16  -0.113  0.015 -0.067 -0.602
 -0.142] inliers 41 abs residual percentiles [0.0155 0.0437 0.0684 0.0846 0.1333 0.4387 4.0416]
panel_b
v 1 [array([8.0e-05, 1.1e-04, 1.1e-04]), array([5.0e-05, 1.2e-04, 1.4e-04]), array([5.0e-05, 8.0e-05, 1.3e-04]), array([6.e-05, 8.e-05, 9.e-05])]
v 2 [array([3.e-05, 5.e-05, 6.e-05]), array([2.e-05, 4.e-05, 7.e-05]), array([6.0e-05, 9.0e-05, 1.1e-04]), array([4.e-05, 8.e-05, 9.e-05])]
v 3 [array([0., 0., 0.]), array([0., 0., 0.]), array([0., 0., 0.]), array([0., 0., 0.])]
panel_c
v1 1 [array([0.43546, 0.72018, 1.05641]), array([0.39963, 0.73051, 0.82217]), array([0.51786, 0.71046, 0.92231]), array([0.53906, 0.82908, 0.96078])]
v1 2 [array([4.e-05, 8.e-05, 1.e-04]), array([3.e-05, 5.e-05, 7.e-05]), array([4.0e-05, 9.0e-05, 1.2e-04]), array([5.e-05, 8.e-05, 9.e-05])]
v1 3 [array([0., 0., 0.]), array([0., 0., 0.]), array([0., 0., 0.]), array([0., 0., 0.])]
v2 1 [array([0.47775, 0.89864, 1.03403]), array([0.39775, 0.61066, 0.66282]), array([0.59011, 0.75237, 0.86454]), array([0.48973, 0.72182, 0.82516])]
v2 2 [array([0.08346, 0.12803, 0.16575]), array([0.06541, 0.13934, 0.1839 ]), array([0.0671 , 0.13035, 0.19219]), array([0.06551, 0.10605, 0.12398])]
v2 3 [array([0., 0., 0.]), array([0., 0., 0.]), array([0., 0., 0.]), array([0., 0., 0.])]
archive branch 1 1.3317149178130978 0.5654467800561295
archive branch 2 0.00019154664867515603 7.320476172422635e-05
archive cv 0.7468899352379126 0.8254496698014469
```

### Turn 5 — `execute_in_remote_docker` (`37260 ms` LLM)
- **Rationale**: Recover calibration from independent panels, then measure scalar and trajectory errors separately on the official split and four complementary folds.
- **Alternatives Weighed**: `Refit only the archived readout` (rejected: The reference-panel errors show that its input features are already misspecified.)
- **ATMS NOGOOD DECLARED**: Refuted `The archived diagonal affine calibration and softsign first branch transfer unchanged.` | Witness: `The archived first branch has panel RMSE 0.56545, whereas a quadratic basis reaches median absolute residuals below 0.00005; coupled cubic calibration also reaches near-noise residuals on two channels before explicit outlier removal.` | Blanket Diagnosis: `The mismatch is already visible in reference panels upstream of readout and time projection.` | Sublated Pivot: `Recover a sparse coupled cubic calibration after gross-outlier screening, invert it, and test the quadratic first branch alongside the archived cosine second branch.`

```bash
cat > /tmp/dag_fit.py <<'PY'
from dag_probe import *
from itertools import combinations_with_replacement
rng=np.random.default_rng(91)
def terms(d): return [()]+[c for k in range(1,d+1) for c in combinations_with_replacement(range(4),k)]
def Aof(u,tt): return np.column_stack([np.prod(u[:,c],axis=1) for c in tt])
def cal(p):
 u,v=p['u'],p['v']; tt=terms(3); A=Aof(u,tt); B=[]
 for j in range(4):
  y=v[:,j]; cy=(j+1)%4; h0=np.abs(y-ls(ones(u),y)@ones(u).T)<2.5
  # Gross corruptions are orders of magnitude larger than calibration noise.
  h=h0.copy(); b=ls(A[h],y[h]); print('cal initial',j,sum(h),np.quantile(abs(A[h]@b-y[h]),[.5,1]),[(c,round(w,6)) for c,w in zip(tt,b) if abs(w)>.001]);
  for thresh in [.01,.003,.001]:
   keep=abs(b)>thresh; b2=np.zeros(len(tt)); b2[keep]=ls(A[h][:,keep],y[h]);
   if np.max(abs(A[h]@b2-y[h]))<.001:b=b2
  B.append(b)
 return tt,np.array(B).T
def inv(x,tt,B):
 from scipy.optimize import root
 z0=(x-B[0])@np.linalg.inv(B[1:5]);return np.array([root(lambda z: Aof(z[None],tt)[0]@B-y,z).x for z,y in zip(z0,x)])
def fitness(F,y,key,splits):
 o=[]
 for i,j in splits:
  p=F[j]@ls(F[i],y[i]);o.append(np.mean(abs(p-y[j])))
 return key,F.shape[1],np.round(o,5)
if __name__=='__main__':
 tt,B=cal(s['panel_a']); p=s['panel_c']; u=p['u']; train=s['train']; z=inv(train['x'],tt,B); print('inverse',np.max(abs(Aof(z,tt)@B-train['x'])),np.max(abs(z)),z[:2]);
 A=poly(u,2); bw=ls(A,p['v1']);print('branch1 coefficients',[(c,np.round(w,5).tolist()) for c,w in zip(terms(2),bw)]);c=poly(z,2)@bw; d=old._branch(p,z,2);
 print('branch2 amplitudes range', np.min(d,axis=0),np.max(d,axis=0)); q=s['panel_b']; bw=ls(ones(q['u']),q['v']); print('panelB',np.round(bw,6)); zm=ones(z)@bw;
 dot=np.sum(c*d,axis=1); cross=(d[:,:,None]*c[:,None,:]).reshape(len(z),-1); zz=np.c_[c,d]; print('telemetry',s['telemetry_vec']);
 Fs={'arch scalar':ones(np.c_[zz,dot]),'arch traj':ones(np.c_[zz,cross]),'arch+z':ones(np.c_[z,zz,cross]),'polyz2':poly(z,2),'polyz3':poly(z,3),'polyc+d2':poly(zz,2),'diag c+d':ones(np.c_[zz,c*d,zz**2]),'atan c+d':ones(np.c_[zz,np.arctan(1.25*zz),cross]),'tanh c+d':ones(np.c_[zz,np.tanh(1.25*zz),cross]),'cubic c+d':ones(np.c_[zz,zz**2,zz**3,cross]),'cubic+dot':ones(np.c_[zz,zz**2,zz**3,dot]),'sin c+d':ones(np.c_[zz,np.sin(1.25*zz),cross]),'rat c+d':ones(np.c_[zz,zz/(1+1.25*zz**2),cross]),'softsign c+d':ones(np.c_[zz,zz/(1+1.25*abs(zz)),cross])};
 ids=np.arange(len(z)); splits=[(ids[:52],ids[52:])]; splits.extend((np.setdiff1d(ids,j),j) for j in np.array_split(rng.permutation(ids),4));
 for k,f in Fs.items(): print(fitness(f,train['y'],k,splits),fitness(f,train['traj'],k,splits)[2]);
 np.savez('/tmp/dag_features.npz',z=z,c=c,d=d,zm=zm,y=train['y'],t=train['traj'])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/dag_fit.py
```
- **Container Exit Code**: `0` (`623 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
), [-7e-05, 2e-05, -2e-05, -3e-05]), ((3,), [0.00014, 3e-05, 6e-05, 4e-05]), ((0, 0), [-5e-05, 2e-05, 0.0, 4e-05]), ((0, 1), [-0.80513, -0.89512, 0.1844, 0.01435]), ((0, 2), [0.00011, -4e-05, -5e-05, -2e-05]), ((0, 3), [-0.80812, 0.58428, -0.69694, 0.06022]), ((1, 1), [0.00013, 7e-05, -0.00017, -0.0001]), ((1, 2), [0.42686, -0.58302, -0.97467, 0.10615]), ((1, 3), [-6e-05, 3e-05, -0.0001, -7e-05]), ((2, 2), [-8e-05, 0.00013, 3e-05, -2e-05]), ((2, 3), [-0.01221, -0.03261, -0.11788, -1.21188]), ((3, 3), [-2e-05, 4e-05, -9e-05, -3e-05])]
branch2 amplitudes range [-1.55807213 -1.33896239 -1.19553125 -1.19912672] [1.41899134 1.23976738 1.21079342 1.61453702]
panelB [[ 0.120028 -0.106927  0.077724  0.077864]
 [-0.516697  0.423138  0.473336  0.323105]
 [ 0.112307 -0.108236  0.62714  -0.597298]
 [-0.090475 -0.726344  0.275128  0.403586]
 [ 0.697476  0.236671  0.285292  0.387846]]
telemetry [1.25 1.25 1.25 0.22]
('arch scalar', 10, array([0.00415, 0.0039 , 0.0039 , 0.00245, 0.00429])) [0.0632  0.0632  0.06202 0.09986 0.08336]
('arch traj', 25, array([0.0047 , 0.0048 , 0.00493, 0.00633, 0.00484])) [0.0062  0.00611 0.00565 0.00585 0.0056 ]
('arch+z', 29, array([0.0052 , 0.00578, 0.00484, 0.00699, 0.0051 ])) [0.00678 0.00628 0.0068  0.00641 0.0063 ]
('polyz2', 15, array([0.34253, 0.3091 , 0.28221, 0.28019, 0.35913])) [0.23187 0.22204 0.22214 0.25384 0.25216]
('polyz3', 35, array([0.5081 , 0.32271, 0.60267, 0.60474, 0.47455])) [0.27979 0.26046 0.39217 0.3993  0.30069]
('polyc+d2', 45, array([0.00778, 0.00923, 0.02118, 0.00985, 0.01227])) [0.02256 0.01391 0.01851 0.02031 0.02245]
('diag c+d', 21, array([0.00538, 0.00436, 0.00495, 0.00421, 0.00603])) [0.08544 0.0726  0.08938 0.10664 0.13349]
('atan c+d', 33, array([0.00555, 0.00531, 0.00457, 0.00556, 0.00563])) [0.00854 0.00887 0.00703 0.00695 0.00729]
('tanh c+d', 33, array([0.00539, 0.00526, 0.00459, 0.00555, 0.00581])) [0.00859 0.00892 0.00704 0.007   0.00732]
('cubic c+d', 41, array([0.0083 , 0.00559, 0.00672, 0.00779, 0.00883])) [0.01709 0.00972 0.01177 0.0127  0.00981]
('cubic+dot', 26, array([0.00537, 0.00533, 0.00442, 0.00344, 0.00595])) [0.09727 0.1137  0.07031 0.08341 0.17762]
('sin c+d', 33, array([0.00525, 0.00476, 0.00496, 0.00655, 0.00684])) [0.01158 0.00798 0.00714 0.00848 0.00863]
('rat c+d', 33, array([0.00635, 0.00526, 0.00477, 0.00553, 0.0052 ])) [0.00841 0.00785 0.00718 0.00672 0.00706]
('softsign c+d', 33, array([0.00657, 0.00525, 0.00477, 0.00612, 0.00518])) [0.00932 0.0072  0.0073  0.00687 0.00765]
```

### Turn 6 — `execute_in_remote_docker` (`38196 ms` LLM)
- **Rationale**: The calibration panel identifies five terms per channel, and the first branch panel identifies four cyclic pair products plus an intercept. Fit those terms from the panels, retain the readouts that passed the prototype folds, and test the exact artifact in an isolated Python process.
- **Alternatives Weighed**: `Fit a dense polynomial directly to calibrated coordinates` (rejected: The dense cubic prototype reached 0.603 scalar MAE on its worst fold, compared with 0.00429 for the branch-based readout.)

```bash
set -e
cat > /holdout_task/node_00.py <<'PY'
"""Recover calibrated coordinates and predict the two branch readouts."""
import numpy as np
from scipy.optimize import least_squares, root


def _ls(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _intercept(features):
    return np.column_stack((np.ones(len(features)), features))


def _calibration_features(z, neighbor):
    return _intercept(np.column_stack((z, neighbor, z * neighbor, z**3)))


def _calibrate(panel, observed):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    width = u.shape[1]
    right = (np.arange(width) + 1) % width
    coefficients = []
    for j, k in enumerate(right):
        design = _calibration_features(u[:, j], u[:, k])
        y = v[:, j]
        robust = least_squares(lambda p: design @ p - y, _ls(design, y),
                               loss='soft_l1', f_scale=.02).x
        error = design @ robust - y
        center = np.median(error)
        mad = np.median(np.abs(error - center))
        clean = np.abs(error - center) <= max(.025, 8 * mad)
        coefficients.append(_ls(design[clean], y[clean]) if clean.sum() >= 5 else robust)
    offset, gain, coupling, product, cubic = np.asarray(coefficients).T
    linear = np.diag(gain)
    linear[np.arange(width), right] = coupling
    starts = np.linalg.solve(linear, (observed - offset).T).T
    result = []
    for value, initial in zip(observed, starts):
        def residual(z):
            return offset + gain*z + (coupling + product*z)*z[right] + cubic*z**3 - value

        def jacobian(z):
            jac = np.diag(gain + product*z[right] + 3*cubic*z**2)
            jac[np.arange(width), right] += coupling + product*z
            return jac

        fit = root(residual, initial, jac=jacobian, tol=1e-10)
        z = fit.x
        if np.max(np.abs(residual(z))) > 1e-8:
            alternate = least_squares(residual, z, jac=jacobian).x
            if np.linalg.norm(residual(alternate)) < np.linalg.norm(residual(z)):
                z = alternate
        result.append(z)
    return np.asarray(result).reshape(observed.shape)


def _branches(panel, z):
    u = np.asarray(panel['u'])

    def pairs(x):
        return _intercept(x * np.roll(x, -1, axis=1))

    first = pairs(z) @ _ls(pairs(u), panel['v1'])
    target = np.asarray(panel['v2'])

    def cosine(x, p):
        return _intercept(np.column_stack((x, np.cos(p[0] * x + p[1:]))))

    def residual(p):
        design = cosine(u, p)
        return (design @ _ls(design, target) - target).ravel()

    parameters = least_squares(residual, np.r_[1.27, np.zeros(u.shape[1])],
                               max_nfev=150).x
    second = cosine(z, parameters) @ _ls(cosine(u, parameters), target)
    return first, second


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    n = len(train['x'])
    measured = np.vstack((train['x'], study['test']['x']))
    calibrated = _calibrate(study['panel_a'], measured)
    first, second = _branches(study['panel_c'], calibrated)
    cross = second[:, :, None] * first[:, None, :]
    scalar = _intercept(np.column_stack((first, second, np.trace(cross, axis1=1, axis2=2))))
    dynamic = _intercept(np.column_stack((first, second, cross.reshape(len(measured), -1))))
    return {'y_hat': scalar[n:] @ _ls(scalar[:n], train['y']),
            'unprojected': dynamic[n:] @ _ls(dynamic[:n], train['traj'])}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`607 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.004167, "public_cv_traj_mae": 0.004401}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=32b902766072a137 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_32b90276 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`22524 ms` LLM)
- **Rationale**: The official isolated run accepted the artifact. Stress-test the same node and frozen sink on held-out coordinate tails, where calibration inversion and branch extrapolation face a stronger test than on random folds.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np

base = Path('/holdout_task')
def load(relative):
    path = base / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_node

fit, basis, sink = [load(p) for p in ('node_00.py', '_frozen_nodes/node_01.py', '_frozen_nodes/node_02.py')]
batch = pickle.loads((base / 'public_batch.pkl').read_bytes())
rng = np.random.default_rng(91)
worst = [0., 0.]
for entry in batch:
    s = entry['study']
    x = s['train']['x']
    n = len(x)
    indices = np.arange(n)
    folds = [(f'random_{j}', a) for j, a in enumerate(np.array_split(rng.permutation(n), 4))]
    q = n // 5
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j])
        folds.extend([(f'x{j}_low', order[:q]), (f'x{j}_high', order[-q:])])
    for name, heldout in folds:
        rows = np.setdiff1d(indices, heldout)
        case = dict(s, train=dict(s['train'], **{k: s['train'][k][rows] for k in ('x', 'y', 'traj')}), test=dict(s['test'], x=x[heldout]))
        a = fit({'__input__': case})
        b = basis({'__input__': case})
        out = sink({'node_00': a, 'node_01': b})
        actual_y, actual_t = (s['train'][k][heldout] for k in ('y', 'traj'))
        assert out['y_hat'].shape == actual_y.shape and out['state_pred'].shape == actual_t.shape
        errors = [float(np.mean(np.abs(out[k]-actual))) for k, actual in [('y_hat', actual_y), ('state_pred', actual_t)]]
        assert np.all(np.isfinite(errors))
        worst = np.maximum(worst, errors)
        print(name, *(f'{e:.6f}' for e in errors))
print('worst scalar and projected trajectory MAE:', *worst)
assert max(worst) <= .038
PY
```
- **Container Exit Code**: `0` (`796 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
random_0 0.003916 0.005284
random_1 0.003886 0.004177
random_2 0.002478 0.004219
random_3 0.004345 0.004718
x0_low 0.003820 0.004356
x0_high 0.004005 0.005595
x1_low 0.004042 0.004196
x1_high 0.004455 0.004637
x2_low 0.003861 0.005198
x2_high 0.004468 0.004103
x3_low 0.004026 0.003972
x3_high 0.004585 0.005705
worst scalar and projected trajectory MAE: 0.004585101650414048 0.005704876502754874

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=32b902766072a137 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_32b90276 (directory).
```

### Turn 8 — `submit_completed_task` (`10745 ms` LLM)
- **Rationale**: The unchanged candidate passed isolated official verification with each node executing once. All random and coordinate-tail folds passed both MAE thresholds on the same artifact.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_00.py` without changing the frozen nodes. The official test accepted it: scalar MAE 0.004167 and trajectory MAE 0.004401. Across 12 stress-test folds, worst MAEs were 0.004585 and 0.005705, both below 0.038.`

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
