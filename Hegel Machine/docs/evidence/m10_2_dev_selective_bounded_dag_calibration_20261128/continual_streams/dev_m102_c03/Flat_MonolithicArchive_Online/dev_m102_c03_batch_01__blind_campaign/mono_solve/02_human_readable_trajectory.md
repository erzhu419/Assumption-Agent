# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_m102_c03_batch_01__blind_campaign`

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
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `455,731` (`441,597` / `143,232` / `14,134`) |
| **Total Episode Wall Time (s)** | `418.99s` |

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

### Turn 1 — `execute_in_remote_docker` (`9442 ms` LLM)
- **Rationale**: Inspect the public contract and authorized campaign archive before fitting or selecting operators.

```bash
ls -lah /holdout_task /tmp/hegel_continual_archive; cat /tmp/hegel_continual_archive/archive_manifest.json; cat /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; cat /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust reference panel alignment ridge nuisance profiling'
```
- **Container Exit Code**: `0` (`285 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
```

### Turn 2 — `execute_in_remote_docker` (`10973 ms` LLM)
- **Rationale**: Inspect the prior solver’s mathematical assumptions and the new panel and trajectory scales before testing transfer or a different model.

```bash
cat /tmp/hegel_continual_archive/dev_m102_c03_batch_00_solver.py; cat /holdout_task/public_asset_index.json; python3 - <<'PY'
import numpy as n,pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n.set_printoptions(precision=5,suppress=True,linewidth=160)
for k,v in s.items():
 print(k)
 for j,a in (v.items() if isinstance(v,dict) else [('',v)]):
  print(j,a.shape, 'min',a.min(axis=0),'max',a.max(axis=0),'std',a.std(axis=0),'first',a[:5])
PY
```
- **Container Exit Code**: `0` (`346 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
3202 0.62268] first [[ 1.10722 -0.10254  1.24779  0.37937]
 [-0.88432 -0.75002  0.80247  0.32326]
 [ 0.75425  0.66497  0.76978 -0.14489]
 [-0.1793  -0.66658 -0.28028 -0.56633]
 [ 0.77047 -1.11606  0.37229  1.2201 ]]
v1 (24, 4) min [-1.17063 -1.3718  -1.72467 -0.90863] max [1.37375 1.71469 0.83683 0.7969 ] std [0.5971  0.66661 0.5926  0.33134] first [[-0.20841 -0.72796 -0.33727  0.07204]
 [-1.17063  0.27397 -0.0427  -0.06346]
 [ 0.02331  0.15706  0.78229  0.03535]
 [-0.02955 -0.22786  0.1528   0.019  ]
 [ 0.19212 -1.22276 -1.21746  0.4859 ]]
v2 (24, 4) min [-0.80441 -0.90129 -0.95468 -0.46695] max [0.50683 0.30143 0.56674 1.11337] std [0.3206  0.34092 0.41392 0.35779] first [[-0.45763 -0.78526 -0.5696  -0.46695]
 [ 0.38869 -0.21088 -0.14745 -0.22671]
 [-0.42832 -0.75937  0.01824  0.00607]
 [-0.05902  0.21497  0.28345 -0.04904]
 [-0.23853 -0.04314 -0.95468  0.13263]]
train
x (72, 4) min [-1.08775 -0.96705 -1.43305 -1.34539] max [1.29346 1.18119 1.18421 1.31727] std [0.67287 0.58714 0.777   0.69979] first [[ 0.27236 -0.53827 -0.86697 -0.67354]
 [ 0.53917 -0.27601  0.1736  -0.91538]
 [ 1.2159   1.18119 -1.02874 -0.84069]
 [ 0.9344   0.31591  0.24252  0.61165]
 [ 0.72359  0.5031  -0.89736 -0.97984]]
y (72,) min -1.265591811490961 max 1.7599724199369133 std 0.5404924942841038 first [0.0301  0.0348  1.75997 0.58533 0.82501]
traj (72, 8) min [-0.72973 -1.43235 -1.81894 -2.0416  -2.19816 -2.28673 -2.35964 -2.41542] max [0.77518 1.4891  1.84749 2.10729 2.26956 2.39784 2.48528 2.56714] std [0.26206 0.50793 0.64219 0.71938 0.76869 0.80418 0.83038 0.85283] first [[ 0.09092  0.1888   0.26277  0.29985  0.32989  0.35658  0.38738  0.39833]
 [ 0.03273  0.06419  0.1063   0.15701  0.18303  0.22279  0.23963  0.25973]
 [-0.45512 -0.74469 -0.80394 -0.7459  -0.65192 -0.55498 -0.45146 -0.35831]
 [ 0.21923  0.44347  0.57305  0.65262  0.71619  0.77001  0.80624  0.8246 ]
 [-0.09643 -0.14514 -0.09686 -0.02155  0.05346  0.13301  0.19688  0.25472]]
summary (4,) min 0.020300088734214048 max 0.8516953507623751 std 0.2969812672482417 first [0.0203  0.5397  0.47477 0.8517 ]
test
x (36, 4) min [-1.09642 -0.78471 -1.46556 -1.29972] max [1.28358 1.15678 1.18886 1.18316] std [0.75797 0.64763 0.78078 0.67177] first [[ 0.70922  0.35646 -0.70355 -0.72689]
 [-0.95874 -0.0101  -0.30142 -0.94485]
 [ 1.07127 -0.52445 -1.17046  0.5468 ]
 [ 1.08264 -0.68226  0.3468  -0.84222]
 [-0.87579 -0.22122 -1.44848 -0.39556]]
telemetry_vec
 (4,) min 0.22 max 1.25 std 0.4460030829489859 first [0.22 1.25 0.22 0.22]
```

### Turn 3 — `execute_in_remote_docker` (`20294 ms` LLM)
- **Rationale**: Characterize each panel separately and estimate trajectory rank and time constants before fitting spatial operators.

```bash
cat /holdout_task/public_asset_index.json; tail -35 /tmp/hegel_continual_archive/dev_m102_c03_batch_00_solver.py; cat >/tmp/probe.py <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; tr=s['train']; Y=tr['traj']; x=tr['x']; y=tr['y']; print('t',t,'traj sv',np.linalg.svd(Y,compute_uv=False)); np.set_printoptions(precision=4,suppress=True)
for pk in ['panel_a','panel_b','panel_c']:
 p=s[pk]; u=p['u']; Z=np.c_[np.ones(len(u)),u];
 for k,v in p.items():
  if k=='u':continue
  c=np.linalg.lstsq(Z,v,rcond=None)[0]; e=v-Z@c; print(pk,k,'affine',c,'res',np.sort(np.linalg.norm(e,axis=1))); keep=np.linalg.norm(e,axis=1)<3; print('refit',np.linalg.lstsq(Z[keep],v[keep],rcond=None)[0]); print('nonlinear residual correlations',np.corrcoef(np.c_[e,u**2,u**3,np.array([u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]).T].T)[:4,4:]);
def fit(r,YY=Y):
 q=-np.expm1(-t[:,None]*np.exp(r)); a=np.linalg.lstsq(q,YY.T,rcond=None)[0].T;return q,a,(YY-a@q.T)
for r in ([.4,1.8],[.2,1,3],[.12,.55,1.4,4]):
 f=least_squares(lambda z:fit(z)[2].ravel(),np.log(r),bounds=(-5,4)); q,a,e=fit(f.x); print('time rates',np.exp(f.x),'MAE',np.mean(abs(e)),'coeff std',a.std(0),'corr',np.corrcoef(np.c_[x,y,a].T)[5:,:5]);
print('affine x-> outputs',np.linalg.lstsq(np.c_[np.ones(len(x)),x],np.c_[y,Y],rcond=None)[0]); print('summary',tr['summary'],[y.mean(),y.std(),Y[:,-1].mean(),Y[:,-1].std()]);
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe.py
```
- **Container Exit Code**: `0` (`809 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[-0.2483 -0.1061 -0.2314 -0.0127  0.1542  0.0571 -0.0778  0.0109  0.7968
  -0.1851  0.0295  0.538   0.1194 -0.0813]
 [ 0.0783  0.2745 -0.2329 -0.0127  0.0217 -0.0565 -0.272   0.1001  0.0045
  -0.0324  0.4528  0.2489 -0.0368 -0.2122]]
panel_c v2 affine [[-0.1244 -0.1583  0.0299  0.1558]
 [-0.4358 -0.0656 -0.1901 -0.0715]
 [-0.0884 -0.3426  0.1162  0.3461]
 [ 0.1916 -0.3866 -0.1054 -0.3093]
 [ 0.0516 -0.0154 -0.518   0.2565]] res [0.0787 0.1011 0.1034 0.1065 0.108  0.1124 0.1149 0.1205 0.1229 0.1323
 0.1353 0.144  0.1642 0.1705 0.175  0.1755 0.1874 0.1922 0.2029 0.2302
 0.2371 0.2387 0.2736 0.3011]
refit [[-0.1244 -0.1583  0.0299  0.1558]
 [-0.4358 -0.0656 -0.1901 -0.0715]
 [-0.0884 -0.3426  0.1162  0.3461]
 [ 0.1916 -0.3866 -0.1054 -0.3093]
 [ 0.0516 -0.0154 -0.518   0.2565]]
nonlinear residual correlations [[-0.769  -0.1667 -0.1751  0.314   0.0536  0.0577  0.053   0.0517  0.3228
  -0.0783 -0.0642 -0.126  -0.0693  0.2237]
 [-0.5486 -0.3744 -0.6395  0.0884  0.0188 -0.0003 -0.1132  0.0572  0.4153
  -0.0644  0.0918 -0.0308  0.0715  0.2684]
 [-0.1803 -0.0875 -0.212  -0.6037  0.0305 -0.0702 -0.1191 -0.0044  0.0235
  -0.22    0.0241  0.229   0.0804 -0.1842]
 [-0.3381  0.755  -0.7133  0.6529 -0.0239 -0.0322 -0.1966  0.22    0.1061
  -0.2764  0.1938 -0.0043 -0.2946  0.1425]]
time rates [0.4039 1.8145] MAE 0.0032332231704483693 coeff std [0.5882 0.6763] corr [[ 0.3178  0.714  -0.1199 -0.447   0.1954]
 [ 0.1437  0.0674 -0.1093  0.1207 -0.1293]]
time rates [0.4027 1.8423 7.6401] MAE 0.0027624636896912305 coeff std [0.5871 0.6776 0.015 ] corr [[ 0.3171  0.7114 -0.1234 -0.4463  0.1895]
 [ 0.1477  0.0742 -0.1068  0.1197 -0.1229]
 [-0.1521 -0.1861 -0.0231 -0.029  -0.1077]]
time rates [ 0.2698  1.5087  1.511  10.8487] MAE 0.0023758089759472944 coeff std [  0.5872 110.1934 110.7226   0.0176] corr [[ 0.3249  0.6764 -0.1401 -0.4314  0.1709]
 [-0.0517  0.2197  0.0954 -0.2398  0.1908]
 [ 0.0525 -0.2176 -0.0956  0.239  -0.1904]
 [-0.0661  0.1196  0.0166 -0.0406 -0.0389]]
affine x-> outputs [[-0.0042  0.0852  0.1697  0.2243  0.263   0.2889  0.3112  0.3278  0.3419]
 [ 0.3313  0.076   0.1618  0.2195  0.2606  0.2916  0.3194  0.3398  0.3565]
 [ 0.1356  0.1126  0.2494  0.3566  0.4381  0.5062  0.5619  0.6075  0.6491]
 [ 0.1831 -0.0471 -0.0953 -0.1246 -0.1446 -0.1611 -0.1694 -0.1789 -0.1879]
 [ 0.0101  0.0019 -0.0156 -0.0422 -0.0746 -0.0995 -0.1272 -0.1503 -0.1698]]
summary [0.0203 0.5397 0.4748 0.8517] [0.019532927766414316, 0.5404924942841038, 0.4754522747560117, 0.852826479700582]
```

### Turn 4 — `execute_in_remote_docker` (`27610 ms` LLM)
- **Rationale**: Separate the spatial prediction error from the near-noise-level temporal reconstruction error. Use panel leave-one-out residuals and spatial baselines to identify which algebraic terms warrant testing.

```bash
cat >/tmp/baseline.py <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_a']; u=p['u']; v=p['v']; U=np.c_[np.ones(len(u)),u]; keep=np.ones(len(u),bool)
for _ in range(5):
 a=np.linalg.lstsq(U[keep],v[keep],rcond=None)[0]; e=np.linalg.norm(v-U@a,axis=1); keep=e<max(.08,3*np.median(e))
np.set_printoptions(precision=5,suppress=True); print('panel A',keep.sum(),a, 'res',np.sort(e));
x=s['train']['x']; z=(x-a[0])@np.linalg.inv(a[1:]); t=s['eval_times']; tr=s['train']; n=52; y=tr['y']; Y=tr['traj']; q=-np.expm1(-t[:,None]*np.array([.4,1.82])); A=np.linalg.lstsq(q,Y.T,rcond=None)[0].T; T=np.c_[y,A]; print('times',t); print('A corr',np.corrcoef(T.T)); print('z ranges',z.min(0),z.max(0))
for pk,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[pk]; print(pk,k)
 for d in (1,2):
  U=PolynomialFeatures(d).fit_transform(p['u']); a=np.linalg.lstsq(U,p[k],rcond=None)[0]; r=p[k]-U@a; loo=r/(1-np.sum(np.linalg.qr(U)[0]**2,axis=1))[:,None]; print('deg',d,'fit',np.sqrt(np.mean(r*r,axis=0)),'loo',np.sqrt(np.mean(loo*loo,axis=0))); print(np.round(a[np.argsort(np.max(abs(a),axis=1))[-8:]],3))
R=[]
def add(name,K,ls):
 for l in ls:
  P=K[n:,:n]@np.linalg.solve(K[:n,:n]+l*np.eye(n),T[:n]); e=np.mean(abs(P-T[n:]),axis=0); h=np.mean(abs((P[:,1:]-T[n:,1:])@q.T)); R.append((name,l,*e,h))
for zz in (x,z):
 for d in range(1,6):
  F=PolynomialFeatures(d).fit_transform(zz); add('p'+str(d),F@F.T,[1e-6,.001,.01,.1,1.])
 for k in [.25,.5,1,2,4,8]:
  d=cdist(zz,zz); add('rbf'+str(k),np.exp(-(d/k)**2),[1e-6,.001,.01,.1]); h=np.sqrt(5)*d/k; add('matern'+str(k),(1+h+h*h/3)*np.exp(-h),[1e-6,.001,.01,.1])
print('ranked y',sorted(R,key=lambda e:e[2])[:10]); print('ranked traj',sorted(R,key=lambda e:e[-1])[:10]); print('poly coefficient patterns:')
p=PolynomialFeatures(2); F=p.fit_transform(z); a=np.linalg.lstsq(F,T,rcond=None)[0]; print(list(zip(p.get_feature_names_out(),a))); print('poly residual stdev',np.std(T-F@a,axis=0)); print('y vs A quadratic res',np.std(y-PolynomialFeatures(2).fit_transform(A)@np.linalg.lstsq(PolynomialFeatures(2).fit_transform(A),y,rcond=None)[0]))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/baseline.py
```
- **Container Exit Code**: `0` (`713 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
14533831373), ('matern4', 0.001, 0.025656891923841872, 0.05651563814332082, 0.06133266835833261, 0.04225693525565784), ('rbf4', 0.001, 0.026418133207342688, 0.04818038098810533, 0.027337665595055377, 0.039279225745689275), ('rbf4', 0.001, 0.028669792023371176, 0.04936267740235824, 0.03247639272361059, 0.033648333756542685), ('matern8', 1e-06, 0.029156779267176424, 0.04594485373443759, 0.026083536855672267, 0.02255108803519506), ('rbf4', 1e-06, 0.030195192163662298, 0.04915229411796192, 0.05490259970571114, 0.0353543594114442), ('p4', 0.1, 0.03522151332845559, 0.07217039622011238, 0.1046501442154462, 0.07132845134560198)]
ranked traj [('matern8', 1e-06, 0.029156779267176424, 0.04594485373443759, 0.026083536855672267, 0.02255108803519506), ('matern8', 1e-06, 0.023303070869056662, 0.047889799835849504, 0.03514280050665985, 0.02307414533831373), ('rbf4', 1e-06, 0.03683821034016712, 0.04646928292545944, 0.042488203004513364, 0.02386160896388927), ('matern4', 1e-06, 0.02147216278782033, 0.048873847042941274, 0.04500543148828144, 0.02714953424617082), ('rbf8', 1e-06, 0.03771688797640023, 0.03183674479071766, 0.034827290494638556, 0.033065575393743385), ('matern4', 0.001, 0.022174792905724, 0.05215231947390596, 0.04014627833649512, 0.03330233812252185), ('rbf4', 0.001, 0.028669792023371176, 0.04936267740235824, 0.03247639272361059, 0.033648333756542685), ('rbf8', 1e-06, 0.0410711120213608, 0.02756512208444184, 0.03693957333791513, 0.03525598708251475), ('rbf4', 1e-06, 0.030195192163662298, 0.04915229411796192, 0.05490259970571114, 0.0353543594114442), ('p3', 0.01, 0.04858420029216581, 0.0416074240230458, 0.04524629951420929, 0.03825992588761047)]
poly coefficient patterns:
[('1', array([-0.05285,  0.31244,  0.12767])), ('x0', array([0.40876, 0.3485 , 0.0923 ])), ('x1', array([0.0456 , 0.71655, 0.17025])), ('x2', array([ 0.18827, -0.16828, -0.06177])), ('x3', array([ 0.0809 , -0.43782,  0.01756])), ('x0^2', array([0.15259, 0.17286, 0.01341])), ('x0 x1', array([ 0.07729, -0.08728, -1.19075])), ('x0 x2', array([-0.01178,  0.01386, -0.00861])), ('x0 x3', array([-0.05437,  0.00505,  1.07597])), ('x1^2', array([-0.05734,  0.28634,  0.00803])), ('x1 x2', array([-0.49818,  0.11061,  0.12164])), ('x1 x3', array([0.04748, 0.06055, 0.03903])), ('x2^2', array([ 0.08421, -0.09586, -0.02921])), ('x2 x3', array([ 1.04477, -0.05208,  0.03567])), ('x3^2', array([ 0.03919, -0.18702,  0.01089]))]
poly residual stdev [0.10287 0.06297 0.08643]
y vs A quadratic res 0.49543232814029947
```

### Turn 5 — `execute_in_remote_docker` (`30079 ms` LLM)
- **Rationale**: Save and check a viable baseline, then inspect panel coefficients and cubic residual structure to identify a model that improves on the gate without relying on a single split.
- **Alternatives Weighed**: `Use a quadratic surrogate alone` (rejected: Its training residual standard deviations of 0.063–0.103 show that it misses substantial structure.)

```bash
cat >/holdout_task/batch_solver.py <<'PY'
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares

def calibrate(study):
    u,v=(np.asarray(study['panel_a'][k]) for k in ('u','v'))
    keep=np.ones(len(u),dtype=bool)
    for _ in range(6):
        um=u[keep].mean(0); vm=v[keep].mean(0)
        gain=np.sum((u[keep]-um)*(v[keep]-vm),axis=0)/np.sum((u[keep]-um)**2,axis=0)
        off=vm-um*gain
        e=np.linalg.norm(v-off-u*gain,axis=1)
        keep=e<=max(.001,3*np.median(e))
    return off,gain

def temporal(t,Y):
    def profile(r):
        q=-np.expm1(-t[:,None]*r)
        a=np.linalg.lstsq(q,Y.T,rcond=None)[0].T
        return q,a
    def residual(r):
        q,a=profile(r)
        return (a@q.T-Y).ravel()
    return profile(least_squares(residual,[.4,1.82],bounds=([.1,.9],[.85,4.])).x)

def solve_batch(study):
    tr=study['train']; n=len(tr['x']); off,gain=calibrate(study)
    x=np.vstack([tr['x'],study['test']['x']]); z=(x-off)/gain
    q,a=temporal(np.asarray(study['eval_times']),np.asarray(tr['traj']))
    target=np.column_stack([tr['y'],a]); pred=[]
    for zz,sc,reg in ((x,4.,1e-6),(z,4.,1e-6),(z,8.,1e-6)):
        h=np.sqrt(5)*cdist(zz,zz[:n])/sc
        k=(1+h+h*h/3)*np.exp(-h)
        pred.append(k[n:]@np.linalg.solve(k[:n]+reg*np.eye(n),target))
    y=.5*(pred[0][:,0]+pred[1][:,0]); state=pred[2][:,1:]@q.T
    return dict(y_hat=y,state_pred=state,diagnostics_vec=np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()]))
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py
cat >/tmp/structure.py <<'PY'
import sys,pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
sys.path.insert(0,'/holdout_task'); from batch_solver import calibrate,temporal
np.set_printoptions(precision=4,suppress=True,linewidth=200)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; off,gain=calibrate(s); z=(tr['x']-off)/gain
q,a=temporal(s['eval_times'],tr['traj']); Y=np.c_[tr['y'],a]
for pkey,vkey,deg in [('panel_b','v',1),('panel_c','v1',2),('panel_c','v2',2)]:
 p=s[pkey]; f=PolynomialFeatures(deg); F=f.fit_transform(p['u']); c=np.linalg.lstsq(F,p[vkey],rcond=None)[0]; print(pkey,vkey); print(list(zip(f.get_feature_names_out(),c))); print('fit to outputs',np.linalg.lstsq(np.c_[np.ones(len(z)),f.transform(z)@c],Y,rcond=None)[0]);
for deg in (2,3,4):
 f=PolynomialFeatures(deg); F=f.fit_transform(z); c=np.linalg.lstsq(F[:52],Y[:52],rcond=None)[0]; print('degree',deg,'CV',np.mean(abs(F[52:]@c-Y[52:]),0));
 if deg==3:
  for j in range(3): print(sorted(zip(f.get_feature_names_out(),c[:,j]),key=lambda a:-abs(a[1]))[:22])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/structure.py
```
- **Container Exit Code**: `0` (`976 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
0725 0.0972]
degree 3 CV [0.0552 0.0366 0.0529]
[('x2 x3', 0.9998037591716038), ('x1 x2', -0.4973698521284618), ('x0', 0.33504840142366843), ('x0 x1 x2', -0.30308571976031173), ('x2^2 x3', 0.2403258885884661), ('x2', 0.22995414828700736), ('x1 x2 x3', -0.1966978828922919), ('x0 x2 x3', 0.18285568553894316), ('x0^2', 0.1731389889368177), ('x0 x1^2', 0.13004644013677652), ('x0 x3^2', 0.12857827985603776), ('x2^2', 0.12344758747731346), ('x0^2 x1', 0.12018744356336843), ('x2 x3^2', -0.08691869947088023), ('x0 x2^2', -0.08588194731205848), ('1', -0.07377199635997611), ('x3', 0.057439895951780985), ('x3^3', -0.05134125741498563), ('x0 x1 x3', 0.04641835112519346), ('x1', -0.040105223778538736), ('x1 x2^2', 0.0389770021988181), ('x1^2', -0.03766501797452556)]
[('x1', 0.6671905871398257), ('x3', -0.4497829760240821), ('x0', 0.44153124645819675), ('1', 0.3130200175129542), ('x1^2', 0.3021393120827766), ('x2', -0.2351961269253292), ('x3^2', -0.21518170061232797), ('x0^2', 0.16636931177164416), ('x1 x2^2', 0.14483763149922452), ('x0 x1', -0.12345477031998014), ('x0 x1^2', -0.11681504430085356), ('x2^2 x3', -0.11162021163818789), ('x2^2', -0.10054061710617034), ('x1^2 x2', 0.09692416096518139), ('x0 x1 x3', 0.09046430655291182), ('x0 x1 x2', -0.07463976484049809), ('x1 x2', 0.0710604044731485), ('x3^3', 0.05812457518274965), ('x0 x2', -0.052826118902055616), ('x0 x3^2', -0.043555528200288504), ('x0^2 x2', 0.03971897526182476), ('x0^2 x1', -0.039615251615800026)]
[('x0 x1', -1.2591846390174972), ('x0 x3', 1.1167348965968262), ('x0 x1 x2', -0.24044801736107202), ('x2^2 x3', 0.20210429842749075), ('x1 x2 x3', -0.1591477246693741), ('x0 x2 x3', 0.15902719907152094), ('x1 x2', 0.12332777110614472), ('x0^2 x1', 0.1149553739347387), ('1', 0.11380029736994687), ('x0 x2^2', -0.10188068017153121), ('x0 x1^2', 0.10082205937332503), ('x0 x3^2', 0.09256933676751516), ('x1', 0.08069235275608609), ('x2 x3^2', -0.07757865918642244), ('x0', 0.059668675271679475), ('x1 x2^2', 0.05427462701136831), ('x0 x1 x3', 0.05360528188262205), ('x3^3', -0.039754791619807675), ('x1^2 x3', 0.03429453834239739), ('x0^2 x2', -0.030853280540319876), ('x2^3', -0.028845932760841055), ('x1^2 x2', 0.02533195111208179)]
degree 4 CV [0.0531 0.0979 0.0948]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7e2c0a67e442fbba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7e2c0a67 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`26532 ms` LLM)
- **Rationale**: The scalar target and fast mode have similar cubic coefficients. Test for cancellation after removing their quadratic terms, and check whether the auxiliary panel identifies the missing nonlinear operator.

```bash
cat >/tmp/modes.py <<'PY'
import pickle,sys,numpy as np
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task'); from batch_solver import calibrate,temporal
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; b,g=calibrate(s); z=(tr['x']-b)/g; q,a=temporal(s['eval_times'],tr['traj']); Y=np.c_[tr['y'],a]; n=52
E=PolynomialFeatures(2).fit_transform(z); C=np.linalg.lstsq(E,Y,rcond=None)[0]; R=Y-E@C
print('post quadratic residual corr',np.corrcoef(R.T),'sv',np.linalg.svd(R,full_matrices=False)[1:]); print('pairwise cancellation',np.std(R[:,2]-.8*R[:,0])); print('linear mix B singular',np.linalg.svd(np.linalg.lstsq(np.c_[np.ones(24),s['panel_b']['u']],s['panel_b']['v'],rcond=None)[0][1:])[1]);
p=s['panel_c']; u=p['u']; v=p['v2']; P=PolynomialFeatures(2).fit_transform(u); H=np.c_[np.ones(len(u)),u,u*u];
for j in range(4):
 w=v[:,j]; print('v2 col',j)
 for name,F in [('linear',H[:,:5]),('sep2',H),('full2',P),('sep3',np.c_[H,u**3]),('sep4',np.c_[H,u**3,u**4]),('sepR',np.c_[np.ones(len(u)),u/(1+.4*u*u),u*u/(1+.4*u*u)]),('sepTrig',np.c_[np.ones(len(u)),np.sin(u),np.cos(u)])]:
  c=np.linalg.lstsq(F,w,rcond=None)[0]; e=w-F@c; hh=np.sum(np.linalg.qr(F)[0]**2,1); print(name,round(np.sqrt(np.mean((e/(1-hh))**2)),6))
 for fun,ifi in [(np.tanh,np.arctanh),(lambda x:x/np.sqrt(1+x*x),lambda x:x/np.sqrt(1-x*x)),(np.sin,np.arcsin)]:
  best=None
  for mat in [H[:,:5],H,P]:
   init=np.linalg.lstsq(mat,ifi(np.clip(w/1.3,-.99,.99)),rcond=None)[0]; fit=least_squares(lambda a:a[0]+np.exp(a[1])*fun(mat@a[2:])-w,np.r_[0,np.log(1.3),init],max_nfev=200); err=np.sqrt(np.mean(fit.fun**2)); print(getattr(fun,'__name__','f'),len(init),round(err,6),np.round(fit.x[:2],3))
E3=PolynomialFeatures(3).fit_transform(z); c=np.linalg.lstsq(E3,Y,rcond=None)[0]; r=Y-E3@c; print('cubic residual',np.std(r,0),np.corrcoef(r.T));
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/modes.py
```
- **Container Exit Code**: `0` (`1483 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
post quadratic residual corr [[1.         0.24353561 0.99037437]
 [0.24353561 1.         0.30627892]
 [0.99037437 0.30627892 1.        ]] sv (array([1.14972441, 0.50893713, 0.06853882]), array([[-0.75468485, -0.15581276, -0.63731716],
       [ 0.16647268, -0.98507707,  0.04370378],
       [ 0.63461613,  0.07311332, -0.7693613 ]]))
pairwise cancellation 0.012485727050077758
linear mix B singular [0.8800622  0.88002054 0.87998567 0.87994004]
v2 col 0
linear 0.098645
sep2 0.007135
full2 0.008861
sep3 0.015115
sep4 0.0015
sepR 0.06504
sepTrig 0.038273
tanh 5 0.065459 [-8.227  2.223]
tanh 9 0.003777 [-0.148  0.736]
tanh 15 0.002286 [-0.233  0.65 ]
<lambda> 5 0.065454 [-6.585  2.085]
<lambda> 9 0.003776 [-0.148  0.932]
<lambda> 15 0.002288 [-0.232  0.846]
sin 5 0.065584 [-11.24    2.466]
sin 9 0.003781 [-0.148  0.408]
sin 15 0.002279 [-0.235  0.325]
v2 col 1
linear 0.102348
sep2 0.008838
full2 0.012049
sep3 0.008888
sep4 0.00226
sepR 0.06658
sepTrig 0.034961
tanh 5 0.064004 [-0.634  0.089]
tanh 9 0.005206 [-0.3    1.483]
tanh 15 0.003471 [2.203 1.969]
<lambda> 5 0.064182 [-0.626  0.232]
<lambda> 9 0.005207 [-0.3    1.685]
<lambda> 15 0.00347 [2.164 2.139]
sin 5 0.063263 [-0.651 -0.089]
sin 9 0.005206 [-0.301  1.138]
sin 15 0.003474 [2.172 1.685]
v2 col 2
linear 0.083304
sep2 0.005206
full2 0.007743
sep3 0.01435
sep4 0.003135
sepR 0.063862
sepTrig 0.039266
tanh 5 0.04755 [-0.395  0.172]
tanh 9 0.002956 [1.456 2.648]
tanh 15 0.00157 [-2.543  2.085]
<lambda> 5 0.047701 [-0.391  0.322]
<lambda> 9 0.002956 [1.43 2.84]
<lambda> 15 0.001569 [-2.377  2.232]
sin 5 0.047344 [-0.425  0.017]
sin 9 0.002956 [1.54  2.334]
sin 15 0.001573 [-2.841  1.851]
v2 col 3
linear 0.158905
sep2 0.00718
full2 0.013112
sep3 0.013649
sep4 0.001911
sepR 0.049788
sepTrig 0.027206
tanh 5 0.115875 [-28.394   3.427]
tanh 9 0.004372 [0.349 1.47 ]
tanh 15 0.002858 [0.186 0.759]
<lambda> 5 0.115638 [-31.962   3.559]
<lambda> 9 0.004372 [0.349 1.672]
<lambda> 15 0.002854 [0.183 0.944]
sin 5 0.116419 [-25.991   3.319]
sin 9 0.004371 [0.35  1.126]
sin 15 0.00287 [0.197 0.465]
cubic residual [0.0187665  0.01711861 0.01546398] [[ 1.         -0.15958286  0.86497976]
 [-0.15958286  1.         -0.35717228]
 [ 0.86497976 -0.35717228  1.        ]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7e2c0a67e442fbba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7e2c0a67 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`32382 ms` LLM)
- **Rationale**: Recover the separable auxiliary law from its own panel before testing compositions on the spatial targets.
- **ATMS NOGOOD DECLARED**: Refuted `An affine-plus-quadratic spatial basis is sufficient for all three outputs.` | Witness: `After removing the full quadratic basis, scalar and fast-mode residuals correlate at 0.9904 and have standard deviations of about 0.103 and 0.086.` | Blanket Diagnosis: `The basis omits a structured contribution shared by the scalar and fast trajectory channels.` | Sublated Pivot: `Keep the calibrated coordinates and quadratic terms while testing rational panel maps and composed operators.`

```bash
cat >/tmp/dag.py <<'PY'
import sys,pickle,numpy as np
from itertools import combinations
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task'); from batch_solver import calibrate,temporal
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; off,gain=calibrate(s); z=(tr['x']-off)/gain; q,a=temporal(s['eval_times'],tr['traj']); T=np.c_[tr['y'],a]; n=52
pairs=list(combinations(range(4),2)); cyc=[(0,1),(0,3),(1,2),(2,3)]
def lin(u): return np.c_[np.ones(len(u)),u]
def cp(u,ps): return np.column_stack([u[:,i]*u[:,j] for i,j in ps])
def c1(u): return np.c_[np.ones(len(u)),cp(u,cyc)]
p=s['panel_c']; b=s['panel_b']; B=np.linalg.lstsq(lin(b['u']),b['v'],rcond=None)[0]; C=np.linalg.lstsq(c1(p['u']),p['v1'],rcond=None)[0]; u=p['u']; v=p['v2']; candidates=[]
for nm,fn,x0,lb,ub in [('rat2',lambda x,k:x*x/(1+k[0]*x*x),[.25],[0],[4]),('rat1',lambda x,k:x*x/(1+k[0]*x+k[1]*x*x),[0,.25],[-.4,0],[.4,4]),('cos',lambda x,k:(1-np.cos(k[0]*x))/k[0]**2,[.6],[.03],[4]),('gauss',lambda x,k:-np.expm1(-k[0]*x*x)/k[0],[.2],[.001],[4]),('tanh',lambda x,k:x*np.tanh(k[0]*x)/k[0],[.6],[.01],[4])]:
 def f(x,k): return np.c_[lin(x),fn(x,k)]
 def err(k):
  F=f(u,k); return (F@np.linalg.lstsq(F,v,rcond=None)[0]-v).ravel()
 opt=least_squares(err,x0,bounds=(lb,ub)); F=f(u,opt.x); W=np.linalg.lstsq(F,v,rcond=None)[0]; print(nm,opt.x,np.std(opt.fun),np.round(W[5:]/W[1:5],3)); candidates.append((np.std(opt.fun),lambda x,f=f,k=opt.x,W=W:f(x,k)@W))
D=min(candidates,key=lambda a:a[0])[1]; ops={'B':lambda x:lin(x)@B,'C':lambda x:c1(x)@C,'D':D}; nodes={'Z':z}; basis={'Z':z};
for _ in range(3):
 for name,x in list(nodes.items()):
  if len(name)!=_+1:continue
  for k,fn in ops.items():
   if k=='B' and name[-1]=='B':continue
   nodes[name+k]=fn(x)
for key,x in nodes.items():
 basis[key]=x; basis[key+'p']=cp(x,pairs); basis[key+'s']=x*x; basis[key+'t']=np.tanh(x); basis[key+'r']=x/(1+.22*x*x)
F0=np.c_[lin(z),z*z,cp(z,pairs)]; base=F0
for stage in range(4):
 scores=[]
 for k,v in basis.items():
  f=np.c_[base,v]; c=np.linalg.lstsq(f[:n],T[:n],rcond=1e-8)[0]; e=f@c-T; scores.append((k,np.mean(abs(e[n:]),0),np.mean(abs(e[:n]),0),np.mean(abs((e[n:,1:])@q.T))))
 for j in range(3): print(stage,j,[(k,np.round(val,4),np.round(train,4),round(h,4)) for k,val,train,h in sorted(scores,key=lambda a:a[1][j])[:6]])
 best=min(scores,key=lambda a:np.sum(a[1])); print('ADDED',best[0]); base=np.c_[base,basis.pop(best[0])]
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/dag.py
```
- **Container Exit Code**: `1` (`457 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rat2 [0.16290736] 0.00016294835144925308 [[0.541 0.544 0.541 0.544]
 [0.548 0.54  0.541 0.542]
 [0.541 0.541 0.544 0.54 ]
 [0.545 0.535 0.542 0.541]]
rat1 [-0.00074028  0.16298688] 0.00016070878435961585 [[0.541 0.544 0.541 0.544]
 [0.548 0.541 0.541 0.543]
 [0.541 0.541 0.544 0.54 ]
 [0.545 0.537 0.542 0.541]]
cos [1.27387499] 0.0003144599296709318 [[1.066 1.074 1.065 1.077]
 [1.093 1.065 1.066 1.071]
 [1.066 1.068 1.076 1.064]
 [1.082 1.062 1.069 1.066]]
gauss [0.29379754] 0.00024246866077994015 [[0.536 0.54  0.536 0.541]
 [0.547 0.536 0.537 0.539]
 [0.536 0.537 0.541 0.536]
 [0.543 0.533 0.538 0.537]]
tanh [0.72541078] 0.00011095786736738095 [[0.544 0.547 0.544 0.546]
 [0.548 0.544 0.544 0.545]
 [0.544 0.544 0.546 0.544]
 [0.547 0.537 0.545 0.544]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7e2c0a67e442fbba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7e2c0a67 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/dag.py", line 22, in <module>
    nodes[name+k]=fn(x)
                  ^^^^^
  File "/tmp/dag.py", line 15, in <lambda>
    opt=least_squares(err,x0,bounds=(lb,ub)); F=f(u,opt.x); W=np.linalg.lstsq(F,v,rcond=None)[0]; print(nm,opt.x,np.std(opt.fun),np.round(W[5:]/W[1:5],3)); candidates.append((np.std(opt.fun),lambda x,f=f,k=opt.x,W=W:f(x,k)@W))
                                                                                                                                                                                                                        ^^^^^^
  File "/tmp/dag.py", line 12, in f
    def f(x,k): return np.c_[lin(x),fn(x,k)]
                                    ^^^^^^^
  File "/tmp/dag.py", line 15, in <lambda>
    opt=least_squares(err,x0,bounds=(lb,ub)); F=f(u,opt.x); W=np.linalg.lstsq(F,v,rcond=None)[0]; print(nm,opt.x,np.std(opt.fun),np.round(W[5:]/W[1:5],3)); candidates.append((np.std(opt.fun),lambda x,f=f,k=opt.x,W=W:f(x,k)@W))
                                                                                                                                                                                                                        ^^^^^^
TypeError: 'numpy.ndarray' object is not callable
```

### Turn 8 — `execute_in_remote_docker` (`17412 ms` LLM)
- **Rationale**: Repair the composition probe and test the near-constant ratio between the linear and even coefficients as a shared panel constraint.

```bash
sed -i 's/def f(x,k): return/def f(x,k,fn=fn): return/' /tmp/dag.py; OPENBLAS_NUM_THREADS=1 python3 /tmp/dag.py | sed -E 's/array\(//g' ; cat >/tmp/aux.py <<'PY'
import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; v=s['panel_c']['v2'];
for name,f in [('tanh',lambda u,k:u*np.tanh(k*u)),('sqrtr',lambda u,k:u*u/np.sqrt(1+k*u*u)),('sin',lambda u,k:u*np.sin(k*u)),('atan',lambda u,k:u*np.arctan(k*u)),('logcosh',lambda u,k:np.log(np.cosh(k*u))),('soft',lambda u,k:np.sqrt(1+k*u*u)-1),('log',lambda u,k:np.log1p(k*u*u)),('rat',lambda u,k:u*u/(1+k*u*u))]:
 def F(p):return np.c_[np.ones(len(u)),u+p[0]*f(u,p[1])]
 def r(p):
  g=F(p);return (g@np.linalg.lstsq(g,v,rcond=None)[0]-v).ravel()
 o=least_squares(r,[.55,.7],bounds=([.01,.03],[10,4]),gtol=1e-12);print(name,o.x,np.std(o.fun))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/aux.py
```
- **Container Exit Code**: `0` (`772 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0384]), 0.0474), ('ZDCDp', [0.0876, 0.0383, 0.0726]), [0.035 , 0.0191, 0.0303]), 0.0622)]
2 2 [('ZCBDs', [0.0516, 0.0516, 0.0402]), [0.0442, 0.0246, 0.0353]), 0.0457), ('ZB', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473), ('ZC', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473), ('Z', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473), ('ZBCB', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473), ('ZCB', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473)]
ADDED ZB
3 0 [('ZC', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473), ('ZBCB', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473), ('ZBs', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473), ('ZCB', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473), ('Z', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473), ('Zs', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473)]
3 1 [('ZDr', [0.1011, 0.033 , 0.083 ]), [0.0438, 0.0183, 0.0351]), 0.07), ('ZDBp', [0.0752, 0.0339, 0.0693]), [0.0322, 0.016 , 0.0258]), 0.0559), ('ZDBCs', [0.0992, 0.0339, 0.0811]), [0.0331, 0.0188, 0.028 ]), 0.0703), ('Zr', [0.0577, 0.035 , 0.046 ]), [0.0478, 0.0237, 0.0384]), 0.0478), ('Zt', [0.0567, 0.0352, 0.0452]), [0.0479, 0.0237, 0.0384]), 0.0474), ('ZDCDp', [0.0876, 0.0383, 0.0726]), [0.035 , 0.0191, 0.0303]), 0.0622)]
3 2 [('ZCBDs', [0.0516, 0.0516, 0.0402]), [0.0442, 0.0246, 0.0353]), 0.0457), ('ZC', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473), ('ZBCB', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473), ('ZBs', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473), ('Z', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473), ('ZCB', [0.0492, 0.0462, 0.0406]), [0.0469, 0.0257, 0.0372]), 0.0473)]
ADDED ZBCB
tanh [0.75029537 0.72582637] 0.00012814000678836162
sqrtr [0.54866141 0.38916502] 9.021867288871261e-05
sin [0.60026608 0.88808658] 0.0003828161878428184
atan [0.68695913 0.80465314] 0.00011944557221603754
logcosh [0.90968056 1.09971454] 9.378537007995264e-05
soft [1.20631574 0.92055988] 0.0001527018558380486
log [1.50107774 0.36362603] 0.00011226599907075672
rat [0.54155543 0.16310424] 0.00018575687586054894

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7e2c0a67e442fbba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7e2c0a67 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`33946 ms` LLM)
- **Rationale**: Test low-dimensional shared nonlinear terms rather than expanding a deep composition dictionary. Refit time constants within each split and measure the current artifact directly for comparison.

```bash
cat >/tmp/shared.py <<'PY'
import pickle,sys,numpy as np
from itertools import combinations
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import ShuffleSplit
sys.path.insert(0,'/holdout_task'); from batch_solver import calibrate,temporal,solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; b,g=calibrate(s); z=(tr['x']-b)/g; t=s['eval_times']; Y=tr['traj']; y=tr['y']; p=s['panel_c']; u=p['u']; v=p['v2'];
lin=lambda z:np.c_[np.ones(len(z)),z]
cyc=lambda z:np.column_stack([z[:,i]*z[:,j] for i,j in [(0,1),(0,3),(1,2),(2,3)]])
S=lambda x:x+.55*x*x/np.sqrt(1+.39*x*x)
W=np.linalg.lstsq(lin(S(u)),v,rcond=None)[0]; B=np.linalg.lstsq(lin(s['panel_b']['u']),s['panel_b']['v'],rcond=None)[0]; C=np.linalg.lstsq(lin(cyc(u)),p['v1'],rcond=None)[0]; d=lin(S(z))@W; bz=lin(z)@B; cz=lin(cyc(z))@C
P=PolynomialFeatures(2).fit_transform(z); E=np.c_[lin(S(z)),cyc(z)]; rng=np.random.default_rng(128)
for train,test in [(np.arange(52),np.arange(52,72)),*list(ShuffleSplit(2,train_size=52,random_state=11).split(z))]:
 q,a=temporal(t,Y[train]); A=np.linalg.lstsq(q,Y.T,rcond=None)[0].T; T=np.c_[y,A]; st=dict(s,train={k:(v[train] if k!='summary' else v) for k,v in tr.items()},test={'x':tr['x'][test]}); out=solve_batch(st); print('BASE',np.mean(abs(out['y_hat']-y[test])),np.mean(abs(out['state_pred']-Y[test]))); print('rates',-np.log1p(-q[0])/t[0]);
 for nm,F in [('D',lin(S(z))),('D+C',E),('D+C+Z',np.c_[E,z]),('P+D',np.c_[P,S(z)])]:
  fit=F@np.linalg.lstsq(F[train],T[train],rcond=1e-10)[0]; print(nm,np.round(np.mean(abs(fit[test]-T[test]),0),4),np.round(np.std(fit[train]-T[train],0),4));
 if test[0]!=52: continue
 for nm,H in [('linear',z),('B',bz),('C',cz),('D',d),('ZC',np.c_[z,cz]),('ZD',np.c_[z,d])]:
  for typ in ['tanh','rad','sin','prod','rational']:
   F0=E; k=H.shape[1];
   def f(par):
    h=H@par[:k]+par[k];
    if typ=='tanh': w=np.tanh(h)
    if typ=='rad': w=h/np.sqrt(1+h*h)
    if typ=='sin': w=np.sin(h)
    if typ=='prod': w=h*(z@par[k+1:])
    if typ=='rational': w=h/(1+(z@par[k+1:])**2)
    return np.c_[F0,w]
   def calc(par,ret=False):
    F=f(par); W=np.linalg.lstsq(F[train],T[train],rcond=1e-9)[0]; return F@W-T
   best=None
   for i in range(5):
    p0=rng.normal(0,.6,k+1+(4 if typ in ('prod','rational') else 0)); opt=least_squares(lambda v:calc(v)[train].ravel(),p0,bounds=(-5,5),max_nfev=100); e=calc(opt.x); score=np.mean(e[train]**2)
    if best is None or score<best[0]: best=(score,e,opt.x)
   print(nm,typ,np.round(np.mean(abs(best[1][train]),0),4),np.round(np.mean(abs(best[1][test]),0),4),round(np.mean(abs(best[1][test,1:]@q.T)),4),np.round(best[2],3))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/shared.py
```
- **Container Exit Code**: `0` (`5687 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
97 -3.541 -4.681 -0.527]
D prod [0.0637 0.0427 0.0509] [0.1372 0.0801 0.116 ] 0.1311 [ 0.586  0.613  0.493  0.652 -0.122 -0.186  0.754  0.031  0.049]
D rational [0.0531 0.0425 0.0422] [0.0731 0.0539 0.0625] 0.0725 [-0.292  0.305  0.092  0.575  0.003  0.002 -0.002 -0.     0.   ]
ZC tanh [0.0422 0.047  0.0337] [0.048  0.0496 0.0407] 0.0385 [-0.599  0.083 -0.021  0.05  -0.164 -0.007 -0.184 -0.032  1.151]
ZC rad [0.0384 0.0468 0.031 ] [0.0624 0.0368 0.0482] 0.0383 [ 0.042 -0.056  0.807 -0.009 -0.3    0.221  0.09   0.071 -0.759]
ZC sin [0.0445 0.0483 0.0367] [0.0651 0.0442 0.0558] 0.0461 [ 4.000e-03 -0.000e+00  0.000e+00 -0.000e+00  1.000e-03 -0.000e+00
  1.000e-03  0.000e+00  1.576e+00]
ZC prod [0.032  0.0478 0.0252] [0.0344 0.0436 0.0276] 0.0245 [ 0.038 -0.002  0.367  0.437 -0.639  0.598  0.429  0.17   0.758 -0.039
 -0.124  1.718 -0.655]
ZC rational [0.055  0.0482 0.0445] [0.1123 0.0413 0.0972] 0.0897 [ 1.374 -0.016 -0.623  0.24   0.233 -0.335 -0.982  0.471 -0.899 -0.147
  1.223 -1.272 -0.061]
ZD tanh [0.0596 0.0421 0.0471] [0.1183 0.0536 0.1033] 0.1018 [-0.069  1.303  0.059  0.049 -0.126  1.027  0.323  0.197 -0.937]
ZD rad [0.0617 0.0377 0.0515] [0.0807 0.0563 0.0674] 0.0756 [-0.    -0.    -0.     0.     0.045 -0.035  0.023  0.002  0.008]
ZD sin [0.0617 0.0377 0.0514] [0.0807 0.0563 0.0674] 0.0756 [-0.    -0.    -0.     0.     0.052 -0.039  0.026  0.003  0.009]
ZD prod [0.061  0.0422 0.049 ] [0.1048 0.0634 0.0874] 0.0959 [-0.135 -0.043 -1.051  0.328 -0.378 -1.791 -0.885 -2.097 -0.098 -0.564
  1.397 -0.17   0.284]
ZD rational [0.0481 0.0438 0.0383] [0.0739 0.0582 0.0629] 0.0719 [-0.026 -0.048 -0.015  0.009  1.179 -0.717  0.259 -1.905  1.043 -0.1
  0.141  0.051  0.003]
BASE 0.05550735872487923 0.03374817126072908
rates [0.39829162 1.80598084]
D [0.3383 0.0723 0.5496] [0.4565 0.0737 0.5956]
D+C [0.0883 0.0681 0.0749] [0.1102 0.0609 0.0949]
D+C+Z [0.1008 0.0998 0.0871] [0.0968 0.0511 0.0835]
P+D [0.1683 0.0893 0.1436] [0.0787 0.0485 0.0687]
BASE 0.06568415900127604 0.02358021645064725
rates [0.40367027 1.80122461]
D [0.4567 0.0784 0.6491] [0.4146 0.075  0.5673]
D+C [0.1028 0.063  0.0806] [0.108  0.062  0.0942]
D+C+Z [0.1085 0.0831 0.0911] [0.0955 0.053  0.0823]
P+D [0.129  0.0767 0.106 ] [0.087  0.0472 0.0763]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7e2c0a67e442fbba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7e2c0a67 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`26347 ms` LLM)
- **Rationale**: Fit interaction-order weights inside each split and inspect dominant higher-order terms to improve extrapolation beyond the designated holdout.
- **Alternatives Weighed**: `Direct auxiliary-panel composition` (rejected: Tested compositions did not improve on the interpolation baseline.)

```bash
cat >/tmp/ratchet.py <<'PY'
import sys,pickle,numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import ShuffleSplit
sys.path.insert(0,'/holdout_task'); from batch_solver import calibrate,temporal
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; off,g=calibrate(s); z=(tr['x']-off)/g; t=s['eval_times']; y=tr['y']; Y=tr['traj']; z=z/.9; p=PolynomialFeatures(6); F=p.fit_transform(z); deg=p.powers_.sum(1); supp=(p.powers_>0).sum(1); nm=p.get_feature_names_out(); G=[]
for i in range(7):
 for j in range(1 if i else 0,min(i,4)+1):
  v=F[:,(deg==i)&(supp==j)]; G.append(v@v.T)
G=np.array(G); splits=[(np.arange(52),np.arange(52,72)),*ShuffleSplit(8,train_size=52,random_state=11).split(z)]; R=[]
for ii,(a,b) in enumerate(splits):
 q,A=temporal(t,Y[a]); T=np.c_[y,np.linalg.lstsq(q,Y.T,rcond=None)[0].T]; P=[]
 for jj,j in enumerate([0,1,2]):
  Ta=T[a,j]; k=G[:,a[:,None],a]; kb=G[:,b[:,None],a]; k=np.concatenate([k,np.eye(len(a))[None]],0); init=np.r_[np.arange(len(G))*-.5,-11.]; init=np.maximum(init,-11.)
  def fun(log):
   w=np.exp(log); K=np.einsum('i,ijk->jk',w,k); L=cho_factor(K); v=cho_solve(L,Ta); Ki=cho_solve(L,np.eye(len(a))); f=.5*Ta@v+np.log(L[0].diagonal()).sum(); d=.5*np.einsum('ijk,jk->i',k,Ki-np.outer(v,v))*w
   return f,d
  opt=minimize(fun,init,jac=True,method='L-BFGS-B',bounds=[(-24,3)]*len(G)+[(-14,-4)],options=dict(maxiter=160)); w=np.exp(opt.x); pred=np.einsum('i,ijk->jk',w[:-1],kb)@np.linalg.solve(np.einsum('i,ijk->jk',w,k),Ta); P.append(pred)
  if ii==0:
   for d in [3,4]:
    S=F[:,deg<=d]; c=np.linalg.solve(S[a].T@S[a]+.01*np.eye(S.shape[1]),S[a].T@Ta); print('coefs',j,d,[(nm[i],round(c[i],3)) for i in np.argsort(-abs(c))[:16]])
 P=np.array(P).T; ey=np.mean(abs(P[:,0]-y[b])); et=np.mean(abs(P[:,1:]@q.T-Y[b])); R.append((ey,et));print(ii,ey,et)
print('mean/max',np.mean(R,0),np.max(R,0))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/ratchet.py
```
- **Container Exit Code**: `0` (`1349 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, ('x0 x1 x2', -0.221), ('x2', 0.203), ('x2^2 x3', 0.172), ('x0^2', 0.143), ('x1 x2 x3', -0.137), ('x0 x2 x3', 0.128), ('x0 x3^2', 0.099), ('x2^2', 0.096), ('x0 x1^2', 0.093), ('x0^2 x1', 0.084), ('1', -0.074), ('x0 x2^2', -0.059), ('x2 x3^2', -0.059)]
coefs 0 4 [('x2 x3', 0.665), ('x1 x2', -0.342), ('x0', 0.281), ('x2', 0.231), ('x0 x1 x2', -0.169), ('x2^3 x3', 0.157), ('x0^2', 0.156), ('x1 x2^2', 0.115), ('x1 x2 x3', -0.112), ('x2^2 x3', 0.107), ('x2 x3^2', -0.102), ('x0 x2 x3', 0.098), ('x1^2 x2 x3', 0.096), ('x0 x3^2', 0.089), ('1', -0.082), ('x0 x1^2', 0.081)]
coefs 1 3 [('x1', 0.596), ('x3', -0.399), ('x0', 0.392), ('1', 0.313), ('x1^2', 0.245), ('x2', -0.209), ('x3^2', -0.172), ('x0^2', 0.133), ('x1 x2^2', 0.108), ('x0 x1', -0.108), ('x2^2 x3', -0.087), ('x0 x1^2', -0.083), ('x2^2', -0.081), ('x0 x1 x3', 0.071), ('x1^2 x2', 0.07), ('x1 x2', 0.057)]
coefs 1 4 [('x1', 0.559), ('x3', -0.371), ('x0', 0.326), ('1', 0.301), ('x2', -0.178), ('x1^2', 0.157), ('x2^2', -0.154), ('x0^2', 0.138), ('x1 x2^2', 0.115), ('x2^2 x3', -0.111), ('x0 x1', -0.104), ('x1^2 x2', 0.091), ('x0 x3', 0.09), ('x2^4', 0.08), ('x1^3', 0.069), ('x0^2 x2', 0.062)]
coefs 2 3 [('x0 x1', -1.003), ('x0 x3', 0.896), ('x0 x1 x2', -0.181), ('x2^2 x3', 0.153), ('x0 x2 x3', 0.115), ('x1 x2 x3', -0.115), ('1', 0.112), ('x1 x2', 0.107), ('x1', 0.077), ('x0 x3^2', 0.077), ('x0 x2^2', -0.075), ('x0^2 x1', 0.074), ('x0 x1^2', 0.065), ('x2 x3^2', -0.05), ('x0', 0.049), ('x1 x2^2', 0.04)]
coefs 2 4 [('x0 x1', -0.725), ('x0 x3', 0.623), ('x0 x3^3', 0.232), ('x1 x2', 0.22), ('x0 x1 x2', -0.2), ('x0 x2 x3', 0.197), ('x2^2 x3', 0.187), ('x0 x1^3', -0.144), ('x1^2 x2 x3', -0.133), ('x0 x1^2 x3', 0.131), ('x0^2', 0.13), ('x0 x1 x3^2', -0.126), ('1', 0.126), ('x0^2 x1', 0.115), ('x1 x2^2', 0.112), ('x0 x2^2 x3', 0.107)]
0 0.02439472201686594 0.02782805883027824
1 0.04151758035368831 0.025842794675732906
2 0.06636148501473041 0.04315281257465686
3 0.05424749728970139 0.030953191128514496
4 0.05632425064974074 0.03210784719782849
5 0.04991197714789788 0.025342210635194727
6 0.034067142055130305 0.05658117546532573
7 0.03193930138523167 0.04348956622888443
8 0.021386410932130267 0.019844581772513664
mean/max [0.04223893 0.03390469] [0.06636149 0.05658118]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7e2c0a67e442fbba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7e2c0a67 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`36020 ms` LLM)
- **Rationale**: Preserve the observed cycle interactions while allowing smooth higher-order modulation, and use only training-fold residuals to weight the candidate kernels.
- **Alternatives Weighed**: `Promote the degree-adaptive polynomial covariance` (rejected: It worsens the designated 52/20 scores and does not reduce the observed worst scalar error.)

```bash
cat >/tmp/structured_cv.py <<'PY'
import sys,pickle,numpy as np
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
sys.path.insert(0,'/holdout_task'); from batch_solver import calibrate,temporal,solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; o,g=calibrate(s); z=(tr['x']-o)/g; y=tr['y']; Y=tr['traj']; t=s['eval_times']; n=len(z)
cyc=np.array([(0,1),(0,3),(1,2),(2,3)]); C=z[:,cyc[:,0]]*z[:,cyc[:,1]]; P=PolynomialFeatures(2).fit_transform(z); Q=P@P.T; L=np.c_[np.ones(n),z]; S=z+.55*z*z/np.sqrt(1+.39*z*z); A=np.c_[L,C,S]; B=A@A.T
mat=lambda d: (1+d+d*d/3)*np.exp(-d)
D=cdist(z,z); M=mat(D*np.sqrt(5)/4); E=(C@C.T)*mat(D*np.sqrt(5)/2); J=(C@C.T)*(L@L.T)**2
U=sum(mat(abs(z[:,i,None]-z[None,:,i])*np.sqrt(5)/1.2) for i in range(4)); V=sum(mat(cdist(z[:,[i,j]],z[:,[i,j]])*np.sqrt(5)/1.6) for i,j in combinations(range(4),2)); Vc=sum(mat(cdist(z[:,[i,j]],z[:,[i,j]])*np.sqrt(5)/1.6) for i,j in cyc)
R={}; kernels={}
for nm,K in [('iso',M),('poly',Q+.03*(z@z.T)**3+.003*(z@z.T)**4),('cycle',Q+E),('cyclepoly',Q+.15*J),('additive',Q+U+V),('cycleadd',B+U+Vc+E),('blend',Q+U+V+E+2*M),('base',B+10*M),('low',B+E+U),('cyclesharp',Q+5*E+.2*U),('cyclelo',Q+.2*E+.1*U),('anova3',Q+.5*U+.3*V+.3*(z@z.T)*V+.5*E)]:
 for reg in (1e-6,3e-5,3e-4,.002): kernels[nm,reg]=K+reg*np.eye(n)
splits=[(np.arange(52),np.arange(52,72)),*ShuffleSplit(12,train_size=52,random_state=11).split(z)]
for a,b in splits:
 q,_=temporal(t,Y[a]); T=np.c_[y,Y]; preds=[]; lo=[]
 for name,K in kernels.items():
  L=cho_factor(K[np.ix_(a,a)]); w=cho_solve(L,T[a]); test=K[np.ix_(b,a)]@w; loo=w/np.diag(cho_solve(L,np.eye(len(a))))[:,None]; preds.append(test); lo.append(np.r_[np.mean(abs(loo[:,0])),np.mean(abs(loo[:,1:]))]); R.setdefault(name,[]).append(np.r_[np.mean(abs(test[:,0]-y[b])),np.mean(abs(test[:,1:]-Y[b]))]);
 preds=np.array(preds); lo=np.array(lo); out=[]
 for j in range(2):
  wt=np.exp(-6*(lo[:,j]/lo[:,j].min()-1)); wt/=wt.sum(); pp=np.einsum('k,kij->ij',wt,preds); out.append(np.mean(abs((pp[:,0]-y[b]) if j==0 else (pp[:,1:]-Y[b]))));
 R.setdefault(('adaptive',0),[]).append(out)
 st=dict(s,train={k:(v[a] if k!='summary' else v) for k,v in tr.items()},test={'x':tr['x'][b]}); out=solve_batch(st); R.setdefault(('current',0),[]).append([np.mean(abs(out['y_hat']-y[b])),np.mean(abs(out['state_pred']-Y[b]))]);
for j in range(2):
 print('metric',j)
 for k,a in sorted(R.items(),key=lambda it:np.mean(it[1],0)[j])[:13]+[(k,R[k]) for k in [('adaptive',0),('current',0)]]: print(k,'fixed/mean/max',np.round([a[0],np.mean(a,0),np.max(a,0)],5))
f=PolynomialFeatures(4); X=f.fit_transform(z); labs=f.get_feature_names_out(); q,_=temporal(t,Y); T=np.c_[y,np.linalg.lstsq(q,Y.T,rcond=None)[0].T];
for j in range(3):
 w=np.linalg.solve(X.T@X+np.diag(.005*3.**(f.powers_.sum(1)-2.)),X.T@T[:,j]); print('terms',j,[(labs[i],round(w[i],3)) for i in np.argsort(-abs(w))[:28]])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/structured_cv.py
```
- **Container Exit Code**: `0` (`1376 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
645 0.02672]
 [0.05879 0.03816]]
('cycle', 0.002) fixed/mean/max [[0.0211  0.018  ]
 [0.03302 0.02681]
 [0.04853 0.03584]]
('cyclelo', 1e-06) fixed/mean/max [[0.0287  0.0173 ]
 [0.0392  0.02684]
 [0.0646  0.04692]]
('cyclelo', 3e-05) fixed/mean/max [[0.02864 0.01733]
 [0.0391  0.02686]
 [0.06481 0.04712]]
('cyclelo', 0.0003) fixed/mean/max [[0.02815 0.01766]
 [0.03898 0.0271 ]
 [0.06626 0.04862]]
('adaptive', 0) fixed/mean/max [[0.02431 0.01584]
 [0.03365 0.02293]
 [0.09501 0.03631]]
('current', 0) fixed/mean/max [[0.01792 0.02312]
 [0.05277 0.03307]
 [0.08361 0.06228]]
terms 0 [('x2 x3', 0.944), ('x1 x2', -0.492), ('x0', 0.342), ('x0 x1 x2', -0.309), ('x2', 0.23), ('x0^2', 0.178), ('x2^2 x3', 0.174), ('x0 x2 x3', 0.16), ('x1 x2 x3', -0.13), ('x2^2', 0.114), ('x2^3 x3', 0.104), ('x1 x2^2', 0.089), ('x0 x3^2', 0.088), ('x2 x3^2', -0.088), ('x0 x1^2', 0.08), ('x0 x1 x3', 0.07), ('1', -0.066), ('x0^2 x1 x2', -0.064), ('x0^2 x1', 0.055), ('x1', -0.05), ('x0 x1 x3^2', -0.048), ('x3', 0.046), ('x0 x1^2 x3', 0.046), ('x1 x2 x3^2', -0.041), ('x0^3 x3', -0.038), ('x1^2 x2 x3', 0.035), ('x1 x2^3', 0.027), ('x0 x2 x3^2', 0.025)]
terms 1 [('x1', 0.68), ('x3', -0.43), ('x0', 0.4), ('x1^2', 0.331), ('1', 0.298), ('x0^2', 0.257), ('x2', -0.238), ('x2^2', -0.171), ('x3^2', -0.158), ('x1 x2^2', 0.124), ('x0 x1^2', -0.103), ('x0 x1 x2', -0.1), ('x1^2 x2', 0.093), ('x0 x1', -0.071), ('x1 x2 x3', -0.063), ('x0^4', -0.06), ('x0^2 x1^2', -0.059), ('x2^4', 0.055), ('x1 x2^3', 0.054), ('x0^2 x1', -0.049), ('x2^2 x3', -0.049), ('x0^2 x1 x2', 0.041), ('x0 x1^3', -0.04), ('x0^2 x2^2', 0.036), ('x2^3 x3', -0.035), ('x0 x1^2 x2', -0.034), ('x0 x1 x3', 0.034), ('x3^4', -0.032)]
terms 2 [('x0 x1', -1.227), ('x0 x3', 1.08), ('x0 x1 x2', -0.258), ('x2^2 x3', 0.135), ('x1 x2', 0.129), ('1', 0.123), ('x0 x2 x3', 0.115), ('x1 x2 x3', -0.113), ('x1 x2^2', 0.086), ('x1', 0.079), ('x0 x3^2', 0.078), ('x0^2 x1 x2', -0.075), ('x0^2 x1', 0.069), ('x0 x1^2 x3', 0.068), ('x2 x3^2', -0.067), ('x2^3 x3', 0.066), ('x0 x1^2', 0.055), ('x0 x1 x3^2', -0.054), ('x0 x1 x3', 0.053), ('x1^2', 0.049), ('x0 x3^3', 0.048), ('x1 x2 x3^2', -0.045), ('x0 x1 x2^2', -0.043), ('x0^2 x1^2', 0.042), ('x1^2 x2', 0.04), ('x0', 0.038), ('x0^3 x2', 0.033), ('x2', -0.033)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7e2c0a67e442fbba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7e2c0a67 (directory).
```

### Turn 12 — `execute_in_remote_docker` (`32569 ms` LLM)
- **Rationale**: Stress-test the leading cycle kernel against regularization and modest smooth corrections before saving the final artifact. Check endpoint error separately from average trajectory error.
- **Alternatives Weighed**: `Keep the isotropic solver because it has the lowest designated scalar MAE.` (rejected: Its worst resplit scalar error is nearly twice that of the cycle-modulated model, which still passes the designated gate.)

```bash
cat >/tmp/final_probe.py <<'PY'
import sys,pickle,numpy as np
from itertools import combinations
from scipy.spatial.distance import cdist
from sklearn.model_selection import ShuffleSplit
from scipy.linalg import cho_factor,cho_solve
sys.path.insert(0,'/holdout_task'); from batch_solver import calibrate,temporal
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; o,g=calibrate(s); z=(tr['x']-o)/g; y=tr['y']; Y=tr['traj']; t=s['eval_times']; n=len(z)
pairs=np.array(list(combinations(range(4),2))); c=z[:,pairs[:,0]]*z[:,pairs[:,1]]; C=c[:,[0,2,3,5]]; l=z@z.T; L=1+l; Q=L+c@c.T+(z*z)@(z*z).T; ck=C@C.T; d=cdist(z,z); H=np.sqrt(5)*d/2; M=(1+H+H*H/3)*np.exp(-H); I=np.eye(n)
S=np.column_stack([z*z/np.sqrt(1+.39*z*z),np.tanh(z),z/(1+.22*z*z)]); D=S@S.T
p=s['panel_c']; u=p['u']; F=np.c_[np.ones(len(u)),u[:,pairs[[0,2,3,5],0]]*u[:,pairs[[0,2,3,5],1]]]; print('C coefficients',np.linalg.lstsq(F,p['v1'],rcond=None)[0]);
splits=[(np.arange(52),np.arange(52,72)),*ShuffleSplit(24,train_size=52,random_state=11).split(z)]
R={}; predictions={}; specs={}
def add(name,K):
 vals=[]; ps=[]
 for a,b in splits:
  T=np.c_[y,Y]; C0=cho_factor(K[np.ix_(a,a)]); p=K[np.ix_(b,a)]@cho_solve(C0,T[a]); e=p-T[b]; vals.append([np.mean(abs(e[:,0])),np.mean(abs(e[:,1:])),np.mean(abs(e[:,-1]))]); ps.append(p)
 R[name]=np.array(vals); predictions[name]=ps
for alpha in [.04,.15,.45]:
 for ridge in [.002,.006,.018]:
  for mi in [0.,.25,1.]:
   for sep in [0.,.4]:
    name=(alpha,ridge,mi,sep); add(name,Q+alpha*ck*(L**2+mi*4*M)+sep*D+ridge*I)
add('cycle',Q+ck*M+.002*I); add('poly',Q+.03*l**3+.003*l**4+.002*I)
for j in range(3):
 print('metric',j)
 for name,r in sorted(R.items(),key=lambda kv:.65*np.mean(kv[1][:,j])+.35*np.max(kv[1][:,j]))[:8]:print(name,np.round([r[0,j],r[:,j].mean(),r[:,j].max()],5))
for name2 in ['cycle','poly']:
 for wt in [.15,.3,.5]:
  r=[]
  for (a,b),p1,p2 in zip(splits,predictions[.15,.002,0.,0.],predictions[name2]):
   e=(1-wt)*p1+wt*p2-np.c_[y,Y][b];r.append([np.mean(abs(e[:,0])),np.mean(abs(e[:,1:])),np.mean(abs(e[:,-1]))]);
  print('blend',name2,wt,np.round([r[0],np.mean(r,0),np.max(r,0)],5))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/final_probe.py
```
- **Container Exit Code**: `0` (`1207 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
C coefficients [[-0.0265697   0.00973803 -0.09436366  0.10883134]
 [-0.81242598  0.12914738  0.8857803   0.14843204]
 [ 0.04056095 -0.90619026  0.03320897  0.81221433]
 [ 0.81656829 -0.20162399  0.82845928 -0.29968504]
 [-0.39384716 -0.77786191 -0.1064827  -0.8437895 ]]
metric 0
(0.04, 0.002, 0.0, 0.0) [0.02013 0.02611 0.04199]
(0.04, 0.002, 0.25, 0.0) [0.02005 0.0264  0.04169]
(0.15, 0.002, 0.25, 0.0) [0.02368 0.02517 0.04406]
(0.15, 0.002, 0.0, 0.0) [0.02429 0.02494 0.04477]
(0.15, 0.006, 0.25, 0.0) [0.01949 0.02586 0.04379]
(0.04, 0.002, 1.0, 0.0) [0.01985 0.02734 0.04121]
(0.15, 0.006, 0.0, 0.0) [0.01935 0.02558 0.045  ]
(0.15, 0.002, 1.0, 0.0) [0.02313 0.02726 0.04457]
metric 1
(0.04, 0.002, 0.0, 0.0) [0.01175 0.02079 0.03347]
(0.04, 0.002, 0.25, 0.0) [0.01169 0.02076 0.03366]
(0.04, 0.002, 1.0, 0.0) [0.01165 0.02093 0.03428]
(0.04, 0.002, 0.0, 0.4) [0.01191 0.02228 0.03514]
(0.04, 0.002, 0.25, 0.4) [0.01137 0.02231 0.03546]
(0.04, 0.002, 1.0, 0.4) [0.01135 0.02271 0.03611]
(0.04, 0.006, 0.0, 0.0) [0.01191 0.02342 0.03666]
(0.04, 0.006, 0.25, 0.0) [0.01153 0.02331 0.03698]
metric 2
(0.04, 0.002, 0.0, 0.0) [0.01346 0.02335 0.03755]
(0.04, 0.002, 0.25, 0.0) [0.01314 0.02349 0.03811]
(0.04, 0.002, 1.0, 0.0) [0.01215 0.02402 0.03985]
(0.04, 0.002, 0.0, 0.4) [0.01339 0.02512 0.04012]
(0.04, 0.002, 0.25, 0.4) [0.0122  0.02535 0.04043]
(0.15, 0.002, 0.25, 0.0) [0.01679 0.02408 0.0443 ]
(0.15, 0.002, 1.0, 0.0) [0.01362 0.02414 0.04473]
(0.04, 0.002, 1.0, 0.4) [0.01252 0.02628 0.04213]
blend cycle 0.15 [[0.0233  0.01436 0.01601]
 [0.02475 0.02133 0.02363]
 [0.04398 0.0385  0.04148]]
blend cycle 0.3 [[0.0223  0.01403 0.01454]
 [0.02534 0.02067 0.02315]
 [0.04392 0.03645 0.03965]]
blend cycle 0.5 [[0.02097 0.01409 0.01258]
 [0.02673 0.02087 0.02385]
 [0.0446  0.03518 0.04114]]
blend poly 0.15 [[0.02501 0.01436 0.01899]
 [0.02439 0.02164 0.02419]
 [0.04372 0.03831 0.04152]]
blend poly 0.3 [[0.02585 0.01461 0.01984]
 [0.02443 0.02108 0.02388]
 [0.04266 0.0358  0.03864]]
blend poly 0.5 [[0.02745 0.01592 0.02129]
 [0.02532 0.0211  0.02445]
 [0.04193 0.03362 0.03681]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7e2c0a67e442fbba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7e2c0a67 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`34650 ms` LLM)
- **Rationale**: Add the higher-order cycle interactions missing from the quadratic basis, with a modest smooth correction only for the scalar target. Verify the exact saved solver against the public schema, the designated gate, and additional resplits in an isolated Python process.
- **Alternatives Weighed**: `Use the same smooth correction for both outputs` (rejected: It marginally reduced worst scalar error but increased worst trajectory and endpoint errors.)

```bash
set -e
cat >/holdout_task/batch_solver.py <<'PY'
"""Calibrated regression with a cycle-interaction covariance."""
import numpy as np

_PAIRS = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
_CYCLE = [0, 2, 3, 5]


def _calibrate(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    keep = np.ones(len(u), dtype=bool)
    for _ in range(6):
        us, vs = u[keep], v[keep]
        um, vm = us.mean(axis=0), vs.mean(axis=0)
        centered = us - um
        gain = np.sum(centered * (vs - vm), axis=0) / np.sum(centered**2, axis=0)
        offset = vm - um * gain
        error = np.linalg.norm(v - offset - u * gain, axis=1)
        keep = error <= max(.001, 3 * np.median(error))
    return offset, gain


def _products(z):
    return z[:, _PAIRS[:, 0]] * z[:, _PAIRS[:, 1]]


def _kernels(left, right):
    a, b = _products(left), _products(right)
    linear = 1 + left @ right.T
    quadratic = linear + a @ b.T + (left**2) @ (right**2).T
    cycle = a[:, _CYCLE] @ b[:, _CYCLE].T
    state = quadratic + .04 * cycle * linear**2
    d = (np.sqrt(5) / 2) * np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    matern = (1 + d + d**2 / 3) * np.exp(-d)
    return state + .04 * cycle * matern, state


def solve_batch(study: dict) -> dict:
    train = study['train']
    offset, gain = _calibrate(study['panel_a'])
    z = (np.asarray(train['x']) - offset) / gain
    test = (np.asarray(study['test']['x']) - offset) / gain
    ky, ks = _kernels(z, z)
    cy, cs = _kernels(test, z)
    ridge = .002 * np.eye(len(z))
    y = cy @ np.linalg.solve(ky + ridge, np.asarray(train['y']))
    state = cs @ np.linalg.solve(ks + ridge, np.asarray(train['traj']))
    endpoint = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), endpoint.mean(), endpoint.std()])}
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
from sklearn.model_selection import ShuffleSplit
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; Y=tr['traj']
def check(study):
 p=m.solve_batch(study); yp,sp,dp=(p[k] for k in ('y_hat','state_pred','diagnostics_vec')); n=len(study['test']['x']); assert yp.shape==(n,) and sp.shape==(n,len(study['eval_times'])) and dp.shape==(4,); assert all(np.isfinite(v).all() for v in p.values()); assert np.allclose(dp,[yp.mean(),yp.std(),sp[:,-1].mean(),sp[:,-1].std()]); return yp,sp
check(s)
def validate(a,b):
 yp,sp=check(dict(s,train=dict(tr,x=x[a],y=y[a],traj=Y[a]),test={'x':x[b]})); err=abs(sp-Y[b]); return np.r_[np.mean(abs(yp-y[b])),err.mean(),err[:,-1].mean(),err.mean(0).max()]
a=np.arange(52); b=np.arange(52,72); fixed=validate(a,b); assert np.max(fixed)<.038; print('designated y / trajectory / endpoint / worst timestamp MAE:',fixed)
for seed in (11,2027):
 r=np.array([validate(a,b) for a,b in ShuffleSplit(24,train_size=52,random_state=seed).split(x)]); print('resplits',seed,'mean:',r.mean(0),'worst:',r.max(0))
PY
```
- **Container Exit Code**: `0` (`1190 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
designated y / trajectory / endpoint / worst timestamp MAE: [0.0200508  0.01174641 0.01345513 0.01600912]
resplits 11 mean: [0.02666477 0.02116353 0.02376652 0.02739378] worst: [0.04169219 0.03346584 0.03755246 0.04054927]
resplits 2027 mean: [0.02483493 0.01844991 0.01994479 0.02452337] worst: [0.05848462 0.04942466 0.05999318 0.06173865]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=513a2550a2ff90b8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_513a2550 (directory).
```

### Turn 14 — `submit_completed_task` (`33496 ms` LLM)
- **Rationale**: The exact saved artifact passed public schema validation and isolated-process checks of all outputs. It meets the designated holdout threshold, but the additional resplits show that the threshold does not hold uniformly and do not establish the exact physical law.
- **Alternatives Weighed**: `Retain the isotropic baseline` (rejected: On the matched resplits, it had higher worst-case scalar and trajectory errors.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema check passed. On the required 52/20 holdout, scalar MAE was 0.02005 and trajectory MAE was 0.01175, both below 0.038. Across 48 additional resplits, the worst MAEs were 0.05848 and 0.04942, respectively.`

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
