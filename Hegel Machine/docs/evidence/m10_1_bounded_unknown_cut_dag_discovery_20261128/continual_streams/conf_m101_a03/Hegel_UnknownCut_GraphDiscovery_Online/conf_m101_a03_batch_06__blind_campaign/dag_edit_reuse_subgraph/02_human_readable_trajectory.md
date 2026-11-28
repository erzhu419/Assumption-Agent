# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m101_a03_batch_06__blind_campaign__reuse_subgraph`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel_M10_1_UnknownCut_DAG_Discovery` |
| **Domain** | `blind_multi_stage_scientific_computation_graph` |
| **Wave / Wormhole Family** | `m10_1_bounded_unknown_cut_dag_discovery` / `m101_blind_dag` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 12` |
| **ATMS Nogoods Recorded** | `3` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `341,899` (`326,546` / `108,288` / `15,353`) |
| **Total Episode Wall Time (s)** | `501.79s` |

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

### Turn 1 — `execute_in_remote_docker` (`15396 ms` LLM)
- **Rationale**: Inspect the exact node interface, panel dimensions, and scoring splits before choosing a cubic basis; a basis that fits Panel A but fails at panel or range boundaries would be rejected before submission.
- **Alternatives Weighed**: `Modify the operator or graph topology` (rejected: The task freezes both downstream nodes and attributes the failure to calibration.)

```bash
cd /holdout_task && cat _dag_spec.json _frozen_nodes/node_01.py _frozen_nodes/node_02.py test_dag_candidate.py && python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('public_batch.pkl','rb'))
def desc(x,k=''):
 if isinstance(x,dict):
  for a,b in x.items(): desc(b,k+'/'+str(a))
 elif isinstance(x,np.ndarray): print(k,x.shape,x.dtype, x.min(),x.max(),x.reshape(-1)[:8])
 else: print(k,type(x).__name__,str(x)[:180])
desc(p)
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust IRLS Cauchy regression'
```
- **Container Exit Code**: `0` (`486 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
 list [{'study': {'eval_times': array([0.25      , 0.64285714, 1.03571429, 1.42857143, 1.82142857,
       2.21428571, 2.60714286, 3.        ]), 'panel_a': {'u': array([[ 0.55011363,  0.7
autogenous_robust_network_assimilation@2262dea7 | assimilate_diffusive_network(laplacian: np.ndarray, damping: np.ndarray, inlet: np.ndarray, initial_state: np.ndarray, sensor_nodes: np.ndarray, observations: np.ndarray, horizon: int, dt: float, *, harmonics: int=5, valve_levels: tuple[float, float]=(0.35, 1.0), gain_floor: float=0.2) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Infer constant transmission and forecast a forward-Euler network | REQ: {'field': 'dt', 'op': '>', 'value': 0}; {'field': 'observation_sample_count', 'op': '>=' | --describe autogenous_robust_network_assimilation.assimilate_diffusive_network
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`29620 ms` LLM)
- **Rationale**: Compare nested forward models with robust row weights and leave-one-out fits. Retain extra terms only if they explain repeatable calibration error, then invert the selected model to supply the frozen node’s two required arrays.
- **Alternatives Weighed**: `Use a full cubic basis without testing simpler bases` (rejected: Leave-one-out error can distinguish necessary interactions from coefficients that only fit panel noise.)

```bash
cd /holdout_task && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_robust_network_assimilation.assimilate_diffusive_network | head -35; cat > /tmp/cal_basis.py <<'PY'
import itertools as it
import pickle
import numpy as np
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
for k,x in st.items():
    print(k, {a: np.shape(b) for a,b in x.items()} if isinstance(x,dict) else np.shape(x))
u,v=(st['panel_a'][k] for k in ('u','v'))
n,d=u.shape
print('u ranges',u.min(0),u.max(0),'v ranges',v.min(0),v.max(0),'test ranges',st['test']['x'].min(0),st['test']['x'].max(0))
terms=[()] + [(j,) for j in range(d)]
terms2=list(it.combinations_with_replacement(range(d),2)); terms3=list(it.combinations_with_replacement(range(d),3)); cubes=[(j,)*3 for j in range(d)]
def design(x,ts): return np.column_stack([np.prod(x[:,t],axis=1) for t in ts])
def fit(A,Y):
    w=np.ones(len(A))
    for scale in (.15,.08,.04,.02,.01,.005,.003,.003,.003,.003):
        b=np.linalg.lstsq(A*w[:,None],Y*w[:,None],rcond=None)[0]
        e=np.linalg.norm(Y-A@b,axis=1)
        w=(1+(e/scale)**2)**-.5
    keep=np.argsort(e)[:-3]
    return np.linalg.lstsq(A[keep],Y[keep],rcond=None)[0],keep
if __name__=='__main__':
 for name, ts in [('affine',terms),('pure_cubes',terms+cubes),('quadratic',terms+terms2),('quad+cubes',terms+terms2+cubes),('homogeneous_cubics',terms+terms3),('full',terms+terms2+terms3)]:
    A=design(u,ts); b,keep=fit(A,v)
    err=np.linalg.norm(A@b-v,axis=1); order=np.argsort(err)
    ee=[]
    if len(ts)<n-5:
        for j in range(n):
            ids=np.arange(n)!=j
            bj,_=fit(A[ids],v[ids]);ee.append(np.linalg.norm(A[j]@bj-v[j]))
    print(name,'p',len(ts),'median/worst retained',np.median(err),err[order[-4]],'outliers',order[-3:],err[order[-3:]],'leave-one-out quantiles',np.quantile(ee,[.5,.85,.9,.95]) if ee else None)
    if name in ('pure_cubes','quad+cubes','homogeneous_cubics','full'):
        for t,c in zip(ts,b):
            if max(abs(c))>.009: print(t,np.round(c,4))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/cal_basis.py
```
- **Container Exit Code**: `0` (`679 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
679 -0.3547  0.0771  0.0319]
(0, 0, 2) [ 0.0343  0.3954  0.0303 -0.0189]
(0, 0, 3) [ 0.0827  0.1533 -0.059  -0.1773]
(0, 1, 1) [-0.139  -0.1462  0.0681 -0.2048]
(0, 1, 2) [-0.0096  0.4373  0.1379 -0.0709]
(0, 1, 3) [-0.3003  0.1678 -0.0848 -0.1475]
(0, 2, 2) [ 0.0548  0.0501  0.0223 -0.0705]
(0, 2, 3) [ 0.03   -0.2583 -0.137   0.0936]
(0, 3, 3) [0.2247 0.1586 0.1187 0.0887]
(1, 1, 1) [-0.0559  0.2486  0.0157 -0.0849]
(1, 1, 2) [ 0.0452 -0.0839  0.0672 -0.0366]
(1, 1, 3) [-0.0072 -0.0032 -0.0257  0.0582]
(1, 2, 2) [ 0.0142  0.0107 -0.0272  0.0226]
(1, 2, 3) [ 0.1019 -0.2407 -0.1689  0.0105]
(1, 3, 3) [-0.0955 -0.0731  0.0895  0.1228]
(2, 2, 2) [ 0.0058 -0.0527  0.0926 -0.0311]
(2, 2, 3) [0.057  0.1919 0.0561 0.0116]
(2, 3, 3) [-0.0495 -0.0082 -0.2368 -0.105 ]
(3, 3, 3) [ 0.071   0.0291 -0.1091  0.0974]
full p 35 median/worst retained 0.10869664485632283 0.45274472346725064 outliers [ 7 17  5] [ 5.30205633 14.03563012 15.86739295] leave-one-out quantiles [ 1.01053621  3.21889451  5.04847574 12.89317264]
() [-0.1478 -0.0199  0.0177  0.102 ]
(0,) [ 1.354   0.2368 -0.2488  0.0949]
(1,) [ 0.6841  1.2339 -0.4091 -0.3706]
(2,) [ 0.5762  0.9367  0.5374 -0.5863]
(3,) [-1.0668 -1.1413  1.5175  2.1854]
(0, 0) [ 1.212   1.2969 -1.361  -1.2332]
(0, 1) [ 1.3849  1.1395 -1.1956 -1.0835]
(0, 2) [-0.6171 -0.6607  0.6931  0.6278]
(0, 3) [-1.8198 -1.9474  2.0434  2.1716]
(1, 1) [ 0.299   0.3199 -0.3355 -0.3043]
(1, 2) [ 0.5645  0.9243 -0.6339 -0.5745]
(1, 3) [-1.49   -1.5944  1.673   1.5161]
(2, 2) [-0.8858 -0.9476  0.9944  0.9013]
(2, 3) [ 1.348   1.4427 -1.1934 -1.3715]
(3, 3) [ 0.7906  0.846  -0.8878 -0.8045]
(0, 0, 0) [-0.0344 -0.2081  0.2187  0.198 ]
(0, 0, 1) [ 1.4974  1.6022 -1.6814 -1.5237]
(0, 0, 2) [-0.3845 -0.4117  0.4317  0.3914]
(0, 0, 3) [-0.8525 -0.9124  0.9575  0.8678]
(0, 1, 1) [ 1.4732  1.5765 -1.6543 -1.4994]
(0, 1, 2) [-1.1902 -1.2742  1.3366  1.2112]
(0, 1, 3) [-3.1101 -3.3283  3.4924  3.1651]
(0, 2, 2) [-0.34   -0.364   0.3821  0.346 ]
(0, 2, 3) [ 0.7075  0.7572 -0.7948 -0.7206]
(0, 3, 3) [ 1.5263  1.633  -1.7137 -1.553 ]
(1, 1, 1) [ 0.1949  0.3684 -0.2187 -0.1982]
(1, 1, 2) [-0.2509 -0.2686  0.282   0.2554]
(1, 1, 3) [-0.7432 -0.7954  0.8347  0.7562]
(1, 2, 2) [-1.2084 -1.293   1.357   1.2296]
(1, 2, 3) [ 2.2717  2.4313 -2.5507 -2.3116]
(1, 3, 3) [ 0.7791  0.8341 -0.8751 -0.793 ]
(2, 2, 2) [ 0.2404  0.2575 -0.1103 -0.2448]
(2, 2, 3) [ 0.4844  0.5182 -0.5437 -0.4927]
(2, 3, 3) [-1.1083 -1.1859  1.2446  1.1277]
(3, 3, 3) [ 0.123   0.1316 -0.1379  0.0348]
```

### Turn 3 — `execute_in_remote_docker` (`35751 ms` LLM)
- **Rationale**: The repeated coefficients support shared nonlinear terms across channels. The isolated official test will detect a bad handoff, incorrect inverse, or either downstream error exceeding tolerance.
- **Alternatives Weighed**: `Unrestricted cubic regression` (rejected: Its leave-one-out median residual was 1.011, while the recovered coefficient pattern supports a much smaller basis.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine forward calibration suffices after rejecting the three Panel A outliers.` | Witness: `Even unrestricted affine fitting had median row residual 0.226 and worst retained residual 0.711. Quadratic-plus-cube fitting recovered four cyclic adjacent products near 0.32 and four own-channel cubes near 0.16, with nonparticipating terms near zero.` | Blanket Diagnosis: `The calibration node omitted cyclic cross-channel dependence and own-channel curvature; the frozen operator does not need to change.` | Sublated Pivot: `Fit the sparse cyclic forward model with robust row weights and invert it for both downstream arrays.`

```bash
cd /holdout_task && cat > node_00.py <<'PY'
import numpy as np


def _features(u):
    n, d = u.shape
    channel = np.broadcast_to(np.eye(d), (n, d, d))
    neighbor = np.roll(u, -1, axis=1)
    return np.concatenate((channel, channel * u[..., None],
                           neighbor[..., None], (u * neighbor)[..., None],
                           (u ** 3)[..., None]), axis=2).reshape(n * d, 2 * d + 3)


def _fit(u, v):
    design = _features(u)
    y = v.ravel()
    d = u.shape[1]
    weights = np.ones(len(u))
    for scale in (0.15, 0.05, 0.015, 0.004, 0.0015, 0.0015):
        w = np.repeat(weights, d)
        p = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)[0]
        residual = np.linalg.norm((design @ p).reshape(v.shape) - v, axis=1)
        weights = 1 / np.hypot(1, residual / scale)
    keep = np.repeat(np.argsort(np.argsort(residual)) < len(u) - 3, d)
    return np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]


def _invert(v, p):
    d = v.shape[1]
    offset, gain = p[:d], p[d:2*d]
    coupling, interaction, cubic = p[-3:]
    j = np.arange(d)
    nxt = (j + 1) % d
    linear = np.diag(gain)
    linear[j, nxt] = coupling
    x = np.linalg.solve(linear, (v - offset).T).T

    def error(x):
        neighbor = x[:, nxt]
        return offset + gain * x + (coupling + interaction * x) * neighbor + cubic * x ** 3 - v

    for _ in range(25):
        r = error(x)
        if np.max(np.abs(r), initial=0.0) < 1e-11:
            break
        jac = np.zeros((len(x), d, d))
        jac[:, j, j] = gain + interaction * x[:, nxt] + 3 * cubic * x ** 2
        jac[:, j, nxt] = coupling + interaction * x
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        trial = x - step
        baseline = np.linalg.norm(r, axis=1)
        for _ in range(12):
            worse = np.linalg.norm(error(trial), axis=1) > baseline
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = x[worse] - step[worse]
        x = trial
    return x


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    panel = study['panel_a']
    p = _fit(panel['u'], panel['v'])
    return {'x_cal_tr': _invert(study['train']['x'], p),
            'x_cal_te': _invert(study['test']['x'], p)}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `1` (`719 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
{"accepted": false, "errors": ["public_cv_y_mae_exceeded:0.367832", "public_cv_traj_mae_exceeded:0.239609"], "cv_metrics": {"public_cv_y_mae": 0.367832, "public_cv_traj_mae": 0.239609}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f70158ce7d3156ba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f70158ce (directory).
```

### Turn 4 — `execute_in_remote_docker` (`28406 ms` LLM)
- **Rationale**: Panel residuals and inverse round-trip error can isolate a fit or solver problem. Comparing forward, inverse, and raw coordinates through the same frozen graph tests the handoff convention without editing downstream nodes.
- **Alternatives Weighed**: `Add polynomial terms immediately` (rejected: First distinguish a poor panel fit from an accurate map applied in the wrong direction; extra terms would not fix a coordinate mismatch.)

```bash
cd /holdout_task && cat public_asset_index.json && cat _frozen_nodes/node_02.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import copy, importlib.util, pickle
import numpy as np
import node_00 as n
st=pickle.load(open('public_batch.pkl','rb'))[0]['study']
def load(name):
 s=importlib.util.spec_from_file_location(name,'_frozen_nodes/'+name+'.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.run_node
op, read=map(load,['node_01','node_02'])
a=st['panel_a']; u,v=a['u'],a['v']; p=n._fit(u,v)
f=lambda x:(n._features(x)@p).reshape(x.shape)
i=lambda x:n._invert(x,p)
print('fit parameters',p,'telemetry',st['telemetry_vec']); e=np.linalg.norm(f(u)-v,axis=1); print('forward row residual median and last four',np.median(e),np.sort(e)[-4:]); print('inverse round-trip on clean panel',np.max(np.abs(i(f(u))-u)))
for k,panel in ((k,st[k]) for k in ('panel_a','panel_b','panel_c')):
 print(k,[(s,np.round(x.mean(0),3),np.round(x.std(0),3)) for s,x in panel.items()]); print('sample', {s:np.round(x[:2],3) for s,x in panel.items()})
print('train x mean/std',st['train']['x'].mean(0),st['train']['x'].std(0))
s=copy.deepcopy(st); t=s['train']; x=t['x']; y=t['y']; tr=t['traj']; s['train']={k:w[:-20] for k,w in [('x',x),('y',y),('traj',tr)]}; s['test']={'x':x[-20:]}
aff=np.column_stack((np.ones(len(u)),u)); keep=np.argsort(e)[:-3]; b=np.linalg.lstsq(aff[keep],v[keep],rcond=None)[0]
for name,fn in [('raw coordinate diagnostic',lambda x:x),('inverse',i),('forward',f),('affine inverse',lambda x:np.linalg.solve(b[1:].T,(x-b[0]).T).T),('affine forward',lambda x:b[0]+x@b[1:])]:
 z=fn(x); out=read({'node_01':op({'__input__':s,'node_00':{'x_cal_tr':z[:-20],'x_cal_te':z[-20:]}})}); print(name,'MAEs',np.mean(abs(out['y_hat']-y[-20:])),np.mean(abs(out['state_pred']-tr[-20:])),'y range',y.min(),y.max(),'traj range',tr.min(),tr.max())
PY
```
- **Container Exit Code**: `0` (`804 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
4872e+01 1.22609625e+01 1.23560361e+01]
inverse round-trip on clean panel 2.220446049250313e-16
panel_a [('u', array([ 0.108, -0.246,  0.025, -0.167]), array([0.688, 0.783, 0.783, 0.649])), ('v', array([ 0.024, -0.098, -0.118, -0.15 ]), array([1.836, 1.861, 1.911, 1.736]))]
sample {'u': array([[ 0.55 ,  0.794,  1.2  , -0.33 ],
       [-0.45 , -1.283,  1.009, -0.597]]), 'v': array([[ 0.889,  1.412,  1.49 , -0.142],
       [-0.905, -1.54 ,  1.001, -0.64 ]])}
panel_b [('u', array([-0.072,  0.099, -0.049, -0.003]), array([0.782, 0.659, 0.681, 0.81 ])), ('v', array([-0.071, -0.036,  0.092, -0.024]), array([0.73 , 0.683, 0.478, 0.67 ]))]
sample {'u': array([[ 0.81 ,  0.395, -0.605,  0.118],
       [ 0.614,  0.135,  0.739,  0.938]]), 'v': array([[-0.694,  0.354,  0.672,  0.184],
       [-0.937, -0.056, -0.129, -0.88 ]])}
panel_c [('u', array([-0.326, -0.154, -0.168,  0.085]), array([0.747, 0.658, 0.729, 0.632])), ('v1', array([0.327, 0.234, 0.144, 0.086]), array([0.595, 0.685, 0.606, 0.682])), ('v2', array([ 0.016,  0.085, -0.169,  0.124]), array([0.355, 0.303, 0.341, 0.313]))]
sample {'u': array([[-0.836, -0.307,  0.849, -0.202],
       [-0.997, -0.926, -0.325,  0.931]]), 'v1': array([[ 0.745,  0.771, -0.679,  0.731],
       [ 0.808,  1.039,  0.942,  0.026]]), 'v2': array([[-0.41 ,  0.148, -0.491, -0.224],
       [ 0.071,  0.69 ,  0.004,  0.314]])}
train x mean/std [-0.31905044 -0.15758755 -0.01342406 -0.04933552] [0.85167332 0.63063063 0.8176068  0.85686291]
raw coordinate diagnostic MAEs 0.4558585789443933 0.379125208610499 y range -2.4742914372823965 3.1246758522344127 traj range -2.002456403179577 3.3705727518535755
inverse MAEs 0.3678320056754957 0.23960911317134545 y range -2.4742914372823965 3.1246758522344127 traj range -2.002456403179577 3.3705727518535755
forward MAEs 0.5028139017190745 0.548111710904395 y range -2.4742914372823965 3.1246758522344127 traj range -2.002456403179577 3.3705727518535755
affine inverse MAEs 0.4712925703998426 0.3791252086104991 y range -2.4742914372823965 3.1246758522344127 traj range -2.002456403179577 3.3705727518535755
affine forward MAEs 0.4261100538545315 0.3791252086104989 y range -2.4742914372823965 3.1246758522344127 traj range -2.002456403179577 3.3705727518535755

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f70158ce7d3156ba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f70158ce (directory).
```

### Turn 5 — `execute_in_remote_docker` (`33733 ms` LLM)
- **Rationale**: Measure how much of the trajectory error is unavoidable under the frozen rank-two output, and test whether nonlinear coordinate terms explain the remaining error on held-out rows. Characterizing the reference panels separately can distinguish omitted structure from a bad calibration inversé.
- **Alternatives Weighed**: `Refine the cyclic calibration coefficients` (rejected: The worst retained panel residual is only 0.000353 and the clean-panel round trip is exact to floating-point precision; coefficient refinement is unlikely to explain the downstream error.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -B - <<'PY'
import pickle,itertools as it
import numpy as np
import node_00 as n
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; p=n._fit(a['u'],a['v']); x=s['train']['x']; t=s['train']['traj']; y=s['train']['y']; d=x.shape[1]
aff=lambda q:np.column_stack((np.ones(len(q)),q))
f=lambda q:(n._features(q)@p).reshape(q.shape)
inv=lambda q:n._invert(q,p)
terms2=list(it.combinations_with_replacement(range(d),2)); terms3=list(it.combinations_with_replacement(range(d),3))
def basis(q,terms): return np.column_stack([aff(q)]+[np.prod(q[:,term],axis=1) for term in terms])
def loo(A,b):
 c=np.linalg.lstsq(A,b,rcond=None)[0]; h=np.sum(np.linalg.qr(A,mode='reduced')[0]**2,axis=1); e=A@c-b
 return np.mean(abs(e)),np.mean(abs(e/np.maximum(1-h,1e-9)[:,None])),c
for name in ('panel_b','panel_c'):
 panel=s[name]; u=panel['u']
 for key,v in panel.items():
  if key=='u': continue
  for desc,terms in [('affine',[]),('quadratic',terms2),('cyclic+cubes',[(i,(i+1)%d) for i in range(d)]+[(i,)*3 for i in range(d)])]:
   ee,cv,c=loo(basis(u,terms),v); print(name,key,desc,ee,cv); 
   if desc=='affine': print(np.round(c,4))
B=np.linalg.lstsq(aff(s['panel_b']['u']),s['panel_b']['v'],rcond=None)[0]
print('panel B singular vals',np.linalg.svd(B[1:],compute_uv=False)); U,sv,V=np.linalg.svd(t,full_matrices=False); print('trajectory singular values',sv,'rank2 MAE',np.mean(abs(t-(U[:,:2]*sv[:2])@V[:2]))); print('centered trajectory singular values',np.linalg.svd(t-t.mean(0),compute_uv=False))
for name,q in [('raw',x),('inverse',inv(x)),('forward',f(x))]:
 z=aff(q)@B
 for desc,terms in [('affine',[]),('quadratic',terms2),('cubic',terms2+terms3)]:
  A=basis(q,terms); c=np.linalg.lstsq(A[:52],t[:52],rcond=None)[0]; ee,cv,_=loo(A,t); print(name,desc,'traj full/train52 tail/loo',ee,np.mean(abs(A[52:]@c-t[52:])),cv)
 print(name,'inverse residual',abs(f(inv(x))-x).max(),'calibrated extrema',q.min(),q.max(),'z extrema',z.min(),z.max())
PY
```
- **Container Exit Code**: `0` (`429 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.4517  0.4398  0.7361  0.0807]]
panel_c v1 quadratic 0.060389201040384234 0.21812081084477972
panel_c v1 cyclic+cubes 0.022016636856542638 0.054998764376457565
panel_c v2 affine 0.07196115290576159 0.09264040904719306
[[-0.0942 -0.0763 -0.1702  0.1582]
 [-0.166  -0.2504  0.2681  0.1203]
 [ 0.1104 -0.2478 -0.2694  0.2355]
 [-0.4326 -0.1064 -0.2394 -0.0422]
 [-0.0014  0.2799  0.0811  0.4055]]
panel_c v2 quadratic 0.0026996691461959933 0.008811358778842123
panel_c v2 cyclic+cubes 0.05079074339585141 0.12390969883139223
panel B singular vals [0.88006833 0.88001386 0.880002   0.87997192]
trajectory singular values [24.00996234  0.40724458  0.26445012  0.0492016   0.04639911  0.04381726
  0.03875468  0.03299419] rank2 MAE 0.008521645849473887
centered trajectory singular values [23.80850902  0.40429197  0.2644501   0.0492003   0.04635302  0.04380297
  0.0386827   0.03107921]
raw affine traj full/train52 tail/loo 0.3193603359665133 0.3790793490132017 0.3476351544206992
raw quadratic traj full/train52 tail/loo 0.059563176145442216 0.08194754691358125 0.07935802583702428
raw cubic traj full/train52 tail/loo 0.018657325206296076 0.07253702535022433 0.04821911897795075
raw inverse residual 7.962741577216548e-12 calibrated extrema -1.6415432467142934 1.9861794677082705 z extrema -2.024560131054075 1.7362132684030358
inverse affine traj full/train52 tail/loo 0.21720296341756354 0.2395659794087388 0.23653622719236436
inverse quadratic traj full/train52 tail/loo 0.0034010058026483683 0.004177542423306424 0.004315469017833788
inverse cubic traj full/train52 tail/loo 0.0027545466510362376 0.006315582100993161 0.005758908354463656
inverse inverse residual 7.962741577216548e-12 calibrated extrema -1.129086861864547 1.134119297026454 z extrema -1.4719177944307205 1.352339451789693
forward affine traj full/train52 tail/loo 0.48575618222233097 0.5487068672294234 0.5278863864641717
forward quadratic traj full/train52 tail/loo 0.19784528106061894 0.34918565016210146 0.2924765518848053
forward cubic traj full/train52 tail/loo 0.0650412112527495 0.3140109119671887 0.2604324528268424
forward inverse residual 7.962741577216548e-12 calibrated extrema -2.694525540600197 4.441890226010006 z extrema -4.644397493285574 2.414742598675138

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f70158ce7d3156ba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f70158ce (directory).
```

### Turn 6 — `execute_in_remote_docker` (`27458 ms` LLM)
- **Rationale**: Read the exact frozen feature equations and inspect coefficient support in both coordinate frames. If a sparse interaction pattern explains the held-out trajectory improvement, use it to test a calibration correction without changing the topology.
- **Alternatives Weighed**: `Add more trajectory modes` (rejected: The frozen rank-two representation is already well within tolerance; adding modes would not fix the missing input dependence.)
- **ATMS NOGOOD DECLARED**: Refuted `Accurately inverting Panel A produces coordinates in which the trajectory is affine after the frozen Panel B map.` | Witness: `After Panel A inversion, an affine trajectory fit gives 0.239566 tail MAE while a quadratic fit gives 0.004178. The frozen rank-two trajectory approximation has only 0.008522 reconstruction MAE.` | Blanket Diagnosis: `The gap is in the input representation of the frozen readout, not Panel A fit precision or the trajectory rank constraint.` | Sublated Pivot: `Retain the robust Panel A inverse and identify the quadratic terms and coordinate convention shared by the trajectory and nonlinear readout.`

```bash
cd /holdout_task && cat _frozen_nodes/node_01.py _frozen_nodes/node_02.py test_dag_candidate.py && OPENBLAS_NUM_THREADS=1 python3 -B - <<'PY'
import pickle, numpy as np
import node_00 as n
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; p=n._fit(a['u'],a['v']); u=n._invert(s['train']['x'],p); b=s['panel_b']; c=s['panel_c']; t=s['train']['traj']; y=s['train']['y']; d=u.shape[1]
aff=lambda x:np.column_stack((np.ones(len(x)),x))
B=np.linalg.lstsq(aff(b['u']),b['v'],rcond=None)[0]
terms=[(i,j) for i in range(d) for j in range(i,d)]
quad=lambda x:np.column_stack((aff(x),*[x[:,i]*x[:,j] for i,j in terms]))
pc=np.linalg.svd(t,full_matrices=False)[2][0]
for name,x in [('u',u),('z',aff(u)@B)]:
 print(name,'quadratic coefficients of trajectory mode and y, rows 1,x_i,x_ix_j:',terms,'\n',np.round(np.linalg.lstsq(quad(x)[:52],np.c_[t@pc,y][:52],rcond=None)[0],4))
for k in ('v1','v2'):
 print('Panel C',k,'quadratic coefficients\n',np.round(np.linalg.lstsq(quad(c['u']),c[k],rcond=None)[0],4))
print('Panel C affine gains/singular values',*[np.linalg.svd(np.linalg.lstsq(aff(c['u']),c[k],rcond=None)[0][1:],compute_uv=False) for k in ('v1','v2')]); print('times',s['eval_times'])
PY
```
- **Container Exit Code**: `0` (`496 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
9 ]
 [-0.1741 -0.1006]
 [-0.0523 -0.0347]
 [ 0.2184  0.139 ]
 [ 0.0686  0.0039]
 [ 0.1974  0.0609]
 [ 0.1776  0.0569]
 [-1.3651 -0.597 ]
 [-0.0638 -0.0176]]
z quadratic coefficients of trajectory mode and y, rows 1,x_i,x_ix_j: [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)] 
 [[-2.3000e-03 -2.4000e-02]
 [ 3.0441e+00  1.1527e+00]
 [ 2.8942e+00  9.7770e-01]
 [-1.1552e+00  1.2108e+00]
 [-3.8060e-01 -8.3970e-01]
 [ 2.8360e-01  1.7070e-01]
 [ 8.2200e-02 -2.3100e-02]
 [ 2.6010e-01  3.0540e-01]
 [-1.7384e+00 -8.5930e-01]
 [ 2.2720e-01  1.3650e-01]
 [-1.7635e+00 -9.3140e-01]
 [-2.2450e-01  1.7800e-02]
 [-8.0400e-02 -2.6000e-02]
 [-6.8800e-02 -1.7300e-02]
 [-1.5450e-01 -1.0790e-01]]
Panel C v1 quadratic coefficients
 [[ 0.1758  0.2704 -0.0795  0.1406]
 [-0.6358 -0.1495 -0.327  -0.59  ]
 [-0.3913 -0.6882  0.0574  0.5441]
 [-0.2163  0.493  -0.6127  0.4617]
 [-0.4486  0.6999  0.8756  0.1098]
 [-0.126  -0.1259 -0.1007 -0.0924]
 [ 0.0718 -0.0689 -0.0092  0.0398]
 [ 0.0023  0.0142  0.0867  0.0136]
 [ 0.0514  0.0138 -0.0141 -0.0328]
 [-0.0386 -0.2033 -0.0434 -0.0039]
 [ 0.0965 -0.002   0.0086 -0.1089]
 [-0.0472 -0.001   0.0582 -0.0478]
 [ 0.0239 -0.0314  0.0893 -0.0588]
 [-0.0024  0.3879  0.1537  0.1312]
 [-0.0536 -0.0383 -0.0786 -0.0462]]
Panel C v2 quadratic coefficients
 [[ 5.950e-02  1.480e-02 -1.414e-01  1.170e-02]
 [-1.384e-01 -3.228e-01  2.955e-01  1.610e-01]
 [ 1.708e-01 -2.324e-01 -3.042e-01  2.465e-01]
 [-4.230e-01 -2.890e-02 -2.228e-01 -1.940e-02]
 [-7.670e-02  2.851e-01  6.450e-02  3.824e-01]
 [-6.400e-02 -1.437e-01  1.300e-01  7.500e-02]
 [-3.000e-03 -1.260e-02  3.000e-04 -3.000e-04]
 [ 1.000e-03 -2.100e-03 -3.600e-03 -1.700e-03]
 [-6.600e-03  1.000e-03 -4.500e-03 -2.100e-03]
 [ 7.770e-02 -1.072e-01 -1.409e-01  1.117e-01]
 [-9.000e-04  3.000e-04 -9.000e-04  5.000e-04]
 [-2.300e-03 -8.800e-03 -7.400e-03 -6.600e-03]
 [-1.962e-01 -1.460e-02 -9.630e-02 -6.700e-03]
 [-1.000e-04  5.700e-03  9.000e-04 -4.300e-03]
 [-2.700e-02  1.288e-01  2.540e-02  1.729e-01]]
Panel C affine gains/singular values [1.01131655 0.9303402  0.88081373 0.87166842] [0.57119523 0.46969009 0.43192445 0.3894763 ]
times [0.25       0.64285714 1.03571429 1.42857143 1.82142857 2.21428571
 2.60714286 3.        ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f70158ce7d3156ba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f70158ce (directory).
```

### Turn 7 — `execute_in_remote_docker` (`34489 ms` LLM)
- **Rationale**: Test whether the Panel C map makes the held-out trajectory affine, and whether a quadratic index retains the frozen tanh readout law. These checks distinguish a panel-derived correction from a missing feature basis and test both error bottlenecks independently.
- **Alternatives Weighed**: `Use Panel C as the missing correction without testing it` (rejected: Panel C shows nonlinear structure, but its relationship to the coordinates consumed by the frozen nodes is not established.)

```bash
cd /holdout_task && sed -n '25,135p' test_dag_candidate.py && OPENBLAS_NUM_THREADS=1 python3 -B - <<'PY'
import pickle,numpy as np
from scipy.optimize import least_squares
import node_00 as n
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; u=n._invert(s['train']['x'],n._fit(a['u'],a['v'])); t=s['train']['traj']; y=s['train']['y']; c=s['panel_c']; d=u.shape[1]
def aff(x): return np.c_[np.ones(len(x)),x]
pairs=[(i,j) for i in range(d) for j in range(i,d)]
def basis(x,ps): return np.c_[aff(x),*[x[:,i]*x[:,j] for i,j in ps]]
def cv(A,T):
 p=np.linalg.lstsq(A[:52],T[:52],rcond=None)[0]; return np.mean(abs(A[52:]@p-T[52:]))
C=np.linalg.lstsq(basis(c['u'],[(i,i) for i in range(d)]),c['v2'],rcond=None)[0]; print('Panel C v2 diagonal model MAE',np.mean(abs(basis(c['u'],[(i,i) for i in range(d)])@C-c['v2']))); print('Panel C v2 diagonal model',np.round(C,4)); print('trajectory CV from Panel C v2',cv(aff(basis(u,[(i,i) for i in range(d)])@C),t)); print('trajectory CV using diagonal quadratic',cv(basis(u,[(i,i) for i in range(d)]),t))
A=basis(u,pairs); modes=np.linalg.svd(t[:52],full_matrices=False)[2]; pt=np.linalg.lstsq(A[:52],t[:52]@modes[:3].T,rcond=None)[0]; print('trajectory coefficients in first three modes',np.round(pt,4)); print('interaction singular values',np.linalg.svd(pt[5:],compute_uv=False))
for name,k in [('affine',[]),('squares',[(i,i) for i in range(d)]),('all quadratic',pairs)]:
 D=basis(u,k); g=np.linalg.lstsq(D[:52],y[:52],rcond=None)[0]; fits=[]
 def pred(theta,D):
  q=D@theta[:-3]; o,amp,f=theta[-3:]; return q+amp*np.tanh(f*q+o)
 for amp in (.15,.35,.65):
  for freq in (0.7,1.7,2.7):
   fits.append(least_squares(lambda p:pred(p,D[:52])-y[:52],np.r_[g,0,amp,freq],bounds=(np.r_[np.full(len(g)+2,-np.inf),.12],np.r_[np.full(len(g)+2,np.inf),4.]),max_nfev=400))
 p=min(fits,key=lambda f:f.cost).x; print(name,'y train/tail',np.mean(abs(pred(p,D[:52])-y[:52])),np.mean(abs(pred(p,D[52:])-y[52:])),'parameters',np.round(p,4))
PY
```
- **Container Exit Code**: `0` (`942 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
float) - vy)))
    tm = float(np.mean(np.abs(np.asarray(out_cv["state_pred"], float) - vtr)))
    cv_metrics = {"public_cv_y_mae": round(ym, 6), "public_cv_traj_mae": round(tm, 6)}
    if ym > 0.038: errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
    if tm > 0.038: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")

ok = len(errs) == 0 and all(v == 1 for v in ec.values())
print(json.dumps({"accepted": ok, "errors": errs, "cv_metrics": cv_metrics, "exec_counts": ec}, allow_nan=False))
sys.exit(0 if ok else 1)
Panel C v2 diagonal model MAE 0.00380410733016496
Panel C v2 diagonal model [[ 0.0578  0.0101 -0.1438  0.0119]
 [-0.1387 -0.3196  0.2951  0.1607]
 [ 0.1706 -0.2268 -0.3075  0.2448]
 [-0.4238 -0.0306 -0.2239 -0.0202]
 [-0.0755  0.2788  0.0664  0.3861]
 [-0.0625 -0.1369  0.1314  0.0753]
 [ 0.0794 -0.0982 -0.1374  0.1124]
 [-0.195  -0.0159 -0.0949 -0.0075]
 [-0.0284  0.1275  0.0223  0.1695]]
trajectory CV from Panel C v2 0.3713070029780455
trajectory CV using diagonal quadratic 0.23105941830147372
trajectory coefficients in first three modes [[-3.0790e-01 -2.5000e-03  4.0000e-04]
 [-1.0590e-01  1.5000e-03 -5.3000e-03]
 [-6.8420e-01  5.0000e-03 -1.8300e-02]
 [ 1.4240e+00  5.8000e-03  3.7800e-02]
 [-3.4996e+00 -4.1000e-02  1.8300e-02]
 [-1.1840e-01  3.0000e-03  1.1000e-03]
 [-1.3472e+00  6.9600e-02  1.6600e-02]
 [-1.7410e-01  8.3000e-03  2.8000e-03]
 [-5.2300e-02  3.8000e-03 -5.0000e-04]
 [ 2.1840e-01 -1.1500e-02 -3.0000e-03]
 [ 6.8600e-02 -2.5000e-03 -1.6000e-03]
 [ 1.9740e-01 -9.3000e-03 -3.1000e-03]
 [ 1.7760e-01 -9.8000e-03 -1.1000e-03]
 [-1.3652e+00  6.8900e-02  1.1600e-02]
 [-6.3800e-02  6.4000e-03  2.3000e-03]]
interaction singular values [1.96555854 0.0053905  0.00405106]
affine y train/tail 0.32475253139068827 0.4005668920913831 parameters [ 0.1015  0.0362  0.1454  0.0559 -0.137  -0.4407  2.4919  4.    ]
squares y train/tail 0.3096834760020541 0.41561118105470546 parameters [-0.1616  0.2692  1.2333  0.5161 -1.107   0.0679  0.247  -0.168   0.0425
  2.7673  0.2269  4.    ]
all quadratic y train/tail 0.04163549571189876 0.04873963784730838 parameters [-0.0486  0.1489  1.0194  0.3313 -0.9738 -0.0871 -0.73   -0.104   0.0113
  0.0971  0.0699  0.0349  0.0574 -0.4743  0.0048 -0.1466  0.3282  1.4345]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f70158ce7d3156ba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f70158ce (directory).
```

### Turn 8 — `execute_in_remote_docker` (`47840 ms` LLM)
- **Rationale**: The trajectory’s quadratic coefficients are nearly rank one. If that factor is mixed into the latent coordinates, the frozen pair products generate factor-by-channel and squared-factor terms. Fit the factor and readout link inside each training fold and compare against unrestricted cubics before choosing a handoff.
- **Alternatives Weighed**: `Use Panel C's second map as the coordinate correction` (rejected: It gives a held-out trajectory MAE of 0.371307, so it does not explain the shared trajectory structure.)

```bash
cat >/tmp/shared_index.py <<'PY'
import sys,pickle,itertools as it
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.optimize import least_squares
import node_00 as n
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; u=n._invert(s['train']['x'],n._fit(a['u'],a['v'])); t=s['train']['traj']; y=s['train']['y']; d=u.shape[1]
def aff(x): return np.c_[np.ones(len(x)),x]
def terms(x,g): return np.column_stack([np.prod(x[:,p],axis=1) for p in it.combinations_with_replacement(range(d),g)])
A=aff(u); Q=terms(u,2); C=terms(u,3); D=np.c_[A,Q]; rng=np.random.default_rng(429)
splits=[(np.arange(52),np.arange(52,72))]+[(j[:52],j[52:]) for j in [rng.permutation(len(u)) for _ in range(4)]]
def pred(p,X,link):
 q=X@p[:-3]; o,b,f=p[-3:]; return q+b*(np.tanh(f*q+o) if link=='tanh' else np.sin(f*q+o))
errs={}
for num,(tr,te) in enumerate(splits):
 m=np.linalg.svd(t[tr],full_matrices=False)[2]; tc=np.linalg.lstsq(D[tr],t[tr]@m[:2].T,rcond=None)[0]; h=Q@tc[5:,0]; h/=np.std(h[tr]); S=D@tc[:,0]; S=(S-np.mean(S[tr]))/np.std(S[tr]); forms={'affine+h':np.c_[A,h], 'quad':D,'quad+h*u':np.c_[D,h[:,None]*u], 'quad+h*u+h2':np.c_[D,h[:,None]*u,h**2], 'quad+S*u+S2':np.c_[D,S[:,None]*u,S**2], 'quad+full_cubic':np.c_[D,C]}
 for name,F in forms.items():
  for link in ['tanh','sin']:
   fits=[]; Ftr=F[tr]; yy=y[tr]; ini=np.linalg.lstsq(Ftr,yy,rcond=None)[0]
   for f in (.7,1.7,3.):
    for b in (.2,.45):
     fits.append(least_squares(lambda p:pred(p,Ftr,link)-yy,np.r_[ini,0,b,f],bounds=(np.r_[np.full(len(ini)+2,-np.inf),.12],np.r_[np.full(len(ini)+2,np.inf),4.]),max_nfev=150))
   p=min(fits,key=lambda f:f.cost).x; loss=np.mean(abs(pred(p,F[te],link)-y[te])); errs.setdefault((name,link),[]).append(loss)
   if num==0: print(name,link,'train,tail',round(np.mean(abs(pred(p,Ftr,link)-yy)),6),round(loss,6),'link params',np.round(p[-3:],4))
 if num==0:
  print('regression residual per time',np.std(t[tr]-D[tr]@np.linalg.lstsq(D[tr],t[tr],rcond=None)[0],axis=0)); print('trajectory top two quadratic scores coefficients',np.round(tc[5:].T,4))
for key,es in errs.items(): print(key,np.round(es,5),'worst',round(max(es),6))
PY
OPENBLAS_NUM_THREADS=1 python3 -B /tmp/shared_index.py
```
- **Container Exit Code**: `0` (`3762 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
+h tanh train,tail 0.072098 0.072212 link params [-0.7432  0.1403  2.368 ]
affine+h sin train,tail 0.071158 0.088257 link params [-0.5263  0.0857  2.1845]
quad tanh train,tail 0.041635 0.04874 link params [-0.1466  0.3282  1.4345]
quad sin train,tail 0.041443 0.050818 link params [-0.1452  0.1323  1.4732]
quad+h*u tanh train,tail 0.039165 0.05665 link params [-0.215   0.2981  1.4021]
quad+h*u sin train,tail 0.035487 0.05085 link params [1.3193 0.4951 0.8851]
quad+h*u+h2 tanh train,tail 0.038859 0.052563 link params [1.6885 2.0456 0.9767]
quad+h*u+h2 sin train,tail 0.03522 0.04945 link params [1.4113 0.546  0.8243]
quad+S*u+S2 tanh train,tail 0.038859 0.052562 link params [1.688  2.045  0.9767]
quad+S*u+S2 sin train,tail 0.03522 0.04945 link params [1.4112 0.546  0.8244]
quad+full_cubic tanh train,tail 0.005384 0.044349 link params [-0.4534  0.1141  2.2448]
quad+full_cubic sin train,tail 0.007068 0.044549 link params [-0.3053  0.0459  1.6794]
regression residual per time [0.00437292 0.00406627 0.00380388 0.00446524 0.00361908 0.00420419
 0.00420544 0.00441629]
trajectory top two quadratic scores coefficients [[-0.1184 -1.3472 -0.1741 -0.0523  0.2184  0.0686  0.1974  0.1776 -1.3652
  -0.0638]
 [ 0.003   0.0696  0.0083  0.0038 -0.0115 -0.0025 -0.0093 -0.0098  0.0689
   0.0064]]
('affine+h', 'tanh') [0.07221 0.08019 0.08936 0.09096 0.06871] worst 0.090964
('affine+h', 'sin') [0.08826 0.07629 0.09457 0.0884  0.06784] worst 0.094568
('quad', 'tanh') [0.04874 0.04229 0.06043 0.07299 0.06328] worst 0.072987
('quad', 'sin') [0.05082 0.04618 0.05697 0.06755 0.06604] worst 0.067551
('quad+h*u', 'tanh') [0.05665 0.0505  0.06403 0.07415 0.06972] worst 0.074153
('quad+h*u', 'sin') [0.05085 0.0378  0.06227 0.07289 0.05909] worst 0.072888
('quad+h*u+h2', 'tanh') [0.05256 0.04664 0.06373 0.07954 0.07782] worst 0.079539
('quad+h*u+h2', 'sin') [0.04945 0.04103 0.06198 0.07637 0.06864] worst 0.076374
('quad+S*u+S2', 'tanh') [0.05256 0.04664 0.06373 0.07954 0.07782] worst 0.079539
('quad+S*u+S2', 'sin') [0.04945 0.04103 0.06198 0.07637 0.06864] worst 0.076374
('quad+full_cubic', 'tanh') [0.04435 0.03225 0.03931 0.04731 0.03101] worst 0.047312
('quad+full_cubic', 'sin') [0.04455 0.03545 0.0445  0.07479 0.03125] worst 0.074787

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f70158ce7d3156ba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f70158ce (directory).
```

### Turn 9 — `execute_in_remote_docker` (`42853 ms` LLM)
- **Rationale**: Inspect cubic coefficients in both coordinate frames and select terms within each training split. If sparse terms improve only the official tail and not the other splits, favor a regularized model over a tail-specific basis.
- **Alternatives Weighed**: `Adopt the unrestricted cubic index` (rejected: It fits training rows closely but misses the 0.038 tolerance on held-out rows; the coefficient patterns may reveal a smaller, better-conditioned basis.)
- **ATMS NOGOOD DECLARED**: Refuted `Interactions generated solely by the trajectory's shared quadratic factor capture the remaining readout dependence.` | Witness: `Adding the factor-by-channel and squared-factor terms gave tail MAE 0.052563 and worst-slice MAE 0.079539. Independent cubic terms reduced those errors to 0.044349 and 0.047312.` | Blanket Diagnosis: `The trajectory identifies one important nonlinear direction, but the readout contains nonlinear terms outside the basis generated by that direction.` | Sublated Pivot: `Compare sparse and regularized cubic bases in both calibrated coordinate frames, then use the surviving nonlinear features in the editable node.`

```bash
cat >/tmp/sparse_cal.py <<'PY'
import sys,pickle,itertools as it
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.optimize import least_squares
import node_00 as n
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; x=n._invert(s['train']['x'],n._fit(a['u'],a['v'])); y=s['train']['y']; t=s['train']['traj']; b=s['panel_b']
A=lambda x:np.c_[np.ones(len(x)),x]
B=np.linalg.lstsq(A(b['u']),b['v'],rcond=None)[0]
terms=[p for k in (2,3) for p in it.combinations_with_replacement(range(4),k)]
def fit(D,y,l=0):
 p=np.linalg.lstsq(D,y,rcond=None)[0]; reg=np.r_[np.zeros(5),np.full(D.shape[1]-5,np.sqrt(l))]
 def pred(p,D):
  q=D@p[:-3]; o,a,f=p[-3:]; return q+a*np.tanh(f*q+o)
 def res(p):return np.r_[pred(p,D)-y,reg*p[:-3]]
 fs=[least_squares(res,np.r_[p,0,a,f],bounds=(np.r_[np.full(len(p)+2,-np.inf),.12],np.r_[np.full(len(p)+2,np.inf),4]),max_nfev=180) for a,f in [(0.15,1),(0.2,2),(0.35,3)]]
 return min(fs,key=lambda p:p.cost).x,pred
rng=np.random.default_rng(429); splits=[np.arange(len(x))]+[rng.permutation(len(x)) for _ in range(4)]
for frame,X in [('u',x),('z',A(x)@B)]:
 D=np.c_[A(X),*[np.prod(X[:,p],axis=1) for p in terms]]
 p,_=fit(D[:52],y[:52]); print(frame,'cubic',[(q,round(c,3)) for q,c in zip(terms,p[5:-3])]); print('link',np.round(p[-3:],3))
 ts=np.linalg.lstsq(D[:52,:15],t[:52],rcond=None)[0]; print('trajectory quadratic norms',np.round(np.linalg.norm(ts[5:],axis=1),3))
 for name,ids in [('all',range(35)),('pure',list(range(15))+[15+i for i,q in enumerate(terms[10:]) if len(set(q))==1]),('no-three',list(range(15))+[15+i for i,q in enumerate(terms[10:]) if len(set(q))<3])]:
  d=D[:,ids];
  for l in (0,.003,.03,.15):
   losses=[]
   for j in splits:
    tr,te=j[:52],j[52:]; p,pred=fit(d[tr],y[tr],l); losses.append(np.mean(abs(pred(p,d[te])-y[te])))
   print(frame,name,l,np.round(losses,5))
 for count in (5,8,11,15):
  losses=[]
  for j in splits:
   tr,te=j[:52],j[52:]; ids=list(range(35)); scale=np.std(D[tr],axis=0); scale[0]=1
   while len(ids)>5+count:
    p,_=fit(D[tr][:,ids],y[tr],.003); score=abs(p[5:-3])*scale[ids[5:]]; k=np.argsort(score)[:min(4,len(ids)-5-count)]; ids=[i for k0,i in enumerate(ids) if k0-5 not in k or k0<5]
   p,pred=fit(D[tr][:,ids],y[tr]); losses.append(np.mean(abs(pred(p,D[te][:,ids])-y[te])))
  print(frame,'sparse',count,np.round(losses,5))
PY
OPENBLAS_NUM_THREADS=1 python3 -B /tmp/sparse_cal.py
```
- **Container Exit Code**: `0` (`31055 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
37 0.04013]
u pure 0 [0.0549  0.04877 0.05962 0.07828 0.06007]
u pure 0.003 [0.05484 0.04895 0.05951 0.07814 0.06045]
u pure 0.03 [0.05432 0.05042 0.05829 0.07683 0.06893]
u pure 0.15 [0.06149 0.05625 0.05296 0.08116 0.06904]
u no-three 0 [0.0619  0.04491 0.04368 0.07341 0.04595]
u no-three 0.003 [0.06182 0.04601 0.04341 0.07275 0.04561]
u no-three 0.03 [0.05225 0.05065 0.04688 0.06593 0.06321]
u no-three 0.15 [0.05231 0.05018 0.05079 0.06795 0.06213]
u sparse 5 [0.073   0.07493 0.07579 0.09677 0.07491]
u sparse 8 [0.05123 0.0702  0.04852 0.07374 0.04068]
u sparse 11 [0.0697  0.04249 0.03851 0.06114 0.04622]
u sparse 15 [0.05737 0.0428  0.04504 0.05674 0.04225]
z cubic [((0, 0), 0.122), ((0, 1), -0.011), ((0, 2), 0.335), ((0, 3), -0.782), ((1, 1), 0.1), ((1, 2), -0.856), ((1, 3), 0.14), ((2, 2), -0.036), ((2, 3), -0.043), ((3, 3), -0.083), ((0, 0, 0), -0.033), ((0, 0, 1), -0.179), ((0, 0, 2), -0.194), ((0, 0, 3), 0.055), ((0, 1, 1), -0.087), ((0, 1, 2), -0.318), ((0, 1, 3), 0.087), ((0, 2, 2), -0.228), ((0, 2, 3), 0.18), ((0, 3, 3), 0.0), ((1, 1, 1), -0.075), ((1, 1, 2), -0.166), ((1, 1, 3), 0.016), ((1, 2, 2), -0.21), ((1, 2, 3), 0.188), ((1, 3, 3), 0.017), ((2, 2, 2), -0.085), ((2, 2, 3), 0.097), ((2, 3, 3), -0.001), ((3, 3, 3), 0.018)]
link [-0.453  0.114  2.245]
trajectory quadratic norms [0.284 0.083 0.261 1.741 0.228 1.766 0.225 0.081 0.069 0.155]
z all 0 [0.04435 0.03225 0.03931 0.04731 0.03101]
z all 0.003 [0.04413 0.03725 0.03294 0.04403 0.03558]
z all 0.03 [0.05859 0.04167 0.05842 0.05293 0.04026]
z all 0.15 [0.06042 0.04049 0.05978 0.05131 0.03913]
z pure 0 [0.05719 0.04896 0.06843 0.07099 0.06573]
z pure 0.003 [0.05734 0.04898 0.06836 0.07001 0.06625]
z pure 0.03 [0.05824 0.05214 0.06989 0.06793 0.07574]
z pure 0.15 [0.06308 0.05209 0.07008 0.07105 0.07553]
z no-three 0 [0.05779 0.05871 0.06751 0.06797 0.04693]
z no-three 0.003 [0.05686 0.05968 0.0648  0.06749 0.05797]
z no-three 0.03 [0.04929 0.08263 0.06881 0.07185 0.06347]
z no-three 0.15 [0.0549  0.08163 0.06798 0.05525 0.0625 ]
z sparse 5 [0.06099 0.05132 0.08106 0.06957 0.08015]
z sparse 8 [0.06437 0.05743 0.06007 0.07169 0.07656]
z sparse 11 [0.05225 0.0442  0.05536 0.06045 0.0652 ]
z sparse 15 [0.05227 0.04085 0.05652 0.05621 0.05814]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f70158ce7d3156ba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f70158ce (directory).
```

### Turn 10 — `execute_in_remote_docker` (`36690 ms` LLM)
- **Rationale**: Test compact Panel C maps through the actual frozen readout and operator conventions. If none improves both errors, use the low-rank trajectory structure to guide the final latent correction rather than treating Panel C as interchangeable with Panel A.
- **Alternatives Weighed**: `Promote the smallest sparse cubic fit` (rejected: None of the tested sparse bases improved worst-slice readout error enough to clear tolerance.)

```bash
cat > /tmp/probe10.py <<'PY'
import sys,pickle,itertools as it,importlib.util
sys.path.insert(0,'/holdout_task')
import numpy as np
import node_00 as n
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; c=s['panel_c']; b=s['panel_b']; t=s['train']['traj']; y=s['train']['y']; raw=s['train']['x']; p=n._fit(a['u'],a['v']); u=n._invert(raw,p)
def aff(x):return np.c_[np.ones(len(x)),x]
def X(x,terms):return np.c_[aff(x),*[np.prod(x[:,q],axis=1) for q in terms]]
q=list(it.combinations_with_replacement(range(4),2)); cube=list(it.combinations_with_replacement(range(4),3)); sq=[(j,j) for j in range(4)]; adj=[(j,(j+1)%4) for j in range(4)]; cu=[(j,)*3 for j in range(4)]
B=np.linalg.lstsq(aff(b['u']),b['v'],rcond=None)[0]; A=lambda x:aff(x)@B; AI=lambda z:np.linalg.solve(B[1:].T,(z-B[0]).T).T
sp=importlib.util.spec_from_file_location('r','/holdout_task/_frozen_nodes/node_02.py'); r=importlib.util.module_from_spec(sp); sp.loader.exec_module(r)
def score(name,z):
 out=r.run_node({'node_01':{'z_tr':z[:52],'z_te':z[52:],'y':y[:52],'traj':t[:52]}}); print(name, np.mean(abs(out['y_hat']-y[52:])),np.mean(abs(out['state_pred']-t[52:])),flush=True)
print('keys',list(s),'telemetry',s.get('telemetry_vec')); print('A fit',p)
for k in ('v1','v2'):
 for name,ts in [('squares',sq),('adj',adj),('own3',cu),('adj+own3',adj+cu),('sq+own3',sq+cu),('q+own3',q+cu)]:
  D=X(c['u'],ts); C=np.linalg.lstsq(D,c[k],rcond=None)[0]; h=np.sum(np.linalg.qr(D)[0]**2,1); err=np.mean(abs((D@C-c[k])/(1-h)[:,None])); print('panel',k,name,round(err,6));
  if name in ('adj+own3','sq+own3'):
   print(np.round(C[5:],4))
   F=lambda x:X(x,ts)@C
   for key,x in [('u',u),('z',A(u))]:
    score(k+name+key+' direct',F(x)); score(k+name+key+' affine out',A(F(x)))
# Check whether the dominant rank-one trajectory term is the only nonlinear part of the trajectory.
z=A(u); D=X(z,q); P=np.linalg.lstsq(D[:52],t[:52],rcond=None)[0]; U,S,V=np.linalg.svd(P[5:],full_matrices=False); print('trajectory quadratic singular values',S,'lead coefs',U[:,0]);
PY
OPENBLAS_NUM_THREADS=1 python3 -B /tmp/probe10.py
```
- **Container Exit Code**: `0` (`2205 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
own3u direct 0.3183916550963664 0.27352418221030794
v1adj+own3u affine out 0.4057076563737 0.27352418221030783
v1adj+own3z direct 0.3984381669317154 0.292715337901301
v1adj+own3z affine out 0.5134678984746942 0.2927153379013011
panel v1 sq+own3 0.053874
[[-0.0049 -0.0073  0.0379 -0.005 ]
 [ 0.0243  0.0174 -0.0088  0.0146]
 [-0.0221  0.0364  0.0316 -0.0076]
 [ 0.0122 -0.0272 -0.0215 -0.0322]
 [ 0.2868  0.0741  0.1374  0.171 ]
 [ 0.22    0.2321 -0.0849 -0.2495]
 [ 0.0872 -0.2862  0.2322 -0.1626]
 [ 0.2064 -0.2192 -0.303  -0.0564]]
v1sq+own3u direct 0.35506952132704317 0.26924859258003614
v1sq+own3u affine out 0.42181294798228486 0.26924859258003614
v1sq+own3z direct 0.4601115468726924 0.31511800947996205
v1sq+own3z affine out 0.528668827827455 0.3151180094799633
panel v1 q+own3 0.110369
panel v2 squares 0.006288
panel v2 adj 0.105977
panel v2 own3 0.11303
panel v2 adj+own3 0.12391
[[-0.0534  0.0307  0.0687  0.0221]
 [ 0.1093  0.0405  0.0125  0.0354]
 [-0.1029 -0.071  -0.2165  0.2383]
 [ 0.054   0.0294  0.028   0.04  ]
 [ 0.1098  0.0713 -0.2102 -0.0548]
 [-0.1432  0.0163  0.1039 -0.0779]
 [-0.1034 -0.0172 -0.2532  0.0624]
 [ 0.0357 -0.0802 -0.1534  0.0914]]
v2adj+own3u direct 0.4353961253154005 0.20167124523276142
v2adj+own3u affine out 0.43803194531959566 0.20167124523276128
v2adj+own3z direct 0.39039640098449907 0.32134704741259945
v2adj+own3z affine out 0.414034611163627 0.32134704741259945
panel v2 sq+own3 0.007924
[[-0.0643 -0.145   0.1294  0.0752]
 [ 0.0795 -0.1006 -0.1398  0.1133]
 [-0.1946 -0.0132 -0.0936 -0.0063]
 [-0.0292  0.1229  0.0226  0.1701]
 [-0.0046 -0.0165 -0.004   0.0013]
 [ 0.0022 -0.0012 -0.0082  0.0043]
 [-0.0005 -0.015  -0.0062 -0.0044]
 [ 0.0023 -0.0063 -0.0002  0.0085]]
v2sq+own3u direct 0.45619270011672075 0.36600972328494524
v2sq+own3u affine out 0.4563330422097979 0.36600972328494497
v2sq+own3z direct 0.3782856876042326 0.2853958092612857
v2sq+own3z affine out 0.39362336091586586 0.2853958092612852
panel v2 q+own3 0.009536
trajectory quadratic singular values [2.53808402 0.01955997 0.0101578  0.00680471 0.0065236  0.00390857
 0.00300884 0.00256035] lead coefs [-0.11185635 -0.03226502 -0.10265657  0.68588531 -0.08966822  0.69579422
  0.088586    0.03169106  0.02712222  0.06093192]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f70158ce7d3156ba (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f70158ce (directory).
```

### Turn 11 — `execute_in_remote_docker` (`75987 ms` LLM)
- **Rationale**: Retain the verified Panel A inverse, fit the latent correction using only each split’s training rows, and test both outputs through the unmodified graph. The clean-process official test then verifies the saved artifact.
- **Alternatives Weighed**: `Apply a Panel C map directly` (rejected: None of the tested maps brought trajectory MAE below 0.20 through the frozen readout.)

```bash
set -e
cd /holdout_task
cat _frozen_nodes/node_02.py
cp node_00.py /tmp/panel_inverse.py
sed '/^def run_node/,$d' /tmp/panel_inverse.py > /tmp/calibration_core.py
cat >> /tmp/calibration_core.py <<'PY'
from itertools import combinations_with_replacement
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _polynomial(x, degree):
    return np.column_stack([np.prod(x[:, q], axis=1)
                            for k in range(degree + 1)
                            for q in combinations_with_replacement(range(x.shape[1]), k)])


def _index(d, y, n, mode):
    a = d[:n]
    if mode == 0:
        # A cubic index with a bounded nonlinear correction of that same index.
        initial = np.linalg.lstsq(a, y, rcond=None)[0]
        penalty = np.r_[np.zeros(15), np.full(20, np.sqrt(0.002))]
        def predict(p, x):
            z = x @ p[:-3]
            return z + p[-2] * np.tanh(p[-1] * z + p[-3])
        fits = [least_squares(lambda p: np.r_[predict(p, a)-y, penalty*p[:-3]],
                             np.r_[initial, 0., .2, f], max_nfev=100,
                             bounds=(np.r_[np.full(37, -np.inf), .2], np.r_[np.full(37, np.inf), 4.]))
                for f in (1., 2.5)]
        return predict(min(fits, key=lambda p: p.cost).x, d)

    # Profile the linear index out of the bounded additive correction.
    # Penalize cubic terms more than quadratic terms to stabilize extrapolation.
    degree = 2 if mode in (1, 2) else 3
    m = 15 if degree == 2 else 35
    a = a[:, :m]
    interaction = d[:, :15 if mode in (2, 4) else 5]
    b = interaction[:n]
    reg = np.r_[np.zeros(5), np.full(10, .003), np.full(m-15, .035)]
    ext = np.vstack((a, np.diag(np.sqrt(reg))))
    pinv = np.linalg.pinv(ext)
    target = np.r_[y, np.zeros(m)]
    py = target - ext @ (pinv @ target)
    project = np.eye(n+m) - ext @ pinv
    proj = project[:, :n]
    shrink = np.r_[.002, np.full(4, .012), np.full(b.shape[1]-5, .040)]
    def fun(p):
        h = np.tanh(b @ p[:-1])
        return np.r_[py - p[-1]*(proj@h), shrink*p[:-1]]
    def jac(p):
        h = np.tanh(b @ p[:-1]); g = np.empty((n+m+len(shrink), len(p)))
        g[:n+m, :-1] = -p[-1]*(proj @ ((1-h*h)[:,None]*b))
        g[:n+m, -1] = -(proj@h)
        g[n+m:] = 0.; g[n+m:, :-1] = np.diag(shrink)
        return g
    base = np.linalg.lstsq(b, y, rcond=None)[0]
    rng = np.random.default_rng(103)
    starts = [base, -base, *rng.normal(0, .65, size=(5,len(base)))]
    fits = [least_squares(fun, np.r_[q, .22], jac=jac, max_nfev=75,
                          bounds=(np.r_[np.full(len(q), -np.inf), -.65], np.r_[np.full(len(q), np.inf), .65]))
            for q in starts]
    p = min(fits, key=lambda w:w.cost).x
    correction = p[-1]*np.tanh(interaction@p[:-1])
    coef = pinv @ (target - np.r_[correction[:n], np.zeros(m)])
    return d[:, :m] @ coef + correction


def _coordinates(inputs, mode, chart):
    study = inputs['__input__']; panel = study['panel_a']
    train = study['train']; n = len(train['x'])
    x = _invert(np.vstack((train['x'], study['test']['x'])), _fit(panel['u'], panel['v']))
    b = study['panel_b']; operator = np.linalg.lstsq(_affine(b['u']), b['v'], rcond=None)[0]
    z = _affine(x) @ operator
    d = _polynomial(x, 3)
    trajectory = train['traj']
    centered = trajectory - trajectory.mean(axis=0)
    modes = np.linalg.svd(centered, full_matrices=False)[2][:2]
    scores = centered @ modes.T
    # Both time modes share the same quadratic coordinate. Estimate that
    # coordinate jointly rather than fitting independent interactions.
    q = d[:, 5:15]; a = d[:, :5]
    independent = q[:n] - a[:n] @ np.linalg.lstsq(a[:n], q[:n], rcond=None)[0]
    weights = np.linalg.lstsq(independent, scores, rcond=None)[0]
    left, _, _ = np.linalg.svd(weights, full_matrices=False)
    f = np.column_stack((a, q @ left[:, 0]))
    scores = f @ np.linalg.lstsq(f[:n], scores, rcond=None)[0]
    signal = _index(d, train['y'], n, mode)
    extra = z[:, 0] if chart == 0 else z[:, 1] if chart == 1 else z[:, 3]
    target = np.column_stack((signal, scores, extra))
    target = (target - target[:n].mean(axis=0)) / np.maximum(target[:n].std(axis=0), 1e-8)
    calibrated = np.linalg.solve(operator[1:].T, (target - operator[0]).T).T
    return {'x_cal_tr': calibrated[:n], 'x_cal_te': calibrated[n:]}
PY
cat > /tmp/select_calibration.py <<'PY'
import pickle, importlib.util, numpy as np
from pathlib import Path
ns={}; core=Path('/tmp/calibration_core.py').read_text(); exec(core, ns)
def frozen(name):
 p=f'/holdout_task/_frozen_nodes/{name}.py'; spec=importlib.util.spec_from_file_location(name,p); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod);return mod.run_node
operator, readout=map(frozen,('node_01','node_02'))
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); rng=np.random.default_rng(429)
# Refit on every split, including the panel coefficients and both latent indices.
orders=[np.arange(n)]+[rng.permutation(n) for _ in range(4)]; results=[]
for mode in range(5):
 for chart in range(3):
  errors=[]
  for j in orders:
   a,b=j[:-20],j[-20:]; t={**s,'train':{k:tr[k][a] for k in ('x','y','traj')},'test':{'x':tr['x'][b]}}
   cal=ns['_coordinates']({'__input__':t},mode,chart); y=readout({'node_01':operator({'__input__':t,'node_00':cal})}); errors.append([np.mean(abs(y['y_hat']-tr['y'][b])),np.mean(abs(y['state_pred']-tr['traj'][b]))])
  es=np.array(errors); print(mode,chart,np.round(es,5).tolist(),flush=True)
  results.append((np.max(es[0])>.038,np.max(es),np.mean(es),mode,chart))
_,worst,_,mode,chart=min(results)
print('selected',mode,chart,'worst',worst,flush=True)
Path('/holdout_task/node_00.py').write_text(core+f'\n\ndef run_node(inputs: dict) -> dict:\n    return _coordinates(inputs, {mode}, {chart})\n')
PY
OPENBLAS_NUM_THREADS=1 python3 -B /tmp/select_calibration.py
OPENBLAS_NUM_THREADS=1 python3 -I -B /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`11899 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
5)]
    p = min(candidates, key=lambda fit: fit.cost).x
    y_hat = _readout(p, _readout_design(z_test))

    traj = features['traj']
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    state_pred = (_affine(z_test) @ _fit(_affine(z), traj @ modes.T)) @ modes
    last = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
0 0 [[0.0441, 0.00881], [0.03213, 0.00777], [0.0379, 0.00845], [0.04633, 0.00941], [0.0315, 0.00892]]
0 1 [[0.04423, 0.00881], [0.03234, 0.00777], [0.03859, 0.00845], [0.04633, 0.00941], [0.0316, 0.00892]]
0 2 [[0.04428, 0.00881], [0.03183, 0.00777], [0.03832, 0.00845], [0.04622, 0.00941], [0.03181, 0.00891]]
1 0 [[0.01446, 0.00881], [0.00987, 0.00777], [0.01368, 0.00845], [0.01776, 0.00941], [0.01315, 0.00892]]
1 1 [[0.01405, 0.00881], [0.01002, 0.00777], [0.01409, 0.00845], [0.01815, 0.00941], [0.01297, 0.00892]]
1 2 [[0.0141, 0.00881], [0.00975, 0.00777], [0.01403, 0.00845], [0.01763, 0.00941], [0.01328, 0.00891]]
2 0 [[0.00673, 0.00881], [0.00597, 0.00777], [0.00912, 0.00845], [0.00859, 0.00941], [0.00505, 0.00892]]
2 1 [[0.00703, 0.00881], [0.00621, 0.00777], [0.00918, 0.00845], [0.00853, 0.00941], [0.00518, 0.00892]]
2 2 [[0.00691, 0.00881], [0.00607, 0.00777], [0.00916, 0.00845], [0.00853, 0.00941], [0.0052, 0.00891]]
3 0 [[0.02941, 0.00881], [0.01479, 0.00777], [0.01911, 0.00845], [0.03637, 0.00941], [0.01741, 0.00892]]
3 1 [[0.02942, 0.00881], [0.01489, 0.00777], [0.0192, 0.00845], [0.03645, 0.00941], [0.01748, 0.00892]]
3 2 [[0.02936, 0.00881], [0.01479, 0.00777], [0.01924, 0.00845], [0.03628, 0.00941], [0.01773, 0.00891]]
4 0 [[0.01803, 0.00881], [0.01078, 0.00777], [0.01192, 0.00845], [0.01776, 0.00941], [0.01472, 0.00892]]
4 1 [[0.01801, 0.00881], [0.01079, 0.00777], [0.01191, 0.00845], [0.01775, 0.00941], [0.01465, 0.00892]]
4 2 [[0.01804, 0.00881], [0.0108, 0.00777], [0.01172, 0.00845], [0.01776, 0.00941], [0.01464, 0.00891]]
selected 2 0 worst 0.009413543915554207
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.006735, "public_cv_traj_mae": 0.008812}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=07a75ae3f3c9f75b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_07a75ae3 (directory).
```

### Turn 12 — `submit_completed_task` (`14384 ms` LLM)
- **Rationale**: The saved artifact passed the official test in an isolated Python process. The five split checks refit the coordinate model using only training rows and verified both outputs through the unchanged graph.
- **Alternatives Weighed**: `Use an unrestricted cubic readout index` (rejected: It exceeded the readout tolerance on the official split; the bounded nonlinear correction stayed below tolerance on all five tested splits.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_00.py`. The frozen nodes were unchanged. The official test passed with readout MAE 0.006735 and trajectory MAE 0.008812, both below 0.038. Each node ran exactly once.`

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
