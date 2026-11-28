# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m102_b03_batch_11__blind_campaign__reuse_subgraph`

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
| **Total Tokens (Prompt / Cached / Out)** | `362,331` (`348,458` / `125,440` / `13,873`) |
| **Total Episode Wall Time (s)** | `516.66s` |

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

### Turn 1 — `execute_in_remote_docker` (`12435 ms` LLM)
- **Rationale**: Inspect the frozen consumers, public measurements, validator, and reference archive before choosing a numerical model.

```bash
cd /holdout_task && cat _dag_spec.json && cat _frozen_nodes/node_01.py _frozen_nodes/node_02.py && cat public_asset_index.json && sed -n '1,250p' test_dag_candidate.py && find /tmp/hegel_continual_archive -maxdepth 3 -type f | sort | head -120 && python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('public_batch.pkl','rb'))
def show(x,p='root'):
 if isinstance(x,dict):
  for k,v in x.items():show(v,p+'/'+str(k))
 else:
  a=np.asarray(x);print(p,a.shape,a.dtype,str(a.ravel()[:10]));
  if np.issubdtype(a.dtype,np.number) and a.size: print(' range',np.nanmin(a),np.nanmax(a))
show(x)
PY
```
- **Container Exit Code**: `0` (`371 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
52202e-01,  1.27903967e+00, -8.14352805e-01,
         -1.24752850e+00],
        [ 7.70653788e-01, -9.12802594e-01, -8.47562677e-01,
         -8.50340881e-01],
        [ 1.03197864e+00, -1.08278844e+00,  5.51062732e-01,
         -1.11541071e+00],
        [-1.09388934e+00, -7.93669545e-02,  5.61949998e-02,
          9.59044328e-02],
        [-9.53435633e-01,  5.61539973e-04,  1.03007609e+00,
         -6.30408786e-01],
        [-9.33321149e-01,  6.66506241e-01,  7.54564691e-01,
          3.32963008e-01],
        [ 8.07370218e-01,  2.62769721e-01, -1.61329682e+00,
         -4.52541912e-01],
        [ 9.24154823e-01,  1.23075775e+00, -1.45952524e+00,
         -1.01684868e+00],
        [ 8.95516684e-01, -1.23427967e+00,  1.10172882e+00,
          4.83977558e-01],
        [-9.33034729e-02, -9.68334997e-01, -1.69793939e-01,
          1.64016268e+00],
        [-2.26634265e-01, -5.13572583e-01,  4.23205291e-01,
         -4.86263220e-01],
        [ 5.50382600e-02, -1.12109105e+00, -3.25306349e-02,
          2.50437161e-01],
        [ 1.10607691e+00, -1.90183040e-01, -1.93857722e+00,
         -9.96356686e-01],
        [ 2.64052363e-01,  8.84006169e-01, -9.04346339e-01,
         -1.45727316e+00],
        [ 4.18547500e-01,  3.48837402e-01,  1.08485731e+00,
         -5.52937461e-01],
        [ 1.06649210e+00, -1.60028381e+00, -8.23407692e-01,
          6.78057393e-01],
        [ 7.15528954e-01, -3.78706486e-01, -7.07356233e-01,
          1.05804587e+00],
        [-1.68119774e-01,  4.91135361e-01, -8.71028372e-02,
         -9.56612832e-01],
        [-1.75332269e+00, -2.89316736e-02,  1.18989070e+00,
         -6.49214906e-01],
        [-1.46345569e+00, -7.91335853e-01,  1.27189974e-01,
         -4.42751746e-01],
        [ 1.30782491e+00, -6.94570934e-01, -1.78055849e+00,
          1.68810156e+00],
        [-5.18729243e-01,  4.84831444e-01,  9.83243430e-01,
          9.63725509e-01],
        [-6.20827461e-01, -4.21198678e-01, -1.26860225e+00,
          4.81403123e-01],
        [ 9.79201111e-01,  9.79372217e-01,  3.10188915e-01,
         -4.61279378e-01],
        [ 1.21201120e+00,  4.07322885e-01, -5.86148273e-01,
         -4.63873247e-01],
        [-6.98547968e-01,  3.08796810e-01,  1.49294935e+00,
          1.16783653e+00],
        [ 9.53806524e-01, -1.34003033e+00,  1.11328189e+00,
          8.49596267e-01]])}, 'telemetry_vec': array([0.22, 0.22, 2.15, 0.22])}}                                                                                                               ]
```

### Turn 2 — `execute_in_remote_docker` (`11831 ms` LLM)
- **Rationale**: Inspect the archive paths and compact data schema without dumping sample arrays; the frozen sink confirms the two required output keys.

```bash
cd /holdout_task && find /tmp/hegel_continual_archive -maxdepth 5 -type f | sort | head -100; sed -n '1,230p' /holdout_task/test_dag_candidate.py; cd /holdout_task; python3 - <<'PY'
import numpy as np,pickle
x=pickle.load(open('public_batch.pkl','rb'))
def show(x,p='root'):
 if isinstance(x,dict):
  for k,v in x.items(): show(v,p+'/'+str(k))
 elif isinstance(x,(list,tuple)) and len(x) and isinstance(x[0],dict):
  print(p,len(x));show(x[0],p+'/0')
 else:
  a=np.asarray(x); print(p,a.shape,a.dtype,repr(a.ravel()[:7]) if a.dtype!=object else str(type(x)))
show(x)
PY
```
- **Container Exit Code**: `0` (`441 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_ = run_dag(st_cv)
    ym = float(np.mean(np.abs(np.asarray(out_cv["y_hat"], float) - vy)))
    tm = float(np.mean(np.abs(np.asarray(out_cv["state_pred"], float) - vtr)))
    cv_metrics = {"public_cv_y_mae": round(ym, 6), "public_cv_traj_mae": round(tm, 6)}
    if ym > 0.038: errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
    if tm > 0.038: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")

ok = len(errs) == 0 and all(v == 1 for v in ec.values())
print(json.dumps({"accepted": ok, "errors": errs, "cv_metrics": cv_metrics, "exec_counts": ec}, allow_nan=False))
sys.exit(0 if ok else 1)
root 1
root/0/study/eval_times (8,) float64 array([0.25      , 0.64285714, 1.03571429, 1.42857143, 1.82142857,
       2.21428571, 2.60714286])
root/0/study/panel_a/u (48, 4) float64 array([ 0.39029514, -0.95170938, -0.4811785 , -0.02139124, -0.17650367,
       -0.54608243, -0.43432813])
root/0/study/panel_a/v (48, 4) float64 array([ 0.80476209, -1.1926999 , -1.06290681,  0.23508629, -0.013812  ,
       -1.01020796, -0.60862339])
root/0/study/panel_b/u (24, 4) float64 array([ 0.9729235 , -0.08929282,  1.11416849, -0.48150752, -0.53147336,
        0.47787631, -1.23689286])
root/0/study/panel_b/v (24, 4) float64 array([-0.82935563, -0.60905359,  0.65146427,  0.70843547,  1.27687552,
        0.75811009,  0.07864918])
root/0/study/panel_c/u (24, 4) float64 array([-1.09462739, -0.73570546, -0.03304023, -0.06170359,  1.19791032,
        0.18671738, -0.64389609])
root/0/study/panel_c/v1 (24, 4) float64 array([ 0.27816325,  0.89445364,  0.55767016, -0.0117454 , -1.13534014,
       -0.52385814, -0.46190537])
root/0/study/panel_c/v2 (24, 4) float64 array([ 0.25764376,  0.04939004, -0.32480842, -0.07386705,  0.20999858,
       -0.36279167,  0.87760554])
root/0/study/train/x (72, 4) float64 array([-1.01442125,  0.6283896 ,  0.94979655, -1.64501329,  0.17590512,
       -0.49675385,  0.95431964])
root/0/study/train/y (72,) float64 array([ 0.00718677,  1.35816327,  0.07499529, -1.14833881,  1.2392626 ,
        0.86471047, -0.36994889])
root/0/study/train/traj (72, 8) float64 array([-0.71774876, -1.39336079, -1.76618577, -1.96977643, -2.10745259,
       -2.20467682, -2.25898867])
root/0/study/train/summary (4,) float64 array([ 0.10141371,  0.76797625, -0.25379409,  1.30898633])
root/0/study/test/x (36, 4) float64 array([ 0.70813585,  0.94243038,  0.72384048, -1.60077778,  1.52631014,
        1.05988666, -0.86296146])
root/0/study/telemetry_vec (4,) float64 array([0.22, 0.22, 2.15, 0.22])
```

### Turn 3 — `execute_in_remote_docker` (`9427 ms` LLM)
- **Rationale**: Read the archived solver before fitting a replacement so the public data can test which calibration and readout terms still transfer.

```bash
wc -l /tmp/hegel_continual_archive/*; sed -n '1,250p' /tmp/hegel_continual_archive/conf_m102_b03_batch_00_solver.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel alignment calibration' 2>&1 | head -20
```
- **Container Exit Code**: `0` (`320 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
o-mode trajectory."""
import numpy as np
from scipy.optimize import least_squares


def _ones(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _align(panel):
    u, v = panel['u'], panel['v']
    # Median of pairwise slopes, followed by inlier least squares.
    i, j = np.triu_indices(len(u), 1)
    dx, dy = u[i] - u[j], v[i] - v[j]
    result = []
    for k in range(u.shape[1]):
        valid = abs(dx[:, k]) > 1e-8
        slope = np.median(dy[valid, k] / dx[valid, k])
        r = v[:, k] - u[:, k] * slope
        r -= np.median(r)
        good = abs(r) <= max(.001, 5 * np.median(abs(r)))
        result.append(_ls(_ones(u[good, k:k+1]), v[good, k]))
    return np.asarray(result).T


def _branch(panel, x, k):
    u, v = panel['u'], panel['v'+str(k)]
    if k == 1:
        def features(z, p):
            w = p[0]*z + p[1:]
            return _ones(np.c_[z, w/(1+abs(w))])
        start = [.84, 0., 0., 0., 0.]
    else:
        def features(z, p):
            return _ones(np.c_[z, np.cos(p[0]*z+p[1:])])
        start = [1.27, 0., 0., 0., 0.]
    def residual(p):
        d = features(u, p)
        return (d @ _ls(d, v) - v).ravel()
    p = least_squares(residual, start, max_nfev=150).x
    return features(x, p) @ _ls(features(u, p), v)


def _predict(study, rows, xnew):
    train = study['train']
    xold = train['x'][rows]
    n = len(xold)
    a, b = _align(study['panel_a'])
    z = (np.vstack((xold, xnew)) - a) / b
    c, d = (_branch(study['panel_c'], z, k) for k in (1,2))
    dot = np.sum(c*d, axis=1)
    sy = _ones(np.c_[d, c, dot])
    syhat = sy[n:] @ _ls(sy[:n], train['y'][rows])
    ts = np.asarray(study['eval_times'])
    trajectories = train['traj'][rows]
    def basis(p):
        return -np.expm1(-ts[:, None]*p)
    def time_residual(p):
        h = basis(p)
        return (h @ _ls(h, trajectories.T) - trajectories.T).ravel()
    h = basis(least_squares(time_residual, [1.8,.4], bounds=([1.,.1],[3.,.95])).x)
    cross = (d[:,:,None]*c[:,None,:]).reshape(len(z), -1)
    st = _ones(np.c_[d,c,cross])
    return syhat, st[n:] @ _ls(st[:n], trajectories) @ np.linalg.pinv(h.T) @ h.T


def solve_batch(study: dict) -> dict:
    y, state = _predict(study, slice(None), study['test']['x'])
    terminal = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}
No matching skill cards found.
```

### Turn 4 — `execute_in_remote_docker` (`29804 ms` LLM)
- **Rationale**: Measure panel fit and compare archived, quadratic, and cross-branch readouts on the same held-out rows. This separates calibration error from missing readout terms.
- **Alternatives Weighed**: `Replace both readouts with a generic regressor` (rejected: First test which calibrated features transfer and whether the scalar residual calls for a specific structural term.)

```bash
cat /tmp/hegel_continual_archive/archive_manifest.json /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json; cat >/tmp/explore_cal.py <<'PY'
import numpy as np,pickle,runpy
from scipy.optimize import least_squares
from itertools import combinations
A=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_b03_batch_00_solver.py'); align,branch,ones,ls=(A[k] for k in ('_align','_branch','_ones','_ls'))
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,t=(s['train'][k] for k in ('x','y','traj')); a,b=align(s['panel_a']); z=(x-a)/b;c,d=(branch(s['panel_c'],z,k) for k in (1,2)); dot=(c*d).sum(1); cd=(c[:,:,None]*d[:,None,:]).reshape(len(x),-1)
np.set_printoptions(precision=4,suppress=True)
print('align',a,b)
for k in (1,2):
 u=s['panel_c']['u'];r=branch(s['panel_c'],u,k)-s['panel_c']['v'+str(k)];print('branch',k,np.sqrt(np.mean(r*r,axis=0)))
u,v=(s['panel_b'][k] for k in ('u','v'));print('b affine matrix and residual',ls(ones(u),v),np.std(ones(u)@ls(ones(u),v)-v,axis=0));print('b corr',np.corrcoef(u.T,v.T)[:4,4:]);
base=np.c_[d,c]; baseline=ones(np.c_[base,dot]);print('baseline scalar coefficients',ls(baseline,y));
for nam,D in {'z':z,'cd base':base,'archived':np.c_[base,dot],'full cross':np.c_[base,cd],'matching cross':np.c_[base,c*d],'squares':np.c_[base,dot,c*c,d*d,z*z],'z quadratic':np.c_[z,*[z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]],'branch paired squares':np.c_[base,*[w[:,i]*w[:,j] for w in (c,d) for i in range(4) for j in range(i,4)],cd]}.items():
 D=ones(D); e=D[52:]@ls(D[:52],np.c_[y,t][:52])-np.c_[y,t][52:];f=D@ls(D,np.c_[y,t])-np.c_[y,t]; print(nam,'val',np.mean(abs(e[:,0])),np.mean(abs(e[:,1:])), 'train',np.mean(abs(f[:,0])),np.mean(abs(f[:,1:])))
res=y-baseline@ls(baseline,y);print('largest residuals indices, y, r, z, dot'); ii=np.argsort(abs(res))[-12:];print(np.c_[ii,y[ii],res[ii],z[ii],dot[ii]])
PY
python3 /tmp/explore_cal.py
```
- **Container Exit Code**: `0` (`437 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
,
    "shared_fanout_nodes": [],
    "topo_order": [
      "node_00",
      "node_01",
      "node_02"
    ]
  },
  "kernel_version": "hegel_shared_math_handoff_kernel_v3_m961_three_layer",
  "wall_ms": 2505.42
}align [-0.0296 -0.1537 -0.1121  0.0333] [1.3058 1.1254 1.3793 1.3177]
branch 1 [0.0005 0.0007 0.0006 0.0006]
branch 2 [0.0003 0.0003 0.0003 0.0003]
b affine matrix and residual [[ 0.12   -0.1069  0.0777  0.0779]
 [-0.5167  0.4231  0.4734  0.3231]
 [ 0.1123 -0.1083  0.6271 -0.5973]
 [-0.0904 -0.7264  0.2751  0.4036]
 [ 0.6976  0.2366  0.2853  0.3878]] [0.0001 0.0001 0.0001 0.0001]
b corr [[-0.6652  0.1207  0.6497  0.422 ]
 [ 0.2919  0.2731  0.75   -0.7102]
 [-0.6363 -0.8635  0.1553  0.6525]
 [ 0.8536  0.6056  0.2515 -0.1077]]
baseline scalar coefficients [-0.0097 -0.418   0.2845 -0.3797 -0.5608 -0.5309 -0.6578  0.9014 -0.6782
  0.2237]
z val 0.21460161091616206 0.16097758743424767 train 0.24862015846055463 0.15294975486869478
cd base val 0.15762572228250224 0.16175280621721308 train 0.1266052918980198 0.11583725162413029
archived val 0.13758219148581197 0.14069548120517802 train 0.10803823836988714 0.09687305519571396
full cross val 0.1270266891663656 0.08773126420941794 train 0.08267833584501226 0.05953256219486226
matching cross val 0.132303290608276 0.14480985349410544 train 0.10490983666377872 0.09715342920131492
squares val 0.11532955871913915 0.09811394615666766 train 0.08348962333217337 0.055981932180362404
z quadratic val 0.1626338110861632 0.15395033592105728 train 0.1577916461502169 0.09857105615975101
branch paired squares val 0.1609433896031324 0.1698099793130341 train 0.04611264155656988 0.03211462954637308
largest residuals indices, y, r, z, dot
[[25.      1.2059  0.1904  0.1285 -0.3345  0.1707 -0.187   0.0054]
 [70.     -1.0053 -0.1915 -1.1315  0.3323  0.7829  0.3737  0.0319]
 [ 4.      1.2393  0.2047  0.0431 -0.4322  0.2678 -0.1712  0.0007]
 [ 1.      1.3582  0.2098  0.1574 -0.3048  0.7732  0.1078  0.0273]
 [20.      0.0151 -0.2196  0.5216  0.3915  0.269   0.2585 -0.0358]
 [11.     -1.5537  0.2285 -0.8792  0.7037 -0.8716  0.032   0.3249]
 [57.      1.351   0.2331  0.0017 -0.2311  0.3522 -0.457  -0.0154]
 [ 3.     -1.1483 -0.2337 -0.703   0.0148  0.1297  0.4282  0.0635]
 [32.     -1.1396  0.2518 -0.801  -1.0687 -0.8403  0.7479 -0.1297]
 [58.      0.7299  0.2706 -0.32    0.3984  0.7601 -0.9721 -0.507 ]
 [29.      0.9361 -0.3466 -1.0949 -0.991   0.9818 -1.0446 -0.0489]
 [69.     -0.1616 -0.3731 -0.3237 -0.7793 -0.5792 -0.773  -0.3763]]
```

### Turn 5 — `execute_in_remote_docker` (`27265 ms` LLM)
- **Rationale**: Panel B exhibits an accurately measured non-diagonal affine map. Testing its placement and direction can distinguish a coordinate mismatch from a missing readout interaction.
- **Alternatives Weighed**: `Treat this as a scalar-only change` (rejected: The trajectory also fails when the archive calibration is applied directly.)
- **ATMS NOGOOD DECLARED**: Refuted `The archived panel-A-only feature grammar transfers unchanged.` | Witness: `Archived scalar CV MAE is 0.13758 and archived trajectory CV MAE is 0.08773, both above 0.038 despite sub-0.001 in-panel branch RMSE.` | Blanket Diagnosis: `Both readouts fail, so missing shared coordinate structure must be tested alongside scalar-specific terms.` | Sublated Pivot: `Retain the calibrated branch forms and test directional and affine applications of the measured panel B map before altering either readout.`

```bash
cat >/tmp/panel_search.py <<'PY'
exec(open('/tmp/explore_cal.py').read().split('np.set_printoptions')[0])
M=ls(ones(s['panel_b']['u']),s['panel_b']['v']); B=M[1:]; h=M[0]; P=np.linalg.inv(B)
print('singular',np.linalg.svd(B)[1]); print('matrix',M)
from scipy.optimize import least_squares
pan=s['panel_c']; u=pan['u']; funcs=[]
for k in (1,2):
 v=pan['v'+str(k)]
 def ft(z,p,k=k):
  w=p[0]*z+p[1:]; return ones(np.c_[z,w/(1+abs(w)) if k==1 else np.cos(w)])
 def fr(p):
  F=ft(u,p);return (F@ls(F,v)-v).ravel()
 q=least_squares(fr,[.84 if k==1 else 1.27,0,0,0,0],max_nfev=150).x; co=ls(ft(u,q),v); funcs.append(lambda z,ft=ft,q=q,co=co: ft(z,q)@co);print('branch',k,q,co)
ops={'I':lambda v:v,'f':lambda v:v@B+h,'r':lambda v:(v-h)@P,'ft':lambda v:v@B.T+h,'rt':lambda v:(v-h)@P.T,'fl':lambda v:v@B,'rl':lambda v:v@P,'flt':lambda v:v@B.T,'rlt':lambda v:v@P.T,'x':lambda v:v*b+a}
C={k:funcs[0](f(z)) for k,f in ops.items()};D={k:funcs[1](f(z)) for k,f in ops.items()}
def eval(F,V):
 r=F[52:]@ls(F[:52],V[:52])-V[52:];return np.mean(abs(r))
ys=[];ts=[]
for i,c in C.items():
 for j,d in D.items():
  dp=(c*d).sum(1); cr=np.einsum('ni,nj->nij',c,d).reshape(len(z),-1); l=np.c_[c,d]
  for name,f in {'base':l,'dot':np.c_[l,dp], 'dot2':np.c_[l,dp,dp**2], 'dot_r':np.c_[l,dp/(1+2.15*dp**2)], 'dot_tanh':np.c_[l,np.tanh(2.15*dp)], 'dot_sin':np.c_[l,np.sin(2.15*dp)], 'cross':np.c_[l,cr], 'quad':np.c_[l,cr,*[w[:,a]*w[:,b] for w in (c,d) for a in range(4) for b in range(a,4)]]}.items():
   F=ones(f);ys.append((eval(F,y),i,j,name)); ts.append((eval(F,t),i,j,name))
print('scalar',sorted(ys)[:25]);print('trajectory',sorted(ts)[:25]);
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/panel_search.py
```
- **Container Exit Code**: `0` (`619 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
11 -0.27998589 -0.04233483  0.23241974]
 [-0.27787088  0.383247    0.01871688  0.0870644 ]
 [ 0.22380977  0.06415264  0.12544993  0.40463172]
 [ 0.05131128  0.03856438 -0.30797409  0.06081789]
 [ 0.20900249  0.18472884  0.02839186 -0.15337655]
 [ 0.18419478 -0.25535431 -0.01294915 -0.05727531]
 [-0.14845932 -0.04234747 -0.08286189 -0.26801907]]
scalar [(0.11326401792264391, 'x', 'r', 'cross'), (0.11415114505908583, 'I', 'rl', 'cross'), (0.11574851171094522, 'I', 'r', 'cross'), (0.11623165128959445, 'x', 'rl', 'cross'), (0.11707784527297413, 'I', 'ft', 'cross'), (0.11789280720902003, 'I', 'flt', 'cross'), (0.1250035403364267, 'I', 'f', 'cross'), (0.12527785728085655, 'I', 'fl', 'cross'), (0.1270266891663655, 'I', 'I', 'cross'), (0.1288670340853291, 'x', 'flt', 'cross'), (0.129361987349917, 'x', 'ft', 'cross'), (0.12985451467476544, 'I', 'rt', 'cross'), (0.13236528717054652, 'x', 'rt', 'cross'), (0.13247631927745646, 'I', 'x', 'cross'), (0.1343012113564514, 'I', 'rlt', 'cross'), (0.1351511182635863, 'I', 'x', 'dot'), (0.1352224069458132, 'I', 'x', 'dot2'), (0.13528940707846127, 'x', 'r', 'dot_tanh'), (0.13564204459515591, 'x', 'rlt', 'cross'), (0.13583760299406616, 'x', 'r', 'dot'), (0.1369753920006977, 'x', 'r', 'dot2'), (0.13758219148581197, 'I', 'I', 'dot'), (0.1380619447824793, 'I', 'I', 'dot2'), (0.13846326684543958, 'x', 'rl', 'dot_tanh'), (0.13862955772238342, 'x', 'ft', 'dot_sin')]
trajectory [(0.06097705527520164, 'x', 'x', 'cross'), (0.0671204891221682, 'I', 'x', 'cross'), (0.07516338815325876, 'x', 'I', 'cross'), (0.08773126420941804, 'I', 'I', 'cross'), (0.09914095103479062, 'I', 'x', 'quad'), (0.11221821114719721, 'fl', 'I', 'cross'), (0.11258450797391095, 'rl', 'ft', 'cross'), (0.11268838794088501, 'r', 'ft', 'cross'), (0.11288274027426062, 'flt', 'ft', 'cross'), (0.11420137452265741, 'x', 'flt', 'cross'), (0.11429854279331393, 'flt', 'flt', 'cross'), (0.11476603882183081, 'x', 'ft', 'cross'), (0.11501919499963043, 'fl', 'x', 'cross'), (0.11564982683407452, 'rl', 'flt', 'cross'), (0.11570752217486999, 'r', 'flt', 'cross'), (0.11596640852336268, 'f', 'I', 'cross'), (0.1163194360707152, 'f', 'x', 'cross'), (0.1168157808343937, 'r', 'rl', 'cross'), (0.1170254731307021, 'flt', 'rl', 'cross'), (0.11795831453739089, 'rlt', 'x', 'cross'), (0.11803493063947923, 'rlt', 'I', 'cross'), (0.11862060117310924, 'rt', 'I', 'cross'), (0.11934860096433672, 'rl', 'rl', 'cross'), (0.12152788320169297, 'rt', 'x', 'cross'), (0.12303846484574285, 'x', 'f', 'cross')]
```

### Turn 6 — `execute_in_remote_docker` (`48650 ms` LLM)
- **Rationale**: Test both calibration orders and their offsets, and inspect the trajectory rank to separate spatial feature errors from temporal model errors.
- **Alternatives Weighed**: `Adopt a direct panel B insertion` (rejected: The measured affine map is accurate, but none of its direct insertions satisfies either readout tolerance.)

```bash
cat /holdout_task/public_asset_index.json; cat /holdout_task/_frozen_nodes/node_0{1,2}.py; cat >/tmp/search6.py <<'PY'
exec(open('/tmp/panel_search.py').read().split('C={')[0])
print('traj sv',np.linalg.svd(t,compute_uv=False),'y quantiles',np.quantile(y,np.linspace(0,1,11)))
A={'I':lambda w:w,'a':lambda w:(w-a)/b,'A':lambda w:w*b+a}; G={k:f for k,f in ops.items() if k not in ('I','x')}; W={'z':z,'x':x,'aa':A['a'](z)}
for k,f in G.items():
 for name,w in list(W.items())[:3]:
  for j,g in A.items(): W[name+k+j]=g(f(w))
C={k:funcs[0](w) for k,w in W.items()};D={k:funcs[1](w) for k,w in W.items()};
def score(F,target):
 coef=ls(F[:52],target[:52]); return np.mean(abs(F[52:]@coef-target[52:])), np.mean(abs(F[:52]@coef-target[:52]))
r=[];q=[]
for i,c in C.items():
 for j,d in D.items():
  cross=(c[:,:,None]*d[:,None,:]).reshape(len(c),-1);base=ones(np.c_[c,d]); dot=(c*d).sum(1)
  for nm,F in (('cross',np.c_[base,cross]),('dot',np.c_[base,dot]),('plain',base),('sparsequad',np.c_[base,c*d,c*c,d*d])):
   if nm in ('cross','sparsequad'):r.append((*score(F,t),i,j,nm))
   if nm in ('dot','plain'):q.append((*score(F,y),i,j,nm))
print('trajectory',sorted(r)[:25]);print('traj train',sorted(r,key=lambda e:e[1])[:12]);print('scalar',sorted(q)[:25]);
for nm,w in {'z':z,'x':x,'Bz':z@B+h}.items():
 c,d=funcs[0](w),funcs[1](w); dot=(c*d).sum(1);d1=np.c_[c,d]; dp=ones(np.c_[d1,dot]); rr=y-dp@ls(dp,y); print(nm,'res corrs',np.corrcoef(np.c_[rr,w**3, np.tanh(w*2.15), dot**2, dot**3].T)[0]);
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/search6.py
```
- **Container Exit Code**: `0` (`2175 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, (0.07667700489825552, 0.03864320179969692, 'z', 'aaflta', 'cross'), (0.08707564370710189, 0.03915002046640207, 'x', 'aarla', 'cross'), (0.09463965580191598, 0.03916372950309443, 'z', 'aaftI', 'cross'), (0.08598353684685289, 0.039165435886818194, 'x', 'aafltI', 'cross'), (0.09415766888902029, 0.03922976380730833, 'x', 'zflta', 'cross')]
scalar [(0.1168945761057012, 0.1646414202044146, 'aafa', 'zrlI', 'dot'), (0.11850161208532188, 0.17631059088989634, 'zfa', 'zrlI', 'dot'), (0.11896399200228425, 0.17214952184039398, 'aafa', 'xrlI', 'dot'), (0.11920023180596417, 0.15971030006927997, 'aafa', 'zrla', 'dot'), (0.11965112564996634, 0.17735023501351316, 'aafa', 'aarlA', 'dot'), (0.11987786153762739, 0.16429875926921086, 'aafa', 'aarlI', 'dot'), (0.12004923688050058, 0.16066923854879728, 'aafa', 'zra', 'dot'), (0.1200567402731616, 0.16706520473005504, 'aafa', 'zrI', 'dot'), (0.12061775647367809, 0.1608988611588082, 'aafa', 'xrla', 'dot'), (0.12184404788957467, 0.17001171559827616, 'aafa', 'zfltA', 'dot'), (0.12186320069081287, 0.16110528654226705, 'aafa', 'zfltI', 'dot'), (0.12213167091889518, 0.17925697772487906, 'zfa', 'zrI', 'dot'), (0.12237895170790218, 0.17466833561878006, 'zfa', 'xrla', 'dot'), (0.12245701083086888, 0.17461789151236878, 'zfa', 'zrla', 'dot'), (0.12249873849006421, 0.16158965504912245, 'aafa', 'xra', 'dot'), (0.12284566095394911, 0.1736862139859242, 'aafa', 'xrI', 'dot'), (0.12333302544987106, 0.18389200043330073, 'zfa', 'xrlI', 'dot'), (0.12340187258611013, 0.1668351099612822, 'aafa', 'aarI', 'dot'), (0.12343293082104256, 0.18294596406298774, 'aafa', 'zrlA', 'dot'), (0.12370042678918389, 0.17572615728666333, 'zfa', 'xra', 'dot'), (0.12391132497911098, 0.16753678864904684, 'aafa', 'aafltA', 'dot'), (0.12398564513299815, 0.16276479450463485, 'aafa', 'xfltI', 'dot'), (0.12419466428657364, 0.16318060229900502, 'aafa', 'aarla', 'dot'), (0.12479258904925594, 0.17939381387485664, 'zfa', 'zfltA', 'dot'), (0.12521454391401507, 0.17092830010030163, 'aafa', 'zftA', 'dot')]
z res corrs [ 1.          0.1410185   0.04163359 -0.07595567  0.11793902 -0.01701001
 -0.0064638   0.01101731 -0.01617549  0.02383882  0.06175037]
x res corrs [ 1.00000000e+00  2.23448099e-01 -4.64521673e-02 -1.45362108e-03
  5.07746535e-02 -2.50169730e-02  5.74277375e-03 -4.50596950e-04
 -1.00921223e-02  3.50282514e-02  3.03145131e-02]
Bz res corrs [ 1.         -0.03215405  0.1769092   0.19776799  0.05866249  0.00800109
 -0.01382889 -0.01393724 -0.00574396  0.23470961 -0.00980645]
```

### Turn 7 — `execute_in_remote_docker` (`40879 ms` LLM)
- **Rationale**: Compare a compact nonlinear scalar readout with smooth spatial residual models across four held-out folds. Read the validator alongside these experiments to keep the exact output contract in view.
- **Alternatives Weighed**: `Select a calibration composition from the previous split` (rejected: No composition reached tolerance, and the best scalar and trajectory scores came from different choices.)

```bash
sed -n '90,230p' /holdout_task/test_dag_candidate.py; cat /holdout_task/_frozen_nodes/node_02.py; cat >/tmp/nonlin7.py <<'PY'
exec(open('/tmp/panel_search.py').read().split('C={')[0])
from scipy.spatial.distance import cdist
from scipy.linalg import qr
from scipy.special import expit
from sklearn.gaussian_process.kernels import Matern
print('sv',np.linalg.svd(t,compute_uv=False)); print('ranges',[(k,np.min(s[k]['u'],0),np.max(s[k]['u'],0)) for k in ('panel_a','panel_b','panel_c')]); print('summary',s['train']['summary'])
X=np.c_[y,t]; rng=np.random.default_rng(3128); folds=np.array_split(rng.permutation(len(y)),4)
def cv(K,F=None):
 err=[]
 for j in folds:
  i=np.setdiff1d(np.arange(len(y)),j); L=K[np.ix_(i,i)]; R=K[np.ix_(j,i)]
  if F is None: pred=R@np.linalg.solve(L,X[i])
  else:
   f=F[i]; V=np.linalg.solve(L,np.c_[X[i],f]); p=ls(np.c_[f.T@V[:,X.shape[1]:]],f.T@V[:,:X.shape[1]]);pred=R@(V[:,:X.shape[1]]-V[:,X.shape[1]:]@p)+F[j]@p
  err.append(abs(pred-X[j]))
 E=np.concatenate(err);return E[:,0].mean(),E[:,1:].mean()
res=[]
for wname,w in [('z',z),('x',x),('f',z@B+h),('r',(z-h)@P)]:
 c,d=(f(w) for f in funcs); dp=(c*d).sum(1); base=ones(np.c_[c,d,dp]); cross=(c[:,:,None]*d[:,None,:]).reshape(len(w),-1)
 for nm,U in [('z',w),('cd',np.c_[c,d]),('zcd',np.c_[w,c,d]),('weightedcd',np.c_[c,0.5*d])]:
  U=U/U.std(0); dist=cdist(U,U)
  for l in [.75,1.5,3,6,12,24]:
   v=dist/l
   for typ,K in [('rbf',np.exp(-.5*v*v)),('matern',(1+v+v*v/3)*np.exp(-v))]:
    for ridge in [1e-7,1e-4,.003,.03]:
     for fn,F in [('simple',ones(w)),('branch',base),('cross',np.c_[base,cross])]:
      e=cv(K+np.eye(len(y))*ridge,F);res.append((*e,wname,nm,l,typ,ridge,fn))
print('GP y',sorted(res)[:9]);print('GP t',sorted(res,key=lambda x:x[1])[:9]);
# Single nonlinear index: jointly fitted direction, partially linear readout.
for wname,w in [('z',z),('x',x),('f',z@B+h),('r',(z-h)@P)]:
 c,d=(f(w) for f in funcs); dot=(c*d).sum(1); base=ones(np.c_[c,d,dot]);
 for name,g in [('tanh',np.tanh),('sin',np.sin),('soft',lambda u:u/(1+abs(u))),('gauss',lambda u:np.exp(-u*u)),('rat',lambda u:u/(1+u*u)),('relu',lambda u:np.logaddexp(0,u))]:
  for mod in [False,True]:
   vals=[]
   for j in folds:
    i=np.setdiff1d(np.arange(len(y)),j)
    def fd(p):
     v=g(w@p[:4]+p[4]);return np.c_[base,v[:,None]*np.c_[np.ones(len(y)),w]] if mod else np.c_[base,v]
    def loss(p):
     f=fd(p)[i];return f@ls(f,y[i])-y[i]
    ps=[least_squares(loss,p,max_nfev=60) for p in np.r_[np.eye(5)*1.5,rng.normal(0,1,(5,5))]]; p=min(ps,key=lambda q:q.cost).x; f=fd(p); vals.append([np.mean(abs(f[j]@ls(f[i],y[i])-y[j])),np.sqrt(np.mean(loss(p)**2))]);
   print(wname,name,mod,np.mean(vals,0))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/nonlin7.py
```
- **Container Exit Code**: `0` (`37960 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.05591319194116261, 'x', 'zcd', 6, 'matern', 0.0001, 'branch'), (0.10528122656291862, 0.055928559723195215, 'x', 'zcd', 6, 'matern', 0.0001, 'simple'), (0.1161323183096198, 0.057579713045273144, 'z', 'zcd', 24, 'matern', 1e-07, 'branch'), (0.11704024989663357, 0.05758930409152743, 'x', 'zcd', 24, 'matern', 1e-07, 'branch'), (0.11778208610006821, 0.05815160985909795, 'x', 'zcd', 24, 'matern', 1e-07, 'simple'), (0.11675693496335983, 0.05883288131868333, 'z', 'zcd', 24, 'matern', 1e-07, 'simple'), (0.10142429122093279, 0.05920311264101435, 'z', 'zcd', 3, 'matern', 1e-07, 'simple'), (0.12152982953154723, 0.05981506936684037, 'x', 'zcd', 6, 'rbf', 0.0001, 'branch'), (0.11346101457066039, 0.06002792543350972, 'z', 'zcd', 6, 'matern', 1e-07, 'simple')]
z tanh False [0.14249286 0.08974876]
z tanh True [0.20845004 0.07952733]
z sin False [0.11845594 0.08636669]
z sin True [0.13858625 0.06610392]
z soft False [0.1358621  0.09425743]
z soft True [0.20841852 0.08082791]
z gauss False [0.10947761 0.08581926]
z gauss True [0.16167435 0.07481166]
z rat False [0.1156744  0.08818904]
z rat True [0.14812715 0.07663948]
z relu False [0.14220526 0.10217817]
z relu True [0.13498568 0.0799505 ]
x tanh False [0.16831686 0.09269743]
x tanh True [0.17280678 0.08300306]
x sin False [0.15205119 0.09480637]
x sin True [0.12914366 0.07744558]
x soft False [0.1607455 0.0931436]
x soft True [0.18393258 0.08066366]
x gauss False [0.14194952 0.09444862]
x gauss True [0.16948295 0.07582196]
x rat False [0.16030501 0.09249592]
x rat True [0.14900549 0.08287291]
x relu False [0.15255078 0.10582977]
x relu True [0.14105811 0.0851103 ]
f tanh False [0.20174564 0.15701656]
f tanh True [0.22064913 0.12843961]
f sin False [0.23595624 0.16098125]
f sin True [0.27911337 0.1335137 ]
f soft False [0.17945195 0.15855109]
f soft True [0.2414996  0.13146446]
f gauss False [0.22959305 0.16073939]
f gauss True [0.26167239 0.12969554]
f rat False [0.21427556 0.16634495]
f rat True [0.22434506 0.1293324 ]
f relu False [0.24260129 0.1729837 ]
f relu True [0.30562573 0.12567835]
r tanh False [0.27593805 0.16949847]
r tanh True [0.31811383 0.13801539]
r sin False [0.25919524 0.16531044]
r sin True [0.28452853 0.13703961]
r soft False [0.27973162 0.16456844]
r soft True [0.31675721 0.13945029]
r gauss False [0.28497983 0.16864127]
r gauss True [0.36489951 0.1337503 ]
r rat False [0.27806991 0.16836862]
r rat True [0.33363771 0.13419841]
r relu False [0.27304651 0.17604114]
r relu True [0.30088153 0.14040156]
```

### Turn 8 — `execute_in_remote_docker` (`37901 ms` LLM)
- **Rationale**: The previous search applied nonlinearities to individual interactions or raw-coordinate indices. This test checks whether the nonlinearity instead acts on the full calibrated score or on paired branch features.
- **Alternatives Weighed**: `Use the Gaussian-process residual model` (rejected: Even its best four-fold settings missed both 0.038 thresholds.)

```bash
cat /tmp/hegel_continual_archive/conf_m102_b03_batch_00_solver.py; cat /holdout_task/_frozen_nodes/node_02.py; sed -n '143,190p' /holdout_task/test_dag_candidate.py; cat >/tmp/struct8.py <<'PY'
exec(open('/tmp/panel_search.py').read().split('C={')[0])
from scipy.optimize import least_squares
from itertools import product
rng=np.random.default_rng(45); ix=rng.permutation(len(y)); j=ix[:18]; i=ix[18:]; Y=np.c_[y,t];
print('studies',len(pickle.load(open('/holdout_task/public_batch.pkl','rb'))),'x bounds',x.min(0),x.max(0),'z bounds',z.min(0),z.max(0),'test bounds',s['test']['x'].min(0),s['test']['x'].max(0)); print('sv',np.linalg.svd(t,compute_uv=False));
def err(F):
 r=F[j]@ls(F[i],Y[i])-Y[j]; return abs(r[:,0]).mean(),abs(r[:,1:]).mean()
ops2={'z':z,'x':x,'Bz':z@B+h,'invz':(z-h)@P}; out=[]; nonlinear=[]
for nm,w in ops2.items():
 c,d=[f(w) for f in funcs]; dp=(c*d).sum(1); cr=(c[:,:,None]*d[:,None,:]).reshape(len(c),-1); base=ones(np.c_[c,d,dp]);
 for un,U in [('w',w),('c',c),('d',d),('c+d',c+d),('c-d',c-d),('w+c',w+c),('w+d',w+d),('cd',c*d)]:
  for g,G in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('rat',lambda a:a/(1+a*a)),('evenrat',lambda a:1/(1+a*a)),('soft',lambda a:a/(1+abs(a))),('abs',abs),('exp',lambda a:np.exp(-a*a))]:
   for k in [.5,1.25,2.15,3.5,5]:
    V=G(k*U); out.append((*err(np.c_[base,V]),nm,un,g,k,'plain')); out.append((*err(np.c_[base,cr,V]),nm,un,g,k,'cross'))
 for fn,V in [('branches',np.c_[c,d]),('branch_dot',np.c_[c,d,dp]),('cross',np.c_[c,d,cr]),('raw',w),('combined',np.c_[w,c,d,dp])]:
  F=ones(V); F=F/np.maximum(F.std(0),1e-8) if False else F/np.sqrt(np.mean(F**2,axis=0));
  for g,G in [('tanh',np.tanh),('sin',np.sin),('soft',lambda a:a/(1+abs(a))),('atan',np.arctan),('rat',lambda a:a/(1+a*a)),('asinh',np.arcsinh)]:
   for mode in [0,1,2]:
    # Squashed full score, possibly added to a linear/cross component.
    A0=ones(w) if mode==1 else (base if mode==2 else np.ones((len(w),1))); Ti=t[:,-1];
    for tr,v in [('y',y),('t',Ti)]:
     amp=max(np.std(v),.5); lin=ls(F[i],v[i]);
     def pred(p):
      non=G(F@p[:-1]); return A0@ls(A0[i],v[i]-p[-1]*non[i])+p[-1]*non
     def loss(p):return (pred(p)-v)[i]
     if fn=='cross' and mode>0:continue
     fits=[least_squares(loss,np.r_[lin/a,a],max_nfev=65) for a in [amp,amp*2,amp*4]]; p=min(fits,key=lambda f:f.cost); nonlinear.append((tr,np.mean(abs((pred(p.x)-v)[j])),np.sqrt(np.mean(p.fun**2)),nm,fn,g,mode))
print('fixed y',sorted(out)[:12]); print('fixed t',sorted(out,key=lambda p:p[1])[:12]);
for tr in ('y','t'):
 print('index',tr,sorted([q for q in nonlinear if q[0]==tr],key=lambda p:p[1])[:16]); print('index training',tr,sorted([q for q in nonlinear if q[0]==tr],key=lambda p:p[2])[:6]);
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/struct8.py | tail -25
```
- **Container Exit Code**: `0` (`47372 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
10380252951, 0.1084445288360413, 'z', 'combined', 'asinh', 0), ('y', 0.098222238280207, 0.10291858780642783, 'x', 'raw', 'rat', 2), ('y', 0.09920089064848434, 0.10902157473084936, 'z', 'combined', 'soft', 0), ('y', 0.1021182234015302, 0.08718271122291278, 'z', 'raw', 'sin', 2), ('y', 0.1024599400551644, 0.0735974884336612, 'x', 'cross', 'asinh', 0)]
index training y [('y', 0.08429659621026282, 0.07131062384012068, 'z', 'cross', 'soft', 0), ('y', 0.08693053017442988, 0.07139758474809418, 'z', 'cross', 'atan', 0), ('y', 0.08306145687680369, 0.07151480226167854, 'z', 'cross', 'asinh', 0), ('y', 0.11989334801452196, 0.07168232991508738, 'x', 'combined', 'asinh', 2), ('y', 0.13086068450790814, 0.0717499259015174, 'x', 'combined', 'atan', 2), ('y', 0.09021930053239881, 0.07214401932607213, 'z', 'cross', 'tanh', 0)]
index t [('t', 0.08153289888348116, 0.11076899817765222, 'z', 'raw', 'sin', 2), ('t', 0.08356824637427657, 0.1105058863136976, 'z', 'raw', 'rat', 2), ('t', 0.08367770575119726, 0.11107055400565785, 'z', 'raw', 'tanh', 2), ('t', 0.08448467637296388, 0.11266924122384768, 'z', 'raw', 'atan', 2), ('t', 0.0892693141608188, 0.11460721850052086, 'z', 'raw', 'asinh', 2), ('t', 0.09153306998523492, 0.08020691343770608, 'x', 'combined', 'rat', 2), ('t', 0.09202214373857694, 0.07910789906212162, 'x', 'combined', 'atan', 2), ('t', 0.10457469301854108, 0.0997010110348199, 'z', 'combined', 'rat', 1), ('t', 0.10521165541588226, 0.09286562193161937, 'x', 'combined', 'rat', 1), ('t', 0.10725252715610012, 0.1638345628971473, 'z', 'branch_dot', 'soft', 0), ('t', 0.10756318444610899, 0.16454514985667756, 'z', 'branch_dot', 'sin', 0), ('t', 0.10771913202938682, 0.1230419721943349, 'z', 'branches', 'soft', 1), ('t', 0.10780082370881294, 0.16453208296624947, 'z', 'branch_dot', 'rat', 0), ('t', 0.10787830000824844, 0.16452772861155812, 'z', 'branch_dot', 'tanh', 0), ('t', 0.10812136764646524, 0.1645132135012985, 'z', 'branch_dot', 'atan', 0), ('t', 0.10839749203936076, 0.12437903789523805, 'z', 'branches', 'asinh', 1)]
index training t [('t', 0.11919786489418661, 0.06892410098784171, 'x', 'cross', 'asinh', 0), ('t', 0.11898526966483328, 0.06898671564012049, 'x', 'cross', 'atan', 0), ('t', 0.1186811604552174, 0.06909863263744571, 'x', 'cross', 'sin', 0), ('t', 0.11813597608138493, 0.06926309192919713, 'x', 'cross', 'tanh', 0), ('t', 0.11447024909507493, 0.07062451105505575, 'x', 'cross', 'rat', 0), ('t', 0.20009237952359413, 0.07111954737983857, 'x', 'combined', 'asinh', 2)]
```

### Turn 9 — `execute_in_remote_docker` (`47128 ms` LLM)
- **Rationale**: Save an executable baseline, check the frozen output contract, and test whether telemetry-sensitive gating modifies branch features before the scalar and trajectory readouts.
- **Alternatives Weighed**: `Gate the final scalar prediction` (rejected: The fitted single-index saturation models did not beat the smooth residual model.)

```bash
set -e; cat /holdout_task/_frozen_nodes/node_02.py; sed -n '150,230p' /holdout_task/test_dag_candidate.py; sed '/^def _predict/,$d' /tmp/hegel_continual_archive/conf_m102_b03_batch_00_solver.py >/holdout_task/node_00.py; cat >>/holdout_task/node_00.py <<'PY'


def _kernel(u):
    delta = u[:, None] - u[None, :]
    return np.exp(-np.sum(delta**2, axis=2) / 4.5)


def _fit_kernel(u, f, target, n):
    K = _kernel(u)
    L = K[:n, :n] + .003*np.eye(n)
    F = f[:n]
    v = np.linalg.solve(L, np.column_stack((target, F)))
    beta = _ls(F.T @ v[:, 1:], F.T @ v[:, 0])
    return f[n:] @ beta + K[n:, :n] @ (v[:, 0] - v[:, 1:] @ beta)


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']
    train, test = s['train'], s['test']
    n = len(train['x'])
    x = np.vstack((train['x'], test['x']))
    a, b = _align(s['panel_a'])
    z = (x-a)/b
    c, d = (_branch(s['panel_c'], z, k) for k in (1, 2))
    yz = np.c_[z,c,d]
    yz /= np.maximum(yz[:n].std(0), 1e-8)
    f = _ones(np.c_[c,d,np.sum(c*d,axis=1)])
    y = _fit_kernel(yz, f, train['y'], n)
    c, d = (_branch(s['panel_c'], x, k) for k in (1, 2))
    cross = (d[:,:,None]*c[:,None,:]).reshape(len(x), -1)
    f = _ones(np.c_[d,c,cross])
    state = f[n:] @ _ls(f[:n],train['traj'])
    return {'y_hat': y, 'TRAJECTORY_KEY': state}
PY
python3 - <<'PY'
import ast
p='/holdout_task/_frozen_nodes/node_02.py'; t=ast.parse(open(p).read());print([(ast.unparse(n.value),n.slice.value) for n in ast.walk(t) if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant)])
PY
key=$(python3 - <<'PY'
import re
s=open('/holdout_task/_frozen_nodes/node_02.py').read(); m=re.findall(r'''([A-Za-z_][A-Za-z_0-9]*)\[['"]([^'"]+)['"]\]''',s); a=next(v for v,k in m if k=='y_hat'); print(next(k for v,k in m if v==a and k!='y_hat'))
PY
); sed -i "s/TRAJECTORY_KEY/$key/" /holdout_task/node_00.py; OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py || true
cat >/tmp/mix9.py <<'PY'
exec(open('/tmp/panel_search.py').read().split('C={')[0])
from itertools import combinations
from scipy.special import expit
from scipy.spatial.distance import cdist
np.set_printoptions(precision=5,suppress=True)
print('align',a,b,'sv',np.linalg.svd(t,compute_uv=False),'n',len(y),'times',s['eval_times']);print('summary matches',np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()]))
Y=np.c_[y,t]; i=np.arange(52);j=np.arange(52,72); print('ranges',z.min(0),z.max(0))
R=[]
for ws,w in [('z',z),('x',x),('b',z@B+h),('r',(z-h)@P)]:
 c,d=(f(w) for f in funcs); cr=(c[:,:,None]*d[:,None,:]).reshape(len(w),-1); base=ones(np.c_[c,d,(c*d).sum(1)]); trend=ones(np.c_[c,d,cr]);
 def eval(F):
  coef=ls(F[i],Y[i]); E=abs(F@coef-Y); return E[j,0].mean(), E[j,1:].mean(), E[i,0].mean(), E[i,1:].mean()
 for k in range(4):
  if ws=='z':
   E=Y-trend@ls(trend,Y); ii=np.argsort(w[:,k]);print('bins',k,[np.round(E[b][:,[0,-1]].mean(0),3) for b in np.array_split(ii,8)])
 projections=[(str(k),w[:,k]) for k in range(4)];
 for k,m in combinations(range(4),2):
  for sign in [-1,1]:projections.append((f'{k}{sign:+d}{m}',w[:,k]+sign*w[:,m])); projections.append((f'{k}*{m}',w[:,k]*w[:,m]));
 projections += [('dot',(c*d).sum(1)), ('sum',w.sum(1)), ('norm',(w*w).sum(1)),('cnorm',(c*c).sum(1)),('dnorm',(d*d).sum(1))];
 for pn,p in projections:
  for shift in [0.,np.median(p),np.quantile(p,.3),np.quantile(p,.7)]:
   h=(p-shift)/max(p.std(),.0001)
   for gn,g in [('hard',h>0),('logit',expit(2.15*h)),('fast',expit(8*h)),('rat',1/(1+2.15*h*h)),('soft',h/(1+2.15*abs(h)))]:
    for dn,D in [('only',np.ones((len(w),1))),('z',ones(w)),('c',ones(c)),('d',ones(d)),('cd',ones(np.c_[c,d]))]:
     for bn,b in [('base',base),('trend',trend)]:R.append((*eval(np.c_[b,g[:,None]*D]),ws,pn,round(shift,3),gn,dn,bn))
print('y',sorted(R)[:10]);print('t',sorted(R,key=lambda v:v[1])[:10]);print('train y',sorted(R,key=lambda v:v[2])[:5]);print('train t',sorted(R,key=lambda v:v[3])[:5]);
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/mix9.py | tail -14
```
- **Container Exit Code**: `0` (`4512 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
8808, 'x', '1-12', -0.875, 'fast', 'only', 'trend'), (0.161296147351273, 0.04731150338830894, 0.06564221746478387, 0.04424925786107821, 'z', '1-12', -0.214, 'rat', 'c', 'trend'), (0.1752372102177449, 0.04734220452071241, 0.06400585830670143, 0.042390116579394434, 'x', '1-12', 0.0, 'rat', 'c', 'trend'), (0.15178952772818816, 0.04738698867414154, 0.0758463993653895, 0.05026780324355954, 'x', '1-12', 0.313, 'fast', 'only', 'trend'), (0.19944102266509814, 0.047568337276354705, 0.05967289039252381, 0.03961248589632768, 'x', '1-12', -0.875, 'fast', 'c', 'trend'), (0.15737298934322103, 0.047752754921325406, 0.07542219471628277, 0.05098272221591427, 'x', '1-12', 0.313, 'soft', 'only', 'trend'), (0.17657534343435644, 0.04786963981244313, 0.06656120987214362, 0.04043158859346022, 'x', '1-12', 0.0, 'rat', 'd', 'trend'), (0.1496239373927032, 0.048603509198316686, 0.07646698205143922, 0.04058498133848244, 'x', '1-12', -0.875, 'soft', 'only', 'trend')]
train y [(0.1891420693849292, 0.07448456036083331, 0.028380205377667205, 0.02483786847273084, 'x', '1', 0.0, 'rat', 'cd', 'trend'), (0.16787507646352967, 0.11526927533843882, 0.031039250959662874, 0.02851364480532746, 'z', '0+12', 0.392, 'fast', 'cd', 'trend'), (0.13919929736309494, 0.10452614177863453, 0.031242090575053987, 0.0268024981240105, 'x', '1', -0.449, 'logit', 'cd', 'trend'), (0.1656861084985256, 0.11081947908596726, 0.03128951177608969, 0.027780411997649017, 'z', '0+12', 0.392, 'soft', 'cd', 'trend'), (0.18555992694458384, 0.13562573625631086, 0.03176616321382771, 0.040224393566255144, 'z', '0-13', 0.0, 'hard', 'cd', 'trend')]
train t [(0.1923811295041941, 0.12541746544545518, 0.06418120549386595, 0.02063079971393917, 'x', 'dnorm', 0.802, 'soft', 'cd', 'trend'), (0.11896572498118496, 0.11348630989971722, 0.04775419901997175, 0.021073657073148754, 'x', '1', 0.364, 'rat', 'cd', 'trend'), (0.15429611318726305, 0.07866065057860223, 0.06376610427096477, 0.02228459927368563, 'z', '2', 0.637, 'soft', 'cd', 'trend'), (0.16724779985745414, 0.13757604326619266, 0.06397220860278967, 0.02259964751843141, 'x', '0*1', 0.0, 'logit', 'cd', 'trend'), (0.16724779985745414, 0.13757604326619266, 0.06397220860278967, 0.02259964751843141, 'x', '0*1', 0.0, 'logit', 'cd', 'trend')]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9906762b418dd6f4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9906762b (directory).
```

### Turn 10 — `execute_in_remote_docker` (`38389 ms` LLM)
- **Rationale**: Test whether a shared coordinate correction explains both residuals, and compare it against regularized nonlinear basis expansions without editing the saved baseline.
- **Alternatives Weighed**: `Promote the best single trajectory gate` (rejected: It was selected on the same split and did not improve the scalar readout.)

```bash
cat >/tmp/joint10.py <<'PY'
exec(open('/tmp/panel_search.py').read().split('C={')[0])
from scipy.optimize import least_squares
from scipy.special import expit
from itertools import combinations
np.set_printoptions(precision=4,suppress=True)
U,S,V=np.linalg.svd(t[:52],full_matrices=False); Y=np.c_[y,t@V[:2].T]; scale=Y[:52].std(0); Y=Y/scale; i=np.arange(52);j=np.arange(52,len(y)); d0=4
print('ref',a,b, 'telemetry',s['telemetry_vec']); print('scalar-state corr',np.corrcoef(Y.T))
def score(P):
 r=(P-Y)*scale;return np.mean(abs(r[j,0])),np.mean(abs(r[j,1:]@V[:2])),np.mean(abs(r[i,0])),np.mean(abs(r[i,1:]@V[:2]))
def gram(w,mode):
 c,d=(f(w) for f in funcs); dp=(c*d).sum(1); cross=np.einsum('ni,nj->nij',c,d).reshape(len(c),-1)
 F=ones(np.c_[c,d, dp if mode!='cross' else cross]); T=ones(np.c_[c,d,cross if mode!='diag' else c*d]);return F,T
for w in (z,x):
 F,T=gram(w,'dot'); A=np.c_[F@ls(F[i],Y[i,0]),T@ls(T[i],Y[i,1:])];print('res corrs',np.corrcoef((A-Y).T),score(A))
for name,fn in [('affine',lambda w:w),('tanh',np.tanh),('sin',np.sin),('soft',lambda w:w/(1+abs(w))),('warp',lambda w:w/(1+w*w))]:
 for base in [z,x]:
  for mode in ['dot','cross','diag']:
   for reg in [.01,.06,.25]:
    def pred(p):
     w=base+fn(z@p[:16].reshape(4,4)+p[16:20])
     F,T=gram(w,mode); return np.c_[F@ls(F[i],Y[i,0]),T@ls(T[i],Y[i,1:])]
    def loss(p):return np.r_[(pred(p)[i]-Y[i]).ravel(),reg*p]
    fit=least_squares(loss,np.zeros(20),max_nfev=65); print('joint',name,base is x,mode,reg,score(pred(fit.x)),flush=True)
# Fit the archive's elementary nonlinearities directly in a shared spatial basis;
# scan common frequency and sharpness, using ridge to stabilize products.
R=[]
for nm,w in [('z',z),('x',x),('Bz',z@B+h)]:
 F,T=gram(w,'dot');
 for k in [.4,.84,1.27,2.15,3.5,5.]:
  q=k*w
  for gn,g in [('soft',q/(1+abs(q))),('cos',np.cos(q)),('sin',np.sin(q)),('rat',q/(1+q*q)),('tanh',np.tanh(q)),('mix',np.c_[np.sin(q),np.cos(q),q/(1+abs(q))])]:
   for order in [0,1,2]:
    E=[g];
    if order>=1:E.append(np.einsum('ni,nj->nij',g,w).reshape(len(w),-1))
    if order==2:E.append(np.einsum('ni,nj->nij',g,g).reshape(len(w),-1))
    G=np.column_stack(E); G-=G[i].mean(0);G/=np.maximum(G[i].std(0),.02)
    for ridge in [.003,.03,.15,1.]:
     def calc(H,target):
      A=np.c_[H,G]; sc=np.maximum(np.sqrt(np.mean(A[i]**2,axis=0)),.02);A=A/sc; D=np.r_[np.full(H.shape[1],ridge*.03),np.full(G.shape[1],ridge)]; return A@np.linalg.solve(A[i].T@A[i]+np.diag(D),A[i].T@target[i]);
     Q=np.c_[calc(F,Y[:,0]),calc(T,Y[:,1:])]; R.append((*score(Q),nm,k,gn,order,ridge))
print('grid y',sorted(R)[:10]);print('grid t',sorted(R,key=lambda a:a[1])[:10]);print('grid joint',sorted(R,key=lambda a:max(a[:2]))[:10])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/joint10.py > /tmp/joint10.log; grep -E 'ref|corr|grid' /tmp/joint10.log; grep '^joint' /tmp/joint10.log | sort -t '(' -k2,2g | head -10; grep '^joint' /tmp/joint10.log | sort -t ',' -k2,2g | head -10
```
- **Container Exit Code**: `0` (`16877 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
int soft False dot 0.25 (0.06890906437473589, 0.086401325142114, 0.041671140164588685, 0.023239119715132754)
joint soft True dot 0.25 (0.07245436956259557, 0.0793345265951649, 0.038714545154507354, 0.025635929790963748)
joint warp True dot 0.25 (0.07445003048986222, 0.07814265732093292, 0.04005034552853756, 0.0277838466752748)
joint warp False dot 0.25 (0.07555255764327508, 0.0811606994871012, 0.04266284667072296, 0.021257287723798104)
joint warp True diag 0.25 (0.07718278604515261, 0.11725939633916194, 0.048923657209927515, 0.05835545564651531)
joint soft True diag 0.25 (0.07847715308834215, 0.12248863127012251, 0.042998124760201824, 0.060372921242665176)
joint warp True diag 0.06 (0.07989269657250578, 0.11652534846653118, 0.05221594536621868, 0.0589493469102168)
joint tanh True diag 0.25 (0.08000683420498025, 0.1484212528325892, 0.050758082893141106, 0.07360491315731922)
joint warp True diag 0.01 (0.08027788658021237, 0.1163218333585411, 0.05222364810547739, 0.05928989566044512)
joint soft True diag 0.06 (0.08076376333199384, 0.11967844525493883, 0.04188637114983772, 0.05959612878233652)
joint sin False cross 0.25 (0.10399243094246238, 0.06213145070231276, 0.02531161315879502, 0.028347565795233565)
joint tanh False cross 0.25 (0.10569169966117684, 0.06543945234233689, 0.027032365904243388, 0.02952290594585052)
joint soft True dot 0.06 (0.0954590326463056, 0.07057733770749326, 0.036967075031023006, 0.032975719509452536)
joint soft True dot 0.01 (0.09554631725903467, 0.07073551908460177, 0.03660280959706894, 0.03271867930572398)
joint affine False cross 0.01 (0.09379013392066718, 0.07483650291607809, 0.02262796523288359, 0.031404417551067744)
joint affine False cross 0.06 (0.09408623576607461, 0.07517716694430954, 0.022615829417569216, 0.031611851072501466)
joint sin False cross 0.01 (0.0968167473037473, 0.07650057456789446, 0.024243864959100785, 0.031017566388411945)
joint sin False cross 0.06 (0.09698428531710344, 0.07659418335415955, 0.02442055152908773, 0.031110071315916296)
joint affine False cross 0.25 (0.11208817985108463, 0.0776663945089667, 0.029238123499485087, 0.023252997137134888)
joint soft False cross 0.06 (0.09404292968149106, 0.07804146221298218, 0.025002861539387782, 0.03547903444988613)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9906762b418dd6f4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9906762b (directory).
```

### Turn 11 — `execute_in_remote_docker` (`62394 ms` LLM)
- **Rationale**: Test additive and pairwise structure with separately tuned readouts, using training-only model selection. Compare the resulting artifact with the prior node on independent outer folds and check the official DAG contract in an isolated process.
- **Alternatives Weighed**: `Promote the best fitted coordinate warp` (rejected: Its scalar MAE remained above tolerance and its trajectory error increased relative to the saved baseline.)

```bash
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cp /holdout_task/node_00.py /tmp/node00_previous.py
sed '/^def _kernel/,$d' /tmp/node00_previous.py >/holdout_task/node_00.py
cat >>/holdout_task/node_00.py <<'PY'

def _rbf(u, width):
    d = np.maximum(np.sum(u*u,axis=1)[:,None]+np.sum(u*u,axis=1)[None,:]-2*u@u.T,0.)
    return np.exp(-d/(2*width**2))


def _design(study):
    x = np.vstack((study['train']['x'], study['test']['x']))
    a,b = _align(study['panel_a'])
    z = (x-a)/b
    c,d = (_branch(study['panel_c'], z,k) for k in (1,2))
    cx,dx = (_branch(study['panel_c'], x,k) for k in (1,2))
    def cross(c,d):
        return (c[:,:,None]*d[:,None,:]).reshape(len(c),-1)
    scalar = _ones(np.c_[c,d,np.sum(c*d,axis=1)])
    state = _ones(np.c_[c,d,cross(c,d)])
    rawstate = _ones(np.c_[cx,dx,cross(cx,dx)])
    n = len(study['train']['x'])
    def norm(v):
        return (v-v[:n].mean(0))/np.maximum(v[:n].std(0),1.e-6)
    Z,C,D,X,CX,DX = map(norm,(z,c,d,x,cx,dx))
    kernels=[]
    for U in (Z,np.c_[Z,C,D]/np.sqrt(3),np.c_[X,CX,DX]/np.sqrt(3)):
        for h in (.65,1.25,2.5,5.):
            kernels.append(_rbf(U,h))
    affine = _ls(_ones(study['panel_b']['u']),study['panel_b']['v'])
    w = _ones(z) @ affine
    for U in (Z,norm(w), np.c_[Z,norm(w)], np.c_[C,D]):
        for h in (.4,.8,1.6,3.2):
            K = np.array([_rbf(v[:,None],h) for v in U.T]); S=K.sum(0)
            kernels.append(S/len(K))
            kernels.append((S*S-np.sum(K*K,axis=0))/(len(K)*(len(K)-1)))
    tel = float(study['telemetry_vec'][2])
    prim = np.c_[z,c,d,w]
    pairs = np.column_stack([z[:,i]+sign*z[:,j] for i in range(4) for j in range(i+1,4) for sign in (-1,1)])
    P=np.c_[prim,pairs]
    F=[]
    for rate in (.84,1.27,tel,3.5):
        q=rate*P
        F.extend((np.sin(q),np.cos(q),q/(1+abs(q)),q/(1+q*q),1/(1+q*q)))
    for u in (z,c,d,w):
        for k in (1.27,tel):
            q=k*u
            for v in (np.sin(q),np.cos(q),q/(1+abs(q)),q/(1+q*q)):
                F.append(cross(z,v))
    F = np.clip(norm(np.column_stack(F)), -15.,15.)
    for U in (F[:,:len(F.T)//2], F):
        kernels.append(U@U.T/U.shape[1])
    return scalar,state,rawstate,kernels,F


def _predict(target, bases, kernels, features, folds, n):
    from scipy.linalg import cho_factor, cho_solve
    q=len(kernels)
    target=np.asarray(target)
    candidates=[]
    def gp(B,I,J,K,r):
        L=cho_factor(K[np.ix_(I,I)] + r*np.eye(len(I)),check_finite=False)
        A=B[I]; Y=target[I]
        Q=cho_solve(L,np.c_[Y,A],check_finite=False)
        m=Y.shape[1]
        beta=_ls(A.T@Q[:,m:]+.005*np.eye(A.shape[1]), A.T@Q[:,:m])
        return B[J]@beta + K[np.ix_(J,I)]@(Q[:,:m]-Q[:,m:]@beta)
    def matching(B,I,J,steps):
        A=B[I]; D=features[I]; G=features[J]; Y=target[I]
        coef=_ls(A,np.c_[Y,D]); H=D-A@coef[:,Y.shape[1]:]
        H=H/np.maximum(np.linalg.norm(H,axis=0),1.e-8)
        Y0=Y-A@coef[:,:Y.shape[1]]; R=Y0.copy(); chosen=[]
        for _ in range(steps):
            p=np.linalg.norm(H.T@R,axis=1)
            p[chosen]=-1
            chosen.append(int(np.argmax(p)))
            E=np.c_[A,D[:,chosen]]
            coef1=_ls(E,Y); R=Y-E@coef1
        return np.c_[B[J],G[:,chosen]]@coef1
    all_idx=np.arange(n)
    test=np.arange(n,len(bases[0]))
    for B in bases:
        for k in range(q):
            for r in (.001,.03,.3):
                candidates.append((B,k,r))
        for steps in (3,7,12):
            candidates.append((B,-1,steps))
    def evaluate(c,I,J):
        B,k,r=c
        return gp(B,I,J,kernels[k],r) if k>=0 else matching(B,I,J,r)
    results=[]
    for c in candidates:
        error=[]
        for J in folds:
            I=np.setdiff1d(all_idx,J,assume_unique=True)
            error.append(np.mean(abs(evaluate(c,I,J)-target[J])))
        v=np.mean(error)+.20*np.max(error)
        results.append(v if np.isfinite(v) else np.inf)
    rank=np.argsort(results)
    keep=[]; seen=set(); best=max(results[rank[0]],1.e-10)
    for k in rank:
        B,index,r=candidates[k]
        key=(id(B),index)
        if key in seen: continue
        if len(keep)>=8 or results[k]>1.25*best: break
        seen.add(key);keep.append(k)
    weights=np.exp(-12*(np.array(results)[keep]/best-1))
    return np.average([evaluate(candidates[k],all_idx,test) for k in keep],axis=0,weights=weights)


def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; train=s['train']; n=len(train['x'])
    scalar,state,rawstate,kernels,features=_design(s)
    rng=np.random.default_rng(1709)
    folds=[np.sort(j) for j in np.array_split(rng.permutation(n),4)]
    y=_predict(np.asarray(train['y'])[:,None], [scalar,scalar[:,:1]],kernels,features,folds,n)[:,0]
    t=_predict(np.asarray(train['traj']),[state,rawstate,scalar,scalar[:,:1]],kernels,features,folds,n)
    return {'y_hat':y,'unprojected':t}
PY
python3 -I /holdout_task/test_dag_candidate.py || true
python3 -I - <<'PY'
import pickle,runpy,numpy as np,copy
p='/holdout_task/'; s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; p0=runpy.run_path('/tmp/node00_previous.py')['run_node']; p1=runpy.run_path(p+'node_00.py')['run_node']; time=runpy.run_path(p+'_frozen_nodes/node_01.py')['run_node']; finish=runpy.run_path(p+'_frozen_nodes/node_02.py')['run_node']; tr=s['train']; ix=np.arange(len(tr['x'])); scores=[]
for jj in np.array_split(np.random.default_rng(417).permutation(len(ix)),4):
 ii=np.setdiff1d(ix,jj); v=copy.deepcopy(s); v['train'].update({k:tr[k][ii] for k in ('x','y','traj')});v['test']={'x':tr['x'][jj]}; h=time({'__input__':v}); r=[]
 for fn in (p0,p1):
  f=finish({'node_00':fn({'__input__':v}),'node_01':h});r.extend([np.mean(abs(f['y_hat']-tr['y'][jj])),np.mean(abs(f['state_pred']-tr['traj'][jj]))]);
 scores.append(r)
print('Independent outer folds [previous y,t; challenger y,t]:',np.array(scores).round(6).tolist());print('Means:',np.mean(scores,axis=0).round(6).tolist(),'worst:',np.max(scores,axis=0).round(6).tolist())
PY
```
- **Container Exit Code**: `0` (`4384 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
{"accepted": false, "errors": ["public_cv_y_mae_exceeded:0.086217", "public_cv_traj_mae_exceeded:0.056467"], "cv_metrics": {"public_cv_y_mae": 0.086217, "public_cv_traj_mae": 0.056467}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}
Independent outer folds [previous y,t; challenger y,t]: [[0.094728, 0.079482, 0.091619, 0.050422], [0.084722, 0.093579, 0.077551, 0.066401], [0.08231, 0.085038, 0.085551, 0.059281], [0.067598, 0.090854, 0.076306, 0.046648]]
Means: [0.08234, 0.087238, 0.082757, 0.055688] worst: [0.094728, 0.093579, 0.091619, 0.066401]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ea205198110f24d4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ea205198 (directory).
```

### Turn 12 — `submit_completed_task` (`13883 ms` LLM)
- **Rationale**: Retained the verified artifact with lower worst-fold errors. The test established that the DAG runs correctly, not that either accuracy target was met.
- **Alternatives Weighed**: `Restore the earlier baseline` (rejected: It had a slightly lower official scalar MAE, but worse worst-fold MAE for both readouts and worse official trajectory MAE.)
- **Submitted Final Answer / Summary**: `Saved `/holdout_task/node_00.py` and verified it in an isolated Python process. All three DAG nodes executed exactly once, but the validator did not accept the candidate: scalar MAE was 0.086217 and trajectory MAE was 0.056467, both above the 0.038 limit.`

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
