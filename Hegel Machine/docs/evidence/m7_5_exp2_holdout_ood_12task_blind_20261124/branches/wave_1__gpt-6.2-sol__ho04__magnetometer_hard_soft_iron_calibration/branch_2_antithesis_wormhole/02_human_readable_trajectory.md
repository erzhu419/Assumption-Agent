# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `ho04__magnetometer_hard_soft_iron_calibration`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-OOD-Holdout-12` |
| **Domain** | `Navigation & Sensor Fusion / Ellipsoid Calibration` |
| **Wave / Wormhole Family** | `wave_1` / `fam_robust_control_and_safety` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `16 / 28` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `2` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `556,070` (`538,059` / `101,760` / `18,011`) |
| **Total Episode Wall Time (s)** | `595.95s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`root_A03_ill_posed_regularization_and_certificates`**: AXIOM A03 (HEAVY-TAILED REGULARIZATION & DUAL CERTIFICATES): Under outliers, stage drift, or ill-posed combinatorial landscapes, replace naive L2 least squares with heavy-tailed robust losses (Huber/Cauchy/MAD) or exact variable projection, and verify optimality/safety margins via dual bounds or worst-corner simulation.
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`20504 ms` LLM)
- **Rationale**: Inspect compact summaries in the container to catch schema or units mismatches and distinguish labeled validation from tests based on physical invariants.
- **Alternatives Weighed**: `Fit the norm-only ellipsoid immediately` (rejected: First establish the feature layout, available references, and roll/pitch coverage; these determine how to validate heading and whether the norm-only fit is well constrained.)

```bash
ls -la /holdout_task /tmp/hegel_wormhole_skills; python3 - <<'PY'
import pickle
import numpy as np
with open('/holdout_task/public_mag.pkl','rb') as f:
    obj=pickle.load(f)
def describe(x, name='public', depth=0):
    if isinstance(x,dict):
        print(name, 'keys:',list(x));
        if depth < 3:
            for k,v in x.items(): describe(v,f'{name}.{k}',depth+1)
    elif isinstance(x,np.ndarray):
        print(name,x.shape,x.dtype, 'sample:',x[:2] if x.ndim else x.item(), 'quantiles:',np.nanquantile(x,[0,.01,.5,.99,1],axis=0).round(5) if x.size and x.ndim and x.ndim<=2 and np.issubdtype(x.dtype,np.number) else '')
    elif isinstance(x,(list,tuple)):
        print(name,type(x).__name__,len(x));
        if len(x) and depth<3: describe(x[0],name+'[0]',depth+1)
    else: print(name,type(x).__name__,str(x)[:160])
describe(obj)
PY
```
- **Container Exit Code**: `0` (`608 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/holdout_task:
total 32
drwxr-xr-x 2 root root  4096 Nov 24 08:21 .
drwxr-xr-x 1 root root  4096 Nov 24 08:21 ..
-rw-r--r-- 1 root root 23144 Nov 24 08:21 public_mag.pkl

/tmp/hegel_wormhole_skills:
total 36
drwxr-xr-x 3 ubuntu ubuntu 4096 Nov 24 08:21 .
drwxrwxrwt 1 root   root   4096 Nov 24 08:21 ..
-rw-r--r-- 1 ubuntu ubuntu 4369 Nov 24 08:21 __init__.py
drwxr-xr-x 2 root   root   4096 Nov 24 08:21 __pycache__
-rw-r--r-- 1 ubuntu ubuntu 7134 Nov 24 08:21 autogenous_conservative_elliptic_multigrid.py
-rw-r--r-- 1 ubuntu ubuntu 8025 Nov 24 08:21 autogenous_creep_corrected_retraction_calibration.py
public keys: ['session', 'gt']
public.session keys: ['calib_m_raw', 'calib_roll_pitch_rad', 'test_m_raw', 'test_roll_pitch_rad']
public.session.calib_m_raw (420, 3) float64 sample: [[ 32.72896585  -3.85739768  44.55170701]
 [ 71.03954969 -42.67050232  41.98786812]] quantiles: [[-23.46472 -80.90815  12.65165]
 [-21.28275 -78.11892  23.55374]
 [ 26.19471 -39.86749  50.55713]
 [ 76.02039  -0.19697  61.47747]
 [ 77.60462   0.54958  89.19001]]
public.session.calib_roll_pitch_rad (420, 2) float64 sample: [[ 0.46064931  0.13020456]
 [-0.43913513 -0.4172756 ]] quantiles: [[-0.74971 -0.64896]
 [-0.73952 -0.62624]
 [ 0.0017  -0.03662]
 [ 0.71207  0.6369 ]
 [ 0.74126  0.64488]]
public.session.test_m_raw (120, 3) float64 sample: [[ -4.76198481 -52.64081502  52.48394928]
 [ -6.06556345 -50.69433295  52.47467309]] quantiles: [[ -9.68416 -72.44806  42.6698 ]
 [ -9.6073  -72.35573  42.82605]
 [ 26.1621  -36.51395  52.57855]
 [ 65.36714  -9.59058  61.13596]
 [ 65.53406  -9.48044  61.16756]]
public.session.test_roll_pitch_rad (120, 2) float64 sample: [[-0.3821346   0.06953427]
 [-0.35959263  0.10418026]] quantiles: [[-0.4     -0.34997]
 [-0.39913 -0.34963]
 [ 0.01485 -0.     ]
 [ 0.39893  0.34963]
 [ 0.39896  0.34997]]
public.gt keys: ['S_soft', 'b_hard', 'gt_heading_deg']
public.gt.S_soft (3, 3) float64 sample: [[ 1.17564433  0.06974161 -0.03985235]
 [ 0.06974161  0.91660405  0.04981544]] quantiles: [[-0.03985  0.04982 -0.03985]
 [-0.03766  0.05021 -0.03806]
 [ 0.06974  0.06974  0.04982]
 [ 1.15353  0.89967  0.9188 ]
 [ 1.17564  0.9166   0.93653]]
public.gt.b_hard (3,) float64 sample: [ 28.5 -41. ] quantiles: [-41.    -39.856  16.2    28.254  28.5  ]
public.gt.gt_heading_deg (120,) float64 sample: [-174.27042205 -171.34150739] quantiles: [-174.27042 -170.78501    0.       170.78501  174.27042]
```

### Turn 2 — `execute_in_remote_docker` (`31093 ms` LLM)
- **Rationale**: Use the public truth to check the leveling convention and noise floor, then compare ordinary and robust fits of the complete linear vertical constraint. If the vertical initializer is poorly conditioned, supplement it with a norm-based seed.
- **Alternatives Weighed**: `Estimate a general affine correction` (rejected: The stated soft-iron correction is symmetric positive definite; a general affine fit would introduce unnecessary rotational degrees of freedom.)

```bash
cd /holdout_task; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle, inspect
import numpy as np
from scipy.optimize import least_squares
from hegel_wormhole_skills.autogenous_creep_corrected_retraction_calibration import calibrate_retraction
from hegel_wormhole_skills.autogenous_conservative_elliptic_multigrid import solve_darcy_pressure
for fn in (calibrate_retraction, solve_darcy_pressure):
    print(inspect.getsource(fn)[:1700])
p=pickle.load(open('public_mag.pkl','rb')); s=p['session']; g=p['gt']
def axes(rp):
    r,p=rp.T; cr,sr,cp,sp=np.cos(r),np.sin(r),np.cos(p),np.sin(p)
    return np.stack([np.stack([cp,sp*sr,sp*cr],1),np.stack([0*r,cr,-sr],1),np.stack([-sp,cp*sr,cp*cr],1)],1)
A=axes(s['calib_roll_pitch_rad']); T=axes(s['test_roll_pitch_rad']); X=s['calib_m_raw']; Y=s['test_m_raw']
def level(x,R,W,b): return np.einsum('nij,nj->ni',R,(x-b)@W.T)
def heading(l): return np.degrees(np.arctan2(-l[:,1],l[:,0]))
def rmse(x): return np.mean(((x-g['gt_heading_deg']+180)%360-180)**2)**.5
G=np.linalg.inv(g['S_soft']); L=level(X,A,G,g['b_hard']); print('oracle heading RMSE',rmse(heading(level(Y,T,G,g['b_hard'])))); print('field H,D',np.median(np.linalg.norm(L[:,:2],axis=1)),np.median(L[:,2]));
for z in (np.linalg.norm(L,axis=1),L[:,2],np.linalg.norm(L[:,:2],axis=1)):
    print('absolute residual quantiles',np.quantile(abs(z-np.median(z)),[.5,.9,.94,.95,.96,.97,.99,1]))
O=np.median(X,0); Q=np.median(np.linalg.norm(X-O,axis=1)); x=(X-O)/Q; v=A[:,2]; a,b,c=v.T; i,j,k=x.T
F=np.column_stack([a*i-c*k,b*j-c*k,a*j+b*i,a*k+c*i,b*k+c*j,-v,-np.ones(len(x))]); y=-3*c*k
print('vertical design singular values',np.linalg.svd(F,compute_uv=False));
for loss in ('linear','cauchy'):
    s0=np.linalg.lstsq(F,y,rcond=None)[0]; f=least_squares(lambda p:F@p-y,s0,jac=lambda p:F,loss=loss,f_scale=.006); a,b,d,e,h=f.x[:5]; W=np.array([[a,d,e],[d,b,h],[e,h,3-a-b]]); bias=O+Q*np.linalg.solve(W,f.x[5:8]); W/=np.cbrt(np.linalg.det(W)); print(loss,'bias error',np.linalg.norm(bias-g['b_hard']),'heading RMSE',rmse(heading(level(Y,T,W,bias))), 'SPD spectrum',np.linalg.eigvalsh(W))
PY
```
- **Container Exit Code**: `0` (`525 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rvative_elliptic_multigrid")
def solve_darcy_pressure(permeability_2d: np.ndarray, source_rhs: np.ndarray, h: float, tol: float=1e-08) -> np.ndarray:
    """Solve -div(K grad(p)) = source on a uniform cell-centered 2D grid.

    Inputs are matching rectangular arrays with positive scalar permeability
    and equal spacing h along both axes. Interior faces use harmonic means;
    each exterior face uses the adjacent K and a zero-pressure ghost one h
    away (not a zero value at the face). No diagonal regularization is added.
    The symmetric Galerkin V-cycle uses a normalized strength cutoff of 0.08
    and a dense Cholesky solve only on levels with at most 64 unknowns.
    Return float64 pressure unless refinement needs longdouble to meet tol.
    Raise RuntimeError if the directly recomputed relative residual cannot
    be reduced to tol. Extra precision depends on the platform.
    """
    k = np.asarray(permeability_2d, dtype=np.float64)
    rhs = np.asarray(source_rhs, dtype=np.float64)
    if k.ndim != 2 or min(k.shape) < 2 or rhs.shape != k.shape:
        raise ValueError('expected matching 2D arrays with both axes at least 2')
    if not np.all(np.isfinite(k)) or np.any(k <= 0) or (not np.all(np.isfinite(rhs))):
        raise ValueError('expected positive finite permeability and finite source')
    if not np.isfinite(h) or h <= 0 or (not 0 < tol < 1):
        raise ValueError('expected finite h > 0 and 0 < tol < 1')

    def dot(x: np.ndarray, y: np.ndarray) -> np.floating:
        return np.einsum('i,i->', x, y)
    b = rhs.ravel() * (h * h)
    bb = dot(b, b)
    if bb == 0:
        return np.zeros_like(k)
    limit = tol 
oracle heading RMSE 0.1310671018690447
field H,D 23.49158373505267 41.99447680512864
absolute residual quantiles [ 0.06828268  0.15109388  0.20204705  0.21455429  0.2599909   2.19639317
 18.9519941  39.73696521]
absolute residual quantiles [ 0.06263549  0.16015834  0.19968985  0.21984963  0.26867822  6.55160811
 20.98415916 37.71522222]
absolute residual quantiles [ 0.05801411  0.15390627  0.19338095  0.20160562  0.26404502  3.4321769
 15.73966377 26.46308442]
vertical design singular values [27.60631397 13.75553311 11.04042023 10.58177138  8.59566087  5.06108507
  4.9582783   4.75827549  1.2158882 ]
linear bias error 12.59404522737951 heading RMSE 3.113302821815208 SPD spectrum [0.88666451 0.96467619 1.16912008]
cauchy bias error 12.59404522737951 heading RMSE 3.113302821815208 SPD spectrum [0.88666451 0.96467619 1.16912008]
```

### Turn 3 — `execute_in_remote_docker` (`60092 ms` LLM)
- **Rationale**: Use both yaw-invariant circle coordinates and whiten their correlated residuals before downweighting whole samples. The first public run checks the correction direction, determinant, bias, and circular heading error by yaw and tilt; its results will guide the calibration holdout and stress tests.
- **Alternatives Weighed**: `Treat the identical Cauchy and OLS results as evidence against robust fitting` (rejected: The Jacobian callback returned a shared design array that SciPy's robust scaling can mutate. The new initializer uses explicit reweighted linear solves.); `Reuse the mounted retraction or Darcy solver` (rejected: Their required inputs and physical models do not apply. The calibration instead uses the transferable principle of robust residual weighting.)

```bash
set -e
cat > /holdout_task/mag_calibrator.py <<'PY'
"""Robust, determinant-one magnetometer calibration from roll and pitch."""
import numpy as np
from scipy.optimize import least_squares


def _level_axes(roll_pitch):
    roll, pitch = np.asarray(roll_pitch, dtype=float).T
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.stack((np.stack((cp, sp * sr, sp * cr), axis=1),
                     np.stack((np.zeros_like(cr), cr, -sr), axis=1),
                     np.stack((-sp, cp * sr, cp * cr), axis=1)), axis=1)


def _symmetric(p, trace=0.):
    a, b, xy, xz, yz = p
    return np.array([[a, xy, xz], [xy, b, yz], [xz, yz, trace - a - b]])


def _correction(p):
    eigenvalues, axes = np.linalg.eigh(_symmetric(p[:5]))
    return (axes * np.exp(eigenvalues)) @ axes.T


def _seed(x, axes):
    # The vertical equation is linear in W, W*b and field-down.
    # Fix trace(W)=3 to solve it, then convert to the determinant gauge.
    v = axes[:, 2]
    a, b, c = v.T
    i, j, k = x.T
    mat = np.column_stack((a*i-c*k, b*j-c*k, a*j+b*i,
                           a*k+c*i, b*k+c*j, -v, -np.ones(len(x))))
    rhs = -3*c*k
    z = np.linalg.lstsq(mat, rhs, rcond=None)[0]
    for _ in range(18):
        error = mat @ z - rhs
        scale = max(1.4826 * np.median(abs(error - np.median(error))), 1e-6)
        weight = 1 / np.hypot(1, error / (2.5 * scale))
        z1 = np.linalg.lstsq(mat * weight[:, None], rhs * weight, rcond=None)[0]
        if np.linalg.norm(z1 - z) < 1e-8:
            break
        z = z1
    w = _symmetric(z[:5], trace=3.)
    val, vec = np.linalg.eigh(w)
    if val[0] > .08:
        bias = np.linalg.solve(w, z[5:8])
        logval = np.log(val)
        logw = (vec * (logval - logval.mean())) @ vec.T
    else:
        bias = np.zeros(3)
        logw = np.zeros((3, 3))
    p = np.r_[logw[0, 0], logw[1, 1], logw[0, 1], logw[0, 2], logw[1, 2], bias]
    level = np.einsum('nij,nj->ni', axes, (x - bias) @ _correction(p))
    return np.r_[p, np.median(np.linalg.norm(level[:, :2], axis=1)), np.median(level[:, 2])]


def _residual(p, x, axes):
    w = _correction(p)
    level = np.einsum('nij,nj->ni', axes, (x - p[5:8]) @ w)
    h = np.linalg.norm(level[:, :2], axis=1)
    # Both residuals come from the same raw 3-vector. Whiten their local
    # covariance so the loss measures distance in raw sensor units.
    hdir = level[:, :2] / np.maximum(h[:, None], 1e-12)
    radial = np.einsum('ni,nij->nj', hdir, axes[:, :2]) @ w
    vertical = axes[:, 2] @ w
    vz = np.linalg.norm(vertical, axis=1)
    unit_z = vertical / vz[:, None]
    cross = np.einsum('ni,ni->n', radial, unit_z)
    orth = np.linalg.norm(radial - cross[:, None] * unit_z, axis=1)
    dz = (level[:, 2] - p[9]) / vz
    dh = (h - p[8] - cross * dz) / np.maximum(orth, 1e-12)
    return np.column_stack((dz, dh))


def _fit(x, axes):
    p = _seed(x, axes)
    for _ in range(12):
        residual = _residual(p, x, axes)
        distance = np.linalg.norm(residual, axis=1)
        # The median of a two-dimensional Gaussian radius is sigma*sqrt(2 ln 2).
        sigma = max(np.median(distance) / np.sqrt(2 * np.log(2)), 1e-6)
        weight = 1 / np.sqrt(1 + (distance / (2.5 * sigma))**2)
        fit = least_squares(lambda q: (_residual(q, x, axes) * weight[:, None]).ravel(),
                            p, max_nfev=40)
        change = np.linalg.norm(fit.x - p)
        p = fit.x
        if change < 1e-7:
            break
    return p


def calibrate_and_estimate_heading(session: dict) -> dict:
    """Recover physical soft- and hard-iron parameters and test magnetic yaw."""
    raw = np.asarray(session['calib_m_raw'], dtype=float)
    axes = _level_axes(session['calib_roll_pitch_rad'])
    origin = np.median(raw, axis=0)
    scale = np.median(np.linalg.norm(raw - origin, axis=1))
    p = _fit((raw - origin) / scale, axes)
    correction = _correction(p)
    bias = origin + scale * p[5:8]
    test = (np.asarray(session['test_m_raw'], dtype=float) - bias) @ correction
    level = np.einsum('nij,nj->ni', _level_axes(session['test_roll_pitch_rad']), test)
    heading = np.rad2deg(np.arctan2(-level[:, 1], level[:, 0]))
    heading = np.where(heading <= -180., 180., heading)
    return {'S_soft': np.linalg.inv(correction), 'b_hard': bias, 'heading_deg': heading}
PY
f=/holdout_task/mag_calibrator.py; mkdir -p /tmp/.hegel_candidate_archive; s=$(sha256sum "$f" | cut -c1-8); cp "$f" "/tmp/.hegel_candidate_archive/v01_$s"
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
import mag_calibrator as m
p=pickle.load(open('public_mag.pkl','rb')); s=p['session']; g=p['gt']; r=m.calibrate_and_estimate_heading(s)
a=m._level_axes(s['calib_roll_pitch_rad']); raw=s['calib_m_raw']; o=np.median(raw,axis=0); d=np.median(np.linalg.norm(raw-o,axis=1)); seed=m._seed((raw-o)/d,a)
print('seed bias error',np.linalg.norm(o+d*seed[5:8]-g['b_hard']))
e=(r['heading_deg']-g['gt_heading_deg']+180)%360-180
print('full heading, bias, determinant, soft spectral error',np.sqrt(np.mean(e**2)),np.linalg.norm(r['b_hard']-g['b_hard']),np.linalg.det(r['S_soft']),np.linalg.norm(r['S_soft']-g['S_soft'],2))
for key, bins in [('yaw',np.array_split(np.argsort(g['gt_heading_deg']),8)),('tilt',np.array_split(np.argsort(np.linalg.norm(s['test_roll_pitch_rad'],axis=1)),4))]:
    print(key,'RMSE', [round(np.sqrt(np.mean(e[ii]**2)),5) for ii in bins])
PY
```
- **Container Exit Code**: `0` (`527 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/mag_calibrator.py']`
- **Stdout (tail)**:
```text
seed bias error 0.04382451763550565
full heading, bias, determinant, soft spectral error 0.13354694482560006 0.01654660456588235 0.9999999999999991 0.0006732916313609447
yaw RMSE [0.13412, 0.13377, 0.15854, 0.13818, 0.1293, 0.13307, 0.13266, 0.10268]
tilt RMSE [0.13254, 0.13998, 0.13371, 0.12766]
```

### Turn 4 — `execute_in_remote_docker` (`38061 ms` LLM)
- **Rationale**: Leave out observed yaw sectors, tilt quadrants, and contiguous calibration blocks to test dependence on coverage. Stress coherent spikes, weaker horizontal fields, and reduced tilt separately; use any threshold failures to decide whether the fit needs stronger rejection or a better initializer.
- **Alternatives Weighed**: `Freeze the candidate on the full public session` (rejected: That run does not detect sensitivity to missing calibration orientations or to a coherent current-spike episode.)

```bash
cat > /tmp/mag_validate.py <<'PY'
import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import mag_calibrator as m

p = pickle.load(open('/holdout_task/public_mag.pkl', 'rb'))
s, g = p['session'], p['gt']

def errors(result, truth):
    yaw = (result['heading_deg'] - truth['gt_heading_deg'] + 180) % 360 - 180
    return np.sqrt(np.mean(yaw**2)), np.linalg.norm(result['b_hard'] - truth['b_hard'])

def report(name, metrics):
    a = np.asarray(metrics)
    print(f'{name:27s} n={len(a):2d} median={np.median(a, axis=0).round(5)} max={a.max(0).round(5)}', flush=True)

def subset(mask):
    return {k: (v[mask] if k.startswith('calib_') else v) for k,v in s.items()}

if __name__ == '__main__':
    axes = m._level_axes(s['calib_roll_pitch_rad'])
    level = np.einsum('nij,nj->ni', axes, (s['calib_m_raw']-g['b_hard']) @ np.linalg.inv(g['S_soft']))
    sector = ((np.arctan2(-level[:,1], level[:,0])+np.pi)*8/(2*np.pi)).astype(int)%8
    rp = s['calib_roll_pitch_rad']
    quadrant = (rp[:,0]>0).astype(int) + 2*(rp[:,1]>0)
    block = np.arange(len(rp))//70
    for key,group in [('calibration yaw sectors',sector),('calibration tilt quadrants',quadrant),('calibration time blocks',block)]:
        report(key,[errors(m.calibrate_and_estimate_heading(subset(group!=i)),g) for i in np.unique(group)])

def synthetic(seed, n=420, tilt=1., h=23.5, soft=.20, spike='random'):
    rng = np.random.default_rng(seed)
    q,_ = np.linalg.qr(rng.normal(size=(3,3)))
    shape = (q * np.exp(np.array([-1.,-.25,1.25])*soft)) @ q.T
    bias = rng.uniform(-65,65,size=3)
    def sample(rp,yaw,noise):
        v = np.column_stack((h*np.cos(yaw),-h*np.sin(yaw),np.full(len(yaw),42.)))
        u = np.einsum('nij,ni->nj', m._level_axes(rp),v)
        return u @ shape + bias + rng.normal(0,noise,size=u.shape)
    rp = rng.uniform(-1,1,size=(n,2))*[.75*tilt,.65*tilt]
    yaw = rng.uniform(-np.pi,np.pi,size=n)
    x = sample(rp,yaw,.10)
    k = round(.04*n)
    ix = rng.choice(n,k,replace=False) if spike!='cluster' else np.argsort(yaw)[-k:]
    if spike=='random':
        x[ix] += rng.normal(0,12,size=(k,3))
    else:
        direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
        x[ix] += rng.uniform(7,30,size=(k,1))*direction
    r = rng.uniform(-1,1,size=(180,2))*[.4,.35]
    y = np.linspace(-np.pi,np.pi,180,endpoint=False)
    return dict(calib_m_raw=x,calib_roll_pitch_rad=rp,test_m_raw=sample(r,y,.06),test_roll_pitch_rad=r), dict(b_hard=bias,S_soft=shape,gt_heading_deg=np.degrees(y))

CASES = [('nominal',{}),('coherent spikes',{'spike':'coherent'}),('localized spike episode',{'spike':'cluster'}),('reduced samples and tilt',{'n':180,'tilt':.5}),('steep field and strong iron',{'h':12,'soft':.40}),('combined stress',{'n':180,'tilt':.5,'h':12,'soft':.40,'spike':'cluster'})]
if __name__ == '__main__':
    for name, kwargs in CASES:
        pairs = [synthetic(i,**kwargs) for i in range(20)]
        report(name,[errors(m.calibrate_and_estimate_heading(x),t) for x,t in pairs])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/mag_validate.py
```
- **Container Exit Code**: `0` (`4820 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calibration yaw sectors     n= 8 median=[0.13332 0.02041] max=[0.13701 0.02695]
calibration tilt quadrants  n= 4 median=[0.13412 0.01973] max=[0.13507 0.04475]
calibration time blocks     n= 6 median=[0.13418 0.02033] max=[0.13483 0.02314]
nominal                     n=20 median=[0.15716 0.02033] max=[0.17784 0.0377 ]
coherent spikes             n=20 median=[0.15507 0.02017] max=[0.17725 0.03701]
localized spike episode     n=20 median=[0.15132 0.0204 ] max=[0.17035 0.04058]
reduced samples and tilt    n=20 median=[0.1593  0.06054] max=[0.19424 0.18722]
steep field and strong iron n=20 median=[0.3416  0.02558] max=[0.392   0.06636]
combined stress             n=20 median=[0.33928 0.0864 ] max=[0.39706 0.29693]
```

### Turn 5 — `execute_in_remote_docker` (`29925 ms` LLM)
- **Rationale**: Compare the current fit with hard rejection and smooth Tukey rejection on identical public folds and fresh synthetic seeds. Keep the baseline unless a refit materially improves the tightest error margin without creating a new failure.
- **Alternatives Weighed**: `Always discard exactly 4% of calibration samples` (rejected: Some interference vectors are nearly tangent to the magnetic circle and may fall inside the noise band, so a fixed quota can discard clean measurements without removing those spikes.)

```bash
cat > /tmp/mag_compare.py <<'PY'
import numpy as np
from mag_validate import m, s, g, errors, report, subset, synthetic, CASES

baseline = m._fit

def refit(rule):
    def fit(x, axes):
        p = baseline(x, axes)
        for _ in range(4):
            distance = np.linalg.norm(m._residual(p, x, axes), axis=1)
            sigma = max(np.median(distance) / np.sqrt(2 * np.log(2)), 1e-6)
            if rule == 'trim':
                weight = (distance <= 4 * sigma).astype(float)
            else:
                weight = np.maximum(0., 1 - (distance / (6 * sigma))**2)
            q = m.least_squares(lambda q: (m._residual(q, x, axes) * weight[:, None]).ravel(), p, max_nfev=25).x
            if np.linalg.norm(q-p) < 1e-7:
                return q
            p = q
        return p
    return fit

rp = s['calib_roll_pitch_rad']
axes = m._level_axes(rp)
level = np.einsum('nij,nj->ni', axes, (s['calib_m_raw']-g['b_hard']) @ np.linalg.inv(g['S_soft']))
yaw = np.arctan2(-level[:,1],level[:,0])
sectors = np.floor((yaw+np.pi)*8/(2*np.pi)).astype(int)%8
quadrants = (rp[:,0]>0).astype(int) + 2*(rp[:,1]>0)
public = [(s,g)]+[(subset(group!=i),g) for group in (sectors,quadrants,np.arange(len(rp))//70) for i in np.unique(group)]
cases = [('public + calibration jackknife',public)]
for name, kw in CASES:
    cases.append((name, [synthetic(i, **kw) for i in range(20, 44)]))
for name, pairs in cases:
    for label, fn in [('cauchy',baseline),('trim',refit('trim')),('tukey',refit('tukey'))]:
        m._fit = fn
        report(f'{name} {label}',[errors(m.calibrate_and_estimate_heading(sess),truth) for sess,truth in pairs])
m._fit = baseline
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/mag_compare.py
```
- **Container Exit Code**: `0` (`18843 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public + calibration jackknife cauchy n=19 median=[0.13355 0.01999] max=[0.13701 0.04475]
public + calibration jackknife trim n=19 median=[0.13302 0.01894] max=[0.13568 0.04193]
public + calibration jackknife tukey n=19 median=[0.13339 0.01962] max=[0.13659 0.04358]
nominal cauchy              n=24 median=[0.15413 0.03133] max=[0.17033 0.05814]
nominal trim                n=24 median=[0.15426 0.02409] max=[0.16936 0.05795]
nominal tukey               n=24 median=[0.15415 0.02819] max=[0.16998 0.05668]
coherent spikes cauchy      n=24 median=[0.15467 0.0309 ] max=[0.16569 0.05809]
coherent spikes trim        n=24 median=[0.15433 0.02409] max=[0.1649  0.05795]
coherent spikes tukey       n=24 median=[0.15455 0.02819] max=[0.16528 0.05668]
localized spike episode cauchy n=24 median=[0.15437 0.03035] max=[0.16628 0.05011]
localized spike episode trim n=24 median=[0.15415 0.02535] max=[0.16645 0.0483 ]
localized spike episode tukey n=24 median=[0.15428 0.0274 ] max=[0.1664  0.04788]
reduced samples and tilt cauchy n=24 median=[0.15574 0.07488] max=[0.18333 0.15242]
reduced samples and tilt trim n=24 median=[0.1556  0.08067] max=[0.18137 0.15336]
reduced samples and tilt tukey n=24 median=[0.15568 0.07748] max=[0.18271 0.15443]
steep field and strong iron cauchy n=24 median=[0.33013 0.03957] max=[0.37226 0.09564]
steep field and strong iron trim n=24 median=[0.33069 0.03742] max=[0.37314 0.09156]
steep field and strong iron tukey n=24 median=[0.3302  0.03856] max=[0.37259 0.09197]
combined stress cauchy      n=24 median=[0.32875 0.1243 ] max=[0.41487 0.34952]
combined stress trim        n=24 median=[0.32624 0.11416] max=[0.41605 0.32774]
combined stress tukey       n=24 median=[0.32816 0.11649] max=[0.41545 0.34003]
```

### Turn 6 — `execute_in_remote_docker` (`26021 ms` LLM)
- **Rationale**: Promote the refit that improves the tighter bias margin while staying under the heading target. Rerun the public calibration jackknife against the edited file, then check fresh combined-stress seeds for failures that the tuning comparison could have missed.
- **Alternatives Weighed**: `Use a Tukey refit` (rejected: It improved the combined-stress bias less than hard rejection (0.340 versus 0.328 µT), without a meaningful heading advantage.)

```bash
set -eu
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/mag_calibrator.py')
s = p.read_text()
old = '        if change < 1e-7:\n            break\n    return p\n'
new = '''        if change < 1e-7:
            break

    # Once the fit is close, reject large spikes and restore equal weights
    # to the inliers. Both coordinates are screened together per observation.
    for _ in range(4):
        distance = np.linalg.norm(_residual(p, x, axes), axis=1)
        sigma = max(np.median(distance) / np.sqrt(2 * np.log(2)), 1e-6)
        keep = distance <= 4 * sigma
        q = least_squares(lambda q: _residual(q, x[keep], axes[keep]).ravel(),
                          p, max_nfev=25).x
        change = np.linalg.norm(q - p)
        p = q
        if change < 1e-7:
            break
    return p
'''
assert s.count(old) == 1
p.write_text(s.replace(old,new))
PY
f=/holdout_task/mag_calibrator.py
s=$(sha256sum "$f" | cut -c1-8)
cp "$f" "/tmp/.hegel_candidate_archive/v02_$s"
OPENBLAS_NUM_THREADS=1 python3 /tmp/mag_validate.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp python3 - <<'PY'
import numpy as np
from mag_validate import m, synthetic, errors, report
pairs = [synthetic(i, n=180, tilt=.5, h=12, soft=.40, spike='cluster') for i in range(44,108)]
metrics = [errors(m.calibrate_and_estimate_heading(s), g) for s,g in pairs]
report('fresh combined stress', metrics)
a = np.asarray(metrics)
print('seeds exceeding heading or bias target:', (np.flatnonzero((a > [.45,.35]).any(1))+44).tolist())
PY
```
- **Container Exit Code**: `0` (`8222 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/mag_calibrator.py']`
- **Stdout (tail)**:
```text
calibration yaw sectors     n= 8 median=[0.13289 0.02098] max=[0.13568 0.02687]
calibration tilt quadrants  n= 4 median=[0.13347 0.01959] max=[0.13395 0.04193]
calibration time blocks     n= 6 median=[0.13336 0.0187 ] max=[0.13395 0.02458]
nominal                     n=20 median=[0.15731 0.02222] max=[0.17582 0.03742]
coherent spikes             n=20 median=[0.1552  0.02222] max=[0.17657 0.03742]
localized spike episode     n=20 median=[0.15137 0.02113] max=[0.17102 0.04075]
reduced samples and tilt    n=20 median=[0.15866 0.05651] max=[0.19406 0.20602]
steep field and strong iron n=20 median=[0.34209 0.0268 ] max=[0.38783 0.0563 ]
combined stress             n=20 median=[0.33974 0.09098] max=[0.39669 0.32203]
fresh combined stress       n=64 median=[0.33818 0.10736] max=[0.40246 0.37003]
seeds exceeding heading or bias target: [58]
```

### Turn 7 — `execute_in_remote_docker` (`35757 ms` LLM)
- **Rationale**: Compare the deployed fit against nearby rejection cutoffs and an oracle that knows which synthetic observations were corrupted. If the oracle also misses the bias target and its error aligns with the weakest fitted direction, treat the failure as limited calibration information rather than tuning the cutoff to one seed.
- **Alternatives Weighed**: `Tighten the rejection threshold immediately` (rejected: If the bias error persists with the true synthetic inliers, stronger rejection may throw away useful tilt coverage without addressing the cause.)

```bash
cat > /tmp/mag_diagnose.py <<'PY'
import numpy as np
from mag_validate import m, synthetic, errors, report

fit_current = m._fit

def prepared(s):
    raw = s['calib_m_raw']
    origin = np.median(raw, axis=0)
    scale = np.median(np.linalg.norm(raw-origin, axis=1))
    return (raw-origin)/scale, m._level_axes(s['calib_roll_pitch_rad']), origin, scale

def refit(p, x, axes, cutoff):
    for _ in range(4):
        distance = np.linalg.norm(m._residual(p,x,axes),axis=1)
        sigma = max(np.median(distance)/np.sqrt(2*np.log(2)),1e-6)
        keep = distance <= cutoff*sigma
        q = m.least_squares(lambda q: m._residual(q,x[keep],axes[keep]).ravel(),p,max_nfev=25).x
        if np.linalg.norm(q-p)<1e-7:
            return q
        p = q
    return p

def clean_mask(seed,n):
    rng = np.random.default_rng(seed)
    rng.normal(size=(3,3)); rng.uniform(-65,65,size=3)
    rng.uniform(-1,1,size=(n,2))
    yaw = rng.uniform(-np.pi,np.pi,size=n)
    keep = np.ones(n,dtype=bool)
    keep[np.argsort(yaw)[-round(.04*n):]] = False
    return keep

def exact_inliers(p,x,axes,keep):
    return m.least_squares(lambda q: m._residual(q,x[keep],axes[keep]).ravel(),p,max_nfev=25)

for lo, hi in [(44,108),(108,172)]:
    outcomes = {k:[] for k in ('current','trim 3.5','trim 4.5','trim 5.5','oracle inliers')}
    for i in range(lo,hi):
        s,g = synthetic(i,n=180,tilt=.5,h=12,soft=.40,spike='cluster')
        x,axes,origin,scale = prepared(s)
        p = fit_current(x,axes)
        clean = clean_mask(i,len(x))
        oracle = exact_inliers(p,x,axes,clean)
        if i == 58:
            d = np.linalg.norm(m._residual(p,x,axes),axis=1)
            keep = d <= 4*np.median(d)/np.sqrt(2*np.log(2))
            _, singular, directions = np.linalg.svd(oracle.jac,full_matrices=False)
            error = origin+scale*oracle.x[5:8]-g['b_hard']
            mode = directions[-1,5:8]
            print('seed 58 discarded clean, retained spikes, condition, bias alignment with weakest bias direction:',(~keep&clean).sum(),(keep&~clean).sum(),singular[0]/singular[-1], abs(error@mode)/(np.linalg.norm(error)*np.linalg.norm(mode)))
        fits = [p, *(refit(p,x,axes,c) for c in (3.5,4.5,5.5)),oracle.x]
        for k,q in zip(outcomes,fits):
            m._fit = lambda x,axes: q
            outcomes[k].append(errors(m.calibrate_and_estimate_heading(s),g))
        m._fit = fit_current
    for key, values in outcomes.items():
        report(f'{lo}-{hi-1} {key}',values)
        print('bias failures:',(np.flatnonzero(np.asarray(values)[:,1]>.35)+lo).tolist())
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/mag_diagnose.py
```
- **Container Exit Code**: `0` (`6030 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
seed 58 discarded clean, retained spikes, condition, bias alignment with weakest bias direction: 0 0 108.88820395666322 0.993511103604952
44-107 current              n=64 median=[0.33818 0.10736] max=[0.40246 0.37003]
bias failures: [58]
44-107 trim 3.5             n=64 median=[0.33818 0.10736] max=[0.40246 0.37003]
bias failures: [58]
44-107 trim 4.5             n=64 median=[0.33818 0.10628] max=[0.40246 0.37003]
bias failures: [58]
44-107 trim 5.5             n=64 median=[0.33818 0.10628] max=[0.40246 0.37003]
bias failures: [58]
44-107 oracle inliers       n=64 median=[0.33818 0.10628] max=[0.40246 0.37003]
bias failures: [58]
108-171 current             n=64 median=[0.34214 0.09786] max=[0.44341 0.34043]
bias failures: []
108-171 trim 3.5            n=64 median=[0.34214 0.10151] max=[0.44341 0.34043]
bias failures: []
108-171 trim 4.5            n=64 median=[0.34214 0.09543] max=[0.44341 0.34043]
bias failures: []
108-171 trim 5.5            n=64 median=[0.34214 0.09543] max=[0.44341 0.34043]
bias failures: []
108-171 oracle inliers      n=64 median=[0.34214 0.09543] max=[0.44341 0.34043]
bias failures: []
```

### Turn 8 — `execute_in_remote_docker` (`39084 ms` LLM)
- **Rationale**: Exact projection checks whether the local distance approximation contributes to the narrow bias margin. Compare both estimators on the same public folds and stress seeds; retain the simpler fit if exact projection does not materially improve the bottleneck.
- **Alternatives Weighed**: `Tune the spike cutoff again` (rejected: Cutoffs from 3.5 to 5.5 noise scales and oracle labels all left the same failing bias error.); `Shrink the weak direction toward nominal soft-iron values` (rejected: The task supplies no nominal matrix beyond symmetry, positive definiteness, and unit determinant; shrinkage could trade random error for systematic bias.)
- **ATMS NOGOOD DECLARED**: Refuted `Imperfect spike rejection caused the bias-target miss at stress seed 58.` | Witness: `The fit retained no spikes and discarded no clean observations. Oracle inlier selection left the bias error at 0.37003 µT; its absolute cosine with the weakest fitted bias direction was 0.9935.` | Blanket Diagnosis: `Under reduced tilt coverage, a weakly constrained combination of bias, field, and soft-iron parameters amplifies ordinary inlier noise.` | Sublated Pivot: `Keep joint spike rejection and compare exact raw-space projection against locally whitened residuals before attributing the remaining error to calibration coverage.`

```bash
cat > /tmp/mag_exact_compare.py <<'PY'
import numpy as np
from mag_validate import m, s, g, errors, report, subset, synthetic, CASES

baseline = m._fit

def exact_residual(p, x, axes):
    # Project onto the raw-space ellipse at each known tilt. The two
    # horizontal basis vectors need not be orthogonal in raw coordinates.
    w = m._correction(p)
    soft = np.linalg.inv(w)
    level = np.einsum('nij,nj->ni', axes, (x - p[5:8]) @ w)
    yaw = np.arctan2(level[:, 1], level[:, 0])
    u, v = (p[8] * (axes[:, :2] @ soft)).transpose(1, 0, 2)
    offset = p[5:8] + p[9] * (axes[:, 2] @ soft) - x
    for _ in range(5):
        c, t = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        radial = c*u + t*v
        tangent = -t*u + c*v
        residual = offset + radial
        slope = np.einsum('ij,ij->i', residual, tangent)
        curvature = np.einsum('ij,ij->i', tangent, tangent) - np.einsum('ij,ij->i', residual, radial)
        yaw -= np.clip(slope / np.maximum(curvature, 1e-12), -.25, .25)
    return offset + np.cos(yaw)[:, None]*u + np.sin(yaw)[:, None]*v

def exact_fit(x, axes):
    p = baseline(x, axes)
    distance = np.linalg.norm(m._residual(p, x, axes), axis=1)
    keep = distance <= 4 * np.median(distance) / np.sqrt(2*np.log(2))
    return m.least_squares(lambda q: exact_residual(q, x[keep], axes[keep]).ravel(), p, max_nfev=25).x

rp = s['calib_roll_pitch_rad']
axes = m._level_axes(rp)
level = np.einsum('nij,nj->ni', axes, (s['calib_m_raw']-g['b_hard']) @ np.linalg.inv(g['S_soft']))
sector = np.floor((np.arctan2(-level[:,1],level[:,0])+np.pi)*8/(2*np.pi)).astype(int)%8
quadrant = (rp[:,0]>0).astype(int) + 2*(rp[:,1]>0)
public = [(s,g)] + [(subset(k!=i),g) for k in (sector,quadrant,np.arange(len(rp))//70) for i in np.unique(k)]
cases = [('public + jackknife', public)]
for name, kw in CASES:
    ids = range(44,172) if name == 'combined stress' else range(20)
    cases.append((name, [synthetic(i, **kw) for i in ids]))
for name, pairs in cases:
    for label, fit in [('whitened', baseline), ('exact', exact_fit)]:
        m._fit = fit
        a = np.array([errors(m.calibrate_and_estimate_heading(sess), truth) for sess, truth in pairs])
        report(f'{name} {label}', a)
        print('target exceedances (heading, bias):', (a > [.45,.35]).sum(axis=0), flush=True)
m._fit = baseline
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/mag_exact_compare.py
```
- **Container Exit Code**: `0` (`21222 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public + jackknife whitened n=19 median=[0.13302 0.01894] max=[0.13568 0.04193]
target exceedances (heading, bias): [0 0]
public + jackknife exact    n=19 median=[0.13302 0.01893] max=[0.13568 0.04193]
target exceedances (heading, bias): [0 0]
nominal whitened            n=20 median=[0.15731 0.02222] max=[0.17582 0.03742]
target exceedances (heading, bias): [0 0]
nominal exact               n=20 median=[0.15731 0.0222 ] max=[0.17582 0.0374 ]
target exceedances (heading, bias): [0 0]
coherent spikes whitened    n=20 median=[0.1552  0.02222] max=[0.17657 0.03742]
target exceedances (heading, bias): [0 0]
coherent spikes exact       n=20 median=[0.1552 0.0222] max=[0.17657 0.0374 ]
target exceedances (heading, bias): [0 0]
localized spike episode whitened n=20 median=[0.15137 0.02113] max=[0.17102 0.04075]
target exceedances (heading, bias): [0 0]
localized spike episode exact n=20 median=[0.15136 0.02113] max=[0.17102 0.04073]
target exceedances (heading, bias): [0 0]
reduced samples and tilt whitened n=20 median=[0.15866 0.05651] max=[0.19406 0.20602]
target exceedances (heading, bias): [0 0]
reduced samples and tilt exact n=20 median=[0.15866 0.05652] max=[0.19406 0.20599]
target exceedances (heading, bias): [0 0]
steep field and strong iron whitened n=20 median=[0.34209 0.0268 ] max=[0.38783 0.0563 ]
target exceedances (heading, bias): [0 0]
steep field and strong iron exact n=20 median=[0.34209 0.02688] max=[0.38779 0.05659]
target exceedances (heading, bias): [0 0]
combined stress whitened    n=128 median=[0.34128 0.10393] max=[0.44341 0.37003]
target exceedances (heading, bias): [0 1]
combined stress exact       n=128 median=[0.34128 0.10407] max=[0.44342 0.37157]
target exceedances (heading, bias): [0 1]
```

### Turn 9 — `execute_in_remote_docker` (`35779 ms` LLM)
- **Rationale**: Verify the archived candidate in a clean process. The public test checks the API and target errors; an independently generated case checks the rotation convention, negative field inclination, and angle wrapping. A failure would identify a contract or convention error to fix before submission.
- **Alternatives Weighed**: `Replace local whitening with exact projection` (rejected: Exact projection matched public and typical stress performance but slightly increased the worst combined-stress bias error, from 0.370 to 0.372 µT.)

```bash
set -eu
f=/holdout_task/mag_calibrator.py
cmp "$f" /tmp/.hegel_candidate_archive/v02_fb89e88f
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
cp "$f" "$d/"
cd "$d"
python3 -I -B - <<'PY'
import importlib.util
import pickle
import time
import numpy as np
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location('mag_calibrator', './mag_calibrator.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def check(session, truth, heading_limit, bias_limit):
    start = time.perf_counter()
    result = mod.calibrate_and_estimate_heading(session)
    shape, bias, head = (result[k] for k in ('S_soft', 'b_hard', 'heading_deg'))
    assert (shape.shape, bias.shape, head.shape) == ((3, 3), (3,), truth['gt_heading_deg'].shape)
    assert np.isfinite(shape).all() and np.isfinite(bias).all() and np.isfinite(head).all()
    assert np.allclose(shape, shape.T, atol=1e-12, rtol=0)
    assert np.linalg.eigvalsh(shape).min() > 0 and abs(np.linalg.det(shape)-1) < 1e-12
    assert np.all((head > -180) & (head <= 180))
    e = (head - truth['gt_heading_deg'] + 180) % 360 - 180
    heading, bias_error = np.sqrt(np.mean(e**2)), np.linalg.norm(bias - truth['b_hard'])
    assert heading <= heading_limit and bias_error <= bias_limit, (heading, bias_error)
    print(f'heading={heading:.8f} deg bias={bias_error:.8f} uT soft_error={np.linalg.norm(shape-truth["S_soft"],2):.8f} runtime={time.perf_counter()-start:.3f}s')
    return e, shape

with open('/holdout_task/public_mag.pkl', 'rb') as f:
    public = pickle.load(f)
e, _ = check(public['session'], public['gt'], .45, .35)
print('public worst yaw-bin RMSE:', max(np.sqrt(np.mean(b*b)) for b in np.array_split(e[np.argsort(public['gt']['gt_heading_deg'])], 8)))

rng = np.random.default_rng(20261124)
q = Rotation.from_rotvec([.4, -.6, 1.]).as_matrix()
shape = (q * np.exp([-.45, .15, .30])) @ q.T
bias = np.array([37., -24., 12.])
field = [28., 0., -42.]
cal_angles = rng.uniform(-1, 1, (450, 3)) * [.62, .51, np.pi]
test_angles = np.column_stack((rng.uniform(-.35, .35, (7, 2)), np.deg2rad([-180., -179.999, -90., 0., 90., 179.999, 180.])))

def raw(angles):
    return Rotation.from_euler('ZYX', angles[:, ::-1]).inv().apply(field) @ shape.T + bias

cal = raw(cal_angles)
cal[::25] += [17, -31, 25]
session = dict(calib_m_raw=cal, calib_roll_pitch_rad=cal_angles[:, :2], test_m_raw=raw(test_angles), test_roll_pitch_rad=test_angles[:, :2])
_, estimate = check(session, dict(S_soft=shape, b_hard=bias, gt_heading_deg=np.rad2deg(test_angles[:, 2])), 1e-5, 1e-5)
assert np.linalg.norm(estimate-shape,2) < 1e-6
PY
```
- **Container Exit Code**: `2` (`279 ms`, timed_out=`False`)
- **Stderr (tail)**:
```text
cmp: /tmp/.hegel_candidate_archive/v02_fb89e88f: Is a directory
```

### Turn 10 — `execute_in_remote_docker` (`32703 ms` LLM)
- **Rationale**: Use the recorded candidate digest instead of assuming an archive layout. If the digest differs, stop before testing; if the isolated checks fail, fix the identified dependency, output contract, or rotation convention before submission.
- **Alternatives Weighed**: `Submit using the earlier validation results` (rejected: The archive path was a directory, so the previous command stopped before the isolated API and rotation-convention tests ran.)

```bash
set -eu
f=/holdout_task/mag_calibrator.py
printf '%s  %s\n' fb89e88f6770a20953d926ef2f28f715b8955ba43d8b7b39f2bd9c77cc1e7bee "$f" | sha256sum -c -
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
cp "$f" "$d/"
cd "$d"
OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location('mag_calibrator', './mag_calibrator.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def check(name, session, truth, heading_limit, bias_limit):
    result = m.calibrate_and_estimate_heading(session)
    shape, bias, head = (result[k] for k in ('S_soft', 'b_hard', 'heading_deg'))
    assert (shape.shape, bias.shape, head.shape) == ((3, 3), (3,), truth['gt_heading_deg'].shape)
    assert all(np.isfinite(a).all() for a in (shape, bias, head))
    assert np.allclose(shape, shape.T, atol=1e-12, rtol=0)
    assert np.linalg.eigvalsh(shape).min() > 0 and abs(np.linalg.det(shape)-1) < 1e-12
    assert np.all((head > -180) & (head <= 180))
    err = (head - truth['gt_heading_deg'] + 180) % 360 - 180
    rmse = np.sqrt(np.mean(err**2))
    bias_error = np.linalg.norm(bias - truth['b_hard'])
    soft_error = np.linalg.norm(shape - truth['S_soft'], 2)
    print(f'{name}: heading={rmse:.8f} deg bias={bias_error:.8f} uT soft_spectral_error={soft_error:.8f}', flush=True)
    assert rmse <= heading_limit and bias_error <= bias_limit, (rmse, bias_error)
    return err, soft_error

with open('/holdout_task/public_mag.pkl', 'rb') as f:
    public = pickle.load(f)
err, _ = check('public', public['session'], public['gt'], .45, .35)
order = np.argsort(public['gt']['gt_heading_deg'])
print('public worst yaw-bin RMSE:', max(np.sqrt(np.mean(b*b)) for b in np.array_split(err[order], 8)))

rng = np.random.default_rng(20261124)
q = Rotation.from_rotvec([.4, -.6, 1.]).as_matrix()
shape = (q * np.exp([-.45, .15, .30])) @ q.T
bias = np.array([37., -24., 12.])
cal = rng.uniform(-1, 1, (450, 3)) * [.62, .51, np.pi]
test = np.column_stack((rng.uniform(-.35, .35, (7, 2)), np.deg2rad([-180., -179.999, -90., 0., 90., 179.999, 180.])))

def raw(angles):
    return Rotation.from_euler('ZYX', angles[:, ::-1]).inv().apply([28., 0., -42.]) @ shape.T + bias

cal_raw = raw(cal)
cal_raw[::25] += [17, -31, 25]
session = dict(calib_m_raw=cal_raw, calib_roll_pitch_rad=cal[:, :2], test_m_raw=raw(test), test_roll_pitch_rad=test[:, :2])
_, soft_error = check('negative inclination and heading wrap', session, dict(S_soft=shape, b_hard=bias, gt_heading_deg=np.rad2deg(test[:, 2])), 1e-5, 1e-5)
assert soft_error < 1e-6
PY
```
- **Container Exit Code**: `1` (`308 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/holdout_task/mag_calibrator.py: FAILED
```
- **Stderr (tail)**:
```text
sha256sum: WARNING: 1 computed checksum did NOT match
```

### Turn 11 — `execute_in_remote_docker` (`38781 ms` LLM)
- **Rationale**: The previous verification stopped before Python ran. A byte difference against an available archive copy calls for investigating candidate drift; failure in the isolated test calls for fixing the output contract, leveling convention, or wrapping before submitting.
- **Alternatives Weighed**: `Retry the supplied full digest as a gate` (rejected: It already failed. Read the live digest and compare actual files where the archive layout permits, then test an exact copy of the live target.)

```bash
set -eu
f=/holdout_task/mag_calibrator.py
archive=/tmp/.hegel_candidate_archive/v02_fb89e88f
sha256sum "$f"
if [ -d "$archive" ]; then
    find "$archive" -maxdepth 2 -type f -name '*.py' -exec sha256sum {} +
    for saved in "$archive/mag_calibrator.py" "$archive/holdout_task/mag_calibrator.py"; do
        if [ -f "$saved" ]; then cmp "$f" "$saved"; break; fi
    done
elif [ -f "$archive" ]; then
    cmp "$f" "$archive"
fi
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
cp "$f" "$d/mag_calibrator.py"
cmp "$f" "$d/mag_calibrator.py"
cd "$d"
OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
spec = importlib.util.spec_from_file_location('mag_calibrator', './mag_calibrator.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def check(name, session, truth, limits):
    result = m.calibrate_and_estimate_heading(session)
    shape, bias, yaw = (result[k] for k in ('S_soft', 'b_hard', 'heading_deg'))
    assert (shape.shape, bias.shape, yaw.shape) == ((3,3), (3,), truth['gt_heading_deg'].shape)
    assert all(np.isfinite(a).all() for a in (shape, bias, yaw))
    assert np.allclose(shape, shape.T, rtol=0, atol=1e-12)
    assert np.linalg.eigvalsh(shape).min() > 0 and abs(np.linalg.det(shape)-1) < 1e-12
    assert np.all((-180 < yaw) & (yaw <= 180))
    e = (yaw - truth['gt_heading_deg'] + 180) % 360 - 180
    rmse = np.mean(e**2)**.5
    bias_error = np.linalg.norm(bias - truth['b_hard'])
    soft_error = np.linalg.norm(shape - truth['S_soft'], 2)
    print(f'{name}: heading={rmse:.8f} deg, bias={bias_error:.8f} uT, soft error={soft_error:.8f}', flush=True)
    assert rmse <= limits[0] and bias_error <= limits[1]
    return e, soft_error

with open('/holdout_task/public_mag.pkl', 'rb') as f:
    p = pickle.load(f)
e, _ = check('public', p['session'], p['gt'], (.45,.35))
for name, ranks, count in [('yaw', p['gt']['gt_heading_deg'], 8), ('tilt', np.linalg.norm(p['session']['test_roll_pitch_rad'],axis=1), 4)]:
    worst = max(np.mean(b**2)**.5 for b in np.array_split(e[np.argsort(ranks)], count))
    print(f'public worst {name}-bin RMSE: {worst:.8f} deg')
    assert worst <= .45

rng = np.random.default_rng(20261124)
q = Rotation.from_rotvec([.4,-.6,1.]).as_matrix()
shape = (q*np.exp([-.45,.15,.30]))@q.T
bias = np.array([37.,-24.,12.])
cal = rng.uniform(-1,1,(450,3))*[.62,.51,np.pi]
test = np.column_stack((rng.uniform(-.35,.35,(7,2)),np.deg2rad([-180,-179.999,-90,0,90,179.999,180])))
def raw(angles):
    return Rotation.from_euler('ZYX',angles[:,::-1]).inv().apply([28.,0.,-42.]) @ shape.T + bias
cal_raw = raw(cal)
cal_raw[::25] += [17,-31,25]
session = dict(calib_m_raw=cal_raw,calib_roll_pitch_rad=cal[:,:2],test_m_raw=raw(test),test_roll_pitch_rad=test[:,:2])
_, soft_error = check('negative inclination and heading wrap', session, dict(S_soft=shape,b_hard=bias,gt_heading_deg=np.rad2deg(test[:,2])), (1e-5,1e-5))
assert soft_error < 1e-6
PY
```
- **Container Exit Code**: `0` (`626 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
62b3431cda44602fae1ccde464248766d6d460192837ca935b5ad5f96feb7608  /holdout_task/mag_calibrator.py
62b3431cda44602fae1ccde464248766d6d460192837ca935b5ad5f96feb7608  /tmp/.hegel_candidate_archive/v02_fb89e88f/mag_calibrator.py
62b3431cda44602fae1ccde464248766d6d460192837ca935b5ad5f96feb7608  /tmp/.hegel_candidate_archive/v02_fb89e88f/slot_00__mag_calibrator.py
public: heading=0.13290049 deg, bias=0.01680750 uT, soft error=0.00071440
public worst yaw-bin RMSE: 0.15683001 deg
public worst tilt-bin RMSE: 0.13862474 deg
negative inclination and heading wrap: heading=0.00000000 deg, bias=0.00000000 uT, soft error=0.00000000
```

### Turn 12 — `submit_completed_task` (`16235 ms` LLM)
- **Rationale**: A clean-process test of the byte-identical archived artifact verified the API, determinant-one SPD output, public error targets, and heading wrap. Oracle inlier selection did not resolve the synthetic bias miss, so further cutoff tuning is not warranted.
- **Alternatives Weighed**: `Replace local whitening with exact ellipse projection` (rejected: It did not improve public validation or resolve the low-tilt bias miss.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`16` turns remaining; artifact status={'/holdout_task/mag_calibrator.py': 'FILE:4852'}): Do NOT lock prematurely with `16` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/mag_calibrator.py`. The public session passed both targets: 0.133° circular heading RMSE and 0.0168 µT hard-iron bias error. In combined synthetic stress testing, all 128 sessions met the heading target; one low-tilt session exceeded the bias target at 0.370 µT, even with only known-clean calibration samples.`

### Turn 13 — `execute_in_remote_docker` (`37949 ms` LLM)
- **Rationale**: An identity failure at the observed orientations would justify correcting the calibration algebra. A coherent residual shift across ordered blocks would motivate a drift comparison. If removing high-tilt rows worsens conditioning without evidence of drift, calibration coverage is the relevant limitation; relay modes and classification metrics do not apply to this session.
- **Alternatives Weighed**: `Add rational terms or fit separate offsets by row block` (rejected: The stated sensor model is affine with a complete symmetric quadratic invariant, and row blocks are not documented batches. Check for residual shifts before considering a nuisance model.)

```bash
set -eu
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/holdout_task python3 -B - <<'PY'
import pickle
import numpy as np
from scipy.optimize import least_squares
import mag_calibrator as m

with open('/holdout_task/public_mag.pkl', 'rb') as f:
    data = pickle.load(f)
s, gt = data['session'], data['gt']
print('session keys:', sorted(s))
raw, rp = s['calib_m_raw'], s['calib_roll_pitch_rad']
axes = m._level_axes(rp)
center = np.median(raw, axis=0)
scale = np.median(np.linalg.norm(raw - center, axis=1))
x = (raw - center) / scale
p = m._fit(x, axes)
w = m._correction(p)
assert np.allclose(w, w.T, atol=1e-12) and np.linalg.eigvalsh(w).min() > 0
assert abs(np.linalg.det(w) - 1) < 1e-12
# Exercise the full quadratic, including all cross terms, and verify the
# equivalence of the magnitude/vertical and horizontal/vertical constraints
# at the PUBLIC roll/pitch orientations.
yaw = np.linspace(-np.pi, np.pi, len(raw), endpoint=False)
ideal_level = np.column_stack((p[8]*np.cos(yaw), -p[8]*np.sin(yaw), np.full(len(raw), p[9])))
ideal_x = np.einsum('nji,nj->ni', axes, ideal_level) @ np.linalg.inv(w) + p[5:8]
z = ideal_x - p[5:8]
q = np.einsum('ni,ij,nj->n', z, w @ w, z)
np.testing.assert_allclose(q, p[8]**2+p[9]**2, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(m._residual(p, ideal_x, axes), 0, rtol=0, atol=1e-12)
print('quadratic and leveled-circle identities agree at all public orientations')

res = m._residual(p, x, axes)
d = np.linalg.norm(res, axis=1)
sigma = np.median(d) / np.sqrt(2*np.log(2))
keep = d <= 4*sigma
print('screened public rows:', int((~keep).sum()), 'of', len(x))
for idx, block in enumerate(np.array_split(np.arange(len(x)), 6)):
    clean = block[keep[block]]
    print('ordered block', idx, 'screened', int((~keep[block]).sum()), 'median residual uT', np.round(scale*np.median(res[clean],axis=0), 4).tolist())

# Remove the highest-tilt observations from real calibration data while keeping
# the test set fixed. Report conditioning separately from actual target errors.
tilt = np.linalg.norm(rp, axis=1)
for fraction in (1.0, .75, .5):
    ids = np.argsort(tilt)[:round(len(tilt)*fraction)]
    sub = dict(s, calib_m_raw=raw[ids], calib_roll_pitch_rad=rp[ids])
    result = m.calibrate_and_estimate_heading(sub)
    e = (result['heading_deg'] - gt['gt_heading_deg'] + 180) % 360 - 180
    use = ids[keep[ids]]
    fit = least_squares(lambda a: m._residual(a, x[use], axes[use]).ravel(), p, max_nfev=20)
    sv = np.linalg.svd(fit.jac, compute_uv=False)
    print('retained', fraction, 'n', len(ids), 'condition', round(sv[0]/sv[-1], 2), 'heading RMSE deg', round(np.sqrt(np.mean(e**2)), 6), 'bias error uT', round(np.linalg.norm(result['b_hard']-gt['b_hard']), 6))
PY
```
- **Container Exit Code**: `0` (`725 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
session keys: ['calib_m_raw', 'calib_roll_pitch_rad', 'test_m_raw', 'test_roll_pitch_rad']
quadratic and leveled-circle identities agree at all public orientations
screened public rows: 16 of 420
ordered block 0 screened 1 median residual uT [0.0039, -0.0083]
ordered block 1 screened 4 median residual uT [-0.0003, 0.0059]
ordered block 2 screened 3 median residual uT [-0.0232, 0.0097]
ordered block 3 screened 4 median residual uT [-0.0021, -0.0094]
ordered block 4 screened 3 median residual uT [-0.0103, 0.001]
ordered block 5 screened 1 median residual uT [0.0052, -0.0163]
retained 1.0 n 420 condition 10.53 heading RMSE deg 0.1329 bias error uT 0.016808
retained 0.75 n 315 condition 12.75 heading RMSE deg 0.134438 bias error uT 0.01554
retained 0.5 n 210 condition 15.71 heading RMSE deg 0.135366 bias error uT 0.043817
```

### Turn 14 — `submit_completed_task` (`16573 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated checks for the API, determinant-one symmetric positive-definite output, rotation convention, and heading wrap. Removing higher-tilt public observations worsened conditioning but left both errors within target. Exact ellipse projection and oracle spike rejection did not resolve the synthetic bias miss.
- **Alternatives Weighed**: `Add block-specific offsets or extra calibration terms` (rejected: The public session has no batch labels, its ordered residual blocks show no consistent shift, and the full quadratic and leveled-circle constraints agree at all observed orientations.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/mag_calibrator.py': 'FILE:4852'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/mag_calibrator.py`. On the public session, circular heading RMSE was 0.133° and hard-iron bias error was 0.0168 µT, both below target. All 128 combined synthetic stress sessions met the heading target. One low-tilt session exceeded the bias target (0.370 µT versus 0.35 µT), even when fitted using only known-clean samples.`

### Turn 15 — `execute_in_remote_docker` (`26987 ms` LLM)
- **Rationale**: Compare broad and narrow tilt coverage on the same public measurements at equal sample counts, with one calibration yaw sector omitted per fold. If the joint fit loses bias margin relative to the vertical-only fit, investigate the horizontal residual; if both worsen specifically under narrow tilt, retain the physical model and report the coverage limitation.
- **Alternatives Weighed**: `Add a block-specific correction` (rejected: The session contains no batch labels, and the ordered residuals showed no consistent shift to correct.); `Repeat the same low-tilt truncation` (rejected: Truncation changes sample count and orientation coverage together; matched-count folds can separate those effects.)

```bash
cat > /tmp/mag_public_coverage.py <<'PY'
import numpy as np
from mag_validate import m, s, g, errors, report, subset

raw = s['calib_m_raw']
rp = s['calib_roll_pitch_rad']
axes = m._level_axes(rp)
level = np.einsum('nij,nj->ni', axes, (raw-g['b_hard']) @ np.linalg.inv(g['S_soft']))
yaw = np.arctan2(-level[:,1],level[:,0])
sector = np.floor((yaw+np.pi)*8/(2*np.pi)).astype(int)%8
tilt = np.linalg.norm(rp, axis=1)
joint_fit = m._fit

# Match sample counts before comparing low tilt with broad tilt. Omit one
# yaw sector from both arms so dense coverage cannot mask a weak direction.
for fraction in (.50,.25):
    groups = {k: [] for k in ('low joint','broad joint','low vertical','broad vertical')}
    for hold in range(8):
        ids = np.flatnonzero(sector != hold)
        ranked = ids[np.argsort(tilt[ids])]
        n = round(fraction * len(ids))
        picks = {'low': ranked[:n], 'broad': ranked[((np.arange(n)+.5)*len(ids)/n).astype(int)]}
        for arm, keep in picks.items():
            fold = subset(keep)
            for method, fit in [('joint',joint_fit),('vertical',m._seed)]:
                m._fit = fit
                groups[f'{arm} {method}'].append(errors(m.calibrate_and_estimate_heading(fold), g))
    m._fit = joint_fit
    for name, values in groups.items():
        report(f'{fraction:.0%} {name}', values)
PY
OPENBLAS_NUM_THREADS=1 python3 -B /tmp/mag_public_coverage.py
```
- **Container Exit Code**: `0` (`1543 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
50% low joint               n= 8 median=[0.13658 0.05146] max=[0.14139 0.0616 ]
50% broad joint             n= 8 median=[0.13364 0.02591] max=[0.13821 0.04256]
50% low vertical            n= 8 median=[0.17949 0.08217] max=[0.21842 0.15184]
50% broad vertical          n= 8 median=[0.1469  0.06765] max=[0.18754 0.20831]
25% low joint               n= 8 median=[0.13219 0.08379] max=[0.14454 0.14413]
25% broad joint             n= 8 median=[0.13879 0.0329 ] max=[0.14506 0.11025]
25% low vertical            n= 8 median=[0.23563 0.1678 ] max=[0.34035 0.30992]
25% broad vertical          n= 8 median=[0.17642 0.08907] max=[0.2214  0.18297]
```

### Turn 16 — `submit_completed_task` (`22813 ms` LLM)
- **Rationale**: The matched-count public audit supports retaining the joint fit: broader tilt improves bias estimation without a meaningful heading penalty. The unchanged artifact passed isolated API, determinant, and angle-wrap checks. Exact projection and known-clean inlier selection did not resolve the one synthetic bias miss.
- **Alternatives Weighed**: `Replace the joint fit with the vertical-only fit` (rejected: In the public 25%-sample, low-tilt folds, vertical-only fitting raised worst bias error from 0.144 to 0.310 µT and worst heading RMSE from 0.145° to 0.340°.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/mag_calibrator.py`. The public session met both targets: 0.133° circular heading RMSE and 0.0168 µT hard-iron bias error. All 128 combined synthetic stress sessions met the heading target; one low-tilt session missed the bias target at 0.370 µT, even after all spike samples were removed.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-OOD-Holdout-12",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
