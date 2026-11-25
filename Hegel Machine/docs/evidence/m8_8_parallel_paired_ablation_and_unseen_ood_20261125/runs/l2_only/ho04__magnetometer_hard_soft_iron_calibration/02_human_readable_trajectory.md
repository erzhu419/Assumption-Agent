# Dialectical Synthesis Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_2_Selective_LiveGPT` | `ho04__magnetometer_hard_soft_iron_calibration`

- **Official Strict Pass**: **`PASS (1.0)`** (`PASSED_ALL_CHECKS`)
- **Blind Public Synthesis Choice**: `antithesis` via `selective_router_confident_single_branch`
- **Synthesis Rationale**: Primary Antithesis branch satisfied public checks under policy; second branch skipped (`1.00u` nominal units).
- **Post-Lock Branch Verdicts**: Thesis (`Bare`) = `NOT_SPAWNED (0T)` | Antithesis (`Wormhole`) = `True` (13T)

## Selected Branch Trajectory Record
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
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `1` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `350,365` (`338,083` / `45,056` / `12,282`) |
| **Total Episode Wall Time (s)** | `432.08s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`root_A03_ill_posed_regularization_and_certificates`**: AXIOM A03 (HEAVY-TAILED REGULARIZATION & DUAL CERTIFICATES): Under outliers, stage drift, or ill-posed combinatorial landscapes, replace naive L2 least squares with heavy-tailed robust losses (Huber/Cauchy/MAD) or exact variable projection, and verify optimality/safety margins via dual bounds or worst-corner simulation.
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`19547 ms` LLM)
- **Rationale**: Inspect compact summaries to establish the input schema, angle and sample ranges, and whether public truth is available for independent heading and bias validation.
- **Alternatives Weighed**: `Fit the raw ellipsoid with ordinary least squares` (rejected: High-leverage current spikes can distort the fit, and a norm-only objective discards the known roll and pitch.)

```bash
ls -la /holdout_task; python - <<'PY'
import pickle
import numpy as np
from scipy import __version__ as scipy_version
with open('/holdout_task/public_mag.pkl', 'rb') as f:
    data = pickle.load(f)
print('SciPy', scipy_version)
def inspect(obj, path='public', depth=0):
    if isinstance(obj, dict) and depth < 4:
        for key, value in obj.items():
            inspect(value, f'{path}.{key}', depth + 1)
    elif isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict) and depth < 4:
        print(path, type(obj).__name__, len(obj))
        for i, value in enumerate(obj[:3]):
            inspect(value, f'{path}[{i}]', depth + 1)
    elif isinstance(obj, np.ndarray):
        print(path, obj.shape, obj.dtype, ('range', float(obj.min()), float(obj.max())) if obj.size and obj.dtype.kind in 'fi' else '')
        if obj.size <= 9:
            print(obj)
    else:
        print(path, type(obj).__name__, str(obj)[:160])
inspect(data)
PY
```
- **Container Exit Code**: `127` (`295 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 32
drwxr-xr-x 2 root root  4096 Nov 25 15:29 .
drwxr-xr-x 1 root root  4096 Nov 25 15:29 ..
-rw-r--r-- 1 root root 23144 Nov 25 15:29 public_mag.pkl
```
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`19004 ms` LLM)
- **Rationale**: The previous command stopped before opening the data because `python` was unavailable. Use `python3` to confirm the input schema, identify which calibration and heading metrics have public ground truth, and determine whether SciPy is available for the fit.

```bash
command -v python3; python3 - <<'PY'
import pickle
import numpy as np
with open('/holdout_task/public_mag.pkl', 'rb') as f:
    data = pickle.load(f)
def inspect(obj, path='public'):
    if isinstance(obj, dict):
        for key, value in obj.items():
            inspect(value, f'{path}.{key}')
    elif isinstance(obj, np.ndarray):
        print(path, obj.shape, obj.dtype)
        if obj.size <= 12:
            print(obj)
        elif obj.dtype.kind in 'fiu':
            print('  min/median/max:', np.min(obj, axis=0), np.median(obj, axis=0), np.max(obj, axis=0))
    elif isinstance(obj, (list, tuple)):
        print(path, type(obj).__name__, len(obj))
        for i, value in enumerate(obj[:2]):
            inspect(value, f'{path}[{i}]')
    else:
        print(path, type(obj).__name__, repr(obj)[:200])
inspect(data)
try:
    import scipy
    print('SciPy:', scipy.__version__)
except ImportError:
    print('SciPy unavailable')
PY
```
- **Container Exit Code**: `0` (`607 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/usr/bin/python3
public.session.calib_m_raw (420, 3) float64
  min/median/max: [-23.46471775 -80.90814953  12.65165156] [ 26.1947076  -39.86748974  50.55712531] [77.60462326  0.54958252 89.19001056]
public.session.calib_roll_pitch_rad (420, 2) float64
  min/median/max: [-0.7497117  -0.64896356] [ 0.00169826 -0.03662319] [0.74126443 0.64488313]
public.session.test_m_raw (120, 3) float64
  min/median/max: [ -9.68416279 -72.4480562   42.66979817] [ 26.16210115 -36.51395367  52.57854876] [65.53405512 -9.48043629 61.16755746]
public.session.test_roll_pitch_rad (120, 2) float64
  min/median/max: [-0.399999   -0.34996619] [ 1.48458614e-02 -1.55257751e-16] [0.39895783 0.34996619]
public.gt.S_soft (3, 3) float64
[[ 1.17564433  0.06974161 -0.03985235]
 [ 0.06974161  0.91660405  0.04981544]
 [-0.03985235  0.04981544  0.93653023]]
public.gt.b_hard (3,) float64
[ 28.5 -41.   16.2]
public.gt.gt_heading_deg (120,) float64
  min/median/max: -174.27042204869176 0.0 174.27042204869176
SciPy: 1.13.0
```

### Turn 3 — `execute_in_remote_docker` (`25936 ms` LLM)
- **Rationale**: Use public truth only for diagnosis: verify the leveling sign convention and measure the noise floor and spike separation before choosing the residual and outlier cutoff.

```bash
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/holdout_task/public_mag.pkl','rb')); s=p['session']; g=p['gt']; A=np.linalg.inv(g['S_soft']); b=g['b_hard']
for split in ('calib','test'):
    x=(s[split+'_m_raw']-b)@A
    r,t=s[split+'_roll_pitch_rad'].T; cr,sr,ct,st=np.cos(r),np.sin(r),np.cos(t),np.sin(t)
    u=np.array([ct,st*sr,st*cr]).T; v=np.array([0*r,cr,-sr]).T; d=np.array([-st,ct*sr,ct*cr]).T
    h1,h2,z=[np.sum(x*j,axis=1) for j in (u,v,d)]; h=np.hypot(h1,h2)
    rr=np.column_stack((h-np.median(h),z-np.median(z))); err=np.linalg.norm(rr,axis=1)
    print(split,'H,D',np.median(h),np.median(z),'residual magnitude quantiles',np.quantile(err,[.25,.5,.75,.9,.94,.95,.96,.97,.99,1]),'inlier component RMS',np.std(rr[err<np.quantile(err,.94)],axis=0))
    if split=='test':
        e=(np.degrees(np.arctan2(-h2,h1))-g['gt_heading_deg']+180)%360-180
        print('oracle heading RMSE',np.sqrt(np.mean(e**2)), 'slice RMSE', [np.sqrt(np.mean(z**2)) for z in np.array_split(e,4)])
PY
```
- **Container Exit Code**: `0` (`405 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calib H,D 23.491583735052668 41.99447680512864 residual magnitude quantiles [ 0.06708326  0.10895839  0.14904871  0.19949635  0.23696348  0.25553282
  0.28236529 13.02482649 24.28308584 41.0771688 ] inlier component RMS [0.08035898 0.08458568]
test H,D 23.490659399027745 41.99837957574918 residual magnitude quantiles [0.04674242 0.06958449 0.10299875 0.13975218 0.1467346  0.14709667
 0.15118998 0.15566491 0.16818619 0.17549634] inlier component RMS [0.05636166 0.05486101]
oracle heading RMSE 0.13106710186904436 slice RMSE [0.12958371767721266, 0.14200136644126116, 0.13385403209795124, 0.11764825074045752]
```

### Turn 4 — `execute_in_remote_docker` (`64110 ms` LLM)
- **Rationale**: The public calibration residual jumps from 0.282 µT at the 96th percentile to 13.0 µT at the 97th. Fit with a robust loss before screening spikes, then measure bias and circular heading independently on the full data and five calibration folds.
- **Alternatives Weighed**: `Initialize with a norm-only ellipsoid fit` (rejected: The public samples cover a limited ellipsoid cap; the known down-axis projections supply a linear initialization without extrapolating the entire ellipsoid from its radius.)

```bash
set -e
cat > /holdout_task/mag_calibrator.py <<'PY'
"""Robust hard- and soft-iron calibration from tilt-tagged magnetometer data."""

import numpy as np
from scipy.optimize import least_squares


def _level_axes(roll_pitch):
    roll, pitch = np.asarray(roll_pitch, dtype=float).T
    sr, cr = np.sin(roll), np.cos(roll)
    st, ct = np.sin(pitch), np.cos(pitch)
    return (np.column_stack((ct, st * sr, st * cr)),
            np.column_stack((np.zeros_like(cr), cr, -sr)),
            np.column_stack((-st, ct * sr, ct * cr)))


def _initial_inverse(x, down):
    # A has trace 3 at this stage; the fitted intercept vector is A @ bias.
    i, j, k = (down * x).T
    dx, dy, dz = down.T
    xx, yy, zz = x.T
    design = np.column_stack((i-k, j-k, dx*yy+dy*xx, dx*zz+dz*xx,
                              dy*zz+dz*yy, -down, -np.ones(len(x))))
    target = -3 * k
    coef = np.linalg.lstsq(design, target, rcond=None)[0]
    rng = np.random.default_rng(241)
    best = np.median(np.abs(design @ coef - target))
    for _ in range(64):
        inds = rng.choice(len(x), min(15, len(x)), replace=False)
        trial, _, rank, _ = np.linalg.lstsq(design[inds], target[inds], rcond=None)
        if rank == 9:
            score = np.median(np.abs(design @ trial - target))
            if score < best:
                coef, best = trial, score
    for _ in range(5):
        error = design @ coef - target
        mad = np.median(np.abs(error - np.median(error)))
        weight = 1 / np.hypot(1, error / max(2 * mad, 1e-10))
        root = np.sqrt(weight)
        coef = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)[0]
    a, b, xy, xz, yz = coef[:5]
    inverse = np.array([[a, xy, xz], [xy, b, yz], [xz, yz, 3-a-b]])
    eig, vec = np.linalg.eigh(inverse)
    eig = np.maximum(eig, 0.05)
    inverse = (vec * eig) @ vec.T
    bias = np.linalg.solve(inverse, coef[5:8])
    inverse /= np.cbrt(np.linalg.det(inverse))
    return inverse, bias


def _inverse(parameters):
    a, b, u, v, w = parameters[:5]
    L = np.array([[np.exp(a), 0., 0.], [u, np.exp(b), 0.],
                  [v, w, np.exp(-a-b)]])
    return L @ L.T


def _fit(x, axes):
    u, v, d = axes
    inverse, bias = _initial_inverse(x, d)
    L = np.linalg.cholesky(inverse)
    y = (x-bias) @ inverse
    h = np.hypot(np.sum(y*u, axis=1), np.sum(y*v, axis=1))
    z = np.sum(y*d, axis=1)
    p = np.r_[np.log(L[0, 0]), np.log(L[1, 1]), L[1, 0], L[2, 0], L[2, 1],
              bias, np.median(h), np.median(z)]

    def residual(p, inds):
        a = _inverse(p)
        y = (x[inds]-p[5:8]) @ a
        uu, vv, dd = u[inds], v[inds], d[inds]
        hx = np.sum(y*uu, axis=1)
        hy = np.sum(y*vv, axis=1)
        h = np.hypot(hx, hy)
        # Whiten the two constraints under isotropic raw sensor noise.
        safe_h = np.maximum(h, 1e-12)
        dh = ((hx/safe_h)[:, None]*uu + (hy/safe_h)[:, None]*vv) @ a
        dz = dd @ a
        dh[h < 1e-12] = (uu @ a)[h < 1e-12]
        l11 = np.linalg.norm(dh, axis=1)
        l21 = np.sum(dh*dz, axis=1)/l11
        l22 = np.linalg.norm(dz-(l21/l11)[:, None]*dh, axis=1)
        rh = (h-p[8])/l11
        rz = (np.sum(y*dd, axis=1)-p[9]-l21*rh)/l22
        return np.column_stack((rh, rz))

    rows = np.arange(len(x))
    e = np.linalg.norm(residual(p, rows), axis=1)
    scale = max(np.median(e)/np.sqrt(2*np.log(2)), 1e-7)
    p = least_squares(lambda q: residual(q, rows).ravel(), p, loss='cauchy',
                      f_scale=2*scale, x_scale='jac', max_nfev=150).x
    for _ in range(2):
        e = np.linalg.norm(residual(p, rows), axis=1)
        sigma = np.median(e)/np.sqrt(2*np.log(2))
        keep = rows[e <= max(4.25*sigma, 1e-7)]
        p = least_squares(lambda q: residual(q, keep).ravel(), p,
                          x_scale='jac', max_nfev=100).x
    return _inverse(p), p[5:8], p[8:10]


def calibrate_and_estimate_heading(session: dict) -> dict:
    """Fit the calibration on the calibration sweep, then return magnetic yaw."""
    raw = np.asarray(session['calib_m_raw'], dtype=float)
    center = np.median(raw, axis=0)
    scale = np.median(np.linalg.norm(raw-center, axis=1))
    inverse, bias, field = _fit((raw-center)/scale,
                                _level_axes(session['calib_roll_pitch_rad']))
    bias = center + scale*bias
    y = (np.asarray(session['test_m_raw'], dtype=float)-bias) @ inverse
    u, v, _ = _level_axes(session['test_roll_pitch_rad'])
    headings = np.rad2deg(np.arctan2(-np.sum(y*v, axis=1), np.sum(y*u, axis=1)))
    headings[headings <= -180] = 180
    soft = np.linalg.solve(inverse, np.eye(3))
    return {'S_soft': (soft+soft.T)/2, 'b_hard': bias,
            'B_ned_uT': scale*np.array([field[0], 0., field[1]]),
            'heading_deg': headings}
PY
mkdir -p /tmp/.hegel_candidate_archive
f=/holdout_task/mag_calibrator.py; digest=$(sha256sum "$f" | cut -c1-8); cp "$f" "/tmp/.hegel_candidate_archive/v01_$digest"
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle, numpy as np
from mag_calibrator import calibrate_and_estimate_heading
p=pickle.load(open('public_mag.pkl','rb')); s=p['session']; g=p['gt']
for fold in [None,0,1,2,3,4]:
    ss=s.copy()
    if fold is not None:
        i=np.arange(len(s['calib_m_raw']))%5!=fold
        ss.update({k:s[k][i] for k in ('calib_m_raw','calib_roll_pitch_rad')})
    o=calibrate_and_estimate_heading(ss)
    err=(o['heading_deg']-g['gt_heading_deg']+180)%360-180
    rmse=lambda x: np.sqrt(np.mean(x*x))
    quarters=[rmse(err[(g['gt_heading_deg']>lo)&(g['gt_heading_deg']<=lo+90)]) for lo in (-180,-90,0,90)]
    print(fold, 'heading RMSE',rmse(err),'worst yaw sector',max(quarters),'bias L2',np.linalg.norm(o['b_hard']-g['b_hard']),'soft L2',np.linalg.norm(o['S_soft']-g['S_soft']),'det',np.linalg.det(o['S_soft']),'field',o['B_ned_uT'])
PY
```
- **Container Exit Code**: `0` (`669 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/mag_calibrator.py']`
- **Stdout (tail)**:
```text
None heading RMSE 0.13290048847434965 worst yaw sector 0.1469878509738223 bias L2 0.016807503317750022 soft L2 0.0008198476961452323 det 1.0000000000000004 field [23.49266578  0.         42.00740983]
0 heading RMSE 0.13225082604356495 worst yaw sector 0.14796165229347216 bias L2 0.019785678845263843 soft L2 0.0010049827545638426 det 1.0000000000000007 field [23.49132557  0.         42.00988094]
1 heading RMSE 0.1338848797583386 worst yaw sector 0.14771300462164053 bias L2 0.017560433352809206 soft L2 0.0008266383355858358 det 1.0000000000000002 field [23.49262127  0.         42.01172141]
2 heading RMSE 0.1344664869512674 worst yaw sector 0.14632887250485818 bias L2 0.01530332515565032 soft L2 0.000861110475403877 det 1.0000000000000002 field [23.49169601  0.         42.00225361]
3 heading RMSE 0.1347507029559078 worst yaw sector 0.15045933250623664 bias L2 0.018715064926489923 soft L2 0.0008397748259604574 det 1.0000000000000007 field [23.49745918  0.         42.01458065]
4 heading RMSE 0.1313142604677251 worst yaw sector 0.14524407631244995 bias L2 0.019004726312953465 soft L2 0.0007978014885494442 det 1.0000000000000007 field [23.49074144  0.         41.99864372]
```

### Turn 5 — `execute_in_remote_docker` (`38396 ms` LLM)
- **Rationale**: The public folds preserve most of the tilt coverage. Remove contiguous blocks and larger tilts, then vary field inclination, soft-iron anisotropy, and spike direction in synthetic sessions to see whether weak excitation or outlier rejection threatens either tolerance.

```bash
set -e
cat > /tmp/mag_stress.py <<'PY'
import pickle
import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, '/holdout_task')
from mag_calibrator import calibrate_and_estimate_heading


def evaluate(session, truth):
    out = calibrate_and_estimate_heading(session)
    e = (out['heading_deg'] - truth['gt_heading_deg'] + 180) % 360 - 180
    assert out['heading_deg'].shape == e.shape
    assert np.all((-180 < out['heading_deg']) & (out['heading_deg'] <= 180))
    S = out['S_soft']
    assert np.allclose(S, S.T) and min(np.linalg.eigvalsh(S)) > 0
    assert abs(np.linalg.det(S)-1) < 1e-10
    assert out['B_ned_uT'][0] > 0 and out['B_ned_uT'][1] == 0
    yaw = truth['gt_heading_deg']
    sectors = [e[(yaw >= lo) & (yaw < lo+90)] for lo in (-180,-90,0,90)]
    rmse = lambda a: np.sqrt(np.mean(a*a))
    return rmse(e), max(rmse(a) for a in sectors if len(a)), np.linalg.norm(out['b_hard']-truth['b_hard'])


def synthetic(seed, n, tilt, h, d, sigma, condition, narrow_spikes):
    rng = np.random.default_rng(seed)
    q = Rotation.random(random_state=rng).as_matrix()
    S = (q * np.array([1/np.sqrt(condition), 1., np.sqrt(condition)])) @ q.T
    b = rng.uniform(-80,80,3)
    field = np.array([h,0,d])
    def sample(count, limit, noise):
        r, p = rng.uniform(-limit,limit,(2,count))
        y = rng.uniform(-np.pi,np.pi,count)
        R = Rotation.from_euler('ZYX', np.column_stack((y,p,r))).as_matrix()
        raw = (field @ R) @ S + b + rng.normal(0,noise,(count,3))
        return raw, np.column_stack((r,p)), np.rad2deg(y)
    cal, cal_rp, _ = sample(n, tilt, sigma)
    ind = rng.choice(n, round(.04*n), replace=False)
    directions = rng.normal(size=(len(ind),3))
    if narrow_spikes:
        directions[:,1:] *= .04
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    cal[ind] += directions*rng.uniform(12,54,(len(ind),1))
    test, test_rp, yaw = sample(240, .45, .065)
    return dict(calib_m_raw=cal, calib_roll_pitch_rad=cal_rp, test_m_raw=test, test_roll_pitch_rad=test_rp), dict(b_hard=b, gt_heading_deg=yaw)

if __name__ == '__main__':
    data = pickle.load(open('/holdout_task/public_mag.pkl','rb'))
    s,g = data['session'],data['gt']
    keys = ('calib_m_raw','calib_roll_pitch_rad')
    x = np.arange(len(s[keys[0]]))
    print('columns: heading RMSE deg, worst yaw sector RMSE deg, bias L2 uT')
    for label, keep in [('full',x>=0)] + [(f'block-{i}',x//84!=i) for i in range(5)] + [('inner tilt',np.max(np.abs(s[keys[1]]),axis=1)<.44)]:
        z = dict(s, **{k:s[k][keep] for k in keys})
        print(label, int(keep.sum()), np.round(evaluate(z,g),5))
    configs = {'public-like':(420,.65,23.5,42,.09,1.35,False), 'narrow tilt':(250,.31,23,42,.12,1.8,False), 'small horizontal':(320,.49,16,49,.13,2.2,True), 'opposite inclination':(240,.45,24,-44,.12,2.1,True), 'near horizontal':(240,.42,44,4,.12,2.1,True)}
    for label, cfg in configs.items():
        a = np.array([evaluate(*synthetic(seed,*cfg)) for seed in range(12)])
        print(label, 'max', np.round(a.max(axis=0),5), 'worst seeds', a.argmax(axis=0), 'misses', np.sum(a[:,[0,2]]>[.45,.35],axis=0))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/mag_stress.py
```
- **Container Exit Code**: `0` (`2084 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
columns: heading RMSE deg, worst yaw sector RMSE deg, bias L2 uT
full 420 [0.1329  0.14699 0.01681]
block-0 336 [0.13188 0.14459 0.01531]
block-1 336 [0.13326 0.14483 0.01869]
block-2 336 [0.13393 0.1477  0.02128]
block-3 336 [0.13472 0.15401 0.03197]
block-4 336 [0.13274 0.14623 0.01689]
inner tilt 158 [0.13577 0.14618 0.04289]
public-like max [0.17984 0.20425 0.05891] worst seeds [4 4 3] misses [0 0]
narrow tilt max [0.20452 0.23452 0.16333] worst seeds [ 3 11  5] misses [0 0]
small horizontal max [0.29891 0.36013 0.13073] worst seeds [0 3 7] misses [0 0]
opposite inclination max [0.20114 0.24021 0.12218] worst seeds [ 3  6 11] misses [0 0]
near horizontal max [0.11044 0.12926 0.07048] worst seeds [ 5  9 11] misses [0 0]
```

### Turn 6 — `execute_in_remote_docker` (`30896 ms` LLM)
- **Rationale**: Confirm the exact stress-tested artifact imports and meets the API and accuracy targets in a clean process. An artifact mismatch or isolated import failure would block submission.
- **Alternatives Weighed**: `Retune the fit after stress testing` (rejected: All tested regimes passed both tolerances; the public result is already within 0.002° of the oracle heading RMSE.)

```bash
set -e
f=/holdout_task/mag_calibrator.py
printf '%s  %s\n' '9701f4a89ddea30fd580f8c5b3c95dee34eaebaef0e32b6886999c3a28862248' "$f" | sha256sum -c -
cd /tmp
python3 -I -B - <<'PY'
import importlib.util
import pickle
import numpy as np

path = '/holdout_task/mag_calibrator.py'
spec = importlib.util.spec_from_file_location('mag_calibrator', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with open('/holdout_task/public_mag.pkl', 'rb') as f:
    data = pickle.load(f)
s, gt = data['session'], data['gt']
out = module.calibrate_and_estimate_heading(s)
heading, soft, bias, field = (out[k] for k in ('heading_deg', 'S_soft', 'b_hard', 'B_ned_uT'))
assert heading.shape == (len(s['test_m_raw']),) and bias.shape == field.shape == (3,) and soft.shape == (3, 3)
assert all(np.isfinite(x).all() for x in (heading, soft, bias, field))
assert np.all((-180 < heading) & (heading <= 180))
assert np.allclose(soft, soft.T) and np.linalg.eigvalsh(soft).min() > 0
assert abs(np.linalg.det(soft) - 1) < 1e-10 and field[0] > 0 and field[1] == 0
err = (heading - gt['gt_heading_deg'] + 180) % 360 - 180
rmse = np.sqrt(np.mean(err**2))
bias_error = np.linalg.norm(bias - gt['b_hard'])
assert rmse <= 0.45 and bias_error <= 0.35
print(f'Isolated import: heading RMSE {rmse:.5f} deg, bias error {bias_error:.5f} uT, det(S_soft) {np.linalg.det(soft):.12f}')
PY
```
- **Container Exit Code**: `1` (`379 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/holdout_task/mag_calibrator.py: FAILED
```
- **Stderr (tail)**:
```text
sha256sum: WARNING: 1 computed checksum did NOT match
```

### Turn 7 — `execute_in_remote_docker` (`27357 ms` LLM)
- **Rationale**: The literal checksum failed before the isolated test ran. A byte-for-byte archive comparison determines whether the target changed; if it matches, verify the exact target's API and public accuracy in a clean process.
- **Alternatives Weighed**: `Rerun all stress tests immediately` (rejected: The tests already passed for the archived candidate. Rerun them only if the target bytes differ and the difference is retained.)

```bash
set -e
f=/holdout_task/mag_calibrator.py
archive=/tmp/.hegel_candidate_archive/v01_9701f4a8
if ! cmp -s "$f" "$archive"; then
    echo 'Target differs from the archived stress-tested candidate (or archive is unavailable).'
    sha256sum "$f"
    if test -f "$archive"; then diff -u "$archive" "$f" | head -n 80 || true; fi
    exit 1
fi
printf 'Archived target matches byte for byte: '
sha256sum "$f"
cd /tmp
OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import importlib.util
import pickle
import numpy as np
path = '/holdout_task/mag_calibrator.py'
spec = importlib.util.spec_from_file_location('mag_calibrator', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with open('/holdout_task/public_mag.pkl', 'rb') as f:
    data = pickle.load(f)
s, gt = data['session'], data['gt']
out = module.calibrate_and_estimate_heading(s)
heading, soft, bias, field = (out[k] for k in ('heading_deg', 'S_soft', 'b_hard', 'B_ned_uT'))
assert heading.shape == (len(s['test_m_raw']),) and bias.shape == field.shape == (3,) and soft.shape == (3, 3)
assert all(np.isfinite(x).all() for x in (heading, soft, bias, field))
assert np.all((-180 < heading) & (heading <= 180))
assert np.allclose(soft, soft.T) and np.linalg.eigvalsh(soft).min() > 0
assert abs(np.linalg.det(soft)-1) < 1e-10 and field[0] > 0 and field[1] == 0
err = (heading - gt['gt_heading_deg'] + 180) % 360 - 180
rmse = np.sqrt(np.mean(err**2))
bias_error = np.linalg.norm(bias - gt['b_hard'])
assert rmse <= 0.45 and bias_error <= 0.35
print(f'Isolated import: heading RMSE {rmse:.5f} deg, bias error {bias_error:.5f} uT, det(S_soft) {np.linalg.det(soft):.12f}')
PY
```
- **Container Exit Code**: `1` (`350 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Target differs from the archived stress-tested candidate (or archive is unavailable).
4e9193028bcf25281a99568ba058fdef8d4456659850c2735ae05352e0d3ba59  /holdout_task/mag_calibrator.py
```
- **Stderr (tail)**:
```text
cmp: /tmp/.hegel_candidate_archive/v01_9701f4a8: Is a directory
```

### Turn 8 — `execute_in_remote_docker` (`35144 ms` LLM)
- **Rationale**: Inspect any source difference and test the live file directly in an isolated process. If the live file regressed, restore or correct it before submission; if it passes, the measured results establish that the actual deliverable meets the tested tolerances.
- **Alternatives Weighed**: `Submit based on the host archive record` (rejected: The live file differs from the recorded candidate, and neither earlier command reached the clean-process test.)

```bash
set -e
f=/holdout_task/mag_calibrator.py
archive=/tmp/.hegel_candidate_archive/v01_9701f4a8
old=$(find "$archive" -type f -name mag_calibrator.py -print -quit)
if [ -n "$old" ]; then
    (diff -u "$old" "$f" || true) | head -n 100
else
    find "$archive" -maxdepth 2 -type f -printf '%P\n' | head -n 20
fi
cd /tmp
OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import sys
import pickle
import numpy as np
sys.path[:0] = ['/holdout_task', '/tmp']
import mag_calibrator
from mag_stress import evaluate, synthetic
assert mag_calibrator.__file__ == '/holdout_task/mag_calibrator.py'
with open('/holdout_task/public_mag.pkl', 'rb') as f:
    public = pickle.load(f)
s, gt = public['session'], public['gt']
out = mag_calibrator.calibrate_and_estimate_heading(s)
for key, shape in [('heading_deg', (len(s['test_m_raw']),)), ('S_soft', (3,3)), ('b_hard', (3,)), ('B_ned_uT', (3,))]:
    assert out[key].shape == shape and np.isfinite(out[key]).all(), key
keys = ('calib_m_raw', 'calib_roll_pitch_rad')
rows = np.arange(len(s[keys[0]]))
public_metrics = [evaluate(s, gt)]
for keep in [rows//84 != i for i in range(5)] + [np.max(np.abs(s[keys[1]]), axis=1) < .44]:
    public_metrics.append(evaluate(dict(s, **{k: s[k][keep] for k in keys}), gt))
print('Public full:', public_metrics[0], 'public worst:', np.max(public_metrics, axis=0), flush=True)
configs = {'public-like': (420,.65,23.5,42,.09,1.35,False), 'narrow tilt': (250,.31,23,42,.12,1.8,False), 'small horizontal': (320,.49,16,49,.13,2.2,True), 'opposite inclination': (240,.45,24,-44,.12,2.1,True), 'near horizontal': (240,.42,44,4,.12,2.1,True)}
all_metrics = list(public_metrics)
for name, cfg in configs.items():
    metrics = [evaluate(*synthetic(seed, *cfg)) for seed in range(12)]
    print(name, np.max(metrics, axis=0), flush=True)
    all_metrics.extend(metrics)
assert np.all(np.asarray(all_metrics)[:, [0,2]] <= [.45,.35])
print('Exact live artifact passed API and stress checks.')
PY
```
- **Container Exit Code**: `0` (`2000 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public full: (0.13290048847434965, 0.1469878509738223, 0.016807503317750022) public worst: [0.13576984 0.15400985 0.04289204]
public-like [0.17984453 0.20425036 0.0589089 ]
narrow tilt [0.20452181 0.23451576 0.16333297]
small horizontal [0.29890549 0.36012583 0.13073274]
opposite inclination [0.20114195 0.240206   0.12217858]
near horizontal [0.11043816 0.12926457 0.0704814 ]
Exact live artifact passed API and stress checks.
```

### Turn 9 — `submit_completed_task` (`16086 ms` LLM)
- **Rationale**: The live artifact passed isolated import, output schema, determinant, and accuracy checks. Public calibration subsets and synthetic stress cases passed both error limits, so no further changes are warranted.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/mag_calibrator.py': 'FILE:4716'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/mag_calibrator.py`. On the public test set, circular heading RMSE was 0.133° and hard-iron bias error was 0.017 µT, both below the requested limits. All 60 synthetic stress cases also passed both limits.`

### Turn 10 — `execute_in_remote_docker` (`35979 ms` LLM)
- **Rationale**: Withhold each extreme tilt direction to detect loss of calibration accuracy that pooled folds might hide, and estimate held-out residual offsets by acquisition block to look for coherent drift. A tolerance miss or offset comparable to the error budget would justify diagnosing the affected regime before changing the fit.
- **Alternatives Weighed**: `Add rational terms, dynamical operators, or batch centering` (rejected: The specified sensor law is static and affine; its known leveling rotation gives exact horizontal and vertical constraints. The public session has no batch labels or classification outcome. Chronological residuals can still reveal drift without inventing batches.)

```bash
set -e
cd /tmp
OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import sys, pickle
import numpy as np
sys.path.insert(0, '/holdout_task')
import mag_calibrator as mag
p = pickle.load(open('/holdout_task/public_mag.pkl', 'rb'))
s, gt = p['session'], p['gt']
raw, rp = s['calib_m_raw'], s['calib_roll_pitch_rad']

def measure(o):
    e = (o['heading_deg'] - gt['gt_heading_deg'] + 180) % 360 - 180
    yaw = gt['gt_heading_deg']
    sectors = [e[(yaw >= k) & (yaw < k+90)] for k in (-180, -90, 0, 90)]
    rmse = lambda a: np.sqrt(np.mean(a**2))
    return rmse(e), max(map(rmse, sectors)), np.linalg.norm(o['b_hard'] - gt['b_hard'])

def audit(name, keep):
    o = mag.calibrate_and_estimate_heading(dict(s, calib_m_raw=raw[keep], calib_roll_pitch_rad=rp[keep]))
    soft = o['S_soft']
    assert np.linalg.eigvalsh(soft).min() > 0 and abs(np.linalg.det(soft)-1) < 1e-10
    assert np.all((-180 < o['heading_deg']) & (o['heading_deg'] <= 180))
    u, v, d = mag._level_axes(rp[~keep])
    m = np.linalg.solve(soft, (raw[~keep] - o['b_hard']).T).T
    level = np.column_stack([np.einsum('ij,ij->i', a, m) for a in (u, v, d)])
    h, down = o['B_ned_uT'][[0, 2]]
    delta = np.column_stack((np.hypot(level[:,0], level[:,1])-h, level[:,2]-down))
    med = np.median(delta, axis=0)
    mad = 1.4826*np.median(np.abs(delta-med), axis=0)
    metrics = measure(o)
    print(name, 'held-out n', len(m), 'heading/sector/bias', np.round(metrics, 5), 'held-out H,D median', np.round(med, 4), 'MAD sigma', np.round(mad, 4))
    return metrics

for prefix in ('calib', 'test'):
    u, v, d = mag._level_axes(s[prefix+'_roll_pitch_rad'])
    level = np.stack((u, v, d), axis=1)
    assert np.allclose(level @ level.transpose(0,2,1), np.eye(3))
    assert np.allclose(np.linalg.det(level), 1)
    x = s[prefix+'_m_raw']
    q = np.einsum('nij,nj->ni', level, x)
    np.testing.assert_allclose(q[:,0]**2+q[:,1]**2, np.sum(x*x,axis=1)-q[:,2]**2, atol=1e-10, rtol=1e-12)

metrics = []
for axis, col in zip(('roll', 'pitch'), rp.T):
    for sign in (-1, 1):
        direction = sign*col
        metrics.append(audit(f'{axis} {sign:+d} outer quartile', direction < np.quantile(direction,.75)))
for i, block in enumerate(np.array_split(np.arange(len(raw)), 5)):
    keep = np.ones(len(raw), dtype=bool)
    keep[block] = False
    metrics.append(audit(f'chronological block {i}', keep))
assert np.all(np.asarray(metrics)[:, [0,2]] <= [.45,.35]), 'Directional or chronological fold missed a target'
print('worst heading/sector/bias:', np.max(metrics,axis=0))
print('module:', mag.__file__, 'session keys:', sorted(s))
PY
```
- **Container Exit Code**: `0` (`766 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
roll -1 outer quartile held-out n 105 heading/sector/bias [0.13411 0.15025 0.03112] held-out H,D median [-0.0037 -0.0213] MAD sigma [0.0913 0.1262]
roll +1 outer quartile held-out n 105 heading/sector/bias [0.13803 0.14749 0.02207] held-out H,D median [ 0.0021 -0.0046] MAD sigma [0.0876 0.0969]
pitch -1 outer quartile held-out n 105 heading/sector/bias [0.13382 0.15139 0.02774] held-out H,D median [ 0.0042 -0.0026] MAD sigma [0.0916 0.0981]
pitch +1 outer quartile held-out n 105 heading/sector/bias [0.13619 0.14747 0.01287] held-out H,D median [-0.0096 -0.0053] MAD sigma [0.0764 0.096 ]
chronological block 0 held-out n 84 heading/sector/bias [0.13188 0.14459 0.01531] held-out H,D median [-0.0096  0.0072] MAD sigma [0.0762 0.1   ]
chronological block 1 held-out n 84 heading/sector/bias [0.13326 0.14483 0.01869] held-out H,D median [ 0.0088 -0.0028] MAD sigma [0.0899 0.0958]
chronological block 2 held-out n 84 heading/sector/bias [0.13393 0.1477  0.02128] held-out H,D median [ 0.0084 -0.0097] MAD sigma [0.1094 0.0871]
chronological block 3 held-out n 84 heading/sector/bias [0.13472 0.15401 0.03197] held-out H,D median [ 0.0008 -0.011 ] MAD sigma [0.0668 0.0981]
chronological block 4 held-out n 84 heading/sector/bias [0.13274 0.14623 0.01689] held-out H,D median [-0.0204 -0.0032] MAD sigma [0.0986 0.0955]
worst heading/sector/bias: [0.13803271 0.15400985 0.03197431]
module: /holdout_task/mag_calibrator.py session keys: ['calib_m_raw', 'calib_roll_pitch_rad', 'test_m_raw', 'test_roll_pitch_rad']
```

### Turn 11 — `submit_completed_task` (`21331 ms` LLM)
- **Rationale**: The live artifact passed isolated execution, API checks, and the determinant-one SPD constraint. Withholding extreme roll or pitch samples and contiguous acquisition blocks did not expose a tolerance failure, so the verified fit is unchanged.
- **Alternatives Weighed**: `Add a time-varying bias correction` (rejected: Held-out chronological blocks had median projection offsets no larger than 0.021 µT, well below the residual noise scale, and all blocks passed both accuracy targets.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/mag_calibrator.py': 'FILE:4716'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/mag_calibrator.py`. On the public test set, circular heading RMSE was 0.133° (limit: 0.45°), and hard-iron bias error was 0.017 µT (limit: 0.35 µT). All 60 synthetic stress cases also met both limits.`

### Turn 12 — `execute_in_remote_docker` (`36758 ms` LLM)
- **Rationale**: Withhold entire yaw sectors from the real calibration sweep to test whether the fit extrapolates to headings missing from calibration. Measure heading and bias separately, and use held-out radius and vertical residuals to diagnose any failure before changing the model.
- **Alternatives Weighed**: `Repeat the tilt and chronological audits or fit unrelated nuisance terms` (rejected: Those folds already passed, and chronological median offsets were at most 0.021 µT. No batch labels or dynamic state are present; loss of calibration yaw coverage is a distinct risk.)

```bash
set -e
cd /tmp
OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import mag_calibrator as mag
assert mag.__file__ == '/holdout_task/mag_calibrator.py'
with open('/holdout_task/public_mag.pkl', 'rb') as f:
    data = pickle.load(f)
s, gt = data['session'], data['gt']
x, rp = s['calib_m_raw'], s['calib_roll_pitch_rad']
u, v, d = mag._level_axes(rp)
# The public reference defines validation strata only; it is never passed to the estimator.
ref = np.linalg.solve(gt['S_soft'], (x - gt['b_hard']).T).T
dot = lambda a, b: np.einsum('ij,ij->i', a, b)
yaw = (np.rad2deg(np.arctan2(-dot(v, ref), dot(u, ref))) + 180) % 360 - 180
all_metrics = []
for lo in (-180, -90, 0, 90):
    held = (yaw >= lo) & (yaw < lo + 90)
    keep = ~held
    out = mag.calibrate_and_estimate_heading(dict(s, calib_m_raw=x[keep], calib_roll_pitch_rad=rp[keep]))
    e = (out['heading_deg'] - gt['gt_heading_deg'] + 180) % 360 - 180
    target_yaw = gt['gt_heading_deg']
    same = (target_yaw >= lo) & (target_yaw < lo + 90)
    soft = out['S_soft']
    m = np.linalg.solve(soft, (x[held] - out['b_hard']).T).T
    h, z = out['B_ned_uT'][[0, 2]]
    vertical = dot(d[held], m) - z
    radius = np.linalg.norm(m, axis=1) - np.hypot(h, z)
    metrics = (np.sqrt(np.mean(e**2)), np.sqrt(np.mean(e[same]**2)), np.linalg.norm(out['b_hard'] - gt['b_hard']))
    all_metrics.append(metrics)
    print(f'yaw [{lo},{lo+90}): held {held.sum()}, full/same-sector RMSE and bias {np.round(metrics,5)}, median held radius/down residual {np.round([np.median(radius), np.median(vertical)],5)} uT')
print('Worst heading/sector/bias:', np.max(all_metrics, axis=0))
assert np.all(np.asarray(all_metrics) <= [.45, .45, .35]), 'Yaw coverage stress exceeded tolerance'
PY
```
- **Container Exit Code**: `0` (`655 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
yaw [-180,-90): held 106, full/same-sector RMSE and bias [0.13365 0.13374 0.02028], median held radius/down residual [ 0.00247 -0.01509] uT
yaw [-90,0): held 99, full/same-sector RMSE and bias [0.13158 0.14646 0.02248], median held radius/down residual [0.00612 0.01221] uT
yaw [0,90): held 106, full/same-sector RMSE and bias [0.13344 0.13012 0.02345], median held radius/down residual [ 6.0e-05 -2.8e-03] uT
yaw [90,180): held 109, full/same-sector RMSE and bias [0.13429 0.12074 0.02089], median held radius/down residual [-0.00231 -0.02091] uT
Worst heading/sector/bias: [0.1342923  0.14646136 0.02345454]
```

### Turn 13 — `submit_completed_task` (`20995 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated execution, output schema and determinant checks, and the public and synthetic accuracy tests. The additional yaw-sector audit found no regression that warrants changing the model.
- **Alternatives Weighed**: `Change the fit to account for uneven yaw coverage` (rejected: Leaving out each full calibration yaw sector produced at most 0.1465° RMSE in the matching test sector and 0.0235 µT bias error.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/mag_calibrator.py`. The public test achieved 0.133° circular heading RMSE and 0.017 µT hard-iron bias error, below the limits of 0.45° and 0.35 µT. All 60 synthetic stress cases also passed both limits.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-OOD-Holdout-12",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/mag_calibrator.py": "FILE:4716"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "heading_rmse_deg": 0.1586,
    "b_hard_err_uT": 0.0192
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"heading_rmse_deg\": 0.1586, \"b_hard_err_uT\": 0.0192}\n__HEGEL_VERIFIER_AUTH_f17b846e7729dfe3aaeac94fb180d789__:{\"strict_pass\": true, \"heading_rmse_deg\": 0.1586, \"b_hard_err_uT\": 0.0192}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 4789.0,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
