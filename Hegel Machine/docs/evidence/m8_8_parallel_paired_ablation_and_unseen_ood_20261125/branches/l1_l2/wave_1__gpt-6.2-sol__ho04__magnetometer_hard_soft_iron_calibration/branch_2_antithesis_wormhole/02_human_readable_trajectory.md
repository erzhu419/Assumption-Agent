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
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `5` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `471,765` (`458,648` / `80,896` / `13,117`) |
| **Total Episode Wall Time (s)** | `503.01s` |

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

### Turn 1 — `execute_in_remote_docker` (`17161 ms` LLM)
- **Rationale**: Inspect the public schema to establish the callable inputs, available reference values, and orientation conventions before choosing validation splits or writing the fitter.

```bash
cd /holdout_task && ls -la && python - <<'PY'
import pickle,numpy as np
with open('public_mag.pkl','rb') as f: d=pickle.load(f)
def show(d,p=''):
 if isinstance(d,dict):
  for k,v in d.items(): show(v,p+'/'+str(k))
 elif isinstance(d,(list,tuple)) and d and isinstance(d[0],dict):
  print(p,type(d).__name__,len(d)); show(d[0],p+'/0')
 elif isinstance(d,np.ndarray):
  print(p,d.shape,d.dtype, np.round(d[:3],5) if d.ndim and d.dtype.kind in 'fiu' else repr(d)[:140])
 else: print(p,type(d).__name__,repr(d)[:200])
show(d)
PY
```
- **Container Exit Code**: `127` (`292 ms`, timed_out=`False`)
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

### Turn 2 — `execute_in_remote_docker` (`17937 ms` LLM)
- **Rationale**: Retry the failed inspection with python3. The public keys and reference arrays will determine the interface and whether to validate against measured headings or use held-out calibration residuals plus synthetic heading checks.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle
import numpy as np
import scipy
print('numpy', np.__version__, 'scipy', scipy.__version__)
with open('public_mag.pkl', 'rb') as f:
    data = pickle.load(f)
def show(value, path=''):
    if isinstance(value, dict):
        for key, item in value.items():
            show(item, f'{path}/{key}')
    elif isinstance(value, (list, tuple)):
        print(path, type(value).__name__, len(value))
        if value:
            show(value[0], path + '/0')
    elif isinstance(value, np.ndarray):
        print(path, value.shape, value.dtype, np.array2string(value[:3] if value.ndim else value, precision=5, threshold=12))
    else:
        print(path, type(value).__name__, repr(value)[:160])
show(data)
PY
```
- **Container Exit Code**: `0` (`647 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
numpy 1.26.4 scipy 1.13.0
/session/calib_m_raw (420, 3) float64 [[ 32.72897  -3.8574   44.55171]
 [ 71.03955 -42.6705   41.98787]
 [ 29.22284  -4.67668  45.65289]]
/session/calib_roll_pitch_rad (420, 2) float64 [[ 0.46065  0.1302 ]
 [-0.43914 -0.41728]
 [ 0.43407  0.19927]]
/session/test_m_raw (120, 3) float64 [[ -4.76198 -52.64082  52.48395]
 [ -6.06556 -50.69433  52.47467]
 [ -7.14329 -48.38551  52.55326]]
/session/test_roll_pitch_rad (120, 2) float64 [[-0.38213  0.06953]
 [-0.35959  0.10418]
 [-0.32861  0.13774]]
/gt/S_soft (3, 3) float64 [[ 1.17564  0.06974 -0.03985]
 [ 0.06974  0.9166   0.04982]
 [-0.03985  0.04982  0.93653]]
/gt/b_hard (3,) float64 [ 28.5 -41.   16.2]
/gt/gt_heading_deg (120,) float64 [-174.27042 -171.34151 -168.41259]
```

### Turn 3 — `execute_in_remote_docker` (`50760 ms` LLM)
- **Rationale**: The leveled field has constant horizontal magnitude and constant vertical component. Fit both with an SPD determinant-one correction and samplewise robust weights. The labeled public check will detect a wrong leveling convention, scale, or bias before testing orientation holdouts.
- **Alternatives Weighed**: `Ellipsoid-only fit` (rejected: It does not use the measured tilt to constrain the shared vertical field component; retain it as a comparison if bias or angular coverage is limiting.)

```bash
cat > /holdout_task/mag_calibrator.py <<'PY'
"""Robust calibration of a three-axis magnetometer from known roll and pitch."""
import numpy as np
from scipy.optimize import least_squares


def _level_axes(roll_pitch):
    roll, pitch = np.asarray(roll_pitch, dtype=float).T
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    z = np.zeros_like(roll)
    return (np.column_stack((cp, sp * sr, sp * cr)),
            np.column_stack((z, cr, -sr)),
            np.column_stack((-sp, cp * sr, cp * cr)))


def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _vertical_seed(x, v):
    # A has trace 3 in this homogeneous linear initializer. Its determinant
    # is normalized afterwards; c is A times the offset in centered units.
    a, b, c = x.T
    u, w, t = v.T
    design = np.column_stack((u*a-t*c, w*b-t*c, u*b+w*a, u*c+t*a,
                              w*c+t*b, -v, -np.ones(len(x))))
    y = -3*t*c
    weight = np.ones(len(x))
    for _ in range(9):
        p = np.linalg.lstsq(design * weight[:, None], y * weight, rcond=None)[0]
        e = design @ p - y
        scale = max(_mad(e), 1e-5)
        weight = 1 / np.sqrt(1 + (e / (2.5*scale))**2)
    a, b, c, d, e = p[:5]
    A = np.array([[a,c,d], [c,b,e], [d,e,3-a-b]])
    if np.linalg.eigvalsh(A)[0] < 0.15:
        return np.eye(3), np.zeros(3)
    return A / np.cbrt(np.linalg.det(A)), np.linalg.solve(A, p[5:8])


def _pack(A, bias, H, D):
    L = np.linalg.cholesky(A)
    return np.r_[bias, np.log(L[0,0]), np.log(L[1,1]), L[1,0], L[2,0], L[2,1], H, D]


def _unpack(p):
    a, b, c, d, e = p[3:8]
    L = np.array([[np.exp(a),0,0], [c,np.exp(b),0], [d,e,np.exp(-a-b)]])
    return L @ L.T


def _fit(raw, angles):
    v = _level_axes(angles)[2]
    center = np.median(raw, axis=0)
    unit = np.median(np.linalg.norm(raw - center, axis=1))
    x = (raw - center) / unit
    A, b = _vertical_seed(x, v)
    z = (x - b) @ A
    vert = np.sum(z*v, axis=1)
    hor = np.linalg.norm(z - vert[:, None]*v, axis=1)
    p = _pack(A, b, np.median(hor), np.median(vert))

    def residual(p):
        A = _unpack(p)
        z = (x - p[:3]) @ A
        vert = np.sum(z*v, axis=1)
        hvec = z - vert[:, None]*v
        hor = np.linalg.norm(hvec, axis=1)
        # Push isotropic raw-space noise through the two invariant gradients.
        gv = v @ A
        gh = (hvec / np.maximum(hor, 1e-12)[:, None]) @ A
        nv = np.linalg.norm(gv, axis=1)
        parallel = np.sum(gh * gv, axis=1) / nv
        nh = np.linalg.norm(gh - (parallel/nv)[:, None]*gv, axis=1)
        rv = (vert - p[9]) / nv
        rh = (hor - p[8] - parallel*rv) / np.maximum(nh, 1e-12)
        return np.column_stack((rv, rh))

    weight = np.ones(len(x))
    for _ in range(7):
        p = least_squares(lambda q: (weight[:, None]*residual(q)).ravel(), p,
                          x_scale='jac', max_nfev=45).x
        r = residual(p)
        scale = max(_mad(r.ravel()), 1e-7)
        weight = 1 / np.sqrt(1 + np.sum(r*r, axis=1) / (2.5*scale)**2)
    for _ in range(2):
        r = residual(p)
        scale = max(_mad(r.ravel()), 1e-7)
        keep = np.linalg.norm(r, axis=1) < 4.3*scale
        p = least_squares(lambda q: residual(q)[keep].ravel(), p,
                          x_scale='jac', max_nfev=45).x
    return _unpack(p), center + unit*p[:3]


def calibrate_and_estimate_heading(session: dict) -> dict:
    """Return hard/soft-iron calibration and tilt-compensated magnetic yaw."""
    A, bias = _fit(np.asarray(session['calib_m_raw'], dtype=float),
                   session['calib_roll_pitch_rad'])
    corrected = (np.asarray(session['test_m_raw'], dtype=float) - bias) @ A
    x, y, _ = _level_axes(session['test_roll_pitch_rad'])
    yaw = np.rad2deg(np.arctan2(-np.sum(y*corrected, axis=1),
                               np.sum(x*corrected, axis=1)))
    yaw[yaw <= -180.] = 180.
    return {'S_soft': np.linalg.inv(A), 'b_hard': bias, 'heading_deg': yaw}
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle, time
import numpy as np
from mag_calibrator import calibrate_and_estimate_heading
p=pickle.load(open('public_mag.pkl','rb')); g=p['gt']; s=p['session']; start=time.monotonic()
o=calibrate_and_estimate_heading(s); e=(o['heading_deg']-g['gt_heading_deg']+180)%360-180
print('RMSE, bias, soft, det, seconds:', np.mean(e*e)**.5,np.linalg.norm(o['b_hard']-g['b_hard']),np.linalg.norm(o['S_soft']-g['S_soft']),np.linalg.det(o['S_soft']),time.monotonic()-start)
print('heading rmse by quartile:',[np.mean(v*v)**.5 for v in np.array_split(e,4)])
PY
```
- **Container Exit Code**: `0` (`629 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/mag_calibrator.py']`
- **Stdout (tail)**:
```text
RMSE, bias, soft, det, seconds: 0.1329004884805871 0.01680750348994527 0.0008198476721201723 0.9999999999999997 0.03969353507272899
heading rmse by quartile: [0.13270532479802347, 0.14819338424762188, 0.13071054616424685, 0.11829171615310495]
```

### Turn 4 — `execute_in_remote_docker` (`38282 ms` LLM)
- **Rationale**: Remove whole calibration orientation blocks to detect dependence on a narrow part of the public trajectory. Narrow tilt coverage and inject coherent dipole spikes in synthetic sessions to check whether the initializer needs a stronger outlier-resistant seed.

```bash
cat > /tmp/mag_stress.py <<'PY'
import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from mag_calibrator import calibrate_and_estimate_heading, _level_axes

with open('/holdout_task/public_mag.pkl', 'rb') as f:
    p = pickle.load(f)
s, g = p['session'], p['gt']

def score(session, truth):
    o = calibrate_and_estimate_heading(session)
    A = o['S_soft']
    assert np.allclose(A, A.T) and np.linalg.eigvalsh(A).min() > 0
    assert np.isclose(np.linalg.det(A), 1, atol=1e-10)
    yaw = o['heading_deg']
    assert yaw.shape == truth['gt_heading_deg'].shape and np.all((yaw > -180) & (yaw <= 180))
    e = (yaw - truth['gt_heading_deg'] + 180) % 360 - 180
    return np.r_[np.sqrt(np.mean(e**2)), max(np.sqrt(np.mean(x**2)) for x in np.array_split(e, 4)), np.linalg.norm(o['b_hard'] - truth['b_hard'])]

print('columns: heading RMSE, worst test quartile RMSE, bias L2')
yr = s['calib_roll_pitch_rad']
ref = (s['calib_m_raw'] - g['b_hard']) @ np.linalg.inv(g['S_soft'])
ax = _level_axes(yr)
yaw = np.arctan2(-np.sum(ref*ax[1], axis=1), np.sum(ref*ax[0], axis=1))
lev = np.sum(ref*ax[2], axis=1)
h = np.sqrt(np.maximum(0, np.sum(ref**2, axis=1) - lev**2))
print('public levels H D, radial and vertical MAD:', np.median(h), np.median(lev), 1.4826*np.median(abs(h-np.median(h))), 1.4826*np.median(abs(lev-np.median(lev))))
for name, dim in [('order', np.arange(len(yaw))), ('estimated yaw', yaw), ('roll', yr[:,0]), ('pitch', yr[:,1])]:
    vals = []
    for block in np.array_split(np.argsort(dim), 4):
        keep = ~np.isin(np.arange(len(yaw)), block)
        ss = dict(s, calib_m_raw=s['calib_m_raw'][keep], calib_roll_pitch_rad=yr[keep])
        vals.append(score(ss, g))
    print(name, 'worst', np.max(vals, axis=0).round(4))

def forward(angles, heading, H, D, S, b):
    u, v, w = _level_axes(angles)
    return (H*np.cos(heading)[:,None]*u - H*np.sin(heading)[:,None]*v + D*w) @ S + b

for coverage, ratio, cond, spike in [(1, 0.45, 1.5, 75), (.5, .45, 2.8, 100), (.25, .45, 2.8, 150), (.5, .30, 2.8, 100)]:
    vals = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(3,3)))
        S = (q * np.array([cond**.5, 1., cond**-.5])) @ q.T
        b = rng.uniform(-75, 75, 3)
        cy = rng.uniform(-np.pi, np.pi, len(yr))
        ty = np.linspace(-np.pi, np.pi, 120)
        ca = yr * coverage
        ta = s['test_roll_pitch_rad']
        m = forward(ca, cy, 50*ratio, 50, S, b) + rng.normal(0, .05, (len(ca),3))
        ii = rng.choice(len(ca), round(.04*len(ca)), replace=False)
        dipole = rng.normal(size=3); dipole /= np.linalg.norm(dipole)
        m[ii] += rng.uniform(.35, 1, len(ii))[:,None] * spike * dipole
        tm = forward(ta, ty, 50*ratio, 50, S, b) + rng.normal(0,.05,(len(ta),3))
        vals.append(score(dict(calib_m_raw=m, calib_roll_pitch_rad=ca, test_m_raw=tm, test_roll_pitch_rad=ta), dict(b_hard=b, gt_heading_deg=np.rad2deg(ty))))
    print('stress', (coverage, ratio, cond, spike), 'worst', np.max(vals, axis=0).round(4))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/mag_stress.py
```
- **Container Exit Code**: `0` (`8330 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
columns: heading RMSE, worst test quartile RMSE, bias L2
public levels H D, radial and vertical MAD: 23.491583735052675 41.99447680512864 0.08601172530465341 0.0928633800800972
order worst [0.1349 0.1538 0.0318]
estimated yaw worst [0.1342 0.1506 0.0297]
roll worst [0.138  0.1517 0.0311]
pitch worst [0.1362 0.1528 0.0277]
stress (1, 0.45, 1.5, 75) worst [0.1438 0.1736 0.0337]
stress (0.5, 0.45, 2.8, 100) worst [0.1755 0.2174 0.0797]
stress (0.25, 0.45, 2.8, 150) worst [33.084  44.4264 42.4474]
stress (0.5, 0.3, 2.8, 100) worst [0.2631 0.3295 0.1072]
```

### Turn 5 — `execute_in_remote_docker` (`30097 ms` LLM)
- **Rationale**: Rerun the same public and synthetic slices against the challenger. Promote it only if it fixes the severe spike case without sacrificing the labeled public margins.
- **Alternatives Weighed**: `Add higher-order magnetic distortion terms` (rejected: The failing stress was generated by the specified affine model; higher-order terms would not address the initialization failure.)
- **ATMS NOGOOD DECLARED**: Refuted `Starting from the full-data vertical regression and iterating residual weights is sufficient even when calibration tilt is narrow and dipole interference is strong.` | Witness: `With one-quarter public tilt coverage, soft-iron condition number 2.8, and 4% coherent spikes up to 150 µT, the worst synthetic session had 33.084° circular RMSE and 42.4474 µT bias error.` | Blanket Diagnosis: `The SPD determinant-one model already includes all six symmetric terms and the exact horizontal and vertical invariants; weak tilt excitation and coherent high-leverage spikes can instead invalidate a full-data start.` | Sublated Pivot: `Keep the physical constraints, initialize from small-subset consensus, and apply robust weights before the first joint optimization.`

```bash
cp /holdout_task/mag_calibrator.py /tmp/mag_v1.py
python3 - <<'PY'
from pathlib import Path
f = Path('/holdout_task/mag_calibrator.py')
s = f.read_text()
a = s.index('    weight = np.ones(len(x))')
b = s.index('\n\ndef _pack', a)
s = s[:a] + '''    def matrix(p):
        a, b, c, d, e = p[:5]
        return np.array([[a,c,d], [c,b,e], [d,e,3-a-b]])

    def rank_seed(p):
        A = matrix(p)
        if np.linalg.eigvalsh(A)[0] <= 0.05:
            return np.inf
        e = (design @ p - y) / np.linalg.norm(v @ A, axis=1)
        return np.median(np.abs(e))

    p = np.linalg.lstsq(design, y, rcond=None)[0]
    best = rank_seed(p)
    # Small random subsets give clean hypotheses even with coherent spikes
    # that exert high leverage in the full vertical regression.
    rng = np.random.default_rng(0)
    for _ in range(64):
        idx = rng.choice(len(x), min(len(x), 24), replace=False)
        q = np.linalg.lstsq(design[idx], y[idx], rcond=None)[0]
        score = rank_seed(q)
        if score < best:
            p, best = q, score
    A = matrix(p)
    if not np.isfinite(best):
        return np.eye(3), np.zeros(3)
    return A / np.cbrt(np.linalg.det(A)), np.linalg.solve(A, p[5:8])
''' + s[b:]
s = s.replace('    weight = np.ones(len(x))\n    for _ in range(7):', '    r = residual(p)\n    scale = max(_mad(r.ravel()), 1e-7)\n    weight = 1 / np.sqrt(1 + np.sum(r*r, axis=1) / (2.5*scale)**2)\n    for _ in range(7):')
f.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/mag_stress.py
```
- **Container Exit Code**: `0` (`4882 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/mag_calibrator.py']`
- **Stdout (tail)**:
```text
columns: heading RMSE, worst test quartile RMSE, bias L2
public levels H D, radial and vertical MAD: 23.491583735052675 41.99447680512864 0.08601172530465341 0.0928633800800972
order worst [0.1349 0.1538 0.0318]
estimated yaw worst [0.1342 0.1506 0.0297]
roll worst [0.138  0.1517 0.0311]
pitch worst [0.1362 0.1528 0.0277]
stress (1, 0.45, 1.5, 75) worst [0.1438 0.1736 0.0337]
stress (0.5, 0.45, 2.8, 100) worst [0.1755 0.2174 0.0797]
stress (0.25, 0.45, 2.8, 150) worst [0.1813 0.2292 0.1409]
stress (0.5, 0.3, 2.8, 100) worst [0.2631 0.3295 0.1072]
```

### Turn 6 — `execute_in_remote_docker` (`32417 ms` LLM)
- **Rationale**: The consensus seed repairs the observed leverage failure without degrading the public slices. Combine weak tilt, small horizontal field, higher noise, and fewer samples to test whether the remaining margin is limited by calibration or by test noise, using true calibration as the comparison.

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/mag_stress.py').read_text()
p = p[:p.index("print('columns:")] + p[p.index('def forward('):p.index('for coverage,')]
Path('/tmp/mag_joint_corners.py').write_text(p + '''
print('columns: circular RMSE, worst test quartile RMSE, bias L2')
for count, coverage, ratio, cond, spike, sigma in [(420,.25,.30,2.8,150,.085), (220,.50,.30,2.8,100,.085), (220,.25,.45,2.8,150,.085), (420,.5,.30,2.8,100,.10)]:
    vals, oracle = [], []
    for seed in range(40):
        rng = np.random.default_rng(1700 + seed)
        q, _ = np.linalg.qr(rng.normal(size=(3,3)))
        S = (q * np.array([cond**.5, 1., cond**-.5])) @ q.T
        b = rng.uniform(-75,75,3)
        ca = s['calib_roll_pitch_rad'][rng.choice(len(s['calib_roll_pitch_rad']), count, replace=False)] * coverage
        ta = s['test_roll_pitch_rad']
        cy = rng.uniform(-np.pi,np.pi,count)
        ty = np.linspace(-np.pi,np.pi,len(ta))
        m = forward(ca, cy, 50*ratio, 50,S,b) + rng.normal(0,sigma,(count,3))
        ii = rng.choice(count,round(.04*count),replace=False)
        dipole = rng.normal(size=3); dipole /= np.linalg.norm(dipole)
        m[ii] += rng.uniform(.35,1,len(ii))[:,None]*spike*dipole
        tm = forward(ta,ty,50*ratio,50,S,b) + rng.normal(0,sigma,(len(ta),3))
        vals.append(score(dict(calib_m_raw=m,calib_roll_pitch_rad=ca,test_m_raw=tm,test_roll_pitch_rad=ta),dict(b_hard=b,gt_heading_deg=np.rad2deg(ty))))
        z = (tm-b) @ np.linalg.inv(S)
        u,v,_ = _level_axes(ta)
        d = (np.rad2deg(np.arctan2(-np.sum(z*v,axis=1),np.sum(z*u,axis=1))-ty)+180)%360-180
        oracle.append(np.sqrt(np.mean(d*d)))
    vals = np.array(vals)
    print((count,coverage,ratio,cond,spike,sigma), 'worst', vals.max(axis=0).round(4), 'worst seeds', vals.argmax(axis=0)+1700, 'oracle at worst heading', round(oracle[vals[:,0].argmax()],4), 'session target misses', np.sum((vals[:,0]>.45)|(vals[:,2]>.35)))
''')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/mag_joint_corners.py
```
- **Container Exit Code**: `0` (`7324 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
columns: circular RMSE, worst test quartile RMSE, bias L2
(420, 0.25, 0.3, 2.8, 150, 0.085) worst [0.4806 0.5647 0.3507] worst seeds [1723 1723 1707] oracle at worst heading 0.4637 session target misses 3
(220, 0.5, 0.3, 2.8, 100, 0.085) worst [0.4856 0.6595 0.1837] worst seeds [1701 1732 1702] oracle at worst heading 0.4834 session target misses 2
(220, 0.25, 0.45, 2.8, 150, 0.085) worst [0.3272 0.4424 0.2489] worst seeds [1701 1732 1722] oracle at worst heading 0.3223 session target misses 0
(420, 0.5, 0.3, 2.8, 100, 0.1) worst [0.5512 0.6726 0.1919] worst seeds [1723 1700 1707] oracle at worst heading 0.5455 session target misses 18
```

### Turn 7 — `execute_in_remote_docker` (`27238 ms` LLM)
- **Rationale**: The near-threshold bias miss may still be fixable. Refit the same simulated sessions with known spike rows removed: if the bias miss persists, it points to limited calibration information; if it disappears, use the paired gap to decide whether outlier rejection needs improvement.
- **Alternatives Weighed**: `Smooth estimated test headings over time` (rejected: The interface specifies pointwise tilt-compensated heading and provides no test-motion model; smoothing could distort genuine maneuvers.)

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/mag_joint_corners.py').read_text()
p = p.replace('vals, oracle = [], []', 'vals, oracle, clean = [], [], []')
p = p.replace('        z = (tm-b)', '        clean_rows = np.ones(count, dtype=bool)\n        clean_rows[ii] = False\n        clean.append(score(dict(calib_m_raw=m[clean_rows],calib_roll_pitch_rad=ca[clean_rows],test_m_raw=tm,test_roll_pitch_rad=ta),dict(b_hard=b,gt_heading_deg=np.rad2deg(ty))))\n        z = (tm-b)')
p = p[:p.index('    vals = np.array(vals)')] + '''    vals, clean, oracle = np.array(vals), np.array(clean), np.array(oracle)
    failing = np.flatnonzero((vals[:,0] > .45) | (vals[:,2] > .35))
    j = vals[:,2].argmax()
    print((count,coverage,ratio,cond,spike,sigma), 'bias worst actual/clean', vals[j,2], clean[j,2], 'max paired delta in bias', np.max(vals[:,2]-clean[:,2]))
    print('failed heading with/without spikes and with true calibration:', [(int(i)+1700, *np.round([vals[i,0],clean[i,0],oracle[i]],4)) for i in failing if vals[i,0] > .45])
'''
Path('/tmp/mag_paired_oracle.py').write_text(p)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/mag_paired_oracle.py
```
- **Container Exit Code**: `0` (`12951 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
columns: circular RMSE, worst test quartile RMSE, bias L2
(420, 0.25, 0.3, 2.8, 150, 0.085) bias worst actual/clean 0.35067520276259007 0.35067521646536737 max paired delta in bias 0.004830551672924038
failed heading with/without spikes and with true calibration: [(1723, 0.4806, 0.4806, 0.4637), (1725, 0.4545, 0.4545, 0.4465)]
(220, 0.5, 0.3, 2.8, 100, 0.085) bias worst actual/clean 0.18365237993117756 0.18365237902112894 max paired delta in bias 0.019760248720345097
failed heading with/without spikes and with true calibration: [(1701, 0.4856, 0.4856, 0.4834), (1732, 0.4586, 0.4586, 0.4575)]
(220, 0.25, 0.45, 2.8, 150, 0.085) bias worst actual/clean 0.2488712372740219 0.24887128718324703 max paired delta in bias 0.03399759941507287
failed heading with/without spikes and with true calibration: []
(420, 0.5, 0.3, 2.8, 100, 0.1) bias worst actual/clean 0.1918753816495763 0.19187539818047725 max paired delta in bias 6.703721916467842e-08
failed heading with/without spikes and with true calibration: [(1700, 0.5137, 0.5137, 0.5093), (1701, 0.5077, 0.5077, 0.4946), (1702, 0.4544, 0.4544, 0.4567), (1708, 0.4529, 0.4529, 0.4464), (1710, 0.4694, 0.4694, 0.4765), (1713, 0.4842, 0.4842, 0.4852), (1720, 0.4965, 0.4965, 0.4937), (1723, 0.5512, 0.5512, 0.5455), (1724, 0.5053, 0.5053, 0.4983), (1725, 0.5312, 0.5312, 0.5254), (1726, 0.4658, 0.4658, 0.4639), (1727, 0.4912, 0.4912, 0.4861), (1729, 0.473, 0.473, 0.472), (1730, 0.4509, 0.4509, 0.4404), (1732, 0.4919, 0.4919, 0.4919), (1733, 0.4505, 0.4505, 0.4474), (1737, 0.4734, 0.4734, 0.4668), (1738, 0.4834, 0.4834, 0.4776)]
```

### Turn 8 — `execute_in_remote_docker` (`35158 ms` LLM)
- **Rationale**: Keep the consensus fitter and verify the exact deliverable in an isolated Python process. The labeled fixture checks both targets; the signed cardinal and endpoint probes catch output-convention errors that aggregate RMSE may miss.
- **Alternatives Weighed**: `Tighten spike rejection to improve the worst bias` (rejected: Removing the exact injected spike rows left the worst bias unchanged to six decimal places.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('mag_calibrator', '/holdout_task/mag_calibrator.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with open('/holdout_task/public_mag.pkl', 'rb') as f:
    data = pickle.load(f)
s, gt = data['session'], data['gt']
o = module.calibrate_and_estimate_heading(s)
S, b, h = (o[k] for k in ('S_soft', 'b_hard', 'heading_deg'))
assert S.shape == (3, 3) and b.shape == (3,) and h.shape == (len(s['test_m_raw']),)
assert np.isfinite(S).all() and np.isfinite(b).all() and np.isfinite(h).all()
assert np.allclose(S, S.T, atol=1e-12, rtol=0) and np.linalg.eigvalsh(S).min() > 0
assert abs(np.linalg.det(S) - 1) < 1e-12
assert np.all((h > -180) & (h <= 180))
err = (h - gt['gt_heading_deg'] + 180) % 360 - 180
rmse = np.sqrt(np.mean(err**2))
bias = np.linalg.norm(b - gt['b_hard'])
assert rmse <= .45 and bias <= .35
print(f'Public: circular RMSE {rmse:.6f} deg; bias error {bias:.6f} uT; determinant {np.linalg.det(S):.15f}')
# Zero roll/pitch isolates the yaw sign and tests the exact negative-180 branch.
probe = dict(s, test_m_raw=np.array([[-1., 0., 0.], [0., -1., 0.], [0., 1., 0.]]) @ S + b, test_roll_pitch_rad=np.zeros((3, 2)))
angles = module.calibrate_and_estimate_heading(probe)['heading_deg']
assert np.all((angles > -180) & (angles <= 180))
assert np.allclose((angles - [180, 90, -90] + 180) % 360 - 180, 0, atol=1e-10)
# Drive atan2 with an exact signed zero to exercise the endpoint directly.
A = np.eye(3)
module._fit = lambda *_: (A, np.zeros(3))
probe['test_m_raw'] = np.array([[-1., 0., 0.]])
probe['test_roll_pitch_rad'] = np.zeros((1, 2))
assert module.calibrate_and_estimate_heading(probe)['heading_deg'][0] == 180.
print('Output contract, cardinal yaw signs, and exact wrap endpoint verified.')
PY
```
- **Container Exit Code**: `0` (`628 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public: circular RMSE 0.132900 deg; bias error 0.016808 uT; determinant 1.000000000000000
Output contract, cardinal yaw signs, and exact wrap endpoint verified.
```

### Turn 9 — `submit_completed_task` (`20665 ms` LLM)
- **Rationale**: Consensus initialization fixed the observed high-leverage failure without degrading the public orientation holdouts. An isolated Python process verified both public targets and the output contract on the exact submitted artifact.
- **Alternatives Weighed**: `Tighten spike rejection` (rejected: Removing every injected spike left the worst synthetic bias unchanged.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/mag_calibrator.py': 'FILE:4511'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/mag_calibrator.py`. On the public fixture, circular heading RMSE was 0.133° and hard-iron bias error was 0.0168 µT, both below target. The soft-iron matrix was symmetric positive-definite with determinant 1, and the heading wrap was verified.

Limitations: In synthetic stress tests with a weak horizontal field and high test noise, heading RMSE reached 0.551° versus 0.546° using the true calibration. The largest synthetic bias error was 0.351 µT and persisted after all injected spikes were removed.`

### Turn 10 — `execute_in_remote_docker` (`42710 ms` LLM)
- **Rationale**: Use the public measurements to check that all ten physical degrees of freedom are locally identifiable and that inlier residuals do not shift systematically with acquisition order or orientation. A rank defect or coherent residual shift would justify investigating a model change; comparison against true calibration shows how much public heading error a calibration change could remove.
- **Alternatives Weighed**: `Add relay, drift-operator, or classification terms` (rejected: The specified measurement equation is affine with symmetric cross-axis coupling; it has no relay state, stochastic drift operator, or class outcome.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -B - <<'PY'
import pickle
import numpy as np
from mag_calibrator import calibrate_and_estimate_heading, _level_axes
with open('public_mag.pkl','rb') as f:
    p=pickle.load(f)
s,g=p['session'],p['gt']
o=calibrate_and_estimate_heading(s)
A=np.linalg.inv(o['S_soft']); x=s['calib_m_raw']-o['b_hard']; m=x@A
u,v,w=_level_axes(s['calib_roll_pitch_rad'])
z=np.einsum('ij,ij->i',m,w); h=np.hypot(np.sum(m*u,axis=1),np.sum(m*v,axis=1)); H,D=np.median(h),np.median(z)
r=np.column_stack((h-H,z-D)); sig=1.4826*np.median(abs(r),axis=0); good=np.linalg.norm(r/sig,axis=1)<4.3
# Rank of the measured invariant sensitivities: three biases, five traceless
# symmetric correction perturbations, and the two common field components.
P=(m-z[:,None]*w)/h[:,None]; E=np.eye(3)
basis=[np.diag([1.,-1.,0.]),np.diag([1.,0.,-1.])]+[np.outer(E[i],E[j])+np.outer(E[j],E[i]) for i,j in [(0,1),(0,2),(1,2)]]
def design(v):
    return np.column_stack((-v@A,*[np.sum(v*(x@M),axis=1)/H for M in basis]))
J=np.vstack((np.column_stack((design(P),-np.ones(len(h)),np.zeros(len(h))))[good],np.column_stack((design(w),np.zeros(len(h)),-np.ones(len(h))))[good]))
sv=np.linalg.svd(J,compute_uv=False)
print('session keys:', sorted(s)); print('calibration rows retained:', good.sum(), 'of', len(good), 'H,D:', np.round([H,D],4), 'radial,vertical MAD:',sig.round(4)); print('scaled invariant sensitivity rank and condition:',np.linalg.matrix_rank(J),round(sv[0]/sv[-1],2))
for name, order in [('time',np.arange(len(h))), ('roll',np.argsort(s['calib_roll_pitch_rad'][:,0])), ('pitch',np.argsort(s['calib_roll_pitch_rad'][:,1])), ('yaw',np.argsort(np.arctan2(-np.sum(m*v,axis=1),np.sum(m*u,axis=1))))]:
    print(name, 'quartile median residuals (uT):', [np.median(r[q[good[q]]],axis=0).round(4).tolist() for q in np.array_split(order,4)])
u,v,_=_level_axes(s['test_roll_pitch_rad']); m=(s['test_m_raw']-g['b_hard'])@np.linalg.inv(g['S_soft']); oracle=np.rad2deg(np.arctan2(-np.sum(m*v,axis=1),np.sum(m*u,axis=1)))
for name, pred in [('fitted',o['heading_deg']),('true calibration',oracle)]:
    e=(pred-g['gt_heading_deg']+180)%360-180; print(name, 'heading RMSE:',round(np.sqrt(np.mean(e*e)),6),'worst quartile:',round(max(np.sqrt(np.mean(q*q)) for q in np.array_split(e,4)),6))
PY
```
- **Container Exit Code**: `0` (`598 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
session keys: ['calib_m_raw', 'calib_roll_pitch_rad', 'test_m_raw', 'test_roll_pitch_rad']
calibration rows retained: 404 of 420 H,D: [23.4914 42.0029] radial,vertical MAD: [0.0833 0.0928]
scaled invariant sensitivity rank and condition: 10 11.19
time quartile median residuals (uT): [[-0.0021, 0.0074], [0.0083, -0.0098], [-0.0031, -0.0026], [-0.006, 0.0091]]
roll quartile median residuals (uT): [[-0.0043, -0.009], [-0.0031, 0.0114], [0.004, -0.0047], [0.0016, -0.0065]]
pitch quartile median residuals (uT): [[0.0054, 0.0003], [-0.0078, -0.0074], [0.006, 0.0044], [-0.0031, 0.0042]]
yaw quartile median residuals (uT): [[-0.0019, -0.0098], [-0.0029, 0.0163], [0.0014, -0.0], [0.0022, -0.0066]]
fitted heading RMSE: 0.1329 worst quartile: 0.148193
true calibration heading RMSE: 0.131067 worst quartile: 0.142001
```

### Turn 11 — `submit_completed_task` (`22136 ms` LLM)
- **Rationale**: The real-data audit supports the affine model: true calibration improved public heading RMSE by only 0.0018°. Paired synthetic tests showed that removing spike rows did not fix the worst bias miss. The unchanged artifact passed isolated-process verification.
- **Alternatives Weighed**: `Add drift or higher-order terms` (rejected: The public invariant sensitivities had full rank, and orientation and acquisition-order quartile residual medians stayed within 0.017 µT of zero, well below the inlier noise scale.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/mag_calibrator.py': 'FILE:4511'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/mag_calibrator.py`. On the public fixture, circular heading RMSE was 0.133° and hard-iron bias error was 0.0168 µT, both below target. The soft-iron matrix was symmetric positive-definite with determinant 1, and heading wrap was verified.

Limitations: In the most demanding synthetic tests, heading RMSE reached 0.551° versus 0.546° with true calibration. The largest bias error was 0.351 µT and persisted after removing all injected spikes.`

### Turn 12 — `execute_in_remote_docker` (`35778 ms` LLM)
- **Rationale**: The earlier block-deletion audit still trained on 75% of the calibration rows. Fit on narrower measured tilt and acquisition bands to distinguish weak excitation from test noise. If a band misses the bias target, its fixed row selection gives a concrete case for comparing initializers or quantifying limited calibration information.
- **Alternatives Weighed**: `Add nonlinear terms or fit a batch offset` (rejected: The public affine sensitivities have full rank, residual shifts were small, and the session has no batch labels or classification outcome.)

```bash
cat > /tmp/mag_excitation_audit.py <<'PY'
import pickle
import sys
import numpy as np

sys.path.insert(0, '/holdout_task')
from mag_calibrator import calibrate_and_estimate_heading, _level_axes

with open('/holdout_task/public_mag.pkl', 'rb') as f:
    data = pickle.load(f)
s, gt = data['session'], data['gt']
angles = s['calib_roll_pitch_rad']
raw = s['calib_m_raw']
n = len(raw)
# Select calibration rows using only acquisition order and measured roll/pitch.
# None of these selections uses the labeled test headings or calibration truth.
order = np.arange(n)
roll, pitch = angles.T
tilt = np.arccos(np.clip(np.cos(roll)*np.cos(pitch), -1, 1))
strata = {'acquisition': order, 'roll': np.argsort(roll),
          'pitch': np.argsort(pitch), 'absolute tilt': np.argsort(tilt)}

u, v, _ = _level_axes(s['test_roll_pitch_rad'])
z = (s['test_m_raw']-gt['b_hard']) @ np.linalg.inv(gt['S_soft'])
oracle = np.rad2deg(np.arctan2(-np.sum(z*v, axis=1), np.sum(z*u, axis=1)))
# Define held-out test slices from measured tilt, not fitted residuals.
test_order = np.argsort(np.linalg.norm(s['test_roll_pitch_rad'], axis=1))
test_groups = np.array_split(test_order, 4)
def error(h):
    e = (h-gt['gt_heading_deg']+180)%360-180
    return np.sqrt(np.mean(e**2)), max(np.sqrt(np.mean(e[k]**2)) for k in test_groups)
print('oracle: overall and worst test tilt-quartile RMSE', np.round(error(oracle), 5))
for name, idx in strata.items():
    for fraction in (0.50, 0.25):
        size = round(fraction*n)
        records = []
        for start in np.unique(np.linspace(0,n-size,5,dtype=int)):
            k = idx[start:start+size]
            out = calibrate_and_estimate_heading(dict(s, calib_m_raw=raw[k], calib_roll_pitch_rad=angles[k]))
            bias = np.linalg.norm(out['b_hard']-gt['b_hard'])
            records.append([*error(out['heading_deg']), bias])
        records = np.array(records)
        print(name, fraction, 'max RMSE / test tilt-quartile RMSE / bias:', records.max(axis=0).round(5), 'target misses:', int(np.sum((records[:,0]>.45)|(records[:,2]>.35))))
PY
OPENBLAS_NUM_THREADS=1 python3 -I -B /tmp/mag_excitation_audit.py
```
- **Container Exit Code**: `0` (`1954 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
oracle: overall and worst test tilt-quartile RMSE [0.13107 0.13447]
acquisition 0.5 max RMSE / test tilt-quartile RMSE / bias: [0.13608 0.14029 0.0348 ] target misses: 0
acquisition 0.25 max RMSE / test tilt-quartile RMSE / bias: [0.13812 0.14451 0.04997] target misses: 0
roll 0.5 max RMSE / test tilt-quartile RMSE / bias: [0.13964 0.15382 0.0567 ] target misses: 0
roll 0.25 max RMSE / test tilt-quartile RMSE / bias: [0.14777 0.15566 0.09327] target misses: 0
pitch 0.5 max RMSE / test tilt-quartile RMSE / bias: [0.14244 0.15351 0.05096] target misses: 0
pitch 0.25 max RMSE / test tilt-quartile RMSE / bias: [0.14852 0.1636  0.07718] target misses: 0
absolute tilt 0.5 max RMSE / test tilt-quartile RMSE / bias: [0.13972 0.15204 0.04413] target misses: 0
absolute tilt 0.25 max RMSE / test tilt-quartile RMSE / bias: [0.14484 0.16081 0.09208] target misses: 0
```

### Turn 13 — `execute_in_remote_docker` (`31811 ms` LLM)
- **Rationale**: Replace the apparent contamination on the measured calibration trajectory with controlled 4% dipole bursts while leaving the labeled test trajectory unchanged. If a burst breaches either target, compare estimators on that fixed case; otherwise retain the verified artifact.
- **Alternatives Weighed**: `Repeat the orientation holdout audit` (rejected: Every measured subset met both targets; repeating those splits would not test whether a temporally concentrated burst can shift the fit.)

```bash
cat > /tmp/mag_transient_audit.py <<'PY'
import pickle
import sys
import numpy as np

sys.path.insert(0, '/holdout_task')
from mag_calibrator import calibrate_and_estimate_heading, _level_axes

with open('/holdout_task/public_mag.pkl', 'rb') as f:
    data = pickle.load(f)
s, gt = data['session'], data['gt']
raw, angles = s['calib_m_raw'], s['calib_roll_pitch_rad']
ref = (raw-gt['b_hard']) @ np.linalg.inv(gt['S_soft'])
u, v, w = _level_axes(angles)
h = np.hypot(np.sum(u*ref, axis=1), np.sum(v*ref, axis=1))
d = np.sum(w*ref, axis=1)
r = np.column_stack((h-np.median(h), d-np.median(d)))
scale = 1.4826*np.median(np.abs(r), axis=0)
keep = np.linalg.norm(r/scale, axis=1) <= 4.3
raw, angles = raw[keep], angles[keep]

def score(m):
    o = calibrate_and_estimate_heading(dict(s, calib_m_raw=m, calib_roll_pitch_rad=angles))
    z = (o['heading_deg'] - gt['gt_heading_deg'] + 180) % 360 - 180
    return [np.sqrt(np.mean(z*z)), np.linalg.norm(o['b_hard']-gt['b_hard'])]

print('Clean measured calibration rows:', len(raw), 'baseline RMSE / bias:', np.round(score(raw), 5))
rng = np.random.default_rng(704)
for amplitude in (0.25, 2, 20, 120):
    scores = []
    count = round(.04*len(raw))
    for start in np.linspace(0, len(raw)-count, 7, dtype=int):
        for _ in range(4):
            dipole = rng.normal(size=3)
            dipole /= np.linalg.norm(dipole)
            m = raw.copy()
            m[start:start+count] += amplitude*dipole
            scores.append(score(m))
    scores = np.array(scores)
    print('Burst amplitude', amplitude, 'worst RMSE / bias', scores.max(axis=0).round(5), 'target misses', np.count_nonzero((scores[:,0]>.45)|(scores[:,1]>.35)))
PY
OPENBLAS_NUM_THREADS=1 python3 -I -B /tmp/mag_transient_audit.py
```
- **Container Exit Code**: `0` (`5210 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Clean measured calibration rows: 404 baseline RMSE / bias: [0.1329  0.01681]
Burst amplitude 0.25 worst RMSE / bias [0.1366  0.03935] target misses 0
Burst amplitude 2 worst RMSE / bias [0.13369 0.0207 ] target misses 0
Burst amplitude 20 worst RMSE / bias [0.13369 0.0207 ] target misses 0
Burst amplitude 120 worst RMSE / bias [0.13369 0.0207 ] target misses 0
```

### Turn 14 — `submit_completed_task` (`21781 ms` LLM)
- **Rationale**: The real-data audit found full rank for the affine calibration and no material residual shift by acquisition order or orientation. Restricted-excitation and burst tests found no additional bottleneck on the public measurements, and the unchanged artifact passed isolated-process verification.
- **Alternatives Weighed**: `Tighten spike rejection` (rejected: Contiguous 4% dipole bursts on the measured trajectory caused no target misses, and removing the exact injected spike rows did not improve the worst synthetic bias.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/mag_calibrator.py`. On the public fixture, circular heading RMSE was 0.133° and hard-iron bias error was 0.0168 µT, both below target. The soft-iron matrix was symmetric positive-definite with determinant 1, and headings used the required (−180°, 180°] range.

Limitations: The hardest synthetic cases exceeded the targets. Worst heading RMSE was 0.551° (0.546° with true calibration), and worst bias error was 0.351 µT even after removing all injected spikes.`

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
