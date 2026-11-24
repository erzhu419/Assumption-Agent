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
| **Turns Used / Max Turns** | `10 / 22` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `1` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `268,381` (`257,495` / `35,840` / `10,886`) |
| **Total Episode Wall Time (s)** | `344.8s` |

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

### Turn 1 — `execute_in_remote_docker` (`20653 ms` LLM)
- **Rationale**: Inspect the public data for field names, orientation coverage, reference labels, and dependencies. These determine the output schema, initialization strategy, and which metrics can be measured directly.
- **Alternatives Weighed**: `Fit the raw sample mean and covariance` (rejected: Nonuniform orientation coverage can shift both moments, and current spikes can dominate the covariance.)

```bash
cd /holdout_task && ls -la && python3 - <<'PY'
import pickle
import numpy as np
import scipy
print('numpy',np.__version__,'scipy',scipy.__version__)
def describe(x, name='root', depth=0):
    if isinstance(x, dict) and depth < 3:
        print(name, 'keys:', list(x))
        for k,v in x.items(): describe(v, f'{name}.{k}', depth+1)
    elif isinstance(x,np.ndarray):
        print(name,x.shape,x.dtype)
        if x.size and x.dtype.kind in 'fiu':
            a=x if x.ndim else x.reshape(1)
            print('  sample:',a[:3], 'quantiles:', np.nanquantile(a,[0,.05,.5,.95,1],axis=0) if x.size>15 else '')
    elif isinstance(x,(list,tuple)):
        print(name,type(x).__name__,len(x))
        if x and depth<3: describe(x[0],name+'[0]',depth+1)
    else: print(name,type(x).__name__,str(x)[:250])
with open('public_mag.pkl','rb') as f: describe(pickle.load(f))
PY
if test -d /tmp/hegel_wormhole_skills; then ls /tmp/hegel_wormhole_skills; grep -nE '^def |^class |MAD|median' /tmp/hegel_wormhole_skills/autogenous_creep_corrected_retraction_calibration.py | head -35; fi
```
- **Container Exit Code**: `0` (`516 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
antiles: [[-3.99998998e-01 -3.49966194e-01]
 [-3.82786253e-01 -3.45423584e-01]
 [ 1.48458614e-02 -1.54390389e-16]
 [ 3.90308433e-01  3.45423584e-01]
 [ 3.98957827e-01  3.49966194e-01]]
root.gt keys: ['S_soft', 'b_hard', 'gt_heading_deg']
root.gt.S_soft (3, 3) float64
  sample: [[ 1.17564433  0.06974161 -0.03985235]
 [ 0.06974161  0.91660405  0.04981544]
 [-0.03985235  0.04981544  0.93653023]] quantiles: 
root.gt.b_hard (3,) float64
  sample: [ 28.5 -41.   16.2] quantiles: 
root.gt.gt_heading_deg (120,) float64
  sample: [-174.27042205 -171.34150739 -168.41259274] quantiles: [-174.27042205 -156.84337984    0.          156.84337984  174.27042205]
__init__.py
__pycache__
autogenous_creep_corrected_retraction_calibration.py
10:def _robust_linear(design: np.ndarray, values: np.ndarray, scale: float | None=None) -> tuple[np.ndarray, float]:
11:    """Return Cauchy IRLS coefficients and residual MAD scale in value units."""
15:        scale = max(float(1.4826 * np.median(np.abs(residual - np.median(residual)))), 1e-08)
20:    noise = max(float(1.4826 * np.median(np.abs(residual - np.median(residual)))), 1e-08)
25:def first_loading_slip(force_N: np.ndarray, specimen_displacement_um: np.ndarray, hold_start: int, hold_end: int, *, threshold_um: float=0.35) -> float | None:
43:        slope = np.median(np.diff(depth[start:stop])[rising] / dx[rising]) if np.any(rising) else 0.0
54:            baseline = np.where(after, np.median(detrended[after]), np.median(detrended[~after]))
56:            scale = max(float(1.4826 * np.median(np.abs(residual - np.median(residual)))), 1e-05)
69:def calibrate_retraction(times_s: list[np.ndarray], forces_N: list[np.ndarray], raw_displacements_um: list[np.ndarray], frame_compliance_um_per_N: float, *, transition_window_s: float=0.015, curvature_to_noise: float=0.8, slip_threshold_um: float=0.35) -> tuple[float, list[float | None]]:
70:    """Return median elastic stiffness (N/µm) and first loading slip forces (N).
76:    Use fixed-rate correction when residual curvature is <= 0.8 hold MAD scales
78:    hold MAD scale, with onset rate fixed and tau bounded between 0.1 median
103:        (hold_fit, noise) = _robust_linear(np.column_stack((np.ones_like(hold_t), hold_t)), hold_h - np.median(hold_h))
110:        y = y - np.median(y)
119:        coherent = np.sqrt(np.mean([np.median(part) ** 2 for part in bins]))
124:            (tau_min, tau_max) = (0.1 * np.median(np.diff(z)), 30 * z[-1])
137:    return (float(np.median(stiffness)), slips)
```

### Turn 2 — `execute_in_remote_docker` (`64345 ms` LLM)
- **Rationale**: Compare the two fits against held-out headings and known bias, including reduced tilt coverage. The oracle identifies how much of the measured heading error remains even with exact calibration, and the slice comparisons show whether the joint refinement improves the worst case.
- **Alternatives Weighed**: `Use the mounted creep-calibration operator directly` (rejected: Its inputs and mechanics do not apply; only its MAD-scaled robust weighting transfers to the magnetometer residuals.)

```bash
cd /holdout_task && sed -n '1,26p' /tmp/hegel_wormhole_skills/autogenous_creep_corrected_retraction_calibration.py && cat > mag_calibrator.py <<'PY'
"""Robust strapdown magnetometer calibration with unknown calibration yaw."""
import numpy as np
from scipy.optimize import least_squares


def _level(rp):
    """Rows of the body-to-level rotation R_y(pitch) R_x(roll)."""
    p, t = rp.T
    sp, cp, st, ct = np.sin(p), np.cos(p), np.sin(t), np.cos(t)
    z = np.zeros_like(p)
    return np.stack((ct, st*sp, st*cp, z, cp, -sp,
                     -st, ct*sp, ct*cp), axis=1).reshape(-1, 3, 3)


def _symmetric(q, trace):
    a, b, c, d, e = q
    return np.array([[a, c, d], [c, b, e], [d, e, trace-a-b]])


def _pack(A):
    return A[0, 0], A[1, 1], A[0, 1], A[0, 2], A[1, 2]


def _exp(q):
    w, u = np.linalg.eigh(_symmetric(q, 0.))
    return (u * np.exp(w)) @ u.T


def _vertical_seed(x, v):
    # A is symmetric and has trace 3 during this linear fit. The known
    # down direction gives v @ (A @ x + c) = D at each attitude.
    a, b, c = x.T
    i, j, k = v.T
    X = np.column_stack((i*a-k*c, j*b-k*c, i*b+j*a,
                         i*c+k*a, j*c+k*b, v, -np.ones(len(x))))
    y = -3*k*c
    rng = np.random.default_rng(176)
    best, score = None, np.inf
    for ids in [slice(None)] + [rng.choice(len(x), min(18, len(x)), replace=False) for _ in range(80)]:
        q = np.linalg.lstsq(X[ids], y[ids], rcond=None)[0]
        A = _symmetric(q[:5], 3.)
        if np.linalg.eigvalsh(A)[0] <= 0.05:
            continue
        s = np.median(np.abs(X @ q - y) / np.linalg.norm(v @ A, axis=1))
        if s < score:
            best, score = q, s
    if best is None:
        return np.eye(3), np.zeros(3)
    q = best
    # Use the heavy-tailed MAD/IRLS principle with the appropriate
    # attitude-dependent projection of raw magnetometer noise.
    for _ in range(7):
        g = np.linalg.norm(v @ _symmetric(q[:5], 3.), axis=1)
        r = (X @ q - y) / g
        s = max(1.4826*np.median(np.abs(r-np.median(r))), 1e-7)
        w = 1 / (g*np.hypot(1., r/(2.5*s)))
        q = np.linalg.lstsq(X*w[:, None], y*w, rcond=None)[0]
        if np.linalg.eigvalsh(_symmetric(q[:5], 3.))[0] <= 0.05:
            q = best
            break
    A = _symmetric(q[:5], 3.)
    return A/np.cbrt(np.linalg.det(A)), -np.linalg.solve(A, q[5:8])


def _fit(x, L, joint=True):
    A, b = _vertical_seed(x, L[:, 2])
    if not joint:
        return A, b
    m = np.einsum('nij,nj->ni', L, (x-b) @ A)
    w, u = np.linalg.eigh(A)
    q = np.r_[_pack((u*np.log(w)) @ u.T), b,
              np.median(np.linalg.norm(m[:, :2], axis=1)), np.median(m[:, 2])]

    def residual(q, ids):
        A = _exp(q[:5]); l = L[ids]
        m = np.einsum('nij,nj->ni', l, (x[ids]-q[5:8]) @ A)
        h = np.maximum(np.hypot(m[:, 0], m[:, 1]), 1e-12)
        g = ((m[:, :2]/h[:, None])[:, :, None] * l[:, :2]).sum(axis=1) @ A
        v = l[:, 2] @ A
        # Whiten the two invariants under isotropic raw-frame noise.
        vn = np.linalg.norm(v, axis=1)
        vu = v/vn[:, None]
        cross = np.sum(g*vu, axis=1)
        zd = (m[:, 2]-q[9])/vn
        rh = (h-q[8]-cross*zd)/np.linalg.norm(g-cross[:, None]*vu, axis=1)
        return np.column_stack((rh, zd))

    ids = np.arange(len(x))
    for _ in range(4):
        r = residual(q, ids)
        s = max(1.4826*np.median(np.abs(r-np.median(r, axis=0))), 1e-7)
        wt = 1/np.hypot(1., np.linalg.norm(r, axis=1)/(2.5*s))
        q = least_squares(lambda p: (residual(p, ids)*wt[:, None]).ravel(), q,
                          max_nfev=40).x
    r = residual(q, ids)
    s = max(1.4826*np.median(np.abs(r-np.median(r, axis=0))), 1e-7)
    inlier = ids[np.linalg.norm(r, axis=1) < 4.25*s]
    if len(inlier) >= 10:
        q = least_squares(lambda p: residual(p, inlier).ravel(), q, max_nfev=40).x
    return _exp(q[:5]), q[5:8]


def calibrate_and_estimate_heading(session: dict) -> dict:
    """Return S_soft, b_hard (µT), and the test headings in degrees."""
    x = np.asarray(session['calib_m_raw'], dtype=float)
    L = _level(np.asarray(session['calib_roll_pitch_rad'], dtype=float))
    origin = np.median(x, axis=0)
    scale = np.median(np.linalg.norm(x-origin, axis=1))
    A, b = _fit((x-origin)/scale, L)
    bias = origin + scale*b
    T = _level(np.asarray(session['test_roll_pitch_rad'], dtype=float))
    m = np.einsum('nij,nj->ni', T, (np.asarray(session['test_m_raw'])-bias) @ A)
    heading = np.degrees(np.arctan2(-m[:, 1], m[:, 0]))
    heading[heading <= -180.] = 180.
    S = np.linalg.inv(A)
    return {'S_soft': (S+S.T)/2, 'b_hard': bias, 'heading_deg': heading}
PY
mkdir -p /tmp/.hegel_candidate_archive && h=$(sha256sum mag_calibrator.py | cut -c1-8) && cp mag_calibrator.py /tmp/.hegel_candidate_archive/v01_$h
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle, numpy as np
import mag_calibrator as c
p=pickle.load(open('public_mag.pkl','rb')); s,g=p['session'],p['gt']; m=s['calib_m_raw']; L=c._level(s['calib_roll_pitch_rad']); LT=c._level(s['test_roll_pitch_rad']); o=np.median(m,axis=0); scale=np.median(np.linalg.norm(m-o,axis=1)); yy=g['gt_heading_deg']
def report(name,A,b):
 z=np.einsum('nij,nj->ni',LT,(s['test_m_raw']-b)@A); d=(np.degrees(np.arctan2(-z[:,1],z[:,0]))-yy+180)%360-180
 print(name, 'rmse/worst quadrant/bias/soft',np.sqrt(np.mean(d*d)),max(np.sqrt(np.mean(d[v]**2)) for v in np.array_split(np.arange(len(d)),4)),np.linalg.norm(b-g['b_hard']),np.linalg.norm(np.linalg.inv(A)-g['S_soft']))
report('oracle',np.linalg.inv(g['S_soft']),g['b_hard'])
for name, ix in [('all',np.ones(len(m),bool)), ('omit +roll',s['calib_roll_pitch_rad'][:,0]<.38), ('omit -pitch',s['calib_roll_pitch_rad'][:,1]>-.33),('inner tilts',np.max(np.abs(s['calib_roll_pitch_rad']),axis=1)<.34)]:
 for joint in [False,True]:
  A,b=c._fit((m[ix]-o)/scale,L[ix],joint); report((name,joint,ix.sum()),A,o+b*scale)
print('API shapes', {k:v.shape for k,v in c.calibrate_and_estimate_heading(s).items()})
PY
```
- **Container Exit Code**: `0` (`654 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/mag_calibrator.py']`
- **Stdout (tail)**:
```text
kage)."""
from __future__ import annotations
import math
import numpy as np
import numpy as np
from scipy.optimize import least_squares
from . import track_wormhole_skill


def _robust_linear(design: np.ndarray, values: np.ndarray, scale: float | None=None) -> tuple[np.ndarray, float]:
    """Return Cauchy IRLS coefficients and residual MAD scale in value units."""
    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    if scale is None:
        residual = values - design @ beta
        scale = max(float(1.4826 * np.median(np.abs(residual - np.median(residual)))), 1e-08)
    for _ in range(12):
        weight = 1.0 / np.hypot(1.0, (values - design @ beta) / scale)
        beta = np.linalg.lstsq(design * weight[:, None], values * weight, rcond=None)[0]
    residual = values - design @ beta
    noise = max(float(1.4826 * np.median(np.abs(residual - np.median(residual)))), 1e-08)
    return (beta, noise)


@track_wormhole_skill("autogenous_creep_corrected_retraction_calibration")
def first_loading_slip(force_N: np.ndarray, specimen_displacement_um: np.ndarray, hold_start: int, hold_end: int, *, threshold_um: float=0.35) -> float | None:
    """Return force at the first persistent positive loading step, or None.
oracle rmse/worst quadrant/bias/soft 0.1310671018690447 0.14200136644126277 0.0 1.576213700060856e-16
('all', False, 420) rmse/worst quadrant/bias/soft 0.14626844485141477 0.16498594168224345 0.047976528718487225 0.0012836990650088808
('all', True, 420) rmse/worst quadrant/bias/soft 0.13290048857319076 0.14819338443677849 0.01680750444332982 0.000819847663394849
('omit +roll', False, 313) rmse/worst quadrant/bias/soft 0.15087613050708434 0.16654889396730843 0.026981006803066747 0.0014547236901789186
('omit +roll', True, 313) rmse/worst quadrant/bias/soft 0.13849101380367085 0.14859973041362212 0.021631818780343426 0.0011875230182111093
('omit -pitch', False, 311) rmse/worst quadrant/bias/soft 0.16554226152280477 0.1954568806968551 0.09867016790203258 0.0017006898542799155
('omit -pitch', True, 311) rmse/worst quadrant/bias/soft 0.1337146991254996 0.15319157710582326 0.03083294215879647 0.0010789976283575456
('inner tilts', False, 84) rmse/worst quadrant/bias/soft 0.34293582882038365 0.4948819560023733 0.25998657337988834 0.005313537320712775
('inner tilts', True, 84) rmse/worst quadrant/bias/soft 0.13518195953711193 0.14094839034384563 0.055875348781302264 0.0017206479698678757
API shapes {'S_soft': (3, 3), 'b_hard': (3,), 'heading_deg': (120,)}
```

### Turn 3 — `execute_in_remote_docker` (`43416 ms` LLM)
- **Rationale**: The public checks favor the joint fit. Test its unchanged implementation on blocked public subsets and independently generated NED rotations to detect convention errors or loss of robustness under tilt restriction, stronger distortion, steep inclination, and correlated spike directions. A failure will identify which regime needs a targeted revision.
- **ATMS NOGOOD DECLARED**: Refuted `The vertical invariant alone gives sufficiently stable heading estimates when calibration tilt coverage is narrow.` | Witness: `On the 84-sample inner-tilt calibration slice, worst heading-quadrant RMSE was 0.495° with the vertical invariant alone versus 0.141° with both invariants.` | Blanket Diagnosis: `With little tilt variation, the down projection weakly constrains horizontal calibration; the yaw-invariant horizontal radius supplies complementary information.` | Sublated Pivot: `Keep the robust vertical initialization and refine with both noise-whitened invariants under the determinant-one constraint.`

```bash
cd /holdout_task && cat >/tmp/mag_stress.py <<'PY'
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from mag_calibrator import calibrate_and_estimate_heading


def metrics(session, soft, bias, yaw):
    out = calibrate_and_estimate_heading(session)
    S, b, h = (out[k] for k in ('S_soft', 'b_hard', 'heading_deg'))
    assert S.shape == (3,3) and b.shape == (3,) and h.shape == yaw.shape
    assert np.all(np.isfinite(np.r_[S.ravel(),b,h]))
    assert np.max(abs(S-S.T)) < 1e-12 and np.linalg.eigvalsh(S)[0] > 0
    assert abs(np.linalg.det(S)-1) < 1e-10 and np.all((-180 < h) & (h <= 180))
    err = (h - yaw + 180) % 360 - 180
    idx = np.clip(np.floor((yaw+180)/90).astype(int), 0, 3)
    return np.array([np.sqrt(np.mean(err**2)), max(np.sqrt(np.mean(err[idx==i]**2)) for i in np.unique(idx)), np.linalg.norm(b-bias), np.linalg.norm(S-soft)])


def report(name, rows):
    a = np.stack(rows)
    print(name, 'cases',len(rows),'max rmse/quadrant/bias/soft:',np.round(a.max(axis=0),6), 'rmse/bias failures:', ((a[:,0]>.45)|(a[:,2]>.35)).sum(), 'worst heading/bias case:',np.argmax(a[:,0]),np.argmax(a[:,2]))


with open('/holdout_task/public_mag.pkl','rb') as f:
    p = pickle.load(f)
s,g = p['session'],p['gt']; rp=s['calib_roll_pitch_rad']
def public(mask):
    return metrics(dict(s,calib_m_raw=s['calib_m_raw'][mask],calib_roll_pitch_rad=rp[mask]),g['S_soft'],g['b_hard'],g['gt_heading_deg'])
report('public all',[public(slice(None))])
report('public leave-one-tilt-cell-out',[public(~((np.digitize(rp[:,0],[-.25,.25])==i)&(np.digitize(rp[:,1],[-.22,.22])==j))) for i in range(3) for j in range(3)])
report('public calibrate on contiguous thirds',[public(ids) for ids in np.array_split(np.arange(len(rp)),3)])


def simulate(seed, mode):
    rng=np.random.default_rng(seed)
    N, extent, axes, field, noise, spike = {
        'nominal':(420,[.75,.65],[1.22,.96,1/1.1712],[31,41],.08,20),
        'short-narrow-tilt':(140,[.35,.28],[1.22,.96,1/1.1712],[29,42],.10,25),
        'steep-field-and-distortion':(420,[.75,.65],[1.65,1.03,1/1.6995],[18,49],.09,35),
        'clustered-yaw-and-spikes':(300,[.60,.50],[1.35,1.05,1/1.4175],[26,-41],.09,50),
    }[mode]
    U=Rotation.random(random_state=rng).as_matrix(); S=(U*np.array(axes))@U.T
    b=rng.uniform(-55,55,3); B=np.array([field[0],0,field[1]])
    def samples(rp, yaw):
        R=Rotation.from_euler('ZYX',np.column_stack((yaw,rp[:,1],rp[:,0])))
        return R.inv().apply(np.broadcast_to(B,(len(rp),3)))@S + b + rng.normal(0,noise,(len(rp),3))
    rp=rng.uniform(-1,1,(N,2))*extent; yaw=rng.uniform(-np.pi,np.pi,N)
    if mode=='clustered-yaw-and-spikes':
        yaw[:int(N*.7)] = rng.normal(.8,.28,int(N*.7))
    m=samples(rp,yaw); k=round(N*.04)
    ids=rng.choice(N,k,replace=False); direction=rng.normal(size=(1 if mode=='clustered-yaw-and-spikes' else k,3)); direction/=np.linalg.norm(direction,axis=1,keepdims=True)
    m[ids]+=direction*rng.uniform(.3,1.7,(k,1))*spike
    tp=rng.uniform(-1,1,(160,2))*[.45,.38]
    ty=np.linspace(-np.pi,np.pi,160,endpoint=False)
    return metrics(dict(calib_m_raw=m,calib_roll_pitch_rad=rp,test_m_raw=samples(tp,ty),test_roll_pitch_rad=tp),S,b,np.degrees(ty))

for mode in ['nominal','short-narrow-tilt','steep-field-and-distortion','clustered-yaw-and-spikes']:
    report(mode,[simulate(seed,mode) for seed in range(24)])
PY
PYTHONPATH=/holdout_task OPENBLAS_NUM_THREADS=1 python3 /tmp/mag_stress.py
```
- **Container Exit Code**: `0` (`4096 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public all cases 1 max rmse/quadrant/bias/soft: [0.1329   0.146988 0.016808 0.00082 ] rmse/bias failures: 0 worst heading/bias case: 0 0
public leave-one-tilt-cell-out cases 9 max rmse/quadrant/bias/soft: [0.134645 0.148539 0.02577  0.001015] rmse/bias failures: 0 worst heading/bias case: 8 2
public calibrate on contiguous thirds cases 3 max rmse/quadrant/bias/soft: [0.138572 0.162466 0.053735 0.001967] rmse/bias failures: 0 worst heading/bias case: 0 0
nominal cases 24 max rmse/quadrant/bias/soft: [0.181394 0.204162 0.03981  0.000893] rmse/bias failures: 0 worst heading/bias case: 1 19
short-narrow-tilt cases 24 max rmse/quadrant/bias/soft: [0.230357 0.273809 0.173645 0.002991] rmse/bias failures: 0 worst heading/bias case: 1 0
steep-field-and-distortion cases 24 max rmse/quadrant/bias/soft: [0.425574 0.519408 0.064128 0.001427] rmse/bias failures: 0 worst heading/bias case: 1 19
clustered-yaw-and-spikes cases 24 max rmse/quadrant/bias/soft: [0.246914 0.287024 0.097698 0.001856] rmse/bias failures: 0 worst heading/bias case: 10 8
```

### Turn 4 — `execute_in_remote_docker` (`34431 ms` LLM)
- **Rationale**: Load the exact artifact in a clean process to detect undeclared dependencies. Extend only the steep-field slice, which had the smallest margin, and compare with true calibration to distinguish a correctable fitting error from test measurement noise.
- **Alternatives Weighed**: `Smooth consecutive test headings` (rejected: The interface specifies no motion model or sampling cadence; smoothing can distort legitimate turns.)

```bash
cd /holdout_task && env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import hashlib, importlib.util, pathlib, pickle
import numpy as np
from scipy.spatial.transform import Rotation
path = pathlib.Path('/holdout_task/mag_calibrator.py')
print('verified artifact:', hashlib.sha256(path.read_bytes()).hexdigest())
spec = importlib.util.spec_from_file_location('mag_calibrator', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def check(s, S, b, yaw):
    out = mod.calibrate_and_estimate_heading(s)
    A, B, h = (out[k] for k in ('S_soft','b_hard','heading_deg'))
    assert A.shape == (3,3) and B.shape == (3,) and h.shape == yaw.shape
    assert all(np.all(np.isfinite(z)) for z in (A,B,h))
    assert np.allclose(A,A.T,atol=1e-12,rtol=0) and np.linalg.eigvalsh(A)[0]>0
    assert abs(np.linalg.det(A)-1)<1e-10 and np.all((-180<h)&(h<=180))
    def rmse(a):
        d = (a-yaw+180)%360-180
        return np.sqrt(np.mean(d*d))
    rp = s['test_roll_pitch_rad']; z = Rotation.from_euler('YX', rp[:,::-1]).apply(np.linalg.solve(S, (s['test_m_raw']-b).T).T)
    return rmse(h), np.linalg.norm(B-b), rmse(np.degrees(np.arctan2(-z[:,1],z[:,0])))

with open('/holdout_task/public_mag.pkl','rb') as f:
    p=pickle.load(f)
g=p['gt']; real=check(p['session'],g['S_soft'],g['b_hard'],g['gt_heading_deg'])
print('public: heading RMSE / bias error / oracle RMSE', real)
assert real[0]<=.45 and real[1]<=.35

rows=[]
for seed in range(96,160):
    r=np.random.default_rng(seed)
    U=Rotation.random(random_state=r).as_matrix(); S=(U*np.array([1.65,1.03,1/1.6995]))@U.T
    b=r.uniform(-55,55,3); field=[18,0,49]
    def raw(rp,yaw):
        att=Rotation.from_euler('ZYX',np.column_stack((yaw,rp[:,1],rp[:,0])))
        return att.inv().apply(np.broadcast_to(field,(len(rp),3)))@S+b+r.normal(0,.09,(len(rp),3))
    rp=r.uniform(-1,1,(420,2))*[.75,.65]; y=r.uniform(-np.pi,np.pi,420)
    m=raw(rp,y); ids=r.choice(len(m),17,replace=False)
    d=r.normal(size=(17,3)); d/=np.linalg.norm(d,axis=1,keepdims=True)
    m[ids]+=d*r.uniform(.3,1.7,(17,1))*35
    tp=r.uniform(-1,1,(160,2))*[.45,.38]; ty=np.linspace(-np.pi,np.pi,160,endpoint=False)
    rows.append(check(dict(calib_m_raw=m,calib_roll_pitch_rad=rp,test_m_raw=raw(tp,ty),test_roll_pitch_rad=tp),S,b,np.degrees(ty)))
a=np.array(rows)
print('steep-field extension maxima (heading, bias, oracle):',a.max(axis=0))
print('largest RMSE excess over oracle:',np.max(a[:,0]-a[:,2]))
print('over target (seed, heading, bias, oracle):',[(i+96,*a[i]) for i in np.flatnonzero((a[:,0]>.45)|(a[:,1]>.35))])
PY
```
- **Container Exit Code**: `0` (`3018 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
verified artifact: c2e97573a1adbb8cab3e2a1a7815669560248385d95a149184b1b70b3f87554b
public: heading RMSE / bias error / oracle RMSE (0.13290048857319076, 0.01680750444332982, 0.13106710186904508)
steep-field extension maxima (heading, bias, oracle): [0.43857211 0.10269663 0.43339663]
largest RMSE excess over oracle: 0.011274182758251705
over target (seed, heading, bias, oracle): []
```

### Turn 5 — `submit_completed_task` (`22114 ms` LLM)
- **Rationale**: The joint fit met both targets in all tested public and synthetic sessions. The isolated verification confirmed the output schema, heading range, and symmetric positive-definite soft-iron matrix with unit determinant. In the steep-field extension, calibration added at most 0.012° RMSE over the true-calibration baseline.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`17` turns remaining; artifact status={'/holdout_task/mag_calibrator.py': 'FILE:4487'}): Do NOT lock prematurely with `17` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/mag_calibrator.py`. On the public session, circular heading RMSE was 0.133° and hard-iron bias error was 0.017 µT, both below their targets.`

### Turn 6 — `execute_in_remote_docker` (`38313 ms` LLM)
- **Rationale**: Independently verify the leveling convention and invariant identity, then check whether contiguous held-out calibration blocks show coherent shifts rather than isolated spikes. A persistent shift coupled with degraded heading or bias would justify investigating drift; otherwise retain the fixed-calibration model. Compare the file directly with the archived baseline to resolve the conflicting artifact identifiers.
- **Alternatives Weighed**: `Add quotient terms, relay logic, or batch-centered classification` (rejected: This session has no relay state, class labels, or batch metadata. The applicable algebraic checks are the rotation convention, norm decomposition, and unit-determinant constraint.)

```bash
cd /holdout_task && env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

path = Path('/holdout_task/mag_calibrator.py')
spec = importlib.util.spec_from_file_location('mag_calibrator', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
archives = sorted(Path('/tmp/.hegel_candidate_archive').glob('v01_*'))
print('baseline archive byte match:', [(p.name, p.read_bytes() == path.read_bytes()) for p in archives])
with open('public_mag.pkl', 'rb') as f:
    p = pickle.load(f)
s, gt = p['session'], p['gt']
x, rp = s['calib_m_raw'], s['calib_roll_pitch_rad']
L = m._level(rp)
ref = Rotation.from_euler('YX', rp[:, ::-1]).as_matrix()
np.testing.assert_allclose(L, ref, atol=1e-14, rtol=0)
assert len(x) == len(rp)

def check(fit):
    out = m.calibrate_and_estimate_heading(dict(s, calib_m_raw=x[fit], calib_roll_pitch_rad=rp[fit]))
    S, b, heading = (out[k] for k in ('S_soft', 'b_hard', 'heading_deg'))
    assert S.shape == (3, 3) and b.shape == (3,) and heading.shape == gt['gt_heading_deg'].shape
    assert np.isfinite(S).all() and np.isfinite(b).all() and np.isfinite(heading).all()
    np.testing.assert_allclose(S, S.T, atol=1e-12, rtol=0)
    assert np.linalg.eigvalsh(S)[0] > 0 and abs(np.linalg.det(S) - 1) < 1e-10
    assert np.all((-180 < heading) & (heading <= 180))
    e = (heading - gt['gt_heading_deg'] + 180) % 360 - 180
    return S, b, float(np.sqrt(np.mean(e**2))), float(np.linalg.norm(b - gt['b_hard']))

def invariants(S, b, ids):
    u = np.linalg.solve(S, (x[ids] - b).T).T
    v = np.einsum('nij,nj->ni', L[ids], u)
    h2 = v[:, 0]**2 + v[:, 1]**2
    np.testing.assert_allclose(h2, np.sum(u*u, axis=1) - v[:, 2]**2, atol=1e-10, rtol=1e-12)
    return np.column_stack((np.sqrt(h2), v[:, 2]))

all_ids = np.arange(len(x))
_, _, rmse, bias = check(all_ids)
print('public full: heading RMSE / bias error', rmse, bias)
assert rmse <= .45 and bias <= .35
rows = []
for block in np.array_split(all_ids, 6):
    fit = np.setdiff1d(all_ids, block)
    S, b, rmse, bias = check(fit)
    z = invariants(S, b, fit)
    center = np.median(z, axis=0)
    mad = 1.4826*np.median(np.abs(z-center), axis=0)
    e = invariants(S, b, block) - center
    shift = np.median(e, axis=0)/mad
    spikes = np.count_nonzero(np.max(np.abs(e)/mad, axis=1) > 5)
    rows.append((rmse, bias, np.max(np.abs(shift))))
    print(f'block {block[0]:3d}–{block[-1]:3d}: heading={rmse:.5f} deg bias={bias:.5f} uT median shift/MAD={shift.round(3)} spikes={spikes}')
a = np.asarray(rows)
print('worst blocked heading / bias / normalized systematic shift:', a.max(axis=0))
assert np.all(a[:, 0] <= .45) and np.all(a[:, 1] <= .35)
PY
```
- **Container Exit Code**: `1` (`470 ms`, timed_out=`False`)
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 13, in <module>
  File "/usr/lib/python3.12/pathlib.py", line 1021, in read_bytes
    with self.open(mode='rb') as f:
         ^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1015, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IsADirectoryError: [Errno 21] Is a directory: '/tmp/.hegel_candidate_archive/v01_080b56a1'
```

### Turn 7 — `execute_in_remote_docker` (`34724 ms` LLM)
- **Rationale**: The prior audit stopped at an archive directory before testing the measurements. Check archive layout without making it a prerequisite, then finish the isolated audit. A rotation or invariant mismatch would require a convention fix; coherent held-out residual shifts coupled with degraded heading or bias would justify investigating drift.

```bash
cd /holdout_task && env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

path = Path('/holdout_task/mag_calibrator.py')
spec = importlib.util.spec_from_file_location('mag_calibrator', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
archive = Path('/tmp/.hegel_candidate_archive/v01_080b56a1')
candidates = sorted(archive.rglob('mag_calibrator.py')) if archive.is_dir() else ([archive] if archive.is_file() else [])
print('host archive direct file matches:', [(str(p), p.read_bytes() == path.read_bytes()) for p in candidates])
with open('public_mag.pkl', 'rb') as f:
    data = pickle.load(f)
s, gt = data['session'], data['gt']
x, rp = s['calib_m_raw'], s['calib_roll_pitch_rad']
L = Rotation.from_euler('YX', rp[:, ::-1]).as_matrix()
np.testing.assert_allclose(mod._level(rp), L, atol=1e-14, rtol=0)
ids = np.arange(len(x))

def check(fit):
    out = mod.calibrate_and_estimate_heading(dict(s, calib_m_raw=x[fit], calib_roll_pitch_rad=rp[fit]))
    S, b, h = (out[k] for k in ('S_soft', 'b_hard', 'heading_deg'))
    assert S.shape == (3, 3) and b.shape == (3,) and h.shape == gt['gt_heading_deg'].shape
    assert all(np.isfinite(a).all() for a in (S, b, h))
    np.testing.assert_allclose(S, S.T, atol=1e-12, rtol=0)
    assert np.linalg.eigvalsh(S)[0] > 0 and abs(np.linalg.det(S) - 1) < 1e-10
    assert np.all((-180 < h) & (h <= 180))
    err = (h - gt['gt_heading_deg'] + 180) % 360 - 180
    return S, b, np.sqrt(np.mean(err**2)), np.linalg.norm(b - gt['b_hard'])

def invariants(S, b):
    u = np.linalg.solve(S, (x - b).T).T
    v = np.einsum('nij,nj->ni', L, u)
    radius = np.hypot(v[:, 0], v[:, 1])
    np.testing.assert_allclose(radius**2, np.sum(u*u, axis=1) - v[:, 2]**2, atol=1e-10, rtol=1e-12)
    return np.column_stack((radius, v[:, 2]))

full = check(ids)[2:]
print('public full: heading RMSE / bias error:', full)
assert full[0] <= .45 and full[1] <= .35
rows = []
for block in np.array_split(ids, 6):
    fit = np.setdiff1d(ids, block)
    S, b, rmse, bias = check(fit)
    z = invariants(S, b)
    center = np.median(z[fit], axis=0)
    mad = 1.4826*np.median(abs(z[fit]-center), axis=0)
    resid = (z[block]-center)/mad
    shift = np.median(resid, axis=0)
    rows.append((rmse, bias, np.max(abs(shift))))
    print(f'block {block[0]:3d}-{block[-1]:3d}: heading={rmse:.5f} deg bias={bias:.5f} uT median shift/MAD={shift.round(3)} spikes={np.count_nonzero(np.max(abs(resid), axis=1)>5)}')
a = np.array(rows)
print('worst blocked heading / bias / absolute median shift in MAD units:', a.max(axis=0))
assert np.all(a[:, 0] <= .45) and np.all(a[:, 1] <= .35)
PY
```
- **Container Exit Code**: `0` (`763 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
host archive direct file matches: [('/tmp/.hegel_candidate_archive/v01_080b56a1/mag_calibrator.py', True)]
public full: heading RMSE / bias error: (0.13290048857319076, 0.01680750444332982)
block   0- 69: heading=0.13172 deg bias=0.01510 uT median shift/MAD=[-0.128  0.127] spikes=1
block  70-139: heading=0.13334 deg bias=0.01894 uT median shift/MAD=[0.114 0.055] spikes=4
block 140-209: heading=0.13337 deg bias=0.01846 uT median shift/MAD=[ 0.182 -0.265] spikes=3
block 210-279: heading=0.13395 deg bias=0.02458 uT median shift/MAD=[-0.093  0.014] spikes=4
block 280-349: heading=0.13359 deg bias=0.02328 uT median shift/MAD=[ 0.018 -0.112] spikes=3
block 350-419: heading=0.13254 deg bias=0.01722 uT median shift/MAD=[-0.211  0.044] spikes=1
worst blocked heading / bias / absolute median shift in MAD units: [0.13394737 0.02457826 0.26513676]
```

### Turn 8 — `submit_completed_task` (`15466 ms` LLM)
- **Rationale**: The isolated audit verified the archived artifact, the NED leveling convention, the wrapped heading range, and the symmetric positive-definite soft-iron matrix with unit determinant. Both targets held across all tested public and synthetic sessions.
- **Alternatives Weighed**: `Add time-dependent bias correction` (rejected: Across six held-out time blocks, the largest median residual shift was 0.27 MAD units and both error metrics remained within tolerance.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/mag_calibrator.py': 'FILE:4487'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/mag_calibrator.py`. On the public session, circular heading RMSE was 0.133° and hard-iron bias error was 0.017 µT, both below the requested limits of 0.45° and 0.35 µT.`

### Turn 9 — `execute_in_remote_docker` (`31520 ms` LLM)
- **Rationale**: Hold out rotating 90° sectors of the public calibration measurements to test whether the joint invariants recover the calibration when azimuth coverage is uneven. Report errors inside the missing sector separately from the full-session targets. A failure would warrant checking fit conditioning and initialization before changing the physical model; a pass supports keeping the verified artifact.
- **Alternatives Weighed**: `Repeat the time-block audit or fit a batch adjustment` (rejected: The prior audit found a maximum median shift of 0.27 MAD units with both targets met, and the session provides no batch variable. Missing yaw coverage tests a different, relevant failure mode.)

```bash
cd /holdout_task && env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

path = Path('/holdout_task/mag_calibrator.py')
spec = importlib.util.spec_from_file_location('mag_calibrator', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('public_mag.pkl', 'rb') as f:
    d = pickle.load(f)
s, gt = d['session'], d['gt']
x, rp = s['calib_m_raw'], s['calib_roll_pitch_rad']
# Use the public reference only to define audit slices, never as fit input.
L = Rotation.from_euler('YX', rp[:, ::-1])
u = np.linalg.solve(gt['S_soft'], (x - gt['b_hard']).T).T
v = L.apply(u)
yaw = np.degrees(np.arctan2(-v[:, 1], v[:, 0]))
y_test = gt['gt_heading_deg']
rows = []
for center in np.arange(-180, 180, 45):
    keep = np.abs((yaw-center+180)%360-180) >= 45
    out = mod.calibrate_and_estimate_heading(dict(s, calib_m_raw=x[keep], calib_roll_pitch_rad=rp[keep]))
    S, b, h = (out[k] for k in ('S_soft', 'b_hard', 'heading_deg'))
    assert S.shape == (3, 3) and b.shape == (3,) and h.shape == y_test.shape
    assert all(np.isfinite(z).all() for z in (S, b, h))
    assert np.allclose(S, S.T, atol=1e-12, rtol=0) and np.linalg.eigvalsh(S)[0] > 0
    assert abs(np.linalg.det(S)-1) < 1e-10 and np.all((-180<h)&(h<=180))
    err = (h-y_test+180)%360-180
    sector = np.abs((y_test-center+180)%360-180) < 45
    row = np.sqrt(np.mean(err**2)), np.sqrt(np.mean(err[sector]**2)), np.linalg.norm(b-gt['b_hard'])
    rows.append(row)
    print(f'omit center={center:4d} train={keep.sum():3d} test in gap={sector.sum():2d} full/gap RMSE={row[0]:.5f}/{row[1]:.5f} deg bias={row[2]:.5f} uT')
a = np.array(rows)
print('worst full-heading RMSE / yaw-gap RMSE / bias error:', a.max(axis=0))
print('slices exceeding 0.45 deg full-session or 0.35 uT:', np.flatnonzero((a[:,0]>.45)|(a[:,2]>.35)).tolist())
PY
```
- **Container Exit Code**: `0` (`745 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
omit center=-180 train=310 test in gap=28 full/gap RMSE=0.13774/0.13764 deg bias=0.02096 uT
omit center=-135 train=314 test in gap=29 full/gap RMSE=0.13365/0.13374 deg bias=0.02028 uT
omit center= -90 train=309 test in gap=31 full/gap RMSE=0.13042/0.14136 deg bias=0.04282 uT
omit center= -45 train=321 test in gap=31 full/gap RMSE=0.13158/0.14646 deg bias=0.02248 uT
omit center=   0 train=328 test in gap=30 full/gap RMSE=0.13340/0.13627 deg bias=0.02547 uT
omit center=  45 train=314 test in gap=31 full/gap RMSE=0.13344/0.13012 deg bias=0.02345 uT
omit center=  90 train=313 test in gap=31 full/gap RMSE=0.13307/0.13127 deg bias=0.03287 uT
omit center= 135 train=311 test in gap=29 full/gap RMSE=0.13429/0.12074 deg bias=0.02089 uT
worst full-heading RMSE / yaw-gap RMSE / bias error: [0.1377418  0.14646136 0.04281825]
slices exceeding 0.45 deg full-session or 0.35 uT: []
```

### Turn 10 — `submit_completed_task` (`12138 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated API, rotation, heading-range, and unit-determinant checks. Both targets held across the public and synthetic evaluations, including all eight public yaw-sector omissions. The steep-field synthetic runs were close to the error obtained with true calibration, so the evidence does not support adding model complexity.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/mag_calibrator.py`. On the public session, circular heading RMSE was 0.133° (limit: 0.45°), and hard-iron bias error was 0.017 µT (limit: 0.35 µT).`

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
