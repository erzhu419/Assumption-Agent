# Dialectical Synthesis Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_2_Selective_LiveGPT` | `ho01__crosswell_anisotropic_clock_tomography`

- **Official Strict Pass**: **`PASS (1.0)`** (`PASSED_ALL_CHECKS`)
- **Blind Public Synthesis Choice**: `antithesis` via `selective_router_confident_single_branch`
- **Synthesis Rationale**: Primary Antithesis branch satisfied public checks under policy; second branch skipped (`1.00u` nominal units).
- **Post-Lock Branch Verdicts**: Thesis (`Bare`) = `NOT_SPAWNED (0T)` | Antithesis (`Wormhole`) = `True` (17T)

## Selected Branch Trajectory Record
# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `ho01__crosswell_anisotropic_clock_tomography`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-OOD-Holdout-12` |
| **Domain** | `Geophysics / Borehole Traveltime Tomography` |
| **Wave / Wormhole Family** | `wave_1` / `fam_guided_wave_shm_localization` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `17 / 18` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `2` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `576,708` (`557,341` / `121,088` / `19,367`) |
| **Total Episode Wall Time (s)** | `834.41s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`16327 ms` LLM)
- **Rationale**: Inspect field names, dimensions, and calibration metadata before deciding which background parameters to fit.

```bash
python - <<'PY'
import pickle, numpy as np
with open('/holdout_task/public_surveys.pkl','rb') as f: x=pickle.load(f)
def show(x,d=0):
 p='  '*d
 if isinstance(x,dict):
  for k,v in x.items():
   print(p,repr(k)); show(v,d+1)
 elif isinstance(x,(tuple,list)) and len(x) and isinstance(x[0],(tuple,list,dict,np.ndarray)):
  print(p,type(x).__name__,len(x));
  for a in x[:5]: show(a,d+1)
 elif isinstance(x,np.ndarray): print(p,x.shape,x.dtype,'range', (np.nanmin(x),np.nanmax(x)) if x.size and np.issubdtype(x.dtype,np.number) else None,'head',x.ravel()[:6])
 else: print(p,repr(x)[:500])
show(x)
PY
```
- **Container Exit Code**: `127` (`303 ms`, timed_out=`False`)
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`19296 ms` LLM)
- **Rationale**: The previous command failed because `python` was unavailable. Retry with `python3` and inspect a compact survey summary to determine which background quantities must be estimated and whether public truth is available for scoring.

```bash
command -v python3; ls -la /holdout_task; python3 - <<'PY'
import pickle, numpy as np, scipy
print('numpy', np.__version__, 'scipy', scipy.__version__)
with open('/holdout_task/public_surveys.pkl','rb') as f:
    obj = pickle.load(f)
def show(x, depth=0):
    p = '  ' * depth
    if isinstance(x, dict):
        for k, v in x.items():
            print(p + str(k) + ':'); show(v, depth+1)
    elif isinstance(x, (list, tuple)) and (not x or isinstance(x[0], (dict, list, tuple, np.ndarray))):
        print(p, type(x).__name__, len(x))
        for v in x[:5]: show(v, depth+1)
    else:
        a = np.asarray(x)
        if a.ndim and a.size and np.issubdtype(a.dtype, np.number):
            print(p, a.shape, a.dtype, 'range', (np.nanmin(a), np.nanmax(a)), 'head', a.ravel()[:8])
        else:
            print(p, repr(x)[:400])
show(obj)
PY
```
- **Container Exit Code**: `0` (`577 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/usr/bin/python3
total 16
drwxr-xr-x 2 root root 4096 Nov 25 15:29 .
drwxr-xr-x 1 root root 4096 Nov 25 15:29 ..
-rw-r--r-- 1 root root 5205 Nov 25 15:29 public_surveys.pkl
numpy 1.26.4 scipy 1.13.0
 list 2
  survey:
    W_m:
       40.0
    H_m:
       80.0
    z_src_m:
       (15,) float64 range (6.0, 74.0) head [ 6.         10.85714286 15.71428571 20.57142857 25.42857143 30.28571429
 35.14285714 40.        ]
    z_rcv_m:
       (16,) float64 range (5.0, 75.0) head [ 5.          9.66666667 14.33333333 19.         23.66666667 28.33333333
 33.         37.66666667]
    T_obs_s:
       (15, 16) float64 range (0.01330233416627665, 0.03317839066777648) head [0.01491584 0.01729941 0.01816543 0.0148971  0.01612493 0.02175942
 0.01714121 0.01843184]
    R_plume_m:
       5.5
  gt:
    plume_xz_m:
       (2,) float64 range (16.5, 34.0) head [16.5 34. ]
    rcv_clock_bias_s:
       (16,) float64 range (-0.002556039023045965, 0.003329653424440951) head [-0.0019168   0.00045573  0.00110994 -0.00255604 -0.00179292  0.0032009
 -0.00228896 -0.0019091 ]
  survey:
    W_m:
       40.0
    H_m:
       80.0
    z_src_m:
       (15,) float64 range (6.0, 74.0) head [ 6.         10.85714286 15.71428571 20.57142857 25.42857143 30.28571429
 35.14285714 40.        ]
    z_rcv_m:
       (16,) float64 range (5.0, 75.0) head [ 5.          9.66666667 14.33333333 19.         23.66666667 28.33333333
 33.         37.66666667]
    T_obs_s:
       (15, 16) float64 range (0.01336071363590472, 0.028705721115872333) head [0.01425288 0.0187468  0.01403325 0.0142902  0.01580415 0.01554096
 0.01899251 0.0161894 ]
    R_plume_m:
       5.5
  gt:
    plume_xz_m:
       (2,) float64 range (26.0, 51.5) head [26.  51.5]
    rcv_clock_bias_s:
       (16,) float64 range (-0.002568385497994811, 0.0034727403303111385) head [-0.0009812   0.00347274 -0.00137483 -0.00143901 -0.00034719 -0.00111101
  0.00170437 -0.00184997]
```

### Turn 3 — `execute_in_remote_docker` (`54819 ms` LLM)
- **Rationale**: Score multiple plume basins after projecting out the exact linear clock and background subspace, then refit without gross skips. Public truth will distinguish localization and clock errors; residuals will show whether skip removal is warranted, and default-thread timing will detect runtime risk.
- **Alternatives Weighed**: `Receiver-mean subtraction before fitting` (rejected: It spreads each gross pick error over the receiver column and removes the column mean of the plume signal.); `Reuse the waveform-localization skill` (rejected: These surveys contain traveltime picks, not waveforms or reference records.)

```bash
cat > /holdout_task/tomography_solver.py <<'PY'
"""Straight-ray inversion of cross-well first arrivals."""
import numpy as np
from scipy.linalg import helmert, qr
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr


class _Rays:
    def __init__(self, width, source, receiver, radius):
        dz = receiver[None, :] - source[:, None]
        length = np.hypot(width, dz)
        self.ux = (width / length).ravel()
        self.uz = (dz / length).ravel()
        self.source = np.broadcast_to(source[:, None], dz.shape).ravel()
        self.length = length.ravel()
        self.radius = radius
        self.clocks = helmert(receiver.size, full=False).T
        self.design = np.column_stack((self.length, -self.length * self.uz**2,
                                       np.tile(self.clocks, (source.size, 1))))

    def plume(self, x, z, derivative=False):
        r = self.radius
        dz = z - self.source
        t = (self.ux * x + self.uz * dz) / r
        p = (self.ux * dz - self.uz * x) / r
        end = self.length / r - t
        envelope = np.exp(-0.5 * p**2)
        path = np.sqrt(2 * np.pi) * r * envelope * (ndtr(end) - ndtr(-t))
        if not derivative:
            return path
        along = envelope * (np.exp(-0.5 * t**2) - np.exp(-0.5 * end**2))
        normal = -p / r * path
        return path, along * self.ux - normal * self.uz, along * self.uz + normal * self.ux


def _linear(design, y, weights):
    return np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)[0]


def _fit(survey):
    w, h = float(survey['W_m']), float(survey['H_m'])
    src, rcv = (np.asarray(survey[k], dtype=float) for k in ('z_src_m', 'z_rcv_m'))
    rays = _Rays(w, src, rcv, float(survey.get('R_plume_m', 5.5)))
    y = 1000 * np.asarray(survey['T_obs_s'], dtype=float).ravel()
    a = rays.design
    lo = np.r_[0., 0., 0., np.full(a.shape[1], -np.inf)]
    hi = np.r_[w, h, np.full(a.shape[1] + 1, np.inf)]

    def prediction(p):
        return a @ p[3:] + p[2] * rays.plume(*p[:2])

    def jac(p):
        g, gx, gz = rays.plume(*p[:2], derivative=True)
        return np.column_stack((p[2] * gx, p[2] * gz, g, a))

    def refine(p, target, keep, scale=None):
        return least_squares(lambda b: (prediction(b) - target)[keep], p,
                             jac=lambda b: jac(b)[keep], bounds=(lo, hi),
                             x_scale='jac', loss='linear' if scale is None else 'soft_l1',
                             f_scale=1. if scale is None else scale,
                             max_nfev=60, ftol=1e-9).x

    bg = _linear(a, y, np.ones(y.size))
    for _ in range(12):
        e = y - a @ bg
        weights = (1 + (e / .38)**2)**(-.25)
        bg = _linear(a, y, weights)

    # Project both data and candidate plume signatures through the same
    # weighted background/clock null space before scoring the coarse grid.
    q = qr(a * weights[:, None], mode='economic')[0]
    gy = y * weights
    gy -= q @ (q.T @ gy)
    d = rays.radius / 2.5
    xy = np.array([(x, z) for x in np.linspace(0, w, int(np.ceil(w/d))+1)
                   for z in np.linspace(0, h, int(np.ceil(h/d))+1)])
    paths = rays.plume(xy[:, :1], xy[:, 1:]).T
    g = paths * weights[:, None]
    g -= q @ (q.T @ g)
    dots = gy @ g
    norm = np.maximum(np.sum(g*g, axis=0), 1e-30)
    merit = np.maximum(dots, 0)**2 / norm
    full = np.ones(y.size, dtype=bool)
    starts = []
    candidates = []
    for i in np.argsort(-merit):
        if any(np.linalg.norm(xy[i] - s) < rays.radius for s in starts):
            continue
        starts.append(xy[i])
        amp = max(dots[i] / norm[i], .001)
        p = np.r_[xy[i], amp, _linear(a, y - amp * paths[:, i], weights)]
        p = refine(p, y, full, .38)
        # Large skips exert no pull while choosing the physical basin.
        keep = np.abs(y - prediction(p)) < 2.
        p = refine(p, y, keep)
        e = y - prediction(p)
        candidates.append((np.minimum(e*e, 4.).sum(), p))
        if len(candidates) == 6:
            break
    p = min(candidates, key=lambda item: item[0])[1]
    return rays, p, y - prediction(p)


def invert_plume_and_clocks(survey: dict) -> dict:
    """Estimate plume center (m) and zero-sum receiver offsets (s)."""
    rays, p, _ = _fit(survey)
    clocks = rays.clocks @ p[5:]
    if np.max(np.abs(clocks)) > 4.:
        shift = brentq(lambda c: np.clip(clocks - c, -4., 4.).sum(),
                       clocks.min() - 4., clocks.max() + 4.)
        clocks = np.clip(clocks - shift, -4., 4.)
    return {'plume_xz_m': p[:2].tolist(), 'rcv_clock_bias_s': (clocks / 1000).tolist()}
PY
cat > /tmp/public_tomo.py <<'PY'
import pickle,time
import numpy as np
from tomography_solver import _fit,invert_plume_and_clocks
samples=pickle.load(open('/holdout_task/public_surveys.pkl','rb'))
for i,sample in enumerate(samples):
    s,gt=sample['survey'],sample['gt']; t=time.monotonic(); r=invert_plume_and_clocks(s); elapsed=time.monotonic()-t
    rays,p,e=_fit(s); good=abs(e)<2; skip=e[~good]; print(i,'params',p[:5], 'actual scipy time',elapsed,'errors',np.linalg.norm(np.subtract(r['plume_xz_m'],gt['plume_xz_m'])), np.sqrt(np.mean(np.subtract(r['rcv_clock_bias_s'],gt['rcv_clock_bias_s'])**2)),'clean rms',np.sqrt(np.mean(e[good]**2)),'skips',np.round(skip,4),'clock sum',sum(r['rcv_clock_bias_s']))
PY
PYTHONPATH=/holdout_task OPENBLAS_NUM_THREADS=1 python3 /tmp/public_tomo.py
PYTHONPATH=/holdout_task python3 /tmp/public_tomo.py
```
- **Container Exit Code**: `0` (`2644 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tomography_solver.py']`
- **Stdout (tail)**:
```text
0 params [16.42106576 34.04212498  0.06490051  0.42012556  0.05911114] actual scipy time 0.039531763177365065 errors 0.08947138398760496 5.301981775131142e-06 clean rms 0.03190903709285461 skips [-5.5062  5.5027  5.4619  5.4507  5.562   5.4965 -5.4956  5.5203 -5.528
 -5.4997] clock sum 5.421010862427522e-19
1 params [25.89839669 51.45256973  0.07134281  0.38011303  0.06862344] actual scipy time 0.03605680982582271 errors 0.11212878064897099 1.2234518105508974e-05 clean rms 0.031614120381364616 skips [-5.5068  5.4651  5.5166 -5.4685  5.5104  5.5262 -5.5114  5.5144  5.5169
  5.493 ] clock sum 8.402566836762659e-19
0 params [16.42106576 34.04212498  0.06490051  0.42012556  0.05911114] actual scipy time 0.32128256000578403 errors 0.08947138398760496 5.301981775131142e-06 clean rms 0.03190903709285461 skips [-5.5062  5.5027  5.4619  5.4507  5.562   5.4965 -5.4956  5.5203 -5.528
 -5.4997] clock sum 5.421010862427522e-19
1 params [25.89839669 51.45256973  0.07134281  0.38011303  0.06862344] actual scipy time 0.29629927687346935 errors 0.11212878064897099 1.2234518105508974e-05 clean rms 0.031614120381364616 skips [-5.5068  5.4651  5.5166 -5.4685  5.5104  5.5262 -5.5114  5.5144  5.5169
  5.493 ] clock sum 8.402566836762659e-19
```

### Turn 4 — `execute_in_remote_docker` (`44757 ms` LLM)
- **Rationale**: Sensor deletion tests whether a few depths dominate recovery. Independent quadrature stresses the exact ray model where plume and receiver clocks are hardest to separate; failures would identify whether to improve basin selection or examine geometric identifiability.

```bash
cat > /tmp/stress_tomo.py <<'PY'
import pickle
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from tomography_solver import _fit, invert_plume_and_clocks

samples = pickle.load(open('/holdout_task/public_surveys.pkl', 'rb'))

def errors(xz, clocks, gt_xz, gt_clocks):
    return np.linalg.norm(np.subtract(xz, gt_xz)), np.sqrt(np.mean(np.subtract(clocks, gt_clocks)**2))

for i, item in enumerate(samples):
    survey, gt = item['survey'], item['gt']
    t, clean_predictions = np.asarray(survey['T_obs_s']), []
    worst = [[], []]
    for j in range(t.shape[0]):
        keep = np.arange(t.shape[0]) != j
        subset = dict(survey, z_src_m=survey['z_src_m'][keep], T_obs_s=t[keep])
        rays, p, _ = _fit(subset)
        clocks = (rays.clocks @ p[5:]) / 1000
        worst[0].append(errors(p[:2], clocks, gt['plume_xz_m'], gt['rcv_clock_bias_s']))
        from tomography_solver import _Rays
        held = _Rays(survey['W_m'], survey['z_src_m'][j:j+1], survey['z_rcv_m'], survey['R_plume_m'])
        pred = held.design @ p[3:] + p[2] * held.plume(*p[:2])
        res = t[j] * 1000 - pred
        clean_predictions.extend(res[abs(res) < 2.])
    for j in range(t.shape[1]):
        keep = np.arange(t.shape[1]) != j
        subset = dict(survey, z_rcv_m=survey['z_rcv_m'][keep], T_obs_s=t[:, keep])
        r = invert_plume_and_clocks(subset)
        clocks = np.asarray(gt['rcv_clock_bias_s'])[keep]
        worst[1].append(errors(r['plume_xz_m'], r['rcv_clock_bias_s'], gt['plume_xz_m'], clocks - clocks.mean()))
    print('public', i, 'worst leave-source-out', np.max(worst[0], axis=0), 'worst leave-receiver-out', np.max(worst[1], axis=0), 'held-source non-skip RMS ms', np.sqrt(np.mean(np.square(clean_predictions))))

rng = np.random.default_rng(728)
nodes, weights = leggauss(64)
u = (nodes + 1) / 2
out = []
for regime in ('public-like', 'wider', 'near-well', 'end-depth', 'weak'):
    for j in range(12):
        w, h = (60., 100.) if regime == 'wider' else (40., 80.)
        n, m = (12, 13) if regime == 'wider' else (15, 16)
        src = np.linspace(.075*h, .925*h, n) + rng.uniform(-.6,.6,n)
        rcv = np.linspace(.0625*h, .9375*h, m) + rng.uniform(-.6,.6,m)
        x, z = rng.uniform(.15,.85)*w, rng.uniform(.17,.83)*h
        if regime == 'near-well': x = 2. if j % 2 else w - 2.
        if regime == 'end-depth': z = 3. if j % 2 else h - 3.
        amp = 0.000022 if regime == 'weak' else rng.uniform(.00005,.00009)
        dz = rcv[None, :] - src[:, None]; dl = np.hypot(w, dz)
        distance_sq = (w*u-x)**2 + (src[:,None,None] + dz[:,:,None]*u-z)**2
        plume = dl * ((np.exp(-distance_sq/(2*5.5**2)) @ weights) / 2)
        s0, eps = rng.uniform(.00034,.00046), rng.uniform(.08,.23)
        clock = rng.uniform(-1,1,m); clock -= clock.mean(); clock *= (0.004 if j % 2 else 0.0032)/max(abs(clock))
        t = dl*s0*(1-eps*(dz/dl)**2) + amp*plume + clock + rng.normal(0,0.000045,dl.shape)
        idx = rng.choice(t.size, round(.04*t.size), replace=False)
        t.flat[idx] += rng.choice([-0.006,0.006], len(idx))
        survey = dict(W_m=w, H_m=h, z_src_m=src, z_rcv_m=rcv, T_obs_s=t, R_plume_m=5.5)
        start = time.monotonic(); result = invert_plume_and_clocks(survey); elapsed = time.monotonic() - start
        a,b = errors(result['plume_xz_m'],result['rcv_clock_bias_s'],[x,z],clock)
        assert abs(sum(result['rcv_clock_bias_s'])) < 1e-12 and max(abs(np.asarray(result['rcv_clock_bias_s']))) <= .004 + 1e-14
        out.append((regime,j,a,b,elapsed,(x,z),result['plume_xz_m']))
for regime in dict.fromkeys(r[0] for r in out):
    rows = [r for r in out if r[0]==regime]
    print(regime,'worst location row',max(rows,key=lambda r:r[2]),'worst clock RMSE s',max(r[3] for r in rows),'max time s',max(r[4] for r in rows),'failed',sum(r[2]>1.8 or r[3]>.00025 for r in rows))
PY
PYTHONPATH=/holdout_task OPENBLAS_NUM_THREADS=1 python3 -u /tmp/stress_tomo.py
```
- **Container Exit Code**: `0` (`5284 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0 worst leave-source-out [1.14619310e-01 6.85004853e-06] worst leave-receiver-out [1.33576495e-01 6.20950816e-06] held-source non-skip RMS ms 0.03489685521622038
public 1 worst leave-source-out [1.45362147e-01 1.51302725e-05] worst leave-receiver-out [1.38634641e-01 1.34580534e-05] held-source non-skip RMS ms 0.034946817169270575
public-like worst location row ('public-like', 11, 0.31782035183125573, 1.6364878916307662e-05, 0.0368770039640367, (30.82786597879661, 15.273590985487003), [30.557323767098723, 15.440374341154644]) worst clock RMSE s 1.882863683874641e-05 max time s 0.03865325916558504 failed 0
wider worst location row ('wider', 7, 0.554157855628265, 1.6043347934784295e-05, 0.03923244494944811, (48.544719691191574, 17.524188733493197), [48.02695873032995, 17.72170965988371]) worst clock RMSE s 2.4858761183793888e-05 max time s 0.04259764798916876 failed 0
near-well worst location row ('near-well', 10, 1.8335031038029568, 2.7725170215054938e-05, 0.045638801995664835, (38.0, 32.41188503769217), [36.209006640195106, 32.80441072702964]) worst clock RMSE s 2.7725170215054938e-05 max time s 0.06117845396511257 failed 1
end-depth worst location row ('end-depth', 9, 3.5377715050213463, 3.758915187548212e-05, 0.04609029204584658, (33.228351504092196, 3.0), [29.717558535838137, 3.4360733376381285]) worst clock RMSE s 3.758915187548212e-05 max time s 0.09360876400023699 failed 1
weak worst location row ('weak', 4, 1.4942967038757826, 8.771962180961199e-06, 0.042004942893981934, (31.39551654871685, 20.794092907241474), [32.787452023862386, 20.250550570118232]) worst clock RMSE s 1.6365607106658196e-05 max time s 0.048066940857097507 failed 0
```

### Turn 5 — `execute_in_remote_docker` (`30795 ms` LLM)
- **Rationale**: Check skip recovery, location uncertainty, the objective at the true center, and whether clock bounds bind. A masking or basin failure warrants a solver change; an indistinguishable true center warrants examining identifiability before adding assumptions.
- **Alternatives Weighed**: `Increase the number of search starts immediately` (rejected: The edge errors may come from weak ray coverage rather than a missed basin; the objective gap and local sensitivity distinguish those failure modes.)

```bash
cp /tmp/stress_tomo.py /tmp/diagnose_tomo.py
python3 - <<'PY'
p='/tmp/diagnose_tomo.py'
s=open(p).read(); i=s.index('for i, item in enumerate(samples):'); j=s.index('rng =',i); s=s[:i]+s[j:]; s=s.replace('start = time.monotonic(); result', '''if regime in ('near-well', 'end-depth', 'weak'):
            rays, p, residual = _fit(survey)
            good = abs(residual) < 2
            a = rays.design[good]
            g, gx, gz = [v[good] for v in rays.plume(*p[:2], derivative=True)]
            X = np.column_stack((p[2]*gx,p[2]*gz,g,a))
            sigma = np.linalg.norm(residual[good])/np.sqrt(good.sum()-len(p))
            cov = sigma**2*np.linalg.pinv(X.T @ X, hermitian=True)
            gt = np.array([x,z,1000*amp])
            g0 = rays.plume(x,z)[good]
            Q = np.column_stack((g0,a))
            target = t.ravel()[good]*1000
            er = target - Q @ np.linalg.lstsq(Q,target,rcond=None)[0]
            delta = (er@er - residual[good]@residual[good])/sigma**2
            if np.linalg.norm(p[:2]-gt[:2]) > 1 or sigma > .065:
                print(regime,j,'truth',gt,'fit',p[:3],'SE',np.sqrt(np.diag(cov)[:3]),'profile delta chi^2',round(delta,3),'residual noise ms',round(sigma,4),'max clock ms',round(max(abs(rays.clocks@p[5:])),4),'mismasked',np.sum(np.flatnonzero(~good)[:,None]==idx[None,:])!=len(idx) or (~good).sum()!=len(idx))
        start = time.monotonic(); result'''); open(p,'w').write(s)
PY
PYTHONPATH=/holdout_task OPENBLAS_NUM_THREADS=1 python3 -u /tmp/diagnose_tomo.py
```
- **Container Exit Code**: `0` (`5232 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
16 mismasked False
end-depth 3 truth [11.81530425  3.          0.06427146] fit [10.49042545  3.29155905  0.05716406] SE [0.70135197 0.52670453 0.00529138] profile delta chi^2 6.382 residual noise ms 0.043 max clock ms 3.993 mismasked False
end-depth 8 truth [2.90963291e+01 7.70000000e+01 5.96800412e-02] fit [3.04859639e+01 7.70290344e+01 7.54120449e-02] SE [2.04585369 0.61631903 0.01484489] profile delta chi^2 0.68 residual noise ms 0.0492 max clock ms 3.192 mismasked False
end-depth 9 truth [33.2283515   3.          0.06388123] fit [29.71755854  3.43607334  0.04720632] SE [2.24345068 0.67730458 0.00936331] profile delta chi^2 2.356 residual noise ms 0.0423 max clock ms 4.0241 mismasked False
end-depth 11 truth [24.76977172  3.          0.05692397] fit [25.7532933   3.18185799  0.06218953] SE [1.16568658 0.47282048 0.00536583] profile delta chi^2 1.374 residual noise ms 0.0439 max clock ms 4.0205 mismasked False
weak 4 truth [3.13955165e+01 2.07940929e+01 2.20000000e-02] fit [3.27874520e+01 2.02505506e+01 2.32741594e-02] SE [0.83300299 0.43198696 0.00213084] profile delta chi^2 3.383 residual noise ms 0.0428 max clock ms 3.199 mismasked False
public-like worst location row ('public-like', 11, 0.31782035183125573, 1.6364878916307662e-05, 0.042925434187054634, (30.82786597879661, 15.273590985487003), [30.557323767098723, 15.440374341154644]) worst clock RMSE s 1.882863683874641e-05 max time s 0.051218484062701464 failed 0
wider worst location row ('wider', 7, 0.554157855628265, 1.6043347934784295e-05, 0.040372905088588595, (48.544719691191574, 17.524188733493197), [48.02695873032995, 17.72170965988371]) worst clock RMSE s 2.4858761183793888e-05 max time s 0.050985143054276705 failed 0
near-well worst location row ('near-well', 10, 1.8335031038029568, 2.7725170215054938e-05, 0.048239629017189145, (38.0, 32.41188503769217), [36.209006640195106, 32.80441072702964]) worst clock RMSE s 2.7725170215054938e-05 max time s 0.08035674202255905 failed 1
end-depth worst location row ('end-depth', 9, 3.5377715050213463, 3.758915187548212e-05, 0.06231035804376006, (33.228351504092196, 3.0), [29.717558535838137, 3.4360733376381285]) worst clock RMSE s 3.758915187548212e-05 max time s 0.09213453810662031 failed 1
weak worst location row ('weak', 4, 1.4942967038757826, 8.771962180961199e-06, 0.04732293402776122, (31.39551654871685, 20.794092907241474), [32.787452023862386, 20.250550570118232]) worst clock RMSE s 1.6365607106658196e-05 max time s 0.06295334198512137 failed 0
```

### Turn 6 — `execute_in_remote_docker` (`38186 ms` LLM)
- **Rationale**: The residual diagnostics distinguish weak coverage from missed skips or an incorrect basin. Test a challenger separately: jointly fitting the clock bounds can change the location where a bound is active, while recovering tightly clustered skips can add ray information. Compare both output errors on the labeled and synthetic slices before replacing the target.
- **Alternatives Weighed**: `Tune a plume amplitude prior to the synthetic edge cases` (rejected: The task specifies positivity but no amplitude range; the synthetic generator's amplitude distribution is not independent evidence for the holdout.)
- **ATMS NOGOOD DECLARED**: Refuted `Once cycle skips are removed, end-depth plume centers are always resolved to 1.8 m by the straight-ray data alone.` | Witness: `The worst end-depth simulation recovered every skip yet missed by 3.538 m; the true center's profiled objective was only 2.356 noise-variance units above the optimum, with a local horizontal standard error of 2.24 m.` | Blanket Diagnosis: `Sparse ray coverage beyond the instrumented depths makes the plume's horizontal position trade off against its amplitude after accounting for background and clocks.` | Sublated Pivot: `Keep the exact ray model, fit the documented clock bounds jointly, and test whether consistently shifted picks can restore useful ray coverage without assuming a plume-amplitude prior.`

```bash
cp /holdout_task/tomography_solver.py /tmp/tomo_baseline.py
python3 - <<'PY'
p='/tmp/tomo_baseline.py'
s=open(p).read().replace('from scipy.optimize import brentq, least_squares','from scipy.optimize import brentq, least_squares, minimize, LinearConstraint, Bounds')
s=s.replace('    p = min(candidates, key=lambda item: item[0])[1]\n    return rays, p, y - prediction(p)', '''    p = min(candidates, key=lambda item: item[0])[1]
    e = y - prediction(p)
    clean = abs(e) < 2.
    col = np.column_stack((np.zeros((rcv.size, 5)), rays.clocks))
    constraint = LinearConstraint(col, -4., 4.)

    def bounded(start, target, mask):
        fit = refine(start, target, mask)
        if np.max(abs(col @ fit)) <= 4.:
            return fit
        # Preserve the zero-sum gauge while fitting the active clock bounds.
        scale = 1. / np.maximum(np.linalg.norm(jac(fit)[mask], axis=0), 1e-7)
        def objective(v):
            b = fit + scale * v
            r = (prediction(b) - target)[mask]
            return 0.5 * (r @ r), scale * (jac(b)[mask].T @ r)
        shift = col @ fit
        result = minimize(objective, np.zeros(len(fit)), jac=True, method='SLSQP',
                          bounds=Bounds((lo-fit)/scale, (hi-fit)/scale),
                          constraints=LinearConstraint(col*scale, -4.-shift, 4.-shift),
                          options={'ftol': 1e-12, 'maxiter': 100})
        return fit + scale * result.x if result.success else fit

    p = bounded(p, y, clean)
    # A common cycle duration permits recovery of otherwise discarded picks.
    # Require clustering within ordinary pick noise before reusing them.
    e = y - prediction(p)
    skip = ~clean
    if skip.sum() >= 4:
        period = np.median(abs(e[skip]))
        noise = max(.01, 1.4826*np.median(abs(e[clean]-np.median(e[clean]))))
        if 4.5 < period < 7.5 and np.max(abs(abs(e[skip])-period)) < 4.*noise:
            target = y.copy()
            target[skip] -= period * np.sign(e[skip])
            p = bounded(p, target, full)
    return rays, p, y - prediction(p)''').replace('    constraint = LinearConstraint(col, -4., 4.)\n','')
open('/tmp/tomography_solver.py','w').write(s)
PY
PYTHONPATH=/tmp OPENBLAS_NUM_THREADS=1 python3 -u /tmp/stress_tomo.py
PYTHONPATH=/tmp OPENBLAS_NUM_THREADS=1 python3 /tmp/public_tomo.py
```
- **Container Exit Code**: `0` (`6718 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0 worst leave-source-out [1.12091848e-01 6.22330543e-06] worst leave-receiver-out [1.32998065e-01 5.67794864e-06] held-source non-skip RMS ms 0.03488287236899667
public 1 worst leave-source-out [1.40887638e-01 1.44682361e-05] worst leave-receiver-out [1.38293531e-01 1.30958500e-05] held-source non-skip RMS ms 0.0348421704233857
public-like worst location row ('public-like', 11, 0.33060098281515393, 1.675458115902957e-05, 0.04517326201312244, (30.82786597879661, 15.273590985487003), [30.53580646725346, 15.428504676152237]) worst clock RMSE s 1.7989813788413426e-05 max time s 0.04955017892643809 failed 0
wider worst location row ('wider', 7, 0.5360628933264828, 1.5846279693653475e-05, 0.05590675491839647, (48.544719691191574, 17.524188733493197), [48.04398635374987, 17.715577213752734]) worst clock RMSE s 2.0517205523921907e-05 max time s 0.057841568952426314 failed 0
near-well worst location row ('near-well', 10, 1.557920348007295, 2.4663128825715518e-05, 0.044711481779813766, (38.0, 32.41188503769217), [36.4601319302218, 32.64836365916297]) worst clock RMSE s 2.4663128825715518e-05 max time s 0.08042074320837855 failed 0
end-depth worst location row ('end-depth', 8, 1.3852390998603803, 3.4371524545060504e-05, 0.08759292494505644, (29.096329092781215, 77.0), [30.48020298374173, 77.0614851014458]) worst clock RMSE s 3.4371524545060504e-05 max time s 0.08759292494505644 failed 0
weak worst location row ('weak', 4, 1.5111027195038231, 8.779708125059854e-06, 0.05784310004673898, (31.39551654871685, 20.794092907241474), [32.80342613384436, 20.24525592871513]) worst clock RMSE s 1.6291479607478495e-05 max time s 0.06885619717650115 failed 0
0 params [16.41850219 34.03880663  0.06491369  0.42012048  0.05910761] actual scipy time 0.048595973988994956 errors 0.09026542889490495 4.8947656869861e-06 clean rms 0.03195439586087098 skips [-5.5059  5.5061  5.4667  5.454   5.558   5.5013 -5.4957  5.5191 -5.5229
 -5.5   ] clock sum 9.215718466126788e-19
1 params [25.90272656 51.4481809   0.07140418  0.38011636  0.06864916] actual scipy time 0.04771638405509293 errors 0.11021497961102453 1.1860238767693307e-05 clean rms 0.03163464512301181 skips [-5.5071  5.4682  5.516  -5.4714  5.5076  5.5246 -5.511   5.5143  5.5159
  5.4946] clock sum 9.0801931945661e-19
```

### Turn 7 — `execute_in_remote_docker` (`33504 ms` LLM)
- **Rationale**: Promote the challenger because it fixes the observed edge failures while preserving public accuracy. Verify the exact installed file in an isolated Python process; the mixed-duration test checks that pick reuse does not degrade recovery when a single cycle period cannot explain the skips.
- **Alternatives Weighed**: `Keep the trimming-only baseline` (rejected: It missed the location target in two edge simulations; the challenger passed both without adding an unsupported plume prior.)

```bash
set -e
cp /tmp/tomography_solver.py /holdout_task/tomography_solver.py
mkdir -p /tmp/tomo_clean_verify
cd /tmp/tomo_clean_verify
env -u PYTHONPATH python3 -I - <<'PY'
import importlib.util, pickle, time
import numpy as np
path = '/holdout_task/tomography_solver.py'
spec = importlib.util.spec_from_file_location('tomography_solver', path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('/holdout_task/public_surveys.pkl', 'rb') as file:
    samples = pickle.load(file)
for index, item in enumerate(samples):
    survey, truth = item['survey'], item['gt']
    observations = np.asarray(survey['T_obs_s'])
    _, _, residual = solver._fit(survey)
    skip = np.flatnonzero(np.abs(residual) > 2.)
    mixed = observations.copy()
    mixed.flat[skip] += np.sign(residual[skip]) * np.resize([-.0005, .0005], len(skip))
    for label, times in [('public', observations), ('mixed skip durations', mixed)]:
        start = time.monotonic()
        result = solver.invert_plume_and_clocks(dict(survey, T_obs_s=times))
        elapsed = time.monotonic() - start
        xz = result['plume_xz_m']
        clocks = result['rcv_clock_bias_s']
        assert set(result) == {'plume_xz_m', 'rcv_clock_bias_s'}
        assert isinstance(xz, list) and isinstance(clocks, list)
        assert len(xz) == 2 and len(clocks) == len(survey['z_rcv_m'])
        assert all(type(v) is float and np.isfinite(v) for v in xz + clocks)
        assert abs(sum(clocks)) < 1e-12 and max(map(abs, clocks)) <= .004 + 1e-12
        location = np.linalg.norm(np.subtract(xz, truth['plume_xz_m']))
        clock = np.sqrt(np.mean(np.square(np.subtract(clocks, truth['rcv_clock_bias_s']))))
        print(index, label, f'location={location:.5f} m clock={clock*1e6:.3f} us runtime={elapsed:.3f} s')
        assert location <= 1.8 and clock <= .00025 and elapsed < 15
PY
```
- **Container Exit Code**: `0` (`2441 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tomography_solver.py']`
- **Stdout (tail)**:
```text
0 public location=0.09027 m clock=4.895 us runtime=0.296 s
0 mixed skip durations location=0.08947 m clock=5.302 us runtime=0.197 s
1 public location=0.11021 m clock=11.860 us runtime=0.299 s
1 mixed skip durations location=0.11213 m clock=12.235 us runtime=0.304 s
```

### Turn 8 — `submit_completed_task` (`17664 ms` LLM)
- **Rationale**: The installed solver passed isolated verification, including mixed-duration skips, and the promoted estimator passed both accuracy targets on every tested synthetic slice. Weak edge coverage remains an identifiability limitation, not a guarantee for unseen surveys.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/tomography_solver.py': 'FILE:6179'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tomography_solver.py`. On the two public surveys, worst-case errors were 0.1103 m for plume location and 11.86 µs for receiver-clock RMSE, with runtime under 0.30 s. All 60 synthetic stress cases met both accuracy targets.`

### Turn 9 — `execute_in_remote_docker` (`42030 ms` LLM)
- **Rationale**: Independent quadrature would expose an endpoint or angle convention error and warrant fixing the ray operator. Adjacent-source differences cancel the receiver clocks exactly, so their observed residuals test the physical model without clock compensation. An added angular term is fit inside each source-deletion fold; only a material predictive gain would justify investigating an omitted dependence.
- **Alternatives Weighed**: `Test drift operators, relay modes, or cohort balancing` (rejected: The survey contains ray arrival times and additive receiver clocks, not a stochastic drift process, relay, batch labels, or classification outcomes.)

```bash
cd /tmp && env -u PYTHONPATH python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
spec = importlib.util.spec_from_file_location('tomo_audit', '/holdout_task/tomography_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_surveys.pkl', 'rb') as f:
    samples = pickle.load(f)
u, weights = leggauss(64)
u = (1+u)/2
for i, item in enumerate(samples):
    s = item['survey']
    src, rcv = s['z_src_m'], s['z_rcv_m']
    w, h, radius = s['W_m'], s['H_m'], s['R_plume_m']
    times = 1000*np.asarray(s['T_obs_s'])
    dz = rcv[None,:]-src[:,None]
    length = np.hypot(w,dz)
    sin2 = (dz/length)**2
    extra = (length*sin2**2).ravel()
    rays, full, err = mod._fit(s)
    maxerr = 0.
    for x,z in ([0.,0.],[w,h], [w,0.], [0.,h], full[:2]):
        r2 = (w*u-x)**2 + (src[:,None,None]+dz[:,:,None]*u-z)**2
        oracle = length*(np.exp(-r2/(2*radius**2)) @ weights)/2
        maxerr = max(maxerr, np.max(abs(oracle.ravel()-rays.plume(x,z))))
    assert maxerr < 1e-10
    prediction = (rays.design @ full[3:] + full[2]*rays.plume(*full[:2])).reshape(times.shape)
    clean = abs(err.reshape(times.shape)) < 2
    c = rays.clocks @ full[5:]
    pairs = clean[1:] & clean[:-1]
    difference = np.diff(times,axis=0)-np.diff(prediction-c,axis=0)
    pair_rms = np.sqrt(np.mean(difference[pairs]**2))
    scores = [[],[]]
    coeffs = []
    for j in range(len(src)):
        rows = np.arange(len(src)) != j
        sub = dict(s,z_src_m=src[rows],T_obs_s=s['T_obs_s'][rows])
        _, start, train_err = mod._fit(sub)
        train_idx = np.repeat(rows,len(rcv)); test_idx = ~train_idx
        retain = train_idx.copy(); retain[train_idx] = abs(train_err) < 2
        a, y = rays.design, times.ravel()
        for order in (0,1):
            design = np.column_stack((a,extra)) if order else a
            # Use identical training masks and the same treatment of skips in both fits.
            def pred(p): return design @ p[3:] + p[2]*rays.plume(*p[:2])
            def jac(p):
                g,gx,gz = rays.plume(*p[:2],derivative=True)
                return np.column_stack((p[2]*gx,p[2]*gz,g,design))[retain]
            p0 = np.r_[start,0.] if order else start
            lo = np.r_[0.,0.,0.,np.full(design.shape[1],-np.inf)]
            hi = np.r_[w,h,np.full(design.shape[1]+1,np.inf)]
            p = least_squares(lambda p:(pred(p)-y)[retain],p0,jac=jac,bounds=(lo,hi),x_scale='jac').x
            held = (y-pred(p))[test_idx]
            scores[order].append(np.mean(np.minimum(held**2, .25**2)))
            if order: coeffs.append(p[-1])
    truth = item['gt']
    loc = np.linalg.norm(full[:2]-truth['plume_xz_m'])
    clk = np.sqrt(np.mean((c/1000-truth['rcv_clock_bias_s'])**2))
    print(f'public {i}: finite integral max error {maxerr:.2e} m; adjacent source difference RMS {pair_rms:.5f} ms; held-source clipped RMS (specified / quartic extension) {np.sqrt(np.mean(scores,axis=1))} ms; quartic coefficient range {min(coeffs):.3g}..{max(coeffs):.3g}; location {loc:.4f} m; clock {1e6*clk:.2f} us')
PY
```
- **Container Exit Code**: `0` (`10276 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0: finite integral max error 9.77e-15 m; adjacent source difference RMS 0.04628 ms; held-source clipped RMS (specified / quartic extension) [0.06141023 0.0615454 ] ms; quartic coefficient range -0.000823..0.000516; location 0.0903 m; clock 4.89 us
public 1: finite integral max error 8.88e-15 m; adjacent source difference RMS 0.04832 ms; held-source clipped RMS (specified / quartic extension) [0.06143745 0.06151548] ms; quartic coefficient range -0.00125..-0.00027; location 0.1102 m; clock 11.86 us
```

### Turn 10 — `submit_completed_task` (`21932 ms` LLM)
- **Rationale**: Independent quadrature confirmed the finite-ray integral to within 9.8e-15 m, and clock-canceling source differences showed residuals consistent with ordinary pick noise. The unchanged installed artifact passed isolated execution, including mixed skip durations. The observed edge failures were corrected in the tested cases, but weak geometric identifiability remains a limitation.
- **Alternatives Weighed**: `Add a fourth-order angular term` (rejected: It slightly worsened held-source prediction on both public surveys.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/tomography_solver.py': 'FILE:6179'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tomography_solver.py`. Across both public surveys, the worst plume-center error was 0.1103 m, the worst clock RMSE was 11.86 µs, and runtime was under 0.31 s per survey. All 60 synthetic stress cases met both accuracy targets. Plumes beyond the instrumented depths remain harder to localize.`

### Turn 11 — `execute_in_remote_docker` (`27438 ms` LLM)
- **Rationale**: Keep the measured geometry and pick noise, but concentrate the documented skip count at one source or receiver. A receiver-specific failure would justify testing clock bounds during basin selection, where a fitted offset can otherwise absorb a burst before the final constrained refit.
- **Alternatives Weighed**: `Repeat the angular-basis audit` (rejected: Independent quadrature and source-deletion validation already support the specified model; a quartic term worsened prediction on both surveys.)

```bash
cd /tmp && env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('tomo', '/holdout_task/tomography_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_surveys.pkl', 'rb') as f:
    items = pickle.load(f)
for k, item in enumerate(items):
    s, gt = item['survey'], item['gt']
    obs = np.asarray(s['T_obs_s'])
    _, _, e = m._fit(s)
    e = e.reshape(obs.shape)
    bad = abs(e) > 2
    base = obs.copy()
    base[bad] -= np.median(abs(e[bad])) * np.sign(e[bad]) / 1000
    n = int(bad.sum())
    print('survey', k, 'observed skip counts by source', bad.sum(axis=1).tolist(), 'by receiver', bad.sum(axis=0).tolist())
    for axis in (0, 1):
        results = []
        for j in range(obs.shape[axis]):
            for end in (0, 1):
                for sign in (-1, 1):
                    y = base.copy()
                    selected = slice(-n, None) if end else slice(n)
                    if axis == 0:
                        y[j, selected] += sign * .006
                    else:
                        y[selected, j] += sign * .006
                    r = m.invert_plume_and_clocks(dict(s, T_obs_s=y))
                    a = np.linalg.norm(np.subtract(r['plume_xz_m'], gt['plume_xz_m']))
                    b = np.sqrt(np.mean(np.square(np.subtract(r['rcv_clock_bias_s'], gt['rcv_clock_bias_s']))))
                    results.append((a, b, j, end, sign))
        print('axis', axis, 'worst center', max(results), 'worst clock', max(results, key=lambda r:r[1]), 'fails', sum(a > 1.8 or b > .00025 for a,b,*_ in results), '/', len(results))
PY
```
- **Container Exit Code**: `0` (`12908 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
survey 0 observed skip counts by source [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 2, 0, 1, 1, 1] by receiver [2, 3, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]
axis 0 worst center (0.10845872289028219, 4.973807833410344e-06, 6, 1, -1) worst clock (0.07818643954572398, 5.241130721810064e-06, 10, 1, -1) fails 0 / 60
axis 1 worst center (25.55709275409375, 0.001157629440777203, 8, 1, 1) worst clock (0.11670922242326125, 0.0014572006248734272, 7, 1, 1) fails 64 / 64
survey 1 observed skip counts by source [1, 0, 0, 0, 0, 2, 1, 1, 1, 0, 2, 0, 0, 1, 1] by receiver [1, 2, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 2, 1]
axis 0 worst center (0.15017758141071633, 1.1300554064901529e-05, 2, 1, -1) worst clock (0.134628915171363, 1.2681004218078922e-05, 2, 0, 1) fails 0 / 60
axis 1 worst center (27.291078341674428, 0.0003108712325491028, 1, 1, 1) worst clock (0.14291359980792648, 0.0014592687485449505, 12, 0, 1) fails 64 / 64
```

### Turn 12 — `execute_in_remote_docker` (`54817 ms` LLM)
- **Rationale**: Run the challenger against the exact burst witnesses and the prior stress suite. Promote it only if it recovers the burst cases without materially reducing the original accuracy margins.
- **Alternatives Weighed**: `Enforce clock bounds earlier without testing cycle assignments` (rejected: Bounds alone do not identify which receiver band is unshifted, especially when the fitted offset can absorb much of the cycle shift.)
- **ATMS NOGOOD DECLARED**: Refuted `A fit that trims large residuals can identify cycle-skipped picks even when most picks at one receiver share the same shift.` | Witness: `On the two measured survey grids with relocated skips, 128 of 128 same-receiver burst tests failed a tolerance, while all 120 same-source burst tests passed.` | Blanket Diagnosis: `The additive clock can absorb a receiver's majority cycle band, reversing which picks appear to be outliers before the clock bounds are enforced.` | Sublated Pivot: `Test both assignments for a well-separated receiver residual band and compare them using the raw picks and the documented clock bounds.`

```bash
mkdir -p /tmp/tomo_burst
python3 - <<'PY'
s=open('/holdout_task/tomography_solver.py').read().replace('def _fit(survey):','def _fit_once(survey):')
pos=s.index('\ndef invert_plume_and_clocks')
s=s[:pos]+'''
def _fit(survey):
    rays, p, residual = _fit_once(survey)
    times = np.asarray(survey['T_obs_s'], dtype=float)
    e = residual.reshape(times.shape)
    best = (rays, p, residual)
    best_cost = np.inf
    for j in range(e.shape[1]):
        order = np.argsort(e[:, j])
        values = e[order, j]
        split = np.argmax(np.diff(values)) + 1
        low, high = values[:split], values[split:]
        if min(len(low), len(high)) < 3 or values[split] - values[split-1] < 3.:
            continue
        period = np.median(high) - np.median(low)
        if not 4.5 < period < 7.5 or max(np.ptp(low), np.ptp(high)) > 2.:
            continue
        # A receiver clock can absorb the majority band. Test the opposite
        # cycle assignment, then compare both solutions on the raw picks.
        if len(low) > len(high):
            rows, shift = order[:split], -period
        else:
            rows, shift = order[split:], period
        if len(rows) <= len(values) / 2:
            continue
        if best_cost == np.inf:
            best_cost = _cycle_cost(rays, p, residual, times.shape)
        corrected = times.copy()
        corrected[rows, j] -= shift / 1000.
        trial_rays, trial_p, trial_e = _fit_once(dict(survey, T_obs_s=corrected))
        trial_e = trial_e.reshape(times.shape).copy()
        trial_e[rows, j] += shift
        trial_e = trial_e.ravel()
        cost = _cycle_cost(trial_rays, trial_p, trial_e, times.shape)
        if cost < best_cost:
            best_cost = cost
            best = (trial_rays, trial_p, trial_e)
    return best


def _cycle_cost(rays, p, residual, shape):
    absolute = np.abs(residual)
    skip = absolute > 3.
    error = absolute.copy()
    if np.any(skip):
        period = np.clip(np.median(absolute[skip]), 4.5, 7.5)
        error[skip] -= period
    clock_excess = np.maximum(np.abs(rays.clocks @ p[5:]) - 4., 0.)
    return np.minimum(error**2, .35**2).sum() + .005 * skip.sum() + shape[0] * (clock_excess @ clock_excess)

''' +s[pos:]
open('/tmp/tomo_burst/tomography_solver.py','w').write(s)
PY
cat >/tmp/burst_tomo.py <<'PY'
import pickle
import numpy as np
import tomography_solver as m
for k,item in enumerate(pickle.load(open('/holdout_task/public_surveys.pkl','rb'))):
    s,gt=item['survey'],item['gt']; obs=np.asarray(s['T_obs_s']); _,_,e=m._fit(s); e=e.reshape(obs.shape)
    bad=abs(e)>2; base=obs.copy(); base[bad]-=np.median(abs(e[bad]))*np.sign(e[bad])/1000
    n=int(bad.sum())
    for axis in (0,1):
        results=[]
        for j in range(obs.shape[axis]):
            for end in (0,1):
                for sign in (-1,1):
                    y=base.copy(); selected=slice(-n,None) if end else slice(n)
                    if axis==0: y[j,selected]+=sign*.006
                    else: y[selected,j]+=sign*.006
                    r=m.invert_plume_and_clocks(dict(s,T_obs_s=y))
                    a=np.linalg.norm(np.subtract(r['plume_xz_m'],gt['plume_xz_m'])); b=np.sqrt(np.mean(np.square(np.subtract(r['rcv_clock_bias_s'],gt['rcv_clock_bias_s']))))
                    results.append((a,b,j,end,sign))
        print(k,axis,'worst location',max(results),'worst clock',max(results,key=lambda r:r[1]),'failures',sum(a>1.8 or b>.00025 for a,b,*_ in results),'/',len(results),flush=True)
PY
cd /tmp/tomo_burst && export OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/tomo_burst && python3 -P -u /tmp/burst_tomo.py && python3 -P -u /tmp/stress_tomo.py
```
- **Container Exit Code**: `0` (`22630 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 0 worst location (0.10845872289028219, 4.973807833410344e-06, 6, 1, -1) worst clock (0.07818643954572398, 5.241130721810064e-06, 10, 1, -1) failures 0 / 60
0 1 worst location (1.105523973048019, 0.000173439594145247, 8, 1, 1) worst clock (0.11670922242326125, 0.0014572006248734272, 7, 1, 1) failures 18 / 64
1 0 worst location (0.15017758141071633, 1.1300554064901529e-05, 2, 1, -1) worst clock (0.134628915171363, 1.2681004218078922e-05, 2, 0, 1) failures 0 / 60
1 1 worst location (15.497651537253493, 0.000973071995538807, 9, 0, 1) worst clock (0.14291359980792648, 0.0014592687485449505, 12, 0, 1) failures 15 / 64
public 0 worst leave-source-out [1.12091848e-01 6.22330543e-06] worst leave-receiver-out [1.32998065e-01 5.67794864e-06] held-source non-skip RMS ms 0.03488287236899667
public 1 worst leave-source-out [1.40887638e-01 1.44682361e-05] worst leave-receiver-out [1.38293531e-01 1.30958500e-05] held-source non-skip RMS ms 0.0348421704233857
public-like worst location row ('public-like', 11, 0.33060098281515393, 1.675458115902957e-05, 0.04346820502541959, (30.82786597879661, 15.273590985487003), [30.53580646725346, 15.428504676152237]) worst clock RMSE s 1.7989813788413426e-05 max time s 0.08303282014094293 failed 0
wider worst location row ('wider', 7, 0.5360628933264828, 1.5846279693653475e-05, 0.04028708813712001, (48.544719691191574, 17.524188733493197), [48.04398635374987, 17.715577213752734]) worst clock RMSE s 2.0517205523921907e-05 max time s 0.043906102888286114 failed 0
near-well worst location row ('near-well', 10, 1.557920348007295, 2.4663128825715518e-05, 0.04951200494542718, (38.0, 32.41188503769217), [36.4601319302218, 32.64836365916297]) worst clock RMSE s 2.4663128825715518e-05 max time s 0.0668867148924619 failed 0
end-depth worst location row ('end-depth', 8, 1.3852390998603803, 3.4371524545060504e-05, 0.08427722495980561, (29.096329092781215, 77.0), [30.48020298374173, 77.0614851014458]) worst clock RMSE s 3.4371524545060504e-05 max time s 0.08427722495980561 failed 0
weak worst location row ('weak', 4, 1.5111027195038231, 8.779708125059854e-06, 0.04788663098588586, (31.39551654871685, 20.794092907241474), [32.80342613384436, 20.24525592871513]) worst clock RMSE s 1.6291479607478495e-05 max time s 0.058967779856175184 failed 0
```

### Turn 13 — `execute_in_remote_docker` (`49057 ms` LLM)
- **Rationale**: A wrong plume basin can distort the residual bands used to detect it. Subtracting only the smooth background keeps band detection independent of the fitted plume and invariant to a receiver's additive clock. Test both assignments against the raw picks, then use the burst and prior stress results to decide whether to install the challenger.
- **Alternatives Weighed**: `Install the first burst challenger` (rejected: It still missed a tolerance in 33 receiver-burst cases, including a 15.50 m localization error.)

```bash
mkdir -p /tmp/tomo_band
python3 - <<'PY'
s=open('/holdout_task/tomography_solver.py').read().replace('def _fit(survey):','def _fit_once(survey):')
pos=s.index('\ndef invert_plume_and_clocks')
s=s[:pos]+'''
def _fit(survey):
    best = _fit_once(survey)
    times = np.asarray(survey['T_obs_s'], dtype=float)
    rays, p, residual = best
    # Remove only the smooth background before looking for two timing bands.
    # An additive clock translates both bands and cannot change their separation.
    background = rays.design[:, :2] @ p[3:5]
    bands = times * 1000 - background.reshape(times.shape)
    best_cost = None
    for j, column in enumerate(bands.T):
        order = np.argsort(column)
        values = column[order]
        gaps = np.diff(values)
        if gaps.size == 0:
            continue
        k = np.argmax(gaps) + 1
        if min(k, len(values)-k) < 3 or gaps[k-1] < 3.3:
            continue
        # Isolated skips do not create a second multi-pick band. Compare both
        # possible clock gauges when there really are two bands.
        period = np.median(values[k:]) - np.median(values[:k])
        if not 4.5 < period < 7.5 or max(np.ptp(values[:k]), np.ptp(values[k:])) > 2.2:
            continue
        if best_cost is None:
            best_cost = _band_cost(residual)
        for rows, shift in ((order[:k], -period), (order[k:], period)):
            corrected = times.copy()
            corrected[rows, j] -= shift/1000
            trial_rays, trial_p, trial_residual = _fit_once(dict(survey, T_obs_s=corrected))
            trial_residual = trial_residual.reshape(times.shape).copy()
            trial_residual[rows, j] += shift
            trial_residual = trial_residual.ravel()
            cost = _band_cost(trial_residual)
            if cost < best_cost:
                best_cost = cost
                best = trial_rays, trial_p, trial_residual
    return best


def _band_cost(residual):
    e = np.abs(residual)
    skip = e > 3.
    if np.any(skip):
        period = np.clip(np.median(e[skip]), 4.5, 7.5)
        e[skip] -= period
    return np.minimum(e**2, .35**2).sum() + .02 * skip.sum()

''' +s[pos:]
open('/tmp/tomo_band/tomography_solver.py','w').write(s)
PY
cd /tmp/tomo_band
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/tomo_band
python3 -P -u /tmp/burst_tomo.py && python3 -P -u /tmp/stress_tomo.py
```
- **Container Exit Code**: `0` (`26869 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 0 worst location (0.10845872289028219, 4.973807833410344e-06, 6, 1, -1) worst clock (0.07818643954572398, 5.241130721810064e-06, 10, 1, -1) failures 0 / 60
0 1 worst location (10.093776007291051, 0.0006233931965272663, 0, 1, -1) worst clock (0.11670922242326125, 0.0014572006248734272, 7, 1, 1) failures 21 / 64
1 0 worst location (0.15017758141071633, 1.1300554064901529e-05, 2, 1, -1) worst clock (0.134628915171363, 1.2681004218078922e-05, 2, 0, 1) failures 0 / 60
1 1 worst location (1.3366022929818313, 0.0001557350697372591, 14, 1, -1) worst clock (0.14291359980792648, 0.0014592687485449505, 12, 0, 1) failures 14 / 64
public 0 worst leave-source-out [1.12091848e-01 6.22330543e-06] worst leave-receiver-out [1.32998065e-01 5.67794864e-06] held-source non-skip RMS ms 0.03488287236899667
public 1 worst leave-source-out [1.40887638e-01 1.44682361e-05] worst leave-receiver-out [1.38293531e-01 1.30958500e-05] held-source non-skip RMS ms 0.0348421704233857
public-like worst location row ('public-like', 11, 0.33060098281515393, 1.675458115902957e-05, 0.04045750107616186, (30.82786597879661, 15.273590985487003), [30.53580646725346, 15.428504676152237]) worst clock RMSE s 1.7989813788413426e-05 max time s 0.12147053191438317 failed 0
wider worst location row ('wider', 7, 0.5360628933264828, 1.5846279693653475e-05, 0.03842542599886656, (48.544719691191574, 17.524188733493197), [48.04398635374987, 17.715577213752734]) worst clock RMSE s 2.0517205523921907e-05 max time s 0.04208055790513754 failed 0
near-well worst location row ('near-well', 10, 1.557920348007295, 2.4663128825715518e-05, 0.04354811902157962, (38.0, 32.41188503769217), [36.4601319302218, 32.64836365916297]) worst clock RMSE s 2.4663128825715518e-05 max time s 0.06919548707082868 failed 0
end-depth worst location row ('end-depth', 8, 1.3852390998603803, 3.4371524545060504e-05, 0.0839963520411402, (29.096329092781215, 77.0), [30.48020298374173, 77.0614851014458]) worst clock RMSE s 3.4371524545060504e-05 max time s 0.0839963520411402 failed 0
weak worst location row ('weak', 4, 1.5111027195038231, 8.779708125059854e-06, 0.05147491884417832, (31.39551654871685, 20.794092907241474), [32.80342613384436, 20.24525592871513]) worst clock RMSE s 1.6291479607478495e-05 max time s 0.05233567696996033 failed 0
```

### Turn 14 — `execute_in_remote_docker` (`56566 ms` LLM)
- **Rationale**: Fit the plume without the suspect receiver to separate physical travel time from its clock, then fit both discrete assignments with a free common cycle duration. The burst suite tests whether this closes the observed failure; the prior suite checks whether it costs accuracy on dispersed skips or weakly covered plumes.
- **Alternatives Weighed**: `Use the background-only band medians as the final cycle duration` (rejected: The plume changes travel time across the two bands; that difference can bias the inferred duration and leave the wrong clock assignment preferred.)

```bash
mkdir -p /tmp/tomo_profile
python3 - <<'PY'
s=open('/holdout_task/tomography_solver.py').read().replace('def _fit(survey):','def _fit_once(survey):')
pos=s.index('\ndef invert_plume_and_clocks')
s=s[:pos]+'''
def _fit(survey):
    best = _fit_once(survey)
    rays, p, residual = best
    times = np.asarray(survey['T_obs_s'], dtype=float)
    nr, nc = times.shape
    y = (1000. * times).ravel()
    a = rays.design
    def prediction(b):
        return a @ b[3:] + b[2] * rays.plume(*b[:2])
    # Screen for a second populated timing band without using the fitted
    # receiver offset. Confirm it against a fit that excludes that receiver.
    smooth = y - a[:, :2] @ p[3:5]
    def split(column, gap):
        order = np.argsort(column)
        gaps = np.diff(column[order])
        if len(gaps) < 5:
            return None
        k = 3 + np.argmax(gaps[2:-2])
        if gaps[k-1] < gap:
            return None
        low, high = order[:k], order[k:]
        period = np.median(column[high]) - np.median(column[low])
        return (low, high, period) if 4.3 < period < 7.7 else None
    def score(b, e):
        ae = abs(e)
        skip = ae > 3.
        if skip.any():
            ae[skip] -= np.clip(np.median(ae[skip]), 4.5, 7.5)
        excess = np.maximum(abs(rays.clocks @ b[5:]) - 4., 0.)
        # Count a skip only after accounting for its waveform-independent
        # cycle duration. The measured clock bound also breaks gauge ties.
        return np.minimum(ae**2, 4.).sum() + .006 * skip.sum() + nr * (excess @ excess)
    best_cost = score(p, residual)
    for j in range(nc):
        if split(smooth.reshape(nr, nc)[:, j], 2.8) is None:
            continue
        keep = np.arange(nc) != j
        _, ref, _ = _fit_once(dict(survey, z_rcv_m=np.asarray(survey['z_rcv_m'])[keep], T_obs_s=times[:, keep]))
        without_clock = y - a[:, :2] @ ref[3:5] - ref[2] * rays.plume(*ref[:2])
        band = split(without_clock.reshape(nr, nc)[:, j], 3.3)
        if band is None:
            continue
        low, high, period = band
        for rows, sign in ((low, -1.), (high, 1.)):
            indices = rows * nc + j
            corrected = y.copy()
            corrected[indices] -= sign * period
            _, trial, _ = _fit_once(dict(survey, T_obs_s=corrected.reshape(times.shape)/1000.))
            error = y - prediction(trial)
            skips = np.where(abs(error) > 3., np.sign(error), 0.)
            skips[indices] = sign
            lo = np.r_[0., 0., 0., np.full(a.shape[1], -np.inf), 4.5]
            hi = np.r_[float(survey['W_m']), float(survey['H_m']), np.full(a.shape[1]+1, np.inf), 7.5]
            def jac(b):
                g, gx, gz = rays.plume(*b[:2], derivative=True)
                return np.column_stack((b[2]*gx, b[2]*gz, g, a, skips))
            fit = least_squares(lambda b: prediction(b[:-1]) + skips*b[-1] - y,
                                np.r_[trial, np.clip(period, 4.5, 7.5)], jac=jac,
                                bounds=(lo, hi), x_scale='jac', loss='soft_l1',
                                f_scale=.15, max_nfev=50).x[:-1]
            error = y - prediction(fit)
            cost = score(fit, error)
            if cost < best_cost:
                best_cost = cost
                best = rays, fit, error
    return best

''' +s[pos:]
open('/tmp/tomo_profile/tomography_solver.py','w').write(s)
PY
cd /tmp/tomo_profile
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/tomo_profile
python3 -P -u /tmp/burst_tomo.py && python3 -P -u /tmp/stress_tomo.py
```
- **Container Exit Code**: `0` (`35010 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 0 worst location (0.10845872289028219, 4.973807833410344e-06, 6, 1, -1) worst clock (0.07818643954572398, 5.241130721810064e-06, 10, 1, -1) failures 0 / 60
0 1 worst location (10.093776007291051, 0.0006233931965272663, 0, 1, -1) worst clock (0.11670922242326125, 0.0014572006248734272, 7, 1, 1) failures 19 / 64
1 0 worst location (0.15017758141071633, 1.1300554064901529e-05, 2, 1, -1) worst clock (0.134628915171363, 1.2681004218078922e-05, 2, 0, 1) failures 0 / 60
1 1 worst location (0.1478215818778399, 0.0014567660507979856, 12, 1, 1) worst clock (0.14291359980792648, 0.0014592687485449505, 12, 0, 1) failures 14 / 64
public 0 worst leave-source-out [1.12091848e-01 6.22330543e-06] worst leave-receiver-out [1.32998065e-01 5.67794864e-06] held-source non-skip RMS ms 0.03488287236899667
public 1 worst leave-source-out [1.40887638e-01 1.44682361e-05] worst leave-receiver-out [1.38293531e-01 1.30958500e-05] held-source non-skip RMS ms 0.0348421704233857
public-like worst location row ('public-like', 11, 0.33060098281515393, 1.675458115902957e-05, 0.047964357072487473, (30.82786597879661, 15.273590985487003), [30.53580646725346, 15.428504676152237]) worst clock RMSE s 1.7989813788413426e-05 max time s 0.1980778449214995 failed 0
wider worst location row ('wider', 7, 0.5360628933264828, 1.5846279693653475e-05, 0.04880083189345896, (48.544719691191574, 17.524188733493197), [48.04398635374987, 17.715577213752734]) worst clock RMSE s 2.0517205523921907e-05 max time s 0.05587971210479736 failed 0
near-well worst location row ('near-well', 10, 1.557920348007295, 2.4663128825715518e-05, 0.046128286980092525, (38.0, 32.41188503769217), [36.4601319302218, 32.64836365916297]) worst clock RMSE s 2.4663128825715518e-05 max time s 0.06289578112773597 failed 0
end-depth worst location row ('end-depth', 8, 1.3852390998603803, 3.4371524545060504e-05, 0.0907373009249568, (29.096329092781215, 77.0), [30.48020298374173, 77.0614851014458]) worst clock RMSE s 3.4371524545060504e-05 max time s 0.0907373009249568 failed 0
weak worst location row ('weak', 4, 1.5111027195038231, 8.779708125059854e-06, 0.05222250986844301, (31.39551654871685, 20.794092907241474), [32.80342613384436, 20.24525592871513]) worst clock RMSE s 1.6291479607478495e-05 max time s 0.07193530001677573 failed 0
```
- **Stderr (tail)**:
```text
/usr/local/lib/python3.12/dist-packages/scipy/optimize/_slsqp_py.py:437: RuntimeWarning: Values in x were outside bounds during a minimize step, clipping to bounds
  fx = wrapped_fun(x)
```

### Turn 15 — `execute_in_remote_docker` (`35636 ms` LLM)
- **Rationale**: Adjacent-source differences remove receiver clocks exactly, and a contiguous skip block affects only the differences at its boundaries. Test whether that independent screen and a weaker skip-count penalty recover the observed burst failures without sacrificing the existing accuracy margins.
- **Alternatives Weighed**: `Choose the assignment with fewer altered picks` (rejected: That rule favors the wrong clock by construction when the skipped picks form the majority at one receiver.)

```bash
set -e
mkdir -p /tmp/tomo_contrast
python3 - <<'PY'
s=open('/tmp/tomo_profile/tomography_solver.py').read()
a=s.index('    # Screen for a second populated')
b=s.index('    def split(column, gap):',a)
s=s[:a]+'''    # Adjacent source differences cancel clocks. Robustly fitting the two
    # background columns here also avoids letting a majority skip band set
    # the clock or warp the background used to detect that band.
    dy = np.diff(1000. * times, axis=0).ravel()
    da = np.diff(a[:, :2].reshape(nr, nc, 2), axis=0).reshape(-1, 2)
    coef = p[3:5]
    for _ in range(8):
        weight = (1. + ((dy - da @ coef) / .18)**2)**(-.25)
        coef = _linear(da, dy, weight)
    smooth = y - a[:, :2] @ coef
''' + s[b:]
s=s.replace('return np.minimum(ae**2, 4.).sum() + .006 * skip.sum() + nr * (excess @ excess)', 'return np.minimum(ae**2, 4.).sum() + .0001 * skip.sum() + nr * (excess @ excess)')
s=s.replace('if split(smooth.reshape(nr, nc)[:, j], 2.8) is None:', 'if split(smooth.reshape(nr, nc)[:, j], 2.6) is None:')
open('/tmp/tomo_contrast/tomography_solver.py','w').write(s)
PY
cd /tmp/tomo_contrast
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/tomo_contrast
python3 -P -u /tmp/burst_tomo.py
python3 -P -u /tmp/stress_tomo.py
```
- **Container Exit Code**: `0` (`34089 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 0 worst location (0.10845872289028219, 4.973807833410344e-06, 6, 1, -1) worst clock (0.07818643954572398, 5.241130721810064e-06, 10, 1, -1) failures 0 / 60
0 1 worst location (0.12228938834975174, 7.532263652247556e-06, 9, 1, 1) worst clock (0.0660382459660155, 1.00936526331257e-05, 7, 0, 1) failures 0 / 64
1 0 worst location (0.15017758141071633, 1.1300554064901529e-05, 2, 1, -1) worst clock (0.134628915171363, 1.2681004218078922e-05, 2, 0, 1) failures 0 / 60
1 1 worst location (0.1322412850875348, 1.2971853082944034e-05, 4, 0, 1) worst clock (0.087305529914238, 1.4329703740715141e-05, 5, 1, 1) failures 0 / 64
public 0 worst leave-source-out [1.12091848e-01 6.22330543e-06] worst leave-receiver-out [1.32998065e-01 5.67794864e-06] held-source non-skip RMS ms 0.03488287236899667
public 1 worst leave-source-out [1.40887638e-01 1.44682361e-05] worst leave-receiver-out [1.38293531e-01 1.30958500e-05] held-source non-skip RMS ms 0.0348421704233857
public-like worst location row ('public-like', 11, 0.33060098281515393, 1.675458115902957e-05, 0.04302531201392412, (30.82786597879661, 15.273590985487003), [30.53580646725346, 15.428504676152237]) worst clock RMSE s 1.7989813788413426e-05 max time s 0.1935855250339955 failed 0
wider worst location row ('wider', 7, 0.5360628933264828, 1.5846279693653475e-05, 0.041283885948359966, (48.544719691191574, 17.524188733493197), [48.04398635374987, 17.715577213752734]) worst clock RMSE s 2.0517205523921907e-05 max time s 0.04841059306636453 failed 0
near-well worst location row ('near-well', 10, 1.557920348007295, 2.4663128825715518e-05, 0.04458473203703761, (38.0, 32.41188503769217), [36.4601319302218, 32.64836365916297]) worst clock RMSE s 2.4663128825715518e-05 max time s 0.06475568516179919 failed 0
end-depth worst location row ('end-depth', 8, 1.3852390998603803, 3.4371524545060504e-05, 0.09840859496034682, (29.096329092781215, 77.0), [30.48020298374173, 77.0614851014458]) worst clock RMSE s 3.4371524545060504e-05 max time s 0.09840859496034682 failed 0
weak worst location row ('weak', 4, 1.5111027195038231, 8.779708125059854e-06, 0.04753855103626847, (31.39551654871685, 20.794092907241474), [32.80342613384436, 20.24525592871513]) worst clock RMSE s 1.6291479607478495e-05 max time s 0.06369502400048077 failed 0
```
- **Stderr (tail)**:
```text
/usr/local/lib/python3.12/dist-packages/scipy/optimize/_slsqp_py.py:437: RuntimeWarning: Values in x were outside bounds during a minimize step, clipping to bounds
  fx = wrapped_fun(x)
```

### Turn 16 — `execute_in_remote_docker` (`29891 ms` LLM)
- **Rationale**: Install the challenger that corrected the observed clock–cycle ambiguity. Import the installed file in isolated Python processes to catch packaging or module-path mistakes, verify output shape, clock constraints and runtime on the public surveys, and confirm that the tested burst and edge margins carry over to the installed artifact.
- **Alternatives Weighed**: `Keep the installed trimming-only solver` (rejected: It failed all 128 receiver-burst cases; the challenger passed all 128 without a regression on the public sensor-deletion or synthetic suites.)

```bash
set -e
install -m 644 /tmp/tomo_contrast/tomography_solver.py /holdout_task/tomography_solver.py
cd /tmp
env -u PYTHONPATH python3 -I - <<'PY'
import importlib.util, pickle, time
import numpy as np
spec = importlib.util.spec_from_file_location('tomography_solver', '/holdout_task/tomography_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_surveys.pkl', 'rb') as f:
    items = pickle.load(f)
for i, item in enumerate(items):
    s, gt = item['survey'], item['gt']
    raw = np.asarray(s['T_obs_s'])
    _, _, residual = m._fit(s)
    skips = np.flatnonzero(abs(residual) > 2.)
    mixed = raw.copy()
    mixed.flat[skips] += np.sign(residual[skips]) * np.resize([-.0005, .0005], skips.size)
    for label, times in [('public', raw), ('mixed cycle durations', mixed)]:
        t = time.monotonic()
        r = m.invert_plume_and_clocks(dict(s, T_obs_s=times))
        elapsed = time.monotonic() - t
        assert set(r) == {'plume_xz_m', 'rcv_clock_bias_s'}
        x, c = r['plume_xz_m'], r['rcv_clock_bias_s']
        assert isinstance(x, list) and len(x) == 2 and isinstance(c, list) and len(c) == raw.shape[1]
        assert all(type(v) is float and np.isfinite(v) for v in x+c)
        assert abs(sum(c)) < 1e-12 and max(map(abs, c)) <= .004 + 1e-12
        a = np.linalg.norm(np.subtract(x, gt['plume_xz_m']))
        b = np.sqrt(np.mean(np.subtract(c, gt['rcv_clock_bias_s'])**2))
        print(f'{i} {label}: {a:.5f} m, {b*1e6:.3f} us, {elapsed:.3f} s', flush=True)
        assert a <= 1.8 and b <= .00025 and elapsed < 15
PY
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 python3 -I -u - <<'PY'
import importlib.util, runpy, sys
spec = importlib.util.spec_from_file_location('tomography_solver', '/holdout_task/tomography_solver.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
runpy.run_path('/tmp/burst_tomo.py')
runpy.run_path('/tmp/stress_tomo.py')
PY
```
- **Container Exit Code**: `0` (`33803 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tomography_solver.py']`
- **Stdout (tail)**:
```text
0 public: 0.09027 m, 4.895 us, 0.300 s
0 mixed cycle durations: 0.08947 m, 5.302 us, 0.299 s
1 public: 0.11021 m, 11.860 us, 0.298 s
1 mixed cycle durations: 0.11213 m, 12.235 us, 0.300 s
0 0 worst location (0.10845872289028219, 4.973807833410344e-06, 6, 1, -1) worst clock (0.07818643954572398, 5.241130721810064e-06, 10, 1, -1) failures 0 / 60
0 1 worst location (0.12228938834975174, 7.532263652247556e-06, 9, 1, 1) worst clock (0.0660382459660155, 1.00936526331257e-05, 7, 0, 1) failures 0 / 64
1 0 worst location (0.15017758141071633, 1.1300554064901529e-05, 2, 1, -1) worst clock (0.134628915171363, 1.2681004218078922e-05, 2, 0, 1) failures 0 / 60
1 1 worst location (0.1322412850875348, 1.2971853082944034e-05, 4, 0, 1) worst clock (0.087305529914238, 1.4329703740715141e-05, 5, 1, 1) failures 0 / 64
public 0 worst leave-source-out [1.12091848e-01 6.22330543e-06] worst leave-receiver-out [1.32998065e-01 5.67794864e-06] held-source non-skip RMS ms 0.03488287236899667
public 1 worst leave-source-out [1.40887638e-01 1.44682361e-05] worst leave-receiver-out [1.38293531e-01 1.30958500e-05] held-source non-skip RMS ms 0.0348421704233857
public-like worst location row ('public-like', 11, 0.33060098281515393, 1.675458115902957e-05, 0.04504302307032049, (30.82786597879661, 15.273590985487003), [30.53580646725346, 15.428504676152237]) worst clock RMSE s 1.7989813788413426e-05 max time s 0.17105442797765136 failed 0
wider worst location row ('wider', 7, 0.5360628933264828, 1.5846279693653475e-05, 0.04058115603402257, (48.544719691191574, 17.524188733493197), [48.04398635374987, 17.715577213752734]) worst clock RMSE s 2.0517205523921907e-05 max time s 0.04334429698064923 failed 0
near-well worst location row ('near-well', 10, 1.557920348007295, 2.4663128825715518e-05, 0.043412637896835804, (38.0, 32.41188503769217), [36.4601319302218, 32.64836365916297]) worst clock RMSE s 2.4663128825715518e-05 max time s 0.059380612103268504 failed 0
end-depth worst location row ('end-depth', 8, 1.3852390998603803, 3.4371524545060504e-05, 0.0841354641597718, (29.096329092781215, 77.0), [30.48020298374173, 77.0614851014458]) worst clock RMSE s 3.4371524545060504e-05 max time s 0.0841354641597718 failed 0
weak worst location row ('weak', 4, 1.5111027195038231, 8.779708125059854e-06, 0.0502234217710793, (31.39551654871685, 20.794092907241474), [32.80342613384436, 20.24525592871513]) worst clock RMSE s 1.6291479607478495e-05 max time s 0.0575610410887748 failed 0
```
- **Stderr (tail)**:
```text
/usr/local/lib/python3.12/dist-packages/scipy/optimize/_slsqp_py.py:437: RuntimeWarning: Values in x were outside bounds during a minimize step, clipping to bounds
  fx = wrapped_fun(x)
```

### Turn 17 — `submit_completed_task` (`16637 ms` LLM)
- **Rationale**: The installed artifact passed isolated execution, output-schema and clock-constraint checks, public sensor-deletion validation, and the burst and synthetic stress suites. The tested receiver-burst ambiguity was resolved without degrading public accuracy; sparse ray coverage remains a geometric limitation.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tomography_solver.py`. Across both public surveys, the worst plume-center error was 0.1103 m, the worst receiver-clock RMSE was 11.86 µs, and runtime was under 0.31 s per survey. All 248 relocated cycle-skip burst tests and 60 synthetic stress cases met both accuracy targets. Plumes beyond the instrumented depths remain harder to localize.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-OOD-Holdout-12",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/tomography_solver.py": "FILE:9596"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "max_pos_err_m": 0.817,
    "max_clk_rmse_s": 0.000113
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"max_pos_err_m\": 0.817, \"max_clk_rmse_s\": 0.000113}\n__HEGEL_VERIFIER_AUTH_7552ff4da18931b32305e62de5812589__:{\"strict_pass\": true, \"max_pos_err_m\": 0.817, \"max_clk_rmse_s\": 0.000113}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 4701.22,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
