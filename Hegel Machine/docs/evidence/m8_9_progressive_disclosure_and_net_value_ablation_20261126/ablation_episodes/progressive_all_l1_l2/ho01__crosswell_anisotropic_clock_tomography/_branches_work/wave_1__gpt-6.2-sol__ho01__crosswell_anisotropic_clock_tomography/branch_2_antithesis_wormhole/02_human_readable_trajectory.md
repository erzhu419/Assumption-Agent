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
| **Turns Used / Max Turns** | `8 / 28` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `6` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `218,759` (`210,508` / `62,208` / `8,251`) |
| **Total Episode Wall Time (s)** | `260.19s` |

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

### Turn 1 — `execute_in_remote_docker` (`16176 ms` LLM)
- **Rationale**: Inspect only the public survey schema and compact data summaries. This will catch input and unit assumptions that could otherwise produce a plausible fit with incorrect geometry or timing.
- **Alternatives Weighed**: `Implement the inverse model before inspecting the public surveys` (rejected: The record keys, array ordering, and available calibration fields determine which parameters must be estimated.)

```bash
ls -l /holdout_task; python3 - <<'PY'
import pickle,numpy as np,scipy
with open('/holdout_task/public_surveys.pkl','rb') as f: d=pickle.load(f)
print('numpy/scipy',np.__version__,scipy.__version__)
def inspect(v,depth=0):
 if depth>4: return
 if isinstance(v,dict):
  for k,x in v.items():
   print('  '*depth,k,end=': '); inspect(x,depth+1)
 elif isinstance(v,(list,tuple)) and v and isinstance(v[0],dict):
  print(type(v).__name__,len(v)); [inspect(x,depth+1) for x in v[:4]]
 else:
  a=np.asarray(v)
  print(type(v).__name__,a.shape,a.dtype, a.ravel()[:10], ('range',float(np.min(a)),float(np.max(a))) if a.size and a.dtype.kind in 'fiu' else '')
inspect(d)
PY
```
- **Container Exit Code**: `0` (`647 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 8
-rw-r--r-- 1 root root 5205 Nov 25 18:49 public_surveys.pkl
numpy/scipy 1.26.4 1.13.0
list 2
   survey:      W_m: float () float64 [40.] ('range', 40.0, 40.0)
     H_m: float () float64 [80.] ('range', 80.0, 80.0)
     z_src_m: ndarray (15,) float64 [ 6.         10.85714286 15.71428571 20.57142857 25.42857143 30.28571429
 35.14285714 40.         44.85714286 49.71428571] ('range', 6.0, 74.0)
     z_rcv_m: ndarray (16,) float64 [ 5.          9.66666667 14.33333333 19.         23.66666667 28.33333333
 33.         37.66666667 42.33333333 47.        ] ('range', 5.0, 75.0)
     T_obs_s: ndarray (15, 16) float64 [0.01491584 0.01729941 0.01816543 0.0148971  0.01612493 0.02175942
 0.01714121 0.01843184 0.02471534 0.02387659] ('range', 0.01330233416627665, 0.03317839066777648)
     R_plume_m: float () float64 [5.5] ('range', 5.5, 5.5)
   gt:      plume_xz_m: list (2,) float64 [16.5 34. ] ('range', 16.5, 34.0)
     rcv_clock_bias_s: ndarray (16,) float64 [-0.0019168   0.00045573  0.00110994 -0.00255604 -0.00179292  0.0032009
 -0.00228896 -0.0019091   0.00332965  0.00124041] ('range', -0.002556039023045965, 0.003329653424440951)
   survey:      W_m: float () float64 [40.] ('range', 40.0, 40.0)
     H_m: float () float64 [80.] ('range', 80.0, 80.0)
     z_src_m: ndarray (15,) float64 [ 6.         10.85714286 15.71428571 20.57142857 25.42857143 30.28571429
 35.14285714 40.         44.85714286 49.71428571] ('range', 6.0, 74.0)
     z_rcv_m: ndarray (16,) float64 [ 5.          9.66666667 14.33333333 19.         23.66666667 28.33333333
 33.         37.66666667 42.33333333 47.        ] ('range', 5.0, 75.0)
     T_obs_s: ndarray (15, 16) float64 [0.01425288 0.0187468  0.01403325 0.0142902  0.01580415 0.01554096
 0.01899251 0.0161894  0.02200229 0.02269381] ('range', 0.01336071363590472, 0.028705721115872333)
     R_plume_m: float () float64 [5.5] ('range', 5.5, 5.5)
   gt:      plume_xz_m: list (2,) float64 [26.  51.5] ('range', 26.0, 51.5)
     rcv_clock_bias_s: ndarray (16,) float64 [-0.0009812   0.00347274 -0.00137483 -0.00143901 -0.00034719 -0.00111101
  0.00170437 -0.00184997  0.0031499   0.00290556] ('range', -0.002568385497994811, 0.0034727403303111385)
```

### Turn 2 — `execute_in_remote_docker` (`58419 ms` LLM)
- **Rationale**: Fit the exact finite-ray integral with analytic derivatives, and remove the receiver-clock gauge with sum-zero contrasts. Public ground truth checks the two outputs independently; residuals grouped by angle and depth test for geometry or anisotropy errors before held-out-ray validation.
- **Alternatives Weighed**: `Fix the background slowness and anisotropy` (rejected: Neither parameter is supplied; the exact angular law lets both be fitted as linear coefficients without shifting their contribution into the clocks.); `Correct every flagged pick by exactly 6 ms` (rejected: First inspect the measured skip residuals. Exclusion makes fewer assumptions if their offsets vary.)

```bash
cat > /holdout_task/tomography_solver.py <<'PY'
"""Straight-ray inversion for one Gaussian plume and receiver clock offsets."""
import numpy as np
from scipy.linalg import helmert
from scipy.optimize import least_squares
from scipy.special import ndtr


class _RayModel:
    def __init__(self, survey):
        self.w, self.h = float(survey['W_m']), float(survey['H_m'])
        self.r = float(survey.get('R_plume_m', 5.5))
        zs = np.asarray(survey['z_src_m'], dtype=float)
        zr = np.asarray(survey['z_rcv_m'], dtype=float)
        self.ns, self.nr = len(zs), len(zr)
        self.zs = np.repeat(zs, self.nr)
        dz = (zr[None, :] - zs[:, None]).ravel()
        length = np.hypot(self.w, dz)
        self.length = length
        self.u, self.v = self.w / length, dz / length
        self.clock = helmert(self.nr).T if self.nr > 1 else np.empty((1, 0))
        self.design = np.column_stack((length / self.w, -length * self.v**2 / self.w,
                                       np.tile(self.clock, (self.ns, 1))))
        self.y = 1000 * np.asarray(survey['T_obs_s'], dtype=float).ravel()

    def kernel(self, x, z, gradient=False):
        r = self.r
        dz = z - self.zs
        a = (self.u*x + self.v*dz) / r
        b = (self.length / r) - a
        d = (-self.v*x + self.u*dz) / r
        cross = np.exp(-0.5*d*d)
        k = np.sqrt(2*np.pi)*cross*(ndtr(b)-ndtr(-a))
        if not gradient:
            return k
        along = cross*(np.exp(-0.5*a*a)-np.exp(-0.5*b*b)) / r
        across = -k*d/r
        return k, along*self.u-across*self.v, along*self.v+across*self.u

    def fit(self, p, y, loss='soft_l1', scale=0.20, keep=slice(None)):
        design = self.design[keep]

        def fun(v):
            return design @ v[3:] + v[2]*self.kernel(*v[:2])[keep] - y

        def jac(v):
            k, kx, kz = self.kernel(*v[:2], gradient=True)
            return np.column_stack((v[2]*kx[keep], v[2]*kz[keep], k[keep], design))

        n = self.design.shape[1]
        return least_squares(fun, p, jac=jac, loss=loss, f_scale=scale,
                             bounds=(np.r_[0, 0, 0, np.full(n, -np.inf)],
                                     np.r_[self.w, self.h, np.full(n+1, np.inf)]),
                             x_scale='jac', max_nfev=100, ftol=1e-8)

    def starts(self):
        A, y = self.design, self.y
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        for _ in range(15):
            err = y - A @ beta
            w = (1 + (err / 0.35)**2)**(-0.25)
            beta = np.linalg.lstsq(A*w[:, None], y*w, rcond=None)[0]
        q, _ = np.linalg.qr(A*w[:, None], mode='reduced')
        v = (y-A @ beta)*w
        v -= q @ (q.T @ v)
        step = self.r * 0.53
        gx = np.linspace(0, self.w, int(np.ceil(self.w / step)) + 1)
        gz = np.linspace(0, self.h, int(np.ceil(self.h / step)) + 1)
        points = np.stack(np.meshgrid(gx, gz), axis=-1).reshape(-1, 2)
        scores, strengths = [], []
        for x, z in points:
            wk = self.kernel(x, z)*w
            wk -= q @ (q.T @ wk)
            norm_sq = wk @ wk
            ip = wk @ v
            scores.append(max(ip, 0.) / np.sqrt(max(norm_sq, 1e-20)))
            strengths.append(max(ip / max(norm_sq, 1e-20), 0.02))
        chosen = []
        for i in np.argsort(scores)[::-1]:
            loc = points[i]
            if any(np.linalg.norm(loc - old[:2]) < 1.5*self.r for old in chosen):
                continue
            beta = np.linalg.lstsq(A*w[:, None], (y-strengths[i]*self.kernel(*loc))*w,
                                   rcond=None)[0]
            chosen.append(np.r_[loc, strengths[i], beta])
            if len(chosen) == 6:
                break
        return chosen

    def solve(self):
        fits = [self.fit(p, self.y) for p in self.starts()]
        best = min(fits, key=lambda f: f.cost)
        keep = np.abs(best.fun) < 2.5
        refined = self.fit(best.x, self.y[keep], loss='soft_l1', scale=0.40, keep=keep)
        return refined.x


def _bound_clocks(clock):
    if np.max(np.abs(clock)) <= 4.0:
        return clock
    lo, hi = np.min(clock)-4, np.max(clock)+4
    for _ in range(48):
        mid = (lo+hi)/2
        if np.clip(clock-mid, -4, 4).sum() > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(clock-(lo+hi)/2, -4, 4)


def invert_plume_and_clocks(survey: dict) -> dict:
    model = _RayModel(survey)
    p = model.solve()
    clocks = _bound_clocks(model.clock @ p[5:]) / 1000
    return {'plume_xz_m': p[:2].tolist(), 'rcv_clock_bias_s': clocks.tolist()}
PY
cat > /tmp/probe_tomo.py <<'PY'
import sys,pickle,time
sys.path.insert(0,'/holdout_task')
import numpy as np
from tomography_solver import _RayModel, _bound_clocks
for row in pickle.load(open('/holdout_task/public_surveys.pkl','rb')):
    a=_RayModel(row['survey']); t=time.monotonic(); p=a.solve(); dt=time.monotonic()-t
    e=a.y-a.design@p[3:]-p[2]*a.kernel(*p[:2]); c=_bound_clocks(a.clock@p[5:])/1000
    b=row['gt']; loc=np.linalg.norm(p[:2]-b['plume_xz_m']); clk=np.sqrt(np.mean((c-b['rcv_clock_bias_s'])**2))
    print('params x,z,R*delta_s,W*s0,W*s0*eps [m,ms]',p[:5], 'error [m,s]',loc,clk,'duration',dt)
    print('core MAD scale (ms)',1.4826*np.median(np.abs(e-np.median(e))),'skips',np.round(e[np.abs(e)>2.5],4))
    core=np.abs(e)<2.5
    for label, mask in [('upper',a.zs<np.median(a.zs)),('lower',a.zs>=np.median(a.zs)),('steep',np.abs(a.v)>.6),('shallow',np.abs(a.v)<=.6)]:
        k=mask&core; print(label,int(k.sum()),'rms ms',np.sqrt(np.mean(e[k]**2)))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/probe_tomo.py
```
- **Container Exit Code**: `0` (`577 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tomography_solver.py']`
- **Stdout (tail)**:
```text
params x,z,R*delta_s,W*s0,W*s0*eps [m,ms] [16.42120931 34.04224971  0.35695217 16.80502842  2.36444469] error [m,s] 0.08940363673906761 5.283220386841697e-06 duration 0.030692653032019734
core MAD scale (ms) 0.03720887529719829 skips [-5.5061  5.5028  5.4619  5.4507  5.562   5.4965 -5.4957  5.5203 -5.528
 -5.4997]
upper 108 rms ms 0.03285111293317257
lower 122 rms ms 0.031051281526236842
steep 85 rms ms 0.03212570525952653
shallow 145 rms ms 0.031781382708571546
params x,z,R*delta_s,W*s0,W*s0*eps [m,ms] [25.89903345 51.45218888  0.3924024  15.20452938  2.74499725] error [m,s] 0.11171458170473594 1.2211173656554394e-05 duration 0.03228986309841275
core MAD scale (ms) 0.0317612944786767 skips [-5.5068  5.4651  5.5167 -5.4685  5.5104  5.5262 -5.5114  5.5144  5.517
  5.493 ]
upper 108 rms ms 0.03168122217012815
lower 122 rms ms 0.031554689607103656
steep 85 rms ms 0.032334960187960296
shallow 145 rms ms 0.03118388968029784

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1fcebc931e61d702 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1fcebc93 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`40095 ms` LLM)
- **Rationale**: Public source and receiver jackknives test dependence on individual acquisition lines. Independently integrated synthetic surveys test whether near-boundary geometry, weaker signals, or sparse coverage expose a localization or clock bottleneck before submission.
- **Alternatives Weighed**: `Correct skipped picks by the public offset of approximately 5.5 ms` (rejected: The specification gives approximately 6 ms, so excluding flagged picks transfers better than assuming the public offset is exact.)

```bash
cat > /tmp/stress_tomo.py <<'PY'
import pickle, sys, time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
sys.path.insert(0, '/holdout_task')
from tomography_solver import invert_plume_and_clocks

rng = np.random.default_rng(501)

def check(s, xy, clocks):
    start = time.perf_counter()
    out = invert_plume_and_clocks(s)
    dt = time.perf_counter() - start
    assert set(out) == {'plume_xz_m', 'rcv_clock_bias_s'}
    assert isinstance(out['plume_xz_m'], list) and len(out['plume_xz_m']) == 2
    assert isinstance(out['rcv_clock_bias_s'], list) and len(out['rcv_clock_bias_s']) == len(clocks)
    assert all(type(v) is float and np.isfinite(v) for v in out['plume_xz_m'] + out['rcv_clock_bias_s'])
    c = np.array(out['rcv_clock_bias_s'])
    assert abs(c.sum()) < 1e-12 and np.max(abs(c)) <= 0.004 + 1e-14
    return (np.linalg.norm(np.array(out['plume_xz_m']) - xy), np.sqrt(np.mean((c-clocks)**2)), dt)

def report(label, scores):
    a = np.asarray(scores)
    print(label, 'n', len(a), 'max [m,s,runtime s]', a.max(axis=0), 'failures [location,clock,runtime]', (a > [1.8, .00025, 15]).sum(axis=0), flush=True)

public=[]; sources=[]; receivers=[]
for entry in pickle.loads(Path('/holdout_task/public_surveys.pkl').read_bytes()):
    s, gt = entry['survey'], entry['gt']
    x = np.asarray(gt['plume_xz_m']); c = gt['rcv_clock_bias_s']
    public.append(check(s,x,c))
    for i in range(len(s['z_src_m'])):
        sources.append(check(dict(s,z_src_m=np.delete(s['z_src_m'],i), T_obs_s=np.delete(s['T_obs_s'],i,axis=0)),x,c))
    for j in range(len(c)):
        sub = np.delete(c,j)
        receivers.append(check(dict(s,z_rcv_m=np.delete(s['z_rcv_m'],j), T_obs_s=np.delete(s['T_obs_s'],j,axis=1)),x,sub-sub.mean()))
report('public full',public)
report('public omit source',sources)
report('public omit receiver; retained clocks recentered',receivers)

for label, count in [('nominal', 20), ('near wells', 24), ('near depth ends', 24), ('sparser weak plume', 20)]:
    results=[]; meta=[]
    for i in range(count):
        w,h=(40.,80.) if label != 'sparser weak plume' else (rng.uniform(30,55), rng.uniform(60,95))
        ns,nr=(15,16) if label != 'sparser weak plume' else (rng.integers(9,13),rng.integers(9,14))
        zs=np.linspace(.075*h,.925*h,ns); zr=np.linspace(.0625*h,.9375*h,nr)
        x,z=rng.uniform(.22*w,.78*w), rng.uniform(.18*h,.82*h)
        if label=='near wells':
            x=[0.0, 1.5, 4., w-4., w-1.5, w][i%6]
        if label=='near depth ends':
            z=[.04*h,.075*h,.12*h,.88*h,.925*h,.96*h][i%6]
        amp=rng.uniform(5.5e-5,1e-4) if label!='sparser weak plume' else rng.uniform(2.5e-5,5.5e-5)
        dz=zr[None,:]-zs[:,None]; length=np.hypot(w,dz)
        times=length*rng.uniform(.00035,.0005)*(1-rng.uniform(.07,.22)*(dz/length)**2)
        # Independently evaluate the exact straight-ray integral by quadrature.
        for j,k in np.ndindex(times.shape):
            times[j,k]+=amp*length[j,k]*quad(lambda t: np.exp(-((w*t-x)**2+(zs[j]+t*dz[j,k]-z)**2)/(2*5.5**2)),0,1)[0]
        c=rng.uniform(-1,1,nr); c-=c.mean(); c*=.004/max(abs(c))
        times+=c+rng.normal(0, 3.5e-5 if label!='sparser weak plume' else 6e-5, times.shape)
        skip=rng.choice(times.size,round(.04*times.size),replace=False)
        times.flat[skip]+=rng.choice([-1,1],len(skip))*rng.uniform(.0055,.0065,len(skip))
        s=dict(W_m=w,H_m=h,z_src_m=zs,z_rcv_m=zr,T_obs_s=times)
        results.append(check(s,(x,z),c)); meta.append((i,w,h,ns,nr,x,z,amp))
    report(label,results)
    for col in (0,1):
        i=np.argmax(np.asarray(results)[:,col]); print('worst case',meta[i],results[i],flush=True)
PY
python3 -I /tmp/stress_tomo.py
```
- **Container Exit Code**: `0` (`6659 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public full n 2 max [m,s,runtime s] [1.11714582e-01 1.22111737e-05 3.35339182e-02] failures [location,clock,runtime] [0 0 0]
public omit source n 30 max [m,s,runtime s] [1.45044899e-01 1.51167496e-05 3.72749250e-02] failures [location,clock,runtime] [0 0 0]
public omit receiver; retained clocks recentered n 32 max [m,s,runtime s] [1.38462049e-01 1.34455528e-05 3.62293802e-02] failures [location,clock,runtime] [0 0 0]
nominal n 20 max [m,s,runtime s] [2.61093393e-01 1.21875160e-05 3.74791441e-02] failures [location,clock,runtime] [0 0 0]
worst case (11, 40.0, 80.0, 15, 16, 18.943125393808046, 14.861699847405946, 6.660458384960485e-05) (0.26109339285610406, 1.0645002542073513e-05, 0.02842833800241351)
worst case (5, 40.0, 80.0, 15, 16, 23.44420820074835, 26.12876708641933, 6.771351963770246e-05) (0.16619921062164467, 1.2187516048259938e-05, 0.031564385863021016)
near wells n 24 max [m,s,runtime s] [1.18046123e+00 2.32156782e-05 4.05506711e-02] failures [location,clock,runtime] [0 0 0]
worst case (16, 40.0, 80.0, 15, 16, 38.5, 17.771964147604578, 6.573649869106133e-05) (1.1804612342935348, 1.8883319176718765e-05, 0.0348189149517566)
worst case (10, 40.0, 80.0, 15, 16, 38.5, 22.420046994659092, 9.400220958151727e-05) (0.6654537770148525, 2.321567821174715e-05, 0.03422398492693901)
near depth ends n 24 max [m,s,runtime s] [1.49567439e+00 1.90305048e-05 4.74554780e-02] failures [location,clock,runtime] [0 0 0]
worst case (11, 40.0, 80.0, 15, 16, 14.995693959390755, 76.8, 7.05366584467044e-05) (1.4956743901789535, 7.899546011904137e-06, 0.03139237500727177)
worst case (19, 40.0, 80.0, 15, 16, 27.18147537726033, 6.0, 9.40624715413008e-05) (1.0433679669843796, 1.9030504763063696e-05, 0.03615060611627996)
sparser weak plume n 20 max [m,s,runtime s] [1.07868719e+00 3.78812218e-05 3.25696520e-02] failures [location,clock,runtime] [0 0 0]
worst case (4, 47.178529533302665, 60.97335405247994, 9, 11, 15.594859868551483, 28.236040194719145, 3.0098500635661733e-05) (1.0786871884597502, 1.8039850702336223e-05, 0.022085315082222223)
worst case (3, 52.00752070149771, 85.69950748029505, 9, 10, 19.178566730225725, 64.34059917008533, 3.9551379470506756e-05) (0.15093879985746964, 3.78812218241813e-05, 0.024240524042397738)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1fcebc931e61d702 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1fcebc93 (directory).
```

### Turn 4 — `submit_completed_task` (`13000 ms` LLM)
- **Rationale**: The archived solver passed clean-process checks for output schema, clock bounds and zero sum, both accuracy targets, and runtime on public surveys, source and receiver jackknives, and independently integrated synthetic stresses. No tested slice warrants another edit.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`24` turns remaining; artifact status={'/holdout_task/tomography_solver.py': 'FILE:4519'}): Do NOT lock prematurely with `24` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tomography_solver.py`. Worst public-survey errors were 0.112 m for plume location and 12.22 µs for receiver clocks. Across the public jackknives and synthetic stress tests, worst errors were 1.496 m and 37.89 µs, and the longest run took 0.048 s. All tested cases met the targets.`

### Turn 5 — `execute_in_remote_docker` (`35781 ms` LLM)
- **Rationale**: Compare the exact angular model with an isotropic ablation on the same unseen source picks. Receiver-specific prediction errors reveal timing offsets that pooled residuals could hide. If the full model still has materially biased receiver slices, investigate that before changing the solver; otherwise retain the physically specified model.
- **Alternatives Weighed**: `Test relay modes, classification balance, and unrelated drift or rational bases` (rejected: This is continuous traveltime inversion with no relay state, class labels, or temporal drift operator. Receiver and ray-angle slices directly test the relevant nuisance separation.)

```bash
cat > /tmp/audit_tomo.py <<'PY'
import pickle
import sys
import time
import numpy as np
sys.path.insert(0, '/holdout_task')
from tomography_solver import _RayModel, invert_plume_and_clocks

with open('/holdout_task/public_surveys.pkl', 'rb') as f:
    entries = pickle.load(f)
for num, entry in enumerate(entries, 1):
    s, gt = entry['survey'], entry['gt']
    a = _RayModel(s)
    start = time.perf_counter()
    out = invert_plume_and_clocks(s)
    elapsed = time.perf_counter() - start
    c = np.asarray(out['rcv_clock_bias_s'])
    loc = np.linalg.norm(np.asarray(out['plume_xz_m']) - gt['plume_xz_m'])
    clk = np.sqrt(np.mean((c - gt['rcv_clock_bias_s'])**2))
    assert set(out) == {'plume_xz_m', 'rcv_clock_bias_s'}
    assert len(out['plume_xz_m']) == 2 and c.shape == (a.nr,)
    assert all(type(x) is float and np.isfinite(x) for x in out['plume_xz_m'] + out['rcv_clock_bias_s'])
    assert abs(c.sum()) < 1e-12 and np.max(abs(c)) <= .004 + 1e-14
    assert loc <= 1.8 and clk <= .00025 and elapsed < 15
    cols = [0, *range(2, a.design.shape[1])]
    residuals = np.empty((2, a.ns, a.nr))
    fold_err = np.empty((2, a.ns, 2))
    for i in range(a.ns):
        train = dict(s, z_src_m=np.delete(s['z_src_m'], i), T_obs_s=np.delete(s['T_obs_s'], i, axis=0))
        held = slice(i*a.nr, (i+1)*a.nr)
        for k in range(2):
            m = _RayModel(train)
            design = a.design
            if k:
                m.design = m.design[:, cols]
                design = design[:, cols]
            p = m.solve()
            residuals[k, i] = a.y[held] - (design[held]@p[3:] + p[2]*a.kernel(*p[:2])[held])
            fold_err[k, i] = (np.linalg.norm(p[:2]-gt['plume_xz_m']), np.sqrt(np.mean((m.clock@p[(4 if k else 5):] - 1000*gt['rcv_clock_bias_s'])**2))*1000)
    # Use the same held-out picks for both models; the 2.5 ms gate lies between
    # the sub-0.1 ms inlier residuals and the observed ~5.5 ms skips.
    core = np.abs(residuals[0]) < 2.5
    steep = (np.abs(a.v) > .6).reshape(core.shape)
    rms = lambda e: 1000*np.sqrt(np.mean(e**2))
    print(f'public {num}: center {loc:.4f} m, clocks {clk*1e6:.2f} us, runtime {elapsed:.4f} s, held-out skips {np.count_nonzero(~core)}')
    for label, e, err in zip(('anisotropic', 'isotropic ablation'), residuals, fold_err):
        by_rcv = [(rms(e[core[:,j],j]), 1000*abs(np.mean(e[core[:,j],j]))) for j in range(a.nr)]
        print(label, 'held-out RMS all/shallow/steep (us)', [round(rms(e[mask]),2) for mask in (core, core & ~steep, core & steep)], 'worst receiver RMS/absolute mean (us)', np.round(np.max(by_rcv,axis=0),2), 'worst fold location (m)/clock (us)', np.round(err.max(axis=0),4))
PY
python3 -I /tmp/audit_tomo.py
```
- **Container Exit Code**: `0` (`3909 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 1: center 0.0894 m, clocks 5.28 us, runtime 0.0304 s, held-out skips 10
anisotropic held-out RMS all/shallow/steep (us) [34.9, 34.81, 35.06] worst receiver RMS/absolute mean (us) [44.91  0.51] worst fold location (m)/clock (us) [0.1143 6.8318]
isotropic ablation held-out RMS all/shallow/steep (us) [810.55, 909.54, 605.42] worst receiver RMS/absolute mean (us) [2322.29 1386.64] worst fold location (m)/clock (us) [  48.2136 5441.8983]
public 2: center 0.1117 m, clocks 12.21 us, runtime 0.0375 s, held-out skips 10
anisotropic held-out RMS all/shallow/steep (us) [34.95, 34.55, 35.62] worst receiver RMS/absolute mean (us) [47.76  0.37] worst fold location (m)/clock (us) [ 0.145  15.1167]
isotropic ablation held-out RMS all/shallow/steep (us) [985.12, 1188.52, 464.99] worst receiver RMS/absolute mean (us) [2284.95 1165.54] worst fold location (m)/clock (us) [  49.2773 7531.5663]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1fcebc931e61d702 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1fcebc93 (directory).
```

### Turn 6 — `submit_completed_task` (`16437 ms` LLM)
- **Rationale**: The clean-process audit confirmed the output schema, clock bounds and zero-sum constraint. The same-pick ablation confirms that the angular term is necessary, and the unchanged solver meets both targets across the tested public jackknife and synthetic slices.
- **ATMS NOGOOD DECLARED**: Refuted `An isotropic background plus receiver offsets can replace the specified angular slowness term.` | Witness: `On the same held-out source picks, worst receiver RMS was 2,322 µs with the isotropic ablation versus 47.76 µs with anisotropy.` | Blanket Diagnosis: `A receiver offset is constant across source depths and cannot represent the angle-dependent background traveltime.` | Sublated Pivot: `Retain the exact angular basis and fit it jointly with zero-sum receiver clock contrasts.`
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`22` turns remaining; artifact status={'/holdout_task/tomography_solver.py': 'FILE:4519'}): Do NOT lock prematurely with `22` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tomography_solver.py`. Worst errors were 0.112 m for plume location and 12.22 µs for receiver clocks on the public surveys, and 1.496 m and 37.89 µs across all stress tests. All tested cases met both accuracy targets and ran in under 0.05 s.`

### Turn 7 — `execute_in_remote_docker` (`27948 ms` LLM)
- **Rationale**: Test the open angular-confounding finding on real picks using source pairs withheld together. Differencing at a fixed receiver cancels its clock exactly. If anisotropy still improves predictions, the existing candidate resolves the finding; if not, investigate whether the earlier gain depended on clock estimation.

```bash
cat > /tmp/tomo_clock_free.py <<'PY'
import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from tomography_solver import _RayModel

with open('/holdout_task/public_surveys.pkl', 'rb') as f:
    entries = pickle.load(f)
for n, entry in enumerate(entries, 1):
    s = entry['survey']
    full = _RayModel(s)
    cols = [0, *range(2, full.design.shape[1])]
    residual = np.empty((2, full.ns - 1, full.nr))
    for i in range(full.ns - 1):
        rows = np.ones(full.ns, dtype=bool)
        rows[i:i+2] = False
        train = dict(s, z_src_m=s['z_src_m'][rows], T_obs_s=s['T_obs_s'][rows])
        a = slice(i*full.nr, (i+1)*full.nr)
        b = slice((i+1)*full.nr, (i+2)*full.nr)
        da = full.design[b] - full.design[a]
        assert np.all(da[:, 2:] == 0)
        for j in range(2):
            model = _RayModel(train)
            if j:
                model.design = model.design[:, cols]
            p = model.solve()
            k = full.kernel(*p[:2])
            residual[j, i] = full.y[b] - full.y[a] - da[:, :2-j] @ p[3:5-j] - p[2]*(k[b]-k[a])
    # Use identical pairs in both models; the gate separates differential skips
    # (about 5.5 ms) from ordinary inlier differences (about 0.05 ms).
    core = np.abs(residual[0]) < 2.5
    rms = 1000*np.sqrt(np.mean(residual[:, core]**2, axis=1))
    receiver_rms = np.array([1000*np.sqrt(np.mean(residual[:, core[:,j], j]**2, axis=1)) for j in range(full.nr)])
    print('survey', n, 'pairs kept', core.sum(), 'held-out differential RMS [anisotropic, isotropic] us', np.round(rms,2), 'worst receiver RMS us', np.round(receiver_rms.max(axis=0),2))
    assert rms[0] < rms[1]
PY
python3 -I /tmp/tomo_clock_free.py
```
- **Container Exit Code**: `0` (`3615 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
survey 1 pairs kept 206 held-out differential RMS [anisotropic, isotropic] us [ 46.23 278.74] worst receiver RMS us [ 68.38 691.16]
survey 2 pairs kept 207 held-out differential RMS [anisotropic, isotropic] us [ 48.36 291.08] worst receiver RMS us [ 69.8  736.84]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1fcebc931e61d702 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1fcebc93 (directory).
```

### Turn 8 — `submit_completed_task` (`13100 ms` LLM)
- **Rationale**: Differencing withheld source pairs at the same receiver cancels its clock exactly. The retained anisotropic model reduced differential RMS from 278.74–291.08 µs to 46.23–48.36 µs, confirming that the improvement does not depend on fitted clocks. The unchanged artifact already passed isolated-process checks for accuracy, runtime, output schema, and clock constraints.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tomography_solver.py`. Worst public-survey errors were 0.112 m for plume location and 12.22 µs for receiver clocks. Across all tested cases, errors stayed within 1.496 m and 37.89 µs, and every run took under 0.05 s.`

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
