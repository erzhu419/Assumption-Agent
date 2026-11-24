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
| **Turns Used / Max Turns** | `23 / 28` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `2` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `999,621` (`973,278` / `196,224` / `26,343`) |
| **Total Episode Wall Time (s)** | `1125.9s` |

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

### Turn 1 — `execute_in_remote_docker` (`26343 ms` LLM)
- **Rationale**: Inspect which background parameters are supplied, how rays are laid out, and whether public truth is available. Those findings determine which parameters must be fitted and which validation metrics can be measured directly.
- **Alternatives Weighed**: `Reuse the pitch-catch waveform localizer directly` (rejected: The specified observations are scalar traveltime picks, not carrier-resolved waveforms; inspect the source only for transferable nuisance-estimation ideas.)

```bash
ls -la /holdout_task; python3 - <<'PY'
import pickle, numpy as np, scipy
with open('/holdout_task/public_surveys.pkl','rb') as f: data=pickle.load(f)
def describe(x, depth=0):
    pad='  '*depth
    if isinstance(x,dict):
        print(pad,'dict keys:',list(x))
        for k,v in x.items():
            print(pad,repr(k)); describe(v,depth+1)
    elif isinstance(x,(list,tuple)) and any(isinstance(v,dict) for v in x):
        print(pad,type(x).__name__,len(x))
        for v in x: describe(v,depth+1)
    elif isinstance(x,(np.ndarray,list,tuple)):
        a=np.asarray(x); print(pad,a.shape,a.dtype,'sample:',a.ravel()[:10]);
        if a.size and a.dtype.kind in 'iuf': print(pad,'min/max/nonfinite:',np.nanmin(a),np.nanmax(a),np.count_nonzero(~np.isfinite(a)))
    else: print(pad,repr(x))
describe(data); print('SciPy',scipy.__version__)
PY
ls -l /tmp/hegel_wormhole_skills; grep -nE '^def |waveform|reference_|gain' /tmp/hegel_wormhole_skills/autogenous_dispersive_wave_tomography.py | head -65
```
- **Container Exit Code**: `0` (`494 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
dict keys: ['plume_xz_m', 'rcv_clock_bias_s']
     'plume_xz_m'
       (2,) float64 sample: [26.  51.5]
       min/max/nonfinite: 26.0 51.5 0
     'rcv_clock_bias_s'
       (16,) float64 sample: [-0.0009812   0.00347274 -0.00137483 -0.00143901 -0.00034719 -0.00111101
  0.00170437 -0.00184997  0.0031499   0.00290556]
       min/max/nonfinite: -0.002568385497994811 0.0034727403303111385 0
SciPy 1.13.0
total 36
-rw-r--r-- 1 ubuntu ubuntu  4369 Nov 24 08:12 __init__.py
drwxr-xr-x 2 root   root    4096 Nov 24 08:12 __pycache__
-rw-r--r-- 1 ubuntu ubuntu  8025 Nov 24 08:12 autogenous_creep_corrected_retraction_calibration.py
-rw-r--r-- 1 ubuntu ubuntu 15402 Nov 24 08:12 autogenous_dispersive_wave_tomography.py
13:def localize_pitch_catch(sensor_xy: np.ndarray, pairs: np.ndarray, waveforms: np.ndarray, plate_dims: np.ndarray, sample_rate_hz: float, carrier_hz: float, *, thickness: float | None=None, reference_sensor_xy: np.ndarray | None=None, reference_pairs: np.ndarray | None=None, reference_waveforms: np.ndarray | None=None, reference_plate_dims: np.ndarray | None=None, reference_thickness: float | None=None, grid_step: float=0.009) -> tuple[tuple[float, float], dict]:
24:    of the target's longest side. The standardized endpoint gain ridge is 1.8 in
35:    if any((h is not None and h <= 0 for h in (thickness, reference_thickness))):
122:    (p, z) = reduce_graph(pairs, waveforms)
126:    ref_args = (reference_sensor_xy, reference_pairs, reference_waveforms, reference_plate_dims)
130:        rs = np.asarray(reference_sensor_xy, float) / span
131:        (rp, rz) = reduce_graph(reference_pairs, reference_waveforms)
132:        rplate = np.asarray(reference_plate_dims, float) / span
140:        both_thicknesses = thickness is not None and reference_thickness is not None
142:            prior[1] *= np.sqrt(reference_thickness / thickness)
143:        same = rs.shape == sensors.shape and np.allclose(rs, sensors, rtol=0, atol=1e-08) and np.allclose(rplate, plate, rtol=0, atol=1e-08) and (not both_thicknesses or abs(reference_thickness / thickness - 1) < 1e-06)
272:    return ((float(xy[0]), float(xy[1])), {'used_twin_subtraction': twin is not None, 'late_coda_correlation': rho, 'launch_periods': float(clock[0]), 'slowness_periods_per_length': float(clock[1] / span), 'reference_launch_periods': None if prior is None else float(prior[0]), 'transferred_slowness_periods_per_length': None if prior is None else float(prior[1] / span), 'scatter_weight': scatter_weight})
```

### Turn 2 — `execute_in_remote_docker` (`34191 ms` LLM)
- **Rationale**: Fit the stated forward law at the known public centers to measure the noise, slip separation, and angular residuals. If residuals are structured instead of separating cleanly, revisit the forward law before choosing a robust loss.
- **Alternatives Weighed**: `Transfer waveform or echo processing` (rejected: The public data contain scalar picks only; the transferable idea is to separate nuisance terms from plume evidence.)

```bash
sed -n '13,35p;69,119p;179,235p' /tmp/hegel_wormhole_skills/autogenous_dispersive_wave_tomography.py; OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle
import numpy as np
from scipy.special import erf
from scipy.optimize import least_squares
with open('/holdout_task/public_surveys.pkl','rb') as f: surveys=pickle.load(f)
for k,c in enumerate(surveys):
 s,g=c['survey'],c['gt']; W=s['W_m']; R=s['R_plume_m']; zs=s['z_src_m'][:,None]; dz=s['z_rcv_m'][None,:]-zs; l=np.hypot(W,dz); x,z=g['plume_xz_m']; p=(W*(z-zs)-dz*x)/l; q=(W*x+dz*(z-zs))/l
 gauss=np.exp(-.5*(p/R)**2)*R*np.sqrt(np.pi/2)*(erf((l-q)/(np.sqrt(2)*R))+erf(q/(np.sqrt(2)*R)))
 A=np.stack((l,-dz**2/l,gauss),axis=-1).reshape(-1,3); b=(1e3*(s['T_obs_s']-g['rcv_clock_bias_s'])).ravel()
 p0=least_squares(lambda v:A@v-b,[.4,.03,.2],loss='cauchy',f_scale=.35).x; e=(b-A@p0).reshape(l.shape); good=np.abs(e)<2.; p1=np.linalg.lstsq(A[good.ravel()],b[good.ravel()],rcond=None)[0]; e=(b-A@p1).reshape(l.shape)
 print('survey',k,'s0, eps, delta_s',p1[0]/1e3,p1[1]/p1[0],p1[2]/1e3,'clock sum',np.sum(g['rcv_clock_bias_s']))
 print('inlier ms RMS/max',np.sqrt(np.mean(e[good]**2)),np.max(np.abs(e[good])),'gross residual ms',np.round(e[~good],3),'per receiver',(~good).sum(axis=0))
 print('plume max/median ms',np.max(gauss)*p1[2],np.median(gauss)*p1[2], 'inlier RMS by angle', [round(np.sqrt(np.mean(e[good & mask]**2)),5) for mask in [np.abs(dz)<12,(np.abs(dz)>=12)&(np.abs(dz)<35),np.abs(dz)>=35]])
PY
```
- **Container Exit Code**: `0` (`505 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
sigma) ** 2 + 1j * (phase[0] * u ** 2 + phase[1] * u)) * (abs(u) <= (radius + 1e-08) / period)
        wave = atom(np.arange(-radius, radius + 1) / period)
        energy = fftconvolve(np.ones(len(t)), abs(wave[::-1]) ** 2, mode='same')
        coda_traces = np.empty((len(t), count))
        ratios = np.empty_like(coda_traces)
        penalty = np.r_[0.002, np.full(4, 0.025), 0.075, np.full(4, 0.32)]
        for i in range(count):
            u = t[None, :] - np.r_[direct[i], walls[i]][:, None]
            packet = atom(u)
            atoms = np.vstack((packet, u * packet / sigma))
            gram = atoms.conj() @ atoms.T
            diag = np.maximum(gram.diagonal().real, 0.15 * period)
            chol = np.linalg.cholesky(gram + np.diag(penalty * diag))
            rhs = atoms.conj() @ base[:, i]
            coef = np.linalg.solve(chol.conj().T, np.linalg.solve(chol, rhs))
            residual = base[:, i] - coef @ atoms
            dot = fftconvolve(residual, wave[::-1].conj(), mode='same')
            cross = fftconvolve(atoms.conj(), wave[None, ::-1], mode='same', axes=-1)
            leverage = np.sum(abs(np.linalg.solve(chol, cross)) ** 2, axis=0)
            remaining = np.maximum(energy - leverage, 0.06 * period)
            safe_energy = np.maximum(energy, 0.1)
            resolved = np.clip(remaining / safe_energy, 0, 1)
            detect = abs(dot) / np.sqrt(remaining * safe_energy) * resolved ** 0.25
            detect[(t < direct[i]) | (t > t[-1] - 0.6)] = 0
            coda_traces[:, i] = detect
            ratios[:, i] = detect / max(abs(coef[0]), 0.012)
        depth = float(np.quantile(ratios, 0.99))
survey 0 s0, eps, delta_s 0.00042011466875072713 0.1406316664179732 6.487975213414033e-05 clock sum 4.336808689942018e-19
inlier ms RMS/max 0.032527776325716774 0.08717676375424688 gross residual ms [-5.513  5.515  5.465  5.462  5.562  5.498 -5.499  5.52  -5.529 -5.504] per receiver [2 3 0 0 2 0 0 0 0 0 1 1 0 1 0 0]
plume max/median ms 0.8943725789172231 0.18795287801309685 inlier RMS by angle [0.02969, 0.03448, 0.03252]
survey 1 s0, eps, delta_s 0.00038004242635738145 0.1807746450155551 7.261058522879244e-05 clock sum -8.673617379884035e-19
inlier ms RMS/max 0.03317851827311705 0.09051888051656931 gross residual ms [-5.516  5.474  5.518 -5.48   5.5    5.522 -5.516  5.525  5.526  5.496] per receiver [1 2 0 0 1 0 0 1 0 1 0 0 0 1 2 1]
plume max/median ms 1.0008387810178119 0.10966367329278241 inlier RMS by angle [0.03217, 0.03195, 0.03574]
```

### Turn 3 — `execute_in_remote_docker` (`59345 ms` LLM)
- **Rationale**: Project out the two background coefficients and zero-sum clocks before comparing plume locations, then refit on the updated slip mask. Public blind recovery will reveal any errors in the projection, units, or nonlinear optimization before held-out-ray and regime testing.
- **Alternatives Weighed**: `Round residuals to exact 6 ms cycle shifts` (rejected: Public shifts cluster around 5.5 ms; use the separation from ordinary pick noise to isolate slips without imposing an exact jump size.)

```bash
cd /holdout_task && cat > tomography_solver.py <<'PY'
"""Straight-ray traveltime inversion for a Gaussian plume and receiver clocks."""
import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.optimize import least_squares
from scipy.special import erf


class _Rays:
    def __init__(self, w, zs, zr, radius):
        self.r = radius
        dz = zr[None, :] - zs[:, None]
        self.length = np.hypot(w, dz).ravel()
        self.u = w / self.length
        self.v = dz.ravel() / self.length
        self.zs = np.repeat(zs, len(zr))
        self.receiver = np.tile(np.arange(len(zr)), len(zs))
        # Last clock is minus the sum of the others. The two background
        # coefficients are s0 and s0*epsilon, expressed in ms/m.
        clocks = np.eye(len(zr), dtype=float)[self.receiver]
        self.design = np.column_stack((self.length, -self.length * self.v**2,
                                       clocks[:, :-1] - clocks[:, -1:]))

    def kernel(self, centers, ind, derivative=False):
        x, z = np.asarray(centers).T
        u, v, length = self.u[ind], self.v[ind], self.length[ind]
        t = z[..., None] - self.zs[ind]
        p = u * t - v * x[..., None]
        q = v * t + u * x[..., None]
        r = self.r
        a, b = q / (np.sqrt(2) * r), (length - q) / (np.sqrt(2) * r)
        normal = np.exp(-0.5 * (p / r)**2)
        g = r * np.sqrt(np.pi / 2) * normal * (erf(a) + erf(b))
        if not derivative:
            return g
        gp = -p / r**2 * g
        gq = normal * (np.exp(-a*a) - np.exp(-b*b))
        return g, np.column_stack((-v*gp + u*gq, u*gp + v*gq))


def _clip_clock(clock):
    """Project onto the zero-sum clock box, in milliseconds."""
    if np.max(np.abs(clock)) <= 4.0:
        return clock
    lo, hi = np.min(clock)-4.0, np.max(clock)+4.0
    for _ in range(52):
        mid = (lo + hi) / 2
        if np.clip(clock-mid, -4., 4.).sum() > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(clock-(lo+hi)/2, -4., 4.)


def _invert(survey, include=None):
    w, h = float(survey['W_m']), float(survey['H_m'])
    zs = np.asarray(survey['z_src_m'], dtype=float)
    zr = np.asarray(survey['z_rcv_m'], dtype=float)
    rays = _Rays(w, zs, zr, float(survey.get('R_plume_m', 5.5)))
    y = 1000. * np.asarray(survey['T_obs_s'], dtype=float).ravel()
    available = np.isfinite(y) if include is None else np.isfinite(y) & include.ravel()
    D = rays.design
    Da, ya = D[available], y[available]
    bg = np.linalg.lstsq(Da, ya, rcond=None)[0]
    for _ in range(12):
        resid = Da @ bg - ya
        weight = (1 + (resid / .25)**2)**(-.25)
        bg = np.linalg.lstsq(Da * weight[:, None], ya * weight, rcond=None)[0]
    good = available & (np.abs(D @ bg-y) < 2.7)

    step = rays.r / 2.
    X, Z = np.meshgrid(np.linspace(0, w, int(np.ceil(w/step))+1),
                       np.linspace(0, h, int(np.ceil(h/step))+1))
    grid = np.column_stack((X.ravel(), Z.ravel()))
    xy = None
    for _ in range(5):
        ind = np.flatnonzero(good)
        Q, U = qr(D[ind], mode='economic')
        coeff_y = Q.T @ y[ind]
        yy = y[ind] - Q @ coeff_y

        def project(g):
            return g - Q @ (Q.T @ g)

        if xy is None:
            best = -1.
            for off in range(0, len(grid), 128):
                block = grid[off:off+128]
                raw = rays.kernel(block, ind)
                q = np.einsum('ij,jk->ik', raw, Q)
                # Profile the positive amplitude and the linear nuisance terms.
                power = np.sum(raw*raw, axis=1) - np.sum(q*q, axis=1)
                dot = np.einsum('ij,j->i', raw, yy)
                score = np.maximum(dot, 0.)**2 / np.maximum(power, 1.e-20)
                winner = int(np.argmax(score))
                if score[winner] > best:
                    xy, best = block[winner], score[winner]

        def profile(pos, jac=False):
            if jac:
                raw, dg = rays.kernel(pos, ind, derivative=True)
                dk = project(dg)
            else:
                raw = rays.kernel(pos, ind)
            k = project(raw)
            denom = max(np.dot(k, k), 1.e-20)
            amp = max(0., np.dot(k, yy) / denom)
            if not jac:
                return amp*k - yy
            if amp == 0.:
                return np.zeros_like(dk)
            da = ((yy - 2*amp*k) @ dk) / denom
            return amp*dk + k[:, None]*da

        xy = least_squares(profile, xy, jac=lambda x: profile(x, True),
                           bounds=([0., 0.], [w, h]), max_nfev=50).x
        g = rays.kernel(xy, ind)
        k = project(g)
        amplitude = max(0., np.dot(k, yy) / max(np.dot(k, k), 1.e-20))
        coef = solve_triangular(U, coeff_y - amplitude * (Q.T @ g))
        prediction = D @ coef + amplitude * rays.kernel(xy, slice(None))
        updated = available & (np.abs(y-prediction) < 2.7)
        if np.array_equal(updated, good):
            break
        good = updated

    clock = _clip_clock(np.r_[coef[2:], -np.sum(coef[2:])])
    return xy, clock/1000., prediction/1000.


def invert_plume_and_clocks(survey: dict) -> dict:
    """Return the plume center in meters and zero-sum receiver biases in seconds."""
    xy, clock, _ = _invert(survey)
    return {'plume_xz_m': xy.tolist(), 'rcv_clock_bias_s': clock.tolist()}
PY
mkdir -p /tmp/.hegel_candidate_archive && cp tomography_solver.py /tmp/.hegel_candidate_archive/v01_$(sha256sum tomography_solver.py | cut -c1-8)
python3 - <<'PY'
import pickle,time
import numpy as np
from tomography_solver import invert_plume_and_clocks
for i,c in enumerate(pickle.load(open('public_surveys.pkl','rb'))):
 t=time.perf_counter(); d=invert_plume_and_clocks(c['survey']); g=c['gt']; clock=np.array(d['rcv_clock_bias_s']); print(i,d['plume_xz_m'], 'meters',np.linalg.norm(d['plume_xz_m']-g['plume_xz_m']), 'clock us',1e6*np.sqrt(np.mean((clock-g['rcv_clock_bias_s'])**2)), 'sum',clock.sum(), 'seconds',time.perf_counter()-t)
PY
```
- **Container Exit Code**: `1` (`843 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tomography_solver.py']`
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
TypeError: unsupported operand type(s) for -: 'list' and 'list'
```

### Turn 4 — `execute_in_remote_docker` (`29799 ms` LLM)
- **Rationale**: Score the saved solver without supplying truth to the inversion, then remove each source and receiver in turn to check whether a single endpoint drives the result. Score retained clocks in their own zero-sum gauge when a receiver is removed.
- **Alternatives Weighed**: `Modify the solver after the failed smoke test` (rejected: The traceback came from subtracting two lists in the test, not from the solver; convert the outputs to arrays for scoring.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle, time
import numpy as np
from tomography_solver import _invert, invert_plume_and_clocks

def metrics(xy, clocks, truth):
    return np.linalg.norm(np.asarray(xy)-truth['plume_xz_m']), 1e6*np.sqrt(np.mean((np.asarray(clocks)-truth['rcv_clock_bias_s'])**2))

for i, case in enumerate(pickle.load(open('public_surveys.pkl', 'rb'))):
    s, g = case['survey'], case['gt']
    t = time.perf_counter()
    result = invert_plume_and_clocks(s)
    elapsed = time.perf_counter()-t
    clocks = np.asarray(result['rcv_clock_bias_s'])
    print('public', i, 'center m / clock us', metrics(result['plume_xz_m'], clocks, g), 'clock sum s', clocks.sum(), 'max abs clock s', np.abs(clocks).max(), 'runtime s', elapsed)
    y = s['T_obs_s']
    _, _, pred = _invert(s)
    good = np.abs(y-pred) < .0027
    print('full-fit inlier RMS us', 1e6*np.sqrt(np.mean((y[good]-pred[good])**2)), 'slips', (~good).sum())
    source, receiver = [], []
    for j in range(y.shape[0]):
        mask = np.ones(y.shape, bool)
        mask[j] = False
        xy, c, pred = _invert(s, mask)
        heldout = (y-pred)[j][good[j]]
        source.append([*metrics(xy,c,g), 1e6*np.sqrt(np.mean(heldout**2))])
    for j in range(y.shape[1]):
        keep = np.arange(y.shape[1]) != j
        reduced = {**s, 'z_rcv_m': np.asarray(s['z_rcv_m'])[keep], 'T_obs_s': y[:,keep]}
        xy, c, _ = _invert(reduced)
        true_clock = np.asarray(g['rcv_clock_bias_s'])[keep]
        true_clock = true_clock - true_clock.mean()
        receiver.append([np.linalg.norm(xy-g['plume_xz_m']), 1e6*np.sqrt(np.mean((c-true_clock)**2))])
    print('worst leave-one-source-out center m / clock us / held-out inlier RMS us', np.max(source,axis=0), 'fold indices', np.argmax(source,axis=0))
    print('worst leave-one-receiver-out center m / retained-clock us', np.max(receiver,axis=0), 'fold indices', np.argmax(receiver,axis=0))
PY
```
- **Container Exit Code**: `1` (`855 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0 center m / clock us (0.08947190669186113, 5.301980093089273) clock sum s 2.168404344971009e-19 max abs clock s 0.0033322092660293848 runtime s 0.008986250963062048
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 17, in <module>
ValueError: operands could not be broadcast together with shapes (15,16) (240,)
```

### Turn 5 — `execute_in_remote_docker` (`53670 ms` LLM)
- **Rationale**: Complete real-data source, receiver, and ray holdouts to detect reliance on particular observations. Use independently integrated stress cases to distinguish slip-rejection failures from weak geometric coverage and determine whether either warrants a solver change.
- **Alternatives Weighed**: `Change the inversion to return matrix-shaped internal predictions` (rejected: Only the validation script needs the matrix shape. The public API returned valid outputs on the first survey, so reshape predictions at the test boundary.)

```bash
cat > /tmp/tomography_validate.py <<'PY'
import pickle
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from tomography_solver import _invert, invert_plume_and_clocks


def metrics(xy, clock, truth_xy, truth_clock):
    return np.linalg.norm(np.asarray(xy)-truth_xy), 1e6*np.sqrt(np.mean((np.asarray(clock)-truth_clock)**2))


def run(s):
    start = time.perf_counter()
    out = invert_plume_and_clocks(s)
    dt = time.perf_counter()-start
    assert set(out) == {'plume_xz_m', 'rcv_clock_bias_s'}
    xy, c = out['plume_xz_m'], out['rcv_clock_bias_s']
    assert isinstance(xy, list) and len(xy) == 2 and isinstance(c, list) and len(c) == len(s['z_rcv_m'])
    assert all(type(x) is float and np.isfinite(x) for x in xy+c)
    assert 0 <= xy[0] <= s['W_m'] and 0 <= xy[1] <= s['H_m']
    assert abs(sum(c)) < 1e-12 and max(map(abs,c)) <= .004 + 1e-15 and dt < 15
    return xy,c,dt

cases = pickle.load(open('/holdout_task/public_surveys.pkl', 'rb'))
for i, case in enumerate(cases):
    s, g = case['survey'], case['gt']
    tx, tc = map(np.asarray, (g['plume_xz_m'], g['rcv_clock_bias_s']))
    xy,c,dt = run(s)
    center, clock = metrics(xy,c,tx,tc)
    assert center <= 1.8 and clock <= 250
    y = s['T_obs_s']
    _, _, pred = _invert(s)
    residual = y-pred.reshape(y.shape)
    good = np.abs(residual) < .0027
    print('public',i,'center m / clock us / runtime s',center,clock,dt,'inlier RMS us',1e6*np.sqrt(np.mean(residual[good]**2)),'slips',(~good).sum())
    source, receiver, ray = [], [], []
    for j in range(y.shape[0]):
        mask = np.ones(y.shape, bool)
        mask[j] = False
        xy,c,pred = _invert(s, mask)
        heldout = (y-pred.reshape(y.shape))[j][good[j]]
        source.append([*metrics(xy,c,tx,tc), 1e6*np.sqrt(np.mean(heldout**2))])
    for j in range(y.shape[1]):
        keep = np.arange(y.shape[1]) != j
        reduced = {**s, 'z_rcv_m': np.asarray(s['z_rcv_m'])[keep], 'T_obs_s': y[:,keep]}
        xy,c,_ = _invert(reduced)
        receiver.append(metrics(xy,c,tx,tc[keep]-tc[keep].mean()))
    for j in range(y.size):
        mask = np.ones(y.size, bool)
        mask[j] = False
        xy,c,pred = _invert(s,mask)
        ray.append([*metrics(xy,c,tx,tc), 1e6*abs(y.ravel()[j]-pred[j]) if good.ravel()[j] else 0.])
    print('worst leave-one-source-out center m / clock us / held-out inlier RMS us', np.max(source,axis=0))
    print('worst leave-one-receiver-out center m / retained-clock us', np.max(receiver,axis=0))
    print('worst leave-one-ray-out center m / clock us / held-out inlier absolute us', np.max(ray,axis=0))
    assert all(np.all(np.max(group,axis=0)[:2] <= [1.8,250]) for group in (source,receiver,ray))

nodes, weights = leggauss(64)
t = (nodes[:,None,None]+1.)/2.
rng = np.random.default_rng(625309)
for label, n in [('public-range',60),('edge-and-corner',60),('varied-geometry',60),('weak-plume',60),('clustered-slips',60)]:
    records = []
    for k in range(n):
        w,h,ns,nr = (rng.uniform(28,65),rng.uniform(55,105),int(rng.integers(10,23)),int(rng.integers(10,23))) if label=='varied-geometry' else (40.,80.,15,16)
        zs = np.linspace(.075*h,.925*h,ns)
        zr = np.linspace(.0625*h,.9375*h,nr)
        if label=='varied-geometry':
            zs += rng.uniform(-.8,.8,ns)
            zr += rng.uniform(-.8,.8,nr)
        if label=='edge-and-corner':
            x = rng.uniform(0,4) if k%2 else w-rng.uniform(0,4)
            z = rng.uniform(0,9) if k%4<2 else h-rng.uniform(0,9)
        else:
            x,z = rng.uniform(.09*w,.91*w),rng.uniform(.13*h,.87*h)
        dz = zr[None,:]-zs[:,None]
        length = np.hypot(w,dz)
        dd = (t*w-x)**2 + (zs[None,:,None]+t*dz-z)**2
        plume = length*np.einsum('q,qij->ij',weights/2,np.exp(-.5*dd/5.5**2))
        s0,eps = rng.uniform(.00030,.00055),rng.uniform(.02,.27)
        amp = rng.uniform(.000021,.000036) if label=='weak-plume' else rng.uniform(.000047,.000100)
        c = rng.uniform(-1,1,nr)
        c -= c.mean()
        c *= .004/np.max(np.abs(c))
        noise = rng.uniform(.000025,.000060)
        y = length*s0*(1-eps*(dz/length)**2)+amp*plume+c+rng.normal(0,noise,length.shape)
        kslip = round(y.size*.04)
        ids = rng.choice(y.size, kslip, replace=False)
        if label=='clustered-slips':
            j = int(rng.integers(nr))
            row = rng.choice(ns,6,replace=False)
            outside = np.flatnonzero(np.arange(y.size)%nr!=j)
            ids = np.r_[row*nr+j,rng.choice(outside,kslip-6,replace=False)]
        signs = rng.choice([-1.,1.],kslip)
        if label=='clustered-slips':
            signs[:6] = signs[0]
        y.flat[ids] += signs*rng.uniform(.0055,.006)
        xy,ch,dt = run(dict(W_m=w,H_m=h,z_src_m=zs,z_rcv_m=zr,T_obs_s=y,R_plume_m=5.5))
        records.append([*metrics(xy,ch,np.array([x,z]),c),dt])
    a = np.asarray(records)
    print(label,'worst center m / clock us / runtime s', np.max(a,axis=0),'number over target',np.sum(a[:,:2]>[1.8,250],axis=0),'worst case indices',np.argmax(a,axis=0))
PY
cd / && env -u PYTHONPATH PYTHONPATH=/holdout_task python3 -B /tmp/tomography_validate.py
```
- **Container Exit Code**: `0` (`10314 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0 center m / clock us / runtime s 0.08947190669186113 5.301980093089273 0.01141557190567255 inlier RMS us 31.909037092861322 slips 10
worst leave-one-source-out center m / clock us / held-out inlier RMS us [ 0.114618    6.85004793 44.6576269 ]
worst leave-one-receiver-out center m / retained-clock us [0.13357577 6.20950973]
worst leave-one-ray-out center m / clock us / held-out inlier absolute us [ 0.10905946  5.90244563 95.56642142]
public 1 center m / clock us / runtime s 0.1121281248088459 12.234506082460276 0.007560038939118385 inlier RMS us 31.614120381372906 slips 10
worst leave-one-source-out center m / clock us / held-out inlier RMS us [ 0.14536082 15.13024564 43.46553094]
worst leave-one-receiver-out center m / retained-clock us [ 0.13863357 13.45800074]
worst leave-one-ray-out center m / clock us / held-out inlier absolute us [ 0.15147462 13.08319223 96.80887045]
public-range worst center m / clock us / runtime s [1.70195898e+00 4.17881341e+01 1.20205330e-02] number over target [0 0] worst case indices [ 2  6 13]
edge-and-corner worst center m / clock us / runtime s [2.33123342e+01 1.43630941e+02 9.85808484e-03] number over target [13  0] worst case indices [58 28 53]
varied-geometry worst center m / clock us / runtime s [ 0.9549769  29.68236694  1.7810174 ] number over target [0 0] worst case indices [14 14 10]
weak-plume worst center m / clock us / runtime s [2.16444126e+00 2.66651345e+01 9.29946615e-03] number over target [1 0] worst case indices [46 33  5]
clustered-slips worst center m / clock us / runtime s [6.41864547e-01 1.89969476e+01 8.39244295e-03] number over target [0 0] worst case indices [54 30 21]
```

### Turn 6 — `execute_in_remote_docker` (`30441 ms` LLM)
- **Rationale**: On missed synthetic cases, compare fits started near truth and at the solver estimate under the known slip mask. A lower objective from the truth start warrants a search change; slip disagreements warrant revisiting rejection; similarly good fits far apart indicate limited localization by the available rays.
- **Alternatives Weighed**: `Immediately add a denser grid or more starting points` (rejected: A wider search helps only if the solver misses a better-fitting basin; it cannot recover location information absent from the picks.)

```bash
python3 - <<'PY'
from pathlib import Path
s = Path('/tmp/tomography_validate.py').read_text()
s = s[:s.index('cases = pickle.load')] + s[s.index('nodes, weights = leggauss(64)'):]
s = s.replace('from tomography_solver import _invert, invert_plume_and_clocks', 'from tomography_solver import _invert, invert_plume_and_clocks, _Rays\nfrom scipy.linalg import qr\nfrom scipy.optimize import least_squares')
s = s.replace('        records.append([*metrics', '''        if np.linalg.norm(np.asarray(xy)-[x,z]) > 1.8:
            # Known-slip mask is diagnostic only; it isolates location search from rejection.
            good = np.setdiff1d(np.arange(y.size),ids)
            rays = _Rays(w,zs,zr,5.5)
            Q,_ = qr(rays.design[good],mode='economic')
            def project(v):
                return v-Q@(Q.T@v)
            target = project(y.ravel()[good]*1000)
            def resid(p):
                g = project(rays.kernel(p,good))
                return target - g*max(0.,(target@g)/(g@g))
            local = least_squares(resid,[x,z],bounds=([0,0],[w,h]))
            fit = least_squares(resid,xy,bounds=([0,0],[w,h]))
            true_err = np.linalg.norm(local.x-[x,z])
            gap = (np.sum(fit.fun**2)-np.sum(local.fun**2))/(noise*1000)**2
            se = noise*1000/np.maximum(np.linalg.svd(local.jac,compute_uv=False),1e-12)
            fitted = _invert(dict(W_m=w,H_m=h,z_src_m=zs,z_rcv_m=zr,T_obs_s=y,R_plume_m=5.5))[2]
            mis = np.count_nonzero((np.abs(y.ravel()-fitted)<.0027) != np.isin(np.arange(y.size),good))
            print('MISS',label,k,'truth',np.round([x,z],2),'solver',np.round(xy,2),'oracle warm',np.round(local.x,2),'oracle solver start',np.round(fit.x,2),'warm error m',round(true_err,2),'solver minus warm SSE/noise^2',round(gap,2),'linearized SE m',np.round(se,2),'slip disagreements',mis)
        records.append([*metrics''')
Path('/tmp/tomography_diagnose.py').write_text(s)
PY
cd / && OPENBLAS_NUM_THREADS=1 PYTHONPATH=/holdout_task python3 -B /tmp/tomography_diagnose.py
```
- **Container Exit Code**: `0` (`3215 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
minus warm SSE/noise^2 -7.69 linearized SE m [0.74 4.05] slip disagreements 0
MISS edge-and-corner 32 truth [38.22  2.1 ] solver [40.    1.76] oracle warm [40.    1.76] oracle solver start [40.    1.76] warm error m 1.81 solver minus warm SSE/noise^2 0.0 linearized SE m [0.83 4.12] slip disagreements 0
MISS edge-and-corner 40 truth [39.57  1.35] solver [35.61  1.47] oracle warm [35.62  1.47] oracle solver start [35.62  1.47] warm error m 3.96 solver minus warm SSE/noise^2 -0.0 linearized SE m [ 1.11 11.05] slip disagreements 0
MISS edge-and-corner 42 truth [36.66 76.64] solver [38.3  78.23] oracle warm [38.31 78.23] oracle solver start [38.31 78.23] warm error m 2.29 solver minus warm SSE/noise^2 -0.0 linearized SE m [0.88 5.78] slip disagreements 0
MISS edge-and-corner 49 truth [3.01 2.2 ] solver [5.74 2.57] oracle warm [5.74 2.57] oracle solver start [5.74 2.57] warm error m 2.76 solver minus warm SSE/noise^2 -0.0 linearized SE m [0.42 0.82] slip disagreements 0
MISS edge-and-corner 54 truth [38.6  78.37] solver [40.  79.8] oracle warm [40.  79.8] oracle solver start [40.  79.8] warm error m 2.01 solver minus warm SSE/noise^2 -0.0 linearized SE m [1.14 6.05] slip disagreements 0
MISS edge-and-corner 56 truth [38.41  1.68] solver [36.46  0.  ] oracle warm [36.45  0.  ] oracle solver start [36.46  0.  ] warm error m 2.58 solver minus warm SSE/noise^2 -0.0 linearized SE m [ 1.54 19.31] slip disagreements 0
MISS edge-and-corner 58 truth [39.91 78.84] solver [17.02 74.41] oracle warm [40.   76.63] oracle solver start [17.02 74.41] warm error m 2.22 solver minus warm SSE/noise^2 -1.17 linearized SE m [0.86 4.31] slip disagreements 0
edge-and-corner worst center m / clock us / runtime s [2.33123342e+01 1.43630941e+02 1.26307569e-02] number over target [13  0] worst case indices [58 28 42]
varied-geometry worst center m / clock us / runtime s [9.54976905e-01 2.96823669e+01 1.61295361e-02] number over target [0 0] worst case indices [14 14 10]
MISS weak-plume 46 truth [20.22 67.13] solver [22.31 67.69] oracle warm [22.31 67.69] oracle solver start [22.31 67.69] warm error m 2.16 solver minus warm SSE/noise^2 -0.0 linearized SE m [0.38 1.11] slip disagreements 0
weak-plume worst center m / clock us / runtime s [2.16444126e+00 2.66651345e+01 9.78767406e-03] number over target [1 0] worst case indices [46 33 50]
clustered-slips worst center m / clock us / runtime s [6.41864547e-01 1.89969476e+01 8.23769206e-03] number over target [0 0] worst case indices [54 30  7]
```

### Turn 7 — `execute_in_remote_docker` (`39506 ms` LLM)
- **Rationale**: Compare an isolated challenger against the unchanged baseline on real public picks and identical synthetic draws. Promote it only if it reduces corner error without sacrificing weak-plume recovery or clock accuracy.
- **Alternatives Weighed**: `Increase grid density alone` (rejected: It cannot correct cases in which the distant solution already fits the measured times better.); `Hard-bound plume amplitude to the public values` (rejected: The task only specifies positive amplitude, so a hard bound would exclude valid weaker or stronger plumes.)
- **ATMS NOGOOD DECLARED**: Refuted `More search alone can resolve the synthetic corner failures once cycle skips are removed.` | Witness: `All 13 corner misses had the correct slip mask; in case 2 the truth-start fit still moved 9.73 m, and in cases 28 and 58 the distant fit had lower residual loss than the truth-start fit.` | Blanket Diagnosis: `Near a survey corner, few rays cross the plume core, allowing changes in center and amplitude to generate nearly indistinguishable pick times.` | Sublated Pivot: `Retain the exact finite-ray model and compare weak, noise-scaled amplitude regularization with the unrestricted fit.`

```bash
set -e
mkdir -p /tmp/tomo_compare
cp /holdout_task/tomography_solver.py /tmp/tomo_compare/tomography_solver.py
python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/tomo_compare/tomography_solver.py')
s=p.read_text().replace('import numpy as np','import os\nimport numpy as np')
s=s.replace('    xy = None\n    for _ in range(5):','    xy = None\n    alpha = 0.\n    amp_ref = .065\n    amp_scale = float(os.environ["TOMO_AMP_SCALE"])\n    for iteration in range(7):')
s=s.replace('score = np.maximum(dot, 0.)**2 / np.maximum(power, 1.e-20)','score = np.maximum(dot + alpha*amp_ref, 0.)**2 / np.maximum(power + alpha, 1.e-20)')
s=s.replace('denom = max(np.dot(k, k), 1.e-20)\n            amp = max(0., np.dot(k, yy) / denom)','denom = max(np.dot(k, k) + alpha, 1.e-20)\n            amp = max(0., (np.dot(k, yy) + alpha*amp_ref) / denom)')
s=s.replace('return amp*k - yy','return np.r_[amp*k - yy, np.sqrt(alpha)*(amp-amp_ref)]').replace('return np.zeros_like(dk)','return np.zeros((len(k)+1, 2))').replace('((yy - 2*amp*k) @ dk) / denom','((yy - 2*amp*k) @ dk) / denom').replace('return amp*dk + k[:, None]*da','return np.vstack((amp*dk + k[:, None]*da, np.sqrt(alpha)*da))')
s=s.replace('np.dot(k, yy) / max(np.dot(k, k), 1.e-20)','(np.dot(k, yy) + alpha*amp_ref) / max(np.dot(k, k) + alpha, 1.e-20)')
s=s.replace('        if np.array_equal(updated, good):\n            break','        if np.array_equal(updated, good):\n            if alpha > 0. or not np.isfinite(amp_scale):\n                break\n            e = (y-prediction)[good]\n            sig = max(1.e-9, 1.4826*np.median(np.abs(e - np.median(e))))\n            alpha = (sig/amp_scale)**2\n            xy = None')
p.write_text(s)
q=Path('/tmp/tomography_validate.py').read_text()
a=q[q.index('cases = pickle.load'):q.index('nodes, weights')]
a=a[:a.index('    source, receiver')]
Path('/tmp/tomo_compare/evaluate.py').write_text(q[:q.index('cases = pickle.load')]+a+q[q.index('nodes, weights'):])
PY
cd /
for scale in inf .022 .035 .050; do
  echo "amplitude prior width (ms/m): $scale"
  OPENBLAS_NUM_THREADS=1 TOMO_AMP_SCALE=$scale PYTHONPATH=/tmp/tomo_compare python3 -B /tmp/tomo_compare/evaluate.py
done
```
- **Container Exit Code**: `0` (`17134 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0] worst case indices [14 26 10]
weak-plume worst center m / clock us / runtime s [2.26722577e+00 3.05684980e+01 1.40803871e-02] number over target [1 0] worst case indices [46 33 29]
clustered-slips worst center m / clock us / runtime s [ 0.65920923 19.16632893  0.01987608] number over target [0 0] worst case indices [54 27 39]
amplitude prior width (ms/m): .035
public 0 center m / clock us / runtime s 0.08947133056628774 5.3019859927747435 0.01556467404589057 inlier RMS us 31.909037093011833 slips 10
public 1 center m / clock us / runtime s 0.11222187057949515 12.24016799820959 0.0142300168517977 inlier RMS us 31.61412103016869 slips 10
public-range worst center m / clock us / runtime s [1.72276318e+00 3.52913356e+01 1.52765668e-02] number over target [0 0] worst case indices [ 2  6 33]
edge-and-corner worst center m / clock us / runtime s [2.05019234e+01 1.43470091e+02 1.96398699e-02] number over target [10  0] worst case indices [28 28 53]
varied-geometry worst center m / clock us / runtime s [ 0.92719687 29.3593974   0.02995533] number over target [0 0] worst case indices [14 26 10]
weak-plume worst center m / clock us / runtime s [2.20491154e+00 2.81869961e+01 1.53124370e-02] number over target [1 0] worst case indices [46 33  5]
clustered-slips worst center m / clock us / runtime s [6.48810338e-01 1.88980498e+01 1.41113589e-02] number over target [0 0] worst case indices [54 30 26]
amplitude prior width (ms/m): .050
public 0 center m / clock us / runtime s 0.08947162435592967 5.301982984251222 0.014728954061865807 inlier RMS us 31.9090370928943 slips 10
public 1 center m / clock us / runtime s 0.1121740670301352 12.237280620628795 0.013441192917525768 inlier RMS us 31.614120536857858 slips 10
public-range worst center m / clock us / runtime s [1.71231740e+00 3.84190847e+01 1.58987590e-02] number over target [0 0] worst case indices [2 6 6]
edge-and-corner worst center m / clock us / runtime s [2.05532517e+01 1.43552087e+02 2.00478411e-02] number over target [11  0] worst case indices [28 28 53]
varied-geometry worst center m / clock us / runtime s [9.41220057e-01 2.93405145e+01 2.86925910e-02] number over target [0 0] worst case indices [14 14 10]
weak-plume worst center m / clock us / runtime s [2.18424940e+00 2.74070714e+01 1.65274760e-02] number over target [1 0] worst case indices [46 33 57]
clustered-slips worst center m / clock us / runtime s [6.45283463e-01 1.89483131e+01 1.72360991e-02] number over target [0 0] worst case indices [54 30 23]
```

### Turn 8 — `execute_in_remote_docker` (`48245 ms` LLM)
- **Rationale**: The point solution can select one end of a broad likelihood ridge. Test whether averaging locations over that ridge reduces the worst center error without degrading well-resolved public cases or clock RMSE. Keep this comparison outside the deliverable until the results support a change.
- **Alternatives Weighed**: `Promote amplitude ridge immediately` (rejected: It reduced corner misses but left a large worst-case error and slightly worsened the weak-plume slice.)

```bash
set -e
mkdir -p /tmp/tomo_marginal
cp /tmp/tomo_compare/evaluate.py /tmp/tomo_marginal/evaluate.py
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/tomography_solver.py')
s = p.read_text().replace('from scipy.special import erf','from scipy.special import erf, log_ndtr')
needle = '    clock = _clip_clock(np.r_[coef[2:], -np.sum(coef[2:])])'
trial = '''    # Integrate over positive amplitude when the center likelihood is broad.
    e = (y - prediction)[ind]
    sig = max(1.e-9, 1.4826*np.median(np.abs(e-np.median(e))))
    sing = np.linalg.svd(profile(xy, True), compute_uv=False)
    if sig / max(sing[-1], 1.e-20) > .75:
        dx = rays.r / 10.
        xx = np.linspace(0., w, int(np.ceil(w/dx))+1)
        zz = np.linspace(0., h, int(np.ceil(h/dx))+1)
        gx, gz = np.meshgrid(xx, zz)
        centers = np.column_stack((gx.ravel(), gz.ravel()))
        alpha = (sig/.035)**2
        prior_mean = .065
        log_mass = []
        for off in range(0, len(centers), 256):
            raw = rays.kernel(centers[off:off+256], ind)
            q = np.einsum('ij,jk->ik', raw, Q)
            power = np.maximum(np.sum(raw*raw, axis=1)-np.sum(q*q, axis=1), 0.) + alpha
            dot = np.einsum('ij,j->i', raw, yy) + alpha*prior_mean
            t = dot/(sig*np.sqrt(power))
            log_mass.append(.5*t*t-.5*np.log(power)+log_ndtr(t))
        log_mass = np.concatenate(log_mass)
        mass = np.exp(log_mass-np.max(log_mass)).reshape(len(zz),len(xx))
        mass[[0,-1],:] *= .5
        mass[:,[0,-1]] *= .5
        xy = np.einsum('i,ij->j', mass.ravel(), centers)/mass.sum()
        g = rays.kernel(xy, ind)
        k = project(g)
        amplitude = max(0., np.dot(k, yy)/max(np.dot(k, k), 1.e-20))
        coef = solve_triangular(U, coeff_y-amplitude*(Q.T@g))
        prediction = D@coef + amplitude*rays.kernel(xy, slice(None))
'''
assert s.count(needle)==1
Path('/tmp/tomo_marginal/tomography_solver.py').write_text(s.replace(needle,trial+needle))
PY
cd /
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/tomo_marginal python3 -B /tmp/tomo_marginal/evaluate.py
```
- **Container Exit Code**: `0` (`9349 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0 center m / clock us / runtime s 0.08947190669186113 5.301980093089273 0.009418026078492403 inlier RMS us 31.909037092861322 slips 10
public 1 center m / clock us / runtime s 0.1121281248088459 12.234506082460276 0.007660878822207451 inlier RMS us 31.614120381372906 slips 10
public-range worst center m / clock us / runtime s [ 1.82726036 41.78813411  0.11456647] number over target [1 0] worst case indices [2 6 2]
edge-and-corner worst center m / clock us / runtime s [1.94967508e+01 1.42973012e+02 1.25925877e-01] number over target [15  0] worst case indices [28 28 50]
varied-geometry worst center m / clock us / runtime s [ 0.84348406 28.83063711  0.06409199] number over target [0 0] worst case indices [14 32 26]
weak-plume worst center m / clock us / runtime s [ 2.24620774 29.01748682  0.1396276 ] number over target [1 0] worst case indices [46 33 52]
clustered-slips worst center m / clock us / runtime s [6.41864547e-01 1.89969476e+01 1.26483040e-02] number over target [0 0] worst case indices [54 30 12]
```

### Turn 9 — `execute_in_remote_docker` (`31921 ms` LLM)
- **Rationale**: The model specifies no upper bound on positive plume amplitude. Stress coherent delays comparable to cycle slips while keeping the center in a well-covered depth range. If errors coincide with rejected inliers, revise slip screening to use the full forward model; otherwise retain the simpler screen.
- **Alternatives Weighed**: `Promote amplitude marginalization` (rejected: It increased corner misses and pushed a previously passing public-range synthetic case over tolerance.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/tomo_compare/evaluate.py')
s = p.read_text()
s = s[:s.index('cases = pickle.load')] + s[s.index('nodes, weights'):]
s = s.replace("[('public-range',60),('edge-and-corner',60),('varied-geometry',60),('weak-plume',60),('clustered-slips',60)]", "[('strong-plume',80),('strong-near-wall',80),('strong-wide-well',80)]")
s = s.replace("label=='varied-geometry'", "label=='strong-wide-well'")
s = s.replace("if label=='edge-and-corner':", "if label=='strong-near-wall':")
s = s.replace("z = rng.uniform(0,9) if k%4<2 else h-rng.uniform(0,9)", "z = rng.uniform(.18*h,.82*h)")
s = s.replace("amp = rng.uniform(.000021,.000036) if label=='weak-plume' else rng.uniform(.000047,.000100)", "amp = rng.uniform(.00010,.00060)")
s = s.replace('        records.append([*metrics', '''        if np.linalg.norm(np.asarray(xy)-[x,z]) > 1.8:
            _, _, pred = _invert(dict(W_m=w,H_m=h,z_src_m=zs,z_rcv_m=zr,T_obs_s=y,R_plume_m=5.5))
            flagged = np.abs(y.ravel()-pred) >= .0027
            actual = np.zeros(y.size,bool)
            actual[ids] = True
            if len(misses) < 5:
                print('miss',label,k,'center error m', round(float(np.linalg.norm(np.asarray(xy)-[x,z])),2), 'peak delay ms',round(1000*amp*plume.max(),2),'inliers wrongly removed',np.sum(flagged & ~actual),'slips retained',np.sum(actual & ~flagged))
            misses.append(k)
        records.append([*metrics''')
s = s.replace('    records = []','    records, misses = [], []')
Path('/tmp/tomography_strong.py').write_text(s)
PY
cd / && OPENBLAS_NUM_THREADS=1 PYTHONPATH=/holdout_task python3 -B /tmp/tomography_strong.py
```
- **Container Exit Code**: `0` (`2730 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
strong-plume worst center m / clock us / runtime s [6.23294463e-01 4.38113850e+01 1.29149088e-02] number over target [0 0] worst case indices [ 2 78  9]
strong-near-wall worst center m / clock us / runtime s [4.27825719e-01 5.12995770e+01 9.74702183e-03] number over target [0 0] worst case indices [37 58 25]
strong-wide-well worst center m / clock us / runtime s [4.64305781e-01 2.82676994e+01 2.17961660e-02] number over target [0 0] worst case indices [66 44 16]
```

### Turn 10 — `execute_in_remote_docker` (`24876 ms` LLM)
- **Rationale**: Keep the total slip rate at 4% while varying how many slips share one receiver and whether they share a sign. If clock errors rise when same-sign slips form a local majority, use the zero-sum gauge and clock bounds to distinguish the competing receiver offsets rather than tightening the ray-level threshold.
- **Alternatives Weighed**: `Change the background-only screen for stronger plumes` (rejected: The strong-plume, near-wall, and wider-well slices all met both targets with the current screen.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
s = Path('/tmp/tomography_strong.py').read_text()
s = s.replace("[('strong-plume',80),('strong-near-wall',80),('strong-wide-well',80)]", "[('clustered-slips',80),('majority-same-sign',80),('majority-mixed-sign',80)]")
s = s.replace('amp = rng.uniform(.00010,.00060)', 'amp = rng.uniform(.000047,.000100)')
s = s.replace("if label=='clustered-slips':\n            j", "if label in ('clustered-slips', 'majority-same-sign', 'majority-mixed-sign'):\n            count = 7 if label == 'clustered-slips' else 9\n            j")
s = s.replace('rng.choice(ns,6,replace=False)', 'rng.choice(ns,count,replace=False)').replace('kslip-6,replace=False', 'kslip-count,replace=False')
s = s.replace("if label=='clustered-slips':\n            signs[:6] = signs[0]", "if label != 'majority-mixed-sign':\n            signs[:count] = signs[0]")
s = s.replace('if np.linalg.norm(np.asarray(xy)-[x,z]) > 1.8:', 'if max(np.asarray(metrics(xy,ch,np.array([x,z]),c)) / [1.8,250]) > 1:')
Path('/tmp/tomography_cluster.py').write_text(s)
PY
cd / && OPENBLAS_NUM_THREADS=1 PYTHONPATH=/holdout_task python3 -B /tmp/tomography_cluster.py
```
- **Container Exit Code**: `0` (`3283 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
clustered-slips worst center m / clock us / runtime s [8.60002431e-01 4.29950149e+01 9.62589984e-03] number over target [0 0] worst case indices [79 24  0]
miss majority-same-sign 0 center error m 0.1 peak delay ms 0.82 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 1 center error m 0.27 peak delay ms 1.1 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 2 center error m 0.09 peak delay ms 1.13 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 3 center error m 0.05 peak delay ms 0.9 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 4 center error m 0.14 peak delay ms 1.03 inliers wrongly removed 6 slips retained 9
majority-same-sign worst center m / clock us / runtime s [1.36862307e+00 1.45370993e+03 9.49919992e-03] number over target [ 0 74] worst case indices [23 14 30]
miss majority-mixed-sign 6 center error m 0.14 peak delay ms 0.85 inliers wrongly removed 6 slips retained 8
miss majority-mixed-sign 7 center error m 0.08 peak delay ms 0.98 inliers wrongly removed 6 slips retained 8
miss majority-mixed-sign 39 center error m 0.35 peak delay ms 0.76 inliers wrongly removed 6 slips retained 8
miss majority-mixed-sign 49 center error m 0.17 peak delay ms 1.31 inliers wrongly removed 6 slips retained 8
miss majority-mixed-sign 68 center error m 0.15 peak delay ms 1.13 inliers wrongly removed 6 slips retained 8
majority-mixed-sign worst center m / clock us / runtime s [4.98221798e-01 1.34873338e+03 9.61568602e-03] number over target [0 6] worst case indices [37  6 69]
```

### Turn 11 — `execute_in_remote_docker` (`36532 ms` LLM)
- **Rationale**: Refit each plausible alternative clock assignment and score both against the same picks. The clustered-slip slice tests whether this fixes the clock ambiguity; the public, ordinary synthetic, and strong-plume slices test whether it mistakes coherent plume delays for slips.
- **Alternatives Weighed**: `Tighten the inlier threshold` (rejected: A narrower threshold still selects the wrong cluster when the current receiver offset is centered on the slipped picks.)
- **ATMS NOGOOD DECLARED**: Refuted `A globally sparse slip rate is sufficient for per-receiver robust fitting to identify its true clock mode.` | Witness: `At 4% overall corruption, the baseline failed the clock target in 74 of 80 same-sign majority cases and 6 of 80 mixed-sign cases; all center estimates still passed.` | Blanket Diagnosis: `When slips dominate one receiver, its fitted offset follows the shifted cluster, and clipping the output afterward cannot recover the correct assignment.` | Sublated Pivot: `Keep the physical plume fit and compare both assignments of a visibly split receiver against the joint, zero-sum traveltime fit.`

```bash
set -e
mkdir -p /tmp/tomo_clock
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/tomography_solver.py')
s = p.read_text().replace('def _invert(survey, include=None):','def _fit(survey, include=None, initial_good=None):')
a = s.index('    Da, ya =')
b = s.index('\n    step =', a)
s = s[:a] + '    if initial_good is None:\n' + ''.join('    '+line if line.strip() else line for line in s[a:b].splitlines(keepends=True)) + '\n    else:\n        good = available & initial_good\n' + s[b:]
a = s.index('\ndef invert_plume_and_clocks')
s = s[:a] + '''
def _invert(survey, include=None):
    fit = _fit(survey, include)
    obs = np.asarray(survey['T_obs_s'], dtype=float)
    available = np.isfinite(obs) if include is None else np.isfinite(obs) & include.reshape(obs.shape)

    def cost(result):
        e = 1000. * np.abs((obs-result[2].reshape(obs.shape))[available])
        # Score a shifted pick as a slip, but charge for the rare slip mode.
        # The shift spans the public and nominal cycle lengths.
        cycle = e - np.clip(e, 5.3, 6.2)
        return np.minimum(np.minimum((e/.11)**2, 32.),
                          8. + (cycle/.20)**2).sum()

    for _ in range(3):
        residual = 1000. * (obs-fit[2].reshape(obs.shape))
        good = available & (np.abs(residual) < 2.7)
        best, best_cost = fit, None
        for j in range(obs.shape[1]):
            for sign in (-1., 1.):
                shifted = available[:, j] & (np.abs(residual[:, j]-sign*5.75) < 1.)
                if shifted.sum() < 3 or good[:, j].sum() < 3:
                    continue
                alt = good.copy()
                alt[:, j] = shifted
                trial = _fit(survey, include, alt.ravel())
                if best_cost is None:
                    best_cost = cost(best)
                new_cost = cost(trial)
                if new_cost < best_cost:
                    best, best_cost = trial, new_cost
        if best is fit:
            break
        fit = best
    return fit

''' + s[a:]
Path('/tmp/tomo_clock/tomography_solver.py').write_text(s)
PY
cd /
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/tomo_clock
python3 -B /tmp/tomography_cluster.py
python3 -B /tmp/tomo_compare/evaluate.py
python3 -B /tmp/tomography_strong.py
```
- **Container Exit Code**: `1` (`6282 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
clustered-slips worst center m / clock us / runtime s [8.60002431e-01 4.29950149e+01 1.88051511e-02] number over target [0 0] worst case indices [79 24 38]
miss majority-same-sign 0 center error m 0.1 peak delay ms 0.82 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 1 center error m 0.27 peak delay ms 1.1 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 2 center error m 0.09 peak delay ms 1.13 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 3 center error m 0.05 peak delay ms 0.9 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 4 center error m 0.14 peak delay ms 1.03 inliers wrongly removed 6 slips retained 9
majority-same-sign worst center m / clock us / runtime s [1.36862307e+00 1.45370993e+03 1.87578320e-02] number over target [ 0 74] worst case indices [23 14 33]
majority-mixed-sign worst center m / clock us / runtime s [ 0.4982218  26.32757945  0.02927484] number over target [0 0] worst case indices [37 27 26]
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/tomo_compare/evaluate.py", line 28, in <module>
    xy,c,dt = run(s)
              ^^^^^^
  File "/tmp/tomo_compare/evaluate.py", line 14, in run
    out = invert_plume_and_clocks(s)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/tomo_compare/tomography_solver.py", line 141, in invert_plume_and_clocks
    xy, clock, _ = _invert(survey)
                   ^^^^^^^^^^^^^^^
  File "/tmp/tomo_compare/tomography_solver.py", line 78, in _invert
    amp_scale = float(os.environ["TOMO_AMP_SCALE"])
                      ~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "<frozen os>", line 685, in __getitem__
KeyError: 'TOMO_AMP_SCALE'
```

### Turn 12 — `execute_in_remote_docker` (`29701 ms` LLM)
- **Rationale**: Compare both clock modes using residuals after accounting for a survey-level cycle length, with a penalty for violating the stated clock bounds. Rerun clustered corruption and the real-data jackknife to see whether the new score fixes the clock ambiguity without changing well-resolved fits.
- **Alternatives Weighed**: `Choose the fit with fewer inferred slips` (rejected: When a receiver has a same-sign slip majority, this favors relabeling the slipped cluster as clean and assigning it the receiver clock.)

```bash
set -e
mkdir -p /tmp/tomo_gauge
python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/tomo_clock/tomography_solver.py')
s = p.read_text()
a = s.index('        e = 1000. * np.abs'); b = s.index('\n\n    for _ in range(3)', a)
s = s[:a] + '''        e = 1000. * np.abs((obs-result[2].reshape(obs.shape))[available])
        slip = (e > 3.8) & (e < 7.5)
        cycle = np.median(e[slip]) if np.any(slip) else 5.75
        # Score timing consistency before counting slips. A wrong clock mode
        # changes the background needed to satisfy the zero-sum gauge.
        error = np.minimum(e, np.abs(e-cycle))
        excess = 1000. * (result[2] - result[3])
        return np.minimum(error**2, .6**2).sum() + .001*slip.sum() + 8.*(excess@excess)
''' + s[b:]
s = s.replace('    return xy, clock/1000., prediction/1000.', '    return xy, clock/1000., prediction/1000., np.r_[coef[2:], -np.sum(coef[2:])]/1000.')
s = s.replace('result[2] - result[3]', 'result[1] - result[3]').replace('    return fit\n', '    return fit[:3]\n')
Path('/tmp/tomo_gauge/tomography_solver.py').write_text(s)
PY
cd / && export OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/tomo_gauge
python3 -B /tmp/tomography_cluster.py
python3 -B /tmp/tomography_validate.py
```
- **Container Exit Code**: `0` (`13910 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
iers wrongly removed 8 slips retained 7
clustered-slips worst center m / clock us / runtime s [8.60002431e-01 1.42048470e+03 2.39024679e-02] number over target [0 2] worst case indices [79 18 62]
miss majority-same-sign 20 center error m 0.14 peak delay ms 0.59 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 29 center error m 0.06 peak delay ms 1.34 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 66 center error m 0.14 peak delay ms 1.23 inliers wrongly removed 6 slips retained 9
majority-same-sign worst center m / clock us / runtime s [1.26386157e+00 1.40995965e+03 3.44193110e-02] number over target [0 3] worst case indices [23 20 68]
majority-mixed-sign worst center m / clock us / runtime s [4.98221798e-01 2.63275795e+01 2.53019899e-02] number over target [0 0] worst case indices [37 27 10]
public 0 center m / clock us / runtime s 0.08947190669186113 5.301980093089273 0.009193829959258437 inlier RMS us 31.909037092861322 slips 10
worst leave-one-source-out center m / clock us / held-out inlier RMS us [ 0.114618    6.85004793 44.6576269 ]
worst leave-one-receiver-out center m / retained-clock us [0.13357577 6.20950973]
worst leave-one-ray-out center m / clock us / held-out inlier absolute us [ 0.10905946  5.90244563 95.56642142]
public 1 center m / clock us / runtime s 0.1121281248088459 12.234506082460276 0.00869798706844449 inlier RMS us 31.614120381372906 slips 10
worst leave-one-source-out center m / clock us / held-out inlier RMS us [ 0.14536082 15.13024564 43.46553094]
worst leave-one-receiver-out center m / retained-clock us [ 0.13863357 13.45800074]
worst leave-one-ray-out center m / clock us / held-out inlier absolute us [ 0.15147462 13.08319223 96.80887045]
public-range worst center m / clock us / runtime s [1.70195898e+00 4.17881341e+01 1.57560110e-02] number over target [0 0] worst case indices [ 2  6 30]
edge-and-corner worst center m / clock us / runtime s [2.33123342e+01 1.43630941e+02 1.80763770e-02] number over target [13  0] worst case indices [58 28 19]
varied-geometry worst center m / clock us / runtime s [9.54976905e-01 2.96823669e+01 2.63621588e-02] number over target [0 0] worst case indices [14 14 18]
weak-plume worst center m / clock us / runtime s [2.16444126e+00 2.66651345e+01 1.81805161e-02] number over target [1 0] worst case indices [46 33 59]
clustered-slips worst center m / clock us / runtime s [ 0.64186455 18.99694762  0.03001834] number over target [0 0] worst case indices [54 30  0]
```

### Turn 13 — `execute_in_remote_docker` (`31121 ms` LLM)
- **Rationale**: A median cycle length computed from a biased clock fit can absorb part of the clock error and favor the wrong mode. Test whether fitting the common shift with the background and clocks resolves the remaining ambiguity, while retaining the clean-pick estimator for the returned result. Compare both clock slices and rerun public jackknives to catch regressions.
- **Alternatives Weighed**: `Promote the previous gauge-aware score unchanged` (rejected: It fixed most majority-slip errors but incorrectly flipped two receivers that the baseline got right.)

```bash
set -e
mkdir -p /tmp/tomo_cycle
python3 - <<'PY'
from pathlib import Path
s = Path('/tmp/tomo_gauge/tomography_solver.py').read_text()
a = s.index('    def cost(result):')
b = s.index('\n    for _ in range(3):', a)
s = s[:a] + '''    rays = _Rays(float(survey['W_m']), np.asarray(survey['z_src_m']),
                 np.asarray(survey['z_rcv_m']), float(survey.get('R_plume_m', 5.5)))
    ind = np.flatnonzero(available.ravel())
    target = 1000. * obs.ravel()[ind]

    def cost(result):
        e = target - 1000. * result[2][ind]
        # The common cycle length is estimated with the background and clock
        # offsets, rather than estimated from their already-biased residuals.
        mode = np.where((np.abs(e) > 3.8) & (np.abs(e) < 7.5), np.sign(e), 0.)
        keep = (np.abs(e) < 1.8) | (mode != 0.)
        design = np.column_stack((rays.design[ind], rays.kernel(result[0], ind), mode))
        beta = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
        clock = np.r_[beta[2:-2], -beta[2:-2].sum()]
        excess = clock - _clip_clock(clock)
        if not np.any(mode):
            cycle = 5.75
        else:
            cycle = np.clip(beta[-1], 5.3, 6.2)
        error = design[:, :-1] @ beta[:-1] + cycle*mode - target
        return np.minimum(error**2, .6**2).sum() + .0004*np.count_nonzero(mode) + 8.*(excess@excess)

''' + s[b:]
Path('/tmp/tomo_cycle/tomography_solver.py').write_text(s)
PY
cd /
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/tomo_cycle
python3 -B /tmp/tomography_cluster.py
python3 -B /tmp/tomography_strong.py
python3 -B /tmp/tomography_validate.py
```
- **Container Exit Code**: `0` (`16084 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
lay ms 1.23 inliers wrongly removed 6 slips retained 9
majority-same-sign worst center m / clock us / runtime s [1.26386157e+00 1.40995965e+03 2.73431439e-02] number over target [0 3] worst case indices [23 20 63]
majority-mixed-sign worst center m / clock us / runtime s [4.98221798e-01 2.63275795e+01 2.45085710e-02] number over target [0 0] worst case indices [37 27 20]
strong-plume worst center m / clock us / runtime s [6.23294463e-01 4.38113850e+01 1.82948650e-02] number over target [0 0] worst case indices [ 2 78 69]
strong-near-wall worst center m / clock us / runtime s [4.27825719e-01 5.12995770e+01 1.64490379e-02] number over target [0 0] worst case indices [37 58 33]
strong-wide-well worst center m / clock us / runtime s [ 0.46430578 28.26769941  0.03676221] number over target [0 0] worst case indices [66 44  0]
public 0 center m / clock us / runtime s 0.08947190669186113 5.301980093089273 0.008986141998320818 inlier RMS us 31.909037092861322 slips 10
worst leave-one-source-out center m / clock us / held-out inlier RMS us [ 0.114618    6.85004793 44.6576269 ]
worst leave-one-receiver-out center m / retained-clock us [0.13357577 6.20950973]
worst leave-one-ray-out center m / clock us / held-out inlier absolute us [ 0.10905946  5.90244563 95.56642142]
public 1 center m / clock us / runtime s 0.1121281248088459 12.234506082460276 0.008006210904568434 inlier RMS us 31.614120381372906 slips 10
worst leave-one-source-out center m / clock us / held-out inlier RMS us [ 0.14536082 15.13024564 43.46553094]
worst leave-one-receiver-out center m / retained-clock us [ 0.13863357 13.45800074]
worst leave-one-ray-out center m / clock us / held-out inlier absolute us [ 0.15147462 13.08319223 96.80887045]
public-range worst center m / clock us / runtime s [1.70195898e+00 4.17881341e+01 1.60404160e-02] number over target [0 0] worst case indices [ 2  6 54]
edge-and-corner worst center m / clock us / runtime s [2.33123342e+01 1.43630941e+02 1.97614469e-02] number over target [13  0] worst case indices [58 28 53]
varied-geometry worst center m / clock us / runtime s [9.54976905e-01 2.96823669e+01 2.58861620e-02] number over target [0 0] worst case indices [14 14 18]
weak-plume worst center m / clock us / runtime s [2.16444126e+00 2.66651345e+01 1.43843731e-02] number over target [1 0] worst case indices [46 33  8]
clustered-slips worst center m / clock us / runtime s [6.41864547e-01 1.38727457e+03 2.26308249e-02] number over target [0 1] worst case indices [54 38 38]
```

### Turn 14 — `execute_in_remote_docker` (`26566 ms` LLM)
- **Rationale**: Compare soft slip-rate penalties with timing-only scoring on both the existing clustered cases and a fresh seed with varying corruption rates. Reject penalties that improve only the fixed-rate slice.
- **Alternatives Weighed**: `Require exactly 4% slips` (rejected: The specification gives an approximate rate, and withholding rays changes the observed slip fraction.); `Always choose the mode with fewer slips` (rejected: A local majority can make the wrong receiver offset appear to explain more clean observations.)

```bash
set -e
mkdir -p /tmp/tomo_rate
python3 - <<'PY'
from pathlib import Path
s = Path('/tmp/tomo_cycle/tomography_solver.py').read_text()
old = 'return np.minimum(error**2, .6**2).sum() + .0004*np.count_nonzero(mode) + 8.*(excess@excess)'
assert s.count(old) == 1
for k, weight in enumerate((0., .008, .025, .08)):
    path = Path('/tmp/tomo_rate') / str(k)
    path.mkdir(exist_ok=True)
    penalty = f'{weight!r} * (np.count_nonzero(mode) - round(.04*len(target)))**2'
    (path/'tomography_solver.py').write_text(s.replace(old, f'return np.minimum(error**2, .6**2).sum() + {penalty} + 8.*(excess@excess)'))
s = Path('/tmp/tomography_cluster.py').read_text()
s = s.replace('kslip = round(y.size*.04)', 'kslip = max(9, round(y.size*rng.uniform(.027,.057)))')
s = s.replace("rng = np.random.default_rng(625309)","rng = np.random.default_rng(194327)")
Path('/tmp/tomography_variable_rate.py').write_text(s)
PY
cd /
export OPENBLAS_NUM_THREADS=1
for k in 0 1 2 3; do
  echo "slip-count penalty candidate $k; fixed-rate stress"
  PYTHONPATH=/tmp/tomo_rate/$k python3 -B /tmp/tomography_cluster.py | grep 'number over target'
  echo "slip-count penalty candidate $k; variable-rate stress"
  PYTHONPATH=/tmp/tomo_rate/$k python3 -B /tmp/tomography_variable_rate.py | grep 'number over target'
done
```
- **Container Exit Code**: `0` (`42861 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
24e+00 1.44883082e+03 2.73492830e-02] number over target [0 2] worst case indices [30 14 47]
majority-same-sign worst center m / clock us / runtime s [6.85976477e-01 1.45219353e+03 2.67480582e-02] number over target [0 5] worst case indices [47 25 54]
majority-mixed-sign worst center m / clock us / runtime s [1.77021198e+00 6.29298244e+01 3.32085569e-02] number over target [0 0] worst case indices [25 25 36]
slip-count penalty candidate 2; fixed-rate stress
clustered-slips worst center m / clock us / runtime s [8.60002431e-01 4.29950149e+01 1.97922429e-02] number over target [0 0] worst case indices [79 24 78]
majority-same-sign worst center m / clock us / runtime s [ 1.26386157 24.41285334  0.028776  ] number over target [0 0] worst case indices [23 23 12]
majority-mixed-sign worst center m / clock us / runtime s [4.98221798e-01 2.63275795e+01 2.41667561e-02] number over target [0 0] worst case indices [37 27  1]
slip-count penalty candidate 2; variable-rate stress
clustered-slips worst center m / clock us / runtime s [1.33988724e+00 1.44883082e+03 2.34307379e-02] number over target [0 4] worst case indices [30 14 47]
majority-same-sign worst center m / clock us / runtime s [6.85976477e-01 1.45219353e+03 2.55227250e-02] number over target [0 6] worst case indices [47 25 35]
majority-mixed-sign worst center m / clock us / runtime s [1.77021198e+00 6.29298244e+01 2.84383451e-02] number over target [0 0] worst case indices [25 25 72]
slip-count penalty candidate 3; fixed-rate stress
clustered-slips worst center m / clock us / runtime s [8.60002431e-01 4.29950149e+01 1.81029178e-02] number over target [0 0] worst case indices [79 24 73]
majority-same-sign worst center m / clock us / runtime s [ 1.26386157 24.41285334  0.03192698] number over target [0 0] worst case indices [23 23 18]
majority-mixed-sign worst center m / clock us / runtime s [4.98221798e-01 2.63275795e+01 2.45433049e-02] number over target [0 0] worst case indices [37 27 78]
slip-count penalty candidate 3; variable-rate stress
clustered-slips worst center m / clock us / runtime s [1.33988724e+00 1.44883082e+03 2.37321029e-02] number over target [0 6] worst case indices [30 14 55]
majority-same-sign worst center m / clock us / runtime s [6.85976477e-01 1.45219353e+03 2.53036532e-02] number over target [0 7] worst case indices [47 25 15]
majority-mixed-sign worst center m / clock us / runtime s [1.55273614e+00 1.39217812e+03 3.08689531e-02] number over target [0 1] worst case indices [25 25 25]
```

### Turn 15 — `execute_in_remote_docker` (`42060 ms` LLM)
- **Rationale**: The previous comparison held the center at a different clean-pick optimum for each clock mode. Reoptimizing against all cycle-corrected picks tests whether that asymmetry caused mode-selection errors. Compare fixed- and variable-rate failures and check strong-plume cases before promoting the change.
- **Alternatives Weighed**: `Use the fixed-rate prior that cleared the fixed-rate tests` (rejected: It increased clock failures when the corruption rate varied within the documented approximate regime.)

```bash
set -e
mkdir -p /tmp/tomo_joint
python3 - <<'PY'
from pathlib import Path
s = Path('/tmp/tomo_cycle/tomography_solver.py').read_text()
a = s.index('        design = np.column_stack'); b = s.index('\n\n    for _ in range(3):', a)
s = s[:a] + '''        # Optimize both candidate assignments under the same full-data model.
        # Slip-corrected picks inform the center as well as the shared cycle length.
        ij = ind[keep]
        t = target[keep]
        B = np.column_stack((rays.design[ij], mode[keep]))
        Q, U = qr(B, mode='economic')
        qy = Q.T @ t
        yy = t - Q @ qy

        def profile(pos, jac=False):
            if jac:
                g, dg = rays.kernel(pos, ij, derivative=True)
                dk = dg - Q @ (Q.T @ dg)
            else:
                g = rays.kernel(pos, ij)
            qg = Q.T @ g
            k = g - Q @ qg
            power = max(k @ k, 1.e-20)
            amp = max(0., k @ yy / power)
            if not jac:
                return amp*k - yy
            if amp == 0.:
                return np.zeros_like(dk)
            da = ((yy - 2*amp*k) @ dk) / power
            return amp*dk + k[:, None]*da

        # With no recognizable slips the cycle column is identically zero.
        if not np.any(mode):
            B = B[:, :-1]
            Q, U = qr(B, mode='economic')
            qy = Q.T @ t
            yy = t - Q @ qy
        pos = least_squares(profile, result[0], jac=lambda p: profile(p, True),
                            bounds=([0., 0.], [survey['W_m'], survey['H_m']]),
                            max_nfev=25).x
        g = rays.kernel(pos, ij)
        qg = Q.T @ g
        k = g - Q @ qg
        amp = max(0., k @ yy / max(k @ k, 1.e-20))
        beta = solve_triangular(U, qy - amp*qg)
        if np.any(mode):
            cycle = float(np.clip(beta[-1], 5.3, 6.2))
            beta = beta[:-1]
        else:
            cycle = 0.
        clock = np.r_[beta[2:], -beta[2:].sum()]
        excess = clock - _clip_clock(clock)
        error = rays.design[ind] @ beta + amp*rays.kernel(pos, ind) + cycle*mode - target
        return np.minimum(error**2, .6**2).sum() + 8.*(excess@excess)
''' + s[b:]
Path('/tmp/tomo_joint/tomography_solver.py').write_text(s)
PY
cd /
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/tomo_joint
for f in /tmp/tomography_cluster.py /tmp/tomography_variable_rate.py /tmp/tomography_strong.py; do
  echo "$f"
  python3 -B "$f"
done
```
- **Container Exit Code**: `0` (`14574 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
error m 0.17 peak delay ms 0.94 inliers wrongly removed 8 slips retained 7
miss clustered-slips 62 center error m 0.04 peak delay ms 1.11 inliers wrongly removed 8 slips retained 7
miss clustered-slips 72 center error m 0.12 peak delay ms 1.12 inliers wrongly removed 8 slips retained 7
clustered-slips worst center m / clock us / runtime s [8.60002431e-01 1.45281754e+03 3.05700242e-02] number over target [0 4] worst case indices [79 43 43]
miss majority-same-sign 29 center error m 0.06 peak delay ms 1.34 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 66 center error m 0.14 peak delay ms 1.23 inliers wrongly removed 6 slips retained 9
majority-same-sign worst center m / clock us / runtime s [1.26386157e+00 1.37209741e+03 2.90661592e-02] number over target [0 2] worst case indices [23 29 35]
majority-mixed-sign worst center m / clock us / runtime s [ 0.4982218  26.32757945  0.02761893] number over target [0 0] worst case indices [37 27 78]
/tmp/tomography_variable_rate.py
miss clustered-slips 14 center error m 0.3 peak delay ms 0.71 inliers wrongly removed 8 slips retained 7
miss clustered-slips 41 center error m 0.29 peak delay ms 0.75 inliers wrongly removed 8 slips retained 7
clustered-slips worst center m / clock us / runtime s [1.33988724e+00 1.44883082e+03 2.52954459e-02] number over target [0 2] worst case indices [30 14 14]
miss majority-same-sign 26 center error m 0.07 peak delay ms 1.34 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 51 center error m 0.13 peak delay ms 0.64 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 58 center error m 0.2 peak delay ms 0.69 inliers wrongly removed 6 slips retained 9
majority-same-sign worst center m / clock us / runtime s [6.85976477e-01 1.42392335e+03 4.31087471e-02] number over target [0 3] worst case indices [47 58 49]
majority-mixed-sign worst center m / clock us / runtime s [1.77021198e+00 6.29298244e+01 3.20942630e-02] number over target [0 0] worst case indices [25 25 72]
/tmp/tomography_strong.py
strong-plume worst center m / clock us / runtime s [6.23294463e-01 4.38113850e+01 1.92047858e-02] number over target [0 0] worst case indices [ 2 78 69]
strong-near-wall worst center m / clock us / runtime s [4.27825719e-01 5.12995770e+01 1.82495860e-02] number over target [0 0] worst case indices [37 58 33]
strong-wide-well worst center m / clock us / runtime s [ 0.46430578 28.26769941  0.03808802] number over target [0 0] worst case indices [66 44  0]
```

### Turn 16 — `execute_in_remote_docker` (`42576 ms` LLM)
- **Rationale**: Withhold the disputed receiver when predicting its absolute clock under the original zero-sum gauge. If this independent prediction separates the clock modes on both corruption regimes, use it to resolve ambiguous assignments; otherwise retain the simpler joint timing score.
- **Alternatives Weighed**: `Use the more complex joint nonlinear refit` (rejected: It left five variable-rate failures and increased fixed-rate failures compared with the simpler timing-only score.)

```bash
set -e
mkdir -p /tmp/tomo_anchor
python3 - <<'PY'
from pathlib import Path
base = Path('/tmp/tomo_cycle/tomography_solver.py').read_text()
needle = '                new_cost = cost(trial)\n                if new_cost < best_cost:'
assert base.count(needle) == 1
trial = '''                new_cost = cost(trial)
                # Infer the disputed receiver's clock from all other receivers.
                # Keeping the full design preserves the original zero-sum gauge.
                other = (rays.receiver[ind] != j) & good.ravel()[ind]
                disputed = rays.receiver[ind] == j
                A = np.column_stack((rays.design[ind], rays.kernel(fit[0], ind)))
                beta, _, rank, _ = np.linalg.lstsq(A[other], target[other], rcond=None)
                choose = new_cost < best_cost
                if rank == A.shape[1]:
                    e = target[disputed] - A[disputed] @ beta
                    original = np.median(e[good[:, j][available[:, j]]])
                    alternative = np.median(e[shifted[available[:, j]]])
                    separation = abs(original) - abs(alternative)
                    if abs(separation) > GATE:
                        choose = separation > 0.
                if choose:'''
for k, gate in enumerate((0., 1., 2.)):
    folder = Path('/tmp/tomo_anchor') / str(k)
    folder.mkdir(exist_ok=True)
    (folder/'tomography_solver.py').write_text(base.replace(needle, trial.replace('GATE', repr(gate))))
PY
cd /
export OPENBLAS_NUM_THREADS=1
for k in 0 1 2; do
  echo "independent clock anchor candidate $k"
  for f in /tmp/tomography_cluster.py /tmp/tomography_variable_rate.py; do
    PYTHONPATH=/tmp/tomo_anchor/$k python3 -B "$f" | grep 'number over target'
  done
done
```
- **Container Exit Code**: `0` (`36843 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[ 0 25] worst case indices [37 26  0]
clustered-slips worst center m / clock us / runtime s [1.33988724e+00 1.44883082e+03 2.40134648e-02] number over target [0 9] worst case indices [30 14 68]
majority-same-sign worst center m / clock us / runtime s [6.85976477e-01 1.39827158e+03 2.50928360e-02] number over target [ 0 11] worst case indices [47 70 20]
majority-mixed-sign worst center m / clock us / runtime s [1.77021198e+00 1.44357885e+03 3.57289871e-02] number over target [ 0 14] worst case indices [25 26 74]
independent clock anchor candidate 1
clustered-slips worst center m / clock us / runtime s [8.60002431e-01 1.45281754e+03 2.59460870e-02] number over target [0 8] worst case indices [79 43 52]
majority-same-sign worst center m / clock us / runtime s [1.36862307e+00 1.39073287e+03 2.94740109e-02] number over target [ 0 10] worst case indices [23 45 67]
majority-mixed-sign worst center m / clock us / runtime s [4.98221798e-01 1.43570093e+03 3.58045490e-02] number over target [ 0 18] worst case indices [37 63 45]
clustered-slips worst center m / clock us / runtime s [1.33988724e+00 1.44883082e+03 2.38970399e-02] number over target [0 7] worst case indices [30 14 17]
majority-same-sign worst center m / clock us / runtime s [6.85976477e-01 1.39827158e+03 2.57122149e-02] number over target [ 0 10] worst case indices [47 70  3]
majority-mixed-sign worst center m / clock us / runtime s [1.77021198e+00 1.44357885e+03 3.05175269e-02] number over target [0 8] worst case indices [25 26 26]
independent clock anchor candidate 2
clustered-slips worst center m / clock us / runtime s [8.60002431e-01 1.45281754e+03 2.48706660e-02] number over target [0 7] worst case indices [79 43 30]
majority-same-sign worst center m / clock us / runtime s [1.36862307e+00 1.38395783e+03 3.04630031e-02] number over target [0 9] worst case indices [23 32 77]
majority-mixed-sign worst center m / clock us / runtime s [4.98221798e-01 1.43570093e+03 3.23023191e-02] number over target [ 0 14] worst case indices [37 63 10]
clustered-slips worst center m / clock us / runtime s [1.33988724e+00 1.44883082e+03 2.57363750e-02] number over target [0 4] worst case indices [30 14 14]
majority-same-sign worst center m / clock us / runtime s [6.85976477e-01 1.39664138e+03 2.70597429e-02] number over target [0 5] worst case indices [47 37 15]
majority-mixed-sign worst center m / clock us / runtime s [1.77021198e+00 1.44357885e+03 3.41059200e-02] number over target [0 6] worst case indices [25 26 26]
```

### Turn 17 — `execute_in_remote_docker` (`35484 ms` LLM)
- **Rationale**: The current score fits unconstrained offsets and penalizes excess afterward. Test whether enforcing the documented bounds during scoring reduces mode-selection errors on both slip-rate regimes. If it does not, retain the simpler timing-only comparison.
- **Alternatives Weighed**: `Infer the disputed clock from other receivers` (rejected: The independent anchor increased clock failures, including mixed-sign failures the timing-only comparison had eliminated.)

```bash
set -e
mkdir -p /tmp/tomo_constrained
python3 - <<'PY'
from pathlib import Path
s = Path('/tmp/tomo_rate/0/tomography_solver.py').read_text()
s = s.replace('from scipy.optimize import least_squares', 'from scipy.optimize import LinearConstraint, least_squares, minimize')
a = s.index('        beta = np.linalg.lstsq(design[keep]'); b = s.index('\n\n    for _ in range(3):', a)
s = s[:a] + '''        has_cycle = np.any(mode)
        if not has_cycle:
            design = design[:, :-1]
        A = design[keep]
        y = target[keep]
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        nclock = obs.shape[1] - 1
        nparam = len(beta)
        # These constraints act in the fit, so an infeasible clock mode can
        # trade against the background only at its actual timing cost.
        C = np.zeros((nclock + 2 + int(has_cycle), nparam))
        C[:nclock, 2:2+nclock] = np.eye(nclock)
        C[nclock, 2:2+nclock] = -1.
        C[nclock+1, 2+nclock] = 1.
        lo = np.r_[np.full(nclock+1, -4.), 0.]
        hi = np.r_[np.full(nclock+1, 4.), np.inf]
        if has_cycle:
            C[-1, -1] = 1.
            lo, hi = np.r_[lo, 5.3], np.r_[hi, 6.2]
        v = C @ beta
        if np.any(v < lo) or np.any(v > hi):
            scale = np.linalg.norm(A, axis=0)
            scale[scale == 0.] = 1.
            B = A / scale
            def objective(p):
                e = B @ p - y
                return .5*(e @ e), B.T @ e
            result = minimize(objective, beta*scale, jac=True, method='SLSQP',
                              constraints=LinearConstraint(C/scale, lo, hi),
                              options={'ftol': 1.e-11, 'maxiter': 80})
            beta = result.x / scale
        error = design @ beta - target
        return np.minimum(error**2, .6**2).sum()
''' + s[b:]
Path('/tmp/tomo_constrained/tomography_solver.py').write_text(s)
PY
cd /
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/tomo_constrained
for f in /tmp/tomography_cluster.py /tmp/tomography_variable_rate.py; do
  python3 -B "$f" | grep 'number over target'
done
```
- **Container Exit Code**: `0` (`12353 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
clustered-slips worst center m / clock us / runtime s [8.60002431e-01 1.45281754e+03 2.47330361e-02] number over target [0 3] worst case indices [79 43 18]
majority-same-sign worst center m / clock us / runtime s [1.26386157e+00 1.37209741e+03 3.09174391e-02] number over target [0 2] worst case indices [23 29 60]
majority-mixed-sign worst center m / clock us / runtime s [ 0.4982218  26.32757945  0.03032146] number over target [0 0] worst case indices [37 27 78]
clustered-slips worst center m / clock us / runtime s [1.33988724e+00 1.44883082e+03 2.56417759e-02] number over target [0 2] worst case indices [30 14 14]
majority-same-sign worst center m / clock us / runtime s [6.85976477e-01 1.42392335e+03 4.33702359e-02] number over target [0 3] worst case indices [47 58 56]
majority-mixed-sign worst center m / clock us / runtime s [1.77021198e+00 6.29298244e+01 3.75995841e-02] number over target [0 0] worst case indices [25 25 48]
```

### Turn 18 — `execute_in_remote_docker` (`33991 ms` LLM)
- **Rationale**: Check whether a smaller rate penalty reduces clock-mode errors without relying on an exact corruption count. Separately, restrict amplitude marginalization to centers whose linearized uncertainty already exceeds 2 m; compare against the unchanged fit to see whether this avoids the ordinary-case regression.
- **Alternatives Weighed**: `Keep the directly constrained clock score` (rejected: It selected the same failing clock modes as the simpler score while requiring a constrained optimizer.)

```bash
set -euo pipefail
mkdir -p /tmp/tomo_last
python3 - <<'PY'
from pathlib import Path
base=Path('/tmp/tomo_rate/0/tomography_solver.py').read_text()
needle='return np.minimum(error**2, .6**2).sum() + 0.0 * (np.count_nonzero(mode) - round(.04*len(target)))**2 + 8.*(excess@excess)'
assert needle in base
for tag, weight in [('timing',0.),('soft',.0015),('soft2',.0035)]:
 p=Path('/tmp/tomo_last')/tag; p.mkdir(exist_ok=True)
 p.joinpath('tomography_solver.py').write_text(base.replace(needle, f'return np.minimum(error**2, .6**2).sum() + {weight}*(np.count_nonzero(mode)-.04*len(target))**2 + 8.*(excess@excess)'))
marg=Path('/tmp/tomo_marginal/tomography_solver.py').read_text()
block=marg[marg.index('    # Integrate over'):marg.index('    clock = _clip_clock(np.r_[coef[2:]')]
for tag, width in [('corner35',.035),('corner20',.020)]:
 p=Path('/tmp/tomo_last')/tag; p.mkdir(exist_ok=True)
 b=block.replace('> .75:', '> 2.:').replace('sig/.035',f'sig/{width}')
 s=base.replace('from scipy.special import erf','from scipy.special import erf, log_ndtr')
 loc=s.index('    clock = _clip_clock(np.r_[coef[2:]')
 p.joinpath('tomography_solver.py').write_text(s[:loc]+b+s[loc:])
PY
cd /
export OPENBLAS_NUM_THREADS=1
for v in timing soft soft2; do
  echo "clock candidate $v"
  for f in /tmp/tomography_cluster.py /tmp/tomography_variable_rate.py; do
    PYTHONPATH=/tmp/tomo_last/$v python3 -B "$f" | grep 'number over target'
  done
done
cp /tmp/tomo_compare/evaluate.py /tmp/tomo_last/evaluate.py
for v in timing corner35 corner20; do
  echo "wide-center-likelihood candidate $v"
  PYTHONPATH=/tmp/tomo_last/$v python3 -B /tmp/tomo_last/evaluate.py | grep -E '^(public|edge|varied|weak|clustered)'
done
```
- **Container Exit Code**: `0` (`46279 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
4 18]
weak-plume worst center m / clock us / runtime s [2.16444126e+00 2.66651345e+01 1.49777711e-02] number over target [1 0] worst case indices [46 33  8]
clustered-slips worst center m / clock us / runtime s [6.41864547e-01 1.38727457e+03 2.23854471e-02] number over target [0 1] worst case indices [54 38 38]
wide-center-likelihood candidate corner35
public 0 center m / clock us / runtime s 0.08947190669186113 5.301980093089273 0.009369445033371449 inlier RMS us 31.909037092861322 slips 10
public 1 center m / clock us / runtime s 0.1121281248088459 12.234506082460276 0.00804246892221272 inlier RMS us 31.614120381372906 slips 10
public-range worst center m / clock us / runtime s [1.70195898e+00 4.17881341e+01 1.63273788e-02] number over target [0 0] worst case indices [ 2  6 54]
edge-and-corner worst center m / clock us / runtime s [ 19.49675079 142.97301215   0.21989951] number over target [14  0] worst case indices [28 28 53]
varied-geometry worst center m / clock us / runtime s [9.54976905e-01 2.96823669e+01 2.69981830e-02] number over target [0 0] worst case indices [14 14 18]
weak-plume worst center m / clock us / runtime s [2.16444126e+00 2.66651345e+01 1.50552101e-02] number over target [1 0] worst case indices [46 33  8]
clustered-slips worst center m / clock us / runtime s [6.41864547e-01 1.38727457e+03 2.26202130e-02] number over target [0 1] worst case indices [54 38 38]
wide-center-likelihood candidate corner20
public 0 center m / clock us / runtime s 0.08947190669186113 5.301980093089273 0.009361874079331756 inlier RMS us 31.909037092861322 slips 10
public 1 center m / clock us / runtime s 0.1121281248088459 12.234506082460276 0.007952624000608921 inlier RMS us 31.614120381372906 slips 10
public-range worst center m / clock us / runtime s [1.70195898e+00 4.17881341e+01 1.58347711e-02] number over target [0 0] worst case indices [ 2  6 54]
edge-and-corner worst center m / clock us / runtime s [ 16.61933732 140.64183595   0.22287394] number over target [13  0] worst case indices [28 28 53]
varied-geometry worst center m / clock us / runtime s [9.54976905e-01 2.96823669e+01 2.70283751e-02] number over target [0 0] worst case indices [14 14 18]
weak-plume worst center m / clock us / runtime s [2.16444126e+00 2.66651345e+01 1.49420542e-02] number over target [1 0] worst case indices [46 33  8]
clustered-slips worst center m / clock us / runtime s [6.41864547e-01 1.38727457e+03 2.25112820e-02] number over target [0 1] worst case indices [54 38 38]
```

### Turn 19 — `execute_in_remote_docker` (`24696 ms` LLM)
- **Rationale**: Compare the saved baseline, clock-only challenger, and hybrid on identical public and synthetic cases, including a fresh geometry and noise seed. Drop gated averaging if it adds center misses outside the poorly covered corner slice; reject the combined fit if it worsens variable-rate clock selection.
- **Alternatives Weighed**: `Promote both changes without testing their interaction` (rejected: The center estimate feeds clock-mode scoring, so the two changes may not preserve each other's gains when combined.)

```bash
set -euo pipefail
mkdir -p /tmp/tomo_hybrid
python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/tomo_last')
corner = (p/'corner20/tomography_solver.py').read_text()
soft = (p/'soft2/tomography_solver.py').read_text()
start = corner.index('    # Integrate over positive amplitude')
end = corner.index('    clock = _clip_clock', start)
pos = soft.index('    clock = _clip_clock(np.r_[coef[2:]')
Path('/tmp/tomo_hybrid/tomography_solver.py').write_text((soft[:pos]+corner[start:end]+soft[pos:]).replace('from scipy.special import erf','from scipy.special import erf, log_ndtr'))
s = Path('/tmp/tomo_last/evaluate.py').read_text().replace('default_rng(625309)', 'default_rng(48107)')
Path('/tmp/tomo_fresh.py').write_text(s)
PY
cd /
export OPENBLAS_NUM_THREADS=1
for candidate in /holdout_task /tmp/tomo_last/soft2 /tmp/tomo_hybrid; do
  printf '\nCandidate %s\n' "$candidate"
  for test in /tmp/tomo_last/evaluate.py /tmp/tomo_fresh.py /tmp/tomography_variable_rate.py; do
    echo "$test"
    PYTHONPATH="$candidate" python3 -B "$test" | grep -E '^public|number over target'
  done
done
```
- **Container Exit Code**: `0` (`37462 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
us 31.909037092861322 slips 10
public 1 center m / clock us / runtime s 0.1121281248088459 12.234506082460276 0.008107939036563039 inlier RMS us 31.614120381372906 slips 10
public-range worst center m / clock us / runtime s [1.70195898e+00 4.17881341e+01 1.68160370e-02] number over target [0 0] worst case indices [ 2  6 32]
edge-and-corner worst center m / clock us / runtime s [ 16.61933732 140.64183595   0.24057792] number over target [13  0] worst case indices [28 28 53]
varied-geometry worst center m / clock us / runtime s [ 0.9549769  29.68236694  0.03143683] number over target [0 0] worst case indices [14 14 22]
weak-plume worst center m / clock us / runtime s [2.16444126e+00 2.66651345e+01 1.52640140e-02] number over target [1 0] worst case indices [46 33  8]
clustered-slips worst center m / clock us / runtime s [ 0.64186455 18.99694762  0.02598957] number over target [0 0] worst case indices [54 30 49]
/tmp/tomo_fresh.py
public 0 center m / clock us / runtime s 0.08947190669186113 5.301980093089273 0.009193265112116933 inlier RMS us 31.909037092861322 slips 10
public 1 center m / clock us / runtime s 0.1121281248088459 12.234506082460276 0.007781354943290353 inlier RMS us 31.614120381372906 slips 10
public-range worst center m / clock us / runtime s [8.62831069e-01 3.32992021e+01 1.66534069e-02] number over target [0 0] worst case indices [46 46 35]
edge-and-corner worst center m / clock us / runtime s [40.78480275 85.72103489  0.21477648] number over target [13  0] worst case indices [42 44 44]
varied-geometry worst center m / clock us / runtime s [7.27357008e-01 3.39114975e+01 2.71776831e-02] number over target [0 0] worst case indices [57 26 16]
weak-plume worst center m / clock us / runtime s [1.99019101e+00 2.14985686e+01 1.59160169e-02] number over target [1 0] worst case indices [45 49 14]
clustered-slips worst center m / clock us / runtime s [6.42416294e-01 2.08827020e+01 1.72639249e-02] number over target [0 0] worst case indices [10 59  2]
/tmp/tomography_variable_rate.py
clustered-slips worst center m / clock us / runtime s [1.33988724e+00 1.44883082e+03 2.31952239e-02] number over target [0 1] worst case indices [30 14 14]
majority-same-sign worst center m / clock us / runtime s [6.85976477e-01 1.45219353e+03 3.62373111e-02] number over target [0 5] worst case indices [47 25 65]
majority-mixed-sign worst center m / clock us / runtime s [1.77021198e+00 6.29298244e+01 2.74410609e-02] number over target [0 0] worst case indices [25 25 72]
```

### Turn 20 — `execute_in_remote_docker` (`21337 ms` LLM)
- **Rationale**: Install the tested hybrid and verify the exact artifact in an isolated Python process. Public schema, clock constraints, accuracy, and jackknife assertions catch installation or integration regressions; synthetic slices quantify the remaining failures without treating them as public-data results.
- **Alternatives Weighed**: `Ship the clock-only challenger` (rejected: The hybrid reduced corner misses from 18 to 13 on the fresh seed, with identical variable-rate clock results and no new public-range or weak-plume misses.)

```bash
set -euo pipefail
cd /holdout_task
cp /tmp/tomo_hybrid/tomography_solver.py tomography_solver.py
sha=$(sha256sum tomography_solver.py | cut -d' ' -f1)
mkdir -p /tmp/.hegel_candidate_archive
cp tomography_solver.py /tmp/.hegel_candidate_archive/v02_${sha:0:8}
printf 'Final candidate SHA-256: %s\n' "$sha"
cd /
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import runpy
import sys
sys.path.insert(0, '/holdout_task')
import tomography_solver
assert tomography_solver.__file__ == '/holdout_task/tomography_solver.py'
for test in ('tomography_validate.py', 'tomo_fresh.py', 'tomography_cluster.py', 'tomography_variable_rate.py', 'tomography_strong.py'):
    print('\n' + test, flush=True)
    runpy.run_path('/tmp/' + test)
PY
```
- **Container Exit Code**: `0` (`28790 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tomography_solver.py']`
- **Stdout (tail)**:
```text
-01 3.39114975e+01 2.70671151e-02] number over target [0 0] worst case indices [57 26 16]
weak-plume worst center m / clock us / runtime s [1.99019101e+00 2.14985686e+01 1.64000371e-02] number over target [1 0] worst case indices [45 49 22]
clustered-slips worst center m / clock us / runtime s [ 0.64241629 20.88270197  0.02717331] number over target [0 0] worst case indices [10 59 58]

tomography_cluster.py
clustered-slips worst center m / clock us / runtime s [8.60002431e-01 4.29950149e+01 2.61687022e-02] number over target [0 0] worst case indices [79 24 38]
majority-same-sign worst center m / clock us / runtime s [ 1.26386157 24.41285334  0.02936335] number over target [0 0] worst case indices [23 23 57]
majority-mixed-sign worst center m / clock us / runtime s [ 0.4982218  26.32757945  0.02655753] number over target [0 0] worst case indices [37 27 74]

tomography_variable_rate.py
miss clustered-slips 14 center error m 0.3 peak delay ms 0.71 inliers wrongly removed 8 slips retained 7
clustered-slips worst center m / clock us / runtime s [1.33988724e+00 1.44883082e+03 2.26073249e-02] number over target [0 1] worst case indices [30 14 14]
miss majority-same-sign 7 center error m 0.35 peak delay ms 0.68 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 25 center error m 0.14 peak delay ms 1.22 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 33 center error m 0.65 peak delay ms 0.93 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 51 center error m 0.13 peak delay ms 0.64 inliers wrongly removed 6 slips retained 9
miss majority-same-sign 58 center error m 0.2 peak delay ms 0.69 inliers wrongly removed 6 slips retained 9
majority-same-sign worst center m / clock us / runtime s [6.85976477e-01 1.45219353e+03 2.54908570e-02] number over target [0 5] worst case indices [47 25 79]
majority-mixed-sign worst center m / clock us / runtime s [1.77021198e+00 6.29298244e+01 3.37806130e-02] number over target [0 0] worst case indices [25 25 72]

tomography_strong.py
strong-plume worst center m / clock us / runtime s [6.23294463e-01 4.38113850e+01 1.99111379e-02] number over target [0 0] worst case indices [ 2 78 69]
strong-near-wall worst center m / clock us / runtime s [4.27825719e-01 5.12995770e+01 1.66525689e-02] number over target [0 0] worst case indices [37 58 78]
strong-wide-well worst center m / clock us / runtime s [ 0.46430578 28.26769941  0.03851638] number over target [0 0] worst case indices [66 44 78]
```

### Turn 21 — `submit_completed_task` (`20123 ms` LLM)
- **Rationale**: The installed artifact passed isolated-process checks for the public API, clock constraints, runtime, and public accuracy. The hybrid preserved public performance and improved corner recovery; synthetic failures remain and held-out survey performance is not yet known.
- **Alternatives Weighed**: `Ship the clock-only solver` (rejected: The verified hybrid reduced fresh-seed corner misses from 18 to 13 of 60 without changing public accuracy or variable-rate clock results.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/tomography_solver.py': 'FILE:9181'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tomography_solver.py`. Both public surveys met the targets: worst plume-center error was 0.112 m, worst clock RMSE was 12.23 µs, and each ran in under 0.01 s. Public source, receiver, and ray holdouts also passed.

Limitations: Some synthetic weak-plume and corner cases exceeded the center-error target. Clock RMSE exceeded the target in 6 of 240 synthetic cases with clustered slips and a varying slip rate.`

### Turn 22 — `execute_in_remote_docker` (`42724 ms` LLM)
- **Rationale**: Independent quadrature can catch missing finite-ray terms, and exact within-receiver differences plus the zero-sum gauge can catch clock leakage into the background or plume coefficients. Source-held-out angular residuals can reveal a missing background term; revisit the forward model only if those checks expose a discrepancy.
- **Alternatives Weighed**: `Test relay modes, Fokker–Planck/HJB drift, or within-batch classification accuracy` (rejected: The surveys provide traveltime picks, not relay states, stochastic dynamics, technical batches, or classification labels. Receiver offsets are the measured nuisance structure to audit.)

```bash
set -euo pipefail
cd /
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 python3 -I -B - <<'PY'
import pickle
import sys
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import qr
sys.path.insert(0, '/holdout_task')
import tomography_solver as solver
assert solver.__file__ == '/holdout_task/tomography_solver.py'
with open('/holdout_task/public_surveys.pkl', 'rb') as f:
    cases = pickle.load(f)
steps, weights = leggauss(64)
steps = (1 + steps[:, None, None])/2
for i, case in enumerate(cases):
    s, gt = case['survey'], case['gt']
    w, h, r = (s[key] for key in ('W_m','H_m','R_plume_m'))
    zs, zr, y = (np.asarray(s[key]) for key in ('z_src_m','z_rcv_m','T_obs_s'))
    start = time.perf_counter()
    out = solver.invert_plume_and_clocks(s)
    dt = time.perf_counter()-start
    assert set(out) == {'plume_xz_m','rcv_clock_bias_s'}
    center, clock = (np.asarray(out[key]) for key in ('plume_xz_m','rcv_clock_bias_s'))
    assert center.shape == (2,) and clock.shape == zr.shape
    assert 0 <= center[0] <= w and 0 <= center[1] <= h
    assert abs(clock.sum()) < 1e-12 and np.max(abs(clock)) <= .004+1e-15
    err_m = np.linalg.norm(center-gt['plume_xz_m'])
    err_us = 1e6*np.sqrt(np.mean((clock-gt['rcv_clock_bias_s'])**2))
    assert err_m <= 1.8 and err_us <= 250 and dt < 15
    rays = solver._Rays(w, zs, zr, r)
    dz = zr[None, :] - zs[:, None]
    centers = np.vstack([center,[0.,0.],[w,h]])
    for x,z in centers:
        dsq = (steps*w-x)**2 + (zs[None,:,None]+steps*dz-z)**2
        numerical = np.hypot(w,dz)*np.einsum('q,qij->ij', weights/2, np.exp(-.5*dsq/r**2))
        np.testing.assert_allclose(rays.kernel([x,z], slice(None)), numerical.ravel(), rtol=1e-11, atol=1e-11)
    _, _, prediction = solver._invert(s)
    clean = abs(y.ravel()-prediction) < .0027
    ind = np.flatnonzero(clean)
    g = rays.kernel(center, ind)
    A = np.column_stack((rays.design[ind,:2],g))
    target = 1e3*y.ravel()[ind]
    full = np.column_stack((A,rays.design[ind,2:]))
    beta = np.linalg.lstsq(full,target,rcond=None)[0][:3]
    # Within-receiver source differences cancel each clock exactly. Add the
    # equally weighted mean across receivers to retain the zero-sum gauge.
    rows, rhs, means, offsets = [], [], [], []
    for j in range(len(zr)):
        block = rays.receiver[ind] == j
        Aj, tj = A[block], target[block]
        a,b = np.triu_indices(len(tj),1)
        rows.append((Aj[a]-Aj[b])/np.sqrt(len(tj)))
        rhs.append((tj[a]-tj[b])/np.sqrt(len(tj)))
        means.append(Aj.mean(axis=0)); offsets.append(tj.mean())
    precision = 1./sum(1./len(Aj) for Aj in [target[rays.receiver[ind] == j] for j in range(len(zr))])
    rows.append(np.sqrt(precision)*np.sum(means,axis=0)[None,:]); rhs.append(np.array([np.sqrt(precision)*sum(offsets)]))
    pairwise = np.linalg.lstsq(np.vstack(rows),np.concatenate(rhs),rcond=None)[0]
    np.testing.assert_allclose(pairwise,beta,rtol=1e-9,atol=1e-10)
    # Score stable public inliers only, and use a separate source holdout for every prediction.
    residual = np.empty_like(y)
    for j in range(len(zs)):
        mask = np.ones(y.shape,bool); mask[j] = False
        _, _, pred = solver._invert(s,mask)
        residual[j] = 1e6*(y[j]-pred.reshape(y.shape)[j])
    angle = np.abs(dz[clean.reshape(y.shape)])
    rr = residual.ravel()[clean]
    buckets = [rr[angle<12],rr[(angle>=12)&(angle<35)],rr[angle>=35]]
    print(f'public {i}: center={err_m:.4f} m, clock RMSE={err_us:.2f} us, runtime={dt:.4f} s; pairwise-vs-joint max parameter difference={max(abs(pairwise-beta)):.3g} ms/m')
    print('source-holdout residual mean / RMS (us) by angle:', [(round(x.mean(),2),round(np.sqrt(np.mean(x*x)),2)) for x in buckets])
PY
```
- **Container Exit Code**: `0` (`788 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0: center=0.0895 m, clock RMSE=5.30 us, runtime=0.0088 s; pairwise-vs-joint max parameter difference=8.88e-16 ms/m
source-holdout residual mean / RMS (us) by angle: [(2.14, 33.06), (-0.91, 37.09), (-1.48, 33.6)]
public 1: center=0.1121 m, clock RMSE=12.23 us, runtime=0.0086 s; pairwise-vs-joint max parameter difference=1.5e-15 ms/m
source-holdout residual mean / RMS (us) by angle: [(0.98, 34.63), (-1.68, 33.32), (1.34, 37.38)]
```

### Turn 23 — `submit_completed_task` (`17532 ms` LLM)
- **Rationale**: The installed solver passed isolated-process checks. Independent quadrature confirmed the finite-ray kernel, and source-held-out residual means stayed within 2.14 µs across angle groups, giving no evidence for adding background terms. The synthetic failures are reported separately; hidden-survey performance is unknown.
- **Alternatives Weighed**: `Replace joint clock projection with explicit within-receiver pairwise differences` (rejected: The two estimates agreed to within 1.5e-15 ms/m on the public data, so the replacement would not improve the fit.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tomography_solver.py`. Both public surveys met the targets: worst plume-center error was 0.112 m, worst receiver-clock RMSE was 12.23 µs, and runtime was under 0.01 s per survey. All public source, receiver, and ray holdouts also passed.

Limitations: Some synthetic corner and weak-plume cases missed the center target. Clock RMSE exceeded the target in 6 of 240 synthetic cases with clustered slips and a varying slip rate.`

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
