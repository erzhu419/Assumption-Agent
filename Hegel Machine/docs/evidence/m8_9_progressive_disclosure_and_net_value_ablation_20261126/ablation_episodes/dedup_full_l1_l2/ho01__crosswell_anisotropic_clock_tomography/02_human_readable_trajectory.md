# Dialectical Synthesis Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_2_Selective_LiveGPT` | `ho01__crosswell_anisotropic_clock_tomography`

- **Official Strict Pass**: **`PASS (1.0)`** (`PASSED_ALL_CHECKS`)
- **Blind Public Synthesis Choice**: `antithesis` via `selective_router_confident_single_branch`
- **Synthesis Rationale**: Primary Antithesis branch satisfied public checks under policy; second branch skipped (`1.00u` nominal units).
- **Post-Lock Branch Verdicts**: Thesis (`Bare`) = `NOT_SPAWNED (0T)` | Antithesis (`Wormhole`) = `True` (12T)

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
| **Turns Used / Max Turns** | `12 / 28` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `6` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `384,492` (`372,095` / `74,112` / `12,397`) |
| **Total Episode Wall Time (s)** | `389.22s` |

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

### Turn 1 — `execute_in_remote_docker` (`15957 ms` LLM)
- **Rationale**: Inspect survey fields, geometry, and any supplied truth before building the forward model so the solver uses the actual input schema and the validation uses only available references.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle
import numpy as np
with open('public_surveys.pkl', 'rb') as f:
    data = pickle.load(f)
def show(x, name='data', depth=0):
    if isinstance(x, dict):
        print(name, 'keys:', list(x)[:20])
        if depth < 4:
            for k, v in list(x.items())[:8]:
                show(v, str(k), depth + 1)
    elif isinstance(x, (list, tuple)):
        print(name, type(x).__name__, len(x))
        if len(x) and depth < 4:
            if np.isscalar(x[0]): print(' ', x[:8])
            else:
                for i, v in enumerate(x[:5]): show(v, f'{name}[{i}]', depth + 1)
    elif isinstance(x, np.ndarray):
        print(name, x.shape, x.dtype, 'sample:', x.ravel()[:8], 'range:', (np.nanmin(x), np.nanmax(x)) if x.size and x.dtype.kind in 'iuf' else '')
    else:
        print(name, repr(x)[:240])
show(data)
PY
```
- **Container Exit Code**: `0` (`559 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
data list 2
data[0] keys: ['survey', 'gt']
survey keys: ['W_m', 'H_m', 'z_src_m', 'z_rcv_m', 'T_obs_s', 'R_plume_m']
W_m 40.0
H_m 80.0
z_src_m (15,) float64 sample: [ 6.         10.85714286 15.71428571 20.57142857 25.42857143 30.28571429
 35.14285714 40.        ] range: (6.0, 74.0)
z_rcv_m (16,) float64 sample: [ 5.          9.66666667 14.33333333 19.         23.66666667 28.33333333
 33.         37.66666667] range: (5.0, 75.0)
T_obs_s (15, 16) float64 sample: [0.01491584 0.01729941 0.01816543 0.0148971  0.01612493 0.02175942
 0.01714121 0.01843184] range: (0.01330233416627665, 0.03317839066777648)
R_plume_m 5.5
gt keys: ['plume_xz_m', 'rcv_clock_bias_s']
plume_xz_m list 2
  [16.5, 34.0]
rcv_clock_bias_s (16,) float64 sample: [-0.0019168   0.00045573  0.00110994 -0.00255604 -0.00179292  0.0032009
 -0.00228896 -0.0019091 ] range: (-0.002556039023045965, 0.003329653424440951)
data[1] keys: ['survey', 'gt']
survey keys: ['W_m', 'H_m', 'z_src_m', 'z_rcv_m', 'T_obs_s', 'R_plume_m']
W_m 40.0
H_m 80.0
z_src_m (15,) float64 sample: [ 6.         10.85714286 15.71428571 20.57142857 25.42857143 30.28571429
 35.14285714 40.        ] range: (6.0, 74.0)
z_rcv_m (16,) float64 sample: [ 5.          9.66666667 14.33333333 19.         23.66666667 28.33333333
 33.         37.66666667] range: (5.0, 75.0)
T_obs_s (15, 16) float64 sample: [0.01425288 0.0187468  0.01403325 0.0142902  0.01580415 0.01554096
 0.01899251 0.0161894 ] range: (0.01336071363590472, 0.028705721115872333)
R_plume_m 5.5
gt keys: ['plume_xz_m', 'rcv_clock_bias_s']
plume_xz_m list 2
  [26.0, 51.5]
rcv_clock_bias_s (16,) float64 sample: [-0.0009812   0.00347274 -0.00137483 -0.00143901 -0.00034719 -0.00111101
  0.00170437 -0.00184997] range: (-0.002568385497994811, 0.0034727403303111385)
```

### Turn 2 — `execute_in_remote_docker` (`35162 ms` LLM)
- **Rationale**: Use the public truth only to diagnose forward-model residuals. The measured gap between ordinary picks and skips will determine the robust fitting scale and whether to discard or correct skipped picks.
- **Alternatives Weighed**: `Use the mounted plate-wave localization operator` (rejected: It requires waveforms and wall arrivals; this survey provides only direct traveltime picks.)

```bash
cat > /tmp/tomo_diagnose.py <<'PY'
import pickle
import numpy as np
from scipy.special import erf
from scipy.optimize import least_squares

def ray_integral(width, source, receiver, x, z, radius):
    dz = receiver - source
    length = np.hypot(width, dz)
    ux, uz = width / length, dz / length
    parallel = ux * x + uz * (z - source)
    normal = -uz * x + ux * (z - source)
    rr = np.sqrt(2) * radius
    return np.sqrt(np.pi / 2) * radius * np.exp(-0.5 * (normal / radius)**2) * (erf((length - parallel) / rr) + erf(parallel / rr))

if __name__ == '__main__':
    with open('/holdout_task/public_surveys.pkl', 'rb') as f:
        data = pickle.load(f)
    for k, row in enumerate(data):
        s, gt = row['survey'], row['gt']
        src = s['z_src_m'][:, None]; rcv = s['z_rcv_m'][None, :]
        dz = rcv - src; w = s['W_m']; l = np.hypot(w, dz)
        y = 1000 * (s['T_obs_s'] - gt['rcv_clock_bias_s']).ravel()
        x, z = gt['plume_xz_m']
        p = ray_integral(w, src, rcv, x, z, s['R_plume_m'])
        a = np.column_stack([l.ravel(), (-dz**2/l).ravel(), p.ravel()])
        b = least_squares(lambda b: a@b-y, [0.4, 0.03, 0.1], jac=lambda b: a, loss='soft_l1', f_scale=0.3).x
        keep = abs(y-a@b) < 3
        b = np.linalg.lstsq(a[keep], y[keep], rcond=None)[0]
        e = (y-a@b).reshape(l.shape)
        print('survey', k, 'shape', l.shape, 's0, eps, amplitude (s/m, dimensionless, s/m):', [b[0]/1000,b[1]/b[0],b[2]/1000])
        print('inlier RMS/max (ms):', np.sqrt(np.mean(e.ravel()[keep]**2)), np.max(abs(e.ravel()[keep])), 'outlier residuals (ms):', np.round(e.ravel()[~keep], 3))
        print('inlier row-median range (ms):', np.round([np.nanmin(np.nanmedian(np.where(abs(e)<3,e,np.nan),axis=1)), np.nanmax(np.nanmedian(np.where(abs(e)<3,e,np.nan),axis=1))], 4), 'ground truth clock sum (s):', np.sum(gt['rcv_clock_bias_s']))
PY
python3 -I /tmp/tomo_diagnose.py
```
- **Container Exit Code**: `1` (`856 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
survey 0 shape (15, 16) s0, eps, amplitude (s/m, dimensionless, s/m): [0.0004923144550983192, 0.30209550590636164, -1.8052175787005504e-06]
inlier RMS/max (ms): 0.9976208108408803 2.534189882218712 outlier residuals (ms): [12.919 12.648 12.658 12.896 12.229 11.534 11.911  9.538  4.192 18.142
  7.323 13.683 16.395 18.879 22.049 12.489 12.806 12.894 12.609 12.83
 12.872 11.825  9.82   7.427  6.992 11.399 15.138 18.297 18.895 22.539
 12.487 13.054 12.902 12.812 12.57  12.243  9.053  6.936  3.241  8.052
  9.496 13.11  14.35  12.75  22.508 12.832 12.581 12.622 11.926 11.716
  9.456  4.47   4.473  4.464  6.836 23.576 12.944 12.025 12.516 12.444
 10.514  9.612  5.772  4.045  3.091  2.369 11.043 11.892 11.668 11.235
 10.211  9.659  7.94   5.912  4.956  3.76   5.244  7.554  5.778  3.282
  6.014  3.124 10.727  9.173  8.54   6.956  8.595  8.894  5.618  6.739
  9.118  9.172 10.268 10.484 10.755 11.003  9.104  8.638  3.097  4.524
  4.584  8.233 10.148 11.684 12.093 12.489 12.18  11.805 11.402 10.853
  9.026  6.363  9.85  11.845 12.316 12.235 12.347 13.218 12.853 23.722
 13.146 10.855  5.6    2.944  3.305  8.314  9.987 12.009 12.248 12.665
 12.546 12.867 12.936 12.74  12.606 12.183 11.427 29.315 14.798  9.659
 10.61  12.66  12.342 12.463 12.644 12.545 12.786 12.51  12.234 12.102
 11.756  8.097  3.112  2.558 10.881 10.313 12.473 12.773 12.177 12.789
 12.966 12.828 12.778 12.978 12.698 14.247  8.918  5.496  8.027  9.925
 12.11  13.107 12.948 12.511 22.356 12.577 12.669 12.73  11.903 15.308
 21.839  3.024  5.835 10.547 12.34  11.945 12.868 12.493 12.    12.747
 12.824 12.585 12.789 15.761 11.533  5.529 18.925  4.395  9.176 11.518
 10.86  12.345 12.796 12.855 13.008 12.587 12.902 13.211]
inlier row-median range (ms): [-0.7484  1.168 ] ground truth clock sum (s): 4.336808689942018e-19
survey 1 shape (15, 16) s0, eps, amplitude (s/m, dimensionless, s/m): [0.0, nan, 0.0]
```
- **Stderr (tail)**:
```text
<3,e,np.nan),axis=1))], 4), 'ground truth clock sum (s):', np.sum(gt['rcv_clock_bias_s']))
/tmp/tomo_diagnose.py:30: RuntimeWarning: invalid value encountered in scalar divide
  print('survey', k, 'shape', l.shape, 's0, eps, amplitude (s/m, dimensionless, s/m):', [b[0]/1000,b[1]/b[0],b[2]/1000])
/usr/local/lib/python3.12/dist-packages/numpy/core/fromnumeric.py:3504: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/usr/local/lib/python3.12/dist-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
Traceback (most recent call last):
  File "/tmp/tomo_diagnose.py", line 31, in <module>
    print('inlier RMS/max (ms):', np.sqrt(np.mean(e.ravel()[keep]**2)), np.max(abs(e.ravel()[keep])), 'outlier residuals (ms):', np.round(e.ravel()[~keep], 3))
                                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/core/fromnumeric.py", line 2810, in max
    return _wrapreduction(a, np.maximum, 'max', axis, None, out,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/core/fromnumeric.py", line 88, in _wrapreduction
    return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: zero-size array to reduction operation maximum which has no identity
```

### Turn 3 — `execute_in_remote_docker` (`13635 ms` LLM)
- **Rationale**: Return a fresh Jacobian to isolate SciPy's robust scaling from the physical design matrix. If the corrected residuals still lack a clear inlier–skip gap, investigate the forward model before fixing a rejection threshold.
- **Alternatives Weighed**: `Treat the large residuals as forward-model failure` (rejected: SciPy rescales the robust Jacobian in place, and the diagnostic returned the same array used for predictions.)

```bash
sed -i 's/jac=lambda b: a,/jac=lambda b: a.copy(),/' /tmp/tomo_diagnose.py && python3 -I /tmp/tomo_diagnose.py
```
- **Container Exit Code**: `0` (`777 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
survey 0 shape (15, 16) s0, eps, amplitude (s/m, dimensionless, s/m): [0.00042011466875072713, 0.14063166641797314, 6.487975213414034e-05]
inlier RMS/max (ms): 0.032527776325716774 0.08717676375424688 outlier residuals (ms): [-5.513  5.515  5.465  5.462  5.562  5.498 -5.499  5.52  -5.529 -5.504]
inlier row-median range (ms): [-0.0192  0.022 ] ground truth clock sum (s): 4.336808689942018e-19
survey 1 shape (15, 16) s0, eps, amplitude (s/m, dimensionless, s/m): [0.00038004242635738145, 0.1807746450155551, 7.261058522879244e-05]
inlier RMS/max (ms): 0.033178518273117054 0.09051888051656931 outlier residuals (ms): [-5.516  5.474  5.518 -5.48   5.5    5.522 -5.516  5.525  5.526  5.496]
inlier row-median range (ms): [-0.0095  0.0205] ground truth clock sum (s): -8.673617379884035e-19
```

### Turn 4 — `execute_in_remote_docker` (`61128 ms` LLM)
- **Rationale**: Profile the shared background and zero-sum clock terms for grid initialization, then fit the exact finite-ray model with a robust loss before refitting ordinary picks. The isolated public-data test measures plume and clock errors separately; source jackknifes test whether either estimate depends on one source depth.
- **Alternatives Weighed**: `Correct all skips by a fixed time offset` (rejected: Public skips are near 5.5 ms, while the problem describes approximately 6 ms. Rejecting separated picks avoids building that difference into the clock estimates.)

```bash
cat > /holdout_task/tomography_solver.py <<'PY'
"""Straight-ray inversion of cross-well picks for a Gaussian plume and receiver clocks."""

import numpy as np
from scipy.linalg import helmert, qr
from scipy.optimize import least_squares
from scipy.special import erf


def _clock_box(clock):
    """Project millisecond biases onto the zero-sum box."""
    if np.max(np.abs(clock)) <= 4.0:
        return clock - clock.mean()
    lo, hi = clock.min() - 4.0, clock.max() + 4.0
    for _ in range(48):
        mid = (lo + hi) / 2
        if np.clip(clock - mid, -4.0, 4.0).sum() > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(clock - (lo + hi) / 2, -4.0, 4.0)


def invert_plume_and_clocks(survey: dict) -> dict:
    w, h = float(survey['W_m']), float(survey['H_m'])
    radius = float(survey.get('R_plume_m', 5.5))
    src = np.asarray(survey['z_src_m'], dtype=float)
    rcv = np.asarray(survey['z_rcv_m'], dtype=float)
    times = np.asarray(survey['T_obs_s'], dtype=float)
    i, j = np.nonzero(np.isfinite(times))
    zs = src[i]
    dz = rcv[j] - zs
    length = np.hypot(w, dz)
    ux, uz = w / length, dz / length
    y = 1000 * times[i, j]
    # Orthogonal clock coordinates impose the gauge in every fit.
    clocks = helmert(len(rcv), full=False).T
    design = np.column_stack((length, -dz * dz / length, clocks[j]))
    p = design.shape[1]
    rr = np.sqrt(2.0) * radius
    cap = np.sqrt(np.pi / 2.0) * radius

    def plume(x, z, deriv=False):
        along = ux * x + uz * (z - zs)
        cross = -uz * x + ux * (z - zs)
        a, b = along / rr, (length - along) / rr
        env = np.exp(-0.5 * (cross / radius)**2)
        ray = cap * env * (erf(a) + erf(b))
        if not deriv:
            return ray
        dp = env * (np.exp(-a*a) - np.exp(-b*b))
        dn = -cross * ray / (radius * radius)
        return ray, dp * ux - dn * uz, dp * uz + dn * ux

    background = least_squares(lambda c: design @ c - y,
                               np.linalg.lstsq(design, y, rcond=None)[0],
                               jac=lambda c: design.copy(), loss='soft_l1',
                               f_scale=0.45).x
    # A broad mask is only for initialization. Nonlinear fits reconsider all picks.
    keep0 = np.abs(y - design @ background) < 2.7
    if keep0.sum() <= p + 3:
        keep0 = np.ones(len(y), dtype=bool)
    q, _ = qr(design[keep0], mode='economic')
    target = y[keep0] - q @ (q.T @ y[keep0])
    step = radius / 2
    xx, zz = np.meshgrid(np.linspace(0, w, max(2, int(np.ceil(w/step)) + 1)),
                         np.linspace(0, h, max(2, int(np.ceil(h/step)) + 1)))
    xy = np.column_stack((xx.ravel(), zz.ravel()))
    projected = plume(xy[:, :1], xy[:, 1:])[:, keep0]
    projected -= (projected @ q) @ q.T
    energy = np.einsum('ij,ij->i', projected, projected)
    cov = np.maximum(projected @ target, 0.0)
    score = cov * cov / np.maximum(energy, 1e-20)

    lower = np.r_[np.full(p, -np.inf), 0., 0., 0.]
    upper = np.r_[np.full(p, np.inf), np.inf, w, h]

    def residual(v):
        c, amp, x, z = v[:p], *v[p:]
        return design @ c + amp * plume(x, z) - y

    def jacobian(v):
        amp, x, z = v[p:]
        ray, dx, dzp = plume(x, z, True)
        return np.column_stack((design, ray, amp * dx, amp * dzp))

    def fit(v, mask, loss, scale):
        return least_squares(lambda c: residual(c)[mask], v,
                             jac=lambda c: jacobian(c)[mask], bounds=(lower, upper),
                             x_scale='jac', loss=loss, f_scale=scale, max_nfev=100)

    starts = []
    best = None
    all_picks = np.ones(len(y), dtype=bool)
    for k in np.argsort(score)[::-1]:
        x, z = xy[k]
        if any(np.hypot(x-a, z-b) < radius for a, b in starts):
            continue
        starts.append((x, z))
        amp = max(cov[k] / max(energy[k], 1e-20), 0.005)
        c = np.linalg.lstsq(design[keep0], (y - amp * plume(x, z))[keep0], rcond=None)[0]
        result = fit(np.r_[c, amp, x, z], all_picks, 'cauchy', 0.25)
        if best is None or result.cost < best.cost:
            best = result
        if len(starts) == 4:
            break

    v = best.x
    # The cycle-skips are separated by several milliseconds from normal picks.
    mask = np.abs(residual(v)) < 2.5
    if mask.sum() > p + 3:
        v = fit(v, mask, 'linear', 1.0).x
    return {'plume_xz_m': v[-2:].tolist(),
            'rcv_clock_bias_s': (_clock_box(clocks @ v[2:p]) / 1000).tolist()}
PY
cat > /tmp/check_tomo_public.py <<'PY'
import importlib.util
import pickle
import time
import numpy as np

spec = importlib.util.spec_from_file_location('tomography_solver', '/holdout_task/tomography_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_surveys.pkl', 'rb') as f:
    data = pickle.load(f)

def check(s, gt):
    start = time.perf_counter()
    got = m.invert_plume_and_clocks(s)
    clock = np.asarray(got['rcv_clock_bias_s'])
    assert abs(clock.sum()) < 1e-12 and np.max(abs(clock)) <= .004 + 1e-14
    return (np.linalg.norm(np.asarray(got['plume_xz_m']) - gt['plume_xz_m']),
            1000 * np.sqrt(np.mean((clock - gt['rcv_clock_bias_s'])**2)),
            time.perf_counter() - start)

for k, row in enumerate(data):
    s, gt = row['survey'], row['gt']
    print(k, 'full (m, ms, s)', check(s, gt), flush=True)
    slices = []
    for i in range(len(s['z_src_m'])):
        leave = dict(s, T_obs_s=s['T_obs_s'].copy())
        leave['T_obs_s'][i] = np.nan
        slices.append(check(leave, gt))
    v = np.asarray(slices)
    print(k, 'leave-one-source-out max (m, ms, s)', v.max(axis=0), 'worst rows', v.argmax(axis=0), flush=True)
PY
python3 -I /tmp/check_tomo_public.py
```
- **Container Exit Code**: `0` (`7919 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tomography_solver.py']`
- **Stdout (tail)**:
```text
0 full (m, ms, s) (0.08947150519271542, 0.005301981224771087, 0.239229696104303)
0 leave-one-source-out max (m, ms, s) [0.11461945 0.00685005 0.38512173] worst rows [7 2 2]
1 full (m, ms, s) (0.11212869049129459, 0.012234516527320712, 0.3966351931449026)
1 leave-one-source-out max (m, ms, s) [0.14536189 0.01513027 0.49234518] worst rows [11  0  6]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=df4d88132b26d914 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_df4d8813 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`32308 ms` LLM)
- **Rationale**: Receiver deletion can expose dependence on a single ray fan; independent quadrature simulations can expose boundary localization or initialization failures that the two interior public plumes do not exercise. If either slice misses tolerance, distinguish optimizer failure from weak geometric sensitivity before changing the solver.

```bash
cat > /tmp/check_tomo_stress.py <<'PY'
import runpy
import numpy as np
from numpy.polynomial.legendre import leggauss

base = runpy.run_path('/tmp/check_tomo_public.py', run_name='stress')
check, data = base['check'], base['data']
for k, row in enumerate(data):
    s, gt = row['survey'], row['gt']
    errors = []
    for j in range(len(s['z_rcv_m'])):
        keep = np.arange(len(s['z_rcv_m'])) != j
        times = s['T_obs_s'][:, keep]
        clock = gt['rcv_clock_bias_s'][keep]
        reduced = dict(s, z_rcv_m=s['z_rcv_m'][keep], T_obs_s=times-clock.mean())
        errors.append(check(reduced, dict(gt, rcv_clock_bias_s=clock-clock.mean())))
    print('public receiver jackknife', k, 'max (m, ms, s)', np.max(errors, axis=0), flush=True)

nodes, weights = leggauss(48)
u = (nodes + 1)/2
out = []
labels = []
for seed in range(40):
    rng = np.random.default_rng(seed)
    w, h = (32., 68.) if seed % 2 else (50., 95.)
    ns, nr = [(12, 11), (15, 16), (19, 21)][seed % 3]
    src = np.linspace(5, h-5, ns) + rng.uniform(-.75, .75, ns)
    rcv = np.linspace(5, h-5, nr) + rng.uniform(-.75, .75, nr)
    fx, fz = [(0.07, 0.15), (.93, .85), (.1, .82), (.89, .18), (.5, .5)][seed % 5]
    x, z = w*fx, h*fz
    a = src[:, None]
    dz = rcv[None, :] - a
    length = np.hypot(w, dz)
    integrand = np.exp(-((w*u - x)**2 + (a[..., None] + dz[..., None]*u - z)**2)/(2*5.5**2))
    path = length * (integrand @ (weights/2))
    clock = rng.uniform(-1, 1, nr)
    clock -= clock.mean()
    clock *= .004/np.max(np.abs(clock))
    s0, eps, amp = rng.uniform(.00033, .0005), rng.uniform(.04, .24), [.000035, .00007, .00012][seed % 3]
    times = s0 * (length - eps*dz**2/length) + amp*path + clock + rng.normal(0, .00007, (ns, nr))
    idx = rng.choice(times.size, round(.04 * times.size), replace=False)
    times.flat[idx] += .006 * rng.choice([-1, 1], len(idx))
    if seed % 4 == 0:
        times.flat[rng.choice(times.size, round(.10*times.size), replace=False)] = np.nan
    s = dict(W_m=w, H_m=h, z_src_m=src, z_rcv_m=rcv, T_obs_s=times)
    out.append(check(s, dict(plume_xz_m=[x, z], rcv_clock_bias_s=clock)))
    labels.append((seed, fx, fz, ns, nr, amp))
a = np.array(out)
print('synthetic 40-case maxima (m, ms, s):', a.max(axis=0), 'worst cases:', [labels[k] for k in a.argmax(axis=0)])
print('failures:', [(labels[k], a[k].tolist()) for k in np.flatnonzero(np.any(a > [1.8, .25, 15], axis=1))])
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/check_tomo_stress.py
```
- **Container Exit Code**: `0` (`2931 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 full (m, ms, s) (0.08947150519271542, 0.005301981224771087, 0.021582755958661437)
0 leave-one-source-out max (m, ms, s) [0.11461945 0.00685005 0.02820957] worst rows [ 7  2 12]
1 full (m, ms, s) (0.11212869049130103, 0.01223451652732094, 0.023689419962465763)
1 leave-one-source-out max (m, ms, s) [0.14536189 0.01513027 0.02672583] worst rows [11  0  3]
public receiver jackknife 0 max (m, ms, s) [0.12391781 0.00569548 0.0227065 ]
public receiver jackknife 1 max (m, ms, s) [0.14837699 0.01346302 0.03078155]
synthetic 40-case maxima (m, ms, s): [3.76923685 0.04101104 0.04746322] worst cases: [(30, 0.07, 0.15, 12, 11, 3.5e-05), (36, 0.93, 0.85, 12, 11, 3.5e-05), (8, 0.89, 0.18, 19, 21, 0.00012)]
failures: [((0, 0.07, 0.15, 12, 11, 3.5e-05), [2.8848452101952335, 0.010804656833146074, 0.01763985212892294]), ((6, 0.93, 0.85, 12, 11, 3.5e-05), [2.6782902550340273, 0.021464955412303467, 0.023715366143733263]), ((30, 0.07, 0.15, 12, 11, 3.5e-05), [3.7692368549203175, 0.015508878441475235, 0.019241129979491234]), ((33, 0.89, 0.18, 12, 11, 3.5e-05), [2.079688574603716, 0.029292591747070027, 0.013939108001068234]), ((36, 0.93, 0.85, 12, 11, 3.5e-05), [2.2501542762638858, 0.041011036330846726, 0.01593795116059482])]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=df4d88132b26d914 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_df4d8813 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`27286 ms` LLM)
- **Rationale**: Compare blind and truth-seeded fits on the same retained picks, then measure local sensitivity and whether recovering skips at a fitted common offset adds useful information. A lower truth-seeded cost would motivate better initialization; matching costs would point to weak information rather than optimizer failure.
- **Alternatives Weighed**: `Increase the number of solver starts immediately` (rejected: More starts cannot help if initialization at the known synthetic center converges to the same displaced minimum.)

```bash
python3 - <<'PY'
from pathlib import Path
s = Path('/holdout_task/tomography_solver.py').read_text()
s = s.replace('def invert_plume_and_clocks(survey: dict) -> dict:', 'def invert_plume_and_clocks(survey: dict, truth=None) -> dict:')
s = s.replace("    return {'plume_xz_m':", """    if truth is not None:
        local = v.copy()
        seeded = fit(np.r_[v[:-2], truth], mask, 'linear', 1.0)
        e = residual(v)
        sigma = np.sqrt(np.sum(e[mask]**2) / (mask.sum() - len(v)))
        jac = jacobian(v)[mask]
        sd = sigma * np.sqrt(np.diag(np.linalg.pinv(jac.T @ jac)))[-2:]
        if np.linalg.norm(v[-2:]-truth) > 1.6:
            near = e.copy()
            near[~mask] -= np.sign(e[~mask])*np.median(np.abs(e[~mask]))
            jj = jacobian(v)
            corrected = v - np.linalg.lstsq(jj, near, rcond=None)[0]
            print('weak slice: truth, fit, truth-seeded, linear SD (m), fitted amplitude, corrected, costs:', np.round(truth, 3), np.round(local[-2:], 3), np.round(seeded.x[-2:], 3), np.round(sd, 3), round(v[-3], 4), np.round(corrected[-2:], 3), np.round([0.5*np.sum(e[mask]**2), seeded.cost], 5), flush=True)
    return {'plume_xz_m':""")
Path('/tmp/tomo_weak_diagnose.py').write_text(s)
s = Path('/tmp/check_tomo_stress.py').read_text()
s = s[:s.index('base = ')] + '''from tomo_weak_diagnose import invert_plume_and_clocks

def check(s, gt):
    got = invert_plume_and_clocks(s, gt['plume_xz_m'])
    return (np.linalg.norm(np.asarray(got['plume_xz_m'])-gt['plume_xz_m']), 1000*np.sqrt(np.mean((np.array(got['rcv_clock_bias_s'])-gt['rcv_clock_bias_s'])**2)), 0.)

''' + s[s.index('nodes, weights'):]
Path('/tmp/check_tomo_weak_diagnose.py').write_text(s)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 /tmp/check_tomo_weak_diagnose.py
```
- **Container Exit Code**: `0` (`1525 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
weak slice: truth, fit, truth-seeded, linear SD (m), fitted amplitude, corrected, costs: [ 3.5  14.25] [ 0.822 13.179] [ 0.822 13.179] [2.38  0.593] 0.0487 [ 0.887 13.184] [0.24812 0.24812]
weak slice: truth, fit, truth-seeded, linear SD (m), fitted amplitude, corrected, costs: [46.5  80.75] [43.867 80.259] [43.867 80.259] [2.368 1.386] 0.0266 [44.002 80.173] [0.28693 0.28693]
weak slice: truth, fit, truth-seeded, linear SD (m), fitted amplitude, corrected, costs: [29.76 57.8 ] [29.615 59.472] [29.614 59.471] [2.89  2.139] 0.0412 [28.213 58.183] [0.22318 0.22318]
weak slice: truth, fit, truth-seeded, linear SD (m), fitted amplitude, corrected, costs: [ 2.24 10.2 ] [0.662 9.804] [0.662 9.804] [1.115 0.242] 0.0837 [0.459 9.768] [0.53352 0.53352]
weak slice: truth, fit, truth-seeded, linear SD (m), fitted amplitude, corrected, costs: [ 3.2  55.76] [ 4.95  55.669] [ 4.949 55.669] [1.454 0.447] 0.0313 [ 4.79  55.739] [0.22267 0.22267]
weak slice: truth, fit, truth-seeded, linear SD (m), fitted amplitude, corrected, costs: [ 3.5  14.25] [ 0.    12.851] [ 0.    12.851] [2.557 0.656] 0.0514 [-5.48  11.803] [0.2736 0.2736]
weak slice: truth, fit, truth-seeded, linear SD (m), fitted amplitude, corrected, costs: [28.48 12.24] [26.494 12.857] [26.494 12.857] [2.055 1.259] 0.022 [26.416 12.998] [0.25888 0.25888]
weak slice: truth, fit, truth-seeded, linear SD (m), fitted amplitude, corrected, costs: [46.5  80.75] [44.721 79.372] [44.724 79.373] [3.335 1.823] 0.0215 [43.93  79.367] [0.24487 0.24487]
synthetic 40-case maxima (m, ms, s): [3.76923685 0.04101104 0.        ] worst cases: [(30, 0.07, 0.15, 12, 11, 3.5e-05), (36, 0.93, 0.85, 12, 11, 3.5e-05), (0, 0.07, 0.15, 12, 11, 3.5e-05)]
failures: [((0, 0.07, 0.15, 12, 11, 3.5e-05), [2.8848452101952335, 0.010804656833146074, 0.0]), ((6, 0.93, 0.85, 12, 11, 3.5e-05), [2.6782902550340273, 0.021464955412303467, 0.0]), ((30, 0.07, 0.15, 12, 11, 3.5e-05), [3.7692368549203175, 0.015508878441475235, 0.0]), ((33, 0.89, 0.18, 12, 11, 3.5e-05), [2.079688574603716, 0.029292591747070027, 0.0]), ((36, 0.93, 0.85, 12, 11, 3.5e-05), [2.2501542762638858, 0.041011036330846726, 0.0])]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=df4d88132b26d914 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_df4d8813 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`28000 ms` LLM)
- **Rationale**: The diagnostic isolates an information limit in the hardest synthetic cases. Test new seeds with near-wall centers and public-like signal and noise to determine whether that failure persists under measured conditions. If it does, inspect ray coverage and residuals before changing the solver.
- **Alternatives Weighed**: `Add an interior location prior` (rejected: No minimum wall clearance is specified; shrinking weak estimates inward could move a real near-wall plume away from its true position.); `Recover skips by estimating a common jump` (rejected: The diagnostic did not show consistent improvement, and the task does not require the jump size to be identical across picks.)
- **ATMS NOGOOD DECLARED**: Refuted `Poor near-wall localization in the weakest synthetic slice is caused by bad initialization.` | Witness: `In the 3.7693 m failure, starting at the true center reached the same wall solution with the same retained-pick cost of 0.27360 ms²; the estimated horizontal standard error was 2.557 m.` | Blanket Diagnosis: `With sparse illumination, low plume amplitude, and noise above the public level, changes in horizontal position can be partly absorbed by changes in amplitude and receiver offsets.` | Sublated Pivot: `Keep the exact robust fit and verify edge coverage in an independent synthetic cohort at the measured public noise level; preserve the weak-signal limitation rather than tuning an unsupported location prior.`

```bash
python3 - <<'PY'
from pathlib import Path
s = Path('/tmp/check_tomo_stress.py').read_text()
s = s[:s.index('base = ')] + '''import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    base = runpy.run_path('/tmp/check_tomo_public.py', run_name='stress')
check = base['check']

''' + s[s.index('nodes, weights'):]
s = s.replace('range(40)', 'range(60, 120)').replace('[(12, 11), (15, 16), (19, 21)]', '[(15, 16), (17, 14), (19, 21)]').replace('[.000035, .00007, .00012]', '[.000055, .00007, .00010]').replace('rng.normal(0, .00007', 'rng.normal(0, .000035').replace('synthetic 40-case maxima', 'synthetic measured-SNR 60-case maxima')
s = s.replace('print(\'failures:\', [(labels[k], a[k].tolist()) for k in np.flatnonzero(np.any(a > [1.8, .25, 15], axis=1))])', '''failures = [(labels[k], a[k].tolist()) for k in np.flatnonzero(np.any(a > [1.8, .25, 15], axis=1))]
print('failures:', failures)
assert not failures''')
Path('/tmp/check_tomo_measured_snr.py').write_text(s)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/check_tomo_measured_snr.py
```
- **Container Exit Code**: `0` (`2824 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
synthetic measured-SNR 60-case maxima (m, ms, s): [0.93176757 0.02008359 0.0493486 ] worst cases: [(87, 0.1, 0.82, 15, 16, 5.5e-05), (96, 0.93, 0.85, 15, 16, 5.5e-05), (80, 0.07, 0.15, 19, 21, 0.0001)]
failures: []

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=df4d88132b26d914 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_df4d8813 (directory).
```

### Turn 8 — `submit_completed_task` (`15710 ms` LLM)
- **Rationale**: The archived artifact passed isolated Python verification, including both output tolerances, clock constraints, and runtime on public and public-noise stress slices. Truth-seeded fits reproduced the hardest synthetic failure, so adding more starts would not address it.
- **Alternatives Weighed**: `Add a location prior to reduce weak-signal error` (rejected: The survey does not specify a wall clearance, so the prior could bias legitimate near-wall locations.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`20` turns remaining; artifact status={'/holdout_task/tomography_solver.py': 'FILE:4437'}): Do NOT lock prematurely with `20` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tomography_solver.py`. Across public jackknife tests and 60 public-noise synthetic surveys, worst-case errors were 0.932 m for plume location and 0.0201 ms for receiver clocks, both within target. In a separate sparse, weak-signal stress test, plume error reached 3.77 m.`

### Turn 9 — `execute_in_remote_docker` (`38205 ms` LLM)
- **Rationale**: The specified angular law gives exactly the L and −Δz²/L columns; the finite Gaussian integral uses both ray endpoints. The next even angular harmonic tests for model mismatch without assuming it belongs in the solver. Held-out errors will show whether omitting clocks distorts plume localization or whether the extra harmonic improves prediction enough to warrant investigating the stated model.
- **Alternatives Weighed**: `Test relay modes, drift operators, or classification batch centering` (rejected: These surveys have none of those variables. The relevant nuisance is an additive receiver clock, which can be tested by holding out rays within each receiver and comparing with a no-clock fit.)

```bash
cat > /tmp/audit_tomo_folds.py <<'PY'
from pathlib import Path
import pickle
import numpy as np
from scipy.special import erf

path = Path('/holdout_task/tomography_solver.py')
source = path.read_text()
# Expose the fitted physical coefficients only in these in-memory audit copies.
needle = "    return {'plume_xz_m':"
assert source.count(needle) == 1

def load_variant(kind):
    text = source
    if kind == 'no_clocks':
        text = text.replace('clocks[j]))', 'clocks[j, :0]))')
        text = text.replace('clocks @ v[2:p]', 'np.zeros(len(rcv))')
    elif kind == 'extra_harmonic':
        text = text.replace('clocks[j]))', 'dz**4 / length**3, clocks[j]))')
        text = text.replace('clocks @ v[2:p]', 'clocks @ v[3:p]')
    text = text.replace(needle, "    return v[:p], v[p], {'plume_xz_m':")
    env = {}
    exec(compile(text, str(path), 'exec'), env)
    return env['invert_plume_and_clocks']

with open('/holdout_task/public_surveys.pkl', 'rb') as f:
    data = pickle.load(f)
for kind in ('exact', 'no_clocks', 'extra_harmonic'):
    invert = load_variant(kind)
    metrics = []
    for row in data:
        s, gt = row['survey'], row['gt']
        obs = s['T_obs_s'] * 1000
        w, r = s['W_m'], s['R_plume_m']
        a, b = np.asarray(s['z_src_m'])[:, None], np.asarray(s['z_rcv_m'])[None, :]
        d = b - a
        L = np.hypot(w, d)
        i, j = np.indices(L.shape)
        for k in range(5):
            test = (i + 2*j) % 5 == k
            beta, amp, out = invert(dict(s, T_obs_s=np.where(test, np.nan, s['T_obs_s'])))
            x, z = out['plume_xz_m']
            clocks = np.asarray(out['rcv_clock_bias_s'])
            assert abs(clocks.sum()) < 1e-12 and max(abs(clocks)) <= .004 + 1e-14
            u = (w*x+d*(z-a))/L
            n = (-d*x+w*(z-a))/L
            I = np.sqrt(np.pi/2)*r*np.exp(-.5*(n/r)**2)*(erf(u/(np.sqrt(2)*r))+erf((L-u)/(np.sqrt(2)*r)))
            pred = L*beta[0] - d**2/L*beta[1] + amp*I + 1000*clocks
            if kind == 'extra_harmonic':
                pred += d**4/L**3*beta[2]
            e = (obs-pred)[test]
            # Retain a tail-sensitive metric as well as ordinary-pick error so an ablation cannot win by rejecting more picks.
            normal = abs(e) < 2.5
            metrics.append([np.sqrt(np.mean(e[normal]**2)), np.mean(np.minimum(abs(e), 2.5)), np.linalg.norm([x,z]-np.array(gt['plume_xz_m'])), 1000*np.sqrt(np.mean((clocks-gt['rcv_clock_bias_s'])**2)), np.sum(~normal)])
    m = np.asarray(metrics)
    print(kind, 'worst fold [ordinary RMS ms, clipped MAE ms, center m, clock RMSE ms, skip count]:', np.round(m.max(axis=0), 5), 'mean clipped MAE ms:', round(m[:, 1].mean(), 5), 'total skips:', int(m[:, -1].sum()))
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/audit_tomo_folds.py && python3 -I /tmp/check_tomo_public.py
```
- **Container Exit Code**: `0` (`9572 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
exact worst fold [ordinary RMS ms, clipped MAE ms, center m, clock RMSE ms, skip count]: [0.04324 0.18686 0.15946 0.01452 3.     ] mean clipped MAE ms: 0.1311 total skips: 20
no_clocks worst fold [ordinary RMS ms, clipped MAE ms, center m, clock RMSE ms, skip count]: [ 0.92596  1.37591 31.29418  1.92048 18.     ] mean clipped MAE ms: 1.18997 total skips: 157
extra_harmonic worst fold [ordinary RMS ms, clipped MAE ms, center m, clock RMSE ms, skip count]: [0.04325 0.18682 0.15713 0.01449 3.     ] mean clipped MAE ms: 0.13116 total skips: 20
0 full (m, ms, s) (0.08947150519271542, 0.005301981224771087, 0.12370589701458812)
0 leave-one-source-out max (m, ms, s) [0.11461945 0.00685005 0.40336928] worst rows [7 2 9]
1 full (m, ms, s) (0.11212869049129459, 0.012234516527320712, 0.19683349411934614)
1 leave-one-source-out max (m, ms, s) [0.14536189 0.01513027 0.4934081 ] worst rows [11  0  1]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=df4d88132b26d914 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_df4d8813 (directory).
```

### Turn 10 — `submit_completed_task` (`14905 ms` LLM)
- **Rationale**: The archived solver passed clean-process verification of its output contract, clock constraints, and runtime. Held-out real picks support the stated angular model and show that explicit receiver clocks are necessary. The truth-seeded stress test reproduced the weak-signal failure, so more optimizer starts are not warranted.
- **Alternatives Weighed**: `Add a fourth-order angular term` (rejected: It did not improve held-out travel-time prediction.); `Omit receiver clocks` (rejected: Worst-fold plume error rose from 0.1595 m to 31.29 m.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`18` turns remaining; artifact status={'/holdout_task/tomography_solver.py': 'FILE:4437'}): Do NOT lock prematurely with `18` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tomography_solver.py`. Across public-data splits and 60 public-noise simulations, worst-case errors were 0.932 m for plume location and 0.0201 ms for receiver clocks, both within target. In a separate sparse, weak-signal stress test, plume error reached 3.77 m.`

### Turn 11 — `execute_in_remote_docker` (`37989 ms` LLM)
- **Rationale**: Contiguous angle holdout can expose anisotropy–clock coupling hidden by interleaved folds. The rational angular identity and depth reflection check the coordinate convention. If a band exceeds tolerance, examine whether the remaining ray geometry weakens parameter sensitivity before changing the model.
- **Alternatives Weighed**: `Repeat the interleaved ray folds or add an angular term` (rejected: Interleaved folds already showed no predictive gain from the extra term; withholding contiguous angle bands tests a different failure mode.)

```bash
cat > /tmp/audit_tomo_angles.py <<'PY'
import contextlib
import io
import runpy
import numpy as np
from scipy.special import erf

with contextlib.redirect_stdout(io.StringIO()):
    audit = runpy.run_path('/tmp/audit_tomo_folds.py')
invert = audit['load_variant']('exact')
for k, row in enumerate(audit['data']):
    s, gt = row['survey'], row['gt']
    w, h, r = s['W_m'], s['H_m'], s['R_plume_m']
    zs = s['z_src_m'][:, None]
    dz = s['z_rcv_m'][None, :] - zs
    length = np.hypot(w, dz)
    y = 1000 * s['T_obs_s']
    folds = np.array_split(np.argsort(np.abs(dz).ravel(), kind='stable'), 5)
    scores = []
    for held in folds:
        mask = np.zeros(dz.size, dtype=bool)
        mask[held] = True
        mask = mask.reshape(dz.shape)
        assert np.all(np.sum(~mask, axis=0) > 0)
        beta, amp, out = invert(dict(s, T_obs_s=np.where(mask, np.nan, s['T_obs_s'])))
        x, z = out['plume_xz_m']
        c = np.array(out['rcv_clock_bias_s'])
        tangent = dz / w
        sin2 = tangent**2 / (1 + tangent**2)
        bg = length * (beta[0] - beta[1] * sin2)
        np.testing.assert_allclose(bg, length * beta[0] - beta[1] * dz**2 / length, atol=1e-12)
        along = (w*x + dz*(z-zs)) / length
        normal = (-dz*x + w*(z-zs)) / length
        ray = np.sqrt(np.pi/2)*r*np.exp(-.5*(normal/r)**2)*(erf(along/(np.sqrt(2)*r))+erf((length-along)/(np.sqrt(2)*r)))
        e = (y - bg - amp*ray - 1000*c)[mask]
        ordinary = abs(e) < 2.5
        scores.append([np.linalg.norm(np.array([x,z])-gt['plume_xz_m']), 1000*np.sqrt(np.mean((c-gt['rcv_clock_bias_s'])**2)), np.sqrt(np.mean(e[ordinary]**2)), int(np.sum(~ordinary))])
    solver = audit['load_variant']('exact')
    _, _, nominal = solver(s)
    _, _, mirror = solver(dict(s, z_src_m=h-s['z_src_m'], z_rcv_m=h-s['z_rcv_m']))
    x, z = nominal['plume_xz_m']
    np.testing.assert_allclose(mirror['plume_xz_m'], [x, h-z], atol=1e-5)
    np.testing.assert_allclose(mirror['rcv_clock_bias_s'], nominal['rcv_clock_bias_s'], atol=1e-9)
    print('survey', k, 'angle bands shallow to steep [center m, clock RMSE ms, ordinary RMS ms, skips]:', np.round(scores, 5).tolist(), flush=True)
    assert np.all(np.asarray(scores)[:, :2] <= [1.8, .25])
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/audit_tomo_angles.py
```
- **Container Exit Code**: `0` (`1454 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
survey 0 angle bands shallow to steep [center m, clock RMSE ms, ordinary RMS ms, skips]: [[0.09572, 0.0082, 0.03194, 0.0], [0.08872, 0.00874, 0.03696, 3.0], [0.1127, 0.00568, 0.03846, 2.0], [0.11566, 0.00622, 0.03718, 1.0], [0.02463, 0.00507, 0.03051, 4.0]]
survey 1 angle bands shallow to steep [center m, clock RMSE ms, ordinary RMS ms, skips]: [[0.07367, 0.01417, 0.03574, 3.0], [0.15306, 0.01348, 0.02944, 1.0], [0.19332, 0.01368, 0.03484, 1.0], [0.0223, 0.01023, 0.0362, 2.0], [0.13175, 0.01263, 0.03461, 3.0]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=df4d88132b26d914 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_df4d8813 (directory).
```

### Turn 12 — `submit_completed_task` (`14136 ms` LLM)
- **Rationale**: The unchanged artifact passed clean-process verification, including runtime and clock constraints. Real-ray holdouts and public-noise stress tests met both tolerances. Truth-seeded fits reproduced the sparse weak-signal failure, so adding optimizer starts would not address the observed limitation.
- **Alternatives Weighed**: `Add an interior location prior` (rejected: No wall clearance is specified, so the prior could bias legitimate near-wall estimates.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tomography_solver.py`. Across public-data splits, worst errors were 0.193 m for plume location and 0.0152 ms for receiver clocks. All 60 simulations at public-data noise levels met both targets. In a separate sparse, weaker, noisier stress test, plume error reached 3.77 m.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-OOD-Holdout-12",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/tomography_solver.py": "FILE:4437"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "max_pos_err_m": 0.077,
    "max_clk_rmse_s": 0.00011
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"max_pos_err_m\": 0.077, \"max_clk_rmse_s\": 0.00011}\n__HEGEL_VERIFIER_AUTH_f4940f740c2395085fa87eff5af9e2f9__:{\"strict_pass\": true, \"max_pos_err_m\": 0.077, \"max_clk_rmse_s\": 0.00011}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 4872.5,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
