# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `vr04__indefinite_helmholtz_wrapped_phase_inversion`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M90a-VoluntaryReuse-4` |
| **Domain** | `Indefinite Acoustic Helmholtz Scattering & Wrapped-Phase Receiver Calibration` |
| **Wave / Wormhole Family** | `m90a_voluntary_reuse` / `fam_indefinite_wave_scattering_negative_control` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `8 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `162,453` (`155,323` / `47,488` / `7,130`) |
| **Total Episode Wall Time (s)** | `244.0s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`17230 ms` LLM)
- **Rationale**: Inspect the fixture schema and sparse solver availability before coding. An input layout mismatch changes the parser, and missing reference outputs would require separate manufactured-solution tests. Print only metadata, not the fixture contents.
- **Alternatives Weighed**: `Fit wrapped phase angles directly` (rejected: Complex phasors remove the angular branch cut and preserve the linear source model after pilot calibration.)

```bash
cd /holdout_task && python3 -I - <<'PY'
import pickle, numpy as np, scipy
with open('public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
print('SciPy:', scipy.__version__)
def describe(obj, depth=0):
    if isinstance(obj, dict):
        for key, value in obj.items():
            print('  '*depth + str(key) + ':', end=' ')
            describe(value, depth+1)
    elif isinstance(obj, (list, tuple)):
        print(type(obj).__name__, len(obj))
        if obj and depth < 4:
            describe(obj[0], depth+1)
    elif isinstance(obj, np.ndarray):
        print('array', obj.shape, obj.dtype)
    else:
        print(repr(obj)[:160])
describe(cases)
PY
```
- **Container Exit Code**: `0` (`664 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
SciPy: 1.13.0
list 2
  problem:     Lx: 1.0
    Ly: 1.0
    k0: 11.5
    beta_imp: 0.85
    n_sq: array (32, 32) float64
    basis: array (6, 32, 32) float64
    pilot_calibration:       known_field_re: array (12,) float64
      known_field_im: array (12,) float64
      amplitude_obs: array (12,) float64
      wrapped_phase_obs: array (12,) float64
    survey:       sensor_ij: array (140, 2) int64
      amplitude_obs: array (140,) float64
      wrapped_phase_obs: array (140,) float64
      tikhonov_alpha: 1e-06
  gt:     source_amplitudes: array (6,) float64
    global_phase_rad: -0.8141122105046681
    predicted_field_abs: array (32, 32) float64
```

### Turn 2 — `execute_in_remote_docker` (`47540 ms` LLM)
- **Rationale**: Fit the specified stencil with pivoted sparse LU. Public output errors and residual quantiles will reveal whether to investigate the forward operator, phase calibration, or survey estimator before stress testing.
- **Alternatives Weighed**: `Ordinary least squares for the survey` (rejected: An arbitrarily large impulse can dominate the fit; paired complex residual weights bound its influence without splitting a receiver into two observations.)

```bash
cd /holdout_task && cat > helmholtz_phase_inversion.py <<'PY'
"""Finite-volume Helmholtz source inversion from calibrated polar telemetry."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _polar(data):
    return np.asarray(data['amplitude_obs']) * np.exp(1j * np.asarray(data['wrapped_phase_obs']))


def _responses(problem):
    n_sq = np.asarray(problem['n_sq'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k = problem['k0']
    idx = np.arange(nx * ny).reshape(nx, ny)
    left = np.concatenate((idx[:-1, :].ravel(), idx[:, :-1].ravel()))
    right = np.concatenate((idx[1:, :].ravel(), idx[:, 1:].ravel()))
    face = np.repeat([hy / hx, hx / hy], [(nx - 1) * ny, nx * (ny - 1)])
    diag = -area * k**2 * n_sq.astype(complex).ravel()
    diag += np.bincount(left, weights=face, minlength=idx.size)
    diag += np.bincount(right, weights=face, minlength=idx.size)
    diag = diag.reshape(idx.shape)
    impedance = -1j * k * problem['beta_imp']
    diag[0, :] += hy * impedance
    diag[-1, :] += hy * impedance
    diag[:, 0] += hx * impedance
    diag[:, -1] += hx * impedance
    flat = idx.ravel()
    matrix = coo_matrix((np.r_[diag.ravel(), -face, -face],
                         (np.r_[flat, left, right], np.r_[flat, right, left])),
                        shape=(idx.size, idx.size)).tocsc()
    basis = np.asarray(problem['basis'])
    rhs = area * basis.reshape(basis.shape[0], idx.size).T
    return splu(matrix).solve(rhs.astype(complex)).reshape(nx, ny, -1)


def _robust_sources(design, obs, alpha):
    # Treat real and imaginary parts of a receiver as one observation when weighting.
    n, m = design.shape
    x = np.stack((design.real, design.imag), axis=1)
    y = np.stack((obs.real, obs.imag), axis=1)
    eye = np.eye(m)

    def fit(weight):
        root = np.sqrt(weight)
        wx = (x * root[:, None, None]).reshape(-1, m)
        wy = (y * root[:, None]).ravel()
        return np.linalg.lstsq(np.vstack((wx, np.sqrt(alpha) * eye)),
                               np.r_[wy, np.zeros(m)], rcond=None)[0]

    rng = np.random.default_rng(39281)
    estimate = fit(np.ones(n))
    best = np.median(np.abs(design @ estimate - obs))
    for _ in range(40):
        sample = rng.choice(n, min(m + 2, n), replace=False)
        trial = np.linalg.lstsq(x[sample].reshape(-1, m), y[sample].ravel(), rcond=None)[0]
        score = np.median(np.abs(design @ trial - obs))
        if score < best:
            estimate, best = trial, score

    # For circular complex Gaussian noise the residual magnitude is Rayleigh;
    # its median equals sqrt(2 log(2)) times the component standard deviation.
    floor = np.finfo(float).eps * max(1., np.max(np.abs(obs)))
    for _ in range(25):
        residual = np.abs(design @ estimate - obs)
        cutoff = max(4.685 * np.median(residual) / np.sqrt(2 * np.log(2)), floor)
        weight = np.maximum(0., 1. - (residual / cutoff)**2)**2
        updated = fit(weight)
        if np.linalg.norm(updated - estimate) <= 1e-10 * max(1., np.linalg.norm(updated)):
            return updated
        estimate = updated
    return estimate


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Return real source coefficients, calibrated phase, and pressure magnitude."""
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _polar(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)
    field_basis = _responses(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    obs = _polar(survey) * np.exp(-1j * phase)
    coefficients = _robust_sources(field_basis[i, j], obs, survey['tikhonov_alpha'])
    return {'source_amplitudes': coefficients,
            'global_phase_rad': phase,
            'predicted_field_abs': np.abs(field_basis @ coefficients)}
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('inversion','/holdout_task/helmholtz_phase_inversion.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
for i,c in enumerate(pickle.load(open('public_helmholtz_cases.pkl','rb'))):
 p,g=c['problem'],c['gt']; o=m.invert_helmholtz_sources_and_phase(p)
 err=lambda k: np.linalg.norm(o[k]-g[k])/np.linalg.norm(g[k])
 v=p['survey']; x,y=v['sensor_ij'].T; a=m._responses(p)[x,y]; z=m._polar(v)*np.exp(-1j*g['global_phase_rad']); truth=a@g['source_amplitudes']; r=np.abs(z-truth)
 print(i, 'errors source, phase, field:', err('source_amplitudes'), abs(np.angle(np.exp(1j*(o['global_phase_rad']-g['global_phase_rad'])))), err('predicted_field_abs'), 'K=',len(z),'residual quantiles',np.quantile(r,[0,.5,.8,.9,.93,.95,1]), 'sensor rms',np.sqrt(np.mean(abs(truth)**2)), 'singular values',np.linalg.svd(np.r_[a.real,a.imag],compute_uv=False))
PY
```
- **Container Exit Code**: `0` (`487 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
0 errors source, phase, field: 0.003043555885058487 0.0006227225657142733 0.002167665953037755 K= 140 residual quantiles [6.47881901e-06 2.04397750e-03 2.75476983e-03 3.74096331e-03
 1.00151822e-01 3.72538342e-01 7.40687115e-01] sensor rms 0.08224338994836831 singular values [0.8506539  0.68277291 0.64396571 0.48967366 0.39137982 0.28720722]
1 errors source, phase, field: 0.005838141247461257 0.00026689595102702407 0.003706654368838503 K= 140 residual quantiles [2.09377251e-04 1.80389227e-03 2.88952944e-03 4.26435661e-03
 9.99000576e-02 4.09010034e-01 6.86029873e-01] sensor rms 0.10347423196268857 singular values [0.88022524 0.69563054 0.63706238 0.44340971 0.39510715 0.28270252]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e417ac48f80751d8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e417ac48 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`41066 ms` LLM)
- **Rationale**: A rectangular-grid cross-check detects swapped face factors or boundary signs and would trigger a stencil fix. Public receiver omissions and manufactured phase-branch and impulse slices test sensitivity to individual receivers and possible phase or regression failures. The isolated-process import verifies the exact artifact without changing it.
- **Alternatives Weighed**: `Retune the robust weights on the two public cases` (rejected: All public errors are well within tolerance; stress the unchanged estimator before considering any tuning.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve
spec = importlib.util.spec_from_file_location('inversion', '/holdout_task/helmholtz_phase_inversion.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cases = pickle.load(open('public_helmholtz_cases.pkl', 'rb'))

def run(problem, gt):
    out = mod.invert_helmholtz_sources_and_phase(problem)
    a, ph, f = (out[k] for k in ('source_amplitudes', 'global_phase_rad', 'predicted_field_abs'))
    assert a.shape == gt['source_amplitudes'].shape and a.dtype.kind == 'f'
    assert isinstance(ph, float) and -np.pi < ph <= np.pi
    assert f.shape == problem['n_sq'].shape and f.dtype.kind == 'f' and np.all(f >= 0)
    return [np.linalg.norm(a-gt['source_amplitudes'])/np.linalg.norm(gt['source_amplitudes']), abs(np.angle(np.exp(1j*(ph-gt['global_phase_rad'])))), np.linalg.norm(f-gt['predicted_field_abs'])/np.linalg.norm(gt['predicted_field_abs'])]

def polar(z):
    return dict(amplitude_obs=np.abs(z), wrapped_phase_obs=np.angle(z))

def independent_field(p, a):
    nx, ny = p['n_sq'].shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    k = p['k0']
    def line(n, ratio, wall):
        d = np.full(n, 2*ratio, dtype=complex)
        d[[0,-1]] = ratio - 1j*k*p['beta_imp']*wall
        return diags([-ratio*np.ones(n-1), d, -ratio*np.ones(n-1)], [-1,0,1])
    H = kron(line(nx, hy/hx, hy), eye(ny)) + kron(eye(nx), line(ny, hx/hy, hx)) - diags(hx*hy*k*k*p['n_sq'].ravel())
    q = np.einsum('m,mij->ij',a,p['basis'])
    u = spsolve(H.tocsc(), hx*hy*q.ravel()).reshape(nx,ny)
    np.testing.assert_allclose(mod._responses(p)@a, u, rtol=1e-9, atol=1e-11)
    return u

scores = {'public': [], 'public leave-one-receiver-out': [], 'rectangular/phase-branch/impulses': []}
rng = np.random.default_rng(2328)
for c in cases:
    p, g = c['problem'], c['gt']
    scores['public'].append(run(p,g))
    sv = p['survey']
    n = len(sv['sensor_ij'])
    for i in range(n):
        keep = np.arange(n) != i
        scores['public leave-one-receiver-out'].append(run(dict(p, survey={k: v[keep] if isinstance(v,np.ndarray) else v for k,v in sv.items()}),g))
    for Lx, Ly, k in [(1.23,.84,10.9), (.83,1.21,12.1)]:
        p1 = dict(p,Lx=Lx,Ly=Ly,k0=k)
        a = g['source_amplitudes']
        u = independent_field(p1,a)
        known = p['pilot_calibration']['known_field_re'] + 1j*p['pilot_calibration']['known_field_im']
        for ph in [-np.pi+1e-6,np.pi-1e-6]:
            pilot = dict(known_field_re=known.real, known_field_im=known.imag, **polar(np.exp(1j*ph)*known))
            for trial in range(8):
                ids = rng.choice(u.size, n, replace=False)
                z = np.exp(1j*ph)*u.ravel()[ids]
                noise = 0.002*(rng.normal(size=n)+1j*rng.normal(size=n))
                # Also target strong-signal receivers, not just random ones.
                bad = np.argsort(np.abs(z))[-10:] if trial%2 else rng.choice(n,10,replace=False)
                noise[bad] += rng.uniform(.1,.8,len(bad))*np.exp(1j*rng.uniform(-np.pi,np.pi,len(bad)))
                survey = dict(sensor_ij=np.column_stack(np.unravel_index(ids,u.shape)), tikhonov_alpha=sv['tikhonov_alpha'], **polar(z+noise))
                gt = dict(source_amplitudes=a, global_phase_rad=ph, predicted_field_abs=np.abs(u))
                scores['rectangular/phase-branch/impulses'].append(run(dict(p1,pilot_calibration=pilot,survey=survey),gt))
for name, values in scores.items():
    worst = np.max(values,axis=0)
    print(name, len(values), 'worst source, phase rad, field:', worst, flush=True)
assert all(np.all(np.max(v,axis=0) <= [.03,.02,.03]) for v in scores.values())
PY
```
- **Container Exit Code**: `0` (`3213 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 2 worst source, phase rad, field: [0.00583814 0.00062272 0.00370665]
public leave-one-receiver-out 280 worst source, phase rad, field: [0.00641197 0.00062272 0.00416389]
rectangular/phase-branch/impulses 64 worst source, phase rad, field: [1.20298744e-02 4.44089210e-16 7.75705763e-03]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e417ac48f80751d8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e417ac48 (directory).
```

### Turn 4 — `submit_completed_task` (`15219 ms` LLM)
- **Rationale**: The exact saved artifact passed isolated-process verification. An independent rectangular-grid solver matched its forward response, and all three outputs passed the public, receiver-omission, and phase-branch impulse tests.
- **Alternatives Weighed**: `Retune survey regression` (rejected: Receiver omissions and targeted impulses exposed no tolerance failure that would justify tuning on the two public cases.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3944'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public cases, 280 leave-one-receiver-out checks, and 64 stress cases, the worst errors were 1.203% for source amplitudes, 0.000623 rad for phase, and 0.776% for field magnitude. All met the specified tolerances.`

### Turn 5 — `execute_in_remote_docker` (`27646 ms` LLM)
- **Rationale**: The independent stencil comparison already checked the relevant discrete algebra. Simultaneously omitting a receiver quadrant and one pilot probes spatial concentration and calibration leverage not covered by single-receiver omissions. A phase failure would direct attention to pilot weighting; a source or field failure with accurate phase would direct attention to survey conditioning and robustness.
- **Alternatives Weighed**: `Audit relay modes, drift operators, and batch classification` (rejected: The specified forward model is a steady-state linear Helmholtz operator, and the fixtures have no relay state, temporal drift input, batch labels, or classification outcome.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('inversion', '/holdout_task/helmholtz_phase_inversion.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def omit(data, keep):
    return {key: value[keep] if isinstance(value, np.ndarray) else value for key, value in data.items()}

def errors(out, gt):
    a = np.linalg.norm(out['source_amplitudes'] - gt['source_amplitudes']) / np.linalg.norm(gt['source_amplitudes'])
    phase = abs(np.angle(np.exp(1j * (out['global_phase_rad'] - gt['global_phase_rad']))))
    field = np.linalg.norm(out['predicted_field_abs'] - gt['predicted_field_abs']) / np.linalg.norm(gt['predicted_field_abs'])
    return np.array([a, phase, field])

with open('public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
limits = np.array([.030, .020, .030])
all_errors = []
for index, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    pilot, survey = p['pilot_calibration'], p['survey']
    nx, ny = p['n_sq'].shape
    i, j = np.asarray(survey['sensor_ij']).T
    groups = 2 * (i >= nx // 2) + (j >= ny // 2)
    pilot_ids = np.arange(len(pilot['known_field_re']))
    for quadrant in range(4):
        keep = groups != quadrant
        reduced_survey = omit(survey, keep)
        rows = []
        for dropped_pilot in pilot_ids:
            reduced_pilot = omit(pilot, pilot_ids != dropped_pilot)
            out = mod.invert_helmholtz_sources_and_phase(dict(p, survey=reduced_survey, pilot_calibration=reduced_pilot))
            rows.append(errors(out, gt))
        rows = np.asarray(rows)
        all_errors.extend(rows)
        print(f'case {index}, excluded quadrant {quadrant}, retained receivers {keep.sum()}: worst [source, phase, field] = {rows.max(axis=0)}')
worst = np.max(all_errors, axis=0)
print(f'{len(all_errors)} real-measurement slices; worst {worst}; headroom {limits - worst}')
assert np.all(worst <= limits)
PY
```
- **Container Exit Code**: `0` (`1253 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case 0, excluded quadrant 0, retained receivers 104: worst [source, phase, field] = [0.00413396 0.00074135 0.00349634]
case 0, excluded quadrant 1, retained receivers 104: worst [source, phase, field] = [0.00229276 0.00074135 0.00180746]
case 0, excluded quadrant 2, retained receivers 109: worst [source, phase, field] = [0.00506239 0.00074135 0.00287116]
case 0, excluded quadrant 3, retained receivers 103: worst [source, phase, field] = [0.00323985 0.00074135 0.00268927]
case 1, excluded quadrant 0, retained receivers 98: worst [source, phase, field] = [0.00595984 0.00058973 0.00393886]
case 1, excluded quadrant 1, retained receivers 104: worst [source, phase, field] = [0.00734819 0.00058973 0.00451474]
case 1, excluded quadrant 2, retained receivers 109: worst [source, phase, field] = [0.00565069 0.00058973 0.00402274]
case 1, excluded quadrant 3, retained receivers 109: worst [source, phase, field] = [0.00749241 0.00058973 0.00436417]
96 real-measurement slices; worst [0.00749241 0.00074135 0.00451474]; headroom [0.02250759 0.01925865 0.02548526]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e417ac48f80751d8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e417ac48 (directory).
```

### Turn 6 — `submit_completed_task` (`14664 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification. An independent rectangular-grid solver matched the forward operator, and every output passed the public, receiver-omission, phase-wrap, and impulse tests.
- **Alternatives Weighed**: `Retune phase calibration or survey weights` (rejected: All 96 combined pilot and receiver-quadrant omission tests passed with ample margin, so retuning on two public cases is not justified.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3944'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public cases and all stress tests, the worst errors were 1.203% for source amplitudes, 0.000741 rad for phase, and 0.776% for field magnitude. All met the specified tolerances.`

### Turn 7 — `execute_in_remote_docker` (`32600 ms` LLM)
- **Rationale**: The independent stencil comparison already checked the relevant algebra; these data have no batch labels or relay states. If the new slices exceed tolerance with phase still accurate, investigate regression leverage. Otherwise retain the verified implementation.
- **Alternatives Weighed**: `Repeat the pilot and receiver omission tests` (rejected: Those passed; relocating measured impulses tests a different failure mode: contamination concentrated at a wall or on receivers with high regression leverage.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('inversion', '/holdout_task/helmholtz_phase_inversion.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
worst = np.zeros(3)
with open('public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
for index, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    a, phase = gt['source_amplitudes'], gt['global_phase_rad']
    s = p['survey']
    i, j = s['sensor_ij'].T
    nx, ny = p['n_sq'].shape
    A = m._responses(p)[i, j]
    clean = np.exp(1j*phase)*(A @ a)
    residual = m._polar(s) - clean
    order = np.argsort(np.abs(residual))
    bad, good = order[-10:], order[:-10]
    # Preserve measured background residuals and relocate the ten measured spikes.
    background = residual.copy()
    background[bad] = residual[good[np.linspace(0,len(good)-1,10,dtype=int)]]
    Q, _ = np.linalg.qr(np.stack((A.real,A.imag),axis=1).reshape(-1,len(a)))
    leverage = np.sum(Q.reshape(len(i),2,-1)**2,axis=(1,2))
    placements = {'highest leverage': np.argsort(leverage)[-10:]}
    for side, distance in [('x low',i),('x high',nx-1-i),('y low',j),('y high',ny-1-j)]:
        placements[side] = np.lexsort((-leverage,distance))[:10]
    for name, ids in placements.items():
        local = []
        for shift in range(10):
            z = clean + background
            z[ids] += np.roll(residual[bad] - background[bad], shift)
            out = m.invert_helmholtz_sources_and_phase(dict(p,survey=dict(s,amplitude_obs=np.abs(z),wrapped_phase_obs=np.angle(z))))
            local.append([np.linalg.norm(out['source_amplitudes']-a)/np.linalg.norm(a), abs(np.angle(np.exp(1j*(out['global_phase_rad']-phase)))), np.linalg.norm(out['predicted_field_abs']-gt['predicted_field_abs'])/np.linalg.norm(gt['predicted_field_abs'])])
        peak = np.max(local,axis=0)
        worst = np.maximum(worst,peak)
        print(index,name,peak)
print('Worst [source relative, phase rad, field relative]:',worst)
assert np.all(worst <= [.030,.020,.030])
PY
```
- **Container Exit Code**: `0` (`1332 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 highest leverage [0.00312879 0.00062272 0.00266303]
0 x low [0.0029369  0.00062272 0.00203721]
0 x high [0.00291595 0.00062272 0.00190904]
0 y low [0.00367835 0.00062272 0.00244549]
0 y high [0.00253484 0.00062272 0.0017928 ]
1 highest leverage [0.00319425 0.0002669  0.00220833]
1 x low [0.00499029 0.0002669  0.00304453]
1 x high [0.00507931 0.0002669  0.00315524]
1 y low [0.00526397 0.0002669  0.00338839]
1 y high [0.00512127 0.0002669  0.00321342]
Worst [source relative, phase rad, field relative]: [0.00526397 0.00062272 0.00338839]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e417ac48f80751d8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e417ac48 (directory).
```

### Turn 8 — `submit_completed_task` (`14478 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification and matched an independent rectangular-grid solver. Receiver and pilot omissions, phase wrapping, and concentrated measured impulses did not expose a tolerance failure that warranted changing the implementation.
- **Alternatives Weighed**: `Retune receiver weights to account for measurement leverage` (rejected: Relocating measured spikes to wall-adjacent and high-leverage receivers produced at most 0.5264% source error, well below the 3% tolerance.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public cases and stress tests, the worst errors were 1.203% for source amplitudes, 0.000741 rad for phase, and 0.776% for field magnitude. All three met their specified tolerances.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M90a-VoluntaryReuse-4",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
