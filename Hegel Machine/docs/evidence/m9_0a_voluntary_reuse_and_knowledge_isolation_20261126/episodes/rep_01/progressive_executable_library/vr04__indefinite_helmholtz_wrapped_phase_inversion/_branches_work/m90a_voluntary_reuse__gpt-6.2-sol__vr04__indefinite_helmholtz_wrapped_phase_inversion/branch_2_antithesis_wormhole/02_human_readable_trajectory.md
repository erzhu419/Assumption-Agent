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
| **Turns Used / Max Turns** | `8 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `7` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `211,837` (`204,408` / `43,136` / `7,429`) |
| **Total Episode Wall Time (s)** | `243.74s` |

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

### Turn 1 — `execute_in_remote_docker` (`17180 ms` LLM)
- **Rationale**: Inspect only the public fixture schema and SciPy availability. This detects field-name, layout, or dependency mismatches before implementation and establishes whether public truth is available to measure each output error directly.
- **Alternatives Weighed**: `SPD iterative solve` (rejected: The specified complex, indefinite matrix does not satisfy the solver assumptions.); `Unwrap survey phase before fitting` (rejected: Complex reconstruction removes the wrap without assuming phase continuity between receivers.)

```bash
ls -lh /holdout_task; python3 - <<'PY'
import pickle
import numpy as np
from scipy import __version__
print('SciPy', __version__)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    data = pickle.load(f)
def describe(v, depth=0):
    if isinstance(v, dict) and depth < 6:
        return {k: describe(x, depth+1) for k, x in list(v.items())[:12]}
    if isinstance(v, (list, tuple)) and depth < 6:
        return [type(v).__name__, len(v), *[describe(x, depth+1) for x in v[:2]]]
    if isinstance(v, np.ndarray):
        return (v.shape, str(v.dtype))
    return v if np.isscalar(v) else type(v).__name__
print(describe(data))
PY
```
- **Container Exit Code**: `0` (`692 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 140K
-rw-r--r-- 1 root root 139K Nov 26 00:46 public_helmholtz_cases.pkl
SciPy 1.13.0
['list', 2, {'problem': {'Lx': 1.0, 'Ly': 1.0, 'k0': 11.5, 'beta_imp': 0.85, 'n_sq': ((32, 32), 'float64'), 'basis': ((6, 32, 32), 'float64'), 'pilot_calibration': {'known_field_re': ((12,), 'float64'), 'known_field_im': ((12,), 'float64'), 'amplitude_obs': ((12,), 'float64'), 'wrapped_phase_obs': ((12,), 'float64')}, 'survey': {'sensor_ij': ((140, 2), 'int64'), 'amplitude_obs': ((140,), 'float64'), 'wrapped_phase_obs': ((140,), 'float64'), 'tikhonov_alpha': 1e-06}}, 'gt': {'source_amplitudes': ((6,), 'float64'), 'global_phase_rad': -0.8141122105046681, 'predicted_field_abs': ((32, 32), 'float64')}}, {'problem': {'Lx': 1.0, 'Ly': 1.0, 'k0': 11.5, 'beta_imp': 0.85, 'n_sq': ((32, 32), 'float64'), 'basis': ((6, 32, 32), 'float64'), 'pilot_calibration': {'known_field_re': ((12,), 'float64'), 'known_field_im': ((12,), 'float64'), 'amplitude_obs': ((12,), 'float64'), 'wrapped_phase_obs': ((12,), 'float64')}, 'survey': {'sensor_ij': ((140, 2), 'int64'), 'amplitude_obs': ((140,), 'float64'), 'wrapped_phase_obs': ((140,), 'float64'), 'tikhonov_alpha': 1e-06}}, 'gt': {'source_amplitudes': ((6,), 'float64'), 'global_phase_rad': -1.3120678293854167, 'predicted_field_abs': ((32, 32), 'float64')}}]
```

### Turn 2 — `execute_in_remote_docker` (`44805 ms` LLM)
- **Rationale**: Measure all three outputs against public truth. Forward-field agreement isolates stencil failures from regression failures, and receiver residuals and singular values reveal whether impulse rejection or ridge bias needs adjustment.
- **Alternatives Weighed**: `Separate robust weights for real and imaginary components` (rejected: A complex impulse corrupts a receiver as a pair; separate weights would make the fit depend on the real/imaginary coordinate orientation.)

```bash
cat > /holdout_task/helmholtz_phase_inversion.py <<'PY'
"""Finite-volume Helmholtz inversion from pilot-calibrated polar data."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _face_laplacian(n):
    diagonal = np.full(n, 2.0)
    diagonal[0] -= 1.0
    diagonal[-1] -= 1.0
    return sparse.diags((-np.ones(n - 1), diagonal, -np.ones(n - 1)),
                        (-1, 0, 1), shape=(n, n), format='csc')


def _responses(problem):
    index_sq = np.asarray(problem['n_sq'])
    nx, ny = index_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k = problem['k0']
    walls = np.zeros((nx, ny))
    walls[0, :] += hy
    walls[-1, :] += hy
    walls[:, 0] += hx
    walls[:, -1] += hx
    diagonal = -area * k**2 * index_sq - 1j * k * problem['beta_imp'] * walls
    matrix = ((hy / hx) * sparse.kron(_face_laplacian(nx), sparse.eye(ny))
              + (hx / hy) * sparse.kron(sparse.eye(nx), _face_laplacian(ny))
              + sparse.diags(diagonal.ravel()))
    basis = np.asarray(problem['basis']).reshape(-1, nx * ny)
    return splu(matrix.tocsc()).solve((area * basis.T).astype(np.complex128))


def _phasors(data):
    return np.asarray(data['amplitude_obs']) * np.exp(1j * np.asarray(data['wrapped_phase_obs']))


def _robust_real_fit(design, observed, alpha):
    # A sensor's real and imaginary components share an outlier weight.
    y = np.r_[observed.real, observed.imag]
    x = np.vstack((design.real, design.imag))
    penalty = np.sqrt(alpha) * np.eye(design.shape[1])

    def fit(weights):
        w = np.sqrt(np.tile(weights, 2))
        return np.linalg.lstsq(np.vstack((w[:, None] * x, penalty)),
                               np.r_[w * y, np.zeros(design.shape[1])], rcond=None)[0]

    coef = fit(np.ones(len(observed)))
    floor = max(np.finfo(float).tiny, 1e-12 * np.median(np.abs(observed)))
    # Convex, smooth approximation to absolute complex residuals for the start.
    for _ in range(25):
        radius = np.abs(design @ coef - observed)
        scale = max(np.median(radius), floor)
        nxt = fit(1 / np.hypot(1, radius / scale))
        change = np.linalg.norm(design @ (nxt - coef)) / np.sqrt(len(observed))
        coef = nxt
        if change <= 1e-5 * scale:
            break
    # Redescend once close: gross impulses no longer bias the final fit.
    for _ in range(25):
        radius = np.abs(design @ coef - observed)
        scale = max(np.median(radius), floor)
        weights = np.maximum(0, 1 - (radius / (4 * scale))**2)**2
        nxt = fit(weights)
        change = np.linalg.norm(design @ (nxt - coef)) / np.sqrt(len(observed))
        coef = nxt
        if change <= 1e-5 * scale:
            break
    return coef


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Recover real source strengths, oscillator phase, and field magnitudes."""
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _phasors(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)
    responses = _responses(problem)
    shape = np.shape(problem['n_sq'])
    survey = problem['survey']
    indices = np.ravel_multi_index(np.asarray(survey['sensor_ij']).T, shape)
    design = responses[indices]
    observed = np.exp(-1j * phase) * _phasors(survey)
    coef = _robust_real_fit(design, observed, survey['tikhonov_alpha'])
    return {'source_amplitudes': coef, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(responses @ coef).reshape(shape)}
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys, pickle
import numpy as np
sys.path.insert(0, '/holdout_task')
import helmholtz_phase_inversion as h
for i, case in enumerate(pickle.load(open('/holdout_task/public_helmholtz_cases.pkl','rb'))):
    p, gt = case['problem'], case['gt']
    ans = h.invert_helmholtz_sources_and_phase(p)
    wave = h._responses(p)
    survey = p['survey']
    design = wave[np.ravel_multi_index(survey['sensor_ij'].T,p['n_sq'].shape)]
    y = np.exp(-1j*gt['global_phase_rad'])*h._phasors(survey)
    errs = [np.linalg.norm(ans[k]-gt[k])/np.linalg.norm(gt[k]) for k in ('source_amplitudes','predicted_field_abs')]
    print(i, 'errors source, field, phase:', *errs, abs(np.angle(np.exp(1j*(ans['global_phase_rad']-gt['global_phase_rad'])))))
    print('design singular values',np.linalg.svd(np.vstack((design.real,design.imag)),compute_uv=False),'response residual quantiles',np.quantile(np.abs(y-design@gt['source_amplitudes']),[0,.5,.9,.94,1]),'field forward agreement',np.linalg.norm(np.abs(wave@gt['source_amplitudes']).reshape(p['n_sq'].shape)-gt['predicted_field_abs'])/np.linalg.norm(gt['predicted_field_abs']))
PY
```
- **Container Exit Code**: `0` (`581 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
0 errors source, field, phase: 0.003049456216725632 0.0021684625279960426 0.0006227225657142733
design singular values [0.8506539  0.68277291 0.64396571 0.48967366 0.39137982 0.28720722] response residual quantiles [6.47881901e-06 2.04397750e-03 3.74096331e-03 3.61614744e-01
 7.40687115e-01] field forward agreement 1.2624230001329256e-15
1 errors source, field, phase: 0.005824781336347079 0.0037006427066568633 0.00026689595102702407
design singular values [0.88022524 0.69563054 0.63706238 0.44340971 0.39510715 0.28270252] response residual quantiles [2.09377251e-04 1.80389227e-03 4.26435661e-03 3.74277075e-01
 6.86029873e-01] field forward agreement 8.893619912307587e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=678dad5702cac3be (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_678dad57 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`45224 ms` LLM)
- **Rationale**: Public forward magnitudes agree with the reference to 1.3e-15 relative error. Leave-one-receiver and leave-one-pilot slices test single-channel dependence; an independent face-by-face stencil on rectangular cells tests axis scaling and wall signs, while phase-wrap and concentrated-impulse slices test calibration and robust fitting separately.
- **Alternatives Weighed**: `Tune the robust fit against the two public cases` (rejected: Both public cases have ample tolerance margin; a failed receiver or geometry slice would provide a concrete reason to change the fit.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy
import pickle
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
sys.path.insert(0, '/holdout_task')
from helmholtz_phase_inversion import invert_helmholtz_sources_and_phase as invert

def error(result, truth, shape):
    assert set(result) == {'source_amplitudes', 'predicted_field_abs', 'global_phase_rad'}
    assert isinstance(result['global_phase_rad'], float) and -np.pi < result['global_phase_rad'] <= np.pi
    values = []
    for key, expected_shape in [('source_amplitudes', truth['source_amplitudes'].shape), ('predicted_field_abs', shape)]:
        a, b = result[key], truth[key]
        assert a.shape == expected_shape and a.dtype.kind == 'f' and np.isfinite(a).all()
        values.append(np.linalg.norm(a-b)/np.linalg.norm(b))
    values.append(abs(np.angle(np.exp(1j*(result['global_phase_rad']-truth['global_phase_rad'])))))
    return values

def independent_field(p, coef):
    nx, ny = p['n_sq'].shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    k = p['k0']
    row, col, value = [], [], []
    for i in range(nx):
        for j in range(ny):
            q = i*ny+j
            diag = -hx*hy*k*k*p['n_sq'][i,j] + 0j
            for di, dj, face, gap in [(1,0,hy,hx),(-1,0,hy,hx),(0,1,hx,hy),(0,-1,hx,hy)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < nx and 0 <= nj < ny:
                    row.append(q); col.append(ni*ny+nj); value.append(-face/gap)
                    diag += face/gap
                else:
                    diag -= 1j*k*p['beta_imp']*face
            row.append(q); col.append(q); value.append(diag)
    H = coo_matrix((value,(row,col)), shape=(nx*ny,nx*ny)).tocsc()
    rhs = hx*hy*(coef @ p['basis'].reshape(len(coef),-1))
    return spsolve(H, rhs).reshape(nx,ny)

def polar(target, z):
    target['amplitude_obs'] = abs(z)
    target['wrapped_phase_obs'] = np.angle(z)

with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
limits = np.array([.030,.030,.020])
all_worst = np.zeros(3)
for case_id, case in enumerate(cases):
    p0, gt = case['problem'], case['gt']
    shape = p0['n_sq'].shape
    for group, keys in [('survey', ('sensor_ij','amplitude_obs','wrapped_phase_obs')), ('pilot_calibration', ('known_field_re','known_field_im','amplitude_obs','wrapped_phase_obs'))]:
        values = []
        for omit in range(len(p0[group][keys[0]])):
            p = dict(p0)
            p[group] = dict(p0[group])
            for key in keys:
                p[group][key] = np.delete(p0[group][key], omit, axis=0)
            values.append(error(invert(p), gt, shape))
        values = np.array(values)
        worst = values.max(axis=0)
        print('public', case_id, group, len(values), 'worst source/field/phase:', worst, 'omitted indices:', values.argmax(axis=0))
        all_worst = np.maximum(all_worst, worst)
    p = copy.deepcopy(p0)
    p.update(Lx=1.22, Ly=.87, k0=12.4, beta_imp=.63)
    rng = np.random.default_rng(230+case_id)
    coef = rng.normal(size=len(gt['source_amplitudes']))
    field = independent_field(p, coef)
    survey, pilot = p['survey'], p['pilot_calibration']
    u = field[tuple(survey['sensor_ij'].T)]
    pilot_field = pilot['known_field_re'] + 1j*pilot['known_field_im']
    values = []
    for phase in [np.pi-1.e-8, -np.pi+1.e-8, np.pi, -.19]:
        truth = {'source_amplitudes':coef, 'predicted_field_abs':abs(field), 'global_phase_rad':phase}
        rot = np.exp(1j*phase)
        polar(pilot, pilot_field*rot)
        for mode in ['clean', 'random impulses', 'strong-channel impulses']:
            z = rot*u
            if mode != 'clean':
                noise = .0015*(rng.normal(size=u.size)+1j*rng.normal(size=u.size))
                z = z+noise
                bad = rng.choice(u.size,10,replace=False) if mode == 'random impulses' else np.argsort(abs(u))[-10:]
                z[bad] += rng.uniform(.25,1.,size=bad.size)*np.exp(1j*rng.uniform(-np.pi,np.pi,size=bad.size))
            polar(survey, z)
            e = error(invert(p), truth, shape)
            if mode == 'clean':
                assert max(e[:2]) < 1e-4, e
            values.append(e)
    worst = np.max(values,axis=0)
    print('rectangular independent stencil',case_id,len(values),'worst source/field/phase:',worst)
    all_worst = np.maximum(all_worst, worst)
print('stress overall worst source/field/phase:',all_worst)
assert np.all(all_worst <= limits), (all_worst, limits)
PY
```
- **Container Exit Code**: `0` (`2790 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0 survey 140 worst source/field/phase: [0.00371191 0.00242758 0.00062272] omitted indices: [115 115   0]
public 0 pilot_calibration 12 worst source/field/phase: [0.00309376 0.00218461 0.00074135] omitted indices: [ 6  6 10]
rectangular independent stencil 0 12 worst source/field/phase: [7.07229226e-03 5.70179379e-03 4.44089210e-16]
public 1 survey 140 worst source/field/phase: [0.00639943 0.00415794 0.0002669 ] omitted indices: [27 27  0]
public 1 pilot_calibration 12 worst source/field/phase: [0.00587938 0.00371314 0.00058973] omitted indices: [0 0 0]
rectangular independent stencil 1 12 worst source/field/phase: [3.45639106e-03 2.24501946e-03 5.55111512e-17]
stress overall worst source/field/phase: [0.00707229 0.00570179 0.00074135]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=678dad5702cac3be (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_678dad57 (directory).
```

### Turn 4 — `submit_completed_task` (`12795 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification, leave-one-channel-out tests on both public cases, and independent rectangular-stencil tests covering phase wraps and concentrated impulses. Every output passed with margin, so no further tuning is warranted.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`24` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3576'}): Do NOT lock prematurely with `24` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Worst errors across public jackknife and synthetic stress tests were 0.707% for source amplitudes, 0.570% for field magnitude, and 0.000741 rad for phase, all within tolerance.`

### Turn 5 — `execute_in_remote_docker` (`33475 ms` LLM)
- **Rationale**: Single-channel omissions cannot detect reliance on a whole receiver region. Hold out each quadrant and the near-wall receivers on both public cases without retuning; a failed slice would motivate checking design conditioning and regional residuals before changing the estimator. Compare pilot weighting and plain least squares to isolate the contributions of calibration and impulse rejection.
- **Alternatives Weighed**: `Add rational source interactions, stochastic drift, or relay dynamics` (rejected: The specified forward operator is linear in the sources and matches the public reference field to roundoff; these terms have no role in that operator.); `Batch-center the survey` (rejected: There are no batches or class labels. Centering spatial receiver groups could remove real acoustic signal, so spatial groups are held out instead.)

```bash
cat > /tmp/audit_helmholtz.py <<'PY'
import pickle
import sys

import numpy as np

sys.path.insert(0, '/holdout_task')
import helmholtz_phase_inversion as h

with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

worst = np.zeros(3)
for ci, case in enumerate(cases):
    p, truth = case['problem'], case['gt']
    s = p['survey']
    ij = s['sensor_ij']
    shape = np.shape(p['n_sq'])
    i, j = ij.T
    depth = np.minimum.reduce([i, j, shape[0]-1-i, shape[1]-1-j])
    blocks = [(f'quadrant_{a}{b}', ((i >= shape[0]//2) == a) & ((j >= shape[1]//2) == b))
              for a in (0, 1) for b in (0, 1)]
    blocks.append(('near_wall', depth <= 2))
    responses = h._responses(p)
    design = responses[np.ravel_multi_index(ij.T, shape)]
    y = h._phasors(s) * np.exp(-1j*truth['global_phase_rad'])
    reference = design @ truth['source_amplitudes']
    residual = np.abs(y-reference)
    pilot = p['pilot_calibration']
    known = pilot['known_field_re'] + 1j*pilot['known_field_im']
    unit_phase = np.angle(np.sum(h._phasors(pilot) * known.conj() / (pilot['amplitude_obs'] * abs(known))))
    base_phase = h.invert_helmholtz_sources_and_phase(p)['global_phase_rad']
    circ = lambda x: abs(np.angle(np.exp(1j*(x-truth['global_phase_rad']))))
    ls = np.linalg.lstsq(np.vstack([design.real, design.imag]), np.r_[y.real, y.imag], rcond=None)[0]
    print('case', ci, 'plain least squares source error', np.linalg.norm(ls-truth['source_amplitudes'])/np.linalg.norm(truth['source_amplitudes']), 'pilot weighted/unit phase errors', circ(base_phase), circ(unit_phase))
    for name, held in blocks:
        reduced = dict(s)
        for key in ('sensor_ij', 'amplitude_obs', 'wrapped_phase_obs'):
            reduced[key] = s[key][~held]
        result = h.invert_helmholtz_sources_and_phase(dict(p, survey=reduced))
        e = np.array([np.linalg.norm(result[k]-truth[k])/np.linalg.norm(truth[k]) for k in ('source_amplitudes', 'predicted_field_abs')] + [circ(result['global_phase_rad'])])
        worst = np.maximum(worst, e)
        pred = design[held] @ result['source_amplitudes']
        print(name, 'held', held.sum(), 'source/field/phase', e, 'held median clean-field discrepancy', np.median(np.abs(pred-reference[held])), 'held median observed discrepancy', np.median(residual[held]))
print('worst spatial-block source/field/phase', worst)
assert np.all(worst <= [.030, .030, .020]), worst
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_helmholtz.py
```
- **Container Exit Code**: `0` (`541 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case 0 plain least squares source error 0.1718031349171875 pilot weighted/unit phase errors 0.0006227225657142733 0.0009396237447449884
quadrant_00 held 36 source/field/phase [0.00407919 0.00348248 0.00062272] held median clean-field discrepancy 0.00046129666485842513 held median observed discrepancy 0.0021681015753226697
quadrant_01 held 36 source/field/phase [0.00226743 0.00179293 0.00062272] held median clean-field discrepancy 0.00014283484283120505 held median observed discrepancy 0.0019011184754797032
quadrant_10 held 31 source/field/phase [0.00500962 0.00284762 0.00062272] held median clean-field discrepancy 0.000360785489916319 held median observed discrepancy 0.001983279394462597
quadrant_11 held 37 source/field/phase [0.00324043 0.00268328 0.00062272] held median clean-field discrepancy 0.0003024438169435599 held median observed discrepancy 0.0022424110991839533
near_wall held 55 source/field/phase [0.00420321 0.00279535 0.00062272] held median clean-field discrepancy 0.0002643711860993874 held median observed discrepancy 0.0020869426066019207
case 1 plain least squares source error 0.2727211285456098 pilot weighted/unit phase errors 0.00026689595102702407 0.000420964056925488
quadrant_00 held 42 source/field/phase [0.00591932 0.00390691 0.0002669 ] held median clean-field discrepancy 0.00031141636975500614 held median observed discrepancy 0.0018144918868116886
quadrant_01 held 36 source/field/phase [0.00726679 0.00450122 0.0002669 ] held median clean-field discrepancy 0.00048369508611311623 held median observed discrepancy 0.00188378669597535
quadrant_10 held 31 source/field/phase [0.00551686 0.00393005 0.0002669 ] held median clean-field discrepancy 0.000508981923982927 held median observed discrepancy 0.0017146209030076015
quadrant_11 held 31 source/field/phase [0.00742722 0.00435208 0.0002669 ] held median clean-field discrepancy 0.00037595349024985866 held median observed discrepancy 0.001997802075792147
near_wall held 46 source/field/phase [0.00711116 0.00494596 0.0002669 ] held median clean-field discrepancy 0.00034510448179122914 held median observed discrepancy 0.00167605626297543
worst spatial-block source/field/phase [0.00742722 0.00494596 0.00062272]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=678dad5702cac3be (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_678dad57 (directory).
```

### Turn 6 — `submit_completed_task` (`14773 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification, public single-receiver and spatial-block holdouts, and independent rectangular-grid tests with phase wrapping and concentrated impulses. The forward model matches the public reference to roundoff, and no tested slice warrants an estimator change.
- **Alternatives Weighed**: `Replace robust regression with ordinary least squares` (rejected: Even with the true phase supplied, ordinary least squares gave 17.2% and 27.3% source errors on the two public cases.); `Give all pilot phases equal weight` (rejected: Equal weighting increased phase error on both public cases.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`22` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3576'}): Do NOT lock prematurely with `22` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Worst observed errors were 0.743% for source amplitudes, 0.570% for field magnitude, and 0.000741 rad for phase. All met the specified tolerances.`

### Turn 7 — `execute_in_remote_docker` (`28873 ms` LLM)
- **Rationale**: Relocate the largest measured complex residuals to high-leverage and near-wall receivers, preserving the empirical residual distribution and the specified contamination rate. This tests whether the robust fit depends on where impulses occur. A tolerance failure would justify inspecting the robust initializer; a pass supports submitting the unchanged artifact.
- **Alternatives Weighed**: `Add nonlinear source terms or batch centering` (rejected: The specified source operator is linear and agrees with the public forward field to roundoff; the inputs contain neither technical batches nor class labels.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import helmholtz_phase_inversion as h

with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
worst = np.zeros(3)
for ci, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    s = p['survey']
    shape = p['n_sq'].shape
    design = h._responses(p)[np.ravel_multi_index(s['sensor_ij'].T, shape)]
    phase = gt['global_phase_rad']
    clean = np.exp(1j * phase) * (design @ gt['source_amplitudes'])
    residual = h._phasors(s) - clean
    n = len(residual)
    n_bad = round(0.07 * n)
    order = np.argsort(np.abs(residual))
    bad, other = order[-n_bad:], order[:-n_bad]
    q, _ = np.linalg.qr(np.vstack([design.real, design.imag]), mode='reduced')
    leverage = np.sum(q[:n]**2 + q[n:]**2, axis=1)
    depth = np.minimum.reduce([s['sensor_ij'][:,0], s['sensor_ij'][:,1], shape[0]-1-s['sensor_ij'][:,0], shape[1]-1-s['sensor_ij'][:,1]])
    for label, strength in [('high leverage', leverage), ('near wall', -depth + leverage)]:
        target = other[np.argsort(strength[other])[-n_bad:]]
        moved = residual.copy()
        moved[bad], moved[target] = residual[target], residual[bad]
        z = clean + moved
        ans = h.invert_helmholtz_sources_and_phase(dict(p, survey=dict(s, amplitude_obs=np.abs(z), wrapped_phase_obs=np.angle(z))))
        errors = [np.linalg.norm(ans[k]-gt[k])/np.linalg.norm(gt[k]) for k in ('source_amplitudes', 'predicted_field_abs')]
        errors.append(abs(np.angle(np.exp(1j*(ans['global_phase_rad']-phase)))))
        print(ci, label, 'source/field/phase', errors)
        worst = np.maximum(worst, errors)
print('worst measured-residual relocation errors:', worst)
assert np.all(worst <= [.03, .03, .02]), worst
PY
```
- **Container Exit Code**: `0` (`658 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 high leverage source/field/phase [0.0022985849745146805, 0.0021727570252260105, 0.0006227225657142733]
0 near wall source/field/phase [0.003721750936303649, 0.0025184458953642565, 0.0006227225657142733]
1 high leverage source/field/phase [0.004079424259488274, 0.002883699440347785, 0.00026689595102702407]
1 near wall source/field/phase [0.005230689141669143, 0.0033616084227209393, 0.00026689595102702407]
worst measured-residual relocation errors: [0.00523069 0.00336161 0.00062272]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=678dad5702cac3be (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_678dad57 (directory).
```

### Turn 8 — `submit_completed_task` (`14290 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification, public receiver and spatial holdouts, independent rectangular-grid and phase-wrap tests, and measured-impulse relocation. No tested slice exposed a failure that warrants another edit.
- **Alternatives Weighed**: `Change the robust initializer` (rejected: Moving measured impulses to high-leverage and near-wall receivers kept every output within tolerance and did not worsen the previous worst-case errors.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public holdouts and stress tests, worst errors were 0.743% for source amplitudes, 0.570% for field magnitude, and 0.000741 rad for phase. All met the specified tolerances.`

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
