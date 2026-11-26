# Dialectical Synthesis Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_2_Selective_LiveGPT` | `vr04__indefinite_helmholtz_wrapped_phase_inversion`

- **Official Strict Pass**: **`PASS (1.0)`** (`PASSED_ALL_CHECKS`)
- **Blind Public Synthesis Choice**: `antithesis` via `selective_router_confident_single_branch`
- **Synthesis Rationale**: Primary Antithesis branch satisfied public checks under policy; second branch skipped (`1.00u` nominal units).
- **Post-Lock Branch Verdicts**: Thesis (`Bare`) = `NOT_SPAWNED (0T)` | Antithesis (`Wormhole`) = `True` (8T)

## Selected Branch Trajectory Record
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
| **Total Tokens (Prompt / Cached / Out)** | `163,082` (`155,801` / `45,056` / `7,281`) |
| **Total Episode Wall Time (s)** | `239.91s` |

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

### Turn 1 — `execute_in_remote_docker` (`15100 ms` LLM)
- **Rationale**: Inspect the public schema to identify the exact input keys and available ground truth before implementing and testing the signed face stencil and robust receiver fit.
- **Alternatives Weighed**: `Fit wrapped phase and amplitude separately` (rejected: Complex reconstruction represents the observations without phase branch-cut discontinuities and keeps both components of each receiver spike together.)

```bash
cd /holdout_task && python3 -I - <<'PY'
import pickle
import numpy as np
import scipy

def describe(obj, depth=0):
    if isinstance(obj, dict) and depth < 5:
        return {k: describe(v, depth+1) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)) and depth < 5:
        return (type(obj).__name__,len(obj),[describe(v, depth+1) for v in obj[:1]])
    if isinstance(obj,np.ndarray):
        return (obj.shape,str(obj.dtype))
    return obj if np.isscalar(obj) else type(obj).__name__

with open('public_helmholtz_cases.pkl','rb') as f:
    cases=pickle.load(f)
print('numpy/scipy', np.__version__,scipy.__version__)
print(describe(cases))
PY
```
- **Container Exit Code**: `0` (`607 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
numpy/scipy 1.26.4 1.13.0
('list', 2, [{'problem': {'Lx': 1.0, 'Ly': 1.0, 'k0': 11.5, 'beta_imp': 0.85, 'n_sq': ((32, 32), 'float64'), 'basis': ((6, 32, 32), 'float64'), 'pilot_calibration': {'known_field_re': ((12,), 'float64'), 'known_field_im': ((12,), 'float64'), 'amplitude_obs': ((12,), 'float64'), 'wrapped_phase_obs': ((12,), 'float64')}, 'survey': {'sensor_ij': ((140, 2), 'int64'), 'amplitude_obs': ((140,), 'float64'), 'wrapped_phase_obs': ((140,), 'float64'), 'tikhonov_alpha': 1e-06}}, 'gt': {'source_amplitudes': ((6,), 'float64'), 'global_phase_rad': -0.8141122105046681, 'predicted_field_abs': ((32, 32), 'float64')}}])
```

### Turn 2 — `execute_in_remote_docker` (`45649 ms` LLM)
- **Rationale**: Implement the signed stencil and measure all three public targets. Compare the forward field at the known source coefficients separately to distinguish stencil errors from regression errors, and inspect residual quantiles to see whether impulse rejection is supported.
- **Alternatives Weighed**: `Ordinary ridge regression alone` (rejected: It gives impulsive receivers the same weight as clean receivers.)

```bash
cat > /holdout_task/helmholtz_phase_inversion.py <<'PY'
"""Finite-volume acoustic source inversion from polar receiver measurements."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _phasors(telemetry):
    return np.asarray(telemetry['amplitude_obs']) * np.exp(
        1j * np.asarray(telemetry['wrapped_phase_obs']))


def _helmholtz(problem):
    n_sq = np.asarray(problem['n_sq'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    k = problem['k0']
    area = hx * hy
    index = np.arange(nx * ny).reshape(nx, ny)
    left = np.r_[index[:-1, :].ravel(), index[:, :-1].ravel()]
    right = np.r_[index[1:, :].ravel(), index[:, 1:].ravel()]
    conductance = np.r_[np.full((nx - 1) * ny, hy / hx),
                        np.full(nx * (ny - 1), hx / hy)]
    boundary = np.zeros((nx, ny))
    boundary[0, :] += hy
    boundary[-1, :] += hy
    boundary[:, 0] += hx
    boundary[:, -1] += hx
    diagonal = (-area * k**2 * n_sq - 1j * k * problem['beta_imp'] * boundary).ravel()
    np.add.at(diagonal, left, conductance)
    np.add.at(diagonal, right, conductance)
    cells = index.ravel()
    matrix = coo_matrix((np.r_[diagonal, -conductance, -conductance],
                         (np.r_[cells, left, right], np.r_[cells, right, left])),
                        shape=(cells.size, cells.size)).tocsc()
    return matrix, area


def _robust_sources(design, observations, alpha):
    """Fit real coefficients with a shared weight per complex observation."""
    n, m = design.shape
    real_design = np.vstack((design.real, design.imag))
    real_obs = np.r_[observations.real, observations.imag]
    penalty = np.sqrt(alpha) * np.eye(m)

    def fit(weights):
        w = np.sqrt(np.tile(weights, 2))
        return np.linalg.lstsq(np.vstack((w[:, None] * real_design, penalty)),
                               np.r_[w * real_obs, np.zeros(m)], rcond=None)[0]

    coeff = fit(np.ones(n))
    floor = np.finfo(float).eps * max(np.max(np.abs(observations)), np.finfo(float).tiny)
    for _ in range(30):
        r = np.abs(observations - design @ coeff)
        cutoff = max(1.8 * np.median(r), floor)
        updated = fit(cutoff / np.maximum(r, cutoff))
        if np.linalg.norm(updated - coeff) <= 1e-10 * np.linalg.norm(updated):
            coeff = updated
            break
        coeff = updated

    # Gross impulses do not contribute to the final inlier regression.
    for _ in range(5):
        r = np.abs(observations - design @ coeff)
        inliers = r <= max(4.0 * np.median(r), floor)
        if inliers.sum() < m:
            break
        updated = fit(inliers)
        if np.array_equal(updated, coeff):
            break
        coeff = updated
    return coeff


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    pilot = problem['pilot_calibration']
    reference = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(reference, _phasors(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    matrix, area = _helmholtz(problem)
    basis = np.asarray(problem['basis'])
    m, nx, ny = basis.shape
    transfer = splu(matrix).solve(area * basis.reshape(m, nx * ny).T)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    design = transfer[i * ny + j]
    observations = _phasors(survey) * np.exp(-1j * phase)
    coeff = _robust_sources(design, observations, survey['tikhonov_alpha'])
    return {'source_amplitudes': coeff, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(transfer @ coeff).reshape(nx, ny)}
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
from scipy.sparse.linalg import splu
spec = importlib.util.spec_from_file_location('inversion', '/holdout_task/helmholtz_phase_inversion.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
for idx, c in enumerate(cases):
    p, gt = c['problem'], c['gt']
    ans = m.invert_helmholtz_sources_and_phase(p)
    def rel(x, y): return np.linalg.norm(x-y)/np.linalg.norm(y)
    print(idx, 'source', rel(ans['source_amplitudes'], gt['source_amplitudes']), 'phase', abs(np.angle(np.exp(1j*(ans['global_phase_rad']-gt['global_phase_rad'])))), 'field', rel(ans['predicted_field_abs'], gt['predicted_field_abs']))
    H, area = m._helmholtz(p)
    basis = p['basis']
    u = splu(H).solve(area * (gt['source_amplitudes'] @ basis.reshape(len(basis),-1)))
    i,j = p['survey']['sensor_ij'].T
    y = m._phasors(p['survey']) * np.exp(-1j*gt['global_phase_rad'])
    r = abs(y-u.reshape(basis.shape[1:])[i,j])
    print('forward field error', rel(abs(u).reshape(basis.shape[1:]), gt['predicted_field_abs']), 'obs amp median', np.median(abs(y)), 'residual quantiles', np.quantile(r,[0,.5,.85,.9,.92,.93,.94,.95,1]))
PY
```
- **Container Exit Code**: `0` (`538 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
0 source 0.003579006557262511 phase 0.0006227225657142733 field 0.002289175829731108
forward field error 8.49587838280063e-16 obs amp median 0.06941243613068682 residual quantiles [6.47881901e-06 2.04397750e-03 2.99061952e-03 3.74096331e-03
 4.01262154e-03 1.00151822e-01 3.61614744e-01 3.72538342e-01
 7.40687115e-01]
1 source 0.004715349089309009 phase 0.00026689595102702407 field 0.003201396421629315
forward field error 7.965998780474151e-16 obs amp median 0.08930816707830316 residual quantiles [2.09377251e-04 1.80389227e-03 3.47499659e-03 4.26435661e-03
 4.83633554e-03 9.99000576e-02 3.74277075e-01 4.09010034e-01
 6.86029873e-01]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=dcbbb1833337b5ee (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_dcbbb183 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`42909 ms` LLM)
- **Rationale**: Test the unchanged candidate in isolation. A public receiver-deletion failure would motivate revising the robust fit; an independently generated rectangular or branch-cut failure would localize a stencil, calibration, or high-leverage contamination issue.
- **Alternatives Weighed**: `Submit on the two nominal public cases` (rejected: Nominal results do not detect reliance on an individual receiver, swapped face lengths on rectangular grids, or failure near the phase branch cut.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import copy
import importlib.util
import pickle
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

spec = importlib.util.spec_from_file_location('inversion', '/holdout_task/helmholtz_phase_inversion.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

def errors(p, truth):
    a = solver.invert_helmholtz_sources_and_phase(p)
    assert set(a) == {'source_amplitudes', 'global_phase_rad', 'predicted_field_abs'}
    for k in ('source_amplitudes', 'predicted_field_abs'):
        assert a[k].shape == truth[k].shape and np.issubdtype(a[k].dtype, np.floating)
        assert np.isfinite(a[k]).all()
    assert isinstance(a['global_phase_rad'], float) and -np.pi < a['global_phase_rad'] <= np.pi
    def rel(k):
        return np.linalg.norm(a[k] - truth[k]) / np.linalg.norm(truth[k])
    return np.array([rel('source_amplitudes'), abs(np.angle(np.exp(1j * (a['global_phase_rad'] - truth['global_phase_rad'])))), rel('predicted_field_abs')])

def report(label, values):
    values = np.asarray(values)
    worst = values.max(axis=0)
    print(label, 'n=', len(values), 'worst (source, phase, field)=', worst, 'argmax=', values.argmax(axis=0), flush=True)
    return bool(np.all(worst <= [0.030, 0.020, 0.030]))

passes = [report('public', [errors(c['problem'], c['gt']) for c in cases])]
for idx, c in enumerate(cases):
    p = c['problem']
    survey = p['survey']
    n = len(survey['sensor_ij'])
    vals = []
    for j in range(n):
        keep = np.arange(n) != j
        reduced = {k: v[keep] if isinstance(v, np.ndarray) else v for k, v in survey.items()}
        vals.append(errors(dict(p, survey=reduced), c['gt']))
    passes.append(report(f'public {idx} leave-one-receiver-out', vals))

def axis(n, g):
    d = np.full(n, 2.0)
    d[[0, -1]] = 1
    sides = np.zeros(n)
    sides[[0, -1]] = 1
    return g * diags([-np.ones(n-1), d, -np.ones(n-1)], [-1, 0, 1]), sides

vals = []
for case_id, c in enumerate(cases):
    for Lx, Ly, k, beta in [(1.17, 0.86, 11.5, 0.85), (0.89, 1.15, 12.1, 0.72)]:
        p = copy.deepcopy(c['problem'])
        p.update(Lx=Lx, Ly=Ly, k0=k, beta_imp=beta)
        m, nx, ny = p['basis'].shape
        hx, hy = Lx / nx, Ly / ny
        tx, sx = axis(nx, hy / hx)
        ty, sy = axis(ny, hx / hy)
        H = kron(tx, eye(ny)) + kron(eye(nx), ty) + diags((-hx*hy*k*k*p['n_sq'] - 1j*k*beta*(hy*sx[:,None]+hx*sy[None,:])).ravel())
        T = splu(H.tocsc()).solve(hx*hy*p['basis'].reshape(m, -1).T)
        coeff = c['gt']['source_amplitudes']
        u = (T @ coeff).reshape(nx, ny)
        i, j = p['survey']['sensor_ij'].T
        clean = u[i,j]
        design = T[i*ny+j]
        q, _ = np.linalg.qr(np.vstack((design.real, design.imag)))
        ls = np.sum(q.reshape(2,len(i),m)**2, axis=(0,2))
        contaminated = np.argsort(ls)[-round(.07*len(i)):]
        pilot = p['pilot_calibration']
        ref = pilot['known_field_re'] + 1j*pilot['known_field_im']
        for seed in range(10):
            rng = np.random.default_rng(seed)
            phase = (-1)**seed * (np.pi - 1e-6)
            rot = np.exp(1j*phase)
            y = clean + .00155*(rng.normal(size=len(i)) + 1j*rng.normal(size=len(i)))
            y[contaminated] += rng.uniform(.1, .7, len(contaminated))*np.exp(1j*rng.uniform(-np.pi,np.pi,len(contaminated)))
            p['survey'].update(amplitude_obs=abs(y), wrapped_phase_obs=np.angle(y*rot))
            pilot.update(amplitude_obs=abs(ref), wrapped_phase_obs=np.angle(ref*rot))
            vals.append(errors(p, dict(source_amplitudes=coeff, global_phase_rad=phase, predicted_field_abs=abs(u))))
passes.append(report('rectangular / high-leverage spikes / phase branch', vals))
assert all(passes), 'A worst-slice threshold failed'
PY
```
- **Container Exit Code**: `0` (`1982 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public n= 2 worst (source, phase, field)= [0.00471535 0.00062272 0.0032014 ] argmax= [1 0 1]
public 0 leave-one-receiver-out n= 140 worst (source, phase, field)= [0.00425068 0.00062272 0.00256123] argmax= [115   0   3]
public 1 leave-one-receiver-out n= 140 worst (source, phase, field)= [0.00542723 0.0002669  0.00359968] argmax= [120   0  27]
rectangular / high-leverage spikes / phase branch n= 40 worst (source, phase, field)= [9.31233081e-03 4.44089210e-16 6.17719491e-03] argmax= [23  0 16]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=dcbbb1833337b5ee (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_dcbbb183 (directory).
```

### Turn 4 — `submit_completed_task` (`14048 ms` LLM)
- **Rationale**: The saved artifact passed clean-process checks on both public cases, all 280 leave-one-receiver-out fits, and 40 independent rectangular-grid stress cases with phase wrapping and high-leverage outliers. No tested slice requires a change.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3605'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public cases, receiver-deletion tests, and synthetic stress tests, the worst errors were 0.932% for source amplitudes, 0.000623 rad for phase, and 0.618% for field magnitude—all below the required tolerances.`

### Turn 5 — `execute_in_remote_docker` (`33359 ms` LLM)
- **Rationale**: Compare against plain ridge to assess the value of spike rejection. Delete receiver quadrants and individual pilots separately to detect spatial dependence or calibration fragility. Test the constant-field identity and transpose symmetry to detect an assembly defect before changing the estimator.
- **Alternatives Weighed**: `Batch centering or relay-mode tests` (rejected: The measured problem has no batch labels, classification outcome, or relay state. Spatial receiver coverage and dependence on individual pilots are testable on the provided data.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.sparse.linalg import splu

spec = importlib.util.spec_from_file_location('inversion', '/holdout_task/helmholtz_phase_inversion.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

limits = np.array([.03, .02, .03])
worst = np.zeros(3)
for idx, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    H, cell_area = solver._helmholtz(p)
    m, nx, ny = p['basis'].shape
    T = splu(H).solve(cell_area * p['basis'].reshape(m, -1).T)
    survey, pilot = p['survey'], p['pilot_calibration']
    i, j = survey['sensor_ij'].T
    phase = solver.invert_helmholtz_sources_and_phase(p)['global_phase_rad']
    A = T[i * ny + j]
    y = solver._phasors(survey) * np.exp(-1j * phase)
    X = np.r_[A.real, A.imag]
    alpha = survey['tikhonov_alpha']
    ridge = np.linalg.solve(X.T @ X + alpha * np.eye(m), X.T @ np.r_[y.real, y.imag])

    def errors(a, ph, field):
        return np.array([np.linalg.norm(a - gt['source_amplitudes']) / np.linalg.norm(gt['source_amplitudes']),
                         abs(np.angle(np.exp(1j * (ph - gt['global_phase_rad'])))),
                         np.linalg.norm(field - gt['predicted_field_abs']) / np.linalg.norm(gt['predicted_field_abs'])])

    def assess(problem):
        ans = solver.invert_helmholtz_sources_and_phase(problem)
        return errors(ans['source_amplitudes'], ans['global_phase_rad'], ans['predicted_field_abs'])

    quad = 2 * (i >= nx / 2) + (j >= ny / 2)
    pilot_errors = [assess(dict(p, pilot_calibration={k: np.delete(v, n) for k, v in pilot.items()})) for n in range(len(pilot['amplitude_obs']))]
    quad_errors = []
    for q in range(4):
        keep = quad != q
        reduced = {k: v[keep] if isinstance(v, np.ndarray) else v for k, v in survey.items()}
        quad_errors.append(assess(dict(p, survey=reduced)))
    qmax = np.max(quad_errors, axis=0)
    pmax = np.max(pilot_errors, axis=0)
    worst = np.maximum(worst, np.maximum(qmax, pmax))
    print(f'case {idx}: plain ridge errors={errors(ridge, phase, np.abs(T @ ridge).reshape(nx, ny))}; quadrant-deletion worst={qmax}; pilot-deletion worst={pmax}')
    constant_action = (-cell_area * p['k0']**2 * p['n_sq']).astype(complex)
    hx, hy = p['Lx'] / nx, p['Ly'] / ny
    impedance = -1j * p['k0'] * p['beta_imp']
    constant_action[[0, -1], :] += hy * impedance
    constant_action[:, [0, -1]] += hx * impedance
    np.testing.assert_allclose(H @ np.ones(nx * ny), constant_action.ravel(), atol=1e-13, rtol=0)
    assert (H - H.T).nnz == 0
print('worst real-measurement audit errors:', worst, '; tolerance fractions:', worst / limits)
assert np.all(worst <= limits)
PY
```
- **Container Exit Code**: `0` (`727 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case 0: plain ridge errors=[0.17182283 0.00062272 0.18377961]; quadrant-deletion worst=[0.00555353 0.00062272 0.00344887]; pilot-deletion worst=[0.00361809 0.00074135 0.00230511]
case 1: plain ridge errors=[2.72670234e-01 2.66895951e-04 1.60987316e-01]; quadrant-deletion worst=[0.00628773 0.0002669  0.0039954 ]; pilot-deletion worst=[0.00473207 0.00058973 0.00321022]
worst real-measurement audit errors: [0.00628773 0.00074135 0.0039954 ] ; tolerance fractions: [0.20959105 0.03706745 0.13318016]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=dcbbb1833337b5ee (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_dcbbb183 (directory).
```

### Turn 6 — `submit_completed_task` (`14014 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated execution, stencil conservation and symmetry checks, receiver and pilot deletion tests, and rectangular-grid tests with phase wrapping and high-leverage spikes. No tested slice warrants a change.
- **Alternatives Weighed**: `Replace robust regression with plain ridge` (rejected: Plain ridge reached 27.3% source error on the public measurements, exceeding the 3% tolerance.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3605'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public cases and stress tests, the worst errors were 0.932% for source amplitudes, 0.000742 rad for phase, and 0.618% for field magnitude—all below the required tolerances.`

### Turn 7 — `execute_in_remote_docker` (`29754 ms` LLM)
- **Rationale**: Earlier checks removed pilots and spatial receiver groups separately. Cross those perturbations on the real measurements to detect whether calibration error is amplified when receiver coverage is reduced. If the crossed worst case exceeds tolerance, its failing output will identify which part of the estimator to investigate.
- **Alternatives Weighed**: `Apply batch centering or add nonlinear stencil terms` (rejected: The public inputs have no batch or class variable, and the independently checked finite-volume equation specifies a linear stencil with signed face differences.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('inversion', '/holdout_task/helmholtz_phase_inversion.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with open('public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

limits = np.array([.03, .02, .03])
worst = np.zeros(3)
labels = [None] * 3
count = 0
for case_id, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    s, pilot = p['survey'], p['pilot_calibration']
    nx, ny = p['n_sq'].shape
    i, j = s['sensor_ij'].T
    # Four half-cavity masks test less spatial diversity and higher effective
    # spike concentration without adding or altering the public measurements.
    for half_id, keep in enumerate((i < nx/2, i >= nx/2, j < ny/2, j >= ny/2)):
        survey = {k: v[keep] if isinstance(v, np.ndarray) else v for k, v in s.items()}
        for leave in range(len(pilot['amplitude_obs'])):
            calibration = {k: np.delete(v, leave) for k, v in pilot.items()}
            ans = module.invert_helmholtz_sources_and_phase(dict(p, survey=survey, pilot_calibration=calibration))
            a, ph, field = (ans[k] for k in ('source_amplitudes', 'global_phase_rad', 'predicted_field_abs'))
            assert a.shape == gt['source_amplitudes'].shape and field.shape == (nx, ny)
            assert -np.pi < ph <= np.pi and np.isfinite(a).all() and np.isfinite(field).all()
            err = np.array([np.linalg.norm(a-gt['source_amplitudes'])/np.linalg.norm(gt['source_amplitudes']), abs(np.angle(np.exp(1j*(ph-gt['global_phase_rad'])))), np.linalg.norm(field-gt['predicted_field_abs'])/np.linalg.norm(gt['predicted_field_abs'])])
            for idx in np.flatnonzero(err > worst):
                labels[idx] = (case_id, half_id, leave)
            worst = np.maximum(worst, err)
            count += 1
print('Crossed public slices:', count, '; worst source, phase, field:', worst, '; tolerance fractions:', worst/limits, '; worst slices:', labels)
assert np.all(worst <= limits)
PY
```
- **Container Exit Code**: `0` (`1144 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Crossed public slices: 96 ; worst source, phase, field: [0.00909404 0.00074135 0.00512673] ; tolerance fractions: [0.30313467 0.03706745 0.17089099] ; worst slices: [(1, 2, 0), (0, 0, 10), (1, 2, 0)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=dcbbb1833337b5ee (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_dcbbb183 (directory).
```

### Turn 8 — `submit_completed_task` (`15157 ms` LLM)
- **Rationale**: The saved artifact passed isolated execution, independent stencil checks, and stress tests for reduced receiver coverage, phase wrapping, and high-leverage outliers. The latest crossed test found no failure that warrants a change.
- **Alternatives Weighed**: `Revise the estimator after the crossed public stress test` (rejected: All 96 half-cavity and pilot-deletion slices passed; the largest error used just 30.3% of its tolerance.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. The worst observed errors were 0.932% for source amplitudes, 0.000742 rad for global phase, and 0.618% for field magnitude. All three met their required tolerances.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-M90a-VoluntaryReuse-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/helmholtz_phase_inversion.py": "FILE:3605"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "all_valid": true,
    "max_source_rel_l2": 0.004235,
    "max_phase_abs_err_rad": 0.000852,
    "max_field_abs_rel_l2": 0.003,
    "elapsed_sec": 0.1437
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"all_valid\": true, \"max_source_rel_l2\": 0.004235, \"max_phase_abs_err_rad\": 0.000852, \"max_field_abs_rel_l2\": 0.003, \"elapsed_sec\": 0.1437}\n__HEGEL_VERIFIER_AUTH_c376278a468037ecde718d67898a0166__:{\"strict_pass\": true, \"all_valid\": true, \"max_source_rel_l2\": 0.004235, \"max_phase_abs_err_rad\": 0.000852, \"max_field_abs_rel_l2\": 0.003, \"elapsed_sec\": 0.1437}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5727.48,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
