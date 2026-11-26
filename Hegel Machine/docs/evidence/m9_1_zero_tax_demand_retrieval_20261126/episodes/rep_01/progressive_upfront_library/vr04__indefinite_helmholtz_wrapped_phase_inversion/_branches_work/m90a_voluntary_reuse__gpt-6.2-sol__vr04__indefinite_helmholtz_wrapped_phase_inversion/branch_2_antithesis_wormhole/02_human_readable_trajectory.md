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
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `4` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `201,761` (`194,596` / `56,704` / `7,165`) |
| **Total Episode Wall Time (s)** | `238.98s` |

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

### Turn 1 — `execute_in_remote_docker` (`16811 ms` LLM)
- **Rationale**: Inspect the public fixture keys, array layout, and reference availability first; these determine the implementation interface and which errors can be measured directly.
- **Alternatives Weighed**: `Reuse the Darcy SPD solver` (rejected: The outgoing impedance and negative wave term violate its symmetry and positive-definiteness assumptions for CG.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle
import numpy as np

def show(x, depth=0):
    p = '  '*depth
    if isinstance(x, dict):
        for k,v in x.items():
            print(f'{p}{k}:', end=' ')
            if isinstance(v,(dict,list,tuple)):
                print(); show(v,depth+1)
            else: show(v,0)
    elif isinstance(x,(tuple,list)):
        print(f'{p}{type(x).__name__} length {len(x)}')
        if x: show(x[0],depth+1)
    elif isinstance(x,np.ndarray):
        print(x.shape,x.dtype,'sample',x.ravel()[:4])
    else: print(repr(x))
with open('public_helmholtz_cases.pkl','rb') as f: cases=pickle.load(f)
show(cases)
PY
```
- **Container Exit Code**: `0` (`642 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
list length 2
  problem: 
    Lx: 1.0
    Ly: 1.0
    k0: 11.5
    beta_imp: 0.85
    n_sq: (32, 32) float64 sample [0.80163524 0.80204006 0.80263744 0.80348888]
    basis: (6, 32, 32) float64 sample [0.000461   0.00215903 0.00802474 0.02367123]
    pilot_calibration: 
      known_field_re: (12,) float64 sample [0.70152797 0.73476006 1.90297056 1.08085754]
      known_field_im: (12,) float64 sample [-0.36152527 -0.65790536 -0.37345604 -0.08335344]
      amplitude_obs: (12,) float64 sample [0.78821059 0.98674527 1.93760065 1.08525412]
      wrapped_phase_obs: (12,) float64 sample [-1.28850398 -1.54397385 -1.0075838  -0.89001087]
    survey: 
      sensor_ij: (140, 2) int64 sample [ 3 29 29 20]
      amplitude_obs: (140,) float64 sample [0.04516173 0.06261364 0.04173044 0.04717036]
      wrapped_phase_obs: (140,) float64 sample [-2.05372953 -0.20593495 -1.60097692  0.96523401]
      tikhonov_alpha: 1e-06
  gt: 
    source_amplitudes: (6,) float64 sample [ 1.35       -0.57020875 -0.72756369 -0.18032955]
    global_phase_rad: -0.8141122105046681
    predicted_field_abs: (32, 32) float64 sample [0.01746595 0.01663818 0.01827211 0.02375379]
```

### Turn 2 — `execute_in_remote_docker` (`48318 ms` LLM)
- **Rationale**: The public ground-truth field independently checks the stencil, while pilot error and the robust-versus-unweighted source errors show whether to adjust calibration or receiver weighting.
- **Alternatives Weighed**: `Fit unwrapped phase and amplitude separately` (rejected: Complex reconstruction removes the arbitrary phase branch and preserves the linear forward response in the signed real sources.)

```bash
cat > /holdout_task/helmholtz_phase_inversion.py <<'PY'
"""Finite-volume Helmholtz inversion from pilot-referenced polar telemetry."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _source_fields(problem):
    """Compute one complex pressure field per source basis function."""
    n_sq = np.asarray(problem['n_sq'])
    basis = np.asarray(problem['basis'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k = problem['k0']
    grid = np.arange(nx * ny).reshape(nx, ny)
    left = np.concatenate((grid[:-1, :].ravel(), grid[:, :-1].ravel()))
    right = np.concatenate((grid[1:, :].ravel(), grid[:, 1:].ravel()))
    conductance = np.repeat([hy / hx, hx / hy], [(nx - 1) * ny, nx * (ny - 1)])

    diagonal = np.array(-area * k**2 * n_sq, dtype=np.complex128, order='C')
    impedance = -1j * k * problem['beta_imp']
    diagonal[0, :] += hy * impedance
    diagonal[-1, :] += hy * impedance
    diagonal[:, 0] += hx * impedance
    diagonal[:, -1] += hx * impedance
    diagonal = diagonal.ravel()
    np.add.at(diagonal, left, conductance)
    np.add.at(diagonal, right, conductance)
    ids = grid.ravel()
    matrix = coo_matrix(
        (np.concatenate((diagonal, -conductance, -conductance)),
         (np.concatenate((ids, left, right)), np.concatenate((ids, right, left)))),
        shape=(ids.size, ids.size),
    ).tocsc()
    rhs = np.asarray(area * basis.reshape(basis.shape[0], -1).T, dtype=np.complex128)
    return splu(matrix).solve(rhs).reshape(nx, ny, basis.shape[0])


def _complex_observations(channel):
    return np.asarray(channel['amplitude_obs']) * np.exp(1j * np.asarray(channel['wrapped_phase_obs']))


def _robust_sources(design, data, alpha):
    """Fit real coefficients, weighting each complex receiver as one observation."""
    n, m = design.shape
    stacked = np.concatenate((design.real, design.imag), axis=0)
    obs = np.concatenate((data.real, data.imag))
    penalty = np.sqrt(alpha) * np.eye(m)
    rhs = np.concatenate((obs, np.zeros(m)))
    floor = np.finfo(float).eps * np.max(np.abs(data))
    weights = np.ones(n)
    a = np.zeros(m)
    for i in range(30):
        scale_rows = np.sqrt(np.tile(weights, 2))
        lhs = np.vstack((stacked * scale_rows[:, None], penalty))
        new_a = np.linalg.lstsq(lhs, rhs * np.concatenate((scale_rows, np.ones(m))), rcond=None)[0]
        if i >= 6 and np.linalg.norm(new_a - a) <= 1e-10 * (1 + np.linalg.norm(new_a)):
            return new_a
        a = new_a
        residual = np.abs(design @ a - data)
        sigma = max(np.median(residual) / np.sqrt(2 * np.log(2)), floor)
        if sigma == 0:
            return a
        if i < 5:
            # Bounded influence before rejecting impulses altogether.
            weights = 1 / np.maximum(1, residual / (1.5 * sigma))
        else:
            weights = np.maximum(0, 1 - (residual / (5 * sigma))**2)**2
    return a


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Recover signed real sources, global phase, and pressure magnitude."""
    pilot = problem['pilot_calibration']
    reference = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(reference, _complex_observations(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    survey = problem['survey']
    fields = _source_fields(problem)
    i, j = np.asarray(survey['sensor_ij']).T
    data = _complex_observations(survey) * np.exp(-1j * phase)
    amplitudes = _robust_sources(fields[i, j], data, survey['tikhonov_alpha'])
    return {'source_amplitudes': amplitudes, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(fields @ amplitudes)}
PY
cat > /tmp/helmholtz_public.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('helmholtz', '/holdout_task/helmholtz_phase_inversion.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
for case in cases:
    p, gt = case['problem'], case['gt']
    out = m.invert_helmholtz_sources_and_phase(p)
    print('geometry', p['Lx'], p['Ly'], p['k0'], p['beta_imp'])
    for key, expected in gt.items():
        error = np.angle(np.exp(1j*(out[key]-expected))) if key == 'global_phase_rad' else np.linalg.norm(out[key]-expected)/np.linalg.norm(expected)
        print(key, abs(error))
    fields = m._source_fields(p)
    survey = p['survey']
    i,j = survey['sensor_ij'].T
    design = fields[i,j]
    true = design @ gt['source_amplitudes']
    obs = m._complex_observations(survey) * np.exp(-1j*gt['global_phase_rad'])
    a_ls = np.linalg.lstsq(np.r_[design.real, design.imag], np.r_[obs.real, obs.imag], rcond=None)[0]
    print('unweighted source error', np.linalg.norm(a_ls-gt['source_amplitudes'])/np.linalg.norm(gt['source_amplitudes']))
    print('signal and true residual quantiles', np.quantile(abs(true), [.1,.5,.9]), np.quantile(abs(obs-true), [.1,.5,.9,.93,.95,1]))
    print('condition', np.linalg.cond(np.r_[design.real, design.imag]), 'true-field forward error', np.linalg.norm(abs(fields @ gt['source_amplitudes']) - gt['predicted_field_abs'])/np.linalg.norm(gt['predicted_field_abs']))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/helmholtz_public.py
```
- **Container Exit Code**: `0` (`472 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
geometry 1.0 1.0 11.5 0.85
source_amplitudes 0.003112524911627291
global_phase_rad 0.0006227225657142733
predicted_field_abs 0.00217793836904194
unweighted source error 0.17180313491718766
signal and true residual quantiles [0.02113297 0.06468471 0.12660188] [0.00076065 0.00204398 0.00374096 0.10015182 0.37253834 0.74068712]
condition 2.9618123405365124 true-field forward error 1.1272979772244929e-15
geometry 1.0 1.0 11.5 0.85
source_amplitudes 0.005678418539034241
global_phase_rad 0.00026689595102702407
predicted_field_abs 0.0036352642498357606
unweighted source error 0.2727211285456088
signal and true residual quantiles [0.04551291 0.08598488 0.16018043] [0.00075374 0.00180389 0.00426436 0.09990006 0.40901003 0.68602987]
condition 3.1136094055719403 true-field forward error 9.93191916978576e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6aaea370bbf4af05 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6aaea370 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`42170 ms` LLM)
- **Rationale**: Receiver deletion tests dependence on individual public sensors. An independent rectangular stencil and branch-cut/impulse stresses test face scaling, circular calibration, and robustness; a failure would identify which component to revise.
- **Alternatives Weighed**: `Retune robust cutoffs on the two public cases` (rejected: Both cases are comfortably within tolerance; test receiver influence and independent geometry before considering changes.)
- **ATMS NOGOOD DECLARED**: Refuted `After exact forward and pilot calibration, all survey channels can be treated as ordinary least-squares observations.` | Witness: `Unweighted source errors were 17.18% and 27.27% on the two public cases; robust errors were 0.31% and 0.57%, and the independent ground-truth forward errors were about 1e-15.` | Blanket Diagnosis: `Sparse receiver impulses contaminate paired real and imaginary residuals, while neither the stencil nor pilot calibration explains the source error.` | Sublated Pivot: `Retain the exact stencil and pilot calibration, but weight and reject impulses by complex receiver residual magnitude.`

```bash
cat > /tmp/helmholtz_stress.py <<'PY'
import importlib.util
import pickle
from copy import deepcopy

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

spec = importlib.util.spec_from_file_location('helmholtz', '/holdout_task/helmholtz_phase_inversion.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

def errors(p, a, phi, field):
    out = solver.invert_helmholtz_sources_and_phase(p)
    assert set(out) == {'source_amplitudes', 'global_phase_rad', 'predicted_field_abs'}
    assert out['source_amplitudes'].shape == a.shape and out['source_amplitudes'].dtype.kind == 'f'
    assert out['predicted_field_abs'].shape == field.shape and out['predicted_field_abs'].dtype.kind == 'f'
    assert isinstance(out['global_phase_rad'], float) and -np.pi < out['global_phase_rad'] <= np.pi
    return np.array([np.linalg.norm(out['source_amplitudes']-a)/np.linalg.norm(a),
                     abs(np.angle(np.exp(1j*(out['global_phase_rad']-phi)))),
                     np.linalg.norm(out['predicted_field_abs']-abs(field))/np.linalg.norm(field)])

def reference(p, a):
    nx, ny = p['n_sq'].shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    k = p['k0']
    op = lil_matrix((nx*ny, nx*ny), dtype=complex)
    for i,j in np.ndindex(nx,ny):
        t = i*ny+j
        op[t,t] = -hx*hy*k*k*p['n_sq'][i,j]
        for ii,jj,length,distance in [(i-1,j,hy,hx),(i+1,j,hy,hx),(i,j-1,hx,hy),(i,j+1,hx,hy)]:
            if 0 <= ii < nx and 0 <= jj < ny:
                op[t,t] += length/distance
                op[t,ii*ny+jj] -= length/distance
            else:
                op[t,t] -= 1j*k*p['beta_imp']*length
    q = np.einsum('m,mij->ij',a,p['basis'])
    return spsolve(op.tocsc(), hx*hy*q.ravel()).reshape(nx,ny)

rng = np.random.default_rng(2047)
loo, synthetic = [], []
for case in cases:
    p, gt = case['problem'], case['gt']
    a, phi, field = (gt[k] for k in ('source_amplitudes','global_phase_rad','predicted_field_abs'))
    for dropped in range(len(p['survey']['sensor_ij'])):
        q = dict(p, survey={k: np.delete(v, dropped, axis=0) if k != 'tikhonov_alpha' else v for k,v in p['survey'].items()})
        loo.append(errors(q, a, phi, field))
    for lx, ly, k0, beta in [(0.82,1.20,10.9,0.65),(1.21,0.78,12.7,1.02)]:
        q = deepcopy(p)
        q.update(Lx=lx, Ly=ly, k0=k0, beta_imp=beta)
        u = reference(q,a)
        np.testing.assert_allclose(solver._source_fields(q) @ a, u, rtol=2e-11, atol=1e-12)
        pilot, survey = q['pilot_calibration'],q['survey']
        ref = pilot['known_field_re']+1j*pilot['known_field_im']
        i,j = survey['sensor_ij'].T
        for phi in [-np.pi+1e-5, np.pi-1e-5]:
            zpilot = np.exp(1j*phi)*ref + .0015*(rng.normal(size=ref.size)+1j*rng.normal(size=ref.size))
            pilot.update(amplitude_obs=abs(zpilot),wrapped_phase_obs=np.angle(zpilot))
            for repeat in range(12):
                z = np.exp(1j*phi)*u[i,j] + .0015*(rng.normal(size=i.size)+1j*rng.normal(size=i.size))
                nbad = round(.07*len(z))
                bad = rng.choice(len(z),nbad,replace=False) if repeat%2 else np.argsort(abs(u[i,j]))[-nbad:]
                z[bad] += rng.uniform(.15,.75,nbad)*np.exp(1j*rng.uniform(-np.pi,np.pi,nbad))
                survey.update(amplitude_obs=abs(z),wrapped_phase_obs=np.angle(z))
                synthetic.append(errors(q,a,phi,u))
for label, vals in [('public leave-one-receiver-out',loo),('rectangular / branch-cut / random and high-signal impulses',synthetic)]:
    worst = np.max(vals,axis=0)
    print(label, len(vals), 'worst source / phase / field:',worst)
    assert np.all(worst <= [.03,.02,.03]), (label,worst)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/helmholtz_stress.py
```
- **Container Exit Code**: `0` (`2853 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public leave-one-receiver-out 280 worst source / phase / field: [0.00626013 0.00062272 0.00409203]
rectangular / branch-cut / random and high-signal impulses 96 worst source / phase / field: [0.00704411 0.0010411  0.00475674]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6aaea370bbf4af05 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6aaea370 (directory).
```

### Turn 4 — `submit_completed_task` (`14620 ms` LLM)
- **Rationale**: Clean-process verification of the unchanged, receipt-bound artifact passed the output contract and all three accuracy thresholds. Public receiver deletions and synthetic rectangular-grid, phase-wrap, and impulse tests support the robust fit over unweighted least squares.
- **Alternatives Weighed**: `Retune the receiver weights` (rejected: The verified implementation passed every tested slice with substantial margin; no measured bottleneck justified retuning.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3716'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across 280 public leave-one-receiver-out runs and 96 synthetic stress cases, worst errors were 0.705% for source amplitudes, 0.00105 rad for phase, and 0.476% for field magnitude. All were within tolerance.`

### Turn 5 — `execute_in_remote_docker` (`32855 ms` LLM)
- **Rationale**: Public truth already confirms the full stencil to about 1e-15. Pilot deletion tests whether one reference dominates calibration, and quadrant deletion tests whether the robust fit relies on one receiver region. A failing slice would distinguish pilot leverage from poor source observability or spatially concentrated contamination.
- **Alternatives Weighed**: `Add rational terms, drift operators, relay modes, or batch centering` (rejected: Polar-to-complex conversion and the finite-volume face differences already give the specified linear source response; there are no drift states, relay modes, batch labels, or class outcomes in this problem.)

```bash
cat > /tmp/helmholtz_audit.py <<'PY'
import runpy
import numpy as np

ns = runpy.run_path('/tmp/helmholtz_public.py')
m, cases = ns['m'], ns['cases']
keys = ('source_amplitudes', 'global_phase_rad', 'predicted_field_abs')
limits = np.array([.03, .02, .03])
worst_pilot, worst_survey = [], []

def errors(p, gt):
    out = m.invert_helmholtz_sources_and_phase(p)
    return np.array([np.linalg.norm(out[keys[0]]-gt[keys[0]])/np.linalg.norm(gt[keys[0]]), abs(np.angle(np.exp(1j*(out[keys[1]]-gt[keys[1]])))), np.linalg.norm(out[keys[2]]-gt[keys[2]])/np.linalg.norm(gt[keys[2]])])

def subset(p, channel, keep):
    return dict(p, **{channel: {k: np.asarray(v)[keep] if np.ndim(v) else v for k,v in p[channel].items()}})

for idx, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    pilot, survey = p['pilot_calibration'], p['survey']
    ref = pilot['known_field_re'] + 1j*pilot['known_field_im']
    for rank in np.argsort(abs(ref)):
        worst_pilot.append(errors(subset(p, 'pilot_calibration', np.arange(ref.size) != rank), gt))
    fields = m._source_fields(p)
    i,j = survey['sensor_ij'].T
    d = fields[i,j]
    nx,ny = fields.shape[:2]
    quadrant = 2*(i >= nx/2)+(j >= ny/2)
    clean = d @ gt[keys[0]]
    obs = m._complex_observations(survey)*np.exp(-1j*gt[keys[1]])
    r = obs-clean
    print('case', idx, 'quadrant diagnostics: n, median absolute residual, fraction above 0.03, retained design condition')
    for group in np.unique(quadrant):
        take = quadrant == group
        retained = d[~take]
        print(int(group), int(take.sum()), round(float(np.median(abs(r[take]))),6), round(float(np.mean(abs(r[take])>.03)),3), round(float(np.linalg.cond(np.r_[retained.real, retained.imag])),3))
        worst_survey.append(errors(subset(p, 'survey', ~take), gt))
for label, rows in [('pilot jackknife',worst_pilot), ('survey quadrant deletion',worst_survey)]:
    worst = np.max(rows,axis=0)
    print(label, len(rows), 'worst source / phase / field:', worst, 'max fraction of tolerance:', max(worst/limits))
assert all(np.all(np.max(rows,axis=0) <= limits) for rows in (worst_pilot,worst_survey))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/helmholtz_audit.py
```
- **Container Exit Code**: `0` (`698 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
geometry 1.0 1.0 11.5 0.85
source_amplitudes 0.003112524911627291
global_phase_rad 0.0006227225657142733
predicted_field_abs 0.00217793836904194
unweighted source error 0.17180313491718766
signal and true residual quantiles [0.02113297 0.06468471 0.12660188] [0.00076065 0.00204398 0.00374096 0.10015182 0.37253834 0.74068712]
condition 2.9618123405365124 true-field forward error 1.1272979772244929e-15
geometry 1.0 1.0 11.5 0.85
source_amplitudes 0.005678418539034241
global_phase_rad 0.00026689595102702407
predicted_field_abs 0.0036352642498357606
unweighted source error 0.2727211285456088
signal and true residual quantiles [0.04551291 0.08598488 0.16018043] [0.00075374 0.00180389 0.00426436 0.09990006 0.40901003 0.68602987]
condition 3.1136094055719403 true-field forward error 9.93191916978576e-16
case 0 quadrant diagnostics: n, median absolute residual, fraction above 0.03, retained design condition
0 36 0.002168 0.083 3.138
1 36 0.001901 0.056 3.511
2 31 0.001983 0.065 3.225
3 37 0.002242 0.081 3.3
case 1 quadrant diagnostics: n, median absolute residual, fraction above 0.03, retained design condition
0 42 0.001814 0.095 3.499
1 36 0.001884 0.056 3.416
2 31 0.001715 0.097 3.312
3 31 0.001998 0.032 3.679
pilot jackknife 24 worst source / phase / field: [0.00571967 0.00074135 0.00364305] max fraction of tolerance: 0.19065550857294475
survey quadrant deletion 8 worst source / phase / field: [0.00727819 0.00062272 0.0044557 ] max fraction of tolerance: 0.24260621066622814

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6aaea370bbf4af05 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6aaea370 (directory).
```

### Turn 6 — `submit_completed_task` (`13360 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification. On real measurements, pilot and receiver-quadrant deletions exposed no calibration or observability bottleneck; the independent rectangular-grid tests also passed near the phase wrap with concentrated receiver impulses. Robust receiver weighting resolves the observed failure of unweighted least squares.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3716'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across 408 validation runs, worst errors were 0.728% for source amplitudes, 0.00105 rad for phase, and 0.476% for field magnitude. All met the specified tolerances.`

### Turn 7 — `execute_in_remote_docker` (`25576 ms` LLM)
- **Rationale**: Test one remaining nuisance directly: predict each public receiver quadrant using sources fitted outside it, then measure whether a phase correction materially reduces the held-out residual. A consistent correction near the phase tolerance would warrant revisiting pilot-to-survey transfer; small or inconsistent corrections would not justify changing the verified solver.
- **Alternatives Weighed**: `Add cohort centering or nonlinear source terms` (rejected: No batch or class variables are present, and the exact linear complex forward map reproduces the public reference fields to about 1e-15.)

```bash
cat > /tmp/helmholtz_phase_transfer.py <<'PY'
import runpy
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize_scalar

ns = runpy.run_path('/tmp/helmholtz_public.py')
m, cases = ns['m'], ns['cases']
for idx, case in enumerate(cases):
    p = case['problem']
    pilot, survey = p['pilot_calibration'], p['survey']
    ref = pilot['known_field_re'] + 1j * pilot['known_field_im']
    phase = np.angle(np.vdot(ref, m._complex_observations(pilot)))
    fields = m._source_fields(p)
    i, j = survey['sensor_ij'].T
    design = fields[i, j]
    data = m._complex_observations(survey) * np.exp(-1j * phase)
    groups = 2*(i >= fields.shape[0]/2)+(j >= fields.shape[1]/2)
    equivalent_phase, loss = defaultdict(list), defaultdict(list)
    for group in np.unique(groups):
        take = groups == group
        train_a = m._robust_sources(design[~take], data[~take], survey['tikhonov_alpha'])
        pred = design[take] @ train_a
        obs = data[take]
        def score(delta):
            return np.median(np.abs(obs - np.exp(1j*delta)*pred))
        fit = minimize_scalar(score, bounds=(-.12, .12), method='bounded')
        equivalent_phase[group] = fit.x
        loss[group] = (score(0), fit.fun)
    print('case', idx, 'quadrant phase corrections (rad) and median residual before/after')
    for group in equivalent_phase:
        print(int(group), round(float(equivalent_phase[group]), 6), np.round(loss[group], 7))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/helmholtz_phase_transfer.py
```
- **Container Exit Code**: `0` (`625 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
geometry 1.0 1.0 11.5 0.85
source_amplitudes 0.003112524911627291
global_phase_rad 0.0006227225657142733
predicted_field_abs 0.00217793836904194
unweighted source error 0.17180313491718766
signal and true residual quantiles [0.02113297 0.06468471 0.12660188] [0.00076065 0.00204398 0.00374096 0.10015182 0.37253834 0.74068712]
condition 2.9618123405365124 true-field forward error 1.1272979772244929e-15
geometry 1.0 1.0 11.5 0.85
source_amplitudes 0.005678418539034241
global_phase_rad 0.00026689595102702407
predicted_field_abs 0.0036352642498357606
unweighted source error 0.2727211285456088
signal and true residual quantiles [0.04551291 0.08598488 0.16018043] [0.00075374 0.00180389 0.00426436 0.09990006 0.40901003 0.68602987]
condition 3.1136094055719403 true-field forward error 9.93191916978576e-16
case 0 quadrant phase corrections (rad) and median residual before/after
0 0.00132 [0.0023372 0.0022921]
1 0.001674 [0.0018858 0.0018354]
2 0.00207 [0.0018621 0.0017706]
3 0.009257 [0.001998  0.0019564]
case 1 quadrant phase corrections (rad) and median residual before/after
0 0.007083 [0.0019345 0.0015479]
1 0.001822 [0.0018129 0.0017691]
2 0.000173 [0.0015688 0.0015603]
3 -0.002065 [0.0018027 0.0018012]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6aaea370bbf4af05 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6aaea370 (directory).
```

### Turn 8 — `submit_completed_task` (`14818 ms` LLM)
- **Rationale**: The saved artifact passed isolated-Python verification on public receiver and pilot deletions and on rectangular-grid, phase-wrap, and impulse stresses. The public forward fields match to about 1e-15, and robust receiver weighting reduced the two public source errors from 17.18% and 27.27% to 0.31% and 0.57%, resolving the observed least-squares failure without changing the stencil or calibration.
- **Alternatives Weighed**: `Add receiver-quadrant phase offsets` (rejected: Post-hoc corrections were below 0.010 rad and varied by case and quadrant; the shared-phase solver already passed all quadrant-deletion slices with margin.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. The worst observed errors were 0.728% for source amplitudes, 0.00105 rad for phase, and 0.476% for field magnitude. All met the specified tolerances.`

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
