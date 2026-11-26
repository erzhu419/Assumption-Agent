# Dialectical Synthesis Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_2_Selective_LiveGPT` | `hd02__3d_high_contrast_robin_amg_multi_rhs_and_nuisance_tomography`

- **Official Strict Pass**: **`PASS (1.0)`** (`PASSED_ALL_CHECKS`)
- **Blind Public Synthesis Choice**: `antithesis` via `selective_router_confident_single_branch`
- **Synthesis Rationale**: Primary Antithesis branch satisfied public checks under policy; second branch skipped (`1.00u` nominal units).
- **Post-Lock Branch Verdicts**: Thesis (`Bare`) = `NOT_SPAWNED (0T)` | Antithesis (`Wormhole`) = `True` (7T)

## Selected Branch Trajectory Record
# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `hd02__3d_high_contrast_robin_amg_multi_rhs_and_nuisance_tomography`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M91-HardDemand-2` |
| **Domain** | `3D High-Contrast Algebraic Multigrid Elliptic PDE & Nuisance-Profiled Source Tomography` |
| **Wave / Wormhole Family** | `m91_hard_demand` / `fam_multiscale_pde_and_robust_assimilation` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `7 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `6` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `179,085` (`172,672` / `49,920` / `6,413`) |
| **Total Episode Wall Time (s)** | `218.68s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`18947 ms` LLM)
- **Rationale**: Inspect the public input schema and coefficient ranges before coding. This catches mismatches in source layout, sensor indexing, and nuisance dimensions and determines which public slices to validate against the stated objectives.
- **Alternatives Weighed**: `Reuse the mounted zero-ghost grid assembly` (rejected: Its Dirichlet boundary term does not match the specified Robin half-cell conductance.)

```bash
python3 - <<'PY'
import pickle
import numpy as np
import scipy
from pathlib import Path
with Path('/holdout_task/public_robin3d_cases.pkl').open('rb') as f:
    data = pickle.load(f)
print('SciPy:', scipy.__version__)
def describe(o, name='public'):
    if isinstance(o, dict) and 'kappa' in o:
        print(name, 'keys:', sorted(o))
        for k, v in o.items():
            if isinstance(v, np.ndarray):
                print(k, v.shape, v.dtype, (v.min(), v.max()) if v.size and v.dtype.kind in 'ifu' else '')
            elif np.isscalar(v):
                print(k, v)
            else:
                print(k, type(v).__name__)
    elif isinstance(o, dict):
        for k, v in o.items(): describe(v, f'{name}.{k}')
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o): describe(v, f'{name}[{i}]')
    else:
        print(name, type(o).__name__)
describe(data)
PY
```
- **Container Exit Code**: `0` (`632 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
SciPy: 1.13.0
public[0].problem keys: ['Lx', 'Ly', 'Lz', 'Y_obs', 'candidate_sources', 'gamma_robin', 'kappa', 'nuisance_design', 'nuisance_ridge', 'sensor_ijk', 'sigma', 'source_ridge']
Lx 1.2
Ly 1.1
Lz 1.0
kappa (20, 20, 20) float64 (0.02, 80.0)
sigma (20, 20, 20) float64 (0.052250000000000005, 0.13775)
gamma_robin 1.5
candidate_sources (8, 20, 20, 20) float64 (1.1240851771000825e-25, 0.9979583217974671)
sensor_ijk (320, 3) int64 (0, 19)
nuisance_design (320, 5) float64 (-0.9938441702975689, 1.0)
nuisance_ridge 0.0015
source_ridge 0.025
Y_obs (320, 4) float64 (-0.9227534433119405, 1.7143269201683184)
public[0].gt.basis_fields ndarray
public[0].gt.single_source_profiled_amplitudes ndarray
public[0].gt.joint_source_weights ndarray
```

### Turn 2 — `execute_in_remote_docker` (`51328 ms` LLM)
- **Rationale**: Implement the harmonic face flux and Robin half-cell conductance directly, then profile both inverse objectives in augmented least squares. Compare all three outputs against the supplied public reference before stressing the 28³ runtime.
- **Alternatives Weighed**: `Implement AMG immediately` (rejected: A single sparse factorization per problem is explicitly allowed; benchmark it before adding a multigrid hierarchy.)

```bash
cat > /holdout_task/robin3d_amg_tomography.py <<'PY'
"""Conservative 3D Robin diffusion-reaction and ridge-profiled tomography."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _robin_operator(kappa, sigma, h, gamma):
    """Assemble integrated cell balances in C order."""
    shape = kappa.shape
    n = kappa.size
    cells = np.arange(n).reshape(shape)
    volume = np.prod(h)
    diagonal = np.broadcast_to(volume * sigma, shape).copy()
    rows, cols, entries = [], [], []
    for axis, step in enumerate(h):
        lo = [slice(None)] * 3
        hi = lo.copy()
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        area = volume / step
        kl, kh = kappa[lo], kappa[hi]
        conductance = (area / step) * (2 * kl * kh / (kl + kh))
        diagonal[lo] += conductance
        diagonal[hi] += conductance
        left, right = cells[lo].ravel(), cells[hi].ravel()
        rows.extend((left, right))
        cols.extend((right, left))
        entries.extend((-conductance.ravel(), -conductance.ravel()))
        for edge in (0, -1):
            face = [slice(None)] * 3
            face[axis] = edge
            face = tuple(face)
            kb = kappa[face]
            diagonal[face] += area * gamma * kb / (kb + gamma * step / 2)
    rows.append(cells.ravel())
    cols.append(cells.ravel())
    entries.append(diagonal.ravel())
    matrix = coo_matrix((np.concatenate(entries),
                         (np.concatenate(rows), np.concatenate(cols))),
                        shape=(n, n)).tocsc()
    return matrix, volume


def _basis_solve(matrix, rhs):
    # The operator is SPD: symmetric minimum-degree ordering limits 3D fill,
    # and diagonal pivots suffice. Reuse the factors for every source.
    factor = splu(matrix, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                  options={'SymmetricMode': True})
    fields = np.empty_like(rhs)
    for i, b in enumerate(rhs):
        u = factor.solve(b)
        threshold = 1e-7 * np.linalg.norm(b)
        for _ in range(2):
            residual = b - matrix @ u
            if np.linalg.norm(residual) <= threshold:
                break
            u += factor.solve(residual)
        else:
            if not np.linalg.norm(b - matrix @ u) <= threshold:
                raise ArithmeticError('Robin solve did not reach the residual tolerance')
        fields[i] = u
    return fields


def _profile(s, z, y, lam_z, lam_s):
    # Apply the orthogonal complement of [Z; sqrt(lam_z) I] to both
    # [S; 0] and [Y; 0]. The added rows retain the nuisance penalty.
    m, p = s.shape[1], z.shape[1]
    z_aug = np.vstack((z, np.sqrt(lam_z) * np.eye(p)))
    data = np.vstack((np.column_stack((s, y)), np.zeros((p, m + y.shape[1]))))
    if p:
        q, sv, _ = np.linalg.svd(z_aug, full_matrices=False)
        q = q[:, sv > np.finfo(float).eps * max(z_aug.shape) * sv[0]]
        data -= q @ (q.T @ data)
    sx, yx = data[:, :m], data[:, m:]
    energy = np.sum(sx * sx, axis=0)
    cross = sx.T @ yx
    # A column in the nuisance span has no identifiable single amplitude.
    noise_floor = (np.finfo(float).eps * max(z_aug.shape)) ** 2
    usable = energy > noise_floor * np.sum(s * s, axis=0)
    amplitudes = np.divide(cross, energy[:, None], out=np.zeros_like(cross),
                           where=usable[:, None])
    joint = np.linalg.lstsq(
        np.vstack((sx, np.sqrt(lam_s) * np.eye(m))),
        np.vstack((yx, np.zeros((m, y.shape[1])))), rcond=None)[0]
    return amplitudes, joint


def solve_3d_robin_tomography(problem: dict) -> dict:
    """Return source response fields, single-source amplitudes and joint weights."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    sigma = np.asarray(problem['sigma'], dtype=float)
    h = np.array([problem['Lx'], problem['Ly'], problem['Lz']]) / kappa.shape
    a, volume = _robin_operator(kappa, sigma, h, float(problem['gamma_robin']))
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    fields = _basis_solve(a, volume * sources.reshape(sources.shape[0], -1))
    fields = fields.reshape(sources.shape)
    i, j, k = np.asarray(problem['sensor_ijk']).T
    s = fields[:, i, j, k].T
    amplitudes, joint = _profile(s, np.asarray(problem['nuisance_design'], dtype=float),
                                 np.asarray(problem['Y_obs'], dtype=float),
                                 float(problem['nuisance_ridge']),
                                 float(problem['source_ridge']))
    return {'basis_fields': fields,
            'single_source_profiled_amplitudes': amplitudes,
            'joint_source_weights': joint}
PY
python3 -I - <<'PY'
import importlib.util, pickle, time
import numpy as np
path = '/holdout_task/robin3d_amg_tomography.py'
spec = importlib.util.spec_from_file_location('solver', path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('/holdout_task/public_robin3d_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
for case in cases:
    t = time.perf_counter()
    result = solver.solve_3d_robin_tomography(case['problem'])
    print('public seconds', time.perf_counter()-t)
    assert result.keys() == case['gt'].keys()
    for key, ref in case['gt'].items():
        actual = result[key]
        rel = np.linalg.norm(actual-ref) / np.linalg.norm(ref)
        print(key, actual.shape, 'relative error', rel)
        assert actual.shape == ref.shape and rel < 1e-7
PY
```
- **Container Exit Code**: `0` (`1018 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin3d_amg_tomography.py']`
- **Stdout (tail)**:
```text
public seconds 0.11881944094784558
basis_fields (8, 20, 20, 20) relative error 2.6565335707331195e-14
single_source_profiled_amplitudes (8, 4) relative error 1.080689990813205e-13
joint_source_weights (8, 4) relative error 5.063183366190329e-13

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=09f8efea7106141a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_09f8efea (directory).
```

### Turn 3 — `execute_in_remote_docker` (`37503 ms` LLM)
- **Rationale**: Test two full-size, 4,000× contrast cases in a clean process. An independent flux calculation detects assembly errors; direct augmented least squares detects profiling errors. Pivot to AMG only if the combined solver time lacks margin under 14 seconds.

```bash
python3 -I - <<'PY'
import importlib.util
import pickle
import time
import numpy as np
from scipy.ndimage import zoom

path = '/holdout_task/robin3d_amg_tomography.py'
spec = importlib.util.spec_from_file_location('solver', path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('/holdout_task/public_robin3d_cases.pkl', 'rb') as f:
    public = pickle.load(f)[0]['problem']

base = dict(public)
base['kappa'] = zoom(public['kappa'], (1.4,)*3, order=0)
base['sigma'] = zoom(public['sigma'], (1.4,)*3, order=1)
base['candidate_sources'] = zoom(public['candidate_sources'], (1., 1.4, 1.4, 1.4), order=1)
base['sensor_ijk'] = np.rint(public['sensor_ijk'] * (27/19)).astype(int)
xyz = np.indices((28,)*3)
blocks = ((xyz[0]//4 + xyz[1]//5 + xyz[2]//6) % 2 == 0)
stress = dict(base, kappa=np.where(blocks, 80., .02), Lx=1.6, Ly=.85, Lz=1.15, gamma_robin=.3)


def check(p, out):
    u = out['basis_fields']
    assert u.shape == (8, 28, 28, 28)
    k = p['kappa']
    h = np.array([p['Lx'], p['Ly'], p['Lz']]) / np.array(k.shape)
    gamma = p['gamma_robin']
    # Divergence form per unit volume, independently of the sparse assembly.
    applied = p['sigma'] * u
    for axis, dx in enumerate(h):
        kk = np.moveaxis(k, axis, -1)
        uu = np.moveaxis(u, axis+1, -1)
        aa = np.moveaxis(applied, axis+1, -1)
        flux = (uu[..., :-1] - uu[..., 1:]) / ((.5 / kk[..., :-1] + .5 / kk[..., 1:]) * dx**2)
        aa[..., :-1] += flux
        aa[..., 1:] -= flux
        for face in (0, -1):
            aa[..., face] += uu[..., face] / (dx * (1/gamma + .5*dx/kk[..., face]))
    q = p['candidate_sources']
    residual = (np.linalg.norm((applied-q).reshape(len(q), -1), axis=1) / np.linalg.norm(q.reshape(len(q), -1), axis=1)).max()
    i, j, k = p['sensor_ijk'].T
    s, z, y = u[:, i, j, k].T, p['nuisance_design'], p['Y_obs']
    m, n, r = s.shape[1], z.shape[1], y.shape[1]
    rz, rs = np.sqrt(p['nuisance_ridge']), np.sqrt(p['source_ridge'])
    singles = []
    for col in s.T:
        a = np.vstack((np.column_stack((z, col)), np.column_stack((rz*np.eye(n), np.zeros(n)))))
        singles.append(np.linalg.lstsq(a, np.vstack((y, np.zeros((n,r)))), rcond=None)[0][-1])
    a = np.vstack((np.column_stack((z,s)), np.diag(np.r_[np.full(n,rz),np.full(m,rs)])))
    joint = np.linalg.lstsq(a, np.vstack((y,np.zeros((n+m,r)))), rcond=None)[0][n:]
    errors = []
    for key, ref in [('single_source_profiled_amplitudes', np.array(singles)), ('joint_source_weights', joint)]:
        assert out[key].shape == ref.shape
        errors.append(np.linalg.norm(out[key]-ref)/np.linalg.norm(ref))
    return residual, *errors

elapsed = 0.
metrics = []
for name, p in [('resized_public', base), ('alternating_blocks', stress)]:
    start = time.perf_counter()
    result = solver.solve_3d_robin_tomography(p)
    sec = time.perf_counter() - start
    elapsed += sec
    row = check(p, result)
    metrics.append(row)
    print(f'{name}: {sec:.3f}s; max residual={row[0]:.3e}; single error={row[1]:.3e}; joint error={row[2]:.3e}', flush=True)
print(f'combined solve time={elapsed:.3f}s')
assert elapsed < 14.
for residual, single, joint in metrics:
    assert residual <= 1e-7 and single < 1e-7 and joint < 1e-7
PY
```
- **Container Exit Code**: `0` (`6099 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
resized_public: 2.221s; max residual=1.106e-12; single error=3.966e-15; joint error=2.011e-14
alternating_blocks: 2.601s; max residual=1.776e-11; single error=2.430e-14; joint error=1.559e-14
combined solve time=4.821s

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=09f8efea7106141a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_09f8efea (directory).
```

### Turn 4 — `submit_completed_task` (`14017 ms` LLM)
- **Rationale**: The saved artifact passed clean-process verification against the public reference and two full-size high-contrast stress cases. Independent flux balances and augmented least-squares fits verified the three outputs.
- **Alternatives Weighed**: `Galerkin-preconditioned CG` (rejected: Reusing one sparse factorization per problem met the measured accuracy and runtime targets with margin.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/robin3d_amg_tomography.py': 'FILE:4654'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/robin3d_amg_tomography.py`. All three outputs matched the public reference within 5.1e-13 relative error. Two full-size 28³ stress cases completed in 4.821 seconds combined, below the 14-second budget, with a worst relative PDE residual of 1.776e-11.`

### Turn 5 — `execute_in_remote_docker` (`32179 ms` LLM)
- **Rationale**: Check conservation and the distinct penalized and unpenalized source objectives on the public measurements in an isolated Python process. A balance failure would trigger a Robin assembly review; a stationarity failure would trigger a profiling review. The full-size run already established substantial runtime margin.
- **Alternatives Weighed**: `Batch-centered classification or relay-mode checks` (rejected: The public problem has continuous observations and a supplied nuisance design, not batches, class labels, or relay modes.)

```bash
python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

path = '/holdout_task/robin3d_amg_tomography.py'
spec = importlib.util.spec_from_file_location('robin3d', path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('/holdout_task/public_robin3d_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

for case in cases:
    p = case['problem']
    out = solver.solve_3d_robin_tomography(p)
    u = out['basis_fields']
    k = p['kappa']
    h = np.array([p['Lx'], p['Ly'], p['Lz']]) / k.shape
    gamma = p['gamma_robin']
    # Interior fluxes cancel in the summed finite-volume balance.
    loss = (u * p['sigma']).sum(axis=(1, 2, 3))
    for axis, step in enumerate(h):
        for edge in (0, -1):
            ku = np.take(k, edge, axis=axis)
            uf = np.take(u, edge, axis=axis + 1)
            loss += (uf / (step * (1/gamma + step/(2*ku)))).sum(axis=(1, 2))
    total = p['candidate_sources'].sum(axis=(1, 2, 3))
    balance = np.max(np.abs(loss-total) / np.abs(total))
    i, j, l = p['sensor_ijk'].T
    s = u[:, i, j, l].T
    z, y = p['nuisance_design'], p['Y_obs']
    rz, rs = p['nuisance_ridge'], p['source_ridge']
    g = z.T @ z + rz*np.eye(z.shape[1])
    def profiled_residual(e):
        return e - z @ np.linalg.solve(g, z.T @ e)
    def normalized_gradient(grad, design):
        scale = np.linalg.norm(design, axis=0)[:, None] * np.linalg.norm(y, axis=0)[None, :]
        return np.max(np.abs(grad) / scale)
    single = out['single_source_profiled_amplitudes']
    e = (s[:, :, None]*single[None, :, :] - y[:, None, :]).reshape(len(y), -1)
    si = np.einsum('km,kmq->mq', s, profiled_residual(e).reshape(len(y), *single.shape))
    w = out['joint_source_weights']
    ji = s.T @ profiled_residual(s @ w - y) + rs*w
    e1, e2 = normalized_gradient(si, s), normalized_gradient(ji, s)
    confounding = np.linalg.norm(z @ np.linalg.solve(g, z.T @ y), axis=0) / np.linalg.norm(y, axis=0)
    print(f'max source-to-loss balance error={balance:.3e}; max normalized single gradient={e1:.3e}; joint gradient={e2:.3e}; nuisance-only fit norm / observation norm={confounding}')
    assert balance < 1e-7 and e1 < 1e-10 and e2 < 1e-10
PY
```
- **Container Exit Code**: `0` (`1065 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
max source-to-loss balance error=1.185e-13; max normalized single gradient=6.905e-16; joint gradient=4.949e-16; nuisance-only fit norm / observation norm=[0.99969657 0.99993669 0.99996196 0.99996342]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=09f8efea7106141a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_09f8efea (directory).
```

### Turn 6 — `submit_completed_task` (`15066 ms` LLM)
- **Rationale**: Isolated-process checks verified the saved artifact against the public reference and two full-size stress cases. The public conservation error was 1.185e-13, and both nuisance-profiled objectives had normalized stationarity errors below 7e-16.
- **Alternatives Weighed**: `Galerkin-preconditioned CG` (rejected: One reused sparse factorization per problem met both the runtime and residual limits with margin.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/robin3d_amg_tomography.py': 'FILE:4654'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin3d_amg_tomography.py`. All three outputs matched the public reference within 5.1e-13 relative error. Two full-size 28³ stress cases took 4.821 seconds combined (budget: 14 seconds), with a worst relative PDE residual of 1.776e-11 (limit: 1e-7).`

### Turn 7 — `submit_completed_task` (`15643 ms` LLM)
- **Rationale**: Clean-process tests verified the saved artifact against public reference outputs, independent full-size flux balances, and direct augmented least-squares fits. The public conservation error was 1.185e-13, and both nuisance-profiled objectives had normalized stationarity errors below 7e-16. The problem has no cohort labels or relay modes to audit.
- **Alternatives Weighed**: `Galerkin-preconditioned CG` (rejected: The specification permits reused sparse factorization, which met both the residual and runtime limits with margin.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin3d_amg_tomography.py`. All three outputs matched the public reference within 5.1e-13 relative error. Two full-size 28³ stress cases took 4.821 seconds combined against a 14-second budget, with a worst relative PDE residual of 1.776e-11 against a 1e-7 limit.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-M91-HardDemand-2",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/robin3d_amg_tomography.py": "FILE:4654"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "all_valid": true,
    "max_pde_rel_residual": 0.0,
    "max_field_rel_l2": 0.0,
    "max_profiled_amp_rel_l2": 0.0,
    "max_joint_weight_rel_l2": 0.0,
    "student_elapsed_sec": 2.8591
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"all_valid\": true, \"max_pde_rel_residual\": 0.0, \"max_field_rel_l2\": 0.0, \"max_profiled_amp_rel_l2\": 0.0, \"max_joint_weight_rel_l2\": 0.0, \"student_elapsed_sec\": 2.8591}\n__HEGEL_VERIFIER_AUTH_db6cda963cc8b794d08f17f124222768__:{\"strict_pass\": true, \"all_valid\": true, \"max_pde_rel_residual\": 0.0, \"max_field_rel_l2\": 0.0, \"max_profiled_amp_rel_l2\": 0.0, \"max_joint_weight_rel_l2\": 0.0, \"student_elapsed_sec\": 2.8591}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 15302.05,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
