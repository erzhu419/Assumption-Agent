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
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `137,359` (`131,498` / `40,704` / `5,861`) |
| **Total Episode Wall Time (s)** | `209.21s` |

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

### Turn 1 — `execute_in_remote_docker` (`16478 ms` LLM)
- **Rationale**: Inspect fixture shapes, coefficient ranges, and solver availability to catch input-layout or dependency mismatches before implementing and timing the solver.
- **Alternatives Weighed**: `Implement immediately against assumed fixture and dependencies` (rejected: Inspection establishes the actual array layout and whether an AMG dependency is available before choosing a solver.)

```bash
cd /holdout_task && python3 - <<'PY'
import importlib.util, pickle
import numpy as np, scipy
print('scipy', scipy.__version__, 'pyamg', importlib.util.find_spec('pyamg'))
with open('public_robin3d_cases.pkl','rb') as f: x=pickle.load(f)
print(type(x), len(x)); print(list(x) if isinstance(x,dict) else '')
def inspect(x, depth=0):
 if isinstance(x,dict) and 'kappa' in x:
  for k,v in x.items():
   if isinstance(v,np.ndarray): print(k,v.shape,v.dtype,np.min(v),np.max(v))
   else: print(k, type(v), repr(v)[:100])
 elif isinstance(x,dict) and depth<2:
  for k,v in list(x.items())[:5]: print('entry',k); inspect(v,depth+1)
 elif isinstance(x,(list,tuple)) and depth<2:
  for v in x[:3]: inspect(v,depth+1)
inspect(x)
PY
```
- **Container Exit Code**: `0` (`604 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
scipy 1.13.0 pyamg None
<class 'list'> 1

entry problem
Lx <class 'float'> 1.2
Ly <class 'float'> 1.1
Lz <class 'float'> 1.0
kappa (20, 20, 20) float64 0.02 80.0
sigma (20, 20, 20) float64 0.052250000000000005 0.13775
gamma_robin <class 'float'> 1.5
candidate_sources (8, 20, 20, 20) float64 1.1240851771000825e-25 0.9979583217974671
sensor_ijk (320, 3) int64 0 19
nuisance_design (320, 5) float64 -0.9938441702975689 1.0
nuisance_ridge <class 'float'> 0.0015
source_ridge <class 'float'> 0.025
Y_obs (320, 4) float64 -0.9227534433119405 1.7143269201683184
entry gt
```

### Turn 2 — `execute_in_remote_docker` (`43674 ms` LLM)
- **Rationale**: The first implementation preserves the exact face conductances and both ridge penalties. The fixture comparison can uncover assembly or profiling errors before full-size timing and independent residual checks.
- **Alternatives Weighed**: `Implement AMG from scratch` (rejected: PyAMG is unavailable, and the specification permits reusable sparse LU; measure its full-size runtime before introducing a multigrid hierarchy.)

```bash
cd /holdout_task && cat > robin3d_amg_tomography.py <<'PY'
"""Cell-centered Robin diffusion and ridge-profiled source tomography."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _robin_matrix(kappa, sigma, lengths, gamma):
    """Assemble integrated finite-volume balances in C order."""
    shape = kappa.shape
    steps = np.asarray(lengths, dtype=float) / shape
    volume = float(np.prod(steps))
    cells = np.arange(kappa.size).reshape(shape)
    diagonal = (volume * np.broadcast_to(sigma, shape)).copy().ravel()
    rows, cols, values = [], [], []
    for axis, step in enumerate(steps):
        area = volume / step
        left = [slice(None)] * 3
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        i, j = cells[left].ravel(), cells[right].ravel()
        kl, kr = kappa[left].ravel(), kappa[right].ravel()
        conductance = (area / step) * (2 * kl * kr / (kl + kr))
        diagonal[i] += conductance
        diagonal[j] += conductance
        rows.extend((i, j))
        cols.extend((j, i))
        values.extend((-conductance, -conductance))
        for end in (0, -1):
            face = [slice(None)] * 3
            face[axis] = end
            face = tuple(face)
            kb = kappa[face].ravel()
            diagonal[cells[face].ravel()] += area * gamma * kb / (kb + gamma * step / 2)
    rows.append(cells.ravel())
    cols.append(cells.ravel())
    values.append(diagonal)
    matrix = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(kappa.size, kappa.size)).tocsc()
    return matrix, volume


def _profile_sources(S, Z, Y, nuisance_ridge, source_ridge):
    """Solve ridge objectives as augmented least squares, without normal equations."""
    n, m = S.shape
    p = Z.shape[1]
    z_aug = np.vstack((Z, np.sqrt(nuisance_ridge) * np.eye(p)))
    s_aug = np.vstack((S, np.zeros((p, m))))
    y_aug = np.vstack((Y, np.zeros((p, Y.shape[1]))))
    single = np.empty((m, Y.shape[1]))
    for i in range(m):
        design = np.column_stack((z_aug, s_aug[:, i]))
        single[i] = np.linalg.lstsq(design, y_aug, rcond=None)[0][-1]
    design = np.block([[z_aug, s_aug],
                       [np.zeros((m, p)), np.sqrt(source_ridge) * np.eye(m)]])
    target = np.vstack((y_aug, np.zeros((m, Y.shape[1]))))
    joint = np.linalg.lstsq(design, target, rcond=None)[0][p:]
    return single, joint


def solve_3d_robin_tomography(problem: dict) -> dict:
    """Return unit-source fields and nuisance-profiled source estimates."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    matrix, volume = _robin_matrix(kappa, problem['sigma'],
                                   [problem[key] for key in ('Lx', 'Ly', 'Lz')],
                                   problem['gamma_robin'])
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    rhs = np.asfortranarray(volume * sources.reshape(sources.shape[0], -1).T)
    # Positive definite stencil: preserve symmetric ordering and reuse one LU.
    factor = splu(matrix, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                  options={'SymmetricMode': True})
    fields = factor.solve(rhs).T.reshape(sources.shape)
    sensors = tuple(np.asarray(problem['sensor_ijk']).T)
    S = fields[(slice(None),) + sensors].T
    single, joint = _profile_sources(S, np.asarray(problem['nuisance_design']),
                                    np.asarray(problem['Y_obs']),
                                    problem['nuisance_ridge'], problem['source_ridge'])
    return {'basis_fields': fields, 'single_source_profiled_amplitudes': single,
            'joint_source_weights': joint}
PY
python3 -I - <<'PY'
import runpy, pickle, time
import numpy as np
impl=runpy.run_path('/holdout_task/robin3d_amg_tomography.py')
case=pickle.load(open('/holdout_task/public_robin3d_cases.pkl','rb'))[0]
p=case['problem']; print('Reference metadata:', type(case['gt']), list(case['gt']))
t=time.perf_counter(); out=impl['solve_3d_robin_tomography'](p); print('seconds:',time.perf_counter()-t)
a,v=impl['_robin_matrix'](p['kappa'],p['sigma'],[p[k] for k in ('Lx','Ly','Lz')],p['gamma_robin']); b=v*p['candidate_sources'].reshape(8,-1).T
print('max residual:',max(np.linalg.norm(a@out['basis_fields'].reshape(8,-1).T-b,axis=0)/np.linalg.norm(b,axis=0)))
for k,x in out.items():
 print(k,x.shape, 'reference max abs error:',np.max(np.abs(x-case['gt'][k])) if k in case['gt'] else 'no matching key')
PY
```
- **Container Exit Code**: `0` (`974 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin3d_amg_tomography.py']`
- **Stdout (tail)**:
```text
Reference metadata: <class 'dict'> ['basis_fields', 'single_source_profiled_amplitudes', 'joint_source_weights']
seconds: 0.127185198944062
max residual: 4.341072518270067e-13
basis_fields (8, 20, 20, 20) reference max abs error: 2.732189474663471e-16
single_source_profiled_amplitudes (8, 4) reference max abs error: 4.0573100434926346e-13
joint_source_weights (8, 4) reference max abs error: 5.07788255887931e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=31ea6a3b11ab0c16 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_31ea6a3b (directory).
```

### Turn 3 — `execute_in_remote_docker` (`41660 ms` LLM)
- **Rationale**: The public fixture agrees with all three references. Full-size timing checks whether LU reuse meets the budget; independent face fluxes and nuisance elimination can catch errors that implementation-derived residuals would miss.
- **Alternatives Weighed**: `Submit based on the public case` (rejected: The 20³ fixture does not establish runtime margin for two 28³ problems.)

```bash
python3 -I - <<'PY'
import pickle, runpy, time
import numpy as np

with open('/holdout_task/public_robin3d_cases.pkl', 'rb') as f:
    case = pickle.load(f)[0]
solve = runpy.run_path('/holdout_task/robin3d_amg_tomography.py')['solve_3d_robin_tomography']
base = case['problem']
public = solve(base)
for key, expected in case['gt'].items():
    assert public[key].shape == expected.shape
    error = np.max(np.abs(public[key] - expected))
    print('public', key, 'max abs error', error, flush=True)
    np.testing.assert_allclose(public[key], expected, rtol=1e-8, atol=1e-9)

rng = np.random.default_rng(17082)
shape = (28, 28, 28)
parent = np.floor((np.arange(28) + 0.5) * 20 / 28).astype(int)
idx = np.ix_(parent, parent, parent)
stress = dict(base, kappa=base['kappa'][idx], sigma=base['sigma'][idx],
              candidate_sources=np.stack([q[idx] for q in base['candidate_sources']]),
              sensor_ijk=np.floor((base['sensor_ijk'] + 0.5) * 28 / 20).astype(int))
checker = np.indices(shape).sum(axis=0) % 2
channel = np.full(shape, 80.0)
channel[11:18, 5:22, 7:21] = 0.02
configs = [('resampled-public', stress),
           ('checkerboard', dict(stress, Lx=0.82, Ly=1.4, Lz=1.08, gamma_robin=0.24,
                                 kappa=np.where(checker, 80.0, 0.02),
                                 sensor_ijk=rng.integers(0, 28, size=(320, 3)))),
           ('high-conductivity-shell', dict(stress, kappa=channel, gamma_robin=0.08))]
timings = []
worst_residual = 0.0
for name, p in configs:
    start = time.perf_counter()
    result = solve(p)
    seconds = time.perf_counter() - start
    timings.append(seconds)
    u = result['basis_fields']
    assert u.shape == (8,) + shape
    k = p['kappa']
    h = np.array([p['Lx'], p['Ly'], p['Lz']]) / shape
    vol = np.prod(h)
    rhs = vol * p['candidate_sources']
    # Apply the stencil as explicit face fluxes, independently of sparse assembly.
    balance = vol * p['sigma'] * u
    for a in range(3):
        uu = np.moveaxis(u, a + 1, -1)
        kk = np.moveaxis(k, a, -1)
        bb = np.moveaxis(balance, a + 1, -1)
        area = vol / h[a]
        resistance = (0.5 * h[a]) * (1 / kk[..., :-1] + 1 / kk[..., 1:])
        flux = area * np.diff(uu, axis=-1) / resistance
        bb[..., :-1] -= flux
        bb[..., 1:] += flux
        for edge in (0, -1):
            bb[..., edge] += area * uu[..., edge] / (1 / p['gamma_robin'] + 0.5 * h[a] / kk[..., edge])
    residual = (np.linalg.norm((balance-rhs).reshape(8,-1), axis=1)
                / np.linalg.norm(rhs.reshape(8,-1), axis=1))
    worst_residual = max(worst_residual, float(max(residual)))
    assert np.all(residual <= 1e-7), residual
    s = u[(slice(None),) + tuple(p['sensor_ijk'].T)].T
    z, y = p['nuisance_design'], p['Y_obs']
    inv_action = np.linalg.solve(z.T @ z + p['nuisance_ridge'] * np.eye(z.shape[1]),
                                 z.T @ np.column_stack((s, y)))
    projected = np.column_stack((s, y)) - z @ inv_action
    gram = s.T @ projected[:, :8]
    cross = s.T @ projected[:, 8:]
    expected = {'single_source_profiled_amplitudes': cross / np.diag(gram)[:, None],
                'joint_source_weights': np.linalg.solve(gram + p['source_ridge'] * np.eye(8), cross)}
    for key, ref in expected.items():
        assert result[key].shape == (8, y.shape[1])
        np.testing.assert_allclose(result[key], ref, rtol=1e-8, atol=1e-9)
    print(name, 'seconds', round(seconds, 4), 'worst RHS relative residual', max(residual),
          'profile max abs errors', {key: float(np.max(np.abs(result[key] - ref))) for key, ref in expected.items()}, flush=True)
worst_pair = sum(sorted(timings)[-2:])
print('Worst two solve times combined:', worst_pair, 's; worst residual:', worst_residual, flush=True)
assert worst_pair < 14.0, worst_pair
PY
```
- **Container Exit Code**: `0` (`9457 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public basis_fields max abs error 2.732189474663471e-16
public single_source_profiled_amplitudes max abs error 4.0573100434926346e-13
public joint_source_weights max abs error 5.07788255887931e-14
resampled-public seconds 2.5023 worst RHS relative residual 1.0955621366292761e-12 profile max abs errors {'single_source_profiled_amplitudes': 1.1189937865196953e-12, 'joint_source_weights': 2.2287727219350018e-14}
checkerboard seconds 2.6 worst RHS relative residual 3.797802141519372e-14 profile max abs errors {'single_source_profiled_amplitudes': 1.608695815447092e-14, 'joint_source_weights': 1.2139161204016702e-14}
high-conductivity-shell seconds 2.9988 worst RHS relative residual 3.211508385043475e-11 profile max abs errors {'single_source_profiled_amplitudes': 1.5274148612576255e-10, 'joint_source_weights': 2.3984937300158826e-13}
Worst two solve times combined: 5.598805292276666 s; worst residual: 3.211508385043475e-11

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=31ea6a3b11ab0c16 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_31ea6a3b (directory).
```

### Turn 4 — `submit_completed_task` (`14656 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification against all public references, independent face-flux balances, and independent nuisance elimination on three full-size conductivity layouts.
- **Alternatives Weighed**: `Replace sparse LU with AMG-preconditioned CG` (rejected: The permitted reusable LU already meets the residual and runtime targets with margin on the full-size stress cases.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/robin3d_amg_tomography.py': 'FILE:3736'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin3d_amg_tomography.py`. All three outputs matched the public reference. Across three 28³ stress cases, the worst relative PDE residual was 3.21e-11, and the two slowest solves took 5.60 s combined, below the 14 s budget.`

### Turn 5 — `execute_in_remote_docker` (`26794 ms` LLM)
- **Rationale**: The stencil and runtime already passed independent full-size checks. This audit targets the remaining relevant failure mode: a nuisance-only change in the public observations must leave the PDE fields unchanged and move both source estimates by exactly the amount permitted by nuisance ridge shrinkage.
- **Alternatives Weighed**: `Apply batch-centering and classification checks` (rejected: The fixture has no batch labels or classification outcome. Its nuisance structure is a measured design matrix with an explicitly specified ridge penalty.)

```bash
python3 -I - <<'PY'
import pickle
import runpy
import numpy as np

with open('/holdout_task/public_robin3d_cases.pkl', 'rb') as f:
    case = pickle.load(f)[0]
p = case['problem']
solve = runpy.run_path('/holdout_task/robin3d_amg_tomography.py')['solve_3d_robin_tomography']
out = solve(p)
for key, ref in case['gt'].items():
    np.testing.assert_allclose(out[key], ref, rtol=1e-8, atol=1e-9)
S = out['basis_fields'][(slice(None),) + tuple(p['sensor_ijk'].T)].T
Z, Y = p['nuisance_design'], p['Y_obs']
lz, ls = p['nuisance_ridge'], p['source_ridge']
# Perturb the real observations only in the measured nuisance subspace.
C = np.cos(np.arange(Z.shape[1] * Y.shape[1]).reshape(Z.shape[1], Y.shape[1]))
D = Z @ C
shifted = solve(dict(p, Y_obs=Y + D))
G = Z.T @ Z + lz * np.eye(Z.shape[1])
# For a nuisance-only shift Z C, its ridge-profiled effect is Z (lz G^-1 C).
profiled_D = Z @ np.linalg.solve(G, lz * C)
profiled_S = S - Z @ np.linalg.solve(G, Z.T @ S)
H = S.T @ profiled_S
R = S.T @ profiled_D
expected = {'single_source_profiled_amplitudes': R / np.diag(H)[:, None],
            'joint_source_weights': np.linalg.solve(H + ls * np.eye(S.shape[1]), R)}
for key, ref in expected.items():
    measured = shifted[key] - out[key]
    np.testing.assert_allclose(measured, ref, rtol=1e-8, atol=1e-9)
    print(key, 'worst nuisance-response error', np.max(np.abs(measured-ref)))
np.testing.assert_array_equal(shifted['basis_fields'], out['basis_fields'])
print('public nuisance rank', np.linalg.matrix_rank(Z), '/', Z.shape[1], '; profiled shift norm ratio', np.linalg.norm(profiled_D) / np.linalg.norm(D))
PY
```
- **Container Exit Code**: `0` (`1502 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
single_source_profiled_amplitudes worst nuisance-response error 1.6876257336040368e-14
joint_source_weights worst nuisance-response error 8.090912922281701e-15
public nuisance rank 5 / 5 ; profiled shift norm ratio 3.4968072001874e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=31ea6a3b11ab0c16 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_31ea6a3b (directory).
```

### Turn 6 — `submit_completed_task` (`14349 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process checks against the public references, independent full-size face-flux balances and nuisance profiling, and an analytic nuisance perturbation test. No change is warranted.
- **Alternatives Weighed**: `Replace reusable sparse LU with multigrid-preconditioned CG` (rejected: The permitted LU approach met both the runtime and residual targets with margin on all full-size stress cases.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/robin3d_amg_tomography.py': 'FILE:3736'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin3d_amg_tomography.py`. All three outputs matched the public reference. Across three 28³ stress cases, the worst relative PDE residual was 3.21e-11, and the two slowest solves took 5.60 s combined against a 14 s budget. The nuisance-only perturbation matched the predicted ridge response to within 1.69e-14.`

### Turn 7 — `submit_completed_task` (`16093 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification against the public reference and independent full-size checks of the finite-volume balance, source estimates, and runtime. The fixture has no batch labels or classification outcome; its nuisance structure was tested directly through the supplied design matrix.
- **Alternatives Weighed**: `Run another audit or replace sparse LU with multigrid` (rejected: Independent full-size flux and profiling checks passed with margin, and the public nuisance perturbation confirmed the specified ridge response. The repeated gate identified no new applicable failure mode.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin3d_amg_tomography.py`. All three outputs matched the public reference. Across three 28³ stress cases, the worst relative PDE residual was 3.21e-11, and the two slowest solves took 5.60 s combined, below the 14 s budget. Both source estimates matched the predicted nuisance-only ridge response within 1.69e-14.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M91-HardDemand-2",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
