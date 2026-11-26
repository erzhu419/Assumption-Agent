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
| **Turns Used / Max Turns** | `8 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `164,557` (`157,165` / `51,840` / `7,392`) |
| **Total Episode Wall Time (s)** | `254.97s` |

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

### Turn 1 — `execute_in_remote_docker` (`19140 ms` LLM)
- **Rationale**: Inspect the public case and installed solver stack to catch schema or dependency mismatches before implementation. If the standard SciPy ordering is too slow at full scale, test a grid-aware ordering without changing the discrete operator.
- **Alternatives Weighed**: `Refactor for each right-hand side` (rejected: All eight source fields share the same operator, and the task reports that repeated factorization exceeds the time budget.)

```bash
ls -lh /holdout_task; python3 -I - <<'PY'
import pickle, importlib.util, os
import numpy as np, scipy
with open('/holdout_task/public_robin3d_cases.pkl','rb') as f: data=pickle.load(f)
print('SciPy',scipy.__version__,'PyAMG available',importlib.util.find_spec('pyamg') is not None,'BLAS threads',os.environ.get('OPENBLAS_NUM_THREADS'))
def describe(x, depth=0):
 if isinstance(x,dict) and depth<2:
  for k,v in x.items():
   print(' '*depth,k,end=': '); describe(v,depth+1)
 elif isinstance(x,(tuple,list)) and depth<2:
  print(type(x).__name__,len(x));
  for i,v in enumerate(x[:4]):
   print(' '*depth,i,end=': '); describe(v,depth+1)
 elif isinstance(x,np.ndarray): print(x.shape,x.dtype,'range', (x.min(),x.max()) if x.size and x.dtype.kind in 'iuf' else None)
 else: print(type(x).__name__, x if np.isscalar(x) else (list(x)[:20] if isinstance(x,dict) else ''))
describe(data)
PY
```
- **Container Exit Code**: `0` (`721 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 1.2M
-rw-r--r-- 1 root root 1.2M Nov 26 01:38 public_robin3d_cases.pkl
SciPy 1.13.0 PyAMG available False BLAS threads None
list 1
 0:   problem: dict ['Lx', 'Ly', 'Lz', 'kappa', 'sigma', 'gamma_robin', 'candidate_sources', 'sensor_ijk', 'nuisance_design', 'nuisance_ridge', 'source_ridge', 'Y_obs']
  gt: dict ['basis_fields', 'single_source_profiled_amplitudes', 'joint_source_weights']
```

### Turn 2 — `execute_in_remote_docker` (`47399 ms` LLM)
- **Rationale**: Public reference comparisons can expose a stencil or profiling error. A full-size solve tests whether factor reuse leaves adequate runtime margin; if it does not, change the ordering before changing the numerical method.
- **Alternatives Weighed**: `Build a custom multigrid hierarchy immediately` (rejected: PyAMG is unavailable, and reusable sparse LU is explicitly allowed. Measure whether it meets the full-size budget before adding complexity.)

```bash
cat > /holdout_task/robin3d_amg_tomography.py <<'PY'
"""Cell-centered Robin finite volumes and ridge-profiled source inversion."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _robin_operator(kappa, sigma, h, gamma):
    """Assemble the integrated cell balance in C-order."""
    vol = np.prod(h)
    ids = np.arange(kappa.size).reshape(kappa.shape)
    diagonal = (vol * sigma).copy()
    rows, cols, values = [], [], []
    for axis, step in enumerate(h):
        area = vol / step
        left = [slice(None)] * 3
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        kl, kr = kappa[left], kappa[right]
        g = (area / step) * (2.0 * kl * kr / (kl + kr))
        diagonal[left] += g
        diagonal[right] += g
        i, j, off = ids[left].ravel(), ids[right].ravel(), -g.ravel()
        rows.extend((i, j))
        cols.extend((j, i))
        values.extend((off, off))
        for endpoint in (0, -1):
            face = [slice(None)] * 3
            face[axis] = endpoint
            face = tuple(face)
            kb = kappa[face]
            diagonal[face] += area * gamma * kb / (kb + 0.5 * step * gamma)
    rows.append(ids.ravel())
    cols.append(ids.ravel())
    values.append(diagonal.ravel())
    return sparse.coo_matrix((np.concatenate(values),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(kappa.size, kappa.size)).tocsc()


def _profile_sources(S, Z, Y, lam_z, lam_s):
    """Project out the nuisance columns in the augmented ridge system."""
    m, p = S.shape[1], Z.shape[1]
    if p:
        Z_aug = np.vstack((Z, np.sqrt(lam_z) * np.eye(p)))
        S = np.pad(S, ((0, p), (0, 0)))
        Y = np.pad(Y, ((0, p), (0, 0)))
        U, singular, _ = np.linalg.svd(Z_aug, full_matrices=False)
        tol = np.finfo(float).eps * max(Z_aug.shape) * singular[0]
        Q = U[:, singular > tol]
        S = S - Q @ (Q.T @ S)
        Y = Y - Q @ (Q.T @ Y)
    denom = np.sum(S * S, axis=0)
    numer = S.T @ Y
    single = np.divide(numer, denom[:, None], out=np.zeros_like(numer),
                       where=denom[:, None] > 0)
    joint = np.linalg.lstsq(np.vstack((S, np.sqrt(lam_s) * np.eye(m))),
                            np.pad(Y, ((0, m), (0, 0))), rcond=None)[0]
    return single, joint


def solve_3d_robin_tomography(problem: dict) -> dict:
    """Solve all candidate fields and fit independent and joint amplitudes."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    sigma = np.asarray(problem['sigma'], dtype=float)
    h = np.array([problem['Lx'], problem['Ly'], problem['Lz']]) / kappa.shape
    A = _robin_operator(kappa, sigma, h, problem['gamma_robin'])
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    m = len(sources)
    rhs = np.asfortranarray(sources.reshape(m, -1).T * np.prod(h))
    lu = splu(A)
    u = np.column_stack([lu.solve(b) for b in rhs.T])
    fields = u.T.reshape(sources.shape)
    i, j, k = np.asarray(problem['sensor_ijk']).T
    single, joint = _profile_sources(
        fields[:, i, j, k].T,
        np.asarray(problem['nuisance_design'], dtype=float),
        np.asarray(problem['Y_obs'], dtype=float),
        problem['nuisance_ridge'], problem['source_ridge'])
    return dict(basis_fields=fields, single_source_profiled_amplitudes=single,
                joint_source_weights=joint)
PY
python3 -I - <<'PY'
import importlib.util, pickle, time
import numpy as np
s=importlib.util.spec_from_file_location('tomography','/holdout_task/robin3d_amg_tomography.py'); mod=importlib.util.module_from_spec(s); s.loader.exec_module(mod)
case=pickle.load(open('/holdout_task/public_robin3d_cases.pkl','rb'))[0]; p=case['problem']
for k,v in p.items():
 a=np.asarray(v); print(k,a.shape,a.dtype,float(a.min()),float(a.max()))
for n in (20,28):
 if n==28:
  idx=np.minimum(np.arange(n)*20//n,19)
  ix=np.ix_(idx,idx,idx)
  p=dict(p,kappa=p['kappa'][ix],sigma=p['sigma'][ix],candidate_sources=np.stack([q[ix] for q in p['candidate_sources']]),sensor_ijk=np.rint(p['sensor_ijk']*27/19).astype(int))
 t=time.perf_counter(); out=mod.solve_3d_robin_tomography(p); print('n',n,'seconds',time.perf_counter()-t)
 h=np.array([p['Lx'],p['Ly'],p['Lz']])/n
 A=mod._robin_operator(p['kappa'],p['sigma'],h,p['gamma_robin']); B=p['candidate_sources'].reshape(8,-1).T*np.prod(h)
 print('max residual',np.max(np.linalg.norm(A@out['basis_fields'].reshape(8,-1).T-B,axis=0)/np.linalg.norm(B,axis=0)))
 if n==20:
  for k,v in out.items():
   ref=case['gt'][k]; print(k,v.shape,'rel diff',np.linalg.norm(v-ref)/np.linalg.norm(ref),'maxabs',np.max(np.abs(v-ref)))
PY
```
- **Container Exit Code**: `0` (`4248 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin3d_amg_tomography.py']`
- **Stdout (tail)**:
```text
Lx () float64 1.2 1.2
Ly () float64 1.1 1.1
Lz () float64 1.0 1.0
kappa (20, 20, 20) float64 0.02 80.0
sigma (20, 20, 20) float64 0.052250000000000005 0.13775
gamma_robin () float64 1.5 1.5
candidate_sources (8, 20, 20, 20) float64 1.1240851771000825e-25 0.9979583217974671
sensor_ijk (320, 3) int64 0.0 19.0
nuisance_design (320, 5) float64 -0.9938441702975689 1.0
nuisance_ridge () float64 0.0015 0.0015
source_ridge () float64 0.025 0.025
Y_obs (320, 4) float64 -0.9227534433119405 1.7143269201683184
n 20 seconds 0.3897161199711263
max residual 7.161173033517655e-13
basis_fields (8, 20, 20, 20) rel diff 4.044663896319073e-17 maxabs 3.469446951953614e-18
single_source_profiled_amplitudes (8, 4) rel diff 1.0262794227518415e-13 maxabs 3.885780586188048e-13
joint_source_weights (8, 4) rel diff 4.995723383022384e-13 maxabs 4.97241137153992e-14
n 28 seconds 2.911635956959799
max residual 1.6954030597410968e-12

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3907acb4df4e86ae (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3907acb4 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`42348 ms` LLM)
- **Rationale**: Verify the exact artifact in an isolated Python process. An independent flux balance can catch a shared assembly error, direct augmented fits can catch a profiling error, and two differently structured full-size cases test the combined runtime budget. If a check fails, revise the corresponding component.
- **Alternatives Weighed**: `Replace sparse LU with custom AMG now` (rejected: The initial full-size timing leaves substantial headroom; switch only if the paired stress test exposes a runtime bottleneck.)

```bash
python3 -I - <<'PY'
import importlib.util, pickle, time
import numpy as np

spec = importlib.util.spec_from_file_location('tomography', '/holdout_task/robin3d_amg_tomography.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin3d_cases.pkl','rb') as f:
    case = pickle.load(f)[0]
base = case['problem']
public = mod.solve_3d_robin_tomography(base)
for key, ref in case['gt'].items():
    np.testing.assert_allclose(public[key], ref, rtol=1e-8, atol=1e-11)

# Independently evaluate each cell's net outward flux, including all six faces.
def residual(p, u):
    k = p['kappa']
    h = np.array([p['Lx'], p['Ly'], p['Lz']]) / k.shape
    vol = np.prod(h)
    gamma = p['gamma_robin']
    balance = vol * p['sigma'] * u
    for axis, step in enumerate(h):
        K = np.moveaxis(k, axis, -1)
        U = np.moveaxis(u, axis + 1, -1)
        F = np.empty(U.shape[:-1] + (U.shape[-1] + 1,))
        F[..., 1:-1] = (vol / step**2) * (U[..., :-1] - U[..., 1:]) / (0.5 / K[..., :-1] + 0.5 / K[..., 1:])
        F[..., 0] = -(vol / step) * U[..., 0] / (1 / gamma + 0.5 * step / K[..., 0])
        F[..., -1] = (vol / step) * U[..., -1] / (1 / gamma + 0.5 * step / K[..., -1])
        balance += np.moveaxis(np.diff(F, axis=-1), -1, axis + 1)
    b = vol * p['candidate_sources']
    return np.linalg.norm((balance - b).reshape(len(b), -1), axis=1) / np.linalg.norm(b.reshape(len(b), -1), axis=1)

def check_inverse(p, out):
    S = out['basis_fields'][:, *p['sensor_ijk'].T].T
    Z, Y = p['nuisance_design'], p['Y_obs']
    z, m = Z.shape[1], S.shape[1]
    Rz = np.sqrt(p['nuisance_ridge']) * np.eye(z)
    Rs = np.sqrt(p['source_ridge']) * np.eye(m)
    single = np.stack([np.linalg.lstsq(np.block([[Z, S[:, j:j+1]], [Rz, np.zeros((z, 1))]]), np.pad(Y, ((0,z),(0,0))), rcond=None)[0][-1] for j in range(m)])
    joint = np.linalg.lstsq(np.block([[Z, S], [Rz, np.zeros((z,m))], [np.zeros((m,z)), Rs]]), np.pad(Y, ((0,z+m),(0,0))), rcond=None)[0][-m:]
    err = []
    for key, ref in [('single_source_profiled_amplitudes', single), ('joint_source_weights', joint)]:
        np.testing.assert_allclose(out[key], ref, rtol=1e-8, atol=1e-10)
        err.append(np.max(np.abs(out[key] - ref)))
    return err

rng = np.random.default_rng(413)
idx = np.arange(28)*20//28
ix = np.ix_(idx,idx,idx)
x,y,z = np.indices((28,28,28))
contrast = [np.where((x//4 + y//5 + z//7)%2 == 0, 80., .02), np.where(rng.random(x.shape) < 0.8, 80., .02)]
total = 0.
for label, k, lengths, gamma in zip(('block interfaces','connected high-conductivity phase'), contrast, ((1.2,1.1,1.0),(1.9,0.8,1.3)), (3.,0.2)):
    p = dict(base, Lx=lengths[0], Ly=lengths[1], Lz=lengths[2], gamma_robin=gamma, kappa=k, sigma=base['sigma'][ix], candidate_sources=np.stack([q[ix] for q in base['candidate_sources']]), sensor_ijk=np.rint(base['sensor_ijk']*27/19).astype(int))
    t = time.perf_counter()
    out = mod.solve_3d_robin_tomography(p)
    elapsed = time.perf_counter()-t
    total += elapsed
    r = residual(p, out['basis_fields'])
    assert out['basis_fields'].shape == (8,28,28,28)
    assert np.max(r) <= 1e-7, r
    print(label, 'seconds', round(elapsed,3), 'worst RHS residual', max(r), 'single/joint max absolute error', check_inverse(p,out))
print('Two full-scale solves:', round(total,3), 's')
assert total < 14.0, total
PY
```
- **Container Exit Code**: `0` (`6704 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
block interfaces seconds 2.85 worst RHS residual 1.2648101699438804e-11 single/joint max absolute error [3.3861802251067274e-15, 2.67841304690819e-15]
connected high-conductivity phase seconds 2.396 worst RHS residual 1.6517391279667495e-11 single/joint max absolute error [6.394884621840902e-14, 6.0706648041808364e-15]
Two full-scale solves: 5.246 s

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3907acb4df4e86ae (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3907acb4 (directory).
```

### Turn 4 — `submit_completed_task` (`14407 ms` LLM)
- **Rationale**: The verified artifact passed the public reference comparison and independent flux-balance and augmented least-squares checks on two distinct full-size conductivity patterns. No further edits are needed.
- **Alternatives Weighed**: `Replace reusable sparse LU with custom AMG` (rejected: Sparse LU reuse is expressly permitted and met both the accuracy and runtime limits with substantial margin.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/robin3d_amg_tomography.py': 'FILE:3440'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin3d_amg_tomography.py`. The solver reuses one sparse LU factorization per case and returns both nuisance-profiled source estimates. All public outputs matched the reference, and two full-size 4,000×-contrast stress cases took 5.246 s combined, with a worst relative PDE residual of 1.66e-11.`

### Turn 5 — `execute_in_remote_docker` (`29136 ms` LLM)
- **Rationale**: The remaining domain-specific checks use the public case to test global conservation and the effect of a nuisance-only observation shift. A conservation failure would put the face or reaction assembly back under review; a shift-response failure would put the ridge profiling back under review. Neither check requires editing the verified artifact.
- **Alternatives Weighed**: `Run batch classification and relay-mode checks` (rejected: The public problem contains continuous measurements and a nuisance design matrix, not batch labels or relay states.)

```bash
cat > /tmp/audit_robin_public.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('tomography', '/holdout_task/robin3d_amg_tomography.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin3d_cases.pkl', 'rb') as f:
    p = pickle.load(f)[0]['problem']
out = mod.solve_3d_robin_tomography(p)
u = out['basis_fields']
kappa = p['kappa']
h = np.array([p['Lx'], p['Ly'], p['Lz']]) / kappa.shape
vol = np.prod(h)
loss = vol * np.sum(u * p['sigma'], axis=(1, 2, 3))
for a, step in enumerate(h):
    for end in (0, -1):
        k = np.take(kappa, end, axis=a)
        v = np.take(u, end, axis=a + 1)
        loss += np.sum(v * (vol / step) / (1 / p['gamma_robin'] + step / (2 * k)), axis=(1, 2))
supply = vol * np.sum(p['candidate_sources'], axis=(1, 2, 3))
balance = np.max(np.abs(loss - supply) / np.abs(supply))
assert balance < 1e-10, balance

# A nuisance-only perturbation changes weights by the finite-ridge leakage,
# not by the response from ignoring nuisance or projecting it out unpenalized.
i, j, k = p['sensor_ijk'].T
S = u[:, i, j, k].T
Z, Y = p['nuisance_design'], p['Y_obs']
lz, ls = p['nuisance_ridge'], p['source_ridge']
C = np.arange(1, Z.shape[1] * Y.shape[1] + 1, dtype=float).reshape(Z.shape[1], Y.shape[1]) / 3
G = Z.T @ Z + lz * np.eye(Z.shape[1])
PS = S - Z @ np.linalg.solve(G, Z.T @ S)
Pdelta = Z @ np.linalg.solve(G, lz * C)
cross = S.T @ Pdelta
expected = (cross / np.sum(S * PS, axis=0)[:, None], np.linalg.solve(S.T @ PS + ls * np.eye(S.shape[1]), cross))
shifted = mod._profile_sources(S, Z, Y + Z @ C, lz, ls)
errors = []
for key, actual, ref in zip(('single_source_profiled_amplitudes', 'joint_source_weights'), shifted, expected):
    delta = actual - out[key]
    np.testing.assert_allclose(delta, ref, rtol=1e-7, atol=1e-10)
    errors.append(np.max(np.abs(delta - ref)))
print('Public worst integrated conservation defect:', balance)
print('Public nuisance-shift max absolute error (single, joint):', errors)
PY
python3 -I /tmp/audit_robin_public.py
```
- **Container Exit Code**: `0` (`1234 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public worst integrated conservation defect: 8.955430632580243e-14
Public nuisance-shift max absolute error (single, joint): [6.323760959325853e-14, 1.6028302816939455e-15]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3907acb4df4e86ae (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3907acb4 (directory).
```

### Turn 6 — `submit_completed_task` (`14240 ms` LLM)
- **Rationale**: Isolated verification passed independent flux-balance and augmented least-squares checks. The public audit also confirmed global conservation to 8.96e-14 relative error and the predicted finite-ridge nuisance response to 6.33e-14 maximum absolute error.
- **Alternatives Weighed**: `Replace sparse LU with custom multigrid` (rejected: Reusing one sparse factorization per case is permitted and met both the accuracy and runtime requirements with substantial margin.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/robin3d_amg_tomography.py': 'FILE:3440'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin3d_amg_tomography.py`. All public outputs matched the reference. Two 28³ stress cases at 4,000× conductivity contrast took 5.246 s combined, with a worst relative PDE residual of 1.66e-11 across all 16 right-hand sides.`

### Turn 7 — `execute_in_remote_docker` (`31947 ms` LLM)
- **Rationale**: Removing each spatial sensor slab and reversing the retained rows tests whether the public entry point keeps source samples, nuisance rows, and observations aligned on noncontiguous subsets. A mismatch against the direct augmented fits would trigger a correction to sensor indexing or profiling; otherwise the verified artifact stays unchanged.
- **Alternatives Weighed**: `Fit a batch-centered classifier` (rejected: The public case contains continuous observations and a nuisance design matrix, with no classification labels or batch identifiers.)

```bash
cat > /tmp/audit_robin_sensor_slices.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('robin3d', '/holdout_task/robin3d_amg_tomography.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin3d_cases.pkl', 'rb') as f:
    case = pickle.load(f)[0]
p = case['problem']
sensors = p['sensor_ijk']
Z, Y = p['nuisance_design'], p['Y_obs']
S = case['gt']['basis_fields'][:, sensors[:, 0], sensors[:, 1], sensors[:, 2]].T
P, M = Z.shape[1], S.shape[1]
Rz = np.sqrt(p['nuisance_ridge']) * np.eye(P)
Rs = np.sqrt(p['source_ridge']) * np.eye(M)
slabs = 3 * sensors[:, 0] // p['kappa'].shape[0]
worst = 0.
for slab in range(3):
    held = slabs == slab
    keep = np.flatnonzero(~held)[::-1]
    z, y, s = Z[keep], Y[keep], S[keep]
    sub = dict(p, sensor_ijk=sensors[keep], nuisance_design=z, Y_obs=y)
    out = mod.solve_3d_robin_tomography(sub)
    single_ref = np.stack([np.linalg.lstsq(np.block([[z, s[:, j:j+1]], [Rz, np.zeros((P, 1))]]), np.pad(y, ((0, P), (0, 0))), rcond=None)[0][-1] for j in range(M)])
    joint_ref = np.linalg.lstsq(np.block([[z, s], [Rz, np.zeros((P, M))], [np.zeros((M, P)), Rs]]), np.pad(y, ((0, P+M), (0, 0))), rcond=None)[0][-M:]
    errors = []
    for key, ref in [('single_source_profiled_amplitudes', single_ref), ('joint_source_weights', joint_ref)]:
        np.testing.assert_allclose(out[key], ref, rtol=1e-8, atol=1e-10)
        errors.append(np.max(np.abs(out[key] - ref)))
    worst = max(worst, *errors)
    print(f'Excluded x-slab {slab}: {held.sum()} held-out sensors; max single/joint coefficient errors {errors}')
print(f'Worst sensor-slice coefficient error: {worst:.3e}')
PY
python3 -I /tmp/audit_robin_sensor_slices.py
```
- **Container Exit Code**: `0` (`2042 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Excluded x-slab 0: 108 held-out sensors; max single/joint coefficient errors [4.574118861455645e-14, 1.4953316362920077e-15]
Excluded x-slab 1: 111 held-out sensors; max single/joint coefficient errors [2.398081733190338e-14, 7.563394355258879e-16]
Excluded x-slab 2: 101 held-out sensors; max single/joint coefficient errors [5.417888360170764e-14, 1.0685896612017132e-15]
Worst sensor-slice coefficient error: 5.418e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3907acb4df4e86ae (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3907acb4 (directory).
```

### Turn 8 — `submit_completed_task` (`14309 ms` LLM)
- **Rationale**: The saved artifact passed isolated verification against the public reference, independent flux balances and augmented least-squares fits, and three public spatial sensor exclusions. The audits found no defect requiring an edit.
- **Alternatives Weighed**: `Replace reusable sparse LU with custom multigrid` (rejected: Factorization reuse is explicitly permitted and passed the full-size accuracy and runtime checks with substantial margin.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin3d_amg_tomography.py`. All three outputs matched the public reference. Two 28³ stress cases at 4,000× conductivity contrast took 5.246 s combined against a 14 s budget. The worst relative PDE residual across all 16 right-hand sides was 1.66e-11, below the 1e-7 limit.`

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
