# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `vr02__large_robin_spd_preconditioned_cg_multi_rhs`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M90a-VoluntaryReuse-4` |
| **Domain** | `Large-Scale High-Contrast Elliptic PDE Preconditioning & Nuisance-Profiled Source Matching` |
| **Wave / Wormhole Family** | `m90a_voluntary_reuse` / `fam_multiscale_pde_and_robust_assimilation` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `7 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `140,664` (`134,408` / `51,584` / `6,256`) |
| **Total Episode Wall Time (s)** | `206.92s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`19957 ms` LLM)
- **Rationale**: Inspect only the fixture schema and numerical ranges, without transferring arrays. This will catch unexpected case nesting, degenerate inputs, or a missing sparse dependency before implementation.
- **Alternatives Weighed**: `Implement before inspecting the fixture and runtime` (rejected: The fixture layout determines how to validate every public case, and SciPy availability determines whether to use its sparse solver.)

```bash
cd /holdout_task && ls -l && python3 -I - <<'PY'
import pickle
import numpy as np
try:
 import scipy
 print('SciPy:', scipy.__version__)
except ImportError:
 print('SciPy unavailable')
with open('public_robin_spd_cases.pkl','rb') as f:
 data=pickle.load(f)
def describe(x, depth=0):
 if isinstance(x,dict):
  print('  '*depth, 'dict keys:', list(x)[:30])
  for key,value in list(x.items())[:20]:
   if isinstance(value, (dict,list,tuple)) and depth<3:
    print('  '*depth,key); describe(value,depth+1)
   elif isinstance(value,np.ndarray):
    print('  '*depth,key,value.shape,value.dtype, (float(value.min()),float(value.max())) if value.size and value.dtype.kind in 'fiu' else '')
   elif np.isscalar(value): print('  '*depth,key,repr(value)[:160])
 elif isinstance(x,(list,tuple)):
  print('  '*depth,type(x).__name__,len(x))
  if depth<3:
   for v in x[:8]: describe(v,depth+1)
 else: print('  '*depth,type(x).__name__)
describe(data)
PY
```
- **Container Exit Code**: `0` (`599 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 1848
-rw-r--r-- 1 root root 1890946 Nov 26 01:38 public_robin_spd_cases.pkl
SciPy: 1.13.0
 list 2
   dict keys: ['problem', 'gt']
   problem
     dict keys: ['Lx', 'Ly', 'kappa', 'sigma', 'gamma_robin', 'candidate_sources', 'sensor_ij', 'nuisance_design', 'nuisance_ridge', 'source_ridge', 'Y_obs']
     Lx 1.4
     Ly 1.2
     kappa (72, 72) float64 (0.022, 75.0)
     sigma (72, 72) float64 (0.06048611111111111, 0.12951388888888887)
     gamma_robin 1.65
     candidate_sources (10, 72, 72) float64 (2.415429504536423e-55, 0.9999489669611482)
     sensor_ij (360, 2) int64 (0.0, 71.0)
     nuisance_design (360, 4) float64 (-0.9976214598368016, 1.0)
     nuisance_ridge 0.001
     source_ridge 0.025
     Y_obs (360, 5) float64 (-1.7728113401552967, 1.2717565657315537)
   gt
     dict keys: ['basis_fields', 'single_source_profiled_amplitudes', 'joint_source_weights']
     basis_fields (10, 72, 72) float64 (6.252411805741707e-05, 0.024076828454340525)
     single_source_profiled_amplitudes (10, 5) float64 (-4.3660760098064815, 3.925287606787953)
     joint_source_weights (10, 5) float64 (-0.346503170240279, 0.27569689064696035)
   dict keys: ['problem', 'gt']
   problem
     dict keys: ['Lx', 'Ly', 'kappa', 'sigma', 'gamma_robin', 'candidate_sources', 'sensor_ij', 'nuisance_design', 'nuisance_ridge', 'source_ridge', 'Y_obs']
     Lx 1.4
     Ly 1.2
     kappa (72, 72) float64 (0.022, 75.0)
     sigma (72, 72) float64 (0.06048611111111111, 0.12951388888888887)
     gamma_robin 1.65
     candidate_sources (10, 72, 72) float64 (5.478021844428302e-52, 0.9999808025664702)
     sensor_ij (360, 2) int64 (0.0, 71.0)
     nuisance_design (360, 4) float64 (-0.999524110790929, 1.0)
     nuisance_ridge 0.001
     source_ridge 0.025
     Y_obs (360, 5) float64 (-0.7309667979431722, 1.5510708159955666)
   gt
     dict keys: ['basis_fields', 'single_source_profiled_amplitudes', 'joint_source_weights']
     basis_fields (10, 72, 72) float64 (6.758480349516637e-05, 0.041475484792134386)
     single_source_profiled_amplitudes (10, 5) float64 (-4.081232224999073, 2.456454688516775)
     joint_source_weights (10, 5) float64 (-0.38191522699830227, 0.474593190792141)
```

### Turn 2 — `execute_in_remote_docker` (`50551 ms` LLM)
- **Rationale**: Compare each output independently against both public fixtures to detect stencil or inverse-fit discrepancies before checking per-mode residuals independently.
- **Alternatives Weighed**: `Use preconditioned CG` (rejected: The stated interface specifies a residual tolerance, not an iterative algorithm; at 5,184 unknowns, sparse factorization can be shared across all ten right-hand sides without tuning a preconditioner.); `Profile using normal equations` (rejected: Orthogonal projection and augmented least squares avoid squaring the condition number.)

```bash
cat > /holdout_task/robin_spd_profiler.py <<'PY'
"""Cell-centered Robin diffusion solves and nuisance-profiled source fits."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _assemble_operator(kappa, sigma, gamma, hx, hy):
    """Assemble shared face conductances and cell-local dissipation."""
    nx, ny = kappa.shape
    cell = np.arange(nx * ny).reshape(nx, ny)
    kx0, kx1 = kappa[:-1, :], kappa[1:, :]
    ky0, ky1 = kappa[:, :-1], kappa[:, 1:]
    gx = (hy / hx) * (2 * kx0 * kx1 / (kx0 + kx1))
    gy = (hx / hy) * (2 * ky0 * ky1 / (ky0 + ky1))

    diag = sigma * (hx * hy)
    diag[:-1, :] += gx
    diag[1:, :] += gx
    diag[:, :-1] += gy
    diag[:, 1:] += gy
    for row in (0, nx - 1):
        kb = kappa[row, :]
        diag[row, :] += hy * gamma * kb / (kb + gamma * (0.5 * hx))
    for col in (0, ny - 1):
        kb = kappa[:, col]
        diag[:, col] += hx * gamma * kb / (kb + gamma * (0.5 * hy))

    left = np.concatenate((cell[:-1, :].ravel(), cell[:, :-1].ravel()))
    right = np.concatenate((cell[1:, :].ravel(), cell[:, 1:].ravel()))
    off = -np.concatenate((gx.ravel(), gy.ravel()))
    main = cell.ravel()
    return coo_matrix(
        (np.concatenate((diag.ravel(), off, off)),
         (np.concatenate((main, left, right)), np.concatenate((main, right, left)))),
        shape=(cell.size, cell.size),
    ).tocsc()


def _profile_sources(samples, nuisance, observations, nuisance_ridge, source_ridge):
    """Project out nuisance in the augmented least-squares space."""
    p = nuisance.shape[1]
    augmented = np.concatenate((samples, observations), axis=1)
    if p:
        z = np.vstack((nuisance, np.sqrt(nuisance_ridge) * np.eye(p)))
        augmented = np.pad(augmented, ((0, p), (0, 0)))
        if nuisance_ridge > 0:
            q, _ = np.linalg.qr(z, mode='reduced')
        else:
            # Without the ridge, only project onto the actual nuisance span.
            q, singular, _ = np.linalg.svd(z, full_matrices=False)
            q = q[:, singular > np.finfo(float).eps * max(z.shape) * singular[0]]
        augmented -= q @ (q.T @ augmented)

    m = samples.shape[1]
    s, y = augmented[:, :m], augmented[:, m:]
    energy = np.einsum('ij,ij->j', s, s)
    cross = s.T @ y
    single = np.divide(cross, energy[:, None], out=np.zeros_like(cross),
                       where=energy[:, None] > 0)
    joint = np.linalg.lstsq(
        np.vstack((s, np.sqrt(source_ridge) * np.eye(m))),
        np.pad(y, ((0, m), (0, 0))), rcond=None,
    )[0]
    return single, joint


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Solve all candidate fields and fit single and joint source coefficients."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    sigma = np.asarray(problem['sigma'], dtype=float)
    nx, ny = kappa.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    operator = _assemble_operator(kappa, sigma, problem['gamma_robin'], hx, hy)
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    rhs = (hx * hy) * sources.reshape(sources.shape[0], nx * ny).T
    # Factor once and solve against all candidate right-hand sides together.
    fields = splu(operator, permc_spec='MMD_AT_PLUS_A').solve(rhs).T.reshape(sources.shape)

    sensor_ij = np.asarray(problem['sensor_ij'])
    samples = fields[:, sensor_ij[:, 0], sensor_ij[:, 1]].T
    single, joint = _profile_sources(
        samples, np.asarray(problem['nuisance_design'], dtype=float),
        np.asarray(problem['Y_obs'], dtype=float), problem['nuisance_ridge'], problem['source_ridge'],
    )
    return dict(basis_fields=fields, single_source_profiled_amplitudes=single,
                joint_source_weights=joint)
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import time
import numpy as np
spec = importlib.util.spec_from_file_location('robin_spd_profiler', '/holdout_task/robin_spd_profiler.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with open('public_robin_spd_cases.pkl','rb') as f:
    cases = pickle.load(f)
for i, case in enumerate(cases):
    start = time.perf_counter()
    result = module.solve_robin_spd_and_profile_sources(case['problem'])
    assert result.keys() == case['gt'].keys()
    print('case', i, 'seconds', round(time.perf_counter()-start, 4))
    for key, expected in case['gt'].items():
        actual = result[key]
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        print(key, 'max abs', np.max(np.abs(actual-expected)))
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)
PY
```
- **Container Exit Code**: `0` (`839 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin_spd_profiler.py']`
- **Stdout (tail)**:
```text
case 0 seconds 0.0178
basis_fields max abs 3.4954678040932663e-15
single_source_profiled_amplitudes max abs 1.7426060594516457e-12
joint_source_weights max abs 1.4280243654241076e-13
case 1 seconds 0.0205
basis_fields max abs 3.382710778154774e-15
single_source_profiled_amplitudes max abs 7.784883848671598e-13
joint_source_weights max abs 1.1091128016005314e-13

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d87eed757c1c674a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d87eed75 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`41284 ms` LLM)
- **Rationale**: Check the exact artifact in an isolated process using direct face-flux balances and independent augmented least-squares references. A residual failure would isolate the forward solve or stencil; a coefficient failure with passing residuals would isolate nuisance profiling.
- **Alternatives Weighed**: `Submit on public output agreement alone` (rejected: An independent per-mode residual check is needed to verify the explicit PDE tolerance, especially for nearly uniform and corner-localized modes.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -B -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('robin_spd_profiler', '/holdout_task/robin_spd_profiler.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with open('/holdout_task/public_robin_spd_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

rng = np.random.default_rng(318)
base = cases[0]['problem']
ii, jj = np.indices(base['kappa'].shape)
q = rng.normal(size=base['candidate_sources'].shape)
q[0] = 0
q[0, 0, -1] = 1
q[1] = 1
q[2] *= 1e-6
problems = [(f'public-{i}', c['problem'], c['gt']) for i, c in enumerate(cases)]
problems += [
    ('checkerboard-strong-robin', dict(base, Lx=2.1, Ly=0.7, gamma_robin=100., kappa=np.where((ii+jj)%2, 0.02, 80.), candidate_sources=q), None),
    ('layered-weak-robin', dict(base, Lx=0.8, Ly=1.9, gamma_robin=1e-4, kappa=np.where(ii < 30, 0.02, 80.), candidate_sources=q, nuisance_ridge=1e-8, source_ridge=1e-7), None),
]
for label, p, gt in problems:
    out = module.solve_robin_spd_and_profile_sources(p)
    assert set(out) == {'basis_fields', 'single_source_profiled_amplitudes', 'joint_source_weights'}
    u = out['basis_fields']
    assert u.shape == p['candidate_sources'].shape and np.isfinite(u).all()
    k = p['kappa']
    nx, ny = k.shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    gamma = p['gamma_robin']
    # Apply the flux balance directly; do not use the implementation's assembled matrix.
    lhs = hx*hy * p['sigma'] * u
    fx = (u[:, :-1, :] - u[:, 1:, :]) * ((hy/hx) / (0.5/k[:-1, :] + 0.5/k[1:, :]))
    fy = (u[:, :, :-1] - u[:, :, 1:]) * ((hx/hy) / (0.5/k[:, :-1] + 0.5/k[:, 1:]))
    lhs[:, :-1, :] += fx
    lhs[:, 1:, :] -= fx
    lhs[:, :, :-1] += fy
    lhs[:, :, 1:] -= fy
    for i in (0, nx-1):
        lhs[:, i, :] += u[:, i, :] * hy / (1/gamma + 0.5*hx/k[i, :])
    for j in (0, ny-1):
        lhs[:, :, j] += u[:, :, j] * hx / (1/gamma + 0.5*hy/k[:, j])
    b = hx*hy * p['candidate_sources']
    residual = np.linalg.norm((lhs-b).reshape(len(u), -1), axis=1)/np.linalg.norm(b.reshape(len(u), -1), axis=1)
    assert residual.max() <= 1e-8, (label, residual)

    i, j = p['sensor_ij'].T
    s = u[:, i, j].T
    z, y = p['nuisance_design'], p['Y_obs']
    n, m = z.shape[1], s.shape[1]
    nz = np.sqrt(p['nuisance_ridge']) * np.eye(n)
    # Solve unprofiled augmented systems independently for reference coefficients.
    single = np.vstack([np.linalg.lstsq(np.block([[z, s[:, [a]]], [nz, np.zeros((n, 1))]]), np.pad(y, ((0, n), (0, 0))), rcond=None)[0][-1] for a in range(m)])
    joint = np.linalg.lstsq(np.block([[z, s], [nz, np.zeros((n, m))], [np.zeros((m, n)), np.sqrt(p['source_ridge'])*np.eye(m)]]), np.pad(y, ((0, n+m), (0, 0))), rcond=None)[0][n:]
    metrics = []
    for name, ref in [('single_source_profiled_amplitudes', single), ('joint_source_weights', joint)]:
        value = out[name]
        assert value.shape == ref.shape and np.isfinite(value).all()
        np.testing.assert_allclose(value, ref, atol=1e-8, rtol=1e-7, err_msg=label+' '+name)
        metrics.append(np.max(np.abs(value-ref)/(1+np.abs(ref))))
    if gt is not None:
        for name in gt:
            np.testing.assert_allclose(out[name], gt[name], atol=1e-9, rtol=1e-8)
    print(f'{label}: worst PDE relative residual {max(residual):.3e} (mode {residual.argmax()}), scaled single/joint errors {metrics[0]:.3e}/{metrics[1]:.3e}')
PY
```
- **Container Exit Code**: `0` (`902 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public-0: worst PDE relative residual 5.473e-12 (mode 9), scaled single/joint errors 4.421e-15/5.459e-16
public-1: worst PDE relative residual 5.627e-12 (mode 5), scaled single/joint errors 5.297e-15/2.115e-15
checkerboard-strong-robin: worst PDE relative residual 1.124e-13 (mode 1), scaled single/joint errors 2.927e-14/4.501e-15
layered-weak-robin: worst PDE relative residual 1.808e-09 (mode 1), scaled single/joint errors 6.050e-14/9.914e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d87eed757c1c674a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d87eed75 (directory).
```

### Turn 4 — `submit_completed_task` (`13054 ms` LLM)
- **Rationale**: The exact submitted artifact passed clean-process checks against independent face-flux balances, augmented least-squares fits, and both public references.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/robin_spd_profiler.py': 'FILE:3698'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/robin_spd_profiler.py`. Both public cases matched all three reference outputs. The worst relative PDE residual across the public and stress cases was 1.81e-9, below the 1e-8 requirement.`

### Turn 5 — `execute_in_remote_docker` (`30743 ms` LLM)
- **Rationale**: Audit the public cases for two task-specific identities without changing the artifact. A conservation failure would direct attention to wall or reaction terms; a stationarity failure would direct attention to nuisance projection or ridge placement.
- **Alternatives Weighed**: `Apply classification, drift, or relay diagnostics` (rejected: The task has no class labels, batch strata, time evolution, or relay modes. Its nuisance mechanism is explicitly specified as ridge regression.)

```bash
cat > /tmp/audit_robin_balance.py <<'PY'
import importlib.util
import pickle
import numpy as np

path = '/holdout_task/robin_spd_profiler.py'
spec = importlib.util.spec_from_file_location('robin_spd_profiler', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin_spd_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

for index, case in enumerate(cases):
    p = case['problem']
    out = mod.solve_robin_spd_and_profile_sources(p)
    k, q = p['kappa'], p['candidate_sources']
    hx, hy = p['Lx']/k.shape[0], p['Ly']/k.shape[1]
    g = p['gamma_robin']
    u = out['basis_fields']
    # Interior faces cancel in the domain-wide balance. Corners lose flux through both walls.
    walls = sum((u[:, i, :] * hy / (1/g + hx/(2*k[i, :]))).sum(axis=1) for i in (0, -1))
    walls += sum((u[:, :, j] * hx / (1/g + hy/(2*k[:, j]))).sum(axis=1) for j in (0, -1))
    reaction = hx*hy * (u * p['sigma']).sum(axis=(1, 2))
    production = hx*hy * q.sum(axis=(1, 2))
    balance = np.max(np.abs((walls + reaction - production) / production))
    assert balance < 1e-8, balance

    # For the positive nuisance ridge, P = I - Z(Z'Z + lambda I)^-1 Z'.
    # Check the resulting stationarity equations directly in sensor space.
    i, j = p['sensor_ij'].T
    s, z, y = u[:, i, j].T, p['nuisance_design'], p['Y_obs']
    inv = z.T @ z + p['nuisance_ridge'] * np.eye(z.shape[1])
    ps = s - z @ np.linalg.solve(inv, z.T @ s)
    py = y - z @ np.linalg.solve(inv, z.T @ y)
    cross = s.T @ py
    single = np.sum(s * ps, axis=0)[:, None] * out['single_source_profiled_amplitudes']
    joint = (s.T @ ps + p['source_ridge'] * np.eye(s.shape[1])) @ out['joint_source_weights']
    single_err, joint_err = (np.linalg.norm(fit-cross) / np.linalg.norm(cross) for fit in (single, joint))
    assert max(single_err, joint_err) < 1e-8, (single_err, joint_err)
    print(f'public-{index}: global balance relative error={balance:.3e}; single/joint stationarity relative errors={single_err:.3e}/{joint_err:.3e}')
PY
OPENBLAS_NUM_THREADS=1 python3 -B -I /tmp/audit_robin_balance.py
```
- **Container Exit Code**: `0` (`622 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public-0: global balance relative error=3.730e-13; single/joint stationarity relative errors=1.315e-13/1.314e-13
public-1: global balance relative error=3.858e-13; single/joint stationarity relative errors=1.234e-13/1.233e-13

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d87eed757c1c674a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d87eed75 (directory).
```

### Turn 6 — `submit_completed_task` (`12427 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process checks against independent face-flux balances and augmented least-squares fits. The public-case audit also confirmed global conservation and stationarity for both inverse fits.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/robin_spd_profiler.py': 'FILE:3698'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/robin_spd_profiler.py`. All three outputs matched both public references. The worst relative PDE residual across public and stress cases was 1.81e-9, below the 1e-8 requirement.`

### Turn 7 — `submit_completed_task` (`15894 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process checks of per-mode flux balance and both inverse fits. The focused public-case audit also confirmed global conservation and stationarity. The repeated gate supplied no new data or failure to investigate.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin_spd_profiler.py`. All three outputs matched both public references. The worst relative PDE residual across public and stress cases was 1.81e-9, below the 1e-8 requirement.`

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
