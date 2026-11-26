# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Bare_ReAct_LiveGPT` | `vr02__large_robin_spd_preconditioned_cg_multi_rhs`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M90a-VoluntaryReuse-4` |
| **Domain** | `Large-Scale High-Contrast Elliptic PDE Preconditioning & Nuisance-Profiled Source Matching` |
| **Wave / Wormhole Family** | `m90a_voluntary_reuse` / `fam_multiscale_pde_and_robust_assimilation` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Bare_ReAct_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `5 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `82,730` (`78,291` / `16,384` / `4,439`) |
| **Total Episode Wall Time (s)** | `276.76s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`23167 ms` LLM)
- **Rationale**: Inspect the public fixture structure and SciPy version to identify array layout, coefficient ranges, and available sparse solver APIs before implementation.

```bash
cd /holdout_task && ls -la && python - <<'PY'
import pickle, scipy, numpy as np
print('SciPy',scipy.__version__)
with open('public_robin_spd_cases.pkl','rb') as f: d=pickle.load(f)
def describe(x,depth=0):
 if isinstance(x,dict):
  print(' '*depth,'dict keys',list(x))
  if depth<3:
   for k,v in list(x.items())[:20]: print(' '*depth,str(k),end=': '); describe(v,depth+1)
 elif isinstance(x,(list,tuple)):
  print(type(x).__name__,len(x))
  for v in x[:2]: describe(v,depth+1)
 elif isinstance(x,np.ndarray): print(x.shape,x.dtype, 'range', (x.min(),x.max()) if x.size and np.issubdtype(x.dtype,np.number) else None)
 else: print(repr(x)[:200])
describe(d)
PY
```
- **Container Exit Code**: `127` (`358 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 1856
drwxr-xr-x 2 root root    4096 Nov 26 00:49 .
drwxr-xr-x 1 root root    4096 Nov 26 00:49 ..
-rw-r--r-- 1 root root 1890946 Nov 26 00:49 public_robin_spd_cases.pkl
```
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`14239 ms` LLM)
- **Rationale**: Use python3 after the missing python executable, and inspect fixture shapes and installed solvers to choose the sparse implementation and verification cases.

```bash
cd /holdout_task && command -v python3 && python3 - <<'PY'
import pickle, importlib.util
import numpy as np
import scipy
print('numpy', np.__version__, 'scipy', scipy.__version__, 'pyamg', bool(importlib.util.find_spec('pyamg')))
with open('public_robin_spd_cases.pkl','rb') as f: data=pickle.load(f)
def show(a, depth=0):
    if isinstance(a, dict):
        print('dict', list(a))
        if depth < 3:
            for k,v in a.items():
                print(' '*depth+str(k)+':',end=' '); show(v,depth+1)
    elif isinstance(a,(list,tuple)):
        print(type(a).__name__,len(a))
        if depth < 3:
            for v in a[:3]: show(v,depth+1)
    elif isinstance(a,np.ndarray): print(a.shape,a.dtype, (a.min(),a.max()) if a.size and np.issubdtype(a.dtype,np.number) else '')
    else: print(repr(a)[:100])
show(data)
PY
```
- **Container Exit Code**: `0` (`569 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/usr/bin/python3
numpy 1.26.4 scipy 1.13.0 pyamg False
list 2
dict ['problem', 'gt']
 problem: dict ['Lx', 'Ly', 'kappa', 'sigma', 'gamma_robin', 'candidate_sources', 'sensor_ij', 'nuisance_design', 'nuisance_ridge', 'source_ridge', 'Y_obs']
  Lx: 1.4
  Ly: 1.2
  kappa: (72, 72) float64 (0.022, 75.0)
  sigma: (72, 72) float64 (0.06048611111111111, 0.12951388888888887)
  gamma_robin: 1.65
  candidate_sources: (10, 72, 72) float64 (2.415429504536423e-55, 0.9999489669611482)
  sensor_ij: (360, 2) int64 (0, 71)
  nuisance_design: (360, 4) float64 (-0.9976214598368016, 1.0)
  nuisance_ridge: 0.001
  source_ridge: 0.025
  Y_obs: (360, 5) float64 (-1.7728113401552967, 1.2717565657315537)
 gt: dict ['basis_fields', 'single_source_profiled_amplitudes', 'joint_source_weights']
  basis_fields: (10, 72, 72) float64 (6.252411805741707e-05, 0.024076828454340525)
  single_source_profiled_amplitudes: (10, 5) float64 (-4.3660760098064815, 3.925287606787953)
  joint_source_weights: (10, 5) float64 (-0.346503170240279, 0.27569689064696035)
dict ['problem', 'gt']
 problem: dict ['Lx', 'Ly', 'kappa', 'sigma', 'gamma_robin', 'candidate_sources', 'sensor_ij', 'nuisance_design', 'nuisance_ridge', 'source_ridge', 'Y_obs']
  Lx: 1.4
  Ly: 1.2
  kappa: (72, 72) float64 (0.022, 75.0)
  sigma: (72, 72) float64 (0.06048611111111111, 0.12951388888888887)
  gamma_robin: 1.65
  candidate_sources: (10, 72, 72) float64 (5.478021844428302e-52, 0.9999808025664702)
  sensor_ij: (360, 2) int64 (0, 71)
  nuisance_design: (360, 4) float64 (-0.999524110790929, 1.0)
  nuisance_ridge: 0.001
  source_ridge: 0.025
  Y_obs: (360, 5) float64 (-0.7309667979431722, 1.5510708159955666)
 gt: dict ['basis_fields', 'single_source_profiled_amplitudes', 'joint_source_weights']
  basis_fields: (10, 72, 72) float64 (6.758480349516637e-05, 0.041475484792134386)
  single_source_profiled_amplitudes: (10, 5) float64 (-4.081232224999073, 2.456454688516775)
  joint_source_weights: (10, 5) float64 (-0.38191522699830227, 0.474593190792141)
```

### Turn 3 — `execute_in_remote_docker` (`73184 ms` LLM)
- **Rationale**: Implement the stencil, multi-RHS solve, and augmented nuisance projection. Compare all three outputs with the public references; a mismatch will guide diagnosis of the PDE or inverse solve.
- **Alternatives Weighed**: `Factor the full sparse matrix once and solve directly` (rejected: Reusing a symmetric incomplete factor retains preconditioned CG as indicated by the task while avoiding a dense system.)

```bash
cd /holdout_task && cat > robin_spd_profiler.py <<'PY'
"""Finite-volume Robin diffusion solves and nuisance-profiled source fits."""

import numpy as np
from scipy.linalg import qr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import LinearOperator, cg, spilu, spsolve_triangular


def _assemble(kappa, sigma, gamma, hx, hy):
    """Assemble the integrated cell balances in C order."""
    cells = np.arange(kappa.size).reshape(kappa.shape)
    diag = (hx * hy) * sigma

    # Each interior face adds a symmetric two-cell flux block.
    wx = (hy / hx) * (2 * kappa[:-1, :] * kappa[1:, :] /
                       (kappa[:-1, :] + kappa[1:, :]))
    wy = (hx / hy) * (2 * kappa[:, :-1] * kappa[:, 1:] /
                       (kappa[:, :-1] + kappa[:, 1:]))
    diag[:-1, :] += wx
    diag[1:, :] += wx
    diag[:, :-1] += wy
    diag[:, 1:] += wy

    # Half-cell diffusion and wall dissipation act in series.
    for i in (0, -1):
        wall = kappa[i, :]
        diag[i, :] += hy * gamma * wall / (wall + 0.5 * hx * gamma)
    for j in (0, -1):
        wall = kappa[:, j]
        diag[:, j] += hx * gamma * wall / (wall + 0.5 * hy * gamma)

    lo = np.concatenate((cells[:-1, :].ravel(), cells[:, :-1].ravel()))
    hi = np.concatenate((cells[1:, :].ravel(), cells[:, 1:].ravel()))
    flux = -np.concatenate((wx.ravel(), wy.ravel()))
    return coo_matrix(
        (np.concatenate((diag.ravel(), flux, flux)),
         (np.concatenate((cells.ravel(), lo, hi)),
          np.concatenate((cells.ravel(), hi, lo)))),
        shape=(cells.size, cells.size),
    ).tocsr()


def _solve_fields(matrix, rhs):
    """Reuse one SPD incomplete-factor preconditioner for all columns."""
    order = reverse_cuthill_mckee(matrix, symmetric_mode=True)
    a = matrix[order, :][:, order].tocsr()
    ilu = spilu(a.tocsc(), permc_spec='NATURAL', diag_pivot_thresh=0,
                drop_tol=1e-5, fill_factor=8,
                options={'SymmetricMode': True, 'Equil': False})
    # Dropping need not preserve symmetry. Use L D L^T, not the raw ILU
    # solve, to keep the preconditioner symmetric positive definite for CG.
    lower = ilu.L
    upper = lower.T.tocsc()
    inv_diag = 1.0 / np.abs(ilu.U.diagonal())

    def precondition(r):
        v = spsolve_triangular(lower, r, lower=True, unit_diagonal=True)
        return spsolve_triangular(upper, inv_diag * v, lower=False,
                                  unit_diagonal=True)

    preconditioner = LinearOperator(a.shape, matvec=precondition, dtype=a.dtype)
    solutions = np.empty_like(rhs)
    for m, b in enumerate(rhs):
        x, info = cg(a, b[order], M=preconditioner, atol=0.0, rtol=1e-11)
        if info != 0:
            raise RuntimeError(f'Conjugate gradient did not converge for source {m}: {info}')
        solutions[m, order] = x
    return solutions


def _profile(s, z, y, lam_z, lam_s):
    """Eliminate the nuisance coefficients in the augmented observation space."""
    n_nuisance = z.shape[1]
    n_sources = s.shape[1]
    # Augmentation turns the nuisance ridge term into an ordinary squared norm.
    nuisance = np.vstack((z, np.sqrt(lam_z) * np.eye(n_nuisance)))
    s_aug = np.vstack((s, np.zeros((n_nuisance, n_sources))))
    y_aug = np.vstack((y, np.zeros((n_nuisance, y.shape[1]))))
    q, _ = qr(nuisance, mode='economic')
    s_profiled = s_aug - q @ (q.T @ s_aug)
    y_profiled = y_aug - q @ (q.T @ y_aug)

    numerator = s_profiled.T @ y_profiled
    norm_sq = np.einsum('ij,ij->j', s_profiled, s_profiled)
    amplitudes = np.divide(numerator, norm_sq[:, None],
                           out=np.zeros_like(numerator), where=norm_sq[:, None] > 0)
    joint_design = np.vstack((s_profiled, np.sqrt(lam_s) * np.eye(n_sources)))
    joint_obs = np.vstack((y_profiled, np.zeros((n_sources, y.shape[1]))))
    weights = np.linalg.lstsq(joint_design, joint_obs, rcond=None)[0]
    return amplitudes, weights


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Solve the source responses and fit single-source and joint models."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    sigma = np.asarray(problem['sigma'], dtype=float)
    nx, ny = kappa.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    matrix = _assemble(kappa, sigma, problem['gamma_robin'], hx, hy)
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    rhs = (hx * hy) * sources.reshape(sources.shape[0], nx * ny)
    fields = _solve_fields(matrix, rhs).reshape(sources.shape)

    i, j = np.asarray(problem['sensor_ij']).T
    amplitudes, weights = _profile(fields[:, i, j].T,
                                   np.asarray(problem['nuisance_design'], dtype=float),
                                   np.asarray(problem['Y_obs'], dtype=float),
                                   problem['nuisance_ridge'], problem['source_ridge'])
    return {'basis_fields': fields,
            'single_source_profiled_amplitudes': amplitudes,
            'joint_source_weights': weights}
PY
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle, time
import numpy as np
from robin_spd_profiler import solve_robin_spd_and_profile_sources
with open('public_robin_spd_cases.pkl','rb') as f: cases = pickle.load(f)
for i, case in enumerate(cases):
    start = time.perf_counter()
    out = solve_robin_spd_and_profile_sources(case['problem'])
    print('case', i, 'seconds', time.perf_counter()-start)
    for key, expected in case['gt'].items():
        print(key, out[key].shape, 'max abs error', np.max(np.abs(out[key]-expected)))
        np.testing.assert_allclose(out[key], expected, rtol=1e-7, atol=1e-9)
PY
```
- **Container Exit Code**: `0` (`111127 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin_spd_profiler.py']`
- **Stdout (tail)**:
```text
case 0 seconds 55.206311790971085
basis_fields (10, 72, 72) max abs error 6.454906054109699e-15
single_source_profiled_amplitudes (10, 5) max abs error 1.425526363618701e-12
joint_source_weights (10, 5) max abs error 1.395827897709978e-13
case 1 seconds 55.295045885955915
basis_fields (10, 72, 72) max abs error 5.3637649877202875e-15
single_source_profiled_amplitudes (10, 5) max abs error 1.4725998198628076e-12
joint_source_weights (10, 5) max abs error 1.2106982083537332e-13
```
- **Stderr (tail)**:
```text
/holdout_task/robin_spd_profiler.py:58: SparseEfficiencyWarning: CSR matrix format is required. Converting to CSR matrix.
  v = spsolve_triangular(lower, r, lower=True, unit_diagonal=True)
/holdout_task/robin_spd_profiler.py:59: SparseEfficiencyWarning: CSR matrix format is required. Converting to CSR matrix.
  return spsolve_triangular(upper, inv_diag * v, lower=False,
/holdout_task/robin_spd_profiler.py:58: SparseEfficiencyWarning: CSR matrix format is required. Converting to CSR matrix.
  v = spsolve_triangular(lower, r, lower=True, unit_diagonal=True)
/holdout_task/robin_spd_profiler.py:59: SparseEfficiencyWarning: CSR matrix format is required. Converting to CSR matrix.
  return spsolve_triangular(upper, inv_diag * v, lower=False,
```

### Turn 4 — `execute_in_remote_docker` (`30533 ms` LLM)
- **Rationale**: Move the repeated triangular applications to native solves while preserving the SPD preconditioner. Recheck reference agreement and independently evaluate face-balance residuals to catch numerical or stencil regressions.
- **Alternatives Weighed**: `Keep SciPy's per-call triangular solver` (rejected: The verified numerical result took 55 seconds per case and repeatedly converted sparse formats.)

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('robin_spd_profiler.py')
s = p.read_text().replace('from scipy.linalg import qr','from scipy.linalg import orth, qr').replace('cg, spilu, spsolve_triangular','cg, spilu, splu')
a = s.index('    lower = ilu.L')
b = s.index('\n    preconditioner =', a)
s = s[:a] + '''    # Factor the triangular matrix once to use native sparse solves in
    # both directions, without a Python-level row loop on each CG step.
    lower = splu(ilu.L, permc_spec='NATURAL', diag_pivot_thresh=0,
                 options={'Equil': False})
    inv_diag = 1.0 / np.abs(ilu.U.diagonal())

    def precondition(r):
        return lower.solve(inv_diag * lower.solve(r), trans='T')
''' + s[b:]
s = s.replace("    q, _ = qr(nuisance, mode='economic')", "    if lam_z > 0:\n        q, _ = qr(nuisance, mode='economic')\n    else:\n        q = orth(nuisance)")
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -B - <<'PY'
import pickle, time
import numpy as np
from robin_spd_profiler import solve_robin_spd_and_profile_sources
with open('public_robin_spd_cases.pkl','rb') as f: cases = pickle.load(f)
for i, case in enumerate(cases):
    p = case['problem']
    t = time.perf_counter()
    out = solve_robin_spd_and_profile_sources(p)
    elapsed = time.perf_counter() - t
    for key, expected in case['gt'].items():
        np.testing.assert_allclose(out[key], expected, rtol=1e-7, atol=1e-9)
    k = p['kappa']; g = p['gamma_robin']; nx, ny = k.shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    u = out['basis_fields']; b = hx*hy*p['candidate_sources']
    r = hx*hy*p['sigma']*u - b
    x = (u[:,:-1,:]-u[:,1:,:])*(hy/hx)*(2/(1/k[:-1,:]+1/k[1:,:]))
    y = (u[:,:,:-1]-u[:,:,1:])*(hx/hy)*(2/(1/k[:,:-1]+1/k[:,1:]))
    r[:,:-1,:] += x; r[:,1:,:] -= x
    r[:,:,:-1] += y; r[:,:,1:] -= y
    for face in (0,-1):
        r[:,face,:] += hy/(1/g + hx/(2*k[face,:]))*u[:,face,:]
        r[:,:,face] += hx/(1/g + hy/(2*k[:,face]))*u[:,:,face]
    residual = np.max(np.linalg.norm(r, axis=(1,2))/np.linalg.norm(b, axis=(1,2)))
    print(f'case {i}: {elapsed:.3f} s, max relative PDE residual {residual:.3e}, reference outputs match')
    assert residual <= 1e-8
PY
```
- **Container Exit Code**: `0` (`1336 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin_spd_profiler.py']`
- **Stdout (tail)**:
```text
case 0: 0.434 s, max relative PDE residual 2.645e-11, reference outputs match
case 1: 0.424 s, max relative PDE residual 2.922e-11, reference outputs match
```

### Turn 5 — `submit_completed_task` (`11146 ms` LLM)
- **Rationale**: The required module is saved, both public cases match the reference outputs, and independent face-balance checks confirm the residual requirement.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/robin_spd_profiler.py` with the Robin finite-volume stencil, a shared preconditioner for all source solves, and both nuisance-profiled fits. All three outputs matched the references on both public cases. The worst relative PDE residual was 2.93×10⁻¹¹, below the 10⁻⁸ requirement.`

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
