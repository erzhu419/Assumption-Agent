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
| **Turns Used / Max Turns** | `6 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `102,408` (`97,617` / `9,856` / `4,791`) |
| **Total Episode Wall Time (s)** | `360.06s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`14218 ms` LLM)
- **Rationale**: Inspect the public input structure and numerical libraries to choose a compatible solver and catch any mismatch with the documented shapes.

```bash
ls -la /holdout_task /pretrain_task; python - <<'PY'
import pickle, importlib.util
from pathlib import Path
for name in ['numpy','scipy','pyamg']:
 print(name,importlib.util.find_spec(name) is not None)
x=pickle.loads(Path('/holdout_task/public_robin_spd_cases.pkl').read_bytes())
def inspect(x, level=0):
 p='  '*level
 if level>3: return
 if isinstance(x,dict):
  print(p,'dict:',list(x)[:30]);
  for k,v in list(x.items())[:18]:
   print(p,repr(k),end=' ')
   if hasattr(v,'shape'): print(v.shape,v.dtype, 'range:',v.min() if v.size else None,v.max() if v.size else None)
   elif isinstance(v,(dict,list,tuple)): print(); inspect(v,level+1)
   else: print(repr(v)[:160])
 elif isinstance(x,(list,tuple)):
  print(p,type(x).__name__,len(x));
  for v in x[:3]: inspect(v,level+1)
 else: print(p,type(x))
inspect(x)
PY
```
- **Container Exit Code**: `127` (`333 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/holdout_task:
total 1856
drwxr-xr-x 2 root root    4096 Nov 26 00:52 .
drwxr-xr-x 1 root root    4096 Nov 26 00:52 ..
-rw-r--r-- 1 root root 1890946 Nov 26 00:52 public_robin_spd_cases.pkl

/pretrain_task:
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 26 00:52 ..
```
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`13846 ms` LLM)
- **Rationale**: The previous command failed because `python` is absent. Use `python3` to identify available numerical libraries and confirm the public fixture layout before choosing the implementation and tests.

```bash
command -v python3; python3 - <<'PY'
import importlib.util, pickle
for name in ('numpy', 'scipy', 'pyamg'):
    print(name, importlib.util.find_spec(name) is not None)
with open('/holdout_task/public_robin_spd_cases.pkl','rb') as f:
    cases=pickle.load(f)
def inspect(obj, depth=0):
    indent='  '*depth
    if isinstance(obj, dict):
        print(indent, list(obj))
        for k,v in obj.items():
            if hasattr(v,'shape'):
                print(indent,k,v.shape,v.dtype, (v.min(),v.max()) if v.size else ())
            elif isinstance(v,(dict,list,tuple)) and depth<3:
                print(indent,k); inspect(v,depth+1)
            else: print(indent,k,repr(v)[:120])
    elif isinstance(obj,(list,tuple)):
        print(indent,type(obj).__name__,len(obj))
        for v in obj[:5]: inspect(v,depth+1)
    else: print(indent,type(obj))
inspect(cases)
PY
```
- **Container Exit Code**: `0` (`480 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/usr/bin/python3
numpy True
scipy True
pyamg False
 list 2
   ['problem', 'gt']
   problem
     ['Lx', 'Ly', 'kappa', 'sigma', 'gamma_robin', 'candidate_sources', 'sensor_ij', 'nuisance_design', 'nuisance_ridge', 'source_ridge', 'Y_obs']
     Lx 1.4
     Ly 1.2
     kappa (72, 72) float64 (0.022, 75.0)
     sigma (72, 72) float64 (0.06048611111111111, 0.12951388888888887)
     gamma_robin 1.65
     candidate_sources (10, 72, 72) float64 (2.415429504536423e-55, 0.9999489669611482)
     sensor_ij (360, 2) int64 (0, 71)
     nuisance_design (360, 4) float64 (-0.9976214598368016, 1.0)
     nuisance_ridge 0.001
     source_ridge 0.025
     Y_obs (360, 5) float64 (-1.7728113401552967, 1.2717565657315537)
   gt
     ['basis_fields', 'single_source_profiled_amplitudes', 'joint_source_weights']
     basis_fields (10, 72, 72) float64 (6.252411805741707e-05, 0.024076828454340525)
     single_source_profiled_amplitudes (10, 5) float64 (-4.3660760098064815, 3.925287606787953)
     joint_source_weights (10, 5) float64 (-0.346503170240279, 0.27569689064696035)
   ['problem', 'gt']
   problem
     ['Lx', 'Ly', 'kappa', 'sigma', 'gamma_robin', 'candidate_sources', 'sensor_ij', 'nuisance_design', 'nuisance_ridge', 'source_ridge', 'Y_obs']
     Lx 1.4
     Ly 1.2
     kappa (72, 72) float64 (0.022, 75.0)
     sigma (72, 72) float64 (0.06048611111111111, 0.12951388888888887)
     gamma_robin 1.65
     candidate_sources (10, 72, 72) float64 (5.478021844428302e-52, 0.9999808025664702)
     sensor_ij (360, 2) int64 (0, 71)
     nuisance_design (360, 4) float64 (-0.999524110790929, 1.0)
     nuisance_ridge 0.001
     source_ridge 0.025
     Y_obs (360, 5) float64 (-0.7309667979431722, 1.5510708159955666)
   gt
     ['basis_fields', 'single_source_profiled_amplitudes', 'joint_source_weights']
     basis_fields (10, 72, 72) float64 (6.758480349516637e-05, 0.041475484792134386)
     single_source_profiled_amplitudes (10, 5) float64 (-4.081232224999073, 2.456454688516775)
     joint_source_weights (10, 5) float64 (-0.38191522699830227, 0.474593190792141)
```

### Turn 3 — `execute_in_remote_docker` (`62161 ms` LLM)
- **Rationale**: Implement the Robin stencil, shared preconditioned CG, and augmented nuisance projection. Comparing all three outputs with the public references will detect stencil, indexing, or objective errors before submission.
- **Alternatives Weighed**: `Sparse direct solve` (rejected: Shared incomplete-Cholesky preconditioning preserves the intended multi-RHS CG approach.)

```bash
cat > /holdout_task/robin_spd_profiler.py <<'PY'
"""Cell-centered Robin diffusion solves and nuisance-profiled source fits."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve_triangular


def _operator(problem):
    k = np.asarray(problem['kappa'], dtype=float)
    nx, ny = k.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    g = problem['gamma_robin']

    kl, kr = k[:-1, :], k[1:, :]
    tx = (hy / hx) * (2 * kl * kr / (kl + kr))
    kl, kr = k[:, :-1], k[:, 1:]
    ty = (hx / hy) * (2 * kl * kr / (kl + kr))

    diag = area * np.asarray(problem['sigma'], dtype=float) + np.zeros_like(k)
    diag[:-1, :] += tx
    diag[1:, :] += tx
    diag[:, :-1] += ty
    diag[:, 1:] += ty
    for i in (0, nx - 1):
        diag[i, :] += hy * (g * k[i, :]) / (g * 0.5 * hx + k[i, :])
    for j in (0, ny - 1):
        diag[:, j] += hx * (g * k[:, j]) / (g * 0.5 * hy + k[:, j])

    grid = np.arange(k.size).reshape(k.shape)
    row = np.r_[grid[1:, :].ravel(), grid[:, 1:].ravel()]
    col = np.r_[grid[:-1, :].ravel(), grid[:, :-1].ravel()]
    off = -np.r_[tx.ravel(), ty.ravel()]
    flat = grid.ravel()
    A = coo_matrix((np.r_[diag.ravel(), off, off],
                    (np.r_[flat, row, col], np.r_[flat, col, row])),
                   shape=(k.size, k.size)).tocsr()

    # Zero-fill incomplete LDL^T. The grid has no triangles, so retained
    # off-diagonals need only be divided by their corresponding pivot.
    pivots = diag.copy()
    for i in range(nx):
        for j in range(ny):
            if i:
                pivots[i, j] -= tx[i - 1, j] ** 2 / pivots[i - 1, j]
            if j:
                pivots[i, j] -= ty[i, j - 1] ** 2 / pivots[i, j - 1]
    d = pivots.ravel()
    lower = coo_matrix((off / d[col], (row, col)), shape=A.shape).tocsr()
    return A, area, lower, d


def _solve_fields(A, B, lower, d):
    """Independent CG recurrences, sharing matrix and triangular solves."""
    upper = lower.T.tocsr()

    def precondition(r):
        t = spsolve_triangular(lower, r, lower=True, unit_diagonal=True)
        return spsolve_triangular(upper, t / d[:, None], lower=False,
                                  unit_diagonal=True)

    X = np.zeros_like(B)
    targets = 1e-11 * np.sqrt(np.sum(B * B, axis=0))
    active = np.flatnonzero(targets > 0)
    # Recompute the true residual whenever the recurrence reports convergence.
    # If round-off caused a premature stop, restart only that column.
    while active.size:
        x = X[:, active]
        r = B[:, active] - A @ x
        z = precondition(r)
        p = z.copy()
        rz = np.einsum('ij,ij->j', r, z)
        restart = []
        for _ in range(A.shape[0]):
            ap = A @ p
            a = rz / np.einsum('ij,ij->j', p, ap)
            x += p * a
            r -= ap * a
            done = np.sqrt(np.sum(r * r, axis=0)) <= targets[active]
            if np.any(done):
                ids = active[done]
                X[:, ids] = x[:, done]
                actual = B[:, ids] - A @ X[:, ids]
                # Accept well inside the specified tolerance even if the
                # recursive residual has fallen below the precision floor.
                restart.extend(ids[np.sqrt(np.sum(actual * actual, axis=0))
                                   > 1e-10 * np.sqrt(np.sum(B[:, ids] ** 2, axis=0))])
                keep = ~done
                if not np.any(keep):
                    break
                active, x, r, p, rz = active[keep], x[:, keep], r[:, keep], p[:, keep], rz[keep]
            z = precondition(r)
            next_rz = np.einsum('ij,ij->j', r, z)
            p = z + p * (next_rz / rz)
            rz = next_rz
        else:
            raise RuntimeError('Preconditioned CG did not converge')
        active = np.asarray(restart, dtype=int)
    return X


def _profile(S, Z, Y, lam_z, lam_s):
    # Augment the data term so nuisance ridge is an ordinary least-squares
    # penalty. Orthogonal residualization applies the same projection to S/Y.
    m, q, p = S.shape[1], Y.shape[1], Z.shape[1]
    data = np.vstack((np.column_stack((S, Y)), np.zeros((p, m + q))))
    if p:
        nuisance = np.vstack((Z, np.sqrt(lam_z) * np.eye(p)))
        data -= nuisance @ np.linalg.lstsq(nuisance, data, rcond=None)[0]
    Sr, Yr = data[:, :m], data[:, m:]

    num = Sr.T @ Yr
    den = np.einsum('ij,ij->j', Sr, Sr)[:, None]
    single = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    joint_design = np.vstack((Sr, np.sqrt(lam_s) * np.eye(m)))
    joint_data = np.vstack((Yr, np.zeros((m, q))))
    joint = np.linalg.lstsq(joint_design, joint_data, rcond=None)[0]
    return single, joint


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Solve the finite-volume PDE for each candidate, then fit sensor data."""
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    A, area, lower, d = _operator(problem)
    rhs = area * sources.reshape(sources.shape[0], A.shape[0]).T
    fields = _solve_fields(A, rhs, lower, d).T.reshape(sources.shape)
    sensors = np.asarray(problem['sensor_ij'])
    S = fields[:, sensors[:, 0], sensors[:, 1]].T
    single, joint = _profile(S, np.asarray(problem['nuisance_design']),
                            np.asarray(problem['Y_obs']), problem['nuisance_ridge'],
                            problem['source_ridge'])
    return {'basis_fields': fields,
            'single_source_profiled_amplitudes': single,
            'joint_source_weights': joint}
PY
cd /holdout_task; PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import pickle, time
import numpy as np
from robin_spd_profiler import solve_robin_spd_and_profile_sources
with open('public_robin_spd_cases.pkl','rb') as f:
    cases=pickle.load(f)
for i, case in enumerate(cases):
    t=time.perf_counter()
    out=solve_robin_spd_and_profile_sources(case['problem'])
    print(i+1, 'seconds', time.perf_counter()-t)
    for k,v in out.items():
        ref=case['gt'][k]
        print(k, v.shape, 'max abs error', np.max(np.abs(v-ref)))
        np.testing.assert_allclose(v, ref, rtol=1e-7, atol=1e-9)
PY
```
- **Container Exit Code**: `124` (`180104 ms`, timed_out=`True`)
- **Filesystem Touched**: `['/holdout_task/robin_spd_profiler.py']`
- **Stderr (tail)**:
```text
[TIMEOUT: command exceeded 180s limit]
```

### Turn 4 — `execute_in_remote_docker` (`18447 ms` LLM)
- **Rationale**: Distinguish expensive triangular substitutions from slow CG convergence or a threaded-library slowdown. The result will determine whether to replace the preconditioner application or adjust convergence handling.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout 20s python3 -u - <<'PY'
import pickle, time
import numpy as np, scipy
import robin_spd_profiler as m
print('scipy',scipy.__version__,flush=True)
p=pickle.load(open('public_robin_spd_cases.pkl','rb'))[0]['problem']; A,area,L,d=m._operator(p); B=area*p['candidate_sources'].reshape(10,-1).T
u=L.T.tocsr(); t=time.perf_counter()
for _ in range(3):
 z=m.spsolve_triangular(L,B,lower=True,unit_diagonal=True)
 z=m.spsolve_triangular(u,z/d[:,None],lower=False,unit_diagonal=True)
print('triangular pair seconds',(time.perf_counter()-t)/3, 'pivots',d.min(),d.max(),flush=True)
r=B.copy(); p0=None
for i in range(710):
 z=m.spsolve_triangular(u,m.spsolve_triangular(L,r,lower=True,unit_diagonal=True)/d[:,None],lower=False,unit_diagonal=True)
 rz=np.sum(r*z,axis=0); p0=z if i==0 else z+p0*(rz/old); ap=A@p0; r-=ap*(rz/np.sum(p0*ap,axis=0)); old=rz
 if i in (0,9,49,99,199,399,699): print(i,np.linalg.norm(r,axis=0)/np.linalg.norm(B,axis=0),time.perf_counter()-t,flush=True)
PY
```
- **Container Exit Code**: `124` (`20481 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
scipy 1.13.0
triangular pair seconds 0.19972903498758873 pivots 0.06931911510685117 257.8517722645011
0 [1.0118793  1.00566264 1.00623059 1.02466424 1.00794439 1.00003606
 1.0085107  1.00233456 1.02563679 1.00100665] 0.8001558189280331
9 [1.01226788 1.00581344 1.00640236 1.02605518 1.00818942 1.00003613
 1.00878067 1.00236632 1.02709763 1.00101712] 2.5976524390280247
49 [1.01226788 1.20738375 1.00640236 1.02605518 1.00818942 1.00003613
 1.00878067 1.00236632 1.02709763 1.00101712] 10.004241549875587
```

### Turn 5 — `execute_in_remote_docker` (`24284 ms` LLM)
- **Rationale**: Replace repeated slow triangular calls with reusable compiled substitutions and test both reference agreement and the true per-source residual. If the change does not restore convergence, inspect the preconditioned recurrence separately.

```bash
cd /holdout_task && python3 - <<'PY'
from pathlib import Path
p = Path('robin_spd_profiler.py')
s = p.read_text().replace('from scipy.sparse import coo_matrix','from scipy.sparse import coo_matrix, eye').replace('from scipy.sparse.linalg import spsolve_triangular','from scipy.sparse.linalg import splu')
a = s.index('    upper = lower.T.tocsr()')
b = s.index('\n    X =', a)
s = s[:a] + '''    # Factor the unit triangular matrix once to reuse compiled sparse
    # forward and transpose substitutions for every CG iteration.
    factor = splu(lower.tocsc() + eye(A.shape[0], format='csc'),
                  permc_spec='NATURAL', diag_pivot_thresh=0)

    def precondition(r):
        return factor.solve(factor.solve(r) / d[:, None], trans='T')
''' + s[b:]
p.write_text(s)
PY
PYTHONDONTWRITEBYTECODE=1 python3 -u - <<'PY'
import pickle, time
import numpy as np
import robin_spd_profiler as m
with open('public_robin_spd_cases.pkl','rb') as f:
    cases = pickle.load(f)
for i, case in enumerate(cases, 1):
    p = case['problem']
    t = time.perf_counter()
    out = m.solve_robin_spd_and_profile_sources(p)
    print(i, 'seconds', time.perf_counter()-t)
    A, area, _, _ = m._operator(p)
    B = area*p['candidate_sources'].reshape(10, -1).T
    R = A @ out['basis_fields'].reshape(10,-1).T - B
    err = max(np.linalg.norm(R,axis=0)/np.linalg.norm(B,axis=0))
    print('max PDE residual', err)
    assert err <= 1e-8
    for k, v in out.items():
        ref = case['gt'][k]
        print(k, v.shape, 'max abs discrepancy', np.max(np.abs(v-ref)))
        np.testing.assert_allclose(v, ref, rtol=1e-7, atol=1e-9)
PY
```
- **Container Exit Code**: `0` (`3852 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin_spd_profiler.py']`
- **Stdout (tail)**:
```text
1 seconds 0.9951376467943192
max PDE residual 2.389441203634722e-11
basis_fields (10, 72, 72) max abs discrepancy 6.6188374225895075e-15
single_source_profiled_amplitudes (10, 5) max abs discrepancy 1.517452830057664e-12
joint_source_weights (10, 5) max abs discrepancy 1.4084566846150892e-13
2 seconds 0.9290929231792688
max PDE residual 2.545565268273765e-11
basis_fields (10, 72, 72) max abs discrepancy 5.9396931817445875e-15
single_source_profiled_amplitudes (10, 5) max abs discrepancy 1.397104654188297e-12
joint_source_weights (10, 5) max abs discrepancy 1.2434497875801753e-13
```

### Turn 6 — `submit_completed_task` (`9968 ms` LLM)
- **Rationale**: The required module is saved in the container and verified against both public cases.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/robin_spd_profiler.py` with shared preconditioned CG and nuisance-profiled source fitting. On both public cases, all three outputs matched the references, and the worst relative PDE residual was 2.55 × 10⁻¹¹, below the required 10⁻⁸.`

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
