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
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `140,593` (`134,338` / `51,584` / `6,255`) |
| **Total Episode Wall Time (s)** | `218.86s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`18963 ms` LLM)
- **Rationale**: Inspect the public input layout to catch mismatches in grid and RHS ordering, and check container limits to choose a representative timing test for the two held-out solves.
- **Alternatives Weighed**: `Start with an external AMG dependency` (rejected: Availability is unverified, and reusable sparse LU is explicitly allowed; benchmark it before adding a dependency.)

```bash
cd /holdout_task && ls -l && python3 -I - <<'PY'
import os, pickle, importlib.util
import numpy as np, scipy
print('numpy/scipy:', np.__version__, scipy.__version__)
print('optional packages:', {n: importlib.util.find_spec(n) is not None for n in ('pyamg','threadpoolctl')})
for name in ('/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max'):
    if os.path.exists(name): print(name, open(name).read().strip())
with open('public_robin3d_cases.pkl','rb') as f: data=pickle.load(f)
def describe(x, depth=0):
    if isinstance(x,dict):
        for k,v in x.items():
            if isinstance(v,np.ndarray): print(k,v.shape,v.dtype, (float(v.min()),float(v.max())) if v.size and v.dtype.kind in 'iuf' else '')
            elif np.isscalar(v): print(k,repr(v))
            elif depth<2: print(k,type(v).__name__); describe(v,depth+1)
    elif isinstance(x,(list,tuple)):
        print(type(x).__name__,len(x))
        for v in x[:3]: describe(v,depth+1)
print(type(data).__name__); describe(data)
PY
```
- **Container Exit Code**: `0` (`588 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 1160
-rw-r--r-- 1 root root 1184036 Nov 26 01:38 public_robin3d_cases.pkl
numpy/scipy: 1.26.4 1.13.0
optional packages: {'pyamg': False, 'threadpoolctl': True}
/sys/fs/cgroup/cpu.max 200000 100000
/sys/fs/cgroup/memory.max 2684354560
list
list 1
problem dict
Lx 1.2
Ly 1.1
Lz 1.0
kappa (20, 20, 20) float64 (0.02, 80.0)
sigma (20, 20, 20) float64 (0.052250000000000005, 0.13775)
gamma_robin 1.5
candidate_sources (8, 20, 20, 20) float64 (1.1240851771000825e-25, 0.9979583217974671)
sensor_ijk (320, 3) int64 (0.0, 19.0)
nuisance_design (320, 5) float64 (-0.9938441702975689, 1.0)
nuisance_ridge 0.0015
source_ridge 0.025
Y_obs (320, 4) float64 (-0.9227534433119405, 1.7143269201683184)
gt dict
basis_fields (8, 20, 20, 20) float64 (1.9923077389427656e-05, 0.02396656177460551)
single_source_profiled_amplitudes (8, 4) float64 (-3.3915227444823284, 2.2876400954505343)
joint_source_weights (8, 4) float64 (-0.10797686395976767, 0.05455092839187196)
```

### Turn 2 — `execute_in_remote_docker` (`52015 ms` LLM)
- **Rationale**: Compare every public output with its reference to detect discretization or profiling errors, then measure random and connected 4,000× contrast at 28³ to decide whether the reused factorization meets the residual and runtime budget.
- **Alternatives Weighed**: `Implement a custom AMG hierarchy first` (rejected: PyAMG is unavailable, and reusable sparse LU is explicitly permitted; benchmark it at full scale before adding solver complexity.)

```bash
cat > /holdout_task/robin3d_amg_tomography.py <<'PY'
"""Cell-centered 3D Robin diffusion and nuisance-profiled source inversion."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _assemble(kappa, sigma, spacing, gamma):
    """Assemble the integrated cell balance in C-order indexing."""
    volume = np.prod(spacing)
    cells = np.arange(kappa.size).reshape(kappa.shape)
    diagonal = (volume * sigma).copy()
    rows, cols, values = [], [], []
    for axis, h in enumerate(spacing):
        area = volume / h
        lo = [slice(None)] * 3
        hi = lo.copy()
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        kl, kr = kappa[lo], kappa[hi]
        conductance = (area / h) * (2 * kl * kr / (kl + kr))
        diagonal[lo] += conductance
        diagonal[hi] += conductance
        left, right = cells[lo].ravel(), cells[hi].ravel()
        rows.extend((left, right))
        cols.extend((right, left))
        values.extend((-conductance.ravel(), -conductance.ravel()))
        for edge in (0, -1):
            boundary = [slice(None)] * 3
            boundary[axis] = edge
            boundary = tuple(boundary)
            kb = kappa[boundary]
            diagonal[boundary] += area * gamma * kb / (kb + 0.5 * h * gamma)
    rows.append(cells.ravel())
    cols.append(cells.ravel())
    values.append(diagonal.ravel())
    return coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(kappa.size, kappa.size)).tocsc()


def _profile(S, Z, Y, lam_z, lam_s):
    """Project out nuisance columns in the ridge-augmented observation space."""
    m, p = S.shape[1], Z.shape[1]
    data = np.column_stack((S, Y))
    if p:
        design = np.vstack((Z, np.sqrt(lam_z) * np.eye(p)))
        data = np.vstack((data, np.zeros((p, data.shape[1]))))
        nuisance = np.linalg.lstsq(design, data, rcond=None)[0]
        data -= design @ nuisance
    s, y = data[:, :m], data[:, m:]
    den = np.einsum('ij,ij->j', s, s)
    num = s.T @ y
    # A source fully in the nuisance span has no identifiable single amplitude.
    tol = (np.finfo(float).eps * max(data.shape) * np.linalg.norm(S, axis=0)) ** 2
    single = np.divide(num, den[:, None], out=np.zeros_like(num),
                       where=(den > tol)[:, None])
    if lam_s:
        s = np.vstack((s, np.sqrt(lam_s) * np.eye(m)))
        y = np.vstack((y, np.zeros((m, y.shape[1]))))
    joint = np.linalg.lstsq(s, y, rcond=None)[0]
    return single, joint


def solve_3d_robin_tomography(problem: dict) -> dict:
    """Solve all candidate responses and fit the two profiled source models."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    spacing = np.array([problem['Lx'], problem['Ly'], problem['Lz']]) / kappa.shape
    A = _assemble(kappa, np.asarray(problem['sigma'], dtype=float), spacing,
                  problem['gamma_robin'])
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    rhs = np.asfortranarray(np.prod(spacing) * sources.reshape(len(sources), -1).T)
    factor = splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                  options={'SymmetricMode': True})
    solution = factor.solve(rhs)
    gap = rhs - A @ solution
    refine = np.linalg.norm(gap, axis=0) > 1e-8 * np.linalg.norm(rhs, axis=0)
    if np.any(refine):
        solution[:, refine] += factor.solve(gap[:, refine])
    fields = solution.T.reshape(sources.shape)
    i, j, k = np.asarray(problem['sensor_ijk']).T
    single, joint = _profile(fields[:, i, j, k].T,
                             np.asarray(problem['nuisance_design'], dtype=float),
                             np.asarray(problem['Y_obs'], dtype=float),
                             problem['nuisance_ridge'], problem['source_ridge'])
    return {'basis_fields': fields,
            'single_source_profiled_amplitudes': single,
            'joint_source_weights': joint}
PY
python3 -I - <<'PY'
import importlib.util, pickle, time, os, resource
import numpy as np
from threadpoolctl import threadpool_info
path='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver', path+'robin3d_amg_tomography.py'); solver=importlib.util.module_from_spec(spec); spec.loader.exec_module(solver)
print(threadpool_info(),flush=True)
with open(path+'public_robin3d_cases.pkl','rb') as f: case=pickle.load(f)[0]
p=case['problem']; t=time.perf_counter(); out=solver.solve_3d_robin_tomography(p)
print('public seconds',time.perf_counter()-t,flush=True)
for key, val in out.items():
    ref=case['gt'][key]; print(key, val.shape, 'max_abs',np.max(abs(val-ref)), 'relative',np.linalg.norm(val-ref)/np.linalg.norm(ref),flush=True)
rng=np.random.default_rng(17); n=28
for mode in ('random','channels'):
    i,j,k=np.indices((n,n,n))
    kap=np.where(rng.random((n,n,n))<0.55,80.,.02) if mode=='random' else np.where(((i//3+j//3)%3==0) | ((j//3+k//4)%3==0),80.,.02)
    q=rng.standard_normal((8,n,n,n)); q[:4]+=1
    p=dict(p,kappa=kap,sigma=np.full(kap.shape,.06),candidate_sources=q,sensor_ijk=rng.integers(n,size=(320,3)))
    t=time.perf_counter(); out=solver.solve_3d_robin_tomography(p); elapsed=time.perf_counter()-t
    h=np.array([p['Lx'],p['Ly'],p['Lz']])/n; v=np.prod(h)
    A=solver._assemble(kap,p['sigma'],h,p['gamma_robin']); b=v*q.reshape(8,-1).T
    rr=np.linalg.norm(A@out['basis_fields'].reshape(8,-1).T-b,axis=0)/np.linalg.norm(b,axis=0)
    print(mode,'seconds',elapsed,'worst residual',max(rr),'rss KiB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
PY
```
- **Container Exit Code**: `0` (`6102 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin3d_amg_tomography.py']`
- **Stdout (tail)**:
```text
[{'user_api': 'blas', 'internal_api': 'openblas', 'num_threads': 32, 'prefix': 'libopenblas', 'filepath': '/usr/local/lib/python3.12/dist-packages/scipy.libs/libopenblasp-r0-24bff013.3.26.dev.so', 'version': '0.3.26.dev', 'threading_layer': 'pthreads', 'architecture': 'Cooperlake'}, {'user_api': 'blas', 'internal_api': 'openblas', 'num_threads': 32, 'prefix': 'libopenblas', 'filepath': '/usr/local/lib/python3.12/dist-packages/numpy.libs/libopenblas64_p-r0-0cf96a72.3.23.dev.so', 'version': '0.3.23.dev', 'threading_layer': 'pthreads', 'architecture': 'Cooperlake'}]
public seconds 0.10634034709073603
basis_fields (8, 20, 20, 20) max_abs 2.732189474663471e-16 relative 2.6564976384719754e-14
single_source_profiled_amplitudes (8, 4) max_abs 3.991251773527438e-13 relative 1.081842862686877e-13
joint_source_weights (8, 4) max_abs 5.147271497918382e-14 relative 5.072790352530213e-13
random seconds 2.514743474079296 worst residual 3.5975971955431573e-12 rss KiB 152460
channels seconds 2.3856723660137504 worst residual 5.707391701435255e-12 rss KiB 157680

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2171af895249973e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2171af89 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`38070 ms` LLM)
- **Rationale**: The first full-scale residual check reused the assembly routine. An independent face-flux balance on an anisotropic grid can catch a shared stencil error, and direct augmented fits can catch a profiling error. If either fails, fix that component; otherwise submit without changing the artifact.
- **Alternatives Weighed**: `Replace the reused LU with custom AMG` (rejected: The allowed LU path meets the timing and residual limits with ample margin; no measured bottleneck justifies a second solver.)

```bash
python3 -I - <<'PY'
import importlib.util
import pickle
import time
import numpy as np
from scipy.linalg import lstsq

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('robin_solver', root + 'robin3d_amg_tomography.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open(root + 'public_robin3d_cases.pkl', 'rb') as f:
    public = pickle.load(f)[0]['problem']
rng = np.random.default_rng(9301)
n = 28
x, y, z = np.indices((n, n, n))
seconds = 0.0
worst = 0.0
for label, kappa, gamma in (
    ('connected high-conductivity bulk', np.where((x >= 8) & (x < 20) & (y >= 8) & (y < 20) & (z >= 8) & (z < 20), .02, 80.), .25),
    ('insulating shell', np.where(np.minimum.reduce([x,y,z,n-1-x,n-1-y,n-1-z]) < 2, .02, 80.), 2.5),
):
    lengths = np.array([1.3, .92, .74]); h = lengths / n
    sigma = .05225 + .0855*x/(n-1)
    q = rng.normal(size=(8,n,n,n)); q[0] = 1.; q[1] += 1.
    q[2] = 0.; q[2,0,0,0] = 1.
    sensors = rng.integers(n, size=(320,3))
    p = dict(public, Lx=lengths[0], Ly=lengths[1], Lz=lengths[2], kappa=kappa,
             sigma=sigma, gamma_robin=gamma, candidate_sources=q, sensor_ijk=sensors)
    start = time.perf_counter()
    out = solver.solve_3d_robin_tomography(p)
    elapsed = time.perf_counter() - start
    seconds += elapsed
    u = out['basis_fields']
    # Evaluate the differential balance directly from the field, without the solver matrix.
    balance = sigma * u
    for axis, step in enumerate(h):
        kk = np.moveaxis(kappa,axis,-1)
        uu = np.moveaxis(u,axis+1,-1)
        rr = np.moveaxis(balance,axis+1,-1)
        flux = np.diff(uu,axis=-1) / ((.5/kk[...,1:] + .5/kk[...,:-1])*step**2)
        rr[...,:-1] -= flux
        rr[...,1:] += flux
        for side in (0,-1):
            rr[...,side] += uu[...,side] / (step*(1/gamma + .5*step/kk[...,side]))
    resid = np.linalg.norm((balance-q).reshape(8,-1),axis=1)/np.linalg.norm(q.reshape(8,-1),axis=1)
    worst = max(worst,float(max(resid)))
    assert np.all(resid <= 1e-7), resid
    S = u[:,*sensors.T].T
    Z, Y = p['nuisance_design'], p['Y_obs']
    k, m = S.shape; nz = Z.shape[1]
    def reference(cols, source_ridge):
        count = cols.shape[1]
        design = np.block([[Z,cols],[np.sqrt(p['nuisance_ridge'])*np.eye(nz),np.zeros((nz,count))],[np.zeros((count,nz)),np.sqrt(source_ridge)*np.eye(count)]])
        return lstsq(design,np.vstack((Y,np.zeros((nz+count,Y.shape[1])))))[0][nz:]
    errors = []
    for key, ref in [('single_source_profiled_amplitudes',np.vstack([reference(S[:,i:i+1],0.) for i in range(m)])),('joint_source_weights',reference(S,p['source_ridge']))]:
        assert out[key].shape == ref.shape
        np.testing.assert_allclose(out[key],ref,rtol=1e-8,atol=1e-9)
        errors.append(float(np.max(np.abs(out[key]-ref))))
    print(f'{label}: {elapsed:.3f}s; max PDE residual {max(resid):.3g}; max independent single/joint errors {errors}')
assert seconds < 14.0, seconds
print(f'two-case total {seconds:.3f}s; worst independent PDE residual {worst:.3g}')
PY
```
- **Container Exit Code**: `0` (`6517 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
connected high-conductivity bulk: 2.855s; max PDE residual 5.88e-11; max independent single/joint errors [9.549694368615746e-12, 1.5230872119076366e-15]
insulating shell: 2.489s; max PDE residual 4.84e-11; max independent single/joint errors [3.4861002973229915e-13, 8.248610128269718e-16]
two-case total 5.344s; worst independent PDE residual 5.88e-11

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2171af895249973e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2171af89 (directory).
```

### Turn 4 — `submit_completed_task` (`14154 ms` LLM)
- **Rationale**: The saved artifact matched all public reference outputs. An isolated Python run independently checked the face-flux balance and both tomography fits on full-size connected-bulk and insulating-shell cases.
- **Alternatives Weighed**: `Replace sparse LU with custom AMG` (rejected: Reusing one factorization per problem is explicitly allowed and meets the tested runtime and residual limits with ample margin.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/robin3d_amg_tomography.py': 'FILE:3939'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin3d_amg_tomography.py`. Two 28³ high-contrast stress cases completed in 5.344 s combined, below the 14 s budget. The worst relative PDE residual was 5.88e-11, below the 1e-7 limit. Both tomography outputs matched independent least-squares fits.`

### Turn 5 — `execute_in_remote_docker` (`30811 ms` LLM)
- **Rationale**: On the provided public observations, check whether the six Robin faces and reaction account for the total source and whether both fits satisfy the original penalized objective after recovering nuisance coefficients. A failure would identify which component to revise; a pass leaves the verified artifact unchanged.
- **Alternatives Weighed**: `Batch-centered classification and relay-mode checks` (rejected: The input contains continuous observations and an explicit nuisance design, not classification labels, batch identifiers, or relay states.)

```bash
python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('robin', root + 'robin3d_amg_tomography.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open(root + 'public_robin3d_cases.pkl', 'rb') as f:
    p = pickle.load(f)[0]['problem']
out = solver.solve_3d_robin_tomography(p)
u = out['basis_fields']
k = p['kappa']
h = np.array([p['Lx'], p['Ly'], p['Lz']]) / k.shape
v = h.prod()
g = p['gamma_robin']
# The interior pairwise fluxes cancel in the global balance.
loss = v * np.sum(p['sigma'] * u, axis=(1, 2, 3))
for a, step in enumerate(h):
    for side in (0, -1):
        kb = np.take(k, side, axis=a)
        ub = np.take(u, side, axis=a+1)
        loss += (v / step) * np.sum(ub / (1/g + .5*step/kb), axis=(1, 2))
load = v * np.sum(p['candidate_sources'], axis=(1, 2, 3))
conservation = np.max(np.abs(loss-load)/np.abs(load))
assert conservation < 1e-7, conservation

i, j, k = p['sensor_ijk'].T
S = u[:, i, j, k].T
Z, Y = p['nuisance_design'], p['Y_obs']
lam_z, lam_s = p['nuisance_ridge'], p['source_ridge']
gram = Z.T @ Z + lam_z * np.eye(Z.shape[1])
single = out['single_source_profiled_amplitudes']
joint = out['joint_source_weights']

def stationarity(design, weights, ridge):
    unexplained = Y - design @ weights
    nuisance = np.linalg.solve(gram, Z.T @ unexplained)
    grad = design.T @ (Z @ nuisance - unexplained) + ridge * weights
    scale = np.linalg.norm(design, axis=0)[:, None] * np.linalg.norm(Y, axis=0)[None, :]
    return np.max(np.abs(grad)/scale)

single_gap = max(stationarity(S[:, m:m+1], single[m:m+1], 0.) for m in range(S.shape[1]))
joint_gap = stationarity(S, joint, lam_s)
assert max(single_gap, joint_gap) < 1e-10, (single_gap, joint_gap)
raw = (S.T @ Y) / np.sum(S*S, axis=0)[:, None]
print(f'Public case: max global balance error {conservation:.3g}; normalized stationarity single {single_gap:.3g}, joint {joint_gap:.3g}; max amplitude shift from accounting for nuisance {np.max(np.abs(single-raw)):.3g}')
PY
```
- **Container Exit Code**: `0` (`1049 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public case: max global balance error 1.18e-13; normalized stationarity single 6.77e-16, joint 5.07e-16; max amplitude shift from accounting for nuisance 195

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2171af895249973e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_2171af89 (directory).
```

### Turn 6 — `submit_completed_task` (`15915 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification, independent full-size face-flux and least-squares checks, and a public-case conservation and stationarity audit. The audit found no defect that warrants another edit.
- **Alternatives Weighed**: `Replace reusable sparse LU with custom AMG` (rejected: The specification permits reusable LU, and the full-size stress tests showed ample margin on both runtime and residual.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/robin3d_amg_tomography.py': 'FILE:3939'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin3d_amg_tomography.py`. Two 28³ high-contrast stress cases took 5.344 s combined against a 14 s budget. The worst relative PDE residual was 5.88e-11 against a 1e-7 limit. Both tomography outputs matched independent least-squares fits.`

### Turn 7 — `submit_completed_task` (`14372 ms` LLM)
- **Rationale**: Independent checks verified the stencil and both fitted outputs, and full-size stress tests left ample residual and runtime margin. Batch classification and relay modes do not apply to this continuous inverse problem.
- **Alternatives Weighed**: `Repeat the public conservation and nuisance audit` (rejected: The unchanged artifact already passed that audit in isolated Python, with global balance error of 1.18e-13 and normalized fit stationarity error of 6.77e-16.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin3d_amg_tomography.py`. Two 28³ high-contrast stress cases took 5.344 s combined against the 14 s budget. The worst relative PDE residual was 5.88e-11 against the 1e-7 limit. Both tomography outputs matched independent least-squares fits.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-M91-HardDemand-2",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/robin3d_amg_tomography.py": "FILE:3939"
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
    "student_elapsed_sec": 2.7327
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"all_valid\": true, \"max_pde_rel_residual\": 0.0, \"max_field_rel_l2\": 0.0, \"max_profiled_amp_rel_l2\": 0.0, \"max_joint_weight_rel_l2\": 0.0, \"student_elapsed_sec\": 2.7327}\n__HEGEL_VERIFIER_AUTH_f249c1d0c9229f274dc5fc32c05b4a3c__:{\"strict_pass\": true, \"all_valid\": true, \"max_pde_rel_residual\": 0.0, \"max_field_rel_l2\": 0.0, \"max_profiled_amp_rel_l2\": 0.0, \"max_joint_weight_rel_l2\": 0.0, \"student_elapsed_sec\": 2.7327}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 15140.66,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
