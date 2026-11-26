# Dialectical Synthesis Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_2_Selective_LiveGPT` | `vr02__large_robin_spd_preconditioned_cg_multi_rhs`

- **Official Strict Pass**: **`PASS (1.0)`** (`PASSED_ALL_CHECKS`)
- **Blind Public Synthesis Choice**: `antithesis` via `dialectical_trajectory_synthesis_medium`
- **Synthesis Rationale**: [blinded_slot:candidate_b->antithesis] Both candidates returned all three outputs with the required shapes and matched both public references. A reported a maximum absolute reference error of 1.64e-12 and passed its assembled-matrix residual check without printing the residuals. B independently reconstructed the finite-volume flux balance and measured public residuals of 5.463e-12 and 5.466e-12. Its worst residual across two anisotropic high-contrast perturbations was 1.381e-10, below the 1e-8 limit. Both of B's coefficient outputs agreed with independent augmented least-squares fits within 9e-15. | B's independent flux check can catch a stencil error that a residual computed with the solver's own matrix would miss. Its separate least-squares checks also verify the two distinct penalty rules: no source penalty for the single-source fits, and source ridge for the joint fit. The mechanical metrics contain no results, and the reported timings are not directly comparable, so runtime does not decide the selection.
- **Post-Lock Branch Verdicts**: Thesis (`Bare`) = `True` (4T) | Antithesis (`Wormhole`) = `True` (7T)

## Selected Branch Trajectory Record
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
| **Turns Used / Max Turns** | `7 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `136,257` (`130,710` / `27,136` / `5,547`) |
| **Total Episode Wall Time (s)** | `183.22s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`19784 ms` LLM)
- **Rationale**: Inspect the actual input layout, coefficient ranges, and available libraries before selecting the solver and validation slices.

```bash
ls -la /holdout_task; python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
for m in ('numpy','scipy'):
    spec=importlib.util.find_spec(m)
    print(m, spec.origin if spec else 'unavailable')
with open('/holdout_task/public_robin_spd_cases.pkl','rb') as f:
    data=pickle.load(f)
print('root:',type(data).__name__,'count:',len(data))
if isinstance(data,dict):
    print('root keys:',list(data)[:20])
    seq=data.get('cases',list(data.values())) if 'kappa' not in data else [data]
else:
    seq=data
for i,c in enumerate(seq):
    if not isinstance(c,dict):
        print(i,type(c).__name__); continue
    print(i,'keys:',list(c))
    p=c.get('problem',c)
    if isinstance(p,dict) and 'kappa' in p:
        print({k:(v.shape,str(v.dtype)) if isinstance(v,np.ndarray) else v for k,v in p.items() if isinstance(v,np.ndarray) or np.isscalar(v)})
        print('kappa and sigma ranges:', [(float(np.min(p[k])),float(np.max(p[k]))) for k in ('kappa','sigma')])
PY
```
- **Container Exit Code**: `0` (`620 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 1856
drwxr-xr-x 2 root root    4096 Nov 26 00:46 .
drwxr-xr-x 1 root root    4096 Nov 26 00:46 ..
-rw-r--r-- 1 root root 1890946 Nov 26 00:46 public_robin_spd_cases.pkl
numpy /usr/local/lib/python3.12/dist-packages/numpy/__init__.py
scipy /usr/local/lib/python3.12/dist-packages/scipy/__init__.py
root: list count: 2
0 keys: ['problem', 'gt']
{'Lx': 1.4, 'Ly': 1.2, 'kappa': ((72, 72), 'float64'), 'sigma': ((72, 72), 'float64'), 'gamma_robin': 1.65, 'candidate_sources': ((10, 72, 72), 'float64'), 'sensor_ij': ((360, 2), 'int64'), 'nuisance_design': ((360, 4), 'float64'), 'nuisance_ridge': 0.001, 'source_ridge': 0.025, 'Y_obs': ((360, 5), 'float64')}
kappa and sigma ranges: [(0.022, 75.0), (0.06048611111111111, 0.12951388888888887)]
1 keys: ['problem', 'gt']
{'Lx': 1.4, 'Ly': 1.2, 'kappa': ((72, 72), 'float64'), 'sigma': ((72, 72), 'float64'), 'gamma_robin': 1.65, 'candidate_sources': ((10, 72, 72), 'float64'), 'sensor_ij': ((360, 2), 'int64'), 'nuisance_design': ((360, 4), 'float64'), 'nuisance_ridge': 0.001, 'source_ridge': 0.025, 'Y_obs': ((360, 5), 'float64')}
kappa and sigma ranges: [(0.022, 75.0), (0.06048611111111111, 0.12951388888888887)]
```

### Turn 2 — `execute_in_remote_docker` (`51494 ms` LLM)
- **Rationale**: Implement the conservative stencil and both inverse objectives, then check every public source mode for PDE error and compare both sets of fitted coefficients to the supplied references.
- **Alternatives Weighed**: `Preconditioned CG` (rejected: The stated interface does not mandate an iterative method, and a sparse factorization at this grid size can be reused across all ten right-hand sides.); `Schur complement formed from normal equations` (rejected: Orthogonal projection in the augmented space handles the same ridge objective without subtracting nearly equal Gram matrices.)

```bash
cat > /holdout_task/robin_spd_profiler.py <<'PY'
"""Cell-centered Robin diffusion solve and ridge-profiled source matching."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _robin_matrix(kappa, sigma, lx, ly, gamma):
    nx, ny = kappa.shape
    hx, hy = lx / nx, ly / ny
    cells = np.arange(nx * ny).reshape(nx, ny)
    wx = (hy / hx) * (2 * kappa[:-1, :] * kappa[1:, :] /
                      (kappa[:-1, :] + kappa[1:, :]))
    wy = (hx / hy) * (2 * kappa[:, :-1] * kappa[:, 1:] /
                      (kappa[:, :-1] + kappa[:, 1:]))

    diag = hx * hy * sigma
    diag[:-1, :] += wx
    diag[1:, :] += wx
    diag[:, :-1] += wy
    diag[:, 1:] += wy
    for i in (0, nx - 1):
        kb = kappa[i, :]
        diag[i, :] += hy * gamma * kb / (kb + 0.5 * hx * gamma)
    for j in (0, ny - 1):
        kb = kappa[:, j]
        diag[:, j] += hx * gamma * kb / (kb + 0.5 * hy * gamma)

    a = np.concatenate((cells[:-1, :].ravel(), cells[:, :-1].ravel()))
    b = np.concatenate((cells[1:, :].ravel(), cells[:, 1:].ravel()))
    w = np.concatenate((wx.ravel(), wy.ravel()))
    d = cells.ravel()
    matrix = coo_matrix((np.concatenate((diag.ravel(), -w, -w)),
                         (np.concatenate((d, a, b)), np.concatenate((d, b, a)))),
                        shape=(d.size, d.size)).tocsc()
    return matrix, hx * hy


def _remove_nuisance(z, data, ridge):
    # Project in the augmented data space so the norm includes the nuisance penalty.
    p = z.shape[1]
    if not p:
        return data
    if ridge > 0:
        z_aug = np.vstack((z, np.sqrt(ridge) * np.eye(p)))
        data = np.vstack((data, np.zeros((p, data.shape[1]))))
        q, _ = np.linalg.qr(z_aug, mode='reduced')
    else:
        u, singular, _ = np.linalg.svd(z, full_matrices=False)
        rank = np.count_nonzero(singular > singular[0] * max(z.shape) * np.finfo(float).eps)
        q = u[:, :rank]
    return data - q @ (q.T @ data)


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Solve all source PDEs and return scalar profiles and joint source weights."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    sigma = np.asarray(problem['sigma'], dtype=float)
    matrix, area = _robin_matrix(kappa, sigma, problem['Lx'], problem['Ly'],
                                 problem['gamma_robin'])
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    m = sources.shape[0]
    rhs = area * sources.reshape(m, kappa.size).T
    # Sparse factorization is shared across all source right-hand sides.
    lu = splu(matrix, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
              options={'SymmetricMode': True})
    basis = lu.solve(rhs).T.reshape(sources.shape)

    sensors = np.asarray(problem['sensor_ij'])
    s = basis[:, sensors[:, 0], sensors[:, 1]].T
    z = np.asarray(problem['nuisance_design'], dtype=float)
    y = np.asarray(problem['Y_obs'], dtype=float)
    profiled = _remove_nuisance(z, np.hstack((s, y)), problem['nuisance_ridge'])
    s, y = profiled[:, :m], profiled[:, m:]
    denom = np.sum(s * s, axis=0)[:, None]
    numer = s.T @ y
    single = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)

    ridge = problem['source_ridge']
    if ridge > 0:
        s = np.vstack((s, np.sqrt(ridge) * np.eye(m)))
        y = np.vstack((y, np.zeros((m, y.shape[1]))))
    joint = np.linalg.lstsq(s, y, rcond=None)[0]
    return {'basis_fields': basis,
            'single_source_profiled_amplitudes': single,
            'joint_source_weights': joint}
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, time, numpy as np
p='/holdout_task/robin_spd_profiler.py'
spec=importlib.util.spec_from_file_location('robin_spd_profiler',p)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
with open('/holdout_task/public_robin_spd_cases.pkl','rb') as f: cases=pickle.load(f)
for i,c in enumerate(cases):
    p=c['problem']; t=time.perf_counter(); out=mod.solve_robin_spd_and_profile_sources(p); elapsed=time.perf_counter()-t
    a,area=mod._robin_matrix(p['kappa'],p['sigma'],p['Lx'],p['Ly'],p['gamma_robin']); b=area*p['candidate_sources'].reshape(10,-1).T
    u=out['basis_fields'].reshape(10,-1).T
    print(i,'seconds',elapsed,'worst mode residual',np.max(np.linalg.norm(a@u-b,axis=0)/np.linalg.norm(b,axis=0)))
    g=c['gt']; print('reference fields:', list(g))
    for key, val in out.items():
        if key in g:
            ref=np.asarray(g[key]); print(key, 'max abs error',np.max(np.abs(val-ref)), 'relative',np.linalg.norm(val-ref)/np.linalg.norm(ref))
PY
```
- **Container Exit Code**: `0` (`537 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin_spd_profiler.py']`
- **Stdout (tail)**:
```text
0 seconds 0.014701599022373557 worst mode residual 5.2707238306983295e-12
reference fields: ['basis_fields', 'single_source_profiled_amplitudes', 'joint_source_weights']
basis_fields max abs error 3.760880495917718e-15 relative 2.0220298672113192e-13
single_source_profiled_amplitudes max abs error 1.814992600657206e-12 relative 3.0607287410635097e-13
joint_source_weights max abs error 1.428301921180264e-13 relative 3.6467384748089415e-13
1 seconds 0.00977808004245162 worst mode residual 5.313446648179597e-12
reference fields: ['basis_fields', 'single_source_profiled_amplitudes', 'joint_source_weights']
basis_fields max abs error 3.83026943495679e-15 relative 1.964047700459796e-13
single_source_profiled_amplitudes max abs error 8.704148513061227e-13 relative 1.9941128048442944e-13
joint_source_weights max abs error 1.067479438177088e-13 relative 2.2807292708117373e-13

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1262f668b9649274 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1262f668 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`31964 ms` LLM)
- **Rationale**: An independent flux balance on anisotropic high-contrast grids can expose an axis or wall-scaling error that a residual using the assembled matrix would miss. Full augmented least squares checks both inverse objectives separately; any mismatch would identify which calculation needs correction.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
path = '/holdout_task/robin_spd_profiler.py'
spec = importlib.util.spec_from_file_location('robin_spd_profiler', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin_spd_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
p = cases[0]['problem']
i, j = np.indices(p['kappa'].shape)
slices = [(f'public-{k}', c['problem'], c['gt']) for k,c in enumerate(cases)]
slices += [('alternating-contrast', dict(p, kappa=np.where((i+j)%2, .02, 80.), Lx=2.4, Ly=.7, gamma_robin=8.), None), ('channel-contrast', dict(p, kappa=np.where((i//9+j//18)%2, 80., .02), Lx=.8, Ly=2.1, gamma_robin=.05), None)]
for name,p,gt in slices:
    got = mod.solve_robin_spd_and_profile_sources(p)
    assert set(got) == {'basis_fields','single_source_profiled_amplitudes','joint_source_weights'}
    u = got['basis_fields']; q = p['candidate_sources']; k = p['kappa']; z = p['nuisance_design']; y = p['Y_obs']; m,nx,ny=q.shape
    assert u.shape == q.shape
    assert got['single_source_profiled_amplitudes'].shape == got['joint_source_weights'].shape == (m,y.shape[1])
    hx,hy=p['Lx']/nx,p['Ly']/ny; gamma=p['gamma_robin']; rhs=hx*hy*q
    balance=hx*hy*p['sigma']*u
    dx = (u[:,1:,:]-u[:,:-1,:])*hy/(.5*hx/k[1:,:]+.5*hx/k[:-1,:]); balance[:,1:,:]+=dx; balance[:,:-1,:]-=dx
    dy = (u[:,:,1:]-u[:,:,:-1])*hx/(.5*hy/k[:,1:]+.5*hy/k[:,:-1]); balance[:,:,1:]+=dy; balance[:,:,:-1]-=dy
    for ix in (0,-1): balance[:,ix,:]+=u[:,ix,:]*hy/(1/gamma+hx/(2*k[ix,:]))
    for iy in (0,-1): balance[:,:,iy]+=u[:,:,iy]*hx/(1/gamma+hy/(2*k[:,iy]))
    residual=np.max(np.linalg.norm((balance-rhs).reshape(m,-1),axis=1)/np.linalg.norm(rhs.reshape(m,-1),axis=1)); assert residual <= 1e-8
    ii,jj=p['sensor_ij'].T; s=u[:,ii,jj].T; nz=z.shape[1]
    def ref(ss,ridge):
        penalty=np.sqrt(np.r_[np.repeat(p['nuisance_ridge'],nz),np.repeat(ridge,ss.shape[1])]); design=np.vstack((np.column_stack((z,ss)),np.diag(penalty)))
        return np.linalg.lstsq(design,np.pad(y,((0,len(penalty)),(0,0))),rcond=None)[0][nz:]
    a=np.vstack([ref(s[:,t:t+1],0.) for t in range(m)]); w=ref(s,p['source_ridge'])
    np.testing.assert_allclose(got['single_source_profiled_amplitudes'],a,rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(got['joint_source_weights'],w,rtol=1e-9,atol=1e-10)
    if gt is not None:
        for key in got: np.testing.assert_allclose(got[key],gt[key],rtol=1e-9,atol=1e-10)
    print(f'{name}: PDE={residual:.3e}, single max abs={np.max(abs(got["single_source_profiled_amplitudes"]-a)):.3e}, joint max abs={np.max(abs(got["joint_source_weights"]-w)):.3e}')
PY
```
- **Container Exit Code**: `0` (`678 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public-0: PDE=5.463e-12, single max abs=8.882e-15, joint max abs=2.137e-15
public-1: PDE=5.466e-12, single max abs=8.549e-15, joint max abs=1.887e-15
alternating-contrast: PDE=7.462e-14, single max abs=1.464e-15, joint max abs=1.568e-15
channel-contrast: PDE=1.381e-10, single max abs=3.331e-15, joint max abs=1.950e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1262f668b9649274 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1262f668 (directory).
```

### Turn 4 — `submit_completed_task` (`12991 ms` LLM)
- **Rationale**: The saved artifact passed isolated Python verification against public references, independently computed face balances, and full augmented least-squares fits for both inverse objectives.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`24` turns remaining; artifact status={'/holdout_task/robin_spd_profiler.py': 'FILE:3532'}): Do NOT lock prematurely with `24` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/robin_spd_profiler.py`. Both public cases and two high-contrast stress cases passed. The worst relative PDE residual was 1.381e-10, below the 1e-8 limit. Both sets of fitted coefficients matched independent least-squares solutions within 9e-15 absolute error.`

### Turn 5 — `execute_in_remote_docker` (`21826 ms` LLM)
- **Rationale**: Test exact source-unit scaling and nuisance-coordinate invariance on both public cases without changing the verified artifact. A failure would trigger inspection of the ridge scaling or nuisance projection.
- **Alternatives Weighed**: `Batch classification, relay, or drift checks` (rejected: The supplied problem is a steady elliptic solve with continuous ridge fits, not a labeled cohort or a time-dependent system.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('robin_spd_profiler', '/holdout_task/robin_spd_profiler.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
solve = module.solve_robin_spd_and_profile_sources
with open('/holdout_task/public_robin_spd_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
for i, case in enumerate(cases):
    p, reference = case['problem'], case['gt']
    # A signed permutation preserves the isotropic nuisance penalty. Scaling all
    # source columns by c requires multiplying the joint ridge by c**2.
    c = 4.0
    z = p['nuisance_design'][:, ::-1].copy()
    z[:, ::2] *= -1
    out = solve(dict(p, nuisance_design=z, candidate_sources=c*p['candidate_sources'], source_ridge=c*c*p['source_ridge']))
    restored = {
        'basis_fields': out['basis_fields']/c,
        'single_source_profiled_amplitudes': out['single_source_profiled_amplitudes']*c,
        'joint_source_weights': out['joint_source_weights']*c,
    }
    for key, result in restored.items():
        np.testing.assert_allclose(result, reference[key], rtol=1e-9, atol=1e-10)
        print(f'public-{i} {key}: max absolute error {np.max(np.abs(result-reference[key])):.3e}')
PY
```
- **Container Exit Code**: `0` (`580 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public-0 basis_fields: max absolute error 3.761e-15
public-0 single_source_profiled_amplitudes: max absolute error 1.808e-12
public-0 joint_source_weights: max absolute error 1.428e-13
public-1 basis_fields: max absolute error 3.830e-15
public-1 single_source_profiled_amplitudes: max absolute error 8.717e-13
public-1 joint_source_weights: max absolute error 1.066e-13

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1262f668b9649274 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1262f668 (directory).
```

### Turn 6 — `submit_completed_task` (`12914 ms` LLM)
- **Rationale**: The saved artifact passed isolated Python verification against public references, independent face balances and least-squares fits, and source-scaling and nuisance-coordinate invariance checks. No remaining failure warrants another edit or test.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`22` turns remaining; artifact status={'/holdout_task/robin_spd_profiler.py': 'FILE:3532'}): Do NOT lock prematurely with `22` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin_spd_profiler.py`. Both public cases and two high-contrast stress cases passed. The worst relative PDE residual was 1.381e-10, below the 1e-8 limit. Both coefficient outputs agreed with independent least-squares fits within 9e-15 absolute error.`

### Turn 7 — `submit_completed_task` (`12341 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification, independent flux and least-squares checks, and source-scaling and nuisance-coordinate invariance checks. The repeated gate identifies no new failure mode applicable to this steady elliptic problem.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin_spd_profiler.py`. Both public cases and two high-contrast stress cases passed. The worst relative PDE residual was 1.381e-10 against a limit of 1e-8. Both coefficient outputs agreed with independent least-squares fits within 9e-15 absolute error.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-M90a-VoluntaryReuse-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/robin_spd_profiler.py": "FILE:3532"
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
    "elapsed_sec": 0.8163
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"all_valid\": true, \"max_pde_rel_residual\": 0.0, \"max_field_rel_l2\": 0.0, \"max_profiled_amp_rel_l2\": 0.0, \"max_joint_weight_rel_l2\": 0.0, \"elapsed_sec\": 0.8163}\n__HEGEL_VERIFIER_AUTH_0578a4d93c90133b400cd63755b1a15d__:{\"strict_pass\": true, \"all_valid\": true, \"max_pde_rel_residual\": 0.0, \"max_field_rel_l2\": 0.0, \"max_profiled_amp_rel_l2\": 0.0, \"max_joint_weight_rel_l2\": 0.0, \"elapsed_sec\": 0.8163}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 6265.63,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
