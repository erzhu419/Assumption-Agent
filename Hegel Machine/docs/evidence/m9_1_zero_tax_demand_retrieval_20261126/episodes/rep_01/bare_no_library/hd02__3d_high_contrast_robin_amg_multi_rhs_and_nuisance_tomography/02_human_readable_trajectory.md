# Dialectical Synthesis Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_2_Selective_LiveGPT` | `hd02__3d_high_contrast_robin_amg_multi_rhs_and_nuisance_tomography`

- **Official Strict Pass**: **`PASS (1.0)`** (`PASSED_ALL_CHECKS`)
- **Blind Public Synthesis Choice**: `antithesis` via `selective_router_confident_single_branch`
- **Synthesis Rationale**: Primary Antithesis branch satisfied public checks under policy; second branch skipped (`1.00u` nominal units).
- **Post-Lock Branch Verdicts**: Thesis (`Bare`) = `NOT_SPAWNED (0T)` | Antithesis (`Wormhole`) = `True` (8T)

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
| **Turns Used / Max Turns** | `8 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `164,533` (`156,959` / `38,016` / `7,574`) |
| **Total Episode Wall Time (s)** | `252.67s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`21564 ms` LLM)
- **Rationale**: Inspect the public case in place to catch schema or axis-order mismatches before assembly and profiling, and record the numerical environment for timing the full-size solves.
- **Alternatives Weighed**: `Implement multigrid first` (rejected: The specification permits factorization reuse; benchmark that simpler method before implementing an iterative hierarchy.)

```bash
ls -lh /holdout_task; python3 -I - <<'PY'
import os, pickle
import numpy as np
import scipy
print('NumPy', np.__version__, 'SciPy', scipy.__version__)
print('thread settings', {k:v for k,v in os.environ.items() if k.endswith('NUM_THREADS')})
with open('/holdout_task/public_robin3d_cases.pkl','rb') as f:
    data=pickle.load(f)
def show(obj, depth=0):
    p='  '*depth
    if isinstance(obj,dict):
        for k,v in obj.items():
            print(p+str(k),end=': ')
            if isinstance(v,np.ndarray) and v.dtype.kind in 'biuf':
                print(v.shape,v.dtype, 'min/max', (float(v.min()),float(v.max())) if v.size else ())
            elif isinstance(v,(dict,list,tuple)) and depth<3:
                show(v,depth+1)
            else:
                print(repr(v)[:140])
    elif isinstance(obj,(list,tuple)):
        print(type(obj).__name__,len(obj))
        if depth<4:
            for v in obj[:3]: show(v,depth+1)
    else:
        print(type(obj).__name__,repr(obj)[:140])
show(data)
PY
```
- **Container Exit Code**: `0` (`656 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 1.2M
-rw-r--r-- 1 root root 1.2M Nov 26 01:38 public_robin3d_cases.pkl
NumPy 1.26.4 SciPy 1.13.0
thread settings {}
list 1
  problem:     Lx: 1.2
    Ly: 1.1
    Lz: 1.0
    kappa: (20, 20, 20) float64 min/max (0.02, 80.0)
    sigma: (20, 20, 20) float64 min/max (0.052250000000000005, 0.13775)
    gamma_robin: 1.5
    candidate_sources: (8, 20, 20, 20) float64 min/max (1.1240851771000825e-25, 0.9979583217974671)
    sensor_ijk: (320, 3) int64 min/max (0.0, 19.0)
    nuisance_design: (320, 5) float64 min/max (-0.9938441702975689, 1.0)
    nuisance_ridge: 0.0015
    source_ridge: 0.025
    Y_obs: (320, 4) float64 min/max (-0.9227534433119405, 1.7143269201683184)
  gt:     basis_fields: (8, 20, 20, 20) float64 min/max (1.9923077389427656e-05, 0.02396656177460551)
    single_source_profiled_amplitudes: (8, 4) float64 min/max (-3.3915227444823284, 2.2876400954505343)
    joint_source_weights: (8, 4) float64 min/max (-0.10797686395976767, 0.05455092839187196)
```

### Turn 2 — `execute_in_remote_docker` (`46194 ms` LLM)
- **Rationale**: Implement the exact stencil and both regression objectives, then compare all three outputs to public ground truth. Any discrepancy will identify which stage needs correction before full-size stress testing.
- **Alternatives Weighed**: `Custom multigrid hierarchy` (rejected: A single shared factorization is permitted and avoids tuning a preconditioner to particular conductivity geometries; full-size timing will determine whether it is sufficient.); `Profile via normal equations` (rejected: Augmented orthogonal projection avoids squaring the condition number of the nuisance design.)

```bash
cat > /holdout_task/robin3d_amg_tomography.py <<'PY'
"""Cell-centered Robin diffusion and ridge-profiled source tomography."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _assemble(kappa, sigma, lengths, gamma):
    shape = kappa.shape
    spacing = np.asarray(lengths, dtype=float) / shape
    volume = np.prod(spacing)
    cells = np.arange(kappa.size).reshape(shape)
    diagonal = volume * np.broadcast_to(sigma, shape).astype(float)
    rows, cols, values = [], [], []
    for axis, h in enumerate(spacing):
        area = volume / h
        left = [slice(None)] * 3
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        kl, kr = kappa[left], kappa[right]
        face = (area / h) * (2 * kl * kr / (kl + kr))
        diagonal[left] += face
        diagonal[right] += face
        i, j, v = cells[left].ravel(), cells[right].ravel(), face.ravel()
        rows.extend((i, j))
        cols.extend((j, i))
        values.extend((-v, -v))
        for side in (0, -1):
            boundary = [slice(None)] * 3
            boundary[axis] = side
            boundary = tuple(boundary)
            kb = kappa[boundary]
            diagonal[boundary] += area * gamma * kb / (kb + 0.5 * h * gamma)
    rows.append(cells.ravel())
    cols.append(cells.ravel())
    values.append(diagonal.ravel())
    matrix = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(kappa.size, kappa.size),
    ).tocsc()
    return matrix, volume


def _profile(sensors, nuisance, observations, nuisance_ridge, source_ridge):
    # Augment by the nuisance penalty, then project onto the orthogonal
    # complement of the augmented nuisance columns. This also handles rank
    # deficiency when the nuisance penalty is zero.
    p = nuisance.shape[1]
    m = sensors.shape[1]
    augmented_z = np.vstack((nuisance, np.sqrt(nuisance_ridge) * np.eye(p)))
    data = np.hstack((sensors, observations))
    data = np.vstack((data, np.zeros((p, data.shape[1]))))
    if p:
        u, singular, _ = np.linalg.svd(augmented_z, full_matrices=False)
        threshold = np.finfo(float).eps * max(augmented_z.shape) * singular[0]
        span = u[:, singular > threshold]
        data -= span @ (span.T @ data)
    X, Y = data[:, :m], data[:, m:]
    norm_sq = np.einsum('ij,ij->j', X, X)
    single = np.divide(X.T @ Y, norm_sq[:, None],
                       out=np.zeros((m, Y.shape[1])), where=norm_sq[:, None] > 0)
    joint = np.linalg.lstsq(
        np.vstack((X, np.sqrt(source_ridge) * np.eye(m))),
        np.vstack((Y, np.zeros((m, Y.shape[1])))), rcond=None,
    )[0]
    return single, joint


def solve_3d_robin_tomography(problem: dict) -> dict:
    """Solve every source basis, then fit signed single and joint weights."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    matrix, volume = _assemble(kappa, problem['sigma'],
                               [problem[key] for key in ('Lx', 'Ly', 'Lz')],
                               problem['gamma_robin'])
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    rhs = volume * sources.reshape(sources.shape[0], -1).T
    # The stencil is SPD. Symmetric minimum-degree ordering limits fill,
    # and the numeric factorization is shared by all right-hand sides.
    factor = splu(matrix, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0,
                  options={'SymmetricMode': True})
    basis = factor.solve(rhs).T.reshape(sources.shape)
    sensors = basis[(slice(None), *np.asarray(problem['sensor_ijk']).T)].T
    single, joint = _profile(sensors, np.asarray(problem['nuisance_design'], dtype=float),
                             np.asarray(problem['Y_obs'], dtype=float),
                             problem['nuisance_ridge'], problem['source_ridge'])
    return {
        'basis_fields': basis,
        'single_source_profiled_amplitudes': single,
        'joint_source_weights': joint,
    }
PY
python3 -I - <<'PY'
import importlib.util, pickle, time
import numpy as np
s=importlib.util.spec_from_file_location('robin','/holdout_task/robin3d_amg_tomography.py'); mod=importlib.util.module_from_spec(s); s.loader.exec_module(mod)
for case in pickle.load(open('/holdout_task/public_robin3d_cases.pkl','rb')):
    t=time.perf_counter(); out=mod.solve_3d_robin_tomography(case['problem']); print('public seconds',time.perf_counter()-t)
    for k, ref in case['gt'].items():
        a=out[k]; print(k,a.shape,'max absolute',np.max(np.abs(a-ref)),'relative',np.linalg.norm(a-ref)/np.linalg.norm(ref)); assert a.shape==ref.shape and np.allclose(a,ref,rtol=1e-7,atol=1e-10)
PY
```
- **Container Exit Code**: `0` (`1074 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin3d_amg_tomography.py']`
- **Stdout (tail)**:
```text
public seconds 0.1589370290748775
basis_fields (8, 20, 20, 20) max absolute 2.732189474663471e-16 relative 2.6564976384719754e-14
single_source_profiled_amplitudes (8, 4) max absolute 4.011790899483003e-13 relative 1.0806262417368052e-13
joint_source_weights (8, 4) max absolute 5.134781488891349e-14 relative 5.064011954109163e-13

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=07d010053b725fbd (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_07d01005 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`44864 ms` LLM)
- **Rationale**: Run three full-size contrast geometries in a clean process. A flux or regression mismatch would identify the stage to correct; exceeding the budget on the slowest pair would justify replacing the shared LU.
- **Alternatives Weighed**: `Check residuals using the production matrix` (rejected: That could miss a shared assembly error; an independent face-flux calculation checks the stencil as well as the solve.)

```bash
python3 -I - <<'PY'
import importlib.util
import itertools
import time

import numpy as np

path = '/holdout_task/robin3d_amg_tomography.py'
spec = importlib.util.spec_from_file_location('robin', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rng = np.random.default_rng(442)
shape = (28, 28, 28)
idx = np.indices(shape)
xyz = (idx + 0.5) / 28
lengths = np.array([1.2, 0.93, 1.07])
h = lengths / shape
vol = np.prod(h)
centers = rng.uniform(0.13, 0.87, (8, 3))
sources = np.stack([np.exp(-np.sum(((xyz - c[:, None, None, None]) / w)**2, axis=0) / 2) for c, w in zip(centers, np.linspace(0.035, 0.11, 8))])
# Interior, face, edge and corner sensors, with different lengths in each direction.
ijk = np.column_stack(np.unravel_index(rng.choice(np.prod(shape), 344, replace=False), shape))
ijk[:8] = np.array(list(itertools.product((0, 27), repeat=3)))
for axis in range(3):
    ijk[8 + 6*axis:14 + 6*axis, axis] = [0, 27, 0, 27, 0, 27]
a, b, c = (ijk.T + 0.5) / 28
Z = np.column_stack((np.ones(len(a)), a, np.sin(np.pi*b), c*c, a*b))
Y = Z @ rng.normal(size=(5, 4)) + rng.normal(scale=0.08, size=(len(a), 4))
lam_z, lam_s = 0.0015, 0.025
geometries = {
    'laminate': (idx[0] // 3) % 2 == 0,
    'blocks-and-channel': (((idx[0] // 4 + idx[1] // 5 + idx[2] // 3) % 2) == 0) | ((idx[1] >= 12) & (idx[1] < 15)),
    'random': rng.random(shape) < 0.43,
}
times = []
for (name, high), gamma in zip(geometries.items(), (0.035, 1.5, 12.0)):
    k = np.where(high, 80.0, 0.02)
    sigma = 0.05225 + 0.0855 * xyz[1]
    p = dict(zip(('Lx', 'Ly', 'Lz'), lengths))
    p.update(kappa=k, sigma=sigma, gamma_robin=gamma, candidate_sources=sources, sensor_ijk=ijk,
             nuisance_design=Z, nuisance_ridge=lam_z, source_ridge=lam_s, Y_obs=Y)
    start = time.perf_counter()
    out = mod.solve_3d_robin_tomography(p)
    times.append(time.perf_counter() - start)
    U = out['basis_fields']
    assert U.shape == (8, *shape)
    # Independent matrix-free balance: apply each oriented flux to its two cells.
    balance = U * (sigma * vol)
    for d in range(3):
        kd = np.moveaxis(k, d, 0)
        ud = np.moveaxis(U, d+1, 1)
        ad = np.moveaxis(balance, d+1, 1)
        flux = (ud[:, :-1] - ud[:, 1:]) * ((vol / h[d]**2) / (0.5/kd[:-1] + 0.5/kd[1:]))
        ad[:, :-1] += flux
        ad[:, 1:] -= flux
        for side in (0, -1):
            ad[:, side] += ud[:, side] * ((vol / h[d]) / (1/gamma + h[d]/(2*kd[side])))
    B = vol * sources
    rel = np.linalg.norm((balance-B).reshape(8, -1), axis=1) / np.linalg.norm(B.reshape(8, -1), axis=1)
    assert np.max(rel) <= 1e-7, rel
    S = U[:, ijk[:, 0], ijk[:, 1], ijk[:, 2]].T
    def regression(X, ridge):
        q = X.shape[1]
        ref_matrix = np.block([[Z, X], [np.sqrt(lam_z)*np.eye(5), np.zeros((5, q))], [np.zeros((q, 5)), np.sqrt(ridge)*np.eye(q)]])
        return np.linalg.lstsq(ref_matrix, np.vstack([Y, np.zeros((5+q, 4))]), rcond=None)[0][5:]
    single = np.vstack([regression(S[:, j:j+1], 0) for j in range(8)])
    joint = regression(S, lam_s)
    errors = []
    for key, ref in [('single_source_profiled_amplitudes', single), ('joint_source_weights', joint)]:
        found = out[key]
        assert found.shape == (8, 4)
        errors.append(np.max(np.abs(found-ref)))
        np.testing.assert_allclose(found, ref, atol=1e-9, rtol=1e-8)
    print(f'{name}: {times[-1]:.3f} s; worst RHS residual {rel.max():.3e}; single/joint max errors {errors[0]:.3e}/{errors[1]:.3e}', flush=True)
worst_pair = sum(sorted(times)[-2:])
print(f'Slowest two cases: {worst_pair:.3f} s', flush=True)
assert worst_pair < 14.0
PY
```
- **Container Exit Code**: `0` (`9303 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
laminate: 2.805 s; worst RHS residual 4.698e-11; single/joint max errors 4.929e-14/3.013e-15
blocks-and-channel: 2.853 s; worst RHS residual 4.923e-12; single/joint max errors 1.386e-13/2.526e-15
random: 2.305 s; worst RHS residual 2.855e-13; single/joint max errors 9.237e-13/1.383e-15
Slowest two cases: 5.658 s

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=07d010053b725fbd (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_07d01005 (directory).
```

### Turn 4 — `submit_completed_task` (`13954 ms` LLM)
- **Rationale**: The unchanged artifact passed clean-process verification against public ground truth, independent face-flux balances, and separate augmented least-squares references for both regression outputs.
- **Alternatives Weighed**: `Replace shared sparse LU with multigrid` (rejected: The permitted shared factorization met the residual and runtime requirements with margin on the full-size stress cases.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/robin3d_amg_tomography.py': 'FILE:4012'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin3d_amg_tomography.py`. Across three full-size, 4,000× contrast cases, the worst relative PDE residual was 4.70e-11 (limit: 1e-7). The two slowest cases took 5.66 s combined (budget: 14.0 s). Both tomography outputs matched independent ridge least-squares fits.`

### Turn 5 — `execute_in_remote_docker` (`32815 ms` LLM)
- **Rationale**: Check global source-outflow balance on the public case to detect omitted or double-counted Robin faces, and compare boundary-only and interior-only fits to independent references to detect sensor indexing or profiling errors masked by pooled agreement.
- **Alternatives Weighed**: `Run batch classification or relay-mode checks` (rejected: The public problem has no batch labels, classification outcomes, or relay states; it specifies a steady PDE and continuous ridge fits.)

```bash
cat > /tmp/audit_robin3d.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('robin', '/holdout_task/robin3d_amg_tomography.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin3d_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
for case in cases:
    p = case['problem']
    out = mod.solve_3d_robin_tomography(p)
    U, k = out['basis_fields'], p['kappa']
    h = np.array([p['Lx'], p['Ly'], p['Lz']]) / k.shape
    vol, gamma = np.prod(h), p['gamma_robin']
    influx = vol * p['candidate_sources'].sum(axis=(1, 2, 3))
    losses = vol * (p['sigma'] * U).sum(axis=(1, 2, 3))
    for axis in range(3):
        for face in (0, -1):
            kb = k.take(face, axis=axis)
            ub = U.take(face, axis=axis + 1)
            losses += (ub * ((vol / h[axis]) / (1 / gamma + 0.5 * h[axis] / kb))).sum(axis=(1, 2))
    conservation = np.max(np.abs(losses - influx) / np.abs(influx))
    print('public global balance, worst RHS:', conservation)
    assert conservation <= 1e-7
    ijk = p['sensor_ijk']
    boundary = ((ijk == 0) | (ijk == np.array(k.shape) - 1)).any(axis=1)
    Z, Y = p['nuisance_design'], p['Y_obs']
    S = U[(slice(None), *ijk.T)].T
    S_ref = case['gt']['basis_fields'][(slice(None), *ijk.T)].T
    lz, ls = p['nuisance_ridge'], p['source_ridge']
    for label, keep in [('interior only', ~boundary), ('boundary only', boundary)]:
        z, y, s, sref = Z[keep], Y[keep], S[keep], S_ref[keep]
        if len(z) <= Z.shape[1]:
            continue
        single, joint = mod._profile(s, z, y, lz, ls)
        def direct(x, penalty):
            a, b = z.shape[1], x.shape[1]
            design = np.block([[z, x], [np.sqrt(lz)*np.eye(a), np.zeros((a,b))], [np.zeros((b,a)), np.sqrt(penalty)*np.eye(b)]])
            return np.linalg.lstsq(design, np.vstack((y, np.zeros((a+b,y.shape[1])))), rcond=None)[0][a:]
        a_ref = np.vstack([direct(sref[:, j:j+1], 0) for j in range(sref.shape[1])])
        w_ref = direct(sref, ls)
        print(label, len(z), 'sensors; max single/joint error:', np.max(np.abs(single-a_ref)), np.max(np.abs(joint-w_ref)))
        np.testing.assert_allclose(single, a_ref, atol=1e-9, rtol=1e-8)
        np.testing.assert_allclose(joint, w_ref, atol=1e-9, rtol=1e-8)
PY
python3 -I /tmp/audit_robin3d.py
```
- **Container Exit Code**: `0` (`1046 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public global balance, worst RHS: 1.1859548852002695e-13
interior only 229 sensors; max single/joint error: 1.1146639167236572e-13 2.581268532253489e-15
boundary only 91 sensors; max single/joint error: 6.039613253960852e-14 2.3332030751888055e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=07d010053b725fbd (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_07d01005 (directory).
```

### Turn 6 — `submit_completed_task` (`15289 ms` LLM)
- **Rationale**: The unchanged artifact passed clean-process verification. The public audit independently confirmed global Robin balance to 1.19e-13 and found no profiling discrepancy in either sensor subset. No further edit is warranted.
- **Alternatives Weighed**: `Replace shared sparse LU with multigrid` (rejected: The permitted shared factorization met both residual and runtime limits with margin on the full-size stress cases.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/robin3d_amg_tomography.py': 'FILE:4012'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin3d_amg_tomography.py`. On three full-size 4,000× contrast stress cases, the worst relative PDE residual was 4.70e-11 (limit: 1e-7), and the two slowest cases took 5.66 s combined (budget: 14.0 s). Both tomography outputs matched independent least-squares fits on the public case, including separate interior and boundary sensor subsets.`

### Turn 7 — `execute_in_remote_docker` (`27507 ms` LLM)
- **Rationale**: Use the public measurement design to test a distinct failure mode: removing a pure nuisance shift entirely would ignore the specified positive nuisance ridge. If either coefficient response disagrees with its analytic prediction, correct that profiling objective.
- **Alternatives Weighed**: `Measure within-batch classification accuracy` (rejected: The public case supplies continuous observations and a nuisance design, not batch or class labels.)

```bash
cat > /tmp/audit_robin3d_nuisance_shift.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('robin', '/holdout_task/robin3d_amg_tomography.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin3d_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
for case in cases:
    p = case['problem']
    out = mod.solve_3d_robin_tomography(p)
    Z, Y = p['nuisance_design'], p['Y_obs']
    S = out['basis_fields'][(slice(None), *p['sensor_ijk'].T)].T
    lz, ls = p['nuisance_ridge'], p['source_ridge']
    shift = np.cos(np.arange(Z.shape[1] * Y.shape[1])).reshape(Z.shape[1], Y.shape[1])
    shifted = mod.solve_3d_robin_tomography(dict(p, Y_obs=Y + Z @ shift))
    # Under the specified nuisance ridge, a pure nuisance shift is not
    # canceled exactly: P Z C = lambda_z Z (Z.T Z + lambda_z I)^-1 C.
    G = Z.T @ Z + lz * np.eye(Z.shape[1])
    P_S = S - Z @ np.linalg.solve(G, Z.T @ S)
    h = S.T @ (lz * Z @ np.linalg.solve(G, shift))
    gram = S.T @ P_S
    expected = dict(single_source_profiled_amplitudes=h / np.diag(gram)[:, None],
                    joint_source_weights=np.linalg.solve(gram + ls * np.eye(S.shape[1]), h))
    for key, ref in expected.items():
        delta = shifted[key] - out[key]
        err = np.max(np.abs(delta - ref))
        print(f'{key}: largest nuisance-shift response {np.max(np.abs(ref)):.3e}; max response error {err:.3e}')
        np.testing.assert_allclose(delta, ref, rtol=1e-8, atol=1e-10)
PY
python3 -I /tmp/audit_robin3d_nuisance_shift.py
```
- **Container Exit Code**: `0` (`1201 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
single_source_profiled_amplitudes: largest nuisance-shift response 1.567e-02; max response error 1.633e-14
joint_source_weights: largest nuisance-shift response 3.930e-04; max response error 4.522e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=07d010053b725fbd (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_07d01005 (directory).
```

### Turn 8 — `submit_completed_task` (`15075 ms` LLM)
- **Rationale**: The unchanged artifact passed clean-process verification against independent face-flux balances and regression solutions. The public nuisance-shift audit also confirmed that the specified positive nuisance ridge preserves the predicted nonzero response in both outputs.
- **Alternatives Weighed**: `Replace shared sparse LU with multigrid` (rejected: The permitted shared factorization met the residual and runtime requirements with margin on the full-size stress cases.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin3d_amg_tomography.py`. Across three full-size 4,000× contrast stress cases, the worst relative PDE residual was 4.70e-11 (limit: 1e-7). The two slowest cases took 5.66 s combined (budget: 14.0 s). Both tomography outputs matched independent least-squares fits, and their responses to a nuisance shift matched the ridge model’s analytic predictions.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-M91-HardDemand-2",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/robin3d_amg_tomography.py": "FILE:4012"
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
    "student_elapsed_sec": 2.5207
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"all_valid\": true, \"max_pde_rel_residual\": 0.0, \"max_field_rel_l2\": 0.0, \"max_profiled_amp_rel_l2\": 0.0, \"max_joint_weight_rel_l2\": 0.0, \"student_elapsed_sec\": 2.5207}\n__HEGEL_VERIFIER_AUTH_7b9bf14ea12b7fb8e3838c2d53df851a__:{\"strict_pass\": true, \"all_valid\": true, \"max_pde_rel_residual\": 0.0, \"max_field_rel_l2\": 0.0, \"max_profiled_amp_rel_l2\": 0.0, \"max_joint_weight_rel_l2\": 0.0, \"student_elapsed_sec\": 2.5207}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 12021.28,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
