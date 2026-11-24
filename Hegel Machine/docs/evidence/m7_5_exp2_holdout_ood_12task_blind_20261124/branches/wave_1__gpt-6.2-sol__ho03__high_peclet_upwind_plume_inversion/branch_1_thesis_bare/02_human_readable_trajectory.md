# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Bare_ReAct_LiveGPT` | `ho03__high_peclet_upwind_plume_inversion`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-OOD-Holdout-12` |
| **Domain** | `Computational Transport PDE / Source Inversion` |
| **Wave / Wormhole Family** | `wave_1` / `fam_multiscale_pde_and_combinatorial_physics` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Bare_ReAct_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `6 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `119,309` (`114,453` / `6,272` / `4,856`) |
| **Total Episode Wall Time (s)** | `172.53s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`16087 ms` LLM)
- **Rationale**: Inspect the public fixture to establish input keys, dimensions, and noise clues; these determine the stencil orientation and inversion model.

```bash
ls -la /holdout_task; python - <<'PY'
import pickle,numpy as np,scipy
p=pickle.load(open('/holdout_task/public_peclet.pkl','rb'))
print('scipy',scipy.__version__)
def show(p,level=0):
 if isinstance(p,dict) and level<3:
  for k,v in p.items():
   print('  '*level,k,end=': ');show(v,level+1)
 elif isinstance(p,np.ndarray):
  print(p.shape,p.dtype, p if p.size<=40 else (p.min(),p.max()))
 else:print(str(p)[:500])
show(p)
PY
```
- **Container Exit Code**: `127` (`307 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 228
drwxr-xr-x 2 root root   4096 Nov 24 08:12 .
drwxr-xr-x 1 root root   4096 Nov 24 08:12 ..
-rw-r--r-- 1 root root 223703 Nov 24 08:12 public_peclet.pkl
```
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`15911 ms` LLM)
- **Rationale**: The Python alias was missing. Use python3 to inspect the input schema and confirm the NumPy/SciPy runtime before implementing the solver.

```bash
command -v python3; python3 - <<'PY'
import pickle,numpy as np,scipy
p=pickle.load(open('/holdout_task/public_peclet.pkl','rb'))
print('scipy',scipy.__version__)
def show(p,level=0):
 if isinstance(p,dict) and level<3:
  for k,v in p.items():
   print('  '*level,k,end=': ');show(v,level+1)
 elif isinstance(p,np.ndarray):
  print(p.shape,p.dtype,p if p.size<=40 else (p.min(),p.max()))
 else:print(str(p)[:500])
show(p)
PY
```
- **Container Exit Code**: `0` (`507 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/usr/bin/python3
scipy 1.13.0
 problem:    N: 96
   L: 1.0
   h: 0.010416666666666666
   ux_face: (97, 96) float64 (1.2000535448361753, 1.9999464551638249)
   uy_face: (96, 97) float64 (0.3000535448361752, 1.0999464551638247)
   D: 0.0025
   kappa: 0.25
   sigma_s: 0.045
   sensor_ij: (18, 2) int64 [[18 20]
 [18 42]
 [18 64]
 [18 82]
 [34 20]
 [34 42]
 [34 64]
 [34 82]
 [50 20]
 [50 42]
 [50 64]
 [50 82]
 [66 20]
 [66 42]
 [66 64]
 [66 82]
 [80 20]
 [80 42]]
   c_obs: (18,) float64 [-3.03737767e-03  5.23514512e-04 -3.49782421e-03  8.50943144e-01
 -2.29858570e-03  1.31783730e-01 -4.19216622e-03 -4.03978902e-03
 -5.23869690e-03  2.01863052e-01 -3.59715158e-03  4.26808488e-03
 -3.14617965e-03  8.48108336e-02 -6.55668947e-01 -2.95458117e-03
 -1.24362664e-02  1.58057311e-02]
 gt_field: (96, 96) float64 (5.355569840726541e-38, 0.23371690417950722)
 gt_source_xy: [0.28, 0.36]
 gt_Q: 4.2
```

### Turn 3 — `execute_in_remote_docker` (`68792 ms` LLM)
- **Rationale**: Implement the flux stencil and two-outlier inversion; the public field comparison detects stencil or axis errors, and recovery and timing detect failures against the stated inversion targets.
- **Alternatives Weighed**: `Repeated forward solves during optimization` (rejected: Adjoint sensor maps reduce each trial to weighted Gaussian sums after one sparse factorization.)

```bash
cat > /holdout_task/peclet_plume.py <<'PY'
"""Exponentially fitted transport and robust inversion of a Gaussian source."""

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _face_rates(velocity, diffusivity, h):
    """Positive- and negative-direction coefficients of flux divided by h."""
    velocity = np.asarray(velocity, dtype=float)
    if diffusivity == 0:
        return np.maximum(velocity, 0) / h, np.maximum(-velocity, 0) / h
    z = np.abs(velocity) * (h / diffusivity)
    # B(z) = z exp(-z) / (1-exp(-z)), avoiding overflow and cancellation.
    ratio = np.ones_like(z)
    np.divide(z * np.exp(-z), -np.expm1(-z), out=ratio, where=z != 0)
    small = (diffusivity / h**2) * ratio
    return small + np.maximum(velocity, 0) / h, small + np.maximum(-velocity, 0) / h


def _operator(ux_face, uy_face, diffusivity, kappa, h):
    xp, xm = _face_rates(ux_face, diffusivity, h)
    yp, ym = _face_rates(uy_face, diffusivity, h)
    diagonal = xp[1:, :] + xm[:-1, :] + yp[:, 1:] + ym[:, :-1] + kappa
    ids = np.arange(diagonal.size).reshape(diagonal.shape)
    left, right = ids[:-1, :].ravel(), ids[1:, :].ravel()
    lower, upper = ids[:, :-1].ravel(), ids[:, 1:].ravel()
    # Each boundary face contributes its center coefficient to the diagonal;
    # the ghost coefficient multiplies zero and has no matrix entry.
    return coo_matrix((
        np.concatenate((diagonal.ravel(), -xm[1:-1, :].ravel(),
                        -xp[1:-1, :].ravel(), -ym[:, 1:-1].ravel(),
                        -yp[:, 1:-1].ravel())),
        (np.concatenate((ids.ravel(), left, right, lower, upper)),
         np.concatenate((ids.ravel(), right, left, upper, lower))),
    ), shape=(ids.size, ids.size)).tocsc()


def forward_peclet_field(ux_face, uy_face, diffusivity, kappa, source_map, h):
    """Solve the steady transport equation in [x-index, y-index] order."""
    source = np.asarray(source_map, dtype=float)
    matrix = _operator(ux_face, uy_face, diffusivity, kappa, h)
    return splu(matrix).solve(source.ravel()).reshape(source.shape)


class _TrimmedFit:
    """Profile nonnegative source strength, excluding the two worst sensors."""

    def __init__(self, observed):
        self.observed = observed
        self.rows = np.arange(observed.size)
        self.i, self.j = np.triu_indices(observed.size, 1)
        self.keep = (self.rows != self.i[:, None]) & (self.rows != self.j[:, None])
        y2 = observed**2
        self.y2 = y2.sum() - y2[self.i] - y2[self.j]

    def grid_loss(self, responses):
        # Evaluate all two-sensor omissions at once for each block of positions.
        z2 = responses**2
        zy = responses * self.observed
        norm = z2.sum(axis=1, keepdims=True) - z2[:, self.i] - z2[:, self.j]
        corr = zy.sum(axis=1, keepdims=True) - zy[:, self.i] - zy[:, self.j]
        np.maximum(corr, 0, out=corr)
        # Recompute nearly cancelled sums directly: the retained response can
        # be orders of magnitude weaker than the two omitted responses.
        weak = norm < 1e-10 * z2.sum(axis=1, keepdims=True)
        a, b = np.nonzero(weak)
        for start in range(0, a.size, 4096):
            s = slice(start, start + 4096)
            mask = self.keep[b[s]]
            norm[a[s], b[s]] = np.sum(z2[a[s]] * mask, axis=1)
            corr[a[s], b[s]] = np.maximum(np.sum(zy[a[s]] * mask, axis=1), 0)
        reduction = np.divide(corr**2, norm, out=np.zeros_like(norm), where=norm > 0)
        return np.min(self.y2 - reduction, axis=1)

    def best(self, response):
        # Direct sums avoid cancellation in both outlier removal and residuals.
        r = response * self.keep
        norm = np.einsum('ij,ij->i', r, r)
        corr = np.einsum('ij,j->i', r, self.observed)
        q = np.divide(np.maximum(corr, 0), norm, out=np.zeros_like(norm), where=norm > 0)
        residual = (q[:, None] * r - self.observed) * self.keep
        loss = np.einsum('ij,ij->i', residual, residual)
        k = np.argmin(loss)
        return loss[k], q[k], residual[k]


def invert_point_source(problem):
    """Estimate the center and peak amplitude of the Gaussian forcing."""
    n, h = problem['N'], problem['h']
    ij = np.asarray(problem['sensor_ij'], dtype=int)
    obs = np.asarray(problem['c_obs'], dtype=float)
    length = n * h
    width = problem['sigma_s'] / length
    # Normalize spatial parameters to [0,1] and concentration by a common
    # factor, without altering the least-squares optimum or sensor weights.
    cells = (np.arange(n) + 0.5) / n
    scale = np.max(np.abs(obs)) or 1.0
    fit = _TrimmedFit(obs / scale)
    lu = splu(_operator(problem['ux_face'], problem['uy_face'],
                        problem['D'], problem['kappa'], h))
    rhs = np.zeros((n*n, obs.size), order='F')
    rhs[ij[:, 0] * n + ij[:, 1], np.arange(obs.size)] = 1.0
    adjoint = np.ascontiguousarray(lu.solve(rhs, trans='T').T.reshape(-1, n, n))

    def evaluate(xy, strength=False):
        dx, dy = (cells[None, :] - np.asarray(xy)[:, None]) / width
        gx, gy = np.exp(-0.5 * dx**2), np.exp(-0.5 * dy**2)
        along_x = np.einsum('sij,i->sj', adjoint, gx)
        response = np.einsum('sj,j->s', along_x, gy)
        loss, q, residual = fit.best(response)
        if strength:
            return q * scale
        derivative_x = np.einsum('sij,i,j->s', adjoint, gx * dx, gy)
        derivative_y = np.einsum('sj,j->s', along_x, gy * dy)
        grad = (2 * q / width) * np.array((residual @ derivative_x, residual @ derivative_y))
        return loss, grad

    count = min(97, max(51, int(np.ceil(2.5 / width)) + 1))
    grid = np.linspace(0., 1., count)
    gauss = np.exp(-0.5 * ((cells[None, :] - grid[:, None]) / width)**2)
    intermediate = np.einsum('xi,sij->xsj', gauss, adjoint)
    responses = np.einsum('xsj,yj->xys', intermediate, gauss).reshape(-1, obs.size)
    score = np.concatenate([fit.grid_loss(responses[k:k+512])
                            for k in range(0, len(responses), 512)]).reshape(count, count)
    best_loss = np.inf
    starts = []
    for idx in np.argsort(score.ravel()):
        xy = grid[np.array(divmod(idx, count))]
        if any(np.max(np.abs(xy - start)) < max(width, 2.5 / (count-1)) for start in starts):
            continue
        starts.append(xy)
        result = minimize(evaluate, xy, jac=True, method='L-BFGS-B',
                          bounds=((0., 1.), (0., 1.)),
                          options={'ftol': 1e-13, 'gtol': 1e-9, 'maxiter': 80})
        if result.fun < best_loss:
            best_loss, best_xy = result.fun, result.x
        if len(starts) == 12:
            break
    return {'source_xy': (length * best_xy).tolist(),
            'strength_Q': float(evaluate(best_xy, strength=True))}
PY
cd /holdout_task && python3 - <<'PY'
import pickle, time, numpy as np
import peclet_plume as m
p=pickle.load(open('public_peclet.pkl','rb'))
a=p['problem']; t=(np.arange(a['N'])+.5)*a['h']; x,y=p['gt_source_xy']; g=p['gt_Q']*np.exp(-0.5*((t[:,None]-x)**2+(t[None,:]-y)**2)/a['sigma_s']**2)
c=m.forward_peclet_field(a['ux_face'],a['uy_face'],a['D'],a['kappa'],g,a['h'])
print('forward max error',np.max(np.abs(c-p['gt_field'])))
print('clean observations',c[tuple(a['sensor_ij'].T)])
s=time.perf_counter(); r=m.invert_point_source(a); print('inversion',r,'runtime',time.perf_counter()-s)
PY
```
- **Container Exit Code**: `0` (`991 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/peclet_plume.py']`
- **Stdout (tail)**:
```text
forward max error 1.942890293094024e-16
clean observations [3.49595195e-05 2.30893576e-03 6.69858757e-11 1.08052057e-18
 4.10054716e-04 1.25286682e-01 1.84184284e-06 4.43072238e-12
 3.05454713e-05 2.00256190e-01 1.06072713e-03 1.68909466e-07
 1.03628309e-06 8.18416760e-02 3.66394722e-02 1.62912767e-04
 3.02573495e-08 1.32457509e-02]
inversion {'source_xy': [0.2725995213080855, 0.3586006021895408], 'strength_Q': 4.295202600657765} runtime 0.11586927995085716
```

### Turn 4 — `execute_in_remote_docker` (`22150 ms` LLM)
- **Rationale**: An independent signed-velocity flux balance detects wrong inflow or ghost-face coefficients and would trigger a stencil correction. Shifted sources and different faulty sensors test whether recovery depends on the public fixture’s source location or outlier identities.

```bash
cd /holdout_task && PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import pickle, time
import numpy as np
from peclet_plume import forward_peclet_field, invert_point_source
rng = np.random.default_rng(37)
n,h,d,k=8,.125,.004,.3
u=rng.normal(size=(n+1,n)); v=rng.normal(size=(n,n+1)); u[2,2]=0; v[2,2]=0
s=rng.uniform(size=(n,n)); c=forward_peclet_field(u,v,d,k,s,h)
def b(p):
    out=np.ones_like(p); np.divide(p,np.expm1(p),out=out,where=p!=0); return out
cx=np.pad(c,((1,1),(0,0))); cy=np.pad(c,((0,0),(1,1)))
jx=d/h*(b(-u*h/d)*cx[:-1]-b(u*h/d)*cx[1:]); jy=d/h*(b(-v*h/d)*cy[:,:-1]-b(v*h/d)*cy[:,1:]); err=np.max(np.abs((np.diff(jx,axis=0)+np.diff(jy,axis=1))/h+k*c-s)); print('signed-velocity flux residual:',err); assert err<1e-10
p=pickle.load(open('public_peclet.pkl','rb'))['problem']; t=(np.arange(p['N'])+.5)*p['h']; ij=tuple(p['sensor_ij'].T)
for x,y,q in [(0.24,0.32,3.8),(0.33,0.37,5.1),(0.39,0.46,4.8),(0.46,0.55,4.4)]:
    s=q*np.exp(-.5*((t[:,None]-x)**2+(t[None,:]-y)**2)/p['sigma_s']**2)
    c=forward_peclet_field(p['ux_face'],p['uy_face'],p['D'],p['kappa'],s,p['h']); obs=c[ij]+rng.normal(0,0.002,len(ij[0])); obs[rng.choice(len(obs),2,replace=False)]+=[.65,-.72]
    start=time.perf_counter(); pred=invert_point_source(dict(p,c_obs=obs)); dt=time.perf_counter()-start
    print((x,y,q),pred,'errors',np.linalg.norm(np.array(pred['source_xy'])-[x,y]),abs(pred['strength_Q']/q-1),'seconds',dt)
PY
```
- **Container Exit Code**: `0` (`1394 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
signed-velocity flux residual: 1.3322676295501878e-15
(0.24, 0.32, 3.8) {'source_xy': [0.24804312736562503, 0.3200179962360265], 'strength_Q': 3.8137679986848623} errors 0.0080431474985964 0.0036231575486480594 seconds 0.11804924090392888
(0.33, 0.37, 5.1) {'source_xy': [0.3270101146378949, 0.36867473351066493], 'strength_Q': 5.1034166886957735} errors 0.003270435100454493 0.000669938959955596 seconds 0.12199976085685194
(0.39, 0.46, 4.8) {'source_xy': [0.3891498564207219, 0.4601222442911924], 'strength_Q': 4.7519350608839455} errors 0.0008588875200612143 0.010013528982511288 seconds 0.10384092899039388
(0.46, 0.55, 4.4) {'source_xy': [0.434593992264975, 0.5378290941798062], 'strength_Q': 4.503374349929592} errors 0.02817083913404389 0.023494170438543627 seconds 0.15844057220965624
```

### Turn 5 — `execute_in_remote_docker` (`21630 ms` LLM)
- **Rationale**: Diagnose the one missed synthetic position tolerance: if fitting with the true faulty sensors removed recovers the source, improve outlier selection; otherwise assess how strongly the remaining sensors constrain the position.

```bash
cd /holdout_task && PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import pickle
import numpy as np
from scipy.optimize import least_squares
from peclet_plume import forward_peclet_field, invert_point_source, _operator
from scipy.sparse.linalg import splu
rng=np.random.default_rng(37)
rng.normal(size=(9,8)); rng.normal(size=(8,9)); rng.uniform(size=(8,8))
p=pickle.load(open('public_peclet.pkl','rb'))['problem']; n,h=p['N'],p['h']; t=(np.arange(n)+.5)*h; ij=tuple(p['sensor_ij'].T); sig=p['sigma_s']
def source(x,y,q): return q*np.exp(-.5*((t[:,None]-x)**2+(t[None,:]-y)**2)/sig**2)
for x,y,q in [(.24,.32,3.8),(.33,.37,5.1),(.39,.46,4.8),(.46,.55,4.4)]:
    clean=forward_peclet_field(p['ux_face'],p['uy_face'],p['D'],p['kappa'],source(x,y,q),h)[ij]; obs=clean+rng.normal(0,.002,len(clean)); faults=rng.choice(len(obs),2,replace=False); obs[faults]+=[.65,-.72]
keep=np.delete(np.arange(len(obs)),faults)
rhs=np.zeros((n*n, len(obs))); rhs[ij[0]*n+ij[1],np.arange(len(obs))]=1
adj=splu(_operator(p['ux_face'],p['uy_face'],p['D'],p['kappa'],h)).solve(rhs,trans='T')
def res(par): return (source(*par).ravel() @ adj - obs)[keep]
fit=least_squares(res,[x,y,q]); pred=invert_point_source(dict(p,c_obs=obs)); print('faults:',faults,'true-fault oracle:',fit.x,'solver:',pred); print('source information singular values:',np.linalg.svd(fit.jac,compute_uv=False)); print('residual norm at truth / oracle:',np.linalg.norm(res((x,y,q))),np.linalg.norm(fit.fun)); print('clean signals:',np.round(clean,4))
PY
```
- **Container Exit Code**: `0` (`1195 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
faults: [15 13] true-fault oracle: [0.43459524 0.53782968 4.50336931] solver: {'source_xy': [0.434593992264975, 0.5378290941798062], 'strength_Q': 4.503374349929592}
source information singular values: [2.29331045 0.22625517 0.02970388]
residual norm at truth / oracle: 0.008258619332294607 0.007194045447778787
clean signals: [0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 2.000e-04 3.000e-04
 0.000e+00 0.000e+00 5.300e-03 7.720e-02 0.000e+00 0.000e+00 2.000e-04
 2.126e-01 1.160e-02 0.000e+00 0.000e+00]
```

### Turn 6 — `submit_completed_task` (`14769 ms` LLM)
- **Rationale**: The required artifact was created and verified in the remote container. The public fixture meets all targets, and an independent signed-velocity flux check confirms the boundary discretization.
- **Alternatives Weighed**: `Add regularization to improve the weakly constrained synthetic case` (rejected: The same position error persisted with the true faulty sensors excluded. No source prior was specified to justify biasing the fit.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/peclet_plume.py`. On the public fixture, the forward field matched the reference to machine precision. Inversion took 0.12 s, with position error 0.0075 L and strength error 2.27%, meeting all stated targets.

Limitation: One synthetic case narrowly missed the position target even when the two faulty sensors were known and excluded.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-OOD-Holdout-12",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
