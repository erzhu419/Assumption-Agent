# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Bare_ReAct_LiveGPT` | `ho01__crosswell_anisotropic_clock_tomography`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-OOD-Holdout-12` |
| **Domain** | `Geophysics / Borehole Traveltime Tomography` |
| **Wave / Wormhole Family** | `wave_1` / `fam_guided_wave_shm_localization` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Bare_ReAct_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `5 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `75,802` (`72,886` / `9,856` / `2,916`) |
| **Total Episode Wall Time (s)** | `103.44s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`12634 ms` LLM)
- **Rationale**: Inspect the survey schema and observed ranges to identify supplied model parameters and how pick times and geometry are represented.

```bash
cd /holdout_task && ls -l && python - <<'PY'
import pickle, numpy as np, scipy
p=pickle.load(open('public_surveys.pkl','rb'))
def show(x,label='',d=0):
 if isinstance(x,dict):
  print(label,'dict',list(x));
  for k,v in x.items(): show(v,str(k),d+1)
 elif isinstance(x,(list,tuple)):
  print(label,type(x).__name__,len(x));
  for v in x[:3]: show(v,label+' item',d+1)
 elif isinstance(x,np.ndarray): print(label,x.shape,x.dtype,x.ravel()[:8], 'min/max', np.nanmin(x),np.nanmax(x))
 else: print(label,repr(x)[:150])
show(p);print('scipy',scipy.__version__)
PY
```
- **Container Exit Code**: `127` (`340 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 8
-rw-r--r-- 1 root root 5205 Nov 25 12:12 public_surveys.pkl
```
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`11944 ms` LLM)
- **Rationale**: Use the available Python interpreter to identify the survey fields, dimensions, and numerical libraries before choosing what to fit.

```bash
cd /holdout_task && command -v python3 && python3 - <<'PY'
import pickle
import numpy as np
p=pickle.load(open('public_surveys.pkl','rb'))
def show(x,label='survey'):
 if isinstance(x,dict):
  print(label, 'dict', list(x));
  for k,v in x.items(): show(v,str(k))
 elif isinstance(x,(list,tuple)):
  print(label,type(x).__name__,len(x));
  for v in x[:5]: show(v,label+' item')
 elif isinstance(x,np.ndarray): print(label,x.shape,x.dtype,x.ravel()[:8],np.nanmin(x),np.nanmax(x))
 else: print(label,repr(x)[:150])
show(p)
try:
 import scipy
 print('scipy',scipy.__version__)
except ImportError: print('no scipy')
PY
```
- **Container Exit Code**: `0` (`590 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/usr/bin/python3
survey list 2
survey item dict ['survey', 'gt']
survey dict ['W_m', 'H_m', 'z_src_m', 'z_rcv_m', 'T_obs_s', 'R_plume_m']
W_m 40.0
H_m 80.0
z_src_m (15,) float64 [ 6.         10.85714286 15.71428571 20.57142857 25.42857143 30.28571429
 35.14285714 40.        ] 6.0 74.0
z_rcv_m (16,) float64 [ 5.          9.66666667 14.33333333 19.         23.66666667 28.33333333
 33.         37.66666667] 5.0 75.0
T_obs_s (15, 16) float64 [0.01491584 0.01729941 0.01816543 0.0148971  0.01612493 0.02175942
 0.01714121 0.01843184] 0.01330233416627665 0.03317839066777648
R_plume_m 5.5
gt dict ['plume_xz_m', 'rcv_clock_bias_s']
plume_xz_m list 2
plume_xz_m item 16.5
plume_xz_m item 34.0
rcv_clock_bias_s (16,) float64 [-0.0019168   0.00045573  0.00110994 -0.00255604 -0.00179292  0.0032009
 -0.00228896 -0.0019091 ] -0.002556039023045965 0.003329653424440951
survey item dict ['survey', 'gt']
survey dict ['W_m', 'H_m', 'z_src_m', 'z_rcv_m', 'T_obs_s', 'R_plume_m']
W_m 40.0
H_m 80.0
z_src_m (15,) float64 [ 6.         10.85714286 15.71428571 20.57142857 25.42857143 30.28571429
 35.14285714 40.        ] 6.0 74.0
z_rcv_m (16,) float64 [ 5.          9.66666667 14.33333333 19.         23.66666667 28.33333333
 33.         37.66666667] 5.0 75.0
T_obs_s (15, 16) float64 [0.01425288 0.0187468  0.01403325 0.0142902  0.01580415 0.01554096
 0.01899251 0.0161894 ] 0.01336071363590472 0.028705721115872333
R_plume_m 5.5
gt dict ['plume_xz_m', 'rcv_clock_bias_s']
plume_xz_m list 2
plume_xz_m item 26.0
plume_xz_m item 51.5
rcv_clock_bias_s (16,) float64 [-0.0009812   0.00347274 -0.00137483 -0.00143901 -0.00034719 -0.00111101
  0.00170437 -0.00184997] -0.002568385497994811 0.0034727403303111385
scipy 1.13.0
```

### Turn 3 — `execute_in_remote_docker` (`44858 ms` LLM)
- **Rationale**: Implement and test robust joint inversion. Public center and clock errors will reveal fitting failures; residual sizes will determine whether to correct cycle skips or discard them.
- **Alternatives Weighed**: `Numerical ray quadrature` (rejected: The straight-ray Gaussian integral is analytic, including truncation at the wells.)

```bash
cd /holdout_task && cat >tomography_solver.py <<'PY'
"""Straight-ray inversion for a Gaussian plume and receiver clock offsets."""

import numpy as np
from scipy.linalg import helmert
from scipy.optimize import least_squares
from scipy.special import ndtr


def _project_clocks(clock):
    """Project onto the zero-sum clock box (units: milliseconds)."""
    if np.max(np.abs(clock)) <= 4:
        return clock
    lo, hi = clock.min() - 4, clock.max() + 4
    for _ in range(50):
        shift = (lo + hi) / 2
        if np.clip(clock - shift, -4, 4).sum() > 0:
            lo = shift
        else:
            hi = shift
    return np.clip(clock - (lo + hi) / 2, -4, 4)


def _invert(survey):
    width, height = survey['W_m'], survey['H_m']
    radius = survey.get('R_plume_m', 5.5)
    sources = np.asarray(survey['z_src_m'])
    receivers = np.asarray(survey['z_rcv_m'])
    data = np.asarray(survey['T_obs_s'], dtype=float).ravel() * 1000
    ns, nr = len(sources), len(receivers)
    source_z = np.repeat(sources, nr)
    dz = np.tile(receivers, ns) - source_z
    length = np.hypot(width, dz)
    c, s = width / length, dz / length
    clock_basis = helmert(nr, full=False).T
    linear = np.column_stack((length / width, -length / width * s**2,
                              np.tile(clock_basis, (ns, 1))))
    nlinear = linear.shape[1]

    def plume(x, z, derivatives=False):
        rel_z = z - source_z
        along = c*x + s*rel_z
        perp = c*rel_z - s*x
        a, b = -along/radius, (length-along)/radius
        taper = np.exp(-0.5*(perp/radius)**2)
        value = taper * (ndtr(b) - ndtr(a))
        if not derivatives:
            return value
        dalong = taper * (np.exp(-0.5*a*a) - np.exp(-0.5*b*b)) / (radius*np.sqrt(2*np.pi))
        dperp = -perp / radius**2 * value
        return value, dalong*c - dperp*s, dalong*s + dperp*c

    def predict(p):
        return linear @ p[:nlinear] + p[-1]*plume(*p[-3:-1])

    def jacobian(p):
        value, dx, dz = plume(*p[-3:-1], derivatives=True)
        return np.column_stack((linear, p[-1]*dx, p[-1]*dz, value))

    # Remove the linear nuisance subspace before looking for plume-shaped signal.
    # A robust background fit supplies weights so cycle skips cannot dominate the scan.
    initial = np.linalg.lstsq(linear, data, rcond=None)[0]
    bg = least_squares(lambda p: linear @ p - data, initial, jac=lambda p: linear.copy(),
                       loss='soft_l1', f_scale=.35).x
    weights = (1 + ((linear @ bg - data) / 1.2)**4)**(-.25)
    q, _ = np.linalg.qr(linear * weights[:, None])
    xy = np.array([(x, z) for x in np.linspace(0, width, int(np.ceil(width/2.5))+1)
                   for z in np.linspace(0, height, int(np.ceil(height/2.5))+1)])
    templates = plume(xy[:, 0, None], xy[:, 1, None]) * weights
    templates -= (templates @ q) @ q.T
    norms = np.maximum(np.linalg.norm(templates, axis=1), 1e-10)
    scores = (templates @ (weights * (data - linear @ bg))) / norms
    lo = np.r_[np.full(nlinear, -np.inf), 0, 0, 0]
    hi = np.r_[np.full(nlinear, np.inf), width, height, np.inf]
    starts = []
    best = None
    for idx in np.argsort(scores)[::-1]:
        center = xy[idx]
        if any(np.linalg.norm(center - prev) < radius for prev in starts):
            continue
        starts.append(center)
        amp = max(scores[idx] / norms[idx], .05)
        init = np.r_[bg, center, amp]
        fit = least_squares(lambda p: predict(p) - data, init, jac=jacobian,
                            bounds=(lo, hi), loss='soft_l1', f_scale=.20, max_nfev=65)
        if best is None or fit.cost < best.cost:
            best = fit
        if len(starts) == 8:
            break

    # Refit the picked inliers without the residual influence of cycle skips.
    p = best.x
    inlier = np.abs(predict(p) - data) < 3.0
    p = least_squares(lambda p: (predict(p) - data)[inlier], p,
                      jac=lambda p: jacobian(p)[inlier], bounds=(lo, hi)).x
    clocks = _project_clocks(clock_basis @ p[2:nlinear])
    return p[-3:-1], clocks / 1000, p, data - predict(p), inlier


def invert_plume_and_clocks(survey: dict) -> dict:
    """Recover plume center (meters) and zero-sum receiver offsets (seconds)."""
    center, clocks, *_ = _invert(survey)
    return {'plume_xz_m': center.tolist(), 'rcv_clock_bias_s': clocks.tolist()}
PY
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,time,numpy as np
from tomography_solver import _invert
for row in pickle.load(open('public_surveys.pkl','rb')):
 t=time.perf_counter(); xy,cl,p,res,inl=_invert(row['survey']); gt=row['gt']; print('time',time.perf_counter()-t,'xy',xy,'error',np.linalg.norm(xy-gt['plume_xz_m']),'clock rmse',np.mean((cl-gt['rcv_clock_bias_s'])**2)**.5,'params',p[:2],p[-1]); print('inlier std',res[inl].std(),'gross ms',res[~inl])
PY
```
- **Container Exit Code**: `0` (`680 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tomography_solver.py']`
- **Stdout (tail)**:
```text
time 0.036602514097467065 xy [16.42106594 34.04212502] error 0.0894712436815506 clock rmse 5.301982457396778e-06 params [16.80502251  2.36444561] 0.894747980710499
inlier std 0.0319090370738482 gross ms [-5.50615978  5.50274298  5.4619393   5.45068342  5.56204882  5.49651389
 -5.49557066  5.52033521 -5.52795654 -5.49967168]
time 0.04453405784443021 xy [25.89839671 51.45256966] error 0.11212879395138935 clock rmse 1.2234518025540354e-05 params [15.20452128  2.74493773] 0.9835645106032401
inlier std 0.031614109182337126 gross ms [-5.5067846   5.46508665  5.51656304 -5.46847209  5.51042192  5.52624913
 -5.51138949  5.51437447  5.51691736  5.49304634]
```

### Turn 4 — `execute_in_remote_docker` (`13568 ms` LLM)
- **Rationale**: Verify the exported API, clock constraints, and public accuracy under the default numerical runtime; if BLAS overhead threatens the limit, reduce the scan’s threaded operations.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,time,numpy as np
from tomography_solver import invert_plume_and_clocks
for row in pickle.load(open('public_surveys.pkl','rb')):
    t=time.perf_counter(); result=invert_plume_and_clocks(row['survey']); elapsed=time.perf_counter()-t
    gt=row['gt']; clocks=np.asarray(result['rcv_clock_bias_s']); center=np.asarray(result['plume_xz_m']); dx=np.linalg.norm(center-gt['plume_xz_m']); rmse=np.sqrt(np.mean((clocks-gt['rcv_clock_bias_s'])**2))
    print(f'{elapsed:.3f} s; center error {dx:.4f} m; clock RMSE {rmse*1e6:.2f} us; clock sum {clocks.sum():.3g} s',flush=True)
    assert elapsed < 15 and dx <= 1.8 and rmse <= .00025
    assert center.shape==(2,) and clocks.shape==row['survey']['z_rcv_m'].shape and abs(clocks.sum())<1e-12 and np.max(np.abs(clocks))<=.004
PY
```
- **Container Exit Code**: `0` (`1614 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.418 s; center error 0.0895 m; clock RMSE 5.30 us; clock sum 1.3e-18 s
0.297 s; center error 0.1121 m; clock RMSE 12.23 us; clock sum -2.6e-18 s
```

### Turn 5 — `submit_completed_task` (`10155 ms` LLM)
- **Rationale**: The saved solver was verified through its public API on both supplied surveys and passed all requested accuracy, clock, and runtime checks.
- **Submitted Final Answer / Summary**: `Implemented and verified `invert_plume_and_clocks` in `/holdout_task/tomography_solver.py`. On the two public surveys, plume center errors were 0.090 m and 0.112 m, clock RMSEs were 5.3 µs and 12.2 µs, and both runs finished in under 0.5 s. Both outputs met the zero-sum and ±4 ms clock constraints.`

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
