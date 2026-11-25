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
| **Turns Used / Max Turns** | `5 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `75,369` (`71,894` / `13,056` / `3,475`) |
| **Total Episode Wall Time (s)** | `115.46s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`13861 ms` LLM)
- **Rationale**: Inspect the supplied survey schema, dimensions, value scales, and available numerical library; these determine which parameters must be estimated and how to implement the fit.

```bash
ls -la /holdout_task; python - <<'PY'
import pickle, numpy as np
with open('/holdout_task/public_surveys.pkl','rb') as f: d=pickle.load(f)
def inspect(o, indent=''):
 if isinstance(o,dict):
  for k,v in o.items():
   print(indent,repr(k),end=': ')
   if isinstance(v,(dict,list,tuple)): print(); inspect(v,indent+'  ')
   elif isinstance(v,np.ndarray): print(v.shape,v.dtype,'min/max',np.min(v),np.max(v),'first',v.ravel()[:8])
   else: print(repr(v))
 elif isinstance(o,(list,tuple)):
  print(indent,type(o).__name__,len(o))
  if len(o) and isinstance(o[0],(dict,list,tuple)):
   for v in o[:5]: inspect(v,indent+'  ')
  else: print(indent,repr(o)[:300])
inspect(d)
import scipy
print('scipy',scipy.__version__)
PY
```
- **Container Exit Code**: `127` (`332 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 16
drwxr-xr-x 2 root root 4096 Nov 25 18:53 .
drwxr-xr-x 1 root root 4096 Nov 25 18:53 ..
-rw-r--r-- 1 root root 5205 Nov 25 18:53 public_surveys.pkl
```
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`12406 ms` LLM)
- **Rationale**: Use python3 after the missing python executable; inspect field names and available dependencies before choosing the fit implementation.

```bash
command -v python3; python3 - <<'PY'
import pickle
with open('/holdout_task/public_surveys.pkl','rb') as f: data=pickle.load(f)
def describe(x,p=''):
 if isinstance(x,dict):
  for k,v in x.items(): describe(v,p+'/'+str(k))
 elif isinstance(x,(tuple,list)):
  print(p,type(x).__name__,len(x))
  if len(x) and isinstance(x[0],(dict,tuple,list)): 
   for i,v in enumerate(x[:3]): describe(v,p+'/'+str(i))
  else: print(repr(x)[:300])
 elif hasattr(x,'shape'): print(p,x.shape,x.dtype,'range',x.min(),x.max(),'start',x.ravel()[:6])
 else: print(p,repr(x)[:300])
describe(data)
import scipy
print('scipy',scipy.__version__)
PY
```
- **Container Exit Code**: `0` (`510 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/usr/bin/python3
 list 2
/0/survey/W_m 40.0
/0/survey/H_m 80.0
/0/survey/z_src_m (15,) float64 range 6.0 74.0 start [ 6.         10.85714286 15.71428571 20.57142857 25.42857143 30.28571429]
/0/survey/z_rcv_m (16,) float64 range 5.0 75.0 start [ 5.          9.66666667 14.33333333 19.         23.66666667 28.33333333]
/0/survey/T_obs_s (15, 16) float64 range 0.01330233416627665 0.03317839066777648 start [0.01491584 0.01729941 0.01816543 0.0148971  0.01612493 0.02175942]
/0/survey/R_plume_m 5.5
/0/gt/plume_xz_m list 2
[16.5, 34.0]
/0/gt/rcv_clock_bias_s (16,) float64 range -0.002556039023045965 0.003329653424440951 start [-0.0019168   0.00045573  0.00110994 -0.00255604 -0.00179292  0.0032009 ]
/1/survey/W_m 40.0
/1/survey/H_m 80.0
/1/survey/z_src_m (15,) float64 range 6.0 74.0 start [ 6.         10.85714286 15.71428571 20.57142857 25.42857143 30.28571429]
/1/survey/z_rcv_m (16,) float64 range 5.0 75.0 start [ 5.          9.66666667 14.33333333 19.         23.66666667 28.33333333]
/1/survey/T_obs_s (15, 16) float64 range 0.01336071363590472 0.028705721115872333 start [0.01425288 0.0187468  0.01403325 0.0142902  0.01580415 0.01554096]
/1/survey/R_plume_m 5.5
/1/gt/plume_xz_m list 2
[26.0, 51.5]
/1/gt/rcv_clock_bias_s (16,) float64 range -0.002568385497994811 0.0034727403303111385 start [-0.0009812   0.00347274 -0.00137483 -0.00143901 -0.00034719 -0.00111101]
scipy 1.13.0
```

### Turn 3 — `execute_in_remote_docker` (`17133 ms` LLM)
- **Rationale**: Estimate noise and skip separation on the public surveys using the known centers and clocks. If residuals do not separate near ±6 ms, use a continuous robust loss rather than a cycle-aware refit.

```bash
OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
from scipy.special import ndtr
from scipy.optimize import least_squares
for d in pickle.load(open('/holdout_task/public_surveys.pkl','rb')):
 s,g=d['survey'],d['gt']; w=s['W_m']; r=s['R_plume_m']; z=s['z_src_m'][:,None]; dz=s['z_rcv_m']-z; l=np.hypot(w,dz); x,p=g['plume_xz_m']; t=(w*x+dz*(p-z))/l; q=(w*(p-z)-dz*x)/l
 I=np.sqrt(2*np.pi)*r*np.exp(-.5*(q/r)**2)*(ndtr((l-t)/r)-ndtr(-t/r)); A=np.stack((l,-dz**2/l,I),axis=-1).reshape(-1,3); y=((s['T_obs_s']-g['rcv_clock_bias_s'])*1000).ravel(); fit=least_squares(lambda p:A@p-y,[.4,.03,.1],loss='soft_l1',f_scale=.2); e=y-A@fit.x
 inl=abs(e)<2; print('nuisance',fit.x,'peak plume ms',max(I.ravel())*fit.x[-1],'inlier std ms',np.std(e[inl]),'absolute-error percentiles',np.percentile(abs(e),[25,50,75,90,95,96,97,99,100]),'outliers',np.round(e[~inl],3)); print('clock sum',sum(g['rcv_clock_bias_s']))
PY
```
- **Container Exit Code**: `0` (`528 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
nuisance [0.42033775 0.05964316 0.06476657] peak plume ms 0.8928124108308431 inlier std ms 0.033205218820529896 absolute-error percentiles [0.01196755 0.02460986 0.03902423 0.06229201 0.07793316 2.45324088
 5.48914807 5.50996075 5.55699107] outliers [-5.506  5.507  5.458  5.458  5.557  5.505 -5.499  5.512 -5.517 -5.496]
clock sum 4.336808689942018e-19
nuisance [0.38024336 0.06917954 0.07285066] peak plume ms 1.0041478807375774 inlier std ms 0.03357097658515113 absolute-error percentiles [0.01008647 0.02091084 0.04146821 0.06560568 0.08568702 2.45590579
 5.49559758 5.51530917 5.51775812] outliers [-5.506  5.466  5.517 -5.483  5.498  5.512 -5.51   5.518  5.518  5.501]
clock sum -1.5178830414797062e-18
```

### Turn 4 — `execute_in_remote_docker` (`51121 ms` LLM)
- **Rationale**: Implement the finite-ray forward model, robust skip rejection, and a coarse-to-fine profiled inversion. Check the public error targets, clock constraints, and runtime to catch model or implementation failures.
- **Alternatives Weighed**: `Joint nonlinear robust regression from a single initial center` (rejected: The plume-center objective can have local minima, and the background and clocks can be solved linearly at each trial center.); `Correct every skipped pick by a fixed cycle offset` (rejected: The public offsets are near 5.5 ms while the specification says approximately 6 ms; rejecting separated skips avoids depending on their exact magnitude.)

```bash
cat > /holdout_task/tomography_solver.py <<'PY'
"""Straight-ray Gaussian plume tomography with receiver clock calibration."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr


def _bounded_clocks(clocks):
    """Project milliseconds onto the zero-sum clock box, if necessary."""
    if np.max(np.abs(clocks)) <= 4.0:
        return clocks
    lo, hi = np.min(clocks) - 4.0, np.max(clocks) + 4.0
    for _ in range(48):
        mid = (lo + hi) / 2
        if np.clip(clocks - mid, -4.0, 4.0).sum() > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(clocks - (lo + hi) / 2, -4.0, 4.0)


def invert_plume_and_clocks(survey: dict) -> dict:
    """Estimate the plume center (m) and receiver timing corrections (s)."""
    w = float(survey['W_m'])
    h = float(survey['H_m'])
    radius = float(survey.get('R_plume_m', 5.5))
    sources = np.asarray(survey['z_src_m'], dtype=float)
    receivers = np.asarray(survey['z_rcv_m'], dtype=float)
    times = 1000 * np.asarray(survey['T_obs_s'], dtype=float).ravel()
    ns, nr = len(sources), len(receivers)
    zs = np.repeat(sources, nr)
    dz = np.tile(receivers, ns) - zs
    length = np.hypot(w, dz)
    ux, uz = w / length, dz / length
    # Linear coefficients are s0 and s0*epsilon, in ms/m, followed by
    # independent clock offsets; the last receiver closes the zero-sum gauge.
    basis = np.vstack((np.eye(nr - 1), -np.ones(nr - 1)))
    linear = np.column_stack((length, -dz * uz, np.tile(basis, (ns, 1))))

    def integral(x, z):
        """Analytic line integral between the two wells, not an infinite ray."""
        depth = z - zs
        along = (ux * x + uz * depth) / radius
        cross = (ux * depth - uz * x) / radius
        return (np.sqrt(2 * np.pi) * radius * np.exp(-0.5 * cross**2)
                * (ndtr(length / radius - along) - ndtr(-along)))

    def dot_columns(a, b):
        return np.einsum('ij,i->j', a, b)

    def matvec(a, b):
        return np.einsum('ij,j->i', a, b)

    # Background-only Huber fit gives a first mask without allowing a skipped
    # pick to drag its entire receiver column away from the other picks.
    weights = np.ones_like(times)
    for _ in range(12):
        coef = np.linalg.lstsq(linear * weights[:, None], times * weights, rcond=None)[0]
        residual = times - matvec(linear, coef)
        weights = np.sqrt(0.25 / np.maximum(0.25, np.abs(residual)))
    mask = np.abs(residual) < 3.0

    def fit_center(mask, seeds):
        a = linear[mask]
        y = times[mask]
        q, _ = np.linalg.qr(a, mode='reduced')
        y_perp = y - matvec(q, dot_columns(q, y))

        def profile(center):
            g = integral(*center)[mask]
            g -= matvec(q, dot_columns(q, g))
            norm2 = g @ g
            if norm2 == 0:
                return 0.0
            return -max(0.0, g @ y_perp)**2 / norm2

        if seeds is None:
            # A whole-domain coarse search avoids a local plume-center minimum.
            step = radius / 2
            xs = np.linspace(0, w, int(np.ceil(w / step)) + 1)
            zs_grid = np.linspace(0, h, int(np.ceil(h / step)) + 1)
            grid = [(x, z) for x in xs for z in zs_grid]
            ranked = sorted(grid, key=profile)
            seeds = []
            for point in ranked:
                if all(np.hypot(point[0] - x, point[1] - z) >= radius for x, z in seeds):
                    seeds.append(point)
                    if len(seeds) == 5:
                        break

        fits = [minimize(profile, seed, method='L-BFGS-B',
                         bounds=((0, w), (0, h)), options={'ftol': 1e-12})
                for seed in seeds]
        best = min(fits, key=lambda fit: fit.fun)
        g_all = integral(*best.x)
        g = g_all[mask]
        g_perp = g - matvec(q, dot_columns(q, g))
        norm2 = g_perp @ g_perp
        amplitude = max(0.0, (g_perp @ y_perp) / norm2) if norm2 else 0.0
        coeff = np.linalg.lstsq(a, y - amplitude * g, rcond=None)[0]
        return best.x, coeff, times - matvec(linear, coeff) - amplitude * g_all

    center, coef, residual = fit_center(mask, None)
    for _ in range(3):
        updated = np.abs(residual) < 2.5
        if np.array_equal(updated, mask):
            break
        mask = updated
        center, coef, residual = fit_center(mask, [center])

    clocks = _bounded_clocks(basis @ coef[2:]) / 1000
    return {'plume_xz_m': center.tolist(), 'rcv_clock_bias_s': clocks.tolist()}
PY
cd /holdout_task && python3 - <<'PY'
import pickle, time
import numpy as np
from tomography_solver import invert_plume_and_clocks
for row in pickle.load(open('public_surveys.pkl','rb')):
    start=time.perf_counter(); result=invert_plume_and_clocks(row['survey']); elapsed=time.perf_counter()-start
    gt=row['gt']; clocks=np.array(result['rcv_clock_bias_s']); center=np.array(result['plume_xz_m'])
    err=np.linalg.norm(center-gt['plume_xz_m']); rmse=np.sqrt(np.mean((clocks-gt['rcv_clock_bias_s'])**2))
    print(result['plume_xz_m'], 'center error:',err,'clock RMSE:',rmse,'clock sum:',clocks.sum(),'time:',elapsed)
    assert err<=1.8 and rmse<=.00025 and elapsed<15
    assert abs(clocks.sum())<1e-12 and np.max(np.abs(clocks))<=.004
PY
```
- **Container Exit Code**: `0` (`831 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tomography_solver.py']`
- **Stdout (tail)**:
```text
[16.421063848806735, 34.04212451114349] center error: 0.08947284730176006 clock RMSE: 5.301974388967698e-06 clock sum: -2.168404344971009e-19 time: 0.023969755042344332
[25.898396747965915, 51.452569046858954] center error: 0.1121290156015376 clock RMSE: 1.2234518988251804e-05 clock sum: 0.0 time: 0.031364190159365535
```

### Turn 5 — `submit_completed_task` (`10486 ms` LLM)
- **Rationale**: The solver is saved in the required location, and both public surveys passed the accuracy, clock-constraint, and runtime checks.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tomography_solver.py`. On both public surveys, plume-center error was at most 0.113 m, receiver-clock RMSE was at most 12.3 µs, and runtime was under 0.04 s per survey. Both results satisfied the clock bounds and zero-sum constraint. Held-out accuracy was not measured.`

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
