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
| **Turns Used / Max Turns** | `7 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `108,059` (`104,317` / `0` / `3,742`) |
| **Total Episode Wall Time (s)** | `137.72s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`12184 ms` LLM)
- **Rationale**: Inspect the public survey schema and available numerical libraries so the solver uses the actual input keys, dimensions, and units.

```bash
cd /holdout_task && python - <<'PY'
import pickle,numpy as np,scipy
p=pickle.load(open('public_surveys.pkl','rb')); print('scipy',scipy.__version__,'public root:',type(p),len(p))
for i,s in (p.items() if isinstance(p,dict) else enumerate(p)):
 print(i,type(s))
 if isinstance(s,dict):
  for k,v in s.items():
   a=np.asarray(v); print(k,a.shape,a.dtype, v if a.size<=16 else (a.flat[:5],np.nanmin(a),np.nanmax(a)) if a.dtype.kind in 'fiu' else '')
 else: print(repr(s)[:500])
PY
```
- **Container Exit Code**: `127` (`307 ms`, timed_out=`False`)
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`11504 ms` LLM)
- **Rationale**: Use python3 to inspect the public survey fields and determine whether SciPy is available for the inversion.

```bash
cd /holdout_task && command -v python3 && python3 - <<'PY'
import pickle
p=pickle.load(open('public_surveys.pkl','rb')); print(type(p),len(p))
for i,s in (p.items() if isinstance(p,dict) else enumerate(p)):
 print(i,type(s))
 if isinstance(s,dict):
  for k,v in s.items(): print(k,getattr(v,'shape',None),str(v)[:240])
 else: print(str(s)[:500])
try:
 import scipy; print('scipy',scipy.__version__)
except ImportError: print('no scipy')
PY
```
- **Container Exit Code**: `0` (`571 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/usr/bin/python3
<class 'list'> 2
0 <class 'dict'>
survey None {'W_m': 40.0, 'H_m': 80.0, 'z_src_m': array([ 6.        , 10.85714286, 15.71428571, 20.57142857, 25.42857143,
       30.28571429, 35.14285714, 40.        , 44.85714286, 49.71428571,
       54.57142857, 59.42857143, 64.28571429, 69.14285714,
gt None {'plume_xz_m': [16.5, 34.0], 'rcv_clock_bias_s': array([-0.0019168 ,  0.00045573,  0.00110994, -0.00255604, -0.00179292,
        0.0032009 , -0.00228896, -0.0019091 ,  0.00332965,  0.00124041,
       -0.00037809,  0.00053325,  0.00150255, -
1 <class 'dict'>
survey None {'W_m': 40.0, 'H_m': 80.0, 'z_src_m': array([ 6.        , 10.85714286, 15.71428571, 20.57142857, 25.42857143,
       30.28571429, 35.14285714, 40.        , 44.85714286, 49.71428571,
       54.57142857, 59.42857143, 64.28571429, 69.14285714,
gt None {'plume_xz_m': [26.0, 51.5], 'rcv_clock_bias_s': array([-9.81201972e-04,  3.47274033e-03, -1.37482804e-03, -1.43901348e-03,
       -3.47187364e-04, -1.11101453e-03,  1.70437425e-03, -1.84997046e-03,
        3.14990149e-03,  2.90555663e-03, 
scipy 1.13.0
```

### Turn 3 — `execute_in_remote_docker` (`9336 ms` LLM)
- **Rationale**: Identify the traveltime field, supplied physics parameters, and array dimensions before implementing the forward model.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
for e in pickle.load(open('public_surveys.pkl','rb')):
 for group,d in e.items():
  print(group)
  for k,v in d.items():
   a=np.asarray(v); print(k,a.shape,a.dtype, a if a.ndim==0 else (a.min(),a.max()))
PY
```
- **Container Exit Code**: `0` (`588 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
survey
W_m () float64 40.0
H_m () float64 80.0
z_src_m (15,) float64 (6.0, 74.0)
z_rcv_m (16,) float64 (5.0, 75.0)
T_obs_s (15, 16) float64 (0.01330233416627665, 0.03317839066777648)
R_plume_m () float64 5.5
gt
plume_xz_m (2,) float64 (16.5, 34.0)
rcv_clock_bias_s (16,) float64 (-0.002556039023045965, 0.003329653424440951)
survey
W_m () float64 40.0
H_m () float64 80.0
z_src_m (15,) float64 (6.0, 74.0)
z_rcv_m (16,) float64 (5.0, 75.0)
T_obs_s (15, 16) float64 (0.01336071363590472, 0.028705721115872333)
R_plume_m () float64 5.5
gt
plume_xz_m (2,) float64 (26.0, 51.5)
rcv_clock_bias_s (16,) float64 (-0.002568385497994811, 0.0034727403303111385)
```

### Turn 4 — `execute_in_remote_docker` (`17066 ms` LLM)
- **Rationale**: Determine whether the straight-ray forward model matches the public data and measure the separation between noise, plume delays, and cycle skips. A mismatch would change the forward model or outlier treatment.

```bash
cd /holdout_task && python3 - <<'PY'
import numpy as np,pickle
from scipy.special import ndtr
from scipy.optimize import least_squares
for e in pickle.load(open('public_surveys.pkl','rb')):
 s,g=e['survey'],e['gt']; a=s['z_src_m'][:,None]; b=s['z_rcv_m'][None,:]; w=s['W_m']; r=s['R_plume_m']; dz=b-a; l=np.hypot(w,dz); xx,zz=g['plume_xz_m']; t=(w*xx+dz*(zz-a))/l; d=(w*(zz-a)-dz*xx)/l
 p=r*np.sqrt(2*np.pi)*np.exp(-.5*(d/r)**2)*(ndtr((l-t)/r)-ndtr(-t/r)); y=1e3*(s['T_obs_s']-g['rcv_clock_bias_s']); A=np.stack([l,l*(dz/l)**2,p],axis=-1).reshape(-1,3); fit=least_squares(lambda v:A@v-y.ravel(),[.4,-.03,.1],loss='huber',f_scale=.35); res=y.ravel()-A@fit.x; keep=abs(res)<3; v=np.linalg.lstsq(A[keep],y.ravel()[keep],rcond=None)[0]; er=y.ravel()-A@v; print('coefs (ms/m)',v,'noise sd (ms)',np.std(er[keep]),'inlier range',er[keep].min(),er[keep].max(),'outliers',np.round(er[~keep],3)); print('clock sum',sum(g['rcv_clock_bias_s']),'plume maximum (ms)',(p*v[-1]).max())
PY
```
- **Container Exit Code**: `0` (`807 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
coefs (ms/m) [ 0.42011467 -0.05908143  0.06487975] noise sd (ms) 0.032527776228506014 inlier range -0.08507092012872164 0.08717676375424688 outliers [-5.513  5.515  5.465  5.462  5.562  5.498 -5.499  5.52  -5.529 -5.504]
clock sum 4.336808689942018e-19 plume maximum (ms) 0.8943725789172232
coefs (ms/m) [ 0.38004243 -0.06870203  0.07261059] noise sd (ms) 0.033178497910216355 inlier range -0.08297605141680364 0.09051888051656931 outliers [-5.516  5.474  5.518 -5.48   5.5    5.522 -5.516  5.525  5.526  5.496]
clock sum -1.5178830414797062e-18 plume maximum (ms) 1.000838781017812
```

### Turn 5 — `execute_in_remote_docker` (`43275 ms` LLM)
- **Rationale**: Implement the finite-ray Gaussian model and robust cycle-skip rejection. Public verification checks for forward-model or clock-sign errors and tests both required accuracy thresholds.
- **Alternatives Weighed**: `Fit all parameters from a single initial center` (rejected: A bounded grid search over the two nonlinear coordinates reduces the chance of settling on a false plume location.)

```bash
cat > /holdout_task/tomography_solver.py <<'PY'
"""Robust straight-ray inversion of cross-well arrival times."""

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.optimize import least_squares
from scipy.special import ndtr


def _clock_projection(clock):
    """Project millisecond offsets onto the zero-sum, ±4 ms clock set."""
    clock = clock - clock.mean()
    if np.max(np.abs(clock)) <= 4.0:
        return clock
    lo, hi = clock.min() - 4.0, clock.max() + 4.0
    for _ in range(48):
        mid = (lo + hi) / 2
        if np.clip(clock - mid, -4.0, 4.0).sum() > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(clock - (lo + hi) / 2, -4.0, 4.0)


def invert_plume_and_clocks(survey: dict) -> dict:
    """Fit plume location and receiver clocks; all input/output times are seconds."""
    width, height = float(survey['W_m']), float(survey['H_m'])
    radius = float(survey.get('R_plume_m', 5.5))
    src = np.asarray(survey['z_src_m'], dtype=float)
    rcv = np.asarray(survey['z_rcv_m'], dtype=float)
    obs = 1000.0 * np.asarray(survey['T_obs_s'], dtype=float).ravel()
    zs = np.repeat(src, rcv.size)
    dz = np.tile(rcv, src.size) - zs
    length = np.hypot(width, dz)
    cos, sin = width / length, dz / length

    # Linear background coefficients are s0 and -s0*eps. Use a
    # contrast basis so the clock offsets cannot absorb a common delay.
    clock_basis = np.eye(rcv.size)[:, :-1] - np.eye(rcv.size)[:, -1:]
    design = np.column_stack((length, length * sin**2,
                              np.tile(clock_basis, (src.size, 1))))

    def template(x, z):
        depth = z - zs
        along = cos * x + sin * depth
        across = cos * depth - sin * x
        # Integrate the Gaussian between the actual source and receiver.
        return (np.sqrt(2 * np.pi) * radius * np.exp(-0.5 * (across / radius)**2)
                * (ndtr((length - along) / radius) - ndtr(-along / radius)))

    # A robust background-only fit gives a first mask. Ordinary plume
    # delays in these surveys are much smaller than a cycle skip.
    init = np.linalg.lstsq(design, obs, rcond=None)[0]
    base = least_squares(lambda v: design @ v - obs, init,
                         jac=lambda v: design.copy(), loss='huber', f_scale=0.18)
    good = np.abs(base.fun) < 2.8
    bounds = ([0.0, 0.0], [width, height])
    gx = np.linspace(0, width, int(np.ceil(2 * width / radius)) + 1)
    gz = np.linspace(0, height, int(np.ceil(2 * height / radius)) + 1)
    grid = np.stack(np.meshgrid(gx, gz), axis=-1).reshape(-1, 2)

    for _ in range(4):
        # Variable projection: eliminate background, clocks and nonnegative
        # plume strength, leaving just the two coordinates to search.
        q, r = qr(design[good], mode='economic')
        qt = q.T.copy()
        target = obs[good]
        qty = qt @ target
        y = target - q @ qty

        def project(p):
            return p - q @ (qt @ p)

        def score(p):
            pp = project(p)
            return max(0.0, float(pp @ y)) / np.sqrt(max(float(pp @ pp), 1e-24))

        def profiled(xz):
            pp = project(template(*xz)[good])
            amp = max(0.0, float(pp @ y)) / max(float(pp @ pp), 1e-24)
            return amp * pp - y

        start = max(grid, key=lambda xy: score(template(*xy)[good]))
        xy = least_squares(profiled, start, bounds=bounds, ftol=1e-10,
                           xtol=1e-9, gtol=1e-9).x
        p = template(*xy)
        pp = project(p[good])
        amp = max(0.0, float(pp @ y)) / max(float(pp @ pp), 1e-24)
        coef = solve_triangular(r, qty - amp * (qt @ p[good]))
        refined = np.abs(obs - design @ coef - amp * p) < 2.8
        if np.array_equal(refined, good):
            break
        good = refined

    clock = _clock_projection(clock_basis @ coef[2:])
    return {'plume_xz_m': xy.tolist(), 'rcv_clock_bias_s': (clock / 1000.0).tolist()}
PY
cd /holdout_task && python3 - <<'PY'
import pickle,time
import numpy as np
from tomography_solver import invert_plume_and_clocks
for e in pickle.load(open('public_surveys.pkl','rb')):
 t=time.perf_counter(); pred=invert_plume_and_clocks(e['survey']); g=e['gt']; d=np.linalg.norm(np.subtract(pred['plume_xz_m'],g['plume_xz_m'])); c=np.sqrt(np.mean(np.subtract(pred['rcv_clock_bias_s'],g['rcv_clock_bias_s'])**2)); print(pred['plume_xz_m'], 'center error m',d,'clock RMSE s',c,'seconds',time.perf_counter()-t); assert d<=1.8 and c<=.00025; assert abs(sum(pred['rcv_clock_bias_s'])) < 1e-12 and max(map(abs,pred['rcv_clock_bias_s'])) <= .004000000000001
PY
```
- **Container Exit Code**: `0` (`857 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tomography_solver.py']`
- **Stdout (tail)**:
```text
[16.42106596102565, 34.04212502430953] center error m 0.08947122544082184 clock RMSE s 5.30198253045244e-06 seconds 0.02781897597014904
[25.898396761653895, 51.45256962712898] center error m 0.11212875774349454 clock RMSE s 1.2234517301432259e-05 seconds 0.014995049219578505
```

### Turn 6 — `execute_in_remote_docker` (`19667 ms` LLM)
- **Rationale**: The public cases pass. Varying geometry, plume strength and skip placement tests whether the initialization misses a plausible plume or misidentifies cycle skips; failures would motivate changing the search or rejection step.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import numpy as np,time
from scipy.special import ndtr
from tomography_solver import invert_plume_and_clocks
rng=np.random.default_rng(47); results=[]; start=time.perf_counter()
for k in range(100):
 w,h=rng.uniform(30,65),rng.uniform(65,105); r=5.5
 a=np.linspace(5,h-6,15)[:,None]; b=np.linspace(4,h-5,16)[None,:]; dz=b-a; l=np.hypot(w,dz)
 x,z=rng.uniform([5,9],[w-5,h-9]); along=(w*x+dz*(z-a))/l; perp=(w*(z-a)-dz*x)/l
 p=r*np.sqrt(2*np.pi)*np.exp(-.5*(perp/r)**2)*(ndtr((l-along)/r)-ndtr(-along/r))
 c=rng.normal(0,.0015,16); c-=c.mean(); c*=min(1,.004/max(abs(c)))
 amp=rng.uniform(.00004,.00012); eps=rng.uniform(.04,.23); s0=rng.uniform(.00032,.00051)
 obs=l*s0*(1-eps*(dz/l)**2)+amp*p+c+rng.normal(0,.00005,l.shape)
 ix=rng.choice(obs.size,10,replace=False); obs.flat[ix]+=rng.choice([-.006,.006],10)
 fit=invert_plume_and_clocks(dict(W_m=w,H_m=h,R_plume_m=r,z_src_m=a.ravel(),z_rcv_m=b.ravel(),T_obs_s=obs)); d=np.linalg.norm(np.asarray(fit['plume_xz_m'])-[x,z]); err=np.sqrt(np.mean((np.asarray(fit['rcv_clock_bias_s'])-c)**2)); results.append((d,err))
 if d>1.8 or err>.00025: print('failure',k,'center',[x,z],'geometry',[w,h],'amp',amp,'errors',d,err)
print('100 simulations: max center and clock errors',np.max(results,axis=0),'elapsed',time.perf_counter()-start)
PY
```
- **Container Exit Code**: `0` (`2991 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
100 simulations: max center and clock errors [1.70072724e+00 4.33227296e-05] elapsed 2.3969566200394183
```

### Turn 7 — `submit_completed_task` (`9775 ms` LLM)
- **Rationale**: The artifact exists and passed the public accuracy, runtime, and clock-constraint checks. All 100 additional synthetic surveys also met both accuracy targets.
- **Submitted Final Answer / Summary**: `Implemented and verified `invert_plume_and_clocks` in `/holdout_task/tomography_solver.py`. On the two public surveys, plume-center errors were 0.089 m and 0.112 m, and receiver-clock RMSEs were 5.3 µs and 12.2 µs. Both runs finished in under 0.03 s and met the clock bounds and zero-sum constraint. Held-out performance was not measured.`

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
