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
| **Turns Used / Max Turns** | `6 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `117,634` (`113,923` / `22,016` / `3,711`) |
| **Total Episode Wall Time (s)** | `149.12s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`14628 ms` LLM)
- **Rationale**: Inspect the supplied fixture to resolve field names, dimensions, and whether background parameters are provided; these determine the forward model and fit.

```bash
cd /holdout_task && ls -la && python - <<'PY'
import pickle
import numpy as np
with open('public_surveys.pkl','rb') as f: d=pickle.load(f)
def describe(d, depth=0):
 print(' '*depth,type(d).__name__,getattr(d,'shape',''))
 if isinstance(d,dict):
  for k,v in d.items():
   print(' '*depth,repr(k),end=': ')
   if isinstance(v,dict): describe(v,depth+2)
   else:
    a=np.asarray(v); print(a.shape,a.dtype,str(v)[:320] if a.size<16 else str(a.flat[:6]))
 elif isinstance(d,(tuple,list)):
  print('length:',len(d))
  for item in d[:4]: describe(item,depth+2)
describe(d)
import scipy
print('scipy',scipy.__version__)
PY
```
- **Container Exit Code**: `127` (`345 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 16
drwxr-xr-x 2 root root 4096 Nov 24 08:12 .
drwxr-xr-x 1 root root 4096 Nov 24 08:12 ..
-rw-r--r-- 1 root root 5205 Nov 24 08:12 public_surveys.pkl
```
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`17092 ms` LLM)
- **Rationale**: Use the available Python 3 interpreter to identify survey fields and determine which numerical libraries can be used.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle, importlib.util
from pprint import pprint
with open('public_surveys.pkl','rb') as f: d=pickle.load(f)
print(type(d),len(d))
def describe(o,depth=0):
 if isinstance(o,dict):
  for k,v in o.items():
   print(' '*depth,k,type(v).__name__,getattr(v,'shape',None),end=' ')
   if isinstance(v,dict): print(); describe(v,depth+2)
   else: print(repr(v)[:245])
 elif isinstance(o,(list,tuple)):
  for v in o[:5]: describe(v,depth+1)
describe(d)
for p in ('numpy','scipy'): print(p,importlib.util.find_spec(p))
PY
```
- **Container Exit Code**: `0` (`519 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
33.        , 37.66666667, 42.33333333, 47.        ,
       51.66666667, 56.33333333, 61.        , 65.66666667, 70.33333333,
       75.        ])
    T_obs_s ndarray (15, 16) array([[0.01491584, 0.01729941, 0.01816543, 0.0148971 , 0.01612493,
        0.02175942, 0.01714121, 0.01843184, 0.02471534, 0.02387659,
        0.01807355, 0.02581797, 0.02816051, 0.02712464, 0.02766659,
        0.03317839],
       [0.01498645, 
    R_plume_m float None 5.5
  gt dict None 
    plume_xz_m list None [16.5, 34.0]
    rcv_clock_bias_s ndarray (16,) array([-0.0019168 ,  0.00045573,  0.00110994, -0.00255604, -0.00179292,
        0.0032009 , -0.00228896, -0.0019091 ,  0.00332965,  0.00124041,
       -0.00037809,  0.00053325,  0.00150255, -0.00097767, -0.00185665,
        0.0023038 ])
  survey dict None 
    W_m float None 40.0
    H_m float None 80.0
    z_src_m ndarray (15,) array([ 6.        , 10.85714286, 15.71428571, 20.57142857, 25.42857143,
       30.28571429, 35.14285714, 40.        , 44.85714286, 49.71428571,
       54.57142857, 59.42857143, 64.28571429, 69.14285714, 74.        ])
    z_rcv_m ndarray (16,) array([ 5.        ,  9.66666667, 14.33333333, 19.        , 23.66666667,
       28.33333333, 33.        , 37.66666667, 42.33333333, 47.        ,
       51.66666667, 56.33333333, 61.        , 65.66666667, 70.33333333,
       75.        ])
    T_obs_s ndarray (15, 16) array([[0.01425288, 0.0187468 , 0.01403325, 0.0142902 , 0.01580415,
        0.01554096, 0.01899251, 0.0161894 , 0.02200229, 0.02269381,
        0.01829848, 0.02293383, 0.02149979, 0.0237031 , 0.02609277,
        0.02200977],
       [0.01430374, 
    R_plume_m float None 5.5
  gt dict None 
    plume_xz_m list None [26.0, 51.5]
    rcv_clock_bias_s ndarray (16,) array([-9.81201972e-04,  3.47274033e-03, -1.37482804e-03, -1.43901348e-03,
       -3.47187364e-04, -1.11101453e-03,  1.70437425e-03, -1.84997046e-03,
        3.14990149e-03,  2.90555663e-03, -2.56838550e-03,  8.78904931e-04,
       -1.90263035e-
numpy ModuleSpec(name='numpy', loader=<_frozen_importlib_external.SourceFileLoader object at 0x7b28850153a0>, origin='/usr/local/lib/python3.12/dist-packages/numpy/__init__.py', submodule_search_locations=['/usr/local/lib/python3.12/dist-packages/numpy'])
scipy ModuleSpec(name='scipy', loader=<_frozen_importlib_external.SourceFileLoader object at 0x7b287b918ef0>, origin='/usr/local/lib/python3.12/dist-packages/scipy/__init__.py', submodule_search_locations=['/usr/local/lib/python3.12/dist-packages/scipy'])
```

### Turn 3 — `execute_in_remote_docker` (`23701 ms` LLM)
- **Rationale**: Measure the public surveys’ noise and skip magnitudes after fitting the stated ray model at the labeled centers. If skips are not close to ±6 ms, the inversion should reject them rather than subtract a fixed shift.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
from scipy.special import erf
from scipy.optimize import least_squares
for entry in pickle.load(open('public_surveys.pkl','rb')):
 s,g=entry['survey'],entry['gt']; w=s['W_m']; r=s['R_plume_m']; z=s['z_src_m'][:,None]; d=s['z_rcv_m'][None,:]-z; L=np.hypot(w,d); u=w/L; v=d/L; y=1000*(s['T_obs_s']-g['rcv_clock_bias_s']); x,p=g['plume_xz_m']; a=p-z; q=u*x+v*a; t=u*a-v*x; K=r*np.sqrt(np.pi/2)*np.exp(-.5*(t/r)**2)*(erf((L-q)/(np.sqrt(2)*r))+erf(q/(np.sqrt(2)*r))); A=np.stack([L,L*v*v,K],axis=-1).reshape(-1,3); fit=least_squares(lambda b: A@b-y.ravel(),[.42,-.03,.1],loss='soft_l1',f_scale=.15); e=(y.ravel()-A@fit.x); print('fit',fit.x,'residual abs quantiles in ms',np.quantile(abs(e),[.25,.5,.75,.9,.95,.96,.97,.99,1]),'outliers:',np.round(e[abs(e)>1.5],3)); print('clock mean',np.mean(g['rcv_clock_bias_s']))
PY
```
- **Container Exit Code**: `0` (`552 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
fit [ 0.42028606 -0.05951203  0.06479411] residual abs quantiles in ms [0.01197    0.02417677 0.03888639 0.06101665 0.07662838 2.45272752
 5.49100826 5.51181541 5.55805947] outliers: [-5.508  5.509  5.46   5.459  5.558  5.504 -5.499  5.514 -5.52  -5.497]
clock mean 2.710505431213761e-20
fit [ 0.3801998  -0.06907494  0.07279236] residual abs quantiles in ms [0.01012756 0.0204387  0.0408559  0.06523566 0.08413355 2.45574316
 5.49571349 5.51628368 5.51950051] outliers: [-5.508  5.468  5.518 -5.482  5.498  5.514 -5.512  5.519  5.52   5.5  ]
clock mean -5.421010862427522e-20
```

### Turn 4 — `execute_in_remote_docker` (`51531 ms` LLM)
- **Rationale**: Implement the finite-ray Gaussian integral and separate linear background and clock parameters from the spatial search. Compare both public predictions to the specified accuracy, clock constraints, and runtime; failures would indicate whether to revise the forward fit, skip filtering, or search.
- **Alternatives Weighed**: `Subtract a fixed cycle-skip offset` (rejected: The public skips are about 5.5 ms, not the nominal 6 ms. Rejecting skips avoids assuming their exact size.)

```bash
cat > /holdout_task/tomography_solver.py <<'PY'
"""Straight-ray inversion for a Gaussian plume and receiver clock offsets."""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


def _bounded_clocks(clocks):
    """Project onto the zero-sum, +/-4 ms clock constraint."""
    if np.max(np.abs(clocks)) <= 4.0:
        return clocks
    lo, hi = np.min(clocks) - 4.0, np.max(clocks) + 4.0
    for _ in range(48):
        mid = (lo + hi) / 2.0
        if np.sum(np.clip(clocks - mid, -4.0, 4.0)) > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(clocks - (lo + hi) / 2.0, -4.0, 4.0)


def invert_plume_and_clocks(survey: dict) -> dict:
    """Estimate the plume center (m) and receiver clock biases (s)."""
    width, height = float(survey['W_m']), float(survey['H_m'])
    radius = float(survey.get('R_plume_m', 5.5))
    src = np.asarray(survey['z_src_m'], dtype=float)
    rcv = np.asarray(survey['z_rcv_m'], dtype=float)
    y = 1000.0 * np.asarray(survey['T_obs_s'], dtype=float).ravel()
    ns, nr = len(src), len(rcv)
    z0 = np.repeat(src, nr)
    dz = np.tile(rcv, ns) - z0
    length = np.hypot(width, dz)
    u, v = width / length, dz / length
    # The two background terms estimate s0 and -s0*epsilon separately.
    # Clock basis columns all sum to zero across the receiver array.
    clock_basis = np.eye(nr)[:, :-1] - 1.0 / nr
    design = np.column_stack((length, length * v**2, np.tile(clock_basis, (ns, 1))))
    scale = np.sqrt(2.0) * radius
    factor = np.sqrt(np.pi / 2.0) * radius

    def kernel(x, z, rows):
        """Integrate the unit-amplitude Gaussian over the finite ray segment."""
        uz, vz = u[rows], v[rows]
        rel_z = z - z0[rows]
        along = uz * x + vz * rel_z
        across = (uz * rel_z - vz * x) / radius
        return factor * np.exp(-0.5 * across**2) * (
            erf((length[rows] - along) / scale) + erf(along / scale)
        )

    # A robust background-only fit removes clock jumps and identifies gross
    # skips before the spatial search. Revisit this classification afterwards.
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(14):
        e = y - design @ beta
        w = (1.0 + (e / 0.18)**2)**(-0.25)
        beta = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)[0]
    good = np.abs(y - design @ beta) < 2.5

    def profile(rows):
        a = design[rows]
        q, r = np.linalg.qr(a, mode='reduced')

        def project(values):
            coords = np.einsum('...i,ij->...j', values, q)
            return values - np.einsum('...j,ij->...i', coords, q)

        target = project(y[rows])

        def fit_plume(x, z):
            k = project(kernel(x, z, rows))
            power = np.sum(k * k, axis=-1)
            correlation = np.maximum(0.0, k @ target if k.ndim == 1 else
                                     np.einsum('ij,j->i', k, target))
            amp = correlation / np.maximum(power, 1e-20)
            return k, amp, correlation * amp

        return fit_plume, target, q, r

    fit, _, _, _ = profile(good)
    gx, gz = np.meshgrid(np.linspace(0.0, width, int(np.ceil(width / 2.0)) + 1),
                         np.linspace(0.0, height, int(np.ceil(height / 2.0)) + 1))
    grid = np.column_stack((gx.ravel(), gz.ravel()))
    gains = []
    for block in np.array_split(grid, max(1, int(np.ceil(len(grid) / 128)))):
        gains.extend(fit(block[:, :1], block[:, 1:],)[2])
    center = grid[np.argmax(gains)]

    for refit in range(4):
        fit, target, q, r = profile(good)

        def residual(p):
            k, amp, _ = fit(*p)
            return amp * k - target

        center = least_squares(residual, center, bounds=([0.0, 0.0], [width, height]),
                               max_nfev=50).x
        _, amp, _ = fit(*center)
        plume = amp * kernel(*center, slice(None))
        beta = np.linalg.solve(r, q.T @ (y[good] - plume[good]))
        new_good = np.abs(y - design @ beta - plume) < 2.5
        if np.array_equal(good, new_good):
            break
        good = new_good

    clocks = _bounded_clocks(clock_basis @ beta[2:])
    return {'plume_xz_m': center.tolist(), 'rcv_clock_bias_s': (clocks / 1000.0).tolist()}
PY
cd /holdout_task && python3 - <<'PY'
import pickle,time,numpy as np
from tomography_solver import invert_plume_and_clocks
for item in pickle.load(open('public_surveys.pkl','rb')):
 start=time.perf_counter(); p=invert_plume_and_clocks(item['survey']); g=item['gt']; c=np.array(p['rcv_clock_bias_s']); e=np.linalg.norm(np.subtract(p['plume_xz_m'],g['plume_xz_m'])); rmse=np.sqrt(np.mean((c-g['rcv_clock_bias_s'])**2)); t=time.perf_counter()-start
 print(p['plume_xz_m'], 'center error m',e,'clock RMSE s',rmse,'sum',sum(c),'max',max(abs(c)),'runtime s',t)
 assert e<=1.8 and rmse<=0.00025 and t<15
 assert abs(sum(c))<1e-12 and max(abs(c))<=.004
PY
```
- **Container Exit Code**: `0` (`929 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tomography_solver.py']`
- **Stdout (tail)**:
```text
[16.42106603321265, 34.04212506948504] center error m 0.08947118302507116 clock RMSE s 5.301982872450558e-06 sum 8.673617379884035e-19 max 0.003332209253617547 runtime s 0.019251981051638722
[25.898396644211832, 51.452569731306546] center error m 0.11212882009434591 clock RMSE s 1.223451877695529e-05 sum -2.168404344971009e-19 max 0.0034568601094382414 runtime s 0.017079284880310297
```

### Turn 5 — `execute_in_remote_docker` (`20400 ms` LLM)
- **Rationale**: Use independent numerical ray integration at centers near both wells, which the public fixtures do not cover. If these fail, revise the spatial initialization or skip filtering before submission.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle
import numpy as np
from tomography_solver import invert_plume_and_clocks
s = pickle.load(open('public_surveys.pkl','rb'))[0]['survey']
a = s['z_src_m'][:,None]; d = s['z_rcv_m'][None,:]-a; w = s['W_m']; L = np.hypot(w,d)
t = np.linspace(0,1,401); wt = np.ones(t.size); wt[[0,-1]] = .5; wt /= (t.size-1)
for seed,(x,z,amp) in enumerate([(3,15,.000075),(37,65,.00009),(11,59,.00013),(30,24,.00013)]):
 rng = np.random.default_rng(seed); c = rng.uniform(-1,1,d.shape[1]); c -= c.mean(); c *= .004/np.max(abs(c));
 g = np.exp(-((w*t-x)**2+(a[...,None]+d[...,None]*t-z)**2)/(2*5.5**2)); y = L*(.0004*(1-.17*(d/L)**2)+amp*np.sum(g*wt,axis=-1))+c+rng.normal(0,.00007,d.shape)
 y.flat[rng.choice(y.size,10,replace=False)] += .006*rng.choice([-1,1],10)
 pred=invert_plume_and_clocks(dict(s,T_obs_s=y)); pe=np.linalg.norm(np.subtract(pred['plume_xz_m'],[x,z])); ce=np.sqrt(np.mean((np.array(pred['rcv_clock_bias_s'])-c)**2)); print(f'center ({x},{z}): {pe:.4f} m, clocks {ce:.8f} s'); assert pe < 1.8 and ce < .00025
PY
```
- **Container Exit Code**: `0` (`961 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
center (3,15): 0.6851 m, clocks 0.00001820 s
center (37,65): 0.7801 m, clocks 0.00001403 s
center (11,59): 0.0917 m, clocks 0.00002132 s
center (30,24): 0.0466 m, clocks 0.00001756 s
```

### Turn 6 — `submit_completed_task` (`11207 ms` LLM)
- **Rationale**: The required artifact was written and verified against both public surveys, including the clock constraints and runtime. Independent synthetic tests also passed.
- **Submitted Final Answer / Summary**: `Implemented `invert_plume_and_clocks` in `/holdout_task/tomography_solver.py`. Both public surveys met the targets: plume-center errors were 0.089 m and 0.112 m, clock RMSEs were 5.3 µs and 12.2 µs, and each inversion took under 0.02 s. Four synthetic tests also met both accuracy targets, including cases near each well. Held-out performance was not measured.`

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
