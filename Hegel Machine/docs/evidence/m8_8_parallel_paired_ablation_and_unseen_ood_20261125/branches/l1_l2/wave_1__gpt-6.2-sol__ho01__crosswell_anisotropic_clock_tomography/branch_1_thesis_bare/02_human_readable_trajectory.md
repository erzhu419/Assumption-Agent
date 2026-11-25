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
| **Total Tokens (Prompt / Cached / Out)** | `78,810` (`74,567` / `26,240` / `4,243`) |
| **Total Episode Wall Time (s)** | `149.2s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`21237 ms` LLM)
- **Rationale**: Inspect the public pickle to identify the actual keys, array shapes, and supplied background parameters; those determine which quantities the solver must estimate.

```bash
cd /holdout_task && ls -l && python - <<'PY'
import pickle, numpy as np, scipy
with open('public_surveys.pkl','rb') as f: p=pickle.load(f)
print('scipy', scipy.__version__)
def show(o,depth=0):
 if isinstance(o,dict):
  for k,v in o.items():
   if isinstance(v,(list,tuple,dict,np.ndarray)): print(' '*depth,k);show(v,depth+1)
   else: print(' '*depth,k,repr(v))
 elif isinstance(o,(list,tuple)) and o and isinstance(o[0],dict):
  print(' '*depth,type(o),len(o))
  for v in o: show(v,depth+1)
 else:
  a=np.asarray(o); print(' '*depth,a.shape,a.dtype,a.ravel()[:12]);
  if a.size and np.issubdtype(a.dtype,np.number): print(' '*depth,'range',np.nanmin(a),np.nanmax(a))
show(p)
PY
```
- **Container Exit Code**: `127` (`323 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 8
-rw-r--r-- 1 root root 5205 Nov 25 15:48 public_surveys.pkl
```
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`15441 ms` LLM)
- **Rationale**: Use python3 after python was unavailable. Inspect the public data to determine geometry, units, known parameters, and whether SciPy is available.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle, numpy as np
try:
 import scipy
 print('scipy', scipy.__version__)
except ImportError:
 print('no scipy')
with open('public_surveys.pkl','rb') as f: p=pickle.load(f)
def show(o,depth=0):
 if isinstance(o,dict):
  for k,v in o.items():
   print(' '*depth,k); show(v,depth+1)
 elif isinstance(o,(list,tuple)) and o and isinstance(o[0],dict):
  print(' '*depth,type(o),len(o))
  for v in o: show(v,depth+1)
 else:
  a=np.asarray(o); print(' '*depth,a.shape,a.dtype,a.ravel()[:12]);
  if a.size and np.issubdtype(a.dtype,np.number): print(' '*depth,'range',np.nanmin(a),np.nanmax(a))
show(p)
PY
```
- **Container Exit Code**: `0` (`506 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
scipy 1.13.0
 <class 'list'> 2
  survey
   W_m
    () float64 [40.]
    range 40.0 40.0
   H_m
    () float64 [80.]
    range 80.0 80.0
   z_src_m
    (15,) float64 [ 6.         10.85714286 15.71428571 20.57142857 25.42857143 30.28571429
 35.14285714 40.         44.85714286 49.71428571 54.57142857 59.42857143]
    range 6.0 74.0
   z_rcv_m
    (16,) float64 [ 5.          9.66666667 14.33333333 19.         23.66666667 28.33333333
 33.         37.66666667 42.33333333 47.         51.66666667 56.33333333]
    range 5.0 75.0
   T_obs_s
    (15, 16) float64 [0.01491584 0.01729941 0.01816543 0.0148971  0.01612493 0.02175942
 0.01714121 0.01843184 0.02471534 0.02387659 0.01807355 0.02581797]
    range 0.01330233416627665 0.03317839066777648
   R_plume_m
    () float64 [5.5]
    range 5.5 5.5
  gt
   plume_xz_m
    (2,) float64 [16.5 34. ]
    range 16.5 34.0
   rcv_clock_bias_s
    (16,) float64 [-0.0019168   0.00045573  0.00110994 -0.00255604 -0.00179292  0.0032009
 -0.00228896 -0.0019091   0.00332965  0.00124041 -0.00037809  0.00053325]
    range -0.002556039023045965 0.003329653424440951
  survey
   W_m
    () float64 [40.]
    range 40.0 40.0
   H_m
    () float64 [80.]
    range 80.0 80.0
   z_src_m
    (15,) float64 [ 6.         10.85714286 15.71428571 20.57142857 25.42857143 30.28571429
 35.14285714 40.         44.85714286 49.71428571 54.57142857 59.42857143]
    range 6.0 74.0
   z_rcv_m
    (16,) float64 [ 5.          9.66666667 14.33333333 19.         23.66666667 28.33333333
 33.         37.66666667 42.33333333 47.         51.66666667 56.33333333]
    range 5.0 75.0
   T_obs_s
    (15, 16) float64 [0.01425288 0.0187468  0.01403325 0.0142902  0.01580415 0.01554096
 0.01899251 0.0161894  0.02200229 0.02269381 0.01829848 0.02293383]
    range 0.01336071363590472 0.028705721115872333
   R_plume_m
    () float64 [5.5]
    range 5.5 5.5
  gt
   plume_xz_m
    (2,) float64 [26.  51.5]
    range 26.0 51.5
   rcv_clock_bias_s
    (16,) float64 [-0.0009812   0.00347274 -0.00137483 -0.00143901 -0.00034719 -0.00111101
  0.00170437 -0.00184997  0.0031499   0.00290556 -0.00256839  0.0008789 ]
    range -0.002568385497994811 0.0034727403303111385
```

### Turn 3 — `execute_in_remote_docker` (`21375 ms` LLM)
- **Rationale**: Use the public labels only to diagnose the noise scale and whether cycle skips can be corrected as fixed offsets. If the large residuals are not concentrated near ±6 ms, the solver should reject them instead of correcting them.
- **Alternatives Weighed**: `Treat background slowness and anisotropy as known` (rejected: Neither value is present in the public survey inputs, so both must be estimated.)

```bash
cd /holdout_task && cat >/tmp/probe_tomo.py <<'PY'
import numpy as np,pickle
from scipy.special import erf
from scipy.optimize import least_squares
P=pickle.load(open('/holdout_task/public_surveys.pkl','rb'))
for p in P:
 s=p['survey']; gt=p['gt']; W=s['W_m']; R=s['R_plume_m']; a=s['z_src_m'][:,None]; b=s['z_rcv_m'][None,:]; dz=b-a; L=np.hypot(W,dz); sn=dz/L; cs=W/L; x,z=gt['plume_xz_m']; u=x*cs+(z-a)*sn; v=(z-a)*cs-x*sn
 g=R*np.sqrt(np.pi/2)*np.exp(-v*v/(2*R*R))*(erf((L-u)/(np.sqrt(2)*R))+erf(u/(np.sqrt(2)*R)))
 X=np.stack([L,-L*sn**2,g],axis=-1).reshape(-1,3)
 y=(s['T_obs_s']-gt['rcv_clock_bias_s']).ravel()*1000
 fit=least_squares(lambda q:X@q-y,[.42,.02,.2],loss='cauchy',f_scale=.25)
 r=y-X@fit.x
 print('params ms/m',fit.x,'epsilon',fit.x[1]/fit.x[0]); print('residual quantiles ms',np.round(np.quantile(r,[0,.01,.025,.05,.1,.25,.5,.75,.9,.95,.975,.99,1]),4)); ind=abs(r)>2; print('large residuals',np.round(np.sort(r[ind]),4),'corrected std',np.std(r-np.sign(r)*ind*6)); print('clock sum',sum(gt['rcv_clock_bias_s']))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe_tomo.py
```
- **Container Exit Code**: `0` (`481 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
params ms/m [0.42012809 0.05911299 0.06487086] epsilon 0.14070230154244823
residual quantiles ms [-5.5288e+00 -5.5016e+00 -7.1200e-02 -5.5700e-02 -4.8600e-02 -2.3900e-02
  3.6000e-03  2.2000e-02  4.2100e-02  6.5100e-02  2.2120e-01  5.5081e+00
  5.5613e+00]
large residuals [-5.5288 -5.5124 -5.5031 -5.4991  5.462   5.4649  5.4988  5.514   5.5197
  5.5613] corrected std 0.10572420199238368
clock sum 4.336808689942018e-19
params ms/m [0.38005783 0.06873795 0.07262196] epsilon 0.18086181779983965
residual quantiles ms [-5.5156e+00 -3.3755e+00 -7.5600e-02 -6.2100e-02 -5.0100e-02 -1.9000e-02
  1.5000e-03  2.3800e-02  4.4100e-02  6.6500e-02  5.4738e+00  5.5200e+00
  5.5252e+00]
large residuals [-5.5156 -5.5149 -5.4802  5.4732  5.4967  5.4994  5.5181  5.5213  5.5248
  5.5252] corrected std 0.10549189060124918
clock sum -1.5178830414797062e-18
```

### Turn 4 — `execute_in_remote_docker` (`67451 ms` LLM)
- **Rationale**: Implement the finite-ray Gaussian model, profile out background and clocks, and verify both public cases against the requested accuracy, clock constraints, and runtime.
- **Alternatives Weighed**: `Fit all quantities jointly from one initial plume position` (rejected: A localized plume can produce competing minima; a coarse profiled search provides starting points without repeatedly optimizing clocks.); `Subtract the nominal 6 ms from every identified skip` (rejected: The public skips cluster near 5.5 ms, so the solver estimates a common jump only when the residuals support one.)

```bash
cat > /holdout_task/tomography_solver.py <<'PY'
"""Joint inversion of straight-ray traveltimes for a Gaussian plume and receiver statics."""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


class _SurveyModel:
    def __init__(self, survey):
        self.w = float(survey['W_m'])
        self.h = float(survey['H_m'])
        self.r = float(survey['R_plume_m'])
        source = np.asarray(survey['z_src_m'], dtype=float)
        receiver = np.asarray(survey['z_rcv_m'], dtype=float)
        dz = (receiver[None, :] - source[:, None]).ravel()
        self.nrcv = nr = receiver.size
        self.a = np.repeat(source, nr)
        self.length = np.hypot(self.w, dz)
        self.cos = self.w / self.length
        self.sin = dz / self.length
        # Clock columns span the sum-zero subspace. The two background
        # coefficients are s0 and the product s0 * epsilon, in ms/m.
        clock = np.vstack((np.eye(nr - 1), -np.ones((1, nr - 1))))
        self.base = np.column_stack((self.length, -self.length * self.sin**2,
                                     np.tile(clock, (source.size, 1))))
        self.y = np.asarray(survey['T_obs_s'], dtype=float).ravel() * 1000.0

    def plume(self, x, z, derivatives=False):
        """Exact Gaussian integral on each finite source-receiver segment."""
        u = x * self.cos + (z - self.a) * self.sin
        v = (z - self.a) * self.cos - x * self.sin
        t0 = u / (np.sqrt(2) * self.r)
        t1 = (self.length - u) / (np.sqrt(2) * self.r)
        e = np.exp(-0.5 * (v / self.r)**2)
        g = self.r * np.sqrt(np.pi / 2) * e * (erf(t0) + erf(t1))
        if not derivatives:
            return g
        gu = e * (np.exp(-t0*t0) - np.exp(-t1*t1))
        gv = -v * g / self.r**2
        return g, gu * self.cos - gv * self.sin, gu * self.sin + gv * self.cos


class _Profile:
    """Eliminate background and clocks from each weighted plume fit."""
    def __init__(self, model, y, weight):
        self.model = model
        self.w = np.sqrt(weight)
        self.design = self.w[:, None] * model.base
        self.q = np.linalg.qr(self.design, mode='reduced')[0]
        self.y = y
        self.target = self.project(y * self.w)

    def project(self, v):
        # Explicit contractions are efficient for the small nuisance subspace.
        coords = np.einsum('ni,...n->...i', self.q, v)
        return v - np.einsum('ni,...i->...n', self.q, coords)

    def refine(self, start):
        m = self.model

        def residual(p):
            return p[2] * self.project(self.w * m.plume(*p[:2])) - self.target

        def jacobian(p):
            g, gx, gz = m.plume(*p[:2], derivatives=True)
            return self.project(self.w * np.array([p[2]*gx, p[2]*gz, g])).T

        return least_squares(residual, start, jac=jacobian,
                             bounds=([0, 0, 0], [m.w, m.h, np.inf]),
                             x_scale=[m.r, m.r, 0.1], max_nfev=80)

    def locate(self):
        m = self.model
        step = m.r / 2
        xs = np.linspace(0, m.w, int(np.ceil(m.w / step)) + 1)
        zs = np.linspace(0, m.h, int(np.ceil(m.h / step)) + 1)
        x, z = (a.ravel() for a in np.meshgrid(xs, zs))
        candidates = []
        for i in range(0, x.size, 64):
            j = slice(i, i + 64)
            template = self.project(self.w * m.plume(x[j, None], z[j, None]))
            cross = np.maximum(0, np.einsum('ij,j->i', template, self.target))
            norm2 = np.einsum('ij,ij->i', template, template)
            candidates.extend(zip(cross**2 / norm2, x[j], z[j], cross / norm2))
        fits = []
        for _, x0, z0, amp in sorted(candidates, reverse=True):
            if all(np.hypot(x0 - x1, z0 - z1) > m.r for x1, z1, _ in fits):
                fits.append((x0, z0, self.refine([x0, z0, amp])))
                if len(fits) == 4:
                    break
        return min((fit for _, _, fit in fits), key=lambda fit: fit.cost).x

    def linear_terms(self, p):
        return np.linalg.lstsq(self.design,
                               self.w * (self.y - p[2] * self.model.plume(*p[:2])),
                               rcond=None)[0]


def _clock_projection(clock):
    """Project onto the box with zero mean, in milliseconds."""
    if np.max(np.abs(clock)) <= 4.0:
        return clock
    lo, hi = min(clock) - 4.0, max(clock) + 4.0
    for _ in range(48):
        mid = (lo + hi) / 2
        if np.clip(clock - mid, -4.0, 4.0).sum() > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(clock - (hi + lo) / 2, -4.0, 4.0)


def invert_plume_and_clocks(survey: dict) -> dict:
    """Return the plume center in meters and the receiver biases in seconds."""
    m = _SurveyModel(survey)
    design, y = m.base, m.y
    # A robust background-only fit identifies the gross skips without letting
    # one such pick move an entire receiver's clock by a substantial fraction.
    weights = np.ones_like(y)
    for _ in range(12):
        w = np.sqrt(weights)
        beta = np.linalg.lstsq(w[:, None] * design, w * y, rcond=None)[0]
        residual = y - np.einsum('ij,j->i', design, beta)
        weights = 1 / np.maximum(1, np.abs(residual) / 0.35)

    prof = _Profile(m, y, np.abs(residual) < 2.75)
    p = prof.locate()
    for _ in range(3):
        beta = prof.linear_terms(p)
        residual = y - np.einsum('ij,j->i', design, beta) - p[2] * m.plume(*p[:2])
        inlier = np.abs(residual) < 2.75
        if np.array_equal(inlier, prof.w):
            break
        prof = _Profile(m, y, inlier)
        p = prof.refine(p).x

    # A coherent signed time jump can be fitted as a nuisance parameter,
    # recovering the useful information in skipped picks. If the outliers do
    # not share a common jump magnitude, keep the inlier-only fit instead.
    beta = prof.linear_terms(p)
    residual = y - np.einsum('ij,j->i', design, beta) - p[2] * m.plume(*p[:2])
    good = prof.w.astype(bool)
    skipped = ~good
    if np.count_nonzero(skipped) >= 4:
        jumps = np.abs(residual[skipped])
        offset = np.median(jumps)
        noise = max(0.015, 1.4826 * np.median(np.abs(residual[good])))
        coherent = np.abs(jumps - offset) < 3.5 * noise
        if np.count_nonzero(coherent) >= 0.75 * jumps.size:
            sign = np.zeros_like(y)
            indices = np.flatnonzero(skipped)[coherent]
            sign[indices] = np.sign(residual[indices])
            m.base = np.column_stack((design, sign))
            prof = _Profile(m, y, good | (sign != 0))
            p = prof.refine(p).x
            beta = prof.linear_terms(p)

    c = beta[2:1 + m.nrcv]
    clock = _clock_projection(np.r_[c, -c.sum()])
    return {'plume_xz_m': p[:2].tolist(), 'rcv_clock_bias_s': (clock / 1000).tolist()}
PY
cd /holdout_task && python3 - <<'PY'
import pickle, time, numpy as np
from tomography_solver import invert_plume_and_clocks
with open('public_surveys.pkl', 'rb') as f:
    for case in pickle.load(f):
        t = time.perf_counter()
        result = invert_plume_and_clocks(case['survey'])
        elapsed = time.perf_counter() - t
        gt = case['gt']
        center = np.linalg.norm(np.asarray(result['plume_xz_m']) - gt['plume_xz_m'])
        clock = np.asarray(result['rcv_clock_bias_s'])
        rmse = np.sqrt(np.mean((clock - gt['rcv_clock_bias_s'])**2))
        print(result['plume_xz_m'], 'center error', center, 'clock RMSE', rmse, 'seconds', elapsed)
        assert center <= 1.8 and rmse <= .00025 and elapsed < 15
        assert abs(clock.sum()) < 1e-14 and np.max(np.abs(clock)) <= .004
PY
```
- **Container Exit Code**: `0` (`830 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tomography_solver.py']`
- **Stdout (tail)**:
```text
[16.419119806808208, 34.03874342610281] center error 0.08968087152188942 clock RMSE 4.859368156702481e-06 seconds 0.015239862026646733
[25.89156160642452, 51.444623763272105] center error 0.12175965175448894 clock RMSE 1.2160443962346202e-05 seconds 0.014147764071822166
```

### Turn 5 — `submit_completed_task` (`13601 ms` LLM)
- **Rationale**: The required file was created and its callable interface, accuracy, clock constraints, and runtime were verified on both public surveys.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/tomography_solver.py` with `invert_plume_and_clocks(survey)`. Both public surveys passed: plume-center errors were 0.09 m and 0.12 m, clock-bias RMSEs were 4.9 µs and 12.2 µs, and each inversion took about 0.015 s. Both clock estimates satisfied the zero-sum and ±4 ms constraints. Held-out performance was not measured.`

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
