# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Bare_ReAct_LiveGPT` | `ho04__magnetometer_hard_soft_iron_calibration`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-OOD-Holdout-12` |
| **Domain** | `Navigation & Sensor Fusion / Ellipsoid Calibration` |
| **Wave / Wormhole Family** | `wave_1` / `fam_robust_control_and_safety` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Bare_ReAct_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `7 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `139,209` (`133,994` / `12,544` / `5,215`) |
| **Total Episode Wall Time (s)** | `185.39s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`14078 ms` LLM)
- **Rationale**: Inspect the public schema and available numerical libraries; the sample layout determines the implementation and available validation.

```bash
cd /holdout_task && python - <<'PY'
import pickle,numpy as np, importlib.util
with open('public_mag.pkl','rb') as f: data=pickle.load(f)
def show(d,p='',depth=0):
 if isinstance(d,dict) and depth<3:
  for k,v in d.items(): show(v,p+'/'+str(k),depth+1)
 elif isinstance(d,(list,tuple)) and d and isinstance(d[0],dict) and depth<3:
  print(p,type(d).__name__,len(d)); show(d[0],p+'/0',depth+1)
 elif hasattr(d,'shape'):
  print(p,d.shape,d.dtype, np.asarray(d).ravel()[:8])
 else: print(p,type(d).__name__,str(d)[:220])
show(data)
print('scipy',bool(importlib.util.find_spec('scipy')))
PY
```
- **Container Exit Code**: `127` (`332 ms`, timed_out=`False`)
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`13892 ms` LLM)
- **Rationale**: Use python3 after the missing python executable; inspect the fixture for input and reference fields and check whether SciPy is available for fitting.

```bash
cd /holdout_task && python3 - <<'PY'
import importlib.util, pickle
for module in ('numpy', 'scipy'):
    print(module, importlib.util.find_spec(module))
with open('public_mag.pkl','rb') as f:
    data=pickle.load(f)
def show(d,p='root'):
    if isinstance(d,dict):
        for k,v in d.items(): show(v,p+'/'+str(k))
    elif isinstance(d,(list,tuple)) and d and isinstance(d[0],dict):
        print(p,type(d).__name__,len(d)); show(d[0],p+'/0')
    elif hasattr(d,'shape'):
        print(p,d.shape,d.dtype,d.ravel()[:8])
    else: print(p,type(d).__name__,str(d)[:220])
show(data)
PY
```
- **Container Exit Code**: `0` (`538 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
numpy ModuleSpec(name='numpy', loader=<_frozen_importlib_external.SourceFileLoader object at 0x729f79231070>, origin='/usr/local/lib/python3.12/dist-packages/numpy/__init__.py', submodule_search_locations=['/usr/local/lib/python3.12/dist-packages/numpy'])
scipy ModuleSpec(name='scipy', loader=<_frozen_importlib_external.SourceFileLoader object at 0x729f79231130>, origin='/usr/local/lib/python3.12/dist-packages/scipy/__init__.py', submodule_search_locations=['/usr/local/lib/python3.12/dist-packages/scipy'])
root/session/calib_m_raw (420, 3) float64 [ 32.72896585  -3.85739768  44.55170701  71.03954969 -42.67050232
  41.98786812  29.22284262  -4.67667589]
root/session/calib_roll_pitch_rad (420, 2) float64 [ 0.46064931  0.13020456 -0.43913513 -0.4172756   0.43406992  0.1992708
  0.4146053  -0.2951089 ]
root/session/test_m_raw (120, 3) float64 [ -4.76198481 -52.64081502  52.48394928  -6.06556345 -50.69433295
  52.47467309  -7.14328655 -48.38551289]
root/session/test_roll_pitch_rad (120, 2) float64 [-0.3821346   0.06953427 -0.35959263  0.10418026 -0.32861013  0.13773825
 -0.28991433  0.16985775]
root/gt/S_soft (3, 3) float64 [ 1.17564433  0.06974161 -0.03985235  0.06974161  0.91660405  0.04981544
 -0.03985235  0.04981544]
root/gt/b_hard (3,) float64 [ 28.5 -41.   16.2]
root/gt/gt_heading_deg (120,) float64 [-174.27042205 -171.34150739 -168.41259274 -165.48367808 -162.55476342
 -159.62584877 -156.69693411 -153.76801945]
```

### Turn 3 — `execute_in_remote_docker` (`22193 ms` LLM)
- **Rationale**: Check the leveling convention against the public reference and measure the residual separation between nominal noise and spikes. A convention mismatch would change the rotation; overlapping residuals would call for soft rather than hard rejection.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
with open('public_mag.pkl','rb') as f: p=pickle.load(f)
s,t=p['session'],p['gt']
for stage in ('calib','test'):
 m=(s[stage+'_m_raw']-t['b_hard'])@np.linalg.inv(t['S_soft'])
 r,q=s[stage+'_roll_pitch_rad'].T
 x=np.cos(q)*m[:,0]+np.sin(q)*(np.sin(r)*m[:,1]+np.cos(r)*m[:,2]); y=np.cos(r)*m[:,1]-np.sin(r)*m[:,2]; z=-np.sin(q)*m[:,0]+np.cos(q)*(np.sin(r)*m[:,1]+np.cos(r)*m[:,2]); h=np.hypot(x,y)
 print(stage,'angle ranges',np.ptp(r),np.ptp(q),'B medians',np.median(h),np.median(z)); print('radial,down absolute residual percentiles',np.percentile(np.abs(np.array([h,z])-np.median([h,z],axis=1)[:,None]),[50,80,90,94,95,96,97,99,100],axis=1))
 if stage=='test':
  d=(np.rad2deg(np.arctan2(-y,x))-t['gt_heading_deg']+180)%360-180
  print('ground truth calibration heading RMSE',np.sqrt(np.mean(d*d)), 'max',np.max(np.abs(d)))
PY
```
- **Container Exit Code**: `0` (`556 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calib angle ranges 1.4909761271821975 1.2938466932496764 B medians 23.491583735052664 41.99447680512864
radial,down absolute residual percentiles [[ 0.05801411  0.06263549]
 [ 0.11475261  0.12428819]
 [ 0.15390627  0.16015834]
 [ 0.19338095  0.19968985]
 [ 0.20160562  0.21984963]
 [ 0.26404502  0.26867822]
 [ 3.4321769   6.55160811]
 [15.73966377 20.98415916]
 [26.46308442 37.71522222]]
test angle ranges 0.7989568253190303 0.699932387266293 B medians 23.490659399027745 41.99837957574918
radial,down absolute residual percentiles [[0.03947432 0.04356784]
 [0.07805903 0.07032867]
 [0.09880145 0.1022795 ]
 [0.11734207 0.11357331]
 [0.11852109 0.12985009]
 [0.12824826 0.13153055]
 [0.13065372 0.13597351]
 [0.14445651 0.14224735]
 [0.16156264 0.16526525]]
ground truth calibration heading RMSE 0.1310671018690439 max 0.3629793030087285
```

### Turn 4 — `execute_in_remote_docker` (`66384 ms` LLM)
- **Rationale**: Implement a determinant-one SPD fit that rejects spikes using the tilt-level cylinder constraints. Public-fixture errors and matrix invariants will reveal if the fit or heading convention needs correction.
- **Alternatives Weighed**: `Fit an ellipsoid using raw norm alone` (rejected: It discards the known roll/pitch constraints and is less informative when tilt coverage is uneven.)

```bash
cat > /holdout_task/mag_calibrator.py <<'PY'
"""Robust hard- and soft-iron calibration from known roll and pitch."""

import numpy as np
from scipy.optimize import least_squares


def _level_rotation(roll_pitch):
    """Body to yaw-rotated NED: Ry(pitch) @ Rx(roll)."""
    roll, pitch = np.asarray(roll_pitch, dtype=float).T
    sr, cr = np.sin(roll), np.cos(roll)
    st, ct = np.sin(pitch), np.cos(pitch)
    return np.stack(((ct, st * sr, st * cr),
                     (np.zeros_like(sr), cr, -sr),
                     (-st, ct * sr, ct * cr)), axis=-1).transpose(1, 0, 2)


def _vertical_start(raw, down):
    """Fit n.T C x - n.T d = D, where d=C b, with trace(C)=3."""
    x, y, z = raw.T
    u, v, w = down.T
    design = np.column_stack((u*x - w*z, v*y - w*z,
                              u*y + v*x, u*z + w*x, v*z + w*y,
                              -u, -v, -w, -np.ones(len(raw))))
    target = -3*w*z
    rng = np.random.default_rng(0)
    best = np.inf
    for i in range(48 if len(raw) > 16 else 1):
        rows = slice(None) if i == 0 else rng.choice(len(raw), 16, replace=False)
        p = np.linalg.lstsq(design[rows], target[rows], rcond=None)[0]
        score = np.median(np.abs(design @ p - target))
        if score < best:
            best, estimate = score, p
    for _ in range(3):
        error = np.abs(design @ estimate - target)
        keep = error <= max(np.quantile(error, 0.85), 4.5*np.median(error))
        estimate = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
    a, b, xy, xz, yz = estimate[:5]
    inverse = np.array([[a, xy, xz], [xy, b, yz], [xz, yz, 3-a-b]])
    if np.linalg.eigvalsh(inverse)[0] > 0:
        center = np.linalg.solve(inverse, estimate[5:8])
        inverse /= np.cbrt(np.linalg.det(inverse))
    else:
        inverse = np.eye(3)
        center = np.linalg.lstsq(np.column_stack((down, np.ones(len(raw)))),
                                 np.sum(down * raw, axis=1), rcond=None)[0][:3]
    return inverse, center


def _inverse_matrix(params):
    a, b, xy, xz, yz = params[:5]
    lower = np.array([[np.exp(a), 0., 0.], [xy, np.exp(b), 0.],
                      [xz, yz, np.exp(-a-b)]])
    return lower @ lower.T


def _fit(raw, rotation):
    # Center and scale raw vectors for conditioning; the correction matrix
    # itself stays dimensionless and has determinant one throughout the fit.
    origin = np.median(raw, axis=0)
    scale = np.median(np.linalg.norm(raw - origin, axis=1))
    raw = (raw - origin) / scale
    inverse, center = _vertical_start(raw, rotation[:, 2, :])
    lower = np.linalg.cholesky(inverse)
    level = np.einsum('nij,nj->ni', rotation, (raw - center) @ inverse)
    start = np.r_[np.log(lower[0, 0]), np.log(lower[1, 1]),
                  lower[1, 0], lower[2, 0], lower[2, 1], center,
                  np.median(np.hypot(level[:, 0], level[:, 1])), np.median(level[:, 2])]

    def residual(params, samples, leveling):
        inverse = _inverse_matrix(params)
        transform = leveling @ inverse
        level = np.einsum('nij,nj->ni', transform, samples - params[5:8])
        h = np.hypot(level[:, 0], level[:, 1])
        # Map cylinder constraint errors to first-order raw-space distances.
        # This accounts for distortion of the measurement noise by the inverse.
        hdir = level[:, :2] / np.maximum(h[:, None], 1e-15)
        gh = np.einsum('ni,nij->nj', hdir, transform[:, :2, :])
        gz = transform[:, 2, :]
        gz_norm = np.linalg.norm(gz, axis=1)
        gz_unit = gz / gz_norm[:, None]
        projection = np.sum(gh * gz_unit, axis=1)
        gh_norm = np.linalg.norm(gh - projection[:, None] * gz_unit, axis=1)
        rz = (level[:, 2] - params[9]) / gz_norm
        rh = (h - params[8] - projection * rz) / np.maximum(gh_norm, 1e-15)
        return np.column_stack((rh, rz))

    def robust_residual(params, width):
        r = residual(params, raw, rotation)
        t = np.sum((r / width)**2, axis=1)
        # One Cauchy loss per sample, not one per coordinate.
        weight = np.sqrt(np.log1p(t) / np.maximum(t, 1e-30))
        return (r * np.where(t < 1e-30, 1., weight)[:, None]).ravel()

    width = max(np.median(np.linalg.norm(residual(start, raw, rotation), axis=1)), 1e-8)
    params = least_squares(robust_residual, start, args=(2*width,), max_nfev=150).x
    width = max(np.median(np.linalg.norm(residual(params, raw, rotation), axis=1)), 1e-8)
    params = least_squares(robust_residual, params, args=(2*width,), max_nfev=100).x
    # Use ordinary least squares for the clean core, after removing spikes.
    distance = np.linalg.norm(residual(params, raw, rotation), axis=1)
    keep = distance <= max(3.5*np.median(distance), 1e-8)
    params = least_squares(lambda p: residual(p, raw[keep], rotation[keep]).ravel(),
                           params, max_nfev=100).x
    return _inverse_matrix(params), origin + scale*params[5:8]


def calibrate_and_estimate_heading(session: dict) -> dict:
    """Return estimated soft-iron matrix, hard-iron bias (uT), and headings (deg)."""
    inverse, bias = _fit(np.asarray(session['calib_m_raw'], dtype=float),
                         _level_rotation(session['calib_roll_pitch_rad']))
    field_body = (np.asarray(session['test_m_raw'], dtype=float) - bias) @ inverse
    field_level = np.einsum('nij,nj->ni', _level_rotation(session['test_roll_pitch_rad']),
                           field_body)
    heading = np.rad2deg(np.arctan2(-field_level[:, 1], field_level[:, 0]))
    return {'S_soft_hat': np.linalg.inv(inverse), 'b_hard_hat': bias,
            'heading_deg': np.where(heading <= -180., 180., heading)}
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle
import numpy as np
from mag_calibrator import calibrate_and_estimate_heading
with open('public_mag.pkl','rb') as f: fixture = pickle.load(f)
a = calibrate_and_estimate_heading(fixture['session']); g = fixture['gt']
d = (a['heading_deg'] - g['gt_heading_deg'] + 180) % 360 - 180
rmse, bias = np.sqrt(np.mean(d*d)), np.linalg.norm(a['b_hard_hat']-g['b_hard'])
s = a['S_soft_hat']
print('heading RMSE', rmse, 'bias error', bias, 'matrix error', np.linalg.norm(s-g['S_soft']), 'determinant', np.linalg.det(s))
assert rmse <= .45 and bias <= .35
assert np.allclose(s,s.T) and np.linalg.eigvalsh(s)[0]>0 and np.isclose(np.linalg.det(s),1)
assert np.all((a['heading_deg'] > -180) & (a['heading_deg'] <= 180))
PY
```
- **Container Exit Code**: `1` (`704 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/mag_calibrator.py']`
- **Stdout (tail)**:
```text
heading RMSE 13.08208964257309 bias error 48.986406421142306 matrix error 0.8301722045061501 determinant 1.0
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
AssertionError
```

### Turn 5 — `execute_in_remote_docker` (`20287 ms` LLM)
- **Rationale**: Fix the transposed leveling matrix. Check it directly against Ry @ Rx before rerunning the public benchmark; if errors remain after the rotation passes, investigate the fit rather than the convention.

```bash
cd /holdout_task && sed -i 's/axis=-1).transpose(1, 0, 2)/axis=-1).transpose(1, 2, 0)/' mag_calibrator.py && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle
import numpy as np
from mag_calibrator import calibrate_and_estimate_heading, _level_rotation
r,p=.3,-.4
rx=np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
ry=np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
np.testing.assert_allclose(_level_rotation([[r,p]])[0],ry@rx)
with open('public_mag.pkl','rb') as f: fixture=pickle.load(f)
a=calibrate_and_estimate_heading(fixture['session']); g=fixture['gt']
d=(a['heading_deg']-g['gt_heading_deg']+180)%360-180
rmse,bias=np.sqrt(np.mean(d*d)),np.linalg.norm(a['b_hard_hat']-g['b_hard'])
s=a['S_soft_hat']
print('heading RMSE',rmse,'bias error',bias,'matrix error',np.linalg.norm(s-g['S_soft']),'determinant',np.linalg.det(s))
assert rmse<=.45 and bias<=.35
assert np.allclose(s,s.T) and np.linalg.eigvalsh(s)[0]>0 and np.isclose(np.linalg.det(s),1)
assert np.all((a['heading_deg']>-180)&(a['heading_deg']<=180))
PY
```
- **Container Exit Code**: `0` (`538 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/mag_calibrator.py']`
- **Stdout (tail)**:
```text
heading RMSE 0.13290048850490904 bias error 0.016807503528075703 matrix error 0.0008198476635844103 determinant 0.9999999999999997
```

### Turn 6 — `execute_in_remote_docker` (`26022 ms` LLM)
- **Rationale**: The public test passed. Test independently generated fields, biases, and distortion matrices to detect dependence on the public fixture’s geometry; a failure would identify a regime where the estimator needs adjustment.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import numpy as np
from mag_calibrator import calibrate_and_estimate_heading, _level_rotation
rng = np.random.default_rng(1234)
for i in range(16):
    q, _ = np.linalg.qr(rng.normal(size=(3,3)))
    e = rng.uniform(-.35,.35,3); e -= e.mean()
    soft = (q*np.exp(e))@q.T
    hard = rng.uniform(-65,65,3)
    hn, bd = rng.uniform(20,34), rng.uniform(-47,47)
    def observations(n, calibration):
        rp = rng.uniform([-0.65,-0.55],[0.65,0.55],(n,2))
        yaw = rng.uniform(-np.pi,np.pi,n)
        level = np.column_stack([hn*np.cos(yaw),-hn*np.sin(yaw),np.full(n,bd)])
        body = np.einsum('nji,nj->ni',_level_rotation(rp),level)
        raw = body@soft + hard + rng.normal(scale=.10 if calibration else .06,size=(n,3))
        if calibration:
            ids = rng.choice(n,int(n*.04),replace=False)
            raw[ids] += rng.normal(scale=18,size=(len(ids),3))
        return raw,rp,np.degrees(yaw)
    c,cp,_ = observations(350,True); t,tp,y = observations(100,False)
    ans = calibrate_and_estimate_heading(dict(calib_m_raw=c,calib_roll_pitch_rad=cp,test_m_raw=t,test_roll_pitch_rad=tp))
    d = (ans['heading_deg']-y+180)%360-180
    rmse = np.sqrt(np.mean(d*d)); bias = np.linalg.norm(ans['b_hard_hat']-hard)
    print(f'{i:02d}: {rmse:.4f} deg, {bias:.4f} uT')
    assert rmse <= .45 and bias <= .35
PY
```
- **Container Exit Code**: `0` (`832 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
00: 0.1136 deg, 0.0144 uT
01: 0.1712 deg, 0.0205 uT
02: 0.1117 deg, 0.0236 uT
03: 0.1147 deg, 0.0347 uT
04: 0.1321 deg, 0.0132 uT
05: 0.1133 deg, 0.0167 uT
06: 0.1328 deg, 0.0326 uT
07: 0.1413 deg, 0.0175 uT
08: 0.1239 deg, 0.0264 uT
09: 0.1559 deg, 0.0163 uT
10: 0.1265 deg, 0.0376 uT
11: 0.1251 deg, 0.0285 uT
12: 0.1489 deg, 0.0427 uT
13: 0.1490 deg, 0.0182 uT
14: 0.1252 deg, 0.0194 uT
15: 0.1226 deg, 0.0304 uT
```

### Turn 7 — `submit_completed_task` (`10880 ms` LLM)
- **Rationale**: The deliverable is saved. Public verification passed for both error targets, the heading range, and the symmetric positive-definite determinant-one matrix; synthetic trials passed across varying fields and distortions.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/mag_calibrator.py`. On the public fixture, circular heading RMSE was 0.133° and hard-iron bias error was 0.017 µT. Both targets were also met in all 16 synthetic trials. Held-out performance was not measured.`

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
