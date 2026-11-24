# Forensic Post-Mortem Trajectory: `gpt-6-luna` | `Bare_ReAct_LiveGPT` | `ho04__magnetometer_hard_soft_iron_calibration`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-OOD-Holdout-12` |
| **Domain** | `Navigation & Sensor Fusion / Ellipsoid Calibration` |
| **Wave / Wormhole Family** | `wave_1` / `fam_robust_control_and_safety` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6-luna` (`gpt-6-luna`) |
| **Arm** | `Bare_ReAct_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `7 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `189,301` (`183,498` / `67,072` / `5,803`) |
| **Total Episode Wall Time (s)** | `229.44s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`11478 ms` LLM)
- **Rationale**: Inspect the provided calibration dataset structure before implementing the estimator.

```bash
python - <<'PY'
import pickle
p='/holdout_task/public_mag.pkl'
x=pickle.load(open(p,'rb'))
print(type(x)); print(x.keys() if isinstance(x,dict) else len(x))
if isinstance(x,dict):
 for k,v in x.items():
  print(k,type(v),getattr(v,'shape',None),repr(v)[:400])
PY
```
- **Container Exit Code**: `127` (`373 ms`, timed_out=`False`)
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`12070 ms` LLM)
- **Rationale**: The previous inspection failed only because `python` is unavailable; use `python3` to inspect the supplied calibration data before implementing the estimator.

```bash
python3 - <<'PY'
import pickle
p='/holdout_task/public_mag.pkl'
x=pickle.load(open(p,'rb'))
print(type(x)); print(x.keys() if isinstance(x,dict) else len(x))
if isinstance(x,dict):
 for k,v in x.items(): print(k,type(v),getattr(v,'shape',None),repr(v)[:400])
PY
```
- **Container Exit Code**: `0` (`567 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
<class 'dict'>
dict_keys(['session', 'gt'])
session <class 'dict'> None {'calib_m_raw': array([[ 32.72896585,  -3.85739768,  44.55170701],
       [ 71.03954969, -42.67050232,  41.98786812],
       [ 29.22284262,  -4.67667589,  45.65289493],
       ...,
       [ 12.45981005, -44.41781384,  59.70725718],
       [ 44.70717721, -73.77116211,  35.89393352],
       [ 22.65179743,  -9.73398028,  50.59111807]]), 'calib_roll_pitch_rad': array([[ 4.60649308e-01,  1.30204556e-01
gt <class 'dict'> None {'S_soft': array([[ 1.17564433,  0.06974161, -0.03985235],
       [ 0.06974161,  0.91660405,  0.04981544],
       [-0.03985235,  0.04981544,  0.93653023]]), 'b_hard': array([ 28.5, -41. ,  16.2]), 'gt_heading_deg': array([-174.27042205, -171.34150739, -168.41259274, -165.48367808,
       -162.55476342, -159.62584877, -156.69693411, -153.76801945,
       -150.8391048 , -147.91019014, -144.98127549,
```

### Turn 3 — `execute_in_remote_docker` (`9674 ms` LLM)
- **Rationale**: Inspect only array dimensions and ground-truth metadata to determine the calibration inputs and verify expected output interface.

```bash
python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('/holdout_task/public_mag.pkl','rb')); s=x['session'];g=x['gt']
for k,v in s.items(): print(k,np.asarray(v).shape)
for k,v in g.items(): print('gt',k,np.asarray(v).shape, repr(v)[:120])
print('keys',s.keys())
PY
```
- **Container Exit Code**: `0` (`569 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
calib_m_raw (420, 3)
calib_roll_pitch_rad (420, 2)
test_m_raw (120, 3)
test_roll_pitch_rad (120, 2)
gt S_soft (3, 3) array([[ 1.17564433,  0.06974161, -0.03985235],
       [ 0.06974161,  0.91660405,  0.04981544],
       [-0.03985235,  0.
gt b_hard (3,) array([ 28.5, -41. ,  16.2])
gt gt_heading_deg (120,) array([-174.27042205, -171.34150739, -168.41259274, -165.48367808,
       -162.55476342, -159.62584877, -156.69693411, -
keys dict_keys(['calib_m_raw', 'calib_roll_pitch_rad', 'test_m_raw', 'test_roll_pitch_rad'])
```

### Turn 4 — `execute_in_remote_docker` (`75291 ms` LLM)
- **Rationale**: Check available numerical libraries before selecting a robust calibration implementation.

```bash
python3 - <<'PY'
import numpy as np, scipy
print(np.__version__,scipy.__version__)
PY
```
- **Container Exit Code**: `0` (`513 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1.26.4 1.13.0
```

### Turn 5 — `execute_in_remote_docker` (`51748 ms` LLM)
- **Rationale**: Run the written module against the provided public sample to verify syntax and calibration accuracy before submitting.

```bash
ls -l /holdout_task/mag_calibrator.py && python3 -m py_compile /holdout_task/mag_calibrator.py && python3 - <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task'); from mag_calibrator import calibrate_and_estimate_heading
x=pickle.load(open('/holdout_task/public_mag.pkl','rb')); o=calibrate_and_estimate_heading(x['session']); g=x['gt']
e=(o['heading_deg']-g['gt_heading_deg']+180)%360-180
print('bias',o['b_hard'], 'b_err',np.linalg.norm(o['b_hard']-g['b_hard']))
print('S_err',np.linalg.norm(o['S_soft']-g['S_soft']))
print('heading rmse',np.sqrt(np.mean(e**2)), 'range',e.min(),e.max())
PY
```
- **Container Exit Code**: `2` (`306 ms`, timed_out=`False`)
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task/mag_calibrator.py': No such file or directory
```

### Turn 6 — `execute_in_remote_docker` (`46716 ms` LLM)
- **Rationale**: Create the requested robust calibration module and verify syntax plus bias and heading accuracy on the supplied public dataset.

```bash
cat > /holdout_task/mag_calibrator.py <<'PY'
import numpy as np
from scipy.optimize import least_squares

def _level(v, rp):
    phi, theta = rp[:, 0], rp[:, 1]
    c, s = np.cos(phi), np.sin(phi)
    x = v[:, 0]
    y = c * v[:, 1] - s * v[:, 2]
    z = s * v[:, 1] + c * v[:, 2]
    ct, st = np.cos(theta), np.sin(theta)
    return np.column_stack((ct * x + st * z, y, -st * x + ct * z))

def calibrate_and_estimate_heading(session: dict) -> dict:
    m = np.asarray(session['calib_m_raw'], dtype=float)
    rp = np.asarray(session['calib_roll_pitch_rad'], dtype=float)
    def unpack(p):
        b = p[:3]
        L = np.array([[np.exp(p[3]), 0., 0.], [p[6], np.exp(p[4]), 0.], [p[7], p[8], np.exp(p[5])]])
        C = L @ L.T
        C /= np.linalg.det(C) ** (1. / 3.)
        return b, C
    def residual(p):
        b, C = unpack(p)
        v = (m - b) @ C.T
        leveled = _level(v, rp)
        norm = np.linalg.norm(v, axis=1)
        return np.r_[norm - np.median(norm), leveled[:, 2] - np.median(leveled[:, 2])]
    p = np.zeros(9)
    p[:3] = np.median(m, axis=0)
    p[3:6] = np.log(1. / 50.)
    result = least_squares(residual, p, loss='soft_l1', f_scale=0.15, max_nfev=3000)
    result = least_squares(residual, result.x, loss='soft_l1', f_scale=0.05, max_nfev=3000)
    b, C = unpack(result.x)
    v = (np.asarray(session['test_m_raw'], dtype=float) - b) @ C.T
    leveled = _level(v, np.asarray(session['test_roll_pitch_rad'], dtype=float))
    heading = (np.degrees(np.arctan2(-leveled[:, 1], leveled[:, 0])) + 180.) % 360. - 180.
    heading = np.where(heading <= -180., 180., heading)
    return {'heading_deg': heading, 'b_hard': b, 'S_soft': np.linalg.inv(C)}
PY
python3 -m py_compile /holdout_task/mag_calibrator.py && python3 - <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from mag_calibrator import calibrate_and_estimate_heading
x=pickle.load(open('/holdout_task/public_mag.pkl','rb')); o=calibrate_and_estimate_heading(x['session']); g=x['gt']
e=(o['heading_deg']-g['gt_heading_deg']+180)%360-180
print('bias error',np.linalg.norm(o['b_hard']-g['b_hard']))
print('S error',np.linalg.norm(o['S_soft']-g['S_soft']))
print('heading RMSE',np.sqrt(np.mean(e**2)))
PY
```
- **Container Exit Code**: `0` (`906 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/mag_calibrator.py']`
- **Stdout (tail)**:
```text
bias error 0.04236697804947909
S error 0.0010289784576652465
heading RMSE 0.1345472420438102
```

### Turn 7 — `submit_completed_task` (`10527 ms` LLM)
- **Rationale**: The requested module exists and has been syntax-checked and evaluated against the public data; both target accuracy thresholds are met.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/mag_calibrator.py`. It compiles and, on the provided public dataset, achieves a hard-iron bias error of 0.0424 µT and heading RMSE of 0.1345° (soft-iron matrix error: 0.00103).`

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
