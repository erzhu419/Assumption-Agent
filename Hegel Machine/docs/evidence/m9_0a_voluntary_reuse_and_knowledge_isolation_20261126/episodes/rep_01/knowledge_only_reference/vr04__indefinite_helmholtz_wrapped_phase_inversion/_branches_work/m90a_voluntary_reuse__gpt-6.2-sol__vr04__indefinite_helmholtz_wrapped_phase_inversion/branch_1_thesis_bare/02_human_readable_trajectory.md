# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Bare_ReAct_LiveGPT` | `vr04__indefinite_helmholtz_wrapped_phase_inversion`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M90a-VoluntaryReuse-4` |
| **Domain** | `Indefinite Acoustic Helmholtz Scattering & Wrapped-Phase Receiver Calibration` |
| **Wave / Wormhole Family** | `m90a_voluntary_reuse` / `fam_indefinite_wave_scattering_negative_control` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Bare_ReAct_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `4 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `59,544` (`57,199` / `29,568` / `2,345`) |
| **Total Episode Wall Time (s)** | `85.41s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`13272 ms` LLM)
- **Rationale**: Inspect the fixture structure and numerical dependencies to identify the actual input keys, available reference outputs, and suitable sparse solver before implementing.

```bash
ls -l /holdout_task; python - <<'PY'
import pickle,numpy as np,scipy
with open('/holdout_task/public_helmholtz_cases.pkl','rb') as f: x=pickle.load(f)
def show(x,d=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print(' '*d,k, end=': ')
   if isinstance(v,dict): print(); show(v,d+2)
   elif hasattr(v,'shape'): print(v.shape,v.dtype)
   elif isinstance(v,(float,int,str)): print(v)
   else: print(type(v),len(v) if hasattr(v,'__len__') else '')
 elif isinstance(x,(tuple,list)):
  print(type(x),len(x)); show(x[0],d+2)
 else: print(type(x))
show(x)
if isinstance(x,dict):
 for k,v in x.items():
  if isinstance(v,(list,tuple)) and v: show(v[0],2)
print('scipy',scipy.__version__)
PY
```
- **Container Exit Code**: `127` (`331 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 140
-rw-r--r-- 1 root root 142215 Nov 26 00:50 public_helmholtz_cases.pkl
```
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`12754 ms` LLM)
- **Rationale**: Retry fixture inspection with python3 after the python executable was unavailable. The actual keys, array shapes, references, and installed solver determine the implementation and public verification.

```bash
command -v python3; ls -la /holdout_task; python3 - <<'PY'
import pickle, numpy as np, scipy
from pathlib import Path
with Path('/holdout_task/public_helmholtz_cases.pkl').open('rb') as f:
    data=pickle.load(f)
def show(x, level=0):
    p=' '*level
    if isinstance(x,dict):
        for k,v in x.items():
            print(p+str(k)+':',end=' ')
            if isinstance(v,(dict,tuple,list)): print(); show(v,level+2)
            elif hasattr(v,'shape'): print(v.shape,v.dtype)
            else: print(repr(v))
    elif isinstance(x,(tuple,list)):
        print(p,type(x).__name__,len(x))
        if x: show(x[0],level+2)
    else: print(p,type(x))
show(data)
print('scipy',scipy.__version__)
PY
```
- **Container Exit Code**: `0` (`584 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/usr/bin/python3
total 148
drwxr-xr-x 2 root root   4096 Nov 26 00:50 .
drwxr-xr-x 1 root root   4096 Nov 26 00:50 ..
-rw-r--r-- 1 root root 142215 Nov 26 00:50 public_helmholtz_cases.pkl
 list 2
  problem: 
    Lx: 1.0
    Ly: 1.0
    k0: 11.5
    beta_imp: 0.85
    n_sq: (32, 32) float64
    basis: (6, 32, 32) float64
    pilot_calibration: 
      known_field_re: (12,) float64
      known_field_im: (12,) float64
      amplitude_obs: (12,) float64
      wrapped_phase_obs: (12,) float64
    survey: 
      sensor_ij: (140, 2) int64
      amplitude_obs: (140,) float64
      wrapped_phase_obs: (140,) float64
      tikhonov_alpha: 1e-06
  gt: 
    source_amplitudes: (6,) float64
    global_phase_rad: -0.8141122105046681
    predicted_field_abs: (32, 32) float64
scipy 1.13.0
```

### Turn 3 — `execute_in_remote_docker` (`40642 ms` LLM)
- **Rationale**: Implement the solver and test against both public references. A mismatch using the reference coefficients would indicate a stencil error; disagreement only in the inversion would point to phase calibration or robust fitting.
- **Alternatives Weighed**: `Ordinary complex least squares` (rejected: A few large complex spikes could dominate the fit; the real-valued coefficients require a joint real/imaginary fit with robust receiver weights.)

```bash
cat > /holdout_task/helmholtz_phase_inversion.py <<'PY'
"""Finite-volume Helmholtz source inversion from polar receiver telemetry."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis_stiffness(n):
    # Path graph Laplacian: only interior faces contribute here.
    main = np.full(n, 2.0)
    main[[0, -1]] = 1.0
    off = -np.ones(n - 1)
    return diags((off, main, off), (-1, 0, 1), shape=(n, n), format="csc")


def _field_responses(problem):
    n_sq = np.asarray(problem['n_sq'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    k0 = problem['k0']
    cell_area = hx * hy
    lap = (hy / hx * kron(_axis_stiffness(nx), eye(ny), format="csc")
           + hx / hy * kron(eye(nx), _axis_stiffness(ny), format="csc"))
    wall_length = np.zeros((nx, ny))
    wall_length[0, :] += hy
    wall_length[-1, :] += hy
    wall_length[:, 0] += hx
    wall_length[:, -1] += hx
    diagonal = -cell_area * k0**2 * n_sq - 1j * k0 * problem['beta_imp'] * wall_length
    operator = lap + diags(diagonal.ravel(), format="csc")
    basis = np.asarray(problem['basis'], dtype=np.complex128)
    rhs = cell_area * basis.reshape(basis.shape[0], nx * ny).T
    return splu(operator).solve(rhs).reshape(nx, ny, basis.shape[0])


def _complex_observations(data):
    return np.asarray(data['amplitude_obs']) * np.exp(1j * np.asarray(data['wrapped_phase_obs']))


def _robust_real_fit(response, observed, alpha):
    # Fit both quadratures with a single weight per receiver. The pilot has
    # already determined the oscillator phase, so the coefficients are real.
    design = np.concatenate((response.real, response.imag), axis=0)
    target = np.concatenate((observed.real, observed.imag))
    m = response.shape[1]
    penalty = np.sqrt(alpha) * np.eye(m)

    def fit(weights):
        root = np.sqrt(np.tile(weights, 2))
        return np.linalg.lstsq(np.vstack((root[:, None] * design, penalty)),
                               np.concatenate((root * target, np.zeros(m))), rcond=None)[0]

    a = fit(np.ones(len(observed)))
    floor = np.finfo(float).eps * max(1.0, np.max(np.abs(observed)))
    # Huber IRLS gets a stable residual scale before rejecting impulses.
    for _ in range(30):
        residual = np.abs(response @ a - observed)
        scale = max(np.median(residual), floor)
        updated = fit(np.minimum(1.0, 1.5 * scale / np.maximum(residual, floor)))
        if np.linalg.norm(updated - a) <= 1e-9 * max(1.0, np.linalg.norm(a)):
            a = updated
            break
        a = updated
    residual = np.abs(response @ a - observed)
    inliers = residual <= 4.0 * max(np.median(residual), floor)
    return fit(inliers) if np.count_nonzero(inliers) >= m else a


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Estimate real source amplitudes, oscillator phase and field magnitude."""
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _complex_observations(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    fields = _field_responses(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    observed = np.exp(-1j * phase) * _complex_observations(survey)
    a = _robust_real_fit(fields[i, j], observed, survey['tikhonov_alpha'])
    return {'source_amplitudes': a, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(fields @ a)}
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -B - <<'PY'
import pickle
import numpy as np
import helmholtz_phase_inversion as h
with open('public_helmholtz_cases.pkl','rb') as f: cases=pickle.load(f)
for n,c in enumerate(cases):
 p,g=c['problem'],c['gt']; y=h.invert_helmholtz_sources_and_phase(p)
 a=g['source_amplitudes']; field=g['predicted_field_abs']; phase=g['global_phase_rad']; s=p['survey']; i,j=s['sensor_ij'].T
 F=h._field_responses(p); A=F[i,j]; z=h._complex_observations(s)*np.exp(-1j*phase)
 rel=lambda x,t: np.linalg.norm(x-t)/np.linalg.norm(t)
 print(n,'source',rel(y['source_amplitudes'],a),'field',rel(y['predicted_field_abs'],field),'phase',abs(np.angle(np.exp(1j*(y['global_phase_rad']-phase)))),'forward',rel(abs(F@a),field))
 r=np.abs(A@a-z); print('true residual percentiles',np.percentile(r,[0,10,50,80,90,92,93,94,95,99,100]),'signal percentiles',np.percentile(abs(A@a),[0,50,100]),'Gram eigenvalues',np.linalg.eigvalsh((A.conj().T@A).real))
PY
```
- **Container Exit Code**: `0` (`502 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
0 source 0.0035790065572615234 field 0.002289175829730886 phase 0.0006227225657142733 forward 1.2624230001329256e-15
true residual percentiles [6.47881901e-06 7.60648847e-04 2.04397750e-03 2.75476983e-03
 3.74096331e-03 4.01262154e-03 1.00151822e-01 3.61614744e-01
 3.72538342e-01 6.29259867e-01 7.40687115e-01] signal percentiles [0.00422408 0.06468471 0.20136088] Gram eigenvalues [0.08248799 0.15317816 0.23978029 0.41469183 0.46617885 0.72361205]
1 source 0.004715349089309256 field 0.003201396421629576 phase 0.00026689595102702407 forward 8.893619912307587e-16
true residual percentiles [2.09377251e-04 7.53740557e-04 1.80389227e-03 2.88952944e-03
 4.26435661e-03 4.83633554e-03 9.99000576e-02 3.74277075e-01
 4.09010034e-01 6.54635093e-01 6.86029873e-01] signal percentiles [0.00373039 0.08598488 0.21292411] Gram eigenvalues [0.07992072 0.15610966 0.19661217 0.40584847 0.48390185 0.77479647]
```

### Turn 4 — `submit_completed_task` (`9248 ms` LLM)
- **Rationale**: The requested module exists, both public cases passed, and the forward stencil reproduced the reference field magnitudes to numerical precision when given the reference amplitudes.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/helmholtz_phase_inversion.py` and verified it on both public cases. The worst relative errors were 0.472% for source amplitudes and 0.321% for field magnitude; the worst circular phase error was 0.000623 rad. All were within the requested tolerances.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M90a-VoluntaryReuse-4",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
