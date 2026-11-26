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
| **Total Tokens (Prompt / Cached / Out)** | `58,653` (`56,362` / `22,912` / `2,291`) |
| **Total Episode Wall Time (s)** | `86.89s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`11277 ms` LLM)
- **Rationale**: Inspect the public input schema and SciPy availability; either can affect the stencil assembly and solver choice.

```bash
cd /holdout_task && ls -lh && python - <<'PY'
import pickle, numpy as np, scipy
x=pickle.load(open('public_helmholtz_cases.pkl','rb'))
print('scipy',scipy.__version__)
def describe(x,d=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print(' '*d,k,end=': '); describe(v,d+2)
 elif isinstance(x,(tuple,list)):
  print(type(x).__name__,len(x));
  if x: describe(x[0],d+2)
 elif isinstance(x,np.ndarray): print(x.shape,x.dtype)
 else: print(repr(x)[:250])
describe(x)
PY
```
- **Container Exit Code**: `127` (`339 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 140K
-rw-r--r-- 1 root root 139K Nov 26 00:49 public_helmholtz_cases.pkl
```
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`12160 ms` LLM)
- **Rationale**: Use python3 to inspect the input schema and determine whether sparse LU is available.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle, numpy as np
try:
 import scipy
 print('scipy',scipy.__version__)
except ImportError:
 print('no scipy')
def describe(x,d=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print(' '*d,k,end=': '); describe(v,d+2)
 elif isinstance(x,(tuple,list)):
  print(type(x).__name__,len(x))
  if x: describe(x[0],d+2)
 elif isinstance(x,np.ndarray): print(x.shape,x.dtype)
 else: print(repr(x)[:200])
describe(pickle.load(open('public_helmholtz_cases.pkl','rb')))
PY
```
- **Container Exit Code**: `0` (`646 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
scipy 1.13.0
list 2
   problem:      Lx: 1.0
     Ly: 1.0
     k0: 11.5
     beta_imp: 0.85
     n_sq: (32, 32) float64
     basis: (6, 32, 32) float64
     pilot_calibration:        known_field_re: (12,) float64
       known_field_im: (12,) float64
       amplitude_obs: (12,) float64
       wrapped_phase_obs: (12,) float64
     survey:        sensor_ij: (140, 2) int64
       amplitude_obs: (140,) float64
       wrapped_phase_obs: (140,) float64
       tikhonov_alpha: 1e-06
   gt:      source_amplitudes: (6,) float64
     global_phase_rad: -0.8141122105046681
     predicted_field_abs: (32, 32) float64
```

### Turn 3 — `execute_in_remote_docker` (`41704 ms` LLM)
- **Rationale**: Implement the indefinite solve and robust inversion, then check public-case tolerances to detect stencil, calibration, or regression errors.
- **Alternatives Weighed**: `Unwrap phases and fit the phase angles directly` (rejected: Polar reconstruction avoids branch-cut errors and permits robust fitting of the linear complex receiver response.)

```bash
cd /holdout_task && cat > helmholtz_phase_inversion.py <<'PY'
"""Finite-volume Helmholtz source inversion from polar receiver data."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _path_laplacian(n):
    edges = -np.ones(n - 1)
    degree = np.zeros(n)
    degree[:-1] += 1
    degree[1:] += 1
    return diags((edges, degree, edges), (-1, 0, 1), shape=(n, n))


def _unit_source_fields(problem):
    n_sq = np.asarray(problem['n_sq'])
    basis = np.asarray(problem['basis'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k0 = problem['k0']

    # Kronecker order follows the C-order index i * ny + j.
    flux = ((hy / hx) * kron(_path_laplacian(nx), eye(ny))
            + (hx / hy) * kron(eye(nx), _path_laplacian(ny)))
    wall = np.zeros((nx, ny))
    wall[0, :] += hy
    wall[-1, :] += hy
    wall[:, 0] += hx
    wall[:, -1] += hx
    wave = -area * k0**2 * n_sq - 1j * k0 * problem['beta_imp'] * wall
    operator = (flux + diags(wave.ravel())).tocsc()
    rhs = np.asfortranarray(area * basis.reshape(basis.shape[0], -1).T,
                            dtype=np.complex128)
    return splu(operator).solve(rhs).reshape(nx, ny, -1)


def _complex_observations(group):
    return (np.asarray(group['amplitude_obs'])
            * np.exp(1j * np.asarray(group['wrapped_phase_obs'])))


def _fit_real_sources(response, data, alpha):
    """Robust complex regression, assigning one weight per receiver."""
    design = np.concatenate((response.real, response.imag), axis=0)
    target = np.concatenate((data.real, data.imag))
    ridge = np.sqrt(alpha) * np.eye(response.shape[1])
    zeros = np.zeros(response.shape[1])

    def fit(weights):
        w = np.tile(np.sqrt(weights), 2)
        return np.linalg.lstsq(np.vstack((w[:, None] * design, ridge)),
                               np.concatenate((w * target, zeros)), rcond=None)[0]

    def residual_and_scale(a):
        residual = np.abs(response @ a - data)
        # Rayleigh median for the radius of circular complex Gaussian noise.
        floor = np.finfo(float).eps * max(np.max(np.abs(data)), np.finfo(float).tiny)
        scale = max(np.median(residual) / np.sqrt(2 * np.log(2)), floor)
        return residual, scale

    a = fit(np.ones(data.size))
    # Huber steps give a stable starting point even when the initial fit is
    # contaminated; bisquare steps then reject the remaining large impulses.
    for redescending in (False, True):
        for _ in range(30):
            residual, scale = residual_and_scale(a)
            if redescending:
                ratio = np.minimum(residual / (4.685 * scale), 1.0)
                weights = (1 - ratio**2)**2
            else:
                cutoff = 1.5 * scale
                weights = cutoff / np.maximum(residual, cutoff)
            updated = fit(weights)
            converged = np.linalg.norm(updated - a) <= 1e-10 * (1 + np.linalg.norm(a))
            a = updated
            if converged:
                break
    return a


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Calibrate receiver phase and infer real source strengths and pressure magnitude."""
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _complex_observations(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    fields = _unit_source_fields(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    data = np.exp(-1j * phase) * _complex_observations(survey)
    amplitudes = _fit_real_sources(fields[i, j], data, survey['tikhonov_alpha'])
    return {'source_amplitudes': amplitudes,
            'global_phase_rad': phase,
            'predicted_field_abs': np.abs(fields @ amplitudes)}
PY
python3 - <<'PY'
import pickle, numpy as np
from helmholtz_phase_inversion import invert_helmholtz_sources_and_phase, _unit_source_fields
with open('public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
for n, case in enumerate(cases):
    p, truth = case['problem'], case['gt']
    got = invert_helmholtz_sources_and_phase(p)
    errs = {}
    for k in ('source_amplitudes', 'predicted_field_abs'):
        assert got[k].shape == truth[k].shape and got[k].dtype == np.float64
        errs[k] = np.linalg.norm(got[k] - truth[k]) / np.linalg.norm(truth[k])
    angle = got['global_phase_rad']
    assert -np.pi < angle <= np.pi
    errs['global_phase_rad'] = abs(np.angle(np.exp(1j * (angle - truth['global_phase_rad']))))
    print(n, errs, 'cond', np.linalg.cond(np.vstack([_unit_source_fields(p)[tuple(p['survey']['sensor_ij'].T)].real, _unit_source_fields(p)[tuple(p['survey']['sensor_ij'].T)].imag])))
    assert errs['global_phase_rad'] <= .020 and max(errs['source_amplitudes'], errs['predicted_field_abs']) <= .030
PY
```
- **Container Exit Code**: `0` (`765 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
0 {'source_amplitudes': 0.003043555885931824, 'predicted_field_abs': 0.0021676659525429382, 'global_phase_rad': 0.0006227225657142733} cond 2.9618123405365107
1 {'source_amplitudes': 0.0058381412505026835, 'predicted_field_abs': 0.003706654370402288, 'global_phase_rad': 0.00026689595102702407} cond 3.11360940557194
```

### Turn 4 — `submit_completed_task` (`10361 ms` LLM)
- **Rationale**: The module was saved, imported, and verified against both public cases. All three errors are below the requested tolerances.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/helmholtz_phase_inversion.py` with the requested inversion function. Both public cases passed: maximum relative error was 0.584% for source amplitudes and 0.371% for field magnitude, and maximum circular phase error was 0.000623 rad.`

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
