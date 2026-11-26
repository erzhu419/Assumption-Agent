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
| **Total Tokens (Prompt / Cached / Out)** | `59,331` (`56,816` / `9,856` / `2,515`) |
| **Total Episode Wall Time (s)** | `92.45s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`12739 ms` LLM)
- **Rationale**: Inspect the public fixture schema and available numerical libraries so the implementation uses the actual input keys and can be checked against the supplied cases.

```bash
ls -lh /holdout_task; python - <<'PY'
import pickle,numpy,scipy
with open('/holdout_task/public_helmholtz_cases.pkl','rb') as f: x=pickle.load(f)
print(numpy.__version__,scipy.__version__)
def describe(x,d=0):
 if isinstance(x,dict):
  print(' '*d, list(x))
  if d<9:
   for k,v in x.items(): print(' '*d,k,end=': '); describe(v,d+2)
 elif isinstance(x,(tuple,list)):
  print(type(x).__name__,len(x));
  if x and d<8:
   for v in (x if isinstance(x,tuple) else x[:1]): describe(v,d+2)
 elif hasattr(x,'shape'): print(x.shape,x.dtype)
 else: print(repr(x)[:150])
describe(x)
PY
```
- **Container Exit Code**: `127` (`350 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 140K
-rw-r--r-- 1 root root 139K Nov 26 00:49 public_helmholtz_cases.pkl
```
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`14270 ms` LLM)
- **Rationale**: Retry with python3 to identify the fixture keys and determine whether SciPy sparse LU is available. If SciPy is absent, select a NumPy-only solver.

```bash
python3 - <<'PY'
import pickle
import numpy as np
try:
 import scipy
 print('SciPy', scipy.__version__)
except ImportError:
 print('SciPy unavailable')
with open('/holdout_task/public_helmholtz_cases.pkl','rb') as f:
 cases=pickle.load(f)
def describe(x, d=0):
 if isinstance(x,dict):
  print('dict')
  for k,v in x.items():
   print(' '*(d+2)+str(k)+': ',end=''); describe(v,d+2)
 elif isinstance(x,(list,tuple)):
  print(type(x).__name__,len(x))
  for v in (x if isinstance(x,tuple) else x[:1]):
   print(' '*(d+2),end=''); describe(v,d+2)
 elif hasattr(x,'shape'):
  print(x.shape,x.dtype)
 else:
  print(repr(x)[:160])
print('NumPy',np.__version__)
describe(cases)
PY
```
- **Container Exit Code**: `0` (`621 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
SciPy 1.13.0
NumPy 1.26.4
list 2
  dict
    problem: dict
      Lx: 1.0
      Ly: 1.0
      k0: 11.5
      beta_imp: 0.85
      n_sq: (32, 32) float64
      basis: (6, 32, 32) float64
      pilot_calibration: dict
        known_field_re: (12,) float64
        known_field_im: (12,) float64
        amplitude_obs: (12,) float64
        wrapped_phase_obs: (12,) float64
      survey: dict
        sensor_ij: (140, 2) int64
        amplitude_obs: (140,) float64
        wrapped_phase_obs: (140,) float64
        tikhonov_alpha: 1e-06
    gt: dict
      source_amplitudes: (6,) float64
      global_phase_rad: -0.8141122105046681
      predicted_field_abs: (32, 32) float64
```

### Turn 3 — `execute_in_remote_docker` (`45754 ms` LLM)
- **Rationale**: Implement sparse complex LU, pilot correlation, and a robust real fit. Compare the forward field independently against the public reference to catch stencil errors, then check all three inversion tolerances to detect calibration or fitting errors.
- **Alternatives Weighed**: `Fit unweighted complex least squares` (rejected: Large receiver spikes would have disproportionate influence on the recovered sources.)

```bash
cat > /holdout_task/helmholtz_phase_inversion.py <<'PY'
"""Finite-volume Helmholtz inversion with pilot phase calibration."""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu


def _basis_fields(problem):
    """Solve once per source basis using a shared sparse LU factorization."""
    n_sq = np.asarray(problem['n_sq'])
    basis = np.asarray(problem['basis'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k0 = problem['k0']
    radiation = -1j * k0 * problem['beta_imp']
    cx, cy = hy / hx, hx / hy
    diagonal = np.asarray(2 * cx + 2 * cy - area * k0**2 * n_sq,
                          dtype=np.complex128)
    diagonal[[0, -1], :] += hy * radiation - cx
    diagonal[:, [0, -1]] += hx * radiation - cy

    # In C order x-neighbors are ny apart; y-neighbors never cross a row.
    x_links = np.full((nx - 1) * ny, -cx)
    y_links = np.full(nx * ny - 1, -cy)
    y_links[ny - 1::ny] = 0
    helmholtz = diags([x_links, y_links, diagonal.ravel(), y_links, x_links],
                      [-ny, -1, 0, 1, ny], format='csc')
    rhs = area * basis.reshape(basis.shape[0], nx * ny).T
    return splu(helmholtz).solve(rhs.astype(np.complex128)).reshape(nx, ny, -1)


def _complex_observations(telemetry):
    return telemetry['amplitude_obs'] * np.exp(1j * telemetry['wrapped_phase_obs'])


def _robust_real_fit(design, observed, alpha):
    """Fit real coefficients, assigning a single robust weight per receiver."""
    m = design.shape[1]
    ridge = np.sqrt(alpha) * np.eye(m)

    def solve(weights):
        w = np.sqrt(weights)
        d, z = design * w[:, None], observed * w
        return np.linalg.lstsq(np.vstack((d.real, d.imag, ridge)),
                               np.concatenate((z.real, z.imag, np.zeros(m))),
                               rcond=None)[0]

    coeff = solve(np.ones(len(observed)))
    # Concentrate the starting fit on the smallest residuals to keep large
    # impulses from inflating the initial inlier-scale estimate.
    keep = max(m, int(np.ceil(0.85 * len(observed))))
    for _ in range(8):
        residual = np.abs(design @ coeff - observed)
        weights = np.zeros(len(observed))
        weights[np.argpartition(residual, keep - 1)[:keep]] = 1
        trial = solve(weights)
        if np.linalg.norm(trial - coeff) <= 1e-9 * (1 + np.linalg.norm(coeff)):
            break
        coeff = trial

    # The median modulus of circular complex Gaussian noise is
    # sqrt(2*log(2)) times the standard deviation of either component.
    residual = np.abs(design @ coeff - observed)
    sigma = np.median(residual) / np.sqrt(2 * np.log(2))
    cutoff = 4.685 * max(sigma, np.finfo(float).eps * np.max(np.abs(observed)))
    if cutoff == 0:
        return coeff
    for _ in range(20):
        residual = np.abs(design @ coeff - observed)
        weights = (1 - np.minimum(residual / cutoff, 1)**2)**2
        trial = solve(weights)
        change = np.linalg.norm(trial - coeff)
        coeff = trial
        if change <= 1e-9 * (1 + np.linalg.norm(coeff)):
            break
    return coeff


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Return real source strengths, oscillator phase, and field magnitude."""
    pilot = problem['pilot_calibration']
    reference = pilot['known_field_re'] + 1j * pilot['known_field_im']
    phase = float(np.angle(np.vdot(reference, _complex_observations(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    fields = _basis_fields(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    observed = _complex_observations(survey) * np.exp(-1j * phase)
    coeff = _robust_real_fit(fields[i, j], observed, survey['tikhonov_alpha'])
    return {'source_amplitudes': coeff, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(fields @ coeff)}
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle
import numpy as np
from helmholtz_phase_inversion import invert_helmholtz_sources_and_phase, _basis_fields
with open('public_helmholtz_cases.pkl','rb') as f:
    cases=pickle.load(f)
for index, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    result = invert_helmholtz_sources_and_phase(p)
    errors=[]
    for key in ('source_amplitudes', 'predicted_field_abs'):
        assert result[key].shape == gt[key].shape and result[key].dtype == np.float64
        errors.append(np.linalg.norm(result[key]-gt[key])/np.linalg.norm(gt[key]))
    phase=result['global_phase_rad']
    assert type(phase) is float and -np.pi < phase <= np.pi
    phase_error=abs(np.angle(np.exp(1j*(phase-gt['global_phase_rad']))))
    print(f'case {index}: source={errors[0]:.6g}, field={errors[1]:.6g}, phase={phase_error:.6g}')
    fields=_basis_fields(p)
    fwd=np.abs(fields @ gt['source_amplitudes'])
    fwd_error=np.linalg.norm(fwd-gt['predicted_field_abs'])/np.linalg.norm(gt['predicted_field_abs'])
    print(f'  forward reference error={fwd_error:.6g}')
    assert fwd_error < 1e-10
    assert max(errors)<=.03 and phase_error<=.02
PY
```
- **Container Exit Code**: `0` (`566 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
case 0: source=0.00304134, field=0.00216737, phase=0.000622723
  forward reference error=1.01811e-15
case 1: source=0.00577186, field=0.00367692, phase=0.000266896
  forward reference error=9.74694e-16
```

### Turn 4 — `submit_completed_task` (`9927 ms` LLM)
- **Rationale**: The required artifact was saved and verified on both public cases. The forward stencil also matched the reference fields to numerical precision.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/helmholtz_phase_inversion.py`. Both public cases passed all tolerances: worst-case relative errors were 0.58% for source amplitudes and 0.37% for field magnitude, and the largest circular phase error was 0.00063 rad.`

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
