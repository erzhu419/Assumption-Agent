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
| **Total Tokens (Prompt / Cached / Out)** | `59,516` (`56,922` / `0` / `2,594`) |
| **Total Episode Wall Time (s)** | `91.89s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`11981 ms` LLM)
- **Rationale**: Inspect the public fixture for exact input keys, array shapes, and available reference outputs; these determine how to wire and verify the solver.

```bash
ls -la /holdout_task; python - <<'PY'
import pickle, numpy as np, scipy
p='/holdout_task/public_helmholtz_cases.pkl'
with open(p,'rb') as f: x=pickle.load(f)
def describe(x,d=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print(' '*d,k,end=': '); describe(v,d+2)
 elif isinstance(x,(tuple,list)):
  print(type(x).__name__,len(x));
  if x and d < 8: describe(x[0],d+2)
 elif isinstance(x,np.ndarray): print(x.shape,x.dtype)
 else: print(repr(x)[:180])
describe(x)
print('SciPy',scipy.__version__)
PY
```
- **Container Exit Code**: `127` (`309 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 148
drwxr-xr-x 2 root root   4096 Nov 26 00:50 .
drwxr-xr-x 1 root root   4096 Nov 26 00:50 ..
-rw-r--r-- 1 root root 142215 Nov 26 00:50 public_helmholtz_cases.pkl
```
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`12211 ms` LLM)
- **Rationale**: Use python3 after the missing python executable. Inspect fixture keys and references to avoid guessing the input schema and to identify the available accuracy tests.

```bash
command -v python3; python3 - <<'PY'
import pickle
import numpy as np
import scipy
with open('/holdout_task/public_helmholtz_cases.pkl','rb') as f:
    data=pickle.load(f)
def show(x,d=0):
    if isinstance(x,dict):
        print('dict')
        for k,v in x.items():
            print(' '*d+str(k)+': ',end=''); show(v,d+2)
    elif isinstance(x,(list,tuple)):
        print(type(x).__name__,len(x))
        if x and d<8: show(x[0],d+2)
    elif isinstance(x,np.ndarray): print(x.shape,x.dtype)
    else: print(repr(x)[:150])
show(data)
print('SciPy',scipy.__version__)
PY
```
- **Container Exit Code**: `0` (`606 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/usr/bin/python3
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
SciPy 1.13.0
```

### Turn 3 — `execute_in_remote_docker` (`47677 ms` LLM)
- **Rationale**: Implement the specified sparse complex stencil and pilot calibration, then test public accuracy and compare forward predictions at the true amplitudes to distinguish stencil errors from fitting errors.
- **Alternatives Weighed**: `Ordinary real least squares` (rejected: Large complex spikes would directly influence the recovered source amplitudes.)

```bash
cat > /holdout_task/helmholtz_phase_inversion.py <<'PY'
"""Finite-volume Helmholtz inversion from calibrated polar telemetry."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _line_laplacian(n):
    """Unit face conductances; exterior faces are handled separately."""
    off = np.ones(n - 1)
    diagonal = np.zeros(n)
    diagonal[:-1] += off
    diagonal[1:] += off
    return diags((-off, diagonal, -off), (-1, 0, 1), format="csc")


def _source_responses(problem, n_sq, basis):
    nx, ny = n_sq.shape
    hx, hy = problem["Lx"] / nx, problem["Ly"] / ny
    area = hx * hy
    k0 = problem["k0"]

    face_length = np.zeros((nx, ny))
    face_length[0, :] += hy
    face_length[-1, :] += hy
    face_length[:, 0] += hx
    face_length[:, -1] += hx
    diagonal = -area * k0**2 * n_sq - 1j * k0 * problem["beta_imp"] * face_length
    operator = (
        (hy / hx) * kron(_line_laplacian(nx), eye(ny), format="csc")
        + (hx / hy) * kron(eye(nx), _line_laplacian(ny), format="csc")
        + diags(diagonal.ravel(), format="csc")
    )
    rhs = area * basis.reshape(basis.shape[0], -1).T
    return splu(operator.tocsc()).solve(rhs.astype(np.complex128))


def _from_polar(telemetry):
    return np.asarray(telemetry["amplitude_obs"]) * np.exp(
        1j * np.asarray(telemetry["wrapped_phase_obs"])
    )


def _robust_real_fit(design, measured, alpha):
    """Fit real coefficients, downweighting each complex channel as a pair."""
    m = design.shape[1]
    target = np.r_[measured.real, measured.imag, np.zeros(m)]
    matrix = np.vstack((design.real, design.imag, np.sqrt(alpha) * np.eye(m)))

    def fit(weights):
        root = np.sqrt(weights)
        scale = np.r_[root, root, np.ones(m)]
        return np.linalg.lstsq(matrix * scale[:, None], target * scale, rcond=None)[0]

    coeff = fit(np.ones(measured.size))
    floor = max(np.finfo(float).tiny,
                1e-12 * np.median(np.abs(measured)))

    # Start with Huber weights to limit the influence of spikes without
    # discarding channels while the residual scale is still settling.
    for _ in range(30):
        residual = np.abs(design @ coeff - measured)
        cutoff = 1.5 * max(floor, np.median(residual))
        updated = fit(cutoff / np.maximum(cutoff, residual))
        if np.linalg.norm(updated - coeff) <= 1e-9 * max(1.0, np.linalg.norm(updated)):
            coeff = updated
            break
        coeff = updated

    # Freeze the robust scale and use a redescending fit so very large
    # amplitude/phase jumps have no leverage on the final coefficients.
    cutoff = 4.0 * max(floor, np.median(np.abs(design @ coeff - measured)))
    for _ in range(25):
        scaled = np.minimum(1.0, np.abs(design @ coeff - measured) / cutoff)
        updated = fit((1.0 - scaled**2)**2)
        if np.linalg.norm(updated - coeff) <= 1e-9 * max(1.0, np.linalg.norm(updated)):
            return updated
        coeff = updated
    return coeff


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Estimate oscillator phase, real source amplitudes, and field modulus."""
    n_sq = np.asarray(problem["n_sq"])
    basis = np.asarray(problem["basis"])
    pilot = problem["pilot_calibration"]
    known = np.asarray(pilot["known_field_re"]) + 1j * np.asarray(pilot["known_field_im"])
    phase = float(np.angle(np.vdot(known, _from_polar(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    response = _source_responses(problem, n_sq, basis)
    survey = problem["survey"]
    ij = np.asarray(survey["sensor_ij"])
    rows = np.ravel_multi_index((ij[:, 0], ij[:, 1]), n_sq.shape)
    measured = _from_polar(survey) * np.exp(-1j * phase)
    coeff = _robust_real_fit(response[rows], measured, survey["tikhonov_alpha"])
    return {
        "source_amplitudes": coeff,
        "global_phase_rad": phase,
        "predicted_field_abs": np.abs(response @ coeff).reshape(n_sq.shape),
    }
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -B - <<'PY'
import pickle
import numpy as np
from helmholtz_phase_inversion import invert_helmholtz_sources_and_phase, _source_responses
with open('public_helmholtz_cases.pkl','rb') as f:
    cases = pickle.load(f)
for i, case in enumerate(cases):
    p, truth = case['problem'], case['gt']
    got = invert_helmholtz_sources_and_phase(p)
    source = got['source_amplitudes']; field = got['predicted_field_abs']; phase = got['global_phase_rad']
    assert source.shape == truth['source_amplitudes'].shape and field.shape == p['n_sq'].shape
    assert source.dtype == field.dtype == np.float64 and isinstance(phase,float) and -np.pi < phase <= np.pi
    def rel(a,b): return np.linalg.norm(a-b)/np.linalg.norm(b)
    errs = (rel(source,truth['source_amplitudes']), abs(np.angle(np.exp(1j*(phase-truth['global_phase_rad'])))), rel(field,truth['predicted_field_abs']))
    response = _source_responses(p,p['n_sq'],p['basis'])
    forward_err = rel(np.abs(response @ truth['source_amplitudes']).reshape(field.shape),truth['predicted_field_abs'])
    print(i, 'errors (source, phase, field):', errs, 'forward error with true sources:', forward_err)
    assert np.all(np.array(errs) <= [0.030, 0.020, 0.030])
PY
```
- **Container Exit Code**: `0` (`495 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
0 errors (source, phase, field): (0.0030518774610387035, 0.0006227225657142733, 0.0021687917468075713) forward error with true sources: 1.2624230001329256e-15
1 errors (source, phase, field): (0.0057510483732139215, 0.00026689595102702407, 0.0036676213996186535) forward error with true sources: 8.893619912307587e-16
```

### Turn 4 — `submit_completed_task` (`10244 ms` LLM)
- **Rationale**: The requested module was saved and verified on both public cases. Forward predictions at the true source amplitudes matched the reference fields to numerical precision.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/helmholtz_phase_inversion.py` using sparse complex LU, pilot-based phase calibration, and robust fitting of real source amplitudes. Both public cases passed all tolerances. The largest observed errors were 0.58% for source amplitudes, 0.00063 rad for phase, and 0.37% for field magnitude.`

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
