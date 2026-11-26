# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `vr04__indefinite_helmholtz_wrapped_phase_inversion`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M90a-VoluntaryReuse-4` |
| **Domain** | `Indefinite Acoustic Helmholtz Scattering & Wrapped-Phase Receiver Calibration` |
| **Wave / Wormhole Family** | `m90a_voluntary_reuse` / `fam_indefinite_wave_scattering_negative_control` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `8 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `161,529` (`154,688` / `35,328` / `6,841`) |
| **Total Episode Wall Time (s)** | `227.61s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`17838 ms` LLM)
- **Rationale**: Inspect only the fixture schema and numerical dependencies. This catches a field-layout mismatch before implementation and determines whether validation can compare all three outputs directly against public references.
- **Alternatives Weighed**: `Fit wrapped phase angles directly` (rejected: Complex reconstruction removes the branch discontinuity and preserves both measured quadratures.)

```bash
cd /holdout_task && python3 -I - <<'PY'
import pickle
import numpy as np
import scipy
with open('public_helmholtz_cases.pkl','rb') as f:
    data=pickle.load(f)
print('scipy',scipy.__version__)
def describe(obj, depth=0):
    if isinstance(obj,dict) and depth<5:
        for k,v in list(obj.items())[:25]:
            print('  '*depth,repr(k),end=': ')
            if isinstance(v,(dict,list,tuple)): print()
            describe(v,depth+1)
    elif isinstance(obj,(list,tuple)) and depth<5:
        print('  '*depth,type(obj).__name__,len(obj))
        if obj: describe(obj[0],depth+1)
    elif isinstance(obj,np.ndarray):
        print(obj.shape,obj.dtype)
    else:
        print(type(obj).__name__,repr(obj)[:110])
describe(data)
PY
```
- **Container Exit Code**: `0` (`611 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
scipy 1.13.0
 list 2
   'problem': 
     'Lx': float 1.0
     'Ly': float 1.0
     'k0': float 11.5
     'beta_imp': float 0.85
     'n_sq': (32, 32) float64
     'basis': (6, 32, 32) float64
     'pilot_calibration': 
       'known_field_re': (12,) float64
       'known_field_im': (12,) float64
       'amplitude_obs': (12,) float64
       'wrapped_phase_obs': (12,) float64
     'survey': 
       'sensor_ij': (140, 2) int64
       'amplitude_obs': (140,) float64
       'wrapped_phase_obs': (140,) float64
       'tikhonov_alpha': float 1e-06
   'gt': 
     'source_amplitudes': (6,) float64
     'global_phase_rad': float -0.8141122105046681
     'predicted_field_abs': (32, 32) float64
```

### Turn 2 — `execute_in_remote_docker` (`46528 ms` LLM)
- **Rationale**: Compare the forward field at the true sources to isolate stencil errors from inversion errors. Survey residual quantiles and design singular values will show whether clipping or regularization needs adjustment.
- **Alternatives Weighed**: `Ordinary ridge fit on all receivers` (rejected: The specified impulse spikes can dominate squared residuals.)

```bash
cd /holdout_task && cat > helmholtz_phase_inversion.py <<'PY'
"""Invert real acoustic sources from phase-calibrated receiver measurements."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _path_laplacian(n):
    off = -np.ones(n - 1)
    degree = np.zeros(n)
    degree[:-1] += 1
    degree[1:] += 1
    return diags((off, degree, off), (-1, 0, 1), format='csc')


def _responses(problem):
    n_sq = np.asarray(problem['n_sq'], dtype=float)
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    k = problem['k0']
    face_length = np.zeros_like(n_sq)
    face_length[0, :] += hy
    face_length[-1, :] += hy
    face_length[:, 0] += hx
    face_length[:, -1] += hx
    diagonal = -hx * hy * k**2 * n_sq - 1j * k * problem['beta_imp'] * face_length
    operator = (hy / hx * kron(_path_laplacian(nx), eye(ny), format='csc')
                + hx / hy * kron(eye(nx), _path_laplacian(ny), format='csc')
                + diags(diagonal.ravel(), format='csc'))
    basis = np.asarray(problem['basis'], dtype=float)
    rhs = (hx * hy * basis.reshape(basis.shape[0], nx * ny).T).astype(complex)
    return splu(operator).solve(rhs).reshape(nx, ny, basis.shape[0])


def _polar(data):
    return np.asarray(data['amplitude_obs']) * np.exp(1j * np.asarray(data['wrapped_phase_obs']))


def _fit_sources(response, observed, alpha):
    # Both quadratures of a receiver must be downweighted together.
    n, m = response.shape
    design = np.concatenate((response.real, response.imag), axis=0)
    target = np.concatenate((observed.real, observed.imag))
    penalty = np.sqrt(alpha) * np.eye(m)
    floor = 64 * np.finfo(float).eps * np.max(np.abs(observed))

    def fit(weight, ridge=True):
        w = np.tile(np.sqrt(weight), 2)
        x, y = design * w[:, None], target * w
        if ridge and alpha > 0:
            x = np.vstack((x, penalty))
            y = np.concatenate((y, np.zeros(m)))
        return np.linalg.lstsq(x, y, rcond=None)[0]

    a = fit(np.ones(n), ridge=False)
    for _ in range(30):
        r = np.abs(response @ a - observed)
        cutoff = max(1.5 * np.median(r), floor, np.finfo(float).tiny)
        updated = fit(cutoff / np.maximum(r, cutoff), ridge=False)
        if np.linalg.norm(updated - a) <= 1e-10 * max(1., np.linalg.norm(a)):
            a = updated
            break
        a = updated

    # Refit on the consensus for full efficiency under small receiver noise.
    previous = None
    for _ in range(8):
        r = np.abs(response @ a - observed)
        keep = r <= max(4.0 * np.median(r), floor)
        if np.array_equal(keep, previous) or keep.sum() < m:
            break
        a = fit(keep)
        previous = keep
    return a


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Return the real source coefficients, shared phase, and field magnitude."""
    pilots = problem['pilot_calibration']
    known = np.asarray(pilots['known_field_re']) + 1j * np.asarray(pilots['known_field_im'])
    phase = float(np.angle(np.vdot(known, _polar(pilots))))
    if phase <= -np.pi:
        phase = float(np.pi)

    field_response = _responses(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    observed = _polar(survey) * np.exp(-1j * phase)
    a = _fit_sources(field_response[i, j], observed, survey['tikhonov_alpha'])
    return {'source_amplitudes': a, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(field_response @ a)}
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/helmholtz_phase_inversion.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
for k,c in enumerate(pickle.load(open('/holdout_task/public_helmholtz_cases.pkl','rb'))):
    p,g=c['problem'],c['gt']; y=m.invert_helmholtz_sources_and_phase(p); f=m._responses(p); i,j=p['survey']['sensor_ij'].T
    rel=lambda a,b: np.linalg.norm(a-b)/np.linalg.norm(b)
    z=m._polar(p['survey'])*np.exp(-1j*g['global_phase_rad']); A=f[i,j]; a=g['source_amplitudes']; r=np.abs(z-A@a); v=abs(A@a)
    print(k, 'metrics',rel(y['source_amplitudes'],a),abs(np.angle(np.exp(1j*(y['global_phase_rad']-g['global_phase_rad'])))),rel(y['predicted_field_abs'],g['predicted_field_abs']), 'forward',rel(abs(f@a),g['predicted_field_abs']))
    print('receiver abs residual',np.quantile(r,[0,.1,.5,.8,.9,.93,.95,.99,1]),'signal',np.quantile(v,[0,.5,1]),'singular values',np.linalg.svd(np.vstack([A.real,A.imag]),compute_uv=False),'alpha',p['survey']['tikhonov_alpha'])
PY
```
- **Container Exit Code**: `0` (`508 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
0 metrics 0.0035790065572618222 0.0006227225657142733 0.0022891758297308683 forward 1.2624230001329256e-15
receiver abs residual [6.47881901e-06 7.60648847e-04 2.04397750e-03 2.75476983e-03
 3.74096331e-03 1.00151822e-01 3.72538342e-01 6.29259867e-01
 7.40687115e-01] signal [0.00422408 0.06468471 0.20136088] singular values [0.8506539  0.68277291 0.64396571 0.48967366 0.39137982 0.28720722] alpha 1e-06
1 metrics 0.004715349089309256 0.00026689595102702407 0.003201396421629576 forward 8.893619912307587e-16
receiver abs residual [2.09377251e-04 7.53740557e-04 1.80389227e-03 2.88952944e-03
 4.26435661e-03 9.99000576e-02 4.09010034e-01 6.54635093e-01
 6.86029873e-01] signal [0.00373039 0.08598488 0.21292411] singular values [0.88022524 0.69563054 0.63706238 0.44340971 0.39510715 0.28270252] alpha 1e-06

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7630227135ac9a8c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76302271 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`39482 ms` LLM)
- **Rationale**: Jackknife the measured survey and pilot channels with fixed settings to detect single-channel dependence. The independent rectangular stencil check would expose face-weight or boundary errors hidden by square fixtures; injected impulses at high-leverage receivers test whether the robust fit retains its margin.
- **Alternatives Weighed**: `Retune robust weights on the two public cases` (rejected: All three outputs have substantial margin; first test whether receiver influence, phase wrapping, or non-square geometry actually exposes a failure.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('helmholtz_phase_inversion', '/holdout_task/helmholtz_phase_inversion.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
solve = solver.invert_helmholtz_sources_and_phase
cases = pickle.load(open('public_helmholtz_cases.pkl', 'rb'))
limits = np.array([0.030, 0.020, 0.030])

def validate(p, g):
    ans = solve(p)
    a, phase, field = (ans[k] for k in ('source_amplitudes', 'global_phase_rad', 'predicted_field_abs'))
    assert a.shape == g['source_amplitudes'].shape and a.dtype.kind == 'f'
    assert field.shape == p['n_sq'].shape and field.dtype.kind == 'f' and np.all(field >= 0)
    assert isinstance(phase, float) and -np.pi < phase <= np.pi
    rel = lambda x, y: np.linalg.norm(x - y) / np.linalg.norm(y)
    return np.array([rel(a, g['source_amplitudes']), abs(np.angle(np.exp(1j*(phase-g['global_phase_rad'])))), rel(field,g['predicted_field_abs'])])

for section, size_key in [('survey','sensor_ij'), ('pilot_calibration','known_field_re')]:
    vals, labels = [], []
    for case_id, case in enumerate(cases):
        p, g = case['problem'], case['gt']
        s = p[section]
        n = len(s[size_key])
        for k in range(n):
            keep = np.arange(n) != k
            vals.append(validate(dict(p, **{section: {key: value[keep] if isinstance(value,np.ndarray) else value for key,value in s.items()}}),g))
            labels.append((case_id,k))
    vals = np.asarray(vals)
    print(section, 'jackknife', len(vals), 'max source/phase/field:', vals.max(0), 'worst indices:', [labels[i] for i in vals.argmax(0)])
    assert np.all(vals <= limits)

# Independently check the volume/boundary stencil on a rectangular cavity,
# then stress phase wrapping and heavy impulses at the highest-leverage sensors.
vals = []
for case in cases:
    p, g = case['problem'], case['gt']
    p = dict(p, Lx=1.19, Ly=0.83)
    F = solver._responses(p)
    n, m = p['n_sq'].shape
    hx, hy = p['Lx']/n, p['Ly']/m
    k, beta = p['k0'], p['beta_imp']
    hf = -hx*hy*k*k*p['n_sq'][...,None]*F
    dx = hy/hx*(F[1:]-F[:-1]); dy = hx/hy*(F[:,1:]-F[:,:-1])
    hf[1:] += dx; hf[:-1] -= dx; hf[:,1:] += dy; hf[:,:-1] -= dy
    hf[[0,-1]] -= 1j*k*beta*hy*F[[0,-1]]
    hf[:,[0,-1]] -= 1j*k*beta*hx*F[:,[0,-1]]
    rhs = hx*hy*np.moveaxis(p['basis'],0,-1)
    residual = np.linalg.norm(hf-rhs)/np.linalg.norm(rhs)
    assert residual < 1e-11, residual
    a = g['source_amplitudes']
    field = F @ a
    i,j = p['survey']['sensor_ij'].T
    A = F[i,j]
    X = np.vstack([A.real, A.imag]); Q = np.linalg.qr(X,mode='reduced')[0]
    lev = (Q**2).sum(1).reshape(2,-1).sum(0)
    hit = np.argsort(lev)[-10:]
    scale = np.median(abs(field[i,j]))
    pilot = p['pilot_calibration']
    ref = pilot['known_field_re']+1j*pilot['known_field_im']
    for phase in (np.pi, -np.pi+1e-6):
        rot = np.exp(1j*phase)
        for seed in range(12):
            rng=np.random.default_rng(seed)
            z=rot*(field[i,j] + scale*0.02*(rng.normal(size=len(i))+1j*rng.normal(size=len(i))))
            z[hit] += scale*8*np.exp(1j*rng.uniform(-np.pi,np.pi,len(hit)))
            pp=dict(p, pilot_calibration=dict(pilot, amplitude_obs=abs(ref), wrapped_phase_obs=np.angle(rot*ref)), survey=dict(p['survey'], amplitude_obs=abs(z), wrapped_phase_obs=np.angle(z)))
            vals.append(validate(pp,dict(source_amplitudes=a,global_phase_rad=phase,predicted_field_abs=abs(field))))
vals = np.asarray(vals)
print('rectangular/branch/high-leverage stress',len(vals),'max source/phase/field:',vals.max(0))
assert np.all(vals <= limits)
PY
```
- **Container Exit Code**: `0` (`2537 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
survey jackknife 280 max source/phase/field: [0.00542723 0.00062272 0.00359968] worst indices: [(1, 120), (0, 0), (1, 27)]
pilot_calibration jackknife 24 max source/phase/field: [0.00473207 0.00074135 0.00321022] worst indices: [(1, 0), (0, 10), (1, 0)]
rectangular/branch/high-leverage stress 48 max source/phase/field: [1.00658383e-02 4.44089210e-16 4.99580907e-03]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7630227135ac9a8c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76302271 (directory).
```

### Turn 4 — `submit_completed_task` (`13284 ms` LLM)
- **Rationale**: The exact artifact passed isolated Python verification. The independent rectangular stencil check, public jackknifes, and phase-wrap and high-leverage impulse stress tests found no failure requiring an edit.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`24` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3476'}): Do NOT lock prematurely with `24` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across both public cases, receiver and pilot jackknifes, and 48 stress tests, worst errors were 1.01% for source amplitudes, 0.000742 rad for phase, and 0.50% for field magnitude. All were within tolerance.`

### Turn 5 — `execute_in_remote_docker` (`27568 ms` LLM)
- **Rationale**: Single-receiver deletion can miss dependence on a whole region. Deleting each quadrant tests that failure on real measurements; if a slice misses tolerance, its design condition number will help distinguish weak spatial coverage from outlier sensitivity before any edit.
- **Alternatives Weighed**: `Add nonlinear terms or stratify by batch` (rejected: The stated stencil is linear in the sources and already matches the public forward fields to numerical precision; no batch labels, relay states, or drift dynamics are specified.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('helmholtz_phase_inversion', '/holdout_task/helmholtz_phase_inversion.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
limits = np.array([0.03, 0.02, 0.03])
worst = np.zeros(3)
with open('public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
for cid, case in enumerate(cases):
    p, g = case['problem'], case['gt']
    s = p['survey']
    f = solver._responses(p)
    i, j = s['sensor_ij'].T
    A = f[i, j]
    X = np.r_[A.real, A.imag]
    q = p['pilot_calibration']
    ref = q['known_field_re'] + 1j*q['known_field_im']
    phase = np.angle(np.vdot(ref, solver._polar(q)))
    z = solver._polar(s) * np.exp(-1j * phase)
    plain = np.linalg.lstsq(X, np.r_[z.real, z.imag], rcond=None)[0]
    print('case', cid, 'survey metadata', sorted(s), 'plain source rel error', np.linalg.norm(plain-g['source_amplitudes'])/np.linalg.norm(g['source_amplitudes']))
    # Delete each spatial quadrant as a block, leaving actual measurement noise and spikes intact.
    groups = 2*(i >= p['n_sq'].shape[0]//2) + (j >= p['n_sq'].shape[1]//2)
    for region in range(4):
        keep = groups != region
        subset = dict(s, **{k: s[k][keep] for k in ('sensor_ij','amplitude_obs','wrapped_phase_obs')})
        y = solver.invert_helmholtz_sources_and_phase(dict(p, survey=subset))
        rel = lambda k: np.linalg.norm(y[k]-g[k])/np.linalg.norm(g[k])
        errors = np.array([rel('source_amplitudes'), abs(np.angle(np.exp(1j*(y['global_phase_rad']-g['global_phase_rad'])))), rel('predicted_field_abs')])
        worst = np.maximum(worst, errors)
        print('held-out quadrant', region, 'remaining', int(keep.sum()), 'condition', round(np.linalg.cond(np.r_[A[keep].real, A[keep].imag]), 3), 'errors', errors)
print('spatial block worst source/phase/field:', worst)
assert np.all(worst <= limits), (worst, limits)
PY
```
- **Container Exit Code**: `0` (`543 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case 0 survey metadata ['amplitude_obs', 'sensor_ij', 'tikhonov_alpha', 'wrapped_phase_obs'] plain source rel error 0.17181974316179813
held-out quadrant 0 remaining 104 condition 3.138 errors [0.00444805 0.00062272 0.00344887]
held-out quadrant 1 remaining 104 condition 3.511 errors [0.00275664 0.00062272 0.00182275]
held-out quadrant 2 remaining 109 condition 3.225 errors [0.00555353 0.00062272 0.00302461]
held-out quadrant 3 remaining 103 condition 3.3 errors [0.00349652 0.00062272 0.00287067]
case 1 survey metadata ['amplitude_obs', 'sensor_ij', 'tikhonov_alpha', 'wrapped_phase_obs'] plain source rel error 0.2726691200590029
held-out quadrant 0 remaining 98 condition 3.499 errors [0.0059749  0.0002669  0.00399188]
held-out quadrant 1 remaining 104 condition 3.416 errors [0.00602678 0.0002669  0.0039954 ]
held-out quadrant 2 remaining 109 condition 3.312 errors [0.00484482 0.0002669  0.00333085]
held-out quadrant 3 remaining 109 condition 3.679 errors [0.00628773 0.0002669  0.00384921]
spatial block worst source/phase/field: [0.00628773 0.00062272 0.0039954 ]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7630227135ac9a8c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76302271 (directory).
```

### Turn 6 — `submit_completed_task` (`15525 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification and the independent stencil check. Removing whole receiver quadrants kept the worst source error below 0.63% on real measurements, so the spatial audit found no failure requiring a change.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`22` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3476'}): Do NOT lock prematurely with `22` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public cases, receiver and pilot jackknifes, spatial holdouts, and phase-wrap and impulse stress tests, the worst errors were 1.01% for source amplitudes, 0.000742 rad for phase, and 0.50% for field magnitude. All were within tolerance.`

### Turn 7 — `execute_in_remote_docker` (`27982 ms` LLM)
- **Rationale**: Quadrant holdouts retain a mix of wall and interior receivers. Fitting each group separately tests whether that mix masks a dependence on receiver location. If a split misses tolerance, check source observability in that split before changing the robust fit.
- **Alternatives Weighed**: `Add batch centering or nonlinear terms` (rejected: The fixtures have no batch labels, and the specified linear stencil already reproduces the public reference fields to numerical precision.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/helmholtz_phase_inversion.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

def rel(x, y):
    return np.linalg.norm(x-y)/np.linalg.norm(y)

worst = np.zeros(4)
for case_id, case in enumerate(cases):
    p, g = case['problem'], case['gt']
    s = p['survey']
    i, j = s['sensor_ij'].T
    nx, ny = p['n_sq'].shape
    near_wall = np.minimum.reduce([i, nx-1-i, j, ny-1-j]) < 5
    for name, keep in [('wall-only', near_wall), ('interior-only', ~near_wall)]:
        subset = dict(s, **{k: s[k][keep] for k in ('sensor_ij', 'amplitude_obs', 'wrapped_phase_obs')})
        y = solver.invert_helmholtz_sources_and_phase(dict(p, survey=subset))
        field = y['predicted_field_abs']
        phase = y['global_phase_rad']
        errors = np.array([
            rel(y['source_amplitudes'], g['source_amplitudes']),
            abs(np.angle(np.exp(1j*(phase-g['global_phase_rad'])))),
            rel(field, g['predicted_field_abs']),
            rel(field[i[~keep],j[~keep]], g['predicted_field_abs'][i[~keep],j[~keep]])])
        worst = np.maximum(worst, errors)
        print(case_id, name, 'fit receivers:', int(keep.sum()), 'source/phase/full field/held-out field:', errors)
print('worst:', worst)
assert np.all(worst <= [0.03, 0.02, 0.03, 0.03]), worst
PY
```
- **Container Exit Code**: `0` (`559 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 wall-only fit receivers: 76 source/phase/full field/held-out field: [0.0094126  0.00062272 0.00678721 0.00716421]
0 interior-only fit receivers: 64 source/phase/full field/held-out field: [0.00536261 0.00062272 0.00359442 0.00325194]
1 wall-only fit receivers: 75 source/phase/full field/held-out field: [0.00296526 0.0002669  0.00247877 0.00251511]
1 interior-only fit receivers: 65 source/phase/full field/held-out field: [0.00581712 0.0002669  0.00406947 0.00350822]
worst: [0.0094126  0.00062272 0.00678721 0.00716421]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7630227135ac9a8c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76302271 (directory).
```

### Turn 8 — `submit_completed_task` (`14243 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification, an independent rectangular stencil check, and phase-wrap and impulse stress tests. Wall-only and interior-only fits also predicted the excluded receivers within tolerance, so the latest audit found no reason to change the solver.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public cases, receiver and pilot holdouts, and stress tests, the worst errors were 1.01% for source amplitudes, 0.000742 rad for phase, and 0.717% for field magnitude, including predictions at held-out receivers. All met the requested tolerances.`

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
