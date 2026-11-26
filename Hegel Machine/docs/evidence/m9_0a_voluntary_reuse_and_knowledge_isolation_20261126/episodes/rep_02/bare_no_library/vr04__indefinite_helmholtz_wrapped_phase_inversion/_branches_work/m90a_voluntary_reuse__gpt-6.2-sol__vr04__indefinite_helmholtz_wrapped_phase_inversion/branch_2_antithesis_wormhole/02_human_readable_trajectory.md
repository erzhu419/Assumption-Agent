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
| **Total Tokens (Prompt / Cached / Out)** | `160,140` (`153,274` / `48,640` / `6,866`) |
| **Total Episode Wall Time (s)** | `226.65s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`15263 ms` LLM)
- **Rationale**: Inspect keys, shapes, reference outputs, and SciPy availability before choosing the implementation and validation harness. This detects schema or dependency mismatches without transferring the data.

```bash
cd /holdout_task && ls -lh && python3 - <<'PY'
import pickle, numpy as np, scipy
with open('public_helmholtz_cases.pkl','rb') as f: data=pickle.load(f)
print('SciPy',scipy.__version__)
def describe(x,d=0):
    pad='  '*d
    if isinstance(x,dict):
        for k,v in x.items():
            print(pad, k, end=': ')
            describe(v,d+1)
    elif isinstance(x,(list,tuple)):
        print(type(x).__name__,len(x))
        if x and d<5: describe(x[0],d+1)
    elif isinstance(x,np.ndarray): print(x.shape,x.dtype)
    else: print(repr(x)[:120])
describe(data)
PY
```
- **Container Exit Code**: `0` (`580 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 140K
-rw-r--r-- 1 root root 139K Nov 26 00:46 public_helmholtz_cases.pkl
SciPy 1.13.0
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

### Turn 2 — `execute_in_remote_docker` (`50949 ms` LLM)
- **Rationale**: Check the stencil separately against the supplied true-source field, then compare robust and ordinary regression and leave each receiver out. This distinguishes forward-model errors from spike sensitivity and regularization bias.
- **Alternatives Weighed**: `Unwrap receiver angles before inversion` (rejected: Rebuilding the complex phasors preserves the telemetry without introducing branch-dependent residuals.)

```bash
cd /holdout_task && cat > helmholtz_phase_inversion.py <<'PY'
"""Finite-volume Helmholtz inversion with calibrated complex receivers."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _polar(channel):
    return np.asarray(channel['amplitude_obs']) * np.exp(1j * np.asarray(channel['wrapped_phase_obs']))


def _phase(pilot):
    reference = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(reference, _polar(pilot))))
    return np.pi if phase == -np.pi else phase


def _path_laplacian(n):
    off = -np.ones(n - 1)
    degree = np.zeros(n)
    degree[:-1] += 1
    degree[1:] += 1
    return sparse.diags((off, degree, off), (-1, 0, 1), shape=(n, n))


def _response(problem):
    n_sq = np.asarray(problem['n_sq'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k = problem['k0']
    face_length = np.zeros_like(n_sq, dtype=float)
    face_length[0, :] += hy
    face_length[-1, :] += hy
    face_length[:, 0] += hx
    face_length[:, -1] += hx
    diagonal = -area * k**2 * n_sq - 1j * k * problem['beta_imp'] * face_length
    matrix = (hy / hx * sparse.kron(_path_laplacian(nx), sparse.eye(ny))
              + hx / hy * sparse.kron(sparse.eye(nx), _path_laplacian(ny))
              + sparse.diags(diagonal.ravel())).tocsc()
    basis = np.asarray(problem['basis'])
    rhs = np.asfortranarray(area * basis.reshape(basis.shape[0], nx * ny).T, dtype=np.complex128)
    return splu(matrix).solve(rhs).reshape(nx, ny, basis.shape[0])


def _robust_fit(design, observed, alpha):
    """Fit real coefficients; downweight the two quadratures jointly."""
    n, m = design.shape
    x = np.concatenate((design.real, design.imag), axis=0)
    y = np.concatenate((observed.real, observed.imag))
    reg = np.sqrt(alpha) * np.eye(m)

    def solve(weights, penalty=None):
        root = np.sqrt(np.tile(weights, 2))
        a, b = x * root[:, None], y * root
        if penalty is not None:
            a = np.vstack((a, penalty))
            b = np.concatenate((b, np.zeros(m)))
        return np.linalg.lstsq(a, b, rcond=None)[0]

    amplitudes = solve(np.ones(n))
    floor = max(np.median(np.abs(observed)) * 1e-12, np.finfo(float).tiny)
    # An L1-like start keeps large spikes from setting the final residual scale.
    for _ in range(25):
        radius = np.abs(design @ amplitudes - observed)
        scale = max(np.median(radius), floor)
        updated = solve(scale / np.maximum(radius, 0.15 * scale))
        if np.linalg.norm(updated - amplitudes) <= 1e-9 * max(np.linalg.norm(amplitudes), floor):
            amplitudes = updated
            break
        amplitudes = updated
    # Fix the scale before the ridge step so shrinkage cannot inflate the cutoff.
    cutoff = 4.0 * max(np.median(np.abs(design @ amplitudes - observed)), floor)
    radius = np.abs(design @ amplitudes - observed)
    weights = np.maximum(1.0 - (np.minimum(radius, cutoff) / cutoff)**2, 0.0)**2
    return solve(weights, reg)


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    phase = _phase(problem['pilot_calibration'])
    response = _response(problem)
    survey = problem['survey']
    ij = np.asarray(survey['sensor_ij'])
    design = response[ij[:, 0], ij[:, 1], :]
    observed = _polar(survey) * np.exp(-1j * phase)
    amplitudes = _robust_fit(design, observed, survey['tikhonov_alpha'])
    return {
        'source_amplitudes': amplitudes,
        'global_phase_rad': phase,
        'predicted_field_abs': np.abs(response @ amplitudes),
    }
PY
cat > /tmp/helmholtz_public.py <<'PY'
import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/helmholtz_phase_inversion.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
def rel(x,y): return np.linalg.norm(x-y)/np.linalg.norm(y)
for i,c in enumerate(cases):
    p,g = c['problem'],c['gt']
    out = s.invert_helmholtz_sources_and_phase(p)
    a,u = g['source_amplitudes'],g['predicted_field_abs']
    survey = p['survey']; ij = survey['sensor_ij']; response = s._response(p)
    D = response[ij[:,0],ij[:,1],:]; z = s._polar(survey)*np.exp(-1j*out['global_phase_rad']); alpha = survey['tikhonov_alpha']
    err = [rel(out['source_amplitudes'],a), abs(np.angle(np.exp(1j*(out['global_phase_rad']-g['global_phase_rad'])))), rel(out['predicted_field_abs'],u)]
    truth_resid = np.abs(z-D@a)
    jack = []
    for j in range(len(z)):
        keep = np.arange(len(z)) != j
        est = s._robust_fit(D[keep],z[keep],alpha)
        jack.append((rel(est,a),rel(np.abs(response@est),u),abs(D[j]@(est-a))))
    regless = s._robust_fit(D,z,0.)
    ordinary = np.linalg.lstsq(np.vstack([D.real,D.imag]),np.r_[z.real,z.imag],rcond=None)[0]
    print(i, 'source/phase/field',err,'LOSO max source/field/clean sensor abs',np.max(jack,0),'LS source',rel(ordinary,a),'unregularized robust source',rel(regless,a))
    print('  clean forward error',rel(np.abs(response@a),u),'signal median',np.median(np.abs(D@a)),'observed residual quantiles',np.quantile(truth_resid,[0,.5,.9,.93,.95,1]),'singular values',np.linalg.svd(np.vstack([D.real,D.imag]),compute_uv=False))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/helmholtz_public.py
```
- **Container Exit Code**: `0` (`1045 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
0 source/phase/field [0.002938992806863936, 0.0006227225657142733, 0.002176953108187411] LOSO max source/field/clean sensor abs [0.00363739 0.00245439 0.00056146] LS source 0.17181974316179813 unregularized robust source 0.002943343985898905
  clean forward error 1.2624230001329256e-15 signal median 0.06468470700735354 observed residual quantiles [3.48695400e-05 2.02539994e-03 3.74544690e-03 1.00180299e-01
 3.72536435e-01 7.40698889e-01] singular values [0.8506539  0.68277291 0.64396571 0.48967366 0.39137982 0.28720722]
1 source/phase/field [0.005694445352195017, 0.00026689595102702407, 0.003674922068473124] LOSO max source/field/clean sensor abs [0.00631443 0.00414603 0.00121338] LS source 0.2726691200590029 unregularized robust source 0.005694461626941781
  clean forward error 8.893619912307587e-16 signal median 0.0859848799274255 observed residual quantiles [2.17014084e-04 1.81359429e-03 4.27393929e-03 9.99116249e-02
 4.09010454e-01 6.86037203e-01] singular values [0.88022524 0.69563054 0.63706238 0.44340971 0.39510715 0.28270252]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6e7cca5f76871179 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6e7cca5f (directory).
```

### Turn 3 — `execute_in_remote_docker` (`32348 ms` LLM)
- **Rationale**: Public errors pass with margin, but dropping one receiver does not test spikes concentrated at influential receivers. Reassign observed spikes to those receivers and move the oscillator phase near both sides of the wrap boundary. A source or field failure would call for revising the robust initializer; a phase failure would call for revising the circular calibration.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/helmholtz_phase_inversion.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

def score(out, gt):
    assert set(out) == {'source_amplitudes', 'global_phase_rad', 'predicted_field_abs'}
    phase = out['global_phase_rad']
    assert isinstance(phase, float) and -np.pi < phase <= np.pi
    for key in ('source_amplitudes', 'predicted_field_abs'):
        assert out[key].shape == gt[key].shape and out[key].dtype.kind == 'f'
    return np.array([
        np.linalg.norm(out['source_amplitudes'] - gt['source_amplitudes']) / np.linalg.norm(gt['source_amplitudes']),
        abs(np.angle(np.exp(1j * (phase - gt['global_phase_rad'])))),
        np.linalg.norm(out['predicted_field_abs'] - gt['predicted_field_abs']) / np.linalg.norm(gt['predicted_field_abs']),
    ])

def polar_fields(z):
    return {'amplitude_obs': np.abs(z), 'wrapped_phase_obs': np.angle(z)}

limits = np.array([.03, .02, .03])
public, stressed = [], []
for case in cases:
    p, gt = case['problem'], case['gt']
    public.append(score(solver.invert_helmholtz_sources_and_phase(p), gt))
    survey, pilot = p['survey'], p['pilot_calibration']
    ij = survey['sensor_ij']
    design = solver._response(p)[ij[:, 0], ij[:, 1]]
    clean = design @ gt['source_amplitudes']
    residual = solver._polar(survey) * np.exp(-1j * gt['global_phase_rad']) - clean
    count = round(.07 * len(clean))
    order = np.argsort(np.abs(residual))
    spikes = residual[order[-count:]]
    inliers = residual[order[:-count]]
    # Group leverage of the real and imaginary rows of each receiver.
    q, _ = np.linalg.qr(np.vstack((design.real, design.imag)))
    lev = (q[:len(ij)]**2 + q[len(ij):]**2).sum(axis=1)
    high = np.argsort(lev)[-count:]
    keep = np.ones(len(clean), dtype=bool)
    keep[high] = False
    cal = solver._polar(pilot) * np.exp(-1j * gt['global_phase_rad'])
    rng = np.random.default_rng(901)
    for phase in (-np.pi + 1e-6, np.pi - 1e-6):
        for trial in range(24):
            noise = np.empty_like(residual)
            noise[keep] = rng.permutation(inliers)
            noise[high] = rng.permutation(spikes)
            rot = np.exp(1j * phase)
            modified = dict(p, pilot_calibration=dict(pilot, **polar_fields(rot * cal)),
                            survey=dict(survey, **polar_fields(rot * (clean + noise))))
            stressed.append(score(solver.invert_helmholtz_sources_and_phase(modified), dict(gt, global_phase_rad=phase)))
for name, values in [('public', public), ('high-leverage spikes / phase branch', stressed)]:
    worst = np.max(values, axis=0)
    print(name, 'n=', len(values), 'worst source/phase/field:', worst, flush=True)
    assert np.all(worst <= limits), (name, worst, limits)
PY
```
- **Container Exit Code**: `0` (`1143 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public n= 2 worst source/phase/field: [0.00569445 0.00062272 0.00367492]
high-leverage spikes / phase branch n= 96 worst source/phase/field: [0.00854462 0.00062272 0.00534957]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6e7cca5f76871179 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6e7cca5f (directory).
```

### Turn 4 — `submit_completed_task` (`13202 ms` LLM)
- **Rationale**: The forward stencil reproduced the public reference fields to floating-point precision. The unchanged artifact passed isolated-process schema and tolerance checks, including spikes concentrated at high-leverage receivers and oscillator phases on both sides of the wrap boundary.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`24` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3562'}): Do NOT lock prematurely with `24` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public cases and 96 stress trials, worst errors were 0.855% for source amplitudes, 0.000623 rad for phase, and 0.535% for field magnitude. All three met the requested tolerances.`

### Turn 5 — `execute_in_remote_docker` (`34323 ms` LLM)
- **Rationale**: Test face signs, axis weights, and corner accumulation independently; a mismatch would require a stencil correction. Holding out entire receiver quadrants while omitting each pilot tests whether the reported margins depend on a single spatial region or calibration channel.
- **Alternatives Weighed**: `Batch centering or nonlinear PDE terms` (rejected: The supplied schema has no cohort or class labels, and the specified linear stencil and common complex rotation already explain the forward model.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/helmholtz_phase_inversion.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

def rel(x, y):
    return np.linalg.norm(x-y)/np.linalg.norm(y)

def metrics(out, g):
    phase = out['global_phase_rad']
    assert isinstance(phase, float) and -np.pi < phase <= np.pi
    for key in ('source_amplitudes', 'predicted_field_abs'):
        assert out[key].shape == g[key].shape and out[key].dtype.kind == 'f'
    return [rel(out['source_amplitudes'],g['source_amplitudes']),
            abs(np.angle(np.exp(1j*(phase-g['global_phase_rad'])))),
            rel(out['predicted_field_abs'],g['predicted_field_abs'])]

rows, flux_checks = [], []
for case in cases:
    p, g = case['problem'], case['gt']
    survey, pilot = p['survey'], p['pilot_calibration']
    ij = survey['sensor_ij']
    nx, ny = p['n_sq'].shape
    # Apply face fluxes independently on square and rectangular domains.
    for stretch in (1., 1.25):
        trial = dict(p, Lx=p['Lx']*stretch)
        u = s._response(trial)
        hx, hy = trial['Lx']/nx, trial['Ly']/ny
        k = trial['k0']
        applied = -(hx*hy*k*k)*trial['n_sq'][...,None]*u
        for axis, weight in ((0,hy/hx),(1,hx/hy)):
            v, av = np.moveaxis(u,axis,0), np.moveaxis(applied,axis,0)
            d = weight*(v[:-1]-v[1:])
            av[:-1] += d
            av[1:] -= d
            face = hy if axis == 0 else hx
            av[[0,-1]] -= 1j*k*trial['beta_imp']*face*v[[0,-1]]
        flux_checks.append(rel(applied,hx*hy*np.moveaxis(trial['basis'],0,-1)))
    groups = 2*(ij[:,0] >= nx//2) + (ij[:,1] >= ny//2)
    response = s._response(p)
    true_receiver = response[ij[:,0],ij[:,1]] @ g['source_amplitudes']
    for q in np.unique(groups):
        keep = groups != q
        kept_survey = dict(survey, **{key:survey[key][keep] for key in ('sensor_ij','amplitude_obs','wrapped_phase_obs')})
        for j in range(len(pilot['known_field_re'])):
            kept_pilot = {key:np.delete(value,j) for key,value in pilot.items()}
            out = s.invert_helmholtz_sources_and_phase(dict(p,survey=kept_survey,pilot_calibration=kept_pilot))
            held = response[ij[~keep,0],ij[~keep,1]] @ out['source_amplitudes']
            rows.append(metrics(out,g) + [rel(held,true_receiver[~keep])])
worst = np.max(rows,axis=0)
print('Independent face-flux residual, worst:', max(flux_checks), flush=True)
print('Quadrant holdout x pilot omission:', len(rows), 'fits; worst source/phase/field/held clean complex:', worst, flush=True)
assert max(flux_checks) < 1e-11
assert np.all(worst[:3] <= [.03,.02,.03]), worst
PY
```
- **Container Exit Code**: `0` (`997 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Independent face-flux residual, worst: 1.1610885287888414e-14
Quadrant holdout x pilot omission: 96 fits; worst source/phase/field/held clean complex: [0.00728528 0.00074135 0.00451466 0.01047859]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6e7cca5f76871179 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6e7cca5f (directory).
```

### Turn 6 — `submit_completed_task` (`14404 ms` LLM)
- **Rationale**: The independent square and rectangular face-flux audit had a worst relative residual of 1.17e-14. The unchanged artifact passed isolated-process checks with high-leverage spikes, phases near the wrap boundary, and combined receiver-quadrant and pilot omissions. No tested slice exposed a bottleneck that warranted a change.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`22` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3562'}): Do NOT lock prematurely with `22` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public cases and stress tests, worst errors were 0.855% for source amplitudes, 0.000742 rad for phase, and 0.535% for field magnitude. All met the requested tolerances.`

### Turn 7 — `execute_in_remote_docker` (`27088 ms` LLM)
- **Rationale**: Earlier audits tested missing regions and spikes at high-leverage receivers separately. Sweep a spatial cluster of measured spikes across the receiver grid to detect bias from localized contamination; a tolerance failure would warrant revisiting the robust fit, not the verified stencil.
- **Alternatives Weighed**: `Add batch centering or nonlinear terms` (rejected: There are no batch or class labels, and the independent face-flux audit already validates the specified linear operator.)

```bash
cat > /tmp/helmholtz_cluster_audit.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/helmholtz_phase_inversion.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

scores = []
for case in cases:
    p, gt = case['problem'], case['gt']
    sv = p['survey']
    ij = np.asarray(sv['sensor_ij'])
    phase = gt['global_phase_rad']
    rot = np.exp(1j * phase)
    a, u = gt['source_amplitudes'], gt['predicted_field_abs']
    clean = s._response(p)[ij[:, 0], ij[:, 1]] @ a
    residual = s._polar(sv) / rot - clean
    n = len(ij)
    count = round(.07 * n)
    order = np.argsort(np.abs(residual))
    spike_idx = order[-count:]
    inlier_idx = order[:-count]
    for anchor in np.unique(ij, axis=0):
        near = np.argsort(np.sum((ij - anchor)**2, axis=1), kind='stable')[:count]
        # Swap measured residuals to cluster the spikes without altering their distribution.
        moved = residual.copy()
        src = np.setdiff1d(spike_idx, near)
        dst = np.intersect1d(near, inlier_idx)
        moved[dst], moved[src] = moved[src], moved[dst]
        z = rot * (clean + moved)
        out = s.invert_helmholtz_sources_and_phase(dict(p, survey=dict(sv, amplitude_obs=np.abs(z), wrapped_phase_obs=np.angle(z))))
        scores.append((np.linalg.norm(out['source_amplitudes'] - a) / np.linalg.norm(a),
                       abs(np.angle(np.exp(1j * (out['global_phase_rad'] - phase)))),
                       np.linalg.norm(out['predicted_field_abs'] - u) / np.linalg.norm(u)))
worst = np.max(scores, axis=0)
print('Measured-spike spatial clusters:', len(scores), 'worst source/phase/field:', worst, flush=True)
assert np.all(worst <= [.03, .02, .03]), worst
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/helmholtz_cluster_audit.py
```
- **Container Exit Code**: `0` (`2297 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Measured-spike spatial clusters: 280 worst source/phase/field: [0.00697669 0.00062272 0.00464293]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6e7cca5f76871179 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6e7cca5f (directory).
```

### Turn 8 — `submit_completed_task` (`11825 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process verification. The 280 measured-spike cluster trials stayed within the earlier stress-test maxima, so localized contamination did not expose a bottleneck that warranted changing the solver.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public cases and stress tests, the worst errors were 0.855% for source amplitudes, 0.000742 rad for global phase, and 0.535% for field magnitude. All met the requested tolerances.`

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
