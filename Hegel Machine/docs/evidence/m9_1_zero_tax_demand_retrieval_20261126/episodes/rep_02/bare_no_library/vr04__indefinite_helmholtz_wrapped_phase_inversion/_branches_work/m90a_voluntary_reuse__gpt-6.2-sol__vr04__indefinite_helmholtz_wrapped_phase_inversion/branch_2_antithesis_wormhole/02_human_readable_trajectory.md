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
| **Turns Used / Max Turns** | `7 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `142,201` (`135,362` / `33,920` / `6,839`) |
| **Total Episode Wall Time (s)** | `220.59s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`16591 ms` LLM)
- **Rationale**: Inspect the public schema and installed numerical libraries before choosing an implementation. This catches input-key mismatches and establishes whether labeled worst-case validation and sparse complex LU are available.
- **Alternatives Weighed**: `Unwrap and fit receiver phase directly` (rejected: Complex reconstruction removes wrapping exactly and keeps amplitude information for identifying corrupted receiver pairs.)

```bash
ls -la /holdout_task; python3 - <<'PY'
import pickle, importlib.util
p='/holdout_task/public_helmholtz_cases.pkl'
with open(p,'rb') as f: x=pickle.load(f)
def show(x,d=0):
 if d>5: return
 if isinstance(x,dict):
  print(' '*d,'dict',list(x));
  for k,v in x.items():
   print(' '*(d+1),k,end=': '); show(v,d+2)
 elif isinstance(x,(tuple,list)):
  print(type(x).__name__,len(x));
  if x: show(x[0],d+1)
 elif hasattr(x,'shape'): print(x.shape,x.dtype)
 else: print(repr(x)[:160])
show(x)
for m in ['numpy','scipy']: print(m, bool(importlib.util.find_spec(m)))
PY
```
- **Container Exit Code**: `0` (`613 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 148
drwxr-xr-x 2 root root   4096 Nov 26 01:38 .
drwxr-xr-x 1 root root   4096 Nov 26 01:38 ..
-rw-r--r-- 1 root root 142215 Nov 26 01:38 public_helmholtz_cases.pkl
list 2
  dict ['problem', 'gt']
   problem:     dict ['Lx', 'Ly', 'k0', 'beta_imp', 'n_sq', 'basis', 'pilot_calibration', 'survey']
     Lx: 1.0
     Ly: 1.0
     k0: 11.5
     beta_imp: 0.85
     n_sq: (32, 32) float64
     basis: (6, 32, 32) float64
     pilot_calibration:       dict ['known_field_re', 'known_field_im', 'amplitude_obs', 'wrapped_phase_obs']
       known_field_re:        known_field_im:        amplitude_obs:        wrapped_phase_obs:      survey:       dict ['sensor_ij', 'amplitude_obs', 'wrapped_phase_obs', 'tikhonov_alpha']
       sensor_ij:        amplitude_obs:        wrapped_phase_obs:        tikhonov_alpha:    gt:     dict ['source_amplitudes', 'global_phase_rad', 'predicted_field_abs']
     source_amplitudes: (6,) float64
     global_phase_rad: -0.8141122105046681
     predicted_field_abs: (32, 32) float64
numpy True
scipy True
```

### Turn 2 — `execute_in_remote_docker` (`50883 ms` LLM)
- **Rationale**: The forward-only field error isolates stencil and indexing defects, while public source and phase errors test the inversion independently. Residual quantiles and the supplied ridge strength will show whether receiver rejection or regularization needs adjustment before stress testing.
- **Alternatives Weighed**: `Ordinary stacked-real least squares` (rejected: It gives impulse-corrupted receivers unbounded influence; retain it as a comparison.)

```bash
cat > /holdout_task/helmholtz_phase_inversion.py <<'PY'
"""Finite-volume Helmholtz inversion from polar receiver measurements."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _responses(problem):
    """Return the complex pressure response to each unit source."""
    index_sq = np.asarray(problem['n_sq'])
    basis = np.asarray(problem['basis'])
    nx, ny = index_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k0 = problem['k0']
    ids = np.arange(nx * ny).reshape(nx, ny)
    cells = ids.ravel()
    left = np.concatenate((ids[:-1, :].ravel(), ids[:, :-1].ravel()))
    right = np.concatenate((ids[1:, :].ravel(), ids[:, 1:].ravel()))
    conductance = np.concatenate((np.full((nx - 1) * ny, hy / hx),
                                  np.full(nx * (ny - 1), hx / hy)))
    diagonal = np.array(-area * k0**2 * index_sq, dtype=np.complex128)
    flux = -1j * k0 * problem['beta_imp']
    diagonal[0, :] += hy * flux
    diagonal[-1, :] += hy * flux
    diagonal[:, 0] += hx * flux
    diagonal[:, -1] += hx * flux
    diagonal = diagonal.ravel()
    np.add.at(diagonal, left, conductance)
    np.add.at(diagonal, right, conductance)
    matrix = coo_matrix((np.concatenate((diagonal, -conductance, -conductance)),
                         (np.concatenate((cells, left, right)),
                          np.concatenate((cells, right, left)))),
                        shape=(cells.size, cells.size)).tocsc()
    rhs = np.asarray(area * basis.reshape(basis.shape[0], -1).T,
                     dtype=np.complex128)
    return splu(matrix).solve(rhs).reshape(nx, ny, basis.shape[0])


def _polar(data):
    return np.asarray(data['amplitude_obs']) * np.exp(
        1j * np.asarray(data['wrapped_phase_obs']))


def _fit(operator, target, alpha):
    """Weight each receiver as a pair of real and imaginary residuals."""
    n, m = operator.shape
    matrix = np.vstack((operator.real, operator.imag, np.sqrt(alpha) * np.eye(m)))
    observation = np.concatenate((target.real, target.imag, np.zeros(m)))

    def solve(weights):
        s = np.sqrt(weights)
        w = np.concatenate((s, s, np.ones(m)))
        return np.linalg.lstsq(matrix * w[:, None], observation * w, rcond=None)[0]

    a = solve(np.ones(n))
    floor = 32 * np.finfo(float).eps * np.max(np.abs(target))
    # Start with bounded-influence weights, then remove gross impulses entirely.
    for cutoff, hard_rejection in ((1.5, False), (5.0, True)):
        for _ in range(25):
            r = np.abs(operator @ a - target)
            scale = max(np.median(r) / np.sqrt(2.0 * np.log(2.0)), floor)
            if scale == 0:
                return a
            if hard_rejection:
                ratio = np.minimum(r / (cutoff * scale), 1.0)
                weights = (1.0 - ratio**2)**2
            else:
                weights = cutoff * scale / np.maximum(r, cutoff * scale)
            next_a = solve(weights)
            done = np.linalg.norm(next_a - a) <= 1e-10 * np.linalg.norm(a)
            a = next_a
            if done:
                break
    return a


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Recover the real source amplitudes, common receiver phase, and field modulus."""
    pilot = problem['pilot_calibration']
    reference = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(reference, _polar(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)
    responses = _responses(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    target = np.exp(-1j * phase) * _polar(survey)
    a = _fit(responses[i, j], target, survey['tikhonov_alpha'])
    return {'source_amplitudes': a, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(responses @ a)}
PY
python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
path = '/holdout_task/helmholtz_phase_inversion.py'
spec = importlib.util.spec_from_file_location('helmholtz_phase_inversion', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
for c, case in enumerate(cases):
    p, gt = case['problem'], case['gt']; s = p['survey']
    result = m.invert_helmholtz_sources_and_phase(p)
    responses = m._responses(p); i,j = np.asarray(s['sensor_ij']).T
    X = responses[i,j]; truth = gt['source_amplitudes']; F = gt['predicted_field_abs']
    d = np.exp(-1j*gt['global_phase_rad'])*m._polar(s)
    err = lambda a,b: np.linalg.norm(a-b)/np.linalg.norm(b)
    ls = np.linalg.lstsq(np.vstack((X.real,X.imag)), np.r_[d.real,d.imag],rcond=None)[0]
    print('case',c,'sensors',len(i),'alpha',s['tikhonov_alpha'],'cond',np.linalg.cond(np.vstack((X.real,X.imag))))
    print('source, circular phase, field',err(result['source_amplitudes'],truth),abs(np.angle(np.exp(1j*(result['global_phase_rad']-gt['global_phase_rad'])))),err(result['predicted_field_abs'],F))
    print('ordinary LS source / forward-only field error',err(ls,truth),err(np.abs(responses@truth),F))
    print('signal RMS',np.linalg.norm(X@truth)/np.sqrt(len(i)),'truth residual quantiles',np.quantile(np.abs(d-X@truth),[.0,.5,.85,.91,.94,.97,1]))
PY
```
- **Container Exit Code**: `0` (`836 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
case 0 sensors 140 alpha 1e-06 cond 2.9618123405365124
source, circular phase, field 0.003112524911603442 0.0006227225657142733 0.0021779383690295113
ordinary LS source / forward-only field error 0.17180313491718777 1.1272979772244929e-15
signal RMS 0.08224338994836831 truth residual quantiles [6.47881901e-06 2.04397750e-03 2.99061952e-03 3.95592022e-03
 3.61614744e-01 4.64060703e-01 7.40687115e-01]
case 1 sensors 140 alpha 1e-06 cond 3.1136094055719403
source, circular phase, field 0.005678418539380476 0.00026689595102702407 0.003635264250011381
ordinary LS source / forward-only field error 0.2727211285456089 9.93191916978576e-16
signal RMS 0.10347423196268851 truth residual quantiles [2.09377251e-04 1.80389227e-03 3.47499659e-03 4.54329085e-03
 3.74277075e-01 5.28755716e-01 6.86029873e-01]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=92e5701a67ae1517 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_92e5701a (directory).
```

### Turn 3 — `execute_in_remote_docker` (`48531 ms` LLM)
- **Rationale**: Public forward errors near machine precision support the stencil, and ordinary least squares misses the source tolerance on both cases. The isolated verification tests failures that would change the implementation: wall scaling on rectangular grids, sensitivity to any one public receiver or pilot, phase branch handling, and rejection of clustered or high-response impulses.
- **Alternatives Weighed**: `Retune the robust estimator on the two public cases` (rejected: The current margins are substantial, and tuning on two cases would not establish robustness to receiver placement or corruption patterns.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

path = '/holdout_task/helmholtz_phase_inversion.py'
spec = importlib.util.spec_from_file_location('helmholtz_phase_inversion', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

rng = np.random.default_rng(71394)
limits = np.array([.03, .02, .03])
def score(p, a, phase, field):
    out = mod.invert_helmholtz_sources_and_phase(p)
    aa, pp, ff = (out[k] for k in ('source_amplitudes', 'global_phase_rad', 'predicted_field_abs'))
    assert aa.shape == a.shape and ff.shape == field.shape
    assert aa.dtype.kind == ff.dtype.kind == 'f' and isinstance(pp, float)
    assert -np.pi < pp <= np.pi and np.all(ff >= 0)
    return np.array([np.linalg.norm(aa-a)/np.linalg.norm(a), abs(np.angle(np.exp(1j*(pp-phase)))), np.linalg.norm(ff-field)/np.linalg.norm(field)])

def polar(z):
    return dict(amplitude_obs=np.abs(z), wrapped_phase_obs=np.angle(z))

def stencil(p, u):
    nx, ny = u.shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    v = -hx*hy*p['k0']**2*p['n_sq']*u
    dx = hy/hx*(u[1:]-u[:-1]); v[1:] += dx; v[:-1] -= dx
    dy = hx/hy*(u[:,1:]-u[:,:-1]); v[:,1:] += dy; v[:,:-1] -= dy
    b = -1j*p['k0']*p['beta_imp']
    v[[0,-1],:] += hy*b*u[[0,-1],:]
    v[:,[0,-1]] += hx*b*u[:,[0,-1]]
    return v

results = {}
def record(group, tag, value):
    results.setdefault(group, []).append((tag, value))

for c, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    a, phase, field = (gt[k] for k in ('source_amplitudes', 'global_phase_rad', 'predicted_field_abs'))
    survey = p['survey']
    record('public', c, score(p,a,phase,field))
    for sensor in range(len(survey['sensor_ij'])):
        mask = np.arange(len(survey['sensor_ij'])) != sensor
        sub = {k: v[mask] if k in ('sensor_ij','amplitude_obs','wrapped_phase_obs') else v for k,v in survey.items()}
        record('public_leave_one_survey_out', (c,sensor), score(dict(p,survey=sub),a,phase,field))
    pilot = p['pilot_calibration']
    for sensor in range(len(pilot['known_field_re'])):
        sub = {k: np.delete(v,sensor) for k,v in pilot.items()}
        record('public_leave_one_pilot_out', (c,sensor), score(dict(p,pilot_calibration=sub),a,phase,field))
    for geom, dims in enumerate(((p['Lx'],p['Ly'],p['k0']), (1.19,.86,12.2), (.86,1.19,10.8))):
        q = dict(p, Lx=dims[0], Ly=dims[1], k0=dims[2]); w = mod._responses(q)
        nx, ny = w.shape[:2]; area = q['Lx']*q['Ly']/(nx*ny)
        err = max(np.linalg.norm(stencil(q,w[:,:,j])-area*q['basis'][j])/np.linalg.norm(area*q['basis'][j]) for j in range(len(a)))
        assert err < 1e-10, (c,geom,err)
        design = w.reshape(nx*ny,-1)
        leverage = np.sum(np.abs(design)**2,axis=1)
        ref = pilot['known_field_re'] + 1j*pilot['known_field_im']
        for trial in range(40):
            mode = ['random','localized','high_response'][trial%3]
            aa = rng.normal(size=a.shape); aa *= np.linalg.norm(a)/np.linalg.norm(aa)
            pp = [np.pi, -np.pi+2e-4, rng.uniform(-np.pi,np.pi)][(trial//3)%3]
            field_complex = w@aa
            flat = rng.choice(nx*ny,size=len(survey['sensor_ij']),replace=False)
            ij = np.column_stack(np.unravel_index(flat,(nx,ny)))
            z = np.exp(1j*pp)*field_complex.ravel()[flat]
            z += .0015*(rng.normal(size=len(z))+1j*rng.normal(size=len(z)))
            count = round(.07*len(z))
            if mode == 'random':
                bad = rng.choice(len(z),size=count,replace=False)
            elif mode == 'localized':
                bad = np.argsort(np.sum((ij-ij[rng.integers(len(z))])**2,axis=1))[:count]
            else:
                bad = np.argsort(leverage[flat])[-count:]
            z[bad] += rng.uniform(.25,.75,size=count)*np.exp(1j*rng.uniform(-np.pi,np.pi,size=count))
            cal = dict(known_field_re=ref.real,known_field_im=ref.imag,**polar(np.exp(1j*pp)*ref))
            obs = dict(sensor_ij=ij,tikhonov_alpha=survey['tikhonov_alpha'],**polar(z))
            record('stress_'+mode,(c,geom,trial),score(dict(q,pilot_calibration=cal,survey=obs),aa,pp,np.abs(field_complex)))

failed = []
for group, values in results.items():
    mat = np.array([v for _,v in values]); ids = mat.argmax(axis=0)
    print(group,'n=',len(mat),'max source/phase/field=',mat.max(axis=0),'worst IDs=',[values[i][0] for i in ids])
    failed.extend((group, tag, v) for tag,v in values if not np.all(v<=limits))
assert not failed, failed[:10]
PY
```
- **Container Exit Code**: `0` (`4161 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public n= 2 max source/phase/field= [0.00567842 0.00062272 0.00363526] worst IDs= [1, 0, 1]
public_leave_one_survey_out n= 280 max source/phase/field= [0.00626013 0.00062272 0.00409203] worst IDs= [(1, 27), (0, 0), (1, 27)]
public_leave_one_pilot_out n= 24 max source/phase/field= [0.00571967 0.00074135 0.00364305] worst IDs= [(1, 0), (0, 10), (1, 0)]
stress_random n= 84 max source/phase/field= [6.94420948e-03 4.44089210e-16 4.17117135e-03] worst IDs= [(0, 0, 24), (0, 0, 0), (0, 1, 9)]
stress_localized n= 78 max source/phase/field= [7.51343500e-03 4.44089210e-16 3.80235486e-03] worst IDs= [(0, 1, 10), (0, 0, 1), (0, 1, 10)]
stress_high_response n= 78 max source/phase/field= [6.69007258e-03 4.44089210e-16 4.70719179e-03] worst IDs= [(0, 1, 2), (0, 0, 2), (0, 1, 2)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=92e5701a67ae1517 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_92e5701a (directory).
```

### Turn 4 — `submit_completed_task` (`14697 ms` LLM)
- **Rationale**: The saved artifact passed isolated Python verification, including rectangular-grid stencil checks, phase wrapping, and random, localized, and high-response receiver corruption. Each output passed its own tolerance.
- **Alternatives Weighed**: `Retune the robust estimator` (rejected: No tested slice approached tolerance; tuning on two public cases would risk overfitting without addressing an observed failure.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3833'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Both public cases, all 304 leave-one-out runs, and 240 synthetic stress cases passed. Worst errors were 0.752% for source amplitudes, 0.000742 rad for phase, and 0.471% for field magnitude.`

### Turn 5 — `execute_in_remote_docker` (`32762 ms` LLM)
- **Rationale**: The stencil already includes the exact pairwise face differences, wall terms, and wave term; the independent stencil check verified their signs and scaling. Unlike individual receiver deletion, spatial blocking can reveal dependence on an entire region, and separate pilot-strength fits can reveal a phase estimate carried by only the strongest references. A failed spatial block would motivate inspecting conditioning and receiver weights; a failed pilot block would motivate inspecting phase residuals by amplitude before changing calibration.
- **Alternatives Weighed**: `Add nonlinear source interactions or batch corrections` (rejected: For a fixed medium the stencil is linear in source amplitudes, and the supplied data specify a common phase and no batch labels. The blocked audit tests the relevant nuisance mechanisms without inventing new inputs.)

```bash
cat > /tmp/audit_helmholtz_receiver_blocks.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('helmholtz_phase_inversion', '/holdout_task/helmholtz_phase_inversion.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)


def rel(a, b):
    return np.linalg.norm(a-b) / np.linalg.norm(b)


def circular(a, b):
    return abs(np.angle(np.exp(1j*(a-b))))


def errors(result, gt):
    return np.array([rel(result['source_amplitudes'], gt['source_amplitudes']),
                     circular(result['global_phase_rad'], gt['global_phase_rad']),
                     rel(result['predicted_field_abs'], gt['predicted_field_abs'])])


rows = []
for ci, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    survey, pilot = p['survey'], p['pilot_calibration']
    i, j = survey['sensor_ij'].T
    nx, ny = p['n_sq'].shape
    quadrant = 2*(i >= nx//2) + (j >= ny//2)
    X = mod._responses(p)[i,j]
    zref = X @ gt['source_amplitudes']
    ref = pilot['known_field_re'] + 1j*pilot['known_field_im']
    order = np.argsort(abs(ref))
    for label, indices in [('weak', order[:len(order)//2]), ('strong', order[len(order)//2:])]:
        sub = {k: v[indices] for k,v in pilot.items()}
        e = errors(mod.invert_helmholtz_sources_and_phase(dict(p, pilot_calibration=sub)), gt)
        print('case', ci, 'pilot half', label, 'source/phase/field', e)
        rows.append(e)
    for q in range(4):
        held = quadrant == q
        sub = {k: v[~held] if k != 'tikhonov_alpha' else v for k,v in survey.items()}
        out = mod.invert_helmholtz_sources_and_phase(dict(p, survey=sub))
        e = errors(out, gt)
        zhat = X[held] @ out['source_amplitudes']
        measured = mod._polar(survey)[held] * np.exp(-1j*out['global_phase_rad'])
        print('case', ci, 'held quadrant', q, 'n', sum(held), 'source/phase/field', e,
              'held complex prediction rel', rel(zhat,zref[held]),
              'held median data residual', np.median(abs(zhat-measured)))
        rows.append(e)
worst = np.max(rows, axis=0)
limits = np.array([.03,.02,.03])
print('blocked audit worst source/phase/field', worst, 'fraction of tolerance', worst/limits)
assert np.all(worst <= limits)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_helmholtz_receiver_blocks.py
```
- **Container Exit Code**: `0` (`592 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case 0 pilot half weak source/phase/field [3.28301632e-03 7.33446180e-05 2.24205481e-03]
case 0 pilot half strong source/phase/field [0.00308881 0.00072366 0.00216921]
case 0 held quadrant 0 n 36 source/phase/field [0.00411957 0.00062272 0.00347249] held complex prediction rel 0.0049516725707414565 held median data residual 0.002337162020993949
case 0 held quadrant 1 n 36 source/phase/field [0.00231099 0.00062272 0.00178255] held complex prediction rel 0.0016713375781965266 held median data residual 0.001885787549023188
case 0 held quadrant 2 n 31 source/phase/field [0.00507833 0.00062272 0.0028672 ] held complex prediction rel 0.008515775641352006 held median data residual 0.001862088432628376
case 0 held quadrant 3 n 37 source/phase/field [0.0032608  0.00062272 0.00270446] held complex prediction rel 0.0050784186772109855 held median data residual 0.0019979816130161836
case 1 pilot half weak source/phase/field [0.0058563  0.00139189 0.00367416]
case 1 pilot half strong source/phase/field [0.00562722 0.00010892 0.00362393]
case 1 held quadrant 0 n 42 source/phase/field [0.00590682 0.0002669  0.00391158] held complex prediction rel 0.004109469011122706 held median data residual 0.0019344766105862219
case 1 held quadrant 1 n 36 source/phase/field [0.00715708 0.0002669  0.0044557 ] held complex prediction rel 0.008101077004995646 held median data residual 0.0018129370100718506
case 1 held quadrant 2 n 31 source/phase/field [0.00539349 0.0002669  0.003853  ] held complex prediction rel 0.009684250816242907 held median data residual 0.0015688194834336173
case 1 held quadrant 3 n 31 source/phase/field [0.00727819 0.0002669  0.00428761] held complex prediction rel 0.005624943237444174 held median data residual 0.0018027374830611763
blocked audit worst source/phase/field [0.00727819 0.00139189 0.0044557 ] fraction of tolerance [0.24260621 0.06959426 0.14852345]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=92e5701a67ae1517 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_92e5701a (directory).
```

### Turn 6 — `submit_completed_task` (`15912 ms` LLM)
- **Rationale**: The saved artifact passed isolated Python verification. Independent stencil checks confirmed the flux signs and rectangular-grid scaling, and the real-data block audit left all three errors below one quarter of their respective tolerances. The task has no batch labels or relay states that would justify additional corrections.
- **Alternatives Weighed**: `Retune receiver rejection or pilot weighting` (rejected: Removing an entire receiver quadrant or half the pilots did not expose a tolerance bottleneck; retuning on two public cases would risk overfitting.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3833'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Both public cases, 304 leave-one-out runs, 240 synthetic stress cases, and 12 receiver and pilot block tests passed. Worst errors were 0.752% for source amplitudes, 0.001392 rad for global phase, and 0.471% for field magnitude.`

### Turn 7 — `submit_completed_task` (`15931 ms` LLM)
- **Rationale**: Independent stencil checks verified face differences, wall flux signs, and rectangular-grid scaling. All three outputs passed their own tolerances, and the data specify neither batch labels nor relay states that would justify further corrections.
- **Alternatives Weighed**: `Repeat the blocked audit or retune the estimator` (rejected: The unchanged artifact already passed isolated execution and the real-data receiver and pilot block audit. The latest gate provides no new slice or observed failure.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Both public cases, 304 leave-one-out runs, 240 synthetic stress cases, and 12 receiver and pilot block tests passed. Worst observed errors were 0.751% for source amplitudes, 0.001392 rad for global phase, and 0.471% for field magnitude.`

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
