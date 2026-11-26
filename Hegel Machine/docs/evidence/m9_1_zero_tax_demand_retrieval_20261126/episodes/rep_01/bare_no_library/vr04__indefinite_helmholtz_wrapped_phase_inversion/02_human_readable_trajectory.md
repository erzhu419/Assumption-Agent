# Dialectical Synthesis Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_2_Selective_LiveGPT` | `vr04__indefinite_helmholtz_wrapped_phase_inversion`

- **Official Strict Pass**: **`PASS (1.0)`** (`PASSED_ALL_CHECKS`)
- **Blind Public Synthesis Choice**: `antithesis` via `selective_router_confident_single_branch`
- **Synthesis Rationale**: Primary Antithesis branch satisfied public checks under policy; second branch skipped (`1.00u` nominal units).
- **Post-Lock Branch Verdicts**: Thesis (`Bare`) = `NOT_SPAWNED (0T)` | Antithesis (`Wormhole`) = `True` (7T)

## Selected Branch Trajectory Record
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
| **Total Tokens (Prompt / Cached / Out)** | `136,532` (`130,392` / `44,800` / `6,140`) |
| **Total Episode Wall Time (s)** | `203.26s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`15428 ms` LLM)
- **Rationale**: Inspect only the fixture structure and SciPy availability. This detects schema mismatches and establishes whether sparse LU and reference-based scoring can be used directly.
- **Alternatives Weighed**: `Implement before inspecting public cases` (rejected: The public pickle schema and available reference outputs determine how to build the validation harness.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle, numpy as np, scipy
with open('public_helmholtz_cases.pkl','rb') as f: data=pickle.load(f)
print('scipy',scipy.__version__)
def describe(x,d=0):
 if isinstance(x,dict):
  for k,v in list(x.items())[:35]:
   print('  '*d,k,end=': '); describe(v,d+1)
 elif isinstance(x,(list,tuple)):
  print(type(x).__name__,len(x))
  if len(x): describe(x[0],d+1)
 elif hasattr(x,'shape'): print(x.shape,x.dtype)
 else: print(str(x)[:150])
describe(data)
PY
```
- **Container Exit Code**: `0` (`599 ms`, timed_out=`False`)
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

### Turn 2 — `execute_in_remote_docker` (`47477 ms` LLM)
- **Rationale**: Compare the forward field at the reference coefficients to isolate stencil errors, then score calibration, coefficients, and field independently to identify any remaining inverse bottleneck.
- **Alternatives Weighed**: `Use all-channel least squares as the final estimator` (rejected: It gives unbounded influence to the stated complex survey spikes; retain it as a public comparison.)

```bash
cat > /holdout_task/helmholtz_phase_inversion.py <<'PY'
"""Finite-volume acoustic source inversion from calibrated polar telemetry."""

import numpy as np
from scipy.sparse import diags, kronsum
from scipy.sparse.linalg import splu


def _axis_stiffness(n, weight):
    edge = np.full(n - 1, -weight)
    diagonal = np.zeros(n)
    diagonal[:-1] += weight
    diagonal[1:] += weight
    return diags((edge, diagonal, edge), (-1, 0, 1), format='csc')


def _basis_response(problem):
    medium = np.asarray(problem['n_sq'], dtype=float)
    basis = np.asarray(problem['basis'])
    nx, ny = medium.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    k = problem['k0']
    volume = hx * hy
    diagonal = (-volume * k**2 * medium).astype(np.complex128)
    flux = -1j * k * problem['beta_imp']
    diagonal[0, :] += hy * flux
    diagonal[-1, :] += hy * flux
    diagonal[:, 0] += hx * flux
    diagonal[:, -1] += hx * flux
    # C-order: y varies fastest, and each corner gets two wall fluxes.
    helmholtz = kronsum(_axis_stiffness(ny, hx / hy),
                        _axis_stiffness(nx, hy / hx), format='csc')
    helmholtz = helmholtz + diags(diagonal.ravel(), format='csc')
    rhs = volume * basis.reshape(len(basis), nx * ny).T
    response = splu(helmholtz).solve(rhs.astype(np.complex128))
    return response.reshape(nx, ny, len(basis))


def _polar(data):
    return np.asarray(data['amplitude_obs']) * np.exp(1j * np.asarray(data['wrapped_phase_obs']))


def _robust_real_fit(response, observed, alpha):
    # Treat each complex residual as one observation when assigning weights.
    m = response.shape[1]
    ridge = np.sqrt(alpha) * np.eye(m)
    target = np.r_[observed.real, observed.imag, np.zeros(m)]
    matrix = np.vstack((response.real, response.imag, ridge))

    def fit(weights):
        root = np.sqrt(weights)
        scale = np.r_[root, root, np.ones(m)]
        return np.linalg.lstsq(matrix * scale[:, None], target * scale, rcond=None)[0]

    estimate = fit(np.ones(len(observed)))
    floor = np.finfo(float).eps * max(np.max(np.abs(observed)), np.finfo(float).tiny)
    for _ in range(50):
        residual = np.abs(observed - response @ estimate)
        cutoff = max(1.6 * np.median(residual), floor)
        updated = fit(cutoff / np.maximum(residual, cutoff))
        converged = np.linalg.norm(updated - estimate) <= 1e-9 * np.linalg.norm(updated)
        estimate = updated
        if converged:
            break

    # Refit the inlier channels without Huber shrinkage of their residuals.
    for _ in range(5):
        residual = np.abs(observed - response @ estimate)
        keep = residual <= max(3.5 * np.median(residual), floor)
        updated = fit(keep)
        converged = np.linalg.norm(updated - estimate) <= 1e-9 * np.linalg.norm(updated)
        estimate = updated
        if converged:
            break
    return estimate


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Recover real source coefficients, oscillator phase, and pressure magnitude."""
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _polar(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    response = _basis_response(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    observed = _polar(survey) * np.exp(-1j * phase)
    coefficients = _robust_real_fit(response[i, j], observed, survey['tikhonov_alpha'])
    return {
        'source_amplitudes': coefficients,
        'global_phase_rad': phase,
        'predicted_field_abs': np.abs(response @ coefficients),
    }
PY
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
path = '/holdout_task/helmholtz_phase_inversion.py'
spec = importlib.util.spec_from_file_location('inversion', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_helmholtz_cases.pkl','rb') as f: cases = pickle.load(f)
for i, case in enumerate(cases):
 p,g = case['problem'],case['gt']; ans = m.invert_helmholtz_sources_and_phase(p)
 print(i, {k: (abs(np.angle(np.exp(1j*(v-g[k])))) if np.isscalar(v) else np.linalg.norm(v-g[k])/np.linalg.norm(g[k])) for k,v in ans.items()})
 b=m._basis_response(p); s=p['survey']; x,y=s['sensor_ij'].T
 A=b[x,y]; z=m._polar(s)*np.exp(-1j*g['global_phase_rad']); r=np.abs(z-A@g['source_amplitudes']); a=np.linalg.lstsq(np.vstack((A.real,A.imag)),np.r_[z.real,z.imag],rcond=None)[0]
 print('design singular values',np.linalg.svd(np.vstack((A.real,A.imag)),compute_uv=False),'true residual quantiles',np.quantile(r,[0,.5,.9,.93,.95,1]),'ols source relerr',np.linalg.norm(a-g['source_amplitudes'])/np.linalg.norm(g['source_amplitudes']),'true full field consistency',np.linalg.norm(np.abs(b@g['source_amplitudes'])-g['predicted_field_abs'])/np.linalg.norm(g['predicted_field_abs']))
PY
```
- **Container Exit Code**: `0` (`557 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/helmholtz_phase_inversion.py']`
- **Stdout (tail)**:
```text
0 {'source_amplitudes': 0.0035790065572618222, 'global_phase_rad': 0.0006227225657142733, 'predicted_field_abs': 0.0022891758297308683}
design singular values [0.8506539  0.68277291 0.64396571 0.48967366 0.39137982 0.28720722] true residual quantiles [6.47881901e-06 2.04397750e-03 3.74096331e-03 1.00151822e-01
 3.72538342e-01 7.40687115e-01] ols source relerr 0.1718031349171875 true full field consistency 1.2624230001329256e-15
1 {'source_amplitudes': 0.004715349089309256, 'global_phase_rad': 0.00026689595102702407, 'predicted_field_abs': 0.003201396421629576}
design singular values [0.88022524 0.69563054 0.63706238 0.44340971 0.39510715 0.28270252] true residual quantiles [2.09377251e-04 1.80389227e-03 4.26435661e-03 9.99000576e-02
 4.09010034e-01 6.86029873e-01] ols source relerr 0.2727211285456098 true full field consistency 8.893619912307587e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=540f92775673bff5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_540f9277 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`43346 ms` LLM)
- **Rationale**: Leave-one-receiver-out tests would expose dependence on a single public channel. An independently assembled unequal-cell forward solve would expose stencil errors; phase-seam and concentrated-spike tests would expose calibration or estimator failures. Any failure identifies the component to revise; otherwise keep the verified candidate unchanged.
- **Alternatives Weighed**: `Submit on the two public aggregate scores` (rejected: The square fixtures do not expose swapped face weights, and aggregate scores do not test dependence on individual receivers or spikes concentrated on high-leverage channels.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
spec = importlib.util.spec_from_file_location('inversion','/holdout_task/helmholtz_phase_inversion.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_helmholtz_cases.pkl','rb') as f:
    cases = pickle.load(f)
rng = np.random.default_rng(380)
limits = np.array([.03,.02,.03])
def scores(ans, gt, shape):
    assert set(ans) == {'source_amplitudes','global_phase_rad','predicted_field_abs'}
    a, p, u = (ans[k] for k in ('source_amplitudes','global_phase_rad','predicted_field_abs'))
    b, q, v = gt
    assert a.shape == b.shape and u.shape == shape and a.dtype.kind == u.dtype.kind == 'f'
    assert np.isfinite(a).all() and np.isfinite(u).all() and (u >= 0).all() and -np.pi < p <= np.pi
    return np.array([np.linalg.norm(a-b)/np.linalg.norm(b),abs(np.angle(np.exp(1j*(p-q)))),np.linalg.norm(u-v)/np.linalg.norm(v)])
for idx, case in enumerate(cases):
    p, gt = case['problem'],case['gt']; s=p['survey']; ij=s['sensor_ij']; shape=p['n_sq'].shape
    ref=tuple(gt[k] for k in ('source_amplitudes','global_phase_rad','predicted_field_abs'))
    public=scores(m.invert_helmholtz_sources_and_phase(p),ref,shape)
    loo=[]
    for k in range(len(ij)):
        mask=np.arange(len(ij))!=k
        sliced=dict(s, **{name:s[name][mask] for name in ('sensor_ij','amplitude_obs','wrapped_phase_obs')})
        loo.append(scores(m.invert_helmholtz_sources_and_phase(dict(p,survey=sliced)),ref,shape))
    worst=np.max(loo,axis=0)
    print('public',idx,public,'leave-one-survey-sensor-out worst',worst)
    assert np.all(public<=limits) and np.all(worst<=limits)
# Independently assemble an unequal-cell finite-volume system by visiting each face.
def forward(p,a):
    n=p['n_sq']; nx,ny=n.shape; hx,hy=p['Lx']/nx,p['Ly']/ny; k=p['k0']; rows=[]; cols=[]; data=[]
    for i,j in np.ndindex(n.shape):
        z=i*ny+j; d=-hx*hy*k*k*n[i,j]+0j
        for di,dj,length,w in ((1,0,hy,hy/hx),(-1,0,hy,hy/hx),(0,1,hx,hx/hy),(0,-1,hx,hx/hy)):
            x,y=i+di,j+dj
            if 0<=x<nx and 0<=y<ny:
                rows.append(z); cols.append(x*ny+y); data.append(-w); d+=w
            else:
                d-=length*1j*k*p['beta_imp']
        rows.append(z); cols.append(z); data.append(d)
    h=coo_matrix((data,(rows,cols)),shape=(n.size,n.size)).tocsc()
    return spsolve(h,hx*hy*np.einsum('m,mij->ij',a,p['basis']).ravel()).reshape(n.shape)
def polar(z):
    return {'amplitude_obs':np.abs(z),'wrapped_phase_obs':np.angle(z)}
stress=[]
for case in cases:
    refp=case['problem']; ij=refp['survey']['sensor_ij']; i,j=ij.T
    for lx,ly,k in ((1.,1.,11.5),(1.23,.83,10.8),(.84,1.24,12.7)):
        p=dict(refp,Lx=lx,Ly=ly,k0=k); shape=p['n_sq'].shape
        a=rng.normal(size=len(p['basis'])); u=forward(p,a); z=u[i,j]
        b=m._basis_response(p)
        assert np.linalg.norm(b@a-u)/np.linalg.norm(u)<1e-11
        design=b[i,j]; gram=np.linalg.inv((design.conj().T@design).real)
        leverage=np.einsum('ij,jk,ik->i',design.conj(),gram,design).real
        for phase in (-np.pi+1e-10,np.pi-1e-10,-.7):
            pilot=rng.normal(size=12)+1j*rng.normal(size=12); rot=np.exp(1j*phase)
            p['pilot_calibration']=dict(polar(rot*pilot),known_field_re=pilot.real,known_field_im=pilot.imag)
            for mode in ('random','high-leverage'):
                noisy=z+.002*(rng.normal(size=len(z))+1j*rng.normal(size=len(z)))
                count=round(.07*len(z)); hit=rng.choice(len(z),count,replace=False) if mode=='random' else np.argsort(leverage)[-count:]
                noisy[hit]+=rng.uniform(.12,.75,count)*np.exp(1j*rng.uniform(-np.pi,np.pi,count))
                p['survey']=dict(refp['survey'],**polar(rot*noisy))
                stress.append(scores(m.invert_helmholtz_sources_and_phase(p),(a,phase,np.abs(u)),shape))
worst=np.max(stress,axis=0)
print('synthetic stress cases',len(stress),'worst [source relative, phase rad, field relative]',worst)
assert np.all(worst<=limits)
PY
```
- **Container Exit Code**: `0` (`2314 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0 [0.00357901 0.00062272 0.00228918] leave-one-survey-sensor-out worst [0.00425068 0.00062272 0.00256123]
public 1 [0.00471535 0.0002669  0.0032014 ] leave-one-survey-sensor-out worst [0.00542723 0.0002669  0.00359968]
synthetic stress cases 36 worst [source relative, phase rad, field relative] [1.14013635e-02 4.44089210e-16 5.79753195e-03]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=540f92775673bff5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_540f9277 (directory).
```

### Turn 4 — `submit_completed_task` (`12870 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification. An independent unequal-cell assembly confirmed the forward stencil, and receiver omissions, phase wrapping, and concentrated spikes did not breach any output tolerance.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3650'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/helmholtz_phase_inversion.py` and verified all three outputs against their tolerances. Across the public cases, receiver jackknifes, and 36 synthetic stress cases, the worst errors were 1.14% for source amplitudes, 0.000623 rad for phase, and 0.580% for field magnitude.`

### Turn 5 — `execute_in_remote_docker` (`31477 ms` LLM)
- **Rationale**: Use the unchanged artifact to check three gaps in the earlier verification: a dominant pilot would motivate revisiting calibration weights, a quadrant failure would motivate diagnosing spatial conditioning or outlier concentration, and a sign-symmetry failure would identify a phase or real-fit convention error.
- **Alternatives Weighed**: `Add nonlinear terms or fit batch corrections` (rejected: The forward equation is linear in source amplitudes, and the public schema has no batch labels or classification outcome. Pilot phase and receiver geometry are the relevant nuisance and coverage factors.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
base = '/holdout_task/'
spec = importlib.util.spec_from_file_location('inversion', base + 'helmholtz_phase_inversion.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(base + 'public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
keys = ('source_amplitudes', 'global_phase_rad', 'predicted_field_abs')
limits = np.array([.03,.02,.03])
def score(ans, gt):
    a, p, u = (ans[k] for k in keys)
    b, q, v = (gt[k] for k in keys)
    assert -np.pi < p <= np.pi
    return np.array([np.linalg.norm(a-b)/np.linalg.norm(b), abs(np.angle(np.exp(1j*(p-q)))), np.linalg.norm(u-v)/np.linalg.norm(v)])
for n, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    pilot, survey = p['pilot_calibration'], p['survey']
    ij = survey['sensor_ij']
    split = np.array(p['n_sq'].shape)//2
    quadrant = 2*(ij[:,0]>=split[0]) + (ij[:,1]>=split[1])
    pilot_runs = []
    for k in range(len(pilot['amplitude_obs'])):
        reduced = {name: np.delete(values, k) for name, values in pilot.items()}
        pilot_runs.append(score(m.invert_helmholtz_sources_and_phase(dict(p,pilot_calibration=reduced)),gt))
    spatial_runs = []
    for quadrant_id in np.unique(quadrant):
        keep = quadrant != quadrant_id
        reduced = dict(survey, **{name:survey[name][keep] for name in ('sensor_ij','amplitude_obs','wrapped_phase_obs')})
        spatial_runs.append(score(m.invert_helmholtz_sources_and_phase(dict(p,survey=reduced)),gt))
    plain = m.invert_helmholtz_sources_and_phase(p)
    # Rotating only the pilot telemetry by pi must flip source signs without changing field magnitude.
    shifted = dict(pilot, wrapped_phase_obs=np.angle(-np.exp(1j*pilot['wrapped_phase_obs'])))
    flipped = m.invert_helmholtz_sources_and_phase(dict(p,pilot_calibration=shifted))
    equivariant = dict(source_amplitudes=-plain[keys[0]],global_phase_rad=np.angle(-np.exp(1j*plain[keys[1]])),predicted_field_abs=plain[keys[2]])
    symmetry = score(flipped,equivariant)
    a, b = np.max(pilot_runs,axis=0), np.max(spatial_runs,axis=0)
    print(n, 'worst pilot omission', a, 'worst quadrant omission', b, 'pi/source-sign symmetry residual', symmetry)
    assert np.all(a <= limits) and np.all(b <= limits) and np.all(symmetry < 1e-10)
PY
```
- **Container Exit Code**: `0` (`701 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 worst pilot omission [0.00361809 0.00074135 0.00230511] worst quadrant omission [0.00555353 0.00062272 0.00344887] pi/source-sign symmetry residual [4.01331410e-16 0.00000000e+00 3.53343571e-16]
1 worst pilot omission [0.00473207 0.00058973 0.00321022] worst quadrant omission [0.00628773 0.0002669  0.0039954 ] pi/source-sign symmetry residual [6.34791528e-16 2.22044605e-16 4.95844886e-16]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=540f92775673bff5 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_540f9277 (directory).
```

### Turn 6 — `submit_completed_task` (`14026 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification, including an independently assembled unequal-cell stencil. The public pilot and quadrant omissions stayed within tolerance, and the phase/source-sign symmetry held to floating-point precision. No observed failure warrants a change.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/helmholtz_phase_inversion.py': 'FILE:3650'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across public cases, receiver and pilot omissions, quadrant omissions, and 36 synthetic stress cases, the worst errors were 1.14% for source amplitudes, 0.000742 rad for phase, and 0.580% for field magnitude. All passed their tolerances.`

### Turn 7 — `submit_completed_task` (`15529 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process verification. The real-data pilot and quadrant omissions stayed within tolerance, and the independent unequal-cell stencil and phase-sign checks found no failure that warrants an edit.
- **Alternatives Weighed**: `Add nonlinear basis terms or batch corrections` (rejected: The specified forward map is linear in the real source coefficients, and the telemetry has no batch variable. The independent stencil and phase-sign checks passed.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/helmholtz_phase_inversion.py`. Across the public cases, receiver and pilot omissions, and 36 synthetic stress cases, the worst errors were 1.14% for source amplitudes, 0.000742 rad for phase, and 0.580% for field magnitude. All were below the required tolerances.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-M90a-VoluntaryReuse-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/helmholtz_phase_inversion.py": "FILE:3650"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "all_valid": true,
    "max_source_rel_l2": 0.004235,
    "max_phase_abs_err_rad": 0.000852,
    "max_field_abs_rel_l2": 0.003,
    "elapsed_sec": 0.1216
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"all_valid\": true, \"max_source_rel_l2\": 0.004235, \"max_phase_abs_err_rad\": 0.000852, \"max_field_abs_rel_l2\": 0.003, \"elapsed_sec\": 0.1216}\n__HEGEL_VERIFIER_AUTH_e3ea5ab0576cae8a5cdf0115e2cb0af0__:{\"strict_pass\": true, \"all_valid\": true, \"max_source_rel_l2\": 0.004235, \"max_phase_abs_err_rad\": 0.000852, \"max_field_abs_rel_l2\": 0.003, \"elapsed_sec\": 0.1216}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5578.86,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
