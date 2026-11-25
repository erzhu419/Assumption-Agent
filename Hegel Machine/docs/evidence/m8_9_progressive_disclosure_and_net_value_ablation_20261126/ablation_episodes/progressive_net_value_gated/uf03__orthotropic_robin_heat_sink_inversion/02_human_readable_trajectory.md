# Dialectical Synthesis Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_2_Selective_LiveGPT` | `uf03__orthotropic_robin_heat_sink_inversion`

- **Official Strict Pass**: **`PASS (1.0)`** (`PASSED_ALL_CHECKS`)
- **Blind Public Synthesis Choice**: `antithesis` via `selective_router_confident_single_branch`
- **Synthesis Rationale**: Primary Antithesis branch satisfied public checks under policy; second branch skipped (`1.00u` nominal units).
- **Post-Lock Branch Verdicts**: Thesis (`Bare`) = `NOT_SPAWNED (0T)` | Antithesis (`Wormhole`) = `True` (15T)

## Selected Branch Trajectory Record
# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `uf03__orthotropic_robin_heat_sink_inversion`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-Unseen-PostFreeze-4` |
| **Domain** | `Thermal Physics / Conservative Finite-Volume PDE Inversion` |
| **Wave / Wormhole Family** | `wave_2` / `fam_unseen_elliptic_robin_inversion` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `15 / 28` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `7` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `515,784` (`498,534` / `95,232` / `17,250`) |
| **Total Episode Wall Time (s)** | `579.7s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`16311 ms` LLM)
- **Rationale**: Inspect compact summaries to establish the input schema, grid and sensor sizes, and whether reference parameters are available. Those findings determine whether to validate against public truth or use held-out sensor predictions and synthetic recovery tests.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle
import numpy as np
with open('public_thermal_cases.pkl','rb') as f:
    cases=pickle.load(f)
def describe(obj, depth=0):
    pad='  '*depth
    if isinstance(obj,dict):
        print(f'{pad}dict keys: {list(obj)}')
        for k,v in obj.items():
            print(f'{pad}{k}:')
            describe(v,depth+1)
    elif isinstance(obj,np.ndarray):
        print(f'{pad}array shape={obj.shape}, dtype={obj.dtype}, range={(obj.min(),obj.max()) if obj.size and np.issubdtype(obj.dtype,np.number) else None}, first={obj.reshape(-1)[:6]}')
    elif isinstance(obj,(list,tuple)) and len(obj) and isinstance(obj[0],(dict,list,tuple,np.ndarray)):
        print(f'{pad}{type(obj).__name__}, len={len(obj)}')
        for v in obj[:8]: describe(v,depth+1)
    else:
        print(f'{pad}{str(obj)[:300]}')
describe(cases)
PY
```
- **Container Exit Code**: `0` (`608 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
list, len=2
  dict keys: ['measurement', 'gt']
  measurement:
    dict keys: ['N', 'kappa', 'gamma_robin', 'sigma_m', 'sensor_ij', 'T_obs']
    N:
      22
    kappa:
      array shape=(22, 22), dtype=float64, range=(0.6176077121657585, 2.0790307364930127), first=[1.08671379 1.0799142  1.0669848  1.04917011 1.02812208 1.00569799]
    gamma_robin:
      2.2
    sigma_m:
      0.085
    sensor_ij:
      array shape=(52, 2), dtype=int64, range=(0, 20), first=[ 9  9 11  0  0 16]
    T_obs:
      array shape=(52,), dtype=float64, range=(-0.012127692222733938, 0.421323026416489), first=[0.2464152  0.1231993  0.19110304 0.1855862  0.1871671  0.12883992]
  gt:
    dict keys: ['source_xy', 'amplitude', 'junction_offset']
    source_xy:
      [0.34, 0.62]
    amplitude:
      18.0
    junction_offset:
      0.045
  dict keys: ['measurement', 'gt']
  measurement:
    dict keys: ['N', 'kappa', 'gamma_robin', 'sigma_m', 'sensor_ij', 'T_obs']
    N:
      22
    kappa:
      array shape=(22, 22), dtype=float64, range=(0.6176077121657585, 2.0790307364930127), first=[1.08671379 1.0799142  1.0669848  1.04917011 1.02812208 1.00569799]
    gamma_robin:
      2.2
    sigma_m:
      0.085
    sensor_ij:
      array shape=(52, 2), dtype=int64, range=(0, 21), first=[18 21  9 16 16 20]
    T_obs:
      array shape=(52,), dtype=float64, range=(-0.07781381443172214, 0.26420885419289625), first=[0.06208414 0.1130059  0.08415969 0.12602429 0.19392087 0.06218506]
  gt:
    dict keys: ['source_xy', 'amplitude', 'junction_offset']
    source_xy:
      [0.68, 0.29]
    amplitude:
      24.5
    junction_offset:
      -0.032
```

### Turn 2 — `execute_in_remote_docker` (`60018 ms` LLM)
- **Rationale**: Build the conservative operator and reuse one factorization for sensor Green functions. Independent cellwise flux balance will expose indexing or boundary errors; the public fits will show whether robust basin selection and refinement meet each recovery tolerance.
- **Alternatives Weighed**: `Reuse the Darcy solver` (rejected: Its zero-ghost boundary operator does not implement the specified Robin loss.)

```bash
cat > /holdout_task/thermal_sink_solver.py <<'PY'
"""Recover a Gaussian heat source and a common thermistor offset."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _factor_operator(kappa, gamma):
    n = kappa.shape[0]
    cells = np.arange(n*n).reshape(n, n)
    left = np.concatenate((cells[:-1, :].ravel(), cells[:, :-1].ravel()))
    right = np.concatenate((cells[1:, :].ravel(), cells[:, 1:].ravel()))
    k = kappa.ravel()
    conductance = 2*k[left]*k[right]/(k[left]+k[right])
    diagonal = np.zeros((n, n))
    diagonal[[0, -1], :] += gamma/n
    diagonal[:, [0, -1]] += gamma/n
    diagonal = diagonal.ravel()
    if n == 1:
        diagonal[0] = 4*gamma
    diagonal += np.bincount(left, conductance, minlength=n*n)
    diagonal += np.bincount(right, conductance, minlength=n*n)
    ids = cells.ravel()
    matrix = coo_matrix((np.r_[diagonal, -conductance, -conductance],
                         (np.r_[ids, left, right], np.r_[ids, right, left])),
                        shape=(n*n, n*n)).tocsc()
    return splu(matrix)


def _trimmed_regression(pred, target):
    """Rank all grid points by trimmed residual energy after profiling gain/offset."""
    m = len(target)
    keep = max(min(4, m), m-int(np.ceil(0.12*m)))
    weights = np.ones_like(pred, dtype=bool)
    for _ in range(5):
        count = weights.sum(axis=1)
        mean_p = (weights*pred).sum(axis=1)/count
        mean_y = (weights*target).sum(axis=1)/count
        pc = pred-mean_p[:, None]
        numerator = (weights*pc*(target-mean_y[:, None])).sum(axis=1)
        denominator = (weights*pc**2).sum(axis=1)
        a = np.maximum(numerator/np.maximum(denominator, np.finfo(float).tiny), 0)
        b = mean_y-a*mean_p
        residual = pred*a[:, None]+b[:, None]-target
        indices = np.argpartition(np.abs(residual), keep-1, axis=1)[:, :keep]
        weights.fill(False)
        np.put_along_axis(weights, indices, True, axis=1)
    return (weights*residual**2).sum(axis=1), a, b


def localize_thermal_source(measurement: dict) -> dict:
    n = int(measurement['N'])
    kappa = np.asarray(measurement['kappa'], dtype=float)
    gamma = float(measurement['gamma_robin'])
    sigma = float(measurement['sigma_m'])
    ij = np.asarray(measurement['sensor_ij'], dtype=int)
    y = np.asarray(measurement['T_obs'], dtype=float)
    cells, inverse = np.unique(n*ij[:, 0]+ij[:, 1], return_inverse=True)
    rhs = np.zeros((n*n, len(cells)))
    rhs[cells, np.arange(len(cells))] = 1/n**2
    green = np.ascontiguousarray(_factor_operator(kappa, gamma).solve(rhs).T)

    centers = (np.arange(n)+0.5)/n
    axis = np.linspace(0, 1, 25)
    grid = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
    gr = np.exp(-0.5*((axis[:, None]-centers)/sigma)**2)
    source = (gr[:, None, :, None]*gr[None, :, None, :]).reshape(len(grid), n*n)
    pred = (source @ green.T)[:, inverse]
    scores, amplitudes, offsets = _trimmed_regression(pred, y)
    seeds = []
    for idx in np.argsort(scores):
        if all(np.linalg.norm(grid[idx]-grid[s]) >= 0.115 for s in seeds):
            seeds.append(idx)
            if len(seeds) == 6:
                break

    x = np.repeat(centers, n)
    z = np.tile(centers, n)
    inv_s2 = 1/sigma**2
    amplitude_scale = max(amplitudes[seeds[0]], np.std(y)/max(np.max(np.ptp(pred, axis=1)), 1e-20), 1e-12)
    lo = np.array([0., 0., 0., -np.inf])
    hi = np.array([1., 1., np.inf, np.inf])

    def evaluate(p):
        xs, ys, scaled_a, b = p
        a = scaled_a*amplitude_scale
        dx, dy = x-xs, z-ys
        g = np.exp(-0.5*inv_s2*(dx*dx+dy*dy))
        basis = np.array([g, g*dx, g*dy])
        f, fx, fy = np.einsum('ij,kj->ki', green, basis)[..., inverse]
        residual = a*f+b-y
        jac = np.column_stack((a*inv_s2*fx, a*inv_s2*fy, amplitude_scale*f, np.ones_like(f)))
        return residual, jac

    # Use a common initial cutoff to compare basins, then estimate its noise
    # component again after continuous localization removes grid mismatch.
    best = seeds[0]
    r0 = amplitudes[best]*pred[best]+offsets[best]-y
    floor = max(1.e-12, 1.e-6*np.ptp(y))

    def cutoff(r):
        return max(floor, 2.5*1.4826*np.median(np.abs(r-np.median(r))))

    def fit(p, scale):
        return least_squares(lambda q: evaluate(q)[0], p,
                             jac=lambda q: evaluate(q)[1], bounds=(lo, hi),
                             loss='cauchy', f_scale=scale, x_scale='jac',
                             ftol=1.e-9, max_nfev=90)

    common = cutoff(r0)
    fits = [fit(np.r_[grid[s], max(amplitudes[s]/amplitude_scale, 1.e-5), offsets[s]], common)
            for s in seeds]
    winner = min(fits, key=lambda f: f.cost)
    for _ in range(2):
        winner = fit(winner.x, cutoff(winner.fun))
    xs, ys, a, b = winner.x
    return {'source_xy': [float(xs), float(ys)],
            'amplitude': float(a*amplitude_scale), 'junction_offset': float(b)}
PY
cd /holdout_task && python3 -I - <<'PY'
import importlib.util, pickle, time
import numpy as np
spec=importlib.util.spec_from_file_location('thermal_sink_solver','/holdout_task/thermal_sink_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for case in pickle.load(open('public_thermal_cases.pkl','rb')):
    t=time.monotonic(); p=m.localize_thermal_source(case['measurement']); g=case['gt']
    e=[np.linalg.norm(np.subtract(p['source_xy'],g['source_xy'])), abs(p['amplitude']/g['amplitude']-1), abs(p['junction_offset']-g['junction_offset'])]
    print('prediction',p,'errors (xy, relative amp, offset)',e,'seconds',time.monotonic()-t,flush=True)
    case_m=case['measurement']; n=case_m['N']; gamma=case_m['gamma_robin']; k=case_m['kappa']
    ii,jj=np.indices((n,n)); x=(ii+.5)/n; y=(jj+.5)/n
    src=g['amplitude']*np.exp(-((x-g['source_xy'][0])**2+(y-g['source_xy'][1])**2)/(2*case_m['sigma_m']**2))/n**2
    u=m._factor_operator(k,gamma).solve(src.ravel()).reshape(n,n)
    flux=np.zeros_like(u)
    for i,j in np.ndindex(n,n):
        for di,dj in ((-1,0),(1,0),(0,-1),(0,1)):
            a,b=i+di,j+dj
            flux[i,j] += (2*k[i,j]*k[a,b]/(k[i,j]+k[a,b])*(u[i,j]-u[a,b]) if 0<=a<n and 0<=b<n else gamma*u[i,j]/n)
    print('independent flux balance max error',np.max(abs(flux-src)), 'global error',abs(np.sum(flux)-np.sum(src)))
    assert np.max(abs(flux-src))<1e-12
PY
```
- **Container Exit Code**: `0` (`2087 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/thermal_sink_solver.py']`
- **Stdout (tail)**:
```text
prediction {'source_xy': [0.3396064793535851, 0.6199422121846537], 'amplitude': 17.983919433076192, 'junction_offset': 0.044977917769018216} errors (xy, relative amp, offset) [0.00039774103479185626, 0.000893364829100407, 2.208223098178197e-05] seconds 0.46913538384251297
independent flux balance max error 3.638175915046671e-16 global error 3.3306690738754696e-16
prediction {'source_xy': [0.6800732246749136, 0.28942468660663656], 'amplitude': 24.492604992308987, 'junction_offset': -0.03198640348159971} errors (xy, relative amp, offset) [0.0005799546133961929, 0.00030183704861275196, 1.3596518400292579e-05] seconds 0.7924734859261662
independent flux balance max error 7.91033905045424e-16 global error 8.881784197001252e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7a1ff1186d03c683 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7a1ff118 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`38143 ms` LLM)
- **Rationale**: Public fits already pass. Sensor deletion tests whether recovery depends on any single thermistor, while independently generated edge-source cases with corrupted high-signal sensors test whether basin selection and amplitude–offset separation remain reliable.

```bash
cat > /tmp/check_thermal_sink.py <<'PY'
import importlib.util
import pickle
import time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

path = '/holdout_task/'
spec = importlib.util.spec_from_file_location('thermal_sink_solver', path+'thermal_sink_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open(path+'public_thermal_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
rng = np.random.default_rng(84)
limits = np.array([.035, .08, .008, 15.])
results = {}

def check(group, label, m, g):
    t = time.monotonic()
    p = solver.localize_thermal_source(m)
    elapsed = time.monotonic()-t
    assert set(p) == {'source_xy', 'amplitude', 'junction_offset'}
    assert isinstance(p['source_xy'], list) and len(p['source_xy']) == 2
    assert all(type(v) is float and np.isfinite(v) for v in [*p['source_xy'], p['amplitude'], p['junction_offset']])
    assert all(0 <= v <= 1 for v in p['source_xy']) and p['amplitude'] > 0
    e = np.array([np.linalg.norm(np.subtract(p['source_xy'], g['source_xy'])), abs(p['amplitude']/g['amplitude']-1), abs(p['junction_offset']-g['junction_offset']), elapsed])
    results.setdefault(group, []).append(e)
    if not np.all(e <= limits):
        print('FAIL', group, label, e, p, g, flush=True)

for case_id, c in enumerate(cases):
    m, g = c['measurement'], c['gt']
    check('public', case_id, m, g)
    for drop in range(len(m['T_obs'])):
        keep = np.arange(len(m['T_obs'])) != drop
        check('public leave-one-sensor-out', (case_id, drop), dict(m, sensor_ij=m['sensor_ij'][keep], T_obs=m['T_obs'][keep]), g)

for trial in range(36):
    n = (18, 22, 30)[trial % 3]
    gamma = (0.45, 2.2, 5.0)[(trial//3) % 3]
    sigma = (.06, .085, .12)[(trial//9) % 3]
    x, y = (np.indices((n, n))+.5)/n
    k = np.exp(.55*np.sin(6*x+trial)*np.cos(5*y)+.35*(x-y))
    xy = rng.uniform(.08, .92, 2)
    if trial % 4 == 0:
        xy[trial//4 % 2] = .035 if trial % 8 == 0 else .965
    a = rng.uniform(14, 32)
    b = rng.uniform(-.075, .075)
    q = a*np.exp(-((x-xy[0])**2+(y-xy[1])**2)/(2*sigma**2))/n**2
    A = lil_matrix((n*n, n*n))
    for i,j in np.ndindex(n,n):
        row = i*n+j
        for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ii,jj = i+di,j+dj
            if 0 <= ii < n and 0 <= jj < n:
                w = 2/(1/k[i,j]+1/k[ii,jj])
                A[row,row] += w
                A[row,ii*n+jj] = -w
            else:
                A[row,row] += gamma/n
    u = spsolve(A.tocsc(),q.ravel())
    ids = rng.choice(n*n,52,replace=False)
    yy = u[ids]+b+rng.normal(0,.0015,len(ids))
    outliers = rng.choice(len(ids),3,replace=False) if trial%2 else np.argsort(u[ids])[-3:]
    yy[outliers] += rng.choice([-1,1],3)*rng.uniform(.12,.45,3)
    check('synthetic edge' if trial%4 == 0 else 'synthetic interior', (trial,n,gamma,sigma), dict(N=n,kappa=k,gamma_robin=gamma,sigma_m=sigma,sensor_ij=np.column_stack(divmod(ids,n)),T_obs=yy), dict(source_xy=xy,amplitude=a,junction_offset=b))

failed = 0
for group, es in results.items():
    es = np.asarray(es)
    bad = np.count_nonzero(np.any(es > limits,axis=1))
    failed += bad
    print(group, 'n=',len(es),'failed=',bad,'worst (position, relative amplitude, offset, seconds):',np.max(es,axis=0))
raise SystemExit(bool(failed))
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/check_thermal_sink.py
```
- **Container Exit Code**: `1` (`5506 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
FAIL synthetic edge (12, 18, 2.2, 0.085) [2.70825269e-02 2.56813631e-01 2.08796658e-04 4.49250422e-02] {'source_xy': [0.8232722287075307, 0.9902850514249851], 'amplitude': 31.700321204377843, 'junction_offset': 0.03425440255577942} {'source_xy': array([0.8135702, 0.965    ]), 'amplitude': 25.222770049753095, 'junction_offset': 0.034045605898185324}
FAIL synthetic edge (28, 22, 0.45, 0.06) [0.03847405 0.49883085 0.00232481 0.06594221] {'source_xy': [0.09135934207531272, 0.9999999994660638], 'amplitude': 44.51339121403033, 'junction_offset': 0.07354525191165202} {'source_xy': array([0.10733597, 0.965     ]), 'amplitude': 29.698742363322886, 'junction_offset': 0.07122044010124672}
public n= 2 failed= 0 worst (position, relative amplitude, offset, seconds): [5.79954613e-04 8.93364829e-04 2.20822310e-05 4.05141721e-02]
public leave-one-sensor-out n= 104 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00122105 0.00314668 0.0003519  0.04690997]
synthetic edge n= 9 failed= 2 worst (position, relative amplitude, offset, seconds): [0.03847405 0.49883085 0.00232481 0.09054121]
synthetic interior n= 27 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00824813 0.0383324  0.00231815 0.08265632]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7a1ff1186d03c683 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7a1ff118 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`26233 ms` LLM)
- **Rationale**: Fit the failing synthetic slice with the injected outliers removed and measure local parameter uncertainty. If the oracle fit recovers the amplitude, improve outlier handling; if it also fails, investigate identifiability before changing the estimator.
- **Alternatives Weighed**: `Constrain the source away from the plate edge` (rejected: The documented source domain includes the edge, and a coordinate clamp could hide amplitude–position ambiguity rather than resolve it.)

```bash
python3 - <<'PY'
p='/tmp/check_thermal_sink.py'
s=open(p).read()
a=s.index('for case_id, c in enumerate(cases):')
b=s.index('for trial in range(36):',a)
s=s[:a]+s[b:]
a=s.index("    check('synthetic edge'")
s=s[:a]+'''    if trial % 4 != 0:
        continue
    from scipy.optimize import least_squares
    yy_clean = yy.copy()
    bad = np.isin(np.arange(len(ids)), outliers)
    yy_clean[bad] = u[ids[bad]] + b
    m = dict(N=n,kappa=k,gamma_robin=gamma,sigma_m=sigma,sensor_ij=np.column_stack(divmod(ids,n)),T_obs=yy)
    rhs = np.zeros((n*n,len(ids)))
    rhs[ids,np.arange(len(ids))] = 1/n**2
    green = solver._factor_operator(k,gamma).solve(rhs).T
    truth = np.r_[xy,a,b]
    def eval(p):
        dx,dy=x-p[0],y-p[1]
        g=np.exp(-.5*(dx**2+dy**2)/sigma**2)
        design=green@np.stack([p[2]*g*dx/sigma**2,p[2]*g*dy/sigma**2,g],axis=-1).reshape(-1,3)
        return p[2]*design[:,2]+p[3],np.column_stack((design,np.ones(len(ids))))
    pred,J=eval(truth)
    fit=least_squares(lambda p:(eval(p)[0]-yy)[~bad],truth,jac=lambda p:eval(p)[1][~bad],bounds=([0,0,0,-np.inf],[1,1,np.inf,np.inf]))
    sd=.0015*np.sqrt(np.diag(np.linalg.inv(J[~bad].T@J[~bad])))
    got=solver.localize_thermal_source(m)
    p=np.r_[got['source_xy'],got['amplitude'],got['junction_offset']]
    r=eval(p)[0]-yy
    print(trial, 'relative amplitude err robust/oracle/1sigma:', np.round([(p[2]-a)/a,(fit.x[2]-a)/a,sd[2]/a],5),'edge truth/oracle xy:',np.round(xy,5),np.round(fit.x[:2],5),'outlier residuals:',np.round(r[bad],5),'oracle inlier residual RMS:',np.sqrt(np.mean(fit.fun**2)))
'''
open('/tmp/diagnose_thermal_edge.py','w').write(s)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/diagnose_thermal_edge.py
```
- **Container Exit Code**: `0` (`1226 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 relative amplitude err robust/oracle/1sigma: [-0.05193 -0.03208  0.07416] edge truth/oracle xy: [0.035   0.83617] [0.03893 0.83501] outlier residuals: [-0.43575  0.2894  -0.36397] oracle inlier residual RMS: 0.0013752509524211124
4 relative amplitude err robust/oracle/1sigma: [-0.03851 -0.0313   0.06552] edge truth/oracle xy: [0.21157 0.965  ] [0.2137 0.9616] outlier residuals: [-0.15574  0.40725  0.31747] oracle inlier residual RMS: 0.0015976347370567471
8 relative amplitude err robust/oracle/1sigma: [-0.00213  0.03392  0.26587] edge truth/oracle xy: [0.035   0.84503] [0.03038 0.84429] outlier residuals: [-0.12508  0.27153  0.41964] oracle inlier residual RMS: 0.0012522683383916894
12 relative amplitude err robust/oracle/1sigma: [0.25681 0.15335 0.21145] edge truth/oracle xy: [0.81357 0.965  ] [0.82027 0.98151] outlier residuals: [-0.43619  0.39156 -0.119  ] oracle inlier residual RMS: 0.0012402327797248667
16 relative amplitude err robust/oracle/1sigma: [-0.01569 -0.01746  0.04408] edge truth/oracle xy: [0.035   0.53463] [0.0386  0.53707] outlier residuals: [ 0.39614 -0.347    0.23626] oracle inlier residual RMS: 0.001390169271931344
20 relative amplitude err robust/oracle/1sigma: [0.00962 0.0116  0.01468] edge truth/oracle xy: [0.57894 0.965  ] [0.57863 0.96751] outlier residuals: [-0.15447  0.37217  0.13687] oracle inlier residual RMS: 0.001070689915357443
24 relative amplitude err robust/oracle/1sigma: [-0.049   -0.04312  0.05593] edge truth/oracle xy: [0.035   0.87015] [0.04039 0.86746] outlier residuals: [ 0.11856 -0.1601   0.40364] oracle inlier residual RMS: 0.0014800494899901958
28 relative amplitude err robust/oracle/1sigma: [0.49883 0.04595 0.10864] edge truth/oracle xy: [0.10734 0.965  ] [0.10384 0.97006] outlier residuals: [-0.16222 -0.32137  0.24809] oracle inlier residual RMS: 0.0014630556566346974
32 relative amplitude err robust/oracle/1sigma: [0.0375  0.04099 0.18213] edge truth/oracle xy: [0.035   0.23891] [0.03004 0.2379 ] outlier residuals: [-0.30081  0.21711 -0.44905] oracle inlier residual RMS: 0.0016289513633003315

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7a1ff1186d03c683 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7a1ff118 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`25061 ms` LLM)
- **Rationale**: Rerun the same public, sensor-deletion, and synthetic slices to test whether rejection fixes trial 28 without regressing well-resolved cases. Trial 12's oracle error distinguishes a remaining noise limit from a failure to reject gross readings.
- **Alternatives Weighed**: `Clamp the source away from the wall or impose an amplitude prior` (rejected: Neither constraint is supported by the task; either could bias valid edge sources.)
- **ATMS NOGOOD DECLARED**: Refuted `Cauchy downweighting alone adequately suppresses gross-error bias in the weakly constrained edge regime.` | Witness: `Trial 28 had 49.9% amplitude error under Cauchy versus 4.6% after removing known outliers. In trial 12 the oracle still missed by 15.3%.` | Blanket Diagnosis: `The independent flux check supports the specified basis and Robin operator. The oracle comparison separates residual outlier influence from the amplitude–position ambiguity that persists with clean sensors.` | Sublated Pivot: `Keep the conservative forward model and robust spatial search, then exclude only readings whose residuals clearly separate from the estimated inlier noise.`

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/thermal_sink_solver.py')
s = p.read_text()
old = '''    xs, ys, a, b = winner.x
'''
new = '''    # Once localized, remove clearly separated stuck readings. Even the
    # small remaining Cauchy influence of a gross outlier can bias the
    # weakly constrained amplitude and wall-normal position together.
    r = winner.fun
    inlier = np.abs(r - np.median(r)) <= 2.4 * cutoff(r)
    if 4 < np.count_nonzero(inlier) < len(y):
        refined = least_squares(lambda q: evaluate(q)[0][inlier], winner.x,
                                jac=lambda q: evaluate(q)[1][inlier],
                                bounds=(lo, hi), x_scale='jac',
                                loss='huber', f_scale=cutoff(r),
                                ftol=1.e-10, max_nfev=90)
        winner = refined
    xs, ys, a, b = winner.x
'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/check_thermal_sink.py
```
- **Container Exit Code**: `1` (`5715 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/thermal_sink_solver.py']`
- **Stdout (tail)**:
```text
FAIL synthetic edge (12, 18, 2.2, 0.085) [0.01781963 0.15335301 0.00031868 0.05178603] {'source_xy': [0.8202669909847655, 0.9815133905712811], 'amplitude': 29.090757835449885, 'junction_offset': 0.03436428095785112} {'source_xy': array([0.8135702, 0.965    ]), 'amplitude': 25.222770049753095, 'junction_offset': 0.034045605898185324}
public n= 2 failed= 0 worst (position, relative amplitude, offset, seconds): [7.46940201e-04 4.24396852e-04 3.25170970e-05 4.20048651e-02]
public leave-one-sensor-out n= 104 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00138615 0.0031111  0.00032274 0.0479429 ]
synthetic edge n= 9 failed= 1 worst (position, relative amplitude, offset, seconds): [0.01781963 0.15335301 0.00125911 0.0738836 ]
synthetic interior n= 27 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00592667 0.0279372  0.00256996 0.07453885]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c9519a067244faf3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c9519a06 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`42782 ms` LLM)
- **Rationale**: Known-inlier refitting eliminated the gross-outlier bias in trial 28. The remaining miss matches the oracle, so test whether summarizing the likelihood improves amplitude recovery across independently randomized wall distances, including sources closer to the wall. Keep the current artifact unchanged unless the challenger improves that slice without harming public recovery.
- **Alternatives Weighed**: `Clamp the source away from a wall` (rejected: That excludes allowed locations and would tailor the result to the fixed wall distance in the first synthetic suite.)

```bash
cp /holdout_task/thermal_sink_solver.py /tmp/thermal_edge_candidate.py
python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/thermal_edge_candidate.py')
s = p.read_text()
s = s.replace('    r = winner.fun\n    inlier', '    r = winner.fun\n    inlier')
s = s.replace('        winner = refined\n    xs, ys, a, b = winner.x', '''        winner = refined
    else:
        inlier = np.ones(len(y), dtype=bool)
    xs, ys, a, b = winner.x

    # On a weakly constrained wall-normal ridge, summarize likelihood over
    # physical source positions instead of selecting the lowest-residual point.
    residual, jac = evaluate(winner.x)
    j = jac[inlier]
    sd = np.linalg.norm(residual[inlier]) / np.sqrt(max(len(j)-4, 1))
    cov = sd**2 * np.linalg.pinv(j.T @ j, rcond=1e-13)
    pos_sd = np.sqrt(np.maximum(np.diag(cov)[:2], 0.0))
    xy = np.array([xs, ys])
    if sd > floor and np.any(np.minimum(xy, 1-xy) < 2.5*pos_sd):
        radius = np.clip(6*pos_sd, 0.008, 0.40)
        low, high = np.maximum(0., xy-radius), np.minimum(1., xy+radius)
        gx, gy = (np.linspace(l, h, 81) for l,h in zip(low,high))
        xx = np.exp(-0.5*((gx[:,None]-centers)/sigma)**2)
        yy = np.exp(-0.5*((gy[:,None]-centers)/sigma)**2)
        # Evaluate the separable Gaussian without a dense (81**2)-by-N**2 array.
        gi = green[inverse[inlier]].reshape(-1,n,n)
        f = ((xx @ gi) @ yy.T).reshape(len(gi),-1).T
        data = y[inlier]
        fm = f.mean(axis=1)
        fc = f-fm[:,None]
        norm2 = np.einsum('ij,ij->i',fc,fc)
        amp = (fc @ (data-data.mean()))/np.maximum(norm2, np.finfo(float).tiny)
        off = data.mean()-amp*fm
        rss = np.sum((f*amp[:,None]+off[:,None]-data)**2,axis=1)
        logw = -0.5 * rss/(sd*sd)
        logw[(amp <= 0.) | (norm2 <= 0.)] = -np.inf
        w = np.exp(logw-np.max(logw)).reshape(len(gx),len(gy))
        w[[0,-1],:] *= 0.5
        w[:,[0,-1]] *= 0.5
        w /= w.sum()
        xs, ys = w.sum(axis=1) @ gx, w.sum(axis=0) @ gy
        a = np.exp(w.ravel() @ np.log(np.maximum(amp,np.finfo(float).tiny)))/amplitude_scale
        b = w.ravel() @ off''')
p.write_text(s)
PY
sed "s|path+'thermal_sink_solver.py'|'/tmp/thermal_edge_candidate.py'|" /tmp/check_thermal_sink.py > /tmp/check_thermal_candidate.py
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -I /tmp/check_thermal_candidate.py
for variant in baseline candidate; do
  if [ "$variant" = baseline ]; then input=/tmp/check_thermal_sink.py; else input=/tmp/check_thermal_candidate.py; fi
  sed -e '/^for case_id, c in enumerate(cases):/,/^for trial in range(36):/{ /^for trial in range(36):/!d; }' -e 's/default_rng(84)/default_rng(1703)/' -e 's/range(36):/range(120):/' -e 's/    if trial % 4 == 0:/    if trial % 2 == 0:/' -e 's/\.035 if trial % 8 == 0 else \.965/rng.uniform(0., .12) if trial % 8 < 4 else rng.uniform(.88, 1.)/' -e 's/if trial%4 == 0 else/if trial%2 == 0 else/' "$input" > "/tmp/check_thermal_unseen_$variant.py"
  log="/tmp/check_thermal_unseen_$variant.log"
  python3 -I "/tmp/check_thermal_unseen_$variant.py" > "$log"
  echo "$variant randomized wall distances"
  tail -n 2 "$log"
done
```
- **Container Exit Code**: `0` (`19849 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
FAIL synthetic edge (12, 18, 2.2, 0.085) [0.01102838 0.09932661 0.00031728 0.05437692] {'source_xy': [0.8177314931202316, 0.9752131631602993], 'amplitude': 27.72806224452013, 'junction_offset': 0.03436288290522547} {'source_xy': array([0.8135702, 0.965    ]), 'amplitude': 25.222770049753095, 'junction_offset': 0.034045605898185324}
FAIL synthetic edge (28, 22, 0.45, 0.06) [0.01020243 0.10025679 0.0011645  0.08035866] {'source_xy': [0.10239227910284054, 0.9739246553956461], 'amplitude': 32.676242880175934, 'junction_offset': 0.07238493850003887} {'source_xy': array([0.10733597, 0.965     ]), 'amplitude': 29.698742363322886, 'junction_offset': 0.07122044010124672}
public n= 2 failed= 0 worst (position, relative amplitude, offset, seconds): [7.46940201e-04 4.24396852e-04 3.25170970e-05 4.37586920e-02]
public leave-one-sensor-out n= 104 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00138615 0.0031111  0.00032274 0.06214574]
synthetic edge n= 9 failed= 2 worst (position, relative amplitude, offset, seconds): [0.01102838 0.10025679 0.00125911 0.08035866]
synthetic interior n= 27 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00592667 0.0279372  0.00256996 0.07551979]
baseline randomized wall distances
synthetic edge n= 60 failed= 15 worst (position, relative amplitude, offset, seconds): [0.08025166 1.47311942 0.00305733 0.12657345]
synthetic interior n= 60 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00480409 0.01411043 0.00327838 0.09884671]
candidate randomized wall distances
synthetic edge n= 60 failed= 11 worst (position, relative amplitude, offset, seconds): [0.06911546 0.31227997 0.00305733 0.13583794]
synthetic interior n= 60 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00480409 0.01411043 0.00327838 0.10074792]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c9519a067244faf3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c9519a06 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`28737 ms` LLM)
- **Rationale**: Compare each remaining randomized edge miss with a known-inlier fit and its noise scale. If the oracle succeeds where the challenger fails, revise the averaging rule; if the oracle also misses along a broad ridge, quantify that limitation rather than adding unsupported constraints.
- **Alternatives Weighed**: `Tune the boundary averaging to the two fixed-distance misses` (rejected: Those misses do not establish a transferable prior on source location or amplitude.)

```bash
echo 'Boundary likelihood candidate misses:'; grep '^FAIL' /tmp/check_thermal_unseen_candidate.log; python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/check_thermal_unseen_candidate.py')
s=p.read_text()
a=s.index("    check('synthetic edge'")
s=s[:a]+'''    if trial % 2:
        continue
    from scipy.optimize import least_squares
    m=dict(N=n,kappa=k,gamma_robin=gamma,sigma_m=sigma,sensor_ij=np.column_stack(divmod(ids,n)),T_obs=yy)
    got=solver.localize_thermal_source(m)
    p=np.r_[got['source_xy'],got['amplitude'],got['junction_offset']]
    if np.linalg.norm(p[:2]-xy)<=.035 and abs(p[2]/a-1)<=.08:
        continue
    good = ~np.isin(np.arange(len(ids)), outliers)
    rhs=np.zeros((n*n,len(ids)))
    rhs[ids,np.arange(len(ids))]=1/n**2
    green=solver._factor_operator(k,gamma).solve(rhs).T
    def ev(p):
        dx,dy=x-p[0],y-p[1]
        g=np.exp(-.5*(dx**2+dy**2)/sigma**2)
        d=green@np.stack([p[2]*g*dx/sigma**2,p[2]*g*dy/sigma**2,g],axis=-1).reshape(-1,3)
        return p[2]*d[:,2]+p[3]-yy,np.column_stack((d,np.ones(len(ids))))
    truth=np.r_[xy,a,b]
    fit=least_squares(lambda p:ev(p)[0][good],truth,jac=lambda p:ev(p)[1][good],bounds=([0,0,0,-np.inf],[1,1,np.inf,np.inf]),gtol=1e-13)
    r,j=ev(fit.x)
    sd=np.sqrt(np.diag(np.linalg.pinv(j[good].T@j[good],rcond=1e-13)))*.0015
    print(trial, 'truth', np.round(truth,5), 'estimate',np.round(p,5), 'oracle',np.round(fit.x,5),'oracle-truth noise-normalized delta RSS',round((np.sum(ev(truth)[0][good]**2)-np.sum(r[good]**2))/.0015**2,3),'oracle sds',np.round(sd,5),'outlier residual min',round(np.min(np.abs(ev(p)[0][~good])),5))
'''
Path('/tmp/diagnose_unseen_edges.py').write_text(s)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/diagnose_unseen_edges.py
```
- **Container Exit Code**: `0` (`4550 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
607  0.04768] oracle [ 0.44952  0.97501 13.45608  0.04768] oracle-truth noise-normalized delta RSS 3.668 oracle sds [2.2000e-03 7.6200e-03 7.1878e-01 4.3000e-04] outlier residual min 0.28835
56 truth [ 0.04179  0.34547 15.31864 -0.03303] estimate [ 1.554000e-02  3.460600e-01  1.964705e+01 -3.307000e-02] oracle [ 0.       0.34568 23.55    -0.03316] oracle-truth noise-normalized delta RSS 5.75 oracle sds [3.404000e-02 2.410000e-03 1.096943e+01 1.210000e-03] outlier residual min 0.20512
62 truth [ 0.10599  0.89039 19.37456 -0.05729] estimate [ 0.13185  0.85758 15.28096 -0.05735] oracle [ 0.13185  0.85758 15.28101 -0.05735] oracle-truth noise-normalized delta RSS 1.751 oracle sds [2.45400e-02 2.45200e-02 2.90745e+00 3.80000e-04] outlier residual min 0.10646
86 truth [ 0.08805  0.93257 14.32798  0.01658] estimate [7.503000e-02 9.468100e-01 1.880232e+01 1.645000e-02] oracle [4.45300e-02 1.00000e+00 3.54348e+01 1.64200e-02] oracle-truth noise-normalized delta RSS 1.229 oracle sds [4.6710000e-01 8.9739000e-01 5.8961107e+02 5.0000000e-04] outlier residual min 0.18842
88 truth [ 0.07233  0.18343 19.0348   0.0512 ] estimate [ 0.05985  0.18346 21.66313  0.05116] oracle [ 0.07191  0.18633 18.93549  0.05114] oracle-truth noise-normalized delta RSS 0.498 oracle sds [4.21800e-02 1.59500e-02 5.26113e+00 4.30000e-04] outlier residual min 0.13156
90 truth [ 2.140000e-03  7.837600e-01  2.037699e+01 -6.909000e-02] estimate [ 1.363000e-02  7.835000e-01  1.837113e+01 -6.872000e-02] oracle [ 1.140000e-02  7.835800e-01  1.867933e+01 -6.871000e-02] oracle-truth noise-normalized delta RSS 0.941 oracle sds [1.026e-02 1.500e-03 1.689e+00 1.160e-03] outlier residual min 0.15563
102 truth [ 0.13443  0.93517 30.97642 -0.06367] estimate [ 0.18372  0.88672 21.74163 -0.0657 ] oracle [ 0.13849  0.92947 29.80316 -0.06418] oracle-truth noise-normalized delta RSS 4.531 oracle sds [4.17000e-03 5.40000e-03 1.17705e+00 4.90000e-04] outlier residual min 0.00269
114 truth [3.950000e-03 7.310000e-01 3.100922e+01 6.128000e-02] estimate [ 0.03089  0.73696 21.90726  0.06134] oracle [ 0.03074  0.73695 21.63054  0.06133] oracle-truth noise-normalized delta RSS 3.135 oracle sds [1.74300e-02 4.49000e-03 4.27992e+00 4.10000e-04] outlier residual min 0.3444

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c9519a067244faf3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c9519a06 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`32502 ms` LLM)
- **Rationale**: Trial 102 shows that the data can support recovery, but one stuck reading had a residual of only 0.00269 at the fitted solution. Test whether fitting the clean majority before setting the rejection threshold recovers that case without regressing public and sensor-deletion results.
- **Alternatives Weighed**: `Tune boundary averaging to the failing cases` (rejected: Most misses persist with known inliers, but trial 102 does not. Test the recoverable outlier failure without imposing a prior on wall distance or amplitude.)

```bash
python3 - <<'PY'
from pathlib import Path
s=Path('/tmp/thermal_edge_candidate.py').read_text()
a=s.index('    # Once localized, remove clearly separated stuck readings.')
b=s.index('    xs, ys, a, b = winner.x',a)
s=s[:a]+'''    # Concentrate the fit on the expected clean majority before estimating
    # the noise scale. A stuck sensor can otherwise partly mask its own
    # residual by pulling a weakly constrained fit toward itself.
    keep = max(5, len(y)-max(1, round(0.06*len(y))))

    def subset_fit(p, mask, loss='linear', scale=1.):
        return least_squares(lambda q: evaluate(q)[0][mask], p,
                             jac=lambda q: evaluate(q)[1][mask],
                             bounds=(lo, hi), x_scale='jac', loss=loss,
                             f_scale=scale, ftol=1.e-10, max_nfev=90)

    if keep < len(y):
        def concentrate(start):
            p, r = start.x, start.fun
            last = None
            for _ in range(6):
                ids = np.sort(np.argpartition(np.abs(r), keep-1)[:keep])
                if last is not None and np.array_equal(ids,last):
                    break
                part = subset_fit(p, ids)
                p, r = part.x, evaluate(part.x)[0]
                last = ids
            return np.partition(r*r, keep-1)[:keep].sum(), p, r

        _, p, r = min((concentrate(f) for f in [winner, *sorted(fits, key=lambda f:f.cost)[:3]]),
                      key=lambda item: item[0])
        inlier = np.abs(r-np.median(r)) <= 2.0*cutoff(r)
        if np.count_nonzero(inlier) <= 4:
            inlier = np.ones(len(y), dtype=bool)
        winner = subset_fit(p, inlier, 'huber', cutoff(r))
    else:
        inlier = np.ones(len(y), dtype=bool)
''' + s[b:]
Path('/tmp/thermal_concentrated.py').write_text(s)
PY
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
sed 's|/tmp/thermal_edge_candidate.py|/tmp/thermal_concentrated.py|' /tmp/check_thermal_candidate.py > /tmp/check_thermal_concentrated.py
sed 's|/tmp/thermal_edge_candidate.py|/tmp/thermal_concentrated.py|' /tmp/check_thermal_unseen_candidate.py > /tmp/check_thermal_unseen_concentrated.py
python3 -I /tmp/check_thermal_concentrated.py > /tmp/thermal_concentrated_fixed.log; tail -n 5 /tmp/thermal_concentrated_fixed.log
python3 -I /tmp/check_thermal_unseen_concentrated.py > /tmp/thermal_concentrated_unseen.log; grep '^FAIL.*(102,' /tmp/thermal_concentrated_unseen.log || true; tail -n 2 /tmp/thermal_concentrated_unseen.log
```
- **Container Exit Code**: `0` (`14096 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
FAIL synthetic edge (28, 22, 0.45, 0.06) [0.01020243 0.10025679 0.0011645  0.14190078] {'source_xy': [0.1023922791538929, 0.9739246553058565], 'amplitude': 32.67624286855184, 'junction_offset': 0.0723849384932182} {'source_xy': array([0.10733597, 0.965     ]), 'amplitude': 29.698742363322886, 'junction_offset': 0.07122044010124672}
public n= 2 failed= 0 worst (position, relative amplitude, offset, seconds): [7.46940201e-04 4.24397503e-04 3.25169384e-05 7.05739069e-02]
public leave-one-sensor-out n= 104 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00138615 0.0031111  0.00032274 0.08196127]
synthetic edge n= 9 failed= 2 worst (position, relative amplitude, offset, seconds): [0.01102838 0.10025679 0.00125911 0.14190078]
synthetic interior n= 27 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00592668 0.02784499 0.00256996 0.0919873 ]
FAIL synthetic edge (102, 18, 2.2, 0.12) [0.06855033 0.29609416 0.00228004 0.05439444] {'source_xy': [0.1813984234320183, 0.885240022750366], 'amplitude': 21.804480951195007, 'junction_offset': -0.0659532829925331} {'source_xy': array([0.13442524, 0.93516665]), 'amplitude': 30.97641709490471, 'junction_offset': -0.06367324353930075}
synthetic edge n= 60 failed= 11 worst (position, relative amplitude, offset, seconds): [0.06855033 0.31227997 0.00305412 0.17484474]
synthetic interior n= 60 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00479838 0.01411043 0.00328789 0.09658196]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c9519a067244faf3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c9519a06 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`28245 ms` LLM)
- **Rationale**: Test whether a small search over sensor membership recovers the known-inlier solution on trial 102 without degrading public accuracy or runtime. If the trimmed objective still prefers the wrong subset, do not promote the added search.
- **Alternatives Weighed**: `Keep adding spatial seeds` (rejected: Spatial multistart did not resolve trial 102; a sensor-membership change directly tests whether a retained reading is masking its own error.)

```bash
python3 - <<'PY'
from pathlib import Path
s = Path('/tmp/thermal_concentrated.py').read_text()
a = s.index('        def concentrate(start):')
b = s.index('        inlier =', a)
s = s[:a] + '''        def concentrate(p, r):
            last = None
            for _ in range(8):
                ids = np.sort(np.argpartition(np.abs(r), keep-1)[:keep])
                if last is not None and np.array_equal(ids, last):
                    break
                part = subset_fit(p, ids)
                p, r = part.x, evaluate(part.x)[0]
                last = ids
            return np.partition(r*r, keep-1)[:keep].sum(), p, r

        best = min((concentrate(f.x, f.fun) for f in [winner, *sorted(fits, key=lambda f: f.cost)[:3]]),
                   key=lambda item: item[0])
        _, p, r = best
        # A retained reading can mask its own error by steering a wall fit.
        # Try swapping out that reading and let every original reading compete
        # again for inclusion after the initial deletion fit.
        if np.min(np.minimum(p[:2], 1-p[:2])) < 2*sigma:
            ids = np.sort(np.argpartition(np.abs(r), keep-1)[:keep])
            residual, jac = evaluate(p)
            u, singular, _ = np.linalg.svd(jac[ids], full_matrices=False)
            leverage = np.sum(u[:, singular > singular[0]*1e-10]**2, axis=1)
            for drop in np.argsort(leverage)[-min(12, keep):]:
                part = subset_fit(p, np.delete(ids, drop))
                trial = concentrate(part.x, evaluate(part.x)[0])
                if trial[0] < best[0]:
                    best = trial
            _, p, r = best
''' + s[b:]
Path('/tmp/thermal_swap.py').write_text(s)
PY
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for suite in concentrated unseen_concentrated; do
  sed 's|/tmp/thermal_concentrated.py|/tmp/thermal_swap.py|' "/tmp/check_thermal_${suite}.py" > "/tmp/check_thermal_swap_${suite}.py"
  python3 -I "/tmp/check_thermal_swap_${suite}.py" > "/tmp/thermal_swap_${suite}.log"
  echo "$suite"; grep '^FAIL.*(102,' "/tmp/thermal_swap_${suite}.log" || true; tail -n 4 "/tmp/thermal_swap_${suite}.log"
done
```
- **Container Exit Code**: `0` (`15688 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
concentrated
public n= 2 failed= 0 worst (position, relative amplitude, offset, seconds): [7.46940201e-04 4.24397503e-04 3.25169384e-05 4.33883001e-02]
public leave-one-sensor-out n= 104 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00138615 0.0031111  0.00032274 0.06668374]
synthetic edge n= 9 failed= 2 worst (position, relative amplitude, offset, seconds): [0.01102838 0.10025679 0.00125911 0.15565369]
synthetic interior n= 27 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00592668 0.02784499 0.00256996 0.10409419]
unseen_concentrated
FAIL synthetic edge (90, 18, 0.45, 0.085) [0.01148615 0.09843752 0.00037462 0.09896924] {'source_xy': [0.013625527934280054, 0.7835036482275043], 'amplitude': 18.371129354767447, 'junction_offset': -0.06871812395107707} {'source_xy': array([0.0021422 , 0.78375827]), 'amplitude': 20.376989751947974, 'junction_offset': -0.0690927425398792}
FAIL synthetic edge (114, 18, 5.0, 0.06) [2.75871173e-02 2.93524391e-01 6.58533544e-05 7.34165760e-02] {'source_xy': [0.030888441270453017, 0.7369555665548367], 'amplitude': 21.90725999125201, 'junction_offset': 0.06134119572461311} {'source_xy': array([0.00395093, 0.73100415]), 'amplitude': 31.009223414407018, 'junction_offset': 0.061275342370208055}
synthetic edge n= 60 failed= 10 worst (position, relative amplitude, offset, seconds): [0.04177103 0.31227997 0.00305412 0.26676491]
synthetic interior n= 60 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00479838 0.01411043 0.00328789 0.09591584]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c9519a067244faf3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c9519a06 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`29677 ms` LLM)
- **Rationale**: Promote the challenger because it fixes a recoverable sensor-masking failure and reduces worst-case edge error while retaining public and interior accuracy. Verify the exact deliverable in isolated Python, checking public runtime with default threading and reporting the remaining noisy edge misses separately.
- **Alternatives Weighed**: `Keep the simpler outlier-rejected point fit` (rejected: It preserves public accuracy but has a substantially worse randomized edge tail and misses a case recoverable by changing the retained sensor subset.); `Tune a source-amplitude or wall-distance prior` (rejected: The task supplies no such prior, and several remaining misses persist even when the injected outliers are known.)

```bash
set -e
cp /tmp/thermal_swap.py /holdout_task/thermal_sink_solver.py
python3 -I - <<'PY'
import importlib.util, math, pickle, time
from pathlib import Path
p = Path('/holdout_task/thermal_sink_solver.py')
spec = importlib.util.spec_from_file_location(p.stem, p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with p.with_name('public_thermal_cases.pkl').open('rb') as f:
    cases = pickle.load(f)
for i, c in enumerate(cases):
    t = time.perf_counter()
    v = m.localize_thermal_source(c['measurement'])
    elapsed = time.perf_counter()-t
    g = c['gt']
    assert set(v) == {'source_xy', 'amplitude', 'junction_offset'}
    assert type(v['source_xy']) is list and len(v['source_xy']) == 2
    assert all(type(x) is float and math.isfinite(x) for x in (*v['source_xy'], v['amplitude'], v['junction_offset']))
    assert v['amplitude'] > 0 and all(0 <= x <= 1 for x in v['source_xy'])
    err = (math.dist(v['source_xy'], g['source_xy']), abs(v['amplitude']/g['amplitude']-1), abs(v['junction_offset']-g['junction_offset']), elapsed)
    print('Public case', i, 'position / relative amplitude / offset / seconds:', err, flush=True)
    assert all(e <= lim for e, lim in zip(err[:3], (.035,.08,.008))) and elapsed < 15
PY
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for suite in concentrated unseen_concentrated; do
  sed "s|'/tmp/thermal_swap.py'|'/holdout_task/thermal_sink_solver.py'|" "/tmp/check_thermal_swap_${suite}.py" > "/tmp/check_thermal_final_${suite}.py"
  log="/tmp/thermal_final_${suite}.log"
  status=0
  python3 -I "/tmp/check_thermal_final_${suite}.py" > "$log" || status=$?
  echo "$suite (suite exit $status)"
  grep ' n= ' "$log"
  grep -q '^synthetic interior .*failed= 0 ' "$log"
  if [ "$suite" = concentrated ]; then
    grep -q '^public leave-one-sensor-out .*failed= 0 ' "$log"
  else
    ! grep -q '^FAIL.*(102,' "$log" || exit 1
  fi
done
```
- **Container Exit Code**: `0` (`17720 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/thermal_sink_solver.py']`
- **Stdout (tail)**:
```text
Public case 0 position / relative amplitude / offset / seconds: (0.0004366878951347448, 0.00042439750254275044, 3.251693840805592e-05, 0.8218594039790332)
Public case 1 position / relative amplitude / offset / seconds: (0.0007469402012306971, 0.00021516133952792504, 1.7993567575626257e-05, 0.7936359760351479)
concentrated (suite exit 1)
public n= 2 failed= 0 worst (position, relative amplitude, offset, seconds): [7.46940201e-04 4.24397503e-04 3.25169384e-05 5.09523561e-02]
public leave-one-sensor-out n= 104 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00138615 0.0031111  0.00032274 0.05628213]
synthetic edge n= 9 failed= 2 worst (position, relative amplitude, offset, seconds): [0.01102838 0.10025679 0.00125911 0.17765573]
synthetic interior n= 27 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00592668 0.02784499 0.00256996 0.10205267]
unseen_concentrated (suite exit 1)
synthetic edge n= 60 failed= 10 worst (position, relative amplitude, offset, seconds): [0.04177103 0.31227997 0.00305412 0.26344679]
synthetic interior n= 60 failed= 0 worst (position, relative amplitude, offset, seconds): [0.00479838 0.01411043 0.00328789 0.09739518]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=68c6d2876043a243 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_68c6d287 (directory).
```

### Turn 11 — `submit_completed_task` (`14937 ms` LLM)
- **Rationale**: The exact submitted artifact passed isolated-Python checks for public accuracy, output schema, and runtime. Sensor swaps fixed the demonstrated outlier-masking failure without regressing public or interior recovery. Remaining synthetic edge misses are reported rather than hidden behind unsupported constraints.
- **Alternatives Weighed**: `Constrain source amplitude or wall distance` (rejected: The task supplies no such prior, and several remaining edge misses persist even when the fit uses only known inliers.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`17` turns remaining; artifact status={'/holdout_task/thermal_sink_solver.py': 'FILE:8964'}): Do NOT lock prematurely with `17` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/thermal_sink_solver.py`. Both public cases passed all tolerances in under 0.83 s each, and all 104 leave-one-sensor-out fits passed. The amplitude tolerance is not guaranteed for noisy sources near a wall, where changes in source position and amplitude can produce nearly indistinguishable readings.`

### Turn 12 — `execute_in_remote_docker` (`43145 ms` LLM)
- **Rationale**: Independent face enumeration would expose a missing boundary or conductance term. Offset-free pairwise contrasts test the relevant algebraic symmetry on real readings. A synthetic near-wall pair on those same real sensor layouts tests whether amplitude differences beyond tolerance remain distinguishable at the earlier stress noise level; it informs whether to pursue another estimator change or report the resolution limit.
- **Alternatives Weighed**: `Apply relay, drift-PDE, or batch-centering checks` (rejected: The documented inputs describe a steady elliptic inverse problem with a common offset, not relay modes, drift dynamics, or labeled technical batches.)

```bash
cat > /tmp/audit_thermal_margins.py <<'PY'
import importlib.util
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

path = Path('/holdout_task/thermal_sink_solver.py')
spec = importlib.util.spec_from_file_location(path.stem, path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with path.with_name('public_thermal_cases.pkl').open('rb') as f:
    cases = pickle.load(f)

for i, case in enumerate(cases):
    m, gt = case['measurement'], case['gt']
    n, k = m['N'], m['kappa']
    sigma, gamma = m['sigma_m'], m['gamma_robin']
    ij = m['sensor_ij']
    ids = n*ij[:, 0]+ij[:, 1]
    data = m['T_obs']
    x, y = (np.indices((n, n))+.5)/n
    # Independently enumerate all four faces: corner cells have two sinks.
    matrix = lil_matrix((n*n, n*n))
    for a, b in np.ndindex(n, n):
        row = a*n+b
        for c, d in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
            if 0 <= c < n and 0 <= d < n:
                w = 2/(1/k[a,b]+1/k[c,d])
                matrix[row, row] += w
                matrix[row, n*c+d] -= w
            else:
                matrix[row, row] += gamma/n
    independent = splu(matrix.tocsc())
    package = solver._factor_operator(k, gamma)
    def basis(xy):
        dx, dy = x-xy[0], y-xy[1]
        src = np.exp(-.5*(dx*dx+dy*dy)/sigma**2)/n**2
        return np.stack((src, src*dx/sigma**2, src*dy/sigma**2), axis=-1).reshape(n*n,3)

    q = basis(gt['source_xy'])
    states = independent.solve(q)
    assert np.max(np.abs(states-package.solve(q))) < 1e-12
    flux_total = gamma/n*(states[:,0].reshape(n,n)[[0,-1],:].sum()+states[:,0].reshape(n,n)[:,[0,-1]].sum())
    assert abs(q[:,0].sum()-flux_total) < 1e-12
    fit = solver.localize_thermal_source(m)
    a, b = fit['amplitude'], fit['junction_offset']
    f, fx, fy = independent.solve(basis(fit['source_xy']))[ids].T
    r = a*f+b-data
    med = np.median(r)
    mad = 1.4826*np.median(abs(r-med))
    good = abs(r-med) <= 6*mad
    assert good.sum() > 4
    rc = r[good]-r[good].mean()
    d = r[good, None]-r[None, good]
    assert np.isclose(np.sum(rc**2), np.sum(d*d)/(2*good.sum()), rtol=1e-12, atol=1e-15)
    amp_contrast = np.dot(f[good]-f[good].mean(),data[good]-data[good].mean())/np.var(f[good])/good.sum()
    jac = np.column_stack((a*fx,a*fy,a*f,np.ones_like(f)))[good]
    sd = np.sqrt(np.diag(np.linalg.inv(jac.T@jac))) * np.linalg.norm(r[good])/np.sqrt(good.sum()-4)
    errs = np.array([np.linalg.norm(np.subtract(fit['source_xy'],gt['source_xy'])),abs(a/gt['amplitude']-1),abs(b-gt['junction_offset'])])
    print('public', i, 'excluded', int((~good).sum()), 'inlier MAD scale', mad, 'contrast-only relative amp delta', abs(amp_contrast/a-1), 'local parameter SDs (x, y, log A, offset)', sd, 'fraction of tolerance', errs/[.035,.08,.008])
    assert np.all(errs <= [.035,.08,.008])

    # Same real plate and sensor layout, but a synthetic source pair close to
    # a wall. Quantify separability for the noise used in the prior edge stress.
    q1 = independent.solve(basis((.006,.74))[:,0])[ids]
    q2 = independent.solve(basis((.025,.74))[:,0])[ids]
    keep = np.argsort(q1)[:-round(.06*len(ids))]
    q1, q2 = q1[keep], q2[keep]
    a1 = 24.
    a2 = a1*np.dot(q1-q1.mean(), q2-q2.mean())/np.sum((q2-q2.mean())**2)
    delta = a1*q1-a2*q2
    separation = np.linalg.norm(delta-delta.mean())/.0015
    print('layout edge pair', i, 'amplitudes', (a1,a2), '8%-tolerance intervals disjoint', max(a1,a2)*.92 > min(a1,a2)*1.08, 'joint separation / noise SD', separation)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/audit_thermal_margins.py
```
- **Container Exit Code**: `0` (`648 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0 excluded 3 inlier MAD scale 0.0013952265028797488 contrast-only relative amp delta 1.2490009027033011e-12 local parameter SDs (x, y, log A, offset) [0.00086718 0.00062632 0.00303288 0.0004818 ] fraction of tolerance [0.0124768  0.00530497 0.00406462]
layout edge pair 0 amplitudes (24.0, 20.195210563700897) 8%-tolerance intervals disjoint True joint separation / noise SD 1.4804237961818358
public 1 excluded 3 inlier MAD scale 0.0011660535733099364 contrast-only relative amp delta 1.531441640167941e-12 local parameter SDs (x, y, log A, offset) [0.00066594 0.00072294 0.00235587 0.00039754] fraction of tolerance [0.02134115 0.00268952 0.0022492 ]
layout edge pair 1 amplitudes (24.0, 20.3158367966241) 8%-tolerance intervals disjoint True joint separation / noise SD 2.729602673995727

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=68c6d2876043a243 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_68c6d287 (directory).
```

### Turn 13 — `submit_completed_task` (`15675 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification. The follow-up audit confirmed the Robin flux balance and showed that offset-free contrasts reproduce the fitted amplitudes on both public cases. Near-wall tests on the public sensor layouts showed that some amplitudes cannot be reliably distinguished at the tested noise level; no further estimator change was justified.
- **Alternatives Weighed**: `Impose a stronger amplitude or wall-distance prior` (rejected: The input contract supplies neither prior, and the same sensor layouts can produce nearly indistinguishable readings for amplitudes with disjoint tolerance intervals.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`15` turns remaining; artifact status={'/holdout_task/thermal_sink_solver.py': 'FILE:8964'}): Do NOT lock prematurely with `15` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/thermal_sink_solver.py`. Both public cases passed all tolerances in under 0.83 s each. All 104 leave-one-sensor-out fits and all 87 synthetic interior cases also passed.

Limitation: 12 of 69 synthetic near-wall cases missed at least one tolerance. Near a wall, changes in source position and amplitude can produce nearly indistinguishable thermistor readings.`

### Turn 14 — `execute_in_remote_docker` (`35124 ms` LLM)
- **Rationale**: Predict groups of real readings excluded from fitting to detect sensor-specific bias that parameter-only checks could miss. Time the more expensive near-wall branch with default threading on both public layouts. A held-out prediction failure would warrant inspecting sensor selection; a runtime failure would warrant changing the contraction implementation rather than the physical model.
- **Alternatives Weighed**: `Repeat the operator and offset-contrast audit` (rejected: The independent flux and offset-contrast checks already passed; repeating them would not test held-out temperatures or near-wall runtime with default threading.)

```bash
cat > /tmp/audit_thermal_prediction.py <<'PY'
import importlib.util
import pickle
import time
from pathlib import Path
import numpy as np

path = Path('/holdout_task/thermal_sink_solver.py')
spec = importlib.util.spec_from_file_location(path.stem, path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with path.with_name('public_thermal_cases.pkl').open('rb') as f:
    cases = pickle.load(f)

for i, case in enumerate(cases):
    m, truth = case['measurement'], case['gt']
    n, sigma = m['N'], m['sigma_m']
    ii, jj = (np.indices((n, n)) + .5) / n
    factor = solver._factor_operator(m['kappa'], m['gamma_robin'])
    sensor = tuple(m['sensor_ij'].T)
    y = m['T_obs']
    def field(xy, a):
        q = (a/n**2) * np.exp(-((ii-xy[0])**2 + (jj-xy[1])**2)/(2*sigma**2))
        return factor.solve(q.ravel()).reshape(n,n)[sensor]
    order = np.lexsort(m['sensor_ij'].T[::-1])
    fold_ids = np.empty(len(y), dtype=int)
    fold_ids[order] = np.arange(len(y)) % 4
    prediction = np.empty_like(y)
    worst = np.zeros(3)
    for fold in range(4):
        keep = fold_ids != fold
        p = solver.localize_thermal_source(dict(m, sensor_ij=m['sensor_ij'][keep], T_obs=y[keep]))
        prediction[~keep] = (field(p['source_xy'], p['amplitude']) + p['junction_offset'])[~keep]
        worst = np.maximum(worst, [np.linalg.norm(np.subtract(p['source_xy'], truth['source_xy'])), abs(p['amplitude']/truth['amplitude']-1), abs(p['junction_offset']-truth['junction_offset'])])
    r = y-prediction
    med = np.median(r)
    scale = 1.4826*np.median(abs(r-med))
    bad = abs(r-med) > 6*scale
    print('public', i, 'held-out residual MAD scale', scale, 'large held-out residuals', int(bad.sum()), 'worst training-fold parameter errors', worst, flush=True)
    assert np.all(worst <= [.035,.08,.008])
    assert scale < .004 and bad.sum() <= 5
    # Exercise the expensive near-wall branch on the same real geometry and
    # conductivity, preserving its real residual pattern and default threading.
    xy = (.006, .74)
    real_residual = y - field(truth['source_xy'], truth['amplitude']) - truth['junction_offset']
    edge = dict(m, T_obs=field(xy, 24.) + truth['junction_offset'] + real_residual)
    start = time.perf_counter()
    p = solver.localize_thermal_source(edge)
    elapsed = time.perf_counter()-start
    print('near wall, default threading', i, 'seconds', elapsed, 'position error', np.linalg.norm(np.subtract(p['source_xy'],xy)), 'relative amplitude error', abs(p['amplitude']/24.-1), flush=True)
    assert elapsed < 15
PY
python3 -I /tmp/audit_thermal_prediction.py
```
- **Container Exit Code**: `0` (`6165 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0 held-out residual MAD scale 0.0013233233131191482 large held-out residuals 3 worst training-fold parameter errors [0.00094335 0.00186332 0.00016723]
near wall, default threading 0 seconds 0.6952279089018703 position error 0.0033360042566472395 relative amplitude error 0.033441712601531615
public 1 held-out residual MAD scale 0.0011743956250742796 large held-out residuals 3 worst training-fold parameter errors [0.00102    0.00220705 0.00015076]
near wall, default threading 1 seconds 0.8989816717803478 position error 0.000845426099391747 relative amplitude error 0.0059335217576975685

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=68c6d2876043a243 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_68c6d287 (directory).
```

### Turn 15 — `submit_completed_task` (`14992 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification. The latest audit found three large held-out residuals per public case, while every grouped fit met parameter tolerances and both near-wall runs on the public layouts finished in under 0.90 seconds. No further change is supported by those results.
- **Alternatives Weighed**: `Impose a stronger source-position or amplitude prior` (rejected: Neither prior is specified, and the near-wall audit found nearly indistinguishable readings for amplitudes with disjoint tolerance intervals.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/thermal_sink_solver.py`. Both public cases met all tolerances in under 0.83 seconds each. All 104 leave-one-sensor-out fits and all 87 synthetic interior cases passed. The grouped holdout audit also recovered the source within tolerance in every fold.

Limitation: 12 of 69 synthetic near-wall cases missed at least one tolerance. Near a wall, changes in source position and amplitude can produce nearly indistinguishable readings.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-Unseen-PostFreeze-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/thermal_sink_solver.py": "FILE:8964"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "max_xy_err": 0.00184,
    "max_amp_rel_err": 0.00296,
    "max_offset_err": 0.00023
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"max_xy_err\": 0.00184, \"max_amp_rel_err\": 0.00296, \"max_offset_err\": 0.00023}\n__HEGEL_VERIFIER_AUTH_286e71d4edd26b921e4defeb60e2dd4d__:{\"strict_pass\": true, \"max_xy_err\": 0.00184, \"max_amp_rel_err\": 0.00296, \"max_offset_err\": 0.00023}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 4785.67,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
