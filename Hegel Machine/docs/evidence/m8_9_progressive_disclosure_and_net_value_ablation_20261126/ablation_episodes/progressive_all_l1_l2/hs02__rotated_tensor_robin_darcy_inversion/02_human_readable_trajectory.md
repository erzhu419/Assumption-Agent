# Dialectical Synthesis Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_2_Selective_LiveGPT` | `hs02__rotated_tensor_robin_darcy_inversion`

- **Official Strict Pass**: **`PASS (1.0)`** (`PASSED_ALL_CHECKS`)
- **Blind Public Synthesis Choice**: `antithesis` via `selective_router_confident_single_branch`
- **Synthesis Rationale**: Primary Antithesis branch satisfied public checks under policy; second branch skipped (`1.00u` nominal units).
- **Post-Lock Branch Verdicts**: Thesis (`Bare`) = `NOT_SPAWNED (0T)` | Antithesis (`Wormhole`) = `True` (10T)

## Selected Branch Trajectory Record
# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `hs02__rotated_tensor_robin_darcy_inversion`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M89-HardStress-4` |
| **Domain** | `Anisotropic Tensor Elliptic PDE & Outlier-Robust Inverse Source Identification` |
| **Wave / Wormhole Family** | `m89_hard_stress` / `fam_multiscale_pde_and_robust_assimilation` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `10 / 28` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `8` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `300,636` (`290,045` / `61,056` / `10,591`) |
| **Total Episode Wall Time (s)** | `326.46s` |

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

### Turn 1 — `execute_in_remote_docker` (`16636 ms` LLM)
- **Rationale**: Inspect fixture keys, shapes, and scales to locate the reference fields and choose forward and inverse checks without transferring full arrays.
- **Alternatives Weighed**: `Use the existing scalar Darcy solver` (rejected: Its zero-ghost Dirichlet stencil omits both Robin conductance and transverse tensor flux.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle
import numpy as np
with open('public_tensor_cases.pkl','rb') as f:
    data=pickle.load(f)
def inspect(x, level=0):
    if isinstance(x, dict):
        for k,v in x.items():
            print('  '*level+str(k)+':',end=' ')
            inspect(v,level+1)
    elif isinstance(x,(list,tuple)) and any(isinstance(v,(dict,tuple)) for v in x):
        print(type(x).__name__,len(x))
        for v in x: inspect(v,level+1)
    else:
        a=np.asarray(x)
        if not a.ndim: print(repr(x))
        elif a.dtype.kind in 'iuf' and a.size: print(a.shape,a.dtype,'range',float(a.min()),float(a.max()),'values',a.tolist() if a.size<=12 else '')
        else: print(a.shape,a.dtype)
inspect(data)
PY
```
- **Container Exit Code**: `0` (`642 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cases: list 2
    problem:       Lx: 1.2
      Ly: 1.0
      k_xx: (22, 22) float64 range 1.0363509910865978 2.24219731617612 values 
      k_xy: (22, 22) float64 range 0.32176035377896317 1.0437379455766584 values 
      k_yy: (22, 22) float64 range 0.5599556333252096 1.5016724120449003 values 
      sigma: (22, 22) float64 range 0.250077479338843 0.39325929752066113 values 
      gamma_robin: 1.65
      u_ambient: 0.4
      basis: (5, 22, 22) float64 range 1.971092819942404e-25 0.9872077221034575 values 
      sensor_ij: (64, 2) int64 range 1.0 19.0 values 
      y_obs: (64,) float64 range -2.1769128090160015 3.6960627392832364 values 
      tikhonov_alpha: 1e-08
    gt:       source_weights: (5,) float64 range 1.5419492109461137 2.3767837702116843 values [2.355978455849119, 2.3767837702116843, 2.3560476507306607, 1.5419492109461137, 1.5469760946511788]
      sensor_offset: -0.2984695515641418
      u_true: (22, 22) float64 range 0.44071671737294293 0.5397746205755551 values 
    problem:       Lx: 1.2
      Ly: 1.0
      k_xx: (22, 22) float64 range 1.0363509910865978 2.24219731617612 values 
      k_xy: (22, 22) float64 range 0.32176035377896317 1.0437379455766584 values 
      k_yy: (22, 22) float64 range 0.5599556333252096 1.5016724120449003 values 
      sigma: (22, 22) float64 range 0.250077479338843 0.39325929752066113 values 
      gamma_robin: 1.65
      u_ambient: 0.4
      basis: (5, 22, 22) float64 range 1.971092819942404e-25 0.9872077221034575 values 
      sensor_ij: (64, 2) int64 range 1.0 19.0 values 
      y_obs: (64,) float64 range -1.9858091674250422 3.910525197770965 values 
      tikhonov_alpha: 1e-08
    gt:       source_weights: (5,) float64 range 1.1856653838234414 2.419012935743922 values [1.2611741711307105, 2.419012935743922, 1.314149026212668, 1.1856653838234414, 2.3934253453776533]
      sensor_offset: -0.08045838188544918
      u_true: (22, 22) float64 range 0.4346920903419648 0.5257877677314957 values
```

### Turn 2 — `execute_in_remote_docker` (`35886 ms` LLM)
- **Rationale**: A forward mismatch would require correcting assembly before tuning the inverse. Centered sensor-response singular values and reference residual quantiles show whether conditioning or spike separation will govern the robust fit.
- **Alternatives Weighed**: `Symmetrize the assembled operator and use CG` (rejected: Symmetrization can change the specified clamped transverse stencil; sparse LU preserves the assembled equations.)

```bash
cat > /holdout_task/tensor_darcy_robin.py <<'PY'
"""Cell-centered tensor–Robin Darcy source inversion."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _operator(problem):
    """Return the integrated finite-volume matrix and ambient RHS."""
    xx = np.asarray(problem['k_xx'], dtype=float)
    xy = np.asarray(problem['k_xy'], dtype=float)
    yy = np.asarray(problem['k_yy'], dtype=float)
    nx, ny = xx.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    ids = np.arange(nx * ny).reshape(nx, ny)
    area = hx * hy
    gamma = problem['gamma_robin']
    boundary = np.zeros_like(xx)
    for i in (0, nx - 1):
        k = xx[i, :]
        boundary[i, :] += hy * gamma * k / (k + gamma * (0.5 * hx))
    for j in (0, ny - 1):
        k = yy[:, j]
        boundary[:, j] += hx * gamma * k / (k + gamma * (0.5 * hy))

    rows = [ids.ravel()]
    cols = [ids.ravel()]
    vals = [(boundary + area * np.asarray(problem['sigma'])).ravel()]

    def add_flux(a, b, terms):
        # One flux per face, outgoing for a and incoming for b.
        a, b = a.ravel(), b.ravel()
        for col, value in terms:
            rows.extend((a, b))
            cols.extend((col.ravel(), col.ravel()))
            v = value.ravel()
            vals.extend((v, -v))

    for axis, normal, ratio in ((0, xx, hy / hx), (1, yy, hx / hy)):
        # In this view the first axis is normal and the second tangential.
        grid = ids if axis == 0 else ids.T
        k = normal if axis == 0 else normal.T
        cross = xy if axis == 0 else xy.T
        a, b = grid[:-1], grid[1:]
        plus = np.minimum(np.arange(grid.shape[1]) + 1, grid.shape[1] - 1)
        minus = np.maximum(np.arange(grid.shape[1]) - 1, 0)
        ka, kb = k[:-1], k[1:]
        g = ratio * (2 * ka * kb / (ka + kb))
        c = (cross[:-1] + cross[1:]) * 0.125
        add_flux(a, b, ((a, g), (b, -g), (a[:, plus], -c),
                        (a[:, minus], c), (b[:, plus], -c), (b[:, minus], c)))

    n = ids.size
    matrix = coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                         np.concatenate(cols))), shape=(n, n)).tocsc()
    return matrix, (boundary * problem['u_ambient']).ravel(), area


def _responses(problem):
    matrix, ambient, area = _operator(problem)
    basis = np.asarray(problem['basis'], dtype=float)
    rhs = np.column_stack((ambient, area * basis.reshape((basis.shape[0], matrix.shape[0])).T))
    return splu(matrix).solve(rhs).reshape((*basis.shape[1:], basis.shape[0] + 1))
PY
python3 -I - <<'PY'
import sys,pickle
import numpy as np
sys.path.insert(0,'/holdout_task')
from tensor_darcy_robin import _responses,_operator
with open('/holdout_task/public_tensor_cases.pkl','rb') as f:
    cases=pickle.load(f)['cases']
for n, case in enumerate(cases):
    p,g=case['problem'],case['gt']; w=g['source_weights']; u=g['u_true']; ij=np.asarray(p['sensor_ij']).T
    A,b,area=_operator(p); rhs=b+area*np.einsum('m,mij->ij',w,p['basis']).ravel()
    z=_responses(p); model=z[...,0]+z[...,1:]@w
    D=z[tuple(ij)][...,1:]; r=p['y_obs']-u[tuple(ij)]-g['sensor_offset']; a=np.sort(np.abs(r))
    print(n,'forward-max',np.max(np.abs(model-u)),'relative-residual',np.linalg.norm(A@u.ravel()-rhs)/np.linalg.norm(rhs),'sv',np.linalg.svd(D-D.mean(axis=0),compute_uv=False),'median abs',np.median(a),'largest abs',a[-10:])
PY
```
- **Container Exit Code**: `0` (`652 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tensor_darcy_robin.py']`
- **Stdout (tail)**:
```text
0 forward-max 2.3314683517128287e-15 relative-residual 3.86152251238858e-14 sv [0.08409948 0.06371736 0.05022973 0.02362306 0.01087366] median abs 0.0008681740323307485 largest abs [2.17903058e-03 2.23496571e-03 2.34767240e-03 2.43520843e-03
 3.33524414e-03 1.85755004e+00 1.95683859e+00 2.31034250e+00
 2.41668910e+00 3.46300655e+00]
1 forward-max 2.1094237467877974e-15 relative-residual 3.7495296826636083e-14 sv [0.07928371 0.06079007 0.05055262 0.02550335 0.01168778] median abs 0.0009248452161155218 largest abs [2.03772634e-03 2.35034610e-03 2.71127992e-03 2.76492287e-03
 3.00045273e-03 2.02587390e+00 2.39353684e+00 2.45124304e+00
 2.89658331e+00 3.50862713e+00]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8761ee16783b07cd (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8761ee16 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`47932 ms` LLM)
- **Rationale**: Use LAD to locate the inlier band and refit with the supplied Tikhonov penalty on sources only. Leave-one-sensor-out tests detect dependence on individual sensors, while high-leverage contamination probes whether the LAD initializer needs a stronger alternative.
- **Alternatives Weighed**: `Fit source weights and offset with ordinary ridge over all sensors` (rejected: Public spikes start at 1.86 while the largest clean residual is 0.00334; fitting all sensors would let spikes dominate the much smaller source signal.)

```bash
sed -i '/^import numpy as np$/a from scipy.optimize import linprog\nfrom scipy.sparse import csr_matrix, eye, hstack, vstack' /holdout_task/tensor_darcy_robin.py
cat >> /holdout_task/tensor_darcy_robin.py <<'PY'


def _robust_sources(response, observed, alpha):
    """Locate the inlier residual band, then fit ridge on source weights only."""
    n, m = response.shape
    center = response.mean(axis=0)
    scale = np.linalg.norm(response - center, axis=0)
    scale[scale == 0] = 1
    design = np.column_stack(((response - center) / scale, np.ones(n)))
    origin = np.median(observed)
    y = observed - origin

    # Gross errors are much larger than the measurement noise. LAD limits
    # their influence while locating the narrow central residual band.
    d = csr_matrix(design)
    slack = -eye(n, format='csr')
    constraints = vstack((hstack((d, slack)), hstack((-d, slack))), format='csr')
    fit = linprog(np.r_[np.zeros(m + 1), np.ones(n)],
                  A_ub=constraints, b_ub=np.r_[y, -y],
                  bounds=[(None, None)] * (m + 1) + [(0, None)] * n,
                  method='highs')
    if not fit.success:
        raise RuntimeError(f'Unable to fit sensor residuals: {fit.message}')

    residual = y - design @ fit.x[:m + 1]
    deviation = np.abs(residual - np.median(residual))
    noise = 1.4826 * np.median(deviation)
    floor = 1e-9 * max(1.0, np.max(np.abs(y)))
    inlier = deviation <= max(4.5 * noise, floor)
    if inlier.sum() < m + 1:
        inlier[np.argsort(deviation)[:m + 1]] = True

    ridge = np.zeros((m, m + 1))
    ridge[np.arange(m), np.arange(m)] = np.sqrt(alpha) / scale
    coef = np.linalg.lstsq(np.vstack((design[inlier], ridge)),
                           np.r_[y[inlier], np.zeros(m)], rcond=None)[0]
    weights = coef[:m] / scale
    return weights, float(origin + coef[m] - center @ weights)


def infer_tensor_darcy_sources(problem: dict) -> dict:
    """Infer source weights, shared sensor bias, and the unbiased field."""
    responses = _responses(problem)
    sensor_ij = np.asarray(problem['sensor_ij'])
    samples = responses[sensor_ij[:, 0], sensor_ij[:, 1]]
    observed = np.asarray(problem['y_obs'], dtype=float) - samples[:, 0]
    weights, offset = _robust_sources(samples[:, 1:], observed,
                                      problem['tikhonov_alpha'])
    return {'source_weights': weights, 'sensor_offset': offset,
            'predicted_field': responses[..., 0] + responses[..., 1:] @ weights}
PY
cat > /tmp/tensor_public_validation.py <<'PY'
import pickle,sys
import numpy as np
sys.path.insert(0,'/holdout_task')
from tensor_darcy_robin import infer_tensor_darcy_sources, _operator, _responses, _robust_sources
with open('/holdout_task/public_tensor_cases.pkl','rb') as f:
    cases=pickle.load(f)['cases']

def errors(w, offset, z, g):
    field=z[...,0]+z[...,1:]@w
    return (np.linalg.norm(w-g['source_weights'])/np.linalg.norm(g['source_weights']),
            abs(offset-g['sensor_offset']), np.max(np.abs(field-g['u_true'])))

for i,case in enumerate(cases):
    p,g=case['problem'],case['gt']
    z=_responses(p); ij=tuple(p['sensor_ij'].T); X=z[ij][:,1:]; y=p['y_obs']-z[ij][:,0]
    a=p['tikhonov_alpha']; n,m=X.shape
    result=infer_tensor_darcy_sources(p)
    assert set(result)=={'source_weights','sensor_offset','predicted_field'}
    w,o,u=[result[k] for k in ('source_weights','sensor_offset','predicted_field')]
    assert w.shape==(m,) and type(o) is float and u.shape==p['k_xx'].shape and np.all(np.isfinite(u))
    A,b,area=_operator(p); rhs=b+area*(w@p['basis'].reshape(m,-1))
    residual=np.linalg.norm(A@u.ravel()-rhs)/np.linalg.norm(rhs)
    assert residual<1e-10
    print('case',i,'public errors',errors(w,o,z,g),'relative FV residual',residual)
    clean=np.abs(y-X@g['source_weights']-g['sensor_offset'])<0.1
    loo=[]
    for k in range(n):
        keep=np.arange(n)!=k
        ww,oo=_robust_sources(X[keep],y[keep],a)
        loo.append((*errors(ww,oo,z,g),abs(y[k]-X[k]@ww-oo)))
    loo=np.array(loo)
    print('LOSO max errors/error on clean held-out sensor',loo[:,:3].max(axis=0),loo[clean,3].max())
    centered=X-X.mean(axis=0)
    U=np.linalg.svd(centered,full_matrices=False)[0]
    rng=np.random.default_rng(77+i)
    stip=[]
    for trial in range(48):
        k=round(n*.1)
        side=U[:,trial % m] * (-1 if trial % 2 else 1)
        contaminated=np.argsort(side)[-k:] if trial%3==0 else rng.choice(n,k,replace=False)
        noise=rng.normal(0,.0015,n)
        yy=X@g['source_weights']+g['sensor_offset']+noise
        yy[contaminated]+=rng.uniform(1.5,4,k)*(1 if trial%3==0 else rng.choice([-1,1],k))
        ww,oo=_robust_sources(X,yy,a)
        stip.append(errors(ww,oo,z,g))
    print('random and high-leverage 10% stress max errors',np.max(stip,axis=0))
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/tensor_public_validation.py
```
- **Container Exit Code**: `0` (`925 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tensor_darcy_robin.py']`
- **Stdout (tail)**:
```text
case 0 public errors (0.013508156233237248, 0.0002478158466358571, 0.0010078598157572394) relative FV residual 3.499033854952922e-14
LOSO max errors/error on clean held-out sensor [0.0229191  0.00173868 0.00246031] 0.003950924351255769
random and high-leverage 10% stress max errors [0.065784   0.00639066 0.00782498]
case 1 public errors (0.03731342313384118, 0.003884479635945043, 0.004579641252252298) relative FV residual 3.4581936687335417e-14
LOSO max errors/error on clean held-out sensor [0.04446097 0.00438796 0.00529212] 0.0032330578270263616
random and high-leverage 10% stress max errors [0.10207541 0.00795294 0.01047586]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fcb5b52af7f2a1c8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_fcb5b52a (directory).
```

### Turn 4 — `execute_in_remote_docker` (`34150 ms` LLM)
- **Rationale**: Compare against ridge with known clean sensors under identical noise and concentrated spikes. A separation failure would justify changing the robust initializer; oracle-level errors would not.
- **Alternatives Weighed**: `Replace LAD immediately with a randomized consensus fit` (rejected: The observed stress error does not yet distinguish outlier-selection failure from error that persists even when the inlier set is known.)

```bash
cat > /tmp/tensor_stress_diagnosis.py <<'PY'
import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from tensor_darcy_robin import _responses, _robust_sources

with open('/holdout_task/public_tensor_cases.pkl', 'rb') as f:
    cases = pickle.load(f)['cases']

for case_id, case in enumerate(cases):
    p, g = case['problem'], case['gt']
    ij = p['sensor_ij']
    Z = _responses(p)
    X = Z[tuple(ij.T)][:, 1:]
    n, m = X.shape
    a = p['tikhonov_alpha']
    w0, o0 = g['source_weights'], g['sensor_offset']
    mu = X @ w0 + o0
    D = np.column_stack((X, np.ones(n)))
    penalty = np.column_stack((np.sqrt(a) * np.eye(m), np.zeros(m)))
    U = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)[0]
    leverage = np.sum(U**2, axis=1)
    rng = np.random.default_rng(269 + case_id)
    errors, oracle_errors, gaps = [], [], []
    for trial in range(240):
        k = (5, 6, 8, 10)[trial % 4]
        mode = trial // 4 % 5
        if mode == 0:
            bad = rng.choice(n, k, replace=False)
        elif mode == 1:
            bad = np.argsort(leverage)[-k:]
        elif mode == 2:
            bad = np.argsort(np.sum((ij - ij[rng.integers(n)])**2, axis=1))[:k]
        else:
            direction = U @ rng.normal(size=m)
            bad = np.argsort(direction if mode == 3 else abs(direction))[-k:]
        keep = np.ones(n, dtype=bool)
        keep[bad] = False
        clean = mu + rng.normal(0, .0015, n)
        y = clean.copy()
        signs = rng.choice([-1, 1], k) if mode < 3 else np.sign(direction[bad])
        y[bad] += signs * rng.uniform(1.5, 4, k)
        w, o = _robust_sources(X, y, a)
        oracle = np.linalg.lstsq(np.vstack((D[keep], penalty)), np.r_[y[keep], np.zeros(m)], rcond=None)[0]
        def metrics(w, o):
            dw = w - w0
            return np.array([np.linalg.norm(dw) / np.linalg.norm(w0), abs(o - o0), np.max(abs(Z[..., 1:] @ dw))])
        errors.append(metrics(w, o)); oracle_errors.append(metrics(oracle[:-1], oracle[-1]))
        residual = abs(y - X @ w - o)
        gaps.append((residual[bad].min(), residual[keep].max()))
    errors, oracle_errors, gaps = map(np.array, (errors, oracle_errors, gaps))
    for k in (6, 10):
        sel = np.arange(240) % 4 == (6, 10).index(k)*2+1
        print(case_id, k, 'spikes: max weight/offset/field errors', errors[sel].max(axis=0), 'oracle max', oracle_errors[sel].max(axis=0), 'max paired excess', (errors[sel]-oracle_errors[sel]).max(axis=0), 'min spike/max clean residual', gaps[sel,0].min(), gaps[sel,1].max())
    failures = np.flatnonzero((gaps[:, 0] < .1) | (gaps[:, 1] > .03))
    print('case', case_id, 'residual separation failures', len(failures), 'first trial indices', failures[:12])
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/tensor_stress_diagnosis.py
```
- **Container Exit Code**: `0` (`1591 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 6 spikes: max weight/offset/field errors [0.09527281 0.00800157 0.01039114] oracle max [0.09527281 0.00800157 0.01039114] max paired excess [9.05178710e-15 3.60822483e-15 1.64018105e-15] min spike/max clean residual 1.5039196850961138 0.005018935377717526
0 10 spikes: max weight/offset/field errors [43.60324645  1.11030268  3.15469798] oracle max [0.08119227 0.0089896  0.00995012] max paired excess [43.52802609  1.10968831  3.14858171] min spike/max clean residual 0.004182288087426045 2.230787390227124
case 0 residual separation failures 12 first trial indices [ 35  74  94  98 114 115 134 135 155 175 194 235]
1 6 spikes: max weight/offset/field errors [0.086238   0.00672231 0.00806861] oracle max [0.086238   0.00672231 0.00806861] max paired excess [0.01307499 0.00065715 0.00071436] min spike/max clean residual 1.5088642855296681 0.005882681269182247
1 10 spikes: max weight/offset/field errors [76.11991283  4.99830052  7.26448217] oracle max [0.12617612 0.00665833 0.0094299 ] max paired excess [76.10313636  4.99674023  7.26226138] min spike/max clean residual 0.0038813968663391485 3.2513347079400985
case 1 residual separation failures 7 first trial indices [ 19  35 119 135 198 219 239]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fcb5b52af7f2a1c8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_fcb5b52a (directory).
```

### Turn 5 — `execute_in_remote_docker` (`37948 ms` LLM)
- **Rationale**: Rerun the identical public, held-out-sensor, and clustered-spike slices to test whether consensus closes the oracle gap without degrading the public fit or finite-volume residual. A remaining oracle gap would warrant revisiting consensus selection, not the stencil.
- **Alternatives Weighed**: `Tighten the LAD residual cutoff` (rejected: The failed LAD fit puts spikes inside the clean residual scale, so a tighter cutoff cannot reliably separate them.); `Increase Tikhonov regularization` (rejected: The clean-sensor oracle already isolates the failure to sensor selection, and increasing the supplied penalty would bias identifiable sources.)
- **ATMS NOGOOD DECLARED**: Refuted `LAD locates the clean sensor residual band even when gross errors are concentrated at high-leverage locations.` | Witness: `In the second public geometry with 10 of 64 sensors corrupted, worst relative source-weight error was 76.12 with LAD versus 0.1262 with the clean-sensor oracle; a spike residual fell to 0.00388.` | Blanket Diagnosis: `Coherent contaminated rows can tilt the fit along weakly observed source directions. The same forward design and noise recover near-oracle estimates when the contaminated rows are excluded.` | Sublated Pivot: `Preserve the verified tensor–Robin stencil and source-only ridge; select clean sensors by fixed-seed consensus scored inside the measured gap between clean residuals and gross spikes.`

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/tensor_darcy_robin.py')
s = p.read_text().replace('from scipy.optimize import linprog\nfrom scipy.sparse import csr_matrix, eye, hstack, vstack\n', '')
start = s.index('def _robust_sources(')
end = s.index('def infer_tensor_darcy_sources(', start)
s = s[:start] + '''def _robust_sources(response, observed, alpha):
    """Find a consensus of clean sensors and refit with source-only ridge."""
    n, m = response.shape
    center = response.mean(axis=0)
    scale = np.linalg.norm(response - center, axis=0)
    scale[scale == 0] = 1
    design = np.column_stack(((response - center) / scale, np.ones(n)))
    origin = np.median(observed)
    y = observed - origin

    # The public clean residuals are below 0.004, and the specified
    # spikes start at 1.5. Score hypotheses in that gap, so large
    # spikes cannot buy a better fit by pulling on high-leverage rows.
    band = 0.05
    rng = np.random.default_rng(831)
    sample_size = min(m + 1, n)
    best = (-1, -np.inf)
    inlier = np.ones(n, dtype=bool)
    for trial in range(193 if n > sample_size else 1):
        sample = slice(None) if trial == 0 else rng.choice(n, sample_size, replace=False)
        coef, _, rank, _ = np.linalg.lstsq(design[sample], y[sample], rcond=None)
        if trial and rank < sample_size:
            continue
        deviation = np.abs(y - design @ coef)
        mask = deviation <= band
        count = int(mask.sum())
        score = (count, -float(deviation[mask] @ deviation[mask]))
        if score > best:
            best, inlier = score, mask
        if count == n:
            break

    # Re-evaluate the band after fitting the consensus without shrinkage.
    # The supplied penalty belongs in the final coefficient estimate only.
    if best[0] >= sample_size:
        coef = np.linalg.lstsq(design[inlier], y[inlier], rcond=None)[0]
        deviation = np.abs(y - design @ coef)
        mask = deviation <= band
        if int(mask.sum()) >= best[0]:
            inlier = mask
    else:
        inlier[:] = True

    ridge = np.zeros((m, m + 1))
    ridge[np.arange(m), np.arange(m)] = np.sqrt(alpha) / scale
    coef = np.linalg.lstsq(np.vstack((design[inlier], ridge)),
                           np.r_[y[inlier], np.zeros(m)], rcond=None)[0]
    weights = coef[:m] / scale
    return weights, float(origin + coef[m] - center @ weights)


''' + s[end:]
p.write_text(s)
PY
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -I /tmp/tensor_public_validation.py
python3 -I /tmp/tensor_stress_diagnosis.py
```
- **Container Exit Code**: `0` (`5746 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tensor_darcy_robin.py']`
- **Stdout (tail)**:
```text
case 0 public errors (0.013508156233237248, 0.0002478158466358571, 0.0010078598157572394) relative FV residual 3.499033854952922e-14
LOSO max errors/error on clean held-out sensor [0.0229191  0.00173868 0.00246031] 0.003950924351255769
random and high-leverage 10% stress max errors [0.065784   0.00639066 0.00782498]
case 1 public errors (0.03731342313384118, 0.003884479635945043, 0.004579641252252298) relative FV residual 3.4581936687335417e-14
LOSO max errors/error on clean held-out sensor [0.04446097 0.00438796 0.00529212] 0.0032330578270263616
random and high-leverage 10% stress max errors [0.10207541 0.00795294 0.01047586]
0 6 spikes: max weight/offset/field errors [0.09527281 0.00800157 0.01039114] oracle max [0.09527281 0.00800157 0.01039114] max paired excess [9.05178710e-15 3.60822483e-15 1.64018105e-15] min spike/max clean residual 1.5039196850961138 0.005018935377717526
0 10 spikes: max weight/offset/field errors [0.08119227 0.0089896  0.00995012] oracle max [0.08119227 0.0089896  0.00995012] max paired excess [8.60422844e-15 3.66373598e-15 1.65102307e-15] min spike/max clean residual 1.5033984176437567 0.004517239349162816
case 0 residual separation failures 0 first trial indices []
1 6 spikes: max weight/offset/field errors [0.086238   0.00672231 0.00806861] oracle max [0.086238   0.00672231 0.00806861] max paired excess [1.42733048e-14 1.63757896e-15 1.77202003e-15] min spike/max clean residual 1.5088642855296681 0.005318305162758774
1 10 spikes: max weight/offset/field errors [0.12617612 0.00665833 0.0094299 ] oracle max [0.12617612 0.00665833 0.0094299 ] max paired excess [8.90953977e-15 2.04003481e-15 1.69309011e-15] min spike/max clean residual 1.4999782911700994 0.00569314285474995
case 1 residual separation failures 0 first trial indices []

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4d3a4af860b6bf8e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_4d3a4af8 (directory).
```

### Turn 6 — `submit_completed_task` (`13298 ms` LLM)
- **Rationale**: Isolated Python verification checked the final artifact's API, public field residuals, leave-one-sensor-out errors, and clustered-spike recovery. The new estimator preserves the public fit and removes the observed LAD masking failure.
- **Alternatives Weighed**: `Increase Tikhonov regularization` (rejected: The consensus fit closes the gap to the known-clean-sensor fit without changing the supplied regularization.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`22` turns remaining; artifact status={'/holdout_task/tensor_darcy_robin.py': 'FILE:5213'}): Do NOT lock prematurely with `22` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tensor_darcy_robin.py`. The forward stencil reproduced both public reference fields within 2.34e-15. Across 480 contamination stress trials, no residual-separation failures were observed. The worst public leave-one-sensor-out relative source-weight error was 4.45%.`

### Turn 7 — `execute_in_remote_docker` (`38705 ms` LLM)
- **Rationale**: Compare the sparse matrix against independent face balances to detect Robin quotient, flux sign, or axis errors, including a rectangular negative-cross-term slice. Test bias equivariance and hold out spatial quadrants of real measurements to detect offset leakage or dependence on a sensor region.
- **Alternatives Weighed**: `Audit classification batches and relay modes` (rejected: This is a steady-state continuous inverse problem with no batches, class labels, or relay state; the specified nuisance is a single additive sensor offset.)

```bash
cat > /tmp/tensor_gate_audit.py <<'PY'
import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from tensor_darcy_robin import infer_tensor_darcy_sources, _operator


def face_balance(p, u):
    xx, xy, yy = (p[k] for k in ('k_xx', 'k_xy', 'k_yy'))
    nx, ny = u.shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    gamma, ambient = p['gamma_robin'], p['u_ambient']
    out = hx*hy*p['sigma']*u
    for i in range(nx):
        for j in range(ny):
            if i+1 < nx:
                lo, hi = max(j-1, 0), min(j+1, ny-1)
                flux = (hy/hx)*2/(1/xx[i,j]+1/xx[i+1,j])*(u[i,j]-u[i+1,j])
                flux -= (xy[i,j]+xy[i+1,j])/8*(u[i,hi]-u[i,lo]+u[i+1,hi]-u[i+1,lo])
                out[i,j] += flux
                out[i+1,j] -= flux
            if j+1 < ny:
                lo, hi = max(i-1, 0), min(i+1, nx-1)
                flux = (hx/hy)*2/(1/yy[i,j]+1/yy[i,j+1])*(u[i,j]-u[i,j+1])
                flux -= (xy[i,j]+xy[i,j+1])/8*(u[hi,j]-u[lo,j]+u[hi,j+1]-u[lo,j+1])
                out[i,j] += flux
                out[i,j+1] -= flux
            out[i,j] += ((int(i==0)+int(i==nx-1))*hy/(1/gamma+hx/(2*xx[i,j])) + (int(j==0)+int(j==ny-1))*hx/(1/gamma+hy/(2*yy[i,j])))*(u[i,j]-ambient)
    return out


with open('/holdout_task/public_tensor_cases.pkl', 'rb') as f:
    cases = pickle.load(f)['cases']
rng = np.random.default_rng(461)
for k, case in enumerate(cases):
    p, g = case['problem'], case['gt']
    u0, w0, o0 = (g[key] for key in ('u_true', 'source_weights', 'sensor_offset'))
    grid = np.asarray(p['sensor_ij'])
    clean = np.abs(p['y_obs']-u0[tuple(grid.T)]-o0)<0.1
    cropped = {**p, **{key: p[key][:13,:19].copy() for key in ('k_xx','k_xy','k_yy','sigma')}}
    cropped['k_xy'] *= -1
    for q in (p, cropped):
        A, b, _ = _operator(q)
        u = rng.normal(size=q['k_xx'].shape)
        actual = (A@u.ravel()-b).reshape(u.shape)
        expected = face_balance(q, u)
        mismatch = np.max(np.abs(actual-expected))
        assert mismatch < 1e-11, mismatch
        print(k, u.shape, 'independent flux discrepancy', mismatch)
    fit = infer_tensor_darcy_sources(p)
    shifted = infer_tensor_darcy_sources({**p, 'y_obs':p['y_obs']+2.25})
    delta_w = np.max(abs(shifted['source_weights']-fit['source_weights']))
    delta_u = np.max(abs(shifted['predicted_field']-fit['predicted_field']))
    delta_o = abs(shifted['sensor_offset']-fit['sensor_offset']-2.25)
    assert max(delta_w, delta_u, delta_o) < 1e-10
    print(k, 'bias equivariance max error', max(delta_w, delta_u, delta_o))
    spatial = []
    mid = (np.array(u0.shape)-1)/2
    for sx, sy in ((0,0),(0,1),(1,0),(1,1)):
        test = ((grid[:,0]>mid[0])==sx)&((grid[:,1]>mid[1])==sy)
        ii, jj = grid[test & clean].T
        res = infer_tensor_darcy_sources({**p,'sensor_ij':grid[~test],'y_obs':p['y_obs'][~test]})
        w, o, u = (res[key] for key in ('source_weights','sensor_offset','predicted_field'))
        e = p['y_obs'][test & clean]-u[ii,jj]-o
        spatial.append((np.linalg.norm(w-w0)/np.linalg.norm(w0),abs(o-o0),np.max(abs(u-u0)),np.max(abs(e))))
    print(k, 'worst spatial holdout: relative source, offset, field max, clean held-out residual max:', np.max(spatial, axis=0))
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/tensor_gate_audit.py
```
- **Container Exit Code**: `0` (`598 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 (22, 22) independent flux discrepancy 3.552713678800501e-15
0 (13, 19) independent flux discrepancy 7.105427357601002e-15
0 bias equivariance max error 1.865174681370263e-14
0 worst spatial holdout: relative source, offset, field max, clean held-out residual max: [0.05146854 0.00213307 0.00579166 0.00550429]
1 (22, 22) independent flux discrepancy 7.105427357601002e-15
1 (13, 19) independent flux discrepancy 3.552713678800501e-15
1 bias equivariance max error 9.103828801926284e-15
1 worst spatial holdout: relative source, offset, field max, clean held-out residual max: [0.07588178 0.00492761 0.00634858 0.00610359]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4d3a4af860b6bf8e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_4d3a4af8 (directory).
```

### Turn 8 — `submit_completed_task` (`14120 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated Python verification. Independent face balances agreed within 7.11e-15, including on a rectangular grid with negative cross terms. Adding a common bias shifted only the estimated offset to numerical precision. Holding out real sensor quadrants gave worst absolute offset and field errors of 0.00493 and 0.00635, respectively.
- **Alternatives Weighed**: `Increase source regularization` (rejected: The stressed fit already matched the known-clean-sensor fit under the supplied penalty; changing it would alter the specified inverse objective.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`20` turns remaining; artifact status={'/holdout_task/tensor_darcy_robin.py': 'FILE:5213'}): Do NOT lock prematurely with `20` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tensor_darcy_robin.py`. The forward stencil reproduced both public reference fields within 2.34e-15. Worst relative source-weight error was 4.45% with one sensor held out and 7.59% with a full quadrant held out. No spike-separation failures occurred in 480 stress trials.`

### Turn 9 — `execute_in_remote_docker` (`29305 ms` LLM)
- **Rationale**: Subsample real sensor records without injecting new noise. Comparing each fit with the known-clean fit on the same subset distinguishes outlier masking, which would warrant changing consensus selection, from geometry-induced loss of precision, which would not.
- **Alternatives Weighed**: `Repeat the flux and common-offset checks` (rejected: The unchanged artifact already passed both to numerical precision; neither check probes reduced sensor coverage.)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from tensor_darcy_robin import _responses, _robust_sources

with open('/holdout_task/public_tensor_cases.pkl', 'rb') as f:
    cases = pickle.load(f)['cases']
rng = np.random.default_rng(9029)
for case_id, case in enumerate(cases):
    p, g = case['problem'], case['gt']
    ij = p['sensor_ij']
    z = _responses(p)[tuple(ij.T)]
    X, y = z[:, 1:], p['y_obs'] - z[:, 0]
    clean = abs(p['y_obs'] - g['u_true'][tuple(ij.T)] - g['sensor_offset']) < .1
    m = X.shape[1]
    D = np.column_stack((X, np.ones(len(X))))
    ridge = np.column_stack((np.sqrt(p['tikhonov_alpha'])*np.eye(m), np.zeros(m)))
    for n in (24, 32, 40):
        worst = np.zeros(3)
        failures = 0
        for trial in range(120):
            if trial % 2:
                distance = np.sum((ij - ij[rng.integers(len(ij))])**2, axis=1)
                keep = np.argsort(distance + rng.uniform(0, .49, len(ij)))[:n]
            else:
                keep = rng.choice(len(ij), n, replace=False)
            c = keep[clean[keep]]
            w, o = _robust_sources(X[keep], y[keep], p['tikhonov_alpha'])
            oracle = np.linalg.lstsq(np.vstack((D[c], ridge)), np.r_[y[c], np.zeros(m)], rcond=None)[0]
            delta = np.r_[w, o] - oracle
            bad = ~clean[keep]
            misclassified = np.sum((abs(y[keep] - X[keep] @ w - o) > .05) != bad)
            failures += bool(misclassified)
            worst = np.maximum(worst, (np.linalg.norm(delta[:-1])/np.linalg.norm(g['source_weights']), abs(delta[-1]), np.max(abs(D @ delta))))
        print(f'case={case_id} training_sensors={n} residual-band failures={failures}/120; worst excess vs clean-only fit [relative weights, offset, sensor prediction]={worst}')
PY
```
- **Container Exit Code**: `0` (`5261 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case=0 training_sensors=24 residual-band failures=0/120; worst excess vs clean-only fit [relative weights, offset, sensor prediction]=[2.56116315e-14 1.31006317e-14 1.48896715e-14]
case=0 training_sensors=32 residual-band failures=0/120; worst excess vs clean-only fit [relative weights, offset, sensor prediction]=[1.72178305e-14 9.93649607e-15 1.04242725e-14]
case=0 training_sensors=40 residual-band failures=0/120; worst excess vs clean-only fit [relative weights, offset, sensor prediction]=[1.67930009e-14 1.25455202e-14 1.24845176e-14]
case=1 training_sensors=24 residual-band failures=0/120; worst excess vs clean-only fit [relative weights, offset, sensor prediction]=[2.11935359e-14 1.00058850e-14 9.97089664e-15]
case=1 training_sensors=32 residual-band failures=0/120; worst excess vs clean-only fit [relative weights, offset, sensor prediction]=[2.27072786e-14 9.63118474e-15 1.09457682e-14]
case=1 training_sensors=40 residual-band failures=0/120; worst excess vs clean-only fit [relative weights, offset, sensor prediction]=[2.37969085e-14 7.84095011e-15 9.04511281e-15]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4d3a4af860b6bf8e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_4d3a4af8 (directory).
```

### Turn 10 — `submit_completed_task` (`13633 ms` LLM)
- **Rationale**: The verified artifact passed isolated Python runs, independent face-balance and offset-equivariance checks, spatial holdouts, and concentrated-spike stress tests. The reduced-coverage audit found no remaining gap to the known-clean-sensor fit.
- **Alternatives Weighed**: `Change the consensus threshold or Tikhonov penalty` (rejected: The estimator already matches the known-clean-sensor fit in the reduced-coverage audit; no remaining outlier-selection error justifies a change.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tensor_darcy_robin.py`. The forward solver reproduced both public reference fields within 2.34e-15. The inverse solver matched fits using known clean sensors across all 720 public-data subsamples, with no missed spikes or false rejections. Worst relative source-weight error was 4.45% when holding out one sensor and 7.59% when holding out a quadrant.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-M89-HardStress-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/tensor_darcy_robin.py": "FILE:5213"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "all_valid": true,
    "max_weight_rel_l2": 0.042679,
    "max_field_rel_l2": 0.012911,
    "max_offset_abs_err": 0.003575,
    "elapsed_sec": 0.2888
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"all_valid\": true, \"max_weight_rel_l2\": 0.042679, \"max_field_rel_l2\": 0.012911, \"max_offset_abs_err\": 0.003575, \"elapsed_sec\": 0.2888}\n__HEGEL_VERIFIER_AUTH_97bfe1459944b09a29254cb7ca139fd3__:{\"strict_pass\": true, \"all_valid\": true, \"max_weight_rel_l2\": 0.042679, \"max_field_rel_l2\": 0.012911, \"max_offset_abs_err\": 0.003575, \"elapsed_sec\": 0.2888}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5117.0,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
