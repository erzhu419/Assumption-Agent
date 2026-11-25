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
| **Total Tokens (Prompt / Cached / Out)** | `302,873` (`291,238` / `52,096` / `11,635`) |
| **Total Episode Wall Time (s)** | `341.15s` |

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

### Turn 1 — `execute_in_remote_docker` (`16530 ms` LLM)
- **Rationale**: Inspect compact public-case metadata to identify the reference fields and verify input shapes before assembly. The reference fields will distinguish stencil or indexing errors from failures of robust inversion.
- **Alternatives Weighed**: `Use the recalled scalar Dirichlet solver` (rejected: It omits the prescribed transverse fluxes and Robin boundary conductances.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle
import numpy as np
with open('public_tensor_cases.pkl','rb') as f:
    data = pickle.load(f)
def describe(x, label='root', depth=0):
    pad = '  '*depth
    if isinstance(x, dict):
        print(pad, label, 'dict', list(x))
        if depth < 4:
            for k,v in list(x.items())[:20]: describe(v, str(k), depth+1)
    elif isinstance(x,(list,tuple)):
        print(pad,label,type(x).__name__,len(x))
        if depth < 3:
            for i,v in enumerate(x[:8]): describe(v,str(i),depth+1)
    elif isinstance(x,np.ndarray):
        print(pad,label,x.shape,x.dtype, 'range', (x.min(),x.max()) if x.size and x.dtype.kind in 'iuf' else '')
    else:
        print(pad,label,repr(x)[:120])
describe(data)
PY
```
- **Container Exit Code**: `0` (`561 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
root dict ['cases']
   cases list 2
     0 dict ['problem', 'gt']
       problem dict ['Lx', 'Ly', 'k_xx', 'k_xy', 'k_yy', 'sigma', 'gamma_robin', 'u_ambient', 'basis', 'sensor_ij', 'y_obs', 'tikhonov_alpha']
         Lx 1.2
         Ly 1.0
         k_xx (22, 22) float64 range (1.0363509910865978, 2.24219731617612)
         k_xy (22, 22) float64 range (0.32176035377896317, 1.0437379455766584)
         k_yy (22, 22) float64 range (0.5599556333252096, 1.5016724120449003)
         sigma (22, 22) float64 range (0.250077479338843, 0.39325929752066113)
         gamma_robin 1.65
         u_ambient 0.4
         basis (5, 22, 22) float64 range (1.971092819942404e-25, 0.9872077221034575)
         sensor_ij (64, 2) int64 range (1, 19)
         y_obs (64,) float64 range (-2.1769128090160015, 3.6960627392832364)
         tikhonov_alpha 1e-08
       gt dict ['source_weights', 'sensor_offset', 'u_true']
         source_weights (5,) float64 range (1.5419492109461137, 2.3767837702116843)
         sensor_offset -0.2984695515641418
         u_true (22, 22) float64 range (0.44071671737294293, 0.5397746205755551)
     1 dict ['problem', 'gt']
       problem dict ['Lx', 'Ly', 'k_xx', 'k_xy', 'k_yy', 'sigma', 'gamma_robin', 'u_ambient', 'basis', 'sensor_ij', 'y_obs', 'tikhonov_alpha']
         Lx 1.2
         Ly 1.0
         k_xx (22, 22) float64 range (1.0363509910865978, 2.24219731617612)
         k_xy (22, 22) float64 range (0.32176035377896317, 1.0437379455766584)
         k_yy (22, 22) float64 range (0.5599556333252096, 1.5016724120449003)
         sigma (22, 22) float64 range (0.250077479338843, 0.39325929752066113)
         gamma_robin 1.65
         u_ambient 0.4
         basis (5, 22, 22) float64 range (1.971092819942404e-25, 0.9872077221034575)
         sensor_ij (64, 2) int64 range (1, 19)
         y_obs (64,) float64 range (-1.9858091674250422, 3.910525197770965)
         tikhonov_alpha 1e-08
       gt dict ['source_weights', 'sensor_offset', 'u_true']
         source_weights (5,) float64 range (1.1856653838234414, 2.419012935743922)
         sensor_offset -0.08045838188544918
         u_true (22, 22) float64 range (0.4346920903419648, 0.5257877677314957)
```

### Turn 2 — `execute_in_remote_docker` (`50984 ms` LLM)
- **Rationale**: Compare forward fields at the true weights before judging inversion. A forward mismatch would point to flux assembly; a matching field with poor source recovery would instead point to sensor weighting or conditioning.
- **Alternatives Weighed**: `Ordinary ridge on all sensor observations` (rejected: The documented spikes can be much larger than the pressure variation that identifies the sources.)

```bash
cat > /holdout_task/tensor_darcy_robin.py <<'PY'
"""Cell-centered tensor Darcy forward model and robust source inversion."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _assemble(problem):
    """Return the integrated cell balance matrix and ambient load."""
    kxx, kxy, kyy, sigma = (np.asarray(problem[key], dtype=float)
                            for key in ('k_xx', 'k_xy', 'k_yy', 'sigma'))
    nx, ny = kxx.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    n = nx * ny
    cells = np.arange(n).reshape(nx, ny)

    def differences(lo, hi):
        lo, hi = lo.ravel(), hi.ravel()
        face = np.arange(lo.size)
        return sparse.csr_matrix((np.tile([1., -1.], (lo.size, 1)).ravel(),
                                  (np.repeat(face, 2), np.column_stack((lo, hi)).ravel())),
                                 shape=(lo.size, n))

    ix, iy = np.arange(nx), np.arange(ny)
    gx = differences(cells, cells[np.minimum(ix + 1, nx - 1), :])
    # Replace the forward difference with the prescribed clamped centered one.
    gx = differences(cells[np.maximum(ix - 1, 0), :],
                     cells[np.minimum(ix + 1, nx - 1), :])
    gy = differences(cells[:, np.maximum(iy - 1, 0)],
                     cells[:, np.minimum(iy + 1, ny - 1)])
    xlo, xhi = cells[:-1, :], cells[1:, :]
    ylo, yhi = cells[:, :-1], cells[:, 1:]
    dx, dy = differences(xlo, xhi), differences(ylo, yhi)
    kx0, kx1 = kxx[:-1, :], kxx[1:, :]
    ky0, ky1 = kyy[:, :-1], kyy[:, 1:]
    normal_x = (hy / hx) * (2 * kx0 * kx1 / (kx0 + kx1))
    normal_y = (hx / hy) * (2 * ky0 * ky1 / (ky0 + ky1))
    cross_x = (kxy[:-1, :] + kxy[1:, :]) / 8
    cross_y = (kxy[:, :-1] + kxy[:, 1:]) / 8
    fx = dx.multiply(normal_x.reshape(-1, 1)) + (
        gy[xlo.ravel()] + gy[xhi.ravel()]).multiply(cross_x.reshape(-1, 1))
    fy = dy.multiply(normal_y.reshape(-1, 1)) + (
        gx[ylo.ravel()] + gx[yhi.ravel()]).multiply(cross_y.reshape(-1, 1))

    gamma = problem['gamma_robin']
    robin = np.zeros_like(kxx)
    for i in (0, -1):
        robin[i, :] += hy * gamma * kxx[i, :] / (gamma * hx / 2 + kxx[i, :])
    for j in (0, -1):
        robin[:, j] += hx * gamma * kyy[:, j] / (gamma * hy / 2 + kyy[:, j])
    area = hx * hy
    matrix = dx.T @ fx + dy.T @ fy + sparse.diags((robin + area * sigma).ravel())
    return matrix.tocsc(), (robin * problem['u_ambient']).ravel(), area


def _forward(problem):
    matrix, ambient, area = _assemble(problem)
    basis = np.asarray(problem['basis'], dtype=float)
    loads = np.column_stack((ambient, area * basis.reshape(len(basis), matrix.shape[0]).T))
    return splu(matrix).solve(loads)


def _robust_fit(design, target, alpha):
    """Estimate sources and a free intercept, rejecting gross sensor spikes."""
    penalty = np.diag(np.r_[np.full(design.shape[1] - 1, np.sqrt(alpha)), 0.])

    def fit(weights):
        root = np.sqrt(weights)
        return np.linalg.lstsq(np.vstack((root[:, None] * design, penalty)),
                               np.r_[root * target, np.zeros(len(penalty))], rcond=None)[0]

    # Start from a robust location/scale before fitting the weak source responses.
    centered = target - np.median(target)
    scale = max(1.4826 * np.median(np.abs(centered)), 1e-12)
    weights = 1 / (1 + (centered / (4 * scale))**2)
    for _ in range(20):
        coef = fit(weights)
        residual = target - design @ coef
        center = np.median(residual)
        scale = max(1.4826 * np.median(np.abs(residual - center)), 1e-12)
        # Give ordinary noise quadratic loss; taper only large residuals.
        updated = 1 / (1 + ((residual - center) / (5 * scale))**4)
        if np.max(np.abs(updated - weights)) < 1e-6:
            break
        weights = updated
    return fit(updated)


def infer_tensor_darcy_sources(problem: dict) -> dict:
    """Infer source weights, a common sensor offset, and the pressure field."""
    shape = np.shape(problem['k_xx'])
    responses = _forward(problem)
    sensors = np.ravel_multi_index(np.asarray(problem['sensor_ij']).T, shape)
    observed = responses[sensors]
    design = np.column_stack((observed[:, 1:], np.ones(len(sensors))))
    target = np.asarray(problem['y_obs'], dtype=float) - observed[:, 0]
    coef = _robust_fit(design, target, problem['tikhonov_alpha'])
    field = responses[:, 0] + responses[:, 1:] @ coef[:-1]
    return {'source_weights': coef[:-1], 'sensor_offset': float(coef[-1]),
            'predicted_field': field.reshape(shape)}
PY
sed -i '/gx = differences(cells, cells\[/d; /# Replace the forward difference/d' /holdout_task/tensor_darcy_robin.py
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle
import numpy as np
from tensor_darcy_robin import _forward, infer_tensor_darcy_sources
with open('public_tensor_cases.pkl','rb') as f:
    cases = pickle.load(f)['cases']
for i,c in enumerate(cases):
    p,g = c['problem'],c['gt']
    v = _forward(p)
    u = (v[:,0] + v[:,1:] @ g['source_weights']).reshape(g['u_true'].shape)
    fit = infer_tensor_darcy_sources(p)
    noise = p['y_obs'] - g['u_true'][tuple(p['sensor_ij'].T)] - g['sensor_offset']
    print(i, 'stencil max abs', np.max(abs(u-g['u_true'])), 'noise absolute order', np.array2string(np.sort(abs(noise))[-14:], precision=5))
    print('source rel', np.linalg.norm(fit['source_weights']-g['source_weights'])/np.linalg.norm(g['source_weights']), 'offset abs', abs(fit['sensor_offset']-g['sensor_offset']), 'field rel', np.linalg.norm(fit['predicted_field']-g['u_true'])/np.linalg.norm(g['u_true']), 'weights', g['source_weights'], fit['source_weights'])
PY
```
- **Container Exit Code**: `0` (`483 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tensor_darcy_robin.py']`
- **Stdout (tail)**:
```text
0 stencil max abs 2.9976021664879227e-15 noise absolute order [1.63101e-03 1.69492e-03 1.79667e-03 2.13711e-03 2.17903e-03 2.23497e-03
 2.34767e-03 2.43521e-03 3.33524e-03 1.85755e+00 1.95684e+00 2.31034e+00
 2.41669e+00 3.46301e+00]
source rel 0.014066838192636523 offset abs 0.00016152025113036483 field rel 0.0007906264419530089 weights [2.35597846 2.37678377 2.35604765 1.54194921 1.54697609] [2.35447111 2.34321752 2.33744548 1.54018466 1.59970612]
1 stencil max abs 2.9976021664879227e-15 noise absolute order [1.66936e-03 1.86282e-03 1.97691e-03 2.01979e-03 2.03773e-03 2.35035e-03
 2.71128e-03 2.76492e-03 3.00045e-03 2.02587e+00 2.39354e+00 2.45124e+00
 2.89658e+00 3.50863e+00]
source rel 0.03749806876175083 offset abs 0.0038933874325717127 field rel 0.007715131336706763 weights [1.26117417 2.41901294 1.31414903 1.18566538 2.39342535] [1.33439871 2.4753487  1.40236229 1.26691323 2.39096908]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1c7f0f15fc8c63fa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1c7f0f15 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`46193 ms` LLM)
- **Rationale**: The independent flux balance catches assembly errors on changed geometries; public leave-one-sensor-out prediction and targeted spikes test whether a few sensors destabilize the inverse fit. Noiseless runs separate regularization bias from contamination sensitivity.
- **Alternatives Weighed**: `Lower the robust-fit cutoff immediately` (rejected: The public cases do not show whether the larger errors come from contamination, shrinkage, or sensor geometry. Stress these separately before changing the fit.)

```bash
cat > /tmp/check_tensor_darcy.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('tensor_darcy_robin', '/holdout_task/tensor_darcy_robin.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_tensor_cases.pkl', 'rb') as stream:
    cases = pickle.load(stream)['cases']
rng = np.random.default_rng(445)

def errors(p, weights, offset, field):
    result = mod.infer_tensor_darcy_sources(p)
    assert set(result) == {'source_weights', 'sensor_offset', 'predicted_field'}
    assert result['source_weights'].shape == weights.shape
    assert result['predicted_field'].shape == field.shape
    assert type(result['sensor_offset']) is float
    return (np.linalg.norm(result['source_weights']-weights)/np.linalg.norm(weights),
            abs(result['sensor_offset']-offset),
            np.linalg.norm(result['predicted_field']-field)/np.linalg.norm(field))

def direct_balance(p, u):
    kxx,kxy,kyy,sigma = [p[k] for k in ('k_xx','k_xy','k_yy','sigma')]
    nx,ny = u.shape
    hx,hy = p['Lx']/nx,p['Ly']/ny
    gamma,ambient = p['gamma_robin'],p['u_ambient']
    w = np.pad(u, 1, mode='edge')
    cx = w[2:,1:-1]-w[:-2,1:-1]
    cy = w[1:-1,2:]-w[1:-1,:-2]
    out = hx*hy*sigma*u
    fx = hy/hx*(2*kxx[:-1]*kxx[1:]/(kxx[:-1]+kxx[1:]))*(u[:-1]-u[1:]) - (kxy[:-1]+kxy[1:])*(cy[:-1]+cy[1:])/8
    fy = hx/hy*(2*kyy[:,:-1]*kyy[:,1:]/(kyy[:,:-1]+kyy[:,1:]))*(u[:,:-1]-u[:,1:]) - (kxy[:,:-1]+kxy[:,1:])*(cx[:,:-1]+cx[:,1:])/8
    out[:-1] += fx
    out[1:] -= fx
    out[:,:-1] += fy
    out[:,1:] -= fy
    for i in (0,-1):
        out[i] += hy*gamma*kxx[i]/(kxx[i]+gamma*hx/2)*(u[i]-ambient)
    for j in (0,-1):
        out[:,j] += hx*gamma*kyy[:,j]/(kyy[:,j]+gamma*hy/2)*(u[:,j]-ambient)
    return out

worst_balance = 0.
for k,c in enumerate(cases):
    p,g = c['problem'],c['gt']
    print('public', k, 'source_rel offset_abs field_rel', errors(p,g['source_weights'],g['sensor_offset'],g['u_true']))
    R = mod._forward(p)
    ij = p['sensor_ij']
    clean = abs(p['y_obs']-g['u_true'][tuple(ij.T)]-g['sensor_offset']) < .05
    held_clean = []
    for j in np.where(clean)[0]:
        keep = np.arange(len(ij)) != j
        result = mod.infer_tensor_darcy_sources(dict(p,sensor_ij=ij[keep],y_obs=p['y_obs'][keep]))
        held_clean.append(abs(result['predicted_field'][tuple(ij[j])]+result['sensor_offset']-p['y_obs'][j]))
    print('public leave-one-clean-sensor-out maximum absolute error', max(held_clean))

base = cases[0]['problem']
for label,change in [('public_geometry',{}), ('negative_cross',{'k_xy': -base['k_xy']}), ('weak_robin_rectangular', {'Lx': 1.85, 'Ly': .65, 'gamma_robin': .16}), ('strong_robin',{'gamma_robin': 6.0})]:
    p = dict(base,**change)
    R = mod._forward(p)
    shape = p['k_xx'].shape
    test_u = rng.normal(size=shape)
    A,b,_ = mod._assemble(p)
    wrong = np.max(abs((A @ test_u.ravel()-b).reshape(shape)-direct_balance(p,test_u)))
    worst_balance = max(worst_balance, wrong)
    np.testing.assert_allclose((A @ test_u.ravel()-b).reshape(shape),direct_balance(p,test_u),atol=1e-12)
    flat = np.ravel_multi_index(p['sensor_ij'].T,shape)
    x = np.column_stack((R[flat,1:],np.ones(len(flat))))
    leverage = (np.linalg.qr(x)[0]**2).sum(axis=1)
    slices = {}
    for mode in ('noiseless','random_spikes','high_leverage_spikes'):
        values = []
        for j in range(24):
            weights = rng.uniform(.5,3,len(p['basis']))
            if j % 2:
                weights *= rng.choice([-1,1],len(weights))
            offset = rng.uniform(-3,3)
            field = (R[:,0] + R[:,1:] @ weights).reshape(shape)
            observations = field.ravel()[flat]+offset
            if mode != 'noiseless':
                observations += rng.normal(0,.0015,len(flat))
                out = rng.choice(len(flat),6,replace=False) if mode == 'random_spikes' else np.argsort(leverage)[-6:]
                observations[out] += rng.uniform(1.5,4,len(out))*rng.choice([-1,1],len(out))
            values.append(errors(dict(p,y_obs=observations),weights,offset,field))
        slices[mode] = np.round(np.max(values,axis=0),7).tolist()
    print(label, slices)
print('stress stencil max absolute balance difference',worst_balance)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check_tensor_darcy.py
```
- **Container Exit Code**: `0` (`1943 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0 source_rel offset_abs field_rel (0.014066838192636523, 0.00016152025113036483, 0.0007906264419530089)
public leave-one-clean-sensor-out maximum absolute error 0.0039549784831817425
public 1 source_rel offset_abs field_rel (0.03749806876175083, 0.0038933874325717127, 0.007715131336706763)
public leave-one-clean-sensor-out maximum absolute error 0.0032371784895942057
public_geometry {'noiseless': [8.15e-05, 6.7e-06, 1.42e-05], 'random_spikes': [0.0714989, 0.0061359, 0.0116775], 'high_leverage_spikes': [0.0765419, 0.007122, 0.017409]}
negative_cross {'noiseless': [8.09e-05, 6e-06, 1.82e-05], 'random_spikes': [0.1060837, 0.0050026, 0.0157489], 'high_leverage_spikes': [0.1148317, 0.0071375, 0.0152022]}
weak_robin_rectangular {'noiseless': [4.25e-05, 2.57e-05, 4.57e-05], 'random_spikes': [0.0614003, 0.0286113, 0.1461497], 'high_leverage_spikes': [0.0828967, 0.0456449, 0.3564296]}
strong_robin {'noiseless': [7.82e-05, 3.5e-06, 7.2e-06], 'random_spikes': [0.0736507, 0.0037978, 0.0081101], 'high_leverage_spikes': [0.1216932, 0.0029109, 0.0081133]}
stress stencil max absolute balance difference 7.105427357601002e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1c7f0f15fc8c63fa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1c7f0f15 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`28812 ms` LLM)
- **Rationale**: Compare the current fit with known clean sensors and with a deployment-available residual-gap refit on the same draws. If known clean sensors do not improve the weak-Robin slice, changing spike rejection is not the remedy.
- **Alternatives Weighed**: `Increase regularization on the current fit` (rejected: It could bias valid source weights without resolving whether the large relative field error comes from missed spikes or weak identification of the field mean.)

```bash
cat > /tmp/diagnose_tensor_fit.py <<'PY'
import numpy as np
from collections import defaultdict

path = '/tmp/check_tensor_darcy.py'
text = open(path).read()
marker = 'worst_balance = 0.'
head, tail = text.split(marker, 1)
exec(head)
original_errors = errors
comparisons = defaultdict(list)

def errors(p, weights, offset, field):
    baseline = original_errors(p, weights, offset, field)
    ij = tuple(p['sensor_ij'].T)
    y = p['y_obs']
    true_residual = y - field[ij] - offset
    clean = abs(true_residual) < .75
    b = mod.infer_tensor_darcy_sources(p)
    detected = abs(y - b['predicted_field'][ij] - b['sensor_offset']) < .75
    oracle = original_errors(dict(p, sensor_ij=p['sensor_ij'][clean], y_obs=y[clean]), weights, offset, field)
    responses = mod._forward(p)
    flat = np.ravel_multi_index(ij, field.shape)
    a = np.column_stack((responses[flat,1:], np.ones(len(flat))))
    z = y - responses[flat,0]
    pmat = np.diag(np.r_[np.full(len(weights), np.sqrt(p['tikhonov_alpha'])), 0.])
    coef = np.linalg.lstsq(np.vstack((a[detected], pmat)), np.r_[z[detected], np.zeros(len(pmat))], rcond=None)[0]
    pred = (responses[:,0] + responses[:,1:] @ coef[:-1]).reshape(field.shape)
    gapfit = (np.linalg.norm(coef[:-1]-weights)/np.linalg.norm(weights), abs(coef[-1]-offset), np.linalg.norm(pred-field)/np.linalg.norm(field))
    comparisons[(p['Lx'], p['gamma_robin'], np.sign(p['k_xy'].mean()))].append((baseline, oracle, gapfit, np.mean(b['predicted_field'] - field), np.sqrt(np.mean(field**2)), np.sum(clean != detected)))
    return baseline

exec(marker + tail)
for key, rows in comparisons.items():
    vals = np.asarray([r[:3] for r in rows]); w = max(rows, key=lambda r:r[0][2]); print('paired slice',key,'worst source/offset/field by baseline, oracle mask with current fit, detected mask with quadratic refit',vals.max(axis=0).round(7).tolist(), 'worst field mean error and truth RMS', w[3:5], 'mask disagreements', sum(r[5] for r in rows))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/diagnose_tensor_fit.py
```
- **Container Exit Code**: `0` (`4959 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.0038933874325717127, 0.007715131336706763)
public leave-one-clean-sensor-out maximum absolute error 0.0032371784895942057
public_geometry {'noiseless': [8.15e-05, 6.7e-06, 1.42e-05], 'random_spikes': [0.0714989, 0.0061359, 0.0116775], 'high_leverage_spikes': [0.0765419, 0.007122, 0.017409]}
negative_cross {'noiseless': [8.09e-05, 6e-06, 1.82e-05], 'random_spikes': [0.1060837, 0.0050026, 0.0157489], 'high_leverage_spikes': [0.1148317, 0.0071375, 0.0152022]}
weak_robin_rectangular {'noiseless': [4.25e-05, 2.57e-05, 4.57e-05], 'random_spikes': [0.0614003, 0.0286113, 0.1461497], 'high_leverage_spikes': [0.0828967, 0.0456449, 0.3564296]}
strong_robin {'noiseless': [7.82e-05, 3.5e-06, 7.2e-06], 'random_spikes': [0.0736507, 0.0037978, 0.0081101], 'high_leverage_spikes': [0.1216932, 0.0029109, 0.0081133]}
stress stencil max absolute balance difference 7.105427357601002e-15
paired slice (1.2, 1.65, 1.0) worst source/offset/field by baseline, oracle mask with current fit, detected mask with quadratic refit [[0.0765419, 0.007122, 0.017409], [0.0727504, 0.0071173, 0.0174006], [0.0900065, 0.0071333, 0.0174252]] worst field mean error and truth RMS (0.006951957994574551, 0.40347841590294475) mask disagreements 0
paired slice (1.2, 1.65, -1.0) worst source/offset/field by baseline, oracle mask with current fit, detected mask with quadratic refit [[0.1148317, 0.0071375, 0.0157489], [0.1159413, 0.0072133, 0.0157705], [0.1132524, 0.0070443, 0.0156741]] worst field mean error and truth RMS (-0.004598024583678012, 0.2940006070807704) mask disagreements 0
paired slice (1.85, 0.16, 1.0) worst source/offset/field by baseline, oracle mask with current fit, detected mask with quadratic refit [[0.0828967, 0.0456449, 0.3564296], [0.0828559, 0.0456042, 0.3561437], [0.0829688, 0.0457134, 0.3569165]] worst field mean error and truth RMS (0.0452974054273634, 0.1271305056846337) mask disagreements 0
paired slice (1.2, 6.0, 1.0) worst source/offset/field by baseline, oracle mask with current fit, detected mask with quadratic refit [[0.1216932, 0.0037978, 0.0081133], [0.1214433, 0.0037951, 0.0081056], [0.1222694, 0.0038086, 0.008144]] worst field mean error and truth RMS (0.0028874834698901564, 0.3684826154678934) mask disagreements 0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1c7f0f15fc8c63fa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1c7f0f15 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`29949 ms` LLM)
- **Rationale**: The isolated challenger tests whether using full quadratic weight on clearly separated inliers improves source recovery without degrading held-out sensor prediction, offset, or field accuracy. Keep the current artifact unless the paired results justify replacing it.
- **Alternatives Weighed**: `Increase source regularization to reduce the weak-Robin field error` (rejected: The oracle result points to weak identification, not misclassified spikes. Stronger shrinkage would impose an unsupported preference for smaller sources.)
- **ATMS NOGOOD DECLARED**: Refuted `Missed sensor spikes drive the large relative field error in the weak-Robin stress slice.` | Witness: `The detected and oracle spike masks agreed on every tested draw, and removing true spikes before fitting changed the worst weak-Robin field error only from 35.64% to 35.61%.` | Blanket Diagnosis: `The worst field error was nearly a uniform level shift: mean error 0.04530, nearly canceled at the sensors by the fitted offset. Weak Robin exchange makes that level poorly identified by noisy observations with a free offset.` | Sublated Pivot: `Keep the exact stencil and free offset. Compare a quadratic inlier refit when the residual gap is clear; judge it on the paired public and stress slices without treating it as a cure for weak identification.`

```bash
python3 - <<'PY'
from pathlib import Path
src = Path('/holdout_task/tensor_darcy_robin.py').read_text()
needle = '        center = np.median(residual)\n        scale = max(1.4826 * np.median(np.abs(residual - center)), 1e-12)\n'
assert src.count(needle) == 1
src = src.replace(needle, needle + '''        # With a clear gap, use the full quadratic information in the inliers.
        magnitude = np.abs(residual - center)
        ordered = np.sort(magnitude)
        first = max(design.shape[1] + 1, (3 * len(target) + 3) // 4)
        for split in range(first, len(target)):
            if ordered[split] > max(0.1, 12 * ordered[split - 1], 12 * scale):
                clean = magnitude <= ordered[split - 1]
                return fit(clean)
''')
Path('/tmp/tensor_darcy_gap.py').write_text(src)
check = Path('/tmp/check_tensor_darcy.py').read_text()
Path('/tmp/check_tensor_darcy_gap.py').write_text(check.replace('/holdout_task/tensor_darcy_robin.py', '/tmp/tensor_darcy_gap.py'))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/check_tensor_darcy_gap.py
```
- **Container Exit Code**: `0` (`1839 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0 source_rel offset_abs field_rel (0.013508156233244478, 0.0002478158466377445, 0.0008653432827755471)
public leave-one-clean-sensor-out maximum absolute error 0.003950924351257573
public 1 source_rel offset_abs field_rel (0.03731342313383632, 0.003884479635946389, 0.007682496822051643)
public leave-one-clean-sensor-out maximum absolute error 0.003233057827025043
public_geometry {'noiseless': [8.15e-05, 6.7e-06, 1.42e-05], 'random_spikes': [0.0709957, 0.0061236, 0.0116584], 'high_leverage_spikes': [0.0900065, 0.0071333, 0.0174252]}
negative_cross {'noiseless': [8.09e-05, 6e-06, 1.82e-05], 'random_spikes': [0.105734, 0.0049714, 0.0156741], 'high_leverage_spikes': [0.1132524, 0.0070443, 0.0149753]}
weak_robin_rectangular {'noiseless': [4.25e-05, 2.57e-05, 4.57e-05], 'random_spikes': [0.0610755, 0.0284587, 0.1462753], 'high_leverage_spikes': [0.0829688, 0.0457134, 0.3569165]}
strong_robin {'noiseless': [7.82e-05, 3.5e-06, 7.2e-06], 'random_spikes': [0.0739042, 0.0038086, 0.0081348], 'high_leverage_spikes': [0.1222694, 0.0029214, 0.008144]}
stress stencil max absolute balance difference 7.105427357601002e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1c7f0f15fc8c63fa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1c7f0f15 (directory).
```

### Turn 6 — `submit_completed_task` (`16010 ms` LLM)
- **Rationale**: The retained artifact passed isolated-process checks of the output contract, public forward references, and an independent flux balance on stressed geometries. The paired comparison did not justify replacing it, and the oracle clean-mask test showed that more aggressive spike rejection would not fix the limiting field error.
- **Alternatives Weighed**: `Quadratic refit after residual-gap spike rejection` (rejected: It did not improve the limiting weak-Robin field error and raised worst source error under high-leverage spikes on the public geometry from 7.65% to 9.00%.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`22` turns remaining; artifact status={'/holdout_task/tensor_darcy_robin.py': 'FILE:4339'}): Do NOT lock prematurely with `22` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tensor_darcy_robin.py`. The forward stencil matched both public reference fields within 3.0e-15. Across the public cases, the worst relative source error was 3.75%, the worst absolute offset error was 0.00389, and the worst relative field error was 0.772%. In the weak-Robin stress test, even knowing which sensors were clean left a worst-case field error of 35.61%: the observations could not reliably separate the field’s mean from the sensor offset.`

### Turn 7 — `execute_in_remote_docker` (`38664 ms` LLM)
- **Rationale**: Use the real measurements to test conservation, tensor sign and axis conventions, and invariance to a common sensor shift. A symmetry failure would require a stencil or offset fix; delete-one instability or spike-mask disagreement would justify revisiting the inverse fit. The offset-profiled singular values distinguish that failure from noise amplification in the field mean.
- **Alternatives Weighed**: `Apply batch centering or relay-mode checks` (rejected: These steady-state measurements contain no batch labels or relay states. A common additive sensor offset is the specified nuisance.)

```bash
cat > /tmp/audit_tensor_public.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('tensor_darcy_robin', '/holdout_task/tensor_darcy_robin.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_tensor_cases.pkl', 'rb') as stream:
    cases = pickle.load(stream)['cases']

for index, case in enumerate(cases):
    p, g = case['problem'], case['gt']
    ij = p['sensor_ij']
    shape = p['k_xx'].shape
    flat = np.ravel_multi_index(ij.T, shape)
    response = mod._forward(p)
    result = mod.infer_tensor_darcy_sources(p)
    u, s, off = (result[key] for key in ('predicted_field', 'source_weights', 'sensor_offset'))
    hx, hy = p['Lx'] / shape[0], p['Ly'] / shape[1]
    gamma, ambient = p['gamma_robin'], p['u_ambient']
    kx, ky = p['k_xx'], p['k_yy']
    x, y = kx[[0,-1], :], ky[:, [0,-1]]
    balance = (hy * gamma * x / (x + gamma * hx / 2) * (u[[0,-1], :] - ambient)).sum()
    balance += (hx * gamma * y / (y + gamma * hy / 2) * (u[:, [0,-1]] - ambient)).sum()
    balance += hx * hy * np.sum(p['sigma'] * u - np.einsum('m,mij->ij', s, p['basis']))
    assert abs(balance) < 1e-11
    shift = 7.5
    transpose = dict(p, Lx=p['Ly'], Ly=p['Lx'], k_xx=ky.T, k_yy=kx.T, k_xy=p['k_xy'].T,
                     sigma=p['sigma'].T, basis=p['basis'].transpose(0,2,1), sensor_ij=ij[:, ::-1], y_obs=p['y_obs']+shift)
    reflected = dict(p, k_xx=kx[::-1], k_yy=ky[::-1], k_xy=-p['k_xy'][::-1], sigma=p['sigma'][::-1],
                     basis=p['basis'][:, ::-1], sensor_ij=np.column_stack((shape[0]-1-ij[:,0], ij[:,1])))
    for transformed, mapped, moved_offset in ((transpose, u.T, off+shift), (reflected, u[::-1], off)):
        r = mod.infer_tensor_darcy_sources(transformed)
        np.testing.assert_allclose(r['source_weights'], s, rtol=0, atol=1e-9)
        np.testing.assert_allclose(r['predicted_field'], mapped, rtol=0, atol=1e-10)
        np.testing.assert_allclose(r['sensor_offset'], moved_offset, rtol=0, atol=1e-10)
    true_noise = p['y_obs'] - g['u_true'][tuple(ij.T)] - g['sensor_offset']
    clean = np.abs(true_noise) < .75
    d = response[flat[clean], 1:]
    d_centered = d - d.mean(axis=0)
    _, sv, vt = np.linalg.svd(d_centered, full_matrices=False)
    # Standard-error multiplier for the field mean; the common sensor offset is profiled out.
    mean_gain = np.linalg.norm((response[:, 1:].mean(axis=0) @ vt.T) / sv)
    worst = np.zeros(3)
    wrong_masks = 0
    for j in range(len(ij)):
        keep = np.arange(len(ij)) != j
        r = mod.infer_tensor_darcy_sources(dict(p, sensor_ij=ij[keep], y_obs=p['y_obs'][keep]))
        resid = p['y_obs'] - r['predicted_field'][tuple(ij.T)] - r['sensor_offset']
        wrong_masks += int((abs(resid[j]) < .75) != clean[j])
        worst = np.maximum(worst, [np.linalg.norm(r['source_weights']-g['source_weights'])/np.linalg.norm(g['source_weights']), abs(r['sensor_offset']-g['sensor_offset']), np.linalg.norm(r['predicted_field']-g['u_true'])/np.linalg.norm(g['u_true'])])
    print(f'public {index}: conservation {abs(balance):.3g}; transpose+shift and reflection pass; centered singular range [{sv[-1]:.4g}, {sv[0]:.4g}]; field-mean noise gain {mean_gain:.4g}; held-out spike-mask errors {wrong_masks}/{len(ij)}; worst delete-one source/offset/field {worst}')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_tensor_public.py
```
- **Container Exit Code**: `0` (`1038 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0: conservation 1.11e-15; transpose+shift and reflection pass; centered singular range [0.01058, 0.08232]; field-mean noise gain 1.719; held-out spike-mask errors 0/64; worst delete-one source/offset/field [0.02578387 0.00161275 0.00310691]
public 1: conservation 9.44e-16; transpose+shift and reflection pass; centered singular range [0.01143, 0.07764]; field-mean noise gain 1.62; held-out spike-mask errors 0/64; worst delete-one source/offset/field [0.04491711 0.00441172 0.00884732]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1c7f0f15fc8c63fa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1c7f0f15 (directory).
```

### Turn 8 — `submit_completed_task` (`18052 ms` LLM)
- **Rationale**: The exact artifact passed isolated-process checks of the output contract, forward references, conservation, axis exchange, reflection, and common-offset invariance. The public delete-one-sensor audit made no spike-classification errors, and the oracle clean-mask comparison showed that changing outlier rejection would not fix the weak-Robin limitation.
- **Alternatives Weighed**: `Quadratic refit after residual-gap spike rejection` (rejected: It did not improve the limiting weak-Robin field error and increased worst source error under high-leverage spikes on the public geometry from 7.65% to 9.00%.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`20` turns remaining; artifact status={'/holdout_task/tensor_darcy_robin.py': 'FILE:4339'}): Do NOT lock prematurely with `20` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tensor_darcy_robin.py`. The forward stencil matched both public reference fields within 3.0e-15. The worst public errors were 3.75% for source weights, 0.00389 for the sensor offset, and 0.772% for the field. Under weak Robin exchange, field error reached 35.64%; even with the true outliers removed, it reached 35.61% because the sensors poorly distinguish the field mean from a common offset.`

### Turn 9 — `execute_in_remote_docker` (`38459 ms` LLM)
- **Rationale**: Move the measured spikes to the highest-leverage sensors to test whether their placement can defeat the current fit. Check the previously untested non-square array against independent flux balance; a mismatch would require an indexing fix. Quantify how much the field mean can change while sensor predictions move by only the measured noise scale, to distinguish a fixable fitting problem from a lack of information.
- **Alternatives Weighed**: `Regularize the free sensor offset` (rejected: No calibration or offset prior is supplied; this could turn a valid common sensor shift into a spurious source change.)

```bash
cat > /tmp/audit_darcy_margin.py <<'PY'
from pathlib import Path
import numpy as np

# Reuse the independent flux balance and isolated import from the first stress test.
exec(Path('/tmp/check_tensor_darcy.py').read_text().split('worst_balance = 0.', 1)[0])
base = cases[0]['problem']
for number, case in enumerate(cases):
    p, g = case['problem'], case['gt']
    ij = p['sensor_ij']
    clean = abs(p['y_obs'] - g['u_true'][tuple(ij.T)] - g['sensor_offset']) < .75
    flat = np.ravel_multi_index(ij.T, g['u_true'].shape)
    a = mod._forward(p)[flat, 1:]
    centered = a[clean] - a[clean].mean(axis=0)
    leverage = np.sum(np.linalg.qr(centered)[0]**2, axis=1)
    clean_ids = np.flatnonzero(clean)
    # Move the observed spikes, including their signs, onto the leverage extremes;
    # retain all measured inlier residuals and the true field.
    spikes = (p['y_obs'] - g['u_true'][tuple(ij.T)] - g['sensor_offset'])[~clean]
    ranked = clean_ids[np.argsort(leverage)[-len(spikes):]]
    y0 = p['y_obs'].copy()
    y0[~clean] -= spikes
    worst = np.zeros(3)
    for k in range(len(spikes)):
        y = y0.copy()
        y[ranked] += np.roll(spikes, k)
        worst = np.maximum(worst, errors(dict(p, y_obs=y), g['source_weights'], g['sensor_offset'], g['u_true']))
    print('relocated measured spikes; public', number, 'worst source/offset/field', worst)

# Use a genuinely non-square array, not merely unequal physical side lengths.
p = dict(base, Lx=1.85, Ly=.65, gamma_robin=.16,
         **{key:base[key][:17, :22] for key in ('k_xx','k_xy','k_yy','sigma')},
         basis=base['basis'][:, :17, :22])
p['sensor_ij'] = base['sensor_ij'][base['sensor_ij'][:, 0] < 17]
nx, ny = p['k_xx'].shape
u = rng.normal(size=(nx, ny))
A,b,_ = mod._assemble(p)
mismatch = np.max(abs((A @ u.ravel()-b).reshape(u.shape)-direct_balance(p,u)))
assert mismatch < 1e-12
print('17x22 weak-Robin pointwise stencil mismatch', mismatch)

# Construct the source perturbation that maximizes the field-mean change at
# fixed centered sensor RMS. The offset cancels its mean exactly at sensors.
for label, p in [('public', base), ('weak rectangular', p)]:
    r = mod._forward(p)
    shape = p['k_xx'].shape
    ij = tuple(p['sensor_ij'].T)
    x = r[np.ravel_multi_index(ij, shape), 1:]
    n = len(x)
    d = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(d, full_matrices=False)
    c = r[:, 1:].mean(axis=0)
    dt = vt.T @ ((vt @ c) / s**2)
    dt *= .001 * np.sqrt(n) / np.linalg.norm(d @ dt)
    offset_change = -float(x.mean(axis=0) @ dt)
    field_change = r[:, 1:] @ dt
    sensor_change = x @ dt + offset_change
    print(label, 'indistinguishability: sensor RMS', np.linalg.norm(sensor_change)/np.sqrt(n),
          'field mean change', field_change.mean(), 'offset change', offset_change,
          'source perturbation norm', np.linalg.norm(dt))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_darcy_margin.py
```
- **Container Exit Code**: `0` (`490 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
relocated measured spikes; public 0 worst source/offset/field [0.01375943 0.00066675 0.00137057]
relocated measured spikes; public 1 worst source/offset/field [0.02768988 0.00289832 0.00559997]
17x22 weak-Robin pointwise stencil mismatch 7.105427357601002e-15
public indistinguishability: sensor RMS 0.0009999999999999998 field mean change 0.013591893259778123 offset change -0.014187015826600466 source perturbation norm 0.6659703613901343
weak rectangular indistinguishability: sensor RMS 0.0009999999999999996 field mean change 0.0681652390640492 offset change -0.06835912091252273 source perturbation norm 0.5053056883934823

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=1c7f0f15fc8c63fa (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_1c7f0f15 (directory).
```

### Turn 10 — `submit_completed_task` (`19861 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process checks of the return contract, conservation, reflection, axis exchange, and common-offset invariance. The non-square stencil matched an independent flux calculation within 7.11e-15, and relocating measured spikes onto high-leverage sensors kept field error below 0.56%. In the weak-Robin test, a source and offset change moved the field mean by 0.0682 while moving sensor predictions by only 0.001 RMS, confirming that spike rejection does not address the limiting uncertainty.
- **Alternatives Weighed**: `Quadratic refit after spike rejection` (rejected: It did not improve the limiting weak-Robin field error and raised worst source error under high-leverage spikes on the public geometry from 7.65% to 9.00%.); `Penalize the common sensor offset` (rejected: No offset calibration is supplied; penalizing the offset would assign a valid common sensor shift to the source field.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tensor_darcy_robin.py`. The forward stencil matched both public reference fields within 3.0e-15. Worst public errors were 3.75% for source weights, 0.00389 for the sensor offset, and 0.772% for the field. Under weak Robin exchange, field error reached 35.64%; removing the known outliers barely changed it (35.61%) because the sensors poorly distinguish the field mean from a common offset.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M89-HardStress-4",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
