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
| **Total Tokens (Prompt / Cached / Out)** | `336,887` (`325,718` / `80,128` / `11,169`) |
| **Total Episode Wall Time (s)** | `326.7s` |

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

### Turn 1 — `execute_in_remote_docker` (`15547 ms` LLM)
- **Rationale**: Inspect the fixture structure and ranges to identify the reference fields and the dimensions and scales that the forward and inverse tests must cover.
- **Alternatives Weighed**: `Reuse the scalar zero-ghost Darcy operator` (rejected: It omits transverse tensor flux and implements a different boundary condition.)

```bash
python3 - <<'PY'
import pickle
import numpy as np
with open('/holdout_task/public_tensor_cases.pkl','rb') as f:
    data=pickle.load(f)
def describe(x, depth=0):
    if isinstance(x, dict):
        for k,v in x.items():
            print(' '*depth, k, end=': ')
            if isinstance(v,(dict,list,tuple)):
                print(); describe(v, depth+2)
            elif isinstance(v,np.ndarray):
                print(v.shape,v.dtype, 'min/max', (v.min(),v.max()) if v.size else None)
            else: print(repr(v)[:200])
    elif isinstance(x,(tuple,list)):
        print(' '*depth,type(x).__name__,len(x))
        for v in x[:8]: describe(v,depth+2)
    else: print(' '*depth,type(x),repr(x)[:200])
describe(data)
PY
```
- **Container Exit Code**: `0` (`539 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cases: 
   list 2
     problem: 
       Lx: 1.2
       Ly: 1.0
       k_xx: (22, 22) float64 min/max (1.0363509910865978, 2.24219731617612)
       k_xy: (22, 22) float64 min/max (0.32176035377896317, 1.0437379455766584)
       k_yy: (22, 22) float64 min/max (0.5599556333252096, 1.5016724120449003)
       sigma: (22, 22) float64 min/max (0.250077479338843, 0.39325929752066113)
       gamma_robin: 1.65
       u_ambient: 0.4
       basis: (5, 22, 22) float64 min/max (1.971092819942404e-25, 0.9872077221034575)
       sensor_ij: (64, 2) int64 min/max (1, 19)
       y_obs: (64,) float64 min/max (-2.1769128090160015, 3.6960627392832364)
       tikhonov_alpha: 1e-08
     gt: 
       source_weights: (5,) float64 min/max (1.5419492109461137, 2.3767837702116843)
       sensor_offset: -0.2984695515641418
       u_true: (22, 22) float64 min/max (0.44071671737294293, 0.5397746205755551)
     problem: 
       Lx: 1.2
       Ly: 1.0
       k_xx: (22, 22) float64 min/max (1.0363509910865978, 2.24219731617612)
       k_xy: (22, 22) float64 min/max (0.32176035377896317, 1.0437379455766584)
       k_yy: (22, 22) float64 min/max (0.5599556333252096, 1.5016724120449003)
       sigma: (22, 22) float64 min/max (0.250077479338843, 0.39325929752066113)
       gamma_robin: 1.65
       u_ambient: 0.4
       basis: (5, 22, 22) float64 min/max (1.971092819942404e-25, 0.9872077221034575)
       sensor_ij: (64, 2) int64 min/max (1, 19)
       y_obs: (64,) float64 min/max (-1.9858091674250422, 3.910525197770965)
       tikhonov_alpha: 1e-08
     gt: 
       source_weights: (5,) float64 min/max (1.1856653838234414, 2.419012935743922)
       sensor_offset: -0.08045838188544918
       u_true: (22, 22) float64 min/max (0.4346920903419648, 0.5257877677314957)
```

### Turn 2 — `execute_in_remote_docker` (`49308 ms` LLM)
- **Rationale**: Check the known-source forward field separately from the inverse fit, then use sensor singular values and residual scales to distinguish a stencil error from regression sensitivity.
- **Alternatives Weighed**: `Use CG with the mounted scalar grid operator` (rejected: The scalar operator omits transverse flux, and the specified clamped variable-coefficient stencil does not guarantee symmetry.)

```bash
cat > /holdout_task/tensor_darcy_robin.py <<'PY'
"""Finite-volume tensor Darcy source inversion with robust sensor fitting."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _assemble(problem):
    """Return the integrated cell operator and ambient boundary load."""
    kxx = np.asarray(problem['k_xx'], dtype=float)
    kyy = np.asarray(problem['k_yy'], dtype=float)
    kxy = np.asarray(problem['k_xy'], dtype=float)
    nx, ny = kxx.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    ids = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, vals = [], [], []

    def add(row, col, val):
        rows.append(row.ravel())
        cols.append(col.ravel())
        vals.append(val.ravel())

    def face(left, right, conductance, cross, up0, down0, up1, down1):
        # Every face contributes equal and opposite integrated outflows.
        for col, coefficient in ((left, conductance), (right, -conductance),
                                 (up0, -cross), (down0, cross),
                                 (up1, -cross), (down1, cross)):
            add(left, col, coefficient)
            add(right, col, -coefficient)

    yp = np.minimum(np.arange(ny) + 1, ny - 1)
    ym = np.maximum(np.arange(ny) - 1, 0)
    xp = np.minimum(np.arange(nx) + 1, nx - 1)
    xm = np.maximum(np.arange(nx) - 1, 0)
    face(ids[:-1], ids[1:], (2 * hy / hx) * kxx[:-1] * kxx[1:] / (kxx[:-1] + kxx[1:]),
         (kxy[:-1] + kxy[1:]) / 8, ids[:-1, yp], ids[:-1, ym], ids[1:, yp], ids[1:, ym])
    face(ids[:, :-1], ids[:, 1:], (2 * hx / hy) * kyy[:, :-1] * kyy[:, 1:] / (kyy[:, :-1] + kyy[:, 1:]),
         (kxy[:, :-1] + kxy[:, 1:]) / 8, ids[xp, :-1], ids[xm, :-1], ids[xp, 1:], ids[xm, 1:])

    gamma = problem['gamma_robin']
    wall = np.zeros_like(kxx)
    for i in (0, -1):
        wall[i] += hy * gamma * kxx[i] / (0.5 * hx * gamma + kxx[i])
    for j in (0, -1):
        wall[:, j] += hx * gamma * kyy[:, j] / (0.5 * hy * gamma + kyy[:, j])
    area = hx * hy
    add(ids, ids, wall + area * np.asarray(problem['sigma']))
    operator = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(ids.size, ids.size)).tocsc()
    return operator, (wall * problem['u_ambient']).ravel(), area


def _responses(problem):
    operator, ambient, area = _assemble(problem)
    basis = np.asarray(problem['basis'], dtype=float)
    m, nx, ny = basis.shape
    rhs = np.column_stack((ambient, area * basis.reshape(m, nx * ny).T))
    return splu(operator).solve(rhs).reshape(nx, ny, m + 1)


def _fit_sources(design, target, alpha):
    """Robustly screen sensor spikes, then fit the retained observations."""
    n, p = design.shape
    ridge = np.sqrt(alpha) * np.eye(p)[:-1]
    zeros = np.zeros(p - 1)

    def fit(weight):
        root = np.sqrt(weight)
        return np.linalg.lstsq(np.vstack((root[:, None] * design, ridge)),
                               np.r_[root * target, zeros], rcond=None)[0]

    coef = np.zeros(p)
    coef[-1] = np.median(target)
    floor = 1e-12 * max(1., np.max(np.abs(target)))
    for _ in range(40):
        residual = target - design @ coef
        scale = max(floor, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        update = fit(1 / (1 + (residual / (2.5 * scale)) ** 2))
        if np.linalg.norm(update - coef) <= 1e-9 * (1 + np.linalg.norm(coef)):
            coef = update
            break
        coef = update
    residual = target - design @ coef
    median = np.median(residual)
    scale = max(floor, 1.4826 * np.median(np.abs(residual - median)))
    keep = np.abs(residual - median) <= 5 * scale
    return fit(keep) if np.count_nonzero(keep) >= p else coef


def infer_tensor_darcy_sources(problem: dict) -> dict:
    """Recover source amplitudes, a common sensor offset, and the pressure grid."""
    response = _responses(problem)
    i, j = np.asarray(problem['sensor_ij']).T
    sensors = response[i, j]
    design = np.column_stack((sensors[:, 1:], np.ones(len(sensors))))
    target = np.asarray(problem['y_obs'], dtype=float) - sensors[:, 0]
    coef = _fit_sources(design, target, float(problem['tikhonov_alpha']))
    return {'source_weights': coef[:-1], 'sensor_offset': float(coef[-1]),
            'predicted_field': response[..., 0] + response[..., 1:] @ coef[:-1]}
PY
python3 -I - <<'PY'
import sys, pickle, numpy as np
sys.path.insert(0,'/holdout_task')
from tensor_darcy_robin import _assemble, _responses, infer_tensor_darcy_sources
with open('/holdout_task/public_tensor_cases.pkl','rb') as f: cases=pickle.load(f)['cases']
for n,c in enumerate(cases):
 p,g=c['problem'],c['gt']; a,b,area=_assemble(p); r=_responses(p); w=g['source_weights']; u=g['u_true']; i,j=p['sensor_ij'].T
 err=p['y_obs']-u[i,j]-g['sensor_offset']; s=np.linalg.svd(r[i,j,1:]-r[i,j,1:].mean(0),compute_uv=False)
 rhs=b+area*(w@p['basis'].reshape(len(w),-1)); z=infer_tensor_darcy_sources(p)
 print(n,'forward max',np.max(abs(r[...,0]+r[...,1:]@w-u)),'FV residual',np.linalg.norm(a@u.ravel()-rhs)/np.linalg.norm(rhs),'singular values',s)
 print('noise sorted absolute',np.round(np.sort(abs(err)),7))
 print('weights truth/fit',w,z['source_weights'],'weight relative',np.linalg.norm(z['source_weights']-w)/np.linalg.norm(w),'offset abs',abs(z['sensor_offset']-g['sensor_offset']),'field max',np.max(abs(z['predicted_field']-u)))
PY
```
- **Container Exit Code**: `0` (`744 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tensor_darcy_robin.py']`
- **Stdout (tail)**:
```text
0e-04
 4.3750000e-04 4.9320000e-04 5.2770000e-04 5.4440000e-04 5.4670000e-04
 5.6560000e-04 5.8010000e-04 5.8170000e-04 5.9310000e-04 6.1840000e-04
 6.5170000e-04 6.9220000e-04 7.3860000e-04 7.7210000e-04 8.0110000e-04
 8.5750000e-04 8.6100000e-04 8.7540000e-04 9.4880000e-04 9.4950000e-04
 1.0236000e-03 1.0248000e-03 1.0336000e-03 1.0427000e-03 1.1198000e-03
 1.1401000e-03 1.1660000e-03 1.1797000e-03 1.1917000e-03 1.2276000e-03
 1.2419000e-03 1.2704000e-03 1.4170000e-03 1.5057000e-03 1.5198000e-03
 1.6310000e-03 1.6949000e-03 1.7967000e-03 2.1371000e-03 2.1790000e-03
 2.2350000e-03 2.3477000e-03 2.4352000e-03 3.3352000e-03 1.8575500e+00
 1.9568386e+00 2.3103425e+00 2.4166891e+00 3.4630066e+00]
weights truth/fit [2.35597846 2.37678377 2.35604765 1.54194921 1.54697609] [2.35588063 2.34606471 2.33962852 1.54150648 1.59906802] weight relative 0.013508156233220322 offset abs 0.0002478158466362457 field max 0.0010078598157561292
1 forward max 4.6629367034256575e-15 FV residual 3.81378837076466e-14 singular values [0.07928371 0.06079007 0.05055262 0.02550335 0.01168778]
noise sorted absolute [5.4100000e-05 5.5000000e-05 1.2390000e-04 1.3520000e-04 1.5730000e-04
 1.6830000e-04 2.0910000e-04 2.1940000e-04 2.5460000e-04 2.5630000e-04
 2.6460000e-04 2.7390000e-04 2.8340000e-04 2.8870000e-04 3.2340000e-04
 3.5290000e-04 3.6250000e-04 3.9440000e-04 3.9770000e-04 4.6910000e-04
 5.3350000e-04 5.6910000e-04 5.9160000e-04 6.0990000e-04 6.1960000e-04
 6.4360000e-04 7.2430000e-04 7.5600000e-04 7.8240000e-04 7.8280000e-04
 9.1250000e-04 9.1420000e-04 9.3550000e-04 9.9300000e-04 9.9830000e-04
 1.0033000e-03 1.0536000e-03 1.1633000e-03 1.2263000e-03 1.2558000e-03
 1.2750000e-03 1.2962000e-03 1.3611000e-03 1.4065000e-03 1.4475000e-03
 1.4766000e-03 1.5907000e-03 1.6135000e-03 1.6272000e-03 1.6402000e-03
 1.6694000e-03 1.8628000e-03 1.9769000e-03 2.0198000e-03 2.0377000e-03
 2.3503000e-03 2.7113000e-03 2.7649000e-03 3.0005000e-03 2.0258739e+00
 2.3935368e+00 2.4512430e+00 2.8965833e+00 3.5086271e+00]
weights truth/fit [1.26117417 2.41901294 1.31414903 1.18566538 2.39342535] [1.33468305 2.47498164 1.40114369 1.26683809 2.39096852] weight relative 0.03731342313382649 offset abs 0.0038844796359441963 field max 0.004579641252250299

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4a4d1fca80d74223 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_4a4d1fca (directory).
```

### Turn 3 — `execute_in_remote_docker` (`48370 ms` LLM)
- **Rationale**: The public forward fields already agree to roundoff. An independent face balance tests whether that agreement extends to rectangular grids, both signs of the cross term, and weak and strong Robin coupling. Sensor holdouts and injected spikes test whether particular sensor placements destabilize the inverse fit; if they do, compare with the known-clean fit before changing the estimator.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from tensor_darcy_robin import _assemble, _responses, _fit_sources, infer_tensor_darcy_sources
with open('/holdout_task/public_tensor_cases.pkl','rb') as f: cases=pickle.load(f)['cases']

def balance(p, u):
    xx, xy, yy = (p[k] for k in ('k_xx','k_xy','k_yy'))
    nx, ny = u.shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    g, amb = p['gamma_robin'], p['u_ambient']
    out = hx*hy*p['sigma']*u
    def v(i,j): return u[min(max(i,0),nx-1),min(max(j,0),ny-1)]
    for i in range(nx):
        for j in range(ny):
            for d in (-1,1):
                k = i+d
                if 0 <= k < nx:
                    out[i,j] += hy/hx*2/(1/xx[i,j]+1/xx[k,j])*(u[i,j]-u[k,j]) - d*(xy[i,j]+xy[k,j])/8*(v(i,j+1)-v(i,j-1)+v(k,j+1)-v(k,j-1))
                else:
                    out[i,j] += hy/(1/g+hx/(2*xx[i,j]))*(u[i,j]-amb)
                k = j+d
                if 0 <= k < ny:
                    out[i,j] += hx/hy*2/(1/yy[i,j]+1/yy[i,k])*(u[i,j]-u[i,k]) - d*(xy[i,j]+xy[i,k])/8*(v(i+1,j)-v(i-1,j)+v(i+1,k)-v(i-1,k))
                else:
                    out[i,j] += hx/(1/g+hy/(2*yy[i,j]))*(u[i,j]-amb)
    return out

rng=np.random.default_rng(489)
for n,c in enumerate(cases):
    p,g=c['problem'],c['gt']; i,j=p['sensor_ij'].T
    r=_responses(p); d=np.c_[r[i,j,1:],np.ones(len(i))]; target=p['y_obs']-r[i,j,0]
    w=g['source_weights']; off=g['sensor_offset']; u=g['u_true']; n_s=len(i); alpha=p['tikhonov_alpha']
    def metrics(coef):
        return [np.linalg.norm(coef[:-1]-w)/np.linalg.norm(w),abs(coef[-1]-off), np.max(abs(r[...,0]+r[...,1:]@coef[:-1]-u))]
    clean=abs(p['y_obs']-u[i,j]-off)<0.1
    cv=[]
    for k in range(n_s):
        train=np.arange(n_s)!=k
        coef=_fit_sources(d[train],target[train],alpha)
        cv.append([*metrics(coef),abs(d[k]@coef-target[k])])
    cv=np.array(cv)
    print(n,'LOSO worst [relative weight, abs offset, max field]',cv[:,:3].max(0),'clean CV RMSE/max',np.sqrt(np.mean(cv[clean,3]**2)),cv[clean,3].max())
    trial=[]; disagreement=[]
    rank=np.sum(np.linalg.qr(d)[0]**2,axis=1)
    for t in range(200):
        idx=rng.choice(n_s,round(n_s*.12),replace=False) if t else np.argsort(rank)[-round(n_s*.12):]
        y=d@np.r_[w,off]+rng.normal(0,0.001,n_s)
        y[idx]+=rng.uniform(1.5,4,len(idx))*rng.choice([-1,1],len(idx))
        coef=_fit_sources(d,y,alpha)
        keep=np.ones(n_s,dtype=bool); keep[idx]=False
        oracle=_fit_sources(d[keep],y[keep],alpha)
        trial.append(metrics(coef)); disagreement.append(np.max(abs(coef-oracle)))
    print(n,'12% contamination 95th/worst',np.quantile(trial,.95,axis=0),np.max(trial,axis=0),'max oracle difference',max(disagreement))
    assert np.max(cv[:,0])<.10 and cv[clean,3].max()<.008
    assert np.max(trial,axis=0)[0]<.12 and max(disagreement)<.05

for shape,sign,gamma in [((7,13),-1,.03),((17,6),1,180.),((1,9),-1,2.),((10,1),1,2.)]:
    nx,ny=shape
    xx=np.exp(rng.uniform(-2,2,shape)); yy=np.exp(rng.uniform(-2,2,shape)); xy=sign*.85*np.sqrt(xx*yy)
    p=dict(Lx=1.7,Ly=.72,k_xx=xx,k_yy=yy,k_xy=xy,sigma=rng.uniform(.1,.4,shape),gamma_robin=gamma,u_ambient=-.65)
    u=rng.normal(size=shape); a,b,area=_assemble(p)
    err=np.max(abs((a@u.ravel()-b).reshape(shape)-balance(p,u)))
    tol=1e-12*max(1,np.max(abs(a@u.ravel())))
    assert err<tol
    iid=np.array(np.unravel_index(np.arange(nx*ny),shape)).T
    # Manufacture basis loads from independently applied physical balance.
    ub=rng.normal(size=(3,nx,ny)); p0=dict(p,u_ambient=0.)
    p['basis']=np.array([balance(p0,q)/area for q in ub]); p['sensor_ij']=iid; p['tikhonov_alpha']=0.
    w=np.array([.9,-1.4,1.2]); off=-.23
    base=_responses(p)[...,0]; truth=base+np.einsum('k,kij->ij',w,ub)
    p['y_obs']=truth.ravel()+off
    worst=np.argsort(np.sum(np.linalg.qr(np.c_[ub.reshape(3,-1).T,np.ones(len(iid))])[0]**2,axis=1))[-max(1,len(iid)//10):]
    p['y_obs'][worst]+=3.
    z=infer_tensor_darcy_sources(p)
    e=np.max(abs(z['predicted_field']-truth))
    print(shape,'sign',sign,'gamma',gamma,'balance max',err,'manufactured field max',e)
    assert set(z)=={'source_weights','sensor_offset','predicted_field'} and type(z['sensor_offset']) is float
    assert z['source_weights'].shape==(3,) and z['predicted_field'].shape==shape
    assert np.max(abs(z['source_weights']-w))<1e-8 and abs(z['sensor_offset']-off)<1e-8 and e<1e-8
PY
```
- **Container Exit Code**: `1` (`940 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 LOSO worst [relative weight, abs offset, max field] [0.0229191  0.00173868 0.00246031] clean CV RMSE/max 0.001212744202627844 0.003950924351257046
0 12% contamination 95th/worst [0.04120853 0.00351192 0.00434326] [0.05236788 0.00551027 0.00672228] max oracle difference 0.06355320852340851

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=4a4d1fca80d74223 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_4a4d1fca (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 56, in <module>
AssertionError
```

### Turn 4 — `execute_in_remote_docker` (`53544 ms` LLM)
- **Rationale**: Public inlier noise is near 0.001 while spikes exceed 1.8, so widen the final screen from five to eight robust scales to avoid dropping clean tail observations. Compare directly against ridge regression on known clean sensors and finish the independent stencil checks that the previous assertion interrupted.
- **Alternatives Weighed**: `Change the finite-volume operator` (rejected: The known-source public fields agree to roundoff; the observed disagreement is in the sensor fit.)

```bash
cp /holdout_task/tensor_darcy_robin.py /tmp/tensor_v01.py
sed -i 's/<= 5 \* scale/<= 8 * scale/' /holdout_task/tensor_darcy_robin.py
cat > /tmp/verify_tensor.py <<'PY'
import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from tensor_darcy_robin import _assemble, _responses, _fit_sources, infer_tensor_darcy_sources
with open('/holdout_task/public_tensor_cases.pkl','rb') as f:
    cases=pickle.load(f)['cases']

def balance(p,u):
    xx,xy,yy=(p[k] for k in ('k_xx','k_xy','k_yy'))
    nx,ny=u.shape; hx,hy=p['Lx']/nx,p['Ly']/ny
    g,amb=p['gamma_robin'],p['u_ambient']
    out=hx*hy*p['sigma']*u
    def v(i,j): return u[min(max(i,0),nx-1),min(max(j,0),ny-1)]
    for i in range(nx):
        for j in range(ny):
            for d in (-1,1):
                k=i+d
                if 0<=k<nx:
                    out[i,j]+=hy/hx*2/(1/xx[i,j]+1/xx[k,j])*(u[i,j]-u[k,j])-d*(xy[i,j]+xy[k,j])/8*(v(i,j+1)-v(i,j-1)+v(k,j+1)-v(k,j-1))
                else:
                    out[i,j]+=hy/(1/g+hx/(2*xx[i,j]))*(u[i,j]-amb)
                k=j+d
                if 0<=k<ny:
                    out[i,j]+=hx/hy*2/(1/yy[i,j]+1/yy[i,k])*(u[i,j]-u[i,k])-d*(xy[i,j]+xy[i,k])/8*(v(i+1,j)-v(i-1,j)+v(i+1,k)-v(i-1,k))
                else:
                    out[i,j]+=hx/(1/g+hy/(2*yy[i,j]))*(u[i,j]-amb)
    return out

rng=np.random.default_rng(489)
fail=[]
for n,c in enumerate(cases):
    p,g=c['problem'],c['gt']; i,j=p['sensor_ij'].T
    r=_responses(p); d=np.c_[r[i,j,1:],np.ones(len(i))]; target=p['y_obs']-r[i,j,0]
    w=g['source_weights']; off=g['sensor_offset']; u=g['u_true']; n_s=len(i); alpha=p['tikhonov_alpha']
    def metrics(coef):
        return [np.linalg.norm(coef[:-1]-w)/np.linalg.norm(w),abs(coef[-1]-off),np.max(abs(r[...,0]+r[...,1:]@coef[:-1]-u))]
    z=infer_tensor_darcy_sources(p)
    print(n,'public [relative weight, abs offset, max field]',metrics(np.r_[z['source_weights'],z['sensor_offset']]), 'known-source forward max',np.max(abs(r[...,0]+r[...,1:]@w-u)))
    clean=abs(p['y_obs']-u[i,j]-off)<0.1
    cv=[]
    for k in range(n_s):
        train=np.arange(n_s)!=k
        coef=_fit_sources(d[train],target[train],alpha)
        cv.append([*metrics(coef),abs(d[k]@coef-target[k])])
    cv=np.array(cv)
    print(n,'LOSO worst',cv[:,:3].max(0),'clean CV RMSE/max',np.sqrt(np.mean(cv[clean,3]**2)),cv[clean,3].max())
    trial=[]; disagreement=[]
    leverage=np.sum(np.linalg.qr(d)[0]**2,axis=1)
    for t in range(200):
        k=round(n_s*.12)
        idx=rng.choice(n_s,k,replace=False) if t else np.argsort(leverage)[-k:]
        y=d@np.r_[w,off]+rng.normal(0,0.001,n_s)
        y[idx]+=rng.uniform(1.5,4,len(idx))*rng.choice([-1,1],len(idx))
        coef=_fit_sources(d,y,alpha)
        keep=np.ones(n_s,dtype=bool); keep[idx]=False
        ridge=np.sqrt(alpha)*np.eye(d.shape[1])[:-1]
        oracle=np.linalg.lstsq(np.vstack((d[keep],ridge)),np.r_[y[keep],np.zeros(len(ridge))],rcond=None)[0]
        trial.append(metrics(coef)); disagreement.append(np.max(abs(coef-oracle)))
    print(n,'12% contamination worst',np.max(trial,axis=0),'max clean-oracle difference',max(disagreement))
    if not (cv[:,0].max()<.10 and cv[clean,3].max()<.008 and np.max(trial,axis=0)[0]<.12 and max(disagreement)<.05): fail.append(('sensor',n))

for shape,sign,gamma in [((7,13),-1,.03),((17,6),1,180.),((1,9),-1,2.),((10,1),1,2.)]:
    nx,ny=shape
    xx=np.exp(rng.uniform(-2,2,shape)); yy=np.exp(rng.uniform(-2,2,shape)); xy=sign*.85*np.sqrt(xx*yy)
    p=dict(Lx=1.7,Ly=.72,k_xx=xx,k_yy=yy,k_xy=xy,sigma=rng.uniform(.1,.4,shape),gamma_robin=gamma,u_ambient=-.65)
    u=rng.normal(size=shape); a,b,area=_assemble(p)
    err=np.max(abs((a@u.ravel()-b).reshape(shape)-balance(p,u)))
    tol=1e-12*max(1,np.max(abs(a@u.ravel())))
    iid=np.array(np.unravel_index(np.arange(nx*ny),shape)).T
    ub=rng.normal(size=(3,nx,ny)); p0=dict(p,u_ambient=0.)
    p['basis']=np.array([balance(p0,q)/area for q in ub]); p['sensor_ij']=iid; p['tikhonov_alpha']=0.
    w=np.array([.9,-1.4,1.2]); off=-.23
    base=_responses(p)[...,0]; truth=base+np.einsum('k,kij->ij',w,ub)
    p['y_obs']=truth.ravel()+off
    worst=np.argsort(np.sum(np.linalg.qr(np.c_[ub.reshape(3,-1).T,np.ones(len(iid))])[0]**2,axis=1))[-max(1,len(iid)//10):]
    p['y_obs'][worst]+=3.
    z=infer_tensor_darcy_sources(p)
    e=np.max(abs(z['predicted_field']-truth))
    print(shape,'sign',sign,'gamma',gamma,'balance max',err,'manufactured field max',e,'weight max',np.max(abs(z['source_weights']-w)),'offset abs',abs(z['sensor_offset']-off))
    assert set(z)=={'source_weights','sensor_offset','predicted_field'} and type(z['sensor_offset']) is float
    assert z['source_weights'].shape==(3,) and z['predicted_field'].shape==shape
    if not (err<tol and np.max(abs(z['source_weights']-w))<1e-8 and abs(z['sensor_offset']-off)<1e-8 and e<1e-8): fail.append(('manufactured',shape))
assert not fail, fail
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_tensor.py
```
- **Container Exit Code**: `1` (`1089 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tensor_darcy_robin.py']`
- **Stdout (tail)**:
```text
0 public [relative weight, abs offset, max field] [0.013508156233220322, 0.0002478158466362457, 0.0010078598157561292] known-source forward max 4.773959005888173e-15
0 LOSO worst [0.0229191  0.00173868 0.00246031] clean CV RMSE/max 0.001212744202627844 0.003950924351257046
0 12% contamination worst [0.05236788 0.00551027 0.00672228] max clean-oracle difference 7.815970093361102e-14
1 public [relative weight, abs offset, max field] [0.03731342313382649, 0.0038844796359441963, 0.004579641252250299] known-source forward max 4.6629367034256575e-15
1 LOSO worst [0.04446097 0.00438796 0.00529212] clean CV RMSE/max 0.0012058842394358671 0.003233057827024835
1 12% contamination worst [0.07054438 0.00575364 0.00693482] max clean-oracle difference 6.750155989720952e-14
(7, 13) sign -1 gamma 0.03 balance max 1.4210854715202004e-14 manufactured field max 4.2557624091443813e-13 weight max 2.90878432451791e-14 offset abs 1.1507461650239748e-13
(17, 6) sign 1 gamma 180.0 balance max 3.552713678800501e-15 manufactured field max 2.7200464103316335e-14 weight max 2.3314683517128287e-15 offset abs 6.106226635438361e-16
(1, 9) sign -1 gamma 2.0 balance max 1.4210854715202004e-14 manufactured field max 1.6247346186777687 weight max 1.434081683447311 offset abs 0.927953464172721
(10, 1) sign 1 gamma 2.0 balance max 7.105427357601002e-15 manufactured field max 5.329070518200751e-15 weight max 2.220446049250313e-15 offset abs 1.7763568394002505e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a1d86a99c0aa1c08 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_a1d86a99 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/verify_tensor.py", line 83, in <module>
    assert not fail, fail
           ^^^^^^^^
AssertionError: [('manufactured', (1, 9))]
```

### Turn 5 — `execute_in_remote_docker` (`25783 ms` LLM)
- **Rationale**: Change only initialization in the regime that failed, then rerun the isolated verification to test whether the sparse-layout error is fixed without degrading the public cases or the independent stencil checks.
- **Alternatives Weighed**: `Adjust the final outlier threshold again` (rejected: A threshold applied after a distorted fit does not reliably recover the masked spike.)
- **ATMS NOGOOD DECLARED**: Refuted `A median-offset start followed by residual-only Cauchy reweighting suffices to isolate a single large spike on a small sensor layout.` | Witness: `On the 1-by-9 manufactured case, one spike at the highest-leverage sensor produced 1.434 maximum weight error despite independent face balance agreeing to 1.42e-14.` | Blanket Diagnosis: `The failure is in the sensor fit, not the PDE stencil: a high-leverage row can distort the source estimate before residual screening begins.` | Sublated Pivot: `Keep the exact stencil and ridge refit; add leave-one-out starting candidates for small sensor layouts and rank them by median absolute residual.`

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/tensor_darcy_robin.py')
s = p.read_text()
old = '''    coef[-1] = np.median(target)
    floor = 1e-12 * max(1., np.max(np.abs(target)))
'''
new = '''    coef[-1] = np.median(target)
    if p < n <= 3 * p:
        # With few redundant rows, a single high-leverage spike can pull
        # the initial fit far enough to hide itself from residual weights.
        # Compare leave-one-out starts on the full set by median residual.
        best = np.median(np.abs(target - design @ coef))
        keep = np.ones(n, dtype=bool)
        for k in range(n):
            keep[k] = False
            candidate = fit(keep)
            keep[k] = True
            residual = target - design @ candidate
            shift = np.median(residual)
            score = np.median(np.abs(residual - shift))
            if score < best:
                best = score
                candidate[-1] += shift
                coef = candidate
    floor = 1e-12 * max(1., np.max(np.abs(target)))
'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_tensor.py
```
- **Container Exit Code**: `0` (`1117 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/tensor_darcy_robin.py']`
- **Stdout (tail)**:
```text
0 public [relative weight, abs offset, max field] [0.013508156233220322, 0.0002478158466362457, 0.0010078598157561292] known-source forward max 4.773959005888173e-15
0 LOSO worst [0.0229191  0.00173868 0.00246031] clean CV RMSE/max 0.001212744202627844 0.003950924351257046
0 12% contamination worst [0.05236788 0.00551027 0.00672228] max clean-oracle difference 7.815970093361102e-14
1 public [relative weight, abs offset, max field] [0.03731342313382649, 0.0038844796359441963, 0.004579641252250299] known-source forward max 4.6629367034256575e-15
1 LOSO worst [0.04446097 0.00438796 0.00529212] clean CV RMSE/max 0.0012058842394358671 0.003233057827024835
1 12% contamination worst [0.07054438 0.00575364 0.00693482] max clean-oracle difference 6.750155989720952e-14
(7, 13) sign -1 gamma 0.03 balance max 1.4210854715202004e-14 manufactured field max 4.2557624091443813e-13 weight max 2.90878432451791e-14 offset abs 1.1507461650239748e-13
(17, 6) sign 1 gamma 180.0 balance max 3.552713678800501e-15 manufactured field max 2.7200464103316335e-14 weight max 2.3314683517128287e-15 offset abs 6.106226635438361e-16
(1, 9) sign -1 gamma 2.0 balance max 1.4210854715202004e-14 manufactured field max 9.769962616701378e-15 weight max 6.772360450213455e-15 offset abs 5.440092820663267e-15
(10, 1) sign 1 gamma 2.0 balance max 7.105427357601002e-15 manufactured field max 5.329070518200751e-15 weight max 2.220446049250313e-15 offset abs 1.7763568394002505e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=60618da636b39c23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_60618da6 (directory).
```

### Turn 6 — `submit_completed_task` (`12967 ms` LLM)
- **Rationale**: The exact submitted artifact passed isolated verification. Leave-one-out initialization reduced the failing nine-sensor case's maximum field error from 1.625 to 9.77e-15 without changing the public results. The rectangular-grid face-balance and return-schema checks also passed.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`22` turns remaining; artifact status={'/holdout_task/tensor_darcy_robin.py': 'FILE:5044'}): Do NOT lock prematurely with `22` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tensor_darcy_robin.py`. Across the two public cases, the worst relative source-weight error was 3.73%, the worst absolute sensor-offset error was 0.00388, and the worst maximum field error was 0.00458. With known source weights, the solver reproduced both reference fields to within 4.8e-15.`

### Turn 7 — `execute_in_remote_docker` (`33916 ms` LLM)
- **Rationale**: Use the public measurements to test two unresolved risks: a sign or axis error exposed by reflection and transpose, and dependence on a group of high-leverage sensors. If the holdout fails, comparison with the known-clean fit distinguishes spike masking from loss of source information.
- **Alternatives Weighed**: `Test batch centering, relay modes, or drift terms` (rejected: The Darcy inputs specify neither batch labels nor relay states or temporal drift. The relevant nuisance parameter is the fitted common sensor offset.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from tensor_darcy_robin import _responses, _fit_sources, infer_tensor_darcy_sources
with open('/holdout_task/public_tensor_cases.pkl','rb') as f:
    cases = pickle.load(f)['cases']

for k, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    i, j = p['sensor_ij'].T
    response = _responses(p)
    design = np.c_[response[i,j,1:], np.ones(len(i))]
    target = p['y_obs'] - response[i,j,0]
    w, offset, field = gt['source_weights'], gt['sensor_offset'], gt['u_true']
    clean = abs(p['y_obs'] - field[i,j] - offset) < .1
    ridge = np.sqrt(p['tikhonov_alpha']) * np.eye(design.shape[1])[:-1]
    def oracle(mask):
        return np.linalg.lstsq(np.vstack((design[mask], ridge)), np.r_[target[mask], np.zeros(len(ridge))], rcond=None)[0]
    result = infer_tensor_darcy_sources(p)
    coef = np.r_[result['source_weights'], result['sensor_offset']]
    oracle_gap = np.max(abs(coef - oracle(clean)))
    assert oracle_gap < 1e-9, (k, oracle_gap)
    # Hold out the most informative clean sensors together, without using their values to fit.
    leverage = np.sum(np.linalg.qr(design, mode='reduced')[0] ** 2, axis=1)
    clean_ids = np.flatnonzero(clean)
    hold = clean_ids[np.argsort(leverage[clean_ids])[-max(1,len(i)//10):]]
    train = np.ones(len(i), dtype=bool)
    train[hold] = False
    fit = _fit_sources(design[train], target[train], p['tikhonov_alpha'])
    hold_err = np.max(abs(design[hold] @ fit - target[hold]))
    hold_gap = np.max(abs(fit - oracle(train & clean)))
    weight_err = np.linalg.norm(fit[:-1]-w) / np.linalg.norm(w)
    field_err = np.max(abs(response[...,0] + response[...,1:] @ fit[:-1] - field))
    assert hold_err < .008 and weight_err < .10 and hold_gap < 1e-9, (k, hold_err,weight_err,hold_gap)
    # Reflect x: K_xy changes sign. Transpose: swap K_xx/K_yy and domain lengths.
    reflect = dict(p, **{name: p[name][::-1,:] for name in ('k_xx','k_yy','sigma')})
    reflect.update(k_xy=-p['k_xy'][::-1,:], basis=p['basis'][:,::-1,:], sensor_ij=np.c_[field.shape[0]-1-i,j])
    transpose = dict(p, Lx=p['Ly'], Ly=p['Lx'], k_xx=p['k_yy'].T, k_yy=p['k_xx'].T,
                     k_xy=p['k_xy'].T, sigma=p['sigma'].T, basis=p['basis'].transpose(0,2,1), sensor_ij=np.c_[j,i])
    symmetry_err = 0.
    for mapped, undo in ((reflect, lambda a: a[::-1,:]), (transpose, lambda a: a.T)):
        out = infer_tensor_darcy_sources(mapped)
        symmetry_err = max(symmetry_err, np.max(abs(undo(out['predicted_field'])-result['predicted_field'])))
        np.testing.assert_allclose(out['source_weights'], result['source_weights'], rtol=0, atol=1e-9)
        assert abs(out['sensor_offset']-result['sensor_offset']) < 1e-10
    assert symmetry_err < 1e-10
    print(f'case {k}: spikes={(~clean).sum()}, clean-oracle gap={oracle_gap:.3g}, high-leverage holdout: max sensor error={hold_err:.6g}, relative weight error={weight_err:.4%}, max field error={field_err:.6g}, oracle gap={hold_gap:.3g}; symmetry field error={symmetry_err:.3g}')
PY
```
- **Container Exit Code**: `0` (`553 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case 0: spikes=5, clean-oracle gap=2.22e-14, high-leverage holdout: max sensor error=0.00179274, relative weight error=1.9483%, max field error=0.0012912, oracle gap=4.26e-14; symmetry field error=3e-15
case 1: spikes=5, clean-oracle gap=2e-15, high-leverage holdout: max sensor error=0.00293803, relative weight error=3.5321%, max field error=0.00408672, oracle gap=3.77e-15; symmetry field error=2.89e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=60618da636b39c23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_60618da6 (directory).
```

### Turn 8 — `submit_completed_task` (`14296 ms` LLM)
- **Rationale**: The saved artifact passed isolated verification, including the previously failing nine-sensor case. The public fits matched ridge fits on known-clean sensors to roundoff, even after withholding the highest-leverage clean sensors as a group. Reflection and axis-transpose checks also agreed to roundoff.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`20` turns remaining; artifact status={'/holdout_task/tensor_darcy_robin.py': 'FILE:5044'}): Do NOT lock prematurely with `20` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tensor_darcy_robin.py`. With known source weights, it reproduced both public reference fields within 4.8e-15. In the public inverse tests, the worst relative source-weight error was 3.73%, the worst absolute sensor-offset error was 0.00388, and the largest pointwise field error was 0.00458. Under 12% injected sensor contamination, the worst relative source-weight error was 7.06%.`

### Turn 9 — `execute_in_remote_docker` (`29470 ms` LLM)
- **Rationale**: Test whether coordinated spikes can mimic a source-response mode and evade the residual screen. Preserve measured inlier noise and compare each fit with the ridge fit on the same retained sensors. A material prediction gap would justify changing the robust initialization; agreement would leave the verified artifact unchanged.
- **Alternatives Weighed**: `Repeat the stencil, reflection, or batch checks` (rejected: The stencil and coordinate transformations already agree to roundoff, and the observations have no batch or temporal labels.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from tensor_darcy_robin import _responses, _fit_sources
with open('/holdout_task/public_tensor_cases.pkl', 'rb') as f:
    cases = pickle.load(f)['cases']
failures = []
for k, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    i, j = p['sensor_ij'].T
    response = _responses(p)[i, j]
    clean = np.abs(p['y_obs'] - gt['u_true'][i, j] - gt['sensor_offset']) < .1
    sensors = response[clean]
    measured = p['y_obs'][clean] - sensors[:, 0]
    d = np.c_[sensors[:, 1:], np.ones(len(sensors))]
    alpha = p['tikhonov_alpha']
    ridge = np.sqrt(alpha) * np.eye(d.shape[1])[:-1]
    directions, _, _ = np.linalg.svd(d[:, :-1] - d[:, :-1].mean(axis=0), full_matrices=False)
    worst_gap = worst_weights = worst_offset = 0.
    for direction in directions.T:
        for fraction in (.10, .15):
            for ranking in (abs(direction), direction, -direction):
                idx = np.argsort(ranking)[-round(fraction * len(d)):]
                keep = np.ones(len(d), dtype=bool)
                keep[idx] = False
                oracle = np.linalg.lstsq(np.vstack((d[keep], ridge)), np.r_[measured[keep], np.zeros(len(ridge))], rcond=None)[0]
                for sign in (-1, 1):
                    y = measured.copy()
                    y[idx] += sign * np.sign(direction[idx]) * np.linspace(1.5, 4., len(idx))
                    fit = _fit_sources(d, y, alpha)
                    gap = np.max(abs(d @ (fit - oracle)))
                    worst_gap = max(worst_gap, gap)
                    worst_weights = max(worst_weights, np.linalg.norm(fit[:-1] - gt['source_weights']) / np.linalg.norm(gt['source_weights']))
                    worst_offset = max(worst_offset, abs(fit[-1] - gt['sensor_offset']))
                    if gap > .001:
                        failures.append((k, fraction, gap))
    print(f'case {k}: worst max sensor prediction gap to clean oracle={worst_gap:.6g}, worst relative weight error={worst_weights:.4%}, worst absolute offset error={worst_offset:.6g}')
assert not failures, (len(failures), max(failures, key=lambda item: item[2]))
PY
```
- **Container Exit Code**: `0` (`614 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case 0: worst max sensor prediction gap to clean oracle=1.35384e-15, worst relative weight error=4.0291%, worst absolute offset error=0.00313115
case 1: worst max sensor prediction gap to clean oracle=2.27289e-15, worst relative weight error=6.4271%, worst absolute offset error=0.0066001

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=60618da636b39c23 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_60618da6 (directory).
```

### Turn 10 — `submit_completed_task` (`12339 ms` LLM)
- **Rationale**: The saved artifact passed isolated verification. Spikes aligned with the public source-response modes changed sensor predictions by at most 2.3e-15 relative to the fit using only known-clean sensors. The audit found no remaining failure that warrants changing the solver.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/tensor_darcy_robin.py`. With known source weights, it reproduced both public reference fields within 4.8e-15. Across the public inverse tests, the worst relative source-weight error was 3.73%, the worst absolute sensor-offset error was 0.00388, and the largest pointwise field error was 0.00458.`



## Official Verifier Report
```json
{
  "benchmark": "Hegel-M89-HardStress-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/tensor_darcy_robin.py": "FILE:5044"
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
    "elapsed_sec": 0.2704
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"all_valid\": true, \"max_weight_rel_l2\": 0.042679, \"max_field_rel_l2\": 0.012911, \"max_offset_abs_err\": 0.003575, \"elapsed_sec\": 0.2704}\n__HEGEL_VERIFIER_AUTH_24d0dd927ed0a3428b7ebbeab44e97e7__:{\"strict_pass\": true, \"all_valid\": true, \"max_weight_rel_l2\": 0.042679, \"max_field_rel_l2\": 0.012911, \"max_offset_abs_err\": 0.003575, \"elapsed_sec\": 0.2704}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 4746.95,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
