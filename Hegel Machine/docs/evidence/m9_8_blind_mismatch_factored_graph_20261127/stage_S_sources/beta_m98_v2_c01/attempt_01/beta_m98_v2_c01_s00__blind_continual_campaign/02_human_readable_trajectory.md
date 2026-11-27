# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_v2_c01_s00__blind_continual_campaign`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M98-Blind-Mismatch-Factored-Lineage-World` |
| **Domain** | `2D Advection-Diffusion-Reaction Tomography & Blind Continual Campaign` |
| **Wave / Wormhole Family** | `wave_1` / `fam_elliptic_transport_and_factored_kkt` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `9 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `217,842` (`209,079` / `89,088` / `8,763`) |
| **Total Episode Wall Time (s)** | `275.54s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`20885 ms` LLM)
- **Rationale**: Inspect the actual input and archived evidence first: array layout, pulse availability, and calibration structure determine which forward model is justified and what can be validated.

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py; ls -la /tmp/hegel_continual_archive /tmp/hegel_wormhole_skills 2>/dev/null; find /tmp/hegel_continual_archive -maxdepth 2 -type f -printf '%p %s bytes\n' 2>/dev/null | head -50; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling' 2>&1 | head -30; python3 - <<'PY'
import pickle,numpy as np,scipy
from pprint import pprint
x=pickle.load(open('/holdout_task/public_batch.pkl','rb')); print('items',len(x),'scipy',scipy.__version__)
y=x[0]; print('item keys',list(y)); d=y.get('campaign_diagnostics'); print('diagnostics',None if d is None else (type(d),len(d))); print('pulse schema',None if not d else {k:np.shape(v) for k,v in d[0].items()})
for k,v in y['study'].items():
 print(k,type(v).__name__,getattr(v,'shape',None), str(v)[:550] if not isinstance(v,np.ndarray) or v.size<20 else (v.dtype,float(v.min()),float(v.max())))
PY
```
- **Container Exit Code**: `0` (`432 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
items 1 scipy 1.13.0
item keys ['study', 'campaign_diagnostics']
diagnostics None
pulse schema None
Lx float None 2.4
Ly float None 2.0
nx int None 14
ny int None 12
kappa_x ndarray (14,) [0.4503445  0.48384869 0.51540574 0.54318167 0.56556224 0.58124677
 0.58932374 0.58932374 0.58124677 0.56556224 0.54318167 0.51540574
 0.48384869 0.4503445 ]
kappa_y ndarray (12,) [0.57243569 0.56757485 0.55338612 0.53101898 0.50228549 0.46951347
 0.43535791 0.40258589 0.3738524  0.35148527 0.33729653 0.33243569]
vel_x ndarray (14,) [ 0.35        0.34500413  0.33015913  0.30588881  0.27288602  0.23209293
  0.18467409  0.13198319  0.07552447  0.01690968 -0.04218784 -0.10008099
 -0.15511704 -0.20572484]
vel_y ndarray (12,) [-0.         -0.05560533 -0.10899563 -0.1580441  -0.20079691 -0.23555099
 -0.26092193 -0.27589907 -0.27988581 -0.27272334 -0.25469696 -0.22652476]
robin_x float None 0.8942552148014796
robin_y float None 1.163864943166582
V_reaction_2d ndarray (14, 12) (dtype('float64'), 0.25, 0.42738476356834465)
mu_shifts ndarray (3,) [0.15 0.55 1.1 ]
source_modes ndarray (8, 14, 12) (dtype('float64'), 1.806275844759665e-17, 0.9998820677290571)
sensor_ij ndarray (32, 2) (dtype('int64'), 0.0, 12.0)
row_bank_id ndarray (96,) (dtype('int64'), 0.0, 3.0)
sensor_calibration_sweeps dict None {'ref_rosette': array([[-1.2       ,  0.        ,  0.        ,  0.        ],
       [-0.4       ,  0.        ,  0.        ,  0.        ],
       [ 0.5       ,  0.        ,  0.        ,  0.        ],
       [ 1.3       ,  0.        ,  0.        ,  0.        ],
       [ 0.        , -1.2       ,  0.        ,  0.        ],
       [ 0.        , -0.4       ,  0.        ,  0.        ],
       [ 0.        ,  0.5       ,  0.        ,  0.        ],
       [ 0.        ,  1.3       ,  0.        ,  0.        ],
       [ 0.        ,  0.        , -1.2       ,
Z_nuisance ndarray (96, 3) (dtype('float64'), -2.9706180053910964, 2.9557581659129353)
y_obs_raw ndarray (96,) (dtype('float64'), -1.1290216666342183, 0.9361792950507306)
aux_manifold_telemetry ndarray (24, 4) (dtype('float64'), -2.116743300512114, 2.6854338834200675)
nuisance_ridge float None 0.08
source_ridge float None 0.04
u_box float None 0.42
total_budget float None 1.4
```

### Turn 2 — `execute_in_remote_docker` (`21588 ms` LLM)
- **Rationale**: Check whether an affine map misses held-out sweep rows and whether the archive supplies direct wall evidence. These results determine the calibration basis and boundary operator before solving the inverse problem.

```bash
cd /holdout_task; cat validate_public_schema.py; find /tmp/hegel_continual_archive -maxdepth 2 -type f -printf '%p %s bytes\n' 2>/dev/null | head -35; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling' 2>/dev/null | grep -i -m5 -E 'nuisance|residualiz'; python3 - <<'PY'
import pickle,numpy as n
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; print('bank rows',s['row_bank_id'].reshape(-1,4)[:9]);print('paired sweep samples',n.c_[x[:18],y[:18]]);print('single banks',{k:(v.shape,v[:12]) for k,v in c['single_bank_sweeps'].items()}); print('telemetry sample',s['aux_manifold_telemetry'][:5]);
a=n.arange(len(x)); tr=a%5!=0; te=~tr
for name,X in [('affine',n.c_[n.ones(len(x)),x]),('separable cubic',n.c_[n.ones(len(x)),x,x*x,x**3]),('full cubic',n.c_[n.ones(len(x)),x,*[n.prod(x[:,z],axis=1) for k in (2,3) for z in __import__('itertools').combinations_with_replacement(range(4),k)]])]:
 b=n.linalg.lstsq(X[tr],y[tr],rcond=None)[0]; print(name,'test rmse',n.sqrt(n.mean((X[te]@b-y[te])**2,axis=0)), 'affine block',b[:5].round(5))
PY
```
- **Container Exit Code**: `0` (`453 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
3.99996073e-01 -6.23876534e-06]
 [ 0.00000000e+00  0.00000000e+00  5.00000000e-01  0.00000000e+00
  -7.34442167e-06 -1.30402419e-05  5.00006078e-01  5.44385264e-06]
 [ 0.00000000e+00  0.00000000e+00  1.30000000e+00  0.00000000e+00
   1.35081874e-05 -1.00519714e-05  1.30000053e+00  1.30010974e-05]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 -1.20000000e+00
   1.60753743e-06 -6.99531776e-06 -1.35219213e-05 -1.19999908e+00]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 -4.00000000e-01
  -2.18226662e-05  2.60092846e-06 -8.86009589e-06 -3.99982947e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  5.00000000e-01
   1.28805788e-05  1.82230657e-05  5.24983341e-06  5.00014855e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.30000000e+00
   9.82351492e-06 -1.53835310e-05  7.41554820e-06  1.30000469e+00]
 [ 1.02269613e+00  1.52335046e+00 -9.56880810e-01  9.50522052e-01
   1.02272204e+00  1.52335196e+00 -9.56884126e-01  9.50522227e-01]
 [ 5.55726829e-01 -9.64421583e-01  2.99392199e-01  3.67804591e-01
   5.55728224e-01 -9.64401684e-01  2.99409035e-01  3.67806468e-01]]
single banks {'bank_id': ((24,), array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])), 'ref_signal': ((24,), array([-1.1, -0.6, -0.1,  0.4,  0.9,  1.4, -1.1, -0.6, -0.1,  0.4,  0.9,
        1.4])), 'raw_readout': ((24,), array([-1.09998341, -0.59999198, -0.10000909,  0.39999383,  0.89998228,
        1.40000173, -1.10000205, -0.6000029 , -0.10000962,  0.40000362,
        0.90000551,  1.39998472]))}
telemetry sample [[-0.37329244  1.2347544   0.72283655  1.68785239]
 [-0.41341243  0.06968354  0.42365226  0.78958434]
 [ 2.68543388  0.09806825  0.60969868 -0.55683525]
 [ 0.84088206 -1.3628343  -1.88870726 -1.40369625]
 [ 0.86208988 -1.26315186 -1.04734072 -1.17971191]]
affine test rmse [8.97459229e-06 9.93773988e-06 7.57840566e-06 5.98987363e-06] affine block [[-0. -0. -0.  0.]
 [ 1.  0.  0.  0.]
 [-0.  1.  0.  0.]
 [ 0.  0.  1.  0.]
 [-0.  0.  0.  1.]]
separable cubic test rmse [8.60664092e-06 1.10217716e-05 7.47586910e-06 7.21078627e-06] affine block [[-1.e-05 -0.e+00  0.e+00  0.e+00]
 [ 1.e+00 -0.e+00  0.e+00 -0.e+00]
 [-0.e+00  1.e+00  0.e+00 -0.e+00]
 [-0.e+00  0.e+00  1.e+00 -1.e-05]
 [ 1.e-05  0.e+00 -0.e+00  1.e+00]]
full cubic test rmse [1.02727774e-05 1.49202896e-05 1.59065803e-05 8.65806947e-06] affine block [[-1.e-05 -0.e+00 -0.e+00  0.e+00]
 [ 1.e+00 -1.e-05  1.e-05 -1.e-05]
 [-1.e-05  1.e+00  1.e-05 -0.e+00]
 [ 0.e+00  0.e+00  1.e+00 -1.e-05]
 [ 1.e-05  1.e-05 -0.e+00  1.e+00]]
```

### Turn 3 — `execute_in_remote_docker` (`61039 ms` LLM)
- **Rationale**: Implement the stipulated baseline and compute the public certificate. A schema failure would identify an output contract issue; a nonzero public gap would direct debugging to projection, active-set constraints, or the dual formula.
- **Alternatives Weighed**: `Fit a nonlinear rosette calibration` (rejected: The cubic expansions worsen worst-channel held-out error on the supplied sweeps.); `Use the candidate-amplitude ridge profiling primitive` (rejected: The contract calls for explicit ridge residualization of both design and response, followed by a separate squared-residual box-simplex objective; this is not a profiled amplitude score.)

```bash
set -e
cd /holdout_task
if test -d /tmp/hegel_continual_archive; then ls -la /tmp/hegel_continual_archive; else echo 'No continual archive mounted'; fi
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes 2>&1 | head -24
cat > batch_solver.py <<'PY'
"""Steady transport tomography with calibrated observations and a box-simplex QP."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, robin):
    face = (kappa[:-1] + kappa[1:]) * (0.5 / h**2)
    left = face[:-1] + np.maximum(velocity[1:-1], 0) / h
    right = face[1:] + np.maximum(-velocity[1:-1], 0) / h
    edge = kappa[[0, -1]] / h**2
    return diags(
        (-np.r_[left, edge[1]], np.r_[edge[0] + robin / h, left + right,
                                       edge[1] + robin / h], -np.r_[edge[0], right]),
        (-1, 0, 1), format='csc',
    )


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1), study['robin_x'])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1), study['robin_y'])
    operator = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
                + diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    rhs = np.stack([mode.ravel(order='F') for mode in study['source_modes']], axis=1)
    ij = np.asarray(study['sensor_ij'])
    sensor_rows = ij[:, 0] + nx * ij[:, 1]
    identity = eye(nx * ny, format='csc')
    observations = []
    for mu in study['mu_shifts']:
        fields = splu(operator + mu * identity).solve(rhs)
        observations.append(fields[sensor_rows])
    terminal_modes = np.stack([field.reshape((nx, ny), order='F') for field in fields.T])
    return np.vstack(observations), terminal_modes


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    affine = np.linalg.lstsq(np.column_stack((np.ones(len(ref)), ref)), raw, rcond=None)[0]
    observed = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return np.linalg.solve(affine[1:].T, (observed - affine[0]).T).T.ravel()


def _box_simplex_qp(hessian, linear, cap, budget):
    """Feasible active set; -1 and +1 mark lower and upper bounds."""
    n = len(linear)
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=np.int8)
    if budget == 0:
        return np.zeros(n), float(linear.min())
    if budget == n * cap:
        w.fill(cap)
        return w, float((hessian @ w + linear).max())
    tol = 1e-11 * max(1.0, np.max(np.abs(linear)), cap * np.max(np.abs(hessian)))
    for _ in range(20 * n + 100):
        g = hessian @ w + linear
        free = np.flatnonzero(active == 0)
        lower = np.flatnonzero(active == -1)
        upper = np.flatnonzero(active == 1)
        if not free.size:
            lo = lower[np.argmin(g[lower])]
            hi = upper[np.argmax(g[upper])]
            if g[hi] <= g[lo] + tol:
                return w, float((g[hi] + g[lo]) / 2)
            active[[lo, hi]] = 0
            continue

        h = hessian[np.ix_(free, free)]
        system = np.ones((len(free) + 1, len(free) + 1))
        system[:-1, :-1] = h
        system[:-1, -1] = -1
        system[-1, -1] = 0
        direction_nu = np.linalg.solve(system, np.r_[-g[free], budget - w.sum()])
        direction, nu = direction_nu[:-1], float(direction_nu[-1])
        # Only truncate when the full equality-constrained step crosses a bound.
        limits = np.full(len(free), np.inf)
        down = (direction < 0) & (w[free] + direction < -1e-14 * cap)
        up = (direction > 0) & (w[free] + direction > cap + 1e-14 * cap)
        limits[down] = -w[free[down]] / direction[down]
        limits[up] = (cap - w[free[up]]) / direction[up]
        hit = int(np.argmin(limits))
        if limits[hit] < 1:
            w[free] += max(0., limits[hit]) * direction
            idx = free[hit]
            active[idx] = 1 if up[hit] else -1
            w[idx] = cap if up[hit] else 0.
            continue

        w[free] = np.clip(w[free] + direction, 0, cap)
        g = hessian @ w + linear
        violation = active * (g - nu)
        release = int(np.argmax(violation))
        if violation[release] <= tol:
            return w, nu
        active[release] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    design, terminal_modes = _forward(study)
    y = _calibrate(study)
    z = np.asarray(study['Z_nuisance'])
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    data = np.column_stack((design, y))
    projected = data - z @ np.linalg.solve(gram, z.T @ data)
    s, y = projected[:, :-1], projected[:, -1]
    ridge = study['source_ridge']
    cap, budget = study['u_box'], study['total_budget']
    w, nu = _box_simplex_qp(s.T @ s + ridge * np.eye(s.shape[1]), -s.T @ y, cap, budget)
    residual = s @ w - y
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = s.T @ residual + ridge * w
    lower = np.maximum(0, g - nu)
    upper = np.maximum(0, nu - g)
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
            'primal_objective': objective,
            'dual_lower_bound': float(objective - g @ w + nu * budget - cap * upper.sum()),
        },
    }
PY
python3 /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import importlib.util,pickle,numpy as np
p=importlib.util.spec_from_file_location('cand','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p);p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; o=m.solve_batch(s); print('weights',o['weights']); print('certificate',o['kkt_certificate']);print('bank layout',s['row_bank_id'].reshape(-1,4)[:6]); print('terminal range',o['terminal_field'].min(),o['terminal_field'].max())
PY
```
- **Container Exit Code**: `0` (`619 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
No continual archive mounted
{
  "ok": true,
  "target": "autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_prim_ridge_profiled_matching",
      "granularity_tier": "L1_compositional_primitive",
      "compiled_code_sha256": "dfdf7838f3618f600b4d7d7df948af526e9de4b41cbe0e429a9a76032e691078",
      "qualification_status": "pretraining_only_l1_primitive_sublation",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        "design (n, p) and values (n,) or (n, q) share leading axis n; ridge >= 0 finite scalar or (p,) vector.",
        "whitener (r, p) must match the SVD Gram whitener of the same weighted design matrix; candidate_energy >= 0."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "fit_ridge_nuisance": "APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)",
        "profile_candidate_amplitudes": "APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)"
      },
      "functions": [
        {
          "function": "profile_candidate_amplitudes",
{"schema_valid": true, "errors": []}
weights [0.03755042 0.42       0.18850144 0.268436   0.01874167 0.24296674
 0.22380374 0.        ]
certificate {'weights': [0.037550419294228414, 0.42, 0.18850143772270464, 0.2684360041653198, 0.018741665696540094, 0.24296673727302404, 0.22380373584818292, 0.0], 'nu_sum': 0.042477875194823815, 'lam_lower': [0.0, 0.0, 0.0, 3.469446951953614e-17, 0.0, 6.938893903907228e-17, 3.469446951953614e-17, 0.004402062061629179], 'lam_upper': [6.938893903907228e-18, 0.0524529053262944, 4.85722573273506e-17, 0.0, 6.938893903907228e-18, 0.0, 0.0, 0.0], 'primal_objective': 0.021031551087179946, 'dual_lower_bound': 0.021031551087179887}
bank layout [[0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]]
terminal range 0.0040982126065512645 0.09923009270115163

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=76ecc103d9e0890e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76ecc103 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`42567 ms` LLM)
- **Rationale**: A schema pass cannot detect swapped grid axes or inconsistent multipliers. Compare each shift and the terminal field against an independent dense stencil, compare the public objective against all feasible active faces, and stress the optimizer near both budget endpoints. Any failure identifies the implementation component to fix.
- **Alternatives Weighed**: `Infer alternate walls from the blind observations` (rejected: The campaign has neither tracer pulses nor an archive; the task specifies the Robin baseline for this case.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import itertools
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open(root+'public_batch.pkl','rb'))[0]['study']
out = m.solve_batch(s)
nx, ny = s['nx'], s['ny']

def dense_axis(suffix, n):
    h = s['L'+suffix]/(n-1)
    k, v = s['kappa_'+suffix], s['vel_'+suffix]
    a = np.zeros((n,n))
    for i in range(n):
        if i in (0,n-1):
            a[i,i] = k[i]/h**2 + s['robin_'+suffix]/h
            a[i,1 if i == 0 else n-2] = -k[i]/h**2
        else:
            a[i,i-1] = -(k[i]+k[i-1])/(2*h**2)-max(v[i],0)/h
            a[i,i+1] = -(k[i]+k[i+1])/(2*h**2)-max(-v[i],0)/h
            a[i,i] = -a[i,i-1]-a[i,i+1]
    return a

ax, ay = dense_axis('x',nx), dense_axis('y',ny)
a = np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag(s['V_reaction_2d'].flatten('F'))
i,j = s['sensor_ij'].T
rhs = s['source_modes'].transpose(0,2,1).reshape(len(out['weights']),-1).T
blocks = []
for mu in s['mu_shifts']:
    u = np.linalg.solve(a+mu*np.eye(nx*ny),rhs)
    blocks.append(u[i+nx*j])
reference = np.vstack(blocks)
observed_design, _ = m._forward(s)
shift_errors = [np.max(np.abs(obs-ref)) for obs,ref in zip(np.split(observed_design,len(blocks)),blocks)]
w = out['weights']; cert = out['kkt_certificate']; nu = cert['nu_sum']
field_error = np.max(np.abs((u@w).reshape(ny,nx).T-out['terminal_field']))
assert max(shift_errors) < 1e-12 and field_error < 1e-12
z = s['Z_nuisance']; p = np.eye(len(z))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
d, y = p@reference, p@m._calibrate(s)
ridge, cap, budget = s['source_ridge'], s['u_box'], s['total_budget']
h = d.T@d+ridge*np.eye(len(w)); c = -d.T@y
best = np.inf
for face in itertools.product((-1,0,1),repeat=len(w)):
    face = np.array(face)
    free = np.where(face==0)[0]; upper = np.where(face==1)[0]
    b = budget-cap*len(upper)
    if b < -1e-12 or b > cap*len(free)+1e-12: continue
    q = np.zeros_like(w); q[upper] = cap
    if len(free):
        hf = h[np.ix_(free,free)]
        q0 = np.linalg.solve(hf,-(h@q+c)[free]); dq = np.linalg.solve(hf,np.ones(len(free)))
        q[free] = q0+dq*((b-q0.sum())/dq.sum())
        if np.min(q) < -1e-12 or np.max(q) > cap+1e-12: continue
    best = min(best,.5*(np.linalg.norm(d@q-y)**2+ridge*(q@q)))
ll, lu = np.array(cert['lam_lower']),np.array(cert['lam_upper'])
g = h@w+c
stationarity = np.linalg.norm(g-nu-ll+lu,np.inf)
assert stationarity < 1e-12 and max(abs(w@ll),abs((cap-w)@lu)) < 1e-12
assert abs(best-cert['primal_objective']) < 1e-12
worst_gap = 0.
rng = np.random.default_rng(23)
for n in (1,2,8,16):
    for rep in range(25):
        x = rng.normal(size=(max(n//2,1),n)); h = x.T@x+ridge*np.eye(n); c = rng.normal(size=n)
        for budget in (0.,1e-8,cap,n*cap/2,n*cap-1e-8,n*cap):
            q,nu = m._box_simplex_qp(h,c,cap,budget)
            g = h@q+c
            gap = g@q - nu*budget - cap*np.minimum(0,g-nu).sum()
            worst_gap = max(worst_gap,abs(gap))
            assert abs(q.sum()-budget) < 1e-12 and np.all((q>=0)&(q<=cap)) and abs(gap)<1e-9
ref = rng.normal(size=(80,4)); x = rng.normal(size=(24,4))
mix = np.eye(4)+rng.normal(scale=.16,size=(4,4)); offset = rng.normal(scale=.2,size=4)
modified = dict(s, sensor_calibration_sweeps={'ref_rosette':ref,'raw_rosette':ref@mix+offset},y_obs_raw=(x@mix+offset).ravel())
cal_error = np.max(np.abs(m._calibrate(modified)-x.ravel()))
assert cal_error < 1e-12
print('shift_errors',shift_errors,'terminal_error',field_error,'exhaustive_objective_error',abs(best-cert['primal_objective']),'stationarity',stationarity,'worst_stress_gap',worst_gap,'mixed_calibration_error',cal_error)
PY
python3 /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`775 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
shift_errors [2.498001805406602e-16, 2.498001805406602e-16, 1.5265566588595902e-16] terminal_error 7.632783294297951e-17 exhaustive_objective_error 4.163336342344337e-17 stationarity 3.0531133177191805e-16 worst_stress_gap 1.4210854715202004e-14 mixed_calibration_error 1.7763568394002505e-15
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=76ecc103d9e0890e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76ecc103 (directory).
```

### Turn 5 — `submit_completed_task` (`13315 ms` LLM)
- **Rationale**: The final artifact passed the public schema check and isolated-process verification. An independent dense stencil matched every shift, exhaustive active-face enumeration matched the public objective, and stress tests covered both budget endpoints.
- **Alternatives Weighed**: `Fit an alternate boundary operator` (rejected: Neither tracer pulses nor a continual archive were available, so the specified Robin fallback applies.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`13` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5323'}): Do NOT lock prematurely with `13` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py`. The public schema check passed. Independent verification matched the terminal field within 7.7e-17, and the worst duality gap across optimizer stress tests was 1.42e-14.`

### Turn 6 — `execute_in_remote_docker` (`35770 ms` LLM)
- **Rationale**: Pairwise, triple-product, and rational terms could expose a calibration error missed by the prior cubic comparison; material improvement on held-out sweep slices would justify revisiting calibration. Holding whole rosettes out tests whether the prescribed nuisance adjustment predicts unseen measurements rather than merely improving the fit.
- **Alternatives Weighed**: `Run batch-classification, relay-mode, or stochastic-drift tests` (rejected: The supplied problem is steady continuous tomography with no class labels, relay states, or time-evolution operator; held-out rosette prediction by shift is the applicable nuisance check.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, itertools, pickle
import numpy as np
p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', p + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(p + 'public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
c = s['sensor_calibration_sweeps']
x, raw = c['ref_rosette'], c['raw_rosette']
# Compare forward calibration models on held-out axis-only and mixed sweeps.
base = np.column_stack((np.ones(len(x)), x))
pairs = np.column_stack([x[:, i] + sign*x[:, j] for i,j in itertools.combinations(range(4),2) for sign in (-1,1)])
triple = np.column_stack([np.prod(x[:, inds], axis=1) for inds in itertools.combinations(range(4),3)])
quotient = np.column_stack([x[:, i]/(1+a*x[:, j]**2)-x[:, i] for i in range(4) for j in range(4) for a in (.25, 1.)])
axis = np.count_nonzero(x,axis=1) == 1
fold = np.zeros(len(x),dtype=int)
for group in (axis,~axis):
    fold[group] = np.arange(group.sum())%5
for name, mat in [('affine',base),('pair/triple',np.column_stack((base,pairs**2,pairs**3,triple))),('rational',np.column_stack((base,quotient)))]:
    errors = np.empty_like(raw)
    for i in range(5):
        tr = fold != i
        errors[~tr] = mat[~tr]@np.linalg.lstsq(mat[tr],raw[tr],rcond=None)[0]-raw[~tr]
    print(name, 'worst-channel RMSE axis/mixed',*[float(np.sqrt(np.mean(errors[group]**2,axis=0)).max()) for group in (axis,~axis) if group.any()])
# Fit sources and nuisance on training observations only, then score held-out rosettes by shift.
a,_ = m._forward(s)
y = m._calibrate(s)
z = s['Z_nuisance']; rid = s['source_ridge']; nr = s['nuisance_ridge']
errs = np.empty((2,len(y)))
for i in range(len(y)//4):
    te = np.arange(len(y))//4 == i
    tr = ~te
    for corrected in (0,1):
        train = np.column_stack((a[tr],y[tr])); zz = z[tr]
        if corrected:
            gram = zz.T@zz+nr*np.eye(z.shape[1])
            train = train-zz@np.linalg.solve(gram,zz.T@train)
        b,v = train[:,:-1],train[:,-1]
        w,_ = m._box_simplex_qp(b.T@b+rid*np.eye(b.shape[1]),-b.T@v,s['u_box'],s['total_budget'])
        guess = a[te]@w
        if corrected:
            guess = guess+z[te]@np.linalg.solve(gram,zz.T@(y[tr]-a[tr]@w))
        errs[corrected,te]=guess-y[te]
for label, error in zip(('unadjusted','ridge nuisance'),errs):
    rmse = [float(np.sqrt(np.mean(e**2))) for e in np.split(error,len(s['mu_shifts']))]
    print(label,'held-out shift RMSE',rmse,'worst',max(rmse))
o = m.solve_batch(s)
k = o['kkt_certificate']; gap = k['primal_objective']-k['dual_lower_bound']
assert abs(gap) < 2e-4
print('isolated candidate gap',gap, 'budget error',abs(o['weights'].sum()-s['total_budget']))
PY
```
- **Container Exit Code**: `0` (`448 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
affine worst-channel RMSE axis/mixed 1.3906305184770435e-05 9.83765344369537e-06
pair/triple worst-channel RMSE axis/mixed 1.397462844603034e-05 2.21967217122913e-05
rational worst-channel RMSE axis/mixed 1.337007038741074e-05 2.3230122281643198e-05
unadjusted held-out shift RMSE [0.3292720664397186, 0.4079911305955631, 0.3136382168502419] worst 0.4079911305955631
ridge nuisance held-out shift RMSE [0.019934529199879297, 0.016811157589140147, 0.015776554507728007] worst 0.019934529199879297
isolated candidate gap 5.898059818321144e-17 budget error 0.0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=76ecc103d9e0890e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76ecc103 (directory).
```

### Turn 7 — `submit_completed_task` (`16669 ms` LLM)
- **Rationale**: The exact candidate passed isolated-process verification. An independent dense stencil matched the terminal field within 7.7e-17, and exhaustive active-face enumeration confirmed the public objective. The follow-up audit supported the existing calibration and nuisance correction without requiring an artifact change.
- **Alternatives Weighed**: `Use nonlinear rosette calibration` (rejected: Pairwise, triple-product, and rational terms more than doubled the worst-channel error on held-out mixed sweeps.); `Infer a different wall operator` (rejected: No tracer pulses or continual archive were available, so the specified Robin fallback applies.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5323'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema check passed, and the worst duality gap across optimizer stress tests was 1.42e-14.`

### Turn 8 — `execute_in_remote_docker` (`28014 ms` LLM)
- **Rationale**: The prior rosette split could leave the same physical location in training at another shift. Holding out whole locations and shifts can expose dependence on that overlap; if the gain collapses, report it as a transfer limitation rather than changing the prescribed objective.
- **Alternatives Weighed**: `Repeat the nonlinear calibration and numerical certificate audits` (rejected: The unchanged candidate already passed the certificate checks, and the held-out sweep comparison showed no calibration upgrade.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
a, _ = m._forward(s)
y, z = m._calibrate(s), s['Z_nuisance']
ns, nk = len(s['sensor_ij']), len(s['mu_shifts'])
# Hold all shifts at a physical sensor out together, then hold out entire shifts.
physical = np.asarray(s['sensor_ij'])
_, sites = np.unique(physical, axis=0, return_inverse=True)
groups = {'physical sensor': np.tile(sites, nk), 'shift': np.repeat(np.arange(nk), ns)}
for label, group in groups.items():
    metrics = np.zeros((2, len(np.unique(group))))
    for j, held in enumerate(np.unique(group)):
        test = group == held
        train = ~test
        zt = z[train]
        data = np.column_stack((a[train], y[train]))
        for adjust in (False, True):
            if adjust:
                gram = zt.T @ zt + s['nuisance_ridge'] * np.eye(z.shape[1])
                projected = data - zt @ np.linalg.solve(gram, zt.T @ data)
            else:
                projected = data
            d, r = projected[:, :-1], projected[:, -1]
            w, _ = m._box_simplex_qp(d.T @ d + s['source_ridge'] * np.eye(d.shape[1]), -d.T @ r, s['u_box'], s['total_budget'])
            pred = a[test] @ w
            if adjust:
                pred += z[test] @ np.linalg.solve(gram, zt.T @ (y[train] - a[train] @ w))
            metrics[int(adjust), j] = np.sqrt(np.mean((pred - y[test])**2))
    print(label, 'folds:', metrics.shape[1], 'worst RMSE raw/corrected:', metrics.max(axis=1).tolist(), 'folds improved:', int(np.sum(metrics[1] < metrics[0])))
    assert np.isfinite(metrics).all()
PY
```
- **Container Exit Code**: `0` (`484 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
physical sensor folds: 32 worst RMSE raw/corrected: [0.7160725520380679, 0.03960509312314318] folds improved: 32
shift folds: 3 worst RMSE raw/corrected: [0.41592487925387034, 0.02154963675582297] folds improved: 3

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=76ecc103d9e0890e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_76ecc103 (directory).
```

### Turn 9 — `submit_completed_task` (`14616 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process execution, independent dense-stencil and objective checks, and optimizer stress tests. The final grouped holdouts confirmed that nuisance correction improves prediction without relying on overlap at the same sensor or shift.
- **Alternatives Weighed**: `Replace affine calibration with nonlinear terms` (rejected: The tested nonlinear terms more than doubled the worst-channel RMSE on held-out mixed sweeps.); `Fit an alternate wall operator` (rejected: Neither diagnostic pulses nor a continual archive were available, so the specified Robin fallback applies.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed, and the worst optimizer stress-test duality gap was 1.42e-14. Ridge nuisance correction improved prediction in all 32 held-out sensor folds and all 3 held-out shift folds.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M98-Blind-Mismatch-Factored-Lineage-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
