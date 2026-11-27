# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_beta_m98_00_s00__blind_continual_campaign`

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
| **Turns Used / Max Turns** | `10 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `248,589` (`238,237` / `72,320` / `10,352`) |
| **Total Episode Wall Time (s)** | `323.08s` |

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

### Turn 1 — `execute_in_remote_docker` (`19599 ms` LLM)
- **Rationale**: Inspect the public schema, array dimensions, diagnostics, and archive inventory to determine which models can be tested and what the validator actually checks.
- **Alternatives Weighed**: `Implement the baseline without data inspection` (rejected: Could miss a reachable calibration or wall-operator variant and misread the output contract.)

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py && if [ -d /tmp/hegel_continual_archive ]; then du -ah /tmp/hegel_continual_archive | sort -h | tail -35; fi; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling' 2>&1; cd /holdout_task && python3 - <<'PY'
import pickle
import numpy as np
with open('public_batch.pkl','rb') as f: batch=pickle.load(f)
print('batch',type(batch),len(batch))
def describe(d,depth=0):
 for k,v in d.items():
  prefix='  '*depth
  if isinstance(v,dict):
   print(prefix,k); describe(v,depth+1)
  elif isinstance(v,np.ndarray):
   print(prefix,k,v.shape,v.dtype,repr(v.ravel()[:8]))
  elif isinstance(v,(tuple,list)):
   print(prefix,k,type(v).__name__,len(v));
   if v and isinstance(v[0],dict): describe(v[0],depth+1)
   elif v: print(prefix,repr(v[0])[:180])
  else: print(prefix,k,repr(v)[:200])
describe(batch[0])
PY
```
- **Container Exit Code**: `0` (`401 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
parse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
batch <class 'list'> 1
 study
   Lx 2.4
   Ly 2.0
   nx 14
   ny 12
   kappa_x (14,) float64 array([0.42130881, 0.454813  , 0.48637006, 0.51414598, 0.53652655,
       0.55221109, 0.56028805, 0.56028805])
   kappa_y (12,) float64 array([0.51535314, 0.5104923 , 0.49630357, 0.47393643, 0.44520295,
       0.41243092, 0.37827536, 0.34550334])
   vel_x (14,) float64 array([0.35      , 0.34500413, 0.33015913, 0.30588881, 0.27288602,
       0.23209293, 0.18467409, 0.13198319])
   vel_y (12,) float64 array([-0.        , -0.05560533, -0.10899563, -0.1580441 , -0.20079691,
       -0.23555099, -0.26092193, -0.27589907])
   robin_x 0.9023444699977767
   robin_y 1.1567834842388094
   V_reaction_2d (14, 12) float64 array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
   mu_shifts (3,) float64 array([0.15, 0.55, 1.1 ])
   source_modes (8, 14, 12) float64 array([0.00097197, 0.00719273, 0.03367342, 0.09973152, 0.1868656 ,
       0.2215022 , 0.1661034 , 0.07880082])
   sensor_ij (32, 2) int64 array([12,  2,  2,  2, 10,  6,  0,  6])
   row_bank_id (96,) int64 array([0, 1, 2, 3, 0, 1, 2, 3])
   sensor_calibration_sweeps
     ref_rosette (80, 4) float64 array([-1.2,  0. ,  0. ,  0. , -0.4,  0. ,  0. ,  0. ])
     raw_rosette (80, 4) float64 array([-1.20001324e+00, -1.11025034e-05,  9.53624036e-06,  8.64706879e-06,
       -4.00001640e-01, -5.74897726e-06, -1.12474641e-06, -7.17806914e-06])
     single_bank_sweeps
       bank_id (24,) int64 array([0, 0, 0, 0, 0, 0, 1, 1])
       ref_signal (24,) float64 array([-1.1, -0.6, -0.1,  0.4,  0.9,  1.4, -1.1, -0.6])
       raw_readout (24,) float64 array([-1.10001991, -0.60000017, -0.09999244,  0.4000215 ,  0.89999102,
        1.39999161, -1.10000557, -0.5999997 ])
   Z_nuisance (96, 3) float64 array([-0.39100862, -0.8425231 , -0.74584925,  0.47635506, -0.94900402,
       -0.20782659, -0.15235741, -0.09525125])
   y_obs_raw (96,) float64 array([ 0.11649357, -0.11772128,  0.24816319, -0.22450744,  0.24200413,
        0.03886201, -0.07405387,  0.31999099])
   aux_manifold_telemetry (24, 4) float64 array([-0.67131433, -0.81728537, -1.31265215,  2.13519789, -1.26179512,
       -0.48693358,  0.62732041, -0.61742346])
   nuisance_ridge 0.08
   source_ridge 0.04
   u_box 0.42
   total_budget 1.4
 campaign_diagnostics None
```

### Turn 2 — `execute_in_remote_docker` (`54434 ms` LLM)
- **Rationale**: The public case is blind, so use the specified wall baseline unless archive evidence supports a different operator. Sweep cross-validation can reveal a needed nonlinear calibration term, while the public validator detects artifact and output-contract mismatches.
- **Alternatives Weighed**: `Assume an identity rosette response` (rejected: The interface requires dynamic calibration even when the public sweep is nearly identity.)

```bash
cd /holdout_task && cat validate_public_schema.py && if [ -d /tmp/hegel_continual_archive ]; then find /tmp/hegel_continual_archive -maxdepth 3 -type f -printf '%s %p\n' | head -35; else echo 'No continual archive mounted'; fi; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling projection' 2>&1 | head -14; python3 - <<'PY'
import pickle,numpy as np
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps']; r=x['ref_rosette']; y=x['raw_rosette']; print('Sweep residual against identity:',np.sqrt(np.mean((r-y)**2,axis=0))); print('sweep rows',r[:22]);
for name,d in [('linear',np.c_[np.ones(len(r)),r]),('quadratic',np.column_stack([np.ones(len(r)),r]+[r[:,i]*r[:,j] for i in range(4) for j in range(i,4)])),('cubic additive',np.c_[np.ones(len(r)),r,r*r,r*r*r])]:
 p=np.empty_like(y)
 for idx in np.array_split(np.random.default_rng(21).permutation(len(r)),8):
  mask=np.ones(len(r),bool);mask[idx]=False;p[idx]=d[idx]@np.linalg.lstsq(d[mask],y[mask],rcond=None)[0]
 print(name,'OOF RMS / worst channel RMS',np.sqrt(np.mean((p-y)**2)),np.sqrt(np.mean((p-y)**2,axis=0)).max())
PY
cat > /holdout_task/batch_solver.py <<'PY'
"""Transport tomography with empirical rosette calibration and a boxed source fit."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _axis(k, v, h, robin):
    n = len(k)
    face = (k[1:] + k[:-1]) / (2 * h**2)
    left = np.r_[-face[:-1] - np.maximum(v[1:-1], 0) / h, -k[-1] / h**2]
    right = np.r_[-k[0] / h**2, -face[1:] - np.maximum(-v[1:-1], 0) / h]
    center = np.r_[k[0] / h**2 + robin / h,
                   -left[:-1] - right[1:],
                   k[-1] / h**2 + robin / h]
    return sparse.diags((left, center, right), (-1, 0, 1), shape=(n, n), format='csc')


def _design(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1), study['robin_x'])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1), study['robin_y'])
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    eye = sparse.eye(nx * ny, format='csc')
    rhs = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    ij = study['sensor_ij']
    rows = ij[:, 0] + nx * ij[:, 1]
    blocks = []
    for mu in study['mu_shifts']:
        fields = splu(operator + mu * eye).solve(rhs)
        blocks.append(fields[rows])
    return np.vstack(blocks), fields.reshape(nx, ny, -1, order='F').transpose(2, 0, 1)


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
    fit = np.linalg.lstsq(np.column_stack((np.ones(len(ref)), ref)), raw, rcond=None)[0]
    obs = study['y_obs_raw'].reshape(-1, 4)
    return np.linalg.solve(fit[1:].T, (obs - fit[0]).T).T.ravel()


def _box_qp(hessian, linear, cap, budget):
    """Primal active-set method; every blocking step preserves the sum."""
    n = len(linear)
    w = np.full(n, budget / n)
    if budget == 0 or budget == n * cap:
        g = hessian @ w + linear
        return w, float(g.min() if budget == 0 else g.max())
    status = np.zeros(n, dtype=int)  # -1 lower, +1 upper, 0 free
    for _ in range(20 * n**2):
        free = np.flatnonzero(status == 0)
        fixed = np.flatnonzero(status)
        if len(free):
            nf = len(free)
            kkt = np.zeros((nf+1, nf+1))
            kkt[:nf, :nf] = hessian[np.ix_(free, free)]
            kkt[:nf, nf] = -1
            kkt[nf, :nf] = 1
            rhs = np.r_[-linear[free] - hessian[np.ix_(free, fixed)] @ w[fixed],
                        budget - w[fixed].sum()]
            answer = np.linalg.solve(kkt, rhs)
            target, nu = answer[:-1], float(answer[-1])
            violation = np.flatnonzero((target < -1e-13) | (target > cap+1e-13))
            if len(violation):
                delta = target - w[free]
                bounds = np.where(target[violation] < 0, 0., cap)
                alpha = (bounds - w[free[violation]]) / delta[violation]
                i = int(np.argmin(alpha))
                w[free] += max(0., alpha[i]) * delta
                j = free[violation[i]]
                w[j] = bounds[i]
                status[j] = -1 if bounds[i] == 0 else 1
                continue
            w[free] = np.clip(target, 0., cap)
        else:
            g = hessian @ w + linear
            nu = float(g[status == 1].max())
        g = hessian @ w + linear
        violation = status * (g - nu)
        j = int(np.argmax(violation))
        if violation[j] <= 1e-12:
            return w, nu
        status[j] = 0
    raise RuntimeError('Boxed source fit did not converge')


def solve_batch(study: dict) -> dict:
    design, terminal_modes = _design(study)
    y = _calibrate(study)
    z = study['Z_nuisance']
    projection = np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    design -= z @ (projection @ design)
    y -= z @ (projection @ y)
    cap, budget, ridge = study['u_box'], study['total_budget'], study['source_ridge']
    w, nu = _box_qp(design.T @ design + ridge * np.eye(design.shape[1]), -design.T @ y, cap, budget)
    residual = design @ w - y
    g = design.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    lower = float(objective - g @ w + nu * budget + cap * np.minimum(0., g - nu).sum())
    return {'weights': w, 'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0., g - nu).tolist(),
                                'lam_upper': np.maximum(0., nu - g).tolist(),
                                'primal_objective': objective, 'dual_lower_bound': lower}}
PY
python3 /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`527 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
Sweep residual against identity: [1.01002221e-05 8.32491505e-06 1.04006360e-05 1.05204914e-05]
sweep rows [[-1.2         0.          0.          0.        ]
 [-0.4         0.          0.          0.        ]
 [ 0.5         0.          0.          0.        ]
 [ 1.3         0.          0.          0.        ]
 [ 0.         -1.2         0.          0.        ]
 [ 0.         -0.4         0.          0.        ]
 [ 0.          0.5         0.          0.        ]
 [ 0.          1.3         0.          0.        ]
 [ 0.          0.         -1.2         0.        ]
 [ 0.          0.         -0.4         0.        ]
 [ 0.          0.          0.5         0.        ]
 [ 0.          0.          1.3         0.        ]
 [ 0.          0.          0.         -1.2       ]
 [ 0.          0.          0.         -0.4       ]
 [ 0.          0.          0.          0.5       ]
 [ 0.          0.          0.          1.3       ]
 [-0.1244219  -0.61407197  0.28610995 -0.26028628]
 [ 0.63891888 -0.67630525 -1.37443547  0.21363624]
 [ 0.48510345 -0.73849227  0.89562373  0.49076324]
 [ 1.05193766 -0.22368983  0.23534991 -0.89495693]
 [ 0.92013983 -0.8716568   0.42081611 -0.6421797 ]
 [-0.15661424 -0.37340276  1.79176661  1.00422422]]
linear OOF RMS / worst channel RMS 1.0432040913491802e-05 1.115677373191871e-05
quadratic OOF RMS / worst channel RMS 1.1327515891653558e-05 1.2479646078793194e-05
cubic additive OOF RMS / worst channel RMS 1.1435650582363054e-05 1.2144968904322699e-05
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c103659f9dec72de (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c103659f (directory).
```

### Turn 3 — `execute_in_remote_docker` (`45631 ms` LLM)
- **Rationale**: The schema passes but cannot detect a transposed field, incorrect wall coefficients, or an internally inconsistent certificate. A dense reference assembly and targeted stress cases check those failure modes in an isolated process before submission.
- **Alternatives Weighed**: `Promote nonlinear calibration` (rejected: Both tested nonlinear models worsened held-out sweep prediction.); `Infer an alternate wall operator` (rejected: There are no public diagnostic pulses or mounted archive; the instructions specify the Robin baseline in this case.)

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes 2>&1 | head -18
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle, numpy as np
from scipy.optimize import minimize
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
nx,ny=s['nx'],s['ny']; dims=(nx,ny); N=nx*ny
A=np.diag(s['V_reaction_2d'].flatten(order='F'))
for y in range(ny):
 for x in range(nx):
  row=x+nx*y
  for a,i,stride,n in [('x',x,1,nx),('y',y,nx,ny)]:
   h=s['L'+a]/(n-1); k=s['kappa_'+a]; v=s['vel_'+a][i]
   if i==0 or i==n-1:
    A[row,row+stride*(1 if i==0 else -1)] -= k[i]/h**2
    A[row,row] += k[i]/h**2+s['robin_'+a]/h
   else:
    for sign in [-1,1]:
     q=(k[i]+k[i+sign])/(2*h*h)+max(-sign*v,0)/h
     A[row,row+stride*sign] -= q
     A[row,row] += q
rhs=np.column_stack([f.flatten(order='F') for f in s['source_modes']]); ix,iy=s['sensor_ij'].T
F=[np.linalg.solve(A+mu*np.eye(N),rhs) for mu in s['mu_shifts']]
S=np.concatenate([f[ix+nx*iy] for f in F]); Sd,Fd=m._design(s)
errors=[np.max(abs(F[-1][:,i]-f.flatten(order='F'))) for i,f in enumerate(Fd)]
blocks=np.split(abs(S-Sd),len(F)); print('design max by shift:',[b.max() for b in blocks],'; terminal modes max:',max(errors)); assert max(b.max() for b in blocks)<1e-12 and max(errors)<1e-12
out=m.solve_batch(s); w=out['weights']; c=out['kkt_certificate']; z=s['Z_nuisance']; yr=m._calibrate(s); P=np.eye(len(yr))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T); X=P@S; y=P@yr; ridge=s['source_ridge']; cap=s['u_box']; budget=s['total_budget']
H=X.T@X+ridge*np.eye(len(w)); b=-X.T@y; g=H@w+b; r=X@w-y; J=(r@r+ridge*(w@w))/2
ll,lu=np.array(c['lam_lower']),np.array(c['lam_upper']); terminal=(F[-1]@w).reshape(dims,order='F')
assert np.max(abs(terminal-out['terminal_field']))<1e-12 and abs(J-c['primal_objective'])<1e-12 and np.array_equal(w,c['weights'])
assert np.max(abs(g-c['nu_sum']-ll+lu))<1e-12 and max(np.max(abs(ll*w)),np.max(abs(lu*(cap-w))))<1e-12 and abs(J-c['dual_lower_bound'])<1e-12
print('public projected observation RMSE by shift:',[float(np.sqrt(np.mean(a*a))) for a in np.split(r,len(F))],'; KKT gap:',J-c['dual_lower_bound'])
aff=np.array([[1.12,.04,0.,-.03],[.06,.93,-.04,0.],[0.,.08,1.31,.05],[-.02,0.,.07,.77]]); off=np.array([.07,-.11,.02,.13]); synthetic=dict(s); sweeps=dict(s['sensor_calibration_sweeps']); ref=sweeps['ref_rosette']; sweeps['raw_rosette']=ref@aff+off; synthetic['sensor_calibration_sweeps']=sweeps; synthetic['y_obs_raw']=(yr.reshape(-1,4)@aff+off).ravel(); recovered=m.solve_batch(synthetic); cal=float(np.max(abs(yr-m._calibrate(synthetic)))); print('affine calibration synthetic inverse max error:',cal); assert cal<1e-12 and np.max(abs(w-recovered['weights']))<1e-12
rng=np.random.default_rng(11); worst=[0.,0.,0.]
for n in [2,3,8,24]:
 for rep in range(80):
  mat=rng.normal(size=(n,n)); H=mat.T@mat/n+0.04*np.eye(n); b=rng.normal(size=n)*10**rng.uniform(-2,2); cap=.42; budget=cap*([0,n,1,n-1][rep%4] if rep<20 else rng.uniform(0,n)); u,nu=m._box_qp(H,b,cap,budget); g=H@u+b
  feas=max(0.,-u.min(),u.max()-cap,abs(u.sum()-budget)); gap=g@u-nu*budget-cap*np.minimum(0,g-nu).sum(); comp=max(np.max(abs(u*np.maximum(0,g-nu))),np.max(abs((cap-u)*np.maximum(0,nu-g)))); worst=list(map(max,zip(worst,[feas,abs(gap),comp])))
  if rep in (20,40,60):
   res=minimize(lambda x: .5*x@H@x+b@x,np.full(n,budget/n),jac=lambda x:H@x+b,method='SLSQP',bounds=[(0,cap)]*n,constraints=[{'type':'eq','fun':lambda x:x.sum()-budget,'jac':lambda x:np.ones(n)}],options={'ftol':1e-11}); assert abs((.5*u@H@u+b@u)-res.fun)<1e-7
assert max(worst)<1e-10; print('320 QPs: worst feasibility, dual gap, complementarity:',worst)
PY
```
- **Container Exit Code**: `1` (`779 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
{"schema_valid": true, "errors": []}
design max by shift: [3.0531133177191805e-16, 1.942890293094024e-16, 2.220446049250313e-16] ; terminal modes max: 2.220446049250313e-16
public projected observation RMSE by shift: [0.006984666927143259, 0.007435894651735843, 0.0089262736361424] ; KKT gap: 9.194034422677078e-17
affine calibration synthetic inverse max error: 1.0547118733938987e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c103659f9dec72de (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c103659f (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 39, in <module>
AssertionError
```

### Turn 4 — `execute_in_remote_docker` (`34303 ms` LLM)
- **Rationale**: The previous run verified the forward operator, orientation, and public certificate. Reproduce the failed stress comparison with diagnostics, and use sorted-gradient lower bounds plus exhaustive face enumeration to decide whether the solver needs a fix.
- **Alternatives Weighed**: `Tune the solver to agree with SLSQP` (rejected: The failing assertion did not report whether the comparison solution was feasible or converged; an independent optimality bound can distinguish solver error from reference error.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
from itertools import product
import numpy as np
from scipy.optimize import minimize
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def exhaustive(H, b, cap, budget):
    best = np.inf
    for status in product((-1, 0, 1), repeat=len(b)):
        status = np.array(status)
        free, fixed = np.flatnonzero(status == 0), np.flatnonzero(status != 0)
        x = np.where(status == 1, cap, 0.)
        left = budget - x.sum()
        n = len(free)
        if left < -1e-13 or left > n * cap + 1e-13:
            continue
        if n:
            hf = H[np.ix_(free, free)]
            unconstrained = np.linalg.solve(hf, -b[free] - H[np.ix_(free, fixed)] @ x[fixed])
            direction = np.linalg.solve(hf, np.ones(n))
            x[free] = unconstrained + direction * ((left - unconstrained.sum()) / direction.sum())
            if x.min() < -1e-12 or x.max() > cap + 1e-12:
                continue
        best = min(best, .5 * x @ H @ x + b @ x)
    return best

rng = np.random.default_rng(11)
worst = np.zeros(4)
reference_discrepancies = []
for n in (2, 3, 8, 24):
    for rep in range(80):
        mat = rng.normal(size=(n, n))
        H = mat.T @ mat / n + .04 * np.eye(n)
        b = rng.normal(size=n) * 10**rng.uniform(-2, 2)
        cap = .42
        budget = cap * ([0, n, 1, n-1][rep % 4] if rep < 20 else rng.uniform(0, n))
        w, nu = m._box_qp(H, b, cap, budget)
        g = H @ w + b
        # Independent exact minimizer of the supporting hyperplane: fill cheapest coordinates first.
        t = np.clip(budget - cap*np.arange(n), 0., cap)
        gap = g @ w - np.sort(g) @ t
        feas = max(0., -w.min(), w.max()-cap, abs(w.sum()-budget))
        comp = max(np.max(abs(w*np.maximum(0, g-nu))), np.max(abs((cap-w)*np.maximum(0, nu-g))))
        f = lambda x: .5*x@H@x+b@x
        err = abs(f(w) - exhaustive(H,b,cap,budget)) if n <= 3 or (n == 8 and rep in (0,2,3,20,40,60)) else 0.
        worst = np.maximum(worst, [feas, abs(gap), comp, err])
        assert max(feas, abs(gap), comp, err) < 1e-10, (n, rep, feas, gap, comp, err)
        if rep in (20,40,60):
            opt = minimize(f, np.full(n,budget/n), jac=lambda x:H@x+b, method='SLSQP', bounds=[(0,cap)]*n, constraints=[{'type':'eq','fun':lambda x:x.sum()-budget,'jac':lambda x:np.ones(n)}], options={'ftol':1e-11})
            if abs(f(w)-opt.fun) >= 1e-7:
                reference_discrepancies.append((n,rep,bool(opt.success),abs(opt.x.sum()-budget),float(f(w)-opt.fun),str(opt.message)))
print('320 cases; worst feasibility, independent gap, complementarity, exhaustive objective error:',worst.tolist())
print('SLSQP discrepancies (n, case, success, budget error, objective delta, message):',reference_discrepancies)
PY
```
- **Container Exit Code**: `0` (`1186 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
320 cases; worst feasibility, independent gap, complementarity, exhaustive objective error: [5.9396931817445875e-15, 8.668621376273222e-13, 2.1056519103483518e-15, 1.7855938949651318e-11]
SLSQP discrepancies (n, case, success, budget error, objective delta, message): [(24, 40, False, 2.0991504312561915e-08, 8.541601417277889e-07, 'Positive directional derivative for linesearch')]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c103659f9dec72de (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c103659f (directory).
```

### Turn 5 — `submit_completed_task` (`16520 ms` LLM)
- **Rationale**: The saved candidate passed isolated schema validation and independent checks of the transport operator, terminal field, and optimization certificate. No further edits are warranted.
- **Alternatives Weighed**: `Replace the active-set solver with SLSQP` (rejected: The sole discrepancy came from a nonconverged SLSQP result; independent optimality bounds and exhaustive face checks supported the active-set solution.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`13` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:4799'}): Do NOT lock prematurely with `13` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py`. The public schema passed, terminal fields matched an independent dense solve, and the worst optimality gap across 320 stress cases was 8.67e-13. The blind campaign uses the specified open-wall Robin baseline because no diagnostic pulses or archive were available.`

### Turn 6 — `execute_in_remote_docker` (`32681 ms` LLM)
- **Rationale**: The additional audit tests whether pairwise or rational effects improve the worst observed calibration slice, with rational-model selection inside the outer folds. The shift-level residuals quantify the effect of profiling the supplied nuisance covariates without inventing batch labels.
- **Alternatives Weighed**: `Audit relay modes, Fokker–Planck/HJB drift, or classification accuracy` (rejected: This study specifies a steady transport inverse problem with continuous measurements, not a relay, stochastic-control model, or labeled classification cohorts.)

```bash
set -e
cat > /tmp/rosette_audit.py <<'PY'
import importlib.util
import itertools
import pickle
import numpy as np

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', p + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open(p + 'public_batch.pkl', 'rb'))[0]
s = item['study']
r = s['sensor_calibration_sweeps']['ref_rosette']
y = s['sensor_calibration_sweeps']['raw_rosette']
single = np.count_nonzero(r, axis=1) == 1
rng = np.random.default_rng(98)
folds = [np.r_[i,j] for i,j in zip(np.array_split(rng.permutation(np.flatnonzero(single)), 4), np.array_split(rng.permutation(np.flatnonzero(~single)), 4))]
aff = np.c_[np.ones(len(r)), r]
pairs = list(itertools.combinations(range(4), 2))
contrasts = [r[:,i] + sign*r[:,j] for i,j in pairs for sign in (-1,1)]
models = {'affine': aff, 'pair_products_and_curvature': np.column_stack([aff, r*r] + [r[:,i]*r[:,j] for i,j in pairs])}
for a in (.001,.01,.1):
    for j,q in enumerate([*r.T, *contrasts]):
        models[f'quotient_{a}_{j}'] = np.column_stack([aff, r*q[:,None]**2/(1+a*q[:,None]**2)])

def predict(x, fit, test):
    return x[test] @ np.linalg.lstsq(x[fit], y[fit], rcond=None)[0]

def nested(names):
    errors = np.empty_like(y)
    selected = []
    for k,test in enumerate(folds):
        train_folds = [fold for j,fold in enumerate(folds) if j != k]
        fit = np.concatenate(train_folds)
        def score(name):
            x = models[name]
            return sum(np.sum((predict(x, np.concatenate([b for j,b in enumerate(train_folds) if i!=j]), a)-y[a])**2) for i,a in enumerate(train_folds))
        name = min(names, key=score)
        selected.append(name)
        errors[test] = predict(models[name],fit,test)-y[test]
    slices = [np.sqrt(np.mean(errors[mask]**2,axis=0)) for mask in (single,~single) if mask.any()]
    return float(np.sqrt(np.mean(errors**2))), float(np.max(slices)), selected

print('Calibration pooled and worst sweep-type/channel outer RMSE:')
for label,names in [('affine',['affine']), ('paired curvature',['pair_products_and_curvature']), ('rational',[n for n in models if n.startswith('quotient')])]:
    print(label, nested(names))
X,_ = m._design(s)
yc = m._calibrate(s)
z = s['Z_nuisance']
P = np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
out = m.solve_batch(s)
res = X@out['weights']-yc
raw = lambda t: [float(np.sqrt(np.mean(a*a))) for a in np.split(t,len(s['mu_shifts']))]
print('Study keys:', sorted(s))
print('Measured residual RMSE by shift, before and after profiling:', raw(res), raw(res-z@(P@res)))
print('Public certificate gap:', out['kkt_certificate']['primal_objective']-out['kkt_certificate']['dual_lower_bound'])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/rosette_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`613 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Calibration pooled and worst sweep-type/channel outer RMSE:
affine (1.0451383422116492e-05, 1.1895223678677425e-05, ['affine', 'affine', 'affine', 'affine'])
paired curvature (1.1712414349236366e-05, 1.2801592349476454e-05, ['pair_products_and_curvature', 'pair_products_and_curvature', 'pair_products_and_curvature', 'pair_products_and_curvature'])
rational (1.1075791566737765e-05, 1.2126591911771503e-05, ['quotient_0.1_4', 'quotient_0.1_14', 'quotient_0.1_11', 'quotient_0.1_8'])
Study keys: ['Lx', 'Ly', 'V_reaction_2d', 'Z_nuisance', 'aux_manifold_telemetry', 'kappa_x', 'kappa_y', 'mu_shifts', 'nuisance_ridge', 'nx', 'ny', 'robin_x', 'robin_y', 'row_bank_id', 'sensor_calibration_sweeps', 'sensor_ij', 'source_modes', 'source_ridge', 'total_budget', 'u_box', 'vel_x', 'vel_y', 'y_obs_raw']
Measured residual RMSE by shift, before and after profiling: [0.328336560488355, 0.2583814352261179, 0.2566214704710772] [0.006984666927143299, 0.007435894651735842, 0.008926273636142389]
Public certificate gap: 1.0061396160665481e-16
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c103659f9dec72de (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c103659f (directory).
```

### Turn 7 — `submit_completed_task` (`17735 ms` LLM)
- **Rationale**: The saved candidate passed isolated schema validation and independent checks of the transport operator, terminal field, and KKT certificate. The follow-up audit found no measured benefit from a more complex calibration.
- **Alternatives Weighed**: `Pairwise or rational rosette calibration` (rejected: Neither improved pooled or worst-slice held-out calibration RMSE.); `Alternate wall operator` (rejected: No pulse measurements or campaign archive were available to identify one.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:4799'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed in an isolated Python process, terminal fields matched an independent dense solve, and the worst optimality gap across 320 stress cases was 8.67e-13. The solver uses the specified open-wall Robin baseline because no diagnostic pulses or campaign archive were available.`

### Turn 8 — `execute_in_remote_docker` (`33679 ms` LLM)
- **Rationale**: Earlier folds tested pairwise and rational terms but did not hold out entire amplitude ranges or assess three-way terms. A clear held-out gain would warrant further calibration work; absent that gain, retain the already verified artifact.
- **Alternatives Weighed**: `Add bank offsets to the estimator` (rejected: This would change the specified objective; bank-level residuals can reveal whether the supplied projection leaves a systematic pattern without changing the solver.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
from itertools import combinations
import numpy as np
p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', p + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open(p + 'public_batch.pkl', 'rb'))[0]['study']
sw = s['sensor_calibration_sweeps']
x, y = sw['ref_rosette'], sw['raw_rosette']
A = np.c_[np.ones(len(x)), x]
B = np.column_stack([A] + [np.prod(x[:, ids], axis=1) for ids in combinations(range(4), 3)])
folds = np.array_split(np.argsort(np.max(abs(x), axis=1)), 5)
for name, X in [('affine', A), ('affine plus three-way products', B)]:
    scores = []
    for ix in folds:
        train = np.ones(len(x), dtype=bool)
        train[ix] = False
        e = X[ix] @ np.linalg.lstsq(X[train], y[train], rcond=None)[0] - y[ix]
        scores.append(np.sqrt(np.mean(e**2, axis=0)).max())
    print(name, 'worst-channel held-out RMSE by amplitude block:', scores)

bank = np.asarray(s['row_bank_id']).ravel()
z = s['Z_nuisance']
X, _ = m._design(s)
w = m.solve_batch(s)['weights']
r = m._calibrate(s) - X@w
profile = r - z @ np.linalg.solve(z.T@z + s['nuisance_ridge']*np.eye(z.shape[1]), z.T@r)
print('Bank layout:', bank.shape, 'observation rosettes:', len(r)//4)
if len(bank) == len(r)//4:
    labels, groups = np.unique(np.round(bank, 3), return_inverse=True)
    print('Bank readout, count, RMSE before profiling, RMSE after profiling, mean after profiling:')
    for i, label in enumerate(labels):
        mask = groups == i
        q = r.reshape(-1, 4)[mask]
        t = profile.reshape(-1, 4)[mask]
        print(float(label), int(mask.sum()), float(np.sqrt(np.mean(q*q))), float(np.sqrt(np.mean(t*t))), float(np.mean(t)))
PY
```
- **Container Exit Code**: `0` (`392 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
affine worst-channel held-out RMSE by amplitude block: [1.2685233850267341e-05, 1.149623429549803e-05, 1.310729056297809e-05, 1.372291813997721e-05, 1.3082009367711555e-05]
affine plus three-way products worst-channel held-out RMSE by amplitude block: [1.2623019482906764e-05, 1.1627908328662757e-05, 1.4017054442459113e-05, 1.3345585767665335e-05, 1.2748708660085168e-05]
Bank layout: (96,) observation rosettes: 24

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c103659f9dec72de (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c103659f (directory).
```

### Turn 9 — `execute_in_remote_docker` (`26520 ms` LLM)
- **Rationale**: The last run skipped the bank comparison because it assumed one readout per rosette. A row-aligned audit will show whether nuisance projection helps consistently across banks or leaves a bank-specific limitation worth reporting; it does not alter the prescribed objective or repeat the verified solver checks.
- **Alternatives Weighed**: `Group entire rosettes by bank` (rejected: The public study has 96 bank readouts for 96 scalar observations, not 24 readouts for 24 rosettes.); `Add three-way calibration terms` (rejected: They did not improve the worst held-out amplitude block.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', p+'batch_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s = pickle.load(open(p+'public_batch.pkl', 'rb'))[0]['study']
sw = s['sensor_calibration_sweeps']
ids = np.asarray(s['row_bank_id']).reshape(-1)
labels, group = np.unique(np.round(ids, 3), return_inverse=True)
print('Sweep keys:', list(sw), 'bank labels/counts:', list(zip(labels.tolist(), np.bincount(group).tolist()))[:28])
if len(labels) > 28:
    print('Readout does not supply repeated bank strata at this resolution.')
else:
    X, _ = m._design(s); y = m._calibrate(s); Z = s['Z_nuisance']
    w = m.solve_batch(s)['weights']; r = y-X@w
    e = r - Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T@r)
    for j, label in enumerate(labels):
        ix = group == j
        print('bank', label, 'n', int(ix.sum()), 'raw RMSE', float(np.sqrt(np.mean(r[ix]**2))), 'profiled RMSE', float(np.sqrt(np.mean(e[ix]**2))), 'profiled mean', float(e[ix].mean()))
PY
```
- **Container Exit Code**: `0` (`393 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Sweep keys: ['ref_rosette', 'raw_rosette', 'single_bank_sweeps'] bank labels/counts: [(0, 24), (1, 24), (2, 24), (3, 24)]
bank 0 n 24 raw RMSE 0.24871491578887583 profiled RMSE 0.00843587530380151 profiled mean 0.0018545856987810822
bank 1 n 24 raw RMSE 0.2546163300918586 profiled RMSE 0.006897906974066837 profiled mean 1.1313553912885676e-05
bank 2 n 24 raw RMSE 0.2586910394846702 profiled RMSE 0.008556640887657816 profiled mean -0.00025372836008251653
bank 3 n 24 raw RMSE 0.3563017691664187 profiled RMSE 0.007283364639025432 profiled mean -0.002105847334804257

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c103659f9dec72de (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c103659f (directory).
```

### Turn 10 — `submit_completed_task` (`19581 ms` LLM)
- **Rationale**: The bank audit confirmed that the improvement from nuisance projection holds within each bank, not just in pooled data. The unchanged artifact already passed isolated schema validation, independent field checks, and optimization stress tests.
- **Alternatives Weighed**: `Add nonlinear rosette terms` (rejected: Pairwise, rational, and three-way extensions did not improve their respective worst held-out calibration slices.); `Add bank-specific offsets` (rejected: The specified projection already lowered residual error in every bank; extra offsets would change the prescribed objective without held-out evidence of benefit.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed in isolated Python, terminal fields matched an independent dense solve, and the worst optimality gap across 320 stress cases was 8.67e-13. Nuisance projection reduced residual error in all four banks. The solver uses the specified open-wall Robin baseline because no diagnostic pulses or campaign archive were available.`

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
