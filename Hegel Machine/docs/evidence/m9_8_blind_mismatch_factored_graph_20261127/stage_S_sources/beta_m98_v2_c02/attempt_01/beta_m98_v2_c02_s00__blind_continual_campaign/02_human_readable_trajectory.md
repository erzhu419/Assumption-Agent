# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_v2_c02_s00__blind_continual_campaign`

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
| **Total Tokens (Prompt / Cached / Out)** | `241,942` (`232,696` / `111,360` / `9,246`) |
| **Total Episode Wall Time (s)** | `295.91s` |

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

### Turn 1 — `execute_in_remote_docker` (`18598 ms` LLM)
- **Rationale**: Inspect the public schema, calibration structure, and archive inventory before choosing a boundary model. This detects missing diagnostics or incompatible array layouts without dumping data.

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py && python3 - <<'PY'
import pickle,numpy as np
items=pickle.load(open('public_batch.pkl','rb'))
print('batch size',len(items))
item=items[0]; print('item keys', list(item)); d=item.get('campaign_diagnostics'); print('diagnostics',None if d is None else (len(d),[list(x) for x in d[:2]]))
for k,v in item['study'].items():
 if isinstance(v,dict):
  print(k,{i:(type(a).__name__,getattr(a,'shape',None),len(a) if isinstance(a,(list,dict)) else None) for i,a in v.items()})
 else:
  a=np.asarray(v); print(k,a.shape,a.dtype, str(v)[:220] if a.size<20 else '')
PY
if [ -d /tmp/hegel_continual_archive ]; then du -ah /tmp/hegel_continual_archive | sort -h | tail -50; else echo 'No continual archive'; fi
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling'
```
- **Container Exit Code**: `0` (`416 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cribe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```

### Turn 2 — `execute_in_remote_docker` (`21574 ms` LLM)
- **Rationale**: Read the exact output contract and compare affine calibration residuals with the bank sweeps. If the residuals reveal structured nonlinearity, fit that transfer before constructing the inverse problem. Check whether nuisance profiling needs the skill’s outputs or only the prescribed small linear solve.
- **Alternatives Weighed**: `Infer a new wall model from the blind observation alone` (rejected: No diagnostic pulses or archive were found, and the task prescribes the open-wall fallback in that case.)

```bash
cd /holdout_task && cat validate_public_schema.py && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance && python3 - <<'PY'
import pickle,numpy as np
items=pickle.load(open('public_batch.pkl','rb')); print('items',len(items),'diagnostic availability',[i.get('campaign_diagnostics') is not None for i in items]); s=items[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; z=c['single_bank_sweeps']; print('bank IDs',s['row_bank_id'].reshape(-1,4)); print('sensor rows',s['sensor_ij']); print('bank sweep summary',{k:(v.shape, v.min(),v.max()) for k,v in z.items()}); print('sweep samples',{k:v[:12] for k,v in z.items()}); print('telemetry range',np.ptp(s['aux_manifold_telemetry'],axis=0)); print('cal range',np.min(x,axis=0),np.max(x,axis=0),'obs raw range',np.min(s['y_obs_raw']),np.max(s['y_obs_raw']));
a=np.c_[np.ones(len(x)),x]; b=np.linalg.lstsq(a,y,rcond=None)[0]; e=y-a@b; print('affine coef',b,'rmse',np.sqrt(np.mean(e**2,axis=0)),'max',abs(e).max(axis=0)); print('cal examples',np.c_[x,y,e][:10]);
PY
```
- **Container Exit Code**: `0` (`437 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
21340652614682
affine coef [[ 2.36450331e-06 -1.15985120e-06  7.13804475e-07  7.81563565e-09]
 [ 9.99999258e-01  2.95088476e-09 -2.31772644e-06  7.04958055e-07]
 [-2.78583639e-06  1.00000236e+00 -1.56007306e-06  2.19707739e-06]
 [ 5.25208368e-07  2.00530456e-06  9.99997844e-01 -2.29827597e-07]
 [ 2.93973554e-07 -1.10749818e-06 -1.42779474e-06  9.99998650e-01]] rmse [9.03526852e-06 9.66249776e-06 1.01939830e-05 1.09747472e-05] max [2.38345334e-05 3.13957529e-05 3.24403004e-05 3.04286034e-05]
cal examples [[-1.20000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  -1.19999263e+00  1.91188163e-06  7.92552738e-06 -3.50791987e-07
   4.11640496e-06  3.07527389e-06  4.43045117e-06  4.87342044e-07]
 [-4.00000000e-01  0.00000000e+00  0.00000000e+00  0.00000000e+00
  -3.99996665e-01 -3.51923532e-06 -8.18894870e-06 -1.65105071e-05
   6.73408980e-07 -2.35820377e-06 -9.82984375e-06 -1.62363395e-05]
 [ 5.00000000e-01  0.00000000e+00  0.00000000e+00  0.00000000e+00
   5.00003869e-01 -1.07068011e-05  2.06220915e-06  8.06231010e-06
   1.87512822e-06 -9.54842538e-06  2.50726790e-06  7.70201544e-06]
 [ 1.30000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   1.30001411e+00 -1.08911781e-05  3.10520163e-06  1.36437104e-05
   1.27091429e-05 -9.73516310e-06  5.40444153e-06  1.27194493e-05]
 [ 0.00000000e+00 -1.20000000e+00  0.00000000e+00  0.00000000e+00
   4.50694856e-06 -1.19999259e+00 -4.65233887e-06  5.27351091e-06
  -1.20055841e-06  1.13947521e-05 -7.23823101e-06  7.90218814e-06]
 [ 0.00000000e+00 -4.00000000e-01  0.00000000e+00  0.00000000e+00
  -2.95915882e-06 -4.00007420e-01 -5.98911606e-06 -7.68057325e-06
  -6.43799668e-06 -5.31674810e-06 -7.32694976e-06 -6.80955793e-06]
 [ 0.00000000e+00  5.00000000e-01  0.00000000e+00  0.00000000e+00
  -2.31226887e-06  4.99990474e-01  1.19899938e-05 -5.61233897e-06
  -3.28385399e-06 -9.54482446e-06  1.20562258e-05 -6.71869330e-06]
 [ 0.00000000e+00  1.30000000e+00  0.00000000e+00  0.00000000e+00
   3.22047462e-06  1.30000915e+00  5.91658320e-06  6.74481536e-06
   4.47755861e-06  7.24946945e-06  7.23087370e-06  3.88079912e-06]
 [ 0.00000000e+00  0.00000000e+00 -1.20000000e+00  0.00000000e+00
  -1.11975872e-05 -1.05382523e-05 -1.20000848e+00  2.37131494e-06
  -1.29318404e-05 -6.97203565e-06 -1.17858719e-05  2.08770619e-06]
 [ 0.00000000e+00  0.00000000e+00 -4.00000000e-01  0.00000000e+00
  -9.84115422e-06 -1.18407345e-05 -4.00005313e-01  4.67073820e-07
  -1.19955742e-05 -9.87876149e-06 -6.88950690e-06  3.67327145e-07]]
```

### Turn 3 — `execute_in_remote_docker` (`52470 ms` LLM)
- **Rationale**: Implement the prescribed wall fallback and certificate, then run the public validator to catch mismatched study keys, output shapes, and certificate violations.
- **Alternatives Weighed**: `Nonlinear rosette transfer` (rejected: The affine fit already matches the public sweep at its observed noise scale.); `Schur-complement ridge objective` (rejected: The specification explicitly minimizes the squared residual after nuisance subtraction, not the jointly profiled objective.)

```bash
cat > /holdout_task/batch_solver.py <<'PY'
"""Steady transport tomography with empirical rosette calibration."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h, robin):
    faces = (0.5 / h**2) * (kappa[:-1] + kappa[1:])
    lower = -faces - np.maximum(velocity[1:], 0) / h
    upper = -faces - np.maximum(-velocity[:-1], 0) / h
    lower[-1] = -kappa[-1] / h**2
    upper[0] = -kappa[0] / h**2
    diag = np.empty_like(kappa, dtype=float)
    diag[1:-1] = -lower[:-1] - upper[1:]
    diag[0] = -upper[0] + robin / h
    diag[-1] = -lower[-1] + robin / h
    return sparse.diags((lower, diag, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1), study['robin_x'])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1), study['robin_y'])
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(reaction, format='csc'))
    ident = sparse.eye(nx * ny, format='csc')
    sources = np.asarray(study['source_modes']).reshape(-1, nx * ny, order='F').T
    ii, jj = np.asarray(study['sensor_ij']).T
    rows = ii + nx * jj
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * ident).solve(sources)
        blocks.append(fields[rows])
    return np.vstack(blocks), fields.T.reshape(-1, nx, ny, order='F')


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    offset, *rows = np.linalg.lstsq(np.c_[np.ones(len(ref)), ref], raw, rcond=None)[0]
    observation = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return np.linalg.solve(np.asarray(rows).T, (observation - offset).T).T.ravel()


def _box_budget_qp(H, b, cap, total):
    """Feasible active-set minimization of 0.5*w.T@H@w - b.T@w."""
    n = len(b)
    if total == 0 or total == n * cap:
        w = np.full(n, 0 if total == 0 else cap, dtype=float)
        g = H @ w - b
        return w, float(g.min() if total == 0 else g.max())
    w = np.full(n, total / n)
    active = np.zeros(n, dtype=int)
    tol = 1e-12
    for _ in range(20 * n * n):
        free = np.flatnonzero(active == 0)
        g = H @ w - b
        if free.size:
            ones = np.ones(free.size)
            inv_g, inv_one = np.linalg.solve(H[np.ix_(free, free)], np.vstack((g[free], ones)).T).T
            nu = float(inv_g.sum() / inv_one.sum())
            p = nu * inv_one - inv_g
            to_bound = np.full(free.size, np.inf)
            down, up = p < -tol, p > tol
            to_bound[down] = -w[free[down]] / p[down]
            to_bound[up] = (cap - w[free[up]]) / p[up]
            j = np.argmin(to_bound)
            if to_bound[j] < 1:
                w[free] += max(0., to_bound[j]) * p
                i = free[j]
                active[i] = 1 if p[j] > 0 else -1
                w[i] = cap if active[i] == 1 else 0.
                continue
            w[free] += p
            g = H @ w - b
        else:
            nu = float((g[active == 1].max() + g[active == -1].min()) / 2)
        violation = active * (g - nu)
        i = np.argmin(violation)
        if violation[i] >= -tol:
            return w, nu
        active[i] = 0
    raise RuntimeError('Box-budget active set did not converge')


def solve_batch(study: dict) -> dict:
    S, last = _forward(study)
    y = _calibrate(study)
    Z = np.asarray(study['Z_nuisance'])
    data = np.column_stack((S, y))
    P = np.linalg.solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    projected = data - Z @ (P @ data)
    A, target = projected[:, :-1], projected[:, -1]
    ridge, cap, total = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    H = A.T @ A + ridge * np.eye(S.shape[1])
    w, nu = _box_budget_qp(H, A.T @ target, cap, total)
    residual = A @ w - target
    value = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = A.T @ residual + ridge * w
    lower = np.maximum(0, g - nu)
    upper = np.maximum(0, nu - g)
    bound = float(value - g @ w + nu * total - cap * upper.sum())
    return {'weights': w, 'terminal_field': np.tensordot(w, last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': lower, 'lam_upper': upper,
                                'primal_objective': value, 'dual_lower_bound': bound}}
PY
cd /holdout_task && python3 - <<'PY'
import pickle
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']
print('velocity keys:',[k for k in s if 'vel' in k])
PY
python3 /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`538 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
velocity keys: ['vel_x', 'vel_y']
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c3c92ce5fd07cd1f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c3c92ce5 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`40558 ms` LLM)
- **Rationale**: Compare the exact artifact against an independently assembled dense operator to catch stencil and indexing errors. Check held-out sweep residuals for calibration lack of fit, and stress active bounds and extreme budgets to catch optimizer failures before submission.
- **Alternatives Weighed**: `Submit on the public schema result alone` (rejected: The schema test would not detect a transposed field, a stencil error, or a certificate inconsistent with the specified objective.)

```bash
set -e
cd /holdout_task
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
p='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('candidate',p)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
items=pickle.load(open('/holdout_task/public_batch.pkl','rb'))

def axis(k,v,h,r):
    n=len(k); a=np.zeros((n,n))
    for i in range(n):
        if i in (0,n-1):
            a[i,i]=k[i]/h**2+r/h; a[i,1 if i==0 else n-2]=-k[i]/h**2
        else:
            kp=(k[i]+k[i+1])/2; km=(k[i]+k[i-1])/2
            a[i,i-1]=-km/h**2-max(v[i],0)/h
            a[i,i+1]=-kp/h**2-max(-v[i],0)/h
            a[i,i]=(kp+km)/h**2+abs(v[i])/h
    return a

worst=np.zeros(7)
for item in items:
    s=item['study']; nx,ny=s['nx'],s['ny']
    ax=axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'])
    ay=axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'])
    L=np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
    U=[np.array([np.linalg.solve(L+mu*np.eye(nx*ny),src.ravel(order='F')).reshape((nx,ny),order='F') for src in s['source_modes']]) for mu in s['mu_shifts']]
    i,j=s['sensor_ij'].T; S=np.concatenate([u[:,i,j].T for u in U]); S_actual,U_actual=m._forward(s)
    o=m.solve_batch(s); w=o['weights']; c=o['kkt_certificate']; lo,hi=c['lam_lower'],c['lam_upper']; cap=s['u_box']
    Z=s['Z_nuisance']; R=np.eye(len(Z))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
    A=R@S; e=A@w-R@m._calibrate(s); g=A.T@e+s['source_ridge']*w
    J=(e@e+s['source_ridge']*(w@w))/2
    sweep=s['sensor_calibration_sweeps']; ref=sweep['ref_rosette']; raw=sweep['raw_rosette']; X=np.c_[np.ones(len(ref)),ref]; err=[]
    for fold in range(5):
        test=np.arange(len(ref))%5==fold
        err.append(np.sqrt(np.mean((raw[test]-X[test]@np.linalg.lstsq(X[~test],raw[~test],rcond=None)[0])**2,axis=0)).max())
    worst=np.maximum(worst,[max(abs(S-S_actual).max(),abs(U[-1]-U_actual).max()),abs(o['terminal_field']-w@U[-1].reshape(len(w),-1).reshape(len(w),nx*ny)).max() if False else abs(o['terminal_field']-np.einsum('k,kij->ij',w,U[-1])).max(),abs(g-c['nu_sum']-lo+hi).max(),max(abs(w*lo).max(),abs((cap-w)*hi).max()),abs(J-c['primal_objective']),abs(J-c['dual_lower_bound']),max(err)])
assert np.all(worst[:6]<1e-9),worst
assert worst[6]<5e-5,worst
print(dict(zip(('forward_max','terminal_max','stationarity_max','complementarity_max','objective_error_max','duality_gap_max','calibration_fold_rmse_max'),worst)))
rng=np.random.default_rng(42); max_gap=0.
for n in (2,8,20):
    for fraction in (0.,0.001,0.25,0.5,0.999,1.):
        for k in range(25):
            D=rng.normal(size=(n,n)); H=D.T@D+0.04*np.eye(n); b=10*rng.normal(size=n); cap=0.42; total=n*cap*fraction
            w,nu=m._box_budget_qp(H,b,cap,total); g=H@w-b; d=g-nu
            gap=d@w-cap*np.minimum(d,0).sum()
            assert w.min()>=-1e-10 and w.max()<=cap+1e-10 and abs(w.sum()-total)<1e-10 and abs(gap)<1e-9,(n,fraction,k,gap)
            max_gap=max(max_gap,abs(gap))
print('450 QP stress cases: worst linearized duality gap',max_gap)
PY
```
- **Container Exit Code**: `1` (`574 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
{'forward_max': 3.191891195797325e-16, 'terminal_max': 1.942890293094024e-16, 'stationarity_max': 4.0939474033052647e-16, 'complementarity_max': 8.608257581381992e-18, 'objective_error_max': 1.3877787807814457e-17, 'duality_gap_max': 5.377642775528102e-17, 'calibration_fold_rmse_max': 1.4007783129833608e-05}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c3c92ce5fd07cd1f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c3c92ce5 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 45, in <module>
  File "/holdout_task/batch_solver.py", line 87, in _box_budget_qp
    raise RuntimeError('Box-budget active set did not converge')
RuntimeError: Box-budget active set did not converge
```

### Turn 5 — `execute_in_remote_docker` (`27373 ms` LLM)
- **Rationale**: Isolate the first failing QP and trace its final bound transitions. The trace will distinguish cycling from slow progress and guide a targeted fix without changing the independently verified forward model.
- **Alternatives Weighed**: `Increase the iteration limit` (rejected: A longer limit would not fix an active-set cycle.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, sys
from collections import deque
import numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rng=np.random.default_rng(42)
for n in (2,8,20):
    for fraction in (0.,0.001,0.25,0.5,0.999,1.):
        for k in range(25):
            D=rng.normal(size=(n,n)); H=D.T@D+0.04*np.eye(n); b=10*rng.normal(size=n); cap=0.42; total=n*cap*fraction
            try: m._box_budget_qp(H,b,cap,total)
            except RuntimeError:
                print('first failing case:',n,fraction,k,'budget',total,flush=True)
                np.savez('/tmp/hegel_qp_repro.npz',H=H,b=b,cap=cap,total=total)
                trace=deque(maxlen=14)
                def tracer(frame,event,arg):
                    if event=='line' and frame.f_code is m._box_budget_qp.__code__ and frame.f_lineno in (57,71,75,79,86):
                        d=frame.f_locals; trace.append((frame.f_lineno,d.get('active',np.array([])).copy().tolist(),str(d.get('p')),str(d.get('to_bound')),d.get('i'),d.get('nu')))
                    return tracer
                sys.settrace(tracer)
                try: m._box_budget_qp(H,b,cap,total)
                except RuntimeError: pass
                finally: sys.settrace(None)
                for row in trace: print(row)
                raise SystemExit(0)
raise SystemExit('No failure reproduced')
PY
```
- **Container Exit Code**: `0` (`464 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
first failing case: 2 0.001 0 budget 0.00084
(75, [0, 0], '[-8.05555605  8.05555605]', '[0.         0.05203365]', 0, -14.694615501852681)
(71, [-1, 0], '[0.]', '[inf]', 0, -17.678165913573167)
(79, [-1, 0], '[0.]', '[inf]', 0, -17.678165913573167)
(86, [-1, 0], '[0.]', '[inf]', 0, -17.678165913573167)
(71, [0, 0], '[-8.05555605  8.05555605]', '[0.         0.05203365]', 0, -14.694615501852681)
(75, [0, 0], '[-8.05555605  8.05555605]', '[0.         0.05203365]', 0, -14.694615501852681)
(71, [-1, 0], '[0.]', '[inf]', 0, -17.678165913573167)
(79, [-1, 0], '[0.]', '[inf]', 0, -17.678165913573167)
(86, [-1, 0], '[0.]', '[inf]', 0, -17.678165913573167)
(71, [0, 0], '[-8.05555605  8.05555605]', '[0.         0.05203365]', 0, -14.694615501852681)
(75, [0, 0], '[-8.05555605  8.05555605]', '[0.         0.05203365]', 0, -14.694615501852681)
(71, [-1, 0], '[0.]', '[inf]', 0, -17.678165913573167)
(79, [-1, 0], '[0.]', '[inf]', 0, -17.678165913573167)
(86, [-1, 0], '[0.]', '[inf]', 0, -17.678165913573167)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c3c92ce5fd07cd1f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c3c92ce5 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`33962 ms` LLM)
- **Rationale**: A lower-bound variable is optimal when its gradient is at least nu, and an upper-bound variable is optimal when its gradient is at most nu. Correct the reversed test, then rerun the saved failure, the full budget stress suite, and the public certificate checks in an isolated process.
- **Alternatives Weighed**: `Replace the active-set solver` (rejected: The trace isolates a reversed sign in the bound-release rule; the stencil, calibration, and certificate formulas need no change.)

```bash
set -e
python3 -I - <<'PY'
from pathlib import Path
p = Path('/holdout_task/batch_solver.py')
s = p.read_text()
old = '        violation = active * (g - nu)\n'
assert s.count(old) == 1
p.write_text(s.replace(old, '        # A lower bound needs g >= nu; an upper bound needs g <= nu.\n        violation = -active * (g - nu)\n'))
PY
python3 /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
worst_gap = worst_budget = 0.
def check(H, b, cap, total):
    global worst_gap, worst_budget
    w, nu = m._box_budget_qp(H, b, cap, total)
    d = H @ w - b - nu
    gap = abs(d @ w - cap * np.minimum(d, 0).sum())
    budget = abs(w.sum() - total)
    assert w.min() >= -1e-10 and w.max() <= cap + 1e-10 and budget < 1e-10 and gap < 1e-9, (len(b), total, w, gap)
    worst_gap = max(worst_gap, gap)
    worst_budget = max(worst_budget, budget)
with np.load('/tmp/hegel_qp_repro.npz') as repro:
    check(repro['H'], repro['b'], float(repro['cap']), float(repro['total']))
rng = np.random.default_rng(42)
for n in (2, 8, 20):
    for fraction in (0., 0.001, 0.25, 0.5, 0.999, 1.):
        for k in range(25):
            D = rng.normal(size=(n,n)); H = D.T @ D + 0.04 * np.eye(n)
            check(H, 10 * rng.normal(size=n), 0.42, n * 0.42 * fraction)
print('Saved reproduction and 450 QPs passed:', {'worst_duality_gap': worst_gap, 'worst_budget_error': worst_budget})
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    s = item['study']; o = m.solve_batch(s); w = o['weights']; c = o['kkt_certificate']; cap = s['u_box']
    S, U = m._forward(s); Z = s['Z_nuisance']; R = np.eye(len(Z)) - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    r = R @ (S @ w - m._calibrate(s)); g = (R @ S).T @ r + s['source_ridge'] * w
    J = 0.5 * (r @ r + s['source_ridge'] * (w @ w)); lo = c['lam_lower']; hi = c['lam_upper']
    errors = {'terminal': np.max(abs(o['terminal_field'] - np.einsum('k,kij->ij', w, U))), 'stationarity': np.max(abs(g - c['nu_sum'] - lo + hi)), 'complementarity': max(np.max(abs(w * lo)), np.max(abs((cap - w) * hi))), 'objective': abs(J - c['primal_objective']), 'duality_gap': abs(J - c['dual_lower_bound'])}
    assert np.array_equal(w, c['weights']) and max(errors.values()) < 1e-9, errors
    print('Public numerical errors:', errors)
PY
```
- **Container Exit Code**: `0` (`687 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
Saved reproduction and 450 QPs passed: {'worst_duality_gap': 1.1368683772161603e-13, 'worst_budget_error': 4.274358644806853e-15}
Public numerical errors: {'terminal': 1.3877787807814457e-17, 'stationarity': 1.3877787807814457e-17, 'complementarity': 8.608257581381992e-18, 'objective': 0.0, 'duality_gap': 3.9898639947466563e-17}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=96ea58fe3935e74b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_96ea58fe (directory).
```

### Turn 7 — `submit_completed_task` (`15649 ms` LLM)
- **Rationale**: The final artifact passed isolated numerical verification. The bound-release correction resolved the reproduced cycle, and the public certificate matched the specified objective with a duality gap of 3.99e-17.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:4700'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema and all 450 optimization stress cases passed. With no diagnostic pulses or campaign archive available, the solver uses the specified open-wall Robin baseline.`

### Turn 8 — `execute_in_remote_docker` (`34021 ms` LLM)
- **Rationale**: Test whether pairwise interactions, cubic terms, or rational saturation improve prediction on held-out real sweeps, selecting rational curvature only inside each training fold. Stratify by sweep type so mixed-channel errors cannot hide in the pooled score. Compare residuals by shift and channel to see where nuisance profiling helps, without changing the prescribed blind wall fallback.
- **Alternatives Weighed**: `Apply classification, relay, or stochastic drift audits` (rejected: The task is steady-state continuous tomography, with no class labels, relay state, or stochastic time evolution. The relevant nuisance audit compares residuals before and after profiling by shift and channel.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, pickle
import numpy as np
p='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: items=pickle.load(f)
for item in items:
    s=item['study']; c=s['sensor_calibration_sweeps']; x=np.asarray(c['ref_rosette']); y=np.asarray(c['raw_rosette'])
    pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
    a=np.c_[np.ones(len(x)),x]; q=np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)])
    t=np.column_stack([x[:,i]*x[:,j]*x[:,k] for i in range(4) for j in range(i,4) for k in range(j,4)])
    u=np.column_stack([x]+[x[:,i:i+1]+sign*x[:,j:j+1] for i,j in pairs for sign in (-1,1)])
    families={'affine':[a],'quadratic':[np.c_[a,q]],'cubic':[np.c_[a,q,t]],'rational':[np.c_[a,u/(1+alpha*u**2)-u] for alpha in (0.01,0.1,1.)]}
    def pred(X,train,test): return X[test]@np.linalg.lstsq(X[train],y[train],rcond=None)[0]
    fold=np.random.default_rng(314).permutation(len(x))%5; strata=np.count_nonzero(x,axis=1); strata=np.where(strata==0,0,np.where(strata==1,1,2))
    for name, designs in families.items():
        err=np.empty_like(y); fold_max=[]
        for k in range(5):
            train=fold!=k; test=~train
            if len(designs)>1:
                sub=np.flatnonzero(train); inner=np.arange(len(sub))%4
                def score(X): return np.mean([np.mean((pred(X,sub[inner!=j],sub[inner==j])-y[sub[inner==j]])**2) for j in range(4)])
                X=min(designs,key=score)
            else: X=designs[0]
            err[test]=pred(X,train,test)-y[test]; fold_max.append(np.sqrt(np.mean(err[test]**2,axis=0)).max())
        print(name,'worst fold-channel RMSE',max(fold_max),'slice RMSE',[(int(j),int(sum(strata==j)),float(np.sqrt(np.mean(err[strata==j]**2,axis=0)).max())) for j in np.unique(strata)])
    S,U=m._forward(s); o=m.solve_batch(s); w=o['weights']; Z=s['Z_nuisance']; yobs=m._calibrate(s)
    raw_res=yobs-S@w; prof_res=raw_res-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@raw_res)
    shape=(len(s['mu_shifts']),-1,4)
    print('shift-by-channel RMS (unprofiled, profiled):',*[np.sqrt(np.mean(r.reshape(shape)**2,axis=1)).round(8).tolist() for r in (raw_res,prof_res)])
    c=o['kkt_certificate']; print('certificate gap',c['primal_objective']-c['dual_lower_bound'],'study technical strata',[k for k in s if 'batch' in k or 'cohort' in k],'diagnostic pulses',item.get('campaign_diagnostics') is not None)
PY
```
- **Container Exit Code**: `0` (`592 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
affine worst fold-channel RMSE 1.5656350554138164e-05 slice RMSE [(1, 16, 1.0515242479084997e-05), (2, 64, 1.2682921349481382e-05)]
quadratic worst fold-channel RMSE 1.706006401992323e-05 slice RMSE [(1, 16, 1.0407662623992394e-05), (2, 64, 1.5339747324763992e-05)]
cubic worst fold-channel RMSE 2.9013188160427497e-05 slice RMSE [(1, 16, 1.347973296611756e-05), (2, 64, 2.18202727020929e-05)]
rational worst fold-channel RMSE 1.708630364560068e-05 slice RMSE [(1, 16, 1.1113632408168027e-05), (2, 64, 1.4006331036356585e-05)]
shift-by-channel RMS (unprofiled, profiled): [[0.15540595, 0.09662555, 0.25820091, 0.18029621], [0.16333727, 0.18388399, 0.13512437, 0.21209592], [0.1292193, 0.18030225, 0.22831943, 0.15898815]] [[0.00863876, 0.00573346, 0.01007922, 0.00800928], [0.00712042, 0.00493468, 0.0067574, 0.00726253], [0.01141296, 0.01078486, 0.0047614, 0.00472592]]
certificate gap 3.9898639947466563e-17 study technical strata [] diagnostic pulses False

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=96ea58fe3935e74b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_96ea58fe (directory).
```

### Turn 9 — `submit_completed_task` (`13490 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated execution and certificate checks. Nuisance profiling reduced RMS in every shift-channel slice, and none of the calibration challengers improved the worst slice.
- **Alternatives Weighed**: `Quadratic, cubic, or rational calibration` (rejected: All three increased the worst held-out fold-channel RMSE and worsened the mixed-channel slice relative to affine calibration.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:4700'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `solve_batch(study)` in `/holdout_task/batch_solver.py`. The public schema and all 450 optimization stress cases passed. The solver uses the specified open-wall Robin fallback because neither diagnostic pulses nor a campaign archive was available.`

### Turn 10 — `submit_completed_task` (`15089 ms` LLM)
- **Rationale**: The exact submitted artifact passed isolated execution and numerical certificate checks. Nuisance profiling reduced residual RMS in every shift-channel slice, and the repeated audit request introduced no new evidence or applicable test.
- **Alternatives Weighed**: `Quadratic, cubic, or rational rosette calibration` (rejected: Each increased the worst held-out fold-channel RMSE and worsened the mixed-channel slice.)
- **Submitted Final Answer / Summary**: `Implemented and verified `solve_batch(study)` in `/holdout_task/batch_solver.py`. The public schema and all 450 optimization stress cases passed. The solver uses the specified open-wall Robin fallback because no diagnostic pulses or campaign archive were available.`

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
