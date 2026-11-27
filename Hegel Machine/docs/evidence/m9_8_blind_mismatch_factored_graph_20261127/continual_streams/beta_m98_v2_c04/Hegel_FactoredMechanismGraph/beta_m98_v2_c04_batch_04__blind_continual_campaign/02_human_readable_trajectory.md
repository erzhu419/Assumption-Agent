# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_v2_c04_batch_04__blind_continual_campaign__upstream_transducer`

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
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `225,054` (`217,099` / `61,824` / `7,955`) |
| **Total Episode Wall Time (s)** | `271.36s` |

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

### Turn 1 — `execute_in_remote_docker` (`12920 ms` LLM)
- **Rationale**: Inspect the composition and public schema to catch any mismatch in item layout or coordinate handling before fitting; look up a robust primitive because spike rows may distort ordinary least squares.

```bash
cd /holdout_task && ls -l && cat batch_solver.py validate_public_schema.py public_asset_index.json && sed -n '1,230p' _verified_downstream.py && ls -l /tmp/hegel_continual_archive && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'robust IRLS Cauchy nonlinear regression'
```
- **Container Exit Code**: `0` (`322 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit | STATUS: APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)
autogenous_prim_robust_affine_projection@abc35d3d | profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | For predictor, target and optional weights (n, m) and predictor sensitivity (n | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.profile_weighted_affine | STATUS: APPLICABLE_SUBROUTINE(real_2d_predictor_3d_sensitivity_2d_target)
autogenous_prim_robust_affine_projection@abc35d3d | propagate_linear_basis(transition: np.ndarray, initial: np.ndarray, increments: np.ndarray) -> np.ndarray | Return a trajectory of shape (steps+1, n, k) from transition (n, n), initial ( | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.propagate_linear_basis | STATUS: APPLICABLE_SUBROUTINE(real_linear_transition_and_2d_initial_basis)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None] | Fit finite vectors time, predictor, and values (n,) with n >= 4, increasing no | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_anchored_relaxation | STATUS: NOT_NEEDED_FOR_STATIC_REGRESSION(1d_monotone_time_relaxation_only)
autogenous_prim_robust_trend_decomposition@1f45421a | fit_local_level_step(predictor: np.ndarray, values: np.ndarray, splits: np.ndarray, *, scale: float, loss_clip: float) -> tuple[int, float] | Fit finite predictor and values (n,) and integer splits (k,) with k >= 1, at l | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | --describe autogenous_prim_robust_trend_decomposition.fit_local_level_step | STATUS: NOT_NEEDED_FOR_STATIC_REGRESSION(1d_step_split_only)
```

### Turn 2 — `execute_in_remote_docker` (`18557 ms` LLM)
- **Rationale**: Inspect sweep coverage and downstream array usage to detect underidentified cross-channel terms, extrapolation, or a pulse layout mismatch before choosing the calibration basis and inverse.

```bash
cd /holdout_task && PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_affine_projection.robust_residual_weights | head -45; cd /holdout_task && python3 - <<'PY'
import numpy as np,pickle
v=pickle.load(open('public_batch.pkl','rb')); print('items',len(v))
for i,it in enumerate(v):
 s=it['study']; print('item',i,'study keys',list(s)); c=s['sensor_calibration_sweeps'];print('cal keys',list(c)); x,y=(np.asarray(c[k]) for k in ('ref_rosette','raw_rosette')); print('cal shape',x.shape,y.shape,'first rows',np.c_[x,y][:8]); print('unique', [len(np.unique(q)) for q in x.T]);print('bounds ref',np.min(x,0),np.max(x,0)); print('single',type(c['single_bank_sweeps']),list(c['single_bank_sweeps'])[:4]);print('obs',np.shape(s['y_obs_raw']),np.percentile(np.asarray(s['y_obs_raw']).reshape(-1,4),[0,50,100],axis=0)); p=s['reference_tracer_pulses'];print('pulses',None if p is None else [(list(q),np.shape(q['y_pulse_raw'])) for q in p]);
PY
sed -n '1,165p' /holdout_task/_verified_downstream.py
```
- **Container Exit Code**: `0` (`449 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
loat((g[high] + g[low]) / 2)
            state[[low, high]] = 0
            continue
        inv_g, inv_one = np.linalg.solve(hessian[np.ix_(free, free)],
                                       np.column_stack((g[free], np.ones(free.sum())))).T
        nu = (inv_g.sum() + budget - w.sum()) / inv_one.sum()
        direction = nu * inv_one - inv_g
        current = w[free]
        trial = current + direction
        hits_lower = trial < 0
        hits_upper = trial > cap
        if (hits_lower | hits_upper).any():
            ratios = np.full(len(current), np.inf)
            ratios[hits_lower] = -current[hits_lower] / direction[hits_lower]
            ratios[hits_upper] = (cap - current[hits_upper]) / direction[hits_upper]
            hit = ratios.argmin()
            w[free] = np.clip(current + ratios[hit] * direction, 0, cap)
            idx = np.flatnonzero(free)[hit]
            state[idx] = 1 if hits_upper[hit] else -1
            w[idx] = cap if hits_upper[hit] else 0
            continue
        w[free] = trial
        g = hessian @ w + linear
        violation = state * (g - nu)
        idx = violation.argmax()
        if violation[idx] <= tol:
            return w, float(nu)
        state[idx] = 0
    raise RuntimeError('Box-budget active set did not converge')


def solve_batch(study: dict) -> dict:
    design, terminal_modes = _forward(study)
    calibrated = _calibrate(study)
    z = np.asarray(study['Z_nuisance'])
    values = np.column_stack((design, calibrated))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    projected = values - z @ np.linalg.solve(gram, z.T @ values)
    x, y = projected[:, :-1], projected[:, -1]
    ridge = float(study['source_ridge'])
    cap, budget = float(study['u_box']), float(study['total_budget'])
    w, nu = _box_budget(x.T @ x + ridge * np.eye(x.shape[1]), -x.T @ y, cap, budget)
    residual = x @ w - y
    g = x.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    delta = g - nu
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(),
            'nu_sum': nu,
            'lam_lower': np.maximum(delta, 0).tolist(),
            'lam_upper': np.maximum(-delta, 0).tolist(),
            'primal_objective': objective,
            'dual_lower_bound': float(objective - g @ w + nu * budget + cap * np.minimum(delta, 0).sum()),
        },
    }
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
KeyError: 'reference_tracer_pulses'
```

### Turn 3 — `execute_in_remote_docker` (`16767 ms` LLM)
- **Rationale**: The first inspection stopped at an absent pulse key. Inspect it as optional and compare observation and sweep ranges: extrapolation would favor a bounded saturation basis over a high-degree polynomial.

```bash
cd /holdout_task && python3 - <<'PY'
import numpy as np,pickle
v=pickle.load(open('public_batch.pkl','rb')); print('items',len(v))
for i,it in enumerate(v):
 s=it['study']; c=s['sensor_calibration_sweeps']; x,y=(np.asarray(c[k],float) for k in ('ref_rosette','raw_rosette')); print(i,'item keys',list(it),'study keys',list(s)); print('cal',x.shape,y.shape,'keys',list(c),'single',type(c.get('single_bank_sweeps')), 'unique',[len(np.unique(q)) for q in x.T]);print('ref percentiles',np.percentile(x,[0,50,100],axis=0));print('raw percentiles',np.percentile(y,[0,50,100],axis=0)); o=np.asarray(s['y_obs_raw']); print('obs',o.shape,np.percentile(o.reshape(-1,4),[0,50,100],axis=0)); p=s.get('reference_tracer_pulses'); print('pulse layout',None if p is None else [(list(q),np.shape(q['y_pulse_raw'])) for q in p]); print('examples',np.round(np.c_[x,y][:4],5))
PY
head -60 _verified_downstream.py
```
- **Container Exit Code**: `0` (`430 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0.08    -0.36407]]
"""Steady advection–diffusion–reaction tomography on a tensor grid."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, length, robin):
    n = len(kappa)
    h = length / (n - 1)
    faces = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = -faces - np.maximum(velocity[1:], 0) / h
    upper = -faces - np.maximum(-velocity[:-1], 0) / h
    upper[0], lower[-1] = -kappa[0] / h**2, -kappa[-1] / h**2
    main = np.r_[-upper[0] + robin / h,
                 -lower[:-1] - upper[1:],
                 -lower[-1] + robin / h]
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'], study['robin_x'])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'], study['robin_y'])
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    op = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
    op += diags(reaction, format='csc')
    modes = np.asarray(study['source_modes'])
    rhs = modes.transpose(0, 2, 1).reshape(len(modes), nx * ny).T
    i, j = np.asarray(study['sensor_ij']).T
    idx = i + nx * j
    identity = eye(nx * ny, format='csc')
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(op + shift * identity).solve(rhs)
        blocks.append(fields[idx])
    return np.vstack(blocks), fields.T.reshape(len(modes), ny, nx).transpose(0, 2, 1)


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = np.asarray(sweeps['ref_rosette']), np.asarray(sweeps['raw_rosette'])
    offset, *mix = np.linalg.lstsq(np.column_stack((np.ones(len(ref)), ref)), raw, rcond=None)[0]
    observed = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return np.linalg.solve(np.asarray(mix).T, (observed - offset).T).T.ravel()


def _box_budget(hessian, linear, cap, budget):
    """Primal active-set solve, with equality multiplier in g - nu convention."""
    n = len(linear)
    w = np.full(n, budget / n)
    if budget == 0 or budget == n * cap:
        g = hessian @ w + linear
        return w, float(g.min() if budget == 0 else g.max())
    state = np.zeros(n, dtype=np.int8)
    tol = 1e-12 * max(1., np.linalg.norm(hessian, np.inf) * cap, np.linalg.norm(linear, np.inf))
    for _ in range(50 * n):
        free = state == 0
        g = hessian @ w + linear
        if not free.any():
```

### Turn 4 — `execute_in_remote_docker` (`28342 ms` LLM)
- **Rationale**: Compare affine, pairwise-product, cubic, and rational saturation bases on held-out sweep rows. The axis sweeps isolate self-channel curvature; the mixed sweeps reveal cross-channel terms that would otherwise bias the inverse.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle, numpy as np
from itertools import combinations
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=map(np.asarray,(c['ref_rosette'],c['raw_rosette'])); n=len(x); o=np.ones((n,1)); pairs=list(combinations(range(4),2)); q=np.stack([x[:,i]*x[:,j] for i,j in pairs],axis=1)
for k,v in c['single_bank_sweeps'].items():
 a=np.asarray(v); print('single',k,a.shape,a.ravel()[:14])
print('axis rows, ref + raw:\n',np.round(np.c_[x,y][:16],6))
def fit(a,b):
 w=np.ones(len(a)); beta=None
 for t in range(40):
  sw=np.sqrt(w[:,None]); beta=np.linalg.lstsq(a*sw,b*sw,rcond=None)[0]; r=np.linalg.norm(a@beta-b,axis=1); z=max(.00003,np.median(r)); w=1/(1+(r/(2*z))**2)
 return beta
blocks={'affine':[], 'self2':[x*x], 'self3':[x*x,x**3], 'odd3':[x**3], 'bilinear':[q], 'quadratic':[x*x,q], 'quad+odd3':[x*x,q,x**3], 'linear+sat1':[x**3/(1+x*x)], 'linear+sat3':[x**3/(1+3*x*x)], 'mixed_sat1':[x*x,q,x**3/(1+x*x)], 'mixed_sat3':[x*x,q,x**3/(1+3*x*x)]}
aff=np.c_[o,x]; r=np.linalg.norm(aff@fit(aff,y)-y,axis=1); print('affine residual largest', [(int(i),round(r[i],5)) for i in np.argsort(r)[-15:]])
for name,b in blocks.items():
 a=np.column_stack([o,x]+b); err=np.empty(n); beta=fit(a,y); rr=np.linalg.norm(a@beta-y,axis=1)
 for f in range(5):
  te=np.arange(n)%5==f; err[te]=np.linalg.norm(a[te]@fit(a[~te],y[~te])-y[te],axis=1)
 good=r<1.; print(name,'cv axis/random p50,p90,max',np.round(np.quantile(err[:16],[.5,.9,1]),6),np.round(np.quantile(err[16:][good[16:]],[.5,.9,1]),6),'train',np.round(np.quantile(rr,[.5,.8,.9]),6))
 if name in ('affine','quad+odd3','mixed_sat1'): print(np.round(beta,5))
PY
```
- **Container Exit Code**: `0` (`569 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
000e+00  1.91390e-01 -0.00000e+00]
 [-0.00000e+00 -3.71200e-01  1.31999e+00  1.13100e-01]
 [ 1.18900e-01 -1.00000e-05 -3.82800e-01  7.80000e-01]
 [ 0.00000e+00 -0.00000e+00 -0.00000e+00  0.00000e+00]
 [-0.00000e+00  0.00000e+00  0.00000e+00 -0.00000e+00]
 [-0.00000e+00 -0.00000e+00 -0.00000e+00 -0.00000e+00]
 [-1.00000e-05 -0.00000e+00  0.00000e+00 -0.00000e+00]
 [-1.80400e-01 -0.00000e+00  0.00000e+00  0.00000e+00]
 [-0.00000e+00 -0.00000e+00  0.00000e+00 -0.00000e+00]
 [-0.00000e+00 -0.00000e+00  0.00000e+00 -1.71600e-01]
 [ 0.00000e+00 -2.81600e-01  0.00000e+00 -0.00000e+00]
 [-0.00000e+00  0.00000e+00  0.00000e+00 -0.00000e+00]
 [-0.00000e+00 -0.00000e+00 -2.90400e-01 -0.00000e+00]
 [ 1.47590e-01 -0.00000e+00 -0.00000e+00 -0.00000e+00]
 [ 0.00000e+00  2.30400e-01  0.00000e+00  0.00000e+00]
 [ 0.00000e+00 -0.00000e+00  2.37610e-01  0.00000e+00]
 [ 0.00000e+00  0.00000e+00  0.00000e+00  1.40400e-01]]
linear+sat1 cv axis/random p50,p90,max [ 0.107245 25.872845 25.998802] [0.318278 0.550853 0.773907] train [0.240619 0.444442 0.592916]
linear+sat3 cv axis/random p50,p90,max [ 0.103186 25.872228 26.012355] [0.317466 0.57886  0.861485] train [0.2485   0.449783 0.639696]
mixed_sat1 cv axis/random p50,p90,max [ 0.047933 25.967032 26.023238] [0.073225 0.165717 0.268666] train [0.045432 0.078936 0.14078 ]
[[-7.30300e-02  6.38500e-02  7.54300e-02 -7.31000e-02]
 [ 7.05140e-01  1.46530e-01 -2.53600e-02 -2.40160e-01]
 [-2.29870e-01  1.10473e+00  1.98960e-01 -3.98000e-03]
 [-4.73000e-03 -3.34910e-01  1.12605e+00  1.14070e-01]
 [ 1.17310e-01 -2.42000e-02 -3.92360e-01  6.69540e-01]
 [ 1.29300e-02 -3.08000e-03 -3.82000e-03 -6.93000e-03]
 [ 8.11000e-03  1.18600e-02 -1.58000e-03 -8.80000e-04]
 [-2.22000e-03  9.81000e-03  3.38200e-02  2.45000e-03]
 [ 4.18000e-03 -1.98000e-03  1.33000e-03  2.59000e-02]
 [-1.77510e-01 -5.98000e-03  6.96000e-03 -5.21000e-03]
 [-4.91000e-03  1.59000e-03 -2.24000e-03  7.00000e-05]
 [-7.16000e-03  8.29000e-03  1.08200e-02 -1.76350e-01]
 [ 5.75000e-03 -2.75740e-01  1.06900e-02 -1.56000e-03]
 [ 3.31000e-03 -1.96000e-03 -1.37300e-02 -5.80000e-04]
 [-4.69000e-03 -9.10000e-04 -2.99060e-01  8.10000e-04]
 [ 5.83140e-01  6.75500e-02  5.66100e-02  2.72000e-02]
 [-1.51100e-02  9.03470e-01 -9.67000e-03  8.97000e-03]
 [ 1.00000e-02 -5.66700e-02  9.60920e-01 -3.57000e-03]
 [-6.95000e-03  3.22500e-02 -5.99000e-03  5.67970e-01]]
mixed_sat3 cv axis/random p50,p90,max [ 0.065905 25.968233 26.022185] [0.119723 0.245737 0.388465] train [0.070942 0.133916 0.187985]
```

### Turn 5 — `execute_in_remote_docker` (`54788 ms` LLM)
- **Rationale**: Implement the measured forward law without changing the verified downstream solver. Check held-out inverse accuracy separately on axis and mixed sweeps, then check pulse shape preservation, identity normalization, observation round-trip error, and the composed schema.
- **Alternatives Weighed**: `Fit a polynomial inverse directly` (rejected: The measured forward law is nearly exact, but its inverse need not be polynomial; iterative inversion can be checked against the original readings.)
- **ATMS NOGOOD DECLARED**: Refuted `Quadratic coupling or the tested rational saturation terms adequately describe the rosette sweep.` | Witness: `Maximum clean mixed-sweep held-out error was 0.596 for the quadratic basis and 0.269 for the better tested rational basis, versus 4.6e-5 after using self-cubics.` | Blanket Diagnosis: `Axis curvature and mixed-channel products are distinct effects; rejecting spike rows does not resolve a missing cubic term.` | Sublated Pivot: `Retain affine mixing and pairwise products, add self-cubics, and fit with rowwise robust weights followed by a clean-row refit.`

```bash
cat > /holdout_task/upstream_transducer.py <<'PY'
"""Robust forward calibration and inversion for the four-channel rosette."""
from itertools import combinations

import numpy as np

_PAIRS = tuple(combinations(range(4), 2))


def _features(x):
    cross = [x[:, i] * x[:, j] for i, j in _PAIRS]
    return np.column_stack((np.ones(len(x)), x, x*x, *cross, x**3))


def _fit(ref, raw):
    a = _features(ref)
    w = np.ones(len(ref))
    for _ in range(30):
        sw = np.sqrt(w)[:, None]
        beta = np.linalg.lstsq(a * sw, raw * sw, rcond=None)[0]
        residual = np.linalg.norm(a @ beta - raw, axis=1)
        scale = max(np.median(residual), 1e-10)
        w = 1 / (1 + (residual / (2 * scale))**2)
    clean = residual < 8 * scale
    if np.linalg.matrix_rank(a[clean]) == a.shape[1]:
        beta = np.linalg.lstsq(a[clean], raw[clean], rcond=None)[0]
    return beta


def _jacobian(x, beta):
    # Axes are (sample, raw output, reference input).
    j = np.broadcast_to(beta[1:5].T, (len(x), 4, 4)).copy()
    j += (2*x)[:, None, :] * beta[5:9].T
    for k, (i, l) in enumerate(_PAIRS, 9):
        j[:, :, i] += x[:, l, None] * beta[k]
        j[:, :, l] += x[:, i, None] * beta[k]
    j += (3*x*x)[:, None, :] * beta[15:19].T
    return j


def _invert(raw, beta):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    x = np.linalg.solve(beta[1:5].T, (y - beta[0]).T).T
    for _ in range(30):
        r = _features(x) @ beta - y
        err = np.linalg.norm(r, axis=1)
        active = err > 1e-12 * (1 + np.linalg.norm(y, axis=1))
        if not np.any(active):
            break
        current = x[active]
        step = np.linalg.solve(_jacobian(current, beta), r[active, :, None])[..., 0]
        t = np.ones(len(current))
        target = y[active]
        for _ in range(16):
            trial = current - t[:, None] * step
            improved = np.linalg.norm(_features(trial) @ beta - target, axis=1) < err[active]
            if np.all(improved):
                break
            t[~improved] *= 0.5
        x[active] = np.where(improved[:, None], trial, current)
        if not np.any(improved):
            break
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a copy of the study with observations and sweeps in reference units."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    beta = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    single = sweeps['single_bank_sweeps']
    calibrated = dict(study, y_obs_raw=_invert(study['y_obs_raw'], beta))
    calibrated['sensor_calibration_sweeps'] = dict(
        sweeps,
        raw_rosette=ref.copy(),
        single_bank_sweeps=dict(single, raw_readout=np.asarray(single['ref_signal']).copy()),
    )
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        calibrated['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], beta)) for pulse in pulses
        ]
    return calibrated
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 validate_public_schema.py /holdout_task/batch_solver.py && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from upstream_transducer import _fit,_features,_invert,calibrate_upstream
from batch_solver import solve_batch
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']
n=len(x); ids=np.arange(n); mask=np.max(np.abs(y),axis=1)<5; errors=np.empty(n); inv=np.empty(n)
for f in range(5):
 te=ids%5==f; b=_fit(x[~te],y[~te]); errors[te]=np.linalg.norm(_features(x[te])@b-y[te],axis=1); valid=te&mask; inv[valid]=np.linalg.norm(_invert(y[valid],b)-x[valid],axis=1)
for name,sl in [('axis',ids<16),('mixed',ids>=16)]:
 z=mask&sl; print(name, 'clean rows',z.sum(),'max forward / inverse',errors[z].max(),inv[z].max()); assert inv[z].max()<2e-4
print('gross row indices',ids[~mask].tolist(),'min held-out residual',errors[~mask].min())
b=_fit(x,y); sc=calibrate_upstream(s); z=sc['y_obs_raw']; print('observation inversion max residual',np.max(np.abs(_features(z.reshape(-1,4))@b-s['y_obs_raw'].reshape(-1,4)))); assert np.max(np.abs(_features(z.reshape(-1,4))@b-s['y_obs_raw'].reshape(-1,4)))<1e-9
s2=dict(s,reference_tracer_pulses=[{'y_pulse_raw':s['y_obs_raw'].reshape(2,12,4).copy(),'label':'probe'}]); p=calibrate_upstream(s2)['reference_tracer_pulses'][0]; assert p['label']=='probe' and p['y_pulse_raw'].shape==(2,12,4) and np.allclose(p['y_pulse_raw'].ravel(),z); assert 'reference_tracer_pulses' not in sc and calibrate_upstream(dict(s,reference_tracer_pulses=None))['reference_tracer_pulses'] is None
c2=sc['sensor_calibration_sweeps']; assert np.array_equal(c2['raw_rosette'],x) and not np.shares_memory(c2['raw_rosette'],x); assert np.array_equal(c2['single_bank_sweeps']['raw_readout'],c['single_bank_sweeps']['ref_signal']); assert not np.array_equal(c['raw_rosette'],x)
out=solve_batch(s); print('isolated import: pulse, identity and composition passed; dual gap',out['kkt_certificate']['primal_objective']-out['kkt_certificate']['dual_lower_bound'])
PY
```
- **Container Exit Code**: `0` (`609 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/upstream_transducer.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
axis clean rows 13 max forward / inverse 3.302533527797024e-05 3.67977021721753e-05
mixed clean rows 64 max forward / inverse 5.400767920288682e-05 5.4366393695667607e-05
gross row indices [13, 14, 15] min held-out residual 25.963240218253127
observation inversion max residual 8.41972325194007e-13
isolated import: pulse, identity and composition passed; dual gap 8.673617379884035e-17

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c2e0422597c74b16 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c2e04225 (directory).
```

### Turn 6 — `submit_completed_task` (`13604 ms` LLM)
- **Rationale**: The verified bundle passes the public schema and isolated composition checks. Held-out inverse errors are below 2e-4 on both sweep slices, all three gross spikes are separated from clean rows, and observation round-trip error is 8.42e-13. The downstream solver remains unchanged.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:199', '/holdout_task/upstream_transducer.py': 'FILE:3020'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented robust four-channel calibration in `upstream_transducer.py` and verified its composition with `batch_solver.py`. The public schema passed. Maximum held-out inverse error was 5.44e-5 across clean sweep rows. Pulse handling and identity normalization passed isolated tests.`

### Turn 7 — `execute_in_remote_docker` (`33287 ms` LLM)
- **Rationale**: Random folds can hide geometry-dependent calibration errors. Hold out every axis row and, separately, the largest mixed excitations; reconsider the basis if either slice loses inverse accuracy or conditioning. Squared pairwise differences are already in the quadratic span. The measured task is static regression on one study, so relay modes, stochastic drift, and within-batch classification metrics do not apply.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit, _features, _invert, _jacobian, calibrate_upstream
from batch_solver import solve_batch
with open('public_batch.pkl', 'rb') as f:
    batch = pickle.load(f)
for item in batch:
    s = item['study']; c = s['sensor_calibration_sweeps']; x, y = map(np.asarray, (c['ref_rosette'], c['raw_rosette']))
    axis = np.count_nonzero(x, axis=1) == 1
    clean = np.max(np.abs(y), axis=1) < 5  # Evaluation mask only; no training rows prefiltered.
    radius = np.max(np.abs(x), axis=1)
    a = _features(x)
    diff = np.column_stack([(x[:, i]-x[:, j])**2 for i in range(4) for j in range(i+1,4)])
    assert np.max(np.abs(diff-a[:,:15]@np.linalg.lstsq(a[:,:15],diff,rcond=None)[0])) < 1e-12
    for name, hold in [('all axis',axis), ('outer mixed quartile',(~axis)&(radius>=np.quantile(radius[~axis],.75)))]:
        test = hold & clean
        b = _fit(x[~hold],y[~hold]); inv = _invert(y[test],b)
        error = np.max(np.linalg.norm(inv-x[test],axis=1))
        fwd = np.max(np.linalg.norm(a[test]@b-y[test],axis=1))
        sigma_min = np.linalg.svd(_jacobian(inv,b),compute_uv=False)[:,-1].min()
        gains = []
        for alpha in (.25,1.,3.):
            # Test additional rational curvature rather than merely replacing the cubic.
            d = np.column_stack((a,x**3/(1+alpha*x*x)))
            dt, yt = d[~hold], y[~hold]; w = np.ones(len(dt))
            for _ in range(30):
                sw = np.sqrt(w)[:,None]; p = np.linalg.lstsq(dt*sw,yt*sw,rcond=None)[0]
                r = np.linalg.norm(dt@p-yt,axis=1)
                w = 1/(1+(r/(2*max(np.median(r),1e-10)))**2)
            gains.append(np.max(np.linalg.norm(d[test]@p-y[test],axis=1)))
        print(name, 'clean rows', test.sum(), 'max forward', fwd, 'max inverse', error, 'min Jacobian singular value', sigma_min, 'rational augmentation forward maxima', gains)
        assert error < 2e-4 and sigma_min > .2, 'Grouped holdout exposed a calibration or conditioning bottleneck'
    calibrated = calibrate_upstream(s)
    b = _fit(x,y); z = calibrated['y_obs_raw'].reshape(-1,4)
    assert np.max(np.abs(_features(z)@b - s['y_obs_raw'].reshape(-1,4))) < 1e-9
    assert np.array_equal(calibrated['sensor_calibration_sweeps']['raw_rosette'], x)
    cert = solve_batch(s)['kkt_certificate']; gap = cert['primal_objective']-cert['dual_lower_bound']
    print('isolated composed dual gap',gap)
    assert abs(gap) < 1e-9
PY
```
- **Container Exit Code**: `0` (`470 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
all axis clean rows 13 max forward 3.4842705928385323e-05 max inverse 4.0350588336801706e-05 min Jacobian singular value 0.5674160360116061 rational augmentation forward maxima [3.396969409477881e-05, 3.5216111460294576e-05, 3.713419774873463e-05]
outer mixed quartile clean rows 16 max forward 5.457742391800326e-05 max inverse 4.672979705888332e-05 min Jacobian singular value 0.5128108442418515 rational augmentation forward maxima [8.876394544259702e-05, 5.454441096564611e-05, 4.4104896190026035e-05]
isolated composed dual gap 8.673617379884035e-17

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c2e0422597c74b16 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c2e04225 (directory).
```

### Turn 8 — `submit_completed_task` (`16614 ms` LLM)
- **Rationale**: The saved bundle passed isolated execution and grouped holdouts on real measurements. The minimum Jacobian singular value across grouped holdouts was 0.513, and the composed downstream dual gap was 8.67e-17. No remaining calibration bottleneck warrants an artifact change.
- **Alternatives Weighed**: `Add rational curvature terms to the cubic model` (rejected: The held-out forward-error differences were at the sweep noise scale and did not establish an inverse-accuracy or conditioning problem.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:199', '/holdout_task/upstream_transducer.py': 'FILE:3020'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented robust four-channel rosette calibration and verified it with the existing batch solver. The public schema passed. Maximum held-out inverse error was 5.44e-5 across clean sweeps and 4.67e-5 on the grouped edge-of-range holdout. Pulse handling and identity normalization also passed.`

### Turn 9 — `execute_in_remote_docker` (`33930 ms` LLM)
- **Rationale**: Hold out whole sign sectors to detect missing cross-cubic interactions that interleaved folds could hide. Reconsider the basis if the fuller model fixes a meaningful error, or reconsider the inverse if the Jacobian is poorly conditioned at actual observations. The public data contain one regression study and no relay or classification outcomes.
- **Alternatives Weighed**: `Repeat the rational and edge-of-range sweep audit` (rejected: Those checks already passed; they do not test whether the model generalizes to held-out combinations of channel signs.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import itertools, pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _features, _fit, _invert, _jacobian, calibrate_upstream
from batch_solver import solve_batch

def robust(A, y):
    w = np.ones(len(y))
    for _ in range(30):
        sw = np.sqrt(w)[:, None]
        b = np.linalg.lstsq(A * sw, y * sw, rcond=None)[0]
        e = np.linalg.norm(A @ b - y, axis=1)
        scale = max(np.median(e), 1e-10)
        w = 1 / (1 + (e / (2 * scale))**2)
    ok = e < 8 * scale
    if np.linalg.matrix_rank(A[ok]) == A.shape[1]:
        b = np.linalg.lstsq(A[ok], y[ok], rcond=None)[0]
    return b

with open('public_batch.pkl', 'rb') as f:
    batch = pickle.load(f)
for item in batch:
    study = item['study']; sweep = study['sensor_calibration_sweeps']
    x = np.asarray(sweep['ref_rosette']); y = np.asarray(sweep['raw_rosette'])
    clean = np.max(np.abs(y), axis=1) < 5  # Evaluation only; training includes EMI rows.
    mixed = np.count_nonzero(x, axis=1) > 1
    code = (x > 0) @ (2**np.arange(4))
    A = _features(x)
    cross_cubic = [x[:, i]**2 * x[:, j] for i in range(4) for j in range(4) if i != j]
    triple = [np.prod(x[:, ids], axis=1) for ids in itertools.combinations(range(4), 3)]
    B = np.column_stack([A, *cross_cubic, *triple])
    records = []
    for k in np.unique(code[mixed]):
        test = mixed & (code == k)
        train = ~test
        valid = test & clean
        if not valid.any():
            continue
        b = _fit(x[train], y[train]); c = robust(B[train], y[train])
        base_fw = np.max(np.linalg.norm(A[valid] @ b - y[valid], axis=1))
        full_fw = np.max(np.linalg.norm(B[valid] @ c - y[valid], axis=1))
        inverse = np.max(np.linalg.norm(_invert(y[valid], b) - x[valid], axis=1))
        records.append((inverse, k, int(valid.sum()), base_fw, full_fw))
    print('Sign-sector holdouts [inverse, sector, n, base forward, full cubic forward]:', sorted(records, reverse=True))
    assert max(r[0] for r in records) < 2e-4
    b = _fit(x, y); calibrated = calibrate_upstream(study)
    obs = calibrated['y_obs_raw'].reshape(-1, 4)
    jac_min = np.linalg.svd(_jacobian(obs, b), compute_uv=False)[:, -1].min()
    roundtrip = np.max(np.abs(_features(obs) @ b - np.asarray(study['y_obs_raw']).reshape(-1, 4)))
    print('Actual observation minimum Jacobian singular value and max round-trip:', jac_min, roundtrip)
    assert jac_min > .2 and roundtrip < 1e-9
    cert = solve_batch(study)['kkt_certificate']
    assert abs(cert['primal_objective'] - cert['dual_lower_bound']) < 1e-9
PY
```
- **Container Exit Code**: `0` (`620 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Sign-sector holdouts [inverse, sector, n, base forward, full cubic forward]: [(5.150238545620386e-05, 5, 4, 5.372906596550681e-05, 7.206901741770643e-05), (3.914581320660211e-05, 7, 9, 3.432516948519702e-05, 0.00028636303417719843), (3.902937491481392e-05, 15, 4, 3.817375730421453e-05, 3.417076305797961e-05), (3.336017527391801e-05, 10, 5, 5.214652844510458e-05, 0.0003951969961382915), (3.1343624741202904e-05, 13, 6, 4.18909362389846e-05, 6.873385020173174e-05), (2.9900534724377958e-05, 9, 5, 4.693155887963878e-05, 6.321066136706468e-05), (2.979214006499282e-05, 3, 7, 3.0111417535716058e-05, 8.29471373723873e-05), (2.9636254810251413e-05, 6, 5, 2.7642550435685293e-05, 0.00010951112803202343), (2.5468165611921242e-05, 1, 5, 3.774653741609121e-05, 0.00010671694451461531), (2.345544114621254e-05, 11, 5, 2.3830484107487615e-05, 6.069525661772511e-05), (2.147289335737322e-05, 4, 1, 2.372826841038785e-05, 1.9931561750249e-05), (1.6832209718267188e-05, 8, 2, 2.3269332829251376e-05, 3.765060447631115e-05), (1.6467020751332107e-05, 14, 2, 1.7806407019756022e-05, 5.454009796690406e-05), (1.53143095168823e-05, 12, 2, 1.9236873277161047e-05, 4.6986530619185655e-05), (1.424346230105865e-05, 2, 1, 2.1887937174289682e-05, 3.641460868068295e-05), (1.1891491183793485e-05, 0, 1, 1.84550868968097e-05, 2.6805649606160874e-05)]
Actual observation minimum Jacobian singular value and max round-trip: 0.6982678364460857 8.41972325194007e-13

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c2e0422597c74b16 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c2e04225 (directory).
```

### Turn 10 — `submit_completed_task` (`16578 ms` LLM)
- **Rationale**: The verified bundle passed isolated execution and grouped holdouts by axis, excitation range, and sign sector. On actual observations, the minimum Jacobian singular value was 0.698 and the maximum round-trip residual was 8.42e-13. The stress tests found no bottleneck that warrants changing the model.
- **Alternatives Weighed**: `Add mixed cubic and three-channel terms` (rejected: The extra terms increased worst sign-sector forward error from 5.37e-5 to 3.95e-4.)
- **Submitted Final Answer / Summary**: `Implemented robust four-channel rosette calibration in `upstream_transducer.py` and verified it with the existing `batch_solver.py`. The public schema, pulse handling, and identity normalization passed. The maximum inverse error was 5.44e-5 on clean held-out sweeps and 5.15e-5 when holding out entire sign sectors.`

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
