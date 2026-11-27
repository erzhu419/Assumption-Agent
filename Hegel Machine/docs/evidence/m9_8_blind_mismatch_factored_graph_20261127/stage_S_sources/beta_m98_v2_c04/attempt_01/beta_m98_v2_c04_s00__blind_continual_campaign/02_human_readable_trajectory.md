# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m98_v2_c04_s00__blind_continual_campaign`

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
| **Total Tokens (Prompt / Cached / Out)** | `208,128` (`200,017` / `100,224` / `8,111`) |
| **Total Episode Wall Time (s)** | `263.05s` |

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

### Turn 1 — `execute_in_remote_docker` (`22678 ms` LLM)
- **Rationale**: Inspect compact schema, array shapes, and archive metadata to catch campaign-specific assumptions before constructing the forward model and testing its fit.
- **Alternatives Weighed**: `Implement the open-wall operator immediately` (rejected: Campaign evidence may identify a different boundary operator; the public study and archive can settle that before implementation.)

```bash
cd /holdout_task && cat public_asset_index.json validate_public_schema.py && printf '\n--- archive ---\n' && if [ -d /tmp/hegel_continual_archive ]; then find /tmp/hegel_continual_archive -maxdepth 4 -type f -printf '%p %s bytes\n' | head -80; fi; printf '\n--- public batch structure ---\n'; cd /holdout_task && python3 - <<'PY'
import pickle, numpy as np
with open('public_batch.pkl','rb') as f: a=pickle.load(f)
print(type(a).__name__,len(a))
def describe(x, p='', depth=0):
 if isinstance(x,dict) and depth < 4:
  for k,v in x.items(): describe(v,p+'/'+str(k),depth+1)
 elif isinstance(x,np.ndarray):
  print(p,x.shape,x.dtype,x.ravel()[:10])
 elif isinstance(x,(list,tuple)):
  print(p,type(x).__name__,len(x));
  if x and depth<4: describe(x[0],p+'/0', depth+1)
 else: print(p,repr(x)[:240])
describe(a[0])
PY
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

### Turn 2 — `execute_in_remote_docker` (`22349 ms` LLM)
- **Rationale**: The previous log was truncated. Recover the archive listing and validator contract, then quantify the affine calibration error to decide whether a richer calibration model is needed.
- **Alternatives Weighed**: `Fit four independent gain and offset curves immediately` (rejected: An affine cross-channel fit and the single-bank sweeps can reveal whether independent calibration would miss rosette coupling.)

```bash
cd /holdout_task && cat validate_public_schema.py; printf '\n--- archive ---\n'; if [ -d /tmp/hegel_continual_archive ]; then du -ah /tmp/hegel_continual_archive | sort -h | tail -40; else echo absent; fi; cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
b=pickle.load(open('public_batch.pkl','rb')); print('studies',len(b)); s=b[0]['study']; print('keys',list(s)); print('modes',np.shape(s['source_modes'])); print('sensors',s['sensor_ij']); print('banks',s['row_bank_id'].reshape(-1,4)); c=s['sensor_calibration_sweeps']; X,Y=c['ref_rosette'],c['raw_rosette']; print('ref range',X.min(0),X.max(0)); print('raw range',Y.min(0),Y.max(0)); D=np.c_[np.ones(len(X)),X]; C=np.linalg.lstsq(D,Y,rcond=None)[0]; print('affine coefficients',C,'rmse',np.sqrt(np.mean((D@C-Y)**2,0))); print('single bank');
for k,v in c['single_bank_sweeps'].items(): print(k,np.shape(v),v[:8]);
print('telemetry first rows',s.get('aux_manifold_telemetry',[])[:6]); print('shifts', s['mu_shifts']); print('diagnostics flags',[i.get('campaign_diagnostics') is not None for i in b]);
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | head -45
```
- **Container Exit Code**: `0` (`500 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ecution_available": true,
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
          "function": "fit_ridge_nuisance",
          "signature": "fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray]",
          "doc_summary": "Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and return coefficients (p,) or (p, q), same-shape residuals, and an (r, p) Gram whitener."
        }
      ],
      "index_cards": [
        "autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance",
        "autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_ridge_profiled_matching",
      "reference_source_txt": null
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`57024 ms` LLM)
- **Rationale**: Implement the specified fallback and exact constrained objective. Check the schema, calibration generalization, and residuals by shift to catch shape, calibration, or forward-model mismatches before independent stress testing.
- **Alternatives Weighed**: `Infer an alternate wall law from the blind observations` (rejected: No diagnostic pulses or archive are available, so the task specifies the open-wall Robin fallback.); `Use the ridge-profile primitive` (rejected: One direct solve applies the specified projection to both the design and observations without computing an unused whitener.)

```bash
cd /holdout_task && cat > batch_solver.py <<'PY'
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
            low = np.argmin(np.where(state < 0, g, np.inf))
            high = np.argmax(np.where(state > 0, g, -np.inf))
            if g[high] <= g[low] + tol:
                return w, float((g[high] + g[low]) / 2)
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
PY
python3 /holdout_task/validate_public_schema.py
python3 -I - <<'PY'
import runpy, pickle, numpy as np
m=runpy.run_path('/holdout_task/batch_solver.py')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
r=m['solve_batch'](s); c=r['kkt_certificate']; w=r['weights']
print('weights',w,'budget error',abs(w.sum()-s['total_budget']),'gap',c['primal_objective']-c['dual_lower_bound'])
S,U=m['_forward'](s); y=m['_calibrate'](s); Z=s['Z_nuisance']; d=y-S@w; e=d-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@d)
print('profiled RMSE by shift',np.sqrt(np.mean(e.reshape(len(s['mu_shifts']),-1)**2, axis=1)))
a=s['sensor_calibration_sweeps']; X=a['ref_rosette']; Y=a['raw_rosette']; D=np.c_[np.ones(len(X)),X]; errors=[]
for k in range(5):
    t=np.arange(len(X))%5==k
    coeff=np.linalg.lstsq(D[~t],Y[~t],rcond=None)[0]
    errors.append(np.sqrt(np.mean((D[t]@coeff-Y[t])**2,axis=0)))
print('worst calibration fold/channel RMSE',np.max(errors))
PY
```
- **Container Exit Code**: `0` (`628 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/batch_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
weights [0.12298362 0.37960467 0.19047567 0.31020724 0.05845026 0.
 0.07599304 0.2622855 ] budget error 0.0 gap 1.3357370765021415e-16
profiled RMSE by shift [0.00990176 0.00884292 0.00872757]
worst calibration fold/channel RMSE 1.370897317614683e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8d7a725cc7872523 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8d7a725c (directory).
```

### Turn 4 — `execute_in_remote_docker` (`37829 ms` LLM)
- **Rationale**: The schema passes but would not catch a transposed forward field or incorrect multipliers. Compare against an independently assembled dense operator and stress the certificate at mixed and fully active bounds; correct any discrepancy before submitting.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy, numpy as np
from scipy.linalg import solve
m=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
nx,ny=s['nx'],s['ny']; modes=s['source_modes']; n=len(modes)
def axis(k,v,h,r):
    a=np.zeros((len(k),len(k)))
    for i in range(len(k)):
        if i in (0,len(k)-1):
            a[i,i]=k[i]/h**2+r/h
            a[i,1 if i==0 else i-1]=-k[i]/h**2
        else:
            km=(k[i-1]+k[i])/2; kp=(k[i]+k[i+1])/2
            a[i,i-1]=-km/h**2-max(v[i],0)/h
            a[i,i+1]=-kp/h**2-max(-v[i],0)/h
            a[i,i]=(kp+km)/h**2+abs(v[i])/h
    return a
ax=axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x']); ay=axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'])
L=np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
rhs=np.column_stack([v.ravel(order='F') for v in modes]); blocks=[]
for shift in s['mu_shifts']:
    U=solve(L+shift*np.eye(nx*ny),rhs).reshape(nx,ny,n,order='F')
    blocks.append(np.array([U[i,j] for i,j in s['sensor_ij']]))
S=np.vstack(blocks); Z=s['Z_nuisance']; P=np.eye(len(S))-Z@solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T); X=P@S
sf,uf=m['_forward'](s); print('dense design and terminal basis max errors',np.max(abs(S-sf)),np.max(abs(U.transpose(2,0,1)-uf)))
np.testing.assert_allclose(S,sf,atol=1e-12); np.testing.assert_allclose(U.transpose(2,0,1),uf,atol=1e-12)
cal=s['sensor_calibration_sweeps']; ref=cal['ref_rosette']; raw=cal['raw_rosette']; coef=np.linalg.lstsq(np.c_[np.ones(len(ref)),ref],raw,rcond=None)[0]; rng=np.random.default_rng(85)
max_stationarity=max_gap=max_budget=max_field=max_obj=0.; patterns=set()
for i in range(101):
    t=s.copy(); observed=m['_calibrate'](s)
    if i:
        cap=s['u_box']; t['total_budget']=float([0.,cap,2*cap,3*cap,7*cap,n*cap,rng.uniform(0,n*cap)][i%7]); observed=S@rng.uniform(0,cap,n)+Z@rng.normal(size=Z.shape[1])+rng.normal(0,0.5,len(S)); t['y_obs_raw']=(observed.reshape(-1,4)@coef[1:]+coef[0]).ravel()
    out=m['solve_batch'](t); w=out['weights']; c=out['kkt_certificate']; nu=c['nu_sum']; lo=np.array(c['lam_lower']); hi=np.array(c['lam_upper']); cap=t['u_box']; B=t['total_budget']; e=X@w-P@observed; g=X.T@e+t['source_ridge']*w; J=.5*(e@e+t['source_ridge']*(w@w)); D=J-g@w+nu*B+cap*np.minimum(g-nu,0).sum()
    np.testing.assert_array_equal(w,c['weights']); assert np.min(w)>=-1e-12 and np.max(w)<=cap+1e-12 and min(lo.min(),hi.min())>=0
    max_budget=max(max_budget,abs(w.sum()-B)); max_gap=max(max_gap,abs(J-D),np.max(abs(lo*w)),np.max(abs(hi*(cap-w)))); max_stationarity=max(max_stationarity,np.max(abs(g-nu-lo+hi))); max_obj=max(max_obj,abs(J-c['primal_objective']),abs(D-c['dual_lower_bound'])); max_field=max(max_field,np.max(abs(out['terminal_field']-U@w))); patterns.add((np.count_nonzero(w<1e-12),np.count_nonzero(w>cap-1e-12)))
print('cases',101,'bound-count patterns',len(patterns),'worst budget, gap/complementarity, stationarity, objective/certificate, field errors:',max_budget,max_gap,max_stationarity,max_obj,max_field)
assert max(max_budget,max_gap,max_stationarity,max_obj,max_field)<1e-10
PY
```
- **Container Exit Code**: `0` (`798 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
dense design and terminal basis max errors 4.996003610813204e-16 2.636779683484747e-16
cases 101 bound-count patterns 28 worst budget, gap/complementarity, stationarity, objective/certificate, field errors: 8.881784197001252e-16 3.552713678800501e-15 4.440892098500626e-15 7.105427357601002e-15 5.551115123125783e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8d7a725cc7872523 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8d7a725c (directory).
```

### Turn 5 — `submit_completed_task` (`13742 ms` LLM)
- **Rationale**: No tracer pulses or archive were available, so the specified Robin fallback applies. An isolated run matched the independent dense forward model and verified feasibility, certificate, and terminal field across 28 bound-count patterns.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`13` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5174'}): Do NOT lock prematurely with `13` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py` with sweep-fitted sensor calibration, the specified open-wall Robin fallback, and a KKT-certified weight solve. The public schema validator passed, and independent numerical checks passed across 101 cases.`

### Turn 6 — `execute_in_remote_docker` (`32526 ms` LLM)
- **Rationale**: Check whether cross-channel nonlinear terms materially improve calibration on held-out sweeps, verify axis row balance, and refit the nuisance projection within each transport holdout. A wall-specific residual would flag a boundary limitation but would not identify a replacement without tracer diagnostics.
- **Alternatives Weighed**: `Apply relay-mode or cohort classification checks` (rejected: The study has neither relay states nor class labels. Held-out wall and shift predictions test whether nuisance fitting masks transport errors.)

```bash
cat > /tmp/audit_m98_rosette.py <<'PY'
import pickle
import runpy
import numpy as np

m = runpy.run_path('/holdout_task/batch_solver.py')
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
aff = np.c_[np.ones(len(x)), x]
pairs = [(i,j) for i in range(4) for j in range(i+1,4)]
quad = np.column_stack([aff, x*x] + [x[:,i]*x[:,j] for i,j in pairs])
cubic = np.column_stack([quad, x*x*x])
rational = np.column_stack([aff, x/(1+x*x)] + [(x[:,i]+sign*x[:,j])/(1+(x[:,i]-sign*x[:,j])**2) for i,j in pairs for sign in (-1,1)])
for name, d in [('independent affine',aff),('coupled affine',aff),('quadratic',quad),('cubic',cubic),('rational',rational)]:
    e = np.empty_like(y)
    for k in range(5):
        test = np.arange(len(x)) % 5 == k
        for j in range(4):
            dj = d[:,[0,j+1]] if name == 'independent affine' else d
            e[test,j] = dj[test] @ np.linalg.lstsq(dj[~test],y[~test,j],rcond=None)[0] - y[test,j]
    worst = max(np.sqrt(np.mean(e[np.arange(len(x)) % 5 == k]**2, axis=0)).max() for k in range(5))
    print(name, 'OOF RMSE', np.sqrt(np.mean(e*e)), 'worst fold/channel', worst)
for a in 'xy':
    n=s['n'+a]; h=s['L'+a]/(n-1)
    op=m['_axis'](s['kappa_'+a],s['vel_'+a],s['L'+a],s['robin_'+a])
    expected=np.zeros(n); expected[[0,-1]]=s['robin_'+a]/h
    err=np.max(abs(op@np.ones(n)-expected))
    print(a,'row balance max error',err)
    assert err < 1e-11
S,_=m['_forward'](s); y=m['_calibrate'](s); Z=s['Z_nuisance']; sensor=np.asarray(s['sensor_ij']); shifts=s['mu_shifts']; wall=np.any((sensor==0)|(sensor==np.array([s['nx']-1,s['ny']-1])),axis=1)
wall=np.tile(wall,len(shifts)); shift_id=np.repeat(np.arange(len(shifts)),len(sensor))
for name,test in [(f'shift {mu}',shift_id==k) for k,mu in enumerate(shifts)]+[('wall',wall),('interior',~wall)]:
    A,z,target=S[~test],Z[~test],y[~test]
    p=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
    X=A-z@(p@A); b=target-z@(p@target)
    w,_=m['_box_budget'](X.T@X+s['source_ridge']*np.eye(S.shape[1]),-X.T@b,s['u_box'],s['total_budget'])
    gamma=p@(target-A@w)
    e=S[test]@w+Z[test]@gamma-y[test]
    print(name,'held-out RMSE',np.sqrt(np.mean(e*e)))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_m98_rosette.py && python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`568 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
independent affine OOF RMSE 1.0429643145258265e-05 worst fold/channel 1.3340597446355298e-05
coupled affine OOF RMSE 1.0683638775120918e-05 worst fold/channel 1.370897317599326e-05
quadratic OOF RMSE 1.1604680522542046e-05 worst fold/channel 1.674675712319921e-05
cubic OOF RMSE 1.2399814839384537e-05 worst fold/channel 1.5752850903547995e-05
rational OOF RMSE 1.3404969494322942e-05 worst fold/channel 1.693034108540095e-05
x row balance max error 3.552713678800501e-15
y row balance max error 1.7763568394002505e-15
shift 0.15 held-out RMSE 0.011911824832209818
shift 0.55 held-out RMSE 0.010138839102095554
shift 1.1 held-out RMSE 0.009613056669473569
wall held-out RMSE 0.008143996640555312
interior held-out RMSE 0.01754110918470291
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8d7a725cc7872523 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8d7a725c (directory).
```

### Turn 7 — `submit_completed_task` (`15459 ms` LLM)
- **Rationale**: The real-data audit found no improvement that justified changing the verified candidate. The independent dense reference agreed with the forward operator and terminal field, and stress tests verified feasibility and the KKT certificate across 28 bound-count patterns.
- **Alternatives Weighed**: `Replace coupled affine calibration` (rejected: Nonlinear models worsened held-out error; the independent affine fit improved worst-fold RMSE by only 3.68e-7 while dropping cross-channel correction.); `Replace the Robin fallback` (rejected: Held-out wall error was lower than held-out interior error, and no tracer pulses or archive identified an alternative boundary law.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:5174'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/batch_solver.py` with sweep-fitted rosette calibration, the specified Robin fallback, and a KKT-certified weight solve. The public schema passed in isolated Python, and independent numerical checks passed across 101 cases. Worst held-out transport RMSE was 0.01754 on interior sensors.`

### Turn 8 — `execute_in_remote_docker` (`26371 ms` LLM)
- **Rationale**: Hold each rosette location out at every shift and fit nuisance coefficients only on the remaining locations. Bank-level errors can expose a failure hidden by pooled RMSE; the single-bank sweeps and calibration range can indicate whether calibration is the cause.
- **Alternatives Weighed**: `Repeat the nonlinear calibration and KKT stress tests` (rejected: Those checks already passed; the remaining question is whether pooled transport error masks a bank-specific failure.)

```bash
cat > /tmp/audit_m98_banks.py <<'PY'
import pickle, runpy
import numpy as np
m = runpy.run_path('/holdout_task/batch_solver.py')
item = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s = item['study']; c = s['sensor_calibration_sweeps']; x,y = c['ref_rosette'],c['raw_rosette']
D = np.c_[np.ones(len(x)),x]; C = np.linalg.lstsq(D,y,rcond=None)[0]
print('affine intercept', C[0], 'mixing perturbation from diagonal', C[1:]-np.diag(np.diag(C[1:])))
for k,v in c['single_bank_sweeps'].items():
    a = np.asarray(v); print('single-bank',k,'shape',a.shape,'first rows',a[:2])
obs = m['_calibrate'](s).reshape(-1,4)
print('observations outside calibration reference range by channel', ((obs<x.min(0))|(obs>x.max(0))).sum(0), 'of',len(obs))
S,_ = m['_forward'](s); Z=s['Z_nuisance']; target=obs.ravel(); n=len(s['sensor_ij']); B=s['total_budget']; cap=s['u_box']; ridge=s['source_ridge']
banks=np.asarray(s['row_bank_id']).ravel(); groups=np.tile(np.arange(n)//4,len(s['mu_shifts'])); errs={}
for profile in (False,True):
    e=np.empty_like(target)
    for group in np.unique(groups):
        test=groups==group; A=S[~test]; b=target[~test]; z=Z[~test]
        if profile:
            p=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T); X=A-z@(p@A); q=b-z@(p@b)
        else:
            X,q=A,b
        w,_=m['_box_budget'](X.T@X+ridge*np.eye(X.shape[1]),-X.T@q,cap,B)
        pred=S[test]@w
        if profile: pred += Z[test]@(p@(b-A@w))
        e[test]=pred-target[test]
    errs[profile]=e
    print('profile',profile,'pooled RMSE',np.sqrt(np.mean(e**2)),'bank RMSE',[(int(k),float(np.sqrt(np.mean(e[banks==k]**2)))) for k in np.unique(banks)])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_m98_banks.py && python3 -I /holdout_task/validate_public_schema.py
```
- **Container Exit Code**: `0` (`551 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
affine intercept [-6.46202984e-08 -6.11832451e-08 -2.58406443e-07  2.21154525e-06] mixing perturbation from diagonal [[ 0.00000000e+00 -2.47836391e-06 -8.26644934e-07 -1.76334543e-06]
 [-2.03206309e-07  0.00000000e+00  1.96456989e-06  8.62478154e-07]
 [-3.16894137e-06 -3.74839037e-06  0.00000000e+00 -6.13086830e-07]
 [ 2.00011943e-06  1.58920966e-06  2.55192390e-06  0.00000000e+00]]
single-bank bank_id shape (24,) first rows [0 0]
single-bank ref_signal shape (24,) first rows [-1.1 -0.6]
single-bank raw_readout shape (24,) first rows [-1.10000562 -0.59998306]
observations outside calibration reference range by channel [0 0 0 0] of 24
profile False pooled RMSE 0.3289795904643022 bank RMSE [(0, 0.27628209150174754), (1, 0.3294749566057618), (2, 0.38032281896333703), (3, 0.3215265090611847)]
profile True pooled RMSE 0.009958781579116414 bank RMSE [(0, 0.011264179127558033), (1, 0.011557843291523578), (2, 0.007348061663907071), (3, 0.009069169696090283)]
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=8d7a725cc7872523 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_8d7a725c (directory).
```

### Turn 9 — `submit_completed_task` (`13684 ms` LLM)
- **Rationale**: The real-data bank holdout confirms that the specified nuisance projection improves both pooled and within-bank prediction. The unchanged artifact passed isolated schema validation and independent dense-reference and KKT stress checks.
- **Alternatives Weighed**: `Omit the nuisance projection` (rejected: Holding out each rosette location across all shifts, the specified projection improved prediction in every sensor bank.); `Use nonlinear rosette calibration` (rejected: The tested nonlinear fits worsened held-out calibration error, and public observations were within the sweep range in all channels.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/batch_solver.py`. The public schema passed in isolated Python, and the forward model, terminal field, and KKT certificate passed 101 numerical checks. Nuisance projection reduced the worst held-out sensor-bank RMSE from 0.3803 to 0.01156.`

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
