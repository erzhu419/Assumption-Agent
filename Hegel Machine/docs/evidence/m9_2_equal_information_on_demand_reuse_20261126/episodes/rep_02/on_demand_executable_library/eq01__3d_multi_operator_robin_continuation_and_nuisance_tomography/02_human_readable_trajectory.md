# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `eq01__3d_multi_operator_robin_continuation_and_nuisance_tomography`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M92-EqualInfo-4` |
| **Domain** | `3D Parametric Elliptic PDE Continuation, Reusable Preconditioning & Profiled Nuisance Tomography` |
| **Wave / Wormhole Family** | `m92_equal_info` / `fam_multiscale_pde_and_preconditioned_operators` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`PASS (1.0)`** |
| **Failure / Outcome Taxonomy** | `PASSED_ALL_CHECKS` |
| **Turns Used / Max Turns** | `10 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `231,694` (`223,053` / `71,680` / `8,641`) |
| **Total Episode Wall Time (s)** | `694.36s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`17311 ms` LLM)
- **Rationale**: Inspect the validator and public study summaries to establish the test interface, coefficient ranges, and available solvers before choosing a reusable preconditioner.
- **Alternatives Weighed**: `Factor each continuation state separately` (rejected: The contract reports that this exceeds the two-study runtime budget.)

```bash
ls -l /holdout_task; sed -n '1,240p' /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'SPD AMG CG sparse preconditioning variable diffusion ridge nuisance profiling'; python3 - <<'PY'
import pickle, importlib.util, numpy as np, scipy
p='/holdout_task/public_eq01_studies.pkl'
x=pickle.load(open(p,'rb')); print('versions',np.__version__,scipy.__version__,'pyamg',importlib.util.find_spec('pyamg')); print(type(x),list(x) if isinstance(x,dict) else len(x))
def walk(o,d=0):
 if d>3: return
 if isinstance(o,dict):
  for k,v in o.items():
   if isinstance(v,np.ndarray): print(k,v.shape,v.dtype, 'range',np.min(v),np.max(v))
   elif isinstance(v,(dict,list,tuple)): print(k,type(v),len(v)); walk(v,d+1)
   else: print(k,v)
 elif isinstance(o,(list,tuple)):
  for i,v in enumerate(o[:3]): print('entry',i); walk(v,d+1)
walk(x)
PY
```
- **Container Exit Code**: `1` (`792 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
on: float=0.76, coarse_size: int=64, seed: int=12) -> LinearOperator | Map a real sparse SPD matrix of shape (n, n) to a symmetric (n, n) V-cycle app | REQ: grid op assumes isotropic scalar coeff>0 + zero Dirichlet ghosts; CG/AMG need square SPD | --describe autogenous_prim_conservative_spd_solve.galerkin_preconditioner | STATUS: APPLICABLE_SUBROUTINE(requires_assembled_csr_spd_matrix)
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes | STATUS: APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)
autogenous_prim_conservative_spd_solve@05cb4387 | conservative_grid_operator(cell_coeff: np.ndarray, *, spacing: float | np.ndarray=1.0) -> sparse.csr_matrix | Map positive cell coefficients of shape (n1,  | REQ: grid op assumes isotropic scalar coeff>0 + zero Dirichlet ghosts; CG/AMG need square SPD | --describe autogenous_prim_conservative_spd_solve.conservative_grid_operator | STATUS: MISMATCHED_BOUNDARY_OR_TENSOR(zero_ghost_scalar_dirichlet_only)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
```
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task': No such file or directory
sed: can't read /holdout_task/validate_public_schema.py: No such file or directory
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
FileNotFoundError: [Errno 2] No such file or directory: '/holdout_task/public_eq01_studies.pkl'
```

### Turn 2 — `execute_in_remote_docker` (`16856 ms` LLM)
- **Rationale**: Locate the public fixtures and inspect the reusable primitives for compatibility with the exact Robin operator. If the fixtures remain absent, build to the stated contract and verify on independently assembled synthetic references.
- **Alternatives Weighed**: `Validate immediately on generated inputs` (rejected: First locate any mounted public study so synthetic stress tests supplement rather than replace available public validation.)

```bash
pwd; ls -la / /pretrain_task; find /pretrain_task /app /workspace /tmp -maxdepth 3 \( -iname '*eq01*' -o -iname '*robin*' -o -name 'validate_public_schema.py' \) -print 2>/dev/null; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_conservative_spd_solve.galerkin_preconditioner; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_conservative_spd_solve.solve_residual_checked_cg; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance; python3 - <<'PY'
import importlib.util, numpy, scipy
for m in ('pyamg','threadpoolctl'):
    print(m, bool(importlib.util.find_spec(m)))
print('numpy',numpy.__version__,'scipy',scipy.__version__)
PY
```
- **Container Exit Code**: `0` (`822 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_profiled_matching.fit_ridge_nuisance",
  "matches": [
    {
      "module_stem": "autogenous_prim_ridge_profiled_matching",
      "module_import": "hegel_wormhole_skills.autogenous_prim_ridge_profiled_matching",
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
      ]
    }
  ]
}
pyamg False
threadpoolctl True
numpy 1.26.4 scipy 1.13.0
```

### Turn 3 — `execute_in_remote_docker` (`61701 ms` LLM)
- **Rationale**: Build a runnable baseline that preserves the exact Robin stencil and reuses one factor across states. The isolated import detects deployment dependency failures before independent field, nuisance, and timing tests.
- **Alternatives Weighed**: `Use the library Galerkin V-cycle as the first candidate` (rejected: A single exact reference factor gives a stronger preconditioner on this fixed-size grid; switch only if measured triangular-solve costs threaten the runtime budget.)

```bash
mkdir -p /holdout_task; cat > /holdout_task/robin_continuation_tomography.py <<'PY'
"""Cell-centered Robin continuation and ridge-profiled source tomography."""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, cg, splu
from threadpoolctl import threadpool_limits


def _operators(shape, spacing, gamma):
    """Precompute C-order face indices and return the operator assembler."""
    ids = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    vol = np.prod(spacing)
    faces = []
    for axis, h in enumerate(spacing):
        grid = np.moveaxis(ids, axis, 0)
        faces.append((grid[:-1].ravel(), grid[1:].ravel(),
                      np.concatenate((grid[0].ravel(), grid[-1].ravel())), h))
    i = np.concatenate([f[0] for f in faces])
    j = np.concatenate([f[1] for f in faces])
    flat = ids.ravel()
    rows = np.concatenate((flat, i, j))
    cols = np.concatenate((flat, j, i))

    def assemble(kappa, sigma):
        diag = vol * sigma.copy()
        weights = []
        for c, d, wall, h in faces:
            lc, ld = kappa[c], kappa[d]
            w = (vol / h**2) * (2.0 * lc * ld / (lc + ld))
            weights.append(w)
            diag[c] += w
            diag[d] += w
            kw = kappa[wall]
            # Add the two opposite walls separately if the axis has one cell.
            robin = (vol / h) * gamma * kw / (kw + 0.5 * h * gamma)
            if shape[len(weights) - 1] == 1:
                np.add.at(diag, wall, robin)
            else:
                diag[wall] += robin
        off = -np.concatenate(weights)
        return sparse.csr_matrix((np.concatenate((diag, off, off)), (rows, cols)),
                                 shape=(flat.size, flat.size))

    return vol, assemble


def _factor(a):
    return splu(a.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                options={'SymmetricMode': True, 'Equil': False})


def _tomography(s, z, y, lam_z, lam_s):
    # Represent P as a Gram operator without subtracting nearly equal energies.
    u, values, _ = linalg.svd(z, full_matrices=False)
    data = np.column_stack((s, y))
    coords = u.T @ data
    perpendicular = data - u @ coords
    shrink = np.sqrt(lam_z) / np.hypot(values, np.sqrt(lam_z))
    weighted = shrink[:, None] * coords
    m = s.shape[1]
    gram = perpendicular[:, :m].T @ perpendicular[:, :m] + weighted[:, :m].T @ weighted[:, :m]
    cross = perpendicular[:, :m].T @ perpendicular[:, m:] + weighted[:, :m].T @ weighted[:, m:]
    energy = np.diag(gram)
    profiled = np.divide(cross, energy[:, None], out=np.zeros_like(cross),
                         where=energy[:, None] > 0.0)
    gram.flat[::m + 1] += lam_s
    return profiled, linalg.solve(gram, cross, assume_a='pos')


def solve_robin_continuation_tomography(study: dict) -> dict:
    """Solve all continuation states and fit snapshot amplitudes at the sensors."""
    base_grid = np.asarray(study['kappa_base'], dtype=float)
    shape = base_grid.shape
    spacing = np.array([study['Lx'], study['Ly'], study['Lz']]) / shape
    vol, assemble = _operators(shape, spacing, float(study['gamma_robin']))
    base = base_grid.ravel()
    mod1, mod2, sigma, smod = (np.asarray(study[key], dtype=float).ravel()
                               for key in ('kappa_mod1', 'kappa_mod2', 'sigma_base', 'sigma_mod'))
    def operator(theta):
        return assemble(base + theta * mod1 + theta**2 * mod2, sigma + theta * smod)

    theta = np.asarray(study['thetas'], dtype=float)
    source = np.asarray(study['source_modes'], dtype=float)
    rhs = vol * source.reshape(source.shape[0], -1).T.copy(order='F')
    norms = np.linalg.norm(rhs, axis=0)
    sensors = np.ravel_multi_index(np.asarray(study['sensor_ijk']).T, shape)
    predictions = np.empty((theta.size, sensors.size, rhs.shape[1]))
    # A geometric center balances endpoint ratios for the quadratic coefficient.
    pivot = int(np.argmin(np.abs(np.log(theta) - np.log(theta).mean())))
    order = np.argsort(np.abs(np.log(theta / theta[pivot])))
    with threadpool_limits(limits=1, user_api='blas'):
        a_ref = operator(theta[pivot])
        factor = _factor(a_ref)
        preconditioner = LinearOperator(a_ref.shape, matvec=factor.solve, dtype=float)
        ref_fields = factor.solve(rhs)
        for k in order:
            a = a_ref if k == pivot else operator(theta[k])
            fields = ref_fields.copy()
            local_factor = None
            for m, norm in enumerate(norms):
                if norm == 0:
                    continue
                b = rhs[:, m]
                u = fields[:, m]
                if k != pivot:
                    if local_factor is None:
                        u, _ = cg(a, b, x0=u, M=preconditioner, rtol=2e-11, atol=0., maxiter=45)
                    else:
                        u = local_factor.solve(b)
                residual = b - a @ u
                if np.linalg.norm(residual) > 2e-9 * norm:
                    if local_factor is None:
                        local_factor = factor if k == pivot else _factor(a)
                    u = u + local_factor.solve(residual)
                fields[:, m] = u
            predictions[k] = fields[sensors]
            if k == theta.size - 1:
                terminal = fields.T.reshape(source.shape)
        stacked = predictions.reshape(-1, rhs.shape[1])
        profiled, joint = _tomography(stacked, np.asarray(study['Z_nuisance'], dtype=float),
                                      np.asarray(study['Y_obs'], dtype=float),
                                      float(study['nuisance_ridge']), float(study['source_ridge']))
    return {'terminal_fields': terminal, 'continuation_sensor_predictions': stacked,
            'profiled_amplitudes': profiled, 'joint_source_weights': joint}
PY
python3 -I - <<'PY'
import importlib.util
p='/holdout_task/robin_continuation_tomography.py'
s=importlib.util.spec_from_file_location('robin',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
print(m.solve_robin_continuation_tomography.__name__)
PY
```
- **Container Exit Code**: `0` (`831 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin_continuation_tomography.py']`
- **Stdout (tail)**:
```text
solve_robin_continuation_tomography

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c275623ad9221a4e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c275623a (directory).
```

### Turn 4 — `execute_in_remote_docker` (`59562 ms` LLM)
- **Rationale**: With the public fixtures absent, test every state and source against independently assembled operators, compare both tomography outputs to augmented least squares, and time two full-size high-contrast studies. The results will identify whether the stencil, preconditioner, or profiling needs revision.
- **Alternatives Weighed**: `Compare only the terminal field` (rejected: It would miss residual failures at other continuation states and errors in sensor stacking or nuisance profiling.)

```bash
cat > /tmp/verify_robin.py <<'PY'
import importlib.util
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import sparse, linalg, ndimage
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

path = '/holdout_task/robin_continuation_tomography.py'
spec = importlib.util.spec_from_file_location('robin', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def fixture(kind):
    rng = np.random.default_rng(217 + kind)
    shape = (28, 28, 28) if kind < 2 else (4, 3, 5)
    n = np.prod(shape)
    grids = np.indices(shape)
    v = ndimage.gaussian_filter(rng.normal(size=shape), 1.7) if kind == 0 else rng.random(shape)
    base = np.exp(np.log(3500.) * (v-v.min()) / np.ptp(v)) * 0.02
    if kind == 0:
        m1 = base*rng.uniform(0.01, 0.25, shape)
        m2 = base*rng.uniform(0.01, 0.12, shape)
    else:
        m1 = np.exp(rng.uniform(-5., 3., shape))
        m2 = np.where(grids[1] < shape[1]//2, 25., 0.02)*rng.uniform(0.7, 1.3, shape)
    theta = np.array([1.85, 0.18, 0.98, 0.42, 1.43, 0.72, 1.67, 1.19, 0.15])
    src = np.stack([np.ones(shape), np.sin(4*grids[0]/shape[0])+0.1, rng.normal(size=shape)])
    if kind == 2:
        src[2] *= 0.
    ijk = np.array(np.unravel_index(np.arange(n),shape)).T
    return dict(Lx=1.7,Ly=0.85,Lz=2.3,kappa_base=base,kappa_mod1=m1,kappa_mod2=m2,sigma_base=rng.uniform(0.01,0.5,shape),sigma_mod=rng.uniform(0.01,0.12,shape),gamma_robin=[0.73,0.006,23.][kind],thetas=theta,source_modes=src,sensor_ijk=ijk,nuisance_ridge=1e-3,source_ridge=0.01)

def reference_operator(s, t):
    k = s['kappa_base'] + t*s['kappa_mod1'] + t*t*s['kappa_mod2']
    sig = s['sigma_base'] + t*s['sigma_mod']
    shape = k.shape
    ids = np.arange(k.size).reshape(shape)
    h = np.array([s['Lx'],s['Ly'],s['Lz']])/shape
    vol = np.prod(h)
    gamma = s['gamma_robin']
    wall_reaction = vol*sig.copy()
    matrices = []
    for axis in range(3):
        lo = np.take(ids,range(shape[axis]-1),axis=axis).ravel()
        hi = np.take(ids,range(1,shape[axis]),axis=axis).ravel()
        face_area = np.prod(np.delete(h,axis))
        flat = k.ravel()
        g = face_area / (0.5*h[axis]*(1/flat[lo]+1/flat[hi]))
        ind = np.arange(len(lo))
        grad = sparse.csr_matrix((np.r_[g**0.5,-g**0.5],(np.r_[ind,ind],np.r_[lo,hi])),shape=(len(lo),k.size))
        matrices.append(grad.T@grad)
        for face in [0,-1]:
            slc = [slice(None)]*3
            slc[axis] = face
            idx = tuple(slc)
            wall_reaction[idx] += face_area/(1/gamma+0.5*h[axis]/k[idx])
    return sum(matrices, sparse.diags(wall_reaction.ravel())).tocsc(),vol

elapsed_total = 0.
for kind in range(3):
    s=fixture(kind)
    rng=np.random.default_rng(700+kind)
    mats, ref = [], []
    with threadpool_limits(limits=1,user_api='blas'):
        for t in s['thetas']:
            a, vol = reference_operator(s,t)
            b = vol*s['source_modes'].reshape(3,-1).T
            mats.append(a)
            ref.append(splu(a,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0).solve(b))
        ref = np.asarray(ref)
        design = ref.reshape(-1,3)
        n = design.shape[0]
        z = np.column_stack([np.ones(n),design[:,0]+1e-3*rng.normal(size=n),rng.normal(size=(n,2))])
        y = design @ rng.normal(size=(3,5)) + z @ rng.normal(size=(4,5)) + 1e-4*rng.normal(size=(n,5))
    s.update(Z_nuisance=z,Y_obs=y)
    # Time the complete implementation with full-volume sensors.
    start=perf_counter()
    out=mod.solve_robin_continuation_tomography(s)
    elapsed=perf_counter()-start
    if kind < 2: elapsed_total+=elapsed
    expected_shapes={'terminal_fields':s['source_modes'].shape,'continuation_sensor_predictions':design.shape,'profiled_amplitudes':(3,5),'joint_source_weights':(3,5)}
    assert out.keys()==expected_shapes.keys()
    for key, shape in expected_shapes.items():
        assert out[key].shape == shape and np.isfinite(out[key]).all() and out[key].dtype.kind=='f',key
    states=out['continuation_sensor_predictions'].reshape(ref.shape)
    assert np.array_equal(out['terminal_fields'].reshape(3,-1).T,states[-1])
    with threadpool_limits(limits=1,user_api='blas'):
        den=np.linalg.norm(b,axis=0)
        valid=den>0
        residuals=np.array([np.linalg.norm((a@u-b)[:,valid],axis=0)/den[valid] for a,u in zip(mats,states)])
        err=np.linalg.norm(states-ref)/np.linalg.norm(ref)
        reg_z=np.sqrt(s['nuisance_ridge'])*np.eye(z.shape[1])
        errors=[]
        for m in np.flatnonzero(valid):
            aug=np.block([[z,design[:,m:m+1]],[reg_z,np.zeros((z.shape[1],1))]])
            ref_m=linalg.lstsq(aug,np.vstack([y,np.zeros((z.shape[1],y.shape[1]))]))[0][-1]
            errors.append(np.linalg.norm(out['profiled_amplitudes'][m]-ref_m)/np.linalg.norm(ref_m))
        aug=np.block([[z,design],[reg_z,np.zeros((z.shape[1],3))],[np.zeros((3,z.shape[1])),np.sqrt(s['source_ridge'])*np.eye(3)]])
        joint=linalg.lstsq(aug,np.vstack([y,np.zeros((z.shape[1]+3,y.shape[1]))]))[0][-3:]
        je=np.linalg.norm(out['joint_source_weights']-joint)/np.linalg.norm(joint)
    print(f'case={kind} time_s={elapsed:.3f} residual_max={residuals.max():.3g} residual_argmax={np.unravel_index(residuals.argmax(),residuals.shape)} field_rel={err:.3g} profile_max={max(errors):.3g} joint_rel={je:.3g}',flush=True)
    assert residuals.max() <= 1e-7 and err < 1e-6 and max(errors)<=1e-5 and je<=0.03
print(f'two_large_studies_seconds={elapsed_total:.3f}',flush=True)
assert elapsed_total<=25.
validator = Path('/holdout_task/validate_public_schema.py')
fixture_path = Path('/holdout_task/public_eq01_studies.pkl')
print('public_validator_present',validator.is_file(),'public_fixture_present',fixture_path.is_file())
PY
python3 -I /tmp/verify_robin.py
rc=$?; if [ "$rc" -ne 0 ]; then exit "$rc"; fi
if [ -f /holdout_task/validate_public_schema.py ] && [ -f /holdout_task/public_eq01_studies.pkl ]; then python3 /holdout_task/validate_public_schema.py; fi
```
- **Container Exit Code**: `0` (`71450 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case=0 time_s=3.299 residual_max=1.96e-11 residual_argmax=(3, 1) field_rel=8.23e-14 profile_max=4.41e-11 joint_rel=1.1e-12
case=1 time_s=9.170 residual_max=9.03e-10 residual_argmax=(8, 0) field_rel=3.16e-13 profile_max=3.36e-10 joint_rel=1.93e-12
case=2 time_s=0.013 residual_max=1.84e-11 residual_argmax=(2, 1) field_rel=1.96e-12 profile_max=5.63e-12 joint_rel=5.91e-13
two_large_studies_seconds=12.469
public_validator_present False public_fixture_present False

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c275623ad9221a4e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c275623a (directory).
```

### Turn 5 — `submit_completed_task` (`14862 ms` LLM)
- **Rationale**: The final artifact passed isolated verification against independently assembled operators and augmented least-squares references, with margin on every tested accuracy and runtime threshold.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`13` turns remaining; artifact status={'/holdout_task/robin_continuation_tomography.py': 'FILE:5743'}): Do NOT lock prematurely with `13` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/robin_continuation_tomography.py`. Two full-size synthetic stress studies ran in 12.47 seconds total. The worst relative residual was 9.03e-10, and both tomography outputs met their accuracy targets. The public fixtures and validator were unavailable.`

### Turn 6 — `execute_in_remote_docker` (`26387 ms` LLM)
- **Rationale**: The existing tests verified the Robin stencil and both tomography outputs against independent references. This focused audit tests whether coefficient maps that shift conductivity between regions exhaust the runtime margin, and whether nearly duplicate nuisance columns degrade amplitude accuracy. A failure would identify which component to revise.
- **Alternatives Weighed**: `Apply batch-classification and relay-mode checks` (rejected: This study has neither classification labels nor relay states; its nuisance model is a specified ridge regression.)

```bash
set -e
if [ -f /holdout_task/validate_public_schema.py ] && [ -f /holdout_task/public_eq01_studies.pkl ]; then
  python3 -I /holdout_task/validate_public_schema.py
fi
python3 -I - <<'PY'
from pathlib import Path
src = Path('/tmp/verify_robin.py').read_text()
addon = '''
original_fixture = fixture
def fixture(kind):
    s = original_fixture(kind)
    rng = np.random.default_rng(905 + kind)
    ii, jj, kk = np.indices(s['kappa_base'].shape)
    # Put the base and modulation on different regions so the endpoint
    # operators are not related by a nearly uniform scalar.
    mask = ii < 14 if kind == 0 else ((ii // 4 + jj // 4 + kk // 4) % 2 == 0)
    s['kappa_base'] = np.where(mask, 70., .02)
    s['kappa_mod1'] = np.where(mask, .003, .2) * rng.uniform(.8, 1.2, mask.shape)
    s['kappa_mod2'] = np.where(mask, .002, 70.) * rng.uniform(.8, 1.2, mask.shape)
    s['sigma_base'] *= .03
    s['sigma_mod'] *= .05
    s['gamma_robin'] = .002 if kind == 0 else 48.
    s['thetas'] = np.array([.15, 1.85, 1.76, 1.66, 1.56, 1.46, .17, .19, .21])
    return s

'''
src = src.replace('elapsed_total = 0.', addon + 'elapsed_total = 0.').replace('for kind in range(3):', 'for kind in range(2):')
src = src.replace('rng.normal(size=(n,2))])', 'rng.normal(size=n), np.ones(n)+1e-8*rng.normal(size=n)])')
Path('/tmp/verify_robin_endpoints.py').write_text(src)
PY
python3 -I /tmp/verify_robin_endpoints.py
```
- **Container Exit Code**: `0` (`73966 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case=0 time_s=9.863 residual_max=3.71e-09 residual_argmax=(1, 0) field_rel=1.91e-11 profile_max=1.23e-07 joint_rel=4.5e-09
case=1 time_s=7.234 residual_max=1.99e-11 residual_argmax=(4, 1) field_rel=5.82e-14 profile_max=1.98e-13 joint_rel=6.02e-14
two_large_studies_seconds=17.097
public_validator_present False public_fixture_present False

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c275623ad9221a4e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c275623a (directory).
```

### Turn 7 — `submit_completed_task` (`13983 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated verification against independently assembled operators and augmented least-squares references. Endpoint conductivity shifts and near-duplicate nuisance columns left all measured errors and the two-study runtime within tolerance.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/robin_continuation_tomography.py': 'FILE:5743'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/robin_continuation_tomography.py`. The harsher pair of full-size synthetic studies completed in 17.10 seconds against a 25-second limit. The worst relative PDE residual was 3.71e-9, and both tomography outputs met their accuracy targets. The public fixtures and validator were unavailable.`

### Turn 8 — `execute_in_remote_docker` (`27520 ms` LLM)
- **Rationale**: Test whether reusing fallback factors improves the endpoint runtime margin without degrading either tomography output. Run the same independent references on both stress suites and keep the verified artifact unchanged unless the challenger improves the bottleneck.
- **Alternatives Weighed**: `Add another reference factor up front` (rejected: It adds setup cost even when the existing reference is sufficient; the challenger reuses only factors already needed for residual correction.)

```bash
set -e
python3 -I - <<'PY'
from pathlib import Path
p = Path('/holdout_task/robin_continuation_tomography.py')
s = p.read_text()
s = s.replace('preconditioner = LinearOperator(a_ref.shape, matvec=factor.solve, dtype=float)', 'factors = {pivot: factor}\n        log_theta = np.log(theta)')
s = s.replace('fields = ref_fields.copy()\n            local_factor = None', 'fields = ref_fields.copy()\n            nearest = min(factors, key=lambda j: abs(log_theta[k] - log_theta[j]))\n            preconditioner = LinearOperator(a.shape, matvec=factors[nearest].solve, dtype=float)\n            local_factor = None')
s = s.replace('local_factor = factor if k == pivot else _factor(a)', 'local_factor = factor if k == pivot else _factor(a)\n                        factors[k] = local_factor')
Path('/tmp/robin_reuse_candidate.py').write_text(s)
for name in ['verify_robin', 'verify_robin_endpoints']:
    t = Path('/tmp/' + name + '.py').read_text()
    t = t.replace(str(p), '/tmp/robin_reuse_candidate.py')
    t = t.replace('elapsed_total = 0.', '''base_factor = mod._factor
factor_count = 0
def counted_factor(a):
    global factor_count
    factor_count += 1
    return base_factor(a)
mod._factor = counted_factor
elapsed_total = 0.''')
    t = t.replace('start=perf_counter()', 'factor_count=0\n    start=perf_counter()').replace('case={kind} time_s=', 'case={kind} factors={factor_count} time_s=')
    Path('/tmp/' + name + '_reuse.py').write_text(t)
PY
python3 -I /tmp/verify_robin_endpoints_reuse.py
python3 -I /tmp/verify_robin_reuse.py
```
- **Container Exit Code**: `0` (`121523 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case=0 factors=6 time_s=6.341 residual_max=3.72e-09 residual_argmax=(1, 0) field_rel=1.81e-11 profile_max=9.39e-08 joint_rel=4.23e-09
case=1 factors=2 time_s=4.193 residual_max=1.99e-11 residual_argmax=(4, 1) field_rel=1.7e-13 profile_max=2.59e-12 joint_rel=1.57e-13
two_large_studies_seconds=10.534
public_validator_present False public_fixture_present False
case=0 factors=1 time_s=2.835 residual_max=1.96e-11 residual_argmax=(3, 1) field_rel=8.23e-14 profile_max=4.41e-11 joint_rel=1.1e-12
case=1 factors=2 time_s=6.756 residual_max=9.03e-10 residual_argmax=(8, 0) field_rel=3.16e-13 profile_max=3.36e-10 joint_rel=1.93e-12
case=2 factors=1 time_s=0.010 residual_max=1.84e-11 residual_argmax=(2, 1) field_rel=1.96e-12 profile_max=5.63e-12 joint_rel=5.91e-13
two_large_studies_seconds=9.590
public_validator_present False public_fixture_present False

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c275623ad9221a4e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_c275623a (directory).
```

### Turn 9 — `execute_in_remote_docker` (`15996 ms` LLM)
- **Rationale**: Promote the faster challenger and run both independent stress suites against the exact installed artifact in isolated Python. If installation or an acceptance threshold fails, retain the verified baseline and investigate the specific failure.
- **Alternatives Weighed**: `Keep the verified single-reference baseline` (rejected: Reusing available factors improved runtime on both stress suites without breaching any accuracy threshold.)

```bash
set -e
cp /tmp/robin_reuse_candidate.py /holdout_task/robin_continuation_tomography.py
python3 -I /tmp/verify_robin_endpoints.py
python3 -I /tmp/verify_robin.py
```
- **Container Exit Code**: `0` (`113946 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin_continuation_tomography.py']`
- **Stdout (tail)**:
```text
case=0 time_s=5.958 residual_max=3.72e-09 residual_argmax=(1, 0) field_rel=1.81e-11 profile_max=9.39e-08 joint_rel=4.23e-09
case=1 time_s=3.914 residual_max=1.99e-11 residual_argmax=(4, 1) field_rel=1.7e-13 profile_max=2.59e-12 joint_rel=1.57e-13
two_large_studies_seconds=9.873
public_validator_present False public_fixture_present False
case=0 time_s=2.583 residual_max=1.96e-11 residual_argmax=(3, 1) field_rel=8.23e-14 profile_max=4.41e-11 joint_rel=1.1e-12
case=1 time_s=7.468 residual_max=9.03e-10 residual_argmax=(8, 0) field_rel=3.16e-13 profile_max=3.36e-10 joint_rel=1.93e-12
case=2 time_s=0.009 residual_max=1.84e-11 residual_argmax=(2, 1) field_rel=1.96e-12 profile_max=5.63e-12 joint_rel=5.91e-13
two_large_studies_seconds=10.051
public_validator_present False public_fixture_present False

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=2cba5ec607a45928 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_2cba5ec6 (directory).
```

### Turn 10 — `submit_completed_task` (`13255 ms` LLM)
- **Rationale**: The installed artifact passed isolated Python verification against independently assembled operators and augmented least-squares references on both stress suites. Nearest-factor reuse improved endpoint runtime while preserving every measured accuracy margin.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin_continuation_tomography.py`. The slower of two full-size synthetic test pairs took 10.05 seconds against the 25-second limit. The worst relative PDE residual was 3.72e-9, and both tomography outputs met their accuracy targets. The public fixtures and validator were unavailable.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/robin_continuation_tomography.py": "FILE:5954"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "all_valid": true,
    "max_terminal_pde_rel_res": 0.0,
    "max_sensor_stack_rel_l2": 0.0,
    "max_profiled_rel_l2": 0.0,
    "max_joint_weights_rel_l2": 0.0,
    "student_elapsed_sec": 5.9744,
    "instance_diagnostics": [
      {
        "seed": 99401,
        "terminal_pde_rel_res": 0.0,
        "sensor_stack_rel_l2": 0.0,
        "profiled_rel_l2": 0.0,
        "joint_rel_l2": 0.0
      },
      {
        "seed": 99402,
        "terminal_pde_rel_res": 0.0,
        "sensor_stack_rel_l2": 0.0,
        "profiled_rel_l2": 0.0,
        "joint_rel_l2": 0.0
      }
    ]
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_terminal_pde_rel_res\": 0.0, \"max_sensor_stack_rel_l2\": 0.0, \"max_profiled_rel_l2\": 0.0, \"max_joint_weights_rel_l2\": 0.0, \"student_elapsed_sec\": 5.9744, \"instance_diagnostics\": [{\"seed\": 99401, \"terminal_pde_rel_res\": 0.0, \"sensor_stack_rel_l2\": 0.0, \"profiled_rel_l2\": 0.0, \"joint_rel_l2\": 0.0}, {\"seed\": 99402, \"terminal_pde_rel_res\": 0.0, \"sensor_stack_rel_l2\": 0.0, \"profiled_rel_l2\": 0.0, \"joint_rel_l2\": 0.0}]}\n__HEGEL_VERIFIER_AUTH_60d9aa82b2935764d5e4bb5fdf22329c__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_terminal_pde_rel_res\": 0.0, \"max_sensor_stack_rel_l2\": 0.0, \"max_profiled_rel_l2\": 0.0, \"max_joint_weights_rel_l2\": 0.0, \"student_elapsed_sec\": 5.9744, \"instance_diagnostics\": [{\"seed\": 99401, \"terminal_pde_rel_res\": 0.0, \"sensor_stack_rel_l2\": 0.0, \"profiled_rel_l2\": 0.0, \"joint_rel_l2\": 0.0}, {\"seed\": 99402, \"terminal_pde_rel_res\": 0.0, \"sensor_stack_rel_l2\": 0.0, \"profiled_rel_l2\": 0.0, \"joint_rel_l2\": 0.0}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 13373.77,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
