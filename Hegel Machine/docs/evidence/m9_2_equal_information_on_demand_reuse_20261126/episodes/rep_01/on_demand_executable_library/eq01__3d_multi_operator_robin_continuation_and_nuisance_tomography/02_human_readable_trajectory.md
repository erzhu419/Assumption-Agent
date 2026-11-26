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
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `299,865` (`288,235` / `78,976` / `11,630`) |
| **Total Episode Wall Time (s)** | `558.99s` |

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

### Turn 1 — `execute_in_remote_docker` (`13635 ms` LLM)
- **Rationale**: Inspect the public validator for contract mismatches and check for reusable solver primitives; if none are available, implement the reference preconditioner with SciPy and test it against the exact stencil.
- **Alternatives Weighed**: `Independent LU per state` (rejected: The stated runtime already exceeds the two-study budget.)

```bash
cd /holdout_task && ls -lh && sed -n '1,220p' validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'SPD AMG CG reusable preconditioning ridge nuisance profiling'; python3 - <<'PY'
import importlib.util, os
for m in ('numpy','scipy','pyamg','threadpoolctl'):
 print(m, bool(importlib.util.find_spec(m)))
print('CPU affinity',len(os.sched_getaffinity(0)))
PY
```
- **Container Exit Code**: `0` (`390 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ter: int=450, refinements: int=4) -> tuple[np.ndarray, float] | Solve a real SPD operator of shape (n, n) against rhs of shape (n,), returning | REQ: grid op assumes isotropic scalar coeff>0 + zero Dirichlet ghosts; CG/AMG need square SPD | --describe autogenous_prim_conservative_spd_solve.solve_residual_checked_cg | STATUS: APPLICABLE_SUBROUTINE(requires_spd_operator_or_matvec)
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance | STATUS: APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)
autogenous_prim_conservative_spd_solve@05cb4387 | galerkin_preconditioner(operator: sparse.spmatrix | sparse.sparray, *, strength_cutoff: float=0.08, max_coarse_fraction: float=0.76, coarse_size: int=64, seed: int=12) -> LinearOperator | Map a real sparse SPD matrix of shape (n, n) to a symmetric (n, n) V-cycle app | REQ: grid op assumes isotropic scalar coeff>0 + zero Dirichlet ghosts; CG/AMG need square SPD | --describe autogenous_prim_conservative_spd_solve.galerkin_preconditioner | STATUS: APPLICABLE_SUBROUTINE(requires_assembled_csr_spd_matrix)
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
numpy True
scipy True
pyamg False
threadpoolctl True
CPU affinity 32
```
- **Stderr (tail)**:
```text
bash: line 1: cd: /holdout_task: No such file or directory
```

### Turn 2 — `execute_in_remote_docker` (`16694 ms` LLM)
- **Rationale**: Locate the public fixtures without copying data, then inspect the solver and ridge primitives to determine whether they fit the exact Robin operator and runtime budget.
- **Alternatives Weighed**: `Assume the public fixtures are absent and use only synthetic tests` (rejected: Only the advertised holdout path was checked; the initial snapshot identifies /pretrain_task as another likely location.)

```bash
pwd; ls -ld /holdout_task /pretrain_task /workspace /app 2>/dev/null; for d in /holdout_task /pretrain_task /workspace /app; do if [ -d "$d" ]; then find "$d" -maxdepth 3 -type f \( -name '*eq01*' -o -name 'validate_public_schema.py' \) -print; fi; done; export PYTHONPATH=/tmp; python3 -m hegel_wormhole_skills --describe autogenous_prim_conservative_spd_solve.galerkin_preconditioner; python3 -m hegel_wormhole_skills --describe autogenous_prim_conservative_spd_solve.solve_residual_checked_cg; python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
```
- **Container Exit Code**: `0` (`418 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
]
}
{
  "ok": true,
  "target": "autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance",
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
```

### Turn 3 — `execute_in_remote_docker` (`16233 ms` LLM)
- **Rationale**: Inspect the V-cycle for symmetry, expensive smoothers, and import dependencies before implementing the solver. Locate the public fixtures by filename only; if they are absent, use an independent synthetic reference test.
- **Alternatives Weighed**: `Use fixed geometric aggregates without comparing preconditioner implementations` (rejected: Fixed aggregates may split high-conductivity paths; inspect how the available algebraic implementation handles those paths before choosing the baseline.)

```bash
ls -la /pretrain_task; ls -la /tmp/hegel_wormhole_skills; sed -n '1,300p' /tmp/hegel_wormhole_skills/autogenous_prim_conservative_spd_solve.py; find / -maxdepth 4 \( -name 'public_eq01_studies.pkl' -o -name 'validate_public_schema.py' \) -print 2>/dev/null
```
- **Container Exit Code**: `0` (`461 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
bottom = lambda r: linalg.cho_solve(factor, r)

    def cycle(r, level=0):
        if level == len(levels):
            return bottom(r)
        (a_level, p, pt, inv) = levels[level]
        z = inv * r
        z += p @ cycle(pt @ (r - a_level @ z), level + 1)
        return z + inv * (r - a_level @ z)

    def apply(r):
        return cycle(np.asarray(r).reshape(n0))
    return LinearOperator((n0, n0), matvec=apply, rmatvec=apply, dtype=float)


@track_wormhole_skill("autogenous_prim_conservative_spd_solve")
def solve_residual_checked_cg(operator: np.ndarray | sparse.spmatrix | sparse.sparray | LinearOperator, rhs: np.ndarray, *, preconditioner: LinearOperator | None=None, tol: float=1e-08, maxiter: int=450, refinements: int=4) -> tuple[np.ndarray, float]:
    """Solve a real SPD operator of shape (n, n) against rhs of shape (n,), returning the float64 solution (n,) and its directly recomputed relative 2-norm residual."""
    (a, b) = (aslinearoperator(operator), np.asarray(rhs, dtype=float))
    if b.ndim != 1 or not b.size or a.shape != (b.size, b.size):
        raise ValueError('expected a square operator and matching nonempty vector')
    if not np.all(np.isfinite(b)) or not 0 < tol < 1 or maxiter < 1 or (refinements < 0):
        raise ValueError('invalid right-hand side or solver controls')
    if preconditioner is not None and preconditioner.shape != a.shape:
        raise ValueError('preconditioner shape must match the operator')
    x = np.zeros_like(b)
    bscale = np.max(np.abs(b))
    if bscale == 0:
        return (x, 0.0)
    bnorm = linalg.norm(b / bscale)
    for attempt in range(refinements + 1):
        r = b - a @ x
        rscale = np.max(np.abs(r))
        if not np.isfinite(rscale):
            break
        if rscale == 0:
            return (x, 0.0)
        unit_r = r / rscale
        rel = float(rscale / bscale * (linalg.norm(unit_r) / bnorm))
        if rel <= tol:
            return (x, rel)
        if attempt == refinements:
            break
        inner_tol = min(0.01, 0.85 * tol / rel)
        (step, _) = cg(a, unit_r, M=preconditioner, rtol=inner_tol, atol=0.0, maxiter=maxiter)
        if not np.all(np.isfinite(step)):
            break
        x += rscale * step
    r = b - a @ x
    if np.all(np.isfinite(r)):
        rel = float(linalg.norm(r / bscale, check_finite=False) / bnorm)
        if rel <= tol:
            return (x, rel)
    raise RuntimeError('CG did not meet the directly recomputed residual tolerance')
```

### Turn 4 — `execute_in_remote_docker` (`64425 ms` LLM)
- **Rationale**: The public fixtures are not mounted. Implement the exact operator and check that the isolated interpreter can compile it; next compare all outputs with independent references and measure the full-size runtime.
- **Alternatives Weighed**: `Use the available Galerkin preconditioner` (rejected: Its coarsening and smoothing performance on this contrast is unmeasured; first test whether one reusable exact factor meets the runtime budget.)

```bash
mkdir -p /holdout_task
cat > /holdout_task/robin_continuation_tomography.py <<'PY'
"""Cell-centered Robin continuation and ridge-profiled source tomography."""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, cg, splu

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import nullcontext

    def threadpool_limits(**kwargs):
        return nullcontext()


class _RobinOperator:
    def __init__(self, shape, lengths, gamma):
        self.shape = shape
        self.h = np.asarray(lengths, dtype=float) / shape
        self.volume = float(np.prod(self.h))
        self.gamma = gamma
        size = int(np.prod(shape))
        ids = np.arange(size).reshape(shape)
        self.faces = []
        self.walls = []
        for axis, h in enumerate(self.h):
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            self.faces.append((ids[tuple(lo)].ravel(), ids[tuple(hi)].ravel(),
                               self.volume / h**2))
            self.walls.append((np.take(ids, [0, shape[axis] - 1], axis=axis).ravel(), h))
        left = np.concatenate([f[0] for f in self.faces])
        right = np.concatenate([f[1] for f in self.faces])
        cells = ids.ravel()
        self.rows = np.concatenate((cells, left, right))
        self.cols = np.concatenate((cells, right, left))

    def assemble(self, kappa, sigma):
        kappa = kappa.ravel()
        diag = self.volume * sigma.ravel()
        off = []
        for i, j, scale in self.faces:
            w = scale * (2 * kappa[i] * kappa[j] / (kappa[i] + kappa[j]))
            diag[i] += w
            diag[j] += w
            off.append(-w)
        for ids, h in self.walls:
            k = kappa[ids]
            wall = (self.volume / h) * (self.gamma * k / (k + self.gamma * h / 2))
            np.add.at(diag, ids, wall)
        values = np.concatenate([diag] + off + off)
        return sparse.csr_matrix((values, (self.rows, self.cols)),
                                 shape=(diag.size, diag.size))


def _profile(sensors, observations, nuisance, ridge_z, ridge_s):
    # An orthogonal transform eliminates the nuisance columns from the
    # augmented least-squares problem without subtracting nearly equal Grams.
    n, p = nuisance.shape
    m = sensors.shape[1]
    rhs = np.zeros((n + p, m + observations.shape[1]), order='F')
    rhs[:n, :m] = sensors
    rhs[:n, m:] = observations
    if p:
        design = np.empty((n + p, p), order='F')
        design[:n] = nuisance
        design[n:] = np.sqrt(ridge_z) * np.eye(p)
        (reflectors, tau), _ = linalg.qr(design, mode='raw', overwrite_a=True,
                                         check_finite=False)
        ormqr = linalg.get_lapack_funcs('ormqr', (reflectors, rhs))
        _, work, _ = ormqr('L', 'T', reflectors, tau, rhs, -1)
        rhs, _, _ = ormqr('L', 'T', reflectors, tau, rhs, int(work[0]), overwrite_c=1)
    s, y = rhs[p:, :m], rhs[p:, m:]
    energy = np.einsum('ij,ij->j', s, s)
    cross = s.T @ y
    amplitudes = np.divide(cross, energy[:, None], out=np.zeros_like(cross),
                           where=energy[:, None] > 0)
    # Solve the small joint source problem directly in square-root form.
    design_s = np.vstack((s, np.sqrt(ridge_s) * np.eye(m)))
    target_s = np.vstack((y, np.zeros((m, y.shape[1]))))
    q, r = linalg.qr(design_s, mode='economic', check_finite=False)
    weights = linalg.solve_triangular(r, q.T @ target_s, check_finite=False)
    return amplitudes, weights


def _solve(study):
    kb, k1, k2, sb, s1 = (np.asarray(study[key], dtype=float) for key in
                         ('kappa_base', 'kappa_mod1', 'kappa_mod2', 'sigma_base', 'sigma_mod'))
    grid = _RobinOperator(kb.shape, [study[key] for key in ('Lx', 'Ly', 'Lz')],
                          float(study['gamma_robin']))
    theta = np.asarray(study['thetas'], dtype=float)
    modes = np.asarray(study['source_modes'], dtype=float).reshape(-1, kb.size)
    rhs = grid.volume * modes
    rhs_norm = np.linalg.norm(rhs, axis=1)
    indices = np.ravel_multi_index(np.asarray(study['sensor_ijk']).T, kb.shape)
    predictions = np.empty((theta.size, indices.size, modes.shape[0]))

    def matrix(t):
        return grid.assemble(kb + t * k1 + t**2 * k2, sb + t * s1)

    # Balance the theta/theta_ref ratios for the positive diffusion terms.
    t_ref = np.sqrt(theta.min() * theta.max())
    reference = matrix(t_ref)
    factor = splu(reference.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0,
                  options={'SymmetricMode': True})
    inverse = LinearOperator(reference.shape, matvec=factor.solve, dtype=float)
    fields = np.zeros_like(rhs)
    last = np.empty_like(fields)
    for k in np.argsort(theta):
        a = matrix(theta[k])
        for m, b in enumerate(rhs):
            if rhs_norm[m] == 0:
                continue
            u, _ = cg(a, b, x0=fields[m], M=inverse, rtol=1e-10, maxiter=240)
            if np.linalg.norm(b - a @ u) > 1e-8 * rhs_norm[m]:
                direct = splu(a.tocsc(), permc_spec='MMD_AT_PLUS_A')
                u = direct.solve(b)
                u += direct.solve(b - a @ u)
            fields[m] = u
        predictions[k] = fields[:, indices].T
        if k == theta.size - 1:
            last[:] = fields

    stacked = predictions.reshape(-1, modes.shape[0])
    amplitudes, weights = _profile(stacked, np.asarray(study['Y_obs'], dtype=float),
                                   np.asarray(study['Z_nuisance'], dtype=float),
                                   study['nuisance_ridge'], study['source_ridge'])
    return {'terminal_fields': last.reshape((-1,) + kb.shape),
            'continuation_sensor_predictions': stacked,
            'profiled_amplitudes': amplitudes,
            'joint_source_weights': weights}


def solve_robin_continuation_tomography(study: dict) -> dict:
    """Solve the continuation family and fit independent and joint source weights."""
    with threadpool_limits(limits=1, user_api='blas'):
        return _solve(study)
PY
python3 -I -m py_compile /holdout_task/robin_continuation_tomography.py
```
- **Container Exit Code**: `0` (`369 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin_continuation_tomography.py']`
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6988662a973992e1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6988662a (directory).
```

### Turn 5 — `execute_in_remote_docker` (`55689 ms` LLM)
- **Rationale**: With the public fixtures unavailable, test two full-size synthetic studies with distinct high-contrast patterns and shuffled states. Compare all residuals and outputs against independent references. If runtime fails, improve the preconditioner; if an output comparison fails, isolate the stencil, ordering, or ridge calculation before changing the solver.
- **Alternatives Weighed**: `Test only output shapes and terminal residuals` (rejected: That would miss a bad intermediate solve, misordered sensors, or an incorrect nuisance projection.)

```bash
cat > /tmp/test_robin_eq01.py <<'PY'
import importlib.util
from time import perf_counter
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

spec = importlib.util.spec_from_file_location('robin', '/holdout_task/robin_continuation_tomography.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def reference(st, t):
    k = st['kappa_base'] + t * st['kappa_mod1'] + t*t * st['kappa_mod2']
    shape = k.shape
    h = np.array([st['Lx'], st['Ly'], st['Lz']]) / shape
    v = np.prod(h)
    xyz = np.indices(shape).reshape(3, -1)
    k = k.ravel()
    cell = np.arange(k.size)
    diag = v * (st['sigma_base'] + t * st['sigma_mod']).ravel()
    diffs, weights = [], []
    for d in range(3):
        start = cell[xyz[d] < shape[d]-1]
        end = start + np.prod(shape[d+1:], dtype=int)
        diffs.append(sparse.csr_matrix((np.tile([1., -1.], start.size), (np.repeat(np.arange(start.size), 2), np.column_stack((start, end)).ravel())), shape=(start.size, k.size)))
        weights.append(2 * v / h[d]**2 / (1/k[start] + 1/k[end]))
        sides = cell[(xyz[d] == 0) | (xyz[d] == shape[d]-1)]
        diag[sides] += (v / h[d]) / (1/st['gamma_robin'] + h[d]/(2*k[sides]))
    d = sparse.vstack(diffs, format='csr')
    return (d.T @ d.multiply(np.concatenate(weights)[:, None]) + sparse.diags(diag)).tocsr(), v

def make_study(case):
    rng = np.random.default_rng(801+case)
    shape = (28,)*3
    x,y,z = np.indices(shape) / 27
    if case == 0:
        base = np.exp(rng.uniform(0, np.log(3500), shape))
        k1 = 0.02 + base * rng.uniform(0.01, 0.9, shape)
        k2 = 0.02 + base * rng.uniform(0.01, 0.9, shape)
        lengths = (1.2, 0.93, 1.61)
    else:
        base = np.where((x<.18) | ((y>.3)&(y<.5)) | (z>.85), 3500., 1.)
        k1 = .01 + 120*np.exp(-90*(z-.55)**2)
        k2 = np.where((x>.75) | ((z>.3)&(z<.5)), 2800., .001)
        lengths = (.75, 1.15, 1.83)
    t = rng.permutation(np.linspace(.15, 1.85, 9))
    sensors = rng.integers(0, 28, (80,3))
    n = len(sensors)*len(t)
    a = rng.normal(size=(n, 15))
    a[:, -1] = a[:, 0] + 1e-8*a[:, 1]
    return dict(Lx=lengths[0], Ly=lengths[1], Lz=lengths[2], kappa_base=base,
        kappa_mod1=k1, kappa_mod2=k2, sigma_base=.01+.015*x, sigma_mod=.002+.012*y,
        gamma_robin=[2., .18][case], thetas=t,
        source_modes=np.stack((np.ones(shape), np.sin(6.3*x)*np.cos(4*y), np.exp(-20*((x-.3)**2+(y-.2)**2+(z-.7)**2)))),
        sensor_ijk=sensors, Z_nuisance=a, Y_obs=rng.normal(size=(n, 5)), nuisance_ridge=1e-6, source_ridge=2e-6)

def run(st):
    capture = []
    real_cg = mod.cg
    def cg(*args, **kwargs):
        steps = [0]
        def tick(_): steps[0] += 1
        sol = real_cg(*args, **kwargs, callback=tick)
        capture.append((sol[0], steps[0]))
        return sol
    mod.cg = cg
    start = perf_counter()
    try:
        out = mod.solve_robin_continuation_tomography(st)
    finally:
        mod.cg = real_cg
    elapsed = perf_counter() - start
    idx = tuple(st['sensor_ijk'].T)
    states = len(st['thetas'])
    truth = np.empty((states, len(idx[0]), 3))
    resid = []
    for j, k in enumerate(np.argsort(st['thetas'])):
        mat, v = reference(st, st['thetas'][k])
        b = v*st['source_modes'].reshape(3, -1).T
        u = np.column_stack([capture[3*j+m][0] for m in range(3)])
        resid.extend(np.linalg.norm(mat@u-b, axis=0) / np.linalg.norm(b,axis=0))
        exact = splu(mat.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0, options={'SymmetricMode':True}).solve(b)
        shaped = exact.T.reshape(st['source_modes'].shape)
        truth[k] = shaped[(slice(None),)+idx].T
        if k == states-1:
            fields = shaped
    s = truth.reshape(-1, 3)
    z,y = st['Z_nuisance'], st['Y_obs']
    p = z.shape[1]
    y_aug = np.vstack((y, np.zeros((p, y.shape[1]))))
    z_aug = np.vstack((z, np.sqrt(st['nuisance_ridge'])*np.eye(p)))
    s_aug = np.vstack((s, np.zeros((p,3))))
    prof = np.stack([linalg.lstsq(np.column_stack((z_aug, col)), y_aug, lapack_driver='gelsd')[0][-1] for col in s_aug.T])
    block = np.vstack((np.column_stack((z_aug,s_aug)), np.column_stack((np.zeros((3,p)), np.sqrt(st['source_ridge'])*np.eye(3)))))
    joint = linalg.lstsq(block, np.vstack((y_aug, np.zeros((3,y.shape[1])))), lapack_driver='gelsd')[0][-3:]
    expected = dict(terminal_fields=fields, continuation_sensor_predictions=s, profiled_amplitudes=prof, joint_source_weights=joint)
    assert out.keys() == expected.keys()
    err = {}
    for name, a in expected.items():
        actual = out[name]
        assert actual.shape == a.shape and actual.dtype.kind == 'f'
        err[name] = np.linalg.norm(actual-a)/np.linalg.norm(a)
    print(f'solve={elapsed:.3f}s max_CG_iters={max(n for _,n in capture)} max_residual={max(resid):.3g} errors={err}', flush=True)
    assert max(resid) <= 1e-7
    assert err['profiled_amplitudes'] <= 1e-5 and err['joint_source_weights'] <= .03
    assert max(err['terminal_fields'], err['continuation_sensor_predictions']) <= 1e-7
    return elapsed

with threadpool_limits(limits=1, user_api='blas'):
    total = sum(run(make_study(i)) for i in range(2))
print(f'Two-study solve time: {total:.3f}s')
assert total <= 25
PY
python3 -I /tmp/test_robin_eq01.py
```
- **Container Exit Code**: `0` (`32418 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
solve=4.680s max_CG_iters=18 max_residual=1.2e-10 errors={'terminal_fields': 6.007665292256776e-14, 'continuation_sensor_predictions': 3.1393291316855593e-13, 'profiled_amplitudes': 4.331276187827248e-12, 'joint_source_weights': 5.770239067407258e-12}
solve=9.059s max_CG_iters=39 max_residual=7.88e-09 errors={'terminal_fields': 6.683759961115021e-11, 'continuation_sensor_predictions': 1.168764581529467e-10, 'profiled_amplitudes': 4.678724114009401e-12, 'joint_source_weights': 2.584366409219452e-09}
Two-study solve time: 13.739s

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6988662a973992e1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6988662a (directory).
```

### Turn 6 — `execute_in_remote_docker` (`36898 ms` LLM)
- **Rationale**: Stress the runtime bottleneck with two full-size, high-contrast cases with weaker wall and reaction terms. If repeated factorizations consume the budget, share each fallback factor across source modes and reassess the trigger against the required residual.
- **Alternatives Weighed**: `Submit after the baseline pass` (rejected: The largest baseline residual approached the 1e-8 fallback trigger. Weaker Robin and reaction terms could trigger repeated factorizations and exhaust the runtime budget.)

```bash
cat > /tmp/stress_robin_eq01.py <<'PY'
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits

# Reuse the full-grid fixtures and independent operator without rerunning baseline tests.
source = Path('/tmp/test_robin_eq01.py').read_text().split('\nwith threadpool_limits(limits=1, user_api=\'blas\'):\n    total =')[0]
exec(compile(source, '/tmp/test_robin_eq01.py', 'exec'))
original_cg, original_lu = mod.cg, mod.splu

def stress(st):
    solves, factors = [], []
    def cg(*args, **kwargs):
        u, info = original_cg(*args, **kwargs)
        solves.append(u)
        return u, info
    class Factor:
        def __init__(self, *args, **kwargs):
            self.factor = original_lu(*args, **kwargs)
            self.calls = []
        def solve(self, rhs):
            u = self.factor.solve(rhs)
            if self is not factors[0]:
                self.calls.append(u)
            return u
    def lu(*args, **kwargs):
        factor = Factor(*args, **kwargs)
        factors.append(factor)
        return factor
    mod.cg, mod.splu = cg, lu
    start = perf_counter()
    try:
        out = mod.solve_robin_continuation_tomography(st)
    finally:
        elapsed = perf_counter() - start
        mod.cg, mod.splu = original_cg, original_lu
    b = st['source_modes'].reshape(3,-1).T
    errors = []
    fallbacks = iter(factors[1:])
    for j,k in enumerate(np.argsort(st['thetas'])):
        a,v = reference(st, st['thetas'][k])
        g = mod._RobinOperator(st['kappa_base'].shape, [st['Lx'],st['Ly'],st['Lz']], st['gamma_robin'])
        t = st['thetas'][k]
        impl = g.assemble(st['kappa_base']+t*st['kappa_mod1']+t**2*st['kappa_mod2'], st['sigma_base']+t*st['sigma_mod'])
        for m in range(3):
            target = v*b[:,m]
            u = solves[3*j+m]
            if np.linalg.norm(g.volume*b[:,m]-impl@u)>1e-8*np.linalg.norm(g.volume*b[:,m]):
                f = next(fallbacks)
                u = f.calls[0] + f.calls[1]
            errors.append(np.linalg.norm(a@u-target)/np.linalg.norm(target))
    print(f'gamma={st["gamma_robin"]:.3g} time={elapsed:.3f}s state_mode_fallbacks={len(factors)-1} max_residual={max(errors):.3g}', flush=True)
    return elapsed, max(errors)

with threadpool_limits(limits=1, user_api='blas'):
    timings = []
    for i in range(2):
        st = make_study(1)
        st['gamma_robin'] = (.08, .04)[i]
        st['sigma_base'] *= .5
        st['sigma_mod'] *= .5
        if i:
            st['kappa_mod2'] = np.flip(st['kappa_mod2'], axis=0).copy()
        timings.append(stress(st))
    print('Combined weaker-wall runtime:', sum(t[0] for t in timings))
    assert max(t[1] for t in timings) <= 1e-7 and sum(t[0] for t in timings) <= 25
PY
python3 -I /tmp/stress_robin_eq01.py
```
- **Container Exit Code**: `1` (`66541 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
gamma=0.08 time=21.620s state_mode_fallbacks=4 max_residual=2.13e-08
gamma=0.04 time=42.759s state_mode_fallbacks=11 max_residual=4.39e-08
Combined weaker-wall runtime: 64.3784253578633

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=6988662a973992e1 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_6988662a (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/stress_robin_eq01.py", line 66, in <module>
    assert max(t[1] for t in timings) <= 1e-7 and sum(t[0] for t in timings) <= 25
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

### Turn 7 — `execute_in_remote_docker` (`42666 ms` LLM)
- **Rationale**: Keep the tight CG target, but allow a directly recomputed residual within the contract margin. Try correction before refactoring, and reuse any fallback factor across source modes. Verify the actual returned fields for every state against the independent operator; if the weak-wall pair still exceeds 25 s, the measured factor count will show whether to improve the preconditioner or the residual calculation.
- **Alternatives Weighed**: `Replace the shared preconditioner` (rejected: The weak-wall run triggered 15 extra factorizations; first remove redundant factors without changing the operator or tomography objective.)

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/robin_continuation_tomography.py')
s = p.read_text()
a = s.index('    fields = np.zeros_like(rhs)')
b = s.index('        predictions[k]', a)
s = s[:a] + '''    fields = np.zeros_like(rhs)
    last = np.empty_like(fields)
    for k in np.argsort(theta):
        a = matrix(theta[k])
        fields = _solve_state(a, rhs, rhs_norm, fields, inverse)
''' + s[b:]
a = s.index('\ndef _solve(study):')
s = s[:a] + '''
def _solve_state(a, rhs, rhs_norm, fields, inverse):
    direct = None
    for m, b in enumerate(rhs):
        norm = rhs_norm[m]
        if norm == 0:
            continue
        u, _ = cg(a, b, x0=fields[m], M=inverse, rtol=1e-10, maxiter=180)
        # Check the actual residual, allowing for cancellation in the nearly
        # constant solution component when reaction and wall transfer are weak.
        r = b - a @ u
        error = np.linalg.norm(r)
        if error > 7e-8 * norm:
            du, _ = cg(a, r, M=inverse, rtol=1e-3, maxiter=100)
            candidate = u + du
            improved = np.linalg.norm(b - a @ candidate)
            if improved < error:
                u, error = candidate, improved
        if error > 7e-8 * norm:
            if direct is None:
                direct = splu(a.tocsc(), permc_spec='MMD_AT_PLUS_A',
                              diag_pivot_thresh=0, options={'SymmetricMode': True})
            candidate = direct.solve(b)
            for _ in range(3):
                r = b - a @ candidate
                improved = np.linalg.norm(r)
                if improved < error:
                    u, error = candidate, improved
                if error <= 7e-8 * norm:
                    break
                candidate = candidate + direct.solve(r)
        fields[m] = u
    return fields

''' + s[a:]
p.write_text(s)
PY
python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/test_robin_eq01.py')
s = p.read_text()
a = s.index('    capture = []')
b = s.index('    elapsed = perf_counter() - start', a)
s = s[:a] + '''    capture = []
    factors = []
    real_state, real_lu = mod._solve_state, mod.splu
    def state(*args, **kwargs):
        sol = real_state(*args, **kwargs)
        capture.append(sol.copy())
        return sol
    def lu(*args, **kwargs):
        factors.append(1)
        return real_lu(*args, **kwargs)
    mod._solve_state, mod.splu = state, lu
    start = perf_counter()
    try:
        out = mod.solve_robin_continuation_tomography(st)
    finally:
        mod._solve_state, mod.splu = real_state, real_lu
''' + s[b:]
s = s.replace('np.column_stack([capture[3*j+m][0] for m in range(3)])', 'capture[j].T').replace('max_CG_iters={max(n for _,n in capture)}','fallback_factors={len(factors)-1}')
a = s.index('\nwith threadpool_limits', s.index('def run(st):'))
s = s[:a] + '''
if __name__ == '__main__':
    import sys
    weak = '--weak' in sys.argv
    def fixture(i):
        if not weak:
            return make_study(i)
        st = make_study(1)
        st['gamma_robin'] = (.08, .04)[i]
        st['sigma_base'] *= .5
        st['sigma_mod'] *= .5
        if i:
            st['kappa_mod2'] = np.flip(st['kappa_mod2'], axis=0).copy()
        return st
    with threadpool_limits(limits=1, user_api='blas'):
        total = sum(run(fixture(i)) for i in range(2))
    print(f'Two-study solve time: {total:.3f}s')
    assert total <= 25
'''
p.write_text(s)
PY
python3 -I /tmp/test_robin_eq01.py --weak
```
- **Container Exit Code**: `0` (`32635 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin_continuation_tomography.py']`
- **Stdout (tail)**:
```text
solve=7.152s fallback_factors=0 max_residual=2.69e-08 errors={'terminal_fields': 2.216271193476449e-10, 'continuation_sensor_predictions': 3.098527174384019e-10, 'profiled_amplitudes': 3.230543027734204e-11, 'joint_source_weights': 2.0838686720664963e-08}
solve=7.787s fallback_factors=0 max_residual=5.67e-08 errors={'terminal_fields': 5.40660405341695e-10, 'continuation_sensor_predictions': 6.124635306644313e-10, 'profiled_amplitudes': 5.125731933998819e-11, 'joint_source_weights': 1.1794918069265361e-08}
Two-study solve time: 14.939s

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c4e0920c6b453b57 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c4e0920c (directory).
```

### Turn 8 — `submit_completed_task` (`14133 ms` LLM)
- **Rationale**: The final artifact passed isolated-process validation against independent references for all 54 state-mode solves and all four outputs across the two weak-wall stress studies. The revised residual check eliminated extra factorizations and brought combined runtime below 25 s.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/robin_continuation_tomography.py': 'FILE:7066'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/robin_continuation_tomography.py`. Two full-size, high-contrast synthetic studies with weak Robin walls completed in 14.939 s combined, with a worst relative residual of 5.67e-8. All four outputs matched independent references within tolerance. The public fixtures were unavailable.`

### Turn 9 — `execute_in_remote_docker` (`27786 ms` LLM)
- **Rationale**: Audit two failure modes not isolated by the earlier full-size tests: an incorrect wall contribution after cancellation of interior fluxes, and loss of finite ridge energy when source and nuisance columns coincide exactly. Either failure would identify which calculation to fix; if both pass and the public fixture remains absent, the existing full-size runtime and accuracy evidence supports submission.
- **Alternatives Weighed**: `Run batch classification, relay-mode, or drift-operator audits` (rejected: This contract has continuous observations and a specified stationary elliptic stencil; it has no batch labels, relay state, or drift operator.)

```bash
if test -f /holdout_task/public_eq01_studies.pkl; then ls -lh /holdout_task/public_eq01_studies.pkl /holdout_task/validate_public_schema.py; fi
python3 -I - <<'PY'
import importlib.util
import numpy as np
from scipy import linalg

path = '/holdout_task/robin_continuation_tomography.py'
spec = importlib.util.spec_from_file_location('robin', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Interior fluxes cancel on a constant field; only reaction and wall flux remain.
shape = (3, 4, 5)
h = np.array([.21, .34, .47])
v = np.prod(h)
gamma = .63
xyz = np.indices(shape)
kappa = 1 + xyz[0] + 8*xyz[1] + 90*xyz[2]
sigma = .01 + .012*xyz[2]
a = mod._RobinOperator(shape, h*np.array(shape), gamma).assemble(kappa, sigma)
expected = v*sigma
for d in range(3):
    on_wall = (xyz[d] == 0) | (xyz[d] == shape[d]-1)
    expected += on_wall * (v/h[d]) / (1/gamma + h[d]/(2*kappa))
flux_error = np.max(np.abs(a@np.ones(kappa.size) - expected.ravel()))
assert flux_error < 1e-11, flux_error
assert (a-a.T).nnz == 0

# Exact source/nuisance overlap: the positive ridge leaves nonzero energy.
# Orthonormal columns make the projected energy and both fits analytic.
rng = np.random.default_rng(54)
z, _ = linalg.qr(rng.normal(size=(90, 3)), mode='economic')
c = rng.normal(size=(3, 4))
y = z@c
rz, rs = 3e-9, 2e-9
p = rz/(1+rz)
amp, joint = mod._profile(z, y, z, rz, rs)
amp_error = np.linalg.norm(amp-c)/np.linalg.norm(c)
want = p/(p+rs)*c
joint_error = np.linalg.norm(joint-want)/np.linalg.norm(want)
assert amp_error <= 1e-5 and joint_error <= .03, (amp_error, joint_error)
print(f'Constant-field flux max abs error={flux_error:.3g}; exact nuisance-overlap relative errors: amplitudes={amp_error:.3g}, joint={joint_error:.3g}')
PY
```
- **Container Exit Code**: `0` (`848 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Constant-field flux max abs error=1.72e-13; exact nuisance-overlap relative errors: amplitudes=2.02e-16, joint=3.11e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c4e0920c6b453b57 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c4e0920c (directory).
```

### Turn 10 — `submit_completed_task` (`16630 ms` LLM)
- **Rationale**: The final artifact passed isolated-process validation against independently assembled operators and least-squares references. The focused audit also confirmed constant-field flux balance and correct profiling when source and nuisance columns coincide.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/robin_continuation_tomography.py': 'FILE:7066'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin_continuation_tomography.py`. Two full-size, high-contrast synthetic studies completed in 14.939 s combined. The worst relative residual across all 54 solves was 5.67e-8, and all four outputs matched independent references within tolerance. The public fixtures were unavailable.`

### Turn 11 — `execute_in_remote_docker` (`27029 ms` LLM)
- **Rationale**: Test whether the shared factor remains effective when theta shifts conductivity between distinct three-dimensional paths. The independent residuals and fallback count distinguish a loss of accuracy from a runtime bottleneck. Use the public fixture instead if it has appeared; batch classification and relay modes are not part of this contract.
- **Alternatives Weighed**: `Repeat the flux and exact nuisance-overlap audits` (rejected: Both passed on the unchanged artifact. Neither probes runtime when the dominant conductivity paths change across continuation states.)

```bash
python3 -I - <<'PY'
import runpy
import numpy as np
from threadpoolctl import threadpool_limits

if __import__('os').path.isfile('/holdout_task/public_eq01_studies.pkl'):
    print('Public fixture is now available; use it for the next validation step.')
    raise SystemExit

suite = runpy.run_path('/tmp/test_robin_eq01.py')
x, y, z = np.indices((28, 28, 28))
with threadpool_limits(limits=1, user_api='blas'):
    total = 0.0
    for i in range(2):
        st = suite['make_study'](1)
        # Separate the conducting paths at low and high theta. Unlike the
        # previous planar slices, these narrow paths cross and turn in 3D.
        backbone = ((y < 4) & (z < 5)) | ((x > 22) & (z < 5)) | ((x > 22) & (y > 20))
        moving_path = (abs(x-y) <= 2) & ((abs(z-x) <= 2) if i else (abs(z+x-27) <= 2))
        st['kappa_base'] = np.where(backbone, 3500., 1.)
        st['kappa_mod1'] = np.where((z == 14) & (y > 9), 220., .01)
        st['kappa_mod2'] = np.where(moving_path, 3500., .001)
        st['gamma_robin'] = (.04, .025)[i]
        st['Lx'], st['Ly'], st['Lz'] = ((.65, 1.4, 1.9), (1.85, .68, 1.2))[i]
        total += suite['run'](st)
print(f'Two-study solve time with changing conductivity paths: {total:.3f}s')
assert total <= 25
PY
```
- **Container Exit Code**: `0` (`30912 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
solve=7.674s fallback_factors=0 max_residual=9.75e-09 errors={'terminal_fields': 2.0169310973447554e-10, 'continuation_sensor_predictions': 1.5810338386557024e-10, 'profiled_amplitudes': 1.8284060146391894e-10, 'joint_source_weights': 2.0120383456783104e-09}
solve=7.611s fallback_factors=0 max_residual=1.91e-08 errors={'terminal_fields': 3.5966268274993136e-10, 'continuation_sensor_predictions': 3.373255135259474e-10, 'profiled_amplitudes': 3.5011238565229345e-10, 'joint_source_weights': 6.577746807901475e-10}
Two-study solve time with changing conductivity paths: 15.286s

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c4e0920c6b453b57 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_c4e0920c (directory).
```

### Turn 12 — `submit_completed_task` (`14734 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process validation with independent references on weak-wall and changing-conductivity-path studies. Neither stress pair required fallback factorization, and both met the residual, output accuracy, and runtime tolerances.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/robin_continuation_tomography.py`. All four outputs matched independent references on the full-size synthetic stress studies. The slowest two-study run took 15.286 s against a 25 s limit; the worst relative residual was 5.67e-8 against a 1e-7 limit. The public fixtures were unavailable.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/robin_continuation_tomography.py": "FILE:7066"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "all_valid": true,
    "max_terminal_pde_rel_res": 1e-10,
    "max_sensor_stack_rel_l2": 0.0,
    "max_profiled_rel_l2": 0.0,
    "max_joint_weights_rel_l2": 0.0,
    "student_elapsed_sec": 7.5173,
    "instance_diagnostics": [
      {
        "seed": 99401,
        "terminal_pde_rel_res": 1e-10,
        "sensor_stack_rel_l2": 0.0,
        "profiled_rel_l2": 0.0,
        "joint_rel_l2": 0.0
      },
      {
        "seed": 99402,
        "terminal_pde_rel_res": 1e-10,
        "sensor_stack_rel_l2": 0.0,
        "profiled_rel_l2": 0.0,
        "joint_rel_l2": 0.0
      }
    ]
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_terminal_pde_rel_res\": 1e-10, \"max_sensor_stack_rel_l2\": 0.0, \"max_profiled_rel_l2\": 0.0, \"max_joint_weights_rel_l2\": 0.0, \"student_elapsed_sec\": 7.5173, \"instance_diagnostics\": [{\"seed\": 99401, \"terminal_pde_rel_res\": 1e-10, \"sensor_stack_rel_l2\": 0.0, \"profiled_rel_l2\": 0.0, \"joint_rel_l2\": 0.0}, {\"seed\": 99402, \"terminal_pde_rel_res\": 1e-10, \"sensor_stack_rel_l2\": 0.0, \"profiled_rel_l2\": 0.0, \"joint_rel_l2\": 0.0}]}\n__HEGEL_VERIFIER_AUTH_a6e04cb749e6c8f0200bc91f6b623606__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_terminal_pde_rel_res\": 1e-10, \"max_sensor_stack_rel_l2\": 0.0, \"max_profiled_rel_l2\": 0.0, \"max_joint_weights_rel_l2\": 0.0, \"student_elapsed_sec\": 7.5173, \"instance_diagnostics\": [{\"seed\": 99401, \"terminal_pde_rel_res\": 1e-10, \"sensor_stack_rel_l2\": 0.0, \"profiled_rel_l2\": 0.0, \"joint_rel_l2\": 0.0}, {\"seed\": 99402, \"terminal_pde_rel_res\": 1e-10, \"sensor_stack_rel_l2\": 0.0, \"profiled_rel_l2\": 0.0, \"joint_rel_l2\": 0.0}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 15190.44,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
