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
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `377,514` (`364,422` / `107,520` / `13,092`) |
| **Total Episode Wall Time (s)** | `845.61s` |

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

### Turn 1 — `execute_in_remote_docker` (`16464 ms` LLM)
- **Rationale**: Inspect the public validator and numerical environment before choosing a reusable factorization or multigrid implementation. The available libraries and BLAS threading affect both the implementation and the runtime test.
- **Alternatives Weighed**: `Factor every continuation operator separately` (rejected: The contract reports that this exceeds the two-study time budget.)

```bash
ls -lh /holdout_task; sed -n '1,240p' /holdout_task/validate_public_schema.py; if test -f /tmp/hegel_knowledge_ref/describe_ref.py; then python3 /tmp/hegel_knowledge_ref/describe_ref.py --search 'SPD algebraic multigrid CG preconditioner'; python3 /tmp/hegel_knowledge_ref/describe_ref.py --search 'ridge nuisance profiling'; fi; python3 - <<'PY'
import importlib.util, os, scipy, numpy as np
print('scipy', scipy.__version__, 'pyamg', importlib.util.find_spec('pyamg'), 'threadpoolctl', importlib.util.find_spec('threadpoolctl'))
print('thread env', {k:v for k,v in os.environ.items() if 'THREAD' in k}); np.show_config()
PY
```
- **Container Exit Code**: `0` (`928 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_robust_affine_projection.profile_weighted_affine
scipy 1.13.0 pyamg None threadpoolctl ModuleSpec(name='threadpoolctl', loader=<_frozen_importlib_external.SourceFileLoader object at 0x7753d3490da0>, origin='/usr/local/lib/python3.12/dist-packages/threadpoolctl.py')
thread env {}
{
  "Compilers": {
    "c": {
      "name": "gcc",
      "linker": "ld.bfd",
      "version": "10.2.1",
      "commands": "cc",
      "args": "-fno-strict-aliasing",
      "linker args": "-Wl,--strip-debug, -fno-strict-aliasing"
    },
    "cython": {
      "name": "cython",
      "linker": "cython",
      "version": "3.0.8",
      "commands": "cython"
    },
    "c++": {
      "name": "gcc",
      "linker": "ld.bfd",
      "version": "10.2.1",
      "commands": "c++",
      "linker args": "-Wl,--strip-debug"
    }
  },
  "Machine Information": {
    "host": {
      "cpu": "x86_64",
      "family": "x86_64",
      "endian": "little",
      "system": "linux"
    },
    "build": {
      "cpu": "x86_64",
      "family": "x86_64",
      "endian": "little",
      "system": "linux"
    }
  },
  "Build Dependencies": {
    "blas": {
      "name": "openblas64",
      "found": true,
      "version": "0.3.23.dev",
      "detection method": "pkgconfig",
      "include directory": "/usr/local/include",
      "lib directory": "/usr/local/lib",
      "openblas configuration": "USE_64BITINT=1 DYNAMIC_ARCH=1 DYNAMIC_OLDER= NO_CBLAS= NO_LAPACK= NO_LAPACKE= NO_AFFINITY=1 USE_OPENMP= HASWELL MAX_THREADS=2",
      "pc file directory": "/usr/local/lib/pkgconfig"
    },
    "lapack": {
      "name": "dep140551260102944",
      "found": true,
      "version": "1.26.4",
      "detection method": "internal",
      "include directory": "unknown",
      "lib directory": "unknown",
      "openblas configuration": "unknown",
      "pc file directory": "unknown"
    }
  },
  "Python Information": {
    "path": "/opt/python/cp312-cp312/bin/python",
    "version": "3.12"
  },
  "SIMD Extensions": {
    "baseline": [
      "SSE",
      "SSE2",
      "SSE3"
    ],
    "found": [
      "SSSE3",
      "SSE41",
      "POPCNT",
      "SSE42",
      "AVX",
      "F16C",
      "FMA3",
      "AVX2",
      "AVX512F",
      "AVX512CD",
      "AVX512_SKX",
      "AVX512_CLX",
      "AVX512_CNL",
      "AVX512_ICL"
    ],
    "not found": [
      "AVX512_KNL",
      "AVX512_KNM"
    ]
  }
}
```
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task': No such file or directory
sed: can't read /holdout_task/validate_public_schema.py: No such file or directory
/usr/local/lib/python3.12/dist-packages/numpy/__config__.py:155: UserWarning: Install `pyyaml` for better output
  warnings.warn("Install `pyyaml` for better output", stacklevel=1)
```

### Turn 2 — `execute_in_remote_docker` (`17310 ms` LLM)
- **Rationale**: Determine whether the public data are mounted elsewhere and inspect the reference methods. If the study is absent, use independently assembled operators and controlled high-contrast cases to validate the implementation.
- **Alternatives Weighed**: `Assume the public study is unavailable and move straight to synthetic tests` (rejected: The first command only checked the advertised path; a bounded search can still recover the supplied study without transferring data.)

```bash
pwd; ls -la / /pretrain_task; find /pretrain_task /holdout_task /workspace /app -maxdepth 3 -type f \( -name '*eq01*' -o -name 'validate_public_schema.py' \) -print 2>/dev/null; mkdir -p /holdout_task; if test -f /tmp/hegel_knowledge_ref/describe_ref.py; then for f in autogenous_prim_conservative_spd_solve.galerkin_preconditioner autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes; do python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe "$f"; done; find /tmp/hegel_knowledge_ref -iname '*reference.txt' | grep -E 'conservative_spd|ridge_profiled'; fi; cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
```
- **Container Exit Code**: `0` (`486 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
SPD flux operator).",
    "operator must be square symmetric positive-definite (or consistent on orthogonal complement) for conjugate-gradient convergence."
  ],
  "function_level_applicability": {
    "conservative_grid_operator": "MISMATCHED_BOUNDARY_OR_TENSOR(zero_ghost_scalar_dirichlet_only)",
    "galerkin_preconditioner": "APPLICABLE_SUBROUTINE(requires_assembled_csr_spd_matrix)",
    "solve_residual_checked_cg": "APPLICABLE_SUBROUTINE(requires_spd_operator_or_matvec)"
  },
  "functions": [
    {
      "function": "galerkin_preconditioner",
      "signature": "galerkin_preconditioner(operator: sparse.spmatrix | sparse.sparray, *, strength_cutoff: float=0.08, max_coarse_fraction: float=0.76, coarse_size: int=64, seed: int=12) -> LinearOperator",
      "doc_summary": "Map a real sparse SPD matrix of shape (n, n) to a symmetric (n, n) V-cycle approximate inverse using smoothed aggregates and Galerkin coarse operators."
    }
  ],
  "reference_source_txt": "/tmp/hegel_knowledge_ref/autogenous_prim_conservative_spd_solve.reference.txt"
}
{
  "ok": true,
  "module_stem": "autogenous_prim_ridge_profiled_matching",
  "compiled_code_sha256": "dfdf7838f3618f600b4d7d7df948af526e9de4b41cbe0e429a9a76032e691078",
  "mathematical_preconditions": [
    "design (n, p) and values (n,) or (n, q) share leading axis n; ridge >= 0 finite scalar or (p,) vector.",
    "whitener (r, p) must match the SVD Gram whitener of the same weighted design matrix; candidate_energy >= 0."
  ],
  "function_level_applicability": {
    "fit_ridge_nuisance": "APPLICABLE_SUBROUTINE(real_design_and_observations_ridge_whitener)",
    "profile_candidate_amplitudes": "APPLICABLE_SUBROUTINE(real_schur_complement_amplitude_profiling)"
  },
  "functions": [
    {
      "function": "profile_candidate_amplitudes",
      "signature": "profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray]",
      "doc_summary": "Profile residual inner products (m,), raw energies (m,), and nuisance cross-products (p, m) using a whitener (r, p), returning amplitudes and remaining energies, both (m,)."
    }
  ],
  "reference_source_txt": "/tmp/hegel_knowledge_ref/autogenous_prim_ridge_profiled_matching.reference.txt"
}
/tmp/hegel_knowledge_ref/autogenous_prim_ridge_profiled_matching.reference.txt
/tmp/hegel_knowledge_ref/autogenous_prim_conservative_spd_solve.reference.txt
200000 100000
```

### Turn 3 — `execute_in_remote_docker` (`61482 ms` LLM)
- **Rationale**: Implement the full contract with one reusable factor and avoid cancellation in the ridge Schur complement. Compilation detects syntax failures; the next step is to measure runtime and independently check finite-volume residuals, state ordering, and both tomography outputs.
- **Alternatives Weighed**: `Implement a smoothed-aggregation V-cycle immediately` (rejected: A single reference factor is simpler and resolves high-contrast structure at the reference exactly; full-size timing will determine whether a cheaper preconditioner is needed.)

```bash
set -e
sed -n '/^def galerkin_preconditioner/,/^def solve_residual_checked_cg/p' /tmp/hegel_knowledge_ref/autogenous_prim_conservative_spd_solve.reference.txt | head -95
sed -n '/^def fit_ridge_nuisance/,/^def profile_candidate_amplitudes/p' /tmp/hegel_knowledge_ref/autogenous_prim_ridge_profiled_matching.reference.txt | head -65
mkdir -p /holdout_task
cat > /holdout_task/robin_continuation_tomography.py <<'PY'
"""Finite-volume Robin continuation and nuisance-profiled tomography."""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, cg, splu


def _assembler(shape, spacing, gamma):
    ids = np.arange(np.prod(shape)).reshape(shape)
    volume = np.prod(spacing)
    faces = []
    walls = []
    for axis, h in enumerate(spacing):
        grid = ids.swapaxes(0, axis)
        faces.append((grid[:-1].ravel(), grid[1:].ravel(), volume / h**2))
        walls.append((grid[[0, -1]].ravel(), volume / h, gamma * h / 2))
    i = np.concatenate([f[0] for f in faces])
    j = np.concatenate([f[1] for f in faces])
    d = ids.ravel()
    rows = np.concatenate((d, i, j))
    cols = np.concatenate((d, j, i))

    def assemble(kappa, sigma):
        k = kappa.ravel()
        w = np.concatenate([scale * (2 * k[a] * k[b] / (k[a] + k[b]))
                            for a, b, scale in faces])
        diag = volume * sigma.ravel() + np.bincount(i, w, len(d)) + np.bincount(j, w, len(d))
        for wall, area, half in walls:
            np.add.at(diag, wall, area * gamma * k[wall] / (half + k[wall]))
        return sparse.csr_matrix((np.concatenate((diag, -w, -w)), (rows, cols)),
                                 shape=(len(d), len(d)))

    return assemble, volume


def _profile(S, Z, Y, lam_z, lam_s):
    # The residual maker for [Z; sqrt(lam_z) I] is an orthogonal
    # projector. Apply it to [S Y; 0] and take inner products without
    # subtracting nearly equal Gram matrices.
    U, t, _ = linalg.svd(Z, full_matrices=False)
    s, y = U.T @ S, U.T @ Y
    ratio = np.sqrt(lam_z) / np.hypot(t, np.sqrt(lam_z))
    tail_s = ratio[:, None] * s
    tail_y = ratio[:, None] * y
    perp_s = S - U @ s
    perp_y = Y - U @ y
    gram = perp_s.T @ perp_s + tail_s.T @ tail_s
    cross = perp_s.T @ perp_y + tail_s.T @ tail_y
    energy = np.diag(gram)[:, None]
    amplitudes = np.divide(cross, energy, out=np.zeros_like(cross), where=energy > 0)
    joint = linalg.solve(gram + lam_s * np.eye(S.shape[1]), cross, assume_a='pos')
    return amplitudes, joint


def _iterate(A, b, preconditioner, guess):
    # Independently check the true residual; recursive CG residuals can
    # drift on high-contrast conductivity fields.
    scale = np.sqrt(b @ b)
    if not scale:
        return np.zeros_like(b)
    x, _ = cg(A, b, x0=guess, M=preconditioner, rtol=2e-10, maxiter=220)
    r = b - A @ x
    if np.sqrt(r @ r) > 8e-9 * scale:
        correction, _ = cg(A, r, M=preconditioner, atol=2e-10 * scale,
                           rtol=0, maxiter=300)
        x += correction
        r = b - A @ x
        if np.sqrt(r @ r) > 5e-8 * scale:
            x = _factor(A).solve(b)
    return x


def _factor(A):
    return splu(A.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0,
                options={'SymmetricMode': True})


def solve_robin_continuation_tomography(study: dict) -> dict:
    """Solve the parametric Robin family and return the four tomography arrays."""
    kb, k1, k2, sb, s1 = (np.asarray(study[name], dtype=float) for name in
                           ('kappa_base', 'kappa_mod1', 'kappa_mod2', 'sigma_base', 'sigma_mod'))
    shape = kb.shape
    thetas = np.asarray(study['thetas'], dtype=float)
    spacing = np.array([study['Lx'], study['Ly'], study['Lz']]) / shape
    assemble, volume = _assembler(shape, spacing, study['gamma_robin'])

    def operator(theta):
        return assemble(kb + theta * k1 + theta**2 * k2, sb + theta * s1)

    # Positive kappa polynomial terms bound the conductance ratios on
    # either side of this geometric center, including on Robin faces.
    reference = np.sqrt(thetas.min() * thetas.max())
    factor = _factor(operator(reference))
    preconditioner = LinearOperator(factor.shape, matvec=factor.solve, dtype=float)
    rhs = (volume * np.asarray(study['source_modes'], dtype=float)).reshape(-1, kb.size)
    sensor = np.ravel_multi_index(np.asarray(study['sensor_ijk']).T, shape)
    S = np.empty((len(thetas), len(sensor), len(rhs)))
    terminal = None
    guess = [None] * len(rhs)
    for k in np.argsort(thetas):
        A = operator(thetas[k])
        fields = np.array([_iterate(A, b, preconditioner, x) for b, x in zip(rhs, guess)])
        S[k] = fields[:, sensor].T
        guess = fields
        if k == len(thetas) - 1:
            terminal = fields.reshape((-1,) + shape)
    S = S.reshape(-1, len(rhs))
    amplitudes, joint = _profile(S, np.asarray(study['Z_nuisance'], dtype=float),
                                 np.asarray(study['Y_obs'], dtype=float),
                                 study['nuisance_ridge'], study['source_ridge'])
    return dict(terminal_fields=terminal, continuation_sensor_predictions=S,
                profiled_amplitudes=amplitudes, joint_source_weights=joint)
PY
python3 -I -m py_compile /holdout_task/robin_continuation_tomography.py
```
- **Container Exit Code**: `0` (`380 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin_continuation_tomography.py']`
- **Stdout (tail)**:
```text
els.append((a, p, pt, inv))
        ac = pt @ (a @ p)
        a = (0.5 * ac + 0.5 * ac.T).tocsr()
    else:
        factor = linalg.cho_factor(a.toarray())
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


def solve_residual_checked_cg(operator: np.ndarray | sparse.spmatrix | sparse.sparray | LinearOperator, rhs: np.ndarray, *, preconditioner: LinearOperator | None=None, tol: float=1e-08, maxiter: int=450, refinements: int=4) -> tuple[np.ndarray, float]:
def fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and return coefficients (p,) or (p, q), same-shape residuals, and an (r, p) Gram whitener."""
    design = np.asarray(design)
    values = np.asarray(values)
    if design.ndim != 2 or values.ndim not in (1, 2) or len(values) != len(design):
        raise ValueError('Design and values must share their observation axis')
    (n, p) = design.shape
    penalty = np.broadcast_to(np.asarray(ridge, dtype=float), (p,))
    if np.any(~np.isfinite(penalty)) or np.any(penalty < 0):
        raise ValueError('Ridge penalties must be finite and nonnegative')
    augmented = np.vstack((design, np.diag(np.sqrt(penalty))))
    (u, s, vh) = np.linalg.svd(augmented, full_matrices=False)
    cutoff = np.finfo(s.dtype).eps * max(augmented.shape) * s.max(initial=0.0)
    keep = s > cutoff
    whitener = vh[keep] / s[keep, None]
    coefficients = whitener.conj().T @ (u[:n, keep].conj().T @ values)
    return (coefficients, values - design @ coefficients, whitener)


def profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray]:

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=312ff60c9dd0f7c3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_312ff60c (directory).
```

### Turn 4 — `execute_in_remote_docker` (`62340 ms` LLM)
- **Rationale**: Use an independent face-incidence assembly to check every state's true residual, separate direct solves and augmented least squares to check tomography, and time the solver alone on two full-size contrast patterns. The results will distinguish a discretization or profiling defect from a preconditioner performance bottleneck.
- **Alternatives Weighed**: `Verify only the output schema` (rejected: Shapes alone would not detect incorrect Robin fluxes, continuation ordering, profiling errors, or a runtime overrun.)

```bash
cat > /tmp/test_robin_eq01.py <<'PY'
import importlib.util
import pickle
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu

path = Path('/holdout_task/robin_continuation_tomography.py')
spec = importlib.util.spec_from_file_location('robin', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def generate(seed, shape, kind):
    rng = np.random.default_rng(seed)
    x,y,z = np.meshgrid(*(np.linspace(-1,1,n) for n in shape), indexing='ij')
    if kind == 0:
        k = 0.03 * 3500**rng.random(shape)
        k1 = 0.2*k*rng.random(shape)
        k2 = k*rng.random(shape)
        gamma, reaction = 0.9, 0.08
    else:
        k = np.where(((x+0.3*z)**2+(y-0.15)**2 < 0.3) | (abs(z)<0.067),105.,0.03)
        k1 = np.where(y>0,0.4,0.001)
        k2 = np.where(x>0,18.,0.0001)
        gamma, reaction = 0.004, 0.00015
    theta = np.array([0.82,0.15,1.7,0.24,1.02,1.85,0.4,1.31,0.59])
    sensors = np.column_stack([rng.integers(n,size=100) for n in shape])
    Z = rng.normal(size=(900,10)); Z[:,5] = Z[:,2]+Z[:,3]
    Z[:,0] = 1.
    Y = rng.normal(size=(900,6)) + Z @ rng.normal(size=(10,6))
    return dict(Lx=1.25,Ly=0.8,Lz=1.7,kappa_base=k,kappa_mod1=k1,kappa_mod2=k2,
                sigma_base=reaction*(1+rng.random(shape)),sigma_mod=reaction*rng.random(shape),gamma_robin=gamma,
                thetas=theta,source_modes=np.array([1+0.3*x,np.exp(-7*((x+0.2)**2+y*y+z*z)),(y+0.4*z)*np.cos(3*x)]),
                sensor_ijk=sensors,Z_nuisance=Z,Y_obs=Y,nuisance_ridge=1e-4,source_ridge=0.001)

def oracle_operator(s, theta):
    k=s['kappa_base']+theta*s['kappa_mod1']+theta**2*s['kappa_mod2']
    sig=s['sigma_base']+theta*s['sigma_mod']; g=s['gamma_robin']
    h=np.array([s['Lx'],s['Ly'],s['Lz']])/k.shape
    v=h.prod(); ids=np.arange(k.size).reshape(k.shape)
    diag=v*sig.copy(); weights=[]; left=[]; right=[]
    for axis in range(3):
        K=np.moveaxis(k,axis,0); I=np.moveaxis(ids,axis,0); D=np.moveaxis(diag,axis,0)
        face=(v/h[axis])/(0.5*h[axis]*(1/K[:-1]+1/K[1:]))
        weights.append(face.ravel()); left.append(I[:-1].ravel()); right.append(I[1:].ravel())
        D[0] += (v/h[axis])/(1/g+0.5*h[axis]/K[0]); D[-1] += (v/h[axis])/(1/g+0.5*h[axis]/K[-1])
    f=np.concatenate(weights); i=np.concatenate(left); j=np.concatenate(right); r=np.arange(len(f))
    incidence=sparse.csr_matrix((np.tile([1.,-1.],(len(f),1)).ravel(), (np.repeat(r,2), np.column_stack((i,j)).ravel())),shape=(len(f),k.size))
    return (incidence.T @ incidence.multiply(f[:,None]) + sparse.diags(diag.ravel())).tocsc(), v

def error(actual, expected):
    return np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-100)

def oracle_profile(s, S):
    Z,Y=s['Z_nuisance'],s['Y_obs']; q=Z.shape[1]; p=S.shape[1]
    rz=np.sqrt(s['nuisance_ridge']); rs=np.sqrt(s['source_ridge'])
    az=np.vstack((Z,rz*np.eye(q))); ay=np.vstack((Y,np.zeros((q,Y.shape[1]))))
    mode=[]
    for x in S.T:
        design=np.column_stack((az,np.r_[x,np.zeros(q)]))
        mode.append(linalg.lstsq(design,ay,lapack_driver='gelsy')[0][-1])
    design=np.block([[Z,S],[rz*np.eye(q),np.zeros((q,p))],[np.zeros((p,q)),rs*np.eye(p)]])
    rhs=np.vstack((ay,np.zeros((p,Y.shape[1]))))
    return np.array(mode),linalg.lstsq(design,rhs,lapack_driver='gelsy')[0][-p:]

def test(s, label, exact):
    # Capture internal solves so residuals can be checked for nonterminal states, too.
    orig = m._iterate; xs = []
    def save(A,b,M,x):
        field=orig(A,b,M,x); xs.append(field); return field
    m._iterate = save
    start=perf_counter()
    try: out=m.solve_robin_continuation_tomography(s)
    finally: m._iterate=orig
    sec=perf_counter()-start
    ks=s['thetas']; p=len(s['source_modes']); sz=s['kappa_base'].shape
    xs=np.asarray(xs).reshape(len(ks),p,-1)
    order=np.argsort(ks); sensors=tuple(s['sensor_ijk'].T); pred=[None]*len(ks); ref=[None]*len(ks)
    res=[]
    for fields,k in zip(xs,order):
        A,v=oracle_operator(s,ks[k]); b=(v*s['source_modes']).reshape(p,-1).T
        res.extend(np.linalg.norm(A@fields.T-b,axis=0)/np.maximum(np.linalg.norm(b,axis=0),1e-100))
        pred[k]=np.column_stack([u.reshape(sz)[sensors] for u in fields]);
        if exact:
            sol=splu(A,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0,options={'SymmetricMode':True}).solve(b)
            ref[k]=np.column_stack([u.reshape(sz)[sensors] for u in sol.T]);
    assert set(out)=={'terminal_fields','continuation_sensor_predictions','profiled_amplitudes','joint_source_weights'}
    np.testing.assert_array_equal(out['terminal_fields'],xs[np.flatnonzero(order==len(ks)-1)[0]].reshape((p,)+sz))
    S=np.vstack(pred)
    np.testing.assert_array_equal(out['continuation_sensor_predictions'],S)
    truth=np.vstack(ref) if exact else S
    a,w=oracle_profile(s,truth)
    ea,ew=error(out['profiled_amplitudes'],a),error(out['joint_source_weights'],w)
    print(label, f'{sec:.3f}s worst residual={max(res):.3g} sensor L2={error(S,truth):.3g} amplitude L2={ea:.3g} joint L2={ew:.3g}', flush=True)
    assert max(res)<=1e-7 and ea<=1e-5 and ew<=0.03
    return sec

if __name__ == '__main__':
    public=Path('/holdout_task/public_eq01_studies.pkl')
    if public.exists():
        data=pickle.loads(public.read_bytes()); print('Public pickle type:',type(data), list(data)[:8] if isinstance(data,dict) else len(data))
    else:
        print('Public study unavailable; testing synthetic studies with independent discretization and augmented QR oracles.',flush=True)
    test(generate(741,(5,4,6),1),'small anisotropic',True)
    t1=test(generate(975,(28,28,28),0),'3500x cell-scale contrast',True)
    t2=test(generate(741,(28,28,28),1),'3500x channels, weak walls, shifting coefficients',True)
    print(f'Full-size pair runtime: {t1+t2:.3f} seconds',flush=True)
    assert t1+t2 <= 25
PY
python3 -I /tmp/test_robin_eq01.py
```
- **Container Exit Code**: `124` (`180106 ms`, timed_out=`True`)
- **Stdout (tail)**:
```text
Public study unavailable; testing synthetic studies with independent discretization and augmented QR oracles.
small anisotropic 0.028s worst residual=1.98e-10 sensor L2=1.81e-12 amplitude L2=9.11e-11 joint L2=9.82e-10
3500x cell-scale contrast 60.477s worst residual=1.74e-10 sensor L2=1.52e-13 amplitude L2=8.82e-13 joint L2=1.81e-12

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=312ff60c9dd0f7c3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_312ff60c (directory).
```
- **Stderr (tail)**:
```text
[TIMEOUT: command exceeded 180s limit]
```

### Turn 5 — `execute_in_remote_docker` (`21498 ms` LLM)
- **Rationale**: Time factorization and CG separately under default and single-threaded BLAS. If threading dominates, replace the repeated vector reductions; if factorization or triangular solves dominate, switch to a cheaper reusable preconditioner.
- **Alternatives Weighed**: `Replace the reference factor with multigrid immediately` (rejected: If repeated BLAS reductions dominate, a different preconditioner alone will not fix the bottleneck; timings can distinguish the two.)

```bash
cat > /tmp/profile_robin_eq01.py <<'PY'
import runpy, sys
from time import perf_counter
ns=runpy.run_path('/tmp/test_robin_eq01.py'); m=ns['m']; gen=ns['generate']
from threadpoolctl import threadpool_info
print(threadpool_info(), flush=True)
for kind in map(int,sys.argv[1:]):
    calls={k:[] for k in ('_factor','cg','_profile')}
    def wrap(name, fn):
        def tagged(*a,**kw):
            start=perf_counter(); out=fn(*a,**kw); sec=perf_counter()-start
            calls[name].append(sec)
            if name=='_factor': print(f'factor {sec:.3f}s',flush=True)
            return out
        return tagged
    saved={name:getattr(m,name) for name in calls}
    for name,fn in saved.items(): setattr(m,name,wrap(name,fn))
    s=gen(975+kind,(28,28,28),kind)
    start=perf_counter(); m.solve_robin_continuation_tomography(s)
    print(kind,perf_counter()-start,{k:(len(v),sum(v),max(v,default=0)) for k,v in calls.items()},flush=True)
    for name,fn in saved.items(): setattr(m,name,fn)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/profile_robin_eq01.py 0 1; timeout 72s python3 -I /tmp/profile_robin_eq01.py 0
```
- **Container Exit Code**: `0` (`67770 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[{'user_api': 'blas', 'internal_api': 'openblas', 'num_threads': 1, 'prefix': 'libopenblas', 'filepath': '/usr/local/lib/python3.12/dist-packages/numpy.libs/libopenblas64_p-r0-0cf96a72.3.23.dev.so', 'version': '0.3.23.dev', 'threading_layer': 'pthreads', 'architecture': 'Cooperlake'}, {'user_api': 'blas', 'internal_api': 'openblas', 'num_threads': 1, 'prefix': 'libopenblas', 'filepath': '/usr/local/lib/python3.12/dist-packages/scipy.libs/libopenblasp-r0-24bff013.3.26.dev.so', 'version': '0.3.26.dev', 'threading_layer': 'pthreads', 'architecture': 'Cooperlake'}]
factor 0.830s
0 4.3015487680677325 {'_factor': (1, 0.8302131581585854, 0.8302131581585854), 'cg': (27, 3.411420145770535, 0.2373506468720734), '_profile': (1, 0.0007017338648438454, 0.0007017338648438454)}
factor 1.034s
1 6.813020583009347 {'_factor': (1, 1.0343032409437, 1.0343032409437), 'cg': (27, 5.732537126401439, 0.361280718119815), '_profile': (1, 0.00046790391206741333, 0.00046790391206741333)}
[{'user_api': 'blas', 'internal_api': 'openblas', 'num_threads': 32, 'prefix': 'libopenblas', 'filepath': '/usr/local/lib/python3.12/dist-packages/numpy.libs/libopenblas64_p-r0-0cf96a72.3.23.dev.so', 'version': '0.3.23.dev', 'threading_layer': 'pthreads', 'architecture': 'Cooperlake'}, {'user_api': 'blas', 'internal_api': 'openblas', 'num_threads': 32, 'prefix': 'libopenblas', 'filepath': '/usr/local/lib/python3.12/dist-packages/scipy.libs/libopenblasp-r0-24bff013.3.26.dev.so', 'version': '0.3.26.dev', 'threading_layer': 'pthreads', 'architecture': 'Cooperlake'}]
factor 0.880s
0 55.46190485986881 {'_factor': (1, 0.8803624080028385, 0.8803624080028385), 'cg': (27, 53.26894504856318, 3.3995982178021222), '_profile': (1, 0.0006651049479842186, 0.0006651049479842186)}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=312ff60c9dd0f7c3 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_312ff60c (directory).
```

### Turn 6 — `execute_in_remote_docker` (`29539 ms` LLM)
- **Rationale**: Default-thread CG took 53.269 s versus 3.411 s with one BLAS thread on the same study, while reference factorization stayed under 0.9 s. Preserve the factor and replace the repeated reductions, then verify the edited artifact under default settings against independent residual, ordering, and tomography checks.
- **Alternatives Weighed**: `Limit BLAS threads for the whole solve` (rejected: Serial reductions remove the measured overhead without an extra dependency or changes to process-wide thread settings.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/robin_continuation_tomography.py')
s = p.read_text().replace('from scipy.sparse.linalg import LinearOperator, cg, splu', 'from scipy.sparse.linalg import LinearOperator, splu')
start = s.index('def _iterate(')
s = s[:start] + '''def _dot(x, y):
    # A serial reduction avoids BLAS thread overhead on these short vectors.
    return np.einsum('i,i->', x, y)


def _pcg(A, b, M, guess, tolerance, maxiter):
    x = np.zeros_like(b) if guess is None else guess.copy()
    r = b.copy() if guess is None else b - A @ x
    threshold = tolerance**2
    if _dot(r, r) <= threshold:
        return x
    p = M @ r
    rho = _dot(r, p)
    for _ in range(maxiter):
        Ap = A @ p
        alpha = rho / _dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        if _dot(r, r) <= threshold:
            break
        z = M @ r
        next_rho = _dot(r, z)
        p *= next_rho / rho
        p += z
        rho = next_rho
    return x


''' + s[start:]
s = s.replace('scale = np.sqrt(b @ b)', 'scale = np.sqrt(_dot(b, b))')
s = s.replace('x, _ = cg(A, b, x0=guess, M=preconditioner, rtol=2e-10, maxiter=220)', 'x = _pcg(A, b, preconditioner, guess, 2e-10 * scale, 220)')
s = s.replace('np.sqrt(r @ r)', 'np.sqrt(_dot(r, r))')
s = s.replace('correction, _ = cg(A, r, M=preconditioner, atol=2e-10 * scale,\n                           rtol=0, maxiter=300)\n        x += correction', 'x += _pcg(A, r, preconditioner, None, 2e-10 * scale, 300)')
p.write_text(s)
PY
python3 -I /tmp/test_robin_eq01.py
```
- **Container Exit Code**: `0` (`25976 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin_continuation_tomography.py']`
- **Stdout (tail)**:
```text
Public study unavailable; testing synthetic studies with independent discretization and augmented QR oracles.
small anisotropic 0.016s worst residual=1.98e-10 sensor L2=2.36e-12 amplitude L2=2.51e-10 joint L2=5.83e-10
3500x cell-scale contrast 3.470s worst residual=1.74e-10 sensor L2=1.52e-13 amplitude L2=9.02e-13 joint L2=1.83e-12
3500x channels, weak walls, shifting coefficients 6.235s worst residual=4.65e-09 sensor L2=3.15e-10 amplitude L2=3.24e-08 joint L2=1.55e-07
Full-size pair runtime: 9.705 seconds

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=37599c55e73a52d7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_37599c55 (directory).
```

### Turn 7 — `submit_completed_task` (`14735 ms` LLM)
- **Rationale**: The final artifact passed clean-process checks for all four outputs, including continuation ordering, independent residuals for every solve, and tomography against augmented least-squares references. All measured errors and the combined runtime were within contract limits.
- **Alternatives Weighed**: `Replace the reusable factor with multigrid` (rejected: The verified implementation meets the two-study runtime budget with over 15 seconds to spare on the available stress cases.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/robin_continuation_tomography.py': 'FILE:5460'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/robin_continuation_tomography.py`. Two full-size synthetic stress studies ran in 9.705 s combined, with a worst relative residual of 4.65e-9. Profiled amplitudes and joint source weights passed independent reference checks. The public study was unavailable in the container.`

### Turn 8 — `execute_in_remote_docker` (`39611 ms` LLM)
- **Rationale**: Check the public study if it is mounted. Otherwise, use targeted fixtures: row sums detect incorrect reaction or Robin wall losses, nonuniform coefficient shifts stress reference-factor reuse, and augmented least squares detects cancellation when the sensor signal nearly lies in the nuisance span. Revise only the component whose check fails.
- **Alternatives Weighed**: `Test relay modes or within-batch classification accuracy` (rejected: This contract has neither relay states nor batch labels; near-alignment between source and nuisance columns is the relevant confounding test.)

```bash
set -e
cat > /tmp/audit_robin_eq01.py <<'PY'
import pickle
from pathlib import Path
import runpy

import numpy as np

n = runpy.run_path('/tmp/test_robin_eq01.py')
m, test, generate = (n[k] for k in ('m','test','generate'))
public = Path('/holdout_task/public_eq01_studies.pkl')
if public.is_file():
    def studies(obj):
        if isinstance(obj, dict):
            if 'kappa_base' in obj and 'Y_obs' in obj:
                yield obj
            else:
                for value in obj.values():
                    yield from studies(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                yield from studies(value)
    found = list(studies(pickle.loads(public.read_bytes())))
    assert found, 'Public file is present but its study layout was not recognized'
    for i, s in enumerate(found):
        test(s, f'public study {i}', True)
else:
    print('Public study still not mounted; audit uses constructed fixtures.', flush=True)

pair = 0.
for kind in (0, 1):
    s = generate(242+kind, (28,28,28), kind)
    x,y,z = np.indices((28,28,28))
    mask = (x+2*y+z)%9 < 4 if not kind else (x+y//4)%12 < 5
    kb = s['kappa_base']
    s['kappa_mod1'] = 0.001*kb
    s['kappa_mod2'] = np.where(mask, 15.*kb, 0.001*kb)
    s['thetas'] = s['thetas'][[5,0,8,2,7,3,6,4,1] if kind else [1,0,8,2,7,3,6,4,5]]
    # The row sum is the integrated reaction and wall loss, independent of interior faces.
    theta = s['thetas'][-1]
    h = np.array([s['Lx'], s['Ly'], s['Lz']])/kb.shape
    A, v = m._assembler(kb.shape, h, s['gamma_robin'])
    kappa = kb+theta*s['kappa_mod1']+theta**2*s['kappa_mod2']
    sigma = s['sigma_base']+theta*s['sigma_mod']
    g = s['gamma_robin']
    expected = v*sigma.copy()
    for axis in range(3):
        for side in (0,-1):
            k = np.take(kappa, side, axis=axis)
            np.swapaxes(expected,0,axis)[side] += (v/h[axis])/(1/g+0.5*h[axis]/k)
    loss_error = np.max(np.abs(A(kappa,sigma)@np.ones(kb.size)-expected.ravel()))
    assert loss_error < 1e-10, loss_error
    print(f'case {kind}: max cell conservation defect {loss_error:.3g}', flush=True)
    pair += test(s, f'strongly nonuniform coefficient shift {kind}', True)
print(f'Shifted two-study runtime: {pair:.3f} s', flush=True)
assert pair <= 25.

# Near alignment with a rank-deficient nuisance basis is where subtracting
# unprojected Gram matrices can erase the one-mode denominator.
rng = np.random.default_rng(31)
Z = rng.normal(size=(900,8))
Z[:,-1] = Z[:,0]+2*Z[:,3]
S = Z[:,:3] + 1e-5*rng.normal(size=(900,3))
Y = Z @ rng.normal(size=(8,4)) + S @ rng.normal(size=(3,4)) + 1e-5*rng.normal(size=(900,4))
s = dict(Z_nuisance=Z,Y_obs=Y,nuisance_ridge=1e-10,source_ridge=1e-8)
a,w = m._profile(S,Z,Y,s['nuisance_ridge'],s['source_ridge'])
aa,ww = n['oracle_profile'](s,S)
ea,ew = n['error'](a,aa),n['error'](w,ww)
print(f'Near-confounded nuisance: amplitude L2={ea:.3g}, joint L2={ew:.3g}', flush=True)
assert ea <= 1e-5 and ew <= 0.03
PY
python3 -I /tmp/audit_robin_eq01.py
```
- **Container Exit Code**: `1` (`1004 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public study still not mounted; audit uses constructed fixtures.

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=37599c55e73a52d7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_37599c55 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/audit_robin_eq01.py", line 50, in <module>
    assert loss_error < 1e-10, loss_error
           ^^^^^^^^^^^^^^^^^^
AssertionError: 0.0005408192481005315
```

### Turn 9 — `execute_in_remote_docker` (`17799 ms` LLM)
- **Rationale**: Correct the audit's wall indexing and rerun it without changing the candidate. If the row-sum discrepancy persists, compare losses cell by cell before editing the solver; otherwise use the same run to check runtime and near-aligned nuisance profiling.
- **Alternatives Weighed**: `Change the production Robin assembly` (rejected: The audit extracts a z-wall in (x, y) order but writes it into a (y, x) view. The square grid hides the mismatch, and the production assembly already passed an independent operator check.)

```bash
set -e
sed -i 's/np.swapaxes(expected,0,axis)\[side\]/np.moveaxis(expected, axis, 0)[side]/' /tmp/audit_robin_eq01.py
python3 -I /tmp/audit_robin_eq01.py
```
- **Container Exit Code**: `0` (`27896 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public study still not mounted; audit uses constructed fixtures.
case 0: max cell conservation defect 1.8e-13
strongly nonuniform coefficient shift 0 5.969s worst residual=1.82e-10 sensor L2=1.49e-13 amplitude L2=1.83e-12 joint L2=1.18e-12
case 1: max cell conservation defect 1.05e-14
strongly nonuniform coefficient shift 1 7.002s worst residual=6.16e-08 sensor L2=5.38e-09 amplitude L2=2.1e-07 joint L2=5.43e-07
Shifted two-study runtime: 12.972 s
Near-confounded nuisance: amplitude L2=5.5e-12, joint L2=3.81e-12

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=37599c55e73a52d7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_37599c55 (directory).
```

### Turn 10 — `submit_completed_task` (`16492 ms` LLM)
- **Rationale**: The exact final artifact passed clean-process checks against independently assembled finite-volume operators and augmented least-squares references, including nonterminal residuals, output ordering, strong coefficient shifts, and near-aligned nuisance columns.
- **Alternatives Weighed**: `Revise the Robin assembly or preconditioner` (rejected: The corrected audit found a maximum conservation defect of 1.8e-13, and strongly nonuniform coefficient shifts remained within both residual and runtime limits.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/robin_continuation_tomography.py': 'FILE:5460'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/robin_continuation_tomography.py`. Across the full-size stress tests, the worst relative residual was 6.16e-8 and the slowest two-study run took 12.972 s, both within contract limits. Both tomography outputs passed independent reference checks. The public dataset was not mounted, so validation used synthetic studies.`

### Turn 11 — `execute_in_remote_docker` (`26722 ms` LLM)
- **Rationale**: Use the public study if mounted; otherwise test the previously unmeasured interaction of 4:1 spacing anisotropy and high conductivity contrast. Independent residuals will distinguish a convergence failure from a pure runtime overrun and determine whether the solver needs revision.
- **Alternatives Weighed**: `Repeat the settled conservation and nuisance audits` (rejected: Those checks already passed on this artifact; the new slice tests stronger directional coupling, which could expose a convergence or runtime bottleneck.)

```bash
set -e
python3 -I - <<'PY'
import runpy
from pathlib import Path
import numpy as np
if Path('/holdout_task/public_eq01_studies.pkl').is_file():
    runpy.run_path('/tmp/audit_robin_eq01.py', run_name='__main__')
else:
    print('Public study is not mounted; testing the interaction of grid anisotropy and conductivity contrast.', flush=True)
    n = runpy.run_path('/tmp/test_robin_eq01.py')
    total = 0.0
    for kind, lengths in enumerate(((0.55, 1.15, 2.2), (2.2, 0.55, 1.15))):
        s = n['generate'](821 + kind, (28, 28, 28), kind)
        s.update(zip(('Lx', 'Ly', 'Lz'), lengths))
        if kind:
            x, y, z = np.indices(s['kappa_base'].shape)
            s['kappa_mod2'] = np.where((y + z//3) % 11 < 5, 17., 0.002) * s['kappa_base']
        s['thetas'] = s['thetas'][[0, 6, 2, 3, 4, 8, 1, 7, 5] if not kind else [5, 0, 8, 2, 7, 3, 6, 4, 1]]
        total += n['test'](s, f'4:1 anisotropic study {kind}', True)
    print(f'Anisotropic two-study runtime: {total:.3f} s', flush=True)
    assert total <= 25.0
PY
```
- **Container Exit Code**: `1` (`25275 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public study is not mounted; testing the interaction of grid anisotropy and conductivity contrast.
4:1 anisotropic study 0 3.636s worst residual=1.76e-10 sensor L2=7.01e-14 amplitude L2=1.82e-12 joint L2=1.84e-12
4:1 anisotropic study 1 7.884s worst residual=1.13e-07 sensor L2=6.74e-09 amplitude L2=1.94e-07 joint L2=3.01e-07

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=37599c55e73a52d7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_37599c55 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 17, in <module>
  File "/tmp/test_robin_eq01.py", line 96, in test
    assert max(res)<=1e-7 and ea<=1e-5 and ew<=0.03
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

### Turn 12 — `execute_in_remote_docker` (`28627 ms` LLM)
- **Rationale**: Compare residuals for the same fields under both assemblies, then measure whether extended-precision iterative refinement closes the gap. This identifies whether to change the fallback or how the finite-volume system is represented numerically.
- **Alternatives Weighed**: `Increase the CG iteration cap` (rejected: More iterations cannot correct an assembly-roundoff or residual-evaluation floor, and the failing state may already use the unchecked direct fallback.)

```bash
set -e
cat > /tmp/anisotropic_robin_eq01.py <<'PY'
import runpy
import numpy as np
n = runpy.run_path('/tmp/test_robin_eq01.py')
m = n['m']
def study(kind):
    s = n['generate'](821 + kind, (28,28,28), kind)
    s.update(zip(('Lx','Ly','Lz'), ((.55,1.15,2.2),(2.2,.55,1.15))[kind]))
    if kind:
        x,y,z = np.indices(s['kappa_base'].shape)
        s['kappa_mod2'] = np.where((y+z//3)%11<5,17.,.002)*s['kappa_base']
    order = ([0,6,2,3,4,8,1,7,5],[5,0,8,2,7,3,6,4,1])[kind]
    s['thetas'] = s['thetas'][order]
    return s
if __name__ == '__main__':
    s = study(1)
    Ax = [n['oracle_operator'](s,t)[0].tocsr() for t in np.sort(s['thetas'])]
    old = m._iterate
    i = 0
    def inspect(A,b,M,g):
        global i
        j=i; i+=1
        u = old(A,b,M,g)
        B = Ax[j//3]; norm = np.linalg.norm(b)
        a = np.linalg.norm(b-A@u)/norm; r = np.linalg.norm(b-B@u)/norm
        if max(a,r)>1e-8:
            f=m._factor(A); Al=A.astype(np.longdouble); bl=b.astype(np.longdouble)
            v=u.copy()
            for _ in range(2):
                v += f.solve(np.asarray(bl-Al@v,dtype=float))
            w=m._factor(B).solve(b)
            print(f'theta={np.sort(s["thetas"])[j//3]} mode={j%3}: orig internal/oracle={a:.3g}/{r:.3g}; refine internal/oracle={np.linalg.norm(b-A@v)/norm:.3g}/{np.linalg.norm(b-B@v)/norm:.3g}; oracle direct internal/oracle={np.linalg.norm(b-A@w)/norm:.3g}/{np.linalg.norm(b-B@w)/norm:.3g}; matrix difference action={np.linalg.norm((A-B)@u)/norm:.3g}',flush=True)
        return u
    m._iterate = inspect
    m.solve_robin_continuation_tomography(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/anisotropic_robin_eq01.py
```
- **Container Exit Code**: `0` (`20525 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
theta=0.59 mode=0: orig internal/oracle=7.05e-09/1.13e-08; refine internal/oracle=5.58e-09/1.08e-08; oracle direct internal/oracle=1.54e-08/1.16e-08; matrix difference action=9.38e-09
theta=0.82 mode=0: orig internal/oracle=1.26e-08/1.57e-08; refine internal/oracle=9.81e-09/1.34e-08; oracle direct internal/oracle=2.3e-08/2.06e-08; matrix difference action=9.97e-09
theta=1.02 mode=0: orig internal/oracle=2.01e-08/2.32e-08; refine internal/oracle=1.51e-08/1.95e-08; oracle direct internal/oracle=3.27e-08/2.89e-08; matrix difference action=1.36e-08
theta=1.31 mode=0: orig internal/oracle=2.73e-08/4.09e-08; refine internal/oracle=2.2e-08/3.68e-08; oracle direct internal/oracle=5.64e-08/4.62e-08; matrix difference action=3.03e-08
theta=1.31 mode=1: orig internal/oracle=9.27e-09/1.26e-08; refine internal/oracle=6.87e-09/1.12e-08; oracle direct internal/oracle=1.75e-08/1.44e-08; matrix difference action=9.52e-09
theta=1.7 mode=0: orig internal/oracle=7.56e-08/8.74e-08; refine internal/oracle=4e-08/5.61e-08; oracle direct internal/oracle=8.91e-08/7.53e-08; matrix difference action=4.29e-08
theta=1.7 mode=1: orig internal/oracle=1.46e-08/1.95e-08; refine internal/oracle=1.14e-08/1.72e-08; oracle direct internal/oracle=2.72e-08/2.33e-08; matrix difference action=1.35e-08
theta=1.85 mode=0: orig internal/oracle=8.81e-08/1.13e-07; refine internal/oracle=4.12e-08/8.03e-08; oracle direct internal/oracle=1.16e-07/8.94e-08; matrix difference action=6.91e-08
theta=1.85 mode=1: orig internal/oracle=1.78e-08/2.67e-08; refine internal/oracle=1.37e-08/2.45e-08; oracle direct internal/oracle=3.63e-08/2.75e-08; matrix difference action=2.17e-08

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=37599c55e73a52d7 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_37599c55 (directory).
```

### Turn 13 — `execute_in_remote_docker` (`25576 ms` LLM)
- **Rationale**: Apply refinement only when the checked residual remains high, and retain the best checked iterate. Rerun the failing anisotropic pair first, then the shifted-coefficient audit to detect regressions in residuals, tomography accuracy, or runtime.
- **Alternatives Weighed**: `Change the finite-volume operator` (rejected: The independent checks support the existing discretization, and mixed-precision refinement already brought the failing slice below tolerance without changing the operator.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/robin_continuation_tomography.py')
s = p.read_text()
old = '''        if np.sqrt(_dot(r, r)) > 5e-8 * scale:
            x = _factor(A).solve(b)
'''
new = '''        if np.sqrt(_dot(r, r)) > 3e-8 * scale:
            # Refine the current field rather than replacing it with an
            # unchecked direct solve. Extended-precision residuals avoid
            # cancellation between large diffusion terms and weak losses.
            factor = _factor(A)
            wide_A = A.astype(np.longdouble)
            wide_b = b.astype(np.longdouble)
            trial = x.copy()
            best_error = _dot(r, r)
            for _ in range(2):
                residual = np.asarray(wide_b - wide_A @ trial, dtype=float)
                trial += factor.solve(residual)
                r = b - A @ trial
                error = _dot(r, r)
                if error < best_error:
                    x = trial.copy()
                    best_error = error
'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
python3 -I - <<'PY'
import runpy
n = runpy.run_path('/tmp/anisotropic_robin_eq01.py')
test = n['n']['test']
total = sum(test(n['study'](kind), f'4:1 anisotropic study {kind}', True) for kind in (0, 1))
print(f'Anisotropic two-study runtime: {total:.3f} s', flush=True)
assert total <= 25.0
PY
python3 -I /tmp/audit_robin_eq01.py
```
- **Container Exit Code**: `0` (`55716 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin_continuation_tomography.py']`
- **Stdout (tail)**:
```text
4:1 anisotropic study 0 3.642s worst residual=1.76e-10 sensor L2=7.01e-14 amplitude L2=1.82e-12 joint L2=1.84e-12
4:1 anisotropic study 1 9.296s worst residual=7.92e-08 sensor L2=6.88e-09 amplitude L2=1.94e-07 joint L2=2.96e-07
Anisotropic two-study runtime: 12.938 s
Public study still not mounted; audit uses constructed fixtures.
case 0: max cell conservation defect 1.8e-13
strongly nonuniform coefficient shift 0 5.579s worst residual=1.82e-10 sensor L2=1.49e-13 amplitude L2=1.83e-12 joint L2=1.18e-12
case 1: max cell conservation defect 1.05e-14
strongly nonuniform coefficient shift 1 7.436s worst residual=5.76e-08 sensor L2=5.4e-09 amplitude L2=2.1e-07 joint L2=5.4e-07
Shifted two-study runtime: 13.015 s
Near-confounded nuisance: amplitude L2=5.5e-12, joint L2=3.81e-12

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=c370a7ed9a564ad8 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_c370a7ed (directory).
```

### Turn 14 — `submit_completed_task` (`14491 ms` LLM)
- **Rationale**: The exact final artifact passed clean-process checks on the previously failing anisotropic case and the shifted-coefficient regression suite. The refinement closed the residual gap without compromising runtime or tomography accuracy.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/robin_continuation_tomography.py`. The final solver passed independent checks on full-size synthetic studies: worst relative residual was 7.92e-8, and the slowest two-study run took 13.015 s. Both tomography outputs met their error tolerances. The public dataset was not mounted.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/robin_continuation_tomography.py": "FILE:6169"
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
    "student_elapsed_sec": 6.0672,
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
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_terminal_pde_rel_res\": 1e-10, \"max_sensor_stack_rel_l2\": 0.0, \"max_profiled_rel_l2\": 0.0, \"max_joint_weights_rel_l2\": 0.0, \"student_elapsed_sec\": 6.0672, \"instance_diagnostics\": [{\"seed\": 99401, \"terminal_pde_rel_res\": 1e-10, \"sensor_stack_rel_l2\": 0.0, \"profiled_rel_l2\": 0.0, \"joint_rel_l2\": 0.0}, {\"seed\": 99402, \"terminal_pde_rel_res\": 1e-10, \"sensor_stack_rel_l2\": 0.0, \"profiled_rel_l2\": 0.0, \"joint_rel_l2\": 0.0}]}\n__HEGEL_VERIFIER_AUTH_957e7fb32f1757f711188719cfbd6be5__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_terminal_pde_rel_res\": 1e-10, \"max_sensor_stack_rel_l2\": 0.0, \"max_profiled_rel_l2\": 0.0, \"max_joint_weights_rel_l2\": 0.0, \"student_elapsed_sec\": 6.0672, \"instance_diagnostics\": [{\"seed\": 99401, \"terminal_pde_rel_res\": 1e-10, \"sensor_stack_rel_l2\": 0.0, \"profiled_rel_l2\": 0.0, \"joint_rel_l2\": 0.0}, {\"seed\": 99402, \"terminal_pde_rel_res\": 1e-10, \"sensor_stack_rel_l2\": 0.0, \"profiled_rel_l2\": 0.0, \"joint_rel_l2\": 0.0}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 13108.98,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
