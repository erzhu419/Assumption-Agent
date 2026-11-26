# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `vr02__large_robin_spd_preconditioned_cg_multi_rhs`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M90a-VoluntaryReuse-4` |
| **Domain** | `Large-Scale High-Contrast Elliptic PDE Preconditioning & Nuisance-Profiled Source Matching` |
| **Wave / Wormhole Family** | `m90a_voluntary_reuse` / `fam_multiscale_pde_and_robust_assimilation` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `7 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `6` / `22` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `183,983` (`177,756` / `47,488` / `6,227`) |
| **Total Episode Wall Time (s)** | `210.19s` |

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

### Turn 1 — `execute_in_remote_docker` (`18514 ms` LLM)
- **Rationale**: Inspect fixture shapes, coefficient ranges, and skill contracts before choosing reusable components. In particular, a profiling cutoff must not discard small but identifiable source modes.
- **Alternatives Weighed**: `Reuse the zero-ghost grid assembler` (rejected: Its boundary conductance differs from the specified series Robin conductance.)

```bash
cd /holdout_task && ls -lh && python3 - <<'PY'
import pickle,numpy as np
with open('public_robin_spd_cases.pkl','rb') as f: x=pickle.load(f)
def inspect(x,d=0):
 if d>3:return
 if isinstance(x,dict):
  print(' '*d,'keys:',list(x.keys()))
  for k,v in x.items():
   if isinstance(v,np.ndarray):
    print(' '*d,k,v.shape,v.dtype,(v.min(),v.max()) if v.size else '')
   elif isinstance(v,(dict,list,tuple)):
    print(' '*d,k);inspect(v,d+1)
   else: print(' '*d,k,repr(v)[:100])
 elif isinstance(x,(list,tuple)):
  print(' '*d,type(x).__name__,len(x))
  for v in x[:5]:inspect(v,d+1)
 else: print(type(x))
inspect(x)
PY
python3 -m hegel_wormhole_skills --describe autogenous_prim_conservative_spd_solve.galerkin_preconditioner
python3 -m hegel_wormhole_skills --describe autogenous_prim_conservative_spd_solve.solve_residual_checked_cg
python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes
```
- **Container Exit Code**: `0` (`792 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
h SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes"
      ]
    }
  ]
}
{
  "ok": true,
  "target": "autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes",
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
      "function_level_applicability": {},
      "functions": [
        {
          "function": "profile_candidate_amplitudes",
          "signature": "profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray]",
          "doc_summary": "Profile residual inner products (m,), raw energies (m,), and nuisance cross-products (p, m) using a whitener (r, p), returning amplitudes and remaining energies, both (m,)."
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

### Turn 2 — `execute_in_remote_docker` (`44939 ms` LLM)
- **Rationale**: The public comparison checks every returned array against its reference independently. Any disagreement will determine whether to inspect stencil assembly, CG convergence, or the profile objective before clean-process stress verification.
- **Alternatives Weighed**: `Profile by subtracting the nuisance correction from raw source energies` (rejected: Augmented projection computes the same objective without subtracting nearly equal energies for source modes aligned with nuisance.)

```bash
cat > /holdout_task/robin_spd_profiler.py <<'PY'
"""Conservative Robin diffusion-reaction solves and ridge-profiled matching."""

import numpy as np
from scipy import sparse
from hegel_wormhole_skills.autogenous_prim_conservative_spd_solve import (
    galerkin_preconditioner,
    solve_residual_checked_cg,
)


def _robin_operator(k, sigma, hx, hy, gamma):
    nx, ny = k.shape
    index = np.arange(k.size).reshape(k.shape)

    # Every interior face contributes a symmetric two-cell flux block.
    wx = (hy / hx) * (2 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :]))
    wy = (hx / hy) * (2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:]))
    diagonal = (hx * hy) * sigma
    diagonal[:-1, :] += wx
    diagonal[1:, :] += wx
    diagonal[:, :-1] += wy
    diagonal[:, 1:] += wy

    # Each exterior face contributes its half-cell/Robin series conductance.
    for i in (0, -1):
        kb = k[i, :]
        diagonal[i, :] += hy * (gamma * kb) / (gamma * 0.5 * hx + kb)
    for j in (0, -1):
        kb = k[:, j]
        diagonal[:, j] += hx * (gamma * kb) / (gamma * 0.5 * hy + kb)

    left = np.concatenate((index[:-1, :].ravel(), index[:, :-1].ravel()))
    right = np.concatenate((index[1:, :].ravel(), index[:, 1:].ravel()))
    weights = np.concatenate((wx.ravel(), wy.ravel()))
    ids = index.ravel()
    return sparse.csr_matrix(
        (np.concatenate((diagonal.ravel(), -weights, -weights)),
         (np.concatenate((ids, left, right)), np.concatenate((ids, right, left)))),
        shape=(nx * ny, nx * ny),
    )


def _profile(s, z, y, lam_z, lam_s):
    # The ridge penalty is an extra set of observations with zero targets.
    # Orthogonally projecting in this augmented space profiles out nuisance.
    p, m = z.shape[1], s.shape[1]
    augmented_z = np.vstack((z, np.sqrt(lam_z) * np.eye(p)))
    values = np.vstack((np.column_stack((s, y)), np.zeros((p, m + y.shape[1]))))
    u, singular, _ = np.linalg.svd(augmented_z, full_matrices=False)
    if singular.size:
        cutoff = np.finfo(float).eps * max(augmented_z.shape) * singular[0]
        u = u[:, singular > cutoff]
        values -= u @ (u.T @ values)
    s_perp, y_perp = values[:, :m], values[:, m:]

    cross = s_perp.T @ y_perp
    energy = np.einsum('ij,ij->j', s_perp, s_perp)[:, None]
    amplitudes = np.divide(cross, energy, out=np.zeros_like(cross), where=energy > 0)
    joint = np.linalg.lstsq(
        np.vstack((s_perp, np.sqrt(lam_s) * np.eye(m))),
        np.vstack((y_perp, np.zeros((m, y.shape[1])))),
        rcond=None,
    )[0]
    return amplitudes, joint


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Solve the candidate fields and fit single-source and joint weights."""
    k = np.asarray(problem['kappa'], dtype=float)
    nx, ny = k.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    sigma = np.asarray(problem['sigma'], dtype=float)
    a = _robin_operator(k, sigma, hx, hy, problem['gamma_robin'])
    preconditioner = galerkin_preconditioner(a)
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    fields = np.empty_like(sources)
    for m, source in enumerate(sources):
        rhs = (hx * hy) * source.ravel()
        solution, _ = solve_residual_checked_cg(
            a, rhs, preconditioner=preconditioner, tol=1e-9, maxiter=600,
        )
        fields[m] = solution.reshape(k.shape)

    sensors = np.asarray(problem['sensor_ij'])
    s = fields[:, sensors[:, 0], sensors[:, 1]].T
    amplitudes, joint = _profile(
        s, np.asarray(problem['nuisance_design'], dtype=float),
        np.asarray(problem['Y_obs'], dtype=float),
        problem['nuisance_ridge'], problem['source_ridge'],
    )
    return {'basis_fields': fields,
            'single_source_profiled_amplitudes': amplitudes,
            'joint_source_weights': joint}
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle, time
import numpy as np
from robin_spd_profiler import solve_robin_spd_and_profile_sources
with open('public_robin_spd_cases.pkl','rb') as f: cases=pickle.load(f)
for i, case in enumerate(cases):
    t=time.perf_counter()
    out=solve_robin_spd_and_profile_sources(case['problem'])
    print(f'case {i}: {time.perf_counter()-t:.3f}s')
    for k, expected in case['gt'].items():
        assert out[k].shape == expected.shape and np.isfinite(out[k]).all()
        error=out[k]-expected
        print(k, 'max abs', np.max(np.abs(error)), 'rel L2', np.linalg.norm(error)/np.linalg.norm(expected))
PY
```
- **Container Exit Code**: `0` (`910 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin_spd_profiler.py']`
- **Stdout (tail)**:
```text
case 0: 0.189s
basis_fields max abs 3.129657991096657e-13 rel L2 3.799350346501559e-12
single_source_profiled_amplitudes max abs 7.361222742474638e-12 rel L2 1.0378258361688766e-12
joint_source_weights max abs 8.885114866075128e-13 rel L2 2.03274573388348e-12
case 1: 0.171s
basis_fields max abs 4.3914893423091517e-13 rel L2 4.042782917990869e-12
single_source_profiled_amplitudes max abs 1.2925882586500848e-11 rel L2 2.979203713576055e-12
joint_source_weights max abs 1.463218435304725e-12 rel L2 2.5553307226626475e-12

[HOST STATIC PACKAGER]: Bundled self-contained `hegel_wormhole_skills` closure into `['/holdout_task/robin_spd_profiler.py']` (`pre_sha=8943314c0e6a` -> `post_sha=3c5e56b87355`). Before `submit_completed_task`, run a clean `python3 -I` verification on the packaged target artifact so `VerificationReceipt` binds to `post_sha`.
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3c5e56b87355e47b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3c5e56b8 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`42262 ms` LLM)
- **Rationale**: Verify the packaged artifact in isolation. Independent face-flux residuals distinguish stencil or convergence errors from profiling errors, while full augmented least-squares references check the two distinct penalty conventions.
- **Alternatives Weighed**: `Verify only against the stored public outputs` (rejected: That would not independently test the face balance or CG convergence on connected high-conductivity regions.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import time
import numpy as np
from scipy.sparse.linalg import splu

spec = importlib.util.spec_from_file_location('robin_spd_profiler', '/holdout_task/robin_spd_profiler.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
solve = mod.solve_robin_spd_and_profile_sources
with open('public_robin_spd_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

def independent_operator_action(p, fields):
    k = p['kappa']
    nx, ny = k.shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    gamma = p['gamma_robin']
    f = fields * p['sigma'] * hx * hy
    dx = (fields[:, :-1, :] - fields[:, 1:, :]) * (2*hy/hx) / (1/k[:-1, :] + 1/k[1:, :])
    dy = (fields[:, :, :-1] - fields[:, :, 1:]) * (2*hx/hy) / (1/k[:, :-1] + 1/k[:, 1:])
    f[:, :-1, :] += dx
    f[:, 1:, :] -= dx
    f[:, :, :-1] += dy
    f[:, :, 1:] -= dy
    if gamma:
        f[:, [0, -1], :] += fields[:, [0, -1], :] * hy / (1/gamma + hx/(2*k[[0, -1], :]))
        f[:, :, [0, -1]] += fields[:, :, [0, -1]] * hx / (1/gamma + hy/(2*k[:, [0, -1]]))
    return f

def profile_reference(p, fields):
    ii, jj = p['sensor_ij'].T
    s = fields[:, ii, jj].T
    z, y = p['nuisance_design'], p['Y_obs']
    nz, ns = z.shape[1], s.shape[1]
    rz, rs = np.sqrt(p['nuisance_ridge']), np.sqrt(p['source_ridge'])
    zeros = lambda n: np.zeros((n, y.shape[1]))
    single = np.stack([np.linalg.lstsq(np.vstack((np.column_stack((z, sm)), np.column_stack((rz*np.eye(nz), np.zeros(nz))))), np.vstack((y, zeros(nz))), rcond=None)[0][-1] for sm in s.T])
    penalty = np.diag(np.r_[np.full(nz, rz), np.full(ns, rs)])
    joint = np.linalg.lstsq(np.vstack((np.column_stack((z, s)), penalty)), np.vstack((y, zeros(nz+ns))), rcond=None)[0][nz:]
    return single, joint

def verify(name, p, expected=None):
    start = time.perf_counter()
    out = solve(p)
    fields = out['basis_fields']
    k = p['kappa']
    nx, ny = k.shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    rhs = p['candidate_sources'] * hx * hy
    residual = (independent_operator_action(p, fields) - rhs).reshape(len(fields), -1)
    worst = np.max(np.linalg.norm(residual, axis=1)/np.linalg.norm(rhs.reshape(len(fields), -1), axis=1))
    basis_error = None
    if expected is None:
        a = mod._robin_operator(k, p['sigma'], hx, hy, p['gamma_robin'])
        reference_fields = splu(a.tocsc()).solve(rhs.reshape(len(fields), -1).T).T.reshape(fields.shape)
        basis_error = np.linalg.norm(fields-reference_fields)/np.linalg.norm(reference_fields)
        assert basis_error < 1e-6, basis_error
    else:
        reference_fields = expected['basis_fields']
        for key, ref in expected.items():
            np.testing.assert_allclose(out[key], ref, atol=1e-9, rtol=1e-7)
    ref_single, ref_joint = profile_reference(p, reference_fields)
    for key, ref in [('single_source_profiled_amplitudes', ref_single), ('joint_source_weights', ref_joint)]:
        np.testing.assert_allclose(out[key], ref, atol=1e-7, rtol=1e-6)
    print(name, 'worst independent PDE residual', f'{worst:.3e}', 'single max abs', f'{np.max(np.abs(out["single_source_profiled_amplitudes"]-ref_single)):.3e}', 'joint max abs', f'{np.max(np.abs(out["joint_source_weights"]-ref_joint)):.3e}', 'basis rel vs LU', basis_error, 'sec', round(time.perf_counter()-start, 3), flush=True)
    assert worst <= 1e-8, (name, worst)

for n, case in enumerate(cases):
    verify(f'public-{n}', case['problem'], case['gt'])
p = cases[0]['problem']
i,j = np.indices(p['kappa'].shape)
verify('full-contrast-checkerboard', dict(p, kappa=np.where((i+j)%2, 80., .02), Lx=1.8, Ly=.75, gamma_robin=.27))
verify('full-contrast-conducting-channels', dict(p, kappa=np.where((i%12<4)|(j%13<3), 80., .02), Lx=.85, Ly=1.65, gamma_robin=.4))
verify('weak-reaction-high-conductivity', dict(p, kappa=np.where(i<6, .02, 80.), sigma=np.full(i.shape, .01), gamma_robin=.15))
PY
```
- **Container Exit Code**: `0` (`1800 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public-0 worst independent PDE residual 7.073e-10 single max abs 7.375e-12 joint max abs 8.844e-13 basis rel vs LU None sec 0.253
public-1 worst independent PDE residual 7.384e-10 single max abs 1.309e-11 joint max abs 1.433e-12 basis rel vs LU None sec 0.171
full-contrast-checkerboard worst independent PDE residual 7.407e-10 single max abs 3.300e-12 joint max abs 5.308e-12 basis rel vs LU 1.4888575151060287e-11 sec 0.165
full-contrast-conducting-channels worst independent PDE residual 7.627e-10 single max abs 2.780e-12 joint max abs 3.684e-13 basis rel vs LU 1.2420939629608322e-12 sec 0.464
weak-reaction-high-conductivity worst independent PDE residual 8.194e-10 single max abs 1.665e-11 joint max abs 1.052e-12 basis rel vs LU 1.3772437142640848e-11 sec 0.184

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3c5e56b87355e47b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3c5e56b8 (directory).
```

### Turn 4 — `submit_completed_task` (`12959 ms` LLM)
- **Rationale**: The packaged artifact passed isolated verification against the public fixtures, independent face-flux residuals, and separate least-squares references for both profiling objectives. Full-contrast checkerboard and conducting-channel cases and a weak-reaction case all met the residual requirement.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/robin_spd_profiler.py': 'FILE:9510'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin_spd_profiler.py`. The worst relative PDE residual across both public cases and three stress cases was 8.194e-10, below the 1e-8 requirement. Both profiling outputs agreed with independent least-squares references to within 1.67e-11 absolute error.`

### Turn 5 — `execute_in_remote_docker` (`31421 ms` LLM)
- **Rationale**: The relevant algebraic risk is that a source signature is nearly explained by the nuisance design and a pooled comparison misses a wall-specific error. Compare both penalties against full augmented fits on disjoint public sensor slices; a mismatch would isolate where to investigate profiling or sensor extraction before submitting.
- **Alternatives Weighed**: `Run classification, relay, or time-evolution diagnostics` (rejected: The task has no class labels, technical batches, relay state, or time-dependent operator.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('robin_spd_profiler', '/holdout_task/robin_spd_profiler.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('public_robin_spd_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

for n, case in enumerate(cases):
    p = case['problem']
    ij = p['sensor_ij']
    shape = np.array(p['kappa'].shape)
    wall = np.any((ij == 0) | (ij == shape - 1), axis=1)
    for label, mask in [('wall', wall), ('interior', ~wall)]:
        sub = dict(p, **{key: p[key][mask] for key in ('sensor_ij', 'nuisance_design', 'Y_obs')})
        ii, jj = sub['sensor_ij'].T
        s = case['gt']['basis_fields'][:, ii, jj].T
        z, y = sub['nuisance_design'], sub['Y_obs']
        m, d = s.shape[1], z.shape[1]
        rz, rs = sub['nuisance_ridge'], sub['source_ridge']
        def reference(x, penalty):
            a = np.hstack((z, x))
            r = np.diag(np.sqrt(np.r_[np.full(d, rz), np.full(x.shape[1], penalty)]))
            rhs = np.vstack((y, np.zeros((r.shape[0], y.shape[1]))))
            return np.linalg.lstsq(np.vstack((a, r)), rhs, rcond=None)[0][d:]
        single = np.concatenate([reference(s[:, j:j+1], 0) for j in range(m)], axis=0)
        joint = reference(s, rs)
        out = mod.solve_robin_spd_and_profile_sources(sub)
        np.testing.assert_allclose(out['single_source_profiled_amplitudes'], single, atol=1e-8, rtol=1e-7)
        np.testing.assert_allclose(out['joint_source_weights'], joint, atol=1e-8, rtol=1e-7)
        g = z.T @ z + rz * np.eye(d)
        energy = np.sum(s*s, axis=0)
        remaining = energy - np.sum((z.T @ s) * np.linalg.solve(g, z.T @ s), axis=0)
        print(f'public-{n} {label} sensors={len(ii)} min remaining energy fraction={min(remaining/energy):.3e} single max abs={np.max(np.abs(out["single_source_profiled_amplitudes"]-single)):.3e} joint max abs={np.max(np.abs(out["joint_source_weights"]-joint)):.3e}')
PY
```
- **Container Exit Code**: `0` (`1402 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public-0 wall sensors=21 min remaining energy fraction=1.378e-02 single max abs=7.812e-11 joint max abs=7.247e-14
public-0 interior sensors=339 min remaining energy fraction=5.754e-02 single max abs=8.098e-12 joint max abs=8.458e-13
public-1 wall sensors=19 min remaining energy fraction=2.236e-02 single max abs=5.503e-11 joint max abs=1.959e-13
public-1 interior sensors=341 min remaining energy fraction=6.456e-02 single max abs=1.182e-11 joint max abs=1.447e-12

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=3c5e56b87355e47b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_3c5e56b8 (directory).
```

### Turn 6 — `submit_completed_task` (`14450 ms` LLM)
- **Rationale**: The packaged artifact passed isolated execution, independent face-flux residual checks, and separate least-squares comparisons for both profiling objectives. The wall-only slice retained as little as 1.378% of a source mode’s energy after nuisance projection, yet both profiling outputs still matched their references. No change is warranted.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/robin_spd_profiler.py': 'FILE:9510'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin_spd_profiler.py`. The worst relative PDE residual was 8.194e-10 across two public cases and three stress cases, below the 1e-8 requirement. Both profiling outputs matched independent least-squares references within 7.82e-11 absolute error, including separate wall and interior sensor checks.`

### Turn 7 — `submit_completed_task` (`15184 ms` LLM)
- **Rationale**: The unchanged packaged artifact has a bound clean-process verification receipt. Independent face-flux checks passed on public and stress cases, and direct least-squares fits verified both profiling objectives even on the wall slice that retained only 1.378% of a source mode's energy after nuisance projection. No unresolved failure warrants another check or edit.
- **Alternatives Weighed**: `Run batch-classification, relay, or time-evolution audits` (rejected: The input contract has no batch labels, relay states, or time dynamics. Nuisance profiling was checked directly on both public sensor slices.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin_spd_profiler.py`. The worst relative PDE residual across two public cases and three stress cases was 8.194e-10, below the 1e-8 requirement. Both profiling outputs matched independent least-squares references within 7.82e-11 absolute error, including separate wall and interior sensor checks.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M90a-VoluntaryReuse-4",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
