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
| **Turns Used / Max Turns** | `7 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `6` / `22` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `186,256` (`179,440` / `45,312` / `6,816`) |
| **Total Episode Wall Time (s)** | `236.84s` |

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

### Turn 1 — `execute_in_remote_docker` (`17992 ms` LLM)
- **Rationale**: Inspect the public schema and coefficient ranges to select relevant stress slices. Read the reusable solver and profiling contracts before adopting them; incompatible boundary or ridge assumptions will require local implementations.
- **Alternatives Weighed**: `Reuse the zero-ghost Dirichlet grid operator` (rejected: It does not implement the specified series half-cell Robin conductance.)

```bash
ls -l /holdout_task; python3 - <<'PY'
import pickle, numpy as np
with open('/holdout_task/public_robin_spd_cases.pkl','rb') as f: obj=pickle.load(f)
def view(o,name,depth=0):
    if isinstance(o,dict) and 'kappa' in o:
        print(name)
        for k,v in o.items():
            if isinstance(v,np.ndarray):
                print(' ',k,v.shape,v.dtype, (float(np.min(v)),float(np.max(v))) if v.size and np.issubdtype(v.dtype,np.number) else '')
            else: print(' ',k,repr(v)[:180])
    elif depth<3 and isinstance(o,(list,tuple,dict)):
        print(name,type(o).__name__,len(o));
        for k,v in (o.items() if isinstance(o,dict) else enumerate(o)): view(v,f'{name}.{k}',depth+1)
    elif isinstance(o,np.ndarray): print(name,o.shape,o.dtype)
    else: print(name,type(o).__name__,repr(o)[:160])
view(obj,'root')
PY
for f in galerkin_preconditioner solve_residual_checked_cg; do python3 -m hegel_wormhole_skills --describe autogenous_prim_conservative_spd_solve.$f; done
for f in fit_ridge_nuisance profile_candidate_amplitudes; do python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.$f; done
```
- **Container Exit Code**: `0` (`933 ms`, timed_out=`False`)
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

### Turn 2 — `execute_in_remote_docker` (`51262 ms` LLM)
- **Rationale**: Evaluate every public RHS against the residual limit and compare all three outputs with their references. A residual failure calls for investigating the stencil or preconditioner; agreement in the fields but not the amplitudes or weights isolates the inverse fit.
- **Alternatives Weighed**: `Profile nuisance by subtracting a Schur complement from the source Gram matrix` (rejected: Projecting the augmented data directly avoids subtracting nearly equal Gram terms when a source resembles a nuisance column.)

```bash
cat > /holdout_task/robin_spd_profiler.py <<'PY'
"""Conservative Robin diffusion solves and nuisance-profiled source fits."""

import numpy as np
from scipy import sparse
from hegel_wormhole_skills.autogenous_prim_conservative_spd_solve import (
    galerkin_preconditioner,
    solve_residual_checked_cg,
)


def _robin_operator(kappa, sigma, hx, hy, gamma):
    nx, ny = kappa.shape
    cells = np.arange(nx * ny).reshape(nx, ny)
    k0, k1 = kappa[:-1, :], kappa[1:, :]
    gx = (hy / hx) * (2 * k0 * k1 / (k0 + k1))
    k0, k1 = kappa[:, :-1], kappa[:, 1:]
    gy = (hx / hy) * (2 * k0 * k1 / (k0 + k1))

    diagonal = np.broadcast_to(sigma, kappa.shape).copy() * (hx * hy)
    diagonal[:-1, :] += gx
    diagonal[1:, :] += gx
    diagonal[:, :-1] += gy
    diagonal[:, 1:] += gy
    for i in (0, -1):
        kb = kappa[i, :]
        diagonal[i, :] += hy * (gamma * kb) / (gamma * 0.5 * hx + kb)
    for j in (0, -1):
        kb = kappa[:, j]
        diagonal[:, j] += hx * (gamma * kb) / (gamma * 0.5 * hy + kb)

    left = np.concatenate((cells[:-1, :].ravel(), cells[:, :-1].ravel()))
    right = np.concatenate((cells[1:, :].ravel(), cells[:, 1:].ravel()))
    conductance = np.concatenate((gx.ravel(), gy.ravel()))
    ids = cells.ravel()
    return sparse.csr_matrix(
        (np.concatenate((diagonal.ravel(), -conductance, -conductance)),
         (np.concatenate((ids, left, right)), np.concatenate((ids, right, left)))),
        shape=(ids.size, ids.size),
    )


def _profile_sources(sources, nuisance, observations, nuisance_ridge, source_ridge):
    # Augment the data by the nuisance penalty rows, then eliminate nuisance
    # using an orthonormal basis of its augmented column space.
    p = nuisance.shape[1]
    m = sources.shape[1]
    design = np.vstack((nuisance, np.sqrt(nuisance_ridge) * np.eye(p)))
    data = np.vstack((np.column_stack((sources, observations)),
                      np.zeros((p, m + observations.shape[1]))))
    if p:
        u, d, _ = np.linalg.svd(design, full_matrices=False)
        u = u[:, d > np.finfo(float).eps * max(design.shape) * d[0]]
        data -= u @ (u.T @ data)
    s, y = data[:, :m], data[:, m:]
    energy = np.einsum('ij,ij->j', s, s)
    original_norm = np.linalg.norm(sources, axis=0)
    identified = np.sqrt(energy) > np.finfo(float).eps * max(design.shape) * original_norm
    cross = s.T @ y
    single = np.divide(cross, energy[:, None], out=np.zeros_like(cross),
                       where=identified[:, None])

    # Only the joint fit includes the source penalty.
    joint_design = np.vstack((s, np.sqrt(source_ridge) * np.eye(m)))
    joint_obs = np.vstack((y, np.zeros((m, y.shape[1]))))
    joint = np.linalg.lstsq(joint_design, joint_obs, rcond=None)[0]
    return single, joint


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Return candidate response fields and the two source estimators."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    sigma = np.asarray(problem['sigma'], dtype=float)
    nx, ny = kappa.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    operator = _robin_operator(kappa, sigma, hx, hy, problem['gamma_robin'])
    preconditioner = galerkin_preconditioner(operator)
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    basis = np.empty_like(sources)
    for k, source in enumerate(sources):
        rhs = (hx * hy) * source.ravel()
        solution, _ = solve_residual_checked_cg(
            operator, rhs, preconditioner=preconditioner, tol=1e-9, maxiter=500,
        )
        basis[k] = solution.reshape(nx, ny)

    i, j = np.asarray(problem['sensor_ij']).T
    single, joint = _profile_sources(
        basis[:, i, j].T,
        np.asarray(problem['nuisance_design'], dtype=float),
        np.asarray(problem['Y_obs'], dtype=float),
        problem['nuisance_ridge'], problem['source_ridge'],
    )
    return {'basis_fields': basis, 'single_source_profiled_amplitudes': single,
            'joint_source_weights': joint}
PY
python3 - <<'PY'
import pickle, sys, time
import numpy as np
sys.path.insert(0, '/holdout_task')
import robin_spd_profiler as r
for i, case in enumerate(pickle.load(open('/holdout_task/public_robin_spd_cases.pkl', 'rb'))):
    t = time.perf_counter()
    p = case['problem']
    out = r.solve_robin_spd_and_profile_sources(p)
    nx, ny = p['kappa'].shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    a = r._robin_operator(p['kappa'], p['sigma'], hx, hy, p['gamma_robin'])
    b = (hx*hy)*p['candidate_sources'].reshape(-1, nx*ny).T
    err = np.linalg.norm(a@out['basis_fields'].reshape(-1,nx*ny).T-b,axis=0)/np.linalg.norm(b,axis=0)
    print('case',i,'seconds',round(time.perf_counter()-t,3),'worst residual',err.max())
    for key, expected in case['gt'].items():
        diff = out[key]-expected
        print(key, out[key].shape, 'abs max', np.max(np.abs(diff)), 'relative norm',np.linalg.norm(diff)/np.linalg.norm(expected))
    assert err.max() <= 1e-8
PY
```
- **Container Exit Code**: `0` (`1586 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin_spd_profiler.py']`
- **Stdout (tail)**:
```text
case 0 seconds 0.178 worst residual 7.07022275805265e-10
basis_fields (10, 72, 72) abs max 3.129657991096657e-13 relative norm 3.799350346501559e-12
single_source_profiled_amplitudes (10, 5) abs max 7.361222742474638e-12 relative norm 1.0378258361688766e-12
joint_source_weights (10, 5) abs max 8.885114866075128e-13 relative norm 2.03274573388348e-12
case 1 seconds 0.304 worst residual 7.385638170974739e-10
basis_fields (10, 72, 72) abs max 4.3914893423091517e-13 relative norm 4.04278291799087e-12
single_source_profiled_amplitudes (10, 5) abs max 1.2925882586500848e-11 relative norm 2.979203713576055e-12
joint_source_weights (10, 5) abs max 1.463218435304725e-12 relative norm 2.5553307226626475e-12

[HOST STATIC PACKAGER]: Bundled self-contained `hegel_wormhole_skills` closure into `['/holdout_task/robin_spd_profiler.py']` (`pre_sha=cc1077ae1d8f` -> `post_sha=419ac5c1d212`). Before `submit_completed_task`, run a clean `python3 -I` verification on the packaged target artifact so `VerificationReceipt` binds to `post_sha`.
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=419ac5c1d21230d9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_419ac5c1 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`41593 ms` LLM)
- **Rationale**: Run the packaged artifact in isolation. The independent flux balance will catch stencil errors, contrast patterns will expose convergence failures, and independent augmented fits will catch incorrect penalties or nuisance elimination. Revise only the component implicated by a failure.
- **Alternatives Weighed**: `Use only the public reference comparisons` (rejected: They do not separately test the Robin flux stencil or convergence for different placements of the full conductivity contrast.)

```bash
cat > /tmp/verify_robin_spd.py <<'PY'
import importlib.util
import pickle
import time
import numpy as np
from scipy.sparse.linalg import spsolve

path = '/holdout_task/robin_spd_profiler.py'
spec = importlib.util.spec_from_file_location('robin_spd_profiler', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
keys = ('basis_fields', 'single_source_profiled_amplitudes', 'joint_source_weights')

def check(name, p, expected=None):
    start = time.perf_counter()
    result = module.solve_robin_spd_and_profile_sources(p)
    rhs = p['candidate_sources']
    k = p['kappa']
    nx, ny = k.shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    gamma = p['gamma_robin']
    u = result['basis_fields']
    i, j = p['sensor_ij'].T
    s = u[:, i, j].T
    z, y = p['nuisance_design'], p['Y_obs']
    m, n, q = s.shape[1], z.shape[1], y.shape[1]
    assert set(result) == set(keys)
    for key, shape in zip(keys, (rhs.shape, (m,q), (m,q))):
        assert result[key].shape == shape and np.isfinite(result[key]).all(), key

    # Independent flux-form application; no call to the assembly routine.
    r = (hx*hy) * (p['sigma']*u-rhs)
    fx = (u[:, :-1, :] - u[:, 1:, :]) * ((hy/hx)/(0.5/k[:-1, :] + 0.5/k[1:, :]))
    fy = (u[:, :, :-1] - u[:, :, 1:]) * ((hx/hy)/(0.5/k[:, :-1] + 0.5/k[:, 1:]))
    r[:, :-1, :] += fx
    r[:, 1:, :] -= fx
    r[:, :, :-1] += fy
    r[:, :, 1:] -= fy
    for a in (0, -1):
        r[:, a, :] += u[:, a, :] * (hy*gamma/(1 + 0.5*hx*gamma/k[a, :]))
        r[:, :, a] += u[:, :, a] * (hx*gamma/(1 + 0.5*hy*gamma/k[:, a]))
    norm = hx*hy*np.linalg.norm(rhs, axis=(1,2))
    relative = np.divide(np.linalg.norm(r, axis=(1,2)), norm, out=np.zeros_like(norm), where=norm > 0)
    assert np.all(u[norm == 0] == 0)
    assert relative.max() <= 1e-8, relative

    a = module._robin_operator(k, p['sigma'], hx, hy, gamma)
    ref = spsolve(a.tocsc(), (hx*hy)*rhs.reshape(m,-1).T).T.reshape(rhs.shape)
    assert np.allclose(u, ref, rtol=1e-7, atol=1e-10)
    zr = np.sqrt(p['nuisance_ridge'])*np.eye(n)
    single = np.zeros((m,q))
    for l in range(m):
        design = np.block([[z, s[:,l:l+1]], [zr, np.zeros((n,1))]])
        single[l] = np.linalg.lstsq(design, np.vstack((y,np.zeros((n,q)))), rcond=None)[0][-1]
    design = np.block([[z,s], [zr,np.zeros((n,m))], [np.zeros((m,n)),np.sqrt(p['source_ridge'])*np.eye(m)]])
    joint = np.linalg.lstsq(design, np.vstack((y,np.zeros((n+m,q)))), rcond=None)[0][-m:]
    for key, target in zip(keys, (ref, single, joint)):
        assert np.allclose(result[key],target,rtol=2e-7,atol=1e-9), key
        if expected is not None:
            assert np.allclose(result[key],expected[key],rtol=2e-7,atol=1e-9), key
    errors = [np.max(np.abs(result[key]-target)) for key,target in zip(keys, (ref,single,joint))]
    print(f'{name}: worst residual={relative.max():.3e}; max errors fields/single/joint={errors}; elapsed={time.perf_counter()-start:.3f}s', flush=True)

with open('/holdout_task/public_robin_spd_cases.pkl','rb') as f:
    cases = pickle.load(f)
for c, case in enumerate(cases):
    check(f'public-{c}', case['problem'], case['gt'])
p = cases[0]['problem']
x, y = np.indices(p['kappa'].shape)
for name, k in [('checkerboard', np.where((x+y)%2, 0.02, 80.0)), ('conducting-stripes', np.where((x//4)%2, 0.02, 80.0)), ('high-conductivity-core', np.where((x>8)&(x<62)&(y>8)&(y<62), 80.0, 0.02))]:
    q = p['candidate_sources'].copy()
    q[0] = 0
    q[1] -= q[2]
    check(name, dict(p, Lx=2.0, Ly=0.8, kappa=k, candidate_sources=q))
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/verify_robin_spd.py
```
- **Container Exit Code**: `0` (`2066 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public-0: worst residual=7.073e-10; max errors fields/single/joint=[3.129653654287967e-13, 1.3100631690576847e-14, 1.6653345369377348e-15]; elapsed=0.236s
public-1: worst residual=7.384e-10; max errors fields/single/joint=[4.3914893423091517e-13, 7.105427357601002e-15, 2.0122792321330962e-15]; elapsed=0.219s
checkerboard: worst residual=7.695e-10; max errors fields/single/joint=[7.326920320460673e-12, 1.5612511283791264e-15, 1.2698175844150228e-15]; elapsed=0.165s
conducting-stripes: worst residual=8.574e-10; max errors fields/single/joint=[1.3169771805217334e-12, 2.7478019859472624e-15, 2.3314683517128287e-15]; elapsed=0.606s
high-conductivity-core: worst residual=7.782e-10; max errors fields/single/joint=[9.273450063407296e-13, 8.43769498715119e-15, 2.4424906541753444e-15]; elapsed=0.281s

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=419ac5c1d21230d9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_419ac5c1 (directory).
```

### Turn 4 — `submit_completed_task` (`13187 ms` LLM)
- **Rationale**: The packaged artifact passed isolated verification of the return schema, independent flux balances, and both inverse fits. The verification receipt is bound to the submitted artifact.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`24` turns remaining; artifact status={'/holdout_task/robin_spd_profiler.py': 'FILE:9801'}): Do NOT lock prematurely with `24` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin_spd_profiler.py`. Both public cases and all three high-contrast stress cases passed. The worst relative PDE residual was 8.57e-10, below the 1e-8 limit. Both source estimators agreed with independent least-squares fits.`

### Turn 5 — `execute_in_remote_docker` (`54093 ms` LLM)
- **Rationale**: Use both public cases to test two distinct failure modes. Axis exchange, signed source contrasts, and global conservation can expose stencil or indexing errors. Adding a nuisance column aligned with a source can expose incorrect ridge handling; if either fit fails against the unprofiled reference, inspect the nuisance elimination rather than changing the PDE solver.
- **Alternatives Weighed**: `Run batch-classification, relay-mode, or time-drift checks` (rejected: The inputs describe a steady PDE and continuous observations, with no batches, class labels, relay modes, or time dynamics.)

```bash
cat > /tmp/audit_robin_spd.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('robin_spd_profiler', '/holdout_task/robin_spd_profiler.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin_spd_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

for idx, case in enumerate(cases):
    p, ref = case['problem'], case['gt']
    k, z = p['kappa'], p['nuisance_design']
    q = p['candidate_sources']
    hx, hy = p['Lx']/k.shape[0], p['Ly']/k.shape[1]
    g = p['gamma_robin']
    # Axis exchange and a signed source contrast give exact transformations of
    # the measured-case fields, without using the solver's assembled matrix.
    expected = ref['basis_fields'].copy()
    expected[0] -= expected[1]
    q_contrast = q.copy()
    q_contrast[0] -= q_contrast[1]
    i, j = p['sensor_ij'].T
    col = expected[0, i, j]
    # Force source/nuisance overlap at the actual positive ridge, leaving the
    # observations unchanged. Compare with unprofiled augmented least squares.
    z_test = np.column_stack((z, col / np.linalg.norm(col)))
    transformed = dict(p, Lx=p['Ly'], Ly=p['Lx'], kappa=k.T,
                       sigma=p['sigma'].T, candidate_sources=q_contrast.swapaxes(1, 2),
                       sensor_ij=p['sensor_ij'][:, ::-1], nuisance_design=z_test)
    result = mod.solve_robin_spd_and_profile_sources(transformed)
    fields = result['basis_fields'].swapaxes(1, 2)
    np.testing.assert_allclose(fields, expected, rtol=1e-7, atol=1e-10)
    # All interior face pairs cancel in the global finite-volume budget.
    lhs = (hx*hy)*(p['sigma']*fields).sum(axis=(1, 2))
    for edge in (0, -1):
        lhs += (fields[:, edge, :] * (hy*g/(1 + 0.5*hx*g/k[edge, :]))).sum(axis=1)
        lhs += (fields[:, :, edge] * (hx*g/(1 + 0.5*hy*g/k[:, edge]))).sum(axis=1)
    balance = np.max(np.abs(lhs - (hx*hy)*q_contrast.sum(axis=(1, 2))) / ((hx*hy)*np.abs(q_contrast).sum(axis=(1, 2))))
    assert balance <= 1e-8, balance
    s, y = fields[:, i, j].T, p['Y_obs']
    nz, m = z_test.shape[1], s.shape[1]
    z_pen = np.sqrt(p['nuisance_ridge'])*np.eye(nz)
    y_aug = np.pad(y, ((0, nz), (0, 0)))
    single = np.stack([np.linalg.lstsq(np.block([[z_test, s[:, l:l+1]], [z_pen, np.zeros((nz, 1))]]), y_aug, rcond=None)[0][-1] for l in range(m)])
    joint_design = np.block([[z_test, s], [z_pen, np.zeros((nz, m))], [np.zeros((m, nz)), np.sqrt(p['source_ridge'])*np.eye(m)]])
    joint = np.linalg.lstsq(joint_design, np.pad(y_aug, ((0, m), (0, 0))), rcond=None)[0][-m:]
    errors = []
    for key, expected_fit in [('single_source_profiled_amplitudes', single), ('joint_source_weights', joint)]:
        np.testing.assert_allclose(result[key], expected_fit, rtol=2e-7, atol=1e-9)
        errors.append(np.max(np.abs(result[key] - expected_fit)))
    print(f'public-{idx} transformed: max field error={np.max(np.abs(fields-expected)):.3e}; relative global balance={balance:.3e}; max single/joint errors={errors}')
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/audit_robin_spd.py
```
- **Container Exit Code**: `0` (`896 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public-0 transformed: max field error=2.018e-13; relative global balance=1.081e-13; max single/joint errors=[6.106226635438361e-14, 1.096345236817342e-15]
public-1 transformed: max field error=5.119e-13; relative global balance=2.426e-13; max single/joint errors=[1.0524914273446484e-13, 2.6645352591003757e-15]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=419ac5c1d21230d9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_419ac5c1 (directory).
```

### Turn 6 — `submit_completed_task` (`14186 ms` LLM)
- **Rationale**: The packaged artifact passed isolated verification. The follow-up audit confirmed axis-swap and signed-source consistency, with a worst relative global balance error of 2.43e-13. Both inverse fits matched independent references, so no change is warranted.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`22` turns remaining; artifact status={'/holdout_task/robin_spd_profiler.py': 'FILE:9801'}): Do NOT lock prematurely with `22` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin_spd_profiler.py`. Both public cases and all three high-contrast stress cases passed. The worst relative PDE residual was 8.57e-10, below the 1e-8 limit. Both source estimators agreed with independent least-squares fits, including when a nuisance column aligned with a source.`

### Turn 7 — `submit_completed_task` (`14771 ms` LLM)
- **Rationale**: The packaged artifact passed isolated execution and independent checks of the Robin flux balance and both inverse fits. The follow-up audit confirmed axis exchange and signed-source consistency, with a worst relative global balance error of 2.43e-13. The verification receipt is bound to the unchanged artifact.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin_spd_profiler.py`. Both public cases and all three high-contrast stress cases passed. The worst relative PDE residual was 8.57e-10, below the 1e-8 limit. Both source estimators matched independent least-squares fits, including when a nuisance column aligned with a source.`

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
