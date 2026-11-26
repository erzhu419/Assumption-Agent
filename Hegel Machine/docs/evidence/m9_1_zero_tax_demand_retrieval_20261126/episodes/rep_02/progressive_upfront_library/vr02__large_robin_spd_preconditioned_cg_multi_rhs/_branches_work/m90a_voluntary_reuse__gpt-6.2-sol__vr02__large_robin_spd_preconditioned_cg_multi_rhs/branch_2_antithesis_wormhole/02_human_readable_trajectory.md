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
| **Total Tokens (Prompt / Cached / Out)** | `184,386` (`177,715` / `45,312` / `6,671`) |
| **Total Episode Wall Time (s)** | `221.62s` |

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

### Turn 1 — `execute_in_remote_docker` (`19444 ms` LLM)
- **Rationale**: Inspect public case dimensions and parameter ranges to select relevant stress slices, and check the mounted solver and profiling contracts before using them.
- **Alternatives Weighed**: `Reuse the mounted zero-ghost grid operator` (rejected: Its Dirichlet boundary conductance does not satisfy the specified half-cell Robin flux.)

```bash
ls -l /holdout_task; python3 - <<'PY'
import pickle
import numpy as np
with open('/holdout_task/public_robin_spd_cases.pkl','rb') as f:
    data=pickle.load(f)
def inspect(obj, path='cases', depth=0):
    if isinstance(obj,dict) and 'kappa' in obj:
        print(path)
        for k,v in obj.items():
            if isinstance(v,np.ndarray):
                print(' ',k,v.shape,v.dtype, (float(v.min()),float(v.max())) if v.size and np.issubdtype(v.dtype,np.number) else '')
            else:
                print(' ',k,repr(v)[:120])
    elif depth < 3 and isinstance(obj,(dict,list,tuple)):
        print(path,type(obj).__name__,len(obj))
        for k,v in (obj.items() if isinstance(obj,dict) else enumerate(obj)):
            inspect(v,f'{path}/{k}',depth+1)
    else:
        print(path,type(obj).__name__,getattr(obj,'shape',''))
inspect(data)
PY
for skill in autogenous_prim_conservative_spd_solve.galerkin_preconditioner autogenous_prim_conservative_spd_solve.solve_residual_checked_cg autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes; do python3 -m hegel_wormhole_skills --describe "$skill"; done
```
- **Container Exit Code**: `0` (`708 ms`, timed_out=`False`)
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

### Turn 2 — `execute_in_remote_docker` (`46073 ms` LLM)
- **Rationale**: Assemble the specified Robin conductance directly, reuse one Galerkin hierarchy for all ten right-hand sides, and allow numerical margin below the requested residual. Compare all three outputs to the public references; a PDE discrepancy would call for a stencil or convergence fix, while an isolated inverse discrepancy would call for checking the profile and ridge conventions.
- **Alternatives Weighed**: `Profile through a difference of raw and nuisance Gram energies` (rejected: That subtraction can lose precision when a sensor response is nearly in the nuisance span; the augmented residual gives the same objective without subtracting energies.)

```bash
cat > /holdout_task/robin_spd_profiler.py <<'PY'
"""Cell-centered Robin diffusion solves and nuisance-profiled source fitting."""

import numpy as np
from scipy import sparse
from hegel_wormhole_skills.autogenous_prim_conservative_spd_solve import (
    galerkin_preconditioner,
    solve_residual_checked_cg,
)


def _robin_operator(kappa, sigma, gamma, hx, hy):
    """Assemble each interior face as one symmetric flux contribution."""
    cells = np.arange(kappa.size).reshape(kappa.shape)
    fx = (2 * hy / hx) * kappa[:-1, :] * kappa[1:, :] / (kappa[:-1, :] + kappa[1:, :])
    fy = (2 * hx / hy) * kappa[:, :-1] * kappa[:, 1:] / (kappa[:, :-1] + kappa[:, 1:])
    diag = hx * hy * np.broadcast_to(sigma, kappa.shape).copy()
    diag[:-1, :] += fx
    diag[1:, :] += fx
    diag[:, :-1] += fy
    diag[:, 1:] += fy
    for i in (0, -1):
        kb = kappa[i, :]
        diag[i, :] += hy * gamma * kb / (0.5 * hx * gamma + kb)
    for j in (0, -1):
        kb = kappa[:, j]
        diag[:, j] += hx * gamma * kb / (0.5 * hy * gamma + kb)
    left = np.concatenate((cells[:-1, :].ravel(), cells[:, :-1].ravel()))
    right = np.concatenate((cells[1:, :].ravel(), cells[:, 1:].ravel()))
    face = np.concatenate((fx.ravel(), fy.ravel()))
    return sparse.csr_matrix(
        (np.concatenate((diag.ravel(), -face, -face)),
         (np.concatenate((cells.ravel(), left, right)),
          np.concatenate((cells.ravel(), right, left)))),
        shape=(kappa.size, kappa.size),
    )


def _profile(S, Z, Y, lam_z, lam_s):
    # Augmentation makes nuisance ridge an ordinary orthogonal projection.
    # Use the norm of the projected column rather than subtracting Gram energies.
    p = Z.shape[1]
    za = np.vstack((Z, np.sqrt(lam_z) * np.eye(p)))
    data = np.pad(np.column_stack((S, Y)), ((0, p), (0, 0)))
    coefficients = np.linalg.lstsq(za, data, rcond=None)[0]
    residual = data - za @ coefficients
    m = S.shape[1]
    sr, yr = residual[:, :m], residual[:, m:]
    energy = np.einsum('ij,ij->j', sr, sr)
    dots = sr.T @ yr
    single = np.divide(dots, energy[:, None], out=np.zeros_like(dots),
                       where=energy[:, None] > 0)
    joint = np.linalg.lstsq(
        np.vstack((sr, np.sqrt(lam_s) * np.eye(m))),
        np.pad(yr, ((0, m), (0, 0))), rcond=None,
    )[0]
    return single, joint


def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Return the converged source fields and both sets of fitted amplitudes."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    nx, ny = kappa.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    A = _robin_operator(kappa, np.asarray(problem['sigma'], dtype=float),
                        problem['gamma_robin'], hx, hy)
    preconditioner = galerkin_preconditioner(A)
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    fields = np.empty_like(sources)
    for index, q in enumerate(sources):
        field, _ = solve_residual_checked_cg(
            A, (hx * hy) * q.ravel(), preconditioner=preconditioner,
            tol=2e-10, maxiter=600,
        )
        fields[index] = field.reshape(kappa.shape)
    ij = np.asarray(problem['sensor_ij'])
    S = fields[:, ij[:, 0], ij[:, 1]].T
    single, joint = _profile(S, np.asarray(problem['nuisance_design'], dtype=float),
                             np.asarray(problem['Y_obs'], dtype=float),
                             problem['nuisance_ridge'], problem['source_ridge'])
    return {'basis_fields': fields,
            'single_source_profiled_amplitudes': single,
            'joint_source_weights': joint}
PY
PYTHONPATH=/holdout_task:/tmp OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle, time
import numpy as np
from robin_spd_profiler import solve_robin_spd_and_profile_sources, _robin_operator
for i, case in enumerate(pickle.load(open('/holdout_task/public_robin_spd_cases.pkl','rb'))):
    p=case['problem']; t=time.perf_counter(); actual=solve_robin_spd_and_profile_sources(p)
    hx=p['Lx']/72; hy=p['Ly']/72
    A=_robin_operator(p['kappa'],p['sigma'],p['gamma_robin'],hx,hy)
    b=p['candidate_sources'].reshape(10,-1).T*hx*hy
    r=np.linalg.norm(A@actual['basis_fields'].reshape(10,-1).T-b,axis=0)/np.linalg.norm(b,axis=0)
    print(i,'seconds',round(time.perf_counter()-t,3),'worst residual',max(r))
    for key, ref in case['gt'].items():
        diff=actual[key]-ref
        print(key,'max abs',np.max(np.abs(diff)),'rel L2',np.linalg.norm(diff)/np.linalg.norm(ref))
PY
```
- **Container Exit Code**: `0` (`1008 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/robin_spd_profiler.py']`
- **Stdout (tail)**:
```text
0 seconds 0.231 worst residual 1.294310461594407e-10
basis_fields max abs 4.0587325167429356e-14 rel L2 6.082411368806458e-13
single_source_profiled_amplitudes max abs 2.8741453661496053e-12 rel L2 3.5612515938507324e-13
joint_source_weights max abs 1.8021695247227854e-13 rel L2 4.841741254072961e-13
1 seconds 0.228 worst residual 1.3651441578648087e-10
basis_fields max abs 1.0295930774617545e-13 rel L2 6.817772158101576e-13
single_source_profiled_amplitudes max abs 2.3667734438959087e-12 rel L2 5.531187743117535e-13
joint_source_weights max abs 2.566835632933362e-13 rel L2 4.726117909387119e-13

[HOST STATIC PACKAGER]: Bundled self-contained `hegel_wormhole_skills` closure into `['/holdout_task/robin_spd_profiler.py']` (`pre_sha=6d1b9dea6dbe` -> `post_sha=5a33c7e94724`). Before `submit_completed_task`, run a clean `python3 -I` verification on the packaged target artifact so `VerificationReceipt` binds to `post_sha`.
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5a33c7e94724aaf4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5a33c7e9 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`45458 ms` LLM)
- **Rationale**: Verify the packaged artifact in isolated Python. Independent face fluxes can expose stencil errors hidden by a self-residual; direct augmented fits can expose ridge or profiling errors; contrast, aspect ratio, and Robin stresses can expose a convergence margin that the two public cases miss.
- **Alternatives Weighed**: `Submit from the initial public comparison` (rejected: That comparison preceded packaging and used the solver's own matrix to measure residuals.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import time
import numpy as np

path = '/holdout_task/robin_spd_profiler.py'
spec = importlib.util.spec_from_file_location('robin_spd_profiler', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin_spd_cases.pkl', 'rb') as f:
    public = pickle.load(f)

basis_key = 'basis_fields'
single_key = 'single_source_profiled_amplitudes'
joint_key = 'joint_source_weights'

def residuals(p, u):
    k = p['kappa']; g = p['gamma_robin']
    nx, ny = k.shape
    dx, dy = p['Lx']/nx, p['Ly']/ny
    b = dx*dy*p['candidate_sources']
    ax = dx*dy*p['sigma']*u
    # Independent face-flux calculation; no use of the solver's sparse matrix.
    xflow = (u[:, :-1, :] - u[:, 1:, :]) * (dy/dx) / (0.5/k[:-1, :] + 0.5/k[1:, :])
    yflow = (u[:, :, :-1] - u[:, :, 1:]) * (dx/dy) / (0.5/k[:, :-1] + 0.5/k[:, 1:])
    ax[:, :-1, :] += xflow; ax[:, 1:, :] -= xflow
    ax[:, :, :-1] += yflow; ax[:, :, 1:] -= yflow
    for i in (0, -1):
        ax[:, i, :] += dy * u[:, i, :] / (1/g + dx/(2*k[i, :]))
    for j in (0, -1):
        ax[:, :, j] += dx * u[:, :, j] / (1/g + dy/(2*k[:, j]))
    norms = np.linalg.norm(b, axis=(1,2))
    err = np.linalg.norm(ax-b, axis=(1,2))
    assert np.all(err[norms == 0] == 0), 'zero source did not give zero flux residual'
    return max(err[norms > 0] / norms[norms > 0], default=0.)

def inverse_reference(p, u):
    ij = p['sensor_ij']; S = u[:, ij[:,0], ij[:,1]].T
    Z = p['nuisance_design']; Y = p['Y_obs']
    nz, ms = Z.shape[1], S.shape[1]
    rz, rs = np.sqrt(p['nuisance_ridge']), np.sqrt(p['source_ridge'])
    target = np.pad(Y, ((0,nz),(0,0)))
    single = np.empty((ms, Y.shape[1]))
    for m in range(ms):
        design = np.vstack((np.column_stack((Z, S[:,m])), np.column_stack((rz*np.eye(nz), np.zeros(nz)))))
        single[m] = np.linalg.lstsq(design, target, rcond=None)[0][-1]
    design = np.block([[Z, S], [rz*np.eye(nz), np.zeros((nz,ms))], [np.zeros((ms,nz)), rs*np.eye(ms)]])
    joint = np.linalg.lstsq(design, np.pad(target, ((0,ms),(0,0))), rcond=None)[0][-ms:]
    return single, joint

def check(name, p, gt=None):
    start = time.perf_counter()
    out = mod.solve_robin_spd_and_profile_sources(p)
    assert set(out) == {basis_key, single_key, joint_key}
    assert out[basis_key].shape == p['candidate_sources'].shape
    assert np.isfinite(out[basis_key]).all()
    residual = residuals(p, out[basis_key]); assert residual <= 1e-8, (name, residual)
    single, joint = inverse_reference(p, out[basis_key])
    for key, ref in ((single_key, single), (joint_key, joint)):
        assert out[key].shape == ref.shape
        np.testing.assert_allclose(out[key], ref, rtol=1e-8, atol=1e-9)
    ref_errors = {}
    if gt is not None:
        for key, ref in gt.items():
            np.testing.assert_allclose(out[key], ref, rtol=1e-8, atol=1e-9)
            ref_errors[key] = float(np.max(np.abs(out[key]-ref)))
    print(name, 'worst PDE residual', residual, 'inverse max abs vs augmented direct', float(np.max(np.abs(out[single_key]-single))), float(np.max(np.abs(out[joint_key]-joint))), 'public max abs', ref_errors, 'seconds', round(time.perf_counter()-start, 3))

for i, case in enumerate(public):
    check(f'public-{i}', case['problem'], case['gt'])
base = public[0]['problem']
i, j = np.indices(base['kappa'].shape)
q = base['candidate_sources'].copy(); q[0] = 0
check('checkerboard, weak Robin, rectangular cells, zero source', dict(base, kappa=np.where((i+j)%2, 80., .02), Lx=2.0, Ly=.7, gamma_robin=.06, candidate_sources=q))
check('high-conductivity stripes, strong Robin, rectangular cells', dict(base, kappa=np.where((i//4)%2, 80., .02), Lx=.65, Ly=1.9, gamma_robin=24.))
check('uniform high conductivity', dict(base, kappa=np.full_like(base['kappa'], 80.)))
PY
```
- **Container Exit Code**: `0` (`2695 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public-0 worst PDE residual 1.2936182914038772e-10 inverse max abs vs augmented direct 1.1102230246251565e-14 2.5673907444456745e-15 public max abs {'basis_fields': 4.0587325167429356e-14, 'single_source_profiled_amplitudes': 2.8741453661496053e-12, 'joint_source_weights': 1.8021695247227854e-13} seconds 0.35
public-1 worst PDE residual 1.3636210122779859e-10 inverse max abs vs augmented direct 5.329070518200751e-15 1.3322676295501878e-15 public max abs {'basis_fields': 1.0295930774617545e-13, 'single_source_profiled_amplitudes': 2.3667734438959087e-12, 'joint_source_weights': 2.566835632933362e-13} seconds 0.258
checkerboard, weak Robin, rectangular cells, zero source worst PDE residual 1.6934489263589033e-10 inverse max abs vs augmented direct 3.4486302702418925e-15 1.4363510381087963e-15 public max abs {} seconds 0.208
high-conductivity stripes, strong Robin, rectangular cells worst PDE residual 1.6048323849538481e-10 inverse max abs vs augmented direct 1.1102230246251565e-14 6.938893903907228e-16 public max abs {} seconds 1.06
uniform high conductivity worst PDE residual 1.6944039647619783e-10 inverse max abs vs augmented direct 1.6484591469634324e-12 3.6461719060687514e-16 public max abs {} seconds 0.188

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5a33c7e94724aaf4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5a33c7e9 (directory).
```

### Turn 4 — `submit_completed_task` (`12846 ms` LLM)
- **Rationale**: The packaged artifact passed isolated Python verification against independent face-flux residuals and direct augmented least-squares fits on both public cases and three stress cases. The verification receipt is bound to the submitted artifact.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`14` turns remaining; artifact status={'/holdout_task/robin_spd_profiler.py': 'FILE:9341'}): Do NOT lock prematurely with `14` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin_spd_profiler.py`. The worst relative PDE residual across the public and stress cases was 1.70e-10, below the 1e-8 limit. Both inverse outputs agreed with independent least-squares fits.`

### Turn 5 — `execute_in_remote_docker` (`34014 ms` LLM)
- **Rationale**: Audit the packaged artifact on the supplied cases without changing it. Global balance can expose missing wall or corner flux, signed superposition can expose source indexing or sign errors, and a controlled shift along the measured nuisance design can expose accidental unpenalized centering or an incorrect ridge response.
- **Alternatives Weighed**: `Apply batch centering or relay-mode tests` (rejected: This task has neither batch labels nor relay states. The nuisance design and its specified ridge penalty determine the appropriate adjustment.)

```bash
cat > /tmp/audit_robin_profile.py <<'PY'
import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('robin_spd_profiler', '/holdout_task/robin_spd_profiler.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin_spd_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

for i, case in enumerate(cases):
    p, gt = case['problem'], case['gt']
    base = mod.solve_robin_spd_and_profile_sources(p)
    fields = base['basis_fields']
    k, g = p['kappa'], p['gamma_robin']
    dx, dy = p['Lx']/k.shape[0], p['Ly']/k.shape[1]
    mass = (dx*dy*p['candidate_sources']).sum(axis=(1,2))
    loss = (dx*dy*p['sigma']*fields).sum(axis=(1,2))
    for edge in (0, -1):
        loss += (dy * fields[:,edge,:] / (1/g + dx/(2*k[edge,:]))).sum(axis=1)
        loss += (dx * fields[:,:,edge] / (1/g + dy/(2*k[:,edge]))).sum(axis=1)
    balance = np.max(np.abs(loss-mass)/np.abs(mass))
    assert balance < 1e-8, (i, 'global flux balance', balance)

    ij = p['sensor_ij']; Z, Y = p['nuisance_design'], p['Y_obs']
    S = fields[:,ij[:,0],ij[:,1]].T
    mz, ms = Z.shape[1], S.shape[1]
    c = np.linspace(-.6,.6,mz)[:,None] * np.linspace(.5,1.5,Y.shape[1])
    dy_obs = Z @ c
    changed = mod.solve_robin_spd_and_profile_sources(dict(p, Y_obs=Y+dy_obs))
    errs = []
    for key, ridge, groups in (
        ('single_source_profiled_amplitudes', 0., [S[:,j:j+1] for j in range(ms)]),
        ('joint_source_weights', p['source_ridge'], [S]),
    ):
        expected = []
        for G in groups:
            d = G.shape[1]
            X = np.column_stack((Z, G))
            penalty = np.diag(np.sqrt(np.r_[np.full(mz,p['nuisance_ridge']),np.full(d,ridge)]))
            coef = np.linalg.lstsq(np.vstack((X, penalty)), np.pad(dy_obs, ((0,mz+d),(0,0))), rcond=None)[0]
            expected.append(coef[mz:])
        expected = np.vstack(expected)
        delta = changed[key] - base[key]
        np.testing.assert_allclose(delta, expected, rtol=1e-8, atol=1e-10)
        errs.append((key, float(np.max(np.abs(delta-expected))), float(np.max(np.abs(expected)))))

    q = p['candidate_sources'].copy(); q[0] -= .7*q[1]
    mixed = mod.solve_robin_spd_and_profile_sources(dict(p, candidate_sources=q))['basis_fields'][0]
    ref = gt['basis_fields'][0] - .7*gt['basis_fields'][1]
    superposition = np.linalg.norm(mixed-ref)/np.linalg.norm(ref)
    assert superposition < 1e-8, (i, 'signed superposition', superposition)
    print('public', i, 'global balance rel', balance, 'signed superposition rel', superposition, 'nuisance response (max error, max expected magnitude)', errs)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_robin_profile.py
```
- **Container Exit Code**: `0` (`1830 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public 0 global balance rel 1.7413312988535563e-13 signed superposition rel 1.2100260510241423e-12 nuisance response (max error, max expected magnitude) [('single_source_profiled_amplitudes', 8.904335602188951e-15, 0.005666343452543999), ('joint_source_weights', 9.872744982653003e-16, 0.0003887552163910329)]
public 1 global balance rel 2.310997197036254e-13 signed superposition rel 1.1756894294283268e-12 nuisance response (max error, max expected magnitude) [('single_source_profiled_amplitudes', 9.179289273131275e-15, 0.004277893982492546), ('joint_source_weights', 2.055145875527742e-15, 0.000282353231413286)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5a33c7e94724aaf4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5a33c7e9 (directory).
```

### Turn 6 — `submit_completed_task` (`13853 ms` LLM)
- **Rationale**: The packaged artifact passed isolated verification. The follow-up audit also confirmed global flux balance, signed source superposition, and the expected response to a ridge-penalized nuisance shift. No change was needed, and the verification receipt remains bound to the artifact.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/robin_spd_profiler.py': 'FILE:9341'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin_spd_profiler.py`. Across both public cases and three stress cases, the worst relative PDE residual was 1.70e-10, below the 1e-8 limit. Both source-fitting outputs agreed with independent least-squares fits.`

### Turn 7 — `submit_completed_task` (`18204 ms` LLM)
- **Rationale**: The unchanged packaged artifact passed clean-process verification and the focused audit of flux balance, signed source superposition, and ridge-penalized nuisance shifts. The repeated gate identifies no new task-relevant failure mode, and the verification receipt remains bound.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/robin_spd_profiler.py`. Across two public cases and three stress cases, the worst relative PDE residual was 1.70e-10, below the 1e-8 limit. Both source-fitting outputs agreed with independent least-squares fits.`

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
