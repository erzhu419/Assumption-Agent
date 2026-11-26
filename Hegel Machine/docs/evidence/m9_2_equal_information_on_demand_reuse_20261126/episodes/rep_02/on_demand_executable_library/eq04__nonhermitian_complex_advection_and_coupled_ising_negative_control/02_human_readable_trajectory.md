# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `eq04__nonhermitian_complex_advection_and_coupled_ising_negative_control`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M92-EqualInfo-4` |
| **Domain** | `Non-Hermitian Complex Advection-Reaction PDE & Non-Separable Pairwise Co-Activation Subset Selection` |
| **Wave / Wormhole Family** | `m92_equal_info` / `fam_indefinite_wave_scattering_negative_control` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`PASS (1.0)`** |
| **Failure / Outcome Taxonomy** | `PASSED_ALL_CHECKS` |
| **Turns Used / Max Turns** | `9 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `196,129` (`187,364` / `51,840` / `8,765`) |
| **Total Episode Wall Time (s)** | `301.78s` |

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

### Turn 1 — `execute_in_remote_docker` (`19896 ms` LLM)
- **Rationale**: Inspect the validator and print only public-case metadata. A schema mismatch would change input handling or the verification harness; a suitable complex-ridge primitive could replace custom enumeration code.
- **Alternatives Weighed**: `SPD CG or AMG for the PDE` (rejected: The specified upwind advection and imaginary reaction produce a non-Hermitian operator.); `Greedy or separable feature selection` (rejected: Positive pair penalties make support costs nonseparable, and all 2,048 supports are cheap to enumerate.)

```bash
ls -lah /holdout_task; sed -n '1,220p' /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'complex Hermitian ridge support enumeration pairwise penalties' 2>&1; python3 - <<'PY'
import pickle, numpy as np, scipy
p='/holdout_task/public_eq04_cases.pkl'
with open(p,'rb') as f: d=pickle.load(f)
print('numpy/scipy:',np.__version__,scipy.__version__)
print('public root:',type(d).__name__,len(d))
def show(x,depth=0):
 if isinstance(x,dict):
  print(' '*depth,'keys:',list(x)[:30])
  if depth<3:
   for k,v in list(x.items())[:12]:
    print(' '*depth,k,type(v).__name__,getattr(v,'shape',''),getattr(v,'dtype',''))
    if isinstance(v,(dict,list,tuple)): show(v,depth+1)
 elif isinstance(x,(list,tuple)):
  print(' '*depth,'count:',len(x))
  if x and depth<3: show(x[0],depth+1)
show(d)
PY
```
- **Container Exit Code**: `1` (`766 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task': No such file or directory
sed: can't read /holdout_task/validate_public_schema.py: No such file or directory
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
FileNotFoundError: [Errno 2] No such file or directory: '/holdout_task/public_eq04_cases.pkl'
```

### Turn 2 — `execute_in_remote_docker` (`19200 ms` LLM)
- **Rationale**: Look for staged public inputs and confirm SciPy and target write access. If the inputs are present, use them for reference comparisons; otherwise verify against an independently assembled stencil and exhaustive reference on generated cases without claiming public validation.
- **Alternatives Weighed**: `Reuse the boxed L0 ridge certifier` (rejected: The search identifies it as a real, boxed regression solver; the required support couples each complex coefficient's two components and adds penalties for active pairs.)

```bash
pwd; ls -ld /pretrain_task /holdout_task /app /workspace /results 2>/dev/null; for d in /pretrain_task /holdout_task /app /workspace; do if [ -d "$d" ]; then find "$d" -maxdepth 3 -type f \( -name '*eq04*' -o -name '*nonhermitian*' -o -name 'validate_public_schema.py' -o -name 'AGENTS.md' \) -printf '%p %s bytes\n'; fi; done; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0; python3 -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'; mkdir -p /holdout_task && test -w /holdout_task && echo 'Target directory is writable'
```
- **Container Exit Code**: `0` (`699 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/pretrain_task
drwxr-xr-x 2 root root 4096 Nov 23 09:31 /pretrain_task
{
  "ok": true,
  "target": "autogenous_sparse_ridge_certification.solve_and_certify_l0",
  "matches": [
    {
      "module_stem": "autogenous_sparse_ridge_certification",
      "module_import": "hegel_wormhole_skills.autogenous_sparse_ridge_certification",
      "granularity_tier": "L2_domain_operator",
      "compiled_code_sha256": "770f6644bb1fd8eb49dbacf51ffa3405a9ca876974e0021d66415d9014d8dfb9",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "n_samples",
          "op": ">=",
          "value": 1
        },
        {
          "field": "n_features",
          "op": "<=",
          "value": 16
        },
        {
          "field": "lam0",
          "op": ">=",
          "value": 0
        },
        {
          "field": "lam2",
          "op": ">=",
          "value": 0
        },
        {
          "field": "M",
          "op": ">=",
          "value": 0
        },
        {
          "field": "eps",
          "op": ">",
          "value": 0
        },
        {
          "field": "max_leaves",
          "op": ">=",
          "value": 1
        }
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {},
      "functions": [
        {
          "function": "solve_and_certify_l0",
          "signature": "solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict",
          "doc_summary": "Certify boxed sparse ridge regression with at most 16 real predictors."
        }
      ],
      "index_cards": [
        "autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0"
      ]
    }
  ]
}
numpy 1.26.4 scipy 1.13.0
Target directory is writable
```

### Turn 3 — `execute_in_remote_docker` (`55807 ms` LLM)
- **Rationale**: Implement the specified stencil and support rule first. Next, compare the exact artifact against a separately assembled scalar stencil and uncompressed ridge reference, including a case where pair penalties change the selected support.
- **Alternatives Weighed**: `Solve each support via normal equations` (rejected: Augmented QR preserves the ridge objective without squaring the condition number; compressing once keeps all support solves small.)

```bash
ls -lah /pretrain_task /holdout_task; cat > /holdout_task/nonhermitian_advection_coupled_select.py <<'PY'
"""Complex upwind forward model and exhaustive coupled ridge selection."""

from itertools import combinations

import numpy as np
from scipy.linalg import lstsq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _responses(problem, basis):
    m, nx, ny = basis.shape
    hx = float(problem['Lx']) / (nx + 1)
    hy = float(problem['Ly']) / (ny + 1)
    dx = problem['nu'] / hx**2
    dy = problem['nu'] / hy**2
    ux = np.asarray(problem['vx'], dtype=float) / hx
    uy = np.asarray(problem['vy'], dtype=float) / hy
    diag = (2 * (dx + dy) + np.abs(ux) + np.abs(uy)
            + np.asarray(problem['sigma'], dtype=float) + 1j * problem['omega'])

    ids = np.arange(nx * ny).reshape(nx, ny)
    rows = np.concatenate((ids.ravel(), ids[1:, :].ravel(), ids[:-1, :].ravel(),
                           ids[:, 1:].ravel(), ids[:, :-1].ravel()))
    cols = np.concatenate((ids.ravel(), ids[:-1, :].ravel(), ids[1:, :].ravel(),
                           ids[:, :-1].ravel(), ids[:, 1:].ravel()))
    values = np.concatenate((diag.ravel(),
                             (-dx - np.maximum(ux[1:, :], 0)).ravel(),
                             (-dx + np.minimum(ux[:-1, :], 0)).ravel(),
                             (-dy - np.maximum(uy[:, 1:], 0)).ravel(),
                             (-dy + np.minimum(uy[:, :-1], 0)).ravel()))
    operator = coo_matrix((values, (rows, cols)), shape=(ids.size, ids.size),
                          dtype=np.complex128).tocsc()
    rhs = basis.reshape(m, ids.size).T.astype(np.complex128)
    return splu(operator).solve(rhs)


def _select(design, observed, lam0, lam2, pairs):
    m = design.shape[1]
    best = np.zeros(m, dtype=np.complex128)
    best_value = float(0.5 * np.vdot(observed, observed).real)
    # Compress the augmented ridge problem once. The discarded residual is
    # independent of the support and is included when evaluating the winner.
    augmented = np.vstack((design, np.sqrt(lam2) * np.eye(m)))
    q, r = np.linalg.qr(augmented, mode='reduced')
    target = q[:observed.size].conj().T @ observed
    best_score = float(0.5 * np.vdot(target, target).real)
    for size in range(1, m + 1):
        for support in combinations(range(m), size):
            submatrix = r[:, support]
            weights = lstsq(submatrix, target, lapack_driver='gelsy',
                            check_finite=False)[0]
            residual = submatrix @ weights - target
            active = {i for i, w in zip(support, weights) if abs(w) > 1e-8}
            penalty = lam0 * len(active) + sum(gamma for u, v, gamma in pairs
                                               if u in active and v in active)
            score = float(0.5 * np.vdot(residual, residual).real + penalty)
            if score < best_score:
                best_score = score
                best.fill(0)
                best[list(support)] = weights
                best_value = penalty

    residual = design @ best - observed
    active = np.abs(best) > 1e-8
    penalty = lam0 * np.count_nonzero(active) + sum(gamma for u, v, gamma in pairs
                                                   if active[u] and active[v])
    value = 0.5 * (np.vdot(residual, residual).real + lam2 * np.vdot(best, best).real)
    return best, float(value + penalty)


def solve_nonhermitian_coupled_inversion(problem: dict) -> dict:
    """Return the complex ridge minimizer selected across all supports."""
    basis = np.asarray(problem['basis'], dtype=float)
    _, nx, ny = basis.shape
    responses = _responses(problem, basis)
    sensors = np.asarray(problem['sensor_ij'], dtype=np.intp)
    design = responses[sensors[:, 0] * ny + sensors[:, 1]]
    observed = (np.asarray(problem['y_obs_re'], dtype=float)
                + 1j * np.asarray(problem['y_obs_im'], dtype=float))
    pairs = [(int(u), int(v), float(gamma)) for u, v, gamma in problem['pair_penalties']]
    weights, objective = _select(design, observed, float(problem['lam0']),
                                 float(problem['lam2']), pairs)
    return {'weights_real': weights.real.copy(),
            'weights_imag': weights.imag.copy(),
            'objective': objective,
            'predicted_field_abs': np.abs(responses @ weights).reshape(nx, ny)}
PY
sed -i '/    best_value = float(0.5 \* np.vdot(observed, observed).real)/d; /                best_value = penalty/d' /holdout_task/nonhermitian_advection_coupled_select.py
```
- **Container Exit Code**: `0` (`369 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/nonhermitian_advection_coupled_select.py']`
- **Stdout (tail)**:
```text
/holdout_task:
total 8.0K
drwxr-xr-x 2 root root 4.0K Nov 26 04:51 .
drwxr-xr-x 1 root root 4.0K Nov 26 04:51 ..

/pretrain_task:
total 8.0K
drwxr-xr-x 2 root root 4.0K Nov 23 09:31 .
drwxr-xr-x 1 root root 4.0K Nov 26 04:51 ..

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=33b107d831d3fd04 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_33b107d8 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`54418 ms` LLM)
- **Rationale**: The public files are absent from the staged container. Use an independent scalar stencil and uncompressed exhaustive reference to distinguish assembly errors from selection errors, and a two-variable test that fails if pair costs do not affect selection.

```bash
cat > /tmp/verify_eq04.py <<'PY'
import importlib.util
import itertools
import time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

path = '/holdout_task/nonhermitian_advection_coupled_select.py'
spec = importlib.util.spec_from_file_location('candidate', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def reference_field(p):
    m, nx, ny = p['basis'].shape
    hx, hy = p['Lx']/(nx+1), p['Ly']/(ny+1)
    ax, ay = p['nu']/hx**2, p['nu']/hy**2
    matrix = lil_matrix((nx*ny, nx*ny), dtype=np.complex128)
    for i, j in itertools.product(range(nx), range(ny)):
        r = i*ny+j
        matrix[r, r] = 2*ax+2*ay+p['sigma'][i,j]+1j*p['omega']
        for di, dj, c in ((-1,0,ax),(1,0,ax),(0,-1,ay),(0,1,ay)):
            if 0 <= i+di < nx and 0 <= j+dj < ny:
                matrix[r, (i+di)*ny+j+dj] = -c
        for axis, h, vel in ((0,hx,p['vx'][i,j]),(1,hy,p['vy'][i,j])):
            c = abs(vel)/h
            matrix[r,r] += c
            step = -1 if vel >= 0 else 1
            ii, jj = (i+step,j) if axis == 0 else (i,j+step)
            if 0 <= ii < nx and 0 <= jj < ny:
                matrix[r,ii*ny+jj] -= c
    matrix = matrix.tocsc()
    rhs = p['basis'].reshape(m,-1).T.astype(complex)
    return matrix, rhs, spsolve(matrix, rhs)

def objective(g,y,w,p):
    a = abs(w)>1e-8
    return (0.5*np.linalg.norm(g@w-y)**2 + 0.5*p['lam2']*np.linalg.norm(w)**2
            + p['lam0']*a.sum() + sum(q for u,v,q in p['pair_penalties'] if a[u] and a[v]))

def reference_select(g,y,p):
    m = g.shape[1]
    w = np.zeros(m,dtype=complex)
    best = w.copy()
    val = objective(g,y,w,p)
    for mask in range(1,1<<m):
        s = [j for j in range(m) if (mask>>j)&1]
        a = np.vstack((g[:,s],np.sqrt(p['lam2'])*np.eye(len(s))))
        b = np.concatenate((y,np.zeros(len(s))))
        w.fill(0)
        w[s] = np.linalg.lstsq(a,b,rcond=None)[0]
        candidate = objective(g,y,w,p)
        if candidate < val:
            best,val = w.copy(),candidate
    return best,val

rng = np.random.default_rng(274)
worst = np.zeros(4)
for name,nu,lengths,omega,speed,ns,lam2 in [
    ('mixed_sign_anisotropic',.04,(1.6,.7),-7.,3.,55,1e-5),
    ('advection_dominated',1e-5,(.8,2.1),2.5,11.,32,1e-6),
    ('rapid_oscillation',.013,(1.,1.),160.,0.,80,1e-8),
    ('redundant_basis_few_sensors',.01,(1.1,.9),-12.,1.4,6,1e-10),
    ('diffusion_and_empty_support',.2,(.7,1.3),.07,.05,70,1e-5),
]:
    n=32
    x,ygrid=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n),indexing='ij')
    basis=np.stack([np.sin(np.pi*(j%4+1)*x)*np.cos(np.pi*(j//4+1)*ygrid) for j in range(11)])
    basis += rng.normal(0,.12,basis.shape)
    if 'redundant' in name:
        basis[1]=basis[0]+1e-7*basis[1]
    sh=(n,n)
    flat=np.r_[0,n-1,n*(n-1),n*n-1,rng.choice(np.arange(1,n*n-1),ns-4,replace=False)]
    p=dict(Lx=lengths[0],Ly=lengths[1],nu=nu,omega=omega,
           vx=speed*rng.uniform(-1,1,sh),vy=speed*rng.uniform(-1,1,sh),
           sigma=rng.uniform(.1,1,sh),basis=basis,
           sensor_ij=np.column_stack(np.unravel_index(flat,sh)),lam2=lam2)
    a,b,field=reference_field(p)
    wanted=np.zeros(11,dtype=complex)
    wanted[[0,2,7]]=[1+1.4j,-2+.6j,.7-1.6j]
    g=field[flat]
    y=g@wanted
    energy=np.vdot(y,y).real
    p.update(y_obs_re=y.real,y_obs_im=y.imag,
             lam0=(2 if 'empty' in name else .025)*energy,
             pair_penalties=[(0,2,.35*energy),(2,7,.4*energy),(1,8,.1*energy)])
    t=time.perf_counter()
    r=mod.solve_nonhermitian_coupled_inversion(p)
    runtime=time.perf_counter()-t
    w=r['weights_real']+1j*r['weights_imag']
    wt,jt=reference_select(g,y,p)
    ft=abs(field@wt).reshape(sh)
    responses=mod._responses(p,basis)
    metrics=np.array([np.linalg.norm(a@responses-b)/np.linalg.norm(b),abs(r['objective']-jt),
           np.linalg.norm(r['predicted_field_abs']-ft)/(np.linalg.norm(ft) or 1),
           abs(objective(g,y,w,p)-r['objective'])])
    worst=np.maximum(worst,metrics)
    assert set(r)=={'weights_real','weights_imag','objective','predicted_field_abs'}
    assert isinstance(r['objective'],float)
    for key, shape in [('weights_real',(11,)),('weights_imag',(11,)),('predicted_field_abs',sh)]:
        assert r[key].shape==shape and r[key].dtype==np.float64 and np.isfinite(r[key]).all()
    assert np.all(metrics < [1e-9,1e-6,1e-5,1e-6]), (name,metrics)
    if 'empty' in name:
        assert not np.any(w) and not np.any(r['predicted_field_abs'])
    print(name, 'errors', metrics.tolist(), 'seconds',round(runtime,4), 'support',np.flatnonzero(abs(w)>1e-8).tolist())

g=np.eye(11,dtype=complex)
y=np.array([2+1j,1+1j]+[0j]*9)
w0,_=mod._select(g,y,.1,.01,[])
w1,_=mod._select(g,y,.1,.01,[(0,1,2.)])
assert np.array_equal(np.flatnonzero(abs(w0)>1e-8),[0,1])
assert np.array_equal(np.flatnonzero(abs(w1)>1e-8),[0])
print('Pair penalty changes support from [0, 1] to [0].')
print('Worst stencil residual, objective error, relative field error, objective self-consistency:',worst.tolist())
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_eq04.py
```
- **Container Exit Code**: `0` (`1514 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
mixed_sign_anisotropic errors [4.975549778092592e-15, 0.0, 5.48522801651526e-16, 0.0] seconds 0.0371 support [0, 7, 10]
advection_dominated errors [2.2717479742444695e-14, 2.6645352591003757e-15, 2.3671645980822195e-15, 2.6645352591003757e-15] seconds 0.0461 support [0, 3, 7]
rapid_oscillation errors [2.738672516425656e-16, 0.0, 3.827555088729577e-16, 0.0] seconds 0.0423 support [0, 7]
redundant_basis_few_sensors errors [1.5204860604298187e-15, 8.673617379884035e-19, 2.7555693278376125e-16, 8.673617379884035e-19] seconds 0.0389 support [2]
diffusion_and_empty_support errors [9.320488157686814e-15, 1.6653345369377348e-16, 0.0, 1.6653345369377348e-16] seconds 0.0506 support []
Pair penalty changes support from [0, 1] to [0].
Worst stencil residual, objective error, relative field error, objective self-consistency: [2.2717479742444695e-14, 2.6645352591003757e-15, 2.3671645980822195e-15, 2.6645352591003757e-15]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=33b107d831d3fd04 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_33b107d8 (directory).
```

### Turn 5 — `submit_completed_task` (`20237 ms` LLM)
- **Rationale**: The isolated-process checks passed on the archived artifact against an independent stencil and exhaustive ridge reference, including a case where the pair penalty changed the optimum. Both numerical errors are below the requested tolerances.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`13` turns remaining; artifact status={'/holdout_task/nonhermitian_advection_coupled_select.py': 'FILE:4158'}): Do NOT lock prematurely with `13` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/nonhermitian_advection_coupled_select.py`. Five synthetic stress cases passed independent verification. Worst objective error: 2.66e-15; worst relative field error: 2.37e-15. The pair penalty also changed the selected support as expected.

Limitations: The public fixtures were unavailable. The solver follows the specified ridge-per-support rule; the literal thresholded objective can also admit nonzero, unpenalized coefficients at or below 1e-8 that are not ridge solutions on any support.`

### Turn 6 — `execute_in_remote_docker` (`30308 ms` LLM)
- **Rationale**: An orthonormal complex design gives a closed-form reference that isolates pair-cost algebra, including the fully connected case. A separate threshold witness quantifies where the literal objective and the prescribed zero-on-omitted-support algorithm differ. If the public files have appeared, use the validator to check the relevant contract.
- **Alternatives Weighed**: `Apply batch, relay, or drift-model diagnostics` (rejected: The inputs define a deterministic PDE and subset optimization with no batches, relay state, or drift model.)

```bash
if [ -f /holdout_task/public_eq04_cases.pkl ] && [ -f /holdout_task/validate_public_schema.py ]; then python3 /holdout_task/validate_public_schema.py || exit; else printf '%s\n' 'Public fixtures or validator still unavailable'; fi
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import numpy as np
p='/holdout_task/nonhermitian_advection_coupled_select.py'
s=importlib.util.spec_from_file_location('solver',p)
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
rng=np.random.default_rng(9401)
q,_=np.linalg.qr(rng.normal(size=(19,11))+1j*rng.normal(size=(19,11)))
y0=np.arange(1,12)*(0.2+0.13j)
lam2=0.03
lam0=0.06
worst=np.zeros(2)
for count in (0,15,55):
    pairs=[(u,v,0.08+rng.random()*0.7) for u in range(11) for v in range(u+1,11)][:count]
    delta=0.5*abs(y0)**2/(1+lam2)-lam0
    gain,mask=max((sum(delta[j] for j in range(11) if b>>j&1)-sum(g for u,v,g in pairs if (b>>u&1) and (b>>v&1)),b) for b in range(2048))
    expected=np.array([y0[j]/(1+lam2) if mask>>j&1 else 0j for j in range(11)])
    w,objective=m._select(q,q@y0,lam0,lam2,pairs)
    errors=[abs(objective-(0.5*np.vdot(y0,y0).real-gain)),np.linalg.norm(w-expected)]
    worst=np.maximum(worst,errors)
    assert max(errors)<1e-11, (count,errors)
print('Orthonormal complex references, 0/15/55 pair costs, worst objective and weight errors:',worst.tolist())
y=np.zeros(11,dtype=complex)
y[0]=1
w,j=m._select(np.eye(11),y,1.,.1,[])
edge=1e-8
jedge=0.5*(1-edge)**2+0.5*.1*edge**2
assert not np.any(w) and jedge<j
print('Literal inactive-disk versus zero-on-omitted-support objective gap:',j-jedge)
PY
```
- **Container Exit Code**: `0` (`899 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public fixtures or validator still unavailable
Orthonormal complex references, 0/15/55 pair costs, worst objective and weight errors: [2.6645352591003757e-15, 1.894711961311747e-15]
Literal inactive-disk versus zero-on-omitted-support objective gap: 9.999999994736442e-09

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=33b107d831d3fd04 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_33b107d8 (directory).
```

### Turn 7 — `submit_completed_task` (`14819 ms` LLM)
- **Rationale**: Isolated-process verification is bound to the saved artifact. Independent PDE and exhaustive ridge references passed both tolerances, and closed-form checks covered zero, sparse, and fully connected pair penalties.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/nonhermitian_advection_coupled_select.py': 'FILE:4158'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/nonhermitian_advection_coupled_select.py`. Across five synthetic PDE stress cases, the worst absolute objective error was 2.66e-15 and the worst relative field error was 2.37e-15. Independent checks also confirmed that pair penalties can change the selected support.

Limitations: Public fixtures were unavailable. The solver follows the prescribed ridge-per-support rule; it does not separately optimize nonzero coefficients at or below the 1e-8 activity threshold.`

### Turn 8 — `execute_in_remote_docker` (`34032 ms` LLM)
- **Rationale**: Previous PDE slices used positive damping and did not test near cancellation of diffusion by reaction. This isolated-process check compares both frequency signs against the independent reference in that regime. A failed stencil residual would point to assembly or solve accuracy; a failed objective or field comparison with a passing residual would point to support selection.
- **Alternatives Weighed**: `Run batch-confounding and relay-mode diagnostics` (rejected: The PDE contract has no batches or relays, and the public fixtures were unavailable on both prior checks.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
from pathlib import Path
import numpy as np

# Load the independent reference helpers without rerunning the settled slices.
helpers = Path('/tmp/verify_eq04.py').read_text().split('rng = np.random.default_rng(274)', 1)[0]
exec(compile(helpers, '/tmp/verify_eq04.py', 'exec'))
rng = np.random.default_rng(712)
n = 32
shape = (n, n)
Lx, Ly, nu = 1.3, .9, .018
hx, hy = Lx/(n+1), Ly/(n+1)
# Tune reaction to cancel the lowest diffusion eigenvalue in the zero-flow limit.
shift = -4*nu*np.sin(np.pi/(2*(n+1)))**2*(hx**-2+hy**-2)
flat = np.r_[0, n-1, n*(n-1), n*n-1, rng.choice(np.arange(1, n*(n-1)), 42, replace=False)]
x = np.arange(1,n+1)/(n+1)
basis = rng.normal(size=(11,n,n))
basis[0] = np.outer(np.sin(np.pi*x), np.sin(np.pi*x))
worst = np.zeros(3)
for omega in (1e-4, -1e-4):
    p = dict(Lx=Lx, Ly=Ly, nu=nu, omega=omega,
             vx=rng.uniform(-1e-4,1e-4,shape), vy=rng.uniform(-1e-4,1e-4,shape),
             sigma=np.full(shape,shift-2e-5), basis=basis,
             sensor_ij=np.column_stack(np.unravel_index(flat,shape)), lam2=1e-4)
    a,b,field = reference_field(p)
    fundamental = basis[0].ravel()
    highest = (basis[0]*(-1)**np.indices(shape).sum(axis=0)).ravel()
    assert np.vdot(fundamental,a@fundamental).real < 0 < np.vdot(highest,a@highest).real
    g = field[flat]
    y = g[:,0]*(2e-4+3e-4j) + g[:,4]*(1e-3-3e-3j) + g[:,7]*(-2e-3+1e-2j)
    p.update(y_obs_re=y.real,y_obs_im=y.imag,lam0=.002,pair_penalties=[(0,4,.01),(0,7,.015)])
    w,j = reference_select(g,y,p)
    result = mod.solve_nonhermitian_coupled_inversion(p)
    expected = abs(field@w).reshape(shape)
    err = np.array([np.linalg.norm(a@mod._responses(p,basis)-b)/np.linalg.norm(b),
                    abs(result['objective']-j),
                    np.linalg.norm(result['predicted_field_abs']-expected)/np.linalg.norm(expected)])
    assert np.all(err < [1e-8,1e-6,1e-5]), (omega,err)
    worst = np.maximum(worst,err)
    print('omega', omega, 'objective', j, 'selected support', np.flatnonzero(abs(w)>1e-8).tolist())
print('Indefinite near-resonant slices: worst stencil residual, absolute objective error, relative field error:', worst.tolist())
PY
```
- **Container Exit Code**: `0` (`1140 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
omega 0.0001 objective 0.0020184569249015846 selected support [0]
omega -0.0001 objective 0.0020184575784714678 selected support [0]
Indefinite near-resonant slices: worst stencil residual, absolute objective error, relative field error: [1.9893758644586085e-11, 4.336808689942018e-19, 3.4994710911797775e-16]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=33b107d831d3fd04 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_33b107d8 (directory).
```

### Turn 9 — `submit_completed_task` (`14745 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process verification against independently assembled PDE and exhaustive ridge references. The added near-resonant cases passed for both frequency signs without worsening the objective or field error.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/nonhermitian_advection_coupled_select.py`. Across seven synthetic PDE stress cases, including two near-resonant cases, the worst absolute objective error was 2.66e-15 and the worst relative field error was 2.37e-15. A separate check confirmed that pair penalties can change the optimal support.

Limitations: Public fixtures were unavailable. The solver follows the prescribed ridge-per-support rule and does not separately optimize nonzero coefficients at or below the 1e-8 activity threshold.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/nonhermitian_advection_coupled_select.py": "FILE:4158"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "all_valid": true,
    "max_weights_rel_l2": 0.0,
    "max_objective_abs_err": 0.0,
    "max_field_rel_l2": 0.0,
    "student_elapsed_sec": 0.339,
    "instance_diagnostics": [
      {
        "seed": 99431,
        "w_rel_l2": 0.0,
        "obj_abs_err": 0.0,
        "field_rel_l2": 0.0
      },
      {
        "seed": 99432,
        "w_rel_l2": 0.0,
        "obj_abs_err": 0.0,
        "field_rel_l2": 0.0
      },
      {
        "seed": 99433,
        "w_rel_l2": 0.0,
        "obj_abs_err": 0.0,
        "field_rel_l2": 0.0
      }
    ]
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_weights_rel_l2\": 0.0, \"max_objective_abs_err\": 0.0, \"max_field_rel_l2\": 0.0, \"student_elapsed_sec\": 0.339, \"instance_diagnostics\": [{\"seed\": 99431, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}, {\"seed\": 99432, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}, {\"seed\": 99433, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}]}\n__HEGEL_VERIFIER_AUTH_31d7f809ea76b06ce7be7d1b39d964c2__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_weights_rel_l2\": 0.0, \"max_objective_abs_err\": 0.0, \"max_field_rel_l2\": 0.0, \"student_elapsed_sec\": 0.339, \"instance_diagnostics\": [{\"seed\": 99431, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}, {\"seed\": 99432, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}, {\"seed\": 99433, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 6507.5,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
