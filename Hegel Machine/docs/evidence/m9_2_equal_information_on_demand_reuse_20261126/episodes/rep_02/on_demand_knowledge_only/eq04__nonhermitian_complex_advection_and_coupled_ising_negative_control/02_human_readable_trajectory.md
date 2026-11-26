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
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `305,979` (`294,066` / `80,896` / `11,913`) |
| **Total Episode Wall Time (s)** | `403.02s` |

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

### Turn 1 — `execute_in_remote_docker` (`17286 ms` LLM)
- **Rationale**: Read only the contract and compact fixture metadata. Use the observed schema to assemble the operator, enumerate all supports, and design independent checks for boundary signs and pairwise penalties.
- **Alternatives Weighed**: `Start from a guessed input schema` (rejected: Inspecting the validator and public cases first can catch incorrect domain-length keys, fixture layout, or return conventions before implementation.)

```bash
ls -lh /holdout_task; sed -n '1,220p' /holdout_task/validate_public_schema.py; python3 -I - <<'PY'
import pickle, numpy as np, scipy
with open('/holdout_task/public_eq04_cases.pkl','rb') as f:
    data=pickle.load(f)
print('versions',np.__version__,scipy.__version__)
def structure(x,d=0):
    if d>=4: return type(x).__name__
    if isinstance(x,dict): return {k:structure(v,d+1) for k,v in list(x.items())[:28]}
    if isinstance(x,(tuple,list)): return (type(x).__name__,len(x),[structure(v,d+1) for v in x[:2]])
    if isinstance(x,np.ndarray): return (x.shape,str(x.dtype))
    return x if isinstance(x,(int,float,str)) else type(x).__name__
print(structure(data))
PY
```
- **Container Exit Code**: `1` (`736 ms`, timed_out=`False`)
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task': No such file or directory
sed: can't read /holdout_task/validate_public_schema.py: No such file or directory
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
FileNotFoundError: [Errno 2] No such file or directory: '/holdout_task/public_eq04_cases.pkl'
```

### Turn 2 — `execute_in_remote_docker` (`19145 ms` LLM)
- **Rationale**: Check metadata only for an alternate fixture or local convention, and confirm NumPy and SciPy are available before writing the solver. If no fixture is mounted, verify against independently assembled stencil and exhaustive subset references.
- **Alternatives Weighed**: `Retry the documented fixture path without inspecting other mounts` (rejected: The prior command established that the directory was absent; a bounded inventory can locate alternate fixtures or establish that independent tests are needed.)

```bash
pwd; ls -ld /holdout_task /pretrain_task /app /workspace /tmp/hegel_knowledge_ref 2>/dev/null || true; for d in /pretrain_task /app /workspace; do if [ -d "$d" ]; then find "$d" -maxdepth 3 -type f \( -name '*.py' -o -name '*.pkl' -o -name 'README*' -o -name 'AGENTS.md' \) -printf '%p (%s bytes)\n'; fi; done | head -100; if [ -f /tmp/hegel_knowledge_ref/describe_ref.py ]; then python3 /tmp/hegel_knowledge_ref/describe_ref.py --search 'exact subset support enumeration complex ridge pairwise penalties'; fi; python3 -I -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)'
```
- **Container Exit Code**: `0` (`667 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
et (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
1.26.4 1.13.0
```

### Turn 3 — `execute_in_remote_docker` (`48133 ms` LLM)
- **Rationale**: Implement the documented stencil and exact support enumeration. Next, compare the final artifact against an independent stencil and subset oracle, including a case where the pair penalties change the selected support.
- **Alternatives Weighed**: `Boxed real L0 certification from the reference catalog` (rejected: The retrieved method assumes real boxed coefficients; all 2,048 complex supports can be evaluated directly.); `Score objectives from the Gram quadratic alone` (rejected: Explicit complex residuals avoid subtracting large nearly equal terms when the fit is tight.)

```bash
mkdir -p /holdout_task
cat > /holdout_task/nonhermitian_advection_coupled_select.py <<'PY'
"""Sparse complex PDE inversion with exhaustive pair-penalized ridge selection."""

from itertools import combinations

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


def _responses(problem):
    basis = np.asarray(problem['basis'], dtype=np.complex128)
    m, nx, ny = basis.shape
    hx = problem['Lx'] / (nx + 1)
    hy = problem['Ly'] / (ny + 1)
    dx = problem['nu'] / hx**2
    dy = problem['nu'] / hy**2
    ux = np.asarray(problem['vx'], dtype=float) / hx
    uy = np.asarray(problem['vy'], dtype=float) / hy
    diag = (2 * (dx + dy) + np.abs(ux) + np.abs(uy)
            + np.asarray(problem['sigma'], dtype=float) + 1j * problem['omega'])
    ids = np.arange(nx * ny).reshape(nx, ny)
    rows = np.concatenate([ids.ravel(), ids[1:, :].ravel(), ids[:-1, :].ravel(),
                           ids[:, 1:].ravel(), ids[:, :-1].ravel()])
    cols = np.concatenate([ids.ravel(), ids[:-1, :].ravel(), ids[1:, :].ravel(),
                           ids[:, :-1].ravel(), ids[:, 1:].ravel()])
    values = np.concatenate([diag.ravel(),
                             (-dx - np.maximum(ux[1:, :], 0)).ravel(),
                             (-dx + np.minimum(ux[:-1, :], 0)).ravel(),
                             (-dy - np.maximum(uy[:, 1:], 0)).ravel(),
                             (-dy + np.minimum(uy[:, :-1], 0)).ravel()])
    operator = csc_matrix((values, (rows, cols)), shape=(nx * ny, nx * ny))
    return splu(operator).solve(basis.reshape(m, -1).T).reshape(nx, ny, m)


def solve_nonhermitian_coupled_inversion(problem: dict) -> dict:
    fields = _responses(problem)
    ij = np.asarray(problem['sensor_ij'], dtype=np.intp)
    g = fields[ij[:, 0], ij[:, 1], :]
    y = (np.asarray(problem['y_obs_re'], dtype=float)
         + 1j * np.asarray(problem['y_obs_im'], dtype=float))
    lam0, lam2 = problem['lam0'], problem['lam2']
    pairs = problem['pair_penalties']
    m = g.shape[1]
    gram = g.conj().T @ g + lam2 * np.eye(m)
    rhs = g.conj().T @ y
    best = 0.5 * float(np.vdot(y, y).real)
    weights = np.zeros(m, dtype=np.complex128)

    # Group equal-sized supports for batched solves; charge each listed pair
    # only when both of its coefficients meet the activation criterion.
    for size in range(1, m + 1):
        indices = np.asarray(list(combinations(range(m), size)), dtype=np.intp)
        restricted = gram[indices[:, :, None], indices[:, None, :]]
        coeffs = np.linalg.solve(restricted, rhs[indices, None])[..., 0]
        candidates = np.zeros((len(indices), m), dtype=np.complex128)
        candidates[np.arange(len(indices))[:, None], indices] = coeffs
        active = np.abs(candidates) > 1e-8
        residual = np.einsum('km,sm->sk', g, candidates) - y
        scores = (0.5 * np.sum(np.abs(residual)**2, axis=1)
                  + 0.5 * lam2 * np.sum(np.abs(coeffs)**2, axis=1)
                  + lam0 * active.sum(axis=1))
        for u, v, gamma in pairs:
            scores += gamma * (active[:, u] & active[:, v])
        i = int(np.argmin(scores))
        if scores[i] < best:
            best = float(scores[i])
            weights = candidates[i]

    return {'weights_real': weights.real, 'weights_imag': weights.imag,
            'objective': best, 'predicted_field_abs': np.abs(fields @ weights)}
PY
```
- **Container Exit Code**: `0` (`340 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/nonhermitian_advection_coupled_select.py']`
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7cca307b0226c84c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7cca307b (directory).
```

### Turn 4 — `execute_in_remote_docker` (`55464 ms` LLM)
- **Rationale**: The independent stencil residual detects sign and boundary-index errors; the augmented least-squares oracle detects support, conjugation, and scoring errors. A forced support change tests the pair penalty directly. Any tolerance failure identifies which layer to revise.
- **Alternatives Weighed**: `Use a forward solve from the candidate as the only test reference` (rejected: It would reproduce stencil and indexing errors instead of detecting them.)

```bash
set -e
cat > /tmp/verify_eq04.py <<'PY'
import importlib.util
import itertools
import pathlib
import time

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

path = '/holdout_task/nonhermitian_advection_coupled_select.py'
spec = importlib.util.spec_from_file_location('candidate', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def forward(p):
    m, nx, ny = p['basis'].shape
    hx, hy = p['Lx']/(nx+1), p['Ly']/(ny+1)
    dx, dy = p['nu']/hx**2, p['nu']/hy**2
    a = lil_matrix((nx*ny, nx*ny), dtype=np.complex128)
    for i, j in itertools.product(range(nx), range(ny)):
        k = i*ny+j
        vx, vy = p['vx'][i,j], p['vy'][i,j]
        a[k,k] = 2*dx + 2*dy + abs(vx)/hx + abs(vy)/hy + p['sigma'][i,j] + 1j*p['omega']
        for ii, jj, h, diff, vel in ((i-1,j,hx,dx,vx), (i+1,j,hx,dx,-vx), (i,j-1,hy,dy,vy), (i,j+1,hy,dy,-vy)):
            if 0 <= ii < nx and 0 <= jj < ny:
                a[k,ii*ny+jj] = -diff-max(vel,0)/h
    a = a.tocsc()
    b = p['basis'].reshape(m,-1).T
    f = spsolve(a,b).reshape(nx,ny,m)
    f_impl = mod._responses(p)
    op_err = np.linalg.norm(a @ f_impl.reshape(-1,m) - b)/np.linalg.norm(b)
    return f, op_err

def oracle(g, y, p):
    m = g.shape[1]
    h = np.vstack((g, np.sqrt(p['lam2'])*np.eye(m)))
    target = np.r_[y, np.zeros(m)]
    best, win = np.inf, None
    for k in range(m+1):
        for support in itertools.combinations(range(m), k):
            w = np.zeros(m, dtype=complex)
            if k:
                w[list(support)] = np.linalg.lstsq(h[:,support], target, rcond=None)[0]
            score = objective(g,y,p,w)
            if score < best:
                best, win = score, w
    return best, win

def objective(g,y,p,w):
    active = np.abs(w)>1e-8
    err = g@w-y
    return (0.5*np.vdot(err,err).real + 0.5*p['lam2']*np.vdot(w,w).real
            + p['lam0']*np.count_nonzero(active)
            + sum(c for u,v,c in p['pair_penalties'] if active[u] and active[v]))

rng = np.random.default_rng(4102)
worst = np.zeros(3)
counts = {}
for label in ('mixed-flow','pair-switch','negative-frequency-correlated','empty'):
    x,y = np.indices((32,32))/31
    basis = rng.normal(size=(11,32,32))
    basis[:4] += np.array([1, x, y, x*y])
    if label == 'negative-frequency-correlated':
        basis[9] = basis[2]+1e-4*basis[9]
    interior = rng.choice(900, size=45, replace=False)
    sensors = np.vstack(([[0,0],[0,31],[31,0],[31,31],[0,15],[16,31]], np.c_[1+interior//30, 1+interior%30]))
    p = dict(Lx=1.7, Ly=0.63, nu=0.0004 if label=='negative-frequency-correlated' else 0.005,
             vx=1.1*np.sin(9*y)+0.4*(x-0.5), vy=0.7*np.cos(7*x)-0.3*y,
             sigma=rng.uniform(-0.6,1.2,(32,32)), omega=-8.0 if label=='negative-frequency-correlated' else 2.3,
             basis=basis, sensor_ij=sensors, lam2=1e-8 if label=='negative-frequency-correlated' else 0.004,
             lam0=0.0003, pair_penalties=[[0,3,0.025],[3,7,0.04],[5,8,0.03]])
    p['vx'][0,:] = 0.0
    p['vy'][:,31] = 0.0
    f, op_err = forward(p)
    g = f[sensors[:,0],sensors[:,1],:]
    truth = np.zeros(11,complex)
    if label != 'empty':
        truth[[0,3,7]] = [1+0.4j,-0.9+0.7j,0.8-0.4j]
    obs = g@truth
    if label == 'pair-switch':
        # Each co-activation alone costs more than the empty model.
        p['pair_penalties'] = [[i,j,1+np.vdot(obs,obs).real] for i,j in itertools.combinations(range(11),2)]
        uncoupled = dict(p, pair_penalties=[])
        _, w0 = oracle(g,obs,uncoupled)
        assert np.count_nonzero(np.abs(w0)>1e-8) > 1
    p.update(y_obs_re=obs.real, y_obs_im=obs.imag)
    ref_score, ref_w = oracle(g,obs,p)
    start = time.perf_counter()
    out = mod.solve_nonhermitian_coupled_inversion(p)
    elapsed = time.perf_counter()-start
    assert out.keys() == {'weights_real','weights_imag','objective','predicted_field_abs'}
    for key, shape in [('weights_real',(11,)),('weights_imag',(11,)),('predicted_field_abs',(32,32))]:
        assert out[key].shape == shape and out[key].dtype == np.float64 and np.all(np.isfinite(out[key]))
    assert isinstance(out['objective'],float)
    w = out['weights_real'] + 1j*out['weights_imag']
    a = np.abs(f@ref_w)
    field_err = np.linalg.norm(out['predicted_field_abs']-a)/(np.linalg.norm(a) or 1.0)
    obj_err = max(abs(out['objective']-ref_score),abs(out['objective']-objective(g,obs,p,w)))
    counts[label] = int(np.count_nonzero(np.abs(w)>1e-8))
    assert op_err < 1e-11 and field_err <= 1e-5 and obj_err <= 1e-6, (label,op_err,field_err,obj_err)
    if label == 'pair-switch':
        assert counts[label] == 1
    if label == 'empty':
        assert counts[label] == 0 and np.all(out['predicted_field_abs'] == 0)
    worst = np.maximum(worst,[op_err,obj_err,field_err])
    print(f'{label}: operator residual {op_err:.3e}, objective error {obj_err:.3e}, field error {field_err:.3e}, active {counts[label]}, solve {elapsed:.3f}s')
print('Worst [operator residual, objective absolute error, field relative L2]:', worst)
print('Public fixture available:', pathlib.Path('/holdout_task/public_eq04_cases.pkl').is_file())
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_eq04.py
if [ -f /holdout_task/validate_public_schema.py ] && [ -f /holdout_task/public_eq04_cases.pkl ]; then python3 /holdout_task/validate_public_schema.py; fi
```
- **Container Exit Code**: `1` (`529 ms`, timed_out=`False`)
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/verify_eq04.py", line 62, in <module>
    basis[:4] += np.array([1, x, y, x*y])
                 ^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (4,) + inhomogeneous part.
```

### Turn 5 — `execute_in_remote_docker` (`15412 ms` LLM)
- **Rationale**: Replace the scalar basis offset with a grid of ones and rerun the isolated verification without changing the artifact. A stencil residual failure would call for revisiting assembly; an objective or field mismatch would call for inspecting support selection or reconstruction.
- **Alternatives Weighed**: `Modify the solver in response to the failure` (rejected: The failure came from stacking a scalar with grid arrays in the test fixture; no solver result was produced.)

```bash
set -e
sed -i 's/np.array(\[1, x, y, x\*y\])/np.stack([np.ones_like(x), x, y, x*y])/' /tmp/verify_eq04.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_eq04.py
if [ -f /holdout_task/validate_public_schema.py ] && [ -f /holdout_task/public_eq04_cases.pkl ]; then python3 /holdout_task/validate_public_schema.py; fi
```
- **Container Exit Code**: `0` (`929 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
mixed-flow: operator residual 2.028e-15, objective error 2.082e-17, field error 1.648e-15, active 7, solve 0.008s
pair-switch: operator residual 1.997e-15, objective error 2.776e-17, field error 3.075e-16, active 1, solve 0.009s
negative-frequency-correlated: operator residual 7.555e-16, objective error 0.000e+00, field error 1.190e-15, active 8, solve 0.008s
empty: operator residual 1.931e-15, objective error 0.000e+00, field error 0.000e+00, active 0, solve 0.008s
Worst [operator residual, objective absolute error, field relative L2]: [2.02783025e-15 2.77555756e-17 1.64771312e-15]
Public fixture available: False

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7cca307b0226c84c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7cca307b (directory).
```

### Turn 6 — `submit_completed_task` (`13482 ms` LLM)
- **Rationale**: The saved artifact passed isolated verification against an independent stencil and exhaustive augmented least-squares oracle, including a case where pair penalties changed the selected support.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/nonhermitian_advection_coupled_select.py': 'FILE:3308'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/nonhermitian_advection_coupled_select.py`. Across four independent stress cases, the worst absolute objective error was 2.78e-17 and the worst relative field error was 1.65e-15, both below tolerance. The public fixtures were not mounted.`

### Turn 7 — `execute_in_remote_docker` (`27108 ms` LLM)
- **Rationale**: An allowed duplicate basis and small positive ridge can make a floating-point Gram block singular, even though the exact ridge system is nonsingular. If that prevents the solver from returning the known singleton optimum, switch to augmented least squares and rerun both this case and the existing stress suite.
- **Alternatives Weighed**: `Audit batch confounding, drift operators, or relay modes` (rejected: The input schema defines a deterministic PDE and support objective, with no cohorts, drift model, or relay state. The public fixtures were not mounted.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import numpy as np
path = '/holdout_task/nonhermitian_advection_coupled_select.py'
spec = importlib.util.spec_from_file_location('candidate', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rng = np.random.default_rng(148)
nx = ny = 32
basis = rng.normal(size=(11,nx,ny))
basis[1] = basis[0]
obs_idx = rng.choice(nx*ny, 40, replace=False)
ij = np.column_stack(divmod(obs_idx, ny))
p = dict(basis=basis, Lx=1., Ly=.7, nu=.003, vx=rng.uniform(-.2,.2,(nx,ny)), vy=rng.uniform(-.4,.4,(nx,ny)), sigma=np.zeros((nx,ny)), omega=-1.5, sensor_ij=ij, lam2=1e-20, lam0=.001, pair_penalties=[[0,1,.01]])
f = mod._responses(p)
y = f[ij[:,0],ij[:,1],0]*(1+.5j)
p.update(y_obs_re=y.real, y_obs_im=y.imag)
out = mod.solve_nonhermitian_coupled_inversion(p)
ref_obj = p['lam0'] + .5*p['lam2']*1.25
field = np.abs(f[...,0]*(1+.5j))
obj_err = abs(out['objective'] - ref_obj)
field_err = np.linalg.norm(out['predicted_field_abs']-field)/np.linalg.norm(field)
print(f'duplicate basis, tiny positive ridge: objective error={obj_err:.3e}, field relative error={field_err:.3e}')
assert obj_err <= 1e-6 and field_err <= 1e-5
PY
```
- **Container Exit Code**: `1` (`717 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7cca307b0226c84c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7cca307b (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 17, in <module>
  File "/holdout_task/nonhermitian_advection_coupled_select.py", line 54, in solve_nonhermitian_coupled_inversion
    coeffs = np.linalg.solve(restricted, rhs[indices, None])[..., 0]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/linalg/linalg.py", line 409, in solve
    r = gufunc(a, b, signature=signature, extobj=extobj)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/linalg/linalg.py", line 112, in _raise_linalgerror_singular
    raise LinAlgError("Singular matrix")
numpy.linalg.LinAlgError: Singular matrix
```

### Turn 8 — `execute_in_remote_docker` (`38455 ms` LLM)
- **Rationale**: The failure is numerical, not a defect in the PDE or penalty model. Keep the ridge explicit in the least-squares system and verify the known singleton optimum, then rerun the original cases to catch any regression in objective or field reconstruction.
- **Alternatives Weighed**: `Fall back to augmented least squares only when the Gram solve raises` (rejected: Ill-conditioned Gram blocks can also return inaccurate coefficients without raising, which can change the selected support.)

```bash
set -e
python3 -I - <<'PY'
from pathlib import Path
p = Path('/holdout_task/nonhermitian_advection_coupled_select.py')
s = p.read_text()
s = s.replace('    gram = g.conj().T @ g + lam2 * np.eye(m)\n    rhs = g.conj().T @ y\n', '    # Keep the ridge rows explicit: forming G^H G can round away a\n    # small positive ridge on linearly dependent sensor columns.\n    design = np.vstack((g, np.sqrt(lam2) * np.eye(m)))\n    target = np.concatenate((y, np.zeros(m, dtype=np.complex128)))\n')
s = s.replace('    # Group equal-sized supports for batched solves; charge each listed pair\n    # only when both of its coefficients meet the activation criterion.\n', '    # Solve each support as augmented least squares, then score the\n    # original objective, charging pairs only when both are active.\n')
s = s.replace('        restricted = gram[indices[:, :, None], indices[:, None, :]]\n        coeffs = np.linalg.solve(restricted, rhs[indices, None])[..., 0]\n', '        coeffs = np.array([np.linalg.lstsq(design[:, support], target, rcond=None)[0]\n                           for support in indices])\n')
assert 'np.linalg.lstsq' in s and 'gram' not in s
p.write_text(s)
PY
cat >> /tmp/verify_eq04.py <<'PY'

# Regression: an allowed duplicate basis and tiny ridge made the former
# normal equations singular. A singleton attains the known optimum to tolerance.
rng = np.random.default_rng(148)
nx = ny = 32
basis = rng.normal(size=(11, nx, ny))
basis[1] = basis[0]
ij = np.column_stack(divmod(rng.choice(nx*ny, 40, replace=False), ny))
p = dict(basis=basis, Lx=1., Ly=.7, nu=.003,
         vx=rng.uniform(-.2,.2,(nx,ny)), vy=rng.uniform(-.4,.4,(nx,ny)),
         sigma=np.zeros((nx,ny)), omega=-1.5, sensor_ij=ij,
         lam2=1e-20, lam0=.001, pair_penalties=[[0,1,.01]])
f, op_err = forward(p)
g = f[ij[:,0],ij[:,1]]
obs = g[:,0]*(1+.5j)
p.update(y_obs_re=obs.real, y_obs_im=obs.imag)
out = mod.solve_nonhermitian_coupled_inversion(p)
w = out['weights_real'] + 1j*out['weights_imag']
field = np.abs(f[...,0]*(1+.5j))
obj_err = max(abs(out['objective'] - (p['lam0'] + .5*p['lam2']*1.25)),
              abs(out['objective'] - objective(g,obs,p,w)))
field_err = np.linalg.norm(out['predicted_field_abs']-field)/np.linalg.norm(field)
assert np.count_nonzero(np.abs(w)>1e-8) == 1
assert op_err < 1e-11 and obj_err <= 1e-6 and field_err <= 1e-5
print(f'duplicate basis, tiny ridge: operator residual {op_err:.3e}, objective error {obj_err:.3e}, field error {field_err:.3e}')
print('Combined worst [operator residual, objective absolute error, field relative L2]:', np.maximum(worst,[op_err,obj_err,field_err]))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_eq04.py
if [ -f /holdout_task/validate_public_schema.py ] && [ -f /holdout_task/public_eq04_cases.pkl ]; then python3 /holdout_task/validate_public_schema.py; fi
```
- **Container Exit Code**: `0` (`1459 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/nonhermitian_advection_coupled_select.py']`
- **Stdout (tail)**:
```text
mixed-flow: operator residual 2.028e-15, objective error 1.388e-17, field error 0.000e+00, active 7, solve 0.055s
pair-switch: operator residual 1.997e-15, objective error 0.000e+00, field error 0.000e+00, active 1, solve 0.060s
negative-frequency-correlated: operator residual 7.555e-16, objective error 1.388e-17, field error 0.000e+00, active 8, solve 0.057s
empty: operator residual 1.931e-15, objective error 0.000e+00, field error 0.000e+00, active 0, solve 0.067s
Worst [operator residual, objective absolute error, field relative L2]: [2.02783025e-15 1.38777878e-17 0.00000000e+00]
Public fixture available: False
duplicate basis, tiny ridge: operator residual 7.264e-16, objective error 0.000e+00, field error 4.055e-16
Combined worst [operator residual, objective absolute error, field relative L2]: [2.02783025e-15 1.38777878e-17 4.05529132e-16]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0cf1efab434b83dc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_0cf1efab (directory).
```

### Turn 9 — `submit_completed_task` (`14958 ms` LLM)
- **Rationale**: The saved artifact passed isolated Python verification against an independently assembled stencil and exhaustive reference, including a pair-penalty support switch and the duplicate-basis case that caused the Gram solve to fail.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/nonhermitian_advection_coupled_select.py': 'FILE:3476'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/nonhermitian_advection_coupled_select.py`. Across five stress cases, the worst absolute objective error was 1.39e-17 and the worst relative field L2 error was 4.06e-16, both below tolerance. The public fixtures were not mounted.`

### Turn 10 — `execute_in_remote_docker` (`41581 ms` LLM)
- **Rationale**: An analytic sine-mode reference checks the upwind signs and complex response near resonance without sharing the sparse LU implementation. An independent QR support oracle checks that the ridge and pair penalties still select the right solution when there are fewer sensors than coefficients. A failed margin identifies whether to revisit the forward solve or support selection.
- **Alternatives Weighed**: `Audit relay modes or batch confounding` (rejected: The contract has no relay state or batch variable, and the public measurements were not mounted at the last verification.)

```bash
set -e
cat > /tmp/audit_eq04_resonance.py <<'PY'
import importlib.util
import numpy as np
from scipy.linalg import lstsq

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/nonhermitian_advection_coupled_select.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rng = np.random.default_rng(2504)
n, m = 32, 11
lx, ly, nu = 1.2, .73, .002
hx, hy = lx/33, ly/33
vx, vy, omega = .11, -.13, -.003
dx, dy, ux, uy = nu/hx**2, nu/hy**2, vx/hx, -vy/hy
j = np.arange(1, n+1)
# A separable upwind operator is diagonally similar to two symmetric
# tridiagonals. Build the reference from their analytic sine modes.
phi = np.sqrt(2/33)*np.sin(np.pi*j[:,None]*j[None,:]/33)
sx = (np.sqrt((dx+ux)/dx))**(j-1)
sy = (np.sqrt(dy/(dy+uy)))**(j-1)
ex = 2*dx+ux-2*np.sqrt(dx*(dx+ux))*np.cos(np.pi*j/33)
ey = 2*dy+uy-2*np.sqrt(dy*(dy+uy))*np.cos(np.pi*j/33)
sigma = -ex[0]-ey[0]-.0005
scale = sx[:,None]*sy[None,:]
basis = rng.normal(size=(m,n,n))
fields = np.stack([scale*(phi@((phi.T@(b/scale)@phi)/(ex[:,None]+ey[None,:]+sigma+1j*omega))@phi.T) for b in basis], axis=-1)
worst = np.zeros(3)
for k in (5, 57):
    ij = np.column_stack(divmod(rng.choice(n*n,k,replace=False),n))
    g = fields[ij[:,0],ij[:,1]]
    truth = np.zeros(m, complex)
    truth[[1,4,8]] = [1.2+.9j,-.8+.5j,1.1-.6j]
    y = g@truth + .002*(rng.normal(size=k)+1j*rng.normal(size=k))
    lam0, lam2 = .0002, .00005
    pairs = [[1,4,.015],[4,8,.009],[0,6,.006]]
    p = dict(Lx=lx,Ly=ly,nu=nu,vx=np.full((n,n),vx),vy=np.full((n,n),vy),sigma=np.full((n,n),sigma),omega=omega,basis=basis,sensor_ij=ij,y_obs_re=y.real,y_obs_im=y.imag,lam0=lam0,lam2=lam2,pair_penalties=pairs)
    def score(w):
        a = np.abs(w)>1e-8
        return .5*np.linalg.norm(g@w-y)**2 + .5*lam2*np.linalg.norm(w)**2 + lam0*a.sum() + sum(c for u,v,c in pairs if a[u] and a[v])
    best, wref = score(np.zeros(m)), np.zeros(m,complex)
    for mask in range(1,1<<m):
        s = [i for i in range(m) if mask>>i&1]
        w = np.zeros(m,complex)
        w[s] = lstsq(np.vstack((g[:,s],np.sqrt(lam2)*np.eye(len(s)))),np.r_[y,np.zeros(len(s))],lapack_driver='gelsy')[0]
        val = score(w)
        if val < best:
            best, wref = val, w
    out = mod.solve_nonhermitian_coupled_inversion(p)
    w = out['weights_real']+1j*out['weights_imag']
    ref_abs = np.abs(fields@wref)
    errs = np.array([np.linalg.norm(mod._responses(p)-fields)/np.linalg.norm(fields),max(abs(out['objective']-best),abs(out['objective']-score(w))),np.linalg.norm(out['predicted_field_abs']-ref_abs)/np.linalg.norm(ref_abs)])
    print(f'Near-resonant operator, {k} sensors: forward relative L2={errs[0]:.3e}, objective absolute error={errs[1]:.3e}, field relative L2={errs[2]:.3e}')
    assert np.all(errs <= [1e-8,1e-6,1e-5]), errs
    worst = np.maximum(worst, errs)
print('Audit worst:', worst)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_eq04_resonance.py
if [ -f /holdout_task/validate_public_schema.py ] && [ -f /holdout_task/public_eq04_cases.pkl ]; then python3 /holdout_task/validate_public_schema.py; else echo 'Public fixtures or validator not mounted'; fi
```
- **Container Exit Code**: `1` (`805 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Near-resonant operator, 5 sensors: forward relative L2=3.646e-11, objective absolute error=1.017e-05, field relative L2=4.686e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0cf1efab434b83dc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_0cf1efab (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/audit_eq04_resonance.py", line 52, in <module>
    assert np.all(errs <= [1e-8,1e-6,1e-5]), errs
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: [3.64580173e-11 1.01696675e-05 4.68576896e-14]
```

### Turn 11 — `execute_in_remote_docker` (`25316 ms` LLM)
- **Rationale**: Split the objective discrepancy into its components and compare both forward maps against the same stencil. If the returned score disagrees on the solver's own map, inspect scoring; if the reference fails the stencil residual, improve the reference before deciding whether to change the solver.
- **Alternatives Weighed**: `Change the solver immediately` (rejected: The reported metric combines the optimum gap and re-scoring on a different forward map; it does not establish which quantity exceeded tolerance.)

```bash
set -e
python3 -I - <<'PY'
from pathlib import Path
p = Path('/tmp/audit_eq04_resonance.py')
s = p.read_text()
s = s.replace('    assert np.all(errs <= [1e-8,1e-6,1e-5]), errs', '''    cand_fields = mod._responses(p)
    gc = cand_fields[ij[:,0],ij[:,1]]
    def candidate_score(z):
        a = np.abs(z)>1e-8
        return .5*np.linalg.norm(gc@z-y)**2 + .5*lam2*np.linalg.norm(z)**2 + lam0*a.sum() + sum(c for u,v,c in pairs if a[u] and a[v])
    print('  returned, analytic optimum, analytic at candidate, candidate at candidate:',out['objective'],best,score(w),candidate_score(w))
    print('  supports:',np.flatnonzero(np.abs(w)>1e-8), np.flatnonzero(np.abs(wref)>1e-8), 'sensor response discrepancy abs:',np.linalg.norm(gc-g), 'target norm:',np.linalg.norm(y))
    b = basis.reshape(m,-1).T
    for name, ff in [('analytic',fields),('sparse LU',cand_fields)]:
        u = ff.reshape(n,n,m)
        r = (2*(dx+dy)+ux+uy+sigma+1j*omega)*u.copy()
        r[1:] -= (dx+ux)*u[:-1]
        r[:-1] -= dx*u[1:]
        r[:,1:] -= dy*u[:,:-1]
        r[:,:-1] -= (dy+uy)*u[:,1:]
        print(' ',name,'stencil residual relative L2:',np.linalg.norm(r.reshape(-1,m)-b)/np.linalg.norm(b))
    assert errs[2] <= 1e-5 and abs(out['objective']-candidate_score(w)) <= 1e-6''')
Path('/tmp/audit_eq04_diagnose.py').write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_eq04_diagnose.py
```
- **Container Exit Code**: `0` (`947 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Near-resonant operator, 5 sensors: forward relative L2=3.646e-11, objective absolute error=1.017e-05, field relative L2=4.686e-14
  returned, analytic optimum, analytic at candidate, candidate at candidate: 0.0010864892815088708 0.001086489282163253 0.0010966589490486775 0.0010864892815104857
  supports: [ 1  5  6  8 10] [ 1  5  6  8 10] sensor response discrepancy abs: 0.0076637856666601145 target norm: 123438160.06298806
  analytic stencil residual relative L2: 1.8568024882156132e-05
  sparse LU stencil residual relative L2: 6.65700771169648e-06
Near-resonant operator, 57 sensors: forward relative L2=3.646e-11, objective absolute error=1.104e+00, field relative L2=5.889e-15
  returned, analytic optimum, analytic at candidate, candidate at candidate: 0.024909792000214565 0.024909245372505526 1.129350793097998 0.024909792000214565
  supports: [1 4 8] [1 4 8] sensor response discrepancy abs: 2.5250949191512047 target norm: 40752615326.52601
  analytic stencil residual relative L2: 1.8568024882156132e-05
  sparse LU stencil residual relative L2: 6.65700771169648e-06
Audit worst: [3.64580173e-11 1.10444100e+00 4.68576896e-14]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0cf1efab434b83dc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_0cf1efab (directory).
```

### Turn 12 — `execute_in_remote_docker` (`29602 ms` LLM)
- **Rationale**: Use an independent QR oracle on the same sensor map to detect support-selection or ridge-solve errors, while retaining the analytic optimum and field comparisons to detect a forward-map tolerance failure. No artifact edit is warranted unless a contract metric fails.
- **Alternatives Weighed**: `Modify the forward solver based on the 1.10 cross-map re-scoring error` (rejected: The near-resonant response magnifies small differences between the two maps; re-scoring fixed weights on the alternate map is not a comparison of minimum objectives.)

```bash
set -e
python3 -I - <<'PY'
from pathlib import Path
s = Path('/tmp/audit_eq04_resonance.py').read_text()
start = s.index('    def score(w):')
stop = s.index("print('Audit worst:", start)
s = s[:start] + '''    out = mod.solve_nonhermitian_coupled_inversion(p)
    w = out['weights_real'] + 1j*out['weights_imag']
    cand_fields = mod._responses(p)
    gc = cand_fields[ij[:,0],ij[:,1]]
    def score(matrix, z):
        a = np.abs(z)>1e-8
        return .5*np.linalg.norm(matrix@z-y)**2 + .5*lam2*np.linalg.norm(z)**2 + lam0*a.sum() + sum(c for u,v,c in pairs if a[u] and a[v])
    def optimum(matrix):
        best, win = score(matrix, np.zeros(m)), np.zeros(m,complex)
        for mask in range(1,1<<m):
            s = [i for i in range(m) if mask>>i&1]
            z = np.zeros(m,complex)
            z[s] = lstsq(np.vstack((matrix[:,s],np.sqrt(lam2)*np.eye(len(s)))),np.r_[y,np.zeros(len(s))],lapack_driver='gelsy')[0]
            val = score(matrix,z)
            if val < best:
                best, win = val, z
        return best, win
    analytic_best, analytic_w = optimum(g)
    matched_best, _ = optimum(gc)
    ref_field = np.abs(fields@analytic_w)
    errs = np.array([abs(out['objective']-score(gc,w)), abs(out['objective']-matched_best), abs(out['objective']-analytic_best)])
    field_err = np.linalg.norm(out['predicted_field_abs']-ref_field)/np.linalg.norm(ref_field)
    print(f'{k} sensors: score consistency {errs[0]:.3e}, matched-map QR optimum gap {errs[1]:.3e}, independent analytic optimum gap {errs[2]:.3e}, field relative L2 {field_err:.3e}', flush=True)
    assert np.all(errs <= 1e-6) and field_err <= 1e-5
    worst = np.maximum(worst, errs)
''' + s[stop:]
Path('/tmp/audit_eq04_matched_map.py').write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_eq04_matched_map.py
```
- **Container Exit Code**: `0` (`1249 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
5 sensors: score consistency 1.615e-15, matched-map QR optimum gap 6.180e-16, independent analytic optimum gap 6.544e-13, field relative L2 4.686e-14
57 sensors: score consistency 0.000e+00, matched-map QR optimum gap 1.212e-09, independent analytic optimum gap 5.466e-07, field relative L2 5.889e-15
Audit worst: [1.61481072e-15 1.21226920e-09 5.46627709e-07]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=0cf1efab434b83dc (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_0cf1efab (directory).
```

### Turn 13 — `submit_completed_task` (`13128 ms` LLM)
- **Rationale**: The saved artifact passed clean-process verification. In the near-resonant audit, the independent analytic optimum met the objective tolerance, and the matched-map QR oracle confirmed support selection to within 1.22e-9.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/nonhermitian_advection_coupled_select.py`. Across seven independent stress cases, the worst absolute objective gap was 5.47e-7 and the worst relative field L2 error was 4.69e-14, both within tolerance. The public fixtures were not mounted.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/nonhermitian_advection_coupled_select.py": "FILE:3476"
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
    "student_elapsed_sec": 0.432,
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
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_weights_rel_l2\": 0.0, \"max_objective_abs_err\": 0.0, \"max_field_rel_l2\": 0.0, \"student_elapsed_sec\": 0.432, \"instance_diagnostics\": [{\"seed\": 99431, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}, {\"seed\": 99432, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}, {\"seed\": 99433, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}]}\n__HEGEL_VERIFIER_AUTH_4d410889add0631ec794b9dc25002871__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_weights_rel_l2\": 0.0, \"max_objective_abs_err\": 0.0, \"max_field_rel_l2\": 0.0, \"student_elapsed_sec\": 0.432, \"instance_diagnostics\": [{\"seed\": 99431, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}, {\"seed\": 99432, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}, {\"seed\": 99433, \"w_rel_l2\": 0.0, \"obj_abs_err\": 0.0, \"field_rel_l2\": 0.0}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 6177.85,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
