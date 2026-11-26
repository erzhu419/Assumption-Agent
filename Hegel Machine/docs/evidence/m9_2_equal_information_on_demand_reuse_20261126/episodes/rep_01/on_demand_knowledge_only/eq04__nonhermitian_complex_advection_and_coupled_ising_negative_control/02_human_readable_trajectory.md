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
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `299,196` (`283,202` / `80,896` / `15,994`) |
| **Total Episode Wall Time (s)** | `492.57s` |

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

### Turn 1 — `execute_in_remote_docker` (`16176 ms` LLM)
- **Rationale**: Inspect the validator and public case schema before coding so any shape, key, or dependency mismatch can be addressed in the implementation without transferring the dataset.
- **Alternatives Weighed**: `SPD iterative solve` (rejected: The oscillatory upwind operator is complex and non-Hermitian.); `Greedy feature selection` (rejected: Pair penalties make gains support-dependent, and the full 2,048-support search is tractable.)

```bash
ls -la /holdout_task; sed -n '1,220p' /holdout_task/validate_public_schema.py; python3 -I - <<'PY'
import scipy, numpy, pickle
print('SciPy',scipy.__version__,'NumPy',numpy.__version__)
with open('/holdout_task/public_eq04_cases.pkl','rb') as f: v=pickle.load(f)
print(type(v),len(v))
if isinstance(v,dict): print(list(v)[:15]); v=next(iter(v.values()))
if isinstance(v,(list,tuple)): print('cases',len(v)); v=v[0]
if isinstance(v,dict): print({k:getattr(x,'shape',type(x).__name__) for k,x in v.items()})
PY
```
- **Container Exit Code**: `1` (`916 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
SciPy 1.13.0 NumPy 1.26.4
```
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task': No such file or directory
sed: can't read /holdout_task/validate_public_schema.py: No such file or directory
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
FileNotFoundError: [Errno 2] No such file or directory: '/holdout_task/public_eq04_cases.pkl'
```

### Turn 2 — `execute_in_remote_docker` (`16964 ms` LLM)
- **Rationale**: Check whether the fixtures or validator were mounted elsewhere and create the required output directory. If no public cases are present, use an independent synthetic oracle and report that limitation.
- **Alternatives Weighed**: `Immediately substitute synthetic fixtures` (rejected: A bounded server-side search can first determine whether the public cases are staged under another path.)

```bash
pwd; ls -la /; for d in /pretrain_task /app /workspace /results /mnt /tmp; do if [ -d "$d" ]; then printf '\n--- %s ---\n' "$d"; ls -la "$d" | head -25; find "$d" -maxdepth 4 -type f \( -name '*eq04*' -o -name 'validate_public_schema.py' -o -name 'nonhermitian_advection_coupled_select.py' -o -name 'AGENTS.md' \) -print; fi; done; mkdir -p /holdout_task
```
- **Container Exit Code**: `0` (`458 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/pretrain_task
total 60
drwxr-xr-x   1 root root 4096 Nov 26 04:50 .
drwxr-xr-x   1 root root 4096 Nov 26 04:50 ..
-rwxr-xr-x   1 root root    0 Nov 26 04:50 .dockerenv
lrwxrwxrwx   1 root root    7 Apr 22  2024 bin -> usr/bin
drwxr-xr-x   2 root root 4096 Apr 22  2024 boot
drwxr-xr-x   5 root root  340 Nov 26 04:50 dev
drwxr-xr-x   1 root root 4096 Nov 26 04:50 etc
drwxr-xr-x   3 root root 4096 Nov 10 14:56 home
lrwxrwxrwx   1 root root    7 Apr 22  2024 lib -> usr/lib
lrwxrwxrwx   1 root root    9 Apr 22  2024 lib64 -> usr/lib64
drwxr-xr-x   2 root root 4096 Nov 10 14:48 media
drwxr-xr-x   2 root root 4096 Nov 10 14:48 mnt
drwxr-xr-x   2 root root 4096 Nov 10 14:48 opt
drwxr-xr-x   2 root root 4096 Nov 23 09:31 pretrain_task
dr-xr-xr-x 650 root root    0 Nov 26 04:50 proc
drwx------   2 root root 4096 Nov 10 14:55 root
drwxr-xr-x   4 root root 4096 Nov 10 14:56 run
lrwxrwxrwx   1 root root    8 Apr 22  2024 sbin -> usr/sbin
drwxr-xr-x   2 root root 4096 Nov 10 14:48 srv
dr-xr-xr-x  13 root root    0 Nov 26 04:50 sys
drwxrwxrwt   1 root root 4096 Nov 26 04:50 tmp
drwxr-xr-x  12 root root 4096 Nov 10 14:48 usr
drwxr-xr-x  11 root root 4096 Nov 10 14:55 var

--- /pretrain_task ---
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 26 04:50 ..

--- /mnt ---
total 8
drwxr-xr-x 2 root root 4096 Nov 10 14:48 .
drwxr-xr-x 1 root root 4096 Nov 26 04:50 ..

--- /tmp ---
total 12
drwxrwxrwt 1 root   root   4096 Nov 26 04:50 .
drwxr-xr-x 1 root   root   4096 Nov 26 04:50 ..
drwxr-xr-x 2 ubuntu ubuntu 4096 Nov 26 04:50 hegel_knowledge_ref
```

### Turn 3 — `execute_in_remote_docker` (`48941 ms` LLM)
- **Rationale**: Implement the five-point boundary-aware operator and enumerate all supports without pruning on nominal cardinality, because the objective counts coefficients using the specified magnitude threshold. Next use an independent stencil and support oracle to detect indexing, conjugation, or pair-penalty errors.
- **Alternatives Weighed**: `Ridge via explicit normal equations` (rejected: Augmented least squares avoids squaring the design condition number while solving the same support problem.)

```bash
set -e
mkdir -p /holdout_task
cat > /holdout_task/nonhermitian_advection_coupled_select.py <<'PY'
"""Complex upwind advection inversion with exhaustive coupled ridge selection."""

import numpy as np
from scipy.linalg import lstsq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _basis_responses(problem):
    basis = np.asarray(problem['basis'], dtype=np.float64)
    m, nx, ny = basis.shape
    hx = problem['Lx'] / (nx + 1)
    hy = problem['Ly'] / (ny + 1)
    dx = problem['nu'] / hx**2
    dy = problem['nu'] / hy**2
    vx = np.asarray(problem['vx'], dtype=np.float64) / hx
    vy = np.asarray(problem['vy'], dtype=np.float64) / hy
    diagonal = (2 * dx + 2 * dy + np.abs(vx) + np.abs(vy)
                + np.asarray(problem['sigma'], dtype=np.float64)
                + 1j * problem['omega'])
    node = np.arange(nx * ny).reshape(nx, ny)
    rows = (node, node[1:, :], node[:-1, :], node[:, 1:], node[:, :-1])
    cols = (node, node[:-1, :], node[1:, :], node[:, :-1], node[:, 1:])
    values = (diagonal,
              -dx - np.maximum(vx[1:, :], 0),
              -dx + np.minimum(vx[:-1, :], 0),
              -dy - np.maximum(vy[:, 1:], 0),
              -dy + np.minimum(vy[:, :-1], 0))
    operator = coo_matrix(
        (np.concatenate([a.ravel() for a in values]),
         (np.concatenate([a.ravel() for a in rows]),
          np.concatenate([a.ravel() for a in cols]))),
        shape=(nx * ny, nx * ny), dtype=np.complex128).tocsc()
    rhs = basis.reshape(m, nx * ny).T.astype(np.complex128)
    return splu(operator).solve(rhs).reshape(nx, ny, m)


def solve_nonhermitian_coupled_inversion(problem: dict) -> dict:
    """Recover complex weights and return the objective and field magnitude."""
    responses = _basis_responses(problem)
    i, j = np.asarray(problem['sensor_ij'], dtype=np.intp).T
    g = responses[i, j, :]
    y = (np.asarray(problem['y_obs_re'], dtype=np.float64)
         + 1j * np.asarray(problem['y_obs_im'], dtype=np.float64))
    lam0, lam2 = problem['lam0'], problem['lam2']
    m = g.shape[1]
    edges = [(int(u), int(v), float(gamma))
             for u, v, gamma in problem['pair_penalties']]
    eye = np.eye(m, dtype=np.complex128)
    # Augmented least squares is equivalent to Hermitian ridge normal
    # equations but avoids squaring the condition number of the design.
    ridge = np.sqrt(lam2)
    best_w = np.zeros(m, dtype=np.complex128)
    best_obj = 0.5 * np.vdot(y, y).real
    for mask in range(1, 1 << m):
        s = [k for k in range(m) if mask & (1 << k)]
        gs = g[:, s]
        fit = lstsq(np.vstack((gs, ridge * eye[:len(s), :len(s)])),
                    np.concatenate((y, np.zeros(len(s), dtype=y.dtype))),
                    cond=0.0, check_finite=False, lapack_driver='gelsy')[0]
        active = np.zeros(m, dtype=bool)
        active[s] = np.abs(fit) > 1e-8
        residual = gs @ fit - y
        obj = (0.5 * np.vdot(residual, residual).real
               + 0.5 * lam2 * np.vdot(fit, fit).real
               + lam0 * np.count_nonzero(active)
               + sum(gamma for u, v, gamma in edges if active[u] and active[v]))
        if obj < best_obj:
            best_obj = obj
            best_w.fill(0)
            best_w[s] = fit
    return {
        'weights_real': best_w.real.copy(),
        'weights_imag': best_w.imag.copy(),
        'objective': float(best_obj),
        'predicted_field_abs': np.abs(responses @ best_w),
    }
PY
```
- **Container Exit Code**: `0` (`358 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/nonhermitian_advection_coupled_select.py']`
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e975499181c9f09b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e9754991 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`55956 ms` LLM)
- **Rationale**: Import the exact artifact in an isolated process and compare it with a separately assembled dense stencil and an SVD ridge oracle. Residual failures would point to assembly or boundary indexing; objective or field mismatches would point to support scoring, weight fitting, or field reconstruction. Changing only one pair penalty tests whether it actually changes the selected support.
- **Alternatives Weighed**: `Validate only array shapes and types` (rejected: A schema check would miss stencil sign errors, incorrect complex conjugation, and a selector that ignores pair penalties.)

```bash
set -e
cat > /tmp/verify_eq04.py <<'PY'
import importlib.util
import itertools
import pathlib
import time

import numpy as np
from scipy.linalg import solve

path = '/holdout_task/nonhermitian_advection_coupled_select.py'
spec = importlib.util.spec_from_file_location('candidate', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
solver = module.solve_nonhermitian_coupled_inversion

def dense_reference(p):
    m, nx, ny = p['basis'].shape
    A = np.zeros((nx*ny, nx*ny), dtype=np.complex128)
    hx, hy = p['Lx']/(nx+1), p['Ly']/(ny+1)
    for i, j in np.ndindex(nx, ny):
        row = i*ny+j
        vx, vy = p['vx'][i,j], p['vy'][i,j]
        A[row,row] = (2*p['nu']/hx**2 + 2*p['nu']/hy**2 + abs(vx)/hx + abs(vy)/hy + p['sigma'][i,j] + 1j*p['omega'])
        for a, b, h in ((i-1,j,hx), (i+1,j,hx), (i,j-1,hy), (i,j+1,hy)):
            if 0 <= a < nx and 0 <= b < ny:
                A[row,a*ny+b] = -p['nu']/h**2
        a = i-1 if vx >= 0 else i+1
        b = j-1 if vy >= 0 else j+1
        if 0 <= a < nx:
            A[row,a*ny+j] -= abs(vx)/hx
        if 0 <= b < ny:
            A[row,i*ny+b] -= abs(vy)/hy
    rhs = p['basis'].reshape(m, -1).T
    return A, rhs, solve(A, rhs, assume_a='gen')

def objective(g, y, p, w):
    active = np.abs(w)>1e-8
    return (np.sum(abs(g@w-y)**2)/2 + p['lam2']*np.sum(abs(w)**2)/2 + p['lam0']*sum(active)
            + sum(gamma for u,v,gamma in p['pair_penalties'] if active[u] and active[v]))

def oracle(g, y, p):
    scores = []
    for k in range(g.shape[1]+1):
        for tup in itertools.combinations(range(g.shape[1]), k):
            w = np.zeros(g.shape[1], dtype=np.complex128)
            if k:
                u, s, vh = np.linalg.svd(g[:, tup], full_matrices=False)
                w[list(tup)] = vh.conj().T @ ((u.conj().T@y) * s/(s*s+p['lam2']))
            scores.append((objective(g,y,p,w), w))
    scores.sort(key=lambda x: x[0]); return scores[0], scores[1][0]-scores[0][0]

worst = np.zeros(4)
for idx, label in enumerate(('mixed-upwind/weak-pair', 'same-data/strong-pair', 'indefinite/negative-frequency', 'duplicate-basis/underdetermined', 'null-support')):
    rng = np.random.default_rng(84 + (0 if idx==1 else idx))
    shape = (32,32)
    basis = rng.normal(size=(11,*shape))
    if idx==3: basis[-1]=basis[-2]
    p = dict(basis=basis, Lx=0.67, Ly=2.3, nu=0.002 if idx==2 else 0.012,
             vx=rng.uniform(-3,3,shape), vy=rng.uniform(-1,1,shape),
             omega=-0.41 if idx==2 else 7.1,
             sigma=rng.uniform(-140,10,shape) if idx==2 else rng.uniform(0,5,shape),
             lam0=0.04, lam2=0.007 if idx!=3 else 1e-6)
    A, rhs, h = dense_reference(p)
    # Normalize the response scale so penalties, data fit and ridge all matter.
    scale = 0.4 / np.median(np.abs(h))
    rhs = rhs * scale
    p['basis'] = basis * scale
    h = h * scale
    ids = np.unique(np.r_[0,31,992,1023,rng.choice(1024,size=4 if idx==3 else 37,replace=False)])
    p['sensor_ij'] = np.column_stack(np.divmod(ids,32))
    g = h[ids]
    truth = np.zeros(11,dtype=complex); truth[[2,6,8]] = [1+0.7j, -0.5+0.6j, 0.2+0.1j]
    y = g@truth + 0.03*(rng.normal(size=len(ids))+1j*rng.normal(size=len(ids)))
    if idx==4: y *= 0
    p.update(y_obs_re=y.real,y_obs_im=y.imag,
             pair_penalties=[(2,6,5000 if idx==1 else 0.03), (8,2,0.012), (6,8,0.032), (0,10,0.12)])
    (ref, w_ref), gap = oracle(g,y,p)
    raw = module._basis_responses(p).reshape(1024,11)
    residual = np.linalg.norm(A@raw-rhs)/np.linalg.norm(rhs)
    t = time.perf_counter(); got = solver(p); elapsed = time.perf_counter()-t
    assert set(got)=={'weights_real','weights_imag','objective','predicted_field_abs'}
    for key, dims in [('weights_real',(11,)),('weights_imag',(11,)),('predicted_field_abs',shape)]:
        assert got[key].shape == dims and got[key].dtype == np.float64
        assert np.isfinite(got[key]).all()
    assert isinstance(got['objective'],float)
    w = got['weights_real'] + 1j*got['weights_imag']
    active = np.abs(w)>1e-8
    if idx==0: assert active[2] and active[6]
    if idx==1: assert not (active[2] and active[6])
    if idx==4: assert not active.any()
    field = abs(h@w_ref).reshape(shape)
    field_error = np.linalg.norm(field-got['predicted_field_abs'])/max(np.linalg.norm(field),1e-30)
    errors = [residual, abs(ref-got['objective']), field_error, abs(objective(g,y,p,w)-got['objective'])]
    print(f'{label}: residual={errors[0]:.3g} objective_error={errors[1]:.3g} field_rel={errors[2]:.3g} objective_consistency={errors[3]:.3g} runner_up_gap={gap:.3g} active={np.where(active)[0]} solver_s={elapsed:.3f}',flush=True)
    assert np.all(np.array(errors) <= [1e-10,1e-6,1e-5,1e-6])
    worst = np.maximum(worst,errors)
print('WORST residual, objective abs, field rel, objective consistency:',worst)
validator=pathlib.Path('/holdout_task/validate_public_schema.py')
cases=pathlib.Path('/holdout_task/public_eq04_cases.pkl')
print('Public validator present:',validator.is_file(),'public fixtures present:',cases.is_file())
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/verify_eq04.py
```
- **Container Exit Code**: `0` (`1816 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
mixed-upwind/weak-pair: residual=9.98e-16 objective_error=5.55e-17 field_rel=1.02e-15 objective_consistency=0 runner_up_gap=0.0382 active=[2 6 8] solver_s=0.069
same-data/strong-pair: residual=9.98e-16 objective_error=2.22e-16 field_rel=1.03e-15 objective_consistency=0 runner_up_gap=0.0117 active=[ 2  8  9 10] solver_s=0.066
indefinite/negative-frequency: residual=1.44e-14 objective_error=1.05e-15 field_rel=2.21e-15 objective_consistency=1.05e-15 runner_up_gap=0.0346 active=[2 6 8] solver_s=0.064
duplicate-basis/underdetermined: residual=1e-15 objective_error=2.78e-17 field_rel=8.52e-16 objective_consistency=5.55e-17 runner_up_gap=0.0322 active=[2 6] solver_s=0.061
null-support: residual=1.01e-15 objective_error=0 field_rel=0 objective_consistency=0 runner_up_gap=0 active=[] solver_s=0.066
WORST residual, objective abs, field rel, objective consistency: [1.43981138e-14 1.05471187e-15 2.20616494e-15 1.05471187e-15]
Public validator present: False public fixtures present: False

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e975499181c9f09b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e9754991 (directory).
```

### Turn 5 — `submit_completed_task` (`14182 ms` LLM)
- **Rationale**: The exact artifact passed isolated verification against an independently assembled dense stencil and an exhaustive SVD ridge reference. Increasing one pair penalty removed the corresponding co-activation on otherwise identical inputs.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`13` turns remaining; artifact status={'/holdout_task/nonhermitian_advection_coupled_select.py': 'FILE:3362'}): Do NOT lock prematurely with `13` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/nonhermitian_advection_coupled_select.py`. All five independent synthetic checks passed; worst absolute objective error was 1.05e-15 and worst relative field error was 2.21e-15. The public fixtures and validator were not mounted, so public-case validation could not be run.`

### Turn 6 — `execute_in_remote_docker` (`45750 ms` LLM)
- **Rationale**: Use exact discrete sine modes to isolate near-resonant complex phase and amplitude errors without relying on the previous dense stencil oracle. The scalar ridge reference also tests overlapping pair penalties and the magnitude threshold. A discrepancy would distinguish a PDE response error from incorrect support scoring.
- **Alternatives Weighed**: `Batch-centering or relay-mode tests` (rejected: The input contract has neither batch variables nor relay modes; those tests cannot reveal a failure of this solver.)

```bash
set -e
if [ -f /holdout_task/validate_public_schema.py ] && [ -f /holdout_task/public_eq04_cases.pkl ]; then OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py; else echo 'Public fixtures and validator are not available.'; fi
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import itertools
import numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/nonhermitian_advection_coupled_select.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
n = 32
Lx, Ly, nu = 0.81, 1.73, 0.015
hx, hy = Lx/33, Ly/33
x = np.arange(1,33)
ks = [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(3,2),(2,3),(3,3),(4,1),(1,4)]
phi = np.stack([2/33 * np.outer(np.sin(np.pi*a*x/33),np.sin(np.pi*b*x/33)) for a,b in ks])
eig = np.array([4*nu*(np.sin(np.pi*a/66)**2/hx**2+np.sin(np.pi*b/66)**2/hy**2) for a,b in ks])
sigma = -eig[0] + 1e-5
edges = [(0,1,.65),(0,2,.42),(1,3,.35),(2,3,.40),(3,4,.38),(4,5,.24),(0,5,.40),(7,10,9.)]
beta = np.array([1.05,.93,.84,.81,.74,.71,.23,0,.19,.11,0]) * np.exp(1j*np.arange(11)*.91)
beta[10] = 5e-9 * (1+.2j)
lam0, lam2 = .075, .13
for omega in (-1e-3, 1e-3):
    denom = eig + sigma + 1j*omega
    gains = abs(denom)/denom
    y = np.einsum('m,mij->ij', beta, phi)
    p = dict(Lx=Lx,Ly=Ly,nu=nu,omega=omega,sigma=np.full((n,n),sigma),vx=np.zeros((n,n)),vy=np.zeros((n,n)),basis=abs(denom)[:,None,None]*phi,sensor_ij=np.indices((n,n)).reshape(2,-1).T,y_obs_re=y.real.ravel(),y_obs_im=y.imag.ravel(),lam0=lam0,lam2=lam2,pair_penalties=edges)
    # Mode orthogonality reduces the reference fit to scalar ridge problems.
    w_unrestricted = gains.conj()*beta/(1+lam2)
    ranking = []
    for flags in itertools.product((False,True),repeat=10):
        w = w_unrestricted * np.r_[flags,True]
        active = abs(w)>1e-8
        val = .5*np.sum(abs(gains*w-beta)**2+lam2*abs(w)**2)+lam0*sum(active)+sum(c for u,v,c in edges if active[u] and active[v])
        ranking.append((val,w))
    ranking.sort(key=lambda t:t[0]); ref,w = ranking[0]
    got = module.solve_nonhermitian_coupled_inversion(p)
    wf = got['weights_real']+1j*got['weights_imag']
    field = abs(np.einsum('m,mij->ij',gains*w,phi))
    jerr = abs(got['objective']-ref)
    ferr = np.linalg.norm(field-got['predicted_field_abs'])/np.linalg.norm(field)
    werr = np.linalg.norm(wf-w)
    print(f'omega={omega:+g} objective_abs={jerr:.3g} field_rel={ferr:.3g} weight_L2={werr:.3g} support_gap={ranking[1][0]-ref:.3g} active={np.flatnonzero(abs(wf)>1e-8)}')
    assert jerr<=1e-6 and ferr<=1e-5 and werr<=1e-6
    assert 0 < abs(wf[10]) < 1e-8
PY
```
- **Container Exit Code**: `1` (`1541 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public fixtures and validator are not available.
omega=-0.001 objective_abs=4.97e-13 field_rel=3.99e-13 weight_L2=5.22e-13 support_gap=0 active=[1 2 4]
omega=+0.001 objective_abs=4.97e-13 field_rel=2.45e-09 weight_L2=4.51e-09 support_gap=0 active=[1 2 4]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e975499181c9f09b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e9754991 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 41, in <module>
AssertionError
```

### Turn 7 — `execute_in_remote_docker` (`54397 ms` LLM)
- **Rationale**: Recheck the exact artifact against the actual tolerances, then distinguish harmless rounding from a structural omission using an orthogonal mode whose response makes a subthreshold weight matter. A failure on that slice would justify adding constrained inactive-weight optimization rather than changing tie-breaking.
- **Alternatives Weighed**: `Force the near-resonant fit to retain an individually subthreshold coefficient` (rejected: The measured difference is below both deliverable tolerances, and the contract does not require that coefficient to be nonzero.)

```bash
set -e
cat > /tmp/verify_eq04_threshold.py <<'PY'
import importlib.util
import itertools
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/nonhermitian_advection_coupled_select.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Lx, Ly, nu = .81, 1.73, .015
hx, hy = Lx/33, Ly/33
x = np.arange(1, 33)
ks = [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(3,2),(2,3),(3,3),(4,1),(1,4)]
phi = np.stack([2/33 * np.outer(np.sin(np.pi*a*x/33), np.sin(np.pi*b*x/33)) for a,b in ks])
eig = np.array([4*nu*(np.sin(np.pi*a/66)**2/hx**2 + np.sin(np.pi*b/66)**2/hy**2) for a,b in ks])
sigma = -eig[0] + 1e-5
edges = [(0,1,.65),(0,2,.42),(1,3,.35),(2,3,.40),(3,4,.38),(4,5,.24),(0,5,.40),(7,10,9.)]

def fixture(omega, scales, beta, lam0, lam2):
    denom = eig + sigma + 1j*omega
    gains = scales*abs(denom)/denom
    y = np.einsum('m,mij->ij', beta, phi)
    p = dict(Lx=Lx,Ly=Ly,nu=nu,omega=omega,sigma=np.full((32,32),sigma),vx=np.zeros((32,32)),vy=np.zeros((32,32)),basis=(scales*abs(denom))[:,None,None]*phi,sensor_ij=np.indices((32,32)).reshape(2,-1).T,y_obs_re=y.real.ravel(),y_obs_im=y.imag.ravel(),lam0=lam0,lam2=lam2,pair_penalties=edges)
    return p, gains

for omega in (-1e-3,1e-3):
    beta = np.array([1.05,.93,.84,.81,.74,.71,.23,0,.19,.11,0])*np.exp(1j*np.arange(11)*.91)
    beta[10] = 5e-9*(1+.2j)
    p, gains = fixture(omega,np.ones(11),beta,.075,.13)
    w_ridge = gains.conj()*beta/(abs(gains)**2+p['lam2'])
    best = (np.inf,None)
    for flags in itertools.product((False,True),repeat=11):
        w = w_ridge * flags
        active = abs(w)>1e-8
        obj = .5*np.sum(abs(gains*w-beta)**2 + p['lam2']*abs(w)**2)+p['lam0']*sum(active)+sum(c for u,v,c in edges if active[u] and active[v])
        if obj < best[0]: best = obj,w
    got = module.solve_nonhermitian_coupled_inversion(p)
    field = abs(np.einsum('m,mij->ij',gains*best[1],phi))
    jerr = abs(got['objective']-best[0]); ferr = np.linalg.norm(field-got['predicted_field_abs'])/np.linalg.norm(field)
    wf = got['weights_real']+1j*got['weights_imag']
    print(f'near-resonant omega={omega:+g}: objective_abs={jerr:.3g}, field_rel={ferr:.3g}, subthreshold_weight={abs(wf[10]):.3g}')
    assert jerr<=1e-6 and ferr<=1e-5 and abs(wf[10])<=1e-8

# An orthogonal mode with a large response makes a subthreshold weight
# physically significant. The inactive disk then has a nonzero boundary optimum.
scales = np.ones(11); scales[0] = 1e8
beta = np.zeros(11,dtype=complex); beta[0] = 2
p, gains = fixture(.7,scales,beta,3.,.13)
got = module.solve_nonhermitian_coupled_inversion(p)
radius = 1e-8
inactive = .5*((2-abs(gains[0])*radius)**2 + p['lam2']*radius**2)
active = 3 + .5*4*p['lam2']/(abs(gains[0])**2+p['lam2'])
assert inactive < active
field = abs(phi[0])*abs(gains[0])*radius
jerr = abs(got['objective']-inactive); ferr = np.linalg.norm(got['predicted_field_abs']-field)/np.linalg.norm(field)
print(f'inactive-disk slice: expected_objective={inactive:.12g}, returned={got["objective"]:.12g}, objective_abs={jerr:.3g}, field_rel={ferr:.3g}',flush=True)
assert jerr <= 1e-6 and ferr <= 1e-5
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/verify_eq04_threshold.py
```
- **Container Exit Code**: `1` (`2007 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
near-resonant omega=-0.001: objective_abs=4.97e-13, field_rel=2.45e-09, subthreshold_weight=4.51e-09
near-resonant omega=+0.001: objective_abs=4.97e-13, field_rel=3.73e-13, subthreshold_weight=0
inactive-disk slice: expected_objective=0.5, returned=2, objective_abs=1.5, field_rel=1

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e975499181c9f09b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e9754991 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/verify_eq04_threshold.py", line 55, in <module>
    assert jerr <= 1e-6 and ferr <= 1e-5
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

### Turn 8 — `execute_in_remote_docker` (`40072 ms` LLM)
- **Rationale**: Look up the prescribed ridge-profiling reference before implementing the constrained refinement. A matching reference can guide stable elimination of the unrestricted coordinates; otherwise use QR projection and a reduced convex disk solve. A supporting-plane bound can exclude masks that cannot improve the incumbent.
- **Alternatives Weighed**: `Clip each coefficient from the unconstrained ridge fit` (rejected: Clipping solves the orthogonal witness but not the coupled quadratic for correlated sensor responses.); `Require penalized coordinates to stay strictly above the cutoff` (rejected: The relaxed solve can leave them unrestricted: any that fall below the cutoff only reduce the actual penalty because all penalties are positive.)
- **ATMS NOGOOD DECLARED**: Refuted `Enumerating unconstrained ridge fits on zero-filled supports covers the objective defined by the strict magnitude cutoff.` | Witness: `The orthogonal high-gain slice attains objective 0.5 on the unpenalized radius-1e-8 boundary; the current solver returns 2 with a zero field.` | Blanket Diagnosis: `Penalty inactivity constrains a complex coefficient to a closed disk; it does not force the coefficient to zero. The disk boundary can produce a measurable response.` | Sublated Pivot: `Keep sparse LU and exact mask enumeration, but for each competitive mask profile out unconstrained coefficients and minimize over the remaining closed disks.`

```bash
set -e
ref=/tmp/hegel_knowledge_ref
python3 "$ref/describe_ref.py" --search 'ridge nuisance profiling variable projection constrained least squares'
ls -1 "$ref" | head -30
```
- **Container Exit Code**: `0` (`466 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ivation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
_skill_contracts.json
autogenous_conservative_elliptic_multigrid.reference.txt
autogenous_prim_box_ridge_bounds.reference.txt
autogenous_prim_conservative_spd_solve.reference.txt
autogenous_prim_reference_panel_balancing.reference.txt
autogenous_prim_ridge_profiled_matching.reference.txt
autogenous_prim_robust_affine_projection.reference.txt
autogenous_prim_robust_trend_decomposition.reference.txt
autogenous_robust_network_assimilation.reference.txt
autogenous_sparse_ridge_certification.reference.txt
describe_ref.py

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e975499181c9f09b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e9754991 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`30383 ms` LLM)
- **Rationale**: Read the two reference notes to separate reusable ridge elimination and lower-bound arguments from assumptions specific to real boxes. For the complex problem, the profiled quadratic's tangent is minimized over each inactive disk in closed form, giving a bound for pruning penalty masks.
- **Alternatives Weighed**: `Apply the catalog's real box certificate directly` (rejected: Independent real bounds do not describe a complex magnitude disk, and the certificate does not price pair co-activation.)

```bash
set -e
ref=/tmp/hegel_knowledge_ref
python3 "$ref/describe_ref.py" --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
python3 "$ref/describe_ref.py" --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound
sed -n '1,135p' "$ref/autogenous_prim_ridge_profiled_matching.reference.txt"
sed -n '1,105p' "$ref/autogenous_prim_box_ridge_bounds.reference.txt"
```
- **Container Exit Code**: `0` (`462 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
inalg.eigvalsh(0.5 * (_G_chk + _G_chk.T))[0])
            else:
                _off_g = np.sum(np.abs(_G_chk), axis=1) - np.abs(np.diag(_G_chk))
                _min_eig = float(np.min(np.diag(_G_chk) - _off_g))
            _scale_eig = max(1.0, float(np.max(_col_sq)), float(np.max(np.abs(_d_chk))))
            if _min_eig < -1e-9 * _scale_eig:
                raise ValueError('diagonal must be a valid PSD minorant of design.T @ design (without ridge)')
            if _min_eig < 0.0:
                _c_chk = np.asarray(center, dtype=float).ravel()
                if _c_chk.size == _p_chk and np.all(np.isfinite(_c_chk)) and np.isfinite(float(box)):
                    _psd_slack_m89 = 0.5 * (-_min_eig + 2e-15 * _scale_eig) * float(np.sum((abs(float(box)) + np.abs(_c_chk)) ** 2))
    ld = np.longdouble
    (x, y, c) = (np.asarray(a, dtype=ld) for a in (design, target, center))
    (n, p) = x.shape
    zero = np.zeros(p, dtype=bool) if fixed_zero is None else np.asarray(fixed_zero, dtype=bool)
    active = np.zeros(p, dtype=bool) if fixed_active is None else np.asarray(fixed_active, dtype=bool)
    if np.any(zero & active) or min(box, activation_cost, ridge) < 0:
        raise ValueError('indicator masks must be disjoint and box and penalties nonnegative')
    d = np.zeros(p, dtype=ld) if diagonal is None else np.asarray(diagonal, dtype=ld)
    (m, l0, l2) = map(ld, (box, activation_cost, ridge))
    guard = 8 * (max(n, p) + 4) * np.finfo(ld).eps
    (ax, ac) = (np.abs(x), np.abs(c))
    r = y - x @ c
    er = guard * (np.abs(y) + ax @ ac)
    g = x.T @ r
    eg = (1 + guard) * (ax.T @ er) + guard * (ax.T @ np.abs(r))
    rlo = np.maximum(np.abs(r) - er, 0)
    dc = d * c
    (q, h) = (rlo @ rlo, dc @ c)
    constant = (q + h) / 2 + g @ c
    z = np.abs(g + dc) + eg + guard * (np.abs(g) + np.abs(dc))
    curv = l2 + d
    b = np.divide(z, curv, out=np.full(p, m, dtype=ld), where=(curv > 0) & (z < m * curv))
    gain = b * (z - curv * b / 2)
    free = ~(zero | active)
    on = l0 - gain
    value = constant + on[active].sum() + np.minimum(on[free], 0).sum()
    err = eg @ ac + guard * (q + h + np.abs(g) @ ac + p * l0 + gain.sum())
    return float(np.nextafter(float(value - err) - _psd_slack_m89, -np.inf))

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=e975499181c9f09b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_e9754991 (directory).
```

### Turn 10 — `execute_in_remote_docker` (`64241 ms` LLM)
- **Rationale**: Profile unrestricted variables by QR and use a disk tangent lower bound to screen competing penalty masks. Re-run the original isolated oracle tests to detect regressions and the high-gain witness to verify the missing boundary solution before closing the nogood.
- **Alternatives Weighed**: `Use independent real box constraints` (rejected: A square includes coefficients whose complex magnitude exceeds the activation cutoff.); `Clip unconstrained ridge coefficients and stop` (rejected: Clipping does not re-optimize correlated coefficients or the unrestricted coordinates.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/nonhermitian_advection_coupled_select.py')
s = p.read_text().replace('from scipy.linalg import lstsq', 'from scipy.linalg import qr, solve_triangular\nfrom scipy.optimize import minimize')
s = s[:s.index('def solve_nonhermitian_coupled_inversion')]
s += '''_CUTOFF = 1e-8


def _disk_fit(design, target):
    """Minimize the profiled quadratic over a product of complex unit disks."""
    adjoint = design.conj().T
    linear = adjoint @ target
    norms = np.sum(np.abs(design)**2, axis=0)
    scale = max(np.max(norms), np.max(np.abs(linear)))
    n = len(linear)
    if scale == 0:
        return np.zeros(n, dtype=np.complex128)

    def project(z):
        return z / np.maximum(1., np.abs(z))

    def value_grad(z):
        x = z[:n] + 1j*z[n:]
        dx = design @ x
        v = (0.5*np.vdot(dx, dx).real - np.vdot(linear, x).real)/scale
        g = (adjoint @ dx - linear)/scale
        return v, np.r_[g.real, g.imag]

    def disk(z):
        return 1. - z[:n]**2 - z[n:]**2

    def disk_jac(z):
        return -2.*np.hstack((np.diag(z[:n]), np.diag(z[n:])))

    # A shifted least-squares objective avoids losing small disk corrections
    # in the constant residual energy. The tangent gap is zero at a minimizer.
    x = np.zeros(n, dtype=np.complex128)
    for iteration in range(1024):
        residual = target - design @ x
        for j in range(n):
            if norms[j] == 0:
                continue
            update = x[j] + np.vdot(design[:, j], residual)/norms[j]
            update /= max(1., abs(update))
            residual -= design[:, j]*(update - x[j])
            x[j] = update
        gradient = (adjoint @ (design @ x) - linear)/scale
        gap = np.sum(np.abs(gradient)) + np.vdot(gradient, x).real
        if gap <= 2e-13:
            break
        if iteration == 3:
            opt = minimize(value_grad, np.r_[x.real, x.imag], jac=True,
                           method='SLSQP',
                           constraints={'type': 'ineq', 'fun': disk, 'jac': disk_jac},
                           options={'ftol': 2e-14, 'maxiter': 160})
            candidate = project(opt.x[:n] + 1j*opt.x[n:])
            if value_grad(np.r_[candidate.real, candidate.imag])[0] < value_grad(np.r_[x.real, x.imag])[0]:
                x = candidate
    # Round just inside the boundary so conversion to float64 weights cannot
    # turn an inactive disk constraint into an activation penalty.
    return project(x) * (_CUTOFF * (1. - 4.*np.finfo(float).eps))


def solve_nonhermitian_coupled_inversion(problem: dict) -> dict:
    """Recover complex weights and return the objective and field magnitude."""
    responses = _basis_responses(problem)
    i, j = np.asarray(problem['sensor_ij'], dtype=np.intp).T
    g = responses[i, j, :]
    y = (np.asarray(problem['y_obs_re'], dtype=np.float64)
         + 1j * np.asarray(problem['y_obs_im'], dtype=np.float64))
    lam0, lam2 = problem['lam0'], problem['lam2']
    m = g.shape[1]
    edges = [(int(u), int(v), float(gamma))
             for u, v, gamma in problem['pair_penalties']]

    def penalty(mask):
        return lam0*mask.bit_count() + sum(
            gamma for u,v,gamma in edges if mask & (1 << u) and mask & (1 << v))

    # QR compresses augmented ridge least squares to m rows; the discarded
    # residual component is constant over all masks.
    q, r = qr(np.vstack((g, np.sqrt(lam2)*np.eye(m))), mode='economic', check_finite=False)
    b = q[:len(y)].conj().T @ y
    best_w = np.zeros(m, dtype=np.complex128)
    best = 0.5*np.vdot(b,b).real
    masks = sorted((penalty(mask), mask) for mask in range(1 << m))
    for cost, mask in masks:
        if cost > best:
            break
        free = [k for k in range(m) if mask & (1 << k)]
        bounded = [k for k in range(m) if not mask & (1 << k)]
        if free:
            uq, ur = qr(r[:, free], mode='full', check_finite=False)
            rot = uq.conj().T @ np.column_stack((b, r[:, bounded]))
        else:
            rot = np.column_stack((b, r))
        k = len(free)
        target, design = rot[k:, 0], rot[k:, 1:]
        base = 0.5*np.vdot(target,target).real
        gain = _CUTOFF*np.sum(np.abs(design.conj().T @ target))
        # The tangent at zero lower-bounds the objective on all the disks.
        if cost + base - gain > best + 32*np.finfo(float).eps*(base + gain + best):
            continue
        w = np.zeros(m, dtype=np.complex128)
        if bounded:
            w[bounded] = _disk_fit(_CUTOFF*design, target)
        if free:
            w[free] = solve_triangular(ur[:k], rot[:k, 0] - rot[:k, 1:] @ w[bounded], check_finite=False)
        # Free coordinates need not exceed the threshold. If they do not,
        # their actual cost only decreases. Every weight vector is feasible
        # for the relaxed mask matching its actual active coordinates.
        actual = sum(1 << j for j, v in enumerate(np.abs(w)) if v > _CUTOFF)
        residual = r @ w - b
        score = 0.5*np.vdot(residual,residual).real + penalty(actual)
        if score < best:
            best, best_w = score, w

    residual = g @ best_w - y
    actual = sum(1 << j for j, v in enumerate(np.abs(best_w)) if v > _CUTOFF)
    objective = (0.5*np.vdot(residual,residual).real
                 + 0.5*lam2*np.vdot(best_w,best_w).real + penalty(actual))
    return {
        'weights_real': best_w.real.copy(),
        'weights_imag': best_w.imag.copy(),
        'objective': float(objective),
        'predicted_field_abs': np.abs(responses @ best_w),
    }
'''
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -I /tmp/verify_eq04.py
python3 -I /tmp/verify_eq04_threshold.py
```
- **Container Exit Code**: `0` (`2280 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/nonhermitian_advection_coupled_select.py']`
- **Stdout (tail)**:
```text
mixed-upwind/weak-pair: residual=9.98e-16 objective_error=9.06e-09 field_rel=1.81e-08 objective_consistency=0 runner_up_gap=0.0382 active=[2 6 8] solver_s=0.026
same-data/strong-pair: residual=9.98e-16 objective_error=8.18e-08 field_rel=2.19e-08 objective_consistency=4.44e-16 runner_up_gap=0.0117 active=[ 2  8  9 10] solver_s=0.118
indefinite/negative-frequency: residual=1.44e-14 objective_error=2.17e-08 field_rel=6.91e-09 objective_consistency=1.03e-15 runner_up_gap=0.0346 active=[2 6 8] solver_s=0.034
duplicate-basis/underdetermined: residual=1e-15 objective_error=4.53e-09 field_rel=2.19e-08 objective_consistency=5.55e-17 runner_up_gap=0.0322 active=[2 6] solver_s=0.017
null-support: residual=1.01e-15 objective_error=0 field_rel=0 objective_consistency=0 runner_up_gap=0 active=[] solver_s=0.005
WORST residual, objective abs, field rel, objective consistency: [1.43981138e-14 8.18457717e-08 2.19254455e-08 1.02695630e-15]
Public validator present: False public fixtures present: False
near-resonant omega=-0.001: objective_abs=3.1e-08, field_rel=1.41e-08, subthreshold_weight=4.51e-09
near-resonant omega=+0.001: objective_abs=3.1e-08, field_rel=1.41e-08, subthreshold_weight=4.51e-09
inactive-disk slice: expected_objective=0.5, returned=0.5, objective_abs=8.88e-16, field_rel=8.44e-16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ae133e99323fd84c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ae133e99 (directory).
```

### Turn 11 — `execute_in_remote_docker` (`44755 ms` LLM)
- **Rationale**: Use a constructed KKT optimum and a strict objective margin over all competing masks to independently verify the new coupled disk solver. A mismatch would warrant correcting the reduced optimization or the free-coordinate recovery before submission.
- **Alternatives Weighed**: `Submit after the orthogonal boundary test` (rejected: That test does not exercise correlated inactive coefficients or their interaction with an unrestricted coefficient.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/nonhermitian_advection_coupled_select.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
n = 32
x = np.arange(1, n+1)
Lx, Ly, nu, omega = .81, 1.73, .015, -.7
modes = [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(3,2),(2,3),(3,3),(4,1),(1,4)]
phi = np.stack([2/33*np.outer(np.sin(np.pi*a*x/33),np.sin(np.pi*b*x/33)) for a,b in modes]).reshape(11,-1).T
eig = np.array([4*nu*((np.sin(np.pi*a/66)/(Lx/33))**2+(np.sin(np.pi*b/66)/(Ly/33))**2) for a,b in modes])
den = eig + 1j*omega
phase = abs(den)/den
C = np.eye(11)
C[:3,:3] = 1e8*np.array([[1.,.45,-.18],[.41,1.,.42],[.21,-.26,1.]])
C[:3,3] = [.4,-.3,.2]
D = phase[:,None]*C
lam0, lam2 = 3., .01
w = np.zeros(11,dtype=complex)
w[:4] = [1e-8*np.exp(.43j), 1e-8*np.exp(-1.14j), 1e-8*(.23+.17j), 8+2j]
# Construct the residual from KKT conditions: two binding disks, one
# interior disk and one unrestricted coefficient. Strict convexity gives
# a unique optimum for the mask in which only coefficient 3 is free.
kkt = np.zeros(11,dtype=complex)
kkt[:2] = np.array([1.9e7, 2.4e7])*w[:2]/1e-8
r = np.linalg.solve(D.conj().T, lam2*w+kkt)
y_modes = D@w+r
y = phi@y_modes
obj = lam0 + .5*(np.vdot(r,r).real+lam2*np.vdot(w,w).real)
# Any mask not freeing coordinate 3 has a large residual in mode 3,
# independent of its other free coordinates. Any mask freeing coordinate 3
# plus another coefficient pays at least 2*lam0.
global_margin = min(.5*(abs(y_modes[3])-1e-8)**2, 2*lam0)-obj
assert global_margin > 0
p = dict(Lx=Lx,Ly=Ly,nu=nu,omega=omega,sigma=np.zeros((n,n)),vx=np.zeros((n,n)),vy=np.zeros((n,n)),basis=(phi@(abs(den)[:,None]*C)).T.reshape(11,n,n),sensor_ij=np.indices((n,n)).reshape(2,-1).T,y_obs_re=y.real,y_obs_im=y.imag,lam0=lam0,lam2=lam2,pair_penalties=[(0,3,1.1),(1,3,.6),(2,3,.9)])
ans = mod.solve_nonhermitian_coupled_inversion(p)
w_fit = ans['weights_real']+1j*ans['weights_imag']
field = abs(phi@(D@w)).reshape(n,n)
jerr = abs(ans['objective']-obj)
ferr = np.linalg.norm(ans['predicted_field_abs']-field)/np.linalg.norm(field)
print(f'correlated disk KKT oracle: objective_abs={jerr:.3g} field_rel={ferr:.3g} other_mask_margin={global_margin:.3g} active={np.flatnonzero(abs(w_fit)>1e-8)}')
assert jerr <= 1e-6 and ferr <= 1e-5
assert np.array_equal(np.flatnonzero(abs(w_fit)>1e-8),[3])
PY
```
- **Container Exit Code**: `0` (`595 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
correlated disk KKT oracle: objective_abs=1.78e-15 field_rel=7.53e-12 other_mask_margin=2.58 active=[3]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ae133e99323fd84c (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_ae133e99 (directory).
```

### Turn 12 — `submit_completed_task` (`15708 ms` LLM)
- **Rationale**: The verified artifact corrected the high-gain boundary case to below 1e-15 on both metrics and matched the independent correlated-disk KKT reference to 1.78e-15 in objective and 7.53e-12 in relative field error.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/nonhermitian_advection_coupled_select.py`. In isolated synthetic tests, the largest absolute objective discrepancy was 8.18e-8 and the largest relative field discrepancy was 2.19e-8, both within tolerance. The public fixtures and validator were not mounted.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/nonhermitian_advection_coupled_select.py": "FILE:6821"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "all_valid": true,
    "max_weights_rel_l2": 2e-08,
    "max_objective_abs_err": 7e-09,
    "max_field_rel_l2": 1e-08,
    "student_elapsed_sec": 0.2073,
    "instance_diagnostics": [
      {
        "seed": 99431,
        "w_rel_l2": 2e-08,
        "obj_abs_err": 6e-09,
        "field_rel_l2": 1e-08
      },
      {
        "seed": 99432,
        "w_rel_l2": 2e-08,
        "obj_abs_err": 7e-09,
        "field_rel_l2": 1e-08
      },
      {
        "seed": 99433,
        "w_rel_l2": 2e-08,
        "obj_abs_err": 6e-09,
        "field_rel_l2": 1e-08
      }
    ]
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_weights_rel_l2\": 2e-08, \"max_objective_abs_err\": 7e-09, \"max_field_rel_l2\": 1e-08, \"student_elapsed_sec\": 0.2073, \"instance_diagnostics\": [{\"seed\": 99431, \"w_rel_l2\": 2e-08, \"obj_abs_err\": 6e-09, \"field_rel_l2\": 1e-08}, {\"seed\": 99432, \"w_rel_l2\": 2e-08, \"obj_abs_err\": 7e-09, \"field_rel_l2\": 1e-08}, {\"seed\": 99433, \"w_rel_l2\": 2e-08, \"obj_abs_err\": 6e-09, \"field_rel_l2\": 1e-08}]}\n__HEGEL_VERIFIER_AUTH_c9286b7f62f59458489bc43ddc68da14__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_weights_rel_l2\": 2e-08, \"max_objective_abs_err\": 7e-09, \"max_field_rel_l2\": 1e-08, \"student_elapsed_sec\": 0.2073, \"instance_diagnostics\": [{\"seed\": 99431, \"w_rel_l2\": 2e-08, \"obj_abs_err\": 6e-09, \"field_rel_l2\": 1e-08}, {\"seed\": 99432, \"w_rel_l2\": 2e-08, \"obj_abs_err\": 7e-09, \"field_rel_l2\": 1e-08}, {\"seed\": 99433, \"w_rel_l2\": 2e-08, \"obj_abs_err\": 6e-09, \"field_rel_l2\": 1e-08}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5352.62,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
