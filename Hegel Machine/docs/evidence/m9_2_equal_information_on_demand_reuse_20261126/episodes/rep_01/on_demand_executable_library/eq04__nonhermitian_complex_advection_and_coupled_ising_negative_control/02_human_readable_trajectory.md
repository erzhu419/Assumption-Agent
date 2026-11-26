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
| **Turns Used / Max Turns** | `14 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `359,026` (`343,965` / `114,304` / `15,061`) |
| **Total Episode Wall Time (s)** | `485.22s` |

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

### Turn 1 — `execute_in_remote_docker` (`14914 ms` LLM)
- **Rationale**: Inspect the public validator to identify the actual input and output conventions before implementing the stencil and enumerating all 2,048 supports. Check whether a reusable primitive meets the complex coupled objective.
- **Alternatives Weighed**: `SPD iteration or independent coefficient thresholding` (rejected: The PDE operator is non-Hermitian and co-activation costs are non-separable.)

```bash
ls -l /holdout_task; sed -n '1,240p' /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'certified box L0 exact subset ridge pairwise' || true
```
- **Container Exit Code**: `0` (`396 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
autogenous_prim_ridge_profiled_matching@dfdf7838 | fit_ridge_nuisance(design: np.ndarray, values: np.ndarray, *, ridge: float | np.ndarray=1e-06) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and retur | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
autogenous_prim_ridge_profiled_matching@dfdf7838 | profile_candidate_amplitudes(residual_dot: np.ndarray, candidate_energy: np.ndarray, nuisance_cross: np.ndarray, whitener: np.ndarray, *, rtol: float=1e-12) -> tuple[np.ndarray, np.ndarray] | Profile residual inner products (m,), raw energies (m,), and nuisance cross-pr | REQ: ridge>=0; whitener must match SVD Gram of same weighted design | --describe autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes
```
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task': No such file or directory
sed: can't read /holdout_task/validate_public_schema.py: No such file or directory
```

### Turn 2 — `execute_in_remote_docker` (`14484 ms` LLM)
- **Rationale**: Check likely mount locations for the validator or fixtures without reading bulk data. If neither is mounted, implement the supplied contract at the required path and verify it against an independent stencil and exhaustive reference.
- **Alternatives Weighed**: `Use boxed real L0 certification` (rejected: The available primitive requires a real boxed model and does not capture a single activation cost per complex coefficient or pairwise co-activation costs.)

```bash
pwd; ls -ld /holdout_task /pretrain_task /workspace /app 2>/dev/null; for d in /pretrain_task /holdout_task /workspace /app; do if test -d "$d"; then find "$d" -maxdepth 3 -type f -printf '%p %s bytes\n' | head -80; fi; done; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
```
- **Container Exit Code**: `0` (`382 ms`, timed_out=`False`)
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
```

### Turn 3 — `execute_in_remote_docker` (`45831 ms` LLM)
- **Rationale**: Implement the stated stencil and enumerate all supports without assuming independent activations. Next compare against a row-by-row stencil and uncompressed exhaustive fits to detect indexing, sign, complex-conjugation, or penalty errors.
- **Alternatives Weighed**: `Solve ridge supports via normal equations` (rejected: QR compression avoids squaring the condition number while keeping each support fit at most 11 by 11.)

```bash
set -e
mkdir -p /holdout_task
cat > /holdout_task/nonhermitian_advection_coupled_select.py <<'PY'
"""Complex upwind PDE inversion with exhaustive pairwise-penalized selection."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _solve_fields(problem, basis):
    m, nx, ny = basis.shape
    n = nx * ny
    hx = problem.get('Lx', 1.0) / (nx + 1)
    hy = problem.get('Ly', 1.0) / (ny + 1)
    dx, dy = problem['nu'] / hx**2, problem['nu'] / hy**2
    vx = np.asarray(problem['vx'], dtype=np.float64).ravel() / hx
    vy = np.asarray(problem['vy'], dtype=np.float64).ravel() / hy
    ids = np.arange(n)
    i, j = ids // ny, ids % ny
    diagonal = (2 * (dx + dy) + np.abs(vx) + np.abs(vy)
                + np.asarray(problem['sigma']).ravel() + 1j * problem['omega'])
    rows, cols, values = [ids], [ids], [diagonal]
    for keep, step, coefficient in (
        (i > 0, -ny, -dx - np.maximum(vx, 0)),
        (i < nx - 1, ny, -dx + np.minimum(vx, 0)),
        (j > 0, -1, -dy - np.maximum(vy, 0)),
        (j < ny - 1, 1, -dy + np.minimum(vy, 0)),
    ):
        rows.append(ids[keep])
        cols.append(ids[keep] + step)
        values.append(coefficient[keep])
    operator = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n), dtype=np.complex128,
    ).tocsc()
    return splu(operator).solve(basis.reshape(m, n).T)


def solve_nonhermitian_coupled_inversion(problem: dict) -> dict:
    """Return the best complex ridge support, its objective and field magnitude."""
    basis = np.asarray(problem['basis'], dtype=np.complex128)
    m, nx, ny = basis.shape
    fields = _solve_fields(problem, basis)
    sensors = np.asarray(problem['sensor_ij'], dtype=np.intp)
    design = fields[sensors[:, 0] * ny + sensors[:, 1], :]
    observed = (np.asarray(problem['y_obs_re'], dtype=np.float64)
                + 1j * np.asarray(problem['y_obs_im'], dtype=np.float64))
    lam0, lam2 = problem['lam0'], problem['lam2']
    pairs = [(1 << int(u) | 1 << int(v), gamma)
             for u, v, gamma in problem['pair_penalties']]

    # Compress the augmented ridge system once. Its orthogonal residual
    # is independent of the support and need not enter the comparison.
    augmented = np.vstack((design, np.sqrt(lam2) * np.eye(m)))
    q, r = np.linalg.qr(augmented, mode='reduced')
    target = q[:len(observed)].conj().T @ observed
    best = 0.5 * np.vdot(target, target).real
    weights = np.zeros(m, dtype=np.complex128)
    for mask in range(1, 1 << m):
        support = [k for k in range(m) if mask & (1 << k)]
        subdesign = r[:, support]
        fitted = np.linalg.lstsq(subdesign, target, rcond=0.0)[0]
        active = sum(1 << k for k, w in zip(support, fitted) if abs(w) > 1e-8)
        penalty = lam0 * active.bit_count() + sum(
            gamma for bits, gamma in pairs if (active & bits) == bits
        )
        residual = subdesign @ fitted - target
        value = 0.5 * np.vdot(residual, residual).real + penalty
        if value < best:
            best = value
            weights.fill(0)
            weights[support] = fitted

    residual = design @ weights - observed
    active = sum(1 << k for k, w in enumerate(weights) if abs(w) > 1e-8)
    objective = (0.5 * np.vdot(residual, residual).real
                 + 0.5 * lam2 * np.vdot(weights, weights).real
                 + lam0 * active.bit_count()
                 + sum(gamma for bits, gamma in pairs if (active & bits) == bits))
    return {
        'weights_real': weights.real.copy(),
        'weights_imag': weights.imag.copy(),
        'objective': float(objective),
        'predicted_field_abs': np.abs(fields @ weights).reshape(nx, ny),
    }
PY
```
- **Container Exit Code**: `0` (`363 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/nonhermitian_advection_coupled_select.py']`
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f6d49365132ec584 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f6d49365 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`50355 ms` LLM)
- **Rationale**: Compare the isolated target against a row-wise stencil and uncompressed exhaustive fits. A stencil discrepancy would trigger an indexing or sign correction; objective or field discrepancies with matching stencils would isolate support fitting or penalty accounting.

```bash
set -e
if test -f /holdout_task/validate_public_schema.py && test -f /holdout_task/public_eq04_cases.pkl; then OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py; fi
cat > /tmp/verify_eq04.py <<'PY'
import importlib.util
import time
from itertools import combinations
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

path = '/holdout_task/nonhermitian_advection_coupled_select.py'
spec = importlib.util.spec_from_file_location('eq04_candidate', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def reference_fields(p):
    a = lil_matrix((1024, 1024), dtype=np.complex128)
    hx, hy = p['Lx']/33, p['Ly']/33
    for i in range(32):
        for j in range(32):
            k = i*32 + j
            a[k,k] = 2*p['nu']/hx**2 + 2*p['nu']/hy**2 + p['sigma'][i,j] + 1j*p['omega']
            for h, v, loc, stride in [(hx,p['vx'][i,j],i,32),(hy,p['vy'][i,j],j,1)]:
                d = p['nu']/h**2
                if loc > 0: a[k,k-stride] = -d
                if loc < 31: a[k,k+stride] = -d
                a[k,k] += abs(v)/h
                if v >= 0 and loc > 0: a[k,k-stride] -= v/h
                if v < 0 and loc < 31: a[k,k+stride] += v/h
    rhs = p['basis'].reshape(11,1024).T
    return spsolve(a.tocsc(), rhs)

def score(g,y,w,p):
    active = np.abs(w)>1e-8
    return 0.5*np.linalg.norm(g@w-y)**2 + 0.5*p['lam2']*np.linalg.norm(w)**2 + p['lam0']*active.sum() + sum(c for i,j,c in p['pair_penalties'] if active[i] and active[j])

def brute(g,y,p):
    w = np.zeros(11, complex)
    best = score(g,y,w,p)
    winner = w.copy()
    for n in range(1,12):
        for s in combinations(range(11), n):
            columns = list(s)
            b = np.vstack((g[:,columns], np.sqrt(p['lam2'])*np.eye(n)))
            w[columns] = np.linalg.lstsq(b, np.r_[y, np.zeros(n)], rcond=None)[0]
            value = score(g,y,w,p)
            if value < best: best, winner = value, w.copy()
            w[columns] = 0
    return best,winner

rng = np.random.default_rng(940)
worst = np.zeros(4)
for name, nu, freq, scale, k in [('coupled',.02,1.7,1,79),('fast_boundary',.00003,-19,9,11),('reaction',.0003,.07,0,42),('collinear',.03,-.8,1,15),('null',.02,3.8,2,23)]:
    v = rng.normal(size=(2,32,32))*scale
    p = dict(Lx=.75,Ly=2.25,nu=nu,omega=freq,vx=v[0],vy=v[1],sigma=rng.uniform(-.3,.7,(32,32)),basis=rng.normal(size=(11,32,32)),lam0=1e-4,lam2=1e-5)
    if name == 'collinear':
        p['basis'][1] = p['basis'][0] + 1e-8*p['basis'][1]
        p['lam0'] = 1e-7
    f = reference_fields(p)
    ids = np.r_[0,31,992,1023, rng.choice(np.setdiff1d(np.arange(1024), [0,31,992,1023]), k-4, replace=False)]
    p['sensor_ij'] = np.column_stack((ids//32,ids%32))
    g = f[ids]
    truth = np.zeros(11, complex)
    truth[[0,1,6]] = [1.1+2j,-.4+1.1j,1.1-.7j]
    y = np.zeros(k,complex) if name == 'null' else g@truth
    p['y_obs_re'], p['y_obs_im'] = y.real,y.imag
    p['pair_penalties'] = [[0,1,10.0],[6,8,.2],[4,2,.1]] if name != 'collinear' else [[6,8,.01],[4,2,.01]]
    if name == 'coupled':
        unpenalized = brute(g,y,dict(p,pair_penalties=[]))[1]
        assert np.all(np.abs(unpenalized[:2]) > 1e-8)
    expected,wref = brute(g,y,p)
    t0 = time.perf_counter()
    got = mod.solve_nonhermitian_coupled_inversion(p)
    elapsed = time.perf_counter()-t0
    assert set(got) == {'weights_real','weights_imag','objective','predicted_field_abs'}
    assert isinstance(got['objective'], float)
    assert all(got[key].shape == (11,) and got[key].dtype == np.float64 for key in ['weights_real','weights_imag'])
    assert got['predicted_field_abs'].shape == (32,32) and got['predicted_field_abs'].dtype == np.float64
    w = got['weights_real']+1j*got['weights_imag']
    if name == 'null': assert not w.any()
    if name == 'coupled': assert not np.all(np.abs(w[:2]) > 1e-8)
    pred = np.abs(f@wref).reshape(32,32)
    ferr = np.linalg.norm(got['predicted_field_abs']-pred)/(np.linalg.norm(pred) or 1.)
    internal = mod._solve_fields(p,np.asarray(p['basis'],complex))
    pde_err = np.linalg.norm(internal-f)/np.linalg.norm(f)
    errors = np.array([abs(expected-got['objective']),ferr,abs(score(g,y,w,p)-got['objective']),pde_err])
    assert np.all(errors <= [1e-6,1e-5,1e-6,1e-11]), (name, errors)
    worst = np.maximum(worst, errors)
    print(name, 'objective/field/score/stencil errors',errors, 'seconds', round(elapsed,3))
print('Worst objective/field/score/stencil errors:', worst)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_eq04.py
```
- **Container Exit Code**: `0` (`1516 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
coupled objective/field/score/stencil errors [0.00000000e+00 1.15577677e-15 0.00000000e+00 9.33160240e-16] seconds 0.051
fast_boundary objective/field/score/stencil errors [2.16840434e-19 1.47627502e-15 0.00000000e+00 1.32730869e-15] seconds 0.051
reaction objective/field/score/stencil errors [0.00000000e+00 4.42568792e-16 0.00000000e+00 0.00000000e+00] seconds 0.052
collinear objective/field/score/stencil errors [0.00000000e+00 8.24769077e-16 6.77626358e-21 8.32826276e-16] seconds 0.049
null objective/field/score/stencil errors [0.00000000e+00 0.00000000e+00 0.00000000e+00 7.13395922e-16] seconds 0.054
Worst objective/field/score/stencil errors: [2.16840434e-19 1.47627502e-15 6.77626358e-21 1.32730869e-15]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f6d49365132ec584 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f6d49365 (directory).
```

### Turn 5 — `submit_completed_task` (`14231 ms` LLM)
- **Rationale**: The saved artifact passed isolated verification against an independently assembled stencil and uncompressed exhaustive fits, including a case where the pairwise penalty changed the selected support.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`13` turns remaining; artifact status={'/holdout_task/nonhermitian_advection_coupled_select.py': 'FILE:3658'}): Do NOT lock prematurely with `13` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/nonhermitian_advection_coupled_select.py`. Five independent stress cases passed; the worst absolute objective error was 2.17e-19 and the worst relative field error was 1.48e-15. Public fixtures were unavailable.`

### Turn 6 — `execute_in_remote_docker` (`39506 ms` LLM)
- **Rationale**: Use the public cases if mounted. Otherwise isolate two failure modes not covered by the original sweep: a phase or edge-order error would require correcting the complex fit or pair accounting, while a feasible inactive coefficient that beats the returned result would distinguish literal threshold minimization from the specified ridge-on-support reference.
- **Alternatives Weighed**: `Audit batch centering, relay modes, or unrelated drift operators` (rejected: This problem has no cohorts, relay states, or fitted drift law. Its relevant algebra is complex phase invariance, undirected pair charges, and the activation threshold.)

```bash
set -e
if test -f /holdout_task/validate_public_schema.py && test -f /holdout_task/public_eq04_cases.pkl; then
  cd /holdout_task
  OPENBLAS_NUM_THREADS=1 python3 validate_public_schema.py
else
  echo 'Public validator and fixtures not available together; using a manufactured solution.'
fi
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import numpy as np
p = '/holdout_task/nonhermitian_advection_coupled_select.py'
s = importlib.util.spec_from_file_location('eq04', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
solve = m.solve_nonhermitian_coupled_inversion

# Identical response columns and a second independent response. Only one
# duplicate incurs co-activation cost, so the penalty must break the tie.
b = np.zeros((11,32,32))
b[0,0,31] = b[1,0,31] = b[2,31,0] = 1.0
z = np.zeros((32,32))
p = dict(nu=1e-4, vx=z, vy=z, sigma=z+2.0, omega=-3.0, basis=b,
         Lx=1.3, Ly=.7, sensor_ij=[[0,31],[31,0]], y_obs_re=[1.,2.],
         y_obs_im=[2.,-1.], lam0=.02, lam2=.001, pair_penalties=[[0,2,10.]])
a = solve(p)
y = np.array(p['y_obs_re']) + 1j*np.array(p['y_obs_im'])
phase = np.exp(1.37j)
c = solve(dict(p, y_obs_re=(phase*y).real, y_obs_im=(phase*y).imag,
               pair_penalties=[[2,0,10.]]))
w = a['weights_real']+1j*a['weights_imag']
w2 = c['weights_real']+1j*c['weights_imag']
e = [abs(a['objective']-c['objective']), np.linalg.norm(w2-phase*w),
     np.linalg.norm(a['predicted_field_abs']-c['predicted_field_abs'])/np.linalg.norm(a['predicted_field_abs'])]
assert np.array_equal(np.flatnonzero(abs(w)>1e-8), [1,2]), w
assert np.all(np.asarray(e)<1e-10), e
print('Phase/edge reversal objective, weights, relative field errors:', e)

# Probe whether the declared threshold is merely a support-reporting
# tolerance or changes the continuous optimum. Use a scaled legal basis.
b = np.zeros_like(b)
b[0,0,31] = 1e8
p = dict(p, basis=b, sensor_ij=[[0,31]], y_obs_re=[.24], y_obs_im=[0.],
         lam0=1., pair_penalties=[])
g = m._solve_fields(p, b.astype(complex))[31,0]
y = 1.5e-8*abs(g)
p['y_obs_re'] = [y]
a = solve(p)
w = 1e-8*np.conj(g)/abs(g)
# Move a few ulps inside the inactive disk so rounding cannot charge lam0.
w *= 1-1e-14
assert abs(w) < 1e-8
j = .5*abs(g*w-y)**2 + .5*p['lam2']*abs(w)**2
print('Literal activation threshold: returned J=', a['objective'], 'feasible inactive J=', j, 'gap=', a['objective']-j)
PY
```
- **Container Exit Code**: `0` (`946 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public validator and fixtures not available together; using a manufactured solution.
Phase/edge reversal objective, weights, relative field errors: [0.0, 2.648876738051731e-15, 3.1855465527785857e-16]
Literal activation threshold: returned J= 0.07193512377265956 feasible inactive J= 0.007992791530295822 gap= 0.06394233224236374

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f6d49365132ec584 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f6d49365 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`58917 ms` LLM)
- **Rationale**: The new bound skips a disk fit only when its best possible linear improvement cannot beat the incumbent. Rerun the original independent sweep to detect regressions in ordinary field predictions; then verify the threshold counterexample against its analytic optimum.
- **Alternatives Weighed**: `Treat 1e-8 as a reporting tolerance only` (rejected: The supported scaled-basis example violates the stated objective tolerance by more than four orders of magnitude.)
- **ATMS NOGOOD DECLARED**: Refuted `Exhaustive ridge fits with omitted coordinates fixed to zero minimize the literal thresholded objective.` | Witness: `A permitted scaled basis yielded J=0.0719351, but an uncharged coefficient inside the 1e-8 disk attained J=0.00799279.` | Blanket Diagnosis: `The activation indicator charges magnitudes strictly above the threshold; zero-only support enumeration omits feasible uncharged reductions in sensor residual.` | Sublated Pivot: `Keep the sparse LU and support enumeration, and fit inactive coordinates on closed complex disks after eliminating the free ridge coordinates.`

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/nonhermitian_advection_coupled_select.py')
s = p.read_text().replace('from scipy.sparse import coo_matrix', 'from scipy.optimize import minimize\nfrom scipy.sparse import coo_matrix')
pos = s.index('\ndef solve_nonhermitian_coupled_inversion')
s = s[:pos] + '''

def _fit_inactive(design, target):
    """Minimize the projected residual on the product of unit complex disks."""
    k = design.shape[1]
    gram = design.conj().T @ design
    rhs = design.conj().T @ target
    energy = gram.diagonal().real
    u = np.zeros(k, dtype=np.complex128)
    # The linearization gap bounds the remaining objective improvement.
    tol = 2e-13 * max(1.0, float(np.vdot(target, target).real))
    for _ in range(64):
        for j in range(k):
            if energy[j] > 0:
                dot = rhs[j] - gram[j] @ u + energy[j] * u[j]
                u[j] = dot / max(energy[j], abs(dot))
        gradient = gram @ u - rhs
        gap = np.vdot(gradient, u).real + np.abs(gradient).sum()
        if gap <= tol:
            break
    else:
        # Refine correlated disk coordinates together when sweeps stall.
        scale = max(np.linalg.norm(gram), np.linalg.norm(rhs))

        def objective(x):
            z = x[:k] + 1j * x[k:]
            h = gram @ z
            value = np.vdot(z, 0.5 * h - rhs).real / scale
            grad = (h - rhs) / scale
            return value, np.r_[grad.real, grad.imag]

        eye = np.eye(k)
        result = minimize(objective, np.r_[u.real, u.imag], jac=True,
                          method='SLSQP', constraints={
                              'type': 'ineq',
                              'fun': lambda x: 1 - x[:k]**2 - x[k:]**2,
                              'jac': lambda x: -2 * np.hstack((eye * x[:k], eye * x[k:])),
                          }, options={'ftol': 1e-13, 'maxiter': 150})
        candidate = result.x[:k] + 1j * result.x[k:]
        candidate /= np.maximum(1, np.abs(candidate))
        if objective(np.r_[candidate.real, candidate.imag])[0] < objective(np.r_[u.real, u.imag])[0]:
            u = candidate
    # Keep the projected boundary on the uncharged side in floating point.
    return ((1 - 4 * np.finfo(float).eps) * 1e-8) * u / np.maximum(1, np.abs(u))
''' + s[pos:]
a, b = s.index('    best = 0.5'), s.index('    residual = design @ weights - observed')
s = s[:a] + '''    threshold = 1e-8

    def penalty(mask):
        return lam0 * mask.bit_count() + sum(
            gamma for bits, gamma in pairs if (mask & bits) == bits
        )

    def active_mask(w):
        return sum(1 << k for k in np.flatnonzero(np.abs(w) > threshold).tolist())

    best = 0.5 * np.vdot(target, target).real
    weights = np.zeros(m, dtype=np.complex128)
    for mask in range(1 << m):
        support = [k for k in range(m) if mask & (1 << k)]
        inactive = [k for k in range(m) if not mask & (1 << k)]
        charge = penalty(mask)
        if charge >= best:
            continue
        qa, ra = np.linalg.qr(r[:, support], mode='reduced')
        projected_target = qa.conj().T @ target
        perpendicular = target - qa @ projected_target
        fitted = np.zeros(m, dtype=np.complex128)
        fitted[support] = np.linalg.solve(ra, projected_target)
        base = 0.5 * np.vdot(perpendicular, perpendicular).real
        value = base + penalty(active_mask(fitted))
        if value < best:
            best, weights = value, fitted.copy()
        if not inactive:
            continue
        # Eliminate free coordinates first. Discarding their activation
        # lower limits is valid: any that become inactive only reduce charge.
        other = r[:, inactive]
        cross = qa.conj().T @ other
        projected = other - qa @ cross
        # The maximum linear improvement over the inactive disks is a
        # bound because the remaining quadratic term is nonnegative.
        gain = threshold * np.abs(projected.conj().T @ perpendicular).sum()
        if base - gain + charge >= best:
            continue
        fitted[inactive] = _fit_inactive(threshold * projected, perpendicular)
        fitted[support] = np.linalg.solve(ra, projected_target - cross @ fitted[inactive])
        err = r @ fitted - target
        value = 0.5 * np.vdot(err, err).real + penalty(active_mask(fitted))
        if value < best:
            best, weights = value, fitted

''' + s[b:]
s = s.replace('    active = sum(1 << k for k, w in enumerate(weights) if abs(w) > 1e-8)\n', '').replace('                 + lam0 * active.bit_count()\n                 + sum(gamma for bits, gamma in pairs if (active & bits) == bits))', '                 + penalty(active_mask(weights)))')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_eq04.py
```
- **Container Exit Code**: `0` (`2285 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/nonhermitian_advection_coupled_select.py']`
- **Stdout (tail)**:
```text
coupled objective/field/score/stencil errors [1.71973726e-09 5.52949343e-09 6.93889390e-18 9.33160240e-16] seconds 0.061
fast_boundary objective/field/score/stencil errors [9.64748810e-12 3.71521492e-09 0.00000000e+00 1.32730869e-15] seconds 0.081
reaction objective/field/score/stencil errors [4.24549285e-13 8.46507276e-09 0.00000000e+00 0.00000000e+00] seconds 0.117
collinear objective/field/score/stencil errors [9.55290595e-13 1.11421448e-08 6.77626358e-21 8.32826276e-16] seconds 0.044
null objective/field/score/stencil errors [0.00000000e+00 0.00000000e+00 0.00000000e+00 7.13395922e-16] seconds 0.007
Worst objective/field/score/stencil errors: [1.71973726e-09 1.11421448e-08 6.93889390e-18 1.32730869e-15]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=322fbd14c0e4b24f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_322fbd14 (directory).
```

### Turn 8 — `execute_in_remote_docker` (`33628 ms` LLM)
- **Rationale**: Test the exact artifact against independently constructed optima for the original threshold failure and a correlated two-coordinate case. If either misses tolerance, the scalar case isolates boundary handling and the correlated case isolates the disk optimizer.

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve

path = '/holdout_task/nonhermitian_advection_coupled_select.py'
spec = importlib.util.spec_from_file_location('candidate', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
n, nu, lx, ly, omega = 32, 1e-4, 1.3, .7, -3.
z = np.zeros((n,n))
t = diags([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1,0,1])
a = nu*(kron(t,eye(n))/(lx/33)**2 + kron(eye(n),t)/(ly/33)**2) + (2+1j*omega)*eye(n*n)
p = dict(nu=nu, vx=z, vy=z, sigma=z+2, omega=omega, Lx=lx, Ly=ly,
         lam2=.001, lam0=1., pair_penalties=[])
threshold = 1e-8

def test(name, basis, ids, y, expected):
    fields = spsolve(a.tocsc(), basis.reshape(11,-1).T.astype(complex))
    g = fields[ids]
    y = y(g)
    wref = expected(g)
    ref_field = np.abs(fields @ wref).reshape(n,n)
    jref = .5*np.linalg.norm(g@wref-y)**2 + .5*p['lam2']*np.linalg.norm(wref)**2
    assert .5*np.linalg.norm(y)**2 < p['lam0'], 'A charged support must be ruled out.'
    out = mod.solve_nonhermitian_coupled_inversion(dict(p, basis=basis,
        sensor_ij=np.column_stack((ids//n,ids%n)), y_obs_re=y.real, y_obs_im=y.imag))
    w = out['weights_real'] + 1j*out['weights_imag']
    assert np.max(np.abs(w)) <= threshold, w
    obj = abs(out['objective']-jref)
    field = np.linalg.norm(out['predicted_field_abs']-ref_field)/np.linalg.norm(ref_field)
    assert obj <= 1e-6 and field <= 1e-5, (obj, field)
    print(name, 'objective', out['objective'], 'absolute error', obj, 'relative field error', field)

b = np.zeros((11,n,n)); b[0,0,31] = 1e8
ids = np.array([31]); unit = np.zeros(11,complex)
def scalar_opt(g):
    w = unit.copy(); w[0] = threshold*np.conj(g[0,0])/abs(g[0,0]); return w

test('scalar boundary', b, ids, lambda g: np.array([1.5*threshold*abs(g[0,0])]), scalar_opt)

# Two correlated columns with one disk-bound coordinate and one interior.
# Prescribe a residual so the first gradient points inward and the second is zero.
b[1,0,31] = 6e7; b[1,31,0] = 7e7
ids = np.array([31, 992]); wref = unit.copy()
wref[:2] = threshold * np.array([np.exp(.7j), .48*np.exp(-1.2j)])
def correlated_target(g):
    h = g[:,:2]; normal = np.array([3e6*np.exp(.7j), 0.])
    return h@wref[:2] + np.linalg.solve(h.conj().T, p['lam2']*wref[:2] + normal)

test('correlated boundary and interior', b, ids, correlated_target, lambda g: wref)
PY
```
- **Container Exit Code**: `0` (`860 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
scalar boundary objective 0.007992791530295534 absolute error 2.949029909160572e-17 relative field error 8.793357063424252e-16
correlated boundary and interior objective 0.012208066511250202 absolute error 2.0016974189296377e-14 relative field error 3.818751790336281e-07

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=322fbd14c0e4b24f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_322fbd14 (directory).
```

### Turn 9 — `execute_in_remote_docker` (`27241 ms` LLM)
- **Rationale**: The analytic boundary tests resolve the omitted-disk counterexample. Check whether the disk solver’s absolute stopping floor can accept a poor relative field for weak observations. If it does, return the unconstrained ridge solution directly when it lies inside all inactive disks.

```bash
set -e
cat > /tmp/verify_eq04_weak.py <<'PY'
import importlib.util
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve

path = '/holdout_task/nonhermitian_advection_coupled_select.py'
spec = importlib.util.spec_from_file_location('eq04', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
n = 32
zero = np.zeros((n,n))
basis = np.zeros((11,n,n))
basis[0,1,1] = 1
basis[1,1,1] = 1
basis[1,30,30] = .3
nu, lam2 = 1e-4, 1e-5
D = diags([-np.ones(31),2*np.ones(32),-np.ones(31)],[-1,0,1])
A = (nu*33**2*(kron(D,eye(n))+kron(eye(n),D)) + (2+1j)*eye(n*n)).tocsc()
F = spsolve(A,basis.reshape(11,-1).T.astype(complex))
sensors = np.array([[1,1],[30,30]])
G = F[sensors[:,0]*n+sensors[:,1],:2]
wref = 1e-9*np.array([1+1j,2-3j])
y = G@wref + np.linalg.solve(G.conj().T, lam2*wref)
out = mod.solve_nonhermitian_coupled_inversion(dict(nu=nu,vx=zero,vy=zero,sigma=zero+2,omega=1.,basis=basis,sensor_ij=sensors,lam0=.01,lam2=lam2,pair_penalties=[],y_obs_re=y.real,y_obs_im=y.imag))
expected = abs(F[:,:2]@wref).reshape(n,n)
relative = np.linalg.norm(out['predicted_field_abs']-expected)/np.linalg.norm(expected)
print('Weak correlated interior optimum: relative field error', relative, flush=True)
assert relative <= 1e-5, relative
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_eq04_weak.py
```
- **Container Exit Code**: `1` (`755 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Weak correlated interior optimum: relative field error 0.2736425043835616

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=322fbd14c0e4b24f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_322fbd14 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/verify_eq04_weak.py", line 28, in <module>
    assert relative <= 1e-5, relative
           ^^^^^^^^^^^^^^^^
AssertionError: 0.2736425043835616
```

### Turn 10 — `execute_in_remote_docker` (`35626 ms` LLM)
- **Rationale**: The weak-signal miss is numerical, not a new structural falsification. Verify the fix against the failed interior case, the ordinary exhaustive reference, and analytic disk-boundary optima before submitting.
- **Alternatives Weighed**: `Only lower the absolute stopping tolerance` (rejected: A fixed floor can still hide large relative field errors as the signal weakens; a feasible unconstrained fit already solves the interior problem.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/nonhermitian_advection_coupled_select.py')
s = p.read_text()
a = s.index('    gram = design.conj().T @ design')
b = s.index('    for _ in range(64):', a)
s = s[:a] + '''    # An interior optimum is an ordinary ridge least-squares fit. Solve
    # directly instead of relying on an objective-gap test at weak signal.
    radius = (1 - 4 * np.finfo(float).eps) * 1e-8
    u = np.linalg.lstsq(design, target, rcond=0.0)[0]
    if np.all(np.abs(u) <= 1):
        return radius * u
    gram = design.conj().T @ design
    rhs = design.conj().T @ target
    energy = gram.diagonal().real
    u /= np.maximum(1, np.abs(u))
    # Only the target component seen by the disk design affects progress.
    tol = 2e-14 * np.sum(np.abs(rhs))
''' + s[b:]
s = s.replace('return ((1 - 4 * np.finfo(float).eps) * 1e-8) * u / np.maximum(1, np.abs(u))', 'return radius * u / np.maximum(1, np.abs(u))')
p.write_text(s)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_eq04.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_eq04_weak.py
sed '/^wref =/,$d' /tmp/verify_eq04_weak.py > /tmp/verify_eq04_disks.py
cat >> /tmp/verify_eq04_disks.py <<'PY'
scale = 1e8
F = F * scale
G = G * scale
for name, m, wref, normal in [
    ('scalar boundary', 1, 1e-8*np.array([np.exp(.7j)]), np.array([3e6*np.exp(.7j)])),
    ('correlated boundary/interior', 2, 1e-8*np.array([np.exp(.7j), .48*np.exp(-1.2j)]), np.array([3e6*np.exp(.7j), 0.])),
]:
    h = G[:m,:m]
    y = h@wref + np.linalg.solve(h.conj().T, lam2*wref + normal)
    b = np.zeros_like(basis)
    b[:m] = basis[:m]*scale
    lam0 = 1.
    assert .5*np.linalg.norm(y)**2 < lam0
    out = mod.solve_nonhermitian_coupled_inversion(dict(nu=nu,vx=zero,vy=zero,sigma=zero+2,omega=1.,basis=b,sensor_ij=sensors[:m],lam0=lam0,lam2=lam2,pair_penalties=[],y_obs_re=y.real,y_obs_im=y.imag))
    w = out['weights_real']+1j*out['weights_imag']
    assert np.max(np.abs(w)) <= 1e-8
    j = .5*np.linalg.norm(h@wref-y)**2 + .5*lam2*np.linalg.norm(wref)**2
    f = np.abs(F[:,:m]@wref).reshape(n,n)
    obj = abs(out['objective']-j)
    rel = np.linalg.norm(out['predicted_field_abs']-f)/np.linalg.norm(f)
    print(name, 'objective error', obj, 'relative field error', rel)
    assert obj <= 1e-6 and rel <= 1e-5
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_eq04_disks.py
```
- **Container Exit Code**: `0` (`2216 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/nonhermitian_advection_coupled_select.py']`
- **Stdout (tail)**:
```text
coupled objective/field/score/stencil errors [1.71973726e-09 5.52949344e-09 6.93889390e-18 9.33160240e-16] seconds 0.044
fast_boundary objective/field/score/stencil errors [9.64748810e-12 3.71521494e-09 4.33680869e-19 1.32730869e-15] seconds 0.048
reaction objective/field/score/stencil errors [4.35207426e-13 8.24608989e-09 1.77635684e-15 0.00000000e+00] seconds 0.07
collinear objective/field/score/stencil errors [9.55290602e-13 1.11421469e-08 6.77626358e-21 8.32826276e-16] seconds 0.027
null objective/field/score/stencil errors [0.00000000e+00 0.00000000e+00 0.00000000e+00 7.13395922e-16] seconds 0.005
Worst objective/field/score/stencil errors: [1.71973726e-09 1.11421469e-08 1.77635684e-15 1.32730869e-15]
Weak correlated interior optimum: relative field error 2.107217550613488e-16
scalar boundary objective error 3.859759734048396e-17 relative field error 1.3078386663136796e-15
correlated boundary/interior objective error 2.588207426157396e-15 relative field error 9.451659834545029e-08

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=807b63a7200827b6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_807b63a7 (directory).
```

### Turn 11 — `submit_completed_task` (`13646 ms` LLM)
- **Rationale**: The final artifact passed isolated verification against an independent stencil, exhaustive ridge references, and analytic threshold optima. The direct interior fit reduced the weak-signal relative field error from 0.274 to 2.11e-16 while preserving the other tolerance margins.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/nonhermitian_advection_coupled_select.py': 'FILE:7026'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/nonhermitian_advection_coupled_select.py`. Eight independent stress cases passed in isolated Python. The worst absolute objective error was 1.72e-9, and the worst relative field error was 9.45e-8, both within tolerance. Public fixtures were unavailable.`

### Turn 12 — `execute_in_remote_docker` (`32153 ms` LLM)
- **Rationale**: An objective can pass tolerance while the field fails when two nearly collinear basis responses cancel. Compare against a known disk-constrained optimum in isolated Python. A miss would justify refining the constrained fit without changing the PDE or support enumeration.
- **Alternatives Weighed**: `Audit cohort confounding, relay states, or drift-law terms` (rejected: The PDE inversion has none of these inputs; near-collinear bases with opposing coefficients directly test its field tolerance.)

```bash
set -e
if test -f /holdout_task/validate_public_schema.py && test -f /holdout_task/public_eq04_cases.pkl; then
  (cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 validate_public_schema.py)
else
  printf '%s\n' 'Public cases are not mounted; running an analytic correlated-disk stress test.'
fi
sed '/^wref =/,$d' /tmp/verify_eq04_weak.py > /tmp/verify_eq04_correlated.py
cat >> /tmp/verify_eq04_correlated.py <<'PY'
# Manufacture the optimum by prescribing its residual to satisfy the disk
# optimality conditions. Opposing coefficients make the field more sensitive
# than the scalar objective to an inaccurate disk solve.
worst_obj = worst_field = 0.
for separation in (.03, .01, .003):
    b = basis.copy() * 1e8
    b[1,30,30] = separation * 1e8
    f = spsolve(A, b.reshape(11,-1).T.astype(complex))
    h = f[sensors[:,0]*n+sensors[:,1],:2]
    wref = 1e-8*np.exp(.7j)*np.array([1., -.95+.05j])
    expected = abs(f[:,:2]@wref).reshape(n,n)
    normal = np.array([30.*np.exp(.7j),0.])
    y = h@wref + np.linalg.solve(h.conj().T, lam2*wref+normal)
    assert .5*np.linalg.norm(y)**2 < .01
    out = mod.solve_nonhermitian_coupled_inversion(dict(nu=nu,vx=zero,vy=zero,sigma=zero+2,omega=1.,basis=b,sensor_ij=sensors,lam0=.01,lam2=lam2,pair_penalties=[],y_obs_re=y.real,y_obs_im=y.imag))
    expected_obj = .5*np.linalg.norm(h@wref-y)**2 + .5*lam2*np.linalg.norm(wref)**2
    err_obj = abs(out['objective']-expected_obj)
    err_field = np.linalg.norm(out['predicted_field_abs']-expected)/np.linalg.norm(expected)
    worst_obj, worst_field = max(worst_obj,err_obj), max(worst_field,err_field)
    print('basis separation:', separation, 'absolute objective error:', err_obj, 'relative field error:', err_field, flush=True)
assert worst_obj <= 1e-6 and worst_field <= 1e-5, (worst_obj,worst_field)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_eq04_correlated.py
```
- **Container Exit Code**: `1` (`811 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public cases are not mounted; running an analytic correlated-disk stress test.
basis separation: 0.03 absolute objective error: 2.8062102280861464e-22 relative field error: 2.5088497312989863e-15
basis separation: 0.01 absolute objective error: 3.2177325828312777e-22 relative field error: 5.098036427883712e-15
basis separation: 0.003 absolute objective error: 4.2470194637231566e-11 relative field error: 1.6135893210328374e-05

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=807b63a7200827b6 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v03_807b63a7 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/verify_eq04_correlated.py", line 42, in <module>
    assert worst_obj <= 1e-6 and worst_field <= 1e-5, (worst_obj,worst_field)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: (4.2470194637231566e-11, 1.6135893210328374e-05)
```

### Turn 13 — `execute_in_remote_docker` (`30387 ms` LLM)
- **Rationale**: Profile out interior coordinates and refine the boundary phases, accepting only a feasible step that lowers the residual. Rerun the failed cancellation case first, then check that the same artifact preserves ordinary, weak-signal, and boundary accuracy.
- **Alternatives Weighed**: `Tighten the generic optimizer’s objective tolerance alone` (rejected: Near-collinear cancellation can amplify field error even when the scalar objective has already converged.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/nonhermitian_advection_coupled_select.py')
s = p.read_text().replace('from scipy.optimize import minimize', 'from scipy.optimize import minimize, root')
pos = s.index('\ndef _fit_inactive')
s = s[:pos] + '''

def _polish_disk_fit(design, target, u):
    """Refine the active disk phases after profiling out interior coordinates."""
    boundary = np.abs(u) >= 1 - 1e-7
    if not np.any(boundary):
        return u
    free = design[:, ~boundary]
    q, r = np.linalg.qr(free, mode='reduced')
    edges = design[:, boundary]
    cross = q.conj().T @ edges
    center = q.conj().T @ target
    edges = edges - q @ cross
    target_perp = target - q @ center
    gram = edges.conj().T @ edges
    rhs = edges.conj().T @ target_perp

    if len(rhs) == 1:
        if rhs[0] == 0:
            return u
        z = rhs / np.abs(rhs)
    else:
        def phase_equations(theta):
            z = np.exp(1j * theta)
            grad = z.conj() * (gram @ z - rhs)
            jac = (z.conj()[:, None] * gram * z).real
            jac[np.diag_indices_from(jac)] -= grad.real
            return grad.imag, jac

        result = root(phase_equations, np.angle(u[boundary]), jac=True, tol=1e-12)
        z = np.exp(1j * result.x)

    candidate = u.copy()
    candidate[boundary] = z
    candidate[~boundary] = np.linalg.solve(r, center - cross @ z)
    if np.any(np.abs(candidate) > 1 + 8 * np.finfo(float).eps):
        return u
    candidate /= np.maximum(1, np.abs(candidate))
    step = design @ (candidate - u)
    improvement = np.vdot(design @ u - target + 0.5 * step, step).real
    return candidate if improvement < 0 else u
''' + s[pos:]
s = s.replace('    # Keep the projected boundary on the uncharged side in floating point.', '    u = _polish_disk_fit(design, target, u)\n    # Keep the projected boundary on the uncharged side in floating point.')
p.write_text(s)
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /tmp/verify_eq04_correlated.py
python3 -I /tmp/verify_eq04.py
python3 -I /tmp/verify_eq04_weak.py
python3 -I /tmp/verify_eq04_disks.py
```
- **Container Exit Code**: `0` (`2503 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/nonhermitian_advection_coupled_select.py']`
- **Stdout (tail)**:
```text
basis separation: 0.03 absolute objective error: 2.8062102280861464e-22 relative field error: 2.5088497312989863e-15
basis separation: 0.01 absolute objective error: 2.208572235516584e-22 relative field error: 2.296648512111951e-15
basis separation: 0.003 absolute objective error: 2.6469779601696886e-22 relative field error: 2.6988396003191084e-15
coupled objective/field/score/stencil errors [1.71973726e-09 5.52949343e-09 0.00000000e+00 9.33160240e-16] seconds 0.054
fast_boundary objective/field/score/stencil errors [9.64748810e-12 3.71521490e-09 0.00000000e+00 1.32730869e-15] seconds 0.051
reaction objective/field/score/stencil errors [4.35207426e-13 8.24608995e-09 1.77635684e-15 0.00000000e+00] seconds 0.084
collinear objective/field/score/stencil errors [9.55290602e-13 1.11421469e-08 6.77626358e-21 8.32826276e-16] seconds 0.035
null objective/field/score/stencil errors [0.00000000e+00 0.00000000e+00 0.00000000e+00 7.13395922e-16] seconds 0.006
Worst objective/field/score/stencil errors: [1.71973726e-09 1.11421469e-08 1.77635684e-15 1.32730869e-15]
Weak correlated interior optimum: relative field error 2.107217550613488e-16
scalar boundary objective error 3.859759734048396e-17 relative field error 1.3078386663136796e-15
correlated boundary/interior objective error 4.163336342344337e-17 relative field error 1.0528107612084873e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=cc20b9aa2e5a32c4 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v04_cc20b9aa (directory).
```

### Turn 14 — `submit_completed_task` (`14986 ms` LLM)
- **Rationale**: The final saved artifact passed clean-process verification. Phase refinement reduced the failing near-collinear case’s relative field error from 1.61e-5 to 2.70e-15 without regressing the ordinary, weak-signal, or threshold-boundary cases.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/nonhermitian_advection_coupled_select.py`. All 11 stress cases passed in isolated Python. The worst absolute objective error was 1.72e-9, and the worst relative field error was 1.11e-8, both within tolerance. Public fixtures were unavailable.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/nonhermitian_advection_coupled_select.py": "FILE:8492"
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
    "student_elapsed_sec": 0.3706,
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
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_weights_rel_l2\": 2e-08, \"max_objective_abs_err\": 7e-09, \"max_field_rel_l2\": 1e-08, \"student_elapsed_sec\": 0.3706, \"instance_diagnostics\": [{\"seed\": 99431, \"w_rel_l2\": 2e-08, \"obj_abs_err\": 6e-09, \"field_rel_l2\": 1e-08}, {\"seed\": 99432, \"w_rel_l2\": 2e-08, \"obj_abs_err\": 7e-09, \"field_rel_l2\": 1e-08}, {\"seed\": 99433, \"w_rel_l2\": 2e-08, \"obj_abs_err\": 6e-09, \"field_rel_l2\": 1e-08}]}\n__HEGEL_VERIFIER_AUTH_515a17dfe6c7fcc4431706df6eafa70a__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_weights_rel_l2\": 2e-08, \"max_objective_abs_err\": 7e-09, \"max_field_rel_l2\": 1e-08, \"student_elapsed_sec\": 0.3706, \"instance_diagnostics\": [{\"seed\": 99431, \"w_rel_l2\": 2e-08, \"obj_abs_err\": 6e-09, \"field_rel_l2\": 1e-08}, {\"seed\": 99432, \"w_rel_l2\": 2e-08, \"obj_abs_err\": 7e-09, \"field_rel_l2\": 1e-08}, {\"seed\": 99433, \"w_rel_l2\": 2e-08, \"obj_abs_err\": 6e-09, \"field_rel_l2\": 1e-08}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 6134.6,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
