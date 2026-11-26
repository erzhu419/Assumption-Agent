# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `eq02__coupled_panel_interferometry_cauchy_and_connected_box_l0`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M92-EqualInfo-4` |
| **Domain** | `Causally Coupled Multi-Center Panel Deconfounding, Interferometric Demodulation & Connected Box-L0 Certification` |
| **Wave / Wormhole Family** | `m92_equal_info` / `fam_sparse_cert_and_combinatorial_pruning` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`PASS (1.0)`** |
| **Failure / Outcome Taxonomy** | `PASSED_ALL_CHECKS` |
| **Turns Used / Max Turns** | `9 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `209,084` (`199,289` / `35,840` / `9,795`) |
| **Total Episode Wall Time (s)** | `318.89s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_06_robust_network_assimilation`**: No empirical nogoods were recorded, but the supplied solver addresses two concrete failure modes: raw least squares lets sparse spikes dominate the fit, and normalizing the drive to unit mean does not by itself identify valve transmission when unknown sensor gains can absorb its scale. The solver propagates the initial condition and each Fourier forcing basis vector through the same discrete dynamics, then eliminates sensor offsets and lower-bounded gains by weighted variable projection. It screens residuals at eight estimated noise scales, averages nested ridge-regularized Fourier fits on the same retained samples within each valve candidate, and flags final outliers beyond five residual MAD scales. The high-transmission regime is selected when the closed-minus-open Cauchy score is positive, but that gap is not a confidence measure: local forcing recovery needs independent projected sensitivities, and valve recovery needs distinguishable calibrated traces, which can fail when the observed initial-condition transient vanishes. [REGIME SPLIT GUARD `closed_minus_open_cauchy_score > 0` — Counter-Regime Warning: A nonpositive gap selects the low-transmission branch by convention. A tie or small gap is not evidence that the valve is closed. With zero observed homogeneous response, changing transmission can be exactly compensated by changing sensor gains if both gain sets meet the floor.] | [BIFURCATION `node_auto_08_split_sparse_ridge_certification` (`row_incidence_component_count`)]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps.
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`15450 ms` LLM)
- **Rationale**: Inspect the local validator to identify the exact leaf-bound and partition checks before choosing the search method. Check for reusable primitives; if none are available, implement against the validator directly. No study data is transferred.
- **Alternatives Weighed**: `Implement the solver before reading the public validator` (rejected: A mathematically valid bound may still fail the validator if it uses a different leaf encoding or numerical tolerance.)

```bash
ls -lh /holdout_task; sed -n '1,260p' /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'certified box L0 diagonal minorant branch bound Cauchy IRLS reference panel alignment'
```
- **Container Exit Code**: `0` (`499 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_robust_affine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar or scale array broadcastable to | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights | STATUS: APPLICABLE_SUBROUTINE(real_residual_array_positive_scale)
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel | STATUS: APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)
```
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task': No such file or directory
sed: can't read /holdout_task/validate_public_schema.py: No such file or directory
```

### Turn 2 — `execute_in_remote_docker` (`14664 ms` LLM)
- **Rationale**: Locate the public fixtures before relying on a certificate formula or claiming validation. Inspect the discovered primitives to identify which can be reused without changing the required leaf representation.
- **Alternatives Weighed**: `Assume the public study and validator are unavailable` (rejected: The initial snapshot names /pretrain_task; a targeted search can distinguish a different mount location from missing fixtures.)

```bash
pwd; ls -la / /pretrain_task; find /pretrain_task /app /workspace /tmp -maxdepth 4 \( -name 'validate_public_schema.py' -o -name 'public_eq02_studies.pkl' -o -name '*interferometry*' \) -print 2>/dev/null; for f in autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound autogenous_prim_box_ridge_bounds.gram_diagonal_minorant autogenous_prim_robust_affine_projection.robust_residual_weights autogenous_prim_reference_panel_balancing.align_reference_panel autogenous_prim_reference_panel_balancing.balanced_cell_weights autogenous_prim_reference_panel_balancing.weighted_column_fill; do PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe "$f"; done
```
- **Container Exit Code**: `0` (`628 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
up and an observed stratum must be present. If there are K such cells, each row in a cell of size m receives weight n/(K*m).",
        "Column fill requires a nonempty real matrix of shape (n, p) and finite, strictly positive row weights of shape (n,). Nonfinite entries are excluded from both numerator and denominator; a column with no finite observations returns zero. Numeric panel and fill arithmetic use float64 and assume intermediate sums, squares, and ratios are within usable float64 range."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "align_reference_panel": "APPLICABLE_SUBROUTINE(real_2d_values_and_controls_per_group)",
        "balanced_cell_weights": "APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)",
        "weighted_column_fill": "APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)"
      },
      "functions": [
        {
          "function": "weighted_column_fill",
          "signature": "weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray",
          "doc_summary": "Return finite-entry weighted means (p,) for a nonempty real matrix values (n, p) with finite positive weights (n,), using zero for unobserved columns."
        }
      ],
      "index_cards": [
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.align_reference_panel",
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights",
        "autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill"
      ]
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`87130 ms` LLM)
- **Rationale**: Build the deliverable despite the missing public mount. The convex envelope guides the search; each reported bound comes from an independently recomputable tangent minorant. The isolated import detects missing dependencies or syntax errors before numerical testing.
- **Alternatives Weighed**: `Enumerate every support` (rejected: Enumeration does not by itself produce the required partition with at most 2048 leaves.); `Use a zero-diagonal dual bound at every node` (rejected: A nonzero PSD minorant tightens the free-coordinate relaxation without changing the certificate representation.)

```bash
mkdir -p /holdout_task; cat > /holdout_task/coupled_interferometry_certified_l0.py <<'PY'
"""Reference-panel calibration and certified box-constrained sparse regression."""
import heapq
import numpy as np


def _panel(panel):
    labels = np.asarray(panel['center_id'])
    strata = np.asarray(panel['stratum'])
    raw = np.asarray(panel['X_panel_raw'], dtype=np.float64)
    controls = np.asarray(panel['controls'], dtype=np.float64)
    z = np.empty_like(raw)
    balance = np.empty(len(raw))
    for label in np.unique(labels):
        rows = labels == label
        c = controls[rows]
        c = c[:, np.sum(np.isfinite(c), axis=0) >= 2]
        offset = np.median(np.nanmean(c, axis=0))
        gain = np.median(np.nanstd(c, axis=0))
        z[rows] = (raw[rows] - offset) / gain
        for s in (0, 1, 2):
            cell = rows & (strata == s)
            balance[cell] = len(raw) / (15 * np.count_nonzero(cell))
    finite = np.isfinite(z)
    denom = balance @ finite
    fill = np.divide(balance @ np.where(finite, z, 0.0), denom,
                     out=np.zeros(z.shape[1]), where=denom > 0)
    return np.where(finite, z, fill)


def _calibrate(h, y):
    def fit(weight):
        sqrt = np.sqrt(weight)
        return np.linalg.lstsq(h * sqrt[:, None], y * sqrt, rcond=None)[0]

    theta = fit(np.ones(len(y)))
    scale = 0.0
    for step in range(100):
        residual = y - np.einsum('ij,j->i', h, theta)
        if step < 16:
            mad = np.median(np.abs(residual - np.median(residual)))
            scale = max(1.8 * mad, 1e-10)
        update = fit(1.0 / (1.0 + (residual / scale)**2))
        converged = np.linalg.norm(update - theta) <= 1e-10 * (1 + np.linalg.norm(theta))
        theta = update
        if converged and step >= 16:
            break
    return theta


def _selection(x, y, ridge, l0, box):
    p = x.shape[1]
    gram = x.T @ x
    xy = x.T @ y
    q = gram.copy()
    ix = np.diag_indices(p)
    q[ix] += ridge
    norms = np.sqrt(np.maximum(np.diag(gram), np.finfo(float).tiny))
    ev = np.linalg.eigvalsh(gram / norms[:, None] / norms[None, :])[0]
    d = max(0.0, ev * 0.995 - 1e-12) * norms**2
    h = gram.copy()
    h[ix] -= d
    a = d + ridge
    hd = np.diag(h)
    qd = np.diag(q)
    if l0 == 0:
        t = np.zeros(p)
        slope = np.zeros(p)
    elif box > 0:
        t = np.minimum(box, np.sqrt(np.divide(2*l0, a, out=np.full(p, np.inf), where=a > 0)))
        slope = l0/t + 0.5*a*t
    else:
        t = np.zeros(p)
        slope = np.zeros(p)
    free_order = np.argsort(-np.abs(xy) / np.sqrt(np.maximum(qd, 1e-300)))
    pad = 1e-12 * (1 + y @ y + box**2 * np.sum(np.abs(gram)))
    best_w = np.zeros(p)
    best = float(0.5 * (y @ y))

    def record(w):
        nonlocal best, best_w
        w = np.where(np.abs(w) > 1e-8, w, 0.0)
        r = x @ w - y
        f = float(0.5 * (r @ r + ridge * (w @ w)) + l0 * np.count_nonzero(w))
        if f < best:
            best, best_w = f, w.copy()

    def polish(w, support):
        j = np.flatnonzero(support)
        result = np.zeros(p)
        if j.size:
            sub = q[np.ix_(j, j)]
            try:
                v = np.linalg.solve(sub, xy[j])
            except np.linalg.LinAlgError:
                v = np.linalg.lstsq(sub, xy[j], rcond=None)[0]
            if np.max(np.abs(v)) > box:
                v = np.clip(v, -box, box)
                r = sub @ v - xy[j]
                for _ in range(350):
                    change = 0.
                    for k in range(len(j)):
                        if sub[k, k] <= 0:
                            continue
                        z = min(box, max(-box, v[k] - r[k]/sub[k, k]))
                        delta = z - v[k]
                        if delta:
                            v[k] = z
                            r += sub[:, k] * delta
                            change = max(change, abs(delta))
                    if change < 1e-10:
                        break
            result[j] = v
        record(result)
        return result

    def sparse_descent(w):
        w = w.copy()
        g = q @ w - xy
        for _ in range(100):
            change = 0.0
            for j in free_order:
                if qd[j] <= 0:
                    continue
                v = g[j] - qd[j]*w[j]
                z = min(box, max(-box, -v/qd[j]))
                if 0.5*qd[j]*z*z + v*z + l0 >= 0 or abs(z) <= 1e-8:
                    z = 0.0
                delta = z - w[j]
                if delta:
                    w[j] = z
                    g += q[:, j]*delta
                    change = max(change, abs(delta))
            if change < 1e-9:
                break
        record(w)
        return w

    def bound(state, center):
        # Tangent to the PSD remainder of the least squares Hessian;
        # minimize the resulting separate box quadratics exactly.
        residual = x @ center - y
        v = x.T @ residual - d * center
        u = np.minimum(box, np.divide(np.abs(v), a, out=np.full(p, box), where=a > 0))
        benefit = u * (np.abs(v) - 0.5*a*u)
        deduction = np.where(state == 1, benefit-l0,
                             np.where(state == 0, np.maximum(benefit-l0, 0.), 0.))
        return float(0.5*(residual @ residual + d @ (center*center)) -
                     (x.T @ residual) @ center - deduction.sum() - pad)

    def relax(state, warm):
        w = warm.copy()
        w[state == -1] = 0.0
        if not box:
            return w, bound(state, w)
        order = [int(j) for j in free_order if state[j] != -1 and qd[j] > 0]
        g = h @ w - xy
        free = state == 0
        for step in range(220):
            change = 0.0
            for j in order:
                v = g[j] - hd[j]*w[j]
                mag = abs(v)
                if free[j] and l0 > 0 and mag <= slope[j] + hd[j]*t[j]:
                    z = 0.0 if mag <= slope[j] else min(t[j], (mag - slope[j])/hd[j])
                else:
                    z = min(box, mag / qd[j])
                z = -z if v > 0 else z
                delta = z - w[j]
                if delta:
                    w[j] = z
                    g += h[:, j]*delta
                    change = max(change, abs(delta))
            if change < 1e-9:
                break
            if step in (3, 11, 31, 79) and bound(state, w) >= best - 1e-7:
                break
        return w, bound(state, w)

    state = np.zeros(p, dtype=np.int8)
    sparse_descent(best_w)
    center, lower = relax(state, best_w)
    sparse_descent(center)
    polish(center, np.abs(center) >= 0.5*t if l0 > 0 and box > 0 else np.ones(p, dtype=bool))
    pending = [(lower, 0, state, center)]
    popped = 0
    leaves = []
    dlist = d.tolist()
    while pending:
        lb, _, state, center = heapq.heappop(pending)
        free = np.flatnonzero(state == 0)
        if lb < best - 1e-7:
            support = (state == 1) | ((state == 0) & (np.abs(center) >= 0.5*t) & (np.abs(center) > 1e-8))
            polish(center, support)
        if lb >= best - 1e-7 or not free.size:
            leaves.append({'fixed_zero': (state == -1).tolist(),
                           'fixed_active': (state == 1).tolist(), 'lb': lb,
                           'minorant_diagonal': dlist, 'minorant_center': center.tolist()})
            continue
        if box and l0:
            ratio = np.minimum(1.0, np.abs(center[free])/t[free])
            score = np.minimum(ratio, 1.0-ratio)
            j = int(free[np.argmax(score)])
        else:
            j = int(free[0])
        for sign in (-1, 1):
            child = state.copy()
            child[j] = sign
            z, low = relax(child, center)
            popped += 1
            heapq.heappush(pending, (low, popped, child, z))
    return {'weights': best_w.tolist(), 'objective': best,
            'certified_lower_bound': min(best, min(leaf['lb'] for leaf in leaves)),
            'subcube_leaves': leaves}


def solve_coupled_interferometry_l0(study: dict) -> dict:
    aligned = _panel(study['panel'])
    strata = np.asarray(study['panel']['stratum'])
    optics = study['interferometry']
    signal = np.asarray(optics['signal_phasor'], dtype=np.complex128)
    reference = np.asarray(optics['ref_phasor'], dtype=np.complex128)
    phase = np.angle(signal * np.conj(reference))
    design = np.column_stack((np.ones(len(strata)), strata == 1, strata == 2, aligned, phase))
    y = np.asarray(study['y_assay_raw'], dtype=np.float64)
    cal = np.asarray(study['calibration_indices'])
    s3 = np.asarray(study['stage3_indices'])
    theta = _calibrate(design[cal], y[cal])
    target = y[s3] - np.einsum('ij,j->i', design[s3], theta)
    cert = _selection(np.asarray(study['X_cand'], dtype=np.float64), target,
                      float(study['lam2']), float(study['lam0']), float(study['M']))
    return {'aligned_panel': aligned, 'calibration_weights': theta,
            'deconfounded_target': target, 'certified_selection': cert}
PY
python3 -I -c "import sys; sys.path.insert(0,'/holdout_task'); import coupled_interferometry_certified_l0; import numpy; print('NumPy', numpy.__version__)"; ls -lh /holdout_task
```
- **Container Exit Code**: `0` (`636 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/coupled_interferometry_certified_l0.py']`
- **Stdout (tail)**:
```text
NumPy 1.26.4
total 16K
drwxr-xr-x 2 root root 4.0K Nov 26 04:52 __pycache__
-rw-r--r-- 1 root root 8.7K Nov 26 04:52 coupled_interferometry_certified_l0.py

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fe0694eb6c16bc15 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_fe0694eb (directory).
```

### Turn 4 — `execute_in_remote_docker` (`51768 ms` LLM)
- **Rationale**: Compare small problems with exhaustive box-face minimization to detect incorrect optima, and independently verify partition coverage and leaf bounds. End-to-end synthetic slices at both supported dimensions test whether correlation, missing values, or calibration glitches consume the stated accuracy, leaf-count, or runtime margins.
- **Alternatives Weighed**: `Use the solver’s reported gap as the sole test` (rejected: A wrong leaf formula or an incomplete support partition can produce a small reported gap.)

```bash
cat > /tmp/check_eq02.py <<'PY'
import itertools, sys, time
import numpy as np
sys.path.insert(0, '/holdout_task')
from coupled_interferometry_certified_l0 import solve_coupled_interferometry_l0, _selection


def verify(cert, x, y, ridge, l0, box):
    p = x.shape[1]
    w = np.asarray(cert['weights'])
    r = x @ w - y
    f = .5*(r@r + ridge*(w@w)) + l0*np.count_nonzero(abs(w)>1e-8)
    assert w.shape == (p,) and np.max(abs(w)) <= box and abs(f-cert['objective']) < 1e-9
    leaves = cert['subcube_leaves']
    assert 1 <= len(leaves) <= 2048, len(leaves)
    allbits = np.arange(1<<p, dtype=np.int32)
    covered = np.zeros(1<<p, dtype=np.int8)
    gram = x.T@x
    for leaf in leaves:
        z, k = leaf['fixed_zero'], leaf['fixed_active']
        assert len(z)==len(k)==p and all(type(v) is bool for v in z+k)
        assert not any(a and b for a,b in zip(z,k))
        zm = sum(1<<j for j,v in enumerate(z) if v)
        km = sum(1<<j for j,v in enumerate(k) if v)
        covered[(allbits & (zm|km)) == km] += 1
        d = np.asarray(leaf['minorant_diagonal']); c = np.asarray(leaf['minorant_center'])
        assert np.min(d) >= 0 and np.linalg.eigvalsh(gram-np.diag(d))[0] >= -1e-8
        g = gram@c - x.T@y
        b = g-d*c; a = d+ridge
        u = np.array([min(box, abs(bj)/aj) if aj>0 else box for aj,bj in zip(a,b)])
        opt = .5*a*u*u - abs(b)*u + l0
        base = .5*np.sum((x@c-y)**2) - g@c + .5*np.dot(d,c*c)
        check = base + sum(0 if zz else oo if kk else min(0,oo) for zz,kk,oo in zip(z,k,opt))
        assert leaf['lb'] <= check + 1e-8, (leaf['lb'],check)
    assert (covered==1).all()
    assert cert['certified_lower_bound'] <= min(q['lb'] for q in leaves)+1e-9
    gap = f-cert['certified_lower_bound']
    assert -1e-9 <= gap <= 1e-5, (gap, len(leaves))
    return len(leaves),gap


def oracle(x,y,ridge,l0,box):
    p=x.shape[1]; q=x.T@x+ridge*np.eye(p); b=x.T@y
    best=.5*(y@y)
    for face in itertools.product((0,1,-1,2),repeat=p):
        f=np.array(face); j=f==2; k=abs(f)==1
        w=np.zeros(p); w[k]=box*f[k]
        if np.any(j): w[j]=np.linalg.solve(q[np.ix_(j,j)],b[j]-q[np.ix_(j,k)]@w[k])
        if np.max(abs(w))>box+1e-10: continue
        r=x@w-y
        best=min(best,.5*(r@r+ridge*(w@w))+l0*np.count_nonzero(abs(w)>1e-8))
    return best

rng=np.random.default_rng(7319)
for seed in range(6):
    x=rng.normal(size=(26,6)); x[:,1]=.91*x[:,0]+.3*x[:,1]
    y=rng.normal(size=26)+x@rng.normal(size=6)
    ridge, l0, box=.04, .3, .18 + .22*seed
    cert=_selection(x,y,ridge,l0,box)
    leaves,gap=verify(cert,x,y,ridge,l0,box)
    err=abs(cert['objective']-oracle(x,y,ridge,l0,box))
    assert err < 2e-7,err
    print('oracle',seed,leaves,gap,err,flush=True)
for p in (18,19):
    for spread in (.12,.72,1.35):
        n=1800; nc=1000; ns=115
        center=np.resize(np.arange(5),n); rng.shuffle(center)
        s=(rng.random(n)<(.23+.10*center)).astype(int)+(rng.random(n)<.22)
        z=rng.normal(size=(n,8)) + .25*s[:,None]; gains=np.exp(rng.uniform(-.65,.65,5)); shifts=rng.uniform(-1.3,1.3,5)
        control=rng.normal(size=(n,18))*gains[center,None]+shifts[center,None]
        raw=z*gains[center,None]+shifts[center,None]
        control[rng.random(control.shape)<.05]=np.nan; raw[rng.random(raw.shape)<.025]=np.nan
        phase=rng.uniform(-2.2,2.2,(n,2)); ref=np.exp(1j*rng.uniform(-np.pi,np.pi,(n,2))); signal=ref*np.exp(1j*phase)
        h=np.column_stack([np.ones(n),s==1,s==2,z,phase]); theta=rng.normal(size=13)
        y=h@theta+.035*rng.normal(size=n); bad=rng.choice(nc,size=120,replace=False); y[bad]+=11*rng.choice([-1,1],size=120)
        x=rng.normal(size=(ns,p))+spread*rng.normal(size=(ns,1)); x/=np.linalg.norm(x,axis=0)
        w=np.zeros(p); w[[1,4,7,10,13]]=[1.9,-.8,1.45,-1.1,.88]
        j=np.arange(nc,nc+ns); y[j]+=x@w
        study={'panel':{'center_id':center,'stratum':s,'X_panel_raw':raw,'controls':control},'interferometry':{'signal_phasor':signal,'ref_phasor':ref},'y_assay_raw':y,'calibration_indices':np.arange(nc),'stage3_indices':j,'X_cand':x,'lam2':.025,'lam0':.16,'M':1.1}
        start=time.monotonic(); result=solve_coupled_interferometry_l0(study); elapsed=time.monotonic()-start
        leaves,gap=verify(result['certified_selection'],x,result['deconfounded_target'],.025,.16,1.1)
        align=np.mean(abs(result['aligned_panel']-z)); rel=np.linalg.norm(result['calibration_weights']-theta)/np.linalg.norm(theta)
        print('full',p,spread,leaves,gap,elapsed,align,rel,flush=True)
        assert align <= .11 and rel <= .14
PY
python3 -I -u /tmp/check_eq02.py
```
- **Container Exit Code**: `0` (`3322 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
oracle 0 1 2.908819851654698e-10 0.0
oracle 1 1 2.2712498548571602e-10 0.0
oracle 2 4 8.412150975800614e-10 0.0
oracle 3 4 5.233715683061746e-10 7.105427357601002e-15
oracle 4 5 6.329323021248001e-08 1.7763568394002505e-15
oracle 5 2 1.1360690166384302e-09 0.0
full 18 0.12 6 7.411937730239515e-11 0.1904051590245217 0.03323900555712972 0.004296943185113547
full 18 0.72 8 1.3182244629206252e-08 0.20278634899295866 0.03428083679272728 0.011600149220700354
full 18 1.35 30 3.034106299537598e-10 0.36671636695973575 0.03442715107882078 0.004507626328803562
full 19 0.12 6 9.212719476181519e-11 0.10463218507356942 0.03299050767488778 0.004989223962940061
full 19 0.72 7 2.528281140001809e-09 0.10167600400745869 0.0287086524385369 0.005215357937866848
full 19 1.35 10 3.0821745156117686e-10 0.23773874598555267 0.03463224146580744 0.005328409771971612

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fe0694eb6c16bc15 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_fe0694eb (directory).
```

### Turn 5 — `execute_in_remote_docker` (`29763 ms` LLM)
- **Rationale**: Use the public fixtures if mounted. Otherwise, independently verify the certificate and time both supported dimensions on paired-contrast problems. A gap, leaf-count, or runtime failure there would justify revisiting the relaxation optimizer before submission.
- **Alternatives Weighed**: `Submit on the existing synthetic results` (rejected: Earlier full-size slices used a shared factor but did not explicitly stress near-redundant pairs with opposing coefficients and active box walls, which could expose slow certification.)

```bash
set -e
if [ -f /holdout_task/validate_public_schema.py ] && [ -f /holdout_task/public_eq02_studies.pkl ]; then
  cd /holdout_task
  python3 -I validate_public_schema.py
else
  echo 'Public studies and validator unavailable; using targeted synthetic stress slices.'
  OPENBLAS_NUM_THREADS=1 python3 -I -u - <<'PY'
import sys, time
import numpy as np
sys.path.insert(0, '/holdout_task')
from coupled_interferometry_certified_l0 import _selection
with open('/tmp/check_eq02.py') as f:
    exec(f.read().split('rng=np.random.default_rng(7319)')[0])
rng = np.random.default_rng(902)
total = 0.
for p in (18, 19):
    n = 105
    # Shared carrier gives a connected design; paired contrasts encourage
    # opposite sparse coefficients, some of which hit the box walls.
    carrier = rng.normal(size=(n, 1))
    directions = rng.normal(size=(n, (p+1)//2))
    x = .9*carrier + .56*np.repeat(directions, 2, axis=1)[:, :p] + .22*rng.normal(size=(n,p))
    x /= np.linalg.norm(x, axis=0)
    truth = np.zeros(p)
    truth[:8] = [1.45,-1.27,-1.4,1.16,1.22,-1.45,0.72,0.]
    y = x @ truth + .025*rng.normal(size=n)
    ridge, l0, box = .014, .027, .95
    start = time.monotonic()
    cert = _selection(x, y, ridge, l0, box)
    elapsed = time.monotonic()-start
    total += elapsed
    leaves, gap = verify(cert, x, y, ridge, l0, box)
    walls = np.count_nonzero(np.isclose(np.abs(cert['weights']),box))
    print(f'p={p} leaves={leaves} gap={gap:.3g} seconds={elapsed:.3f} walls={walls}')
    assert walls >= 1, 'stress slice did not exercise active box constraints'
assert total <= 18., total
PY
fi
```
- **Container Exit Code**: `0` (`620 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public studies and validator unavailable; using targeted synthetic stress slices.
p=18 leaves=13 gap=8.83e-10 seconds=0.070 walls=3
p=19 leaves=8 gap=5.01e-09 seconds=0.026 walls=1

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fe0694eb6c16bc15 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_fe0694eb (directory).
```

### Turn 6 — `submit_completed_task` (`14196 ms` LLM)
- **Rationale**: The isolated Python runs verified the final artifact against exhaustive small-box optima, independently recomputed leaf bounds and partition coverage, and exercised both supported dimensions with active box constraints. All measured synthetic margins passed; public-fixture performance remains unverified.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/coupled_interferometry_certified_l0.py': 'FILE:8877'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_interferometry_certified_l0.py`. Synthetic verification passed at both supported dimensions, including cases with active box constraints. The largest full-pipeline certification gap was 1.32e-8, and the slowest study took 0.367 s. The public studies and validator were not available in the container.`

### Turn 7 — `execute_in_remote_docker` (`32599 ms` LLM)
- **Rationale**: Run the public validator if the fixtures are mounted. Otherwise, test whether nearly redundant pairwise contrasts slow the box subproblems enough to break the certificate gap, leaf count, or runtime limits. A failure would justify improving the subproblem solver; a pass would not justify changing calibration.
- **Alternatives Weighed**: `Test relay modes, drift equations, or within-batch classification` (rejected: The study contract has no relay state, drift equation, or classification labels. Center nuisance is handled by the prescribed reference-panel alignment.)

```bash
set -e
if [ -f /holdout_task/public_eq02_studies.pkl ] && [ -f /holdout_task/validate_public_schema.py ]; then
  cd /holdout_task
  python3 -I validate_public_schema.py
else
  echo 'Public fixtures unavailable; testing connected, nearly redundant contrasts.'
  OPENBLAS_NUM_THREADS=1 timeout 30s python3 -I -u - <<'PY'
import time
with open('/tmp/check_eq02.py') as f:
    exec(f.read().split('rng=np.random.default_rng(7319)')[0])
rng = np.random.default_rng(17021)
total = 0.
for p in (18, 19):
    n = 112
    common = rng.normal(size=(n, 1))
    base = rng.normal(size=(n, (p+1)//2))
    x = .55*common + np.repeat(base, 2, axis=1)[:, :p] + .105*rng.normal(size=(n,p))
    x /= np.linalg.norm(x, axis=0)
    w = np.zeros(p)
    w[:8] = [1.25, -1.17, -.93, 1.06, 1.08, -1.14, -.78, .75]
    y = x @ w + .008*rng.normal(size=n)
    ridge, l0, box = .002, .003, .93
    start = time.monotonic()
    cert = _selection(x, y, ridge, l0, box)
    elapsed = time.monotonic()-start
    total += elapsed
    leaves, gap = verify(cert, x, y, ridge, l0, box)
    active = np.abs(cert['weights']) > 1e-8
    walls = np.count_nonzero(np.isclose(np.abs(cert['weights']), box))
    pair_corr = min(x[:,j]@x[:,j+1] for j in range(0,p-1,2))
    print(f'p={p} min_pair_corr={pair_corr:.4f} active={active.sum()} walls={walls} leaves={leaves} gap={gap:.3g} solver_s={elapsed:.3f}')
print(f'two-study solver_s={total:.3f}')
assert total <= 18., total
PY
fi
```
- **Container Exit Code**: `0` (`1107 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public fixtures unavailable; testing connected, nearly redundant contrasts.
p=18 min_pair_corr=0.9907 active=6 walls=1 leaves=16 gap=5.66e-08 solver_s=0.227
p=19 min_pair_corr=0.9904 active=6 walls=2 leaves=14 gap=2.01e-09 solver_s=0.175
two-study solver_s=0.402

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=fe0694eb6c16bc15 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_fe0694eb (directory).
```

### Turn 8 — `submit_completed_task` (`14713 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated end-to-end checks, exhaustive small-problem comparisons, and independent leaf-bound and partition verification. The final stress slice exercised active box walls and pair correlations above 0.99 at both supported dimensions without approaching the runtime or certificate limits.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/coupled_interferometry_certified_l0.py': 'FILE:8877'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_interferometry_certified_l0.py`. In isolated Python runs, the solver matched exhaustive small-problem optima and passed independent certificate checks at both supported dimensions. The nearly redundant, box-constrained pair took 0.402 s total. The public studies and validator were unavailable, so public-data performance remains unverified.`

### Turn 9 — `submit_completed_task` (`18238 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated execution, independent certificate checks, and a two-study stress run with active box walls and pair correlations above 0.99 in 0.402 s. The requested real-data audit requires the missing public fixtures; another synthetic run would not close that gap.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_interferometry_certified_l0.py`. The solver matched exhaustive small-problem optima and passed independent certificate and partition checks at both supported dimensions. The worst synthetic certificate gap was 6.33e-8, below the 1e-5 limit. The public studies and validator were unavailable, so real-data performance remains unverified.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/coupled_interferometry_certified_l0.py": "FILE:8877"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "all_valid": true,
    "max_panel_mae": 0.045663,
    "max_calibration_rel_l2": 0.017114,
    "max_deconfounded_mae": 0.197961,
    "max_stage3_w_vs_clean_rel_l2": 0.171897,
    "max_duality_gap": 1e-09,
    "student_elapsed_sec": 0.1052,
    "instance_diagnostics": [
      {
        "seed": 99411,
        "p": 18,
        "panel_mae": 0.045663,
        "cal_rel_l2": 0.016924,
        "deconf_mae": 0.176937,
        "w_vs_clean_rel_l2": 0.171897,
        "cert_ok": true,
        "cert_reason": "ok"
      },
      {
        "seed": 99412,
        "p": 19,
        "panel_mae": 0.045333,
        "cal_rel_l2": 0.017114,
        "deconf_mae": 0.197961,
        "w_vs_clean_rel_l2": 0.096685,
        "cert_ok": true,
        "cert_reason": "ok"
      }
    ]
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_panel_mae\": 0.045663, \"max_calibration_rel_l2\": 0.017114, \"max_deconfounded_mae\": 0.197961, \"max_stage3_w_vs_clean_rel_l2\": 0.171897, \"max_duality_gap\": 1e-09, \"student_elapsed_sec\": 0.1052, \"instance_diagnostics\": [{\"seed\": 99411, \"p\": 18, \"panel_mae\": 0.045663, \"cal_rel_l2\": 0.016924, \"deconf_mae\": 0.176937, \"w_vs_clean_rel_l2\": 0.171897, \"cert_ok\": true, \"cert_reason\": \"ok\"}, {\"seed\": 99412, \"p\": 19, \"panel_mae\": 0.045333, \"cal_rel_l2\": 0.017114, \"deconf_mae\": 0.197961, \"w_vs_clean_rel_l2\": 0.096685, \"cert_ok\": true, \"cert_reason\": \"ok\"}]}\n__HEGEL_VERIFIER_AUTH_c6fb69260fbae3ad4d53cf996e782683__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_panel_mae\": 0.045663, \"max_calibration_rel_l2\": 0.017114, \"max_deconfounded_mae\": 0.197961, \"max_stage3_w_vs_clean_rel_l2\": 0.171897, \"max_duality_gap\": 1e-09, \"student_elapsed_sec\": 0.1052, \"instance_diagnostics\": [{\"seed\": 99411, \"p\": 18, \"panel_mae\": 0.045663, \"cal_rel_l2\": 0.016924, \"deconf_mae\": 0.176937, \"w_vs_clean_rel_l2\": 0.171897, \"cert_ok\": true, \"cert_reason\": \"ok\"}, {\"seed\": 99412, \"p\": 19, \"panel_mae\": 0.045333, \"cal_rel_l2\": 0.017114, \"deconf_mae\": 0.197961, \"w_vs_clean_rel_l2\": 0.096685, \"cert_ok\": true, \"cert_reason\": \"ok\"}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5902.46,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
