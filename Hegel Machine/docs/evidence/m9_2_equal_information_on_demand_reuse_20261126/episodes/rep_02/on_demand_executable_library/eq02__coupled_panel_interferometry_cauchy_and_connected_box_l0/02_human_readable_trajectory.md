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
| **Turns Used / Max Turns** | `10 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `33` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `238,112` (`227,929` / `60,544` / `10,183`) |
| **Total Episode Wall Time (s)** | `342.21s` |

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

### Turn 1 — `execute_in_remote_docker` (`12890 ms` LLM)
- **Rationale**: Inspect the solver contract and discover available primitives before choosing an implementation; this detects certificate formula mismatches without transferring any datasets.
- **Alternatives Weighed**: `Implement branching before reading the validator` (rejected: The verifier's exact lower-bound formulas determine which certificate construction will work.)

```bash
ls -lah /holdout_task; sed -n '1,260p' /holdout_task/validate_public_schema.py; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'certified box L0 bounds Cauchy IRLS reference panel alignment imputation'
```
- **Container Exit Code**: `0` (`386 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights | STATUS: APPLICABLE_SUBROUTINE(real_residual_array_positive_scale)
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound | STATUS: APPLICABLE_SUBROUTINE(real_symmetric_box_separable_l0_ridge)
autogenous_prim_reference_panel_balancing@a0bd4b96 | balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray | Return positive mean-one weights (n,) from nonempty finite real vectors groups | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights | STATUS: APPLICABLE_SUBROUTINE(discrete_group_and_stratum_vectors)
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | --describe autogenous_prim_reference_panel_balancing.weighted_column_fill | STATUS: APPLICABLE_SUBROUTINE(real_2d_matrix_with_nan_and_positive_weights)
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_psd_minorant)
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks | STATUS: APPLICABLE_SUBROUTINE(real_2d_design_matrix)
```
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task': No such file or directory
sed: can't read /holdout_task/validate_public_schema.py: No such file or directory
```

### Turn 2 — `execute_in_remote_docker` (`16149 ms` LLM)
- **Rationale**: Locate the validator and studies by filename and inspect the available primitives. If the fixtures are absent, implement from the written contract and verify with synthetic stress cases instead.
- **Alternatives Weighed**: `Assume the verifier and studies are unavailable` (rejected: The expected path is absent, but the initial snapshot names /pretrain_task; a bounded search can locate the files without transferring data.)

```bash
pwd; ls -la /pretrain_task /app /workspace 2>/dev/null; find /pretrain_task /app /workspace -maxdepth 4 -type f \( -iname '*schema*' -o -iname '*eq02*' -o -iname '*interferometry*' \) -printf '%p %s bytes\n' 2>/dev/null; export PYTHONPATH=/tmp; for f in autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound autogenous_prim_box_ridge_bounds.gram_diagonal_minorant autogenous_prim_reference_panel_balancing.balanced_cell_weights autogenous_prim_reference_panel_balancing.weighted_column_fill autogenous_prim_robust_affine_projection.robust_residual_weights; do python3 -m hegel_wormhole_skills --describe "$f"; done
```
- **Container Exit Code**: `0` (`537 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
_only_l1_primitive_sublation",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        "scale > 0 finite scalar or broadcastable array; loss in {'huber', 'cauchy'}.",
        "weights >= 0 when provided; sensitivity must have nonzero weighted norm above gain_floor for identifiable slope."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "propagate_linear_basis": "APPLICABLE_SUBROUTINE(real_linear_transition_and_2d_initial_basis)",
        "profile_weighted_affine": "APPLICABLE_SUBROUTINE(real_2d_predictor_3d_sensitivity_2d_target)",
        "robust_residual_weights": "APPLICABLE_SUBROUTINE(real_residual_array_positive_scale)"
      },
      "functions": [
        {
          "function": "robust_residual_weights",
          "signature": "robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray",
          "doc_summary": "For residuals of shape S and a positive scalar or scale array broadcastable to S, return Huber or Cauchy IRLS weights of shape S."
        }
      ],
      "index_cards": [
        "autogenous_prim_robust_affine_projection@abc35d3d | propagate_linear_basis(transition: np.ndarray, initial: np.ndarray, increments: np.ndarray) -> np.ndarray | Return a trajectory of shape (steps+1, n, k) from transition (n, n), initial ( | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.propagate_linear_basis",
        "autogenous_prim_robust_affine_projection@abc35d3d | profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | For predictor, target and optional weights (n, m) and predictor sensitivity (n | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.profile_weighted_affine",
        "autogenous_prim_robust_affine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar or scale array broadcastable to | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights"
      ]
    }
  ]
}
```

### Turn 3 — `execute_in_remote_docker` (`18556 ms` LLM)
- **Rationale**: Read the diagonal certificate implementation to match its lower-bound formula and confirm which numerical libraries are available before writing the solver.
- **Alternatives Weighed**: `Use residual-dual certificates immediately` (rejected: Without the public validator, its dual-vector sign convention is unknown; the diagonal primitive documents the minorant convention.)

```bash
ls -la /; find /tmp/hegel_wormhole_skills -maxdepth 2 -name '*box_ridge*' -o -name '*reference_panel*'; sed -n '1,260p' /tmp/hegel_wormhole_skills/autogenous_prim_box_ridge_bounds.py; python3 -I -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)'
```
- **Container Exit Code**: `0` (`599 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
raise ValueError('diagonal must be a valid PSD minorant of design.T @ design (d_j <= ||X_:,j||_2^2)')
            _G_chk = (_X_chk.T @ _X_chk) - np.diag(_d_chk)
            if _p_chk <= 1536:
                _min_eig = float(np.linalg.eigvalsh(0.5 * (_G_chk + _G_chk.T))[0])
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
1.26.4 1.13.0
```

### Turn 4 — `execute_in_remote_docker` (`79934 ms` LLM)
- **Rationale**: Implement the written contract using the inspected minorant formula. Compilation detects artifact or syntax failures before independent certificate and worst-slice tests.
- **Alternatives Weighed**: `Enumerate supports with exact box fits` (rejected: Enumeration alone does not give a compact partition certificate and spends work on supports a convex lower bound can eliminate.)

```bash
mkdir -p /holdout_task; cat > /holdout_task/coupled_interferometry_certified_l0.py <<'PY'
"""Panel calibration and globally bounded connected box-L0 selection."""
import heapq
import numpy as np
from scipy.optimize import minimize


def _align(panel):
    group = np.asarray(panel['center_id'])
    strata = np.asarray(panel['stratum'])
    raw = np.asarray(panel['X_panel_raw'], dtype=np.float64)
    controls = np.asarray(panel['controls'], dtype=np.float64)
    aligned = np.empty_like(raw)
    weight = np.empty(len(group))
    for center in np.unique(group):
        rows = group == center
        ref = controls[rows]
        ref = ref[:, (np.isfinite(ref).sum(axis=0) >= 2)]
        offset = np.median(np.nanmean(ref, axis=0))
        gain = np.median(np.nanstd(ref, axis=0))
        aligned[rows] = (raw[rows] - offset) / gain
        for s in (0, 1, 2):
            cell = rows & (strata == s)
            weight[cell] = 1.0 / cell.sum()
    weight /= weight.mean()
    known = np.isfinite(aligned)
    fill = np.nansum(aligned * weight[:, None], axis=0) / (weight @ known)
    return np.where(known, aligned, fill)


def _calibrate(h, y):
    theta = np.linalg.lstsq(h, y, rcond=None)[0]
    for multiplier in (2.5, 1.5):
        residual = y - np.einsum('ij,j->i', h, theta)
        scale = max(1e-8, multiplier * 1.4826 * np.median(np.abs(residual - np.median(residual))))
        for _ in range(75):
            residual = y - np.einsum('ij,j->i', h, theta)
            root = 1 / np.hypot(1, residual / scale)
            nxt = np.linalg.lstsq(h * root[:, None], y * root, rcond=None)[0]
            change = np.linalg.norm(nxt - theta)
            theta = nxt
            if change <= 1e-10 * (1 + np.linalg.norm(theta)):
                break
    return theta


def _select(x, y, l2, l0, m):
    p = x.shape[1]
    g = x.T @ x
    b = x.T @ y
    yy = 0.5 * (y @ y)
    jj = np.arange(p)
    qmat = g + np.diag(np.full(p, l2))
    norms = np.sqrt(np.maximum(np.diag(g), np.finfo(float).tiny))
    eig = np.linalg.eigvalsh(g / norms[:, None] / norms[None, :])[0]
    d = max(0., eig - 2e-12) * np.diag(g)
    r = g - np.diag(d)
    k = d + l2
    rd = np.diag(r)
    qdiag = rd + k
    t = np.full(p, m)
    np.sqrt(np.divide(2*l0, k, out=np.full(p, np.inf), where=k > 0), out=t)
    t = np.minimum(m, t)
    slope = np.divide(l0, t, out=np.zeros(p), where=t > 0) + 0.5*k*t
    best = yy
    weights = np.zeros(p)
    cache = set()

    def offer(support, initial):
        nonlocal best, weights
        key = support.tobytes()
        if key in cache or l0 * support.sum() >= best:
            return
        cache.add(key)
        idx = jj[support]
        if not len(idx):
            return
        a = qmat[np.ix_(idx, idx)]
        v = b[idx]
        try:
            w = np.linalg.solve(a, v)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(a, v, rcond=None)[0]
        if np.any(np.abs(w) > m):
            def fun(z):
                az = a @ z
                return (0.5*az-v) @ z, az-v
            w = minimize(fun, np.clip(w, -m, m), jac=True, method='L-BFGS-B',
                         bounds=[(-m,m)]*len(idx), options={'ftol': 1e-15, 'gtol': 1e-10}).x
        w[np.abs(w) <= 1e-8] = 0.
        full = np.zeros(p)
        full[idx] = w
        err = x @ full - y
        obj = float(0.5*(err @ err + l2*(w @ w)) + l0*np.count_nonzero(w))
        if obj < best:
            best, weights = obj, full

    def relax(zero, active, start):
        free = ~(zero | active)
        ids = jj[~zero]
        c = np.where(zero, 0., start).copy()
        rc = r @ c
        for sweep in range(110):
            change = 0.
            for j in ids:
                z = b[j] - rc[j] + rd[j]*c[j]
                az = abs(z)
                if free[j] and az <= slope[j]:
                    u = 0.
                elif free[j] and az <= slope[j] + rd[j]*t[j] and rd[j] > 0:
                    u = (az-slope[j])/rd[j]
                else:
                    u = min(m, az/qdiag[j]) if qdiag[j] > 0 else m
                u = np.copysign(u, z)
                delta = u-c[j]
                if delta:
                    rc += delta*r[j]
                    c[j] = u
                    change = max(change, abs(delta))
            if change < 2e-9:
                break
        rc = r @ c
        z = np.abs(b-rc)
        u = np.minimum(m, np.divide(z, k, out=np.full(p,m), where=k > 0))
        on = l0 - u*(z - 0.5*k*u)
        const = yy - 0.5*(c @ rc)
        lb = const + on[active].sum() + np.minimum(on[free],0.).sum()
        # A downward rounding margin for the float64 Gram representation.
        lb -= 4e-13 * (1 + abs(yy) + np.abs(c) @ (np.abs(g) @ np.abs(c)) + p*l0 + m*np.abs(b).sum())
        frac = np.divide(np.abs(c), t, out=np.zeros(p), where=t > 0)
        return float(lb), c, frac, on

    leaves = []
    pending = []
    serial = 0
    tol = 8e-6

    def add(zero, active, start):
        nonlocal serial
        lb, c, frac, on = relax(zero, active, start)
        free = ~(zero | active)
        if lb < best-tol or not pending and not leaves:
            offer(active | (free & (frac >= 0.5)), c)
            offer(active | (free & (on < 0)), c)
        if lb >= best-tol or not free.any():
            leaves.append((zero, active, lb, c))
        else:
            score = np.minimum(frac, 1-frac)
            j = jj[free][np.argmax(score[free] - 1e-12*np.abs(on[free]))]
            serial += 1
            heapq.heappush(pending, (lb, serial, zero, active, c, j))

    mask = np.zeros(p, dtype=bool)
    add(mask, mask, np.zeros(p))
    while pending:
        lb, _, zero, active, c, j = heapq.heappop(pending)
        if lb >= best-tol:
            leaves.append((zero, active, lb, c))
            continue
        nz, na = zero.copy(), active.copy()
        nz[j] = True
        na[j] = True
        add(zero, na, c)
        add(nz, active, c)
    diagonal = d.tolist()
    return {'weights': weights.tolist(), 'objective': best,
            'certified_lower_bound': min(best, min(leaf[2] for leaf in leaves)),
            'subcube_leaves': [{'fixed_zero': z.tolist(), 'fixed_active': a.tolist(),
                                'lb': lb, 'minorant_diagonal': diagonal, 'minorant_center': c.tolist()}
                               for z, a, lb, c in leaves]}


def solve_coupled_interferometry_l0(study: dict) -> dict:
    aligned = _align(study['panel'])
    stratum = np.asarray(study['panel']['stratum'])
    optical = study['interferometry']
    signal = np.asarray(optical['signal_phasor'], dtype=np.complex128)
    ref = np.asarray(optical['ref_phasor'], dtype=np.complex128)
    phase = np.angle(signal * np.conj(ref))
    h = np.column_stack((np.ones(len(stratum)), stratum == 1, stratum == 2, aligned, phase))
    y = np.asarray(study['y_assay_raw'], dtype=np.float64)
    cal = np.asarray(study['calibration_indices'])
    target = np.asarray(study['stage3_indices'])
    theta = _calibrate(h[cal], y[cal])
    response = y[target] - np.einsum('ij,j->i', h[target], theta)
    selection = _select(np.asarray(study['X_cand'], dtype=np.float64), response,
                        float(study['lam2']), float(study['lam0']), float(study['M']))
    return {'aligned_panel': aligned, 'calibration_weights': theta,
            'deconfounded_target': response, 'certified_selection': selection}
PY
python3 -I -m py_compile /holdout_task/coupled_interferometry_certified_l0.py
```
- **Container Exit Code**: `0` (`400 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/coupled_interferometry_certified_l0.py']`
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f9497b40df24e9ab (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f9497b40 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`40136 ms` LLM)
- **Rationale**: The public fixtures are unavailable in this container. Run connected synthetic designs at both required widths and independently check each leaf bound and the complete support partition. Compare small box-constrained cases against exhaustive fits to distinguish a certificate defect from an incumbent-fit defect.
- **Alternatives Weighed**: `Verify only on an easy independent design` (rejected: That can miss weak bounds on correlated designs, excessive leaf counts, and inaccurate fits at the box boundary.)

```bash
cat > /tmp/eq02_stress.py <<'PY'
import importlib.util, numpy as np, time
from scipy.optimize import lsq_linear
path='/holdout_task/coupled_interferometry_certified_l0.py'
spec=importlib.util.spec_from_file_location('eq02',path)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def check(x,y,l2,l0,m,sol):
    p=x.shape[1]; w=np.array(sol['weights']); f=.5*np.sum((x@w-y)**2)+.5*l2*(w@w)+l0*np.count_nonzero(abs(w)>1e-8)
    assert np.max(abs(w))<=m and abs(f-sol['objective'])<1e-8
    ids=np.arange(1<<p,dtype=np.int64); seen=np.zeros(len(ids),dtype=np.uint8); bits=1<<np.arange(p); g=x.T@x
    slack=0
    for leaf in sol['subcube_leaves']:
        assert all(type(v) is bool for name in ['fixed_active','fixed_zero'] for v in leaf[name]); a=np.array(leaf['fixed_active']); z=np.array(leaf['fixed_zero']); assert not (a&z).any()
        seen[(ids & (bits@(a|z))) == (bits@a)]+=1
        c=np.array(leaf['minorant_center']); d=np.array(leaf['minorant_diagonal']); assert min(d)>=0 and np.linalg.eigvalsh(g-np.diag(d))[0]>=-1e-9
        r=y-x@c; h=x.T@r; v=h+d*c; k=d+l2
        u=np.clip(np.divide(v,k,out=m*np.sign(v),where=k>0),-m,m)
        on=l0+.5*k*u*u-v*u
        lb=.5*(r@r)+h@c+.5*np.dot(d,c*c)+on[a].sum()+np.minimum(on[~(a|z)],0).sum()
        slack=max(slack,leaf['lb']-lb)
    assert np.all(seen==1) and slack<=1e-9
    return f-sol['certified_lower_bound'],len(sol['subcube_leaves'])

def design(rng,n,p,case):
    x=rng.normal(size=(n,p))/np.sqrt(n)
    if case=='cluster':
        x=.48*x+.88*np.repeat(rng.normal(size=(n,(p+2)//3))/np.sqrt(n),3,axis=1)[:,:p]
    if case=='collinear':
        x=.17*x+.98*np.repeat(rng.normal(size=(n,(p+1)//2))/np.sqrt(n),2,axis=1)[:,:p]
    if case=='scaled': x*=np.geomspace(.15,4,p)
    if case=='bridge':
        x=np.zeros((n,p)); rr=np.arange(n-1); j=rng.integers(p,size=n-1); x[rr,j]=rng.normal(size=n-1); x[-1,:]=.07; x/=np.linalg.norm(x,axis=0)
    return x

for p in [18,19]:
    for i,case in enumerate(['dense','cluster','collinear','scaled','bridge']):
        rng=np.random.default_rng(31+p+i); x=design(rng,170,p,case); truth=np.zeros(p); truth[[1,5,8,12,16]]=[1.8,-.65,1.05,-.70,1.25]; y=x@truth+.08*rng.normal(size=len(x)); l2,l0,m=.025,.115,.84
        t=time.monotonic(); sol=mod._select(x,y,l2,l0,m); elapsed=time.monotonic()-t
        gap,count=check(x,y,l2,l0,m,sol); print(p,case,'seconds',round(elapsed,4),'gap',gap,'leaves',count,flush=True)

for seed in range(4):
    rng=np.random.default_rng(seed); n,p=50,7; x=design(rng,n,p,'cluster'); y=x@rng.normal(size=p)+.1*rng.normal(size=n); l2,l0,m=.025,.11,.67
    sol=mod._select(x,y,l2,l0,m); oracle=.5*y@y
    for mask in range(1,1<<p):
        a=((mask>>np.arange(p))&1).astype(bool); k=a.sum(); xx=np.vstack((x[:,a],np.sqrt(l2)*np.eye(k))); yy=np.r_[y,np.zeros(k)]
        fit=lsq_linear(xx,yy,bounds=(-m,m),tol=1e-13); oracle=min(oracle,fit.cost+l0*k)
    assert abs(sol['objective']-oracle)<1e-7, (seed,sol['objective'],oracle)
print('Exhaustive small box oracle: 4/4')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/eq02_stress.py
```
- **Container Exit Code**: `0` (`1793 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
18 dense seconds 0.0037 gap 7.627676268384675e-12 leaves 1
18 cluster seconds 0.0743 gap 1.1423084700368236e-11 leaves 14
18 collinear seconds 0.0271 gap 9.760636743294526e-12 leaves 14
18 scaled seconds 0.0009 gap 1.6470824704128972e-11 leaves 1
18 bridge seconds 0.0011 gap 6.035172361862351e-12 leaves 1
19 dense seconds 0.0015 gap 6.302292021587164e-12 leaves 2
19 cluster seconds 0.0712 gap 7.995382134140527e-12 leaves 26
19 collinear seconds 0.0208 gap 9.414247159611477e-12 leaves 16
19 scaled seconds 0.0016 gap 1.326450060901152e-11 leaves 1
19 bridge seconds 0.001 gap 6.594502721668505e-12 leaves 1
Exhaustive small box oracle: 4/4

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f9497b40df24e9ab (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f9497b40 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`53774 ms` LLM)
- **Rationale**: Exercise both supported widths through the full pipeline. Independent panel and handoff calculations isolate upstream failures; per-leaf primitive checks and full support coverage isolate certificate failures. Compare Cauchy calibration with OLS under the same glitches, and run the public validator if its fixtures are now available.
- **Alternatives Weighed**: `Submit after Stage-3 verification alone` (rejected: The prior tests did not exercise the public entry point, the calibration handoff, or runtime across the full pipeline.)

```bash
cat > /tmp/eq02_end_to_end.py <<'PY'
import importlib.util
import pathlib
import sys
import time
import numpy as np

root = pathlib.Path('/holdout_task')
spec = importlib.util.spec_from_file_location('eq02', root / 'coupled_interferometry_certified_l0.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.path.insert(0, '/tmp')
from hegel_wormhole_skills.autogenous_prim_box_ridge_bounds import box_ridge_cube_lower_bound

for p, seed in [(18, 62), (19, 100)]:
    rng = np.random.default_rng(seed)
    counts = np.array([[120, 40, 30], [37, 149, 44], [45, 32, 120], [183, 37, 36], [35, 129, 44]])
    c, s = np.repeat(np.arange(5), counts.sum(axis=1)), np.repeat(np.tile(np.arange(3), 5), counts.ravel())
    n = len(c)
    latent = rng.normal(size=(n, 8)) + (s[:, None] - 1) * np.linspace(-0.4, 0.4, 8)
    offsets = np.array([-2.8, 0.9, 2.5, -1.5, 4.1])
    gains = np.array([0.56, 1.45, 2.15, 0.91, 1.75])
    raw = offsets[c, None] + gains[c, None] * latent
    standards = offsets[c, None] + gains[c, None] * rng.normal(size=(n, 18))
    raw[rng.random(raw.shape) < .025] = np.nan
    standards[rng.random(standards.shape) < .05] = np.nan
    phase = rng.uniform(-np.pi, np.pi, (n, 2))
    carrier = rng.uniform(-np.pi, np.pi, (n, 2))
    ref, sig = np.exp(1j*carrier), np.exp(1j*(carrier+phase))
    h = np.column_stack((np.ones(n), s == 1, s == 2, latent, phase))
    theta = rng.normal(size=13) * 0.5
    idx = rng.permutation(n)
    dst, cal = idx[:220], idx[220:]
    x = rng.normal(size=(len(dst), p)) / np.sqrt(len(dst))
    x = .42*x + .85*np.repeat(rng.normal(size=(len(dst), (p+2)//3))/np.sqrt(len(dst)), 3, axis=1)[:, :p]
    target = np.zeros(p)
    target[[1, 5, 8, 12, 16]] = [1.2, -.6, 1.05, -.69, .8]
    y = h@theta + .05*rng.normal(size=n)
    y[dst] += x@target
    glitches = rng.choice(cal, int(.12*len(cal)), replace=False)
    y[glitches] += rng.choice([-11., 11.], len(glitches))
    study = dict(panel=dict(center_id=c, stratum=s, X_panel_raw=raw, controls=standards),
                 interferometry=dict(signal_phasor=sig, ref_phasor=ref), y_assay_raw=y,
                 calibration_indices=cal, stage3_indices=dst, X_cand=x, lam2=.025, lam0=.115, M=.8)
    tick = time.monotonic()
    out = mod.solve_coupled_interferometry_l0(study)
    elapsed = time.monotonic() - tick
    z, th, dest, sol = (out[k] for k in ('aligned_panel', 'calibration_weights', 'deconfounded_target', 'certified_selection'))
    assert (z.shape, th.shape, dest.shape) == ((n, 8), (13,), (len(dst),))
    assert z.dtype == th.dtype == dest.dtype == np.float64
    aligned = raw.copy()
    for i in range(5):
        ref_i = standards[c == i]
        columns = [v[np.isfinite(v)] for v in ref_i.T]
        columns = [v for v in columns if len(v) >= 2]
        aligned[c == i] = (raw[c == i] - np.median([v.mean() for v in columns])) / np.median([v.std() for v in columns])
    cell_weight = 1/counts[c, s]
    known = np.isfinite(aligned)
    for j in range(8):
        take = known[:, j]
        aligned[~take, j] = np.average(aligned[take, j], weights=cell_weight[take])
    np.testing.assert_allclose(z, aligned, atol=1e-14)
    np.testing.assert_allclose(dest, y[dst]-np.column_stack((h[:, :3], aligned, phase))[dst]@th, atol=1e-13)
    panel_mae = np.mean(abs(z-latent))
    cal_rel = np.linalg.norm(th-theta)/np.linalg.norm(theta)
    ols = np.linalg.lstsq(np.column_stack((h[:, :3], aligned, phase))[cal], y[cal], rcond=None)[0]
    ols_rel = np.linalg.norm(ols-theta)/np.linalg.norm(theta)
    leaves = sol['subcube_leaves']
    w = np.array(sol['weights'])
    f = .5*np.linalg.norm(x@w-dest)**2 + .5*.025*(w@w) + .115*np.count_nonzero(abs(w)>1e-8)
    assert abs(f-sol['objective']) < 1e-9 and np.max(abs(w)) <= .8
    gap = sol['objective']-sol['certified_lower_bound']
    ids = np.arange(1<<p); bits = 1<<np.arange(p); covered = np.zeros(len(ids), dtype=np.int16)
    for leaf in leaves:
        a, zmask = (np.asarray(leaf[k]) for k in ('fixed_active','fixed_zero'))
        assert a.dtype == zmask.dtype == bool and a.shape == zmask.shape == (p,) and not (a&zmask).any()
        covered[(ids & (bits@(a|zmask))) == (bits@a)] += 1
        expected = box_ridge_cube_lower_bound(x, dest, np.asarray(leaf['minorant_center']), .8, .115,
                    ridge=.025, fixed_zero=zmask, fixed_active=a, diagonal=np.asarray(leaf['minorant_diagonal']))
        assert leaf['lb'] <= expected + 1e-9
    print(f'p={p} time={elapsed:.4f}s panel_MAE={panel_mae:.6f} calibration_relL2={cal_rel:.6f} OLS_relL2={ols_rel:.6f} gap={gap:.3g} leaves={len(leaves)}', flush=True)
    assert panel_mae <= .11 and cal_rel <= .14 and 0 <= gap <= 1e-5 and 1 <= len(leaves) <= 2048
    assert (covered == 1).all()
    assert sol['certified_lower_bound'] == min(leaf['lb'] for leaf in leaves)
    assert elapsed <= 9
PY
python3 -I /tmp/eq02_end_to_end.py
if [ -f /holdout_task/validate_public_schema.py ] && [ -f /holdout_task/public_eq02_studies.pkl ]; then python3 /holdout_task/validate_public_schema.py; fi
```
- **Container Exit Code**: `0` (`1670 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
p=18 time=0.1969s panel_MAE=0.035630 calibration_relL2=0.014965 OLS_relL2=0.321055 gap=5.77e-08 leaves=17
p=19 time=0.2351s panel_MAE=0.050206 calibration_relL2=0.016609 OLS_relL2=0.203815 gap=7.7e-12 leaves=16

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f9497b40df24e9ab (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f9497b40 (directory).
```

### Turn 7 — `submit_completed_task` (`16510 ms` LLM)
- **Rationale**: Clean-process tests verified the exact artifact at both supported widths, including the calibration handoff, every leaf bound, and exhaustive support coverage. Four small box-constrained cases also matched an exhaustive oracle.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`11` turns remaining; artifact status={'/holdout_task/coupled_interferometry_certified_l0.py': 'FILE:7289'}): Do NOT lock prematurely with `11` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_interferometry_certified_l0.py`. Both synthetic end-to-end studies passed: worst panel MAE was 0.0502, worst calibration relative L2 error was 0.0166, and the largest certificate gap was 5.77e-8. Combined runtime was 0.43 seconds. The public studies and validator were not mounted.`

### Turn 8 — `execute_in_remote_docker` (`27347 ms` LLM)
- **Rationale**: Use real fixtures if they are now mounted. Otherwise test two previously untested support geometries with near-interchangeable columns and an active box constraint. A gap, leaf-count, or runtime failure would call for tighter node bounds rather than a change to the calibration model.
- **Alternatives Weighed**: `Apply relay, drift, or classification checks` (rejected: This contract defines a continuous assay regression and box-constrained support selection; it has no relay states, drift operator, or class labels.)

```bash
set -e
if [ -f /holdout_task/public_eq02_studies.pkl ] && [ -f /holdout_task/validate_public_schema.py ]; then python3 -I /holdout_task/validate_public_schema.py; else printf '%s\n' 'Public studies and validator are not both available; running synthetic support-tie audit.'; fi
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util, time
import numpy as np
pname='/holdout_task/coupled_interferometry_certified_l0.py'
s=importlib.util.spec_from_file_location('eq02',pname); mod=importlib.util.module_from_spec(s); s.loader.exec_module(mod)
for p in (18,19):
    for case in ('mixed_aliases','signed_shared_factor'):
        rng=np.random.default_rng(750+p); n=180
        noise=rng.normal(size=(n,p))/np.sqrt(n)
        if case=='mixed_aliases':
            shared=np.repeat(rng.normal(size=(n,(p+2)//3))/np.sqrt(n),3,axis=1)[:,:p]
            x=.12*noise+.98*shared
            truth=np.tile([.84,0,0],(p+2)//3)[:p]
        else:
            x=.36*noise+.91*rng.normal(size=(n,1))/np.sqrt(n)*np.where(np.arange(p)%2,1,-1)
            truth=np.where(np.arange(p)%3==0,.65,0.)
        y=x@truth+.02*rng.normal(size=n)
        t=time.perf_counter(); sol=mod._select(x,y,.025,.105,.73); elapsed=time.perf_counter()-t
        leaves=sol['subcube_leaves']; gap=sol['objective']-sol['certified_lower_bound']; w=np.asarray(sol['weights'])
        objective=.5*np.sum((x@w-y)**2)+.5*.025*(w@w)+.105*np.count_nonzero(abs(w)>1e-8)
        print(f'p={p} {case}: gap={gap:.3g} leaves={len(leaves)} time={elapsed:.3f}s',flush=True)
        assert abs(objective-sol['objective'])<1e-9 and max(abs(w))<=.73
        assert 0<=gap<=1e-5 and 1<=len(leaves)<=2048 and elapsed<=9
        assert abs(min(leaf['lb'] for leaf in leaves)-sol['certified_lower_bound'])<1e-9
PY
```
- **Container Exit Code**: `0` (`832 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public studies and validator are not both available; running synthetic support-tie audit.
p=18 mixed_aliases: gap=8.05e-12 leaves=8 time=0.029s
p=18 signed_shared_factor: gap=6.89e-06 leaves=1 time=0.000s
p=19 mixed_aliases: gap=6.27e-06 leaves=12 time=0.074s
p=19 signed_shared_factor: gap=4.59e-12 leaves=3 time=0.010s

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=f9497b40df24e9ab (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_f9497b40 (directory).
```

### Turn 9 — `submit_completed_task` (`14935 ms` LLM)
- **Rationale**: Clean-process tests verified the saved artifact at both supported widths, including the calibration handoff, independent lower bounds, and complete support coverage. Near-tied supports and signed shared factors remained within the gap and leaf limits, so the stress audit did not justify an artifact change.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/coupled_interferometry_certified_l0.py': 'FILE:7289'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_interferometry_certified_l0.py`. Both synthetic end-to-end studies passed in 0.43 seconds combined, with worst panel MAE of 0.0502 and worst relative calibration error of 0.0166. The largest certificate gap across all stress tests was 6.89e-6, below the 1e-5 limit. The public studies and validator were unavailable.`

### Turn 10 — `submit_completed_task` (`19622 ms` LLM)
- **Rationale**: The verified artifact meets the measured tolerances at both supported widths. Independent checks covered the calibration handoff, leaf bounds, and full support partition; the additional near-tie audit did not reveal a failure requiring a change.
- **Alternatives Weighed**: `Repeat the clean-process synthetic audit` (rejected: The saved artifact is already verified, and repeating those checks would not resolve the missing public-data validation.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_interferometry_certified_l0.py`. Two synthetic end-to-end studies ran in 0.43 seconds combined, with worst panel MAE of 0.0502 and worst relative calibration error of 0.0166. Across stress tests, the largest certificate gap was 6.89e-6 and the largest partition had 26 leaves. The public studies and validator were not mounted, so public-data performance remains unverified.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/coupled_interferometry_certified_l0.py": "FILE:7289"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "all_valid": true,
    "max_panel_mae": 0.045663,
    "max_calibration_rel_l2": 0.023469,
    "max_deconfounded_mae": 0.205521,
    "max_stage3_w_vs_clean_rel_l2": 0.169728,
    "max_duality_gap": 2.379e-06,
    "student_elapsed_sec": 0.0934,
    "instance_diagnostics": [
      {
        "seed": 99411,
        "p": 18,
        "panel_mae": 0.045663,
        "cal_rel_l2": 0.023469,
        "deconf_mae": 0.179751,
        "w_vs_clean_rel_l2": 0.169728,
        "cert_ok": true,
        "cert_reason": "ok"
      },
      {
        "seed": 99412,
        "p": 19,
        "panel_mae": 0.045333,
        "cal_rel_l2": 0.021079,
        "deconf_mae": 0.205521,
        "w_vs_clean_rel_l2": 0.093493,
        "cert_ok": true,
        "cert_reason": "ok"
      }
    ]
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_panel_mae\": 0.045663, \"max_calibration_rel_l2\": 0.023469, \"max_deconfounded_mae\": 0.205521, \"max_stage3_w_vs_clean_rel_l2\": 0.169728, \"max_duality_gap\": 2.379e-06, \"student_elapsed_sec\": 0.0934, \"instance_diagnostics\": [{\"seed\": 99411, \"p\": 18, \"panel_mae\": 0.045663, \"cal_rel_l2\": 0.023469, \"deconf_mae\": 0.179751, \"w_vs_clean_rel_l2\": 0.169728, \"cert_ok\": true, \"cert_reason\": \"ok\"}, {\"seed\": 99412, \"p\": 19, \"panel_mae\": 0.045333, \"cal_rel_l2\": 0.021079, \"deconf_mae\": 0.205521, \"w_vs_clean_rel_l2\": 0.093493, \"cert_ok\": true, \"cert_reason\": \"ok\"}]}\n__HEGEL_VERIFIER_AUTH_a95a904aeac8cbd02fa21877f2847681__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_panel_mae\": 0.045663, \"max_calibration_rel_l2\": 0.023469, \"max_deconfounded_mae\": 0.205521, \"max_stage3_w_vs_clean_rel_l2\": 0.169728, \"max_duality_gap\": 2.379e-06, \"student_elapsed_sec\": 0.0934, \"instance_diagnostics\": [{\"seed\": 99411, \"p\": 18, \"panel_mae\": 0.045663, \"cal_rel_l2\": 0.023469, \"deconf_mae\": 0.179751, \"w_vs_clean_rel_l2\": 0.169728, \"cert_ok\": true, \"cert_reason\": \"ok\"}, {\"seed\": 99412, \"p\": 19, \"panel_mae\": 0.045333, \"cal_rel_l2\": 0.021079, \"deconf_mae\": 0.205521, \"w_vs_clean_rel_l2\": 0.093493, \"cert_ok\": true, \"cert_reason\": \"ok\"}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 6013.33,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
