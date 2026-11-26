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
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `253,139` (`242,088` / `62,976` / `11,051`) |
| **Total Episode Wall Time (s)** | `358.58s` |

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

### Turn 1 — `execute_in_remote_docker` (`14358 ms` LLM)
- **Rationale**: Inspect the validator and reference catalog first: a mismatch in how the validator replays leaf bounds would invalidate an otherwise optimal fit and change the certificate implementation.

```bash
ls -lh /holdout_task; sed -n '1,260p' /holdout_task/validate_public_schema.py; if test -f /tmp/hegel_knowledge_ref/describe_ref.py; then python3 /tmp/hegel_knowledge_ref/describe_ref.py --search 'box L0 certified diagonal minorant dual branch bound'; python3 /tmp/hegel_knowledge_ref/describe_ref.py --search 'Cauchy IRLS reference panel alignment imputation'; fi
```
- **Container Exit Code**: `0` (`360 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ontrol col with >=2 finite obs & spread>0 per group | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights
autogenous_prim_robust_affine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar or scale array broadcastable to | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_robust_affine_projection.robust_residual_weights
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_reference_panel_balancing.align_reference_panel
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_reference_panel_balancing.weighted_column_fill
autogenous_prim_robust_trend_decomposition@1f45421a | cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float] | Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank bef | REQ: 1D strictly increasing t>=0 (n>=4), full-rank design, scale>0, >=3 pts/side for step | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_robust_trend_decomposition.cauchy_linear_fit
autogenous_control_panel_deconfounding@aa502f6d | align_control_panel(X: np.ndarray, centers: np.ndarray, signal_columns: tuple[int, ...], control_columns: tuple[int, ...] | None=None) -> np.ndarray | Return signal values in control-referenced coordinates, in input row order | REQ: {'field': 'observations_per_contributing_control', 'op': '>=', 'value': 2}; {'field': 'u | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_control_panel_deconfounding.align_control_panel
```
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task': No such file or directory
sed: can't read /holdout_task/validate_public_schema.py: No such file or directory
```

### Turn 2 — `execute_in_remote_docker` (`16181 ms` LLM)
- **Rationale**: Inspect the reported task mount and reference contracts. If the public fixtures are absent there too, implement against the stated schema and verify with a separate synthetic exact oracle rather than reporting a public validation score.
- **Alternatives Weighed**: `Implement against the task prose without locating the available fixtures` (rejected: The reported starting directory may contain a public validator or equivalent that determines how certificate bounds are replayed.)

```bash
pwd; ls -la /; if test -d /pretrain_task; then find /pretrain_task -maxdepth 3 -type f -printf '%s %p\n' | sort -k2 | head -100; fi; for item in autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound autogenous_sparse_ridge_certification.solve_and_certify_l0 autogenous_prim_reference_panel_balancing.align_reference_panel autogenous_prim_robust_trend_decomposition.cauchy_linear_fit; do python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe "$item"; done; find /tmp/hegel_knowledge_ref -iname '*.reference.txt' | grep -E 'box_ridge|sparse_ridge|panel_balancing|robust_trend'
```
- **Container Exit Code**: `0` (`453 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"Inputs are finite real arrays convertible to float64, with magnitudes that keep intermediate residuals, model evaluations, and least-squares calculations representable. The linear fit requires a nonempty (n,p) design and an (n,) response with n >= p >= 1 and numerical full column rank, including after Cauchy row weighting. Supplied scale and noise_floor must be positive finite scalars.",
    "The relaxation fit requires aligned length-n vectors, n >= 4, strictly increasing nonnegative times measured from the supplied onset, and a predictor numerically independent of the intercept. The onset rate is independently specified and nonnegative, noise_scale is positive, curvature_ratio is nonnegative, 2 <= bins <= n, and 0 < tau_min < tau_max. The saturation branch requires a positive onset rate, a numerically full-rank fitted Jacobian, and a positive slope not active at its lower bound.",
    "The step fit requires aligned length-n vectors and a nonempty integer vector of candidate split indices, each leaving at least three samples on either side. Each candidate design must have full column rank: the predictor cannot be constant within both segments. The scale and loss_clip are positive finite scalars. Interpretation as a level step assumes a shared local trend on both sides."
  ],
  "function_level_applicability": {
    "cauchy_linear_fit": "APPLICABLE_SUBROUTINE(real_full_rank_design_and_1d_target)",
    "fit_anchored_relaxation": "NOT_NEEDED_FOR_STATIC_REGRESSION(1d_monotone_time_relaxation_only)",
    "fit_local_level_step": "NOT_NEEDED_FOR_STATIC_REGRESSION(1d_step_split_only)"
  },
  "functions": [
    {
      "function": "cauchy_linear_fit",
      "signature": "cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float]",
      "doc_summary": "Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank before and after weighting, and positive scale if supplied and positive floor, returning (coefficients (p,), floored residual MAD)."
    }
  ],
  "reference_source_txt": "/tmp/hegel_knowledge_ref/autogenous_prim_robust_trend_decomposition.reference.txt"
}
/tmp/hegel_knowledge_ref/autogenous_prim_box_ridge_bounds.reference.txt
/tmp/hegel_knowledge_ref/autogenous_prim_robust_trend_decomposition.reference.txt
/tmp/hegel_knowledge_ref/autogenous_sparse_ridge_certification.reference.txt
/tmp/hegel_knowledge_ref/autogenous_prim_reference_panel_balancing.reference.txt
```

### Turn 3 — `execute_in_remote_docker` (`21715 ms` LLM)
- **Rationale**: Read the exact minorant replay and Cauchy update before writing the solver. The bound algebra determines the certificate payload, and the installed libraries determine which numerical routines can be used.
- **Alternatives Weighed**: `Use the small-p solver unchanged` (rejected: Its documented applicability stops at 16 predictors; this contract requires 18 or 19 in a single connected block.)

```bash
ls -la /pretrain_task /holdout_task 2>/dev/null; for f in /tmp/hegel_knowledge_ref/{autogenous_prim_box_ridge_bounds,autogenous_sparse_ridge_certification,autogenous_prim_reference_panel_balancing,autogenous_prim_robust_trend_decomposition}.reference.txt; do printf '\n--- %s ---\n' "$f"; grep -nE '^def |minorant_center|dual_nu|def .*lower|fixed_active|MAD' "$f" | head -45; done; sed -n '/^def box_ridge_cube_lower_bound/,/^def /p' /tmp/hegel_knowledge_ref/autogenous_prim_box_ridge_bounds.reference.txt | head -125; sed -n '/^def cauchy_linear_fit/,/^def /p' /tmp/hegel_knowledge_ref/autogenous_prim_robust_trend_decomposition.reference.txt | head -90; python3 -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)'
```
- **Container Exit Code**: `0` (`664 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
l2) = map(ld, (box, activation_cost, ridge))
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
def cauchy_linear_fit(design: np.ndarray, values: np.ndarray, *, scale: float | None=None, noise_floor: float=1e-08) -> tuple[np.ndarray, float]:
    """Fit finite design (n,p) and values (n,) with n >= p >= 1, full column rank before and after weighting, and positive scale if supplied and positive floor, returning (coefficients (p,), floored residual MAD)."""
    (a, y) = (_real(design), _real(values))
    noise_floor = _scalar(noise_floor)
    if scale is not None:
        scale = _scalar(scale)
    if a.ndim != 2 or not 1 <= a.shape[1] <= a.shape[0] or y.shape != (a.shape[0],):
        raise ValueError('Expected design (n,p) and values (n,), with n >= p >= 1.')
    if noise_floor <= 0 or (scale is not None and scale <= 0):
        raise ValueError('Scale and noise floor must be positive.')
    (beta, _, rank, _) = np.linalg.lstsq(a, y, rcond=None)
    if rank != a.shape[1]:
        raise ValueError('Design must have full column rank.')
    if scale is None:
        scale = max(noise_floor, _mad(y - a @ beta))
    for _ in range(12):
        root_weight = scale / np.hypot(scale, y - a @ beta)
        (beta, _, rank, _) = np.linalg.lstsq(a * root_weight[:, None], y * root_weight, rcond=None)
        if rank != a.shape[1]:
            raise ValueError('Weighted design must have full column rank.')
    return (beta, max(noise_floor, _mad(y - a @ beta)))


def fit_anchored_relaxation(time: np.ndarray, predictor: np.ndarray, values: np.ndarray, *, onset_rate: float, noise_scale: float, curvature_ratio: float, bins: int, tau_bounds: tuple[float, float]) -> tuple[float, float, float | None]:
1.26.4 1.13.0
```

### Turn 4 — `execute_in_remote_docker` (`82337 ms` LLM)
- **Rationale**: Implement the three stages and use the convex envelope to choose minorant centers, while keeping each leaf independently replayable. Next test partition coverage, bound replay, and the leaf and runtime limits on connected designs.
- **Alternatives Weighed**: `Enumerate all supports and return their bounds` (rejected: A fully enumerated partition cannot meet the 2048-leaf limit for a connected 18- or 19-column study.)

```bash
mkdir -p /holdout_task
cat > /holdout_task/coupled_interferometry_certified_l0.py <<'PY'
"""Reference-calibrated interferometry and certified boxed sparse regression."""

import heapq

import numpy as np
from scipy.optimize import lsq_linear


def _align(panel, strata):
    groups, center = np.unique(panel['center_id'], return_inverse=True)
    values = np.array(panel['X_panel_raw'], dtype=np.float64, copy=True)
    controls = np.asarray(panel['controls'], dtype=np.float64)
    for c in range(groups.size):
        rows = center == c
        ref = controls[rows]
        finite = np.isfinite(ref)
        valid = finite.sum(axis=0) >= 2
        ref = np.where(finite[:, valid], ref[:, valid], np.nan)
        offset = np.median(np.nanmean(ref, axis=0))
        gain = np.median(np.nanstd(ref, axis=0))
        values[rows] = (values[rows] - offset) / gain
    cell = 3 * center + strata
    counts = np.bincount(cell, minlength=3 * groups.size)
    w = (cell.size / (3 * groups.size * counts[cell]))[:, None]
    finite = np.isfinite(values)
    fill = np.sum(w * np.where(finite, values, 0), axis=0) / np.sum(w * finite, axis=0)
    return np.where(finite, values, fill)


def _cauchy(design, y):
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    # Re-estimate scale during warm-up so glitches cannot permanently inflate it.
    for i in range(100):
        r = y - design @ beta
        if i < 15:
            sigma = max(1e-8, 1.4826 * np.median(np.abs(r - np.median(r))))
        scale = 1.5 * sigma
        root_w = scale / np.hypot(scale, r)
        new = np.linalg.lstsq(design * root_w[:, None], y * root_w, rcond=None)[0]
        delta = np.linalg.norm(new - beta)
        beta = new
        if i >= 15 and delta < 1e-10 * (1 + np.linalg.norm(beta)):
            break
    return beta


class _BoxSearch:
    def __init__(self, x, y, l0, l2, box):
        self.x, self.y = x, y
        self.l0, self.l2, self.box = l0, l2, box
        self.p = x.shape[1]
        self.bits = 1 << np.arange(self.p)
        self.full = (1 << self.p) - 1
        g = x.T @ x
        self.g = (g + g.T) * 0.5
        self.b = x.T @ y
        self.h = self.g + l2 * np.eye(self.p)
        self.hdiag = self.h.diagonal().copy()
        # Normalize the Gram matrix before extracting a PSD diagonal minorant.
        norm = np.sqrt(np.maximum(self.g.diagonal(), np.finfo(float).tiny))
        corr = self.g / norm[:, None] / norm[None, :]
        rho = max(0., np.linalg.eigvalsh(corr)[0] - 1e-11)
        self.d = rho * self.g.diagonal()
        self.k = self.d + l2
        self.koff = self.g - np.diag(self.d)
        self.q = self.koff.diagonal().copy()
        self.t = np.minimum(box, np.sqrt(np.divide(2*l0, self.k, out=np.full(self.p, np.inf), where=self.k > 0)))
        self.slope = l0 / np.where(self.t > 0, self.t, 1.0) + 0.5 * self.k * self.t
        self.switch = self.slope + self.q * self.t
        self.xl = x.astype(np.longdouble)
        self.yl = y.astype(np.longdouble)
        self.dl = self.d.astype(np.longdouble)
        self.upper = float(0.5 * (y @ y))
        self.best = np.zeros(self.p)
        self.seen = set()

    def _try_support(self, support):
        mask = int(self.bits[support].sum())
        if mask in self.seen:
            return
        self.seen.add(mask)
        if not mask:
            return
        hs = self.h[np.ix_(support, support)]
        try:
            w = np.linalg.solve(hs, self.b[support])
        except np.linalg.LinAlgError:
            w = np.full(int(support.sum()), np.inf)
        if np.max(np.abs(w)) > self.box:
            xs = self.x[:, support]
            ys = self.y
            if self.l2:
                xs = np.vstack((xs, np.sqrt(self.l2)*np.eye(xs.shape[1])))
                ys = np.r_[ys, np.zeros(xs.shape[1])]
            w = lsq_linear(xs, ys, bounds=(-self.box, self.box), method='bvls', tol=1e-12).x
        w = np.clip(w, -self.box, self.box)
        w[np.abs(w) <= 1e-8] = 0.
        r = self.x[:, support] @ w - self.y
        val = float(0.5*(r @ r + self.l2*(w @ w)) + self.l0*np.count_nonzero(w))
        if val < self.upper:
            self.upper = val
            self.best = np.zeros(self.p)
            self.best[support] = w

    def _relax(self, zero, active, initial):
        # Convex envelope of each diagonal quadratic plus its activation cost.
        # Its tangent dual is exactly the separable minorant recorded in the leaf.
        w = initial.copy()
        w[zero] = 0.
        residual = self.b - self.koff @ w
        order = np.flatnonzero(~zero).tolist()
        q, h, slope, switch, box = self.q, self.hdiag, self.slope, self.switch, self.box
        for _ in range(100):
            movement = 0.
            for j in order:
                z = residual[j] + q[j] * w[j]
                az = abs(z)
                if active[j] or az > switch[j]:
                    v = min(box, az / h[j]) if h[j] > 0 else box
                elif az <= slope[j]:
                    v = 0.
                else:
                    v = min(box, (az - slope[j]) / q[j]) if q[j] > 0 else box
                if z < 0.:
                    v = -v
                dv = v - w[j]
                if dv:
                    residual -= dv * self.koff[:, j]
                    w[j] = v
                    movement = max(movement, abs(dv))
            if movement < 2e-10 * (1. + box):
                break
        return w

    def _leaf(self, z, a, initial):
        zero = (z & self.bits) != 0
        active = (a & self.bits) != 0
        free = ~(zero | active)
        w = self._relax(zero, active, initial)
        magnitude = np.abs(w)
        self._try_support(active | (free & (magnitude > np.maximum(1e-8, 0.5*self.t))))
        if z == 0 and a == 0:
            self._try_support(magnitude > 1e-8)
        c = w.astype(np.longdouble)
        r = self.yl - self.xl @ c
        g = self.xl.T @ r
        grad = np.abs(g + self.dl*c)
        k = self.dl + self.l2
        v = np.minimum(self.box, np.divide(grad, k, out=np.full(self.p, self.box, dtype=np.longdouble), where=k>0))
        gain = v * (grad - 0.5*k*v)
        on = self.l0 - gain
        const = 0.5*(r @ r + (self.dl*c) @ c) + g @ c
        val = const + on[active].sum() + np.minimum(on[free], 0).sum()
        guard = 4e-13*(1 + abs(const) + np.abs(g) @ np.abs(c) + gain.sum() + self.p*self.l0)
        lb = float(val - guard)
        leaf = {'fixed_zero': zero.tolist(), 'fixed_active': active.tolist(), 'lb': lb,
                'minorant_diagonal': self.d.tolist(), 'minorant_center': w.tolist()}
        frac = np.divide(magnitude, self.t, out=np.zeros(self.p), where=self.t>0)
        score = np.minimum(frac, 1.-frac)
        score[~free] = -np.inf
        branch = int(np.argmax(score)) if np.any(free) else None
        return lb, leaf, branch, w

    def solve(self):
        heap = []

        def add(z, a, initial):
            lb, leaf, branch, w = self._leaf(z, a, initial)
            heapq.heappush(heap, (lb, z, a, leaf, branch, w))

        add(0, 0, self.best)
        while heap[0][0] < self.upper - 8e-6:
            if len(heap) >= 2048:
                raise RuntimeError('Box-L0 partition exceeded the certificate budget')
            _, z, a, _, j, w = heapq.heappop(heap)
            if j is None:
                raise RuntimeError('Box-L0 full-support bound did not close')
            bit = 1 << j
            add(z | bit, a, w)
            add(z, a | bit, w)
        lower = min(self.upper, heap[0][0])
        return {'weights': self.best.tolist(), 'objective': self.upper,
                'certified_lower_bound': lower, 'subcube_leaves': [node[3] for node in heap]}


def solve_coupled_interferometry_l0(study: dict) -> dict:
    """Align references, estimate the optical calibration, and certify support selection."""
    panel = study['panel']
    strata = np.asarray(panel['stratum'], dtype=int)
    aligned = _align(panel, strata)
    optics = study['interferometry']
    signal = np.asarray(optics['signal_phasor'], dtype=np.complex128)
    reference = np.asarray(optics['ref_phasor'], dtype=np.complex128)
    phase = np.angle(signal * np.conj(reference))
    h = np.column_stack((np.ones(strata.size), strata == 1, strata == 2, aligned, phase))
    y = np.asarray(study['y_assay_raw'], dtype=np.float64)
    cal = np.asarray(study['calibration_indices'])
    target = np.asarray(study['stage3_indices'])
    theta = _cauchy(h[cal], y[cal])
    deconfounded = y[target] - h[target] @ theta
    cert = _BoxSearch(np.asarray(study['X_cand'], dtype=np.float64), deconfounded,
                      float(study['lam0']), float(study['lam2']), float(study['M'])).solve()
    return {'aligned_panel': aligned, 'calibration_weights': theta,
            'deconfounded_target': deconfounded, 'certified_selection': cert}
PY
sed -n '20,100p' /tmp/hegel_knowledge_ref/autogenous_prim_reference_panel_balancing.reference.txt
ls -lh /holdout_task
```
- **Container Exit Code**: `0` (`354 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/coupled_interferometry_certified_l0.py']`
- **Stdout (tail)**:
```text
e mean-one weights (n,) from nonempty finite real vectors groups (n,) and strata (n,) with every observed group–stratum combination present."""
    groups = _real_array(groups, 'groups')
    strata = _real_array(strata, 'strata')
    if groups.ndim != 1 or not groups.size or strata.shape != groups.shape:
        raise ValueError('groups and strata must be matching nonempty vectors')
    if not np.isfinite(groups).all() or not np.isfinite(strata).all():
        raise ValueError('groups and strata must be finite')
    (group_levels, group_codes) = np.unique(groups, return_inverse=True)
    (stratum_levels, stratum_codes) = np.unique(strata, return_inverse=True)
    pairs = np.column_stack((group_codes, stratum_codes))
    (_, inverse, counts) = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    if counts.size != group_levels.size * stratum_levels.size:
        raise ValueError('every group must contain every observed stratum')
    return groups.size / counts.size / counts[inverse.reshape(-1)]


def weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return finite-entry weighted means (p,) for a nonempty real matrix values (n, p) with finite positive weights (n,), using zero for unobserved columns."""
    values = _real_array(values, 'values').astype(float, copy=False)
    weights = _real_array(weights, 'weights').astype(float, copy=False)
    if values.ndim != 2 or not values.shape[0] or (not values.shape[1]):
        raise ValueError('values must be a nonempty matrix')
    if weights.shape != (values.shape[0],):
        raise ValueError('weights must be a vector matching value rows')
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('weights must be finite and strictly positive')
    observed = np.isfinite(values)
    support = weights @ observed
    numerator = weights @ np.where(observed, values, 0.0)
    if not np.isfinite(support).all() or not np.isfinite(numerator).all():
        raise ValueError('weighted column sums must be finite')
    return np.divide(numerator, support, out=np.zeros(values.shape[1]), where=support > 0)
total 12K
-rw-r--r-- 1 root root 8.6K Nov 26 04:52 coupled_interferometry_certified_l0.py

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ea6eb8eea7bc253b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ea6eb8ee (directory).
```

### Turn 5 — `execute_in_remote_docker` (`65181 ms` LLM)
- **Rationale**: Replay the bounds and indicator partition independently and compare against exhaustive boxed fits at small p. If correlated 18- or 19-column designs exceed the leaf or runtime allowance, the measurements will identify where to strengthen the relaxation or branching.

```bash
cat > /tmp/verify_eq02.py <<'PY'
import importlib.util
import itertools
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import lsq_linear

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solution', root / 'coupled_interferometry_certified_l0.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def replay(x, y, l0, l2, box, cert):
    n, p = x.shape
    w = np.asarray(cert['weights'])
    leaves = cert['subcube_leaves']
    assert w.shape == (p,) and np.max(abs(w)) <= box and 1 <= len(leaves) <= 2048
    obj = .5 * (np.linalg.norm(x @ w - y)**2 + l2 * (w @ w)) + l0 * np.count_nonzero(abs(w) > 1e-8)
    assert_allclose(cert['objective'], obj, rtol=0, atol=2e-10)
    cube = np.arange(1 << p)
    covered = np.zeros(cube.size, bool)
    bits = 1 << np.arange(p)
    base = x.T @ x
    xy = x.T @ y
    eig = {}
    lower = np.inf
    min_slack = np.inf
    for leaf in leaves:
        zeros, actives = (leaf[k] for k in ('fixed_zero', 'fixed_active'))
        assert len(zeros) == len(actives) == p and all(type(i) is bool for i in zeros + actives)
        z, a = np.array(zeros), np.array(actives)
        assert not np.any(z & a)
        match = ((cube & bits[z].sum()) == 0) & ((cube & bits[a].sum()) == bits[a].sum())
        assert not covered[match].any()
        covered |= match
        free = ~(z | a)
        if 'dual_nu' in leaf:
            nu = np.asarray(leaf['dual_nu'])
            assert nu.shape == (n,)
            k, linear = l2, x.T @ nu
            constant = nu @ y - .5 * (nu @ nu)
        else:
            d, c = (np.asarray(leaf[k]) for k in ('minorant_diagonal', 'minorant_center'))
            assert d.shape == c.shape == (p,) and np.all(d >= 0)
            dt = tuple(d)
            if dt not in eig:
                eig[dt] = np.linalg.eigvalsh(base - np.diag(d))[0]
            assert eig[dt] >= -1e-10
            k, linear = d + l2, xy - (base - np.diag(d)) @ c
            constant = .5 * (y @ y - c @ (base - np.diag(d)) @ c)
        magnitude = abs(linear)
        amp = np.minimum(box, np.divide(magnitude, k, out=np.full(p, box), where=k > 0))
        active = l0 - amp * (magnitude - .5*k*amp)
        replayed = constant + active[a].sum() + np.minimum(active[free], 0).sum()
        assert leaf['lb'] <= replayed + 2e-9, (leaf['lb'], replayed)
        min_slack = min(min_slack, replayed - leaf['lb'])
        lower = min(lower, leaf['lb'])
    assert covered.all()
    assert cert['certified_lower_bound'] <= lower + 2e-9
    gap = obj - cert['certified_lower_bound']
    assert -2e-10 <= gap <= 1e-5
    return gap, len(leaves), min_slack

def design(rng, n, p, kind):
    x = rng.normal(size=(n,p))
    if kind == 'paired':
        x[:,1::2] = .97*x[:,:2*(p//2):2] + .245*x[:,1::2]
    elif kind == 'common':
        x = .3*x + rng.normal(size=(n,1))
    elif kind == 'ar':
        for j in range(1,p):
            x[:,j] = .88*x[:,j-1] + .475*x[:,j]
    elif kind == 'rankdef':
        x[:,-2:] = x[:,:2]
    return x / np.linalg.norm(x, axis=0)

def study_for(seed, p, kind):
    rng = np.random.default_rng(seed)
    cells = np.repeat(np.arange(15), rng.integers(15, 75, 15))
    rng.shuffle(cells)
    center, strata = cells // 3, cells % 3
    n = cells.size
    offsets = np.array([-2.0, .5, 1., -.8, 2.])
    gains = np.array([.5, .95, 1.6, .85, 1.2])
    true = rng.normal(size=(n,8)) + .4*strata[:,None]
    raw = true * gains[center,None] + offsets[center,None]
    raw[rng.random(raw.shape) < .025] = np.nan
    controls = rng.normal(size=(n,18)) * gains[center,None] + offsets[center,None]
    controls[rng.random(controls.shape) < .05] = np.nan
    phase = rng.uniform(-np.pi, np.pi, size=(n,2))
    carrier = rng.uniform(-np.pi, np.pi, size=(n,2))
    theta = rng.normal(size=13) * .4
    target = np.arange(n-110,n)
    cal = np.arange(n-110)
    x = design(rng, len(target), p, kind)
    w = np.zeros(p)
    w[rng.choice(p, 5, False)] = rng.choice([-1,1],5)*rng.uniform(.75,1.7,5)
    h = np.column_stack((np.ones(n), strata == 1, strata == 2, true, phase))
    y = h @ theta + rng.normal(0,.035,n)
    glitches = rng.choice(cal, int(.12*len(cal)), False)
    y[glitches] += np.clip(2*rng.standard_cauchy(len(glitches)), -11, 11)
    y[target] += x @ w
    study = dict(panel=dict(center_id=center, stratum=strata, controls=controls, X_panel_raw=raw), interferometry=dict(signal_phasor=np.exp(1j*(phase+carrier)), ref_phasor=np.exp(1j*carrier)), calibration_indices=cal, stage3_indices=target, y_assay_raw=y, X_cand=x, lam0=.14, lam2=.05, M=1.0)
    return study, theta, true

for seed, kind in enumerate(('iid','paired','ar','common','rankdef','paired','ar','common'), 31):
    p = 18 + seed % 2
    study, theta, true = study_for(seed, p, kind)
    t = perf_counter()
    try:
        result = mod.solve_coupled_interferometry_l0(study)
    except RuntimeError as error:
        print('FAIL', seed, kind, perf_counter()-t, str(error), flush=True)
        continue
    dt = perf_counter()-t
    cert = result['certified_selection']
    gap, leaves, slack = replay(study['X_cand'], result['deconfounded_target'], study['lam0'], study['lam2'], study['M'], cert)
    h = np.column_stack((np.ones(len(true)), study['panel']['stratum']==1, study['panel']['stratum']==2, result['aligned_panel'], np.angle(study['interferometry']['signal_phasor']*study['interferometry']['ref_phasor'].conj())))
    ix = study['stage3_indices']
    assert_allclose(result['deconfounded_target'], study['y_assay_raw'][ix] - h[ix] @ result['calibration_weights'], atol=1e-12)
    print(f'{seed} {kind} p={p}: panel_mae={np.mean(abs(result["aligned_panel"]-true)):.4f} theta_rel={np.linalg.norm(result["calibration_weights"]-theta)/np.linalg.norm(theta):.4f} gap={gap:.3g} leaves={leaves} time={dt:.4f} replay_slack={slack:.3g}', flush=True)

for seed in range(6):
    rng = np.random.default_rng(580+seed)
    x = design(rng, 25, 8, ('iid','ar','paired','common','rankdef','common')[seed])
    y = x @ rng.normal(size=8) + rng.normal(0,.2,25)
    l0, l2, box = (.07, .025, .4)
    cert = mod._BoxSearch(x, y, l0, l2, box).solve()
    gap, leaves, slack = replay(x, y, l0, l2, box, cert)
    values = [.5*y@y]
    for k in range(1,9):
        for s in itertools.combinations(range(8), k):
            aug = np.vstack((x[:,s], np.sqrt(l2)*np.eye(k)))
            sol = lsq_linear(aug, np.r_[y,np.zeros(k)], bounds=(-box,box), method='bvls').x
            values.append(.5*np.linalg.norm(aug@sol-np.r_[y,np.zeros(k)])**2 + l0*k)
    assert abs(cert['objective'] - min(values)) < 2e-10
    print(f'boxed exhaustive oracle {seed}: gap={gap:.3g} leaves={leaves}', flush=True)
PY
if test -f /holdout_task/validate_public_schema.py; then python3 /holdout_task/validate_public_schema.py; else echo 'Public validator not mounted'; fi
python3 -I /tmp/verify_eq02.py
```
- **Container Exit Code**: `0` (`2980 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public validator not mounted
31 iid p=19: panel_mae=0.0532 theta_rel=0.0317 gap=6.2e-12 leaves=3 time=0.0089 replay_slack=5.86e-12
32 paired p=18: panel_mae=0.0516 theta_rel=0.0173 gap=7.03e-06 leaves=15 time=0.2625 replay_slack=2.58e-12
33 ar p=19: panel_mae=0.0466 theta_rel=0.0256 gap=1.34e-07 leaves=6 time=0.1966 replay_slack=2.39e-12
34 common p=18: panel_mae=0.0488 theta_rel=0.0156 gap=2.74e-12 leaves=9 time=0.0502 replay_slack=2.74e-12
35 rankdef p=19: panel_mae=0.0420 theta_rel=0.0399 gap=4.65e-12 leaves=25 time=0.0420 replay_slack=4.07e-12
36 paired p=18: panel_mae=0.0406 theta_rel=0.0118 gap=2.11e-12 leaves=10 time=0.1701 replay_slack=2.11e-12
37 ar p=19: panel_mae=0.0356 theta_rel=0.0148 gap=2.98e-12 leaves=38 time=0.2974 replay_slack=2.81e-12
38 common p=18: panel_mae=0.0346 theta_rel=0.0054 gap=5.24e-12 leaves=19 time=0.0255 replay_slack=3.87e-12
boxed exhaustive oracle 0: gap=4.08e-12 leaves=1
boxed exhaustive oracle 1: gap=1.48e-12 leaves=7
boxed exhaustive oracle 2: gap=1.27e-12 leaves=2
boxed exhaustive oracle 3: gap=5.87e-12 leaves=1
boxed exhaustive oracle 4: gap=2.67e-12 leaves=9
boxed exhaustive oracle 5: gap=2.15e-11 leaves=1

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ea6eb8eea7bc253b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ea6eb8ee (directory).
```

### Turn 6 — `submit_completed_task` (`16710 ms` LLM)
- **Rationale**: An isolated Python process verified the final artifact on connected 18- and 19-column designs, including correlated and rank-deficient cases. Independent replay confirmed complete, disjoint partitions and valid leaf bounds; the largest certificate used 38 leaves.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`12` turns remaining; artifact status={'/holdout_task/coupled_interferometry_certified_l0.py': 'FILE:8736'}): Do NOT lock prematurely with `12` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_interferometry_certified_l0.py`. Eight synthetic connected studies passed certificate replay, and six smaller cases matched exhaustive boxed optimization. The public studies and validator were not mounted, so public validation remains unverified.`

### Turn 7 — `execute_in_remote_docker` (`30602 ms` LLM)
- **Rationale**: Look for the public measurements once more. On synthetic data, test whether correlated, nearly tied supports exhaust the certificate budget and whether holding out a center exposes a calibration weakness relative to OLS. Classification and relay checks do not apply to this regression contract.

```bash
find /holdout_task /pretrain_task /workspace /app /mnt /data -maxdepth 4 -type f \( -name 'public_eq02_studies.pkl' -o -name 'validate_public_schema.py' \) -print 2>/dev/null; cat > /tmp/audit_eq02.py <<'PY'
from pathlib import Path
exec(Path('/tmp/verify_eq02.py').read_text().split('for seed, kind in enumerate')[0])

# Targets near inclusion boundaries and reduced-rank blocks can reveal weak leaf bounds.
for seed, kind in enumerate(('paired','ar','common','rankdef')*3, 90):
    p = 18 + seed % 2
    rng = np.random.default_rng(seed)
    x = design(rng, 48, p, kind)
    box = (.20, .55, 1.2)[(seed-90)//4]
    l2, l0 = .005, .50*box**2
    w = box*rng.choice([-1.,1.],p)*rng.uniform(.7,1.3,p)
    if kind == 'paired':
        w[1::2] = -w[:2*(p//2):2]
    y = x @ w + rng.normal(0,.15*box,x.shape[0])
    start = perf_counter()
    try:
        cert = mod._BoxSearch(x,y,l0,l2,box).solve()
    except RuntimeError as e:
        print(f'{seed} {kind}: FAIL {perf_counter()-start:.3f}s {e}', flush=True)
        continue
    dt = perf_counter()-start
    gap, leaves, slack = replay(x,y,l0,l2,box,cert)
    print(f'{seed} {kind}: {dt:.3f}s gap={gap:.3g} leaves={leaves}', flush=True)

# Check whether a single bad reference center dominates held-out calibration prediction.
for seed in (31, 35, 38):
    study, theta, _ = study_for(seed, 19, 'paired')
    strata = study['panel']['stratum']
    groups = study['panel']['center_id']
    phase = np.angle(study['interferometry']['signal_phasor']*study['interferometry']['ref_phasor'].conj())
    h = np.column_stack((np.ones(len(strata)),strata==1,strata==2,mod._align(study['panel'],strata),phase))
    errors = []
    for c in np.unique(groups):
        cal = study['calibration_indices']
        train = cal[groups[cal]!=c]
        test = cal[groups[cal]==c]
        y = study['y_assay_raw']
        robust = mod._cauchy(h[train],y[train])
        ols = np.linalg.lstsq(h[train],y[train],rcond=None)[0]
        errors.append((np.median(abs(y[test]-h[test]@robust)),np.median(abs(y[test]-h[test]@ols)),np.linalg.norm(robust-theta)/np.linalg.norm(theta)))
    print('held-out center',seed,'worst [Cauchy MedAE, OLS MedAE, Cauchy theta rel L2]',np.max(errors,axis=0).round(5).tolist())
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_eq02.py
```
- **Container Exit Code**: `0` (`1247 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
90 paired: 0.001s gap=5.69e-13 leaves=1
91 ar: 0.016s gap=7.49e-13 leaves=9
92 common: 0.013s gap=7.45e-06 leaves=14
93 rankdef: 0.033s gap=3.51e-06 leaves=35
94 paired: 0.000s gap=1.95e-12 leaves=1
95 ar: 0.016s gap=3.34e-12 leaves=11
96 common: 0.008s gap=7.51e-06 leaves=10
97 rankdef: 0.047s gap=3.58e-12 leaves=41
98 paired: 0.001s gap=7.45e-12 leaves=1
99 ar: 0.021s gap=1.11e-11 leaves=9
100 common: 0.000s gap=7.11e-12 leaves=1
101 rankdef: 0.071s gap=1.35e-11 leaves=43
held-out center 31 worst [Cauchy MedAE, OLS MedAE, Cauchy theta rel L2] [0.07595, 0.34022, 0.03662]
held-out center 35 worst [Cauchy MedAE, OLS MedAE, Cauchy theta rel L2] [0.08959, 0.30199, 0.04953]
held-out center 38 worst [Cauchy MedAE, OLS MedAE, Cauchy theta rel L2] [0.04631, 0.2612, 0.01133]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ea6eb8eea7bc253b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ea6eb8ee (directory).
```

### Turn 8 — `submit_completed_task` (`15682 ms` LLM)
- **Rationale**: Independent replay in clean Python processes verified disjoint and complete support partitions on connected, correlated, rank-deficient, and near-tied designs. The largest certificate used 43 leaves, and the final artifact remained unchanged after verification.
- **Alternatives Weighed**: `Replace or retune the Cauchy calibration` (rejected: Cauchy calibration had lower worst-center median absolute error than OLS in all three synthetic audits, and its held-out coefficient error stayed within tolerance.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/coupled_interferometry_certified_l0.py': 'FILE:8736'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/coupled_interferometry_certified_l0.py`. All 20 synthetic connected cases passed certificate replay, and six smaller cases matched exhaustive optimization. The public studies and validator were not mounted, so public validation remains unverified.`

### Turn 9 — `execute_in_remote_docker` (`28473 ms` LLM)
- **Rationale**: Test a distinct failure mode: many simultaneously ambiguous inclusion decisions can exhaust the leaf or runtime budget even when individual near-ties pass. If it fails, strengthen the bound or branching before submission. Run the public validator if its fixtures are now mounted.

```bash
cat > /tmp/eq02_boundary.py <<'PY'
from pathlib import Path
exec(Path('/tmp/verify_eq02.py').read_text().split('for seed, kind in enumerate')[0])

public = root / 'public_eq02_studies.pkl'
validator = root / 'validate_public_schema.py'
if public.is_file() and validator.is_file():
    import subprocess, sys
    subprocess.run([sys.executable, '-I', str(validator)], cwd=root, check=True)
else:
    print('Public fixtures unavailable; the following are synthetic tolerance tests.', flush=True)

# Weakly correlated predictors with all marginal scores near the activation
# boundary can require a larger partition than sparse-signal examples.
worst = []
for i in range(12):
    rng = np.random.default_rng(20260+i)
    p, n = 18 + i%2, 96
    common = rng.normal(size=(n,1))
    x = rng.normal(size=(n,p)) + (0.18 + 0.15*(i%4))*common*rng.choice([-1,1],p)
    x /= np.linalg.norm(x, axis=0)
    l0, l2, box = .125, .015, (.45 if i%3 == 0 else 1.)
    b = .50 * rng.choice([-1,1],p) * rng.uniform(.94,1.06,p)
    y = x @ np.linalg.solve(x.T @ x,b)
    t = perf_counter()
    try:
        cert = mod._BoxSearch(x,y,l0,l2,box).solve()
        dt = perf_counter()-t
        gap, leaves, slack = replay(x,y,l0,l2,box,cert)
        worst.append((dt,i,gap,leaves))
        print(f'case={i} p={p} M={box}: gap={gap:.4g} leaves={leaves} seconds={dt:.3f}', flush=True)
    except RuntimeError as e:
        print(f'case={i} p={p} M={box}: FAIL after {perf_counter()-t:.3f}s: {e}', flush=True)
print('two-slowest combined seconds:', sum(z[0] for z in sorted(worst,reverse=True)[:2]), flush=True)
PY
python3 -I /tmp/eq02_boundary.py
```
- **Container Exit Code**: `0` (`6702 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public fixtures unavailable; the following are synthetic tolerance tests.
case=0 p=18 M=0.45: gap=3.441e-12 leaves=31 seconds=0.232
case=1 p=19 M=1.0: gap=3.21e-12 leaves=16 seconds=0.389
case=2 p=18 M=1.0: gap=3.078e-06 leaves=15 seconds=0.375
case=3 p=19 M=0.45: gap=4.579e-12 leaves=20 seconds=0.399
case=4 p=18 M=1.0: gap=3.523e-12 leaves=87 seconds=1.480
case=5 p=19 M=1.0: gap=3.467e-12 leaves=14 seconds=0.276
case=6 p=18 M=0.45: gap=2.51e-11 leaves=7 seconds=0.018
case=7 p=19 M=1.0: gap=4.505e-12 leaves=11 seconds=0.117
case=8 p=18 M=1.0: gap=2.78e-12 leaves=27 seconds=0.400
case=9 p=19 M=0.45: gap=4.035e-12 leaves=31 seconds=0.278
case=10 p=18 M=1.0: gap=3.32e-12 leaves=19 seconds=0.295
case=11 p=19 M=1.0: gap=4.065e-12 leaves=18 seconds=0.298
two-slowest combined seconds: 1.880324193276465

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=ea6eb8eea7bc253b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_ea6eb8ee (directory).
```

### Turn 10 — `submit_completed_task` (`23666 ms` LLM)
- **Rationale**: An isolated Python process verified the unchanged final artifact. Independent replay confirmed complete, disjoint partitions and valid leaf bounds on correlated, rank-deficient, and near-threshold synthetic designs.
- **Alternatives Weighed**: `Strengthen the bound or branching rule` (rejected: The simultaneous near-threshold stress cases stayed within all tested limits, using at most 87 of the 2048 permitted leaves.)
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_interferometry_certified_l0.py`. All 32 synthetic connected cases passed independent certificate replay, and six smaller cases matched exhaustive boxed optimization. The public studies and schema validator were not mounted, so public validation remains unverified.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/coupled_interferometry_certified_l0.py": "FILE:8736"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "all_valid": true,
    "max_panel_mae": 0.045663,
    "max_calibration_rel_l2": 0.01807,
    "max_deconfounded_mae": 0.200075,
    "max_stage3_w_vs_clean_rel_l2": 0.171072,
    "max_duality_gap": 5.226e-06,
    "student_elapsed_sec": 0.088,
    "instance_diagnostics": [
      {
        "seed": 99411,
        "p": 18,
        "panel_mae": 0.045663,
        "cal_rel_l2": 0.017083,
        "deconf_mae": 0.177276,
        "w_vs_clean_rel_l2": 0.171072,
        "cert_ok": true,
        "cert_reason": "ok"
      },
      {
        "seed": 99412,
        "p": 19,
        "panel_mae": 0.045333,
        "cal_rel_l2": 0.01807,
        "deconf_mae": 0.200075,
        "w_vs_clean_rel_l2": 0.095582,
        "cert_ok": true,
        "cert_reason": "ok"
      }
    ]
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_panel_mae\": 0.045663, \"max_calibration_rel_l2\": 0.01807, \"max_deconfounded_mae\": 0.200075, \"max_stage3_w_vs_clean_rel_l2\": 0.171072, \"max_duality_gap\": 5.226e-06, \"student_elapsed_sec\": 0.088, \"instance_diagnostics\": [{\"seed\": 99411, \"p\": 18, \"panel_mae\": 0.045663, \"cal_rel_l2\": 0.017083, \"deconf_mae\": 0.177276, \"w_vs_clean_rel_l2\": 0.171072, \"cert_ok\": true, \"cert_reason\": \"ok\"}, {\"seed\": 99412, \"p\": 19, \"panel_mae\": 0.045333, \"cal_rel_l2\": 0.01807, \"deconf_mae\": 0.200075, \"w_vs_clean_rel_l2\": 0.095582, \"cert_ok\": true, \"cert_reason\": \"ok\"}]}\n__HEGEL_VERIFIER_AUTH_d1c9b51f1186bd55c0316463443418a6__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_panel_mae\": 0.045663, \"max_calibration_rel_l2\": 0.01807, \"max_deconfounded_mae\": 0.200075, \"max_stage3_w_vs_clean_rel_l2\": 0.171072, \"max_duality_gap\": 5.226e-06, \"student_elapsed_sec\": 0.088, \"instance_diagnostics\": [{\"seed\": 99411, \"p\": 18, \"panel_mae\": 0.045663, \"cal_rel_l2\": 0.017083, \"deconf_mae\": 0.177276, \"w_vs_clean_rel_l2\": 0.171072, \"cert_ok\": true, \"cert_reason\": \"ok\"}, {\"seed\": 99412, \"p\": 19, \"panel_mae\": 0.045333, \"cal_rel_l2\": 0.01807, \"deconf_mae\": 0.200075, \"w_vs_clean_rel_l2\": 0.095582, \"cert_ok\": true, \"cert_reason\": \"ok\"}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5616.82,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
