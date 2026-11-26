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
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `356,966` (`343,115` / `74,112` / `13,851`) |
| **Total Episode Wall Time (s)** | `473.7s` |

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

### Turn 1 — `execute_in_remote_docker` (`15631 ms` LLM)
- **Rationale**: Read the validator to detect certificate-convention mismatches before implementation, and inspect only public study structure to choose compatible validation slices.

```bash
ls -lah /holdout_task; sed -n '1,260p' /holdout_task/validate_public_schema.py; if test -f /tmp/hegel_knowledge_ref/describe_ref.py; then python3 /tmp/hegel_knowledge_ref/describe_ref.py --search 'box L0 Cauchy IRLS reference panel'; fi; python3 - <<'PY'
import pickle
p='/holdout_task/public_eq02_studies.pkl'
x=pickle.load(open(p,'rb'))
print(type(x),len(x))
if isinstance(x,dict): print(list(x))
else:
 for a in x:
  print({k:getattr(v,'shape',type(v).__name__) for k,v in a.items()} if isinstance(a,dict) else type(a))
PY
```
- **Container Exit Code**: `1` (`440 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
rata (n,); >=1 control col with >=2 finite obs & spread>0 per group | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_reference_panel_balancing.balanced_cell_weights
autogenous_prim_robust_affine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar or scale array broadcastable to | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_robust_affine_projection.robust_residual_weights
autogenous_prim_reference_panel_balancing@a0bd4b96 | align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray | Return aligned real values (n, p) using real controls (n, q) and finite real g | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_reference_panel_balancing.align_reference_panel
autogenous_prim_reference_panel_balancing@a0bd4b96 | weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray | Return finite-entry weighted means (p,) for a nonempty real matrix values (n,  | REQ: discrete groups/strata (n,); >=1 control col with >=2 finite obs & spread>0 per group | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_reference_panel_balancing.weighted_column_fill
autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(design: np.ndarray, target: np.ndarray, center: np.ndarray, box: float, activation_cost: float, *, ridge: float=0.0, fixed_zero: np.ndarray | None=None, fixed_active: np.ndarray | None=None, diagonal: np.ndarray | None=None) -> float | Return a scalar box-ridge lower bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant
```
- **Stderr (tail)**:
```text
ls: cannot access '/holdout_task': No such file or directory
sed: can't read /holdout_task/validate_public_schema.py: No such file or directory
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
FileNotFoundError: [Errno 2] No such file or directory: '/holdout_task/public_eq02_studies.pkl'
```

### Turn 2 — `execute_in_remote_docker` (`17655 ms` LLM)
- **Rationale**: Locate the public or pretraining fixtures and read the relevant reference formulas. This determines whether to validate against the supplied schema or use an independent synthetic oracle until the public files are available.
- **Alternatives Weighed**: `Assume the public validator's dual convention` (rejected: A sign or objective-offset mismatch could invalidate otherwise correct certificates; first locate the validator, then use the documented quadratic-minorant form if it is unavailable.)

```bash
pwd; ls -lah /; ls -lah /pretrain_task /workspace /app /results 2>/dev/null; find /pretrain_task /workspace /app /results /tmp -maxdepth 3 -type f \( -name '*eq02*' -o -name '*schema*' -o -name '*.reference.txt' \) -print 2>/dev/null | head -70; r=/tmp/hegel_knowledge_ref/describe_ref.py; python3 "$r" --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound; python3 "$r" --describe autogenous_prim_reference_panel_balancing.align_reference_panel; python3 "$r" --describe autogenous_prim_robust_affine_projection.robust_residual_weights; python3 "$r" --search 'dual nu certified box l0 perspective'
```
- **Container Exit Code**: `0` (`610 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
er bound for design (n, p), target (n,), and cent | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound
autogenous_prim_box_ridge_bounds@18dcec21 | gram_diagonal_minorant(design: np.ndarray) -> np.ndarray | Return a nonnegative vector d (p,) for design X (n, p) such that X | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_box_ridge_bounds.gram_diagonal_minorant
autogenous_prim_box_ridge_bounds@18dcec21 | row_incidence_blocks(design: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray] | Return (blocks, uncovered_rows) for design (n, p), with each block a pair of 1 | REQ: G-diag(d)>=0 PSD minorant (NO ridge in minorant); box>=0, l0>=0; finite arrays | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_box_ridge_bounds.row_incidence_blocks
autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
autogenous_prim_conservative_spd_solve@05cb4387 | conservative_grid_operator(cell_coeff: np.ndarray, *, spacing: float | np.ndarray=1.0) -> sparse.csr_matrix | Map positive cell coefficients of shape (n1,  | REQ: grid op assumes isotropic scalar coeff>0 + zero Dirichlet ghosts; CG/AMG need square SPD | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_conservative_spd_solve.conservative_grid_operator
autogenous_prim_conservative_spd_solve@05cb4387 | galerkin_preconditioner(operator: sparse.spmatrix | sparse.sparray, *, strength_cutoff: float=0.08, max_coarse_fraction: float=0.76, coarse_size: int=64, seed: int=12) -> LinearOperator | Map a real sparse SPD matrix of shape (n, n) to a symmetric (n, n) V-cycle app | REQ: grid op assumes isotropic scalar coeff>0 + zero Dirichlet ghosts; CG/AMG need square SPD | python3 /tmp/hegel_knowledge_ref/describe_ref.py --describe autogenous_prim_conservative_spd_solve.galerkin_preconditioner
```

### Turn 3 — `execute_in_remote_docker` (`25733 ms` LLM)
- **Rationale**: Check the reference source for dual signs, ridge placement, and per-state activation terms before implementing the certificate evaluator. Confirm available numerical libraries so the solver and independent oracle can run in isolation.
- **Alternatives Weighed**: `Apply the catalog solver unchanged` (rejected: Its advertised limit is 16 predictors; the required connected problems have 18 or 19.)

```bash
r=/tmp/hegel_knowledge_ref; python3 "$r/describe_ref.py" --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | tail -25; grep -nE '^(def |class )|dual_nu|subcube_leaves|schema' "$r/autogenous_sparse_ridge_certification.reference.txt"; sed -n '1,220p' "$r/autogenous_prim_box_ridge_bounds.reference.txt"; grep -nE '^(def |class )|nan|ddof|median|bincount' "$r/autogenous_prim_reference_panel_balancing.reference.txt"; grep -nE '^(def |class )|cauchy|1\.0 /' "$r/autogenous_prim_robust_affine_projection.reference.txt"; python3 -I -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)'
```
- **Container Exit Code**: `0` (`662 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
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
13:def _real_array(x: np.ndarray, name: str) -> np.ndarray:
20:def align_reference_panel(values: np.ndarray, controls: np.ndarray, groups: np.ndarray) -> np.ndarray:
31:    result = np.full(values.shape, np.nan)
44:        (offset, gain) = (np.median(means), np.median(spreads))
48:        result[rows] = (np.where(np.isfinite(block), block, np.nan) - offset) / gain
52:def balanced_cell_weights(groups: np.ndarray, strata: np.ndarray) -> np.ndarray:
69:def weighted_column_fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
13:def propagate_linear_basis(transition: np.ndarray, initial: np.ndarray, increments: np.ndarray) -> np.ndarray:
30:def profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
64:def robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray:
73:        return 1.0 / np.maximum(1.0, u)
74:    if loss == 'cauchy':
75:        inverse = 1.0 / np.hypot(1.0, u)
77:    raise ValueError('loss must be huber or cauchy')
1.26.4 1.13.0
```

### Turn 4 — `execute_in_remote_docker` (`20910 ms` LLM)
- **Rationale**: Read the solver and diagonal formulas without output truncation to confirm the certificate conventions and distinguish reusable bounding methods from assumptions specific to the smaller reference problem.
- **Alternatives Weighed**: `Use only a box-ridge fit as each minorant center` (rejected: Its bound can be loose on free indicators; a perspective relaxation selects a center that accounts for ambiguous activations.)

```bash
sed -n '1,210p' /tmp/hegel_knowledge_ref/autogenous_sparse_ridge_certification.reference.txt; sed -n '38,92p' /tmp/hegel_knowledge_ref/autogenous_prim_box_ridge_bounds.reference.txt
```
- **Container Exit Code**: `0` (`339 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
9 = 0.0
    if diagonal is not None:
        _X_chk = np.asarray(design, dtype=float)
        _d_chk = np.asarray(diagonal, dtype=float).ravel()
        if _X_chk.ndim != 2 or _d_chk.size != _X_chk.shape[1] or not np.all(np.isfinite(_d_chk)) or not np.all(np.isfinite(_X_chk)):
            raise ValueError('diagonal must be a finite 1-D vector matching design columns')
        _p_chk = _X_chk.shape[1]
        if _p_chk > 0:
            _col_sq = np.sum(_X_chk * _X_chk, axis=0)
            if np.any(_d_chk > _col_sq + 1e-9 * np.maximum(1.0, _col_sq)):
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
```

### Turn 5 — `execute_in_remote_docker` (`105383 ms` LLM)
- **Rationale**: Implement the causal handoff and independently checkable bounds. Compilation catches artifact errors before numerical testing; the next step is to replay the certificate and compare against a small exact oracle, then stress the connected 18- and 19-column cases.
- **Alternatives Weighed**: `Enumerate all supports` (rejected: Even an exact optimum from enumeration would not provide the required compact certificate.)

```bash
mkdir -p /holdout_task; cat > /holdout_task/coupled_interferometry_certified_l0.py <<'PY'
"""Reference-panel calibration and certified connected box-L0 selection."""

import heapq
import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize


def _aligned(panel):
    groups, group = np.unique(panel['center_id'], return_inverse=True)
    stratum = np.asarray(panel['stratum'], dtype=int)
    raw = np.asarray(panel['X_panel_raw'], dtype=np.float64)
    controls = np.asarray(panel['controls'], dtype=np.float64)
    aligned = np.empty_like(raw)
    for i in range(len(groups)):
        rows = group == i
        ref = controls[rows]
        finite = np.isfinite(ref)
        keep = finite.sum(axis=0) >= 2
        ref = np.where(finite[:, keep], ref[:, keep], np.nan)
        offset = np.median(np.nanmean(ref, axis=0))
        gain = np.median(np.nanstd(ref, axis=0, ddof=0))
        aligned[rows] = (raw[rows] - offset) / gain
    cell = 3 * group + stratum
    weight = len(cell) / (3 * len(groups) * np.bincount(cell)[cell])
    ok = np.isfinite(aligned)
    fill = (weight @ np.where(ok, aligned, 0.0)) / (weight @ ok)
    return np.where(ok, aligned, fill)


def _cauchy(H, y):
    theta = np.linalg.lstsq(H, y, rcond=None)[0]
    floor = 1e-10 * max(1.0, np.median(np.abs(y)))
    # Start with broader downweighting, then adapt the scale to inlier scatter.
    for scale_factor, limit in ((3.0, 14), (1.7, 24), (1.35, 42)):
        for _ in range(limit):
            r = y - H @ theta
            mad = 1.4826 * np.median(np.abs(r - np.median(r)))
            root_weight = 1.0 / np.hypot(1.0, r / max(floor, scale_factor * mad))
            new = np.linalg.lstsq(root_weight[:, None] * H, root_weight * y, rcond=None)[0]
            if np.linalg.norm(new - theta) <= 1e-10 * (1 + np.linalg.norm(theta)):
                theta = new
                break
            theta = new
    return theta


def _box_l0(X, y, lam0, lam2, M):
    p = X.shape[1]
    bits = 1 << np.arange(p)
    full = (1 << p) - 1
    ld = np.longdouble
    xl, yl = X.astype(ld), y.astype(ld)
    gl, hl, cl = xl.T @ xl, xl.T @ yl, (yl @ yl) / 2
    G, h = gl.astype(float), hl.astype(float)
    A = G + lam2 * np.eye(p)
    scale = np.sqrt(np.maximum(0, np.diag(G)))
    safe_scale = np.where(scale > 0, scale, 1.)
    corr = G / safe_scale[:, None] / safe_scale[None, :]
    # Scaling the columns before taking the smallest eigenvalue gives a
    # nonnegative, scale-aware PSD minorant of G (without the ridge term).
    eig = np.linalg.eigvalsh(corr)[0]
    d = max(0., eig - 1e-10) * scale ** 2
    Q = G - np.diag(d)
    q = d + lam2
    lipschitz = max(1e-12, np.linalg.eigvalsh(Q)[-1])
    a = np.full(p, M)
    nz = q > 0
    a[nz] = np.minimum(M, np.sqrt(2 * lam0 / q[nz]))
    slope = np.divide(lam0, a, out=np.zeros(p), where=a > 0) + .5 * q * a
    # Worst-case magnitude of the expanded quadratic on the box, used for
    # outward rounding of the bounds obtained in extended precision.
    guard = float(64 * (max(X.shape) + 4) * np.finfo(ld).eps *
                  (1 + cl + M * np.sum(np.abs(xl).T @ np.abs(yl)) +
                   M * M * np.sum(np.abs(xl).T @ np.abs(xl)) + p * (lam0 + M*M*lam2)))
    guard += 2e-10

    def bound(center, zero, active):
        c = center.astype(ld)
        dd = d.astype(ld)
        g = hl - gl @ c
        const = cl - (c @ gl @ c) / 2 + (dd * c) @ c / 2
        linear = np.abs(g + dd * c)
        b = np.divide(linear, q.astype(ld), out=np.full(p, M, dtype=ld), where=(q > 0) & (linear < M * q))
        on = lam0 - b * (linear - q * b / 2)
        val = const + on[active].sum() + np.minimum(on[~(zero | active)], 0).sum()
        return float(np.nextafter(float(val) - guard, -np.inf))

    def relaxation(start, zero, active):
        # Eliminate relaxed indicators via the box perspective of q*w^2/2.
        # The resulting coordinate penalty is linear up to a and quadratic above it.
        free = ~(zero | active)
        threshold = np.where(free, slope / lipschitz, 0.)
        anchor = np.where(free, a, 0.)
        shrink = 1.0 / (1.0 + q / lipschitz)
        x = start.copy()
        x[zero] = 0.
        z = x.copy()
        momentum = 1.
        for _ in range(320):
            u = z + (h - Q @ z) / lipschitz
            mag = np.abs(u)
            linear = np.maximum(0., mag - threshold)
            new = np.sign(u) * np.minimum(M, np.where(linear <= anchor, linear, mag * shrink))
            new[zero] = 0.
            step = new - x
            if np.max(np.abs(step)) <= 2e-10 * (1 + M):
                return new
            if (z - new) @ step > 0:
                z, momentum = new, 1.
            else:
                nxt = .5 * (1. + np.sqrt(1. + 4. * momentum * momentum))
                z, momentum = new + ((momentum - 1.) / nxt) * step, nxt
            x = new
        return x

    best = np.zeros(p)
    upper = float(cl)
    tried = set()

    def primal(mask, start):
        nonlocal best, upper
        if mask in tried or not mask:
            return
        tried.add(mask)
        idx = np.flatnonzero(bits & mask)
        B, f = A[np.ix_(idx, idx)], h[idx]
        try:
            z = solve(B, f, assume_a='pos', check_finite=False)
        except np.linalg.LinAlgError:
            z = start[idx]
        if np.any(np.abs(z) > M) or np.max(np.abs(B @ z - f)) > 1e-7:
            def fun(v):
                g = B @ v
                return .5 * (v @ g) - f @ v, g - f
            z = minimize(fun, np.clip(z, -M, M), jac=True, method='L-BFGS-B',
                         bounds=[(-M, M)] * len(idx), options={'ftol': 1e-15, 'gtol': 1e-10}).x
            interior = np.abs(z) < M
            if interior.any():
                rhs = f[interior] - B[np.ix_(interior, ~interior)] @ z[~interior]
                try:
                    polish = solve(B[np.ix_(interior, interior)], rhs, assume_a='pos', check_finite=False)
                    if np.all(np.abs(polish) <= M):
                        z[interior] = polish
                except np.linalg.LinAlgError:
                    pass
        w = np.zeros(p)
        w[idx] = np.where(np.abs(z) > 1e-8, z, 0.)
        r = X @ w - y
        obj = .5 * (r @ r + lam2 * (w @ w)) + lam0 * np.count_nonzero(w)
        if obj < upper:
            upper, best = float(obj), w

    heap = []
    leaves = []
    serial = 0
    tol = 3e-6

    def visit(zmask, amask, start, parent_lb=-np.inf, parent_center=None):
        nonlocal serial
        zero, active = (bits & zmask) != 0, (bits & amask) != 0
        free = ~(zero | active)
        if parent_center is not None:
            pb = bound(parent_center, zero, active)
            if pb >= upper - tol:
                leaves.append((zero, active, pb, parent_center))
                return
        if not free.any():
            # Solve the fixed-support box QP even when the relaxation is slow.
            idx = np.flatnonzero(active)
            c = relaxation(start, zero, active)
            if idx.size:
                primal(amask, c)
                if np.array_equal(best != 0, active):
                    c = best
            lb = bound(c, zero, active)
            if lb < upper - tol:
                # Tighten this leaf using its own box fit rather than the incumbent.
                if idx.size:
                    B, f = A[np.ix_(idx, idx)], h[idx]
                    def fun(v):
                        g = B @ v
                        return .5 * v @ g - f @ v, g - f
                    c = c.copy()
                    c[idx] = minimize(fun, c[idx], jac=True, method='L-BFGS-B',
                                      bounds=[(-M, M)] * len(idx),
                                      options={'ftol': 1e-15, 'gtol': 1e-11}).x
                    lb = bound(c, zero, active)
            leaves.append((zero, active, lb, c))
            return
        c = relaxation(start, zero, active)
        lb = bound(c, zero, active)
        if parent_center is not None and pb > lb:
            lb, witness = pb, parent_center
        else:
            witness = c
        if lb < upper - tol:
            frac = np.divide(np.abs(c), a, out=np.ones(p), where=a > 0)
            for cutoff in ((.3, .6, .9) if not (zmask | amask) else (.55,)):
                primal(amask | int(bits[free & (frac >= cutoff)].sum()), c)
        if lb >= upper - tol:
            leaves.append((zero, active, lb, witness))
        else:
            serial += 1
            heapq.heappush(heap, (lb, serial, zmask, amask, c, witness))

    visit(0, 0, best)
    while heap:
        lb, _, zmask, amask, c, witness = heapq.heappop(heap)
        if lb >= upper - tol:
            leaves.append(((bits & zmask) != 0, (bits & amask) != 0, lb, witness))
            continue
        if len(heap) + len(leaves) + 2 > 2048:
            raise ArithmeticError('Box-L0 certificate exceeded 2048 leaves')
        free = (bits & (zmask | amask)) == 0
        frac = np.clip(np.divide(np.abs(c), a, out=np.ones(p), where=a > 0), 0., 1.)
        j = int(np.argmax(np.where(free, frac * (1. - frac), -1.)))
        bit = int(bits[j])
        visit(zmask | bit, amask, c, lb, witness)
        visit(zmask, amask | bit, c, lb, witness)

    lower = min(l[2] for l in leaves)
    if not 0 <= upper - lower <= 1e-5:
        raise ArithmeticError(f'Box-L0 certificate gap is {upper - lower:g}')
    diagonal = d.tolist()
    return {'schema': 'hegel_box_l0_cert_schema_v2',
            'weights': best.tolist(), 'objective': upper, 'certified_lower_bound': lower,
            'subcube_leaves': [dict(fixed_zero=z.tolist(), fixed_active=b.tolist(), lb=lb,
                                    minorant_diagonal=diagonal, minorant_center=c.tolist())
                               for z, b, lb, c in leaves]}


def solve_coupled_interferometry_l0(study: dict) -> dict:
    panel = study['panel']
    aligned = _aligned(panel)
    optical = study['interferometry']
    signal = np.asarray(optical['signal_phasor'], dtype=np.complex128)
    ref = np.asarray(optical['ref_phasor'], dtype=np.complex128)
    phase = np.angle(signal * np.conj(ref))
    s = np.asarray(panel['stratum'])
    H = np.column_stack((np.ones(len(s)), s == 1, s == 2, aligned, phase))
    y = np.asarray(study['y_assay_raw'], dtype=np.float64)
    cal = np.asarray(study['calibration_indices'])
    target = np.asarray(study['stage3_indices'])
    theta = _cauchy(H[cal], y[cal])
    residual = y[target] - H[target] @ theta
    selection = _box_l0(np.asarray(study['X_cand'], dtype=np.float64), residual,
                        float(study['lam0']), float(study['lam2']), float(study['M']))
    return {'aligned_panel': aligned, 'calibration_weights': theta,
            'deconfounded_target': residual, 'certified_selection': selection}
PY
python3 -I -m py_compile /holdout_task/coupled_interferometry_certified_l0.py; sed -n '110,300p' /tmp/hegel_knowledge_ref/autogenous_sparse_ridge_certification.reference.txt | grep -nE 'schema|certified_lower_bound|subcube_leaves|dual_nu|return '
```
- **Container Exit Code**: `0` (`484 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/coupled_interferometry_certified_l0.py']`
- **Stdout (tail)**:
```text
14:        return lb
17:        return ld(upper) - ld(eps)
22:        return {'beta': best, 'upper_bound': upper, 'tree_leaves': [{'fixed_zero': np.flatnonzero(zero & bits).tolist(), 'fixed_active': np.flatnonzero(active & bits).tolist(), 'lower_bound': lb} for ((zero, active), (lb, _)) in sorted(partition.items())]}
24:        return result({(0, 0): (residual_dual(h, 0, 0), best)})
29:            return result({(0, 0): (lb, b)})
66:            return result({(0, 0): (lb, combined)})
75:            return (witness(b, zero, active), b)
78:            return (lb, start)
109:            return (float((0.5 * hv + q) @ v), hv + q)
113:        return (trial, b) if trial > lb else (lb, start)
185:            return result(partition)
191:    return result({root: (lb, best)})

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bdfa466e92e1b23e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bdfa466e (directory).
```

### Turn 6 — `execute_in_remote_docker` (`50360 ms` LLM)
- **Rationale**: Replay every witness and the full Boolean partition in isolation, and compare small boxed problems with exhaustive support fits. The connected synthetic slices test calibration contamination, box saturation, and correlation; a failure identifies which stage or certificate mechanism needs revision.
- **Alternatives Weighed**: `Check only the returned objective and gap` (rejected: A plausible objective and gap would not detect overlapping subcubes or an invalid leaf witness.)

```bash
cat > /tmp/eq02_check.py <<'PY'
import importlib.util, itertools, os, pickle, time
import numpy as np
from scipy.optimize import lsq_linear
p='/holdout_task/coupled_interferometry_certified_l0.py'
sp=importlib.util.spec_from_file_location('eq02',p); mod=importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)

def replay(X,y,l0,l2,M,cert):
    n,p=X.shape; G=X.T@X; w=np.asarray(cert['weights']); err=X@w-y
    obj=.5*(err@err+l2*(w@w))+l0*np.count_nonzero(abs(w)>1e-8)
    assert max(abs(w))<=M and abs(obj-cert['objective'])<1e-9
    leaves=cert['subcube_leaves']; assert 1<=len(leaves)<=2048
    bits=1<<np.arange(p); covers=np.zeros(1<<p,dtype=np.uint8); seen_diag={}; replay_slack=0.
    for leaf in leaves:
        for key in ('fixed_zero','fixed_active'):
            assert len(leaf[key])==p and all(type(v) is bool for v in leaf[key])
        z,a=(np.array(leaf[key]) for key in ('fixed_zero','fixed_active')); assert not (z&a).any()
        free=~(a|z); free_bits=bits[free]; mask=np.array([sum(bits[a])]);
        for b in free_bits: mask=np.r_[mask,mask+b]
        assert not covers[mask].any(); covers[mask]=1
        dl=tuple(leaf['minorant_diagonal']); d=np.array(dl); c=np.array(leaf['minorant_center']); assert c.shape==d.shape==(p,)
        if dl not in seen_diag:
            assert np.min(d)>=0 and np.linalg.eigvalsh(G-np.diag(d))[0]>=-1e-9
            seen_diag[dl]=True
        r=y-X@c; g=X.T@r; t=np.abs(g+d*c); q=d+l2
        b=np.minimum(M,np.divide(t,q,out=np.full(p,M),where=q>0)); on=l0-b*(t-q*b/2)
        replay=.5*(r@r+(d*c)@c)+g@c+on[a].sum()+np.minimum(on[free],0).sum()
        assert leaf['lb']<=replay+1e-8,(leaf['lb'],replay)
        replay_slack=max(replay_slack,replay-leaf['lb'])
    assert covers.all()
    L=min(v['lb'] for v in leaves); assert L==cert['certified_lower_bound'] and 0<=obj-L<=1e-5
    return len(leaves),obj-L,replay_slack

def make(seed,p,corr,bound):
    rng=np.random.default_rng(seed)
    n=850; ncal=630; ns=n-ncal
    group=np.tile(np.arange(5),170); rng.shuffle(group)
    s=np.array([rng.choice(3,p=np.roll([.65,.25,.10],int(g)%3)) for g in group]);
    z=rng.normal(size=(n,8))+.25*s[:,None]; phase=rng.uniform(-2.9,2.9,size=(n,2)); ref=np.exp(1j*rng.uniform(-np.pi,np.pi,size=(n,2)))
    offset=np.array([2,-1,3,-2,.4])[group,None]; gain=np.array([.8,1.1,1.5,2.1,.7])[group,None]
    raw=z*gain+offset; ctrl=rng.normal(size=(n,18))*gain+offset
    raw[rng.random(raw.shape)<.025]=np.nan; ctrl[rng.random(ctrl.shape)<.05]=np.nan
    theta=rng.normal(size=13); H=np.column_stack([np.ones(n),s==1,s==2,z,phase]); y=H@theta+.055*rng.normal(size=n)
    bad=rng.choice(ncal,int(.12*ncal),replace=False); y[bad]+=np.clip(2*rng.standard_cauchy(len(bad)),-11,11)
    X=(np.sqrt(1-corr)*rng.normal(size=(ns,p))+np.sqrt(corr)*rng.normal(size=(ns,1)))/np.sqrt(ns)
    w=np.zeros(p); w[[1,3,6,8,11]]=[.6,-.7,1.2,-1.1,.87] if bound else [.9,-1.5,1.7,-1.5,1.15]; y[ncal:]+=X@w
    return dict(panel=dict(center_id=group,stratum=s,X_panel_raw=raw,controls=ctrl),interferometry=dict(signal_phasor=ref*np.exp(1j*phase),ref_phasor=ref),y_assay_raw=y,calibration_indices=np.arange(ncal),stage3_indices=np.arange(ncal,n),X_cand=X,lam0=.09,lam2=.08,M=.88 if bound else 2.0),z,theta

if __name__=='__main__':
    public='/holdout_task/public_eq02_studies.pkl'
    if os.path.isfile(public):
        studies=pickle.load(open(public,'rb')); print('PUBLIC FIXTURE STRUCTURE',type(studies),list(studies)[:4] if isinstance(studies,dict) else len(studies))
    else: print('PUBLIC FIXTURES UNAVAILABLE')
    for seed in range(12):
        p=18+seed%2; corr=[.02,.4,.78,.91][seed//3]; st,z,theta=make(seed,p,corr,bool(seed%3)); t=time.perf_counter()
        try: out=mod.solve_coupled_interferometry_l0(st)
        except ArithmeticError as e:
            print('FAIL', seed,p,corr,round(time.perf_counter()-t,3),str(e)); continue
        elapsed=time.perf_counter()-t; mae=np.abs(out['aligned_panel']-z).mean(); rel=np.linalg.norm(out['calibration_weights']-theta)/np.linalg.norm(theta)
        cert=out['certified_selection']; print('SLICE',seed,p,corr,'seconds',round(elapsed,4),'mae',round(mae,4),'rel',round(rel,4),'cert',replay(st['X_cand'],out['deconfounded_target'],st['lam0'],st['lam2'],st['M'],cert),flush=True)
    for seed in range(4):
        rng=np.random.default_rng(seed+20); X=rng.normal(size=(23,9)); X[:,1]+=.9*X[:,0]; y=rng.normal(size=23); M=.18+seed*.08; l0=.30; l2=.03
        cert=mod._box_l0(X,y,l0,l2,M); replay(X,y,l0,l2,M,cert); ref=np.dot(y,y)/2
        for k in range(1,10):
            for inds in itertools.combinations(range(9),k):
                A=np.vstack((X[:,inds],np.sqrt(l2)*np.eye(k))); b=np.r_[y,np.zeros(k)]; z=lsq_linear(A,b,bounds=(-M,M),tol=1e-13).x
                r=A@z-b; ref=min(ref,.5*(r@r)+k*l0)
        assert abs(ref-cert['objective'])<1e-7,(ref,cert['objective']); print('ORACLE',seed,'error',abs(ref-cert['objective']),flush=True)
PY
python3 -I /tmp/eq02_check.py
```
- **Container Exit Code**: `0` (`5150 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
PUBLIC FIXTURES UNAVAILABLE
SLICE 0 18 0.02 seconds 0.1915 mae 0.0359 rel 0.0037 cert (3, 2.0152945978679782e-10, 2.0152768342995842e-10)
SLICE 1 19 0.02 seconds 0.1868 mae 0.0604 rel 0.0569 cert (7, 2.816654475878977e-06, 2.0044410575792426e-10)
SLICE 2 18 0.02 seconds 0.1693 mae 0.0469 rel 0.0074 cert (4, 2.0036239334331185e-10, 2.0037660419802705e-10)
SLICE 3 19 0.4 seconds 0.1888 mae 0.0447 rel 0.0104 cert (6, 2.0188650751151727e-10, 2.0188295479783847e-10)
SLICE 4 18 0.4 seconds 0.1868 mae 0.0373 rel 0.0067 cert (6, 2.0033219527704205e-10, 2.003335275446716e-10)
SLICE 5 19 0.4 seconds 0.2035 mae 0.0486 rel 0.0348 cert (10, 2.0042634218953026e-10, 2.0043344761688786e-10)
SLICE 6 18 0.78 seconds 0.1998 mae 0.0479 rel 0.0222 cert (6, 2.0165558112239523e-10, 2.0165735747923463e-10)
SLICE 7 19 0.78 seconds 0.1302 mae 0.0457 rel 0.012 cert (3, 2.0046364568315767e-10, 2.0046719839683647e-10)
SLICE 8 18 0.78 seconds 0.1796 mae 0.0499 rel 0.0149 cert (3, 2.0055779259564588e-10, 2.0057200345036108e-10)
SLICE 9 19 0.91 seconds 0.1849 mae 0.0337 rel 0.0132 cert (3, 2.0224888430675492e-10, 2.0224799612833522e-10)
SLICE 10 18 0.91 seconds 0.1478 mae 0.0374 rel 0.0092 cert (4, 2.0040147319377866e-10, 2.0040324955061806e-10)
SLICE 11 19 0.91 seconds 0.3183 mae 0.0401 rel 0.0121 cert (22, 2.0052581817253667e-10, 2.0053647631357308e-10)
ORACLE 0 error 1.7763568394002505e-15
ORACLE 1 error 0.0
ORACLE 2 error 1.7763568394002505e-15
ORACLE 3 error 0.0

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bdfa466e92e1b23e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bdfa466e (directory).
```

### Turn 7 — `execute_in_remote_docker` (`27817 ms` LLM)
- **Rationale**: The initial checks passed. Tighter boxes and more strongly correlated columns test whether ambiguous activations threaten the leaf or runtime budget; a failure would call for improving the bound or branching rule before submission.

```bash
set -e; cat > /tmp/eq02_boundary.py <<'PY'
import runpy, time
import numpy as np
q=runpy.run_path('/tmp/eq02_check.py'); replay=q['replay']; mod=q['mod']; make=q['make']
worst=np.zeros(4); times=[]
for seed in range(8):
    p=18+seed%2
    st,z,theta=make(80+seed,p,.94 if seed<4 else .985,True)
    X=st['X_cand']; st['lam0']=(.018,.035,.065,.105)[seed%4]; st['lam2']=.025 if seed<4 else .006; st['M']=.45 if seed%2 else .7
    # Increase the target-only signal on two similar columns to exercise box saturation and ambiguous supports.
    st['y_assay_raw'][st['stage3_indices']]+=1.8*X[:,1]+1.3*X[:,2]-1.1*X[:,6]
    t=time.perf_counter(); out=mod.solve_coupled_interferometry_l0(st); t=time.perf_counter()-t; times.append(t)
    c=out['certified_selection']; leaves,gap,_=replay(X,out['deconfounded_target'],st['lam0'],st['lam2'],st['M'],c)
    mae=np.mean(abs(out['aligned_panel']-z)); rel=np.linalg.norm(out['calibration_weights']-theta)/np.linalg.norm(theta)
    saturated=np.count_nonzero(np.isclose(abs(np.asarray(c['weights'])),st['M']))
    print('CONNECTED BOX',seed,p,'leaves',leaves,'gap',gap,'saturated',saturated,'seconds',round(t,3),flush=True)
    worst=np.maximum(worst,[mae,rel,gap,leaves])
assert worst[0]<=.11 and worst[1]<=.14 and sum(sorted(times)[-2:])<=18
print('WORST mae, calibration rel-L2, certificate gap, leaves:',worst.tolist(),'slowest pair seconds:',sum(sorted(times)[-2:]))
PY
python3 -I /tmp/eq02_boundary.py
if test -f /holdout_task/validate_public_schema.py; then python3 -I /holdout_task/validate_public_schema.py; else printf '%s\n' 'Public validator not mounted'; fi
```
- **Container Exit Code**: `0` (`7286 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
CONNECTED BOX 0 18 leaves 4 gap 2.0047252746735467e-10 saturated 14 seconds 0.175
CONNECTED BOX 1 19 leaves 19 gap 2.0023449565087503e-10 saturated 8 seconds 0.261
CONNECTED BOX 2 18 leaves 2 gap 2.0036239334331185e-10 saturated 9 seconds 0.15
CONNECTED BOX 3 19 leaves 29 gap 2.0019541580040823e-10 saturated 6 seconds 0.313
CONNECTED BOX 4 18 leaves 21 gap 2.0036239334331185e-10 saturated 8 seconds 0.283
CONNECTED BOX 5 19 leaves 336 gap 2.0017765223201422e-10 saturated 4 seconds 3.048
CONNECTED BOX 6 18 leaves 12 gap 2.0043344761688786e-10 saturated 4 seconds 0.247
CONNECTED BOX 7 19 leaves 166 gap 2.0021317936880223e-10 saturated 6 seconds 1.331
WORST mae, calibration rel-L2, certificate gap, leaves: [0.046866185850839036, 0.021096482297952843, 2.0047252746735467e-10, 336.0] slowest pair seconds: 4.3784817981068045
Public validator not mounted

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=bdfa466e92e1b23e (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_bdfa466e (directory).
```

### Turn 8 — `execute_in_remote_docker` (`29263 ms` LLM)
- **Rationale**: Extend the separable bound to the specified inactive interval. Replay the same certificates to catch regressions in gap, coverage, or runtime, and test a connected instance where feasible uncharged coefficients improve on the exact-zero incumbent.
- **Alternatives Weighed**: `Submit the exact-zero certificate unchanged` (rejected: The objective permits nonzero coefficients of magnitude at most 1e-8 without an activation charge, so a bound for exact-zero states can overstate the literal optimum.)

```bash
set -e
python3 - <<'PY'
from pathlib import Path
p = Path('/holdout_task/coupled_interferometry_certified_l0.py')
s = p.read_text()
old = '''        on = lam0 - b * (linear - q * b / 2)
        val = const + on[active].sum() + np.minimum(on[~(zero | active)], 0).sum()
'''
new = '''        on = lam0 - b * (linear - q * b / 2)
        # The specified count treats coefficients up to 1e-8 as inactive.
        # Allow that interval in the bound; it also bounds the exact-zero cube.
        small = np.minimum(b, 1e-8)
        off = -small * (linear - q * small / 2)
        val = const + on[active].sum() + off[zero].sum()
        val += np.minimum(on[~(zero | active)], off[~(zero | active)]).sum()
'''
assert old in s
p.write_text(s.replace(old,new))
PY
python3 -I /tmp/eq02_check.py
python3 -I /tmp/eq02_boundary.py
python3 -I - <<'PY'
import numpy as np, runpy
q=runpy.run_path('/tmp/eq02_check.py'); mod=q['mod']
p=19; X=np.eye(p)+.02; y=np.full(p,.25); l0=3.; l2=.05; M=.8
cert=mod._box_l0(X,y,l0,l2,M)
w=np.full(p,1e-8); r=X@w-y; tiny_objective=.5*(r@r+l2*(w@w))
assert cert['certified_lower_bound'] <= tiny_objective < cert['objective']
print('THRESHOLD FEASIBILITY: uncharged candidate improves exact-zero incumbent by',cert['objective']-tiny_objective,'and certified gap is',cert['objective']-cert['certified_lower_bound'])
PY
```
- **Container Exit Code**: `0` (`11863 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/coupled_interferometry_certified_l0.py']`
- **Stdout (tail)**:
```text
.0604 rel 0.0569 cert (7, 2.839288679723495e-06, 2.2834640844848764e-08)
SLICE 2 18 0.02 seconds 0.1702 mae 0.0469 rel 0.0074 cert (4, 1.710804298227231e-08, 2.4816515065140266e-08)
SLICE 3 19 0.4 seconds 0.1949 mae 0.0447 rel 0.0104 cert (6, 1.503786606349422e-08, 1.744151134630556e-08)
SLICE 4 18 0.4 seconds 0.1788 mae 0.0373 rel 0.0067 cert (6, 1.5862333668792417e-08, 2.5472081333077767e-08)
SLICE 5 19 0.4 seconds 0.2015 mae 0.0486 rel 0.0348 cert (10, 1.7168442667525596e-08, 2.7687008241628064e-08)
SLICE 6 18 0.78 seconds 0.169 mae 0.0479 rel 0.0222 cert (6, 1.444079167356449e-08, 1.562398210808169e-08)
SLICE 7 19 0.78 seconds 0.1554 mae 0.0457 rel 0.012 cert (3, 1.3886596761381043e-08, 1.4760933808588561e-08)
SLICE 8 18 0.78 seconds 0.1856 mae 0.0499 rel 0.0149 cert (3, 4.921425045267824e-09, 1.1565482793685078e-08)
SLICE 9 19 0.91 seconds 0.1777 mae 0.0337 rel 0.0132 cert (3, 1.0062430000346012e-08, 1.603331245547679e-08)
SLICE 10 18 0.91 seconds 0.1649 mae 0.0374 rel 0.0092 cert (4, 1.6030451632786935e-08, 1.6030448080073256e-08)
SLICE 11 19 0.91 seconds 0.3203 mae 0.0401 rel 0.0121 cert (22, 2.8569754562113303e-08, 2.8569754562113303e-08)
ORACLE 0 error 1.7763568394002505e-15
ORACLE 1 error 0.0
ORACLE 2 error 1.7763568394002505e-15
ORACLE 3 error 0.0
CONNECTED BOX 0 18 leaves 4 gap 1.8083738950736006e-09 saturated 14 seconds 0.2
CONNECTED BOX 1 19 leaves 19 gap 1.1177053949040783e-08 saturated 8 seconds 0.209
CONNECTED BOX 2 18 leaves 2 gap 3.922423275071196e-09 saturated 9 seconds 0.171
CONNECTED BOX 3 19 leaves 29 gap 4.4108912078399953e-08 saturated 6 seconds 0.32
CONNECTED BOX 4 18 leaves 21 gap 1.7525181306154991e-09 saturated 8 seconds 0.287
CONNECTED BOX 5 19 leaves 336 gap 3.5386813479476587e-09 saturated 4 seconds 2.454
CONNECTED BOX 6 18 leaves 12 gap 2.000574639282604e-08 saturated 4 seconds 0.234
CONNECTED BOX 7 19 leaves 166 gap 5.2679464346283567e-08 saturated 6 seconds 1.158
WORST mae, calibration rel-L2, certificate gap, leaves: [0.046866185850839036, 0.021096482297952843, 5.2679464346283567e-08, 336.0] slowest pair seconds: 3.611842192010954
THRESHOLD FEASIBILITY: uncharged candidate improves exact-zero incumbent by 6.554999831820396e-08 and certified gap is 6.575001310071826e-08

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=07686a3209288819 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_07686a32 (directory).
```

### Turn 9 — `submit_completed_task` (`14113 ms` LLM)
- **Rationale**: The final artifact passed isolated certificate replay, exhaustive small-instance comparisons, and connected 18- and 19-column stress tests within the specified tolerances.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/coupled_interferometry_certified_l0.py': 'FILE:10997'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_interferometry_certified_l0.py`. Across 20 synthetic studies, the worst certificate gap was 2.84e-6, the largest certificate had 336 leaves, and the two slowest solves took 3.61 seconds combined. Four exhaustive reference solves matched to numerical precision. The public studies and schema validator were not mounted.`

### Turn 10 — `execute_in_remote_docker` (`37746 ms` LLM)
- **Rationale**: Use public measurements if mounted. Independently reconstruct cell-balanced imputation and the optical-to-assay handoff to detect convention errors, then compare Cauchy and OLS on held-out centers to expose calibration degradation that pooled scores could hide. Replay the certificate on the resulting target in an isolated process.
- **Alternatives Weighed**: `Audit classifier recall, relay modes, or drift operators` (rejected: This task has a continuous calibration target, optical phase demodulation, and a quadratic box objective; those checks would not exercise a supported mechanism.)

```bash
set -e
find /holdout_task /pretrain_task /workspace /app /results -maxdepth 3 -type f \( -name 'public_eq02_studies.pkl' -o -name 'validate_public_schema.py' \) -print 2>/dev/null || true
python3 -I - <<'PY'
from pathlib import Path
import pickle, runpy, time
import numpy as np
q=runpy.run_path('/tmp/eq02_check.py'); mod=q['mod']; replay=q['replay']

def studies(obj):
    if isinstance(obj, dict):
        if {'panel','interferometry','X_cand'} <= obj.keys(): yield obj
        else:
            for value in obj.values(): yield from studies(value)
    elif isinstance(obj, (list,tuple)):
        for value in obj: yield from studies(value)

public=Path('/holdout_task/public_eq02_studies.pkl')
if public.is_file():
    with public.open('rb') as f: cases=list(studies(pickle.load(f)))
    assert cases, 'Public file contains no studies in the expected schema'
    label='PUBLIC'
else:
    print('Public studies not mounted; using synthetic measurements with known clean calibration targets.')
    cases=[q['make'](80+i,18+i%2,.985,True)[0] for i in (4,5,6,7)]
    label='SYNTHETIC'
rows=[]
for i,st in enumerate(cases):
    p=st['panel']; g=np.asarray(p['center_id']); s=np.asarray(p['stratum']); raw=np.asarray(p['X_panel_raw']); ctrl=np.asarray(p['controls']); ref=np.empty_like(raw)
    levels=np.unique(g)
    for level in levels:
        ix=g==level; cols=ctrl[ix].T; cols=[v[np.isfinite(v)] for v in cols]; cols=[v for v in cols if len(v)>=2]
        ref[ix]=(raw[ix]-np.median([v.mean() for v in cols]))/np.median([v.std(ddof=0) for v in cols])
    for j in range(raw.shape[1]):
        means=[]; mass=[]
        for level in levels:
            for k in range(3):
                v=ref[(g==level)&(s==k),j]; ok=np.isfinite(v); means.append(v[ok].sum()/len(v)); mass.append(ok.mean())
        ref[~np.isfinite(ref[:,j]),j]=sum(means)/sum(mass)
    optical=st['interferometry']; phase=np.angle(np.exp(1j*(np.angle(optical['signal_phasor'])-np.angle(optical['ref_phasor']))))
    H=np.column_stack((np.ones(len(g)),s==1,s==2,ref,phase)); y=np.asarray(st['y_assay_raw']); cal=np.asarray(st['calibration_indices']); target=np.asarray(st['stage3_indices'])
    t=time.perf_counter(); out=mod.solve_coupled_interferometry_l0(st); elapsed=time.perf_counter()-t
    np.testing.assert_allclose(out['aligned_panel'],ref,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(out['deconfounded_target'],y[target]-H[target]@out['calibration_weights'],rtol=1e-11,atol=1e-11)
    leaves,gap,_=replay(st['X_cand'],out['deconfounded_target'],st['lam0'],st['lam2'],st['M'],out['certified_selection'])
    errors=[]
    for level in levels:
        train=cal[g[cal]!=level]; hold=cal[g[cal]==level]
        if not len(hold): continue
        robust=mod._cauchy(H[train],y[train]); ols=np.linalg.lstsq(H[train],y[train],rcond=None)[0]
        errors.append([np.median(abs(y[hold]-H[hold]@t)) for t in (robust,ols)])
    rows.extend(errors)
    print(label,i,'leaves',leaves,'gap',gap,'seconds',round(elapsed,3),'held-center residual medians, worst [Cauchy, OLS]',np.max(errors,axis=0).tolist())
print('Worst held-center median residual [Cauchy, OLS]:',np.max(rows,axis=0).tolist())
PY
if test -f /holdout_task/validate_public_schema.py; then python3 -I /holdout_task/validate_public_schema.py; else echo 'Public schema validator not mounted'; fi
```
- **Container Exit Code**: `0` (`2634 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public studies not mounted; using synthetic measurements with known clean calibration targets.
SYNTHETIC 0 leaves 34 gap 5.233913213942287e-08 seconds 0.352 held-center residual medians, worst [Cauchy, OLS] [0.09596396406328278, 0.36292920942355944]
SYNTHETIC 1 leaves 20 gap 4.144680154638536e-09 seconds 0.118 held-center residual medians, worst [Cauchy, OLS] [0.08856236676656848, 0.32383161715662623]
SYNTHETIC 2 leaves 3 gap 1.3116906671939432e-08 seconds 0.025 held-center residual medians, worst [Cauchy, OLS] [0.12914585028882453, 0.4716153425853049]
SYNTHETIC 3 leaves 32 gap 4.355855054427593e-08 seconds 0.19 held-center residual medians, worst [Cauchy, OLS] [0.11616391900241574, 0.2674120751280564]
Worst held-center median residual [Cauchy, OLS]: [0.12914585028882453, 0.4716153425853049]
Public schema validator not mounted

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=07686a3209288819 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_07686a32 (directory).
```

### Turn 11 — `submit_completed_task` (`15250 ms` LLM)
- **Rationale**: The final artifact passed isolated checks of cell-balanced imputation, the optical-to-assay handoff, and certificate coverage and bounds. On the synthetic held-center audit, the worst median residual was 0.129 for Cauchy calibration versus 0.472 for OLS.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/coupled_interferometry_certified_l0.py': 'FILE:10997'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/coupled_interferometry_certified_l0.py`. Across synthetic stress tests, the largest certificate had 336 leaves, the worst certified gap was 2.84e-6, and the two slowest solves took 3.61 seconds combined. Four exhaustive reference solves matched to numerical precision. The public studies and schema validator were not mounted, so public validation remains unverified.`

### Turn 12 — `execute_in_remote_docker` (`28371 ms` LLM)
- **Rationale**: The contract does not require normalized candidate columns. Test whether the 1e-5 absolute gap still holds at unit-variance feature scale. If it fails, use the result to distinguish a too-loose witness from an incumbent that does not optimize within the uncharged coefficient interval.
- **Alternatives Weighed**: `Repeat the held-center calibration audit` (rejected: It already confirmed the benefit of robust calibration; repeating it would not test the remaining absolute-gap risk.)

```bash
set -e
python3 -I - <<'PY'
from pathlib import Path
import os, pickle, runpy, time
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
q=runpy.run_path('/tmp/eq02_check.py'); solve=q['mod'].solve_coupled_interferometry_l0; replay=q['replay']
def studies(obj):
    if isinstance(obj,dict):
        if {'panel','interferometry','X_cand'} <= obj.keys(): yield obj
        else:
            for v in obj.values(): yield from studies(v)
    elif isinstance(obj,(list,tuple)):
        for v in obj: yield from studies(v)
public=Path('/holdout_task/public_eq02_studies.pkl')
if public.is_file():
    with public.open('rb') as f: cases=list(studies(pickle.load(f)))
    assert cases, 'No public studies found in fixture'
    cases=list(enumerate(cases)); label='PUBLIC'
else:
    print('Public fixture unavailable; testing the unnormalized candidate scale allowed by the contract.')
    label='SYNTHETIC'
    cases=[]
    for i,corr in enumerate((.05,.45,.85,.96)):
        st,_,_=q['make'](106+i,18+i%2,corr,True)
        X=st['X_cand']; n=len(X); s=st['stage3_indices']
        b=np.zeros(X.shape[1]); b[[1,3,6,8,11]]=[.6,-.7,1.2,-1.1,.87]
        st['y_assay_raw'][s]+=(np.sqrt(n)-1)*(X@b)
        st['X_cand']=X*np.sqrt(n); st['lam0']*=n; st['lam2']*=n
        cases.append((i,st))
for i,st in cases:
    X=st['X_cand']; t=time.perf_counter()
    try:
        out=solve(st)
        leaves,gap,_=replay(X,out['deconfounded_target'],st['lam0'],st['lam2'],st['M'],out['certified_selection'])
        print(label,i,X.shape,'seconds',round(time.perf_counter()-t,3),'leaves',leaves,'gap',gap,flush=True)
    except (ArithmeticError,AssertionError) as e:
        print(label,i,X.shape,'seconds',round(time.perf_counter()-t,3),'FAIL',repr(e),flush=True)
PY
if test -f /holdout_task/validate_public_schema.py; then python3 -I /holdout_task/validate_public_schema.py; fi
```
- **Container Exit Code**: `0` (`762 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Public fixture unavailable; testing the unnormalized candidate scale allowed by the contract.
SYNTHETIC 0 (220, 18) seconds 0.017 leaves 2 gap 1.1199234393188817e-06
SYNTHETIC 1 (220, 19) seconds 0.024 leaves 3 gap 5.395393429807882e-07
SYNTHETIC 2 (220, 18) seconds 0.023 leaves 3 gap 2.034268476336365e-06
SYNTHETIC 3 (220, 19) seconds 0.021 leaves 2 gap 1.7685366344721842e-06

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=07686a3209288819 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_07686a32 (directory).
```

### Turn 13 — `submit_completed_task` (`15366 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated certificate replay on normalized and unnormalized connected designs. The worst unnormalized gap was 2.04e-6 against the 1e-5 limit, so the scale audit did not warrant another edit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/coupled_interferometry_certified_l0.py`. Synthetic stress tests verified a worst certificate gap of 2.84e-6, a maximum of 336 leaves, and a combined runtime of 3.61 seconds for the two slowest solves. Four exhaustive reference solves matched to numerical precision. The public studies and schema validator were not mounted, so public validation remains unverified.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M92-EqualInfo-4",
  "strict_pass": true,
  "reward": 1.0,
  "artifact_audit": {
    "artifact_status": {
      "/holdout_task/coupled_interferometry_certified_l0.py": "FILE:10997"
    },
    "missing_or_empty": [],
    "all_present": true
  },
  "verifier_metrics": {
    "strict_pass": true,
    "reward": 1.0,
    "all_valid": true,
    "max_panel_mae": 0.045663,
    "max_calibration_rel_l2": 0.017548,
    "max_deconfounded_mae": 0.19899,
    "max_stage3_w_vs_clean_rel_l2": 0.171413,
    "max_duality_gap": 2.349e-06,
    "student_elapsed_sec": 0.0969,
    "instance_diagnostics": [
      {
        "seed": 99411,
        "p": 18,
        "panel_mae": 0.045663,
        "cal_rel_l2": 0.016892,
        "deconf_mae": 0.177103,
        "w_vs_clean_rel_l2": 0.171413,
        "cert_ok": true,
        "cert_reason": "ok"
      },
      {
        "seed": 99412,
        "p": 19,
        "panel_mae": 0.045333,
        "cal_rel_l2": 0.017548,
        "deconf_mae": 0.19899,
        "w_vs_clean_rel_l2": 0.096112,
        "cert_ok": true,
        "cert_reason": "ok"
      }
    ]
  },
  "verifier_exit_code": 0,
  "verifier_stdout": "{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_panel_mae\": 0.045663, \"max_calibration_rel_l2\": 0.017548, \"max_deconfounded_mae\": 0.19899, \"max_stage3_w_vs_clean_rel_l2\": 0.171413, \"max_duality_gap\": 2.349e-06, \"student_elapsed_sec\": 0.0969, \"instance_diagnostics\": [{\"seed\": 99411, \"p\": 18, \"panel_mae\": 0.045663, \"cal_rel_l2\": 0.016892, \"deconf_mae\": 0.177103, \"w_vs_clean_rel_l2\": 0.171413, \"cert_ok\": true, \"cert_reason\": \"ok\"}, {\"seed\": 99412, \"p\": 19, \"panel_mae\": 0.045333, \"cal_rel_l2\": 0.017548, \"deconf_mae\": 0.19899, \"w_vs_clean_rel_l2\": 0.096112, \"cert_ok\": true, \"cert_reason\": \"ok\"}]}\n__HEGEL_VERIFIER_AUTH_f0633b7e28a80632e4536b496a538957__:{\"strict_pass\": true, \"reward\": 1.0, \"all_valid\": true, \"max_panel_mae\": 0.045663, \"max_calibration_rel_l2\": 0.017548, \"max_deconfounded_mae\": 0.19899, \"max_stage3_w_vs_clean_rel_l2\": 0.171413, \"max_duality_gap\": 2.349e-06, \"student_elapsed_sec\": 0.0969, \"instance_diagnostics\": [{\"seed\": 99411, \"p\": 18, \"panel_mae\": 0.045663, \"cal_rel_l2\": 0.016892, \"deconf_mae\": 0.177103, \"w_vs_clean_rel_l2\": 0.171413, \"cert_ok\": true, \"cert_reason\": \"ok\"}, {\"seed\": 99412, \"p\": 19, \"panel_mae\": 0.045333, \"cal_rel_l2\": 0.017548, \"deconf_mae\": 0.19899, \"w_vs_clean_rel_l2\": 0.096112, \"cert_ok\": true, \"cert_reason\": \"ok\"}]}\n",
  "verifier_stderr": "",
  "verifier_wall_ms": 5497.03,
  "failure_taxonomy": "PASSED_ALL_CHECKS"
}
```
